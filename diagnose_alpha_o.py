"""
Diagnose α_o homogeneity: does per-step overhead correlate with per-step compute density?

Approach:
  1. Fit the parallel-only model (Model C: α_c·TO_c + α_m·TO_m)
  2. For each sequential data point, compute:
     - E_parallel_pred = α_c·TO_c + α_m·TO_m  (what parallel model predicts)
     - E_overhead = E_measured - E_parallel_pred  (unexplained by compute+memory)
     - overhead_per_step = E_overhead / n_seq_steps
     - TO_per_step = TO_compute / n_seq_steps  (compute density)
  3. Check correlation between overhead_per_step and TO_per_step

If per-step overhead DECREASES with compute density, we have a physically
grounded basis for a correction term.

Author: Muntaser Syed
Date: May 2026
"""

import csv
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.to_model import predict_to, TO_MODELS, get_seq_steps


def load_csv(path):
    points = []
    with open(path) as f:
        for row in csv.DictReader(f):
            points.append(row)
    return points


def analyze_gpu(csv_path, gpu_name, has_torchaudio):
    rows = load_csv(csv_path)

    # Compute TO predictions
    data = []
    for row in rows:
        alg = row["algorithm"]
        N = int(row["signal_length"])
        B = int(row["batch_size"])
        E = float(row["energy_per_call_j"])
        if alg not in TO_MODELS or E <= 0:
            continue
        result = predict_to(alg, N, B)
        n_seq = get_seq_steps(alg, N, B, has_torchaudio=has_torchaudio)
        data.append({
            "alg": alg, "N": N, "B": B, "E": E,
            "to_c": result["to_compute"], "to_m": result["to_memory"],
            "n_seq": n_seq,
        })

    # Split into parallel and sequential
    parallel = [d for d in data if d["n_seq"] == 0 and d["to_c"] > 0]
    sequential = [d for d in data if d["n_seq"] > 0]

    # Fit parallel-only model (Model C)
    X_par = np.array([[d["to_c"], d["to_m"]] for d in parallel])
    y_par = np.array([d["E"] for d in parallel])
    alpha = np.linalg.solve(X_par.T @ X_par, X_par.T @ y_par)
    alpha_c, alpha_m = alpha

    y_pred_par = X_par @ alpha
    ss_res = np.sum((y_par - y_pred_par) ** 2)
    ss_tot = np.sum((y_par - np.mean(y_par)) ** 2)
    r2_par = 1 - ss_res / ss_tot

    print(f"\n{'='*90}")
    print(f"  {gpu_name} (torchaudio={has_torchaudio})")
    print(f"{'='*90}")
    print(f"  Parallel-only fit (Model C): α_c={alpha_c*1e15:.2f} fJ/TO, "
          f"α_m={alpha_m*1e15:.2f} fJ/TO, r²={r2_par:.4f}, n={len(parallel)}")

    # For each sequential point: compute overhead per step and TO per step
    print(f"\n  {'Algorithm':18s} {'N':>6s} {'B':>4s} {'E_meas':>10s} {'E_par_pred':>10s} "
          f"{'E_overhead':>10s} {'n_seq':>7s} {'uJ/step':>9s} {'TO/step':>12s} {'log10(TO/s)':>11s}")
    print(f"  {'-'*110}")

    seq_data = []
    for d in sequential:
        E_par_pred = alpha_c * d["to_c"] + alpha_m * d["to_m"]
        E_overhead = d["E"] - E_par_pred
        overhead_per_step_uJ = (E_overhead / d["n_seq"]) * 1e6 if d["n_seq"] > 0 else 0
        to_per_step = d["to_c"] / d["n_seq"] if d["n_seq"] > 0 else 0

        print(f"  {d['alg']:18s} {d['N']:>6d} {d['B']:>4d} "
              f"{d['E']:>10.4e} {E_par_pred:>10.4e} {E_overhead:>10.4e} "
              f"{d['n_seq']:>7d} {overhead_per_step_uJ:>9.1f} "
              f"{to_per_step:>12.0f} {np.log10(to_per_step) if to_per_step > 0 else 0:>11.2f}")

        if E_overhead > 0 and to_per_step > 0:
            seq_data.append({
                "alg": d["alg"], "N": d["N"],
                "overhead_per_step_uJ": overhead_per_step_uJ,
                "to_per_step": to_per_step,
                "log_to_per_step": np.log10(to_per_step),
                "n_seq": d["n_seq"],
                "E_overhead": E_overhead,
            })

    if not seq_data:
        print("  No valid sequential data points.")
        return

    # Correlation analysis
    log_tos = np.array([d["log_to_per_step"] for d in seq_data])
    overheads = np.array([d["overhead_per_step_uJ"] for d in seq_data])
    log_overheads = np.log10(np.maximum(overheads, 1e-6))

    # Pearson correlation (linear)
    corr_linear = np.corrcoef(log_tos, overheads)[0, 1]
    # Pearson correlation (log-log)
    corr_loglog = np.corrcoef(log_tos, log_overheads)[0, 1]

    print(f"\n  CORRELATION ANALYSIS:")
    print(f"  Pearson(log10(TO/step), overhead_µJ/step) = {corr_linear:.3f}")
    print(f"  Pearson(log10(TO/step), log10(overhead_µJ/step)) = {corr_loglog:.3f}")

    # Group by algorithm to see the pattern more clearly
    print(f"\n  PER-ALGORITHM MEAN (grouped):")
    print(f"  {'Algorithm':18s} {'Mean uJ/step':>12s} {'Mean TO/step':>14s} {'n_pts':>6s}")
    print(f"  {'-'*55}")
    algs = sorted(set(d["alg"] for d in seq_data))
    alg_means = []
    for alg in algs:
        pts = [d for d in seq_data if d["alg"] == alg]
        mean_overhead = np.mean([d["overhead_per_step_uJ"] for d in pts])
        mean_to = np.mean([d["to_per_step"] for d in pts])
        print(f"  {alg:18s} {mean_overhead:>12.1f} {mean_to:>14.0f} {len(pts):>6d}")
        alg_means.append({"alg": alg, "mean_overhead": mean_overhead, "mean_to": mean_to})

    # Per-algorithm correlation
    if len(alg_means) >= 4:
        alg_log_tos = np.array([np.log10(d["mean_to"]) for d in alg_means if d["mean_to"] > 0])
        alg_overheads = np.array([d["mean_overhead"] for d in alg_means if d["mean_to"] > 0])
        alg_log_overheads = np.log10(np.maximum(alg_overheads, 1e-6))
        corr_alg = np.corrcoef(alg_log_tos, alg_log_overheads)[0, 1]
        print(f"\n  Per-algorithm Pearson(log10(TO/step), log10(overhead)) = {corr_alg:.3f}")

        # Fit a power law: overhead = a * (TO/step)^b
        # log(overhead) = log(a) + b * log(TO/step)
        coeffs = np.polyfit(alg_log_tos, alg_log_overheads, 1)
        b_slope = coeffs[0]
        a_intercept = 10**coeffs[1]
        print(f"  Power law fit: overhead_µJ = {a_intercept:.2f} × (TO/step)^{b_slope:.3f}")
        print(f"  (Negative slope means overhead DECREASES with compute density)")

    # Summary
    print(f"\n  SUMMARY:")
    print(f"  If slope is negative and |correlation| > 0.5, then compute density")
    print(f"  does explain per-step overhead variation, supporting a density-aware α_o.")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    local_csv = base / "data" / "results" / "all_results.csv"
    if local_csv.exists():
        analyze_gpu(str(local_csv), "RTX 4090 Laptop", has_torchaudio=True)

    server_csv = base / "data" / "server_results" / "results" / "all_results.csv"
    if server_csv.exists():
        analyze_gpu(str(server_csv), "A100 SXM4", has_torchaudio=False)
