"""
Diagnose kernel-launch-count approach for sequential step counting.

Instead of counting outer-loop iterations, count total CUDA kernel launches.
Each torch operation on a CUDA tensor that produces a new tensor (not a view)
triggers at least one kernel launch, incurring Python-to-CUDA dispatch overhead.

Kernel launch counts derived from source code analysis of:
  algorithms/adaptive.py, algorithms/estimation.py,
  algorithms/decomposition.py, algorithms/compression.py

Author: Muntaser Syed
Date: May 2026
"""

import csv
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.to_model import predict_to, TO_MODELS

# =========================================================================
# KERNEL LAUNCHES PER OUTER ITERATION (from source code analysis)
# =========================================================================
# Each entry: (outer_iterations_func, kernels_per_outer_iter)
# Views (slice, unsqueeze, squeeze, transpose, reshape) = 0 kernels
# Each torch op producing new data = 1 kernel (conservative)

KERNELS_PER_ITER = {
    # Adaptive: algorithms/adaptive.py
    "lms":       7,   # flip, mul+sum, sub, scalar*tensor*tensor+tensor
    "nlms":     11,   # LMS(7) + pow+sum+add(norm:3) + div(1)
    "rls":      12,   # 4x bmm, inv, 2x div, 3x add/sub, flip, mul
    "apa_p4":   15,   # P*flip+stack(5), 2x bmm, solve, eye+mul+add(3), bmm+mul+add(3)

    # Estimation: algorithms/estimation.py
    "kalman":   15,   # 7x matmul, inv, 4x add/sub, 2x indexing-copy, squeeze
    "ekf":      22,   # kalman(15) + sin(1) + cos+diag_embed(2) + bmm upgrades(4)
    "ukf":      43,   # cholesky+mul+add(3), zeros+copy(2), 4x2 sigma(8),
                      # matmul+sin+mul+add(4), 4x mean(4), sub(1),
                      # bmm+mul+add(3), 8x2 cov inner(16), add+mul symmetry(2)
    "particle_1k": 20,  # randn_like+mul+add(3), sub+pow+sum(3), mul(1),
                        # max+sub(2), exp(1), sum+div(2), cumsum(1),
                        # arange+rand+add+div(4), searchsorted(1), clamp(1), gather(1)

    # Decomposition: algorithms/decomposition.py
    "fastica":  13,   # bmm(1), tanh(1), pow+sub(2), bmm+div+mean+transpose+mul+sub(6), QR(1), 2 extra
    "nmf":      14,   # 4x bmm(4), 4x transpose(0-view), 2x add(2), 2x mul(2), 2x div(2), 4x misc(4)
    "pca":       0,   # pca_lowrank is a C++ function — no Python dispatch overhead
                      # (cuSOLVER overhead absorbed by alpha_c, like SVD)

    # Compression: algorithms/compression.py
    "mdct_audio": 11,  # mul(window:1), matmul+unsqueeze+squeeze(2), abs+pow(2),
                       # add+log+mul(3), exp(1), clamp(1), div+round(2) → ~11-12
}


def get_outer_iters(alg, N, B, has_torchaudio=True):
    """Return number of outer Python-loop iterations."""
    M = 32  # filter_length default
    if alg in ("lms", "nlms"):
        return B * max(0, min(N, M + 200) - M)
    elif alg == "rls":
        return B * max(0, min(N, M + 100) - M)
    elif alg == "apa_p4":
        P = 4
        return B * max(0, min(N, M + P + 100) - (M + P))
    elif alg in ("kalman", "ekf"):
        return B * min(N, 200)
    elif alg == "ukf":
        return B * min(N, 100)
    elif alg == "particle_1k":
        return B * min(N, 200)
    elif alg in ("fastica", "nmf"):
        return B * 50
    elif alg == "pca":
        return 0  # NOT a Python-loop sequential algorithm
    elif alg == "mdct_audio":
        return B * max(1, min((N - 2*512) // 512 + 1, 50))
    elif alg == "iir_butter4":
        if has_torchaudio:
            return 0
        else:
            return B * N
    return 0


def get_kernel_launches(alg, N, B, has_torchaudio=True):
    """Return total kernel launches = outer_iters × kernels_per_iter."""
    if alg == "iir_butter4":
        if has_torchaudio:
            return 0
        else:
            # IIR Python fallback: each step is ~5 ops (see filters.py fallback)
            return B * N * 5
    kpi = KERNELS_PER_ITER.get(alg, 0)
    outer = get_outer_iters(alg, N, B, has_torchaudio)
    return outer * kpi


def load_csv(path):
    points = []
    with open(path) as f:
        for row in csv.DictReader(f):
            points.append(row)
    return points


def fit_model(X, y, n_params=3):
    """Fit via NNLS. Returns alphas and r²."""
    from scipy.optimize import nnls
    valid = (y > 0) & np.isfinite(y) & np.isfinite(X).all(axis=1)
    if X.shape[1] > 2:
        valid &= (X[:, 0] > 0)
    Xv, yv = X[valid], y[valid]
    alpha, _ = nnls(Xv, yv)
    y_pred = Xv @ alpha
    ss_res = np.sum((yv - y_pred) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return alpha, r2, y_pred, yv, valid


def run_head_to_head(data, alpha, label=""):
    """Run head-to-head ranking and return (correct, total, details)."""
    pairs = [
        ("fft", "direct_dft"), ("fir_direct", "fir_fft"),
        ("lms", "rls"), ("kalman", "ukf"),
        ("periodogram", "welch"), ("cnn_denoiser", "lstm_denoiser"),
        ("lstm_denoiser", "transformer_denoiser"), ("wiener", "cnn_denoiser"),
    ]
    correct, total = 0, 0
    details = []
    # Build lookup: (alg, N) → (E_meas, E_pred)
    lookup = {}
    for d in data:
        E_pred = alpha[0] * d["to_c"] + alpha[1] * d["to_m"] + alpha[2] * d["n_seq"]
        lookup[(d["alg"], d["N"])] = (d["E"], E_pred)

    for alg_a, alg_b in pairs:
        ns_a = sorted(set(N for (a, N) in lookup if a == alg_a))
        ns_b = sorted(set(N for (a, N) in lookup if a == alg_b))
        common = sorted(set(ns_a) & set(ns_b))
        for n in common:
            ea_meas, ea_pred = lookup[(alg_a, n)]
            eb_meas, eb_pred = lookup[(alg_b, n)]
            meas_a_wins = ea_meas < eb_meas
            pred_a_wins = ea_pred < eb_pred
            match = meas_a_wins == pred_a_wins
            total += 1
            if match:
                correct += 1
            status = "✓" if match else "✗"
            winner_meas = alg_a if meas_a_wins else alg_b
            winner_pred = alg_a if pred_a_wins else alg_b
            details.append(f"  {status} {alg_a} vs {alg_b} (N={n}): meas={winner_meas}, pred={winner_pred}")
    return correct, total, details


def analyze_gpu(csv_path, gpu_name, has_torchaudio):
    rows = load_csv(csv_path)

    data = []
    for row in rows:
        alg = row["algorithm"]
        N = int(row["signal_length"])
        B = int(row["batch_size"])
        E = float(row["energy_per_call_j"])
        if alg not in TO_MODELS or E <= 0:
            continue
        result = predict_to(alg, N, B)
        data.append({
            "alg": alg, "N": N, "B": B, "E": E,
            "to_c": result["to_compute"], "to_m": result["to_memory"],
        })

    print(f"\n{'='*90}")
    print(f"  {gpu_name}")
    print(f"{'='*90}")

    # === Approach A: outer iterations only ===
    for d in data:
        d["n_seq"] = get_outer_iters(d["alg"], d["N"], d["B"], has_torchaudio)

    X_a = np.array([[d["to_c"], d["to_m"], d["n_seq"]] for d in data])
    y = np.array([d["E"] for d in data])
    alpha_a, r2_a, _, _, _ = fit_model(X_a, y)
    correct_a, total_a, details_a = run_head_to_head(data, alpha_a)

    print(f"\n  --- Approach A: outer iterations as n_seq ---")
    print(f"  α_c={alpha_a[0]*1e15:.2f} fJ/TO, α_m={alpha_a[1]*1e15:.2f} fJ/TO, "
          f"α_o={alpha_a[2]*1e6:.1f} µJ/iter")
    print(f"  r² = {r2_a:.4f}")
    print(f"  Ranking: {correct_a}/{total_a} = {correct_a/total_a*100:.0f}%")
    for line in details_a:
        print(line)

    # === Approach B: kernel launch counts ===
    for d in data:
        d["n_seq"] = get_kernel_launches(d["alg"], d["N"], d["B"], has_torchaudio)

    X_b = np.array([[d["to_c"], d["to_m"], d["n_seq"]] for d in data])
    alpha_b, r2_b, _, _, _ = fit_model(X_b, y)
    correct_b, total_b, details_b = run_head_to_head(data, alpha_b)

    print(f"\n  --- Approach B: kernel launches as n_seq ---")
    print(f"  α_c={alpha_b[0]*1e15:.2f} fJ/TO, α_m={alpha_b[1]*1e15:.2f} fJ/TO, "
          f"α_o={alpha_b[2]*1e6:.1f} µJ/launch")
    print(f"  r² = {r2_b:.4f}")
    print(f"  Ranking: {correct_b}/{total_b} = {correct_b/total_b*100:.0f}%")
    for line in details_b:
        print(line)

    # === Per-algorithm overhead consistency check ===
    print(f"\n  --- Per-launch overhead consistency (Approach B) ---")
    print(f"  {'Algorithm':18s} {'Outer iters':>11s} {'KPL':>5s} {'Total launches':>14s} "
          f"{'E_overhead(J)':>13s} {'µJ/launch':>10s}")
    print(f"  {'-'*80}")

    seq_overheads = []
    for d in data:
        n_launches = get_kernel_launches(d["alg"], d["N"], d["B"], has_torchaudio)
        if n_launches == 0:
            continue
        E_par = alpha_b[0] * d["to_c"] + alpha_b[1] * d["to_m"]  # parallel-only prediction
        # Use parallel-only coefficients for cleaner overhead estimate
        E_overhead = d["E"] - E_par
        if E_overhead <= 0:
            continue
        uj_per_launch = E_overhead / n_launches * 1e6
        outer = get_outer_iters(d["alg"], d["N"], d["B"], has_torchaudio)
        kpi = KERNELS_PER_ITER.get(d["alg"], 0)
        if kpi == 0:
            continue
        print(f"  {d['alg']:18s} {outer:>11d} {kpi:>5d} {n_launches:>14d} "
              f"{E_overhead:>13.4e} {uj_per_launch:>10.1f}")
        seq_overheads.append({"alg": d["alg"], "uj": uj_per_launch})

    if seq_overheads:
        all_ujs = [s["uj"] for s in seq_overheads]
        print(f"\n  All-point µJ/launch: mean={np.mean(all_ujs):.1f}, "
              f"median={np.median(all_ujs):.1f}, "
              f"std={np.std(all_ujs):.1f}, "
              f"CV={np.std(all_ujs)/np.mean(all_ujs)*100:.1f}%")

        # Per-algorithm means
        algs = sorted(set(s["alg"] for s in seq_overheads))
        alg_means = []
        for alg in algs:
            vals = [s["uj"] for s in seq_overheads if s["alg"] == alg]
            m = np.mean(vals)
            alg_means.append(m)
            print(f"    {alg:18s}: mean={m:.1f} µJ/launch")
        print(f"\n  Per-algorithm µJ/launch range: {min(alg_means):.0f} – {max(alg_means):.0f} "
              f"(ratio {max(alg_means)/min(alg_means):.1f}×)")

    # === Comparison ===
    print(f"\n  === COMPARISON ===")
    print(f"  {'Metric':30s} {'A (iters)':>12s} {'B (launches)':>14s}")
    print(f"  {'-'*60}")
    print(f"  {'r²':30s} {r2_a:>12.4f} {r2_b:>14.4f}")
    print(f"  {'Ranking accuracy':30s} {correct_a}/{total_a} = {correct_a/total_a*100:.0f}%"
          f"{'':>4s}{correct_b}/{total_b} = {correct_b/total_b*100:.0f}%")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    local_csv = base / "data" / "results" / "all_results.csv"
    if local_csv.exists():
        analyze_gpu(str(local_csv), "RTX 4090 Laptop", has_torchaudio=True)

    server_csv = base / "data" / "server_results" / "results" / "all_results.csv"
    if server_csv.exists():
        analyze_gpu(str(server_csv), "A100 SXM4", has_torchaudio=False)
