"""
TOMLSignals - Improved Analysis with Log-Space Fitting
=======================================================
Fixes:
  1. Log-log fitting (standard for multi-decade energy data)
  2. Per-GPU IIR sequential step correction
  3. Robust error metrics (median, percentiles)
  4. GPU utilization filtering

Author: Muntaser Syed
Date: May 2026
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.to_model import predict_to, TO_MODELS, get_seq_steps

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =========================================================================
# DATA LOADING
# =========================================================================

def load_csv(path, gpu_name="Unknown"):
    points = []
    with open(path) as f:
        for row in csv.DictReader(f):
            gpu = row.get("gpu_name", gpu_name)
            if not gpu or gpu == "":
                gpu = gpu_name
            points.append({
                "algorithm": row["algorithm"],
                "category": row["category"],
                "N": int(row["signal_length"]),
                "B": int(row["batch_size"]),
                "e_meas": float(row["energy_per_call_j"]),
                "dp": float(row["delta_power_w"]),
                "t_us": float(row["time_per_call_us"]),
                "idle_w": float(row.get("idle_power_w", 0)),
                "gpu": gpu,
            })
    return points


def load_json_iir(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "algorithm": d["algorithm"], "category": d["category"],
        "N": d["signal_length"], "B": d["batch_size"],
        "e_meas": d["energy_per_call_j"], "dp": d["delta_power_w"],
        "t_us": d["time_per_call_us"], "idle_w": d.get("idle_power_w", 0),
        "gpu": d.get("gpu_name", "Unknown"),
    }


def enrich(points, has_torchaudio=False):
    """Add TO predictions and seq_steps to each point."""
    for p in points:
        alg, N, B = p["algorithm"], p["N"], p["B"]
        if alg in TO_MODELS:
            r = predict_to(alg, N, B)
            p["to_c"] = r["to_compute"]
            p["to_m"] = r["to_memory"]
            p["to_t"] = r["to_total"]
            # IIR seq steps depend on whether torchaudio is available
            if alg == "iir_butter4" and has_torchaudio:
                p["seq"] = 0  # fused C++ kernel, single launch
            else:
                p["seq"] = get_seq_steps(alg, N, B)
        else:
            p["to_c"] = 0
            p["to_m"] = 0
            p["to_t"] = 0
            p["seq"] = 0


# =========================================================================
# LOG-SPACE FITTING
# =========================================================================

def fit_loglog_2param(points):
    """
    Fit ln(E) ~ ln(α_c · TO_c + α_m · TO_m) in log space.

    Minimizes sum of (ln(E_meas) - ln(E_pred))² which is equivalent to
    minimizing relative error. Standard for multi-decade data.

    Uses scipy.optimize.minimize with bounds (α_c, α_m > 0).
    """
    to_c = np.array([p["to_c"] for p in points], dtype=np.float64)
    to_m = np.array([p["to_m"] for p in points], dtype=np.float64)
    e = np.array([p["e_meas"] for p in points], dtype=np.float64)
    ln_e = np.log(e)

    def objective(params):
        a_c, a_m = np.exp(params)  # work in log space for positivity
        e_pred = a_c * to_c + a_m * to_m
        e_pred = np.maximum(e_pred, 1e-30)  # prevent log(0)
        return np.sum((np.log(e_pred) - ln_e) ** 2)

    # Initialize with linear-space OLS
    X = np.column_stack([to_c, to_m])
    alpha_init = np.linalg.lstsq(X, e, rcond=None)[0]
    alpha_init = np.maximum(alpha_init, 1e-20)
    x0 = np.log(alpha_init)

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12})
    a_c, a_m = np.exp(result.x)

    e_pred = a_c * to_c + a_m * to_m
    e_pred = np.maximum(e_pred, 1e-30)

    # R² in log space
    ln_pred = np.log(e_pred)
    ss_res = np.sum((ln_e - ln_pred) ** 2)
    ss_tot = np.sum((ln_e - np.mean(ln_e)) ** 2)
    r2_log = 1 - ss_res / ss_tot

    # R² in linear space
    ss_res_lin = np.sum((e - e_pred) ** 2)
    ss_tot_lin = np.sum((e - np.mean(e)) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot_lin

    # Relative errors
    rel_err = np.abs(e - e_pred) / e * 100
    log_ratio = np.abs(np.log10(e_pred / e))  # log10 accuracy

    return {
        "alpha_c": a_c, "alpha_m": a_m,
        "alpha_c_fJ": a_c * 1e15, "alpha_m_fJ": a_m * 1e15,
        "r2_log": r2_log, "r2_lin": r2_lin,
        "mean_rel_err": float(np.mean(rel_err)),
        "median_rel_err": float(np.median(rel_err)),
        "p90_rel_err": float(np.percentile(rel_err, 90)),
        "mean_log10_err": float(np.mean(log_ratio)),
        "n": len(points),
        "e_pred": e_pred,
    }


def fit_loglog_3param(points):
    """
    Fit ln(E) ~ ln(α_c · TO_c + α_m · TO_m + α_o · seq) in log space.
    """
    to_c = np.array([p["to_c"] for p in points], dtype=np.float64)
    to_m = np.array([p["to_m"] for p in points], dtype=np.float64)
    seq = np.array([p["seq"] for p in points], dtype=np.float64)
    e = np.array([p["e_meas"] for p in points], dtype=np.float64)
    ln_e = np.log(e)

    def objective(params):
        a_c, a_m, a_o = np.exp(params)
        e_pred = a_c * to_c + a_m * to_m + a_o * seq
        e_pred = np.maximum(e_pred, 1e-30)
        return np.sum((np.log(e_pred) - ln_e) ** 2)

    # Initialize
    X = np.column_stack([to_c, to_m, seq])
    try:
        from scipy.optimize import nnls
        alpha_init, _ = nnls(X, e)
    except Exception:
        alpha_init = np.linalg.lstsq(X, e, rcond=None)[0]
    alpha_init = np.maximum(alpha_init, 1e-20)
    x0 = np.log(alpha_init)

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-12})
    a_c, a_m, a_o = np.exp(result.x)

    e_pred = a_c * to_c + a_m * to_m + a_o * seq
    e_pred = np.maximum(e_pred, 1e-30)

    ln_pred = np.log(e_pred)
    ss_res = np.sum((ln_e - ln_pred) ** 2)
    ss_tot = np.sum((ln_e - np.mean(ln_e)) ** 2)
    r2_log = 1 - ss_res / ss_tot

    ss_res_lin = np.sum((e - e_pred) ** 2)
    ss_tot_lin = np.sum((e - np.mean(e)) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot_lin

    rel_err = np.abs(e - e_pred) / e * 100

    return {
        "alpha_c": a_c, "alpha_m": a_m, "alpha_o": a_o,
        "alpha_c_fJ": a_c * 1e15, "alpha_m_fJ": a_m * 1e15,
        "alpha_o_uJ": a_o * 1e6,
        "r2_log": r2_log, "r2_lin": r2_lin,
        "mean_rel_err": float(np.mean(rel_err)),
        "median_rel_err": float(np.median(rel_err)),
        "p90_rel_err": float(np.percentile(rel_err, 90)),
        "n": len(points),
        "e_pred": e_pred,
    }


# =========================================================================
# ANALYSIS
# =========================================================================

def analyze_gpu(points, label, has_torchaudio=False):
    print(f"\n{'='*80}")
    print(f"  {label} ({len(points)} points)")
    print(f"{'='*80}")

    enrich(points, has_torchaudio=has_torchaudio)

    # Filter valid
    valid = [p for p in points if p["e_meas"] > 0 and p["to_t"] > 0]
    parallel = [p for p in valid if p["seq"] == 0]
    sequential = [p for p in valid if p["seq"] > 0]
    print(f"  Valid: {len(valid)} (parallel: {len(parallel)}, sequential: {len(sequential)})")

    results = {}

    # ---- Model A: 2-param, parallel only, log-log ----
    if len(parallel) >= 3:
        fitA = fit_loglog_2param(parallel)
        results["A_parallel_2p"] = fitA
        print(f"\n  --- Model A: 2-param log-log (parallel only, n={fitA['n']}) ---")
        print(f"  α_c = {fitA['alpha_c_fJ']:.3f} fJ/TO")
        print(f"  α_m = {fitA['alpha_m_fJ']:.3f} fJ/TO")
        print(f"  α_m/α_c = {fitA['alpha_m']/fitA['alpha_c']:.1f}× (memory wall)")
        print(f"  r²(log) = {fitA['r2_log']:.4f}")
        print(f"  r²(lin) = {fitA['r2_lin']:.4f}")
        print(f"  Median relative error: {fitA['median_rel_err']:.1f}%")
        print(f"  Mean relative error: {fitA['mean_rel_err']:.1f}%")
        print(f"  P90 relative error: {fitA['p90_rel_err']:.1f}%")

        # Store predictions
        for i, p in enumerate(parallel):
            p["e_pred"] = fitA["e_pred"][i]

    # ---- Model B: 3-param log-log (all data) ----
    if len(valid) >= 4 and len(sequential) > 0:
        fitB = fit_loglog_3param(valid)
        results["B_all_3p"] = fitB
        print(f"\n  --- Model B: 3-param log-log (all data, n={fitB['n']}) ---")
        print(f"  α_c = {fitB['alpha_c_fJ']:.3f} fJ/TO")
        print(f"  α_m = {fitB['alpha_m_fJ']:.3f} fJ/TO")
        print(f"  α_o = {fitB['alpha_o_uJ']:.1f} µJ/step ({fitB['alpha_o']*1e3:.3f} mJ/step)")
        print(f"  r²(log) = {fitB['r2_log']:.4f}")
        print(f"  r²(lin) = {fitB['r2_lin']:.4f}")
        print(f"  Median relative error: {fitB['median_rel_err']:.1f}%")
        print(f"  Mean relative error: {fitB['mean_rel_err']:.1f}%")

        # Store predictions for all
        for i, p in enumerate(valid):
            p["e_pred_3p"] = fitB["e_pred"][i]

    # ---- Model comparison ----
    print(f"\n  --- MODEL COMPARISON ---")
    print(f"  {'Model':55s} {'r²(log)':>8s} {'r²(lin)':>8s} {'Med%':>7s} {'Mean%':>7s} {'n':>5s}")
    print(f"  {'-'*90}")
    for name, f in results.items():
        print(f"  {name:55s} {f['r2_log']:>8.4f} {f['r2_lin']:>8.4f} "
              f"{f['median_rel_err']:>6.1f}% {f['mean_rel_err']:>6.1f}% {f['n']:>5d}")

    # ---- Per-category breakdown (parallel, 2-param) ----
    if "A_parallel_2p" in results:
        fitA = results["A_parallel_2p"]
        print(f"\n  --- Per-category (parallel, 2-param log-log) ---")
        by_cat = defaultdict(list)
        for p in parallel:
            by_cat[p["category"]].append(p)

        for cat in sorted(by_cat.keys()):
            pts = by_cat[cat]
            if len(pts) >= 3:
                cf = fit_loglog_2param(pts)
                print(f"  {cat:20s}: r²(log)={cf['r2_log']:.4f}, "
                      f"median_err={cf['median_rel_err']:.1f}%, n={cf['n']}")

    # ---- Per-algorithm detail (parallel) ----
    if parallel and parallel[0].get("e_pred") is not None:
        print(f"\n  --- Per-algorithm (parallel, sorted by error) ---")
        for p in parallel:
            p["rel_err"] = abs(p["e_meas"] - p["e_pred"]) / p["e_meas"] * 100

        by_alg = defaultdict(list)
        for p in parallel:
            by_alg[p["algorithm"]].append(p)

        alg_summary = []
        for alg, pts in by_alg.items():
            errs = [p["rel_err"] for p in pts]
            alg_summary.append((alg, np.median(errs), np.mean(errs), len(pts)))

        alg_summary.sort(key=lambda x: x[1])
        print(f"  {'Algorithm':25s} {'Med%':>8s} {'Mean%':>8s} {'n':>4s}")
        for alg, med, mean, n in alg_summary:
            flag = " *** " if med > 100 else ""
            print(f"  {alg:25s} {med:>7.1f}% {mean:>7.1f}% {n:>4d}{flag}")

    # ---- Head-to-head ranking ----
    if parallel and parallel[0].get("e_pred") is not None:
        print(f"\n  --- Head-to-head ranking accuracy ---")
        pairs = [
            ("fft", "direct_dft"), ("fir_direct", "fir_fft"),
            ("periodogram", "welch"), ("dwt_haar", "dwt_db4"),
            ("cnn_denoiser", "lstm_denoiser"), ("wiener", "cnn_denoiser"),
        ]
        correct, total = 0, 0
        for a, b in pairs:
            pts_a = {p["N"]: p for p in parallel if p["algorithm"] == a}
            pts_b = {p["N"]: p for p in parallel if p["algorithm"] == b}
            for n in sorted(set(pts_a) & set(pts_b)):
                pa, pb = pts_a[n], pts_b[n]
                meas_rank = a if pa["e_meas"] < pb["e_meas"] else b
                pred_rank = a if pa["e_pred"] < pb["e_pred"] else b
                ok = meas_rank == pred_rank
                correct += ok
                total += 1
                s = "✓" if ok else "✗"
                print(f"  {s} {a} vs {b} N={n}: meas={meas_rank}, pred={pred_rank}")
        if total:
            print(f"\n  Ranking accuracy: {correct}/{total} = {correct/total*100:.0f}%")

    return results, valid, parallel, sequential


# =========================================================================
# FIGURES
# =========================================================================

def plot_scatter_loglog(all_results, fig_dir):
    if not HAS_MPL:
        return

    colors = {
        "transform": "#1f77b4", "filter": "#ff7f0e", "adaptive": "#2ca02c",
        "estimation": "#d62728", "spectral": "#9467bd", "decomposition": "#8c564b",
        "compression": "#e377c2", "ml_enhanced": "#7f7f7f",
    }

    n_gpus = len(all_results)
    fig, axes = plt.subplots(1, n_gpus, figsize=(7 * n_gpus, 6), squeeze=False)

    for idx, (label, res, valid, parallel, sequential) in enumerate(all_results):
        ax = axes[0, idx]

        # Plot parallel (filled) and sequential (hollow)
        for p in parallel:
            if p.get("e_pred"):
                c = colors.get(p["category"], "gray")
                ax.scatter(p["e_pred"], p["e_meas"], c=c, s=25, alpha=0.7,
                          edgecolors="none", zorder=3)

        for p in sequential:
            if p.get("e_pred_3p"):
                c = colors.get(p["category"], "gray")
                ax.scatter(p["e_pred_3p"], p["e_meas"], c=c, s=25, alpha=0.5,
                          edgecolors=c, facecolors="none", linewidths=1, zorder=2)

        # Perfect line + 2x bands
        vals = [p["e_meas"] for p in valid if p["e_meas"] > 0]
        lo, hi = min(vals) * 0.3, max(vals) * 3
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=1, alpha=0.5, label="Perfect")
        ax.fill_between([lo, hi], [lo*0.5, hi*0.5], [lo*2, hi*2],
                       alpha=0.08, color="gray", label="2× band")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("TOML Predicted Energy (J)", fontsize=12)
        ax.set_ylabel("Measured Energy (J)", fontsize=12)

        r2 = res.get("A_parallel_2p", {}).get("r2_log", 0)
        med = res.get("A_parallel_2p", {}).get("median_rel_err", 0)
        n_par = res.get("A_parallel_2p", {}).get("n", 0)
        ax.set_title(f"{label}\nParallel: r²(log)={r2:.3f}, median err={med:.0f}%, n={n_par}",
                     fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2, which="both")

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=8, label=cat)
                       for cat, c in colors.items()]
    legend_elements.append(Line2D([0], [0], marker='o', color='gray',
                                  markerfacecolor='none', markersize=8,
                                  label='sequential (3-param)'))
    axes[0, -1].legend(handles=legend_elements, loc='lower right', fontsize=7)

    plt.tight_layout()
    path = fig_dir / "fig1_loglog_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# MAIN
# =========================================================================

def main():
    base = Path(__file__).resolve().parent
    fig_dir = base / "figures"
    fig_dir.mkdir(exist_ok=True)

    all_results = []

    # Local (4090) — has torchaudio
    local_csv = base / "data" / "results" / "all_results.csv"
    if local_csv.exists():
        local = load_csv(str(local_csv), "RTX 4090 Laptop")
        res, valid, par, seq = analyze_gpu(local, "RTX 4090 Laptop GPU",
                                           has_torchaudio=True)
        all_results.append(("RTX 4090 Laptop", res, valid, par, seq))

    # Server (A100) — no torchaudio (Python fallback)
    server_csv = base / "data" / "server_results" / "results" / "all_results.csv"
    if server_csv.exists():
        server = load_csv(str(server_csv), "A100-SXM4-40GB")
        iir_dir = base / "data" / "server_results" / "results" / "filter"
        existing = {(r["algorithm"], r["N"]) for r in server if r["algorithm"] == "iir_butter4"}
        for jf in sorted(iir_dir.glob("iir_butter4_*.json")):
            p = load_json_iir(str(jf))
            if (p["algorithm"], p["N"]) not in existing:
                server.append(p)
                existing.add((p["algorithm"], p["N"]))

        res, valid, par, seq = analyze_gpu(server, "A100-SXM4-40GB",
                                           has_torchaudio=False)
        all_results.append(("A100-SXM4-40GB", res, valid, par, seq))

    # ---- Cross-GPU comparison ----
    if len(all_results) == 2:
        print(f"\n{'='*80}")
        print(f"  CROSS-GPU COMPARISON")
        print(f"{'='*80}")
        r0, r1 = all_results[0][1], all_results[1][1]
        for model_key in ["A_parallel_2p", "B_all_3p"]:
            if model_key in r0 and model_key in r1:
                f0, f1 = r0[model_key], r1[model_key]
                print(f"\n  {model_key}:")
                print(f"    α_c: {f0['alpha_c_fJ']:.2f} vs {f1['alpha_c_fJ']:.2f} fJ/TO "
                      f"(ratio {f0['alpha_c']/f1['alpha_c']:.2f}×)")
                print(f"    α_m: {f0['alpha_m_fJ']:.2f} vs {f1['alpha_m_fJ']:.2f} fJ/TO "
                      f"(ratio {f0['alpha_m']/f1['alpha_m']:.2f}×)")
                if "alpha_o" in f0:
                    print(f"    α_o: {f0['alpha_o_uJ']:.1f} vs {f1['alpha_o_uJ']:.1f} µJ/step "
                          f"(ratio {f0['alpha_o']/f1['alpha_o']:.2f}×)")

    # ---- Figures ----
    if HAS_MPL and all_results:
        print(f"\n  GENERATING FIGURES")
        plot_scatter_loglog(all_results, fig_dir)

    # ---- Summary ----
    summary = {}
    for label, res, _, _, _ in all_results:
        summary[label] = {k: {kk: vv for kk, vv in v.items() if kk != "e_pred"}
                          for k, v in res.items()}
    with open(base / "analysis_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved analysis_summary_v2.json")


if __name__ == "__main__":
    main()
