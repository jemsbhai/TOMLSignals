"""
TOMLSignals - Publication Figure Generator
============================================
Generates 8 publication-quality figures for IEEE MLSP 2026 paper.
Output: PDF files in paper/figures/ for LaTeX inclusion.

Figures:
  1. Predicted vs measured scatter (4-param, both GPUs)
  2. Signal length scaling (key algorithms)
  3. Head-to-head energy pairs
  4. FIR direct vs FFT crossover
  5. Classical vs ML denoising comparison
  6. Cross-GPU coefficient comparison
  7. CPU vs GPU energy comparison
  8. NCU validation: predicted vs measured instruction counts

Usage:
  python generate_paper_figures.py          # all figures
  python generate_paper_figures.py --fig 1  # single figure

Author: Muntaser Syed
Date: May 2026
"""

import csv
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.to_model import predict_to, TO_MODELS, get_seq_steps, get_fused_steps

# =========================================================================
# STYLE
# =========================================================================

# IEEE-compatible: Times font, >=8pt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

CATEGORY_COLORS = {
    "transform":     "#1f77b4",
    "filter":        "#ff7f0e",
    "adaptive":      "#2ca02c",
    "estimation":    "#d62728",
    "spectral":      "#9467bd",
    "decomposition": "#8c564b",
    "compression":   "#e377c2",
    "ml_enhanced":   "#17becf",
}

COLUMN_WIDTH = 3.39  # inches (IEEE double-column)
FIG_DIR = Path(__file__).resolve().parent / "paper" / "figures"


# =========================================================================
# DATA LOADING
# =========================================================================

def load_gpu_data(csv_path, gpu_name, has_torchaudio=True):
    """Load GPU results and compute TO predictions."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in list(r.keys()):
                if k in ("algorithm", "category", "precision", "gpu_name"):
                    continue
                try:
                    r[k] = float(r[k])
                except (ValueError, KeyError):
                    pass
            r["gpu_name"] = gpu_name
            # TO predictions
            alg = r["algorithm"]
            N = int(r["signal_length"])
            B = int(r["batch_size"])
            if alg in TO_MODELS:
                to = predict_to(alg, N, B)
                r["to_compute"] = to["to_compute"]
                r["to_memory"] = to["to_memory"]
                r["to_total"] = to["to_total"]
                r["n_seq_steps"] = get_seq_steps(alg, N, B,
                                                  has_torchaudio=has_torchaudio)
                r["n_fused_steps"] = get_fused_steps(alg, N, B)
            else:
                r["to_compute"] = 0
                r["to_memory"] = 0
                r["to_total"] = 0
                r["n_seq_steps"] = 0
                r["n_fused_steps"] = 0
            rows.append(r)
    return rows


def fit_4param(rows):
    """Fit 4-parameter model, return coefficients and predictions."""
    from scipy.optimize import nnls

    X = np.array([[r["to_compute"], r["to_memory"],
                    r["n_seq_steps"], r["n_fused_steps"]] for r in rows])
    y = np.array([r["energy_per_call_j"] for r in rows])

    valid = (y > 0) & (X[:, 0] > 0) & np.isfinite(y) & np.isfinite(X).all(axis=1)
    X_v, y_v = X[valid], y[valid]
    alpha, _ = nnls(X_v, y_v)
    y_pred = X_v @ alpha

    ss_res = np.sum((y_v - y_pred) ** 2)
    ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Store predictions
    pred_dict = {}
    idx = 0
    for i, r in enumerate(rows):
        if valid[i]:
            pred_dict[(r["algorithm"], int(r["signal_length"]))] = y_pred[idx]
            idx += 1

    return {
        "alpha_c": alpha[0], "alpha_m": alpha[1],
        "alpha_o": alpha[2], "alpha_f": alpha[3],
        "r2": r2, "pred": pred_dict, "valid": valid,
        "y_valid": y_v, "y_pred": y_pred,
    }


def load_cpu_data(csv_path):
    """Load CPU benchmark results."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in list(r.keys()):
                if k in ("algorithm", "category", "precision", "cpu_name"):
                    continue
                try:
                    r[k] = float(r[k])
                except (ValueError, KeyError):
                    pass
            rows.append(r)
    return rows


# =========================================================================
# FIGURE 1: Predicted vs Measured Scatter
# =========================================================================

def fig1_scatter(gpu_4090, gpu_a100, fit_4090, fit_a100):
    """Log-log scatter: TOML predicted vs measured energy, both GPUs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * COLUMN_WIDTH + 0.4, COLUMN_WIDTH))

    for ax, rows, fit, title in [
        (ax1, gpu_4090, fit_4090, f"RTX 4090 ($r^2={fit_4090['r2']:.3f}$)"),
        (ax2, gpu_a100, fit_a100, f"A100 SXM4 ($r^2={fit_a100['r2']:.3f}$)"),
    ]:
        for r in rows:
            key = (r["algorithm"], int(r["signal_length"]))
            if key in fit["pred"] and r["energy_per_call_j"] > 0:
                c = CATEGORY_COLORS.get(r["category"], "gray")
                ax.scatter(fit["pred"][key], r["energy_per_call_j"],
                          c=c, s=12, alpha=0.7, edgecolors="none", zorder=2)

        vals = [r["energy_per_call_j"] for r in rows
                if (r["algorithm"], int(r["signal_length"])) in fit["pred"]
                and r["energy_per_call_j"] > 0]
        preds = [fit["pred"][(r["algorithm"], int(r["signal_length"]))]
                 for r in rows
                 if (r["algorithm"], int(r["signal_length"])) in fit["pred"]
                 and r["energy_per_call_j"] > 0]
        lo = min(min(vals), min(preds)) * 0.3
        hi = max(max(vals), max(preds)) * 3
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted Energy (J)")
        ax.set_ylabel("Measured Energy (J)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")

    # Legend
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                       markersize=5, label=cat.replace("_", " "))
               for cat, c in CATEGORY_COLORS.items()]
    ax2.legend(handles=handles, loc="lower right", fontsize=6, ncol=1,
               framealpha=0.8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_scatter.pdf")
    print("  Saved fig1_scatter.pdf")
    plt.close()


# =========================================================================
# FIGURE 2: Signal Length Scaling
# =========================================================================

def fig2_scaling(gpu_4090, fit_4090):
    """Energy vs N for representative algorithms."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.85))

    show = ["fft", "fir_direct", "kalman", "cnn_denoiser", "transformer_denoiser"]
    styles = [("o", "-"), ("s", "-"), ("^", "-"), ("D", "-"), ("v", "-")]

    for alg, (mk, ls) in zip(show, styles):
        pts = sorted([r for r in gpu_4090 if r["algorithm"] == alg],
                     key=lambda r: r["signal_length"])
        if not pts:
            continue
        ns = [int(r["signal_length"]) for r in pts]
        e_meas = [r["energy_per_call_j"] for r in pts]
        e_pred = [fit_4090["pred"].get((alg, n), None) for n in ns]

        ax.plot(ns, e_meas, marker=mk, linestyle=ls, markersize=4,
                label=alg.replace("_", " "))
        pred_ns = [n for n, p in zip(ns, e_pred) if p is not None]
        pred_es = [p for p in e_pred if p is not None]
        if pred_es:
            ax.plot(pred_ns, pred_es, marker=mk, linestyle="--",
                    markersize=3, alpha=0.4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Signal Length $N$")
    ax.set_ylabel("Energy per Call (J)")
    ax.legend(fontsize=7, loc="upper left")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([256, 1024, 4096, 16384])

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_scaling.pdf")
    print("  Saved fig2_scaling.pdf")
    plt.close()


# =========================================================================
# FIGURE 3: Head-to-Head Pairs
# =========================================================================

def fig3_headtohead(gpu_4090, fit_4090):
    """Head-to-head energy pairs: correct vs incorrect predictions."""
    pairs = [
        ("fft", "direct_dft"), ("fir_direct", "fir_fft"),
        ("lms", "rls"), ("kalman", "ukf"),
        ("periodogram", "welch"), ("cnn_denoiser", "lstm_denoiser"),
        ("lstm_denoiser", "transformer_denoiser"), ("wiener", "cnn_denoiser"),
    ]

    correct_x, correct_y = [], []
    wrong_x, wrong_y = [], []
    labels_correct, labels_wrong = [], []

    by_alg = defaultdict(dict)
    for r in gpu_4090:
        by_alg[r["algorithm"]][int(r["signal_length"])] = r

    for a, b in pairs:
        common_n = set(by_alg[a].keys()) & set(by_alg[b].keys())
        for n in sorted(common_n):
            ra, rb = by_alg[a][n], by_alg[b][n]
            ea, eb = ra["energy_per_call_j"], rb["energy_per_call_j"]
            pa = fit_4090["pred"].get((a, n))
            pb = fit_4090["pred"].get((b, n))
            if pa is None or pb is None:
                continue
            meas_a_wins = ea < eb
            pred_a_wins = pa < pb
            if meas_a_wins == pred_a_wins:
                correct_x.append(ea)
                correct_y.append(eb)
            else:
                wrong_x.append(ea)
                wrong_y.append(eb)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.9))
    ax.scatter(correct_x, correct_y, c="#2ca02c", s=18, alpha=0.7,
               label=f"Correct ({len(correct_x)})", zorder=2)
    ax.scatter(wrong_x, wrong_y, c="#d62728", s=25, marker="x", linewidths=1.5,
               label=f"Incorrect ({len(wrong_x)})", zorder=3)

    all_v = correct_x + correct_y + wrong_x + wrong_y
    if all_v:
        lo, hi = min(all_v) * 0.3, max(all_v) * 3
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Algorithm A Energy (J)")
    ax.set_ylabel("Algorithm B Energy (J)")
    ax.set_title("Head-to-Head Ranking Accuracy")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_headtohead.pdf")
    print("  Saved fig3_headtohead.pdf")
    plt.close()

    total = len(correct_x) + len(wrong_x)
    print(f"    Ranking: {len(correct_x)}/{total} = "
          f"{len(correct_x)/total*100:.0f}%")


# =========================================================================
# FIGURE 4: FIR Crossover
# =========================================================================

def fig4_fir_crossover(gpu_4090):
    """FIR direct vs FFT-based: energy crossover with signal length."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.8))

    for alg, label, mk in [("fir_direct", "FIR Direct", "o"),
                             ("fir_fft", "FIR FFT", "s")]:
        pts = sorted([r for r in gpu_4090 if r["algorithm"] == alg],
                     key=lambda r: r["signal_length"])
        if not pts:
            continue
        ns = [int(r["signal_length"]) for r in pts]
        es = [r["energy_per_call_j"] for r in pts]
        ax.plot(ns, es, marker=mk, label=label, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Signal Length $N$")
    ax.set_ylabel("Energy per Call (J)")
    ax.set_title("FIR Direct vs. FFT-based Convolution")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([256, 1024, 4096, 16384])

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_fir_crossover.pdf")
    print("  Saved fig4_fir_crossover.pdf")
    plt.close()


# =========================================================================
# FIGURE 5: Classical vs ML Denoising
# =========================================================================

def fig5_classical_vs_ml(gpu_4090):
    """Energy comparison: Wiener vs CNN/LSTM/Transformer denoisers."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.85))

    denoisers = ["wiener", "cnn_denoiser", "lstm_denoiser", "transformer_denoiser"]
    labels = ["Wiener", "CNN", "LSTM", "Transformer"]
    markers = ["o", "s", "^", "D"]

    for alg, label, mk in zip(denoisers, labels, markers):
        pts = sorted([r for r in gpu_4090 if r["algorithm"] == alg],
                     key=lambda r: r["signal_length"])
        if not pts:
            continue
        ns = [int(r["signal_length"]) for r in pts]
        es = [r["energy_per_call_j"] for r in pts]
        ax.plot(ns, es, marker=mk, label=label, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Signal Length $N$")
    ax.set_ylabel("Energy per Call (J)")
    ax.set_title("Classical vs. ML-Enhanced Denoising")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([256, 1024, 4096, 16384])

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_denoising.pdf")
    print("  Saved fig5_denoising.pdf")
    plt.close()


# =========================================================================
# FIGURE 6: Cross-GPU Coefficient Comparison
# =========================================================================

def fig6_cross_gpu(gpu_4090, gpu_a100):
    """Cross-GPU: energy scatter (same algorithm, different hardware)."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.9))

    lookup_4090 = {(r["algorithm"], int(r["signal_length"])): r
                   for r in gpu_4090}
    lookup_a100 = {(r["algorithm"], int(r["signal_length"])): r
                   for r in gpu_a100}

    common = set(lookup_4090.keys()) & set(lookup_a100.keys())
    for key in sorted(common):
        r4 = lookup_4090[key]
        ra = lookup_a100[key]
        e4 = r4["energy_per_call_j"]
        ea = ra["energy_per_call_j"]
        if e4 > 0 and ea > 0:
            c = CATEGORY_COLORS.get(r4["category"], "gray")
            ax.scatter(e4, ea, c=c, s=12, alpha=0.7, edgecolors="none")

    all_e = [r["energy_per_call_j"] for r in gpu_4090 if r["energy_per_call_j"] > 0] + \
            [r["energy_per_call_j"] for r in gpu_a100 if r["energy_per_call_j"] > 0]
    lo, hi = min(all_e) * 0.3, max(all_e) * 3
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RTX 4090 Energy (J)")
    ax.set_ylabel("A100 SXM4 Energy (J)")
    ax.set_title("Cross-GPU Energy Comparison")
    ax.set_aspect("equal", adjustable="datalim")

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                       markersize=5, label=cat.replace("_", " "))
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=5.5, ncol=2,
              framealpha=0.8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_cross_gpu.pdf")
    print("  Saved fig6_cross_gpu.pdf")
    plt.close()


# =========================================================================
# FIGURE 7: CPU vs GPU Comparison
# =========================================================================

def fig7_cpu_vs_gpu(comparison_path):
    """CPU/GPU energy ratio by category — the three-regime result."""
    rows = []
    with open(comparison_path) as f:
        for r in csv.DictReader(f):
            r["ratio"] = float(r["ratio"])
            r["N"] = int(float(r["N"]))
            rows.append(r)

    # Group by category
    cat_order = ["transform", "filter", "spectral", "ml_enhanced",
                 "compression", "decomposition", "estimation", "adaptive"]
    cat_labels = ["Transform", "Filter", "Spectral", "ML-Enhanced",
                  "Compression", "Decomp.", "Estimation", "Adaptive"]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 1.0))

    positions = []
    medians = []
    bp_data = []
    for i, cat in enumerate(cat_order):
        ratios = [r["ratio"] for r in rows if r["category"] == cat]
        if ratios:
            bp_data.append(ratios)
            positions.append(i)
            medians.append(np.median(ratios))

    bp = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=3))

    for i, (patch, cat) in enumerate(zip(bp["boxes"], cat_order)):
        color = CATEGORY_COLORS.get(cat, "gray")
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=1.0, color="k", linestyle="--", lw=1, alpha=0.7, zorder=0)
    ax.annotate("GPU wins $\\uparrow$", xy=(0.02, 0.75), xycoords="axes fraction",
                fontsize=7, color="#2ca02c")
    ax.annotate("CPU wins $\\downarrow$", xy=(0.02, 0.15), xycoords="axes fraction",
                fontsize=7, color="#d62728")

    ax.set_yscale("log")
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CPU / GPU Energy Ratio")
    ax.set_title("CPU vs. GPU Energy Efficiency by Category")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_cpu_vs_gpu.pdf")
    print("  Saved fig7_cpu_vs_gpu.pdf")
    plt.close()


# =========================================================================
# FIGURE 8: NCU Validation
# =========================================================================

def fig8_ncu_validation(ncu_path):
    """NCU-predicted vs measured instruction counts for validated algorithms."""
    with open(ncu_path) as f:
        ncu = json.load(f)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.85))

    algs, pred, meas = [], [], []
    for data in ncu:
        alg_name = data["algorithm"]
        if "fp32_total" not in data or data["fp32_total"] == 0:
            continue
        measured = data["fp32_total"]
        N = data.get("N", 4096)
        B = data.get("B", 1)
        if alg_name in TO_MODELS:
            to = predict_to(alg_name, N, B)
            predicted_fma = to["to_compute"] / 5000
            if predicted_fma > 0:
                algs.append(alg_name)
                pred.append(predicted_fma)
                meas.append(measured)

    if not pred:
        print("    No NCU data matched TO models, skipping fig8")
        return

    pred, meas = np.array(pred), np.array(meas)
    ratios = meas / pred

    ax.scatter(pred, meas, c="#1f77b4", s=25, alpha=0.7, zorder=2)

    # Label key points
    for i, alg in enumerate(algs):
        if alg in ("fft", "direct_dft", "cnn_denoiser", "svd"):
            ax.annotate(alg.replace("_", " "), (pred[i], meas[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=6)

    lo = min(min(pred), min(meas)) * 0.3
    hi = max(max(pred), max(meas)) * 3
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TO-Predicted FP32 Ops")
    ax.set_ylabel("NCU Measured FP32 Ops")
    ax.set_title("Nsight Compute Validation")
    ax.set_aspect("equal", adjustable="datalim")

    # Annotation with median ratio
    med_ratio = np.median(ratios)
    ax.text(0.05, 0.92, f"Median ratio: {med_ratio:.2f}$\\times$",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_ncu_validation.pdf")
    print("  Saved fig8_ncu_validation.pdf")
    plt.close()


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=int, help="Generate single figure (1-8)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    gpu_4090 = load_gpu_data(
        base / "data" / "results" / "all_results.csv",
        "RTX 4090", has_torchaudio=True)
    gpu_a100 = load_gpu_data(
        base / "data" / "server_results" / "results" / "all_results.csv",
        "A100 SXM4", has_torchaudio=False)

    print(f"  4090: {len(gpu_4090)} points, A100: {len(gpu_a100)} points")

    print("Fitting 4-parameter model...")
    fit_4090 = fit_4param(gpu_4090)
    fit_a100 = fit_4param(gpu_a100)
    print(f"  4090 r²={fit_4090['r2']:.4f}, A100 r²={fit_a100['r2']:.4f}")

    print(f"\nGenerating figures to {FIG_DIR}/")

    if args.fig is None or args.fig == 1:
        print("\nFig 1: Predicted vs Measured Scatter")
        fig1_scatter(gpu_4090, gpu_a100, fit_4090, fit_a100)

    if args.fig is None or args.fig == 2:
        print("\nFig 2: Signal Length Scaling")
        fig2_scaling(gpu_4090, fit_4090)

    if args.fig is None or args.fig == 3:
        print("\nFig 3: Head-to-Head Pairs")
        fig3_headtohead(gpu_4090, fit_4090)

    if args.fig is None or args.fig == 4:
        print("\nFig 4: FIR Crossover")
        fig4_fir_crossover(gpu_4090)

    if args.fig is None or args.fig == 5:
        print("\nFig 5: Classical vs ML Denoising")
        fig5_classical_vs_ml(gpu_4090)

    if args.fig is None or args.fig == 6:
        print("\nFig 6: Cross-GPU Comparison")
        fig6_cross_gpu(gpu_4090, gpu_a100)

    if args.fig is None or args.fig == 7:
        print("\nFig 7: CPU vs GPU")
        comp_path = base / "data" / "cpu_vs_gpu_comparison.csv"
        if comp_path.exists():
            fig7_cpu_vs_gpu(comp_path)
        else:
            print("  SKIP: cpu_vs_gpu_comparison.csv not found")

    if args.fig is None or args.fig == 8:
        print("\nFig 8: NCU Validation")
        ncu_path = base / "data" / "ncu_profiles" / "ncu_summary.json"
        if ncu_path.exists():
            fig8_ncu_validation(ncu_path)
        else:
            print("  SKIP: ncu_summary.json not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
