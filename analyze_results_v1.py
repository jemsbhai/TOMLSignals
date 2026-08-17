"""
TOMLSignals - Camera-Ready Analysis Pipeline (v1)
=================================================
Single source of every number and figure in the camera-ready paper.

Model (F-022): E = a_c*TO_c + a_m*TO_m + a_o*S_o + a_f*S_f, unweighted NNLS,
with S_o = measured GPU commands per invocation (shared/launch_counts.py,
EXP-CR-005 census) for every algorithm. Data: the submission data set
(analyze_cv.load_gpu, v0 loaders unmodified).

Outputs
  data/camera_ready/paper_numbers_v1.json   every number (fit, LOAO ranges,
      error distributions in-sample / LOAO / LOCO / leave-one-category-out,
      head-to-head with pairs, regimes, per-algorithm table, launch-term share,
      effective fJ/TO, cross-GPU transfer, baseline, NCU validation table 37/37)
  paper/numbers_v1.tex                       LaTeX macros for the headline numbers
  paper/figures/*.pdf                        regenerated figures (v0 PDFs copied
      once to paper/figures/v0_submitted/)

Usage (repo root):  python analyze_results_v1.py [--no-figures]
Console is ASCII-only.
Author: Muntaser Syed        Date: August 2026
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import analyze_cv as cv  # noqa: E402  (v0 loaders, metrics, head-to-head, regime labels)
import analyze_estimators as est  # noqa: E402  (fit, loao, loco, locato, partial_transfer)
from shared.launch_counts import get_launch_count  # noqa: E402
from shared.to_model import predict_to  # noqa: E402

OUT_DIR = BASE / "data" / "camera_ready"
NUMBERS_JSON = OUT_DIR / "paper_numbers_v1.json"
MACROS_TEX = BASE / "paper" / "numbers_v1.tex"
NCU_V1 = BASE / "data" / "ncu_profiles" / "ncu_summary_v1.json"
FIG_DIR = BASE / "paper" / "figures"
FIG_V0_DIR = FIG_DIR / "v0_submitted"

GPU_SHORT = {"RTX 4090": "a", "A100 SXM4": "b"}   # macro suffixes

# NCU validation classes (F-023)
NCU_CLASS = {
    "svd": "factorization", "pca": "factorization", "music": "factorization",
    "kalman": "padded", "ekf": "padded", "ukf": "padded", "apa_p4": "padded", "fastica": "padded",
    "jpeg_q50": "padded", "lms": "padded", "nlms": "padded", "particle_1k": "padded", "nmf": "padded",
    "fir_direct": "padded",   # cuDNN conv kernel executes a 32-output-channel tile for 1 channel (32.3x, F-001 data)
    "median": "comparison-only",
}
NCU_CLASS_LABEL = {"analytical": "kernel is the algorithm", "padded": "small-tensor / tiny-matrix library kernels",
                   "factorization": "implementation-dependent factorization", "comparison-only": "comparison-only (no FP32)"}


# =========================================================================
# DATA + FIT
# =========================================================================

def build(label, csv_rel, gpu_name, has_ta, iir_dir):
    points, valid, hashes = cv.load_gpu(csv_rel, gpu_name, has_ta, iir_dir)
    X, y = cv.design(valid)
    S = np.array([get_launch_count(p.algorithm, p.signal_length, p.batch_size, has_torchaudio=has_ta)
                  for p in valid], dtype=float)
    X[:, 2] = S
    return valid, X, y, hashes


def gpu_analysis(label, valid, X, y):
    algs = np.array([p.algorithm for p in valid])
    cats = np.array([p.category for p in valid])
    alpha = est.fit(X, y, "ls")
    p_in = X @ alpha
    p_loao, fold_alpha = est.loao(valid, X, y, "ls")
    p_loco = est.loco(valid, X, y, "ls")
    p_locat = est.locato(valid, X, y, "ls")
    fa = np.array([fold_alpha[a] for a in sorted(fold_alpha)])
    names = sorted(fold_alpha)
    stab = {}
    for j, (name, scale, unit) in enumerate(zip(cv.COEF, cv.COEF_SCALE, cv.COEF_UNIT)):
        col = fa[:, j] * scale
        jmin, jmax = int(np.argmin(col)), int(np.argmax(col))
        stab[name] = {"in_sample": float(alpha[j] * scale), "min": float(col[jmin]), "max": float(col[jmax]),
                      "min_holdout": names[jmin], "max_holdout": names[jmax], "unit": unit}
    out = {"label": label, "n_points": int(len(valid)), "alpha": cv.alpha_dict(alpha),
           "coefficient_stability": stab, "schemes": {}, "predictions": {}}
    for scheme, p in [("in", p_in), ("loao", p_loao), ("loco", p_loco), ("locato", p_locat)]:
        hh = cv.head_to_head(valid, p)
        by_reg = {}
        for reg in ["parallel", "python-loop", "fused-sequential"]:
            m = np.array([cv.regime_of(q) == reg for q in valid])
            if m.sum() >= 2:
                by_reg[reg] = cv.metrics(y[m], p[m])
        by_cat = {}
        for c in sorted(set(cats.tolist())):
            m = cats == c
            by_cat[c] = cv.metrics(y[m], p[m])
        out["schemes"][scheme] = {"metrics": cv.metrics(y, p), "head_to_head": hh,
                                  "by_regime": by_reg, "by_category": by_cat}
    per_alg = {}
    for a in sorted(set(algs.tolist())):
        m = algs == a
        per_alg[a] = {"category": str(cats[m][0]), "n": int(m.sum()),
                      "regime": cv.regime_of(valid[int(np.argmax(m))]),
                      "mdape_in": float(np.median(np.abs(y[m] - p_in[m]) / y[m] * 100)),
                      "mdape_loao": float(np.median(np.abs(y[m] - p_loao[m]) / y[m] * 100)),
                      "ratio_in": float(np.median(p_in[m] / y[m])),
                      "ratio_loao": float(np.median(p_loao[m] / y[m])),
                      "S_o": [int(v) for v in X[m, 2]]}
    out["per_algorithm"] = per_alg
    # launch-term share
    share = alpha[2] * X[:, 2] / np.maximum(p_in, 1e-300)
    out["launch_share_by_regime"] = {}
    for reg in ["parallel", "python-loop", "fused-sequential"]:
        m = np.array([cv.regime_of(q) == reg for q in valid])
        if m.sum():
            out["launch_share_by_regime"][reg] = {"median": float(np.median(share[m])), "max": float(np.max(share[m])),
                                                  "n": int(m.sum())}
    # effective energy per compute TO (parallel regime)
    par = np.array([cv.regime_of(q) == "parallel" for q in valid])
    eff = y[par] / X[par, 0] * 1e15
    tc = X[par, 0]
    dec = np.floor(np.log10(tc)).astype(int)
    out["effective_fJ_per_TO"] = {
        "per_algorithm_median": {a: float(np.median(eff[algs[par] == a])) for a in sorted(set(algs[par].tolist()))},
        "per_decade": {str(int(d)): {"n": int((dec == d).sum()), "median": float(np.median(eff[dec == d])),
                                     "min": float(eff[dec == d].min()), "max": float(eff[dec == d].max())}
                       for d in sorted(set(dec.tolist()))},
        "overall_min": float(eff.min()), "overall_max": float(eff.max()), "n": int(par.sum())}
    # single-parameter baseline (FLOP-counting analogue)
    t = np.array([p.to_total for p in valid], dtype=float)
    a1 = float(np.dot(t, y) / np.dot(t, t))
    out["baseline_1param"] = {"alpha_fJ_per_TO": a1 * 1e15, "metrics": cv.metrics(y, a1 * t)}
    out["predictions"] = {"in": p_in.tolist(), "loao": p_loao.tolist()}
    return out, alpha, p_in


def cross_gpu(fits, data):
    out = {}
    for src in fits:
        for tgt in fits:
            if src == tgt:
                continue
            valid_t, X_t, y_t = data[tgt]
            alg_t = np.array([p.algorithm for p in valid_t])
            key = f"{src} -> {tgt}"
            out[key] = {}
            for name, keep in [("full", (0, 1, 2, 3)), ("keep_c_m_refit_o_f", (0, 1)),
                               ("keep_c_f_refit_m_o", (0, 3)), ("in_sample", ())]:
                a = est.partial_transfer(X_t, y_t, fits[src], keep, "ls", alg_t)
                p = X_t @ a
                hh = cv.head_to_head(valid_t, p)
                out[key][name] = {"alpha": cv.alpha_dict(a), "metrics": cv.metrics(y_t, p),
                                  "head_to_head": f"{hh['correct']}/{hh['total']}"}
    return out


# =========================================================================
# NCU VALIDATION TABLE (37 algorithms)
# =========================================================================

def ncu_table():
    if not NCU_V1.exists():
        return None
    with open(NCU_V1) as f:
        d = json.load(f)
    rows = {}
    for e in d.get("v0_entries", []):
        alg, N, B = e["algorithm"], int(e["N"]), int(e["B"])
        meas = float(e.get("fp32_total", 0.0))
        rows[alg] = {"N": N, "B": B, "measured_fp32": meas, "source": "F-001",
                     "n_kernels": int(e.get("n_kernels", 0)), "dram_words": float(e.get("dram_words", 0.0))}
    for e in d.get("v1_entries", []):
        if e["experiment"] != "EXP-CR-003":
            continue
        alg, N, B = e["algorithm"], int(e["N"]), int(e["B"])
        meas = float(e.get("fp32_total_per_invocation", e.get("measured", {}).get("fp32_total", 0.0)))
        rows[alg] = {"N": N, "B": B, "measured_fp32": meas, "source": "EXP-CR-003 " + e["mode"],
                     "n_kernels": int(e.get("measured", {}).get("n_kernels", e.get("census_kernels", 0)) or 0),
                     "dram_words": float(e.get("dram_words_per_invocation", e.get("measured", {}).get("dram_words", 0.0)) or 0.0)}
    for alg, r in rows.items():
        pred = predict_to(alg, r["N"], r["B"])
        r["analytical_fp32"] = pred["to_compute"] / 5000.0
        r["analytical_mem_words"] = pred["to_memory"] / 10000.0
        r["ratio"] = r["measured_fp32"] / r["analytical_fp32"] if r["analytical_fp32"] > 0 else None
        r["mem_ratio"] = r["dram_words"] / r["analytical_mem_words"] if r["analytical_mem_words"] > 0 and r["dram_words"] else None
        r["class"] = NCU_CLASS.get(alg, "analytical")
    fft_sweep = [{"N": int(e["N"]), "measured_fp32": e["measured"]["fp32_total"], "n_kernels": e["measured"]["n_kernels"],
                  "split_radix": e["split_radix_ops"], "cooley_tukey": e["cooley_tukey_ops"],
                  "ratio_split_radix": e["ratio_meas_over_split_radix"], "ratio_cooley_tukey": e["ratio_meas_over_cooley_tukey"],
                  "kernel": e["kernel_names"][0][0] if e.get("kernel_names") else ""}
                 for e in d.get("v1_entries", []) if e["experiment"] == "EXP-CR-002"]
    classes = {}
    for c in ["analytical", "padded", "factorization"]:
        rs = [r["ratio"] for a, r in rows.items() if r["class"] == c and r["ratio"]]
        classes[c] = {"n": len(rs), "median_ratio": float(np.median(rs)) if rs else None,
                      "min": float(min(rs)) if rs else None, "max": float(max(rs)) if rs else None,
                      "algorithms": sorted(a for a, r in rows.items() if r["class"] == c)}
    return {"rows": rows, "classes": classes, "fft_sweep": fft_sweep, "n_algorithms": len(rows)}


# =========================================================================
# LATEX MACROS
# =========================================================================

def _f(x, nd=1):
    return f"{x:.{nd}f}"


def macros(summary):
    g = summary["gpus"]
    lines = ["% Auto-generated by analyze_results_v1.py from data/camera_ready/paper_numbers_v1.json",
             f"% {summary['started_utc']}  do not edit by hand"]

    def add(name, val):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for label, s in GPU_SHORT.items():
        if label not in g:
            continue
        r = g[label]
        A = r["alpha"]
        add(f"npts{s}", r["n_points"])
        add(f"alphaC{s}", _f(A["alpha_c_fJ_per_TO"], 1))
        add(f"alphaM{s}", _f(A["alpha_m_fJ_per_TO"], 1))
        add(f"alphaO{s}", _f(A["alpha_o_uJ_per_launch"], 1))
        add(f"alphaF{s}", _f(A["alpha_f_uJ_per_step"], 1))
        st = r["coefficient_stability"]
        add(f"alphaCrange{s}", f"{_f(st['alpha_c']['min'],1)}, {_f(st['alpha_c']['max'],1)}")
        add(f"alphaMrange{s}", f"{_f(st['alpha_m']['min'],0)}, {_f(st['alpha_m']['max'],0)}")
        add(f"alphaOrange{s}", f"{_f(st['alpha_o']['min'],0)}, {_f(st['alpha_o']['max'],0)}")
        add(f"alphaFrange{s}", f"{_f(st['alpha_f']['min'],1)}, {_f(st['alpha_f']['max'],1)}")
        add(f"alphaOoverF{s}", _f(A["alpha_o_uJ_per_launch"] / A["alpha_f_uJ_per_step"], 2) if A["alpha_f_uJ_per_step"] > 0 else "n/a")
        # fusion payoff: a Python loop issuing K commands per timestep fused into one kernel costs
        # K*alpha_o versus alpha_f per timestep; K = 7 (LMS) to 54 (UKF) measured in EXP-CR-005
        if A["alpha_f_uJ_per_step"] > 0:
            ratio = A["alpha_o_uJ_per_launch"] / A["alpha_f_uJ_per_step"]
            add(f"fusionGainMin{s}", _f(7 * ratio, 0))
            add(f"fusionGainMax{s}", _f(54 * ratio, 0))
        for scheme, tag in [("in", ""), ("loao", "LOAO"), ("loco", "LOCO"), ("locato", "LOCAT")]:
            m = r["schemes"][scheme]["metrics"]
            hh = r["schemes"][scheme]["head_to_head"]
            add(f"mdape{tag}{s}", _f(m["mdape_pct"], 1))
            add(f"mape{tag}{s}", _f(m["mape_pct"], 1))
            add(f"withinTwo{tag}{s}", _f(m["frac_within_2x"] * 100, 0))
            add(f"withinThree{tag}{s}", _f(m["frac_within_3x"] * 100, 0))
            add(f"withinOneHalf{tag}{s}", _f(m["frac_within_1p5x"] * 100, 0))
            add(f"rTwoLin{tag}{s}", _f(m["r2_linear"], 3))
            add(f"rTwoLog{tag}{s}", _f(m["r2_log10"], 2))
            add(f"hh{tag}{s}", f"{hh['correct']}/{hh['total']}")
            add(f"hhPct{tag}{s}", _f(hh["accuracy"] * 100, 0))
        for reg, tag in [("parallel", "Par"), ("python-loop", "Loop"), ("fused-sequential", "Fused")]:
            m = r["schemes"]["in"]["by_regime"].get(reg)
            if m:
                add(f"mdape{tag}{s}", _f(m["mdape_pct"], 1))
                add(f"withinTwo{tag}{s}", _f(m["frac_within_2x"] * 100, 0))
                add(f"n{tag}{s}", m["n"])
        b = r["baseline_1param"]["metrics"]
        add(f"baseRtwo{s}", _f(b["r2_linear"], 2))
        add(f"baseMdape{s}", _f(b["mdape_pct"], 0))
        add(f"baseWithinTwo{s}", _f(b["frac_within_2x"] * 100, 0))
        eff = r["effective_fJ_per_TO"]
        add(f"effMin{s}", _f(eff["overall_min"], 0))
        add(f"effMax{s}", _f(eff["overall_max"], 0))
        for a_, key in [("direct_dft", "effDft"), ("fft", "effFft"), ("median", "effMedian"), ("svd", "effSvd"), ("pca", "effPca")]:
            v = eff["per_algorithm_median"].get(a_)
            if v is not None:
                add(f"{key}{s}", _f(v, 0))
        sh = r["launch_share_by_regime"]
        if "parallel" in sh:
            add(f"launchSharePar{s}", _f(sh["parallel"]["median"] * 100, 1))
    if "RTX 4090" in g and "A100 SXM4" in g:
        ao = g["RTX 4090"]["alpha"]["alpha_o_uJ_per_launch"] / g["A100 SXM4"]["alpha"]["alpha_o_uJ_per_launch"]
        add("alphaOhostRatio", _f(ao, 1))
        cg = summary.get("cross_gpu_transfer", {})
        for key, tag in [("RTX 4090 -> A100 SXM4", "AtoB"), ("A100 SXM4 -> RTX 4090", "BtoA")]:
            if key in cg:
                for name, t2 in [("full", "Full"), ("keep_c_m_refit_o_f", "KeepCM"), ("keep_c_f_refit_m_o", "KeepCF")]:
                    m = cg[key][name]["metrics"]
                    add(f"xfer{tag}{t2}Mdape", _f(m["mdape_pct"], 0))
                    add(f"xfer{tag}{t2}WithinTwo", _f(m["frac_within_2x"] * 100, 0))
                    add(f"xfer{tag}{t2}HH", cg[key][name]["head_to_head"])
        # pooled head-to-head
        ha = g["RTX 4090"]["schemes"]["in"]["head_to_head"]
        hb = g["A100 SXM4"]["schemes"]["in"]["head_to_head"]
        add("hhPooled", f"{ha['correct'] + hb['correct']}/{ha['total'] + hb['total']}")
        add("hhPooledPct", _f((ha["correct"] + hb["correct"]) / (ha["total"] + hb["total"]) * 100, 0))
        la = g["RTX 4090"]["schemes"]["loao"]["head_to_head"]
        lb = g["A100 SXM4"]["schemes"]["loao"]["head_to_head"]
        add("hhPooledLOAO", f"{la['correct'] + lb['correct']}/{la['total'] + lb['total']}")
        add("hhPooledLOAOPct", _f((la["correct"] + lb["correct"]) / (la["total"] + lb["total"]) * 100, 0))
    ncu = summary.get("ncu_validation")
    if ncu:
        add("ncuNalgs", ncu["n_algorithms"])
        for c, tag in [("analytical", "Analytical"), ("padded", "Padded"), ("factorization", "Fact")]:
            cl = ncu["classes"].get(c)
            if cl and cl["n"]:
                add(f"ncu{tag}N", cl["n"])
                add(f"ncu{tag}Median", _f(cl["median_ratio"], 2))
                add(f"ncu{tag}Min", _f(cl["min"], 2))
                add(f"ncu{tag}Max", _f(cl["max"], 0 if cl["max"] > 20 else 2))
        for row in ncu["fft_sweep"]:
            add(f"fftRatioN{row['N']}".replace("N256", "TwoFiveSix").replace("N1024", "OneK").replace("N4096", "FourK")
                .replace("N16384", "SixteenK").replace("N65536", "SixtyFourK"), _f(row["ratio_split_radix"], 2))
    return "\n".join(lines) + "\n"


# =========================================================================
# FIGURES
# =========================================================================

def figures(summary, data, alphas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.ticker as mticker
    import generate_paper_figures as gpf   # v0 style (rcParams, colors, widths) + unchanged figures 4, 5, 6, 7

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_V0_DIR.mkdir(parents=True, exist_ok=True)
    for f in FIG_DIR.glob("fig*.pdf"):
        tgt = FIG_V0_DIR / f.name
        if not tgt.exists():
            shutil.copy2(f, tgt)
    CW, COL = gpf.COLUMN_WIDTH, gpf.CATEGORY_COLORS

    # ---- Fig 1: predicted vs measured, both GPUs (true column width) -----
    fig, axes = plt.subplots(1, 2, figsize=(CW, CW * 0.6))
    for ax, label in zip(axes, ["RTX 4090", "A100 SXM4"]):
        valid, X, y = data[label]
        p = np.array(summary["gpus"][label]["predictions"]["in"])
        m = summary["gpus"][label]["schemes"]["in"]["metrics"]
        for cat in COL:
            mask = np.array([q.category == cat for q in valid])
            if mask.any():
                ax.scatter(p[mask], y[mask], c=COL[cat], s=6, alpha=0.75, edgecolors="none", zorder=2)
        lo = min(p.min(), y.min()) * 0.3
        hi = max(p.max(), y.max()) * 3
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.6, zorder=1)
        ax.plot([lo, hi], [2 * lo, 2 * hi], color="gray", ls=":", lw=0.5, alpha=0.7, zorder=1)
        ax.plot([lo, hi], [lo / 2, hi / 2], color="gray", ls=":", lw=0.5, alpha=0.7, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted (J)", fontsize=7, labelpad=1)
        ax.set_title(label, fontsize=7, pad=2)
        ax.text(0.03, 0.97, f"median {m['mdape_pct']:.0f}%\n{m['frac_within_2x']*100:.0f}% within 2$\\times$",
                transform=ax.transAxes, fontsize=5.5, va="top")
        ax.tick_params(labelsize=6, pad=1)
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("Measured (J)", fontsize=7, labelpad=1)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=3.5, label=cat.replace("_", " "))
               for cat, c in COL.items()]
    axes[1].legend(handles=handles, loc="lower right", fontsize=4.2, ncol=1, framealpha=0.85,
                   handletextpad=0.2, borderpad=0.3, labelspacing=0.2)
    plt.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "fig1_scatter.pdf")
    plt.close(fig)
    print("  Saved fig1_scatter.pdf")

    # ---- Fig 2: scaling (measured solid, predicted dashed, SAME color) ---
    valid, X, y = data["RTX 4090"]
    p_in = np.array(summary["gpus"]["RTX 4090"]["predictions"]["in"])
    fig, ax = plt.subplots(figsize=(CW, CW * 0.75))
    show = ["fft", "fir_direct", "kalman", "cnn_denoiser", "transformer_denoiser"]
    marks = ["o", "s", "^", "D", "v"]
    for alg, mk in zip(show, marks):
        idx = [i for i, q in enumerate(valid) if q.algorithm == alg]
        idx.sort(key=lambda i: valid[i].signal_length)
        # one point per N: keep the largest-batch configuration (the paper's convention)
        byN = {}
        for i in idx:
            n = valid[i].signal_length
            if n not in byN or valid[i].batch_size > valid[byN[n]].batch_size:
                byN[n] = i
        ii = [byN[n] for n in sorted(byN)]
        ns = [valid[i].signal_length for i in ii]
        line, = ax.plot(ns, [y[i] for i in ii], marker=mk, linestyle="-", markersize=4, label=alg.replace("_", " "))
        ax.plot(ns, [p_in[i] for i in ii], marker=mk, linestyle="--", markersize=3, alpha=0.7,
                color=line.get_color(), markerfacecolor="none")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Signal Length $N$")
    ax.set_ylabel("Energy per Call (J)")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([256, 1024, 4096, 16384])
    ax.legend(fontsize=6.5, loc="upper left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_scaling.pdf")
    plt.close(fig)
    print("  Saved fig2_scaling.pdf")

    # ---- Fig 3: head-to-head (4090) --------------------------------------
    hh = summary["gpus"]["RTX 4090"]["schemes"]["in"]["head_to_head"]
    ok = [(r["E_a_meas"], r["E_b_meas"]) for r in hh["pairs"] if r["correct"]]
    bad = [(r["E_a_meas"], r["E_b_meas"]) for r in hh["pairs"] if not r["correct"]]
    fig, ax = plt.subplots(figsize=(CW, CW * 0.75))
    if ok:
        ax.scatter(*zip(*ok), c="#2ca02c", s=18, alpha=0.75, label=f"Correct ({len(ok)})", zorder=2)
    if bad:
        ax.scatter(*zip(*bad), c="#d62728", s=28, marker="x", linewidths=1.5, label=f"Incorrect ({len(bad)})", zorder=3)
    allv = [v for pr in ok + bad for v in pr]
    lo, hi = min(allv) * 0.3, max(allv) * 3
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Algorithm A Energy (J)")
    ax.set_ylabel("Algorithm B Energy (J)")
    ax.set_title(f"Head-to-Head Ranking, RTX 4090 ({hh['correct']}/{hh['total']})", fontsize=8)
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_headtohead.pdf")
    plt.close(fig)
    print("  Saved fig3_headtohead.pdf")

    # ---- Figs 4, 5, 6, 7: data unchanged, v0 functions -------------------
    rows4090 = gpf.load_gpu_data(BASE / "data" / "results" / "all_results.csv", "RTX 4090", has_torchaudio=True)
    rowsA100 = gpf.load_gpu_data(BASE / "data" / "server_results" / "results" / "all_results.csv", "A100 SXM4",
                                 has_torchaudio=False)
    gpf.fig4_fir_crossover(rows4090)
    gpf.fig5_classical_vs_ml(rows4090)
    gpf.fig6_cross_gpu(rows4090, rowsA100)
    comp = BASE / "data" / "cpu_vs_gpu_comparison.csv"
    if comp.exists():
        gpf.fig7_cpu_vs_gpu(comp)

    # ---- Fig 8: NCU validation, 37 algorithms, three classes -------------
    ncu = summary.get("ncu_validation")
    if ncu:
        fig, ax = plt.subplots(figsize=(CW, CW * 0.8))
        style = {"analytical": ("#1f77b4", "o", "kernel is the algorithm"),
                 "padded": ("#ff7f0e", "s", "small-tensor library kernels"),
                 "factorization": ("#8c564b", "^", "iterative factorization")}
        for cls, (c, mk, lab) in style.items():
            xs = [r["analytical_fp32"] for a, r in ncu["rows"].items() if r["class"] == cls and r["ratio"]]
            ys = [r["measured_fp32"] for a, r in ncu["rows"].items() if r["class"] == cls and r["ratio"]]
            if xs:
                ax.scatter(xs, ys, c=c, marker=mk, s=22, alpha=0.8, label=f"{lab} ({len(xs)})", zorder=2)
        allx = [r["analytical_fp32"] for r in ncu["rows"].values() if r["ratio"]]
        ally = [r["measured_fp32"] for r in ncu["rows"].values() if r["ratio"]]
        lo, hi = min(min(allx), min(ally)) * 0.3, max(max(allx), max(ally)) * 3
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)
        for a in ("fft", "direct_dft", "svd", "kalman", "transformer_denoiser", "esprit"):
            r = ncu["rows"].get(a)
            if r and r["ratio"]:
                ax.annotate(a.replace("_", " "), (r["analytical_fp32"], r["measured_fp32"]),
                            textcoords="offset points", xytext=(4, 3), fontsize=5.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Analytical FP32 operations")
        ax.set_ylabel("Nsight Compute FP32 instr.")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=6, loc="upper left")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig8_ncu_validation.pdf")
        plt.close(fig)
        print("  Saved fig8_ncu_validation.pdf")

    # ---- Fig 9: effective energy per compute TO, both GPUs overlaid ------
    fig, ax = plt.subplots(figsize=(CW, CW * 0.62))
    gpu_marker = {"RTX 4090": ("o", "--"), "A100 SXM4": ("^", ":")}
    for label, (mk, ls) in gpu_marker.items():
        valid, X, y = data[label]
        par = np.array([cv.regime_of(q) == "parallel" for q in valid])
        for cat in COL:
            m = par & np.array([q.category == cat for q in valid])
            if m.any():
                ax.scatter(X[m, 0], y[m] / X[m, 0] * 1e15, c=COL[cat], marker=mk, s=11, alpha=0.75,
                           edgecolors="none", zorder=2)
        ac = summary["gpus"][label]["alpha"]["alpha_c_fJ_per_TO"]
        ax.axhline(ac, color="k", ls=ls, lw=0.8, alpha=0.7, zorder=1)
    ac_a = summary["gpus"]["RTX 4090"]["alpha"]["alpha_c_fJ_per_TO"]
    ac_b = summary["gpus"]["A100 SXM4"]["alpha"]["alpha_c_fJ_per_TO"]
    ax.text(0.99, 0.02, f"$\\alpha_c$: {ac_a:.1f} (RTX 4090, dashed), {ac_b:.1f} fJ/TO (A100, dotted)",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute TOs per invocation")
    ax.set_ylabel("Energy per compute TO (fJ)")
    cat_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=4, label=cat.replace("_", " "))
                   for cat, c in COL.items()]
    gpu_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=4, label="RTX 4090"),
                   Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=4, label="A100 SXM4")]
    leg1 = ax.legend(handles=cat_handles, loc="upper right", fontsize=5, ncol=2, framealpha=0.85,
                     handletextpad=0.2, borderpad=0.3, labelspacing=0.2, columnspacing=0.6)
    ax.add_artist(leg1)
    ax.legend(handles=gpu_handles, loc="center right", fontsize=5, framealpha=0.85,
              handletextpad=0.2, borderpad=0.3, labelspacing=0.2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_energy_per_to.pdf")
    plt.close(fig)
    print("  Saved fig9_energy_per_to.pdf")


# =========================================================================
# MAIN
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("TOMLSignals camera-ready pipeline v1")
    print(f"started {started}")
    summary = {"pipeline": "analyze_results_v1", "started_utc": started, "environment": cv.environment_info(),
               "model": "E = a_c TO_c + a_m TO_m + a_o S_o + a_f S_f; S_o = measured GPU commands (kernels+memcpy+memset) "
                        "per invocation for every algorithm (F-022); unweighted NNLS",
               "input_hashes": {}, "gpus": {}}
    data, alphas = {}, {}
    for label, csv_rel, gpu_name, has_ta, iir_dir in cv.GPUS:
        try:
            valid, X, y, hashes = build(label, csv_rel, gpu_name, has_ta, iir_dir)
        except FileNotFoundError as e:
            print(f"  WARNING missing data for {label}: {e}")
            continue
        summary["input_hashes"].update(hashes)
        out, alpha, p_in = gpu_analysis(label, valid, X, y)
        summary["gpus"][label] = out
        data[label] = (valid, X, y)
        alphas[label] = alpha
        m = out["schemes"]["in"]["metrics"]
        ml = out["schemes"]["loao"]["metrics"]
        A = out["alpha"]
        hh = out["schemes"]["in"]["head_to_head"]
        hl = out["schemes"]["loao"]["head_to_head"]
        print(f"\n  {label}: n={out['n_points']}  alpha_c {A['alpha_c_fJ_per_TO']:.1f} fJ/TO  alpha_m {A['alpha_m_fJ_per_TO']:.1f}  "
              f"alpha_o {A['alpha_o_uJ_per_launch']:.1f} uJ/cmd  alpha_f {A['alpha_f_uJ_per_step']:.1f} uJ/step")
        print(f"    in-sample: MdAPE {m['mdape_pct']:.1f}%  within2x {m['frac_within_2x']*100:.1f}%  within3x {m['frac_within_3x']*100:.1f}%  "
              f"r2lin {m['r2_linear']:.3f}  r2log {m['r2_log10']:.2f}  H2H {hh['correct']}/{hh['total']}")
        print(f"    LOAO:      MdAPE {ml['mdape_pct']:.1f}%  within2x {ml['frac_within_2x']*100:.1f}%  within3x {ml['frac_within_3x']*100:.1f}%  "
              f"r2lin {ml['r2_linear']:.3f}  r2log {ml['r2_log10']:.2f}  H2H {hl['correct']}/{hl['total']}")
        for reg, mm in out["schemes"]["in"]["by_regime"].items():
            print(f"    regime {reg:17s} n={mm['n']:3d} MdAPE {mm['mdape_pct']:5.1f}%  within2x {mm['frac_within_2x']*100:4.0f}%")
        fails = [f"{r['pair']} N={r['N']}" for r in hh["pairs"] if not r["correct"]]
        print("    H2H failures: " + "; ".join(fails))
    if len(data) == 2:
        summary["cross_gpu_transfer"] = cross_gpu(alphas, data)
        ao = summary["gpus"]["RTX 4090"]["alpha"]["alpha_o_uJ_per_launch"] / summary["gpus"]["A100 SXM4"]["alpha"]["alpha_o_uJ_per_launch"]
        print(f"\n  alpha_o(4090)/alpha_o(A100) = {ao:.2f}")
        for k, v in summary["cross_gpu_transfer"].items():
            print(f"  {k}: " + "  ".join(f"{n}: MdAPE {d['metrics']['mdape_pct']:.1f}% within2x {d['metrics']['frac_within_2x']*100:.0f}% H2H {d['head_to_head']}"
                                        for n, d in v.items()))
    ncu = ncu_table()
    if ncu:
        summary["ncu_validation"] = ncu
        summary["input_hashes"][NCU_V1.relative_to(BASE).as_posix()] = cv.sha256_file(NCU_V1)
        print(f"\n  NCU validation: {ncu['n_algorithms']} algorithms")
        for c, cl in ncu["classes"].items():
            if cl["n"]:
                print(f"    {c:14s} n={cl['n']:2d} median ratio {cl['median_ratio']:.2f} [{cl['min']:.2f}, {cl['max']:.2f}]  {cl['algorithms']}")
        for a, r in sorted(ncu["rows"].items(), key=lambda kv: (kv[1]['class'], kv[0])):
            print(f"      {a:22s} {r['class']:15s} N={r['N']:5d} meas {r['measured_fp32']:13.0f} analytical {r['analytical_fp32']:13.0f} "
                  f"ratio {(r['ratio'] if r['ratio'] else float('nan')):8.3f}  {r['source']}")
        print("    fft sweep: " + ", ".join(f"N={s['N']}: {s['ratio_split_radix']:.3f}" for s in ncu["fft_sweep"]))
    with open(NUMBERS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    MACROS_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(MACROS_TEX, "w", encoding="utf-8") as f:
        f.write(macros(summary))
    print(f"\n  Saved {NUMBERS_JSON.relative_to(BASE).as_posix()} and {MACROS_TEX.relative_to(BASE).as_posix()}")
    if not args.no_figures and len(data) == 2:
        print("\n  Figures:")
        figures(summary, data, alphas)
    print(f"  git: {summary['environment'].get('git_commit')}  dirty={summary['environment'].get('git_dirty')}")


if __name__ == "__main__":
    main()
