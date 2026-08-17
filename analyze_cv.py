"""
TOMLSignals - Camera-Ready Analysis (EXP-CR-001)
=================================================
Error distribution, cross-validation, and cross-GPU transfer for the
four-parameter energy model  E = a_c*TO_c + a_m*TO_m + a_o*S_o + a_f*S_f.

This script does NOT modify analyze_results.py (v0, frozen at submission).
It imports the v0 loaders and TO prediction code, so the data set and the
design matrix are identical to the submitted paper. The in-sample fit is
checked against v0 fit_four_parameter() before anything else is reported.

Computes, per GPU:
  1. In-sample fit (must reproduce Table 3 of the submitted paper) plus a
     scale-free error distribution: MAPE, MdAPE, p90 APE, fraction of points
     within 1.5x / 2x / 3x, geometric-mean multiplicative error, r2 in linear
     space (what the submitted paper reported) and r2 in log10 space.
  2. Leave-one-algorithm-out (LOAO) cross-validation: fit on 36 algorithms,
     predict every configuration of the held-out algorithm.
  3. Leave-one-configuration-out (LOCO) cross-validation: fit on n-1 points,
     predict the held-out point.
  4. Head-to-head ranking accuracy in-sample, under LOAO, and under LOCO
     (same 8 pairs as v0).
  5. Cross-GPU coefficient transfer: apply one GPU's coefficients to the other
     GPU's data (full transfer), transfer subsets of coefficients and refit the
     rest (partial transfer), and single-scalar recalibration.
  6. Single-parameter baseline E = a*TO_total with the same metric set.

Outputs (directory created if missing):
  data/camera_ready/exp_cr_001_cv_results.json
  data/camera_ready/exp_cr_001_per_algorithm.csv
Console output is ASCII-only (Windows cp1252 safe).

Usage (from repo root):  python analyze_cv.py

Author: Muntaser Syed
Date: August 2026
"""

import copy
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:  # never let a Unicode print from an imported module crash the run
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import analyze_results as ar  # noqa: E402  (v0, frozen: loaders + TO predictions)
from scipy.optimize import nnls  # noqa: E402

OUT_DIR = BASE / "data" / "camera_ready"

# Head-to-head pairs: identical to analyze_results.analyze_gpu (v0).
PAIRS = [
    ("fft", "direct_dft", "FFT vs DFT"),
    ("fir_direct", "fir_fft", "FIR direct vs FFT"),
    ("lms", "rls", "LMS vs RLS"),
    ("kalman", "ukf", "Kalman vs UKF"),
    ("periodogram", "welch", "Periodogram vs Welch"),
    ("cnn_denoiser", "lstm_denoiser", "CNN vs LSTM denoiser"),
    ("lstm_denoiser", "transformer_denoiser", "LSTM vs Transformer"),
    ("wiener", "cnn_denoiser", "Wiener vs CNN"),
]

# (label, results csv, gpu_name string used by v0, has_torchaudio, IIR rerun dir)
GPUS = [
    ("RTX 4090", "data/results/all_results.csv",
     "NVIDIA GeForce RTX 4090 Laptop GPU", True, None),
    ("A100 SXM4", "data/server_results/results/all_results.csv",
     "NVIDIA A100-SXM4-40GB", False, "data/server_results/results/filter"),
]

COEF = ["alpha_c", "alpha_m", "alpha_o", "alpha_f"]
COEF_SHORT = ["c", "m", "o", "f"]
COEF_SCALE = [1e15, 1e15, 1e6, 1e6]          # J/TO -> fJ/TO ; J/step -> uJ/step
COEF_UNIT = ["fJ/TO", "fJ/TO", "uJ/launch", "uJ/step"]


# =========================================================================
# DATA (replicates analyze_results.main() exactly)
# =========================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gpu(csv_rel, gpu_name, has_torchaudio, iir_dir_rel):
    """Load one GPU's data exactly as analyze_results.main() does (v0)."""
    csv_path = BASE / csv_rel
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    points = ar.load_csv(str(csv_path), gpu_name=gpu_name)
    hashes = {Path(csv_rel).as_posix(): sha256_file(csv_path)}
    if iir_dir_rel:
        iir_dir = BASE / iir_dir_rel
        existing = {(p.algorithm, p.signal_length) for p in points
                    if p.algorithm == "iir_butter4"}
        for jf in sorted(iir_dir.glob("iir_butter4_*.json")):
            p = ar.load_json(str(jf))
            key = (p.algorithm, p.signal_length)
            if key not in existing:
                points.append(p)
                existing.add(key)
                hashes[jf.relative_to(BASE).as_posix()] = sha256_file(jf)
    ar.compute_to_predictions(points, has_torchaudio=has_torchaudio)
    # Validity: analyze_gpu filter (E>0, TO_total>0) AND fit_four_parameter
    # filter (E>0, TO_c>0, finite). In v0 both applied to the fitted set.
    valid = [p for p in points
             if p.energy_per_call_j > 0 and p.to_total > 0 and p.to_compute > 0
             and np.isfinite(p.energy_per_call_j)]
    return points, valid, hashes


def design(points):
    X = np.array([[p.to_compute, p.to_memory, p.n_seq_steps, p.n_fused_steps]
                  for p in points], dtype=float)
    y = np.array([p.energy_per_call_j for p in points], dtype=float)
    return X, y


def regime_of(p):
    if p.n_fused_steps > 0:
        return "fused-sequential"
    if p.n_seq_steps > 0:
        return "python-loop"
    return "parallel"


# =========================================================================
# FITTING AND METRICS
# =========================================================================

def fit_nnls(X, y):
    alpha, _ = nnls(X, y)
    return alpha


def metrics(y_true, y_pred):
    """Scale-free error metrics for a set of (measured, predicted) energies."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    res = y_true - y_pred
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    pos = y_pred > 0
    if int(pos.sum()) >= 2:
        lt = np.log10(y_true[pos])
        lp = np.log10(y_pred[pos])
        sst = float(np.sum((lt - lt.mean()) ** 2))
        r2_log = 1.0 - float(np.sum((lt - lp) ** 2)) / sst if sst > 0 else float("nan")
        gm = float(10 ** np.mean(np.abs(lp - lt)))
    else:
        r2_log, gm = float("nan"), float("nan")

    ape = np.abs(res) / y_true * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pos,
                         np.maximum(y_pred / y_true, y_true / np.where(pos, y_pred, 1.0)),
                         np.inf)
    return {
        "n": n,
        "n_nonpositive_pred": int((~pos).sum()),
        "r2_linear": float(r2_lin),
        "r2_log10": float(r2_log),
        "mape_pct": float(np.mean(ape)),
        "mdape_pct": float(np.median(ape)),
        "p90_ape_pct": float(np.percentile(ape, 90)),
        "max_ape_pct": float(np.max(ape)),
        "frac_within_1p5x": float(np.mean(ratio <= 1.5)),
        "frac_within_2x": float(np.mean(ratio <= 2.0)),
        "frac_within_3x": float(np.mean(ratio <= 3.0)),
        "geo_mean_mult_error": gm,
    }


def alpha_dict(alpha):
    d = {}
    for name, val, scale, unit in zip(COEF, alpha, COEF_SCALE, COEF_UNIT):
        d[name] = float(val)
        d[name + "_" + unit.replace("/", "_per_")] = float(val * scale)
    return d


def head_to_head(valid, pred):
    """Ranking accuracy over PAIRS. pred is aligned with valid.
    Dict-comprehension per algorithm reproduces v0 exactly (last point at a
    given N wins if an algorithm has several batch sizes at that N)."""
    idx = {id(p): i for i, p in enumerate(valid)}
    correct = total = 0
    rows = []
    for alg_a, alg_b, label in PAIRS:
        pts_a = {p.signal_length: p for p in valid if p.algorithm == alg_a}
        pts_b = {p.signal_length: p for p in valid if p.algorithm == alg_b}
        for n in sorted(set(pts_a) & set(pts_b)):
            pa, pb = pts_a[n], pts_b[n]
            ea, eb = pa.energy_per_call_j, pb.energy_per_call_j
            qa, qb = pred[idx[id(pa)]], pred[idx[id(pb)]]
            meas_a = ea < eb
            pred_a = qa < qb
            ok = bool(meas_a == pred_a)
            correct += ok
            total += 1
            rows.append({"pair": label, "N": int(n),
                         "measured_winner": alg_a if meas_a else alg_b,
                         "predicted_winner": alg_a if pred_a else alg_b,
                         "correct": ok,
                         "E_a_meas": float(ea), "E_b_meas": float(eb),
                         "E_a_pred": float(qa), "E_b_pred": float(qb)})
    return {"correct": int(correct), "total": int(total),
            "accuracy": float(correct / total) if total else float("nan"),
            "pairs": rows}


# =========================================================================
# CROSS-VALIDATION
# =========================================================================

def loao(valid, X, y):
    """Leave-one-algorithm-out. Returns out-of-sample predictions aligned with
    valid, plus the per-fold coefficient vector."""
    algs = sorted(set(p.algorithm for p in valid))
    pred = np.full(len(valid), np.nan)
    fold_alpha = {}
    for a in algs:
        test = np.array([p.algorithm == a for p in valid])
        alpha = fit_nnls(X[~test], y[~test])
        pred[test] = X[test] @ alpha
        fold_alpha[a] = alpha
    return pred, fold_alpha


def loco(X, y):
    """Leave-one-configuration-out (one benchmark point held out at a time)."""
    n = len(y)
    pred = np.empty(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        alpha = fit_nnls(X[mask], y[mask])
        pred[i] = X[i] @ alpha
        mask[i] = True
    return pred


def partial_transfer(X_t, y_t, alpha_src, keep):
    """Transfer the coefficients in `keep` (column indices) from the source
    GPU unchanged; refit the remaining coefficients on the target GPU by NNLS
    on the residual. keep=() is the in-sample fit; keep=(0,1,2,3) is full
    transfer."""
    keep = list(keep)
    refit = [j for j in range(4) if j not in keep]
    alpha = np.zeros(4)
    if keep:
        alpha[keep] = alpha_src[keep]
        resid = y_t - X_t[:, keep] @ alpha_src[keep]
    else:
        resid = y_t.copy()
    if refit:
        a_refit, _ = nnls(X_t[:, refit], resid)
        alpha[refit] = a_refit
    return alpha


# =========================================================================
# ANALYSIS PER GPU
# =========================================================================

def analyze_gpu(label, valid, X, y):
    out = {"label": label, "n_points": int(len(valid))}
    print("\n" + "=" * 78)
    print(f"  {label}: {len(valid)} valid points")
    print("=" * 78)

    # ---- 1. In-sample fit, cross-checked against v0 ---------------------
    alpha = fit_nnls(X, y)
    v0 = ar.fit_four_parameter(copy.deepcopy(valid))
    v0_alpha = np.array([v0["alpha_c"], v0["alpha_m"], v0["alpha_o"], v0["alpha_f"]])
    same = np.allclose(alpha, v0_alpha, rtol=1e-8, atol=0.0)
    print(f"\n  [check] in-sample NNLS reproduces v0 fit_four_parameter: {same}")
    print(f"          v0 r2 (linear) = {v0['r_squared']:.4f}")
    if not same:
        print("  ERROR: coefficient mismatch vs v0 -- aborting this GPU")
        print("     this:", alpha)
        print("     v0:  ", v0_alpha)
        out["error"] = "coefficient mismatch vs v0"
        return out, None
    pred_in = X @ alpha
    m_in = metrics(y, pred_in)
    out["in_sample"] = {"alpha": alpha_dict(alpha), "metrics": m_in,
                        "v0_r2_linear": float(v0["r_squared"])}
    print("\n  In-sample coefficients:")
    for name, val, scale, unit in zip(COEF, alpha, COEF_SCALE, COEF_UNIT):
        print(f"    {name:8s} = {val*scale:10.2f} {unit}")
    print_metrics("In-sample (all points)", m_in)

    # by regime
    out["in_sample"]["metrics_by_regime"] = {}
    for reg in ["parallel", "python-loop", "fused-sequential"]:
        mask = np.array([regime_of(p) == reg for p in valid])
        if mask.sum() >= 2:
            mreg = metrics(y[mask], pred_in[mask])
            out["in_sample"]["metrics_by_regime"][reg] = mreg
            print_metrics(f"In-sample, regime = {reg}", mreg)

    hh_in = head_to_head(valid, pred_in)
    out["in_sample"]["head_to_head"] = hh_in
    print(f"\n  Head-to-head (in-sample): {hh_in['correct']}/{hh_in['total']} "
          f"= {hh_in['accuracy']*100:.0f}%")

    # ---- 6. Single-parameter baseline -----------------------------------
    t = np.array([p.to_total for p in valid], dtype=float)
    a1 = float(np.dot(t, y) / np.dot(t, t))
    m1 = metrics(y, a1 * t)
    out["baseline_1param"] = {"alpha_fJ_per_TO": a1 * 1e15, "metrics": m1}
    print_metrics("Baseline E = a*TO_total (1 parameter)", m1)

    # ---- 2. LOAO ---------------------------------------------------------
    pred_loao, fold_alpha = loao(valid, X, y)
    m_loao = metrics(y, pred_loao)
    lstm_mask = np.array([p.algorithm == "lstm_denoiser" for p in valid])
    m_loao_ex = metrics(y[~lstm_mask], pred_loao[~lstm_mask]) if lstm_mask.any() else None
    hh_loao = head_to_head(valid, pred_loao)
    out["loao"] = {"metrics_all": m_loao,
                   "metrics_excluding_lstm": m_loao_ex,
                   "head_to_head": hh_loao,
                   "fold_alpha": {a: alpha_dict(v) for a, v in fold_alpha.items()}}
    print_metrics("LOAO out-of-sample (all points)", m_loao)
    if m_loao_ex:
        print_metrics("LOAO out-of-sample, excluding lstm_denoiser "
                      "(alpha_f unidentifiable when LSTM is held out)", m_loao_ex)
    print(f"\n  Head-to-head (LOAO out-of-sample): {hh_loao['correct']}/{hh_loao['total']} "
          f"= {hh_loao['accuracy']*100:.0f}%")
    for r in hh_loao["pairs"]:
        flag = "ok " if r["correct"] else "XX "
        print(f"    {flag}{r['pair']:24s} N={r['N']:<6d} meas={r['measured_winner']:22s} "
              f"pred={r['predicted_winner']}")

    # coefficient stability across LOAO folds
    fa = np.array([fold_alpha[a] for a in sorted(fold_alpha)])
    names = sorted(fold_alpha)
    print("\n  Coefficient range across LOAO folds (held-out algorithm driving the extreme):")
    stab = {}
    for j, (name, scale, unit) in enumerate(zip(COEF, COEF_SCALE, COEF_UNIT)):
        col = fa[:, j] * scale
        jmin, jmax = int(np.argmin(col)), int(np.argmax(col))
        stab[name] = {"min": float(col[jmin]), "min_holdout": names[jmin],
                      "max": float(col[jmax]), "max_holdout": names[jmax],
                      "in_sample": float(alpha[j] * scale), "unit": unit,
                      "n_folds_zero": int(np.sum(col == 0))}
        print(f"    {name:8s} in-sample {alpha[j]*scale:9.2f}  min {col[jmin]:9.2f} "
              f"({names[jmin]})  max {col[jmax]:9.2f} ({names[jmax]})  {unit}"
              f"{'  [zero in %d fold(s)]' % stab[name]['n_folds_zero'] if stab[name]['n_folds_zero'] else ''}")
    out["loao"]["coefficient_stability"] = stab

    # ---- 3. LOCO ---------------------------------------------------------
    pred_loco = loco(X, y)
    m_loco = metrics(y, pred_loco)
    hh_loco = head_to_head(valid, pred_loco)
    out["loco"] = {"metrics_all": m_loco, "head_to_head": hh_loco}
    print_metrics("LOCO out-of-sample (leave one configuration out)", m_loco)
    print(f"\n  Head-to-head (LOCO out-of-sample): {hh_loco['correct']}/{hh_loco['total']} "
          f"= {hh_loco['accuracy']*100:.0f}%")

    # ---- 3b. Leave-one-category-out (strongest generalization test) ------
    cats = sorted(set(p.category for p in valid))
    pred_locat = np.full(len(valid), np.nan)
    locat = {}
    print("\n  Leave-one-category-out (fit on 7 categories, predict the 8th):")
    for cat in cats:
        test = np.array([p.category == cat for p in valid])
        a_cat = fit_nnls(X[~test], y[~test])
        pred_locat[test] = X[test] @ a_cat
        mcat = metrics(y[test], pred_locat[test])
        locat[cat] = {"metrics": mcat, "alpha": alpha_dict(a_cat)}
        print(f"    {cat:14s} n={int(test.sum()):<3d} MdAPE {mcat['mdape_pct']:7.1f}  "
              f"within2x {mcat['frac_within_2x']*100:5.0f}%  max {mcat['max_ape_pct']:8.1f}%")
    m_locat = metrics(y, pred_locat)
    hh_locat = head_to_head(valid, pred_locat)
    out["locato"] = {"metrics_all": m_locat, "per_category": locat,
                     "head_to_head": hh_locat}
    print_metrics("Leave-one-category-out, all points pooled", m_locat)
    print(f"\n  Head-to-head (leave-one-category-out): {hh_locat['correct']}/{hh_locat['total']} "
          f"= {hh_locat['accuracy']*100:.0f}%")

    # ---- per-algorithm and per-category tables ---------------------------
    per_alg = []
    print("\n  Per-algorithm APE (%):  in-sample MdAPE | LOAO MdAPE | LOAO max | LOCO MdAPE | LOAO within 2x")
    print("  " + "-" * 96)
    algs = sorted(set(p.algorithm for p in valid),
                  key=lambda a: (next(p.category for p in valid if p.algorithm == a), a))
    for a in algs:
        mask = np.array([p.algorithm == a for p in valid])
        cat = next(p.category for p in valid if p.algorithm == a)
        reg = sorted(set(regime_of(p) for p in valid if p.algorithm == a))
        ape_in = np.abs(y[mask] - pred_in[mask]) / y[mask] * 100
        ape_lo = np.abs(y[mask] - pred_loao[mask]) / y[mask] * 100
        ape_lc = np.abs(y[mask] - pred_loco[mask]) / y[mask] * 100
        with np.errstate(divide="ignore", invalid="ignore"):
            r_lo = np.where(pred_loao[mask] > 0,
                            np.maximum(pred_loao[mask] / y[mask], y[mask] / np.where(pred_loao[mask] > 0, pred_loao[mask], 1.0)),
                            np.inf)
        row = {"gpu": label, "algorithm": a, "category": cat, "regime": "/".join(reg),
               "n": int(mask.sum()),
               "mdape_in_sample": float(np.median(ape_in)),
               "mdape_loao": float(np.median(ape_lo)),
               "max_ape_loao": float(np.max(ape_lo)),
               "mdape_loco": float(np.median(ape_lc)),
               "frac_within_2x_loao": float(np.mean(r_lo <= 2.0))}
        per_alg.append(row)
        print(f"  {a:22s} {cat:14s} n={row['n']:<3d} {row['mdape_in_sample']:8.1f} | "
              f"{row['mdape_loao']:8.1f} | {row['max_ape_loao']:8.1f} | "
              f"{row['mdape_loco']:8.1f} | {row['frac_within_2x_loao']*100:5.0f}%")
    out["per_algorithm"] = per_alg

    per_cat = {}
    print("\n  Per-category LOAO MdAPE (%) and fraction within 2x:")
    for cat in sorted(set(p.category for p in valid)):
        mask = np.array([p.category == cat for p in valid])
        mcat = metrics(y[mask], pred_loao[mask])
        per_cat[cat] = mcat
        print(f"    {cat:14s} n={int(mask.sum()):<3d} MdAPE {mcat['mdape_pct']:7.1f}  "
              f"within2x {mcat['frac_within_2x']*100:5.0f}%  r2_log {mcat['r2_log10']:6.3f}")
    out["loao"]["per_category"] = per_cat

    return out, {"alpha": alpha, "X": X, "y": y, "valid": valid}


def print_metrics(title, m):
    print(f"\n  {title}:")
    print(f"    n={m['n']}  r2_linear={m['r2_linear']:.4f}  r2_log10={m['r2_log10']:.4f}"
          f"  geo-mean mult. error={m['geo_mean_mult_error']:.3f}x")
    print(f"    MAPE={m['mape_pct']:.1f}%  MdAPE={m['mdape_pct']:.1f}%  "
          f"p90={m['p90_ape_pct']:.1f}%  max={m['max_ape_pct']:.1f}%")
    print(f"    within 1.5x: {m['frac_within_1p5x']*100:.1f}%   within 2x: "
          f"{m['frac_within_2x']*100:.1f}%   within 3x: {m['frac_within_3x']*100:.1f}%"
          + (f"   [{m['n_nonpositive_pred']} non-positive predictions]"
             if m['n_nonpositive_pred'] else ""))


# =========================================================================
# CROSS-GPU TRANSFER
# =========================================================================

def cross_gpu(fits):
    """fits: dict label -> {alpha, X, y, valid}."""
    labels = list(fits)
    results = {}
    print("\n" + "=" * 78)
    print("  CROSS-GPU COEFFICIENT TRANSFER")
    print("=" * 78)
    for src, tgt in itertools.permutations(labels, 2):
        a_src = fits[src]["alpha"]
        X_t, y_t, valid_t = fits[tgt]["X"], fits[tgt]["y"], fits[tgt]["valid"]
        key = f"{src} -> {tgt}"
        res = {}
        # full transfer
        p_full = X_t @ a_src
        m_full = metrics(y_t, p_full)
        hh_full = head_to_head(valid_t, p_full)
        res["full_transfer"] = {"metrics": m_full, "head_to_head": hh_full}
        # scalar recalibration of the transferred prediction
        k = float(np.dot(p_full, y_t) / np.dot(p_full, p_full))
        m_k = metrics(y_t, k * p_full)
        res["scalar_recalibration"] = {"k": k, "metrics": m_k}
        # partial transfers: every subset of coefficients kept from source
        partial = {}
        for r in range(0, 5):
            for keep in itertools.combinations(range(4), r):
                a = partial_transfer(X_t, y_t, a_src, keep)
                m = metrics(y_t, X_t @ a)
                hh = head_to_head(valid_t, X_t @ a)
                name = ",".join(COEF_SHORT[j] for j in keep) if keep else "(none: in-sample refit)"
                partial[name] = {"kept_from_source": [COEF[j] for j in keep],
                                 "alpha": alpha_dict(a), "metrics": m,
                                 "head_to_head_accuracy": hh["accuracy"],
                                 "head_to_head": f"{hh['correct']}/{hh['total']}"}
        res["partial_transfer"] = partial
        results[key] = res

        print(f"\n  {key}")
        print(f"    full transfer (all 4 coefficients from {src}):")
        print(f"      MdAPE {m_full['mdape_pct']:.1f}%  within2x {m_full['frac_within_2x']*100:.0f}%  "
              f"r2_log {m_full['r2_log10']:.3f}  head-to-head {hh_full['correct']}/{hh_full['total']}")
        print(f"    scalar recalibration (one fitted scale k={k:.3f} on top of full transfer):")
        print(f"      MdAPE {m_k['mdape_pct']:.1f}%  within2x {m_k['frac_within_2x']*100:.0f}%  "
              f"r2_log {m_k['r2_log10']:.3f}")
        print(f"    partial transfer (kept from source | refit on target), sorted by MdAPE:")
        print(f"      {'kept':26s} {'MdAPE%':>7s} {'within2x':>9s} {'r2_log':>7s} {'H2H':>6s}   refit coefficients")
        order = sorted(partial.items(), key=lambda kv: kv[1]["metrics"]["mdape_pct"])
        for name, d in order:
            refit = [c for c in COEF if c not in d["kept_from_source"]]
            ad = d["alpha"]
            refit_str = "  ".join(f"{c}={ad[c + '_' + u.replace('/', '_per_')]:.1f}"
                                  for c, u in zip(COEF, COEF_UNIT) if c in refit)
            print(f"      {name:26s} {d['metrics']['mdape_pct']:7.1f} "
                  f"{d['metrics']['frac_within_2x']*100:8.0f}% {d['metrics']['r2_log10']:7.3f} "
                  f"{d['head_to_head']:>6s}   {refit_str}")
    return results


# =========================================================================
# MAIN
# =========================================================================

def environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__}
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover
        info["scipy"] = "unknown"
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(BASE),
                             capture_output=True, text=True, timeout=10)
        info["git_commit"] = sha.stdout.strip() if sha.returncode == 0 else "unavailable"
        st = subprocess.run(["git", "status", "--porcelain"], cwd=str(BASE),
                            capture_output=True, text=True, timeout=10)
        info["git_dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except Exception:  # pragma: no cover
        info["git_commit"] = "unavailable"
        info["git_dirty"] = None
    return info


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("TOMLSignals EXP-CR-001: error distribution, cross-validation, cross-GPU transfer")
    print(f"started {started}")

    summary = {"experiment": "EXP-CR-001", "started_utc": started,
               "environment": environment_info(), "input_hashes": {},
               "pairs": [list(p) for p in PAIRS], "gpus": {},
               "metric_definitions": {
                   "r2_linear": "1 - SS_res/SS_tot on raw energies in J "
                                "(definition used by v0 fit_four_parameter and the submitted paper)",
                   "r2_log10": "1 - SS_res/SS_tot on log10(energy)",
                   "mape_pct/mdape_pct/p90_ape_pct/max_ape_pct": "mean / median / 90th pct / max of "
                                "|E_pred - E_meas| / E_meas * 100",
                   "frac_within_kx": "fraction of points with max(E_pred/E_meas, E_meas/E_pred) <= k",
                   "geo_mean_mult_error": "10 ** mean(|log10(E_pred/E_meas)|)",
                   "loao": "leave-one-algorithm-out: fit on all other algorithms, predict all "
                           "configurations of the held-out algorithm",
                   "loco": "leave-one-configuration-out: fit on n-1 points, predict the held-out point",
                   "locato": "leave-one-category-out: fit on 7 categories, predict the 8th",
                   "coefficient_stability": "range of each fitted coefficient across the LOAO folds",
                   "cross_gpu_transfer": "coefficients fitted on the source GPU applied to the target "
                                         "GPU; partial_transfer keeps the listed coefficients from the "
                                         "source and refits the rest on the target by NNLS on the residual",
               }}
    fits = {}
    per_alg_rows = []
    for label, csv_rel, gpu_name, has_ta, iir_dir in GPUS:
        try:
            points, valid, hashes = load_gpu(csv_rel, gpu_name, has_ta, iir_dir)
        except FileNotFoundError as e:
            print(f"\n  WARNING: missing data for {label}: {e}")
            continue
        summary["input_hashes"].update(hashes)
        print(f"\nLoaded {len(points)} points for {label} ({len(valid)} valid, "
              f"torchaudio={'yes' if has_ta else 'no'})")
        X, y = design(valid)
        out, fit = analyze_gpu(label, valid, X, y)
        summary["gpus"][label] = out
        if fit is not None:
            fits[label] = fit
            per_alg_rows.extend(out["per_algorithm"])

    if len(fits) == 2:
        summary["cross_gpu_transfer"] = cross_gpu(fits)

    # ---- write outputs ---------------------------------------------------
    json_path = OUT_DIR / "exp_cr_001_cv_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    csv_path = OUT_DIR / "exp_cr_001_per_algorithm.csv"
    if per_alg_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_alg_rows[0].keys()))
            w.writeheader()
            w.writerows(per_alg_rows)
    print("\n" + "=" * 78)
    print(f"  Saved: {json_path.relative_to(BASE).as_posix()}")
    print(f"  Saved: {csv_path.relative_to(BASE).as_posix()}")
    print("  Input SHA-256:")
    for k, v in summary["input_hashes"].items():
        print(f"    {v[:16]}...  {k}")
    print(f"  git: {summary['environment'].get('git_commit')}  "
          f"dirty={summary['environment'].get('git_dirty')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
