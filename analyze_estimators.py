"""
TOMLSignals - Camera-Ready Analysis (EXP-CR-004)
=================================================
Estimator comparison and launch-term / utilization diagnostics for the
four-parameter model  E = a_c*TO_c + a_m*TO_m + a_o*S_o + a_f*S_f.

Reuses analyze_cv.py (EXP-CR-001) for data loading, design matrix, metrics,
head-to-head scoring and fold logic, so every number here is on the same 138 /
126 points. Nothing in analyze_results.py (v0) or analyze_cv.py is modified.

Part 1  Estimators, all run through identical folds:
          ls       v0 unweighted NNLS on raw joules (must reproduce EXP-CR-001)
          rel      relative-error NNLS: min sum ((E_pred - E)/E)^2, alpha >= 0
                   (same linear model, same physical coefficients; every
                   configuration weighted equally instead of by its joules)
          rel_alg  relative-error NNLS with each ALGORITHM weighted equally
                   regardless of its number of configurations (robustness check)
        For each: in-sample, LOAO, LOCO, leave-one-category-out, head-to-head,
        coefficient values and LOAO ranges, per-algorithm MdAPE, cross-GPU
        full transfer and two partial transfers.
Part 2  Launch-term diagnostic (RTX 4090, B = 1, the 23 algorithms in the v0
        NCU summary): residual energy per kernel launch after a_c*T_c + a_m*T_m,
        energy per NCU FP32 instruction, and a trial NNLS with a per-launch term
        on those 23 points.
Part 3  Utilization diagnostic (both GPUs, parallel regime): effective energy
        per compute TO, E_meas / T_c, binned by decade of T_c.

Outputs:  data/camera_ready/exp_cr_004_estimators.json
Console is ASCII-only.   Usage (repo root):  python analyze_estimators.py

Author: Muntaser Syed
Date: August 2026
"""

import itertools
import json
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

import analyze_cv as cv  # noqa: E402  (EXP-CR-001, unmodified)
from scipy.optimize import nnls  # noqa: E402

OUT_DIR = BASE / "data" / "camera_ready"
NCU_SUMMARY = BASE / "data" / "ncu_profiles" / "ncu_summary.json"

ESTIMATORS = ["ls", "rel", "rel_alg"]


# =========================================================================
# ESTIMATORS
# =========================================================================

def fit(X, y, estimator, alg=None):
    """Return alpha (4,) for the given estimator.
    ls      : nnls(X, y)                          (v0)
    rel     : nnls(X / y, 1)                      (relative-error, equal weight per point)
    rel_alg : nnls(X * w, y * w), w = 1/(y*sqrt(n_alg))  (equal weight per algorithm)
    """
    if estimator == "ls":
        a, _ = nnls(X, y)
        return a
    if estimator == "rel":
        w = 1.0 / y
    elif estimator == "rel_alg":
        alg = np.asarray(alg)
        counts = {a_: int(np.sum(alg == a_)) for a_ in set(alg.tolist())}
        w = 1.0 / (y * np.sqrt(np.array([counts[a_] for a_ in alg.tolist()])))
    else:
        raise ValueError(estimator)
    a, _ = nnls(X * w[:, None], y * w)
    return a


def loao(valid, X, y, estimator):
    algs_all = np.array([p.algorithm for p in valid])
    pred = np.full(len(valid), np.nan)
    fold_alpha = {}
    for a in sorted(set(algs_all.tolist())):
        test = algs_all == a
        alpha = fit(X[~test], y[~test], estimator, algs_all[~test])
        pred[test] = X[test] @ alpha
        fold_alpha[a] = alpha
    return pred, fold_alpha


def loco(valid, X, y, estimator):
    algs_all = np.array([p.algorithm for p in valid])
    n = len(y)
    pred = np.empty(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        alpha = fit(X[mask], y[mask], estimator, algs_all[mask])
        pred[i] = X[i] @ alpha
        mask[i] = True
    return pred


def locato(valid, X, y, estimator):
    algs_all = np.array([p.algorithm for p in valid])
    cats = np.array([p.category for p in valid])
    pred = np.full(len(valid), np.nan)
    for c in sorted(set(cats.tolist())):
        test = cats == c
        alpha = fit(X[~test], y[~test], estimator, algs_all[~test])
        pred[test] = X[test] @ alpha
    return pred


def partial_transfer(X_t, y_t, alpha_src, keep, estimator, alg_t):
    keep = list(keep)
    refit = [j for j in range(4) if j not in keep]
    alpha = np.zeros(4)
    if keep:
        alpha[keep] = alpha_src[keep]
        resid = y_t - X_t[:, keep] @ alpha_src[keep]
    else:
        resid = y_t.copy()
    if refit:
        if estimator == "ls":
            a_refit, _ = nnls(X_t[:, refit], resid)
        else:
            # relative weighting by the TARGET measured energy (not the residual)
            if estimator == "rel":
                w = 1.0 / y_t
            else:
                alg_t = np.asarray(alg_t)
                counts = {a_: int(np.sum(alg_t == a_)) for a_ in set(alg_t.tolist())}
                w = 1.0 / (y_t * np.sqrt(np.array([counts[a_] for a_ in alg_t.tolist()])))
            a_refit, _ = nnls(X_t[:, refit] * w[:, None], resid * w)
        alpha[refit] = a_refit
    return alpha


# =========================================================================
# PART 1: ESTIMATOR COMPARISON
# =========================================================================

def part1_gpu(label, valid, X, y):
    algs_all = np.array([p.algorithm for p in valid])
    out = {}
    print("\n" + "=" * 78)
    print(f"  PART 1  {label}: {len(valid)} points, estimators {ESTIMATORS}")
    print("=" * 78)

    preds = {}   # estimator -> dict(scheme -> pred array)
    for est in ESTIMATORS:
        alpha = fit(X, y, est, algs_all)
        p_in = X @ alpha
        p_loao, fold_alpha = loao(valid, X, y, est)
        p_loco = loco(valid, X, y, est)
        p_locat = locato(valid, X, y, est)
        preds[est] = {"in": p_in, "loao": p_loao, "loco": p_loco, "locato": p_locat}
        fa = np.array([fold_alpha[a] for a in sorted(fold_alpha)])
        names = sorted(fold_alpha)
        stab = {}
        for j, (name, scale, unit) in enumerate(zip(cv.COEF, cv.COEF_SCALE, cv.COEF_UNIT)):
            col = fa[:, j] * scale
            jmin, jmax = int(np.argmin(col)), int(np.argmax(col))
            stab[name] = {"in_sample": float(alpha[j] * scale), "min": float(col[jmin]),
                          "min_holdout": names[jmin], "max": float(col[jmax]),
                          "max_holdout": names[jmax], "unit": unit}
        res = {"alpha": cv.alpha_dict(alpha), "coefficient_stability": stab}
        for scheme, p in preds[est].items():
            m = cv.metrics(y, p)
            hh = cv.head_to_head(valid, p)
            res[scheme] = {"metrics": m, "head_to_head": f"{hh['correct']}/{hh['total']}",
                           "hh_pairs": hh["pairs"] if scheme in ("in", "loao") else None}
            by_reg = {}
            for reg in ["parallel", "python-loop", "fused-sequential"]:
                mask = np.array([cv.regime_of(q) == reg for q in valid])
                if mask.sum() >= 2:
                    by_reg[reg] = cv.metrics(y[mask], p[mask])
            res[scheme]["by_regime"] = by_reg
        res["per_algorithm"] = {}
        for a in sorted(set(algs_all.tolist())):
            mask = algs_all == a
            res["per_algorithm"][a] = {
                "mdape_in": float(np.median(np.abs(y[mask] - preds[est]["in"][mask]) / y[mask] * 100)),
                "mdape_loao": float(np.median(np.abs(y[mask] - preds[est]["loao"][mask]) / y[mask] * 100)),
                "signed_median_ratio_in": float(np.median(preds[est]["in"][mask] / y[mask])),
            }
        out[est] = res

    # ---- printing --------------------------------------------------------
    print("\n  Coefficients (in-sample) and LOAO range:")
    print(f"    {'coef':8s} " + " ".join(f"{est:>28s}" for est in ESTIMATORS))
    for name in cv.COEF:
        cells = []
        for est in ESTIMATORS:
            s = out[est]["coefficient_stability"][name]
            cells.append(f"{s['in_sample']:8.1f} [{s['min']:7.1f},{s['max']:7.1f}]")
        print(f"    {name:8s} " + " ".join(f"{c:>28s}" for c in cells)
              + f"   {out['ls']['coefficient_stability'][name]['unit']}")
    for est in ESTIMATORS:
        s = out[est]["coefficient_stability"]
        print(f"    {est:8s} extremes driven by: alpha_c min {s['alpha_c']['min_holdout']}, "
              f"alpha_m max {s['alpha_m']['max_holdout']}, alpha_o min {s['alpha_o']['min_holdout']}")

    print("\n  Metrics (rows) x estimator/scheme (columns):")
    schemes = ["in", "loao", "loco", "locato"]
    cols = [(est, sc) for est in ESTIMATORS for sc in schemes]
    print("    " + " " * 22 + " ".join(f"{est}/{sc:>6s}" for est, sc in cols))
    rows = [("r2_linear", "r2_linear", "{:9.3f}"), ("r2_log10", "r2_log10", "{:9.3f}"),
            ("MdAPE %", "mdape_pct", "{:9.1f}"), ("MAPE %", "mape_pct", "{:9.1f}"),
            ("p90 APE %", "p90_ape_pct", "{:9.1f}"), ("max APE %", "max_ape_pct", "{:9.1f}"),
            ("within 1.5x", "frac_within_1p5x", "{:9.3f}"), ("within 2x", "frac_within_2x", "{:9.3f}"),
            ("within 3x", "frac_within_3x", "{:9.3f}"), ("geo mult err", "geo_mean_mult_error", "{:9.3f}")]
    for title, key, fmt in rows:
        print(f"    {title:22s} " + " ".join(fmt.format(out[est][sc]["metrics"][key]).rjust(10)
                                             for est, sc in cols))
    print(f"    {'head-to-head':22s} " + " ".join(out[est][sc]["head_to_head"].rjust(10)
                                                for est, sc in cols))

    print("\n  By regime, in-sample MdAPE % / within 2x:")
    for reg in ["parallel", "python-loop", "fused-sequential"]:
        cells = []
        for est in ESTIMATORS:
            m = out[est]["in"]["by_regime"].get(reg)
            cells.append(f"{m['mdape_pct']:6.1f} / {m['frac_within_2x']*100:4.0f}%" if m else "   n/a")
        print(f"    {reg:18s} " + "   ".join(f"{est}: {c}" for est, c in zip(ESTIMATORS, cells)))

    print("\n  Head-to-head failures, in-sample (LS vs REL):")
    for est in ["ls", "rel"]:
        fails = [f"{r['pair']} N={r['N']}" for r in out[est]["in"]["hh_pairs"] if not r["correct"]]
        print(f"    {est:4s} {out[est]['in']['head_to_head']}: " + "; ".join(fails))

    print("\n  Per-algorithm MdAPE % (in-sample | LOAO)  and signed median ratio pred/meas (in-sample):")
    print(f"    {'algorithm':22s} {'ls in':>7s} {'ls LOAO':>8s} {'rel in':>7s} {'rel LOAO':>9s} "
          f"{'ls ratio':>9s} {'rel ratio':>10s}")
    for a in sorted(set(algs_all.tolist()),
                    key=lambda q: (next(p.category for p in valid if p.algorithm == q), q)):
        L, R = out["ls"]["per_algorithm"][a], out["rel"]["per_algorithm"][a]
        print(f"    {a:22s} {L['mdape_in']:7.1f} {L['mdape_loao']:8.1f} {R['mdape_in']:7.1f} "
              f"{R['mdape_loao']:9.1f} {L['signed_median_ratio_in']:9.2f} {R['signed_median_ratio_in']:10.2f}")
    return out, {est: fit(X, y, est, algs_all) for est in ESTIMATORS}


def part1_cross_gpu(fits, data):
    """fits: label -> {est: alpha}; data: label -> (valid, X, y)."""
    print("\n" + "=" * 78)
    print("  PART 1b  CROSS-GPU TRANSFER BY ESTIMATOR")
    print("=" * 78)
    out = {}
    labels = list(fits)
    for src, tgt in itertools.permutations(labels, 2):
        valid_t, X_t, y_t = data[tgt]
        alg_t = np.array([p.algorithm for p in valid_t])
        key = f"{src} -> {tgt}"
        out[key] = {}
        print(f"\n  {key}")
        print(f"    {'estimator':9s} {'variant':32s} {'MdAPE%':>7s} {'within2x':>9s} {'r2_log':>7s} {'H2H':>6s}")
        for est in ESTIMATORS:
            a_src = fits[src][est]
            variants = [("full transfer (all 4)", (0, 1, 2, 3)),
                        ("keep c,m; refit o,f on target", (0, 1)),
                        ("keep c,f; refit m,o on target", (0, 3)),
                        ("target in-sample (refit all)", ())]
            out[key][est] = {}
            for name, keep in variants:
                a = partial_transfer(X_t, y_t, a_src, keep, est, alg_t)
                p = X_t @ a
                m = cv.metrics(y_t, p)
                hh = cv.head_to_head(valid_t, p)
                out[key][est][name] = {"alpha": cv.alpha_dict(a), "metrics": m,
                                       "head_to_head": f"{hh['correct']}/{hh['total']}"}
                print(f"    {est:9s} {name:32s} {m['mdape_pct']:7.1f} {m['frac_within_2x']*100:8.0f}% "
                      f"{m['r2_log10']:7.3f} {hh['correct']:>3d}/{hh['total']}")
    return out


# =========================================================================
# PART 2: LAUNCH-TERM DIAGNOSTIC (4090, B = 1, NCU-profiled algorithms)
# =========================================================================

def part2_launch_diagnostic(valid, X, y, alpha_ls, alpha_rel):
    print("\n" + "=" * 78)
    print("  PART 2  LAUNCH-TERM DIAGNOSTIC (RTX 4090, B = 1, v0 NCU summary)")
    print("=" * 78)
    if not NCU_SUMMARY.exists():
        print(f"  ncu_summary.json not found at {NCU_SUMMARY}; skipping")
        return None
    with open(NCU_SUMMARY) as f:
        ncu = json.load(f)
    by_key = {(p.algorithm, p.signal_length, p.batch_size): (i, p) for i, p in enumerate(valid)}
    rows = []
    for r in ncu:
        key = (r["algorithm"], int(r["N"]), int(r["B"]))
        if key not in by_key:
            print(f"    no benchmark point for {key}; skipped")
            continue
        i, p = by_key[key]
        e = y[i]
        e_cm_ls = alpha_ls[0] * p.to_compute + alpha_ls[1] * p.to_memory
        e_pred_ls = X[i] @ alpha_ls
        e_pred_rel = X[i] @ alpha_rel
        nk = int(r["n_kernels"])
        fp32 = float(r["fp32_total"])
        rows.append({
            "algorithm": p.algorithm, "N": p.signal_length, "regime": cv.regime_of(p),
            "n_kernels": nk, "fp32_instr": fp32,
            "E_meas_uJ": e * 1e6, "E_pred_ls_uJ": e_pred_ls * 1e6, "E_pred_rel_uJ": e_pred_rel * 1e6,
            "E_cm_ls_uJ": e_cm_ls * 1e6,
            "ratio_meas_pred_ls": e / e_pred_ls if e_pred_ls > 0 else float("inf"),
            "resid_per_kernel_uJ": (e - e_cm_ls) / nk * 1e6,
            "E_per_kernel_uJ": e / nk * 1e6,
            "pJ_per_fp32_instr": (e / fp32 * 1e12) if fp32 > 0 else float("nan"),
            "eff_fJ_per_TOc": e / p.to_compute * 1e15 if p.to_compute > 0 else float("nan"),
        })
    rows.sort(key=lambda d: d["resid_per_kernel_uJ"])
    print("\n  alpha_c(ls) = %.2f fJ/TO -> %.1f pJ per counted MAC (5000 TO)" % (alpha_ls[0] * 1e15, alpha_ls[0] * 5000 * 1e12))
    print(f"    {'algorithm':22s} {'N':>6s} {'kern':>5s} {'E_meas uJ':>10s} {'E_pred uJ':>10s} "
          f"{'meas/pred':>9s} {'resid/kern uJ':>13s} {'E/kern uJ':>10s} {'pJ/FP32':>8s} {'eff fJ/TOc':>10s} regime")
    for d in rows:
        print(f"    {d['algorithm']:22s} {d['N']:6d} {d['n_kernels']:5d} {d['E_meas_uJ']:10.1f} "
              f"{d['E_pred_ls_uJ']:10.1f} {d['ratio_meas_pred_ls']:9.2f} {d['resid_per_kernel_uJ']:13.1f} "
              f"{d['E_per_kernel_uJ']:10.1f} {d['pJ_per_fp32_instr']:8.1f} {d['eff_fJ_per_TOc']:10.1f} {d['regime']}")

    # trial fit on these points: E = a_c T_c + a_m T_m + a_o S_o + a_f S_f + a_k n_kernels
    idx = [by_key[(d["algorithm"], d["N"], 1)][0] for d in rows]
    Xk = np.column_stack([X[idx], np.array([d["n_kernels"] for d in rows], dtype=float)])
    yk = y[idx]
    a4, _ = nnls(X[idx], yk)
    a5, _ = nnls(Xk, yk)
    w = 1.0 / yk
    a4r, _ = nnls(X[idx] * w[:, None], yk * w)
    a5r, _ = nnls(Xk * w[:, None], yk * w)
    trial = {}
    print("\n  Trial fits on these B=1 points only (n=%d):" % len(rows))
    for name, a, Xm in [("4-param ls", a4, X[idx]), ("5-param ls (+launch)", a5, Xk),
                        ("4-param rel", a4r, X[idx]), ("5-param rel (+launch)", a5r, Xk)]:
        m = cv.metrics(yk, Xm @ a)
        ak = a[4] * 1e6 if len(a) > 4 else float("nan")
        trial[name] = {"alpha": [float(v) for v in a], "alpha_k_uJ_per_kernel": ak, "metrics": m}
        print(f"    {name:24s} MdAPE {m['mdape_pct']:6.1f}%  within2x {m['frac_within_2x']*100:4.0f}%  "
              f"r2_log {m['r2_log10']:6.3f}  alpha_c {a[0]*1e15:6.1f} fJ/TO  alpha_m {a[1]*1e15:6.1f}  "
              f"alpha_k {ak:7.1f} uJ/kernel")
    # correlation between residual and kernel count among parallel-regime rows
    par = [d for d in rows if d["regime"] == "parallel"]
    if len(par) >= 4:
        r_res = np.array([d["E_meas_uJ"] - d["E_cm_ls_uJ"] for d in par])
        r_k = np.array([d["n_kernels"] for d in par], dtype=float)
        rho = spearman(r_res, r_k)
        print(f"\n  Spearman(residual after c+m, n_kernels) over {len(par)} parallel-regime rows: {rho:.3f}")
    else:
        rho = float("nan")
    return {"rows": rows, "trial_fits": trial, "spearman_resid_vs_kernels_parallel": rho}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# =========================================================================
# PART 3: UTILIZATION DIAGNOSTIC (effective fJ per compute TO by decade of T_c)
# =========================================================================

def part3_utilization(label, valid, X, y, alpha_ls, alpha_rel):
    print("\n" + "=" * 78)
    print(f"  PART 3  EFFECTIVE ENERGY PER COMPUTE TO, {label} (parallel regime)")
    print("=" * 78)
    par = [(p, X[i], y[i]) for i, p in enumerate(valid) if cv.regime_of(p) == "parallel"]
    tc = np.array([xx[0] for _, xx, _ in par])
    e = np.array([ee for _, _, ee in par])
    eff = e / tc * 1e15
    dec = np.floor(np.log10(tc)).astype(int)
    out = {"alpha_c_ls_fJ": alpha_ls[0] * 1e15, "alpha_c_rel_fJ": alpha_rel[0] * 1e15, "bins": []}
    print(f"    alpha_c: ls {alpha_ls[0]*1e15:.1f} fJ/TO, rel {alpha_rel[0]*1e15:.1f} fJ/TO")
    print(f"    {'T_c decade':12s} {'n':>3s} {'median E/T_c fJ/TO':>19s} {'min':>9s} {'max':>10s}   algorithms (min .. max)")
    for d in sorted(set(dec.tolist())):
        m = dec == d
        vals = eff[m]
        names = [q.algorithm for (q, _, _), mm in zip(par, m) if mm]
        imin, imax = int(np.argmin(vals)), int(np.argmax(vals))
        out["bins"].append({"decade": int(d), "n": int(m.sum()), "median_eff_fJ_per_TO": float(np.median(vals)),
                            "min": float(vals.min()), "min_alg": names[imin],
                            "max": float(vals.max()), "max_alg": names[imax]})
        print(f"    1e{d:<10d} {int(m.sum()):3d} {np.median(vals):19.1f} {vals.min():9.1f} {vals.max():10.1f}   "
              f"{names[imin]} .. {names[imax]}")
    # per algorithm effective fJ/TO (median over its parallel configs), sorted
    per_alg = {}
    for q, xx, ee in par:
        per_alg.setdefault(q.algorithm, []).append(ee / xx[0] * 1e15)
    ranked = sorted(per_alg.items(), key=lambda kv: np.median(kv[1]))
    out["per_algorithm_median_eff_fJ_per_TO"] = {k: float(np.median(v)) for k, v in ranked}
    print("\n    Per-algorithm median E/T_c (fJ/TO), ascending:")
    line = []
    for k, v in ranked:
        line.append(f"{k}={np.median(v):.0f}")
    for i in range(0, len(line), 6):
        print("      " + "  ".join(line[i:i + 6]))
    return out


# =========================================================================
# MAIN
# =========================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("TOMLSignals EXP-CR-004: estimator comparison, launch-term and utilization diagnostics")
    print(f"started {started}")
    summary = {"experiment": "EXP-CR-004", "started_utc": started,
               "environment": cv.environment_info(), "input_hashes": {},
               "estimators": {"ls": "nnls(X, y) (v0)",
                              "rel": "nnls(X/y, 1): minimize sum ((E_pred-E)/E)^2, alpha>=0",
                              "rel_alg": "as rel with weights 1/(E*sqrt(n_configs_of_algorithm))"},
               "gpus": {}, "part2_launch_diagnostic_4090": None, "part3_utilization": {}}
    data, fits = {}, {}
    for label, csv_rel, gpu_name, has_ta, iir_dir in cv.GPUS:
        try:
            points, valid, hashes = cv.load_gpu(csv_rel, gpu_name, has_ta, iir_dir)
        except FileNotFoundError as e:
            print(f"\n  WARNING: missing data for {label}: {e}")
            continue
        summary["input_hashes"].update(hashes)
        X, y = cv.design(valid)
        data[label] = (valid, X, y)
        out, est_fits = part1_gpu(label, valid, X, y)
        # control: ls must reproduce EXP-CR-001
        m_ls = out["ls"]["in"]["metrics"]
        print(f"\n  [control] ls in-sample MdAPE {m_ls['mdape_pct']:.1f}%, r2_linear {m_ls['r2_linear']:.4f}, "
              f"H2H {out['ls']['in']['head_to_head']}  (EXP-CR-001: 44.3 / 0.9467 / 24/30 on 4090; "
              f"68.0 / 0.9823 / 24/30 on A100)")
        summary["gpus"][label] = out
        fits[label] = est_fits
    if len(fits) == 2:
        summary["cross_gpu_transfer"] = part1_cross_gpu(fits, data)
    if "RTX 4090" in data:
        valid, X, y = data["RTX 4090"]
        summary["part2_launch_diagnostic_4090"] = part2_launch_diagnostic(
            valid, X, y, fits["RTX 4090"]["ls"], fits["RTX 4090"]["rel"])
        if NCU_SUMMARY.exists():
            summary["input_hashes"][NCU_SUMMARY.relative_to(BASE).as_posix()] = cv.sha256_file(NCU_SUMMARY)
    for label in data:
        valid, X, y = data[label]
        summary["part3_utilization"][label] = part3_utilization(
            label, valid, X, y, fits[label]["ls"], fits[label]["rel"])

    json_path = OUT_DIR / "exp_cr_004_estimators.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n" + "=" * 78)
    print(f"  Saved: {json_path.relative_to(BASE).as_posix()}")
    for k, v in summary["input_hashes"].items():
        print(f"    {v[:16]}...  {k}")
    print(f"  git: {summary['environment'].get('git_commit')}  dirty={summary['environment'].get('git_dirty')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
