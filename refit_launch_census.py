"""
TOMLSignals - EXP-CR-006: Refit with Measured Launch Counts
============================================================
Refits E = a_c*TO_c + a_m*TO_m + a_o*S_o + a_f*S_f with S_o taken from the
EXP-CR-005 kernel-launch census, under the identical fold machinery of
EXP-CR-001 / EXP-CR-004 (analyze_cv.py, analyze_estimators.py, unmodified).

S_o definitions compared (estimator: v0 unweighted NNLS, F-020 decision):
  v0         model counts (KERNELS_PER_ITER, hand-derived): must reproduce EXP-CR-001
  a-kernels  census kernel count, but only for algorithms where v0 has S_o > 0
             (Python-loop + IIR fallback); S_o = 0 elsewhere, as in the paper
  a-all      as a-kernels but counting kernels + memcpy + memset
  b-kernels  unified: census kernel count for EVERY algorithm (svd, pca,
             esprit, music, transformer, ... included)
  b-all      unified, kernels + memcpy + memset

For every variant and GPU: coefficients (+ LOAO range), full error distribution
in-sample / LOAO / LOCO / leave-one-category-out, head-to-head, by-regime (v0
regime labels), per-algorithm MdAPE and signed ratio, launch-term share of the
predicted energy, and cross-GPU transfer.

Outputs: data/camera_ready/exp_cr_006_refit.json.  Console is ASCII-only.
Usage (repo root):  python refit_launch_census.py

Author: Muntaser Syed
Date: August 2026
"""

import csv
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
import analyze_estimators as est  # noqa: E402  (EXP-CR-004, unmodified; loao/loco/locato/partial_transfer)

OUT_DIR = BASE / "data" / "camera_ready"
CENSUS_CSV = OUT_DIR / "exp_cr_005_kernel_census.csv"
VARIANTS = ["v0", "a-kernels", "a-all", "b-kernels", "b-all"]
FOCUS = ["svd", "pca", "esprit", "music", "transformer_denoiser", "mdct_audio", "iir_butter4",
         "lms", "nlms", "rls", "apa_p4", "kalman", "ekf", "ukf", "particle_1k", "fastica", "nmf",
         "median", "direct_dft", "fft", "cnn_denoiser", "lstm_denoiser", "jpeg_q50", "welch", "wiener"]


# =========================================================================
# CENSUS
# =========================================================================

def load_census():
    """(alg, N, B, variant) -> dict(kernels, memcpy, memset) from repeat 0."""
    if not CENSUS_CSV.exists():
        raise FileNotFoundError(CENSUS_CSV)
    table = {}
    with open(CENSUS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["repeat_idx"]) != 0:
                continue
            key = (row["algorithm"], int(row["N"]), int(row["B"]), row["variant"])
            table[key] = {"kernels": int(float(row["n_kernels"])),
                          "memcpy": int(float(row["n_memcpy"])),
                          "memset": int(float(row["n_memset"]))}
    return table


def census_count(table, alg, N, B, has_torchaudio, mode):
    if alg == "iir_butter4":
        variant = "torchaudio" if has_torchaudio else "fallback"
    else:
        variant = "default"
    row = table.get((alg, N, B, variant))
    if row is None:
        return None
    if mode == "kernels":
        return row["kernels"]
    return row["kernels"] + row["memcpy"] + row["memset"]


def build_S_o(valid, table, has_torchaudio, variant):
    """Return the S_o column for the given variant, and a list of missing keys."""
    S = np.zeros(len(valid))
    missing = []
    for i, p in enumerate(valid):
        v0 = float(p.n_seq_steps)
        if variant == "v0":
            S[i] = v0
            continue
        scheme, mode = variant.split("-")           # a|b, kernels|all
        c = census_count(table, p.algorithm, p.signal_length, p.batch_size, has_torchaudio, mode)
        if c is None:
            missing.append((p.algorithm, p.signal_length, p.batch_size))
            S[i] = v0
            continue
        if scheme == "a":
            S[i] = c if v0 > 0 else 0.0
        else:
            S[i] = c
    return S, missing


# =========================================================================
# ANALYSIS
# =========================================================================

def analyze_variant(label, valid, X0, y, S, variant):
    X = X0.copy()
    X[:, 2] = S
    algs = np.array([p.algorithm for p in valid])
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
        stab[name] = {"in_sample": float(alpha[j] * scale), "min": float(col[jmin]),
                      "min_holdout": names[jmin], "max": float(col[jmax]),
                      "max_holdout": names[jmax], "unit": unit}
    res = {"alpha": cv.alpha_dict(alpha), "coefficient_stability": stab, "schemes": {}}
    for scheme, p in [("in", p_in), ("loao", p_loao), ("loco", p_loco), ("locato", p_locat)]:
        hh = cv.head_to_head(valid, p)
        by_reg = {}
        for reg in ["parallel", "python-loop", "fused-sequential"]:
            mask = np.array([cv.regime_of(q) == reg for q in valid])   # v0 regime labels
            if mask.sum() >= 2:
                by_reg[reg] = cv.metrics(y[mask], p[mask])
        res["schemes"][scheme] = {"metrics": cv.metrics(y, p),
                                  "head_to_head": f"{hh['correct']}/{hh['total']}",
                                  "hh_pairs": hh["pairs"] if scheme in ("in", "loao") else None,
                                  "by_regime": by_reg}
    # per-algorithm
    per_alg = {}
    for a in sorted(set(algs.tolist())):
        m = algs == a
        per_alg[a] = {"mdape_in": float(np.median(np.abs(y[m] - p_in[m]) / y[m] * 100)),
                      "mdape_loao": float(np.median(np.abs(y[m] - p_loao[m]) / y[m] * 100)),
                      "ratio_in": float(np.median(p_in[m] / y[m])),
                      "ratio_loao": float(np.median(p_loao[m] / y[m])),
                      "n": int(m.sum())}
    res["per_algorithm"] = per_alg
    # launch-term share of predicted energy (in-sample), by v0 regime
    share = alpha[2] * X[:, 2] / np.maximum(p_in, 1e-300)
    res["launch_share_by_regime"] = {}
    for reg in ["parallel", "python-loop", "fused-sequential"]:
        mask = np.array([cv.regime_of(q) == reg for q in valid])
        if mask.sum():
            res["launch_share_by_regime"][reg] = {"median": float(np.median(share[mask])),
                                                  "max": float(np.max(share[mask])),
                                                  "n": int(mask.sum())}
    res["S_o_stats"] = {"n_nonzero": int(np.sum(S > 0)), "max": float(S.max()),
                        "sum": float(S.sum())}
    return res, alpha, X


def print_gpu(label, out, valid):
    print("\n" + "=" * 96)
    print(f"  {label}: {len(valid)} points  (estimator: v0 unweighted NNLS)")
    print("=" * 96)
    print("\n  Coefficients (in-sample [LOAO min, max]):")
    print(f"    {'coef':8s} " + " ".join(f"{v:>26s}" for v in VARIANTS))
    for name in cv.COEF:
        cells = []
        for v in VARIANTS:
            s = out[v]["coefficient_stability"][name]
            cells.append(f"{s['in_sample']:7.1f} [{s['min']:6.1f},{s['max']:6.1f}]")
        print(f"    {name:8s} " + " ".join(f"{c:>26s}" for c in cells) + f"  {out['v0']['coefficient_stability'][name]['unit']}")
    ao = {v: out[v]["alpha"]["alpha_o_uJ_per_launch"] for v in VARIANTS}
    af = {v: out[v]["alpha"]["alpha_f_uJ_per_step"] for v in VARIANTS}
    print("    alpha_o/alpha_f: " + "  ".join(f"{v}={ao[v]/af[v] if af[v] > 0 else float('nan'):.2f}" for v in VARIANTS))
    print("    S_o non-zero points / max / total: " + "  ".join(
        f"{v}={out[v]['S_o_stats']['n_nonzero']}/{out[v]['S_o_stats']['max']:.0f}/{out[v]['S_o_stats']['sum']:.0f}" for v in VARIANTS))

    print("\n  Metrics (rows) x variant/scheme (columns):")
    schemes = ["in", "loao", "loco", "locato"]
    cols = [(v, sc) for v in VARIANTS for sc in schemes]
    print("    " + " " * 14 + " ".join(f"{v[:9]}/{sc:>6s}" for v, sc in cols))
    rows = [("r2_linear", "r2_linear", "{:.3f}"), ("r2_log10", "r2_log10", "{:.3f}"),
            ("MdAPE %", "mdape_pct", "{:.1f}"), ("MAPE %", "mape_pct", "{:.1f}"),
            ("p90 APE %", "p90_ape_pct", "{:.1f}"), ("max APE %", "max_ape_pct", "{:.1f}"),
            ("within 1.5x", "frac_within_1p5x", "{:.3f}"), ("within 2x", "frac_within_2x", "{:.3f}"),
            ("within 3x", "frac_within_3x", "{:.3f}"), ("geo mult err", "geo_mean_mult_error", "{:.3f}")]
    for title, key, fmt in rows:
        print(f"    {title:14s} " + " ".join(fmt.format(out[v]["schemes"][sc]["metrics"][key]).rjust(16) for v, sc in cols))
    print(f"    {'head-to-head':14s} " + " ".join(out[v]["schemes"][sc]["head_to_head"].rjust(16) for v, sc in cols))

    print("\n  By regime (v0 labels), in-sample MdAPE % / within 2x  and launch-term share of prediction (median, max):")
    for reg in ["parallel", "python-loop", "fused-sequential"]:
        cells = []
        for v in VARIANTS:
            m = out[v]["schemes"]["in"]["by_regime"].get(reg)
            sh = out[v]["launch_share_by_regime"].get(reg)
            cells.append(f"{v}: {m['mdape_pct']:5.1f}/{m['frac_within_2x']*100:3.0f}% share {sh['median']*100:4.1f}%/{sh['max']*100:5.1f}%"
                         if m and sh else f"{v}: n/a")
        print(f"    {reg:17s} " + " | ".join(cells))

    print("\n  Head-to-head failures in-sample:")
    for v in VARIANTS:
        fails = [f"{r['pair']} N={r['N']}" for r in out[v]["schemes"]["in"]["hh_pairs"] if not r["correct"]]
        print(f"    {v:10s} {out[v]['schemes']['in']['head_to_head']}: " + "; ".join(fails))

    print("\n  Per-algorithm in-sample MdAPE % (signed median ratio pred/meas) by variant:")
    print(f"    {'algorithm':22s} " + " ".join(f"{v:>16s}" for v in VARIANTS) + "   [LOAO MdAPE b-all]")
    algs_sorted = sorted(out["v0"]["per_algorithm"],
                         key=lambda a: (next(p.category for p in valid if p.algorithm == a), a))
    for a in algs_sorted:
        cells = [f"{out[v]['per_algorithm'][a]['mdape_in']:6.1f} ({out[v]['per_algorithm'][a]['ratio_in']:4.2f})" for v in VARIANTS]
        print(f"    {a:22s} " + " ".join(f"{c:>16s}" for c in cells) + f"   {out['b-all']['per_algorithm'][a]['mdape_loao']:6.1f}")


def cross_gpu(fits, data):
    print("\n" + "=" * 96)
    print("  CROSS-GPU TRANSFER BY VARIANT (LS)")
    print("=" * 96)
    out = {}
    for src, tgt in itertools.permutations(list(fits), 2):
        valid_t, y_t = data[tgt]["valid"], data[tgt]["y"]
        alg_t = np.array([p.algorithm for p in valid_t])
        key = f"{src} -> {tgt}"
        out[key] = {}
        print(f"\n  {key}")
        print(f"    {'variant':10s} {'transfer':30s} {'MdAPE%':>7s} {'within2x':>9s} {'r2_log':>7s} {'H2H':>6s}")
        for v in VARIANTS:
            a_src = fits[src][v]
            X_t = data[tgt]["X"][v]
            out[key][v] = {}
            for name, keep in [("full transfer (all 4)", (0, 1, 2, 3)),
                               ("keep c,m; refit o,f on target", (0, 1)),
                               ("keep c,f; refit m,o on target", (0, 3)),
                               ("target in-sample (refit all)", ())]:
                a = est.partial_transfer(X_t, y_t, a_src, keep, "ls", alg_t)
                p = X_t @ a
                m = cv.metrics(y_t, p)
                hh = cv.head_to_head(valid_t, p)
                out[key][v][name] = {"alpha": cv.alpha_dict(a), "metrics": m,
                                     "head_to_head": f"{hh['correct']}/{hh['total']}"}
                print(f"    {v:10s} {name:30s} {m['mdape_pct']:7.1f} {m['frac_within_2x']*100:8.0f}% "
                      f"{m['r2_log10']:7.3f} {hh['correct']:>3d}/{hh['total']}")
    return out


# =========================================================================
# MAIN
# =========================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("TOMLSignals EXP-CR-006: refit with measured launch counts")
    print(f"started {started}")
    table = load_census()
    print(f"census rows (repeat 0): {len(table)}")
    summary = {"experiment": "EXP-CR-006", "started_utc": started, "environment": cv.environment_info(),
               "input_hashes": {CENSUS_CSV.relative_to(BASE).as_posix(): cv.sha256_file(CENSUS_CSV)},
               "variants": VARIANTS, "gpus": {}}
    data, fits = {}, {}
    for label, csv_rel, gpu_name, has_ta, iir_dir in cv.GPUS:
        try:
            points, valid, hashes = cv.load_gpu(csv_rel, gpu_name, has_ta, iir_dir)
        except FileNotFoundError as e:
            print(f"\n  WARNING: missing data for {label}: {e}")
            continue
        summary["input_hashes"].update(hashes)
        X0, y = cv.design(valid)
        out, fits[label], Xs = {}, {}, {}
        for v in VARIANTS:
            S, missing = build_S_o(valid, table, has_ta, v)
            if missing and v != "v0":
                print(f"  WARNING [{label} {v}]: {len(missing)} configurations without census row, v0 S_o kept: {missing[:6]}")
            res, alpha, X = analyze_variant(label, valid, X0, y, S, v)
            res["missing_census"] = [list(m) for m in missing]
            out[v] = res
            fits[label][v] = alpha
            Xs[v] = X
        m0 = out["v0"]["schemes"]["in"]["metrics"]
        print(f"\n  [control] {label} v0: MdAPE {m0['mdape_pct']:.1f}%  r2_linear {m0['r2_linear']:.4f}  "
              f"H2H {out['v0']['schemes']['in']['head_to_head']}  (EXP-CR-001: 44.3 / 0.9467 / 24/30 on 4090; "
              f"68.0 / 0.9823 / 24/30 on A100)")
        print_gpu(label, out, valid)
        summary["gpus"][label] = out
        data[label] = {"valid": valid, "y": y, "X": Xs}
    if len(fits) == 2:
        summary["cross_gpu_transfer"] = cross_gpu(fits, data)
    json_path = OUT_DIR / "exp_cr_006_refit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n" + "=" * 96)
    print(f"  Saved: {json_path.relative_to(BASE).as_posix()}")
    for k, v in summary["input_hashes"].items():
        print(f"    {v[:16]}...  {k}")
    print(f"  git: {summary['environment'].get('git_commit')}  dirty={summary['environment'].get('git_dirty')}")
    print("=" * 96)


if __name__ == "__main__":
    main()
