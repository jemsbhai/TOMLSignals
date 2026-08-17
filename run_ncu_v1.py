"""
TOMLSignals - EXP-CR-002 / EXP-CR-003: Nsight Compute, v1 driver
==================================================================
EXP-CR-002  cuFFT FP32 instruction count vs N (256 .. 65536, B = 1): tests the
            split-radix count 4N log2 N - 6N + 8 at every size, not only N = 4096.
EXP-CR-003  The 14 algorithms not profiled in F-001 (v0 run_ncu_profile.py):
            lms, nlms, rls, apa_p4, kalman, ekf, ukf, particle_1k, fastica, nmf,
            music, esprit, dwt_db4, iir_butter4 (torchaudio path).

Method for Python-loop algorithms (fixed iteration counts, kernel sequence
periodic with period K = kernels per iteration from the EXP-CR-005 census):
  window A: --launch-skip 0        --launch-count s+K   (setup + first iteration)
  window B: --launch-skip s+K      --launch-count 2K    (two full periods)
  per-iteration = B / 2 (alignment-independent inside the loop);
  per-invocation = iterations * per-iteration + max(A - per-iteration, 0).
  A/B halves are compared to confirm periodicity. About 100 to 170 kernel
  replays per algorithm instead of 1,300 to 7,800.
Everything else is profiled in full (2 to 44 kernels).

Uses v0's metric list, CSV parser and kernel summarizer (run_ncu_profile.py,
unmodified) and v0's harness (profile_single.py: 3 warmups, cudaProfilerStart
around one invocation). Writes data/ncu_profiles/ncu_summary_v1.json
(= v0 entries tagged as such + new entries) and per-config raw CSVs under
data/ncu_profiles/v1/. v0's ncu_summary.json is not touched.

Usage (repo root):  python run_ncu_v1.py [--only fft lms ...] [--ncu <ncu.bat>]
Author: Muntaser Syed        Date: August 2026
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import run_ncu_profile as v0ncu  # noqa: E402  (v0, unmodified: NCU_METRICS, parse_ncu_csv, summarize_kernels)
from shared.to_model import predict_to, get_seq_steps, SEQ_STEPS  # noqa: E402

NCU_BAT_DEFAULT = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat"
OUT_DIR = BASE / "data" / "ncu_profiles"
RAW_DIR = OUT_DIR / "v1"
V0_SUMMARY = OUT_DIR / "ncu_summary.json"
V1_SUMMARY = OUT_DIR / "ncu_summary_v1.json"
CENSUS_CSV = BASE / "data" / "camera_ready" / "exp_cr_005_kernel_census.csv"

# EXP-CR-002: cuFFT sweep (full profiles)
FFT_SWEEP = [("fft", N, 1) for N in (256, 1024, 4096, 16384, 65536)]

# EXP-CR-003: full profiles (few kernels)
FULL_CONFIGS = [
    ("dwt_db4", 4096, 1),
    ("iir_butter4", 4096, 1),      # torchaudio fused path on the 4090
    ("music", 1024, 1),
    ("esprit", 1024, 1),
]

# EXP-CR-003: Python-loop algorithms, one-iteration windows (K from census)
LOOP_CONFIGS = [
    # (alg, N, B)
    ("lms", 4096, 1), ("nlms", 4096, 1), ("rls", 4096, 1), ("apa_p4", 4096, 1),
    ("kalman", 4096, 1), ("ekf", 4096, 1), ("ukf", 4096, 1), ("particle_1k", 4096, 1),
    ("fastica", 1024, 1), ("nmf", 1024, 1),
]


def census_kernels_per_iter():
    """K (kernels per outer iteration) and setup kernels s from the census CSV."""
    if not CENSUS_CSV.exists():
        raise FileNotFoundError(CENSUS_CSV)
    out = {}
    with open(CENSUS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["repeat_idx"]) != 0 or row["variant"] != "default":
                continue
            alg = row["algorithm"]
            outer = int(float(row["outer_iters"] or 0))
            if outer <= 0:
                continue
            kernels = int(float(row["n_kernels"]))
            K = kernels // outer
            s = kernels - K * outer
            out.setdefault(alg, (K, s, outer, kernels))   # counts identical across N
    return out


def run_ncu(ncu_bat, alg, N, B, launch_skip=None, launch_count=None, tag="", timeout=1800):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / f"ncu_{alg}_N{N}_B{B}{('_' + tag) if tag else ''}.csv"
    cmd = [f'"{ncu_bat}"', "--metrics", ",".join(v0ncu.NCU_METRICS), "--csv",
           "--profile-from-start", "off"]
    if launch_skip is not None:
        cmd += ["--launch-skip", str(launch_skip)]
    if launch_count is not None:
        cmd += ["--launch-count", str(launch_count)]
    cmd += ["python", "profile_single.py", "--alg", alg, "--N", str(N), "--B", str(B)]
    cmd_str = " ".join(cmd)
    t0 = time.perf_counter()
    result = subprocess.run(cmd_str, capture_output=True, text=True, timeout=timeout,
                            cwd=str(BASE), shell=True)
    dt = time.perf_counter() - t0
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(f"ncu rc={result.returncode} for {alg} N={N} B={B} {tag}: {tail}")
    kernels = v0ncu.parse_ncu_csv(result.stdout)
    return kernels, dt, csv_path


def totals_of(kernels):
    t = v0ncu.summarize_kernels(kernels)
    return {"fp32_fma": t["fp32_fma"], "fp32_add": t["fp32_add"], "fp32_mul": t["fp32_mul"],
            "fp32_total": t["fp32_total"], "int_total": t.get("int_total", 0.0),
            "fp_misc": t.get("fp_misc", 0.0), "dram_bytes": t["dram_bytes_total"],
            "dram_words": t["dram_words_32bit"], "n_kernels": len(kernels)}


def kernel_names(kernels, n=6):
    c = Counter(k.get("_kernel_name", "?")[:70] for k in kernels)
    return c.most_common(n)


def split_radix(N):
    return 4 * N * math.log2(N) - 6 * N + 8


def cooley_tukey(N):
    return 5 * N * math.log2(N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these algorithms")
    ap.add_argument("--ncu", default=NCU_BAT_DEFAULT)
    ap.add_argument("--skip-fft", action="store_true")
    args = ap.parse_args()
    if not Path(args.ncu).exists():
        print(f"ncu not found at {args.ncu}; pass --ncu")
        sys.exit(2)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"TOMLSignals EXP-CR-002/003 Nsight Compute v1 driver, started {started}")
    print(f"ncu: {args.ncu}")
    kpi = census_kernels_per_iter()
    print("census K (kernels/iter), s (setup), iterations: " +
          ", ".join(f"{a}={v[0]}/{v[1]}/{v[2]}" for a, v in sorted(kpi.items())))

    entries = []
    # ---- EXP-CR-002: FFT sweep -------------------------------------------
    if not args.skip_fft and (not args.only or "fft" in args.only):
        print("\n=== EXP-CR-002: cuFFT FP32 instruction count vs N (B = 1) ===")
        print(f"  {'N':>6s} {'kern':>4s} {'FP32 meas':>11s} {'split-radix':>11s} {'ratio':>6s} "
              f"{'Cooley-Tukey':>12s} {'ratio':>6s} {'model MACs':>11s} {'ratio':>6s} {'DRAM words':>10s}  time")
        for alg, N, B in FFT_SWEEP:
            try:
                kernels, dt, path = run_ncu(args.ncu, alg, N, B)
            except Exception as e:
                print(f"  N={N}: ERROR {e}")
                continue
            t = totals_of(kernels)
            pred = predict_to(alg, N, B)
            model_macs = pred["to_compute"] / 5000.0
            sr, ct = split_radix(N), cooley_tukey(N)
            e = {"experiment": "EXP-CR-002", "algorithm": alg, "N": N, "B": B, "mode": "full",
                 "measured": t, "split_radix_ops": sr, "cooley_tukey_ops": ct,
                 "ratio_meas_over_split_radix": t["fp32_total"] / sr,
                 "ratio_meas_over_cooley_tukey": t["fp32_total"] / ct,
                 "to_pred_macs": model_macs, "ratio_fp_actual_vs_predicted": t["fp32_total"] / model_macs if model_macs else None,
                 "to_pred_mem_words": pred["to_memory"] / 10000.0,
                 "kernel_names": kernel_names(kernels), "csv": str(path.relative_to(BASE)), "seconds": dt}
            entries.append(e)
            print(f"  {N:6d} {t['n_kernels']:4d} {t['fp32_total']:11.0f} {sr:11.0f} {t['fp32_total']/sr:6.3f} "
                  f"{ct:12.0f} {t['fp32_total']/ct:6.3f} {model_macs:11.0f} "
                  f"{(t['fp32_total']/model_macs if model_macs else float('nan')):6.3f} {t['dram_words']:10.0f}  {dt:5.0f}s")

    # ---- EXP-CR-003: full profiles ---------------------------------------
    print("\n=== EXP-CR-003: full profiles ===")
    print(f"  {'algorithm':14s} {'N':>6s} {'kern':>4s} {'FP32 meas':>11s} {'model MACs':>11s} {'ratio':>7s} "
          f"{'DRAM words':>10s} {'model words':>11s} {'ratio':>7s}  time")
    for alg, N, B in FULL_CONFIGS:
        if args.only and alg not in args.only:
            continue
        try:
            kernels, dt, path = run_ncu(args.ncu, alg, N, B)
        except Exception as e:
            print(f"  {alg}: ERROR {e}")
            continue
        t = totals_of(kernels)
        pred = predict_to(alg, N, B)
        model_macs = pred["to_compute"] / 5000.0
        model_words = pred["to_memory"] / 10000.0
        e = {"experiment": "EXP-CR-003", "algorithm": alg, "N": N, "B": B, "mode": "full",
             "measured": t, "fp32_total_per_invocation": t["fp32_total"],
             "to_pred_macs": model_macs, "ratio_fp_actual_vs_predicted": t["fp32_total"] / model_macs if model_macs else None,
             "to_pred_mem_words": model_words,
             "ratio_mem_actual_vs_predicted": t["dram_words"] / model_words if model_words else None,
             "kernel_names": kernel_names(kernels), "csv": str(path.relative_to(BASE)), "seconds": dt}
        entries.append(e)
        print(f"  {alg:14s} {N:6d} {t['n_kernels']:4d} {t['fp32_total']:11.0f} {model_macs:11.0f} "
              f"{(t['fp32_total']/model_macs if model_macs else float('nan')):7.3f} {t['dram_words']:10.0f} "
              f"{model_words:11.0f} {(t['dram_words']/model_words if model_words else float('nan')):7.3f}  {dt:5.0f}s")

    # ---- EXP-CR-003: Python-loop windows ---------------------------------
    print("\n=== EXP-CR-003: Python-loop algorithms, one-iteration windows ===")
    print(f"  {'algorithm':12s} {'N':>6s} {'K':>3s} {'s':>2s} {'iters':>5s} {'FP32/iter':>10s} {'halves A|B':>14s} "
          f"{'FP32/call est':>13s} {'model MACs':>11s} {'ratio':>7s} {'DRAM/iter':>10s}  time")
    for alg, N, B in LOOP_CONFIGS:
        if args.only and alg not in args.only:
            continue
        if alg not in kpi:
            print(f"  {alg}: no census entry; skipped")
            continue
        K, s, outer_census, kern_census = kpi[alg]
        outer = int(SEQ_STEPS[alg](N, B)) if alg in SEQ_STEPS else outer_census
        try:
            kA, dtA, pathA = run_ncu(args.ncu, alg, N, B, launch_skip=0, launch_count=s + K, tag="winA")
            kB, dtB, pathB = run_ncu(args.ncu, alg, N, B, launch_skip=s + K, launch_count=2 * K, tag="winB")
        except Exception as e:
            print(f"  {alg}: ERROR {e}")
            continue
        tA, tB = totals_of(kA), totals_of(kB)
        # periodicity check: first K vs second K kernels of window B
        h1, h2 = totals_of(kB[:K]), totals_of(kB[K:2 * K])
        per_iter = tB["fp32_total"] / 2.0
        setup = max(tA["fp32_total"] - per_iter, 0.0)
        est_total = setup + outer * per_iter
        dram_iter = tB["dram_words"] / 2.0
        pred = predict_to(alg, N, B)
        model_macs = pred["to_compute"] / 5000.0
        model_words = pred["to_memory"] / 10000.0
        e = {"experiment": "EXP-CR-003", "algorithm": alg, "N": N, "B": B, "mode": "loop-window",
             "K_kernels_per_iter": K, "setup_kernels": s, "iterations": outer, "census_kernels": kern_census,
             "window_A": tA, "window_B": tB, "window_B_first_half": h1, "window_B_second_half": h2,
             "fp32_per_iter": per_iter, "fp32_setup": setup, "fp32_total_per_invocation": est_total,
             "dram_words_per_iter": dram_iter, "dram_words_per_invocation": setup * 0 + outer * dram_iter,
             "to_pred_macs": model_macs, "ratio_fp_actual_vs_predicted": est_total / model_macs if model_macs else None,
             "to_pred_mem_words": model_words,
             "ratio_mem_actual_vs_predicted": (outer * dram_iter) / model_words if model_words else None,
             "kernel_names_B": kernel_names(kB, 10),
             "csv": [str(pathA.relative_to(BASE)), str(pathB.relative_to(BASE))],
             "seconds": dtA + dtB,
             "model_seq_steps": int(get_seq_steps(alg, N, B))}
        entries.append(e)
        flag = "" if (h1["fp32_total"] == h2["fp32_total"]) else "  [halves differ]"
        print(f"  {alg:12s} {N:6d} {K:3d} {s:2d} {outer:5d} {per_iter:10.0f} "
              f"{h1['fp32_total']:6.0f}|{h2['fp32_total']:6.0f} {est_total:13.0f} {model_macs:11.0f} "
              f"{(est_total/model_macs if model_macs else float('nan')):7.3f} {dram_iter:10.0f}  {dtA + dtB:5.0f}s{flag}")

    # ---- write v1 summary (v0 entries + new) -----------------------------
    v0_entries = []
    if V0_SUMMARY.exists():
        with open(V0_SUMMARY) as f:
            for r in json.load(f):
                r = dict(r)
                r["experiment"] = "F-001 (v0)"
                r["mode"] = "full"
                v0_entries.append(r)
    out = {"created_utc": started, "ncu": args.ncu, "v0_entries": v0_entries, "v1_entries": entries}
    with open(V1_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  Saved {V1_SUMMARY.relative_to(BASE)}: {len(v0_entries)} v0 entries + {len(entries)} new; raw CSVs in {RAW_DIR.relative_to(BASE)}/")


if __name__ == "__main__":
    main()
