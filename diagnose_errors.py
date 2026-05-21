"""
Diagnostic analysis: Identify error sources and patterns.
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.to_model import predict_to, TO_MODELS, get_seq_steps


def load_csv(path, gpu_name="Unknown"):
    points = []
    with open(path) as f:
        for row in csv.DictReader(f):
            points.append(row | {"gpu_name": row.get("gpu_name", gpu_name)})
    return points


def analyze_errors(points, label):
    print(f"\n{'='*80}")
    print(f"  DIAGNOSTIC: {label}")
    print(f"  {len(points)} data points")
    print(f"{'='*80}")

    results = []
    for p in points:
        alg = p["algorithm"]
        N = int(p["signal_length"])
        B = int(p["batch_size"])
        e_meas = float(p["energy_per_call_j"])
        dp = float(p["delta_power_w"])
        t_us = float(p["time_per_call_us"])

        if alg not in TO_MODELS or e_meas <= 0:
            continue

        r = predict_to(alg, N, B)
        seq = get_seq_steps(alg, N, B)

        results.append({
            "alg": alg, "cat": p["category"], "N": N, "B": B,
            "e_meas": e_meas, "dp": dp, "t_us": t_us,
            "to_c": r["to_compute"], "to_m": r["to_memory"], "to_t": r["to_total"],
            "seq": seq,
        })

    # ---- Error pattern 1: by signal length ----
    print(f"\n  --- Error by signal length (parallel algorithms only) ---")
    by_n = defaultdict(list)
    for r in results:
        if r["seq"] == 0:
            by_n[r["N"]].append(r)

    for N in sorted(by_n.keys()):
        pts = by_n[N]
        # Fit alpha on just these points
        to_t = np.array([r["to_t"] for r in pts])
        e = np.array([r["e_meas"] for r in pts])
        alpha = np.dot(to_t, e) / np.dot(to_t, to_t)
        pred = alpha * to_t
        rel_err = np.abs(e - pred) / e * 100
        print(f"  N={N:>6d}: {len(pts):>3d} pts, "
              f"mean_err={rel_err.mean():.1f}%, median_err={np.median(rel_err):.1f}%, "
              f"alpha={alpha*1e15:.2f} fJ/TO")

    # ---- Error pattern 2: by batch size ----
    print(f"\n  --- Error by batch size (parallel only) ---")
    by_b = defaultdict(list)
    for r in results:
        if r["seq"] == 0:
            by_b[r["B"]].append(r)

    for B in sorted(by_b.keys()):
        pts = by_b[B]
        to_t = np.array([r["to_t"] for r in pts])
        e = np.array([r["e_meas"] for r in pts])
        if len(pts) < 2:
            continue
        alpha = np.dot(to_t, e) / np.dot(to_t, to_t)
        pred = alpha * to_t
        rel_err = np.abs(e - pred) / e * 100
        print(f"  B={B:>5d}: {len(pts):>3d} pts, "
              f"mean_err={rel_err.mean():.1f}%, median_err={np.median(rel_err):.1f}%, "
              f"alpha={alpha*1e15:.2f} fJ/TO")

    # ---- Error pattern 3: worst offenders ----
    print(f"\n  --- Top 20 worst relative errors (parallel only) ---")
    parallel = [r for r in results if r["seq"] == 0]
    # Global alpha for parallel
    to_t = np.array([r["to_t"] for r in parallel])
    e = np.array([r["e_meas"] for r in parallel])
    alpha_global = np.dot(to_t, e) / np.dot(to_t, to_t)

    for r in parallel:
        r["e_pred"] = alpha_global * r["to_t"]
        r["rel_err"] = abs(r["e_meas"] - r["e_pred"]) / r["e_meas"] * 100
        r["ratio"] = r["e_pred"] / r["e_meas"]

    parallel.sort(key=lambda x: x["rel_err"], reverse=True)
    print(f"  {'Alg':20s} {'N':>6s} {'B':>5s} {'E_meas':>10s} {'E_pred':>10s} "
          f"{'Ratio':>7s} {'Err%':>7s} {'dP(W)':>7s} {'t(us)':>10s}")
    for r in parallel[:20]:
        print(f"  {r['alg']:20s} {r['N']:>6d} {r['B']:>5d} "
              f"{r['e_meas']:>10.4e} {r['e_pred']:>10.4e} "
              f"{r['ratio']:>7.2f}x {r['rel_err']:>6.1f}% "
              f"{r['dp']:>7.1f} {r['t_us']:>10.1f}")

    # ---- Error pattern 4: overpredict vs underpredict ----
    over = [r for r in parallel if r["ratio"] > 1]
    under = [r for r in parallel if r["ratio"] <= 1]
    print(f"\n  --- Over/under prediction (parallel only) ---")
    print(f"  Overpredicted:  {len(over):>3d} points, "
          f"mean ratio={np.mean([r['ratio'] for r in over]):.2f}x")
    print(f"  Underpredicted: {len(under):>3d} points, "
          f"mean ratio={np.mean([r['ratio'] for r in under]):.2f}x")

    # ---- Error pattern 5: by category ----
    print(f"\n  --- Error by category (parallel only) ---")
    by_cat = defaultdict(list)
    for r in parallel:
        by_cat[r["cat"]].append(r)

    for cat in sorted(by_cat.keys()):
        pts = by_cat[cat]
        errs = [r["rel_err"] for r in pts]
        ratios = [r["ratio"] for r in pts]
        print(f"  {cat:20s}: n={len(pts):>3d}, "
              f"mean_err={np.mean(errs):.1f}%, median_err={np.median(errs):.1f}%, "
              f"mean_ratio={np.mean(ratios):.2f}x")

    # ---- Error pattern 6: GPU utilization proxy ----
    # Delta power / TDP as a utilization proxy
    print(f"\n  --- Error vs delta power (GPU utilization proxy) ---")
    dp_bins = [(0, 10), (10, 30), (30, 60), (60, 100), (100, 200)]
    for lo, hi in dp_bins:
        pts = [r for r in parallel if lo <= r["dp"] < hi]
        if pts:
            errs = [r["rel_err"] for r in pts]
            print(f"  ΔP={lo:>3d}-{hi:>3d}W: n={len(pts):>3d}, "
                  f"mean_err={np.mean(errs):.1f}%, median_err={np.median(errs):.1f}%")

    # ---- Sequential algorithm analysis ----
    sequential = [r for r in results if r["seq"] > 0]
    if sequential:
        print(f"\n  --- Sequential algorithms ({len(sequential)} points) ---")
        print(f"  {'Alg':20s} {'N':>6s} {'B':>5s} {'seq':>7s} {'E_meas':>10s} "
              f"{'TO_total':>10s} {'E/seq(mJ)':>10s} {'dP(W)':>7s}")
        for r in sorted(sequential, key=lambda x: (x["alg"], x["N"])):
            e_per_seq = r["e_meas"] / r["seq"] * 1000 if r["seq"] > 0 else 0
            print(f"  {r['alg']:20s} {r['N']:>6d} {r['B']:>5d} {r['seq']:>7d} "
                  f"{r['e_meas']:>10.4e} {r['to_t']:>10.2e} "
                  f"{e_per_seq:>10.3f} {r['dp']:>7.1f}")


# Main
base = Path(__file__).resolve().parent

local_csv = base / "data" / "results" / "all_results.csv"
if local_csv.exists():
    local = load_csv(str(local_csv), "RTX 4090 Laptop")
    analyze_errors(local, "RTX 4090 Laptop GPU")

server_csv = base / "data" / "server_results" / "results" / "all_results.csv"
if server_csv.exists():
    server = load_csv(str(server_csv), "A100-SXM4-40GB")
    # Add IIR rerun JSONs
    iir_dir = base / "data" / "server_results" / "results" / "filter"
    existing = {(r["algorithm"], int(r["signal_length"])) for r in server
                if r["algorithm"] == "iir_butter4"}
    for jf in sorted(iir_dir.glob("iir_butter4_*.json")):
        with open(jf) as f:
            d = json.load(f)
        key = (d["algorithm"], d["signal_length"])
        if key not in existing:
            server.append({k: str(v) for k, v in d.items() if k != "params"})
            existing.add(key)
    analyze_errors(server, "A100-SXM4-40GB")
