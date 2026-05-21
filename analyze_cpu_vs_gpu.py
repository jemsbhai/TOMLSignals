"""
TOMLSignals - CPU vs GPU Energy Comparison Analysis
======================================================
Compares energy-per-signal-call between CPU (i9-14900HX) and GPU (RTX 4090 Laptop).
GPU energy is normalized by batch size (B=2048) for per-signal comparison.

Key question: When should a signal processing engineer use GPU vs CPU for energy?

Author: Muntaser Syed
Date: May 2026
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_csv(path):
    """Load results CSV into list of dicts with numeric conversion."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            # Convert numeric fields
            for k in r:
                if k in ("algorithm", "category", "precision", "cpu_name"):
                    continue
                try:
                    r[k] = float(r[k])
                except (ValueError, KeyError):
                    pass
            rows.append(r)
    return rows


def main():
    cpu_path = Path("data/cpu_results/all_cpu_results.csv")
    gpu_path = Path("data/results/all_results.csv")

    if not cpu_path.exists():
        print(f"ERROR: {cpu_path} not found")
        sys.exit(1)
    if not gpu_path.exists():
        print(f"ERROR: {gpu_path} not found")
        sys.exit(1)

    cpu_data = load_csv(cpu_path)
    gpu_data = load_csv(gpu_path)

    print("=" * 90)
    print("  TOMLSignals: CPU vs GPU Energy Comparison")
    print("  CPU: Intel Core i9-14900HX  |  GPU: NVIDIA RTX 4090 Laptop")
    print("=" * 90)

    # Build lookup: (algorithm, signal_length) -> row
    cpu_lookup = {}
    for r in cpu_data:
        key = (r["algorithm"], int(r["signal_length"]))
        cpu_lookup[key] = r

    gpu_lookup = {}
    for r in gpu_data:
        key = (r["algorithm"], int(r["signal_length"]))
        gpu_lookup[key] = r

    # Find common (algorithm, signal_length) pairs
    common_keys = sorted(set(cpu_lookup.keys()) & set(gpu_lookup.keys()))

    if not common_keys:
        print("No common (algorithm, signal_length) pairs found!")
        print(f"CPU algorithms: {sorted(set(r['algorithm'] for r in cpu_data))}")
        print(f"GPU signal lengths: {sorted(set(int(r['signal_length']) for r in gpu_data))}")
        sys.exit(1)

    print(f"\n  Common data points: {len(common_keys)}")
    print(f"  CPU benchmarks: {len(cpu_data)}")
    print(f"  GPU benchmarks: {len(gpu_data)}")

    # ===================================================================
    # Table 1: Per-algorithm comparison
    # ===================================================================
    print(f"\n{'=' * 90}")
    print(f"  {'Algorithm':<22} {'N':>6} {'CPU E/call':>12} {'GPU E/call':>12} "
          f"{'GPU E/sig':>12} {'Ratio':>8} {'Winner':>8}")
    print(f"  {'':22} {'':>6} {'(J)':>12} {'(J, B=2048)':>12} "
          f"{'(J, /B)':>12} {'CPU/GPU':>8} {'':>8}")
    print(f"  {'-' * 86}")

    results = []
    categories = defaultdict(list)

    for alg, N in common_keys:
        cpu_r = cpu_lookup[(alg, N)]
        gpu_r = gpu_lookup[(alg, N)]

        cpu_energy = float(cpu_r["energy_per_call_j"])
        gpu_energy_batch = float(gpu_r["energy_per_call_j"])
        gpu_batch = int(gpu_r["batch_size"])
        gpu_energy_per_sig = gpu_energy_batch / gpu_batch

        # Ratio: CPU/GPU. >1 means GPU is more efficient
        ratio = cpu_energy / gpu_energy_per_sig if gpu_energy_per_sig > 0 else float("inf")
        winner = "GPU" if ratio > 1 else "CPU"

        cpu_time = float(cpu_r["time_per_call_us"])
        gpu_time = float(gpu_r["time_per_call_us"]) / gpu_batch * 1e6  # per-signal

        row = {
            "algorithm": alg,
            "category": cpu_r["category"],
            "N": N,
            "cpu_energy_j": cpu_energy,
            "gpu_energy_batch_j": gpu_energy_batch,
            "gpu_energy_per_sig_j": gpu_energy_per_sig,
            "ratio": ratio,
            "winner": winner,
            "cpu_time_us": cpu_time,
            "cpu_delta_w": float(cpu_r["delta_power_w"]),
            "gpu_delta_w": float(gpu_r["delta_power_w"]),
        }
        results.append(row)
        categories[cpu_r["category"]].append(row)

        print(f"  {alg:<22} {N:>6} {cpu_energy:>12.6e} {gpu_energy_batch:>12.6e} "
              f"{gpu_energy_per_sig:>12.6e} {ratio:>8.1f}x {winner:>8}")

    # ===================================================================
    # Summary statistics
    # ===================================================================
    ratios = [r["ratio"] for r in results]
    gpu_wins = sum(1 for r in results if r["winner"] == "GPU")
    cpu_wins = sum(1 for r in results if r["winner"] == "CPU")

    print(f"\n{'=' * 90}")
    print(f"  SUMMARY")
    print(f"  {'-' * 86}")
    print(f"  Total comparisons: {len(results)}")
    print(f"  GPU more efficient: {gpu_wins} ({gpu_wins/len(results)*100:.0f}%)")
    print(f"  CPU more efficient: {cpu_wins} ({cpu_wins/len(results)*100:.0f}%)")
    print(f"  Median CPU/GPU ratio: {np.median(ratios):.1f}x")
    print(f"  Mean CPU/GPU ratio: {np.mean(ratios):.1f}x")
    print(f"  Min ratio: {min(ratios):.2f}x ({[r for r in results if r['ratio']==min(ratios)][0]['algorithm']})")
    print(f"  Max ratio: {max(ratios):.1f}x ({[r for r in results if r['ratio']==max(ratios)][0]['algorithm']})")

    # ===================================================================
    # Per-category summary
    # ===================================================================
    print(f"\n{'=' * 90}")
    print(f"  PER-CATEGORY SUMMARY")
    print(f"  {'-' * 86}")
    print(f"  {'Category':<20} {'GPU wins':>10} {'CPU wins':>10} {'Median ratio':>14}")
    print(f"  {'-' * 58}")

    for cat in sorted(categories.keys()):
        cat_results = categories[cat]
        cat_gpu = sum(1 for r in cat_results if r["winner"] == "GPU")
        cat_cpu = sum(1 for r in cat_results if r["winner"] == "CPU")
        cat_ratio = np.median([r["ratio"] for r in cat_results])
        print(f"  {cat:<20} {cat_gpu:>10} {cat_cpu:>10} {cat_ratio:>14.1f}x")

    # ===================================================================
    # Per-signal-length summary
    # ===================================================================
    print(f"\n{'=' * 90}")
    print(f"  PER-SIGNAL-LENGTH SUMMARY")
    print(f"  {'-' * 86}")
    print(f"  {'N':<10} {'GPU wins':>10} {'CPU wins':>10} {'Median ratio':>14}")
    print(f"  {'-' * 48}")

    for N in sorted(set(r["N"] for r in results)):
        n_results = [r for r in results if r["N"] == N]
        n_gpu = sum(1 for r in n_results if r["winner"] == "GPU")
        n_cpu = sum(1 for r in n_results if r["winner"] == "CPU")
        n_ratio = np.median([r["ratio"] for r in n_results])
        print(f"  {N:<10} {n_gpu:>10} {n_cpu:>10} {n_ratio:>14.1f}x")

    # ===================================================================
    # Interesting crossover points
    # ===================================================================
    print(f"\n{'=' * 90}")
    print(f"  CROSSOVER ANALYSIS: Algorithms where winner changes with N")
    print(f"  {'-' * 86}")

    # Group by algorithm, check if winner changes
    alg_results = defaultdict(list)
    for r in results:
        alg_results[r["algorithm"]].append(r)

    for alg in sorted(alg_results.keys()):
        ar = sorted(alg_results[alg], key=lambda x: x["N"])
        winners = [r["winner"] for r in ar]
        if len(set(winners)) > 1:  # crossover exists
            print(f"  {alg}:")
            for r in ar:
                print(f"    N={r['N']:>6}: CPU/GPU = {r['ratio']:.2f}x -> {r['winner']}")

    # ===================================================================
    # Save comparison CSV
    # ===================================================================
    out_path = Path("data/cpu_vs_gpu_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["algorithm", "category", "N", "cpu_energy_j",
              "gpu_energy_batch_j", "gpu_energy_per_sig_j",
              "ratio", "winner", "cpu_time_us", "cpu_delta_w", "gpu_delta_w"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n  Saved comparison to {out_path}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
