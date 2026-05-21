"""Quick diagnostic: find outliers in the corrected TO model."""
import csv, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from shared.to_model import predict_to, TO_MODELS, get_seq_steps

base = Path(".")

def load_and_analyze(csv_path, gpu_name):
    points = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            alg = row["algorithm"]
            N = int(row["signal_length"])
            B = int(row["batch_size"])
            E = float(row["energy_per_call_j"])
            if alg in TO_MODELS and E > 0:
                r = predict_to(alg, N, B)
                points.append({
                    "alg": alg, "N": N, "B": B, "E": E,
                    "to_c": r["to_compute"], "to_m": r["to_memory"],
                    "to_t": r["to_total"], "seq": get_seq_steps(alg, N, B),
                })

    print(f"\n{'='*80}")
    print(f"  {gpu_name}: {len(points)} points")
    print(f"{'='*80}")

    # Show TO compute range
    to_vals = sorted(points, key=lambda p: p["to_c"], reverse=True)
    print(f"\n  Top 10 by TO_compute:")
    print(f"  {'Algorithm':25s} {'N':>6s} {'TO_compute':>14s} {'Energy(J)':>12s} {'E/TO_c':>12s}")
    for p in to_vals[:10]:
        ratio = p["E"] / p["to_c"] if p["to_c"] > 0 else 0
        print(f"  {p['alg']:25s} {p['N']:>6d} {p['to_c']:>14.2e} {p['E']:>12.4e} {ratio:>12.4e}")

    print(f"\n  Bottom 10 by TO_compute:")
    for p in to_vals[-10:]:
        ratio = p["E"] / p["to_c"] if p["to_c"] > 0 else 0
        print(f"  {p['alg']:25s} {p['N']:>6d} {p['to_c']:>14.2e} {p['E']:>12.4e} {ratio:>12.4e}")

    # E/TO_c ratio spread — should be roughly constant for a good model
    ratios = [p["E"]/p["to_c"] for p in points if p["to_c"] > 0]
    print(f"\n  E/TO_compute ratio (should be ~constant for good fit):")
    print(f"    min: {min(ratios):.4e}")
    print(f"    max: {max(ratios):.4e}")
    print(f"    spread: {max(ratios)/min(ratios):.0f}x")

    # Show sequential algorithms
    seq_pts = [p for p in points if p["seq"] > 0]
    par_pts = [p for p in points if p["seq"] == 0]
    print(f"\n  Sequential ({len(seq_pts)} pts): {sorted(set(p['alg'] for p in seq_pts))}")
    print(f"  Parallel ({len(par_pts)} pts)")

    # Show MDCT specifically
    mdct_pts = [p for p in points if "mdct" in p["alg"]]
    if mdct_pts:
        print(f"\n  MDCT data points:")
        for p in mdct_pts:
            print(f"    N={p['N']} B={p['B']} E={p['E']:.4e} TO_c={p['to_c']:.4e} seq={p['seq']}")

# Load 4090
csv_4090 = base / "data/results/all_results.csv"
if csv_4090.exists():
    load_and_analyze(csv_4090, "RTX 4090 Laptop")

# Load A100
csv_a100 = base / "data/server_results/results/all_results.csv"
if csv_a100.exists():
    load_and_analyze(csv_a100, "A100 SXM4")
