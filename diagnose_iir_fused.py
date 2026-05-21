"""
Diagnose IIR fused-sequential behavior.
Compare per-step energy across sequential algorithm categories.
"""
import csv
from pathlib import Path
import numpy as np

def load_csv(path):
    points = []
    with open(path) as f:
        for row in csv.DictReader(f):
            points.append(row)
    return points

# Sequential algorithms and their step counts
def get_steps(alg, N, B):
    """Return (n_steps, category) for sequential algorithms."""
    if alg == "iir_butter4":
        return B * N, "fused_sequential"
    elif alg in ("lms", "nlms"):
        M = 32
        return B * max(0, min(N, M + 200) - M), "python_loop"
    elif alg == "rls":
        M = 32
        return B * max(0, min(N, M + 100) - M), "python_loop"
    elif alg == "apa_p4":
        M, P = 32, 4
        return B * max(0, min(N, M + P + 100) - (M + P)), "python_loop"
    elif alg in ("kalman", "ekf"):
        return B * min(N, 200), "python_loop"
    elif alg == "ukf":
        T = min(N, 100)
        return B * T * (1 + 4 + 9), "python_loop"  # ~14 inner ops
    elif alg == "particle_1k":
        return B * min(N, 200), "python_loop"
    elif alg in ("fastica", "nmf"):
        return B * 50, "python_loop"
    elif alg == "pca":
        return B, "python_loop"
    elif alg == "mdct_audio":
        return B * max(1, min((N - 1024) // 512 + 1, 50)), "python_loop"
    return 0, "parallel"


print("=" * 90)
print("  PER-STEP ENERGY COMPARISON: Python-loop vs Fused-sequential")
print("=" * 90)

for gpu_name, csv_path, has_torchaudio in [
    ("RTX 4090 Laptop", "data/results/all_results.csv", True),
    ("A100 SXM4", "data/server_results/results/all_results.csv", False),
]:
    print(f"\n  --- {gpu_name} (torchaudio={has_torchaudio}) ---")
    points = load_csv(csv_path)

    print(f"  {'Algorithm':20s} {'N':>6s} {'B':>5s} {'Energy(J)':>12s} {'Steps':>8s} "
          f"{'uJ/step':>10s} {'Category':>18s}")
    print(f"  {'-'*85}")

    categories = {}
    for row in points:
        alg = row["algorithm"]
        N = int(row["signal_length"])
        B = int(row["batch_size"])
        E = float(row["energy_per_call_j"])

        steps, cat = get_steps(alg, N, B)
        if steps == 0 or cat == "parallel":
            continue

        # For IIR on 4090 (torchaudio available), it's fused sequential
        # For IIR on A100 (no torchaudio), it's python loop
        if alg == "iir_butter4" and not has_torchaudio:
            cat = "python_loop"

        uj_per_step = E / steps * 1e6 if steps > 0 else 0

        print(f"  {alg:20s} {N:>6d} {B:>5d} {E:>12.4e} {steps:>8d} "
              f"{uj_per_step:>10.1f} {cat:>18s}")

        if cat not in categories:
            categories[cat] = []
        categories[cat].append(uj_per_step)

    print(f"\n  Summary:")
    for cat, vals in sorted(categories.items()):
        print(f"    {cat:20s}: median={np.median(vals):>10.1f} µJ/step  "
              f"mean={np.mean(vals):>10.1f} µJ/step  "
              f"range=[{min(vals):.1f}, {max(vals):.1f}]  n={len(vals)}")

    if "fused_sequential" in categories and "python_loop" in categories:
        ratio = np.median(categories["python_loop"]) / np.median(categories["fused_sequential"])
        print(f"\n    Python-loop / Fused-sequential ratio: {ratio:.0f}×")
