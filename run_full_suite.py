"""
TOMLSignals - Full Benchmark Suite Runner
==========================================
Runs all 37 algorithms across signal lengths with:
  - Auto-calibrated batch sizes (ensures t/call >= 1ms for reliable power measurement)
  - Per-algorithm idle baseline measurement
  - Full cooldown between algorithms

Usage:
  python run_full_suite.py --duration 5
  python run_full_suite.py --quick
  python run_full_suite.py --category transform --duration 10
"""

import sys
import argparse
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.harness import (
    benchmark_algorithm, calibrate_batch_size,
    save_result, save_results_csv, BenchmarkResult,
)
from algorithms.transforms import TRANSFORMS
from algorithms.filters import FILTERS
from algorithms.adaptive import ADAPTIVE
from algorithms.estimation import ESTIMATION
from algorithms.spectral import SPECTRAL
from algorithms.decomposition import DECOMPOSITION
from algorithms.compression import COMPRESSION
from algorithms.ml_enhanced import ML_ENHANCED

ALL_ALGORITHMS = {
    "transform": TRANSFORMS,
    "filter": FILTERS,
    "adaptive": ADAPTIVE,
    "estimation": ESTIMATION,
    "spectral": SPECTRAL,
    "decomposition": DECOMPOSITION,
    "compression": COMPRESSION,
    "ml_enhanced": ML_ENHANCED,
}

# Signal lengths to test (adjusted per category)
SIGNAL_LENGTHS_DEFAULT = [256, 1024, 4096, 16384]
SIGNAL_LENGTHS_QUICK = [1024, 4096]

# Some algorithms need different signal lengths
CUSTOM_LENGTHS = {
    "direct_dft": [256, 512, 1024, 2048],  # O(N^2), keep small
    "iir_butter4": [1024, 4096, 16384],
    "music": [256, 512, 1024],
    "esprit": [256, 512, 1024],
    "svd": [256, 512, 1024],
    "pca": [256, 512, 1024],
    "fastica": [256, 512, 1024],
    "nmf": [256, 512, 1024],
    "jpeg_q50": [4096, 16384],  # 64x64, 128x128 images
    "mdct_audio": [4096, 16384, 65536],
}

# Max batch sizes per algorithm (memory safety limits)
MAX_BATCH = {
    "transformer_denoiser": 512,   # self-attention is O(N^2) in memory
    "direct_dft": 512,             # O(N^2) DFT matrix
    "fastica": 256,                # multiple matrix ops in iteration loop
    "nmf": 256,                    # multiple matrix ops in iteration loop
    "svd": 256,                    # full SVD is memory-heavy
    "kalman": 128,                 # large state matrices
    "ekf": 128,
    "ukf": 128,
    "particle_1k": 128,            # 1000 particles per batch
    "apa_p4": 128,
    "rls": 128,
}
DEFAULT_MAX_BATCH = 2048


def run_suite(quick: bool = False, categories: list = None, duration_s: float = 5.0,
              target_time_ms: float = 1.0):
    results = []
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = SIGNAL_LENGTHS_QUICK if quick else SIGNAL_LENGTHS_DEFAULT
    device = torch.device("cuda")

    total_algs = sum(
        len(CUSTOM_LENGTHS.get(name, lengths)[:2] if quick else CUSTOM_LENGTHS.get(name, lengths))
        for cat_name, algorithms in ALL_ALGORITHMS.items()
        if not categories or cat_name in categories
        for name in algorithms
    )
    completed = 0

    for cat_name, algorithms in ALL_ALGORITHMS.items():
        if categories and cat_name not in categories:
            continue

        print(f"\n{'='*70}")
        print(f"  Category: {cat_name} ({len(algorithms)} algorithms)")
        print(f"{'='*70}")

        for alg_name, (setup_fn, run_fn, defaults) in algorithms.items():
            alg_lengths = CUSTOM_LENGTHS.get(alg_name, lengths)
            if quick:
                alg_lengths = alg_lengths[:2]

            max_b = MAX_BATCH.get(alg_name, DEFAULT_MAX_BATCH)

            for N in alg_lengths:
                completed += 1
                print(f"\n  [{completed}/{total_algs}] {alg_name} | N={N} | ", end="", flush=True)

                try:
                    # Auto-calibrate batch size
                    batch_size = calibrate_batch_size(
                        setup_fn=setup_fn,
                        run_fn=run_fn,
                        signal_length=N,
                        precision="fp32",
                        device=device,
                        extra_params=defaults,
                        target_time_ms=target_time_ms,
                        max_batch=max_b,
                    )
                    print(f"B={batch_size} | ", end="", flush=True)

                    # Brief cooldown after calibration
                    torch.cuda.empty_cache()
                    time.sleep(1.0)

                    result = benchmark_algorithm(
                        name=alg_name,
                        category=cat_name,
                        setup_fn=setup_fn,
                        run_fn=run_fn,
                        signal_length=N,
                        batch_size=batch_size,
                        precision="fp32",
                        duration_s=duration_s,
                        warmup_iters=50,
                        extra_params=defaults,
                    )
                    print(f"Δ={result.delta_power_w:.1f}W  "
                          f"E/call={result.energy_per_call_j:.4e}J  "
                          f"t/call={result.time_per_call_us:.1f}us  "
                          f"idle={result.idle_power_w:.1f}W@{result.idle_temp_c:.0f}°C  "
                          f"Twait={result.thermal_wait_s:.0f}s  "
                          f"samples={result.power_samples}/{result.idle_samples}")
                    results.append(result)
                    save_result(result, str(output_dir / cat_name))

                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    torch.cuda.empty_cache()

    # Save combined CSV
    if results:
        csv_path = str(output_dir / "all_results.csv")
        save_results_csv(results, csv_path)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {len(results)} benchmarks completed")
    print(f"{'='*70}")
    print(f"\n{'Algorithm':25s} {'N':>6s} {'B':>5s} {'Idle(W)':>8s} {'T(°C)':>6s} {'ΔPow(W)':>8s} "
          f"{'E/call(J)':>12s} {'t/call(us)':>10s} {'Twait':>6s}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: (x.category, x.algorithm, x.signal_length)):
        print(f"  {r.algorithm:23s} {r.signal_length:>6d} {r.batch_size:>5d} "
              f"{r.idle_power_w:>8.1f} {r.idle_temp_c:>6.0f} {r.delta_power_w:>8.1f} "
              f"{r.energy_per_call_j:>12.4e} {r.time_per_call_us:>10.1f} "
              f"{r.thermal_wait_s:>5.0f}s")

    # Flag any remaining negative deltas
    negatives = [r for r in results if r.delta_power_w < 0]
    if negatives:
        print(f"\n  WARNING: {len(negatives)} benchmarks still show negative delta power:")
        for r in negatives:
            print(f"    {r.algorithm} N={r.signal_length} B={r.batch_size}: "
                  f"Δ={r.delta_power_w:.1f}W (idle={r.idle_power_w:.1f}W, active={r.mean_power_w:.1f}W)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOMLSignals Full Benchmark Suite")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer signal lengths, shorter duration")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only one category")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds per benchmark (default: 5)")
    parser.add_argument("--target-time", type=float, default=1.0,
                        help="Target ms per call for batch calibration (default: 1.0)")
    args = parser.parse_args()

    cats = [args.category] if args.category else None
    run_suite(quick=args.quick, categories=cats, duration_s=args.duration,
              target_time_ms=args.target_time)
