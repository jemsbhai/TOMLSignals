"""
TOMLSignals - CPU Benchmark Runner
====================================
Runs all 37 signal processing algorithms on CPU with power measurement.

Usage:
    python run_cpu_suite.py                    # full suite
    python run_cpu_suite.py --category filter  # single category
    python run_cpu_suite.py --algo fft         # single algorithm
    python run_cpu_suite.py --quick            # N=1024 only, 3s measurement

Results saved to data/cpu_results/
"""

import argparse
import sys
import time
from pathlib import Path

from shared.cpu_harness import (
    LHMSensorReader,
    benchmark_cpu_algorithm,
    save_cpu_result,
    save_cpu_results_csv,
)
from cpu_algorithms import CPU_ALGORITHMS


# Signal lengths matching GPU benchmarks
SIGNAL_LENGTHS = [256, 1024, 4096, 16384]

# Measurement duration (seconds) — longer for short algorithms
# to accumulate enough power samples above the noise floor
DURATION_S = 5.0

# Algorithms that are very slow per call at large N — use shorter
# measurement but more time per call still gives measurable ΔP
SLOW_ALGORITHMS = {
    "transformer_denoiser",  # O(N²) attention
    "music",                 # inner loop over freq bins
    "ukf",                   # Cholesky + sigma points per step
    "nmf",                   # iterative matrix factorization
    "fastica",               # iterative convergence
}


def run_suite(
    categories=None,
    algorithms=None,
    signal_lengths=None,
    duration_s=DURATION_S,
    output_dir="data/cpu_results",
    lhm=None,
):
    """Run the full CPU benchmark suite.

    Args:
        categories: List of categories to run (None = all).
        algorithms: List of algorithm names to run (None = all).
        signal_lengths: List of N values (None = all 4).
        duration_s: Active measurement duration per benchmark.
        output_dir: Directory for individual JSON results.
        lhm: LHMSensorReader (created if None).

    Returns:
        List of CPUBenchmarkResult.
    """
    if signal_lengths is None:
        signal_lengths = SIGNAL_LENGTHS

    if lhm is None:
        lhm = LHMSensorReader()

    results = []
    total_benchmarks = 0
    completed = 0

    # Count total benchmarks
    for cat_name, algs in CPU_ALGORITHMS.items():
        if categories and cat_name not in categories:
            continue
        for alg_name in algs:
            if algorithms and alg_name not in algorithms:
                continue
            total_benchmarks += len(signal_lengths)

    print(f"\nTOMLSignals CPU Benchmark Suite")
    print(f"{'=' * 60}")
    print(f"  Algorithms: {total_benchmarks // len(signal_lengths)}")
    print(f"  Signal lengths: {signal_lengths}")
    print(f"  Total benchmarks: {total_benchmarks}")
    print(f"  Measurement duration: {duration_s}s per benchmark")
    est_time = total_benchmarks * (30 + duration_s)  # ~30s settling + measurement
    print(f"  Estimated time: {est_time/60:.0f} minutes")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    t_suite_start = time.time()

    for cat_name, algs in CPU_ALGORITHMS.items():
        if categories and cat_name not in categories:
            continue

        print(f"\n  CATEGORY: {cat_name}")
        print(f"  {'-' * 50}")

        for alg_name, (setup_fn, run_fn) in algs.items():
            if algorithms and alg_name not in algorithms:
                continue

            for N in signal_lengths:
                completed += 1
                pct = completed / total_benchmarks * 100
                elapsed = time.time() - t_suite_start
                if completed > 1:
                    eta = elapsed / (completed - 1) * (total_benchmarks - completed)
                else:
                    eta = 0

                print(f"\n  [{completed}/{total_benchmarks} {pct:.0f}%] "
                      f"{alg_name} N={N} (ETA: {eta/60:.0f}m)")

                # Adjust duration for slow algorithms at large N
                d = duration_s
                if alg_name in SLOW_ALGORITHMS and N >= 4096:
                    d = max(duration_s, 10.0)  # longer to get enough samples

                try:
                    result = benchmark_cpu_algorithm(
                        name=alg_name,
                        category=cat_name,
                        setup_fn=setup_fn,
                        run_fn=run_fn,
                        signal_length=N,
                        batch_size=1,
                        precision="fp32",
                        duration_s=d,
                        warmup_iters=5,
                        idle_duration_s=10.0,
                        lhm=lhm,
                    )
                    results.append(result)

                    # Save individual result
                    save_cpu_result(result, output_dir)

                    # Save running CSV after each result (crash recovery)
                    csv_path = str(Path(output_dir) / "all_cpu_results.csv")
                    save_cpu_results_csv(results, csv_path)

                except Exception as e:
                    print(f"    ERROR: {alg_name} N={N}: {e}")
                    import traceback
                    traceback.print_exc()

    # Final summary
    total_time = time.time() - t_suite_start
    print(f"\n{'=' * 60}")
    print(f"  CPU Benchmark Suite Complete")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: {len(results)} benchmarks")
    print(f"  Saved to: {output_dir}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOMLSignals CPU Benchmark Runner")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only this category")
    parser.add_argument("--algo", type=str, default=None,
                        help="Run only this algorithm")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: N=1024 only, 3s measurement")
    parser.add_argument("--signal-lengths", type=int, nargs="+",
                        default=None, help="Custom signal lengths")
    parser.add_argument("--duration", type=float, default=DURATION_S,
                        help="Measurement duration in seconds")
    args = parser.parse_args()

    categories = [args.category] if args.category else None
    algorithms = [args.algo] if args.algo else None

    if args.quick:
        signal_lengths = [1024]
        duration = 3.0
    else:
        signal_lengths = args.signal_lengths
        duration = args.duration

    run_suite(
        categories=categories,
        algorithms=algorithms,
        signal_lengths=signal_lengths,
        duration_s=duration,
    )
