"""
TOMLSignals - Quick Test
=========================
Tests the benchmark harness with FFT and FIR to verify power monitoring works.
Run: python test_harness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.harness import benchmark_algorithm, save_result
from algorithms.transforms import TRANSFORMS
from algorithms.filters import FILTERS


def main():
    print("=" * 60)
    print("TOMLSignals - Harness Test")
    print("=" * 60)

    # Test 1: FFT
    print("\n--- Test 1: FFT (N=4096, B=1, 5s) ---")
    setup_fn, run_fn, defaults = TRANSFORMS["fft"]
    result = benchmark_algorithm(
        name="fft", category="transform",
        setup_fn=setup_fn, run_fn=run_fn,
        signal_length=4096, batch_size=1,
        duration_s=5.0, extra_params=defaults,
    )
    print(f"  Iterations: {result.iterations}")
    print(f"  Power: {result.delta_power_w:.1f}W")
    print(f"  Energy/call: {result.energy_per_call_j:.6e} J")
    print(f"  Time/call: {result.time_per_call_us:.1f} us")
    print(f"  Temp: {result.mean_temp_c:.0f}C")
    path = save_result(result, "data/test")
    print(f"  Saved: {path}")

    # Test 2: FIR Direct
    print("\n--- Test 2: FIR Direct (N=4096, M=64, B=1, 5s) ---")
    setup_fn, run_fn, defaults = FILTERS["fir_direct"]
    result = benchmark_algorithm(
        name="fir_direct", category="filter",
        setup_fn=setup_fn, run_fn=run_fn,
        signal_length=4096, batch_size=1,
        duration_s=5.0, extra_params=defaults,
    )
    print(f"  Iterations: {result.iterations}")
    print(f"  Power: {result.delta_power_w:.1f}W")
    print(f"  Energy/call: {result.energy_per_call_j:.6e} J")
    print(f"  Time/call: {result.time_per_call_us:.1f} us")

    # Test 3: Wiener (has divisions)
    print("\n--- Test 3: Wiener Filter (N=4096, B=1, 5s) ---")
    setup_fn, run_fn, defaults = FILTERS["wiener"]
    result = benchmark_algorithm(
        name="wiener", category="filter",
        setup_fn=setup_fn, run_fn=run_fn,
        signal_length=4096, batch_size=1,
        duration_s=5.0, extra_params=defaults,
    )
    print(f"  Iterations: {result.iterations}")
    print(f"  Power: {result.delta_power_w:.1f}W")
    print(f"  Energy/call: {result.energy_per_call_j:.6e} J")
    print(f"  Time/call: {result.time_per_call_us:.1f} us")

    print("\n" + "=" * 60)
    print("Harness test complete. If you see power readings above,")
    print("the setup is working. Ready to run the full suite.")
    print("=" * 60)


if __name__ == "__main__":
    main()
