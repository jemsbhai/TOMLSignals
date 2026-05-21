"""
TOMLSignals - Signal Processing Benchmark Harness
===================================================
Runs short-duration SP algorithms in a measurement loop with GPU power monitoring.

Unlike TOMLCloud (60-second sustained workloads), SP algorithms complete in
microseconds. This harness loops them for a configurable duration to accumulate
stable power measurements.

Measurement protocol:
  1. Setup algorithm state on GPU
  2. Warmup (50 iterations to trigger JIT, cache fills)
  3. Thermal settle: wait until GPU temp stabilizes (ΔT < 1°C over 5s)
  4. Measure per-algorithm idle baseline (3s, discard first 1s)
  5. Short cooldown (1s)
  6. Measurement loop: continuous algorithm execution with power sampling
  7. Compute delta power = mean_active - idle_baseline

Author: Muntaser Syed
Date: May 2026
"""

import time
import json
import gc
import csv
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import torch

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    algorithm: str
    category: str
    signal_length: int
    batch_size: int
    precision: str
    iterations: int
    duration_s: float
    idle_power_w: float
    idle_power_std_w: float
    mean_power_w: float
    power_std_w: float
    delta_power_w: float
    total_energy_j: float
    delta_energy_j: float
    energy_per_call_j: float
    mean_temp_c: float
    max_temp_c: float
    mean_clock_mhz: int
    time_per_call_us: float
    power_samples: int
    idle_samples: int
    idle_temp_c: float
    thermal_wait_s: float
    thermal_timed_out: bool
    gpu_name: str
    params: Dict[str, Any] = field(default_factory=dict)


class GPUPowerSampler:
    """Lightweight GPU power sampler for SP benchmarks."""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 50):
        if not HAS_PYNVML:
            raise RuntimeError("pynvml not available")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.interval_s = interval_ms / 1000.0
        self._powers: List[float] = []
        self._temps: List[int] = []
        self._clocks: List[int] = []
        self._sampling = False
        self._thread: Optional[threading.Thread] = None

    def _read(self):
        p = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        t = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        c = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
        return p, t, c

    def measure_idle(self, duration_s: float = 3.0, discard_s: float = 1.0) -> dict:
        """Measure idle GPU power with initial discard period.
        
        Args:
            duration_s: Total measurement time (including discard period).
            discard_s: Initial period to discard (GPU settling time).
            
        Returns:
            Dict with idle_power_w, idle_power_std_w, n_samples.
        """
        all_powers = []
        t0 = time.time()
        while time.time() - t0 < duration_s:
            p, _, _ = self._read()
            elapsed = time.time() - t0
            if elapsed >= discard_s:
                all_powers.append(p)
            time.sleep(self.interval_s)
        
        if not all_powers:
            return {"idle_power_w": 0.0, "idle_power_std_w": 0.0, "n_samples": 0}
        
        return {
            "idle_power_w": float(np.mean(all_powers)),
            "idle_power_std_w": float(np.std(all_powers)),
            "n_samples": len(all_powers),
        }

    def start(self):
        self._powers, self._temps, self._clocks = [], [], []
        self._sampling = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._sampling:
            p, t, c = self._read()
            self._powers.append(p)
            self._temps.append(t)
            self._clocks.append(c)
            time.sleep(self.interval_s)

    def stop(self) -> Dict:
        self._sampling = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._powers:
            return {}
        return {
            "mean_power_w": float(np.mean(self._powers)),
            "power_std_w": float(np.std(self._powers)),
            "mean_temp_c": float(np.mean(self._temps)),
            "max_temp_c": int(np.max(self._temps)),
            "mean_clock_mhz": int(np.mean(self._clocks)),
            "n_samples": len(self._powers),
        }

    def get_gpu_name(self) -> str:
        """Get GPU device name."""
        name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        return name

    def shutdown(self):
        pynvml.nvmlShutdown()

    def wait_for_thermal_settle(
        self,
        threshold_c: float = 1.0,
        window_s: float = 5.0,
        timeout_s: float = 120.0,
        poll_s: float = 1.0,
    ) -> dict:
        """Wait until GPU temperature stabilizes.
        
        Polls GPU temperature every poll_s seconds. Considers temperature
        settled when the max-min spread over the last window_s seconds is
        <= threshold_c. This ensures thermal equilibrium before baseline
        measurement, preventing hot-GPU bias from prior benchmarks.
        
        Args:
            threshold_c: Max temp spread (°C) within window to consider settled.
            window_s: Rolling window duration for stability check.
            timeout_s: Max wait time (seconds). Proceeds with warning if exceeded.
            poll_s: Polling interval.
            
        Returns:
            Dict with settled_temp_c, wait_time_s, timed_out.
        """
        temps = []
        times = []
        t0 = time.time()
        
        while True:
            _, t, _ = self._read()
            now = time.time()
            temps.append(t)
            times.append(now)
            
            # Keep only readings within the window
            cutoff = now - window_s
            while times and times[0] < cutoff:
                times.pop(0)
                temps.pop(0)
            
            # Check if we have a full window and it's stable
            elapsed = now - t0
            if elapsed >= window_s and len(temps) >= 3:
                spread = max(temps) - min(temps)
                if spread <= threshold_c:
                    return {
                        "settled_temp_c": float(temps[-1]),
                        "wait_time_s": elapsed,
                        "timed_out": False,
                    }
            
            # Timeout check
            if elapsed >= timeout_s:
                return {
                    "settled_temp_c": float(temps[-1]),
                    "wait_time_s": elapsed,
                    "timed_out": True,
                }
            
            time.sleep(poll_s)


def calibrate_batch_size(
    setup_fn: Callable,
    run_fn: Callable,
    signal_length: int,
    precision: str,
    device: torch.device,
    extra_params: Dict,
    target_time_ms: float = 1.0,
    max_batch: int = 4096,
) -> int:
    """Find minimum batch_size so each call takes >= target_time_ms.
    
    This ensures the GPU sustains enough load for reliable power measurement.
    NVML samples at ~50ms intervals; we need each run_fn call to keep the GPU
    busy long enough that power draw is representative.
    
    Args:
        target_time_ms: Minimum desired time per call in milliseconds.
        max_batch: Upper bound on batch size (memory safety).
        
    Returns:
        Calibrated batch_size.
    """
    batch_size = 1
    while batch_size <= max_batch:
        try:
            state = setup_fn(
                signal_length=signal_length,
                batch_size=batch_size,
                precision=precision,
                device=device,
                **extra_params,
            )
            # Warmup
            for _ in range(5):
                run_fn(state)
            torch.cuda.synchronize()
            
            # Time 20 iterations
            t0 = time.time()
            for _ in range(20):
                run_fn(state)
            torch.cuda.synchronize()
            t_per_call_ms = (time.time() - t0) / 20 * 1000
            
            # Clean up
            del state
            torch.cuda.empty_cache()
            
            if t_per_call_ms >= target_time_ms:
                return batch_size
            
            batch_size *= 2
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return max(1, batch_size // 2)
            raise
    
    return max_batch


def benchmark_algorithm(
    name: str,
    category: str,
    setup_fn: Callable,
    run_fn: Callable,
    signal_length: int = 4096,
    batch_size: int = 1,
    precision: str = "fp32",
    duration_s: float = 10.0,
    warmup_iters: int = 50,
    extra_params: Optional[Dict] = None,
    gpu_index: int = 0,
) -> BenchmarkResult:
    """
    Benchmark a signal processing algorithm with per-algorithm baseline.

    Measurement protocol:
      1. Setup + warmup
      2. Cooldown (empty cache, sleep cooldown_s)
      3. Measure idle baseline (3s, discard first 1s)
      4. Short cooldown (1s)
      5. Run measurement loop
      6. Compute delta = active - idle

    Args:
        name: Algorithm name.
        category: Category name.
        setup_fn: Setup callable returning state dict.
        run_fn: Run callable taking state dict.
        signal_length: Input signal length N.
        batch_size: Batch size B.
        precision: "fp32" or "fp16".
        duration_s: Measurement duration.
        warmup_iters: Warmup iterations.
        extra_params: Algorithm-specific parameters.
        gpu_index: GPU device index.

    Returns:
        BenchmarkResult with energy and timing data.
    """
    device = torch.device("cuda")

    if extra_params is None:
        extra_params = {}

    # Setup
    state = setup_fn(
        signal_length=signal_length,
        batch_size=batch_size,
        precision=precision,
        device=device,
        **extra_params,
    )

    # Warmup
    for _ in range(warmup_iters):
        run_fn(state)
    torch.cuda.synchronize()

    # === THERMAL SETTLE ===
    # Critical: GPU must return to thermal equilibrium before baseline.
    # Heavy algorithms (transformer at N=16384) can raise temp by 10-20°C;
    # measuring idle power while hot inflates baseline (leakage power scales
    # with temperature) and can cause negative deltas.
    torch.cuda.empty_cache()
    gc.collect()
    sampler = GPUPowerSampler(gpu_index=gpu_index)
    gpu_name = sampler.get_gpu_name()
    thermal = sampler.wait_for_thermal_settle(
        threshold_c=1.0,   # settled = max-min spread < 1°C over 5s window
        window_s=5.0,
        timeout_s=120.0,
    )
    idle_temp = thermal["settled_temp_c"]
    thermal_wait = thermal["wait_time_s"]
    thermal_timed_out = thermal["timed_out"]

    # === MEASURE PER-ALGORITHM IDLE BASELINE ===
    # 3 seconds total, discard first 1s (GPU settling)
    idle_stats = sampler.measure_idle(duration_s=3.0, discard_s=1.0)
    idle_power = idle_stats["idle_power_w"]
    idle_std = idle_stats["idle_power_std_w"]
    idle_samples = idle_stats["n_samples"]

    # Short pause before measurement
    time.sleep(1.0)

    # === MEASUREMENT LOOP ===
    sampler.start()
    iterations = 0
    t_start = time.time()

    while time.time() - t_start < duration_s:
        run_fn(state)
        iterations += 1

    torch.cuda.synchronize()
    t_end = time.time()
    stats = sampler.stop()

    actual_duration = t_end - t_start
    mean_power = stats.get("mean_power_w", 0)
    power_std = stats.get("power_std_w", 0)
    power_samples = stats.get("n_samples", 0)
    delta_power = mean_power - idle_power
    total_energy = mean_power * actual_duration
    delta_energy = delta_power * actual_duration

    result = BenchmarkResult(
        algorithm=name,
        category=category,
        signal_length=signal_length,
        batch_size=batch_size,
        precision=precision,
        iterations=iterations,
        duration_s=actual_duration,
        idle_power_w=idle_power,
        idle_power_std_w=idle_std,
        mean_power_w=mean_power,
        power_std_w=power_std,
        delta_power_w=delta_power,
        total_energy_j=total_energy,
        delta_energy_j=delta_energy,
        energy_per_call_j=delta_energy / iterations if iterations > 0 else 0,
        mean_temp_c=stats.get("mean_temp_c", 0),
        max_temp_c=stats.get("max_temp_c", 0),
        mean_clock_mhz=stats.get("mean_clock_mhz", 0),
        time_per_call_us=(actual_duration / iterations * 1e6) if iterations > 0 else 0,
        power_samples=power_samples,
        idle_samples=idle_samples,
        idle_temp_c=idle_temp,
        thermal_wait_s=thermal_wait,
        thermal_timed_out=thermal_timed_out,
        gpu_name=gpu_name,
        params=extra_params,
    )

    sampler.shutdown()
    return result


def save_result(result: BenchmarkResult, output_dir: str):
    """Save a benchmark result as JSON."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    params_str = "_".join(f"{k}{v}" for k, v in sorted(result.params.items()))
    if params_str:
        params_str = f"_{params_str}"

    fname = f"{result.algorithm}_N{result.signal_length}_B{result.batch_size}_{result.precision}{params_str}.json"

    data = {
        "algorithm": result.algorithm,
        "category": result.category,
        "signal_length": result.signal_length,
        "batch_size": result.batch_size,
        "precision": result.precision,
        "iterations": result.iterations,
        "duration_s": result.duration_s,
        "idle_power_w": result.idle_power_w,
        "idle_power_std_w": result.idle_power_std_w,
        "mean_power_w": result.mean_power_w,
        "power_std_w": result.power_std_w,
        "delta_power_w": result.delta_power_w,
        "total_energy_j": result.total_energy_j,
        "delta_energy_j": result.delta_energy_j,
        "energy_per_call_j": result.energy_per_call_j,
        "mean_temp_c": result.mean_temp_c,
        "max_temp_c": result.max_temp_c,
        "mean_clock_mhz": result.mean_clock_mhz,
        "time_per_call_us": result.time_per_call_us,
        "power_samples": result.power_samples,
        "idle_samples": result.idle_samples,
        "idle_temp_c": result.idle_temp_c,
        "thermal_wait_s": result.thermal_wait_s,
        "thermal_timed_out": result.thermal_timed_out,
        "gpu_name": result.gpu_name,
        "params": result.params,
    }

    with open(p / fname, "w") as f:
        json.dump(data, f, indent=2)

    return p / fname


def save_results_csv(results: List[BenchmarkResult], path: str):
    """Save all results to a single CSV for analysis."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "algorithm", "category", "signal_length", "batch_size", "precision",
        "iterations", "duration_s", "idle_power_w", "idle_power_std_w",
        "mean_power_w", "power_std_w", "delta_power_w",
        "total_energy_j", "delta_energy_j",
        "energy_per_call_j", "mean_temp_c", "max_temp_c",
        "mean_clock_mhz", "time_per_call_us",
        "power_samples", "idle_samples",
        "idle_temp_c", "thermal_wait_s", "thermal_timed_out",
        "gpu_name",
    ]

    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fields}
            writer.writerow(row)

    print(f"Saved {len(results)} results to {p}")
