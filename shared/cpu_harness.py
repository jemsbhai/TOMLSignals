"""
TOMLSignals - CPU Benchmark Harness
=====================================
Runs signal processing algorithms on CPU (numpy/scipy/sklearn) with
power measurement via LibreHardwareMonitor's HTTP API.

Measurement protocol (per algorithm, per signal length):
  1. Wait for CPU thermal settle (±1°C over 5s window)
  2. Wait for low CPU load (<10% over 3s window)
  3. Idle baseline: 10 seconds of CPU power + temp + load + clock polling
     Also monitors GPU power to confirm GPU idle throughout.
  4. Execute benchmark in a tight loop (min 2 seconds active)
  5. Record: ΔP, time, energy, temps, clocks, loads, GPU idle power

Power source: Intel RAPL via LibreHardwareMonitor HTTP API (~217 Hz from Python).
CPU: Intel Core i9-14900HX (8P + 16E cores, 55W base / 157W turbo).

Author: Muntaser Syed
Date: May 2026
"""

import time
import json
import gc
import csv
import threading
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple

import numpy as np


# =========================================================================
# LibreHardwareMonitor HTTP API Sensor Reader
# =========================================================================

class LHMSensorReader:
    """Reads CPU and GPU sensors from LibreHardwareMonitor's HTTP API.

    LHM must be running as Administrator with Remote Web Server enabled.
    Default endpoint: http://localhost:8085/data.json

    Sensor tree structure (from empirical inspection):
      root -> Children[0] (computer) -> Children[i] (hardware)
        -> Children[j] (sensor category: Powers, Temperatures, Clocks, Load)
          -> Children[k] (individual sensor: Text, Value)
    """

    def __init__(self, url: str = "http://localhost:8085/data.json",
                 cpu_index: int = 1, gpu_index: int = 6):
        """
        Args:
            url: LHM web server data endpoint.
            cpu_index: Index of CPU in hardware children list.
                       i9-14900HX is at index 1 on this machine.
            gpu_index: Index of NVIDIA GPU in hardware children list.
                       RTX 4090 Laptop GPU is at index 6 on this machine.
        """
        self.url = url
        self.cpu_index = cpu_index
        self.gpu_index = gpu_index
        self._validate_connection()

    def _validate_connection(self):
        """Verify LHM is reachable and sensor indices are correct."""
        try:
            tree = self._fetch_tree()
            hw_list = tree["Children"][0]["Children"]
            cpu_text = hw_list[self.cpu_index]["Text"]
            gpu_text = hw_list[self.gpu_index]["Text"]
            if "i9" not in cpu_text.lower() and "intel" not in cpu_text.lower():
                raise RuntimeError(
                    f"CPU index {self.cpu_index} points to '{cpu_text}', "
                    f"expected Intel CPU. Hardware list: "
                    f"{[h['Text'] for h in hw_list]}"
                )
            if "4090" not in gpu_text and "nvidia" not in gpu_text.lower():
                raise RuntimeError(
                    f"GPU index {self.gpu_index} points to '{gpu_text}', "
                    f"expected NVIDIA GPU. Hardware list: "
                    f"{[h['Text'] for h in hw_list]}"
                )
            print(f"  LHM connected: CPU='{cpu_text}', GPU='{gpu_text}'")
        except urllib.error.URLError:
            raise RuntimeError(
                "Cannot connect to LHM at {self.url}. "
                "Ensure LibreHardwareMonitor is running as Admin "
                "with Options > Remote Web Server enabled."
            )

    def _fetch_tree(self) -> dict:
        """Fetch the full sensor tree from LHM."""
        with urllib.request.urlopen(self.url) as resp:
            return json.loads(resp.read())

    def _get_sensor_category(self, hw_index: int, category_name: str) -> list:
        """Get all sensors in a category (e.g., 'Powers', 'Temperatures')."""
        tree = self._fetch_tree()
        hw = tree["Children"][0]["Children"][hw_index]
        for cat in hw["Children"]:
            if cat["Text"] == category_name:
                return cat["Children"]
        return []

    def _parse_value(self, value_str: str) -> float:
        """Parse LHM sensor value string like '19.7 W' or '45.2 °C'."""
        if not value_str or value_str.strip() == "":
            return 0.0
        # Strip units: W, °C, MHz, %, etc.
        parts = value_str.strip().split()
        try:
            return float(parts[0].replace(",", ""))
        except (ValueError, IndexError):
            return 0.0

    def read_cpu_power(self) -> Dict[str, float]:
        """Read CPU power sensors.

        Returns:
            Dict with 'package_w', 'cores_w'.
        """
        sensors = self._get_sensor_category(self.cpu_index, "Powers")
        result = {"package_w": 0.0, "cores_w": 0.0}
        for s in sensors:
            val = self._parse_value(s.get("Value", ""))
            if "Package" in s["Text"]:
                result["package_w"] = val
            elif "Cores" in s["Text"]:
                result["cores_w"] = val
        return result

    def read_cpu_temp(self) -> Dict[str, float]:
        """Read CPU temperature sensors.

        Returns:
            Dict with 'package_c' (CPU Package temperature).
        """
        sensors = self._get_sensor_category(self.cpu_index, "Temperatures")
        result = {"package_c": 0.0}
        for s in sensors:
            if "Package" in s["Text"] or "CPU Package" in s["Text"]:
                result["package_c"] = self._parse_value(s.get("Value", ""))
                break
        # Fallback: first temperature sensor
        if result["package_c"] == 0.0 and sensors:
            result["package_c"] = self._parse_value(sensors[0].get("Value", ""))
        return result

    def read_cpu_clock(self) -> Dict[str, float]:
        """Read CPU clock sensors.

        Returns:
            Dict with 'max_clock_mhz' (highest active core clock).
        """
        sensors = self._get_sensor_category(self.cpu_index, "Clocks")
        clocks = []
        for s in sensors:
            if "Core" in s["Text"] and "#" in s["Text"]:
                val = self._parse_value(s.get("Value", ""))
                if val > 0:
                    clocks.append(val)
        return {
            "max_clock_mhz": max(clocks) if clocks else 0.0,
            "mean_clock_mhz": np.mean(clocks) if clocks else 0.0,
        }

    def read_cpu_load(self) -> Dict[str, float]:
        """Read CPU load sensors.

        Returns:
            Dict with 'total_pct' (CPU Total load percentage).
        """
        sensors = self._get_sensor_category(self.cpu_index, "Load")
        result = {"total_pct": 0.0}
        for s in sensors:
            if "Total" in s["Text"]:
                result["total_pct"] = self._parse_value(s.get("Value", ""))
                break
        return result

    def read_gpu_power(self) -> Dict[str, float]:
        """Read GPU power to verify it's idle during CPU benchmarks."""
        sensors = self._get_sensor_category(self.gpu_index, "Powers")
        result = {"gpu_power_w": 0.0}
        for s in sensors:
            # LHM typically reports GPU Power or Board Power
            val = self._parse_value(s.get("Value", ""))
            if val > 0:
                result["gpu_power_w"] = val
                break
        return result

    def read_all(self) -> Dict[str, float]:
        """Read all relevant sensors in a single fetch.

        More efficient than individual calls since it makes one HTTP request.

        Returns:
            Dict with cpu_package_w, cpu_cores_w, cpu_temp_c,
            cpu_max_clock_mhz, cpu_mean_clock_mhz, cpu_load_pct, gpu_power_w.
        """
        tree = self._fetch_tree()
        hw_list = tree["Children"][0]["Children"]
        cpu_hw = hw_list[self.cpu_index]
        gpu_hw = hw_list[self.gpu_index]

        result = {
            "cpu_package_w": 0.0,
            "cpu_cores_w": 0.0,
            "cpu_temp_c": 0.0,
            "cpu_max_clock_mhz": 0.0,
            "cpu_mean_clock_mhz": 0.0,
            "cpu_load_pct": 0.0,
            "gpu_power_w": 0.0,
            "gpu_load_pct": 0.0,
        }

        # Parse CPU sensors
        for cat in cpu_hw["Children"]:
            if cat["Text"] == "Powers":
                for s in cat["Children"]:
                    val = self._parse_value(s.get("Value", ""))
                    if "Package" in s["Text"]:
                        result["cpu_package_w"] = val
                    elif "Cores" in s["Text"]:
                        result["cpu_cores_w"] = val
            elif cat["Text"] == "Temperatures":
                for s in cat["Children"]:
                    if "Package" in s["Text"] or "CPU Package" in s["Text"]:
                        result["cpu_temp_c"] = self._parse_value(s.get("Value", ""))
                        break
                if result["cpu_temp_c"] == 0.0 and cat["Children"]:
                    result["cpu_temp_c"] = self._parse_value(
                        cat["Children"][0].get("Value", ""))
            elif cat["Text"] == "Clocks":
                clocks = []
                for s in cat["Children"]:
                    if "Core" in s["Text"] and "#" in s["Text"]:
                        val = self._parse_value(s.get("Value", ""))
                        if val > 0:
                            clocks.append(val)
                if clocks:
                    result["cpu_max_clock_mhz"] = max(clocks)
                    result["cpu_mean_clock_mhz"] = np.mean(clocks)
            elif cat["Text"] == "Load":
                for s in cat["Children"]:
                    if "Total" in s["Text"]:
                        result["cpu_load_pct"] = self._parse_value(
                            s.get("Value", ""))
                        break

        # Parse GPU sensors
        for cat in gpu_hw["Children"]:
            if cat["Text"] == "Powers":
                for s in cat["Children"]:
                    val = self._parse_value(s.get("Value", ""))
                    if val > 0:
                        result["gpu_power_w"] = val
                        break
            elif cat["Text"] == "Load":
                for s in cat["Children"]:
                    if s["Text"] == "GPU Core":
                        result["gpu_load_pct"] = self._parse_value(
                            s.get("Value", ""))
                        break

        return result


# =========================================================================
# CPU Power Sampler (background thread)
# =========================================================================

class CPUPowerSampler:
    """Polls CPU and GPU power in a background thread during benchmarks.

    Uses LHMSensorReader for all sensor access. Sampling rate is limited
    by HTTP API latency (~4.6ms per poll = ~217 Hz).
    """

    def __init__(self, lhm: LHMSensorReader, interval_ms: int = 10):
        """
        Args:
            lhm: Initialized LHMSensorReader.
            interval_ms: Target interval between polls (actual rate limited
                        by HTTP latency, ~5ms minimum).
        """
        self.lhm = lhm
        self.interval_s = interval_ms / 1000.0
        self._samples: List[Dict[str, float]] = []
        self._timestamps: List[float] = []
        self._sampling = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start background power sampling."""
        self._samples = []
        self._timestamps = []
        self._sampling = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._sampling:
            try:
                reading = self.lhm.read_all()
                self._samples.append(reading)
                self._timestamps.append(time.perf_counter())
            except Exception:
                pass  # Skip failed reads, don't crash the sampler
            time.sleep(self.interval_s)

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return summary statistics.

        Returns:
            Dict with mean/std for all sensors over the sampling window.
        """
        self._sampling = False
        if self._thread:
            self._thread.join(timeout=2.0)

        if not self._samples:
            return {}

        cpu_pkg = [s["cpu_package_w"] for s in self._samples]
        cpu_cores = [s["cpu_cores_w"] for s in self._samples]
        cpu_temp = [s["cpu_temp_c"] for s in self._samples]
        cpu_max_clk = [s["cpu_max_clock_mhz"] for s in self._samples]
        cpu_mean_clk = [s["cpu_mean_clock_mhz"] for s in self._samples]
        cpu_load = [s["cpu_load_pct"] for s in self._samples]
        gpu_power = [s["gpu_power_w"] for s in self._samples]
        gpu_load = [s["gpu_load_pct"] for s in self._samples]

        return {
            "cpu_package_w_mean": float(np.mean(cpu_pkg)),
            "cpu_package_w_std": float(np.std(cpu_pkg)),
            "cpu_cores_w_mean": float(np.mean(cpu_cores)),
            "cpu_temp_c_mean": float(np.mean(cpu_temp)),
            "cpu_temp_c_max": float(np.max(cpu_temp)),
            "cpu_max_clock_mhz_mean": float(np.mean(cpu_max_clk)),
            "cpu_mean_clock_mhz_mean": float(np.mean(cpu_mean_clk)),
            "cpu_load_pct_mean": float(np.mean(cpu_load)),
            "cpu_load_pct_max": float(np.max(cpu_load)),
            "gpu_power_w_mean": float(np.mean(gpu_power)),
            "gpu_power_w_max": float(np.max(gpu_power)),
            "gpu_load_pct_mean": float(np.mean(gpu_load)),
            "gpu_load_pct_max": float(np.max(gpu_load)),
            "n_samples": len(self._samples),
        }


# =========================================================================
# Thermal + Load Settling
# =========================================================================

def wait_for_thermal_settle(
    lhm: LHMSensorReader,
    threshold_c: float = 1.0,
    window_s: float = 5.0,
    poll_s: float = 1.0,
) -> Dict[str, Any]:
    """Wait until CPU package temperature stabilizes.

    Polls CPU temperature every poll_s seconds. Temperature is considered
    settled when the max-min spread over the last window_s seconds is
    <= threshold_c.

    No timeout — we wait as long as needed. The i9-14900HX can spike to
    100°C under turbo load and takes significant time to cool.

    Args:
        lhm: LHMSensorReader instance.
        threshold_c: Max temp spread (°C) within window to consider settled.
        window_s: Rolling window duration for stability check.
        poll_s: Polling interval.

    Returns:
        Dict with settled_temp_c, wait_time_s.
    """
    temps = []
    times = []
    t0 = time.time()

    while True:
        reading = lhm.read_cpu_temp()
        temp = reading["package_c"]
        now = time.time()
        temps.append(temp)
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
                }

        # Progress feedback every 15 seconds
        if elapsed > 0 and int(elapsed) % 15 == 0 and int(elapsed) > 0:
            spread = max(temps) - min(temps) if temps else 0
            current = temps[-1] if temps else 0
            print(f"    Thermal settling: {current:.1f}°C "
                  f"(spread {spread:.1f}°C, target ≤{threshold_c}°C) "
                  f"[{elapsed:.0f}s]")

        time.sleep(poll_s)


def wait_for_low_load(
    lhm: LHMSensorReader,
    threshold_pct: float = 10.0,
    window_s: float = 3.0,
    poll_s: float = 1.0,
) -> Dict[str, Any]:
    """Wait until CPU load drops below threshold.

    Waits for CPU Total load to stay below threshold_pct for an entire
    window_s period. This ensures background processes (Windows Update,
    antivirus, indexing) are not contaminating measurements.

    No timeout — we wait as long as needed.

    Args:
        lhm: LHMSensorReader instance.
        threshold_pct: Max CPU load percentage.
        window_s: Window over which load must remain low.
        poll_s: Polling interval.

    Returns:
        Dict with settled_load_pct, wait_time_s.
    """
    loads = []
    times = []
    t0 = time.time()

    while True:
        reading = lhm.read_cpu_load()
        load = reading["total_pct"]
        now = time.time()
        loads.append(load)
        times.append(now)

        # Keep only readings within the window
        cutoff = now - window_s
        while times and times[0] < cutoff:
            times.pop(0)
            loads.pop(0)

        # Check if all loads in window are below threshold
        elapsed = now - t0
        if elapsed >= window_s and len(loads) >= 2:
            if max(loads) <= threshold_pct:
                return {
                    "settled_load_pct": float(loads[-1]),
                    "wait_time_s": elapsed,
                }

        # Progress feedback every 15 seconds
        if elapsed > 0 and int(elapsed) % 15 == 0 and int(elapsed) > 0:
            current = loads[-1] if loads else 0
            peak = max(loads) if loads else 0
            print(f"    Load settling: {current:.1f}% "
                  f"(peak {peak:.1f}%, target ≤{threshold_pct}%) "
                  f"[{elapsed:.0f}s]")

        time.sleep(poll_s)


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class CPUBenchmarkResult:
    """Result from a single CPU benchmark run."""
    algorithm: str
    category: str
    signal_length: int
    batch_size: int
    precision: str
    iterations: int
    duration_s: float
    # CPU power
    idle_power_w: float
    idle_power_std_w: float
    mean_power_w: float
    power_std_w: float
    delta_power_w: float
    # Energy
    total_energy_j: float
    delta_energy_j: float
    energy_per_call_j: float
    # CPU state
    mean_temp_c: float
    max_temp_c: float
    idle_temp_c: float
    mean_clock_mhz: float
    # Timing
    time_per_call_us: float
    # Load
    idle_load_pct: float
    active_load_pct: float
    # GPU verification
    gpu_idle_power_w: float
    gpu_active_power_w: float
    # Sampling
    power_samples: int
    idle_samples: int
    # Settling
    thermal_wait_s: float
    load_wait_s: float
    # Metadata
    cpu_name: str = "Intel Core i9-14900HX"
    params: Dict[str, Any] = field(default_factory=dict)


# =========================================================================
# CPU Benchmark Function
# =========================================================================

def benchmark_cpu_algorithm(
    name: str,
    category: str,
    setup_fn: Callable,
    run_fn: Callable,
    signal_length: int = 4096,
    batch_size: int = 1,
    precision: str = "fp32",
    duration_s: float = 5.0,
    warmup_iters: int = 10,
    idle_duration_s: float = 10.0,
    extra_params: Optional[Dict] = None,
    lhm: Optional[LHMSensorReader] = None,
) -> CPUBenchmarkResult:
    """
    Benchmark a signal processing algorithm on CPU with power measurement.

    Measurement protocol:
      1. Setup + warmup
      2. Wait for CPU thermal settle (±1°C over 5s)
      3. Wait for low CPU load (<10% over 3s)
      4. Idle baseline: 10s of CPU + GPU power polling
      5. Short pause (1s)
      6. Measurement loop: tight execution loop with power sampling
      7. Compute ΔP = active_power − idle_power, E = ΔP × time

    Args:
        name: Algorithm name (e.g., 'fft', 'lms').
        category: Category name (e.g., 'transform', 'adaptive').
        setup_fn: Returns state dict for the algorithm.
        run_fn: Executes algorithm given state dict. Must be CPU-only.
        signal_length: Input signal length N.
        batch_size: Batch size B (for consistency; most CPU algos use B=1).
        precision: 'fp32' or 'fp64'.
        duration_s: Active measurement duration in seconds.
        warmup_iters: Number of warmup iterations.
        idle_duration_s: Idle baseline measurement duration in seconds.
        extra_params: Algorithm-specific parameters.
        lhm: LHMSensorReader instance (created if None).

    Returns:
        CPUBenchmarkResult with energy and timing data.
    """
    if extra_params is None:
        extra_params = {}

    if lhm is None:
        lhm = LHMSensorReader()

    # === SETUP ===
    state = setup_fn(
        signal_length=signal_length,
        batch_size=batch_size,
        precision=precision,
        **extra_params,
    )

    # === WARMUP ===
    for _ in range(warmup_iters):
        run_fn(state)

    # Force garbage collection before settling
    gc.collect()

    # === THERMAL SETTLE ===
    print(f"    Waiting for thermal settle...")
    thermal = wait_for_thermal_settle(lhm, threshold_c=1.0, window_s=5.0)
    idle_temp = thermal["settled_temp_c"]
    thermal_wait = thermal["wait_time_s"]
    print(f"    Settled at {idle_temp:.1f}°C ({thermal_wait:.1f}s)")

    # === LOAD SETTLE ===
    print(f"    Waiting for low CPU load...")
    load_settle = wait_for_low_load(lhm, threshold_pct=10.0, window_s=3.0)
    load_wait = load_settle["wait_time_s"]
    print(f"    Load settled at {load_settle['settled_load_pct']:.1f}% ({load_wait:.1f}s)")

    # === IDLE BASELINE ===
    print(f"    Measuring idle baseline ({idle_duration_s}s)...")
    idle_sampler = CPUPowerSampler(lhm, interval_ms=10)
    idle_sampler.start()
    time.sleep(idle_duration_s)
    idle_stats = idle_sampler.stop()

    idle_power = idle_stats.get("cpu_package_w_mean", 0.0)
    idle_power_std = idle_stats.get("cpu_package_w_std", 0.0)
    idle_load = idle_stats.get("cpu_load_pct_mean", 0.0)
    idle_gpu_power = idle_stats.get("gpu_power_w_mean", 0.0)
    idle_gpu_load = idle_stats.get("gpu_load_pct_mean", 0.0)
    idle_samples = idle_stats.get("n_samples", 0)
    print(f"    Idle: {idle_power:.1f}W ± {idle_power_std:.1f}W, "
          f"load={idle_load:.1f}%, GPU_load={idle_gpu_load:.1f}%, "
          f"samples={idle_samples}")

    # Short pause before measurement
    time.sleep(1.0)

    # === MEASUREMENT LOOP ===
    print(f"    Measuring ({duration_s}s)...")
    active_sampler = CPUPowerSampler(lhm, interval_ms=10)
    active_sampler.start()

    iterations = 0
    t_start = time.perf_counter()

    while time.perf_counter() - t_start < duration_s:
        run_fn(state)
        iterations += 1

    t_end = time.perf_counter()
    active_stats = active_sampler.stop()

    # === COMPUTE RESULTS ===
    actual_duration = t_end - t_start
    mean_power = active_stats.get("cpu_package_w_mean", 0.0)
    power_std = active_stats.get("cpu_package_w_std", 0.0)
    delta_power = mean_power - idle_power
    total_energy = mean_power * actual_duration
    delta_energy = delta_power * actual_duration
    active_load = active_stats.get("cpu_load_pct_mean", 0.0)
    active_gpu_power = active_stats.get("gpu_power_w_mean", 0.0)
    active_gpu_load = active_stats.get("gpu_load_pct_mean", 0.0)
    active_samples = active_stats.get("n_samples", 0)

    time_per_call = actual_duration / iterations * 1e6 if iterations > 0 else 0
    energy_per_call = delta_energy / iterations if iterations > 0 else 0

    print(f"    Active: {mean_power:.1f}W (Δ{delta_power:.1f}W), "
          f"{iterations} iters in {actual_duration:.2f}s, "
          f"{time_per_call:.1f} µs/call, "
          f"load={active_load:.1f}%, GPU_load={active_gpu_load:.1f}%, "
          f"samples={active_samples}")
    print(f"    Energy: {energy_per_call:.6e} J/call")

    result = CPUBenchmarkResult(
        algorithm=name,
        category=category,
        signal_length=signal_length,
        batch_size=batch_size,
        precision=precision,
        iterations=iterations,
        duration_s=actual_duration,
        idle_power_w=idle_power,
        idle_power_std_w=idle_power_std,
        mean_power_w=mean_power,
        power_std_w=power_std,
        delta_power_w=delta_power,
        total_energy_j=total_energy,
        delta_energy_j=delta_energy,
        energy_per_call_j=energy_per_call,
        mean_temp_c=active_stats.get("cpu_temp_c_mean", 0.0),
        max_temp_c=active_stats.get("cpu_temp_c_max", 0.0),
        idle_temp_c=idle_temp,
        mean_clock_mhz=active_stats.get("cpu_mean_clock_mhz_mean", 0.0),
        time_per_call_us=time_per_call,
        idle_load_pct=idle_load,
        active_load_pct=active_load,
        gpu_idle_power_w=idle_gpu_power,
        gpu_active_power_w=active_gpu_power,
        power_samples=active_samples,
        idle_samples=idle_samples,
        thermal_wait_s=thermal_wait,
        load_wait_s=load_wait,
        params=extra_params,
    )

    return result


# =========================================================================
# I/O
# =========================================================================

def save_cpu_result(result: CPUBenchmarkResult, output_dir: str) -> Path:
    """Save a CPU benchmark result as JSON."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    params_str = "_".join(f"{k}{v}" for k, v in sorted(result.params.items()))
    if params_str:
        params_str = f"_{params_str}"

    fname = (f"{result.algorithm}_N{result.signal_length}"
             f"_B{result.batch_size}_{result.precision}{params_str}.json")

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
        "idle_temp_c": result.idle_temp_c,
        "mean_clock_mhz": result.mean_clock_mhz,
        "time_per_call_us": result.time_per_call_us,
        "idle_load_pct": result.idle_load_pct,
        "active_load_pct": result.active_load_pct,
        "gpu_idle_power_w": result.gpu_idle_power_w,
        "gpu_active_power_w": result.gpu_active_power_w,
        "power_samples": result.power_samples,
        "idle_samples": result.idle_samples,
        "thermal_wait_s": result.thermal_wait_s,
        "load_wait_s": result.load_wait_s,
        "cpu_name": result.cpu_name,
        "params": result.params,
    }

    filepath = p / fname
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def save_cpu_results_csv(results: List[CPUBenchmarkResult], path: str):
    """Save all CPU results to a single CSV for analysis."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "algorithm", "category", "signal_length", "batch_size", "precision",
        "iterations", "duration_s",
        "idle_power_w", "idle_power_std_w",
        "mean_power_w", "power_std_w", "delta_power_w",
        "total_energy_j", "delta_energy_j", "energy_per_call_j",
        "mean_temp_c", "max_temp_c", "idle_temp_c", "mean_clock_mhz",
        "time_per_call_us",
        "idle_load_pct", "active_load_pct",
        "gpu_idle_power_w", "gpu_active_power_w",
        "power_samples", "idle_samples",
        "thermal_wait_s", "load_wait_s",
        "cpu_name",
    ]

    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fields}
            writer.writerow(row)

    print(f"Saved {len(results)} CPU results to {p}")


# =========================================================================
# QUICK VALIDATION
# =========================================================================

if __name__ == "__main__":
    print("CPU Harness Validation")
    print("=" * 50)

    lhm = LHMSensorReader()

    # Read all sensors
    print("\nCurrent sensor readings:")
    reading = lhm.read_all()
    for key, val in sorted(reading.items()):
        print(f"  {key}: {val:.1f}")

    # Test thermal settle (should be near-instant at idle)
    print("\nTesting thermal settle...")
    thermal = wait_for_thermal_settle(lhm, threshold_c=1.0, window_s=5.0)
    print(f"  Settled at {thermal['settled_temp_c']:.1f}°C "
          f"in {thermal['wait_time_s']:.1f}s")

    # Test load settle
    print("\nTesting load settle...")
    load = wait_for_low_load(lhm, threshold_pct=10.0, window_s=3.0)
    print(f"  Load settled at {load['settled_load_pct']:.1f}% "
          f"in {load['wait_time_s']:.1f}s")

    # Test power sampling (5 seconds)
    print("\nTesting power sampling (5s)...")
    sampler = CPUPowerSampler(lhm, interval_ms=10)
    sampler.start()
    time.sleep(5.0)
    stats = sampler.stop()
    print(f"  CPU Package: {stats['cpu_package_w_mean']:.1f}W "
          f"± {stats['cpu_package_w_std']:.1f}W")
    print(f"  Samples: {stats['n_samples']} "
          f"({stats['n_samples']/5:.0f} Hz)")
    print(f"  GPU power: {stats['gpu_power_w_mean']:.1f}W ")
    print(f"  GPU load: {stats['gpu_load_pct_mean']:.1f}% "
          f"(max {stats['gpu_load_pct_max']:.1f}%)")

    print("\nValidation complete.")
