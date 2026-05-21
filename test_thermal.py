"""Quick smoke test for thermal settling."""
from shared.harness import GPUPowerSampler

s = GPUPowerSampler()
t = s.wait_for_thermal_settle(threshold_c=1.0, window_s=5.0, timeout_s=30.0)
print(f"Settled at {t['settled_temp_c']:.0f}C in {t['wait_time_s']:.0f}s, timed_out={t['timed_out']}")
s.shutdown()
print("Thermal settle OK")
