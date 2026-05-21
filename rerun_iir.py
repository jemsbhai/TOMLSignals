"""Re-run just the IIR benchmarks — no scipy dependency."""
from shared.harness import benchmark_algorithm, calibrate_batch_size, save_result
from algorithms.filters import FILTERS
import torch

# Pre-compute Butterworth order=4, cutoff=0.2 coefficients
# These are the exact outputs of scipy.signal.butter(4, 0.2)
# Verified on local machine.
IIR_B = [0.004824343357716229, 0.019297373430864916, 0.028946060146297373, 0.019297373430864916, 0.004824343357716229]
IIR_A = [1.0, -2.369513007182038, 2.313988414415881, -1.054665405878568, 0.18737949236818502]

def setup_iir_no_scipy(signal_length, batch_size, precision, device, order=4, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    b_t = torch.tensor(IIR_B, device=device, dtype=dtype)
    a_t = torch.tensor(IIR_A, device=device, dtype=dtype)
    return {"x": x, "b": b_t, "a": a_t, "order": order, "_lfilter": None}

_, run, kw = FILTERS["iir_butter4"]
dev = torch.device("cuda")

for N in [1024, 4096, 16384]:
    B = calibrate_batch_size(setup_iir_no_scipy, run, N, "fp32", dev, kw,
                             target_time_ms=1.0, max_batch=2048)
    print(f"IIR N={N} B={B}", flush=True)
    r = benchmark_algorithm("iir_butter4", "filter", setup_iir_no_scipy, run,
                            signal_length=N, batch_size=B, precision="fp32",
                            duration_s=10.0, extra_params=kw)
    print(f"  delta={r.delta_power_w:.1f}W  E/call={r.energy_per_call_j:.4e}  "
          f"t/call={r.time_per_call_us:.1f}us  idle={r.idle_power_w:.1f}W@{r.idle_temp_c:.0f}C")
    save_result(r, "data/results/filter")

print("\nIIR re-run complete.")
