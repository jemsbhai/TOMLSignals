"""
Diagnose DST and Welch NCU discrepancies.

DST: non-power-of-2 FFT size issue
Welch: non-FFT term accounting
"""
import torch
import time
import math

device = "cuda"
N = 4096
n_warmup = 50
n_measure = 200

def bench(fn, label):
    """Benchmark a function, return mean time in microseconds."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_measure):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_measure * 1e6


print("=" * 70)
print("PART 1: DST — non-power-of-2 FFT cost")
print("=" * 70)

# Compare FFT times for different sizes around 8194
sizes = [4096, 8192, 8194, 16384]
print(f"\n{'Size':>8s} {'Factorization':>30s} {'Time (us)':>12s} {'Relative':>10s}")
print("-" * 65)

times = {}
for sz in sizes:
    x = torch.randn(1, sz, device=device, dtype=torch.float32)
    t = bench(lambda: torch.fft.fft(x), f"FFT({sz})")
    times[sz] = t
    # Quick factorization
    n = sz
    factors = []
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241]:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        factors.append(n)
    fact_str = " × ".join(str(f) for f in factors)
    rel = t / times[4096]
    print(f"{sz:>8d} {fact_str:>30s} {t:>12.1f} {rel:>10.2f}x")

print(f"\n  Key finding: FFT(8194) vs FFT(8192) timing ratio = {times[8194]/times[8192]:.2f}x")
print(f"  FFT(8194) vs FFT(16384) timing ratio = {times[8194]/times[16384]:.2f}x")

print()
print("=" * 70)
print("PART 2: DST — full pipeline breakdown")
print("=" * 70)

x = torch.randn(1, N, device=device, dtype=torch.float32)

# Time the full DST
def full_dst():
    L = 2 * (N + 1)
    z = torch.zeros(1, L, device=x.device, dtype=torch.float32)
    z[:, 1:N+1] = x
    z[:, N+2:] = -x.flip(dims=[1])
    Z = torch.fft.fft(z)
    X = -Z[:, 1:N+1].imag

t_dst = bench(full_dst, "full DST")

# Time just the FFT(8194) portion
z_pre = torch.randn(1, 8194, device=device, dtype=torch.float32)
t_fft_8194 = bench(lambda: torch.fft.fft(z_pre), "FFT(8194)")

# Time FFT(8192) for comparison  
z_pow2 = torch.randn(1, 8192, device=device, dtype=torch.float32)
t_fft_8192 = bench(lambda: torch.fft.fft(z_pow2), "FFT(8192)")

# Time just the setup (copy + negate) 
def dst_setup_only():
    L = 2 * (N + 1)
    z = torch.zeros(1, L, device=x.device, dtype=torch.float32)
    z[:, 1:N+1] = x
    z[:, N+2:] = -x.flip(dims=[1])

t_setup = bench(dst_setup_only, "DST setup")

print(f"\n  Full DST:           {t_dst:8.1f} us")
print(f"  Setup (copy+negate):{t_setup:8.1f} us")
print(f"  FFT(8194):          {t_fft_8194:8.1f} us")
print(f"  FFT(8192):          {t_fft_8192:8.1f} us")
print(f"\n  FFT dominates: {t_fft_8194/t_dst*100:.0f}% of DST time")
print(f"  Bad FFT size penalty: FFT(8194)/FFT(8192) = {t_fft_8194/t_fft_8192:.2f}x")

# What if DST used padded power-of-2?
def dst_padded():
    L = 2 * (N + 1)
    L_pad = 1
    while L_pad < L:
        L_pad *= 2  # next power of 2 = 16384
    z = torch.zeros(1, L_pad, device=x.device, dtype=torch.float32)
    z[:, 1:N+1] = x
    z[:, N+2:L] = -x.flip(dims=[1])
    Z = torch.fft.fft(z)
    X = -Z[:, 1:N+1].imag

t_dst_padded = bench(dst_padded, "DST padded")
print(f"\n  DST with padded FFT(16384): {t_dst_padded:8.1f} us")
print(f"  Speedup vs current:         {t_dst/t_dst_padded:.2f}x")

print()
print("=" * 70)
print("PART 3: Welch — operation breakdown")
print("=" * 70)

W = 256
hop = 128
K = (N - W) // hop + 1  # 31 segments

x_welch = torch.randn(1, N, device=device, dtype=torch.float32)
window = torch.hann_window(W, device=device, dtype=torch.float32)

# Full welch
def full_welch():
    segments = x_welch.unfold(1, W, hop)
    segments_w = segments * window
    X = torch.fft.fft(segments_w)
    Pxx = torch.abs(X) ** 2
    Pxx_avg = Pxx.mean(dim=1)

t_welch = bench(full_welch, "full welch")

# Component times
segments_pre = x_welch.unfold(1, W, hop)
t_unfold = bench(lambda: x_welch.unfold(1, W, hop), "unfold")

segments_w_pre = segments_pre * window
t_window = bench(lambda: segments_pre * window, "window")

X_pre = torch.fft.fft(segments_w_pre)
t_fft = bench(lambda: torch.fft.fft(segments_w_pre), "batched FFT")

t_power = bench(lambda: torch.abs(X_pre) ** 2, "abs**2")
t_power_direct = bench(lambda: X_pre.real**2 + X_pre.imag**2, "real²+imag²")

Pxx_pre = torch.abs(X_pre) ** 2
t_mean = bench(lambda: Pxx_pre.mean(dim=1), "mean")

print(f"\n  Full Welch:                 {t_welch:8.1f} us")
print(f"  Components:")
print(f"    unfold (memory only):     {t_unfold:8.1f} us")
print(f"    window multiply:          {t_window:8.1f} us")
print(f"    batched FFT (K={K}×W={W}):{t_fft:8.1f} us")
print(f"    abs(X)**2:                {t_power:8.1f} us")
print(f"    real²+imag² (direct):     {t_power_direct:8.1f} us")
print(f"    mean over segments:       {t_mean:8.1f} us")
print(f"\n  K={K} segments, W={W}, total data points = {K*W}")

# TO model comparison
fft_ops_W = 4 * W * math.log2(W) - 6 * W + 8
print(f"\n  Split-radix FFT(256) ops:   {fft_ops_W:.0f}")
print(f"  Total FFT ops (K×above):    {K * fft_ops_W:.0f}")
print(f"  Window ops (K×W):           {K * W}")
print(f"  Power ops (3×K×W):          {3 * K * W}")
print(f"  Mean ops (K×W + W):         {K * W + W}")
print(f"  Total predicted:            {K * fft_ops_W + K*W + 3*K*W + K*W + W:.0f}")
print(f"  NCU actual:                 341,601")
print(f"  Ratio:                      {341601 / (K * fft_ops_W + K*W + 3*K*W + K*W + W):.3f}")
