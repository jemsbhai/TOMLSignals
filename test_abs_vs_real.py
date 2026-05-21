"""
Test: torch.abs(X)**2 vs X.real**2 + X.imag**2
Measures FP32 instruction difference to explain periodogram NCU discrepancy.

Run from TOMLSignals directory:
  python test_abs_vs_real.py
"""
import torch
import time

device = "cuda"
N = 4096
B = 1
n_warmup = 50
n_measure = 200

# Create complex FFT output (simulating what periodogram does)
x_real = torch.randn(B, N, device=device, dtype=torch.float32)
X = torch.fft.fft(x_real)

# --- Method A: torch.abs(X) ** 2  (what the code does) ---
# Warmup
for _ in range(n_warmup):
    _ = torch.abs(X) ** 2
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(n_measure):
    Pa = torch.abs(X) ** 2
torch.cuda.synchronize()
t_abs = (time.perf_counter() - start) / n_measure

# --- Method B: X.real**2 + X.imag**2  (mathematical minimum) ---
for _ in range(n_warmup):
    _ = X.real**2 + X.imag**2
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(n_measure):
    Pb = X.real**2 + X.imag**2
torch.cuda.synchronize()
t_direct = (time.perf_counter() - start) / n_measure

# --- Method C: Full periodogram with abs (current implementation) ---
for _ in range(n_warmup):
    _ = torch.abs(torch.fft.fft(x_real)) ** 2 / N
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(n_measure):
    Pc = torch.abs(torch.fft.fft(x_real)) ** 2 / N
torch.cuda.synchronize()
t_full_abs = (time.perf_counter() - start) / n_measure

# --- Method D: Full periodogram with direct (optimized) ---
for _ in range(n_warmup):
    Xd = torch.fft.fft(x_real)
    _ = (Xd.real**2 + Xd.imag**2) / N
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(n_measure):
    Xd = torch.fft.fft(x_real)
    Pd = (Xd.real**2 + Xd.imag**2) / N
torch.cuda.synchronize()
t_full_direct = (time.perf_counter() - start) / n_measure

# Verify numerical equivalence
print("=== Numerical equivalence ===")
print(f"  max |abs(X)**2 - (real**2+imag**2)|: {(Pa - Pb).abs().max().item():.2e}")
print(f"  Allclose: {torch.allclose(Pa, Pb, atol=1e-4)}")
print()

print("=== Timing (power spectrum only, no FFT) ===")
print(f"  torch.abs(X)**2:          {t_abs*1e6:8.1f} us")
print(f"  X.real**2 + X.imag**2:    {t_direct*1e6:8.1f} us")
print(f"  Speedup (direct/abs):     {t_abs/t_direct:8.2f}x")
print()

print("=== Timing (full periodogram: FFT + power + /N) ===")
print(f"  abs-based (current):      {t_full_abs*1e6:8.1f} us")
print(f"  direct (optimized):       {t_full_direct*1e6:8.1f} us")
print(f"  Speedup (direct/abs):     {t_full_abs/t_full_direct:8.2f}x")
print()

print("=== Implication for TO model ===")
print(f"  The TO model counts Re**2 + Im**2 = 3N = {3*N} ops (mathematical minimum).")
print(f"  torch.abs(X)**2 computes sqrt(Re**2+Im**2) then squares -- wasting the sqrt.")
print(f"  NCU shows 53,246 extra FP32 ops beyond FFT = ~{53246/N:.1f}N ops per element.")
print(f"  Mathematical minimum is 4N (3N power + N division).")
print(f"  Implementation overhead factor: ~{53246/(4*N):.1f}x")
