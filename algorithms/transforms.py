"""
TOMLSignals - Transform Algorithms (Category 1)
=================================================
FFT, Direct DFT, DCT-II, DST-II, DWT, STFT, Hilbert Transform

Canonical implementations:
  - FFT: torch.fft.fft (canonical)
  - Direct DFT: explicit DFT matrix multiply (reference O(N^2) implementation)
  - DCT-II: Makhoul's algorithm — reorder + FFT + complex twiddle (canonical FFT-based DCT)
  - DST-II: Antisymmetric extension + FFT (standard FFT-based DST)
  - DWT: pywt.Wavelet for filter coefficients + torch.nn.functional.conv1d (canonical GPU DWT)
  - STFT: torch.stft (canonical)
  - Hilbert: torch.fft.fft + analytic signal mask (standard)
"""

import torch
import numpy as np


# ---- 1. FFT (Cooley-Tukey via torch) ----

def setup_fft(signal_length, batch_size, precision, device, **kw):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    return {"x": x}

def run_fft(state):
    torch.fft.fft(state["x"])
    torch.cuda.synchronize()


# ---- 2. Direct DFT (matrix multiply) ----

def setup_direct_dft(signal_length, batch_size, precision, device, **kw):
    dtype = torch.complex64 if precision == "fp32" else torch.complex64
    N = signal_length
    n = torch.arange(N, device=device, dtype=torch.float32)
    k = n.unsqueeze(1)
    W = torch.exp(-2j * np.pi * k * n / N).to(dtype)  # DFT matrix
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32).to(dtype)
    return {"x": x, "W": W}

def run_direct_dft(state):
    torch.matmul(state["x"], state["W"])
    torch.cuda.synchronize()


# ---- 3. DCT-II (Makhoul's FFT-based algorithm) ----
#
# DCT-II: X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi * k * (2n+1) / (2N))
#
# Makhoul (1980) algorithm:
#   1. Reorder: v[m] = x[2m] for m=0..N/2-1, v[N-1-m] = x[2m+1] for m=0..N/2-1
#      (equivalently: even-indexed elements, then odd-indexed reversed)
#   2. V = FFT(v)
#   3. X[k] = 2 * Re(V[k] * exp(-j * pi * k / (2N)))
#
# Operations: N-point FFT + N complex multiplies + N real extractions

def setup_dct(signal_length, batch_size, precision, device, **kw):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    N = signal_length
    # Precompute complex twiddle factors: exp(-j * pi * k / (2N))
    k = torch.arange(N, device=device, dtype=torch.float32)
    twiddle = torch.exp(-1j * torch.pi * k / (2 * N))  # complex64, shape (N,)
    return {"x": x, "N": N, "twiddle": twiddle}

def run_dct(state):
    x = state["x"]
    N = state["N"]
    twiddle = state["twiddle"]
    # Step 1: Makhoul reordering — even indices, then reversed odd indices
    v = torch.cat([x[:, ::2], x[:, 1::2].flip(dims=[1])], dim=1)
    # Step 2: FFT
    V = torch.fft.fft(v.float())
    # Step 3: Complex twiddle multiply, then extract real part
    X = 2.0 * (V * twiddle).real
    torch.cuda.synchronize()


# ---- 4. DST-II (antisymmetric extension + FFT) ----
#
# DST-II: X[k] = 2 * sum_{n=0}^{N-1} x[n] * sin(pi * (k+1) * (2n+1) / (2N))
#
# Standard FFT method via odd antisymmetric extension of length 2(N+1):
#   z[0] = 0, z[n] = x[n-1] for n=1..N, z[N+1] = 0,
#   z[2(N+1)-n] = -x[n-1] for n=1..N
# Then: DST-II[k] = -Im(FFT(z)[k+1]) for k=0..N-1
#
# Operations: 2(N+1)-point FFT + extraction

def setup_dst(signal_length, batch_size, precision, device, **kw):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    return {"x": x, "N": signal_length}

def run_dst(state):
    x = state["x"]
    N = state["N"]
    B = x.shape[0]
    # Antisymmetric extension of length 2(N+1)
    L = 2 * (N + 1)
    z = torch.zeros(B, L, device=x.device, dtype=torch.float32)
    z[:, 1:N+1] = x.float()           # z[1..N] = x[0..N-1]
    # z[N+1] = 0 (already zero)
    z[:, N+2:] = -x.float().flip(dims=[1])  # z[N+2..2N+1] = -x[N-1..0]
    # FFT of extended signal
    Z = torch.fft.fft(z)
    # Extract DST-II coefficients: X[k] = -Im(Z[k+1]) for k=0..N-1
    X = -Z[:, 1:N+1].imag
    torch.cuda.synchronize()


# ---- 5. DWT (Discrete Wavelet Transform) ----
#
# Single-level DWT via convolution with stride-2 downsampling.
# Filter coefficients generated canonically from pywt.Wavelet objects.
# GPU execution via torch.nn.functional.conv1d (pywt itself is CPU-only).

def setup_dwt(signal_length, batch_size, precision, device, wavelet="haar", **kw):
    import pywt
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, 1, signal_length, device=device, dtype=dtype)
    # Canonical wavelet filter coefficients from PyWavelets
    w = pywt.Wavelet(wavelet)
    lo = torch.tensor(w.dec_lo, device=device, dtype=dtype).reshape(1, 1, -1)
    hi = torch.tensor(w.dec_hi, device=device, dtype=dtype).reshape(1, 1, -1)
    return {"x": x, "lo": lo, "hi": hi}

def run_dwt(state):
    x = state["x"]
    lo, hi = state["lo"], state["hi"]
    # Single-level DWT: convolution + stride-2 downsampling
    approx = torch.nn.functional.conv1d(x, lo, stride=2, padding=lo.shape[-1]//2)
    detail = torch.nn.functional.conv1d(x, hi, stride=2, padding=hi.shape[-1]//2)
    torch.cuda.synchronize()


# ---- 6. STFT ----

def setup_stft(signal_length, batch_size, precision, device, window_size=256, hop_length=128, **kw):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=torch.float32)
    window = torch.hann_window(window_size, device=device)
    return {"x": x, "window": window, "n_fft": window_size, "hop": hop_length}

def run_stft(state):
    torch.stft(
        state["x"], n_fft=state["n_fft"], hop_length=state["hop"],
        window=state["window"], return_complex=True
    )
    torch.cuda.synchronize()


# ---- 7. Hilbert Transform ----

def setup_hilbert(signal_length, batch_size, precision, device, **kw):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=torch.float32)
    N = signal_length
    # Precompute analytic signal mask
    h = torch.zeros(N, device=device)
    h[0] = 1
    h[N//2] = 1
    h[1:N//2] = 2
    return {"x": x, "h": h}

def run_hilbert(state):
    X = torch.fft.fft(state["x"])
    Xa = X * state["h"]
    xa = torch.fft.ifft(Xa)
    torch.cuda.synchronize()


# ---- Registry ----

TRANSFORMS = {
    "fft": (setup_fft, run_fft, {}),
    "direct_dft": (setup_direct_dft, run_direct_dft, {}),
    "dct": (setup_dct, run_dct, {}),
    "dst": (setup_dst, run_dst, {}),
    "dwt_haar": (setup_dwt, run_dwt, {"wavelet": "haar"}),
    "dwt_db4": (setup_dwt, run_dwt, {"wavelet": "db4"}),
    "stft": (setup_stft, run_stft, {"window_size": 256, "hop_length": 128}),
    "hilbert": (setup_hilbert, run_hilbert, {}),
}
