"""
TOMLSignals - Filtering Algorithms (Category 2)
=================================================
FIR direct, FIR-FFT, IIR, Median, Savitzky-Golay, Wiener, Matched, Bandpass bank

Canonical implementations:
  - FIR direct: torch.nn.functional.conv1d (canonical)
  - FIR-FFT: torch.fft (canonical)
  - IIR: torchaudio.functional.lfilter (canonical GPU IIR)
  - Median: torch unfold + median (no canonical GPU median filter)
  - Savitzky-Golay: scipy.signal.savgol_coeffs for kernel, torch.nn.functional.conv1d for GPU
  - Wiener: torch.fft (standard frequency-domain Wiener)
  - Matched: torch.fft (standard FFT-based correlation)
  - Filterbank: torch.nn.functional.conv1d (canonical)
"""

import torch
import torch.nn.functional as F
import numpy as np


# ---- 8. FIR Filter (direct convolution) ----

def setup_fir_direct(signal_length, batch_size, precision, device, filter_length=64, **kw):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, 1, signal_length, device=device, dtype=dtype)
    h = torch.randn(1, 1, filter_length, device=device, dtype=dtype)
    return {"x": x, "h": h, "filter_length": filter_length}

def run_fir_direct(state):
    F.conv1d(state["x"], state["h"], padding=state["filter_length"]//2)
    torch.cuda.synchronize()


# ---- 9. FIR Filter (FFT overlap-save) ----

def setup_fir_fft(signal_length, batch_size, precision, device, filter_length=64, **kw):
    dtype = torch.float32  # FFT requires float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    h = torch.randn(filter_length, device=device, dtype=dtype)
    # Zero-pad to next power of 2
    N = 1
    while N < signal_length + filter_length - 1:
        N *= 2
    H = torch.fft.fft(h, n=N)
    return {"x": x, "H": H, "N": N}

def run_fir_fft(state):
    X = torch.fft.fft(state["x"], n=state["N"])
    Y = X * state["H"]
    torch.fft.ifft(Y)
    torch.cuda.synchronize()


# ---- 10. IIR Filter (Butterworth, Direct Form II Transposed) ----
#
# IIR is inherently sequential (each output depends on previous outputs).
# Uses torchaudio.functional.lfilter when available (optimized C++/CUDA),
# falls back to pure-torch Direct Form II transposed otherwise.
# Both implement the same mathematical operation:
#   y[n] = b[0]*x[n] + s_0
#   s_i  = b[i+1]*x[n] - a[i+1]*y[n] + s_{i+1}
# Same TO count regardless of implementation.

try:
    import torchaudio.functional as _AF
    _HAS_TORCHAUDIO = True
except (ImportError, OSError):
    _HAS_TORCHAUDIO = False


def _lfilter_torch(x, a, b):
    """Pure-torch Direct Form II Transposed IIR filter.
    Fallback for environments without torchaudio.
    x: (B, N), a: (order+1,), b: (order+1,)
    """
    B, N = x.shape
    order = a.shape[0] - 1
    # Normalize by a[0]
    b = b / a[0]
    a = a / a[0]
    # State variables
    s = torch.zeros(B, order, device=x.device, dtype=x.dtype)
    y = torch.empty_like(x)
    for n in range(N):
        xn = x[:, n]
        yn = b[0] * xn + s[:, 0]
        y[:, n] = yn
        for i in range(order - 1):
            s[:, i] = b[i + 1] * xn - a[i + 1] * yn + s[:, i + 1]
        s[:, order - 1] = b[order] * xn - a[order] * yn
    return y


def setup_iir(signal_length, batch_size, precision, device, order=4, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    from scipy.signal import butter
    b, a = butter(order, 0.2)
    b_t = torch.tensor(b, device=device, dtype=dtype)
    a_t = torch.tensor(a, device=device, dtype=dtype)
    if _HAS_TORCHAUDIO:
        return {"x": x, "b": b_t, "a": a_t, "order": order, "_lfilter": _AF.lfilter}
    else:
        return {"x": x, "b": b_t, "a": a_t, "order": order, "_lfilter": None}

def run_iir(state):
    if state["_lfilter"] is not None:
        state["_lfilter"](state["x"], state["a"], state["b"], clamp=False)
    else:
        _lfilter_torch(state["x"], state["a"], state["b"])
    torch.cuda.synchronize()


# ---- 11. Median Filter ----

def setup_median(signal_length, batch_size, precision, device, window_size=7, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, 1, signal_length, device=device, dtype=dtype)
    return {"x": x, "window_size": window_size}

def run_median(state):
    x = state["x"]
    w = state["window_size"]
    # Unfold into sliding windows, take median
    x_unfold = x.unfold(2, w, 1)
    torch.median(x_unfold, dim=-1)
    torch.cuda.synchronize()


# ---- 12. Savitzky-Golay Filter ----
#
# Canonical approach: scipy.signal.savgol_coeffs computes the least-squares
# polynomial smoothing kernel, then we apply it as a 1D convolution on GPU.
# This is equivalent to scipy.signal.savgol_filter but runs on CUDA.

def setup_savgol(signal_length, batch_size, precision, device, window_size=7, poly_order=3, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, 1, signal_length, device=device, dtype=dtype)
    # Canonical SG kernel via scipy
    from scipy.signal import savgol_coeffs
    coeffs = savgol_coeffs(window_size, poly_order)
    kernel = torch.tensor(coeffs, device=device, dtype=dtype).reshape(1, 1, -1)
    return {"x": x, "kernel": kernel, "window_size": window_size}

def run_savgol(state):
    F.conv1d(state["x"], state["kernel"], padding=state["window_size"]//2)
    torch.cuda.synchronize()


# ---- 13. Wiener Filter (frequency domain) ----

def setup_wiener(signal_length, batch_size, precision, device, noise_power=0.01, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    noise_var = torch.tensor(noise_power, device=device, dtype=dtype)
    return {"x": x, "noise_var": noise_var}

def run_wiener(state):
    X = torch.fft.fft(state["x"])
    Pxx = torch.abs(X) ** 2
    H = Pxx / (Pxx + state["noise_var"])  # Wiener filter: division is the key TO
    Y = X * H
    torch.fft.ifft(Y)
    torch.cuda.synchronize()


# ---- 14. Matched Filter ----

def setup_matched(signal_length, batch_size, precision, device, template_length=64, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    template = torch.randn(template_length, device=device, dtype=dtype)
    # FFT-based correlation
    N = 1
    while N < signal_length + template_length - 1:
        N *= 2
    T_conj = torch.conj(torch.fft.fft(template, n=N))
    return {"x": x, "T_conj": T_conj, "N": N}

def run_matched(state):
    X = torch.fft.fft(state["x"], n=state["N"])
    Y = X * state["T_conj"]
    torch.fft.ifft(Y)
    torch.cuda.synchronize()


# ---- 15. Bandpass Filter Bank ----

def setup_filterbank(signal_length, batch_size, precision, device, n_channels=32, filter_length=64, **kw):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, 1, signal_length, device=device, dtype=dtype)
    # Random filter bank (n_channels output channels)
    filters = torch.randn(n_channels, 1, filter_length, device=device, dtype=dtype)
    return {"x": x, "filters": filters, "filter_length": filter_length}

def run_filterbank(state):
    F.conv1d(state["x"], state["filters"], padding=state["filter_length"]//2)
    torch.cuda.synchronize()


# ---- Registry ----

FILTERS = {
    "fir_direct": (setup_fir_direct, run_fir_direct, {"filter_length": 64}),
    "fir_fft": (setup_fir_fft, run_fir_fft, {"filter_length": 64}),
    "iir_butter4": (setup_iir, run_iir, {"order": 4}),
    "median": (setup_median, run_median, {"window_size": 7}),
    "savgol": (setup_savgol, run_savgol, {"window_size": 7, "poly_order": 3}),
    "wiener": (setup_wiener, run_wiener, {"noise_power": 0.01}),
    "matched_filter": (setup_matched, run_matched, {"template_length": 64}),
    "filterbank_32ch": (setup_filterbank, run_filterbank, {"n_channels": 32, "filter_length": 64}),
}
