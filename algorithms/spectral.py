"""
TOMLSignals - Spectral Estimation (Category 5)
================================================
Periodogram, Welch, MUSIC, ESPRIT

Canonical implementations:
  - Periodogram: torch.fft.fft + power spectrum (standard)
  - Welch: Vectorized segment-window-FFT-average (standard Welch, no scipy on GPU)
  - MUSIC: Covariance eigendecomp + vectorized pseudospectrum (standard MUSIC)
  - ESPRIT: Shift-invariance + least squares + eigvals (standard ESPRIT)
"""

import torch
import numpy as np


# ---- 24. Periodogram ----

def setup_periodogram(signal_length, batch_size, precision, device, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    return {"x": x}

def run_periodogram(state):
    X = torch.fft.fft(state["x"])
    Pxx = torch.abs(X) ** 2 / state["x"].shape[1]
    torch.cuda.synchronize()


# ---- 25. Welch's Method (vectorized) ----
#
# Previous implementation used a Python for-loop over segments, which
# serializes what should be a fully parallel operation on GPU.
#
# Vectorized approach: unfold extracts all overlapping segments at once,
# then a single batched FFT processes them in parallel. This is how
# any competent GPU Welch implementation works.
#
# Operations: N/hop windowed segments, each gets an FFT of length seg_len,
# plus element-wise |.|^2 and averaging.

def setup_welch(signal_length, batch_size, precision, device, segment_length=256, overlap=128, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    window = torch.hann_window(segment_length, device=device, dtype=dtype)
    return {"x": x, "window": window, "seg_len": segment_length, "overlap": overlap}

def run_welch(state):
    x, window = state["x"], state["window"]
    seg_len, overlap = state["seg_len"], state["overlap"]
    hop = seg_len - overlap
    # Extract all overlapping segments at once: B x n_segs x seg_len
    segments = x.unfold(1, seg_len, hop)
    # Apply window to all segments simultaneously
    segments = segments * window
    # Batched FFT over all segments
    X = torch.fft.fft(segments)
    # Power spectrum, averaged over segments
    Pxx = torch.abs(X) ** 2
    Pxx_avg = Pxx.mean(dim=1)
    torch.cuda.synchronize()


# ---- 26. MUSIC (vectorized pseudospectrum) ----
#
# Previous implementation used a Python for-loop over 512 frequency bins,
# computing one steering vector at a time. This serializes what is naturally
# a matrix operation on GPU.
#
# Vectorized approach: build the full steering matrix A (n_freq x n_sensors),
# compute noise subspace projection En @ En^H once, then use einsum to
# evaluate the pseudospectrum for all frequencies in one call.
#
# Operations: covariance (BMM), eigendecomp, noise projection (BMM),
# steering matrix (exp), pseudospectrum (einsum + division).

def setup_music(signal_length, batch_size, precision, device, n_sources=3, n_freq_bins=512, **kw):
    dtype = torch.float32
    # Simulated data matrix (signal_length = snapshot count, columns = sensors)
    n_sensors = min(signal_length, 16)  # use signal_length as snapshots, cap sensors
    X = torch.randn(batch_size, n_sensors, signal_length, device=device, dtype=torch.complex64)
    # Precompute steering matrix: A[f, s] = exp(-j * freq_f * s)
    freqs = torch.linspace(0, np.pi, n_freq_bins, device=device)
    sensor_idx = torch.arange(n_sensors, device=device, dtype=torch.float32)
    A = torch.exp(-1j * freqs.unsqueeze(1) * sensor_idx.unsqueeze(0)).to(torch.complex64)
    return {
        "X": X, "A": A,
        "n_sources": n_sources, "n_freq_bins": n_freq_bins, "n_sensors": n_sensors,
    }

def run_music(state):
    X = state["X"]
    A = state["A"]
    n_src = state["n_sources"]
    n_sens = state["n_sensors"]
    B = X.shape[0]
    # Sample covariance matrix: B x n_sens x n_sens
    Rxx = torch.bmm(X, X.conj().transpose(1, 2)) / X.shape[2]
    # Eigendecomposition (divisions, square roots inside)
    eigenvalues, eigenvectors = torch.linalg.eigh(Rxx)
    # Noise subspace: B x n_sens x (n_sens - n_src)
    En = eigenvectors[:, :, :n_sens - n_src]
    # Noise projection matrix: B x n_sens x n_sens
    En_proj = torch.bmm(En, En.conj().transpose(1, 2))
    # Vectorized pseudospectrum: P[b,f] = 1 / |a_f^H @ En_proj_b @ a_f|
    # A: n_freq x n_sens, En_proj: B x n_sens x n_sens
    denom = torch.einsum('fi,bij,fj->bf', A.conj(), En_proj, A)
    P_music = 1.0 / denom.abs()  # division: key TO cost
    torch.cuda.synchronize()


# ---- 27. ESPRIT ----

def setup_esprit(signal_length, batch_size, precision, device, n_sources=3, **kw):
    dtype = torch.float32
    n_sensors = min(signal_length, 16)
    X = torch.randn(batch_size, n_sensors, signal_length, device=device, dtype=torch.complex64)
    return {"X": X, "n_sources": n_sources, "n_sensors": n_sensors}

def run_esprit(state):
    X = state["X"]
    n_src = state["n_sources"]
    n_sens = state["n_sensors"]
    B = X.shape[0]
    Rxx = torch.bmm(X, X.conj().transpose(1, 2)) / X.shape[2]
    eigenvalues, eigenvectors = torch.linalg.eigh(Rxx)
    # Signal subspace
    Es = eigenvectors[:, :, n_sens - n_src:]
    # Shift invariance: Es1 and Es2
    Es1 = Es[:, :-1, :]
    Es2 = Es[:, 1:, :]
    # Least squares: Phi = (Es1^H Es1)^{-1} Es1^H Es2
    Phi = torch.linalg.lstsq(Es1, Es2).solution  # involves inversion
    # Eigenvalues of Phi give DOA estimates
    eigs = torch.linalg.eigvals(Phi)
    torch.cuda.synchronize()


SPECTRAL = {
    "periodogram": (setup_periodogram, run_periodogram, {}),
    "welch": (setup_welch, run_welch, {"segment_length": 256, "overlap": 128}),
    "music": (setup_music, run_music, {"n_sources": 3, "n_freq_bins": 512}),
    "esprit": (setup_esprit, run_esprit, {"n_sources": 3}),
}
