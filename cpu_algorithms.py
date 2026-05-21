"""
TOMLSignals - CPU Algorithm Implementations (All 37 Algorithms)
================================================================
CPU counterparts of all GPU signal processing algorithms using the
natural libraries a practitioner would use: numpy, scipy, sklearn,
pywt, and PyTorch in CPU mode (for ML models).

Default parallelism is left intact (MKL/OpenBLAS threading for numpy/scipy,
sklearn joblib for decomposition). This represents what a practitioner
would actually get.

Parameters (filter lengths, state dimensions, iterations) match the
GPU implementations exactly for apples-to-apples comparison.

Author: Muntaser Syed
Date: May 2026
"""

import numpy as np
from scipy import signal as sig
from scipy import fft as sp_fft

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    from sklearn.decomposition import PCA as skPCA, FastICA as skFastICA, NMF as skNMF
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import torch
import torch.nn as nn


# =========================================================================
# CATEGORY 1: TRANSFORMS
# =========================================================================

# ---- 1. FFT ----

def setup_fft(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x}

def run_fft(state):
    np.fft.fft(state["x"])


# ---- 2. Direct DFT (matrix multiply) ----

def setup_direct_dft(signal_length, batch_size, precision, **kw):
    dtype = np.complex64 if precision == "fp32" else np.complex128
    N = signal_length
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N).astype(dtype)
    x = (np.random.randn(batch_size, N) + 0j).astype(dtype)
    return {"x": x, "W": W}

def run_direct_dft(state):
    state["x"] @ state["W"]


# ---- 3. DCT-II ----

def setup_dct(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x}

def run_dct(state):
    sp_fft.dct(state["x"], type=2)


# ---- 4. DST-II ----

def setup_dst(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x}

def run_dst(state):
    sp_fft.dst(state["x"], type=2)


# ---- 5. DWT Haar ----

def setup_dwt_haar(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x}

def run_dwt_haar(state):
    for i in range(state["x"].shape[0]):
        pywt.dwt(state["x"][i], "haar")


# ---- 6. DWT db4 ----

def setup_dwt_db4(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x}

def run_dwt_db4(state):
    for i in range(state["x"].shape[0]):
        pywt.dwt(state["x"][i], "db4")


# ---- 7. STFT ----

def setup_stft(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x, "nperseg": 256, "noverlap": 128}

def run_stft(state):
    for i in range(state["x"].shape[0]):
        sig.stft(state["x"][i], nperseg=state["nperseg"],
                 noverlap=state["noverlap"])


# ---- 8. Hilbert Transform ----

def setup_hilbert(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x}

def run_hilbert(state):
    for i in range(state["x"].shape[0]):
        sig.hilbert(state["x"][i])


# =========================================================================
# CATEGORY 2: FILTERS
# =========================================================================

# ---- 9. FIR Direct ----

def setup_fir_direct(signal_length, batch_size, precision,
                     filter_length=64, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    h = np.random.randn(filter_length).astype(dtype)
    return {"x": x, "h": h}

def run_fir_direct(state):
    for i in range(state["x"].shape[0]):
        np.convolve(state["x"][i], state["h"], mode="full")


# ---- 10. FIR FFT ----

def setup_fir_fft(signal_length, batch_size, precision,
                  filter_length=64, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    h = np.random.randn(filter_length).astype(dtype)
    return {"x": x, "h": h}

def run_fir_fft(state):
    for i in range(state["x"].shape[0]):
        sig.fftconvolve(state["x"][i], state["h"], mode="full")


# ---- 11. IIR Butterworth 4th order ----

def setup_iir_butter4(signal_length, batch_size, precision, order=4, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    sos = sig.butter(order, 0.2, btype="low", output="sos")
    return {"x": x, "sos": sos.astype(dtype)}

def run_iir_butter4(state):
    for i in range(state["x"].shape[0]):
        sig.sosfilt(state["sos"], state["x"][i])


# ---- 12. Median Filter ----

def setup_median(signal_length, batch_size, precision, window_size=7, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x, "kernel_size": window_size}

def run_median(state):
    for i in range(state["x"].shape[0]):
        sig.medfilt(state["x"][i], kernel_size=state["kernel_size"])


# ---- 13. Savitzky-Golay ----

def setup_savgol(signal_length, batch_size, precision, window_size=7, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x, "window_length": window_size, "polyorder": 3}

def run_savgol(state):
    for i in range(state["x"].shape[0]):
        sig.savgol_filter(state["x"][i],
                          window_length=state["window_length"],
                          polyorder=state["polyorder"])


# ---- 14. Wiener Filter ----

def setup_wiener(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x}

def run_wiener(state):
    for i in range(state["x"].shape[0]):
        sig.wiener(state["x"][i])


# ---- 15. Matched Filter (FFT correlation) ----

def setup_matched_filter(signal_length, batch_size, precision,
                         template_length=64, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    template = np.random.randn(template_length).astype(dtype)
    return {"x": x, "template": template}

def run_matched_filter(state):
    for i in range(state["x"].shape[0]):
        sig.correlate(state["x"][i], state["template"], mode="full",
                      method="fft")


# ---- 15b. Filterbank (32 channels) ----

def setup_filterbank_32ch(signal_length, batch_size, precision,
                          n_channels=32, filter_length=64, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    filters = np.random.randn(n_channels, filter_length).astype(dtype)
    return {"x": x, "filters": filters, "n_channels": n_channels}

def run_filterbank_32ch(state):
    for i in range(state["x"].shape[0]):
        for c in range(state["n_channels"]):
            np.convolve(state["x"][i], state["filters"][c], mode="full")


# =========================================================================
# CATEGORY 3: ADAPTIVE FILTERS
# =========================================================================

# ---- 16. LMS ----

def setup_lms(signal_length, batch_size, precision, filter_length=32, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    N = signal_length
    M = filter_length
    x = np.random.randn(batch_size, N).astype(dtype)
    d = np.random.randn(batch_size, N).astype(dtype)
    return {"x": x, "d": d, "M": M, "N": N, "mu": np.dtype(dtype).type(0.01)}

def run_lms(state):
    x, d, M, N, mu = state["x"], state["d"], state["M"], state["N"], state["mu"]
    B = x.shape[0]
    for b in range(B):
        w = np.zeros(M, dtype=x.dtype)
        for n in range(M, min(N, M + 200)):
            x_vec = x[b, n-M:n][::-1].copy()
            y = np.dot(w, x_vec)
            e = d[b, n] - y
            w += mu * e * x_vec


# ---- 17. NLMS ----

def setup_nlms(signal_length, batch_size, precision, filter_length=32, **kw):
    return setup_lms(signal_length, batch_size, precision, filter_length)

def run_nlms(state):
    x, d, M, N, mu = state["x"], state["d"], state["M"], state["N"], state["mu"]
    eps = 1e-8
    B = x.shape[0]
    for b in range(B):
        w = np.zeros(M, dtype=x.dtype)
        for n in range(M, min(N, M + 200)):
            x_vec = x[b, n-M:n][::-1].copy()
            y = np.dot(w, x_vec)
            e = d[b, n] - y
            norm = np.dot(x_vec, x_vec) + eps
            w += (mu / norm) * e * x_vec


# ---- 18. RLS ----

def setup_rls(signal_length, batch_size, precision, filter_length=32, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    N = signal_length
    M = filter_length
    x = np.random.randn(batch_size, N).astype(dtype)
    d = np.random.randn(batch_size, N).astype(dtype)
    return {"x": x, "d": d, "M": M, "N": N, "lam": np.dtype(dtype).type(0.99)}

def run_rls(state):
    x, d, M, N, lam = state["x"], state["d"], state["M"], state["N"], state["lam"]
    B = x.shape[0]
    for b in range(B):
        w = np.zeros(M, dtype=x.dtype)
        P = np.eye(M, dtype=x.dtype) * 100
        for n in range(M, min(N, M + 100)):
            x_vec = x[b, n-M:n][::-1].copy()
            y = np.dot(w, x_vec)
            e = d[b, n] - y
            Px = P @ x_vec
            denom = lam + x_vec @ Px
            k = Px / denom
            w += k * e
            P = (P - np.outer(k, x_vec @ P)) / lam


# ---- 19. APA (Affine Projection, P=4) ----

def setup_apa_p4(signal_length, batch_size, precision,
                 filter_length=32, proj_order=4, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    N = signal_length
    M, P = filter_length, proj_order
    x = np.random.randn(batch_size, N).astype(dtype)
    d = np.random.randn(batch_size, N).astype(dtype)
    return {"x": x, "d": d, "M": M, "P": P, "N": N,
            "mu": np.dtype(dtype).type(0.1)}

def run_apa_p4(state):
    x, d, M, P, N, mu = (state["x"], state["d"], state["M"],
                          state["P"], state["N"], state["mu"])
    B = x.shape[0]
    for b in range(B):
        w = np.zeros(M, dtype=x.dtype)
        for n in range(M + P, min(N, M + P + 100)):
            A = np.zeros((M, P), dtype=x.dtype)
            for p in range(P):
                A[:, p] = x[b, n-M-p:n-p][::-1].copy()
            d_vec = d[b, n-P+1:n+1].copy()
            y_vec = A.T @ w
            e_vec = d_vec - y_vec
            AtA = A.T @ A + 1e-6 * np.eye(P, dtype=x.dtype)
            z = np.linalg.solve(AtA, e_vec)
            w += mu * A @ z


# =========================================================================
# CATEGORY 4: STATE ESTIMATION
# =========================================================================

# ---- 20. Kalman Filter ----

def setup_kalman(signal_length, batch_size, precision, state_dim=4, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    N = signal_length
    n_s = state_dim
    F = np.eye(n_s, dtype=dtype) + 0.1 * np.random.randn(n_s, n_s).astype(dtype)
    obs_dim = max(1, n_s // 2)
    H = np.random.randn(obs_dim, n_s).astype(dtype)
    Q = np.eye(n_s, dtype=dtype) * 0.01
    R = np.eye(obs_dim, dtype=dtype) * 0.1
    z = np.random.randn(batch_size, N, obs_dim).astype(dtype)
    return {"F": F, "H": H, "Q": Q, "R": R, "z": z, "N": N, "n_s": n_s}

def run_kalman(state):
    F, H, Q, R, z = state["F"], state["H"], state["Q"], state["R"], state["z"]
    N, n_s = state["N"], state["n_s"]
    B = z.shape[0]
    steps = min(N, 200)
    for b in range(B):
        x = np.zeros(n_s, dtype=z.dtype)
        P = np.eye(n_s, dtype=z.dtype)
        for t in range(steps):
            x = F @ x
            P = F @ P @ F.T + Q
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            y = z[b, t] - H @ x
            x = x + K @ y
            P = P - K @ H @ P


# ---- 21. Extended Kalman Filter ----

def setup_ekf(signal_length, batch_size, precision, state_dim=4, **kw):
    return setup_kalman(signal_length, batch_size, precision, state_dim)

def run_ekf(state):
    F, H, Q, R, z = state["F"], state["H"], state["Q"], state["R"], state["z"]
    N, n_s = state["N"], state["n_s"]
    B = z.shape[0]
    steps = min(N, 200)
    for b in range(B):
        x = np.zeros(n_s, dtype=z.dtype)
        P = np.eye(n_s, dtype=z.dtype)
        for t in range(steps):
            # Nonlinear state transition
            x_pred = F @ x + 0.1 * np.sin(x)
            # Jacobian
            J = F + 0.1 * np.diag(np.cos(x))
            P = J @ P @ J.T + Q
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            y = z[b, t] - H @ x_pred
            x = x_pred + K @ y
            P = P - K @ H @ P


# ---- 22. Unscented Kalman Filter ----

def setup_ukf(signal_length, batch_size, precision, state_dim=4, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    N = signal_length
    n_s = state_dim
    # Match GPU: stabilize F by scaling spectral radius to 0.95
    F_raw = np.eye(n_s, dtype=dtype) + 0.1 * np.random.randn(n_s, n_s).astype(dtype)
    eigvals = np.linalg.eigvals(F_raw)
    spectral_radius = np.max(np.abs(eigvals))
    F = (F_raw * (0.95 / spectral_radius)).astype(dtype)
    obs_dim = max(1, n_s // 2)
    H = np.random.randn(obs_dim, n_s).astype(dtype)
    Q = np.eye(n_s, dtype=dtype) * 0.01
    R = np.eye(obs_dim, dtype=dtype) * 0.1
    z = np.random.randn(batch_size, N, obs_dim).astype(dtype)
    return {"F": F, "H": H, "Q": Q, "R": R, "z": z, "N": N, "n_s": n_s}

def run_ukf(state):
    F, H, Q, R, z = state["F"], state["H"], state["Q"], state["R"], state["z"]
    N, n_s = state["N"], state["n_s"]
    B = z.shape[0]
    steps = min(N, 100)
    n_sigma = 2 * n_s + 1
    alpha, beta, kappa = 1.0, 2.0, 0.0
    lam = alpha ** 2 * (n_s + kappa) - n_s

    # Weights
    Wm = np.full(n_sigma, 1.0 / (2 * (n_s + lam)), dtype=z.dtype)
    Wc = Wm.copy()
    Wm[0] = lam / (n_s + lam)
    Wc[0] = lam / (n_s + lam) + (1 - alpha ** 2 + beta)

    for b in range(B):
        x = np.zeros(n_s, dtype=z.dtype)
        P = np.eye(n_s, dtype=z.dtype)
        for t in range(steps):
            # Sigma points
            scaled_P = (n_s + lam) * P
            scaled_P = (scaled_P + scaled_P.T) / 2  # ensure symmetry
            try:
                L = np.linalg.cholesky(scaled_P)
            except np.linalg.LinAlgError:
                scaled_P += 1e-4 * np.eye(n_s, dtype=z.dtype)
                L = np.linalg.cholesky(scaled_P)
            sigmas = np.zeros((n_sigma, n_s), dtype=z.dtype)
            sigmas[0] = x
            for i in range(n_s):
                sigmas[1 + i] = x + L[i]
                sigmas[1 + n_s + i] = x - L[i]
            # Propagate
            for i in range(n_sigma):
                sigmas[i] = F @ sigmas[i] + 0.1 * np.sin(sigmas[i])
            # Mean and covariance
            x_pred = Wm @ sigmas
            P = Q.copy()
            for i in range(n_sigma):
                d = sigmas[i] - x_pred
                P += Wc[i] * np.outer(d, d)
            P = (P + P.T) / 2
            # Update
            z_pred = H @ x_pred
            y = z[b, t] - z_pred
            S = R.copy()
            Pxz = np.zeros((n_s, H.shape[0]), dtype=z.dtype)
            for i in range(n_sigma):
                dz = H @ sigmas[i] - z_pred
                S += Wc[i] * np.outer(dz, dz)
                dx = sigmas[i] - x_pred
                Pxz += Wc[i] * np.outer(dx, dz)
            K = Pxz @ np.linalg.inv(S)
            x = x_pred + K @ y
            P = P - K @ S @ K.T


# ---- 23. Particle Filter (1000 particles) ----

def setup_particle_1k(signal_length, batch_size, precision,
                      n_particles=1000, state_dim=4, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    N = signal_length
    obs_dim = max(1, state_dim // 2)
    z = np.random.randn(batch_size, N, obs_dim).astype(dtype)
    return {"z": z, "N": N, "n_s": state_dim, "P": n_particles}

def run_particle_1k(state):
    z, N, n_s, n_p = state["z"], state["N"], state["n_s"], state["P"]
    B = z.shape[0]
    steps = min(N, 200)
    obs_dim = max(1, n_s // 2)
    H = np.random.randn(obs_dim, n_s).astype(z.dtype)
    for b in range(B):
        particles = np.random.randn(n_p, n_s).astype(z.dtype)
        weights = np.ones(n_p, dtype=z.dtype) / n_p
        for t in range(steps):
            # Propagate
            particles += np.random.randn(n_p, n_s).astype(z.dtype) * 0.1
            # Weight
            obs_pred = particles @ H.T  # (n_p, obs_dim)
            diff = obs_pred - z[b, t]
            log_w = -0.5 * np.sum(diff ** 2, axis=1)
            log_w -= np.max(log_w)
            weights = np.exp(log_w)
            weights /= np.sum(weights)
            # Systematic resample
            cumsum = np.cumsum(weights)
            positions = (np.arange(n_p) + np.random.rand()) / n_p
            indices = np.searchsorted(cumsum, positions)
            indices = np.clip(indices, 0, n_p - 1)
            particles = particles[indices]


# =========================================================================
# CATEGORY 5: SPECTRAL ESTIMATION
# =========================================================================

# ---- 24. Periodogram ----

def setup_periodogram(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x}

def run_periodogram(state):
    for i in range(state["x"].shape[0]):
        sig.periodogram(state["x"][i])


# ---- 25. Welch ----

def setup_welch(signal_length, batch_size, precision, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    return {"x": x, "nperseg": 256}

def run_welch(state):
    for i in range(state["x"].shape[0]):
        sig.welch(state["x"][i], nperseg=state["nperseg"])


# ---- 26. MUSIC ----

def setup_music(signal_length, batch_size, precision,
                n_sources=3, n_freq_bins=512, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    n_sensors = min(signal_length, 16)
    # X: (n_sensors, signal_length) snapshots
    X = np.random.randn(batch_size, n_sensors, signal_length).astype(dtype)
    return {"X": X, "n_sources": n_sources, "n_freq_bins": n_freq_bins,
            "n_sensors": n_sensors}

def run_music(state):
    X, n_src, n_bins = state["X"], state["n_sources"], state["n_freq_bins"]
    n_sensors = state["n_sensors"]
    B = X.shape[0]
    for b in range(B):
        # Covariance
        R = X[b] @ X[b].T / X.shape[2]
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(R)
        # Noise subspace
        En = eigvecs[:, :n_sensors - n_src]
        En_proj = En @ En.T
        # Pseudospectrum
        freqs = np.linspace(0, np.pi, n_bins)
        spectrum = np.zeros(n_bins, dtype=X.dtype)
        for f_idx in range(n_bins):
            a = np.exp(-1j * freqs[f_idx] * np.arange(n_sensors))
            a = a.astype(np.complex64 if X.dtype == np.float32 else np.complex128)
            denom = np.real(a.conj() @ En_proj @ a)
            spectrum[f_idx] = 1.0 / (np.abs(denom) + 1e-10)


# ---- 27. ESPRIT ----

def setup_esprit(signal_length, batch_size, precision, n_sources=3, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    n_sensors = min(signal_length, 16)
    X = np.random.randn(batch_size, n_sensors, signal_length).astype(dtype)
    return {"X": X, "n_sources": n_sources, "n_sensors": n_sensors}

def run_esprit(state):
    X, n_src = state["X"], state["n_sources"]
    n_sensors = state["n_sensors"]
    B = X.shape[0]
    for b in range(B):
        R = X[b] @ X[b].T / X.shape[2]
        eigvals, eigvecs = np.linalg.eigh(R)
        Es = eigvecs[:, n_sensors - n_src:]
        Es1 = Es[:-1, :]
        Es2 = Es[1:, :]
        Phi, _, _, _ = np.linalg.lstsq(Es1, Es2, rcond=None)
        freqs = np.angle(np.linalg.eigvals(Phi))


# =========================================================================
# CATEGORY 6: DECOMPOSITION
# =========================================================================

# ---- 28. SVD ----

def setup_svd(signal_length, batch_size, precision, n_features=64, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    X = np.random.randn(batch_size, signal_length, n_features).astype(dtype)
    return {"X": X}

def run_svd(state):
    for i in range(state["X"].shape[0]):
        np.linalg.svd(state["X"][i], full_matrices=False)


# ---- 29. PCA ----

def setup_pca(signal_length, batch_size, precision,
              n_features=64, n_components=8, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    X = np.random.randn(batch_size, signal_length, n_features).astype(dtype)
    return {"X": X, "n_components": n_components}

def run_pca(state):
    for i in range(state["X"].shape[0]):
        pca = skPCA(n_components=state["n_components"])
        pca.fit_transform(state["X"][i])


# ---- 30. FastICA ----

def setup_fastica(signal_length, batch_size, precision,
                  n_features=16, n_components=4, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    X = np.random.randn(batch_size, signal_length, n_features).astype(dtype)
    return {"X": X, "n_components": n_components}

def run_fastica(state):
    for i in range(state["X"].shape[0]):
        ica = skFastICA(n_components=state["n_components"],
                        max_iter=50, tol=1e-4)
        try:
            ica.fit_transform(state["X"][i])
        except Exception:
            pass  # FastICA can fail to converge on random data


# ---- 31. NMF ----

def setup_nmf(signal_length, batch_size, precision,
              n_features=64, n_components=8, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    # NMF requires non-negative data
    X = np.abs(np.random.randn(batch_size, signal_length, n_features)).astype(dtype)
    return {"X": X, "n_components": n_components}

def run_nmf(state):
    for i in range(state["X"].shape[0]):
        nmf = skNMF(n_components=state["n_components"],
                    max_iter=50, init="random")
        nmf.fit_transform(state["X"][i])


# =========================================================================
# CATEGORY 7: COMPRESSION
# =========================================================================

# ---- 32. JPEG (DCT-based) ----

def setup_jpeg_q50(signal_length, batch_size, precision, quality=50, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    import math
    side = int(math.sqrt(signal_length))
    side = (side // 8) * 8
    # Standard JPEG luminance quantization table
    Q = np.array([
        [16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]
    ], dtype=dtype)
    if quality < 50:
        S = 5000.0 / quality
    else:
        S = 200.0 - 2.0 * quality
    Q = np.floor((S * Q + 50) / 100).clip(1, 255)
    # Pre-compute DCT basis
    basis = np.zeros((8, 8), dtype=dtype)
    for i in range(8):
        for j in range(8):
            if i == 0:
                basis[i, j] = 1.0 / np.sqrt(8)
            else:
                basis[i, j] = np.sqrt(2.0 / 8) * np.cos(
                    np.pi * (2 * j + 1) * i / 16)
    img = np.random.randn(batch_size, side, side).astype(dtype)
    return {"img": img, "Q": Q, "basis": basis, "side": side}

def run_jpeg_q50(state):
    img, Q, basis, side = state["img"], state["Q"], state["basis"], state["side"]
    B = img.shape[0]
    for b in range(B):
        for i in range(0, side, 8):
            for j in range(0, side, 8):
                block = img[b, i:i+8, j:j+8]
                dct_block = basis @ block @ basis.T
                quantized = np.round(dct_block / Q)
                dequantized = quantized * Q


# ---- 33. MDCT Audio Codec ----

def setup_mdct_audio(signal_length, batch_size, precision,
                     frame_size=512, **kw):
    dtype = np.float32 if precision == "fp32" else np.float64
    x = np.random.randn(batch_size, signal_length).astype(dtype)
    # MDCT basis
    n = np.arange(2 * frame_size, dtype=dtype)
    k = np.arange(frame_size, dtype=dtype)
    basis = np.cos(np.pi / frame_size * (n[:, None] + 0.5 + frame_size / 2)
                   * (k[None, :] + 0.5)).astype(dtype)
    # Window
    window = np.sin(np.pi / (2 * frame_size)
                    * (np.arange(2 * frame_size, dtype=dtype) + 0.5))
    return {"x": x, "basis": basis, "window": window,
            "frame_size": frame_size, "N": signal_length}

def run_mdct_audio(state):
    x, basis, window = state["x"], state["basis"], state["window"]
    fs, N = state["frame_size"], state["N"]
    B = x.shape[0]
    n_frames = max(1, min((N - 2 * fs) // fs + 1, 50))
    for b in range(B):
        for f_idx in range(n_frames):
            start = f_idx * fs
            frame = x[b, start:start + 2 * fs] * window
            coeffs = basis.T @ frame
            power = np.abs(coeffs) ** 2
            mask = np.exp(np.log(power + 1e-10) * 0.5)
            quantized = np.round(coeffs / (mask + 1e-10))


# =========================================================================
# CATEGORY 8: ML-ENHANCED (PyTorch CPU mode)
# =========================================================================

class _CNN1DDenoiser(nn.Module):
    def __init__(self, channels=32, n_layers=3):
        super().__init__()
        layers = [nn.Conv1d(1, channels, 7, padding=3), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Conv1d(channels, channels, 7, padding=3), nn.ReLU()]
        layers += [nn.Conv1d(channels, 1, 7, padding=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _LSTMDenoiser(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


class _TransformerDenoiser(nn.Module):
    def __init__(self, d_model=64, nhead=4, d_ff=128, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.output_proj(x)


# ---- 34. CNN Denoiser (CPU) ----

def setup_cnn_denoiser(signal_length, batch_size, precision,
                       channels=32, n_layers=3, **kw):
    dtype = torch.float32
    model = _CNN1DDenoiser(channels, n_layers).to(dtype).eval()
    x = torch.randn(batch_size, 1, signal_length, dtype=dtype)
    return {"model": model, "x": x}

def run_cnn_denoiser(state):
    with torch.no_grad():
        state["model"](state["x"])


# ---- 35. LSTM Denoiser (CPU) ----

def setup_lstm_denoiser(signal_length, batch_size, precision,
                        hidden_size=128, **kw):
    dtype = torch.float32
    model = _LSTMDenoiser(hidden_size).to(dtype).eval()
    x = torch.randn(batch_size, signal_length, 1, dtype=dtype)
    return {"model": model, "x": x}

def run_lstm_denoiser(state):
    with torch.no_grad():
        state["model"](state["x"])


# ---- 36. Transformer Denoiser (CPU) ----

def setup_transformer_denoiser(signal_length, batch_size, precision,
                               d_model=64, nhead=4, d_ff=128, n_layers=2,
                               **kw):
    dtype = torch.float32
    model = _TransformerDenoiser(d_model, nhead, d_ff, n_layers).to(dtype).eval()
    x = torch.randn(batch_size, signal_length, 1, dtype=dtype)
    return {"model": model, "x": x}

def run_transformer_denoiser(state):
    with torch.no_grad():
        state["model"](state["x"])


# =========================================================================
# REGISTRY
# =========================================================================

CPU_ALGORITHMS = {
    "transform": {
        "fft":              (setup_fft, run_fft),
        "direct_dft":       (setup_direct_dft, run_direct_dft),
        "dct":              (setup_dct, run_dct),
        "dst":              (setup_dst, run_dst),
        "dwt_haar":         (setup_dwt_haar, run_dwt_haar),
        "dwt_db4":          (setup_dwt_db4, run_dwt_db4),
        "stft":             (setup_stft, run_stft),
        "hilbert":          (setup_hilbert, run_hilbert),
    },
    "filter": {
        "fir_direct":       (setup_fir_direct, run_fir_direct),
        "fir_fft":          (setup_fir_fft, run_fir_fft),
        "iir_butter4":      (setup_iir_butter4, run_iir_butter4),
        "median":           (setup_median, run_median),
        "savgol":           (setup_savgol, run_savgol),
        "wiener":           (setup_wiener, run_wiener),
        "matched_filter":   (setup_matched_filter, run_matched_filter),
        "filterbank_32ch":  (setup_filterbank_32ch, run_filterbank_32ch),
    },
    "adaptive": {
        "lms":              (setup_lms, run_lms),
        "nlms":             (setup_nlms, run_nlms),
        "rls":              (setup_rls, run_rls),
        "apa_p4":           (setup_apa_p4, run_apa_p4),
    },
    "estimation": {
        "kalman":           (setup_kalman, run_kalman),
        "ekf":              (setup_ekf, run_ekf),
        "ukf":              (setup_ukf, run_ukf),
        "particle_1k":      (setup_particle_1k, run_particle_1k),
    },
    "spectral": {
        "periodogram":      (setup_periodogram, run_periodogram),
        "welch":            (setup_welch, run_welch),
        "music":            (setup_music, run_music),
        "esprit":           (setup_esprit, run_esprit),
    },
    "decomposition": {
        "svd":              (setup_svd, run_svd),
        "pca":              (setup_pca, run_pca),
        "fastica":          (setup_fastica, run_fastica),
        "nmf":              (setup_nmf, run_nmf),
    },
    "compression": {
        "jpeg_q50":         (setup_jpeg_q50, run_jpeg_q50),
        "mdct_audio":       (setup_mdct_audio, run_mdct_audio),
    },
    "ml_enhanced": {
        "cnn_denoiser":     (setup_cnn_denoiser, run_cnn_denoiser),
        "lstm_denoiser":    (setup_lstm_denoiser, run_lstm_denoiser),
        "transformer_denoiser": (setup_transformer_denoiser, run_transformer_denoiser),
    },
}


# =========================================================================
# QUICK VALIDATION
# =========================================================================

if __name__ == "__main__":
    print("CPU Algorithm Validation — testing all 37 algorithms")
    print("=" * 60)

    total, passed, failed = 0, 0, []

    for cat_name, algorithms in CPU_ALGORITHMS.items():
        print(f"\n  Category: {cat_name}")
        for alg_name, (setup_fn, run_fn) in algorithms.items():
            total += 1
            try:
                N = 256 if alg_name in ("direct_dft", "music", "esprit",
                                         "svd", "pca", "fastica", "nmf") else 1024
                state = setup_fn(signal_length=N, batch_size=1, precision="fp32")
                run_fn(state)
                print(f"    ✓ {alg_name} (N={N})")
                passed += 1
            except Exception as e:
                print(f"    ✗ {alg_name}: {e}")
                failed.append(alg_name)

    print(f"\n{'='*60}")
    print(f"  Passed: {passed}/{total}")
    if failed:
        print(f"  Failed: {failed}")
