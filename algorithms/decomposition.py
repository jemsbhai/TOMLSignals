"""
TOMLSignals - Decomposition (Category 6)
==========================================
SVD, PCA, ICA (FastICA), NMF

Canonical implementations:
  - SVD: torch.linalg.svd (canonical)
  - PCA: torch.pca_lowrank (canonical GPU PCA via randomized SVD)
  - FastICA: Manual implementation (no canonical GPU ICA library exists)
  - NMF: Manual multiplicative update (no canonical GPU NMF library exists)
"""

import torch
import numpy as np


# ---- 28. SVD ----

def setup_svd(signal_length, batch_size, precision, device, n_features=64, **kw):
    dtype = torch.float32
    X = torch.randn(batch_size, signal_length, n_features, device=device, dtype=dtype)
    return {"X": X}

def run_svd(state):
    torch.linalg.svd(state["X"], full_matrices=False)
    torch.cuda.synchronize()


# ---- 29. PCA (via torch.pca_lowrank) ----
#
# Previous implementation computed the full covariance matrix then ran
# eigendecomposition — O(N*D^2 + D^3) where D = n_features.
#
# torch.pca_lowrank uses randomized SVD (Halko et al. 2011), which is:
#   1. The canonical PyTorch API for PCA
#   2. More efficient for low-rank approximations: O(N*D*k) where k = n_components
#   3. What sklearn.decomposition.PCA uses internally (via ARPACK/randomized)
#   4. Numerically more stable than eigendecomposition of the covariance matrix
#
# Operations: centering (N*D subtractions) + randomized SVD (matrix multiplies,
# QR factorizations, a small SVD of a k×k matrix).

def setup_pca(signal_length, batch_size, precision, device, n_features=64, n_components=8, **kw):
    dtype = torch.float32
    X = torch.randn(batch_size, signal_length, n_features, device=device, dtype=dtype)
    return {"X": X, "n_components": n_components}

def run_pca(state):
    X = state["X"]
    n_c = state["n_components"]
    # Center the data (mean subtraction)
    X_centered = X - X.mean(dim=1, keepdim=True)
    # torch.pca_lowrank: returns (U, S, V) like truncated SVD
    # Input must be 2D, so process each batch element
    # For batch_size=1 (our benchmark default), this is a single call
    for i in range(X_centered.shape[0]):
        torch.pca_lowrank(X_centered[i], q=n_c)
    torch.cuda.synchronize()


# ---- 30. FastICA ----
#
# No canonical GPU ICA library exists. This is a faithful implementation of
# the FastICA algorithm (Hyvärinen & Oja, 2000) with tanh nonlinearity.
#
# Operations per iteration: whitening (eigendecomp + matrix multiply),
# then per ICA iteration: matrix multiply (W @ X_white), tanh (15,000 TOs each),
# outer product, QR orthogonalization.

def setup_ica(signal_length, batch_size, precision, device, n_features=16, n_components=4, max_iter=50, **kw):
    dtype = torch.float32
    X = torch.randn(batch_size, n_features, signal_length, device=device, dtype=dtype)
    return {"X": X, "n_components": n_components, "max_iter": max_iter}

def run_ica(state):
    X = state["X"]
    n_c = state["n_components"]
    max_iter = state["max_iter"]
    B, D, N = X.shape
    # Whiten via PCA
    X_centered = X - X.mean(dim=2, keepdim=True)
    C = torch.bmm(X_centered, X_centered.transpose(1, 2)) / N
    eigvals, eigvecs = torch.linalg.eigh(C)
    # Whitening matrix: D_sqrt_inv @ V^T
    D_sqrt_inv = torch.diag_embed(1.0 / torch.sqrt(eigvals[:, -n_c:] + 1e-8))  # division + sqrt
    K = torch.bmm(D_sqrt_inv, eigvecs[:, :, -n_c:].transpose(1, 2))
    X_white = torch.bmm(K, X_centered)
    # FastICA iteration with tanh nonlinearity
    W = torch.randn(B, n_c, n_c, device=X.device, dtype=X.dtype)
    W, _ = torch.linalg.qr(W)
    for _ in range(max_iter):
        WX = torch.bmm(W, X_white)
        g = torch.tanh(WX)  # tanh: 15,000 TOs each
        g_prime = 1 - g ** 2
        W_new = torch.bmm(g, X_white.transpose(1, 2)) / N - g_prime.mean(dim=2, keepdim=True).transpose(1, 2) * W
        W, _ = torch.linalg.qr(W_new)  # orthogonalize
    S = torch.bmm(W, X_white)
    torch.cuda.synchronize()


# ---- 31. NMF (multiplicative update) ----
#
# No canonical GPU NMF library exists. This implements Lee & Seung (2001)
# multiplicative update rules, which is the standard NMF algorithm.
#
# Operations per iteration: 4 matrix multiplies (W^T V, W^T W H, V H^T, W H H^T)
# + 2 element-wise divisions.

def setup_nmf(signal_length, batch_size, precision, device, n_features=64, n_components=8, max_iter=50, **kw):
    dtype = torch.float32
    V = torch.abs(torch.randn(batch_size, n_features, signal_length, device=device, dtype=dtype)) + 0.01
    return {"V": V, "n_components": n_components, "max_iter": max_iter, "n_features": n_features}

def run_nmf(state):
    V = state["V"]
    n_c = state["n_components"]
    max_iter = state["max_iter"]
    B, F, N = V.shape
    # Initialize W and H (non-negative)
    W = torch.abs(torch.randn(B, F, n_c, device=V.device, dtype=V.dtype)) + 0.01
    H = torch.abs(torch.randn(B, n_c, N, device=V.device, dtype=V.dtype)) + 0.01
    for _ in range(max_iter):
        # Update H: H *= (W^T V) / (W^T W H + eps)
        WtV = torch.bmm(W.transpose(1, 2), V)
        WtWH = torch.bmm(torch.bmm(W.transpose(1, 2), W), H) + 1e-8
        H = H * WtV / WtWH  # element-wise division
        # Update W: W *= (V H^T) / (W H H^T + eps)
        VHt = torch.bmm(V, H.transpose(1, 2))
        WHHt = torch.bmm(W, torch.bmm(H, H.transpose(1, 2))) + 1e-8
        W = W * VHt / WHHt  # element-wise division
    torch.cuda.synchronize()


DECOMPOSITION = {
    "svd": (setup_svd, run_svd, {"n_features": 64}),
    "pca": (setup_pca, run_pca, {"n_features": 64, "n_components": 8}),
    "fastica": (setup_ica, run_ica, {"n_features": 16, "n_components": 4, "max_iter": 50}),
    "nmf": (setup_nmf, run_nmf, {"n_features": 64, "n_components": 8, "max_iter": 50}),
}
