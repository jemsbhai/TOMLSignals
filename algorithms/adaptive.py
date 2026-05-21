"""
TOMLSignals - Adaptive Filtering (Category 3)
===============================================
LMS, NLMS, RLS, Affine Projection
"""

import torch
import numpy as np


# ---- 16. LMS ----

def setup_lms(signal_length, batch_size, precision, device, filter_length=32, **kw):
    dtype = torch.float32
    N = signal_length
    M = filter_length
    # Input signal and desired signal
    x = torch.randn(batch_size, N, device=device, dtype=dtype)
    d = torch.randn(batch_size, N, device=device, dtype=dtype)
    w = torch.zeros(batch_size, M, device=device, dtype=dtype)
    mu = 0.01
    return {"x": x, "d": d, "w": w.clone(), "M": M, "N": N, "mu": mu}

def run_lms(state):
    x, d, M, N, mu = state["x"], state["d"], state["M"], state["N"], state["mu"]
    B = x.shape[0]
    w = torch.zeros(B, M, device=x.device, dtype=x.dtype)
    for n in range(M, min(N, M + 200)):  # limit iterations for benchmarking
        x_vec = x[:, n-M:n].flip(1)
        y = torch.sum(w * x_vec, dim=1, keepdim=True)
        e = d[:, n:n+1] - y
        w = w + mu * e * x_vec
    torch.cuda.synchronize()


# ---- 17. NLMS ----

def setup_nlms(signal_length, batch_size, precision, device, filter_length=32, **kw):
    return setup_lms(signal_length, batch_size, precision, device, filter_length)

def run_nlms(state):
    x, d, M, N = state["x"], state["d"], state["M"], state["N"]
    mu, B = state["mu"], x.shape[0]
    eps = 1e-8
    w = torch.zeros(B, M, device=x.device, dtype=x.dtype)
    for n in range(M, min(N, M + 200)):
        x_vec = x[:, n-M:n].flip(1)
        y = torch.sum(w * x_vec, dim=1, keepdim=True)
        e = d[:, n:n+1] - y
        norm = torch.sum(x_vec ** 2, dim=1, keepdim=True) + eps
        w = w + (mu / norm) * e * x_vec  # division here
    torch.cuda.synchronize()


# ---- 18. RLS ----

def setup_rls(signal_length, batch_size, precision, device, filter_length=32, **kw):
    dtype = torch.float32
    N = signal_length
    M = filter_length
    x = torch.randn(batch_size, N, device=device, dtype=dtype)
    d = torch.randn(batch_size, N, device=device, dtype=dtype)
    return {"x": x, "d": d, "M": M, "N": N, "lam": 0.99}

def run_rls(state):
    x, d, M, N, lam = state["x"], state["d"], state["M"], state["N"], state["lam"]
    B = x.shape[0]
    w = torch.zeros(B, M, 1, device=x.device, dtype=x.dtype)
    P = torch.eye(M, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1) * 100
    for n in range(M, min(N, M + 100)):  # fewer iterations, heavier per step
        x_vec = x[:, n-M:n].flip(1).unsqueeze(2)  # B x M x 1
        y = torch.bmm(w.transpose(1, 2), x_vec).squeeze()
        e = d[:, n] - y
        Px = torch.bmm(P, x_vec)  # B x M x 1
        denom = lam + torch.bmm(x_vec.transpose(1, 2), Px).squeeze(-1)  # B x 1
        k = Px / denom.unsqueeze(-1)  # gain vector, division here
        w = w + k * e.unsqueeze(-1).unsqueeze(-1)
        P = (P - torch.bmm(k, torch.bmm(x_vec.transpose(1, 2), P))) / lam
    torch.cuda.synchronize()


# ---- 19. Affine Projection ----

def setup_apa(signal_length, batch_size, precision, device, filter_length=32, proj_order=4, **kw):
    dtype = torch.float32
    N = signal_length
    M = filter_length
    P = proj_order
    x = torch.randn(batch_size, N, device=device, dtype=dtype)
    d = torch.randn(batch_size, N, device=device, dtype=dtype)
    return {"x": x, "d": d, "M": M, "N": N, "P": P, "mu": 0.1}

def run_apa(state):
    x, d, M, N, P, mu = state["x"], state["d"], state["M"], state["N"], state["P"], state["mu"]
    B = x.shape[0]
    w = torch.zeros(B, M, device=x.device, dtype=x.dtype)
    eps = 1e-6
    for n in range(M + P, min(N, M + P + 100)):
        # Build input matrix A: B x M x P
        A = torch.stack([x[:, n-M-p:n-p].flip(1) for p in range(P)], dim=2)
        d_vec = d[:, n-P+1:n+1].unsqueeze(2)  # B x P x 1
        y_vec = torch.bmm(A.transpose(1, 2), w.unsqueeze(2))  # B x P x 1
        e_vec = d_vec - y_vec
        # APA update: w += mu * A * (A^T A + eps I)^{-1} * e
        ATA = torch.bmm(A.transpose(1, 2), A) + eps * torch.eye(P, device=x.device).unsqueeze(0)
        ATA_inv_e = torch.linalg.solve(ATA, e_vec)  # involves division/LU
        w = w + mu * torch.bmm(A, ATA_inv_e).squeeze(2)
    torch.cuda.synchronize()


ADAPTIVE = {
    "lms": (setup_lms, run_lms, {"filter_length": 32}),
    "nlms": (setup_nlms, run_nlms, {"filter_length": 32}),
    "rls": (setup_rls, run_rls, {"filter_length": 32}),
    "apa_p4": (setup_apa, run_apa, {"filter_length": 32, "proj_order": 4}),
}
