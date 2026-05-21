"""
TOMLSignals - State Estimation (Category 4)
=============================================
Kalman, EKF, UKF, Particle Filter
"""

import torch
import numpy as np


# ---- 20. Kalman Filter ----

def setup_kalman(signal_length, batch_size, precision, device, state_dim=4, **kw):
    dtype = torch.float32
    N = signal_length  # time steps
    n_s = state_dim
    # State transition and observation matrices
    F = torch.eye(n_s, device=device, dtype=dtype) + 0.1 * torch.randn(n_s, n_s, device=device, dtype=dtype)
    H = torch.randn(n_s // 2 or 1, n_s, device=device, dtype=dtype)
    Q = torch.eye(n_s, device=device, dtype=dtype) * 0.01
    R = torch.eye(H.shape[0], device=device, dtype=dtype) * 0.1
    # Observations
    z = torch.randn(batch_size, N, H.shape[0], device=device, dtype=dtype)
    return {"F": F, "H": H, "Q": Q, "R": R, "z": z, "N": N, "n_s": n_s}

def run_kalman(state):
    F, H, Q, R, z = state["F"], state["H"], state["Q"], state["R"], state["z"]
    N, n_s = state["N"], state["n_s"]
    B = z.shape[0]
    n_o = H.shape[0]
    x = torch.zeros(B, n_s, device=z.device, dtype=z.dtype)
    P = torch.eye(n_s, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(B, 1, 1)
    steps = min(N, 200)
    for t in range(steps):
        # Predict
        x = torch.matmul(F, x.unsqueeze(-1)).squeeze(-1)
        P = torch.matmul(F, torch.matmul(P, F.T)) + Q
        # Update
        S = torch.matmul(H, torch.matmul(P, H.T)) + R  # innovation covariance
        K = torch.matmul(P, torch.matmul(H.T, torch.linalg.inv(S)))  # Kalman gain (INVERSION)
        y = z[:, t, :] - torch.matmul(H, x.unsqueeze(-1)).squeeze(-1)
        x = x + torch.matmul(K, y.unsqueeze(-1)).squeeze(-1)
        P = P - torch.matmul(K, torch.matmul(H, P))
    torch.cuda.synchronize()


# ---- 21. Extended Kalman Filter ----

def setup_ekf(signal_length, batch_size, precision, device, state_dim=4, **kw):
    return setup_kalman(signal_length, batch_size, precision, device, state_dim)

def run_ekf(state):
    F, H, Q, R, z = state["F"], state["H"], state["Q"], state["R"], state["z"]
    N, n_s = state["N"], state["n_s"]
    B = z.shape[0]
    x = torch.zeros(B, n_s, device=z.device, dtype=z.dtype)
    P = torch.eye(n_s, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(B, 1, 1)
    steps = min(N, 200)
    for t in range(steps):
        # Nonlinear predict: f(x) = F*x + 0.1*sin(x) (nonlinearity)
        x_pred = torch.matmul(F, x.unsqueeze(-1)).squeeze(-1) + 0.1 * torch.sin(x)
        # Jacobian ~= F + 0.1*diag(cos(x))
        J = F.unsqueeze(0) + 0.1 * torch.diag_embed(torch.cos(x))
        P = torch.bmm(J, torch.bmm(P, J.transpose(1, 2))) + Q
        # Update (same as Kalman)
        S = torch.matmul(H, torch.bmm(P, H.T.unsqueeze(0).expand(B, -1, -1))) + R
        K = torch.bmm(P, torch.matmul(H.T, torch.linalg.inv(S)))
        y = z[:, t, :] - torch.matmul(H, x_pred.unsqueeze(-1)).squeeze(-1)
        x = x_pred + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
        P = P - torch.bmm(K, torch.matmul(H, P))
    torch.cuda.synchronize()


# ---- 22. Unscented Kalman Filter ----

def setup_ukf(signal_length, batch_size, precision, device, state_dim=4, **kw):
    dtype = torch.float32
    N = signal_length
    n_s = state_dim
    # UKF requires a stable system (spectral radius < 1) because Cholesky
    # demands strict positive-definiteness of P at every step. An unstable F
    # causes P to diverge, unlike Kalman/EKF which use inv() that tolerates
    # ill-conditioning. We scale F to ensure stability.
    F_raw = torch.eye(n_s, device=device, dtype=dtype) + 0.1 * torch.randn(n_s, n_s, device=device, dtype=dtype)
    spectral_radius = torch.linalg.eigvals(F_raw).abs().max().item()
    F = F_raw * (0.95 / spectral_radius)  # ensure spectral radius = 0.95
    H = torch.randn(n_s // 2 or 1, n_s, device=device, dtype=dtype)
    Q = torch.eye(n_s, device=device, dtype=dtype) * 0.01
    R = torch.eye(H.shape[0], device=device, dtype=dtype) * 0.1
    z = torch.randn(batch_size, N, H.shape[0], device=device, dtype=dtype)
    return {"F": F, "H": H, "Q": Q, "R": R, "z": z, "N": N, "n_s": n_s}

def run_ukf(state):
    F, H, Q, R, z = state["F"], state["H"], state["Q"], state["R"], state["z"]
    N, n_s = state["N"], state["n_s"]
    B = z.shape[0]
    x = torch.zeros(B, n_s, device=z.device, dtype=z.dtype)
    P = torch.eye(n_s, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(B, 1, 1)
    # UKF parameterization: alpha=1.0 gives all non-negative weights, equivalent
    # to the cubature Kalman filter (Arasaratnam & Haykin, 2009). The standard
    # alpha=1e-3 produces w_c0 ~ -1e6 which destroys P's positive-definiteness
    # in float32 without a square-root formulation.
    alpha, beta, kappa = 1.0, 2.0, 0.0
    lam = alpha**2 * (n_s + kappa) - n_s
    n_sigma = 2 * n_s + 1
    steps = min(N, 100)
    eps_I = 1e-8 * torch.eye(n_s, device=z.device, dtype=z.dtype).unsqueeze(0)
    for t in range(steps):
        # Sigma points via Cholesky
        L = torch.linalg.cholesky((n_s + lam) * P + eps_I)
        sigma = torch.zeros(B, n_sigma, n_s, device=z.device, dtype=z.dtype)
        sigma[:, 0, :] = x
        for i in range(n_s):
            sigma[:, i+1, :] = x + L[:, :, i]
            sigma[:, n_s+i+1, :] = x - L[:, :, i]
        # Propagate through nonlinear model
        sigma_pred = torch.matmul(F, sigma.transpose(1, 2)).transpose(1, 2) + 0.1 * torch.sin(sigma)
        # Weighted mean and covariance
        w_m0 = lam / (n_s + lam)
        w_c0 = w_m0 + (1 - alpha**2 + beta)
        w_i = 1.0 / (2 * (n_s + lam))
        x_pred = w_m0 * sigma_pred[:, 0] + w_i * sigma_pred[:, 1:].sum(dim=1)
        # Covariance
        diff = sigma_pred - x_pred.unsqueeze(1)
        P = Q.unsqueeze(0) + w_c0 * torch.bmm(diff[:, 0:1].transpose(1, 2), diff[:, 0:1])
        for i in range(1, n_sigma):
            P = P + w_i * torch.bmm(diff[:, i:i+1].transpose(1, 2), diff[:, i:i+1])
        # Enforce symmetry (accumulation of rounding errors)
        P = 0.5 * (P + P.transpose(1, 2))
        x = x_pred
    torch.cuda.synchronize()


# ---- 23. Particle Filter (SIR) ----

def setup_particle(signal_length, batch_size, precision, device, state_dim=4, n_particles=1000, **kw):
    dtype = torch.float32
    N = signal_length
    n_s = state_dim
    z = torch.randn(batch_size, N, n_s // 2 or 1, device=device, dtype=dtype)
    return {"z": z, "N": N, "n_s": n_s, "n_p": n_particles}

def run_particle(state):
    z, N, n_s, n_p = state["z"], state["N"], state["n_s"], state["n_p"]
    B = z.shape[0]
    device, dtype = z.device, z.dtype
    particles = torch.randn(B, n_p, n_s, device=device, dtype=dtype)
    weights = torch.ones(B, n_p, device=device, dtype=dtype) / n_p
    steps = min(N, 200)
    for t in range(steps):
        # Predict: random walk
        particles = particles + 0.1 * torch.randn_like(particles)
        # Weight: likelihood based on distance to observation
        obs = z[:, t, :].unsqueeze(1)  # B x 1 x n_o
        dist = torch.sum((particles[:, :, :obs.shape[2]] - obs) ** 2, dim=2)
        log_w = -0.5 * dist
        log_w = log_w - log_w.max(dim=1, keepdim=True).values
        weights = torch.exp(log_w)  # exp is expensive
        weights = weights / weights.sum(dim=1, keepdim=True)  # division (normalize)
        # Systematic resampling
        cumsum = torch.cumsum(weights, dim=1)
        u = (torch.arange(n_p, device=device, dtype=dtype) + torch.rand(B, 1, device=device)) / n_p
        indices = torch.searchsorted(cumsum, u)  # comparison-based
        indices = indices.clamp(0, n_p - 1)
        particles = torch.gather(particles, 1, indices.unsqueeze(-1).expand(-1, -1, n_s))
    torch.cuda.synchronize()


ESTIMATION = {
    "kalman": (setup_kalman, run_kalman, {"state_dim": 4}),
    "ekf": (setup_ekf, run_ekf, {"state_dim": 4}),
    "ukf": (setup_ukf, run_ukf, {"state_dim": 4}),
    "particle_1k": (setup_particle, run_particle, {"state_dim": 4, "n_particles": 1000}),
}
