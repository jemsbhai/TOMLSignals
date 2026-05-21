"""
TOMLSignals - Compression (Category 7)
========================================
JPEG pipeline (DCT + quantize + entropy), MDCT audio codec
"""

import torch
import numpy as np


# ---- 32. JPEG Pipeline (DCT + quantize + zigzag scan) ----

def setup_jpeg(signal_length, batch_size, precision, device, quality=50, block_size=8, **kw):
    dtype = torch.float32
    # Treat signal_length as image side length (square image)
    side = int(np.sqrt(signal_length)) or 64
    side = (side // block_size) * block_size  # ensure divisible
    img = torch.randn(batch_size, 1, side, side, device=device, dtype=dtype) * 128 + 128
    # Precompute DCT basis for 8x8 blocks
    n = torch.arange(block_size, device=device, dtype=dtype)
    k = n.unsqueeze(1)
    dct_basis = torch.cos(np.pi * (2 * n + 1) * k / (2 * block_size))
    dct_basis[0] *= 1 / np.sqrt(block_size)
    dct_basis[1:] *= np.sqrt(2 / block_size)
    # Quantization matrix (standard JPEG luminance)
    Q = torch.tensor([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], device=device, dtype=dtype)
    # Scale by quality
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    Q = torch.clamp(torch.floor((Q * scale + 50) / 100), min=1)
    return {"img": img, "dct_basis": dct_basis, "Q": Q, "block_size": block_size, "side": side}

def run_jpeg(state):
    img = state["img"]
    basis = state["dct_basis"]
    Q = state["Q"]
    bs = state["block_size"]
    side = state["side"]
    B = img.shape[0]
    # Extract 8x8 blocks
    blocks = img.unfold(2, bs, bs).unfold(3, bs, bs)  # B x 1 x H/8 x W/8 x 8 x 8
    blocks = blocks.contiguous().view(-1, bs, bs)
    # 2D DCT: basis @ block @ basis^T
    dct_coeffs = torch.matmul(basis, torch.matmul(blocks, basis.T))
    # Quantize (division -- expensive in TOs)
    quantized = torch.round(dct_coeffs / Q)
    # Dequantize (multiply, for decoder simulation)
    dequantized = quantized * Q
    torch.cuda.synchronize()


# ---- 33. MDCT Audio Codec ----

def setup_mdct(signal_length, batch_size, precision, device, frame_size=1024, **kw):
    dtype = torch.float32
    x = torch.randn(batch_size, signal_length, device=device, dtype=dtype)
    N = frame_size
    # MDCT basis
    n = torch.arange(2 * N, device=device, dtype=dtype)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    basis = torch.cos(np.pi / N * (n + 0.5 + N / 2) * (k + 0.5))  # N x 2N
    # Window
    window = torch.sin(np.pi / (2 * N) * (torch.arange(2 * N, device=device, dtype=dtype) + 0.5))
    # Psychoacoustic masking threshold (simplified: exponential)
    bark_scale = torch.exp(-0.5 * torch.arange(N, device=device, dtype=dtype) / N)  # exp is expensive
    return {"x": x, "basis": basis, "window": window, "bark": bark_scale, "N": N}

def run_mdct(state):
    x, basis, window, bark, N = state["x"], state["basis"], state["window"], state["bark"], state["N"]
    B, L = x.shape
    n_frames = (L - 2 * N) // N + 1
    n_frames = max(1, min(n_frames, 50))  # cap for benchmarking
    for i in range(n_frames):
        start = i * N
        frame = x[:, start:start + 2*N] * window
        # MDCT transform
        X = torch.matmul(basis, frame.unsqueeze(-1)).squeeze(-1)  # MAC
        # Psychoacoustic masking (exp/log operations)
        power = torch.abs(X) ** 2
        mask = torch.log(power + 1e-10) * bark  # log is expensive
        threshold = torch.exp(mask)  # exp is expensive
        # Quantize above threshold (division)
        scale = torch.clamp(threshold, min=1e-6)
        quantized = torch.round(X / scale)
    torch.cuda.synchronize()


COMPRESSION = {
    "jpeg_q50": (setup_jpeg, run_jpeg, {"quality": 50, "block_size": 8}),
    "mdct_audio": (setup_mdct, run_mdct, {"frame_size": 512}),
}
