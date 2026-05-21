"""
TOMLSignals - Transistor Operation Model
==========================================
Derives TO (Transistor Operation) counts for all 37 signal processing algorithms
as analytic functions of signal length N, batch size B, and algorithm parameters.

Grounded in the TOML framework (Syed et al., FLAIRS 2026) with TO costs at 45nm
reference node, consistent with Horowitz (ISSCC 2014) energy-per-operation data.

Methodology:
  - Each floating-point operation on a GPU dispatches a full FMA unit.
    Therefore: 1 FP op (add, mul, or FMA) = 1 MAC = 5,000 TOs.
  - Transcendentals (exp, div, sqrt, sin, tanh) use iterative algorithms
    (Newton-Raphson, polynomial approximation) requiring multiple FMA cycles,
    hence higher TO costs.
  - Memory costs distinguish SRAM (on-chip, reused within a kernel) from
    HBM (off-chip weight/data loads).
  - Operation counts are derived from the algorithm's mathematical definition,
    not from GPU kernel implementation details, making them architecture-independent.

TO Cost Table (45nm reference, all three TOML papers):
  MAC (FP32):     5,000 TOs   (1 FMA dispatch)
  Division:      15,000 TOs   (iterative, ~3 FMA cycles)
  Square root:   15,000 TOs   (iterative, ~3 FMA cycles)
  Exp:           18,000 TOs   (polynomial approx + range reduction)
  Sin/Cos:       18,000 TOs   (polynomial approximation)
  Tanh:          15,000 TOs   (exp-based or polynomial)
  Sigmoid:       18,000 TOs   (exp + division)
  ReLU:             100 TOs   (comparison + mux)
  GELU:          20,000 TOs   (erf approximation)
  Softmax/elem:  25,000 TOs   (exp + sum + division)
  Comparison:        50 TOs   (single comparator)
  Mem SRAM:         192 TOs/word  (on-chip, 32-bit)
  Mem HBM:       10,000 TOs/word  (off-chip, 32-bit)

Author: Muntaser Syed
Date: May 2026
"""

import math
from typing import Dict

# =========================================================================
# TO COST TABLE (consistent across FLAIRS-39, IC2E 2026, MLSys 2027)
# =========================================================================

TO = {
    "mac":       5_000,     # FP32 multiply-accumulate (1 FMA dispatch on GPU)
    "div":      15_000,     # Division (iterative, ~3× MAC)
    "sqrt":     15_000,     # Square root (iterative, ~3× MAC)
    "exp":      18_000,     # Exponential (polynomial + range reduction)
    "sin":      18_000,     # Sine (polynomial approximation)
    "cos":      18_000,     # Cosine (polynomial approximation)
    "tanh":     15_000,     # Hyperbolic tangent
    "sigmoid":  18_000,     # Sigmoid = exp + div
    "relu":        100,     # ReLU (comparison + mux)
    "gelu":     20_000,     # GELU (erf approximation)
    "softmax":  25_000,     # Softmax per element (exp + sum + div)
    "cmp":          50,     # Comparison
    "abs":       5_000,     # Absolute value (same as 1 MAC: mask + select)
    "neg":       5_000,     # Negation (sign bit flip, but dispatches full FMA)
    "mem_sram":    192,     # SRAM read or write (32-bit word)
    "mem_hbm":  10_000,     # HBM read or write (32-bit word)
}


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def _log2(n: int) -> float:
    """log base 2."""
    return math.log2(n) if n > 0 else 0


# =========================================================================
# HELPER: FFT TO COUNT
# =========================================================================

def fft_tos_cooley_tukey(N: int) -> Dict[str, float]:
    """
    [ORIGINAL — preserved for comparison]
    TO count for N-point complex FFT (Cooley-Tukey radix-2).

    Standard operation count (Cooley & Tukey, 1965):
      - N/2 · log₂(N) butterfly operations
      - Each butterfly: 1 complex multiply + 1 complex add + 1 complex subtract
      - Complex multiply: 4 real multiplies + 2 real adds = 6 FP ops
      - Complex add/sub: 2 real adds each = 4 FP ops
      - Per butterfly: 10 real FP ops

    On GPU, each FP op dispatches 1 FMA unit = 1 MAC.
    Total: N/2 · log₂(N) · 10 MACs
    """
    if N <= 1:
        return {"to_compute": 0, "to_memory": 0}
    n_butterflies = (N // 2) * _log2(N)
    # 10 real FP ops per butterfly, each = 1 MAC
    to_compute = n_butterflies * 10 * TO["mac"]
    return {"to_compute": to_compute, "n_butterflies": n_butterflies}


def _is_pow2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def fft_tos(N: int) -> Dict[str, float]:
    """
    TO count for N-point complex FFT (split-radix).

    Split-radix FFT operation count (Duhamel & Vetterli, 1990):
      For N = 2^m (m >= 2):
        Total real FP ops = 4N·log₂(N) − 6N + 8

      This counts real additions + real multiplications. On GPU, each
      FP op dispatches 1 FMA unit = 1 MAC.

    For non-power-of-2 sizes, the formula is mathematically invalid.
    We approximate by padding to the next power of 2, which represents
    a lower bound: cuFFT's mixed-radix/Bluestein algorithms for sizes
    with large prime factors are typically MORE expensive than padding.
    (Validated: FFT(8194) is 4.6x slower than FFT(16384) on RTX 4090.)

    Validated against Nsight Compute hardware instruction counts:
      N=4096: formula gives 172,040 ops vs 173,055 measured (99.4% match)

    Reference: P. Duhamel and M. Vetterli, "Fast Fourier Transforms:
    A Tutorial Review and a State of the Art," Signal Processing, vol. 19,
    no. 4, pp. 259–299, 1990.
    """
    if N <= 1:
        return {"to_compute": 0, "to_memory": 0}
    # Split-radix only valid for N = 2^m; pad otherwise
    N_eff = N if _is_pow2(N) else _next_pow2(N)
    # Split-radix: 4N·log₂(N) − 6N + 8 real FP ops
    n_ops = 4 * N_eff * _log2(N_eff) - 6 * N_eff + 8
    to_compute = n_ops * TO["mac"]
    return {"to_compute": to_compute, "n_ops": n_ops, "N_eff": N_eff}


# =========================================================================
# CATEGORY 1: TRANSFORMS
# =========================================================================

def to_fft(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """FFT: N-point complex FFT via Cooley-Tukey."""
    f = fft_tos(N)
    to_compute = B * f["to_compute"]
    # Memory: read N words from HBM, write N complex words (2N) to HBM
    to_memory = B * (N + 2 * N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_direct_dft_v0(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    [ORIGINAL — preserved for comparison]
    Direct DFT with 8 real ops per complex MAC.
    """
    to_compute = B * N * N * 8 * TO["mac"]
    to_memory = (B * 2 * N + 2 * N * N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_direct_dft(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    Direct DFT: X = x @ W, complex matrix multiply.

    x: (B, N) complex, W: (N, N) complex.
    Complex matrix multiply: B · N² complex MACs.

    On FMA hardware, each complex MAC = 4 FMA instructions:
      (a+bi)(c+di) accumulated into (e+fi):
        e += a·c   (1 FMA)
        e -= b·d   (1 FMA)
        f += a·d   (1 FMA)
        f += b·c   (1 FMA)

    Validated against Nsight Compute:
      N=1024: formula gives 4,194,304 FMA vs 4,198,400 measured (99.9% match)

    Total: B · N² · 4 MACs
    """
    # B * N^2 complex MACs, each = 4 FMA on FMA hardware
    to_compute = B * N * N * 4 * TO["mac"]
    # Memory: load x (2BN complex words) + W (2N² complex words) from HBM
    to_memory = (B * 2 * N + 2 * N * N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_dct(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    DCT-II via Makhoul's algorithm:
      1. Reorder: N index operations (no arithmetic)
      2. N-point FFT
      3. N complex twiddle multiplies (4 real muls + 2 real adds each)
      4. N real extractions + N multiplications (factor of 2.0)

    Total: FFT(N) + N · 6 MACs + N · 1 MAC
    """
    f = fft_tos(N)
    to_fft_part = B * f["to_compute"]
    to_twiddle = B * N * 6 * TO["mac"]      # complex multiply with twiddle
    to_scale = B * N * TO["mac"]              # multiply by 2.0
    to_compute = to_fft_part + to_twiddle + to_scale
    to_memory = B * (N + 2 * N) * TO["mem_hbm"]  # read real, write real
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_dst(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    DST-II via antisymmetric extension:
      1. Build extended signal of length L = 2(N+1): N copies + N negations
      2. L-point FFT
      3. Extract N imaginary parts + N negations

    Total: 2N MACs (negations) + FFT(L) + N MACs (negations)
    """
    L = 2 * (N + 1)
    f = fft_tos(L)
    to_copy_negate = B * 2 * N * TO["mac"]   # negations + copies
    to_fft_part = B * f["to_compute"]
    to_extract = B * N * TO["mac"]            # negate imaginary parts
    to_compute = to_copy_negate + to_fft_part + to_extract
    to_memory = B * (N + L + N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_dwt(N: int, B: int = 1, wavelet: str = "haar", **kw) -> Dict[str, float]:
    """
    Single-level DWT: two convolutions (lo + hi) with stride 2.

    Each conv: ceil(N/2) output points, each = K MACs (filter length K).
    Total: 2 · ceil(N/2) · K MACs.

    Haar: K=2, db4: K=8.
    """
    K = {"haar": 2, "db4": 8}.get(wavelet, 2)
    n_out = (N + 1) // 2  # ceil(N/2)
    to_compute = B * 2 * n_out * K * TO["mac"]
    # Memory: read N words, write 2 * n_out words
    to_memory = B * (N + 2 * n_out) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_stft(N: int, B: int = 1, window_size: int = 256, hop_length: int = 128, **kw) -> Dict[str, float]:
    """
    STFT: Windowed FFT of overlapping frames.

    Number of frames: F = floor((N - window_size) / hop_length) + 1
    Per frame: window_size multiplications (windowing) + FFT(window_size)
    Total: F · (W MACs + FFT(W))
    """
    W = window_size
    F_frames = max(1, (N - W) // hop_length + 1)
    f = fft_tos(W)
    to_window = F_frames * W * TO["mac"]      # element-wise window multiply
    to_fft_part = F_frames * f["to_compute"]
    to_compute = B * (to_window + to_fft_part)
    # Memory: read N input + W window, write F * (W/2+1) * 2 complex output
    n_freq = W // 2 + 1
    to_memory = B * (N + F_frames * n_freq * 2) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_hilbert(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    Hilbert transform: analytic signal via FFT.

    Steps: FFT(N) + N multiplications (mask) + IFFT(N)
    Mask values are 0, 1, 2 — but GPU still dispatches FMA for each.
    Total: 2 · FFT(N) + N MACs
    """
    f = fft_tos(N)
    to_compute = B * (2 * f["to_compute"] + N * TO["mac"])
    to_memory = B * (N + 2 * N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


# =========================================================================
# CATEGORY 2: FILTERS
# =========================================================================

def to_fir_direct(N: int, B: int = 1, filter_length: int = 64, **kw) -> Dict[str, float]:
    """
    FIR direct convolution: conv1d with kernel of length K.

    N output points, each = K MACs (multiply-accumulate with filter taps).
    Total: N · K MACs.
    """
    K = filter_length
    to_compute = B * N * K * TO["mac"]
    to_memory = B * (N + K + N) * TO["mem_hbm"]  # input + kernel + output
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_fir_fft(N: int, B: int = 1, filter_length: int = 64, **kw) -> Dict[str, float]:
    """
    FIR via FFT: FFT(x) * H then IFFT.

    Pad to M = next_pow2(N + K - 1).
    Steps: FFT(M) + M complex multiplies + IFFT(M)
    Complex multiply: 6 FP ops each.
    Total: 2 · FFT(M) + M · 6 MACs
    """
    K = filter_length
    M = _next_pow2(N + K - 1)
    f = fft_tos(M)
    to_compute = B * (2 * f["to_compute"] + M * 6 * TO["mac"])
    to_memory = B * (N + 2 * M) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_iir(N: int, B: int = 1, order: int = 4, **kw) -> Dict[str, float]:
    """
    IIR Direct Form II Transposed (order P):

    Per time step:
      y[n] = b[0]*x[n] + s[0]                        → 1 MAC + 1 add = 2 MACs
      s[i] = b[i+1]*x[n] - a[i+1]*y[n] + s[i+1]     → 2 MACs + 1 add = 3 MACs (×(P-1))
      s[P-1] = b[P]*x[n] - a[P]*y[n]                 → 2 MACs

    Per step: 2 + 3(P-1) + 2 = 3P + 1 MACs
    Total: N · (3P + 1) MACs

    IIR is inherently sequential — cannot be parallelized across time.
    """
    P = order
    ops_per_step = 3 * P + 1
    to_compute = B * N * ops_per_step * TO["mac"]
    to_memory = B * (N + N + P) * TO["mem_hbm"]  # input + output + state
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_median_v0(N: int, B: int = 1, window_size: int = 7, **kw) -> Dict[str, float]:
    """
    [ORIGINAL — preserved for comparison]
    Median filter with spurious unfold MACs.
    """
    W = window_size
    n_windows = N - W + 1
    to_sort = n_windows * W * _log2(W) * TO["cmp"]
    to_unfold = n_windows * W * TO["mac"]
    to_compute = B * (to_sort + to_unfold)
    to_memory = B * (N + n_windows) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_median(N: int, B: int = 1, window_size: int = 7, **kw) -> Dict[str, float]:
    """
    Median filter: sliding window + sort-based median.

    (N - W + 1) windows, each sorted via comparison network.
    Sorting W elements: O(W log W) comparisons.
    Total: (N - W + 1) · W · log₂(W) comparisons

    Nsight Compute confirms: zero FP32 ops. torch.median uses
    integer comparison sorting only. No floating-point arithmetic.
    The TO cost is purely comparison-based (TO["cmp"] = 50 TOs each).
    """
    W = window_size
    n_windows = N - W + 1
    # Sorting: ~W * log2(W) comparisons per window
    to_sort = n_windows * W * _log2(W) * TO["cmp"]
    to_compute = B * to_sort
    to_memory = B * (N + n_windows) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_savgol(N: int, B: int = 1, window_size: int = 7, **kw) -> Dict[str, float]:
    """
    Savitzky-Golay: convolution with precomputed kernel of length W.

    Identical to FIR direct with K = W.
    Total: N · W MACs
    """
    return to_fir_direct(N, B, filter_length=window_size)


def to_wiener(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    Frequency-domain Wiener filter:
      1. FFT(x)                           → FFT(N)
      2. |X|² = Re² + Im²                → 2N muls + N adds = 3N MACs
      3. H = |X|² / (|X|² + σ²)          → N adds + N divisions
      4. Y = X * H                        → N complex muls = 6N MACs
      5. IFFT(Y)                          → FFT(N)

    Total: 2·FFT(N) + 3N + N + N·(div) + 6N MACs
    """
    f = fft_tos(N)
    to_fft_2x = 2 * f["to_compute"]
    to_power = 3 * N * TO["mac"]              # |X|^2
    to_add = N * TO["mac"]                     # Pxx + noise_var
    to_div = N * TO["div"]                     # division
    to_cmul = 6 * N * TO["mac"]               # complex multiply X*H
    to_compute = B * (to_fft_2x + to_power + to_add + to_div + to_cmul)
    to_memory = B * (N + 2 * N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_matched(N: int, B: int = 1, template_length: int = 64, **kw) -> Dict[str, float]:
    """
    FFT-based matched filter (correlation):
      1. FFT(x, M)                → FFT(M)
      2. X * conj(T)              → M complex multiplies = 6M MACs
      3. IFFT                     → FFT(M)

    M = next_pow2(N + K - 1). Template FFT is precomputed.
    """
    K = template_length
    M = _next_pow2(N + K - 1)
    f = fft_tos(M)
    to_compute = B * (2 * f["to_compute"] + M * 6 * TO["mac"])
    to_memory = B * (N + 2 * M) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_filterbank(N: int, B: int = 1, n_channels: int = 32, filter_length: int = 64, **kw) -> Dict[str, float]:
    """
    Bandpass filter bank: C parallel FIR filters, each length K.

    C channels × N output points × K MACs each.
    Total: C · N · K MACs
    """
    C, K = n_channels, filter_length
    to_compute = B * C * N * K * TO["mac"]
    to_memory = B * (N + C * K + C * N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


# =========================================================================
# CATEGORY 3: ADAPTIVE FILTERS
# =========================================================================

def to_lms(N: int, B: int = 1, filter_length: int = 32, **kw) -> Dict[str, float]:
    """
    LMS adaptive filter (T iterations, filter length M):
      Per step:
        - x_vec extraction + flip: M ops (memory)
        - y = w · x_vec (dot product): M MACs
        - e = d[n] - y: 1 MAC
        - w += mu * e * x_vec: M muls + M adds = 2M MACs

      Per step: 3M + 1 MACs
      Total: T · (3M + 1) MACs

    T = min(N, M + 200) - M iterations (as benchmarked).
    """
    M = filter_length
    T = min(N, M + 200) - M
    T = max(T, 0)
    ops_per_step = 3 * M + 1
    to_compute = B * T * ops_per_step * TO["mac"]
    to_memory = B * (N + N + M) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_nlms(N: int, B: int = 1, filter_length: int = 32, **kw) -> Dict[str, float]:
    """
    NLMS: LMS + normalization.
      Per step (additional to LMS):
        - ||x||² = sum(x²): M MACs
        - mu / ||x||²: 1 division
        - normalized update: M MACs

      Per step: 4M + 1 MACs + 1 division
    """
    M = filter_length
    T = min(N, M + 200) - M
    T = max(T, 0)
    to_dot = T * M * TO["mac"]           # w · x
    to_error = T * TO["mac"]              # d - y
    to_norm = T * M * TO["mac"]           # ||x||²
    to_div = T * TO["div"]                # mu / norm
    to_update = T * 2 * M * TO["mac"]    # mu_n * e * x_vec
    to_compute = B * (to_dot + to_error + to_norm + to_div + to_update)
    to_memory = B * (N + N + M) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_rls(N: int, B: int = 1, filter_length: int = 32, **kw) -> Dict[str, float]:
    """
    RLS adaptive filter (T iterations, filter length M):
      Per step:
        - y = w^T · x: M MACs
        - e = d[n] - y: 1 MAC
        - Px = P · x: M² MACs (matrix-vector)
        - denom = λ + x^T · Px: M MACs + 1 MAC
        - k = Px / denom: M divisions
        - w += k · e: M MACs
        - P update: M² MACs (outer product) + M² MACs (subtract) + M² divs

      Per step: ~3M² + 3M MACs + M divisions
      Total: T · (3M² + 3M + M·div)

    T = min(N, M + 100) - M iterations.
    """
    M = filter_length
    T = min(N, M + 100) - M
    T = max(T, 0)
    to_matmul = T * M * M * TO["mac"]        # P @ x
    to_dot = T * M * TO["mac"]                 # x^T @ Px
    to_error = T * (M + 1) * TO["mac"]        # y = w^T x, e = d - y
    to_gain_div = T * M * TO["div"]            # k = Px / denom
    to_w_update = T * M * TO["mac"]            # w += k * e
    to_p_update = T * 2 * M * M * TO["mac"]   # outer product + P subtract
    to_p_div = T * M * M * TO["div"]           # P /= lambda
    to_compute = B * (to_matmul + to_dot + to_error + to_gain_div +
                      to_w_update + to_p_update + to_p_div)
    to_memory = B * (N + N + M * M) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_apa(N: int, B: int = 1, filter_length: int = 32, proj_order: int = 4, **kw) -> Dict[str, float]:
    """
    Affine Projection Algorithm (T iterations, filter M, projection order P):
      Per step:
        - Build A matrix: M·P copies (memory only)
        - y = A^T · w: M·P MACs
        - e = d - y: P MACs
        - A^T A: M·P² MACs
        - solve (A^T A)^{-1} e: ~P³/3 MACs (LU) + P² MACs (back-sub)
        - w += mu · A · z: M·P MACs

      Per step: ~2MP + P³/3 + P² + P MACs, plus P³/3 divisions (LU pivots)
    """
    M, P = filter_length, proj_order
    T = min(N, M + P + 100) - (M + P)
    T = max(T, 0)
    to_yv = T * M * P * TO["mac"]             # A^T w
    to_err = T * P * TO["mac"]                  # d - y
    to_ata = T * M * P * P * TO["mac"]         # A^T A
    to_solve = T * (P * P * P // 3) * TO["mac"] + T * (P * P) * TO["div"]
    to_update = T * M * P * TO["mac"]           # A @ z
    to_compute = B * (to_yv + to_err + to_ata + to_solve + to_update)
    to_memory = B * (N + N + M + M * P) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


# =========================================================================
# CATEGORY 4: STATE ESTIMATION
# =========================================================================

def to_kalman(N: int, B: int = 1, state_dim: int = 4, **kw) -> Dict[str, float]:
    """
    Kalman filter (T steps, state dim S, observation dim S/2):
      Per step:
        - x_pred = F · x: S² MACs
        - P_pred = F P F^T + Q: 2S³ MACs + S² MACs
        - Innovation: y = z - H x_pred: S·(S/2) MACs
        - S = H P H^T + R: 2·(S/2)·S² MACs
        - K = P H^T S^{-1}: S²·(S/2) MACs + (S/2)³/3 divisions (inverse)
        - x update: S·(S/2) MACs
        - P update: S²·(S/2) MACs

      Per step: ~6S³ MACs + S³/24 divisions (approximate)
    """
    S = state_dim
    T = min(N, 100)  # capped at 100 steps in benchmark
    obs_dim = max(1, S // 2)

    to_predict = T * (S * S + 2 * S * S * S + S * S) * TO["mac"]
    to_innovation = T * (S * obs_dim) * TO["mac"]
    to_kalman_gain = T * (2 * obs_dim * S * S + obs_dim * obs_dim * S) * TO["mac"]
    to_inverse = T * (obs_dim ** 3) * TO["div"]
    to_update = T * (S * obs_dim + S * S * obs_dim) * TO["mac"]
    to_compute = B * (to_predict + to_innovation + to_kalman_gain + to_inverse + to_update)
    to_memory = B * (N * obs_dim + S * S + S) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_ekf(N: int, B: int = 1, state_dim: int = 4, **kw) -> Dict[str, float]:
    """
    Extended Kalman Filter: same structure as Kalman + Jacobian computation.

    Jacobian via finite differences: S forward evaluations of nonlinear model,
    each involving S MACs + S tanh evaluations.
    Additional: S² tanh per step (nonlinear model).
    """
    S = state_dim
    T = min(N, 100)
    obs_dim = max(1, S // 2)

    # Kalman base cost
    kf = to_kalman(N, B=1, state_dim=S)
    to_kalman_base = kf["to_compute"]

    # Jacobian via implicit differentiation in nonlinear model
    # Nonlinear model: F @ x + 0.1 * sin(x) → S² MACs + S sin
    to_nonlinear = T * (S * S * TO["mac"] + S * TO["sin"])
    # Jacobian construction: ~S evaluations
    to_jacobian = T * S * (S * TO["mac"] + S * TO["sin"])

    to_compute = B * (to_kalman_base + to_nonlinear + to_jacobian)
    to_memory = B * (N * obs_dim + S * S + S) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_ukf(N: int, B: int = 1, state_dim: int = 4, **kw) -> Dict[str, float]:
    """
    Unscented Kalman Filter (T steps, state dim S):
      - 2S+1 sigma points per step
      - Each sigma point: S MACs (state transform) + S sin evaluations
      - Cholesky: S³/3 MACs + S² sqrt
      - Weighted mean: (2S+1) · S MACs
      - Weighted covariance: (2S+1) · S² MACs (outer products)
      - Symmetry enforcement: S² MACs

    Per step: ~S³/3 + (2S+1)(S + S²) + S² sqrt
    """
    S = state_dim
    T = min(N, 100)
    n_sigma = 2 * S + 1

    to_cholesky = T * (S ** 3 // 3) * TO["mac"] + T * S * TO["sqrt"]
    to_sigma_gen = T * 2 * S * S * TO["mac"]  # x ± L columns
    to_propagate = T * n_sigma * (S * S * TO["mac"] + S * TO["sin"])
    to_mean = T * n_sigma * S * TO["mac"]
    to_cov = T * n_sigma * S * S * TO["mac"]  # outer products
    to_symmetry = T * S * S * TO["mac"]

    to_compute = B * (to_cholesky + to_sigma_gen + to_propagate +
                      to_mean + to_cov + to_symmetry)
    to_memory = B * (N * S + S * S + n_sigma * S) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_particle(N: int, B: int = 1, n_particles: int = 1000, state_dim: int = 4, **kw) -> Dict[str, float]:
    """
    Particle filter (T steps, P particles, state dim S):
      Per step:
        - Propagate: P · S MACs (state transition) + P · S random draws
        - Weight: P · S MACs (likelihood) + P exp + P divisions (normalize)
        - Resample: P comparisons + P copies

    Random number generation: ~10 MACs per sample (Mersenne Twister/Box-Muller).
    """
    S = state_dim
    P = n_particles
    T = min(N, 100)

    to_propagate = T * P * S * TO["mac"]
    to_random = T * P * S * 10 * TO["mac"]     # RNG
    to_weight = T * P * S * TO["mac"]           # likelihood
    to_exp = T * P * TO["exp"]                   # exp(log-likelihood)
    to_normalize = T * (P * TO["mac"] + P * TO["div"])  # sum + divide
    to_resample = T * P * TO["cmp"]              # systematic resampling

    to_compute = B * (to_propagate + to_random + to_weight +
                      to_exp + to_normalize + to_resample)
    to_memory = B * (N * S + P * S) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


# =========================================================================
# CATEGORY 5: SPECTRAL ESTIMATION
# =========================================================================

def to_periodogram(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    Periodogram: FFT then |X|²/N.
      1. FFT(N)
      2. |X|² = Re² + Im²: 3N MACs
      3. / N: N divisions

    Total: FFT(N) + 3N MACs + N divisions
    """
    f = fft_tos(N)
    to_compute = B * (f["to_compute"] + 3 * N * TO["mac"] + N * TO["div"])
    to_memory = B * (N + N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_welch(N: int, B: int = 1, segment_length: int = 256, overlap: int = 128, **kw) -> Dict[str, float]:
    """
    Welch's method (vectorized):
      - K segments of length W with hop H
      - K = floor((N - W) / H) + 1
      - Per segment: W multiplications (window) + FFT(W)
      - Power: K · W MACs (|.|²)
      - Average: W MACs (mean over K segments)

    Total: K · (W + FFT(W) + W) + W MACs
    """
    W = segment_length
    H = W - overlap
    K = max(1, (N - W) // H + 1)
    f = fft_tos(W)
    to_window = K * W * TO["mac"]
    to_fft_all = K * f["to_compute"]
    to_power = K * W * 3 * TO["mac"]    # |X|² for complex
    to_avg = W * TO["mac"]               # mean over segments
    to_compute = B * (to_window + to_fft_all + to_power + to_avg)
    to_memory = B * (N + K * W) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_music(N: int, B: int = 1, n_sources: int = 3, n_freq_bins: int = 512, **kw) -> Dict[str, float]:
    """
    MUSIC algorithm (vectorized):
      n_sensors = min(N, 16), N = snapshot count

      1. Covariance: R = X X^H / N → P² · N complex MACs
      2. Eigendecomp (eigh): ~P³ MACs
      3. Noise subspace En: P × (P - n_src) selection
      4. Noise projection: En @ En^H → P²(P-n_src) complex MACs
      5. Pseudospectrum (einsum): F frequencies, each → P² complex MACs + 1 div

    P = n_sensors, F = n_freq_bins
    """
    P = min(N, 16)
    n_src = n_sources
    F = n_freq_bins

    to_cov = P * P * N * 8 * TO["mac"]        # complex BMM
    to_eig = P * P * P * TO["mac"]              # eigendecomposition
    to_proj = P * P * (P - n_src) * 8 * TO["mac"]  # noise projection
    to_steer = F * P * TO["exp"]                # steering vectors (complex exp)
    to_pseudo = F * P * P * 8 * TO["mac"]      # einsum a^H En_proj a
    to_div = F * TO["div"]                      # 1/denom

    to_compute = B * (to_cov + to_eig + to_proj + to_steer + to_pseudo + to_div)
    to_memory = B * (P * N + P * P + F * P) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_esprit(N: int, B: int = 1, n_sources: int = 3, **kw) -> Dict[str, float]:
    """
    ESPRIT: shift-invariance + eigenvalues.
      P = min(N, 16), N = snapshots

      1. Covariance: P² · N complex MACs
      2. Eigendecomp: P³ MACs
      3. Signal subspace selection: slicing (no compute)
      4. Shift: Es1 = Es[:-1], Es2 = Es[1:] (no compute)
      5. Least squares: lstsq(Es1, Es2) → ~n_src²·(P-1) MACs + n_src³/3 divs
      6. Eigenvalues of Phi: n_src³ MACs
    """
    P = min(N, 16)
    n_src = n_sources

    to_cov = P * P * N * 8 * TO["mac"]
    to_eig = P * P * P * TO["mac"]
    to_lstsq = n_src * n_src * (P - 1) * 8 * TO["mac"] + (n_src ** 3 // 3) * TO["div"]
    to_eigvals = (n_src ** 3) * TO["mac"]

    to_compute = B * (to_cov + to_eig + to_lstsq + to_eigvals)
    to_memory = B * (P * N + P * P) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


# =========================================================================
# CATEGORY 6: DECOMPOSITION
# =========================================================================

def to_svd(N: int, B: int = 1, n_features: int = 64, **kw) -> Dict[str, float]:
    """
    Truncated SVD of (N × D) matrix:
      Golub-Kahan bidiagonalization + QR iteration.
      Dominant cost: ~6·N·D² MACs (for N > D).
    """
    D = n_features
    to_compute = B * 6 * N * D * D * TO["mac"]
    to_memory = B * (N * D + D * D + N * D) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_pca_v0(N: int, B: int = 1, n_features: int = 64, n_components: int = 8, **kw) -> Dict[str, float]:
    """
    [ORIGINAL — preserved for comparison]
    PCA with single-pass formula (missed power iterations).
    """
    D, k = n_features, n_components
    to_center = N * D * TO["mac"] + D * TO["div"]
    to_project = N * D * k * TO["mac"]
    to_qr = N * k * k * TO["mac"]
    to_svd_small = k * k * k * TO["mac"]
    to_backproj = D * k * k * TO["mac"]
    to_compute = B * (to_center + to_project + to_qr + to_svd_small + to_backproj)
    to_memory = B * (N * D + D * k) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_pca(N: int, B: int = 1, n_features: int = 64, n_components: int = 8,
           niter: int = 2, **kw) -> Dict[str, float]:
    """
    PCA via torch.pca_lowrank (randomized SVD, Halko et al. 2011).

    torch.pca_lowrank(A, q=k, center=True, niter=2) internally calls
    svd_lowrank which performs:
      1. Centering: N·D subtractions + D divisions (mean)
      2. Initial projection: Q = A @ R → N·D·q MACs
      3. QR(Q): N·q matrix (cost dominated by cuSOLVER overhead)
      4. Power iterations (niter=2), each:
         - Z = A.T @ Q → D·N·q MACs
         - QR(Z): D·q matrix
         - Q = A @ Z → N·D·q MACs
         - QR(Q): N·q matrix
      5. B = Q.T @ A → q·N·D MACs
      6. SVD(B): q×D matrix → 6·q·D² MACs (lower bound)
      7. U = Q @ U_small → N·q·q MACs

    Total matmul MACs: N·D·q·(1 + 2·niter + 1) + 6·q·D² + N·q²

    Note: QR factorizations dominate actual GPU time (74% from timing)
    but are cuSOLVER overhead not captured by operation counting.
    The remaining 10.6× gap between this formula and NCU is from
    cuSOLVER QR/SVD overhead on small matrices (Finding F-009).
    """
    D, k = n_features, n_components
    n_passes = 1 + 2 * niter + 1  # initial + 2 per iter + final B computation

    # Centering (done inside pca_lowrank with center=True)
    to_center = N * D * TO["mac"] + D * TO["div"]

    # Matrix multiplications through A (the dominant algorithmic cost)
    to_matmuls = n_passes * N * D * k * TO["mac"]

    # QR factorizations: counted as Householder cost 2·m·n²
    # (1 + niter) QR of N×q + niter QR of D×q
    to_qr_big = (1 + niter) * 2 * N * k * k * TO["mac"]
    to_qr_small = niter * 2 * D * k * k * TO["mac"]

    # Small SVD of q×D matrix (lower bound)
    to_svd_small = 6 * k * D * D * TO["mac"]

    # Final back-projection U = Q @ U_small
    to_backproj = N * k * k * TO["mac"]

    to_compute = B * (to_center + to_matmuls + to_qr_big + to_qr_small +
                      to_svd_small + to_backproj)
    to_memory = B * (N * D + D * k) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_fastica(N: int, B: int = 1, n_features: int = 16, n_components: int = 4,
               max_iter: int = 50, **kw) -> Dict[str, float]:
    """
    FastICA (Hyvärinen & Oja, 2000), T iterations:
      Whitening (once):
        - Covariance: D²·N MACs
        - Eigendecomp: D³ MACs
        - Whitening: k·D·N MACs + k divisions + k sqrt

      Per ICA iteration:
        - WX = W @ X_white: k²·N MACs
        - g = tanh(WX): k·N tanh
        - g' = 1 - g²: 2·k·N MACs
        - W_new = g @ X^T / N - mean(g') * W: k²·N + k² MACs + k² divs
        - QR: k³ MACs

    Total: whitening + T · (k²N + kN·tanh + 2kN + k²N + k³) MACs
    """
    D, k, T = n_features, n_components, max_iter

    # Whitening
    to_cov = D * D * N * TO["mac"]
    to_eig = D * D * D * TO["mac"]
    to_whiten = k * D * N * TO["mac"]
    to_sqrt_inv = k * TO["sqrt"] + k * TO["div"]

    # ICA iterations
    to_wx = T * k * k * N * TO["mac"]
    to_tanh = T * k * N * TO["tanh"]
    to_gprime = T * 2 * k * N * TO["mac"]
    to_w_update = T * (k * k * N + k * k) * TO["mac"]
    to_w_div = T * k * k * TO["div"]
    to_qr = T * k * k * k * TO["mac"]

    to_compute = B * (to_cov + to_eig + to_whiten + to_sqrt_inv +
                      to_wx + to_tanh + to_gprime + to_w_update + to_w_div + to_qr)
    to_memory = B * (D * N + k * N + k * k) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_nmf(N: int, B: int = 1, n_features: int = 64, n_components: int = 8,
           max_iter: int = 50, **kw) -> Dict[str, float]:
    """
    NMF multiplicative update (Lee & Seung, 2001), T iterations:
      V: (D × N), W: (D × k), H: (k × N)

      Per iteration:
        H update:
          W^T V: k·D·N MACs
          W^T W: k·D·k MACs
          (W^T W) H: k·k·N MACs
          H *= num/denom: k·N divisions

        W update:
          V H^T: D·k·N MACs
          W (H H^T): D·k·k + k·k·N MACs
          W *= num/denom: D·k divisions

      Per iteration: ~4kDN + 2k²N + 2Dk² MACs + (kN + Dk) divisions
    """
    D, k, T = n_features, n_components, max_iter

    per_iter_mac = (2 * k * D * N + k * D * k + k * k * N +   # H update
                    D * k * N + D * k * k + k * k * N)          # W update
    per_iter_div = k * N + D * k

    to_compute = B * T * (per_iter_mac * TO["mac"] + per_iter_div * TO["div"])
    to_memory = B * (D * N + D * k + k * N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


# =========================================================================
# CATEGORY 7: COMPRESSION
# =========================================================================

def to_jpeg_v0(N: int, B: int = 1, quality: int = 50, **kw) -> Dict[str, float]:
    """
    [ORIGINAL — preserved for comparison]
    JPEG with optimized 29-MAC DCT-8 and IDCT (did not match implementation).
    """
    n_blocks = max(1, N // 64)
    to_dct_2d = n_blocks * 4 * 8 * 29 * TO["mac"]
    to_quant = N * TO["div"]
    to_dequant = N * TO["mac"]
    to_compute = B * (to_dct_2d + to_quant + to_dequant)
    to_memory = B * (N + N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_jpeg(N: int, B: int = 1, quality: int = 50, **kw) -> Dict[str, float]:
    """
    JPEG pipeline (as benchmarked):
      Signal of length N reshaped to sqrt(N) × sqrt(N) image.
      n_blocks = (side/8)² blocks of 8×8.

      Per block:
        - 2D DCT via matmul: basis(8×8) @ block(8×8) @ basis.T(8×8)
          = 2 × 8³ = 1,024 MACs per block
        - Quantization: 64 divisions (dct_coeffs / Q)
        - Rounding: 64 round ops (not counted as FP MAC)
        - Dequantization: 64 multiplications (quantized * Q)

      Total: n_blocks × (1024 + 64 + 64) MACs + n_blocks × 64 divisions

      Note: Implementation does NOT include IDCT.
      Note: cuBLAS overhead on 8×8 batched matmul is 16–65× the
      mathematical op count (Finding F-008), absorbed by α_c.
    """
    import math as _math
    side = int(_math.sqrt(N))
    side = (side // 8) * 8
    n_blocks = max(1, (side // 8) ** 2)
    pixels = n_blocks * 64

    to_matmul = n_blocks * 2 * 8 * 8 * 8 * TO["mac"]  # 2 matmuls per block
    to_quant = pixels * TO["div"]                       # division by Q
    to_dequant = pixels * TO["mac"]                     # multiply by Q
    to_compute = B * (to_matmul + to_quant + to_dequant)
    to_memory = B * (pixels + pixels) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_mdct_v0(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    [ORIGINAL — preserved for comparison]
    MDCT via FFT-based approach (does NOT match implementation).
    """
    M = N // 2
    f = fft_tos(M)
    to_window = N * TO["mac"]
    to_mdct_fft = f["to_compute"] + M * 6 * TO["mac"]
    to_quant = M * TO["div"]
    to_dequant = M * TO["mac"]
    to_imdct = f["to_compute"] + M * 6 * TO["mac"]
    to_ola = N * TO["mac"]
    to_compute = B * (to_window + to_mdct_fft + to_quant + to_dequant + to_imdct + to_ola)
    to_memory = B * (N + N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_mdct(N: int, B: int = 1, frame_size: int = 512, **kw) -> Dict[str, float]:
    """
    MDCT audio codec pipeline (as benchmarked):
      Python loop over F frames, each of length 2*frame_size.
      F = min((N - 2*frame_size) // frame_size + 1, 50)

      Per frame:
        - Window: 2*frame_size multiplications
        - MDCT via matmul: basis(frame_size × 2*frame_size) @ frame → frame_size·2*frame_size MACs
        - Power spectrum: |X|² → 3*frame_size MACs
        - Psychoacoustic masking: frame_size log + frame_size mul (bark) + frame_size exp
        - Quantization: frame_size divisions + frame_size round

    Total per frame: 2*fs + fs*2*fs + 3*fs + fs + fs MACs + fs log + fs exp + fs div

    Validated against Nsight Compute:
      N=4096, frame_size=512: formula gives 3,702,272 ops vs 3,895,808 measured (1.05 ratio)
    """
    fs = frame_size
    n_frames = max(1, min((N - 2 * fs) // fs + 1, 50))

    # Per-frame compute
    to_window = 2 * fs * TO["mac"]
    to_matmul = fs * 2 * fs * TO["mac"]         # basis @ frame
    to_power = 3 * fs * TO["mac"]                # |X|^2
    to_log = fs * TO["exp"]                       # log (same cost class as exp)
    to_bark_mul = fs * TO["mac"]                  # mask * bark_scale
    to_exp = fs * TO["exp"]                       # exp(mask)
    to_quant = fs * TO["div"]                     # X / scale

    per_frame = to_window + to_matmul + to_power + to_log + to_bark_mul + to_exp + to_quant
    to_compute = B * n_frames * per_frame
    to_memory = B * (N + N) * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


# =========================================================================
# CATEGORY 8: ML-ENHANCED SIGNAL PROCESSING
# =========================================================================

def to_cnn_denoiser_v0(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    [ORIGINAL — preserved for comparison]
    1D CNN denoiser with WRONG architecture parameters.
    Had Conv1d(1,16,7) + Conv1d(16,32,5) + Conv1d(32,1,3) = 2768N MACs.
    """
    to_conv1 = N * 1 * 16 * 7 * TO["mac"]
    to_relu1 = 16 * N * TO["relu"]
    to_conv2 = N * 16 * 32 * 5 * TO["mac"]
    to_relu2 = 32 * N * TO["relu"]
    to_conv3 = N * 32 * 1 * 3 * TO["mac"]
    to_compute = B * (to_conv1 + to_relu1 + to_conv2 + to_relu2 + to_conv3)
    to_memory = B * (N + N) * TO["mem_hbm"]
    to_mem_weights = (1*16*7 + 16*32*5 + 32*1*3) * TO["mem_sram"]
    to_memory += to_mem_weights
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_cnn_denoiser(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    1D CNN denoiser (as benchmarked):
      CNN1DDenoiser(channels=32, n_layers=3):
        Conv1d(1, 32, 7, padding=3) → ReLU
        Conv1d(32, 32, 7, padding=3) → ReLU
        Conv1d(32, 1, 7, padding=3)

      Layer 1: N · 1 · 32 · 7 = 224N MACs + 32N ReLU
      Layer 2: N · 32 · 32 · 7 = 7168N MACs + 32N ReLU
      Layer 3: N · 32 · 1 · 7 = 224N MACs

    Total: 7616N MACs + 64N ReLU

    Validated against Nsight Compute:
      N=4096: formula gives 31,195,136 MACs vs 31,460,992 FMA (1.009 ratio)
    """
    C = 32  # channels
    K = 7   # kernel size for all layers
    to_conv1 = N * 1 * C * K * TO["mac"]
    to_relu1 = C * N * TO["relu"]
    to_conv2 = N * C * C * K * TO["mac"]
    to_relu2 = C * N * TO["relu"]
    to_conv3 = N * C * 1 * K * TO["mac"]
    to_compute = B * (to_conv1 + to_relu1 + to_conv2 + to_relu2 + to_conv3)
    to_memory = B * (N + N) * TO["mem_hbm"]  # input + output
    # Weights are small enough for SRAM
    to_mem_weights = (1*C*K + C*C*K + C*1*K) * TO["mem_sram"]
    to_memory += to_mem_weights
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_lstm_denoiser_v0(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    [ORIGINAL — preserved for comparison]
    LSTM denoiser with WRONG parameters: H=64, num_layers=2.
    Actual implementation: H=128, num_layers=1.
    """
    H = 64
    to_lstm1 = N * 4 * (1 + H) * H * TO["mac"]
    to_sig1 = N * 3 * H * TO["sigmoid"]
    to_tanh1 = N * H * TO["tanh"]
    to_elem1 = N * 4 * H * TO["mac"]
    to_lstm2 = N * 4 * (H + H) * H * TO["mac"]
    to_sig2 = N * 3 * H * TO["sigmoid"]
    to_tanh2 = N * H * TO["tanh"]
    to_elem2 = N * 4 * H * TO["mac"]
    to_linear = N * H * TO["mac"]
    to_compute = B * (to_lstm1 + to_sig1 + to_tanh1 + to_elem1 +
                      to_lstm2 + to_sig2 + to_tanh2 + to_elem2 + to_linear)
    n_weights = 4 * (1 + H) * H + 4 * (H + H) * H + H * 1
    to_memory = B * N * TO["mem_hbm"] + n_weights * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_lstm_denoiser(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    LSTM denoiser (as benchmarked):
      LSTMDenoiser(hidden_size=128):
        LSTM(input_size=1, hidden_size=128, num_layers=1, batch_first=True)
        + Linear(128, 1)

    LSTM per step (1 layer, input_size=1, H=128):
      4 gates × (input_size + hidden_size) × hidden_size MACs
      = 4 × 129 × 128 = 66,048 MACs/step
      + 3 × 128 sigmoid + 128 tanh + 4 × 128 element-wise MACs
    Linear: 128 MACs/step

    Total per step: 66,048 + 512 + 128 = 66,688 MACs + 384 sigmoid + 128 tanh
    Over N steps: N × above

    Validated against Nsight Compute:
      N=1024: gate matmuls = 67,633,152 vs 68,682,752 FMA (1.016 ratio)
    """
    H = 128
    # Single LSTM layer
    to_lstm = N * 4 * (1 + H) * H * TO["mac"]
    to_sig = N * 3 * H * TO["sigmoid"]
    to_tanh = N * H * TO["tanh"]
    to_elem = N * 4 * H * TO["mac"]  # element-wise gate ops (c_t, h_t)

    # Linear output
    to_linear = N * H * TO["mac"]

    to_compute = B * (to_lstm + to_sig + to_tanh + to_elem + to_linear)
    # Weight memory: loaded from HBM
    n_weights = 4 * (1 + H) * H + H * 1
    to_memory = B * N * TO["mem_hbm"] + n_weights * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


def to_transformer_denoiser(N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    Transformer denoiser (as benchmarked):
      Linear(1, 64) → 2-layer TransformerEncoder(d_model=64, nhead=4, d_ff=128) → Linear(64, 1)

    Per transformer layer:
      - Q, K, V projections: 3 · N · 64 · 64 MACs
      - Attention scores: 4 heads × N × N × 16 MACs  (d_head = 64/4 = 16)
      - Softmax: 4 · N · N softmax elements
      - Attention × V: 4 × N × N × 16 MACs
      - Output projection: N × 64 × 64 MACs
      - FFN: N × 64 × 128 + N × 128 × 64 MACs
      - 2× LayerNorm: 2 × N × 64 × 12,000 TOs
      - ReLU: N × 128

    Self-attention is O(N²) — this is visible in the benchmarks.
    """
    d, h, d_ff, L = 64, 4, 128, 2
    d_k = d // h  # 16

    # Input/output projections
    to_proj_in = N * 1 * d * TO["mac"]
    to_proj_out = N * d * 1 * TO["mac"]

    # Per transformer layer
    to_qkv = 3 * N * d * d * TO["mac"]
    to_attn_score = h * N * N * d_k * TO["mac"]
    to_softmax = h * N * N * TO["softmax"]
    to_attn_v = h * N * N * d_k * TO["mac"]
    to_out_proj = N * d * d * TO["mac"]
    to_ffn = N * d * d_ff * TO["mac"] + N * d_ff * d * TO["mac"]
    to_relu = N * d_ff * TO["relu"]
    to_layernorm = 2 * N * d * 12_000  # layernorm TOs from TOMLTransformers

    to_per_layer = (to_qkv + to_attn_score + to_softmax + to_attn_v +
                    to_out_proj + to_ffn + to_relu + to_layernorm)

    to_compute = B * (to_proj_in + L * to_per_layer + to_proj_out)
    # Weights in HBM
    n_weights = 1 * d + 3 * d * d * L + d * d * L + (d * d_ff + d_ff * d) * L + d * 1
    to_memory = B * N * TO["mem_hbm"] + n_weights * TO["mem_hbm"]
    return {"to_compute": to_compute, "to_memory": to_memory,
            "to_total": to_compute + to_memory}


# =========================================================================
# SEQUENTIAL STEP COUNTS
# =========================================================================
# GPU kernel launch overhead (~10-20µs per launch on modern GPUs) is
# negligible for algorithms that execute as a single fused kernel (FFT,
# conv1d, matmul) but dominates for algorithms with Python for-loops
# where each iteration launches tiny kernels. This is a real hardware
# cost not captured by operation counting.
#
# n_seq_steps(alg, N, B) returns the number of sequential Python loop
# iterations. Each iteration incurs ~5-20 GPU kernel launches worth of
# overhead (kernel dispatch, Python-CUDA synchronization, etc.).
# The three-parameter model E = α_c·TO_compute + α_m·TO_memory + α_o·n_seq
# captures this overhead with a per-step energy cost α_o.

def _seq_lms(N, B=1, filter_length=32, **kw):
    M = filter_length
    return B * max(0, min(N, M + 200) - M)

def _seq_nlms(N, B=1, filter_length=32, **kw):
    return _seq_lms(N, B, filter_length)

def _seq_rls(N, B=1, filter_length=32, **kw):
    M = filter_length
    return B * max(0, min(N, M + 100) - M)

def _seq_apa(N, B=1, filter_length=32, proj_order=4, **kw):
    M, P = filter_length, proj_order
    return B * max(0, min(N, M + P + 100) - (M + P))

def _seq_kalman(N, B=1, **kw):
    return B * min(N, 200)

def _seq_ekf(N, B=1, **kw):
    return B * min(N, 200)

def _seq_ukf_v0(N, B=1, state_dim=4, **kw):
    """[ORIGINAL — preserved for comparison]
    Counted inner-loop iterations (sigma generation + covariance accumulation)
    as separate sequential steps. This was inconsistent with all other _seq_*
    functions, which count only outer-loop iterations."""
    T = min(N, 100)
    n_sigma = 2 * state_dim + 1
    return B * T * (1 + state_dim + n_sigma)  # ~14 inner ops per outer step


def _seq_ukf(N, B=1, state_dim=4, **kw):
    """UKF outer-loop iterations only.

    The UKF implementation has inner Python loops (sigma generation: n_s
    iterations, covariance accumulation: n_sigma-1 iterations), but these
    produce lightweight kernel launches (simple assignment/add) whose overhead
    is fundamentally different from outer-loop iterations (which include
    cholesky, matmul, bmm).

    Counting only outer iterations is consistent with all other _seq_*
    functions (Kalman, EKF, LMS, RLS, etc.) which count outer loops only
    despite having multiple kernel launches per iteration.

    Evidence (from diagnose_alpha_o.py):
      - Kernel launches per outer iter: UKF ~37, Kalman ~17 (ratio 2.2x)
      - Measured overhead per outer iter: UKF 15,610 uJ, Kalman 8,413 uJ
        (ratio 1.86x on 4090, 2.21x on A100) — consistent with kernel count.
      - Old 14x multiplier gave UKF 1,122 uJ/step vs Kalman 8,413 uJ/step
        (7.5x lower), which is physically inconsistent: both use the same
        Python-to-CUDA dispatch path on the same hardware.
    """
    T = min(N, 100)
    return B * T  # outer-loop iterations only

def _seq_particle(N, B=1, **kw):
    return B * min(N, 200)

def _seq_fastica(N, B=1, max_iter=50, **kw):
    return B * max_iter

def _seq_nmf(N, B=1, max_iter=50, **kw):
    return B * max_iter

# Cache torchaudio availability at import time (for local machine)
try:
    import torchaudio.functional as _AF_check  # noqa: F401
    _LOCAL_HAS_TORCHAUDIO = True
except (ImportError, OSError):
    _LOCAL_HAS_TORCHAUDIO = False


def _seq_iir(N, B=1, has_torchaudio=None, **kw):
    # torchaudio.lfilter = 0 steps (fused C++ kernel)
    # Pure-torch fallback = N steps (Python loop)
    # has_torchaudio: None=auto-detect, True/False=override
    if has_torchaudio is None:
        has_torchaudio = _LOCAL_HAS_TORCHAUDIO
    if has_torchaudio:
        return 0  # fused kernel, no Python loop
    else:
        return B * N  # Python loop fallback

def _seq_pca_v0(N, B=1, **kw):
    """[ORIGINAL — preserved for comparison]
    Counted batch-element loop as sequential. But pca_lowrank is a single
    C++ function call — no Python dispatch overhead per internal kernel."""
    return B

def _seq_pca(N, B=1, **kw):
    # pca_lowrank is a C++ function (like cuSOLVER SVD).
    # Internal kernel launches happen inside PyTorch C++ code,
    # not through Python dispatch. Overhead is cuSOLVER overhead,
    # absorbed by alpha_c (same regime as SVD, which has seq_steps=0).
    return 0

def _seq_zero(N, B=1, **kw):
    return 0

SEQ_STEPS = {
    # Transforms — all GPU-parallel (cuFFT, conv1d, matmul)
    "fft": _seq_zero, "direct_dft": _seq_zero, "dct": _seq_zero,
    "dst": _seq_zero, "dwt_haar": _seq_zero, "dwt_db4": _seq_zero,
    "stft": _seq_zero, "hilbert": _seq_zero,
    # Filters — parallel except IIR
    "fir_direct": _seq_zero, "fir_fft": _seq_zero,
    "iir_butter4": _seq_iir,
    "median": _seq_zero, "savgol": _seq_zero, "wiener": _seq_zero,
    "matched_filter": _seq_zero, "filterbank_32ch": _seq_zero,
    # Adaptive — all sequential Python loops
    "lms": _seq_lms, "nlms": _seq_nlms, "rls": _seq_rls,
    "apa_p4": lambda N, B=1, **kw: _seq_apa(N, B, filter_length=32, proj_order=4),
    # Estimation — all sequential
    "kalman": _seq_kalman, "ekf": _seq_ekf, "ukf": _seq_ukf,
    "particle_1k": _seq_particle,
    # Spectral — all vectorized
    "periodogram": _seq_zero, "welch": _seq_zero,
    "music": _seq_zero, "esprit": _seq_zero,
    # Decomposition — SVD/PCA parallel, ICA/NMF iterative
    "svd": _seq_zero, "pca": _seq_pca,
    "fastica": _seq_fastica, "nmf": _seq_nmf,
    # Compression — MDCT has Python loop, JPEG is parallel
    "jpeg_q50": _seq_zero, "mdct_audio": lambda N, B=1, **kw: B * max(1, min((N - 2*512) // 512 + 1, 50)),
    # ML — all fused kernels (cuDNN LSTM, torch.nn)
    "cnn_denoiser": _seq_zero, "lstm_denoiser": _seq_zero,
    "transformer_denoiser": _seq_zero,
}


# =========================================================================
# KERNEL LAUNCHES PER OUTER ITERATION
# =========================================================================
# Each torch operation on a CUDA tensor that produces a new tensor (not a
# view) triggers at least one CUDA kernel launch. Views (slice, unsqueeze,
# squeeze, transpose, reshape) are free. Counts derived from source code
# analysis of algorithms/*.py implementations.
#
# The sequential overhead alpha_o captures per-kernel-launch dispatch cost:
#   - Python-to-CUDA dispatch latency (~5-20 us per launch)
#   - GPU idle power during dispatch gaps
#   - Python interpreter overhead between operations
#
# Using kernel launches instead of loop iterations reduces per-algorithm
# overhead variation from 8x to 2.5x on RTX 4090 (CV 34% vs 130%),
# and improves 3-parameter model r^2 from 0.569 to 0.932 on RTX 4090.

KERNELS_PER_ITER = {
    # Adaptive (algorithms/adaptive.py)
    "lms":       7,   # flip, mul+sum, sub, scalar*tensor*tensor+tensor
    "nlms":     11,   # LMS(7) + pow+sum+add(norm:3) + div(1)
    "rls":      12,   # 4x bmm, inv, 2x div, 3x add/sub, flip, mul
    "apa_p4":   15,   # P*flip+stack(5), 2x bmm, solve, eye+mul+add(3), bmm+mul+add(3)
    # Estimation (algorithms/estimation.py)
    "kalman":   15,   # 7x matmul, inv, 4x add/sub, 2x indexing-copy
    "ekf":      22,   # kalman(15) + sin, cos, diag_embed, bmm upgrades
    "ukf":      43,   # cholesky+alloc(3), 4x2 sigma inner(8), propagate(4),
                      # mean(4), diff(1), init_cov(3), 8x2 cov inner(16),
                      # symmetry(2), copy(2)
    "particle_1k": 20, # randn+mul+add(3), sub+pow+sum(3), mul(1), max+sub(2),
                       # exp(1), sum+div(2), cumsum(1), arange+rand+add+div(4),
                       # searchsorted(1), clamp(1), gather(1)
    # Decomposition (algorithms/decomposition.py)
    "fastica":  13,   # bmm(1), tanh(1), pow+sub(2), bmm+div+mean+mul+sub(5),
                      # transpose+mul(1), QR(1), 2 extra
    "nmf":      14,   # 4x bmm(4), 2x add_eps(2), 2x mul(2), 2x div(2),
                      # 2x transpose(0-view), 4x misc(4)
    "pca":       0,   # pca_lowrank is C++, not Python dispatch (seq_steps=0)
    # Compression (algorithms/compression.py)
    "mdct_audio": 11, # mul(window:1), matmul(2), abs+pow(2), add+log+mul(3),
                      # exp(1), clamp(1), div+round(2) -> ~11-12
    # Filters
    "iir_butter4": 5, # Python fallback: ~5 ops per time step
}


def get_seq_steps(algorithm: str, N: int, B: int = 1, **kw) -> int:
    """Return total CUDA kernel launches for Python-loop sequential algorithms.

    Computed as: outer_loop_iterations × kernel_launches_per_iteration.

    This replaced the previous iteration-only counting (preserved in _v0
    functions). Kernel launches are the physical event that incurs
    Python-to-CUDA dispatch overhead, making them the correct unit for
    the sequential overhead term alpha_o.

    Kwargs are passed through to the algorithm-specific function.
    Use has_torchaudio=True/False to override IIR detection for cross-GPU analysis.
    """
    if algorithm not in SEQ_STEPS:
        return 0
    outer_iters = SEQ_STEPS[algorithm](N, B, **kw)
    kpl = KERNELS_PER_ITER.get(algorithm, 1)  # default 1 if not in table
    return outer_iters * kpl


def get_fused_steps(algorithm: str, N: int, B: int = 1, **kw) -> int:
    """Return number of fused-sequential timesteps.

    Fused-sequential algorithms run as a single C++/CUDA kernel that
    processes N timesteps serially (no Python dispatch overhead). The
    energy cost is proportional to N at low GPU utilization, captured
    by a separate coefficient alpha_f.

    Currently only cuDNN LSTM at B<=1 is classified as fused-sequential.
    At B>1, batch parallelism allows the GPU to hide sequential latency.

    IIR with torchaudio is also fused-sequential, but with ~5000x less
    compute per step than LSTM. A single alpha_f cannot cover both,
    so IIR remains as seq_steps=0 (3 outlier points on 4090).

    Physical mechanism (distinct from Python-loop alpha_o):
      - No Python interpreter or CUDA driver dispatch overhead
      - Cost is serial execution time x GPU power at low SM utilization
      - alpha_o/alpha_f ratio: ~14x on 4090, ~1x on A100 (Finding F-014)
    """
    if algorithm == "lstm_denoiser" and B <= 1:
        return N  # cuDNN LSTM at B=1 processes N timesteps serially
    return 0


# =========================================================================
# REGISTRY
# =========================================================================

TO_MODELS = {
    # Category 1: Transforms
    "fft":              to_fft,
    "direct_dft":       to_direct_dft,
    "dct":              to_dct,
    "dst":              to_dst,
    "dwt_haar":         lambda N, B=1, **kw: to_dwt(N, B, wavelet="haar"),
    "dwt_db4":          lambda N, B=1, **kw: to_dwt(N, B, wavelet="db4"),
    "stft":             to_stft,
    "hilbert":          to_hilbert,
    # Category 2: Filters
    "fir_direct":       to_fir_direct,
    "fir_fft":          to_fir_fft,
    "iir_butter4":      lambda N, B=1, **kw: to_iir(N, B, order=4),
    "median":           to_median,
    "savgol":           to_savgol,
    "wiener":           to_wiener,
    "matched_filter":   to_matched,
    "filterbank_32ch":  lambda N, B=1, **kw: to_filterbank(N, B, n_channels=32, filter_length=64),
    # Category 3: Adaptive
    "lms":              to_lms,
    "nlms":             to_nlms,
    "rls":              to_rls,
    "apa_p4":           lambda N, B=1, **kw: to_apa(N, B, filter_length=32, proj_order=4),
    # Category 4: Estimation
    "kalman":           to_kalman,
    "ekf":              to_ekf,
    "ukf":              to_ukf,
    "particle_1k":      lambda N, B=1, **kw: to_particle(N, B, n_particles=1000),
    # Category 5: Spectral
    "periodogram":      to_periodogram,
    "welch":            to_welch,
    "music":            to_music,
    "esprit":           to_esprit,
    # Category 6: Decomposition
    "svd":              to_svd,
    "pca":              to_pca,
    "fastica":          to_fastica,
    "nmf":              to_nmf,
    # Category 7: Compression
    "jpeg_q50":         to_jpeg,
    "mdct_audio":       to_mdct,
    # Category 8: ML-enhanced
    "cnn_denoiser":     to_cnn_denoiser,
    "lstm_denoiser":    to_lstm_denoiser,
    "transformer_denoiser": to_transformer_denoiser,
}


def predict_to(algorithm: str, N: int, B: int = 1, **kw) -> Dict[str, float]:
    """
    Predict total transistor operations for an algorithm.

    Args:
        algorithm: Algorithm name (must be in TO_MODELS registry).
        N: Signal length.
        B: Batch size.
        **kw: Algorithm-specific parameters.

    Returns:
        Dict with to_compute, to_memory, to_total.
    """
    if algorithm not in TO_MODELS:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         f"Available: {list(TO_MODELS.keys())}")
    return TO_MODELS[algorithm](N, B, **kw)


# =========================================================================
# QUICK VALIDATION
# =========================================================================

if __name__ == "__main__":
    print(f"{'Algorithm':25s} {'N':>6s} {'B':>5s} {'TO_compute':>14s} {'TO_memory':>14s} "
          f"{'TO_total':>14s} {'MCER':>8s}")
    print("-" * 95)

    test_cases = [
        ("fft", 4096, 1),
        ("direct_dft", 1024, 1),
        ("fir_direct", 4096, 1),
        ("fir_fft", 4096, 1),
        ("iir_butter4", 4096, 1),
        ("wiener", 4096, 1),
        ("lms", 4096, 1),
        ("rls", 4096, 1),
        ("kalman", 4096, 1),
        ("ukf", 4096, 1),
        ("welch", 4096, 1),
        ("music", 1024, 1),
        ("svd", 1024, 1),
        ("pca", 1024, 1),
        ("fastica", 1024, 1),
        ("nmf", 1024, 1),
        ("cnn_denoiser", 4096, 1),
        ("lstm_denoiser", 4096, 1),
        ("transformer_denoiser", 4096, 1),
    ]

    for alg, N, B in test_cases:
        r = predict_to(alg, N, B)
        mcer = r["to_memory"] / r["to_compute"] if r["to_compute"] > 0 else float("inf")
        print(f"  {alg:23s} {N:>6d} {B:>5d} {r['to_compute']:>14.2e} {r['to_memory']:>14.2e} "
              f"{r['to_total']:>14.2e} {mcer:>8.2f}")
