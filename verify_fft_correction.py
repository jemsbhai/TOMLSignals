"""
Verify split-radix FFT correction against NCU ground truth.
Compares old (Cooley-Tukey) vs new (split-radix) predictions.
"""
import json
import sys
sys.path.insert(0, ".")
from shared.to_model import fft_tos, fft_tos_cooley_tukey, TO, _next_pow2, _log2

# Load NCU data
with open("data/ncu_profiles/ncu_summary.json") as f:
    ncu_data = {r["algorithm"]: r for r in json.load(f)}

# Algorithms to verify
algorithms = [
    "fft", "direct_dft", "dct", "dst", "stft", "hilbert",
    "fir_fft", "matched_filter", "wiener", "periodogram", "welch",
]

print(f"{'Algorithm':20s} {'NCU FP32':>12s} {'Old pred':>12s} {'New pred':>12s} "
      f"{'Old ratio':>10s} {'New ratio':>10s}")
print("-" * 80)

for alg in algorithms:
    if alg not in ncu_data:
        print(f"  {alg:18s}  -- not in NCU data --")
        continue

    ncu = ncu_data[alg]
    actual = ncu["fp32_total"]
    N = ncu["N"]

    # Compute old and new FFT op counts for comparison
    # We need the raw MAC count (before multiplying by TO["mac"])
    # Old: N/2 * log2(N) * 10
    old_fft_ops = (N // 2) * _log2(N) * 10
    # New: 4N*log2(N) - 6N + 8
    new_fft_ops = 4 * N * _log2(N) - 6 * N + 8

    old_pred = ncu["to_pred_macs"]

    if alg == "direct_dft":
        # Old: N^2 * 8, New: N^2 * 4
        new_pred = N * N * 4
    elif alg == "fft":
        new_pred = new_fft_ops
    elif alg == "dct":
        # DCT = FFT(N) + N*6 (twiddle) + N*1 (scale)
        new_pred = new_fft_ops + N * 6 + N * 1
    elif alg == "dst":
        # DST = 2N (negate) + FFT(L) + N (extract), L=2*(N+1)
        # fft_tos now pads non-power-of-2 to next power of 2
        L = 2 * (N + 1)
        L_eff = L if (L & (L-1)) == 0 else _next_pow2(L)
        new_fft_L = 4 * L_eff * _log2(L_eff) - 6 * L_eff + 8
        new_pred = 2 * N + new_fft_L + N
    elif alg == "stft":
        # STFT: F frames, each = W window muls + FFT(W)
        W = 256; hop = 128
        F = max(1, (N - W) // hop + 1)
        new_fft_W = 4 * W * _log2(W) - 6 * W + 8
        new_pred = F * (W + new_fft_W)
    elif alg == "hilbert":
        # 2*FFT(N) + N
        new_pred = 2 * new_fft_ops + N
    elif alg in ("fir_fft", "matched_filter"):
        # 2*FFT(M) + M*6, M = next_pow2(N+K-1), K=64
        M = _next_pow2(N + 64 - 1)
        new_fft_M = 4 * M * _log2(M) - 6 * M + 8
        new_pred = 2 * new_fft_M + M * 6
    elif alg == "wiener":
        # 2*FFT(N) + 3N + N + N(div as MAC approx) + 6N
        new_pred = 2 * new_fft_ops + 3 * N + N + N + 6 * N  # div counted separately in TOs but as MACs here
    elif alg == "periodogram":
        # FFT(N) + 3N + N(div)
        new_pred = new_fft_ops + 3 * N + N
    elif alg == "welch":
        # K*(W + FFT(W) + 3W) + W
        W = 256; H = 128
        K = max(1, (N - W) // H + 1)
        new_fft_W = 4 * W * _log2(W) - 6 * W + 8
        new_pred = K * (W + new_fft_W + 3 * W) + W
    else:
        new_pred = old_pred  # fallback

    old_ratio = actual / old_pred if old_pred > 0 else float('inf')
    new_ratio = actual / new_pred if new_pred > 0 else float('inf')

    print(f"  {alg:18s} {actual:>12,.0f} {old_pred:>12,.0f} {new_pred:>12,.0f} "
          f"{old_ratio:>10.3f} {new_ratio:>10.3f}")

print()
print("Target: new ratios closer to 1.0 than old ratios.")
print("Perfect match = 1.000")
