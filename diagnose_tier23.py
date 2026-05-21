"""
Diagnose Tier 2/3 TO model discrepancies against NCU ground truth.

Algorithms: SVD (3.41x), CNN denoiser (2.80x), PCA (65.73x),
            JPEG (69.14x), MDCT (14.63x), Median (0x FP32)
"""
import json
import math

# Load NCU data
with open("data/ncu_profiles/ncu_summary.json") as f:
    ncu_data = {r["algorithm"]: r for r in json.load(f)}


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =====================================================================
header("CNN DENOISER — Architecture parameter mismatch")
# =====================================================================

ncu = ncu_data["cnn_denoiser"]
N = ncu["N"]  # 4096

# TO model (WRONG): Conv1d(1,16,7) + Conv1d(16,32,5) + Conv1d(32,1,3)
old_macs = (1*16*7 + 16*32*5 + 32*1*3) * N
print(f"  Old TO model (wrong arch):    {old_macs:>14,} MACs")

# Implementation (CORRECT): CNN1DDenoiser(channels=32, n_layers=3)
# Conv1d(1,32,7) + Conv1d(32,32,7) + Conv1d(32,1,7)
new_macs = (1*32*7 + 32*32*7 + 32*1*7) * N
print(f"  Corrected (actual arch):      {new_macs:>14,} MACs")
print(f"  NCU FP32 FMA:                 {ncu['fp32_fma']:>14,.0f}")
print(f"  NCU FP32 total:               {ncu['fp32_total']:>14,.0f}")
print(f"  Old ratio:                    {ncu['fp32_total']/old_macs:>14.3f}")
print(f"  New ratio (FMA):              {ncu['fp32_fma']/new_macs:>14.3f}")
print(f"  New ratio (total):            {ncu['fp32_total']/new_macs:>14.3f}")
print(f"\n  VERDICT: Straightforward fix — TO model had wrong Conv1d parameters.")
print(f"  Extra {ncu['fp32_total']-ncu['fp32_fma']:,} non-FMA ops = {(ncu['fp32_total']-ncu['fp32_fma'])/N:.0f}N")
print(f"  likely bias addition ({ncu['fp32_mul']:,} muls = {ncu['fp32_mul']/N:.0f}N for weight scaling)")


# =====================================================================
header("SVD — Iterative algorithm, 6ND² is lower bound")
# =====================================================================

ncu = ncu_data["svd"]
N_svd, D = ncu["N"], 64  # 1024 x 64

# Current model: 6*N*D^2
old_macs = 6 * N_svd * D * D
# Better estimates from matrix computation literature
bidiag = 4 * N_svd * D * D - 4 * D**3 // 3
backaccum = 2 * N_svd * D * D + 2 * D**3

print(f"  Matrix shape: {N_svd} x {D}")
print(f"  NCU kernels: {ncu['n_kernels']}")
print(f"  NCU FP32 FMA:                 {ncu['fp32_fma']:>14,.0f}")
print(f"  NCU FP32 total:               {ncu['fp32_total']:>14,.0f}")
print(f"")
print(f"  --- Operation count estimates ---")
print(f"  Current TO (6ND²):            {old_macs:>14,}")
print(f"  Bidiagonalization (4ND²-4D³/3):{bidiag:>13,}")
print(f"  Back-accumulation (2ND²+2D³): {backaccum:>14,}")
print(f"  Bidiag + BackAccum:           {bidiag+backaccum:>14,}")
print(f"")
print(f"  Ratios to NCU FP32 total:")
print(f"    Current 6ND²:               {ncu['fp32_total']/old_macs:>14.2f}x")
print(f"    Bidiag+BackAccum:           {ncu['fp32_total']/(bidiag+backaccum):>14.2f}x")
print(f"")
print(f"  Implied total cost:           ~{ncu['fp32_total']/(N_svd*D*D):.1f} ND²")
print(f"  {ncu['n_kernels']} kernels confirms iterative QR/D&C SVD with many small launches.")
print(f"  VERDICT: 6ND² captures bidiag only. Full pipeline is ~{ncu['fp32_total']/(N_svd*D*D):.0f}ND²")
print(f"  including QR iteration + singular vector accumulation.")


# =====================================================================
header("PCA — torch.pca_lowrank power iterations undercounted")
# =====================================================================

ncu = ncu_data["pca"]
N_pca, D_pca, k = ncu["N"], 64, 8  # 1024 x 64, q=8, niter=2

# Current model
old_macs = N_pca*D_pca + N_pca*D_pca*k + N_pca*k*k + k*k*k + D_pca*k*k
# = 65536 + 524288 + 65536 + 512 + 4096 = 659,968

# Corrected: trace through torch.pca_lowrank(A, q=8, center=True, niter=2)
# svd_lowrank internals:
niter = 2
# Initial: A @ R → N*D*q MACs
init_proj = N_pca * D_pca * k
init_qr = 2 * N_pca * k * k  # QR of N×q

# Per power iteration (2 iters): A.T@Q + QR + A@Z + QR
per_iter = (N_pca * D_pca * k +    # A.T @ Q: D×N @ N×q
            2 * D_pca * k * k +     # QR of D×q
            N_pca * D_pca * k +     # A @ Z: N×D @ D×q
            2 * N_pca * k * k)      # QR of N×q
total_iters = niter * per_iter

# Post-iteration
post_b = N_pca * D_pca * k    # B = Q.T @ A: q×N @ N×D... wait, it's q×N @ N×D
post_svd_b = 6 * k * D_pca * D_pca  # SVD of q×D (tiny)
post_u = N_pca * k * k        # U = Q @ Ub: N×q @ q×q

# Centering (done twice: once external, once inside pca_lowrank)
center_ext = 2 * N_pca * D_pca  # mean + subtract (external)
center_int = 2 * N_pca * D_pca  # mean + subtract (internal pca_lowrank center=True)

corrected = init_proj + init_qr + total_iters + post_b + post_svd_b + post_u + center_ext + center_int

print(f"  Matrix shape: {N_pca} x {D_pca}, q={k}, niter={niter}")
print(f"  NCU kernels: {ncu['n_kernels']}")
print(f"  NCU FP32 FMA:                 {ncu['fp32_fma']:>14,.0f}")
print(f"  NCU FP32 total:               {ncu['fp32_total']:>14,.0f}")
print(f"")
print(f"  --- Operation count estimates ---")
print(f"  Current TO model:             {old_macs:>14,}")
print(f"  Corrected (matmuls+QR only):  {corrected:>14,}")
print(f"")
print(f"  Breakdown:")
print(f"    Centering (2x):             {center_ext+center_int:>14,}")
print(f"    Initial A@R + QR:           {init_proj+init_qr:>14,}")
print(f"    Power iters ({niter}x):       {total_iters:>14,}")
print(f"    Post (B, SVD(B), U):        {post_b+post_svd_b+post_u:>14,}")
print(f"")
print(f"  Ratios to NCU FP32 total:")
print(f"    Current TO model:           {ncu['fp32_total']/old_macs:>14.2f}x")
print(f"    Corrected estimate:         {ncu['fp32_total']/corrected:>14.2f}x")
print(f"")
print(f"  VERDICT: Remaining {ncu['fp32_total']/corrected:.1f}x gap likely from:")
print(f"    - QR factorization overhead on GPU (cuSOLVER blocked Householder)")
print(f"    - Small SVD of {k}x{D_pca} matrix via cuSOLVER")
print(f"    - {ncu['n_kernels']} kernel launches with per-kernel overhead")


# =====================================================================
header("JPEG — TO model counts optimized DCT-8, impl uses matmul")
# =====================================================================

ncu = ncu_data["jpeg_q50"]
N_jpeg = ncu["N"]  # 4096
side = int(math.sqrt(N_jpeg))  # 64
n_blocks = (side // 8) ** 2    # 64

# Current TO model (uses 29-MAC optimized DCT-8, counts fwd+inv)
old_dct = n_blocks * 4 * 8 * 29  # forward+inverse, 2 passes each
old_quant = N_jpeg  # divisions
old_dequant = N_jpeg  # multiplies
old_total_macs = old_dct + old_quant + old_dequant

# Actual implementation uses matmul for 2D DCT (NOT optimized DCT-8)
# basis @ (blocks @ basis.T): two 8x8 matmuls per block
# No IDCT in the implementation!
matmul_inner = n_blocks * 8 * 8 * 8  # blocks @ basis.T
matmul_outer = n_blocks * 8 * 8 * 8  # basis @ result
quant_div = n_blocks * 64  # division by Q
round_ops = n_blocks * 64  # torch.round
dequant_mul = n_blocks * 64  # multiply by Q
impl_total = matmul_inner + matmul_outer + quant_div + dequant_mul

print(f"  Image: {side}x{side}, {n_blocks} blocks of 8x8")
print(f"  NCU kernels: {ncu['n_kernels']}")
print(f"  NCU FP32 FMA:                 {ncu['fp32_fma']:>14,.0f}")
print(f"  NCU FP32 total:               {ncu['fp32_total']:>14,.0f}")
print(f"")
print(f"  --- Operation count estimates ---")
print(f"  Current TO model (DCT-8 opt): {old_total_macs:>14,}")
print(f"  Impl (matmul, no IDCT):       {impl_total:>14,}")
print(f"")
print(f"  Ratios to NCU FP32 total:")
print(f"    Current TO model:           {ncu['fp32_total']/old_total_macs:>14.2f}x")
print(f"    Implementation estimate:    {ncu['fp32_total']/impl_total:>14.2f}x")
print(f"")
print(f"  NCU FMA breakdown: {ncu['fp32_fma']:,.0f} FMA = {ncu['fp32_fma']/n_blocks:.0f} per block")
print(f"  Expected per block (2 matmuls): {2*8*8*8} FMA")
print(f"  Ratio per block: {ncu['fp32_fma']/(n_blocks * 2 * 512):.1f}x")
print(f"")
print(f"  VERDICT: Two issues:")
print(f"    1. TO model uses 29-MAC DCT-8 but impl uses 512-MAC matmul (8x8 @ 8x8)")
print(f"    2. TO model counts IDCT but implementation does NOT do IDCT")
print(f"    3. Remaining ~{ncu['fp32_total']/impl_total:.0f}x gap needs empirical investigation")


# =====================================================================
header("MDCT — Python loop + many small kernels")
# =====================================================================

ncu = ncu_data["mdct_audio"]
N_mdct = ncu["N"]  # 4096
frame_size = 512  # from COMPRESSION registry

# Count frames: n_frames = (L - 2*N) // N + 1, capped at 50
L = N_mdct
n_frames = max(1, min((L - 2*frame_size) // frame_size + 1, 50))

print(f"  Signal length: {N_mdct}, frame_size: {frame_size}")
print(f"  n_frames: {n_frames} (capped at 50)")
print(f"  NCU kernels: {ncu['n_kernels']}")
print(f"  NCU FP32 FMA:                 {ncu['fp32_fma']:>14,.0f}")
print(f"  NCU FP32 total:               {ncu['fp32_total']:>14,.0f}")
print(f"")

# Per frame operations (from implementation):
# frame = x[:, start:start+2*N] * window → 2*frame_size muls
# X = basis @ frame → frame_size × 2*frame_size matmul (frame_size output)
# power = abs(X)**2 → frame_size muls+adds+sqrt
# mask = log(power) * bark → frame_size log + frame_size muls
# threshold = exp(mask) → frame_size exp
# scale = clamp(threshold) → frame_size comparisons
# quantized = round(X / scale) → frame_size divs + frame_size round

per_frame_matmul = frame_size * 2 * frame_size  # basis @ frame
per_frame_window = 2 * frame_size
per_frame_power = 3 * frame_size  # abs**2
per_frame_log = frame_size  # log ops
per_frame_exp = frame_size  # exp ops
per_frame_mul_bark = frame_size
per_frame_div = frame_size  # X / scale

total_matmul = n_frames * per_frame_matmul
total_other = n_frames * (per_frame_window + per_frame_power +
                          per_frame_log + per_frame_exp +
                          per_frame_mul_bark + per_frame_div)

print(f"  Per-frame matmul (basis@frame): {per_frame_matmul:,} MACs")
print(f"  Total matmul ({n_frames} frames):    {total_matmul:,} MACs")
print(f"  Total other ops:              {total_other:,}")
print(f"  Grand total estimate:         {total_matmul + total_other:,}")
print(f"")

# Current TO model
to_model_fft = 4 * (frame_size) * math.log2(frame_size) - 6 * frame_size + 8
old_total = N_mdct + to_model_fft + frame_size*6 + frame_size + frame_size + to_model_fft + frame_size*6 + N_mdct
print(f"  Current TO model (FFT-based): {old_total:,.0f}")
print(f"  Ratio current TO:             {ncu['fp32_total']/old_total:.2f}x")
print(f"  Ratio corrected:              {ncu['fp32_total']/(total_matmul+total_other):.2f}x")
print(f"")
print(f"  VERDICT: Implementation uses matmul-based MDCT (not FFT) + Python loop")
print(f"  over {n_frames} frames + psychoacoustic masking (log/exp). TO model assumed")
print(f"  FFT-based MDCT codec without psychoacoustic processing.")
print(f"  77 kernels = ~{ncu['n_kernels']//n_frames} kernels per frame iteration.")


# =====================================================================
header("MEDIAN — Zero FP32, comparison-only")
# =====================================================================

ncu = ncu_data["median"]
N_med = ncu["N"]
W = 7  # default window_size

print(f"  N={N_med}, window_size={W}")
print(f"  NCU FP32 total:               {ncu['fp32_total']:>14,.0f}")
print(f"  NCU int_total:                {ncu['int_total']:>14,.0f}")
print(f"")
n_windows = N_med - W + 1
sort_cmp = n_windows * W * math.log2(W)
print(f"  Predicted comparisons (sort): {sort_cmp:>14,.0f}")
print(f"  NCU integer ops:              {ncu['int_total']:>14,.0f}")
print(f"  Ratio (int ops / pred cmp):   {ncu['int_total']/sort_cmp:>14.1f}x")
print(f"")
print(f"  VERDICT: torch.median uses integer comparison sorting, zero FP32.")
print(f"  TO model should use TO['cmp'] exclusively, no MACs.")
print(f"  Current model already uses TO['cmp'] for sorting but also")
print(f"  adds unfold copy MACs which don't exist in hardware.")

print(f"\n{'='*70}")
print(f"  SUMMARY OF CORRECTIONS NEEDED")
print(f"{'='*70}")
print(f"""
  1. CNN Denoiser: Fix Conv1d params → channels=32, kernel=7 for all layers
     Expected ratio after fix: ~1.01 (near-perfect)

  2. SVD: Change 6ND² to literature-derived estimate. NCU shows ~{ncu_data['svd']['fp32_total']/(1024*64*64):.0f}ND².
     137 kernels = iterative QR + D&C SVD overhead.

  3. PCA: Rederive for pca_lowrank(q=8, niter=2): initial proj + 2 power
     iterations + final SVD. Current model misses power iterations entirely.

  4. JPEG: Fix two bugs:
     a) Use 8x8 matmul cost (512 MACs/block), not optimized DCT-8 (29 MACs)
     b) Remove IDCT — implementation doesn't do inverse transform
     Remaining gap needs empirical investigation.

  5. MDCT: Rederive for matmul-based MDCT + psychoacoustic masking (log/exp).
     Current model assumes FFT-based codec without masking.

  6. Median: Remove unfold MACs. Use TO['cmp'] only.
""")
