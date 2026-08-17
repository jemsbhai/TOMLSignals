# TOMLSignals — Research Findings Log

**Project**: TOMLSignals (TOML Paper 4, IEEE MLSP 2026)  
**Authors**: Muntaser Syed, Marius Silaghi, Sheikh Abujar, Sharun Akter Khushbu  
**Policy**: This document is append-only. Entries are never deleted or rewritten. Corrections are recorded as separate errata entries referencing the original finding number.

---

## Finding F-001: cuFFT Implements Split-Radix, Not Cooley-Tukey

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `shared/to_model.py` (fft_tos), `data/ncu_profiles/ncu_summary.json`, `verify_fft_correction.py`

### Observation

The textbook Cooley-Tukey FFT operation count (N/2 · log₂(N) · 10 real FP ops) overpredicts cuFFT's actual FP32 instruction count by ~30% on both RTX 4090 and A100 GPUs.

### Evidence

Nsight Compute hardware profiling at N=4096, B=1:

| Metric | Cooley-Tukey prediction | Split-radix prediction | NCU measured |
|--------|------------------------|----------------------|--------------|
| FP32 ops | 245,760 | 172,040 | 173,055 |
| Ratio to NCU | 0.704 | **1.006** | — |

The Duhamel & Vetterli (1990) split-radix formula `4N·log₂(N) − 6N + 8` matches hardware to **99.4% accuracy**.

### Cascade effect

Nine algorithms use FFT as a subroutine. After correction:

| Algorithm | Old ratio (Cooley-Tukey) | New ratio (split-radix) |
|-----------|------------------------|----------------------|
| fft | 0.704 | **1.006** |
| dct | 0.705 | **0.964** |
| stft | 0.660 | **1.002** |
| hilbert | 0.787 | 1.120 |
| fir_fft | 0.648 | 0.900 |
| matched_filter | 0.656 | 0.910 |
| wiener | 0.851 | 1.191 |
| periodogram | 0.837 | 1.201 |
| welch | 0.978 | 1.432 |

Pure-FFT algorithms (fft, stft, dct) converge to within 4% of hardware. Compound algorithms show 10–20% residual from non-FFT terms (torch.abs overhead, division implementation), which is implementation overhead absorbed by the α_c calibration coefficient.

### Significance

- First empirical validation (to our knowledge) of split-radix operation counts against GPU hardware instruction profiling.
- The 30% Cooley-Tukey overprediction propagates through every FFT-based energy model in the literature.
- Directly applicable to energy-aware algorithm selection in signal processing and ML.

### Reference

P. Duhamel and M. Vetterli, "Fast Fourier Transforms: A Tutorial Review and a State of the Art," *Signal Processing*, vol. 19, no. 4, pp. 259–299, 1990.

---

## Finding F-002: Complex MAC = 4 FMA on FMA Hardware

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `shared/to_model.py` (to_direct_dft), `data/ncu_profiles/ncu_summary.json`

### Observation

The standard complex multiply-accumulate count of 8 real ops (4 muls + 4 adds) overpredicts by 2× on FMA hardware. On GPUs with fused multiply-add units, a complex MAC requires exactly 4 FMA instructions:

```
(a+bi)(c+di) accumulated into (e+fi):
  e += a·c   (1 FMA)
  e -= b·d   (1 FMA)
  f += a·d   (1 FMA)
  f += b·c   (1 FMA)
```

### Evidence

Nsight Compute at N=1024, B=1 (direct DFT = N² complex MACs):

| Metric | 8-op prediction | 4-FMA prediction | NCU measured |
|--------|----------------|-----------------|--------------|
| FP32 ops | 8,388,608 | 4,194,304 | 4,229,120 |
| Ratio to NCU | 0.504 | **1.008** | — |

The 4-FMA formula matches hardware to **99.2% accuracy**. The 34,816 residual ops (0.8%) are reduction/accumulation operations across thread blocks.

### Significance

- Any energy model using the textbook "8 ops per complex MAC" will overpredict by 2× on all modern GPUs with FMA units (i.e., all GPUs since ~2010).
- Affects: DFT, complex convolution, beamforming, MIMO processing, and any algorithm with complex-valued matrix operations.

---

## Finding F-003: Non-Power-of-2 FFT Sizes Incur Catastrophic Energy Penalties

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `diagnose_welch_dst.py`, `shared/to_model.py` (fft_tos, _is_pow2)

### Observation

FFT of size 8194 (= 2 × 17 × 241) is **4.6× more expensive** than FFT of size 16384 (= 2¹⁴) on an RTX 4090, despite being half the size. The large prime factor 241 forces cuFFT into an expensive mixed-radix or Bluestein code path.

### Evidence

Wall-clock timing on RTX 4090 Laptop GPU:

| FFT size | Factorization | Time (µs) | Relative to FFT(4096) |
|----------|--------------|-----------|----------------------|
| 4096 | 2¹² | 24.8 | 1.00× |
| 8192 | 2¹³ | 17.2 | 0.69× |
| 8194 | 2 × 17 × 241 | 85.8 | **3.45×** |
| 16384 | 2¹⁴ | 18.6 | 0.75× |

Key ratios:
- FFT(8194) / FFT(8192) = **4.99×** (two extra elements → 5× slower)
- FFT(8194) / FFT(16384) = **4.62×** (half-size input → 4.6× slower)

### Impact on TO model

The split-radix formula `4N·log₂(N) − 6N + 8` is only valid for N = 2^m. Applying it to N=8194 produces a mathematically invalid result. Our guard pads to the next power of 2, which serves as a principled lower bound.

DST-II uses antisymmetric extension of length L = 2(N+1). For N=4096, L=8194 — an accidentally terrible FFT size:

| Model | DST ratio (actual/predicted) |
|-------|-----------------------------|
| Unguarded split-radix (invalid) | 3.84 |
| Power-of-2 padded (lower bound) | **1.80** |

### Practical implication

Signal processing pipelines on GPU should always pad to power-of-2 FFT sizes. The memory cost of padding (up to 2× for worst case) is negligible compared to the 5× execution time penalty of a bad factorization.

---

## Finding F-004: torch.abs(X)**2 vs X.real²+X.imag² — Implementation Overhead

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `test_abs_vs_real.py`, `algorithms/spectral.py`

### Observation

The PyTorch expression `torch.abs(X)**2` (where X is complex) computes `sqrt(Re²+Im²)` then squares the result — mathematically equivalent to `Re²+Im²` but computationally wasteful. Nsight Compute shows **13N FP32 ops** for the abs-then-square path versus a mathematical minimum of **3N ops** (2 multiplies + 1 add per element).

### Evidence

NCU periodogram breakdown (N=4096), extra FP32 ops beyond FFT:

| Component | Extra FMA | Extra add | Extra mul | Total extra |
|-----------|----------|----------|----------|-------------|
| Measured | +12,288 (3N) | +8,192 (2N) | +32,766 (~8N) | 53,246 (~13N) |
| Mathematical minimum | — | — | — | 4N = 16,384 |

Despite 3.2× more instructions, wall-clock timing shows **no difference** (65.8 µs vs 66.0 µs). The GPU is not compute-bound at this scale — the extra instructions fit within memory latency hiding.

### Decision

The TO model counts the mathematical minimum (3N for |X|²). The implementation overhead is absorbed by the α_c calibration coefficient. This is consistent with the project methodology: TO counts are architecture-independent mathematical operations, not library-specific instruction counts.

---

## Finding F-005: CNN and LSTM Denoiser TO Models Had Wrong Architecture Parameters

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `shared/to_model.py`, `algorithms/ml_enhanced.py`, `data/ncu_profiles/ncu_summary.json`

### Observation

The TO model for the CNN denoiser used incorrect Conv1d parameters (channels=16, mixed kernel sizes) instead of the actual implementation (channels=32, kernel_size=7 for all layers). The LSTM denoiser TO model used hidden_size=64 with 2 layers instead of hidden_size=128 with 1 layer.

### Evidence

**CNN Denoiser** (N=4096):

| Model | Architecture | Predicted MACs | NCU FP32 | Ratio |
|-------|-------------|---------------|----------|-------|
| Old | Conv1d(1,16,7)+Conv1d(16,32,5)+Conv1d(32,1,3) | 11,337,728 | 31,727,232 | 2.798 |
| Corrected | Conv1d(1,32,7)+Conv1d(32,32,7)+Conv1d(32,1,7) | 31,195,136 | 31,727,232 | **1.017** |

**LSTM Denoiser** (N=1024):

| Model | Architecture | Predicted FMA | NCU FMA | Ratio |
|-------|-------------|--------------|---------|-------|
| Old | LSTM(1,64,layers=2)+Linear(64,1) | ~51M | 68,682,752 | 1.366 |
| Corrected | LSTM(1,128,layers=1)+Linear(128,1) | 67,633,152 | 68,682,752 | **1.016** |

### Root cause

TO model formulas were written from memory or an earlier architecture spec, not verified against the actual `ml_enhanced.py` implementation. A systematic cross-check of all TO model parameters against implementation code would have caught this.

### Lesson

Always derive TO formulas by reading the implementation source, not from architecture descriptions.

---

## Finding F-006: MDCT Implementation Uses Matmul, Not FFT

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `shared/to_model.py`, `algorithms/compression.py`, `data/ncu_profiles/ncu_summary.json`

### Observation

The TO model assumed the MDCT codec used an FFT-based transform. The actual implementation uses explicit matrix multiplication (basis @ frame) with a Python loop over frames, plus psychoacoustic masking (log, exp operations) not counted in the original model.

### Evidence

N=4096, frame_size=512, n_frames=7:

| Model | Approach | Predicted ops | NCU FP32 | Ratio |
|-------|---------|--------------|----------|-------|
| Old | FFT-based, no masking | 25,104 | 3,895,808 | 155.2 |
| Corrected | Matmul + psychoacoustic | 3,695,104 | 3,895,808 | **1.054** |

The 77 NCU kernels (~11 per frame) confirm the Python-loop structure.

### Significance

This is the largest single correction in the TO model (155× → 1.05×). It underscores that TO formulas must be derived from the actual implementation, not from the textbook algorithm description.

---

## Finding F-007: Median Filter Uses Zero Floating-Point Operations

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `shared/to_model.py`, `data/ncu_profiles/ncu_summary.json`

### Observation

`torch.median` performs sorting entirely via integer comparison operations. Nsight Compute confirms exactly zero FP32 instructions. The previous TO model incorrectly included unfold-copy MACs (143M TOs) alongside comparison TOs.

### Evidence

- NCU fp32_total: **0** (zero)
- NCU int_total: 35,324,920
- Old TO_compute: 4,018,729 (cmp) + 143,150,000 (spurious MACs) = 147,168,729 TOs
- New TO_compute: 4,018,729 TOs (comparison-only)

### Significance

Median filter is a pure-comparison algorithm on GPU. Energy models must distinguish comparison-only algorithms from arithmetic algorithms — they have fundamentally different hardware utilization patterns (integer ALU vs FP32 FMA units).

---

## Finding F-008: JPEG cuBLAS Overhead Amortizes with Block Count

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `profile_scaling.py`, `data/ncu_profiles/scaling/jpeg_*.csv`

### Observation

NCU profiling of the JPEG pipeline at four image sizes shows that cuBLAS FMA-per-block is constant at 66,368 for 4–64 blocks, then drops to 17,216 at 256 blocks. The mathematical minimum is 1,024 FMA/block (two 8×8 matmuls). The overhead ratio ranges from 64.8× (small images) to 16.8× (larger images).

### Evidence

| Image | Blocks | FMA/block | Overhead vs theoretical |
|-------|--------|----------|------------------------|
| 16×16 | 4 | 66,368 | 64.8× |
| 32×32 | 16 | 66,368 | 64.8× |
| 64×64 | 64 | 66,368 | 64.8× |
| 128×128 | 256 | 17,216 | 16.8× |

### Significance

The overhead is from cuBLAS thread dispatch on 8×8 matrices, not from the algorithm. It amortizes at scale. The TO model should use the mathematical operation count (2 × 8³ = 1,024 FMA/block), and the implementation overhead is absorbed by α_c.

---

## Finding F-009: SVD Operation Count Is Implementation-Dependent and Not Capturable by a Universal Polynomial

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `profile_scaling.py`, `data/ncu_profiles/scaling/svd_*.csv`, `parse_scaling_ncu.py`

### Observation

NCU profiling of `torch.linalg.svd` at 9 (N, D) combinations reveals that the FP32-ops-per-ND² ratio varies from 10 to 69, depending on matrix shape. A simple `a·ND² + b·D³` formula fitted on D-varying data fails to predict N-varying data (errors up to 108%). cuSOLVER uses different internal algorithms for different matrix aspect ratios.

### Evidence

**Vary N (D=64 fixed):**

| N | FP32 total | FP32/ND² |
|------|-----------|----------|
| 256 | 72,714,565 | 69.35 |
| 512 | 67,928,965 | 32.39 |
| 1024 | 86,405,413 | 20.60 |
| 2048 | 108,684,522 | 12.96 |
| 4096 | 170,023,400 | 10.13 |

**Vary D (N=1024 fixed):**

| D | FP32 total | FP32/ND² |
|-----|-----------|----------|
| 16 | 11,085,796 | 42.29 |
| 32 | 44,033,629 | 41.99 |
| 64 | 86,405,413 | 20.60 |
| 128 | 416,871,524 | 24.85 |

Key observation: N=256 has MORE FP32 ops than N=512 (72.7M vs 67.9M), confirming a large N-independent overhead from QR iteration and D&C decomposition.

Least-squares fit `a·ND² + b·D³` gives a=20.5, b=34.3, but cross-validation errors reach 108%.

### Decision

Retain `6·ND²` as the theoretical lower bound (Householder bidiagonalization cost from Golub & Van Loan). Report the NCU-measured ratio (3.41× at N=1024, D=64) as “implementation overhead” in the paper. This is an honest representation: the iterative QR/D&C phase of SVD has convergence-dependent cost that no fixed formula can capture.

### Significance

This finding extends to PCA (which internally calls SVD on a small matrix). The 10.6× PCA gap is 74% cuSOLVER QR overhead + 51% small-matrix SVD overhead (from timing decomposition). Operation counts for factorization-heavy algorithms are inherently implementation-dependent on GPU, unlike FFT and matmul which have predictable costs.

---

## Finding F-010: Sequential Algorithm Energy Is Dominated by Python Dispatch, Not Computation

**Date**: 2026-05-17  
**Status**: Validated  
**Relevant files**: `diagnose_iir_fused.py`, `algorithms/filters.py`, `shared/to_model.py`

### Observation

Per-step energy for sequential GPU algorithms falls into two categories separated by **three orders of magnitude**:

- **Python-loop sequential** (LMS, NLMS, RLS, Kalman, etc.): median 5,902 µJ/step on RTX 4090
- **Fused-sequential** (IIR via torchaudio.lfilter): median 5.0 µJ/step on RTX 4090
- **Ratio: 1,190×**

Both execute the same class of inherently sequential algorithm (each output depends on previous outputs). The only difference is the implementation: Python for-loop with per-step kernel launches vs. a single fused C++ kernel.

### Evidence

**RTX 4090 Laptop:**

| Category | Median µJ/step | Mean µJ/step | Range | n |
|----------|---------------|-------------|-------|---|
| Fused-sequential | 5.0 | 4.6 | [3.1, 5.8] | 3 |
| Python-loop | 5,902 | 9,144 | [1,089, 65,966] | 44 |

**Cross-GPU (Python-loop only):**

| GPU | Median µJ/step |
|-----|---------------|
| RTX 4090 Laptop | 5,902 |
| A100 SXM4 | 840 |
| Ratio | 7.0× |

The 7× cross-GPU ratio for Python-loop overhead reflects the 4090 laptop’s slower CPU and higher Python interpreter latency vs. the A100 server.

### Significance

1. **The α_o coefficient in the 3-parameter model captures Python dispatch overhead, not sequential computation cost.** The 1,190× gap proves that the actual FP computation in sequential algorithms is energetically negligible compared to the per-step interpreter + kernel launch overhead.

2. **Fused-sequential algorithms are a third energy regime** that neither the parallel TO model nor the sequential α_o term can capture. IIR with torchaudio is excluded from sequential step counting on the 4090 (3 data points treated as outliers).

3. **Practical implication for algorithm designers:** Converting a Python-loop sequential algorithm to a fused C++/CUDA kernel reduces per-step energy by ~1,000×. This is a far larger gain than any algorithmic optimization within the loop body.

### Model treatment

- **4090 (torchaudio available):** IIR seq_steps = 0, treated as outlier (3 points out of 138)
- **A100 (torchaudio unavailable):** IIR seq_steps = N, correctly classified as Python-loop sequential
- Runtime detection via `_LOCAL_HAS_TORCHAUDIO` flag in `to_model.py`

---

## Finding F-011: Per-Category OLS Fit Was Overwriting Global Predictions (Bug Fix)

**Date**: 2026-05-17  
**Status**: Fixed  
**Relevant files**: `analyze_results.py`

### Bug

`fit_two_parameter()` stored predictions back into `DataPoint.e_predicted` as a side effect. The per-category diagnostic fits (run after the main model fits) overwrote the global predictions with values from tiny, ill-conditioned per-category fits. The compression category (5 points, r² = -16 trillion) produced predictions of 9,959 J for JPEG — physically impossible.

### Impact

- JPEG per-algorithm error: 6,816,514% → **84%** after fix
- MDCT per-algorithm error: 2,041,964% → **42%** after fix
- Head-to-head ranking accuracy: 57% → **77%** after fix
- Model r² and α coefficients were never affected (computed correctly, just not stored correctly)

### Fix

Added `store_predictions=False` parameter to `fit_two_parameter()`. Per-category diagnostic fits now pass this flag to avoid overwriting global predictions.

---

## Finding F-012: Head-to-Head Ranking Failures Reveal Three Model Limitations

**Date**: 2026-05-17  
**Status**: Documented  
**Relevant files**: `analyze_results.py` (head-to-head section)

### Observation

After the bug fix, head-to-head ranking accuracy is 77% (23/30) on both GPUs. The 7 failures on the 4090 decompose into three root causes:

### 1. Single α_o cannot distinguish dense vs sparse sequential steps (4 failures)

**Kalman vs UKF (N=256,1024,4096):** Model predicts UKF 7× more expensive (1400 vs 200 steps). Measured: UKF is actually cheaper because its denser inner loop (matrix operations) amortizes kernel launch overhead better.

- Kalman: 8,413 µJ/step (tiny per-step compute, mostly kernel launch)
- UKF: 1,115 µJ/step (dense matrix ops per step)

**LMS vs RLS (N=16384):** Model predicts RLS cheaper (100 vs 200 steps). Measured: RLS is more expensive because per-step work is O(M²) vs O(M).

Fix would require per-algorithm α_o or a per-step compute density term — adds model complexity without clear generalization benefit.

### 2. LSTM fused-sequential, same as IIR (2 failures)

**CNN vs LSTM (N=4096,16384 at B=1):** cuDNN LSTM at B=1 processes one timestep at a time (inherently sequential) but is classified as parallel (seq_steps=0). Predicted 0.019J, measured 0.257J. Same regime as IIR with torchaudio (Finding F-010).

### 3. Measurement noise at small N (1 failure)

**Periodogram vs Welch (N=256):** Both predictions are ~0.0019J. The difference is within measurement noise at this scale.

### Significance

The 23% failure rate is dominated by the single-α_o limitation (4/7 failures). This is a known trade-off: a single overhead coefficient provides a simple, interpretable model but cannot capture the 8× variation in per-step overhead across algorithms.

---

## Finding F-013: Sequential Overhead Scales with Kernel Launches, Not Loop Iterations

**Date**: 2026-05-18  
**Status**: Validated  
**Relevant files**: `shared/to_model.py` (KERNELS_PER_ITER, get_seq_steps), `diagnose_kernel_launches.py`, `diagnose_alpha_o.py`, `algorithms/estimation.py`

### Observation

The three-parameter model's sequential overhead term α_o should scale with CUDA kernel launches, not Python loop iterations. Each `torch.*` operation on a GPU tensor that produces a new tensor (not a view) triggers a kernel launch, incurring Python-to-CUDA dispatch overhead (~5-20 µs) plus GPU idle power during the dispatch gap.

Different algorithms have different numbers of kernel launches per outer loop iteration (7 for LMS, 43 for UKF), making loop iterations a poor proxy.

### Evidence

**UKF step counting inconsistency (root cause of 3 Kalman-vs-UKF ranking failures):**

The original `_seq_ukf` uniquely counted inner-loop iterations as separate steps (1,400 = 100 outer × 14 inner), while all other 11 `_seq_*` functions counted only outer iterations. This was a coding inconsistency, not a physics insight.

Source code analysis of kernel launches per outer iteration:

| Algorithm | Outer iters | Kernels/iter | Total launches |
|-----------|------------|-------------|----------------|
| LMS | 200 | 7 | 1,400 |
| NLMS | 200 | 11 | 2,200 |
| RLS | 100 | 12 | 1,200 |
| APA | 100 | 15 | 1,500 |
| Kalman | 200 | 15 | 3,000 |
| EKF | 200 | 22 | 4,400 |
| UKF | 100 | 43 | 4,300 |
| Particle | 200 | 20 | 4,000 |
| FastICA | 50 | 13 | 650 |
| NMF | 50 | 14 | 700 |
| MDCT | varies | 11 | varies |

**Per-launch overhead consistency (RTX 4090):**

| Approach | Per-alg overhead range | CV | 3-param r² |
|----------|----------------------|-----|------------|
| Loop iterations | 1,122–8,961 µJ (8.0×) | ~130% | 0.569 |
| Kernel launches | 234–594 µJ (2.5×) | 34% | **0.932** |

**Cross-GPU ranking accuracy:**

| Approach | RTX 4090 | A100 |
|----------|----------|------|
| Loop iterations (old) | 23/30 = 77% | 23/30 = 77% |
| Kernel launches (new) | 23/30 = 77% | 23/30 = 77% |

Ranking accuracy is preserved while r² improves dramatically.

### PCA reclassification

`torch.pca_lowrank` is a C++ function (like `torch.linalg.svd`). Its internal kernel launches occur within PyTorch C++ code without Python dispatch overhead. PCA's overhead is cuSOLVER overhead, absorbed by α_c — the same regime as SVD (which always had seq_steps=0). PCA seq_steps changed from B to 0.

### Irreducible Kalman-vs-UKF ranking flip

The Kalman vs UKF ranking is GPU-dependent: on 4090 (laptop CPU, high per-iteration dispatch), fewer outer iterations wins → UKF cheaper. On A100 (server CPU, low per-iteration dispatch), more inner-loop launches wins → Kalman cheaper. No single α_o resolves this. With kernel launches, UKF (4,300) > Kalman (3,000), correctly predicting the A100 ranking but not the 4090 ranking (and vice versa for iteration counting).

### Significance

1. Kernel launches are the physically correct unit for sequential dispatch overhead — each launch is the event that incurs the overhead cost.
2. The 3-parameter model r² on the RTX 4090 improves from 0.569 to 0.932 without adding parameters.
3. Per-algorithm overhead consistency improves from 8× to 2.5× range.
4. PCA correctly reclassified as parallel (C++ function, not Python dispatch).

---

## Finding F-014: Fused-Sequential Algorithms Require a Separate Overhead Coefficient

**Date**: 2026-05-18  
**Status**: Validated  
**Relevant files**: `shared/to_model.py` (get_fused_steps), `analyze_results.py` (fit_four_parameter), `diagnose_fused_seq.py`

### Observation

cuDNN LSTM at B=1 processes N timesteps sequentially within a single fused C++/CUDA kernel. It has zero Python dispatch overhead but incurs per-timestep serial execution cost at low GPU SM utilization. This is a physically distinct energy regime from both parallel algorithms and Python-loop sequential algorithms.

### Evidence

**LSTM at B=1 prediction errors (3-param vs 4-param):**

| GPU | N | 3-param error | 4-param error |
|-----|------|---------------|---------------|
| 4090 | 1024 | 93.2% | **13.3%** |
| 4090 | 4096 | 92.5% | **3.9%** |
| 4090 | 16384 | 92.2% | **0.3%** |
| A100 | 1024 | 90.2% | **7.1%** |
| A100 | 4096 | 89.8% | **1.4%** |
| A100 | 16384 | 89.7% | **0.1%** |

**Fitted coefficients:**

| Coefficient | RTX 4090 | A100 | Physical meaning |
|-------------|----------|------|------------------|
| alpha_o | 385.4 uJ/launch | 125.2 uJ/launch | Python dispatch overhead |
| alpha_f | 55.7 uJ/step | 45.3 uJ/step | Fused-sequential per-timestep cost |
| alpha_o / alpha_f | 6.9x | 2.8x | Python dispatch is 3-7x more expensive |

**Ranking improvement:**

| GPU | 3-param | 4-param |
|-----|---------|----------|
| 4090 | 23/30 = 77% | **24/30 = 80%** |
| A100 | 23/30 = 77% | **24/30 = 80%** |

CNN vs LSTM denoiser at N=16384 fixed on BOTH GPUs.

### Physical justification

The two overhead terms capture physically distinct mechanisms:

1. **alpha_o (Python-loop dispatch)**: CPU-side cost. Python interpreter executes each loop iteration, dispatching CUDA kernels through the driver. Dominated by CPU speed: 385 uJ/launch on 4090 (laptop CPU) vs 125 uJ/launch on A100 (server CPU).

2. **alpha_f (fused-sequential)**: GPU-side cost. A single C++ kernel (cuDNN LSTM) processes timesteps serially. No Python interpreter or CUDA driver dispatch. The cost is serial execution time x GPU power at low SM utilization. More consistent across GPUs: 56 vs 45 uJ/step (1.2x ratio vs 3.1x for alpha_o).

The alpha_f cross-GPU consistency (1.2x) vs alpha_o inconsistency (3.1x) further confirms these are different mechanisms: alpha_o scales with CPU speed, alpha_f scales with GPU characteristics.

### Scope

Currently only LSTM at B<=1 is classified as fused-sequential. IIR with torchaudio is also fused-sequential but has ~5000x less compute per step, so a single alpha_f cannot cover both. IIR remains as seq_steps=0 on 4090 (3 outlier data points).

---

## Finding F-015: CPU vs GPU Energy Comparison

**Date**: 2026-05-20  
**Status**: Validated  
**Relevant files**: `shared/cpu_harness.py`, `cpu_algorithms.py`, `run_cpu_suite.py`, `analyze_cpu_vs_gpu.py`, `data/cpu_results/all_cpu_results.csv`, `data/cpu_vs_gpu_comparison.csv`

### Summary

Compared energy-per-signal between CPU (Intel i9-14900HX, RAPL via LibreHardwareMonitor) and GPU (RTX 4090 Laptop, NVML) across 129 common (algorithm, signal_length) pairs. GPU energy normalized by batch size (B=2048 for most algorithms) for per-signal comparison.

### Key results

| Metric | Value |
|--------|-------|
| Total comparisons | 129 |
| GPU more efficient | 86 (67%) |
| CPU more efficient | 43 (33%) |
| Median CPU/GPU ratio | 8.8x |
| Mean CPU/GPU ratio | 90.7x |
| Min ratio (CPU wins most) | 0.01x (NLMS) |
| Max ratio (GPU wins most) | 3556.6x (SavGol N=256) |

### Per-category results

| Category | GPU wins | CPU wins | Median ratio |
|----------|----------|----------|--------------|
| transform | 29 | 1 | 24.1x |
| filter | 28 | 3 | 14.2x |
| spectral | 11 | 1 | 74.8x |
| ml_enhanced | 12 | 0 | 24.2x |
| compression | 4 | 0 | 22.9x |
| decomposition | 2 | 6 | 0.4x |
| estimation | 0 | 16 | 0.1x |
| adaptive | 0 | 16 | 0.0x |

### Three energy regimes

1. **Parallel-batched algorithms** (transforms, filters, spectral, ML, compression): GPU wins decisively. The GPU processes B=2048 signals in a single kernel launch; per-signal cost is amortized. Median 14-75x more GPU-efficient.

2. **Sequential algorithms** (adaptive filters, state estimation): CPU wins 100% (32/32). The GPU's Python-loop dispatch overhead (alpha_o = 385 uJ/launch) makes sequential algorithms catastrophically expensive. NLMS is 100x more efficient on CPU.

3. **Decomposition** (SVD, PCA, FastICA, NMF): Mixed. CPU wins 6/8. These algorithms involve eigendecomposition/factorization that doesn't parallelize across batch dimension on GPU. Exceptions: SVD at large N where cuSOLVER amortizes.

### Crossover points

- **SVD**: CPU wins at N=256 (0.59x), GPU wins at N>=1024 (2.45x). cuSOLVER kernel launch overhead amortizes at larger matrix sizes.
- **PCA**: GPU wins at N=256 (1.78x), CPU wins at N>=1024 (0.30x). sklearn PCA is efficient for tall matrices.
- **ESPRIT**: GPU batch size varies (B=512 at N=256, B=2 at N=1024). At low batch sizes, GPU loses its amortization advantage.

### Signal length effect

| N | GPU wins | CPU wins | Median ratio |
|---|----------|----------|--------------|
| 256 | 23 | 11 | 23.1x |
| 1024 | 22 | 13 | 10.1x |
| 4096 | 20 | 10 | 8.6x |
| 16384 | 21 | 9 | 4.3x |

GPU advantage decreases with N. At small N, the per-signal compute is tiny and GPU's parallel kernel launch amortizes well. At large N, CPU's per-signal compute grows but remains single-threaded, while GPU's batch-amortized advantage is diluted.

### Methodology

- CPU power: RAPL "CPU Package" via LibreHardwareMonitor HTTP API, ~217 Hz polling from Python
- Protocol: thermal settle (±1°C over 5s), load settle (<10% over 3s), 10s idle baseline, 5s active measurement
- GPU idle verified via LHM GPU Core Load = 0% during all CPU benchmarks
- CPU batch size B=1 (natural CPU usage). GPU batch size B=2048 (or smaller for algorithms that can't batch).
- Energy per signal: CPU = delta_P × time / iterations. GPU = energy_per_call / batch_size.

### Significance

This is the first systematic energy comparison across 37 signal processing algorithms on matched hardware (same laptop). The three-regime result (parallel → GPU, sequential → CPU, decomposition → mixed) provides actionable guidance for practitioners. The finding that sequential algorithms are 10-100x more CPU-efficient challenges the assumption that "GPU is always better for compute."

---

## Finding F-016: CPU Benchmarking Methodology Quirks and Notes

**Date**: 2026-05-20  
**Status**: Documented  
**Relevant files**: `shared/cpu_harness.py`, `cpu_algorithms.py`, `check_gpu_sensors2.py`

### 1. LHM GPU Power Sensor Returns Stale Max Value

LibreHardwareMonitor's "GPU Package" power reading for the RTX 4090 Laptop GPU consistently reports 593.5 W regardless of actual GPU state. This value matches the historical max (likely from a prior CUDA workload) and never updates to reflect current draw. At 0% GPU Core Load and 210 MHz clock, actual idle power is ~1-5 W (confirmed by NVML in the GPU harness).

**Workaround**: GPU idle during CPU benchmarks is verified via GPU Core Load (0.0% throughout all 147 benchmarks), not GPU power. The stuck power reading is logged but not used for any calculation.

**Implication for reproducibility**: Researchers using LHM on NVIDIA laptop GPUs should validate GPU power readings against NVML or nvidia-smi before trusting them. GPU Core Load is a reliable alternative for idle verification.

### 2. CPU Idle Power Variability and Noise Floor

CPU Package idle power (RAPL via LHM) ranged from 11-20 W across the 147 benchmarks, with per-measurement standard deviation of 0.5-5 W. This is significantly noisier than GPU NVML measurements (sub-watt precision at idle).

Sources of variability:
- Background OS processes (Windows services, indexing, telemetry)
- P-core vs E-core scheduling shifts at idle
- Thermal-dependent leakage current (idle power correlates with die temperature)

Active power (delta above idle) ranged from 10-59 W depending on algorithm. For lightweight algorithms at small N (e.g., FFT N=256 at delta_P=13.5 W), the idle variability (up to 5 W) represents ~37% relative noise. For compute-heavy algorithms (e.g., Transformer N=4096 at delta_P=72 W), noise is <7%.

**Mitigation**: Thermal settling (±1°C/5s) and load settling (<10%/3s) before each measurement reduces but does not eliminate this variability. The 10-second idle baseline provides a per-measurement reference that captures the current background state.

**Implication**: CPU energy-per-call values for lightweight algorithms at small N should be interpreted as order-of-magnitude estimates, not precise measurements. The GPU-vs-CPU comparison conclusions (F-015) are robust because the category-level patterns (GPU wins parallel by 10-100x, CPU wins sequential by 10-100x) far exceed the measurement noise.

### 3. MDCT at N=256 Physically Impossible

MDCT audio codec uses frame_size=512, requiring a minimum signal length of 2×512=1024 samples. At N=256, the frame slice exceeds the signal length. Both CPU and GPU benchmarks skip this configuration (GPU has MDCT only at N>=4096, CPU run produced 147/148 results with MDCT N=256 as the sole failure). This is expected and consistent across platforms.

### 4. CPU UKF Required Parameter Matching for Apples-to-Apples Comparison

The textbook UKF uses alpha=1e-3, which produces w_c0 ≈ -1e6 in float32 — a large negative weight that destroys covariance matrix positive-definiteness. The GPU implementation (estimation.py) uses alpha=1.0 (equivalent to the cubature Kalman filter, Arasaratnam & Haykin 2009) and stabilizes the state transition matrix F by scaling its spectral radius to 0.95.

The CPU implementation was matched to these parameters (alpha=1.0, spectral radius scaling) for fair comparison. The initial CPU implementation with alpha=0.5 failed Cholesky decomposition; matching the GPU's alpha=1.0 resolved this.

**Implication for the paper**: The UKF implementation is technically a cubature Kalman filter (CKF). This should be noted when describing the algorithm. The energy comparison remains valid because both CPU and GPU execute identical mathematical operations.

---

## F-017: Paper Submitted to IEEE MLSP 2026
**Date**: 2026-05-22
**Evidence**: Submitted via CMT. 6-page paper, double-blind, 8 figures, 3 tables, 11 references. Anonymous code at https://anonymous.4open.science/r/TOMLSignals-8810/.
**Key results reported**: r²=0.947 (RTX 4090), r²=0.982 (A100 SXM4), 80% head-to-head ranking (24/30) on both GPUs, NCU validation (99.4% split-radix FFT, 99.2% complex MAC), three sequential energy regimes (α_o/α_f = 2.8-6.9×), CPU vs GPU comparison (CPU wins 100% sequential, GPU wins 84/85 parallel).
**Corrections applied during writing**: Garcia2017 reference was hallucinated and replaced with Bridges2016. Desoli2017 title corrected. 1,190× gap claim replaced with conservative α ratio (3-7×). Head-to-head failure analysis corrected from "5/6 Kalman-UKF" to "4/6 sequential pairs per GPU".

---

## F-018: MLSP 2026 Decision and Reviewer Findings (Camera-Ready Scope)
**Date**: 2026-08-17
**Status**: Documented
**Evidence**: OpenReview decision (accept as poster, published in proceedings); four official reviews (ratings 6, 5, 4, 3; confidences 5, 2, 3, 2). No author rebuttal was submitted.

### Reviewer points and their disposition
1. DHCe (3, weak reject): r2 across five decades overstates accuracy; 4 parameters on ~130 points with no held-out set; "hardware validated" oversells because NCU compares instruction counts, not energy.
   - The submitted r2 (0.947 / 0.982) is computed in LINEAR space by v0 fit_four_parameter (sum of squared residuals on raw joules), not on the log-log axes of Fig. 1. The paper never stated the definition. Linear r2 is dominated by the joule-scale points; the reviewer's concern stands in that form.
   - Disposition: EXP-CR-001 (analyze_cv.py) reports MdAPE, within-1.5x/2x/3x, log-space r2, LOAO / LOCO / leave-one-category-out cross-validation, head-to-head under LOAO, and cross-GPU transfer. Wording of "hardware validation" to be split into two tiers: NCU validates instruction counts (inputs to TO_c); NVML power validates energy predictions.
2. AQaA (5): why 23 of 37 profiled and which; head-to-head section unclear; Fig. 4 colors; split-radix only at N = 4096?
   - Profiled (v0 run_ncu_profile.py): fft, direct_dft, dct, dst, dwt_haar, stft, hilbert, fir_direct, fir_fft, wiener, matched_filter, savgol, median, filterbank_32ch, periodogram, welch, svd, pca, cnn_denoiser, lstm_denoiser, transformer_denoiser, jpeg_q50, mdct_audio. Not profiled: lms, nlms, rls, apa_p4, kalman, ekf, ukf, particle_1k, fastica, nmf, music, esprit, dwt_db4, iir_butter4. No technical obstacle recorded.
   - Disposition: EXP-CR-003 profiles the remaining 14; EXP-CR-002 profiles FFT at N = 256 to 65536; head-to-head subsection rewritten; Fig. 4 solid/dashed pairs recolored.
3. qq3z (4): novelty over TOML core; missing related work (Latif et al. 2026 TCC; Latif et al. 2025 IEEE Access; Fischer 2025 arXiv:2509.22092); generalization to unseen architectures.
   - All three references verified live on 2026-08-17 (DOIs 10.1109/TCC.2026.3700971, 10.1109/ACCESS.2025.3554728, 10.48550/arXiv.2509.22092).
   - Disposition: explicit contributions-beyond-TOML paragraph; cross-GPU transfer analysis in EXP-CR-001; references added.
4. A6Ww (6): narrow algorithm/hardware set; no action beyond the above.

### Model facts established during triage (for the camera-ready text)
- alpha_f is identified only by cuDNN LSTM at B = 1 (get_fused_steps); LOAO with LSTM held out cannot estimate alpha_f. Report as a one-exemplar regime.
- FLAIRS-39 TOML paper DOI: 10.32473/flairs.39.1.141781 (published title "TOML Transistor Operations for Machine Learning: A Physics-Grounded Energy Efficiency Framework"; authors Syed, Silaghi, Abujar, Akter Khushbu).
- TOMLCloud is under review (CloudCom 2026) and can only be cited as submitted for publication.
- MLSP camera-ready: 6 pages including references, no extension; upload via OpenReview "Camera Ready Revision"; de-anonymize paper and code repository.
- Camera-ready experiments are logged in LOGBOOK.md (EXP-CR-001, 002, 003), created 2026-08-17.

---

## F-019: The Reported r2 Reflects the Joule-Scale Points; Least-Squares Coefficients Are Set by One to Three Algorithms
**Date**: 2026-08-17
**Status**: Validated (EXP-CR-001, analyze_cv.py, same data and design matrix as the submission; v0 coefficient check passed)
**Relevant files**: analyze_cv.py, data/camera_ready/exp_cr_001_cv_results.json, exp_cr_001_per_algorithm.csv, exp_cr_001_console.txt, LOGBOOK.md EXP-CR-001

### Observation
The four-parameter model's in-sample r2 of 0.947 (RTX 4090) and 0.982 (A100) is computed on raw joules and is dominated by the largest-energy configurations. Scale-free metrics on the same fit: median absolute percentage error 44.3% (4090) and 68.0% (A100); 69.6% and 54.8% of configurations within 2x of measurement; 80.4% and 74.6% within 3x; log-space r2 of -0.19 and +0.14. Head-to-head ranking is 24/30 on both GPUs, as reported.

### Evidence
| Metric | 4090 in-sample | 4090 LOAO | A100 in-sample | A100 LOAO |
|---|---|---|---|---|
| r2 linear | 0.947 | 0.731 | 0.982 | 0.654 |
| r2 log10 | -0.19 | -0.25 | 0.14 | 0.06 |
| MdAPE | 44.3% | 49.4% | 68.0% | 76.2% |
| within 2x | 69.6% | 60.9% | 54.8% | 47.6% |
| head-to-head | 24/30 | 22/30 | 24/30 | 22/30 |

Leave-one-configuration-out is indistinguishable from in-sample (MdAPE 45.2 / 68.1, head-to-head 24/30 on both), so the gap is not overfitting. Coefficient range across the 37 leave-one-algorithm-out folds: 4090 alpha_c 5.5 to 13.7 fJ/TO (in-sample 13.6; minimum when transformer_denoiser is held out), alpha_m 58 to 232 fJ/TO (in-sample 72; maximum when the transformer is held out); A100 alpha_o 55 to 126 uJ/launch (in-sample 125; minimum when the three IIR Python-fallback points are held out). Consequently every A100 adaptive and estimation algorithm is overpredicted 2 to 5x (lms 380%, nlms 286%, rls 280%, ukf 173%, ekf 138%, apa 130%, kalman 102%). Coefficients transferred from the other GPU predict a GPU's typical configuration better than its own fit (A100 alpha_c, alpha_m on the 4090 with alpha_o, alpha_f refit: MdAPE 37.0%, 74% within 2x, versus 44.3% and 69.6% for the 4090's own fit).

A fat tail of algorithms is 10x to 300x underpredicted on both GPUs independent of the estimator: median (torch.median sorts each window; NCU shows 35.3M integer instructions where the TO model counts 4.0M TOs of comparisons), svd, pca, music and esprit (cuSOLVER iterative factorizations; 137 and 166 kernels per svd and pca call), iir_butter4 on the 4090 (fused kernel outside the alpha_f scope of F-014), and filterbank_32ch on the 4090 (unexplained: NCU FP ratio 1.02 at B = 1). direct_dft is 3 to 4x overpredicted on both GPUs although its instruction count matches NCU to 99.2%.

### Significance
- The accuracy claims in the submitted paper (r2 as the headline; "points cluster tightly along the identity line across five orders of magnitude") must be replaced by the error distribution above. Reviewer DHCe's objection is confirmed in substance, though the reported r2 was linear-space, not log-log.
- Unweighted linear least squares over five decades of energy yields coefficients that describe one to three algorithms rather than the population; a relative-error weighted NNLS keeps the linear physical model and gives each configuration equal weight (to be evaluated in EXP-CR-004 before any change to the paper).
- The three qualitative results (three sequential regimes, CPU vs GPU regimes, split-radix and complex-MAC corrections) and the ranking accuracy do not depend on this finding.

### Errata to earlier findings
- F-014 reported r2 improvements for the 4-parameter model on both GPUs; those r2 values are linear-space and remain correct as computed, but the per-LSTM error improvements there (3.5% to 4.2% in-sample) are in-sample values for the algorithm that alone identifies alpha_f; the LOAO error for LSTM is 92% and 90%.
- F-017 lists r2 = 0.947 / 0.982 as key results; the values are correct, the interpretation is superseded by this finding.

### Erratum 2026-08-17 (from EXP-CR-004)
The fat-tail list above misstates two directions: filterbank_32ch on the 4090 is 2x OVER-predicted (signed median ratio 1.98), not under; music is mixed across N (median ratio 1.62 on the 4090, 1.74 on the A100). esprit is 50x under on the 4090 but 1.75x over on the A100. The consistent 2 to 6x under-predictions on both GPUs are jpeg (0.16 / 0.29), welch (0.26 / 0.49), wiener (0.37 / 0.66) and periodogram (0.39 / 0.81); the 30x to 300x under-predictions are median, svd and pca on both GPUs and iir_butter4 (fused) on the 4090.

---

## F-020: Energy per Counted Operation Spans 60x Across FP Kernels; the Estimator Is Not the Fix; the IIR Fallback Launch Count Is Undercounted 4.4x
**Date**: 2026-08-17
**Status**: Validated for the estimator comparison and the energy-per-TO spread (EXP-CR-004 on the submission data set); Hypothesis for the launch-count corrections (profiler validation pending, EXP-CR-005)
**Relevant files**: analyze_estimators.py, data/camera_ready/exp_cr_004_estimators.json, exp_cr_004_console.txt, shared/to_model.py (KERNELS_PER_ITER), algorithms/filters.py (_lfilter_torch), LOGBOOK.md EXP-CR-004

### Observation
1. Estimator. A relative-error NNLS (minimize sum ((E_pred - E)/E)^2, alpha >= 0, same linear model) does not improve the 4090 (MdAPE 45.9% vs 44.3%; within 2x 58.7% vs 69.6%) and improves the A100 (MdAPE 39.6% vs 68.0%; within 2x 72.2% vs 54.8%) almost entirely by lowering alpha_o from 125 to 38 uJ/launch, i.e. by removing the influence of the three IIR Python-fallback points. Under either estimator head-to-head ranking is 22 to 24 / 30 in-sample and under LOAO. Relative-error fitting underpredicts the largest workloads about 2.5x (transformer 0.38, cnn 0.44 on the 4090) because squared relative error is bounded at -100% below and unbounded above.
2. Energy per counted compute TO (E_meas / T_c, parallel regime) is not a constant: per-algorithm medians on the 4090 run from direct_dft 4 fJ/TO through cnn 12, transformer 14, fft 18, fir 30, dct 52, welch 87, dwt_haar 242, to esprit 898, svd 1541, pca 6321, and median 208,864 fJ/TO; the A100 ordering is the same (direct_dft 5 ... svd 1010, pca 3062, median 260,058). By decade of T_c per invocation the median falls from about 90 fJ/TO at 1e10 to 12 to 14 fJ/TO at 1e13 to 1e14. The fitted alpha_c (13.6 / 15.2 fJ/TO) is the energy per TO of the largest, highest-throughput kernels, not of the population.
3. Launch counts. (a) KERNELS_PER_ITER["iir_butter4"] = 5, but the fallback _lfilter_torch issues about 22 launches per sample at order 4 (2 for y[n] plus 1 copy, 5 for each of three state updates, 4 for the last), so the A100 IIR points carry S_o about 4.4x too small; 125 uJ / 4.4 = 28 uJ per launch matches the 30 to 55 uJ implied by the other A100 sequential algorithms. (b) On the four NCU-profiled configurations that are in the fitted data, the residual energy per cuSOLVER launch after alpha_c T_c + alpha_m T_m is svd 457 uJ (137 kernels) and pca 396 uJ (166 kernels), against alpha_o = 385 uJ per Python-loop launch: counting library-internal launches in S_o would predict svd and pca within about 15% with no new parameter (137 x 385 = 53 mJ vs 64 mJ; 166 x 385 = 64 mJ vs 66 mJ), and it reinterprets alpha_o as a per-launch GPU idle cost rather than a Python-interpreter cost.

### Evidence
See LOGBOOK EXP-CR-004 results tables (estimator comparison, Part 3 per-algorithm and per-decade fJ/TO). The 19 parallel-algorithm B = 1 configurations needed for the launch diagnostic are not in the fitted data; the B = 1 JSONs on disk are from an earlier harness (5 s runs, 17 W idle, no thermal fields) and were not used.

### Decision
- Keep the v0 unweighted NNLS estimator (unchanged from the accepted paper); report LOAO ranges of all coefficients in Table 3.
- Fix launch counts rather than the estimator: correct the IIR fallback count after profiler validation; validate every KERNELS_PER_ITER entry with a torch.profiler census (EXP-CR-005); count library-internal launches for svd, pca, music, esprit in S_o if the census confirms the per-launch cost.
- Report the energy-per-TO spread as the paper's accuracy statement: the constant-energy-per-operation assumption holds within about 3x for high-throughput kernels and fails by orders of magnitude for sort, iterative factorization and serial code.

### Significance
This is the physics behind reviewer DHCe's objection: energy per operation on a GPU is a throughput variable. Ranking accuracy is robust to it (22 to 24 / 30 under every fold, estimator and cross-GPU transfer); absolute prediction is not. The 60x spread across FP kernels is itself a measured result of the paper.

---

## F-021: Measured Kernel-Launch Census: KERNELS_PER_ITER Is 2x Low for Four Algorithms and 4x Low for the IIR Fallback; cuSOLVER Launch Counts per Call
**Date**: 2026-08-17
**Status**: Validated (EXP-CR-005, Nsight Systems 2023.4.4 kernel traces on the RTX 4090, one bracketed invocation per configuration after 3 warmups; all 264 (algorithm, N, B, variant) configurations of both GPUs' data sets; deterministic across repeats)
**Relevant files**: census_kernels.py, data/camera_ready/exp_cr_005_kernel_census.{csv,json}, exp_cr_005_console.txt, shared/to_model.py (KERNELS_PER_ITER), LOGBOOK.md EXP-CR-005

### Observation
1. Kernels per outer iteration, table vs measured: lms 7 / 7.0; nlms 11 / 12.0; rls 12 / 13.0; particle_1k 20 / 21.0; nmf 14 / 15.1; mdct_audio 11 / 11.0 (within 10%); ukf 43 / 54.1 (+26%); apa_p4 15 / 28.0; kalman 15 / 31.0; ekf 22 / 39.0; fastica 13 / 27.1 (all about 2x); iir_butter4 Python fallback 5 / 22 per sample (17 kernels + 5 device-to-device memcpys at B = 1, 22 kernels at B = 2). Counts do not depend on N for the Python-loop algorithms (fixed iteration counts) and scale exactly with N for the fallback IIR and with frames for mdct. Kalman, EKF, UKF and APA also issue 3, 3, 11 and 4 memcpys per iteration (info flags from cuSOLVER inverse/solve routines).
2. Library-internal launches per call for parallel algorithms: svd 245 / 244 / 137 at N = 256 / 512 / 1024 (cuSOLVER switches from the Householder ormtr/gesvd path to batched Jacobi gesvdbj at N = 1024), plus 16 memcpys and 4 memsets; pca 166 at every N (+11, +3); esprit 19 (batched, B >= 2) or 44 (B = 1), 5 to 7 memcpys; transformer 24; music 16 to 18. All other parallel algorithms launch 1 to 14 kernels per call (fft 2, direct_dft 1, fir_direct 1, filterbank 1, cnn 10, wiener 10 to 11, dst 9 to 14). cuDNN LSTM at B = 1 launches 6 to 8 kernels at every N.
3. Per-command cost implied on the 4090 (energies from EXP-CR-004): svd N = 1024 64.4 mJ over 157 commands = 410 uJ each; pca 66.0 mJ over 180 = 367 uJ each; the Python-loop per-launch cost after correcting the counts falls from 385 uJ to roughly 200 to 385 uJ depending on algorithm.

### Significance
- The submitted paper's dispatch term used launch counts that are 2x low for four of eleven Python-loop algorithms and 4x low for the three A100 IIR points that dominate the A100 alpha_o fit (F-019); the sentence "ranging from 7 for LMS to 43 for UKF, determined by source code analysis" becomes "7 to 54, measured with Nsight Systems".
- Library-internal launches (cuSOLVER factorizations, eigendecompositions, transformer layers) were not counted at all although they cost the same order per launch as Python-issued launches; svd and pca, the two largest under-predictions in the paper, are quantitatively explained by their launch counts.
- Launch counts are now a measured input of the model, on the same footing as the NCU-validated instruction counts. The refit (EXP-CR-006) decides between (a) corrected counts for the Python-loop and IIR-fallback algorithms only and (b) a unified S_o that counts every GPU command of every algorithm.

### Errata to earlier findings
- F-013: the KERNELS_PER_ITER values for apa_p4, kalman, ekf, ukf and fastica were 1.3x to 2.1x low, and iir_butter4 (fallback) was 4.4x low; the qualitative conclusion (launches, not iterations) stands, the "8x to 2.5x" spread was computed with the wrong counts.
- F-014 / F-010: the A100 alpha_o = 125 uJ/launch was fitted with the 4.4x-undercounted IIR fallback; the population per-launch cost on the A100 is 30 to 55 uJ.

---

## F-022: Camera-Ready Model Decision: S_o = Measured GPU Commands per Invocation (Unified), and Framing
**Date**: 2026-08-17
**Status**: Decided (EXP-CR-006 results; confirmed by Muntaser Syed)
**Relevant files**: refit_launch_census.py, data/camera_ready/exp_cr_006_refit.json, exp_cr_006_console.txt, LOGBOOK.md EXP-CR-006

### Decision
1. Model: E = alpha_c T_c + alpha_m T_m + alpha_o S_o + alpha_f S_f with S_o = number of GPU commands (kernel launches + memcpy + memset) issued per invocation, measured with Nsight Systems for every configuration, applied to every algorithm (Python loops and library-internal launches alike). Estimator: unweighted NNLS (F-020). This is variant b-all of EXP-CR-006.
2. Resulting model on the submission data set: RTX 4090 alpha_c 13.6 fJ/TO, alpha_m 70.1 fJ/TO, alpha_o 233.8 uJ/command [LOAO 227, 248], alpha_f 55.6 uJ/step; A100 alpha_c 15.2, alpha_m 184.2, alpha_o 29.5 [29.5, 31.9], alpha_f 45.3. Accuracy: MdAPE 26.6% / 19.9% (LOAO 33.5 / 30.5), within 2x 77.5% / 73.0% (LOAO 70.3 / 68.3), within 3x 87.0% / 84.9%, r2 linear 0.971 / 0.992, r2 log10 0.67 / 0.56, head-to-head 27/30 and 22/30 (LOAO 25 and 20). To be regenerated by the v1 analysis pipeline after EXP-CR-002 / EXP-CR-003 (NCU may correct T_c for the 14 unprofiled algorithms).
3. Framing: the camera-ready is the archival, introductory paper for the framework applied to signal processing. It presents the model as it now stands and does not narrate the submitted version. Improvements are the method; findings (three regimes, host-dependent per-command cost 234 vs 30 uJ, throughput spread) are results; limitations are stated once as scope. No overclaiming, no apologizing.
4. Assumption to state in the paper: launch counts measured on the RTX 4090 (torch 2.6.0, CUDA 12.4) are applied to the A100 configurations; an A100 census (EXP-CR-007) is optional verification.

---

## F-023: Nsight Compute Completes the Instruction-Count Validation (37 of 37) and the cuFFT Sweep: Two Populations, and the Split-Radix Count Holds for Single-Kernel Plans
**Date**: 2026-08-17
**Status**: Validated (EXP-CR-002 and EXP-CR-003, run_ncu_v1.py, Nsight Compute 2024.1.0 on the RTX 4090; N = 4096 FFT reproduces F-001 exactly)
**Relevant files**: run_ncu_v1.py, data/ncu_profiles/ncu_summary_v1.json, data/ncu_profiles/v1/*.csv, ncu_v1_console.txt, LOGBOOK.md EXP-CR-002 and EXP-CR-003

### Observation
1. cuFFT (real-input transform via torch.fft on the 4090), FP32 instructions vs the split-radix count 4N log2 N - 6N + 8: 0.958 (N = 256), 1.057 (1024), 1.006 (4096), 0.654 (16384), 0.662 (65536); vs Cooley-Tukey 5N log2 N: 0.62, 0.72, 0.70, 0.47, 0.48. cuFFT runs a single radix-16 kernel (vector_fft_r2c, 16 elements per thread) up to 4096, an EPT-32 symmetric real-input kernel at 16384, and a two-pass 128 x 256 factorization at 65536.
2. The 14 algorithms not in F-001 fall into two populations. Analytically tractable kernels: esprit 0.98, rls 1.09, dwt_db4 1.13, fused IIR 0.77 (measured / analytical FP32 instructions). Tiny-matrix library calls inside Python loops execute fixed padded tiles regardless of the 4x4 problem: kalman 526x, ekf 395x, apa 192x, fastica 88x (QR), ukf 32x (Cholesky), with lms 4.0x and nlms 5.4x from per-thread and reduction overhead on 32-vectors; particle 8.6x (Box-Muller sampling, exp, prefix scan) and nmf 5.4x (skinny k = 8 matmuls as 32x32 tiles) in between; music 3.7x (cuSOLVER eigensolver plus spectrum search), the same class as svd/pca (F-009).
3. Per-launch fixed instruction cost is energetically negligible on the laptop (kalman: 16.4M FP32 instructions = 1.1 mJ against 1.6 J of launch cost) and is what the launch term prices; on the server host (30 uJ per command) it becomes visible for fastica, nmf and particle, matching their EXP-CR-006 residuals.
4. DRAM traffic exceeds the analytical T_m by 2 to 19x in every full profile (F-001 found 5 to 9x for the FFT-based composites): eager execution materializes intermediates.

### Decision
- T_c stays analytical (mathematical operation counts). Coefficients from EXP-CR-006 (b-all) stand.
- The paper's NCU figure covers all 37 algorithms with the two populations labeled; the split-radix paragraph reports the sweep (validated within 6% at N <= 4096; cuFFT beats even the split-radix count by 35% at N >= 16384 by changing plan); MUSIC joins SVD/PCA as implementation-dependent factorizations.
- Memory-count under-counting (materialized intermediates) is recorded as the next modeling target beyond this paper.

### Significance
The instruction-count validation is complete and honest: where the analytical count is the algorithm's kernel (FFT, GEMM, convolution, ESPRIT, RLS, DWT, fused IIR) it is right to within 2 to 13% (FFT within 6% up to 4096); where the algorithm calls a dense-linear-algebra routine on a 4x4 problem, the executed instructions are a fixed cost per launch, which is the physics of the launch term and not a failure of operation counting.

### Addendum 2026-08-17 (camera-ready pipeline)
fir_direct is reclassified from "analytical" to "padded": its measured/analytical instruction ratio is 32.3 (F-001 data, 8,458,241 vs 262,144), the same instruction count as the 32-channel filterbank (8,585,248), i.e. cuDNN executes a 32-output-channel convolution tile for one channel. Class counts become analytical 22 (median 1.03, range 0.59 to 1.80), padded 11 (3.98 to 526), factorization 3 (3.41 to 11.2).

---

## F-024: The Fusion Payoff Is K alpha_o / alpha_f per Timestep, Not alpha_o / alpha_f
**Date**: 2026-08-17
**Status**: Validated (arithmetic on the F-022 coefficients and the F-021 census)
**Relevant files**: analyze_results_v1.py (macros fusionGainMin/Max), paper/main.tex

### Observation
The submitted paper compared one Python-loop launch (alpha_o) to one fused timestep (alpha_f) and reported a "3 to 7x per-step" difference. A Python loop issues K commands per timestep (measured K = 7 for LMS to 54 for UKF, F-021), so fusing it into one kernel that executes one serial step per timestep changes the per-timestep cost from K alpha_o to alpha_f. With the F-022 coefficients: RTX 4090 alpha_o / alpha_f = 4.21, payoff 29x (K = 7) to 227x (K = 54); A100 alpha_o / alpha_f = 0.65 (a fused step costs one and a half commands), payoff 4.6x to 35x. alpha_o(4090) / alpha_o(A100) = 7.9.

### Significance
The camera-ready states the payoff as K alpha_o / alpha_f with the measured K range (29 to 227x on the laptop host, 5 to 35x on the server host) and describes alpha_o / alpha_f as the cost of a fused step in commands (a quarter of a command on the laptop, one and a half on the server). The earlier phrasing "kernel fusion pays nothing on a server" (used briefly in the drafting) was wrong and is not in the paper.

### Camera-ready status
Draft built locally on the real data at 6 pages (pdflatex + bibtex with the MLSP 2026 mlspconf.sty and IEEEbib.bst): no undefined references, no overfull lines, no non-ASCII characters, no double-dash in prose, Table 3 at 9 pt (guideline minimum), abstract about 190 words, copyright notice 979-8-3195-0884-3/26 from the official 2026 template. Files: paper/main.tex, paper/refs.bib, paper/numbers_v1.tex (generated), paper/figures/*.pdf (generated by analyze_results_v1.py). Final compilation on Overleaf by Muntaser.

---
