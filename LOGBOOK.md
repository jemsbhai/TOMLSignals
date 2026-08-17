# TOMLSignals - Experimental Logbook

**Project**: TOMLSignals (IEEE MLSP 2026, accepted as poster, in proceedings)
**Researcher**: Muntaser Syed
**Policy**: Append-only. Every experiment gets a planned entry BEFORE it runs and a
results entry AFTER it completes. Corrections are dated addenda that reference the
original entry; nothing is deleted or rewritten. Publishable observations go to
`findings.md` (F-xxx) once the logbook entry is complete.

Entries before EXP-CR-001 (development through submission, May 2026) are recorded
only in `findings.md` (F-001 to F-017). This logbook starts at the camera-ready
revision.

---

## EXP-CR-001: Error Distribution, Cross-Validation, and Cross-GPU Transfer

**Date**: 2026-08-17 (EDT)
**Researcher**: Muntaser Syed
**Type**: computational (re-analysis of existing measurements, no new GPU runs)
**Status**: completed 2026-08-17 (run at 16:26 UTC)

### Motivation
MLSP 2026 reviewer DHCe (weak reject) argued that r2 = 0.947 / 0.982 overstates
accuracy across five decades of energy and that fitting 4 parameters on ~130
points with no held-out set does not distinguish a fit from a prediction.
Reviewer qq3z (weak accept) asked whether the framework predicts energy on an
unseen architecture without refitting coefficients. The submitted r2 was
computed in linear space (v0 `fit_four_parameter`), which is dominated by the
few joule-scale points; the paper never stated the definition.

### Hypothesis
H1. The four-parameter model's out-of-sample error under leave-one-algorithm-out
(LOAO) is close to its in-sample error (median absolute percentage error within a
few points of the in-sample value and most points within 2x), because 4
coefficients cannot memorize 138 / 126 points and each coefficient is identified
by many algorithms. Exception expected: alpha_f is identified only by cuDNN LSTM
at B = 1, so the LSTM fold reverts to the 3-parameter error (~90%).
H2. Head-to-head ranking accuracy under LOAO stays at or near 24/30 on both GPUs.
H3. Cross-GPU transfer: alpha_c and alpha_f transfer between the two GPUs within
their in-sample ratios (1.1x and 1.2x), while alpha_m and alpha_o do not (2.6x
and 3.1x); full transfer of all four coefficients therefore degrades error
mainly on memory-bound and Python-loop algorithms, and refitting only alpha_m
and alpha_o on the target recovers most of the in-sample accuracy.

### Independent variables
- Fold scheme: in-sample, LOAO (37 folds), LOCO (leave one configuration out),
  leave-one-category-out (8 folds)
- Coefficient transfer set: every subset of {alpha_c, alpha_m, alpha_o, alpha_f}
  kept from the source GPU, remainder refit on the target
- Direction: RTX 4090 -> A100 SXM4 and A100 SXM4 -> RTX 4090

### Dependent variables / metrics
- r2 in linear space (v0 definition) and in log10 space
- MAPE, MdAPE, p90 APE, max APE (percent of measured energy)
- Fraction of points within 1.5x / 2x / 3x of measurement
- Geometric-mean multiplicative error 10^mean(|log10(E_pred/E_meas)|)
- Head-to-head ranking accuracy over the same 8 pairs / 30 comparisons as v0
- Coefficient range across LOAO folds

### Control conditions
- Data set identical to the submission: `data/results/all_results.csv` (4090,
  138 points), `data/server_results/results/all_results.csv` plus the three IIR
  rerun JSONs (A100, 126 points); SHA-256 of every input recorded in the output
- Design matrix identical to v0: `analyze_results.compute_to_predictions`
  (unmodified) with has_torchaudio = True (4090) / False (A100)
- Fit identical to v0: scipy NNLS on [TO_c, TO_m, S_o, S_f]; the script aborts if
  its in-sample coefficients differ from v0 `fit_four_parameter` (rtol 1e-8)
- Baseline: single-parameter E = a * TO_total with the same metric set

### Protocol
1. `python analyze_cv.py` from the repo root (Windows PowerShell)
2. Paste console output into the results section below
3. Outputs: `data/camera_ready/exp_cr_001_cv_results.json`,
   `data/camera_ready/exp_cr_001_per_algorithm.csv`
4. Commit script and outputs; record commit SHA below

### Environment
- **Hardware**: analysis only (any CPU); measurements were RTX 4090 Laptop GPU
  and A100 SXM4 40 GB as documented in findings.md F-001 to F-017
- **Software**: Windows, Python 3.12, numpy / scipy (versions recorded in JSON)
- **Git commit**: 8df5da82a3e2618b8b1d734f19f21f1c4665964f (working tree dirty at run: analyze_cv.py, LOGBOOK.md, findings.md uncommitted; committed immediately after)
- **Config**: none (script constants only)
- **Input SHA-256 (first 16)**: 955d5fbc6cf9f6fb all_results.csv (4090); b3532cc3c62405fd all_results.csv (A100); 577c409951c148f6 / 9c2faf46d374873d / 0fe148c8d9b9b24b IIR rerun JSONs (A100 N=1024/16384/4096)
- **Seeds**: none (NNLS is deterministic)

### Results
Console: `data/camera_ready/exp_cr_001_console.txt`. Both GPUs pass the v0 coefficient check (rtol 1e-8); v0 r2 = 0.9467 (4090), 0.9823 (A100).

| Metric | 4090 in | 4090 LOAO | 4090 LOCO | A100 in | A100 LOAO | A100 LOCO |
|---|---|---|---|---|---|---|
| r2 linear | 0.947 | 0.731 | 0.819 | 0.982 | 0.654 | 0.896 |
| r2 log10 | -0.19 | -0.25 | -0.19 | 0.14 | 0.06 | 0.13 |
| MdAPE % | 44.3 | 49.4 | 45.2 | 68.0 | 76.2 | 68.1 |
| MAPE % | 53.3 | 60.1 | 54.4 | 93.5 | 100.2 | 94.4 |
| p90 APE % | 99.4 | 99.6 | 99.5 | 261 | 261 | 261 |
| within 1.5x | 39.9% | 31.9% | 38.4% | 34.9% | 26.2% | 34.1% |
| within 2x | 69.6% | 60.9% | 68.8% | 54.8% | 47.6% | 54.8% |
| within 3x | 80.4% | 78.3% | 80.4% | 74.6% | 71.4% | 74.6% |
| geo-mean mult. err | 2.95x | 3.30x | 2.98x | 2.48x | 2.77x | 2.50x |
| head-to-head | 24/30 | 22/30 | 24/30 | 24/30 | 22/30 | 24/30 |

Leave-one-category-out pooled: 4090 MdAPE 61.6, within 2x 51%, H2H 21/30; A100 MdAPE 72.4, within 2x 42%, H2H 23/30. Worst held-out categories: 4090 ml_enhanced 84.7% (0% within 2x), transform 71.4%; A100 adaptive 282%, compression 244%, estimation 120%.

In-sample by regime: 4090 parallel (94) MdAPE 51.9 / within 2x 55%; python-loop (41) 31.9 / 100%; fused (3) 3.9 / 100%. A100 parallel (79) 36.6 / 70%; python-loop (44) 134.4 / 25%; fused (3) 1.4 / 100%.

Baseline E = a*TO_total: MdAPE 84.9 / 89.4, within 2x 20% / 33%, r2 linear 0.20 / 0.19.

LOAO coefficient range (in-sample value in brackets):
- 4090: alpha_c 5.54 to 13.73 fJ/TO [13.61], min with transformer_denoiser held out; alpha_m 57.9 to 231.7 [71.8], max with transformer held out; alpha_o 363.6 to 401.2 uJ/launch [385.4]; alpha_f 0 (LSTM out) to 58.4 [55.7].
- A100: alpha_c 7.61 to 15.32 [15.20], min with transformer held out; alpha_m 163.5 to 323.0 [184.3], max with transformer held out; alpha_o 54.85 to 126.0 [125.2], min with iir_butter4 (Python fallback) held out; alpha_f 0 to 47.9 [45.3].

Per-algorithm in-sample MdAPE >= 90%: 4090: direct_dft 295 (over), iir_butter4 100.0, pca 99.7, median 99.3, svd 99.1, esprit 98.4, filterbank_32ch 97.7, music 92.0 (all under). A100: lms 380, nlms 286, rls 280, mdct_audio 261, direct_dft 198, ukf 173, ekf 138, apa_p4 130, fft 107, kalman 102 (over); pca 99.3, median 98.6, svd 98.3 (under). Best: lstm 3.5 / 4.2, transformer 5.7 / 1.7, ekf 4.3 (4090), ukf 5.0 (4090), hilbert 4.4 (A100), cnn 9.4 / 18.9. LOAO barely moves any algorithm except lstm (92% and 90%, alpha_f = 0), transformer (61.5% and 50.7%), filterbank_32ch (158% on 4090) and iir_butter4 (56.5% on A100).

Cross-GPU transfer: full 4090 -> A100 MdAPE 61.6 (A100 own fit 68.0), within 2x 44% (54.8%), H2H 24/30; full A100 -> 4090 MdAPE 54.2 (own fit 44.3), within 2x 51% (69.6%), H2H 24/30. Best partial: A100 alpha_c and alpha_m applied to the 4090 with alpha_o, alpha_f refit gives MdAPE 37.0 and within 2x 74%, better than the 4090's own least-squares fit. Scalar recalibration hurts or is neutral (k = 0.348: 82.7%; k = 1.112: 55.0%). Head-to-head is 24/30 under every transfer variant.

### Observations
1. H1 (no overfitting) holds: LOCO is indistinguishable from in-sample; LOAO costs 5 to 8 MdAPE points and 7 to 9 points of within-2x; ranking stays 22 to 24 / 30. Four coefficients cannot memorize 138 points. Excluding LSTM from LOAO changes aggregates by less than 1 point; the alpha_f identifiability caveat is real but numerically negligible.
2. In-sample accuracy is far below what r2 = 0.947 / 0.982 conveys: median error 44% / 68%, 30% / 45% of configurations more than 2x off, log-space r2 about zero (-0.19 / +0.14). The submitted sentence "points cluster tightly along the identity line across five orders of magnitude" is not supported by the data.
3. Unweighted linear least squares is controlled by the largest-energy algorithms. transformer_denoiser sets alpha_c and alpha_m on both GPUs (holding it out moves the 4090's alpha_c by 2.5x and alpha_m by 3.2x). The three IIR Python-fallback points set alpha_o on the A100 (S_o = O(N) launches; holding them out halves alpha_o from 125 to 55 uJ/launch), which is why every A100 adaptive and estimation algorithm is overpredicted 2 to 5x. The A100 r2 = 0.982 is largely IIR at N = 16384 plus the transformer.
4. Coefficients from the other GPU predict a GPU's typical configuration better than its own least-squares fit (A100 alpha_c, alpha_m on the 4090: MdAPE 37 vs 44; 4090 coefficients on the A100: 58 to 62 vs 68). The physical picture (alpha_c comparable across GPUs, alpha_m architecture-dependent) survives; the fitted values are hostage to the fit objective, not to physics.
5. Fat tail of TO-model (not estimator) failures, 10x to 300x under: median (comparison-only model; torch.median sorts each window: NCU 35.3M integer instructions vs 4.0M TOs of comparisons counted, ~440x), svd / pca / music / esprit (cuSOLVER iterative factorizations, 137 to 166 kernels per call for svd/pca), iir_butter4 on the 4090 (fused kernel outside the F-014 alpha_f scope), filterbank_32ch on the 4090 (NCU FP ratio 1.02 at B = 1, yet 40x under at B = 2048: unexplained). direct_dft is 3 to 4x OVER on both GPUs despite the 99.2% NCU instruction match: energy per FMA in a large dense complex matmul at high utilization is well below alpha_c.
6. Leave-one-category-out collapses when the coefficient-setting category is removed (4090 ml_enhanced held out: 85%; A100 adaptive held out: 282%), which is observation 3 from the other side.

### Interpretation
DHCe's concern is confirmed in substance, though not for the stated reason (the r2 was linear-space, not log-log): the reported r2 measures the fit to the joule-scale points and says nothing about the 100 uJ points. Three separable causes: (a) the estimator (unweighted linear LS across five decades: representativeness, not overfitting); (b) missing physics for library-launch-dominated and comparison-only algorithms; (c) utilization-dependent energy per operation (large dense matmuls at the top, tiny kernels at the bottom). Cross-validation is not the problem. Consequences for the camera-ready: report this error distribution in place of the tight-cluster claim; decide the estimator (a relative-error weighted NNLS keeps the linear physical model and gives each of the 138 configurations equal weight; see EXP-CR-004 proposal); fix TO counts where NCU evidence exists (median integer work; kernel-launch counts) rather than describe them. Nothing in these results changes the qualitative claims (three regimes, CPU vs GPU, split-radix and complex-MAC corrections, ranking accuracy); it changes the accuracy claims.

### Artifacts
- data/camera_ready/exp_cr_001_cv_results.json
- data/camera_ready/exp_cr_001_per_algorithm.csv

### Addendum 2026-08-17 (after EXP-CR-004)
Observation 5 above misread the direction of two algorithms: filterbank_32ch on the 4090 is 2x over-predicted (signed median ratio 1.98), not under; music is mixed across N (median ratio 1.62 on the 4090). esprit is 50x under on the 4090 only. F-019 carries the same erratum. The remaining statements stand.

---

## EXP-CR-002: Nsight Compute Validation of the Split-Radix FFT Count Across N

**Date**: 2026-08-17 (EDT)
**Researcher**: Muntaser Syed
**Type**: hardware (Nsight Compute instruction counts, RTX 4090 Laptop GPU)
**Status**: planned

### Motivation
Reviewer AQaA asked whether the split-radix correction is effective only at
N = 4096, the single size profiled for FFT in F-001. The paper claims the
correction propagates to all FFT-based algorithms at every N; that claim needs
the same instrument at more than one size.

### Hypothesis
H1. cuFFT's FP32 instruction count for a batch-1 complex FFT matches the
Duhamel-Vetterli split-radix count 4N log2 N - 6N + 8 to within a few percent
for every power-of-two N from 1024 to 65536 (predicted counts: 34,824;
172,040; 819,208; 3,801,096).
H2. At N = 256 (predicted 6,664) the ratio measured/predicted rises above 1
because fixed per-kernel costs (twiddle setup, thread-block reduction) are no
longer negligible against a small butterfly network.
H3. If cuFFT changes kernel plan (single-kernel vs multi-pass) across this range,
the ratio may step at the plan boundary; the step, if present, is small relative
to the 30% Cooley-Tukey overprediction.

### Independent variables
- N in {256, 1024, 4096, 16384, 65536}, B = 1, FP32, torch.fft.fft (cuFFT)

### Dependent variables / metrics
- FP32 FMA + FADD + FMUL instruction counts (sm__sass_thread_inst_executed_op_*),
  summed over all kernels of one invocation
- Number of kernels launched
- Ratio measured / split-radix and measured / Cooley-Tukey (5N log2 N)

### Control conditions
- Same profiling harness as F-001: `profile_single.py --alg fft --N <N> --B 1`,
  3 warmup calls, cudaProfilerStart/Stop around one call
- Same NCU metric list and CSV parsing as `run_ncu_profile.py` / F-001
- The N = 4096 measurement must reproduce F-001 (173,055 FP32 ops, 2 kernels)

### Protocol
1. Write `run_fft_scaling_ncu.ps1` (loop over N, `ncu --csv --profile-from-start off`)
   and `parse_fft_scaling.py` (parser + ratio table + JSON)
2. Run the PowerShell script; confirm N = 4096 reproduces F-001 before reading
   the other sizes
3. Paste parser output here; save `data/ncu_profiles/scaling/fft_N*_B1.csv` and
   `data/camera_ready/exp_cr_002_fft_scaling.json`

### Environment
- **Hardware**: RTX 4090 Laptop GPU
- **Software**: Windows, Python 3.12, PyTorch (version recorded), Nsight Compute
  2024.1.0 (`C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat`)
- **Git commit**: TBD at run
- **Seeds**: not applicable (deterministic instruction counts)

### Results / Observations / Interpretation
[to be filled after run]

---

## EXP-CR-003: Nsight Compute Profiling of the 14 Algorithms Not Profiled in F-001

**Date**: 2026-08-17 (EDT)
**Researcher**: Muntaser Syed
**Type**: hardware (Nsight Compute instruction counts, RTX 4090 Laptop GPU)
**Status**: planned

### Motivation
Reviewer AQaA asked why only 23 of 37 algorithms were profiled and which ones.
`run_ncu_profile.py` (v0) profiled the algorithms whose invocation is a bounded
set of library kernels; not profiled were the ten Python-loop sequential
algorithms (lms, nlms, rls, apa_p4, kalman, ekf, ukf, particle_1k, fastica, nmf),
music, esprit, dwt_db4, and iir_butter4. No technical obstacle was recorded.
`profile_single.py` supports all 37, so the correct response is to complete
the validation rather than explain the gap.

### Hypothesis
H1. Python-loop algorithms with closed-form per-step arithmetic (lms, nlms, rls,
apa_p4, kalman, ekf) match their TO_c formulas within ~20% once instruction
counts are summed over all kernels; dispatch overhead is not visible in
instruction counts, so these ratios test TO_c independently of alpha_o.
H2. Algorithms with library factorizations inside the loop (ukf: Cholesky;
particle_1k: resampling; music/esprit: cuSOLVER eigendecomposition; fastica/nmf:
iterative updates with fixed max_iter) show implementation-dependent ratios,
as SVD and PCA did (F-009), because the TO model uses analytical lower bounds.
H3. dwt_db4 behaves like dwt_haar (ratio ~1.5x from convolution overhead);
iir_butter4 (torchaudio lfilter, fused kernel) matches the 4th-order biquad
count within ~20%.

### Independent variables
- Algorithm (14), N following the v0 convention (4096 for filters/transforms/
  spectral, 1024 for decomposition/estimation-scale checks), B = 1

### Dependent variables / metrics
- FP32 instruction counts summed over all kernels; kernel count; DRAM bytes
- Ratio measured / TO_c-predicted MACs (same definition as `ncu_summary.json`)

### Control conditions
- Same NCU metric list, harness, and parser as F-001; v0 `ncu_summary.json` is
  not overwritten (new script writes `ncu_summary_v1.json` = v0 entries + new)
- Timeout raised from 300 s to accommodate 1,200 to 4,400 kernel replays

### Protocol
1. Write `run_ncu_profile_v1.py`: v0 configs plus the 14 new (alg, N, B) tuples,
   `--only-new` flag, output `ncu_summary_v1.json`, per-algorithm CSV kept
2. Run for the 14 algorithms; note per-algorithm wall time and any timeout
3. Regenerate Fig. 3 (NCU validation) with all profiled algorithms

### Environment
- **Hardware**: RTX 4090 Laptop GPU
- **Software**: Windows, Python 3.12, PyTorch, Nsight Compute 2024.1.0
- **Git commit**: TBD at run
- **Seeds**: not applicable

### Results / Observations / Interpretation
[to be filled after run]

---

## EXP-CR-004: Estimator Comparison and Launch-Term / Utilization Diagnostics

**Date**: 2026-08-17 (EDT)
**Researcher**: Muntaser Syed
**Type**: computational (existing measurements and existing NCU profiles; no new GPU runs)
**Status**: completed 2026-08-17 (run at 16:40 UTC); Part 2 incomplete, see Results

### Motivation
EXP-CR-001 (F-019) showed that unweighted linear least squares over five
decades of energy yields coefficients set by one to three algorithms
(transformer_denoiser for alpha_c / alpha_m; the IIR Python-fallback points for
the A100's alpha_o), median errors of 44% / 68%, and a fat tail of algorithms
10x to 300x under (median, svd, pca, music, esprit, iir_butter4 on the 4090,
filterbank_32ch on the 4090) plus direct_dft 3 to 4x over. Before any change to
the paper, decide with data (a) whether a relative-error estimator gives
population-representative coefficients, and (b) whether the tail is explained
by a per-kernel-launch cost and by utilization-dependent energy per operation.

### Hypotheses
H1. Relative-error NNLS (minimize sum of ((E_pred - E)/E)^2 subject to alpha >= 0,
same linear model, same physical coefficients) lowers in-sample and LOAO MdAPE on
both GPUs and raises the within-2x fraction by at least 10 points, at the cost of
larger absolute error on the largest-energy configurations and a lower linear r2;
head-to-head accuracy is unchanged or better.
H2. Under relative-error NNLS the LOAO coefficient ranges shrink (no single
held-out algorithm moves alpha_c or alpha_m by more than 1.5x) and alpha_c on the
two GPUs stays within 1.5x of each other.
H3. For the 23 NCU-profiled algorithms at B = 1 on the 4090, the residual energy
after alpha_c*T_c + alpha_m*T_m, divided by the NCU kernel count, is positive and
of similar magnitude (tens to hundreds of uJ) across the launch-heavy algorithms
(svd 137, pca 166, mdct 77, transformer 24, dst 12 kernels), and refitting those
23 points with an added per-launch term reduces their MdAPE substantially.
H4. Effective energy per compute TO (E_meas / T_c) for parallel configurations
falls with per-invocation T_c and flattens at large T_c, consistent with fixed
per-invocation costs dominating small workloads; direct_dft at large T_c sits
below alpha_c.

### Independent variables
- Estimator: v0 unweighted NNLS ("ls"); relative-error NNLS ("rel"); relative-error
  NNLS with each algorithm weighted equally regardless of its number of
  configurations ("rel_alg", robustness check only)
- Fold scheme: in-sample, LOAO, LOCO, leave-one-category-out; cross-GPU full
  transfer and the two most interpretable partial transfers
- Diagnostic subsets: 23 NCU-profiled algorithms at B = 1 (4090); parallel-regime
  configurations binned by decade of T_c (both GPUs)

### Dependent variables / metrics
- Same metric set as EXP-CR-001; per-algorithm MdAPE under each estimator;
  coefficient values and LOAO ranges under each estimator
- Residual per kernel (uJ), energy per NCU FP32 instruction (pJ), effective fJ/TO

### Control conditions
- Data, design matrix and validity filter identical to EXP-CR-001 (imports
  analyze_cv.load_gpu; SHA-256 recorded); "ls" must reproduce EXP-CR-001 exactly
- NCU inputs: v0 data/ncu_profiles/ncu_summary.json (unmodified)

### Protocol
1. `python analyze_estimators.py` from the repo root
2. Paste console output; save data/camera_ready/exp_cr_004_estimators.json and
   exp_cr_004_console.txt
3. Decide estimator and whether to proceed to kernel-count collection for all
   configurations (would become EXP-CR-005)

### Environment
- **Hardware**: analysis only
- **Software**: Windows, Python 3.12, numpy / scipy (recorded in JSON)
- **Git commit**: c05076beaa22a3645cfe51058c8aafd1257d9afd (working tree dirty at run: analyze_estimators.py and LOGBOOK.md uncommitted)
- **Seeds**: none

### Results
Console: `data/camera_ready/exp_cr_004_console.txt`; JSON: `exp_cr_004_estimators.json`. Control lines reproduce EXP-CR-001 exactly (ls: 44.3 / 0.9467 / 24-30 on the 4090; 68.0 / 0.9823 / 24-30 on the A100).

**Part 1, estimators (in-sample | LOAO):**

| | 4090 ls | 4090 rel | A100 ls | A100 rel |
|---|---|---|---|---|
| alpha_c fJ/TO | 13.6 [5.5, 13.7] | 5.5 [5.0, 10.7] | 15.2 [7.6, 15.3] | 8.0 [7.0, 12.1] |
| alpha_m fJ/TO | 71.8 [58, 232] | 137.7 [109, 166] | 184.3 [164, 323] | 170.0 [130, 250] |
| alpha_o uJ/launch | 385.4 [364, 401] | 308.6 [300, 326] | 125.2 [55, 126] | 38.1 [37, 43] |
| alpha_f uJ/step | 55.7 | 61.8 | 45.3 | 49.2 |
| MdAPE % | 44.3 / 49.4 | 45.9 / 50.4 | 68.0 / 76.2 | 39.6 / 43.8 |
| MAPE % | 53.3 / 60.1 | 47.3 / 55.4 | 93.5 / 100.2 | 44.1 / 52.3 |
| max APE % | 357 / 362 | 103 / 263 | 515 / 515 | 103 / 173 |
| within 2x | 69.6 / 60.9 | 58.7 / 52.2 | 54.8 / 47.6 | 72.2 / 60.3 |
| within 3x | 80.4 / 78.3 | 79.0 / 73.9 | 74.6 / 71.4 | 82.5 / 78.6 |
| r2 linear | 0.947 / 0.731 | 0.731 / 0.688 | 0.982 / 0.654 | 0.544 / 0.523 |
| head-to-head | 24 / 22 | 24 / 22 | 24 / 22 | 23 / 23 |

rel_alg is indistinguishable from rel (all metrics within 1 point). By regime, in-sample MdAPE (within 2x): 4090 parallel ls 51.9 (55%) vs rel 58.0 (40%); python-loop ls 31.9 (100%) vs rel 24.2 (98%). A100 parallel ls 36.6 (70%) vs rel 47.2 (67%); python-loop ls 134.4 (25%) vs rel 31.3 (80%). Under rel the largest workloads are underpredicted about 2.5x (signed median ratio pred/meas: 4090 transformer 0.38, cnn 0.44; A100 transformer 0.52) and most mid-size parallel algorithms sit at 0.4 to 0.6, i.e. rel trades the top of the range for the sequential middle. Cross-GPU: rel does not transfer better than ls (rel full transfer 65.3% / 75.0% MdAPE vs ls 61.6% / 54.2%).

**Part 2, launch diagnostic:** only 4 of 23 NCU configurations have a matching row in the fitted data set (mdct, pca, svd, lstm). The B = 1 configurations of the 19 parallel algorithms are not in `all_results.csv` (the paper's 138 points use B = 2048 or the largest feasible batch for parallel algorithms). B = 1 JSONs for those algorithms exist on disk but were produced by an earlier harness (5 s duration, idle 17 W, no thermal or sample fields, e.g. `fft_N4096_B1_fp32.json`) and are not comparable to the paper's measurements. On the four matched rows: residual energy per kernel after alpha_c T_c + alpha_m T_m is svd 457 uJ (137 kernels, E 64.4 mJ, model 1.8 mJ), pca 396 uJ (166 kernels, E 66.0 mJ, model 0.3 mJ), mdct 252 uJ (77 kernels, python-loop, S_o already counted), lstm 11.1 mJ per kernel (fused serial steps, not a launch cost). The svd and pca residual per launch (396 to 457 uJ) is the same magnitude as the Python-loop per-launch coefficient alpha_o = 385 uJ.

**Part 3, effective energy per compute TO (parallel regime):** per-algorithm median E_meas / T_c on the 4090: direct_dft 4, filterbank 8, music 9, cnn 12, lstm 13, transformer 14, fft 18, dst 29, fir_fft 30, fir_direct 31, matched 32, hilbert 32, stft 38, wiener 51, dct 52, periodogram 54, savgol 79, dwt_db4 79, welch 87, jpeg 174, dwt_haar 242, esprit 898, svd 1541, pca 6321, iir (fused, 4090) 76,290, median 208,864 fJ/TO. A100: direct_dft 5, esprit 10, music 10, cnn 13, filterbank 13, lstm 14, transformer 15, fft 20 ... jpeg 178, svd 1010, pca 3062, median 260,058. By decade of T_c per invocation (4090): 1e10: median 93 (n=17), 1e11: 37, 1e12: 37, 1e13: 12, 1e14: 14 fJ/TO. So the FP-throughput algorithms span 4 to about 250 fJ/TO (60x), and the sort / factorization / serial algorithms sit 3 to 5 orders of magnitude higher; alpha_c (13.6 / 15.2 under ls) equals the energy per TO of the largest, highest-throughput workloads (dense complex GEMM, transformer, CNN, cuDNN LSTM), not of the population.

**Code finding during interpretation (not from the run):** `KERNELS_PER_ITER["iir_butter4"] = 5` in shared/to_model.py ("Python fallback: ~5 ops per time step"), but the fallback `_lfilter_torch` in algorithms/filters.py issues about 22 kernel launches per sample for order 4 (mul, add, copy for y[n]; mul, mul, sub, add, copy for each of three state updates; mul, mul, sub, copy for the last). The A100's three IIR points therefore carry S_o about 4.4x too small; their per-launch cost of 125 uJ divided by 4.4 is about 28 uJ, in line with the 30 to 55 uJ that the other A100 sequential algorithms imply. This is a launch-count error, not a per-launch cost difference. To be validated by counting launches with torch.profiler (EXP-CR-005).

### Observations
1. H1 fails on the 4090 and holds on the A100. rel gives no MdAPE gain on the 4090 (45.9 vs 44.3) and loses 11 points of within-2x; on the A100 it halves the median error (39.6 vs 68.0) and adds 17 points of within-2x, but almost entirely by removing the IIR-driven alpha_o inflation, which the launch-count correction above addresses at the source. Ranking is 22 to 24 / 30 under either estimator.
2. H2 holds: LOAO coefficient ranges shrink under rel (no held-out algorithm moves alpha_c by more than 2x or alpha_o by more than 15%); alpha_c across GPUs 5.5 vs 8.0 (1.45x). But rel underpredicts the largest workloads 2.5x because squared relative error is bounded below (-100%) and unbounded above, which biases the fit low; the two estimators describe different ends of a 60x spread that no single coefficient covers.
3. H3 is supported where testable: svd and pca residual per cuSOLVER launch (457 and 396 uJ) equals alpha_o (385 uJ) within 20%, i.e. counting library launches in S_o would predict both to within about 15% without a new parameter (137 x 385 = 53 mJ vs 64 measured; 166 x 385 = 64 mJ vs 66 measured). The B = 1 diagnostic on the other 19 algorithms needs fresh measurements.
4. H4 holds: energy per TO falls from about 90 fJ at 1e10 T_c to 12 to 14 fJ at 1e13 to 1e14, and direct_dft sits at 4 to 5 fJ/TO. Effective energy per counted operation is a throughput (utilization) variable spanning 60x across FP kernels and 10^3 to 10^5 x for sort, factorization and serial kernels.
5. Correction to EXP-CR-001 observation 5: filterbank_32ch on the 4090 is 2x OVER-predicted (median ratio 1.98; LOAO 2.6x), not under; music is mixed across N (median ratio 1.62 on the 4090, 1.74 on the A100); esprit is 50x under on the 4090 but 1.75x over on the A100. jpeg (0.16 / 0.29), welch (0.26 / 0.49), wiener (0.37 / 0.66) and periodogram (0.39 / 0.81) are the consistent 2 to 6x under-predictions.

### Interpretation
Switching the estimator is not the fix: it moves the error between regimes without shrinking the spread, and its A100 gain is an artifact removable by correcting the IIR launch count. Keep the v0 unweighted NNLS (unchanged from the accepted paper), report the LOAO ranges of every coefficient in Table 3, and fix the launch counts: (i) correct KERNELS_PER_ITER for the IIR fallback and validate all entries with a profiler census; (ii) count library-internal launches (cuSOLVER svd, pca, eigendecomposition in music and esprit) in S_o, which the svd/pca residual per launch justifies quantitatively and which reinterprets alpha_o as a per-launch GPU idle cost rather than a Python-interpreter cost. The Part 3 spread is the paper's honest accuracy statement: the constant-energy-per-operation assumption holds within about 3x for high-throughput kernels and fails by orders of magnitude for sort, iterative factorization and serial code; that sentence, with the numbers, replaces "points cluster tightly along the identity line".

---

## EXP-CR-005: Kernel-Launch Census for Every Benchmarked Configuration

**Date**: 2026-08-17 (EDT)
**Researcher**: Muntaser Syed
**Type**: hardware (torch.profiler kernel counts on the RTX 4090; no energy measured)
**Status**: completed 2026-08-17 (nsys backend; 178 rows + 10 smoke rows, all 264 unique configurations covered)

### Motivation
F-020: the model's launch counts S_o come from a hand-derived table
(KERNELS_PER_ITER, "derived from source code analysis") that is wrong by about
4.4x for the IIR fallback, and library-internal launches (cuSOLVER svd/pca,
eigendecomposition in music/esprit) are not counted at all although the svd/pca
residual per launch (396 to 457 uJ) equals alpha_o (385 uJ). Every launch count
the model uses should be measured, the same way NCU measured the instruction
counts (F-001, F-002).

### Hypotheses
H1. For the ten Python-loop algorithms (lms, nlms, rls, apa_p4, kalman, ekf, ukf,
particle_1k, fastica, nmf) and mdct_audio, the profiler kernel count per outer
iteration matches KERNELS_PER_ITER within +-20%; for iir_butter4 (fallback) it is
about 22 per sample, not 5.
H2. svd, pca, music and esprit launch hundreds of kernels per call (svd 137 and
pca 166 at N = 1024 per F-001 NCU) with counts that vary slowly with N; the
other parallel algorithms launch at most about 15 kernels per call, so a
per-launch term is negligible (< 5% of energy) for them at the paper's batch sizes.
H3. Kernel counts are deterministic across repeats except possibly cuSOLVER
iterative routines (convergence-dependent), and independent of B for library ops.
H4. lstm_denoiser at B = 1 launches a small fixed number of kernels (cuDNN fused
recurrence), confirming its per-step cost is serial execution, not launches.

### Independent variables
- Every (algorithm, N, B) configuration present in the 4090 or A100 data set
  (union), all algorithms; iir_butter4 profiled on both the torchaudio path
  (4090) and the forced Python fallback (A100 configs)

### Dependent variables
- CUDA kernel launches, memcpy and memset counts per invocation (torch.profiler,
  CUDA activity), top kernel names by count; model S_o from get_seq_steps;
  ratio census / model; implied kernels per outer iteration

### Control conditions
- Same setup_fn / run_fn / defaults as the harness (ALL_ALGORITHMS), 3 warmup
  calls, one profiled call after torch.cuda.synchronize(); repeat = 2 for svd,
  pca, music, esprit to check determinism
- Configuration list read from the two all_results.csv files (SHA-256 recorded)

### Protocol
1. `python census_kernels.py` from the repo root (resumable: appends rows to
   data/camera_ready/exp_cr_005_kernel_census.csv; use --algs to restrict)
2. Paste the summary tables printed at the end
3. Outputs: exp_cr_005_kernel_census.csv, exp_cr_005_kernel_census.json

### Environment
- **Hardware**: RTX 4090 Laptop GPU
- **Software**: Windows, Python 3.12, PyTorch (version recorded), torch.profiler / Kineto
- **Git commit**: TBD at run
- **Seeds**: not applicable (kernel counts)

### Addendum 2026-08-17 (before the run)
torch 2.6.0+cu124 (Windows wheel) reports `supported_activities() = {CPU}`: Kineto is built without CUPTI, so torch.profiler cannot see kernels on this machine (its "Self CUDA" column is op-level CUDA-event timing, not launches). Backend switched to Nsight Systems 2023.4.4 (`nsys profile --trace=cuda --capture-range=cudaProfilerApi` per configuration in a child process; counts from `nsys stats --report cuda_gpu_kern_sum` and `cuda_gpu_mem_time_sum`). Same protocol otherwise (harness setup, 3 warmups, one bracketed invocation). The census script auto-selects nsys when torch.profiler lacks CUDA activity.

### Results
Console: `data/camera_ready/exp_cr_005_console.txt` (+ `exp_cr_005_smoke.txt` for fft/svd); CSV/JSON: `exp_cr_005_kernel_census.{csv,json}`. Backend nsys 2023.4.4; torch 2.6.0+cu124; RTX 4090 Laptop GPU. Every configuration was measured (no ERROR rows); repeats identical for svd, pca, music, esprit, fastica, nmf, particle.

**A. Python-loop algorithms, kernels per outer iteration, table (KERNELS_PER_ITER) vs census:**

| algorithm | table | census kernels/iter | ratio | memcpy per iter | outer iters | model S_o | census kernels |
|---|---|---|---|---|---|---|---|
| lms | 7 | 7.0 | 1.00 | 0 | 200 | 1400 | 1401 |
| nlms | 11 | 12.0 | 1.09 | 0 | 200 | 2200 | 2401 |
| rls | 12 | 13.0 | 1.09 | 0 | 100 | 1200 | 1304 |
| apa_p4 | 15 | 28.0 | 1.87 | 4 | 100 | 1500 | 2801 |
| kalman | 15 | 31.0 | 2.07 | 3 | 200 | 3000 | 6203 |
| ekf | 22 | 39.0 | 1.77 | 3 | 200 | 4400 | 7803 |
| ukf | 43 | 54.1 | 1.26 | 11 | 100 | 4300 | 5406 |
| particle_1k | 20 | 21.0 | 1.05 | 0 | 200 | 4000 | 4203 |
| fastica | 13 | 27.1 | 2.08 | 0 | 50 | 650 | 1354 |
| nmf | 14 | 15.1 | 1.08 | 0 | 50 | 700 | 756 |
| mdct_audio | 11 | 11.0 | 1.00 | 0 | 7/31/50 | 77/341/550 | 77/341/550 |
| iir_butter4 fallback (A100 configs) | 5 | 17.0 at B=1, 22.0 per sample at B=2 | 3.4 / 4.4 | 5 per sample at B=1, 0 at B=2 | B*N | 5*B*N | 17411 (N=1024, B=1), 69635 (4096, 1), 274647 (16384, 1); 22531 (1024, 2), 90115 (4096, 2) |

Counts are identical across N for every Python-loop algorithm (fixed iteration counts), so S_o does not depend on N except for mdct and the IIR fallback. IIR fallback: at B = 1 the five per-sample slice assignments (y[:, n] and four s[:, i]) are executed as 4-byte device-to-device memcpys, at B = 2 as copy kernels; kernels + memcpys = 22 per sample at both B, exactly the source-code count in F-020.

**B. Parallel and fused algorithms, kernels per invocation (min..max over the paper's configurations), memcpy, memset:** jpeg 10..13; pca 166 (11, 3) at all N; svd 245 / 244 / 137 at N = 256 / 512 / 1024 (16, 4); filterbank 1; fir_direct 1; fir_fft 7..9; iir torchaudio 8..9 (1 memcpy); matched 8..10; median 2; savgol 1; wiener 10..11; cnn 10; lstm 6..8 at every N including B = 1 (fused recurrence); transformer 24; esprit 19 (batched B >= 2) or 44 (B = 1, N = 512), 5..7 memcpy; music 16..18 (1 memcpy); periodogram 6..7; welch 7; dct 6..7; direct_dft 1 (1 memset); dst 9..14; dwt 2; fft 2 (3 at N = 16384); hilbert 5..6; stft 3. cuSOLVER changes algorithm for svd at N = 1024 (Householder ormtr/gesvd path with 244 to 245 kernels at N <= 512, batched Jacobi gesvdbj with 137 at N = 1024); pca (pca_lowrank) is a fixed 166-kernel schedule.

### Observations
1. H1 half holds. Six of twelve table entries are within 10% (lms, nlms, rls, particle, nmf, mdct); ukf is 26% low; apa, kalman, ekf and fastica are about 2x low; the IIR fallback is 3.4x (kernels) or 4.4x (kernels + memcpys) low. The submitted paper's "K ranging from 7 for LMS to 43 for UKF, determined by source code analysis" is 7 to 54 by measurement, and F-013's per-launch overhead spread was computed with the wrong K for four algorithms.
2. H2 holds: svd (137 to 245), pca (166), esprit (19 to 44), transformer (24) and music (16 to 18) are the launch-heavy parallel algorithms; every other parallel algorithm launches at most 14 kernels per call.
3. H3 holds: counts are deterministic across repeats and independent of B for library ops (svd, pca, music) and of N for Python loops; svd's count depends on N through cuSOLVER's algorithm choice.
4. H4 holds: cuDNN LSTM launches 6 to 8 kernels at every N at B = 1; its per-step cost is serial execution inside those kernels, not launches.
5. Per-command cost check on the 4090 (F-020 numbers, census counts): svd N = 1024 64.4 mJ over 137 + 16 + 4 = 157 GPU commands = 410 uJ each; pca 66.0 mJ over 166 + 11 + 3 = 180 = 367 uJ each; the Python-loop per-launch cost after correcting the counts is 385 uJ x (old S_o / census S_o), i.e. roughly 200 to 385 uJ depending on algorithm. Same order; consistent with one per-command cost for library and Python-issued launches.

### Interpretation
The dispatch term's inputs were partly wrong (2x for four algorithms, 4x for the IIR fallback that dominates the A100 fit) and incomplete (library launches uncounted). Both are now measured. Next: EXP-CR-006 refits the four-parameter model with census launch counts, in two definitions to be compared on identical folds: (a) census counts for the Python-loop and IIR-fallback algorithms only (S_o = 0 elsewhere, as in the paper), and (b) unified S_o = all GPU commands issued per invocation (kernels + memcpy + memset) for every algorithm, which is the same physics applied without exception. Modeling decision needed before the run: count kernels only, or kernels + memcpy + memset (the IIR B = 1 vs B = 2 result argues for all commands, since a 4-byte D2D copy is a launch with the same latency).

---

## EXP-CR-006: Refit of the Four-Parameter Model with Measured Launch Counts

**Date**: 2026-08-17 (EDT)
**Researcher**: Muntaser Syed
**Type**: computational (existing energy measurements + EXP-CR-005 census; no new GPU runs)
**Status**: completed 2026-08-17 (run at 19:39 UTC)

### Motivation
F-021: the model's launch counts were 2x low for four Python-loop algorithms,
4.4x low for the IIR fallback that dominates the A100 alpha_o fit, and zero for
library-internal launches (svd 137 to 245, pca 166, esprit 19 to 44,
transformer 24, music 16 to 18 per call) although svd and pca residuals imply
the same per-launch cost as Python-issued launches. Refit with measured counts
and decide the S_o definition for the camera-ready.

### Hypotheses
H1. With census counts for the Python-loop and IIR-fallback algorithms only
(variant a), the A100 alpha_o falls from 125 to 30 to 60 uJ/launch, the A100
python-loop MdAPE falls from 134% to below 50%, the 4090 python-loop error
changes by less than 10 points, and head-to-head stays at 24/30 or better.
H2. With a unified S_o counting every GPU command of every algorithm (variant b),
svd and pca move from about 99% APE (30x to 100x under) to within 2x on both
GPUs, esprit and music improve, alpha_o stays within 1.5x of variant a, and the
batched parallel algorithms' predictions change by less than 5% (their launch
share is negligible), so within-2x rises by roughly the svd/pca/esprit share of
the data (5 to 10 points).
H3. With corrected counts alpha_o(4090) / alpha_o(A100) is 5 to 8 (host-
dependent per-launch cost) and alpha_o / alpha_f on the A100 is about 1 (not 2.8),
while on the 4090 it stays between 4 and 7.
H4. Counting kernels + memcpy + memset fits at least as well as kernels only and
removes the B-dependence of the IIR fallback per-sample count.

### Independent variables
- S_o definition: v0 (must reproduce EXP-CR-001); a-kernels; a-all; b-kernels;
  b-all (a = census counts only where v0 had S_o > 0; b = unified, every
  algorithm; kernels vs kernels + memcpy + memset)
- Fold scheme: in-sample, LOAO, LOCO, leave-one-category-out; cross-GPU full
  transfer and keep-c,m-refit-o,f

### Dependent variables
- Same metric set as EXP-CR-001; coefficients with LOAO ranges; per-algorithm
  MdAPE and signed ratio; launch-term share of predicted energy by regime

### Control conditions
- Estimator: v0 unweighted NNLS (F-020 decision); data and design identical to
  EXP-CR-001 except the S_o column; census values from repeat 0 (repeats identical)
- Regime labels for the by-regime tables are the v0 labels in every variant

### Protocol
1. `python refit_launch_census.py` from the repo root
2. Paste console; save data/camera_ready/exp_cr_006_refit.json and console
3. Decide the S_o definition and record it in findings.md

### Environment
- **Hardware**: analysis only
- **Software**: Windows, Python 3.12, numpy / scipy
- **Git commit**: 762a68241df53923de9a73084d12ffe597cdf0fc (working tree dirty at run: refit_launch_census.py, LOGBOOK.md uncommitted)
- **Seeds**: none

### Results
Console: `data/camera_ready/exp_cr_006_console.txt`; JSON: `exp_cr_006_refit.json`. Both `[control]` lines reproduce EXP-CR-001 (44.3 / 0.9467 / 24-30; 68.0 / 0.9823 / 24-30). Census rows used: 160 (repeat 0). No configuration lacked a census row.

**Coefficients (in-sample [LOAO min, max]):**

| | 4090 v0 | 4090 a-all | 4090 b-kernels | 4090 b-all | A100 v0 | A100 a-all | A100 b-kernels | A100 b-all |
|---|---|---|---|---|---|---|---|---|
| alpha_c fJ/TO | 13.6 [5.5, 13.7] | 13.6 | 13.6 | 13.6 [5.5, 13.7] | 15.2 [7.6, 15.3] | 15.2 | 15.2 | 15.2 [7.6, 15.3] |
| alpha_m fJ/TO | 71.8 [58, 232] | 71.9 | 69.9 | 70.1 [56, 230] | 184.3 [164, 323] | 184.5 | 184.1 | 184.2 [163, 323] |
| alpha_o uJ/cmd | 385.4 [364, 401] | 233.7 [226, 248] | 262.3 [256, 281] | 233.8 [227, 248] | 125.2 [55, 126] | 29.5 [29.5, 31.8] | 38.2 [36.3, 38.2] | 29.5 [29.5, 31.9] |
| alpha_f uJ/step | 55.7 | 55.7 | 55.6 | 55.6 | 45.3 | 45.3 | 45.3 | 45.3 |
| alpha_o / alpha_f | 6.9 | 4.2 | 4.7 | 4.2 | 2.8 | 0.65 | 0.84 | 0.65 |

alpha_c, alpha_m and alpha_f are unchanged by the S_o definition to three digits (the launch term's median share of predicted energy for batched parallel algorithms is 4.5% on the 4090 and 0.3% on the A100 under b-*). a-kernels equals b-kernels and a-all equals b-all on alpha_o to 0.1 uJ.

**Metrics, in-sample | LOAO:**

| | 4090 v0 | 4090 a-all | 4090 b-kernels | 4090 b-all | A100 v0 | A100 a-all | A100 b-kernels | A100 b-all |
|---|---|---|---|---|---|---|---|---|
| MdAPE % | 44.3 / 49.4 | 33.5 / 44.9 | 24.9 / 32.3 | 26.6 / 33.5 | 68.0 / 76.2 | 19.9 / 30.5 | 31.1 / 37.5 | 19.9 / 30.5 |
| MAPE % | 53.3 / 60.1 | 47.1 / 53.7 | 40.6 / 47.3 | 40.8 / 47.4 | 93.5 / 100.2 | 41.8 / 47.2 | 45.1 / 50.3 | 41.4 / 46.9 |
| p90 APE % | 99.4 | 99.4 | 91.3 | 92.0 | 261 | 98.6 | 93.0 | 92.4 |
| within 1.5x | 39.9 / 31.9 | 50.0 / 44.2 | 58.0 / 52.2 | 55.1 / 50.0 | 34.9 / 26.2 | 59.5 / 53.2 | 57.9 / 52.4 | 59.5 / 54.0 |
| within 2x | 69.6 / 60.9 | 69.6 / 60.9 | 77.5 / 70.3 | 77.5 / 70.3 | 54.8 / 47.6 | 73.0 / 68.3 | 76.2 / 71.4 | 73.0 / 68.3 |
| within 3x | 80.4 / 78.3 | 80.4 / 78.3 | 87.7 / 84.8 | 87.0 / 84.8 | 74.6 / 71.4 | 84.9 / 81.7 | 84.9 / 81.7 | 84.9 / 81.7 |
| r2 linear | 0.947 / 0.731 | 0.970 / 0.758 | 0.972 / 0.756 | 0.971 / 0.757 | 0.982 / 0.654 | 0.992 / 0.917 | 0.992 / 0.919 | 0.992 / 0.916 |
| r2 log10 | -0.19 / -0.25 | -0.18 / -0.24 | 0.68 / 0.62 | 0.67 / 0.62 | 0.14 / 0.06 | 0.22 / 0.15 | 0.58 / 0.52 | 0.56 / 0.49 |
| geo mult err | 2.95 / 3.30 | 2.80 / 3.12 | 1.77 / 1.97 | 1.78 / 1.98 | 2.48 / 2.77 | 2.03 / 2.22 | 1.81 / 1.97 | 1.79 / 1.96 |
| head-to-head | 24 / 22 | 26 / 24 | 27 / 25 | 27 / 25 | 24 / 22 | 22 / 20 | 22 / 20 | 22 / 20 |

LOCO is within 1.5 points of in-sample for every variant. Leave-one-category-out b-all: 4090 MdAPE 36.6, within 2x 59%, H2H 24; A100 35.7, 64%, 21.

**By regime (v0 labels), in-sample MdAPE / within 2x:** 4090 parallel: v0 51.9 / 55% -> b-all 42.5 / 67%; python-loop 31.9 / 100% -> 8.7 / 100%; fused 3.9 -> 3.6. A100 parallel 36.6 / 70% -> 37.0 / 70%; python-loop 134.4 / 25% -> 10.5 / 77%; fused 1.4 -> 1.4.

**Per-algorithm in-sample MdAPE (signed ratio pred/meas), v0 -> b-all.** 4090: pca 99.7 (0.00) -> 34.7 (0.65); svd 99.1 (0.01) -> 38.0 (0.62); esprit 98.4 (0.02) -> 34.2 (0.66); music 92.0 (1.62) -> 65.6 (1.66); lms 65.9 (1.66) -> 1.9 (1.01); rls 58.6 -> 6.8; nlms 40.9 -> 6.7; apa 32.4 -> 9.7; kalman 29.5 -> 4.6; ukf 5.0 -> 1.9; ekf 4.3 -> 13.5; mdct 52.5 -> 7.0; fastica 35.1 -> 17.8; nmf 10.6 -> 27.1; particle 17.7 -> 25.0; transformer 5.7 -> 2.8; dwt_haar 32.8 -> 17.3; dst 41.3 -> 30.5; hilbert 38.7 -> 28.7; periodogram 61.3 -> 47.3; wiener 62.9 -> 53.1; dct 52.5 -> 45.7; matched 41.4 -> 37.2; savgol 30.2 -> 26.7; fft 47.3 (1.47) -> 59.6 (1.60); unchanged tail: direct_dft 296 (3.96 over), median 99.1 (0.01), iir fused 91.5 (0.08), filterbank 97.0 (1.97 over), jpeg 82.5 (0.17), welch 67.9 (0.32), fir_direct 42.0 (0.58), stft 42.4 (0.58). A100: lms 379.6 (4.80) -> 13.2 (1.13); nlms 286 -> 1.9; rls 280 -> 2.3; mdct 261 -> 6.0; ukf 173 -> 3.7; ekf 138 -> 10.0; apa 130 -> 19.6; kalman 101 -> 10.9; pca 99.3 (0.01) -> 82.3 (0.18); svd 98.3 (0.02) -> 86.0 (0.14); particle 88.6 (1.89) -> 53.1 (0.47); fastica 68.2 (0.32) -> 84.3 (0.16); nmf 45.8 (0.54) -> 84.7 (0.15); esprit 75.2 (1.75) -> 85.3 (1.85); music 74.0 -> 74.2; unchanged: direct_dft 198 (2.98 over), median 98.5, fft 108 (2.08 over), jpeg 71 (0.29), filterbank 63 (1.63 over), welch 50 (0.49).

**Cross-GPU transfer (b-all):** 4090 -> A100 full 52.8% / 48% (v0 61.6 / 44); keep c,f refit m,o 20.0% / 73%; keep c,m refit o,f 34.9% / 66%. A100 -> 4090 full 85.7% / 44% (v0 54.2 / 51; alpha_o differs 7.9x between hosts); keep c,m refit o,f 24.6% / 78% / H2H 27; keep c,f refit m,o 35.0% / 73%.

### Observations
1. H1 holds: with census counts the A100 alpha_o falls to 29.5 (all commands) or 38.2 (kernels) uJ, LOAO range within 8%, and the A100 python-loop MdAPE falls from 134% to 10.5% (77% within 2x); the 4090 python-loop error falls from 31.9% to 8.7%. Head-to-head improves on the 4090 (24 -> 26 in variant a; Kalman vs UKF at N = 256 to 4096 now correct) but drops on the A100 (24 -> 22: Kalman vs UKF at 256 to 4096 now wrong). F-013's note stands in mirror image: on the 4090 UKF is measured cheaper and the census (fewer commands for UKF: 6507 vs 6804) predicts it; on the A100 Kalman is measured cheaper (UKF's 11 device-to-host syncs per iteration cost more on that host), which a single per-command cost cannot express.
2. H2 holds on the 4090: unified S_o takes pca / svd / esprit from about 99% under to 35 to 38% (ratio 0.62 to 0.66), music from 92% to 66%, transformer from 5.7% to 2.8%, and the composite transforms improve 5 to 15 points; batched parallel predictions move by less than 5% (median launch share 4.5%). Overall 4090: MdAPE 44.3 -> 26.6, within 2x 69.6 -> 77.5, within 3x 80.4 -> 87.0, log10 r2 -0.19 -> 0.67, head-to-head 24 -> 27 / 30, LOAO 25 / 30. On the A100 the launch term at 29.5 uJ recovers only part of svd / pca (0.01 -> 0.18) and does not help esprit / music: cuSOLVER's synchronizing launches cost roughly 200 to 250 uJ each on the A100 versus 30 uJ for the Python-loop launches issued by the fast host, so per-command cost is not universal there. On the 4090 the two are 410 vs 234 uJ, close enough for one coefficient.
3. H3 holds: alpha_o(4090) / alpha_o(A100) = 7.9 (all commands) or 6.9 (kernels), consistent with F-010's directly measured 7x per-step ratio; alpha_o / alpha_f = 4.2 on the 4090 and 0.65 on the A100. The submitted "3.1x dispatch ratio due to laptop-CPU interpreter latency" and "alpha_o / alpha_f from 2.8x (A100) to 6.9x (4090)" must be replaced: per-command cost is host-dominated (8x between hosts), and on the server it is below the fused per-step cost.
4. H4 holds for the sequential regime (all commands: 8.7 vs 11.3 on the 4090, 10.5 vs 30.5 on the A100, because Kalman/EKF/UKF/APA's per-iteration device-to-host copies are synchronizing launches) and is a wash elsewhere (4090 MdAPE 26.6 vs 24.9, A100 19.9 vs 31.1; within-2x identical or within 3 points; ranking identical). fastica, nmf and particle become 3 to 6x under-predicted on the A100 under the corrected alpha_o: their per-iteration work is dominated by real kernel execution (bmm, QR, sort) that T_c under-counts, i.e. a compute-count problem, not a launch problem.
5. The remaining tail after the refit is the F-020 throughput spread: median (sort), fused IIR on the 4090, direct_dft 3 to 4x over (peak-efficiency complex GEMM), fft 1.6 to 2x over (cuFFT), filterbank 2x over (cuDNN conv), and the composite FFT-based algorithms 0.3 to 0.6 under, whose NCU DRAM traffic is 5 to 9x the analytical T_m (F-001 memory ratios): eager execution materializes every intermediate. That is a memory-count question for a follow-on, not for the camera-ready.

### Interpretation
Adopt the unified S_o counting every GPU command per invocation (kernels + memcpy + memset), variant b-all, as the camera-ready model: it is the same physics as the paper's dispatch term applied without exception, its inputs are measured (Nsight Systems) like the instruction counts (Nsight Compute), it removes the special-casing of Python loops versus library loops, and it is the best or tied-best variant on every metric that matters (sequential-regime error, within-2x, log-space r2, ranking) with coefficients that barely move under LOAO. Report honestly what remains: median error 27% / 20%, 77% / 73% of configurations within 2x, 87% / 85% within 3x, ranking 27 / 30 and 22 / 30 (25 and 20 under LOAO), the host-dependence of the per-command cost, and the tail. Assumption to state: launch counts were measured on the 4090 (torch 2.6.0 / CUDA 12.4) and applied to the A100 configurations; an A100 census (EXP-CR-007, about one Lambda hour) would verify the library counts on that architecture.

### Decision needed
Confirm b-all as the model for the camera-ready (record in findings F-022), or choose b-kernels.

---
