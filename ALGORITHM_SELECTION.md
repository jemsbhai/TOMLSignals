# TOMLSignals - Expanded Algorithm Suite v2

## Design Philosophy

We want three things from this suite:

1. **Breadth**: cover every major SP category so reviewers see their area
2. **Depth**: multiple algorithms per task for head-to-head energy comparison
3. **Ablations**: systematically vary parameters to show how TO profiles shift

---

## Full Algorithm Suite (32 algorithms, 8 categories)

### Category 1: Transforms (7 algorithms)

| # | Algorithm | Complexity | TO profile | Implementation |
|---|-----------|-----------|------------|----------------|
| 1 | **FFT** (Cooley-Tukey, radix-2) | O(N log N) | Complex MAC, butterfly memory | torch.fft.fft |
| 2 | **Direct DFT** | O(N^2) | Complex MAC, dense matrix | Manual matmul |
| 3 | **DCT-II** | O(N log N) | Real MAC only | scipy.fft.dct / torch |
| 4 | **DST-II** (Discrete Sine) | O(N log N) | Real MAC only | scipy.fft.dst |
| 5 | **DWT** (Discrete Wavelet, Haar + Daubechies-4) | O(N) per level | Real MAC, multi-scale memory | pywt / manual conv |
| 6 | **STFT** | O(W * N/hop * log W) | Windowed FFT + overlap memory | torch.stft |
| 7 | **Hilbert Transform** | O(N log N) | FFT + mask + IFFT | scipy.signal.hilbert |

**Comparisons:**
- FFT vs Direct DFT: at what N does FFT become cheaper in TOs (not just FLOPs)?
- DCT vs DFT: real vs complex arithmetic, same structure. TO savings from real-only?
- DWT Haar vs Daubechies-4: 2-tap vs 8-tap filter. How does wavelet length affect TOs?
- FFT vs Hilbert: Hilbert is FFT + extra work. How much does the spectral masking add?

**Ablations:**
- Signal length: N = 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
- Precision: FP32 vs FP16
- Batch size: 1, 16, 64, 256

---

### Category 2: Filtering (8 algorithms)

| # | Algorithm | Complexity | TO profile | Implementation |
|---|-----------|-----------|------------|----------------|
| 8 | **FIR filter** (direct convolution) | O(N*M) | Pure MAC, streaming | torch.nn.Conv1d |
| 9 | **FIR filter** (FFT overlap-save) | O(N log N) | Complex MAC + memory overhead | Manual FFT-convolve |
| 10 | **IIR filter** (direct form II, Butterworth) | O(N*K) | MAC + sequential dependency | scipy.signal.lfilter / manual |
| 11 | **Median filter** | O(N*W log W) | Comparisons only, no MACs | scipy.ndimage.median_filter |
| 12 | **Savitzky-Golay filter** | O(N*W) | Polynomial fit MACs per window | scipy.signal.savgol_filter |
| 13 | **Wiener filter** (frequency domain) | O(N log N) | FFT + division + IFFT | scipy.signal.wiener |
| 14 | **Matched filter** | O(N*M) or O(N log N) | Correlation = convolution | scipy.signal.correlate |
| 15 | **Bandpass filter bank** (32 channels) | O(32*N*M) or O(32*N log N) | Parallel filters, memory for all channels | Manual or torchaudio |

**Comparisons:**
- FIR direct vs FIR-FFT: crossover point in TOs vs FLOPs for M = 8, 16, 32, 64, 128, 256, 512
- FIR vs IIR: same frequency response (e.g., 4th-order Butterworth). IIR is fewer ops but sequential.
- Median vs Savitzky-Golay: both smooth signals. Median is comparisons; SG is MACs. Which is cheaper?
- Wiener vs matched filter: both optimize SNR. Different TO profiles (division vs convolution).
- Matched filter direct vs FFT: same computation, direct O(N*M) vs FFT O(N log N).

**Ablations:**
- Filter length M: 4, 8, 16, 32, 64, 128, 256, 512, 1024
- Signal length N: 1024, 4096, 16384, 65536
- Filter bank channels: 8, 16, 32, 64
- Precision: FP32 vs FP16

---

### Category 3: Adaptive Filtering (4 algorithms)

| # | Algorithm | Complexity per sample | TO profile | Implementation |
|---|-----------|---------------------|------------|----------------|
| 16 | **LMS** | O(M) MACs | Simple MAC, streaming, low memory | Manual |
| 17 | **NLMS** (Normalized LMS) | O(M) MACs + 1 division | LMS + one division per step for normalization | Manual |
| 18 | **RLS** | O(M^2) MACs + rank-1 update | Dense matrix ops, division for gain | Manual |
| 19 | **Affine Projection** (APA, order P) | O(M*P + P^3) | Between LMS and RLS complexity | Manual |

**Comparisons:**
- LMS vs NLMS: identical except NLMS adds one division per step. What is the TO overhead of that single division across 10,000 adaptation steps?
- LMS vs RLS: classic convergence-vs-complexity tradeoff. TOML adds energy as a third axis.
- LMS vs NLMS vs APA vs RLS: full Pareto frontier of convergence speed vs TO cost.

**Ablations:**
- Filter length M: 16, 32, 64, 128, 256
- Number of adaptation steps: 1000, 10000, 100000
- APA projection order P: 2, 4, 8

---

### Category 4: State Estimation (4 algorithms)

| # | Algorithm | Complexity per step | TO profile | Implementation |
|---|-----------|-------------------|------------|----------------|
| 20 | **Kalman filter** (standard) | O(N_s^3) MACs + matrix inversion | Division + sqrt in inversion | Manual / filterpy |
| 21 | **Extended Kalman filter** (EKF) | O(N_s^3) + Jacobian eval | Kalman + nonlinear function eval | Manual |
| 22 | **Unscented Kalman filter** (UKF) | O((2*N_s+1) * N_s^2) | Sigma point propagation, no Jacobian | Manual / filterpy |
| 23 | **Particle filter** (SIR) | O(N_p * N_s) + resampling | Random memory access, comparisons | Manual |

**Comparisons:**
- Kalman vs EKF vs UKF: for a nonlinear system. Kalman is cheapest in FLOPs but wrong for nonlinear. EKF adds Jacobian. UKF avoids Jacobian but uses 2N+1 sigma points. TO comparison reveals true energy cost of handling nonlinearity.
- Kalman vs Particle filter: for large state dimension. Kalman has O(N_s^3) with expensive inversions. Particle filter has O(N_p * N_s) with cheap per-particle ops but expensive resampling.

**Ablations:**
- State dimension N_s: 2, 4, 8, 16, 32, 64
- Particle count N_p: 50, 100, 500, 1000, 5000, 10000
- Time steps: 100, 1000

---

### Category 5: Spectral Estimation (4 algorithms)

| # | Algorithm | Complexity | TO profile | Implementation |
|---|-----------|-----------|------------|----------------|
| 24 | **Periodogram** | O(N log N) | Single FFT + magnitude^2 | Manual |
| 25 | **Welch's method** | O(S * W log W) | S segment FFTs + averaging | scipy.signal.welch |
| 26 | **MUSIC** | O(N^3) eigendecomp + O(N * N_freq) search | Eigendecomp dominates: div, sqrt | Manual |
| 27 | **ESPRIT** | O(N^3) eigendecomp + O(N * K) LS | Eigendecomp + least squares | Manual |

**Comparisons:**
- Periodogram vs Welch: Welch is S periodograms averaged. How does energy scale with segments?
- Welch vs MUSIC: both estimate PSD/frequencies. MUSIC is way more TOs per element due to eigendecomp nonlinear operations. What is the TO ratio?
- MUSIC vs ESPRIT: both subspace methods, similar eigendecomp cost, different post-processing.

**Ablations:**
- Signal length N: 128, 256, 512, 1024, 2048
- Welch segment count S: 4, 8, 16, 32
- MUSIC/ESPRIT subspace dimension: 4, 8, 16
- Number of frequency search points: 256, 512, 1024

---

### Category 6: Decomposition / Subspace (4 algorithms)

| # | Algorithm | Complexity | TO profile | Implementation |
|---|-----------|-----------|------------|----------------|
| 28 | **SVD** (Singular Value Decomposition) | O(min(M,N)^2 * max(M,N)) | Iterative: divisions, square roots | torch.linalg.svd |
| 29 | **PCA** (via eigendecomposition) | O(N^2 * D) covariance + O(D^3) eigen | Matmul + eigendecomp | torch.pca_lowrank |
| 30 | **ICA** (FastICA) | Iterative: tanh/exp + matmul | Nonlinear activations (tanh = 15,000 TOs) each iteration | sklearn.decomposition.FastICA |
| 31 | **NMF** (Non-negative Matrix Factorization) | Iterative: matmul + division | Division at every update step | sklearn.decomposition.NMF |

**Comparisons:**
- SVD vs PCA: PCA is SVD on the covariance matrix. Which path has fewer TOs?
- PCA vs ICA: PCA uses eigendecomp (divisions, sqrt). ICA uses iterative tanh (15,000 TOs each). For the same dimensionality reduction, which costs more in transistor operations?
- ICA vs NMF: both iterative decompositions but ICA uses tanh while NMF uses division. Different nonlinear TO profiles.

**Ablations:**
- Matrix dimensions: 64x64, 128x128, 256x256, 512x512, 1024x1024
- Number of components: 4, 8, 16, 32
- ICA/NMF iterations: 10, 50, 100, 200

---

### Category 7: Compression / Coding (2 algorithms)

| # | Algorithm | Steps | TO profile | Implementation |
|---|-----------|-------|------------|----------------|
| 32 | **JPEG pipeline** (encode) | DCT-8x8 + quantize + zigzag + Huffman | DCT (MAC), quantize (division), Huffman (comparison + memory) | Manual per-block |
| 33 | **MDCT audio codec** (encode) | MDCT + psychoacoustic + quantize + Huffman | MDCT (MAC), masking (exp/log), quantize (div) | Manual |

**Comparisons:**
- JPEG step-by-step breakdown: which step dominates in TOs? DCT, quantization, or Huffman?
- JPEG vs MDCT: both are transform + quantize + entropy code. DCT vs MDCT transforms + the psychoacoustic model's exp/log operations.

**Ablations:**
- JPEG quality factor: 10, 50, 90 (affects quantization division values)
- Block size: 8x8, 16x16
- Image/signal size: 256x256, 512x512, 1024x1024

---

### Category 8: ML-Enhanced Signal Processing (3 algorithms)

| # | Algorithm | Architecture | TO profile | Implementation |
|---|-----------|-------------|------------|----------------|
| 34 | **1D CNN denoiser** (3 layers) | Conv1d + ReLU + skip | MAC + ReLU (100 TOs, very cheap) | PyTorch |
| 35 | **LSTM denoiser** (1 layer, 128 hidden) | LSTM + linear | MAC + sigmoid/tanh (18k/15k TOs) | PyTorch |
| 36 | **Small transformer denoiser** (2 layers, d=64) | Self-attention + FFN + softmax + layernorm | MAC + softmax (25k TOs) + layernorm | PyTorch |

**Comparisons (classical vs ML):**
- Wiener filter vs 1D CNN vs LSTM vs Transformer for denoising: same task, same input SNR, same output quality target. Which is cheapest in TOs at what signal length?
- FIR filter vs 1D CNN: both are effectively learned convolutions. Is a trained CNN more or less TO-efficient than a designed FIR for the same filtering task?

**Comparisons (ML architectures):**
- CNN (cheap ReLU) vs LSTM (expensive sigmoid/tanh) vs Transformer (expensive softmax): for the same task, how do activation function TOs change the energy ranking?
- This directly bridges to the TOMLTransformers work.

**Ablations:**
- Signal length: 256, 1024, 4096
- Model size: small (3 layers) vs medium (6 layers)
- Batch size: 1, 16, 64

---

## Cross-Cutting Ablations (applied to ALL algorithms)

### A1: Signal Length Scaling
How does total TO scale with input size N?
- N = 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
- Plot: TO vs N for each algorithm family (log-log)
- Expectation: FFT is O(N log N) in FLOPs but what in TOs? Memory component
  may change the effective scaling.

### A2: Precision Scaling
FP32 vs FP16 for all algorithms.
- FP32 MAC = 7,500 TOs, FP16 MAC = 5,000 TOs (1.5x)
- But memory also changes: FP16 is half the bytes = half the HBM TOs
- Which algorithms benefit more from FP16: compute-bound or memory-bound?

### A3: Batch Size Scaling
How does energy per sample change with batch size?
- Batch = 1, 4, 16, 64, 256
- Weight loading is amortized across the batch (same as transformer prefill).
  Algorithms with large filter coefficients should show better batch scaling.

### A4: CPU vs GPU
Run the same algorithms on CPU (via numpy) and GPU (via torch.cuda).
- CPU has no HBM, only DRAM and L1/L2 cache. Different memory hierarchy.
- GPU has HBM + SRAM. Massively parallel but higher idle power.
- For which algorithms is GPU more energy-efficient? TOML should predict the
  crossover based on parallelism and memory access patterns.

### A5: MCER Ranking
Compute MCER for all 36 algorithms at a reference signal length (N=4096).
- Rank by MCER from most compute-bound to most memory-bound.
- Hypothesis: transforms and filters (structured memory) are compute-bound.
  Decomposition methods (large matrix loads) are memory-bound.
  Adaptive filters (streaming, tiny state) are compute-bound.

### A6: Nonlinear Operation Amplification
For each algorithm, compute:
  TO_nonlinear / TO_total
Where TO_nonlinear = TOs from divisions, sqrt, exp, log, sigmoid, tanh, softmax.
- Hypothesis: FLOPs underestimates energy of algorithms with many nonlinear ops
  (Kalman, MUSIC, ICA, NMF, Wiener, psychoacoustic models) by the largest margin.
- Plot: FLOPs prediction error vs nonlinear TO fraction. Should be positively
  correlated.

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Algorithms | 36 |
| Categories | 8 |
| Head-to-head comparisons | 16 |
| Cross-cutting ablations | 6 |
| Signal lengths tested | 11 |
| Estimated total experiment runs | ~800 |
| Estimated GPU time (4090, local) | ~2 hours |

---

## Paper Figure Plan (updated)

1. **MCER ranking** bar chart: all 36 algorithms ranked by compute-vs-memory ratio
2. **Signal length scaling**: log-log plot, TO vs N for representative algorithms
3. **FIR crossover**: FIR-direct vs FIR-FFT in TOs vs FLOPs as function of filter length M
4. **Nonlinear amplification**: scatter plot of FLOPs error vs nonlinear TO fraction
5. **Head-to-head pairs**: grouped bars for each pair showing FLOPs rank vs TOML rank
6. **Precision impact**: FP32 vs FP16 TO ratio across algorithm categories
7. **Classical vs ML denoising**: energy per sample at multiple signal lengths
8. **Batch size efficiency**: energy per sample vs batch size for selected algorithms

---

## Head-to-Head Comparison Matrix (16 pairs)

| # | Algorithm A | Algorithm B | Task | What FLOPs says | What TOML predicts |
|---|------------|------------|------|----------------|-------------------|
| 1 | FFT | Direct DFT | Transform | FFT wins for N>16 | DFT may win for small N (simpler memory) |
| 2 | DCT | DFT | Transform | Same complexity | DCT cheaper (real-only, fewer TOs per MAC) |
| 3 | DWT Haar | DWT Daub-4 | Transform | Haar cheaper (2-tap vs 8-tap) | Same ranking but ratio is not 4x |
| 4 | FIR direct | FIR-FFT | Filtering | FFT wins past M~64 | Crossover shifts to M~128+ (complex overhead) |
| 5 | FIR | IIR | Filtering | IIR fewer ops (lower order) | IIR penalized by sequential execution on GPU |
| 6 | Median | Savitzky-Golay | Smoothing | SG fewer comparisons | Median is comparison-only (50 TO each), SG is MAC (5000 TO). Median wins |
| 7 | Wiener | Matched filter | SNR optimization | Similar complexity | Wiener has divisions (15k TO each), matched filter is pure MAC |
| 8 | LMS | NLMS | Adaptive | Almost identical | NLMS adds 1 division per step (~15k TO overhead) |
| 9 | LMS | RLS | Adaptive | RLS is M x costlier | Ratio differs due to memory locality |
| 10 | Kalman | UKF | Nonlinear estimation | UKF is (2N+1)x per step | Kalman has matrix inversion (div TOs), UKF avoids it |
| 11 | Kalman | Particle filter | State estimation | Depends on N_s vs N_p | Kalman has O(N_s^3) with inversions; PF has random memory |
| 12 | Welch | MUSIC | Spectral estimation | MUSIC is much costlier | Ratio is even larger in TOs due to eigendecomp nonlinear ops |
| 13 | SVD | PCA | Decomposition | PCA adds covariance step | Depends on M/N ratio |
| 14 | ICA | NMF | Decomposition | Similar iteration cost | ICA uses tanh (15k TO), NMF uses division (15k TO). Similar TO but different bottleneck |
| 15 | JPEG DCT vs quant step | JPEG step breakdown | Compression | DCT dominates FLOPs | Quantization divisions may be significant in TOs |
| 16 | Wiener vs 1D CNN | Classical vs ML denoising | Denoising | Depends on model size | Crossover at what signal length? |
