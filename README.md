# TOMLSignals: Transistor-Level Energy Analysis of Signal Processing Algorithms

Code, data, and profiler traces for

> M. Syed, M. Silaghi, S. Abujar, S. Akter Khushbu,
> "Transistor-Level Energy Analysis of Signal Processing Algorithms: Beyond Operation Counting,"
> IEEE International Workshop on Machine Learning for Signal Processing (MLSP), Atlanta, 2026.

The paper applies the TOML framework (Transistor Operations, a physics-grounded cost unit anchored
to CMOS switching costs) to 37 signal processing algorithms in 8 categories, measured on an RTX 4090
(laptop host) and an A100 SXM4 (server). The energy model is

    E = alpha_c * T_c + alpha_m * T_m + alpha_o * S_o + alpha_f * S_f

with T_c and T_m analytical operation and memory counts, S_o the number of GPU commands (kernel
launches, memory copies, memsets) per invocation measured with Nsight Systems, and S_f the number of
fused-sequential timesteps. Analytical counts are checked against Nsight Compute instruction counts
for all 37 algorithms.

## Layout

    algorithms/            the 37 algorithms (PyTorch, SciPy, scikit-learn, PyWavelets, torchaudio)
    shared/to_model.py     analytical TO counts per algorithm; get_seq_steps (v0) and get_seq_steps_v1 (measured S_o)
    shared/launch_counts.py measured GPU-command counts served from the Nsight Systems census
    run_full_suite.py      energy benchmark harness (NVML / RAPL, thermal settling, per-algorithm idle baseline)
    data/results/          RTX 4090 measurements (all_results.csv + per-configuration JSON)
    data/server_results/   A100 SXM4 measurements
    data/cpu_vs_gpu_comparison.csv
    data/ncu_profiles/     Nsight Compute instruction counts: ncu_summary.json (23 algorithms, F-001) and
                           ncu_summary_v1.json (all 37 + cuFFT size sweep), raw CSVs under v1/
    data/camera_ready/     Nsight Systems launch census (exp_cr_005_kernel_census.csv), cross-validation
                           and refit outputs, paper_numbers_v1.json (every number in the paper)
    paper/                 main.tex, refs.bib, numbers_v1.tex (generated), figures/ (generated)
    LOGBOOK.md             experiment log (EXP-CR-001 to EXP-CR-006 for the camera-ready)
    findings.md            numbered findings F-001 to F-024 with evidence, corrections recorded as errata

## Reproducing the paper's numbers and figures

The measurements are in the repository; the analysis needs numpy, scipy and matplotlib only
(the model code imports torch/torchaudio if present, for local torchaudio detection, but the
analysis does not need a GPU).

    python analyze_results_v1.py

writes `data/camera_ready/paper_numbers_v1.json`, `paper/numbers_v1.tex` (LaTeX macros used by
main.tex) and `paper/figures/*.pdf`, and prints the coefficients, error distributions,
leave-one-algorithm-out results, head-to-head ranking, cross-GPU transfer and the 37-algorithm
Nsight Compute table. Then compile `paper/main.tex` with the MLSP style (`mlspconf.sty`,
`IEEEbib.bst`).

The scripts behind the model decision, in the order they were run (all logged in LOGBOOK.md):

    analyze_cv.py             EXP-CR-001  error distribution, LOAO/LOCO/leave-one-category-out, cross-GPU transfer
    analyze_estimators.py     EXP-CR-004  estimator comparison, launch-term and energy-per-TO diagnostics
    census_kernels.py         EXP-CR-005  Nsight Systems launch census for every benchmark configuration (nsys backend)
    refit_launch_census.py    EXP-CR-006  refit with measured launch counts (five S_o definitions compared)
    run_ncu_v1.py             EXP-CR-002/003  Nsight Compute: cuFFT size sweep and the 14 algorithms not in F-001

Re-measuring requires the GPUs and Nsight tools: `run_full_suite.py` for energy (LibreHardwareMonitor
for the CPU RAPL counters on Windows), `census_kernels.py` for launch counts (Nsight Systems 2023.4 or
later; falls back to torch.profiler where its CUDA activity is available), `run_ncu_v1.py` for
instruction counts (Nsight Compute 2024.1). Kernel counts were measured on the RTX 4090 with
PyTorch 2.6 / CUDA 12.4 and applied to both GPUs' configurations, as stated in the paper.

## Provenance and conventions

Every experiment has a planned entry in LOGBOOK.md before it runs and a results entry after; input
files are recorded by SHA-256 in the JSON outputs; superseded code is kept as v0 alongside v1 rather
than overwritten (`analyze_results.py` / `analyze_results_v1.py`, `run_ncu_profile.py` /
`run_ncu_v1.py`, `get_seq_steps` / `get_seq_steps_v1`). Corrections to earlier findings are recorded as
dated errata in findings.md, never by editing the original entry.

## Related

TOML (the framework): M. Syed, M. Silaghi, S. Abujar, S. Akter Khushbu, "TOML Transistor Operations
for Machine Learning: A Physics-Grounded Energy Efficiency Framework," FLAIRS-39, 2026,
doi:10.32473/flairs.39.1.141781.
