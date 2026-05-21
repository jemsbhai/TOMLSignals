"""
TOMLSignals - Core Analysis: TOML Predictions vs Measured Energy
=================================================================
Fits E = α_c × TO_compute + α_m × TO_memory per GPU.
Generates figures for the MLSP 2026 paper.

Usage:
  python analyze_results.py

Author: Muntaser Syed
Date: May 2026
"""

import csv
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.to_model import predict_to, TO_MODELS, TO, get_seq_steps, get_fused_steps

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping figures")


# =========================================================================
# DATA LOADING
# =========================================================================

@dataclass
class DataPoint:
    algorithm: str
    category: str
    signal_length: int
    batch_size: int
    precision: str
    delta_power_w: float
    energy_per_call_j: float
    time_per_call_us: float
    idle_power_w: float
    idle_temp_c: float
    mean_temp_c: float
    mean_clock_mhz: int
    power_samples: int
    idle_samples: int
    thermal_timed_out: bool
    gpu_name: str
    # Computed
    to_compute: float = 0.0
    to_memory: float = 0.0
    to_total: float = 0.0
    n_seq_steps: int = 0
    n_fused_steps: int = 0
    e_predicted: float = 0.0


def load_csv(path: str, gpu_name: str = "Unknown") -> List[DataPoint]:
    """Load results CSV into DataPoint list."""
    points = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gpu = row.get("gpu_name", gpu_name)
            if not gpu or gpu == "":
                gpu = gpu_name
            points.append(DataPoint(
                algorithm=row["algorithm"],
                category=row["category"],
                signal_length=int(row["signal_length"]),
                batch_size=int(row["batch_size"]),
                precision=row["precision"],
                delta_power_w=float(row["delta_power_w"]),
                energy_per_call_j=float(row["energy_per_call_j"]),
                time_per_call_us=float(row["time_per_call_us"]),
                idle_power_w=float(row["idle_power_w"]),
                idle_temp_c=float(row.get("idle_temp_c", 0)),
                mean_temp_c=float(row.get("mean_temp_c", 0)),
                mean_clock_mhz=int(row.get("mean_clock_mhz", 0)),
                power_samples=int(row.get("power_samples", 0)),
                idle_samples=int(row.get("idle_samples", 0)),
                thermal_timed_out=row.get("thermal_timed_out", "False") == "True",
                gpu_name=gpu,
            ))
    return points


def load_json(path: str) -> DataPoint:
    """Load a single JSON result into a DataPoint."""
    with open(path, "r") as f:
        d = json.load(f)
    return DataPoint(
        algorithm=d["algorithm"],
        category=d["category"],
        signal_length=d["signal_length"],
        batch_size=d["batch_size"],
        precision=d["precision"],
        delta_power_w=d["delta_power_w"],
        energy_per_call_j=d["energy_per_call_j"],
        time_per_call_us=d["time_per_call_us"],
        idle_power_w=d["idle_power_w"],
        idle_temp_c=d.get("idle_temp_c", 0),
        mean_temp_c=d.get("mean_temp_c", 0),
        mean_clock_mhz=d.get("mean_clock_mhz", 0),
        power_samples=d.get("power_samples", 0),
        idle_samples=d.get("idle_samples", 0),
        thermal_timed_out=d.get("thermal_timed_out", False),
        gpu_name=d.get("gpu_name", "Unknown"),
    )


# =========================================================================
# TO PREDICTION
# =========================================================================

def compute_to_predictions(points: List[DataPoint], has_torchaudio: bool = True):
    """Compute TO predictions and sequential step counts for each data point.

    Args:
        has_torchaudio: Whether torchaudio was available on the GPU that
            generated this data. Affects IIR sequential step classification.
            4090 Laptop: True (torchaudio available, IIR runs as fused kernel)
            A100 Lambda: False (torchaudio unavailable, IIR uses Python fallback)
    """
    missing = set()
    for p in points:
        if p.algorithm in TO_MODELS:
            result = predict_to(p.algorithm, p.signal_length, p.batch_size)
            p.to_compute = result["to_compute"]
            p.to_memory = result["to_memory"]
            p.to_total = result["to_total"]
            p.n_seq_steps = get_seq_steps(p.algorithm, p.signal_length, p.batch_size,
                                          has_torchaudio=has_torchaudio)
            p.n_fused_steps = get_fused_steps(p.algorithm, p.signal_length, p.batch_size)
        else:
            missing.add(p.algorithm)
    if missing:
        print(f"  WARNING: No TO model for: {missing}")


# =========================================================================
# FITTING
# =========================================================================

def fit_two_parameter(points: List[DataPoint], store_predictions: bool = True) -> Dict:
    """
    Fit E = α_c × TO_compute + α_m × TO_memory (no intercept).
    Uses ordinary least squares via normal equations.

    Args:
        store_predictions: If True, store predictions back into DataPoint.e_predicted.
            Set False for diagnostic fits (per-category) to avoid overwriting.

    Returns dict with alpha_c, alpha_m, r_squared, residuals, etc.
    """
    n = len(points)
    if n < 3:
        return {"error": "Too few data points"}

    # Build design matrix X (n × 2) and target y (n,)
    X = np.array([[p.to_compute, p.to_memory] for p in points])
    y = np.array([p.energy_per_call_j for p in points])

    # Filter out invalid points
    valid = (y > 0) & (X[:, 0] > 0) & np.isfinite(y) & np.isfinite(X).all(axis=1)
    X_valid = X[valid]
    y_valid = y[valid]
    n_valid = valid.sum()

    if n_valid < 3:
        return {"error": f"Only {n_valid} valid points after filtering"}

    # OLS: α = (X^T X)^{-1} X^T y
    XtX = X_valid.T @ X_valid
    Xty = X_valid.T @ y_valid

    # Check conditioning
    cond = np.linalg.cond(XtX)
    if cond > 1e12:
        print(f"  WARNING: Ill-conditioned design matrix (cond={cond:.2e})")

    alpha = np.linalg.solve(XtX, Xty)
    alpha_c, alpha_m = alpha

    # Predictions
    y_pred = X_valid @ alpha
    residuals = y_valid - y_pred

    # R² (no intercept version: 1 - SS_res / SS_total)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Relative errors
    rel_errors = np.abs(residuals) / y_valid * 100
    mean_rel_error = np.mean(rel_errors)
    median_rel_error = np.median(rel_errors)

    # Store predictions back
    if store_predictions:
        idx = 0
        for i, p in enumerate(points):
            if valid[i]:
                p.e_predicted = y_pred[idx]
                idx += 1

    return {
        "alpha_c": alpha_c,
        "alpha_m": alpha_m,
        "r_squared": r_squared,
        "n_points": int(n_valid),
        "n_filtered": int(n - n_valid),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "mean_rel_error_pct": mean_rel_error,
        "median_rel_error_pct": median_rel_error,
        "max_rel_error_pct": float(np.max(rel_errors)),
        "residuals": residuals,
        "y_valid": y_valid,
        "y_pred": y_pred,
        "cond_number": cond,
        "alpha_c_per_to_fJ": alpha_c * 1e15,  # Convert J/TO to fJ/TO
        "alpha_m_per_to_fJ": alpha_m * 1e15,
    }


def fit_single_parameter(points: List[DataPoint]) -> Dict:
    """
    Fit E = α × TO_total (single parameter baseline for comparison).
    """
    n = len(points)
    to_total = np.array([p.to_total for p in points])
    y = np.array([p.energy_per_call_j for p in points])

    valid = (y > 0) & (to_total > 0) & np.isfinite(y) & np.isfinite(to_total)
    to_valid = to_total[valid]
    y_valid = y[valid]

    # OLS: α = (to^T to)^{-1} to^T y
    alpha = np.dot(to_valid, y_valid) / np.dot(to_valid, to_valid)

    y_pred = alpha * to_valid
    residuals = y_valid - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    rel_errors = np.abs(residuals) / y_valid * 100

    return {
        "alpha": alpha,
        "r_squared": r_squared,
        "n_points": int(valid.sum()),
        "mean_rel_error_pct": float(np.mean(rel_errors)),
        "alpha_per_to_fJ": alpha * 1e15,
    }


def fit_three_parameter(points: List[DataPoint]) -> Dict:
    """
    Fit E = α_c × TO_compute + α_m × TO_memory + α_o × n_seq_steps (no intercept).

    The overhead term α_o × n_seq_steps captures the per-kernel-launch energy cost
    of Python-to-CUDA round trips in sequential algorithms (kernel launch
    latency × GPU idle power). This is a real hardware cost invisible to
    operation counting.
    """
    X = np.array([[p.to_compute, p.to_memory, p.n_seq_steps] for p in points])
    y = np.array([p.energy_per_call_j for p in points])

    valid = (y > 0) & (X[:, 0] > 0) & np.isfinite(y) & np.isfinite(X).all(axis=1)
    X_valid = X[valid]
    y_valid = y[valid]
    n_valid = valid.sum()

    if n_valid < 4:
        return {"error": f"Only {n_valid} valid points"}

    # Non-negative least squares (all coefficients must be >= 0)
    try:
        from scipy.optimize import nnls
        alpha, rnorm = nnls(X_valid, y_valid)
    except ImportError:
        alpha = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]

    alpha_c, alpha_m, alpha_o = alpha

    y_pred = X_valid @ alpha
    residuals = y_valid - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    rel_errors = np.abs(residuals) / y_valid * 100

    # Store predictions
    idx = 0
    for i, p in enumerate(points):
        if valid[i]:
            p.e_predicted = y_pred[idx]
            idx += 1

    return {
        "alpha_c": alpha_c,
        "alpha_m": alpha_m,
        "alpha_o": alpha_o,
        "r_squared": r_squared,
        "n_points": int(n_valid),
        "mean_rel_error_pct": float(np.mean(rel_errors)),
        "median_rel_error_pct": float(np.median(rel_errors)),
        "max_rel_error_pct": float(np.max(rel_errors)),
        "alpha_c_fJ": alpha_c * 1e15,
        "alpha_m_fJ": alpha_m * 1e15,
        "alpha_o_uJ": alpha_o * 1e6,  # µJ per sequential step
    }


def fit_four_parameter(points: List[DataPoint]) -> Dict:
    """
    Fit E = α_c·TO_c + α_m·TO_m + α_o·n_seq + α_f·n_fused (no intercept).

    Two sequential overhead regimes:
      α_o: Python-loop dispatch overhead per kernel launch (CPU-side cost)
      α_f: Fused-sequential per-timestep overhead (GPU-side cost, low utilization)

    These are physically distinct mechanisms with a measured 14× gap on
    RTX 4090 (Finding F-014). Python dispatch involves interpreter + CUDA
    driver overhead; fused-sequential is serial execution within a single
    C++/CUDA kernel.
    """
    X = np.array([[p.to_compute, p.to_memory, p.n_seq_steps, p.n_fused_steps]
                  for p in points])
    y = np.array([p.energy_per_call_j for p in points])

    valid = (y > 0) & (X[:, 0] > 0) & np.isfinite(y) & np.isfinite(X).all(axis=1)
    X_valid = X[valid]
    y_valid = y[valid]
    n_valid = valid.sum()

    if n_valid < 5:
        return {"error": f"Only {n_valid} valid points"}

    try:
        from scipy.optimize import nnls
        alpha, rnorm = nnls(X_valid, y_valid)
    except ImportError:
        alpha = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]

    alpha_c, alpha_m, alpha_o, alpha_f = alpha

    y_pred = X_valid @ alpha
    residuals = y_valid - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    rel_errors = np.abs(residuals) / y_valid * 100

    # Store predictions
    idx = 0
    for i, p in enumerate(points):
        if valid[i]:
            p.e_predicted = y_pred[idx]
            idx += 1

    return {
        "alpha_c": alpha_c,
        "alpha_m": alpha_m,
        "alpha_o": alpha_o,
        "alpha_f": alpha_f,
        "r_squared": r_squared,
        "n_points": int(n_valid),
        "mean_rel_error_pct": float(np.mean(rel_errors)),
        "median_rel_error_pct": float(np.median(rel_errors)),
        "max_rel_error_pct": float(np.max(rel_errors)),
        "alpha_c_fJ": alpha_c * 1e15,
        "alpha_m_fJ": alpha_m * 1e15,
        "alpha_o_uJ": alpha_o * 1e6,
        "alpha_f_uJ": alpha_f * 1e6,
    }


# =========================================================================
# ANALYSIS
# =========================================================================

def analyze_gpu(points: List[DataPoint], gpu_name: str, has_torchaudio: bool = True) -> Dict:
    """Full analysis for one GPU."""
    print(f"\n{'='*70}")
    print(f"  GPU: {gpu_name}")
    print(f"  Data points: {len(points)}")
    print(f"  torchaudio available: {has_torchaudio}")
    print(f"{'='*70}")

    compute_to_predictions(points, has_torchaudio=has_torchaudio)

    # Filter data quality
    valid_points = [p for p in points if p.energy_per_call_j > 0 and p.to_total > 0]
    negative = [p for p in points if p.energy_per_call_j <= 0]
    if negative:
        print(f"\n  WARNING: {len(negative)} points with non-positive energy filtered:")
        for p in negative:
            print(f"    {p.algorithm} N={p.signal_length} B={p.batch_size}: "
                  f"E={p.energy_per_call_j:.4e} J, ΔP={p.delta_power_w:.1f}W")

    print(f"\n  Valid data points: {len(valid_points)}")

    # Two-parameter fit: E = α_c × TO_compute + α_m × TO_memory
    fit2 = fit_two_parameter(valid_points)
    if "error" in fit2:
        print(f"  FIT ERROR: {fit2['error']}")
        return fit2

    print(f"\n  --- Model A: E = α_c·TO_compute + α_m·TO_memory (all data) ---")
    print(f"  α_c = {fit2['alpha_c']:.6e} J/TO  ({fit2['alpha_c_per_to_fJ']:.3f} fJ/TO)")
    print(f"  α_m = {fit2['alpha_m']:.6e} J/TO  ({fit2['alpha_m_per_to_fJ']:.3f} fJ/TO)")
    print(f"  r² = {fit2['r_squared']:.6f}")
    print(f"  Mean relative error: {fit2['mean_rel_error_pct']:.1f}%")

    # Three-parameter fit: E = α_c·TO_compute + α_m·TO_memory + α_o·n_seq_steps
    fit3 = fit_three_parameter(valid_points)
    if "error" not in fit3:
        print(f"\n  --- Model B3: E = α_c·TO_compute + α_m·TO_memory + α_o·n_seq (3-param) ---")
        print(f"  α_c = {fit3['alpha_c']:.6e} J/TO  ({fit3['alpha_c_fJ']:.3f} fJ/TO)")
        print(f"  α_m = {fit3['alpha_m']:.6e} J/TO  ({fit3['alpha_m_fJ']:.3f} fJ/TO)")
        print(f"  α_o = {fit3['alpha_o']:.6e} J/launch ({fit3['alpha_o_uJ']:.1f} µJ/launch)")
        print(f"  r² = {fit3['r_squared']:.6f}")
        print(f"  Mean relative error: {fit3['mean_rel_error_pct']:.1f}%")
    else:
        fit3 = None

    # Four-parameter fit: + α_f·n_fused_steps
    fit4 = fit_four_parameter(valid_points)
    if "error" not in fit4:
        print(f"\n  --- Model B: E = α_c·TO_c + α_m·TO_m + α_o·n_seq + α_f·n_fused (4-param) ---")
        print(f"  α_c = {fit4['alpha_c']:.6e} J/TO  ({fit4['alpha_c_fJ']:.3f} fJ/TO)")
        print(f"  α_m = {fit4['alpha_m']:.6e} J/TO  ({fit4['alpha_m_fJ']:.3f} fJ/TO)")
        print(f"  α_o = {fit4['alpha_o']:.6e} J/launch ({fit4['alpha_o_uJ']:.1f} µJ/launch)")
        print(f"  α_f = {fit4['alpha_f']:.6e} J/step ({fit4['alpha_f_uJ']:.2f} µJ/fused_step)")
        print(f"  r² = {fit4['r_squared']:.6f}")
        print(f"  Mean relative error: {fit4['mean_rel_error_pct']:.1f}%")
        print(f"  Median relative error: {fit4['median_rel_error_pct']:.1f}%")
        if fit4['alpha_f'] > 0:
            print(f"  α_o/α_f ratio: {fit4['alpha_o']/fit4['alpha_f']:.1f}×")
    else:
        fit4 = None

    # Parallel-only fit: 2-parameter on algorithms with 0 sequential + 0 fused steps
    parallel_points = [p for p in valid_points if p.n_seq_steps == 0 and p.n_fused_steps == 0]
    sequential_points = [p for p in valid_points if p.n_seq_steps > 0]
    fused_points = [p for p in valid_points if p.n_fused_steps > 0]
    print(f"\n  Data split: {len(parallel_points)} parallel, {len(sequential_points)} python-loop, {len(fused_points)} fused-sequential")

    fit2_par = fit_two_parameter(parallel_points)
    if "error" not in fit2_par:
        print(f"\n  --- Model C: E = α_c·TO_compute + α_m·TO_memory (parallel only) ---")
        print(f"  α_c = {fit2_par['alpha_c']:.6e} J/TO  ({fit2_par['alpha_c_per_to_fJ']:.3f} fJ/TO)")
        print(f"  α_m = {fit2_par['alpha_m']:.6e} J/TO  ({fit2_par['alpha_m_per_to_fJ']:.3f} fJ/TO)")
        print(f"  r² = {fit2_par['r_squared']:.6f}")
        print(f"  Mean relative error: {fit2_par['mean_rel_error_pct']:.1f}%")
    else:
        fit2_par = {}

    # Single-parameter fit for comparison
    fit1 = fit_single_parameter(valid_points)
    print(f"\n  --- Single-parameter model (baseline): E = α · TO_total ---")
    print(f"  α = {fit1['alpha']:.6e} J/TO  ({fit1['alpha_per_to_fJ']:.3f} fJ/TO)")
    print(f"  r² = {fit1['r_squared']:.6f}")
    print(f"  Mean relative error: {fit1['mean_rel_error_pct']:.1f}%")

    # Model comparison summary
    print(f"\n  --- MODEL COMPARISON ---")
    print(f"  {'Model':50s} {'r²':>10s} {'Err%':>8s} {'n':>5s}")
    print(f"  {'-'*75}")
    print(f"  {'A: α_c·TO_c + α_m·TO_m (all data)':50s} {fit2['r_squared']:>10.4f} {fit2['mean_rel_error_pct']:>7.1f}% {fit2['n_points']:>5d}")
    if fit3:
        print(f"  {'B3: + α_o·n_seq (3-param)':50s} {fit3['r_squared']:>10.4f} {fit3['mean_rel_error_pct']:>7.1f}% {fit3['n_points']:>5d}")
    if fit4:
        print(f"  {'B: + α_o·n_seq + α_f·n_fused (4-param)':50s} {fit4['r_squared']:>10.4f} {fit4['mean_rel_error_pct']:>7.1f}% {fit4['n_points']:>5d}")
    if fit2_par and 'r_squared' in fit2_par:
        print(f"  {'C: α_c·TO_c + α_m·TO_m (parallel only)':50s} {fit2_par['r_squared']:>10.4f} {fit2_par['mean_rel_error_pct']:>7.1f}% {fit2_par['n_points']:>5d}")
    print(f"  {'D: α·TO_total (baseline)':50s} {fit1['r_squared']:>10.4f} {fit1['mean_rel_error_pct']:>7.1f}% {fit1['n_points']:>5d}")

    # Per-category breakdown
    print(f"\n  --- Per-category r² (two-parameter) ---")
    categories = sorted(set(p.category for p in valid_points))
    for cat in categories:
        cat_pts = [p for p in valid_points if p.category == cat]
        if len(cat_pts) >= 3:
            cat_fit = fit_two_parameter(cat_pts, store_predictions=False)
            if "error" not in cat_fit:
                print(f"  {cat:20s}: r²={cat_fit['r_squared']:.4f}  "
                      f"n={cat_fit['n_points']}  "
                      f"err={cat_fit['mean_rel_error_pct']:.1f}%")

    # Per-algorithm residual table
    print(f"\n  --- Per-algorithm results ---")
    print(f"  {'Algorithm':23s} {'N':>6s} {'B':>5s} {'E_meas(J)':>12s} {'E_pred(J)':>12s} "
          f"{'Rel.Err%':>9s} {'TO_comp':>12s} {'TO_mem':>12s} {'MCER':>6s}")
    print(f"  {'-'*100}")
    for p in sorted(valid_points, key=lambda x: (x.category, x.algorithm, x.signal_length)):
        if p.e_predicted > 0:
            rel_err = abs(p.energy_per_call_j - p.e_predicted) / p.energy_per_call_j * 100
            mcer = p.to_memory / p.to_compute if p.to_compute > 0 else float('inf')
            print(f"  {p.algorithm:23s} {p.signal_length:>6d} {p.batch_size:>5d} "
                  f"{p.energy_per_call_j:>12.4e} {p.e_predicted:>12.4e} "
                  f"{rel_err:>8.1f}% {p.to_compute:>12.2e} {p.to_memory:>12.2e} {mcer:>6.2f}")

    # Head-to-head comparisons
    print(f"\n  --- Head-to-head pairs (same N, TOML correctly ranks?) ---")
    pairs = [
        ("fft", "direct_dft", "FFT vs DFT"),
        ("fir_direct", "fir_fft", "FIR direct vs FFT"),
        ("lms", "rls", "LMS vs RLS"),
        ("kalman", "ukf", "Kalman vs UKF"),
        ("periodogram", "welch", "Periodogram vs Welch"),
        ("cnn_denoiser", "lstm_denoiser", "CNN vs LSTM denoiser"),
        ("lstm_denoiser", "transformer_denoiser", "LSTM vs Transformer"),
        ("wiener", "cnn_denoiser", "Wiener vs CNN"),
    ]
    correct, total_pairs = 0, 0
    for alg_a, alg_b, label in pairs:
        pts_a = {p.signal_length: p for p in valid_points if p.algorithm == alg_a}
        pts_b = {p.signal_length: p for p in valid_points if p.algorithm == alg_b}
        common_n = set(pts_a.keys()) & set(pts_b.keys())
        for n in sorted(common_n):
            pa, pb = pts_a[n], pts_b[n]
            meas_a_wins = pa.energy_per_call_j < pb.energy_per_call_j
            pred_a_wins = pa.e_predicted < pb.e_predicted
            match = meas_a_wins == pred_a_wins
            if match:
                correct += 1
            total_pairs += 1
            winner_meas = alg_a if meas_a_wins else alg_b
            winner_pred = alg_a if pred_a_wins else alg_b
            status = "✓" if match else "✗"
            print(f"  {status} {label} (N={n}): "
                  f"meas={winner_meas}, pred={winner_pred}")
    if total_pairs > 0:
        print(f"\n  Ranking accuracy: {correct}/{total_pairs} = {correct/total_pairs*100:.0f}%")

    return {
        "gpu_name": gpu_name,
        "fit2": fit2,
        "fit3": fit3,
        "fit2_par": fit2_par,
        "fit1": fit1,
        "valid_points": valid_points,
    }


# =========================================================================
# FIGURES
# =========================================================================

def plot_scatter(results: List[Dict], output_dir: Path):
    """Main scatter plot: predicted vs measured energy (both GPUs)."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6),
                              squeeze=False)

    colors = {
        "transform": "#1f77b4",
        "filter": "#ff7f0e",
        "adaptive": "#2ca02c",
        "estimation": "#d62728",
        "spectral": "#9467bd",
        "decomposition": "#8c564b",
        "compression": "#e377c2",
        "ml_enhanced": "#7f7f7f",
    }

    for idx, res in enumerate(results):
        ax = axes[0, idx]
        fit2 = res["fit2"]
        points = res["valid_points"]

        for p in points:
            if p.e_predicted > 0 and p.energy_per_call_j > 0:
                c = colors.get(p.category, "gray")
                ax.scatter(p.e_predicted, p.energy_per_call_j,
                          c=c, s=25, alpha=0.7, edgecolors="none")

        # Perfect prediction line
        all_vals = [p.energy_per_call_j for p in points if p.e_predicted > 0]
        all_preds = [p.e_predicted for p in points if p.e_predicted > 0]
        lo = min(min(all_vals), min(all_preds)) * 0.5
        hi = max(max(all_vals), max(all_preds)) * 2
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="Perfect")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("TOML Predicted Energy (J)", fontsize=12)
        ax.set_ylabel("Measured Energy (J)", fontsize=12)
        fit_best = res.get("fit3") or res["fit2"]
        r2 = fit_best['r_squared']
        err = fit_best['mean_rel_error_pct']
        model_label = "3-param" if res.get("fit3") else "2-param"
        ax.set_title(f"{res['gpu_name']}\n"
                     f"r²={r2:.4f}, err={err:.1f}% ({model_label})",
                     fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=8, label=cat)
                       for cat, c in colors.items()]
    legend_elements.append(Line2D([0], [0], linestyle='--', color='k',
                                  label='Perfect prediction'))
    axes[0, -1].legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    path = output_dir / "fig1_predicted_vs_measured.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close()


def plot_mcer_analysis(results: List[Dict], output_dir: Path):
    """MCER vs relative error — shows where compute vs memory model matters."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results:
        points = res["valid_points"]
        mcers, errors, names = [], [], []
        for p in points:
            if p.e_predicted > 0 and p.to_compute > 0:
                mcer = p.to_memory / p.to_compute
                rel_err = abs(p.energy_per_call_j - p.e_predicted) / p.energy_per_call_j * 100
                mcers.append(mcer)
                errors.append(rel_err)
                names.append(p.algorithm)

        label = res["gpu_name"].split("-")[0].strip()  # "NVIDIA A100" or "NVIDIA GeForce RTX 4090"
        ax.scatter(mcers, errors, s=20, alpha=0.6, label=label)

    ax.set_xlabel("MCER (TO_memory / TO_compute)", fontsize=12)
    ax.set_ylabel("Relative Prediction Error (%)", fontsize=12)
    ax.set_title("Prediction Error vs Memory-to-Compute Energy Ratio")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fig2_mcer_vs_error.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_signal_length_scaling(results: List[Dict], output_dir: Path):
    """Energy vs N for key algorithms — shows scaling behavior."""
    if not HAS_MPL:
        return

    # Pick representative algorithms from different categories
    show_algs = ["fft", "fir_direct", "wiener", "lstm_denoiser", "transformer_denoiser"]

    for res in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        points = res["valid_points"]
        gpu_short = res["gpu_name"].replace("NVIDIA ", "")

        for alg in show_algs:
            alg_pts = sorted([p for p in points if p.algorithm == alg],
                            key=lambda p: p.signal_length)
            if not alg_pts:
                continue
            ns = [p.signal_length for p in alg_pts]
            e_meas = [p.energy_per_call_j for p in alg_pts]
            e_pred = [p.e_predicted for p in alg_pts if p.e_predicted > 0]
            ns_pred = [p.signal_length for p in alg_pts if p.e_predicted > 0]

            ax.plot(ns, e_meas, 'o-', label=f"{alg} (measured)", markersize=5)
            if e_pred:
                ax.plot(ns_pred, e_pred, 's--', alpha=0.5, markersize=4,
                       label=f"{alg} (TOML)")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Signal Length N", fontsize=12)
        ax.set_ylabel("Energy per Call (J)", fontsize=12)
        ax.set_title(f"Signal Length Scaling — {gpu_short}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        gpu_tag = gpu_short.replace(" ", "_").replace("-", "_")
        path = output_dir / f"fig3_scaling_{gpu_tag}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close()


def plot_cross_gpu(results: List[Dict], output_dir: Path):
    """Cross-GPU comparison: same algorithm, different hardware."""
    if not HAS_MPL or len(results) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Match algorithms by (algorithm, N)
    data_by_key = {}
    for i, res in enumerate(results):
        for p in res["valid_points"]:
            key = (p.algorithm, p.signal_length)
            if key not in data_by_key:
                data_by_key[key] = [None, None]
            data_by_key[key][i] = p.energy_per_call_j

    gpu_a_vals, gpu_b_vals = [], []
    for key, (va, vb) in data_by_key.items():
        if va is not None and vb is not None and va > 0 and vb > 0:
            gpu_a_vals.append(va)
            gpu_b_vals.append(vb)

    if gpu_a_vals:
        ax.scatter(gpu_a_vals, gpu_b_vals, s=20, alpha=0.6)
        lo = min(min(gpu_a_vals), min(gpu_b_vals)) * 0.5
        hi = max(max(gpu_a_vals), max(gpu_b_vals)) * 2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="Equal energy")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Energy per Call (J) — {results[0]['gpu_name']}", fontsize=10)
        ax.set_ylabel(f"Energy per Call (J) — {results[1]['gpu_name']}", fontsize=10)
        ax.set_title("Cross-GPU Energy Comparison")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fig4_cross_gpu.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# =========================================================================
# MAIN
# =========================================================================

def main():
    base = Path(__file__).resolve().parent
    fig_dir = base / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ---- Load local (4090) data ----
    local_csv = base / "data" / "results" / "all_results.csv"
    if local_csv.exists():
        local_points = load_csv(str(local_csv), gpu_name="NVIDIA GeForce RTX 4090 Laptop GPU")
        print(f"Loaded {len(local_points)} local (4090) data points")
    else:
        local_points = []
        print("WARNING: No local results found")

    # ---- Load server (A100) data ----
    server_csv = base / "data" / "server_results" / "results" / "all_results.csv"
    server_points = []
    if server_csv.exists():
        server_points = load_csv(str(server_csv), gpu_name="NVIDIA A100-SXM4-40GB")
        print(f"Loaded {len(server_points)} server (A100) data points from CSV")

        # Add IIR rerun JSONs (not in the main CSV)
        iir_dir = base / "data" / "server_results" / "results" / "filter"
        existing_iir = {(p.algorithm, p.signal_length) for p in server_points
                        if p.algorithm == "iir_butter4"}
        for jf in sorted(iir_dir.glob("iir_butter4_*.json")):
            p = load_json(str(jf))
            key = (p.algorithm, p.signal_length)
            if key not in existing_iir:
                server_points.append(p)
                existing_iir.add(key)
                print(f"  Added IIR rerun: N={p.signal_length} B={p.batch_size}")
    else:
        print("WARNING: No server results found")

    # ---- Analyze each GPU ----
    all_results = []

    if local_points:
        res_local = analyze_gpu(local_points, "NVIDIA GeForce RTX 4090 Laptop GPU",
                                has_torchaudio=True)
        all_results.append(res_local)

    if server_points:
        res_server = analyze_gpu(server_points, "NVIDIA A100-SXM4-40GB",
                                 has_torchaudio=False)  # torchaudio unavailable on Lambda
        all_results.append(res_server)

    # ---- Cross-GPU comparison ----
    if len(all_results) == 2:
        print(f"\n{'='*70}")
        print(f"  CROSS-GPU COMPARISON")
        print(f"{'='*70}")
        f1 = all_results[0]["fit2"]
        f2 = all_results[1]["fit2"]
        print(f"\n  {'Metric':30s} {'4090 Laptop':>15s} {'A100 SXM4':>15s} {'Ratio':>10s}")
        print(f"  {'-'*75}")
        print(f"  {'α_c (fJ/TO)':30s} {f1['alpha_c_per_to_fJ']:>15.3f} "
              f"{f2['alpha_c_per_to_fJ']:>15.3f} "
              f"{f1['alpha_c_per_to_fJ']/f2['alpha_c_per_to_fJ']:>10.2f}×")
        print(f"  {'α_m (fJ/TO)':30s} {f1['alpha_m_per_to_fJ']:>15.3f} "
              f"{f2['alpha_m_per_to_fJ']:>15.3f} "
              f"{f1['alpha_m_per_to_fJ']/f2['alpha_m_per_to_fJ']:>10.2f}×")
        print(f"  {'α_m / α_c (memory wall)':30s} "
              f"{f1['alpha_m']/f1['alpha_c']:>15.1f}× "
              f"{f2['alpha_m']/f2['alpha_c']:>15.1f}×")
        print(f"  {'r²':30s} {f1['r_squared']:>15.6f} {f2['r_squared']:>15.6f}")
        print(f"  {'Mean relative error':30s} {f1['mean_rel_error_pct']:>14.1f}% "
              f"{f2['mean_rel_error_pct']:>14.1f}%")

    # ---- Generate figures ----
    if HAS_MPL and all_results:
        print(f"\n{'='*70}")
        print(f"  GENERATING FIGURES")
        print(f"{'='*70}")
        plot_scatter(all_results, fig_dir)
        plot_mcer_analysis(all_results, fig_dir)
        plot_signal_length_scaling(all_results, fig_dir)
        if len(all_results) == 2:
            plot_cross_gpu(all_results, fig_dir)

    # ---- Summary JSON ----
    summary = {}
    for res in all_results:
        gpu = res["gpu_name"]
        entry = {
            "model_A_2param_all": {
                "alpha_c": res["fit2"]["alpha_c"],
                "alpha_m": res["fit2"]["alpha_m"],
                "r_squared": res["fit2"]["r_squared"],
                "mean_rel_error": res["fit2"]["mean_rel_error_pct"],
                "n_points": res["fit2"]["n_points"],
            },
            "model_D_1param": {
                "alpha": res["fit1"]["alpha"],
                "r_squared": res["fit1"]["r_squared"],
                "mean_rel_error": res["fit1"]["mean_rel_error_pct"],
            },
        }
        if res.get("fit3"):
            entry["model_B_3param_all"] = {
                "alpha_c": res["fit3"]["alpha_c"],
                "alpha_m": res["fit3"]["alpha_m"],
                "alpha_o": res["fit3"]["alpha_o"],
                "alpha_o_uJ_per_step": res["fit3"]["alpha_o_uJ"],
                "r_squared": res["fit3"]["r_squared"],
                "mean_rel_error": res["fit3"]["mean_rel_error_pct"],
                "n_points": res["fit3"]["n_points"],
            }
        if res.get("fit2_par") and "r_squared" in res["fit2_par"]:
            entry["model_C_2param_parallel"] = {
                "alpha_c": res["fit2_par"]["alpha_c"],
                "alpha_m": res["fit2_par"]["alpha_m"],
                "r_squared": res["fit2_par"]["r_squared"],
                "mean_rel_error": res["fit2_par"]["mean_rel_error_pct"],
                "n_points": res["fit2_par"]["n_points"],
            }
        summary[gpu] = entry
    with open(base / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved analysis_summary.json")


if __name__ == "__main__":
    main()
