"""
Parse NCU scaling CSV files and determine operation count formulas.
Run after: .\run_scaling_ncu.ps1
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

SCALING_DIR = Path("data/ncu_profiles/scaling")


def parse_ncu_csv(filepath):
    """Parse NCU CSV output, sum metrics across all kernels."""
    totals = {}
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            # Skip ==PROF== lines at the top
            lines = []
            for line in f:
                if line.startswith('"') or line.startswith('ID'):
                    lines.append(line)
                elif lines:  # already found header, keep all subsequent lines
                    lines.append(line)
            if not lines:
                print(f"  No CSV data found in {filepath}")
                return None

            reader = csv.DictReader(lines)
            for row in reader:
                metric = row.get("Metric Name", "")
                value_str = row.get("Metric Value", "0")
                # Strip quotes and commas from numbers like "643,008"
                value_str = value_str.replace('"', '').replace(',', '').strip()
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                totals[metric] = totals.get(metric, 0) + value
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return None

    fma = totals.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
    fadd = totals.get("sm__sass_thread_inst_executed_op_fadd_pred_on.sum", 0)
    fmul = totals.get("sm__sass_thread_inst_executed_op_fmul_pred_on.sum", 0)
    fp32_total = fma + fadd + fmul
    int_total = totals.get("sm__sass_thread_inst_executed_op_integer_pred_on.sum", 0)
    dram_read = totals.get("dram__bytes_read.sum", 0)
    dram_write = totals.get("dram__bytes_write.sum", 0)

    return {
        "fp32_fma": fma, "fp32_add": fadd, "fp32_mul": fmul,
        "fp32_total": fp32_total, "int_total": int_total,
        "dram_bytes": dram_read + dram_write,
    }


def analyze_svd():
    print("=" * 80)
    print("  SVD SCALING ANALYSIS")
    print("=" * 80)

    # Vary N, fixed D=64
    print(f"\n  --- Vary N, fixed D=64 ---")
    print(f"  {'N':>6s} {'FP32_total':>14s} {'FP32/ND^2':>12s} {'FP32_FMA':>14s}")
    D = 64
    data_N = {}
    for N in [256, 512, 1024, 2048, 4096]:
        fpath = SCALING_DIR / f"svd_N{N}_D64.csv"
        if not fpath.exists():
            print(f"  {N:>6d}  -- file not found --")
            continue
        result = parse_ncu_csv(fpath)
        if result is None:
            continue
        data_N[N] = result
        nd2 = N * D * D
        print(f"  {N:>6d} {result['fp32_total']:>14,.0f} {result['fp32_total']/nd2:>12.2f} {result['fp32_fma']:>14,.0f}")

    # Vary D, fixed N=1024
    print(f"\n  --- Vary D, fixed N=1024 ---")
    print(f"  {'D':>6s} {'FP32_total':>14s} {'FP32/ND^2':>12s} {'FP32_FMA':>14s}")
    N = 1024
    data_D = {}
    for D in [16, 32, 64, 128]:
        fpath = SCALING_DIR / f"svd_N1024_D{D}.csv"
        if not fpath.exists():
            print(f"  {D:>6d}  -- file not found --")
            continue
        result = parse_ncu_csv(fpath)
        if result is None:
            continue
        data_D[D] = result
        nd2 = N * D * D
        print(f"  {D:>6d} {result['fp32_total']:>14,.0f} {result['fp32_total']/nd2:>12.2f} {result['fp32_fma']:>14,.0f}")

    # Fit: FP32_total = a * N * D^2 + b * D^3
    if len(data_D) >= 2:
        print(f"\n  --- Least-squares fit: FP32 = a*N*D^2 + b*D^3 ---")
        Ds = sorted(data_D.keys())
        A_mat = np.array([[1024 * D**2, D**3] for D in Ds])
        b_vec = np.array([data_D[D]["fp32_total"] for D in Ds])
        # Least squares
        coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        a_coeff, b_coeff = coeffs
        print(f"  a (ND^2 coefficient): {a_coeff:.4f}")
        print(f"  b (D^3 coefficient):  {b_coeff:.4f}")
        print(f"\n  Predictions vs actual:")
        for D in Ds:
            pred = a_coeff * 1024 * D**2 + b_coeff * D**3
            actual = data_D[D]["fp32_total"]
            print(f"    D={D:>4d}: pred={pred:>14,.0f}  actual={actual:>14,.0f}  error={abs(pred-actual)/actual*100:.1f}%")

        # Cross-validate on N-varying data
        if data_N:
            print(f"\n  Cross-validation (vary N, D=64):")
            D = 64
            for N in sorted(data_N.keys()):
                pred = a_coeff * N * D**2 + b_coeff * D**3
                actual = data_N[N]["fp32_total"]
                print(f"    N={N:>5d}: pred={pred:>14,.0f}  actual={actual:>14,.0f}  error={abs(pred-actual)/actual*100:.1f}%")


def analyze_jpeg():
    print(f"\n{'='*80}")
    print(f"  JPEG SCALING ANALYSIS")
    print(f"{'='*80}")

    print(f"\n  {'N':>6s} {'Side':>6s} {'Blocks':>8s} {'FP32_total':>14s} {'FP32/block':>12s} {'FMA/block':>12s}")
    for N in [256, 1024, 4096, 16384]:
        side = int(math.sqrt(N))
        side = side - (side % 8)
        n_blocks = (side // 8) ** 2
        if n_blocks == 0:
            continue
        fpath = SCALING_DIR / f"jpeg_N{N}_side{side}.csv"
        if not fpath.exists():
            print(f"  {N:>6d}  -- file not found --")
            continue
        result = parse_ncu_csv(fpath)
        if result is None:
            continue
        print(f"  {N:>6d} {side:>6d} {n_blocks:>8d} {result['fp32_total']:>14,.0f} "
              f"{result['fp32_total']/n_blocks:>12,.0f} {result['fp32_fma']/n_blocks:>12,.0f}")

    print(f"\n  Expected per block: 2 x (8x8 matmul) = 1,024 FMA")
    print(f"  If FMA/block is constant across sizes → fixed overhead per block (cuBLAS)")
    print(f"  If FMA/block decreases with more blocks → overhead is amortized")


if __name__ == "__main__":
    if not SCALING_DIR.exists():
        print(f"No scaling data found at {SCALING_DIR}")
        print(f"Run: .\\run_scaling_ncu.ps1 first")
        sys.exit(1)
    analyze_svd()
    analyze_jpeg()
