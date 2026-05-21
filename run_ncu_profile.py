"""
TOMLSignals - Batch Nsight Compute Profiling
=============================================
Runs ncu for each algorithm and collects instruction counts.
Saves raw ncu CSV output and a summary comparison with TO predictions.

Usage:
  python run_ncu_profile.py
  python run_ncu_profile.py --algs fft direct_dft fir_direct

Author: Muntaser Syed
Date: May 2026
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.to_model import predict_to, TO_MODELS

# Key ncu metrics for TO validation
NCU_METRICS = [
    # FP32 instructions (each = 1 FMA dispatch = 1 MAC in TOML terms)
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",   # FP32 FMA
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",   # FP32 ADD
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",   # FP32 MUL
    # FP16
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",   # FP16 FMA
    "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",   # FP16 ADD
    "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",   # FP16 MUL
    # FP64
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",   # FP64 FMA
    "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",   # FP64 ADD
    "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",   # FP64 MUL
    # Integer (for comparisons, indexing)
    "sm__sass_thread_inst_executed_op_integer_pred_on.sum",
    # Memory
    "dram__bytes_read.sum",     # HBM reads
    "dram__bytes_write.sum",    # HBM writes
    # Special function unit (transcendentals: sin, cos, exp, rsqrt)
    "sm__sass_thread_inst_executed_op_fp_misc_pred_on.sum",
]

# Algorithms to profile with N and B
PROFILE_CONFIGS = [
    # Transforms — the main suspects for TO overcounting
    ("fft",           4096, 1),
    ("direct_dft",    1024, 1),
    ("dct",           4096, 1),
    ("dst",           4096, 1),
    ("dwt_haar",      4096, 1),
    ("stft",          4096, 1),
    ("hilbert",       4096, 1),
    # Filters
    ("fir_direct",    4096, 1),
    ("fir_fft",       4096, 1),
    ("wiener",        4096, 1),
    ("matched_filter",4096, 1),
    ("savgol",        4096, 1),
    ("median",        4096, 1),
    ("filterbank_32ch",4096, 1),
    # Spectral
    ("periodogram",   4096, 1),
    ("welch",         4096, 1),
    # Decomposition (parallel only)
    ("svd",           1024, 1),
    ("pca",           1024, 1),
    # ML
    ("cnn_denoiser",  4096, 1),
    ("lstm_denoiser", 1024, 1),
    ("transformer_denoiser", 1024, 1),
    # Compression
    ("jpeg_q50",      4096, 1),
    ("mdct_audio",    4096, 1),
]


def run_ncu(alg, N, B, output_dir):
    """Run ncu for a single algorithm and return parsed metrics."""
    csv_path = output_dir / f"ncu_{alg}_N{N}_B{B}.csv"

    cmd = [
        r'"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat"',
        "--metrics", ",".join(NCU_METRICS),
        "--csv",
        "--profile-from-start", "off",
        "python", "profile_single.py",
        "--alg", alg, "--N", str(N), "--B", str(B),
    ]
    cmd_str = " ".join(cmd)

    print(f"  Profiling {alg} N={N} B={B}...", end="", flush=True)

    try:
        result = subprocess.run(
            cmd_str, capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).resolve().parent),
            shell=True,
        )
        if result.returncode != 0:
            print(f" ERROR (rc={result.returncode})")
            if result.stderr:
                # Print last 5 lines of stderr
                for line in result.stderr.strip().split("\n")[-5:]:
                    print(f"    {line}")
            return None

        # Parse ncu CSV output from stdout
        metrics = parse_ncu_csv(result.stdout)
        print(f" OK ({len(metrics)} kernels)")
        return metrics

    except subprocess.TimeoutExpired:
        print(" TIMEOUT")
        return None
    except Exception as e:
        print(f" EXCEPTION: {e}")
        return None


def parse_ncu_csv(stdout):
    """Parse ncu --csv long-format output into per-kernel metric dicts.
    
    NCU CSV format: one row per (kernel, metric) pair.
    Columns: ID, ..., Kernel Name, ..., Metric Name, Metric Unit, Metric Value
    Values may contain commas inside quotes (e.g., "39,552").
    """
    import io
    lines = stdout.strip().split("\n")
    
    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if '"ID"' in line or line.startswith('ID,'):
            header_idx = i
            break
    
    if header_idx is None:
        return []
    
    # Parse CSV properly (handles quoted commas)
    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))
    
    # Group by kernel ID
    from collections import defaultdict
    kernels = defaultdict(dict)
    for row in reader:
        kid = row.get("ID", "")
        metric_name = row.get("Metric Name", "")
        metric_value = row.get("Metric Value", "0")
        # Clean value: remove commas, quotes, whitespace
        metric_value = metric_value.replace(",", "").replace('"', '').strip()
        try:
            val = float(metric_value)
        except ValueError:
            val = 0.0
        kernels[kid][metric_name] = val
        # Also store kernel name for reference
        if "Kernel Name" in row:
            kernels[kid]["_kernel_name"] = row["Kernel Name"]
    
    return list(kernels.values())


def summarize_kernels(kernel_metrics):
    """Sum instruction counts across all kernels for one algorithm run."""
    # Now each kernel_metrics entry is a dict with metric names as keys
    totals = {m: 0.0 for m in NCU_METRICS}
    
    for km in kernel_metrics:
        for metric in NCU_METRICS:
            if metric in km:
                totals[metric] += km[metric]

    # Derived: total FP32 instructions
    fp32_fma = totals.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
    fp32_add = totals.get("sm__sass_thread_inst_executed_op_fadd_pred_on.sum", 0)
    fp32_mul = totals.get("sm__sass_thread_inst_executed_op_fmul_pred_on.sum", 0)
    totals["fp32_total"] = fp32_fma + fp32_add + fp32_mul
    totals["fp32_fma"] = fp32_fma
    totals["fp32_add"] = fp32_add
    totals["fp32_mul"] = fp32_mul

    # FP16
    fp16_fma = totals.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum", 0)
    fp16_add = totals.get("sm__sass_thread_inst_executed_op_hadd_pred_on.sum", 0)
    fp16_mul = totals.get("sm__sass_thread_inst_executed_op_hmul_pred_on.sum", 0)
    totals["fp16_total"] = fp16_fma + fp16_add + fp16_mul

    # Special functions (transcendentals)
    totals["fp_misc"] = totals.get("sm__sass_thread_inst_executed_op_fp_misc_pred_on.sum", 0)

    # Integer
    totals["int_total"] = totals.get("sm__sass_thread_inst_executed_op_integer_pred_on.sum", 0)

    # Memory
    dram_read = totals.get("dram__bytes_read.sum", 0)
    dram_write = totals.get("dram__bytes_write.sum", 0)
    totals["dram_bytes_total"] = dram_read + dram_write
    totals["dram_words_32bit"] = (dram_read + dram_write) / 4

    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algs", nargs="*", default=None,
                        help="Specific algorithms to profile")
    args = parser.parse_args()

    output_dir = Path("data/ncu_profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = PROFILE_CONFIGS
    if args.algs:
        configs = [(a, n, b) for a, n, b in configs if a in args.algs]

    print(f"Profiling {len(configs)} algorithm configurations with Nsight Compute")
    print(f"Output: {output_dir}\n")

    results = []

    for alg, N, B in configs:
        kernel_metrics = run_ncu(alg, N, B, output_dir)
        if kernel_metrics is None:
            continue

        totals = summarize_kernels(kernel_metrics)
        n_kernels = len(kernel_metrics)

        # Get TO prediction
        to_pred = predict_to(alg, N, B)
        to_compute_macs = to_pred["to_compute"] / 5000  # Convert TOs back to MACs
        to_memory_words = to_pred["to_memory"] / 10000   # Convert TOs to HBM words

        ratio_fp = totals["fp32_total"] / to_compute_macs if to_compute_macs > 0 else 0
        ratio_mem = totals["dram_words_32bit"] / to_memory_words if to_memory_words > 0 else 0

        results.append({
            "algorithm": alg, "N": N, "B": B,
            "n_kernels": n_kernels,
            "fp32_fma": totals["fp32_fma"],
            "fp32_add": totals["fp32_add"],
            "fp32_mul": totals["fp32_mul"],
            "fp32_total": totals["fp32_total"],
            "fp16_total": totals.get("fp16_total", 0),
            "fp_misc": totals.get("fp_misc", 0),
            "int_total": totals.get("int_total", 0),
            "dram_bytes": totals["dram_bytes_total"],
            "dram_kb": totals["dram_bytes_total"] / 1024,
            "dram_words": totals["dram_words_32bit"],
            "to_pred_macs": to_compute_macs,
            "to_pred_mem_words": to_memory_words,
            "ratio_fp_actual_vs_predicted": ratio_fp,
            "ratio_mem_actual_vs_predicted": ratio_mem,
        })

    # Print summary
    print(f"\n{'='*100}")
    print(f"  NCU PROFILING SUMMARY: Actual vs Predicted Instruction Counts")
    print(f"{'='*100}")
    print(f"  {'Algorithm':20s} {'N':>6s} {'Kernels':>8s} {'FP32 actual':>14s} "
          f"{'TO pred MACs':>14s} {'FP ratio':>9s} {'DRAM words':>12s} "
          f"{'TO mem words':>12s} {'Mem ratio':>9s}")
    print(f"  {'-'*110}")

    for r in results:
        print(f"  {r['algorithm']:20s} {r['N']:>6d} {r['n_kernels']:>8d} "
              f"{r['fp32_total']:>14.0f} {r['to_pred_macs']:>14.0f} "
              f"{r['ratio_fp_actual_vs_predicted']:>9.3f}× "
              f"{r['dram_words']:>12.0f} {r['to_pred_mem_words']:>12.0f} "
              f"{r['ratio_mem_actual_vs_predicted']:>9.3f}×")

    # Save JSON
    with open(output_dir / "ncu_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_dir / 'ncu_summary.json'}")

    # Key findings
    if results:
        ratios = [r["ratio_fp_actual_vs_predicted"] for r in results if r["ratio_fp_actual_vs_predicted"] > 0]
        if ratios:
            print(f"\n  FP instruction ratio (actual/predicted):")
            print(f"    Mean:   {sum(ratios)/len(ratios):.3f}×")
            print(f"    Median: {sorted(ratios)[len(ratios)//2]:.3f}×")
            print(f"    Range:  {min(ratios):.3f}× — {max(ratios):.3f}×")


if __name__ == "__main__":
    main()
