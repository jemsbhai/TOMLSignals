"""
TOMLSignals - Nsight Compute Profiling Script
==============================================
Runs a single algorithm once for ncu to profile.
Captures actual GPU instruction counts (FMA, ADD, MUL, memory bytes)
to validate/calibrate the theoretical TO model.

Usage:
  ncu --metrics sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum --csv python profile_single.py --alg fft --N 4096 --B 1

Author: Muntaser Syed
Date: May 2026
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms.transforms import TRANSFORMS
from algorithms.filters import FILTERS
from algorithms.adaptive import ADAPTIVE
from algorithms.estimation import ESTIMATION
from algorithms.spectral import SPECTRAL
from algorithms.decomposition import DECOMPOSITION
from algorithms.compression import COMPRESSION
from algorithms.ml_enhanced import ML_ENHANCED

ALL_ALGORITHMS = {}
for d in [TRANSFORMS, FILTERS, ADAPTIVE, ESTIMATION, SPECTRAL,
          DECOMPOSITION, COMPRESSION, ML_ENHANCED]:
    ALL_ALGORITHMS.update(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", required=True, help="Algorithm name")
    parser.add_argument("--N", type=int, default=4096, help="Signal length")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--list", action="store_true", help="List available algorithms")
    args = parser.parse_args()

    if args.list:
        for name in sorted(ALL_ALGORITHMS.keys()):
            print(f"  {name}")
        return

    if args.alg not in ALL_ALGORITHMS:
        print(f"Unknown algorithm: {args.alg}")
        print(f"Available: {sorted(ALL_ALGORITHMS.keys())}")
        sys.exit(1)

    setup_fn, run_fn, defaults = ALL_ALGORITHMS[args.alg]
    device = torch.device("cuda")

    # Setup
    state = setup_fn(
        signal_length=args.N,
        batch_size=args.B,
        precision="fp32",
        device=device,
        **defaults,
    )

    # Warmup (not profiled by ncu if we use --kernel-id)
    for _ in range(args.warmup):
        run_fn(state)

    # Profile this call — ncu will capture all kernels launched here
    torch.cuda.cudart().cudaProfilerStart()
    run_fn(state)
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
