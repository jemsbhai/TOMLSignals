"""
Profile SVD and JPEG at multiple sizes for operation count scaling.
Usage: Run each command through ncu (see bottom of file for commands).
"""
import argparse
import sys
import json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algorithms.decomposition import DECOMPOSITION
from algorithms.compression import COMPRESSION


def profile_svd(N, D):
    """Profile SVD at specific N, D."""
    setup_fn, run_fn, _ = DECOMPOSITION["svd"]
    device = torch.device("cuda")
    state = setup_fn(signal_length=N, batch_size=1, precision="fp32",
                     device=device, n_features=D)
    for _ in range(3):
        run_fn(state)
    torch.cuda.cudart().cudaProfilerStart()
    run_fn(state)
    torch.cuda.cudart().cudaProfilerStop()


def profile_jpeg(N):
    """Profile JPEG at specific signal_length (image side = sqrt(N))."""
    setup_fn, run_fn, defaults = COMPRESSION["jpeg_q50"]
    device = torch.device("cuda")
    state = setup_fn(signal_length=N, batch_size=1, precision="fp32",
                     device=device, **defaults)
    for _ in range(3):
        run_fn(state)
    torch.cuda.cudart().cudaProfilerStart()
    run_fn(state)
    torch.cuda.cudart().cudaProfilerStop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", required=True, choices=["svd", "jpeg"])
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--D", type=int, default=64, help="n_features for SVD")
    args = parser.parse_args()

    if args.alg == "svd":
        print(f"Profiling SVD: N={args.N}, D={args.D}", file=sys.stderr)
        profile_svd(args.N, args.D)
    elif args.alg == "jpeg":
        import math
        side = int(math.sqrt(args.N))
        side = (side // 8) * 8
        n_blocks = (side // 8) ** 2
        print(f"Profiling JPEG: N={args.N}, side={side}, n_blocks={n_blocks}", file=sys.stderr)
        profile_jpeg(args.N)


if __name__ == "__main__":
    main()
