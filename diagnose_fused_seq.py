"""
Diagnose 4-parameter model with fused-sequential term.

E = α_c·TO_c + α_m·TO_m + α_o·n_python_launches + α_f·n_fused_steps

Fused-sequential algorithms:
  - LSTM denoiser at B=1: cuDNN processes one timestep at a time internally
  - IIR with torchaudio (4090 only): torchaudio.lfilter runs as fused C++ kernel

These have NO Python dispatch overhead (α_o), but DO have per-timestep
sequential execution cost at low GPU utilization (α_f).

Author: Muntaser Syed
Date: May 2026
"""

import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.to_model import predict_to, TO_MODELS

# Kernel launches per outer iteration (from F-013)
KERNELS_PER_ITER = {
    "lms": 7, "nlms": 11, "rls": 12, "apa_p4": 15,
    "kalman": 15, "ekf": 22, "ukf": 43, "particle_1k": 20,
    "fastica": 13, "nmf": 14, "mdct_audio": 11, "iir_butter4": 5,
}


def get_outer_iters(alg, N, B, has_torchaudio=True):
    """Outer Python-loop iterations."""
    M = 32
    if alg in ("lms", "nlms"):
        return B * max(0, min(N, M + 200) - M)
    elif alg == "rls":
        return B * max(0, min(N, M + 100) - M)
    elif alg == "apa_p4":
        return B * max(0, min(N, M + 4 + 100) - (M + 4))
    elif alg in ("kalman", "ekf"):
        return B * min(N, 200)
    elif alg == "ukf":
        return B * min(N, 100)
    elif alg == "particle_1k":
        return B * min(N, 200)
    elif alg in ("fastica", "nmf"):
        return B * 50
    elif alg == "mdct_audio":
        return B * max(1, min((N - 2*512) // 512 + 1, 50))
    elif alg == "iir_butter4" and not has_torchaudio:
        return B * N
    return 0


def get_python_launches(alg, N, B, has_torchaudio=True):
    """Total Python-loop kernel launches."""
    outer = get_outer_iters(alg, N, B, has_torchaudio)
    kpl = KERNELS_PER_ITER.get(alg, 1)
    return outer * kpl


def get_fused_steps(alg, N, B, has_torchaudio=True):
    """Fused-sequential steps (cuDNN/C++ kernel, no Python dispatch).

    LSTM at B=1: cuDNN LSTM processes N timesteps sequentially.
    IIR with torchaudio: torchaudio.lfilter processes N samples sequentially.

    For LSTM, B>1 means the batch dimension allows parallelism across
    sequences, so the GPU can hide latency. At B=1, it's purely serial.
    We use a threshold: B <= threshold means fused-sequential.
    """
    if alg == "lstm_denoiser" and B <= 1:
        return N  # cuDNN LSTM at B=1 is serial
    if alg == "iir_butter4" and has_torchaudio:
        return B * N  # torchaudio.lfilter is fused-sequential
    return 0


def load_csv(path):
    points = []
    with open(path) as f:
        for row in csv.DictReader(f):
            points.append(row)
    return points


def run_rankings(data, alpha):
    """Run head-to-head ranking with 4-param model."""
    pairs = [
        ("fft", "direct_dft"), ("fir_direct", "fir_fft"),
        ("lms", "rls"), ("kalman", "ukf"),
        ("periodogram", "welch"), ("cnn_denoiser", "lstm_denoiser"),
        ("lstm_denoiser", "transformer_denoiser"), ("wiener", "cnn_denoiser"),
    ]
    lookup = {}
    for d in data:
        E_pred = (alpha[0] * d["to_c"] + alpha[1] * d["to_m"] +
                  alpha[2] * d["n_python"] + alpha[3] * d["n_fused"])
        lookup[(d["alg"], d["N"])] = (d["E"], E_pred)

    correct, total = 0, 0
    details = []
    for alg_a, alg_b in pairs:
        common = sorted(set(N for a, N in lookup if a == alg_a) &
                        set(N for a, N in lookup if a == alg_b))
        for n in common:
            ea_m, ea_p = lookup[(alg_a, n)]
            eb_m, eb_p = lookup[(alg_b, n)]
            match = (ea_m < eb_m) == (ea_p < eb_p)
            total += 1
            if match:
                correct += 1
            status = "✓" if match else "✗"
            wm = alg_a if ea_m < eb_m else alg_b
            wp = alg_a if ea_p < eb_p else alg_b
            details.append(f"  {status} {alg_a} vs {alg_b} (N={n}): meas={wm}, pred={wp}")
    return correct, total, details


def analyze_gpu(csv_path, gpu_name, has_torchaudio):
    rows = load_csv(csv_path)
    data = []
    for row in rows:
        alg = row["algorithm"]
        N = int(row["signal_length"])
        B = int(row["batch_size"])
        E = float(row["energy_per_call_j"])
        if alg not in TO_MODELS or E <= 0:
            continue
        result = predict_to(alg, N, B)
        data.append({
            "alg": alg, "N": N, "B": B, "E": E,
            "to_c": result["to_compute"], "to_m": result["to_memory"],
            "n_python": get_python_launches(alg, N, B, has_torchaudio),
            "n_fused": get_fused_steps(alg, N, B, has_torchaudio),
        })

    print(f"\n{'='*90}")
    print(f"  {gpu_name} (torchaudio={has_torchaudio})")
    print(f"{'='*90}")

    # Count data by category
    n_python = sum(1 for d in data if d["n_python"] > 0)
    n_fused = sum(1 for d in data if d["n_fused"] > 0)
    n_parallel = sum(1 for d in data if d["n_python"] == 0 and d["n_fused"] == 0)
    print(f"  Data split: {n_parallel} parallel, {n_python} python-loop, {n_fused} fused-sequential")

    # Show fused-sequential data points
    print(f"\n  Fused-sequential data points:")
    for d in data:
        if d["n_fused"] > 0:
            print(f"    {d['alg']:20s} N={d['N']:>6d} B={d['B']:>4d} "
                  f"E={d['E']:.4e} n_fused={d['n_fused']:>6d}")

    # === 3-parameter model (current, no fused term) ===
    X3 = np.array([[d["to_c"], d["to_m"], d["n_python"]] for d in data])
    y = np.array([d["E"] for d in data])
    alpha3, _ = nnls(X3, y)
    y_pred3 = X3 @ alpha3
    ss_res3 = np.sum((y - y_pred3) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_3 = 1 - ss_res3 / ss_tot
    c3, t3, d3 = run_rankings(
        [{**d, "n_fused": 0} for d in data],  # dummy n_fused for 3-param
        list(alpha3) + [0])  # alpha_f = 0

    print(f"\n  --- 3-parameter (current): E = α_c·TO_c + α_m·TO_m + α_o·n_python ---")
    print(f"  α_c={alpha3[0]*1e15:.2f} fJ/TO, α_m={alpha3[1]*1e15:.2f} fJ/TO, "
          f"α_o={alpha3[2]*1e6:.1f} µJ/launch")
    print(f"  r² = {r2_3:.4f}")
    print(f"  Ranking: {c3}/{t3} = {c3/t3*100:.0f}%")
    for line in d3:
        print(line)

    # === 4-parameter model: + α_f·n_fused ===
    X4 = np.array([[d["to_c"], d["to_m"], d["n_python"], d["n_fused"]] for d in data])
    alpha4, _ = nnls(X4, y)
    y_pred4 = X4 @ alpha4
    ss_res4 = np.sum((y - y_pred4) ** 2)
    r2_4 = 1 - ss_res4 / ss_tot
    c4, t4, d4 = run_rankings(data, alpha4)

    print(f"\n  --- 4-parameter: E = α_c·TO_c + α_m·TO_m + α_o·n_python + α_f·n_fused ---")
    print(f"  α_c={alpha4[0]*1e15:.2f} fJ/TO, α_m={alpha4[1]*1e15:.2f} fJ/TO, "
          f"α_o={alpha4[2]*1e6:.1f} µJ/launch, α_f={alpha4[3]*1e6:.2f} µJ/fused_step")
    print(f"  r² = {r2_4:.4f}")
    print(f"  Ranking: {c4}/{t4} = {c4/t4*100:.0f}%")
    for line in d4:
        print(line)

    # Show LSTM predictions
    print(f"\n  LSTM denoiser predictions (4-param):")
    print(f"  {'N':>6s} {'B':>4s} {'E_meas':>10s} {'E_pred_3p':>10s} {'E_pred_4p':>10s} "
          f"{'Err_3p':>8s} {'Err_4p':>8s} {'n_fused':>8s}")
    for d in data:
        if d["alg"] == "lstm_denoiser":
            ep3 = alpha3[0]*d["to_c"] + alpha3[1]*d["to_m"] + alpha3[2]*d["n_python"]
            ep4 = alpha4[0]*d["to_c"] + alpha4[1]*d["to_m"] + alpha4[2]*d["n_python"] + alpha4[3]*d["n_fused"]
            err3 = abs(d["E"] - ep3) / d["E"] * 100
            err4 = abs(d["E"] - ep4) / d["E"] * 100
            print(f"  {d['N']:>6d} {d['B']:>4d} {d['E']:>10.4e} {ep3:>10.4e} {ep4:>10.4e} "
                  f"{err3:>7.1f}% {err4:>7.1f}% {d['n_fused']:>8d}")

    # Show IIR predictions (if fused)
    print(f"\n  IIR predictions (4-param):")
    for d in data:
        if d["alg"] == "iir_butter4":
            ep3 = alpha3[0]*d["to_c"] + alpha3[1]*d["to_m"] + alpha3[2]*d["n_python"]
            ep4 = alpha4[0]*d["to_c"] + alpha4[1]*d["to_m"] + alpha4[2]*d["n_python"] + alpha4[3]*d["n_fused"]
            err3 = abs(d["E"] - ep3) / d["E"] * 100
            err4 = abs(d["E"] - ep4) / d["E"] * 100
            print(f"  {d['N']:>6d} {d['B']:>4d} {d['E']:>10.4e} {ep3:>10.4e} {ep4:>10.4e} "
                  f"{err3:>7.1f}% {err4:>7.1f}% n_fused={d['n_fused']:>6d}")

    # Comparison
    print(f"\n  === COMPARISON ===")
    print(f"  {'Metric':30s} {'3-param':>12s} {'4-param':>12s}")
    print(f"  {'-'*58}")
    print(f"  {'r²':30s} {r2_3:>12.4f} {r2_4:>12.4f}")
    print(f"  {'Ranking':30s} {c3}/{t3} = {c3/t3*100:.0f}%{'':>4s}{c4}/{t4} = {c4/t4*100:.0f}%")
    print(f"  {'α_o / α_f ratio':30s} {'':>12s} {alpha4[2]/alpha4[3] if alpha4[3] > 0 else 'inf':>12.0f}×")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    local_csv = base / "data" / "results" / "all_results.csv"
    if local_csv.exists():
        analyze_gpu(str(local_csv), "RTX 4090 Laptop", has_torchaudio=True)

    server_csv = base / "data" / "server_results" / "results" / "all_results.csv"
    if server_csv.exists():
        analyze_gpu(str(server_csv), "A100 SXM4", has_torchaudio=False)
