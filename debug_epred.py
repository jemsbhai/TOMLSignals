"""
Debug: trace E_pred computation for JPEG and MDCT.
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from shared.to_model import predict_to, TO_MODELS, get_seq_steps

# Manually compute what E_pred should be for JPEG
test_cases = [
    ("jpeg_q50", 4096, 2048),
    ("jpeg_q50", 16384, 512),
    ("mdct_audio", 4096, 1),
    ("mdct_audio", 16384, 1),
    ("cnn_denoiser", 4096, 2048),
    ("fft", 4096, 2048),
    ("transformer_denoiser", 16384, 1),
]

print("=== TO model predictions ===")
print(f"  {'Algorithm':25s} {'N':>6s} {'B':>5s} {'TO_compute':>14s} {'TO_memory':>14s} {'TO_total':>14s} {'seq':>6s}")
for alg, N, B in test_cases:
    r = predict_to(alg, N, B)
    seq = get_seq_steps(alg, N, B, has_torchaudio=True)
    print(f"  {alg:25s} {N:>6d} {B:>5d} {r['to_compute']:>14.6e} {r['to_memory']:>14.6e} "
          f"{r['to_total']:>14.6e} {seq:>6d}")

# Now simulate the OLS fit
print("\n=== Simulating OLS fit (Model A: 2-param all data) ===")
import csv
from pathlib import Path

points = []
with open("data/results/all_results.csv") as f:
    for row in csv.DictReader(f):
        alg = row["algorithm"]
        N = int(row["signal_length"])
        B = int(row["batch_size"])
        E = float(row["energy_per_call_j"])
        if alg in TO_MODELS and E > 0:
            r = predict_to(alg, N, B)
            if r["to_compute"] > 0:
                points.append({
                    "alg": alg, "N": N, "B": B, "E": E,
                    "to_c": r["to_compute"], "to_m": r["to_memory"],
                })

X = np.array([[p["to_c"], p["to_m"]] for p in points])
y = np.array([p["E"] for p in points])

print(f"  n_points: {len(points)}")
print(f"  X range: to_c=[{X[:,0].min():.2e}, {X[:,0].max():.2e}]")
print(f"  X range: to_m=[{X[:,1].min():.2e}, {X[:,1].max():.2e}]")
print(f"  y range: [{y.min():.4e}, {y.max():.4e}]")

# Check condition number
XtX = X.T @ X
cond = np.linalg.cond(XtX)
print(f"  Condition number of X^T X: {cond:.2e}")

# OLS
alpha = np.linalg.solve(XtX, X.T @ y)
print(f"  alpha_c = {alpha[0]:.6e}")
print(f"  alpha_m = {alpha[1]:.6e}")

# Predictions
y_pred = X @ alpha
print(f"\n  y_pred range: [{y_pred.min():.4e}, {y_pred.max():.4e}]")

# Check specific algorithms
for alg_check in ["jpeg_q50", "mdct_audio", "cnn_denoiser", "fft"]:
    idxs = [i for i, p in enumerate(points) if p["alg"] == alg_check]
    if idxs:
        print(f"\n  {alg_check}:")
        for i in idxs:
            p = points[i]
            manual_pred = alpha[0] * p["to_c"] + alpha[1] * p["to_m"]
            print(f"    N={p['N']:>6d} B={p['B']:>5d}: "
                  f"E_meas={p['E']:.4e}  "
                  f"y_pred[{i}]={y_pred[i]:.4e}  "
                  f"manual={manual_pred:.4e}  "
                  f"match={np.isclose(y_pred[i], manual_pred)}")

# Now simulate 3-param fit (Model B)
print("\n=== Simulating 3-param NNLS fit (Model B) ===")
from scipy.optimize import nnls

X3 = np.array([[p["to_c"], p["to_m"], get_seq_steps(p["alg"], p["N"], p["B"], has_torchaudio=True)] 
               for p in points])
alpha3, rnorm = nnls(X3, y)
print(f"  alpha_c = {alpha3[0]:.6e}")
print(f"  alpha_m = {alpha3[1]:.6e}")
print(f"  alpha_o = {alpha3[2]:.6e}")

y_pred3 = X3 @ alpha3
print(f"  y_pred3 range: [{y_pred3.min():.4e}, {y_pred3.max():.4e}]")

for alg_check in ["jpeg_q50", "mdct_audio", "cnn_denoiser"]:
    idxs = [i for i, p in enumerate(points) if p["alg"] == alg_check]
    if idxs:
        print(f"\n  {alg_check}:")
        for i in idxs:
            p = points[i]
            seq = get_seq_steps(p["alg"], p["N"], p["B"], has_torchaudio=True)
            manual = alpha3[0]*p["to_c"] + alpha3[1]*p["to_m"] + alpha3[2]*seq
            print(f"    N={p['N']:>6d} B={p['B']:>5d} seq={seq}: "
                  f"y_pred3={y_pred3[i]:.4e}  manual={manual:.4e}")

# Now simulate parallel-only fit (Model C) — this is what gets printed for parallel algs
print("\n=== Simulating parallel-only 2-param fit (Model C) ===")
par_idxs = [i for i, p in enumerate(points) 
            if get_seq_steps(p["alg"], p["N"], p["B"], has_torchaudio=True) == 0]
X_par = X[par_idxs]
y_par = y[par_idxs]

print(f"  n_parallel: {len(par_idxs)}")
cond_par = np.linalg.cond(X_par.T @ X_par)
print(f"  Condition number: {cond_par:.2e}")

alpha_par = np.linalg.solve(X_par.T @ X_par, X_par.T @ y_par)
print(f"  alpha_c = {alpha_par[0]:.6e}")
print(f"  alpha_m = {alpha_par[1]:.6e}")

y_pred_par = X_par @ alpha_par
print(f"  y_pred_par range: [{y_pred_par.min():.4e}, {y_pred_par.max():.4e}]")

for alg_check in ["jpeg_q50", "mdct_audio", "cnn_denoiser"]:
    local_idxs = [j for j, i in enumerate(par_idxs) if points[i]["alg"] == alg_check]
    if local_idxs:
        print(f"\n  {alg_check}:")
        for j in local_idxs:
            i = par_idxs[j]
            p = points[i]
            manual = alpha_par[0]*p["to_c"] + alpha_par[1]*p["to_m"]
            print(f"    N={p['N']:>6d} B={p['B']:>5d}: "
                  f"y_pred_par={y_pred_par[j]:.4e}  manual={manual:.4e}")
