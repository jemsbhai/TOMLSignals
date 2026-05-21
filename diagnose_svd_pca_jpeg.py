"""
Diagnose SVD, PCA, JPEG remaining discrepancies.
Isolate component costs to identify where NCU gaps originate.
"""
import torch
import time
import math
import numpy as np

device = "cuda"
n_warmup = 50
n_measure = 200

def bench(fn, label=""):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_measure):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_measure * 1e6


print("=" * 70)
print("  JPEG: Isolating the 64.8x per-block overhead")
print("=" * 70)

# Reproduce exact JPEG setup
bs = 8
side = 64
n_blocks = (side // bs) ** 2  # 64
img = torch.randn(1, 1, side, side, device=device, dtype=torch.float32) * 128 + 128
n = torch.arange(bs, device=device, dtype=torch.float32)
k = n.unsqueeze(1)
dct_basis = torch.cos(np.pi * (2*n+1) * k / (2*bs))
dct_basis[0] *= 1/np.sqrt(bs)
dct_basis[1:] *= np.sqrt(2/bs)
Q = torch.ones(bs, bs, device=device) * 16  # simplified Q

# Extract blocks
blocks = img.unfold(2,bs,bs).unfold(3,bs,bs).contiguous().view(-1,bs,bs)
print(f"\n  blocks shape: {blocks.shape}")  # should be (64, 8, 8)

# Test: batched matmul with different sizes
print(f"\n  --- Batched matmul scaling test ---")
print(f"  {'n_blocks':>10s} {'Shape':>20s} {'Time(us)':>10s} {'FMA/call':>12s}")
for nb in [1, 4, 16, 64, 256]:
    test_blocks = torch.randn(nb, 8, 8, device=device)
    test_basis = torch.randn(8, 8, device=device)
    t = bench(lambda: torch.matmul(test_basis, torch.matmul(test_blocks, test_basis.T)))
    expected_fma = nb * 2 * 8**3
    print(f"  {nb:>10d} ({nb},8,8)@(8,8)      {t:>10.1f} {expected_fma:>12,}")

# Test: does unfold+contiguous+view affect the matmul cost?
print(f"\n  --- Unfold vs pre-allocated blocks ---")
t_unfold = bench(lambda: torch.matmul(dct_basis, torch.matmul(
    img.unfold(2,bs,bs).unfold(3,bs,bs).contiguous().view(-1,bs,bs), dct_basis.T)))
t_prealloc = bench(lambda: torch.matmul(dct_basis, torch.matmul(blocks, dct_basis.T)))
print(f"  With unfold+view: {t_unfold:.1f} us")
print(f"  Pre-allocated:    {t_prealloc:.1f} us")

# Test: single large matmul vs batched small matmul (are they equivalent?)
print(f"\n  --- Equivalent large matmul ---")
blocks_flat = blocks.view(64*8, 8)  # (512, 8)
t_flat = bench(lambda: torch.matmul(blocks_flat, dct_basis.T))
t_batched = bench(lambda: torch.matmul(blocks, dct_basis.T))
print(f"  Batched (64,8,8)@(8,8): {t_batched:.1f} us")
print(f"  Flat (512,8)@(8,8):     {t_flat:.1f} us")


print(f"\n{'='*70}")
print(f"  SVD: Component breakdown for 1024x64 matrix")
print(f"{'='*70}")

N_svd, D_svd = 1024, 64
X_svd = torch.randn(1, N_svd, D_svd, device=device, dtype=torch.float32)

t_full_svd = bench(lambda: torch.linalg.svd(X_svd, full_matrices=False))
t_svd_2d = bench(lambda: torch.linalg.svd(X_svd[0], full_matrices=False))

print(f"\n  torch.linalg.svd({N_svd}x{D_svd}, full_matrices=False):")
print(f"    Batched (1,{N_svd},{D_svd}): {t_full_svd:.1f} us")
print(f"    2D ({N_svd},{D_svd}):        {t_svd_2d:.1f} us")

# Compare with equivalent-cost matmul
A_test = torch.randn(N_svd, D_svd, device=device)
B_test = torch.randn(D_svd, D_svd, device=device)
t_matmul = bench(lambda: torch.matmul(A_test, B_test))
print(f"    Matmul {N_svd}x{D_svd} @ {D_svd}x{D_svd}: {t_matmul:.1f} us (for reference)")

# Try different formulas
ratios = {
    "6ND^2": 6 * N_svd * D_svd**2,
    "14ND^2": 14 * N_svd * D_svd**2,
    "20ND^2": 20 * N_svd * D_svd**2,
    "4ND^2+22D^3/3": int(4*N_svd*D_svd**2 + 22*D_svd**3/3),
    "2(4ND^2+8D^3)": int(2*(4*N_svd*D_svd**2 + 8*D_svd**3)),
}
ncu_svd = 85_804_453
print(f"\n  NCU FP32 total: {ncu_svd:,}")
print(f"  {'Formula':>25s} {'Predicted':>14s} {'Ratio':>8s}")
for name, pred in ratios.items():
    print(f"  {name:>25s} {pred:>14,} {ncu_svd/pred:>8.2f}")


print(f"\n{'='*70}")
print(f"  PCA: Component breakdown for pca_lowrank(1024x64, q=8, niter=2)")
print(f"{'='*70}")

N_pca, D_pca, q_pca = 1024, 64, 8
X_pca = torch.randn(N_pca, D_pca, device=device, dtype=torch.float32)

# Full pca_lowrank
t_pca = bench(lambda: torch.pca_lowrank(X_pca, q=q_pca, center=True, niter=2))
print(f"\n  Full pca_lowrank: {t_pca:.1f} us")

# Component: centering
X_centered = X_pca - X_pca.mean(0)
t_center = bench(lambda: X_pca - X_pca.mean(0))
print(f"  Centering:        {t_center:.1f} us")

# Component: A @ R (projection)
R = torch.randn(D_pca, q_pca, device=device)
t_proj = bench(lambda: torch.matmul(X_centered, R))
print(f"  A @ R ({N_pca}x{D_pca} @ {D_pca}x{q_pca}): {t_proj:.1f} us")

# Component: QR of N×q
Q_test = torch.randn(N_pca, q_pca, device=device)
t_qr_big = bench(lambda: torch.linalg.qr(Q_test))
print(f"  QR({N_pca}x{q_pca}):      {t_qr_big:.1f} us")

# Component: QR of D×q
Z_test = torch.randn(D_pca, q_pca, device=device)
t_qr_small = bench(lambda: torch.linalg.qr(Z_test))
print(f"  QR({D_pca}x{q_pca}):       {t_qr_small:.1f} us")

# Component: SVD of q×D
B_test = torch.randn(q_pca, D_pca, device=device)
t_svd_small = bench(lambda: torch.linalg.svd(B_test, full_matrices=False))
print(f"  SVD({q_pca}x{D_pca}):      {t_svd_small:.1f} us")

# Estimated total from components
# 1 center + 1 initial A@R + 1 QR(N,q) + 2*(A.T@Q + QR(D,q) + A@Z + QR(N,q)) + B=Q.T@A + SVD(B) + U=Q@Us
n_iter = 2
est_components = (t_center + t_proj + t_qr_big +
                  n_iter * (t_proj + t_qr_small + t_proj + t_qr_big) +
                  t_proj + t_svd_small + t_proj * (q_pca/D_pca))  # rough
print(f"\n  Sum of components: ~{est_components:.1f} us")
print(f"  Full pca_lowrank:   {t_pca:.1f} us")
print(f"  Ratio:              {t_pca/est_components:.2f}x")

# What fraction is QR?
total_qr_time = t_qr_big * (1 + 2*n_iter) + t_qr_small * (2*n_iter)
total_matmul_time = t_proj * (1 + 4*n_iter + 1) + t_proj * (q_pca/D_pca)
print(f"\n  QR time ({1+2*n_iter} big + {2*n_iter} small): {total_qr_time:.1f} us ({total_qr_time/t_pca*100:.0f}% of total)")
print(f"  Matmul time ({1+4*n_iter+1} calls): {total_matmul_time:.1f} us ({total_matmul_time/t_pca*100:.0f}% of total)")
print(f"  SVD({q_pca}x{D_pca}) time: {t_svd_small:.1f} us ({t_svd_small/t_pca*100:.0f}% of total)")
