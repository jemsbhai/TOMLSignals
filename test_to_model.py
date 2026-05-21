"""Quick validation of TO model."""
from shared.to_model import predict_to, TO_MODELS

print(f"{'Algorithm':25s} {'N':>6s} {'B':>5s} {'TO_compute':>14s} {'TO_memory':>14s} "
      f"{'TO_total':>14s} {'MCER':>8s}")
print("-" * 95)

test_cases = [
    ("fft", 4096, 1),
    ("direct_dft", 1024, 1),
    ("fir_direct", 4096, 1),
    ("fir_fft", 4096, 1),
    ("iir_butter4", 4096, 1),
    ("wiener", 4096, 1),
    ("lms", 4096, 1),
    ("rls", 4096, 1),
    ("kalman", 4096, 1),
    ("ukf", 4096, 1),
    ("welch", 4096, 1),
    ("music", 1024, 1),
    ("svd", 1024, 1),
    ("pca", 1024, 1),
    ("fastica", 1024, 1),
    ("nmf", 1024, 1),
    ("cnn_denoiser", 4096, 1),
    ("lstm_denoiser", 4096, 1),
    ("transformer_denoiser", 4096, 1),
]

for alg, N, B in test_cases:
    r = predict_to(alg, N, B)
    mcer = r["to_memory"] / r["to_compute"] if r["to_compute"] > 0 else float("inf")
    print(f"  {alg:23s} {N:>6d} {B:>5d} {r['to_compute']:>14.2e} {r['to_memory']:>14.2e} "
          f"{r['to_total']:>14.2e} {mcer:>8.2f}")

print(f"\nTotal algorithms in model: {len(TO_MODELS)}")
