"""Verify Tier 2/3 corrections against NCU."""
import json, sys, math
sys.path.insert(0, ".")
from shared.to_model import TO

with open("data/ncu_profiles/ncu_summary.json") as f:
    ncu_data = {r["algorithm"]: r for r in json.load(f)}

print("=== CNN Denoiser ===")
ncu = ncu_data["cnn_denoiser"]
N = ncu["N"]
old = (1*16*7 + 16*32*5 + 32*1*3) * N
new = (1*32*7 + 32*32*7 + 32*1*7) * N
print(f"  Old (wrong arch): {old:>12,}  ratio: {ncu['fp32_total']/old:.3f}")
print(f"  New (correct):    {new:>12,}  ratio: {ncu['fp32_total']/new:.3f}")
print(f"  NCU FP32 total:   {ncu['fp32_total']:>12,.0f}")

print("\n=== MDCT ===")
ncu = ncu_data["mdct_audio"]
N = ncu["N"]
fs = 512
nf = max(1, min((N - 2*fs) // fs + 1, 50))
new_ops = nf * (2*fs + fs*2*fs + 3*fs + fs + fs)
old_ops = N + int(4*256*math.log2(256)-6*256+8) + 256*6 + 256 + 256 + int(4*256*math.log2(256)-6*256+8) + 256*6 + N
print(f"  Old (FFT-based):  {old_ops:>12,}  ratio: {ncu['fp32_total']/old_ops:.3f}")
print(f"  New (matmul):     {new_ops:>12,}  ratio: {ncu['fp32_total']/new_ops:.3f}")
print(f"  NCU FP32 total:   {ncu['fp32_total']:>12,.0f}")
print(f"  n_frames={nf}, frame_size={fs}")

print("\n=== Median ===")
ncu = ncu_data["median"]
N = ncu["N"]
W = 7
nw = N - W + 1
old_cmp = nw * W * math.log2(W)
old_unfold = nw * W
print(f"  NCU FP32:         {ncu['fp32_total']:>12,.0f} (zero - confirmed)")
print(f"  NCU int_total:    {ncu['int_total']:>12,.0f}")
print(f"  Old TO had {old_unfold*TO['mac']:,} TOs from spurious unfold MACs")
print(f"  New TO: comparison-only = {old_cmp*TO['cmp']:,.0f} TOs")

print("\n=== LSTM (flagged - not fixed yet) ===")
ncu = ncu_data["lstm_denoiser"]
N = ncu["N"]
correct_fma = N * 4 * (1 + 128) * 128
print(f"  Impl: LSTM(1,128,layers=1) + Linear(128,1)")
print(f"  TO model: H=64, layers=2 (WRONG)")
print(f"  NCU FMA:          {ncu['fp32_fma']:>12,.0f}")
print(f"  Correct pred:     {correct_fma:>12,}")
print(f"  Ratio:            {ncu['fp32_fma']/correct_fma:.3f}")
