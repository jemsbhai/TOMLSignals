#!/bin/bash
# TOMLSignals Lambda Labs Setup & Run Script
# Usage: bash run_lambda.sh [--duration 10]
#
# Prerequisites: Lambda Labs instance with NVIDIA GPU + CUDA
# Tested on: A100 40GB SXM4

set -e

echo "============================================"
echo "  TOMLSignals - Lambda Labs Benchmark Setup"
echo "============================================"

# Parse args
DURATION=${1:-10}
if [ "$1" = "--duration" ]; then
    DURATION=$2
fi

# Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -q PyWavelets torchaudio scikit-learn pynvml 2>/dev/null || \
pip install -q PyWavelets torchaudio scikit-learn nvidia-ml-py 2>/dev/null

# Verify GPU
echo ""
echo "[2/4] Verifying GPU..."
python3 << 'PYEOF'
import torch
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
name = pynvml.nvmlDeviceGetName(h)
if isinstance(name, bytes):
    name = name.decode('utf-8')
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  GPU: {name}')
print(f'  Memory: {mem:.1f} GB')
print(f'  CUDA: {torch.version.cuda}')
print(f'  PyTorch: {torch.__version__}')
pynvml.nvmlShutdown()
PYEOF

# Quick smoke test
echo ""
echo "[3/4] Smoke test (all algorithms)..."
python3 << 'PYEOF'
from algorithms.transforms import TRANSFORMS
from algorithms.filters import FILTERS
from algorithms.adaptive import ADAPTIVE
from algorithms.estimation import ESTIMATION
from algorithms.spectral import SPECTRAL
from algorithms.decomposition import DECOMPOSITION
from algorithms.compression import COMPRESSION
from algorithms.ml_enhanced import ML_ENHANCED
import torch
dev = torch.device('cuda')
ok, fail = 0, 0
for cat, algs in [('transform', TRANSFORMS), ('filter', FILTERS), ('adaptive', ADAPTIVE),
                   ('estimation', ESTIMATION), ('spectral', SPECTRAL),
                   ('decomposition', DECOMPOSITION), ('compression', COMPRESSION),
                   ('ml_enhanced', ML_ENHANCED)]:
    for name, (setup, run, kw) in algs.items():
        try:
            state = setup(1024, 1, 'fp32', dev, **kw)
            run(state)
            ok += 1
        except Exception as e:
            print(f'  FAIL: {name} - {e}')
            fail += 1
print(f'  {ok} passed, {fail} failed')
if fail > 0:
    print('  WARNING: Some algorithms failed. Check output above.')
PYEOF

# Run full suite
echo ""
echo "[4/4] Running full benchmark suite (duration=${DURATION}s per benchmark)..."
echo "  Estimated time: 60-90 minutes"
echo "  Results will be saved to data/results/"
echo ""

python3 run_full_suite.py --duration ${DURATION}

echo ""
echo "============================================"
echo "  DONE - Results in data/results/"
echo "============================================"
