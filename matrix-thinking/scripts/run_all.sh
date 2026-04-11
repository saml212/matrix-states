#!/bin/bash
# H100 Experiment Queue — Run All
# Usage: bash run_all.sh
#
# Prerequisites:
# - WikiText-103 tokenized data at /data/wikitext103_tokenized/
# - FA3 installed: pip install flash_attn_3 --find-links ...
# - This script runs experiments SEQUENTIALLY on a single GPU.
#   For multi-GPU parallel: modify each script to use torchrun.

set -e  # Exit on first error

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="/results"

echo "=========================================="
echo " H100 Experiment Queue"
echo " $(date)"
echo "=========================================="

# Verify environment
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPUs: {torch.cuda.device_count()}')
try:
    from flash_attn_interface import flash_attn_func
    print('FA3: OK')
except: print('FA3: MISSING (using SDPA fallback)')
"

# Copy tokenized data to local storage if needed
if [ ! -f /data/wikitext103_tokenized/train.pt ]; then
    echo "ERROR: No tokenized data at /data/wikitext103_tokenized/"
    echo "Copy from SSD: cp -r /path/to/wikitext103_tokenized /data/"
    exit 1
fi

# Warmup torch.compile cache
echo ""
echo "=== Warmup (torch.compile cache) ==="
python3 "$SCRIPTS_DIR/exp1a_matrix_v3.py" --max-steps 10 2>/dev/null || true

# Run experiments
echo ""
echo "=== Experiment 1a: Matrix V3 (Kronecker K=4) ==="
python3 "$SCRIPTS_DIR/exp1a_matrix_v3.py"

echo ""
echo "=== Experiment 1b: Vector Transformer Baseline ==="
python3 "$SCRIPTS_DIR/exp1b_vector_baseline.py"

echo ""
echo "=== Experiment 1c: Matrix V2 (flatten) ==="
python3 "$SCRIPTS_DIR/exp1c_matrix_v2_flatten.py"

echo ""
echo "=== Experiment 2b: Householder Projections ==="
python3 "$SCRIPTS_DIR/exp2b_householder.py"

echo ""
echo "=== Experiment 2c: RowThenColumn Projections ==="
python3 "$SCRIPTS_DIR/exp2c_rowthencolumn.py"

echo ""
echo "=== Experiment 2d: Bilinear Projections ==="
python3 "$SCRIPTS_DIR/exp2d_bilinear.py"

echo ""
echo "=== Experiment 3b: GELU Activation ==="
python3 "$SCRIPTS_DIR/exp3b_gelu.py"

echo ""
echo "=== Experiment 3c: No Activation ==="
python3 "$SCRIPTS_DIR/exp3c_no_activation.py"

echo ""
echo "=== Experiment 4b: Iterative Shared ×6 ==="
python3 "$SCRIPTS_DIR/exp4b_iterative_shared.py"

echo ""
echo "=== Experiment 4c: Iterative Shared ×12 ==="
python3 "$SCRIPTS_DIR/exp4c_iterative_12.py"

echo ""
echo "=========================================="
echo " ALL EXPERIMENTS COMPLETE"
echo " Results saved to: $RESULTS_DIR"
echo " $(date)"
echo "=========================================="

# Print summary
python3 -c "
import json, os, glob
results = []
for f in sorted(glob.glob('$RESULTS_DIR/*/result.json')):
    r = json.load(open(f))
    results.append(r)

print()
print(f'{\"Experiment\":<35} {\"Params\":>12} {\"Best PPL\":>10} {\"Time\":>8}')
print('-' * 65)
for r in sorted(results, key=lambda x: x['best_val_ppl']):
    t = r['training_time_s'] / 60
    print(f'{r[\"experiment\"]:<35} {r[\"params\"]:>12,} {r[\"best_val_ppl\"]:>10.1f} {t:>7.0f}m')
"
