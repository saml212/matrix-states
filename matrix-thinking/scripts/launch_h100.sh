#!/bin/bash
# Launch all H100 experiments in parallel across 8 GPUs.
# Each GPU runs one experiment independently.
#
# Usage: bash launch_h100.sh
#
# Prerequisites:
# - Reasoning data at /data/reasoning/ (copy from SSD)
# - WikiText-103 tokenized at /data/wikitext103_tokenized/ (for baselines)
# - Python environment with PyTorch, transformers

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="/results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo " Matrix Thinker H100 Experiments"
echo " $(date)"
echo " 8 GPUs, parallel execution"
echo "=========================================="

# Verify environment
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# GPU 0: Main thinker (mat_dim=16, 2 layers, 200 thoughts)
CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT_DIR/exp_thinker_main.py" \
    --config thinker_d16_deep &

# GPU 1: Wider matrices (mat_dim=24)
CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT_DIR/exp_thinker_main.py" \
    --config thinker_d24_wide &

# GPU 2: Widest matrices (mat_dim=32)
CUDA_VISIBLE_DEVICES=2 python3 "$SCRIPT_DIR/exp_thinker_main.py" \
    --config thinker_d32 &

# GPU 3: Deeper thinking per step (3 layers)
CUDA_VISIBLE_DEVICES=3 python3 "$SCRIPT_DIR/exp_thinker_main.py" \
    --config thinker_d16_3layer_4thought &

# GPU 4: Minimal layers, many thoughts (1 layer, 200 thoughts)
CUDA_VISIBLE_DEVICES=4 python3 "$SCRIPT_DIR/exp_thinker_main.py" \
    --config thinker_d16_1layer_20thought &

# GPU 5: Frobenius attention ablation (no 3D)
CUDA_VISIBLE_DEVICES=5 python3 "$SCRIPT_DIR/exp_thinker_main.py" \
    --config thinker_d16_deep &
    # TODO: add --frobenius flag to switch attention type

# GPU 6: Vector transformer baseline
CUDA_VISIBLE_DEVICES=6 python3 "$SCRIPT_DIR/exp1b_vector_baseline.py" &

# GPU 7: Matrix thinker with flattening
CUDA_VISIBLE_DEVICES=7 python3 "$SCRIPT_DIR/exp1c_matrix_v2_flatten.py" &

echo ""
echo "All 8 experiments launched. Waiting for completion..."
wait

echo ""
echo "=========================================="
echo " ALL EXPERIMENTS COMPLETE"
echo " $(date)"
echo "=========================================="

# Print summary
python3 -c "
import json, glob
results = []
for f in sorted(glob.glob('$RESULTS_DIR/*/result.json')) + sorted(glob.glob('$RESULTS_DIR/*/solidification_results.json')):
    try:
        r = json.load(open(f))
        results.append(r)
    except: pass

print(f'{\"Experiment\":<40} {\"Params\":>12} {\"Best PPL\":>10}')
print('-' * 62)
for r in sorted(results, key=lambda x: x.get('best_val_ppl', 999999)):
    name = r.get('experiment', 'unknown')
    params = r.get('params', 0)
    ppl = r.get('best_val_ppl', float('inf'))
    print(f'{name:<40} {params:>12,} {ppl:>10.1f}')
"
