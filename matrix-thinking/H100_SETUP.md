# H100 Environment

## Access (current 8×H100 pod)
```
ssh root@154.57.34.103 -p 44178 -i ~/.ssh/id_ed25519
```
Volume: `/toy_story_slam/` (200GB persistent)

## Stack
- Python 3.12 + PyTorch 2.9.1+cu128 (CUDA 12.8)
- 1x NVIDIA H100 80GB HBM3
- Flash Attention available via SDPA backend

## File Layout
```
/root/
  run_exp2_iterative.py    ← current experiment script
  exp2_stdout.log          ← stdout/stderr
  src/
    matrix_thinker.py      ← v1 model (3D attention, PonderNet)
    matrix_model.py        ← v1 bilinear
    matrix_model_v2.py     ← v2 multiplicative
    data.py                ← data utils
  data/
    reasoning/             ← OpenR1-Math tokenized
      train.pt, val.pt, meta.json
  results/
    best.pt                ← Run 8 checkpoint
    exp2_iterative/        ← Exp 2 results (script, logs, checkpoint, summary)
```

## Quick Commands
```bash
# Check if training is running
ps aux | grep python

# Check progress
tail -20 /root/exp2_stdout.log

# GPU status
nvidia-smi

# Kill training
kill $(pgrep -f run_exp)
```

## torch.compile (for future experiments)
```python
model = torch.compile(model, dynamic=False, fullgraph=True)
# Requires: fixed shapes, no data-dependent control flow
# Cache at /tmp/torchinductor_root/ — doesn't survive pod restarts
# Always do 1 warmup run first
```

## Multi-GPU (for future 8×H100)
```bash
torchrun --standalone --nproc_per_node=8 train.py
```

## Precision
- bfloat16 autocast for forward/backward (H100 tensor cores: 990 TFLOPS)
- float32 for optimizer states (AdamW)
- SDPA auto-selects flash attention backend with bfloat16

## Pod Setup Checklist (new pod)
```bash
# 1. Install deps
pip install --break-system-packages datasets transformers tokenizers huggingface_hub

# 2. Redirect HF cache to volume (container disk is only 20GB)
rm -rf /root/.cache/huggingface
ln -s /toy_story_slam/.cache/huggingface /root/.cache/huggingface

# 3. Set HF token
export HF_TOKEN=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

# 4. Upload scripts from matrix-thinking/h100_scripts/
# 5. Run data prep (WikiText: 46s, OpenR1-Math full: ~8 min)
# 6. Launch experiment
```

## Lessons Learned (Hard-Won)

### VRAM Budget (mat_dim=32, 8 layers, T=8, H100 80GB)
| Batch/GPU | VRAM | Status |
|-----------|------|--------|
| 32 | 14 GB | Works but wastes 66GB |
| 96 | 47 GB | **Safe max** — room for eval |
| 112 | 50 GB | Training fits, **eval OOMs** |
| 128+ | OOM | Don't try |

The bottleneck is the logits tensor: (B, L, 50257) in bf16.
batch=96 × 512 × 50257 × 2 = 5GB just for logits. With backward ~10GB.

### DDP Eval Timeout
Eval runs on rank 0 only. Other ranks wait at `dist.barrier()`.
NCCL default timeout is 10 minutes. If eval takes longer, ALL ranks abort.
**Fix:** `dist.init_process_group("nccl", timeout=timedelta(minutes=30))`
**AND** cap eval to 50 batches max.

### HuggingFace Cache
HF downloads go to `/root/.cache/huggingface/` by default (container disk, 20GB).
Large datasets fill it instantly. Symlink to volume on first setup.

### PyTorch 2.4 Quirks
- `nn.MultiheadAttention`: pass EITHER `attn_mask` OR `is_causal=True`, not both
- `torch.utils.checkpoint`: FutureWarning about `cpu.amp.autocast` — harmless
- `killall python3` may not match `/usr/bin/python` — use `pkill -f` instead

## Parameter Matching

Matrix models are extremely parameter-efficient. Thinking layers add almost
nothing — embeddings dominate. At mat_dim=32:
- Embeddings: ~3.2M params (63%)
- 8 thinking layers: ~0.2M params (4%)
- Output head: ~1.6M params (31%)

Comparison: LoopFormer (ICLR 2026) at n_embd=96, 2 blocks × 8 loops = 5.3M params.
Script ready at `run_loopformer_baseline.py`.
