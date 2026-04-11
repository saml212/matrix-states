# H100 Training Environment — Exact Specs

## Base Stack
- Python 3.12 + PyTorch 2.9.1+cu128 (CUDA 12.8)
- Flash Attention 3 (Hopper-specific TMA/warp-specialized kernels)
- torch.compile with Triton backend + inductor

## Install (30 seconds)
```bash
# FA3 Hopper kernel — prebuilt wheel, no compilation needed
pip install --break-system-packages \
  flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

# Verify
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

## FA3 Calling Convention (DIFFERENT from FA2)
```python
# FA3: inputs are (batch, seqlen, heads, head_dim) — NOT transposed
y = flash_attn_func(q, k, v, causal=True)

# FA2/SDPA: inputs are (batch, heads, seqlen, head_dim) — transposed
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## torch.compile Settings
```python
model = torch.compile(model, dynamic=False, fullgraph=True)
# dynamic=False — static shapes, maximum optimization
# fullgraph=True — no graph breaks, one kernel graph
# Requires: fixed batch size, fixed seq len, no data-dependent control flow
```

## torch.compile Cache
- Persists at /tmp/torchinductor_root/
- Never clear between runs unless you change the script
- Cache does NOT survive pod restarts
- Always do 1 throwaway warmup run first
- Cache is script-specific — different .py files trigger recompilation

## Multi-GPU (DDP / torchrun)
```bash
torchrun --standalone --nproc_per_node=8 train.py
```

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
```

## Hardware Checks
```python
import torch
print(f"Python: {__import__('sys').version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPUs: {torch.cuda.device_count()}")

try:
    from flash_attn_interface import flash_attn_func
    print("FA3 Hopper: OK")
except ImportError:
    print("FA3 Hopper: MISSING")

# Compile test
@torch.compile(dynamic=False)
def test_fn(x): return x * 2 + 1
test_fn(torch.randn(8, device='cuda'))
print("torch.compile: OK")
```

## Bandwidth Test
```python
import torch, time
x = torch.randn(256*1024*1024, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20): y = x * 2.0
torch.cuda.synchronize()
bw = 20 * x.numel() * 2 * 2 / (time.perf_counter() - t0) / 1e9
print(f'Memory bandwidth: {bw:.0f} GB/s')  # Good: >1500, Great: >2000
```

## Key Performance Numbers
| Backend | Import | H100 Speed | Notes |
|---------|--------|-----------|-------|
| FA3 Hopper | flash_attn_interface.flash_attn_func | 0.47ms | H100-only, TMA + warp spec |
| FA2 | flash_attn.flash_attn_interface.flash_attn_func | 0.73ms | Any GPU |
| PyTorch SDPA | F.scaled_dot_product_attention | 0.77ms | Built-in, flash auto-selected |
| cuDNN SDPA | SDPBackend.CUDNN_ATTENTION | 13.5ms | AVOID. 18x slower |
