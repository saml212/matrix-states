# Proposal: What to Build and Run on H100s

## What we've proven
- Matrix thoughts solidify (rank drops monotonically across thinking steps)
- 3D matrix product attention works (rows at pos s couple with cols at pos t)
- The architecture trains and produces valid predictions on reasoning data
- Matrix V2 beat vector transformer on byte-level prediction (1.94 vs 2.03 BPC)

## What to build before H100s (in order)

### 1. True autoregressive thought generation
**Current:** Iterates over same positions, refining in place.
**Need:** Each thought APPENDED to sequence as new position.

Implementation (start simple, no KV cache):
```python
for t in range(max_thoughts):
    x = run_transformer(full_sequence)
    new_thought = project(x[:, -1:])          # generate from last position
    sequence = torch.cat([sequence, new_thought], dim=1)  # APPEND
    if rank_converged(new_thought): break      # crystallized → output token
```

COCONUT does replacement (not what we want). We do appending.
KV cache adds speed later — input tokens computed once, each thought
appends one KV pair. Memory for 200 thoughts: ~20MB. Trivial.

### 2. Rank-based collapse + PonderNet weighted loss
When latest thought's σ₂/σ₁ < threshold → collapse to vector → predict token.
Training: PonderNet weighted loss across all steps (differentiable, no REINFORCE).
Inference: truly stop when rank converges.
Use `torch.linalg.svdvals` for rank (gradients only through singular values, stable).

### 3. Thinking layer stays per-position (2D)
Research conclusion: cross-position coupling in FFN gives diminishing returns
when attention already provides 3D coupling. The iterative shared-weight
application creates implicit cross-position coupling through attention
across iterations. Add sequential scan only as follow-up experiment.

### 4. Certainty-driven mode switching (Phase 2, NOT first H100 run)
EMA certainty tracker with detached gradients (no backprop through EMA).
Per-position masking for variable iteration counts within a batch.
Start with FixedScheduleController, swap to EMACertaintyController after
base results are in.

Research agent built full implementation:
- FixedScheduleController (constant iterations, use first)
- EMACertaintyController (EMA-based adaptive, use second)
- IteratedThinkingBlock with per-position halting masks
- Warmup period: fixed iterations until EMA has enough history

## H100 Deployment Details

### Flash Attention 3: CANNOT use it
FA3 requires scalar dot-product scores. Our matrix product scores
(einsum 'bhlik,bhmjk->bhlmij') produce a MATRIX per pair. FA3's tiling
trick doesn't work. Use chunked PyTorch attention instead.

### Memory management
The (B,H,L,L,hd,hd) intermediate is the bottleneck:
- L=512, H=4, hd=4, B=16: ~512MB forward. Feasible.
- L=2048: ~16GB. Needs chunked computation (chunk_size=64, 32x reduction).
- Gradient checkpointing on the attention: halves peak memory, +33% compute.

### Precision
- bfloat16 for matrix scores (H100 tensor cores: 990 TFLOPS)
- float32 for softmax accumulation (precision-sensitive)
- float32 for scan state if using transfer-matrix approach

### Parallelism
- FSDP (not DDP) with auto_wrap per transformer block
- static_graph=True if shapes are fixed
- gradient_as_bucket_view=True

## H100 Experiment Plan (8 GPUs, 1 hour)

Training data: OpenR1-Math-220k reasoning (46M tokens).
Max thoughts: 200 (uncapped — let the model decide).
Eval: validation perplexity + rank profiles + thinking step distributions.

```
GPU 0: mat_dim=16, 2 thinking layers, max_thoughts=200 (MAIN)
GPU 1: mat_dim=24, 2 thinking layers, max_thoughts=200 (wider)
GPU 2: mat_dim=32, 1 thinking layer, max_thoughts=200 (widest)
GPU 3: mat_dim=16, 3 thinking layers, max_thoughts=200 (deeper per step)
GPU 4: mat_dim=16, 1 thinking layer, max_thoughts=200 (minimal per step)
GPU 5: mat_dim=16, 2 thinking layers, Frobenius attention (no 3D — ablation)
GPU 6: Vector transformer baseline, matched compute, same reasoning data
GPU 7: Matrix thinker with flattening (measures no-flatten vs flatten)
```

## What we're measuring

1. **Do thoughts solidify at scale?** Rank profiles over iterations at 50M+ params
2. **How many thoughts does the model learn to use?** Halt distribution
3. **Does 3D attention help?** GPU 0 vs GPU 5
4. **Does matrix beat vectors on reasoning?** GPU 0 vs GPU 6
5. **Which matrix dimension works best?** GPU 0 vs 1 vs 2
6. **Does true autoregressive thought generation differ from refinement-in-place?**

## Build order
1. Rewrite matrix_thinker.py → true autoregressive (append, not refine)
2. Add rank-based collapse + PonderNet loss
3. Build chunked matrix attention for memory efficiency
4. Build H100 launch script (FSDP, bf16 autocast, proper paths)
5. Test everything on Mac Mini with reasoning data
6. Build baseline scripts (vector, flatten) for GPU 6-7
7. Run on H100s
