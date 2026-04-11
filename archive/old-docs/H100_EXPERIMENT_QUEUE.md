# H100 Experiment Queue

## Prerequisites (run on Mac Mini first)
- [running] V2 matrix vs vector comparison (byte-level, 2M params, confirms basic operation works)
- [ ] Build V3 architecture (matrix-native ops, no flattening, SwiGLU)
- [ ] Smoke test V3 locally (forward, backward, shape check, gradient check)
- [ ] Verify V3 runs on CUDA (not just MPS)
- [ ] Set up WikiText-103 tokenized benchmark (BPE tokenizer, perplexity metric)
- [ ] Run V3 for 100 steps locally to confirm training loop works

## Experiment Queue (8×H100, ~1 hour session)

All experiments: WikiText-103, BPE tokenized, matched parameter count,
same optimizer/schedule, 3 random seeds each.

### Priority 1: Does matrix thinking beat vectors?
```
Experiment 1a: Matrix V3 (mat_dim=16, 6 layers, Kronecker K=4) — ~50M params
Experiment 1b: Standard transformer (d=512, 6 layers) — ~50M params (matched)
Experiment 1c: Matrix V3 (mat_dim=16, 12 layers) — ~100M params
Experiment 1d: Standard transformer (d=768, 8 layers) — ~100M params (matched)
Metric: WikiText-103 perplexity
```

### Priority 2: Which matrix-native operation is best?
```
Experiment 2a: Kronecker K=4 projections (current plan)
Experiment 2b: Householder K=4 reflections (fewest params)
Experiment 2c: RowThenColumn with nonlinearity (cheapest expressive option)
Experiment 2d: Flatten→Linear (current V2, as baseline to measure what we lose)
All at ~50M params, same architecture otherwise
```

### Priority 3: Does iterative refinement help?
```
Experiment 3a: 6 layers, no iteration (standard forward pass)
Experiment 3b: 2 shared-weight layers × 6 iterations (PonderNet halting)
Experiment 3c: 2 shared-weight layers × 12 iterations (more thinking)
All at ~50M params
```

### Priority 4: Activation function sweep
```
Experiment 4a: GELU in bottleneck
Experiment 4b: SwiGLU in bottleneck
Experiment 4c: No activation (pure multiplicative nonlinearity)
Experiment 4d: Spectral activation (apply nonlinearity to singular values)
All at ~50M params
```

## Expected Timeline
- 8 experiments × 3 seeds = 24 runs
- ~50M params on H100: ~15-30 min per run for 50K steps
- Total: ~6-12 hours of H100 compute
- With 8 GPUs in parallel: ~1-2 hours wall clock

## What We Need From Results
1. Perplexity numbers comparable to published baselines (GPT-2 small = 35.1 ppl on WikiText-103)
2. Parameter efficiency: if matrix model at 50M matches vector model at 200M, that's the story
3. Training curves: do matrix models learn faster per step, or just converge to same place?
4. Rank dynamics at scale: does the rank-difficulty correlation get stronger with more params?
