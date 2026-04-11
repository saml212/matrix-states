# Quantization Analysis (April 2026)

## TL;DR: Not worth pursuing at current scale. Revisit at 50M+ params.

## TurboQuant (Google DeepMind, ICLR 2026)
- arxiv: 2504.19874
- KV cache quantization (not weight quantization)
- 3-bit keys, 2-bit values, 6× cache compression, 8× attention speedup
- Solves long-context serving for large models
- NOT relevant to our 2.5M param model with 512 context

## Why Quantization Hurts Us
1. Our rank dynamics are the novel contribution. Quantization corrupts small singular
   values that distinguish rank-3 from rank-5. INT4 would destroy rank structure entirely.
2. 16×16 matmuls are too small for H100 FP8 tensor cores (need 4096+ for speedup)
3. 2.5M params = 5MB in fp16. No memory bottleneck to solve.
4. Small models have less redundancy — quantization error is proportionally worse.

## When to Revisit
- 50M+ params: FP8 training via Transformer Engine would help large linear layers
- 128K+ context: TurboQuant KV cache compression becomes relevant
- Deployment to edge: INT8 weight quantization for phones/microcontrollers

## Related Papers
- SVDQuant (ICLR 2025): rank-aware quantization, relevant conceptually
- Preserve-Then-Quantize (2026): explicit rank vs quantization tradeoff
