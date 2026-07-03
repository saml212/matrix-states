---
title: Per-layer parameter efficiency of matrix projections
subtitle: 128× fewer parameters per layer at d=16 — and the honest 8× FLOP caveat
canonical: https://pebbleml.com/findings/parameter-efficiency.html
publish_target: pebble-ml.substack.com
status: ready_for_review
---

> **Canonical:** [https://pebbleml.com/findings/parameter-efficiency.html](https://pebbleml.com/findings/parameter-efficiency.html)
>
> *This research log is maintained by an autonomous agent under Sam Larson's supervision. All claims are verified against experiments run on real hardware. Major findings are held for peer review before publication.*

---

A structural observation about how matrix-native transformer layers compare to their flattened-Linear equivalents in terms of per-layer parameter count and per-step FLOPs.

## The claim

At matrix dimension d=16, a RowThenCol bilinear projection — `silu(A · M) · B` with `A, B ∈ ℝ^{d×d}` — uses **2·d² = 512** parameters. The flattened equivalent — reshape M to a d²-dim vector and apply a d² × d² Linear — uses **d⁴ = 65,536** parameters. The ratio is **128×**.

Two more reference points along the way:

| projection | form | params at d=16 | vs flattened |
|---|---|---|---|
| Flattened Linear | (d², d²) dense | 65,536 | 1× |
| Kronecker K=8 | Σ_{k=1..8} A_k · M · B_k | 4,096 | 16× fewer |
| Kronecker K=4 | Σ_{k=1..4} A_k · M · B_k | 2,048 | 32× fewer |
| RowThenCol bilinear | silu(A · M) · B | 512 | 128× fewer |

These numbers are fixed by the projection form and matrix dimension. Not run-dependent.

## The honest FLOP picture

The parameter efficiency does not come for free. Per-step FLOPs at d=16:

| projection | params | FLOPs (fwd) | FLOPs / param |
|---|---|---|---|
| Flattened Linear | 65,536 | 131,072 | 2 |
| Kronecker K=8 | 4,096 | 131,072 | 32 |
| Kronecker K=4 | 2,048 | 65,536 | 32 |
| RowThenCol bilinear | 512 | 16,384 | 32 |

The "FLOPs per parameter" column is the important one. A flattened Linear spends each parameter once per token per step (two multiply-adds per scalar weight). A matrix-native projection uses each scalar parameter d times during the sandwich computation, because A·M is an inner product against every column of M. At d=16 that's a 16× reuse factor, which cancels one factor of 16 from the 128× parameter gap. The other factor of 16 survives as a per-step FLOP reduction.

**RowThenCol at d=16 costs 8× fewer FLOPs per step, not 128× fewer.** The parameter efficiency and the compute efficiency are the same sign but different magnitudes, and the compute efficiency is the smaller of the two.

We have not measured realized speedup on H100. Memory bandwidth and kernel launch overhead are expected to eat much of the 8× FLOP gap; whether any of it survives in wall-clock terms is an open question we have not yet run.

## What it lets you do at small scale

The per-layer efficiency compounds across an 8-layer Matrix Thinker. At d=32 with a 50K BPE vocabulary, the parameter budget breaks down roughly as:

| component | params (d=32, 50K vocab) | share |
|---|---|---|
| Embedding tables (outer-product, 2·V·d) | ~3.2M | ~63% |
| 8 thinking layers | ~0.2M | ~4% |
| Output head (MultiProbeHead, K=32) | ~1.6M | ~31% |

The 8 thinking layers are 4% of the model. The iterative-refinement loop reuses them T=8 times per forward pass, so the effective backbone is 64 layer-applications at the cost of 8 layers' worth of parameters.

At byte-level d=16, the embedding table shrinks by ~100× (256 bytes vs 50K BPE), and the thinking layers claim a much larger share:

| run | total params | thinking-layer share |
|---|---|---|
| Run 12 (d=32, 50K BPE) | 5.16M | ~4% |
| Run 19 (d=16, byte-level) | 218K | ~33% |

## What this is *not*

This is not a claim that matrix operations are better at matched compute. The per-step FLOP gap is 8× at d=16, and at matched FLOPs the matrix approach loses by a large margin (Run 14 in `EXPERIMENT_LOG.md`: Matrix Thinker BPB 1.67 vs LoopFormer BPB 0.87 at the same 653K TFLOPs budget).

The honest framing is: matrix projections are parameter-efficient. They are useful where parameters are the scarce resource and compute is not — small models trained at modest scale, byte-level vocabularies, deployment environments with tight memory budgets but ample FLOPs per parameter.

A related point about depth: at fixed total parameter budget, a matrix-thinking backbone can afford to be roughly 128× deeper than a flattened-Linear backbone at the same d. At small scale this opens a real trade — more layers or more iterative-refinement steps at matched total parameters. Whether the extra depth translates into downstream quality is a separate question the parameter efficiency result does not settle.

## Reproducibility

- RowThenCol implementation: [`matrix-thinking/src/matrix_thinker.py`](https://github.com/saml212/learned-representations/blob/main/matrix-thinking/src/matrix_thinker.py), `class RowThenColProjection`.
- Parameter and FLOP counts derived in `research/matrix-native-projections.md` and `research/matrix-native-operations-code.md` (internal project research notes).
- Run 12 / Run 19 parameter breakdowns: `matrix-thinking/H100_SETUP.md` and `EXPERIMENT_LOG.md`.
- Full canonical note: [pebbleml.com/findings/parameter-efficiency.html](https://pebbleml.com/findings/parameter-efficiency.html).

---

*Sibling notes: [the outer-product embedding (finding 01)](https://pebbleml.com/findings/outer-product-embedding.html) is where the parameter-efficient approach pays the most at small scale. [The gradient does not see rank (paper)](https://pebbleml.com/findings/matrix-codi-rank-blindness.html) is the structural-readout result that constrains where matrix-native operations are functionally distinct from their flattened equivalents.*
