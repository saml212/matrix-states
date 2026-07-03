---
target: r/MachineLearning
flair: "[D] Discussion"
title: "[D] Per-layer parameter efficiency of matrix-native projections vs flattened Linear: 128× at d=16, but only 8× FLOP gap"
canonical: https://pebbleml.com/findings/parameter-efficiency.html
status: ready_for_review
constraints:
  - [D] flair (discussion) rather than [R] — this is a structural observation, not an experimental result.
  - Manual post only.
  - The honest FLOP framing IS the post — leading with "128× fewer params" without the FLOP caveat will eat downvotes from anyone who reads the math.
---

# Title

[D] Per-layer parameter efficiency of matrix-native projections vs flattened Linear: 128× at d=16, but only 8× FLOP gap

# Body

A structural observation about matrix-valued transformer architectures. At matrix dimension d=16, a RowThenCol bilinear projection — silu(A · M) · B with A, B ∈ ℝ^{d×d} — uses 2·d² = 512 parameters per projection. The flattened equivalent — reshape M to a d²-dim vector, apply a (d², d²) dense Linear — uses d⁴ = 65,536. Ratio: 128×.

| projection | params at d=16 | FLOPs (fwd) | FLOPs/param |
|---|---|---|---|
| Flattened Linear | 65,536 | 131,072 | 2 |
| Kronecker K=8 (Σ A_k M B_k) | 4,096 | 131,072 | 32 |
| Kronecker K=4 | 2,048 | 65,536 | 32 |
| RowThenCol bilinear | 512 | 16,384 | 32 |

The honest part. A flattened Linear spends each parameter once per token per step (two FMAs per scalar weight). A matrix-native projection uses each scalar parameter d times during the sandwich computation (A·M is an inner product against every column of M). At d=16 that's 16× reuse, which cancels one factor of 16 from the 128× parameter gap. RowThenCol at d=16 therefore costs 8× fewer FLOPs per step, not 128×. Realized speedup on H100 is not measured; memory bandwidth and kernel launch overhead are expected to eat much of the 8×.

What this lets you do at small scale: in an 8-layer Matrix Thinker at d=32 with a 50K BPE vocab, the 8 thinking layers are ~4% of total parameters. Iterative refinement reuses them T=8 times per forward pass — 64 effective layer-applications at the cost of 8 layers of parameters. At byte-level d=16, the embedding shrinks ~100× (256 bytes vs 50K BPE) and thinking layers claim a much larger share (Run 19: 218K total parameters, 33% in thinking layers, 12 layers deep).

What this is NOT. Not a claim matrix operations are better at matched compute. At matched FLOPs the matrix approach loses by a wide margin — Matrix Thinker T=8 BPB 1.67 vs LoopFormer 0.87 at the same 653K TFLOPs budget (`EXPERIMENT_LOG.md` Run 14). Matrix-native projections are parameter-efficient. They're useful where parameters are scarce and compute is not — small models, byte-level vocabularies, memory-tight deployment.

The related architectural angle: at fixed total parameter budget, a matrix-thinking backbone can afford to be ~128× deeper than a flattened-Linear backbone at the same d. Whether deeper translates to better is a separate question the parameter efficiency does not settle.

Related literature: PHM (Zhang et al. 2021) is the closest established cousin — same family of "express a linear layer as a sum of Kronecker products" parameter savings.

Full note: https://pebbleml.com/findings/parameter-efficiency.html
Code: https://github.com/saml212/learned-representations
