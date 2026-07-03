---
target: r/MachineLearning
flair: "[R] Research"
title: "[R] Rank enrichment in matrix-valued token representations: effective rank rises monotonically across iterative refinement under a bilinear output head"
canonical: https://pebbleml.com/findings/rank-enrichment.html
status: ready_for_review
constraints:
  - Manual post only.
  - Single-seed result; lead with the caveat.
  - The Reddit angle is the direction-reversal across output heads, not just the enrichment alone — frame it as a structural finding about *what* shapes rank trajectories, not a "matrices are great" post.
---

# Title

[R] Rank enrichment in matrix-valued token representations: effective rank rises monotonically across iterative refinement under a bilinear output head

# Body

A single-seed observation from a matrix-valued transformer (Matrix Thinker, d=32, 8 shared thinking layers iterated T=8 times, 5.16M params, 2.2B-token training run on OpenR1-Math). Effective rank — exp(−Σ p_i log p_i) over the normalized singular values of the matrix-valued token, averaged over 512 held-out positions — rises monotonically across iterations under a bilinear output head:

| iter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| eff rank | 5.02 | 5.41 | 5.67 | 5.83 | 5.93 | 6.02 | 6.09 | 6.12 |

This runs counter to the depth-driven rank-collapse direction (Dong et al. 2021 and follow-ups) the prior literature documents. The interesting part isn't the enrichment alone — it's that the direction of the rank trajectory tracks the output head, with backbone, training data, and compute budget held identical.

| configuration | rank trajectory | T=8 BPB |
|---|---|---|
| Frobenius attn + MultiProbeHead (32 bilinear probes) | 5.02 → 6.12 | 1.670 |
| Frobenius attn + vector-collapse head | drops (direction only) | ~1.72 |
| 3D matrix-product attention | 2.75 → 2.66 | 2.457 |

The MultiProbeHead computes K=32 bilinear scores per token (probe_k = u_k^T M v_k). Each probe is sensitive to the rank-1 direction u_k v_k^T. A matrix that spreads its singular energy across multiple probe axes produces a richer logit signal than one that concentrates — gradient descent on the cross-entropy loss therefore rewards maintaining linearly independent rank-1 components, and over iterations builds them up. The vector-collapse head's effective per-class scoring matrix is rank-1 by construction (all vocab entries share the same column directions, differ only in row scalings) and rewards concentration. 3D matrix-product attention imposes pairwise consistency that reduces the matrix's degrees of freedom.

Caveats:

- Single seed in each condition. n=1 across the board. Trajectory is monotonic across 8 iterations under MultiProbeHead which makes a pure-noise explanation strained, but I do not report confidence intervals.
- Toy scale (5.16M params). The direction-reversal may weaken or disappear at >10M.
- No causal test yet. Rank rises *and* BPB improves under MultiProbeHead, but I have not shown rank causes the BPB. A forced rank-projection ablation (project M to rank k at eval, measure accuracy) would separate causation from correlation.
- The cross-condition BPB comparison is weak evidence — the 3D matrix-product run uses 1000 steps instead of 3000 and a different attention mechanism, the vector-collapse run is on smaller data with different optimizer settings. Direction of rank trajectory within each run is the cleaner signal.
- Output-head confound is real and unresolved. It's possible the enrichment reflects implicit regularization from bilinear probe geometry rather than a property of the backbone responding to gradient signal.

The companion finding ([finding 06 / rank-blindness paper](https://pebbleml.com/findings/matrix-codi-rank-blindness.html), ICML 2026 MI workshop submission) shows the opposite case in a different architecture: matrix-CODI's flatten-then-project readout produces rank-indifferent gradients regardless of the readout's nonlinearity in Z, and rank-k SVD truncation of the matrix latent does not affect accuracy.

Full note with figures, discussion, and reproducibility pointers: https://pebbleml.com/findings/rank-enrichment.html

Code: https://github.com/saml212/learned-representations
