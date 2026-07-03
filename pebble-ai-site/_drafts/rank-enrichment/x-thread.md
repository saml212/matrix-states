---
target: x.com
canonical: https://pebbleml.com/findings/rank-enrichment.html
status: ready_for_review
constraints:
  - 5 tweets. Hook stands alone.
  - Last tweet = canonical link.
  - Best @-mentions for tweet 5: researchers from the outreach list working on rank dynamics, structured representations, or bilinear pooling. Pick from the list manually.
---

# Tweet 1 (hook)

a result that runs against the rank-collapse direction the literature documents:

in a matrix-valued transformer with a bilinear output head, effective rank of the token matrices RISES monotonically across iterative refinement steps. 5.02 → 6.12 over 8 iterations.

🧵

# Tweet 2 (the finding)

trajectory under the MultiProbeHead output (single training run, 5.16M params, 2.2B OpenR1-Math tokens):

iter:    1    2    3    4    5    6    7    8
e-rank: 5.02 5.41 5.67 5.83 5.93 6.02 6.09 6.12

monotonic. counter to "depth drives collapse."

# Tweet 3 (the reversal)

same backbone, same data, same compute budget — change ONLY the output head and the rank trajectory reverses:

· bilinear probes (K=32): rank rises 5.02 → 6.12
· vector-collapse head: rank drops
· 3D matrix-product attn: rank drops 2.75 → 2.66, BPB 29% worse

the head shapes the gradient.

# Tweet 4 (mechanism + caveats)

why: the MultiProbeHead computes K bilinear scores per token. each probe reads a distinct rank-1 direction. spreading singular energy across probe axes produces richer logits → backbone is incentivized to maintain independent components → rank rises during iteration.

caveats: single seed. toy scale. no causal test yet.

# Tweet 5 (link)

companion result: in matrix-CODI (different arch, flatten readout) rank-k SVD truncation has zero effect on accuracy because the readout's Jacobian is constant in Z. the gradient there does NOT see rank.

so: rank dynamics depend on the read-out.

full note: https://pebbleml.com/findings/rank-enrichment.html
