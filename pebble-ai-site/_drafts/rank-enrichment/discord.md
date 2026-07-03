---
target: research-oriented Discord servers
canonical: https://pebbleml.com/findings/rank-enrichment.html
status: ready_for_review
---

# Long form (#papers / #interpretability channels)

cross-posting from a small lab — would value structural critique:

Matrix Thinker (5.16M params, d=32, 8 shared thinking layers iterated T=8, trained on 2.2B OpenR1-Math tokens) shows monotonically RISING effective rank across iterative refinement under a bilinear output head: 5.02 → 5.41 → 5.67 → ... → 6.12.

direction-reversal under output-head changes with backbone held fixed:
· MultiProbeHead (K=32 bilinear probes): rises 5.02 → 6.12
· vector-collapse head: drops
· 3D matrix-product attention: drops 2.75 → 2.66, T=8 BPB 29% worse than Frobenius-attn baseline

interpretation: bilinear probes reward independent rank-1 components → backbone maintains them. vector-collapse head's per-class scoring matrix is rank-1 by construction → backbone is rewarded for concentrating energy.

caveats: single-seed across the board. toy scale. no causal rank-projection ablation yet (planned). cross-condition BPB comparison is weak evidence (different step counts, attention mechanisms).

companion result for the case where this fails: in matrix-CODI (separate arch, flatten-then-project readout), rank-k SVD truncation has zero effect on accuracy — readout's Jacobian is constant in Z, gradient cannot see rank. the broader claim from the two together is that rank dynamics in trained transformers are partly a property of the read-out's gradient shape, not just the backbone.

interested in: prior work I should be citing on output-head shaping of backbone dynamics, or counterexamples.

paper: https://pebbleml.com/findings/rank-enrichment.html

# Short form (#share / general)

matrix-valued transformer, single training run: effective rank rises 5.02 → 6.12 across 8 iterative refinement steps under a bilinear output head. flips direction under vector-collapse and 3D matrix-product. single-seed, output-head-dependent, no causal test yet.

https://pebbleml.com/findings/rank-enrichment.html
