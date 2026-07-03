---
target: linkedin.com
canonical: https://pebbleml.com/findings/output-head-dynamics.html
status: ready_for_review
audience: interpretability researchers, lab leads
attached_image: output_head_dynamics.svg
note: Wait at least 5 days after the rank-enrichment LinkedIn post.
---

A follow-on to the rank-enrichment finding I shared last week:

**The output head, not the backbone, determines whether the rank of token representations rises or falls during iterative refinement.**

Three configurations of the Matrix Thinker architecture, same matrix-thinking layers and update rule:

· Bilinear MultiProbeHead — effective rank **rises** 5.05 → 6.13 across 8 iterations.
· Vector-collapse output head — rank falls.
· 3D matrix-product attention — rank falls 2.75 → 2.66, downstream BPB 29% worse than the comparable Frobenius-attention baseline.

Same backbone. Same multiplicative thinking layers. Opposite internal dynamics. The output mechanism shapes the gradient signal reaching the backbone — and the shape of that signal dictates whether the model's representations diversify or concentrate during iteration.

**Why this matters for interpretability:**

The transformer-rank literature treats rank dynamics as a property of the architecture — depth drives collapse, residual connections mitigate it, MLPs help, etc. The implicit assumption is that the read-out layer is thin and does not meaningfully alter what happens below it. The result above complicates that. If the head can flip the *sign* of rank trajectories — not just shift magnitudes, but reverse direction — then claims about "depth drives X" are claims about a particular training incentive, not about the architecture in isolation.

When the same probe is applied to two models that differ only in head, it can be reading different things.

This sits next to the rank-blindness paper at architectural extreme: in matrix-CODI under a flatten readout, the readout's Jacobian is constant in the matrix latent Z, the gradient is rank-indifferent by construction, and rank-k SVD truncation of Z has zero effect on accuracy. The output mechanism shapes whether rank is a functional observable *at all*, and the sign and magnitude of the trajectory if it is.

Caveats: single-seed runs, toy scale (5.16M parameters), cross-run comparison is unmatched on training corpus and step count. The within-run direction is the cleaner signal; the magnitudes are weaker. Causal rank-projection ablation is queued.

Full note with all three configurations and limitations:
https://pebbleml.com/findings/output-head-dynamics.html

---

*Pebble is an independent research lab. The research log is maintained by an autonomous agent under my supervision; all claims are verified against experiments run on real hardware.*
