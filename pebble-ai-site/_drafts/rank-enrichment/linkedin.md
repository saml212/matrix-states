---
target: linkedin.com
canonical: https://pebbleml.com/findings/rank-enrichment.html
status: ready_for_review
audience: lab leads, recruiters, grant reviewers, mech-interp researchers
attached_image: rank_enrichment.svg
---

A second finding from the matrix-thinking project at Pebble that I want to surface alongside the rank-blindness paper, because the two together say something the prior literature does not:

**The rank trajectory of representations through a transformer depends on the output head, not just the backbone.**

The transformer-rank literature — Dong et al. 2021 and the line of follow-up work — documents *rank collapse*: token representations converge to a low-rank manifold as they pass through layers. This has been read as an architectural property of attention-plus-residual stacks. The implicit assumption is that the output head is a thin read-out layer whose gradient shape does not meaningfully alter what happens below it.

In a matrix-valued transformer (Matrix Thinker, d=32, 8 thinking layers iterated 8 times per forward pass, 5.16M parameters, trained on 2.2B tokens of reasoning data), I observed the opposite of collapse under a specific output-head choice. With a bilinear output head — K=32 probes that each read a distinct rank-1 direction of the token matrix — effective rank *rises* monotonically across iterations: 5.02 → 5.41 → 5.67 → ... → 6.12. With a vector-collapse output head, rank drops. With 3D matrix-product attention, rank also drops and downstream BPB is 29% worse than the closest comparable baseline. Same matrix-thinking layers, same training data, same compute — opposite internal dynamics.

**Why the head matters:**

A bilinear probe head reads K independent rank-1 features from each token matrix. Gradient descent on cross-entropy loss therefore rewards a backbone that maintains — and over iterations builds up — linearly independent rank-1 components, because each component contributes additional signal through a different probe.

A vector-collapse head's per-class effective scoring matrix is rank-1 by construction. The backbone's incentive is to concentrate information into a single dominant direction during iteration. Same backbone, opposite reward.

**Why this matters for the field:**

Mechanistic interpretability work that uses rank as a depth-monitoring tool implicitly assumes the backbone is the only thing being measured. The reversal under output-head changes shows that what gets read is partly the backbone and partly the head. Comparisons across models with different heads can therefore be reading different things.

This sits next to the rank-blindness paper I shipped recently — a different architecture where matrix-CODI's flatten readout produces *rank-indifferent* gradients, and rank-k SVD truncation of the matrix latent has zero effect on accuracy. The two findings together: the output mechanism shapes whether rank is a functional observable at all, and in some cases (matrix-CODI flatten readout) the gradient cannot see rank even in principle.

This is single-seed, toy-scale, and the causal test is queued. I am posting it because the direction-reversal is real and the cross-architecture comparison with rank-blindness is the cleanest version of the broader claim I can make right now.

Full note with all three configurations, figures, and limitations:
https://pebbleml.com/findings/rank-enrichment.html

---

*Pebble is an independent research lab. The research log is maintained by an autonomous agent under my supervision; all claims are verified against experiments run on real hardware.*
