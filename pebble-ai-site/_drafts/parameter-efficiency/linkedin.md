---
target: linkedin.com
canonical: https://pebbleml.com/findings/parameter-efficiency.html
status: ready_for_review
audience: engineers, lab leads, infrastructure people, people building on tight memory budgets
attached_image: parameter_efficiency.svg
---

A structural observation from the matrix-thinking project at Pebble that has implications for parameter-constrained model design:

**At matrix dimension d=16, a matrix-native bilinear projection uses 128× fewer parameters than the flattened-Linear equivalent. The honest per-step FLOP gap is 8×, not 128×.**

The per-layer math is straightforward. A standard transformer's projection between d²-dimensional vectors is a (d², d²) dense Linear with d⁴ parameters — 65,536 at d=16. A matrix-native projection silu(A · M) · B with A, B ∈ ℝ^{d×d} uses 2·d² = 512 parameters, a 128× reduction.

The catch: a matrix-native projection uses each scalar parameter d times during the sandwich computation (the column-wise inner products in A·M), so the per-step FLOP reduction is 16× smaller than the parameter reduction. At d=16, the realized FLOP gap is 8× rather than 128×.

**Where this matters:**

In an 8-layer Matrix Thinker backbone at d=32 with a 50K BPE vocabulary, the 8 thinking layers occupy ~4% of the parameter budget — embeddings and the output head dominate. With iterative refinement that reuses the same 8 layers T=8 times per forward pass, the model gets 64 effective layer-applications at the cost of 8 layers' worth of parameters.

The regime where the parameter gap binds most directly is byte-level: at vocabulary 256, the embedding table stops dominating. A 218K-parameter byte-level Matrix Thinker (12 layers deep, d=16) puts a third of its parameter budget in the backbone. A flattened-Linear backbone at the same depth and d would cost ~3.1M parameters in projections alone — fifteen times the total Run 19 model.

**What this is not:**

Not a claim that matrix operations are better at matched compute. At matched FLOPs the matrix approach loses to the vector baseline by a wide margin — 1.67 vs 0.87 BPB at the same 653K TFLOPs budget in our cross-experiment comparison.

Matrix-native projections are *parameter-efficient*. They are useful where parameters are the scarce resource and compute is not: small models trained at modest scale, byte-level vocabularies, deployment environments with tight memory budgets but ample FLOPs per parameter.

The related architectural lever: at fixed total parameter budget, a matrix-thinking backbone can afford to be roughly 128× deeper than a flattened-Linear backbone at the same d. Whether the extra depth translates into downstream quality is a separate question the parameter efficiency result does not settle.

Full note with all four projection families, FLOP comparisons, and parameter allocation tables:
https://pebbleml.com/findings/parameter-efficiency.html

---

*Pebble is an independent research lab. The research log is maintained by an autonomous agent under my supervision; all claims are verified against experiments run on real hardware.*
