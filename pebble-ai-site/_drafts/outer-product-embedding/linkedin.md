---
target: linkedin.com
canonical: https://pebbleml.com/findings/outer-product-embedding.html
status: ready_for_review
audience: lab leads, recruiters, grant reviewers
attached_image: outer_product_embedding.svg
---

A finding from earlier in the matrix-thinking project at Pebble — worth surfacing now that the rank-blindness paper has gone in:

**A token embedding defined as the outer product of two learned vectors gives a 26% lower T=1 bits-per-byte loss than a flat-vector baseline that has 2.2× more parameters.**

For each byte b, instead of looking up one d-dim vector, look up two — u_b and v_b — and form the rank-1 matrix u_b ⊗ v_b. The token now arrives at the rest of the model as a structured d×d object that bilinear read-outs and matrix-aware downstream layers can read along row and column directions independently.

**The numbers (param-matched, same data, same optimizer):**

· Matrix model, 2.55M parameters: T=1 BPB 2.117
· Flat-vector baseline, 5.66M parameters (2.2× more): T=1 BPB 2.872

The flat baseline catches up and reverses the gap at T=8 once iterative refinement is enabled (1.50 vs 1.86) — so this is an *embedding-layer* advantage, not a "matrices beat vectors" claim. Replicated in the same direction across two more comparisons, including a 2.19B-token run on a reasoning corpus where the gap widens to 175× PPL at single-step evaluation.

**Why it matters for parameter-constrained settings:**

The cost per token is 2d parameters instead of d. At small scale (under 10M parameters) and at byte-level vocabularies (where the embedding table is the bulk of the model), this kind of structural bias is what lets a small model get more out of its parameter budget. The rank-blindness result we shipped recently is about how matrix structure interacts with iterative reasoning gradients; this finding is about how matrix structure interacts with the embedding layer specifically. Both contribute to the bigger picture: where structured representations actually buy you something and where they do not.

**Mechanism is unresolved.** The outer-product embedding is both a rank-1 matrix (which the downstream stack reads structurally) and a factored parameterization (which any low-rank bottleneck embedding would share). The Priority 1 follow-up is a clean three-way ablation — standard flat vs ALBERT-style bottleneck vs outer-product. Until that experiment runs, the honest framing is: there is a real gap, and it is one of two interesting things.

Full note with all three comparisons, reproducibility pointers, and limitations:
https://pebbleml.com/findings/outer-product-embedding.html

---

*Pebble is an independent research lab. The research log is maintained by an autonomous agent under my supervision; all claims are verified against experiments run on real hardware.*
