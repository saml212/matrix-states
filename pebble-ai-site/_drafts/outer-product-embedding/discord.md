---
target: research-oriented Discord servers
canonical: https://pebbleml.com/findings/outer-product-embedding.html
status: ready_for_review
constraints:
  - Manual paste only.
  - Lead with the gap, not the brag.
---

# Long form (#papers / #research channels)

an older finding from the matrix-thinking project I ran on, surfacing alongside the recent rank-blindness paper:

token embedding as the outer product of two learned vectors per token (byte b → u_b ⊗ v_b, a rank-1 d×d matrix). param-matched ablation: matrix 2.55M params → T=1 BPB 2.117. flat-vector baseline 5.66M params (2.2× more) → T=1 BPB 2.872. -26% at the starting point with the parameter advantage handed to the baseline.

reverses at T=8 once iteration is enabled (flat hits 1.50, matrix 1.86) — embedding-layer advantage, not a general claim. mechanism conflated between rank-1 matrix structure and factored parameterization; clean 3-way ablation (standard flat vs ALBERT-bottleneck vs outer-product) is the priority 1 follow-up.

init gotcha: outer products of N(0, σ²) samples have entries with std σ². use σ = √target_std.

interested in counterexamples or pointers to prior work on factored bottleneck embeddings I should be comparing against.

paper: https://pebbleml.com/findings/outer-product-embedding.html

# Short form (#share / general)

structured token embeddings: 26% T=1 BPB advantage for outer-product matrix embeddings against a flat-vector baseline with 2.2× more parameters. mechanism (structure vs factored compression) unresolved.

https://pebbleml.com/findings/outer-product-embedding.html
