---
target: x.com
canonical: https://pebbleml.com/findings/outer-product-embedding.html
status: ready_for_review
constraints:
  - 5 tweets. Hook stands alone.
  - Last tweet = canonical link.
  - Before posting, tag 2-3 researchers from researcher-outreach-list.md who work on token representations / structured embeddings.
---

# Tweet 1 (hook)

a structured token embedding I've been testing beats a flat-vector baseline at T=1 by 26% BPB.

the kicker: the flat baseline has 2.2× MORE parameters than the matrix model.

🧵

# Tweet 2 (the embedding)

it's the outer product of two learned vectors per token:

byte b → embed(b) = u_b ⊗ v_b ∈ ℝ^{d×d}, rank-1.

2d free params per token, d² entries. downstream layers can read row and column directions independently via bilinear probes and RowThenCol projections.

# Tweet 3 (the comparisons)

three comparisons, all same direction at T=1:

· Run 22 (param-matched): matrix 2.55M → BPB 2.117. flat 5.66M → BPB 2.872. -26%.
· Round 2 (2.19B tokens): matrix 5.15M → BPB 2.12. LoopFormer 5.33M → BPB 4.29. -51%.
· Run 18 (10× param-asymm): matrix 2.4M → 2.18. flat 24M → 3.22.

# Tweet 4 (honest caveats)

what this is NOT:

· not a claim that matrices beat vectors in general. flat baselines close + reverse the gap at T=8 with iteration (Run 22 flat at T=8 hits BPB 1.50 vs matrix 1.86).
· not FLOPs-matched. matrix *operations* at matched FLOPs lose to vector ops (different finding).
· mechanism unresolved: structure vs factored parameterization. clean 3-way ablation pending.

# Tweet 5 (link out)

initialization gotcha for anyone replicating: outer products of N(0, σ²) samples have entries with std σ², not σ. use σ = √target_std on each table.

full note: https://pebbleml.com/findings/outer-product-embedding.html
