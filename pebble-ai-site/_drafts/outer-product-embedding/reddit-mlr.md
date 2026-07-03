---
target: r/MachineLearning
flair: "[R] Research"
title: "[R] Outer-product matrix embeddings beat flat-vector baselines at T=1 by 26% BPB despite the baseline having 2.2× more parameters"
canonical: https://pebbleml.com/findings/outer-product-embedding.html
status: ready_for_review
constraints:
  - Manual post only.
  - Tue/Wed 9am PT for max news.smol.ai scrape window.
  - Within 30 min of posting, drop the GitHub link in a self-reply comment.
---

# Title

[R] Outer-product matrix embeddings beat flat-vector baselines at T=1 by 26% BPB despite the baseline having 2.2× more parameters

# Body

Defining a token embedding as the outer product of two learned vectors — byte b → u_b ⊗ v_b, a rank-1 d×d matrix — gives a lower single-step bits-per-byte loss than a standard flat-vector embedding at comparable or worse parameter counts. Three comparisons in this direction.

The cleanest is a param-matched ablation. Same data, optimizer, step count. Matrix model (d=16, 2.55M params) reaches T=1 BPB 2.117. Flat-vector baseline (d_model=256, 5.66M params, 2.2× more parameters than the matrix model) reaches T=1 BPB 2.872. The flat model wins at T=8 with iteration (1.50 vs 1.86) but loses at the starting point by 26% despite the parameter advantage.

A larger comparison on 2.19B tokens of OpenR1-Math reasoning data: Matrix Thinker at 5.15M params reaches T=1 PPL 140.6 / BPB 2.12. Tokens-matched LoopFormer at 5.33M params reaches L=1 PPL 24,587.7 / BPB 4.29 — a 175× PPL gap at single-step. LoopFormer recovers and reverses at L=8 (PPL 26.0 vs Matrix Thinker T=8 72.4). The lesson is the same: vector baselines close the gap with iteration; the embedding-layer advantage is at T=1.

The honest mechanism question is unresolved. The outer-product embedding is simultaneously (a) a rank-1 matrix that downstream bilinear read-outs and RowThenCol projections can read along row and column directions, and (b) a factored parameterization with 2d free parameters per token producing a d²-dim object. Every comparison above conflates the two. A flat-vector model with a comparably bottlenecked embedding (e.g., ALBERT-style (V, m)→(m, d²) factorization, m=2d) might close the gap. The Priority 1 follow-up is a clean three-way ablation — standard flat vs ALBERT-bottleneck vs outer-product, downstream backbone held fixed.

One initialization gotcha: outer products of independent N(0, σ²) samples have entries with std σ², not σ. Use σ = √target_std on the two embedding tables or you get degenerate embeddings.

This is single-seed, all runs under 25M parameters, and the T=1 framing is narrow (the iteration story reverses for high-param flat baselines). The follow-on note on rank dynamics during iterative refinement is at [finding 02](https://pebbleml.com/findings/rank-enrichment.html).

Full note, all three comparisons with reproducibility pointers: https://pebbleml.com/findings/outer-product-embedding.html

Code: https://github.com/saml212/learned-representations
