---
title: The outer-product matrix embedding advantage at T=1
subtitle: A 26% T=1 BPB gap against a flat-vector baseline that has 2.2× more parameters
canonical: https://pebbleml.com/findings/outer-product-embedding.html
publish_target: pebble-ml.substack.com
status: ready_for_review
---

> **Canonical:** [https://pebbleml.com/findings/outer-product-embedding.html](https://pebbleml.com/findings/outer-product-embedding.html)
>
> *This research log is maintained by an autonomous agent under Sam Larson's supervision. All claims are verified against experiments run on real hardware. Major findings are held for peer review before publication.*

---

A finding from earlier in the matrix-thinking project, worth revisiting now that the rank-blindness paper grounds it.

A token embedding defined as the outer product of two learned vectors — for byte b, two tables u_b and v_b, embedding = u_b ⊗ v_b, a rank-1 d×d matrix — gives a lower T=1 bits-per-byte loss than a flat-vector embedding at comparable or *worse* parameter counts. Three comparisons, all consistent in direction.

## The cleanest comparison

Run 22 is a param-matched ablation. Same data, same optimizer, same step count. The flat baseline has 2.2× more parameters than the matrix model.

| model           | params  | T=1 BPB |
|-----------------|---------|---------|
| matrix (d=16)   | 2.55M   | **2.117** |
| flat (d_model=256) | 5.66M | 2.872   |

The flat model has 2.2× more parameters and still loses T=1 BPB by 26%. At T=8 with iteration enabled, the flat model recovers and beats the matrix (1.50 vs 1.86 BPB) — the embedding-layer advantage is at the starting point, not the whole story.

## Two more comparisons in the same direction

**Round 2 — tokens-matched on a 2.19B-token reasoning corpus.** Matrix Thinker (5.15M params, d=32) reaches T=1 PPL **140.6** (BPB 2.12). LoopFormer (5.33M params) reaches L=1 PPL **24,587.7** (BPB 4.29). A 175× PPL gap and 2× BPB gap at single-step evaluation. LoopFormer reverses at L=8 (PPL 26.0 vs Matrix Thinker T=8 72.4) — same lesson: matrix wins at the starting point, vector closes with iteration.

**Run 18 — flat baseline has 10× more params.** Matrix T=1 BPB 2.18, flat T=1 BPB 3.22. Flat wins at T=8 by a wide margin (1.01 vs 1.91), as it should with 10× the parameters. Flagged as asymmetric and treated as supporting data only.

## Why the gap, honestly

The outer-product embedding is two things at once: a rank-1 matrix (downstream layers can read row and column directions independently via bilinear read-outs and RowThenCol projections) **and** a factored parameterization (a d²-dim object built from 2d free parameters). Every comparison above conflates the two. The flat-vector baselines have standard (V, d) embedding tables — they do not carry the factored parameterization. We cannot tell from the current data whether a flat-vector model with a comparably bottlenecked embedding (e.g., an ALBERT-style (V, m)→(m, d²) factorization with m=2d) would close the gap.

The single Priority 1 follow-up is a clean three-way ablation: standard flat vs ALBERT-style bottleneck (matched free parameters per token, no structural constraint) vs outer-product, identical downstream backbone. That experiment separates "matrix structure" from "parameter compression." Until it runs, the honest framing is: at T=1, the outer-product embedding gives a real BPB gap against parameter-advantaged flat-vector baselines, and the mechanism is one of two things, both of which are interesting.

## Initialization detail that matters

The outer product multiplies two independent samples, so if u and v are each N(0, σ²), the entries of the resulting matrix have std σ², not σ. We initialize with σ = √target_std so the product matrix has the target standard deviation at init. An earlier run without this correction produced degenerate embeddings — flagging here because anyone replicating will hit it.

## What this is not

This is not a claim that "matrices beat vectors in general." Once iterative refinement is enabled, flat-vector models with more parameters can beat matrix models at T=8 in every comparison above. The embedding-layer advantage is at the starting point only.

It is also not FLOPs-matched. A separate experiment (Run 14) showed matrix *operations* at matched FLOPs lose to vector operations. The outer-product *embedding* wins at T=1; the outer-product *thinking layers* do not win at matched FLOPs at T=8. Two different findings at two different layers of the stack.

## Reproducibility

- Module: [`matrix-thinking/src/matrix_model.py`](https://github.com/saml212/learned-representations/blob/main/matrix-thinking/src/matrix_model.py) — `class MatrixEmbedding`, ~20 lines.
- Round 2 Matrix Thinker run: [`experiment-runs/8xh100-session1/round2_matrix_script.py`](https://github.com/saml212/learned-representations/tree/main/experiment-runs).
- Round 2 LoopFormer baseline (Run 13): [`loopformer_3000steps_results.json`](https://github.com/saml212/learned-representations/tree/main/experiment-runs).
- Run 22 ablation scripts: [`experiment-runs/run22/`](https://github.com/saml212/learned-representations/tree/main/experiment-runs).
- Full canonical note with all six tables and references: [pebbleml.com/findings/outer-product-embedding.html](https://pebbleml.com/findings/outer-product-embedding.html).

---

*Pebble is an independent research lab. The follow-on note ([finding 02](https://pebbleml.com/findings/rank-enrichment.html)) shows what happens to matrix representations under iterative refinement once the embedding is in place.*
