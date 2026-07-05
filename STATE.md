# Project State

**Last updated:** 2026-07-04

This document is the project dashboard. Anyone returning to the project (you, a collaborator, a grant reader, an experimenter agent) should read this first to answer: where is the project right now?

---

## The Thesis

Language is a powerful cognitive tool. It encodes most of accumulated human knowledge, and any useful model has to interpret it and communicate in it. It developed under specific constraints — embodied, serial, social communication at ~50 bit/s — and machines operate under different constraints. We investigate whether using language as the cognitive medium for machines is an artificial constraint that limits abstract reasoning.

Language models inherit two layers of abstraction from a research culture that started with text and built outward: tokenizers, which impose linguistic structure on raw data, and flat-vector token representations, which leave the internal complexity of a token implicit. Fedorenko et al. 2024 *Nature* shows the human language network is dissociable from reasoning, math, and theory of mind. The brain treats language as a communication tool, with separate networks for thinking.

We investigate whether better-fitting representations enable stronger generalization and clearer abstract reasoning, both in the model's internal computation and in inference-time reasoning (the GPT-o1 "think before answering" regime).

The work proceeds along three threads:

**Thread 1 — byte-level inputs.** The model ingests data the way it lives in computers: raw bytes. Text as UTF-8, images as pixels, audio as samples, code as source bytes. Removing the tokenization layer lets the model develop its own vocabulary from the data.

**Thread 2 — matrix-valued token representations.** Each token is a d×d matrix. The matrix structure provides a measurable, differentiable observable — rank — that quantifies how many independent reasoning paths a representation holds in superposition. Rank can be computed from a single matrix; for a vector representation it can only be estimated across an ensemble. The matrix structure also affords contextualized token embeddings, where each token's matrix encodes learned pairwise interactions across a local window of inputs. The embedding carries relational structure from the start.

**Thread 3 — inference-time reasoning with structured representations.** We test whether matrix tokens make abstract reasoning measurable during inference-time compute. The setting is continuous-reasoning models like CODI and COCONUT, where the model "thinks" by generating latent representations before producing an output. The empirical question: does matrix rank track the number of distinct reasoning paths a model holds during this process? An affirmative answer would resolve a current dispute in the literature about whether superposition of reasoning paths is a structural property or a phenomenological description.

The unifying question: can structured representations enable stronger experience and generalization than language-shaped baselines, in a way that scales to inference-time reasoning?

---

## Mathematical Foundations

### Outer products and rank-1 matrices

For vectors `u, v ∈ ℝ^d`, the outer product `u ⊗ v` is the `d × d` matrix with entries `(u ⊗ v)[i, j] = u[i] · v[j]`. Every outer product has rank 1: every row is a scalar multiple of `v`, every column is a scalar multiple of `u`. Outer products have `d²` entries but only `2d` degrees of freedom.

Our byte embedding is a rank-1 outer product: `byte_b → u_b ⊗ v_b` with `u_b, v_b ∈ ℝ^16`, producing a 16×16 matrix.

### Rank as sum of rank-1 components

A matrix `M` of rank `r` can be written as a sum of `r` outer products:

```
M = Σᵢ₌₁ʳ uᵢ ⊗ vᵢ
```

The rank is the smallest such `r`. Equivalently, via the Singular Value Decomposition `M = UΣV^T`, the rank is the number of nonzero singular values.

### Continuous (differentiable) rank

Discrete rank is non-differentiable. For training and measurement, three continuous proxies:

**Stable rank:** `‖M‖_F² / ‖M‖_2² = (Σᵢ σᵢ²) / σ₁²`. Always between 1 and `rank(M)`.

**Participation ratio:** `(Σᵢ σᵢ)² / Σᵢ σᵢ²`.

**Effective rank (entropy of singular values):** `effective_rank(M) = exp(H(p))` where `pᵢ = σᵢ / Σⱼ σⱼ` and `H(p) = -Σᵢ pᵢ log pᵢ`.

All three measure how spread out the singular value distribution is. We use stable rank as the primary metric.

### Superposition encoding

Suppose a continuous reasoning model holds `r` distinct hypotheses. If each hypothesis `hᵢ` corresponds to a (row-pattern, column-pattern) pair `(uᵢ, vᵢ)`, the matrix encoding all hypotheses simultaneously is:

```
M = Σᵢ αᵢ (uᵢ ⊗ vᵢ)
```

where `αᵢ` is the confidence weight on hypothesis `i`. By construction, `rank(M) ≤ r`. If the `(uᵢ, vᵢ)` pairs are linearly independent in matrix space, `rank(M) = r` exactly.

A bilinear probe `(u_w, v_w)` reads from this matrix as:

```
logit(w) = u_w^T M v_w = Σᵢ αᵢ ⟨u_w, uᵢ⟩ ⟨v_w, vᵢ⟩
```

The matrix can hold up to `d` linearly independent hypothesis encodings before they interfere. **The hypothesis is that this is the number we should be measuring during reasoning.**

### Why rank is the relevant capacity

The CoT2 paper (arXiv 2505.23648) argues that parallelism in continuous-vector reasoning is bounded by embedding dimension `d`. The matrix analog: parallelism in continuous-matrix reasoning is bounded by matrix rank, which can range from 1 to `d`. Rank gives a measurable structural property that vectors can only approximate via ensemble statistics.

---

## What We've Built and Shown

### Findings that survived attack

- **Outer-product matrix embedding gives better per-parameter representations at T=1.** Reproduced across every configuration tested. T=1 BPB 2.12 (matrix d=32) vs 4.29 (vector LoopFormer baseline) at matched parameters. Held in Round 1, Round 2, byte-level d=16, and the partial param-matched ablation (Run 22).

- **Rank enrichment is an emergent novel phenomenon.** With MultiProbeHead output, effective rank rises during iterative refinement (5.02 → 6.12 across 8 iterations in Round 2). Not reported in any prior literature.

- **The output head determines representation dynamics.** MultiProbeHead drives enrichment (rank rises). Vector-collapse output drives solidification (rank falls). 3D matrix-product attention drives solidification with worse BPB. Same backbone, different output head, different rank trajectory. Novel empirical finding.

- **130× parameter efficiency per layer** for matrix operations versus standard vector layers at d=16.

- **Thought interleaving works mechanically.** When toggled at inference time, thoughts contribute (Run 25 sweep: 10.6% benefit at N=4 with toggle ablation).

### Honest negative results

- **Matrix operations lose at matched FLOPs.** LoopFormer FLOPs-matched: BPB 0.87 vs Matrix d=32: BPB 1.67. The quality gap is algorithmic, not a speed problem. Confirmed by the cheap-ops waterfall (22 ideas brainstormed, 5 validated, none close the gap). **[CORRECTION 2026-07-02]** An independent FLOPs-accounting audit found this was never FLOPs-matched in either direction — LoopFormer's cited best (BPB 0.87) occurred at step 21,500 (not "~step 40K") and used only ~0.5-0.6× of Matrix Thinker's total compute, while Matrix Thinker was itself undertrained relative to its own budget. The converged, genuinely FLOPs-matched gap is unmeasured; this does not mean matrix ops secretly win, but the "2× at matched FLOPs" framing is not supported by the runs as executed. See EXPERIMENT_LOG.md, "FLOPs-accounting audit of Runs 12-15 (2026-07-02)." Stage G (`matrix-thinking/STAGE_G_DESIGN.md`) is designed to measure the matched-FLOPs gap properly. **[UPDATE 2026-07-02, Stage G Wave A/B]** Stage G's Wave 0-R2 gap baseline (byte vocab, d=32, matched 3,000-step budget) measured a genuine, properly matched-tokens gap (matrix 3.5552 vs vector 3.2511 BPB, G=0.3040) and separately FALSIFIED the undertraining hypothesis (H_d) at this regime — extended 3× budget widens the gap, it does not close it (`STAGE_G_DESIGN.md` §13). The follow-on Wave A/B component-swap screen (§14) then found the gap **has a named mechanism**: none of 5 training-setup/plumbing axes (init scale, embedding rank, iteration conditioning, depth structure, output-head tying) recover any of it (all ≤+0.006 `recovered_frac`), but relaxing the Kronecker-separable `RowThenColProjection` restriction on the attention/thinking projections to a dense rank-swept bottleneck recovers ~64% of G at matched params (3/3 seeds ≥0.5) — while using *fewer* FLOPs than what it replaces. A capacity-matched vector control shows most of the apparent "inversion" seen at higher rank is extra parameters, not the projection swap (vector wins by 0.115 BPB at matched capacity), and the per-FLOP tax survives everywhere measured (≈16.5× even at the cheapest matrix winner). See `STAGE_G_DESIGN.md` §14 for the full table.

- **Thought interleaving does not beat adding layers.** Run 25 sweep at 288K params: thought config A scored BPB 3.535, no-thought config E (just more layers) scored BPB 3.524. The thoughts are dead weight at this scale.

- **3D matrix attention drives solidification and worse BPB.** Confirmed dead end. Drop it.

- **PonderNet halting collapses at small scale.** Run 8 expected_steps converged to 1.0. Use fixed iterations or LoopFormer-style consistency training instead.

- **Cross-domain generalization via matrix structure** was attacked by 6 fatal arguments before any experiment ran (research/hypothesis-attack-april2026.md). The hypothesis as originally stated is wrong (reshape equivalence, distribution mismatch).

- **The original PHM + learned-byte-segmentation project (Phase 1-2)** died at 26 experiments. PHM converges to nilpotent algebra rather than learning meaningful structure. Learned segmentation loses to fixed-stride at the scales tested. Archived to `archive/byte-agnostic/`.

### Complete experiment record (26 experiments)

| Era | Count | Outcome |
|---|---|---|
| Runs 1-7: PHM + Byte-Agnostic | 7 | Dead end. Archived. |
| Runs 8-9: First H100 | 2 | PonderNet collapsed; iterative refinement helps (9.8% benefit) |
| Runs 10-11: 8×H100 Round 1 | 2 | Matrix T=1 wins 175× over vector T=1; vector T=8 wins on iteration |
| Run 12: Round 2 (MultiProbeHead) | 1 | **Rank enrichment discovered: 5.02 → 6.12 — novel finding** |
| Runs 13-14: LoopFormer comparison | 2 | We lose 2× at matched FLOPs — **[CORRECTION 2026-07-02] not actually matched; see "Honest negative results" above** |
| Run 15: Optimized matrix thinker | 1 | Marginal speedup, marginal quality drop |
| Runs 16-17: d=16 experiments | 2 | First byte-level matrix model, BPB 1.91 |
| Run 18: Critical ablation (not param-matched) | 1 | Flat 24M vs Matrix 2.4M — unfair comparison |
| Run 19: Byte-level d=16 | 1 | 218K params, BPB 3.560, 33% thinking |
| Runs 20-21: 3D attention | 2 | Solidification confirmed dead end |
| Run 22: Param-matched ablation (still not clean) | 1 | 5.66M flat vs 2.55M matrix — still mismatched |
| Runs 23-24: Thought interleaving | 2 | BPB 3.535 (no thoughts) vs 3.538 (with thoughts) — depth wins |
| Run 25: Full sweep (5 configs) | 5 | "Just add layers" beats every thought config |

Full details with exact numbers: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

---

## What the Field Has Shown Us

A multi-agent research session (April 9, 2026) investigated 11 topics in parallel. Key findings that bear on the project:

### Continuous reasoning research is maturing

- **CODI** (Feb 2025, EMNLP 2025, arXiv 2502.21074) hits 43.7% on GSM8K with simpler training than COCONUT (34.1%). Joint teacher-student via shared weights, L1 distillation at the `:` token across all layers. Complete public code at github.com/zhenyi4/codi. **Strongest baseline for our matrix-CODI experiment.**

- **CoT2** (May 2025, ICLR 2026, arXiv 2505.23648) proposes parallelism-vs-dimension theoretical framework. Argues parallelism in continuous reasoning scales with embedding dimension, but only proves this via existence constructions and accuracy curves. **Never measures rank or any structural property.** Closest published prior art for the rank-superposition argument.

- **The Illusion of Superposition** (2026, arXiv 2604.06374) is a rebuttal: argues fine-tuned COCONUT reaches 96.6% without latent tokens, entity-probes show no stepwise computation. Superposition is contested. **The matrix-CODI experiment can adjudicate this dispute.**

- **Reasoning by Superposition** (May 2025, arXiv 2505.12514) hand-designs continuous thoughts as `t_c = (1/√|V_c|) Σ u_v` over BFS frontier vertices. Rank equals frontier size by construction, but the paper never measures whether trained models actually realize this rank.

### JEPA is gaining momentum

- LeCun left Meta in February 2026 and raised ~$1B for AMI Labs to pursue JEPA exclusively.
- LeJEPA (Nov 2025) finally solved collapse via SIGReg — single hyperparameter, no stop-gradient, no EMA. 20 lines of code, drops in.
- LLM-JEPA (Sep 2025) added JEPA aux losses to standard LLMs with statistically significant gains on GSM8K and other benchmarks.
- V-JEPA 2, V-JEPA 2.1, VL-JEPA — scaled video and vision-language JEPA.
- **No byte-level JEPA exists** as of April 2026. Confirmed gap.

### Pure-sensor models match language-supervised models at scale

- **DINOv3** (Aug 2025, ViT-7B, no text, 1.7B images): first SSL model to beat weakly-supervised peers across the board.
- **Web-SSL** (Apr 2025, Meta): controlled comparison shows CLIP's prior advantage was data, not language.
- **Object Binding paper** (Oct 2025): object binding emerges in DINOv2/MAE but NOT in supervised ViTs. Self-supervision specifically.

### Discrete vocabularies have lost the text race

- **Meta abandoned Chameleon** (VQ multimodal) for **BLT** (byte-level, tokenizer-free). Most important data point against learned vocabularies for text.
- VQ won vision/video, lost text, contested for audio.
- Emu3.5 (34.1B, Oct 2025) is the largest native discrete-token multimodal model, but uses separate text BPE + visual VQ codebooks.

### Structure-vs-scale debate is unresolved at language-modeling scale

- **HELM** (May 2025, NeurIPS 2025): first billion-parameter fully hyperbolic LLM. Reports +0.5-2.3 points over Euclidean baselines on MMLU/ARC. The architecture commits hyperbolic everywhere — no half measures. **Existence proof that pervasive structured architecture works at scale.**
- **Brehmer et al.** (TMLR 2025): often misread as pro-scaling. Actually shows equivariant models maintain ~2x compute-efficiency advantage at every budget tested across 10^16-10^19 FLOPs.
- The pattern: structure wins when pervasive (HELM), loses when bolted on. Half-commitments do not work.

### Neuroscience case for non-linguistic cognition is mainstream

- **Fedorenko et al. 2024 *Nature*:** "Language is primarily a tool for communication rather than thought." fMRI dissociates language network from reasoning, math, theory of mind.
- Zaslavsky 2018 *PNAS*: languages near information bottleneck optimum for **communication** between brains. Channel-capacity argument: language is optimized for ~50 bit/s inter-brain channel; machines have no such constraint.
- Grid cells, sparse coding, predictive coding: well-established neural primitives that compute below language.

### Critical gap nobody has filled

**Nobody has measured rank as a structural correlate of reasoning capacity in continuous-reasoning models.** Three lines of work have come within one step of this question and none have taken it. The theorists define superposition but never measure it. The dimensional-collapse work measures rank but not in reasoning models. The interpretability folks study reasoning but use feature dictionaries, not geometric rank. **This is the unfilled gap the matrix-CODI experiment targets.**

---

## The Narrowed Hypothesis — STATUS (April 2026)

After the matrix-CODI experiments (Rounds 1-9 + positive-control Round PC) the
narrowed hypothesis has been tested and the relevant sub-claims have failed:

- **H1 (correlation rank ↔ reasoning paths):** FAILED. Four flat rank-k curves
  (Rounds 1, 2, 3, 6). 3-seed replication of flatten readout: accuracy tight
  at 81.5 ± 1.2pp but Z_rank varies by 3× (seeds 42 → rank 4, 7 → rank 12, 1337
  → rank 13). Rank is decoupled from accuracy.
- **H2 (capacity bound):** Not meaningfully testable given H1 failure. Rank is
  not being used, so there is no capacity bound to observe.
- **H3 (causation via rank truncation):** FAILED. Rank-k truncation has no
  effect on accuracy for k ≥ 1 in all tested configurations.

The bolt-on matrix-CODI configuration does not use rank structure to encode
reasoning. Mechanism: the flatten-then-project readout has a constant Jacobian
in Z, so the gradient cannot distinguish rank-1 from full-rank Z during
training. This has been verified with a positive control — a nonlinear-in-Z
readout (bilinear+GELU) that breaks the constant-Jacobian property.

Positive-control result: **bilinear+GELU also produces a flat rank-k curve**
(Spearman r = -0.13, p = 0.14). The failure is deeper than readout linearity
alone; the CODI distillation objective itself produces rank-indifferent
gradients regardless of how Z is consumed.

**Publication status:** workshop paper written for ICML MI Workshop 2026
(deadline May 8) documenting the negative result + diagnosis + positive-control
falsification test — **submitted and accepted** (see "Workshop paper outcome"
below and `matrix-thinking/submissions/icml-mi-workshop-2026/`). The
now-superseded writing brief and results-consolidation scratch that fed the
paper are archived at `archive/matrix-thinking-workshop-era/PAPER_WRITER_BRIEF.md`
and `archive/matrix-thinking-workshop-era/PAPER_RESULTS_SUMMARY.md`.

**What the failure does NOT imply:** the broader matrix-thinking thesis is NOT
decided by these experiments. All experiments here bolt a matrix bottleneck
onto a vector-pretrained model with a vector teacher signal (CODI distillation
from a vector-output teacher). The failure modes are specific to this bolt-on
setup. A matrix-native architecture trained end-to-end on a task that rewards
rank-K structure has not been tested. That is Chapter 2.

**Workshop paper outcome:** *The Gradient Does Not See Rank* was submitted to
ICML MI Workshop 2026 and **accepted**. Position-decomposition follow-up work
(`rank_aware_v1`, 2026-04-29) independently reproduced the mechanism from a
different angle: even on a constructed multi-target task, matrix-CODI composes
via **position** (one rank-1 value per latent slot), not within-position
spectral rank — forcing Z to rank-1 throughout training did not hurt accuracy.
Consistent finding, two routes: bolt-on matrix-CODI never uses rank.

---

## Chapter 2 — STATUS (2026-07-04): CONFIRMED through real data; six programs closed (exactness-mechanism study, Wave 0/1/F/geo3, now closed)

Chapter 2 ran and gave the field's first positive result for matrix-native
rank: **when a task provably requires `rank(Z) ≥ K`, gradient descent trained
from scratch both develops effective rank ≈ K and makes rank causally
necessary.** This resolves the open question left by the workshop paper — the
earlier rank-blindness was **task-specific (ProsQA was rank-1-solvable), not a
property of the gradient.**

**What got built and why the original Chapter 2 plan changed:** the original
synthetic-task plan (Task A, K-parallel-entity-tracking with a single-entity
query) was killed by a design-gauntlet attack *before any GPU ran*: in a
full-attention model, "hold K items" is trivially satisfiable via K
*positions* at rank-1 each (the same position-decomposition escape
`rank_aware_v1` found empirically), and a rank-1 matrix `Z=u⊗v₀` is not
low-capacity — its free vector side already holds `d` items. So the naive
K≈P crossover prediction was mathematically wrong (the real threshold would
have been K≈P·d, i.e. a flat curve everywhere testable). The gauntlet
produced **Task D** instead: a from-scratch matrix-native transformer trained
under a **hard single-Z bottleneck** on K key→value bindings, with a
**provable** `rank(Z) ≥ K` lower bound for exact continuous recovery (proof:
stacking K independent keys/values, `rank(V) = rank(Z·K_mat) ≤ rank(Z)`).
Critical design decision: the readout must be the **pinned linear unbind**
`Z·key`, scored by absolute cosine — **never argmax over a codebook** — because
under argmax decoding a rank-1 matrix can recover ≈d bindings (Nichani, Lee &
Bietti, ICLR 2025, arXiv:2412.06538), which would silently collapse the
provable bound. Full spec, proof, and audit trail:
`matrix-thinking/chapter2/TASK_D_PREREGISTRATION.md`.

**Results (d=8, d=16 — the confirmed regime):** effective rank tracks K almost
exactly (d=16: K=1→2.4, 4→4.7, 8→8.2, 12→11.8, 16→15.1; Spearman ρ=1.0). The
causal test (train-time `force_rank_k`, the primary test — not post-hoc
truncation, which was uninformative in the CODI work) shows a sharp step at
k≈K: at d=8,K=4, force-rank ≤3 gives 0.0 recovery, force-rank=4 gives 0.97.
Full write-up with citations: `matrix-thinking/chapter2/TASK_D_WRITEUP.md`.

**Honest limitation, corrected (2026-07-03) — the d≥32 "trainability
frontier" was a step-budget artifact; the real frontier is exactness:**
Stage 0 (`matrix-thinking/chapter2/STAGE0_DESIGN.md` §12-14, closed
2026-07-03) ran a full diagnostic and found the original d≥32 wall
(effective rank ≈1, recovery ≈0 at Task D's 8K-step budget) is
substantially a step-budget artifact — every d=32 baseline seed tested
(17/17 across Wave 0, an extended-budget arm, and a 100K-step probe)
transitions reliably, onset 6-16K steps, and effective rank recruits to K
once budget suffices (final eff. rank 3.7/7.3-7.8/14.6 at K=4/8/16 — Task
D's M1 pattern holds at d=32). But even at 100K steps (10x Task D's
original budget), with trajectories confirmed flat/plateaued rather than
climbing, the formal pass bar (`recovered_frac@0.9 > 0.7`) still FAILS —
best observed is 0.65 (K=8), a genuine converged plateau (cos 0.83-0.91),
not undertraining. Contrast: d=16 reaches genuinely exact solutions
(`recovered_frac@0.9` = 1.00 at every tested hop including h=21, Task E's
40K round). So the honest frontier is not trainability (transitions are
reliable) — it's exactness, and it degrades with `d`. *Why* the d≥32
write plateaus sub-exact rather than reaching 1.0 is open, named Stage
0.5, and is explicitly NOT answered by Stage 0. Full data and
hypothesis-by-hypothesis verdicts: `STAGE0_DESIGN.md` §12-14.

**Chapter 2.5 — Task E (reasoning transfer, launched 2026-07-01, running on
Brev 8×H100):** Task D is associative memory (one lookup, no composition) —
the open question is whether the causally-necessary rank-K matrix *composes*
correctly under repeated self-application (`Zʰ`) at hop-depths never seen in
training, i.e. whether SGD's Z has genuine operator/eigenstructure, not just a
rank-sufficient lookup table. Design + full attack trail:
`matrix-thinking/chapter2/NEXT_EXPERIMENT_DESIGN.md`. Two build-time FATALs
were caught and fixed by the audit gauntlet before any compute ran: (1) the
injectivity check on the key→value mapping had a `-1` tolerance that couldn't
detect a single merge — "K edges" silently stopped implying rank≥K (the same
miscounting trap that killed MNNS); (2) the permutation-based hop-depth
generator sampled a *general* random permutation rather than a single
Hamiltonian K-cycle, so short cycles made "held-out" hop depths periodically
collapse into in-distribution or trivial queries (measured: 100% collapse at
K=4, the `H_extra=8` probe was 100% dead at K=8). Both fixed; re-audited clean;
smoke-passed on the H100. Primary decision metric M3_E: held-out-hop recovery
vs. the C_MLP shortcut floor vs. the analytic ideal-Z ceiling. CONFIRM = matrix
thinking's compositional-reasoning premise survives its first reasoning test;
FALSIFY = rank is causally necessary for recall but functionally inert for
reasoning (still a clean, publishable negative).

**Sequencing decided:** if Task E CONFIRMs → real-data reasoning transfer
(matrix-native on OpenR1-Math, already tokenized on the H100 side) is next,
sequenced *after* Task E specifically to avoid reintroducing every confound
Task D was built to eliminate. If it FALSIFIES → publish as a companion
negative to the workshop paper.

**Real-data link — Wave 1 CLOSED, CONFIRMED on all three legs (2026-07-03):**
a parallel thread (`matrix-thinking/DELTANET_REALDATA_DESIGN.md`) asks the
same rank-necessity question as Chapter 2, but on a production fast-weight
kernel (DeltaNet's `chunk_delta_rule`) with real GPT-2-tokenized text
instead of a bespoke encoder or constructed vector grammar. Its Wave 0
originally value-collapsed 10/10 seeds (caught clean at zero premise-valid
checkpoints, never reported as a finding); a mini-audit traced it to a
hop-index gather bug, and the fix + a pre-registered anti-collapse NCE loss
produced a rerun that is 10/10 collapse-free (K=16: rec@0.9 0.996–0.999,
entity-subspace rank 15.6–15.7/16). Wave A then found a graded K-exactness
frontier at fixed d_state=64 (K=8 near-exact through several held-out hops,
K=16 partial, K=24/32 collapsed beyond h=1), rank recruited 94–99% of
target at every K. **Causal close (2026-07-03, §17):** Bprobe's train-time
force-rank arm reproduced the train-time-forcing-breaks-SGD failure a
**third** time (fr16 at `k=K=16`, a provable no-op, collapsed 3/3 —
entity-subspace rank 9.85–10.41 vs. the unconstrained arm's own 15.6–15.7),
so the full Wave B grid was correctly judged moot and never launched
(mirroring `DELTANET_CAUSAL_RANK_DESIGN.md`'s identical decision). The
pre-registered fallback — eval-time SVD truncation of the archived
`Z_dump` states (`--save-z` is on by default in this harness, so no
retrain was needed) — closed the causal question instead: **CONFIRMED,
rank causally load-bearing at K∈{8,16,24,32}** (a hard ceiling reached
exactly at k=K, never before or after, at every cell) **but graded across
a multi-rank window, not the synthetic design's razor cliff at
k=K−1→K** — the pre-registered caveat (trained rank sits slightly under K;
keys are non-orthonormal, unlike the synthetic construction) landing
exactly as predicted, not a hedge that never fired. This is the project's
first demonstration of genuine, causally-verified rank-K relational
binding in a production architecture on real tokenized surface forms.
Full arc, exact numbers, and the closing verdict:
`matrix-thinking/DELTANET_REALDATA_DESIGN.md` §14–§17.

**Wave 2 (Waves C+D, CLOSED 2026-07-04, §19):** real-corpus LM pretrain
(OpenR1-Math vs. WikiText, d_state=64, seeds{0,1,2}) plus inference-time
rank-truncation grid, 12/12 cells. Headline: reasoning-dense text
(OpenR1-Math) is measurably more truncation-damage-sensitive than
narrative text (WikiText) at low-to-moderate k (8/16/24), converging to
the same noise floor by k≈48 of d_state=64 — consistent in direction
across every cell tested, including a within-token-class check that
partially rules out a symbol-density confound. Counter-intuitive finding:
layer-0 effective rank *falls* (not rises) as training proceeds in BOTH
corpora (Pearson r vs. val-loss trajectory: +0.92 openr1, +0.91 wikitext —
both decreasing together), the opposite of the "SGD recruits more rank as
it learns" intuition from the controlled causal-rank chain; read as a
general LM training dynamic, not reasoning-specific, since it doesn't
replicate cleanly at layer 1. Wave 2 closes the DeltaNet real-data
program's record — no further waves are pre-registered beyond §7's
manifest Reserve row.

**Exactness mechanism study (living, NOT one of the five closed
programs — the active follow-on) — why does real-text composition fall
short of the synthetic razor cliff, and can the gap be closed?**
`matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md`. Wave 0/1 (CLOSED
2026-07-04): effective-key geometry is the whole attribution story — three
independent arms (frozen orthonormal, GPT-2 span, gram-matched) all
converge to the *same* non-orthonormal write-geometry attractor regardless
of input geometry (raw embedding geometry is causally irrelevant), and a
surgical orthonormal-key pin (i-strong) achieves **PERFECT K=32
composition** (1.00/1.00/1.00 recovery at h=1/2/3) — proving the exact
solution is architecturally reachable; SGD just doesn't find it under the
current objective. Wave F (soft fix attempt, CLOSED 2026-07-04): an
orthogonality penalty and ZCA whitening both move recovery in the right
direction but land 20-25× short of the pre-registered bars (K=32 h=4:
0.014-0.026 vs. bar ≥0.5) — an honest negative that motivated the next,
structural attempt. **K=48 rider (CLOSED):** the frontier extends past
d/2 (gram deviation keeps growing, composition gone by h≥2); the
i-strong pin's own dimensional guard correctly refuses K=48 (train+
held-out identity vectors exceed d_state=64), fencing that boundary as
designed. **F-geo-3 (differentiable per-episode key orthogonalization, the
trainable version of i-strong) — fix wave CLOSED, program CLOSED.** A
first 6-cell Wave 1 batch (K∈{16,32}×3 seeds, `geo3_n_iter=12`,
20,000/20,000 steps each) landed 2026-07-04: **K=16 clears the
pre-registered minimum-publishable bar on all 3 seeds** (h=4 recovery
0.95-1.00 vs. a bar of ≥0.8, baseline was 0.42-0.47), but **K=32,
despite a ~50× headline improvement (h=4 recovery 0.39-0.50 vs.
baseline 0.009), failed the admissibility criterion on all 3 seeds** (a
numerical eigh fallback triggered on a small fraction of steps). The
follow-on escalation (K=32 ×3 seeds at `geo3_n_iter=20`) closed the
admissibility gap cleanly — **0/3 → 3/3 admissible, zero fallback steps
at any seed — and the behavioral numbers did not move** (largest
per-seed delta 0.0042), confirming the fallback steps never degraded
training. **Final verdict: K=16 bar HIT 3/3 (h=4 0.98 mean vs. bar
≥0.8); K=32 improves ~43-56× over baseline (mean ≈48×) but narrowly
misses its ≥0.5 headline bar on the mean (0.4368)**, with the residual
attributed to the pre-registered **outcome F** (stable-not-just-
orthogonal geometry: measured cross-episode key drift 0.90-0.94, HIGH
band, predicted and confirmed via the §14.6 gating diagnostic before
the wave ran, not fit after the fact) — a named mechanism, not an
unexplained shortfall. h=1 no-sacrifice holds at both K (K=32 h=1
actually exceeds baseline by +0.21); h=21 literal-depth collapse is
unchanged (orthogonalization fixes write interference, not iteration
compounding). Full verdict in `EXPERIMENT_LOG.md` ("F-geo-3 WAVE
VERDICT" + "F-geo-3 escalation VERDICT", both 2026-07-04) and the full
per-cell write-up in `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md`
§16. **This closes the exactness-mechanism study (Wave 0/1/F/geo3) in
its entirety** — the next step is a stability-targeted follow-on
design (named as a direction in §14.8, not yet designed) or the
already-gated Chapter 3 scale-up (see "Then" below), not a further
iteration inside this design (no fix-fishing, per the anti-Goodhart
rule this program held to throughout).

---

## Path Forward (updated 2026-07-04)

### Now — Saturation campaign (2-month uptime-metered window)

**The grant meters UPTIME, not utilization**: the box bills while RUNNING and
cannot be stopped (`brev stop` unsupported on this instance type; only
delete). The user confirmed 2026-07-03: hardware is granted for two months,
use-it-or-lose-it. Effective budget ≈ 192 GPU-h/day × window ≈ 10,000+ GPU-h,
not the 1.6k previously assumed. **Strategy: keep the box saturated with
audited experiments for the whole window; idle time is the only true waste.**
Discipline unchanged: no un-audited work launches just to fill GPUs — the
pipeline (design → adversarial attack → build → independent audit → bounded
waves) has enough parallel workstreams to stay ahead of the hardware.

**Campaign ledger (2026-07-01 → 07-04): five programs designed, attacked,
built, audited, run, and closed** — Task D/E (bespoke synthetic causal rank +
composition), Stage 0 (d-frontier), DeltaNet synthetic (production
architecture causal rank), Stage G (matrix-vs-vector gap mechanism named),
DeltaNet real-data (rank-K binding + composition on real tokenized text,
causal close via eval-truncation, plus a closed Wave 2 real-corpus-LM
follow-on). Two threads opened by those closures were active as of 2026-07-04
early — the exactness mechanism study (why real-text composition
undershoots the synthetic razor cliff) and Stage G's gated H_e
task-swap check (below). **The exactness mechanism study is now fully
CLOSED** (Wave 0/1/F/geo3, including the geo3 escalation, see above);
**Stage G's H_e check is now also CLOSED** on its primary question (below;
one small anomaly left open, not a blocker). ~600+ GPU-h total
across the campaign. Full verdicts in EXPERIMENT_LOG.md (dated
2026-07-01..04, table of contents at the start of that date range) and the
five design docs (`DELTANET_REALDATA_DESIGN.md`,
`DELTANET_CAUSAL_RANK_DESIGN.md`, `STAGE_G_DESIGN.md`,
`chapter2/STAGE0_DESIGN.md`, `DELTANET_RD_EXACTNESS_DESIGN.md`). Workshop
paper drafted at `matrix-thinking/submissions/neurips-ws-2026/` (awaiting
user review: author block, venue, figures, title, appendix).

**Other closures (2026-07-04):**
- **Stage-G H_e Wave C — CLOSED on its primary question.**
20K showed neither arm composing (flagged then as not-yet-triaged); 40K
calibration (seed 0) resolved that as a genuine late-transition budget
effect, not a bug — vector fully composes at 40K (h1/h2/h3 chance-adjusted
1.0/1.0/1.0) while matrix does not (1.0/0.027/0.013), firing the
pre-registered decision rule for the full 6-cell manifest (4 more cells:
matrix baseline s1, matrix `h_b_factored_r4` s0+s1 — the H_b Wave-B
projection winner — vector baseline s1). All 6 cells complete, no
timeouts/NaN/crashes, 27.5 GPU-h total on GPU 7 alone (no contention with
GPUs 0-6's concurrent waves). **Verdict: `h_b_factored_r4` does NOT rescue
matrix composition** (`recovered_frac` on the seed-stable h=3 metric: +0.5%
seed 0, −0.6% seed 1 — both ≈0, an order of magnitude below the `≥0.5`
dominant-site bar — despite running at 2.69× the baseline's params). **The
vector-composes/matrix-cannot inversion is seed-stable at hop-depth 3**
(4/4 matrix-family cells flat at chance across the full 40K trajectory;
both vector seeds clear matrix by 30+ points, seed 1 still climbing at
cutoff so 0.661 is a lower bound). **Held-out hop generalization (h=4/5/7)
is at chance for ALL 6 cells, matrix and vector alike** — a separate,
uniformly negative axis. **Open, unresolved anomaly (not a blocker):**
matrix baseline's hop-depth-2 result is NOT seed-stable — seed 0 stays flat
at chance through 40K, seed 1 undergoes a sharp, clean phase transition to
full composition at steps 18K–22K with no proposed mechanism; any h=2-
specific claim (either direction) is unsupported pending more seeds. Full
table, verdict derivation, and the anomaly write-up:
`EXPERIMENT_LOG.md`, "Stage-G H_e 40K MANIFEST VERDICT" (2026-07-04);
design-doc results section: `STAGE_G_DESIGN.md` §15. Archive:
`experiment-runs/2026-07-05_stageg_he40k/` + SSD mirror.
- **ReserveMH (DeltaNet multi-head causal-rank generality) — CLOSED, not
  in flight**: every attention head independently recruits full rank K=32
  at H∈{2,4}; the H=1 qualifier on the synthetic causal-rank claim is
  lifted (`EXPERIMENT_LOG.md`, 2026-07-04 early).
- The deeper-hop training probe and the RD Wave 2 instrumented-LM builder
  (both listed as in-flight as of 2026-07-03) are **now CLOSED** — see the
  DeltaNet real-data paragraphs above and `EXPERIMENT_LOG.md`'s
  "deephop program CLOSED" and "Wave 2 (Waves C+D) results" entries.
- **F-geo-3 escalation (listed as in-flight as of 2026-07-04 early) is now
  CLOSED**, and with it the entire exactness-mechanism study (Wave
  0/1/F/geo3): K=32 admissibility fixed 0/3→3/3 with zero behavioral
  change; K=16 bar HIT, K=32 bar narrowly missed and attributed to the
  pre-registered outcome F. See the exactness-mechanism paragraph above,
  `EXPERIMENT_LOG.md`'s "F-geo-3 escalation VERDICT" entry, and
  `DELTANET_RD_EXACTNESS_DESIGN.md` §16.
- **KEY-ANCHORING wave (the outcome-F follow-on) — RUN, verdict DESCRIPTIVE,
  not yet the confirmed headline (2026-07-05, 10.98 GPU-h, `youthful-indigo-
  turkey`).** This supersedes the earlier "outcome F follow-on... not yet
  designed" note — it is now designed (`KEY_ANCHORING_DESIGN.md`, Rev 5,
  5 attack/verify rounds), built, and run. **Behavioral result is strong:**
  candidate (d) (learned-λ anchor blend) K=32 h=4 `rec@0.9` clears the ≥0.5
  bar in 3/3 seeds (0.559/0.615/0.665, mean 0.613 vs. this wave's own
  fresh-reference 0.4105 — a +0.20 absolute lift), λ lands interior (not
  pin-rediscovery, not unrecruited) in 6/6 seeds at both K, K=16 has no
  regression, and value-Gram deviation drops to roughly half the reference's
  own (a disclosed bonus finding). **But the wave cannot be assigned its
  pre-registered Outcome A/A′/A″/B/C** (`KEY_ANCHORING_DESIGN.md` §3.5):
  `keyanchor_wave1_manifest()` never wired `--drift-probe` into the admitted
  candidate-(d)/(c) cells (only the reference arms got it), so item 5
  (pre-NS blend drift), item 6 at final admission (table conditioning), and
  §3.7's `engaged_frac` were never measured on the actual trained runs —
  and the standalone tool that would have supplied them
  (`keyanchor_drift_diagnostic.py`) crashed on its first invocation (a
  `log_every` self-inconsistency with the harness's own registered logging
  cadence) and was never re-run, silently also skipping Gate 1's pre-spend
  check. Verdict: **DESCRIPTIVE tier for the mechanistic claim, pending a
  cheap instrumentation-only follow-up** — not a hypothesis failure, a
  build gap caught by an independent verdict pass. Full tables + gap
  narrative: `EXPERIMENT_LOG.md`'s "KEY-ANCHORING WAVE VERDICT" entry (this
  date) and `KEY_ANCHORING_DESIGN.md` §9. Archive:
  `experiment-runs/2026-07-05_keyanchor_wave/` + SSD mirror.
  **CONFIRMATORY WAVE RUN (2026-07-05, same day) — gap closed, literal
  outcome is C, not A.** `--wave keyanchor-confirm` re-ran candidate (d)
  K=32 seeds {0,1,2} + a K=16 spot check with `drift_probe=True` wired
  into the real 20,000-step training loop (fixes committed in `5963616`).
  Item 5 (pre-NS drift) and h4 both clear their own bars decisively (3/3,
  plus K=16) and λ is interior 4/4 — but §3.7's per-entity `engaged_frac`
  (the metric built specifically to catch a pooled statistic masking a
  disengaged majority) reads **under 14% in every leg** (13.08/3.74/4.67%
  at K=32, 11.21% at K=16), and §3.7's own text routes any `engaged_frac
  <50%` to **Outcome C ("mechanism not engaged... not an admissible test
  of the hypothesis either way") regardless of aggregate drift**. Seed 1
  also independently fails item 6a (2/3, not 3/3, admissible under the
  full items-1–6 stack at K=32). **Tier: the §1 interaction hypothesis is
  now measured-and-not-admissible (was: unmeasured/UNASSIGNABLE); the h4
  behavioral result stays DESCRIPTIVE**, unchanged in kind, changed in
  reason (a measured mechanistic null under a real behavioral positive,
  not a missing measurement). Verification note: the confirm cells reuse
  Wave-1's own seed integers (not 3 new independent seeds) — h4/λ land
  within 0.0001–0.001 of Wave-1's own per-seed numbers (GPU nondeterminism
  at a fixed seed, not fresh replication; do not double-count as 6
  independent seeds). A `Rev 6` λ-scaled `engaged_frac` threshold
  re-derivation is registered as a stub (not scheduled, no retroactive
  rescoring authorized — needs its own attack round first). Full tables:
  `EXPERIMENT_LOG.md`'s "KEY-ANCHORING CONFIRMATORY WAVE VERDICT" entry
  and `KEY_ANCHORING_DESIGN.md` §9.6. Archive:
  `experiment-runs/2026-07-05_keyanchor_confirm/` + SSD mirror.
- **SCALE-TRANSFER Track B — measurement waves complete, geo3-in-LM
  DOUBLE-BARRED, selectivity main effects read (2026-07-05, ≈1.5–2 GPU-h
  actual, GPU 7 only).** `TRACKB_REDESIGN.md` Rev 3's Wave −1 stability
  smoke measured `skip_rate=0.6319` (103/163 steps) against the `≤0.01`
  bar with a PROBATIVE positive control (196/326 calls ≥6 duplicated
  selected rows, max 32/32) — a **second independent barrier** to
  geo3-in-LM after the original β-uniformity no-launch (Gini 0.099). Cells
  3/4 and Wave 2 REFUSED; the interaction/headline bars are UNCOMPUTABLE,
  not merely unevaluated. Readout scoped to selectivity main effects only
  (Cells 1/2/2R/comparator, Wave 1's real 18-cell manifest — candidate 2/
  entmax never promoted past Wave −1, traced to the launcher's own
  `--surviving-mechanisms` default): val-loss +5% bar FAILS on openr1 for
  all three arms (candidate 1 and its STE comparator fail together →
  hard selectivity implicated, not an STE artifact) and PASSES on
  wikitext for all three; same-instrument Gram-deviation shows openr1
  Cell-2-vs-2R disjoint (targeting matters) but wikitext overlapping
  (INCONCLUSIVE, write-concentration not targeting) — **combined headline
  verdict INCONCLUSIVE** per the registered same-outcome-both-corpora rule.
  One positional-concentration band breach (Cell 2, openr1 s0 L1, ≈2% over);
  BUDGET-PARTIAL stamping is uncomputable from the archived data (a real
  instrumentation gap, filed not assumed clean — see the [LEARN] block).
  Full table/verdicts: `EXPERIMENT_LOG.md` "SCALE-TRANSFER Track B
  measurement waves" entry; `TRACKB_REDESIGN.md` §14. Archive:
  `experiment-runs/2026-07-05_trackb_wave/` + SSD mirror.
- **SCALE-TRANSFER Track D Phase 1 (pretrained-model measurement) — CLOSED
  (2026-07-04, ≈0.9 GPU-h, GPU 6 only).** The write-geometry signature
  exists in production fixed-state models (RWKV-7 1.5B — the design's 2.9B
  pick is a broken HF conversion on this stack, documented substitution —
  and Falcon-Mamba-7B) and is far MORE extreme than our 14M attractor
  (RWKV-7 raw Gram dev ≈10.9 at K=16/d=64 vs our 0.6–4.4 band; ≈70% of the
  collapse ceiling). BUT the registered no-fixed-state negative control
  (Qwen2.5-1.5B, resolving `SCALE_TRANSFER_DESIGN.md` §12 Q4) shows the
  same magnitude — the signature is dominated by generic trained-LM key
  anisotropy (massive activations, Sun et al. 2024) and is NOT attributable
  to the delta-rule write mechanism at this measurement tier. Phase 2
  (graft) premise weakened; it stays unauthorized. Full table + caveats:
  `SCALE_TRANSFER_DESIGN.md` §6.8; archive
  `experiment-runs/2026-07-04_track_d/` + SSD.
- **SCALE-TRANSFER Track C Wave 1 (rung-1) harvest (2026-07-04, ≈0.08 GPU-h
  probe, GPU 0 only) — geometry leg only, persists 14M→98M on this 2-point
  read** (control 14M raw gram-dev 21.93 vs. rung-1 98M 27.82, both well
  above the K=64/d=64 random anchor 7.94 and below collapse 63.50; gets
  slightly worse, not better, with scale); frontier-probe transplant and
  fix-effect-at-scale (§5.5 items 2/3) NOT run. Data-mix axis (MAJOR-5)
  stays an open gap — Wave C's checkpoints are no longer on the box, so no
  same-instrument mix-vs-clean reading exists yet (proxy: val loss +0.28
  nats both corpora from mixing; whole-state rank flat). Full tables/caveats:
  `SCALE_TRANSFER_DESIGN.md` §5.9; archive
  `experiment-runs/2026-07-04_trackc_rung1/` + SSD.
- **SCALE-TRANSFER Track C Wave 2 (rung-2, 392M) + mixcontrol HARVESTED
  2026-07-05 (probe ≈1.04 GPU-h, GPU 7 only) — attractor WORSENS
  monotonically on the 3-point read 14M→98M→392M.** All 6 rung-2 runs
  (391.87M params, 91,552 steps ≈1.5B tok/run, ext mixes, 128.3 GPU-h
  measured) + 6 mixcontrol runs completed clean (0 skips/NaN/timeouts).
  Anchor-normalized gram-dev span-frac: 0.252 (14M) → 0.358 (98M) → **0.389
  (392M)** (raw 21.93 → 27.82 → 28.10 on the corpus-matched archived-4
  subset; rung-2 anchors 5.61/63.50 at K=64,d=128). Trajectory (11 ckpts,
  seed 0): fast drop 31.6→29.8 by step 11k, then plateaus ≈5× above random
  — no dissolution within budget. **Mixcontrol closes §5.9's mix-axis gap
  at 14M: ext-mix geometry 21.74 vs orig-mix 21.93 (Δ −0.19, seed noise) —
  mixes move val loss (+0.06 nats), not geometry**, supporting scale as the
  rung-1→rung-2 driver (98M interaction closes with wave-1ext, GPU 6).
  Geometry leg only; raw-only probe; rung-3 now un-gated per Rev 2.2
  (launches at 1.5B tok/run token-matched to rung-2). Full tables:
  `SCALE_TRANSFER_DESIGN.md` §5.10; archive
  `experiment-runs/2026-07-05_trackc_rung2/` + SSD (30MB training JSONs
  SSD-only). Launch artifacts: `experiment-runs/2026-07-04_trackc_rung2/`.

**Scale-up doctrine (user directive 2026-07-03):** deploy plenty of
adversarial design/attack teams and independent code audits on everything;
max out ALL levers — data (more corpora beyond the 43.7M-token OpenR1 slice +
WikiText already on box), memory (80GB/GPU is barely touched by probe-scale
models; scale d_state/batch/model where the question warrants), compute
(8×H100 continuous). Sonnet subs do the work; orchestrator stays top-level
and verifies key claims itself.

**Constructive-demonstration mandate (user directive 2026-07-03):** don't
just map failures — demonstrate positive capability. Every attribution
program carries a pre-registered FIX/demo wave as its deliverable (the
exactness program's Wave F: once the mechanism is named, demonstrate the
intervention that moves the K-frontier — headline target: K=32 held-out-hop
recovery from ≈0.05 floor to ≥0.5). Findings docs and papers frame the
demonstrated path forward, with failure maps as supporting evidence.

### Then — Chapter 3: matrix-native on real data (Task E gate PASSED; exactness-mechanism study now CLOSED — scale program not yet designed)

Byte-level input, matrix tokens throughout, multi-modal training. A dedicated
research pass (`research/bytes-vs-tokens-matrix-native-june2026.md`, cross-
checked by an external deep-research pass) settled a design question that was
previously just an assumption: **hold the tokenizer standard (GPT-2/BPE) for
the primary scaled matrix-native experiment; treat byte-level input as a
separate, high-priority follow-on ablation, not bundled with the matrix
change.** Reasoning: (1) bundling two unproven architectural changes (matrix
representation + byte input) makes any result uninterpretable — this is the
same discipline the byte-level LM field uses on itself (BLT holds architecture
fixed and varies only the tokenizer); (2) byte-level models do not currently
beat BPE on math/reasoning benchmarks at matched compute; (3) matrix-native-on-
BPE is *already* a novel, unoccupied combination, so there's no novelty
pressure to rush bytes in. The counter-evidence that keeps bytes a
**high-priority**, not "someday," follow-on: 2025-2026 literature (Tokenization
Counts, arXiv:2402.14903; BitTokens, arXiv:2510.06824) shows number/token
granularity causally changes arithmetic reasoning, so tokenization is a held-
fixed confound with an **open interaction**, not a proven-orthogonal one — do
not claim it as inert in any write-up. When the byte ablation is designed, use
a TPR-style outer-product byte-window embedding (Smolensky 1990; TP-Transformer,
arXiv:1910.06611), not a naive tokenizer swap, since a structured byte-window
construction is the principled way rank could plausibly interact with byte
granularity.

### Backup — Pivot direction

If Task E falsifies, publish that as the decisive result and reassess before
committing further compute to matrix-thinking. Candidate pivots (unordered):
- Byte-level JEPA with LeJEPA SIGReg (Thread 1 of the original thesis, no
  matrix commitment required)
- Continuous-reasoning extensions without matrix structure (e.g. SIM-CoT-style
  step-level supervision done properly)
- Other mech-interp directions surfaced by the workshop paper reviews

---

## Hardware

- **Brev 8×H100 80GB (active, 2026-07-01 onward)** — "nvidia-pebble /
  youthful-indigo-turkey", GCP asia-southeast1-c, via an NVIDIA Brev
  accelerator-lab grant. **[CORRECTED 2026-07-03]** The grant is a 2-month
  uptime-metered hardware window, not a 1.6k GPU-h utilization budget; the
  instance cannot be stopped (`brev stop` unsupported — delete only) and
  bills while RUNNING, so the operative budget is ~192 GPU-h/day for the
  window (~10k+ GPU-h) and the strategy is full saturation with audited
  work. SSH via the Brev CLI alias
  `youthful-indigo-turkey` (see `matrix-thinking/H100_SETUP.md`). Python
  venv at `/home/nvidia/tdenv` (torch 2.12+cu13; the base image ships no
  torch/conda). Task D used ~76 GPU-h (~5% of budget) for a complete,
  decisive, written-up result — this experiment class is cheap; do not
  over-provision compute before proving the code, per the audit discipline
  below.
- Prior/legacy: single H100 80GB HBM3 pods (cloud rental, `/toy_story_slam/`
  volume) — used through the matrix-CODI workshop-paper era. Endpoint in
  `matrix-thinking/H100_SETUP.md` is stale; superseded by the Brev cluster.
- Local SSD at `/Volumes/1TB_SSD/learned-representations/` holds large data and checkpoints (not in repo)
  - `data/` — 2.3GB (WikiText-103, CIFAR-10)
  - `checkpoints/` — 219MB (model checkpoints from completed experiments)

---

## Documentation Map

*(Refreshed 2026-07-04 during a documentation consolidation pass — see
`consolidation-manifest.md` at repo root, temporary, for the full
file-by-file move/merge record of that pass.)*

**Root (this repo's top level, kept ≤8 files by design):**
- **README.md** — public-facing 1-page summary
- **STATE.md** (this file) — project dashboard, single source of truth
- **EXPERIMENT_LOG.md** — chronological, append-only history of every
  experiment with exact numbers; the 2026-07-01→04 campaign section has its
  own table-of-contents header grouping entries by program thread
- **references.md** — bibliography organized by topic
- **CLAUDE.md** — workflow rules, hard rules learned from prior experiments, and the `[LEARN]` block convention for the learnings DB
- **AGENTS.md** — the same content as CLAUDE.md, kept in sync, for the Codex CLI harness (`.codex/`)
- **AUTOPILOT_HANDOFF.md** — agentic harness (hooks, skills, notification layer); setup + phase roadmap

**matrix-thinking/ (living docs; closed-program docs carry a `STATUS:
CLOSED` header with a one-paragraph verdict rather than being archived,
since they're the primary source for a closed finding, not disposable
scratch):**
- **matrix-thinking/QUEUE.md** — engineering queue; trimmed to a banner +
  pointer table (2026-07-04) — the ~570-line pre-2026-07 body moved to
  `archive/matrix-thinking-workshop-era/QUEUE_historical.md`
- **matrix-thinking/KILL_LIST.md** — experiments killed by attack-agent review with recorded fatal flaws; still actively cited by current Chapter 2 design docs
- **matrix-thinking/CONTROL_A_HISTORY.md** — consolidated history + the
  previously-undocumented 2026-04-28 result for the Control A null-baseline
  experiment (added 2026-07-04; supersedes 6 archived design/audit docs)
- **matrix-thinking/H100_SETUP.md** — pod environment + the perpetual/unattended sweep pattern
- **matrix-thinking/DELTANET_CAUSAL_RANK_DESIGN.md**, **DELTANET_REALDATA_DESIGN.md**, **STAGE_G_DESIGN.md**, **chapter2/STAGE0_DESIGN.md**, **chapter2/TASK_D_PREREGISTRATION.md**, **chapter2/TASK_D_WRITEUP.md**, **chapter2/NEXT_EXPERIMENT_DESIGN.md** (Task E design), **chapter2/TASK_E_FINDINGS.md** — the five closed 2026-07-01→03 programs; each carries a `STATUS: CLOSED` header
- **matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md** — CLOSED through §16 (Wave 0/1/F/geo3, including the geo3 escalation); the stability-targeted follow-on (§14.8) is now `KEY_ANCHORING_DESIGN.md` (below), run
- **matrix-thinking/KEY_ANCHORING_DESIGN.md** — Rev 5, RUN 2026-07-05 (§9), CONFIRMATORY WAVE RUN same day (§9.6): candidate (d) clears K=32 h=4 ≥0.5 in 3/3 seeds with λ interior 6/6; the confirm wave closed §9.3's item-5/6/`engaged_frac` measurement gap and the literal §3.5 outcome is **C** ("mechanism not engaged"), not A — `engaged_frac` reads <14% in every leg despite item 5/h4/λ all clearing their own bars, per §3.7's own override clause. h4 stays DESCRIPTIVE tier (real, reproducible behavioral result; now a measured mechanistic null underneath it, not a missing measurement) — see STATE.md's key-anchoring bullet above and `EXPERIMENT_LOG.md`'s "KEY-ANCHORING CONFIRMATORY WAVE VERDICT" entry
- **matrix-thinking/stageg/** — Stage G's H_e task-swap harness (built, calibration run, Wave C gated open — see "In flight" above)
- **matrix-thinking/scripts/** — runnable training scripts
- **matrix-thinking/src/**, **chapter2/*.py** — model code
- **matrix-thinking/submissions/** — `icml-mi-workshop-2026/` (accepted; checklist and 5-round review are closed historical records) and `neurips-ws-2026/` (draft in progress, no figures yet)

**Elsewhere:**
- **experiment-runs/_auto_sync/WORKFLOW_FOR_AGENTS.md** — how the autonomous pod monitor, wakeup poll, and pull loop operate (agent-facing runbook for continuous GPU utilization)
- **experiment-runs/README.md** — the hybrid archive policy (≤25MB tracked in git; full archive, including large payloads, on the SSD) — source of truth also mirrored in `CLAUDE.md`'s Data section
- **research/** — individual research notes (see research/README.md for index; well-organized, no restructuring needed as of 2026-07-04)
- **experiment-runs/** — completed runs with scripts and results, size-capped per the hybrid policy above
- **archive/** — dead ends and historical material, including three folders added 2026-07-04 (`matrix-thinking-workshop-era/`, `chapter2-gauntlet/`, `team-output-2026-04-28/`) — see archive/README.md
