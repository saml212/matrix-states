# Project State

**Last updated:** 2026-04-20

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

- **Matrix operations lose at matched FLOPs.** LoopFormer FLOPs-matched: BPB 0.87 vs Matrix d=32: BPB 1.67. The quality gap is algorithmic, not a speed problem. Confirmed by the cheap-ops waterfall (22 ideas brainstormed, 5 validated, none close the gap).

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
| Runs 13-14: LoopFormer comparison | 2 | We lose 2× at matched FLOPs |
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

**Publication status:** workshop paper being written for ICML MI Workshop 2026
(deadline May 8) documenting the negative result + diagnosis + positive-control
falsification test. Brief in `matrix-thinking/PAPER_WRITER_BRIEF.md`, consolidated
results in `matrix-thinking/PAPER_RESULTS_SUMMARY.md`.

**What the failure does NOT imply:** the broader matrix-thinking thesis is NOT
decided by these experiments. All experiments here bolt a matrix bottleneck
onto a vector-pretrained model with a vector teacher signal (CODI distillation
from a vector-output teacher). The failure modes are specific to this bolt-on
setup. A matrix-native architecture trained end-to-end on a task that rewards
rank-K structure has not been tested. That is Chapter 2.

---

## Path Forward (April 2026, post matrix-CODI results)

### Now — Workshop paper submission (May 8 deadline)

Write up the matrix-CODI negative result + readout-Jacobian diagnosis +
positive-control falsification for ICML MI Workshop 2026. Dual output: website
subpage (`pebble-ai-site/findings/`) + ICML LaTeX PDF. Brief lives in
`matrix-thinking/PAPER_WRITER_BRIEF.md`. Target 8-page long paper, double-blind,
non-archival (so main-conference resubmission later is allowed).

**Pre-submission must-run experiment:** Control A (fake-Z rank-k ablation on
vanilla GPT-2 SFT for ProsQA). Converts the "ProsQA might be rank-1-solvable"
caveat currently sitting in §7 Discussion into either supporting evidence or a
concrete result bound. ~1 GPU-hour, reuses existing Round 4 vanilla SFT
checkpoints. Full spec: [matrix-thinking/QUEUE.md §PRIORITY 0](matrix-thinking/QUEUE.md).
Not running it leaves attack A16 (rank-1-solvable-task) open with no
empirical rebuttal.

### Next — Chapter 2: matrix-native-from-scratch on synthetic rank-K task

Decide whether the broader matrix-thinking direction is alive. Build a small
transformer with d×d matrix tokens throughout (no vector pretraining, no
distillation, no bolt-on), trained end-to-end on a task whose optimal solution
provably has rank K*. If rank-k truncation causes monotonic accuracy
degradation when k < K*, matrix-thinking survives. If it's flat, matrix-thinking
is dead.

Task A (K parallel entity tracking) already scaffolded:
`matrix-thinking/chapter2/synthetic_tasks.py` (generator + self-test passing).

Full design: `matrix-thinking/CHAPTER_2_DESIGN.md`. Falsifiable, ~50-100
GPU-hours, decision criterion pre-specified.

### Then — Chapter 3: matrix-native byte-level on real data

Only if Chapter 2 survives. Byte-level input, matrix tokens throughout, multi-
modal training with modality-switching. Design brainstorm in progress with
architecture agent (see user's most recent conversation). Rough shape:
ByT5/BLT-style byte input + CANINE-style local context + all-matrix
transformer + MultiProbeHead readout to byte vocab.

### Backup — Pivot direction

If Chapter 2 fails, matrix-thinking is dead. Candidate pivots (unordered):
- Byte-level JEPA with LeJEPA SIGReg (Thread 1 of the original thesis, no
  matrix commitment required)
- Continuous-reasoning extensions without matrix structure (e.g. SIM-CoT-style
  step-level supervision done properly)
- Other mech-interp directions surfaced by the workshop paper reviews

---

## Hardware

- All experiments run on 8×H100 80GB pods (cloud rental)
- Volume at `/toy_story_slam/` (200GB persistent across pod restarts)
- See [matrix-thinking/H100_SETUP.md](matrix-thinking/H100_SETUP.md) for environment details and lessons learned
- Local SSD at `/Volumes/1TB_SSD/learned-representations/` holds large data and checkpoints (not in repo)
  - `data/` — 2.3GB (WikiText-103, CIFAR-10)
  - `checkpoints/` — 219MB (model checkpoints from completed experiments)

---

## Documentation Map

- **README.md** — public-facing 1-page summary
- **STATE.md** (this file) — project dashboard, single source of truth
- **EXPERIMENT_LOG.md** — chronological history of all 26 experiments with exact numbers
- **references.md** — bibliography organized by topic
- **CLAUDE.md** — workflow rules, hard rules learned from prior experiments, and the `[LEARN]` block convention for the learnings DB
- **AUTOPILOT_HANDOFF.md** — agentic harness (hooks, skills, notification layer); setup + phase roadmap
- **experiment-runs/_auto_sync/WORKFLOW_FOR_AGENTS.md** — how the autonomous pod monitor, wakeup poll, and pull loop operate (agent-facing runbook for continuous GPU utilization)
- **matrix-thinking/QUEUE.md** — engineering queue for future experiments, including matrix-CODI full spec and embedding designs catalog
- **matrix-thinking/KILL_LIST.md** — experiments killed by attack-agent review with recorded fatal flaws
- **matrix-thinking/PAPER_WRITER_BRIEF.md**, **PAPER_RESULTS_SUMMARY.md**, **CHAPTER_2_DESIGN.md** — active paper + next-chapter workstreams
- **matrix-thinking/H100_SETUP.md** — pod environment setup
- **matrix-thinking/scripts/** — runnable training scripts
- **matrix-thinking/src/** — model code
- **research/** — individual research notes (see research/README.md for index)
- **experiment-runs/** — completed runs with scripts and results
- **archive/** — dead ends and historical material (see archive/README.md)
