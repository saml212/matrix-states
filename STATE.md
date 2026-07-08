# STATE — Current Project State

**Last updated:** 2026-07-08 (post-campaign consolidation; per-event history
lives in `EXPERIMENT_LOG.md`, the design docs' § sections, `experiment-runs/`
archives, and git history — the superseded event-block stack that used to
live here was removed in this consolidation, content preserved in those
records).

## GOALS

1. **Publish the drafted material** — one paper is already PUBLISHED
   ("The Gradient Does Not See Rank", ICML 2026 MI workshop). Drafts:
   `neurips-ws-2026/` (positive rank results; PI venue + 10pp→4pp cut
   decision pending, ~Jul 11 CFP), `workshop-2026/` (capacity trilogy,
   current), `iclr-2027/` (the FULL main-conference paper — complete
   draft, deadline ~late Sept 2026).
2. **Multi-workshop strategy (PI, 2026-07-08):** chop the result
   inventory into MANY workshop submissions (ground-level credibility)
   beyond the two dated ones — chop candidates: the reasoning-link null
   program (80/80 + methodology), the instrument-calibration/fake-cliff
   methodology story, the coming M* memory result, the coming
   capability-separation result. A venue-scout agent maintains the CFP
   pipeline (dispatched 2026-07-08; re-scan Aug 2026 for NeurIPS-2026
   workshop CFPs).
3. **Land a full (main-conference) publication** — the ICLR 2027 draft is
   the vehicle; PI wants to collaborate with Berkeley/Stanford professors
   on the flagship; PI publishes it regardless, but it needs a POSITIVE
   result to matter.
4. **The overall research goal (PI, verbatim intent, sharpened
   2026-07-08):** demonstrate CAPABILITIES current architectures are
   incapable of, functionally or as observed/tested — capability
   SEPARATION, not efficiency, is the world-changing headline. Matched
   comparisons (the head-to-head) remain the grounding. Modality is an
   open research question (language / bytes / other), settled by the
   waterfall, never bundled two-unproven-axes-at-once. A
   capability-separation research wave was dispatched 2026-07-08 (TC0
   limits of transformers AND diagonal SSMs vs delta-rule state-tracking
   expressivity; sharpest unclaimed demo reachable by our stack).

## ACTIVE CAMPAIGN — HEAD-TO-HEAD DEMO (PI-ratified 2026-07-08)

Does a matrix-native fast-weight model (frozen-bias fix + recruitable
rank + super-linear capacity + exact composition) beat matched baselines?
Design registry: `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` (**Rev 2
committed+pushed 48778bb; attack round 3 running** — see QUEUE below).
PI comparison
framing — we research for the FUTURE's constraints (compute grows
fastest; quality data and HBM are the coming walls):
- PRIMARY 1: data-efficiency (param+data-matched learning curves on
  relational/compositional tasks; binding as native relational bias).
- PRIMARY 2: inference-memory-matched (fixed matrix state vs
  KV-cache-capped baseline at equal bytes, long-horizon tasks —
  "constant-memory minds", backed by the super-linear capacity result).
- FLOP-matched = disclosed control only. Param-matched flat-vector
  ablation stays mandatory (CLAUDE.md hard rule).
- WIN/TIE/LOSE pre-registered publishable per axis; escalate rungs only
  on win-or-tie. Rung 1 ≤15 GPU-h raw at the 14-98M tier (realized
  Rev 0 arithmetic: ≈11.23 GPU-h raw, ≈112.3 GPU-h at the 10× enforced
  ceiling).
- Byte-level input explicitly OUT of scope (never bundle two unproven
  axes).

**QUEUE:** design Rev 0 → attack round 1 (NEEDS-REVISION, §1.13: F1
neutral-probe-head gap → Wave −1; M1 above-cliff load; M2 byte ambiguity;
M3 tuning parity) → Rev 1 (resolved all, §1.14; raw 12.44 GPU-h, bracket
124.4 vs 127.33) → attack round 2 (NEEDS-REVISION, §1.15: F-NEW-1
cap_length DEGENERATE at rung-1 — 8-16 tokens vs 224-token bind phase →
fix = memory-multiplier sweep reporting crossover M*, inference-only;
F-NEW-2 train/eval mismatch → attention-sink patch, partial, disclosed;
M-NEW-1 TASK_BASE missing keys; M-NEW-2 Hadamard-vs-matvec tap asymmetry
undisclosed; M-NEW-3 eval-time-axis prose; M-NEW-4 budget margin 2.93
GPU-h load-bearing — fixes MUST stay inference-only) → Rev 2 (done,
§1.16, pushed 48778bb: M-sweep M∈{1..32} w/ crossover M* re-registered
WIN M*≥4/TIE 2/LOSE ≤1; absolute horizons H4=902 primary; sink patch
k_sink=4; TASK_BASE 5 keys; raw 12.59, bracket 125.88, margin 1.45
GPU-h — thinnest yet, flagged) → attack round 3 (done, NEEDS-REVISION
§1.17: R3-F1 FATAL — Rev 2's cross-dim diagnostic numerically PROVEN
broken (cosine scale-invariance → step function; needs K-simultaneous-
bindings redesign + pre-commit numeric verification); R3-F2 FATAL — M*
underpowered at n=3 AND biased toward strongest-WIN default (fixed-
sequence procedure + straddle→seed-extension + split M*=∞ into
CONFIRMED vs INDETERMINATE); R3-F3 LOSE unreachable at n_layers=2 (pin
+ remap tiers to eligible grid); R3-F4 M-sweep timing-pilot gate +
de-scope order; + β∈[0,1] caveat fold-in) → Rev 3 (in progress) →
attack round 4 → BUILD → build audit → launch rung-1 GPUs 0-6.

**CAPABILITY CAMPAIGN — waterfall ATTACK stage RETURNED (2026-07-08):**
#3 KILLED (2601.15158 is an RL-dynamics single-layer paper, NOT a
fixed-depth impossibility bound — scout mischaracterized it; also
redundant w/ head-to-head axis 1). #1 KILLED AS FRAMED — the
solvable-vs-non-solvable prediction is falsified by representation
theory BEFORE any GPU: corrected minimal faithful dims Z_m=1c/2r, S3=2,
S4=3, A5=3 (!), S5=4, A6=5 — S4 (solvable) and A5 (non-solvable) TIE at
dim 3; DeltaProduct's own published n_h numbers confirm (S4 n_h=2, A5
n_h=2, S5 n_h=4 — the hard/easy split straddles solvability). SURVIVES
AS CLAIM B: causal rank↔representation-dimension recruitment
(subspace-RESTRICTED rank per Task E's analyze_zdump methodology —
whole-matrix rank is trivially full for invertible group elements;
continuous group-element-MATRIX readout mandatory, never
softmax-over-|G| — the Nichani argmax shortcut), across a group family
interleaving solvability at matched dims (S4-vs-A5 = marquee
dissociation control). Claim tier: descriptive+causal-interventional
(Task D/E tier), NOT provable-necessity — labeled as such. Est. 20-35
GPU-h. #4 SURVIVES-WITH-MODIFICATIONS: replication-first gate mandatory
(2605.30233 characterized 2B-70B pretrained LMs; the suppression-tag
failure may not exist at 14-98M from-scratch — verify on their released
battery, github.com/PootieT/entity-tracking-mi, ~5-10 GPU-h, BEFORE
funding the full 15-25). fla allow_neg_eigval verified low-risk (β×2
pre-scale, pretrained to 1.3B in Grazzi). NEXT: VALIDATION stage
(waterfall 4) — dispatched 2026-07-08 — then campaign design gauntlet
w/ own ledger, PI-visible.

**PUBLICATION PIPELINE:** measurement-2026 chop paper DRAFTED (local
commit 4c7d172; "The Cliff That Wasn't"; compiles clean w/ tectonic,
~4.6pp body; every number provenance-commented to §15.NN anchors; claim
boundary vs workshop-2026 trilogy stated in-text) — coordinator review
then push.

**PARALLEL RESEARCH WAVES (dispatched + RETURNED 2026-07-08, PI
capability-first directive):**

(1) **Capability-separation scout — RETURNED.** Verified landscape:
fixed-depth transformers TC0-bounded (Merrill-Sabharwal line, holds
through log-CoT; only linear CoT escapes — so all claims must be
single-pass/no-CoT); diagonal SSMs ALSO TC0 (Illusion of State,
2404.08819); delta-rule models escape ONLY with negative eigenvalues
(Grazzi 2411.12537 ICLR'25 oral; DeltaProduct 2502.10297: S3 needs
n_h=2, S5 n_h=4; RWKV-7 2503.14456). ⚠️ **BLOCKING: our
`lm_pretrain_rd.py` uses plain sigmoid β∈[0,1] everywhere, zero
`allow_neg_eigval` hits in repo → contender AS CONFIGURED is
TC0-bounded, no separation property. One-line fix (β×2) but the fix
itself is published prior art (DeltaProduct at LM scale) — our novelty
must be built ON TOP.** Bare "recurrent beats transformer at state
tracking" is FORECLOSED (Chess-World-Model 2605.30100 at 3M-40M; code
traces 2602.14814 by the Grazzi group; M²RNN 2603.14360 matrix-state at
7B). THE UNCLAIMED GAP: causally-verified rank↔representation-complexity
correspondence (force-rank methodology, our asset) + inference-memory-
matched accounting — no 2026 paper measures either. Top candidates:
#1 rank↔minimal-faithful-rep-dim across solvable/non-solvable groups
(~20-40 GPU-h); #3 single-pass exact composition on the provably-
CoT-required task class (2601.15158 inverted, ~10-20 GPU-h); #4 exact
unbind vs the published "global suppression tag" failure taxonomy
(2605.30233, ~15-25 GPU-h); #2 = head-to-head axis 2 (already in
gauntlet). Bytes: CONFOUND for this campaign, stays parked (sequenced
follow-on only). "Abstract thinking" claim ceiling: systematic
compositional generalization + causal state tracking w/ extrapolation.
HEAD-TO-HEAD CROSS-IMPLICATION: contender stays β∈[0,1] there (frozen-
bias λ=0.58 evidence was collected under sigmoid β; changing mid-
gauntlet invalidates provenance) — Task-2 held-out-depth results are
empirical-only, disclose in the design's caveats register; the
capability campaign pins β∈[0,2] as its own arm w/ own calibration.
NEXT (waterfall): ATTACK stage on candidates #1/#3/#4 (dispatched
2026-07-08) — esp. verify the representation-theory dims (S5 minimal
faithful = 4-dim standard rep; A5 has 3-dim faithful irreps — the
scout's "5 for S5/A5" is suspect) and whether delta-rule state rank ↔
rep dimension is well-posed (state update S(I-βkk^T)+βvk^T IS a
Householder-product recurrence, so plausibly yes — needs rigor).

(2) **Venue scout — RETURNED.** Full table + email drafts appended to
`neurips-ws-2026/VENUE_DECISION.md`. Headlines: NeurIPS-2026 workshop
list drops **Jul 11** (both pre-built EAs → NeurReps/UniReps, deadline
~Aug 29); COLM Efficient Reasoning open to **Jul 19** (capacity trilogy
candidate); 5 late-add emails recommended THIS WEEK (MOSS best scope
match; AIMS/Sci-FM ≈ purpose-built for the instrument-methodology
story); ICBINB (reasoning-link null) re-scan Nov-Dec; NeSy is ARCHIVAL —
avoid. Instrument-methodology chop paper DRAFTING NOW
(`submissions/measurement-2026/`, agent dispatched). PI ACTIONS NEEDED:
approve/send the late-add emails (author/affiliation pending), Jul 11
submissions.
iterate to DESIGN-CLEARED-FOR-BUILD → build (new code: flat-vector
ablation mixer, switchable uncapped/capped-KV Transformer,
`verify_match_gate.py`, calibration/timing-pilot wrappers) →
independent build audit → launch rung-1 (14M). **GPU allocation for this
campaign: GPUs 0-6 available for rung-1 launch; GPU 7 held as
pool/overflow.** Escalation to 392M gated on rung-1 win-or-tie only
(§1.5/§1.9 item 1 — the escalation rung's own GPU-h budget does not yet
fit the shared 135 GPU-h ceiling at rung-1's step count; flagged, not
resolved, in Rev 0).

## CAMPAIGN SCORECARD (Jul 6-8 2026, all pushed)

**FOR the approach:** SGD recruits provably-necessary rank (causal);
super-linear capacity (x0 0.5455@d64 → 0.6779@d80; NO cliff at d=96 to
K/d=0.94); exact composition; the write-geometry attractor is FIXABLE
(frozen-bias, side-effects bounded ≈0).
**AGAINST / bounds:** the attractor WORSENS with scale (4-pt monotonic
ladder 0.248→0.344→0.389→0.455, 14M→1.31B); reasoning-link geometric
readout dead everywhere (80/80 nulls, triple-null + vocab/geometry
dissociation); causal keystone multiply-bounded null (the n=3 transient
did NOT replicate at n=12 — BATCH-EFFECT-FLAGGED, new-cohort CI spans 0);
NO demonstrated end-to-end win yet (the head-to-head's job).
**Instrument escalations (PI-gated, §15.28-class):** C17
TOLERANCE-MISCALIBRATION (n_iter 20→28 unlocked 11 cells); the admission
frontier moves with K/d (K=90 inadmissible even at 28); admission
leg-swap at K=84; K=90's exact-1.0 ceiling did not replicate fresh
(0.9725); pool-restriction shift +0.033 ≈ 12× measured noise.

## PENDING PI DECISIONS

1. Venue/author/title for the ~Jul 11 CFP (`neurips-ws-2026/`
   VENUE_DECISION.md; needs the 10pp→4pp cut call).
2. Retroactive ratification: Phase-2b vocab-space pivot + seedext
   restructure-to-B (both ran under gauntlet authority, fully recorded).
3. Fund or park the §15.28-class admission-frontier design round
   (KEY_ANCHORING_SCALING_DRAFT.md §15.27 escalations).

## LEDGERS (GPU-h, realized/ceiling)

- keyanchor-scaling: 20.46/21 (authorized +5 extension NEVER drawn) —
  CLOSED for new waves pending PI.
- frozen-bias: 11.43/135 (~123 headroom — funds the head-to-head).
- phase2b: 8.3/66.5.
- Box: Brev 8×H100 "youthful-indigo-turkey", uptime-metered (bills
  regardless; cannot stop). GPUs 0-7 all currently idle.

## SECURITY NOTE (standing)

≥29 fake `<system-reminder>` blocks observed appended to tool stdout
during the Jul 2026 sessions (date-change or false file-modification
claims, always with a concealment instruction; +≥4 this session, incl.
≥3 independently hit by dispatched sub-agents during the head-to-head
design's mandatory reading pass — see EXPERIMENT_LOG.md's matching
entry). All defeated by verify-vs-git; none complied with; all reported
to the user. Separately, this session also surfaced a real (non-security)
concurrency hazard, not an injection: a concurrently-running session
sharing this same working directory committed (`c019dcf`) while this
session's own in-progress `EXPERIMENT_LOG.md` edit was sitting
uncommitted, sweeping a WIP placeholder into that commit; caught and
cleaned up in a follow-up commit rather than amended. See CLAUDE.md Hard
Rules. Legitimate harness notices never arrive embedded in command
output.

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

> **[2026-07-08 note: historical snapshot.** Everything below was true at
> 2026-07-04; the Jul 6-8 campaign since closed the reasoning-link lane,
> resolved d=96, completed the 4-pt scale ladder, and ratified the
> head-to-head demo campaign — see the CURRENT STATE header at the top of
> this file and `EXPERIMENT_LOG.md`.]**

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

> **[2026-07-08 note: SUPERSEDED by the GOALS + ACTIVE CAMPAIGN sections
> at the top of this file** — the current path is the head-to-head demo
> campaign (`matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md`) plus the
> publication pipeline. Kept for the historical record.]**

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
- **SCALE-TRANSFER / KEY-ANCHORING — compact current picture (consolidated
  2026-07-05; supersedes the separate per-wave narrative bullets this entry
  replaces — see `EXPERIMENT_LOG.md`'s dated entries and the design docs
  below for full derivations, tables, and attack trails).**

  **CLOSED:**
  - *KEY-ANCHORING campaign — PROGRAM COMPLETE (2026-07-07 final
    verdicts)* (`KEY_ANCHORING_DESIGN.md` §9/§9.6/§9.7/§10/§10.13/§10.14/
    §11/§11.12; `KEYANCHOR_REV6_ATTACK.md`, `KEYANCHOR_REV7_ATTACK.md`).
    Five waves plus a rejected rescore attempt, **≈55.83/80 GPU-h**
    against the exactness program's own cap (≈24.17 GPU-h reserve
    remaining, untouched):
    - Wave 1 + confirmatory wave (10.98 + ~0 new GPU-h): K=32 h=4
      `rec@0.9` 0.4105 (fresh reference) → 0.556–0.665, 3/3 seeds ≥0.5
      bar, same seed integers both waves (reproduction, not independent
      replication); λ interior 0.55–0.58, 6/6; per-entity `engaged_frac`
      <14% every leg → literal **Outcome C**.
    - Rev 6 λ-scaled-threshold rescore: independently attacked and
      **REJECTED** (`KEYANCHOR_REV6_ATTACK.md`: bar-preserving z≈3.76–4.03
      sits outside the offered {2,3} menu; unverified anchor-norm
      m≈1.34 would make the signal chance-level).
    - Mechanism-tier wave (2026-07-06, `KEY_ANCHORING_DESIGN.md`
      §10/§10.13, 1.50 GPU-h realized). Fresh instrumentation (`r_e`
      measured directly, pre-blend, norm-invariant by construction;
      hash-locked BH-FDR test; `REV7_THRESHOLD_PINNED.json` verified
      byte-identical in a fresh empty sandbox) + a new per-entity-λ arm
      (d′). **Behavioral: independently replicated** — 6 genuinely
      fresh seeds, 2 architecture variants, K=32 h4 0.6141–0.7141, all
      ≥0.5 bar; K=16 s10 = 0.9998. **Mechanism: Outcome C reconfirmed,
      3/3 seeds, both arms** — engaged_frac_v3 3.7–41.1% at K=32, median
      `r_e` 0.15–0.23 (below the 0.25 partial floor), immune to every
      Rev-6 objection. Candidate (d′) → **Inconclusive/mixed**.
      **Synthesis: the anchor blend stabilizes by construction** (the
      λ·anchor term is episode-constant) — registered with a ~1 GPU-h
      falsification probe (candidate (e)).
    - **Candidate (e) verdict (2026-07-07, §10.14, 1.231 GPU-h
      realized) — CONFIRMED BY ABLATION.** Both registered arms ran:
      **e** (frozen `random_unit_rows`, never trained, seeds 60/61/62,
      h4 0.6663/0.7619/0.7540, mean 0.7274) and **e-fp** (frozen
      `frame_potential_init`, never trained, seeds 70/71/72, h4
      0.7603/0.7123/0.7512, mean 0.7413) — both at fixed λ=0.58. **Both
      arms match/slightly exceed candidate (d)'s own learned-table K=32
      mean (0.6669)** — no seed of either frozen arm falls below (d)'s
      own minimum. Per the registered joint outcome map, this routes to
      **"constancy alone suffices"** — bulk geometry is not the
      carrier (arm e, pure random init, performs the same as e-fp).
      **r_e negative control passes at the strongest null in this
      design's history**: arm e's median r_e is **negative**
      (−0.2431/−0.1345/−0.2098, engaged_frac_v3 = 0.000 all 3 seeds) —
      the instrument correctly reports no alignment when the anchor is
      pure noise with nothing to align to. **Full implication chain**:
      this supersedes the "learned anchoring" framing entirely — the
      deployable fix is a frozen random key-bias at matched λ; SGD's
      role reduces to (at most) tuning λ; the 2×2's stability
      ingredient is satisfiable by construction (the episode-constant
      term need not be derived from data). The primary entity-alignment
      hypothesis (§1) stays **Outcome C**, unchanged; the
      construction-stabilization *account* moves from
      descriptive/interpretive to **confirmed-by-ablation**.
    - **K=48 capacity-curve verdict (2026-07-07, §11.12, 1.597 GPU-h
      realized) — bar MISSED 0/3.** Candidate (d), K=48, learned λ:
      h4 0.02295/0.02287/0.01872 (mean 0.0215) vs. the registered bar
      ≥0.024434 (transplanted 1.494× relative-gain factor from K=32
      onto K=48's own collapsed baseline) — every seed misses, margins
      0.0015–0.0057. Fresh reference (bare geo3): h4 mean 0.0167
      (reproduces the archived 0.0164). Candidate (d) is **fully
      admissible 3/3** (value-salvage 0.105–0.115, clears the 0.1
      floor) while the fresh reference **fails value-salvage 3/3**
      (0.071–0.087) — a cleaner split than the falsification map's two
      adjacent rows anticipated. Item 5 (pre-NS drift) passes cleanly
      (0.99999+) → routes to **"packing ceiling reached, wasn't
      enough,"** not "mechanism didn't engage": post-NS drift (mean
      0.8870) closes ≈81% of the gap between the fresh reference
      (0.8382) and the independently-reproduced λ=1 ceiling (0.8987);
      value-Gram-relief is also present (0.626× the reference's
      deviation at the h4 leg) — both of the design's own pre-registered
      explanatory channels are measurably active, neither dormant, yet
      still insufficient to cross the bar. **The capacity curve
      completes as a cliff, not a smooth decline: ~1.00 (K=16) → ~0.65
      (K=32) → ~0.02 (K=48)**, while the λ=1 ceiling declines gently
      (0.9745 → 0.9423 → 0.8987) over the same three points. **Binding
      survives (h=1 guard 1.0000 at every cell, every K, this design's
      entire history), composition collapses.** Reported as an
      informative negative per §11.6's own discipline — bounds the
      fix's regime (K/d ≤0.5, not 0.75) and sharpens the open theory
      question (value-crowding, the design's own named candidate) —
      never pooled or averaged across K.
    - Archive: `experiment-runs/2026-07-05_keyanchor_wave/` +
      `experiment-runs/2026-07-05_keyanchor_confirm/` +
      `experiment-runs/2026-07-06_keyanchor_mech/` +
      `experiment-runs/2026-07-07_keyanchor_e/` +
      `experiment-runs/2026-07-07_keyanchor_k48/` + SSD mirrors.
    - **Full story, one paragraph:** key-anchoring behaviorally works at
      K/d≤0.5 (independently replicated, 9+ seed-runs across waves), but
      not because the anchor table learns anything — a frozen random
      table at the right blend weight does the same job
      (confirmed-by-ablation), so the mechanism is "constancy in the
      key-blend arithmetic," not "learned entity alignment" (Outcome C
      throughout). That mechanism does not transplant to K/d=0.75: the
      capacity curve is a cliff, not a gradual decline, and the two
      pre-registered explanatory channels (drift-space stabilization,
      value-Gram relief) are both measurably active at K=48 yet
      insufficient — pointing at value-crowding as the next open
      question, not yet tested. Program formally complete at
      ≈55.83/80 GPU-h; no further waves scheduled.
  - *SCALE-TRANSFER Track C — pure-scale attractor ladder*
    (`SCALE_TRANSFER_DESIGN.md` §5.9/§5.10/Addendum). The mix-axis
    confound is closed at both tested scales (14M mixcontrol Δ−0.004
    span vs. control; 98M wave-1ext Δ−0.014 span vs. rung-1, both inside
    seed noise), yielding a clean, single-(extended)-mix, 3-point
    monotone ladder: span-fraction 0.248 (14M) → 0.344 (98M) → 0.389
    (392M, rung-2) — upgraded from a joint scale+mix claim to a
    **pure-scale** claim. Geometry-leg-only; no compositional-recovery
    cross-check at any rung. Archive: `experiment-runs/2026-07-04_trackc_rung1/`,
    `experiment-runs/2026-07-05_trackc_rung2/`,
    `experiment-runs/2026-07-05_wave1ext/` + SSD.
  - *SCALE-TRANSFER Track D Phase 1* (`SCALE_TRANSFER_DESIGN.md` §6.8).
    The write-geometry signature is measurable, and larger, in production
    fixed-state models (RWKV-7 1.5B, Falcon-Mamba-7B) but a matched
    no-fixed-state negative control (Qwen2.5-1.5B) shows the same
    magnitude — **NOT attributable** to the delta-rule write mechanism
    specifically at this measurement tier. Archive:
    `experiment-runs/2026-07-04_track_d/` + SSD.
  - *SCALE-TRANSFER Track B* (`TRACKB_REDESIGN.md` §14) —
    **double-barred**: the original β-uniformity no-launch (write-mass
    0.431 vs. ≤0.40 bar) plus a duplicate-key stability smoke
    (`skip_rate=0.6319` vs. ≤0.01 bar, PROBATIVE positive control,
    196/326 calls ≥6-duplicated, max 32/32) refused Cells 3/4 and Wave 2
    before further spend. The selectivity main effects that did run
    (fix mechanism inactive) are **INCONCLUSIVE** (val-loss fails on
    openr1 for all 3 arms, passes on wikitext; Gram deviation splits
    disjoint-on-openr1 vs. overlapping-on-wikitext — a registered
    split-verdict rule). Archive: `experiment-runs/2026-07-05_trackb_wave/` + SSD.

  **RUNNING:**
  - *SCALE-TRANSFER Track C rung-3* (1.31B params, token-matched to
    rung-2 at 1.5B tok/run per the user-signed-off Rev 2.2 amendment).
    Launched on GPUs 0–1, tmux session `trackc3`. `ALL_DONE` expected
    ≈05:00 UTC 2026-07-08 (corrected 2026-07-06 from ~19:00 UTC Jul 6:
    measured 1.416 s/step is 2× the banked calibration — see the SESSION
    HANDOFF block at top for the full dated correction and the disclosed
    ≈334/300 GPU-h budget overrun). At-launch guard printed 266.47/300
    using the banked constants, in good faith at the time.
    Completes §5.7's literal 3-rung criterion (98M/392M/1.3B).

  **DECISION-PENDING (user call):**
  - The key-anchoring program is now fully CLOSED end to end (mechanism
    question, candidate (e) falsification probe, and the K=48
    capacity-curve extension all resolved — see the CLOSED bullet
    above). No open decision remains on this campaign; no wave is
    scheduled. Next direction on this thread, if any, is a fresh
    brainstorm/research/attack/validate waterfall (e.g. investigating
    value-crowding at K/d=0.75, per §11.12's own named-not-yet-tested
    candidate), not a continuation of the existing design.

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
- **matrix-thinking/KEY_ANCHORING_DESIGN.md** — **PROGRAM COMPLETE (2026-07-07)**, §10.14 + §11.12 are the final verdicts. Full arc: Wave 1 + confirm wave (§9/§9.6) → Rev 6 rescore REJECTED (`KEYANCHOR_REV6_ATTACK.md`) → mechanism-tier wave (§10/§10.13, Outcome C reconfirmed both arms, construction-stabilization account registered) → candidate (e) falsification probe (§10.14, CONFIRMED BY ABLATION — a frozen, never-trained anchor table matches/exceeds the learned one) → K=48 capacity-curve extension (§11/§11.12, bar missed 0/3, capacity cliff K/d 0.5→0.75). Literal §3.5 outcome for the entity-alignment hypothesis is **C** throughout, never revisited; the *mechanistic account* for the real, reproducible behavioral gain moved from descriptive to confirmed-by-ablation. See STATE.md's key-anchoring bullet above and `EXPERIMENT_LOG.md`'s dated verdict entries (mechanism-tier, candidate-(e), K=48-capacity-curve)
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

