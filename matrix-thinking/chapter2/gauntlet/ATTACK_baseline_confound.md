# ATTACK: Interpretability of the Chapter 2 P-slot / K-item Crossover Design

Target: whether a positive OR negative result from the proposed Chapter 2
experiment (P-slot matrix bottleneck, K-item joint-readout task, rank-k
ablation crossover at k≈⌈K/P⌉, param-matched vector baseline) would actually
mean what the design claims. Not attacking feasibility — attacking meaning.

Sources read: `matrix-thinking/CHAPTER_2_DESIGN.md`, `STATE.md` (Math
Foundations + narrowed-hypothesis status), `matrix-thinking/KILL_LIST.md`,
`experiment-runs/2026-04-29_rank_aware_v1/SYNTHESIS.md`,
`matrix-thinking/PAPER_RESULTS_SUMMARY.md`, `matrix-thinking/src/
matrix_output_heads.py`, `matrix-thinking/scripts/run_matrix_codi.py`.

---

## 1. Reshape-equivalence / is matrix structure load-bearing?

**Verdict: FATAL AS CURRENTLY SPECIFIED. Fixable, but requires specifying
an architecture detail the design doc currently leaves silent.**

The design's own Gauntlet item 3 already concedes the sharpest form of this
attack and only half-answers it: Frobenius attention `⟨Q_i,K_j⟩_F` is
literally the dot product of flattened vectors (`tr(QK^T) = vec(Q)·vec(K)`).
This is not a hypothetical — the project has already killed a proposal
(MVAttn) for exactly this reason. `KILL_LIST.md` F1: *"MVAttn with d=16 and
H=4 is GPT-2 attention with head_dim=256 — a known, uninteresting variant...
Bilinearity in (Q_i, K_j) is not new."* F3 quotes the CLAUDE.md reshape rule
directly. So Chapter 2's attention block contributes **zero** structural
advantage by the project's own prior verdict. Everything rests on one
component: the matrix MLP `Z → ReLU(Z W_1) W_2`.

Here is why that component does not obviously survive either. `ReLU(Z W_1)
W_2` with `W_1, W_2 ∈ ℝ^{d×d}` right-multiplies every **row** of Z by the
same d×d weight, independently per row, then applies ReLU, then
right-multiplies again. This is mathematically a **weight-tied MLP with
hidden width d, applied identically to each of the d row-blocks of the
flattened d²-vector.** It uses only `2d²` parameters but achieves an
effective hidden width of `d²` (d hidden units × d rows) — because of
weight-sharing across rows, not because of anything intrinsically
"matrix." A generic (unstructured, fully dense) vector MLP matched to the
same `2d²` parameter budget is forced into a hidden width of
`2d²/(2·d²) ≈ 1` — i.e. a single hidden unit. That comparison (d²-effective-
width weight-tied matrix MLP vs. 1-hidden-unit dense vector MLP) is not a
test of "matrix vs. vector," it's a test of "weight-sharing vs. no
weight-sharing" dressed up as the former. **The design doc never specifies
the vector baseline's MLP architecture at all** — it only says "a dim-d²
vector per slot, holding items via superposition." That silence is the
crack the whole result can fall through: if whoever builds it reaches for a
standard dense MLP at matched total params, the matrix arm wins for a
reason that has nothing to do with rank, superposition, or reasoning
capacity — it wins because it's the only arm allowed to have a sensible
hidden width.

**The control that defeats this:** add a **third arm** — "reshape-
equivalent structured vector": operate on the same d²-vector but reshape it
to d blocks of size d, apply the identical weight-tied
`ReLU(block · W_1) W_2` MLP (same W_1, W_2 shapes, same param count) block-
wise, with no notion of "matrix" anywhere in the code (no SVD-friendly
data structure, just a for-loop or batched einsum over blocks). This arm is
byte-for-byte compute-equivalent to the matrix MLP, but coded and thought
of as pure vector engineering.
- If the reshape-equivalent arm matches the matrix arm's accuracy AND
  produces the same rank-k crossover **when you reshape its flattened
  state back to d×d for SVD post hoc**, the experiment has shown that the
  advantage is weight-sharing, and "rank"/"matrix thinking" is a costume,
  not a mechanism — exactly the outcome the project's reshape-equivalence
  rule predicts and exactly what Chapter 2 is supposed to be capable of
  falsifying.
- Only if the reshape-equivalent arm does noticeably worse than the true
  matrix arm (same param count, same effective hidden width, same
  operations, different only in whether SVD-decomposable structure is
  imposed on the intermediate state, e.g. via a symmetric or
  orthogonally-initialized W_1/W_2) is there a genuine claim that
  "matrixness" itself (not just the compute pattern it happens to imply)
  is doing something. This is a strong, non-obvious thing to have to show,
  and the current design doesn't ask for it.
- Also apply rank-k ablation/effective-rank measurement **identically** to
  all three arms (matrix, generic vector, reshape-equivalent vector) by
  reshaping every arm's slot state to d×d before running SVD. If the
  generic vector arm's *reshaped* hidden state also shows a bend at
  k≈⌈K/P⌉, "rank" was never matrix-specific — it's a property of any
  sufficiently capable d²-dim representation once you choose to look at it
  through an SVD lens.

## 2. Crossover confound (task difficulty vs. rank bottleneck)

**Verdict: FIXABLE, mostly already addressed by construction, but two gaps
remain.**

Holding K and P fixed while sweeping k at inference (the design's actual
rank-k truncation protocol) is a reasonable within-condition control — it
does isolate "does removing rank hurt" from "is this task instance hard,"
since K is constant within a single curve. Two confounds survive:

1. **Sequence length / interleaving grows with K.** If K parallel entity
   streams interleave over more total steps as K grows, then a K>P run is
   simultaneously (a) demanding more capacity per the design's hypothesis
   AND (b) longer / harder to optimize for reasons unrelated to slot rank
   (more opportunities for attention dilution, more steps to backprop
   through). A bend at K>P could be "sequences got longer" rather than
   "P slots ran out of rank." **Control:** decorrelate K from total
   sequence length — fix total steps and vary only how many distinct
   streams share them, or report a length-matched K sweep as a
   robustness check.
2. **Base (full-rank, k=k_max) accuracy must be checked comparable across
   K≤P and K>P before the truncation curve is interpreted.** If full-rank
   accuracy already degrades as K grows past P (plausible — it's simply a
   harder instance of the task even for an architecture with "enough"
   rank), then the observed "bend" at low k could be relative degradation
   measured from an already-lowered ceiling, not evidence the bottleneck
   specifically produced the failure. **Control:** report rank-k accuracy
   normalized to that condition's own full-rank accuracy (relative
   degradation), not absolute accuracy, and separately confirm full-rank
   accuracy is comparable across K≤P vs K>P conditions (or explicitly
   model the confound if it isn't).
3. **Optimization-difficulty confound.** Verify training accuracy is near
   ceiling (loss plateaued) for every (K,P) cell separately before
   interpreting its eval-time rank-k curve. An undertrained large-K run
   will show degraded accuracy at every k, which can mimic a "bend" for
   reasons having nothing to do with capacity.

The genuinely load-bearing control here is the vector baseline from
Attack 1: only if it does **not** show the same K>P bend at the same k*
(or shows a different threshold consistent with vector capacity, e.g. at
k related to d rather than P) does "P slots ran out of matrix rank" survive
as the specific explanation, as opposed to "any bottlenecked architecture
degrades once demand exceeds supply," which is a much less interesting
and non-matrix-specific claim.

## 3. Rank measurement reliability

**Verdict: FATAL AS PLANNED, fixable with seed count + pre-registration +
scatter reporting.**

This project already has direct, on-point evidence that rank is a noisy
readout at this exact scale. `PAPER_RESULTS_SUMMARY.md` Table 2 (matrix-
CODI, d=16, flatten readout, 3 seeds): accuracy tight at **81.51% ± 1.2pp**
while **Z_rank varied 3×: 4, 12, 13** across seeds 42/7/1337. Even more
pointed: `rank_aware_v1/SYNTHESIS.md` shows a baseline (unconstrained)
matrix-CODI run hitting **effective rank 13.22 — near-full-rank** — yet a
force-rank=1 training run scored *higher* accuracy (61.72% vs 58.99% mean).
That is, in a closely analogous d=16 setting, high measured rank coexisted
with rank being **functionally decorative** (forcibly collapsing it to 1
did not hurt, and if anything helped). This is the single most relevant
piece of internal evidence against trusting a correlational match between
measured rank and ⌈K/P⌉ as sufficient proof that rank is functionally used.
A crossover that "looks right" on a mean-rank-vs-K plot could still be
sitting on the same kind of seed noise / decorative-rank phenomenon
documented here.

Concrete gaps in the current design:
- **No seed count specified.** Given the demonstrated 3× spread at matched
  accuracy, and given the target signal (⌈K/P⌉) is typically a difference
  of 1-3 integer units between adjacent K/P configs, the design needs
  enough seeds to distinguish signal from a noise floor this large.
  Minimum: 5 seeds per (K,P) cell, more if early runs show high variance.
- **Metric inconsistency between documents.** `STATE.md` Math Foundations
  explicitly commits: *"We use stable rank as the primary metric."*
  `CHAPTER_2_DESIGN.md`'s evaluation section instead lists *"Effective
  rank of Z"* without mentioning stable rank or participation ratio. These
  are different functions of the singular spectrum with different
  sensitivity to noise (effective rank/entropy is generally more sensitive
  to small singular values than stable rank). Pin down ONE primary metric
  before running, and report the other two as secondary checks — don't
  choose post hoc based on which one shows a cleaner bend.
- **Report full per-seed scatter, not a mean-only summary.** Table 2's
  finding was only visible because per-seed ranks were reported
  individually; a mean-rank-vs-K line plot would have hidden the 3×
  spread entirely and looked like a clean trend.

## 4. Does SVD rank-k truncation at inference test the mechanism?

**Verdict: FATAL AS THE SOLE TEST, fixable by promoting the train-time
constraint (already proven in this codebase) to the primary causal test.**

Two distinct failure modes make post-hoc truncation an ambiguous
instrument, and both have precedent in this project:

1. **Instrument-insensitivity, not null mechanism.** The design's own
   evaluation plan says to use "the same protocol as the main paper" for
   rank-k truncation. That protocol (`PAPER_RESULTS_SUMMARY.md` Table 1)
   produced **flat curves across four different readout architectures**
   (linear flatten, bilinear, bilinear+GELU, SVD-augmented, quadratic),
   including nonlinear-in-Z readouts specifically designed to break the
   constant-Jacobian objection. That flatness was eventually attributed to
   the CODI distillation objective being rank-blind — a claim about *that
   architecture*, not a validation that the *measurement instrument
   itself* reliably detects rank use when it exists. Reusing this exact
   protocol for Chapter 2 without a **positive control that is known (or
   constructed) to use rank k** — e.g. a hand-planted Reasoning-by-
   Superposition-style latent, or simply re-running the truncation eval on
   `rank_aware_v1`'s own force-rank=1 vs. unconstrained checkpoint pair,
   where the ground truth (forced rank 1) is already known — risks
   reporting "matrix-thinking is dead" when the correct conclusion is "the
   post-hoc truncation test doesn't detect rank use in this family of
   models, full stop." That distinction changes the decision made at the
   end of Chapter 2 (kill the whole direction vs. fix the test).
2. **Distribution shift at inference.** Truncating Z to rank k post hoc
   feeds the downstream readout (and, if truncation happens mid-stack,
   subsequent layers) an input it never saw during training. A flat curve
   is compatible with "rank wasn't used" but equally compatible with "the
   readout / later layers are robust to this particular out-of-
   distribution perturbation," which says nothing about whether the
   original full-rank representation was functionally exploiting rank
   during normal (non-truncated) operation.

**The cleaner causal test, already validated in this codebase:** train-
time hard rank constraints, exactly as `run_matrix_codi.py`'s
`--force-rank-during-training` flag and `rank_aware_v1` already implement
(`force_rank_k` truncates Z to rank k **every training step**, not just at
eval). Sweep k ∈ {1, ..., ⌈K/P⌉-1, ⌈K/P⌉, ⌈K/P⌉+1, full} at **train time**
and look for where accuracy craters. This tests capacity directly — the
model never gets a chance to develop a workaround the constraint would
break, because the constraint is present throughout optimization. This
should be the **primary** test; post-hoc SVD truncation should be reported
only as a secondary consistency check, not the deciding evidence.

**Additional controls against "any perturbation of this size hurts":**
- **Magnitude-matched random-noise ablation:** replace the discarded
  singular directions with random noise of matched Frobenius norm rather
  than zeroing them. If this degrades accuracy similarly to rank-k
  truncation, the "bend" reflects generic information removal, not
  specifically rank.
- **Bottom-k-kept control:** keep the k *weakest* singular directions
  instead of the top-k. If bottom-k performs comparably to top-k for
  matched k, the SVD ordering (i.e., "rank" as a privileged basis) isn't
  actually what the readout depends on.

---

## Minimal control set required for an interpretable result (either way)

1. **Fully specify the vector baseline's MLP** at true param match
   (generic dense, not silently weight-tied/block-structured). Add a
   third **reshape-equivalent structured vector** arm (same block-
   weight-tied MLP as the matrix arm, computed on the flattened vector) to
   isolate weight-sharing from "matrixness." Apply rank-k
   ablation/effective-rank measurement identically to all three arms by
   reshaping every arm's state to d×d for SVD.
2. **Decorrelate K (task demand) from sequence length**, and report
   rank-k accuracy **normalized to each condition's own full-rank
   accuracy**, plus confirm training-accuracy convergence per (K,P) cell
   before interpreting eval curves.
3. **≥5 seeds per (K,P,arm) cell**, report full per-seed scatter (not
   mean-only), and **pre-register one primary rank metric** (resolve the
   STATE.md-vs-CHAPTER_2_DESIGN.md stable-rank/effective-rank
   inconsistency before running).
4. **Promote train-time hard-rank-constraint sweep to the primary causal
   test** (proven protocol, already in `run_matrix_codi.py` via
   `force_rank_k`), with post-hoc SVD truncation as secondary. Validate
   the truncation instrument itself with a positive control (planted-rank
   task, or re-running it on the existing `rank_aware_v1` force-rank=1
   checkpoints where ground truth is known) before trusting a flat curve
   as a negative result. Add magnitude-matched random-noise and
   bottom-k-kept controls to rule out generic-perturbation explanations.

Without (1) and (4) in particular, either outcome of Chapter 2 is
ambiguous: a "bend" could be weight-sharing plus a possibly-insensitive
instrument happening to show something; a "flat" curve could be an
insensitive instrument reproducing the same false-negative pattern already
seen in the matrix-CODI work, mistakenly read as a second confirmation
that matrix-thinking is dead.
