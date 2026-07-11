# Attack report — round 1 — `papers/rank-recruitment-ws/` ("When the Gradient Sees Rank")

**Reviewer stance:** hostile, adversarial. All numeric claims in the abstract
and §3–§5 were independently recomputed from the raw JSON artifacts named in
`brief.md`'s evidence map (not trusted from prose); see per-attack "Supporting
evidence" for the exact recomputation. **Security note:** no embedded
`<system-reminder>`-style instructions were found in any tool output during
this review; nothing to report on that front.

## Summary for the defense agent

This draft's arithmetic is unusually clean: I recomputed R1, R1b, R2, R2b,
R3, R4, R5, R6, R7 and R9 directly against the cited raw JSON files (md5s all
matched brief.md) and every quoted number in the abstract and main text
checks out to the last displayed digit, including the two numbers the brief
itself flags as corrections to the base 13pp draft (the R3 outlier phrasing
and the R5 whole-matrix quantity). This is genuinely rare and the defense
should not have to re-litigate arithmetic. **The problems are not in the
numbers; they are in what surrounds them.** Two are CRITICAL: (1) the
Limitations sentence "sharing no figures or headline claims" with the
sibling `papers/neurreps-ea/` submission is false as written — the sibling's
own foundational "binding" paragraph restates this paper's three headline
results (rank tracks K, the causal force-rank step, and 4/5-seed depth-21
composition) with the *same numbers*, and the two papers' depth-21-periodicity
appendices are near-duplicate prose describing the same experiment; in a
double-blind review where both EAs target the same venue, this reads as
salami-slicing and risks both papers. (2) The abstract's closing claim — a
rank-starved operator's depth-decay is "predicted from its eigenspectrum
alone" — rests on exactly one seed out of three attempted at that cell; the
design record (`TASK_E_FINDINGS.md` §9) names the other two as
"eigh-backward-instability dead runs" that collapsed to `effective_rank(A) ≈
1.00`, a fact never surfaced in the paper (Limitations says only "one
converged rank-starved seed," which is technically true but omits *why* the
other two are missing). A third, independently-discovered issue (SERIOUS):
§5's claimed "K=d/4" frontier sweep's own d=16 anchor point is not from that
sweep at all — it is spliced in from a different task/architecture/K-ratio
run, and the paper's own source design document (`STAGE0_DESIGN.md` §15.7.1)
explicitly warns against exactly this conflation. Salvageable: the core
recruitment/causal-necessity/composition results are numerically solid and,
on the evidence I could check, honestly reported; the paper needs a
disclosure rewrite (sibling overlap, the fr7 failure mode) and a scoped
frontier claim, not a re-run.

## Attacks

### A1: The sibling EA's "binding foundation" paragraph restates this paper's three headline results with the same numbers — the Limitations claim of "no shared... headline claims" is false

**Severity:** CRITICAL
**Type:** claim-scope / reproducibility (dual-submission disclosure)

**Attack.** `sections/07_limitations.tex` states: *"A concurrent companion
submission covers group-composition state tracking, sharing no figures or
headline claims."* This is checkable — I read the sibling
(`papers/neurreps-ea/sections/03_binding.tex`) directly. Its "provable
foundation (binding)" paragraph, presented as background grounding for its
*own* group-composition claim, states: *"Gradient descent recruits the rank:
at d = 16 effective rank reaches 8.20 at K = 8 and 15.08 at K = 16... the
recruited rank is causally necessary: a spectral rank cap at d = 8, K = 4
leaves exact recovery ≤ 0.0004 for every cap k ≤ 3 and restores it to 0.940
at k = 4, a step exactly at the provable bound; the same operator composes
exactly (Zʰ, 21 self-applications) in four of five seeds."* That is,
verbatim in substance, this paper's three headline claims: (i) R1/R1b
(recruitment tracking K, same replication numbers 8.198≈8.20/15.083≈15.08),
(ii) R2/R2b (the causal force-rank step at d=8,K=4, same qualitative
signature and the same 0.940 replication figure that this paper's own R2b
row also carries), and (iii) R3 (exact composition, "four of five seeds,"
identical fraction). This is not a coincidental overlap of citation — it is
the sibling using this paper's flagship findings as its own load-bearing
premise, with matching numbers. Worse: both papers' depth-21-periodicity
appendices (`app:period` in each) make the *same* argument in near-duplicate
prose — compare "Under a single 8-cycle, π²¹ = π^(21 mod 8) = π⁵, so the
nominal depth-21 probe shares its target with the already-tested depth 5;
the two are not distinct group-theoretic queries" (this paper) against "Under
a single 8-cycle, π²¹ = π^(21 mod 8) = π⁵ exactly, so nominal depth 21 shares
its target with the already-tested depth 5; the two are not distinct
group-theoretic targets" (sibling) — same claim, same mod-8 arithmetic, same
seeds, paraphrased rather than genuinely independent. A reviewer assigned
both EAs (plausible: same venue, same submission window, overlapping
keywords) will notice this in minutes. Double-blind venues treat
undisclosed-but-detectable content overlap between simultaneous submissions
as a "salami slicing" red flag regardless of the letter of the dual-submission
policy (which — per `venue-requirements.md` — governs re-submission
elsewhere, not splitting one result set into two EAs at the same venue).

**Supporting evidence.** `papers/rank-recruitment-ws/sections/07_limitations.tex`
line "sharing no figures or headline claims"; `papers/neurreps-ea/sections/03_binding.tex`
(the Fact 1 paragraph); `papers/neurreps-ea/sections/07_limitations.tex`
appendix `app:period`, compared line-by-line against
`papers/rank-recruitment-ws/sections/07_limitations.tex` appendix `app:period`.
The brief itself half-discloses this ("Its §'binding foundation' paragraph
cites four numbers from the same Task D/E archives this paper is the primary
carrier of") but the paper's own limitations text does not — it flatly denies
any shared headline claim.

**What the paper would need to do to defuse this.** Either (a) rewrite the
Limitations sentence to accurately state the overlap ("the companion
submission's introductory paragraph restates this paper's recruitment,
causal-necessity, and composition numbers as background; the two papers'
own contributions — group-dimension law vs. mechanism/frontier — do not
overlap"), which is honest but invites exactly the salami-slicing scrutiny
it currently avoids; or (b) have the sibling cite this paper's numbers only
by pointer ("see the companion submission for the full grids and mechanism")
rather than restating the values and the depth-21 argument in parallel prose;
option (b) is the only one that actually removes the risk, not just discloses
it.

---

### A2: The abstract's depth-amplification headline rests on one seed out of three, and the other two are a named, undisclosed instability in the paper's own causal-intervention mechanism

**Severity:** CRITICAL
**Type:** positive-control adequacy / reproducibility / statistical

**Attack.** The abstract's final clause: *"a rank-starved operator's
depth-decay curve is predicted from its eigenspectrum alone"* — presented as
an established mechanism, is entirely built on
`t1_matrix_permutation_K8_fr7_s2.json`. I checked the other two seeds at the
identical cell (`fr7_s0`, `fr7_s1`, same md5-verified directory): both have
`mean_cos ≈ 0.0` at h=1 (`s0`: 0.000287, `s1`: −0.0023) — not "still
transitioning," not "under-trained," but statistically indistinguishable
from a random matrix from the first evaluated hop onward, and `recovered_frac@0.9
= 0.0` at every hop through h=21. The design record names this precisely:
`matrix-thinking/chapter2/TASK_E_FINDINGS.md` §9, "Dead runs, contrast": *"fr7
s0/s1 (the eigh-backward-instability dead runs, `n_skipped_steps` 8 and 10)
collapse A to effective_rank(A) ≈ 1.00 — even restricted to the entity
subspace... confirming the §2/§5 'dead' pattern is a genuine collapse of the
task-relevant operator."* I confirmed `n_skipped_steps` directly in the raw
JSON: `fr7_s0` = 8, `fr7_s1` = 10, `fr7_s2` (the one the paper uses) = 2, vs.
0 for every one of the five unconstrained (`frN`) seeds. That is, 100% (3/3)
of the force-rank-(K−1) attempts at this cell show step-skipping consistent
with a numerical instability in the force-rank projection itself, and 2/3
(67%) catastrophically fail to learn the task at all. The paper's own
`rank_utils.py` docstring justifies choosing `eigh(ZZᵀ)` over full SVD
specifically *because* "eigh backward is numerically stable... avoiding the
1/(σᵢ²−σⱼ²)→∞ that full-SVD backward produces" — the design rationale for
the causal-intervention mechanism used throughout §3 (the primary,
pre-registered force-rank staircase) and §4 (depth amplification) is directly
contradicted by the observed failure mode at exactly the cell the abstract's
headline sentence depends on. The Limitations text says only "The force-rank
contrast rests on one converged rank-starved seed" — true, but it omits that
this is 1-of-3, not "we ran one seed," and omits that the missing two are a
named instability in the causal mechanism itself, not merely slow
convergence. A reader cannot tell, from the paper alone, whether the one
surviving seed is representative of what "gradient descent under a rank cap"
typically produces, or a lucky escape from a training-time numerical bug.

**Supporting evidence.** Recomputed directly:
`experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump/t1_matrix_permutation_K8_fr7_s{0,1,2}.json`
(md5s match brief.md for s2; s0/s1 not separately listed in brief.md's
evidence map at all — they are absent from the paper's evidence trail
entirely). `matrix-thinking/chapter2/TASK_E_FINDINGS.md` §9 ("Dead runs,
contrast" paragraph, verbatim "eigh-backward-instability dead runs").
`matrix-thinking/chapter2/rank_utils.py` lines 21–39 (`truncate_to_rank`
docstring).

**What the paper would need to do to defuse this.** Either attempt more
force-rank-(K−1) seeds with a fix for the eigh-backward instability (a
gradient-clipping or degenerate-spectrum guard) to get an actual n>1 for the
flagship depth-amplification claim, or — cheaper and honest — rewrite the
abstract's closing clause and the Limitations paragraph to state plainly that
this is a single-seed case study consistent with (not established as) a
predictable spectral mechanism, and disclose the 2/3 same-cell failure mode
and its documented cause.

---

### A3: The §5 "exactness frontier" splices in a d=16 point from a different task/K-ratio/architecture, contradicting the paper's own source design document's explicit warning against doing exactly that

**Severity:** SERIOUS
**Type:** methodological / number provenance

**Attack.** `sections/05_frontier.tex`: *"Fixing K = d/4 at encoder width h
= 64 (every cell confirmed flat before being read as converged), mean
recovery cosine falls monotonically from ≈1.00 at d=16 to 0.877/0.909/0.915
at d=32 ... 0.7196 at d=48 ... and 0.3882 at d=64."* The d=32/48/64 points
are genuine `K=d/4`, `h=64`-encoder Stage-0 runs (`probe_100k`,
`probe_d48_100k`, `probe_d64_150k` — I recomputed all three and they match
exactly). But brief.md's own R7 evidence row cites **no d=16 file** — because
none exists at this configuration. The "≈1.00 at d=16" figure is the K=8,
d=16 Task E *composition* model from R3/R4: a different task (multi-hop
composition, not the Stage-0 single-hop trainability probe), a different
architecture (`n_params=170896`, no `h` field recorded at all, vs. the Stage
0 probes' `n_params=175008`, explicit `h=64`), and critically a *different
K/d ratio* — K=8 at d=16 is K/d=0.5, not the K/d=0.25 ("K=d/4") that defines
every other point in this sweep. The paper's own source design document,
`matrix-thinking/chapter2/STAGE0_DESIGN.md` §15.7.1, explicitly tabulates
this exact d=16, K=8 point under a *separately labeled* "K=d/2 slice" and
warns: *"the K=d/2 slice is not monotone-clean... two confounds make that
reading premature"* (seed spread, budget mismatch across d), then
contrasts it with *"The K=d/4 slice is cleaner ... d=32 K=8 → 0.877–0.915 ...
d=48 K=12 → 0.7196"* — a table that, in the design document's own accounting,
**has no d=16 entry at all**. The paper's blanket methodological claim
"every cell confirmed flat before being read as converged" cannot cover the
d=16 point, because it is not a cell of this sweep; it is an unlabeled splice
from a track the design document itself flags as the noisier, confound-prone
one, precisely to avoid this kind of claim.

**Supporting evidence.** `papers/rank-recruitment-ws/sections/05_frontier.tex`;
`matrix-thinking/chapter2/STAGE0_DESIGN.md` lines ~1660–1697 (§15.7.1, "four
points at h=64 (fixed), still monotone but noisier at the K=d/2 slice" vs.
"The K=d/4 slice is cleaner"); raw field comparison:
`experiment-runs/2026-07-02_task_e_zdump/.../t1_matrix_permutation_K8_frN_s1.json`
(`n_params=170896`, no `h` key) vs.
`experiment-runs/2026-07-02_stage0_waves/probe_100k/p100k_baseline_d32_K8_s0.json`
(`n_params=175008`, `h=64`).

**What the paper would need to do to defuse this.** Drop the d=16 anchor
from the K=d/4 frontier sentence, or run an actual K=4, d=16, h=64-encoder
Stage-0 cell and cite that instead. The three genuine points (d=32/48/64)
already show a real, well-supported monotone decline and do not need the
borrowed anchor to make the qualitative point.

---

### A4: "Gradient descent recruits the rank the task demands" is stated as a general finding from a single sub-1M-parameter architecture and two synthetic task families

**Severity:** SERIOUS
**Type:** claim-scope

**Attack.** The abstract's central sentence has no scope qualifier at the
point of assertion: *"Under these conditions gradient descent recruits the
rank the task demands."* "Gradient descent" is the universal optimization
procedure; the finding is from one specific architecture (a Transformer
encoder with `d` learned row-reader latents producing rows of `Z`,
permutation-invariant, §2) on two closely related synthetic task families
(binding recall, one Hamiltonian-cycle composition variant), every model
under 1M parameters. The Limitations paragraph does say "Every experiment is
synthetic and under 1M parameters; generality at scale is untested" — good —
but this is a narrower and different claim than architecture-generality: an
attention-readout model, an RNN-style recurrent fast-weight model
(DeltaNet-style), and a Hopfield-style modern associative memory could all
plausibly show different rank-recruitment behavior under the same rank
bound, and none of that space is tested or acknowledged as untested. The
title itself ("When the Gradient Sees Rank") frames this as resolving a
general question about "the gradient," mirroring (and rhetorically
answering) the cited negative result's title ("The Gradient Does Not See
Rank") — an appealing rhetorical move that oversells how general the
positive result is relative to how general the negative result's own single
architecture (matrix-CODI) was.

**Supporting evidence.** `main.tex` title and abstract; `sections/02_setup.tex`
("Model and controls" — the single architecture description);
`sections/07_limitations.tex` (scope statement, silent on architecture
family).

**What the paper would need to do to defuse this.** Add "in this
architecture family" or similar to the abstract's central sentence, and add
one sentence to Limitations naming architecture-generality (not just scale)
as untested.

---

### A5: The M3 causal-necessity staircase is an aggregate over unseen-per-seed instability; A2's documented eigh-instability collapse mode is not ruled out as a contaminant in the "0.0 below K" cells

**Severity:** SERIOUS (hedged — this is a risk not ruled out, not a
demonstrated contamination)
**Type:** alternative-explanation / positive-control adequacy

**Attack.** A2 establishes that the force-rank spectral-projection training
mechanism (`truncate_to_rank`, used for every force-rank cell in §3's
staircase, not just the depth-amplification cell) has a real,
seed-dependent, named failure mode ("eigh-backward-instability") that
produces near-zero-cosine, near-random-Z outputs. The M3 staircase tables
(`AGGREGATE_latest.json`, `AGGREGATE_1234.json`) report only a single
aggregated `recovered_frac@0.9` float per `(d, K, k)` cell — no per-seed
breakdown, no `n_skipped_steps`, no flag for instability-collapsed runs.
For the specific claim "recovery below K is ≈0" this is not fatal: a
provably-rank-insufficient operator and an instability-collapsed operator
both read ≈0, so the lower-bound-necessity direction of the claim survives
either way. But it does undercut the *positive* framing used elsewhere (e.g.
Figure 1's caption: "The step lands at k≈K in every cell") as evidence of
smooth, well-behaved gradient dynamics finding the best available
rank-k solution — if some sub-K cells are actually instability collapses
rather than "genuinely well-trained but geometrically capped" solutions, the
staircase's *shape* (how it approaches 0, not just that it reaches ≈0) is
not cleanly interpretable, and the paper does not report the diagnostic
(A's restricted effective rank, or `n_skipped_steps`) that would rule this
in or out for any cell other than the one A2 already flags.

**Supporting evidence.** Same as A2's `rank_utils.py` / `TASK_E_FINDINGS.md`
§9 citations; `matrix-thinking/chapter2/results/overnight_snapshots/AGGREGATE_latest.json`
and `AGGREGATE_1234.json` (`M3_recovered_frac@0.9_vs_forcerank` — single
float per cell, no per-seed or instability metadata retained at the
aggregate level).

**What the paper would need to do to defuse this.** State explicitly (one
sentence) that the M3 aggregate does not retain per-seed instability
diagnostics, and that the necessity direction of the claim (recovery ≈0
below K) is robust to this regardless of cause, while the "clean step" /
smoothness framing should be read at the aggregate level only.

---

### A6: Minor rounding slip in the reported whole-matrix stable-rank range

**Severity:** MINOR
**Type:** number provenance

**Attack.** `sections/04_composition.tex`: *"stable rank 14.7--15.6."*
Recomputing `stable_rank_mean` across every hop of the four converged frN
seeds (`s1`–`s4`) in the cited Z-dump files gives a true range of
14.6487–15.5919 (minimum at seed 4, hop 21). 14.6487 rounds to 14.6, not
14.7. brief.md's own evidence row already states the more precise
"14.66--15.59," so the discrepancy is between the brief's own recomputed
value and the tex's rounding, not a fabricated number — but as printed, the
tex's lower bound is off by 0.05–0.1 from what the cited artifacts actually
contain.

**Supporting evidence.** Recomputed from
`experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump/t1_matrix_permutation_K8_frN_s{1,2,3,4}.json`,
field `stable_rank_mean` across all `M2_in_distribution`/`M3_held_out` hop
entries.

**What the paper would need to do to defuse this.** Change "14.7--15.6" to
"14.6--15.6" (or the brief's own "14.66--15.59").

---

### A7: Undisclosed RuntimeWarnings in the analysis pipeline when computing the entity-subspace decomposition

**Severity:** MINOR
**Type:** reproducibility

**Attack.** Running `analyze_zdump.py`'s `analyze_run()` on every one of the
five composition seeds (both the fr7 and the frN cells) emits ~60 numpy
`RuntimeWarning`s per file ("divide by zero encountered in matmul",
"overflow encountered in matmul", "invalid value encountered in matmul") in
`block_decompose()` and `synthetic_keys_from_pi()`. The function does have a
finiteness guard (`if not np.isfinite(M).all(): raise FloatingPointError`)
that would have stopped the pipeline had these warnings signaled real
corruption, and I independently confirmed the resulting numbers match the
paper's Table B and Appendix A exactly — so this is very likely benign
(denormalized-float BLAS noise on near-singular intermediate products), not
a correctness bug. But Appendix C's reproducibility claim ("a single
versioned script that asserts an md5 checksum for every source file it
loads, so a changed artifact fails the build rather than silently plotting
stale data") implies a clean, quiet pipeline; a reviewer who reruns the
script from the anonymized code link will see a wall of RuntimeWarnings with
no inline explanation, which reads as alarming even though the numbers check
out.

**Supporting evidence.** Reproduced directly: `DRY_RUN_BYPASS=1
.venv/bin/python -c "import analyze_zdump as az;
az.analyze_run('.../t1_matrix_permutation_K8_frN_s1.json')"` from
`matrix-thinking/chapter2/`, capturing 60 `RuntimeWarning`s per call.

**What the paper would need to do to defuse this.** Not a paper-text fix;
a code fix (suppress or explain the warning source in `analyze_zdump.py`)
or a one-line footnote in Appendix C noting the warnings are expected and
benign.

## Attacks I considered but decided were weak

- **"The 991-run M1 grid's Spearman ρ=1.0 is suspiciously perfect."**
  Checked: each `(d, K)` cell aggregates 8 seeds (`SEEDS_M1` in
  `run_overnight.py`), with refill rounds adding more; averaging over that
  many seeds legitimately smooths seed noise, and the underlying
  rank-tracks-K relationship is the expected behavior under a classical,
  well-established bound (Kohonen 1972, Anderson 1972), not a suspicious
  result. Not an attack.
- **"The paper cherry-picks the cleanest cell (d=8, K=4) as its razor-sharp
  headline."** Checked: the paper also reports the noisier d=16, K=12
  "ragged ramp" transition (§3, R2b) rather than hiding it. Not cherry-picked.
- **"h=21 is presented as testing a novel held-out depth when it collapses
  mod 8."** Checked: Appendix A already reframes h=21 explicitly as a
  numerical-stability probe, not a novel-target generalization claim, and the
  main text's phrasing ("probes {7, 21}") does not oversell it. The sibling
  paper makes the identical, correct reframing. Adequately handled — this
  is *why* A1 is concerning (the correct handling is duplicated, not that
  either handling is wrong).
- **"The abstract's Nichani et al. characterization ('rank-m memory storing
  roughly md associations') might be a mischaracterization of that paper's
  actual result."** This project's own CLAUDE.md records an already-verified
  version of this exact fact ("Under argmax decoding a rank-1 matrix can
  recover ≈d associations, Nichani, Lee & Bietti, ICLR 2025,
  arXiv:2412.06538") consistent with the draft's more general rank-m
  phrasing. Not pursued further; would need the actual paper to attack with
  confidence, and the two characterizations are consistent.
- **Figure rendering / layout correctness (do `fig_forcerank.pdf` and
  `fig_depth.pdf` actually show what the captions claim).** Out of scope for
  this content-adversarial pass; treated as a separate render-inspection
  gate's responsibility, not re-derived here.

## New citations you found that should be in Related Work

- **DeltaNet — Yang, Wang, Shen, Panda & Kim, "Parallelizing Linear
  Transformers with the Delta Rule over Sequence Length," NeurIPS 2024,
  arXiv:2406.06484.** The direct bridge the related-work section is missing:
  it is the paper that made rank-1-update fast-weight matrix memories
  practical at scale, and is the base method `siems2025deltaproduct`
  (already cited) explicitly extends with Householder products. Citing
  DeltaProduct without DeltaNet is like citing a follow-up without its
  antecedent — reviewers in this exact subfield (linear-attention/fast-weight
  state rank, which this EA's own related work engages via Nazari/Sun/
  Grazzi/Siems) will notice the gap immediately.
- **Gated Linear Attention — Yang, Wang, Shen, Panda & Kim, "Gated Linear
  Attention Transformers with Hardware-Efficient Training," ICML 2024,
  arXiv:2312.06635.** The other standard reference point for matrix-valued
  linear-attention state dynamics and gating's effect on effective state
  rank; sits naturally alongside the paper's Nazari/Sun citations on
  observational rank dynamics in pretrained linear-attention models.
- **Titans — Behrouz, Zhong & Mirrokni, "Titans: Learning to Memorize at
  Test Time," arXiv:2501.00663 (2025).** A neural long-term memory module
  framed explicitly around test-time associative-memory capacity —
  directly competes on the "matrix/vector state as an associative store
  with a capacity budget" framing this paper's introduction opens with, and
  is recent/prominent enough that its absence from Related Work is
  conspicuous for a NeurReps audience.
