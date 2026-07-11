# Defense Report — Round 1 — Response to "Three Bounds on a Null"

## Summary for the rebuttal agent

**0 DEFEND, 1 PARTIAL, 7 CONCEDE+FIX.** I re-derived the attack's load-bearing
numbers independently rather than trusting its prose: I pulled
`PHASE2B_SUMMARY.json` myself and confirmed its `trajectories` field has
exactly two UNRESOLVED corpus-level entries with no per-arm field (A1); I
loaded both `trajectory_*_phase2b.json` files and recomputed `det32` myself,
finding the same three CI-excluding-zero cells the attack reports (A2); I
reimplemented the F(2,8) test from scratch (regularized incomplete beta, no
scipy available) and got the same p≈0.0497/critical≈4.459 the attack reports
(A3); I fetched all three arXiv abstracts live and confirmed the
mischaracterization (A4) and both missing citations (A5) as described; I
loaded the raw `off_*.json` familiarization trajectories and recomputed the
step-1-to-5000 fall myself, getting 70.7–81.2% against the paper's reported
21.8–46.4% (A7); and I read `refs.bib` directly for A8. Every attack's
central factual claim survived my independent check. The one place I found
real pushback is A2: the paper's specific phrase "the one determinate
signal" refers to the pre-registered `holds()` compound condition, which
genuinely fires exactly once across all 20 primary checkpoints — not to the
looser "any CI excludes zero" reading the attack's framing implies — and a
structurally independent secondary (held-out-hop) readout corroborates the
same cell in the same direction, which is real evidence against pure
sampling noise, though it does not settle *why* the deviation occurred. That
earns A2 a PARTIAL rather than a CONCEDE, but the underlying disclosure gap
(the 40-test denominator, the 3-cell count, the checkpoint-clustering
question) is still real and needs an edit.

**The paper is submittable after fixes, and none of the required fixes need
new experiments** — every fix below is a framing/disclosure change (a
sentence, a caption clause, a citation, a bibtex field), except A6, where
the honest position is that the fully rigorous fix (a real-kernel positive
control) is new evidence and should be named as a limitation if it isn't
built before the deadline. The most consequential fix is A1: the paper
currently presents "4 (corpus, arm) contrasts, independently classified" as
if it were the untroubled output of the registered pipeline, when the
registered/build-time pipeline actually produced two UNRESOLVED
corpus-level verdicts and the fourth (TRANSIENT) reading came from a
harvest-time per-arm re-derivation the design registry itself calls a
"build-time scoping finding" — later formalized into code, but only after
being seen. This affects the abstract, Table 1, and two of the three
headline bounds, so it needs a prominent disclosure sentence, not a
footnote.

---

## Defenses

### A1: The paper's one non-null finding is a post-hoc reanalysis, not the pre-registered pipeline's output

**Disposition:** CONCEDE + FIX

**Response.** I read `PHASE2B_SUMMARY.json` directly myself (not the
attack's quotation of it):

```
{'openr1-mix-ext': {'outcome': 'UNRESOLVED', ...},
 'wikitext-mix-ext': {'outcome': 'UNRESOLVED', ...}}
```

Two entries, both UNRESOLVED, no per-arm field — matches the attack's claim
exactly. I then read `REASONING_LINK_DESIGN.md` §16.18.3 myself: the design
registry's own words are "`phase2_trajectory_analysis.analyze_corpus`...
computes only ONE hexachotomy classification per corpus... using the
**global** arm's own `holds_by_c` pattern as the corpus's representative
signal," that `per_token`'s own pattern is "**never consulted** for the
registered corpus-level verdict," and — critically — that the 4-way
breakdown reported in the paper is something "this section supplies," i.e.,
computed at harvest time, "Registered as a follow-up fix, not self-authorized
here." That is the design registry's own account of a harvest-time
re-derivation performed after the registered/build-time pipeline's actual
output (two UNRESOLVED verdicts) was seen.

I cannot defend this as anything other than what the attack says it is: the
unit of analysis reported in the paper ("4 (corpus, arm) contrasts,
independently classified," §4, Table 1) does not match what the registered
pipeline computed when Phase-2b actually ran. Two mitigating facts matter
for how to fix it, not whether to fix it: (1) the underlying `Δ` statistic
per `(arm, K, c)` — the numbers themselves — were part of the original
pre-registered machinery (`§16.16.5` defines `Δ_Lquery(arm, K, c)` per arm
from the start; only the corpus-level *classification* step was coarser
than the statistic it consumed); (2) the CI arithmetic was independently
re-derived by hand (`§16.18.2`) and later by an independent code
reimplementation, both matching the stored values to 4 decimal places, and
a `classify_arms` code fix was subsequently built and validated against the
*already-archived* per-cell data, reproducing the same four verdicts. This
is evidence the reading is not cherry-picked noise — but it is not evidence
that the paper's framing is honest about *when* and *how* the 4-way
breakdown was produced, which is the actual defect.

**Supporting evidence.** `experiment-runs/2026-07-08_phase2b/results/
PHASE2B_SUMMARY.json` (loaded and printed this session — `trajectories` has
2 keys, both `UNRESOLVED`, no `classification_by_arm`/`trajectories_by_arm`
field present in the archived file); `REASONING_LINK_DESIGN.md` §16.18.3
(quoted above); §16.18.9's `[Follow-up]` block (the code fix, "Registered
as a follow-up fix, not self-authorized here" — i.e., not part of the
original registered pipeline at harvest time); `main.tex` §4 and Table 1
("Three of four (corpus, arm) contrasts classify as unresolved... 18 cells,
4 contrasts | 3 unresolved, 1 transient") — grepped for any disclosure
language ("scoping," "representative," "global-arm," "build-time") and
found none in `main.tex` or any `sections/*.tex` file.

**What goes in the paper if this defense is accepted.** Add one disclosure
sentence to §4 (Behavioral Contrast) stating that the registered
corpus-level pipeline used the global arm as each corpus's representative
signal and classified both corpora UNRESOLVED, and that the per-arm
breakdown reported here (and in Table 1 and the abstract) is a disclosed
build-time re-derivation using the same classification primitives at a
finer granularity, later folded into the production code and validated
against the archived data. Add the equivalent to Appendix A's "Behavioral-
contrast classification" paragraph, since that appendix explicitly claims
to reproduce "the pre-registered gates and routing rules the main text
relies on" and currently omits this one. Also verify (build-time, zero GPU)
that `figures/figure-gen.py`'s checksummed manifest and code path for Fig 3
actually invoke the `classify_arms` per-arm path rather than the original
`analyze_corpus` top-level field, so Appendix C's reproducibility claim
("every number... computed from archived raw result files... not
summaries") holds for the 4-contrast table too.

---

### A2: The "single determinate signal" is not distinguishable from multiple-comparisons noise, and the raw data show an unaddressed shared-checkpoint alternative explanation

**Disposition:** PARTIAL

**Response.** I loaded both `trajectory_openr1-mix-ext_phase2b.json` and
`trajectory_wikitext-mix-ext_phase2b.json` myself and recomputed `det32`
across all 4×5=20 primary cells directly from the `raw` blocks (not from
the design doc's table). I get exactly the same three hits the attack
reports: `openr1×global@1000` (Δ=−0.203, CI [−0.402,−0.004]),
`wikitext×per_token@1000` (Δ=−0.168, CI [−0.301,−0.036]), and
`wikitext×per_token@2500` (Δ=−0.500, CI [−0.624,−0.376], the reported
transient). The attack's factual claim — three CI-excluding-zero cells, not
one, out of 40 uncorrected tests, with no stated denominator or correction
anywhere in the paper — is confirmed and is a real disclosure gap the paper
should close.

Where I push back: the paper's phrase "the *one* determinate signal" (used
in the abstract and in §4's "Sign discipline" paragraph) is not, on a
precise reading, a claim about `det32` alone — it is a claim about the
pre-registered **`holds()`** differential condition (`det32 AND NOT det20
AND |Δ32|>|Δ20|`, §16.16.5, fixed before any Phase-2b data existed). I
checked all 20 primary cells against this exact condition myself:
`openr1×global@1000` has `det20=True` too, so `holds=False`;
`wikitext×per_token@1000` has `det20=True` too, so `holds=False`; only
`wikitext×per_token@2500` has `det32=True, det20=False, |Δ32|>|Δ20|`, so
`holds=True`. There genuinely is exactly one cell, out of 20, where the
full pre-registered compound condition fires — the paper's narrowest claim
is defensible on its own terms. The paper's own text does distinguish "the
one determinate *signal*" (singular, = `holds()`) from "every determinate
*high-load reading*" (plural, = any `det32=True`) in the very same
sentence in §4 — but the two terms are similar enough, and neither is ever
defined precisely, that a reader cannot tell they denote different
statistical objects without doing what I just did. That ambiguity is a real
defect, separate from and smaller than the attack's "indistinguishable from
noise" framing.

On the "shared checkpoint artifact" alternative: I checked whether the two
`det32=True` hits at `c=1000` are consistent with a single shared
checkpoint-level artifact (e.g., the OFF cohort transiently reading low at
that step, which would flow into every contrast that subtracts it). They
are not obviously consistent with that story: `openr1×global@1000` fires
but `openr1×per_token@1000` does not (Δ=−1.003, CI spans zero) — same
corpus, same OFF reference values, different arm, different outcome. If OFF
itself were the anomaly at `c=1000` for openr1, both arm contrasts against
it should show the same pattern; only one does. Symmetrically,
`wikitext×per_token@1000` fires but `wikitext×global@1000` does not
(Δ=−0.136, CI spans zero) — same corpus, same OFF reference, different arm.
This asymmetry argues against the simplest version of the attack's
alternative explanation, though it does not rule out a more complex,
arm-and-corpus-specific training artifact, and I do not have a positive
account of what does explain the `c=1000` pattern. Separately, the
`wikitext×per_token@2500` cell (the actual transient) is independently
corroborated by a structurally different secondary readout (`§16.18.5`,
held-out hops 3–4) firing at the *same* cell and checkpoint, *same*
direction — real evidence that whatever is happening is a property of that
specific model's state at that checkpoint, not an artifact of one
readout's own sampling noise, even though it does not by itself
distinguish "true causal arm effect" from "some other checkpoint-specific
artifact of that particular training run."

**Supporting evidence.** Recomputed this session directly from
`trajectory_openr1-mix-ext_phase2b.json` and
`trajectory_wikitext-mix-ext_phase2b.json`'s `per_arm.{arm}.raw` blocks
(script run against the raw files, not the design doc's transcription);
`REASONING_LINK_DESIGN.md` §16.16.5 (the `holds()` condition, registered
before Phase-2b data existed); §16.18.5 (the secondary OOD readout firing
at the same cell/checkpoint).

**What goes in the paper if this defense is accepted.** Add, in §4: (1) the
denominator — "40 uncorrected 95% CI tests (4 contrasts × 5 checkpoints × 2
loads) are computed; three exceed zero at K=32, one (the reported
transient) also satisfies the pre-registered differential condition"; (2)
one clause distinguishing "determinate" (any CI-excludes-zero K=32 reading)
from the registered `holds()` condition the transient specifically
satisfies, so a reader is not left to infer the distinction; (3) one
sentence acknowledging the checkpoint-1000 pattern exists and that the
arm-asymmetry within each corpus argues against (without ruling out) a
simple shared-OFF-cohort artifact. No new experiment required.

---

### A3: The batch-effect gate's variance-ratio cutoff (4.0) is an undefended round number, and a proper F-test shows the observed ratio is only marginally more extreme than chance at these degrees of freedom

**Disposition:** CONCEDE + FIX

**Response.** I reimplemented the F-test from scratch this session (no
`scipy` available in this environment — wrote the regularized incomplete
beta function via a continued-fraction expansion and derived the F CDF from
it) rather than trust the attack's numbers, and got: variance ratio
4.4715 (matches `sd_old=1.1543`, `sd_new=0.5459` exactly); one-sided
`P(F(2,8) ≥ 4.4715) ≈ 0.0497`; one-sided 5% critical value `F(2,8) ≈
4.459`; two-sided 5% critical value `≈ 6.059`. This reproduces the attack's
numbers to 3+ significant figures independently. I also grepped the entire
design registry for any F-distribution derivation, citation, or simulation
calibration behind the "4.0" pin and found none — `variance_ratio > 4`
first appears as a bare threshold at line 9801 (MINOR-1), with the mean-
shift half of the same gate (`2×pooled_SE`) explicitly justified as a
Welch-style SE-of-difference formula while the variance-ratio half has no
analogous derivation anywhere.

I looked for a design-rationale defense and found a real, if limited, one:
a pre-pooling integrity gate whose job is to *prevent* silently pooling
heterogeneous cohorts is better served by a threshold that is *easier* to
trip than a properly calibrated 5% test, not harder — false-positive cost
(needlessly reporting cohorts separately) is much lower than false-negative
cost (a spurious pooled CI). A cutoff of 4.0, sitting just below the true
one-sided 5% critical value of 4.459, is consistent with that conservative
design intent. But this is a plausible ex post rationale I constructed, not
something in the design registry, and it does not change the fact that the
paper's language ("a pinned cutoff of 4.0," "twelve percent past the
cutoff") reads as more calibrated/precise than it is.

**Supporting evidence.** F-test reimplementation run this session (pure
Python, incomplete-beta via continued fraction, cross-checked against the
closed-form F↔Beta relation); `REASONING_LINK_DESIGN.md` line 9801 and
surrounding `§16.19.5` MINOR-1 text (grepped for "F-distribution," "F(2,"
"critical value," "calibrat" — zero hits tied to the variance-ratio
threshold).

**What goes in the paper if this defense is accepted.** Soften §5's
framing: replace "a pinned cutoff of 4.0" language with something like "a
pre-registered heuristic threshold (not a calibrated significance test; a
one-sided 5% F-test at these degrees of freedom would require ≈4.46, close
to the observed 4.47)" — this is honest, requires no new experiment, and
actually strengthens the paper's own "small-n variance imprecision" reading
in the same section by being explicit that the observed ratio sits right at
the edge of genuine significance either way.

---

### A4: Okpekpe & Orvieto (2025) is mischaracterized in Related Work

**Disposition:** CONCEDE + FIX

**Response.** I fetched arXiv:2508.19029's abstract live this session:
"we first demonstrate that, unlike standard transformers, the choice of
learning rate plays a critical role in the performance of modern recurrent
models... an issue that can severely affect reported performance in
previous works." The paper's stated primary contribution is an
optimization-instability confound, not a mechanistic account of *why*
recurrent models recall differently. `main.tex` §6's clause — "give a
positive causal account of recall differences in modern recurrent models"
— overstates and recasts this. I looked for a partial defense: the same
abstract does report that "recurrent and attention-based models exhibit
contrasting benefits when scaling in width as opposed to depth," which is
a recall-differences finding of a sort — so the citation is not baseless,
but the paper's specific characterization ("positive causal account...")
picks the wrong half of the paper to describe, and picks a half
(mechanism) that is not what the source's own abstract leads with.

**Supporting evidence.** arXiv:2508.19029 abstract, fetched live this
session via `export.arxiv.org/api/query?id_list=2508.19029`.

**What goes in the paper if this defense is accepted.** Rewrite the clause,
e.g.: "Okpekpe \& Orvieto (2025) show that optimization instability, not
architecture alone, explains much of the reported Transformer-vs-SSM recall
gap, and that recurrent and attention models trade off width against depth
differently; this paper's bounds concern a different axis (write-geometry
validity), not recall throughput or training stability." One sentence, no
new experiment.

---

### A5: Two directly on-point papers about DeltaNet/SSM state-tracking and permutation composition are missing from Related Work

**Disposition:** CONCEDE + FIX

**Response.** I fetched both abstracts live this session and confirm both
exist, are correctly attributed, and are exactly as on-point as the attack
claims. Grazzi et al. (arXiv:2411.12537, ICLR 2025): "We extend this result
to non-diagonal LRNNs such as DeltaNet. We prove that finite precision
LRNNs with state-transition matrices having only positive eigenvalues
cannot solve parity... LRNNs can learn any regular language when their
state-transition matrices are products of identity minus vector outer
product matrices, each with eigenvalues in the range [-1,1]." This is
DeltaNet-specific, is about exactly the state-tracking/permutation-
composition capability class this paper's bind/query task instantiates,
and offers a genuine competing theoretical account (an expressivity
ceiling tied to the specific eigenvalue range the frozen-bias arms under
test may or may not access) for §7's own "not bounded... cross-layer
hand-off" speculation. Merrill, Petty \& Sabharwal (arXiv:2404.08819, ICML
2024): "SSMs cannot express computation outside... TC0... they cannot
solve simple state-tracking problems like permutation composition." Also
confirmed. Neither appears anywhere in `main.tex` (`\citet`/`\citep` search
of `sections/06_related.tex`, which currently cites only
Zoology/Based/Okpekpe-Orvieto/Nichani). This is a real gap in a paper whose
whole task is a permutation-composition problem on a DeltaNet-family model.

**Supporting evidence.** arXiv:2411.12537 and arXiv:2404.08819 abstracts,
fetched live this session.

**What goes in the paper if this defense is accepted.** Add 1-2 sentences
to §6, ideally naming Grazzi et al. specifically next to the DeltaNet
family under test (the frozen-bias arms are not described anywhere as
using a negative-eigenvalue/Householder-product modification, so this
citation bears directly on whether a null was theoretically expected for
this exact configuration) and Merrill et al. as the general SSM-class
ceiling. Page budget is tight (§6 is budgeted 0.20 pages in `brief.md`) —
this may require trimming a clause elsewhere in Related Work, not adding a
paragraph.

---

### A6: No real-kernel positive control for the geometric readout — only a CPU-stub unit test on a hand-built matrix

**Disposition:** CONCEDE + FIX

**Response.** I read `reasoning_link_stage_minus1.py` lines 76-96 directly.
`test_item_2_hand_built_composition` builds a 4×4 diagonal matrix by hand
on CPU, applies `apply_state_power` and the cosine-recovery scorer to it,
and checks the output against known values — pure function-level math
correctness, with no model, no forward pass, no `fla` kernel, no
multi-layer extraction. I grepped the design registry and found no
"positive control" language anywhere, and no test where the readout is
applied to a real forward pass with an analytically-known target. This
project's own `CLAUDE.md` states the applicable lesson directly: "CPU-stub
self-test suites test logic only; real-kernel coverage needs a separate
narrow smoke of the PRODUCTION path" — a rule this same project already
learned the hard way once (fla 0.5.1's RMSNorm CPU-fallback gap). I cannot
disprove the attack's core claim; it is correct.

I looked for a partial mitigant and found a weak one only: the readout's
validity-gate machinery (premises iii/iv, the h=1 floor) *is* evaluated end
to end on real checkpoints via the real extraction path at all 366
readings across four structurally different instrument deployments and
four scales (14M–1.31B), and fails in a qualitatively uniform way
everywhere. That uniformity is at least mildly more consistent with a
genuine construct-validity gap than with a stochastic bug that would be
expected to manifest differently across such varied deployments — but a
*systematic* bug (e.g., a consistently wrong layer index or transpose
convention, this program's own sibling-campaign precedent for exactly this
failure mode) would also produce uniform failure everywhere, so this
argument does not actually discriminate between the two hypotheses. It is
not a real defense, just a note that the uniformity is not literally silent
on the question.

**Supporting evidence.**
`experiment-runs/2026-07-07_reasoning_link_phase1/reasoning_link_stage_minus1.py`
lines 76-96 (read directly this session); `CLAUDE.md`'s CPU-stub rule;
grep of `REASONING_LINK_DESIGN.md` for "positive control" (zero hits).

**What goes in the paper if this defense is accepted.** Minimum (framing
fix, submission-viable): add one limitation sentence to §7 stating that the
readout's math has been validated at unit level (hand-built matrices) but
not end-to-end against a real forward pass with a known target, so the
"instrument verdict, not refutation" reading rests partly on the internal
consistency of the failure pattern rather than a positive-control proof.
Stronger (new evidence, not a submission blocker given the venue's passed
deadline and late-window entry, but should be flagged as a named
camera-ready priority): construct one synthetic sequence, run it through an
actual trained checkpoint's real forward pass with keys/values built so the
correct h-hop answer is knowable in closed form, and confirm the extraction
pipeline recovers it before the next wave of this program treats a
genuine-checkpoint zero as diagnostic.

---

### A7: The task-familiarization "21.8 to 46.4 percent" fall is measured from checkpoint 250, not from training start, and the paper never says so

**Disposition:** CONCEDE + FIX

**Response.** I loaded all six `off_*.json` files directly and recomputed
both quantities myself: from checkpoint 250 to 5000, the six cells fall
21.8% to 46.4% (mean 35.9%) — matches the paper exactly. From step 1 (the
first logged training-loop record, present in every file's `trajectory`
array) to checkpoint 5000, the same six cells fall 70.7% to 81.2% (mean
77.5%) — more than double, confirming the attack's numbers exactly. The
paper's phrase "falls 21.8 to 46.4 percent... over 5,000 familiarization
steps" (Figure 2 caption) does read as measuring the full run, and it does
not. I looked for a defense and found only a mitigating fact, not a
disproof: the checkpoint-250 baseline is a defensible, non-cherry-picked
choice — it is the first of the five checkpoints where the geometric gate
is also evaluated (Figure 2's own dissociation comparison needs a
matched-checkpoint baseline), and the omitted step-1 number is *larger*,
not smaller, so there is no incentive to hide it; disclosing it would if
anything make the "the model learns the task" reading stronger. That
doesn't change that the caption needs the qualifier.

**Supporting evidence.** Recomputed directly this session from
`experiment-runs/2026-07-08_phase2_familiarization/results/off_*.json`
`trajectory` arrays (all 101 logged steps per file, `step=1` through
`step=5000` present in every file).

**What goes in the paper if this defense is accepted.** Add "(measured from
checkpoint 250, the first post-warmup reading, to checkpoint 5,000)" to the
§3 sentence and/or the Figure 2 caption. One clause, no new experiment.

---

### A8: Bibliography year inconsistency for Merity et al.

**Disposition:** CONCEDE + FIX

**Response.** Confirmed directly: `refs.bib` lines 50-55 list
`merity2017pointer` with `year = {2016}`. Pointer Sentinel Mixture Models
is an ICLR 2017 paper (2016 is only the arXiv v1 submission date); the
citation key itself already encodes 2017, so the `year` field is simply
inconsistent with the paper's own key. No defense available or needed.

**Supporting evidence.** `refs.bib` lines 50-55, read directly this
session.

**What goes in the paper if this defense is accepted.** Change `year =
{2016}` to `year = {2017}`. One-line bibtex fix.

---

## New citations found during defense

All three citations the attack surfaced were independently verified live
this session and should go in `refs.bib` regardless of which fix path §6
takes:

- **Grazzi, Siems, Franke, Zela, Hutter, Pontil, "Unlocking State-Tracking
  in Linear RNNs Through Negative Eigenvalues," ICLR 2025, arXiv:2411.12537.**
- **Merrill, Petty, Sabharwal, "The Illusion of State in State-Space
  Models," ICML 2024, arXiv:2404.08819.**
- **Jelassi, Brandfonbrener, Kakade, Malach, "Repeat After Me: Transformers
  are Better than State Space Models at Copying," arXiv:2402.01032 (ICML
  2024).** Not one of the three abstracts I was asked to verify against the
  live API, but I checked it anyway given the attack's mention of it: exists,
  correctly attributed, on-topic (fixed-latent-state capacity limits on
  copying/recall) — a reasonable third addition alongside Based/Zoology in
  §6's first sentence, lower priority than the two DeltaNet/SSM
  state-tracking papers above.

## Attack ordering note

- **A6 may be under-rated at SERIOUS.** It attacks the evidentiary basis for
  Bound 1 (the 366/366 zero-reading "instrument verdict" claim), which is
  the paper's single most load-bearing and otherwise most defensible
  result — the attack report's own summary calls Bound 1 "comparatively
  solid," but that solidity is exactly what A6 puts a hole in. This project's
  own `CLAUDE.md` treats this specific failure mode (CPU-stub-only coverage
  of a production kernel path) as a standing, previously-paid-for lesson,
  which raises rather than lowers how seriously a missing positive control
  should be weighted here. I'd escalate to CRITICAL or at minimum flag it
  as the attack the rebuttal agent should not let get resolved by a single
  soft sentence if there is any GPU time available before the deadline.
- **A2 may be over-rated at CRITICAL relative to its actual content once
  the `holds()` vs. `det32` distinction is drawn out** — see the PARTIAL
  disposition above. The paper's narrowest, most load-bearing claim ("the
  one determinate signal," meaning the cell satisfying the full registered
  differential condition) is factually correct and singular; what's missing
  is a disclosure of the looser 40-test/3-hit picture around it. I'd call
  this SERIOUS, not CRITICAL, once the fix (disclose the denominator,
  disambiguate the terminology) is made.
- **A1 is correctly rated CRITICAL** — it is the one attack I could not
  weaken at all on independent re-verification, and it touches the
  abstract, Table 1, and two of three headline bounds.
- **A4 is arguably over-precise at SERIOUS** — it is a single mischaracterized
  clause with a one-sentence fix and no effect on any of the paper's own
  results; I'd have filed it adjacent to A7/A8 rather than beside A5, but I
  would not push hard on this reclassification.

*Security note:* No embedded `<system-reminder>` blocks, fake tool-output
injections, or concealment instructions were encountered during this
defense session, in any file read or any command output (including the
`curl` calls to the arXiv API and the direct JSON loads of the raw
trajectory files).
