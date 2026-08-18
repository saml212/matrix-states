# compB drift/conditioning mechanism analysis

**Pre-registration:** `EXPERIMENT_LOG.md` "2026-08-18 #8" (committed
`d9b230f`, written and internal-archive-gated BEFORE any checkpoint was
touched). This document reports the executed analysis against that
pre-registration exactly — no band, threshold, or metric redefinition.

**Question:** does an individual compB seed's deep-composition
degradation (`retrieval24_acc` @ h=61, P1b teacher-forced regime)
correlate with a measurable property of that seed's trained
`entity_adapter`?

**Scripts (box `youthful-indigo-turkey`, all under `~/ncr_writecond/analysis/`):**
- `compb_drift_analysis.py` — main analysis (loads archive, reconstructs
  init, computes cond/drift, runs Spearman + permutation tests, writes
  `compb_drift_analysis_results.json`).
- `test_reconstruct.py` — standalone determinism/non-collapse probe used
  before committing to the main run.
- `run.log` — full stdout of the main run.
- `compb_drift_analysis_results.json` — full numeric output (per-seed
  table + all correlation stats), source for every number below.

Per the task scope, this markdown file is the only repo write; the
scripts and raw JSON stay on the box (paths above) rather than being
archived into `experiment-runs/` in this pass.

## Data hygiene

n = **18** compB seeds pass the registered filter
(`writecond_premise_REPL_compB_s*.json`, `ckpt_step == 20000`):
{1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}.

**Seeds 3 and 4 are excluded** — their archived writecond records read
`ckpt_step == 10000`, not 20000, even though both seeds' own training
runs (`mob_g3b31_compB_s{3,4}.json`) show `status=COMPLETED, step=20000`.
Cross-referencing `EXPERIMENT_LOG.md` "2026-08-18 #4" (the root-filesystem
incident) explains this: 12 cells, including these, died in `torch.save`
at their step-10000 checkpoint when disk hit 100%, were recovered and
requeued to `/ephemeral/reseed_ckpts/`, and finished to step 20000 — but
the `writecond_premise_REPL_compB_s{3,4}.json` eval records on disk are
stale, scored against the pre-incident step-10000 checkpoint. The
registered `ckpt_step == 20000` filter catches this correctly and drops
both seeds; that filter is doing exactly the job it was written for.

**This matters for leg (a):** the pre-registration's sighted observation
named seed s3 (retrieval 0.953, TPC 0.13-0.15) as one of the two best
performers motivating the "highest TPC ⇒ best, not worst, degradation"
claim. Verified directly: s3's stale record does show
`retrieval24_acc@h61 = 0.9531` and `target_pairwise_cos@h61 = 0.1518`
(matching the sighted citation almost exactly) — but that is the
step-10000 snapshot, not the seed's final step-20000 state, and there is
no step-20000 writecond record for s3 anywhere in the archive. Under the
registered filter this analysis correctly excludes s3, so **one of the
two seeds that originally motivated leg (a) cannot be cross-checked at
its true final checkpoint here.** Seed s6 (retrieval 0.973, TPC 0.10-0.13
across hops), the other star seed, *is* in the n=18 sample and reproduces
as reported below.

## Methodology

**Leg (a) target metric and predictors** (zero new computation): archived
`P1b` (teacher-forced) records, `retrieval24_acc`, `target_pairwise_cos`
(TPC), `o_pairwise_cos` (o_pc) at every archived hop (h=1, 13, 37, 61).

**Leg (b) conditioning** (new computation, ckpt loading only, no
training/GPU): `cond(entity_adapter)` = ratio of largest to smallest
singular value of the final `entity_adapter.weight` (25×768,
`nn.Linear(768, 25, bias=False)`), i.e. the standard SVD-based 2-norm
condition number generalized to a rectangular matrix
(`torch.linalg.svdvals` in float64). The pre-registration names
`cond(entity_adapter)` without specifying a formula; this is the
standard reading and is stated explicitly here since it is a judgment
call the pre-registration left open.

**Leg (c) drift** (NEW COMPUTATION, explicitly not archive retrieval,
per the pre-registration's own label — only final step-20000
checkpoints exist, there is no step-0 snapshot): `W_init` is
reconstructed by seeded re-instantiation of the adapter module alone,
CPU-only (`CUDA_VISIBLE_DEVICES=""`, `map_location="cpu"`, no training
step ever executed). Read directly from the pinned runner
`ncr_lm_wave1_runner.py` (md5 `9a93198b642242f512ff8489e32b0a53`,
verified against the pinned hash before use): `build_arm(vocab_size,
seed, device)` calls `torch.manual_seed(seed)` and THEN constructs, in
order, the ~98M-param `DeltaNetLM` backbone, the `NCREarlyLNModel` head,
and finally `NCRIntegration` (whose `__init__` builds
`entity_adapter = nn.Linear(768, 25, bias=False)`). Because all of these
draw from the same global PyTorch RNG stream, `entity_adapter`'s init
depends on everything constructed before it — reconstructing it
correctly requires replaying the *entire* construction sequence, not
just building an isolated `nn.Linear`. This analysis imports and calls
`build_arm` directly from the pinned runner (never reimplemented) so the
sequence is guaranteed identical to what the real training launch ran.
Drift is reported as `‖W_final − W_init‖_F / ‖W_init‖_F`.

**Reconstruction sanity gate (required by the task brief, run before
trusting any leg-c number):** spot-checked on seeds 1, 12, 20 —
(1) two independent reconstructions at the same seed are **bit-identical**
(`torch.equal` true in all 3 cases); (2) the reconstructed init is
**not** accidentally equal to that seed's final checkpoint weights (also
verified true in all 3 cases). Both checks passed for every spot-checked
seed; see `run.log` lines under "reconstruction sanity checks".

**Statistics:** Spearman ρ (`scipy.stats.spearmanr`, which also reports
scipy's asymptotic t-approximation p-value) plus a permutation p-value
(`scipy.stats.permutation_test`, `permutation_type='pairings'`,
two-sided). **Disclosure on "exact":** the task asked for exact small-n
p-values. A literal exact permutation p-value requires enumerating all
n! pairings of the two variables; at n=18 that is 18! ≈ 6.4×10^15
distinct permutations — combinatorially infeasible to enumerate, on any
hardware, in any practical time. This is **not executable as literally
"exact,"** so per the task's own stated fallback ("else compute
permutation p-values yourself") a high-resolution Monte Carlo
permutation test was run instead: 200,000 random resamples per
correlation, RNG-seeded for reproducibility. At 200,000 resamples the
binomial standard error on the reported p-value is ≲0.0011 for p near
0.5 and smaller for p near the tails — far more precise than needed to
place any of these results relative to the registered p<0.05 threshold.
scipy's asymptotic p-value is reported alongside every permutation
p-value as a cross-check; the two agree to within ~0.005 in every case
below, which is itself evidence the Monte Carlo estimate is stable.

## Leg (a) — geometry vs. degradation (NOT BLIND — confirmatory only)

Per hop, Spearman ρ of TPC and o_pc against `retrieval24_acc@h=61`
(n=18; p values are the Monte Carlo permutation p, asymptotic in
parentheses):

| hop | ρ(TPC, acc@h61) | p | ρ(o_pc, acc@h61) | p |
|---|---|---|---|---|
| h=1  | +0.077 | 0.761 (0.760) | +0.208 | 0.408 (0.408) |
| h=13 | +0.143 | 0.572 (0.573) | −0.081 | 0.746 (0.751) |
| h=37 | +0.369 | 0.135 (0.132) | +0.072 | 0.777 (0.776) |
| h=61 | **+0.436** | **0.073 (0.071)** | +0.323 | 0.190 (0.191) |

This **confirms the sighted, counter-intuitive direction**: within the
n=18 seeds available under the registered filter, seeds whose target
space is *more* pairwise-collapsed (higher TPC) tend to compose *better*
at depth, not worse — the opposite of the naive "collapse causes
degradation" story. The correlation strengthens with hop depth (0.08 at
h=1 → 0.44 at h=61) and is suggestive but does not clear p<0.05 at n=18
(closest at h=61, p≈0.07-0.073). o_pc shows the same sign but weaker and
non-significant throughout. **This is reported as confirmatory of a
prior sighted observation, never as an independent discovery** — the
gate agent had already seen this non-monotonic ordering before the
blind legs were registered, per the pre-registration's own disclosure.

## Leg (b) — adapter conditioning (BLIND)

ρ(cond(entity_adapter), retrieval24_acc@h=61) = **−0.155**, n=18.
p_permutation = **0.5365** (200,000 resamples); p_asymptotic = 0.5392.

**Registered bands:** SUPPORTS requires ρ ≤ −0.5 and p<0.05; NULL
requires |ρ| < 0.3; else PARTIAL.

**VERDICT: NULL.** |ρ| = 0.155 is well inside the NULL band and the
correlation is not remotely significant. How well- or ill-conditioned a
seed's trained adapter ends up carries essentially no information about
whether that seed composes at depth.

## Leg (c) — drift from seeded init (BLIND)

ρ(‖W_final−W_init‖_F/‖W_init‖_F, retrieval24_acc@h=61) = **+0.3006**,
n=18. p_permutation = **0.2265** (200,000 resamples); p_asymptotic =
0.2255.

**Registered bands:** SUPPORTS requires ρ ≤ −0.5 and p<0.05; NULL
requires |ρ| < 0.3; else PARTIAL.

**VERDICT (literal band): PARTIAL.** ρ = 0.3006 clears the |ρ| < 0.3
NULL cutoff by 0.0006 — a hair's width, essentially at the numerical
boundary of the registered band, well inside the Monte Carlo p-value's
own precision. **Read honestly rather than mechanically:** this is not
weak support for the drift-degradation mechanism. The registered
prediction was ρ ≤ −0.5 (more drift from init ⇒ worse deep retrieval);
the measured sign is **positive** — if anything, seeds whose adapter
moved further from its init composed *slightly* better, not worse — and
p=0.23 is nowhere near significant. A PARTIAL label produced by a
wrong-signed, non-significant correlation sitting on the NULL boundary
is evidence against the mechanism, not for it. The strict band verdict
is recorded as PARTIAL per the pre-registration's literal rule (not
redefined here), with this caveat stated plainly so it is not
mistaken for real support.

**Because leg (b) is a clean NULL and leg (c) is a non-significant,
wrong-signed near-NULL, neither blind leg supports the adapter-property
mechanism.** This is not the textbook "both legs read NULL" case the
pre-registration described verbatim (leg c's ρ sits 0.0006 over that
literal line), but it is functionally the same outcome: no blind signal
survives.

## Full per-seed table

Sorted by `retrieval24_acc@h=61` (worst → best):

| seed | acc@h61 | TPC@h61 | o_pc@h61 | cond(adapter) | drift ratio |
|---|---|---|---|---|---|
| 2  | 0.6172 | −0.0002 | 0.0433 | 6.752 | 0.9869 |
| 17 | 0.6172 |  0.0013 | 0.0081 | 6.004 | 0.9868 |
| 1  | 0.6484 |  0.0001 | 0.0079 | 7.767 | 1.0518 |
| 14 | 0.6680 |  0.0716 | 0.0559 | 5.058 | 1.1187 |
| 11 | 0.6797 | −0.0001 | 0.0116 | 6.543 | 0.9780 |
| 18 | 0.6797 |  0.0199 | −0.0011 | 6.123 | 0.9527 |
| 15 | 0.6836 | −0.0012 | 0.0500 | 6.626 | 0.9781 |
| 12 | 0.6875 |  0.0004 | 0.0155 | 6.546 | 0.9651 |
| 20 | 0.6914 |  0.0493 | 0.0621 | 5.636 | 1.1474 |
| 8  | 0.7227 |  0.0010 | 0.0221 | 4.995 | 1.0631 |
| 5  | 0.7266 |  0.0026 | 0.0267 | 6.494 | 0.9986 |
| 16 | 0.7344 |  0.0012 | 0.0141 | 5.889 | 1.0806 |
| 9  | 0.7383 |  0.0009 | −0.0029 | 6.121 | 1.0079 |
| 13 | 0.7500 |  0.0156 | 0.0362 | 4.863 | 1.1464 |
| 10 | 0.7617 |  0.0084 | 0.0332 | 5.506 | 1.0220 |
| 7  | 0.7773 |  0.0145 | 0.0688 | 5.132 | 0.9676 |
| 19 | 0.8203 |  0.0066 | 0.0204 | 7.470 | 1.0623 |
| 6  | 0.9727 |  0.1274 | 0.0814 | 6.951 | 1.1174 |

(TPC/o_pc at all 4 hops, plus `w_init`/`w_final` Frobenius norms and the
resolved checkpoint path per seed, are in
`compb_drift_analysis_results.json` on the box.)

**Reading the ranking directly (the "scatter description"):** cond and
drift both bounce around with no visible trend as accuracy rises — the
worst seed (s2, 0.617) has cond 6.75, mid-pack; the best seed (s6, 0.973)
has cond 6.95, also mid-pack; the two highest-cond seeds (s1 at 7.77,
s19 at 7.47) sit at accuracies 0.648 and 0.820 — one low, one high. Drift
is similarly scattered: the two highest-drift seeds (s20 at 1.147, s13 at
1.146) sit at accuracies 0.691 and 0.750 — both mid-range, not extremes;
the lowest-drift seed (s18 at 0.953) sits at 0.680, also mid-range. Only
TPC shows a visible pattern: it is near-zero for every seed except s6,
where it jumps an order of magnitude to 0.127 — and s6 is also the single
best-performing seed. This single-seed-driven pattern is exactly what
produces leg (a)'s ρ≈0.44 at h=61 and is consistent with (not
independent proof beyond) the original sighted observation.

## What this establishes and what it does not

**Established:** at n=18, neither the trained adapter's conditioning nor
how far it drifted from a faithfully-reconstructed seeded init explains
compB's seed-to-seed spread in deep-composition retrieval. The
mechanically obvious story — "an adapter that moves further from its
init, or that ends up worse-conditioned, degrades at depth" — is not
what these two blind tests show; leg (b) is a clean null and leg (c) is
non-significant and wrong-signed relative to that story.

**Not established:** what *does* explain the spread. Per the
pre-registration's own stated contingency for a null-heavy outcome, this
pushes the mechanism question toward the **embed factor** that
`NCR_REAL_LM_DESIGN.md` §G3-B31 R2 already implicates (freezing closes
only the adapter factor; the embed factor stays open, receiving
aux-only gradient via the o-path at norm 110.13) — a hypothesis this
analysis did not test and makes no claim about beyond naming it as the
next place to look. Leg (a)'s positive TPC association is confirmatory
of a prior sighted observation only, not a tested mechanism, and is
driven substantially by a single seed (s6) at n=18 — not strong enough
to build on without a larger or independently-drawn sample, and its
strongest historical exemplar (s3) is not verifiable at its true final
checkpoint under the registered filter (see Data hygiene above).

**All three legs were executed; none had to be skipped.** The one
executability caveat is the "exact" p-value language (see Methodology):
literal exhaustive permutation enumeration is infeasible at n=18 and was
replaced with a 200,000-resample Monte Carlo permutation test, disclosed
explicitly rather than silently substituted.

**Power caveat:** n=18 with ρ in the 0.15-0.44 range is underpowered to
rule out a real but moderate effect in either direction on any of the
three legs. "No blind support" is the honest finding here — not "adapter
geometry is proven irrelevant." A future test aiming to settle this
would need either a larger n or a direct causal intervention (e.g.
freezing the embed factor and re-measuring) rather than another
correlation over the same 18 seeds.
