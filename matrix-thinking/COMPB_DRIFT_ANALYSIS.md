# compB drift/conditioning mechanism analysis — RESULT: both blind legs fail to support the hypothesis

**Executed 2026-08-18** per the pre-registration in `EXPERIMENT_LOG.md`
2026-08-18 #8 (committed `d9b230f`, before any checkpoint was touched).
Analysis agent wrote and ran the scripts; it ended its turn before
reporting, so this write-up is the coordinator's, from the completed
raw output. Artifacts: `experiment-runs/2026-08-18_compb_drift/`
(script, reconstruction test, run log, results JSON — repo + SSD).

## What was asked

Does an individual compB seed's deep-composition degradation
(`retrieval24_acc` @ h=61, exact-write P1b regime) correlate with a
measurable property of that seed's trained `entity_adapter`?

## Data hygiene (worked as designed)

- n = **18** compB seeds at `ckpt_step == 20000`.
- **Seeds 3 and 4 were EXCLUDED at `ckpt_step == 10000`** — leftovers
  from the 2026-08-18 root-filesystem incident, caught by the
  registered step guard rather than silently averaged in. This is the
  guard the COMPD audit demanded, doing exactly its job.
- Init reconstruction sanity-checked: re-instantiation is
  deterministic (bit-identical across two draws) AND differs from the
  final checkpoint — i.e. it is a real init, not an accidental copy.

## Results

| leg | quantity | sightedness | Spearman ρ | permutation p (200k) | registered verdict |
|---|---|---|---|---|---|
| (b) | adapter condition number | **BLIND** | **−0.155** | 0.537 | **NULL** (|ρ| < 0.3) |
| (c) | drift ‖W_final−W_init‖_F/‖W_init‖_F | **BLIND** | **+0.301** | 0.226 | **PARTIAL** |
| (a) | `target_pairwise_cos` | not blind | +0.436 | 0.073 | confirmatory only |
| (a) | `o_pairwise_cos` | not blind | +0.323 | 0.190 | confirmatory only |

Per-seed values (cond, drift ratio, retrieval@h61) are in the results
JSON and the run log; ranges: cond 4.86–7.77, drift ratio 0.953–1.147,
retrieval 0.617–0.973.

## Verdict — stated plainly

**Neither blind leg supports the adapter-property mechanism.**

- **Leg (b) is NULL outright**: how well- or ill-conditioned the
  trained adapter is carries essentially no information about whether
  that seed composes at depth (ρ = −0.155, p = 0.54).
- **Leg (c) is PARTIAL, and its SIGN IS OPPOSITE to the registered
  prediction.** The pre-registration predicted ρ ≤ −0.5 (more drift →
  worse retrieval); the measurement is ρ = **+0.301** (more drift →
  *better* retrieval), not significant. A PARTIAL label in the wrong
  direction is not weak support for the hypothesis — it is evidence
  against it, and is recorded as such.
- **Leg (a) confirms the sighted observation** it was registered to
  confirm: the correlation with target-space collapse is **positive**
  (ρ = +0.436, p = 0.073) — seeds whose target space is *more*
  collapsed compose *better*, the reverse of the naive reading. Not
  significant at n = 18, and NOT claimed as a discovery: a prior agent
  had already seen this ordering, which is why the pre-registration
  labelled this leg confirmatory.

## What this establishes and what it does not

**Established:** compB's seed-to-seed spread at depth is *not*
explained by the trained entity adapter's conditioning, nor by how far
it moved from initialization. The obvious mechanical story — "letting
the adapter drift wrecks the geometry, and wrecked geometry fails at
depth" — is not what the data show.

**Not established:** what *does* explain the spread. Per the
pre-registration's own contingency, this pushes the mechanism question
toward the **embed factor** that `NCR_REAL_LM_DESIGN.md` §G3-B31 R2
already implicates (the freeze closes the adapter factor only; the
embed factor stays open, receiving aux-only gradient norm 110.13 via
the o-side path). That is a hypothesis this analysis did not test.

**Unchanged:** the freeze effect itself. Whether or not we can explain
*why*, the arm-level result stands — frozen adapters compose at depth
(median 0.996–1.000 across both aux settings) and trainable ones do
not (0.279 cosine / 0.707 contrastive), with complete separation and
exact p = 1.55e-04 in the pre-registered cosine contrast.

**Honest note on power:** n = 18 with ρ ≈ 0.3 is underpowered to
resolve a moderate effect; leg (c) is "no support," not "proven
absent." A future test wanting to settle it should pre-register a
larger n or a direct intervention (e.g. freeze the embed factor and
re-measure) rather than another correlation on the same 18 seeds.
