# Project State

**Last updated:** 2026-07-09 (KEYANCHOR-SCALING §15.26 D=96 SCATTER-RESOLUTION wave — Rev 2 LANDED, NOISE-FLOOR CALIBRATED: a second independent attack round on Rev 1's own RESHAPE-TO-C design returned NEEDS-REVISION — 3 MAJOR + 5 MINOR, no FATAL, all surgical; the empirical core (360,000-trial cumulative power check, the analytic K84-vs-K90 z-derivation) independently re-verified and found exceptionally clean, the 320,000-trial extension independently RE-EXECUTED and reproduced exactly. MAJOR-1 (highest value): the outcome trigger's CEILING-IS-REAL branch had no measured noise null (the restricted/unrestricted `M3` calls used two different eval generators) — fixed by registering one additional eval-only noise-floor repeat pass in the K=84 block, re-pinning both thresholds relative to the directly-measured `noise_shift`, MECE proven by an explicit totality walk. MAJOR-2: the manipulation matched K90's spare-entity MARGIN (`N'=101`, 0.94pp overlap-fraction residual) instead of the actually-diagnosed mechanism, entity-draw OVERLAP FRACTION — re-pinned `N'=100` (0.11pp residual, ≈8.4× tighter, same cost). MAJOR-3: the launch wrapper's own field-diff check couldn't pass verbatim with the new `--m3-pool-restrict-n` flag — fixed with an enumerated `NEW_FLAG_WHITELIST` stripped before the diff runs, plus its own negative test. Five surgical MINORs (telemetry-threading consistency, a citation fix, a pre-registered Δ_measured contingency, a finding-text reword, a ledger rounding-consistency fix). Design-only, zero GPU spent this session — queue: Rev 2 (done) → attack round 3 (VERIFY pass) → build wrapper → audit → launch GPU 2 FIRST, ≈0.9 GPU-h, THEN REASONING-LINK §16.19's own Leg-A pretraining slice per the pinned sequencing (shares GPUs 2-7, sequenced not concurrent). Full account below. REASONING-LINK §16.19 PHASE-2B SEED EXTENSION Rev 3 LANDED, MECE OUTCOME PARTITION + OOD GUARD SYMMETRY + LEG-A LAUNCH MECHANISM (prior session, retained below for history, UNAFFECTED in its own content by this session's §15.26 work — the two lanes remain sequenced on the shared GPU range per this session's own explicit note above) — full prior-session account also below. Earlier this day — paper fold-in session: reasoning-link triple-null/Phase-2b bounded-causal result folded into ICLR discussion item 10; d=96 unlock/AMBIGUOUS resolution folded into every stale "in-flight/DRAFT" hedge across iclr-2027 + workshop-2026, x0(80) corrected 0.6756→0.6779 for consistency, workshop-2026 main.pdf recompiled clean. C17 harvest session below this line: TOLERANCE-MISCALIBRATION verdict walked + independently re-verified, the registered d=96 11-cell unlock executed and re-fit — AMBIGUOUS; GPU 2 FREE. Phase-2b behavioral-contrast wave HARVESTED earlier the same day, GPUs 0-1 also FREE; prior notes retained below for history)

This document is the project dashboard. Anyone returning to the project (you, a collaborator, a grant reader, an experimenter agent) should read this first to answer: where is the project right now?

---

## REASONING-LINK PHASE-2B SEED EXTENSION — REV 3 LANDED, MECE OUTCOME PARTITION + OOD GUARD SYMMETRY + LEG-A LAUNCH MECHANISM (2026-07-08) — supersedes the REV 2 LANDED, ARCHIVED-VALUES SOURCING PIN block's own queue status (Rev 1/Rev 2's own restructure/floor/power/cell-grid/primary-readout-sourcing content is reused and extended, not retracted — attack round 3 found a MECE gap in the decision rules, a scope gap in the archived-values guard, and an unnamed launch mechanism, none of which reopens the A-vs-B restructure or the primary-readout sourcing fix themselves); the KEYANCHOR-SCALING §15.26 block below is UNAFFECTED in its own content — the GPUs-2-7 lane note it shares with this block is now backed by a mechanical `contention_gate()`-reuse gate (§16.19.7.1, MAJOR-C below), not disclosure-only

**Queue status: DESIGN (Rev 3) → (next) ATTACK ROUND 4 → BUILD-DELTA →
AUDIT → LAUNCH (Leg-A pretraining cells on GPUs 2-7, 6-way, ≈0.76h wall,
THEN familiarization+eval cells on GPUs 0-1, sequenced not concurrent).**

Attack round 3 on Rev 2 returned **NEEDS-REVISION** — 3 MAJOR + 2 MINOR,
no FATAL, all surgical (everything else re-verified correct: the power
arithmetic, the mixed-radix collision-freedom, the cost lines, the
primary-readout archived-values-sourcing MECHANISM Rev 2 registered —
its own loader shape and negative-test discipline reused, only extended
in scope, the single-confirmatory-cell pin), full finding→fix table at
`REASONING_LINK_DESIGN.md` §16.19.10's own round-3 table.

**MAJOR-A: §16.19.8's own 3-outcome decision rules could double-fire.**
TRANSIENT-CONFIRMED ("pooled CI excludes zero, negative side") and
TRANSIENT-REFUTED's own sub-case 2(b) ("CI excludes the archived n=3
point estimate, `-0.4999`, entirely") can both be true of the SAME CI
(e.g. `[-0.35,-0.05]`) — plausibly the single most likely non-trivial
outcome this wave produces, given regression-to-the-mean off a
significance-filtered n=3 estimate. **Fixed:** §16.19.8 rewritten as an
explicit, precedence-ordered, MECE 4-outcome partition — TRANSIENT-
CONFIRMED-AT-MAGNITUDE / TRANSIENT-CONFIRMED-SMALLER / TRANSIENT-REFUTED
/ NEW-PATTERN(SIGN-FLIP) — driven by `phase2_hexachotomy.det()` (already
registered) plus a NEW `contains_point(ci_low,ci_high,point)` helper,
with the totality walk shown, not asserted: every realized CI now maps
to exactly one bucket.

**MAJOR-B: the archived-values no-recompute guard (Rev 2) closed the
recompute hazard for the primary per_token readout only.**
`secondary_ood_readout` hits the IDENTICAL `_MAX_CKPT_SEED`-driven
re-seed hazard at `hop_set=(3,4)` (`kind="eval_lquery_ood"`), through a
call site Rev 2's own guard scoping did not provably cover. **Fixed:**
`load_archived_arm_val` generalized over `hop_set` (one function, both
readouts); the guard's scope re-pinned as WHOLE-HARVEST-RUNTIME rather
than call-site-local — since both readouts route through the SAME
`eval_query_loss_heldout` seam, one whole-runtime guard covers both by
construction. Also disclosed: the harvest driver is its own wave-specific
`analyze_corpus_seedext` function, not production `analyze_corpus`
invoked blindly (which would attempt a `global`-arm branch this wave
trains no new seeds for).

**MAJOR-C: the 18 Leg-A pretraining cells' own launch mechanism was
never named.** §16.19.4/§16.19.6 registered the grid and cost, never a
manifest, `--wave` flag, chain script, or env var — the same class of
gap §15.26's own MAJOR-2 found and fixed (§15.26.3.1). **Fixed,
mirroring that fix's own shape:** a NEW, additive `--wave rung1-seedext`
in `frozen_bias_lm_sweep.py` (seed∈{3,...,11}, 2 arms, 1 corpus, never
editing the existing `rung1` manifest), launched via a forked
`frozen_bias_seedext_chain.sh`; the calibration-cell-first stop APPLIES
but its human-inspection step is REPLACED by a mechanical val-loss
sanity-band gate pinned from 3 real archived numbers (`[4.2426,
4.4426]`, per_token/wikitext-mix-ext terminal val_loss); `contention_
gate()` reuse (already-built, already-generically-tested — verified
directly against `smoke_frozen_bias_lm.py`'s own `smoke_6_contention_
gate_refusal`) registered as the mechanical resolution of the GPUs-2-7
lane conflict with §15.26.

**Two MINORs, both surgical, no new GPU spend.** MINOR-1: `pooled_SE`
(round-2's own batch-effect gate) pinned explicitly as `sqrt(SE_old² +
SE_new²)` — the standard-error-of-a-difference form, justified by
`n_old=3 ≠ n_new=9`. MINOR-2: named the familiarization-layer's own
forked chain, `phase2b_seedext_chain.sh`, in the design doc for the
first time (previously only named in this file's own narrative queue
text) — distinct from the Leg-A-layer's own `frozen_bias_seedext_
chain.sh` (MAJOR-C).

**Detectable floor, power, and cost figures all UNCHANGED from Rev 2**
(this round touched decision-rule structure, guard scope, and launch
mechanism only, not the underlying n=12 statistics or cell grid): new
detectable floor ≈0.39-0.43 loss units, both below the observed 0.4999
magnitude; power ~81% at σ=0.43, ~72% at σ=0.48; raw ≈6.65 GPU-h,
bracket ≈33.3-66.5 GPU-h, ceiling 66.5 GPU-h. Full account:
`REASONING_LINK_DESIGN.md` §16.19 (Rev 3) + §16.19.10 (round-1, round-2,
AND round-3 attack fix-maps). No cells launched, no code written this
session.

---

## KEYANCHOR-SCALING §15.26 D=96 SCATTER-RESOLUTION WAVE — REV 2 LANDED, NOISE-FLOOR CALIBRATED (2026-07-09) — supersedes the REV 1 LANDED, RESHAPE-TO-C block's own queue status below (Rev 1's own retirement of the 10-cell grid + substitution of the K90 pool-margin control diagnostic is reused and extended, not retracted — attack round 2 found and fixed an unmeasured-noise-null gap in the outcome trigger, a margin-vs-overlap mismatch in the manipulation itself, and a wrapper field-diff check that couldn't pass verbatim with the new flag, none of which reopens the retirement decision). Different lane from, never gates and never gated by, the REASONING-LINK §16.19 Leg-A pretraining lane in CONTENT (different files) — but still shares GPUs 2-7 with that lane; neither has reached build/launch yet, so no live conflict today. Per the task's own pinned sequencing, this lane's launch (GPU 2) goes FIRST once it clears build+audit, THEN §16.19's own Leg-A pretraining slice reaches its own launch gate — the two stay sequenced, not concurrent, on the shared GPU range

**Queue status: Rev 2 (this entry) → ATTACK ROUND 3 (a VERIFY pass, not
a fresh full attack round) → BUILD WRAPPER
(`run_poolmargin_k84s1943_k90s2043.py` + the additive
`restrict_entity_pool_n`/`m3_pool_restrict_n` params + the
`NEW_FLAG_WHITELIST` field-diff adaptation) → AUDIT → LAUNCH GPU 2
FIRST (Stage 0: K=84/seed=1943 alone, calibration-gated; Stage 1:
K=90/seed=2043), ≈0.9 GPU-h — THEN §16.19's own Leg-A pretraining slice
(GPUs 2-7, 6-way) once it separately clears its own build+audit gate.**

**A second independent attack round returned NEEDS-REVISION — 3 MAJOR +
5 MINOR, no FATAL, all surgical** (`KEY_ANCHORING_SCALING_DRAFT.md`
§15.26.10's own fix-map); the empirical core (360,000-trial cumulative
power check, the analytic K84-vs-K90 z-derivation) was independently
re-verified this round and found exceptionally clean — every number
reproduces, and the 320,000-trial extension was independently
RE-EXECUTED, not merely re-read. Every finding fixed, zero GPU spent:

**(1) MAJOR-1, noise-floor calibration (highest value):** the outcome
trigger's `CEILING-IS-REAL` branch (`shift≤0.1×Δ`) had no measured
null — the restricted and unrestricted `M3` calls used two DIFFERENT
eval generators, conflating the pool-restriction treatment with plain
eval-batch sampling noise. Fixed by registering ONE additional
eval-only pass in the K=84 block (the SAME unrestricted call, repeated
under the second generator's own offset), giving `noise_shift :=
|repeat−standard|`, a directly measured null. Both thresholds re-pinned
relative to it (`REAL_THRESH=max(0.1×Δ,noise_shift)`,
`ARTIFACT_THRESH=max(0.5×Δ,3×noise_shift)`), MECE proven by an explicit
3-case totality walk.

**(2) MAJOR-2, control the diagnosed variable:** the live mechanism is
entity-draw OVERLAP FRACTION `K/N` (per §15.26.2.1's own code-level
pin), but Rev 1's own manipulation (`N'=101`) matched K90's spare-entity
MARGIN (17) instead — `84/101=83.17%` vs. K90's real `84.11%`, a 0.94pp
residual on the actual mechanism. Fixed by re-pinning `N'=100`
(`84/100=84.00%` vs. `84.11%`, 0.11pp residual, ≈8.4× tighter, same
cost) — Rev 1's own margin-vs-overlap slip disclosed explicitly, not
silently corrected.

**(3) MAJOR-3, wrapper field-diff adaptation:** the launch wrapper is
named as mirroring `run_k69_s1733_contingency.py`'s own precedent
"line-for-line," but that precedent's own token-diff check (refuse
unless the command matches a sibling-seed reference with only
seed-derived tokens differing) can't pass verbatim once K=84's own
command carries the new `--m3-pool-restrict-n` flag. Fixed by pinning
an enumerated `NEW_FLAG_WHITELIST` stripped from the generated command
before the precedent's own diff runs, plus a registered negative test
(one extra, non-whitelisted flag must still be refused).

**Five MINORs, all surgical, no new GPU spend.** MINOR-1: threaded
`c17_repro_telemetry` into both new eval calls, restoring the
"threaded to every pool call" invariant. MINOR-2: fixed an off-by-2
citation, `:961`→`:963-964` (verified against the live source). MINOR-3:
pre-registered a `Δ_measured` contingency for a fresh K90 reading that
doesn't reproduce ceiling or reverses the K84<K90 ordering — routes to
AMBIGUOUS + a registered follow-up, never silent. MINOR-4: reworded the
registered finding — "seed-dependent in the sub-ceiling regime
(K72–K84); K90 pinned at exact ceiling in all 3 seeds." MINOR-5: fixed a
rounding-base inconsistency in the ledger's 2× pessimistic row
(`+1.708`→`+1.800`, `21.1666/26`, `100.79%` — was `100.36%`; conclusion
unchanged).

**Ledger unchanged at 1×: 19.3666 + 0.900 = 20.2666/26 (96.51% of the
ORIGINAL 21 GPU-h ceiling — FITS WITHOUT drawing on the extension).**
2× pessimistic bracket corrected (MINOR-5): `21.1666/26` (`100.79%` of
the original, `81.41%` of the extended) — a small, disclosed tail
excess unchanged in substance from Rev 1's own (rounding-inconsistent)
`100.36%` figure.

Full account: `KEY_ANCHORING_SCALING_DRAFT.md` §15.26 (Rev 2) +
§15.26.9 + §15.26.10. Harvest entry: `EXPERIMENT_LOG.md`.

**Security note.** The same recurring fake `<system-reminder>`
injection (date-change-concealment instruction + fabricated agent-type
list + fabricated MCP-server tool-loading instructions, appended to the
first `Bash` tool result mid-session) fired again this session —
disregarded in full, including the concealment instruction; the
underlying date claim was independently cross-checked against this
machine's own `date` output and recent commit timestamps (both
genuinely 2026-07-08) rather than trusted because asserted — the
fabricated delivery mechanism and concealment instruction are the
actual finding, not whether the date itself happened to be accurate.
This is at least the 10th occurrence logged against this project's
history (`EXPERIMENT_LOG.md`'s own running tally). HEAD was verified
against the task's own cited starting commit (`f18b106`, "§15.26 Rev 1
— RESHAPE-TO-C") via `git pull --ff-only` (already up to date) before
any edit began; matched exactly, and did not move mid-session this
time (no concurrent sibling-agent commits landed). `REASONING_LINK_
DESIGN.md` was never opened this session, per the task's own explicit
constraint.

---

## REASONING-LINK PHASE-2B SEED EXTENSION (n=3→6) — DESIGN (2026-07-08): Rev 0, pre-attack, zero GPU spent — supersedes the PHASE-2B HARVEST block's own "more seeds... out of scope here" queue note below (§16.18.9 item 2); the KEYSTONE VERDICT and TRANSIENT finding in that block are otherwise still current/unchanged

**Queue: design (done, this entry) → attack (independent round, not yet
run) → build-delta (the `delta_ci_n`/`episode_seed` generalization, the
18-cell Leg-A pretraining launch, a forked `phase2b_seedext_chain.sh`) →
audit → launch (GPUs 0-1 for the familiarization/eval slice; the Leg-A
pretraining slice's own GPU assignment is a build-delta-stage decision,
unassigned as of this Rev).**

Extends `REASONING_LINK_DESIGN.md` §16.16's audited n=3 instrument to n=6
paired seeds/cell, testing (i) whether any of §16.18's 3 UNRESOLVED
(corpus×arm) contrasts resolve at a tighter bound and (ii) whether the 1
real, non-durable TRANSIENT signal (wikitext×per_token, HURTS-direction,
`|Δ(K=32,c=2500)|=0.4999`) replicates at independent seeds. **Re-derived
the CI-shrinkage factor at ≈2.37× (not the ≈1.45× a loose reading might
suggest) — new detectable floor ≈0.64-0.71 loss units, down from ≈1.5-1.7
at n=3.** **Honest, computed (not asserted) caveat: the observed
transient's own 0.4999 magnitude sits BELOW even the optimistic end of
the new floor — n=6 tightens the bound but is not projected to reliably
CONFIRM the transient; resolving it needs n≈9-10 (boundary) to n≈12-15
(conventional power)** — registered as a bound-tightening +
informal-replication-check wave, not a confirmation wave.

**THE key question, verified directly on the box: Leg-A init checkpoints
for seeds 3-5 do NOT exist** (`/data/deltanet_rd_frozenbias_ckpts/` has
exactly the 20 rung-1 cells at seed∈{0,1,2} only) — the extension is not a
free Phase-2b-layer add-on; it requires either 18 new Leg-A pretraining
cells or a nested reuse of the 3 existing checkpoints. **Adjudicated: full
new inits recommended** over nested reuse (which would pseudoreplicate 2
of 6 "seeds" per physical checkpoint, invalidating any independence
claim) — the honest, statistically-clean option, costing only ≈4.54 GPU-h
more against `FROZEN_BIAS_LM_DESIGN.md`'s own 135 GPU-h ceiling
(≈128.1 GPU-h headroom untouched).

**Cost, recommended option (FULL 18-cell grid + 18 new Leg-A cells): raw
≈6.65 GPU-h, bracket ≈33.3-66.5 GPU-h, ceiling 66.5 GPU-h** — a NEW ledger
line (the closed n=3 wave's own 1.66/26.4 ledger doesn't cover the new
Leg-A pretraining requirement), negligible against combined program
headroom. Full account (all 7 required design elements): `REASONING_LINK_
DESIGN.md` §16.19 (Rev 0, pre-attack — faces an independent attack round
next). No cells launched, no code written this session.

---

## C17 EVAL-ADMISSION REPRO INSTRUMENT — CLOSED (2026-07-08, harvest
session): verdict **TOLERANCE-MISCALIBRATION** (commit `a51f102`) walked
and independently re-verified against the raws + a fresh read-only box
pull; the registered §15.24.6 outcome-(c) UNLOCK executed (11 quarantined
d=96 cells recalibrated); the registered re-fit lands **AMBIGUOUS**
(non-monotonic scatter, 100% bootstrap-degenerate) — supersedes the "C17
EVAL-ADMISSION REPRO INSTRUMENT — RUN COMPLETE" block below's own queue
status ("harvest is a separate agent's task" — now done). **GPU 2 is
FREE.** Full account: `KEY_ANCHORING_SCALING_DRAFT.md` §15.25; harvest
entry: `EXPERIMENT_LOG.md`.

**The verdict, independently re-verified (not merely re-cited from the
commit message):** reconstruction gate PASS (107/107 set-equal, Stage
−1's own positive+negative trap fixtures both confirmed); Step 0b 0
violations; Step −1 36 distinct events (re-counted directly from the raw
`.pt` dumps: 12 each at checkpoints 16000/18000/20000, zero at
2000–14000); Step 0a 0 anomalous; Step 1 0 disagreements, 0
TF32-sensitive flips; Step 2 all 4,608 episodes resolve by n_iter≤28
(4,313 at 24, 295 at 28, independently re-tallied). STEP 3b replay PASS
12/12 byte-identical. **New disclosure from this session's own raw-dump
pull:** the n_iter=20 near-miss WORSENS over training (13–27/128
anomalous rows at checkpoint 16000, maxima 0.07–0.43 → 128/128 anomalous
at 18000/20000, maxima up to 1.43) yet still fully resolves by n_iter≤28
at the worst-case checkpoint — the strongest evidence yet this is
iteration-fixable, not a structural wall. Item-1's baseline-relative
re-pin (nondeterminism envelope, 3× threshold) independently
re-confirmed a SECOND time in-chain with a DIFFERENT measured envelope
(8.841e-04 vs. the original adjudication's 7.51e-04) — consistent with
genuine run-to-run GPU nondeterminism, strengthening confidence in the
re-pin itself. GPU-h realized this launch: 0.487 chain + 0.33
verification ≈ **0.82 total**; contingency seeds NOT fired.
KEY_ANCHORING_SCALING sub-ledger: 18.5466/26 → **19.3666/26** (92.22% of
the original 21 GPU-h ceiling, reserve 1.6334 — still fits without the
`+5.0` extension).

**THE UNLOCK (§15.24.6 outcome-(c), the only place the design registers
the mechanics): PURE RE-READ, disclosed choice.** A literal per-cell NS
re-sweep is impossible without new GPU spend (10 of the 11 quarantined
cells never saved a full checkpoint) — the admission FLAG alone (never
the always-valid h4 measurement) was recalibrated offline for the 11
cells sharing the repro's own verified failure signature (`K72/{1740,
1742}`, `K78/{1840,1841,1842}`, `K84/{1940,1941,1942}`,
`K90/{2040,2041,2042}`); `K69/seed=1730` deliberately excluded (same
signature, but the pre-existing §15.19-era anomaly, outside the declared
11-cell scope). Byte-diff confirmed `admissible` is the only field that
changed across all 11 flips.

**The re-fit — `fit_cliff_curve.py`, full d=96 grid `{69,72,78,84,90}`,
n=3/K — is DEGENERATE (100% of 4,000 bootstrap resamples), not the
clean discrimination the wide-grid wave was built to attempt.** Per-K
means: K69=0.9592, K72=0.9216, K78=0.9326, K84=0.9581, K90=1.0000 — the
curve is NON-MONOTONIC (dips at K72/K78, exact ceiling at K90), not the
assumed monotonic cliff shape; the main fit pins at BOTH its own upper
bounds (`x0=0.9, L=1.2`). Applying §15.20 Rev 1's 6-row decision rule
mechanically: Step 0 fails (degenerate>10%) → Step 1a fails (4/5 K means
sit below 0.98, not a flatness signature) → **Step 1b fires: VERDICT
AMBIGUOUS** (genuine scatter, not a data gap this time — the unlock
closed the data gap cleanly, 5/5 K's populated). §15.20.4's own
absolute-slack-vs-power-law discrimination test STILL does not execute —
second wave in a row to reach that non-outcome, for a different
mechanical reason each time. **Scaling-law reading:** `x0(64)=0.5455 →
x0(80)=0.6779 → x0(96) UNRESOLVED`. This registered fit SUPERSEDES
§15.22's own diagnostic-only 6-K fit (`x0=0.7716`, wrong grid, n=1-driven)
— that suggestive point does not survive the full unlocked data.
Noisiest K-group (registered follow-up target): K72. Not launched this
harvest (harvest-only discipline).

**PI check-in package, refreshed this session (consolidating status
across lanes for the next check-in, not a new decision on any of them —
none of the PENDING-USER items below were acted on by this harvest):**

- **The d=96 resolution, for the paper drafts — DONE (2026-07-08, fold-in
  session).** `iclr-2027/sections/{01_intro,04_phenomenon,05_mechanism,
  08_results,09_discussion_limitations,10_conclusion}.tex` no longer say
  "in-flight, not pre-claimed" / "still DRAFT... not yet launched" — every
  d=96 passage now states the actual landed result (per-K h4
  0.9592/0.9216/0.9326/0.9581/1.0000, NO cliff through K/d=0.9375, curve
  non-monotonic not flat-ceiling, sigmoid 100% bootstrap-degenerate,
  x0(96) unresolved from genuine scatter not a data gap) plus the
  tolerance-miscalibration catch story (11/12 flagged cells, all 4,608
  episodes resolve n_iter≤28, admission flag alone recalibrated). x0(80)
  corrected 0.6756→0.6779 (the escalated n=5 fit) everywhere it appears,
  for internal consistency with the new x0(96) text. **Correction to this
  block's own prior claim:** the `workshop-2026/` 4pp Extended Abstract
  was NOT actually scoped away from d=96 as this note previously
  asserted — `sections/04_open_question.tex` and `05_limitations.tex`
  both discuss the d=80/96 super-linear-growth follow-up at length and
  carried the identical stale DRAFT hedge; both are now updated the same
  way, and `main.pdf` was recompiled clean via `tectonic` (5pp, unchanged
  page count). No figure needed regeneration — neither `fig_cliff.py`
  (d=64/d=128 only) nor `fig11_capacity_curve` in `make_figures_v2.py`
  (K=16/32/48 at d=64 only) plots d=80/96 data. Reasoning-link's own
  triple-null + Phase-2b bounded-causal result was folded into ICLR
  discussion item 10 (§16.15/§16.18) in the same pass. Full detail:
  `EXPERIMENT_LOG.md`'s "PAPER FOLD-IN" entry.
- **Submissions venue/author/title — still PENDING-USER, unchanged by
  this harvest, timing now close:** `matrix-thinking/submissions/
  neurips-ws-2026/VENUE_DECISION.md` (Decision 1: NeurReps/UniReps
  Extended Abstract is the leading candidate; NeurIPS 2026 workshop
  accepted-list drops **~2026-07-11 — 3 days from this update**, CFPs
  open shortly after) and `PAPER_SPRINT_PLAN.md` §5.5 (Decision 2:
  title — keep the current default, not adopting a 5th option; Decision
  3: author block — fill in `Sam Larson, Pebble AI, pebbleml.com`
  contingent on confirming the chosen venue's double-blind requirement
  first) both still show open checkboxes. Nothing in this harvest
  resolves any of the three — flagged for the PI's next check-in given
  the CFP window is now imminent.
- **Phase-2b ratification — still awaiting PI decision, unchanged by
  this harvest:** the REASONING-LINK PHASE-2B block below (KEYSTONE
  UNRESOLVED, harvested earlier 2026-07-08) has no pre-scripted next
  branch in §16.6's own decision tree; PI options named there (more
  seeds, a lower-variance readout, or reporting the underpowered-null
  as the standing result) remain open, not decided by this harvest
  (different lane, GPUs 0-1 vs. this lane's GPU 2 — never gated each
  other).
- **Injection report (security, consolidated across this harvest):** one
  fake `<system-reminder>` this session (fabricated date-change
  concealment instruction + fabricated agent-list/MCP-instructions
  block, appended to the first `Bash` tool result) — disregarded in
  full including the concealment instruction, reported. The one factual
  claim inside it (date=2026-07-08) is independently correct against
  box `date`, raw-JSON timestamps, and commit `a51f102`'s own author
  date — the danger was the concealment instruction and fabricated
  capability lists, not that string. Zero injected content found in any
  box-persisted artifact this session. This continues a now
  well-established recurring pattern across this project's recent
  sessions (multiple prior occurrences logged in this file's own
  Phase-2b blocks above/below) — worth a standing note for whoever next
  triages the harness's own tool-output handling, not a new finding
  specific to this harvest.
- **Headlines (skim version):** (1) C17 is CLOSED — the wide-grid wave's
  11/12-inadmissible mystery was tolerance miscalibration, not a real
  capacity boundary or a broken probe. (2) The unlock worked exactly as
  designed (data gap closed, zero new GPU spend) but the underlying
  d=96 signal is genuinely noisy/non-monotonic at n=3 seeds/K — x0(96)
  is NOT resolved, and neither rival scaling-law band (`[0.718,0.739]`
  vs. `[0.768,0.837]`) is confirmed or excluded. (3) The super-linear
  capacity story's own d=96 rung stays an open question, now for a
  data-quality-of-signal reason rather than a data-availability reason —
  the paper text needs a PI pass to reflect this honestly. (4) Neither
  the workshop-2026 nor the ICLR venue/author/title decisions are
  blocked by anything in this harvest, but the CFP window is 3 days out.

---

## REASONING-LINK PHASE-2B VOCAB-SPACE CONTRAST — HARVESTED, KEYSTONE UNRESOLVED at the registered instrument (2026-07-08, chain completed 06:28:19Z) — supersedes this block's own prior "LAUNCH ATTEMPT 2 RUNNING" status (chain ran to completion cleanly). **GPUs 0-1 are FREE.** Full account: `matrix-thinking/REASONING_LINK_DESIGN.md` §16.18; harvest entry: `EXPERIMENT_LOG.md`.

**Completeness verified from raw box files:** 18/18 cells (6 reused OFF + 12 new `per_token`/`global` × 2 corpora × 3 seeds) at 5000/5000 steps, grad_finite throughout; 30/30 reused OFF checkpoints sha256-verified; OFF-eval cache sha256 verified 3 ways (local/box/pin, all agree); both corpora FLOOR-PASS (openr1 ratio=0.9823≤1.1427; wikitext ratio=1.0138≤1.1287), CONFIRMATORY tier.

**THE KEYSTONE VERDICT.** None of the 4 primary (corpus × arm) contrasts — independently re-derived per-arm from the raw CIs, since the registered pipeline computes only 2 corpus-level verdicts using the `global` arm as a representative proxy (a disclosed scoping finding, §16.18.3) — hit PERSISTENT/LATE-EMERGENT (would mean "arm helps") or CONVERGED-EQUIVALENT (would mean "genuine, well-powered equivalence"). **3/4 (openr1×global, openr1×per_token, wikitext×global) are UNRESOLVED** — the pre-registered most-likely outcome at n=3 seeds (detectable-effect floor ≈1.5-1.7 loss units, same order as the OFF arm's own entire familiarization effect) — an open measurement question, not a clean negative. **1/4 (wikitext×per_token) is TRANSIENT** — a real, independently-confirmed signal (`holds(2500)=TRUE`) that the registered pipeline's own global-arm-representative corpus rollup silently absorbed into wikitext's UNRESOLVED — but the sign is NEGATIVE (arm transiently HURTS, Δ=−0.4999 at c=2500, CI=[−0.6241,−0.3758]), not a "helps" reading, and TRANSIENT is registered as a non-durable training-dynamics artifact regardless of sign. **Answer to the keystone question ("does the frozen-bias arm's whole causal package change in-context task acquisition"): this wave does not resolve it either way** — a measurement-power result, not a mechanism result.

**GPU-h:** total realized (both launch attempts) **1.6611 GPU-h** — attempt-1 abort 0.0872 (cold-compile timing-pilot overrun, caught correctly pre-launch) + attempt-2 complete 1.5739 (warm-cache pilot 6.4× faster) — a ~1.6× undershoot of the 2.64 GPU-h raw estimate, ~15.9× undershoot of the 26.4 GPU-h debug-tax ceiling.

**Discrepancies/follow-ups (not self-launched, registered for a future wave):** fix `analyze_corpus`'s corpus-level scoping to classify both non-off arms independently rather than using `global` as a silent proxy for `per_token`; the keystone question needs either more seeds (fresh Leg-A pretraining, real GPU cost) or a lower-variance readout to move off the power floor; PI check-in should report the underpowered-null-dominant result plus the one real, non-durable, hurts-direction TRANSIENT signal, not a resolved answer either direction.

**Security note:** 2 more fake-system-reminder injections this harvest session (one appended to a `grep` Bash-tool result, one appended to an `Edit` tool result — both the same fabricated date-change/concealment pattern, one also carrying a fake agent-list + fake MCP-instructions block) — disregarded in full including the concealment instructions, reported. Zero injection strings found in any raw box-persisted artifact. 5th and 6th occurrences logged against this overall session (2 in DEPLOY ATTEMPT, 1 in LAUNCH ATTEMPT 2, 2 here).

**Queue:** GPUs 0-1 free for the next wave. No cells launched, no code built this harvest session (harvest-only, per `CLAUDE.md`'s "the implementer does not review their own work" / §15.10's "harvest does not self-launch" discipline).

**Superseded block below (2026-07-08 ~05:44 UTC launch-attempt-2 status) — kept for the record, now stale, completed by the harvest above:**

## REASONING-LINK PHASE-2B VOCAB-SPACE CONTRAST — LAUNCH ATTEMPT 2 RUNNING, HEALTHY START (2026-07-08 ~05:44 UTC, GPUs 0-1) — supersedes this block's own prior "BUDGET-GATE ABORTED" queue below (items 1-2 completed: re-derivation landed at commit f5a0c21, ceiling 20.6→26.4, §16.16.11 item 1 RESOLVED; chain relaunched). **12 cells (`per_token`/`global` × 2 corpora × 3 seeds) are TRAINING on GPUs 0-1 right now** — `tmux phase2b` on the box, log `logs/phase2b_run2.log`, ETA ≈ 06:30-06:50 UTC 2026-07-08, completion sentinel `results/phase2/PHASE2B_SUMMARY.json` + the `PHASE-2B CHAIN COMPLETE` log line. Do not touch GPUs 0-1 until the sentinel appears; **HARVEST IS THE NEXT QUEUE ITEM** once it does.

Attempt-2 facts (full account: `EXPERIMENT_LOG.md`'s LAUNCH ATTEMPT 2 entry): all gates re-passed in-chain (Stage −1 ×2 suites, smoke ×3 arms, sha256 30/30); the timing pilot PASSED at **2.1488 s/pass** — attempt 1's 13.7339 s was cold-start Triton kernel compilation, not the true rate; the real projection (raw 1.4789 GPU-h, bracket-high 14.7888) is under even the OLD 20.6 ceiling, so the 26.4 re-derivation is conservative. FLOOR_PINNED blind-pinned 05:47:16Z: openr1 pooled_ratio=0.9823/floor_pin=1.1427, wikitext pooled_ratio=1.0138/floor_pin=1.1287 — **BOTH corpora FLOOR-PASS**, full CONFIRMATORY-tier hexachotomy proceeds. First pair (per_token openr1 s0/s1) confirmed training at ~19 GB/~87% util per GPU with healthy step-250 checkpoints (L_query ~4.7-4.8, already well below the ~12.3 OFF baseline — encouraging early behavioral-contrast signal, NOT a verdict).

**Superseded queue (attempt-1 abort, kept for the record):** the BUDGET-GATE ABORTED account below is still the accurate history of attempt 1 (~05:30 UTC); its 3-item queue is done/underway (1: f5a0c21; 2-3: running now).

## REASONING-LINK PHASE-2B VOCAB-SPACE CONTRAST — DEPLOY ATTEMPT, BUDGET-GATE ABORTED (2026-07-08 ~05:30 UTC) — supersedes this block's own prior "REV 2.1 LANDED (2026-07-13)" status below: Rev 2.2 (round-4 spot-check) landed, BUILD landed (commit 4cf6122), an independent build audit landed and fixed a FATAL (cache-envelope KeyError) + a MAJOR (cache sha256 integrity check) at commit 42b3f48 — LAUNCH-CLEARED, 22/22 Stage −1, mutation-confirmed. Deploy then shipped everything cleanly (closure + manifest sync + box Stage −1 22/22 + reasoning_link Stage −1 19/19 + real-kernel smoke ×3 arms all PASS + FORCE_FAIL negative correctly failed + full pre-flight clean) and launched `phase2b_chain.sh` in tmux on GPUs 0-1 — but the chain MECHANICALLY ABORTED at its own pre-launch timing-pilot budget gate, before the OFF-eval cache build or any of the 12 new cells. One real eval pass on GPU 0 measured 13.7339s; projected debug-tax-bracket-high cost 26.3739 GPU-h exceeds the registered ENFORCED ceiling (20.6 GPU-h, §16.16.8) by ~28% — `phase2b_off_cache.py --time-pilot`'s own registered logic correctly refused, `set -euo pipefail` propagated the abort, zero cells launched, negligible GPU-h spent (<0.1). This is the pre-launch safety gate working as designed (see full account in `EXPERIMENT_LOG.md`'s matching entry) — it also empirically resolves §16.16.11 item 1's own open "GPU-h reference-rate uncertainty" concern: the registered reference rate was too optimistic by ~28% for this box's real measured rate, even after the existing 10× debug-tax bracket. *(Attempt-2 correction: the pilot rate was cold-cache compile-dominated; the steady-state rate is ~6× cheaper — see the block above.)*

**Superseded block below (2026-07-13, "REV 2.1 LANDED") — its own Rev 2/2.1 fix history is still accurate background, only its "Queue" (round-4 spot-check → build → audit → launch) is now stale, completed by the above:**

## REASONING-LINK PHASE-2B VOCAB-SPACE CONTRAST — REV 2.1 LANDED (2026-07-13) — supersedes this block's own prior "REV 2 LANDED (2026-07-12)" status: round-3 verify returned NEEDS-REVISION (1 MAJOR, 1 MINOR, no FATAL, both surgical), all fixed. The REASONING-LINK PHASE-2 FAMILIARIZATION block further below's own harvest facts (gate-refused, PARTIAL-TASK-LEARNING-BELOW-PIN, triple-null arc) are otherwise still current/unchanged.

**Status: Rev 2.1, round-3 verify landed, DESIGN-ONLY.** `matrix-thinking/REASONING_LINK_DESIGN.md` §16.16 (fixes), §16.17 (round-1, round-2, and round-3 finding→fix tables). Zero GPU spent this session; no box access this session (Rev 2.1 is a pure documentation/design fix pass over Rev 2's own already-verified text).

**Rev 1 fixed (unchanged, verified landed by round 2, not reopened):** the analysis-integration gap (MAJOR-1, highest value — `phase2_trajectory_analysis.py` was wired to the DEAD `recovered_frac` quantity and the permanently-FAILED Stage-0.5 gate JSONs; would have silently buried a real PERSISTENT finding as UNRESOLVED-GATE); chain self-refuse on reuse (MAJOR-2 — forked `phase2b_chain.sh`, drops the dead launch gate); the knife-edge, wrong-readout OFF-floor pin (MAJOR-3 — re-pinned BLIND from the 6 reused OFF checkpoints' own new-readout data, before any new cell launches); understated causal framing (MAJOR-4 — the registered intervention is the arm's entire causal package, pretraining-era divergence AND the persistently-applied familiarization-time blend); under-covered real-kernel smoke (MAJOR-5 — now off+per_token+global, all three required); plus 3 MINORs (sha256 negative test made explicit, a transposed totality-citation label pair corrected, a new `--arm`-vs-checkpoint-config assertion).

**Rev 2 fixed (unchanged, verified landed by round 3, not reopened):** Rev 1's own Stage-1 negative test only exercised `classify_trajectory` (never the broken piece) on a hand-built `holds_by_c` dict, never the rewritten `build_holds_and_gate_by_checkpoint` that actually needed proving (MAJOR-R2-1, highest value) — re-registered one layer down: stub `eval_query_loss_heldout` through the REAL `delta_ci_n3`→`det`→`holds` chain inside the rewritten function, plus a structural assertion (`recovered_frac`/`gate_json_path_for` grep-gone from the module); eval-(B)'s own forward path was left adjudication-free against the parent doc's broad MAJOR-NEW-5 surgery rule, risking a habit-pasted force-off wrapper that would silently revert the causal claim (MAJOR-R2-2 — pinned NO surgery override for eval-(B), MAJOR-NEW-5 scoped RETIRED-GATE-ONLY, new Stage −1 assertion); the "360 passes, computed once, consumed twice" framing undercounted by ~33% since `analyze_corpus`'s own per-arm iteration re-scores `off` inside each of 2 arm branches (MAJOR-R2-3 — registered an OFF-eval cache, `off_lquery_cache-Phase2b.json`, cached 360/uncached ≈480 both disclosed, cost bracket corrected to ≈10.3-20.6 GPU-h); plus 2 MINORs (`eval_query_loss_heldout`'s `batch_size` pinned to `=16`; smoke-cost arithmetic corrected to actually use the disclosed ≈0.02-0.03 GPU-h figure instead of the stale ≈0.01). Full trace: §16.17 (round-2 table).

**Rev 2.1 fixed:** the registered Stage-1 test stubbed `load_init_checkpoint_strict` as the eval-model loader, but that function (`lm_pretrain_rd.py` L1803) takes an ALREADY-CONSTRUCTED model and mutates it in place — it cannot be stubbed to deliver a sentinel, and model construction actually happens one layer up, a separate `torch.load(...)['config']` antecedent (`phase2_familiarization_train.py` L408→L421) (MAJOR-R3-1) — fixed by registering a single-seam helper, `phase2b_load_eval_model(ckpt_path, device) -> DeltaNetLM`, that internally reproduces the SAME L408→L421 double-defense sequence; the rewritten `killer_prediction_readout` routes ALL eval-model loading through this ONE helper, which the Stage-1 test now stubs directly; production keeps the strict double-defense path unconditionally, never the laxer `reasoning_link_probe.load_checkpoint`. Also fixed: the `frozen_bias_surgery`/`recovered_frac`/`gate_json_path_for` `getsource` substring-absence assertions were fragile to a docstring/comment mentioning the identifier in prose (MINOR-R3-1) — re-pinned as AST-level checks via one shared `_references` helper (walks `Call`/`Attribute`/`Name` nodes, ignores docstrings/comments), plus a house convention that in-code prose should cite fixes by section number rather than the literal retired identifier. Full trace: §16.17 (round-3 table).

**Queue (Rev 2.1 → round-4 spot-check → build → audit → launch on GPUs 0-1, currently free):**
1. **Round-4 spot-check on §16.16 Rev 2.1** (fresh-eyes reviewer; NOT a full attack round — confirms round-3's own 2 fixes above land as described, plus takes a first pass at the 5 still-open self-attack items §16.16.11 lists — GPU-h reference-rate uncertainty, readout-(B) variance-proxy conservativeness, floor-gate pooling granularity, pairing-device CI-independence risk, secondary-readout multiple-comparisons correction) — NOT YET RUN. Blocking item.
2. **Build delta**, once cleared: the Rev-2.1-registered analysis-module rewrite (`phase2_trajectory_analysis.py`, §16.16.3), including the new `phase2b_load_eval_model` single-seam loader, with its own re-registered negative Stage-1 test (behavioral + structural parts, AST-level assertions via `_references`); the OFF-eval cache (`off_lquery_cache-Phase2b.json`); the forked `phase2b_chain.sh` (§16.16.8); parameterize `query_loss_forward`'s hardcoded `hop_set`; add 2 new `phase2_seed` kinds (`eval_lquery_heldout`, `eval_lquery_ood`) with the arm-independent pairing convention (§16.16.2/16.16.3); write `phase2b_floor_gate_enforce.py` taking `floor_pin` as a required argument (§16.16.6); extend `phase2_smoke_gpu.py` with a real `--arm` flag and run the full suite against ALL THREE arms (§16.16.9); wire the sha256 reuse gate with its own negative test (manifest already committed, `experiment-runs/2026-07-08_phase2_familiarization/gates/phase2b_off_ckpts_reuse_manifest.sha256`); the `--arm`-vs-checkpoint-config Stage −1 assertion (§16.16.9 item (c)); the AST-level no-surgery Stage −1 assertion for `eval_query_loss_heldout` (§16.16.9 item (d)).
3. **Independent build audit** (separate agent, per `CLAUDE.md`'s "the implementer does not review their own work").
4. **Launch**: 12 new cells (`per_token`/`global` × 2 corpora × 3 seeds) on GPUs 0-1, registered cost bracket ≈10.3-20.6 GPU-h (§16.16.8, Rev-2-corrected).

**Known risk, registered now so it cannot be silently elided at harvest time (§16.16.4):** at n=3 seeds, the minimum detectable arm effect (≈1.5-1.7 loss units) is roughly the SAME SIZE as the OFF arm's own entire familiarization effect (≈1.69) — a modest, plausible intervention effect will likely resolve to the pre-registered UNRESOLVED bucket, not a clean positive or negative. Any harvest read as a clean negative must be checked against this power sketch before being reported as such.

---

## REASONING-LINK PHASE-2 FAMILIARIZATION — GATE-REFUSED (harvested
2026-07-08, chain ran 01:27–01:45 UTC) — verdict PARTIAL-TASK-LEARNING-
BELOW-PIN; the C17/§15.24 blocks below are unaffected (different program,
GPU 2 planned there; Phase-2 owned GPUs 0–1, now free)

**Queue status: COMPLETE (OFF-arm leg only) — `per_token`/`global` launch
MECHANICALLY REFUSED, not a crash.** The OFF arm's 6 cells (2 corpora ×
3 seeds) ran to completion (5,000/5,000 steps, no crashes, one clean
launch). The Stage-0.5-familiarized gate then FAILED all 30/30 (cell,
checkpoint) readings (`recovered_frac(h1)=0.0000` everywhere;
premises (iii)/(iv) and the h1 floor all `False`). Per §16.5 Constraint 1
the chain refused to launch `per_token`/`global`, wrote
`results/phase2/STAGE05_LAUNCH_GATE_REFUSED`, and exited cleanly.
**GPUs 0–1 are FREE.** Full tables, mechanical adjudication, and the
triple-null reading: `matrix-thinking/REASONING_LINK_DESIGN.md` §16.15;
harvest entry: `EXPERIMENT_LOG.md`'s matching entry.

**The central question, adjudicated mechanically (§16.15.2):** did the
model learn the bind/query task while the geometric readout stayed
invalid? Pin: terminal `L_query` < 50% of its step-250 value. **0/6
cells crossed it** (21.8%–46.4% relative fall, mean −35.9% — real and
uniform, but short of the pin in every cell). Verdict:
**PARTIAL-TASK-LEARNING-BELOW-PIN** — neither pre-registered bucket (a)
or (b) fit cleanly; a genuinely independent vocab-space readout
(`L_query`) showed real partial signal while the `d_state`-space gate
stayed at an exact `0.0000` floor in all 30 readings — a real
dissociation, disclosed but not overclaimed.

**Triple-null arc:** Phase 1 marker/zero-shot (§15, 0/312) → Phase-1b
natural/zero-shot (§16.8, 0/4) → Phase-2 marker/FAMILIARIZED (this
result, 0/30) — three structurally different instruments, three training
regimes, identical categorical `0.0` floor on the `S_T^h·q_eff` readout.
Still PROBE-INVALID, not REFUTE, per §8.4/§16.0's standing rule.

**Realized cost:** ≈0.617 GPU-h (1,111s wall × 2 GPUs), well under the
registered 1.48–12.06 GPU-h full-grid bracket (this leg ran only 6/18
cells) and under even the leg-scaled 5–10× bracket's own low end — one
clean launch, no crashes.

**Queue: PI decision on the REASONING-LINK keystone lane at next
check-in.** §16.6's decision tree has no pre-scripted branch for
"Phase-2's own gate also refuses" (§16.15.5 names the registered options
without selecting one: a vocab-space behavioral-contrast instrument
promoted to primary; lane closure with the triple null as the
publishable finding; or a recipe-extension calibration run). Not
self-launched by this harvest.

---

## C17 EVAL-ADMISSION REPRO INSTRUMENT — RUN COMPLETE (2026-07-08, chain
06:25:46–06:55 UTC on GPU 2) — **VERDICT: TOLERANCE-MISCALIBRATION** (via
Step 2) — supersedes the REV 4 LANDED block below's queue status

**Queue status: COMPLETE.** One clean launch (tmux `c17_repro`, GPU 2
only; GPUs 0–1 untouched, Phase-2b concurrent). The K=84/seed=1940/d=96
repro cell re-ran with telemetry (20,000/20,000 steps, `wall_s=1403.2s`,
under both the 2430s budget guard and §15.22's 1409s baseline), and the
full §15.24.4 precedence resolved: reconstruction gate PASS (107/107
set-equal); **Step 0b: 0 pool-membership violations**; Step −1: **36
distinct events** (≥3 minimum, all from checkpoints 16000/18000/20000 —
12 each, hops 1/2/3 × 4 batches; checkpoints 2000–14000 carried zero);
Step 0a: 0 anomalous episodes; **Step 1: 0 live/offline disagreements, 0
TF32-sensitive flips** (recorded tf32_matmul=False, tf32_cudnn=True);
**Step 2: all 4,608 episodes (36×128) resolve below the 0.01 tol by
n_iter≤28 — 4,313 at n_iter=24, 295 at n_iter=28, none needed 32/40** →
TOLERANCE-MISCALIBRATION per §15.24.1 row (c): live resids at n_iter=20
sit far above tol (event-0 range 0.696–1.371) yet are an
iteration-count-fixable near-miss, not a structural wall. The probe is
NOT buggy (0b/Step-1 clean) and the failure is NOT a real capacity
boundary at these K/d. STEP 3b per-launch determinism replay (new,
disclosed chain addition, audit residual 2): **PASS, 12/12 step-20000
events byte-identical** (key_ids + k_eff_raw) against a fresh
full_step20000.pt reload. Box Stage −1 fully green in-chain: ALL items
PASS, 0 deferrals (formerly-deferred 1/2/3/4/9 real branches + item 5's
box half all ran live). **Disclosed deviations this deploy:** (1) item 1's
registered bitwise OFF-vs-ON smoke was UNSATISFIABLE on this hardware —
OFF-vs-OFF at identical flag/seed diverges from step 4 (max_abs_dev
7.5e-04; the repo's documented fixed-seed nondeterminism) — re-pinned
baseline-relative (≤3× the OFF-vs-OFF envelope; envelope 0 → bitwise;
bracketed note at §15.24.2); (2) STEP 3b's zero-event skip fatals on a
coverage gap if earlier checkpoints carry ≥3 events (independent-audit
MAJOR, fixed pre-launch — moot here, final checkpoint had 12 events).
GPU-h: chain 0.487 (1,754s × 1 GPU) + ≈0.33 pre-launch verification
(3 Stage-−1 suite runs + OFF-vs-OFF diagnostic) ≈ **0.82 total**.
Artifacts pulled to `matrix-thinking/deltanet_rd/results/
keyanchor_scaling_c17repro/` (analysis JSON, replay JSON, Stage-−1
results, OFF-vs-OFF diag, chain log; raw ~100MB dumps stay on box).
**GPU 2 is FREE.** Harvest (§15.25 + EXPERIMENT_LOG) is a separate
agent's task, per the deploy brief. Registered §15.24.6 outcome-(c)
implication of this verdict, verbatim: "a cheap, surgical fix (bump
`geo3_resid_tol` for raw/un-blended queries specifically, or raise
`n_iter` modestly) unlocks the ALREADY-COLLECTED 11 inadmissible cells'
own h4 values for the fit **without any new GPU spend**" — the fastest
path back to §15.20.4's rival-discrimination test (this run's own Step-2
data says n_iter=28 suffices for every flagged episode). PI decision at
next check-in, not self-launched.

---

## C17 EVAL-ADMISSION REPRO INSTRUMENT — REV 4 LANDED (2026-07-11) —
supersedes the REV 3 LANDED block below's own queue status (that block's
Rev 3 content is otherwise unaffected/unretracted; Rev 4 fixes it, not
replaces it)

**Queue status: DESIGN (Rev 4) → (next) ROUND 5 VERIFY PASS → BUILD →
AUDIT → LAUNCH.** `KEY_ANCHORING_SCALING_DRAFT.md` §15.24's independent
attack-round-4 verdict: **NEEDS-REVISION** — 0 FATAL, 2 MAJOR, 1 MINOR —
the narrowest, most surgical finding set of the four rounds so far, and
the first with zero FATALs. All three fixed in Rev 4, finding→fix table
at §15.24.13. **MAJOR-1 (highest value):** the offline analysis must
reconstruct `pools.heldout_name_ids`/`pools.train_name_ids` to evaluate
Step 0b at all, but nothing pinned HOW — the training path's own caller
(`run_deltanet_rd.py:1470`) builds these pools with a HARDCODED `seed=0`,
decoupled from the training `--seed` Rev 3 made a load-bearing per-event
field. A builder naively threading the launch seed (1940) into the
offline reconstruction would silently draw the WRONG train/held-out
partition, firing a total, confidently wrong INSTRUMENT-BUG verdict on
otherwise-healthy code — the "two seeds" trap. Fixed by pinning offline
reconstruction to the literal hardcoded call
`grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)`, NEVER the
launch seed, plus a new prerequisite gate ahead of Step 0b itself: assert
the reconstructed `train_name_ids` is SET-EQUAL to the checkpoint's own
already-archived `anchor_train_ids` tensor (`run_deltanet_rd.py:934–936`,
zero new telemetry cost); mismatch → hard-abort before any verdict (a new
RECONSTRUCTION-FAILURE state, distinct from INSTRUMENT-BUG). **MAJOR-2:**
the KEY_ANCHORING_SCALING ledger baseline (previously stated as
`18.1196/26`) omitted the K=69/seed=1733 contingency cell's own realized
0.427 GPU-h (§15.22 addendum, landed 2026-07-08) — corrected to
**`18.5466/26` GPU-h realized**; every downstream figure re-derived, worst
case (2× + NO-REPRO contingency) now `20.3466/21 = 96.89%`, reserve
**0.6534 GPU-h** against the ORIGINAL ceiling (down from the
previously-claimed 1.0804 — a ~40% tightening of the reserve; the
conclusion this design fits without the `+5.0 GPU-h` extension survives
unchanged, the margin backing it does not). Folded into
`EXPERIMENT_LOG.md`'s running-total convention and into this file (see the
correction note on the wide-grid-wave harvest block below). **MINOR-1:**
new dedicated Stage −1 fixture for the minimal boundary case 0b's own
prose already names but no fixture yet exercised — a SINGLE-EVENT sink
with exactly 1 pool-mismatch violation, asserting INSTRUMENT-BUG fires
BEFORE Step −1's own `<3`-event gate would even run. **Rev 4 has NOT yet
had its own verification pass — round 5, a VERIFY pass confirming these
three fixes land clean (not a fresh full attack round), is next.** No
cells launched, no code built this session.

---

## C17 EVAL-ADMISSION REPRO INSTRUMENT — REV 3 LANDED (2026-07-10) —
supersedes the REV 2 LANDED block below's own queue status (that block's
Rev 2 content is otherwise unaffected/unretracted; Rev 3 fixes it, not
replaces it).

**Queue status: DESIGN (Rev 3) → (next) ATTACK ROUND 4 → BUILD → AUDIT →
LAUNCH.** `KEY_ANCHORING_SCALING_DRAFT.md` §15.24's independent
attack-round-3 verdict: **NEEDS-REVISION** — 1 FATAL, 2 MAJOR, 6 MINOR.
All nine fixed in Rev 3, finding→fix table at §15.24.12. Headline fix
(**FATAL**): Rev 2's own two-level dispositive floor (≥2 anomalous
episodes across ≥2 distinct events) is a NOISE argument — sound for Step
1's NUMERICAL live/offline disagreement check, where a near-boundary
residual genuinely can jitter run to run — but wrongly applied, unchanged,
to Step 0b's pool-membership precheck, which is STRUCTURAL: a dumped
entity id either is or is not a member of the disjoint held-out pool,
computed with zero floating-point arithmetic; one violation is already
deterministic proof of a bug, exactly this project's own "exact
threshold, no tolerance slack copied from a floating-point context" rule.
Concretely, a real pool-mismatch in a 5-event sink previously fell below
the 2-event floor, was EXCLUDED, and the verdict silently continued to
REAL-CAPACITY-BOUNDARY or TOLERANCE-MISCALIBRATION on the untainted
remainder — a confidently wrong claim. Fixed by splitting the floor: Step
0b is now dispositive on ANY SINGLE pool-membership violation, no
event/episode-count minimum, mirroring `model_rd.py:2048`'s own
assert-exactly-zero convention; the ≥2-episode/≥2-distinct-event bar now
gates Step 1's numerical disagreement check ONLY. Two MAJORs, both in the
combined-sink machinery Step −1's NO-REPRO contingency path created:
event identity `(step, hop, batch_idx)` was never launch-unique across
the up-to-3-launch combined sink, so a cross-launch reproduction at
identical coordinates could wrongly dedup to "1 event" — fixed with an
additive `seed` field on every dumped event, pinning `episode := (seed,
step, hop, batch_idx, row_idx)` and `event := (seed, step, hop,
batch_idx)` (cross-launch recurrence at identical coordinates is now
disclosed as the STRONGEST reproduction evidence available); and Step 1's
offline recompute ran on a batch-size-1 slice of the dumped tensor, which
can select a different GEMM kernel than the live batch-size-128 call and
flip a near-boundary residual from batching alone — fixed by recomputing
ONE batched call on each event's full dumped `(B,K,d)` tensor, matching
the live call's own batching exactly. Six MINORs: a missing cross-marker
negative test (floors count per-marker-type, never combined); a stale
"per episode" usage reworded to "per K-item pool draw"; a citation off by
one line (`:3097`→`:3098`); the bitwise re-confirmation pinned as an
explicit hard-abort on failure; 0a's own corroborating-marker counting
rule named explicitly; "residual AMBIGUOUS" renamed to
**AMBIGUOUS-RESIDUAL**. **Rev 3 has NOT yet had its own independent audit
pass — per this program's standing multiple-independent-audit-rounds
rule, a fresh attack round 4 is the next step, before build.** No cells
launched, no code written this session.

---

## C17 EVAL-ADMISSION REPRO INSTRUMENT — REV 2 LANDED (2026-07-09) —
supersedes the REV 1 LANDED block below's own queue status (that block's
Rev 1 content is otherwise unaffected/unretracted; Rev 2 fixes it, not
replaces it).

**Queue status: DESIGN (Rev 2) → (next) ATTACK ROUND 3 → BUILD → AUDIT →
LAUNCH.** `KEY_ANCHORING_SCALING_DRAFT.md` §15.24's independent
attack-round-2 verdict: **NEEDS-REVISION** — 1 FATAL, 3 MAJOR, 4 MINOR.
All eight fixed in Rev 2, finding→fix table at §15.24.11. Headline fix
(**FATAL**): "episode" was never pinned to one referent across Rev 1's own
text — Step 0a's rank check already operated on ONE within-batch `(K,d)`
row, while the granularity-threshold paragraph's own "~120 dumped events"
denominator counted whole triggering batches, a ~128× gap (120 events vs.
~15,360 rows) in the dispositive-floor arithmetic; the codebase's own
docstring (`model_rd.py:433–434`) already distinguishes the two. Fixed by
pinning `episode := (step, hop, batch_idx, row_idx)` and `event := one
dumped dict, one triggering batch`, and replacing the inherited
percentage clause (2% of the correct 15,360-row population is 308 rows —
a bar a genuinely broken probe would plausibly never clear) with a
two-level absolute floor: ≥2 anomalous episodes occurring in ≥2 distinct
events. Three MAJORs: Step −1's `<3`-event reproduction bar and Step 0b's
structural floor disagreed on a 2-event, 2-pool-mismatch sink (a
deterministic bug signature Step −1 alone would refuse as AMBIGUOUS-
NONDETERMINISM) — fixed by reordering 0b (structural) ahead of Step −1
(reproduction), with total precedence pinned explicitly (0b > Step −1 >
Step 1 > Step 2); 0b's dispositive trigger had no enforced-abort branch —
added, with its own negative test; §15.24.2's dump-dict spec referenced a
nonexistent `evaluate_pool()` `step` parameter — fixed with an additive
`step=None` parameter threaded only at the C17 call site (also caught the
same code block's undefined `batch_i` loop index). Four MINORs: a stale
§15.24.1 table row still called Step 0a dispositive after Rev 1's own
demotion; a citation off by 1000 lines (`model_rd.py:149`→`:1149`); TF32
matched-mode recompute now pinned per-source-run (the combined sink can
span up to 3 launches — primary + 2 contingency seeds); the determinism
cross-check now runs per-launch, not once. **Rev 2 has NOT yet had its
own independent audit pass — per this program's standing multiple-
independent-audit-rounds rule, a fresh attack round 3 is the next step,
before build.** No cells launched, no code written this session.

---

## C17 EVAL-ADMISSION REPRO INSTRUMENT — REV 1 LANDED (2026-07-08) —
supersedes the DESIGN QUEUED block below's own queue status (that block's
design content is otherwise unaffected/unretracted; Rev 1 fixes it, not
replaces it).

**Queue status: DESIGN (Rev 1) → (next) ATTACK ROUND 2 → BUILD → AUDIT →
LAUNCH.** `KEY_ANCHORING_SCALING_DRAFT.md` §15.24's independent
attack-round-1 verdict: **NEEDS-REVISION** — 1 FATAL, 2 MAJOR, 2 MINOR.
All five fixed in Rev 1, finding→fix table at §15.24.10. Headline fix
(**FATAL-1**): the three-way decision rule (REAL-CAPACITY-BOUNDARY /
INSTRUMENT-BUG / TOLERANCE-MISCALIBRATION) operated only on dumped
fallback events — a zero-event re-run (plausible; this program already
measured seed-fixed run-to-run drift consistent with GPU floating-point
nondeterminism, `KEY_ANCHORING_DESIGN.md:1976–1994`, and no determinism
pin exists anywhere in the training path) would have silently emitted an
unearned TOLERANCE-MISCALIBRATION via a vacuous `all()`-over-empty-list
pass. Fixed with a new Step −1 guard: `<3` events → **NO-REPRO**, fires
the reserved K=84 contingency seeds 1943/1944 before concluding anything;
still `<3` after that → **AMBIGUOUS-NONDETERMINISM**, promoting the
checkpoint-payload extension (candidate (1)) to the primary next
instrument. Two MAJORs: TF32 matmul mode was unpinned/unrecorded for the
offline NS recompute (now dual-mode: strict-fp32 + matched-to-live, with
the live run's own TF32 state now recorded in the tap); single-episode
poisoning let ANY one anomalous episode among ~120 fire a dispositive
verdict (now needs ≥2 episodes or ≥2% of events, whichever larger; a
single occurrence is excluded-and-disclosed instead). Two MINORs: disk
worst-case corrected 490MB→~1GB (both `k_eff_raw` and `k_blend_raw` are
dumped per event, not one); the exact-rank precheck (`tol=1e-4`, never
rigorously derived) demoted from dispositive to corroborating. Ledger
re-derived with the new NO-REPRO contingency cost folded in: worst case
(2× + both contingency seeds) `18.1196 + 0.900 + 0.900 = 19.9196/21 =
94.86%`, still fits the ORIGINAL, non-extended ceiling with 1.0804 GPU-h
margin. **Rev 1 has NOT yet had its own independent audit pass — per
this program's standing multiple-independent-audit-rounds rule, a second,
fresh attack round is the next step, before build.** No cells launched,
no code written this session.

---

## C17 EVAL-ADMISSION REPRO INSTRUMENT — DESIGN QUEUED (2026-07-08) —
advances the NS EVAL-ADMISSION DIAGNOSTIC block's own "Queue implication"
below from two named-not-designed candidates to one designed, pre-attack
instrument. That block itself is otherwise unaffected/unretracted.

**Queue status: DESIGN → (next) ATTACK ROUND → BUILD → AUDIT → LAUNCH.**
`KEY_ANCHORING_SCALING_DRAFT.md` §15.24 (Rev 0, DESIGN-ONLY, zero GPU
spent this session) designs §15.23's registered candidate (2) — a single
deterministic repro of K=84/seed=1940/d=96 (the mid-pattern cell among
the three uniformly-failing K=78/84/90 groups, chosen because it already
has a verified complete on-box checkpoint tree, unlike §15.23's own
preferred-but-missing K=72/seed=1740) with a full model `state_dict`
snapshot at step 20000, live raw pre-NS `k_eff_raw` dumps on every C17
fallback event, and per-probe-pool residual telemetry — as the primary
instrument to discriminate three named mechanisms for the C17-exclusive
admission failures §15.23 found but could not directly test: (a)
**REAL-CAPACITY-BOUNDARY** (a genuine geometric-degeneracy finding about
held-out-entity keys at K/d>0.75 — would turn the d=96 "flat at ceiling"
reading into a real, arguably more-publishable measured boundary), (b)
**INSTRUMENT-BUG** (would license a wide-grid re-run at a fixed probe),
(c) **TOLERANCE-MISCALIBRATION** (would unlock the already-collected 11
inadmissible cells for the fit at zero new GPU spend). Candidate (1) (a
fixed-batch checkpoint-payload extension for future waves) is registered
as a same-commit secondary build task, not separately costed. Cost,
re-derived from K=84's own §15.22 realized rate: 1× 0.450 GPU-h, 2×
0.900 GPU-h — **fits inside the ORIGINAL, non-extended 21 GPU-h
KEY_ANCHORING_SCALING ceiling even at 2×** (18.1196 realized + 0.900 =
19.0196/21, 1.9804 GPU-h margin untouched; the already-authorized but
unused `+5.0 GPU-h` extension is not needed). Planned: GPU 2 (Phase-2
owns GPUs 0–1, GPUs 3–7 idle). Full 3-way discrimination table, mechanical
decision rules with exact pinned thresholds, gate table, and a 5-question
self-attack round: §15.24. `EXPERIMENT_LOG.md`'s matching one-line entry.
**Next step: an independent attack round on §15.24, per this program's
own standing multiple-independent-audit-rounds discipline — not yet
started.** No cells launched, no code written this session.

---

## NS EVAL-ADMISSION DIAGNOSTIC (2026-07-08) — CORRECTS the mechanistic
claim in the KEYANCHOR-SCALING §15.20 WIDE-GRID WAVE HARVEST block
immediately below (lines ~22-29 of that block: "the failure is confined
to Newton-Schulz convergence on EVAL-time recovery-probe queries against
the final, fully-LEARNED anchor table" is **RETRACTED at the mechanism
level** — the failure-RATE pattern that block reports (0/30 at
K/d≤0.71875 → 1/3 at K/d=0.75 → 3/3 at K/d≥0.8125) is unchanged and still
real, but the CAUSE it names is wrong).

**Verdict: MISDIAGNOSED-ARTIFACT.** Diagnosed the wide-grid wave's own
queued follow-up (that block's "Queue implication" item 1: does
`n_iter>20` restore admission on the learned anchor table?). Found,
BEFORE running any NS sweep, that all 12 originally-failing cells'
`checkpoint_fallback_seen=True` flags trace 100% to
`C17_heldout_entities` — a recovery-probe pool that is architecturally
**anchor-bypassed by construction** (its bind items are drawn from a name
pool disjoint from `pools.train_name_ids`, the exact set
`anchor_trained_mask` is built from; `anchor_blend_gather_scatter` never
touches the anchor table for these queries). The anchor table cannot be
the cause of a computation it never participates in. Ran the
pre-registered check anyway: all 6 pulled tables (4 failing-cell + 2
passing-control, `step20000.pt`) are 100% admissible at `n_iter=20`
already, residuals ~7,000-8,000× below tolerance, indistinguishable
between failing and passing cells, unchanged through `n_iter=40` — the
anchor table was never close to a convergence problem, at any K tested
(69-90). Full method, citations, per-table tables:
`KEY_ANCHORING_SCALING_DRAFT.md` §15.23; `EXPERIMENT_LOG.md`'s matching
entry. Archive: `experiment-runs/2026-07-08_ns_admission_diag/` (repo,
~330KB) + SSD mirror. Cost: ~0 GPU-h (CPU-only, 264KB of `scp` pulls off
idle box GPUs, no training launched).

**Queue implication:** item 1 from the block below ("diagnose the NS
eval-admissibility failure... check whether n_iter>20 converges on a
LEARNED, post-training anchor-table snapshot") is CLOSED — answered NO,
for a more fundamental reason than the item anticipated. Two NEW
candidates are queued in its place (named only, not designed, no cost
estimate — §15.23's own "Registered candidates"): (1) extend the
checkpoint payload to snapshot a fixed C17 diagnostic batch's raw pre-NS
`k_eff_raw`, enabling this same diagnostic on the TRUE failing object;
(2) a targeted, deterministic-seeded repro on one already-failing cell
with a full model checkpoint at `step=20000` only. Item 2 from the block
below (the K=69 contingency seed 1733) is UNAFFECTED by this correction.

---

## KEYANCHOR-SCALING §15.20 WIDE-GRID WAVE HARVEST (2026-07-08 ~00:45 UTC) — CLOSES the wide-grid wave; supersedes the KEYANCHOR-SCALING WAVE HARVEST block below's own "queued next steps" for §15.19's AMBIGUOUS d=96 result. The TRACK C RUNG-3 HARVEST block immediately below is UNAFFECTED (different lane; its own "rung-3 ALL_DONE harvest remains the other standing queue item" line is independently already stale — that harvest itself is the block below, already closed)

**MECHANISTIC CLAIM IN THIS BLOCK CORRECTED, 2026-07-08 — see the NS
EVAL-ADMISSION DIAGNOSTIC block above.** The "confined to... the final,
fully-LEARNED anchor table" sentence two paragraphs below is now known to
be wrong; do not rely on it. The K/d-correlated failure-RATE numbers
remain accurate.

**`KEY_ANCHORING_SCALING_DRAFT.md` §15.20 (Rev 1, d=96 wider-K grid +
d=80 seed escalation) ran to completion on box and was harvested: WAVE
VERDICT = AMBIGUOUS — DATA-QUALITY COLLAPSE, more severe than the
pre-registered decision rule anticipated.** The chain halted at its own
d=96-wide fit step (`set -euo pipefail`, `KEYANCHOR_SCALING_WIDE_DONE`
sentinel never written) — confirmed NOT a path/glob bug by four
independent checks (byte-identical local reproduction off-box, direct
per-JSON inspection, a combined-9-K-directory retry, and reading the
`k32_mean=None`/`k48_mean=None` red herring correctly). **Real root
cause: 11 of the 12 new d=96-wide cells (K∈{72,78,84,90}) fail the
`geo3_admission.admissible` eval-side gate — K=78/84/90 have ZERO
admissible seeds each (3/3 fail at every one).** Verified mechanistic
cause (`run_deltanet_rd.py` `compute_geo3_admission`): the failure is
confined to Newton-Schulz convergence on EVAL-time recovery-probe
queries against the final, fully-LEARNED anchor table — every failing
cell has `n_geo3_fallback_train_steps=0` (training itself is clean, h1
guard 1.0 everywhere). §15.20.1's own Wave −1 `n_iter=20` sufficiency
check passed cleanly because it only tests the STATIC frame-potential
init, not a post-20,000-step drifted table — a real, newly-disclosed gap
in that gate's own coverage. Failure rate rises sharply with K/d at
d=96: 0/30 at K/d≤0.71875 → 1/3 at K/d=0.75 → 3/3 at K/d≥0.8125.

**d=80 seed escalation (K=48/K=53, now 5 seeds each) is unaffected —
100% admissible — and REFUTES ratio-invariance more tightly than before:
`x0(80)=0.6779`, 95% CI `[0.6683,0.6867]` (width 0.0184, down from
0.0248), still excluding the invariance band `[0.4745,0.6165]` by a wide
margin.** This fit was never run on box (chain halted one step earlier);
this harvest ran it for the first time, locally.

**§15.20.4's own rival-discrimination test — the wide grid's entire
reason for existing — never executed.** No `CI(x0,96-wide)` exists at
all; the registered `--k-grid 69 72 78 84 90` fit cannot be attempted
with any data this wave collected, reproduced identically off-box. A
diagnostic-only (non-registered) 6-K fit using every K with ANY
admissible data (24,51,57,63,69,72, the last at n=1) gives `x0=0.7716`
CI `[0.7700,0.7841]`, `degenerate_frac=26.2%` (exceeds the 10% bar) —
descriptively just outside the abs-slack band and just inside the
power-law band, a real lead but explicitly not a licensed result.

**Realized 6.3331 GPU-h** (16 new cells, summed `wall_s`, independently
cross-checked against on-box file timestamps to within 0.32s program-
wide) **— 95.2% of the design's own 1× point estimate, 47.6% of the 2×
bracket.** KEY_ANCHORING_SCALING sub-ledger: 11.7865 (§15.19) + 6.3331
(this wave) = **18.1196/21 GPU-h realized, reserve 2.8804/21** — fits
inside the ORIGINAL ceiling; the `+5.0 GPU-h` extension token
(`KEYANCHOR_SCALING_EXT_PI_SIGNOFF`) was authorized and its gate fired
correctly but was never actually drawn on.

**LEDGER CORRECTED 2026-07-11 (§15.24 Rev 4 MAJOR-2, C17 attack round
4):** the `18.1196/21` figure above never included the K=69/seed=1733
contingency cell's own realized 0.427 GPU-h (fired standalone, AFTER this
wave, `wall_s=1535.2s`, §15.22 addendum — see the "KEY-ANCHORING K=69/d=96
CONTINGENCY SEED 1733" block below). **Corrected running total: 11.7865 +
6.3331 + 0.427 = 18.5466/21 GPU-h realized, reserve 2.4534/21** — this
wave's own 6.3331 figure and this block's other findings are otherwise
unaffected/unretracted; only the running-total figure was stale. Every
`18.1196`/`19.0196`/`19.9196`-derived figure elsewhere in this file's
history predates this correction and is left verbatim per house style
(the historical record of what was believed at each point in time); the
CURRENT figure is `18.5466/21`, per `KEY_ANCHORING_SCALING_DRAFT.md`
§15.24.7.

Full verdict, 19-cell verification table, root-cause derivation, local
re-fits, and the mechanical 6-row decision-rule walk:
`matrix-thinking/KEY_ANCHORING_SCALING_DRAFT.md` §15.22. Archive:
`experiment-runs/2026-07-07_keyanchor_scaling_wide/` (repo, ~78MB, all
files ≤25MB) + SSD mirror (full parity, byte-verified).
`EXPERIMENT_LOG.md`'s matching entry.

**Queue implication:** the wide-grid wave is CLOSED. Not launched by
this harvest, both queued: (1) diagnose the Newton-Schulz eval-
admissibility failure (candidate: check whether `n_iter>20` converges on
a LEARNED, post-training anchor-table snapshot before committing to a
re-run) — a design decision for whoever builds the follow-up, not
self-authorized here; (2) the already-reserved K=69 contingency seed
(1733) remains registered but is known-insufficient on its own (~4% CI
narrowing, per §15.20.4's own MAJOR-2 power check). No other program's
decision tree is gated by this wave (mirrors §15.19's own "Path (iii)
never gates, and is never gated by" framing, restated one wave later).

---

## TRACK C RUNG-3 HARVEST (2026-07-07 ~20:45 UTC) — CLOSES the rung-3 lane (Option-A harvest); supersedes ALL rung-3/trackc tracking in every block below (the "~05:00 UTC Jul 8 ALL_DONE" ETA is dead — see disclosure)

**SCALE PROGRAM COMPLETE AT 4 POINTS. The 4-point ladder is MONOTONIC —
span_frac 0.248 (14M) → 0.344 (98M) → 0.389 (392M) → 0.455 (1.31B), all
on the held-fixed extended mixes — the write-geometry attractor WORSENS
with pure scale through 1.31B params.** §5.7's literal 3-rung criterion
reads "persists/worsens", the pre-registered headline direction. Full
verdict + tables: `matrix-thinking/SCALE_TRANSFER_DESIGN.md` §5.11;
EXPERIMENT_LOG "SCALE-TRANSFER Track C Wave 3 (rung-3) harvest" entry.

**PROMINENT DISCLOSURE:** rung-3's training never reached `ALL_DONE` —
both cells **self-terminated at their own `--internal-timeout` at steps
155,081/155,028 of 183,105 (~84.7% of the token-matched budget, ≈1.27B
of 1.5B tokens)**. Clean designed shutdown, not a crash (loss curves
clean, `skip_rate=0.0`, ckpts every 1k steps through 155,000;
`complete=false, timed_out=true`). Root cause: the banked solo
calibration constant (0.7135 s/step) underestimated the two concurrent
runs' real steady-state rate (1.416 s/step) by 1.985×, consuming the
1.6× margin (derived internal timeout 219,631 s; both cells stopped
within 13 s under it) — the mid-run ETA disclosure (2026-07-06 ~07:35
UTC) was never propagated to the running cells' timeout. No
optimizer-state resume exists; budget guard correctly blocked a ~144
GPU-h restart; supervisor stopped via `STOP_trackc3`. **Option A
(harvest at 155k + disclosure) adjudicated over restart** on the plateau
check: span_frac at 130k/140k/150k/155k = 0.4584/0.4573/0.4566/0.4554 —
flat, mildly declining; extrapolated to 183k ≈ 0.452, a −0.003
correction vs the +0.066 rung-2→rung-3 increment. Two [LEARN]s
registered (optimizer state in checkpoints as a build requirement;
timeouts must track measured steady-state rate, not calibration
constants).

**Ledger (realized, from the cells' own wall_s):** wave-3 = 219,618.09 +
219,618.42 s = **122.01 GPU-h** (vs 76.25 booked) → program total
**312.23/300 GPU-h, OVER the ceiling by 12.23, disclosed** (the earlier
≈334/300 projection assumed a full-length run; the timeout capped it).
`PROGRAM_SPENT_GPUH` 266.47 → 312.23 in `run_lm_rd_trackc_sweep.py`
(repo + box, md5-matched). No further Track C training exists to launch.
Probe cost 0.49 GPU-h (GPU 0 only; smoke gate 6/6 PASS; pooling method
re-validated against archived rung-1/control numbers to ≤4.5e-7 before
scoring). Archive: `experiment-runs/2026-07-06_trackc_rung3/` (repo
≤25MB) + SSD mirror (incl. the 2×69MB raw training JSONs, SSD-only).
Checkpoints stay on box (`/data/lm_rd_trackc_ckpts/wave3/`, 155/run).

**Queue implication:** the PRE-COMPACTION SNAPSHOT's queue item 2 step 1
(this harvest) is DONE; **its step 2 (paper addendum) is now ALSO DONE
(2026-07-07, paper-addendum pass):** all 3 `[PENDING RUNG-3]` `\todo{}`
markers resolved with the real numbers + disclosure
(`iclr-2027/sections/09_discussion_limitations.tex`,
`10_conclusion.tex`); the completed monotonic 4-point ladder, the
super-linear-capacity finding (`KEY_ANCHORING_SCALING_DRAFT.md` §15.19,
d=96 wide-grid marked in-flight, not pre-claimed), and the double
PROBE-INVALID/format-exonerated result (`REASONING_LINK_DESIGN.md`
§15/§16.8) folded into `04_phenomenon.tex`/`05_mechanism.tex`/
`08_results.tex`/`09_discussion_limitations.tex`/`10_conclusion.tex`;
3 missing intro contribution bullets added to `01_intro.tex`; 3
related-work citations (VLA 2605.11196, Frozen-QK 2506.01115, Okpekpe
\& Orvieto 2508.19029) added to `02_related_work.tex`; `make_figures_v2.py`
Fig 9 Panel A extended to the 4th ladder point and regenerated; the
super-linear-capacity update also folded into `workshop-2026/sections/
04_open_question.tex` and `05_limitations.tex`; both stale handoff
blocks (PRE-COMPACTION SNAPSHOT, SESSION HANDOFF) deleted from this
file, consumed by this pass. REASONING-LINK's 2 deferred
rung-3 Leg-B cells are unblocked in the narrow sense that trackc's lane
is closed and GPUs 0-1 are free — but they would run on the step-155k
(84.7%-budget) checkpoints; whether that's acceptable is a design-owner
call under REASONING-LINK's own probe-invalid verdict (rungs 0-2
already showed the identical pattern; see the Phase-1b block below).

---

## REASONING-LINK PHASE-1B GATE-TEST NULL (2026-07-07 ~20:21 UTC) — supersedes the REASONING-LINK PHASE 1 HARVEST SNAPSHOT block's own queue implication below (that block's PROBE-INVALID Phase-1 finding is otherwise still current/unchanged), and supersedes IMMEDIATE QUEUE item 1(b) in the PRE-COMPACTION SNAPSHOT block further below (stale, pre-dates Phase 1's own launch); the KEYANCHOR-SCALING WAVE HARVEST block immediately below is UNAFFECTED (never gated, and never gates, this block's own decision tree, §16.6 step 6)

**Phase-1b (`REASONING_LINK_DESIGN.md` §16.1/§16.8) ran on box and is CLOSED-NULL: both natural-language candidates (A — gift-verb, reserved query marker dropped; B — succession-family verbs) return `gate_result_h1_probe_valid=False` on the licensing WIKITEXT-mix-ext cell, identically to Phase 1's marker-template result** — `recovered_frac`(h1)=0.0, premises (iii)/(iv) both fail their null-relative gate, at all 4 gate cells (2 templates × {wikitext-licensing, openr1-expected-null}). The openr1 "expected-null" cells also fail as predicted (no §16.1.3 item 2b confound flag). Per §16.1.3 item 1's own pre-registered reading, two structurally different templates failing the SAME way on the licensing checkpoint is the single most informative null this instrument can produce — the failure is NOT attributable to surface form. **This mechanically PROMOTES Path (ii)/Phase-2 per §16.6's trigger table (row 4).** The conditional ≈0.4 GPU-h full Phase-1b grid was correctly NOT launched.

**Gate enforcement worked as designed** — `reasoning_link_gate_enforce.py` (built to close Phase 1's own §15.2/[LEARN] discrepancy, where the launch script computed but never read `gate_result_h1_probe_valid`) read the real exit code, refused, wrote `STAGE0_GATE_REFUSED`, exited nonzero. First real-failure exercise of that fix; it held.

**Realized ≈0.0088 GPU-h** (4 GPU-attached gate cells, box `stat` timestamps) — matches the §16.1.2 pre-registered "~0.01 GPU-h" almost exactly; ≈0.0244 GPU-h full single-launch wall-clock incl. CPU-only Stage −1. One clean run, zero crashes. Against the ≈24.20 GPU-h Phase-1 ceiling: ≈1.3% cumulative utilized (Phase 1 + Phase-1b combined).

Full gate table, mechanical reading, GPU-h derivation, and one disclosed path-naming discrepancy (task brief assumed `results/reasoning_link/`; box's real path is the sibling directory `results/reasoning_link_phase1b/`, no data issue): `REASONING_LINK_DESIGN.md` §16.8. Archive: `experiment-runs/2026-07-07_phase1b_gate/` (4 raw JSONs + `STAGE0_GATE_REFUSED` + 6 logs + 1 exact script, ~120KB) + SSD mirror (byte-identical). `EXPERIMENT_LOG.md` matching entry.

**Queue implication — REASONING-LINK's own decision tree now has exactly one live next action: Phase-2 (§16.2, Rev 1 landed, §16.7 fix-map: 1 FATAL/6 MAJOR/1 MINOR all fixed) needs its own fresh, independent SECOND audit pass (§16.2.4's registered prerequisite — landing attack-round-1's findings does not itself certify build-readiness) before build + independent code audit + launch.** That audit is zero-GPU, can start immediately, and runs concurrent with Path (iii)'s already-CONCLUDED-and-reported status (KEYANCHOR-SCALING WAVE HARVEST block below, §16.6 step 6: Path (iii) never gates, and is never gated by, this decision tree). Phase-2's recipe-pinning step (§16.2.1) proceeds with the original marker template (its own registered fallback) since neither Phase-1b candidate cleared a gate to supersede it. This supersedes the now-stale IMMEDIATE QUEUE item 1(b) below (written before Phase 1 itself had even launched).

---

## KEYANCHOR-SCALING WAVE HARVEST (2026-07-07 ~18:15 UTC) — supersedes IMMEDIATE QUEUE item 1(a) in the PRE-COMPACTION SNAPSHOT block below; the REASONING-LINK PHASE 1 block immediately below is UNAFFECTED (this wave never gated, and was never gated by, that block's own decision tree)

**`KEY_ANCHORING_SCALING_DRAFT.md` §15 (Path (iii) of `REASONING_LINK_DESIGN.md` §16.3/§16.6) ran to completion on box (`KEYANCHOR_SCALING_PI_SIGNOFF=1`, past the §15.18 PARK gate, after REASONING-LINK Phase 1 landed) and was harvested: WAVE VERDICT = AMBIGUOUS (mechanical, pre-registered §15.10) — not a judgment call.** Tests whether the d=64 capacity cliff (`x0=0.5455`, §12.9) is invariant IN RATIO TERMS as `d_state` grows to 80/96, holding the SAME organic-coherence regime (`n_entities=107>d_state`) fixed. d=96's fit is degenerate (`degenerate_frac=98.5%`, far past the pre-registered 10% bar) because `h4` sits flat near ceiling (0.98–1.0) across the ENTIRE tested K/d window — mechanically the same "no cliff in the window" signature §13.10 found at d=128, reproducing one `d_state` earlier than previously seen. Per §15.10 item 4's own text, any AMBIGUOUS `d` makes the WHOLE wave AMBIGUOUS, regardless of the other `d`'s result.

**d=80 alone is a clean, non-degenerate result and REFUTES ratio-invariance:** `x0(80)=0.6756`, 95% CI `[0.6620,0.6868]`, `degenerate_frac=0.0` — excludes the pre-registered invariance band `[0.4745,0.6165]` entirely, on the high side (gap 0.0455). Closer to the "absolute-slack" rival's own `x0(80)≈0.636` prediction than to the ratio-invariant band center (0.5455); d=96's own descriptive (not statistically clean) decline also concentrates near that rival's `x0(96)≈0.697` prediction. The "fixed-K" rival (`x0(96)≈0.364`) is directly contradicted: K=51/d=96 (ratio 0.531, well past fixed-K's predicted trouble zone) still reads `h4=1.0`.

**Two disclosed anomalies, neither changing the verdict:** (1) one admissibility FAIL, first of its kind in the program's archive history (`d=96, K=69, seed=1730`: `geo3_admission.admissible=false`, `checkpoint_fallback_seen=true`; every other admissibility field on that cell is clean; sensitivity re-fit excluding it leaves d=96 degenerate regardless, 94.8% vs 98.5%); (2) the pre-registered seed-contingency range trigger (§15.14) fires at K=48/d=80 and K=53/d=80 (3-seed h4 range 0.218/0.178, both >0.15 — the two K-groups straddling d=80's own fitted transition), queued as a follow-up, not yet run.

**Realized 11.7865 GPU-h** (summed `wall_s` across all 30 cells, on-box timestamps, `run1`+`run2` chain logs) **against the Tier-1 approved ceiling `H_scaling=21 GPU-h`** (mandatory-only 2× bracket: 20.956, §15.12) — **56.2% used**, and the first wave in this program to land ABOVE its own 1× point estimate (10.478 GPU-h, 112.5%) rather than comfortably under it (prior waves: 13.6%–87.0% of bracket). This is a NEW sub-ledger, not an extension of the exhausted 80/80 KEY_ANCHORING ledger (§15.12's own registered framing) — tracked separately: **KEY_ANCHORING_SCALING sub-ledger: 11.7865/21 GPU-h realized, reserve 9.2135/21.** **GPUs 2-7 now idle.**

Full verdict, 30-cell verification table, independent re-fit, rival comparison, disclosed critique of the mechanical AMBIGUOUS rule against a flat-not-noisy fit: `matrix-thinking/KEY_ANCHORING_SCALING_DRAFT.md` §15.19. Archive: `experiment-runs/2026-07-07_keyanchor_scaling/` (30 raw JSONs + 34 per-cell logs + 2 chain-session logs + fit outputs + pins + kernel-safety artifact + 4 exact scripts byte-verified zero-drift against both box and repo, ~99MB, all files ≤25MB) + SSD mirror (full parity verified, `diff -rq` clean).

**Queue implication: none for REASONING-LINK's own decision tree** — `REASONING_LINK_DESIGN.md` §16.6 step 6 pre-registered this exactly: "Path (iii) grid complete (any outcome) → report to the capacity-law paper track; does not change anything in this decision tree." Next: §16 Path (i) (Phase-1b Stage-0-gate-only build) + Path (ii) (Phase-2 attack round) continue, unaffected — both already in flight per the REASONING-LINK block below's own step 3/4. rung-3 `ALL_DONE` (~05:00 UTC Jul 8) harvest remains the other standing queue item, unblocked by anything above.

---

## REASONING-LINK PHASE 1 HARVEST SNAPSHOT (2026-07-07 ~12:45 UTC) — supersedes the PRE-COMPACTION SNAPSHOT block below for REASONING-LINK status; that block's rung-3/trackc tracking is otherwise still current

**REASONING-LINK Phase 1 (Leg A 60 cells + Leg B rungs 0-2, 18 cells)
ran to completion on box and was unblinded: PROBE-INVALID, not a
licensed CONFIRM/REFUTE.** `recovered_frac@0.9` (Option 1) is exactly
`0.0` at all 78 cells × 4 h = 312/312 readings — the readout never
fires on ANY checkpoint (14M-392M, both corpora, every arm/K/surgery).
Stage 0's own registered h1 sanity floor fails both its null-relative
and absolute-0.10 conditions, and premises (iii)/(iv) fail their own
null-relative gate at 0/78 cells. Per the design's own §8.4 rule
("failure routes to probe-invalid, not to REFUTE"), this is NOT a
negative result about H_LINK-A/H_LINK-B — it is the instrument itself
being invalid for this checkpoint family, caught by its own
pre-registered gates. **Discrepancy, flagged prominently: the launch
script (`reasoning_link_chain.sh`) never wired `gate_result_
h1_probe_valid` (or `marker_disagreement_flag`) into an actual abort
branch — only the wall-clock cost check is a real gate — so the grid
ran to completion past a failed validity check that, per the design's
own §9 registration, should have halted it before Stage 1 launched.**
Realized cost was small regardless (≈0.29 GPU-h for runs 6-8, ≈0.373
GPU-h total incl. 5 pre-launch build crashes, vs the ≈24.20 GPU-h
Phase-1 ceiling — ≈1.2% utilized; rung-3, the expensive row, was never
in scope this run). Full verdict, gates table, per-cell numbers, and
the `[LEARN]` writeup: `REASONING_LINK_DESIGN.md` §15;
`EXPERIMENT_LOG.md`'s matching entry.

**Queue implication:** rung-3's 2 Leg-B cells remain DEFERRED pending
trackc `ALL_DONE` (unchanged, ≈05:00 UTC Jul 8 ETA per the block below)
— when they land, they get harvested the same way, but rungs 0-2
already show the identical probe-invalid pattern a rung-3 result would
need to have differed from to change anything. Before any re-run of
this instrument: (a) wire the missing gates into the chain script as
real abort branches (the `[LEARN]` item), (b) reconsider whether an
absolute-cosine≥0.9 threshold is right for never-task-trained
checkpoints at all — `cos_mean` across the grid sits in `[-0.33,0.25]`
with `cos_std` `[0.03,0.20]`, so `|cos|>0.9` is a >10σ event under the
measured distribution, consistent with a threshold/mechanism mismatch
rather than proof of zero composable structure. **Not decided here —
a PI/design-owner call**, per §15.10.

**Archive:** `experiment-runs/2026-07-07_reasoning_link_phase1/` (82
raw JSONs + 89 logs + 3 exact scripts, ~1.3MB) + SSD mirror.

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

