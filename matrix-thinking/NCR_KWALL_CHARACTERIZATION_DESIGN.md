# NCR K-WALL CHARACTERIZATION — K∈{26,28,30} ON THE LIVE K=24 RUNG

**STATUS: DRAFT-R1 — POST-AUDIT-1, AWAITING AUDIT ROUND 2 (not
build-released, not queue-eligible).**

**Mandate.** `NCR_KLADDER_DESIGN.md` §A4-ADJUDICATION (2026-08-06,
`matrix-thinking/NCR_KLADDER_DESIGN.md:1999-2004`) stamped the K-ladder
design SPENT and named this document as the successor for design-round
dispatch: *"a K∈{26,28,30} wall-characterization anchored on the LIVE
K=24 rung, aimed at the RECOVERY leg (the leg A4.4 shows was never the
target of any prior fix). Enters the standard ceremony (design draft →
internal-archive sweep incl. verifying 26–30 is genuinely open → audit
→ adjudication) before any cell exists."* The same recommendation is
independently recorded at `NCR_KLADDER_ATTACK_R2.md:776-782` (§6,
finding 4): *"the wall sits between K=24 and K=32 and has now been
measured three times without being crossed. Characterizing that wall
at K∈{26,28,30} on the K=24 recipe is a handful of GPU-hours, is
anchored on a live rung rather than a dead one, and answers a question
the archive actually leaves open."* `STATE.md:11-13` records this
document's own filename as the in-flight successor. This draft
supersedes no other document's verdict; `NCR_KLADDER_DESIGN.md` stays
SPENT and the 07-12 mapping-law CLOSED/BLOCKED ruling (§3 below) is
treated as fixed ground truth, not re-litigated.

---

## §1 HYPOTHESIS (one sentence)

**Rev 1 restates this hypothesis per §A1-ADJUDICATION D1/D4 (R0 had an
unqualified "wall" framing and reported the rank leg as non-gating;
see §R1 below for the full disposition table).**

Under the exact recipe that already brackets a CONVERGED-ROBUST rung
(K=24, n=12, `indist_min=1.000` uniformly) against a measured
80K-budget non-ROBUST rung (K=32, 0/4 CONVERGED at 1×) — earlyln
free-write, tight-spare `d=K+1`, encoder hidden `h=64` fixed
(non-binding throughout this range), 80,000 training steps as the
PRIMARY, explicitly budget-disclosed instrument — the runner's own
**conjunctive** Gate-1 CONVERGED verdict (`indist_min =
min(recovered_frac@0.9` for `h∈{1,2,3}) ≥ 0.9` **AND**
`aer_mean ≥ 0.9·K`, `_cell_gate1`, `ncr_earlyln_scale.py:317-329` —
stated as a conjunction here per D4, not the recovery leg reported
"not gating" as R0 wrongly had it) resolves into additional K-scale
structure inside the untested K∈{26,28,30} window. This is now an
explicitly **budget-conditioned** question, not a permanent-
trainability one: `experiment-runs/2026-07-12_ncr_k32_budget/`
(re-derived directly from the raw JSONs, §3) shows the SAME
conjunctive Gate-1 rate at K=32 moving **0/4→1/4→2/4 CONVERGED across
1×/2×/4× budget** — so any K∈{26,28,30} rung reading sub-ROBUST at 80K
could be genuinely walled or merely slow-converging, and this design's
headline is **the 80K-budget convergence frontier over
K∈{24,26,28,30,32}**, never "the wall" (D1). The pre-registered
CONDITIONAL 4-cell 160K arm (§4, §5) is the built-in disambiguator
between those two readings, triggered only at the K where it is
needed. This remains answerable for a low, capped GPU-hour spend (§4,
≤15 GPU-h inclusive of the conditional arm) without touching the rank
leg's own separate accounting (it is a gate COMPONENT, not a separate
signal — D4), the ortho/NS-polar mechanism, or launching any fresh
K≥32 cell (the K=32 leg of the disambiguation reuses already-archived
data, §4, §7).

---

## §2 CONFIG FAMILY — exact, with the evidence that picks it

**Choice: tight-spare `d=K+1`, `h=64` fixed, earlyln free-write
(no Newton–Schulz / orthogonal projection), 80,000-step budget, run via
`ncr_earlyln_scale.py --cell --K {26,28,30} --d-override {27,29,31}
--seed {0,1,2,3} --steps 80000` — the exact `--d-override` mechanism
`NCR_NEXT_LEVER_DESIGN.md` §2.1/§5 built and that Probe A already ran
at K=16/K=24 (`ncr_earlyln_scale.py:212-250`, docstring at `:219`: *"lets
a job set..."*). NOT the `d=2K` parked-relic convention
(`GRID_SHAPES`, `ncr_earlyln_scale.py:75-93`), and NOT
`ncr_ortho_write.py --arm free`'s 320K-step / hardcoded-far-depth
harness.** Both rejections are evidence-based, not a coin flip:

**(a) `d=K+1` is the config the LIVE anchor is measured under; `d=2K`
is not.** `STATE.md:114-116` calls out the load-bearing finding in
these words: *"free-write K=24 / d=25 recovers 1.0 at ALL far depths,
cond 1.0, 4/4 seeds — the wall is K=32-specific, K=24 already works."*
`d=25 = K+1`, not `d=2K=48`. **Rev 1 correction (KW1.5, D6):** R0
cited the 2026-07-12 mapping-law Q2 n=4→n=12 seed extension as
`d=2K=48` evidence against that convention. It is not — every one of
those 12 cells (`q2_K24_seedext/earlyln_K24_s{4..11}.json` and
`q2_K24_seedext_orig0-3/earlyln_K24_s{0..3}.json`) carries `"d": 25,
"d_override": 25`, re-verified by direct `json.load` this revision —
i.e. it is `d=K+1` evidence, this design's OWN config, not the
rejected one (§3's table already used this same run set correctly, as
`d=K+1` evidence — R0 contradicted itself between §2(a) and §3 on one
run set; this revision reconciles both to the correct reading). The
conclusion ("`d=2K` is not clean") **survives on other evidence, cited
correctly here:** `experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K24_s{0-3}.json`
(K=24, `d=48=2K`, 80,000 steps) reads `indist_min=0.000` in **4/4**
seeds, `AER/K=0.734–0.746` — DEAD by both legs, re-derived by
`json.load` this revision; and
`experiment-runs/2026-07-12_ncr_earlyln_budget2x/earlyln_K24_s{0-3}.json`
(same config, 160,000 steps, 2×) reads `indist_min=0.000` in **4/4**,
`AER/K=0.733–0.758` — still DEAD at double budget. This is a stronger
and cleaner rejection than the misattributed far-depth figure ever
was: `d=2K` at K=24 fails the PRIMARY recovery leg outright, at two
budgets, not merely the noisier far-depth leg. K=32's own full `d(K)`
grid (`d∈{33,40,48,64}` — `d=K+1=33` INCLUDED, not only `d=2K=64`)
independently reads **uniformly TRAINABILITY-DEAD** at every `d`
tested — *"every arm lands TRAINABILITY-DEAD (0/4 fully Gate-1
CONVERGED)... front is pinned at the trivial K−3=29 rung in all 16
cells, every d, every seed — zero far-depth signal anywhere"*
(`EXPERIMENT_LOG.md:8632-8637`). Because that sentence covers `d=K+1`
too, it does **not** by itself discriminate between the two
conventions at K=32 (R0's aggravator error, also KW1.5) — it
establishes only that K=32 fails broadly at 80K/1×, addressed on its
own terms, budget-inclusive, in §3 below. There is no clean anchor
anywhere in the archive under `d=2K`, at any budget tested; there is
one under `d=K+1`, at 80K (K=24) — not yet confirmed at K=32 at any
budget (§3). A wall-characterization study needs a live point to
characterize distance from — only `d=K+1` supplies one.

**(b) The `d=2K` underperformance has an independently confirmed
mechanism, which is further reason not to chase it here.** The 07-12
Q3 mechanism analysis (CPU-only, zero GPU, independently opus-audited
CLEAN) found the over-parameterized `d=2K` write space produces
**7.1×–14.1× more normalized leakage** than `d=K+1` at the same K, with
the leakage moving into the entity-adjacent cross-terms rather than
staying self-contained (`EXPERIMENT_LOG.md:8543,8560-8578`). More
headroom trains *worse* here, not better — consistent with `d=K+1`
being the convention worth extending, not `d=2K`.

**(c) `ncr_ortho_write.py --arm free` (the harness that produced the
STATE.md-cited clean K=24/dead K=32 result) is NOT reused verbatim
here, for a concrete, code-verified reason: its far-depth ladder is
hardcoded, not K-parameterized, and would silently break at all three
new K's.** `ncr_ortho_write.py:87-89`: `REALISTIC_DEPTHS = (5, 12, 20,
29, 40, 61)`, `PRIMARY_HSTAR = 40` — literal K=32 values (`29=K-3`,
`40=K+8`, `61=2K-3`), asserted novel only via `r = h % K; assert r not
in (0,1,2,3)` (`ncr_ortho_write.py:248-251`), never re-derived per K.
Checked mechanically for this design (not assumed, per CLAUDE.md's
exact-threshold rule): at K=26, `29 % 26 = 3` — **hits the forbidden
set, hard `AssertionError`**; at K=28, `29 % 28 = 1` — same failure; at
K=30, `61 % 30 = 1` — same failure. All three new K's would crash this
harness's far-depth eval outright. This is exactly the class of
mod-K periodicity trap CLAUDE.md's hard rules warn about (fixed depths
reused across a K sweep silently collapsing into trivial residues).
The alternative — `ncr_earlyln_scale.py`'s `nt._gen_grid(K)` — is
already K-parameterized (`ladder_residue=K-3`, `ladder = m·K-3` for
`m∈{1,2,4,8,...}`, `h_star=8K-3`), already regression-tested against
every hand-typed K it was meant to reproduce
(`ncr_task.py:150-152`), and was mechanically extended to
K∈{20,32,48,64,96,128,192,256} by the identical additive pattern this
design now reuses for K∈{26,28,30} (`ncr_task.py:154-162`). No new
residue arithmetic is invented; the far-depth (Gate-2/secondary) ladder
comes free and correct from code already in the repo.

**(d) `h=64` fixed is reused unmodified and is NOT the constraint in
this range — verified, not assumed.** The binding rank ceiling is
`min(d, h+1)`. For K∈{26,28,30}, `d=K+1∈{27,29,31}` and `h+1=65` in
every cell, so `min(d,65)=d` always — `h` never binds, exactly as it
never bound at K=16/24/32 either (`h(K)=2K` was a fix `NCR_KLADDER_DESIGN.md`
needed only at K≥64, `ncr_earlyln_scale.py:75-93`'s own `GRID_SHAPES`
table already fixes `h=64` through K=32). No `h(K)` change, no
`GRID_SHAPES` edit to the `h` field, is licensed or needed by this
design.

**(e) Budget: 80,000 steps, matching the parked-relic cost SHAPE per
the dispatch mandate, NOT the 320K budget the clean STATE.md anchor
was measured at — flagged honestly, not silently substituted.** The
STATE.md "recovers 1.0 at ALL far depths" K=24 result is a 320K-step
cell (`NCR_ORTHO_WRITE.md`'s CEILING AMENDMENT, `:353-355`, `:187-193`).
At 80K steps the SAME `d=K+1` K=24 recipe is *also* clean on the
PRIMARY metric this design uses (§3 table below proves this directly
from the raw JSONs, not by assumption) but is *not* clean on the
secondary far-depth metric (§3's Q2 citation). §3 and §5 build this
distinction into the design rather than paper over it — this is the
one place this document deliberately trades completeness for the
mandate's cost ceiling, and it is disclosed here, in §3, and again in
§7 (non-goals), not just once.

**Build note (additive-only, mirrors the existing 2026-07-11
extension exactly, not new code pattern):** `GRIDS[K]` and
`GRID_SHAPES[K]` currently have no entries for K∈{26,28,30} —
`nt.claim_config(K,...)` (`ncr_task.py:196-208`) and the CLI's
`--K` (`choices=sorted(GRID_SHAPES)`, `ncr_earlyln_scale.py:857`) both
require `K` to already be a dict key. The build adds, verbatim,
`GRID_SHAPES[26]=dict(d=52,h=64)`, `[28]=dict(d=56,h=64)`,
`[30]=dict(d=60,h=64)` (the `d` field is unused by `--d-override`
cells but kept schema-consistent) and `for _K_new in (26,28,30):
GRIDS[_K_new] = _gen_grid(_K_new)` alongside the existing loop at
`ncr_task.py:161-162`. This does not touch any existing K's entry
(the file's own regression assert at `:150-152` only checks
K∈{14,15,16,24} and is unaffected) — the exact "additive only" discipline
`ncr_earlyln_scale.py:80-84` already documents for the last K-ladder
extension, and the discipline `NCR_KLADDER_ATTACK_R2.md` finding A4.13
found this program had NOT been honoring for the `h(K)=2K` build (this
design does not repeat that mistake: nothing shared is mutated in
place, only new keys are added).

---

## §3 INTERNAL-ARCHIVE SWEEP

**Rev 1 note (D1/D6):** R0's sweep proved K∈{25..31} is open on the
K-string axis (correct, independently re-confirmed below) but never
swept the BUDGET axis at this design's own primary metric — a miss
the audit (KW1.2) traced mechanically to method: every one of R0's
five searches was keyed on the literal string `K`, which is invisible
to a budget wave run at a FIXED K. This revision re-runs the sweep
keyed on the axes the design's decisions actually turn on, states
those axes explicitly (KW1.2's discharge condition), and folds the
budget-response finding into every downstream section (§1, §4, §5).

**Axes swept, Rev 1 (explicit):**
(i) K-string axis (unchanged method from R0, reconfirmed below).
(ii) Every prior measurement of `indist_min`/Gate-1 CONVERGED at ANY
training budget, at `d=K+1`, any K.
(iii) Every prior recorded ruling on budget-vs-convergence for this
arm.
(iv) The on-box queue pool's live directories — **only partially**:
this session swept the repo-tracked `matrix-thinking/queue/jobs/pending/`
only; `~/queue/{fallback_pool,claimed}` on the box were NOT swept
(KW2.7 — R0 claimed this coverage and it was false; corrected below).

**(ii)/(iii) — the budget axis, the miss corrected.** Directly
re-derived from the raw JSONs this revision (not cited from prose):

| K | d | budget | seed 0 | seed 1 | seed 2 | seed 3 | CONVERGED rate |
|---|---|---|---|---|---|---|---|
| 32 | 33 | 1× (80,000 steps) | 0.4643 | 0.5170 | 0.6875 | 0.8711 | **0/4** |
| 32 | 33 | 2× (160,000 steps) | 0.7944 | 0.9015 | 0.5818 | 0.8865 | **1/4** |
| 32 | 33 | 4× (320,000 steps) | 0.8754 | 0.9124 | 0.9118 | 0.8965 | **2/4** |

(`indist_min` values shown; "CONVERGED rate" applies the FULL
conjunctive gate — `indist_min≥0.9` **AND** `aer_mean≥0.9×32=28.8`,
per D4 — not `indist_min` alone. Sources:
`experiment-runs/2026-07-12_ncr_mappinglaw_wave1/dratio_K32_d33/earlyln_K32_s{0-3}.json`
(1×), `experiment-runs/2026-07-12_ncr_k32_budget/budget2x_earlyln_K32_s{0-3}.json`
(2×), `.../budget4x_earlyln_K32_s{0-3}.json` (4×);
`eval.points[h].reads.binexp['recovered_frac@0.9']` for `h∈{1,2,3}`,
`deep_probe.A_eff_rank`, `train.step`, `status` — all 12 cells
independently `json.load`-verified this revision: 12/12
`status=COMPLETED`, `d==33`, `train.step` matches the requested budget
exactly.) Every seed improves with budget; the rate improves
monotonically 0/4→1/4→2/4, never regressing, but never reaches the 3/4
CONVERGED-ROBUST bar even at 4×. This is the exact archive record
`EXPERIMENT_LOG.md:8845` calls *"THE FINAL K-AXIS PROBE"* and *"This
CLOSES the K-axis book at K=32"* — cited and adjudicated below (KW1.8/
D3), not silently missed as R0 left it.

Corroborating context on a DIFFERENT K (not this design's own
anchor's config, cited for pattern only): `EXPERIMENT_LOG.md:8495-8496`
records K=16, `d=32=2K`, 320K-step budget escalation: *"Gate-1
convergence keeps improving (1/4→3/4→4/4 CONVERGED across
1×/2×/4×)"* — the SAME qualitative budget-responsiveness, on the
rejected `d=2K` convention, at a different K. Budget-responsive slow
convergence is not `d=K+1`-specific or K=32-specific in this archive;
it is a general pattern, which strengthens rather than weakens the
case that an 80K-only read cannot be trusted as a permanent
trainability verdict.

**Consequence, stated plainly (D1).** R0's §2(e)/§3/§7 argued
80K↔320K equivalence using ONLY the K=24 comparison, where
`indist_min` is pinned at 1.000 in all 12 seeds — a saturated point
with zero power to detect a budget effect. The comparison that DOES
have power (K=32, unsaturated 0.46–0.91 band, the same band
K∈{26,28,30} will land in) shows non-equivalence. This design's
headline is corrected accordingly: **the 80K-budget convergence
frontier**, with the CONDITIONAL 160K arm (§4/§5) as the built-in
disambiguator, never a permanent "trainability wall" claim from the
80K read alone.

**(i) K-string axis, reconfirmed, with KW2.7's scope corrections
applied:**
- `grep -rn` for `K=2[5-9]`, `K=3[01]`, `K25`..`K31` across
  `EXPERIMENT_LOG.md`, every `matrix-thinking/*.md`, `STATE.md`,
  `matrix-thinking/KILL_LIST.md` — two textual hits, both false
  positives (`EXPERIMENT_LOG.md:8569`: K24's `d=25=K+1`, not a K=25
  cell; `NCR_MAPPING_LAW_DESIGN.md:471`: `"1.25K=30"`, a `d`-value not
  a K-value). **Correction (KW2.7):** a broader loose search finds
  MORE hits, all adjudicating non-NCR — `EXPERIMENT_LOG.md:4690` (a
  KEY-ANCHORING sigmoid-transition width, "K≈31-39", not an NCR cell),
  `EXPERIMENT_LOG.md:1860` (training-step notation "18K→25K"), the
  DeltaNet lowercase-`k` truncation-rank family
  (`DELTANET_CAUSAL_RANK_DESIGN.md`), and `stageg/task_he.py`'s
  alphabet-size bound. R0's "only two hits" undercounted these; the
  conclusion is unaffected (none are NCR K-values) but the claim is
  restated honestly as "the only two NCR-relevant hits."
- `grep` over `matrix-thinking/queue/jobs/**/*.json` for
  `"K": 2[5-9]`/`"K": 3[01]` and `--K 2[5-9]`/`--K 3[01]` — zero hits.
  **Correction (KW2.7):** R0 claimed this covered "pending +
  fallback_pool + claimed, all lanes." Only
  `matrix-thinking/queue/jobs/pending/` exists in this repo;
  `fallback_pool/` and `claimed/` live on the box under `~/queue/`
  (`idle_fallback_daemon.sh:28-30`) and were NOT swept this session
  (nor, despite the claim, in R0). The local result stands; sweeping
  the on-box pool is added as a mandatory pre-launch red-team task
  (§6).
- `find experiment-runs -iname "*K2[5-9]*" -o -iname "*K3[01]*"` —
  zero directories. **Correction (KW2.7):** filename-only `find` is
  insufficient by this program's own counterexample — K=20 has real,
  on-record results (4/4 DEAD, `d=40`, `NOVEL_ARCH_WATERFALL.md:4325-4348`)
  with ZERO matching filenames anywhere (never archived off-box). The
  load-bearing check is the JSON-**content** grep (above and the code
  dicts below), not this filename `find`; restated as supplementary,
  non-load-bearing.
- `archive/` grepped the same way — zero hits.
- Code-level: `GRIDS`/`GRID_SHAPES` (both `ncr_task.py` and
  `ncr_earlyln_scale.py`) list every K they define; 26/28/30 absent
  from both (confirmed by direct read, §2 above).

**Conclusion: K∈{26,28,30} is open on the K-string axis** —
reconfirmed, unchanged from R0. **What was NOT open, and is folded in
here (D1/D6): the BUDGET-response question at K=32/`d=K+1`**, which
the archive already answers in the table above, and the standing
ruling on it, next.

**KW1.8 / D3 — the standing ruling, cited and adjudicated (silently
missed in R0).** `EXPERIMENT_LOG.md:8845` (2026-07-13, the
budget-rescue harvest): *"This CLOSES the K-axis book at K=32."*
`EXPERIMENT_LOG.md:8885`: *"no further K-axis probe is recommended or
licensed."* `NOVEL_ARCH_WATERFALL.md:5071` (the same finding's full
record, §11.6), its own scope paragraph in full: *"Closed: whether
budget alone rescues K=32's tight-spare wall into anything licensing
further K-escalation — no ... no further budget probe at K=32 is
licensed or recommended by this record."*

**Adjudicated per §A1-ADJUDICATION D3 (binding, quoted verbatim):**
*"the 07-13 'closes the K-axis book' ruling is adjudicated NARROW per
its own scope paragraph ('Closed: whether budget alone rescues K=32's
tight-spare wall into anything licensing further K-escalation — no'):
it bars upward escalation (K≥48 stays BLOCKED; parked cells stay
parked), it does NOT bar the below-32 characterization this design
performs, which the §A4-ADJUDICATION mandate + two judge-tier rounds
name as genuinely open."* This design's K∈{26,28,30} grid and its
CONDITIONAL 160K arm at K∈{26,28,30} (§4/§5) are both DOWNWARD
interpolation on a live rung, not further K≥32 escalation — the
ruling's own subject list (K=48's job band, the unparked 2K-reference,
`parked_k24plus`) does not include them. This design does not re-open
K=32's own escalation question; where the conditional arm's trigger
lands at K=30 (i.e. no sub-ROBUST rung is found in {26,28,30}), it
REUSES K=32's already-archived budget table (above) as a free
disambiguator rather than launching a new K=32 cell (§4, §7 non-goal
unchanged).

**The n=12 K=24 table, corrected (KW1.6, D6 — R0 selectively re-pulled
3 of 8 available cells).** All 8 re-derived by direct `json.load` from
`experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext/earlyln_K24_s{4..11}.json`
this revision:

| K | d | seed | `indist_min` | AER/K | gpu_h |
|---|---|---|---|---|---|
| 24 | 25 | 4 | 1.000 | 0.9997 | 0.519 |
| 24 | 25 | 5 | 1.000 | 0.9999 | 0.525 |
| 24 | 25 | 6 | 1.000 | 0.9999 | 0.491 |
| 24 | 25 | 7 | 1.000 | 0.9999 | 0.489 |
| 24 | 25 | 8 | 1.000 | 0.9995 | 0.520 |
| 24 | 25 | 9 | 1.000 | 1.0000 | 0.486 |
| 24 | 25 | 10 | 1.000 | 0.9996 | 0.503 |
| 24 | 25 | 11 | 1.000 | 0.9997 | 0.503 |

**8/8**, not R0's 3/8 (`indist_min=1.000`, `AER/K≥0.9995` in every
one). Joined to the original seeds 0–3, the full n=12 at K=24 reads
**12/12 CONVERGED**, strengthening (not merely reproducing) R0's
claim. §5's fixed-denominator rule (every seed counted, no selection)
is honored here by actually including every available seed, not
merely asserting the practice while sampling 3 of 8.

**KW1.7 / D6 — the 320K-vs-80K far-depth comparison is a confounded
instrument, disclosed (does not touch the primary `indist_min`
leg).** R0's §2(e) attributed the 320K "recovers 1.0 at ALL far
depths" anchor vs. the 80K far-depth metric's 33% figure entirely to
budget. The two come from different harnesses over different residue
sets: the 320K anchor (`ncr_ortho_write.py --arm free`) tests
`REALISTIC_DEPTHS=(5,12,20,29,40,61)`, whose effective hops at K=24
are `[5,12,20,5,16,13]` — 5 distinct residues, max 20, and **h=29
collides with h=5** (both ≡5 mod 24, re-verified this revision:
`29 % 24 = 5 = 5 % 24`). The 80K figure is `ncr_earlyln_scale.py`'s
Gate-2 `sweep_min_rec`, the MINIMUM over every K-residue at
`h_star=8K-3=189` — a strictly harder, whole-sweep metric over a
different residue set. Part of the 320K-clean/80K-not-clean gap is
therefore the LADDER, not the budget alone; this design does not lean
on that comparison beyond what it can support — it bears on the
SECONDARY far-depth leg only (already reported-not-gating, unaffected)
and NOT on the PRIMARY `indist_min` leg, which is harness-common and
where KW1.1's budget-responsiveness finding (above) is the operative
one.

**Reading the bracket, restated budget-conditionally (D1).** At 80K:
K=16 and K=24 are CONVERGED-ROBUST (`indist_min=1.000` in
4+4+8=16 sampled seeds combined — corrected from R0's 4+4+3, KW1.6).
K=32 is 0/4 CONVERGED (`indist_min∈[0.464,0.871]`). At 160K: K=32
rises to 1/4. At 320K: K=32 rises to 2/4 — still short of
CONVERGED-ROBUST at every budget tested. The rank leg (`AER/K`) clears
its own 0.9 bar in every K=32 seed at every budget (0.928–0.966 at 1×;
similarly at 2×/4×) even as the recovery leg lags — the
leg-dissociation A4.4 first identified holds at every budget tested,
not only at 80K.

**The honest power caveat, on a DIFFERENT metric (unchanged from R0,
still correctly scoped).** `EXPERIMENT_LOG.md:8663-8677`'s n=4→n=12
extension found the FAR-DEPTH metric noisy (33% success on the looser
metric, 0/12 on the strict one) — measured on `front`/`sweep_min_rec`
(Gate 2), NOT on `indist_min` (Gate 1), which reads 1.000 in all 12
re-pulled seeds (corrected above) with zero disagreement. §5 does not
borrow this caveat for the primary leg it was never measured on.

---

## §4 CELL GRID + PRICING

**Grid: K∈{26,28,30} × seed∈{0,1,2,3} = 12 cells, single arm (free
only), Part A only (single-relation; no discriminator/bank cells) —
unchanged from R0.** **Rev 1 adds (D1): a CONDITIONAL 4-cell 160K
disambiguator arm**, pre-registered here, launched only if triggered.

**Command (primary, unchanged CLI surface from R0):**

```
ncr_earlyln_scale.py --cell --K {26,28,30} --d-override {27,29,31} \
  --seed {0,1,2,3} --steps 80000 --ceiling-gpuh 0.75 \
  --outdir results_kwall_characterization \
  --stop-file results_kwall_characterization/STOP
```

(`--ceiling-gpuh` lowered from R0's 1.25 to 0.75 — see the unified cap
accounting below; a deliberate, disclosed trade to make room for the
conditional arm inside the mandate's shared 15h cap. Nominal cost is
unaffected.)

**Trigger rule for the conditional 160K arm (D1, exact).** Let
`K_trig` = the smallest K in the ordered list `(26, 28, 30, 32)` whose
primary 80K rate is NOT CONVERGED-ROBUST (`rate<3/4`), evaluated only
over K's with 4/4 `status=="COMPLETED"` primary cells (K=32's "rate"
is the ALREADY-ARCHIVED 0/4 value from
`experiment-runs/2026-07-12_ncr_mappinglaw_wave1/dratio_K32_d33/`,
fixed, not re-measured).
- If `K_trig ∈ {26,28,30}`: launch the 4-cell 160K arm AT `K_trig`.
- If `K_trig == 32`: **no new cells are launched.** The K=32
  disambiguation is already on record (§3's budget table:
  0/4→1/4→2/4 across 1×/2×/4×) and is cited directly, at $0
  incremental GPU-h. This is the FRONTIER-AT-K\*=30 case (§5) — the
  best possible primary outcome — and it costs nothing extra to
  disambiguate because the archive already did the work.
- Precondition (D5): `K_trig` is read only from a K with 4/4 COMPLETED
  cells. An `INCOMPLETE-AT-K` rung (§5) defers the trigger decision
  until it resolves.

**Command (conditional, if triggered):**

```
ncr_earlyln_scale.py --cell --K {K_trig} --d-override {K_trig+1} \
  --seed {0,1,2,3} --steps 160000 --ceiling-gpuh 1.50 \
  --outdir results_kwall_characterization_160k \
  --stop-file results_kwall_characterization_160k/STOP
```

**Pricing — recomputed from raw JSONs this revision, not
FLOP-extrapolated from a different config family.**

Closed-form (`F(K,d,h)=76Kh²+4dh²+12K²h+4Kdh+4d²h`, `h=64`, free arm,
no `NS(d)` term — same formula as R0, re-verified by direct execution
this revision):

| K | d | F(K,d,64) | F / F(24,25,64) |
|---|---|---|---|
| 24 | 25 | 8,636,672 | 1.000 |
| 26 | 27 | 9,421,568 | **1.091** |
| 28 | 29 | 10,216,704 | **1.183** |
| 30 | 31 | 11,022,080 | **1.276** |
| 32 | 33 | 11,837,696 | 1.371 |

(**Correction, KW2.4:** the spread is **1.09×–1.28×** — "~9–28%" —
not R0's "1.17×–1.28×"/"~15–28%"; R0's error was in the conservative
direction, no budget risk, corrected here.)

**80K nominal per cell, K∈{26,28,30}** — the FLOP ratio applied to the
K=24 measured mean (`0.4680 GPU-h`, mean of the 4 raw `gpu_h` values
in `experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/earlyln_K24_s{0-3}.json`,
recomputed this revision), **not R0's flat 0.50h (KW2.5 — the spread
was computed and then not applied):**

| K | 80K nominal (h/cell) | ×4 seeds |
|---|---|---|
| 26 | 0.5105 | 2.042 |
| 28 | 0.5536 | 2.214 |
| 30 | 0.5973 | 2.389 |
| **12-cell total** | | **≈6.65 GPU-h** |

**160K nominal per cell (conditional arm, worst-case K=30).** The
K=32 budget wave's own 2×/1× empirical ratio
(`1.0510/0.5688 = 1.848×`, recomputed this revision from
`experiment-runs/2026-07-12_ncr_k32_budget/budget2x_*.json` vs.
`.../dratio_K32_d33/*.json` mean `gpu_h`) applied to the 80K nominal
above:

| K | 160K nominal (h/cell) | ×4 seeds |
|---|---|---|
| 26 | 0.9434 | 3.774 |
| 28 | 1.0230 | 4.092 |
| 30 | 1.1037 | 4.415 |

**Unified ceiling accounting (D1 — "total stays ≤15 cap"):**

| | per-cell ceiling | ×N cells | nominal (worst-K) |
|---|---|---|---|
| Primary (12 cells, 80K) | 0.75 h | **9.00 h** | ≈6.65 h |
| Conditional (4 cells, 160K, if triggered) | 1.50 h | **6.00 h** | ≈4.42 h (K=30) |
| **Combined worst case (unconditional sum)** | | **15.00 h** | ≈11.06 h (K=30 trigger) / ≈6.65 h (no trigger) |

`12×0.75 + 4×1.50 = 15.00` exactly bounds the pessimistic case of all
16 possible cells (12 primary + 4 conditional) simultaneously hitting
their training-phase ceiling, regardless of which K triggers — no
mutual-exclusivity argument is needed to hold this bound. **This is a
real, disclosed trade against R0's per-cell margin:** R0's 1.25h/cell
(≈2.5× nominal) was affordable only as the sole arm; sharing the
mandate's 15h cap with the conditional arm requires trimming to 0.75h
(≈1.26–1.47× the corrected per-K nominal) and 1.50h (≈1.36× the
worst-case 160K nominal). Both margins stay ABOVE every empirically
observed max/nominal ratio in the archive to date — the largest ever
seen is 1.206× (K32, 2×-budget, seed 3: `1.2685/1.0510`); every
1×-budget cell ever run has stayed within 1.06× of its own K's mean.
The margin is tighter than R0's but still exceeds the worst real
variance on record.

**D5 — eval-inclusive ceiling handling (KW2.2, corrected not merely
disclosed).** `gpu_h` (used throughout this pricing) is
`elapsed_s/3600` measured end-to-end (`ncr_earlyln_scale.py:303-304`)
— i.e. `gpu_h` already INCLUDES eval; every pricing figure above is
eval-inclusive. What is NOT eval-inclusive is the runtime `ceiling_s`
ENFORCEMENT itself (`train_earlyln_cell`, `:198-201`) — it checks only
during training, so a cell whose training finishes right at the
ceiling can still add eval time afterward, unbounded by the check.
Measured directly this revision across every archived cell (K=16/24/32,
1×/2×): eval overhead is **0.7%–1.5% of total elapsed**, max observed
**45.5s = 0.0126 GPU-h** (K32, 2×, seed 3). Worst-case eval-inclusive
total: `15.00h + 16 cells × 0.0126h ≈ 15.20 GPU-h` — disclosed
explicitly as the true pessimistic bound, not silently rounded back to
15.0.

**D5 — ABORTED-BUDGET / MISSING / non-COMPLETED cell rule
(KW2.2/KW2.3).**
- A cell with `status=="ABORTED-BUDGET"` (`train_earlyln_cell`,
  `:198-201`) is retried ONCE automatically with no ceiling change
  (the standard resume-safe supervisor loop skips only
  `status=="COMPLETED"`, `:243-245` — an ABORTED-BUDGET file is NOT
  skipped and would otherwise silently re-run and re-abort
  indefinitely). If it aborts a SECOND time, that seed is flagged
  `PERSISTENTLY-ABORTED` and excluded from its K's rate WITH
  mandatory disclosure (never silently folded into `n_seeds`, unlike
  `discover_seeds_by_K`'s raw file-glob behavior, `:351-371`, which
  would otherwise count it against the K).
- A K with fewer than 4/4 `status=="COMPLETED"` cells (any mix of
  MISSING, ABORTED-BUDGET, PERSISTENTLY-ABORTED) is `INCOMPLETE-AT-K`
  — not classified into any §5 band, not folded into `n_seeds` for
  rate purposes, and not used to evaluate the conditional-arm trigger
  (above) until it resolves.

**Job-spec template (KW2.6 — pool-conformance artifact, specified
here for the build stage; this design remains DRAFT and creates no job
JSONs itself).** Every cell, primary and conditional, gets a
`queue/jobs/pending/*.json` entry in job-108's own 8-field format
(`id, lane, hypothesis, cmd, gpu_h_estimate, output_dir,
validity_check, notes`) with an ABSOLUTE interpreter/working-directory
`cmd` (not the CWD-relative `--outdir` R0 left implicit) and a
`validity_check` that asserts, in addition to job-108's
`status=='COMPLETED'`/`train.step==<budget>`/`'eval' in d`/
`blank_out.passed is True`: **`d == K+1` and `d_override == K+1`** —
closing the exact `d=K+1`-vs-`d=2K` filename-collision risk
`EXPERIMENT_LOG.md:8452` already forced a workaround for once (the
fresh `results_kwall_characterization*` outdirs already avoid the
collision; this `validity_check` addition is defense-in-depth against
a mis-flagged cell silently harvesting under the wrong convention).

---

## §5 PRE-REGISTERED OUTCOME BANDS

**Metric definitions — CONVERGED gate stated as the runner's own
conjunction (D4; R0's "reported not gating" framing for the rank leg
is WITHDRAWN — it was false against the code, KW2.1):**
- Per-seed `indist_min = min(recovered_frac@0.9` at `h∈{1,2,3})`
  (`:319`) and `aer_mean = mean(A_eff_rank)` (`:321`), both from
  `_cell_gate1` (`ncr_earlyln_scale.py:317-329`).
- Per-seed label: **CONVERGED** iff `indist_min ≥ 0.9`
  (`CONVERGED_INDIST_BAR`, `:95`) **AND** `aer_mean ≥ 0.9·K`
  (`AEFF_RANK_FRAC_BAR`, `:97`) — the code's own conjunction (`:322`),
  not `indist_min` alone. **PARTIAL** (`indist_min∈[0.5,0.9)`, gate
  fails) / **DEAD** (`indist_min<0.5`).
- Per-K `rate = (#seeds CONVERGED)/4`, read ONLY on a K with 4/4
  `status=="COMPLETED"` cells (§4 D5); anything else is
  `INCOMPLETE-AT-K` (re-run, not classified — KW2.3's discharge).
- **Rank leg is a GATE COMPONENT, not a separate "reported only"
  signal (D4 correction of KW2.1):** every CONVERGED count above
  already reflects both legs. A seed failing ONLY the rank leg
  (`indist_min≥0.9` but `aer_mean<0.9K`) is scored PARTIAL/DEAD by the
  SAME rule as a pure recovery failure — this design does not attempt
  to separate the two (R0's claim that it did was false); flagged as a
  residual open question this design does not resolve, though `AER/K`
  has stayed comfortably above 0.9 throughout K≤32 in this program's
  own record so far (§3).

**Guard against A4.9's defect (unchanged, still holds):** the per-K
label is a **rate over the full fixed n=4**, never a median over a
gate-passing subset — every seed counts in the denominator regardless
of outcome.

**K\* domain extended to {24,26,28,30} (D2, discharging KW1.4's
`(0,0,0)` gap).** K=24's rate is fixed at its already-measured archive
value (n=12, 12/12 CONVERGED, `indist_min=1.000` uniformly — a
stronger anchor than a fresh n=4 read, §3). K=32's rate is fixed at
its already-measured 80K-budget archive value (0/4, §3) for the
PRIMARY (80K) classification below; the CONDITIONAL arm (§4) may
revise the picture at whichever K it targets, reported as a separate
budget-verdict qualifier (below), never folded back into the PRIMARY
label itself.

**The DEMONSTRATED partition (D2 — replaces R0's three
non-exhaustive, non-mutually-exclusive bands; KW1.3/KW1.4 fully
discharged).** Let `r26, r28, r30 ∈ {0,1,2,3,4}` be the CONVERGED
seed-counts at K=26/28/30 (80K, primary), and fix `r24=4` (ROBUST,
archive) and `r32=0` (NOT ROBUST, archive, §3). Define
`ROBUST(r) := r≥3`. Classification is the following **total, ordered
decision procedure** (first matching rule fires — a partition by
construction; no case unhandled, none matches twice):

1. If `ROBUST(r30)` and `r32≤1`: **FRONTIER-AT-K\*=30**.
2. Else if `ROBUST(r28)` and `r30≤1`: **FRONTIER-AT-K\*=28**.
3. Else if `ROBUST(r26)` and `r28≤1`: **FRONTIER-AT-K\*=26**.
4. Else if `ROBUST(r24)` [always true] and `r26≤1`: **FRONTIER-AT-K\*=24**.
5. Else if `r26 ≥ r28 ≥ r30` (monotone non-increasing): **GRADUAL-DECAY**.
6. Else: **NON-MONOTONE-UNRESOLVED**.

Each fired rule additionally checks whether the boolean ROBUST-sequence
`[ROBUST(r24), ROBUST(r26), ROBUST(r28), ROBUST(r30), ROBUST(r32)]`
is itself monotone (`True…True False…False`); if not, the band
carries a `[NON-MONOTONE]` tag, disclosing an internal dip/recovery
rather than hiding it inside a single K\* (discharging KW1.3(iii)'s
"band (a) can name two walls at once").

**Renamed from R0's "WALL-AT-K\*" to "FRONTIER-AT-K\*" (D1):** the
label reports where the 80K CONVERGED-ROBUST frontier currently sits,
not a claim that the drop below it is permanent — that claim is only
licensed after the CONDITIONAL arm's disambiguation (below), or, at
`K\*=30`, by the ALREADY-ARCHIVED K=32 budget table (§3), which shows
the frontier moving but never clearing ROBUST even at 4×.

**Rule (1) subsumes R0's separate "NO-WALL-BELOW-32" band (discharges
KW1.3(i)):** since `r32=0≤1` always, `ROBUST(r30)` alone triggers rule
(1) — "no wall below 32" is now reported as `FRONTIER-AT-K*=30`,
exactly the reading D2 specifies ("make 'no wall below 32' the K\*=30
sub-case").

**Exhaustiveness and mutual exclusivity — DEMONSTRATED, not asserted
(D2).** The decision procedure was executed against all `5³=125`
reachable `(r26,r28,r30)` outcomes this revision (a total function
with a single `return` per branch is a partition by construction;
verified by direct enumeration, not assumed). Counts:

| Band | Count |
|---|---|
| FRONTIER-AT-K\*=24 | 18 |
| FRONTIER-AT-K\*=24 [NON-MONOTONE] | 4 |
| FRONTIER-AT-K\*=26 | 12 |
| FRONTIER-AT-K\*=28 | 8 |
| FRONTIER-AT-K\*=28 [NON-MONOTONE] | 12 |
| FRONTIER-AT-K\*=30 | 8 |
| FRONTIER-AT-K\*=30 [NON-MONOTONE] | 42 |
| GRADUAL-DECAY | 15 |
| NON-MONOTONE-UNRESOLVED | 6 |
| **Total** | **125 / 125** |

Representative rows (covering every band, including the exact cases
the audit named):

| `(r26,r28,r30)` | Band |
|---|---|
| `(0,0,0)` | FRONTIER-AT-K\*=24 (KW1.4's named gap — now covered) |
| `(0,0,1)` | FRONTIER-AT-K\*=24 |
| `(4,3,0)` | FRONTIER-AT-K\*=28 |
| `(0,4,0)` | FRONTIER-AT-K\*=28 [NON-MONOTONE] |
| `(4,4,2)` | GRADUAL-DECAY (KW1.3(ii)'s ambiguous case — now ONE reading: 2/4 is not ≤1/4, so no cliff fires) |
| `(2,2,2)` | GRADUAL-DECAY (flat plateau — no longer fires nothing, KW1.3(iv)) |
| `(4,0,4)` | FRONTIER-AT-K\*=30 [NON-MONOTONE] (KW1.3(iii)'s "two walls at once" case — now one K\*, flagged) |
| `(4,4,4)` | FRONTIER-AT-K\*=30 |
| `(2,4,2)` | NON-MONOTONE-UNRESOLVED |
| `(3,2,1)` | GRADUAL-DECAY |

Any auditor can re-run the six-rule procedure above against all 125
outcomes to re-check this table; it is a ~15-line function, not a
large artifact, and is reproduced in full above (not merely
referenced) so Round 2 can re-check it without re-deriving it.

**The CONDITIONAL 160K disambiguator's report (D1).** If the trigger
(§4) fires at `K_trig∈{26,28,30}`, its 4-cell 160K rate is reported
ALONGSIDE the PRIMARY 80K classification above as a budget-verdict
qualifier at `K_trig`, never substituted into the 80K label itself:
- **CONFIRMED-WALL-AT-160K:** `K_trig`'s rate stays `≤1/4` at 160K (no
  material improvement) — the strongest evidence this design can
  produce that the drop is architectural, not merely slow.
- **SLOW-CONVERGENCE-AT-160K:** `K_trig`'s rate reaches `≥3/4`
  (CONVERGED-ROBUST) at 160K — the frontier moves with budget;
  recommend the flagship's "last live rung" consider `K_trig` a
  BUDGET-CONDITIONAL candidate (never unconditional) and flag a 320K
  confirmation as future PI-gated work (§7 non-goal, unchanged).
- **PARTIAL-IMPROVEMENT-AT-160K:** `K_trig`'s rate improves but lands
  at `2/4` (not `≤1/4`, not `≥3/4`) — mirrors K=32's OWN archived
  pattern exactly (0/4→1/4→2/4, never reaching ROBUST at any budget
  tested, §3); reported as genuinely ambiguous, motivating (not
  resolving into) a 320K follow-on, exactly as the K=32 precedent
  itself was left unresolved (`EXPERIMENT_LOG.md:8887-8896`, "Not
  established: whether an even larger budget... would behave
  differently").

At `K_trig=32` (i.e. `FRONTIER-AT-K\*=30`, no sub-ROBUST rung inside
{26,28,30}): the qualifier is read directly off the ALREADY-ARCHIVED
table (§3) — K=32 is **PARTIAL-IMPROVEMENT-AT-160K/320K**
(0/4→1/4→2/4, matching the middle case above) — reported at $0
incremental cost, per §4.

**INCOMPLETE-AT-K (D5, replaces R0's under-specified silence on
MISSING/non-COMPLETED cells, KW2.2/KW2.3).** Any K with fewer than 4/4
`status=="COMPLETED"` cells is not classified by the procedure above;
it is re-run (with the retry-once/flag rule, §4) until 4/4 COMPLETED,
then classified. No partial rate is ever read into the table.

**Every band is informative and reportable** — none requires a
follow-on wave to be publishable as a characterization result; every
FRONTIER-AT-K\* outcome localizes the 80K frontier to a resolution
better than the current archive; GRADUAL-DECAY produces the K-ladder's
own missing scaling-curve data point; NON-MONOTONE-UNRESOLVED is
itself a legitimate, publishable pre-registered outcome (an
unclassifiable result was the defect this design no longer has).

---

## §6 POOL-ELIGIBILITY STATEMENT

This grid satisfies `matrix-thinking/queue/idle_fallback_daemon.sh`'s
own pool contract (header, `:10-16`): *"the pool holds ONLY flat
specs — each fully audited + queue-eligible, independently runnable in
any order, carrying its own cost ceiling, with NO intra-wave
dependencies, stage gates, or staged-escalation semantics."* Verified
against this design's own structure, not asserted:
- **Independent, with one honest exception (Rev 1 disclosure).** No
  primary cell's launch condition depends on another cell's result —
  all 12 primary cells run in any order or fully in parallel, exactly
  as R0 stated (unlike the SPENT K-ladder design,
  `NCR_KLADDER_ATTACK_R2.md` finding A4.12). **The CONDITIONAL 4-cell
  160K arm (§4) is NOT flat-independent of the primary** — its launch
  (and which K it targets) depends on the primary 12-cell harvest.
  This is a genuine, disclosed, single-stage dependency, reintroduced
  at far smaller scale than the SPENT ladder's multi-stage design it
  was praised for avoiding. Resolution: the PRIMARY 12-cell grid is
  the flat, pool-eligible spec (unchanged from R0); the CONDITIONAL
  arm is a separate, pre-registered FOLLOW-UP spec, generated only
  after the primary's harvest determines `K_trig` (§4) — it is not
  submitted to the pool alongside the primary as a second flat item,
  and its own job spec (§4's template) is produced at that later point,
  not now.
- **Own cost ceiling:** the primary 12 cells carry `--ceiling-gpuh
  0.75` (§4, lowered from R0's 1.25 — see §4's unified cap
  accounting); the conditional 4 cells, if triggered, carry
  `--ceiling-gpuh 1.50`. Both enforced by the runner's own existing
  ceiling mechanism (`train_earlyln_cell`'s `ceiling_s` argument,
  `:198-201`) — training-phase only; §4 D5 discloses the small
  eval-inclusive overshoot (≈15.20h true worst case, not 15.00h).
- **Audited + queue-eligible only after this draft clears its own
  audit round** — still explicitly NOT queue-eligible (status header,
  now DRAFT-R1); the pool contract's "ceremony gate stays upstream of
  it" applies here exactly as written.
- **Queue-pool sweep scope, corrected (KW2.7).** §3's internal sweep
  covered `matrix-thinking/queue/jobs/pending/` (the only queue
  directory tracked in this repo) and found zero K∈{26,28,30} hits;
  `~/queue/{fallback_pool,claimed}` on the box were NOT swept this
  session. **Added as a mandatory pre-launch red-team task:** sweep
  both on-box directories for K∈{26,28,30} content before this design
  (or its conditional follow-up) is promoted to the pool.
- **No standing restriction bites.** The `STATE.md:39-40` "NO NCR job
  queue-eligible" restriction (2026-07-30) is scoped to the in-LM
  write-conditioning claim pivot; this design makes no in-LM claim and
  no claim pivot — it characterizes an already-cleared toy-scale
  mechanism (S11 earlyln free-write; NCR core mechanism NOVEL per
  `research/novelty-gate-2026-07-27.md`) at new K values, the same kind
  of additive K-extension the 2026-07-11 queue-system build already
  did without a fresh novelty gate. **Per KW2.10's discharge condition:
  this reading is not this design's ruling to make** — it needs a
  coordinator record in STATE.md/EXPERIMENT_LOG at adjudication time,
  not self-clearance inside this file. `STATE.md`'s 2026-08-06 tick
  already records the coordinator routing this document
  "DRAFT-R0 → audit → adjudication → build → pool," consistent with
  this reading; the formal ruling itself stays outside this file, per
  the audit's own instruction.

---

## §7 NON-GOALS

- **No K≥32 cell.** K=32 is CLOSED-AT-THIS-K per the mapping-law
  harvest (`EXPERIMENT_LOG.md:8632-8637`), and the 07-13 budget-rescue
  wave's own "closes the K-axis book" ruling bars further UPWARD
  escalation (§3, D3) — this design does not re-run K=32, does not
  extend past it, and does not touch the K=48 WAVE-1b block
  (`:8638-8640`). **Rev 1 (D1):** the CONDITIONAL disambiguator's
  `K_trig=32` case (§4/§5) reuses the ALREADY-ARCHIVED
  `experiment-runs/2026-07-12_ncr_k32_budget/` table at $0 incremental
  GPU-h — citing existing data is not "a K≥32 cell" under this
  non-goal; no new K=32 launch is licensed or performed by this
  design at any revision.
- **No NS-polar / orthogonal-write machinery anywhere.** Free arm
  only; `ncr_ortho_write.py`'s `ortho` arm, `newton_schulz_polar`, and
  `NS_ITER_DEFAULT`/`NS_POWER_DEFAULT` are not invoked, not tuned, not
  discussed as a candidate fix. That mechanism's verdict of record is
  FAIL (`NCR_ORTHO_WRITE.md` §9) and stays FAIL.
- **No `h(K)` relitigating.** `h=64` fixed, reused unmodified — §2(d)
  shows it is not the binding constraint anywhere in this range; no
  `GRID_SHAPES["h"]` edit is proposed or needed.
- **No new far-depth residue arithmetic.** The Gate-2/secondary ladder
  for K∈{26,28,30} comes from the already-audited, regression-tested
  `_gen_grid(K)` formula (§2), extended by the identical additive
  pattern already used for every other post-2026-07-11 K. No K+8 /
  2K-1 coprime-probe scheme (`NCR_KLADDER_DESIGN.md` §3) is imported —
  that scheme belongs to the SPENT ortho-write ladder's own harness and
  is not needed here.
- **No discriminator/bank (Part B) cells.** Single-relation only.
- **No budget-instrument unification claim — corrected and
  strengthened (D1, KW1.1).** R0 claimed 80K and 320K agree on the
  PRIMARY (`indist_min`) leg, evidenced only at K=24 (a saturated
  point with zero power to detect disagreement). Rev 1 corrects this:
  the K=32 budget table (§3) shows the primary leg does NOT agree
  across budgets in the unsaturated regime (0/4→1/4→2/4 CONVERGED
  across 1×/2×/4×) — budget materially changes the PRIMARY verdict,
  which is exactly why the CONDITIONAL 160K arm (§4/§5) exists. This
  design does not claim budget-instrument unification in any
  direction; it pre-registers the CONDITIONAL arm as the mechanism for
  handling the disagreement at whichever K needs it, within the same
  ≤15 GPU-h cap (§4). A 320K confirmation beyond the conditional 160K
  arm, if the disambiguator itself reads PARTIAL-IMPROVEMENT-AT-160K
  (§5), remains future PI-gated work, not part of this design's cap.
- **No mapping-law / WAVE-1b relitigating.** The K=32 `d(K)`-grid
  CLOSED verdict and the K=48 BLOCKED staging rule both stand as
  written; nothing here reopens either.
- **No claim of a flagship-level capability result.** This is a
  trainability-characterization filler wave, not a capability-
  separation or scaling-law submission; §5's outcomes feed the
  flagship's "last live rung" bookkeeping at most, never its headline.

---

*Draft-R0, 2026-08-06. Written from direct reads of
`NCR_KLADDER_DESIGN.md`, `NCR_KLADDER_ATTACK_R2.md`, `STATE.md`,
`EXPERIMENT_LOG.md`, `NCR_ORTHO_WRITE.md`, `matrix-thinking/ncr/ncr_earlyln_scale.py`,
`matrix-thinking/ncr/ncr_task.py`, `matrix-thinking/ncr/ncr_ortho_write.py`,
`matrix-thinking/queue/jobs/pending/108_laneA_main_K48_s0.json`,
`matrix-thinking/queue/idle_fallback_daemon.sh`, and the raw per-seed
JSONs in `experiment-runs/2026-07-12_ncr_nextlever_wave/` and
`experiment-runs/2026-07-12_ncr_mappinglaw_wave1/` (fields read via
`json.load` this session, not transcribed from prose). No repo file
other than this one was created or modified; no command was run on
the box; no job was launched; no git mutation was made.*

## §A1-ADJUDICATION — AUDIT ROUND 1 VERDICT ADOPTED: **REV-REQUIRED** (Fable, 2026-08-06)

Audit `NCR_KWALL_ATTACK_R1.md` (judge tier, frame-attack charter):
REV-REQUIRED, 4 FATAL / 7 MAJOR / 7 MINOR. Coordinator verified the
decisive findings against raws before adopting: KW1.1 CONFIRMED —
`experiment-runs/2026-07-12_ncr_k32_budget/` exists and the 07-13 log
entry records K=32 CONVERGED improving 1/4 (2×) → 2/4 (4×) with
budget, so this draft's fixed "K=32 = 0/4 DEAD" band anchor is an
80K-budget artifact (CLAUDE.md's recorded plateau-vs-slow-convergence
failure mode, verbatim); KW1.8's uncited ruling located and read in
full. The draft's verified-clean core (mod-K crash finding, grid
constructors, FLOP arithmetic, K∈{25..31} openness — independently
re-swept by the audit incl. SSD superset and git -S) carries forward.

**BINDING DISPOSITIONS for Rev 1 (audit's recommended shape ADOPTED):**
- **D1 (KW1.1/KW1.2):** keep all 12 cells; RE-REGISTER the headline as
  the **80K-budget convergence frontier** over K∈{26,28,30} (a budget-
  conditioned claim, never "the wall"); add the pre-registered
  CONDITIONAL 4-cell 160K arm at the first sub-ROBUST rung as the
  speed-vs-wall disambiguator (~1.2–1.5 GPU-h, total stays ≤15 cap);
  redo the §3 sweep keyed on config axes (budget included), not K
  strings.
- **D2 (KW1.3/KW1.4):** rebuild §5 as a DEMONSTRATED partition — K*
  domain extended to {24,26,28,30}, every one of the 125 rate outcomes
  classified exactly once, the (0,0,0) wall-at-24→26 case included;
  the demonstration table goes IN the design (audit re-checks it).
- **D3 (KW1.8):** the 07-13 "closes the K-axis book" ruling is
  adjudicated NARROW per its own scope paragraph ("Closed: whether
  budget alone rescues K=32's tight-spare wall into anything licensing
  further K-escalation — no"): it bars upward escalation (K≥48 stays
  BLOCKED; parked cells stay parked), it does NOT bar the below-32
  characterization this design performs, which the §A4-ADJUDICATION
  mandate + two judge-tier rounds name as genuinely open. Rev 1 cites
  this disposition where it cites the mandate.
- **D4 (KW2.1):** do not fork instrument semantics — the runner's
  CONVERGED conjunction (recovery AND rank) stays as-is; the design
  must STATE the gate accurately (rank leg is a gate component, not
  "reported not gating") and disclose it in every band definition.
- **D5 (KW2.2/KW2.3):** eval-inclusive ceiling handling; ABORTED-BUDGET
  cells excluded from band denominators WITH mandatory disclosure and
  resume-skip (no infinite re-abort); explicit MISSING/non-COMPLETED
  rule before `harvest()` folds anything.
- **D6 (KW1.5–KW1.7, KW2.x MINORs):** address each per the audit's
  discharge conditions; citation corrections exactly as found (the
  KW1.5/KW1.6 conclusions survive on the corrected evidence — say so
  plainly, no silent swap).

Rev 1 → fresh audit round (same two-part charter) → adjudication →
only then build/audit → placement red-team (10–50 tier) → pool.

---

## §R1 REVISION 1 (2026-08-06)

Rev 1, dispatched per §A1-ADJUDICATION's binding dispositions D1–D6.
Every KW-finding addressed below with an exact section reference; §1–§8
[sic, §1–§7] and §A1-ADJUDICATION themselves are UNCHANGED as
historical record EXCEPT where a disposition explicitly required
rewriting a section's content in place — those in-place rewrites are
noted here and marked inline (§1, §2(a), §3, §4, §5, §6, §7 all carry
"Rev 1"/"D#"/"KW#.#" markers at the exact rewritten passages). This
design now carries status **DRAFT-R1 — POST-AUDIT-1, AWAITING AUDIT
ROUND 2**.

| Finding | Disposition | Where fixed |
|---|---|---|
| **KW1.1 (FATAL)** — 80K budget premise refuted by an unswept archive wave; K=32's own rate moves 0/4→1/4→2/4 CONVERGED with budget | Headline re-registered as "the 80K-budget convergence frontier," never "the wall"; K=32's budget table re-derived directly from all 12 raw JSONs this revision (independently reproduced to the digit); pre-registered CONDITIONAL 4-cell 160K disambiguator added, with an exact trigger rule and a unified ≤15 GPU-h cap covering both arms | §1, §3, §4, §5 |
| **KW1.2 (FATAL)** — archive sweep scoped to the K-string axis only, missed the budget axis the claim actually turns on | Sweep axes stated explicitly (K-string, budget, prior `d=K+1` measurements at any budget, prior rulings, on-box pool — with the pool sweep's own incompleteness disclosed, KW2.7); the three missed 2026-07-12 waves + `NOVEL_ARCH_WATERFALL.md` §11.4/§11.6 cited and reconciled | §3 |
| **KW1.3 (FATAL)** — three bands are not a partition (25/125 unclassifiable, band (c)⊂band (a), band (b) self-contradictory on 12/125, multiple K\* possible) | Rebuilt as a 6-rule ordered decision procedure over `(r26,r28,r30)`; demonstrated exhaustive AND mutually exclusive over all 125 outcomes by direct enumeration (counts + representative rows given, reproducible from the ~15-line rule set printed in full); "no wall below 32" is now the K\*=30 sub-case (rule 1) — the old band (c) is retired, not duplicated; non-monotone ROBUST-sequences get a disclosed `[NON-MONOTONE]` tag instead of silently picking one of several candidate K\*s | §5 |
| **KW1.4 (FATAL)** — K\*=24 inexpressible; `(0,0,0)` unrepresentable / actively mis-classified as GRADUAL-DECAY | K\* domain extended to {24,26,28,30}; `(0,0,0)` now classifies as FRONTIER-AT-K\*=24 (verified directly in the 125-outcome enumeration, not merely asserted) | §5 |
| **KW1.5 (MAJOR)** — §2(a) misattributed the n=12 Q2 seed extension to `d=2K`; all 12 cells are actually `d=K+1`, contradicting §3's own correct use of the same run set | Misattribution struck; correct `d=2K` rejection evidence substituted (K24/d48 at 80K AND 160K, both 4/4 DEAD by both legs, re-derived from raw JSONs this revision); the K=32 `d(K)`-grid sentence rescoped (it covers `d=K+1` too, so it does not by itself discriminate the two conventions — that discrimination now rests on the corrected K24/d48 evidence) | §2(a) |
| **KW1.6 (MAJOR)** — §3 selectively re-pulled 3 of 8 available n=12 seed-extension cells while §5 asserted "no selection effect to disclose" | All 8 re-pulled and reported (`indist_min=1.000`, `AER/K≥0.9995` in 8/8, re-verified by direct `json.load`); full n=12 at K=24 now stated as 12/12 CONVERGED | §3 |
| **KW1.7 (MAJOR)** — the 320K-vs-80K far-depth comparison confounds budget with a ladder/residue-set difference (h=29≡h=5 mod 24 collision) | Disclosed as a confounded instrument (different harnesses, different residue coverage, mod-arithmetic re-verified: `29 % 24 = 5 % 24`); explicitly scoped to the SECONDARY far-depth leg only — does not touch the primary-leg budget finding (KW1.1), which is harness-common | §3 |
| **KW1.8 (MAJOR)** — an uncited standing ruling ("this CLOSES the K-axis book at K=32") never adjudicated | Cited verbatim (`EXPERIMENT_LOG.md:8845/8885`, `NOVEL_ARCH_WATERFALL.md:5071`) and adjudicated NARROW per §A1-ADJUDICATION D3 (quoted in full); this design's license is stated against that narrow ruling wherever it is invoked | §3, §7 |
| **KW2.1 (MAJOR)** — "rank leg, reported not gating" is false against the code; CONVERGED is a conjunction (`_cell_gate1`) | Corrected throughout: the CONVERGED gate is stated as the recovery-AND-rank conjunction in every band definition (D4); the false "reported not gating" framing is withdrawn, not merely softened | §1, §5 |
| **KW2.2 (MAJOR)** — the ceiling is enforced training-only; an `ABORTED-BUDGET` cell silently deflates a K's rate and would re-abort indefinitely under the standard supervisor loop | Eval-inclusive worst case computed from raw elapsed-time fields and disclosed (≈15.20 GPU-h, not 15.00); explicit retry-once-then-`PERSISTENTLY-ABORTED`-with-disclosure rule added; such cells are excluded from `n_seeds`/rate, never silently folded in | §4, §5 |
| **KW2.3 (MAJOR)** — no band states what a MISSING/non-COMPLETED cell does to the rate | `INCOMPLETE-AT-K` added as a first-class, unclassified, re-run-not-guessed state, used consistently by the band procedure and the conditional-arm trigger | §4, §5 |
| **KW2.4 (MINOR)** — FLOP spread lower bound wrong (stated 1.17×, actual 1.09×) | Corrected to 1.09×–1.28× / "~9–28%", re-verified by direct execution of the pinned formula this revision | §4 |
| **KW2.5 (MINOR)** — the corrected FLOP spread was computed and then not applied to the flat 0.50h/cell pricing | Applied: per-K 80K nominal is now 0.5105/0.5536/0.5973 h (K=26/28/30), not a flat 0.50h | §4 |
| **KW2.6 (MINOR)** — no conforming pool-spec artifact; CWD-relative `--outdir`; no `validity_check` pinned | Job-108-format template specified in full (absolute paths, `d==K+1`/`d_override==K+1` validity-check addition) for the build stage; this design stays DRAFT and creates no job JSONs itself, so the actual specs are correctly deferred, not fabricated here | §4 |
| **KW2.7 (MINOR)** — §3 claimed a queue-pool sweep scope ("pending + fallback_pool + claimed") that does not exist locally | Scope corrected to what was actually searched (`jobs/pending/` only, locally); on-box `fallback_pool/`/`claimed/` sweep added as a mandatory pre-launch red-team task; the K=20 filename-vs-content-grep gap and the two extra loose-search false hits are disclosed | §3, §6 |
| **KW2.8 (MINOR)** — the build's own `--smoke`/t5 self-tests would exercise the new K's at `GRID_SHAPES`' default `d=2K`, not the `d=K+1` this design runs | Not a design-content defect (no §-text asserted otherwise); flagged here as a build-stage instruction: add a `d=K+1` micro-cell smoke (the `t10` pattern) at one new K before build-release | (build-stage instruction only; no section rewrite required) |
| **KW2.9 (MINOR)** — harvest emits `SUB4-DISCLOSED-ONLY(n=0)` rows for 12 unrelated K's | The audit itself classified this "no false verdict" — no design text was asserting otherwise; no change made | (no change — audit's own finding required none) |
| **KW2.10 (MINOR)** — the `STATE.md:39-40` scoping question needs a coordinator record, not design-doc self-clearance | The design's reading is unchanged (already correct per the audit) but is now explicitly marked as NOT this file's ruling to make; `STATE.md`'s existing 2026-08-06 tick (recording the coordinator's own routing of this document) is cited as the standing record, per the audit's own discharge instruction | §6 |

**Numbers that moved as a direct, disclosed consequence of the
KW1.1/D1 fix (not independent changes):** the 12-cell primary's
per-cell ceiling: `1.25h → 0.75h`; the primary's stated 80K nominal:
flat `≈0.50h → 0.5105/0.5536/0.5973h` (K=26/28/30, KW2.5); the FLOP
spread: `1.17×–1.28× → 1.09×–1.28×` (KW2.4); the total worst-case cost
figure: `15.0h (primary only) → 15.0h (primary+conditional, same
number, now covering 16 possible cells not 12) / ≈15.20h
eval-inclusive` (KW2.2); K=32's fixed archive anchor gained a THIRD
column (budget) it did not have in R0 — 0/4 (1×) is now reported
alongside 1/4 (2×) and 2/4 (4×), not alone.

**Not re-litigated in Rev 1 (out of scope for this pass, flagged for
Round 2 to check for completeness):** §2's `d=K+1`-vs-`d=2K` config
choice itself (only its citation was corrected, KW1.5 — the choice
stands); §2(c)'s mod-K crash arithmetic and §2(d)'s `h`-never-binds
argument (both independently verified CLEAN by the audit, untouched);
the Gate-2/secondary far-depth ladder generation (`_gen_grid(K)`,
unchanged); the K=48 WAVE-1b block and the K=32 `d(K)`-grid CLOSED
verdict (both stand as written, §7).

**Disposition not fully implementable inside this file (disclosed,
per the task's own instruction to report it):** KW2.10's discharge
condition explicitly directs the scoping ruling into
STATE.md/EXPERIMENT_LOG "at adjudication time, before pool
insertion — not inside the design document." This revision cites the
existing STATE.md record and states the design's reading, but does
not — and structurally cannot, being scoped to this one file — write
the formal coordinator ruling itself. That step remains outstanding
outside this document.

*Rev 1, 2026-08-06. Written from direct reads of
`NCR_KWALL_ATTACK_R1.md` (966 lines, every FATAL/MAJOR/MINOR and the
verified-clean list), this file's own §A1-ADJUDICATION, and fresh
`json.load` re-derivation of every load-bearing number from
`experiment-runs/2026-07-12_ncr_k32_budget/` (12 cells),
`experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext/` (8
cells) and `q2_K24_seedext_orig0-3/` (4 cells),
`experiment-runs/2026-07-11_ncr_earlyln_scale/` and
`experiment-runs/2026-07-12_ncr_earlyln_budget2x/` (K24/d48 cells, 8
total), `experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/` (K16/
K24/K32 cells), `EXPERIMENT_LOG.md` (lines 8495-8496, 8632-8637,
8663-8677, 8845, 8885-8896), `NOVEL_ARCH_WATERFALL.md` (§11.6,
line 5071), `STATE.md`, and `matrix-thinking/ncr/ncr_earlyln_scale.py`
(lines 198-266, 317-329, 351-406). No repo file other than this one
was created or modified; no command was run on the box; no job was
launched; no git mutation was made.*
