# NCR K-WALL CHARACTERIZATION — K∈{26,28,30} ON THE LIVE K=24 RUNG

**STATUS: DRAFT-R9 — POST-AUDIT-9, AWAITING NARROW AUDIT ROUND 10 (not
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
the archive actually leaves open."* `STATE.md:24-26` (line renumbered
this revision, content unchanged — KW3.7) records this
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
is not.** `STATE.md:128` (line renumbered this revision, content
unchanged — KW3.7) calls out the load-bearing finding in
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
licensed."* Its own scope paragraph, `EXPERIMENT_LOG.md:8887-8888`,
verbatim: *"Closed: whether budget alone rescues K=32's tight-spare
wall into anything licensing further K-escalation — no."*
**Citation correction (Rev 2, KW3.8):** R0/Rev 1 attributed this
sentence to `NOVEL_ARCH_WATERFALL.md:5071` and labelled it "in full" —
both wrong. `NOVEL_ARCH_WATERFALL.md:5071`'s own wording differs
verbatim (*"Closed: whether more compute (budget) rescues K=32's
tight-spare Gate-1 wall into something that licenses further
K-escalation — no"*); the sentence quoted above is
`EXPERIMENT_LOG.md`'s, not waterfall's. What waterfall `:5071` DOES
independently confirm, quoted separately and correctly attributed
here: *"no further budget probe at K=32 is licensed or recommended by
this record."* Both sources agree in substance (D3's ruling is
unaffected — it was never sourced from the misattributed sentence
alone) but the composite quotation was a two-source splice presented
as one paragraph; corrected to two properly attributed quotes.

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
its own 0.9 bar in every K=32 seed at every budget
(**0.9269–0.9679 at 1×**, corrected this revision — KW3.10 found the
prior "0.928–0.966" wrong at both ends, re-derived directly from
`aer_mean/K` = 0.9269/0.9287/0.9440/0.9679 for seeds 0–3; similarly at
2×/4×) even as the recovery leg lags — the
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

**Rev 3 (F1, KW4.1–KW4.4): delivery model changed — ONE self-contained
ORCHESTRATOR job, not 16 independently pool-dispatched cells.**
§A3-ADJUDICATION locates the root cause of all three FATALs in one
place: §A2's E1 prescribed a cumulative budget gate while §6's own
cited pool contract (`idle_fallback_daemon.sh:10-16`) forbids
intra-wave dependencies between pool specs — a cumulative cap across
independently-dispatched cells IS such a dependency, and no revision
inside that frame could close it. Rev 3 changes what the pool sees:
the pool holds exactly ONE flat, independent spec — "run the
orchestrator" — carrying its own single cost ceiling (§6, 15.50h). All
16 possible cells run STRICTLY SEQUENTIALLY, on ONE GPU, INSIDE that
one job, dispatched by the orchestrator itself — never through the
multi-worker `queue_worker.sh`/`idle_fallback_daemon.sh` pool paths
KW4.3 found carry no budget state at all. This is the audit's own
"Option B" (`NCR_KWALL_ATTACK_R3.md` §9, item 1) minus the
reservation-ledger machinery a PARALLEL-dispatch Option B would have
needed — because dispatch is strictly serial, there is no "batch" to
reserve against; the orchestrator's own realized-spend ledger (the
ORCHESTRATOR CONTRACT below) is always exactly the true cumulative
spend at every gate check, by construction of the sequencing itself,
not by a convention that could be violated. This occupies exactly 1 of
the cluster's 8 GPUs; the other 7 remain free for concurrent pool work
— saturation is a fleet-level property, not a claim this one job makes
about itself.

**Per-cell-attempt subprocess shape (issued BY the orchestrator; not a
command a human or the pool runs standalone — the orchestrator itself
is the only pool artifact this design produces, §6).** Primary arm,
one subprocess call per (K, seed, attempt):

```
ncr_earlyln_scale.py --cell --K {26,28,30} --d-override {27,29,31} \
  --seed {0,1,2,3} --steps 80000 --ceiling-gpuh 1.20 \
  --outdir /home/nvidia/ncr/results_kwall_characterization/K{K}_s{seed}_attempt{n} \
  --stop-file /home/nvidia/ncr/results_kwall_characterization/STOP
```

**Absolute paths (KW5.11, Rev 4).** `NCR_ROOT=/home/nvidia/ncr`,
matching job-108's own `cmd`/`output_dir`/`validity_check` convention
verbatim — `queue_worker.sh` runs both `cmd` and `validity_check` from
the WORKER's own CWD (`:157`, `:162`), not the spec's, so a relative
results tree would land wherever the supervisor happened to be started
from. Every path this design specifies (the two results trees, the
smoke outdir, the ledger, the report, the STOP sentinels) is
`${NCR_ROOT}/...`, shown here and in the conditional/smoke command
blocks below; elsewhere in this document `results_kwall_
characterization/` is informal shorthand for the same absolute path,
not a second, relative convention.

**Attempt dirs are ARCHIVE ONLY (F1's KW4.1 overwrite fix, Rev 3;
G2, Rev 4 — closes KW5.2, `harvest()` never reads them).** `{n}∈{1,2}`:
attempt 1's outdir is never reused for attempt 2. `run_earlyln_cell`'s
existing `--outdir` flag (already part of the R0 CLI surface) is the
ONLY mechanism needed. The overwrite KW4.1 found — `:240-266` skips
only on `status=="COMPLETED"`, so a retry's write to the SAME fixed
filename clobbers the first attempt's `elapsed_s` — cannot occur here,
because the two attempts are never given the same directory; a first
attempt's full JSON record survives untouched under `attempt1/` even
after `attempt2/` is written. **What Rev 4 fixes: these
per-attempt directories are a debugging/archival record the
orchestrator itself never reads back — `harvest()` never globs inside
them, recursively or otherwise.** See G2's canonical-path contract
below for what `harvest()` actually reads. (The same attempt-indexed,
archive-only convention applies to the conditional-arm command below.)

**`--ceiling-gpuh` values (E2, Rev 2, unchanged in substance) restored
to the job-108 house convention** (`≥2×nominal, floor 1.0h` — verified
verbatim against `queue/jobs/pending/108_laneA_main_K48_s0.json`'s own
notes field: *"`--ceiling-gpuh` is 2x the estimate (floor 1.0h) as the
real safety bound"*). Per-K minimum under this convention is
`max(2×nominal, 1.0)` = 1.0211/1.1073/1.1946 h for K=26/28/30
(recomputed this revision, §4 pricing below); **1.20h is used as one
shared value across the 3-K command** because it is `≥2×nominal` for
EVERY K in the batch (it exceeds even K=30's 1.1946h minimum), so one
CLI invocation stays valid for all three K's without under-covering
any of them. **Rev 3 (KW4.4): this CLI value — 1.20h primary, 2.32h
conditional (below) — is now ALSO the exact value the orchestrator's
ledger CHARGES at every gate check** (ORCHESTRATOR CONTRACT below);
there is no longer a separate per-K "charged" figure distinct from the
"enforced" one. The per-K `max(2×nominal,1.0)` numbers above remain
only as the INFORMATIONAL justification for why 1.20h clears every
K's own floor — they do not feed the gate.

**Trigger rule for the conditional 160K arm (F2, KW4.5, Rev 3 —
replaces D1's rule, which deadlocked and did not compose with interval
logic).** The trigger is evaluated as an explicit function of the
per-K resolution-state vector `(state_26, state_28, state_30)` — the
SAME per-K states §4's D5/E4 rule (below) computes for band
classification, consumed here by a SEPARATE downstream function,
because `classify()` (§5) and the trigger's K-scan are different rules
over the same triple and — per the 11 cases enumerated below — CAN
disagree even when both individually "decide."

**Per-K resolution state (shared with D5/E4; `n_completed` from
`harvest()`'s status-based count, never file-glob presence):**

| `n_completed(K)` | State | Contribution to the trigger scan |
|---|---|---|
| 4 | `EXACT` | one fixed `r`-value |
| 3, `r_known∈{0,1,3}` | `DECIDED` (interval-resolved) | one fixed `r`-value — `ROBUST(r_known)==ROBUST(r_known+1)` for every value except 2, so the candidate is unambiguous |
| 3, `r_known=2` | `AMBIGUOUS` | two candidate `r`-values `{2,3}` — the ONE point where `ROBUST(r):=r≥3` straddles the boundary |
| ≤2 | `UNRESOLVED` | excluded from candidacy (F2); blocks the scan IF reached (below) |

**Scope note on the `DECIDED` collapse (KW5.13, Rev 4).** "Shared with
D5/E4" describes only the `n_completed` COUNT that produces this
table — the table's `DECIDED` row's "one fixed `r`-value" collapse is
valid **for the trigger's `ROBUST`-only K-scan below, and ONLY for
that scan.** Band classification (D5/E4, later in this section) never
uses this collapse: it evaluates the six-rule `classify()` procedure
at BOTH interval candidates (`r_known` and `r_known+1`) for EVERY
incomplete-cell state — `r_known∈{0,1,2,3}` alike — because a
same-`ROBUST`-ness pair can still land in different bands (rule 4
tests `r≤1` exactly, not `ROBUST(r)`; counterexample: `r_known=1` at
K=26 with `r28=r30=0` gives `classify(1,0,0)=FRONTIER-AT-K*=24` but
`classify(2,0,0)=GRADUAL-DECAY` — different bands from a `DECIDED`
state whose trigger-scan reading is unambiguously "not ROBUST" either
way). A build implementer must read "one fixed `r`-value" as
scoped to the trigger scan only, never as license to skip the
two-candidate evaluation in the D5/E4 band procedure below.

**Trigger decision procedure (pseudocode):**

```
def trigger(states_26_28_30):
    branches = cross_product_of_AMBIGUOUS(states)   # 1, 2, 4, or 8 candidate (r26,r28,r30) triples (2^k for k AMBIGUOUS K's — §R5 KW6.12)
    K_trigs = set()
    for triple in branches:
        # scan K=26,28,30,32 in order; r24=4 (fixed ROBUST), r32=0 (fixed archive)
        kt = smallest_K_with_rate_below_3(triple)
        if kt requires reading an UNRESOLVED K's status to decide:
            # (§R8 K7/KW9.10 — every return below is now the SAME
            # 4-tuple shape, (K_trig, resolution, resolution_detail,
            # diag): `resolution` sat at index 2 in the old DECIDED
            # branches but index 0 in the old TRIGGER-UNRESOLVED
            # branches (a 2-tuple there, a 3-tuple in the G5 copy
            # below); the redundant "DECIDED" wrapper is dropped
            # (resolution alone already says DECIDED vs UNRESOLVED).
            # `diag` is OVERLOADED, not split by copy (§R9 m6 — closes
            # KW10.7's residue: the prior comment claimed a per-copy
            # split that does not hold). This copy's own `diag` is
            # always `blocking_K`. The G5 copy below carries
            # `blocking_K` from ITS OWN line-parallel early return AND
            # `band_blocked_K_trig` from its separate G5-precondition
            # return — never both from one call, but both reachable
            # from that one function. The schema (`:1548-1549`) keeps
            # these as two distinct fields; a consumer must key off
            # the RETURN SITE, never tuple position alone, to know
            # which one `diag` holds.
            return (None, "TRIGGER-UNRESOLVED", None, blocking_K)   # F2: a K that cannot resolve cannot trigger
        K_trigs.add(kt)
    if len(K_trigs) == 1:
        return (K_trigs.pop(), "unanimous", None, None)
    else:
        # (§R7 J2 — closes KW8.2's FATAL) `resolution` is now the bare
        # enum literal "tie-break-min"; the human-readable candidate
        # list moves to its OWN field, `resolution_detail` (schema,
        # below), so universal assertion 6's exact-membership test
        # (`trigger["resolution"] in {"unanimous","tie-break-min",
        # "TRIGGER-UNRESOLVED"}`, unchanged) sees the literal, not a
        # formatted string it can never match.
        return (min(K_trigs), "tie-break-min",
                f"candidates were {sorted(K_trigs)}", None)
```

- If `K_trig ∈ {26,28,30}`: launch the 4-cell 160K arm AT `K_trig`.
- If `K_trig == 32`: **no new cells are launched.** The K=32
  disambiguation is already on record (§3's budget table:
  0/4→1/4→2/4 across 1×/2×/4×) and is cited directly, at $0
  incremental GPU-h. This is the FRONTIER-AT-K\*=30 case (§5) — the
  best possible primary outcome.
- **`TRIGGER-UNRESOLVED`:** no conditional arm launches. This is a
  disclosed, terminating, publishable outcome — NOT a deferral that
  waits for the blocking K to resolve (a `PERSISTENTLY-ABORTED`-caused
  `UNRESOLVED` state is, by E4's own bounded-retry rule, TERMINAL and
  can never resolve — Rev 2's "defers … until it resolves" precondition
  deadlocked against exactly this, KW4.5's first defect). The primary
  80K `classify()` band is still reported if it is itself decidable
  (§5) — only the 160K disambiguation is unavailable.
- **Tie-break rationale (F2, in-text, attackable):** the SMALLEST
  disagreeing candidate K runs — nearest the live K=24 rung, and the
  most informative single choice for the frontier claim §1 makes (a
  wall confirmed at the smaller K is evidence the larger K's own
  ambiguity does not need to independently re-establish).

**Trigger precondition — the conditional arm requires a DECIDED band,
not merely a DECIDED `K_trig` (G5, Rev 4 — closes KW5.5).** The K-scan
above and the §5 band classification are, as stated, independent
functions of the same per-K states, and CAN disagree: re-executed this
revision over the full 1000-vector reachable state space, **371 of
1000 vectors had the K-scan return `DECIDED` while the whole-study
primary band was `INCOMPLETE-AT-K`** — dispatching a PAID ≤9.28 GPU-h
4-cell conditional arm to budget-qualify a `K_trig` inside a study
whose own 80K headline is, by §5's own rule, unreportable as a
frontier claim. This was never pre-registered either way
(`NCR_KWALL_ATTACK_R4.md` §2, KW5.5). Rev 4 closes it with one
precondition, checked AFTER the K-scan produces a `DECIDED` result and
BEFORE anything is dispatched: **the conditional arm fires only if the
whole-study primary band (§5, evaluated over the SAME triple via the
interval logic below) is ALSO decided — i.e. NOT `INCOMPLETE-AT-K`.**
If the K-scan says `DECIDED` but the band says `INCOMPLETE-AT-K`, the
trigger's result is overridden to `TRIGGER-UNRESOLVED`, disclosing the
K-scan's own candidate `K_trig` as a non-dispatched, band-blocked value
(`orchestrator_report.json`'s `trigger.band_blocked_K_trig` field,
below) rather than silently dropping it. This is discharge option (a)
of KW5.5's two — SUPPRESS, with disclosure — not option (b) — RUN,
with a qualifier sentence: a conditional spend that budget-qualifies a
K whose own primary read stays unreportable regardless buys a
standalone 160K datum at real GPU-h cost while the study's own
headline is still `INCOMPLETE-AT-K`; a future revision may adopt
option (b) if a reader is found who wants that standalone datum badly
enough to spend the GPU-h on it, but this design does not pre-register
that trade.

```
def trigger(states_26_28_30):
    branches = cross_product_of_AMBIGUOUS(states)   # 1, 2, 4, or 8 candidate (r26,r28,r30) triples (2^k for k AMBIGUOUS K's — §R5 KW6.12)
    K_trigs = set()
    for triple in branches:
        # scan K=26,28,30,32 in order; r24=4 (fixed ROBUST), r32=0 (fixed archive)
        kt = smallest_K_with_rate_below_3(triple)
        if kt requires reading an UNRESOLVED K's status to decide:
            # (§R8 K7/KW9.10 — same normalised 4-tuple shape as the
            # first copy above: (K_trig, resolution, resolution_detail,
            # diag).)
            return (None, "TRIGGER-UNRESOLVED", None, blocking_K)   # F2: a K that cannot resolve cannot trigger
        K_trigs.add(kt)
    if len(K_trigs) == 1:
        result = (K_trigs.pop(), "unanimous", None, None)
    else:
        # (§R7 J2, same fix as the first copy above) bare literal +
        # separate resolution_detail field.
        result = (min(K_trigs), "tie-break-min",
                  f"candidates were {sorted(K_trigs)}", None)
    # G5 precondition -- checked only once the K-scan itself decides:
    band = classify_with_interval_logic(states_26_28_30)   # sec.5's own procedure, SAME triple
    if band == "INCOMPLETE-AT-K":
        band_blocked_K_trig = result[0]   # the K-scan's own candidate K_trig -- disclosed, not dropped (§R8 K7: index 0, not the old index 1, now that the tuple is normalised)
        return (None, "TRIGGER-UNRESOLVED", None, band_blocked_K_trig)
    return result
```

**Re-run over the full 1000-vector reachable state space (executed
this revision — same `classify()`/K-scan implementation independently
re-verified against every prior round's own figures before trusting
the new number, not hand-derived).** Old split (K-scan alone, Rev 3's
rule, pre-G5): `DECIDED` **844**, `TRIGGER-UNRESOLVED` **156** —
reproduces the audit's own figures exactly. Of the 844, **371 were
paid-on-unresolved** (`DECIDED` while the whole-study band was
`INCOMPLETE-AT-K`) and the remaining **473** were genuinely
`(DECIDED, DECIDE)` — the audit's own joint-distribution figures
(`NCR_KWALL_ATTACK_R4.md` §2), reproduced to the digit. **New split
under the G5 precondition: `DECIDED` **473**, `TRIGGER-UNRESOLVED`
**527** — 0 of the 473 are paid-on-unresolved, by construction.**
`473+371=844` and `156+371=527` confirm arithmetically that the new
split is exactly the old split with the 371 defect cases moved from
`DECIDED` to `TRIGGER-UNRESOLVED` — not an independent recomputation
that happens to agree, the SAME recomputation partitioned one more
way. **The 11-configuration ambiguity table below is UNCHANGED by
G5:** every one of those 11 configurations is, by its own "Band (both
agree)" column, a case where the band ALREADY decides at both interval
candidates — i.e. `classify_with_interval_logic` never returns
`INCOMPLETE-AT-K` on any of them — so the new precondition cannot fire
on any row in that table; its `K_trig` values, tie-breaks, and
rationale stand exactly as printed below.

**The 11 ambiguous configurations (KW4.5's finding, independently
re-executed this revision — not merely cited).** A ~40-line Python
sweep (executed this revision, not hand-checked) enumerated every
singly-incomplete configuration — 3 choices of incomplete K × 4 values
of `r_known` × 25 combinations of the other two K's full 0–4 range =
300 configs — and flagged every one where `classify()` gives the SAME
band at both interval candidates (E4 would say "DECIDE") while the
OLD trigger rule's `K_trig` differs between them. **Exactly 11**,
matching the audit's count to the digit:

| # | Incomplete K | `r_known` | Candidate triples | Band (both agree) | `K_trig` candidates | Tie-break |
|---|---|---|---|---|---|---|
| 1 | 26 | 2 | (2,0,3) / (3,0,3) | FRONTIER-AT-K\*=30 [NON-MONOTONE] | {26,28} | **26** |
| 2 | 26 | 2 | (2,0,4) / (3,0,4) | FRONTIER-AT-K\*=30 [NON-MONOTONE] | {26,28} | **26** |
| 3 | 26 | 2 | (2,1,3) / (3,1,3) | FRONTIER-AT-K\*=30 [NON-MONOTONE] | {26,28} | **26** |
| 4 | 26 | 2 | (2,1,4) / (3,1,4) | FRONTIER-AT-K\*=30 [NON-MONOTONE] | {26,28} | **26** |
| 5 | 26 | 2 | (2,2,0) / (3,2,0) | GRADUAL-DECAY | {26,28} | **26** |
| 6 | 26 | 2 | (2,2,1) / (3,2,1) | GRADUAL-DECAY | {26,28} | **26** |
| 7 | 26 | 2 | (2,2,2) / (3,2,2) | GRADUAL-DECAY | {26,28} | **26** |
| 8 | 26 | 2 | (2,2,3) / (3,2,3) | FRONTIER-AT-K\*=30 [NON-MONOTONE] | {26,28} | **26** |
| 9 | 26 | 2 | (2,2,4) / (3,2,4) | FRONTIER-AT-K\*=30 [NON-MONOTONE] | {26,28} | **26** |
| 10 | 28 | 2 | (3,2,2) / (3,3,2) | GRADUAL-DECAY | {28,30} | **28** |
| 11 | 28 | 2 | (4,2,2) / (4,3,2) | GRADUAL-DECAY | {28,30} | **28** |

Every one of the 11 has `r_known=2` — the same single point KW4.7
independently flags as the ROBUST boundary interval logic cannot
disambiguate (§4 D5/E4 below) — and every disagreement is between
exactly two adjacent K's, so `min()` always resolves it within this
11-row, BAND-AGREEING, singly/doubly-incomplete domain: **no 3-way tie
is ever produced BY A BAND-AGREEING CONFIGURATION WITH AT MOST TWO
SIMULTANEOUSLY-INCOMPLETE K's** (verified by an extended sweep of the
two-K-simultaneously-incomplete case, 4-way candidate cross-products
over all 3 K-pairs × 4 `r_known` values × 5 third-K values = 240
configs checked: the maximum number of distinct `K_trig` values in any
band-agreeing configuration in THIS domain is **2**, never 3+).
Composition confirmed against every reachable interval-logic outcome —
the case analysis above is exhaustive over the singly/doubly-incomplete
domain, not illustrative. **Scope correction (KW5.6, Rev 4 — §R5
KW6.16 corrects this line's attribution; it was KW5.6's fix, not
KW5.13's, per §R4's own "Where fixed" column): the
UNQUALIFIED lead sentence ("every disagreement is between exactly two
adjacent K's") is FALSE over the FULL reachable state space, which
also permits all THREE of K∈{26,28,30} incomplete simultaneously.**
Re-swept this revision over the full non-`UNRESOLVED` per-K state
space (`9³=729` combinations, executed — matches the audit's own
figures to the count): candidate-`K_trig`-set sizes are
`{1: 612, 2: 102, 3: 14, 4: 1}` — 15 of 729 configurations (the `3`s
and the single `4`) DO produce 3-or-4-way ties. The 4-way case is
exactly "all three K's `AMBIGUOUS` at `r_known=2`," whose candidate set
is `{26,28,30,32}`. `min()` is still total over any candidate set (it
resolves a 4-way tie exactly as it resolves a 2-way one), and all 15
of these wider-tie cases land in `INCOMPLETE-AT-K` for the band
regardless (so, as of G5 above, they are ALSO `TRIGGER-UNRESOLVED` —
none of them reaches the tie-break at all) — nothing breaks, but the
narrower, correctly-scoped sentence is the one that is true, and is
the one that stands as of this revision.

**Command (conditional, if triggered) — same attempt-indexed-outdir
convention as the primary arm:**

```
ncr_earlyln_scale.py --cell --K {K_trig} --d-override {K_trig+1} \
  --seed {0,1,2,3} --steps 160000 --ceiling-gpuh 2.32 \
  --outdir /home/nvidia/ncr/results_kwall_characterization_160k/K{K_trig}_s{seed}_attempt{n} \
  --stop-file /home/nvidia/ncr/results_kwall_characterization_160k/STOP
```

`--ceiling-gpuh 2.32h` — `≥2×nominal` for every possible
`K_trig∈{26,28,30}` (worst case K=30's minimum is `2.3121h`, §4
pricing below; 2.32h clears it and every smaller-K minimum) — and,
per the same KW4.4 fix above, this is the value the ledger charges
directly, not a separate per-K figure.

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

**160K nominal per cell (conditional arm, worst-case K=30) — Rev 2
correction (KW3.11).** Three archive 2×/1× empirical ratios exist, not
one: K16 `1.9355`, K24/d48 `1.8580`, K32 `1.0510/0.5688 = 1.8477`. Rev
1 applied the MINIMUM of the three (1.848×) — the audit found this
understates the 160K nominal by up to 5% (a non-conservative choice
inside a pricing figure). **Rev 2 applies the MAXIMUM of the three
(1.9355×, K16) instead** — the conservative choice, since this figure
now feeds a per-cell ceiling floor (E2, informational sizing per
KW4.4) and the orchestrator's cumulative ledger (F1), both of which are
safety bounds that should never be built on an understated nominal:

| K | 160K nominal (h/cell) | ×4 seeds |
|---|---|---|
| 26 | 0.9882 | 3.953 |
| 28 | 1.0716 | 4.286 |
| 30 | 1.1561 | 4.624 |

(vs. Rev 1's 0.9434/1.0230/1.1037 — every value moves up ≈4.5–4.8%,
`0.5105×1.9355=0.9882` (§R5 KW6.11: corrected from `0.5106`, a stray
digit inconsistent with the already-verified `0.5105` at K=26's own
80K-nominal row above — the product `0.9882` is unchanged, this fixes
only the operand), `0.5536×1.9355=1.0716`,
`0.5973×1.9355=1.1561`, all directly executed this revision, not hand
math.)

**ORCHESTRATOR CONTRACT (F1, Rev 3 — replaces E1's launcher-side
cumulative-realized-GPU-h check, which KW4.1–KW4.3 found blind to
aborted spend, derived from a false premise, and unimplementable in
the repo's real dispatch path). Specified precisely enough for a
build-stage implementer to write the script from this design alone —
this document remains DRAFT and creates no code.**

**Cell order (deterministic, K-major).** Primary arm: K=26
(seed 0,1,2,3), then K=28 (seed 0,1,2,3), then K=30 (seed 0,1,2,3) —
12 cells. All 12 reach a terminal state (below) before the trigger
(§4, above) is evaluated even once. If triggered, the 4 conditional
cells at `K_trig` (seed 0,1,2,3) run next, same discipline. Exactly
ONE cell-attempt is ever in flight — never parallel, never split
across GPUs.

**Per-cell-attempt dispatch loop — the abort/retry state machine (Rev
4, G1/G3: write-ahead ledger record + exit-code-exact branching;
replaces the "unconditional ledger update after return" / "non-zero
exit ⇒ ABORTED-BUDGET" text KW5.1/KW5.3 found unsafe).**
For each (K, seed) in cell order:
1. **Attempt 1.** HARD GATE check (below). Refused → append a
   `GATE-REFUSED` row to `ledger.attempts` (`elapsed_h=0.0`,
   `outdir=null`, `d_override` recorded from the CLI value this
   attempt would have carried — §R5 H4 below); `realized_gpu_h` is
   UNCHANGED (only a `0.0`-valued row is added — §R5 KW6.5(i)); move
   to the next cell (treated identically to MISSING by the
   resolution-state table above and by `harvest()`, §4 D5/E4 — G1's
   cell-level resume rule, below, skips this cell by its DERIVED
   terminal state, §R6 I3, never by an absent row: every touched
   cell/attempt now always has a row, GATE-REFUSED included, so
   "no row of any kind" could never distinguish it from an untouched
   cell in the first place).
   Admitted → **write-ahead (G1):** BEFORE calling `subprocess.run`,
   the orchestrator sets `ledger.open_attempt = {K, seed, arm,
   attempt_n, charged_ceiling: ceiling(this_attempt), dispatch_ts:
   time.monotonic()}` and persists `ORCHESTRATOR_LEDGER.json`
   ATOMICALLY via `rn.atomic_write_json` (`matrix-thinking/ncr/
   run_ncr.py:105-109` — writes `<path>.tmp`, then `os.replace(tmp,
   path)`; §R5 H1 below) — the ONLY ledger write that ever happens
   BEFORE a subprocess runs, so a mid-attempt orchestrator death
   leaves a detectable gap, never a silent one (recovery procedure
   below). Then run the subprocess (command shape above, `attempt1`
   outdir). The orchestrator's OWN wall-clock timer — `t0`
   (`dispatch_ts` above), `t1` immediately after `subprocess.run`
   returns, `attempt_elapsed_h=(t1-t0)/3600` — is the measurement,
   **not** the cell JSON's `gpu_h` field (KW4.1: that field is
   assigned only on the `COMPLETED` path, `ncr_earlyln_scale.py:304`,
   and is absent entirely on the `ABORTED-BUDGET` early-return path,
   `:262-266`).

   **On return — CLASSIFY, then (if `COMPLETED`) COPY, then FOLD (§R5
   H2: copy-then-fold, replacing "fold unconditionally, before status
   is inspected" — KW6.2 found a crash between the OLD fold and the
   copy silently drops a `COMPLETED` cell from `harvest()`'s count
   forever; KW6.10 found the old row was written before its own
   `status` field was known):**
   1. **Classify (exit-code-exact branch, G3, unchanged from Rev 4
      except the new default arm marked `*`):**
      - exit code `3` (the `--stop-file` sentinel,
        `ncr_earlyln_scale.py:196`) → `STOPPED-BY-OPERATOR` (never
        retried — see "operator stop," below).
      - cell JSON written with `status=="COMPLETED"` → `COMPLETED`
        (terminal).
      - cell JSON written with `status=="ABORTED-BUDGET"` →
        `ABORTED-BUDGET-1`, proceed to step 2 (below).
      - any OTHER non-zero exit, no `COMPLETED`/`ABORTED-BUDGET` JSON
        on disk → `CRASHED-1` (a deterministic crash — shape error,
        OOM, import failure — never a coin-flip seed the way a budget
        abort is; disclosed distinctly, per KW5.3), proceed to step 2
        under the SAME gates.
      - `*` **exit code `0`, no `COMPLETED`/`ABORTED-BUDGET` JSON on
        disk (§R5 KW6.5(iii) — reachable if `main` exits cleanly
        without ever dispatching, or a future CLI path returns early)
        → `CRASHED-1`,** the same default as any other JSON-less
        non-zero exit. (The full exit-code × JSON cross-product,
        including the two branches this makes UNREACHABLE, is the
        unified table in §R5 H4 below — this bullet list is the
        single source of truth for the LIVE build, superseding it in
        case of any apparent conflict.)
   2. **Copy — `COMPLETED` only (G2's trigger condition, unchanged).
      ATOMIC** (§R5 H2): copy the archival attempt JSON to
      `<canonical_path>.tmp` inside the canonical directory, then
      `os.replace(<canonical_path>.tmp, canonical_path)`. A crash
      mid-copy leaves only an orphaned `.tmp` file — unmatched by
      `discover_seeds_by_K`'s glob, harmless — never a truncated
      canonical file (closes KW6.2's second face). The pre-copy
      `os.path.exists(canonical_path)` exists-check (G2, below) fires
      first, unchanged, and still aborts loudly on a genuine
      duplicate.
   3. **Fold — `status` is already known; never written as a
      placeholder (KW6.10 closed as a byproduct of this reordering,
      not by a separate two-phase write).** `ledger.realized_gpu_h +=
      attempt_elapsed_h` UNCONDITIONALLY, `ledger.open_attempt` is
      cleared, and a terminal row carrying the step-1 classification
      is appended to `ledger.attempts`; the ledger is persisted
      ATOMICALLY. For `COMPLETED` this step is now strictly AFTER
      step 2's copy — copy-then-fold — so a `COMPLETED` row can never
      exist in `ledger.attempts` unless its canonical file already
      exists on disk (the crash-window walk, §R5 H2 below).
2. **Retry (attempt 2), from `ABORTED-BUDGET-1` OR `CRASHED-1` only —
   never from `STOPPED-BY-OPERATOR`.** HARD GATE **and** RETRY GATE
   (below), both checked against the ledger AS IT STANDS after attempt
   1 — already updated in step 1 (and, if attempt 1 crashed rather than
   returned, closed by the G1 recovery procedure on the next restart
   BEFORE this check ever runs) — so there is no staleness: strict
   sequencing plus write-ahead recovery means the ledger is exactly
   current, or conservatively over-stated, at every check. Both pass →
   run attempt 2 (`attempt2` outdir, identical write-ahead +
   classify-copy-fold discipline, §R5 H2); `COMPLETED` → state
   `COMPLETED`; `ABORTED-BUDGET`/`CRASHED` again (whichever recurs) →
   the attempt-2 row records that outcome and the CELL derives to
   `PERSISTENTLY-ABORTED` (terminal — feeds interval logic as an
   unknown-numerator seed regardless of which of the two produced it,
   D5/E4 below). **`PERSISTENTLY-ABORTED` is a DERIVED CELL state, and
   is NEVER itself an `attempts[].status` value (§R5 KW6.5(ii)) — it
   is derived from `ledger.attempts` rows by the rule: a cell is
   `PERSISTENTLY-ABORTED` iff its attempt-2 row exists and is
   non-`COMPLETED`, OR its attempt-1 row is non-`COMPLETED` and no
   attempt-2 row exists and the retry gate closed it.** Either gate
   fails → append a `GATE-REFUSED` row for attempt 2
   (`elapsed_h=0.0`, `outdir=null`, `d_override=K+1` — the SAME
   treatment attempt 1's refusal gets, §R5 KW6.5(i); this is what
   gives `COMPLETE-DEGRADED`'s *primary-retry-refused* sub-case (below)
   the positive on-disk evidence its `validity_check` disk-evidence
   assertion, §R5 H4, depends on) — the cell derives to
   `PERSISTENTLY-ABORTED` immediately via the rule above, attempt 2 is
   never dispatched, and `realized_gpu_h` is unchanged (only the
   `0.0`-valued row is added). (Retraining is still from scratch — the
   harness has no
   checkpoint resume, only a `status=="COMPLETED"` skip, `:243-245` —
   so a retry still costs close to a full ceiling again; what Rev 3
   fixed is that the retry's OWN spend, successful or not, is now
   always counted and never overwrites the first attempt's record, per
   the attempt-indexed outdir above; what Rev 4 fixes is that the
   ledger is no longer silently short if the orchestrator itself dies
   mid-attempt, G1 below.)

**Operator stop (G3, `STOPPED-BY-OPERATOR`).** On exit code `3`, the
orchestrator does not retry, does not advance to the next cell, and
does not evaluate the trigger — it flushes the ledger (a normal,
non-open terminal write, since exit code 3 means `subprocess.run` DID
return) and exits immediately, with `run_status="STOPPED-BY-OPERATOR"`
(G4, below) written to `orchestrator_report.json`. This is terminal for
the WHOLE wave, not merely the one cell — an operator touching the
STOP file is asking the run to stop, not asking one cell to be marked
aborted so the next one can be dispatched. Before this fix, `sys.
exit(3)` was read as a generic non-zero exit and misfiled as
`ABORTED-BUDGET-1`, immediately re-dispatching attempt 2 (which exits
3 again in seconds) and converting the kill switch into a
mass-`PERSISTENTLY-ABORTED` of every remaining cell in under a minute
(KW5.3).

**Gate check points, exact — checked immediately BEFORE every
dispatch (attempt 1 or the retry), never after:**
1. **HARD GATE.** `ledger.realized_gpu_h + ceiling(this_attempt) ≤
   15.00`, else refuse. `ceiling(this_attempt)` is the CLI
   `--ceiling-gpuh` value passed to THIS attempt — `1.20` for every
   primary attempt, `2.32` for every conditional attempt. **The
   charged value and the enforced value are the same number by
   definition (KW4.4 closed)** — there is no separate per-K
   `max(2×nominal,1.0)` figure inside the gate; that figure's only
   remaining role is the informational sizing table below.
2. **RETRY GATE (attempt-2 dispatches only, subordinate to 1 — both
   must pass).** `ledger.realized_gpu_h < 12.00`, else the cell goes
   straight to `PERSISTENTLY-ABORTED` with no second attempt,
   reserving the residual `15.00−12.00=3.00` GPU-h band exclusively
   for completing outstanding first attempts.

**Ledger persistence + write-ahead recovery (G1, Rev 4 — replaces the
"never resets `realized_gpu_h` to 0" text KW5.1 found insufficient: it
guards a *fresh-start* reset but says nothing about a *mid-attempt*
death, which under Rev 3's contract left the ledger silently short by
up to one full ceiling and reopened budget an earlier crash had
already consumed, with nothing capping how many times that could
repeat — CLAUDE.md's resume-safe-supervisor rule; Rev 5, §R5 H1,
additionally fixes the ledger's OWN write durability and adds a
recovery branch for a ledger it cannot read at all).**

`/home/nvidia/ncr/results_kwall_characterization/ORCHESTRATOR_LEDGER.json`
carries `realized_gpu_h`, a completed-attempt log `attempts[]` (one row
per TERMINAL attempt: `{K, seed, arm, attempt_n, elapsed_h, status,
outdir, d_override, ceiling_charged}` — `d_override` and
`ceiling_charged` are new this revision, §R5 H4/H5 below), and exactly
one `open_attempt` slot (`null` when nothing is in flight — dispatch
is strictly serial, so at most one can ever be open at a time).
`ceiling_charged` is `true` iff `elapsed_h` is a gate-admitted ceiling
value rather than a measured wall-clock value (every
`CRASHED-RECOVERED` row, and every row produced by disk reconstruction
below, regardless of its `status`).

**Every ledger write is ATOMIC (§R5 H1 — closes KW6.1: the document
previously specified `open_attempt` writes and terminal-row folds with
no atomicity or durability contract at all).** The repo already ships
the primitive this design now names explicitly: `rn.atomic_write_json`
(`matrix-thinking/ncr/run_ncr.py:105-109` — serializes to
`<path>.tmp` via `json.dump`, then `os.replace(tmp, path)`, the same
POSIX-atomic-rename helper `ncr_earlyln_scale.py` already uses for
every cell JSON, `:265` and `:307`). The orchestrator uses this SAME
helper for `ORCHESTRATOR_LEDGER.json` on every one of its writes — the
`open_attempt` write BEFORE `subprocess.run`, and the classify-copy-
fold sequence's fold AFTER it returns (above) — never a bare
`json.dump(ledger, open(path,"w"))`, which truncates the file first
and is exactly the FATAL KW6.1 found: a crash mid-write of that form
leaves a truncated, unparseable ledger. Under `rn.atomic_write_json`,
an observer can only ever see the FULLY-PRIOR ledger state or the
FULLY-NEW one, never a partial file — the truncation window KW6.1
found is closed by construction, not by a recovery branch. The
recovery branch below (step 0) exists for the residual case: a ledger
made unparseable by something OTHER than this design's own writer (a
manually-edited file, a filesystem fault, a foreign process).

**Recovery procedure, run on ANY (re)start, BEFORE any gate check and
BEFORE the cell-order walk resumes:**
0. **Read `ORCHESTRATOR_LEDGER.json` (§R5 H1).** If it parses, proceed
   to step 1 unchanged. **If it is MISSING or UNPARSEABLE —
   CONSERVATIVE RECONSTRUCTION FROM DISK, PER-ATTEMPT-DIRECTORY, never
   a fresh start (§R6 I1 — replaces the Rev-5 per-CELL procedure
   KW7.1 found under-charging: it wrote at most one row per (K, seed)
   regardless of how many attempt directories existed on disk, and it
   gated the conditional arm on its CANONICAL directory rather than its
   own archival evidence, silently erasing real spend whenever that arm
   crashed before its first copy).** Reconstruction now iterates every
   `(K, seed)` in the full cell order — primary 12 UNCONDITIONALLY,
   plus the conditional 4 UNCONDITIONALLY (never gated on the
   conditional CANONICAL directory's existence — closes KW7.1(b): the
   conditional cells reconstruct from the SAME archival-attempt-dir
   evidence the primary cells do) — and for each one, every
   `attempt{n}/` directory, `n∈{1,2}`, in that cell's own archival tree
   (`.../K{K}_s{seed}_attempt{n}/`).

   **0.0 Canonical sanity pass (once per (K, seed), before the
   per-attempt loop).** Read the canonical-path file if one exists:
   parses with `status=="COMPLETED"` → `canonical_state=OK` (its own
   `elapsed_s` is kept for 0.2's bootstrap fallback); present but
   unparseable, OR parses with `status!="COMPLETED"` →
   **QUARANTINE-RENAME** (`os.rename(canonical_path, canonical_path +
   f".CORRUPT-{int(time.time())}")`), `canonical_state=CORRUPT` — the
   missing else-branch KW7.1(c) found: the file is no longer AT
   `canonical_path` afterward, so G2's pre-copy
   `os.path.exists(canonical_path)` check (§4, above) can never trip
   against it on a later re-run — the corrupt evidence is preserved on
   disk (renamed, not deleted) but is no longer in G2's way; absent →
   `canonical_state=ABSENT`.

   **0.1 Per-attempt-directory reconstruction — a TOTAL function (every
   reachable disk state maps to exactly one outcome row; state
   totality, verified below).** For each attempt directory `n∈{1,2}`,
   cross this attempt's own evidence against the shared
   `canonical_state` (0.0) — snapshotted ONCE, before the per-attempt
   loop, and deliberately NOT refreshed by a sibling slot's own PROMOTE
   write within the same pass (§R7 KW8.8 — every downstream consumer of
   `ledger.attempts` counts DISTINCT `(K,seed)` pairs, never raw row
   count — "the G2+H2 identity," above, and the `COMPLETE`/
   `COMPLETE-DEGRADED` canonical-count assertions, below — so a stale
   snapshot causing an occasional harmless double-`COMPLETED`-row
   double-charge is absorbed by that set-based counting, not a
   correctness gap; re-reading `canonical_state` per-slot would only
   change which slot is credited with the PROMOTE, never whether the
   cell is correctly derived `COMPLETED`) — and `charged_ceiling(arm)`
   (`1.20` primary / `2.32` conditional — the only place `arm` enters
   this table; it selects a NUMBER, never a branch). **Charging rule
   (§R7 J6 — restores the row-wise invariant "ledger row ≥ that
   attempt's true spend," closing KW8.6: every row below charges FULL
   `charged_ceiling(arm)` EXCEPT a row whose `status=="COMPLETED"`
   (completed work, promoted or already consistent), which charges
   MEASURED `rec["elapsed_s"]/3600` (§R9 m5 — closes KW10.6's residue:
   the TOP-LEVEL field, `ncr_earlyln_scale.py:302`, never the nested
   `rec["train"]["elapsed_s"]` at `:202`, per KW9.11's same
   disambiguation) PLUS a STARTUP ALLOWANCE `s=0.0053` GPU-h
   — derivation, "True spend, worst case," below.** A parseable JSON
   whose `status` key is missing or not one of the 6 enumerated values
   is treated as UNPARSEABLE (§R7 KW8.9 — row 10's treatment, full
   ceiling; 0.0 already quarantines the identical case on the canonical
   side, `status!="COMPLETED"` is true for a missing key there too, so
   this is the same rule applied consistently to attempt JSONs):

   | Attempt dir | Attempt JSON | `canonical_state` | Row appended (`attempt_n` = this directory's own `n`) |
   |---|---|---|---|
   | absent | (absent) | OK | none — this attempt slot never ran; the canonical is accounted for by the sibling slot's own row, or by 0.2's bootstrap if NEITHER slot has evidence |
   | absent | (absent) | CORRUPT | none — see 0.2's bootstrap (canonical already quarantined at 0.0) |
   | absent | (absent) | ABSENT | none — genuinely never dispatched, cell available for dispatch |
   | present | parseable, `status=="COMPLETED"` | OK | `{attempt_n, elapsed_h:rec["elapsed_s"]/3600 + s (MEASURED + STARTUP ALLOWANCE — §R8 K7/KW9.11: the TOP-LEVEL field, `ncr_earlyln_scale.py:302`, never the nested `rec["train"]["elapsed_s"]` at `:202`, which excludes `z_dump`/the deep probe/the Axis-C lock/the trust screen/`blank_out_check`/`eval_cell`), status:"COMPLETED", outdir:<canonical>, d_override:K+1, ceiling_charged:false}` — already consistent with canonical, no copy needed |
   | present | parseable, `status=="COMPLETED"` | CORRUPT | **PROMOTE (§R6 I2):** copy this attempt's own JSON to `<canonical_path>.tmp`, `os.replace(<canonical_path>.tmp, canonical_path)` — the same atomic pattern G2 uses live. Row as above (measured + `s`, `ceiling_charged:false`) |
   | present | parseable, `status=="COMPLETED"` | ABSENT | **PROMOTE (§R6 I2)** — same copy+rename; this is the "before copy" crash window's fix (crash-window table, above): the science exists and is now USED, never discarded. Row as above |
   | present | parseable, `status!="COMPLETED"` (e.g. `ABORTED-BUDGET`) | OK | `{attempt_n, elapsed_h:charged_ceiling(arm) (§R7 J6 — FULL ceiling; measurement alone is no longer trusted to cover true spend for a non-`COMPLETED` reconstructed row, closing KW8.6's Class-1 leak), status:<the JSON's own status>, outdir:null, d_override:K+1, ceiling_charged:true}` — canonical's `COMPLETED` is attributed to the sibling attempt slot |
   | present | parseable, `status!="COMPLETED"` | CORRUPT | quarantine already ran (0.0); row as above (full ceiling) — "cell treated per its attempt evidence" (§R6 I1) |
   | present | parseable, `status!="COMPLETED"` | ABSENT | row as above (full ceiling) |
   | present | unparseable (or missing/invalid `status`, §R7 KW8.9) | any | `{attempt_n, elapsed_h:charged_ceiling(arm) (FULL ceiling — no measurement is readable), status:"CRASHED-RECOVERED", outdir:null, d_override:K+1, ceiling_charged:true}` |
   | present | absent (mid-attempt crash, before any status write) | any | same treatment as unparseable — no measurement exists either way: `{attempt_n, elapsed_h:charged_ceiling(arm), status:"CRASHED-RECOVERED", outdir:null, d_override:K+1, ceiling_charged:true}` |

   This is **24 exhaustively-covered state-space cells**: 2 (attempt
   dir present/absent) × 3 (attempt JSON parseable/unparseable/absent)
   × 3 (`canonical_state` OK/CORRUPT/ABSENT) × 2 (arm) = 36 nominal,
   minus the 12 impossible `dir=absent` × `JSON∈{parseable,
   unparseable}` × canonical(3) × arm(2) combinations (a JSON cannot
   exist without its directory) = 24 valid — **12 core `(dir, JSON,
   canonical)` states, rendered as 11 table rows** (§R7 KW8.11 — rows
   10/11 each merge all three `canonical_state` values into one row
   while the `parseable` triples split into two sub-case rows each,
   `3+6+1+1=11` physical rows for 12 core states; "12" is the count the
   `12×2=24` arithmetic actually needs, not the row count, which this
   sentence previously conflated) × the 2-way `arm` ceiling-value split
   noted inline (`arm` changes no branch, only which number
   `charged_ceiling` reads, and which archival tree root is scanned).
   **Every dispatched attempt provably leaves an `attempt{n}/`
   directory** (`os.makedirs(outdir,
   exist_ok=True)`, `ncr_earlyln_scale.py:237`, before training and
   before the resume-skip — §R6 I6/KW7.15's premise, stated explicitly
   rather than left implicit) — so "attempt dir absent" is sound
   evidence this design's own dispatch path never ran it; the precise
   claim is "left no evidence this design's dispatch path can
   produce," never "provably never ran" (KW7.15 — absence of evidence a
   THIS design's writer would leave is not proof of absence against
   every possible cause). **The mirror gap — a crash BEFORE
   `os.makedirs` (interpreter start, `torch`/CUDA import, arg parsing)
   leaves no attempt directory and is charged `0` for a window in which
   real GPU time can be spent (§R7 KW8.12) — is the exact same
   process-startup term `s` prices for a COMPLETED row, above; it is
   disclosed here rather than charged, because charging requires
   evidence this window structurally cannot leave (KW7.15's own point).
   **(§R8 K7/KW9.8 — retracts "and is bounded by the same derivation":
   this window has no term and no multiplicity cap in `T`'s expression
   below, and its occurrence count is capped only by the restart count,
   which no section bounds — `T ≤ 15.3737h` is an honest bound on
   spend for which disk EVIDENCE survives, not on all true spend. It
   is instead screened SYSTEMATICALLY by the mandated micro-smoke gate
   below — a systematic failure in this window is a smoke-test kill,
   not a per-run leak — and each TRANSIENT occurrence is small (bounded
   by `s`, ≈19 s), left to the disclosed `0.30h` margin, the same
   honesty J6 applies everywhere else.)**

   **0.2 Cell-level bootstrap fallback (the residual case: canonical
   evidence survives with NO `COMPLETED` row yet accounted for by 0.1
   for that cell — e.g. an attempt tree pruned after copying, OR — §R7
   J3, closes KW8.3's MAJOR — a `(K,seed)` whose surviving attempt
   dir(s) recorded only a NON-`COMPLETED` outcome, `ABORTED-BUDGET-1`
   say, while the canonical file independently shows the cell actually
   completed).** For any `(K, seed)` where **0.1 appended NO row with
   `status=="COMPLETED"`** (§R7 J3 — widened from Rev 6's "0.1 appended
   ZERO rows," which left a canonical `COMPLETED` file unaccounted for
   whenever 0.1 had already written ≥1 NON-`COMPLETED` row for that
   cell; KW8.3 found 30/200 cell compositions hit exactly this gap, 6
   of them additionally derived NON-terminal and re-dispatched by step
   3 onto a cell whose science already existed) and `canonical_state≠
   ABSENT` at 0.0: let `bootstrap_n = max(every attempt_n already
   recorded by 0.1 for this cell, default 0) + 1` (§R7 J3 — replaces
   Rev 6's hard-coded `attempt_n:1`, which would silently COLLIDE with
   an existing attempt-1 row exactly in the gap case above; a
   bootstrap row's `attempt_n` is a RECONSTRUCTION LABEL, not a future
   dispatch number, and may exceed `2` — harmless, because every cell a
   bootstrap row lands on derives TERMINAL (`COMPLETED` via this same
   row, or `PERSISTENTLY-ABORTED` via the CORRUPT branch below) and is
   therefore skipped in full at step 3, never re-numbered for dispatch;
   confirmed by direct execution — 0/200 abort-trips in the corrected
   sweep, below, including every composition where `bootstrap_n>2`
   fires). If `canonical_state` was `OK`, append ONE row
   `{K, seed, arm, attempt_n:bootstrap_n, elapsed_h:<the canonical
   JSON's own rec["elapsed_s"]>/3600 + s (MEASURED + STARTUP ALLOWANCE,
   §R7 J6 — canonical is a byte copy of the completing attempt's own
   JSON and retains its fields; **§R8 K7/KW9.11: this is the TOP-LEVEL
   `rec["elapsed_s"]`, `ncr_earlyln_scale.py:302`, never the nested
   `rec["train"]["elapsed_s"]` at `:202` — reading the nested field
   under-charges by the whole post-train instrument sequence, a term
   orders of magnitude larger than `s`**), status:"COMPLETED",
   outdir:<canonical path>, d_override:K+1, ceiling_charged:false}`; if
   `canonical_state` was `CORRUPT` (already quarantined at 0.0), append
   ONE conservative row `{K, seed, arm, attempt_n:bootstrap_n,
   elapsed_h:charged_ceiling(arm) (FULL
   ceiling — no attempt-dir evidence survives to measure),
   status:"CRASHED-RECOVERED", outdir:null, d_override:K+1,
   ceiling_charged:true}` — something produced the quarantined file;
   one full ceiling is the conservative floor, now scoped to only this
   rare filesystem-anomaly case rather than the general rule.

   `realized_gpu_h` = the sum of every reconstructed row's `elapsed_h`
   (0.1 + 0.2 together); `open_attempt = null`. Persist the
   reconstructed ledger ATOMICALLY (`rn.atomic_write_json`), then
   proceed to step 1. **Reconstruction charges ≥ one ceiling-or-measured
   amount per attempt for which disk evidence exists (§R6 I1 — replaces
   the refuted "no path through this step re-opens budget" argument:
   positivity of one row said nothing about whether the SUM replaced a
   larger true spend) — never zero for a dispatched attempt, and never
   more than one row per attempt directory found, closing KW7.1(a)/(b)/
   (c) with the same per-attempt-directory rule.** A `(K, seed)` with NO
   disk evidence at all — neither attempt directory nor canonical file
   — gets no row and remains genuinely available for dispatch.

   **200-state cell-level composition, RE-RUN under the §R7 J3 fix
   (closes KW8.3's MAJOR).** The 0.0/0.1/0.2 rules above were
   transcribed verbatim (`recon_r7.py`, this revision's session
   scratchpad, following the same 5 attempt-1 states × 5 attempt-2
   states × 4 raw-canonical states × 2 arms = 200 shape the audit's own
   `recon.py` used) and executed twice — once against Rev 6's OLD guard
   ("0.1 appended ZERO rows") and once against the amended NEW guard
   ("0.1 appended no `COMPLETED` row") — counting two things: **orphans**
   (`canonical_state==OK` — a `COMPLETED` canonical file survives 0.0's
   quarantine pass — with no `COMPLETED` row in the reconstructed
   ledger) and **abort-trips** (the cell derives NON-TERMINAL, i.e.
   step 3 would dispatch a further attempt, while `canonical_state==OK`
   — the redispatched attempt's own eventual `COMPLETED` JSON then trips
   G2's pre-copy exists-check on completion):

   | Guard | Orphans | Abort-trips |
   |---|---|---|
   | OLD (Rev 6, "0.1 appended ZERO rows") | **30 / 200** | **6 / 200** |
   | NEW (§R7 J3, "0.1 appended no `COMPLETED` row") | **0 / 200** | **0 / 200** |

   The OLD-guard run reproduces the audit's own KW8.3 figures exactly
   (30 orphans, 6 abort-trips — three unique `(a1,a2,raw_canonical)`
   compositions × 2 arms each, matching KW8.3's own worked table
   digit-for-digit). The NEW-guard run drives both counts to 0/200: 72
   of the 200 NEW-guard states produce a bootstrap row with
   `attempt_n>2` (the "reconstruction label, not a dispatch number"
   case flagged above); every one of those 72 derives `COMPLETED` or
   `PERSISTENTLY-ABORTED` — both terminal — confirmed by the same
   0-abort-trip count, not asserted separately. Re-run by direct
   execution, not hand-checked, per the round-8 scope's own instruction.
1. If `open_attempt` is `null`, nothing is dangling — proceed to step 3.
2. **A non-null `open_attempt` means the orchestrator died between
   writing it and clearing it — mid-attempt (§R5 H2 distinguishes a
   genuine crash from a crash that landed after the copy but before the
   fold; §R6 I2 adds a check that runs FIRST, before either of those
   two branches, so a provably-`COMPLETED` attempt is never discarded
   regardless of its attempt number).** Before closing anything:
   1. **(§R6 I2 — runs BEFORE any branch that could lead to a
      `PERSISTENTLY-ABORTED` derivation.)** Read
      `.../K{open_attempt.K}_s{open_attempt.seed}_attempt{
      open_attempt.attempt_n}/earlyln_K{open_attempt.K}_s{
      open_attempt.seed}.json` (deterministic from the fields
      `open_attempt` already carries). If it parses and
      `status=="COMPLETED"`: **PROMOTE** — copy it to
      `<canonical_path>.tmp`, `os.replace(<canonical_path>.tmp,
      canonical_path)` (skip the copy if a canonical file already
      exists there and itself parses `COMPLETED` — nothing to do), then
      append a terminal row `status="COMPLETED"`, `elapsed_h=
      rec["elapsed_s"]/3600 + s` (MEASURED + STARTUP ALLOWANCE, §R7
      J6 — **§R8 K7/KW9.11: the TOP-LEVEL field, `:302`, never the
      nested `rec["train"]["elapsed_s"]` at `:202`** — the
      subprocess's own timer survived in its JSON even though the
      orchestrator's did not, but it is the SAME understated timer 0.1's
      table uses, `t0` set at `ncr_earlyln_scale.py:257`, so the same
      allowance applies here for the same reason, "True spend, worst
      case," below),
      `outdir=<canonical path>`, `ceiling_charged:false`,
      `d_override=open_attempt.K+1`, and skip the two branches below.
      **This closes KW7.2:** the prior text discarded exactly this case
      whenever the crash landed in the "before copy" window, which for
      an attempt-2 crash left no retry and silently shrank the harvest
      denominator (a demonstrably-`COMPLETED` cell scored as never
      having completed); the completed science is now reused on
      attempt 1 or attempt 2 alike, never re-run from scratch.
   2. Otherwise (the JSON is missing, unparseable, or parses
      non-`COMPLETED`), check whether a canonical-path file ALREADY
      exists for `(open_attempt.K, open_attempt.seed)` with
      `status=="COMPLETED"`:
      - **YES — canonical file + dangling open record (§R5 H2's named
        "between copy and fold" crash window): this is PROOF OF
        COMPLETION, not a crash to conservatively write off.** Append a
        terminal row `status="COMPLETED"`, `elapsed_h=open_attempt.
        charged_ceiling` (still the gate-admitted ceiling, not a
        measurement — the timer that would have measured it died with
        the process), `outdir=<canonical path>`, `ceiling_charged=true`,
        `d_override=open_attempt.K+1`. This restores `COMPLETED ⇒
        canonical`'s converse (G2's own contract already guarantees
        `canonical ⇒ COMPLETED`) — **with both directions now holding,
        the harvest-patch-unnecessary claim (below) is true BY THE PAIR,
        not by one direction alone.**
      - **NO — genuine crash (the "before copy" and "mid-copy" windows,
        now reached ONLY when 2.1's own-JSON check above ALSO found no
        proof of completion; the atomic copy-to-temp+rename means a
        mid-copy crash leaves no canonical file, only a harmless
        orphaned `.tmp`, so it is indistinguishable on disk from
        "before copy" and gets the same treatment).** Close it
        conservatively, as before: append a terminal row
        `status="CRASHED-RECOVERED"`, `elapsed_h=
        open_attempt.charged_ceiling`, `ceiling_charged=true`,
        `d_override=open_attempt.K+1`. That cell's attempt/retry state
        machine treats `CRASHED-RECOVERED` exactly like `CRASHED-n` (G3)
        for retry-gating and interval-logic purposes (D5/E4, below) — a
        crash the orchestrator recovers FROM restart is not
        distinguishable, from the cell's perspective, from a crash it
        observed directly.
   3. **Either branch:** before persisting the closure, verify no LIVE
      process still holds the GPU this attempt was assigned (§R5
      KW6.17 — e.g. `nvidia-smi --query-compute-apps`); a kernel
      OOM-kill of the Python parent alone can orphan a live CUDA
      process the tmux-session-kill premise elsewhere in this document
      assumes does not happen. If one is found, ABORT LOUDLY rather
      than re-dispatching attempt 2 onto a GPU attempt 1 may still be
      running on.
   Then set `open_attempt=null` and persist ATOMICALLY.
3. **Cell-level resume, stated CELL-WISE and normatively (§R6 I3 —
   closes KW7.3: the Rev-5 wording mixed a cell-wise skip clause with
   an attempt-wise "no ledger row" resume clause, and the two disagreed
   whenever reconstruction had written a row at `attempt_n:2` with no
   `attempt_n:1` row for the same cell — a state §R6 I1's per-attempt
   reconstruction no longer produces in the ordinary case, but the rule
   itself must not depend on that being true).** A cell whose DERIVED
   state is TERMINAL — `COMPLETED`, `PERSISTENTLY-ABORTED` (the rule
   above: iff its attempt-2 row exists and is non-`COMPLETED`, OR its
   attempt-1 row is non-`COMPLETED`, no attempt-2 row exists, and the
   retry gate closed it), or `STOPPED-BY-OPERATOR` — is skipped IN
   FULL, regardless of which individual `attempt_n` rows exist for it.
   This is never re-gated: a restart at `realized≈13h` can never turn
   an already-`COMPLETED` cell into `GATE-REFUSED`, because terminal
   status comes from the ledger record, never from a fresh HARD-GATE
   re-decision.

   **Attempt numbering is authoritative and single-sourced (§R6 I3's
   one precedence sentence): reconstruction's own `attempt_n` values
   (§R6 I1 — each attempt directory's own number) are the SAME
   numbering space normal dispatch writes into; there is no separate
   reconstruction-only convention left to disagree with it.** For a
   cell whose derived state is NON-terminal, dispatch resumes at
   `attempt_n = max(every recorded attempt_n for this cell, default 0)
   + 1` — always `≤2`, because a cell already carrying an
   `attempt_n:2` row is, by the derivation rule above, either already
   terminal (skipped) or `COMPLETED` (not dispatched again at all).
   Never below the recorded maximum — an already-numbered attempt is
   never re-dispatched — and never a fresh `attempt_n:1` for a cell
   that already carries an `attempt_n:1` row of any status.
4. Once every dangling record is closed (step 2) and every
   already-terminal cell is skipped in full (step 3), dispatch resumes
   at the next non-terminal cell's `max(attempt_n)+1` — normal
   operation from here.

A restarted orchestrator therefore never resets `realized_gpu_h` to 0
(Rev 3, unchanged) AND never leaves a crashed attempt's spend
unrecorded past the NEXT restart (Rev 4, new) — together these close
the "reopens budget" failure mode: every crash→restart cycle adds AT
MOST one attempt's `charged_ceiling` to the ledger before any FURTHER
dispatch decision is made, no matter how many cycles occur. (Residual,
disclosed, and unfixable without external metering: if the box is torn
down and NEVER restarts again while an attempt is open, the on-disk
ledger under-reports true spend by up to one `charged_ceiling`
forever — G1 guarantees the gap closes BEFORE the next gate check, not
that a process which never checks again leaves a perfectly accurate
historical record. This is the same declarative-enforcement limit §4's
own "nothing external enforces this ceiling" finding already
discloses, not a new one. §R5 H1 additionally closes the ONE path that
previously escaped even this bound — a ledger left unparseable by a
truncated write — via atomicity (no truncated write is observable) and
step 0's disk reconstruction (an unparseable ledger from any other
cause is rebuilt conservatively, never treated as fresh, §R5 H1
above); the residual box-never-restarts case is unaffected, since no
recovery procedure of any kind can run if nothing ever runs it again.)

**G2 — canonical-path harvest contract (Rev 4, closes KW5.2: the
attempt-indexed outdirs (F1/Rev 3) broke `harvest()`'s flat,
non-recursive glob outright, and the natural "fix" — a recursive glob —
would have silently corrupted the fixed-denominator-4 guard on any
retried seed instead).** On attempt ACCEPTANCE — i.e. the instant a
subprocess's cell JSON reads `status=="COMPLETED"` — the orchestrator
COPIES that JSON from its archival attempt directory
(`.../K{K}_s{seed}_attempt{n}/earlyln_K{K}_s{seed}.json`) to the
CANONICAL FLAT PATH `discover_seeds_by_K`/`harvest()` already
non-recursively glob, unmodified: `.../earlyln_K{K}_s{seed}.json`,
one file per (K, seed), ever. Before copying, the orchestrator checks
`os.path.exists(canonical_path)`; if it already exists, the copy
ABORTS LOUDLY (raises) instead of overwriting. **This exists-check fires
only on a genuine invariant violation, once more (§R7 J3 — closes
KW8.3's second face, a REGRESSION §R6 introduced and this revision
corrects, not a residual of the original design).** The dispatch loop
above only ever advances a cell to attempt 2 from `ABORTED-BUDGET-1`/
`CRASHED-1`, never from `COMPLETED`, so no cell can produce two
`COMPLETED` attempts in NORMAL (live, non-reconstructed) operation.
**Reconstruction was a SECOND, independent producer of the same trip
under Rev 6's 0.2 guard:** a cell holding a `COMPLETED` canonical file
but a non-empty, non-`COMPLETED` 0.1 row set (e.g. attempt 1
`ABORTED-BUDGET`, attempt 2 dir absent) failed Rev 6's "0.1 appended
ZERO rows" guard, left the canonical file unaccounted for in the
ledger, derived NON-TERMINAL, and was re-dispatched by step 3 — the
redispatched attempt's own eventual `COMPLETED` JSON then tripped this
exists-check for real, mid-run, on a cell whose science already
existed (KW8.3). §R7 J3 widens 0.2's guard to "no `COMPLETED` row
appended" (above), which the corrected 200-state composition sweep
(embedded above) confirms drives this producer to **0/200** — so "an
operator re-running the orchestrator against a dirty, pre-existing
results directory" is, once again, the ONLY way to trip it, this time
verified by direct execution rather than assumed by inspection.

**The exact hazard this closes (KW5.2's duplicate-seed scenario).**
The audit found that switching `discover_seeds_by_K` to a RECURSIVE
glob over the attempt-indexed tree — the obvious fix for the flat-glob
break — would let two files sharing the SAME basename in different
`attempt{n}/` directories both count toward `n_seeds` (no dedupe in
`tuple(sorted(seeds_by_K[K]))`), silently inflating a retried K's
denominator to 5 and breaking the A4.9 fixed-denominator-4 guard three
places rely on. G2 makes this unreachable by construction, not by
adding a dedupe step: the directory `harvest()` reads is FLAT, `glob`
stays non-recursive (unchanged code), and the exists-check guarantees
AT MOST ONE file per (K, seed) is ever present there — there is no
second file for a duplicate-basename bug to find, whether or not a
future build implementer is ever tempted to "fix" discovery with
recursion (which this design also makes unnecessary, since the attempt
dirs are never read by `harvest()` at all, per the archive-only
convention above).

**Harvest contract, stated explicitly: `harvest()` reads canonical
paths ONLY, never attempt-indexed subdirectories, and the denominator
stays exactly 4** (A4.9 guard, unaffected — G2 changes WHERE a file
lands, never how many count toward `n_seeds`).

**A bonus simplification this creates for D5/E4's "Enforcement point"
build instruction below, TRUE BY THE PAIR (§R5 H2 — corrects Rev 4's
one-directional claim, KW6.2):** the identity `discover_seeds_by_K`'s
glob-presence count over the CANONICAL directory == a status-based
`n_completed` count requires BOTH directions, and Rev 5 is the first
revision where both are actually established:
- **canonical ⇒ `COMPLETED`** (G2, unchanged since Rev 4): a
  canonical-path file is written ONLY on `COMPLETED` acceptance, so no
  `ABORTED-BUDGET`/`CRASHED`/`PERSISTENTLY-ABORTED`/`GATE-REFUSED` cell
  can ever produce one to be miscounted.
- **`COMPLETED` ⇒ canonical** (§R5 H2, NEW this revision): the
  copy-then-fold ordering above means a `COMPLETED` row can only ever
  be written — by the normal fold OR by the recovery-closure branch —
  AFTER its canonical file is confirmed on disk. A crash between copy
  and fold no longer produces a `COMPLETED` row with no canonical
  file (the recovery-closure branch checks for exactly this and
  writes the row only once the file is confirmed present, above); no
  other code path writes a `COMPLETED` row at all.

With both directions holding, the identity is exact, not
approximate-in-one-direction, and the `harvest()` code-patch D5/E4
previously specified (file-presence → status-based) remains NO LONGER
NEEDED — subsumed by G2+H2's copy-then-fold discipline TOGETHER, not
by G2 alone as Rev 4 claimed. (Confirmed against the actual code,
unchanged this revision: `harvest()`'s non-recursive glob at
`ncr_earlyln_scale.py:358-380` is exactly the mechanism this relies on.)

**Crash-window walk (§R5 H2 — the four windows a crash on the
`COMPLETED` path can land in, and each one's recovery outcome. §R6
NOTE: the first two rows are UPDATED this revision — §R6 I2 (recovery
procedure step 2.1, above) now reads the attempt-dir JSON BEFORE
declaring a genuine crash, so these two windows no longer discard
completed science; see §R6 for why this forced contact with an
otherwise-settled table):**

| Window | On-disk state at crash | Recovery outcome |
|---|---|---|
| Before copy starts | attempt-dir JSON is `COMPLETED`; no canonical file, no `.tmp`; `open_attempt` dangling | **§R6 I2:** step 2.1 reads the attempt-dir JSON first, finds `COMPLETED`, PROMOTES it (copy+atomic-rename) ⇒ `COMPLETED` row, MEASURED `elapsed_h` (§R9 m5 — closes KW10.6: `rec["elapsed_s"]/3600`, the TOP-LEVEL field, `:302`, never the nested `rec["train"]["elapsed_s"]` at `:202`) — the completed science IS reused, never re-run from scratch (closes KW7.2) |
| Mid-copy (`.tmp` being written, not yet renamed) | identical observable state to "before copy" — the `.tmp` is unmatched by `discover_seeds_by_K`'s glob | identical outcome: §R6 I2's promotion fires the same way — `COMPLETED` row, measured `elapsed_h` |
| Between copy and fold (`os.replace` completed; fold has not run) | canonical file EXISTS with `status=="COMPLETED"`; `open_attempt` still dangling | canonical file + dangling record ⇒ PROOF OF COMPLETION (the fixed case): `COMPLETED` row written, full ceiling charged, `COMPLETED ⇒ canonical` restored |
| After fold | canonical file exists; `COMPLETED` terminal row exists; `open_attempt` is `null` | normal — nothing dangling, no recovery action |

**Trigger evaluation point + harvest invocations, exact.**
1. Once all 12 primary cells are terminal, `harvest()` runs ONCE over
   `/home/nvidia/ncr/results_kwall_characterization/` (the canonical
   directory, G2 above) to compute each K's resolution state (table
   above) and, from it, both the §5 `classify()` band candidates and
   the §4 trigger resolution (now gated by G5's DECIDED-band
   precondition, above). This is the ONLY point the trigger is
   evaluated.
2. If a conditional arm is dispatched, its 4 cells run (same state
   machine, `--steps 160000 --ceiling-gpuh 2.32`,
   `/home/nvidia/ncr/results_kwall_characterization_160k/`, its own
   canonical directory under the SAME G2 contract), then the
   orchestrator makes a SECOND, SEPARATE `harvest()` call against that
   directory's own canonical files and merges the two independent
   results into one report (§5's 160K qualifier band). This is a
   second CALL to the single-`outdir` `harvest(outdir, seeds_by_K)`
   function, not a single call spanning both trees — the two arms'
   results never share a directory.
3. If no conditional arm is dispatched (`K_trig==32`'s $0 branch, or
   `TRIGGER-UNRESOLVED` — now including the G5 band-blocked case), the
   second `harvest()` call is simply skipped (over the primary results
   only, plus — for `K_trig==32` — the already-archived §3 table cited
   at $0 cost) so the final report always has the same shape regardless
   of branch.

**Primary/conditional canonical-directory disjointness (§R7 J5 — part
of KW8.5's discharge).** The primary canonical directory
(`/home/nvidia/ncr/results_kwall_characterization/`, G2 above) and the
conditional canonical directory
(`/home/nvidia/ncr/results_kwall_characterization_160k/`, cited just
above) are DISJOINT sibling directories under the same
`/home/nvidia/ncr/` root — neither is a prefix or subdirectory of the
other, and `harvest()` is always called with exactly one of the two as
its `outdir` argument (never a call spanning both, per "Trigger
evaluation point + harvest invocations," above), so a `COMPLETED` file
copied into one tree by G2's exists-check can never collide with, or be
mistaken for, a file in the other. This is a static, build-time
invariant, not a per-report runtime check; **the build asserts it
directly (§R8 K7/KW9.9 — corrects a dangling in-document reference:
this cites item (d) of the BINDING BUILD CHARTER, which lives in the
attack reports, not in this design's own §1–§7/§A/§R sections — this
design has no §9 of its own. The citation is `NCR_KWALL_ATTACK_R7.md`
§9, item 4(d), restated unchanged as item 4 of
`NCR_KWALL_ATTACK_R8.md` §8) before the first conditional dispatch**
— `validity_check`'s new universal assertion 7 (below) checks per-report
disk EVIDENCE for a claimed `qualifier_band`, which is a different,
complementary guarantee (evidence exists) from this one (the two trees
cannot be confused with each other).

**Output JSON (`orchestrator_report.json`), required fields (Rev 4:
`run_status` enum exhaustive (G4); `attempts[].status` enum exhaustive
per G1/G3's actual reachable states (KW5.3); `open_attempt` exposed for
transparency; `trigger.band_blocked_K_trig` discloses G5's suppressed
cases; `band.incomplete_at_K`/`candidate_bands` and a `smoke` block
close KW5.9's missing fields; `gpu_id`/`git_commit` added, also KW5.9.
Rev 5, §R5 H4/H5: `run_status` gains
`EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`; every `attempts[]` row gains
`d_override`/`ceiling_charged`; `charged_vs_measured` and
`stop_file_path` are new top-level fields, closing KW6.4/KW6.8/KW6.13
against the actual schema below — this table is the SINGLE SOURCE OF
TRUTH for both enums; it supersedes any other enumeration anywhere
else in this document, live or historical, per the precedence sentence
in the unified-enum table below):**

```
{
  "run_status": "COMPLETE" | "COMPLETE-DEGRADED" |
    "STOPPED-BY-OPERATOR" | "EXHAUSTED-BUDGET" |
    "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE",
  "ledger": {
    "realized_gpu_h_final": <float>, "hard_gate_cap": 15.00,
    "retry_gate_threshold": 12.00, "declared_pool_ceiling": 15.50,
    "open_attempt": null,
    "attempts": [ {"K":int,"seed":int,"arm":"primary"|"conditional",
      "attempt_n":int (1 or 2 for a LIVE-DISPATCHED attempt; a
        RECONSTRUCTION BOOTSTRAP row, §R6 I1/§R7 J3, may carry a value
        >2 as a reconstruction LABEL, never a future dispatch number —
        §R8 K6, closes KW9.6's MAJOR: 72/200 reconstructed compositions
        legitimately emit `attempt_n=3`, which the old `1|2` schema
        forbade outright),"elapsed_h":float,
      "status":"COMPLETED"|"ABORTED-BUDGET"|"CRASHED"|
        "CRASHED-RECOVERED"|"GATE-REFUSED"|"STOPPED-BY-OPERATOR",
      "outdir":str|null, "d_override":int,
      "ceiling_charged":bool}, ... ] },
  "charged_vs_measured": {"measured_gpu_h":float,
    "ceiling_charged_gpu_h":float, "ceiling_charged_fraction":float},
  "stop_file_path": str|null,
  "smoke": {"K26":"PASS"|"FAIL", "K28":"PASS"|"FAIL",
    "K30":"PASS"|"FAIL"},
  "primary": { "per_K": { "26": {...}, "28": {...}, "30": {...} } },
  "trigger": { "resolution": "unanimous"|"tie-break-min"|
    "TRIGGER-UNRESOLVED", "resolution_detail": str|null, "K_trig": int|null,
    "candidate_set": [int,...]|null, "blocking_K": int|null,
    "band_blocked_K_trig": int|null },
  "conditional": {"launched": bool, "per_seed": [...],
    "qualifier_band": str|null} | null,
  "band": {"label": str, "non_monotone_tag": bool,
    "interval_resolved_Ks": [int,...],
    "incomplete_at_K": [int,...]|null,
    "candidate_bands": [str,...]|null},
  "wall_clock": {"start": <iso8601>, "end": <iso8601>},
  "gpu_id": int, "git_commit": str,
  "n_cells_attempted": int, "n_attempts_total": int
}
```

**Unified `attempts[].status` × `run_status` enum table (§R5 H4 — the
SINGLE SOURCE OF TRUTH). Precedence: this table supersedes the
JSON-schema block immediately above wherever they might ever drift
(they should not — both are edited together), and it supersedes the
enumerations in the FROZEN historical sections §R4 (KW5.3's row and the
"numbers that moved" line — cited by NAME only, not by line number,
per §R6 KW7.8: Rev 5 shifted §R4 by +472 lines, so a numeric pointer
written today goes stale the next time anything ABOVE §R4 is revised,
exactly as happened to the Rev-5 pointers this fixes) and any frozen
`§A`-section text — those stay byte-identical as historical record
(house convention) and are simply outranked, never edited, by this
table when the two disagree (§R5 KW6.5/KW6.9 below). **The same
outranking covers arithmetic, not only enumerations (§R6 KW7.12):
§R4's KW5.1 discharge row still carries the refuted "the crash case is
actually TIGHTER … than the completing case" claim; it is outranked,
on the identical footing, by §4's "True spend, worst case" derivation
(`T ≤ 15.3737h`, §R7 J6's two-class re-derivation, above).**

*`attempts[].status` (6 reachable values; `PERSISTENTLY-ABORTED` is
correctly ABSENT — it is a derived CELL state, never an attempt
status, §R5 KW6.5(ii)):*

| Value | Reachable via | Typical `attempt_n` |
|---|---|---|
| `COMPLETED` | classify-copy-fold, exit code 0 or non-zero AFTER a `COMPLETED` JSON is on disk; OR recovery's canonical-file+dangling-record branch (§R5 H2, "between copy and fold"); OR recovery's attempt-JSON promotion branch (§R6 I2, "before copy"/"mid-copy"); OR ledger reconstruction's per-attempt-directory table / bootstrap fallback (§R6 I1) | 1 or 2; **>2 possible via a 0.2 OK-bootstrap row (§R8 K6/KW9.6 — a RECONSTRUCTION LABEL, not a dispatch number)** |
| `ABORTED-BUDGET` | exit 0, `status=="ABORTED-BUDGET"` JSON on disk | 1 or 2 |
| `CRASHED` | any non-zero exit (or exit 0 with no JSON, §R5 KW6.5(iii)) with no `COMPLETED`/`ABORTED-BUDGET` JSON | 1 or 2 |
| `CRASHED-RECOVERED` | recovery's genuine-crash branch (dangling `open_attempt`, no canonical file, §R5 H2); OR ledger reconstruction's per-attempt-directory table, unparseable/absent-JSON rows (§R6 I1, written at that directory's OWN `attempt_n` — no longer forced to `2`) | 1 or 2; **>2 possible via a 0.2 CORRUPT-bootstrap row (§R8 K6/KW9.6, same reconstruction-label reasoning)** |
| `GATE-REFUSED` | HARD or RETRY gate refusal, pre-dispatch — no subprocess ever runs (§R5 KW6.5(i): now DOES produce a row, `elapsed_h=0.0`, `outdir=null`) | 1 or 2 |
| `STOPPED-BY-OPERATOR` | exit code 3 (`--stop-file` sentinel, checked training-only, strictly before any JSON write) | 1 or 2 |

*Derived CELL state (never an `attempts[].status` value):*
**`COMPLETED` takes precedence (§R7 KW8.7 — the derivation below is
otherwise not a function: a cell whose attempt-1 row is `COMPLETED`
and whose attempt-2 row exists and is not, e.g. attempt 1
parseable-`COMPLETED` / attempt 2 unparseable, satisfies BOTH clauses
below simultaneously, 24/200 compositions; no dispatch hazard either
way since both are terminal and skipped in full — but a spurious
`PERSISTENTLY-ABORTED` reading could otherwise satisfy
`COMPLETE-DEGRADED`'s throttle-evidence clause with nothing actually
throttled). A cell with ANY `COMPLETED` row is `COMPLETED`, full stop —
the clause below is evaluated only once that is ruled out:*
`PERSISTENTLY-ABORTED` iff (no row is `COMPLETED`, and) the cell's
attempt-2 row exists and is non-`COMPLETED`, OR its attempt-1 row is
non-`COMPLETED`, no attempt-2 row exists, and the retry gate closed it
(§R5 KW6.5(ii)).

*Exit-code × on-disk-JSON cross-product (9 cells; verified against the
real code, §7 INTEGRITY below and the R5 audit's own code reads).
`GATE-REFUSED` is ORTHOGONAL to this table — it occurs pre-dispatch,
before any exit code or JSON can exist:*

| exit code \ on-disk JSON | no JSON | `COMPLETED` JSON | `ABORTED-BUDGET` JSON |
|---|---|---|---|
| `0` | `CRASHED` (§R5 KW6.5(iii), the NEW default arm) | `COMPLETED` | `ABORTED-BUDGET` |
| `3` (`sys.exit(3)`, `ncr_earlyln_scale.py:196-197`) | `STOPPED-BY-OPERATOR` | **UNREACHABLE** — `sys.exit(3)` fires strictly before any JSON write (`:196-197` precedes both `:262-266` and `:307`) | **UNREACHABLE**, same reason |
| other non-zero | `CRASHED` | `COMPLETED` (an already-written JSON is authoritative over a later non-zero exit — the science is done) | `ABORTED-BUDGET` (same) |

*`run_status` (5 values — adds `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`,
§R5 H5; corrects `COMPLETE`/`COMPLETE-DEGRADED` for mutual exclusivity
and exhaustiveness, §R5 KW6.6; each carries its own disk-evidence
assertion, §R5 H4/KW6.7 — full definitions and assertions below):*
`COMPLETE`, `COMPLETE-DEGRADED`, `STOPPED-BY-OPERATOR`,
`EXHAUSTED-BUDGET`, `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`.

**`run_status` enum, defined exhaustively (G4, Rev 4 — closes KW5.4,
where the schema offered two undefined values and the job spec's own
`validity_check` rejected the design's own pre-registered degraded
outcome. Rev 5, §R5 H4/H5: `COMPLETE`/`COMPLETE-DEGRADED` corrected for
mutual exclusivity and exhaustiveness — KW6.6 — and a fifth value,
`EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`, added — KW6.8 — each value below
now also states the disk-evidence `validity_check` requires of it,
KW6.4/KW6.7).**
- **`COMPLETE`.** Defined SOLELY as: no budget-caused refusal of any
  dispatch — first attempt OR retry — anywhere in the run (§R5
  KW6.6(i) deletes "no `GATE-REFUSED` anywhere in the run" as a
  SEPARATE criterion; a `PERSISTENTLY-ABORTED` cell whose refusal came
  from a denied RETRY, not a `GATE-REFUSED` first attempt, is a
  budget-caused refusal too, and this single criterion now catches it
  without a second, disagreeing test). The trigger was evaluated, and
  — if it fired — every conditional cell's first attempt and retry
  were likewise free of budget-caused refusal. **Disk-evidence
  assertion, quantified over the full outcome space (§R6 I4 — replaces
  §R5 H4's unconditional 12-canonical count, which KW7.4 showed
  rejects every legitimately-reportable `INCOMPLETE-AT-K` run even
  though a non-budget cause, e.g. a twice-`CRASHED` seed, is fully
  compatible with `COMPLETE`):**
  - If `band.interval_resolved_Ks` is empty AND `band.incomplete_at_K`
    is `null` (no K is incomplete): `COMPLETE` ⇔ exactly 12 canonical
    PRIMARY files exist, each `status=="COMPLETED"` — the original,
    unchanged 12-canonical assertion, still the ONLY check when nothing
    is disclosed as incomplete.
  - Otherwise (some K IS disclosed incomplete): `COMPLETE` ⇔ for every
    primary `K∈{26,28,30}` named in `band.interval_resolved_Ks` or
    `band.incomplete_at_K`, that K's canonical primary count is `<4`,
    AND every OTHER primary K's canonical count is exactly `4` — the
    per-K canonical counts are consistent with what the report's own
    `band` object discloses, never a bare magic number — **AND (§R7
    J1 — closes KW8.1's FATAL, a REGRESSION that reopened the no-op
    hole `§A6-ADJUDICATION` had named SETTLED: unlike
    `COMPLETE-DEGRADED`'s own evidence clause below, the three checks
    just above are satisfiable by an EMPTY ledger and an empty
    filesystem — a report disclosing incompleteness at every primary K
    vacuously satisfies "consistent with `band`" when every disclosed
    count reads `0<4`) every primary `(K, seed)` pair, `K∈{26,28,30}`,
    `seed∈{0,1,2,3}`, has ≥1 row in `ledger.attempts`; AND the primary
    canonical count equals `len({(a["K"],a["seed"]) for a in
    ledger.attempts if a["arm"]=="primary" and
    a["status"]=="COMPLETED"})`** — the same two positive-evidence
    clauses `COMPLETE-DEGRADED` already carries (below), now required
    here too: a `COMPLETE` verdict must be backed by a per-cell attempt
    record, never merely a self-reported `band` object. **AND (§R8 K2 —
    closes KW9.2's MAJOR: the two clauses above are evidence of a ROW,
    not evidence of WORK — satisfiable by 12 zero-cost `GATE-REFUSED`
    rows at `elapsed_h=0.0`, one per primary pair, with nothing ever
    dispatched) `len({(a["K"],a["seed"]) for a in ledger.attempts if
    a["arm"]=="primary" and a["status"]=="COMPLETED"}) >= 1`** — a
    `COMPLETE` verdict must show that SOMETHING primary actually
    completed; a run whose every primary row is a refusal is not
    `COMPLETE` under any disclosed `band`.
  In both cases: equivalently, the canonical count for a fully-complete
  K equals the number of that K's `(K,seed)` pairs carrying a
  `COMPLETED` row in `ledger.attempts` (unique by G2's exists-check +
  §R5 H2's `COMPLETED ⇒ canonical` invariant, so counting rows and
  counting canonical files agree).
- **`COMPLETE-DEGRADED`.** Every primary cell got its first attempt
  (the hard gate never refused a PRIMARY cell's first attempt — the
  12-cell baseline sweep completed), but the hard/retry gates
  throttled something downstream of that baseline for budget reasons
  alone. THREE pre-registered sub-cases now (§R5 KW6.6(ii) adds the
  third — the enumeration was not exhaustive before), all disclosed
  via `attempts[]` and none treated as a bug: (i) *primary-retry-
  refused* — a primary cell's attempt-2 retry was denied by the HARD
  or RETRY gate rather than the state machine reaching a natural
  `PERSISTENTLY-ABORTED` after both attempts ran; that cell still
  follows D5/E4's interval logic exactly as any other incomplete cell
  — hence FEWER than 12 canonical primaries is the NORMAL disk state
  for this sub-case, not an anomaly (§R6 I4/KW7.4 — see the corrected
  assertion below, which this sub-case could never satisfy under §R5
  H4's unconditional 12-canonical wording). (ii) *conditional-
  throttled* — the trigger fired (`DECIDED`, and the G5 band
  precondition held), but 1-4 of the conditional arm's 4 cells' FIRST
  attempts were refused by the hard gate before the 15.00 cap was
  reached. (iii) **`conditional-retry-refused` (NEW, §R5 KW6.6(ii))**
  — the trigger fired and every conditional cell got its first
  attempt, but a conditional cell's RETRY was refused by the HARD or
  RETRY gate — the case neither (i) (scoped to primary) nor (ii)
  (scoped to a conditional FIRST attempt) covers, which the parent
  sentence already implied but the enumeration omitted. Any
  budget-caused throttle strictly downstream of the completed 12-cell
  primary baseline belongs to this label, whether or not it fits one
  of (i)-(iii) by name. **Disk-evidence assertion, corrected to the
  count it actually implies (§R6 I4 — replaces §R5 H4's "same
  12-canonical-primaries condition as `COMPLETE`", which sub-case (i)
  can never satisfy by its own construction — KW7.4's FATAL):
  `COMPLETE-DEGRADED` ⇔ every primary `(K, seed)` pair has SOME
  terminal disposition in `ledger.attempts` (the no-op's hole: zero
  rows for all 12 cells fails this even though 0 canonical == 0
  `COMPLETED` rows would otherwise trivially match), AND the number of
  canonical primary files equals the number of distinct primary
  `(K,seed)` pairs carrying a `COMPLETED` row in `ledger.attempts`
  (the identity G2+H2 guarantee — §5's interval logic is free to leave
  this below 12), AND at least one `GATE-REFUSED` or `PERSISTENTLY-
  ABORTED`-deriving row exists somewhere in `ledger.attempts`** (the
  positive evidence that a throttle actually occurred, distinguishing
  this from `COMPLETE`). **AND (§R8 K2 — mirrors the `COMPLETE` fix
  above, closing the same KW9.2 hole this label shares with it: a
  12-row ledger of pure `GATE-REFUSED` refusals at `elapsed_h=0.0`
  otherwise satisfies every clause above with nothing ever dispatched)
  `len({(a["K"],a["seed"]) for a in ledger.attempts if
  a["arm"]=="primary" and a["status"]=="COMPLETED"}) >= 1`** — this
  label's own prose definition already says the 12-cell primary
  baseline "completed" (sub-case (i) leaves 11 canonical, sub-cases
  (ii)/(iii) leave 12; both trivially clear `>=1`), so this clause only
  rejects the disk states the prose never licensed in the first place.
- **`STOPPED-BY-OPERATOR`.** The `--stop-file` sentinel was seen (G3).
  Terminal for the whole run at whatever point it occurred; never a
  gate refusal, never retried. **Evidence, relocated (§R6 I6/KW7.11 —
  `validity_check`'s universal assertion 1, below, excludes this label
  from the accept-set BEFORE any per-`run_status` branch ever runs, so
  a disk-evidence assertion written as a `validity_check` branch for
  this label is unreachable code; the stop-file marker check —
  `report["stop_file_path"] is not None and
  os.path.exists(report["stop_file_path"])` — is instead the
  orchestrator's OWN pre-write self-check, asserted before
  `orchestrator_report.json` is ever written with this label, where it
  can actually fire and prevent a false report from existing at all).**
- **`EXHAUSTED-BUDGET`.** The hard gate refused a PRIMARY cell's OWN
  FIRST ATTEMPT — the 12-cell baseline itself could not be completed
  inside the ceiling. Reachable in principle, not vacuous: 12 primary
  first attempts at the shared `1.20h` ceiling sum to `14.40h`, inside
  `15.00` only if every one is admitted with essentially no headroom
  left for anything else. More severe than `COMPLETE-DEGRADED`: most
  or all of §5's K's will read `INCOMPLETE-AT-K`. **Disk-evidence
  assertion, now WITH its negative half (§R6 I5 — closes KW7.5: §R5
  H4's positive-only clause let a report claiming `EXHAUSTED-BUDGET`
  alongside a COMPLETE 12-cell baseline pass, a label its own disk
  evidence refutes):**
  `ledger.realized_gpu_h_final > 13.80` (`= 15.00 − 1.20`, the exact
  threshold at which even one more primary-ceiling admission is
  impossible — ledger-EVIDENCED spend, not merely a claimed label, and
  what the no-op JSON below fails) **AND fewer than 12 canonical
  PRIMARY files exist AND at least one primary-arm, first-attempt
  (`attempt_n==1`) `GATE-REFUSED` row exists in `ledger.attempts`**
  (positive evidence the 12-cell baseline was itself refused, exactly
  what the label claims — a 12-canonical-primaries disk state can
  never satisfy this alongside the negative clause, so `EXHAUSTED-
  BUDGET` and `COMPLETE`/`COMPLETE-DEGRADED` are now provably disjoint
  on disk, not merely by prose). **PLUS (§R7 J4 — closes KW8.4's
  MAJOR: this label's clause set was a strict SUBSET of
  `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`'s, below, so a suspect run — an
  `EXHAUSTED-BUDGET` disk state that is mostly `CRASHED-RECOVERED`/
  reconstructed noise, exactly the case the disclosure paragraph just
  below names — could evade `-SUSPECT-OVERCHARGE`'s binding
  resubmission protection by mislabelling itself plain
  `EXHAUSTED-BUDGET` instead, which the OLD clause set could not
  distinguish from a genuine one):** `charged_vs_measured.
  ceiling_charged_fraction <= 0.50` (the fraction is defined just
  below; **this makes the two labels a DICHOTOMY, not merely
  disjoint-by-prose: for any ledger satisfying the shared `>13.80`/
  `<12`/`GATE-REFUSED` base clauses, `ceiling_charged_fraction` is a
  single real number that is EITHER `≤0.50` (this label) OR `>0.50`
  (the next label) — never both, never neither, so exactly one of the
  two `EXHAUSTED-BUDGET*` labels can ever be the CORRECT claim for a
  given ledger**, closing the mislabelling escape by construction, not
  by disclosure alone). **Disclosure (§R5 H5, KW6.8 — an
  `EXHAUSTED-BUDGET` verdict is not always a genuine budget result):**
  repeated crash→restart cycles that each reach the dispatch point
  before dying burn a full ceiling of LEDGER charge for ≈0 GPU-h of
  real compute (CLAUDE.md's own mandated supervisor loop can produce
  exactly this against a systematic mid-attempt kill). An
  `EXHAUSTED-BUDGET` verdict whose `attempts[]` is dominated by
  `CRASHED-RECOVERED`/reconstructed rows indicates an ENVIRONMENT
  FAULT, not a budget result — see `charged_vs_measured` and the
  `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE` value, next (§R7 J4's fraction
  clause is what now ROUTES such a ledger to the correct label instead
  of leaving both technically satisfiable).
- **`EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE` (NEW, §R5 H5 — KW6.8's
  operator escape hatch).** The orchestrator computes
  `charged_vs_measured.ceiling_charged_fraction =
  ceiling_charged_gpu_h / realized_gpu_h_final` (`ceiling_charged_gpu_h`
  = the sum of `elapsed_h` over every `attempts[]` row with
  `ceiling_charged==true` — every `CRASHED-RECOVERED` row and every
  row produced by §R5 H1's disk reconstruction, regardless of its
  `status`). Whenever an `EXHAUSTED-BUDGET` verdict (above) would
  otherwise fire AND `ceiling_charged_fraction > 0.50`, the
  orchestrator reports THIS value instead. **Same disk-evidence
  assertion as `EXHAUSTED-BUDGET`, negative half included (§R6 I5:
  `realized_gpu_h_final > 13.80` AND fewer than 12 canonical PRIMARY
  files AND ≥1 primary-arm first-attempt `GATE-REFUSED` row), PLUS
  `charged_vs_measured.ceiling_charged_fraction > 0.50`** (§R7 J4 — the
  mirror half of the dichotomy above: this label's own clause set now
  differs from plain `EXHAUSTED-BUDGET`'s by exactly one strict
  inequality on the same shared quantity, `>0.50` here vs `≤0.50`
  there, so the two are exhaustive AND mutually exclusive over the
  shared base disk state, never merely "the same three clauses PLUS
  one more" as Rev 6 had it, which let plain `EXHAUSTED-BUDGET` also
  satisfy this label's disk state by omission).
  Reportable, and routed to `completed/` by `validity_check` (below) —
  it is a legitimate, disclosed terminal state, not a bug — but
  **resubmission is NEVER automatic: only an explicit coordinator
  adjudication, with a fresh ledger, may re-run the affected cells.**
  The pool's ordinary "resubmitting resumes cleanly" advice
  (`STOPPED-BY-OPERATOR`'s own paragraph, below) does NOT apply here —
  resuming this ledger as-is would re-gate every cell against a budget
  that is mostly environment-fault noise, not real spend, which is
  precisely the failure KW6.8 found and this value exists to flag
  instead of hiding.

All FIVE are DISCLOSED, TERMINATING, non-buggy outcomes of the
orchestrator behaving exactly as designed (finishing, being told to
stop, protecting its own ceiling, or flagging a suspect ceiling) —
none indicates a code defect. The job-spec `validity_check` (below)
accepts the four that represent the orchestrator completing its own
logic (`COMPLETE`/`COMPLETE-DEGRADED`/`EXHAUSTED-BUDGET`/
`EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`) as `completed/`, never
`failed/`. `STOPPED-BY-OPERATOR` is the one exception: a deliberate
human action to interrupt the run is not itself a completed-or-
gracefully-degraded RESULT of this design's own logic, so it stays
outside the accept-set; resubmitting the job resumes cleanly (G1's
cell-level resume rule skips every already-terminal cell) at no cost
beyond the ceiling already spent.

**Worst-case bound, derived for the sequential model — the
parallel-batch hole no longer exists, and (Rev 4, G1) the mid-attempt-
death hole no longer exists either.** Rev 2's induction broke (KW4.2)
because `realized_before_last_batch` was read from `COMPLETED`-only
JSONs on disk at admission time, and nothing prevented MULTIPLE
dispatches from reading the SAME stale value before any of them
finished — the "batch" was undefined and could be arbitrarily large
(the audit's own abort-free counterexample reached 20.92h under the
most natural reading of the old trigger wording). **That hole is
structural to CONCURRENT dispatch — it does not exist when dispatch
cannot be concurrent.** F1 removes concurrency itself: one GPU, one
subprocess in flight at a time, ledger updated synchronously in the
SAME process immediately after each attempt returns and BEFORE the
next gate check runs. There is no "batch" to be atomic about, and no
reservation/release bookkeeping is needed (what a PARALLEL Option B
would have required) — every gate check sees the exact true cumulative
spend of every prior attempt, aborted or completed. **What Rev 3's
version of this argument did NOT cover (KW5.1): an orchestrator that
DIES before an attempt returns never executes the "immediately after
each attempt returns" ledger update at all — the true cumulative spend
argument silently assumed every dispatched attempt eventually returns.
G1's write-ahead record + recovery procedure (above) closes exactly
this gap, and the induction below is extended to cover it.**

Induction: define `R_k` as `ledger.realized_gpu_h` immediately before
the `k`-th gate check ever performed, across the WHOLE run INCLUDING
every restart. Claim: `R_k` equals the sum, over every attempt
strictly prior to the `k`-th, of EITHER its true measured `elapsed_h`
(if it returned normally) OR its `charged_ceiling` (if it was
recovered as `CRASHED-RECOVERED`, G1) — exact-or-conservative in every
case, never an omission. This holds by induction: a normal return
updates the ledger, in-process, before the next gate check (the
sequencing argument above, unchanged); a recovery-closure ALSO updates
the ledger, before the cell-order walk resumes and therefore before
the next gate check (G1's recovery procedure, above) — so BOTH kinds
of "prior attempt" are folded into `R_k` before it is ever read for a
gate decision, regardless of how many crash/restart cycles happened in
between. **This is what actually answers KW5.1's "nothing caps the
number of cycles" finding: the CYCLE COUNT no longer matters, because
every cycle's dangling attempt is priced into the ledger before the
NEXT cycle's first gate check, so the induction's premise survives any
number of restarts.**

Let attempt `N` be the LAST attempt ever admitted before the program
ends. By the hard gate, `R_{N-1} + ceiling(N) ≤ 15.00`. Two cases:
- **It returns** (`COMPLETED`, `ABORTED-BUDGET`, or `CRASHED` — any
  exit letting `subprocess.run` return): its true `elapsed_h` can
  exceed `ceiling(N)` by at most the single-attempt tail —
  **correcting KW5.7: BOTH the eval-overhead term (max observed
  0.0126 GPU-h, KW3.9's figure) and the `log_every=500`-step
  training-ceiling-check granularity term (KW4.9 — max observed
  0.0031 GPU-h) can apply to the SAME attempt**, not "only one of the
  two" as Rev 3 claimed: a run whose last check before `ceiling_s`
  passes at step `S` proceeds 500 more steps past it, exits the loop
  `COMPLETED`, and then still runs its full (unbounded) eval phase.
  The tail is their SUM: `0.0126+0.0031=0.0157` GPU-h.
- **It never returns** (the orchestrator itself dies): the LEDGER
  value `R_N`, as computed by the NEXT restart's recovery step, is
  `R_{N-1} + charged_ceiling(N) ≤ 15.00` EXACTLY — no tail term at
  all, because the recovery charge is the gate-admitted ceiling value,
  not a measurement. **This bounds the LEDGER, not TRUE GPU-hours
  consumed (§R5 H3 — deletes the false "this is TIGHTER than the
  return case" claim this replaced, KW6.3): the two diverge whenever a
  recovered attempt's true elapsed time exceeded its charged ceiling
  before the crash, which the ledger — having died with the process
  that would have measured it — has no way to observe or bound at the
  recovery step.**

```
R_N  ≤  15.00 + 0.0157  =  15.0157 GPU-h    (bound on ledger.realized_gpu_h — the HARD GATE's own quantity)
```

**True spend, worst case (§R5 H3 — KW6.3's honest replacement for the
deleted claim above; §R7 NOTE: RE-OPENED this revision — KW8.6 found
the KW7.7 sentence §R6 added here claims a premise I1 does not
establish, and I1 in fact WEAKENS a different premise this derivation
actually needs; the two-class re-derivation below is the fix, J6).**
Let `τ=0.0157` (the combined per-attempt tail derived above), let
`s=0.0053` GPU-h (the STARTUP ALLOWANCE — derivation just below), and
let `T=Σt_i` be the TRUE elapsed GPU-hours across every attempt ever
dispatched, recovered or not. `R_N ≤ 15.00 + τ` (derived above, the
LEDGER's own tight bound, unaffected by anything below — it governs
the LIVE dispatch loop's own timer, which J6 does not touch).

**Deriving `s` (§R7 J6 — closes the false premise, honestly, from the
design's own existing overhead terms; no fresh measurement is possible
from this desk, so this is a bound derived from evidence already on
record, not a new one asserted).** A RECONSTRUCTED `COMPLETED` row's
`elapsed_h` is measured from the SUBPROCESS's own internal timer,
`t0=time.time()` at `ncr_earlyln_scale.py:257` — read directly this
revision: `t0` sits AFTER `os.makedirs(outdir, exist_ok=True)` (`:237`),
the resume-skip check (`:240-245`), `nt.claim_config` (`:247`),
`torch.manual_seed` (`:248`), and `NCREarlyLNModel(...).to(device)`
(`:249` — CUDA init + model build), and after the `rec` dict assembly
including `rn.git_commit()`/`nm.n_params(model)` (`:250-255`). The
LIVE orchestrator's own timer, by contrast, starts at `dispatch_ts`
(`t0` in the dispatch loop, above) — BEFORE `subprocess.run` is even
called — so it captures process spawn, interpreter startup, and every
line up to `:257` that the subprocess's own `elapsed_s` does not. This
is exactly the "third, unpriced term (interpreter/CUDA-init/
model-build startup)" `§R4`'s own frozen KW5.7 disposition already
named and estimated at **"true single-attempt tail ≈0.016–0.021h"**
(frozen table, below). That figure is the TOTAL single-attempt tail
(eval-overhead + log-interval + this startup term); subtracting the
already-priced portion, `τ=0.0157`, isolates the startup-specific
residual: `0.016−0.0157=0.0003` to `0.021−0.0157=0.0053` GPU-h. Taking
the CONSERVATIVE (upper) end of this existing, previously-audited
range: **`s=0.0053` GPU-h (≈19s)** — a plausible order of magnitude for
Python/CUDA/model-build startup on this hardware, and, honestly, an
ESTIMATE inherited from the same source the `0.30h` margin already
relied on for this identical term, not a freshly-proven worst case
(the same caveat KW5.7's own discharge carried; a direct timed
measurement of `run_earlyln_cell`'s own `:237-257` span on the box
would tighten this at build time, per the same-era frozen disposition's
own suggestion — `§R3` below, "Round 4 may tighten it once the build
stage's actual orchestrator-process overhead ... is measured rather
than estimated" — never yet acted on, still open).

Two classes of reconstructed/recovered row now exist, keyed by the
schema's own `ceiling_charged` field (§R7 J6):
- **Class 1 — `ceiling_charged==true`** (every `CRASHED-RECOVERED`
  row, every reconstructed non-`COMPLETED` row post-J3/J6, and the
  live recovery "canonical file + dangling record" branch, above): charged its full
  `charged_ceiling(arm) ≥ 1.20`. `leak_i ≤ τ` per row (unchanged
  argument — the charge is a gate-admitted ceiling, and true elapsed
  can exceed it by at most the single-attempt tail). Every leaking
  attempt charged ≥1.20 of `R_N`'s ≤15.0157 total, so at most
  `⌊15.0157/1.20⌋ = 12` such rows can exist.
- **Class 2 — `ceiling_charged==false`, `status=="COMPLETED"`**
  (0.1's rows 4-6, 0.2's OK bootstrap, and the live recovery
  "before copy"/"mid-copy" PROMOTE branch, above — §R7 J6): charged
  measured `elapsed_h + s`. `leak_i ≤ s` per row under the estimate
  just derived (the allowance is sized to the SAME estimate's upper
  bound, so it covers the term by construction under that estimate —
  the residual honesty is that the estimate itself, not the charging
  rule, is where uncertainty remains). Bounded by the total number of
  attempt slots this design can ever produce: **≤32** (16 cells ×
  2 attempts — 12 primary + 4 conditional — the same loose-but-sound
  structural cap KW8.6's own discharge condition proposed, made
  slightly tighter here by naming its source: a live-folded completed
  attempt has NO leak at all, since the orchestrator's own accurate
  `dispatch_ts`-based timer folded it; only a RECONSTRUCTED completed
  row can leak this term, and reconstruction can, in the worst case —
  a restart forced after every single attempt — touch every one of
  the 32 slots).

Hence:

```
T  ≤  R_N + 12·τ + 32·s
   ≤  15.0157 + 12(0.0157) + 32(0.0053)
   =  15.0157 + 0.1884 + 0.1696
   =  15.3737 GPU-h
```

(equivalently `15.00 + 13(0.0157) + 32(0.0053) = 15.2041 + 0.1696 =
15.3737` — the `15.2041` figure is the Class-1-only bound this
derivation used to stop at; it is NOT the true worst-case bound
post-J6, which is `15.3737`.) `15.3737h` exceeds the tight LEDGER
bound `15.0157h` by `0.3580h` and the disclosed `15.20h` rounding by
`0.1737h`. **Both remain inside the declared `15.50h` pool ceiling**
(margin remaining: `15.50−15.3737=0.1263h`) — no declared ceiling
changes as a result of this correction; see the "Rounding
conservatively" paragraph below, whose own numbers this correction
forces contact with (disclosed in §R7). The
`≤15.00`/`15.0157` figures stay correct as bounds on
`ledger.realized_gpu_h` — the quantity the HARD GATE and
`validity_check`'s `<=15.50` assertion actually read — never as a
bound on true consumed GPU-hours, which is honestly `≤15.3737h`.

**The one term G1 changes:** every term in Rev 3's induction was
already a true measured `elapsed_h`; the ONE new kind of term is a
crash-recovered attempt's `charged_ceiling` substituting for an
`elapsed_h` that no longer exists to measure. Because
`charged_ceiling` is exactly the value the hard gate already verified
fits (`R_{N-1}+ceiling(N)≤15.00`), substituting it can only ever match
or UNDERSTATE what a completing attempt would have contributed at that
same admission — it never inflates `R_N` past the completing-attempt
case above. (Residual, disclosed: if the box is torn down and never
restarts while an attempt is open, the on-disk record under-reports
true spend by up to one `charged_ceiling` forever — this is the same
declarative-enforcement limit already disclosed in the recovery
paragraph above, not a new gap in the bound the GATE enforces.)

**This is still, in essence, one derivation — not `15.00 + 16×0.0126`
as in Rev 2.** Rev 2's `16×` term existed because its induction had to
price EVERY admitted cell's unpriced tail (any of them could, under
concurrent dispatch, have been "the one seen stale"). Under strict
sequencing (Rev 3) plus write-ahead recovery (Rev 4), every attempt
BEFORE the last one is already fully reflected in `R_{N-1}` — either
its own tail was added to the ledger before the next gate check ran,
or its `charged_ceiling` was recovered before the next gate check ran.
Only the ONE truly-last attempt's tail is ever unpriced, and it is
unpriced by at most `0.0157h` regardless of whether it completes or
crashes. **Sequential admission does not merely dodge KW4.2 — it
structurally converts an N-attempt unpriced term into a 1-attempt
term, and G1 shows that term is bounded the same way whether the last
attempt completes or the orchestrator crashes trying to run it.**

Rounding conservatively (retaining the prior revisions' disclosed
`≈15.20h` figure as the reported internal bound — `0.0031h` looser
than Rev 3's `15.0126h` claim, per the KW5.7 correction; **§R5 H3
corrects what this figure covers: it is a rounding of the tight
LEDGER bound `15.0157h`, and it now sits BELOW the true
worst-case-spend bound derived above, not above it as prior
revisions implied** — the disclosed `15.20h` was never itself the
true-spend ceiling) and adding a **stated — not derived; a policy
choice, disclosed as such — supervisor margin**, covering the
log-interval overshoot's own disclosed contention-variance
("proportionally more under exactly the contention the ceiling exists
to survive," KW4.9), the process-startup term (subprocess
spawn/interpreter/import latency, ledger-file I/O, the two
`harvest()` invocations) the cell-level `ceiling_s` check does not
model — **now EXPLICITLY PRICED, not merely margin-absorbed, for every
reconstructed `COMPLETED` row via the startup allowance `s=0.0053`
(§R7 J6, above); what remains for the margin to cover is the residual
uncertainty IN that estimate, not the whole unpriced term** — **and
now (§R7 J6, correcting §R5 H3's `0.0041h` figure, which used a
Class-1-only accounting) the `0.1737h` by which the two-class true
worst-case-spend bound `15.3737h` exceeds the disclosed `15.20h`
figure — the margin is doing three jobs, all disclosed here, none
silently** — fixed at a round, generous
**0.30 GPU-h** (`0.30 ≫ 0.1737`, so the margin absorbs this addition
with `0.1263h` of headroom remaining, still comfortable though visibly
less slack than the pre-J6 `0.0041h`-vs-`0.30h` comparison implied):

```
declared program ceiling = 15.20  (disclosed, conservative-rounded internal bound)
                          + 0.30  (supervisor margin, stated)
                          = 15.50 GPU-h
```

This is the ONE ceiling the orchestrator's pool spec carries (§6) —
flat, independent, derived by induction over a genuinely sequential,
crash-recoverable admission process. **A disclosed consequence, not a
hidden failure mode:** in the pathological case where all 12 primary
cells consume their full ceiling before the conditional arm is
considered, the hard gate correctly THROTTLES OR REFUSES part or all
of the conditional arm rather than exceeding 15.00 — degrading
gracefully to `run_status="COMPLETE-DEGRADED"` (reduced conditional
coverage) or, in the more severe case where even the primary sweep
cannot complete, `"EXHAUSTED-BUDGET"` (G4, above) — never to a silent
overrun. In the intended, non-adversarial case the numbers stay far
inside this envelope: nominal primary (≈6.65h) + nominal conditional
worst-case (K=30, ≈4.62h) = **≈11.27h**, informational only, not the
enforced bound.

**Ceiling reference table (E2, unchanged values — informational
sizing justification ONLY; Rev 3: the gate charges the CLI value
directly, 1.20/2.32, not these per-K figures — KW4.4):**

| | per-cell ceiling (`≥2×nominal`, floor 1.0h) | ×N cells (informational sum) |
|---|---|---|
| Primary K=26 (80K) | 1.0211 h | 4.084 h |
| Primary K=28 (80K) | 1.1073 h | 4.429 h |
| Primary K=30 (80K) | 1.1946 h | 4.778 h |
| Conditional K=26 (160K) | 1.9764 h | 7.906 h |
| Conditional K=28 (160K) | 2.1432 h | 8.573 h |
| Conditional K=30 (160K, worst case) | 2.3121 h | 9.248 h |

(All 6 values re-derived by direct execution of `max(2×nominal, 1.0)`
against the corrected nominals above; shown only to justify why
1.20h/2.32h clear every K's own floor — the enforced/charged bound is
the ORCHESTRATOR CONTRACT's 15.50h above, never a sum of these.)

**Margin claim, corrected (KW3.2; digit fixed, KW4.10).** Rev 1's
supporting sentence *"every 1×-budget cell ever run has stayed within
1.06× of its own K's mean"* is **deleted — it was false** (the audit
found 4 of 24 archive config groups exceed 1.06×, max 1.092×). The
TRUE archive-wide figure, verified CLEAN by the audit over all 97
completed cells / 24 groups and independently re-executed this
revision to the same digit: **the largest max/nominal ratio ever
observed in this program is 1.2069×, rounding to 1.207×** (K32,
2×-budget, seed 3: `1.2685/1.0510` — Rev 2's "1.206×" was a truncation
in the flattering direction, KW4.10). Under the restored `≥2×nominal`
ceilings, this leaves `2.00/1.2069 ≈ 1.657×` (≈1.66×) of headroom
beyond the worst spike ever recorded — a substantially WIDER safety
margin than Rev 1's 0.75h/1.50h trim (which was `1.26×`–`1.47×`
nominal, i.e. inside 1.5× of the worst archive spike, the exact
contention risk KW2.2 flagged). This margin bounds a SINGLE training
run's own variance around its ceiling — a property of the harness,
independent of and unaffected by the ORCHESTRATOR CONTRACT's ledger
fix above.

**D5 — eval-inclusive ceiling handling (KW2.2, corrected not merely
disclosed).** `gpu_h` (used throughout this pricing) is
`elapsed_s/3600` measured end-to-end (`ncr_earlyln_scale.py:303-304`)
— i.e. `gpu_h` already INCLUDES eval; every pricing figure above is
eval-inclusive. What is NOT eval-inclusive is the runtime `ceiling_s`
ENFORCEMENT itself (`train_earlyln_cell`, `:198-201`) — it checks only
during training, so a cell whose training finishes right at the
ceiling can still add eval time afterward, unbounded by the check.
Measured directly this revision across every archived cell (K=16/24/32,
1×/2×, n=64): eval overhead is **0.35%–1.58% of total elapsed**
(corrected this revision — KW3.9 found Rev 1's "0.7%–1.5%" scope wrong
at both ends; the max ABSOLUTE figure was already right and is
unchanged), max observed **45.5s = 0.0126 GPU-h** (K32, 2×, seed 3).
This is exactly the per-cell figure the ORCHESTRATOR CONTRACT's
worst-case derivation (above) uses; the `≈15.20`/`15.50` GPU-h program
bounds already reflect it.

**D5/E4 — ABORTED-BUDGET / MISSING / non-COMPLETED cell rule for BAND
classification, Rev 2 mechanization (KW2.2/KW2.3/KW3.4), Rev 3
cross-referenced to the orchestrator's own gates (F1) rather than a
launcher that no longer exists in this form.** Rev 1's version
deadlocked (`PERSISTENTLY-ABORTED` could never reach 4/4 COMPLETED, so
its K could never be classified) and left the retry/exclusion logic
unmechanized, with no named enforcement point against `harvest()`'s
actual behavior. Rev 2's bounded, mechanized rule stands (the band
side of E4 was independently verified correct by round 3, KW4.5's own
text: *"delivered exactly as the disposition specified"*); Rev 3 only
repoints its cross-references from the retired launcher-side E1 check
to the orchestrator's HARD/RETRY gates it now shares with the trigger
rule (§4, above), and fixes KW4.7/KW4.8:

- **Retry, bounded.** A cell whose attempt 1 lands in
  `status=="ABORTED-BUDGET"` (`train_earlyln_cell`, `:198-201`),
  `CRASHED-1`, or `CRASHED-RECOVERED` (G1/G3, Rev 4 — a deterministic
  crash or a mid-attempt orchestrator death gets the SAME retry
  treatment as a budget abort, never a separate weaker/stronger path)
  is retried AT MOST ONCE — subject to the orchestrator's RETRY GATE
  (`realized_gpu_h<12.00`, above) — with no ceiling change. If the
  retry ALSO fails to reach `COMPLETED` (any of the same three
  non-completing outcomes again, or the retry itself is refused by the
  RETRY GATE), that seed becomes **`PERSISTENTLY-ABORTED` — a TERMINAL
  state, never retried again, regardless of remaining budget.** (A
  `CRASHED`-family outcome is disclosed distinctly from
  `ABORTED-BUDGET` in `attempts[].status` — KW5.3's "not a coin-flip
  seed" point — but both feed the identical bounded-retry/interval-logic
  machinery below; the distinction is for a reader diagnosing WHY a
  seed is incomplete, not for how it is counted.)
- **Denominator, fixed at 4 (A4.9 guard preserved — KW3.4's
  "denominator contradiction" closed).** `PERSISTENTLY-ABORTED`,
  `GATE-REFUSED`, and MISSING (never-attempted) cells are NOT excluded
  from `n_seeds`; the rate denominator for every K stays exactly 4,
  matching the partition's own `r∈{0,1,2,3,4}` domain. What is unknown
  is the NUMERATOR contribution of the incomplete seed(s), not the
  denominator — resolved by interval logic, next. This is the SAME
  `n_completed`-based resolution-state computation §4's trigger rule
  (above) consumes — one shared computation, two independent
  downstream decision functions (`classify()` here, the trigger's
  K-scan there), which is exactly why the two can (and per KW4.5's 11
  enumerated cases, do) disagree even when each individually decides.
- **Interval logic for a K with exactly one terminal-aborted
  (`PERSISTENTLY-ABORTED`, from any of `ABORTED-BUDGET`/`CRASHED`/
  `CRASHED-RECOVERED`), MISSING, or `GATE-REFUSED` cell (E4, exact).**
  Let `r_known` = the CONVERGED count among that K's 3 resolved seeds.
  Evaluate the §5 six-rule classification procedure TWICE — once with
  `r = r_known` and once with `r = r_known + 1` (the other two K's
  `r`-values held fixed) — **for EVERY value of `r_known∈{0,1,2,3}`,
  never collapsed to one candidate even where the trigger's
  `ROBUST`-only scan can (KW5.13 — the scope note on the resolution-
  state table above)** — and compare the resulting bands (the
  `[NON-MONOTONE]` tag included):
  - **Same band both ways ⇒ DECIDE.** Report that band, with a
    disclosure flag naming which K's rate was interval-resolved and
    from which incomplete-cell state.
  - **Different bands ⇒ the STUDY reports `INCOMPLETE-AT-K` (KW4.8
    fix — this is a study-level verdict, not a per-K band the
    six-rule procedure can return; §5's partition is a function of the
    FULL triple and yields exactly one label, never one per K).**
    `INCOMPLETE-AT-K` is reported as its own outcome, orthogonal to
    the 125-outcome partition (§5), explicitly EXCLUDED from frontier
    claims, and disclosed carrying the affected K(s) as a field —
    never silently forced into either candidate band.
  - **Decide-rate, disclosed (KW4.7 — not previously stated).**
    Independently re-executed this revision (a ~40-line sweep, not
    hand-counted): at `r_known=2` — precisely the value where
    `ROBUST(r):=r≥3` straddles the boundary, the case a sub-ROBUST
    rung is expected to produce — the two candidates give DIFFERENT
    bands (fail to decide) in **16/25=64%** (K=26 incomplete),
    **17/25=68%** (K=28), and **25/25=100%** (K=30) of the surrounding
    configurations; with two K's each singly-incomplete, the
    cross-product decides in only **45–54%** of configurations. A
    single terminal abort at K=30 with `r_known=2` therefore
    **guarantees** `INCOMPLETE-AT-K` for the study band (no further
    recourse — the retry is exhausted, the state is terminal). This is
    not a logic error; it is a reliability property a reader should
    not assume is better than it is.
- **Two or more incomplete cells at one K, or incomplete cells at
  MULTIPLE K's simultaneously.** With ≥2 incomplete cells at a single
  K, the interval width exceeds what a two-way comparison can resolve;
  that K is `INCOMPLETE-AT-K` (band) / `UNRESOLVED` (trigger,
  excluded from candidacy per F2) UNCONDITIONALLY, no candidate
  comparison performed. If DIFFERENT K's each have exactly one
  incomplete cell at the same time, interval logic is applied
  compositionally: evaluate over the full cross-product of each
  affected K's two candidate `r`-values (`2^m` candidates for `m`
  singly-incomplete K's); if every candidate yields the SAME band,
  decide (disclosing all `m` interval-resolved K's); otherwise
  `INCOMPLETE-AT-K` for the affected K's, both/all candidate bands
  disclosed. (The trigger rule, §4 above, runs the identical
  cross-product independently for `K_trig` and applies its own
  tie-break when it disagrees — the two need not agree with each
  other on whether they can decide, per KW4.5's 11 cases.)
- **Enforcement point, named (closes KW3.4's "no enforcement point"
  defect; Rev 4 — no `harvest()` code patch needed anymore, subsumed by
  G2).** Rev 2/Rev 3's version of this bullet instructed a build-stage
  patch to `harvest()`'s `n_seeds`/`gate_eligible` computation
  (`ncr_earlyln_scale.py:380-406`), because `discover_seeds_by_K`'s
  glob counts FILE PRESENCE, and a file existed on disk for an
  `ABORTED-BUDGET`/`PERSISTENTLY-ABORTED` cell too under the old
  (pre-orchestrator) harness usage — so `n_seeds` silently counted it
  even though that cell never reached `COMPLETED`. **Under G2's
  canonical-path contract (§4, "Trigger evaluation point + harvest
  invocations"), this is no longer possible: a canonical-path file is
  written ONLY on `COMPLETED` acceptance, so
  `discover_seeds_by_K`'s existing file-glob-presence count over the
  canonical directory IS a status-based `n_completed` count, by
  construction — no `ABORTED-BUDGET`/`CRASHED`/`CRASHED-RECOVERED`/
  `PERSISTENTLY-ABORTED`/`GATE-REFUSED` cell can ever produce a file
  there to be miscounted.** `harvest()`'s existing code
  (`:358-380`) is therefore the correct instrument AS-IS against the
  canonical directory — no patch is specified or needed. **Naming
  which fields, exactly (§R5 KW6.15 — "correct AS-IS" needs scope: two
  of `harvest()`'s own computed fields do NOT mean what this design
  needs on an interval-resolved K):** this design consumes
  `per_K[K]["n_converged"]` (a COUNT) and `n_seeds` (read AS
  `n_completed`) ONLY. `harvest()`'s own `rate` (`n_converged/n_seeds`,
  a 3-denominator RATE on an incomplete K, not the count `classify()`
  needs), `gate_eligible` (`n_seeds>=4`, `False` on any interval-
  resolved K), and `gate1_label` (e.g. `"SUB4-DISCLOSED-ONLY(n=3)"`)
  are computed against a 3-denominator and are NEVER read by this
  design's band procedure — an implementer pointed at `per_K[K][
  "rate"]` on an interval-resolved K would read `0.667` where this
  design means the count `r=2`. The rule that DOES still need a
  build-stage implementer to apply it (this design stays DRAFT and
  edits no code) is the classification logic itself:
  - `n_completed==4` → classify normally (as today).
  - `n_completed==3`, the 4th `MISSING`/`PERSISTENTLY-ABORTED`/
    `GATE-REFUSED` → apply the interval-logic rule above.
  - `n_completed≤2` → `INCOMPLETE-AT-K` unconditionally.
  This logic, plus G2's copy-on-accept discipline and the
  orchestrator's own ledger/gate machinery (§4, above), are together
  the points that actually enforce this design's stated rules against
  the harness's real behavior — one fewer moving part than Rev 3
  specified, not one more.

**Job-spec template (KW2.6, Rev 3 rescoped to F1's delivery model —
pool-conformance artifact, specified here for the build stage; this
design remains DRAFT and creates no job JSONs itself).** Rev 2 gave
every one of up to 16 CELLS its own pool entry; **F1 replaces that
with exactly ONE** `queue/jobs/pending/*.json` entry, job-108's own
8-field format (`id, lane, hypothesis, cmd, gpu_h_estimate,
output_dir, validity_check, notes`), whose `cmd` is an ABSOLUTE
interpreter/working-directory invocation of the (build-stage)
orchestrator script — never a single `ncr_earlyln_scale.py` cell
command directly. `output_dir` is
`/home/nvidia/ncr/results_kwall_characterization` (absolute, KW5.11 —
matching every other path in §4). `gpu_h_estimate: 15.50` (§4's
derived-plus-margin ceiling). **`validity_check`, REWRITTEN against
the actual ledger/report schema (§R5 H4 — closes KW6.4/KW6.7/KW6.14;
Rev 4's version is superseded, not merely amended: it asserted a
field, `d_override_of(a)`, that does not exist anywhere on-disk and
that opening the cell JSON cannot supply for a `CRASHED`/
`GATE-REFUSED` row either, throwing on exactly the degraded runs G4
exists to route to `completed/`; it also accepted a total no-op).**
Asserts, over the orchestrator's OWN `orchestrator_report.json`
(schema above) — UNIVERSAL assertions first, then EXACTLY ONE
per-`run_status` disk-evidence assertion selected by the report's own
claimed value:

*Universal (every accepted report must satisfy all of these):*
1. `run_status in {"COMPLETE","COMPLETE-DEGRADED","EXHAUSTED-BUDGET",
   "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE"}` (every value G4/H5 defines
   as a non-buggy completion of the orchestrator's own logic —
   `STOPPED-BY-OPERATOR` is deliberately excluded, G4 above).
2. `ledger.realized_gpu_h_final <= 15.50`.
3. `abs(ledger.realized_gpu_h_final - sum(a["elapsed_h"] for a in
   ledger.attempts)) <= 1e-6` (§R5 KW6.14 — was an exact float
   equality used as a job-routing gate; an epsilon tolerance survives
   accumulation-order differences that exact equality does not, without
   weakening the check's purpose of catching a real bookkeeping slip).
4. `all(a["K"] + 1 == a["d_override"] for a in ledger.attempts)`
   (§R5 KW6.4 — REWRITTEN as pure ledger arithmetic over the new
   `d_override` field, §R5 H4's schema addition above, recorded by the
   orchestrator AT DISPATCH/GATE-CHECK TIME from the CLI value it
   itself passes or would have passed — never by opening a cell JSON
   that may not exist. Defined for EVERY row, `GATE-REFUSED` included,
   closing the exact `d=K+1`-vs-`d=2K` filename-collision risk
   `EXPERIMENT_LOG.md:8452` already forced a workaround for once).
5. `all(v == "PASS" for v in smoke.values())` (§R5 KW6.13 — the
   `smoke` block is populated by the orchestrator itself before it
   ever dispatches, below; this re-assertion is redundant with the
   build-release gate by design — cheap defense-in-depth, not new
   information).
6. `band["label"] in {"FRONTIER-AT-K*=24", "FRONTIER-AT-K*=26",
   "FRONTIER-AT-K*=28", "FRONTIER-AT-K*=30", "GRADUAL-DECAY",
   "NON-MONOTONE-UNRESOLVED", "INCOMPLETE-AT-K"}` and
   `trigger["resolution"] in {"unanimous","tie-break-min",
   "TRIGGER-UNRESOLVED"}` (§R5 KW6.7, placeholder expanded §R6 I6/
   KW7.14 — the six literal band labels are §5's own six-rule
   `classify()` outcomes, EXHAUSTIVE and mutually exclusive over that
   procedure's own domain, §5 above; `INCOMPLETE-AT-K` is the seventh,
   study-level label §5 also defines. The `[NON-MONOTONE]` tag is a
   SEPARATE boolean field, `band["non_monotone_tag"]`, never part of
   the label string — not asserted here, nothing in this design reads
   it as part of the enum. The three 160K qualifier bands
   — `CONFIRMED-WALL-AT-160K`/`SLOW-CONVERGENCE-AT-160K`/
   `PARTIAL-IMPROVEMENT-AT-160K` — are `conditional.qualifier_band`, a
   DIFFERENT field this assertion does not touch. **`trigger["resolution"]`
   is now UNCHANGED text but a NEWLY-TRUE assertion (§R7 J2 — closes
   KW8.2's FATAL): the trigger pseudocode's f-string previously emitted
   `f"tie-break-min, candidates were {...}"` into THIS field, which is
   not a member of the 3-value set above under exact string equality —
   every tie-break resolution failed this assertion and was routed to
   `failed/` after the full ≤15 GPU-h was spent. The producer is fixed
   at the source (trigger pseudocode, above) to emit the bare literal;
   this assertion's TEXT does not need to change, only the value it is
   now actually given. The candidate list moves to the new
   `trigger["resolution_detail"]` field, schema above — un-asserted,
   informational.**)
7. **(§R7 J5, NEW — closes KW8.5's MAJOR: no assertion anywhere
   previously read the CONDITIONAL canonical directory at all.)** If
   `conditional is not None and conditional["qualifier_band"] is not
   None` (any report carrying a 160K qualifier band, whether from a
   paid conditional dispatch or the $0 `K_trig==32` archive citation,
   §5 above): EITHER (a) `conditional["launched"]==True` AND exactly 4
   files exist in the conditional canonical directory
   (`results_kwall_characterization_160k/`, disjoint from the primary
   canonical directory by construction, above), each `status==
   "COMPLETED"`, matching `len({(a["K"],a["seed"]) for a in
   ledger.attempts if a["arm"]=="conditional" and
   a["status"]=="COMPLETED"})==4`; OR (b) `conditional["launched"]==
   False` AND `trigger["K_trig"]==32` (the $0-branch archive citation —
   no conditional cells were ever dispatched, so no conditional disk
   evidence is required or possible). A `qualifier_band` satisfying
   NEITHER (a) NOR (b) — e.g. `launched=True` with 0 conditional
   canonical files, KW8.5's own fabricated-evidence payload, below —
   FAILS. **(§R8 K1 — retracts the prior scope note, which called the
   partial-conditional case "hypothetical future" and "not currently
   defined." KW9.1's FATAL: it IS defined — G4's `COMPLETE-DEGRADED`
   sub-case (ii)/(iii), and §5's qualifier-band paragraph above now
   states the 4/4 precondition explicitly. The mirror clause below is
   the enforcement point that used to be missing.)**
   If `qualifier_band is None` AND `conditional is not None` AND
   `conditional["launched"]==True`: **(§R8 K1/K7 — new; also closes
   KW9.7's MAJOR, a paid conditional arm silently absent from the
   ledger, by the same clause)** let `n_cond_canon` = the number of
   files in the conditional canonical directory with
   `status=="COMPLETED"`; then `n_cond_canon == len({(a["K"],a["seed"])
   for a in ledger.attempts if a["arm"]=="conditional" and
   a["status"]=="COMPLETED"})` (the disk evidence and the ledger agree
   on what actually completed — real conditional spend can never be
   invisible to the ledger, whether or not a band is claimed) **AND
   `n_cond_canon < 4`** (a throttled arm has SOME real completions but
   never all 4 — 4/4 conditional canonical with `qualifier_band is
   None` contradicts §5's now-unconditional "4/4 triggers a band" rule
   and correctly FAILS here). Otherwise (`conditional is None`, or
   `qualifier_band is None` and `launched` is `False` or absent):
   **(§R9 m3 — closes KW10.4's residue: this arm previously asserted
   NOTHING, so real paid conditional spend stayed invisible whenever
   the report's OWN `conditional` block claimed non-dispatch,
   regardless of what is actually on disk — the identical
   never-invisible guarantee the branch above already gives a
   THROTTLED arm, extended here to a claimed-ABSENT one)** assert the
   conditional canonical directory
   (`results_kwall_characterization_160k/`) contains **0** files with
   `status=="COMPLETED"` — the conditional arm was never dispatched,
   so no conditional disk evidence may exist; if it does, this branch
   FAILS (real spend cannot hide behind a report's own claim of
   non-dispatch).
8. **(§R8 K3, NEW — closes KW9.3's MAJOR, the exact sibling of
   universal assertion 3 above, which already does this for
   `realized_gpu_h_final`: `validity_check` previously TRUSTED
   `charged_vs_measured.ceiling_charged_gpu_h`/`ceiling_charged_
   fraction` as a self-report, even though both are a pure function of
   `ledger.attempts` the check already has open. A ledger that is 93%
   `CRASHED-RECOVERED` noise could self-declare a low fraction and
   evade J4's dichotomy below entirely.)** Recompute
   `ccgh_recomputed = sum(a["elapsed_h"] for a in ledger.attempts if
   a["ceiling_charged"])` and assert `abs(charged_vs_measured.
   ceiling_charged_gpu_h - ccgh_recomputed) <= 1e-6`; then, IF
   `ledger.realized_gpu_h_final > 0` (guarding the division — a
   genuine no-op reads `0/0`, undefined, and cannot satisfy either
   `EXHAUSTED-BUDGET*` label's own `>13.80` base clause regardless, so
   skipping the fraction half here costs nothing reachable), assert
   `abs(charged_vs_measured.ceiling_charged_fraction -
   ccgh_recomputed/ledger.realized_gpu_h_final) <= 1e-6`. A report
   whose self-reported fraction disagrees with its own ledger — KW9.3's
   own payload, a 93%-ceiling-charged ledger self-declaring `0.20` —
   now FAILS here, before any per-`run_status` branch (including J4's
   `<=0.50`/`>0.50` dichotomy, which reads this field but never
   recomputed it) is ever reached.

*Per-`run_status` (exactly one branch fires, matching the report's own
claimed value — §R5 H4, closes KW6.7's no-op hole by construction;
rewritten §R6 I4/I5 — quantified over the FULL pre-registered §5
outcome space, closing KW7.4's FATAL, which showed every branch below
rejected the design's own `INCOMPLETE-AT-K` and `COMPLETE-DEGRADED`
sub-case (i) outcomes):*
- `run_status=="COMPLETE"` ⇒ **(§R6 I4)** if `band["interval_resolved_
  Ks"]==[]` and `band["incomplete_at_K"] is None`: exactly 12 files
  matching `earlyln_K{26,28,30}_s{0-3}.json` exist in the PRIMARY
  canonical directory and every one parses with `status=="COMPLETED"`
  **AND (§R9 m7 — closes KW10.8's residue: the strict branch carried
  NO ledger clause at all, so a ledger that was empty, or 12
  zero-cost primary `GATE-REFUSED` rows, still PASSED beside 12
  genuine canonical files, under-reporting `realized_gpu_h_final`
  vacuously — the primary-arm twin of KW9.7/KW10.4) every primary
  `(K,seed)` pair has ≥1 row in `ledger.attempts` (J1(a)), AND the
  primary canonical count (12) equals `len({(a["K"],a["seed"]) for a
  in ledger.attempts if a["arm"]=="primary" and
  a["status"]=="COMPLETED"})` (J1(b)) — trivially `12==12` for any
  legitimate run, free** (this and the base filesystem check together
  are unchanged from §R5's own strict case in substance, now with the
  same positive-evidence pair the OTHERWISE branch already carries).
  OTHERWISE (some K is
  disclosed incomplete): for every `K∈{26,28,30}` named in either
  field, that K's canonical-file count in the primary directory is
  `<4`; for every K named in NEITHER field, that K's canonical-file
  count is exactly `4` — a filesystem check tied to the report's own
  disclosed `band` fields, not a bare count — **AND (§R7 J1 — closes
  KW8.1's FATAL regression) every primary `(K,seed)` pair has ≥1 row
  in `ledger.attempts`, AND the primary canonical count equals
  `len({(a["K"],a["seed"]) for a in ledger.attempts if
  a["arm"]=="primary" and a["status"]=="COMPLETED"})`** — the same two
  positive-evidence clauses `COMPLETE-DEGRADED`'s branch already
  carries, below; the OTHERWISE branch alone, without these, was
  satisfiable by an empty ledger (`attempts=[]`) claiming every primary
  K incomplete, since `0<4` holds vacuously for all three. **AND (§R8
  K2 — closes KW9.2's MAJOR: the two clauses just above are evidence of
  a ROW, not evidence of WORK, and are satisfiable by 12 zero-cost
  `GATE-REFUSED` rows at `elapsed_h=0.0`)
  `len({(a["K"],a["seed"]) for a in ledger.attempts if
  a["arm"]=="primary" and a["status"]=="COMPLETED"}) >= 1`.**
- `run_status=="COMPLETE-DEGRADED"` ⇒ **(§R6 I4, replaces the
  12-canonical filesystem check sub-case (i) can never pass)** every
  primary `(K,seed)`, `K∈{26,28,30}`, `seed∈{0,1,2,3}`, has AT LEAST
  ONE row in `ledger.attempts` (a terminal disposition exists for all
  12 cells — the no-op's escape hatch), AND the canonical-file count in
  the primary directory equals `len({(a["K"],a["seed"]) for a in
  ledger.attempts if a["arm"]=="primary" and a["status"]=="COMPLETED"})`
  (the G2+H2 identity — never required to equal 12), AND at least one
  row in `ledger.attempts` has `status=="GATE-REFUSED"` or contributes
  to a `PERSISTENTLY-ABORTED` derivation (the positive evidence a
  throttle actually occurred). **AND (§R8 K2 — mirrors the `COMPLETE`
  branch's fix above, closing the same KW9.2 hole this label shares
  with it) `len({(a["K"],a["seed"]) for a in ledger.attempts if
  a["arm"]=="primary" and a["status"]=="COMPLETED"}) >= 1`.**
- `run_status=="STOPPED-BY-OPERATOR"` ⇒ excluded from the accept-set
  entirely (universal assertion 1) — this branch never fires here; the
  stop-file marker check lives in the orchestrator's own pre-write
  self-check instead (§R6 I6/KW7.11, G4 above), never here.
- `run_status=="EXHAUSTED-BUDGET"` ⇒ **(§R6 I5, adds the negative
  half)** `ledger.realized_gpu_h_final > 13.80` (`=15.00−1.20`, the
  exact threshold at which even one more primary-ceiling admission is
  impossible — LEDGER-evidenced spend, per G4's definition above; this
  alone is what fails the audit's no-op example,
  `realized_gpu_h_final=0.0`) **AND** fewer than 12 canonical PRIMARY
  files exist in the primary directory **AND** at least one row in
  `ledger.attempts` has `arm=="primary"`, `attempt_n==1`, and
  `status=="GATE-REFUSED"` (closes KW7.5: without this, a 12-canonical
  baseline whose ledger happens to read `>13.80` — e.g. every primary
  cell landing near its own ceiling — passed under a label its own
  disk evidence refuted) **AND (§R7 J4 — closes KW8.4's MAJOR)
  `charged_vs_measured.ceiling_charged_fraction <= 0.50`** (without
  this, a ledger that is mostly `CRASHED-RECOVERED`/reconstructed
  environment-fault noise could mislabel itself plain
  `EXHAUSTED-BUDGET` and evade the next label's binding
  resubmission-adjudication protection, below).
- `run_status=="EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE"` ⇒ the same three
  base clauses as `EXHAUSTED-BUDGET` (§R6 I5: `>13.80` / `<12`
  canonical / ≥1 primary first-attempt `GATE-REFUSED`) **PLUS**
  `charged_vs_measured.ceiling_charged_fraction > 0.50` — **the strict
  mirror of the `<=0.50` clause just added to `EXHAUSTED-BUDGET` (§R7
  J4): the two labels now PARTITION the shared base disk state exactly
  at the `0.50` threshold — `ceiling_charged_fraction` is a single
  real number that is either `<=0.50` or `>0.50`, never both, never
  neither, so for any ledger satisfying the three base clauses, EXACTLY
  ONE of the two `EXHAUSTED-BUDGET*` labels is the correct claim.**

**In-text test list (§R6 I4, extended §R7 J1/J2/J4/J5, further
extended §R8 K1/K2/K3/K4, precision-corrected §R9 M1/m1 — every
§5-reportable outcome type PASSES (a throttled conditional arm PASSES
by reporting `qualifier_band=null`, per K1) with the evidence clause
named; every R5/R6/R7/R8/R9 adversarial JSON FAILS; re-run this
revision as a 24-payload suite (`vcheck_r8_rev.py` +
`drive_vcheck_r8_rev.py`, re-executed this revision against the
`L6`-corrected text; this revision's session scratchpad — a
DIFFERENTIAL harness that runs every payload through both the
pre-Rev-8 transcription and the current amended one, so the delta is
executed, not asserted), not hand-checked — 24/24 match expectation
under the amended text (failure-reason lists printed, not just
verdicts — see below), and, as a regression check, 24/24 ALSO match
under the pre-Rev-8 transcription run side-by-side, confirming **SIX**
payloads (B1, B1', B2, B2', B3-NEG, B4) flip PASS→FAIL and
B3-AMENDED (the legitimate 3/4-throttled report, listed above) is
newly **emittable** under the amended §5 rule — its `validity_check`
verdict is UNCHANGED, PASS under both transcriptions; pre-Rev-8 §5's
unconditional wording gave a compliant implementer no way to EMIT a
`null` band in the first place, which is precisely the FATAL K1
closed — these SIX flips plus B3-AMENDED's newly-sanctioned emission
are the ONLY behavioural deltas WITHIN THE 24-PAYLOAD SUITE (§A10/n1
scoping, per KW11.2: the four §R9 teeth-probes OUTSIDE the suite —
D1/D1', m7's `COMPLETE`/strict ledger clause, and D2/D2', m3's U7
Otherwise-arm assertion — also flip PASS(OLD)→FAIL(NEW), by
construction and by design, as the forced-fail demonstrations of
their new clauses); nothing else regressed (§R9 m1 —
corrects KW10.2's "five, and B3-AMENDED is one of them" miscount; the
prior tally hid B2' and B3-NEG because the driver auto-matched an
`"N/A(old had no rule)"` OLD-expectation string as a pass regardless
of the actual OLD verdict — this revision's harness does not carry
that auto-match). **Every payload this suite REBUILDS from a prior
round's non-closing numbers is disclosed here, not asserted PASS in
silence: A6/A6' (§R8 K4, closing KW9.4) and L6 below (§R9 M1, closing
KW10.1) — no other payload among the 24 is a rebuild.**

*Reportable outcomes, traced against the rewritten branches above:*
- `COMPLETE`, 12/12 canonical, nothing disclosed incomplete — PASSES
  the strict-12 clause (unchanged base case).
- **`INCOMPLETE-AT-K` (§R6 I4's walked case — KW7.4's own construction:
  one primary seed `CRASHED` on both attempts, 11 canonical primaries,
  `run_status="COMPLETE"` — CRASHED is not budget-caused, so `COMPLETE`
  is the correct claim here, per G4's own "no budget-caused refusal"
  definition, with `band["incomplete_at_K"]=[30]` naming the affected
  K):** `COMPLETE`'s OTHERWISE branch fires (`incomplete_at_K` is
  non-`None`); K=30's canonical count is 3 (`<4`, consistent with the
  disclosed field), K=26 and K=28 each read 4 — **PASSES.** (Under
  Rev-5's unconditional 12-canonical clause this same payload FAILED —
  the FATAL this discharges.)
- **`COMPLETE-DEGRADED` sub-case (i) *primary-retry-refused* (§R6 I4's
  second walked case): attempt-1 `CRASHED` on one primary seed,
  attempt-2 `GATE-REFUSED` by the retry gate, 11 canonical primaries,**
  the retry gate's interval logic DECIDES a real band (or the study
  reports `INCOMPLETE-AT-K` if it does not — either is compatible).
  All 12 `(K,seed)` pairs carry ≥1 row (11 `COMPLETED`, 1
  `CRASHED`+`GATE-REFUSED`); canonical count (11) equals the count of
  `COMPLETED` rows (11); ≥1 `GATE-REFUSED` row exists — **PASSES all
  three clauses.** (Under Rev-5's identical-to-`COMPLETE` 12-canonical
  clause this sub-case could NEVER pass by its own construction — the
  second half of KW7.4's FATAL this discharges.)
- `COMPLETE-DEGRADED` sub-case (ii)/(iii) — 12/12 primary canonical (the
  throttle is downstream, on the conditional arm), ≥1 `GATE-REFUSED`
  row on a conditional attempt — PASSES (12 canonical == 12
  `COMPLETED` primary rows; the throttle-evidence clause is satisfied
  by the conditional row, which the assertion does not restrict to
  `arm=="primary"`).
- `EXHAUSTED-BUDGET`, `realized=14.55`, 9 canonical primaries, a
  primary K's first attempt genuinely `GATE-REFUSED` — PASSES all
  three clauses.
- **`EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE` (§R9 M1 — REBUILT
  self-consistent, closing KW10.1's MAJOR: the prior payload's own
  number did not close — L5's disk state above structurally CAPS
  `ceiling_charged_gpu_h` at `4.80` (only 3 non-canonical pairs remain
  once 9 are canonical, and ≥1 of those 3 must be a zero-charge
  `GATE-REFUSED`), so `0.71×14.55=10.3305` was never reachable while
  holding L5's `realized=14.55` fixed — this bullet therefore uses its
  OWN disk state, not L5's, exactly as K4 built A6's own state rather
  than reusing L5's; composition restated per §A10/N1 — the §R9
  draft's `COMPLETED` row declared `elapsed_h=2.00`, which NO primary
  attempt can produce under the design's own rules (`--ceiling-gpuh
  1.20` + τ=0.0157 tail + s=0.0053 startup allowance ⇒ maximum
  reachable primary row `1.2210`; KW11.1); replaced with R10 §4's
  option (A), executed PASS by the R10 audit — which is the
  composition Rev-8's harness ACTUALLY ran, so the in-text spec and
  the historically-executed payload are now IDENTICAL, fully retiring
  the KW9.4/KW10.1/KW11.1 substitution lineage):** 12 primary
  `(K,seed)` pairs — 10 single-attempt `CRASHED-RECOVERED`
  (`ceiling_charged=true`, `1.20` each → `12.00`), 1 pair `CRASHED`
  on BOTH attempts (`ceiling_charged=true` on each row,
  `1.20+1.20=2.40`, 0 canonical files), 1 pair `GATE-REFUSED` at
  `attempt_n=1` (`0.0`) — `10+1+1=12` pairs, 0 canonical files
  total. Self-reported `charged_vs_measured`:
  `ceiling_charged_gpu_h=14.40` (`=12×1.20`, matching the 12
  `ceiling_charged=true` rows exactly, so universal assertion 8
  PASSES), `realized_gpu_h_final=14.40` (`=12.00+2.40+0.00`, so
  universal assertion 3 PASSES: `|14.40−14.40|=0`),
  `ceiling_charged_fraction=14.40/14.40` exactly (`=1.0000` — the
  EXACT quotient, per m4's same rounding discipline). Claimed
  `run_status="EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE"`: the three base
  clauses hold (`14.40>13.80` ✓, canonical count `0<12` ✓, the
  `GATE-REFUSED` pair's row has `attempt_n==1` ✓), PLUS the mirror
  clause `ceiling_charged_fraction>0.50` → `14.40/14.40>0.50` ✓ —
  **PASSES**, with U1/U2/U3/U8 and all three base clauses confirmed
  PASSING first, by execution (R10's `r10_l6fix.py`, option (A):
  failure-reason list `[]`; round 11 to re-derive independently).
- `STOPPED-BY-OPERATOR` with a real stop-file on disk — correctly
  routes to `failed/` via universal assertion 1 (by design, per G4
  above — this is not a `validity_check` failure, it is the intended
  exclusion).
- **`tie-break-min` resolution (§R7 J2's own verification payload,
  extended §R8 K7/KW9.10 — the field J2 created was itself never
  exercised): `COMPLETE`, 12/12 canonical, `trigger.resolution
  ="tie-break-min"`, `trigger.K_trig=26`, `trigger.candidate_set=
  [26,28]`, `trigger.resolution_detail="candidates were [26, 28]"`
  (now set, matching what the fixed pseudocode's tie-break branch
  actually writes).** Universal assertion 6's `trigger["resolution"] in
  {"unanimous","tie-break-min","TRIGGER-UNRESOLVED"}` now sees the
  BARE literal (the trigger pseudocode fix, above) — exact membership
  holds — **PASSES.** (Under Rev 6's f-string producer, the identical
  disk state carried `resolution="tie-break-min, candidates were
  [26, 28]"`, which is NOT a member of the 3-value set — **FAILED**
  universal assertion 6 and was routed to `failed/` after the full
  ≤15 GPU-h was spent; this is KW8.2's FATAL, now discharged.)
- **LEGITIMATE conditional-throttled 3/4, qualifier band `null` (§R8
  K1 — closes KW9.1's FATAL; this is the exact payload the finding
  named): `COMPLETE-DEGRADED`, 12/12 primary canonical, the trigger
  DECIDED at `K_trig=26`, 3 of the 4 `K=26` conditional cells
  `COMPLETED` (real canonical files), the 4th conditional cell
  `GATE-REFUSED` before its first attempt, `conditional.launched=true`,
  `conditional.qualifier_band=null`** (per the amended §5 rule above:
  a throttled arm reports no band).** `COMPLETE-DEGRADED`'s branch:
  every primary pair has ≥1 row ✓, canonical count (12) equals
  distinct `COMPLETED` primary rows (12) ✓, a `GATE-REFUSED` row exists
  ✓, ≥1 distinct `COMPLETED` primary pair (§R8 K2) ✓. Universal
  assertion 7's amended clause: `qualifier_band is None` and
  `launched==True`, so the NEW mirror fires — conditional canonical
  count (3, matching 3 `COMPLETED` conditional canonical files) equals
  the ledger's distinct conditional `COMPLETED` count (3) ✓, AND `3<4`
  ✓ — **PASSES.** (Under the pre-Rev-8 text, this exact payload —
  routed through U7's OLD clause (a), which requires exactly 4
  conditional files whenever `qualifier_band` is non-`None` — only
  reached that branch if an implementer, following the OLD §5 text's
  unconditional "is reported" wording, set `qualifier_band=
  "SLOW-CONVERGENCE-AT-160K"` on a 3/4 read; that mislabelled version
  correctly FAILS both before and after this revision, since a band on
  a sub-4 conditional read was never valid. The FATAL was that the OLD
  design gave the implementer NO OTHER way to report this state
  honestly — U7 was silent on `qualifier_band=None`, so there was no
  path to a legitimate PASS for a genuinely-throttled arm. §R8 K1
  supplies that path: report `null`, not a band, and the mirror clause
  now validates it. A run that spent up to `2.32×3=6.96` GPU-h
  conditional plus the ≤`14.40` GPU-h primary baseline, honestly
  reported, is no longer routed to `failed/`.)

*Adversarial JSONs from R5+R6, re-traced against the rewritten
assertions — still killed by name:*
- **Audit-R5 no-op** (`run_status="COMPLETE"`, `attempts=[]`,
  `realized_gpu_h_final=0.0`): `band` carries no disclosed
  incompleteness in a genuine no-op, so `COMPLETE`'s strict-12 clause
  fires — 0 canonical files ≠ 12 — **FAILS.** Also fails
  `COMPLETE-DEGRADED`'s new identity: 0 canonical == 0 `COMPLETED` rows
  passes that ONE clause, but "every primary `(K,seed)` has ≥1 row" —
  **FAILS**, none of the 12 cells has any row at all.
- **Near-miss #1** (12 canonical primaries, `run_status="EXHAUSTED-
  BUDGET"`, `realized=14.40`): the NEW negative clause requires fewer
  than 12 canonical files — 12 is not `<12` — **FAILS** (this is
  exactly KW7.5's fix: the old positive-only clause let this through).
- **Near-miss #2** (`STOPPED-BY-OPERATOR`, `attempts=[]`, no stop-file
  marker): excluded outright by universal assertion 1, independent of
  the marker check — **FAILS**, unchanged.
- **Near-miss #3's disk state MIS-claimed as anything other than
  `COMPLETE`** (11 canonical, one seed `CRASHED`-`CRASHED`, no
  `GATE-REFUSED` anywhere): claiming `EXHAUSTED-BUDGET` — `realized≈
  8.0h` is not `>13.80` — **FAILS**; claiming `STOPPED-BY-OPERATOR` —
  excluded outright — **FAILS**. (Claimed as `COMPLETE` with
  `incomplete_at_K` disclosed, it correctly PASSES — see the reportable
  list above; this is the fix, not a hole — `validity_check` verifies
  disk evidence for the CLAIMED label, it does not re-derive the
  uniquely correct label from raw disk state, which is G4's job, not a
  job-routing gate's.)

*Adversarial JSONs from R7's own audit (KW8.1/KW8.4/KW8.5), re-traced
against the §R7 J1/J4/J5 clauses — killed by name:*
- **ENHANCED no-op** (`run_status="COMPLETE"`, `attempts=[]`,
  `realized_gpu_h_final=0.0`, 0 canonical files, `band.incomplete_at_K
  =[26,28,30]` — KW8.1's own evidence payload; a byte-identical second
  payload swaps in `band.interval_resolved_Ks=[26,28,30]` instead,
  traced identically): `COMPLETE`'s OTHERWISE branch fires (both
  fields name all three K's); the per-K filesystem clause PASSES
  vacuously (`0<4` holds for all three, nothing is named in NEITHER
  field to require `==4`) — **but §R7 J1's new clause (a), "every
  primary `(K,seed)` pair has ≥1 row in `ledger.attempts`," reads 0 of
  12 required pairs covered against `attempts=[]`** — **FAILS.** (Under
  §R6's un-amended OTHERWISE branch this payload PASSED — the FATAL
  regression KW8.1 found and this revision closes; §A6-ADJUDICATION had
  named "the no-op rejection" SETTLED, and this is the same hole,
  reopened by a branch that lacked `COMPLETE-DEGRADED`'s own evidence
  clause, now restored.)
- **Suspect run mislabelled plain `EXHAUSTED-BUDGET` (§R8 K4 — REBUILT
  self-consistent, closing KW9.4's MAJOR: the prior payload's own
  numbers did not close — `9×1.20+3×0.0=10.80`, not the claimed
  `14.40`, so it died on universal assertion 3's bookkeeping check
  before ever reaching the J4 clause it existed to exercise, giving
  ZERO coverage of that clause).** 12 primary `(K,seed)` pairs: 9
  single-attempt `CRASHED-RECOVERED` (`attempt_n=1`, `1.20` each,
  `ceiling_charged=true` → `9×1.20=10.80`); 1 pair retried and crashed
  again (`attempt_n=1` AND `attempt_n=2`, both `CRASHED-RECOVERED` at
  `1.20`, `ceiling_charged=true` → `2.40` — this is the pair
  contributing the 10th and 11th `CRASHED-RECOVERED` **row** over only
  10 **pairs** (9 single-row pairs + this 1 double-row pair), spelled
  out here precisely because an ambiguous row-vs-pair count is exactly
  the class of error this fix is closing); 1 pair `COMPLETED`
  (measured, `elapsed_h=1.00`, `ceiling_charged=false`, 1 canonical
  file on disk); 1 pair `GATE-REFUSED` at `attempt_n=1`
  (`elapsed_h=0.0`, `ceiling_charged=false`) — `9+1+1+1=12` pairs,
  `9+2+1+1=13` rows, `11` of them `CRASHED-RECOVERED`. Self-reported
  `charged_vs_measured`: `ceiling_charged_gpu_h=13.20`
  (`=10.80+2.40`, matching the 11 `ceiling_charged=true` rows exactly —
  so universal assertion 8, §R8 K3, PASSES: nothing to catch here),
  `realized_gpu_h_final=14.20` (`=10.80+2.40+1.00+0.0`, so universal
  assertion 3 PASSES: `|14.20-14.20|=0`), `ceiling_charged_fraction=
  13.20/14.20` **exactly** (`0.9296` to 4 dp — §R9 m4, closes
  KW10.5: the declared field is the EXACT quotient, never the
  rounded 4-dp literal, which trips U8's `1e-6` tolerance). Claimed
  `run_status="EXHAUSTED-BUDGET"`: all
  three base clauses hold (`14.20>13.80` ✓, canonical count `1<12` ✓,
  the `GATE-REFUSED` pair's row has `attempt_n==1` ✓) — **but §R7 J4's
  clause, `ceiling_charged_fraction<=0.50`, reads `13.20/14.20>0.50`**
  —
  **FAILS on J4's clause, exactly as intended, with U3 and U8 both
  PASSING first** (traceable: U1 ✓, U2 ✓, U3 ✓, U8 ✓, base clauses
  1-3 ✓, J4 clause ✗). **A6' — the same ledger, correctly claimed
  `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`** instead: the same three base
  clauses hold, PLUS the mirror clause `ceiling_charged_fraction>0.50`
  → `13.20/14.20>0.50` ✓ — **PASSES**, evading nothing (this exact disk
  state correctly routes to `completed/` only under the `-SUSPECT-
  OVERCHARGE` label — KW8.4's MAJOR, still closed, now with a negative
  test that actually exercises the clause protecting it).
- **Fabricated conditional arm** (a genuine 12/12 `COMPLETE` primary
  run whose report additionally carries `conditional={"launched":true,
  "per_seed":[],"qualifier_band":"SLOW-CONVERGENCE-AT-160K"}` with
  ZERO conditional canonical files and ZERO conditional ledger rows):
  `qualifier_band` is non-`None`, so universal assertion 7 (§R7 J5)
  fires — clause (a) requires `launched==True` AND 4 conditional
  canonical files; this payload has 0, so `0==4` is **False** — clause
  (a) fails; clause (b) requires `launched==False`, but this payload
  claims `launched==True` — clause (b) also fails; NEITHER holds —
  **FAILS universal assertion 7**, rejected before any per-`run_status`
  branch is even reached. (Under §R6 — which had NO conditional-arm
  disk-evidence assertion anywhere — this payload PASSED outright,
  reporting a fabricated ≤9.248-GPU-h-priced 160K datum with zero
  supporting evidence — KW8.5's MAJOR, now closed.)

*Adversarial JSONs from R8's own audit (KW9.1/KW9.2/KW9.3/KW9.7),
re-traced against the §R8 K1/K2/K3 clauses — killed by name:*
- **Zero-cost no-op via 12 `GATE-REFUSED` rows, claimed `COMPLETE`**
  (`run_status="COMPLETE"`, `band.incomplete_at_K=[26,28,30]`, 0
  canonical files, `ledger.attempts` = 12 `GATE-REFUSED` rows at
  `elapsed_h=0.0`, one per primary pair): the OTHERWISE branch's
  original two clauses both PASS vacuously (every pair has ≥1 row ✓;
  `0` canonical `==` `0` distinct `COMPLETED` rows ✓) — **but §R8 K2's
  new clause, `len({...COMPLETED primary pairs...}) >= 1`, reads `0 >=
  1` as False** — **FAILS.** (Pre-Rev-8, this payload PASSED and
  routed a 0-GPU-h, nothing-ever-dispatched run to `completed/` — the
  same hole KW8.1 named, reopened one layer down: J1 required a row,
  not work. KW9.2's MAJOR, now closed.) The identical ledger claimed
  `COMPLETE-DEGRADED` fails the same way, on the mirror clause added to
  that branch.
- **Payload B2 — A6's ledger with `ceiling_charged_fraction`
  MIS-DECLARED `0.20`, `ceiling_charged_gpu_h` declared CORRECTLY**
  (same disk state as the corrected A6 above — 11 `CRASHED-RECOVERED`
  rows summing `13.20`, `realized=14.20`, TRUE fraction `13.20/14.20`
  exactly, `0.9296` to 4 dp — the report's own `charged_vs_measured`
  block declares `ceiling_charged_gpu_h=13.20` (matching truth) WITH
  `ceiling_charged_fraction=0.20` (wrong)): `validity_check`'s OLD
  clauses read
  the fraction field as given — `0.20<=0.50` — and PASS as plain
  `EXHAUSTED-BUDGET`, evading `-SUSPECT-OVERCHARGE`'s binding
  resubmission protection on a ledger that is 93% environment-fault
  noise. **§R8 K3's new universal assertion 8 recomputes
  `ceiling_charged_gpu_h` from `ledger.attempts` directly
  (`sum(elapsed_h for rows with ceiling_charged=true) = 13.20`,
  matching the declared value — the `ccgh` half PASSES), then
  recomputes `ceiling_charged_fraction = 13.20/14.20 = 0.9296…` and
  finds `abs(0.9296… − 0.20) > 1e-6`** — **FAILS on universal
  assertion 8's FRACTION half**, before the per-`run_status` branch
  (and its now-moot
  J4 clause) is ever reached. (KW9.3's MAJOR, now closed: the field is
  recomputed, never trusted.) **The sibling payload B2'** (§R9 m1 —
  disclosed here, closes KW10.2's crossed narration: the prior text
  described B2''s own mechanism under B2's name) instead declares
  `ceiling_charged_gpu_h=2.84` WITH `ceiling_charged_fraction=0.20`
  TOGETHER — internally self-consistent (`2.84/14.20=0.20` exactly)
  but both wrong against the true ledger: assertion 8's `ccgh` half
  recomputes `13.20`, `abs(13.20−2.84) > 1e-6` — **FAILS on the `ccgh`
  half**, before the fraction half is even reached.
- **The paid conditional arm silently absent from the ledger** (a
  genuine 12/12 `COMPLETE` primary run, `conditional={"launched":true,
  "per_seed":[0,1,2,3],"qualifier_band":null}`, 4 conditional canonical
  files genuinely on disk — up to `2.32×4=9.28` GPU-h of real
  conditional spend — but ZERO conditional rows in `ledger.attempts`):
  pre-Rev-8, `qualifier_band is None` meant universal assertion 7 was
  silent altogether, and no per-`run_status` branch reads the
  conditional tree — **PASSES**, a real spend invisible to the ledger.
  **§R8 K1's mirror clause fires** (`launched==True`, `qualifier_band
  is None`): conditional canonical count (4) is compared against the
  ledger's distinct conditional `COMPLETED` count (0) — `4 != 0` —
  **FAILS.** (KW9.7's MINOR, closed by the same clause that fixes
  KW9.1, exactly as the report's own discharge condition predicted.) A
  second adversarial variant — the SAME 4 conditional canonical files,
  but this time WITH 4 matching ledger rows, still `qualifier_band=
  null` — also correctly **FAILS**, on the `n_cond_canon < 4` half of
  the same clause (4/4 conditional completion with no band contradicts
  §5's own now-unconditional "4/4 always reports a band" rule).

**No per-cell job specs are created** — the orchestrator is the only
pool artifact this design produces (§6).

**KW2.8/KW3.13/KW4.6 close-out — the REFUTED accepted-risk replaced
with a real, owned gate (F3, Rev 3).** Round 3 found the "d=K+1
micro-smoke... recorded above" cross-reference FALSE: no live section
specified it (only `§R1`, frozen and non-operative by house
convention); `validity_check` (above) is a harvest-time assertion on a
COMPLETED production cell, not a smoke test, and cannot catch a
first-forward-pass shape crash the way a smoke test does — it is never
called a smoke anywhere in this document. Rev 3 specifies a real
per-K micro-smoke as an explicit BUILD-RELEASE GATE, distinct from and
run strictly BEFORE any production cell:

- **What runs.** One short `d`-override cell per K∈{26,28,30}:
  `ncr_earlyln_scale.py --cell --K {K} --d-override {K+1} --seed 0
  --steps 500 --ceiling-gpuh 0.05
  --outdir /home/nvidia/ncr/results_kwall_smoke/K{K}` (absolute path,
  KW5.11; the `t10` micro-cell pattern `§R1`'s frozen KW2.8 row already
  named as the right shape, now actually wired to a live gate instead
  of left inside a frozen, non-operative section). 500 steps exercises
  one full forward/backward pass and one optimizer step at each new
  K's actual `d=K+1` shape; it is not expected to converge and is not
  scored on `indist_min`. **Budget and placement, disclosed (KW5.10,
  Rev 4).** 3 cells × `0.05h` ceiling = `≤0.15 GPU-h`, run BEFORE the
  orchestrator is queue-eligible and therefore outside the
  orchestrator's own `15.50h` ledger and outside any pool spec — total
  disclosed program spend is `≤15.50 + ≤0.15 = ≤15.65 GPU-h`. The 3
  micro-smokes run on a GPU verified idle by the build/red-team stage
  (§6) before the orchestrator job is promoted to the pool — they are
  NOT covered by `queue_worker.sh`'s free-GPU gate (`:107-115`), which
  governs pool jobs only, so this is a manual pre-launch check, not an
  automatic one.
- **Pass criterion (exact, build-checkable).** The subprocess exits
  without an uncaught exception, AND the resulting JSON has
  `status ∈ {"COMPLETED","ABORTED-BUDGET"}` (either is fine — a smoke
  only proves the config RUNS, never that it converges), AND `K`, `d`,
  `d_override` in the JSON equal `K`, `K+1`, `K+1` respectively (the
  shape actually built is the one asked for, not a silently-defaulted
  `d=2K`).
- **Gate placement.** Build-release gate: the orchestrator script is
  not queue-eligible until all 3 micro-smokes (K=26,28,30) pass —
  specified here precisely enough for the build stage to implement
  without further design input, the same deferral precedent already
  accepted for the job-spec template above (design-complete,
  implementation deferred to build, not a design gap).
- **Populating `orchestrator_report.json`'s `smoke` block (§R5
  KW6.13 — the field was previously unfillable: the 3 micro-smokes run
  as a manual build/red-team gate BEFORE the orchestrator ever
  executes, and the orchestrator was given no way to learn their
  results).** At startup, BEFORE any gate check or dispatch, the
  orchestrator reads
  `/home/nvidia/ncr/results_kwall_smoke/K{K}/earlyln_K{K}_s0.json` for
  each `K∈{26,28,30}` and applies the Pass criterion above; `smoke[
  "K{K}"]` is set to `"PASS"`/`"FAIL"` accordingly. If any file is
  missing, unparseable, or fails the criterion, the orchestrator
  REFUSES TO DISPATCH ANYTHING and exits before touching the ledger —
  the same gate the manual build/red-team check already enforces,
  now also enforced by the orchestrator's own startup path so
  `validity_check`'s `all(v=="PASS" for v in smoke.values())`
  assertion (above) is checking a field that is always populated by
  the time a report exists.
- **`t4b`'s K-list extension** (the OTHER half of R1's original ask)
  stays deferred to the build-stage smoke-test checklist, same
  disposition as Rev 1/Rev 2 — a build-time task, not this draft's own
  claim about K∈{26,28,30}'s trainability. The +3 default-`d=2K` smoke
  cells the build's own `--smoke`/t5 path already runs remain harmless
  extra coverage of an already-rejected convention (§2(a)/(b)), never
  a substitute for the gate specified above, which is this design's
  own owned smoke for its own config family.

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
- **Rev 2 (KW3.12), zero-cost transparency addition.** `harvest()`
  already emits `gate1.A_eff_rank_mean` and the `AEFF_RANK_FRAC_BAR`
  per cell at no extra compute cost; the report table for this design
  includes these two raw fields alongside `indist_min` for every
  harvested cell. This does not separate the two legs into distinct
  gates (still not attempted, per D4's binding narrowing above) — it
  only makes both legs' raw numbers visible in the same table a reader
  already sees `indist_min` in, closing the residual at zero
  incremental GPU-h.

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
(D2), and Rev 2 REGENERATES this table by EXECUTION (E5, KW3.5).**
Round 2 found the printed table did not reproduce from the six-rule
procedure + the `[NON-MONOTONE]` tag rule exactly as printed (2 of 125
outcomes landed in a row the table omitted, and the representative
row `(2,4,2)` carried the wrong tag). Rev 2 does not change the rule
text — the six numbered rules above and the tag rule below are
UNCHANGED from Rev 1 — it re-executes them and reports what they
actually produce. The decision procedure was run this revision as a
15-line Python function (`classify(r26,r28,r30)` implementing rules
1–6 verbatim, plus the ROBUST-sequence monotonicity check for the
`[NON-MONOTONE]` tag) against all `5³=125` reachable `(r26,r28,r30)`
outcomes:

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
| NON-MONOTONE-UNRESOLVED | 4 |
| **NON-MONOTONE-UNRESOLVED [NON-MONOTONE]** | **2** |
| **Total** | **125 / 125** |

**What changed vs. Rev 1's printed table (KW3.5's discharge, option
(b) — add the missing row, fix the mislabeled row; the tag rule itself
is untouched):** the `NON-MONOTONE-UNRESOLVED` band splits into two
rows because 2 of its 6 members — `(2,3,2)` and `(2,4,2)` — have
ROBUST-sequence `[True,False,True,False,False]`, which is genuinely
non-monotone, so the SAME tag rule that already applies to every other
band applies to them too: `NON-MONOTONE-UNRESOLVED` proper now has 4
members (`(2,0,1) (2,0,2) (2,1,2) (3,4,2)`) and
`NON-MONOTONE-UNRESOLVED [NON-MONOTONE]` has 2 (`(2,3,2) (2,4,2)`).
`GRADUAL-DECAY` is unaffected by construction (`r26≥r28≥r30` forces a
monotone boolean sequence, so it never needed the tag) — this is why
only the residual band moved.

Representative rows (covering every band, including the exact cases
the audit named; `(2,4,2)`'s tag corrected):

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
| `(2,4,2)` | **NON-MONOTONE-UNRESOLVED [NON-MONOTONE]** (corrected this revision — Rev 1 listed this untagged, which contradicted the design's own tag rule; ROBUST-sequence `[T,F,T,F,F]` is non-monotone) |
| `(3,2,1)` | GRADUAL-DECAY |

Any auditor can re-run the six-rule procedure above against all 125
outcomes to re-check this table; it is a ~15-line function, not a
large artifact, and is reproduced in full above (not merely
referenced). **Row-count checksum: 10 band-rows, Σ=125/125, matching
the six-rule-plus-tag procedure executed exactly as printed — this
revision closes E5/KW3.5 by actually running the enumeration, not by
hand-editing counts.**

**The CONDITIONAL 160K disambiguator's report — Rev 2 correction (E3,
KW3.3/KW3.6).** Round 2 found the `$0` K=32 reuse branch scored at a
DIFFERENT budget than the paid branch: the paid branch produces exactly
one new datum (a 4-cell rate AT 160K), but Rev 1's `K_trig=32` case
substituted K=32's ARCHIVED 320K rate to produce a label
(`PARTIAL-IMPROVEMENT`) the paid branch's own rule, applied honestly at
the matched 160K budget, does not support — K=32's actual 160K rate is
1/4, which the pre-registered rule below reads as
`CONFIRMED-WALL-AT-160K`, not `PARTIAL-IMPROVEMENT`. Rev 2 discharge
option (i) from the audit: apply the rule honestly at matched budget
everywhere, and disclose the 320K datum as separate context, never as a
branch output. If the trigger (§4) fires at `K_trig∈{26,28,30,32}`
(32 now included on the SAME footing as the paid K's, not a special
case — see below), its 4-cell **160K** rate — paid for `K_trig∈{26,28,30}`,
already-archived for `K_trig=32` — is reported ALONGSIDE the PRIMARY
80K classification above as a budget-verdict qualifier at `K_trig`,
never substituted into the 80K label itself, and NEVER computed from a
320K datum:

**For the PAID branch (`K_trig∈{26,28,30}`), the qualifier band is
reported ONLY on 4/4 conditional completion (§R8 K1 — closes KW9.1's
FATAL: a prior scope note in `validity_check`, below, wrongly called
this "hypothetical future"; it is G4's own pre-registered
`COMPLETE-DEGRADED` sub-case (ii)/(iii), not a future case). The `$0`
`K_trig=32` archive branch below is NOT subject to this 4/4 gate — by
construction it has zero conditional completions; it reports its band
unconditionally, straight from the already-archived table, per U7
clause (b) below (§R9 m2 — closes KW10.3's contradiction against this
same pre-registered $0 branch, three sentences below).** If the
conditional arm is THROTTLED — the hard/retry gate
refuses 1-4 of the 4 conditional cells' first attempts (G4 sub-case
(ii)), or refuses a conditional cell's retry (G4 sub-case (iii)) —
before all 4 reach `COMPLETED`, the run reports `conditional.
qualifier_band = null`. The cells that DID complete are disclosed as
DATA ONLY (their raw per-seed `gate1`/`indist_min` records, alongside
the `COMPLETE-DEGRADED` primary verdict and the throttle evidence in
`ledger.attempts`) — never rounded into one of the three named bands
above, which are pre-registered against a full `n=4` rate exactly as
the runner's own S9.5 hard rule already treats any sub-4 rung
elsewhere (`ncr_earlyln_scale.py:397-403`, "a sub-4-seed rung is NEVER
gated as CONVERGED-ROBUST/TRAINABILITY-DEAD ... it is disclosed data
only" — the identical policy applied here to the conditional arm's own
160K rung). This is a genuinely reportable, publishable outcome (per
"every band is informative," below) — a legitimate ≤15 GPU-h run that
happens to land here is never routed to `failed/` for having reported
it honestly (`validity_check`'s mirror clause, below, is the
enforcement point).
- **CONFIRMED-WALL-AT-160K:** `K_trig`'s rate stays `≤1/4` at 160K —
  the strongest evidence this design can produce that the drop is
  architectural, not merely slow. **Gloss corrected (KW3.6):** this
  includes the case `0/4→1/4`, which IS some improvement, just not
  enough to matter — reworded from Rev 1's "(no material improvement)",
  which was literally false for exactly this transition; the correct
  reading is "does not clear CONVERGED-ROBUST at 160K," not "does not
  improve at all."
- **SLOW-CONVERGENCE-AT-160K:** `K_trig`'s rate reaches `≥3/4`
  (CONVERGED-ROBUST) at 160K — the frontier moves with budget;
  recommend the flagship's "last live rung" consider `K_trig` a
  BUDGET-CONDITIONAL candidate (never unconditional) and flag a 320K
  confirmation as future PI-gated work (§7 non-goal, unchanged).
- **PARTIAL-IMPROVEMENT-AT-160K:** `K_trig`'s rate improves but lands
  at exactly `2/4` (not `≤1/4`, not `≥3/4`) at 160K — reported as
  genuinely ambiguous, motivating (not resolving into) a 320K
  follow-on. **This band is reachable ONLY from a 160K datum reading
  2/4 — never by importing a 320K figure (E3's fix: the `/320K` label
  suffix Rev 1 used is deleted; no band in this design is ever decided
  by a budget the paid branch does not produce).**

**At `K_trig=32`:** the qualifier is now read from the SAME
ALREADY-ARCHIVED table (§3), at the MATCHED 160K row only — K=32's
160K rate is **1/4** (only seed 1 clears, `indist_min=0.9015`), so by
the rule above K=32 is **`CONFIRMED-WALL-AT-160K`** — reported at $0
incremental cost, per §4, on the exact same footing as a paid
`K_trig∈{26,28,30}` cell reading 1/4 at 160K would be. **Disclosed
separately, as archive context ONLY, never as a band determinant:** the
SAME table shows a further rise to 2/4 at 4× (320K,
`EXPERIMENT_LOG.md:8887-8896`, "Not established: whether an even
larger budget... would behave differently") — this is additional
information a reader may want, not part of the pre-registered
160K-matched decision, and it is never substituted into the label the
way Rev 1 did.

**INCOMPLETE-AT-K (D5/E4, Rev 2 mechanization, KW2.2/KW2.3/KW3.4; Rev 3
KW4.8 wording fix).** Rev 1 said an incomplete K is "re-run... until
4/4 COMPLETED, then classified" — this deadlocks against a TERMINAL
`PERSISTENTLY-ABORTED` seed (§4's bounded-retry rule), which by
construction can never reach 4/4 COMPLETED. Rev 2 replaces this with
§4's mechanized rule (E4, cross-referenced here rather than restated
in full): a K with exactly one `PERSISTENTLY-ABORTED`/MISSING/
`GATE-REFUSED` cell is resolved by INTERVAL LOGIC (evaluate the
six-rule procedure at both possible values of the unresolved seed;
same band ⇒ decide with disclosure, different bands ⇒ the study
reports `INCOMPLETE-AT-K`); a K with ≥2 incomplete cells, or multiple
K's whose cross-product of interval candidates disagrees, likewise
yields `INCOMPLETE-AT-K`. **`INCOMPLETE-AT-K` is a STUDY-LEVEL verdict,
not a per-K band (KW4.8) — §5's six-rule procedure is a function of
the FULL `(r26,r28,r30)` triple and returns exactly ONE label; there is
no per-K band object for it to attach to.** It sits orthogonal to the
125-outcome partition demonstrated below (that partition is exhaustive
over its OWN domain — complete triples — and stays exact;
`INCOMPLETE-AT-K` is what the study reports when no complete triple can
be read at all), reported, disclosed with the affected K(s) carried as
a field, and explicitly EXCLUDED from frontier claims — never silently
forced into a band or silently dropped from `n_seeds` (the denominator
stays fixed at 4 throughout, per the A4.9 guard). No partial rate is
ever read into the table as if it were a full-n=4 verdict. **Related
to, but NO LONGER independent of, `TRIGGER-UNRESOLVED` (§4) — corrected
this revision, G5, closing KW5.5.** `classify()` and the trigger's
K-scan remain different functions of the same per-K resolution states,
and the reverse direction stays genuinely unreachable (the trigger
scan only reaches an unresolved K when it needs that K's own status,
per §4's pseudocode, so `TRIGGER-UNRESOLVED` while the band DECIDEs
is 0/1000 over the full reachable state space — re-verified this
revision). **But the forward direction — "a K can be
band-`INCOMPLETE-AT-K` while the 160K trigger still resolves" — was
TRUE through Rev 3 and is FALSE as of Rev 4's G5 precondition (§4):**
the trigger now checks the SAME whole-study band before dispatching,
and forces its own result to `TRIGGER-UNRESOLVED` whenever that band
is `INCOMPLETE-AT-K` (371 of 1000 reachable state vectors were exactly
this case, re-executed this revision — §4's G5 subsection). The two
computations are therefore no longer fully independent: the trigger's
`DECIDED` result now IMPLIES the band decides too, though the band
deciding does not by itself guarantee the K-scan reaches a `DECIDED`
`K_trig` (the K-scan can still separately return `TRIGGER-UNRESOLVED`
via its own `blocking_K` path, unaffected by G5).

**KW2.9 (Rev 2, now fully discharged — the one sentence R1 asked for).**
`harvest()`'s existing report format also emits `SUB4-DISCLOSED-ONLY(n=0)`
rows for every K in the shared `GRID_SHAPES` dict that lies OUTSIDE
this design's own K∈{24,26,28,30,32} outcome space — i.e. every OTHER
design's K value tracked in the same shared dict. A harvest reader of
THIS design's report should read only the K∈{24,26,28,30,32} rows;
the `SUB4-DISCLOSED-ONLY(n=0)` rows for unrelated K's are a harness-wide
artifact of a shared dictionary, not a signal about this
characterization, and are not part of this design's outcome space.

**Every band is informative and reportable** — none requires a
follow-on wave to be publishable as a characterization result; every
FRONTIER-AT-K\* outcome localizes the 80K frontier to a resolution
better than the current archive; GRADUAL-DECAY produces the K-ladder's
own missing scaling-curve data point; NON-MONOTONE-UNRESOLVED is
itself a legitimate, publishable pre-registered outcome (an
unclassifiable result was the defect this design no longer has).

---

## §6 POOL-ELIGIBILITY STATEMENT

**Rev 3 (F1): pool-eligibility attaches to the ORCHESTRATOR SPEC, not
to the 16 cells.** This is the structural fix to KW4.3's finding that
§6 asserted both "16 independent pool-eligible cells" AND "a
cumulative program cap across them" — mutually exclusive under the
pool's own contract. Rev 3 resolves the contradiction by changing which
object the contract is checked against: ONE job, "run the
orchestrator," is what ships to `queue/jobs/pending/` (job-spec
template, §4, above); the 16 cells (12 primary + up to 4 conditional)
are dispatched BY that job, sequentially, on the one GPU it occupies —
entirely internal to its execution, invisible to and unenforced by
`queue_worker.sh`/`idle_fallback_daemon.sh`. This grid satisfies
`matrix-thinking/queue/idle_fallback_daemon.sh`'s own pool contract
(header, `:10-16`): *"the pool holds ONLY flat specs — each fully
audited + queue-eligible, independently runnable in any order,
carrying its own cost ceiling, with NO intra-wave dependencies, stage
gates, or staged-escalation semantics."* Verified against the
orchestrator spec's own structure, not asserted:
- **Independent — genuinely, not "with one honest exception."** Rev 2
  disclosed the conditional arm's dependency on the primary harvest as
  a pool-level exception to flatness (KW4.3's structural complaint was
  exactly that this exception contradicted the flat-spec claim
  alongside it). Under F1 that dependency no longer touches the pool
  at all: the primary→trigger→conditional sequencing (§4's cell order,
  trigger evaluation point, harvest invocations) is entirely INTERNAL
  to the one orchestrator job. From the pool's perspective there is
  exactly one flat, independently-runnable-in-any-order item — nothing
  about it depends on any OTHER pool spec's state or result. (Unlike
  the SPENT K-ladder design's multi-stage pool submission,
  `NCR_KLADDER_ATTACK_R2.md` finding A4.12, which this design still
  does not repeat.)
- **Own cost ceiling (F1, replaces E1/E2's launcher-side cumulative
  check, which KW4.1–KW4.3 found blind/false-premised/unimplementable
  in the real dispatch path).** The orchestrator spec carries exactly
  ONE declared ceiling: **15.50 GPU-h** — `15.20h` (the disclosed,
  conservatively-rounded figure for the tight LEDGER bound, which the
  derivation proves is `15.0157h`, §4 — Rev 4's corrected figure,
  extending Rev 3's `15.0126h` to cover the mid-attempt-crash case, G1,
  and the KW5.7-corrected single-attempt tail; **§R5 H3 originally
  derived a TRUE-spend worst case of `15.2041h`; §R7 J6 found that
  figure incomplete (Class-1/ceiling-charged leaks only) and
  re-derived it two-class, `15.3737h` — `0.1737h` above the disclosed
  `15.20h` — still comfortably inside the `15.50h` total, §4 "True
  spend, worst case"**) plus a **stated 0.30h supervisor margin** (§4,
  covering three disclosed terms — contention variance, the RESIDUAL
  process-startup estimation uncertainty (most of this term is now
  explicitly priced per-row via the `s=0.0053` startup allowance, §R7
  J6, not merely margin-absorbed), and the `15.3737h` shortfall). This is
  enforced ENTIRELY inside the
  orchestrator's own process (the HARD/RETRY gates and the wall-clock
  ledger, §4) — no external launcher, no cross-cell coordination, and
  no dependence on `queue_worker.sh`/`idle_fallback_daemon.sh` carrying
  any budget state (they carry none, and now need none: KW4.3's
  finding that neither script has a notion of "batch" or "cumulative
  spend" is moot, because the ONE thing the pool dispatches doesn't
  need either from them). Per-cell CLI ceilings (`1.20`/`2.32`) are
  still enforced by the runner's own existing training-phase mechanism
  (`train_earlyln_cell`'s `ceiling_s` argument, `:198-201`) AND are now
  the SAME values the orchestrator's ledger charges (KW4.4 closed, §4).
  Disclosed total program spend, INCLUDING the micro-smokes (KW5.10,
  §4): `≤15.50 + ≤0.15 = ≤15.65 GPU-h`.
- **Audited + queue-eligible only after this draft clears its own
  audit round** — still explicitly NOT queue-eligible (status header,
  now DRAFT-R4); the pool contract's "ceremony gate stays upstream of
  it" applies here exactly as written.
- **Queue-pool sweep scope, corrected (KW2.7, unchanged from Rev 2).**
  §3's internal sweep covered `matrix-thinking/queue/jobs/pending/`
  (the only queue directory tracked in this repo) and found zero
  K∈{26,28,30} hits; `~/queue/{fallback_pool,claimed}` on the box were
  NOT swept this session. **Still a mandatory pre-launch red-team
  task:** sweep both on-box directories for K∈{26,28,30} content before
  this design's orchestrator job is promoted to the pool.
- **Resource/placement red-team, restated for the orchestrator model
  (Rev 2, KW3.16; Rev 3 repoints items (i)/(iii); Rev 4 repoints (ii)
  and adds (vi)–(ix) for G1–G5; Rev 5, §R5 H6, adds (x)–(xiv) for
  H1/H2/H4/H5/KW6.17).** CLAUDE.md's ceremony tiers require a
  pre-launch resource/placement red-team for any 10–50 GPU-h wave;
  this design's declared ceiling (15.50h) sits in that tier. Before
  launch, the red-team round must: (i) verify the build-stage
  orchestrator script's ledger/gate implementation against the
  ORCHESTRATOR CONTRACT (§4) EXACTLY — cell order, dispatch loop, gate
  check points, attempt-indexed outdirs, ledger persistence and the G1
  write-ahead/recovery procedure — not just against this design's
  prose in the abstract; (ii) verify G2's copy-on-accept +
  exists-check-fails-loudly discipline is actually applied (no
  `harvest()` code patch is needed under G2, §4 — the red-team confirms
  the CANONICAL directory stays flat and one-file-per-cell, not that a
  patch exists); (iii) confirm the orchestrator genuinely occupies
  exactly ONE GPU with no concurrent cell dispatch anywhere in its own
  process (the property the whole worst-case derivation, §4, depends
  on — a build that silently parallelizes cells across GPUs "for
  speed" would reopen KW4.2's hole); (iv) verify the 3 build-release
  micro-smokes (F3, §4) actually ran and passed before the orchestrator
  was marked queue-eligible; (v) complete KW2.7's still-outstanding
  on-box `fallback_pool/`/`claimed/` sweep; **(vi) verify G1's recovery
  procedure with a synthetic test — kill the orchestrator process
  mid-attempt (e.g. `SIGKILL` during a real `subprocess.run`), restart
  it, and confirm the ledger shows exactly one `CRASHED-RECOVERED` row
  charged at the FULL ceiling, not a silent gap or a double-charge;
  (vii) verify G3's exit-code branch with a synthetic `--stop-file`
  touch mid-run — confirm the orchestrator reports
  `STOPPED-BY-OPERATOR` and dispatches NOTHING further, never
  `ABORTED-BUDGET`/`PERSISTENTLY-ABORTED`; (viii) verify G4's
  `validity_check` accept-set matches §4 exactly (`COMPLETE`/
  `COMPLETE-DEGRADED`/`EXHAUSTED-BUDGET`/
  `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE` → `completed/`,
  `STOPPED-BY-OPERATOR` → `failed/`, §R5 H4/H5); (ix) verify G5's
  trigger precondition is wired — a synthetic state where the K-scan
  alone would return `DECIDED` but the whole-study band is
  `INCOMPLETE-AT-K` must produce `TRIGGER-UNRESOLVED` with a disclosed
  `band_blocked_K_trig`, never a dispatched conditional arm; **(x, §R5
  H1) truncate/corrupt `ORCHESTRATOR_LEDGER.json` mid-file and restart
  — confirm CONSERVATIVE RECONSTRUCTION fires (recovery step 0) and
  the run does NOT resume at `realized_gpu_h=0`; (xi, §R5 H2) kill the
  orchestrator in the copy-then-fold window specifically — after the
  archival JSON reads `COMPLETED` but before/mid/after the canonical
  copy — and confirm each of the four crash-window outcomes above,
  not only the pre-Rev-5 "before copy" case; (xii, §R5 H4) run a cell
  through a genuine `GATE-REFUSED` (e.g. a deliberately tiny
  `--hard-gate-cap` in a throwaway config) and confirm the resulting
  report PASSES `validity_check` (a `GATE-REFUSED` row must not itself
  cause a false `failed/` routing); (xiii, §R5 KW6.17) confirm recovery
  reaps/verifies the assigned GPU is free before closing a dangling
  `open_attempt` — simulate an orphaned CUDA process surviving a
  parent-only kill and confirm recovery aborts loudly rather than
  re-dispatching onto it; (xiv, §R5 H5) synthetically force
  `ceiling_charged_fraction > 0.50` (repeated forced crash-recoveries)
  and confirm the report reads `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`,
  not plain `EXHAUSTED-BUDGET`, and that no automatic resubmission
  occurs.**
- **No standing restriction bites.** The `STATE.md:53` "NO NCR job
  queue-eligible" restriction (line renumbered this revision — content
  unchanged, KW3.7) is scoped to the in-LM write-conditioning claim
  pivot; this design makes no in-LM claim and no claim pivot — it
  characterizes an already-cleared toy-scale mechanism (S11 earlyln
  free-write; NCR core mechanism NOVEL per
  `research/novelty-gate-2026-07-27.md`) at new K values, the same kind
  of additive K-extension the 2026-07-11 queue-system build already
  did without a fresh novelty gate. **Per KW2.10's discharge condition:
  this reading is not this design's ruling to make** — it needs a
  coordinator record in STATE.md/EXPERIMENT_LOG at adjudication time,
  not self-clearance inside this file. `STATE.md:24-26` (line
  renumbered this revision, KW3.7) already records the coordinator
  routing this document "DRAFT-R0 → audit → adjudication → build →
  pool," consistent with this reading; the formal ruling itself stays
  outside this file, per the audit's own instruction; the KW2.10
  coordinator ruling itself is separately on record at
  `EXPERIMENT_LOG.md`'s 2026-08-06 entry (commit `eaf42e6`), per
  round 2's own verification.

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
- **Only ONE K gets budget-qualified — disclosed explicitly, not left
  implicit (Rev 2, KW3.16).** The CONDITIONAL 160K arm (§4/§5)
  disambiguates speed-vs-wall at exactly `K_trig`, whichever single K
  that is. Every OTHER FRONTIER-AT-K\* label in the same report stays
  an 80K-only read, not budget-qualified by this design — e.g. if
  `K_trig=28`, the design says nothing about whether K=26's own 80K
  rate (already ROBUST, by construction, for rule 3 to have skipped it)
  would also hold at 160K; it simply wasn't measured. This is a scope
  limit of the ≤15 GPU-h cap, not an oversight: a full budget sweep at
  every K would cost roughly 4× this design's cap. A reader of the
  eventual harvest should not read "FRONTIER-AT-K\*=28" as implying
  K=26 was budget-checked too.
- **The ORCHESTRATOR CONTRACT's gates (F1, superseding E1) and E4 (the
  bounded-retry / interval-logic incomplete-cell rule) apply uniformly
  to BOTH arms, stated once here to avoid ambiguity (Rev 2, KW3.16;
  Rev 3 repoints from the retired launcher-side E1 check).** Neither
  rule is primary-arm-only or conditional-arm-only: the orchestrator's
  single ledger and HARD/RETRY gates (§4) govern every dispatch in the
  whole program, in the SAME sequential stream, regardless of which arm
  a cell belongs to — there is no separate ledger, no separate gate,
  and no separate GPU for the conditional arm. A conditional-arm cell
  that hits `ABORTED-BUDGET`, `CRASHED`, or `CRASHED-RECOVERED` (G1/G3,
  Rev 4) is subject to the identical 1-retry-then-`PERSISTENTLY-ABORTED`
  state machine and the identical interval-logic classification
  treatment as a primary cell — there is no separate, weaker, or
  unspecified abort-handling path for the conditional arm. **G5's
  DECIDED-band trigger precondition (§4, Rev 4) is a PRECONDITION on
  whether the conditional arm dispatches at all, not a separate rule
  for how it is treated once dispatched — once triggered, everything
  in this bullet applies unchanged.**
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

## §A2-ADJUDICATION — AUDIT ROUND 2 VERDICT ADOPTED: **REV-REQUIRED** (Fable, 2026-08-06)

`NCR_KWALL_ATTACK_R2.md`: 0 FATAL / 5 MAJOR / 11 MINOR; 13/18 prior
findings DISCHARGED, 4 PARTIAL, 1 declined. Core VERIFIED CLEAN by
independent recomputation (budget wave to 4 decimals; partition
125/125 exhaustive+exclusive; §A1-ADJUDICATION byte-identical to
HEAD~1). The five MAJORs are ops/cost bookkeeping; all forcing.

**BINDING DISPOSITIONS for Rev 2:**
- **E1 (KW3.1):** replace blanket retry-once with a retry rule
  SUBORDINATED to an enforceable cumulative program cap: retries
  permitted only while cumulative realized GPU-h ≤ 12.0, hard program
  cap 15.00 INCLUSIVE of retries, conditional arm, and eval — the
  worst case is bounded by construction (budget-capped, not
  count-capped). State the bound's derivation.
- **E2 (KW3.2):** per-cell ceilings return to the job-108 house
  convention (≥2× nominal, floor 1.0h); the GLOBAL cap (E1) does the
  program bounding, so no per-cell trim is needed. Delete the false
  "within 1.06×" support claim; cite the audit-verified 1.206×
  archive-wide max instead.
- **E3 (KW3.3):** the $0 K=32 reuse branch is scored at MATCHED budget
  (160K vs 160K) under the same band rule as the paid branch; any
  320K-dependent label is unreachable and must be removed from the
  reachable-outcome set (the 320K datum may appear as archive-only
  disclosure, never as a branch output).
- **E4 (KW3.4):** mechanize the incomplete-cell logic: 1 retry max →
  PERSISTENTLY-ABORTED is terminal; denominators stay fixed at 4
  (A4.9 guard preserved); a K with a terminal-aborted/MISSING cell is
  decided by INTERVAL LOGIC — if both possible values of the missing
  cell map to the same band, decide; otherwise INCOMPLETE-AT-K
  (reported, disclosed, excluded from frontier claims). harvest()'s
  treatment must be specified to implement exactly this.
- **E5 (KW3.5):** regenerate the 125-outcome table from the rules as
  written (add the missing NON-MONOTONE-UNRESOLVED row; fix the
  (2,4,2) row's tag); the demonstration counts must reproduce under
  re-execution.
- **E6:** every PARTIAL (KW1.3, KW2.2, KW2.3, KW2.8) and the declined
  KW2.9 gets an explicit close-out row: fully discharged, or
  accepted-risk with a one-sentence justification the next audit can
  attack. No silent leftovers.

Rev 2 → FOCUSED audit round 3 (E1–E6 discharge verification +
partition re-execution + bound recomputation; fresh judge) →
adjudication → build ceremony (spec generation + harvest script get
their own build audit) → placement red-team → pool.

---

## §R2 REVISION 2 (2026-08-06)

Rev 2, dispatched per §A2-ADJUDICATION's binding dispositions E1–E6.
Every finding from `NCR_KWALL_ATTACK_R2.md` gets a row below — all 5
MAJOR (KW3.1–KW3.5) and all 11 MINOR (KW3.6–KW3.16), plus the four
round-1 PARTIALs (KW1.3, KW2.2, KW2.3, KW2.8) and the declined KW2.9,
per E6's "no silent leftovers" instruction. §1–§7,
`§A1-ADJUDICATION`, `§R1`, and `§A2-ADJUDICATION` are UNCHANGED as
historical record EXCEPT where a disposition explicitly required
rewriting a section's content in place — every such rewrite is listed
in the "Where fixed" column below. This design now carries status
**DRAFT-R2 — POST-AUDIT-2, AWAITING FOCUSED AUDIT ROUND 3**.

**Every number introduced or changed this revision was recomputed from
cited raw artifacts or derived by visible arithmetic — none copied
from prose.** The 160K nominal figures were re-derived using the
MAXIMUM (not Rev 1's minimum) of the three archive 2×/1× ratios; every
restored ceiling is `max(2×nominal, 1.0h)`, executed directly; the E1
worst-case bound is derived by induction, not asserted; the 125-outcome
table was regenerated by literally executing the unchanged six-rule
procedure, not hand-edited.

| Finding | Disposition | Where fixed |
|---|---|---|
| **KW3.1 (MAJOR)** — the "15.00h unconditional bound" is broken by the revision's own retry-once rule; true worst case was 30.00h/30.20h | **DISCHARGED.** Blanket retry-once replaced by E1's cumulative-realized-GPU-h program cap: launcher checks `realized_gpu_h` from `COMPLETED` cell JSONs before every launch/retry; hard gate refuses admission once `realized_gpu_h+ceiling(cell)>15.00`; retry sub-gate closes at `realized_gpu_h≥12.00`. Worst case derived by induction on admission order: `≤15.00+16×0.0126=15.2016≈15.20h` — bounded by construction (budget-capped, not count-capped), matching (not exceeding) the pre-existing eval-inclusive disclosure. | §4 (new "E1 — the cumulative-realized-GPU-h program cap" subsection, replaces "Unified ceiling accounting"); §6 (ceiling-mechanism cross-reference) |
| **KW3.2 (MAJOR)** — the 0.75h/1.50h ceiling trim worsens KW2.2's contention risk; the "within 1.06×" safety claim is false (4/24 archive groups exceed it, max 1.092×); violates job-108's own `≥2×nominal, floor 1.0h` convention | **DISCHARGED.** Per-cell ceilings restored to the job-108 convention: primary 1.20h (≥2×nominal for all of K∈{26,28,30}), conditional 2.32h (≥2×nominal for all of K_trig∈{26,28,30}). E1's global cap does the program-level bounding, so no per-cell trim is needed. The false "within 1.06×" sentence is deleted; the TRUE archive-wide max/nominal ratio (1.206×, verified by the audit over 97 cells/24 groups) is cited instead, now leaving `2.00/1.206≈1.66×` of headroom beyond the worst spike ever recorded — wider than Rev 1's margin, not narrower. | §4 (primary + conditional command blocks; new "Ceiling reference table" + "Margin claim, corrected" subsections); §6 |
| **KW3.3 (MAJOR)** — the `$0` K=32 branch is scored against a 320K datum the paid branch never produces; K=32's own MATCHED 160K rate (1/4) gives `CONFIRMED-WALL-AT-160K` under the design's own rule, not the reported `PARTIAL-IMPROVEMENT-AT-160K/320K` | **DISCHARGED** (audit discharge option (i) — apply the rule honestly at matched budget). `K_trig=32` is now scored at its MATCHED 160K rate only (1/4 → `CONFIRMED-WALL-AT-160K`), on the same footing as any paid K. The `/320K` label suffix is deleted from every reachable band; the archived 320K datum (0/4→1/4→2/4) is retained ONLY as disclosed context, never as a branch determinant. | §5 ("The CONDITIONAL 160K disambiguator's report" rewritten in full, incl. the `K_trig=32` case) |
| **KW3.4 (MAJOR)** — D5's retry/`PERSISTENTLY-ABORTED`/`INCOMPLETE-AT-K` rules deadlock, contradict the A4.9 fixed-n=4 guard, and are unmechanized with no named enforcement point against `harvest()`'s actual `gate_eligible` computation | **DISCHARGED.** Retry bounded at exactly 1 (subordinate to E1's retry gate); a second failure is TERMINAL `PERSISTENTLY-ABORTED`, never retried again — no deadlock. Denominator stays fixed at 4 always (A4.9 preserved) — `PERSISTENTLY-ABORTED`/MISSING seeds are not excluded from `n_seeds`; their unknown numerator contribution is resolved by INTERVAL LOGIC (evaluate the six-rule procedure at both possible outcomes of the incomplete seed; same band ⇒ decide+disclose, different ⇒ `INCOMPLETE-AT-K`); ≥2 incomplete cells at one K, or a disagreeing multi-K cross-product, is `INCOMPLETE-AT-K` unconditionally. Enforcement point named: `harvest()`'s `n_seeds`/`gate_eligible` (`ncr_earlyln_scale.py:380-406`) currently counts FILE PRESENCE (`discover_seeds_by_K`'s glob), not `status=="COMPLETED"` — verified by direct code read this revision — and must be patched (build-stage instruction, same deferral precedent as KW2.6's job-spec template) to compute `n_completed` from status before applying the rule above. | §4 (D5/E4 rule rewritten in full); §5 (`INCOMPLETE-AT-K` section rewritten, cross-references §4); §7 (new bullet: E1/E4 apply uniformly to both arms) |
| **KW3.5 (MAJOR)** — the printed 125-outcome table does not reproduce from the rules as written (2 rows missing/mislabeled); `(2,4,2)` contradicts the design's own tag rule | **DISCHARGED.** Rule text is UNCHANGED (six rules + tag rule, verbatim); the table was regenerated by literally EXECUTING that unchanged procedure this revision (15-line function, all `5³=125` outcomes enumerated). `NON-MONOTONE-UNRESOLVED` splits into 4 untagged + 2 `[NON-MONOTONE]` members (`(2,3,2)`,`(2,4,2)` have ROBUST-sequence `[T,F,T,F,F]`, genuinely non-monotone); `(2,4,2)`'s representative row retagged to match. Checksum: 10 band-rows, Σ=125/125. | §5 ("Exhaustiveness and mutual exclusivity" table + representative-rows table, both regenerated) |
| **KW3.6 (MINOR)** — conditional-arm gloss "`≤1/4` (no material improvement)" is false for the `0/4→1/4` case, which IS an improvement | **DISCHARGED.** Reworded to "does not clear CONVERGED-ROBUST at 160K" — correct for every rate in the `≤1/4` range, including `0/4→1/4`. | §5 (same E3 rewrite region) |
| **KW3.7 (MINOR)** — every `STATE.md` line citation is stale by ~13 lines despite the closing note claiming a fresh re-read | **DISCHARGED.** Re-verified against the live file this revision: `:39-40`→`:53`, `:114-116`→`:128`, `:11-13`→`:24-26`. All three corrected in place; quoted text unchanged (content was never wrong, only the line numbers). | §1, §2(a), §6 (two citations) |
| **KW3.8 (MINOR)** — the D3 scope quote is a two-source splice (`EXPERIMENT_LOG.md:8887` + `NOVEL_ARCH_WATERFALL.md:5071`) presented as one paragraph "in full"; waterfall `:5071`'s actual wording differs | **DISCHARGED.** Re-attributed to two separately-quoted, correctly-sourced sentences: `EXPERIMENT_LOG.md:8887-8888` verbatim for the first clause, `NOVEL_ARCH_WATERFALL.md:5071` verbatim (its actual wording, re-read this revision) for the second. "In full" framing removed. D3's ruling is unaffected — it was never sourced from the misattributed splice alone. | §3 (KW1.8/D3 paragraph) |
| **KW3.9 (MINOR)** — eval-overhead range "0.7%–1.5%" is wrong in scope (actual 0.35%–1.58%, n=64); the absolute max (45.5s/0.0126h) was already right | **DISCHARGED.** Percentage range corrected to 0.35%–1.58% (n=64); the max absolute figure (0.0126 GPU-h) is unchanged and is the figure E1's bound derivation actually uses, so the ≈15.20h arithmetic was never affected. | §4 (D5 eval-inclusive paragraph) |
| **KW3.10 (MINOR)** — "AER/K … (0.928–0.966 at 1×)" is wrong at both ends; actual range 0.9269–0.9679 | **DISCHARGED.** Corrected to 0.9269–0.9679, re-derived from the same 4 per-seed `aer_mean/K` values already in the design's own K=32 budget table (§3). | §3 ("Reading the bracket" paragraph) |
| **KW3.11 (MINOR)** — the 2×/1× ratio 1.848 used for 160K nominal is the LOWEST of three available archive ratios (K16 1.9355, K24/d48 1.858, K32 1.8477), understating the 160K nominal by up to 5% | **DISCHARGED.** Switched to the MAXIMUM of the three (1.9355, K16) — the conservative choice, since this ratio now feeds a per-cell ceiling floor (E2) and the program cap (E1). 160K nominal revised: 0.9882/1.0716/1.1561 (K=26/28/30), up ≈4.5–4.8% from Rev 1's figures, all re-derived by direct execution. | §4 (160K nominal table + all downstream ceiling/margin arithmetic) |
| **KW3.12 (MINOR)** — D4's narrowing of KW2.1 dropped leg-attribution, but the residual is closable at zero cost since `harvest()` already emits `gate1.A_eff_rank_mean`/`A_eff_rank_bar` per cell | **DISCHARGED.** Both fields are now specified as included verbatim in the per-cell report table alongside `indist_min`, at zero incremental GPU-h. Does not separate the two legs into distinct gates (D4's binding narrowing stands) — only makes both legs' raw numbers visible together. | §5 (metric-definitions bullet list, new item) |
| **KW3.13 (MINOR)** — same substance as KW2.8 (below): `t4b`'s K-list extension silently dropped, "+3 smoke cells at `d=2K`" consequence undisclosed | **ACCEPTED-RISK**, one sentence (see KW2.8 row — same finding, same disposition, same location). | §4 (Job-spec template subsection) |
| **KW3.14 (MINOR)** — KW2.9 declined, and `§R1` mischaracterizes round 1's actual finding ("required none" vs. R1's real text, "worth one sentence...") | **ACCEPTED-RISK, superseded rather than retroactively edited.** `§R1` is historical and frozen by house convention — its mischaracterization is not rewritten. Instead, KW2.9 itself is now DISCHARGED for real this revision (see below), which is what R1 actually asked for; the mischaracterization becomes moot going forward rather than corrected in place. One-sentence justification: fixing the underlying gap supersedes editing a frozen historical claim about whether the gap existed. | (no historical-section edit — see KW2.9 row, §5) |
| **KW3.15 (MINOR)** — `§R1`'s KW1.2 row claims §11.4 "cited and reconciled"; §11.4 appears nowhere in the design (substance carried by `EXPERIMENT_LOG.md:8495-8496` instead; conclusion unaffected) | **ACCEPTED-RISK.** `§R1` is historical/frozen; the audit itself confirms this is cosmetic (the underlying conclusion is unaffected, substance is correctly carried elsewhere). One-sentence justification: a citation slip in a frozen historical revision-log entry, with the substantive conclusion independently verified correct, is not worth reopening a frozen section for. | (no edit — historical section stays frozen) |
| **KW3.16 (MINOR)** — the 10–50 GPU-h resource/placement red-team requirement appears only in historical sections, not the revised body; conditional-arm abort handling and the single-K budget-qualification scope limit are undisclosed | **DISCHARGED**, three parts. (a) Resource/placement red-team requirement now stated explicitly in the living body with four concrete pre-launch checks. (b) New §7 bullet states E1/E4 apply uniformly to both arms — no separate, weaker abort path for the conditional arm. (c) New §7 bullet discloses explicitly that only `K_trig` gets budget-qualified; every other FRONTIER label stays 80K-only. | §6 (new red-team bullet); §7 (two new bullets) |
| **KW1.3 (round-1 FATAL, PARTIAL after R2)** — partition logic confirmed exhaustive+exclusive (125/125) by the audit, but the printed demonstration table did not reproduce | **FULLY DISCHARGED.** The underlying partition logic was never in question (round 2's own independent re-execution confirmed it); E5 fixes the one remaining defect — the printed table now matches a fresh execution exactly (10 band-rows, 125/125). Nothing left open. | §5 (E5 table regeneration) |
| **KW2.2 (round-1 MAJOR, PARTIAL after R2)** — eval-inclusive worst case + ABORTED-BUDGET rule were added, but (a) the ceiling trim increased contention/abort risk, (b) retry was unmechanized/unpriced, (c) `harvest()` still emits a gated verdict for a K containing an ABORTED cell | **FULLY DISCHARGED.** (a) closed by E2 (restored ceilings, wider margin than ever). (b) closed by E1 (bounded, budget-gated retry with a derived worst case) + E4 (bounded retry count, terminal `PERSISTENTLY-ABORTED`). (c) closed by E4's named `harvest()` patch (`n_completed` from `status`, not file-glob presence) — specified precisely for the build stage, the same deferral pattern already accepted for KW2.6's job-spec template (design-complete, implementation deferred to build, not a design gap). | §4 (E1, E2, D5/E4 subsections) |
| **KW2.3 (round-1 MAJOR, PARTIAL after R2)** — `INCOMPLETE-AT-K` added as a state, but it deadlocked against `PERSISTENTLY-ABORTED`, and MISSING-vs-ABORTED were lumped together despite the harness treating them differently | **FULLY DISCHARGED.** Deadlock closed by the bounded-retry/terminal-state rule (E4). MISSING and `PERSISTENTLY-ABORTED` are now treated identically and correctly by construction: both feed the SAME interval-logic rule off a `harvest()` patch that counts `status=="COMPLETED"` uniformly (not glob presence), removing the asymmetry the audit found rather than merely disclosing it. | §4, §5 (same E4 material) |
| **KW2.8 (round-1 MINOR, PARTIAL after R2)** — the `d=K+1` micro-smoke instruction survived, but `t4b`'s K-list extension was dropped without comment and the "+3 smoke cells at `d=2K`" consequence was undisclosed | **ACCEPTED-RISK**, one sentence: the `t4b` K-list extension is deferred to the build-stage smoke-test checklist (a build-time task, not this draft's own trainability claim), and the +3 default-`d=2K` smoke cells are harmless extra coverage of an already-rejected convention (§2(a)/(b)), not a gap in this design's own `d=K+1` claim — the build stage must still run the specified `d=K+1` micro-smoke before release. | §4 (Job-spec template subsection, new paragraph) |
| **KW2.9 (round-1 MINOR, declined/NOT-DISCHARGED after R2)** — `harvest()` emits `SUB4-DISCLOSED-ONLY(n=0)` rows for K's outside this design's own outcome space; R1 said "worth one sentence... so a harvest reader is not misled" but added none | **FULLY DISCHARGED, R2.** The one sentence R1 asked for is now added: a harvest reader of this design's report is told explicitly that `SUB4-DISCLOSED-ONLY(n=0)` rows for K's outside {24,26,28,30,32} are a harness-wide artifact of a shared `GRID_SHAPES` dict belonging to OTHER designs, not part of this characterization's outcome space. | §5 ("INCOMPLETE-AT-K" section, new "KW2.9" paragraph) |

**Numbers that moved as a direct, disclosed consequence of the
E1–E6 fixes (not independent changes):** primary per-cell ceiling
`0.75h→1.20h`; conditional per-cell ceiling `1.50h→2.32h`; the
program-level worst-case bound `15.00h (false, broken by retry)
→15.2016h (true, derived by induction)`; 160K nominal per K
`0.9434/1.0230/1.1037→0.9882/1.0716/1.1561` (KW3.11's conservative-ratio
switch); the `K_trig=32` disambiguator label
`PARTIAL-IMPROVEMENT-AT-160K/320K→CONFIRMED-WALL-AT-160K`; the
125-outcome table's residual band `NON-MONOTONE-UNRESOLVED: 6→4+2
(split, tagged)`; three `STATE.md` line citations corrected
(`:39-40→:53`, `:114-116→:128`, `:11-13→:24-26`); the eval-overhead
percentage range `0.7%–1.5%→0.35%–1.58%` (its absolute-max input to
every downstream bound, 0.0126h, is unchanged); the AER/K range at
K=32/1× `0.928–0.966→0.9269–0.9679`.

**Not re-litigated in Rev 2 (out of scope for this pass, unchanged from
Rev 1, flagged for Round 3 to check for completeness):** §2's
`d=K+1`-vs-`d=2K` config choice itself; §2(c)'s mod-K crash arithmetic
and §2(d)'s `h`-never-binds argument (both independently verified
CLEAN by round 2, untouched again); the Gate-2/secondary far-depth
ladder generation (`_gen_grid(K)`, unchanged); the K=48 WAVE-1b block
and the K=32 `d(K)`-grid CLOSED verdict (both stand as written, §7);
the six-rule classification procedure's RULE TEXT itself (E5 changed
only the table's fidelity to that unchanged text, per the audit's own
discharge option (b)); D4's leg-attribution narrowing (still not
separated, only made more transparent by KW3.12's zero-cost fix).

**Disposition not fully implementable inside this file (disclosed, same
structural limit R1 already flagged for KW2.10, now applying
additionally to E4's `harvest()` patch and E1's launcher check):** both
E1 (the launcher-side cumulative-GPU-h admission check) and E4 (the
`harvest()` `n_completed`-vs-file-glob patch) are specified here
PRECISELY ENOUGH for a build-stage implementer to follow exactly, but
neither is — and structurally cannot be, this document being a design
draft that edits no code — actually implemented as code by this
revision. This is the same deferral pattern already accepted for
KW2.6's job-spec template (CLEAN-10 in round 2's own verdict); Round 3
should verify the build stage's actual launcher/harvest code against
these specifications, not merely against this design's prose a second
time.

*Rev 2, 2026-08-06. Written from direct reads of
`NCR_KWALL_ATTACK_R2.md` (694 lines, every MAJOR/MINOR, the full
discharge-verification table, and the verified-clean list), this
file's own `§A2-ADJUDICATION`, fresh execution of the six-rule
partition procedure (Python, all `5³=125` outcomes, reproduced in this
revision's own scratch computation before being transcribed into the
table above), fresh arithmetic for every restored ceiling/nominal/bound
(`max(2×nominal,1.0)`, the E1 induction bound, the KW3.11 ratio switch —
all executed, not hand-computed), a fresh `grep`/read of `STATE.md` for
the three stale line citations (KW3.7), a fresh read of
`NOVEL_ARCH_WATERFALL.md:5065-5080` and `EXPERIMENT_LOG.md:8880-8897`
for the KW3.8 quote-splice correction, and a fresh read of
`matrix-thinking/ncr/ncr_earlyln_scale.py:350-406`
(`discover_seeds_by_K`, `harvest`) confirming the file-glob-vs-status
`gate_eligible` defect KW3.4/E4 names. No repo file other than this one
was created or modified; no command was run on the box; no job was
launched; no git mutation was made.*

## §A3-ADJUDICATION — AUDIT ROUND 3 VERDICT ADOPTED: **REV-REQUIRED**; DELIVERY MODEL CHANGED (Fable, 2026-08-06)

`NCR_KWALL_ATTACK_R3.md`: 3 FATAL / 3 MAJOR / 5 MINOR. E2/E3/E5
DISCHARGED (independently reproduced, third consecutive round for the
partition and budget-wave numbers); E1 NOT-DISCHARGED, E4/E6 PARTIAL.
The E1 FATALs (KW4.1–4.3) share one root cause the coordinator OWNS:
§A2's E1 prescribed a cumulative budget gate while §6 (per the A4.12
flat-spec pool contract the coordinator also imposed) requires cells
to be independent pool specs — a cumulative cap across independent
specs IS the forbidden intra-wave dependency. The two rulings were
contradictory as written; no revision inside that frame could close
KW4.1–4.3.

**BINDING DISPOSITIONS for Rev 3:**
- **F1 (KW4.1–KW4.4; supersedes E1's mechanism, keeps its intent):**
  DELIVERY MODEL CHANGE — the experiment ships as ONE self-contained
  ORCHESTRATOR job (the chained-launcher pattern the pool contract
  itself prescribes for sequenced work): a single queue/pool spec,
  one GPU, cells run sequentially inside it. The orchestrator keeps
  its own realized-spend ledger from wall-clock elapsed (counts
  ABORTED attempts' true spend; attempt-indexed output files — no
  overwrite of a first attempt's record), enforces per-cell CLI
  ceilings (the ENFORCED 1.20/2.32 values are also the charged
  values — KW4.4 unification) plus the 15.00 cumulative gate and
  12.00 retry gate internally. The queue-level spec carries ONE
  ceiling (15.20 + supervisor margin) and is flat/independent —
  §6 rewritten to claim pool-eligibility for the ORCHESTRATOR SPEC,
  not the cells.
- **F2 (KW4.5):** PERSISTENTLY-ABORTED/INCOMPLETE-AT-K rungs are
  EXCLUDED from 160K-trigger candidacy (a K that cannot resolve
  cannot trigger); pre-registered tie-break when interval logic
  leaves K_trig ambiguous between two K's: the SMALLEST candidate K
  runs (rationale: nearest the live rung, most informative for the
  frontier claim; stated in-text, attackable). The orchestrator
  implements trigger resolution as an explicit function of the
  post-interval-logic band vector.
- **F3 (KW4.6):** the accepted-risk's false cross-reference is
  replaced by a real, specified per-K micro-smoke (d-override
  short-run per K∈{26,28,30}) wired into the BUILD gate in the live
  body; harvest-time validity_check stays but is not called a smoke.
- **F4:** all 5 MINORs per their discharge conditions; every E4/E6
  PARTIAL completed or re-justified with an attackable sentence.

Rev 3 → focused audit round 4 (F1 delivery model + F2 trigger logic +
ledger arithmetic; fresh judge) → adjudication → build ceremony
(orchestrator script + specs + harvest patch, own build audit) →
placement red-team → pool.

---

## §R3 REVISION 3 (2026-08-06)

Rev 3, dispatched per §A3-ADJUDICATION's binding dispositions F1–F4.
Every finding from `NCR_KWALL_ATTACK_R3.md` gets a row below — all 3
FATAL (KW4.1–KW4.3), all 3 MAJOR (KW4.4–KW4.6), all 5 MINOR
(KW4.7–KW4.11) — plus the two round-3 PARTIALs (E4, E6), per F4's "no
silent leftovers" instruction. §1–§7, `§A1-ADJUDICATION`, `§R1`,
`§A2-ADJUDICATION`, `§R2`, and `§A3-ADJUDICATION` are UNCHANGED as
historical record EXCEPT where a disposition explicitly required
rewriting a section's content in place — every such rewrite is listed
in the "Where fixed" column below. This design now carries status
**DRAFT-R3 — POST-AUDIT-3, AWAITING FOCUSED AUDIT ROUND 4**.

**The core change (F1) is a delivery-model change, not a bookkeeping
patch: the experiment ships as ONE self-contained orchestrator job —
single queue spec, one GPU, sequential cells — with an internal
wall-clock ledger, attempt-indexed output files, unified
charged-equals-enforced ceilings, and the 15.00/12.00 internal gates.
Every number below was recomputed or derived visibly this revision:**
the trigger's 11 ambiguous configurations were found by a ~40-line
Python sweep (all 300 singly-incomplete configurations enumerated,
matching the audit's count to the digit) and independently
cross-checked against a 240-config two-K-simultaneously-incomplete
sweep (max 2 distinct `K_trig` values in any band-agreeing case, never
3+); the interval-logic decide-rates (KW4.7) were independently
re-executed and matched the audit exactly (64%/68%/100%, 45–54%); the
sequential worst-case bound (`15.0126h`) is derived by induction over
a genuinely non-concurrent admission process, not asserted; the
1.206×→1.207× correction (KW4.10) is a direct re-division
(`1.2685/1.05105=1.20685...`).

| Finding | Disposition | Where fixed |
|---|---|---|
| **KW4.1 (FATAL)** — `ABORTED-BUDGET` cells burn full-ceiling GPU-h but are invisible to the gate on two independent grounds (status filter + missing `gpu_h` field), and a retry overwrites the first attempt's `elapsed_s`; true worst case 28.80h/39.80h against a claimed ≤15.20h | **DISCHARGED.** The orchestrator's own wall-clock timer (`t0`/`t1` around each `subprocess.run`) measures every attempt's `elapsed_h` directly — never the cell JSON's `gpu_h` field, which is absent on the abort path by construction of the harness. The ledger updates UNCONDITIONALLY, before `status` is even inspected — no status filter can hide an aborted attempt. Attempt-indexed outdirs (`attempt1/`, `attempt2/`) mean a retry never overwrites the first attempt's record — all three grounds KW4.1 named (status filter, missing field, overwrite) are closed by construction, not by patching the old file-glob mechanism. | §4 (ORCHESTRATOR CONTRACT — dispatch loop steps 1–2; attempt-indexed outdir paragraph) |
| **KW4.2 (FATAL)** — the induction's premise is false (`realized_before_last_batch` is a COMPLETED-only read, not true cumulative spend); an abort-free counterexample reaches 20.92h; a second, conflicting reading of the trigger rule exists | **DISCHARGED.** Concurrency itself is removed — one GPU, one subprocess in flight, ledger updated synchronously in-process immediately after each attempt. There is no "batch" for the premise to be stale about: `realized_gpu_h` at every gate check IS the true cumulative spend, by construction of strict sequencing, not by convention. KW4.2's counterexample (dispatching K=28/K=30 while K=26 is still training) cannot occur under this model. The reading ambiguity is also closed: the trigger is evaluated at exactly ONE point (§4's "trigger evaluation point"), after all 12 primary cells are terminal — never earlier, so there is no second reading to disagree with the first. | §4 (ORCHESTRATOR CONTRACT — cell order; worst-case derivation; trigger evaluation point) |
| **KW4.3 (FATAL)** — E1 has no owner in the repo's real dispatch path (`queue_worker.sh`/`idle_fallback_daemon.sh` carry no budget state and no "batch"), and a cumulative cap across independent pool specs is itself the forbidden intra-wave dependency; §6 asserted both simultaneously | **DISCHARGED** — Option B (audit's own §9 recommendation), simplified: sequential dispatch needs no reservation ledger, only Option B's parallel-dispatch case would have. The orchestrator script is the owner — one purpose-built process the pool dispatches as a SINGLE job; `queue_worker.sh`/`idle_fallback_daemon.sh` need no budget state because neither ever sees more than "run this one job." §6 rewritten in full: pool-eligibility is claimed for the orchestrator spec, never for the 16 cells — the contradiction dissolves because the two claims are no longer about the same object. | §4 (delivery-model paragraph; ORCHESTRATOR CONTRACT); §6 (rewritten in full) |
| **KW4.4 (MAJOR)** — `ceiling(cell)` denotes two different numbers: the gate charged `max(2×nominal,1.0h)` (per-K) while the runner enforced the shared CLI `1.20`/`2.32`; under-charge up to 1.14h, true idealized bound ≈16.34h | **DISCHARGED.** The gate now charges the CLI value DIRECTLY (`1.20` primary, `2.32` conditional) — the exact same number the runner enforces, by definition (ORCHESTRATOR CONTRACT gate 1). The per-K `max(2×nominal,1.0)` figures are relabeled explicitly INFORMATIONAL (sizing justification for why the shared CLI value clears every K's own floor) and no longer feed any gate arithmetic — no under-charge is possible. | §4 (command-block ceiling paragraph; ORCHESTRATOR CONTRACT gate 1; "Ceiling reference table" relabel) |
| **KW4.5 (MAJOR)** — the trigger precondition ("defers … until it resolves") is unchanged from Rev 1 and deadlocks against a TERMINAL `PERSISTENTLY-ABORTED` state; 11 configurations exist where interval logic decides the band but `K_trig` remains ambiguous between two K's | **DISCHARGED.** New trigger rule (F2): `K_trig` is evaluated independently via the SAME interval-candidate cross-product used for band classification (not derived from the band label), with an explicit tie-break — smallest candidate K — whenever the resulting set has more than one value. All 11 audit-found configurations are enumerated in the design and resolve by construction under `min()` (re-verified this revision by direct execution, exact match to the audit's count and rows). The "defers … until it resolves" deadlock is retired: a K that cannot resolve is excluded from trigger candidacy (F2) and the whole run reports `TRIGGER-UNRESOLVED` — a terminating, disclosed outcome, never an infinite wait. | §4 (Trigger rule for the conditional 160K arm, rewritten in full — resolution-state table, pseudocode, 11-config enumeration table); §4 (D5/E4 rule, cross-referenced) |
| **KW4.6 (MAJOR)** — the KW2.8/KW3.13 accepted-risk rests on a false internal cross-reference: the `d=K+1` micro-smoke it claims is "recorded above" exists nowhere in the living body, only in the frozen, non-operative `§R1` | **DISCHARGED (F3).** A real per-K micro-smoke is specified as an explicit BUILD-RELEASE GATE: a 500-step `d`-override cell per K∈{26,28,30}, exact pass criterion (no uncaught exception; `status∈{COMPLETED,ABORTED-BUDGET}`; `K`/`d`/`d_override` match the requested shape), run strictly before any production cell. Kept explicitly distinct from, and never conflated with, the harvest-time `validity_check` (a post-hoc correctness assertion on a COMPLETED production cell — never called a smoke anywhere in this document). | §4 (KW2.8/KW3.13/KW4.6 close-out, rewritten in full) |
| **KW4.7 (MINOR)** — interval logic frequently cannot decide, undisclosed: at `r_known=2` bands differ in 64%/68%/100% of configurations (K=26/28/30); two singly-incomplete K's decide in only 45–54% | **DISCHARGED.** Figures independently re-derived this revision by direct execution (exact match to the audit's: 16/25, 17/25, 25/25; 43/80, 38/80, 36/80) and disclosed inline as a new bullet in the interval-logic paragraph, with the practical consequence stated plainly (a terminal abort at K=30 with `r_known=2` guarantees `INCOMPLETE-AT-K`). | §4 (D5/E4 rule, new "Decide-rate, disclosed" bullet) |
| **KW4.8 (MINOR)** — "`INCOMPLETE-AT-K` for that K" is a category error: the six-rule procedure returns one global label for the triple, not a per-K band | **DISCHARGED.** Reworded throughout: `INCOMPLETE-AT-K` is stated explicitly as a STUDY-LEVEL verdict orthogonal to the 125-outcome partition, carrying the affected K(s) as a disclosure field — never implying a per-K band object the procedure cannot produce. | §4 (D5/E4 rule); §5 (INCOMPLETE-AT-K section, rewritten) |
| **KW4.9 (MINOR)** — "actual TRAINING time is bounded above by its ceiling" is false: the `elapsed>ceiling_s` test fires only at `log_every=500`-step boundaries, so training can overshoot by up to one interval; unpriced in the induction | **DISCHARGED.** The claim is no longer made unqualified. The worst-case derivation explicitly prices the `log_every` overshoot (max observed 0.0031 GPU-h) as one of the two possible single-attempt overrun terms (the other being eval overhead), and its own disclosed contention-variance is folded into the stated supervisor margin rather than silently ignored. | §4 (ORCHESTRATOR CONTRACT, worst-case derivation) |
| **KW4.10 (MINOR)** — `1.206×` is a truncation of the true `1.2069×` (rounds to `1.207×`); the error direction flatters the margin | **DISCHARGED.** Corrected to `1.2069×`, rounding to `1.207×`, re-derived directly (`1.2685/1.05105`); the headroom figure (`2.00/1.2069≈1.657×`, "≈1.66×") is unchanged since it was already computed from the correct ratio, only the reported digit moves. | §4 (Margin claim, corrected) |
| **KW4.11 (MINOR)** — one 2-line diff hunk (the KW3.7 `STATE.md` line-renumber fix) falls outside every section `§R2`'s "Where fixed" column claims — it lands in the pre-§1 mandate preamble, attributed to "§1" | **ACCEPTED-COSMETIC, not retroactively edited — same precedent as `§R2`'s own KW3.14/KW3.15 rows.** `§R2` is historical and frozen by house convention; the underlying fix (correct `STATE.md` line numbers) is genuinely present and correct at the cited location — only its section attribution in a frozen revision-log TABLE is off by one block. One-sentence justification: editing a frozen table to correct a table-attribution slip, when the substantive fix it describes is real, visible, and unaffected, trades a documented correction for an invisible retcon; not worth reopening a frozen section for. | (no edit — `§R2`'s table stays frozen; recorded here per F4's "no silent leftovers") |
| **E4 (round-3 PARTIAL — band side delivered exactly as specified, trigger side deadlocked and non-composing)** | **FULLY DISCHARGED.** The band side was independently verified correct by round 3 itself (KW4.5: "delivered exactly as the disposition specified") and is unchanged in substance this revision — only its cross-references are repointed from the retired E1 launcher check to the orchestrator's own HARD/RETRY gates (§4). The trigger side (KW4.5's forcing defect) is fixed by F2's new rule in full — see the KW4.5 row above. Nothing left open on either side. | §4 (Trigger rule; D5/E4 rule) |
| **E6 (round-3 PARTIAL — coverage complete, but (1) KW3.1/KW3.4's `§R2` rows overstated given this round's findings, and (2) the KW2.8/KW3.13 accepted-risk was REFUTED)** | **FULLY DISCHARGED, both problems, differently.** (1) KW3.1's and KW3.4's `§R2` "DISCHARGED" rows are **SUPERSEDED, not retroactively edited** (the same frozen-section precedent `§R2` itself set for KW3.14/KW3.15): the mechanism KW3.1 discharged (the launcher-side E1 cumulative check) no longer exists in this design at all, replaced by F1's orchestrator ledger; KW3.4's band-side fix stands unchanged, and its trigger-side gap (KW4.5) is now closed by F2. Both `§R2` rows describe a mechanism this revision retired, not a live defect. (2) KW4.6's refutation is fixed for real by F3's owned, build-gated micro-smoke — see the KW4.6 row above; no second false cross-reference is introduced this time. | §4, §6 (throughout — the delivery-model change itself is the discharge); this table (supersession stated explicitly for KW3.1/KW3.4) |

**Numbers that moved as a direct, disclosed consequence of the F1–F4
fixes (not independent changes):** the declared program-level bound
`≤15.20h (E1, false — broken by KW4.1–KW4.3) → 15.0126h (true,
sequential-model induction) → 15.50h (declared pool ceiling: 15.20h
disclosed-conservative + 0.30h stated supervisor margin)`; `ceiling
(cell)` for gate purposes `per-K max(2×nominal,1.0) →` the shared CLI
value `1.20/2.32` (KW4.4, no more charged/enforced split); the pool
artifact count `16 possible per-cell job specs → 1 orchestrator job
spec` (job-spec template, §4); the margin ratio `1.206×→1.2069×/1.207×`
(KW4.10); the trigger mechanism `single scan, no interval resolution,
one deadlocking precondition → interval-resolved per-K states + an
independent K_trig cross-product + a stated tie-break`, with 11
previously-silent ambiguous configurations now enumerated and resolved
in-text; the `INCOMPLETE-AT-K` framing `implied per-K band →` explicit
study-level verdict (KW4.8).

**Not re-litigated in Rev 3 (out of scope for this pass, unchanged
from Rev 2, flagged for Round 4 to check for completeness):** §2's
`d=K+1`-vs-`d=2K` config choice itself; §2(c)'s mod-K crash arithmetic
and §2(d)'s `h`-never-binds argument (both independently verified
CLEAN by round 3, untouched again); the Gate-2/secondary far-depth
ladder generation (`_gen_grid(K)`, unchanged); the K=48 WAVE-1b block
and the K=32 `d(K)`-grid CLOSED verdict (both stand as written, §7);
the six-rule classification procedure's RULE TEXT itself and the
125-outcome partition (E5, independently reproduced a THIRD time by
round 3 — `NCR_KWALL_ATTACK_R3.md` §3/§8 — and not touched this
revision); the `$0` K=32 reuse branch's matched-160K scoring (E3,
likewise independently reproduced a third time and unchanged); D4's
leg-attribution narrowing (still not separated, unaffected by the
delivery-model change).

**Disposition not fully implementable inside this file (disclosed,
same structural limit `§R1`/`§R2` already flagged for KW2.10/E1/E4,
now extended to the whole ORCHESTRATOR CONTRACT):** F1 specifies the
orchestrator's cell order, per-cell command shape, ledger update
points, gate check points, abort/retry state machine, trigger
evaluation point, harvest invocations, and output JSON schema
PRECISELY ENOUGH for a build-stage implementer to follow exactly, but
none of it is — and structurally cannot be, this document being a
design draft that edits no code — actually implemented as the
orchestrator script, the `harvest()` patch, or the job-spec JSON by
this revision. This is the same deferral pattern already accepted for
KW2.6's job-spec template and E4's `harvest()` patch in Rev 1/Rev 2
(CLEAN-10 / CLEAN precedent both rounds); Round 4 should verify the
build stage's actual orchestrator script against these specifications
line-by-line — cell order, ledger semantics, gate arithmetic, the
attempt-indexed outdir convention, and the trigger's tie-break — not
merely re-read this design's prose a fourth time. Separately, and by
design rather than omission: the **0.30h supervisor margin is STATED,
not derived** — F1's own mandate calls for "one ceiling ≈15.20 +
a stated supervisor margin," and this revision is explicit that 0.30h
is a policy choice sized generously against the disclosed but
unbounded contention-variance of the `log_every` overshoot (KW4.9),
not a tight statistical bound; Round 4 may tighten it once the build
stage's actual orchestrator-process overhead (subprocess spawn,
interpreter/import latency) is measured rather than estimated.

*Rev 3, 2026-08-06. Written from direct reads of
`NCR_KWALL_ATTACK_R3.md` (974 lines, every FATAL/MAJOR/MINOR, the
per-disposition summary, and the "what I could not break" list), this
file's own `§A3-ADJUDICATION`, and fresh Python execution (not
hand-checked) of: (1) the `classify()` six-rule procedure combined with
the OLD trigger rule over all 300 singly-incomplete `(incomplete_K,
r_known, other two K's)` configurations, isolating the 11 cases where
band agrees but `K_trig` disagrees — reproduced exactly (11/11,
matching the audit's rows to the digit); (2) an extended 240-config
sweep of the two-K-simultaneously-incomplete case, confirming no
band-agreeing configuration ever produces more than 2 distinct
`K_trig` candidates; (3) the KW4.7 decide-rate figures, independently
re-executed over the same `r_known=2`/two-K-incomplete domains
(exact match: 16/25, 17/25, 25/25, 43/80, 38/80, 36/80). Also read
`matrix-thinking/ncr/ncr_earlyln_scale.py` (lines 191–266, 303–304,
351–406) confirming the `log_every=500` overshoot window (KW4.9) and
the `gpu_h`-assignment/overwrite defects (KW4.1) at their cited lines,
and `matrix-thinking/queue/idle_fallback_daemon.sh` (header, `:10-16`)
for the pool contract text §6 is checked against. No repo file other
than this one was created or modified; no command was run on the box;
no job was launched; no git mutation was made. The two scratch Python
scripts used for the enumerations above are session-local working
files, not part of this design and not committed.*

## §A4-ADJUDICATION-KWALL — AUDIT ROUND 4 VERDICT ADOPTED: **REV-REQUIRED**, CONTRACT-LINE FIXES ONLY (Fable, 2026-08-06)

`NCR_KWALL_ATTACK_R4.md`: 2 FATAL / 3 MAJOR / 8 MINOR; F3 DISCHARGED,
F4 discharged-with-one-defect, F1 NOT / F2 PARTIAL. The audit's own
structural ruling is adopted with it: the delivery model STANDS (K_trig
proven total over all 1000 reachable vectors, zero excluded-K
dispatches; partition + pricing chains reproduce again; §6's
"self-enforced ceiling" honesty confirmed against both dispatch
scripts) — every discharge is contract-line surgery, no new model.
The 0.30h margin ruling (acceptable as-declared, do NOT tighten) is
adopted verbatim.

**BINDING DISPOSITIONS for Rev 4:**
- **G1 (KW5.1):** write-ahead attempt pricing — BEFORE each dispatch
  the orchestrator persists an attempt-open record (cell, ceiling,
  monotonic start); ledger recovery on any (re)start FIRST closes
  every dangling open record by charging its FULL ceiling
  (conservative), then resumes. A crash can now only over-charge,
  never re-open budget. State the revised bound argument.
- **G2 (KW5.2):** attempt dirs stay as archive; on attempt acceptance
  the orchestrator COPIES the accepted output to the canonical flat
  path harvest() already globs (one cell = exactly one canonical
  file; overwrite forbidden by an exists-check that fails loudly).
  Denominator-4 guard untouched. State the harvest contract
  explicitly: harvest reads canonical paths ONLY.
- **G3 (KW5.3):** exit-code classification table: stop-file exit (3) →
  STOPPED-BY-OPERATOR (terminal for the wave, never an abort, never
  retried, disclosed); ceiling abort → ABORTED-BUDGET-n; any other
  non-zero → CRASHED-n (retry-eligible under the same gates); enum
  enumerated in the output-JSON field spec.
- **G4 (KW5.4):** define the report `run_status` enum exhaustively
  (COMPLETE, COMPLETE-DEGRADED — the pre-registered degradation
  outcomes enumerated — STOPPED-BY-OPERATOR, EXHAUSTED-BUDGET);
  the job spec's validity_check accepts every enum value that §5
  pre-registers as reportable — graceful degradation must land in
  completed/, not failed/.
- **G5 (KW5.5):** pre-registered trigger precondition: the conditional
  arm dispatches ONLY if the triggering K's own primary band is
  DECIDED (not INCOMPLETE-AT-K, not interval-ambiguous); otherwise
  TRIGGER-UNRESOLVED. Re-run the 1000-vector sweep under this rule
  and embed the new decide/unresolved split.
- **G6:** the 8 MINORs + F4's one defect + the unattributed diff hunk
  per their discharge conditions; §R4 rows for everything.

Rev 4 → audit round 5 (contract-line verification, expected terminal)
→ adjudication → BUILD ceremony.

---

## §R4 REVISION 4 (2026-08-06)

Rev 4, dispatched per §A4-ADJUDICATION-KWALL's binding dispositions
G1–G6. Every finding from `NCR_KWALL_ATTACK_R4.md` gets a row below —
both FATAL (KW5.1–KW5.2), all 3 MAJOR (KW5.3–KW5.5), all 8 MINOR
(KW5.6–KW5.13) — with KW5.8's row also explicitly covering F4's one
defect and the audit's separately-flagged "unattributed diff hunk"
(§6 INTEGRITY; §4 F4 `DISCHARGED-WITH-ONE-DEFECT`), independently
re-confirmed this revision to be the SAME finding under three names,
not three separate ones. §1–§7, `§A1-ADJUDICATION`, `§R1`,
`§A2-ADJUDICATION`, `§R2`, `§A3-ADJUDICATION`, `§R3`, and
`§A4-ADJUDICATION-KWALL` are UNCHANGED as historical record EXCEPT
where a disposition explicitly required rewriting a section's content
in place — every such rewrite is listed in the "Where fixed" column
below. This design now carries status **DRAFT-R4 — POST-AUDIT-4,
AWAITING AUDIT ROUND 5**.

**MD5 verification, run before and after this revision's edits (all
seven historical sections, byte-identical):**

| Section | MD5 (pre-edit) | MD5 (post-edit) |
|---|---|---|
| `§A1-ADJUDICATION` | `16afc36df0678c2e0bc75a4255d387a4` | `16afc36df0678c2e0bc75a4255d387a4` |
| `§R1` | `be21d7a231e2cdf0fcd661bf8df865f7` | `be21d7a231e2cdf0fcd661bf8df865f7` |
| `§A2-ADJUDICATION` | `38f1cf368c876a95adcf8029c8f2364e` | `38f1cf368c876a95adcf8029c8f2364e` |
| `§R2` | `6d68f9c8be9f3970531d530338b066a5` | `6d68f9c8be9f3970531d530338b066a5` |
| `§A3-ADJUDICATION` | `c3199e7e13872fabd408e35e3ef7f88c` | `c3199e7e13872fabd408e35e3ef7f88c` |
| `§R3` | `43d4510bb9ed7af3abb9321111d1eed5` | `43d4510bb9ed7af3abb9321111d1eed5` |
| `§A4-ADJUDICATION-KWALL` | `bae588d2031106eec0de0a461efb43ab` | `bae588d2031106eec0de0a461efb43ab` |

**The core change this round is NOT a delivery-model change — the
audit's own structural ruling (adopted verbatim in
`§A4-ADJUDICATION-KWALL`) is that no fifth delivery model is needed,
only contract-line surgery inside the existing ORCHESTRATOR CONTRACT.
Every number below was recomputed or derived visibly this revision:**
the revised worst-case bound (`15.0157h`, extending Rev 3's `15.0126h`
to cover the crash-recovery case AND correcting KW5.7's combined
single-attempt tail) is derived by an induction extended to cover
restart cycles, not asserted; the G5 trigger-precondition resweep
(the full 1000-vector reachable state space, old split `844/156`
reproducing the audit exactly, new split `473/527` with 0
paid-on-unresolved) was executed fresh this revision, cross-checked
two independent ways (a direct resweep under the amended rule, and
arithmetic on the audit's own joint-distribution figures:
`473+371=844`, `156+371=527`); the KW5.6 candidate-`K_trig`-set-size
resweep (the full `9³=729` non-`UNRESOLVED` state space) reproduces
the audit's own `{1:612,2:102,3:14,4:1}` to the count, including an
independently-verified check (not merely re-cited) that all 15
wide-tie cases land in band `INCOMPLETE-AT-K`.

| Finding | Disposition | Where fixed |
|---|---|---|
| **KW5.1 (FATAL)** — a mid-attempt orchestrator death zero-counts that attempt's spend; the ledger under-counts on the box's own documented restart path (`queue_worker.sh`'s from-scratch-retry reclaim + "killing the tmux session kills its in-flight job too") and nothing external backstops it, so `15.0126h` was not a bound; a second, smaller face: a restart's `GATE-REFUSED` can disagree with `harvest()`'s on-disk view of an already-`COMPLETED` cell | **DISCHARGED.** G1's write-ahead attempt record (`ledger.open_attempt`, written BEFORE every `subprocess.run`) plus a mandatory recovery procedure (run before ANY gate check or cell-walk resume on restart) closes both faces: (i) a dangling `open_attempt` is charged its FULL `charged_ceiling` before anything else happens, so a crash can only over-charge the ledger, never leave it silently short — "nothing caps the number of cycles" no longer matters, since every cycle's dangling record is priced in before the next cycle's first gate check (the induction is extended, by construction, to cover the crash case); (ii) cell-level resume explicitly never re-gates a cell with an existing terminal ledger row, so a restart at `realized≈13h` cannot turn an already-`COMPLETED` cell into `GATE-REFUSED` — the ledger is now the one source of truth both the dispatch loop and `harvest()`'s canonical-path view (G2) trace back to. Revised bound: `R_N ≤ 15.00+0.0157=15.0157h` (the `0.0157h` tail itself corrected per KW5.7, below); the crash case is actually TIGHTER (`R_N≤15.00` exactly, no tail) than the completing case. | §4 (dispatch loop — write-ahead + exit-code branch; "Ledger persistence + write-ahead recovery," rewritten in full; "Worst-case bound," rewritten in full) |
| **KW5.2 (FATAL)** — the attempt-indexed outdirs (F1's KW4.1 fix) break `harvest()`'s flat, non-recursive glob outright — a null read after the full ≤15 GPU-h has been spent; the obvious build "fix" (a recursive glob) corrupts the A4.9 fixed-denominator-4 guard instead (duplicate basenames across attempt dirs inflate `n_seeds` to 5 on a retried seed); the second `harvest()` call over "combined results" isn't expressible in `harvest()`'s single-`outdir` signature either | **DISCHARGED.** G2 keeps attempt dirs as an archive-only record `harvest()` never reads, recursively or otherwise, and instead has the orchestrator COPY each `COMPLETED` attempt's JSON to the canonical flat path `discover_seeds_by_K`/`harvest()` already glob, UNMODIFIED — one file per (K,seed), ever, enforced by an exists-check that ABORTS LOUDLY rather than overwrites. This makes the duplicate-seed corruption scenario unreachable by construction — no second file for a duplicate-glob bug to find, whether or not a future build implementer is ever tempted toward recursion — rather than closed by a dedupe step. The "second `harvest()` call" ambiguity is resolved explicitly: a SEPARATE call against the conditional arm's own canonical directory, merged afterward, never one call spanning two trees. Bonus: this also makes D5/E4's previously-specified `harvest()` code patch (file-glob-presence → status-based `n_completed`) unnecessary — a canonical file now only ever exists for a `COMPLETED` cell BY CONSTRUCTION, one fewer build-stage change than Rev 3 specified. | §4 ("Attempt dirs are ARCHIVE ONLY," rewritten; "Trigger evaluation point + harvest invocations" — new "G2 — canonical-path harvest contract" subsection; D5/E4 "Enforcement point," rewritten) |
| **KW5.3 (MAJOR)** — "`ABORTED-BUDGET` (or a non-zero subprocess exit)" folds the `--stop-file` kill switch's `sys.exit(3)` into a budget abort, converting an operator stop into a mass-`PERSISTENTLY-ABORTED` of every remaining cell in under a minute; the same conflation misfiles genuine crashes (shape bug, OOM, import failure) as budget aborts | **DISCHARGED.** G3 branches on the EXACT exit code, never its non-zero-ness: exit `3` → `STOPPED-BY-OPERATOR` (terminal for the WHOLE wave — the orchestrator flushes the ledger and dispatches nothing further, never retried); a cell JSON with `status=="ABORTED-BUDGET"` → `ABORTED-BUDGET-n` (path unchanged); any OTHER non-zero exit with no such JSON → `CRASHED-n`, disclosed distinctly from a budget abort (KW5.3's "not a coin-flip seed" point) but retry-gated identically. All four reachable non-`COMPLETED` attempt states (`ABORTED-BUDGET`, `CRASHED`, `CRASHED-RECOVERED` (G1), `STOPPED-BY-OPERATOR`) are enumerated in `attempts[].status`. | §4 (dispatch loop — exit-code-exact branch; new "Operator stop" paragraph; Output JSON schema — `attempts[].status` enum) |
| **KW5.4 (MAJOR)** — `run_status`'s two schema values are never defined anywhere; the job spec's own `validity_check` asserts `run_status=="COMPLETE"`, routing the design's own pre-registered graceful-degradation outcome to `failed/` | **DISCHARGED.** G4 defines all four `run_status` values exhaustively — `COMPLETE`; `COMPLETE-DEGRADED` (with its two pre-registered sub-cases, *primary-retry-refused* and *conditional-throttled*, enumerated); `STOPPED-BY-OPERATOR`; and the new `EXHAUSTED-BUDGET` for the more severe case where even the primary sweep's own baseline can't complete (shown reachable: 12 primary first attempts at the shared `1.20h` ceiling sum to `14.40h`, inside `15.00` only with essentially no headroom left). `validity_check` now asserts `run_status in {"COMPLETE","COMPLETE-DEGRADED","EXHAUSTED-BUDGET"}` (every value representing the orchestrator's own logic completing, never a bug) plus a new self-consistency check (`realized_gpu_h_final == sum(elapsed_h)`) — `STOPPED-BY-OPERATOR` is deliberately EXCLUDED (a human action, not a completed-or-degraded RESULT of this design's own logic; resubmission resumes cleanly via G1's cell-level resume rule, no data lost). | §4 (Output JSON schema — new "`run_status` enum, defined" subsection; Job-spec template `validity_check`, rewritten) |
| **KW5.5 (MAJOR)** — the trigger dispatches a PAID ≤9.28 GPU-h 4-cell conditional arm in 371/1000 reachable state vectors where the primary band is `INCOMPLETE-AT-K` (§5 explicitly excludes that band from frontier claims) — never pre-registered either way | **DISCHARGED** (audit's discharge option (a) — SUPPRESS with disclosure, not option (b) run-and-qualify). G5 adds one precondition, checked only once the K-scan itself decides: the conditional arm fires only if the whole-study primary band is ALSO decided, never `INCOMPLETE-AT-K`; otherwise the trigger's result is overridden to `TRIGGER-UNRESOLVED`, disclosing the K-scan's own candidate as `band_blocked_K_trig` rather than silently dropping it. Re-executed over the full 1000-vector state space this revision: old split (K-scan alone) `DECIDED` 844 / `TRIGGER-UNRESOLVED` 156, with 371 of the 844 paid-on-unresolved (reproducing the audit's own figures to the digit — model fidelity confirmed before trusting the new number); new split under G5, `DECIDED` **473** / `TRIGGER-UNRESOLVED` **527**, with **0 paid-on-unresolved by construction** (`473+371=844`, `156+371=527` — the new split is exactly the old split with the 371 defect cases moved, confirmed two independent ways). The 11-configuration ambiguity table is UNAFFECTED (every row there is band-agreeing by definition, so G5 cannot fire on any of them). | §4 (Trigger rule — new "Trigger precondition — the conditional arm requires a DECIDED band" subsection + amended pseudocode); §5 (INCOMPLETE-AT-K section — "Related to, but no longer independent of, `TRIGGER-UNRESOLVED`," rewritten) |
| **KW5.6 (MINOR)** — "no 3-way tie is ever produced by this rule" is stated unqualified but is false over the full reachable state space, which permits all three of K∈{26,28,30} incomplete simultaneously; candidate-`K_trig`-set sizes reach 3 (14 cases) and 4 (1 case) among the non-`UNRESOLVED` states | **DISCHARGED** (both discharge options taken — restate scope AND report the extended figures, stronger than either alone). The lead sentence is now explicitly scoped to "a band-agreeing configuration with at most two simultaneously-incomplete K's" (the domain the 11-row table and the 240-config sweep actually cover); the full `9³=729`-space sweep is re-executed and reported: `{1: 612, 2: 102, 3: 14, 4: 1}`, matching the audit's own figures to the count. Independently re-verified this revision (not merely re-cited): all 15 of the wide-tie (≥3-candidate) cases land in band `INCOMPLETE-AT-K` — and, as of G5 above, are therefore ALSO `TRIGGER-UNRESOLVED`, so none of them ever reaches the tie-break. Nothing breaks; the narrower sentence is simply the one that is true, and is the one that stands. | §4 (11-configuration ambiguity table's closing paragraph, rewritten) |
| **KW5.7 (MINOR)** — "only one of the two [overrun terms] can apply to a single attempt" is false — the eval-overhead tail and the log-interval-overshoot tail can co-occur on the same attempt; a third, unpriced term (interpreter/CUDA-init/model-build startup) sits outside every priced term while `15.0126h` is presented as the tight derived bound; true single-attempt tail ≈0.016–0.021h | **DISCHARGED.** The worst-case bound derivation (rewritten for G1 regardless) now sums both terms explicitly — `0.0126+0.0031=0.0157` GPU-h — replacing "only one can apply" with the SUM, exactly per the audit's discharge condition. The process-startup term is explicitly named as the thing the STATED (not derived) `0.30h` supervisor margin carries, unchanged in substance from Rev 3, now stated beside the corrected tail rather than the old, narrower one. Impact on the declared numbers: nil — `15.20h`/`15.50h` both still clear the corrected `15.0157h` tight bound with room to spare (`0.1843h` at the `15.20h` rounding, vs Rev 3's `0.1874h`). | §4 (Worst-case bound derivation, rewritten in full) |
| **KW5.8 (MINOR) — the SAME finding as F4's one defect (§4 `DISCHARGED-WITH-ONE-DEFECT`) and the audit's "unattributed diff hunk" (§0/§6 INTEGRITY); independently re-confirmed this revision via a fresh `git diff HEAD~2 HEAD~1 -- matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md` (9 hunks, matching the audit's own count exactly; the hunk with no attributing row is the §7 E1/E4-uniformity-bullet rewrite, old `:1162-1175` → new `:1491-1507`)** — §R3's "every such rewrite is listed in the 'Where fixed' column below" is false: §7's bullet WAS substantively rewritten that revision and no row names §7 | **ACCEPTED-COSMETIC, NOT retroactively edited** — same precedent §R2 set for KW3.14/KW3.15 and §R3 set for KW4.11. The audit itself found this precedent did NOT transfer to `§R3` at the time of the R4 audit, because `§R3` was still the LIVE revision log for the round then under review. As of THIS revision, `§R3` is no longer live — this section (§R4) is now the live revision log, and `§R3` becomes historical/frozen exactly like `§R2` before it (confirmed byte-identical before/after this revision, MD5 table above). The precedent therefore DOES transfer now: the underlying fix (§7's bullet is genuinely, correctly rewritten — re-verified this revision, unaffected by anything in this table) is real; only a frozen table's own self-description of its edit list is imprecise. One-sentence justification: reopening a frozen table to fix a table-attribution slip, when the substantive fix it under-describes is real, visible, and unaffected, trades a documented correction for an invisible retcon — not worth it. Recorded here per G6's "no silent leftovers" instruction, and cross-identified explicitly so a future round does not re-discover the same finding under three different names. | (no edit — `§R3`'s table stays frozen, confirmed byte-identical; recorded here per G6 only) |
| **KW5.9 (MINOR)** — `orchestrator_report.json` carries no field for the KW4.8 fix (`INCOMPLETE-AT-K`'s affected-K disclosure, the candidate bands), nor for the F3 smoke results, per-attempt exit codes, assigned GPU, or git commit | **DISCHARGED.** Schema extended: `band.incomplete_at_K`, `band.candidate_bands`, a `smoke` block (`K26`/`K28`/`K30` PASS\|FAIL), `gpu_id`, `git_commit`; per-attempt exit codes are now covered by `attempts[].status`'s expanded enum (G1/G3, above) and `trigger.band_blocked_K_trig` (G5). | §4 (Output JSON schema, rewritten in full) |
| **KW5.10 (MINOR)** — the 3 micro-smokes (≤0.15 GPU-h) run outside the 15.50h ceiling, outside any pool spec, with no stated GPU or free-GPU-gate discipline | **DISCHARGED.** One disclosure added: total disclosed program spend is `≤15.50+≤0.15=≤15.65 GPU-h`; the 3 micro-smokes run on a GPU verified idle by the build/red-team stage before the orchestrator is promoted to the pool, explicitly NOT covered by `queue_worker.sh`'s free-GPU gate (which governs pool jobs only) — a manual pre-launch check, stated as such rather than left implicit. | §4 (KW2.8/KW3.13/KW4.6 close-out, "What runs" bullet); §6 ("Own cost ceiling" bullet) |
| **KW5.11 (MINOR)** — every results/ledger/smoke path in this design is RELATIVE while the house convention (job 108) is absolute everywhere, and `queue_worker.sh` runs both `cmd` and `validity_check` from its OWN CWD | **DISCHARGED.** `NCR_ROOT=/home/nvidia/ncr` stated explicitly, matching job-108's own convention; the primary command, conditional command, smoke command, ledger path, and job-spec `output_dir` are all shown in absolute form; a normative sentence states that any other `results_kwall_characterization/`-style mention elsewhere in this document is informal shorthand for the same absolute path, never a second, relative convention. | §4 (primary/conditional/smoke command blocks; ledger persistence paragraph; job-spec template `output_dir`) |
| **KW5.12 (MINOR, cosmetic)** — `1.05105` vs `1.0510` both used as the same K32-seed-3 denominator; both round to `1.207` | **ACCEPTED-COSMETIC, no edit needed.** Re-verified this revision: the LIVE body (§4's "160K nominal per cell" paragraph and its "Margin claim, corrected" paragraph) already uses ONE consistent digit, `1.0510`, in both places — there is no live-body inconsistency to fix. The only divergent copy (`1.05105`) lives inside the FROZEN `§R3` disposition table, which — per the same house convention just re-affirmed at KW5.8, above — stays frozen rather than retroactively edited. Both digits round to the identical `1.207`; no reader-facing number is at risk either way. | (no edit — the live body was already internally consistent; the frozen table stays frozen) |
| **KW5.13 (MINOR)** — the §4 resolution-state table's `DECIDED → one fixed r-value` collapse is valid for the trigger's `ROBUST`-only scan ONLY; band classification must evaluate `classify()` at BOTH interval candidates for every `r_known`, not just the `AMBIGUOUS` case — a build implementer reading the "shared" table literally could collapse to one candidate and mis-report a decided band (counterexample: `r_known=1` at K=26 gives different bands at `r=1` vs `r=2`) | **DISCHARGED.** A scope-note paragraph added directly after the resolution-state table states the distinction explicitly, with the audit's own counterexample reproduced; the D5/E4 interval-logic bullet cross-references it and now states "for EVERY value of `r_known∈{0,1,2,3}`, never collapsed to one candidate even where the trigger's `ROBUST`-only scan can." | §4 (resolution-state table — new "Scope note on the `DECIDED` collapse" paragraph; D5/E4 interval-logic bullet, amended) |

**Numbers that moved as a direct, disclosed consequence of the G1–G6
fixes (not independent changes):** the derived tight worst-case bound
`15.0126h (Rev 3, false on the crash path per KW5.1) → 15.0157h
(Rev 4, true — extends the induction to cover crash-recovery AND
corrects KW5.7's combined tail)`; the declared `15.20h`/`15.50h`
figures are UNCHANGED (both still clear the corrected tight bound with
room to spare); the trigger's `DECIDED`/`TRIGGER-UNRESOLVED` split
`844/156 (K-scan alone) → 473/527 (K-scan + G5's band precondition)`,
with paid-on-unresolved `371 → 0`; the candidate-`K_trig`-set-size
sweep's scope `≤2-simultaneously-incomplete claim (unqualified, false)
→ full 9³=729-space figures {1:612,2:102,3:14,4:1} (qualified, true)`;
`run_status`'s reachable values `2 (undefined) → 4 (defined: COMPLETE/
COMPLETE-DEGRADED/STOPPED-BY-OPERATOR/EXHAUSTED-BUDGET)`;
`attempts[].status`'s reachable values `3 (COMPLETED/ABORTED-BUDGET/
GATE-REFUSED) → 6 (+ CRASHED, CRASHED-RECOVERED, STOPPED-BY-OPERATOR)`;
disclosed total program spend `15.50h (smoke cost undisclosed) →
≤15.65h (smoke cost folded in)`; the `harvest()` build-stage
instruction `a required code patch (file-glob-presence → status-based
n_completed) → no patch needed (subsumed by G2's copy-on-accept
discipline)`.

**Not re-litigated in Rev 4 (out of scope for this pass, unchanged
from Rev 3, flagged for Round 5 to check for completeness):** §2's
`d=K+1`-vs-`d=2K` config choice itself; §2(c)'s mod-K crash arithmetic
and §2(d)'s `h`-never-binds argument (both independently verified
CLEAN by round 4, untouched again); the Gate-2/secondary far-depth
ladder generation (`_gen_grid(K)`, unchanged); the K=48 WAVE-1b block
and the K=32 `d(K)`-grid CLOSED verdict (both stand as written, §7);
the six-rule `classify()` procedure's RULE TEXT itself and the
125-outcome partition (E5, reproduced again by round 4's own audit AND
by this revision's own sweep script — `18/4/12/8/12/8/42/15/4/2`,
Σ=125 — and not touched this revision); the `$0` K=32 reuse branch's
matched-160K scoring (E3, likewise reproduced again and unchanged);
D4's leg-attribution narrowing (still not separated, unaffected); the
11-configuration ambiguity table's ROWS themselves (unaffected by G5,
per the KW5.5 row above — only its closing paragraph's SCOPE sentence
was corrected, KW5.6); the `0.30h` supervisor margin (the audit's own
ruling — ACCEPTABLE AS-DECLARED, do NOT tighten — is adopted verbatim
by the adjudication and left untouched this revision).

**Disposition not fully implementable inside this file (disclosed,
same structural limit `§R1`/`§R2`/`§R3` already flagged, now covering
G1–G5 as a set):** every one of G1's write-ahead/recovery procedure,
G2's canonical-path copy-on-accept/exists-check, G3's exit-code
branch, G4's `run_status` enum/`validity_check` update, and G5's
trigger precondition is specified PRECISELY ENOUGH for a build-stage
implementer to follow exactly, but none of it is — and structurally
cannot be, this document being a design draft that edits no code —
actually implemented as the orchestrator script, the job-spec JSON, or
a `harvest()` patch (which, per G2's bonus simplification above, this
revision now specifies is NOT needed at all) by this revision. Round
5's audit is charged with contract-line verification of the build
stage against these five mechanisms specifically, per the
adjudication's own framing ("expected terminal").

*Rev 4, 2026-08-06. Written from direct reads of
`NCR_KWALL_ATTACK_R4.md` (861 lines, every FATAL/MAJOR/MINOR finding
in full, the per-disposition summary, and the "what I could not break"
list), this file's own `§A4-ADJUDICATION-KWALL`, and fresh Python
execution (session-local scratch script, not committed) of: (1) the
six-rule `classify()` procedure against all `5³=125` complete triples,
reproducing the 125-outcome partition to the row and spot-checked
against four of the design's own worked examples (`(0,0,0)`, `(4,0,4)`,
`(2,4,2)`, `(4,4,2)`); (2) the full 1000-vector reachable per-K state
space (10 states/K: `EXACT`×5, `DECIDED`×3, `AMBIGUOUS`×1,
`UNRESOLVED`×1) under BOTH the pre-G5 trigger rule (reproducing the
audit's own 844/156 split and 371-paid-on-unresolved figure exactly,
confirming the model's fidelity before trusting the new number) AND
the post-G5 rule (yielding the new 473/527 split with 0
paid-on-unresolved, cross-checked arithmetically against the audit's
own joint-distribution figures, `NCR_KWALL_ATTACK_R4.md` §2); (3) the
`9³=729` non-`UNRESOLVED` state space for the KW5.6
candidate-`K_trig`-set-size resweep, reproducing `{1:612,2:102,3:14,
4:1}` exactly and independently confirming (not merely re-citing) that
all 15 wide-tie cases land in band `INCOMPLETE-AT-K`. Also re-ran
`git diff HEAD~2 HEAD~1 -- matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`
this revision (9 hunks, matching the audit's own count exactly) to
independently locate and confirm the unattributed hunk (the §7
E1/E4-uniformity-bullet rewrite) is the SAME finding as KW5.8/F4's
defect, not a separate one. Section-level MD5 of all seven historical
sections (`§A1-ADJUDICATION` through `§A4-ADJUDICATION-KWALL`,
`§R1`–`§R3`) was taken BEFORE this revision's edits and re-verified
byte-identical AFTER (table above). No repo file other than this one
was created or modified; no command was run on the box; no job was
launched; no git mutation was made.*

## §A5-ADJUDICATION — AUDIT ROUND 5 VERDICT ADOPTED: **REV-REQUIRED**, ~8-LINE SURGERY; ROUND 6 SCOPE NARROWED (Fable, 2026-08-06)

`NCR_KWALL_ATTACK_R5.md`: 2 FATAL / 6 MAJOR / 9 MINOR; G5 PASS
(five-fold-verified numerics, 0 illegal dispatches), G1/G2/G4 FAIL,
G3/G6 PARTIAL. The conditional build-release checklist in the report
is ACCEPTED as the build phase's charter once Rev 5 clears.

**BINDING DISPOSITIONS for Rev 5:**
- **H1 (KW6.1):** every ledger write uses the repo's existing
  atomic-write primitive (cite it); unparseable-ledger recovery =
  CONSERVATIVE RECONSTRUCTION from disk artifacts: canonical files ⇒
  COMPLETED rows (measured spend unavailable ⇒ charge full ceiling);
  attempt-dir evidence without a canonical file ⇒ charge full
  ceiling, no retry credit. No path re-opens budget.
- **H2 (KW6.2):** ORDERING FIX — canonical copy precedes the terminal
  COMPLETED ledger row (copy-then-fold). Recovery treats a canonical
  file + dangling open record as proof of completion: charge full
  ceiling, write the COMPLETED row, never skip-and-lose. This
  restores COMPLETED⇒canonical; with G2's canonical⇒COMPLETED both
  directions hold and the harvest-patch-unnecessary claim becomes
  true BY THE PAIR — state it that way.
- **H3 (KW6.3):** correct the bound prose: worst-case TRUE spend
  15.2041h (show the arithmetic), absorbed within the declared 15.20
  + 0.30 margin structure; delete the false "crash case ≤15.00
  exactly" sentence; declared ceilings unchanged.
- **H4 (KW6.4–KW6.7):** ONE unified enum table as single source of
  truth (attempts[].status × run_status), cross-producted against
  every reachable terminal state incl. (exit 0, no JSON) and
  GATE-REFUSED; validity_check rewritten against the actual schema,
  accepting exactly the §5-reportable set — and closing the no-op
  hole: each run_status value carries its own disk-evidence
  assertion (COMPLETE ⇔ 12 canonical primaries; STOPPED-BY-OPERATOR
  with attempts=[] valid only with the stop-file's own marker;
  EXHAUSTED-BUDGET requires ledger-evidenced spend).
- **H5 (KW6.8):** the EXHAUSTED-BUDGET report gains a
  charged-vs-measured breakdown; if recovered/full-ceiling charges
  exceed 50% of the ledger, run_status = EXHAUSTED-BUDGET-SUSPECT-
  OVERCHARGE — reportable, lands in completed/, resubmission ONLY by
  explicit coordinator adjudication with a fresh ledger (never
  automatic). Disclosed, escape-hatched, still budget-safe.
- **H6:** the 9 MINORs + §R5 rows for all 17 findings.

**ROUND 6 SCOPE (binding on the next audit):** crash-path composition
(H1/H2 state-machine walk), the unified enum/validity_check text, and
the H3 arithmetic ONLY. The numeric sweeps (partition, trigger,
pricing) are FIVE-FOLD verified and excluded — do not re-run them.

---

## §R5 REVISION 5 (2026-08-06)

**Scope discipline (house convention).** §1–§7 are the LIVE body and
were edited in place per H1–H6 below — every rewrite is listed in the
"Where fixed" column of the table below, same convention as §R4.
§A1-ADJUDICATION through §A5-ADJUDICATION (and §R1–§R4 inside that
range) are UNCHANGED as historical record. Verified, not asserted:
the byte range from `## §A1-ADJUDICATION` to the end of the pre-Rev-5
file is **MD5-IDENTICAL before and after this revision**
(`df44dee31a86dc8afeed11f3d3e51024`, both). Full-file hashes, for the
record: whole file before Rev 5 = `bd8d30b783e3e23b3eec587dc253fc05`;
live body (§1–§7) before = `eeb62b6c236a0101759bb310fdcbe13d`, after =
`ef8a45873b2c2455f12aa1db41879ab5` (changed, as expected — every H1–H6
edit landed here). Status header updated to **DRAFT-R5 —
POST-AUDIT-5, AWAITING NARROW AUDIT ROUND 6**.

**Partition/trigger/pricing sections — contact disclosure (per the
coordinator's own instruction: note prominently, do not silently
touch).** The 125-outcome partition, the trigger's 1000-vector sweep
and 11-configuration table, and the core pricing chain (`F(24,25,64)`,
the per-K nominals, the ceiling-table figures, the `1.2069` margin
ratio) are **NOT re-derived or altered** — every verified NUMBER
stands exactly as G5/G6 five-fold-verified it. Three edits physically
touch text adjacent to or inside that territory, disclosed here:
(1) **H3** adds a NEW derived quantity, `T≤15.2041 GPU-h`, built
ENTIRELY from already-verified inputs (`τ=0.0157`, the `1.20h` primary
ceiling floor, the `15.00h` hard cap) — no pricing INPUT changes, only
a new arithmetic combination of existing ones, inside the "Worst-case
bound" subsection (not the "Ceiling reference table"/margin-ratio
subsections, which are untouched); (2) **KW6.11** (H6) changes ONE
digit, `0.5106→0.5105`, at the K=26 160K-nominal derivation line,
aligning a stray transcription with the ALREADY-VERIFIED `0.5105`
figure used two lines above it — the displayed PRODUCT (`0.9882`) is
unchanged, only the operand; (3) **KW6.12** (H6) corrects a FALSE CODE
COMMENT in the trigger pseudocode (`# 1, 2, or 4 candidate` →
`# 1, 2, 4, or 8 candidate`, both occurrences) — the audit's own
finding that this was "harmless to the results" (the 729-space sweep
already confirmed `{1:612,2:102,3:14,4:1}`) is unaffected; the fix is
presentational, not numeric. Round 6 should re-verify these three are
what they claim to be (a pure arithmetic add-on, a digit-alignment,
and a comment fix) rather than re-running any sweep.

**H1–H6 disposition summary (against the frozen §A5-ADJUDICATION
directives):**
- **H1 (KW6.1).** Every ledger write now goes through the repo's
  existing atomic-write primitive, named explicitly:
  **`rn.atomic_write_json`, `matrix-thinking/ncr/run_ncr.py:105-109`**
  (serializes to `<path>.tmp`, `os.replace(tmp, path)`). Unparseable-
  ledger recovery is CONSERVATIVE RECONSTRUCTION FROM DISK (new
  recovery step 0): canonical files ⇒ `COMPLETED` rows charged at the
  full ceiling (measured spend unavailable); attempt-dir evidence
  without a canonical file ⇒ one `CRASHED-RECOVERED` row at
  `attempt_n=2` (full ceiling, no retry credit — the cell derives
  TERMINAL immediately). No path re-opens budget: every row charges a
  positive amount, and a cell with zero disk evidence gets no row and
  stays genuinely available. Where: §4 "Ledger persistence + write-
  ahead recovery" (atomicity statement, schema, recovery step 0),
  dispatch loop step "write-ahead."
- **H2 (KW6.2).** Attempt-completion sequence rewritten CLASSIFY →
  COPY (if `COMPLETED`, atomic temp+rename) → FOLD. Recovery's
  dangling-`open_attempt` branch now checks for a canonical file first:
  found ⇒ proof of completion, charge full ceiling, write `COMPLETED`
  (never skip-and-lose); not found ⇒ genuine crash, `CRASHED-RECOVERED`
  as before. Crash-window table (compressed):

  | Window | Recovery outcome |
  |---|---|
  | Before copy | no canonical file ⇒ `CRASHED-RECOVERED`, full ceiling |
  | Mid-copy (`.tmp`, not renamed) | same observable as "before copy" ⇒ same outcome |
  | Between copy and fold | canonical file exists ⇒ `COMPLETED` row written, full ceiling |
  | After fold | nothing dangling — no action |

  The harvest-patch-unnecessary claim now holds BY THE PAIR:
  canonical⇒`COMPLETED` (G2, unchanged) + `COMPLETED`⇒canonical (this
  ordering, new) — stated as such, replacing Rev 4's one-directional
  claim. KW6.10 (two-phase write) is CLOSED AS A BYPRODUCT: status is
  now always classified before any row is ever written, so no
  `RETURNED-UNCLASSIFIED` placeholder is needed. Where: §4 dispatch
  loop "On return" block, recovery step 2, G2's "bonus simplification"
  paragraph (rewritten with the crash-window table).
- **H3 (KW6.3).** Corrected bound sentence, verbatim as it now reads in
  §4: *"the LEDGER value `R_N`... bounds the LEDGER, not TRUE
  GPU-hours consumed... This bounds the LEDGER, not TRUE GPU-hours
  consumed (§R5 H3 — deletes the false 'this is TIGHTER than the
  return case' claim this replaced, KW6.3)."* The false "≤15.00
  exactly" tightness claim is deleted. True-spend derivation added in
  full (`τ=0.0157`, `leak_i≤τ`, at most `⌊15.0157/1.20⌋=12` attempts
  can leak): **`T ≤ 15.00 + (1+12)(0.0157) = 15.2041 GPU-h`** —
  exceeds the tight ledger bound by `0.1884h` and the disclosed
  `15.20h` figure by `0.0041h`, both absorbed inside the declared
  `15.50h` (the `0.30h` margin now discloses three jobs: contention
  variance, process-startup, and this `0.0041h` shortfall). Declared
  ceilings (`15.00`/`15.20`/`15.50`) UNCHANGED. Where: §4 "Worst-case
  bound" (the two-case derivation, the new "True spend, worst case"
  subsection), "Rounding conservatively" paragraph, §6 "Own cost
  ceiling" bullet.
- **H4 (KW6.4–KW6.7).** ONE unified `attempts[].status`×`run_status`
  enum table is now the single source of truth (§4, immediately after
  the output-JSON schema), with an explicit precedence sentence naming
  it authoritative over the JSON-schema block (edited together, should
  never drift) and over the FROZEN historical enumerations in §R4
  (KW5.3's row, the "numbers that moved" line) — those stay
  byte-identical and are simply outranked, never edited. Cross-product
  table covers all 9 exit-code×JSON cells including the new `(exit 0,
  no JSON) → CRASHED` default arm, marks the two exit-3 cells
  UNREACHABLE against the real code, and states `GATE-REFUSED` is
  orthogonal (pre-dispatch, no exit code or JSON exists). `validity_
  check` REWRITTEN against the real schema: universal assertions
  (accept-set, `<=15.50`, `abs(...)<=1e-6` sum-equality — KW6.14 —,
  pure-ledger `K+1==d_override` — KW6.4 —, smoke-all-PASS, band/
  trigger enum membership) PLUS exactly one per-`run_status`
  disk-evidence assertion: `COMPLETE`/`COMPLETE-DEGRADED` ⇔ 12
  canonical primary files on disk (a real filesystem check, not a
  self-reported count); `STOPPED-BY-OPERATOR` ⇔ the stop-file marker
  exists on disk; `EXHAUSTED-BUDGET`/`-SUSPECT-OVERCHARGE` ⇔
  `realized_gpu_h_final > 13.80` (ledger-evidenced spend). **Confirmed:
  the audit's no-op JSON (`run_status="COMPLETE"`, `attempts=[]`,
  `realized_gpu_h_final=0.0`) now FAILS validity_check** — the
  `COMPLETE` branch's disk check finds 0 canonical files, not 12 — and
  fails under every OTHER `run_status` label it could instead claim
  (traced through all 5 in the validity_check block's own closing
  paragraph). Every legitimate degraded outcome (a genuine
  `COMPLETE-DEGRADED` with a real `GATE-REFUSED` row, a genuine
  `EXHAUSTED-BUDGET` with `realized>13.80` and low ceiling-charged
  fraction, a genuine `STOPPED-BY-OPERATOR` with the stop-file present)
  passes each branch by construction. Where: §4 output JSON schema,
  new unified-enum table, `run_status` enum definitions
  (`COMPLETE`/`COMPLETE-DEGRADED` rewritten for KW6.6), `validity_
  check` (rewritten in full), micro-smoke "Gate placement" (smoke-
  population clause for KW6.13), D5/E4 "Enforcement point" (KW6.15
  field-naming sentence).
- **H5 (KW6.8).** New `charged_vs_measured` report field
  (`measured_gpu_h`/`ceiling_charged_gpu_h`/`ceiling_charged_fraction`)
  backed by a new ledger-row field `ceiling_charged: bool`. Rule:
  `EXHAUSTED-BUDGET` + `ceiling_charged_fraction > 0.50` ⇒ report
  `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE` instead — same disk-evidence
  requirement as `EXHAUSTED-BUDGET` plus the fraction test; routed to
  `completed/` by `validity_check` (disclosed, not a bug), but
  resubmission is explicitly NEVER automatic — coordinator adjudication
  with a fresh ledger only. Disclosure paragraph added inside `EXHAUSTED-
  BUDGET`'s own definition naming the environment-fault failure mode
  directly. Where: §4 output JSON schema, `run_status` enum (new value
  + `EXHAUSTED-BUDGET` disclosure paragraph), `validity_check`,
  ledger-row schema (`ceiling_charged` field).
- **H6.** All 9 MINORs dispositioned (table below) + this table itself.

**§R5 finding-by-finding table (all 17):**

| Finding | Severity | Disposition | Where |
|---|---|---|---|
| KW6.1 | FATAL | FIXED (H1) | §4 ledger persistence/recovery |
| KW6.2 | FATAL | FIXED (H2) | §4 dispatch loop, recovery step 2, G2 |
| KW6.3 | MAJOR | FIXED (H3) | §4 worst-case bound, §6 own-cost-ceiling |
| KW6.4 | MAJOR | FIXED (H4) | §4 ledger schema, validity_check |
| KW6.5 | MAJOR | FIXED (H4) | §4 dispatch loop, unified enum table |
| KW6.6 | MAJOR | FIXED (H4) | §4 run_status enum |
| KW6.7 | MAJOR | FIXED (H4) | §4 validity_check |
| KW6.8 | MAJOR | FIXED (H5) | §4 run_status enum, schema, validity_check |
| KW6.9 | MINOR | **ACCEPTED-BY-ADDENDUM** (H6) — see below | this §R5 note (not §R4, which stays frozen) |
| KW6.10 | MINOR | FIXED, subsumed by H2 | §4 dispatch loop "On return" |
| KW6.11 | MINOR | FIXED (1-digit) | §4 160K nominal derivation line |
| KW6.12 | MINOR | FIXED (comment) | §4 trigger pseudocode, both occurrences |
| KW6.13 | MINOR | FIXED | §4 micro-smoke "Gate placement" |
| KW6.14 | MINOR | FIXED | §4 validity_check |
| KW6.15 | MINOR | FIXED | §4 D5/E4 "Enforcement point" |
| KW6.16 | MINOR | FIXED (attribution) | §4 11-config table closing paragraph |
| KW6.17 | MINOR | FIXED | §4 recovery step 2 |

**KW6.9 — ACCEPTED-BY-ADDENDUM, attackable sentence.** The audit's
discharge asks for edits INSIDE §R4's own "Where fixed" table
(attributing §7's E1/E4-uniformity bullet to KW5.3/KW5.5, and §6's
red-team items (vi)–(ix)/"Own cost ceiling" bullet to KW5.1/KW5.7/
KW5.10). §R4 is a FROZEN historical section under this revision's own
house convention and this agent's explicit mandate — it is not edited.
The correction is recorded here instead, as the authoritative
attribution: **§7 (E1/E4-uniformity bullet) belongs to KW5.3/KW5.5;
§6 (red-team items (vi)–(ix), "Own cost ceiling" bullet) belongs to
KW5.1/KW5.7/KW5.10.** *Attackable:* a reader who consults §R4 alone,
without also reading §R5, still sees the incomplete attribution table
— this addendum does not retroactively fix that document. Mitigated,
not eliminated: (i) this is pure bookkeeping with zero effect on the
design's mechanism, ceilings, or any reported number; (ii) §R5 is
newer and, per this document's own front-to-back reading convention,
is the record a round-6+ auditor consults last and treats as
authoritative on precedence; (iii) the SAME pattern (an unattributed
hunk in a live-at-the-time revision log) was accepted as cosmetic for
§A3/§A4's separator-append artifact by round 4's own ruling — this is
the equivalent move one layer up, applied to content instead of
punctuation, and equally non-load-bearing.

**Not fully implemented, with reason:** none. All six binding
dispositions (H1–H6) and all 17 findings have a disposition recorded
above. The one item that is a DEFERRED BUILD-STAGE TASK rather than a
design gap — the same deferral pattern already accepted for the
job-spec template and the micro-smokes — is KW2.7's on-box
`fallback_pool/`/`claimed/` sweep, unchanged from Rev 4 and still
listed in the audit's own conditional build-release checklist (§9),
not a Rev-5 discharge item.

## §A6-ADJUDICATION — AUDIT ROUND 6 VERDICT ADOPTED: **REV-REQUIRED**; RECONSTRUCTION + VALIDITY ONLY REMAIN (Fable, 2026-08-06)

`NCR_KWALL_ATTACK_R6.md`: 2 FATAL / 3 MAJOR / 10 MINOR. SETTLED this
round (excluded from all future rounds): crash windows 1–8 incl. the
copy-then-fold closure of KW6.2 (verified line-exact), the H3
arithmetic (two routes, 13-multiplier provenance), the enum precedence
mechanism, and the no-op rejection. KW6.9's outcome stands on the
audit's REPLACEMENT justification (editing frozen §R4 would destroy
the §A1→EOF MD5 identity), which is ADOPTED. The audit's §9 build-
charter additions (per-attempt-dir reconstruction contract; Rev-6
validity one-liner; red-team items i–xiv; negative-test-run-to-
completion; the os.makedirs build assertion) are ADOPTED into the
build charter.

**BINDING DISPOSITIONS for Rev 6:**
- **I1 (KW7.1):** reconstruction rebuilt PER-ATTEMPT from attempt
  dirs (the ground truth of dispatches): each attempt dir ⇒ one
  ledger row, charged at measured elapsed if its JSON parses, else
  full ceiling; conditional-arm cells reconstruct from the same
  attempt-dir evidence; a present-but-unparseable canonical file ⇒
  quarantine-rename (canonical.CORRUPT-<ts>), cell treated per its
  attempt evidence (full ceiling, NOT COMPLETED), exists-check no
  longer trips. State totality: every disk state maps to exactly one
  rule.
- **I2 (KW7.2):** before declaring PERSISTENTLY-ABORTED, recovery
  consults the attempt-dir JSON: parseable + COMPLETED ⇒ promote to
  canonical via the same temp+atomic-rename path, charge measured
  elapsed — a completed cell is never lost to a pre-copy crash.
- **I3 (KW7.3):** one precedence sentence: reconstruction's attempt
  numbering is authoritative; resume at attempt_n = max(reconstructed
  n)+1 when ≤2 and the cell is non-terminal; never below, never
  re-dispatch a numbered attempt.
- **I4 (KW7.4):** validity_check branches quantified over the FULL
  pre-registered outcome space: INCOMPLETE-AT-K runs are reportable
  and must PASS with their own evidence clause (per-K canonical
  counts consistent with the reported band vector); COMPLETE's
  12-primaries assertion applies only when no K is INCOMPLETE;
  COMPLETE-DEGRADED sub-case (i)'s assertion corrected to the count
  it actually implies. Test list in-text: every §5-reportable outcome
  passes, every R5/R6 adversarial JSON fails.
- **I5 (KW7.5):** label-vs-disk consistency: EXHAUSTED-BUDGET
  additionally requires at least one gated-out cell with no canonical
  evidence — a label contradicting a complete disk state is rejected.
- **I6:** 10 MINORs incl. the KW7.9 §6-hunk disclosure; §R6 rows for
  all 15 findings.

**ROUND 7 SCOPE (binding):** the I1–I3 reconstruction contract and the
I4–I5 validity text ONLY. Everything else is settled and excluded.

---

## §R6 REVISION 6 (2026-08-06)

**Scope discipline (house convention, same as §R4/§R5).** §1–§7 are
the LIVE body; every edit this revision implements a binding I1–I6
disposition from `§A6-ADJUDICATION` above and is listed in the
disposition table below. `§A1-ADJUDICATION` through
`§A6-ADJUDICATION` (and `§R1`–`§R5` inside that range) are UNCHANGED
as historical record — verified, not asserted (MD5 block below). This
design now carries status **DRAFT-R6 — POST-AUDIT-6, AWAITING NARROW
AUDIT ROUND 7** (status header updated at the top of this file).

**MD5 verification, run before and after this revision's edits.**

| Quantity | Value |
|---|---|
| Whole file, before this revision | `cee7e8136a63028ab420f30ab2769cf4` |
| Frozen range `## §A1-ADJUDICATION` → EOF, before | `9d07f2879e25de26cab512465ba8aa90` (960 lines) |
| Frozen range `## §A1-ADJUDICATION` → EOF, after | `9d07f2879e25de26cab512465ba8aa90` (960 lines) — **IDENTICAL, independently reproduced** |
| Live body (§1–§7), before | `e92b343202888e9a948769cd5ff5843b` (2495 lines) |
| Live body (§1–§7), after (pre-`§R6`-append; recomputed as the LAST step, after two small collateral cross-reference fixes the I1/I3 edits required — see below) | `68440ddc8fc7408168daa8ce4ef2f090` (2748 lines) — **changed, as expected**: every I1–I6 edit landed here |

**No whole-file "after" hash is stated (§R6, following §R5's own
convention, not repeating KW7.10's mistake for a different reason):**
a hash of the finished file, printed inside that same file, is a
fixed point that cannot be satisfied by construction — the moment the
value is written, the file (and its true hash) changes. `§R5` avoided
this correctly for the whole file but its one attempt at a **non**
self-referential figure (live body after) still failed to reproduce,
per KW7.10 — a real computation slip, not a self-reference issue. The
two figures given above (live body before/after) are recomputed here
directly from `git show HEAD:…` and the current on-disk file, not
carried forward from any prior claim.

**Line-count arithmetic, visibly derived.** `3455` (pre-Rev-6 total)
`= 2495` (live body) `+ 960` (frozen) `+ 0` (no separator line double
counted — confirmed by direct `wc -l` on each range). Live body grew
`2748 − 2495 = 253` lines this revision (two more than the I1–I6 edits
alone: KW7.1/KW7.3's reconstruction rewrite left two SHORT downstream
cross-references stale — a dispatch-loop forward-pointer into the old
step-3 wording, and the unified enum table's `CRASHED-RECOVERED`/
`COMPLETED` "Reachable via" cells, which named the old fixed
`attempt_n:2` convention and the old single-branch reconstruction path
by name; both were corrected in place as a direct, in-scope
consequence of I1/I3, not a new finding); frozen stayed `960`;
`2748 + 960 = 3708` (file length immediately before this `§R6` section
was appended, confirmed by direct `wc -l`). This `§R6` section's own
length is additive on top of `3708` — the file's final `wc -l` after
this append is the visible check, not a value pre-declared here.

**Settled-section contact — disclosed prominently, per the
coordinator's own instruction (three edits physically land in text the
`§A6-ADJUDICATION` scope named SETTLED; none re-derives or reverses a
verified number).**
1. **The unified-enum-table precedence sentence** (§4, "Unified
   `attempts[].status` × `run_status` enum table") is inside "the enum
   precedence mechanism," named settled. Two MINOR, non-numeric fixes
   landed there: KW7.8's stale `§R4` line pointers (Rev 5 shifted §R4
   by +472 lines; the pointers now targeted unrelated §R1 prose) are
   replaced with name-only citations, which cannot go stale the same
   way; KW7.12's gap (the precedence sentence covered enumerations but
   not `§R4`'s still-live refuted-TIGHTER-claim row) is closed by one
   added clause. Neither edit touches the mechanism's LOGIC or any
   verified number — the table it describes is untouched.
2. **The "Crash-window walk" table** (§4, §R5 H2) is inside "crash
   windows 1–8," named settled. I2's fix to the recovery procedure's
   dangling-`open_attempt` branch (a genuinely in-scope KW7.2 edit)
   makes the table's OLD "Before copy starts"/"Mid-copy" rows
   factually wrong the moment I2 lands (they describe the discarded,
   pre-fix behavior). Left uncorrected, the table would directly
   contradict the procedure it summarizes two paragraphs above it —
   worse than the contact. Both rows are updated to state I2's
   promotion outcome; the "Between copy and fold"/"After fold" rows
   (KW6.2's actual settled FATAL closure) are untouched.
3. **The H3 "True spend, worst case" paragraph** (§4, §R5 H3) is
   inside "the H3 arithmetic," named settled. Two MINOR, symbol-only
   fixes landed there: KW7.6 renames `L_final`→`R_N` (the audit found
   the SYMBOL wrong, not the NUMBER — `15.2041` is unchanged, re-derived
   below); KW7.7 adds one sentence noting the paragraph's monotone-
   ledger premise is now GUARANTEED by I1's per-attempt reconstruction
   rather than merely assumed — both audit-suggested discharges for
   these two findings, applied with zero numeric re-derivation.
   **Re-verified after the edit, not merely asserted:**
   `13 × 0.0157 = 0.2041`; `15.00 + 0.2041 = 15.2041` — unchanged, H3
   still **PASSES**.

**Disposition table — all 15 R6 findings.**

| Finding | Severity | §R6 disposition | Where |
|---|---|---|---|
| KW7.1 | FATAL | **FIXED (I1).** Reconstruction rewritten per-attempt-directory; 24-state total function (2 dir × 3 JSON × 3 canonical, 12 invalid `dir=absent`×`JSON∈{parseable,unparseable}` combinations removed = 24 valid, `arm` a ceiling-value split of 12 core rows), quarantine-rename rule for a corrupt canonical, conditional arm scanned unconditionally. | §4, recovery procedure step 0 (0.0/0.1/0.2) |
| KW7.2 | MAJOR | **FIXED (I2).** Both the live dangling-`open_attempt` branch AND reconstruction's per-attempt table now read the attempt's own archival JSON and PROMOTE a provably-`COMPLETED` attempt before any `CRASHED-RECOVERED`/`PERSISTENTLY-ABORTED` path can fire. | §4, recovery procedure step 2.1; step 0.1 table rows |
| KW7.3 | MAJOR | **FIXED (I3).** Resume rule restated cell-wise, keyed off DERIVED terminal state, never "no ledger row"; one precedence sentence makes reconstruction's own `attempt_n` the sole numbering authority (`max(attempt_n)+1`, never below, never re-dispatch a numbered attempt). | §4, recovery procedure step 3–4 |
| KW7.4 | FATAL | **FIXED (I4).** `COMPLETE`'s 12-canonical assertion now conditional on `band.interval_resolved_Ks`/`incomplete_at_K` both being empty; `COMPLETE-DEGRADED`'s assertion replaced with the identity (terminal disposition × canonical-count-equals-`COMPLETED`-row-count × throttle evidence). Both G4's prose and the job-spec `validity_check` branches updated in lockstep. | §4, G4 enum definitions; Job-spec `validity_check` per-`run_status` branches |
| KW7.5 | MAJOR | **FIXED (I5).** `EXHAUSTED-BUDGET`/`-SUSPECT-OVERCHARGE` gain the negative clause: `<12` canonical primaries AND ≥1 primary-arm first-attempt `GATE-REFUSED` row, alongside the existing `>13.80` clause — provably disjoint from `COMPLETE`/`COMPLETE-DEGRADED` on disk now, not by prose alone. | §4, G4 enum definitions; `validity_check` per-`run_status` branches |
| KW7.6 | MINOR | **FIXED**, forced contact with the settled H3 paragraph (disclosed above) — `L_final`→`R_N` rename, zero numeric change. | §4, "True spend, worst case" |
| KW7.7 | MINOR | **DISCHARGED AS A SIDE EFFECT OF I1**, plus one disclosure sentence, forced contact with the settled H3 paragraph (disclosed above) — the bound's monotone-ledger premise is now true by construction (I1 never reduces `realized_gpu_h`), not merely assumed. | §4, "True spend, worst case" |
| KW7.8 | MINOR | **FIXED**, forced contact with the settled precedence sentence (disclosed above) — stale `§R4` line pointers replaced with name-only citations. | §4, unified enum table precedence sentence |
| KW7.9 | MINOR | **ADDENDUM, not a live edit** — the correct attribution (§6 red-team items (viii), (x)–(xiv) belong under H1/H2/H4/H5's "Where fixed") is recorded in the addendum below; `§R5`'s own table stays frozen (editing it would break the `§A1→EOF` MD5 identity, the same trade `§A6-ADJUDICATION` already ratified for KW6.9). | Addendum below; `§R5` itself untouched |
| KW7.10 | MINOR | **DISCLOSED, not fixed** — `§R5`'s claimed live-body-`after` MD5 is frozen and cannot be corrected in place; the CORRECT figures for the `§R4→§R5` transition are recorded in the addendum below for anyone who needs them. `§R6`'s own MD5 block (above) was independently recomputed from `git show`/the live file, not carried forward from any prior claim. | Addendum below; `§R5` itself untouched |
| KW7.11 | MINOR | **FIXED.** `STOPPED-BY-OPERATOR`'s G4 paragraph now states the stop-file marker check is the orchestrator's own pre-write self-check, not a `validity_check` branch (which cannot reach it — universal assertion 1 excludes the label first). | §4, G4 `STOPPED-BY-OPERATOR` bullet |
| KW7.12 | MINOR | **FIXED**, forced contact with the settled precedence sentence (disclosed above) — one clause extends the precedence sentence's scope from enumerations to arithmetic, naming `§R4`'s KW5.1 row as outranked by "True spend, worst case." | §4, unified enum table precedence sentence |
| KW7.13 | MINOR | **ADOPTED per `§A6-ADJUDICATION`** (no further edit needed — the adjudication already ruled the audit's REPLACEMENT justification, the MD5-identity argument, in as KW6.9's operative reasoning). | `§A6-ADJUDICATION` above (no live-body edit) |
| KW7.14 | MINOR | **FIXED.** Universal assertion 6's placeholder expanded to the six literal §5 band labels plus `"INCOMPLETE-AT-K"`; the `[NON-MONOTONE]` tag and the 160K qualifier bands are noted as separate, un-asserted fields. | §4, Job-spec `validity_check` universal assertion 6 |
| KW7.15 | MINOR | **FIXED.** `os.makedirs(outdir, exist_ok=True)` / `ncr_earlyln_scale.py:237` cited explicitly as the premise behind "attempt dir absent ⇒ no evidence"; "provably never ran" softened to "left no evidence this design's dispatch path can produce." | §4, recovery procedure step 0.1 |

**Addendum — KW7.9 (§6 red-team attribution, corrected without editing
frozen `§R5`).** `§R5`'s own disposition table names only H3 ("§6 'Own
cost ceiling' bullet") among its "Where fixed" entries touching §6; the
two hunks that actually added red-team items (x)–(xiv) and updated
item (viii)'s accept-set belong, by content, under **H1** (items x, xi
— the ledger-corruption and copy-window tests), **H2** (item xi,
shared), **H4** (items viii, xii — the `validity_check`/`GATE-REFUSED`
tests), and **H5** (items xiii, xiv — the GPU-reap and
forced-overcharge tests). This correction is recorded here as the
attribution of record; `§R5`'s table itself is not retroactively
edited, for the same reason `§A6-ADJUDICATION` already gave for KW6.9
(the `§A1→EOF` MD5 identity is a real cross-round integrity
instrument, worth more than one bookkeeping cell).

**Addendum — KW7.10 (`§R4`→`§R5` transition MD5, corrected without
editing frozen `§R5`).** `§R5`'s own claimed live-body-`after` figure
(`ef8a45873b2c2455f12aa1db41879ab5`) does not reproduce against the
`§R4→§R5` boundary (KW7.10, independently confirmed by this round's
audit, not re-litigated here). The two hashes that DO verify from that
same transition — whole file before Rev 5
(`bd8d30b783e3e23b3eec587dc253fc05`) and live body before
(`eeb62b6c236a0101759bb310fdcbe13d`) — are unaffected and stand. No
corrected `§R4→§R5` live-body-after figure is manufactured here (this
round has no way to reconstruct the exact intermediate file state
`§R5`'s edits produced immediately before its own append, and
guessing one would create a second unverifiable claim); the reader is
directed to `git show` against the `§R4`→`§R5` commit boundary
directly if that specific historical figure is ever needed.

**Confirmation — the two outcomes `§A6-ADJUDICATION` named for direct
walk-through both now PASS `validity_check`, and every R5/R6
adversarial JSON still FAILS (traced in full, in-text, at the Job-spec
`validity_check` section's new "In-text test list" — recapped here for
the record):**
- `INCOMPLETE-AT-K` (one primary seed `CRASHED` on both attempts, 11
  canonical primaries, `run_status="COMPLETE"`, `band.incomplete_at_K`
  naming the affected K) — **PASSES** `COMPLETE`'s OTHERWISE branch
  (per-K canonical counts consistent with the disclosed field). Failed
  under Rev 5's unconditional 12-canonical clause; this is KW7.4's
  FATAL, discharged.
- `COMPLETE-DEGRADED` sub-case (i) *primary-retry-refused* (11
  canonical primaries, one `GATE-REFUSED` retry row) — **PASSES** the
  new identity assertion (terminal disposition for all 12 cells ×
  canonical-count-equals-`COMPLETED`-row-count × ≥1 `GATE-REFUSED`
  row). Could never pass Rev 5's identical-to-`COMPLETE` 12-canonical
  clause by its own construction; this is the second half of KW7.4's
  FATAL, discharged.
- Audit-R5 no-op, near-miss #1 (`EXHAUSTED-BUDGET` at 12 canonical),
  and near-miss #2 (`STOPPED-BY-OPERATOR`, no stop-file) — **all
  three still FAIL**, by the assertions named in-text at the Job-spec
  section above; near-miss #1 is the one whose failure mode CHANGED
  (previously incorrectly accepted, KW7.5's FATAL-adjacent hole — now
  correctly rejected by the new negative clause).

**Round 7 should, per `§A6-ADJUDICATION`'s own binding scope, verify
I1–I5 only — plus, since three MINOR fixes forced contact with
sections that scope named settled (disclosed above, all three
non-numeric), confirm those three contacts are exactly what they
claim: a symbol rename and a monotonicity note in the H3 paragraph, a
citation-only fix in the precedence sentence, and two crash-window-
table rows updated to match I2's now-live promotion behavior — none of
which re-opens the H3 number, the enum precedence mechanism's logic,
or crash windows 3/4/5/6/7/8 (unaffected, still settled).**

## §A7-ADJUDICATION — AUDIT ROUND 7 VERDICT ADOPTED: **REV-REQUIRED** (Fable, 2026-08-06)

`NCR_KWALL_ATTACK_R7.md`: 2 FATAL / 4 MAJOR / 6 MINOR. NEWLY SETTLED
(excluded henceforth): the 24-state derivation + impossibility
argument, 24/24 totality, quarantine placement, promotion preemption
of every PERSISTENTLY-ABORTED path, resume numbering (0/200
violations), the 6 legitimate validity outcomes PASSING (KW7.4
discharged), integrity/citations. The report's §9 build charter
(R5 + R6's five + R7's five additions, incl. wiring the 13-payload and
200-composition suites as build unit tests with forced-fail negatives)
is ADOPTED as the standing charter.

**BINDING DISPOSITIONS for Rev 7:**
- **J1 (KW8.1):** COMPLETE's OTHERWISE branch gains a positive-
  evidence clause — a non-empty incomplete_at_K requires, per listed
  K, attempt-dir evidence of two terminal non-COMPLETED attempts;
  attempts=[] can never claim incompleteness. Re-trace the R5 no-op
  against the amended branch in-text.
- **J2 (KW8.2):** trigger.resolution is a BARE ENUM LITERAL; the
  candidate list moves to a separate resolution_detail field;
  universal assertion 6 unchanged. The trigger pseudocode's f-string
  is corrected at both cited lines.
- **J3 (KW8.3):** rule 0.2's guard keyed on "no COMPLETED row for
  this cell" (not "zero rows appended"); re-derive the 200-state
  composition counts under the fix; G2's abort-loudly sentence
  re-worded to the truth.
- **J4 (KW8.4):** EXHAUSTED-BUDGET and EXHAUSTED-BUDGET-SUSPECT-
  OVERCHARGE made mutually exclusive: the plain label additionally
  asserts recovered-charge-fraction ≤ 50%; exactly one label can hold
  for any ledger.
- **J5 (KW8.5):** conditional-arm disk evidence: any report carrying
  qualifier_band asserts either the 4 conditional canonical files
  (disjoint from primary canonical space — assert the disjointness)
  or the $0-branch archive citation; fabricated qualifier_band with
  no evidence must fail.
- **J6 (KW8.6):** restore the row-wise invariant "ledger row ≥ that
  attempt's true spend": reconstruction/recovery rows charge FULL
  CEILING except promotion rows (completed work), which charge
  measured elapsed PLUS a startup allowance derived from the design's
  own overhead terms (t0 sits after CUDA init — cite it); the H3
  bound re-derived visibly under the corrected charging, and the
  false "I1 establishes the premise" sentence corrected.
- **J7:** 6 MINORs + §R7 rows for all 12 findings.

**ROUND 8 SCOPE (binding):** J1–J6 verification + re-running the two
audit scripts (vcheck/recon suites) against the amended text. All
PASS-marked material above is excluded.

## §R7 REVISION 7 (2026-08-06)

**Scope discipline (house convention, same as §R4/§R5/§R6).** §1–§7 are
the LIVE body; every edit this revision implements a binding J1–J7
disposition from `§A7-ADJUDICATION` above and is listed in the
disposition table below. `§A1-ADJUDICATION` through
`§A7-ADJUDICATION` (and `§R1`–`§R6` inside that range) are UNCHANGED
as historical record — verified, not asserted (MD5 block below). This
design now carries status **DRAFT-R7 — POST-AUDIT-7, AWAITING NARROW
AUDIT ROUND 8 (not build-released, not queue-eligible)** (status
header updated at the top of this file).

**MD5 verification, run before and after this revision's edits.**

| Quantity | Value |
|---|---|
| Whole file, before this revision | `4e03ed5d13b2139e123ab079bbc0517e` (3927 lines) |
| Frozen range `## §A1-ADJUDICATION` → end-of-pre-Rev-7 content, before | `3805e7dac8893f272f51fb62210e28be` (1179 lines) |
| Frozen range `## §A1-ADJUDICATION` → end-of-pre-Rev-7 content, after | `3805e7dac8893f272f51fb62210e28be` (1179 lines) — **IDENTICAL, independently reproduced** (§R7 uses `§R5`'s precise phrasing throughout, per KW8.10's own discharge, below — never the ambiguous "→ EOF" wording `§R6` regressed to) |
| Live body (§1–§7), before | `68440ddc8fc7408168daa8ce4ef2f090` (2748 lines) |
| Live body (§1–§7), after (pre-`§R7`-append) | `55ba3e9a9289e10f5e7fde5864c21970` (3116 lines) — **changed, as expected**: every J1–J7 edit landed here. **(§R8 K5 — corrects KW9.5's MAJOR: the figure previously stated here, `1f93fa4ca8ee7333d573d5d095b37453`, matched no prefix range of the committed file at all — a brute-force sweep by the R8 audit over every plausible line range found nothing. Recomputed directly against `7a0917d` (`sed -n '1,3116p' <file> | md5`) for this correction; the "before" row, the frozen-range rows, and the arithmetic `3927=2748+1179`/`3116-2748=368`/`3116+1179=4295` all independently verified exact by the R8 audit and are untouched — nothing settled was disturbed, only this one stale figure.)** |

**No whole-file "after" hash is stated (§R7, following §R5/§R6's own
convention — a hash of the finished file, printed inside that same
file, is a fixed point that cannot be satisfied by construction).**
Both figures above are recomputed directly from the on-disk file
before and after this revision's edits, not carried forward from any
prior claim.

**Line-count arithmetic, visibly derived.** `3927` (pre-Rev-7 total)
`= 2748` (live body) `+ 1179` (frozen) `+ 0` (no separator line double
counted — confirmed by direct `wc -l` on each range). Live body grew
`3116 − 2748 = 368` lines this revision (J1–J7's edits: the trigger
pseudocode fix ×2, the schema `resolution_detail` field, the
primary/conditional disjointness paragraph, the 0.1 table's charging
rewrite + KW8.8/KW8.9 disclosures + KW8.11 wording fix + KW8.12
cross-reference, the 200-composition re-run table + its embedded
0/200 counts, the 0.2 bootstrap guard/attempt_n/charging rewrite, the
live-recovery PROMOTE-branch allowance, the G2 abort-loudly reword,
the derived-cell-state KW8.7 precedence clause, the `COMPLETE`
OTHERWISE-branch J1 clauses (G4 prose + `validity_check` branch,
twice), the `EXHAUSTED-BUDGET`/`-SUSPECT-OVERCHARGE` J4 dichotomy (G4
prose + `validity_check` branches, twice), the H3 paragraph's full
two-class re-derivation, the "Rounding conservatively" forced-contact
fix, the unified-enum-table precedence sentence's forced numeric
update, the §6 "Own cost ceiling" bullet's forced numeric update, the
universal assertion 7 (§R7 J5), and four new in-text test-list
entries); frozen stayed `1179`; `3116 + 1179 = 4295` (file length
immediately before this `§R7` section was appended, confirmed by
direct `wc -l`). This `§R7` section's own length is additive on top of
`4295` — the file's final `wc -l` after this append is the visible
check, not a value pre-declared here.

**Settled-section contact — disclosed prominently, per the house
convention `§R6` re-affirmed (three edits physically land in text a
prior round's own adjudication named settled or PASS; none re-derives
or reverses a verified LOGIC path, only propagates a corrected NUMBER
that same edit produced).**
1. **The H3 "True spend, worst case" paragraph itself** (§4) is the
   DIRECT, AUTHORIZED target of J6/KW8.6 — not a forced incidental
   contact but the round's own primary edit site. Recorded here anyway
   for completeness: `§A7-ADJUDICATION` did not blanket-exclude it (J6
   explicitly reopens it), and the audit's own §0 SCOPE note excluding
   "the H3 arithmetic itself" from general re-verification refers to
   the NUMBER `15.2041`, which this revision does not silently
   overturn — it re-derives a DIFFERENT, MORE COMPLETE bound
   (`15.3737h`) under a two-class charging model the OLD bound never
   accounted for, with the arithmetic shown in full, not asserted.
2. **The unified-enum-table precedence sentence** (§4, "the same
   outranking covers arithmetic … by §4's 'True spend, worst case'
   derivation") is inside "the unified-enum-table precedence
   sentence," named settled by `§A6-ADJUDICATION`'s scope and touched
   again by `§R6` itself (KW7.8/KW7.12, disclosed there). One
   NUMERIC-ONLY edit lands here this revision: the cited figure
   `T ≤ 15.2041h` is updated to `T ≤ 15.3737h`, the direct, necessary
   consequence of J6's re-derivation — leaving this cross-reference
   uncorrected would have the document contradict itself two sections
   apart. The sentence's LOGIC (that this derivation outranks `§R4`'s
   frozen KW5.1 row) is unchanged.
3. **§6 POOL-ELIGIBILITY STATEMENT's "Own cost ceiling" bullet**
   restates the same H3 figures for the pool-spec audience. Two
   NUMERIC-ONLY edits land here: `15.2041h`→`15.3737h` and the
   `0.0041h`/`0.1737h` shortfall-vs-margin figures, both the same
   direct, necessary consequence of J6. The bullet's STRUCTURE (one
   declared `15.50h` ceiling = `15.20h` rounded bound + `0.30h` stated
   margin) is unchanged — the margin still comfortably absorbs the
   corrected shortfall (`0.30 ≫ 0.1737`, `0.1263h` headroom remaining).

**Confirmation — both re-run suites executed to completion this
revision, not hand-checked (per `§A7-ADJUDICATION`'s own ROUND 8 SCOPE
instruction, executed a round early since the fixes and their
verification were produced together):**
- **200-state cell-level composition** (`recon_r7.py`, this revision's
  session scratchpad, same 5×5×4×2=200 shape as the audit's own
  `recon.py`): OLD guard reproduces the audit's own KW8.3 figures
  exactly, **30/200 orphans, 6/200 abort-trips**; NEW guard (§R7 J3)
  drives both to **0/200**. Embedded in full at the 200-state
  composition table, above.
- **`validity_check` payload suite** (`vcheck_r7.py`, this revision's
  session scratchpad, transcribing the amended universal +
  per-`run_status` assertions verbatim): all 6 legitimate outcomes
  (L1–L6) PASS; all 7 R5/R6/R7 adversarial payloads (A1–A7, including
  KW8.1's ENHANCED no-op ×2, KW8.4's mislabelled-suspect run, and
  KW8.5's fabricated-conditional-arm payload) FAIL, each by the named
  clause; a 14th, newly-added `tie-break-min` legitimate payload
  PASSES (§R6's own producer bug, KW8.2, made this the one
  legitimately-reportable outcome that used to fail). **14/14 payloads
  match expectation.**

**Disposition table — all 12 R7 findings.**

| Finding | Severity | §R7 disposition | Where |
|---|---|---|---|
| KW8.1 | FATAL | **FIXED (J1).** `COMPLETE`'s OTHERWISE branch gains the same two positive-evidence clauses `COMPLETE-DEGRADED` already carried: every primary `(K,seed)` has ≥1 row in `ledger.attempts`, AND canonical count equals the distinct-`COMPLETED`-primary-row count. The ENHANCED no-op (`attempts=[]`, `band.incomplete_at_K`/`interval_resolved_Ks` disclosed) now FAILS clause (a), traced in-text. | §4, G4 `COMPLETE` definition; Job-spec `validity_check` `COMPLETE` branch; in-text test list |
| KW8.2 | FATAL | **FIXED (J2).** Trigger pseudocode's f-string, both cited sites, now returns the bare literal `"tie-break-min"`; the candidate list moves to the new `trigger.resolution_detail` schema field. Universal assertion 6's text is unchanged, as directed — only the value it is now given changed. A `tie-break-min` payload traced PASSING in-text (previously FAILED under the f-string producer). | §4, trigger pseudocode (both copies); Output JSON schema; in-text test list |
| KW8.3 | MAJOR | **FIXED (J3).** 0.2's guard re-keyed from "0.1 appended ZERO rows" to "0.1 appended no `COMPLETED` row"; bootstrap `attempt_n` now `max(recorded,0)+1`, never a colliding hard-coded `1`. 200-state composition re-run by execution: OLD guard reproduces 30 orphans/6 abort-trips exactly; NEW guard drives both to 0/200. G2's "only way to trip it" sentence corrected to name reconstruction as a (now-closed) second producer. | §4, recovery procedure step 0.2; G2 exists-check paragraph; embedded 200-state composition table |
| KW8.4 | MAJOR | **FIXED (J4).** `EXHAUSTED-BUDGET` gains `ceiling_charged_fraction<=0.50`; `-SUSPECT-OVERCHARGE`'s existing `>0.50` is now its exact mirror. The two labels partition the shared base disk state exactly at `0.50` — a dichotomy, stated and traced (A6 payload FAILS plain `EXHAUSTED-BUDGET`, PASSES the `-SUSPECT-OVERCHARGE` claim instead). | §4, G4 `EXHAUSTED-BUDGET`/`-SUSPECT-OVERCHARGE` definitions; Job-spec `validity_check` branches; in-text test list |
| KW8.5 | MAJOR | **FIXED (J5).** New universal assertion 7: a non-null `conditional.qualifier_band` requires EITHER 4 conditional canonical files (`launched=true`) XOR the $0-branch archive citation (`launched=false`, `K_trig==32`). Primary/conditional canonical-directory disjointness stated explicitly (sibling paths, neither a prefix of the other) and cross-referenced to the existing §9(d) build assertion. Fabricated-`qualifier_band` payload traced FAILING both clauses in-text. | §4, new universal assertion 7; new disjointness paragraph; in-text test list |
| KW8.6 | MAJOR | **FIXED (J6).** Row-wise invariant restored: reconstruction/recovery rows charge FULL CEILING except `status=="COMPLETED"` ("promotion") rows, which charge measured `elapsed_h` PLUS a startup allowance `s=0.0053` GPU-h, derived from `§R4`'s own frozen KW5.7 estimate (`≈0.016–0.021h` true single-attempt tail, minus the already-priced `τ=0.0157h`) and cited against `ncr_earlyln_scale.py:237-257`. H3 re-derived two-class: `T ≤ R_N + 12τ + 32s = 15.3737` GPU-h, still inside the declared `15.50h` (`0.1263h` headroom). The false "I1 establishes the premise" sentence replaced with the honest two-class accounting. | §4, 0.1/0.2 reconstruction tables; live-recovery PROMOTE branch; H3 "True spend, worst case" (RE-OPENED, full re-derivation); "Rounding conservatively" (forced numeric contact); unified-enum-table precedence sentence + §6 "Own cost ceiling" (forced numeric contacts, disclosed above) |
| KW8.7 | MINOR | **FIXED.** One precedence clause: a cell with any `COMPLETED` row is `COMPLETED`, full stop — the `PERSISTENTLY-ABORTED` clause is evaluated only once that is ruled out. Closes the 24/200 dual-derivation states; harmless in practice (both were terminal) but now a genuine function. | §4, "Derived CELL state" definition |
| KW8.8 | MINOR | **DISCHARGED, disclosure only.** One sentence: the `canonical_state` snapshot is deliberate, not refreshed by a sibling slot's own PROMOTE within the same 0.1 pass; every downstream consumer counts DISTINCT `(K,seed)` pairs (set-based), which is what makes an occasional harmless double-`COMPLETED`-row safe. | §4, 0.1 intro paragraph |
| KW8.9 | MINOR | **FIXED.** One clause: a parseable JSON with a missing or non-enum `status` key is treated as unparseable (row 10, full ceiling) — the same rule 0.0 already applies to the canonical side. | §4, 0.1 intro paragraph; row 10 (0.1 table) |
| KW8.10 | MINOR | **ADDENDUM, not a live edit** (`§R6` is frozen — the same REPLACEMENT precedent `§A6-ADJUDICATION` ratified for KW6.9 and `§R6` itself applied to KW7.9/KW7.10) — recorded below. | Addendum below; `§R6` itself untouched |
| KW8.11 | MINOR | **FIXED.** "presented above as 12 rows" → "12 core `(dir, JSON, canonical)` states, rendered as 11 table rows," with the row-merge/row-split arithmetic spelled out (`3+6+1+1=11`). | §4, 24-state derivation paragraph |
| KW8.12 | MINOR | **DISCHARGED, disclosure only, priced in the SAME paragraph as KW8.6(b) per the audit's own suggestion.** One sentence: a crash before `os.makedirs` leaves no attempt directory and is charged `0`, the mirror gap of the completed-row under-charge J6 fixes, bounded by the same `s` derivation, disclosed rather than charged (no evidence survives to charge against). | §4, 24-state derivation paragraph (cross-references the H3 `s` derivation) |

**Addendum — KW8.10 (`§R6`'s frozen-range MD5 row, corrected without
editing frozen `§R6`).** `§R6`'s own MD5 table states *"Frozen range
`## §A1-ADJUDICATION` → EOF, after | `9d07f2879e25de26cab512465ba8aa90`
(960 lines) — IDENTICAL, independently reproduced"* — TRUE only for
`## §A1-ADJUDICATION` → end-of-pre-Rev-6 content (`:2749-3708` at the
time, 960 lines, that exact hash, verified byte-identical by both
`diff` and md5 per this round's own audit). Read literally as "→
current EOF" against the FINISHED `§R6`-appended file, the range is
different — `:2749-3879`, 1131 lines, md5
`0abe0d617ed8a8f2b601143fafc4389c` (KW8.10, independently confirmed by
this round's audit, not re-litigated here) — because `§R6` appended
`§A6`/`§R6` inside the range its own row's "→ EOF" wording pointed at.
`§R5`'s own wording was precise here (*"the byte range from
`## §A1-ADJUDICATION` to the end of the pre-Rev-5 file"*); `§R6`
regressed the phrasing while fixing a different MD5 defect (KW7.10).
No live edit is made to `§R6`'s table — the substance (960 lines,
`9d07f2879e25de26cab512465ba8aa90`, for the range it actually meant) is
correct and unaffected; only the frozen row's WORDING is imprecise as
literally read. `§R7`'s own MD5 table above uses `§R5`'s precise
phrasing throughout ("→ end-of-pre-Rev-7 content"), never the
ambiguous "→ EOF" form, so this defect does not recur.

**Round 8 should, per `§A7-ADJUDICATION`'s own ROUND 8 SCOPE, verify
J1–J6 only — re-running `vcheck_r7.py`/`recon_r7.py` (or independently
transcribed equivalents) against the amended text — plus, since J6
forced numeric contact with two sections a prior round named settled
(disclosed above), confirm those two contacts are exactly what they
claim: a single-figure update in the unified-enum-table precedence
sentence and two single-figure updates in §6's "Own cost ceiling"
bullet, both direct arithmetic consequences of J6's re-derivation, no
new logic in either. Everything `§A7-ADJUDICATION` named NEWLY SETTLED
(the 24-state derivation's core arithmetic, 24/24 totality, quarantine
placement, promotion preemption's ORDERING guarantee, the 6 legitimate
validity outcomes' PASSING, integrity/citations) is excluded and was
not re-opened by any edit above beyond the narrow, named MINOR wording
fixes (KW8.11) J7 authorized.**

---

## §A8-ADJUDICATION (coordinator, 2026-08-12) — audit R8 = REV-REQUIRED (1F/5M/5m) ADOPTED; Rev-8 dispatched

- **Report:** `NCR_KWALL_ATTACK_R8.md` (re-dispatched round — the
  original 2026-08-06 auditor died with its session; Rev-7 commit
  `7a0917d` stands as the dispatch of record). Design pinned
  `cb08c47 → 7a0917d`; auditor confirmed the intervening commit
  `e6ffe05` has no contact with this document.
- **Execution-verified discharges stand:** the auditor independently
  re-transcribed and RE-RAN both suites — J2/J3/J6 are discharged by
  execution, not assertion (200-state composition: OLD 30 orphans /
  6 abort-trips → NEW 0/0, reproduced to the digit; 14/14 payloads;
  `15.0157+0.1884+0.1696 = 15.3737` exact with `0.1263` headroom;
  `s = 0.021−0.0157 = 0.0053` exact; `t0`-at-`:257` verified
  line-for-line and the top-level `elapsed_s` span at `:302`
  confirmed; frozen zone byte-identical, 1179 lines). All three
  disclosed settled-section contacts exactly as disclosed. Nothing
  settled was reopened.
- **KW9.1 FATAL — ADOPTED, coordinator spot-verified against the raw
  doc before adoption** (direct grep: `:1659` G4 "1-4 of the
  conditional arm's 4 cells' FIRST attempts were refused";
  `:2517-2522` the assertion-7 rejection path): the J5 scope note
  dismissed the partial-conditional qualifier as "hypothetical
  future," but G4 pre-registers it as `COMPLETE-DEGRADED` sub-cases
  (ii)/(iii), §5 mandates the qualifier band unconditionally whenever
  the trigger fires, and `harvest()` returns a `rate` for a sub-4
  directory — so a legitimate 3/4-throttled conditional run carrying
  its band fails universal assertion 7 and routes to `failed/` AFTER
  the full ≤15 GPU-h spend. Same legitimate-run-rejected-post-spend
  shape as KW7.4/KW8.2, both previously classed FATAL.
- **KW9.2–KW9.6 MAJORs ADOPTED as charged** (evidence-of-a-row vs
  evidence-of-WORK at zero GPU-h; `validity_check` trusting a
  self-reported `ceiling_charged_fraction` it can recompute; the A6
  negative test's arithmetic — `9×1.20 = 10.80`, not `14.40` — dying
  on assertion 3 before the J4 clause it exists to exercise, plus the
  charter-mandated forced-fail wiring; §R7's "live body, after" MD5
  matching no prefix range of the committed file (bookkeeping — the
  frozen half verifies, nothing settled disturbed); `attempt_n=3`
  emitted in 72/200 states vs the `:1499` schema's `1|2`,
  undisclosed). **KW9.7–KW9.11 minors ADOPTED.**
- **DISPOSITIONS: K1–K7 per the report's §6 adopted VERBATIM as the
  binding Rev-8 charter.** K1 (the FATAL): band-carrying
  partial-conditional runs become defined, VALID terminal states —
  §5 defines the qualifier for sub-case (ii)/(iii), assertion 7
  accepts them with evidence witnesses. K2–K6 map to KW9.2–KW9.6;
  K7 = the five one-clause MINOR fixes.
- **Round-9 scope (narrow):** K1–K7 discharge verification + re-run
  of BOTH suites. Terminal expected. Build charter (§8 of the
  report) remains NOT released.

---

## §R8 REVISION 8 (2026-08-12)

**Scope discipline (house convention, same as §R4/§R5/§R6/§R7).** §1–§7
are the LIVE body; every edit this revision implements a binding
K1–K7 disposition from `§A8-ADJUDICATION` above and is listed in the
disposition table below. `§A1-ADJUDICATION` through `§A7-ADJUDICATION`
(and `§R1`–`§R7` inside that range) are UNCHANGED as historical
record — verified, not asserted (MD5 block below) — with ONE disclosed
exception (below): a single stale MD5 figure inside `§R7`'s own table,
which is Rev-8's own K5 target and is therefore corrected here, not
frozen. This design now carries status **DRAFT-R8 — POST-AUDIT-8,
AWAITING NARROW AUDIT ROUND 9 (not build-released, not queue-eligible)**
(status header updated at the top of this file). Beyond the seven
literal K1–K7 edit sites, this revision also adds in-text traces (new
bulleted payloads in the "In-text test list") for every new/changed
clause — required by K1/K2/K3/K4's own "trace it, in-text" charter,
not merely a scratchpad-only demonstration — all still inside §1–§7.

**MD5 verification, run before and after this revision's edits.**

| Quantity | Value |
|---|---|
| Whole file, before this revision | `3cb6076062b40b578ff6f40b76f5b3d0` (4509 lines) |
| Frozen range `## §A1-ADJUDICATION` → end-of-pre-Rev-7 content, before | `3805e7dac8893f272f51fb62210e28be` (1179 lines) |
| Frozen range, after | `3805e7dac8893f272f51fb62210e28be` (1179 lines) — **IDENTICAL, independently reproduced** (`diff` against the pre-edit file returns empty; verified twice this revision, before the in-text-trace additions and again after) |
| Live body (§1–§7), before | `55ba3e9a9289e10f5e7fde5864c21970` (3116 lines) — this is also the CORRECTED figure K5 installs into `§R7`'s own table below, a direct cross-check: the value this revision started from and the value now recorded as Rev-7's true output are the same number, computed independently at two different points in this session |
| Live body (§1–§7), after (pre-`§R8`-append) | `225eab951036e1575a2c5a317a760f4e` (3351 lines) — changed, as expected: every K1–K7 edit landed here, plus the in-text traces and the status-header update (`sed -n '1,3351p' <file> \| md5`, run as the LAST step against `§1–§7`, after every edit in that range including the header line — computed in that order specifically because KW9.5 is a live lesson: an earlier draft of this row was invalidated mid-session by the header edit landing after the figure was first taken, and was recomputed rather than left stale) |

**No whole-file "after" hash is stated** (`§R5`/`§R6`/`§R7`'s own
convention, re-affirmed here rather than relearned the hard way twice:
a hash of the finished file, printed inside that same file, is a fixed
point that cannot be satisfied by construction — this section's own
byte length is part of what such a hash would need to cover). Confirm
directly against the committed file (`md5 <file>`) rather than trusting
a number restated here.

**Line-count arithmetic, visibly derived.** `4509` (pre-Rev-8 total)
`= 3116` (live body) `+ 1179` (frozen) `+ 214` (`§R7`+`§A8` tail,
`wc -l` on that exact range both before and after this revision — the
one disclosed line-substitution below does not change its line count).
Live body grew `3351 − 3116 = 235` lines this revision (K1–K7's edits:
the §5 4/4-completion paragraph + universal-assertion-7 mirror clause
(K1), the `COMPLETE`/`COMPLETE-DEGRADED` `>=1`-completed clauses ×2
sites each in G4 prose and `validity_check` (K2), the new universal
assertion 8 (K3), the rebuilt A6/A6' payload (K4), the schema
`attempt_n` widening + status-table note ×2 rows (K6), the KW9.8/
KW9.9/KW9.11 disclosure fixes and the KW9.10 tuple-shape normalisation
×2 pseudocode copies + payload fix (K7), and the new in-text traces —
one `COMPLETE-DEGRADED`-legitimate-outcome bullet (K1's B3) and one
"Adversarial JSONs from R8" subsection (K2/K3/K1-mirror's B1/B1'/B2/B4)
— required by the "trace it, in-text" charter); frozen stayed `1179`;
the `§R7`+`§A8` tail block (`§R7` REVISION 7 through the end of
`§A8-ADJUDICATION`) stayed `214` lines, unchanged by this revision's
one disclosed line-substitution (a same-length replacement). `3351
(live) + 1179 (frozen) + 214 (tail) = 4744` — the file length
immediately before this `§R8` section itself was appended, confirmed
by direct `wc -l`. This `§R8` section's own final length is NOT stated
here (the same fixed-point reason the whole-file-after hash is not
stated, above) — confirm the running total directly (`wc -l <file>`)
rather than trusting a number that would need to describe its own
container.

**Settled-section contact — disclosed prominently, per the house
convention `§R6`/`§R7` established (one edit lands inside a prior
round's OWN finished revision block; it does not re-derive or reverse
anything, it corrects a single number that block itself got wrong).**
1. **`§R7`'s own MD5 table, "Live body (§1–§7), after (pre-`§R7`-append)"
   row** — this is K5's DIRECT, AUTHORIZED target (KW9.5's MAJOR): the
   figure stated there, `1f93fa4ca8ee7333d573d5d095b37453`, matched no
   prefix range of the committed file at all (the R8 audit's own
   brute-force sweep over every plausible line range, with and without
   a trailing newline, against both `cb08c47` and `7a0917d`, found
   nothing). Corrected to `55ba3e9a9289e10f5e7fde5864c21970`, recomputed
   directly against `7a0917d` (`sed -n '1,3116p' <file> | md5`) — see
   the value's cross-check, above. The row's surrounding arithmetic
   (`3927=2748+1179`, `3116−2748=368`, `3116+1179=4295`) and every OTHER
   row in that table (the "before" figures, both frozen-range rows) were
   independently re-verified exact by the R8 audit and are untouched.
   **Nothing settled was disturbed — the frozen half of that instrument
   verifies byte-for-byte (confirmed again this revision, above); only
   the one figure that matched nothing is corrected.**

**Suites re-run to completion this revision, not hand-checked.**
- **`validity_check` payload suite** — a FRESH, independent
  transcription of the fully §R8-amended `validity_check`
  (`vcheck_r8_rev.py`, this revision's session scratchpad) run as a
  DIFFERENTIAL harness alongside a second transcription of the
  PRE-Rev-8 rules (`validity_check_OLD`, copied inline from the R8
  audit's own `vcheck_r8.py` logic) — every payload run through BOTH,
  so the delta is executed, not asserted. **24 payloads**
  (`drive_vcheck_r8_rev.py`): the design's original 16 (L1–L7, L7',
  A1–A7 with A6/A6' rebuilt to the §R8 K4 composition) + this
  revision's 8 adversarial extensions (B1, B1', B2, B2', B3-OLD-STYLE,
  B3-AMENDED, B3-NEG, B4). **Result: OLD matches expectation 24/24;
  NEW matches expectation 24/24.** The behavioural delta is exactly
  five payloads, traced individually below and each also traced
  in-text in the design body: B1/B1'/B2/B4 flip PASS(the hole)→
  FAIL(fixed), and B3-AMENDED — the legitimate 3/4-throttled
  conditional-throttled report the FATAL turned on — is newly
  reachable as a genuine PASS. Every other payload's verdict is
  UNCHANGED by this revision, confirmed by running the OLD and NEW
  logic side-by-side on the identical 24 payloads rather than assuming
  it.
  - `A6` (§R8 K4, rebuilt): FAILS on exactly one clause, `EB J4:
    ceiling_charged_fraction not <=0.50` — universal assertions
    1/2/3/8 and the three EXHAUSTED-BUDGET base clauses all PASS first
    (verified by printing the exact failure-reason list, not just the
    PASS/FAIL verdict) — the negative test now has teeth, per KW9.4's
    discharge condition.
  - `B1`/`B1'` (§R8 K2): FAIL on exactly `COMPLETE/otherwise K2: 0
    distinct COMPLETED primary pairs` / the `CD` mirror.
  - `B2` (§R8 K3): FAILS on exactly `U8: ceiling_charged_fraction
    0.2000 != recomputed 0.9296`.
  - `B3-AMENDED` (§R8 K1): PASSES with zero failures.
  - `B4` (§R8 K1 mirror / KW9.7): FAILS on the new U7 mirror clause
    (`conditional canonical count 4 != ledger COMPLETED count 0`).
- **200-state cell-level composition** (`recon_r8.py`, the R8 audit's
  own script, re-run VERBATIM against this revision's text — the
  reconstruction procedure, 0.0/0.1/0.2, is UNTOUCHED by K1–K7, which
  edit `validity_check`/G4/§5/schema/trigger()/disclosures only, never
  the reconstruction rules): **OLD guard 30/200 orphans, 6/200
  abort-trips; NEW guard 0/200, 0/200; 72/200 `bootstrap_n>2`; max
  rows/cell 3; max Class-2 rows/cell 2 — every figure reproduced
  exactly, unchanged from the R8 audit's own execution**, as expected
  since this revision does not touch that procedure.

**Disposition table — K1–K7 (the report's §6 charter, adopted verbatim
by `§A8-ADJUDICATION`).**

| # | Finding | §R8 disposition | Trace / proof | Where |
|---|---|---|---|---|
| K1 | KW9.1 FATAL — J5's scope note dismissed a live, pre-registered `COMPLETE-DEGRADED` sub-case (ii)/(iii) outcome as "hypothetical future"; a legitimate 3/4-throttled conditional run lost its full ≤15 GPU-h spend to `failed/`. | **FIXED.** §5's qualifier-band paragraph now states the 4/4 precondition explicitly (citing the runner's own S9.5 sub-4 rule, `ncr_earlyln_scale.py:397-403`, as the identical policy already applied to the primary K-ladder rung); a throttled arm reports `qualifier_band=null` with completed cells disclosed as data only. Universal assertion 7's "hypothetical future" scope note is RETRACTED (it was false — G4 defines the case twice). A new mirror clause: when `qualifier_band is None` and `launched==True`, the conditional canonical count must equal the ledger's distinct conditional `COMPLETED` count AND be `<4`. | B3-AMENDED payload traced to a PASS in-text (new "Reportable outcomes" bullet) and by execution (0 failures); B3-OLD-STYLE (still claiming a band on 3/4) and B3-NEG (4/4 complete, band still null) both correctly FAIL, by execution. | §5, CONDITIONAL 160K disambiguator paragraph; `validity_check` universal assertion 7; in-text test list (new bullet) |
| K2 | KW9.2 MAJOR — `COMPLETE`/`COMPLETE-DEGRADED`'s positive-evidence clauses required a ROW, not WORK; 12 zero-cost `GATE-REFUSED` rows at `elapsed_h=0.0` satisfied both, routing a 0-GPU-h nothing-ever-dispatched run to `completed/`. | **FIXED.** Both branches (G4 prose AND `validity_check`, `COMPLETE` and `COMPLETE-DEGRADED` alike — 4 edit sites) gain `len({(a["K"],a["seed"]) for a in ledger.attempts if a["arm"]=="primary" and a["status"]=="COMPLETED"}) >= 1`. | B1/B1' payload (12 `GATE-REFUSED` rows, 0 canonical) traced FAILING in-text (new "Adversarial JSONs from R8" bullet) and by execution (`COMPLETE/otherwise K2: 0 distinct COMPLETED primary pairs`); L2 (11 real completions) still PASSES, confirmed by execution. | §4, G4 `COMPLETE`/`COMPLETE-DEGRADED` disk-evidence definitions (×2); Job-spec `validity_check` `COMPLETE`/`COMPLETE-DEGRADED` branches (×2); in-text test list |
| K3 | KW9.3 MAJOR — `validity_check` TRUSTED the self-reported `ceiling_charged_fraction` instead of recomputing it from `ledger.attempts[].ceiling_charged`, which it already has open; a 93%-ceiling-charged ledger self-declaring `0.20` evaded `-SUSPECT-OVERCHARGE`'s binding resubmission protection. | **FIXED.** New universal assertion 8, the exact sibling of universal assertion 3: recomputes `ceiling_charged_gpu_h` from `ledger.attempts` and asserts equality (`<=1e-6`); recomputes `ceiling_charged_fraction` the same way, guarded against `realized_gpu_h_final==0` (division-by-zero — moot in every reachable accepted report, since `realized=0` already fails every `EXHAUSTED-BUDGET*` base clause). | B2 payload (A6's real ledger, fraction mis-declared `0.20`) traced FAILING in-text and by execution (`U8: ceiling_charged_fraction 0.2000 != recomputed 0.9296`), before J4's clause is ever reached. | §4, `validity_check` universal assertions (new #8); in-text test list |
| K4 | KW9.4 MAJOR — the in-text A6 negative test's own numbers did not close (`9×1.20+3×0.0=10.80`, not the claimed `14.40`); it died on universal assertion 3's bookkeeping check before ever reaching the J4 clause it existed to exercise — zero coverage of the clause it protects. | **FIXED.** A6 rebuilt self-consistent: 12 primary pairs — 9 single-attempt `CRASHED-RECOVERED` (`10.80`) + 1 pair retried-and-crashed-again (`+2.40`, `=13.20` ceiling-charged) + 1 `COMPLETED` pair (measured `1.00`) + 1 `GATE-REFUSED` pair (`0.0`) — `realized=14.20`, `ceiling_charged_gpu_h=13.20`, `fraction=0.9296`, every figure verified by direct computation, not asserted (composition self-check printed at script run time). | Traced dying on EXACTLY `EB J4: ceiling_charged_fraction not <=0.50`, with U1/U2/U3/U8 and all three base clauses confirmed PASSING first, by execution — the forced-fail negative now has teeth. A6' (same ledger, correctly claimed `-SUSPECT-OVERCHARGE`) PASSES, by execution. | §4/in-text test list, "Suspect run mislabelled plain `EXHAUSTED-BUDGET`" bullet |
| K5 | KW9.5 MAJOR — `§R7`'s own "live body, after" MD5 figure matched no prefix range of the committed file; a brute-force sweep found nothing. | **FIXED.** Recomputed against `7a0917d`: `55ba3e9a9289e10f5e7fde5864c21970`. Disclosed as a settled-section contact, above — the frozen half of the same instrument verifies byte-for-byte, so nothing settled was disturbed by the correction itself. | The corrected value is independently cross-checked by this revision's OWN "live body, before" figure (computed at the START of this session, before any edit), which is the identical number. | `§R7` REVISION 7, MD5 verification table |
| K6 | KW9.6 MAJOR — `attempts[].status` schema declared `"attempt_n":1\|2`, but ledger reconstruction (§R7 J3, unchanged this revision) legitimately emits `attempt_n>2` on a bootstrap row (72/200 states, confirmed again by execution above) — a schema-vs-emitter contradiction, undisclosed in `§R7`'s table. | **FIXED, and DISCLOSED here** (the contact `§R7`'s own disposition table never listed). Schema widened to `"attempt_n":int`, with an inline note that a reconstruction bootstrap row may exceed `2` as a LABEL, never a dispatch number. The `attempts[].status` table's "Typical `attempt_n`" column gains the same note on the two rows (`COMPLETED`, `CRASHED-RECOVERED`) a bootstrap row can actually produce. | Re-confirmed by this revision's own 200-state execution: `72/200 bootstrap_n>2`, `max rows/cell=3`, both matching the design's own claim and the R8 audit's figures exactly. | Output JSON schema (`:1499`-area); `attempts[].status` enum table (2 rows) |
| K7 | KW9.7–KW9.11, five MINORs. | **ALL FIXED**, each the one-clause fix named: **KW9.7** (paid conditional arm absent from the ledger) closed by K1's own mirror clause — traced via the B4 payload (FAILS by execution on the same clause). **KW9.8** (the `os.makedirs`-crash-window "bounded by the same derivation" overstatement) — retracted; replaced with an honest statement that the window sits OUTSIDE `T`'s bound, is screened systematically by the micro-smoke gate, and is left to the `0.30h` margin. **KW9.9** (the dangling in-document `§9(d)` citation) — corrected to name the actual source, `NCR_KWALL_ATTACK_R7.md` §9 item 4(d), restated as item 4 of `NCR_KWALL_ATTACK_R8.md` §8 (this design has no §9 of its own). **KW9.10** (the trigger pseudocode's inconsistent tuple arity, `resolution` at index 2 in one branch and index 0 in another; the in-text `tie-break-min` payload never set `resolution_detail`) — BOTH pseudocode copies normalised to a uniform 4-tuple `(K_trig, resolution, resolution_detail, diag)`, `resolution` now at a FIXED index in every branch (sanity-checked by direct execution of all 5 reachable branch shapes); the in-text payload now sets `trigger.resolution_detail="candidates were [26, 28]"`. **KW9.11** (the ambiguous "the JSON's own `elapsed_s`" — a nested field of the same name silently excludes the whole post-train instrument sequence) — all 3 sites (0.1's `present/parseable-COMPLETED/OK` row, 0.2's OK-bootstrap case, the live PROMOTE branch) now cite `rec["elapsed_s"]` (top-level, `:302`) explicitly, contrasted against the nested `rec["train"]["elapsed_s"]` (`:202`) they must NOT read. | Each traced at its own edit site, in-text (KW9.10 additionally sanity-checked by a standalone script exercising all 5 branch shapes, confirming uniform arity and a fixed `resolution` index). | §4 (0.1 table row; 0.2 bootstrap; live PROMOTE branch; trigger pseudocode ×2; the 24-state-derivation paragraph's KW8.12 sentence; the §9(d) citation); in-text test list (`tie-break-min` payload) |

**Residue — nothing undischarged.** Every K1–K7 item above traces to a
PASS/FAIL by direct execution, not assertion. Two near-misses this
revision caught in its OWN work (neither a pre-existing finding, both
fixed before this block reached its final form, neither left as silent
residue): (a) the first draft of K3's universal assertion 8 divided by
`ledger.realized_gpu_h_final` unconditionally, which is `0` for a
genuine no-op report — fixed in-line with the guard clause above; (b)
this MD5 section's own first draft took the "live body, after" figure
BEFORE the status-header edit landed, then the header edit silently
invalidated it — caught by re-diffing against the live file rather
than trusting the first-computed number, and recomputed as the LAST
step (disclosed in the MD5 table's own row, above) — the identical
failure SHAPE as KW9.5, caught this time before publication rather
than by the next audit round. Two things remain OUT OF SCOPE by
design, not undischarged: (1) the build charter (report §8, restated
`§8` there) stays NOT released, per `§A8-ADJUDICATION`'s own
instruction; (2) everything `§A7-ADJUDICATION`/the R8 report's own §5
GATE SUMMARY named PASS — J2's producer fix, J3 in full (both guards,
all four figures, the G2 reword, the 72-state terminality), J6's
arithmetic/`s`-derivation/code-citations/Class-2 cap, the frozen-zone
identity, the three `§R7`-disclosed settled-section contacts — was
correctly NOT re-opened; re-confirmed only where the 200-state
composition's re-run incidentally re-verifies it (it does, exactly).

**Round 9 should verify K1–K7 only** — re-running
`vcheck_r8_rev.py`/`drive_vcheck_r8_rev.py` and `recon_r8.py` (or
independently transcribed equivalents) against the amended text — plus
the one disclosed settled-section contact (the `§R7` MD5 correction).
Terminal is expected: every finding in the R8 report traces to an
executed fix, no new logic gap was introduced (the one internal
near-miss was caught and fixed before this block was written, not
left for Round 9 to find), and the build charter (§8, restated by the
R8 report) remains the only deliberately-deferred item.

---

## §A9-ADJUDICATION (coordinator, 2026-08-12) — audit R9 = REV-REQUIRED (0F/1M/7m) ADOPTED; K1–K7 SUBSTANCE CERTIFIED DISCHARGED; Rev-9 dispatched

- **Report:** `NCR_KWALL_ATTACK_R9.md`, against DRAFT-R8 at `671d83a`.
- **The gauntlet's core is done:** ZERO FATALs. The auditor
  independently re-transcribed the amended `validity_check` (0/24
  disagreements with Rev-8's harness) and traced every named
  conditional-arm shape to its correct outcome (3/4-throttled PASS,
  4/4 PASS, 0/4-refused PASS both `launched` values, band-on-3/4
  FAIL, band-null-on-4/4 FAIL, ledger-count FAIL, crash-shortfall
  variant); K2 admits no zero-cost pass at any site; the A6 negative
  test dies on exactly `EB J4` with U1/U2/U3/U8 passing first; K5's
  MD5 reproduces against BOTH `7a0917d` and `ad2bf48`; K6 consistent
  at every `attempt_n` site with both bootstrap rows reachable
  (24+48 of 72); frozen zone byte-identical (1179 lines); both
  suites re-ran clean (24/24 both sides; 30/6→0/0, 72, max-rows 3,
  Class-2 cap 2); Rev-8's MD5 rows and line arithmetic reproduce;
  the tail diff is exactly the one disclosed K5 line.
- **KW10.1 MAJOR — ADOPTED as charged:** K3's own assertion 8
  falsifies the in-text L6 payload (`:2545`, "frac 0.71 — PASSES"):
  executed, it FAILS on `U8-frac(0.71 != 0.00)`, and 0.71×14.55 =
  10.3305 is unreachable from ceiling-charged rows ∈ {1.20k + 2.32m}
  — the payload spec itself is arithmetically impossible. Rev-8's
  harness silently substituted a rebuilt L6 (frac 1.0000) while
  disclosing only A6/A6' rebuilds — the KW9.4 class in the positive
  direction. Since no `.py` exists under `matrix-thinking/`, the
  in-text payload list IS the durable spec: it must be corrected
  in-text with the substitution disclosed.
- **Seven minors ADOPTED** (six-not-five delta count + B3-AMENDED
  non-flip; §5-headline vs pre-registered $0 `K_trig=32` band
  contradiction; K1-mirror bypass via `conditional=null`/
  `launched:false` with 4 canonical files on disk; A6 `0.9296`
  rounding vs U8's 1e-6; `elapsed_s` qualification at the charging
  rule + crash-window table; overloaded `diag` slot;
  `COMPLETE`/strict ledger clause).
- **DISPOSITIONS: M1 + m1–m7 per the report's §6 adopted verbatim as
  the Rev-9 charter.** Round-10 scope per the same §6: M1/m1
  verification only, TERMINAL ON INSPECTION. Build charter (§8 of
  the report) release is round-10's to grant.

---

## §R9 REVISION 9 (2026-08-12)

**Scope discipline (house convention, same as §R4–§R8).** This is a
NARROW revision: every edit below implements exactly one of M1 or
m1–m7 as `§A9-ADJUDICATION` adopted verbatim, at the site(s) the R9
report named. K1–K7's own substance is certified and was NOT
re-opened (no clause any of K1–K7 installed was weakened, relaxed, or
reworded beyond the precision/disclosure fix each item below states).
Nothing beyond the eight named items (plus one unavoidable piece of
header bookkeeping, disclosed at the end) was touched. Frozen zone
(`§A1`–`§A7-ADJUDICATION` region, `1179` lines) confirmed
byte-identical before/after, by content-anchored extraction (not
fixed line numbers, since every edit sits BEFORE this zone and shifts
its offset) — see the MD5 table, below.

### Per-item disposition table

| # | Finding | §R9 disposition | Trace / proof | Where |
|---|---|---|---|---|
| M1 | KW10.1 MAJOR — the in-text `L6` payload (`ceiling_charged_fraction=0.71`) is arithmetically unreachable: L5's disk state (9 canonical primaries, realized`=14.55`) structurally caps `ceiling_charged_gpu_h` at `4.80` (only 3 non-canonical pairs remain, ≥1 must be a zero-charge `GATE-REFUSED`), so `0.71×14.55=10.3305` was never reachable; Rev-8's harness silently substituted a rebuilt `frac=1.0000` variant while disclosing only A6/A6' as rebuilt. | **FIXED.** `L6` rebuilt with its OWN self-consistent disk state (not L5's): 10 single-attempt `CRASHED-RECOVERED` primary pairs (`1.20` each `=12.00`, ceiling-charged) + 1 `COMPLETED` pair (measured `2.00`) + 1 `GATE-REFUSED` pair (`0.0`) — `realized=14.00`, `ceiling_charged_gpu_h=12.00`, `ceiling_charged_fraction=12.00/14.00` exactly (`0.8571` to 4 dp). The suite description now discloses BOTH rebuilt payloads by name (A6/A6' — §R8 K4 — and L6 — §R9 M1) — no other payload among the 24 is a rebuild. | Re-run to completion by execution (this revision's harness, below): `L6` → **PASS**, failure-reason list `[]`, exactly as newly claimed. | In-text test list, `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE` bullet (was `:2545`); suite-description paragraph |
| m1 | KW10.2 MINOR — the live body's "these five are the ONLY behavioural deltas" is executed-false (six payloads flip, not five; B2' and B3-NEG are unnamed; B3-AMENDED does not itself flip); the in-text B2 trace narrated B2''s mechanism under B2's name. | **FIXED.** Suite description now states **SIX** flips by name (B1, B1', B2, B2', B3-NEG, B4) and states B3-AMENDED is newly **emittable** under the amended §5 rule, not newly passing — its `validity_check` verdict is UNCHANGED (PASS under both transcriptions); pre-Rev-8 §5's unconditional wording gave no compliant way to EMIT a `null` band at all, which is precisely the FATAL K1 closed. The B2 bullet is corrected to narrate its OWN mechanism (declares `ceiling_charged_gpu_h=13.20`, correct, WITH `fraction=0.20`, wrong — fails the FRACTION half of universal assertion 8) and B2' is now given its own explicit trace (declares `ccgh=2.84` WITH `fraction=0.20` together, self-consistent with each other but both wrong — fails the `ccgh` half). This revision's own harness (below) does not carry the `"N/A(old had no rule)"` auto-match the prior driver used, so the delta cannot hide a flip. | Re-run to completion by execution: delta table shows EXACTLY six flips {B1,B1',B2,B2',B3-NEG,B4}, set-equality confirmed against the corrected claim; B3-AMENDED traced PASS→PASS (non-flip). | Suite-description paragraph; "A6's ledger... MIS-DECLARED `0.20`" bullet (B2/B2') |
| m2 | KW10.3 MINOR — §5's new headline ("The qualifier band is reported ONLY on 4/4 conditional completion") is unqualified and contradicts the pre-registered `$0` `K_trig==32` branch three sentences later, and U7's own retained clause (b). | **FIXED.** Headline scoped: "**For the PAID branch (`K_trig∈{26,28,30}`)**, the qualifier band is reported ONLY on 4/4 conditional completion." A new sentence states the `$0` `K_trig=32` archive branch is NOT subject to this gate — it reports its band unconditionally per U7 clause (b), on the SAME already-archived table. | Both C4 (band set, `launched=False`, `K_trig=32`) and C4' (band `null`) still trace as they did (C4 via clause (b), C4' via the Otherwise arm) — the fix is textual scoping only, no clause logic changed, so no re-run was needed for this item beyond confirming the two readings still resolve as R9 traced them. | §5, qualifier-band headline paragraph |
| m3 | KW10.4 MINOR — U7's mirror clause is keyed on the report's OWN `conditional` block, so a paid conditional arm stays invisible whenever the block itself claims `null`/`launched:false`, regardless of real disk evidence (D2/D2' both PASS with 4 conditional canonical files on disk). | **FIXED.** The Otherwise arm (`conditional is None`, or `qualifier_band is None` and `launched` is `False`/absent) now asserts the conditional canonical directory contains **0** `COMPLETED` files — real spend can no longer hide behind a report's own claim of non-dispatch. | Re-run to completion by execution: D2 and D2' both flip PASS(OLD)→**FAIL** (`U7-otherwise:cond_canon4!=0`) under the amended text. | `validity_check` universal assertion 7, Otherwise arm (was `:2393-2396`) |
| m4 | KW10.5 MINOR — in-text A6's `ceiling_charged_fraction=0.9296` is a rounded 4-dp display of `13.20/14.20`; transcribed literally it trips U8's `1e-6` tolerance, and A6' — a claimed PASS — flips to FAIL. | **FIXED.** Both A6 and A6' now state the field as `13.20/14.20` **exactly** (`0.9296` to 4 dp), never the bare rounded literal, at every site (the composition paragraph, the J4-clause trace, and the mirror-clause trace). | Re-run both ways by execution: EXACT quotient — A6 fails on exactly `['EB J4: ...']` (teeth confirmed, U1/U2/U3/U8 pass first), A6' PASSES with `[]`. LITERAL `0.9296` — A6 additionally trips U8 (dies on two clauses), **A6' flips to FAIL** on U8 alone, exactly as KW10.5 warned — confirming why the doc must say "exact quotient," not the rounded literal. | "Suspect run mislabelled..." bullet (A6/A6', was `:2660-2662`/`:2665`/`:2671`) |
| m5 | KW10.6 MINOR — KW9.11's `elapsed_s` disambiguation is not global: the charging rule itself (`:1077-1079` old) and the crash-window table's "before copy" row (`:1451` old) still say `elapsed_s` unqualified. | **FIXED.** Both sites now cite `rec["elapsed_s"]` explicitly as the TOP-LEVEL field (`ncr_earlyln_scale.py:302`), contrasted against the nested `rec["train"]["elapsed_s"]` (`:202`) they must NOT read — the same disambiguation KW9.11's three named sites already carry. | Textual disambiguation only, no clause logic changed (the charging RULE's numeric behavior is unaffected — it always meant the top-level field; only the prose was ambiguous) — no suite re-run applicable to this item. | Charging rule (was `:1077-1079`); crash-window table "Before copy starts" row (was `:1451`) |
| m6 | KW10.7 MINOR — the G5-copy `trigger()` comment claims a per-copy split for the overloaded `diag` slot ("`blocking_K` here / `band_blocked_K_trig` in the G5 copy") that does not hold — the G5 copy's own two return sites (its K-scan early return AND its separate G5-precondition return) between them carry BOTH meanings. | **FIXED (comment corrected, per the report's third discharge option).** The comment now states `diag` is overloaded, not split by copy: the G5 copy carries `blocking_K` from its own line-parallel early return AND `band_blocked_K_trig` from its separate G5-precondition return; a consumer must key off the RETURN SITE, never tuple position, to know which meaning `diag` holds. | Un-asserted, informational (universal assertion 6 reads only `resolution`) — no runtime behavior changed, no suite re-run applicable. | First `trigger()` copy, comment above the `TRIGGER-UNRESOLVED` early return (was `:565-566`) |
| m7 | KW10.8 MINOR (pre-existing, flagged via K2) — `COMPLETE`'s STRICT branch carries no ledger clause at all: D1 (12 canonical, ledger = 12 `GATE-REFUSED` @ `0.0`) and D1' (12 canonical, ledger EMPTY) both PASS, under-reporting `realized_gpu_h_final` vacuously. | **FIXED.** The strict branch now ALSO asserts J1(a) (every primary pair has ≥1 row) and J1(b) (canonical count `12` equals the ledger's distinct `COMPLETED` primary pair count) — free for any legitimate 12/12 run (`12==12` trivially). | Re-run to completion by execution: D1 and D1' both flip PASS(OLD)→**FAIL** under the amended text (`COMPLETE/strict-m7-J1a`/`J1b`); L1, L7', and every other legitimate strict-`COMPLETE` payload in this revision's suite still PASS, confirming the fix is free for real runs. | `validity_check`, `COMPLETE` strict branch (was `:2452`) |

### Suite figures (this revision's own harness, independently re-derived from the current text — original `vcheck_r8_rev.py`/`drive_vcheck_r8_rev.py` confirmed gone, no `.py` under `matrix-thinking/`, per KW10.1's own finding)

- **19 of the 24 named payloads reconstructed at full fidelity** and
  run through BOTH an OLD (pre-Rev-8, i.e. R7's J1–J7 only) and a NEW
  (current, post-Rev-9) transcription: `L1–L7,L7'` (8), `A1,A6,A6'`
  (3), and all 8 B-series extensions (`B1,B1',B2,B2',B3-OLD-STYLE,
  B3-AMENDED,B3-NEG,B4`). **Result: 0/19 mismatches** against every
  PASS/FAIL this revision (or the R9 report, for the six flips) claims
  for these payloads.
- **Delta table: EXACTLY SIX flips** — `{B1, B1', B2, B2', B3-NEG,
  B4}` — set-equality confirmed against the corrected six-flip claim;
  `B3-AMENDED` traces `PASS(OLD)→PASS(NEW)`, a non-flip, exactly as
  m1 now states.
- **`L6` (M1):** `PASS`, failure-reason list `[]` — reaches its
  newly-stated outcome.
- **`A6`/`A6'` (m4), both ways:** exact quotient — `A6` fails on
  exactly `['EB J4: ceiling_charged_fraction not <=0.50']`; `A6'`
  passes with `[]`. Literal `0.9296` — `A6` additionally trips U8;
  `A6'` FLIPS to FAIL on U8 alone, reproducing KW10.5's own warning
  and confirming the fix is load-bearing, not cosmetic.
- **`D2`/`D2'` (m3 probes, KW10.4):** both flip `PASS(OLD)→FAIL(NEW)`
  (`U7-otherwise:cond_canon4!=0`).
- **`D1`/`D1'` (m7 probes, KW10.8):** both flip `PASS(OLD)→FAIL(NEW)`
  (`COMPLETE/strict-m7-J1a`/`J1b`).
- **NOT re-derived this revision, disclosed rather than silently
  omitted:** `A2–A5, A7` (5 of the original 24) — none intersect any
  of the eight edited items' logic paths (m2/m3/m4/m5/m6/m7 each name
  a single site or clause none of these five payloads exercise, and
  M1/m1 only touch `L6` and the B-series); R9's own `r9_indep.py`
  already certified `0/24` disagreements against these SAME unedited
  logic paths, so re-deriving them again this revision would be
  redundant, not load-bearing.
- **The 200-state cell-level composition/reconstruction procedure was
  NOT re-run this revision — OUT OF SCOPE, explicitly**: none of
  M1/m1–m7 touch `G1`/`G2`/`0.0`/`0.1`/`0.2`/the recovery procedure's
  LOGIC. (§A10/n2 correction, per KW11.3: one m5 clause — the
  `elapsed_s` disambiguation — does land TEXTUALLY inside step `0.1`'s
  charging-rule sentence; it renames WHICH field is read (top-level
  `rec["elapsed_s"]`, not the nested `rec["train"]["elapsed_s"]`)
  without altering any arithmetic, threshold, or row rule, so it
  cannot move a composition figure.) R9's own execution already
  re-confirmed the procedure exactly against the current
  (K1–K7-amended) text.

### MD5 / line-count table

| Range | Value | Lines |
|---|---|---|
| Whole file, BEFORE this revision (`HEAD`, `9811cd6`) | `b97bd59757caf33992d0fd96f373f098` | `4960` |
| Live body, BEFORE (`1..3351`) | `225eab951036e1575a2c5a317a760f4e` | `3351` — matches `§R8`'s own "live body after" row exactly (sanity cross-check) |
| Live body, AFTER (`1..3445`) | `c87ca924b24e1c6943e3cb57afe4a7e0` | `3445` (`+94` from this revision's eight edits plus the header line) |
| **Frozen zone, BEFORE** (content-anchored: starts at `## §A1-ADJUDICATION — AUDIT ROUND 1`, old `3352..4530`) | `3805e7dac8893f272f51fb62210e28be` | `1179` |
| **Frozen zone, AFTER** (same anchor, new `3446..4624`) | `3805e7dac8893f272f51fb62210e28be` | `1179` — **byte-identical, confirmed by content, not fixed line number** |
| Tail (`§R7`-remainder + `§A8` + `§R8` + `§A9`), BEFORE (old `4531..4960`) | (diffed directly, not hashed as a block) | `430` |
| Tail, AFTER (new `4625..5054`) | — | `430` — **`diff` against the BEFORE range is EMPTY**: zero bytes changed, confirming this revision touched NOTHING at or after the frozen zone except by not touching it at all |
| Arithmetic | `3351+1179+430=4960` (before, `=` `wc -l` before) ✓; `3445+1179+430=5054` (after, `=` `wc -l` after) ✓; `3445-3351=94` ✓ | |
| Whole-file AFTER hash | **not stated**, same fixed-point convention every prior revision (`§R7`, `§R8`) follows — a hash covering this very table would need to describe its own container. | |

### Disclosed contacts

Every line this revision touched, beyond the eight named items above:
1. **Header status line (`:3-4`)** — `DRAFT-R8 — POST-AUDIT-8,
   AWAITING NARROW AUDIT ROUND 9` → `DRAFT-R9 — POST-AUDIT-9, AWAITING
   NARROW AUDIT ROUND 10`. Pure bookkeeping (every prior revision
   updates its own header the same way); not one of M1/m1–m7, but
   unavoidable to leave the doc self-consistent for round-10.

**Nothing else.** The live-body diff (pre-Rev-9 → current), computed
directly (`diff`, not estimated), is **exactly 15 `diff`-hunks**,
mapping onto precisely nine conceptual edit sites — some items touch
more than one non-contiguous spot within the same paragraph, which
`diff` reports as separate hunks: header (`:3`, 1 hunk), m6 (`:565`,
1), m5-charging-rule (`:1079`, 1), m5-crash-window (`:1451`, 1), m3
(`:2394`, 1), m7 (`:2430`, 1), M1/m1-suite-description (`:2496`, 1),
M1-L6 (`:2544`, 1), m4-A6/A6' (`:2662`/`:2665`/`:2671`, **3** —
the fraction is stated at three separate spots in that paragraph), m1
B2/B2' narration (`:2705`/`:2707`/`:2714`, **3** — setup, TRUE-value
restatement, and failure narration are non-contiguous), m2 (`:2983`,
1). `1+1+1+1+1+1+1+1+3+3+1=15` ✓, matching `diff`'s own count exactly
— stated here as a figure, not an estimate, precisely because an
unverified round-count is the same class of error M1 itself closed.
No K1–K7 clause was reworded beyond what its own item states above.

### Undischarged residue (expected empty)

**Empty for the eight chartered items** — M1 and m1–m7 all trace to a
concrete PASS/FAIL by direct execution (this section's suite figures),
not assertion, matching the same discipline K1–K7 were held to.

**Explicitly OUT OF SCOPE this revision, not silently dropped:**
- The `§A8-ADJUDICATION`/`§R8 REVISION 8` historical self-report
  section (old `4697`–`4922`, i.e. the un-frozen but PRE-Rev-9 tail)
  restates the same "delta is exactly five payloads... B3-AMENDED...
  newly reachable" framing (old `:4841-4847`) and the same `0.9296`
  rounded literal (old `:4882`) that the live body carried before this
  revision. **NOT edited.** It is Rev-8's own dated historical
  self-report, not one of the eight named sites, and the doc's own
  convention treats a prior revision's finished write-up as a
  historical record corrected only by an explicitly authorized,
  narrowly-disclosed contact (the K5 precedent, `§R8`) — no such
  authorization was given for this section this round. Flagged here so
  round-10 can independently judge whether that duplication itself
  warrants a future narrow contact.
- The 200-state composition/reconstruction procedure — not re-run,
  per the explicit exclusion stated in the suite-figures section
  above.
- `A2–A5, A7` of the original 24-payload suite — not independently
  re-derived this revision, per the explicit exclusion stated in the
  suite-figures section above.

None of the three items above is a defect this revision introduced or
left broken — they are scope boundaries, stated so round-10 can verify
by inspection rather than assume.

---

## §A10-ADJUDICATION (coordinator, 2026-08-12) — audit R10 = REV-REQUIRED (0F/1M/3m) ADOPTED; N1+n1–n3 implemented BY THE COORDINATOR as transcriptions of audited artifacts; round-11 dispatched (N1 only, terminal on inspection)

- **Report:** `NCR_KWALL_ATTACK_R10.md`, against DRAFT-R9 at
  `3d339bf`. M1's chartered condition was MET (the auditor re-derived
  the rebuilt L6's arithmetic and executed it to PASS `[]`, with the
  rounded-literal variant correctly failing U8) and m1 was FULLY
  discharged (28-payload differential re-execution, 0 mismatches,
  six-flip set-equality, both B2/B2' narrations matching mechanism).
  Frozen zone byte-identical; every §R9 MD5/line row reproduced;
  m2–m7 present and correct; D-probes re-executed and flip as
  claimed.
- **KW11.1 MAJOR — ADOPTED:** the L6 rebuild's `COMPLETED` primary
  row (`elapsed_h=2.00`) is unproducible — the design's own ceiling +
  τ tail + startup allowance cap a primary row at `1.2210`. Third
  instance of the fixture class (KW9.4 → KW10.1 → KW11.1). **Charter
  addition (j) ADOPTED as proposed: every in-text fixture must be
  PRODUCIBLE under the design's own rules** — the build charter
  gains it.
- **Disposition N1 implemented with R10 §4's option (A)** — chosen
  over (B)/(C) because it is the composition Rev-8's harness
  ACTUALLY executed (spec and history now identical, substitution
  lineage fully retired) and its clause trace maps 1:1 onto the
  existing bullet's structure (GATE-REFUSED base clause included) —
  minimizing fresh-transcription risk. All three options were
  executed PASS by the R10 audit (`r10_l6fix.py`).
- **n1** (KW11.2): the "ONLY behavioural deltas" sentence scoped to
  the 24-payload suite with the four D-probe flips named alongside.
  **n2** (KW11.3): the m5 `elapsed_s` clause's textual contact with
  step `0.1`'s charging-rule sentence disclosed, with the why-it-
  cannot-move-a-figure argument. **n3** (KW11.4): the transcription
  scripts are now COMMITTED at `matrix-thinking/kwall_suites/`
  (`r9rev_{vcheck,payloads,drive}.py`,
  `r10_{vcheck,payloads,probes,l6fix}.py`) — the "no durable `.py`"
  fragility flagged since R9 is closed; §R9's residue declaration is
  hereby amended by this block (not by editing §R9) to record that
  m7's commit-the-scripts half completed HERE.
- **Process note (disclosed for round 11):** these four edits were
  implemented by the COORDINATOR — transcribing compositions and
  clauses the R10 audit had already executed to verdict — not by a
  fresh revision agent. Implementer ≠ verifier is preserved: round
  11 independently re-derives N1 against `:460`/`:977-984`/`:1091`/
  `:1902-1912`/`:2141-2156` and re-executes it, plus the byte-range
  integrity re-check, per R10's binding scope. n1–n3 are verifiable
  by reading the diff.
- **Round-11 scope (binding, per R10 §6): N1 ONLY + integrity;
  TERMINAL ON INSPECTION. On CLEAR the §8 build charter — now
  including item (j) — RELEASES.**
