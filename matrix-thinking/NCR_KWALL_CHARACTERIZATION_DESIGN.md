# NCR K-WALL CHARACTERIZATION — K∈{26,28,30} ON THE LIVE K=24 RUNG

**STATUS: DRAFT-R2 — POST-AUDIT-2, AWAITING FOCUSED AUDIT ROUND 3 (not
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

**Command (primary, unchanged CLI surface from R0/Rev 1):**

```
ncr_earlyln_scale.py --cell --K {26,28,30} --d-override {27,29,31} \
  --seed {0,1,2,3} --steps 80000 --ceiling-gpuh 1.20 \
  --outdir results_kwall_characterization \
  --stop-file results_kwall_characterization/STOP
```

**Rev 2 (E2, KW3.2): `--ceiling-gpuh` restored to the job-108 house
convention** (`≥2×nominal, floor 1.0h` — verified verbatim against
`queue/jobs/pending/108_laneA_main_K48_s0.json`'s own notes field:
*"`--ceiling-gpuh` is 2x the estimate (floor 1.0h) as the real safety
bound"*), not Rev 1's 0.75h trim. Per-K minimum under this convention
is `max(2×nominal, 1.0)` = 1.0211/1.1073/1.1946 h for K=26/28/30
(recomputed this revision, §4 pricing below); **1.20h is used as one
shared value across the 3-K command** because it is `≥2×nominal` for
EVERY K in the batch (it exceeds even K=30's 1.1946h minimum), so one
CLI invocation stays valid for all three K's without under-covering
any of them. Rev 1's 0.75h trim is deleted — it violated the job-108
floor discipline and, per KW3.2, inverted the exact contention
mitigation KW2.2 asked for. Program-level cost bounding is now done by
the **cumulative cap (E1, below)**, not by shrinking individual
ceilings — no per-cell trim is needed.

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
  --seed {0,1,2,3} --steps 160000 --ceiling-gpuh 2.32 \
  --outdir results_kwall_characterization_160k \
  --stop-file results_kwall_characterization_160k/STOP
```

**Rev 2 (E2):** `--ceiling-gpuh` restored to `2.32h` — `≥2×nominal`
for every possible `K_trig∈{26,28,30}` (worst case K=30's minimum is
`2.3121h`, §4 pricing below; 2.32h clears it and every smaller-K
minimum). Rev 1's 1.50h trim (≈1.36× nominal, below job-108's 2×
convention) is deleted for the same reason as the primary ceiling.

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
now feeds a per-cell ceiling floor (E2) and the program cumulative cap
(E1), both of which are safety bounds that should never be built on an
understated nominal:

| K | 160K nominal (h/cell) | ×4 seeds |
|---|---|---|
| 26 | 0.9882 | 3.953 |
| 28 | 1.0716 | 4.286 |
| 30 | 1.1561 | 4.624 |

(vs. Rev 1's 0.9434/1.0230/1.1037 — every value moves up ≈4.5–4.8%,
`0.5106×1.9355=0.9882`, `0.5536×1.9355=1.0716`,
`0.5973×1.9355=1.1561`, all directly executed this revision, not hand
math.)

**E1 — the cumulative-realized-GPU-h program cap (KW3.1's fix,
replacing Rev 1's "sum of trimmed per-cell ceilings" bound, which was
false — see KW3.1: the trim did not survive its own retry rule and the
true worst case under it was 30.00h, double the mandate's cap).**

**The enforceable rule, stated exactly (who checks, when, on what
data):** before dispatching ANY cell-launch command — a first attempt
on a not-yet-attempted cell, OR a retry of an `ABORTED-BUDGET` cell
(E4, below) — the launcher (the same resume-safe supervisor-loop
process that already gates on `status=="COMPLETED"` for skip-vs-resume,
per the repo's on-box queue directive) computes
`realized_gpu_h = Σ gpu_h` read fresh from EVERY cell JSON currently on
disk under `results_kwall_characterization*/` whose `status` is
`COMPLETED` (the `gpu_h` field is `elapsed_s/3600`, already
eval-inclusive, `ncr_earlyln_scale.py:303-304`), and applies, in order:
1. **HARD PROGRAM GATE.** If `realized_gpu_h + ceiling(cell) > 15.00`,
   the launch is refused outright — no cell, first attempt or retry,
   is ever dispatched once its OWN training-phase ceiling (E2's
   restored `max(2×nominal,1.0h)` value) would push the projected total
   past the 15.00 hard cap. For a batch of cells dispatched together
   (e.g. one per GPU, per the repo's packing doctrine), the check is
   applied to the WHOLE batch atomically —
   `realized_gpu_h + Σ ceiling(cell_i in batch) ≤ 15.00` — never
   per-cell in isolation, so simultaneous launches cannot jointly
   overshoot a bound only checked one cell at a time.
2. **RETRY GATE (subordinate to 1).** If `realized_gpu_h ≥ 12.00`, no
   RETRY is dispatched — an `ABORTED-BUDGET` cell hitting this
   condition is flagged `PERSISTENTLY-ABORTED` (E4) immediately, with
   no second attempt, reserving the residual `15.00−12.00=3.00` GPU-h
   band exclusively for completing outstanding FIRST attempts, never
   for a second attempt at a cell that already failed once.

**Worst-case bound, derived (not asserted).** By induction on
admission order: the hard gate (1) never admits a cell or batch unless
`realized_gpu_h`-so-far + that admission's own ceiling(s) stays
`≤15.00`; each admitted cell's actual TRAINING time is bounded above by
its ceiling (the existing `ceiling_s` enforcement in
`train_earlyln_cell`); the only quantity the admission check does NOT
price is EVAL overhead (unenforced by `ceiling_s`, empirically bounded
at `≤0.0126 GPU-h/cell`, the corrected max — KW3.9, below). So for the
LAST batch admitted before the gate closes: `realized_before_last_batch
+ Σ ceiling(last batch) ≤ 15.00`, and once that batch finishes,
`realized_final = realized_before_last_batch + Σ(training_i + eval_i)
≤ realized_before_last_batch + Σ ceiling(last batch) + Σ eval_i ≤ 15.00
+ Σ eval_i`. Since there are at most 16 distinct cells in the WHOLE
program (12 primary + 4 conditional) and `eval_i ≤ 0.0126h` each, this
is a batching-strategy-independent bound:

```
realized_final ≤ 15.00 + 16 × 0.0126 = 15.2016 ≈ 15.20 GPU-h
```

— matching, not exceeding, the pre-existing eval-inclusive disclosure
(D5, below), and this time actually TRUE (KW3.1's 30.00h defect is
closed by construction: bounded by budget, not by a fixed retry count).
The retry gate (2) does not loosen this bound — it is a stricter,
disclosed sub-constraint that keeps retries from crowding out
first-attempt cells inside the same 15.00 envelope, nothing more.
**A disclosed consequence, not a hidden failure mode:** in the
pathological case where all 12 primary cells consume their full
ceiling before the conditional arm is considered, the hard gate can
correctly THROTTLE OR REFUSE part or all of the conditional arm rather
than exceed 15.00 — the mechanism degrades gracefully to
`INCOMPLETE-AT-K`/reduced conditional coverage, never to a silent
overrun. In the intended, non-adversarial case the numbers stay far
inside this envelope: nominal primary (≈6.65h) + nominal conditional
worst-case (K=30, ≈4.62h) = **≈11.27h**, informational only, not the
enforced bound.

**Ceiling reference table (E2, restored job-108 convention;
informational — the ENFORCED bound is E1's 15.00/15.20, not a sum of
these):**

| | per-cell ceiling (`≥2×nominal`, floor 1.0h) | ×N cells (informational sum) |
|---|---|---|
| Primary K=26 (80K) | 1.0211 h | 4.084 h |
| Primary K=28 (80K) | 1.1073 h | 4.429 h |
| Primary K=30 (80K) | 1.1946 h | 4.778 h |
| Conditional K=26 (160K) | 1.9764 h | 7.906 h |
| Conditional K=28 (160K) | 2.1432 h | 8.573 h |
| Conditional K=30 (160K, worst case) | 2.3121 h | 9.248 h |

(All 6 values re-derived this revision by direct execution of
`max(2×nominal, 1.0)` against the corrected nominals above; the
informational per-arm ceiling-sums — e.g. all 16 cells simultaneously
at ceiling would sum to `13.29h primary + 9.25h conditional(K30) =
22.54h` — are NOT the enforced bound and are shown only to make clear
why E1's cumulative, realized-GPU-h check, not a static ceiling-sum, is
the mechanism that actually holds the program to ≤15.20h.)

**Margin claim, corrected (KW3.2).** Rev 1's supporting sentence *"every
1×-budget cell ever run has stayed within 1.06× of its own K's mean"*
is **deleted — it was false** (the audit found 4 of 24 archive config
groups exceed 1.06×, max 1.092×). The TRUE archive-wide figure, verified
CLEAN by the audit over all 97 completed cells / 24 groups, is cited
instead: **the largest max/nominal ratio ever observed in this program
is 1.206×** (K32, 2×-budget, seed 3: `1.2685/1.0510`). Under the
restored E2 ceilings (`≥2×nominal`), this leaves `2.00/1.206 ≈ 1.66×`
of headroom beyond the worst spike ever recorded — a substantially
WIDER safety margin than Rev 1's 0.75h/1.50h trim (which was
`1.26×`–`1.47×` nominal, i.e. inside 1.5× of the worst archive spike,
the exact contention risk KW2.2 flagged). Restoring the convention
fixes the substance of KW3.2, not merely its citation.

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
This is exactly the per-cell figure E1's worst-case derivation (above)
uses; the ≈15.20 GPU-h program bound already reflects it.

**D5/E4 — ABORTED-BUDGET / MISSING / non-COMPLETED cell rule, Rev 2
mechanization (KW2.2/KW2.3/KW3.4).** Rev 1's version deadlocked
(`PERSISTENTLY-ABORTED` could never reach 4/4 COMPLETED, so its K could
never be classified) and left the retry/exclusion logic unmechanized,
with no named enforcement point against `harvest()`'s actual behavior.
Rev 2 replaces both bullets with a bounded, mechanized rule:

- **Retry, bounded.** A cell with `status=="ABORTED-BUDGET"`
  (`train_earlyln_cell`, `:198-201`) is retried AT MOST ONCE — subject
  to E1's retry gate (`realized_gpu_h<12.00`, above) — with no ceiling
  change. Retraining is from scratch (the harness only skips on
  `status=="COMPLETED"`, `:243-245`; there is no checkpoint resume, so
  a retry consumes close to a full ceiling again — this is exactly why
  the retry is bounded at 1 and gated on cumulative budget, not
  unconditional). If the retry ALSO fails to reach `COMPLETED` (a
  second `ABORTED-BUDGET`, or the retry itself is refused by E1's retry
  gate), that seed becomes **`PERSISTENTLY-ABORTED` — a TERMINAL state,
  never retried again, regardless of remaining budget.**
- **Denominator, fixed at 4 (A4.9 guard preserved — KW3.4's
  "denominator contradiction" closed).** `PERSISTENTLY-ABORTED` and
  MISSING (never-attempted) cells are NOT excluded from `n_seeds`; the
  rate denominator for every K stays exactly 4, matching the
  partition's own `r∈{0,1,2,3,4}` domain. What is unknown is the
  NUMERATOR contribution of the incomplete seed(s), not the
  denominator — resolved by interval logic, next.
- **Interval logic for a K with exactly one terminal-aborted or
  MISSING cell (E4, exact).** Let `r_known` = the CONVERGED count among
  that K's 3 resolved seeds. The incomplete seed's true outcome is
  unknown but bounded: it either would have been CONVERGED (contributing
  `+1`) or not (contributing `+0`). Evaluate the §5 six-rule
  classification procedure TWICE — once with that K's `r = r_known` and
  once with `r = r_known + 1` (the other two K's `r`-values held fixed
  at their own resolved values) — and compare the resulting bands
  (including the `[NON-MONOTONE]` tag, which is part of the band for
  this comparison):
  - **Same band both ways ⇒ DECIDE.** Report that band, with a
    disclosure flag naming which K's rate was interval-resolved and
    from which incomplete-cell state (`PERSISTENTLY-ABORTED` or
    MISSING).
  - **Different bands ⇒ `INCOMPLETE-AT-K` for that K.** Reported as its
    own outcome, explicitly EXCLUDED from frontier claims (never
    silently forced into either candidate band), and disclosed with
    both candidate bands shown side by side.
- **Two or more incomplete cells at one K, or incomplete cells at
  MULTIPLE K's simultaneously.** With ≥2 incomplete cells at a single
  K, the interval width exceeds what a two-way comparison can resolve
  (up to 3 candidate `r` values); Rev 2 does not attempt the wider
  interval — that K is `INCOMPLETE-AT-K` UNCONDITIONALLY, no candidate
  comparison performed. If DIFFERENT K's each have exactly one
  incomplete cell at the same time, interval logic is applied
  compositionally: evaluate the classification over the full
  cross-product of each affected K's two candidate `r`-values (`2^m`
  candidates for `m` singly-incomplete K's); if every candidate in the
  cross-product yields the SAME band, decide (disclosing all `m`
  interval-resolved K's); otherwise `INCOMPLETE-AT-K` for the affected
  K's, both/all candidate bands disclosed. Either rule is acceptable to
  state outright per E4; Rev 2 states both rather than leaving the
  multi-incomplete case silent.
- **Enforcement point, named (closes KW3.4's "no enforcement point"
  defect).** `harvest()`'s existing `n_seeds`/`gate_eligible` computation
  (`ncr_earlyln_scale.py:380-406`) is **not** the right instrument
  as-is: `n_seeds = len(seeds_by_K[K])` comes from `discover_seeds_by_K`,
  which globs `earlyln_K{K}_s{seed}.json` FILE PRESENCE — a file exists
  on disk for an `ABORTED-BUDGET`/`PERSISTENTLY-ABORTED` cell too, so
  `n_seeds` silently counts it and `gate_eligible=n_seeds>=4` reads
  `True` even though that cell never reached `COMPLETED` (verified by
  direct code read this revision — this is the "band contamination"
  KW2.2/KW3.4 named, concretely located). **Build-stage instruction
  (this design stays DRAFT and edits no code; specified here so the
  build stage implements exactly this, the same deferral pattern
  already used for the KW2.6 job-spec template):** `harvest()` must
  compute `n_completed = Σ 1[cells[seed]["status"]=="COMPLETED"]`
  (status-based, not file-glob-based) per K, and:
  - `n_completed==4` → classify normally (as today).
  - `n_completed==3`, the 4th `MISSING` or `PERSISTENTLY-ABORTED` →
    apply the interval-logic rule above; emit the decided band + the
    interval-resolved disclosure flag, OR `INCOMPLETE-AT-K`.
  - `n_completed≤2` → `INCOMPLETE-AT-K` unconditionally, no candidate
    comparison.
  This patch is the harvest-side counterpart to the launcher-side E1
  cumulative check — together they are the two points that actually
  enforce this design's stated rules against the harness's real
  behavior, closing the gap KW3.4 found between "the rule is written"
  and "the rule has an enforcement point."

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

**KW2.8/KW3.13 close-out (accepted-risk, Rev 2).** The build's own
`--smoke`/t5 self-test path exercises new K's at `GRID_SHAPES`'
default `d=2K`, not this design's own `d=K+1` config, and would run 3
extra smoke cells under an already-rejected convention (§2(a)/(b)).
R1's discharge asked for two things: a `d=K+1` micro-smoke instruction
(recorded above, in the `validity_check`/job-spec instructions) AND an
extension of `t4b`'s own K-list — the second half was silently dropped
in Rev 1. **Accepted as risk, one sentence:** the `t4b` K-list
extension is deferred to the build-stage smoke-test checklist (a
build-time task, not this draft's own claim about K∈{26,28,30}'s
trainability), and the +3 default-`d=2K` smoke cells are harmless
extra coverage of a convention already rejected on its own evidence
(§2(a)/(b)), not a gap in this design's `d=K+1` claim — the build stage
must still run the `d=K+1` micro-smoke this design specifies before
release, which is the smoke test that actually matters for this
design's own config family.

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

**INCOMPLETE-AT-K (D5/E4, Rev 2 mechanization, KW2.2/KW2.3/KW3.4).**
Rev 1 said an incomplete K is "re-run... until 4/4 COMPLETED, then
classified" — this deadlocks against a TERMINAL `PERSISTENTLY-ABORTED`
seed (§4's bounded-retry rule), which by construction can never reach
4/4 COMPLETED. Rev 2 replaces this with §4's mechanized rule (E4,
cross-referenced here rather than restated in full): a K with exactly
one `PERSISTENTLY-ABORTED`/MISSING cell is resolved by INTERVAL LOGIC
(evaluate the six-rule procedure at both possible values of the
unresolved seed; same band ⇒ decide with disclosure, different bands
⇒ `INCOMPLETE-AT-K`); a K with ≥2 incomplete cells, or multiple K's
whose cross-product of interval candidates disagrees, is
`INCOMPLETE-AT-K` — reported, disclosed, and explicitly EXCLUDED from
frontier claims, never silently forced into a band or silently dropped
from `n_seeds` (the denominator stays fixed at 4 throughout, per the
A4.9 guard). No partial rate is ever read into the table as if it were
a full-n=4 verdict.

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
- **Own cost ceiling (Rev 2, E1/E2 — restored per-cell ceilings, global
  program cap does the bounding).** The primary 12 cells carry
  `--ceiling-gpuh 1.20` (§4 — restored to the job-108 `≥2×nominal`
  convention, NOT Rev 1's 0.75h trim); the conditional 4 cells, if
  triggered, carry `--ceiling-gpuh 2.32`. Both per-cell ceilings are
  still enforced by the runner's own existing training-phase mechanism
  (`train_earlyln_cell`'s `ceiling_s` argument, `:198-201`); the
  PROGRAM-level bound is now E1's cumulative-realized-GPU-h check (the
  launcher reads `gpu_h` from every `COMPLETED` cell JSON before each
  launch/retry and refuses admission once `realized_gpu_h +
  ceiling(cell) > 15.00`) — this is what makes ≤15.20h GPU-h
  (eval-inclusive, derived in §4) an actual bound rather than an
  aspirational sum, closing KW3.1's finding that Rev 1's per-cell-sum
  bound broke under its own retry rule.
- **Audited + queue-eligible only after this draft clears its own
  audit round** — still explicitly NOT queue-eligible (status header,
  now DRAFT-R2); the pool contract's "ceremony gate stays upstream of
  it" applies here exactly as written.
- **Queue-pool sweep scope, corrected (KW2.7).** §3's internal sweep
  covered `matrix-thinking/queue/jobs/pending/` (the only queue
  directory tracked in this repo) and found zero K∈{26,28,30} hits;
  `~/queue/{fallback_pool,claimed}` on the box were NOT swept this
  session. **Added as a mandatory pre-launch red-team task:** sweep
  both on-box directories for K∈{26,28,30} content before this design
  (or its conditional follow-up) is promoted to the pool.
- **Resource/placement red-team, made explicit in the living body
  (Rev 2, KW3.16).** CLAUDE.md's ceremony tiers require a pre-launch
  resource/placement red-team for any 10–50 GPU-h wave; this design's
  true worst case (≈15.20h, E1) sits in that tier, but Rev 1's only
  red-team task named in §6 was the on-box pool sweep above. Rev 2
  states the requirement here explicitly rather than leaving it only in
  `§A1-ADJUDICATION` (historical): before launch, the red-team round
  must (i) verify E1's cumulative-cap check and E2's restored ceilings
  against whatever launcher script the build stage actually produces
  (not just against this design's prose), (ii) verify E4's
  `harvest()` `n_completed`-vs-`n_seeds` patch is actually applied
  before any K is classified, (iii) confirm the packing density (cells
  per GPU) the build intends, since E1's batch-atomic admission check
  depends on knowing the batch size, and (iv) complete KW2.7's
  still-outstanding on-box `fallback_pool/`/`claimed/` sweep.
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
- **E1 (the cumulative program cap) and E4 (the bounded-retry /
  interval-logic incomplete-cell rule) apply uniformly to BOTH arms,
  stated once here to avoid ambiguity (Rev 2, KW3.16).** Neither rule
  is primary-arm-only or conditional-arm-only: the launcher's
  cumulative-GPU-h check (§4) gates every launch/retry in the whole
  program regardless of which arm it belongs to, and a conditional-arm
  cell that hits `ABORTED-BUDGET` is subject to the identical
  1-retry-then-`PERSISTENTLY-ABORTED` rule and the identical
  interval-logic classification treatment as a primary cell — there is
  no separate, weaker, or unspecified abort-handling path for the
  conditional arm.
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
