# NCR K-WALL CHARACTERIZATION — K∈{26,28,30} ON THE LIVE K=24 RUNG

**STATUS: DRAFT-R4 — POST-AUDIT-4, AWAITING AUDIT ROUND 5 (not
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
    branches = cross_product_of_AMBIGUOUS(states)   # 1, 2, or 4 candidate (r26,r28,r30) triples
    K_trigs = set()
    for triple in branches:
        # scan K=26,28,30,32 in order; r24=4 (fixed ROBUST), r32=0 (fixed archive)
        kt = smallest_K_with_rate_below_3(triple)
        if kt requires reading an UNRESOLVED K's status to decide:
            return ("TRIGGER-UNRESOLVED", blocking_K)   # F2: a K that cannot resolve cannot trigger
        K_trigs.add(kt)
    if len(K_trigs) == 1:
        return ("DECIDED", K_trigs.pop(), "unanimous")
    else:
        return ("DECIDED", min(K_trigs), f"tie-break-min, candidates were {sorted(K_trigs)}")
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
    branches = cross_product_of_AMBIGUOUS(states)   # 1, 2, or 4 candidate (r26,r28,r30) triples
    K_trigs = set()
    for triple in branches:
        # scan K=26,28,30,32 in order; r24=4 (fixed ROBUST), r32=0 (fixed archive)
        kt = smallest_K_with_rate_below_3(triple)
        if kt requires reading an UNRESOLVED K's status to decide:
            return ("TRIGGER-UNRESOLVED", blocking_K)   # F2: a K that cannot resolve cannot trigger
        K_trigs.add(kt)
    if len(K_trigs) == 1:
        result = ("DECIDED", K_trigs.pop(), "unanimous")
    else:
        result = ("DECIDED", min(K_trigs), f"tie-break-min, candidates were {sorted(K_trigs)}")
    # G5 precondition -- checked only once the K-scan itself decides:
    band = classify_with_interval_logic(states_26_28_30)   # sec.5's own procedure, SAME triple
    if band == "INCOMPLETE-AT-K":
        band_blocked_K_trig = result[1]   # the K-scan's own candidate -- disclosed, not dropped
        return ("TRIGGER-UNRESOLVED", None, band_blocked_K_trig)
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
domain, not illustrative. **Scope correction (KW5.13, Rev 4): the
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
`0.5106×1.9355=0.9882`, `0.5536×1.9355=1.0716`,
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
1. **Attempt 1.** HARD GATE check (below). Refused → state
   `GATE-REFUSED`, ledger unchanged, move to the next cell (treated
   identically to MISSING by the resolution-state table above and by
   `harvest()`, §4 D5/E4 — G1's cell-level resume rule, below, is what
   keeps this true across a restart too). Admitted →
   **write-ahead (G1):** BEFORE calling `subprocess.run`, the
   orchestrator sets `ledger.open_attempt = {K, seed, arm, attempt_n,
   charged_ceiling: ceiling(this_attempt), dispatch_ts:
   time.monotonic()}` and persists `ORCHESTRATOR_LEDGER.json` — the
   ONLY ledger write that ever happens BEFORE a subprocess runs, so a
   mid-attempt orchestrator death leaves a detectable gap, never a
   silent one (recovery procedure below). Then run the subprocess
   (command shape above, `attempt1` outdir). The orchestrator's OWN
   wall-clock timer — `t0` (`dispatch_ts` above), `t1` immediately
   after `subprocess.run` returns, `attempt_elapsed_h=(t1-t0)/3600` —
   is the measurement, **not** the cell JSON's `gpu_h` field (KW4.1:
   that field is assigned only on the `COMPLETED` path,
   `ncr_earlyln_scale.py:304`, and is absent entirely on the
   `ABORTED-BUDGET` early-return path, `:262-266`). On return, `ledger.
   realized_gpu_h += attempt_elapsed_h` UNCONDITIONALLY, `ledger.
   open_attempt` is cleared, and a terminal row is appended to
   `ledger.attempts` — before the resulting JSON's `status` is even
   inspected. **Exit-code-exact branch (G3, replaces "non-zero exit ⇒
   ABORTED-BUDGET"):**
   - exit code `3` (the `--stop-file` sentinel,
     `ncr_earlyln_scale.py:196`) → state `STOPPED-BY-OPERATOR`
     (never retried — see "operator stop," below).
   - cell JSON written with `status=="COMPLETED"` → state `COMPLETED`
     (terminal).
   - cell JSON written with `status=="ABORTED-BUDGET"` → state
     `ABORTED-BUDGET-1`, proceed to step 2.
   - any OTHER non-zero exit, no `COMPLETED`/`ABORTED-BUDGET` JSON on
     disk → state `CRASHED-1` (a deterministic crash — shape error,
     OOM, import failure — never a coin-flip seed the way a budget
     abort is; disclosed distinctly, per KW5.3), proceed to step 2
     under the SAME gates.
2. **Retry (attempt 2), from `ABORTED-BUDGET-1` OR `CRASHED-1` only —
   never from `STOPPED-BY-OPERATOR`.** HARD GATE **and** RETRY GATE
   (below), both checked against the ledger AS IT STANDS after attempt
   1 — already updated in step 1 (and, if attempt 1 crashed rather than
   returned, closed by the G1 recovery procedure on the next restart
   BEFORE this check ever runs) — so there is no staleness: strict
   sequencing plus write-ahead recovery means the ledger is exactly
   current, or conservatively over-stated, at every check. Both pass →
   run attempt 2 (`attempt2` outdir, identical write-ahead +
   ledger-update + exit-code-branch discipline); `COMPLETED` → state
   `COMPLETED`; `ABORTED-BUDGET`/`CRASHED` again (whichever recurs) →
   state `PERSISTENTLY-ABORTED` (terminal — feeds interval logic as an
   unknown-numerator seed regardless of which of the two produced it,
   D5/E4 below). Either gate fails → state `PERSISTENTLY-ABORTED`
   immediately, attempt 2 is never dispatched, zero additional ledger
   spend. (Retraining is still from scratch — the harness has no
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
repeat — CLAUDE.md's resume-safe-supervisor rule).**

`/home/nvidia/ncr/results_kwall_characterization/ORCHESTRATOR_LEDGER.json`
carries `realized_gpu_h`, a completed-attempt log `attempts[]` (one row
per TERMINAL attempt: `{K, seed, arm, attempt_n, elapsed_h, status,
outdir}`), and exactly one `open_attempt` slot (`null` when nothing is
in flight — dispatch is strictly serial, so at most one can ever be
open at a time). The file is rewritten after EVERY ledger update — an
`open_attempt` write BEFORE `subprocess.run`, and an `open_attempt`
clear + `attempts[]` append + `realized_gpu_h` increment AFTER it
returns — not only at the end.

**Recovery procedure, run on ANY (re)start, BEFORE any gate check and
BEFORE the cell-order walk resumes:**
1. Read `ORCHESTRATOR_LEDGER.json`. If `open_attempt` is `null`,
   nothing is dangling — proceed to step 3.
2. **A non-null `open_attempt` means the orchestrator died between
   writing it and clearing it — mid-attempt.** Close it
   conservatively: append a terminal row to `attempts[]` with
   `status="CRASHED-RECOVERED"`, `elapsed_h=open_attempt.
   charged_ceiling` (the FULL ceiling that attempt was gated on — not
   a measured value, which no longer exists once the process that
   would have measured it is dead), `ledger.realized_gpu_h +=
   open_attempt.charged_ceiling`, then set `open_attempt=null` and
   persist. That cell's attempt/retry state machine treats
   `CRASHED-RECOVERED` exactly like `CRASHED-n` (G3) for retry-gating
   and interval-logic purposes (D5/E4, below) — a crash the
   orchestrator recovers FROM restart is not distinguishable, from the
   cell's perspective, from a crash it observed directly.
3. **Cell-level resume (closes KW5.1's second, smaller gap — the
   `GATE-REFUSED`-vs-`harvest()` disagreement).** Walk cell order
   (above) from the start, but a cell/attempt with an existing
   TERMINAL row already in `ledger.attempts[]` (`COMPLETED`,
   `ABORTED-BUDGET`, `CRASHED`, `CRASHED-RECOVERED`,
   `PERSISTENTLY-ABORTED`, or `STOPPED-BY-OPERATOR`) is **never
   re-gated** — its state comes from the ledger record, not a fresh
   HARD-GATE decision, so a restart at `realized≈13h` can never turn
   an already-`COMPLETED` cell into `GATE-REFUSED`: the ledger is
   consulted FIRST, and `GATE-REFUSED` is only ever produced for a
   cell/attempt with no ledger row of any kind yet. This is also what
   makes the "`GATE-REFUSED` treated identically to MISSING by
   `harvest()`" claim (dispatch loop step 1, above) correct exactly as
   stated, on a restart as much as on a first pass: a `GATE-REFUSED`
   cell/attempt never ran a subprocess, so it never produced a
   canonical output file (G2, below) either — `harvest()`'s view and
   the orchestrator's own ledger view agree by construction, because
   both ultimately trace back to "did a subprocess for this cell ever
   COMPLETE," never to two independently-maintained records that could
   disagree.
4. Once every dangling record is closed and every already-terminal
   cell is skipped, dispatch resumes from the first cell/attempt with
   no ledger row — normal operation from here.

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
discloses, not a new one.)

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
ABORTS LOUDLY (raises) instead of overwriting. **This exists-check can
only fire on a genuine invariant violation** — the dispatch loop above
only ever advances a cell to attempt 2 from `ABORTED-BUDGET-1`/
`CRASHED-1`, never from `COMPLETED`, so no cell can produce two
`COMPLETED` attempts in normal operation; the only way to trip it is
an operator re-running the orchestrator against a dirty, pre-existing
results directory, which is exactly the case that SHOULD fail loudly
rather than silently mis-harvest.

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
build instruction below:** because a canonical-path file is written
ONLY on `COMPLETED` acceptance, `discover_seeds_by_K`'s existing
file-glob-presence count over the CANONICAL directory is now IDENTICAL
to a status-based `n_completed` count by construction — no
`ABORTED-BUDGET`/`CRASHED`/`PERSISTENTLY-ABORTED`/`GATE-REFUSED` cell
can ever produce a canonical file to be miscounted. The `harvest()`
code-patch D5/E4 previously specified (file-presence → status-based)
is therefore NO LONGER NEEDED as a separate change; it is subsumed by
G2's copy-on-accept discipline. (Confirmed against the actual code,
unchanged this revision: `harvest()`'s non-recursive glob at
`ncr_earlyln_scale.py:358-380` is exactly the mechanism this relies on.)

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

**Output JSON (`orchestrator_report.json`), required fields (Rev 4:
`run_status` enum exhaustive (G4); `attempts[].status` enum exhaustive
per G1/G3's actual reachable states (KW5.3); `open_attempt` exposed for
transparency; `trigger.band_blocked_K_trig` discloses G5's suppressed
cases; `band.incomplete_at_K`/`candidate_bands` and a `smoke` block
close KW5.9's missing fields; `gpu_id`/`git_commit` added, also KW5.9):**

```
{
  "run_status": "COMPLETE" | "COMPLETE-DEGRADED" |
    "STOPPED-BY-OPERATOR" | "EXHAUSTED-BUDGET",
  "ledger": {
    "realized_gpu_h_final": <float>, "hard_gate_cap": 15.00,
    "retry_gate_threshold": 12.00, "declared_pool_ceiling": 15.50,
    "open_attempt": null,
    "attempts": [ {"K":int,"seed":int,"arm":"primary"|"conditional",
      "attempt_n":1|2,"elapsed_h":float,
      "status":"COMPLETED"|"ABORTED-BUDGET"|"CRASHED"|
        "CRASHED-RECOVERED"|"GATE-REFUSED"|"STOPPED-BY-OPERATOR",
      "outdir":str}, ... ] },
  "smoke": {"K26":"PASS"|"FAIL", "K28":"PASS"|"FAIL",
    "K30":"PASS"|"FAIL"},
  "primary": { "per_K": { "26": {...}, "28": {...}, "30": {...} } },
  "trigger": { "resolution": "unanimous"|"tie-break-min"|
    "TRIGGER-UNRESOLVED", "K_trig": int|null,
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

**`run_status` enum, defined exhaustively (G4, Rev 4 — closes KW5.4,
where the schema offered two undefined values and the job spec's own
`validity_check` rejected the design's own pre-registered degraded
outcome).**
- **`COMPLETE`.** Every primary cell's first attempt was admitted by
  the hard gate (no `GATE-REFUSED` anywhere in the run), the trigger
  was evaluated, and — if it fired — every conditional cell's first
  attempt was likewise admitted. No budget-caused refusal anywhere.
- **`COMPLETE-DEGRADED`.** Every primary cell got its first attempt
  (the hard gate never refused a PRIMARY cell's first attempt — the
  12-cell baseline sweep completed), but the hard/retry gates
  throttled something downstream of that baseline for budget reasons
  alone. Two pre-registered sub-cases, both disclosed via `attempts[]`
  and neither treated as a bug: (i) *primary-retry-refused* — a
  primary cell's attempt-2 retry was denied by the HARD or RETRY gate
  rather than the state machine reaching a natural
  `PERSISTENTLY-ABORTED` after both attempts ran; that cell still
  follows D5/E4's interval logic exactly as any other incomplete cell.
  (ii) *conditional-throttled* — the trigger fired (`DECIDED`, and the
  G5 band precondition held), but 1-4 of the conditional arm's 4
  cells' first attempts were refused by the hard gate before the 15.00
  cap was reached — the graceful degradation the ORIGINAL text already
  disclosed ("the hard gate correctly THROTTLES OR REFUSES part or all
  of the conditional arm... degrading gracefully... never to a silent
  overrun"), now given an actual reportable label.
- **`STOPPED-BY-OPERATOR`.** The `--stop-file` sentinel was seen (G3).
  Terminal for the whole run at whatever point it occurred; never a
  gate refusal, never retried.
- **`EXHAUSTED-BUDGET`.** The hard gate refused a PRIMARY cell's OWN
  FIRST ATTEMPT — the 12-cell baseline itself could not be completed
  inside the ceiling. Reachable in principle, not vacuous: 12 primary
  first attempts at the shared `1.20h` ceiling sum to `14.40h`, inside
  `15.00` only if every one is admitted with essentially no headroom
  left for anything else. More severe than `COMPLETE-DEGRADED`: most
  or all of §5's K's will read `INCOMPLETE-AT-K`.

All four are DISCLOSED, TERMINATING, non-buggy outcomes of the
orchestrator behaving exactly as designed (finishing, being told to
stop, or protecting its own ceiling) — none indicates a defect. The
job-spec `validity_check` (below) accepts the three that represent the
orchestrator completing its own logic (`COMPLETE`/`COMPLETE-DEGRADED`/
`EXHAUSTED-BUDGET`) as `completed/`, never `failed/`. `STOPPED-BY-
OPERATOR` is the one exception: a deliberate human action to interrupt
the run is not itself a completed-or-gracefully-degraded RESULT of
this design's own logic, so it stays outside the accept-set;
resubmitting the job resumes cleanly (G1's cell-level resume rule
skips every already-terminal cell) at no cost beyond the ceiling
already spent.

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
- **It never returns** (the orchestrator itself dies): `R_N`, as
  computed by the NEXT restart's recovery step, is `R_{N-1} +
  charged_ceiling(N) ≤ 15.00` EXACTLY — no tail term at all, because
  the recovery charge is the gate-admitted ceiling value, not a
  measurement. This is TIGHTER than the return case.

```
R_N  ≤  15.00 + 0.0157  =  15.0157 GPU-h
```

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
`≈15.20h`, tighter than but consistent with the true `15.0157h` this
revision's derivation proves — `0.0031h` looser than Rev 3's
`15.0126h` claim, per the KW5.7 correction, still comfortably inside
the disclosed figure) and adding a **stated — not derived; a policy
choice, disclosed as such — supervisor margin**, covering the
log-interval overshoot's own disclosed contention-variance
("proportionally more under exactly the contention the ceiling exists
to survive," KW4.9) plus orchestrator-process overhead the cell-level
`ceiling_s` check does not model (subprocess spawn/interpreter/import
latency, ledger-file I/O, the two `harvest()` invocations), fixed at a
round, generous **0.30 GPU-h**:

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
  canonical directory — no patch is specified or needed. The rule that
  DOES still need a build-stage implementer to apply it (this design
  stays DRAFT and edits no code) is the classification logic itself:
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
derived-plus-margin ceiling). **`validity_check`, updated for G4 (Rev
4 — closes KW5.4: the old assertion `run_status=="COMPLETE"` rejected
this design's own pre-registered graceful-degradation outcome,
routing a legitimate result to `failed/`),** asserts, over the
orchestrator's OWN `orchestrator_report.json` (schema above):
`run_status in {"COMPLETE","COMPLETE-DEGRADED","EXHAUSTED-BUDGET"}`
(every value G4 defines as a non-buggy completion of the
orchestrator's own logic — `STOPPED-BY-OPERATOR` is deliberately
excluded, see G4 above), `ledger.realized_gpu_h_final <= 15.50`,
`ledger.realized_gpu_h_final == sum(a["elapsed_h"] for a in
ledger.attempts)` (self-consistency — the reported total must equal
the sum of its own logged rows, catching a bookkeeping slip directly
rather than trusting the running total blindly), and — closing the
exact `d=K+1`-vs-`d=2K` filename-collision risk
`EXPERIMENT_LOG.md:8452` already forced a workaround for once —
`all(a["K"]+1 == d_override_of(a) for a in ledger.attempts)`, i.e.
every logged attempt's underlying cell carried `d==K+1`/
`d_override==K+1`, checked across the WHOLE ledger rather than
per-cell (the attempt-indexed outdirs already avoid the collision by
construction; this is defense-in-depth against a mis-flagged attempt
silently harvesting under the wrong convention). **No per-cell job
specs are created** — the orchestrator is the only pool artifact this
design produces (§6).

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
  conservatively-rounded internal worst case; the tight derivation
  proves `15.0157h`, §4 — Rev 4's corrected figure, extending Rev 3's
  `15.0126h` to cover the mid-attempt-crash case, G1, and the
  KW5.7-corrected single-attempt tail) plus a **stated 0.30h
  supervisor margin** (§4). This is enforced ENTIRELY inside the
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
  and adds (vi)–(ix) for G1–G5).** CLAUDE.md's ceremony tiers require a
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
  `COMPLETE-DEGRADED`/`EXHAUSTED-BUDGET` → `completed/`,
  `STOPPED-BY-OPERATOR` → `failed/`); (ix) verify G5's trigger
  precondition is wired — a synthetic state where the K-scan alone
  would return `DECIDED` but the whole-study band is `INCOMPLETE-AT-K`
  must produce `TRIGGER-UNRESOLVED` with a disclosed
  `band_blocked_K_trig`, never a dispatched conditional arm.**
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
