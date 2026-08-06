# NCR K-WALL CHARACTERIZATION — K∈{26,28,30} ON THE LIVE K=24 RUNG

**STATUS: DRAFT-R0 — AWAITING AUDIT ROUND (not build-released, not
queue-eligible).**

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

Under the exact recipe that already brackets a clean pass (K=24) against
a measured failure (K=32) — earlyln free-write, tight-spare `d=K+1`,
encoder hidden `h=64` fixed (non-binding throughout this range), 80,000
training steps — Gate-0's **in-distribution recovery leg**
(`indist_min = min(recovered_frac@0.9` for `h∈{1,2,3})`, the leg
`NCR_KLADDER_ATTACK_R2.md` finding A4.4 identifies as the one that
actually fails at K=32 while the rank leg passes) resolves into
additional K-scale structure inside the untested K∈{26,28,30} window —
i.e., the archive's current 8-K-wide bracket (K=24 clean, K=32 dead) is
not yet known to be a single atomic step, a gradual decline, or a step
still located exactly at K=32, and this is answerable for a low,
capped GPU-hour spend without touching the rank leg, the ortho/NS-polar
mechanism, or any K≥32 cell.

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
`d=25 = K+1`, not `d=2K=48`. The `d=2K` config at the same K was
independently measured and is NOT clean: the 2026-07-12 mapping-law
Q2 extension (K=24, `d=48=2K`, n=4→n=12) reads *"Reliability under the
strict whole-sweep metric is low: `sweep_min_rec` HOLD in 0/12,
DEGRADED in 1/12, FAIL in 11/12; the looser `front≥h*` metric clears
in 4/12 (33%)"* (`EXPERIMENT_LOG.md:8672-8677`). At `d=2K`, K=32's own
d(K) grid (which includes `d=64=2K` as one of its four tested `d`
values) reads **uniformly TRAINABILITY-DEAD** — *"every arm lands
TRAINABILITY-DEAD (0/4 fully Gate-1 CONVERGED)... front is pinned at
the trivial K−3=29 rung in all 16 cells, every d, every seed — zero
far-depth signal anywhere"* (`EXPERIMENT_LOG.md:8632-8637`). There is
no clean anchor anywhere in the archive under `d=2K`; there is one
under `d=K+1`. A wall-characterization study needs a live point to
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

**Claim under test: K∈{25,26,27,28,29,30,31} is genuinely untouched
anywhere in this repo.** Verified by direct search this session, not
assumed:
- `grep -rn` for `K=2[5-9]`, `K=3[01]`, `K25`..`K31` across
  `EXPERIMENT_LOG.md`, every `matrix-thinking/*.md`, `STATE.md`,
  `matrix-thinking/KILL_LIST.md` — the only two textual hits
  (`EXPERIMENT_LOG.md:8569`: *"at K=25→K+1..."*, meaning K24's
  `d=25=K+1`, not a K=25 cell; `NCR_MAPPING_LAW_DESIGN.md:471`:
  `"1.25K=30"`, meaning `d=1.25×24=30`, a `d`-value not a K-value) —
  are both false positives on inspection, not real K=25/K=30 cells.
- `grep` over every `matrix-thinking/queue/jobs/**/*.json` (pending +
  fallback_pool + claimed, all lanes) for `"K": 2[5-9]`/`"K": 3[01]`
  and `--K 2[5-9]`/`--K 3[01]` — zero hits.
- `find experiment-runs -iname "*K2[5-9]*" -o -iname "*K3[01]*"` —
  zero directories.
- `archive/` grepped the same way — zero hits.
- Code-level: `GRIDS`/`GRID_SHAPES` (both `ncr_task.py` and
  `ncr_earlyln_scale.py`) list every K they define; 26/28/30 are
  absent from both dicts (confirmed by direct read, §2 above).

**Conclusion: K∈{26,28,30} is open.** No cell, job spec, archived
result, or standing block references it. This clears the "internal
sweep must not redo or contradict our own recorded work" gate
(CLAUDE.md, Waterfall Process) — there is nothing to redo, and the
only adjacent standing block (`EXPERIMENT_LOG.md:8638-8640`, the
WAVE-1b K=48 block) is scoped to K=48's own `d(K)` grid under the
`d=2K`/rebuild premise this design explicitly does not use (§2) and
therefore does not apply here; this is stated, not silently assumed.

**What the archive DOES say about the two flanking, already-measured
K's, at the SAME `d=K+1`, 80,000-step recipe this design reuses —
re-derived directly from the raw per-seed JSONs this session, not
copied from summary prose (file paths + the exact field read given for
each number):**

| K | d | seed | h1 | h2 | h3 | `indist_min` | label | mean A_eff_rank | AER/K | gpu_h |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 17 | 0 | 1.000 | 1.000 | 1.000 | **1.000** | CONVERGED | 15.999 | 1.000 | 0.407 |
| 16 | 17 | 1–3 | 1.000 | 1.000 | 1.000 | **1.000** | CONVERGED | 15.999–16.000 | 1.000 | 0.380–0.397 |
| 24 | 25 | 0–3 | 1.000 | 1.000 | 1.000 | **1.000** | CONVERGED | 23.924–23.998 | 0.997–1.000 | 0.442–0.498 |
| 24 | 25 | 4,10,11 (n=12 ext.) | 1.000 | 1.000 | 1.000 | **1.000** | CONVERGED | (not re-pulled; log-cited) | — | — |
| 32 | 33 | 0 | 0.900 | 0.710 | 0.464 | **0.464** | DEAD | 29.65–29.73 | 0.928 | 0.594 |
| 32 | 33 | 1 | 0.913 | 0.743 | 0.517 | **0.517** | PARTIAL | 29.40–30.05 | 0.930 | 0.566 |
| 32 | 33 | 2 | 0.952 | 0.854 | 0.688 | **0.688** | PARTIAL | 29.97–30.41 | 0.941 | 0.574 |
| 32 | 33 | 3 | 0.990 | 0.951 | 0.871 | **0.871** | PARTIAL | 30.93–31.04 | 0.966 | 0.542 |

Sources: `experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/earlyln_K{16,24}_s{0-3}.json`
(`eval.points[h].reads.binexp['recovered_frac@0.9']` for `h∈{1,2,3}`,
`deep_probe.A_eff_rank`, `gpu_h`), `experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext/earlyln_K24_s{4,10,11}.json`
(same fields), `experiment-runs/2026-07-12_ncr_mappinglaw_wave1/dratio_K32_d33/earlyln_K32_s{0-3}.json`
(same fields) — read directly this session via `json.load`, not
transcribed from prose. The `0.871` best-seed figure and the "3/4
PARTIAL" characterization in `EXPERIMENT_LOG.md:8642-8644` reproduce
exactly against these raws (seed 3: `indist_min=0.871`; seeds 1–2:
PARTIAL; seed 0: `0.464`, which is actually DEAD by the module's own
`<0.5` bar, a finer disclosure than the log's "3/4 PARTIAL" summary —
seed 0 is the 4th seed and is DEAD, not PARTIAL; **0/4 CONVERGED, 3/4
PARTIAL, 1/4 DEAD** is the precise breakdown).

**Reading the bracket.** At the exact metric this design makes primary
(`indist_min`, `ncr_earlyln_scale.py:319`, bars at `:95-96`): K=16 and
K=24 are **perfectly** clean (`indist_min=1.000` in every one of 4+4+3
sampled seeds, no seed below 1.000, not merely above the 0.9 bar). K=32
is **uniformly not CONVERGED** (0/4, `indist_min∈[0.464,0.871]`, none
reaching 0.9). The rank leg (`AER/K`) clears its own 0.9 bar in every
K=32 seed (0.928–0.966) even as the recovery leg fails — this is the
raw evidence underneath A4.4's finding, reproduced directly here at
80K steps (A4.4 itself cites the 320K numbers; this table shows the
SAME leg-dissociation already holds at 80K, which is why 80K is a
scientifically honest budget for this specific metric, not merely a
cost compromise — see §2(e)).

**The honest power caveat, on a DIFFERENT metric — cited so it is not
silently reused for the wrong one.** `EXPERIMENT_LOG.md:8663-8677`'s
n=4→n=12 extension at K=24 found the FAR-DEPTH metric
(`sweep_min_rec`/`front≥h*`) is noisy — a 33% per-seed success rate on
the looser metric, 0/12 on the strict one, "a near coin flip" per
`NCR_KLADDER_ATTACK_R2.md` finding A4.8. That noise is measured on
`front`/`sweep_min_rec` (Gate 2, far-depth), not on `indist_min`
(Gate 1, this design's primary metric) — the SAME n=12 extension shows
`indist_min=1.000` in all three re-pulled seeds (4, 10, 11) with zero
disagreement from the n=4 read. The two metrics are not equally noisy;
§5 uses this asymmetry explicitly and does not borrow A4.8's coin-flip
caveat for a metric it was never measured on.

---

## §4 CELL GRID + PRICING

**Grid: K∈{26,28,30} × seed∈{0,1,2,3} = 12 cells, single arm (free
only), Part A only (single-relation; no discriminator/bank cells).**
n=4 is both the assignment's stated minimum and what the GPU-hour cap
affords at a comfortable per-cell safety margin (below) — raising n
would mean either fewer K's or a thinner ceiling margin, a trade this
design does not make silently.

**Command (per cell, mirrors the parked-relic job-spec style at
`matrix-thinking/queue/jobs/pending/108_laneA_main_K48_s0.json`, same
CLI surface, different flags per §2):**

```
ncr_earlyln_scale.py --cell --K {26,28,30} --d-override {27,29,31} \
  --seed {0,1,2,3} --steps 80000 --ceiling-gpuh 1.25 \
  --outdir results_kwall_characterization \
  --stop-file results_kwall_characterization/STOP
```

**Pricing — empirically grounded, not FLOP-extrapolated from a
different config family.** The parked K48 relic (`d=2K=96`) cost
~1.154 GPU-h nominal at 80K steps (`queue/jobs/pending/108_laneA_main_K48_s0.json`)
— but that cell is a materially bigger write (`d=96` vs this design's
`d=27-31`) under a config family §2 already rejected. The directly
relevant, ALREADY-MEASURED rate is Probe A's own `d=K+1`, 80K-step
cells: K=16 (`d=17`) 0.380–0.407 GPU-h/cell, K=24 (`d=25`)
0.442–0.498 GPU-h/cell (raw `gpu_h` field, same JSONs as §3's table).
Closed-form cross-check (`P(d,h)=40h²+4dh+46h+d`,
`F(K,d,h)=76Kh²+4dh²+12K²h+4Kdh+4d²h`, `h=64`, no `NS(d)` term since the
free arm never runs Newton–Schulz): `F(26,27,64)=9,421,568`,
`F(28,29,64)=10,216,704`, `F(30,31,64)=11,022,080` — a mild 1.17×–1.28×
spread over the measured K=24 rate, consistent with the ~28% larger
FLOP count. **Nominal estimate: ≈0.50 GPU-h/cell** (rounds the
measured 0.44–0.50 K=24 range up slightly to cover the ~15-28% FLOP
growth to K=30). **Per-cell ceiling: 1.25 GPU-h** (~2.5× nominal,
matching the safety-margin shape of the parked relic's own "2×
nominal, floor 1.0h" convention, `queue/jobs/pending/108...json`
`"notes"` field) — resume-safe by construction (`ncr_earlyln_scale.py`'s
own whole-cell skip-if-COMPLETED, inherited, same as the parked
relics).

| | per-cell | ×12 cells |
|---|---|---|
| Nominal (empirically grounded) | ≈0.50 h | **≈6.0 GPU-h** |
| Hard ceiling (abort trigger) | 1.25 h | **15.0 GPU-h** |

**Total ≤15 GPU-h, exactly the mandate's cap, at the pessimistic
every-cell-hits-ceiling reading; realistically ≈6 GPU-h.** No trim
order is needed (unlike `NCR_KLADDER_DESIGN.md`'s multi-hundred-GPU-h
grid) — the full 12-cell grid fits inside the cap even at 2.5× the
measured rate, and every cell is independently resumable, so a
mid-run interruption loses at most one cell's partial progress, not
the wave.

---

## §5 PRE-REGISTERED OUTCOME BANDS

**Metric definitions (exact reuse of `ncr_earlyln_scale.py`'s own
pinned bars — no new threshold invented, per CLAUDE.md's
exact-structural-threshold rule):**
- Per-seed `indist_min = min(recovered_frac@0.9` at `h∈{1,2,3})`
  (`:319`). Per-seed label: **CONVERGED** (`≥0.9`, `CONVERGED_INDIST_BAR`,
  `:95`) / **PARTIAL** (`[0.5,0.9)`, `PARTIAL_INDIST_BAR`, `:96`) /
  **DEAD** (`<0.5`).
- Per-K CONVERGED-rate `= (#seeds CONVERGED)/4`. Per-K label (module's
  own vocabulary, `:19-20`, reused verbatim): **CONVERGED-ROBUST**
  (`≥3/4`) / **CONVERGED-PARTIAL** (`1-2/4`) / **TRAINABILITY-DEAD**
  (`0/4`).
- **Rank leg, reported not gating:** mean `A_eff_rank/K` per K
  (`AEFF_RANK_FRAC_BAR=0.9`, `:97`) — expected to clear throughout
  this range (it clears even at K=32 where recovery fails, §3's
  table), reported to confirm the wall stays localized to the
  recovery leg rather than silently becoming a rank story, per A4.4.
- The per-seed CONVERGED/PARTIAL/DEAD label is reported alongside the
  collapsed per-K rate at every K — not just the rate — so a
  "qualitatively least-dead" result (as K=32 itself is, per
  `EXPERIMENT_LOG.md:8642`) is not flattened into an indistinguishable
  `TRAINABILITY-DEAD` bucket the way the module's coarse per-K label
  alone would do.

**Guard against A4.9's defect (a WIN/verdict decided on a
data-dependent n=2 median with no floor):** this design's per-K label
is a **rate over the full fixed n=4**, never a median over a
gate-passing subset — every seed counts in the denominator regardless
of outcome, so there is no selection effect to disclose and no
sub-population to condition on.

**Three bands over the pattern across K∈{26,28,30}, read against the
fixed archive brackets K=24 (CONVERGED-ROBUST, 4/4, `indist_min=1.000`
uniformly) and K=32 (TRAINABILITY-DEAD by rate, 0/4, but qualitatively
graded 0.464–0.871, §3):**

**(a) WALL-AT-K\*.** Some `K*∈{26,28,30}` is the last CONVERGED-ROBUST
rung (`rate(K*)≥3/4`) AND the very next tested rung (K*+2, or K=32 if
`K*=30`) drops to `rate≤1/4` — i.e. a genuine multi-seed swing, not a
one-seed wobble (a `3/4→2/4` step does NOT qualify here; see band (b)).
**Licenses:** report `K*` as the wall's location to ±2 precision (an
8×→2× tightening of the current bracket); recommend `K*` (if `>24`)
replace K=24 as the flagship real-LM campaign's provisional "last live
rung" per `STATE.md:127-134`'s un-gating logic — a free trainability
upgrade if `K*∈{26,28,30}`; motivate a follow-on mechanism probe
(analogous to the 07-12 Q3 leakage analysis, CPU-only, zero GPU) asking
what changes structurally at `K*→K*+2`, since a sharp step is a more
tractable mechanistic target than a smooth decline.

**(b) GRADUAL-DECAY.** CONVERGED-rate declines across {26,28,30}
without a single ≥2-seed step matching band (a)'s criterion (e.g.
`4/4→3/4→2/4` feeding into K=32's `0/4`, or any monotonic sequence
whose largest single-rung drop is ≤1 seed). **Licenses:** the wall is
not a discrete architectural transition inside this window but a
continuous K-dependent degradation; fit a coarse scaling curve across
the now six-point 80K-budget series K∈{16,24,26,28,30,32} (`indist_min`
or CONVERGED-rate vs K) — itself a small, disclosed, publishable
characterization, not a scaling LAW claim (six points, one budget, one
d-convention; no periodicity or extrapolation claim beyond this
range); flag as a candidate DATA POINT for whether the 320K-budget
"clean-at-K=24" recipe would also rescue the intermediate K's — an
explicitly NOT-committed follow-on (§7), since answering it needs a
320K-budget sub-wave this design's ≤15 GPU-h cap does not cover.

**(c) NO-WALL-BELOW-32.** All three of K∈{26,28,30} read
CONVERGED-ROBUST (`≥3/4`), matching K=24's clean profile. **Licenses:**
the entire measured wall stays exactly where the archive already put
it — the single 30→32 step — with nothing finer to find below it;
recommend the flagship's provisional "last live rung" move from K=24
to K=30 outright (three additional relational-capacity units at zero
newly-measured risk); redirect any future mechanism hunt at the
absolute-K-specific effect the mapping-law harvest already flagged
(`EXPERIMENT_LOG.md:8650-8661`, the "bounded to K≤24, not a general
`d(K)` law" scope correction A4.6 also cites) rather than a
relative-headroom story, since relative headroom (`1/d`) varies
smoothly across 24→30→32 while trainability would not have.

**INDETERMINATE-AT-K flag (disclosed, not folded into any band):** any
tested K landing at exactly `2/4` CONVERGED is reported with this flag
— it is a real, defined label (`CONVERGED-PARTIAL`) under the module's
own vocabulary, but per §3's honest power caveat this design does not
treat a `3/4→2/4` transition alone as licensing band (a)'s "wall
found here" conclusion; it is read as consistent with band (b)
(gradual) unless a same-direction, larger-magnitude drop appears at
the next rung too.

**All three bands are informative and reportable** — none requires a
follow-on wave to be publishable as a characterization result; (a) and
(c) each localize the wall to a resolution better than the current
archive; (b) produces the K-ladder's own missing scaling-curve data
point at essentially zero incremental discussion cost.

---

## §6 POOL-ELIGIBILITY STATEMENT

This grid satisfies `matrix-thinking/queue/idle_fallback_daemon.sh`'s
own pool contract (header, `:10-16`): *"the pool holds ONLY flat
specs — each fully audited + queue-eligible, independently runnable in
any order, carrying its own cost ceiling, with NO intra-wave
dependencies, stage gates, or staged-escalation semantics."* Verified
against this design's own structure, not asserted:
- **Independent:** no cell's launch condition depends on another
  cell's result. Unlike the SPENT K-ladder design (`NCR_KLADDER_ATTACK_R2.md`
  finding A4.12: Stage-0 blocks everything, K=48 gates K=64), all 12
  cells here can run in any order or fully in parallel — the §5 bands
  are read at harvest time, over the complete 12-cell set, never as a
  pre-launch gate on a later cell.
- **Own cost ceiling:** every cell carries `--ceiling-gpuh 1.25`
  (§4), enforced by the runner's own existing ceiling mechanism
  (`els.train_earlyln_cell`'s `ceiling_s` argument, the same mechanism
  job 108 and every other parked/live cell already uses).
- **Audited + queue-eligible only after this draft clears its own
  audit round** — this document is explicitly NOT queue-eligible yet
  (status header); the pool contract's "ceremony gate stays upstream
  of it" applies here exactly as written.
- **No standing restriction bites.** The `STATE.md:39-40` "NO NCR job
  queue-eligible" restriction (2026-07-30) is scoped to the in-LM
  write-conditioning claim pivot; this design makes no in-LM claim and
  no claim pivot — it characterizes an already-cleared toy-scale
  mechanism (S11 earlyln free-write; NCR core mechanism NOVEL per
  `research/novelty-gate-2026-07-27.md`) at new K values, the same kind
  of additive K-extension the 2026-07-11 queue-system build already
  did without a fresh novelty gate. If the audit round disagrees with
  this reading it should say so explicitly rather than silently defer.

---

## §7 NON-GOALS

- **No K≥32 cell.** K=32 is CLOSED-AT-THIS-K per the mapping-law
  harvest (`EXPERIMENT_LOG.md:8632-8637`); this design does not
  re-run it, does not extend past it, and does not touch the K=48
  WAVE-1b block (`:8638-8640`).
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
- **No budget-instrument unification claim.** This design does not
  attempt to show 80K and 320K give the same verdict on every metric —
  §3 explicitly shows they diverge on the far-depth leg and only
  argues equivalence on the primary (`indist_min`) leg, evidenced
  directly (n=4 and n=12 agreement at K=24), not assumed by analogy to
  the primary leg. A 320K confirmation sub-wave, if band (a) or (b)
  motivates one, is future PI-gated work, not part of this design or
  its ≤15 GPU-h cap.
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
