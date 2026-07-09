# CAPABILITY-SEPARATION — group-element-recovery rank/representation-dimension test

## §1 DESIGN — Stage 1 (Rev 4, post-attack-round-4 revision, 2026-07-08) —
does a from-scratch matrix-state model recruit subspace-restricted state
rank equal to a group's minimal faithful real representation dimension
`d_min(G)`, tracking dimension not solvability?

Status: **Rev 4, pre-attack (round 5 pending), zero GPU spent.** Rev 0
(§1.1-§1.12) was attacked (§1.13: NEEDS-REVISION — 1 FATAL-as-written + 3
MAJOR + 2 minor); every finding was resolved by Rev 1 (mapped in §1.14).
Rev 1 was attacked again (§1.15: NEEDS-REVISION — 2 MAJOR + 5 minor,
narrow scope, all cheap); every finding is resolved by Rev 2 (mapped
in §1.16), including an EXECUTED A5 generator-class verification
(CA2-M1) that confirms Rev 1's existing numbers were already computed
from the class-correct generator — no recalibration required. Rev 2 was
attacked a third time (§1.17: NEEDS-REVISION, narrow — 1 MAJOR + 5 minor);
every finding is resolved by Rev 3 (mapped in §1.18), the headline
fix being an EXECUTED power simulation (CA3-M1) that unconditionally
bumps the marquee S4/A5 comparison to n=5 seeds and replaces bare
CI-overlap with a pre-registered TOST equivalence test — verified to
clear a 70% power bar against a real 1.0-rank-unit gap across the entire
reconstructed noise grid, no further escalation required. Rev 3 was
attacked a fourth time (§1.19: NEEDS-REVISION, narrow — 1 MAJOR,
one-paragraph fix); the finding is resolved by this Rev 4 (mapped in
§1.20) — the escalation triggers (general escalation-to-5 and the
marquee n=7 trigger) were uncosted and the hard-abort only projected the
BASE 58-cell sweep once upfront, a genuine stop-and-ask gap at literal
worst case (`19.65 + 2.40 + 12.60 = 34.65 GPU-h > the 30 GPU-h cap`);
fixed by wiring a live budget re-check into both escalation-trigger code
paths with a pre-registered yield order (marquee outranks general),
no design-content change. This design came through the full waterfall (brainstorm →
research → attack → validation, `STATE.md` "CAPABILITY CAMPAIGN"); the
hypothesis, group family, readout (Option A), controls, budget, and
pre-registered verdicts below are **BINDING** per the 2026-07-08
validation verdict (BUILD-WITH-CHANGES) and are not relitigated here —
this document's job is to make them buildable without further design
decisions, to the rigor bar `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.13-§1.17 set
(numerically-executed formulas, not asserted ones; a self-attack register
with teeth).

---

### 1.0 Reading list this design builds on (context, not repeated here)

- **`CLAUDE.md`** Hard Rules — the ones with direct teeth here: readout
  must force EXACT CONTINUOUS recovery, never argmax/nearest-neighbor
  (the Nichani shortcut); a hard single-state (`P=1`) bottleneck is
  required for within-representation rank to be load-bearing, verified via
  a gradient-based blank-out test, not a vacuous shape check;
  single-full-K-cycle permutation lesson (periodicity confounds); integer/
  structural correctness checks need EXACT thresholds, not
  floating-point-tolerance slack (the `_assert_injective` `-1` bug); a
  calibration run before a big sweep is mandatory; dead cells need
  2-2.5× budget retests before being called dead (the three-budget-
  artifacts pattern); multiple independent adversarial audit rounds catch
  different bugs each round; hold tokenization/every second axis FIXED
  when testing a primary hypothesis.
- **`STATE.md`** "CAPABILITY CAMPAIGN" section (waterfall record) — Claim B
  as validated: causal rank↔representation-dimension recruitment across a
  group family interleaving solvability at matched dimensions (S4-vs-A5 =
  the marquee dissociation control); the PSL(2,7) real-dim correction
  (dim 6, complex-conjugate 3-dim irrep pair, not real); the required
  changes (readout Option A + Procrustes/scale degauging; Stage 1 =
  Task D/E's exact bespoke architecture; re-derived budget). GOALS item 4
  (2026-07-08): the overall research direction is capability SEPARATION,
  not efficiency — this design is that program's second concrete test
  after the head-to-head demo.
- **`matrix-thinking/chapter2/TASK_D_PREREGISTRATION.md`** — the exact
  architecture, hard-bottleneck machinery, and rank-necessity proof
  pattern this design reuses. Read in full for this design; every
  reused mechanism below cites exact section/function names, not a
  paraphrase.
- **`matrix-thinking/chapter2/TASK_E_FINDINGS.md`** — the
  subspace-restriction methodology (§9, `analyze_zdump.py`) this design
  generalizes, and the rank-inflation trap (§4) it exists to avoid
  repeating: whole-matrix rank is trivially uninformative once an
  encoder has ambient capacity beyond the task-relevant subspace.
- **`matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md`** §1.13-§1.17 — the
  gauntlet-rigor bar this design is held to: attack rounds now
  numerically EXECUTE design formulas (Rev 2→3's cross-dimension
  diagnostic was caught FATAL by numerical execution, not review) rather
  than accepting them asserted. §1.3-1.9's self-attack register,
  §1.6-1.7's cost/gate structure, and §1.11-1.12's sequencing/
  reproducibility conventions are the template this document follows.
- **`matrix-thinking/capability_separation/verify_option_a_readout.py`**
  — this design's own executed verification (§1.3 below): the five
  reference representations `rho_G` and the Procrustes/scale degauging
  pipeline, both run on-machine (numpy, no GPU) before this doc was
  written, not asserted.

---

### 1.1 The hypothesis, the honest framing, and pre-registered outcomes

**One-sentence hypothesis:** a from-scratch matrix-state model
(`matrix-thinking/chapter2/model_v4.py`'s `BindingEncoder`/
`MatrixMemoryModel`, extended per §1.4 below), trained on
group-element-recovery tasks under the SAME hard single-state (`P=1`)
bottleneck Task D/E use, with Option A's fixed-reference readout, recruits
subspace-restricted state rank equal to the group's minimal faithful REAL
representation dimension `d_min(G)` — and force-ranking below `d_min(G)`
collapses accuracy at exactly `d_min(G)` per group, tracking
**dimension**, not **solvability** (the S4-vs-A5 dissociation: both
`d_min=3`, S4 solvable, A5 non-solvable — they must land together).

This is a **demonstration-or-bound wave**, mirroring Task D/E's own
convention (`TASK_D_PREREGISTRATION.md` §1: "Both outcomes are decisive
and publishable"): every outcome below is pre-registered as reportable
BEFORE any GPU cell launches.

| Outcome | Trigger | What it means |
|---|---|---|
| **CONFIRM** | Restricted rank tracks `d_min(G)` (Spearman ρ≥0.8 across the 5-group family, §1.5 M1) AND force-rank below `d_min(G)` collapses while force-rank at/above `d_min(G)` reaches ≥0.9×ceiling (§1.5 M3) AND S4/A5 land TOGETHER (dimension) not apart (solvability) | This project's second causal rank-necessity confirmation (after Task D/E), now on a genuinely different necessity argument (representation-theoretic minimality, §1.3 below, not Task D's stacked-independent-values argument) and a genuinely different task family (group word-composition, not parallel key-value binding). Gates Stage 2 (multi-hop word problems, §1.11). |
| **FALSIFY** | Flat rank (recruitment does not track `d_min(G)`) OR a rank-deficient force-rank cell already reaches ceiling (rank isn't actually binding) | Publishable negative: the provable-necessity ↔ SGD-recruitment correspondence, robustly confirmed in Task D's parallel-binding setting and Task E's iterated-composition setting, does NOT survive the shift to "realize a faithful group representation via compositional word recovery." Framed as an honest boundary case of when the correspondence holds, not a refutation of Task D/E. |
| **INCONCLUSIVE** | Mixed across the family (some groups confirm, others don't) OR S4/A5 DO diverge by solvability rather than landing together | Diagnose before scaling (leak? optimization wall? degauging artifact?) — mirrors Task D/E's own "diagnose, don't scale past an unexplained result" discipline. |

**What this does NOT show (scope, stated up front, Task D §9's convention):**
this is associative/compositional word recovery, not multi-hop reasoning
over held-out relational structure the way Task E's `Zʰ` iteration was —
Stage 2 (§1.11) is the reasoning-transfer step, gated on this Stage 1
result, not assumed by it.

---

### 1.2 The group family — reference representations (Option A pin)

**Family, dimensions, solvability (settled by the validation verdict —
representation-theoretic minimal REAL faithful dimensions, independently
confirmed against DeltaProduct's own published `n_h` numbers per
`STATE.md`'s waterfall record):**

| Group | `d_min(G)` | Solvable? | Realization used | `|G|` |
|---|---|---|---|---|
| S3 | 2 | Yes | 2-dim standard rep (D3, symmetries of the equilateral triangle) | 6 |
| S4 | 3 | Yes | 3-dim cube-rotation rep | 24 |
| A5 | 3 | **No** | 3-dim icosahedral rotation rep | 60 |
| S5 | 4 | No | 4-dim standard rep (zero-sum hyperplane in ℝ⁵) | 120 |
| A6 | 5 | No | 5-dim standard/deleted-permutation rep (zero-sum hyperplane in ℝ⁶) | 360 |

S4 (`d_min=3`, solvable) vs A5 (`d_min=3`, non-solvable) is the **marquee
dissociation control**: if rank recruitment tracks `d_min(G)` and not
solvability, these two land together; STATE.md's kill of the original
solvability-indexed hypothesis (candidate #1, "KILLED AS FRAMED") is
exactly this observation made BEFORE any GPU ran, now built into Stage 1's
own pre-registered CONFIRM condition rather than left as an assumption.
**The family's `d_min=2` rung is solvable-only** — no non-solvable group
has a 2-dimensional real faithful representation (the smallest
non-solvable group, A5, already needs 3) — so there is no dim-2
dissociation pair; disclosed, not fixable within this family.

**PSL(2,7), real dim 6 — DROPPED, not included in Stage 1.** Reasoning
(the validation verdict's "principled slot or drop" instruction):
(1) **Construction risk.** PSL(2,7)'s minimal faithful REAL representation
requires combining a complex-conjugate PAIR of 3-dimensional irreps
(PSL(2,7) has no 3-dimensional irrep of real type — Frobenius-Schur
indicator complex, not real) into one genuine 6-real-dimensional
irreducible representation. This is a substantially more error-prone
construction than any of the five reps above — all five of which reduce
to either (a) a rotation group realized via Rodrigues' formula with
axes derived from an explicit, checkable polytope (cube, icosahedron), or
(b) the standard/deleted-permutation representation of `S_n`/`A_n`
(zero-sum hyperplane, exact enumeration of all group elements, no
generator-axis geometry to get right at all) — and I was able to
numerically construct and verify all five of those cleanly in this
session (§1.3 below); PSL(2,7)'s complex-irrep-pairing construction was
not attempted, and getting it wrong would silently invalidate one cell
without being caught until deep into analysis. (2) **Marginal value.**
PSL(2,7) at dim 6 would extend the ladder's ceiling above A6 (dim 5) but
does not fill a currently-missing solvable/non-solvable-matched-dimension
slot — the family's ONE dissociation pair (S4/A5 at dim 3) is already
present; PSL(2,7) would be a standalone 6th rung, not a second
dissociation control. (3) **Budget.** Dropping it also lets Stage 1 fit
comfortably under its dedicated 30 GPU-h ceiling (§1.6) at 5 groups rather
than 6. **Correction to the binding content's own cost shorthand:** the
validation verdict's cost line references "~6 groups"; this design uses
**5** (S3/S4/A5/S5/A6), with PSL(2,7) registered as a Stage-1.5/Stage-2
follow-on if the 5-group ladder CONFIRMs and a higher-dimension rung is
later wanted, not silently absorbed into "6."

**Z_3 vs S3 for the dim-2 rung:** the validation verdict permits either.
S3 is used here (not Z_3) because it continues the Sym/Alt-group ladder
narrative cleanly (S3→S4→A5→S5→A6) and gives a non-abelian dim-2 case,
which is a strictly harder test of order-sensitive word composition
(§1.4) than an abelian group would be.

---

### 1.3 Executed verification — reference representations + the
Procrustes/scale degauging pipeline

**Script (repo-committed, reproducible):**
`matrix-thinking/capability_separation/verify_option_a_readout.py`
(numpy + stdlib only, no torch/GPU, ~1s wall-clock, deterministic given
`RNG_SEED=20260709`). Run: `python3 verify_option_a_readout.py`. This is
NOT a toy-only exercise kept separate from the real pipeline — Part 1's
group-construction code and Part 2's degauging-recovery code are the
literal functions Stage 1's eval harness will import (`bfs_closure`,
`standard_rep_group`, `fit_scale`, `fit_orthogonal_intertwiner`,
`score_eval`), not a throwaway simulation with different logic.

#### 1.3.1 Part 1 — the five reference representations, constructed and verified

Method: S3/S4 built via `bfs_closure()` of two rotation-matrix generators
(Rodrigues' formula, `rot_axis_angle()`); A5 built the same way, with
generator axes derived directly from the icosahedron's 12 vertices
(`(0,±1,±φ)` and even permutations, φ = golden ratio) — a 5-fold axis
through a vertex and a 3-fold axis through an adjacent face's centroid
(found from the vertex's 5 nearest neighbors, keeping a pair that are
themselves mutually adjacent — i.e. a genuine triangular face), with an
explicit `vertex_set_preserved()` check that each generator is a real
icosahedron symmetry BEFORE closing the group; S5/A6 built via full
enumeration of `itertools.permutations` restricted to the zero-sum
hyperplane (`standard_rep_group()`/`hyperplane_basis()`), which exercises
every group element directly rather than closing from generators.

**Executed output (verbatim, this session):**

```
[S3] dim=2  |group|=6 (expect 6): OK  |  all orthogonal: True  |  all det=+-1: True  |  PASS=True
[S4] dim=3  |group|=24 (expect 24): OK  |  all orthogonal: True  |  all det=+-1: True  |  PASS=True
[A5] dim=3  |group|=60 (expect 60): OK  |  all orthogonal: True  |  all det=+-1: True  |  PASS=True
  A5 simplicity argument: A5 is simple (only normal subgroups {e}, A5); image order == |A5| == 60
  forces kernel = {e} => the representation is FAITHFUL as a consequence of group simplicity.
[S5] dim=4  |group|=120 (expect 120): OK  |  all orthogonal: True  |  all det=+-1: True  |  PASS=True
[A6] dim=5  |group|=360 (expect 360): OK  |  all orthogonal: True  |  all det=+-1: True  |  PASS=True
```

All five: correct order, exactly orthogonal (`M @ M.T == I` to float
tolerance), `det=±1` (proper representations of the abstract group's
elements). S4/S3/S5/A6's faithfulness follows directly from exact
enumeration matching `|G|` (every element distinct); A5's follows from
the simplicity argument stated and checked above (order-60 image of a
simple group of order 60 forces trivial kernel) — a structural argument,
not a spot-check.

#### 1.3.2 Part 2 — the Procrustes/scale degauging pipeline, executed

**The gauge freedom being tested.** The model may implement `rho_G` up to
a SINGLE orthogonal change of basis + scale: `g ↦ c·Q·rho_G(g)·Qᵀ` for one
fixed `(Q, c)` shared across every `g` (this is exactly representation
equivalence). This is NOT a plain single-sided vector Procrustes fit —
`Q` appears on BOTH sides via conjugation.

**Recovery method (Schur's-lemma intertwiner recovery, not textbook
Procrustes):**
1. **Scale.** `c_hat` = median over the FITTING set of
   `‖Z(g)‖_F / ‖rho(g)‖_F` (orthogonal conjugation preserves Frobenius
   norm exactly, so this ratio is `|c|` up to noise — this design's exact
   analog of Task E §9's fitted isotropic scale `c*`).
2. **Orthogonal part.** `Z(g)·Q − c_hat·Q·rho(g) = 0` is LINEAR in `Q`.
   Vectorized via `vec(AXB) = (Bᵀ ⊗ A) vec(X)`:
   `[(I⊗Z(g)) − c_hat·(rho(g)ᵀ⊗I)] · vec(Q) = 0`, stacked over the
   fitting set. `Q_raw` = the right-singular vector of the stack's
   smallest singular value, reshaped; `Q_hat` = the nearest orthogonal
   matrix to `Q_raw` (polar decomposition, `SVD(Q_raw) → U·Vᵀ`).
3. **Score on a DISJOINT eval set** — the fit/eval split is the mandatory
   guard: `(Q_hat, c_hat)` is fit ONLY from fitting-set elements; recovery
   quality is judged ONLY on eval-set elements never used to compute
   either. **This is the single biggest registered risk in this design**
   (see §1.9 item 1) — a Procrustes fit evaluated on its OWN fitting set
   could trivially "rescue" an unfaithful solution by curve-fitting; the
   disjoint eval set is what prevents that.

**Why a rank-deficient corruption CANNOT be rescued (a provable
guarantee, verified below, not just claimed):** rank is an invariant of
orthogonal conjugation — `rank(Q·A·Qᵀ) = rank(A)` for any orthogonal `Q`.
So if a "model" is genuinely rank-deficient (fails to recruit full rank on
the entity subspace — exactly the Stage-1 FALSIFY condition), NO `(Q, c)`
can degauge it back onto a full-rank reference. The negative control below
builds a corrupted family by conjugating with a RANK-DEFICIENT `Q_def`
(one row of the true `Q` zeroed, not re-orthogonalized) instead of the
true orthogonal `Q`, and runs the identical fit/score pipeline on it.

**Executed output (verbatim, this session, S4's 24-element group, 14
fit / 10 eval elements, disjoint):**

```
[setup] |S4| = 24 elements; fit set = 14, eval set = 10, disjoint = True
[ground truth] c_true=1.7, Q_true orthogonal (||QQ^T-I||=5.40e-16), det(Q_true)=1.0000

=== A: exact conjugation, zero noise (sanity check) ===
  EVAL SET (n=10, never used to fit c_hat/Q_hat): mean_cos=1.000000  mean_rel_err=0.000000  recovered_frac@0.9=1.0000

=== B: exact conjugation + noise(std=0.03) ===
  EVAL SET (n=10, never used to fit c_hat/Q_hat): mean_cos=0.999617  mean_rel_err=0.028647  recovered_frac@0.9=1.0000

[negative control] rank(Qdef)=2 (of 3) -- Q_true's last row zeroed
[negative control] rank(Z_corrupt(g)) on eval set (pre-noise): [2] (expect <= 2, never 3)

=== C: rank-deficient corruption (NEGATIVE CONTROL -- must NOT be rescued) ===
  EVAL SET (n=10, never used to fit c_hat/Q_hat): mean_cos=0.616102  mean_rel_err=0.923411  recovered_frac@0.9=0.0000
```

**Summary table (eval-set metrics only; fit-set never scored):**

| condition | mean_cos | rel_err | rec@0.9 | intertwiner sv gap |
|---|---|---|---|---|
| A: exact conjugation, zero noise | 1.0000 | 0.0000 | 1.0000 | 7.7675 |
| B: exact conjugation + noise(std=0.03) | 0.9996 | 0.0286 | 1.0000 | 7.5839 |
| C: rank-deficient corruption (negative control) | 0.6161 | 0.9234 | 0.0000 | 1.4850 |

All 5 pre-registered assertions in the script pass (exact case
`mean_cos>0.9999`; noisy case `mean_cos>0.95` and `recovered_frac@0.9≥0.9`;
corrupted case `mean_cos` at least 0.3 below the honest noisy case's, and
`recovered_frac@0.9<0.5`). **The degauging pipeline correctly recovers the
true gauge under realistic noise, and provably cannot rescue a
rank-deficient corruption on the held-out eval set — the fit/eval split
has teeth, verified, not asserted.** The intertwiner singular-value gap
(7.58-7.77 for the honest conditions vs 1.49 for the corrupted one) is a
useful secondary diagnostic: a genuine intertwiner produces a
well-separated near-null direction in the linear system; the corrupted
case does not, because no exact intertwiner exists.

---

### 1.3.3 Executed verification (Rev 1) — query-coverage calibration
(CA1-F1 fix) and the M1 Spearman null (CA1-M1 fix)

**Why this section exists.** Attack round 1 (§1.13) found Rev 0's
query-coverage bar mathematically impossible for 2 of 5 groups (CA1-F1)
and Rev 0's M1 Spearman bar statistically undisclosed (CA1-M1). Both fixes
are numerically executed here, not asserted — the same discipline §1.3
already applies to the reference representations and degauging pipeline.

**Script 1 (repo-committed):**
`matrix-thinking/capability_separation/coverage_calibration.py` (numpy +
stdlib only, deterministic `RNG_SEED=20260710`, ~15s wall-clock). Verifies
each group's PERMUTATION stand-in generating set (order-matched to §1.4's
table — a 3-cycle+transposition for S3, a 4-cycle+3-cycle for S4, a
5-cycle+3-cycle for A5, a transposition+5-cycle for S5, a 3-cycle+5-cycle
sharing one point for A6) closes to the correct `|G|` via `bfs_closure`
(the same closure discipline §1.3.1 uses on the matrix realizations), then
Monte Carlo-simulates the ACTUAL held-out sampler (`N=50` words/trial,
`L~Uniform{9,16}`, i.i.d. generator draws — §1.4's exact rule) against a
pathological UNDERSAMPLED negative control (`L` fixed at `1`, a
DETERMINISTIC ceiling of `|generating set|` distinct elements reachable,
not merely a statistical tail — a plausible real bug, e.g. the held-out
length sampler silently defaulting to the trivial case).

**Executed output (verbatim, this session):**

```
========================================================================================
STEP 0 -- verify each generating set's closure matches |G| (faithfulness
of the permutation stand-in, same discipline as S1.3's bfs_closure check)
========================================================================================
  [S3] closure size = 6  (expect 6): PASS  |  gen set size = 3
  [S4] closure size = 24  (expect 24): PASS  |  gen set size = 4
  [A5] closure size = 60  (expect 60): PASS  |  gen set size = 4
  [S5] closure size = 120  (expect 120): PASS  |  gen set size = 3
  [A6] closure size = 360  (expect 360): PASS  |  gen set size = 4

========================================================================================
STEP 1 -- Monte Carlo: N=50 words/trial, 20000 trials/group/sampler, L~Uniform{9..16}
========================================================================================
Group    |G| d_min   healthy: mean     p1     p5   min  |   undersamp: mean   max
---------------------------------------------------------------------------------
S3         6     2        6.00 (100.0%)      6      6     5  |          3.00 (50.0%)     3
S4        24     3       21.12 (88.0%)     18     19    15  |          4.00 (16.7%)     4
A5        60     3       33.78 (56.3%)     28     30    25  |          4.00 ( 6.7%)     4
S5       120     4       37.41 (31.2%)     31     33    27  |          3.00 ( 2.5%)     3
A6       360     5       44.90 (12.5%)     40     41    35  |          4.00 ( 1.1%)     4

========================================================================================
STEP 2 -- calibrated bar selection: largest 5%-of-|G| bar such that
(a) it clears the healthy-sampler 1st-percentile floor (>=99% of
healthy N=50 draws pass) AND (b) it strictly exceeds the
undersampled (L=1 pathological) ceiling -- note undersamp max is a
DETERMINISTIC hard cap at L=1 (== |gens|, never exceedable by
construction, not just a statistical tail), so bar>u_max alone
already guarantees a 100%, not just high-probability, fail rate
for that corruption mode -- both conditions numerically checked.
========================================================================================
Group    |G|  healthy p1   undersamp max  bar (frac)  bar (count)  p1 margin  undersamp mult
--------------------------------------------------------------------------------------------
S3         6           6               3        0.80          4.8        1.2            1.60
S4        24          18               4        0.70         16.8        1.2            4.20
A5        60          28               4        0.45         27.0        1.0            6.75
S5       120          31               3        0.25         30.0        1.0           10.00
A6       360          40               4        0.10         36.0        4.0            9.00

Bars (fraction of |G|): {'S3': 0.8, 'S4': 0.7, 'A5': 0.45, 'S5': 0.25, 'A6': 0.1}

========================================================================================
STEP 3 -- CA1-M3 fit-set diversity floor: n_fit=30 words/trial, 20000 trials/group, bar = min(3*d_min(G), floor(0.8*|G|))
========================================================================================
Group    |G| d_min  bar=min(3d,.8|G|)   fit p1  fit mean   margin
-----------------------------------------------------------------
S3         6     2                  4        5      5.98        1  PASS
S4        24     3                  9       14     17.29        5  PASS
A5        60     3                  9       19     23.62       10  PASS
S5       120     4                 12       21     25.05        9  PASS
A6       360     5                 15       25     28.12       10  PASS
```

**Non-integer bar clarifying note (CA3-m5 fix, Rev 3).** STEP 2's "bar
(count)" column above shows the RAW fractional threshold
(`frac × |G|`, e.g. S3's `0.80×6=4.8`) — since a realized coverage draw
is always an integer count of distinct group elements, the check
actually compares against `⌈frac × |G|⌉` (the ceiling), not the raw
fractional value shown in the table. §1.4's bar table already reports
the correct integer thresholds (S3 `≥5`, S4 `≥17`, A5 `≥27`, S5 `≥30`,
A6 `≥36` — each the ceiling of its STEP-2 row above, e.g.
`⌈4.8⌉=5`, `⌈16.8⌉=17`); this note exists solely to prevent a reader from
treating STEP 2's raw fractional column as the operative pass/fail cutoff.

**Deviation from the naive "80% for S3/S4/A5" default, disclosed
explicitly.** The recommended family-consistent starting point (≥80% of
`|G|` for the three smaller groups) does NOT survive execution for S4
(healthy p1 = 18/24 = 75% < 80%) or A5 (healthy p1 = 28/60 = 46.7%, far
below 80%) at the pinned `N=50` floor — adopting it uncorrected would have
reproduced CA1-F1's own failure mode (a bar a healthy sampler cannot
reliably clear) one level down, on groups where it happens to look
plausible at a glance. Each group's bar above is instead the largest
5%-of-`|G|` step that (a) clears the healthy sampler's 1st-percentile
floor by ≥1 element and (b) strictly exceeds the undersampled sampler's
DETERMINISTIC ceiling — a per-group calibration, not a single inherited
percentage. S3 is the only group where 80% survives on its own merits.

**Script 2 (repo-committed):**
`matrix-thinking/capability_separation/spearman_null_calibration.py`
(numpy-free, stdlib only, exact enumeration of all `5!=120` permutations,
deterministic). Independently re-derives the M1 null distribution from
scratch this Rev-1 pass (not copied from §1.13's own numbers).

**Executed output (verbatim, this session):**

```
================================================================================
CA1-M1 fix -- exact permutation null for the M1 Spearman bar
================================================================================
d_min(G) sequence (S3,S4,A5,S5,A6) = [2, 3, 3, 4, 5]
midranks (S4/A5 tie at value 3)     = [1.0, 2.5, 2.5, 4.0, 5.0]

total permutations enumerated = 120 (5! = 120)
max achievable rho (perfect ordering respecting the S4/A5 tie) = 0.9747
next-highest achievable rho (one non-tie misordering)          = 0.8721  (cliff size = 0.1026)

P(rho >= 0.8 | null) = 8/120 = 6.6667%
P(rho >= 0.9 | null) = 2/120 = 1.6667%

top 4 distinct achievable rho values: [0.9747, 0.8721, 0.8208, 0.7182]

ASSERTIONS PASSED -- independently reproduces the attack round 1 figures (S1.13 CA1-M1: 8/120~=6.67%, tie cap 0.9747) via fresh exhaustive enumeration.
```

**This independently reproduces §1.13's own CA1-M1 figures exactly**
(`8/120≈6.67%`, tie cap `0.9747`) via a from-scratch exhaustive
enumeration, not a copy — see §1.5 for how these numbers are used in M1's
bar decision, and §1.4/§1.4.1 for how the coverage numbers above are used
in the query-coverage bar and the degauging fit-set split.

---

### 1.3.4 Executed verification (Rev 2) — A5 order-5 generator CLASS
verification (CA2-M1 fix)

**Why this section exists.** Attack round 2 (§1.15, CA2-M1) found that
`coverage_calibration.py`'s A5 permutation stand-in never verified its
order-5 generator `g5=(1,2,3,4,0)` corresponds to the SAME A5 conjugacy
class as `verify_option_a_readout.py`'s real icosahedral `2π/5` rotation
generator `g5_a5` — A5's order-5 elements split into two non-conjugate
classes (5A/5B, the same fact behind A5 having two inequivalent 3-dim real
irreps), and the round's own empirical probe showed the choice is NOT
cosmetic: swapping to the other class drops mean coverage 33.79→31.76 and
the 1st-percentile floor 28→26 — **below** the pinned bar of 27 (a 16×
false-block rate increase). The round explicitly flagged its own
parenthetical guess at the two classes' trace values (`φ-1` vs `-φ`) as
**unverified** and instructed: derive the correct values from the actual
rep matrices, and use either an explicit A5-isomorphism or a
cycle-structure/character comparison — not a repeat of the same
unverified-guess mistake one level down.

**The correct mechanism, settled by attack round 3 (§1.17's "A5
mathematical story"), is the JOINT GENERATING PAIR, not conjugacy class in
isolation.** Round 3 checked, by brute force over all `120` `τ∈S5`
(A5's outer automorphism group `Out(A5)=Z/2` acts by conjugation by an odd
permutation), whether any automorphism maps the generating PAIR
`{g5,g3}` to `{g5²,g3}`: every `τ` with `τg5τ⁻¹=g5²` does exist (the
5A/5B swap is real) but **NONE of those same `τ` also fixes `g3` up to
inversion** — so no automorphism carries one generating pair to the other.
The empirical coverage difference (33.79 vs 31.76 mean, above) is
therefore a genuine **pair-level Cayley-graph difference** (the two
generating pairs produce non-isomorphic Cayley graphs with different
mixing statistics), not merely "wrong conjugacy class" read in isolation
— the isomorphism check below verifies the right thing (the actual
generator-matched pair used by the random-walk sampler), and this
framing sentence names the mechanism precisely rather than leaving
"class" as a loose stand-in for "pair."

**Method — explicit generator-matched group isomorphism, exhaustively
checked, not a trusted character-table lookup.** `verify_a5_generator_class()`
(new function, `coverage_calibration.py`, imports
`verify_option_a_readout.py`'s own A5 generator construction directly — no
re-implemented geometry, no drift risk) runs a **simultaneous BFS closure**
of BOTH the permutation group (`coverage_calibration.py`'s `GROUPS["A5"]`)
and the matrix group (`verify_option_a_readout.py`'s `g5_a5`/`g3_a5`),
multiplying by GENERATOR-MATCHED pairs (`perm_gens[i] ↔ matrix_gens[i]`) at
every step, both left- and right-multiplied (mirroring exactly how each
script's own `bfs_closure` already builds its group). The map
permutation→matrix is checked to be a **well-defined function** at every
one of the 60 elements — the instant the SAME permutation is reached via
two different generator-words that assign it TWO DIFFERENT matrices, the
candidate correspondence is refuted. `consistent AND both element counts
== 60` is a constructive, EXHAUSTIVE proof of a genuine group isomorphism
(every element checked, not sampled) — strictly stronger evidence than a
character/trace match alone, which only constrains one invariant.

**Character cross-check, executed first (not trusted from the
parenthetical).** Direct computation from the real rep matrices:
`trace(g5_a5) = 1.618034 = φ` and `trace(g5_a5²) = -0.618034 = 1-φ`. **This
does NOT match attack round 2's own parenthetical guess** (`φ-1≈0.618` vs
`-φ≈-1.618`) — the guess was checked against the actual matrices and found
wrong in detail (though correctly flagged as needing verification rather
than being trusted), confirming the instruction to verify rather than
inherit the parenthetical was warranted.

**Executed output (verbatim, this session,
`python3 coverage_calibration.py`):**

```
========================================================================================
CA2-M1 -- A5 order-5 generator CLASS verification
========================================================================================
  order(g5_a5) = 5   order(g5_a5^2) = 5   order(g3_a5) = 3
  trace(g5_a5)   = 1.618034   (== phi = 1.618034? True)
  trace(g5_a5^2) = -0.618034   (== 1-phi = -0.618034? True)
  [note] the attack's own parenthetical guess ('traces phi-1=0.6180 vs -phi=-1.6180') does NOT match either computed trace above -- computed directly from the rep matrices, not trusted from the parenthetical, per the CA2-M1 instruction.

  Testing candidate generator-matched isomorphisms (full 60-element
  simultaneous BFS closure, well-definedness checked at every step):
    [FAIL] g5 <-> g5_a5,    g3 <-> g3_a5   (naive as-labeled pairing)  (perm elems=60, matrix elems=46)
    [PASS] g5 <-> g5_a5,    g3 <-> g3_a5^-1 (inverse-labeled g3; SAME symmetric set)  (perm elems=60, matrix elems=60)
    [FAIL] g5 <-> g5_a5^2,  g3 <-> g3_a5   (wrong-class control)  (perm elems=60, matrix elems=33)
    [FAIL] g5 <-> g5_a5^2,  g3 <-> g3_a5^-1 (wrong-class control, inverse-labeled)  (perm elems=60, matrix elems=31)

  VERDICT: coverage_calibration.py's CURRENT stand-in (g5, unmodified) IS
  the CORRECT class -- ...
```

**Reading the result.** Exactly ONE of the four candidates closes to the
full 60-element group with zero inconsistencies: `g5 ↔ g5_a5` (the
generator AS ALREADY WRITTEN, unmodified) paired with `g3 ↔ g3_a5⁻¹` — the
inverse-labeled match is immaterial to the actual random-walk sampler,
because the SYMMETRIC generating set `{g5,g5⁻¹,g3,g3⁻¹}` already contains
both `g3_a5` and `g3_a5⁻¹` regardless of which one is internally called
"primary" (the sampler draws from the full symmetric set, never from `g3`
alone). Both `g5_a5²` (wrong-class) controls FAIL to close to the full
group under either labeling of `g3` (46, 33, and 31 elements respectively,
all `<60`) — cleanly separating the correct class from the incorrect one;
this is not a near-miss, the wrong-class candidates do not even form a
valid homomorphism, let alone an isomorphism.

**Verdict: the current stand-in was ALREADY the right class.**
`coverage_calibration.py`'s `g5` generator, as written before this
revision, is exhaustively verified (all 60 elements, not sampled) to be
A5-class-conjugate to the real icosahedral generator it stands in for.
**No re-calibration is required — §1.3.3's existing numbers stand exactly
as computed**, now with executed evidence (rather than an unverified
assumption) that the permutation stand-in used to compute them was the
class-correct one. A full re-run of `coverage_calibration.py` (STEP 0-3,
this session, immediately following the class verification above)
reproduces §1.3.3's table BYTE-IDENTICAL, as expected since the generator
itself is unchanged — see §1.3.3 for the reproduced STEP 0-3 output
(unchanged) and §1.4.1 below (§1.4.1 step 4) for the new STEP 4 (CA2-m2
fix, eval-set diversity floor) added in this same run.

**Script (repo-committed, reproducible):**
`matrix-thinking/capability_separation/coverage_calibration.py`
(`verify_a5_generator_class`, `_simultaneous_closure`,
`_get_real_a5_generators` — new this revision; imports
`verify_option_a_readout.part1_reference_representations` directly rather
than re-deriving the icosahedron geometry, avoiding drift between the two
scripts). Run: `python3 coverage_calibration.py` (runs the class
verification first, then STEP 0-4).

---

### 1.4 The task, architecture, and readout

**Task — group-element-recovery via word composition (the Barrington-style
word problem, not Task D's parallel key-value binding).** Per sample: a
WORD `w = g_{i1} g_{i2} ... g_{iL}`, each `i_t` drawn i.i.d. uniform from a
SYMMETRIC generating set `S_G = S_G⁻¹` (size 3-4, table below — a genuine
bidirectional random walk on `G`'s Cayley graph, standard practice,
avoiding a "words only grow in one direction" degeneracy). Target =
`rho_G(g_{i1} · g_{i2} · ... · g_{iL})`, computed exactly by multiplying
the pinned reference generator matrices (well-defined for any word,
`§1.3`'s verified matrices). Word length `L` varies per sample:
`L ~ Uniform{1,...,L_train_max=8}` at train time;
`L ~ Uniform{9,...,L_test_max=16}` for the HELD-OUT generalization split
(Task D's C5 "longer sequences than trained," directly ported).

**Why the periodicity-confound lesson does NOT re-apply here (checked, not
assumed).** `CLAUDE.md`'s Hamiltonian-K-cycle lesson (Task E's periodicity
bug: a fixed permutation self-applied `h` times is periodic in `h mod
cycle_length`, silently collapsing "held-out" hops) applies to REPEATED
self-application of a SINGLE fixed operator (`Zʰ`). This task has no such
structure: each of the `L` tokens is an INDEPENDENTLY drawn generator, not
one operator applied `L` times, so there is no fixed period for word
length to collide with. By the standard theory of random walks on finite
groups, the distribution over the resulting element approaches uniform as
`L` grows (mixing) with no periodic collapse. **Query-coverage
pre-registration (the concrete, checkable bar — CA1-F1 fix, Rev 1:
replaces Rev 0's mathematically-impossible bar, "≥200 distinct elements
for A5/S5/A6," which contradicted the group table above — `|A5|=60`,
`|S5|=120`, a group cannot yield more distinct elements than it has, and
gate 3's coverage check could never pass for A5, the marquee dissociation
partner, or S5. The replacement is a per-group, NUMERICALLY CALIBRATED
percentage-of-`|G|` bar, executed via Monte Carlo simulation of the actual
sampler — full derivation, deviation disclosure, and raw output in
§1.3.3):** for each group's eval-word set (§1.4.1, `N=50` words), (a) the
set of DISTINCT resulting group elements reached must clear the
group-specific bar below —

| Group | `|G|` | calibrated bar | bar (count) | healthy-sampler p1 | undersampled (L=1) ceiling |
|---|---|---|---|---|---|
| S3 | 6 | ≥80% of `|G|` | ≥5 | 6 | 3 |
| S4 | 24 | ≥70% of `|G|` | ≥17 | 18 | 4 |
| A5 | 60 | ≥45% of `|G|` | ≥27 | 28 | 4 |
| S5 | 120 | ≥25% of `|G|` | ≥30 | 31 | 3 |
| A6 | 360 | ≥10% of `|G|` | ≥36 | 40 | 4 |

every bar clears the healthy sampler's 1st-percentile floor with a ≥1-
element margin and strictly exceeds the undersampled sampler's
DETERMINISTIC ceiling (by ≥1.6×, and by ≥4.2× for S4-A6) — checked
directly by enumerating the eval batch's realized targets; (b) exact
token-sequence
(word) overlap between train and eval is prevented by construction (a
disjoint `rd_episode_seed`-style mixed-radix RNG stream per split,
precedent: `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.3.1's F1b fix,
`reasoning_link_probe.py`'s `episode_seed`) — note this is DIFFERENT from
Task D's continuous-vector freshness guarantee: because `G` is finite, the
same RESULTING group element inevitably recurs across samples (unlike
Task D's near-zero-probability-of-repeat continuous vectors); only exact
WORD (token-sequence) repetition is what must be, and is, prevented.

**Generating sets (symmetric, from §1.3's verified matrices):**

| Group | Generating set | Size |
|---|---|---|
| S3 | `{r, r⁻¹=r², s}` (`r`=120° rotation order 3, `s`=reflection, self-inverse) | 3 |
| S4 | `{r, r⁻¹, s, s⁻¹}` (`r`=90° about z order 4, `s`=120° about (1,1,1) order 3) | 4 |
| A5 | `{g5, g5⁻¹, g3, g3⁻¹}` (icosahedral 5-fold, 3-fold rotations) | 4 |
| S5 | `{t, c, c⁻¹}` (`t`=transposition, self-inverse; `c`=5-cycle order 5) | 3 |
| A6 | `{(123), (123)⁻¹, (23456), (23456)⁻¹}` — standard 3-cycle + 5-cycle generators of A6 | 4 |

A6's generating pair is the standard, well-known one (any 3-cycle plus a
5-cycle sharing exactly one point generates `A_n` for the relevant small
`n`) — not independently re-verified via `bfs_closure` on just these 2
generators in §1.3, because §1.3 already verified the STRONGER claim (the
full 360-element representation, built by exhaustively enumerating every
even permutation directly, not by closing 2 generators); the standard
closure-order smoke on these two specific generators is already
executed, see §1.3.3 (STEP 0: `[A6] closure size = 360 (expect 360):
PASS | gen set size = 4`) — a cheap, load-bearing sanity check, not a
build-time TODO (minor fix, Rev 4, §1.20: STEP 0's executed closure
check subsumes the earlier "will still run before launch" phrasing).

**Architecture (Task D/E's exact bespoke encoder, THREE required
extensions — CA1-m1 fix, Rev 1: Rev 0 undercounted this as "ONE"; CA2-M2
fix, Rev 2: pins the third, the variable-length batching scheme, as a
disclosed extension rather than leaving it for the builder to invent —
see below).** Reuses `matrix-thinking/chapter2/model_v4.py`'s
`BindingEncoder` (`d`, `h`, `n_layers`, `n_heads`, `n_refine`
constructor signature, `nn.TransformerEncoder` + learned `row_queries` +
`MultiheadAttention` reader + `row_norm` + `row_out` — same class,
same core module stack, EXTENDED by the three deltas below, not an
unmodified forward pass — CA3-m5 fix, Rev 3: "same forward pattern" (Rev
2's wording) overstated the reuse, since delta (1) inserts a new
positional-embedding addition before the encoder and delta (2) replaces
the input embedding entirely, both changing what the forward pass
actually computes) and `MatrixMemoryModel.encode`'s
`force_rank_k`-via-`rank_utils.truncate_to_rank` path (§1.5 below), with
**one architectural delta, required because group composition is
non-abelian and Task D's `BindingEncoder` is otherwise permutation-
EQUIVARIANT (no positional signal) by construction** (it was correct for
Task D's genuinely order-INDEPENDENT set-of-bindings task, and would
silently break here): a learned absolute positional embedding
(`nn.Embedding(L_max, h)`), added to each generator token's embedding
before the `nn.TransformerEncoder`, tags word position 1..L. **Three
deltas total, all structurally required by the task change, not a
broader redesign (CA1-m1 fix — Rev 0's "ONLY change" sentence undercounted
the second, which was already disclosed two bullets below but never
folded into this count; CA2-M2 fix — Rev 1 disclosed the third's NEED,
§1.9 item 9's word-length discussion, but never pinned its concrete
mechanism, leaving a real methodological choice for the builder to
invent):** (1) this positional embedding (order-sensitivity, above), (2)
the input embedding itself — a single generator-index embedding (below)
replacing Task D's `[key; value]` concatenation, because this task has no
separate value to bind, the WORD is the whole composition, and (3) the
variable-per-sample word-length batching scheme (below — CA2-M2 fix).
`CLAUDE.md`'s "hold every second axis fixed when testing a primary
hypothesis" rule is honored because all three deltas are direct, minimal
consequences of the SAME task-family change (word composition replacing
parallel key-value binding, which is simultaneously non-abelian AND
variable-length, unlike Task D's fixed-`K` parallel bindings), not
independent design choices stacked on top of it.

- **Input:** each token is a generator-index one-hot/embedding
  (`nn.Embedding(|S_G|, h)`), NOT Task D's `[key; value]` concatenation
  (there is no separate value here — the task IS the composition; this is
  delta (2) above).
- **Batching scheme for variable word length (delta (3), CA2-M2 fix,
  Rev 2 — pinned, not left for the builder to invent).**
  `BindingEncoder.forward` (`matrix-thinking/chapter2/model_v4.py`) takes
  fixed-shape `(B, K, h)` tensors and contains no padding or attention-mask
  logic anywhere — Task D always ran a single fixed `K` per launch and
  never exercised variable per-sample sequence length. This task samples
  `L ~ Uniform{1..8}` (train) / `{9..16}` (eval) PER SAMPLE (§1.4 above),
  so a naive per-sample-variable-`L` batch would require the builder to
  invent padding + `key_padding_mask` plumbing — landing directly in
  `CLAUDE.md`'s own hard-rule gotcha class (`nn.MultiheadAttention` in
  this repo's PyTorch version requires explicit `attn_mask` OR
  `is_causal`, not both — untested mask code is exactly the kind of
  silent-footgun surface that rule exists to flag). **Pinned scheme:
  PER-BATCH-FIXED-`L`** — each batch samples ONE `L` from the pinned
  distribution (`Uniform{1..8}` train / `{9..16}` eval); every episode
  within that batch shares it; `L` varies ACROSS batches, not within one.
  **Justification vs. padding+mask, decided explicitly (not defaulted
  to):** per-batch-fixed-`L` needs ZERO new padding/mask code — every
  batch is a plain fixed-shape `(B, L, h)` tensor, `BindingEncoder`'s
  existing forward pass is reused completely unmodified (no
  `key_padding_mask` argument threaded through at all), eliminating the
  mask-gotcha risk outright rather than requiring it to be gotten right.
  Padding+mask would instead add real, currently-untested plumbing to
  `BindingEncoder` (a class this design otherwise reuses byte-for-byte)
  for a benefit that does not apply here: padding's usual motivation is
  packing MULTIPLE distinct sequence lengths into the SAME batch for
  throughput; this design's `L` range is small (`1..16`) and its
  step-budget cost is already dominated by the 58-cell sweep structure
  (§1.4.2, Rev 3's cell count; §1.6), not by batch-packing efficiency.
  **Consequence, stated
  explicitly:** per-batch `L`-homogeneity is a real deviation from
  literal i.i.d.-per-sample `L` sampling — but it is BENIGN for this
  task specifically, because episodes are i.i.d. GIVEN `L` (the word's
  distinct generator draws, and the target group element they compose to,
  do not depend on which OTHER episodes share the same batch), and the
  `L`-distribution is preserved in EXPECTATION across training (every
  batch still draws its shared `L` from the same pinned distribution, so
  across the full training run each `L` value is sampled proportionally
  often — only the WITHIN-batch correlation changes, not the marginal
  distribution any single episode experiences). **Loss-scale note (CA3-m4
  fix, Rev 3 — round 3 adjudicated SAFE, disclosed here in one sentence
  rather than left implicit):** per-batch-fixed-`L` introduces no
  loss-scale variation across batches of different `L`, because the
  training loss (§1.4 below, Task D's cosine-loss primitive) is a SINGLE
  scalar cosine distance per episode, averaged over the batch — there is
  no per-token or per-position accumulation term whose magnitude would
  grow with `L`, unlike e.g. a summed per-step cross-entropy loss would.
  **Smoke item added (build
  gate, §1.7 gate 7's blank-out/pre-training smoke list):** batches at
  `L=1` and `L=16` (the distribution's two extremes) both forward AND
  backward cleanly through `GroupWordEncoder` before any training cell
  launches — a minimal, cheap, direct test of the one legitimately new
  shape-handling surface this extension introduces.
- **`d_state(G) = d_min(G) + 2`** (uniform margin rule across the family —
  enough ambient headroom for the force-rank grid `{d_min−1, d_min,
  d_min+1}` (§1.4.2) to fit inside `[1, d_state]`, and for the
  unconstrained arm to show genuine over-recruitment above `d_min(G)` if
  it happens, rather than being architecturally capped at exactly
  `d_min(G)` and unable to demonstrate a real M1 finding either way).
- **`h=32`** (HALVED from Task D/E's `h=64`) — an economization justified
  by `d_state` being much smaller here (2-7, vs Task D/E's 16) and the
  task's shorter sequence length (`L≤16` vs Task D's `K≤16` two-vector
  bindings or Task E's up-to-21-hop compositions); **explicitly
  calibration-pending** (§1.7 gate 1) — the mandatory calibration cell
  re-derives this before the sweep trusts it, per `CLAUDE.md`'s
  calibration-first rule applied literally to this new economization, not
  assumed safe.
- **`n_layers=3, n_heads=4, n_refine=1`** — unchanged from Task D's
  defaults (`run_task_d.py`'s own CLI defaults), no reason found to alter
  these.
- **Output:** `Z = encoder(word_tokens) ∈ ℝ^{d_state × d_state}`, exactly
  `BindingEncoder.forward`'s return value — no separate query/unbind step
  (unlike Task D's `unbind(Z, query_keys)`; there is nothing to query
  here, the whole word maps to one element).

**Readout — Option A, pinned, no learned readout weights beyond the
encoder itself (the "no generic MLP, no argmax" discipline, Task D §2's
exact reasoning, re-applied):** the target for word `w` is the
BLOCK-EMBEDDED reference `rho_G_embedded(product(w)) = rho_G(product(w))
⊕ I_{d_state − d_min(G)}` (block-diagonal, identity on the ambient
complement — this is a fixed, non-learned embedding choice, resolving the
otherwise-unavoidable dimension mismatch between `Z` (`d_state×d_state`)
and `rho_G(g)` (`d_min(G)×d_min(G)`) without introducing a trainable
readout that could launder rank). **Training loss:** cosine loss between
flattened `Z` and flattened `rho_G_embedded(product(w))` — Task D's exact
loss primitive (`task_d.py::recovery_cosine`-equivalent), applied
directly, no MLP, no argmax. **Scoring metric (post-hoc, on held-out
words):** Option A's Procrustes/scale-degauged cosine (§1.3.2, fit/eval
split mandatory) and `recovered_frac@0.9` (Task D/E convention).

**Blank-out test (mandatory, reused verbatim from `run_task_d.py::smoke()`
step [5]):** identical method — `Z = model.encode(word_tokens)` detached
to a leaf, gradient of a downstream function of `Z` w.r.t. the raw word
tokens is checked to be `None` (no path around `Z`), and
`inspect.signature` on the scoring function is checked to admit only
`Z` (plus the fixed reference target) as inputs. Simplifies slightly
relative to Task D's version (no separate `query_keys` argument to check),
but the exact `torch.autograd.grad`-based method is unchanged — this
project's `CLAUDE.md` hard rule is explicit that this must be a real
gradient-based check, not a code-inspection-only shape check.

**Force-rank:** `rank_utils.truncate_to_rank(Z, k)` applied verbatim
inside `MatrixMemoryModel.encode`'s existing `force_rank_k` branch — same
`eigh`-based, degenerate-spectrum-safe implementation Task D/E already
use and this project already smoke-tested for NaN/Inf-free backward
(`run_task_d.py::smoke()` step [2]).

#### 1.4.1 Subspace-restriction analysis — generalizing `analyze_zdump.py` via the pinned-ρ ideal trajectory

Task E §9's `analyze_zdump.py` derived its "entity subspace" `U` from a
SINGLE fixed target operator `z_ideal`'s own SVD, because Task E's task
had exactly one operator per episode living in the SAME ambient space as
`Z`. Capability-separation has no single fixed target the same way — every
word has a DIFFERENT target — so this design generalizes the precedent
rather than reusing it literally (stated honestly, not overclaimed as a
verbatim reuse):

1. Dump `Z(w)` for a held-out sample of `N≥50` words per group (spanning
   the query-coverage bar, §1.4).
2. Derive the model's OWN dominant `d_min(G)`-dimensional subspace `U`
   (`d_state × d_min(G)`) via SVD of the empirical covariance
   `Σ_w Z(w)·Z(w)ᵀ` over that sample — a DATA-DRIVEN, non-circular
   derivation (from the model's own outputs, not assumed from `rho_G`),
   directly analogous in spirit to `entity_subspace()`'s SVD-of-the-target
   derivation, adapted because there is no single target here.
3. Restrict: `A(w) = Uᵀ·Z(w)·U` (`d_min(G) × d_min(G)`) for every held-out
   `w`.
4. **Fit/eval split, pinned (CA1-M3 fix, Rev 1 — Rev 0 left this
   implicit).** Split the `N=50` held-out words 60/40 by index (disjoint,
   no overlap): `n_fit=30` (fits the Procrustes/scale degauging
   `(Q_hat, c_hat)`, §1.3.2) / `n_eval=20` (scored ONLY, never used to fit
   either parameter — the same fit/eval discipline §1.3.2 already verified
   has teeth, now applied to the real pipeline's exact split ratio).
   **Draw cadence, pinned (CA2-m1 fix, Rev 2 — Rev 1 left "how often is
   this `N=50` sample redrawn" unstated).** **Fresh-per-cell** — every
   sweep cell (58 cells total post-Rev-3, §1.4.2/§1.6) draws its OWN
   `N=50` held-out
   sample from that cell's own seeded RNG stream, the same
   `rd_episode_seed`-style per-cell seeding this design already pins for
   train/eval word disjointness (§1.4), NOT one shared `N=50` sample reused
   across all cells in a group (10 for S3/S5/A6, 14 for S4/A5 post-CA3-M1(a))
   — following Task D's own `seed+10_000`
   precedent (a fixed per-purpose offset from the cell's base seed) rather
   than inventing a new convention. **Consequence, computed exactly (not
   asserted):** attack round 2's own executed false-block-rate figures
   (§1.15 — independently reproduced at `N=400,000`/group, 20× this
   design's own Monte Carlo scale) give a PER-CELL coverage-check false
   -block exposure of **≈2.75%** (the per-group rates — S3≈0%, S4≈0.068%,
   A5≈0.075%, S5≈0.135%, A6≈0.002% — summed across the ORIGINAL
   10-cells/group structure, the operative fresh-per-cell exposure over
   the full pre-Rev-3 50-cell
   sweep — the Rev-3 post-bump figure is derived just below) — small
   enough that a single false block is an expected,
   planned-for event at this sweep size, not a design flaw, PROVIDED it is
   handled by an explicit retry, not silently ignored. **Post-retry
   exposure, stated precisely (CA3-m2 fix, Rev 3 — round 3 found "far
   below the ≈2.75%" too vague for a two-consecutive-miss event).** The
   probability of a group failing BOTH independent draws (retry-once rule
   below) is `p_g²` per group; summed over the 10-cells/group structure
   the ≈2.75% figure's own denominator uses:
   `Σ_g 10·p_g² = 10·(0² + 0.00068² + 0.00075² + 0.00135² + 0.00002²)
   ≈ 2.85×10⁻⁵` — **S5-dominated** (`10·0.00135²≈1.82×10⁻⁵` is ~64% of the
   total, since S5 has both the largest single-draw rate and the biggest
   group order among the family). **Consistency note with §1.4.2's Rev-3
   cell-count bump (CA3-M1, below):** the `10` multiplier above is the
   PRE-Rev-3 uniform per-group cell count this ≈2.85×10⁻⁵ figure was
   computed against; after S4/A5 bump to 14 cells/group (§1.4.2), the
   Rev-3-consistent figure is `10·p_S3²+14·p_S4²+14·p_A5²+10·p_S5²+10·p_A6²
   ≈ 3.26×10⁻⁵` — still S5-dominated, still five orders of magnitude below
   1, i.e. still an expected-never-to-fire event at this sweep size, not a
   design flaw. **Retry-once rule
   (CA2-m1 fix, Rev 2, symmetric with the existing fit-set redraw guard
   below; itemized as its own build item, CA3-m1 fix, Rev 3, §1.12):** if a
   cell's `N=50` coverage check fails, resample the eval set
   ONCE with a shifted seed (the same cell seed `+ 1`), log the retry
   (cell id, both seeds, both realized coverage counts) to the run's
   output, and fail HARD on a second consecutive miss (two independent
   `N=50` draws both failing the calibrated bar is far below the ≈2.75%
   single-draw rate — a second miss is evidence of a real problem, not
   sampling noise, and must stop the cell rather than retry indefinitely).
   **Fit-set diversity floor.** The `n_fit=30` subset must realize
   `≥min(3·d_min(G), ⌊0.8·|G|⌋)` DISTINCT target elements. **Justification,
   reworded for precision (CA2-m3 fix, Rev 2 — Rev 1's phrasing implied
   naive linear-equation counting, which is not the actual mechanism).**
   Rev 1 described this as making the `d_min(G)²`-unknown orthogonal-
   intertwiner linear system "comfortably overdetermined" — imprecise:
   raw equation-vs-unknown counting does not, by itself, pin `Q`, because
   for a SINGLE fit element `g`, the linear constraint
   `Z(g)·Q = c·Q·ρ(g)` has a solution set of dimension equal to `ρ(g)`'s
   OWN centralizer (any `Q' = Q_true·C` with `C` commuting with `ρ(g)`
   also solves that one element's constraint) — a single `g` alone leaves
   `Q` badly underdetermined regardless of how many "equations" its `d²`
   entries nominally supply. **The real mechanism is Schur's lemma +
   irreducibility:** all five pinned reference representations are
   irreducible AND of real type (verified §1.3.1 — orthogonal, correct
   order, faithful; standard/deleted-permutation or icosahedral-rotation
   constructions, none complex- or quaternionic-type), so the centralizer
   of the FULL group image `ρ(G)` is exactly the scalar matrices (Schur),
   and — generically — as few as **~2** well-chosen (non-commuting)
   sampled elements already intersect their individual centralizers down
   to that same scalar-only limit, pinning `Q` up to the scale already
   handled separately by `c_hat` (§1.3.2 step 1). **EXECUTED, not merely
   asserted (CA3-m3 fix, Rev 3 — round 3 found this claim stated but never
   run against the actual pinned generators).**
   `coverage_calibration.py::verify_centralizer_dims` computes, for EACH
   of the five groups, the null-space dimension of
   `X ↦ (g₁·X − X·g₁, g₂·X − X·g₂)` over `d_min(G)×d_min(G)` real
   matrices, where `(g₁,g₂)` are the SAME 2 generators pinned in this
   design's own generating-set table (§1.4) — not generic/sampled
   elements — via the stacked-Kronecker/SVD method (`vec(g·X−X·g) =
   (I⊗g − gᵀ⊗I)·vec(X)`, same convention `fit_orthogonal_intertwiner`
   already uses, §1.3.2). **Executed output (verbatim, this session):**

   ```
   ========================================================================================
   CA3-m3 -- centralizer-dimension check on the 2 PINNED generators per group
   ========================================================================================
   Group   dim  null_dim   smallest_sv     next_sv       gap
   ---------------------------------------------------------
   S3        2         1      1.90e-16      1.7321    1.7321  PASS
   S4        3         1      4.89e-16      1.0000    1.0000  PASS
   A5        3         1      1.01e-16      1.1386    1.1386  PASS
   S5        4         1      2.62e-16      0.6126    0.6126  PASS
   A6        5         1      2.95e-16      0.6773    0.6773  PASS
   ```

   All five groups: null space dimension exactly `1` (spanned by the
   identity, i.e. scalar matrices only), with a well-separated
   smallest-vs-next-smallest singular-value gap (0.61-1.73, vs the
   near-machine-epsilon smallest sv ~1e-16) — a genuine isolated null
   space, not a numerically ambiguous near-tie. **The 2 pinned generators
   alone (before any additional fit-set elements are even drawn) already
   pin `Q` to scale, for all five groups, EXECUTED.** §1.4.1's "~2 generic
   elements suffice" claim is confirmed on the actual pinned pair, not
   just on the theoretical "generic element" argument. **The floor's actual
   SIZE is therefore a noise-margin choice, not an equation-counting
   requirement** — `3·d_min(G)` is anchored empirically to §1.3.2's own
   toy verification ratio (14 fit elements for a `d=3` group under
   `noise(std=0.03)`, ≈4.7×`d`), chosen conservatively BELOW that ratio
   (a floor, not a match) while comfortably ABOVE the ~2-element
   theoretical sufficiency point, giving robustness margin against
   realistic per-checkpoint noise rather than against a naive
   equation-counting shortfall. The number itself is unchanged from Rev 1
   — only the justification is corrected. Executed via
   `coverage_calibration.py::fit_set_diversity_check`
   (`n_fit=30`, 20,000 trials/group, §1.3.3) — every group clears its floor
   by ≥1 element at the 1st percentile (S3: bar 4, p1 5; S4: bar 9, p1 14;
   A5: bar 9, p1 19; S5: bar 12, p1 21; A6: bar 15, p1 25). If a real
   fit-set draw fails the floor (rare, ≤1% by construction), redraw the
   fit/eval split from the same `N=50` sample before fitting — a redraw
   guard, not a silent pass/fail toggle. **Eval-set diversity floor (CA2-m2
   fix, Rev 2 — Rev 1 had no analog on the scoring subset).** The
   DISJOINT `n_eval=20` scoring subset has no fitting role (it never
   touches `(Q_hat, c_hat)`), so it needs no equation-sufficiency argument
   — but an eval draw that happens to collapse onto very few distinct
   targets would still silently "score," just as a poorly-powered
   estimate of the checkpoint's real accuracy, not a caught error. A
   MODEST floor (deliberately weaker than the fit floor, anchored to the
   same MC machinery): `≥min(2·d_min(G), ⌊0.6·|G|⌋)` distinct elements.
   Executed via `coverage_calibration.py::eval_set_diversity_check`
   (`n_eval=20`, 20,000 trials/group, new this revision, §1.3.3-adjacent
   STEP 4) — every group clears its floor by ≥2 elements at the 1st
   percentile (S3: bar 3, p1 5, margin 2; S4: bar 6, p1 10, margin 4; A5:
   bar 6, p1 14, margin 8; S5: bar 8, p1 14, margin 6; A6: bar 10, p1 17,
   margin 7) — empirically fine, as attack round 2 already noted, now with
   an enforced floor rather than an unguarded assumption. Score
   `recovered_frac@0.9` and mean cosine on the DISJOINT `n_eval=20`
   subset (after its own floor clears, with the SAME retry-once-then-fail
   rule as the coverage check above if it doesn't) — this is the Stage-1
   decision metric (§1.5 M1/M3).
5. Report whole-matrix effective rank of raw `Z(w)` too (Task D/E's M1
   convention), but flagged NON-decisive by default — Task E §4's own
   retracted lesson (whole-matrix rank saturates toward the ambient
   ceiling once anything fills an unconstrained complement, becoming
   uninformative) is the reason the restricted metric above is primary,
   not the raw one; this design does not assume the SAME inflation
   mechanism recurs here (Task E's arose specifically from REPEATED
   self-application `Zʰ` compounding structure into the complement across
   many hops — this task has no such repeated self-application, single
   forward pass per word), but it re-runs the SAME check as a safety net
   rather than assuming the raw metric is trustworthy without it (the
   standing "never trust an instrument without recalibrating it for the
   new context" discipline, cf. the C17 tolerance-miscalibration lesson).

---

### 1.4.2 Arms, controls, and the force-rank grid

**Arm 1 — unconstrained rank (M1/M2-style recruitment measurement).**
Train with no `force_rank_k`; measure restricted effective rank (§1.4.1)
vs `d_min(G)` across the family (M1), and a post-hoc rank-`k` truncation
curve at eval time (§1.5 M2, Task D/E convention, `truncate_to_rank`
applied to a trained checkpoint's `Z` across `k=1..d_state`).

**Arm 2 — force-rank training grid, straddling `d_min(G)`:**
`k ∈ {d_min(G)−1, d_min(G), d_min(G)+1}` per group — the pre-registered
minimum. **Justification for exactly this grid (not wider):** `d_min(G)−1`
tests the FALSIFY boundary — below the group's minimal faithful
dimension, NO faithful representation of any kind exists (this follows
directly from the DEFINITION of `d_min(G)`, a representation-theoretic
fact independent of architecture, giving this design's exact-recovery
"necessity" argument — different from, and arguably tighter than, Task
D's stacked-independent-values proof, since it is definitional rather than
constructed): exact recovery should be near-chance. `d_min(G)` tests the
CONFIRM point: recovery should reach ≥0.9×ceiling. `d_min(G)+1` tests
SUFFICIENCY, not just necessity — confirming the step is genuinely AT
`d_min(G)` and performance does not keep climbing past it (Task D's own
M3 criterion band, "`k=K±1`," directly reused). A wider grid (e.g.
`d_min−2`) is not pre-registered as a minimum but is cheap to add if
attack round 1 wants more resolution; not added by default to keep Stage
1 inside its budget (§1.6).

**Seed allocation (C3-analog, ≥3 seeds minimum per the validation
verdict, economized by cell type — justified below, not a uniform-3
default). CA3-M1(a) fix, Rev 3 — S4 and A5 get an UNCONDITIONAL bump to
5 seeds (default, not contingent on the general escalation trigger below)
at the two cell types that carry the marquee dissociation claim
(unconstrained and `k=d_min`), because attack round 3's power simulation
(§1.4.2.1 below, executed) found the general n=3 escalation trigger
structurally unable to fire in the marquee's own dangerous regime (both
groups independently clearing their own within-group bars while
differing FROM EACH OTHER is not "ambiguous" by either group's own CI —
it is a genuine two-sample comparison the general trigger was never
built to catch). S3/S5/A6 are unaffected (unconditional n=3 remains
correct for them — they are not part of the marquee comparison):**

| Cell type | S3 / S5 / A6 seeds | S4 / A5 seeds (CA3-M1(a)) | Justification |
|---|---|---|---|
| Unconstrained (M1/M2) | 3 | **5 (unconditional)** | Needs clean statistics for the family-wide Spearman ρ (§1.5) — the headline recruitment claim. S4/A5 additionally carry the marquee's M1 restricted-rank TOST comparison (§1.5), which the power simulation below requires n=5 to clear its pre-registered power bar. |
| Force-rank `k = d_min(G)` | 3 | **5 (unconditional)** | The critical boundary point — this is where the "step at `k=K±1`" claim lives; needs full precision. S4/A5 additionally carry the marquee's step-location comparison at this exact grid point. |
| Force-rank `k = d_min(G)−1` | 2 | 2 (unchanged) | Expected to show a CLEAR near-chance signal (below the definitional minimum, no faithful representation exists at all) — a flanking/sanity cell, not a close call, mirrors `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.6's own quarter-budget-stress-point economization pattern (applied here via seed count, not step-budget, since step-budget is already the primary lever being kept short, §1.6). Not part of the marquee comparison (§1.5's marquee check is scoped to the unconstrained arm's restricted rank and the `k=d_min` step, not the flanking cells) — no bump needed. |
| Force-rank `k = d_min(G)+1` | 2 | 2 (unchanged) | Same reasoning: expected to clear ceiling clearly, a sufficiency check, not the primary boundary; not part of the marquee comparison. |

**Pre-registered escalation-to-5 trigger (per the validation verdict,
GENERAL case — S3/S5/A6, and S4/A5's own flanking cells):** if any cell
type's within-family Spearman ρ or M3 step-margin (§1.5) is
AMBIGUOUS (CI straddles the pre-registered bar), extend THAT cell type to
5 seeds before drawing a conclusion — mirrors
`REASONING_LINK_DESIGN.md` §16.19/§16.20's n=3→n=12 precedent and
`HEAD_TO_HEAD_DEMO_DESIGN.md` §1.8's seed-extension contingency
(gated the same way: variance-ratio check before pooling old/new
cohorts, `var_ratio > 4.0` → flag, don't silently pool).

**Marquee-specific escalation trigger (CA3-M1(d) fix, Rev 3 — a
`§1.4.2`-style trigger scoped to the marquee comparison itself, not the
general per-cell-type trigger above, since the marquee's TOST test can be
independently ambiguous even when neither group's own M1/M3 CI is).** If
the S4-vs-A5 TOST equivalence test (§1.5) returns NEITHER "declare
equivalence" NOR "reject equivalence" (i.e. the two one-sided tests split
— one rejects, one does not, the standard TOST INCONCLUSIVE outcome),
extend BOTH S4 and A5's unconstrained and `k=d_min` cells to **n=7**
before drawing a conclusion (the power simulation, §1.4.2.1 below, shows
n=7 measurably improves the equivalence-declaration side without
changing the already-saturated dissociation-detection power) — gated by
the same variance-ratio check (`var_ratio > 4.0` → flag, don't silently
pool) as the general trigger, applied to the pooled n=5→n=7 cohort.

**Cell count per group.** S3/S5/A6 unchanged: 3 (unconstrained) + 3
(`k=d_min`) + 2 (`k=d_min−1`) + 2 (`k=d_min+1`) = **10 cells/group**.
S4/A5 (CA3-M1(a) bump): 5 (unconstrained) + 5 (`k=d_min`) + 2
(`k=d_min−1`) + 2 (`k=d_min+1`) = **14 cells/group**. Five groups:
`10+14+14+10+10` = **58 cells total** (was 50 pre-Rev-3; §1.6 derives the
exact cost delta below).

**Controls (C-analogs, Task D §5's exact numbering scheme, restated for
this task):**
- **C1 (train-time force-rank, primary causal test).** As above —
  `rank_utils.truncate_to_rank` at train time, not post-hoc-only.
- **C3 (seeds + primary/secondary rank metric).** ≥3 seeds (economized
  above); primary metric = restricted effective rank (§1.4.1), secondary
  = restricted stable rank (`rank_utils.stable_rank`, same function,
  applied to the restricted `A(w)`, not raw `Z(w)`).
- **C5 (held-out generalization).** Word length `L∈{9..16}` at eval,
  never seen at `L∈{1..8}` train time (§1.4); query-coverage
  pre-registration (§1.4) as the concrete spanning bar.
- **C2 analog (param-matched flat-vector ablation) — EXPLICITLY
  DEFERRED, not silently dropped.** `CLAUDE.md`'s hard rule ("the
  param-matched flat-vector ablation blocks ALL downstream decisions...
  run it first") is a standing rule for END-TO-END capability claims; Task
  D itself descoped its own C2 (`HRRVectorMemory`) to "Stage-1b" for the
  identical reason (`model_v4.py`'s own comment: "NOT PART OF THE STAGE-1
  GATE... As-is, M4 is uninterpretable"). This design follows that exact
  precedent: Stage 1's question is narrower ("does SGD recruit
  provably-necessary MATRIX rank when the necessity argument is
  representation-theoretic minimality"), answerable without a vector
  control the same way Task D's Stage-1 gate was; a param-matched vector
  control (Option B/`learned-embedding` case, likewise deferred per the
  validation verdict) is registered as a Stage-1b/Stage-2 follow-on, not
  run here. Flagged again in §1.9 (self-attack item 6).

#### 1.4.2.1 CA3-M1(b)/(c) — the marquee TOST equivalence test, procedure and EXECUTED power simulation

**Why TOST, not bare CI-overlap (CA3-M1(b), replacing §1.5's old
"overlapping CIs" marquee check).** Attack round 3 found bare CI-overlap
at `n=3` badly underpowered for the marquee's actual job (disconfirming
"S4/A5 land together" when they genuinely don't) — CI-overlap is not a
real hypothesis test, it is a crude heuristic with no controlled error
rate, and a "no significant difference" reading from two wide, non-
overlapping-by-little CIs is not evidence FOR equivalence, only an
absence of evidence against it. **Two One-Sided Tests (TOST) is the
standard fix**: it directly tests the pre-registered claim ("the true
gap is small enough to call the same") against an explicit margin,
rather than testing the uninteresting null ("the true gap is exactly
zero").

**Equivalence margin, pinned: `±0.5` rank-units.** Justification: half
the unit spacing of the `d_min(G)` ladder (`[2,3,3,4,5]`, adjacent-rung
spacing `=1`) — a true S4/A5 gap smaller than half a rung is
scientifically compatible with "both track dimension 3" (it cannot be
confused with either group actually sitting at a DIFFERENT rung of the
ladder), while a gap at or beyond a full rung-spacing would not be.

**Test procedure, pinned exactly: Welch's unpaired TOST (two one-sided
Welch t-tests, Satterthwaite-approximated degrees of freedom), NOT
paired-per-seed.** S4 and A5 share literal seed-index labels by this
project's convention (§1.7 gate 1's "seed=0" example) and IDENTICAL
architecture dimensions (`d_state=5`, `|S_G|=4` for both, since both have
`d_min=3`) — so an S4-seed-`k` and an A5-seed-`k` run share bit-identical
INITIAL weights under the same `torch.manual_seed(k)`. But S4 and A5
train on entirely DIFFERENT data (distinct Cayley graphs / generating
sets, different mixing statistics, §1.3.4), and assuming that
shared-init correlation survives 8,000 steps of divergent-data training,
without evidence, would be exactly the kind of unverified assumption
this gauntlet exists to catch. Welch's unpaired test is the conservative
default that does not require this assumption; if a future build-time
audit of real checkpoints shows detectable init-correlation surviving
training (not assumed here), a paired analysis is available as a
non-load-bearing robustness cross-check, not the primary decision
procedure.

**Applied to:** (1) M1's restricted effective rank (unconstrained arm,
`n=5` per group after CA3-M1(a)) — the primary, continuous, rank-unit
metric the `±0.5` margin is scaled to. (2) M3's step location is, by
construction, the SAME shared grid point (`k=3`) for both groups (both
`d_min=3`) — so the marquee's "step locations... indistinguishable"
clause (§1.5) is satisfied when EACH group independently clears its own
M3 CONFIRM criterion at that grid point (near-chance at `k=2`, `≥0.9×`
ceiling at `k=3`); this is already checked per-group and does not need a
second continuous equivalence test on top of (1).

**EXECUTED power simulation (CA3-M1(c), not asserted).** Script
(repo-committed): `matrix-thinking/capability_separation/
marquee_power_simulation.py` (numpy+scipy, deterministic
`RNG_SEED=20260711`). The original attack-round-3 simulation script is
not repo-committed (attack rounds are ephemeral agents); this
INDEPENDENTLY RE-DERIVES a comparable simulation from round 3's own
reported description (§1.17) — a plausible per-seed noise grid for the
restricted-rank metric, bracketing the M1 CONFIRM band's width
(`[0.7,1.3]×d_min`, i.e. `±0.9` rank-units around `d_min=3`) on its
tighter side, and validated by reproducing round 3's own reported n=3
bare-CI-overlap miss-rate ranges (a qualitative match, not a byte-exact
reproduction, since the original script is unavailable):

```
SANITY CHECK -- bare CI-overlap 'missed' rate at n=3 (ORIGINAL pre-Rev-3 method,
reconstructed noise grid). Target (attack round 3, S1.17): 0.5-gap missed 71-97%,
1.0-gap missed 11-80%.
====================================================================================================
  sigma    gap=0.5 missed    gap=1.0 missed
-------------------------------------------
   0.15             70.9%             11.0%
   0.20             85.3%             35.6%
   0.25             91.4%             56.7%
   0.30             94.5%             70.9%
   0.35             96.0%             79.6%

Reconstructed grid spans: 0.5-gap missed 70.9-96.0% (target 71-97%); 1.0-gap missed 11.0-79.6% (target 11-80%).
Qualitative match confirmed -- grid adopted for the production table below.
```

**Production table (the Rev-3-adopted design: `n=5`, Welch TOST,
margin=`±0.5`):**

```
PRODUCTION -- Welch TOST at n=5, margin=+-0.5 rank-units  [Rev-3 ADOPTED design]
====================================================================================================
  sigma    P(declare equiv | gap=0)   P(reject equiv | gap=0.5)   P(reject equiv | gap=1.0)
-------------------------------------------------------------------------------------------
   0.15                       99.8%                       95.3%                      100.0%
   0.20                       94.2%                       95.4%                      100.0%
   0.25                       76.4%                       95.3%                      100.0%
   0.30                       53.2%                       95.4%                      100.0%
   0.35                       33.1%                       95.8%                      100.0%

Minimum P(correctly reject equivalence | gap=1.0) across the grid: 100.0%  (CLEARS the pre-registered 70% bar)
```

**Reading the table.** `P(reject equiv | gap=0.5)` (the margin boundary)
sits near the nominal `95%` level across the whole grid, as TOST's own
construction predicts (this column confirms correct test SIZE, not a
power figure per se). `P(reject equiv | gap=1.0)` — the real power
figure CA3-M1(c) requires — is **100% across the entire reconstructed
noise grid**, comfortably clearing the pre-registered 70% bar; **no
escalation to `n=7` is required on this axis.** `P(declare equiv |
gap=0)` is noise-sensitive (`~100%` at the low end of the grid down to
`~33%` at the high end) — **flagged, not papered over**: at the noisy
end of the plausible grid, TOST's conservative construction means the
marquee comparison may render INCONCLUSIVE (triggering the CA3-M1(d)
`n=7` escalation above) even when the true story is "both track
dimension 3," rather than falsely confirming or falsely denying
dimension-tracking — a safe failure mode (conservative, not
misleading), but a real, disclosed possibility this design does not
eliminate. An `n=7` comparison row (same script, `run_production_table`)
shows this column measurably improves at `n=7` (`60.8%` at the noisiest
grid point vs `33.1%` at `n=5`) — the basis for the CA3-M1(d) escalation
trigger above, not adopted as the default because the pre-registered
70%-power bar is scoped to the dissociation-detection direction, which
`n=5` already saturates.

---

### 1.5 Pre-registered measurements & decision criteria

Fixed per-group operating point: `d_state(G) = d_min(G)+2`,
`L_train∈{1..8}`, `L_test∈{9..16}` (held-out), symmetric generating set
(§1.4). Success metric: `cos(degauged Z(w), rho_G(product(w))) > τ`
(τ=0.9 primary, Task D/E convention), scored via §1.3.2's fit/eval-split
Procrustes/scale degauging pipeline on the §1.4.1 restricted operator.

**M1 — restricted effective rank vs `d_min(G)`** (unconstrained arm,
across the 5-group family).
- CONFIRM: Spearman ρ(`d_min(G)`, restricted_eff_rank) ≥ 0.8 across the
  family AND restricted_eff_rank ∈ `[0.7·d_min, 1.3·d_min]` per group
  (Task D M1's exact band).
- **Statistical weight, disclosed exactly (CA1-M1 fix, Rev 1 — Rev 0 left
  this undisclosed).** The family's `d_min(G)` sequence is `[2,3,3,4,5]`
  (S3,S4,A5,S5,A6) with a TIE at `d_min=3` (S4/A5 — the marquee
  dissociation pair, intentionally). Under the exact permutation null (all
  `5!=120` equally-likely rank assignments of the measured metric,
  midrank-adjusted for the tie; independently re-derived from scratch this
  Rev-1 pass, §1.3.3, byte-identical to §1.13's own figures):
  `P(ρ≥0.8 | null) = 8/120 ≈ 6.667%`, and the tie caps the MAXIMUM
  achievable ρ at `0.9747` (perfect ordering respecting the S4/A5 tie).
  **M1 alone is corroborating-only, not independently decisive at this
  bar** — `ρ≥0.8` is not conventionally significant at `α=0.05` on its own
  (one-sided 6.67%). The Stage-1 CONFIRM verdict's actual statistical
  weight rests on **M3's per-group paired-seed causal force-rank test +
  the S4-vs-A5 marquee dissociation check**, both genuine independent
  causal tests — M1's role is a corroborating trend check, stated
  explicitly here rather than left implicit.
- **Bar choice: `ρ≥0.8` kept, NOT tightened to `ρ≥0.9`, evaluated and
  declined (CA1-M1's own suggested alternative).** `ρ≥0.9` is nominally
  achievable (`0.9747 ≥ 0.9`, exact `P(ρ≥0.9|null)=2/120≈1.667%`, §1.3.3)
  and would be a stronger disclosed p-value — but the achievable-ρ set
  under this exact 5-point/1-tie configuration is DISCRETE with a steep
  cliff just below the maximum: the next-highest achievable value after
  `0.9747` (perfect ordering) is `0.8721` — independently enumerated, a
  drop of `~0.10` from a SINGLE non-tie pairwise misordering among
  S3/S5/A6's three non-tied points. A `ρ≥0.9` bar is therefore satisfied
  ONLY by (essentially) perfect ranking; any one ordinary-noise
  misordering collapses it straight to `0.8721`, BELOW even the ORIGINAL
  `0.8` bar's own value. Since M1 is corroborating-only (above),
  tightening it to a near-exact-match bar would convert a soft trend
  check into a brittle pass/fail gate disproportionate to its actual role
  in the CONFIRM decision — declined. `ρ≥0.8` (satisfied by any of the 8
  permutations scoring `≥0.8208`, i.e. tolerating one non-tie
  misordering) is kept as the better-calibrated corroborating bar.
- FALSIFY: restricted_eff_rank stays ≤2 (or flat across the family) while
  recovery accuracy is high. (If observed with a PASSING blank-out test,
  this contradicts §1.4's definitional necessity argument and signals a
  readout/degauging leak to debug, not a valid result until ruled out —
  Task D M1's exact contingency, re-applied.)

**M2 — post-hoc rank-`k` truncation curve** (unconstrained checkpoint,
`k=1..d_state`, τ=0.9).
- Knee `k*` = smallest `k` with `acc(k) ≥ 0.9·acc(k=d_state)`.
- CONFIRM (CA3-m5 fix, Rev 3 — reworded, was a stray sentence fragment):
  `k* ∈ [d_min(G)−1, d_min(G)+1]` in ≥2/3 seeds at S3/S5/A6's default
  `n=3` (escalating to the ≥4/5 bar only if the general escalation
  trigger fires for that group, §1.4.2), OR in ≥4/5 seeds directly at
  S4/A5's unconditional `n=5` (CA3-M1(a), Rev 3, §1.4.2 — no trigger
  needed, the larger seed count is the default for these two groups).

**M3 — train-time force-rank-`k` (PRIMARY causal test)** — the force-rank
grid arm.
- CONFIRM: recovery is near-chance for `k=d_min(G)−1` AND ≥0.9×ceiling for
  `k∈{d_min(G), d_min(G)+1}`, with a monotone step at `k=d_min(G)` — Task
  D M3's exact criterion, ported.
- HARD FALSIFY (premise dead for this task family): `k=d_min(G)−1`
  reaches ≥0.9×ceiling at ANY group — a below-minimal-dimension solution
  exists despite the pinned block-embedded readout ⇒ investigate a
  degauging/embedding leak first (§1.9 item 1); if real, this task family
  is rank-blind in the way Task D's premise would have been.

**Marquee dissociation check (S4 vs A5, both `d_min=3`) — CA3-M1(b)/(d)
fix, Rev 3: bare CI-overlap REPLACED with a pre-registered TOST
equivalence test.** The two groups' M1 restricted-rank values (`n=5`
each, CA3-M1(a)) must PASS a Welch TOST equivalence test against a
`±0.5` rank-unit margin (§1.4.2.1 — full procedure, justification, and
the EXECUTED power simulation confirming this design clears a 70% power
bar against a real 1.0-unit gap) — i.e. TOST must declare equivalence,
not merely "CIs happen to overlap." M3's step locations are checked via
each group's own independent M3 CONFIRM criterion at the shared grid
point `k=3` (§1.4.2.1) — both groups must independently CONFIRM there.
If TOST returns neither "declare" nor "reject" (an INCONCLUSIVE split
between the two one-sided tests), the CA3-M1(d) marquee-specific
escalation trigger (§1.4.2) extends both groups to `n=7` before a
verdict is drawn. This IS the CONFIRM condition's "land together, not
apart" clause, made a concrete, checkable, EXECUTED-power comparison,
not just prose.

**Overall Stage-1 verdict:** CONFIRM if M1 CONFIRM AND M3 CONFIRM (M2
corroborating) AND the marquee check passes. FALSIFY if M3 HARD FALSIFY OR
(M1 FALSIFY with a passing blank-out test). Otherwise INCONCLUSIVE →
diagnose before Stage 2 (§1.11).

---

### 1.6 Cost arithmetic

**Rate derivation (two independent anchors, both cited — the validation
verdict's own cost-derivation instruction).**

- **Task D anchor.** Realized total: **~76 GPU-h**
  (`EXPERIMENT_LOG.md:1556`, "~76 GPU-h on the new Brev 8×H100 cluster";
  independently restated `STATE.md:984`, "Task D used ~76 GPU-h (~5% of
  budget)"). Stage-1-class rate (the relevant one for THIS design, not the
  realized total, most of which was Stage 2 + overnight-sweep refill
  excess per `EXPERIMENT_LOG.md:1619-1625`): the pre-registered Stage-1
  minimal-gate estimate was **~5-10 GPU-h for ~18 short runs**
  (`TASK_D_PREREGISTRATION.md` §7) → **≈0.28-0.56 GPU-h/run**.
- **Task E anchor.** The 80K-step K-wall-resolution round: **6 runs ×
  ~2.4 GPU-h ≈ 14.5 GPU-h serial-sum** (`EXPERIMENT_LOG.md:2230`,
  independently restated `TASK_E_FINDINGS.md` §10 line 710) → by linear
  step-budget scaling, an 8,000-step-equivalent run (this design's
  planned default, below) costs **≈2.4 × 8000/80000 ≈ 0.24 GPU-h/run**
  (an explicit, disclosed linear-scaling assumption, not a directly
  measured 8K-step Task E rate — flagged for the calibration gate to
  confirm or correct).
- **Correction to the validation verdict's own cost line:** the binding
  content cites "Task D's realized 11.9 GPU-h total" as an anchor; this
  figure could not be verified anywhere in `EXPERIMENT_LOG.md`,
  `STATE.md`, or `TASK_D_WRITEUP.md` (`grep`-searched exhaustively). The
  nearest matching number found, `EXPERIMENT_LOG.md:4984`
  ("Stage 1 + Stage 2 combined = 11.9072 GPU-h"), belongs to a DIFFERENT,
  unrelated program (the coherence-dose-response wave), not Task D. Per
  `CLAUDE.md`'s "verify before claiming" rule, this design uses the
  independently-verified **~76 GPU-h** Task D total and the **~5-10
  GPU-h / ~18-run** Stage-1 estimate instead, and flags the discrepancy
  here rather than silently propagating an unverifiable number.

**Both anchors converge to roughly 0.2-0.5 GPU-h/cell at an 8,000-step
budget for a Task-D/E-class bespoke encoder** (`h=64`, ~171K params, at
`d=16`). This design's encoder is smaller in TWO ways Task D/E's rate
doesn't reflect (`d_state≤7` vs 16, `h=32` vs 64) — plausibly cheaper —
but this is a DISCLOSED, calibration-pending economization (§1.4), not
assumed. **Planning rate: 0.3 GPU-h/cell** (mid-range of the two anchors,
at an 8,000-step planned default step budget — see below).

**Step budget.** Planned default: **8,000 steps** (Task D's own original
per-run budget, `run_task_d.py`'s CLI default region). **Pre-registered
2-2.5× escalation rule** (not assumed away): any cell showing the "flat
loss for most of budget, then late transition" signature this project has
now seen three independent times (Task E K=8 20K→40K, Stage 0 d≥32
8K→100K, Task E K=12/16 40K→80K — `EXPERIMENT_LOG.md`'s "three-budget-
artifacts" finding) gets retested at 2-2.5× budget before being called
dead, exactly per `CLAUDE.md`'s hard rule. This is a contingency, not
baked into the base estimate below (mirrors Task D/E's own convention of
costing the first-look budget separately from contingency).

| Item | Cells / basis | GPU-h |
|---|---|---|
| Main sweep (58 cells × 0.3 GPU-h) — CA3-M1(a) fix, Rev 3: was 50 cells/15.0 GPU-h; S4/A5 bump to 14 cells/group (§1.4.2) | 5 groups: 10+14+14+10+10=58 cells/group (§1.4.2) | 17.4 |
| Calibration-first wave (§1.7 gate 1) — 5 cells (1/group, unconstrained, seed=0), **reused within the 58 sweep cells above, not double-charged** | 5 cells | 0.0 (already counted) |
| Degauging-pipeline validation on a REAL trained checkpoint (§1.7 gate 1) | CPU-only, numpy | ≈0.0 |
| Contingency margin — 2-2.5× escalation rule firing on 1-2 cells (unchanged from Rev 2: this margin covers "1-2 cells" needing a longer step budget, a rate-of-occurrence assumption independent of total cell count, not proportional to the 50→58 bump) | ~2 cells re-run at 2.25× | 1.35 |
| **β∈[0,2] fla positive-control smoke** (§1.4.3 below) — forward/backward/grad-check (~0.05) + one reproduced DeltaProduct Fig.5 point on a minimal S4/A5 instance, `L≤4`, `n_h∈{1,2}` (~0.85) | nominal | **0.90** |
| Group-construction + generator-closure smoke (§1.3, §1.4) — CPU-only, numpy | — | ≈0.0 |
| MATCH/verification overhead (blank-out test, query-coverage check, injectivity-guard negative test, marquee power simulation) — CPU-only | — | ≈0.0 |
| **Raw total** | | **≈19.65 GPU-h** |

**CA1-M2 fix, Rev 1 — cost ceiling reconciled.** Rev 0's stated raw total
("≈17.4-18.3 GPU-h") carried ~1 GPU-h with no itemized source above the
itemized rows' own sum (§1.13 CA1-M2). Fixed by itemizing the previously-
vague `<1.0` β-smoke row to a specific **0.90 GPU-h** (forward/backward/
grad-check plus the actual Fig.5-reproduction training run, not a nominal
upper bound) — the Rev-1/Rev-2 raw total equaled the itemized sum exactly:
`15.0 + 1.35 + 0.90 = 17.25` GPU-h, no unaccounted margin.

**CA3-M1(a) fix, Rev 3 — cost delta from the marquee seed bump, DERIVED
exactly (not copied from the attack's own estimate).** The main-sweep row
alone changes: `58 cells × 0.3 GPU-h/cell = 17.4` GPU-h (was `50×0.3=15.0`),
a delta of exactly **`+2.4` GPU-h** on that row. The contingency and
β-smoke rows are unchanged (both are rate-of-occurrence estimates over
"1-2 cells" / a fixed nominal smoke, not row totals that scale with the
sweep's cell count). New raw total: `17.4 + 1.35 + 0.90 = 19.65` GPU-h —
exactly `19.65 − 17.25 = +2.4` GPU-h over the Rev-2 raw total, confirming
attack round 3's own `~2.4 GPU-h` estimate (§1.17, CA3-M1) via independent
derivation from the cell table rather than by copying it.

**Stage-1 dedicated ledger: 30 GPU-h cap, PI-visible, does NOT draw the
frozen-bias ledger** (per the validation verdict's explicit instruction —
this campaign is budgeted separately from the head-to-head demo's 135
GPU-h program ceiling). Raw ≈19.65 GPU-h (Rev 3, was ≈17.25 GPU-h pre-Rev-3)
leaves **≈34.5% margin** (`(30−19.65)/30`) under the 30 GPU-h cap (Rev 3
recomputation of CA1-M2's reconciled-figure convention — the marquee
seed bump's `+2.4` GPU-h costs `8.0` percentage points of margin, `42.5%
→ 34.5%`, landing inside the pre-bump `12.75` GPU-h margin exactly as
attack round 3 anticipated, §1.17) — comfortably
wider than `HEAD_TO_HEAD_DEMO_DESIGN.md`'s
own razor-thin margins (≈1-2%), appropriate for a first-look Stage-1
gate at this much smaller scale (`d_state≤7` vs the head-to-head's 14M+
param models).

**Circuit breaker — deliberately NOT a literal 10×-of-raw multiplier
(flagged as a design decision, not silently inherited from
`HEAD_TO_HEAD_DEMO_DESIGN.md`'s convention).** That design's 10× bracket
exists because ITS cells are large (0.25 GPU-h/cell at 14M params,
scaling to 28 GPU-h/cell at 392M) and a runaway sweep could plausibly
consume the WHOLE shared 135 GPU-h ceiling; a literal `10×19.65≈196.5 GPU-h`
bracket here (Rev 3's raw total; was `10×17.25≈172.5 GPU-h` pre-Rev-3)
would be nonsensical overkill relative to this campaign's
own explicitly-stated 30 GPU-h cap. **This design's circuit breaker
instead mechanically enforces the 30 GPU-h dedicated cap directly**, via
the same timing-pilot-based abort machinery precedent
(`phase2b_off_cache.py --time-pilot`, §1.7 gate 2): if the calibration
cell's measured real rate projects the 58-cell sweep to exceed 30 GPU-h
(incl. the pre-registered contingency margin), the chain hard-aborts
before spending it, and this design's cell/seed grid gets re-scoped
before relaunch, not silently over-spent.

**CA4-M1 fix, Rev 4 — the hard-abort also guards the escalation triggers,
not just the base sweep.** Attack round 4 (§1.19) found both escalation
triggers (the GENERAL escalation-to-5 trigger, §1.4.2, inherited since
Rev 0 and uncaught through 3 rounds; and the marquee-specific `n=7`
trigger, §1.4.2, CA3-M1(d)) uncosted, with the hard-abort above only
projecting the BASE 58-cell sweep once upfront — a build agent had no
instruction to budget-check before launching an escalation-triggered
cell. **Worst-case arithmetic, disclosed explicitly:** base raw total
`19.65` + marquee `n=7` escalation (`+2.40`, if TOST returns
INCONCLUSIVE on both S4 and A5) + general escalation-to-5 at its own
literal maximum (every remaining ambiguous-eligible cell type escalates:
`S3`/`S5`/`A6`'s all 4 cell types 3→5/3→5/2→5/2→5 each, plus `S4`/`A5`'s
2 flanking cell types 2→5 each — `42` extra cells `× 0.3` GPU-h/cell
`= 12.60`) = `19.65 + 2.40 + 12.60 = **34.65 GPU-h > the 30 GPU-h cap by
4.65`. **Fix: the hard-abort re-checks actual-spend-to-date + projected
cost BEFORE launching ANY escalation-triggered cells** (general
escalation-to-5 OR the marquee `n=7` escalation), not only once upfront
against the base sweep. If the projection (actual spend so far +
already-committed escalations + the newly-triggered escalation's cost)
would exceed the 30 GPU-h cap, the guard converts the overrun into a
**pre-registered priority allocation** rather than a silent over-spend or
an undirected stop: the **marquee `n=7` escalation outranks all general
escalations** (it guards the design's central S4-vs-A5 dissociation
claim, §1.5 — the one result this design cannot ship without); **general
escalations are granted in a pinned order — by cell type's group `|G|`
ascending** (S3 `|G|=6` first, then S4/A5 `|G|=24/60` tied, then S5
`|G|=120`, then A6 `|G|=360` last; justified as cheapest information
first — a small-`|G|` group's ambiguous cell is resolved by the fewest
additional GPU-h per unit of statistical resolution, so exhausting
remaining budget on it first buys the most de-ambiguation per dollar) —
until the projection hits the cap. **Anything denied is REPORTED as
budget-denied in the final harvest** (a distinct status from "ran, came
back ambiguous" or "skipped by design"), not silently dropped — the
harvest script must be able to distinguish "this cell's ambiguity was
never resolved because budget ran out" from "this cell resolved cleanly
at its base seed count," since the two have different evidentiary
weight for §1.5's decision criteria. This is a budget-arbitration
mechanism only — it does not change which cells are ELIGIBLE to
escalate (§1.4.2's triggers, unchanged) or the decision criteria
(§1.5, unchanged); it only orders which eligible escalations actually
launch when the 30 GPU-h cap would otherwise be breached.

**β∈[0,2] fla positive-control smoke, registered, non-load-bearing:**
**0.90 GPU-h** (CA1-M2 fix, Rev 1 — itemized: ~0.05 GPU-h
forward/backward/grad-check + ~0.85 GPU-h to actually reach the
qualitative split below on the minimal instance, replacing Rev 0's
un-itemized "<1.0") on a minimal `fla` DeltaNet layer
with `allow_neg_eigval=True`, β-gate range `[0,2]` (not this repo's
current plain-sigmoid `β∈[0,1]`, per Grazzi et al. 2411.12537), `n_h∈{1,2}`
Householder products, trained on a MINIMAL S4 (or A5) word-problem
instance (short words, `L≤4`) to reproduce the qualitative split
DeltaProduct (arXiv:2502.10297) reports in its Fig. 5 — `n_h=1` fails
state tracking on this group family, `n_h=2` succeeds (matches
`STATE.md`'s own verified citation: "DeltaProduct's own published `n_h`
numbers confirm S4 `n_h=2`, A5 `n_h=2`, S5 `n_h=4`"). **Explicitly NOT
load-bearing for Stage 1's hypothesis** — Stage 1 uses the bespoke
`BindingEncoder` architecture throughout, not `fla`/DeltaNet. Its purpose
is to validate the `fla` `allow_neg_eigval` code path works correctly on
this repo's stack BEFORE Stage 2 or any later wave that might want a
production delta-rule kernel instead of the bespoke encoder — registered
and run during Stage 1's build, scored and reported, but never gates
Stage 1's CONFIRM/FALSIFY/INCONCLUSIVE verdict.

---

### 1.7 Gates

Reused verbatim from this program's own precedent (Task D/E,
`HEAD_TO_HEAD_DEMO_DESIGN.md` §1.7), not re-derived:

1. **Calibration-first, 5 cells (1/group, unconstrained arm, seed=0),
   mandatory per `CLAUDE.md`.** Run to completion BEFORE the remaining 45
   sweep cells launch. Does FOUR duties in sequence (CA1-M3 fix, Rev 1 —
   (b) below is a new, concrete acceptance test; Rev 0 had no ground-truth
   criterion for a real checkpoint at all): (a) confirms convergence at the
   planned 8,000-step budget (or triggers the 2-2.5× escalation rule,
   §1.6, before the rest of the sweep inherits a bad assumption);
   (b) **synthetic-injection acceptance test, run BEFORE any real
   checkpoint's degauging output is trusted.** Since no ground-truth
   `(Q,c)` exists on a real checkpoint, the PRODUCTION eval harness (the
   exact imported `verify_option_a_readout.py` functions wired into the
   real pipeline per gate 6 below — not a standalone numpy re-run) is fed
   a SYNTHETIC injection instead: a known `(Q_true, c_true)`-conjugated
   `rho_G` trajectory (`Z_synth(w) = c_true · Q_true · rho_G(w) · Q_trueᵀ +
   noise(std=0.03)`, §1.3.2's exact ground truth and noise level) written
   into the harness's real on-disk dump format/shapes/dtypes, for S4
   (reusing §1.3.2's own setup, `n_fit=14`/`n_eval=10` there — or the
   Rev-1-pinned `n_fit=30`/`n_eval=20` split, §1.4.1 step 4, if the
   production harness is already wired to it). Acceptance requires, on
   the disjoint eval subset: **recovery** — `mean_cos > 0.95` AND
   `recovered_frac@0.9 ≥ 0.9` (§1.3.2's own noisy-condition thresholds,
   unchanged) — AND **the rank-deficient negative control holds in the
   SAME harness** — inject §1.3.2's `Q_def` (rank-2-of-3) corruption
   through the identical production code path, require `mean_cos` at
   least `0.3` below the honest recovery's value AND
   `recovered_frac@0.9 < 0.5` (§1.3.2's exact corrupted-condition
   thresholds). This catches wiring/shape/dtype bugs in the PRODUCTION
   harness specifically — a standalone numpy re-run of
   `verify_option_a_readout.py` cannot, by construction, catch those; only
   after (b) passes does (c) **the Procrustes/scale degauging pipeline
   (§1.3.2) get validated ON the 5 REAL trained checkpoints** — this is
   the single biggest registered risk in this design (§1.9 item 1), and
   the validation verdict explicitly requires this to happen on the
   calibration cell, not deferred; (d) measures the REAL per-cell
   wall-clock rate, superseding §1.6's planning-estimate rate for the
   remaining 53-cell sweep's own go/no-go (Rev 3: `58−5=53`, was `45`
   pre-Rev-3's 50-cell total).
2. **Timing pilot, mechanical enforced abort.** One real cell per group
   measured for wall-clock (folds into gate 1 above, since the
   calibration cells ARE one real cell per group); if the projected
   58-cell sweep cost exceeds the 30 GPU-h dedicated cap (§1.6), the
   chain hard-aborts before spending it.
3. **Enforced aborts with negative tests.** The injectivity-style guard
   this task needs is simpler than Task E's (`_assert_injective`'s
   rank-K-eff exact threshold) because words are freely generated (no
   merge-collision risk the way Task E's key/value pool had) — but the
   query-coverage check (§1.4) gets its own negative test
   (`_test_coverage_guard_detects_undersampling`, precedent:
   `task_e.py::_test_injectivity_guard_detects_merge`) proven to
   actually fire before being trusted, per `CLAUDE.md`'s "never trust a
   'proves the check has teeth' test without running it to completion"
   rule. **Re-specified against the corrected per-group bars (CA1-F1 fix,
   Rev 1, §1.4/§1.3.3):** the negative test plants the `L=1` pathological
   sampler (§1.4) for EACH of the 5 groups and asserts the coverage check
   fails every time; this cannot flake, because the undersampled
   sampler's reach (`|generating set|` ≤ 4 elements, a DETERMINISTIC
   ceiling) sits strictly below every group's calibrated bar (`≥5` for
   S3 up to `≥36` for A6, §1.4's table) — proven directly by
   `coverage_calibration.py`'s `undersampled_trial` (§1.3.3), not merely
   claimed.
4. **Sha closure.** Every shipped script/config gets a sha256 manifest,
   verified against the box copy before launch and re-verified after
   completion (standing project convention).
5. **PI-signoff token.** `CAPABILITY_SEP_PI_SIGNOFF=1` required before any
   GPU cell — one token (this campaign's scope is small enough that a
   second, MATCH-GATE-style token is not warranted the way
   `HEAD_TO_HEAD_DEMO_DESIGN.md`'s three-way matched comparison needed
   one). **Cross-reference closed (CA2-m4 fix, Rev 2 — Rev 1 flagged this
   for attack round 1 but the round-1 verdict, §1.13, never explicitly
   adjudicated it, leaving the reference dangling into Rev 1).** Attack
   round 2 (§1.15) adjudicated the single-token simplification SAFE: the
   second, MATCH-GATE-style token in `HEAD_TO_HEAD_DEMO_DESIGN.md` exists
   specifically to gate a three-way PARAMETER/FLOP/BYTE-matched comparison
   where a silent asymmetry between arms could invalidate the whole
   comparison; this design has no such matched-comparison invariant at
   risk — the C2 vector-ablation control is EXPLICITLY DEFERRED (§1.4.2,
   §1.9 item 6), so there is no second arm whose match-integrity a second
   token would need to separately gate. One token is sufficient and the
   simplification stands, closed.
6. **Reference-representation + degauging-pipeline verification (this
   design's MATCH-GATE analog).** `verify_option_a_readout.py`'s Part 1
   (group construction) and Part 2 (degauging recovery) must both PASS
   (all 5 + 5 assertions, §1.3) — already run and passing as of this
   Rev 0 (§1.3), re-verified against the sha-closed script copy on-box
   before launch (gate 4), not re-derived from scratch by a second
   independent agent the way `HEAD_TO_HEAD_DEMO_DESIGN.md`'s two-pass
   MATCH-GATE requires (this design's readout is a closed-form numpy
   computation, not a three-architecture parameter/FLOP/byte match with
   room for silent asymmetry — an independent build audit, §1.11, is the
   equivalent safeguard here).
7. **Blank-out test (§1.4), mandatory pre-training,** reused verbatim
   from `run_task_d.py::smoke()` step [5] — must pass before any training
   cell launches, exactly as Task D requires.

---

### 1.8 Pre-registered analysis

**CI machinery.** Paired-seed CI, same `t(n-1,.975)·s/√n` construction
this program's `delta_ci_n`/Task D-E convention already uses; at `n=3`,
`df=2, t=4.303`. Escalation to `n=5` (§1.4.2's trigger) uses `df=4,
t=2.776`.

**Multiple comparisons.** Five groups × (M1 Spearman test + M3 step test)
= 10 nominal tests, plus the S4/A5 marquee-dissociation comparison — this
design does NOT apply a downward alpha-correction across groups (the
pre-registered semantics is "does the pattern hold ACROSS the family,"
read via the family-wide Spearman ρ in M1, which is already a single
test over all 5 points, not 5 independent tests) but DOES disclose, in
any write-up, that M3's per-group step tests are 5 separate nominal-95%
tests (family-wise ≈`1-0.95^5≈22.6%` under a global null) — stated
plainly, mirroring `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.8's own disclosed
family-wise-inflation convention rather than silently presenting 5 clean
95% results.

**Seed-extension contingency.** Per §1.4.2's trigger — extend the
AMBIGUOUS cell type only, gated through the same batch-effect
variance-ratio check (`var_ratio > 4.0` → flag, don't silently pool)
`REASONING_LINK_DESIGN.md` §16.19.5 registered and `HEAD_TO_HEAD_DEMO_DESIGN.md`
§1.8 reuses.

---

### 1.9 Self-attack register (minimum 5 per house convention; 9 here,
seeded from the validation verdict's required list plus this design's
own additions)

1. **The degauging-rescue risk is the #1 risk in this design, restated
   precisely.** §1.3.2's toy verification proves the fit/eval split has
   teeth in a CONTROLLED numpy setting (known ground-truth `Q_true`,
   `c_true`, a clean rank-deficient corruption). The real risk is
   whether this holds on a REAL trained checkpoint's noisier, less
   clean failure modes (e.g. a model that's PARTIALLY right — correct on
   some words, wrong on others, rather than uniformly rank-deficient) —
   this is exactly why gate 1 (§1.7) requires validating the pipeline on
   the 5 real calibration checkpoints BEFORE trusting the main sweep's
   numbers, not deferred to post-hoc analysis.
2. **Task E's three-budget-artifacts warning — dead cells need 2-2.5×
   budget retests, pre-registered here as a rule (§1.6), not a hope.**
   This design's 8,000-step planned default is CHEAPER than Task E's own
   40K-80K step budgets that hit this pattern three times; there is a
   real risk the compositional word-recovery task (closer in spirit to
   Task E's multi-hop composition than Task D's flat lookup) needs a
   longer budget than planned. The escalation rule exists specifically so
   an under-budgeted cell is retested, not declared FALSIFY prematurely
   (the exact mistake `CLAUDE.md`'s hard rule was written to prevent).
3. **Does `k=d_min(G)−1` failing reflect the representation-theoretic
   necessity bound, or a trivial optimization/vanishing-gradient artifact
   of `truncate_to_rank`'s `eigh`-based projection at very small `k`
   (`k=1` for S3)?** Disambiguation: the blank-out test (§1.4) and the
   NaN/Inf-free degenerate-spectrum backward check
   (`run_task_d.py::smoke()` step [2], already verified for `k` down to
   1 on `d=16` matrices — re-run at THIS design's much smaller `d_state`
   as part of gate 4's sha-closed smoke) rule out a numerical-instability
   confound; a genuine optimization-difficulty confound (rather than a
   representation-theoretic one) would show up as a NON-monotone or
   noisy `k=d_min−1` curve across seeds rather than a clean near-chance
   plateau — the seed count at this cell (2, escalating to 5 if
   ambiguous, §1.4.2) is sized to catch this distinction, not just
   confirm a single run.
4. **The whole-matrix-rank trivially-full trap (Task E §4's precedent) —
   why the restricted metric, not the raw one, is primary.** Stated
   directly in §1.4.1 above; re-flagged here because it is the single
   most consequential instrument-choice decision in this design and the
   validation verdict specifically named it as a required self-attack
   item.
5. **The learned-embedding case (Option B, emergent representation, no
   pinned reference) is explicitly deferred, not tested here.** This
   design ONLY tests whether the model can be MADE to align with a
   PINNED faithful representation under direct supervision (§1.4). It
   does NOT test whether a model trained on some OTHER, unsupervised or
   weakly-supervised group-task objective would SPONTANEOUSLY discover a
   faithful representation without ever seeing `rho_G` during training —
   that is a strictly harder, more interesting claim, correctly scoped
   out per the validation verdict's explicit Option-A/Option-B split.
6. **The C2 vector-ablation control is deferred (§1.4.2), following Task
   D's own precedent exactly** — but this means Stage 1's CONFIRM/FALSIFY
   verdict, on its own, cannot yet distinguish "the MATRIX structure is
   load-bearing" from "ANY sufficiently-parameterized state of dimension
   `≥d_min(G)²`-ish would do the same thing." Flagged, not resolved here
   — the Stage-1b/Stage-2 follow-on (§1.11) is where this gets closed,
   exactly mirroring how Task D's own Stage-1 gate left this open until a
   dedicated, properly param/state-matched control was designed
   separately.
7. **Does the direct block-diagonal-embedded cosine-loss training design
   (§1.4) leave GENUINE orthogonal gauge freedom, or does cosine loss's
   rotation-sensitivity already pin the basis during training (leaving
   only Task-E-style SCALE freedom, not a full `Q`)?** Cosine similarity
   between flattened matrices is scale-invariant but NOT
   rotation-invariant in general — so DIRECT supervision against a FIXED
   embedded target plausibly already pins the model toward THAT specific
   basis during training, in which case §1.3.2's full orthogonal-`Q`
   degauging machinery is a CONSERVATIVE superset (harmlessly recovers
   `Q_hat≈I` if training already pinned the basis) rather than a
   necessary correction. This is deliberately the SAFE default (verified
   in §1.3.2 to correctly reject a genuine rank-deficient corruption even
   with the full `Q` freedom available, i.e. the extra freedom does not
   appear to weaken the guard in the tested case) — but a full `Q` fit
   has more free parameters than a scale-only fit, which could in
   principle make the guard LAXER on messier real-checkpoint failure
   modes than the clean toy case tested. **Recommend gate 1's calibration
   cells explicitly check whether `Q_hat≈I` (up to noise) on real trained
   checkpoints; if consistently true, the sweep's PRIMARY decision metric
   can safely simplify to Task E's tighter scale-only degauging, with
   full-`Q` degauging kept only as a robustness cross-check** — not
   resolved by this Rev 0, explicitly flagged for gate 1 to determine
   empirically rather than assumed either way.
8. **`h=32` (halved from Task D/E's `h=64`) is a disclosed,
   calibration-pending economization, not a validated choice (§1.4,
   §1.6).** If the calibration cells (gate 1) show `h=32` cannot reach
   the convergence band Task D/E's own precedent predicts (mirroring
   `CLAUDE.md`'s calibration-first rule catching exactly this kind of
   silent-ceiling risk), the design reverts to `h=64` and the cost table
   (§1.6) is recomputed BEFORE the 45-cell remainder of the sweep
   launches — not discovered after the fact.
9. **Word-length held-out generalization (`L∈{9..16}`) may be a WEAK test
   for the small groups in this family.** For S3 (`|G|=6`) and S4
   (`|G|=24`), a random walk of length `L_train_max=8` over a 3-4-element
   generating set very likely already reaches most or all of the group
   (a mixing-time argument, not verified numerically in this Rev 0) —
   meaning "held-out length" mostly tests whether the model correctly
   handles LONGER compositions of ALREADY-SEEN target elements (a
   genuine, if narrower, compounding-error/state-tracking-depth test)
   rather than testing generalization to genuinely NEW group elements the
   way it does for A5/S5/A6 (larger groups, slower mixing). This is a
   real, disclosed asymmetry in the strength of the held-out claim across
   the family's dimension range, not fixable without either shrinking
   `L_train_max` for small groups (changing the family's uniformity) or
   accepting the asymmetry — accepted here, flagged for the write-up's
   limitations section regardless of Stage 1's verdict.

---

### 1.10 What this design does and does NOT do

**Does:** pins five reference representations, constructed and
NUMERICALLY VERIFIED (orthogonal, correct order, faithful, §1.3) rather
than asserted; specifies a Procrustes/scale degauging pipeline with a
mandatory fit/eval split, NUMERICALLY VERIFIED to recover the true gauge
under noise and to provably fail to rescue a rank-deficient corruption
(§1.3.2); reuses Task D/E's exact bespoke architecture and blank-out/
force-rank machinery with exactly one, structurally-necessary addition
(positional embeddings, §1.4); generalizes Task E's subspace-restriction
methodology to a task family with no single fixed target operator
(§1.4.1); pins a cost table derived from two independently-cited,
verified rate anchors (§1.6), correcting an unverifiable figure in the
validation verdict's own cost line; pre-registers CONFIRM/FALSIFY/
INCONCLUSIVE with the marquee S4-vs-A5 dissociation as an explicit,
checkable comparison, not just prose (§1.5).

**Does NOT do:** run any GPU cell (Rev 0 is design-only, zero GPU spent);
resolve whether Stage 1's direct-supervision training design leaves
genuine orthogonal gauge freedom or only scale freedom (§1.9 item 7,
explicitly deferred to gate 1's empirical check); include a param-matched
vector-ablation control (§1.4.2, §1.9 item 6, deferred per Task D's own
precedent); include PSL(2,7) (§1.2, dropped with reasons given); resolve
whether `h=32` is an adequate economization (§1.9 item 8, calibration-
gated); test Option B (emergent, unsupervised representation discovery,
§1.9 item 5); test multi-hop reasoning over the recovered representations
(that is Stage 2, gated on this result, §1.11).

---

### 1.11 Sequencing

design (this doc, Rev 0) → attack round 1 (§1.13, NEEDS-REVISION) → Rev 1
(§1.14, this revision) → attack round 2 (pending) → iterate to
DESIGN-CLEARED-FOR-BUILD (per this program's standing gauntlet discipline,
`HEAD_TO_HEAD_DEMO_DESIGN.md`'s own §1.11 precedent) → build (new code
needed: `GroupWordEncoder` — `BindingEncoder` + positional embedding,
§1.4; the word-sampling grammar per group, §1.4, incl. the query-coverage
guard + its negative test; the block-diagonal embedding of `rho_G`;
`verify_option_a_readout.py`'s functions imported directly into the eval
harness, not re-implemented; the β∈[0,2] `fla` smoke, §1.6; the 30 GPU-h
hard-abort enforcement wrapper, §1.6's circuit breaker, CA1-m2 fix Rev 1)
→ independent
build audit (separate agent, `CLAUDE.md`'s "the implementer does not
review their own work" rule) → launch alongside/after the head-to-head
demo's own rung-1 (GPU allocation TBD at build time, this campaign's own
dedicated 30 GPU-h ledger, §1.6) → harvest → Stage-1 verdict (§1.5) → **IF
CONFIRM or diagnosed-INCONCLUSIVE:** Stage 2 design addendum (multi-hop
word problems — composing SEVERAL separately-recovered group elements
into a further relational query, ~50-120 GPU-h per the validation
verdict's re-derived estimate, itself gated through the SAME
attack→build→audit gauntlet) → **IF FALSIFY:** write up as a companion
negative to Task D/E, closing this line, per §1.1's pre-registered framing.

---

### 1.12 Reproducibility pointers

- **This design's own executed verification:**
  `matrix-thinking/capability_separation/verify_option_a_readout.py`
  (repo-committed, numpy + stdlib only, deterministic `RNG_SEED=20260709`,
  §1.3's exact output reproduces on re-run).
- **CA1-F1/CA1-M3 fix's own executed verification (Rev 1):**
  `matrix-thinking/capability_separation/coverage_calibration.py`
  (repo-committed, numpy + stdlib only, deterministic `RNG_SEED=20260710`,
  §1.3.3/§1.4/§1.4.1's query-coverage and fit-set-diversity tables both
  reproduce on re-run).
- **CA1-M1 fix's own executed verification (Rev 1):**
  `matrix-thinking/capability_separation/spearman_null_calibration.py`
  (repo-committed, numpy-free stdlib only, exact enumeration, no RNG —
  §1.3.3/§1.5's M1 p-values reproduce on re-run).
- **CA3-M1(c) fix's own executed verification (Rev 3):**
  `matrix-thinking/capability_separation/marquee_power_simulation.py`
  (repo-committed, numpy+scipy, deterministic `RNG_SEED=20260711` —
  §1.4.2.1's bare-CI sanity check and the `n=5`/`n=7` TOST power tables
  both reproduce on re-run).
- **CA3-m3 fix's own executed verification (Rev 3):**
  `matrix-thinking/capability_separation/coverage_calibration.py`'s
  `verify_centralizer_dims` (same file/RNG as the CA1-F1/CA1-M3 entry
  above — §1.4.1's centralizer-dimension table reproduces on re-run).
- **Reused architecture:** `matrix-thinking/chapter2/model_v4.py`
  (`BindingEncoder`, `MatrixMemoryModel.encode`/`.unbind` pattern —
  `.unbind` itself NOT reused, no query mechanism in this task, §1.4).
- **Reused rank machinery:** `matrix-thinking/chapter2/rank_utils.py`
  (`truncate_to_rank`, `effective_rank`, `stable_rank` — all three reused
  verbatim, applied to the restricted operator `A(w)`, §1.4.1, not raw
  `Z`).
- **Reused blank-out test:** `matrix-thinking/chapter2/run_task_d.py`
  (`smoke()` step [5], lines 199-214 — the `torch.autograd.grad`
  leaf-detach + `inspect.signature` method).
- **Generalized subspace-restriction precedent:**
  `matrix-thinking/chapter2/analyze_zdump.py`
  (`entity_subspace`, `block_decompose`, `principal_angles_deg` — cited
  for methodology/precedent per §1.4.1; NOT imported verbatim, since this
  design's `U` derivation (covariance-SVD over a word sample) differs
  from Task E's (SVD of a single `z_ideal`), as explained in §1.4.1).
- **Rate anchors:** `EXPERIMENT_LOG.md:1556` (Task D total),
  `EXPERIMENT_LOG.md:1619-1625` (Task D idle/refill-excess breakdown),
  `TASK_D_PREREGISTRATION.md` §7 (Task D Stage-1 estimate),
  `EXPERIMENT_LOG.md:2230` and `TASK_E_FINDINGS.md` §10 line 710 (Task E
  80K-round rate), `STATE.md:984` (independent restatement of the Task D
  total).
- **Gate/cost-discipline precedent:** `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md`
  §1.6-§1.9 (cost-table format, self-attack register conventions, the
  numerically-executed-formula bar this design's §1.3 meets).
- **DeltaProduct citation (β-smoke, §1.6):** Siems et al., DeltaProduct,
  arXiv:2502.10297; Grazzi et al., arXiv:2411.12537 (ICLR 2025 oral,
  `allow_neg_eigval`/negative-eigenvalue state-tracking mechanism).
- **New, not yet built:** `GroupWordEncoder` (positional-embedding
  extension of `BindingEncoder`); the per-group word-sampling grammar +
  query-coverage guard + its negative test **+ the per-batch-fixed-`L`
  batching sampler** (CA2-M2 fix, Rev 2 — §1.4's third pinned extension;
  the `L=1`/`L=16` forward+backward smoke item, §1.4); the block-diagonal
  `rho_G` embedding; the β∈[0,2] `fla` smoke harness; **the 30 GPU-h
  hard-abort enforcement wrapper** (§1.6's circuit breaker — projects the
  58-cell sweep's cost from the calibration cell's real per-cell rate and
  mechanically hard-aborts before exceeding the dedicated cap; CA1-m2
  fix, Rev 1 — Rev 0 designed this mechanism in §1.6 but never itemized
  it in this build list); **the retry-once coverage-check orchestration**
  (CA3-m1 fix, Rev 3 — §1.4.1 step 4 pins the mechanism (resample the
  `N=50` eval set ONCE with `cell_seed+1` on a coverage-bar miss, log
  both draws, fail hard on a second consecutive miss) and the
  eval-set-diversity-floor variant that uses the SAME retry-once
  machinery (§1.4.1 step 4's `eval_set_diversity_check` floor), but
  neither was itemized as its own build item before this revision — both
  require the production eval harness, which does not exist until build,
  §1.11; treat as a real, load-bearing build-time obligation, not
  settled evidence, the same standing caveat §1.14/§1.16 already applied
  to the synthetic-injection acceptance test and the batching scheme);
  **the live escalation-budget guard** (CA4-M1(b) fix, Rev 4 — §1.6's
  hard-abort re-check wired INTO the escalation-trigger code path for
  BOTH variants (the general escalation-to-5 trigger, §1.4.2, and the
  marquee `n=7` trigger, §1.4.2), enforcing the pre-registered yield
  order (marquee outranks general; general escalations pinned by
  ascending group `|G|`) and the budget-denied report status, **plus its
  negative test**: a simulated over-budget projection (actual spend +
  committed escalations pushed artificially close to the 30 GPU-h cap)
  must deny the correct escalation(s) in the pinned order and log each
  denial with a `budget-denied` status, not silently skip or over-spend —
  this is a real, load-bearing build-time obligation, not settled
  evidence, until the negative test is executed against the built
  harness).

---

**QUEUE (STATE.md, appended per this design's commit):** Design Rev 3
committed (this commit, following the CA2-m5-established convention of
updating this footer every revision) → attack round 4 next.

---

### 1.13 ATTACK ROUND 1 VERDICT (independent fresh-eyes agent, 2026-07-08): NEEDS-REVISION

Recorded per the gauntlet-bookkeeping hard rule before dispatching Rev 1.
The round re-executed `verify_option_a_readout.py` verbatim (byte-exact
match) AND independently re-implemented the constructions via DIFFERENT
methods (deleted-permutation reps for S3/S4; the binary-icosahedral
QUATERNION construction for A5 — element-order distribution
{1:1,2:15,3:20,5:24} matches A5's conjugacy classes exactly;
Gram-Schmidt hyperplane bases for S5/A6; character-theoretic minimality
re-derivations for all five). Findings binding on Rev 1:

**FATAL-AS-WRITTEN:**

- **CA1-F1 — the query-coverage threshold is mathematically impossible
  for 2 of 5 groups.** §1.4's gate bar "≥200 distinct elements for
  A5/S5/A6" contradicts the doc's own group table: |A5|=60, |S5|=120 —
  a group cannot yield more distinct elements than it has. As
  pre-registered, gate 3's mandatory coverage check can never pass for
  A5 (the marquee dissociation partner!) or S5, silently blocking the
  C5 control exactly where it matters most. → Rev 1: replace with a
  family-consistent percentage-of-|G| bar (e.g. ≥80% for S3/S4/A5, a
  calibrated lower % for S5/A6 at their larger orders), re-run gate 3's
  negative test against the corrected numbers.

**MAJOR:**

- **CA1-M1 — M1's Spearman bar is statistically weaker than it reads.**
  Exact null (brute-force over 5!=120 permutations, x=[2,3,3,4,5] with
  the S4/A5 tie): P(ρ≥0.8 | null) = 8/120 ≈ 6.67% — NOT conventionally
  significant alone at α=0.05 (and ρ=1.0 is impossible; the tie caps
  ρ at 0.9747). Mitigated by CONFIRM requiring M1 AND M3 AND marquee
  (M3's per-group paired-seed causal test is the workhorse). → Rev 1:
  disclose the exact one-sided permutation p-value; state explicitly
  that M1 alone is underpowered/corroborating-only and the claim rests
  on M3+marquee.
- **CA1-M2 — cost-table ceiling unreconciled.** Itemized rows sum to
  ≤≈17.34 GPU-h (50 cells×0.3=15.0 + contingency 1.35 + β-smoke <1.0)
  but the stated raw total reads "≈17.4-18.3" — the upper bound carries
  ~1 GPU-h with no itemized source. Safety conclusion unchanged (39-42%
  margin under the 30 GPU-h cap) but the arithmetic must reconcile. →
  Rev 1: itemize or trim the range.
- **CA1-M3 — degauging pipeline parameters unpinned for the REAL runs.**
  Only "N≥50 words" is stated; no n_fit/n_eval ratio; gate 1's
  real-checkpoint validation has no concrete acceptance criterion (no
  ground-truth Q exists on a real checkpoint). → Rev 1: pin the split
  (and its sampling rule) + define gate 1's acceptance test (e.g.
  synthetic-injection: plant a known-Q conjugated ideal state in the
  real pipeline's shapes and require recovery + the rank-deficient
  negative to hold there).

**MINOR:** CA1-m1 — "ONLY change to Task D's encoder" undercounts the
input-embedding change (single generator-index embedding replaces the
[key;value] concat — disclosed elsewhere in the same section, not
counted here). CA1-m2 — the 30 GPU-h hard-abort enforcement wrapper is
not itemized in §1.12's build list.

**Verified clean this round (attack's own list):** all five reference
representations (independent constructions, orthogonality, faithfulness,
order, char-theory minimality); the degauging guard's teeth under fresh
adversarial instances (rank-deficient corruption NOT rescued:
mean_cos=0.663, sv-gap 11.06→1.26) plus a NEW fit-set-bias test (no
degradation, 8 seeds); the corrected cost anchors (Task D ~76 GPU-h at
EXPERIMENT_LOG:1556 confirmed; the 11.9 figure confirmed to belong to
the unrelated coherence-dose-response wave); the cell count (5×10=50);
the OWN-30-GPU-h-hard-abort circuit-breaker design (correctly declines
the impossible inherited 10× bracket); the S4-vs-A5 training-coverage
confound EMPIRICALLY CLEARED (both reach 100% element coverage by
L=5-6; a mild A5 visit-frequency imbalance at L=8, max/min≈6.8×, noted
as a footnote, non-blocking); gauge-freedom scoping; dim-2 disclosure;
β-smoke non-load-bearing status; Stage-2 gating; ledger separation.

---

### 1.14 REV 1 CHANGES — finding → resolution map

Every §1.13 finding, mapped to its exact Rev 1 resolution. Nothing below
is a disclaimer-only close — each row points at new or rewritten design
text (or a new, executed, repo-committed script), not a footnote.

| Finding | Resolution (Rev 1) | Where |
|---|---|---|
| **CA1-F1** (FATAL) — query-coverage bar "≥200 distinct elements for A5/S5/A6" impossible (`|A5|=60`, `|S5|=120`) | Replaced with a per-group, NUMERICALLY CALIBRATED percentage-of-`|G|` bar: S3 ≥80%, S4 ≥70%, A5 ≥45%, S5 ≥25%, A6 ≥10% — each the largest 5%-step clearing the healthy-sampler 1st-percentile floor (≥1-element margin) while strictly exceeding a pathological (`L=1`) undersampled sampler's DETERMINISTIC ceiling. Derived via a new Monte Carlo script (`coverage_calibration.py`, 20,000 trials/group, executed this session), NOT the naive uniform-80% default — that default is explicitly shown to fail for S4 (p1=75%) and A5 (p1=46.7%) and disclosed as a deviation, not silently adopted. Gate 3's negative test re-specified against the corrected bars for all 5 groups. | §1.3.3 (new), §1.4, §1.7 gate 3 |
| **CA1-M1** (MAJOR) — M1's Spearman bar statistically undisclosed | Exact one-sided permutation p-values disclosed (`P(ρ≥0.8|null)=8/120≈6.67%`, `P(ρ≥0.9|null)=2/120≈1.67%`, tie cap `ρ≤0.9747`), independently re-derived from scratch this Rev-1 pass (`spearman_null_calibration.py`, exact enumeration) — reproduces §1.13's own figures exactly. M1 stated explicitly as corroborating-only; CONFIRM's statistical weight attributed to M3 + the marquee dissociation check. `ρ≥0.9` evaluated and EXPLICITLY DECLINED (not silently kept at 0.8): achievable but brittle — the next-highest achievable ρ after the perfect-ordering maximum is 0.8721, an ~0.10 cliff, so a 0.9 bar would fail on any single ordinary-noise misordering, undermining M1's corroborating role. | §1.3.3 (new), §1.5 |
| **CA1-M2** (MAJOR) — cost-table ceiling unreconciled (~1 GPU-h unaccounted) | β-smoke row itemized from vague `<1.0` to a specific **0.90 GPU-h** (0.05 smoke + 0.85 Fig.5-reproduction run); raw total now equals the itemized sum exactly (`15.0+1.35+0.90=17.25` GPU-h, was "≈17.4-18.3" with no source for the extra ~1). Margin recomputed to a single reconciled figure, **≈42.5%** (was an unreconciled "≈40-42%" range). | §1.6 |
| **CA1-M3** (MAJOR) — degauging pipeline's real-run n_fit/n_eval split and gate-1 acceptance test unpinned | Split pinned: 60/40 of the `N=50` eval words (`n_fit=30`/`n_eval=20`, disjoint by index), plus a fit-set diversity floor (`≥min(3·d_min(G), ⌊0.8·|G|⌋)` distinct elements, executed and cleared by ≥1 element at p1 for all 5 groups, `coverage_calibration.py::fit_set_diversity_check`). Gate 1's acceptance test pinned: a synthetic-injection check on the PRODUCTION eval harness (known `(Q_true,c_true)`-conjugated `rho_G` trajectory, §1.3.2's exact noise/tolerance numbers) requiring (a) recovery (`mean_cos>0.95`, `recovered_frac@0.9≥0.9`) and (b) the rank-deficient negative control (`Q_def`) to hold in the SAME harness (`mean_cos` ≥0.3 below (a), `recovered_frac@0.9<0.5`) — run BEFORE any real checkpoint's degauging output is trusted. | §1.4.1 step 4, §1.7 gate 1 |
| **CA1-m1** (minor) — "ONLY change to Task D's encoder" undercounted the input-embedding delta | Corrected to "TWO required extensions": (1) the positional embedding, (2) the generator-index input embedding replacing Task D's `[key;value]` concat — both stated as direct, minimal consequences of the same task-family change, not independent design choices. | §1.4 |
| **CA1-m2** (minor) — 30 GPU-h hard-abort enforcement wrapper not itemized in the build list | Added to §1.12's "New, not yet built" list and to §1.11's sequencing parenthetical. | §1.11, §1.12 |

**Nothing from §1.13's "Verified clean" list was touched** — the five
reference representations, the degauging guard's teeth (rank-deficient
corruption + fit-set-bias tests), the corrected cost anchors' provenance,
the cell count, the OWN-30-GPU-h circuit-breaker design, the S4-vs-A5
training-coverage-confound clearance, gauge-freedom scoping, dim-2
disclosure, β-smoke non-load-bearing status, Stage-2 gating, and ledger
separation all stand as attack round 1 verified them, unmodified in Rev 1.

**What this revision could not fully close (flagged, not papered over):**
CA1-M3's synthetic-injection acceptance test is pinned at the DESIGN
level (exact tolerances, exact procedure) but has not been RUN — it
requires the production eval harness code, which does not exist until
build (§1.11); an attack-round-2 agent should treat gate 1's acceptance
test as a real, load-bearing build-time obligation, not settled evidence,
the same way CA1-M2's cost table was until this revision itemized it.

---

### 1.15 ATTACK ROUND 2 VERDICT (independent fresh-eyes agent, 2026-07-08): NEEDS-REVISION

Recorded per the gauntlet-bookkeeping hard rule before dispatching Rev 2.
All §1.13 resolutions verified GENUINELY resolved; all three companion
scripts re-run BYTE-EXACT and independently re-implemented (different
algorithms, different RNG, N=400,000/group MC vs the design's 20,000 —
every §1.3.3/§1.5/§1.6 numeric reproduced; false-block probabilities
computed: S3≈0%, S4≈0.068%, A5≈0.075%, S5≈0.135%, A6≈0.002%; joint
family exposure ≈0.28% shared-draw / ≈2.75% per-cell-draw; fit-floor
0/400,000 failures, Wilson-CI upper ≈0.001%; Spearman enumeration exact
via independent recursive generator + scipy cross-check). Findings
binding on Rev 2 (narrow scope — all cheap; round expects a fast Rev 2
clear):

**MAJOR:**

- **CA2-M1 — A5's coverage-calibration generator is CLASS-unverified.**
  A5's order-5 elements split into two non-conjugate classes (cycle
  type (5), single odd-length part → splits; the same fact behind A5's
  two inequivalent 3-dim real irreps). `coverage_calibration.py`'s
  permutation stand-in never verifies its 5-cycle corresponds to the
  SAME class as the real icosahedral 2π/5 generator in
  `verify_option_a_readout.py`. Empirically: swapping to the other
  class (g5²) drops mean coverage 33.79→31.76 and p1 28→**26 — below
  the pinned bar 27** — false-block 0.077%→1.274% (16×). Only A5 has
  this ambiguity among the five. → Rev 2: verify the correspondence
  explicitly (trace/character check between the stand-in and the
  matrix generator) or re-calibrate A5's bar with the verified class.
- **CA2-M2 — variable-per-sample word length has NO batching spec.**
  `BindingEncoder.forward` requires fixed-shape (B,K,h), no padding
  mask anywhere in the reused code; Task D always used fixed K per run
  (never exercised variable lengths); this task samples L~Uniform{1..8}
  train / {9..16} eval PER SAMPLE. The builder would have to invent
  padding+key_padding_mask vs per-batch-fixed-L — a real methodological
  choice (the CLAUDE.md nn.MultiheadAttention mask gotcha class).
  → Rev 2: pin the scheme as a disclosed THIRD architectural extension
  (recommend per-batch-fixed-L for zero mask-code risk — batch groups
  share one L, L varies across batches; justify either way).

**MINOR:** CA2-m1 — pin the coverage-check draw cadence (once-per-group
vs fresh-per-cell; Task D's own seed+10_000 precedent implies per-cell
→ ~2.75% operative sweep exposure) + add a disclosed retry-once rule
symmetric with the existing fit-diversity redraw guard [= the round's
single highest-value change]. CA2-m2 — no diversity floor on the
n_eval=20 scoring subset (empirically fine, unguarded). CA2-m3 —
fit-floor JUSTIFICATION imprecise (raw d² equation-count doesn't pin Q
— the null space is ρ(g)'s centralizer; the real mechanism is
Schur/irreducibility w/ ~2 generic elements sufficing exactly, plus
noise margin anchored to §1.3.2's 4.7×d toy ratio) — reword, keep the
number. CA2-m4 — gate 5's "flagged for attack round 1" cross-reference
was never answered: close the loop (round 2 adjudicated the
single-token simplification SAFE — C2 deferred, no matched-comparison
invariant at risk). CA2-m5 — STATE queue line stale (coordinator note).

**Verified clean this round:** all §1.14 resolutions; all three scripts
byte-exact + independently reproduced; cost anchors at exact cited
lines; ρ≥0.8 keep-decision sound; injection-spec deferral honest and
correctly sequenced; no stale 200-element/80% leftovers; h=32
disclosure ×4 consistent; chapter2 citations all verified; S4-vs-A5
coverage confound independently reproduced; self-attack item 7's
rotation-sensitivity math verified.

---

### 1.16 REV 2 CHANGES — finding → resolution map

Every §1.15 finding, mapped to its exact Rev 2 resolution. Narrow scope
per the dispatch instruction — nothing below relitigates material §1.13
or §1.15 already verified clean; each row points at new or rewritten
design text (or a new, executed, repo-committed script change), not a
footnote.

| Finding | Resolution (Rev 2) | Where |
|---|---|---|
| **CA2-M1** (MAJOR) — A5's coverage-calibration generator never verified against the SAME conjugacy class as the real icosahedral generator; attack's own parenthetical trace guess (`φ-1`/`-φ`) explicitly flagged unverified | New `verify_a5_generator_class()` in `coverage_calibration.py` (imports `verify_option_a_readout.py`'s real A5 generators directly, no re-implemented geometry) runs an EXPLICIT, generator-matched simultaneous-BFS isomorphism check across the full 60-element group. Character cross-check computed directly from the rep matrices first (`trace(g5_a5)=φ`, `trace(g5_a5²)=1-φ`) — confirms the attack's own parenthetical guess did NOT match, validating the instruction to verify rather than trust it. Result: the CURRENT stand-in (`g5`, unmodified) IS the correct class — exhaustively proven (all 60 elements, zero inconsistencies) via `g5↔g5_a5`, `g3↔g3_a5⁻¹` (an immaterial inverse-labeling, since the symmetric generating set `{g5,g5⁻¹,g3,g3⁻¹}` already contains both `g3_a5` and `g3_a5⁻¹`); both `g5_a5²` (wrong-class) controls FAIL to close to the full group under either labeling. **§1.3.3's numbers do NOT move** — a full re-run (this session) reproduces STEP 0-3 byte-identical. Executed output pasted in full. | §1.3.4 (new) |
| **CA2-M2** (MAJOR) — variable-per-sample word length has no batching spec; builder would have to invent padding+mask vs per-batch-fixed-`L`, landing in the `CLAUDE.md` `nn.MultiheadAttention` mask-gotcha class | Pinned as the disclosed THIRD architectural extension: **per-batch-fixed-`L`** (each batch samples one `L` from the pinned distribution; all episodes in that batch share it; `L` varies across batches). Justified explicitly against padding+mask (zero new mask code vs. untested plumbing added to a class this design otherwise reuses byte-for-byte, for a packing-efficiency benefit this design's small `L` range and sweep-dominated cost structure don't need). Consequence disclosed: per-batch `L`-homogeneity is benign because episodes are i.i.d. GIVEN `L`, and the `L`-distribution is preserved in expectation across training. Smoke item added: `L=1` and `L=16` both forward/backward cleanly. | §1.4 |
| **CA2-m1** (minor) — coverage-check draw cadence unpinned (once-per-group vs fresh-per-cell) | Pinned: **fresh-per-cell** (Task D's own `seed+10_000` precedent). Operative per-cell exposure **≈2.75%** stated from attack round 2's own executed `N=400,000`/group figures (not re-derived — already-executed evidence cited directly). Retry-once rule added, symmetric with the existing fit-set redraw guard: resample once with `cell_seed+1` on failure, log the retry, fail hard on a second consecutive miss. | §1.4.1 step 4 |
| **CA2-m2** (minor) — no diversity floor on the `n_eval=20` scoring subset | New `eval_set_diversity_check()` in `coverage_calibration.py`, bar `≥min(2·d_min(G), ⌊0.6·|G|⌋)` (deliberately weaker than the fit floor — the eval set does no fitting). Executed: every group clears by ≥2 elements at p1 (S3 margin 2, S4 margin 4, A5 margin 8, S5 margin 6, A6 margin 7). | §1.4.1 step 4 |
| **CA2-m3** (minor) — fit-floor justification imprecise (implied naive `d²`-equation counting) | Reworded: for a single fit element, the linear system's null space is `ρ(g)`'s own centralizer (not zero) — Schur's lemma + irreducibility (all five reps verified irreducible + real-type, §1.3.1) is the actual mechanism pinning `Q` to scalars over the full group image, with ~2 generic elements generically sufficing. The `3·d_min(G)` floor is reframed as a noise-margin choice anchored to §1.3.2's own 4.7×`d` toy ratio, not an equation-sufficiency requirement. The number is unchanged — only the justification. | §1.4.1 step 4 |
| **CA2-m4** (minor) — gate 5's "flagged for attack round 1" cross-reference never answered | Closed: attack round 2 (§1.15) adjudicated the single-PI-signoff-token simplification SAFE (the C2 vector-ablation control is explicitly deferred, so there is no second matched-comparison arm needing a second token the way `HEAD_TO_HEAD_DEMO_DESIGN.md`'s three-way comparison does) — written directly into the gate text. | §1.7 gate 5 |
| **CA2-m5** (minor) — STATE queue footer stale (still read "Rev 0 committed → attack round 1 next" after two revisions and two attack rounds) | Updated to "Rev 2 committed → attack round 3 next." | §1.12 footer |

**Nothing from §1.15's "Verified clean this round" list was touched** —
all §1.14 resolutions, all three companion scripts' byte-exactness and
independent reproduction, the cost-anchor citations, the `ρ≥0.8`
keep-decision, the injection-spec deferral sequencing, the absence of
stale 200-element/80% leftovers, the `h=32` disclosure's ×4 consistency,
the chapter2 citations, the S4-vs-A5 coverage-confound reproduction, and
self-attack item 7's rotation-sensitivity math all stand as attack round
2 verified them, unmodified in Rev 2.

**What this revision could not fully close (flagged, not papered over):**
the per-batch-fixed-`L` batching scheme (CA2-M2) is pinned at the DESIGN
level — mechanism, justification, consequence, and smoke item — but, like
CA1-M3's synthetic-injection acceptance test before it, has not been RUN;
it requires `GroupWordEncoder`, which does not exist until build (§1.11).
An attack-round-3 agent should treat the `L=1`/`L=16` smoke item as a
real, load-bearing build-time obligation, not settled evidence. The
CA2-m1 retry-once rule is likewise pinned but unexecuted for the same
reason (no production coverage-check harness exists yet to run it
against) — flagged for the same build-time treatment.

---

### 1.17 ATTACK ROUND 3 VERDICT (independent fresh-eyes agent, 2026-07-08): NEEDS-REVISION (narrow — 1 MAJOR)

Recorded per the gauntlet-bookkeeping hard rule before dispatching Rev 3.
All three companion scripts reproduced BYTE-IDENTICAL on fresh re-run;
§1.13/§1.14/§1.15 byte-integrity confirmed via git diff; all §1.15
resolutions verified genuinely resolved.

**THE A5 MATHEMATICAL STORY, settled (round 3's deep item):**
- `_simultaneous_closure` is a genuine ISOMORPHISM proof, not mere
  homomorphism-compatibility (inductive well-definedness over the full
  60-element closure + equal-cardinality bijectivity — the standard
  coset-enumeration argument). The 5A/5B split is real (g conjugate to
  g⁻¹ in A5, NOT to g²; traces φ vs 1−φ in the same irrep).
- The automorphism-invariance objection RESOLVED by brute force over
  all 120 τ∈S5: every τ with τg5τ⁻¹=g5² is odd (Out(A5)=Z/2 does swap
  5A/5B) but NONE fixes g3 up to inversion — **no automorphism maps
  the generating pair {g5,g3} to {g5²,g3}**. So CA2-M1's finding was
  REAL and the correct mechanism is the JOINT GENERATING PAIR changing
  the Cayley graph (different mixing statistics, 33.79 vs 31.76 mean
  coverage), not conjugacy class per se. The doc's fix verifies the
  right thing; the framing sentence should name the pair-level
  mechanism (folded into Rev 3 as a wording item).

**MAJOR (binding on Rev 3):**

- **CA3-M1 — the marquee S4-vs-A5 dissociation check is UNDERPOWERED
  at n=3 with no dedicated escalation trigger.** The check is a
  required AND-condition for CONFIRM (§1.5) and the design's entire
  dimension-vs-solvability point — yet simulated power at n=3
  (t(2)=4.303): a REAL 0.5-rank-unit S4/A5 gap is missed (CIs wrongly
  overlap) 71-97% of the time; even a 1.0-unit gap missed 11-80%. The
  §1.4.2 escalation trigger never fires in the dangerous regime (both
  groups clearing their own bars while differing from each other) —
  as written the control can almost never disconfirm dimension-tracking
  even if solvability were the true story. → Rev 3: UNCONDITIONAL n=5
  seeds for S4 and A5's k=d_min + unconstrained cells (default, not
  contingent; ~2.4 GPU-h, inside the 12.75 GPU-h margin) AND/OR a
  TOST-style equivalence test with a pre-registered margin instead of
  bare CI-overlap; re-run the power simulation at the chosen design
  and paste the table (executed, not asserted).

**MINOR:** CA3-m1 — §1.12 doesn't itemize the retry-once orchestration
as its own build item. CA3-m2 — post-retry exposure stated precisely:
correctly weighted Σ_g 10·p_g² ≈ 2.85×10⁻⁵ (S5-dominated), not just
"far below". CA3-m3 — fold the executed centralizer check (round 3
verified: every group's 2 pinned generators already reduce the
centralizer to exactly dim 1 = scalars) into the companion script —
converts §1.4.1's asserted "~2 generic elements suffice" into an
executed claim. CA3-m4 — one-sentence note that per-batch-L loss-scale
variation doesn't arise (single cosine loss per episode, no
token-count accumulation; round 3 adjudicated SAFE). CA3-m5 —
cosmetic: "same forward pattern" overstatement; non-integer bar
clarifying note; stray sentence fragment in §1.5 M2.

**Verified clean this round:** the A5 isomorphism check (re-proven +
re-run, wrong-class controls fail); per-batch-fixed-L (loss-scale
concern adjudicated SAFE); retry-once + n_eval floor (re-run, ≥2
margins); Schur rewording (numerically EXECUTED beyond the doc's own
assertion); gate-5 closure; footer; byte-integrity of all records.

---

### 1.18 REV 3 CHANGES — finding → resolution map

Every §1.17 finding, mapped to its exact Rev 3 resolution. Narrow scope
per the dispatch instruction — nothing below relitigates material §1.13/
§1.15 already verified clean; each row points at new or rewritten design
text (or a new/extended, executed, repo-committed script), not a
footnote.

| Finding | Resolution (Rev 3) | Where |
|---|---|---|
| **CA3-M1** (MAJOR) — the marquee S4-vs-A5 dissociation check underpowered at `n=3` with no dedicated escalation trigger; simulated a REAL 0.5-unit gap missed 71-97% of the time, a 1.0-unit gap missed 11-80% | (a) UNCONDITIONAL bump to `n=5` for S4/A5's unconstrained and `k=d_min` cells (default, not contingent) — cell count `10→14`/group for S4/A5, `50→58` total, DERIVED cost delta `+2.4` GPU-h (`15.0→17.4` main-sweep row), new raw total `≈19.65` GPU-h, new margin `≈34.5%` (was `≈42.5%`), independently confirming attack round 3's own `~2.4` GPU-h estimate. (b) Bare CI-overlap REPLACED with a pre-registered Welch TOST equivalence test, margin `±0.5` rank-units (half the `d_min` ladder's unit spacing), justified as unpaired (S4/A5 share init-seed labels but train on entirely different data — pairing would require an unverified surviving-correlation assumption). (c) EXECUTED power simulation (`marquee_power_simulation.py`, new script) over a reconstructed noise grid (validated against round 3's own reported n=3 miss-rate ranges): `n=5`+TOST clears **100%** power against a real 1.0-unit gap across the entire grid (≫ the 70% bar) — no escalation to `n=7` required on the power-of-interest axis; the equivalence-DECLARATION side (`gap=0`) is noise-sensitive (`~100%→~33%` across the grid) and flagged, not papered over. (d) A marquee-specific `n=7` escalation trigger added for the case TOST itself returns INCONCLUSIVE (§1.4.2), distinct from the general per-cell-type trigger; §1.5's CONFIRM/marquee text rewritten to reference the TOST procedure directly. | §1.4.2, §1.4.2.1 (new), §1.5, §1.6 |
| **CA3-m1** (minor) — the retry-once coverage-check orchestration (+ the eval-diversity-floor variant reusing it) not itemized as its own build item | Added to §1.12's "New, not yet built" list, flagged as a real, load-bearing build-time obligation per the same standing convention §1.14/§1.16 already applied to the synthetic-injection test and the batching scheme. | §1.12 |
| **CA3-m2** (minor) — post-retry exposure stated only as "far below ≈2.75%," not precise | Stated exactly: `Σ_g 10·p_g² ≈ 2.85×10⁻⁵`, S5-dominated (`~64%` of the total) — plus a disclosed Rev-3-consistency note giving the post-cell-bump figure (`≈3.26×10⁻⁵` using 14 for S4/A5), still S5-dominated, still negligible. | §1.4.1 step 4 |
| **CA3-m3** (minor) — the "~2 generic elements suffice" centralizer claim asserted, never executed against the actual pinned generators | New `coverage_calibration.py::verify_centralizer_dims` (+ `_get_real_generator_pairs`, `_centralizer_null_dim`) computes the null-space dimension of the 2-pinned-generator commutator map for all 5 groups via stacked-Kronecker/SVD; EXECUTED output pasted (all 5 groups: null_dim=1, well-separated singular-value gaps 0.61-1.73 vs ~1e-16). §1.4.1's text updated from asserted to executed. | §1.4.1 step 4, `coverage_calibration.py` |
| **CA3-m4** (minor) — no explicit note that per-batch-L doesn't introduce loss-scale variation | One-sentence note added: the training loss is a single scalar cosine distance per episode (no per-token/per-position accumulation), so no loss-scale term grows with `L`. | §1.4 |
| **CA3-m5** (minor) — "same forward pattern" overstatement; non-integer STEP-2 bar column unclarified; §1.5 M2's stray sentence fragment | "Same forward pattern" corrected to "same core module stack, EXTENDED by the three deltas" with the specific reason (new positional-embedding addition + replaced input embedding both change the actual forward computation). Non-integer-bar clarifying note added (STEP 2's raw fractional column vs. the operative integer `⌈·⌉` threshold already used in §1.4's table). M2's CONFIRM criterion reworded into one clean sentence, threading through S4/A5's new `n=5` default alongside S3/S5/A6's `n=3`. | §1.4, §1.3.3, §1.5 |
| **A5 wording item** (§1.17's own list, folded into Rev 3) — name the joint-generating-pair mechanism in the §1.3.4 framing sentence | New paragraph added: no automorphism of A5 maps `{g5,g3}` to `{g5²,g3}` (verified by round 3 over all 120 `τ∈S5`) — the correct mechanism is the pair-level Cayley-graph difference, not conjugacy class read in isolation; the isomorphism check itself was already verifying the right thing. | §1.3.4 |

**Nothing from §1.17's "Verified clean this round" list was touched** —
the A5 isomorphism check, per-batch-fixed-L's loss-scale adjudication
(now additionally given its own one-sentence note per CA3-m4, not
reversed), the retry-once + n_eval floor re-run, the Schur rewording,
gate-5 closure, the footer, and byte-integrity of §1.13/§1.14/§1.15/
§1.16/§1.17 all stand as attack round 3 verified them, unmodified in
Rev 3 (verified via `git diff` before this commit — see the commit
message).

**What this revision could not fully close (flagged, not papered
over):** the equivalence-DECLARATION side of the marquee TOST test
(`P(declare equivalence | gap=0)`) is genuinely noise-sensitive at
`n=5` — from `~100%` at the low end of the reconstructed noise grid down
to `~33%` at the high end (§1.4.2.1) — meaning that if the REAL
per-checkpoint restricted-rank noise lands toward the high end of this
design's plausible range, the marquee comparison may render
INCONCLUSIVE (triggering the CA3-M1(d) `n=7` escalation) even when the
true underlying story is "S4 and A5 genuinely track dimension together."
This is a safe (conservative, not misleading) failure mode, not a
correctness bug, but it is NOT eliminated by this revision — an
attack-round-4 agent should treat gate 1's calibration cells as the
first real opportunity to measure the ACTUAL per-checkpoint noise level
and confirm which end of the grid this design's real runs land on,
rather than assuming the favorable end. The retry-once orchestration
(CA3-m1) and the centralizer check's integration into a future
production eval harness (CA3-m3) remain pinned-but-unexecuted-in-
production for the same reason CA1-M3/CA2-M2 flagged before them: the
production harness does not exist until build (§1.11).

---

*(End §1 records. Rev 0 → §1.13 → Rev 1 → §1.15 → Rev 2 (§1.16) →
§1.17 → Rev 3 (§1.18). Rev 3 resolves CA3-M1 (marquee power, EXECUTED)
+ 5 minors + the A5 wording item. Attack round 4 next.)*

---

### 1.19 ATTACK ROUND 4 VERDICT (independent fresh-eyes agent, 2026-07-08): NEEDS-REVISION (narrow — 1 MAJOR, one-paragraph fix)

Recorded per the gauntlet-bookkeeping hard rule before dispatching Rev 4.
All four companion scripts byte-reproduced; SHA-256 of every working-tree
script/doc matches the 5852585 blobs; §1.13-§1.17 records byte-intact;
the TOST power core INDEPENDENTLY re-implemented via a different
statistical primitive (scipy ttest_ind on margin-shifted samples, 100K
trials, own seed) — matches the doc's table to ≤0.5pp at every cell
incl. the n=7 column. Everything §1.18 added verified clean: cell
recount 58 exact; 19.65/34.5% exact; TOST margin correctly pinned to
the CONTINUOUS effective_rank metric (discreteness foreclosed); scope
choice (bump only unconstrained+k=d_min) adjudicated CORRECT; centralizer
tol=1e-8 verified robust across 8 orders of magnitude (true nulls at
~1e-16 vs next-smallest 0.61-1.73); exposure figures 2.848e-5/3.258e-5
re-derived exactly; contingency row disjoint from bump costs (no
double-count).

**MAJOR (binding on Rev 4):**

- **CA4-M1 — the escalation triggers are uncosted and the hard-abort
  doesn't guard them.** The marquee n=7 trigger (Rev 3's own addition)
  costs +2.4 GPU-h (→22.05, 26.5% margin — fits, realistic case); but
  the GENERAL escalation-to-5 trigger (inherited since Rev 0, uncaught
  through 3 rounds) is also uncosted, and §1.6's circuit-breaker only
  projects the BASE 58-cell sweep once upfront. Literal worst case
  (every ambiguous-eligible cell type escalates + marquee n=7):
  19.65 + 2.40 + 12.60 = **34.65 GPU-h > the 30 GPU-h cap by 4.65**.
  A build agent has no instruction on budget-checking before launching
  escalation cells — a genuine stop-and-ask gap. → Rev 4: one clause
  in §1.6 (the hard-abort re-checks actual-spend-to-date + projected
  escalation cost BEFORE launching ANY escalation-triggered cells,
  general or marquee; pre-registered priority order for which
  escalations yield if budget is short — marquee outranks general) +
  one new §1.12 build item + disclose the worst-case arithmetic.

**MINOR:** stale-wording nit on A6's "will still run before launch"
(subsumed by STEP 0's executed closure check — clarify, don't re-run).

**Verified clean (what round 4 tried and failed to break):** TOST
target quantity; Welch-vs-paired justification; centralizer threshold;
contingency/bump disjointness; n_fit-floor × seed-bump orthogonality;
build-manifest completeness EXCEPT the CA4-M1 item.

---

*(End §1 records. Rev 0 → §1.13 → Rev 1 → §1.15 → Rev 2 → §1.17 →
Rev 3 (§1.18) → §1.19 NEEDS-REVISION (CA4-M1 escalation budget guard).
Rev 4 in progress — expected one-paragraph class.)*

---

### 1.20 REV 4 CHANGES — finding → resolution map

Every §1.19 finding, mapped to its exact Rev 4 resolution. One-paragraph-
class scope per the dispatch instruction — nothing below relitigates
material §1.13/§1.15/§1.17 already verified clean; each row points at
new or rewritten design text, not a footnote.

| Finding | Resolution (Rev 4) | Where |
|---|---|---|
| **CA4-M1** (MAJOR) — the escalation triggers (general escalation-to-5, §1.4.2, inherited since Rev 0; and the marquee `n=7` trigger, §1.4.2, CA3-M1(d)) are uncosted, and §1.6's circuit-breaker only projects the BASE 58-cell sweep once upfront — literal worst case `19.65 + 2.40 + 12.60 = 34.65 GPU-h > the 30 GPU-h cap by 4.65`; no instruction to budget-check before launching an escalation cell | New paragraph appended to §1.6's circuit-breaker discussion: the hard-abort now re-checks actual-spend-to-date + projected escalation cost BEFORE launching ANY escalation-triggered cell (general or marquee), not only once upfront against the base sweep. The worst-case arithmetic is disclosed explicitly (base `19.65` + marquee `n=7` `2.40` + general-max `12.60` `= 34.65`, independently re-derived from the cell table: `42` extra cells at `0.3` GPU-h/cell — `10` each for S3/S5/A6's 4 cell types 3→5/3→5/2→5/2→5, `6` each for S4/A5's 2 flanking cell types 2→5, matching attack round 4's own `12.60` figure). If the projection would breach the cap, a **pre-registered yield order** converts the overrun into a priority allocation instead of a silent over-spend: the marquee `n=7` escalation outranks all general escalations (it guards the central S4-vs-A5 dissociation claim, §1.5); general escalations are granted by ascending group `|G|` (S3 → S4/A5 → S5 → A6 — cheapest information first) until the cap is hit; anything denied is reported with an explicit `budget-denied` status in the final harvest, distinct from "resolved at base seed count." A matching build item — the live guard wired into both escalation-trigger code paths, plus its negative test (a simulated over-budget projection must deny in the pinned order and log the denial) — is added to §1.12's "New, not yet built" list. This is a budget-arbitration mechanism only: it does not change §1.4.2's escalation eligibility rules or §1.5's decision criteria. | §1.6, §1.12 |
| **MINOR** — stale-wording nit on A6's "will still run before launch" (STEP 0's executed closure check, §1.3.3, already subsumes it) | Reworded in place: the standard closure-order smoke on A6's two pinned generators is "already executed, see §1.3.3," quoting the STEP 0 PASS line, rather than described as a future build-step action. | §1.4 (the A6 generating-set paragraph) |

**Nothing from §1.19's "Verified clean" list was touched** — the TOST
target quantity, Welch-vs-paired justification, centralizer threshold,
contingency/bump disjointness, n_fit-floor × seed-bump orthogonality,
and every §1.13-§1.19 record's byte-integrity all stand as attack round
4 verified them, unmodified in Rev 4 (verified via `git diff` before
this commit — see the commit message). This revision touches only
§1.6 (new paragraph), §1.12 (new build item), §1.4 (one clarifying
reword), the top-of-doc status block, and adds this §1.20 — no other
design content changed.

---

*(End §1 records. Rev 0 → §1.13 → Rev 1 → §1.15 → Rev 2 → §1.17 →
Rev 3 → §1.19 → Rev 4 (§1.20) resolves CA4-M1 (escalation budget guard)
+ 1 minor (A6 wording). Attack round 5 next.)*

---

### 1.21 ATTACK ROUND 5 VERDICT (micro-round, independent fresh-eyes agent, 2026-07-08): **DESIGN-CLEARED-FOR-BUILD**

Recorded per the gauntlet-bookkeeping hard rule; the build stage
dispatches against THIS verdict. Scope = Rev 4's delta + coherence with
unchanged sections. All confirmed: the 42-cell/12.60 worst-case
arithmetic independently re-derived from the cell table (S3/S5/A6:
10 extra cells/group; S4/A5: 6 each — flanking only, since
unconstrained+d_min are already unconditionally n=5 and thus not
general-escalation-eligible); marquee +2.40 re-derived (8 cells);
34.65 > 30 by 4.65 airtight; yield order TOTAL; the guard is
arbitration-only with no §1.4.2/§1.5 contradiction (a budget-denied
marquee escalation falls through to the pre-existing INCONCLUSIVE
catch-all); §1.12 build item runnable; §1.13-§1.19 byte-intact
(hunk-boundary verified).

**Three MINOR notes, safe defaults pinned here for the build agent
(no Rev 5 needed):** (1) S4/A5 "tied" contradicts the ascending-|G|
wording (24≠60) — pinned tie-break: canonical name order, S4 before A5
(adjudicated non-consequential in the worst case: 4.95 remains after
marquee+S3, S4+A5 need 3.6 together). (2) The denied-marquee-escalation
chain is implicit — made explicit here: **if the marquee n=7 escalation
is budget-denied, TOST's INCONCLUSIVE stands and the denial is
disclosed in the harvest; §1.5's CONFIRM cannot be reached without the
declared equivalence.** (3) Stale QUEUE footer — superseded by this
section; the operative build-item set is §1.12's "New, not yet built"
list.

*(End §1 records. Rev 0 → four NEEDS-REVISION rounds → Rev 4 →
**§1.21 DESIGN-CLEARED-FOR-BUILD**. BUILD STAGE ACTIVE. Stage-1 ledger
30 GPU-h, budget-guarded; launch sequenced vs the h2h box load.)*

---

### 1.22 INDEPENDENT BUILD AUDIT VERDICT (2026-07-08/09): NEEDS-FIXES

Recorded per the gauntlet-bookkeeping hard rule before the fix stage.
Build commit a3defcc (12 files). All 11 smoke sections re-run green;
all 4 companion scripts byte-reproduced; the 58-cell/19.65/34.65
arithmetic re-derived and exact; 5/6 mutations CAUGHT cleanly (coverage
bars pinned by value; yield order; TOST margin — independently
Monte-Carlo'd, Type-I ≈0% at null, REJECT power 99.9→64% across σ;
resume-safety verified at run_manifest level; escalation guard caught
DOUBLY). Mutation (d) mixed-L NOT caught — adjudicated
avoidance-by-construction (only safe samplers exist), residual risk
flagged for future non-sampler batch construction. All 3 builder
judgment calls SOUND (blank-out leaf adaptation verified with a
planted-leak test — catches a closure leak the honest path misses;
TOST REJECT mirror-test statistically well-behaved; gate-1b scope
faithful to the design's own duty split).

**FINDINGS (binding on the fix stage):**
- **BA-F1 (MAJOR) — the §1.7 gate-2 circuit-breaker is DEAD CODE:**
  check_base_sweep_projection/BaseSweepAbort never called by
  run_manifest()/main()/any smoke. The calibration-first gate does NOT
  actually block the 53 remaining cells. → wire into the CLI flow
  (calibration-only → gate → sweep), add smoke section + negative test.
- **BA-F2 (MODERATE) — M2 (post-hoc rank-k truncation curve) has zero
  implementation** — pre-registered corroborating measurement, no code.
  → build the truncation-curve module (unconstrained checkpoint Z,
  k=1..d_state, knee k*) + unit test.
- **BA-F3 (MINOR) — S4/A5 share byte-identical eval word-index
  structure** (eval seed ignores group name; both have |gens|=4) — an
  undisclosed second correlation channel beyond the adjudicated
  init-weight one; direction conservative but accidental. → salt the
  seed derivation by group name.
- **BA-F4 (MINOR) — escalation allocator is skip-and-continue,** which
  can grant a cheaper lower-priority item after denying a
  higher-priority one in a narrow headroom window (A5→S5 transition);
  the shipped negative test's headroom never lands there. → hard-stop-
  on-first-denial semantics OR a targeted test pinning the window.
- **BA-F5 (MINOR) — beta_fla_smoke's declared L_max=4 never used**
  (Fig.5 reproduction would run off-spec L∈{1..8} on box). → apply cap.
- **BA-F6 (COSMETIC)** — blank_out.py dead no-op. Also ADD the
  blank-out planted-leak negative test the audit prototyped (formal
  test, mirroring the coverage guard's discipline).
- Non-finding recorded: float64 degauging boundary works via incidental
  numpy promotion — make the .astype(np.float64) contract EXPLICIT
  while in there (robustness, not correctness).
- Deploy-stage checklist (7 items incl. box smoke, real-GPU
  calibration gate, β-smoke re-run, general-escalation detection
  before reliance, group-salted seeds, SHA closure) recorded in the
  audit transcript — binding on deploy.

---

### 1.23 FIX STAGE CHANGES — §1.22 finding → resolution map

Every §1.22 finding, mapped to its exact fix-stage resolution. All 12
smoke sections re-run green (`smoke_capability_sep.py`, was 11 sections —
+1 for the new `truncation_curve.py` module, BA-F2); each fix's own
negative/positive test run to completion, output shown in the fix
transcript. Scoped strictly to §1.22's findings — nothing else touched.

| Finding | Resolution | Where |
|---|---|---|
| **BA-F1** (MAJOR) — `check_base_sweep_projection`/`BaseSweepAbort` dead code, never called; the calibration-first gate did not actually block the 53 remaining sweep cells | `run_capability_sep.py` gets an explicit two-step CLI: `--calibration-only` runs the 5 calibration cells and writes a calibration report (`calibration_report_path`/`write_calibration_report`, the real measured per-cell rate); `--sweep` REFUSES to start unless `gate_sweep_launch` clears all three preconditions in order — (i) `load_and_validate_calibration_report` (report exists, parses, matches `GROUP_NAMES`/cell count/the requested `--steps`), (ii) `budget_guard.check_base_sweep_projection` on the MEASURED rate (raises `BaseSweepAbort` if it would blow the 30 GPU-h cap), (iii) the PI-signoff token. Smoke section added: a synthetic over-rate calibration report aborts `--sweep` via `BaseSweepAbort` (negative, run to completion); a healthy rate passes; missing-report/stale-steps/no-PI-signoff refusals also proven. Exercised via the REAL CLI end-to-end (`--calibration-only` then `--sweep`, tiny steps, CPU), not just the internal smoke. | `run_capability_sep.py` |
| **BA-F2** (MODERATE) — M2 (post-hoc rank-k truncation curve) had zero implementation | New module `truncation_curve.py`: `truncation_curve(model, name, base_seed, device)` runs `readout.py`'s production pipeline at k=1..d_state on a FIXED eval-word sample (comparable curve across k), reusing the existing `force_rank_k` threading (`GroupWordModel.encode` → `rank_utils.truncate_to_rank`) — genuinely post-hoc, the checkpoint is never retrained per k. `knee_from_curve` pins S1.5's exact rule (smallest k with `recovered_frac_90(k) >= 0.9*recovered_frac_90(k=d_state)`, Task D/E's `run_stage1.py` M2 convention, ported field names). Unit test: synthetic Z built as a rank-`r_true` ORTHOGONAL PROJECTOR (tied singular values @ 1.0, not a generic random rank-r matrix — a generic construction was tried first and kneed early, since a few dominant directions already capture most Frobenius energy; the tied-projector construction gives the exact identity `cos(rank-k trunc, Z) = sqrt(k/r_true)`, producing a genuinely sharp knee) — curve knees EXACTLY at the planted rank (r_true=4, d_state=7), verified `k>=r_true` lossless / `k<r_true` degraded. | `truncation_curve.py` (new) |
| **BA-F3** (MINOR) — S4/A5 (both `\|gens\|=4`, `d_state=5`) drew byte-identical eval word-index sequences (eval seed ignores group name) | `groups.group_seed_salt(name)` (sha256-hashed, deterministic) added; salted into `group_task.sample_eval_words`'s RNG seed, `readout._split_with_diversity_retry`'s shuffle seed, and `run_capability_sep.train_and_eval_cell`'s TRAINING-batch generator seed (the model-INIT seed, `torch.manual_seed(seed)`, is deliberately left UNSALTED — that's the separately-adjudicated shared-init channel S1.4.2.1's Welch-unpaired TOST already accounts for). Determinism per (group, seed, N) preserved (proven, not assumed). Negative-then-positive test: S4 vs A5 sequences now DIFFER at the same nominal seed; same-group re-draw still reproduces exactly; `check_coverage_with_retry`'s `final_seed` still round-trips consistently through the salted `sample_eval_words`. `gate1_synthetic_injection.py` and `force_rank_arms.py`'s existing S4-only pipelines re-verified green under the new salted seeds (different specific draw, same pass/fail outcome). | `groups.py`, `group_task.py`, `readout.py`, `run_capability_sep.py` |
| **BA-F4** (MINOR) — escalation allocator was skip-and-continue, could grant a cheaper lower-priority item right after denying a pricier higher-priority one in a narrow headroom window (A5→S5 transition) | `BudgetGuard.run_escalation_pass` changed to HARD-STOP-ON-FIRST-DENIAL: once any request in a pass is denied, every subsequent request (pinned order) is recorded `budget-denied` WITHOUT being evaluated against the live projection — a denial ends the pass. New negative test `_test_a5_to_s5_hard_stop_window` derives the EXACT headroom (spend_to_date=23.70, programmatically, not hand-picked) where the OLD semantics (reproduced inline via direct `request_general_escalation` calls, for this regression check only) wrongly grants `S5:unconstrained` (cost 0.6) immediately after denying `A5:k_dmin_plus_1` (cost 0.9); the NEW `run_escalation_pass` denies both, and every request from the first denial onward. The pre-existing `_test_near_cap_denies_in_pinned_order` (headroom 10.35) still passes unchanged — that headroom happens to exhaust tightly enough that old/new semantics coincidentally agree there, which is exactly why it never caught this bug and why the new targeted test was needed. | `budget_guard.py` |
| **BA-F5** (MINOR) — `beta_fla_smoke`'s declared `L_max=4` never threaded through `reproduce_fig5`'s sampler (would run off-spec `L∈{1..8}` on box) | `L_max` promoted to a module constant `FIG5_L_MAX`; `group_task.sample_train_batch` gets an optional `l_hi` override (default `None` → unchanged `{1..8}` production range, so the real 58-cell sweep is untouched) that `reproduce_fig5` now passes explicitly, plus an in-loop assertion. New CPU-only test `_test_fig5_l_max_cap_applied` (independent of whether real `fla` is installed, unlike `reproduce_fig5`'s own training loop) proves both halves directly: capped draws never exceed 4; the SAME seed uncapped (the pre-fix behavior) reaches L up to 8 routinely (94/200 draws), so the cap is not vacuous. | `beta_fla_smoke.py`, `group_task.py` |
| **BA-F6** (COSMETIC) — `blank_out.py` dead no-op (`if model.encoder.d_state: pass`) | Dead no-op removed. New formal negative test `run_blank_out_planted_leak_test`: a `scoring_fn` closure that reads `tok_embed` directly (bypassing Z) is CAUGHT (nonzero grad on the leaf-detached path) while the honest path still reports `None` under the identical mechanism — mirrors the coverage guard's/diversity-floor's own planted-failure test discipline. (First draft of the leak closure used a `0.0 *` coefficient and was itself caught as a false negative during this fix — a zero coefficient has zero LOCAL GRADIENT too, not just zero forward value; fixed to a nonzero `1e-3` coefficient, now genuinely demonstrates the mechanism has teeth.) | `blank_out.py` |
| Non-finding — float64 degauging boundary worked via incidental numpy promotion | `readout.dump_Z` now does `.astype(np.float64)` explicitly at the torch→numpy readout boundary, with a one-line comment naming the promotion chain it replaces. No numeric behavior change (same effective dtype throughout, verified via the full smoke suite and `gate1_synthetic_injection.py`'s unchanged acceptance numbers). | `readout.py` |

**Not done exactly as scoped:** nothing — all six findings plus the
non-finding were addressed within the stated scope; no scope creep (the
deploy-stage checklist above is explicitly deploy-stage, out of this
fix stage's mandate).

*(End §1 records. §1.21 CLEARED → build a3defcc → §1.22 audit
NEEDS-FIXES → **§1.23 FIX STAGE COMPLETE** (all 12 smoke sections green).
Scoped re-audit next → deploy.)*

---

### 1.24 SCOPED RE-AUDIT VERDICT (2026-07-09): FIXES-VERIFIED-CLEARED (one residual)

Recorded per the gauntlet-bookkeeping hard rule. Fix commit a555012
(isolated diff: exactly 10 files, no scope creep — the naive
79ada6a..a555012 range includes an unrelated interleaved h2h commit).
All six fixes independently re-verified with the re-auditor's own runs:
BA-F1 real-CLI end-to-end (refusal without report; over-rate 60.25
aborts; healthy 19.65 clears; mutual-exclusion; PI-signoff env gate);
BA-F3 salt verified both directions + init-seed deliberately unsalted
w/ documented adjudication; BA-F4 the 23.70-headroom window
programmatically derived, old-semantics wrong-grant reproduced, new
hard-stop verified; BA-F5 cap verified + production path untouched;
BA-F6 leak caught at 1e-3, dead code gone; float64 explicit; 12/12
smoke sections green.

**RESIDUAL (fold into deploy): RA-1 — the truncation-curve planted-rank
test cannot falsify the 0.9 knee-threshold constant** (the tied-
orthogonal-projector construction makes rf@0.9 a binary step; every
multiplier 0.01-0.99 yields the same knee). Production constant
verified correct on disk vs §1.5. → deploy stage adds a companion test
with a non-tied spectrum so a threshold mutation is falsifiable.

**DEPLOY STAGE AUTHORIZED** per §1.22's 7-item checklist + RA-1.

*(End §1 records. §1.21 CLEARED → build a3defcc → §1.22 NEEDS-FIXES →
fixes a555012 → **§1.24 FIXES-VERIFIED-CLEARED. DEPLOY ACTIVE** —
GPUs 0-6 free while h2h is parked on the §1.3.1 instrument redesign.)*

---

### 1.25 GATE-1 DIAGNOSIS (2026-07-09): HARD-STOP FOR DESIGN ROUND — models healthy, TWO instrument defects proven; the M1 signal already visible through the corrected lens

Recorded per the gauntlet-bookkeeping hard rule. The h2h §1.21
precedent applied and INVERTED: there the objective lacked the
capability pressure; here the objective provably forces it (direct
ρ_G supervision) and the models demonstrably have it — the
instruments could not see it. 0.38 GPU-h spent; artifacts at
results/gate1_diagnosis/ on box.

**DEFECT 1 (FATAL to M1/M3 as instrumented) — §1.4.1 step 2's
UNCENTERED covariance-SVD is mathematically degenerate for Option A
targets:** Z ≈ c·(ρ⊕I) is orthogonal ⇒ ZZᵀ ≈ c²·I — isotropic, zero
subspace information; the constant identity complement systematically
outranks the 2 weakest ρ directions (principal cosines always capture
exactly d_min−2). PROOF: an ambient synthetic injection shows a
PERFECT model FAILS the production bars (mean_cos 0.711, rec90 0.15);
the shipped gate-1(b) injection had skipped entity_subspace_from_words
(disclosed in its docstring) — which is exactly why it passed while
every real checkpoint read −0.02..0.17. **VALIDATED FIX: CENTER the
covariance** (group-mean of a nontrivial irrep = 0 cancels the
constant block): synthetic 0.9996/1.00 = oracle; real checkpoints
restored w/ razor d_min spectral gaps (A5@20K [1,.95,.80,.007,.003];
A6@20K gap at exactly 5).
**DEFECT 2 (FATAL as pinned) — M1/M3's decision surface was pinned
EXCLUSIVELY to L∈{9..16} words while nn.Embedding positional rows
8..15 NEVER TRAIN** (zero gradient at L_train≤8): every scored word
fed N(0,1) rows; causal clamp-probe recovers +0.04..+0.12; ρ-block
signal at L9-16 ≈ 0 at BOTH 8K and 20K (budget-independent, as the
defect predicts) — the 58-cell sweep would have produced
uninterpretable zeros.
**EXONERATED:** Procrustes/scale degauging (Q̂≈I — §1.9 item 7
answered: scale-only suffices once U is fixed); training loop;
fit/eval split; closure (17/17 md5).
**CONVERGENCE re-read:** the A5/A6 "under-convergence" was largely a
last-batch-loss logging artifact (per-batch-fixed-L makes final loss
L-dependent; per-L profiles nearly identical); 20K steps still gives
genuine, material trained-length improvement → pin 20K for A5/A6.

**M1 PREVIEW (the headline, through the corrected lens, train-length):
restricted effective rank 1.85/2.74/2.64/3.73/3.91-4.54 vs d_min
2/3/3/4/5 — Spearman ρ=0.9747 (the achievable max under the S4/A5
tie); every group inside the [0.7,1.3]·d_min band; the marquee pair
lands together (2.74 vs 2.77).** CAVEAT (registered): restriction to
d_min dims partially favors rank≈d_min for near-orthogonal blocks —
M3's causal force-rank arms remain the decisive test. The campaign is
healthy.

**PINNED FOR THE DESIGN ROUND (Rev 5, all pre-validated):** (1)
centered covariance in §1.4.1 step 2 + gate-1(b) injection extended to
AMBIENT d_state so the U step is inside the acceptance test; (2)
M1/M3 measurement lengths re-pinned to fresh TRAIN-support words;
L9-16 becomes the disclosed C5 out-of-support control; (3) 20K steps
for A5/A6 per the pre-registered escalation rule (cost trivial at the
measured rate); (4) per-L eval-based convergence reporting replaces
last-batch loss; (5) scale-only degauging primary, full-Q cross-check.

---

*(End §1 records. ... → §1.24 deploy CLEARED → calibration → **§1.25
HARD-STOP: two instrument defects proven + fixes pre-validated; M1
preview POSITIVE (ρ=0.9747)** → Rev 5 design round ACTIVE → micro
attack → scoped build fix → calibration re-check → sweep.)*
