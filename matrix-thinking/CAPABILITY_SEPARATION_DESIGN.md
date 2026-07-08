# CAPABILITY-SEPARATION — group-element-recovery rank/representation-dimension test

## §1 DESIGN — Stage 1 (Rev 1, post-attack-round-1 revision, 2026-07-08) —
does a from-scratch matrix-state model recruit subspace-restricted state
rank equal to a group's minimal faithful real representation dimension
`d_min(G)`, tracking dimension not solvability?

Status: **Rev 1, pre-attack (round 2 pending), zero GPU spent.** Rev 0
(§1.1-§1.12) was attacked (§1.13: NEEDS-REVISION — 1 FATAL-as-written + 3
MAJOR + 2 minor); every finding is resolved by this Rev 1 (mapped in
§1.14). This design came through the full waterfall (brainstorm →
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
even permutation directly, not by closing 2 generators); the build step
will still run the standard closure-order smoke (below) on these two
specific generators as a cheap, load-bearing sanity check before launch.

**Architecture (Task D/E's exact bespoke encoder, TWO required
extensions — CA1-m1 fix, Rev 1: Rev 0 undercounted this as "ONE," see
below).** Reuses `matrix-thinking/chapter2/model_v4.py`'s
`BindingEncoder` (`d`, `h`, `n_layers`, `n_heads`, `n_refine`
constructor signature, `nn.TransformerEncoder` + learned `row_queries` +
`MultiheadAttention` reader + `row_norm` + `row_out` — same class,
same forward pattern) and `MatrixMemoryModel.encode`'s
`force_rank_k`-via-`rank_utils.truncate_to_rank` path (§1.5 below), with
**one architectural delta, required because group composition is
non-abelian and Task D's `BindingEncoder` is otherwise permutation-
EQUIVARIANT (no positional signal) by construction** (it was correct for
Task D's genuinely order-INDEPENDENT set-of-bindings task, and would
silently break here): a learned absolute positional embedding
(`nn.Embedding(L_max, h)`), added to each generator token's embedding
before the `nn.TransformerEncoder`, tags word position 1..L. **Two
deltas total, both structurally required by the task change, not a
broader redesign (CA1-m1 fix — Rev 0's "ONLY change" sentence undercounted
the second, which was already disclosed two bullets below but never
folded into this count):** (1) this positional embedding
(order-sensitivity, above), and (2) the input embedding itself — a single
generator-index embedding (below) replacing Task D's `[key; value]`
concatenation, because this task has no separate value to bind, the WORD
is the whole composition. `CLAUDE.md`'s "hold every second axis fixed when
testing a primary hypothesis" rule is honored because both deltas are
direct, minimal consequences of the SAME task-family change (word
composition replacing parallel key-value binding), not two independent
design choices stacked on top of it.

- **Input:** each token is a generator-index one-hot/embedding
  (`nn.Embedding(|S_G|, h)`), NOT Task D's `[key; value]` concatenation
  (there is no separate value here — the task IS the composition; this is
  delta (2) above).
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
   **Fit-set diversity floor:** the `n_fit=30` subset must realize
   `≥min(3·d_min(G), ⌊0.8·|G|⌋)` DISTINCT target elements — enough that the
   `d_min(G)²`-unknown orthogonal-intertwiner linear system (§1.3.2 step 2)
   is comfortably overdetermined, a deliberately conservative floor
   relative to §1.3.2's own toy ratio (14 fit elements for a `d=3` group,
   ≈4.7×`d`). Executed via `coverage_calibration.py::fit_set_diversity_check`
   (`n_fit=30`, 20,000 trials/group, §1.3.3) — every group clears its floor
   by ≥1 element at the 1st percentile (S3: bar 4, p1 5; S4: bar 9, p1 14;
   A5: bar 9, p1 19; S5: bar 12, p1 21; A6: bar 15, p1 25). If a real
   fit-set draw fails the floor (rare, ≤1% by construction), redraw the
   fit/eval split from the same `N=50` sample before fitting — a redraw
   guard, not a silent pass/fail toggle. Score `recovered_frac@0.9` and
   mean cosine on the DISJOINT `n_eval=20` subset — this is the Stage-1
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
default):**

| Cell type | Seeds | Justification |
|---|---|---|
| Unconstrained (M1/M2) | 3 | Needs clean statistics for the family-wide Spearman ρ (§1.5) — the headline recruitment claim. |
| Force-rank `k = d_min(G)` | 3 | The critical boundary point — this is where the "step at `k=K±1`" claim lives; needs full precision. |
| Force-rank `k = d_min(G)−1` | 2 | Expected to show a CLEAR near-chance signal (below the definitional minimum, no faithful representation exists at all) — a flanking/sanity cell, not a close call, mirrors `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.6's own quarter-budget-stress-point economization pattern (applied here via seed count, not step-budget, since step-budget is already the primary lever being kept short, §1.6). |
| Force-rank `k = d_min(G)+1` | 2 | Same reasoning: expected to clear ceiling clearly, a sufficiency check, not the primary boundary. |

**Pre-registered escalation-to-5 trigger (per the validation verdict):**
if any cell type's within-family Spearman ρ or M3 step-margin (§1.5) is
AMBIGUOUS (CI straddles the pre-registered bar), extend THAT cell type to
5 seeds before drawing a conclusion — mirrors
`REASONING_LINK_DESIGN.md` §16.19/§16.20's n=3→n=12 precedent and
`HEAD_TO_HEAD_DEMO_DESIGN.md` §1.8's seed-extension contingency
(gated the same way: variance-ratio check before pooling old/new
cohorts, `var_ratio > 4.0` → flag, don't silently pool).

**Cell count per group:** 3 (unconstrained) + 3 (`k=d_min`) + 2
(`k=d_min−1`) + 2 (`k=d_min+1`) = **10 cells/group**. Five groups: **50
cells total** (§1.6).

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
- CONFIRM: `k* ∈ [d_min(G)−1, d_min(G)+1]` in ≥4/5... (n=3 minimum here,
  so: in ≥2/3 seeds, escalating to 5-seed's ≥4/5 bar only if the
  escalation trigger fires, §1.4.2).

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

**Marquee dissociation check (S4 vs A5, both `d_min=3`):** the two
groups' M1 restricted-rank values and M3 step locations must be
statistically indistinguishable (overlapping CIs) from each other — this
IS the CONFIRM condition's "land together, not apart" clause, made a
concrete, checkable comparison, not just prose.

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
| Main sweep (50 cells × 0.3 GPU-h) | 5 groups × 10 cells/group (§1.4.2) | 15.0 |
| Calibration-first wave (§1.7 gate 1) — 5 cells (1/group, unconstrained, seed=0), **reused within the 50 sweep cells above, not double-charged** | 5 cells | 0.0 (already counted) |
| Degauging-pipeline validation on a REAL trained checkpoint (§1.7 gate 1) | CPU-only, numpy | ≈0.0 |
| Contingency margin — 2-2.5× escalation rule firing on 1-2 cells | ~2 cells re-run at 2.25× | 1.35 |
| **β∈[0,2] fla positive-control smoke** (§1.4.3 below) — forward/backward/grad-check (~0.05) + one reproduced DeltaProduct Fig.5 point on a minimal S4/A5 instance, `L≤4`, `n_h∈{1,2}` (~0.85) | nominal | **0.90** |
| Group-construction + generator-closure smoke (§1.3, §1.4) — CPU-only, numpy | — | ≈0.0 |
| MATCH/verification overhead (blank-out test, query-coverage check, injectivity-guard negative test) — CPU-only | — | ≈0.0 |
| **Raw total** | | **≈17.25 GPU-h** |

**CA1-M2 fix, Rev 1 — cost ceiling reconciled.** Rev 0's stated raw total
("≈17.4-18.3 GPU-h") carried ~1 GPU-h with no itemized source above the
itemized rows' own sum (§1.13 CA1-M2). Fixed by itemizing the previously-
vague `<1.0` β-smoke row to a specific **0.90 GPU-h** (forward/backward/
grad-check plus the actual Fig.5-reproduction training run, not a nominal
upper bound) — the raw total now equals the itemized sum exactly:
`15.0 + 1.35 + 0.90 = 17.25` GPU-h, no unaccounted margin.

**Stage-1 dedicated ledger: 30 GPU-h cap, PI-visible, does NOT draw the
frozen-bias ledger** (per the validation verdict's explicit instruction —
this campaign is budgeted separately from the head-to-head demo's 135
GPU-h program ceiling). Raw ≈17.25 GPU-h leaves **≈42.5% margin**
(`(30−17.25)/30`) under the 30 GPU-h cap (CA1-M2 fix, Rev 1 — a single
reconciled figure, not Rev 0's own unreconciled "≈40-42%" range) — comfortably
wider than `HEAD_TO_HEAD_DEMO_DESIGN.md`'s
own razor-thin margins (≈1-2%), appropriate for a first-look Stage-1
gate at this much smaller scale (`d_state≤7` vs the head-to-head's 14M+
param models).

**Circuit breaker — deliberately NOT a literal 10×-of-raw multiplier
(flagged as a design decision, not silently inherited from
`HEAD_TO_HEAD_DEMO_DESIGN.md`'s convention).** That design's 10× bracket
exists because ITS cells are large (0.25 GPU-h/cell at 14M params,
scaling to 28 GPU-h/cell at 392M) and a runaway sweep could plausibly
consume the WHOLE shared 135 GPU-h ceiling; a literal `10×17.25≈172.5 GPU-h`
bracket here would be nonsensical overkill relative to this campaign's
own explicitly-stated 30 GPU-h cap. **This design's circuit breaker
instead mechanically enforces the 30 GPU-h dedicated cap directly**, via
the same timing-pilot-based abort machinery precedent
(`phase2b_off_cache.py --time-pilot`, §1.7 gate 2): if the calibration
cell's measured real rate projects the 50-cell sweep to exceed 30 GPU-h
(incl. the pre-registered contingency margin), the chain hard-aborts
before spending it, and this design's cell/seed grid gets re-scoped
before relaunch, not silently over-spent.

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
   remaining 45-cell sweep's own go/no-go.
2. **Timing pilot, mechanical enforced abort.** One real cell per group
   measured for wall-clock (folds into gate 1 above, since the
   calibration cells ARE one real cell per group); if the projected
   50-cell sweep cost exceeds the 30 GPU-h dedicated cap (§1.6), the
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
   one; flagged for attack round 1 to confirm this simplification is
   safe).
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
  query-coverage guard + its negative test; the block-diagonal `rho_G`
  embedding; the β∈[0,2] `fla` smoke harness; **the 30 GPU-h hard-abort
  enforcement wrapper** (§1.6's circuit breaker — projects the 50-cell
  sweep's cost from the calibration cell's real per-cell rate and
  mechanically hard-aborts before exceeding the dedicated cap; CA1-m2
  fix, Rev 1 — Rev 0 designed this mechanism in §1.6 but never itemized
  it in this build list).

---

**QUEUE (STATE.md, appended per this design's commit):** Design Rev 0
committed (this commit) → attack round 1 next.

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

*(End §1 Rev 1 records. Rev 0 attacked → NEEDS-REVISION (§1.13: CA1-F1
impossible coverage bar, 3 MAJOR, 2 minor). Rev 1 (§1.14) resolves all 6
findings. Attack round 2 next.)*
