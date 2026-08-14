# NCR WRITE-CONDITIONING — ADVERSARIAL ATTACK ROUND 1

**Target:** `matrix-thinking/NCR_WRITE_CONDITIONING_DESIGN.md` (DRAFT-R0, 693 lines), repo commit `68a7c68`.
**Agent:** independent round-1 attack (frame → arithmetic → instrument, per the K-wall `§A1-ADJUDICATION` charter style).
**Date:** 2026-08-13. Read in full. Every cited number checked against the RAW artifacts, not against prose.

## VERDICT: **BLOCKED**

**5 FATAL · 11 MAJOR · 7 minor.** Not REV-REQUIRED: three of the five FATALs
(F1, F2, F3) are not defects *in* the design's mechanism family — they are
defects in the premise, the derivation's sign, and the primary arm's
objective, each independently sufficient to make all 13 Stage-1 cells
produce an uninterpretable verdict. A revision cannot rescue a wave whose
baseline has no shallow-depth capability to preserve. The correct next
spend is a ≈0–0.83 GPU-h premise-establishing measurement (D1), not a 19.2
GPU-h mechanism wave.

**Most dangerous finding: F1.** The design's own cited artifacts show 24-way
retrieval is **at chance at h=1** — one application of `Z`, no repeated
squaring, no power iteration — in all three contrastive-grid cells. There
is no depth-destroyed capability. Under `§3.6`'s final clause an all-NULL
Stage 1 would be recorded as *"the read-collapse mechanism is robust to both
read-side and write-side intervention at this architecture/scale, a
materially different and more serious finding"* — a **wrong** conclusion
published off a wave that could not have shown otherwise.

**Evidence base.** Raw JSONs at
`/Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/`
(`mob_g3b31_{primary,compA,compB}_s0.json`); runner md5 re-verified from the
repo copy; `NCR_REAL_LM_DESIGN.md` §G3-B24–B32; `NCR_ORTHO_WRITE.md`
§9.1/§10.7; `NCR_ORTHO_FALLBACK_DESIGN.md` §3.1/§3.2/§B8; `EXPERIMENT_LOG.md`
Z-dump complement entry; `matrix-thinking/ncr/ncr_task.py` `GRIDS`.
Numeric demonstrations in
`/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad/attack{,2,3,4,5,6,7}.py`.

---

# PART I — FATAL

## F1 (FATAL, frame) — The premise is falsified by the artifacts the design cites: there is no shallow-depth capability for depth to destroy

`§0` asserts the mechanism as *"`binexp_read`'s repeated squaring … is power
iteration toward `Z`'s own dominant singular direction — measured `Z` s1/s2 ≈
1.21 is already enough to erase query-discriminability by h=61."*

**`retrieval24_acc`, `full_graft` arm, raw JSONs (chance = 1/24 = 0.04167, n=64, SD = 0.0250):**

| cell | h=1 | h=2 | h=3 | h=5 | h=12 | h=20 | h=29 | h=40 | h=61 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| compB 0994 | **0.03125** | 0.06250 | 0.03125 | 0.01562 | 0.06250 | 0.09375 | 0.03125 | 0.06250 | 0.01562 | 0.0451 |
| PRIMARY 0992 | **0.10938** | 0.03125 | 0.01562 | 0.01562 | 0.06250 | 0.04688 | 0.03125 | 0.04688 | 0.04688 | 0.0451 |
| compA 0993 | **0.01562** | 0.03125 | 0.03125 | 0.00000 | 0.04688 | 0.03125 | 0.06250 | 0.04688 | 0.10938 | 0.0417 |

At `h=1` the read applies `Z` exactly **once**. There is no repeated squaring,
no `Z^h`, no power iteration, nothing for "depth" to erode. All three cells
sit at chance (0.031 / 0.109 / 0.016; the 0.109 is +2.7 SD and is 1 of 54
recorded draws — and it belongs to the cell §G3-B32 labels NULL-BY-COLLAPSE,
not to a healthy arm). `recovered_frac@0.9 = 0.0` at every hop in compB.
`offtarget_margin` = −0.007…+0.003 at every hop.

`§G3-B26` records the same for the pre-contrastive checkpoint: 24-way NN
retrieval 0.031–0.062 at h ∈ {1,2,3,61} — chance at h=1 too.

The design *states* both halves of this ("24-way retrieval stayed at chance
**throughout**", §0) and then draws a conclusion inconsistent with them. The
non-discriminative common mode is present at depth 1 and is not a depth
artifact: `mean_cos` (full_graft − backbone) is 0.31–0.38 at **every** hop
including h=1, i.e. `o` is aligned to the target *cone* by ~0.35 and to the
*correct* target by nothing, uniformly across depth.

**Consequence.** Spectral conditioning of `Z` changes how `Z^h` degrades
relative to `Z^1`. It cannot manufacture a `Z^1` association that was never
learned. Every Stage-1 arm is predicted NULL independent of how well its
mechanism works, and §3.6's all-NULL clause would record a false structural
finding. **This is a launch-losing defect.**

**What would falsify F1:** a recorded cell in which `full_graft` retrieval24
at h ∈ {1,2,3} clears chance by a pre-registered margin. None exists in the
archive. See D1.

## F2 (FATAL, frame/derivation) — §1.1's stated success target is the numerical signature of total collapse

`§1.1`: *"`sin θ_h = 0` for every h, by construction, not by extrapolation.
**This is the target every mechanism below is measured against.**"*

`§1.2` operationalizes the same symbol as `sin θ_61 = √(1 − 0.9961²) ≈ 0.088`
from `|cos(o_h61, Z's top singular direction)| = 0.9961`. Under that
definition **small `sin θ` IS the collapse** — the whole §1.2/§1.3
apparatus fits `sin θ` *decaying* toward zero and §1.3 requires `sin θ_{h*}`
to stay **above** 0.20. §1.1 declares the failure condition to be the target.

**Re-derived numerically** (d=25, `Z = cQ`, Q orthogonal, scale-managed read,
`attack2.py` N1b):

| h | sin θ_h | mean pairwise cos across 24 queries |
|---|---|---|
| 0 | 0.9973 | −0.0094 |
| 1 | 0.9982 | +0.0027 |
| 3 | 0.9955 | −0.0054 |
| 20 | 0.9760 | +0.0100 |
| 61 | 0.9918 | +0.0105 |

`sin θ_h` is **constant at ≈ sin θ₀ ≈ 0.99**, not 0. The correct statement is
`sin θ_h = sin θ₀`. The design's own equation says so: `sin θ_h ≈ C·ρ^h` with
`ρ = 1` gives `C`, not 0. The paragraph is introduced as *"the exact endpoint
(no approximation)"* and is the section every downstream band inherits.

The *substantive* claim of §1.1 (a scaled-orthogonal `Z` preserves all
pairwise inner products exactly at every `h`, so cosine scoring is untouched)
is **correct and verified above** — the error is confined to the sentence
naming the metric, but that sentence is what the calibration cell, `ρ_required`
and Band 2 are anchored to.

## F3 (FATAL, mechanism) — The PRIMARY arm's penalty is minimized by the identity map; its optimum destroys the capability while scoring Band-2 PERFECT

`L_conformal = λ_b·‖Z_raw − ĉ·I_d‖_F²/d²` with `ĉ = tr(Z_raw)/d`.

The task's ideal write is `z_ideal = K_mat·P·K_matᵀ` — a K-cycle permutation
on the entity subspace (`NCR_ORTHO_FALLBACK_DESIGN.md` §3.1, verbatim). At
`K=24, d=25` (`attack3.py`/`attack4.py` N7/N7b):

```
ideal K-cycle write Z* :  s1/s2 = 1.000000 ,  retrieval24@h=61 = 1.000
L_conformal(Z*)   = 3.9936e-02      <- the PRIMARY penalty AT THE CORRECT SOLUTION
L_conformal(c*I)  = 0.0             <- the penalty's GLOBAL MINIMUM
‖Z* − ĉI‖_F² / ‖Z*‖_F² = 0.9984     <- the "deviation" the penalty removes IS ~100% of Z
```

Descending `L_conformal` from the solution toward its own minimum:

| t (mix toward c·I) | s1/s2 | L_conformal | cos(o,true)@h=3 | cos(o,true)@h=61 |
|---|---|---|---|---|
| 0.00 (ideal) | 1.00000 | 3.99e-02 | +1.0000 | +1.0000 |
| 0.20 | 1.00000 | 2.56e-02 | +0.7911 | +0.0008 |
| 0.50 | 1.00000 | 9.98e-03 | +0.2236 | +0.0960 |
| 0.90 | 1.00000 | 3.99e-04 | +0.0013 | +0.0120 |
| 1.00 (`c·I`) | 1.00000 | 9.2e-13 | +0.0000 | −0.0000 |

The penalty falls monotonically, the capability is destroyed, and **`s1/s2`
— the design's Band-2 "mechanism check" — reads 1.00000 at every point
including total capability loss.** At `Z = c·I`, `Z^h q ∝ q`: the read returns
the query, so h-hop retrieval is at chance *by construction*, while §3.6
Band 2 scores a perfect PASS. The arm's pre-registered taxonomy would label
this "mechanism achieved, no behavioural gain" — a maximally misleading
write-up of a self-inflicted result.

**The design's own cited source says the target is dispensable.**
`EXPERIMENT_LOG.md`'s Z-dump complement entry (commit 64c59d9), which §2(b)
invokes as its motivation, records verbatim:

> **Dispensability:** the rank-7-forced converged seed (fr7 s2) has the
> scaffold amputated (fD=0.0002, fA=0.990) yet recovers h≤7 — **the identity
> is unconstrained-SGD's default, not load-bearing.**

The same entry also fixes the energy split as `fA ≈ K/d`, `fD ≈ (d−K)/d`. At
`K=24, d=25` the c*·I complement is **one dimension, ~4% of the state**; the
finding was measured at `K=8,d=16` (fD=0.50) and `K=12,d=16` (fD=0.25). The
design transplants a `d ≫ K` complement observation onto a `d = K+1`
configuration where the complement barely exists, and quotes the entry's
positive half while omitting its dispensability clause.

**Answering the charter's question directly:** does a write-side loss on `Z`
evade §G3-B32's read-side NULL's *letter* and its *mechanism*? **Both** — it
has no dependence on `o`, the target embedding, or the adapter, so it cannot
inherit the saturated-instrument pathology, and it attacks a different
mechanism. Mechanism (b) does not die of §G3-B32. It dies of F3.

## F4 (FATAL, arithmetic) — The pre-registered mechanism target does not prevent the collapse it is registered to prevent; Bands 2 and 3 are decoupled

`ρ_target = 1.10` provisional (§1.3, §2(c)), "derived bound" 1.19 (§1.3).
Simulating the **best case the hinge can deliver** — spectrum `(r, 1, 1, …, 1)`
(only `s1/s2 = r` off-flat, everything else perfectly equal), top singular
pair aligned to the query cone, queries drawn at pairwise cos 0.19 to match
compB's TPC 0.21 (`attack7.py`):

| s1/s2 = r | o_pc@h=1 | o_pc@h=3 | o_pc@h=20 | **o_pc@h=61** |
|---|---|---|---|---|
| 1.00 | +0.1868 | +0.1868 | +0.1868 | **+0.1868** |
| 1.01 | +0.1895 | +0.1950 | +0.2460 | +0.3941 |
| **1.02** | +0.1922 | +0.2034 | +0.3139 | +0.6220 |
| 1.05 | +0.2003 | +0.2293 | +0.5385 | +0.8809 |
| **1.10 (target)** | +0.2138 | +0.2749 | +0.7967 | **+0.8773** |
| **1.19 (bound)** | +0.2380 | +0.3611 | +0.8859 | **+0.8770** |
| **1.21 (measured)** | +0.2433 | +0.3804 | +0.8850 | **+0.8770** |

**At h*=61 the target (1.10), the bound (1.19) and the measured collapsed
value (1.21) are numerically indistinguishable — 0.877 all three.** The
quantity has saturated. Only `r ≤ ~1.02` preserves the input cone.

The independent analytic route agrees (`attack.py` N3): requiring
`sin θ_61 ≥ 0.20` under **any admissible** constant gives
`s1/s2 ≤ 1.026 (C=1)` to `1.054 (C = tan θ₀ = 4.90)`.

**Therefore an arm that HITS its own pre-registered mechanism target still
fails behaviourally, by the design's own model.** Band-2 PASS + Band-3 NULL
is the *expected* outcome, and §3.6 has no label for "the target was wrong" —
it would be filed as a clean behavioural NULL. This is precisely the
"uninterpretable verdict" the charter asked me to hunt.

`§3.3`'s calibration is not a remedy here: it re-fits `C` in a model that F2
mis-signs and M1 shows is inadmissible, and it cannot repair the fact that
the quantity the hinge controls saturates above ~1.02.

## F5 (FATAL, control design) — The shuffled/placebo control is not a null; it delivers the active ingredient

`§4`: *"an anchor penalty of identical form, magnitude, and update frequency
— but toward a RANDOM, per-example, DETACHED target `R_rand` (a fresh random
orthogonal matrix …). Same gradient perturbation/regularization pressure,
**zero information about conformal/orthogonal structure relevant to
composition**."*

`R_rand` is **orthogonal**. `§1.1` states the target explicitly: *"If `Z = c·Q`
with `Q` orthogonal … `sin θ_h = 0` for every h … This is the target every
mechanism below is measured against"*, and `§2`'s own table row for (a) reads
*"no — any orthogonal `Q`"* as the acceptable outcome. Anchoring `Z_raw`
toward a random orthogonal matrix is therefore a **maximally
flatness-inducing** intervention that lands squarely in the target class —
arguably a *better* mechanism than (b), since it does not collapse onto the
identity (F3).

The pre-registered interpretation inverts: *"if the placebo ALSO improves
retrieval, the conclusion flips to 'extra regularization/gradient-budget
reallocation helps, not specifically flatness'."* The single most
informative outcome of the whole wave would be read exactly backwards.

**A correct null** anchors toward a matrix that matches the *nuisance*
(gradient magnitude, update frequency, Frobenius scale) while randomizing the
*structure*: e.g. a random matrix with the **same singular-value profile as
`Z_raw`** (structure-randomized, spectrum-matched), or an i.i.d. Gaussian
anchor at matched `‖·‖_F` (whose condition number is large, not 1).

---

# PART II — MAJOR

## M1 — The fitted constant `C ≈ 9,900` is not "4 orders off an O(1) expectation"; it is outside the admissible range of a bounded quantity

`§1.2` reproduces exactly: `C = 0.088 / (1/1.21)^61 = 9871.7`. But
`sin θ ∈ [0,1]` by definition, and `C·ρ^h > 1` for every `h < 48.2`
(`attack.py` N2). The fitted model asserts `sin θ > 1` at every depth the
program has ever evaluated except h=61. `C = 9900` requires
`cos θ₀ = 1.01×10⁻⁴` — a query orthogonal to `v₁` to four decimals, for
*every* query. The generic value is `C = tan θ₀ ≈ √(d−1) = 4.90`.

The design frames this as an uncertainty to be re-fit. It is a
**mis-specification**: §3.3's calibration cells would re-fit a model that
cannot represent the data, and the R² < 0.5 escape hatch is measured against
the wrong functional form.

Backing out the effective rate with an admissible constant (`attack.py` N3):

| admissible C | implied ρ_eff | implied effective s1/s2 |
|---|---|---|
| 4.90 (generic query) | 0.9363 | **1.068** |
| 1.00 (design's own O(1)) | 0.9610 | **1.041** |

The **measured global `s1/s2 = 1.21` (ρ = 0.826) is not the operative decay
rate** — the operative rate is ~1.04–1.07. Combined with F4's requirement
(≤1.02–1.05), the honest reading is that the write is *already*
approximately as flat as the theory needs and still fails — a direct
prediction that spectral conditioning will not move the outcome.

## M2 — Non-normality: `§1.1` substitutes singular values for eigenvalues in one line, and the substitution is false for the matrices this program measures

`§1.1` sets up power iteration correctly with `ρ := |λ₂/λ₁|` (eigenvalues),
then writes *"`ρ = s₂/s₁ = 1`"* and every subsequent section — the bound
(§1.3), the penalty (§2(c)), Band 2 (§3.6) — operates on **singular values**.
For non-normal `Z` these are unrelated.

Counterexample at the design's own number (`attack2.py` N5(i)): embed
`[[0, 1.21], [1, 0]]` in d=25.

```
s1/s2 = 1.2100     |λ1| = |λ2| = 1.1000     |λ2/λ1| = 1.0000
pairwise cos across 24 queries:  h=1: −0.0075   h=61: −0.0096
```

**`s1/s2` exactly equal to the "measured collapsed" 1.21, and zero
directional collapse at any depth.** The premise "s1/s2 = 1.21 ⇒ collapse"
does not hold as a matrix fact.

This is not an exotic worry: the program's own `spectral_diagnostics()`
(`ncr_ortho_write.py:197`), which §3.2 ports verbatim, measures
`depart_normality` precisely because these operators are non-normal
(free_K24 depNrm 0.004–0.009; the ortho_K32 arms far worse).

## M3 — Mechanism (c)'s hinge is satisfied at exactly zero cost by a rank-2 collapse

`L_spectral` penalizes only `s1/s2`. Spectrum `(1, 1, 0.05, …, 0.05)`
(`attack2.py` N5(ii)):

```
s1/s2 = 1.000000  ->  hinge = EXACTLY 0 (penalty fully satisfied)
```

while the read is confined to a 2-dimensional subspace — 24 targets in a
25-dim space projected onto 2 dimensions, retrieval destroyed. §2(c)'s
pre-attack guards only `Z → 0`; this route is unguarded and is the *cheapest*
descent direction (equalize the top two, let the rest go).

The design's "existence proof" is about a **different quantity**: `§1.3` cites
toy `cond(A) = s1/s_min` of 1.0–1.1, then sets the target on `s1/s2`. Three
quantities are used interchangeably across §1.3/§2(c)/§3.6: `s1/s_min` (the
evidence), `s1/s2` (the penalty and the band), and `|λ2/λ1|` (the derivation).
The derivation actually requires the **whole spectrum flat**.

Compounding: at `d = K+1` the global ratio can be dominated by the single
spare direction, which `NCR_ORTHO_WRITE.md` §10.7 records as receiving
*"zero pressure"* and random-walking. §2(c)'s own failure-mode 3 flags
this, but the fix is to *score* the entity block alongside — the **penalty**
still pulls on the global ratio, so the cheapest gradient path is to move the
task-irrelevant spare direction and leave the entity block untouched.

## M4 — The `‖Z_raw‖_F²` normalization in `L_spectral` is unnecessary AND is an active escape hatch

`L_spectral = λ_c · max(0, s1/s2 − ρ_target)² / ‖Z_raw‖_F²`, justified as
*"normalized so shrinking Z can't evade it."*

`s1/s2` is already scale-invariant (`attack2.py` N6):

| scale a | s1/s2 of `aZ` | L_spectral |
|---|---|---|
| 0.1 | 1.138914 | 2.718e-04 |
| 1.0 | 1.138914 | 2.718e-06 |
| 10.0 | 1.138914 | 2.718e-08 |
| 100.0 | 1.138914 | 2.718e-10 |

(a) The guard was **never needed** — `Z → αZ` leaves the ratio fixed.
(b) It **creates** a runaway: `L ∝ 1/‖Z‖²`, so gradient descent zeroes the
penalty by growing `‖Z‖_F` without touching the ratio.

This is not hypothetical. `NCR_ORTHO_WRITE.md` §10.7 **measured** exactly this
degree of freedom on this write path: *"The cosine read is scale-invariant and
the pre-scale σ̂ is detached, so the loss exerts **zero pressure on σ_max**
(it drifts up freely — observed 5→13)."* The proposed normalization converts a
free drift into a *rewarded* one.

The correct guard against the true degeneracy (`s2 → 0`, ratio undefined) is
an epsilon floor on `s2`, or — better — abandoning the ratio entirely (D3).

## M5 — Band-1 applied to the design's own baseline: compB VIOLATES the paired bar; §0/§4's "far below the bar / healthy" is false against the raw JSON

`§0`: *"compB, 0994: TPC 0.196–0.228, **far below both** the paired-drift bar
and the 0.50 absolute tripwire."* `§4`: *"TPC 0.196–0.228 (**healthy**…)"*.

Recomputed from `mob_g3b31_compB_s0.json` (`attack4.py` N9):

| hop | TPC_fg | TPC_bo | bar = TPC_bo+0.15 | slack | Band-1 |
|---|---|---|---|---|---|
| h=1 | 0.20829 | 0.07678 | 0.22678 | +0.01849 | ok |
| h=2 | 0.20852 | 0.06743 | 0.21743 | +0.00892 | ok |
| h=3 | 0.21127 | 0.07821 | 0.22821 | +0.01694 | ok |
| h=5 | 0.21051 | 0.07660 | 0.22660 | +0.01608 | ok |
| h=12 | 0.19598 | 0.06434 | 0.21434 | +0.01837 | ok |
| h=20 | 0.21889 | 0.07831 | 0.22831 | +0.00942 | ok |
| h=29 | 0.22797 | 0.08390 | 0.23390 | +0.00593 | ok |
| **h=40** | **0.22725** | 0.07637 | 0.22637 | **−0.00088** | **VIOLATED** |
| h=61 | 0.20656 | 0.06633 | 0.21633 | +0.00978 | ok |

`§G3-B32` records this correctly (*"NULL-BY-COLLAPSE by the letter … single
paired-bar exceedance at h=40 (0.2272 vs bar 0.2264, +0.0008)"*). The
write-conditioning design drops the exceedance and promotes the cell to
"healthy / far below the bar." **The design's chosen baseline fails the
design's own §3.6 Band-1, which voids a cell "regardless of retrieval24."**

Worse, this is a *systemic* band-fragility, not a one-off. Paired difference
across 9 hops: mean 0.13855, SD 0.00657 → the 0.15 bar sits **1.74 SD** above
the mean. Per-hop exceedance ≈ 0.041; **P(≥1 of 9 hops) ≈ 0.31 per cell**
(normal approx, independence; hops are correlated so treat as an upper bound —
but the single available cell *did* trip, 1/9). Across 13 Stage-1 cells that
is an expectation of ~4 cells voided by eval noise before retrieval is even
looked at.

## M6 — Band-3's claimed provenance is false in two ways, and the baseline already scores the design's "never yet observed" PARTIAL signature

`§3.6` item 3: *"**NULL/COLLAPSE:** ≤ 2×chance = 0.0833 (**verbatim §G3-B29
frozen rule** — matches the measured unconditioned baseline exactly)."*

1. §G3-B29's rule is *"retrieval24 **MAX over ALL eval points/splits** =
   0.0417 (both LRs) ≤ 2×chance."* The design silently re-scopes it to the
   single point `h*=61`.
2. Under the MAX form, **the baseline FAILS it**: compB's max over eval points
   is **0.09375 at h=20 > 0.0833**. It does not "match the measured
   unconditioned baseline exactly."

`§3.6` item 4 registers the depth-decay PARTIAL signature — *"clears
WIN/PARTIAL at shallow-mid depth (h≤20) but decays toward chance by h*=61"* —
noting §G3-B32's *"no depth-decay PARTIAL signature anywhere."* compB's raw
numbers are **0.09375 at h=20 (inside the design's PARTIAL band) → 0.01562 at
h=61 (chance)**. The unconditioned NULL baseline scores the design's own
PARTIAL signature.

Statistics at n=64 (`attack4.py` N10): SD at chance = 0.0250; PARTIAL floor
0.0833 = **1.67 SD**; per-point false-PARTIAL probability 0.048; over 9 hops ×
2 arms ≈ **0.59 per cell**. The band has no statistical control whatsoever.
(The WIN bar 0.1917 = 6.0 SD is adequately conservative; the failure is
entirely on the PARTIAL side, which is where the "informative" outcomes live.)

## M7 — The +0.15 WIN margin is a category transplant, and no aggregation or multiplicity rule exists

`§3.6`: *"WIN: > chance+0.15 = 0.192 (**reuses the exact +0.15
absolute-margin convention §G3-B31 R1 already established for TPC, applied
here for internal consistency**)."* §G3-B31 R1's +0.15 is a **paired drift
tolerance on a cosine similarity** — a NULL-side allowance. Re-using the
numeral as a **positive-evidence margin on an accuracy metric** is aspiration
citing precedent. There is no CI requirement, no rule for aggregating 4 seeds
(the only pointer is an unreproduced gesture at `NCR_ORTHO_FALLBACK_DESIGN.md`
§A1.3's `1≤p≤3/4` trigger), and no multiple-comparison control across
4 seeds × 9 hops × 2 arms × 3 arms. Metric quantum is 1/64 = 0.015625.

## M8 — §1's frozen claim box attributes a measurement to cells that never produced it, and "provably" is unearned

`§1`: *"a capability the measured unconditioned write **provably** lacks
(§G3-B26/B32: **s1/s2=1.21 ⇒ retrieval24 at chance** at every tested depth,
**in every one of three independently trained arms, 0992/0993/0994**)."*

`s1/s2 = 1.21` was measured in §G3-B26 on `mob_g3b24_s0` — a **pre-contrastive**
checkpoint whose target space was catastrophically collapsed (TPC 0.9962,
trained adapter s1/s2 4.67, `o_pairwise` 1.00000, `o_var` 6.95e-08). **No
spectral measurement of `Z` exists for 0992/0993/0994** — I checked every
field of all three JSONs. The claim box transplants a number across
checkpoints and training regimes and presents it as a property of the three
contrastive arms.

The transplant is also numerically inconsistent (`attack6.py` N12b): with
queries drawn at compB's own TPC and the top singular pair aligned to the
query cone, a global `s1/s2 = 1.21` produces `o_pc@h=1 ≈ 0.24`; compB
**measures 0.7995**. Reproducing 0.80 at h=1 needs an effective top gap of
order 3–5×, not 1.21 — i.e. either the 1.21 does not describe compB's `Z`, or
the h=1 collapse has a non-spectral cause (a query-independent additive
component, which no spectral penalty touches). Either way §0's causal
sentence is unsupported.

`§1.3` then disowns the very bound §0 leans on (*"barely tighter than the
already-measured collapsed value, which is not credible as an actionable
target"*). §0 and §1.3 cannot both stand.

## M9 — Calibration runs at the wrong horizon and fits a mis-specified pooled model

- Cells 0.1/0.2 run **5,000 steps**; every Stage-1 cell runs **20,000**.
  CLAUDE.md's rule is explicit: *"A calibration run (**one real training run at
  the target config**) before a big sweep is mandatory."* The program's own
  record shows the relevant spectra move enormously with training (adapter
  s1/s2 1.01 at init → 4.67 at 20K, §G3-B26). A decay law fitted at 5K and
  applied to score 20K cells is instrument-relative extrapolation of exactly
  the kind CLAUDE.md's "instrument-relative admission" rule forbids.
- The gate says the points *"fit the log-linear decay law
  `log(sin θ) = log C + h·log ρ`"* as **one** regression. The six runs have six
  different `λ` and therefore six different `Z` and six different `ρ`. A pooled
  (C, ρ) fit is mis-specified; the R² < 0.5 escape hatch is measured on the
  wrong model. Correct form: per-run (C, ρ), then regress `ρ_fit` against the
  measured spectral ratio — which is also the only way to test M1/M2's claim
  that the two are unrelated.

## M10 — Arm (a) at n=2 is unscoreable under §3.6's own FAIL band

`§3.6`: *"**Mechanism-level FAIL (Gate-0…)**: if the conditioning term
prevents in-dist `recovered_frac@0.9 ≥0.9` in **≥3/4 seeds** … pre-registered
as a LIVE risk for **mechanism (a) specifically** given its direct
precedent."* `§3.4` gives mechanism (a) **2 seeds**. The band explicitly
scoped to (a) cannot be evaluated on (a). No split-result rule covers n=2
either.

**Answering the charter's question:** (a) should **not** be a Stage-1 arm in
wave-1. It carries 2.44 GPU-h (18% of nominal) against zero empirical
evidence that any hard orthogonal parametrization trains at this scale
(NS-polar: Gate-0 dead 4/4 at this exact K/d; damped-polar Stage 0: FAIL,
39/39 zero-recovery, retry never executed; expm/Cayley Stage 1: never
launched). Note the irony worth recording: **(a) is the only mechanism in the
family that actually reaches §1.1's stated target** (`expm(W−Wᵀ)` is exactly
scaled-orthogonal, and `NCR_ORTHO_FALLBACK_DESIGN.md` §3.1 Rev-1's spare-sign
parity argument — which I verified — makes the K-cycle target reachable inside
`SO(d)`), while the two PRIMARY arms target quantities the derivation does not
require (F3, M3). Keep the 0.12 GPU-h canary; drop the n=2 cell; promote to
n≥4 in wave-2 on ENGAGE.

## M11 — Cell 0.0's gate is undefined, unsequenced, and omits its own most load-bearing deliverable

- **Undefined criterion:** *"`o_pairwise_cos` at h≥3 within the **0989-family
  noise band** of the original 0.989–0.992."* Job 0989 is the decode-isolation
  probe; it produced no `o_pairwise_cos` family and no noise band. There is
  exactly one recorded value per hop from one cell. An unfalsifiable gate.
- **Unsequenced:** the §3.3 table lists 0.0 alongside 0.1–0.3 with no ordering
  constraint; §6 item 3 flags this as an open question and leaves it open. As
  written, ~1.4 GPU-h of calibration can be spent before the premise is
  checked.
- **Omitted deliverable:** the compB config has **no recorded `Z` spectrum at
  all** (M8). Cell 0.0 is the only place that number can come from, and Band 2
  is unscoreable without it (there is no baseline Δ). §3.3 frames 0.0 as
  replication only.

**Answering the charter's question:** no Stage-1 conclusion survives a 0.0
failure — and the finding 0.0 is meant to replicate is *already* contradicted
by the raw JSON (M5), so 0.0 as specified cannot replicate it. The real
precondition is F1's h=1 capability check, which no cell covers.

---

# PART III — minor

- **m1.** *"0.1/0.2's **42** (strength, h, sin θ) points"* (§3.3) and *"~40
  controlled points"* (§1.2). The runner's ladder is `train_hops (1,2,3)` +
  `deep_ladder (5,12,20,29,40,61)` = 9 hops (verified in the config block of
  all three JSONs). 6 runs × 9 hops = **54**.
- **m2.** §2 calls the pin *"the pinned `9a93198b…` **v2/v3**-instrumented
  runner (§G3-B27/B31)"*. §G3-B27's v2 patch md5 is `f307a7fd…`; `9a93198b…` is
  the v3 contrastive runner. **The pin claim itself is VERIFIED CLEAN:**
  `md5(experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/ncr_lm_wave1_runner.py)
  = 9a93198b642242f512ff8489e32b0a53`, matching §G3-B31 and §G3-B32.
- **m3.** §G3-B31 **N1** records a carried obligation: 4 runner docstrings cite
  the VOID `0.065–0.081` frozen-init basis, left unedited *only* to preserve
  the smoked md5, *"correct at next runner patch."* This design **is** the next
  runner patch and does not mention the obligation.
- **m4.** Arm (a) carries the 1.4× expm multiplier inside 2.32, then the global
  1.4× contingency on top — an effective 1.96×. Defensible (different
  purposes), undisclosed.
- **m5.** §1.3's 0.20 floor is verified as arithmetic (`NCR_REAL_LM_DESIGN.md:5120`:
  cosine SD between independent random unit vectors = 1/√25 = 0.20) but is a
  **cos-scale null SD applied to a sin-scale quantity**. The chain
  `sin θ_{h*} ≥ 0.20 ⇒ retrieval24 > chance` is asserted, never derived, and
  never validated — even though both quantities were measured on the same
  checkpoint (`mob_g3b24_s0`: sin θ_61 = 0.088, ret24 = 0.031–0.062).
- **m6.** No effective-distance control. `h*=61 ≡ 13 (mod 24)`; `GRIDS[24]` has
  `h_star = 189`, `ladder_residue = 21`, so h=61 is off the K=24 ladder
  entirely (§6 item 1 discloses the choice but not the consequence). An
  eval-only point at raw **h=13** (same effective distance, 1 squaring-round
  instead of ~6) costs ≈0 and is the single cheapest discriminator between
  "`Z` encodes the cycle and squaring destroys it" and "`Z` never encoded it" —
  the exact discrimination F1 says is missing.
- **m7.** Runway. ~19.2 GPU-h across 3 GPUs ≈ 4.5–5 h wall (13 Stage-1 cells,
  ⌈13/3⌉ = 5 waves). The standing durable-queue doctrine requires ≥2-day
  on-box runway; no successor payload for GPUs 4/6/7 is named. Placement itself
  is sound: STATE.md 2026-08-13 confirms GPUs 4/6/7 free, Jacobians on 0/2/3,
  K-wall orchestrator on GPU 5 (≤15.50 GPU-h), PI server on GPU 1; the
  no-packing ruling is correct at 73–80% SM util.

---

# PART IV — VERIFIED CLEAN (what survived the round)

Recorded so a later round does not re-litigate these.

| item | status |
|---|---|
| Runner md5 pin `9a93198b642242f512ff8489e32b0a53` | **VERIFIED** against the repo artifact |
| Budget arithmetic | **VERIFIED**: recomputed Stage-0 **2.1912** (doc 2.21), Stage-1 **11.5040** (doc 11.5), nominal **13.6952** (13.7), ×1.4 = **19.1733** (19.2), headroom to 30 = **10.83** (doc ~10). Every per-cell line reproduces from 0.83 GPU-h/cell × step fraction × multiplier |
| `ĉ = tr(Z)/d` as `argmin_c ‖Z − cI‖_F²` | **VERIFIED** (derivation in §2(b) is correct) |
| expm reachability of the K-cycle target inside `SO(d)` | **VERIFIED** against `NCR_ORTHO_FALLBACK_DESIGN.md` §3.1 Rev-1 (spare-sign parity: `det(Q) = (−1)^{K−1}·s`, K even ⇒ `s=−1` gives `det=+1`); the citation is accurate and the superseded FATAL det-parity claim is correctly not reused |
| expm FLOP estimate | **VERIFIED**: 24·d³ at d=25 = 3.75×10⁵, negligible vs a 98M backbone |
| `h*=61 mod 24 = 13`, clear of residues {0,1,2,3} | **VERIFIED**; `GRIDS[24].h_star = 189` also correct |
| Placement reuse (6.86 GB, 73–80% SM, 0.83 GPU-h/cell) | **VERIFIED** against §G3-B31 PLACEMENT and the JSONs' `gpu_h` (0.8293) |
| §1.1's substantive orthogonal-invariance claim | **VERIFIED** numerically (only the metric-naming sentence is wrong — F2) |
| Blank-out / P=1 battery reuse | Sound; a genuine invariant, correctly bundled at no GPU cost |
| §2(b) failure-mode 3 (degenerate `ĉ`) | A real risk, correctly identified at design time |
| Novelty charter (§5) construction | Well-formed; MuonSSM rank-1-vs-full-rank and transition-vs-write distinctions are the right axes. **Not adjudicated here** — that is the triple-sweep's job, and it is moot until the premise is re-established |

---

# PART V — BINDING DISPOSITION PROPOSAL

Ordered. **D1–D5 are blocking**; the design cannot re-enter the build queue
until each is discharged in the registry.

**D1 (blocking, F1 + M8 + M11 + m6) — PREMISE CELL, before any mechanism spend.**
Replace §3.3 cell 0.0 with a premise-establishing measurement, run FIRST and
gating everything:
  1. **Eval-only** on the retained `mob_g3b31_compB_s0` checkpoint if it
     survives on the box (≈0 GPU-h); otherwise one fresh 20,000-step cell at
     compB's exact recipe (0.83 GPU-h).
  2. Must record: exact `torch.linalg.svd` spectrum of `Z` — **global and
     entity-block `A = UᵀZU`** — the number that exists nowhere for this
     config; `depart_normality` and `|λ2/λ1|` (M2); `retrieval24`,
     `offtarget_margin`, `o_pairwise_cos` at raw **h ∈ {1, 13, 37, 61}**
     (13 and 37 are ≡ 13 mod 24 — same effective distance, different squaring
     counts).
  3. **GATE (pre-registered, hard):** `full_graft` retrieval24 at h=1 must
     exceed chance by a stated margin (at n=64, chance + 4 SD = 0.142; or raise
     the eval n and re-derive). **If it does not clear, write-conditioning is
     not the binding lever and this design does not launch** — the real
     question becomes why the h=1 write→read association never forms, which is
     a different (and cheaper) investigation.

**D2 (blocking, F2 + F4 + M1 + M9) — Rewrite §1 and re-derive the target.**
  - Correct the metric: the success condition is `sin θ_h ≈ sin θ₀` (constant),
    not `sin θ_h = 0`.
  - Delete the `C ≈ 9,900` fit (inadmissible, M1) and the 1.19 bound built on
    it.
  - **Re-pre-register the instrument** as `o_pairwise_cos(h)` — already
    measured, directly coupled to retrieval, and free of the sinθ→retrieval
    inferential gap (m5).
  - Re-derive `ρ_required` from `o_pc(h)`; the honest current estimate is
    `s1/s2_eff ≤ 1.02–1.05`, and note explicitly that the quantity **saturates
    above ~1.02**, so the target must be stated on the effective (whole-
    spectrum) ratio, not the global top-two ratio.
  - Calibration cells run at **20,000 steps at the target config** (CLAUDE.md),
    with **per-run** (C, ρ) fits and a `ρ_fit` vs measured-spectrum regression,
    not one pooled fit; 54 points, not 42.

**D3 (blocking, F3 + M3 + M4) — Retire (b) and (c) as specified; replace with one smooth conformality penalty on the entity block.**
```
A = Uᵀ Z_raw U                       # entity block, U from the existing az.entity_subspace machinery
G = Aᵀ A
L_conf = λ · ‖ G − (tr(G)/K)·I_K ‖_F²  /  (tr(G)/K)²
```
Properties, all verified above or trivially: scale-invariant (no `‖Z‖`
normalization needed, M4); smooth and degeneracy-free (no SVD, no power
iteration, no singular-vector chatter at `s1≈s2`, M3/§2(c) failure-mode 2);
**exactly zero at every `c·Q` including the K-cycle target** (F3's table:
`L_correct(Z*) = 0`, `L_correct(c·I) = 0`, `L_correct(50/50 mush) = 5.18e-03`);
localized to the task subspace by construction (M3); one `K×K` matmul
(`O(K³)` ≈ 1.4×10⁴ FLOPs, cheaper than the proposed top-2 power iteration).
This single term subsumes the intent of both (b) and (c) without either's
degenerate optimum.

**D4 (blocking, F5) — Fix the placebo.** `R_rand` orthogonal is the active
ingredient, not a null. Replace with a **spectrum-matched, structure-randomized**
anchor (random matrix carrying `Z_raw`'s own singular-value profile) or a
Frobenius-matched Gaussian anchor. Re-state the pre-registered interpretation:
under the corrected control, placebo-improves ⇒ nuisance-gradient effect;
under the *current* control, placebo-improves would have been *evidence for*
the flatness hypothesis.

**D5 (blocking, M5 + M6 + M7) — Re-anchor every band to the raw artifacts.**
  - Band 1: state that compB **violates** the paired bar at h=40 (0.22725 vs
    0.22637) and is NULL-BY-COLLAPSE by the letter; then either re-register the
    rule (hop-median, or a paired CI) or accept and disclose the ~31%/cell
    noise-void rate. Do not describe the baseline as "healthy / far below the
    bar."
  - Band 3: strike the false "verbatim §G3-B29 / matches the baseline exactly"
    provenance (§G3-B29's rule is MAX-over-all-points, which compB's 0.09375
    fails). Recompute PARTIAL/WIN from n=64 binomial statistics with an
    explicit seed-aggregation rule and a multiplicity correction across
    9 hops × 2 arms × 4 seeds. Record that compB itself scores the item-4
    depth-decay PARTIAL signature (0.09375@h=20 → 0.01562@h=61).
  - Justify the WIN margin on its own statistics, not by transplant from a TPC
    drift tolerance (M7).

**D6 (non-blocking) — Correct §1's frozen claim box.** `s1/s2 = 1.21` is a
`mob_g3b24_s0` measurement on a target-collapsed pre-contrastive checkpoint,
not a 0992/0993/0994 measurement. Strike "provably."

**D7 (non-blocking, M10) — Re-scope arm (a).** Canary 0.3 only in wave-1
(0.12 GPU-h). Drop the n=2 Stage-1 cell (unscoreable under §3.6's own ≥3/4
rule). Promote to n≥4 in wave-2 on canary ENGAGE. Reallocate the freed 2.32
GPU-h to the corrected control (D4) and the premise cell (D1) — **not** to more
seeds of a penalty whose target is mis-specified.

**D8 (non-blocking) — m1, m2, m3, m4, m6, m7** as listed; in particular carry
§G3-B31 **N1**'s docstring obligation into whatever patch lands, and name a
successor payload for GPUs 4/6/7 so the ≥2-day runway doctrine holds.

**Ceremony note (against §6 item 6).** This round found 5 FATALs, three of
them in the frame rather than the build. That is the signature of a
>50-GPU-h/publication-bound tier, not a 10–50 tier — regardless of the price
tag. Recommend the **full multi-round gauntlet** on the revised design, with
D1's premise cell adjudicated *before* the mechanism family is re-designed.

---

*Round-1 attack agent. Read-only except this file. No STATE.md /
EXPERIMENT_LOG.md edits, no commit, no box contact.*
