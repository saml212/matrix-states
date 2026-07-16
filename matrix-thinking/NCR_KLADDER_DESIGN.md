# NCR ORTHO-WRITE K-LADDER — SCALE-UP DESIGN

**DRAFT-STAGE-1 (PRE-ATTACK, CONDITIONAL on the ortho-write verdict), dated
2026-07-16.** This document is written BLIND to the live run in
`experiment-runs/2026-07-16_ncr_ortho_write/` / any `results_ortho_write`
path — no such path was read while drafting this design. This is a
CONDITIONAL design: it does not authorize any GPU spend by itself. It
executes ONLY if the running orthogonal-write pre-registration
(`NCR_ORTHO_WRITE.md` §4) returns **WIN** or the pre-registered **PARTIAL**
band for Part A. See §9 for the exact branch. Everything costed below is a
plan, not a launch.

**Reused, unmodified inputs (grounding every number below):**
- `NCR_ORTHO_WRITE.md` §2 (Newton–Schulz polar write, `NS_ITER_DEFAULT=40`,
  `NS_POWER_DEFAULT=12`), §3 (h\* re-registration method), §4 (WIN/PARTIAL/
  NULL/FAIL bands, Gate-0), §6 (measured rates), the CEILING AMENDMENT
  (2.8 h primary / 4.24 h discriminator, 320K steps, measured on-box).
- `NOVEL_ARCH_WATERFALL.md:3899-3901` — the pinned param/FLOP formulas
  `P(d,h)=40h²+4dh+46h+d`, `F(K,d,h)=76Kh²+4dh²+12K²h+4Kdh+4d²h` — and its
  own documented finding that FLOP-ratio scaling was **empirically refuted**
  at small K (4 measured cells flat within noise despite 1.688×–2.102× FLOP
  spread — kernel-launch/small-batch overhead-bound, not compute-bound).
- `ncr_task.py` `_gen_grid(K)`: `ladder_residue=K-3`, `ladder = m·K-3` for
  `m∈{1,2,4,8,16,32,64,128}`, synthetic `h_star=8K-3` — already defined for
  every `K∈{20,32,48,64,96,128,192,256}`, reused verbatim below (no new
  grid-generation code needed).
- `ncr_ortho_write.py`: `NS_ITER_DEFAULT=40`, `NS_POWER_DEFAULT=12`,
  `TRAIN_BATCH=256` (`run_ncr.py:78`), `d=K+1` tight-spare convention,
  encoder hidden `h=64` fixed across K (`GRID_SHAPES[K]["h"]`).

---

## §1 HYPOTHESIS (one sentence)

If the Newton–Schulz orthogonal write cracks realistic-depth composition at
K∈{24,32} (WIN or PARTIAL), the SAME write mechanism — trained fresh at each
larger K∈{48,64,96,128} under the untested `d=K+1` extrapolation — will keep
producing a near-orthogonal, well-conditioned entity operator whose recovery
at a K-relative realistic depth (`h*=K+8`, "one full relational cycle plus a
novel residue") stays ≥0.9, i.e., the crack is a property of the
orthogonal-write mechanism and not an artifact special to K≈32; the ladder's
job is to find out whether that holds flat, decays gracefully, or hits a new
cliff.

---

## §2 GRID + PER-K STATE DIMENSION + FLOPs/MEMORY

**Grid justification.** The task-specified `K∈{48,64,96,128}` is KEPT. Ratio
between successive rungs is 1.5×/1.33×/1.5×/1.33× — a moderate geometric
step, not a pure power-of-2 jump (32→64→128→256 was considered and
rejected: a 256 jump has zero validated calibration point anywhere near it,
and a 4-point ladder needs resolution near the boundary where uncertainty is
highest, not one giant untested leap). Four points is the minimum for a
crude power-law/decay fit with one point to spare for a sanity check.

**d convention: `d=K+1` (tight-spare), UNCHANGED — carried by extrapolation,
not validation.** This is the single biggest structural assumption in this
design (flagged again in §5). Precedent: tight-spare `d=K+1` reached 4/4
Gate-1 CONVERGED at 1× budget at BOTH K=16 and K=24
(`NCR_NEXT_LEVER_DESIGN.md` §2.1), and the currently-running wave extends it
to K=32. **Nothing has validated it past K=32.** A sharper way to see the
risk: the convention holds the ABSOLUTE headroom fixed at exactly one
dimension (`d-K=1`) while the RELATIVE headroom `1/d` shrinks monotonically
— 5.9% at K=16, 4.0% at K=24, 3.0% at K=32, and by this ladder: **2.0% at
K=48, 1.5% at K=64, 1.0% at K=96, 0.78% at K=128.** If trainability depends
on relative (not absolute) spare capacity, this ladder could fail Gate-0 for
a reason that has nothing to do with orthogonality — the calibration cell
(§5) exists partly to catch exactly this confound. Pre-registered fallback
if Gate-0 fails at the calibration K under `d=K+1`: retest that SAME K once
at `d=1.25K` before concluding anything about the K-ladder itself (mirrors
the FAIL clause structure in `NCR_ORTHO_WRITE.md` §4 Part A) — not a silent
swap.

**Params/FLOPs — closed form, `h=64` fixed (encoder hidden unchanged
across K, `GRID_SHAPES[K]["h"]`), no new formula invented:**

`P(d,h) = 40h² + 4dh + 46h + d`
`F(K,d,h) = 76Kh² + 4dh² + 12K²h + 4Kdh + 4d²h`

Plus the ortho-write-SPECIFIC term absent from the base formula (the
Newton–Schulz polar projection itself, `n_iter=40` matmul-pairs + `n_power=12`
power-iteration matvecs on a `d×d` matrix per example, forward-FLOP count,
`2mnk` convention): `NS(d) = 160d³ + 48d²`. Verified against the K=32/d=33
cell: `NS(33)=5,802,192`; `P(33,64)=175,265` — matches the pre-registration's
own disclosed "≈175K params" for the K=32 ortho cell almost exactly.

| K | d=K+1 | K/d | 1/d (rel. headroom) | P(d,h) params | F(K,d,h) | NS(d) | F+NS | ratio vs K=32 |
|---|---|---|---|---|---|---|---|---|
| 32 (ref) | 33 | 0.9697 | 3.0% | 175,265 | 11,837,696 | 5,802,192 | 17,639,888 | 1.000 |
| 48 | 49 | 0.9796 | 2.0% | 179,377 | 18,731,264 | 18,939,088 | 37,670,352 | 2.136 |
| 64 | 65 | 0.9846 | 1.5% | 183,489 | 26,280,192 | 44,142,800 | 70,422,992 | 3.992 |
| 96 | 97 | 0.9897 | 1.0% | 191,713 | 43,344,128 | 146,479,312 | 189,823,440 | 10.761 |
| 128 | 129 | 0.9922 | 0.78% | 199,937 | 63,029,504 | 344,269,008 | 407,298,512 | 23.090 |

**Two findings worth flagging, not just the table.** (1) Params stay
trivial end to end (175K→200K, +14%) — the "never the constraint" precedent
holds for PARAMS. (2) The Newton–Schulz term overtakes the base task's own
FLOPs at K≈48 (`NS=18.94M` vs `F=18.73M`) and dominates by K=128 (`NS`
5.5× `F`). **This K-ladder is, computationally, increasingly a
Newton–Schulz-cost ladder, not an NCR-task-cost ladder** — a direct
consequence of `n_iter=40` being held FIXED (a cubic-in-`d` cost) while `K`
scales. This is presented as an observation, not a proposal to change
`n_iter` (a science parameter change requires its own pre-registration
amendment, triggered only if the calibration cell shows `n_iter=40`
insufficient at high `d` — see §5, §7).

**Memory (order-of-magnitude, `TRAIN_BATCH=256`).** Params+optimizer(Adam,
2×)+base activations stay kilobyte/example-trivial as before. The NEW driver
is the NS backward graph: autograd must retain ≈2 `d×d` tensors per
iteration for `n_iter=40` iterations (the un-checkpointed default) ≈
`320·d²` bytes/example:

| K | NS backward mem/example | NS backward mem/batch (256) |
|---|---|---|
| 32 (ref) | ~0.35 MB | ~89 MB |
| 48 | ~0.77 MB | ~197 MB |
| 64 | ~1.35 MB | ~346 MB |
| 96 | ~3.01 MB | ~772 MB |
| 128 | ~5.33 MB | ~1.36 GB |

At K=32 this is a small slice of the PI-reported 1.7GB total. By K=128 it
could be a **comparable-or-larger slice** (~1.4GB vs. the ~1.6-1.7GB
"everything else" baseline) — a real, K-growing memory driver, still
trivial relative to an 80GB H100, but no longer negligible the way the base
model's memory always has been. Confirm with real `nvidia-smi` numbers at
the calibration cell (§5), don't trust this estimate blind.

---

## §3 h\* SELECTION PER K — explicit mod-K arithmetic (avoiding the collapse)

Generalizes `NCR_ORTHO_WRITE.md` §3's method exactly rather than inventing a
new one. At K=32 that section picked `h*=40=K+8` (residue 8, novel,
∉{0,1,2,3}), and reused the audited `GRIDS[32]` ladder's own first two rungs
(`m=1`→29=K-3, `m=2`→61=2K-3, both residue `K-3`) as the "nearer" and
"stretch" checkpoints. The SAME schema generalizes cleanly to every K in
this ladder because `2K-3 ≡ K-3 (mod K)` for any K — the ladder's own two
lowest rungs are always on the SAME residue class, and `K+8` is always a
DIFFERENT, independently novel residue (`(K+8) mod K = 8`, and `8∉{0,1,2,3}`
for every K≥48). Shallow sanity points `{5,12,20}` are kept as absolute
(not K-scaled) depths — they were never K-scaled at K=32 either (`5,12,20 <
32`), and for K≥48 they remain trivially `<K`, so their residue equals their
raw value, trivially novel.

| K | near = K-3 | h\* = K+8 (primary) | stretch = 2K-3 | synthetic h_star=8K-3 (disclaimed, NOT chased) |
|---|---|---|---|---|
| 48 | 45 (mod 48 = 45) | **56** (mod 48 = 8) | 93 (mod 48 = 45) | 381 |
| 64 | 61 (mod 64 = 61) | **72** (mod 64 = 8) | 125 (mod 64 = 61) | 509 |
| 96 | 93 (mod 96 = 93) | **104** (mod 96 = 8) | 189 (mod 96 = 93) | 765 |
| 128 | 125 (mod 128 = 125) | **136** (mod 128 = 8) | 253 (mod 128 = 125) | 1021 |

Per-K realistic ladder: `{5, 12, 20, K-3, K+8, 2K-3}`. All six residues
verified ∉{0,1,2,3} for all four K (mechanical check, not assumed): `5, 12,
20` are literal values `<K`, always novel; `K-3` and `2K-3` share residue
`K-3` which is `≥45` for K≥48, never 0-3; `K+8` has residue `8`, never 0-3.
Training-depth multiple of the primary target grows with K (`h*/3`): 18.7×
(K48), 24× (K64), 34.7× (K96), 45.3× (K128) — all comfortably past the "≥10×
training depth" bar the K=32 pre-registration used, growing MORE ambitious
(not less) as K increases, since `h*=K+8` scales with K while training depth
stays pinned at 3.

**Part B (discriminator) needs no new arithmetic.** Depth there is
compositions-of-DISTINCT-operators (`L`), never a power of one matrix, so
`L mod anything` collapse cannot occur BY CONSTRUCTION (§4b's three guards).
`L∈{1,2,3}` train / `{5,8,12,16,20,24,32,40}` eval / `L*=32` stays IDENTICAL
across every K in this ladder — nothing to re-derive.

---

## §4 SEEDS + CELL GRID + GPU-h PRICING

**Measured base rates (the ONLY pricing inputs, per instruction — from
`NCR_ORTHO_WRITE.md`'s CEILING AMENDMENT, K=32, 320K steps, on-box):**
primary (single-relation) cell **2.8 h**; discriminator (R=4 bank) cell
**4.24 h** (the measured WORST case).

**Cost-scaling assumption, stated explicitly AS AN ASSUMPTION (not fact):**
project cost with the `(F+NS)` FLOP ratio from §2 — i.e., assume
COMPUTE-BOUND scaling from the K=32 measured rate. This is deliberately the
CONSERVATIVE (over-pricing) choice for planning/ceiling purposes: the
project's own established finding at small K (`NOVEL_ARCH_WATERFALL.md`
§9.10/§11) is that these cells are actually OVERHEAD-BOUND — measured rates
were FLAT within noise despite up to 2.1× FLOP spread. That precedent argues
the TRUE cost at K=48/64 will likely track well BELOW the FLOP-ratio
estimate. But the FLOP spread in THIS ladder reaches 23× (not 2×), driven by
the cubic `NS(d)` term (§2) — a regime the small-K precedent never tested,
and where compute-bound behavior becoming real is entirely plausible. Given
the last ceiling was mispriced 2× LOW by trusting a similar per-cell
extrapolation (the 3.0h→6.0h amendment), this design prices the CEILING off
the pessimistic assumption and gates actual spend behind a calibration cell
(§5) that measures the truth before commit.

**Nominal (FLOP-ratio-scaled) worst-case per-cell hours:**

| K | primary (×2.8h) | disc (×4.24h) |
|---|---|---|
| 48 | 5.98 h | 9.06 h |
| 64 | 11.18 h | 16.93 h |
| 96 | 30.13 h | 45.63 h |
| 128 | 64.65 h | 97.90 h |

**Cell grid (this design's scope, cost-trimmed by construction, not by
seed-count alone).** Part A (single-relation, ortho arm ONLY — the
free-write baseline is NOT re-run at each new K; it is already established
DEAD at K=24/32 both behaviorally and mechanistically, §4's own pinned
baseline-reuse precedent, and this wave is conditional on ortho already
having WON/PARTIAL'd against it) runs at **all four K, n=4 seeds each** —
the primary trainability + scaling-law axis this design exists to answer.
Part B (discriminator, ortho-bank ONLY, same free-bank-reuse logic) runs
ONLY at the two ENDPOINTS, **K=48 and K=128, n=4 seeds each** — enough to
check the mod-K-trap-safe compositional result generalizes to a near point
and a far point without paying for the full 4-K sweep on both arms.
Contingency: if K=48's ortho-bank cell breaks the expected pattern, the two
skipped middle K's (64, 96) for Part B become a pre-registered follow-on,
not silently dropped forever.

**Nominal (worst-case, FLOP-scaled) total for this grid:**

| arm | K48 | K64 | K96 | K128 | subtotal |
|---|---|---|---|---|---|
| A (n=4) | 23.9 h | 44.7 h | 120.5 h | 258.6 h | 447.7 h |
| B (n=4, endpoints only) | 36.2 h | — | — | 391.6 h | 427.8 h |
| **total** | | | | | **≈875.5 h** |

This number is a SICKER-PRICE upper bound, not a plan — K=128 alone (A+B,
650.2 h) is 74% of it. It exists to make the case for §5/§7's gating
concrete: if the compute-bound assumption is even roughly right, K=128
could eat this entire wave's budget by itself. **Planning cap for the
COMMITTED sweep (post-calibration): 150 GPU-h**, chosen as roughly 2× the
prior ortho-write wave's actual spend (≈77 GPU-h, 24 cells) to cover the
ladder's added scope — an adjustable planning number, not an external
constraint. Pinned trim order if the calibration-corrected projection still
exceeds the cap (mirrors `NCR_NEXT_LEVER_DESIGN.md`'s own trim-highest-cost-
first precedent):
1. Drop Part B at K=128 (already provisionally deferred — see §5).
2. Trim Part A K=96 from n=4 → n=2, then → n=1 (confirmatory-only,
   disclosed per the standing sub-4-seed-never-gates rule).
3. Drop Part B at K=48.
4. Trim Part A K=128 from n=4 → n=1.
5. **Floor, never trimmed:** Part A at K=48 and K=64, n=4 — the two
   nearest, cheapest, most decisive "does the crack survive AT ALL past
   K=32" rungs; this is the wave's minimum viable deliverable.

---

## §5 CALIBRATION-CELL-FIRST PLAN (mandatory, before any sweep cell launches)

CLAUDE.md's calibration rule exists for exactly two failure modes this
ladder is at real risk of: (a) a convergence ceiling — Gate-0 silently
capping below 0.9 at larger K under an unvalidated `d=K+1` (§2's
shrinking-relative-headroom risk), and (b) a "bigger model" guess that
diverges or blows the wall-clock budget before a sweep's compute is
committed (§4's 875h sticker-price risk). A THIRD, ortho-write-specific risk
belongs here too: **NS convergence quality is untested past d=33.**
`n_iter=40` was calibrated to fully orthogonalize a "measured cond#≈8500 at
random init" case at the K=24/32 build; condition numbers of random d×d
matrices grow with d, so `n_iter=40` might not fully orthogonalize the
K=128/d=129 write — a failure mode invisible to Gate-0 (training could still
converge) but fatal to the far-depth claim (an imperfectly orthogonal
operator still decays weak modes, just more slowly).

**Stage 0 (mandatory, blocks everything else): ONE calibration cell, K=128
(the top target config, deliberately the hardest, not the cheapest), Part A
ortho arm, single seed, run to completion (or to its own tighter abort
trigger, below) BEFORE any other cell in this ladder launches, packed or
not.** It answers three things at once:
1. **Real wall-clock** → computes a realized-vs-predicted ratio against the
   FLOP-scaled 64.65h estimate (§4), which corrects the pricing for K=48/
   64/96 too (interpolated, not re-derived from scratch — same qualitative
   regime assumed unless Stage 0 says otherwise).
2. **Gate-0 pass/fail** → catches the §2 relative-headroom convergence-
   ceiling risk directly, at the hardest point on the ladder.
3. **Orthogonality quality at convergence** (`orthogonality_error`,
   already-instrumented per `NCR_ORTHO_WRITE.md` §2) → catches the
   NS-convergence risk above. Target: `‖QᵀQ-I‖_F` at machine-precision-ish,
   matching the K=32 behavior; a materially non-zero residual at K=128 is
   an explicit ABORT-and-redesign trigger (candidate fix: raise `n_iter`, a
   science-parameter change requiring its own pre-registration amendment —
   not a silent bump).

**Decision rule after Stage 0:**
- **PASS** (Gate-0 holds, orthogonality tight, wall-clock finite and
  reported): re-price the full grid using the realized rate (not the FLOP
  ratio), apply §4's trim order against the 150h cap using REAL numbers,
  launch the committed sweep K=48→64→96→128 in that order (nearest-first,
  not all at once — see next bullet).
- **STAGED escalation, not a flat launch.** Even after PASS, commit K=48
  first (cheapest, closest to the validated K=32 regime), confirm its own
  real rate and Gate-0 result, THEN advance to K=64, etc. — each rung's
  real numbers refine the next rung's price before it is committed. This
  costs a little wall-clock in sequencing but removes the single biggest
  risk in this design (a blind flat commit against a possibly-wrong cost
  model).
- **FAIL** (Gate-0 dead, or `orthogonality_error` not tight): STOP. Do not
  launch K=96/K=128 under `d=K+1`. Escalate per §2's pre-registered fallback
  (retest K=128 once at `d=1.25K`) before concluding anything about the
  ladder's upper end. K=48/K=64 may still be launched independently (they
  are closer to the validated regime and the calibration failure may be
  K=128-specific, e.g., the shrinking relative-headroom effect) — but ONLY
  after their own cheap rate-probe confirms sane per-cell cost (a 500-step
  probe per K, mirroring `NCR_NEXT_LEVER_DESIGN.md`'s own Phase-0a pattern,
  ≈0.05 GPU-h aggregate, negligible).
- **Part B (disc) at K=128 is explicitly DEFERRED, not committed, pending
  Stage 0's realized-vs-predicted ratio.** If Stage 0 shows costs tracking
  close to the FLOP ratio (genuinely compute-bound), Part B at K=128
  (nominal 97.9h/cell, heaviest single cell type in this design) is very
  likely out of this wave's budget and should be scoped as a separate,
  PI-gated follow-on rather than folded in here.

---

## §6 DRAFT WIN/NULL BANDS PER K + THE RECOVERY-vs-K SCALING-LAW READOUT

**Per-K bands** (Gate-0 precondition unchanged and K-general: `min over
h∈{1,2,3} rec@0.9 ≥0.9` AND `mean A_eff_rank ≥0.9K`; thresholds below mirror
`NCR_ORTHO_WRITE.md` §4 Part A exactly, generalized by K's own re-registered
depths from §3):

- **WIN(K):** median rec@0.9 at `h*=K+8` ≥0.9 across Gate-0-passing seeds;
  free-write baseline (extrapolated-dead per §4's reuse logic, not
  freshly re-measured) stays <0.5 at the same depth; mechanistic
  corroboration unchanged (departure-from-normality ≤0.02, cond#≤~2,
  min|λ|/c\*≥0.9) — these thresholds describe a well-orthogonalized
  operator and are not themselves K-dependent.
- **PARTIAL(K):** median rec@0.9 ≥0.9 at the `K-3`/`2K-3` checkpoints but
  <0.9 at `h*=K+8` — the wall cracks to one full cycle but not past it.
- **NULL(K):** rec@0.9 <0.9 at every depth ≥ `K-3`, Gate-0 still passes —
  the write trains but buys no far-depth gain at this K.
- **FAIL(K):** Gate-0 DEAD in ≥3/4 seeds — the constraint (or the untested
  `d=K+1` convention at this K) breaks trainability.

**The scaling-law readout — three pre-registered shapes, decided before
data, not after:**
- **(a) FLAT-HOLD:** WIN at all four K. Publishable as "the orthogonal
  write's realistic-depth composition ceiling is K-general" — the strongest
  possible outcome, no evidence of a new wall inside this ladder's range.
- **(b) GRACEFUL DECAY:** WIN/PARTIAL at K=48/64, degrading toward
  NULL by K=96/128, with `rec@h*(K)` roughly monotonically falling. Report
  the fitted trend (simple monotonic fit against K or log K — four points is
  enough to characterize direction and rough rate, not enough for a tight
  power-law exponent) and the point estimate for where `rec@h*` crosses back
  under 0.9 — i.e., report a NEW, higher K-wall location, not just "it got
  worse."
- **(c) CLIFF:** WIN holds at K=48 (and maybe 64), then a sharp drop to
  NULL/FAIL at a specific rung. Cross-check against the mechanistic
  diagnostics: if the cliff co-occurs with `orthogonality_error` going
  non-tight (NS not fully converging at that d, §5's third risk), it is an
  NS-iteration-count problem (plausibly fixable); if Gate-0 itself fails
  with clean orthogonality, it is a genuine SGD-trainability boundary
  (harder, matches the free-write K-axis wall's own character).

All three outcomes are informative and none require a follow-up experiment
to be reportable — this is itself a discipline point (mirrors this
project's own "negative results are data" rule).

---

## §7 ABORT CRITERIA + CEILING PRICING (headroom from the measured WORST case)

**Two DISTINCT thresholds, not one** — the amendment this design makes over
the prior ceiling-mispricing incident (3.0h→6.0h, §CEILING AMENDMENT):
a **budget-planning ceiling** (how much to reserve) is not the same number
as a **runaway-abort trigger** (when to kill a hung/stuck cell), and
conflating them is exactly what under-priced the last one.

**Budget-planning ceiling per cell**, priced from the measured WORST case
(4.24h, the discriminator rate — used as the base even for Part A cells'
ceiling, per instruction to price off worst-case not mean) scaled by the
§2 FLOP ratio, with a 2× margin (tighter margins were tried before and
under-priced; this design deliberately doubles the prior 1.4-1.5× margin
given the ladder reaches untested K):

`ceiling(K) = 2 × ratio(K) × 4.24 h`

| K | ceiling (solo) |
|---|---|
| 48 | 18.1 h |
| 64 | 33.9 h |
| 96 | 91.3 h |
| 128 | 195.8 h |

A 196h SOLO ceiling for one cell is not a real operating threshold — it is
a worst-case sticker price. **The actual runaway-abort trigger is an
early-step-rate projection, not the ceiling above:** after the first 2,000
of 320,000 steps (<1% of budget, mirrors `NCR_NEXT_LEVER_DESIGN.md`'s own
500-step rate-probe discipline, cost negligible), extrapolate total
wall-clock; if the extrapolation exceeds **24h for the Stage-0 calibration
cell** or **1.5× that cell's own ceiling(K)** for any sweep cell, ABORT
immediately rather than waiting for the full ceiling to elapse. This turns
"is this cell healthy" into a decision made in minutes, not days.

**Gate-0 abort (mirrors `NCR_ORTHO_WRITE.md` §4 Part A FAIL clause
verbatim):** if a K-rung's Gate-0 fails in ≥3/4 seeds, STOP that rung
immediately — do not spend the remaining seed budget chasing it with more
compute. This is a design question (§2's fallback, or a `n_iter` amendment
per §5), never a "try again with more steps" question.

**Packed-cell ceiling correction (feeds §8):** when N cells share a GPU,
each cell's OWN wall-clock lengthens by the contention factor (§8 estimates
~1.3× at N=2); the per-cell abort trigger above must use the CONTENDED
projection, not the solo one, or a healthy packed cell gets killed for
looking "slow" relative to a solo baseline it was never running under.

---

## §8 SATURATION-PACKING PLAN (PI directive 2026-07-16)

**Rejected baseline:** current cells draw ~44-59% SM and ~1.7GB on an
80GB H100 — both badly under-saturated. §2's memory analysis shows this
profile is NOT expected to persist across the whole ladder (NS backward
memory grows as `d²`, reaching an estimated ~1.4GB extra by K=128 on top of
the ~1.6-1.7GB baseline) — meaning the RIGHT packing factor is **K-dependent**,
not one fixed N for the whole ladder.

**K=48/64 (near the measured 44-59%/1.7GB profile): pack N=2 cells/GPU.**
Predicted per-GPU memory: `2 × (1.7-2.5 GB) ≈ 3.4-5.0 GB` — trivial against
80GB, memory is not the limiter here. Predicted per-GPU SM utilization:
`2 × (44-59%) ≈ 88-118%` — near-saturating, plausibly mildly oversubscribed
at the high end. **Contention-factor assumption (stated explicitly, to be
confirmed empirically, not asserted as fact): each packed cell runs ~1.3×
slower in wall-clock than solo**, giving a net GPU-throughput gain of
`2/1.3 ≈ 1.54×` cells-per-GPU-hour despite the per-cell slowdown — a real
saturation win. First packed batch at each new N is itself a mini-calibration:
watch realized `nvidia-smi`/DCGM SM% and per-cell step-rate for the first
~5,000 steps before trusting the 1.3× assumption for the rest of the rung.

**K=96/128 (NS-dominated, §2): default to N=1 (no packing) until measured
otherwise.** The mechanism argument: `NS(d)` is a genuine `d³` FLOP driver
that plausibly raises SINGLE-cell SM utilization well above the 44-59%
baseline at these K (more real matmul work per step, less kernel-launch-
overhead-bound) — packing N=2 here risks `2×(70-90%) = 140-180%`, real
oversubscription with a much worse contention penalty than the 1.3×
estimated at K=48/64. **This is a prediction, not a measurement** — Stage
0's calibration cell (§5) reports real SM% for K=128 solo; if it comes in
close to the 44-59% baseline after all (i.e., NS's extra FLOPs did NOT
translate into higher utilization, consistent with the overhead-bound
precedent persisting), N=2 packing should be extended to K=96/128 too, and
the ceiling in §7 re-corrected with the SAME contention-factor logic.

**Re-priced ceiling under packing:** `packed_ceiling(K, N) = ceiling(K) ×
contention_factor(N)` — at K=48/64, N=2: `18.1h×1.3=23.5h` / `33.9h×1.3=44.1h`.
At K=96/128 (N=1, no packing, pending confirmation): unchanged from §7.

---

## §9 BRANCH LOGIC — WIN vs PARTIAL of the running ortho-write experiment

This design executes ONLY under one of the two conditions below; the
`NCR_ORTHO_WRITE.md` §4 FAIL/NULL bands do not license any part of this
document.

**Under WIN (Part A: `rec@0.9` at `h*=40` ≥0.9 at K=32, mechanistically
corroborated):** run the FULL K-ladder as designed above — `h*=K+8` at every
rung, the complete §4 cell grid (subject to §5's calibration gate and §4's
trim order). The ambitious per-K target (`K+8`, proportionally deeper as K
grows) is licensed because K=32 already cleared the equivalent ask.

**Under PARTIAL (Part A: `rec@0.9`≥0.9 only at `h∈{20 OR 29}`, i.e. short of
`h*=40`, "wall cracked, shallower"):** the K-32 result did NOT clear the
`K+8`-equivalent target, so committing every rung to the SAME (proportionally
harder, since `h*=K+8` grows with K) ambitious ask is premature. Two
changes from the WIN branch: (1) **re-register the per-K primary target one
notch down** — use the `K-3` checkpoint (the ladder's own audited `m=1`
rung) as the PARTIAL-branch `h*`, not `K+8`; report `K+8` as a stretch
checkpoint instead of the primary bar. (2) **shrink the committed grid to
K=48 only, n=4 seeds, Part A alone, no Part B, no Stage-0 calibration cell
needed at this reduced scope** (a single K=48 cell's nominal FLOP-scaled
cost, 23.9h for n=4, is already inside the 150h cap without any grid
commitment past it) — the question under PARTIAL is narrower ("does even
the SHALLOW crack survive ANY scaling past K=32"), and answering it cheaply
before spending on K=64/96/128 is the correct sequencing. If K=48 clears
its shallower bar, escalate to the WIN branch's full ladder as a follow-on;
if it doesn't, the K-ladder line closes here, cheaply.

**Under NULL or FAIL:** this document does not execute. No cells launch.

---

## §A1 ATTACK ROUND 1 (2026-07-16, pre-build, independent)

Adversarial pre-build review. Every §2/§3/§4/§7/§8 number was recomputed from
the pinned formulas; every load-bearing citation was checked against
`NCR_ORTHO_WRITE.md`, `NOVEL_ARCH_WATERFALL.md:3899-3901`, `ncr_task.py`,
`ncr_ortho_write.py`, and `chapter2/model_v4.py`. Arithmetic is CLEAN
throughout (see "verified clean" at the end). The killers are structural.

### FATAL

**A1.1 — FATAL. The `h=64` encoder-hidden cap makes Gate-0 STRUCTURALLY
UNPASSABLE at K=96 and K=128 (and razor-thin at K=64). The write cannot
reach rank K.**
- Defective claim (§2): "*`h=64` fixed (encoder hidden unchanged across K…)*"
  and finding (1): "*Params stay trivial end to end (175K→200K, +14%) — the
  'never the constraint' precedent holds for PARAMS.*" The design treats
  `h=64` as a param-count question ONLY and never asks what it does to the
  **write rank**.
- Evidence: the write is produced by `BindingEncoder`
  (`chapter2/model_v4.py:52-63`): `self.row_out = nn.Linear(h, d)`, then
  `Z = self.row_out(q)` where `q` is `(B, d, h)`. Every one of the `d` rows of
  `Z` is the image of an `h`-vector under ONE shared `h→d` linear map, so
  **`rank(Z) ≤ h + 1 = 65`** for all K (the `+1` is the bias). Gate-0 requires
  `mean A_eff_rank ≥ 0.9·K` (design §6; enforced verbatim at
  `ncr_earlyln_scale.py:322`, `AEFF_RANK_FRAC_BAR*K`), and `A_eff_rank` is
  `effective_rank(A)` of the entity operator `A` derived from `Z`
  (`chapter2/analyze_zdump.py:166`, `= exp(entropy(σ))≤ rank`), so
  `A_eff_rank ≤ rank(Z) ≤ 65`. Then:
  - K=48 (d=49): cap `min(d,h)+1≈49`, bar `0.9·48=43.2` → feasible (d<h; no cap).
  - K=64 (d=65): cap `≈65`, bar `0.9·64=57.6` → RAZOR-THIN (needs eff-rank
    57.6 against a hard structural ceiling of 65; a soft-trained encoder rarely
    fills 89% of its rank ceiling).
  - K=96 (d=97): cap `≈65`, bar `0.9·96=86.4` → **IMPOSSIBLE (86.4 > 65)**.
  - K=128 (d=129): cap `≈65`, bar `0.9·128=115.2` → **IMPOSSIBLE (115.2 > 65)**.
- Why the precedent is silent: the validated regime (K=16/24/32) has `d ≤ 33 <
  h=64`, so the `h` cap NEVER binds there (`A_eff_rank≈31` at K=32 sits under
  `d=33`, not under `h`). The ladder is the FIRST time `d` crosses 64
  (K≥64) — exactly where the cap bites — and it does so with `h` held fixed.
  "Carried by extrapolation" (§2) is thus doubly unvalidated: the design flags
  the relative-headroom axis but MISSES the write-rank-expressivity axis.
- The design's own safety nets misdiagnose it: (a) the Stage-0 calibration cell
  is at K=128 (§5) — it will fail Gate-0 for THIS reason, and the design would
  misattribute it to relative-headroom or NS-convergence. (b) The §2 fallback
  (retest at `d=1.25K` = 160 at K=128) makes it strictly WORSE — rank still
  capped at 65, bar still 115.2, in an even larger ambient space. (c) The §5
  `orthogonality_error` abort WOULD fire (NS-polar of a rank-65 `Z` in d=129 is
  a partial isometry; scale-normalized `‖QᵀQ−I‖_F ≈ 11`, not machine
  precision), but the pre-registered remediation ("raise `n_iter`") CANNOT fix
  it: a zero singular value is a fixed point of the NS map `x←1.5x−0.5x³`, so no
  number of iterations orthogonalizes a rank-deficient write.
- Minimal fix: `h` (encoder hidden) MUST scale with K so `h ≳ d = K+1` (e.g.
  `h = max(64, K+ margin)`), which is what actually lets `row_out` write rank-K.
  This is NOT a patch: the `40h²` param term then grows as K² (invalidating the
  §2 "params trivial" finding and the entire §2/§4/§7/§8 cost model), and the
  FLOP/NS tables must be recomputed. BUILD-BLOCKED until `h(K)` is redesigned
  and every downstream table redone. (Alternatively cap the ladder at K≤48 —
  the only rung whose `d<h` keeps the write full-rank in the validated regime —
  but that abandons the scaling-law premise.)

### MAJOR

**A1.2 — MAJOR. The free-write "<0.5 at same depth" WIN(K) clause is never
measured at any new K, and the cited precedent is mischaracterized.**
- Defective text (§4): "*the free-write baseline is NOT re-run at each new K; it
  is already established DEAD at K=24/32 …, §4's own pinned baseline-reuse
  precedent*", carried into §6 WIN(K): "*free-write baseline (extrapolated-dead
  …, not freshly re-measured) stays <0.5 at the same depth*".
- Evidence: `NCR_ORTHO_WRITE.md` §4 pins the K=32 free clause as **measured**
  ("*reads < 0.5 at h=40 (measured; … re-analyzed on the identical realistic
  ladder from the archived z-dumps, CPU, free)*", §4 + the "Pinned baselines"
  block). That precedent REUSES AN EXISTING MEASUREMENT (a K=32 z-dump that
  physically exists). No free-write z-dump exists at K∈{48,64,96,128} — there is
  nothing to re-analyze — so the design is not "reusing a measurement," it is
  EXTRAPOLATING across K and calling it reuse. The WIN(K) band therefore
  contains a gate clause that can never be checked, so WIN(K) collapses to a
  single-arm ortho claim — which directly undercuts the PI's capability-
  SEPARATION headline (a separation claim needs both arms AT that K).
- Minimal fix: measure a free-write anchor at ≥1 new K (minimum: K=48, the floor
  cell; a fresh free run is cheap relative to the wave), OR restate WIN(K) as an
  explicit single-arm claim with the free comparison labeled
  "extrapolated-from-K≤32, NOT measured at this K." Do not pre-register an
  unverifiable clause inside a WIN band.

**A1.3 — MAJOR. The mechanistic corroboration thresholds are asserted
K-independent, but their far-depth meaning compounds over `h*=K+8` and goes
vacuous at large K — contra the CLAUDE.md instrument-relative rule.**
- Defective text (§6 WIN(K)): "*mechanistic corroboration unchanged
  (departure-from-normality ≤0.02, cond#≤~2, min|λ|/c\*≥0.9) — these thresholds
  … are not themselves K-dependent*", with §6 also claiming the bands "*mirror
  `NCR_ORTHO_WRITE.md` §4 Part A exactly*".
- Evidence: the far-depth recovery these numbers corroborate decays as
  `(min|λ|/c*)^h` (`NCR_ORTHO_WRITE.md` §1). The SAME threshold `min|λ|/c*=0.9`
  leaves the weak mode at `0.9^40 = 0.015` at K=32 (h*=40) but `0.9^136 = 4·10⁻⁷`
  at K=128 (h*=136) — total annihilation. So a threshold that is meaningful
  corroboration at K=32 is satisfied by operators far too imperfect to win at
  K=128; the corroboration has NO discriminating power at large K. CLAUDE.md's
  hard rule is explicit: "*the … n_iter-sufficiency frontier MOVES with K/d …
  Never carry an admission profile derived at one K/d to another without
  re-validating.*" Asserting flat K-independence violates it.
- Minimal fix: tie corroboration tightness to `h*` (require
  `(min|λ|/c*)^(K+8) ≥` a fixed fidelity, likewise for cond#/departure), OR
  explicitly demote the three numbers to non-binding sanity checks and state
  the behavioral `rec@0.9` is the SOLE gate. Either way, drop the false
  "not K-dependent" and "mirror exactly" wording.

**A1.4 — MAJOR. The Stage-0 orthogonality abort trigger is a pre-registered
decision gate with no pinned threshold (adjudication-by-vibes).**
- Defective text (§5): "*Target: `‖QᵀQ-I‖_F` at machine-precision-ish … a
  **materially non-zero** residual at K=128 is an explicit ABORT-and-redesign
  trigger*".
- Evidence: "machine-precision-ish" and "materially non-zero" are not numbers.
  `orthogonality_error` is a scale-NORMALIZED Frobenius residual
  (`ncr_ortho_write.py:144-151`) that is never exactly zero and (per A1.1) reads
  ≈11 under rank-deficiency and O(1e-2) under mild NS under-convergence — utterly
  different regimes that "materially non-zero" cannot distinguish. Pre-
  registration discipline (the doc's own record-first framing; the frozen §4
  bands) requires a frozen number for any abort gate.
- Minimal fix: pin an exact numeric tolerance (e.g. scale-normalized
  `orthogonality_error ≤ 1e-2` per the code's own self-test t4 bound, or the
  realized K=32 ortho value once available) AND a forced-fail negative test, per
  the CLAUDE.md "run the negative unit test that proves the check has teeth"
  rule.

**A1.5 — MAJOR. Residue-8 monoculture: the primary bar has FIXED effective
relational depth at every K, so §6's "K-general depth composition" readout
overstates what the ladder measures.**
- Defective framing (§1 / §6a): "*one full relational cycle plus a novel
  residue*" and "*the orthogonal write's realistic-depth composition ceiling is
  K-general*".
- Evidence: every primary `h*=K+8` has residue 8 (design §3), so the effective
  target is `π^8` at EVERY K; and for a faithfully-learned orthogonal operator
  `Z^K≈I`, so `Z^(K+8)≈Z^8` — the "+K" contributes nothing but a physical-
  application stress. The full realistic ladder's forward effective depths are
  `{5,8,12,20}` (fixed, K-independent) plus `π^{-3}` (near=K-3 AND stretch=2K-3
  both `≡ -3 mod K`). No ladder point probes a forward effective depth that
  GROWS with K. So the "scaling law" measures orthogonality-survival over `K+8`
  physical applications with EFFECTIVE relational depth pinned at ≤20 — not
  deeper compositional generalization. (Aggravator: residue 8 has identical
  special structure `gcd(8,K)=8 ⇒ 8 sub-cycles` at every ladder K, so a
  residue-8-specific effect would masquerade identically across all four rungs.)
- Minimal fix: add a novel primary checkpoint whose effective (mod-K) depth
  GROWS with K and whose gcd-with-K varies (e.g. a residue coprime to K giving a
  single long cycle), OR restate the §1/§6 readout precisely as
  "physical-application-fidelity scaling," dropping "compositional depth
  generalization" at the primary. (Note: this monoculture is inherited from the
  frozen K=32 pre-reg, so the fix is claim-scoping, not a build blocker on its
  own — but the ladder AMPLIFIES a single-K confound into a four-point "law.")

**A1.6 — MAJOR. The pinned trim order provably cannot reach the 150 GPU-h cap
in the compute-bound worst case it is built for.**
- Defective claim (§4): "*Pinned trim order if the calibration-corrected
  projection still exceeds the cap*" (steps 1–5), against "*Planning cap … 150
  GPU-h*".
- Evidence (recomputed on the design's own §4 nominal table): applying ALL four
  trims — drop B@K128 (−391.6), K96 A n4→n1 (−90.4), drop B@K48 (−36.2), K128 A
  n4→n1 (−194.0) — leaves `K48 A 23.9 + K64 A 44.7 + K96 A n1 30.13 + K128 A n1
  64.65 = 163.4 h`, still **> 150 h**. Step 5 is the floor (K48+K64 only), so
  K96/K128 A at n1 are neither trimmable-to-zero nor floor — the trim order
  bottoms out at 163.4 h and cannot satisfy its own cap. (The floor deliverable,
  68.6 h, IS protected — that part is clean.)
- Minimal fix: add explicit "drop K96 A entirely / drop K128 A entirely" steps
  before the floor, OR raise the cap to ≥165 h, OR state plainly that the cap
  binds only under the overhead-bound (calibration-corrected) regime and the
  compute-bound worst case is PI-gated, not trim-order-satisfiable. (Real-world
  stakes are low — uptime-metered grant, cap is "adjustable" — but the stated
  mechanism is internally inconsistent.)

### MINOR

**A1.7 — MINOR. §6 claims PARTIAL(K) "mirror[s] §4 Part A exactly," but the
checkpoint set differs.** Frozen K=32 PARTIAL (`NCR_ORTHO_WRITE.md` §4) is
"`h∈{20 OR 29=K-3}`"; design §6 PARTIAL(K) is "`K-3 / 2K-3`" — it drops the
shallow-20 checkpoint and substitutes the deeper 2K-3. A silent re-derivation of
a frozen band. Fix: say "generalized (20→2K-3)," not "exactly."

**A1.8 — MINOR. §7 runaway-abort formula uses the SOLO ceiling for packed
cells.** §7 states the trigger as "1.5× that cell's own `ceiling(K)`" while the
same section's packed-correction paragraph and §8 require the CONTENDED
projection — a literal reader kills healthy packed cells. Fix: write
"1.5× `packed_ceiling(K,N)`" for packed cells.

**A1.9 — MINOR. The `d=1.25K` fallback is under-motivated by, and mildly in
tension with, the only d-variation evidence in the cited source.** It is
well-posed (`K≤d` holds at d=60/80/120/160, task defined — attack-surface-2's
narrow question passes), BUT `NCR_ORTHO_WRITE.md` §5 shows the MORE-headroom
(2K) convention was catastrophically ill-conditioned (K24@d48 cond#≈2951,
"polar barely helps") vs the K+1 tight-spare being "far healthier." Moving
toward 2K to rescue trainability is thus a DIAGNOSTIC ("does headroom matter"),
not a presumed rescue, and its own cost is un-repriced (`NS(160)≈656M` at K=128,
~1.9× `NS(129)`). Fix: reframe as a diagnostic and cite the §5 cautionary point.

**A1.10 — MINOR. Single-seed Stage-0 drives a branch/redesign decision despite
documented high trainability seed-variance.** §5's FAIL clause (Gate-0 dead at
K=128 → don't launch upper ladder / go to `d=1.25K`) fires on ONE seed, but the
project's own record (`CLAUDE.md` / STATE §1.40: "one fresh seed cleared the
bar") warns trainability is seed-variance-heavy. Fix: require a 2nd calibration
seed before the FAIL→redesign branch fires (cheap; one extra cell).

**A1.11 — MINOR. §9 PARTIAL branch and §6 bands are not fully reconciled.** §9
moves the primary to `K-3` under PARTIAL, but §6's WIN(K)/PARTIAL(K)/NULL(K)
bands are written only for the `K+8` primary; the branch does not restate the
full band set at the moved bar. Fix: give the PARTIAL-branch band mapping
explicitly.

### Verified CLEAN (attack surfaces exercised, no defect found)
- **All §2 tables** (params `166784+257d`; FLOP `311296K+16384d+768K²+256Kd+256d²`
  at h=64; `NS=160d³+48d²`; F+NS; ratio-vs-K32; `1/d`; K/d; memory `320·d²`/ex ×256):
  every cell recomputed EXACT, including `NS(33)=5,802,192`, `P(33,64)=175,265`.
- **§3 mod-K residue table**: all residues recomputed EXACT (near=K-3, h*≡8,
  stretch=2K-3, synthetic=8K-3); all `∉{0,1,2,3}` verified; `h*/3` multiples
  (18.7×/24×/34.7×/45.3×) EXACT. Matches `ncr_task._gen_grid` (ladder_residue=K-3,
  ladder=m·K-3, h_star=8K-3) and GRIDS is pre-defined for all four K.
- **§4 pricing**: per-cell (5.98/9.06/11.18/16.93/30.13/45.63/64.65/97.90 h),
  subtotals (A 447.7 / B 427.8 / total 875.5), K128=650.2h=74% — all EXACT.
- **§7 ceiling** (`2·ratio·4.24`: 18.1/33.9/91.3/195.8) and **§8 packed**
  (×1.3: 23.5/44.1; `2/1.3=1.54×`) — EXACT.
- **Attack-surface-8 (disc-rate ceiling distorting trim order)**: does NOT hold.
  The trim order (§4) prices A at 2.8h and B at 4.24h correctly; the disc-rate
  +2× conservatism lives ONLY in the §7 sticker-ceiling, which does not feed the
  trim order. (Separately flawed via A1.6, but not by double-counting.)
- **Code-claim checks**: `d=K+1` (`ncr_ortho_write.py:286`, overrides
  GRID_SHAPES d=2K), `h=64` (`:287`, `GRID_SHAPES[K]["h"]`), `NS_ITER_DEFAULT=40`
  / `NS_POWER_DEFAULT=12` (`:82-83`), `orthogonality_error` instrumented
  (`:144`) — all match the design's "reused inputs."

### VERDICT: **BUILD-BLOCKED** (A1.1 is fatal: K=96/K=128 Gate-0 is
unpassable because the fixed `h=64` write caps `rank(Z)≤65 < 0.9·K`; the
calibration cell, the `d=1.25K` fallback, and the `n_iter` remediation all fail
to address it). Re-registration required before any build: scale `h(K)` and
redo every §2/§4/§7/§8 table, then clear A1.2–A1.6.

---

## §A1-ADJUDICATION (coordinator, 2026-07-16 — recorded before dispatching Rev 1)

**A1.1 FATAL: CONFIRMED against the raw code by the coordinator directly**
(per the conflicting-claims/tiebreak rule — verified, not taken on the
attacker's word): `chapter2/model_v4.py:52` `row_out = nn.Linear(h, d)` with
`Z = row_out(q)`, `q:(B,d,h)` ⇒ `Z = QW + 1bᵀ` ⇒ `rank(Z) ≤ h+1 = 65`;
`ncr_earlyln_scale.py:75` `GRID_SHAPES` pins `h=64` for ALL K∈{14..128};
`ncr_earlyln_scale.py:322` gate requires `aer_mean ≥ 0.9·K` = 86.4 (K=96) /
115.2 (K=128) — both `> 65`, structurally unpassable. Context: `h=64 = 8K`
at the original K=8; the h/K ratio silently shrank as K grew and the
validated regime (K≤32, bar 28.8) never touched the ceiling.

**Disposition:** BUILD-BLOCKED verdict ACCEPTED. Rev 1 dispatched with:
(a) h must scale with K — coordinator steer: `h=2K` preserves the ratio at
the LAST VALIDATED rung (K=32, h=64=2K), making the rank ceiling `2K+1 ≥
0.9K` trivially and minimizing extrapolation distance (h=8K matches only the
K=8 rung and 4×-overshoots the validated ratio); Rev 1 may argue an
alternative but must justify it against this default and re-derive EVERY
§2/§4/§7/§8 cost/memory table for the chosen h(K);
(b) A1.2–A1.6 each addressed with an exact, testable change (A1.2: measured
free-write confirmatory cells n=1 per new K, or an explicit one-arm
reframing of WIN(K) — no extrapolated baseline inside a WIN clause;
A1.3: mechanistic thresholds re-derived per-K from band arithmetic, e.g.
`min|λ| ≥ floor^(1/h*(K))`, honoring the instrument-relative rule;
A1.4: exact orthogonality_error tolerance + a forced-fail negative test;
A1.5: honest reframe of effective-depth claims + at least one added
non-residue-8 novel probe depth per K; A1.6: trim order re-derived to
actually reach the 150h cap);
(c) minors A1.7–A1.11 folded in. Rev 1 output → fresh independent ATTACK
ROUND 2 (multi-round rule) before any build authorization. The design
remains CONDITIONAL on the ortho-write verdict regardless.
