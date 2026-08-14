# NCR write-conditioning — ATTACK ROUND 4

**Target:** `matrix-thinking/NCR_WRITE_CONDITIONING_DESIGN.md` §`DRAFT-R4
(Rev-4, 2026-08-13)` (lines 2946–3957) at commit `8c665f7`, ONLY.
**Date:** 2026-08-13. **Agent:** round-4 adversarial audit (independent of the
reviser).
**Frozen, not re-litigated:** `NCR_WRITECOND_ATTACK_R3.md`'s 13 verified-clean
items (V1–V13); the premise-battery facts (`§PREMISE BATTERY HARVEST`); the
novelty verdicts in `research/writecond-novelty-2026-08-13.md` RE-ENTRY 2.
**Prior reports:** R1 (BLOCKED 5F/11M/7m), R2 (BLOCKED 5F/12M/8m), R3 (BLOCKED
3F/8M/8m).

---

## VERDICT: **REV-REQUIRED** (4 FATAL / 11 MAJOR / 10 minor)

**Operational meaning — stated first so it cannot be misread.** REV-REQUIRED
means the same thing as BLOCKED for GPU spend: **no Stage-1 cell may launch
before Rev-5**, and **Stage 0′ may not run as carded**. It differs from R1–R3's
BLOCKED in *kind*, and the difference is the point of this round:

- **The mechanism this round was scoped to verify is CLEAN.** D1's
  re-specified loss, D1.3's zero-set proof, D2's `ortho` removal, and D3's
  detach all survive independent re-derivation *and* execution at compB's own
  measured key geometry. I re-derived the proof from scratch and it is
  correct; the "L_key flat directions = {u wᵀ}", "Z_ideal w = 0", and
  "L_transverse = 0 ⟺ u = 0" steps each reproduce numerically (§V1–V4). F1,
  F2, F3 of R3 are genuinely closed.
- **Every FATAL below is in the instrumentation / adjudication / launch-card
  layer**, and each has a named repair that does not re-open D1–D3.

The four FATALs are: an adjudication band that is *not* scale-quotiented
(D4's central claim is false, executed); a `λ_t` commitment resting on an
argument that holds in Z-space and fails in parameter space, selected by an
instrument that provably cannot discriminate; a Stage 0′ script that cannot
run (three wrong checkpoint paths, one `AttributeError`, one non-existent
function, one un-scoreable item); and a Band-0 gate that voids CONTROL B by
construction with no self-correcting path.

**FATAL** here = *would produce an uninterpretable verdict, void a
pre-registered cell, or spend GPU-h that cannot answer the question.*

---

## What is VERIFIED CLEAN (recorded so Rev-5/R5 does not re-derive it)

| # | item | how verified |
|---|---|---|
| **V1** | **D1.3's zero-set proof is CORRECT.** At compB's measured geometry (mean pairwise cos 0.2200, matching `target_pairwise_cos = 0.2193` @h=61 in the raw artifact), `L_key` is flat to `1.9e-12 → 2.4e-11` across `‖u‖ = 0 → 100` while `L_transverse` tracks `‖u‖²/v̄²` exactly; the joint zero occurs only at `u=0`. | executed, `r4_1_zeroset.py` |
| **V2** | `Z_ideal w = 0`: `max_b ‖Z_ideal w‖ = 1.7e-05` at `‖Z_ideal‖_F = 23.4` (fp32), i.e. zero to float noise. R3 measured `2.6e-05`. | executed |
| **V3** | **D1.2's `w` construction is right.** For `keys_v (B,24,25)`, `torch.linalg.svd(..., full_matrices=True)` returns `S (B,24)` and `Vh (B,25,25)`; `Vh[:,-1,:]` is unit-norm with `max|w·kᵢ| = 2.2e-07` (R3: `2.0e-07`). Row rank is 24 in 64/64 episodes. | executed |
| **V4** | **D3's detach DOES cover `w`'s SVD route.** With `keys_v` detached before the SVD, `Vh.requires_grad = False` — no SVD backward, no path to `entity_adapter`/`embed`. With `keys_v` undetached the route carries real gradient. The "second door" D3 names is real and the single top-of-function detach closes it. | executed, `r4_4_svd.py` §A |
| **V5** | **D1.3's literal orthogonality claim holds in Z-space.** `⟨∂L_key/∂Z, ∂L_trans/∂Z⟩_F = 0.000e+00` exactly (`∂L_key/∂Z` has row space ⊆ span(keys); `∂L_trans/∂Z ∈ ℝ^d ⊗ span(w)`; `span(keys) ⊥ w`). | executed, `r4_2_gradortho.py` Test 1 |
| **V6** | **D5.2's partition has no hole and no double-fire.** 18,018 constructed outcomes (1001 `x` values × 6 `GAP` values × 3 recipes, including the `GAP = 0.15/0.1500001` boundary): exactly one of NULL/WIN/PARTIAL fires every time. M5(a)'s hole is closed structurally. | executed, `r4_8_bands.py` §B |
| **V7** | **D5.1's compB anchors and D5.3's Band-4 per-hop references are exact against the raw artifacts.** compB `P0=0.066406`, `P1b=0.976562`; per-hop P0 `h=5/12/20/40/61 = 0.046875 / 0.035156 / 0.031250 / 0.074219 / 0.066406` — all match the design's digits. | `writecond_premise_SUPP.json`, `_P0P1b.json` |
| **V8** | **The 127.8 correction is right.** `NCR_WRITECOND_ATTACK_R2.md:111-118` reads `{148.7, 170.8, 195.2, 167.7, 127.8}` across the five *measured* geometries; 127.8 (compA) is the floor. | raw table |
| **V9** | **D6's arithmetic reproduces to the digit.** `3×4.861 + 4.861 + 0.486 + 4.861 + 0.05 = 24.841`; `+0.10 = 24.94`; `×1.4 = 34.92`; low/high `23.67 / 26.22`. M6b's `2.481 GPU-h` = `0.8116 + 0.8293 + 0.8403` exactly (the three `mob_g3b31_*` result JSONs). | executed + raw JSONs |
| **V10** | **The language obligation is discharged inside the design.** No surviving "provably fails" in any claim context in lines 2946–3957; the only occurrence is the explicit prohibition. (One record-hygiene exception outside the section — m9.) | grep |
| **V11** | **Band 0's witness fields are real.** `rec["teacher_force_check"]["active"]`, `["ncr_zero_grad_checks_passed"]` (runner:1305/1416/1439) and `config.teacher_force_operator` all exist in the recorded result JSONs. `--ceiling-gpuh`, `--freeze-entity-adapter`, `--ortho-reg-weight` all exist as flags. | runner + `mob_g3b31_compB_s0.json` |
| **V12** | **The ≈0.1 GPU-h Stage 0′ cost claim is conservative.** The premise battery's comparable read-only cells measured `elapsed_s` 5.75–15.02 at n=256. Item 6's added Adam sweep (4 λ × 3000 steps on `(256,25,25)`) is seconds on an H100. | raw JSONs + timing of the local equivalent |
| **V13** | **m1's dead-code elimination is correct.** Neither sub-term needs `Z_ideal` at runtime: `L_key` is defined from `Z_sgd` on raw `(keys_v, values_v)`; `L_transverse` needs only `w` and `v̄²`. Dropping the line is right. | inspection of D1.1/D1.2 |

---

# FATAL FINDINGS

### F1 (FATAL — adjudication) — D4 does **not** quotient the global scale; Band 2's two hard gates are both scale-sensitive, and a *perfectly reading* operator scores INCONCLUSIVE-BY-MECHANISM

D4 claims the fix "sidesteps M1's exact failure mode **by construction**,
because the calibrated targets are properties of the *residual*, not sensitive
to which global scale the encoder happens to have settled on … the calibration
table already reports RMS **relative** per-key error, which factors out exactly
this degree of freedom."

**Both halves of that sentence are false.** `L_key` divides by `‖vᵢ‖²` — it
factors out the scale of **v**, not the scale of **Z**. A `10×`-larger `Z`
rescales `L_key` by `~100×` and the target `3e-4` does *not* rescale with it;
it is an absolute constant. Meanwhile `binexp_read` renormalises at every
squaring, so the read is *exactly* invariant to a positive global scale on `Z`
(M1, frozen).

**Executed** (`r4_3_scale.py`, n=256, compB geometry, real `nm.binexp_read`,
real `retrieval24` construction):

CASE 1 — a **perfect** operator `Z = c·Z_ideal`:

| `c` | `L_key` | `‖Zw‖` | **Band 2** | `retr24@61` | Band 3 |
|---|---|---|---|---|---|
| 0.01 | 9.801e-01 | 0.0000 | **FAIL** | 1.0000 | WIN |
| 0.10 | 8.100e-01 | 0.0000 | **FAIL** | 1.0000 | WIN |
| 1.00 | 2.148e-12 | 0.0000 | pass | 1.0000 | WIN |
| 1.50 | 2.500e-01 | 0.0000 | **FAIL** | 1.0000 | WIN |
| 10.0 | 8.100e+01 | 0.0000 | **FAIL** | 1.0000 | WIN |
| 100 | 9.801e+03 | 0.0002 | **FAIL** | 1.0000 | WIN |

A `1.5×` global scale — nothing else changed, the read literally cannot see it
— converts a perfect `retrieval24@61 = 1.0000` into
`INCONCLUSIVE-BY-MECHANISM`. That is M1's failure mode, unfixed, now promoted
into a **hard pre-registered gate** that sits *above* the primary signal in the
band order.

**The transverse half is worse, because it fails in the permissive
direction.** `‖Z_sgd w‖ ≤ 3` is an absolute threshold on a quantity that
scales linearly with `‖Z‖_F`. R3's own F1 stated the calibration **with its
conditioning intact**: *"the calibrated requirement from the table above is
`‖Z w‖ ≲ 3` at `‖Z‖_F ≈ 25`, i.e. transverse gain ≤ ~12% of the operator
norm."* DRAFT-R4's D1.5 transcribes it as a bare `‖Z_sgd w‖ ≲ 3` and drops the
`at ‖Z‖_F ≈ 25` qualifier. The consequence is not hypothetical — my
parametrised probe (F2 below) produces trained operators with `‖Z‖_F ≈ 1.0` and
`‖Zw‖ = 0.20`: **the transverse gate passes comfortably on an operator that
reads at chance**, because the operator itself has collapsed toward zero.

CASE 3 — the scale-invariant forms behave correctly:

| operator | `‖Zw‖/‖Z‖_F` | `L_key` at the optimal rescale `c*` | `retr24@61` |
|---|---|---|---|
| `c=1` perfect | 0.0000 | 2.158e-12 | 1.0000 |
| `c=10` perfect | 0.0000 | 2.212e-12 | 1.0000 |
| `c=0.01` perfect | 0.0000 | 2.221e-12 | 1.0000 |
| `c=1` transverse-100 | 0.9705 | 2.378e-11 | 0.0625 |
| `c=0.01` transverse-100 | 0.9705 | 2.390e-11 | 0.0508 |

The closed-form `c*` rescale D4 explicitly declined to add ("No additional
closed-form `c*`-rescale machinery is added") is four lines and makes `L_key`
exactly invariant; the ratio `‖Zw‖/‖Z‖_F` is one line and is precisely R3's own
12% statement.

**Repair (binding).** Band 2 becomes:
`L_key(c*·Z_sgd) ≤ 3e-4` **AND** `‖Z_sgd w‖ / ‖Z_sgd‖_F ≤ 0.12`, with
`c* = Σ⟨Z kᵢ, vᵢ⟩ / Σ‖Z kᵢ‖²` computed per episode, and the reduction
statistic (median / p90 across the eval batch) **named explicitly** — the
current text names none. Also state the band applies to the eval batch at
`h*`, since neither quantity is a function of `h`.

---

### F2 (FATAL — wave-wasting) — the `λ_t` commitment rests on an orthogonality argument that is true in Z-space and **false in parameter space**, and Stage 0′ item 6 provably cannot discriminate `λ_t`

D1.3's "structural bonus" states the two terms' gradients are orthogonal
subspaces and concludes: *"they do not compete for the same minimum … `λ_t`
only has to be large enough to make the transverse direction converge **within
the training budget** — a speed question, not a competing-minima question."*
Self-attack item 2 then declines to sweep `λ_t` on that basis, committing all
three PRIMARY recipes to one value chosen by Stage 0′ item 6.

**(a) The orthogonality is real in Z-space and gone in parameter space.**
Training moves `θ`, not `Z`; the pullback through `∂Z/∂θ` does not preserve the
split. Measured on the **real** encoder (`ncr_models.NCRModel` →
`chapter2/model_v4.BindingEncoder(d=25,h=64)`, **173,209 params**, exactly
`NCR_PARAM_EXACT`):

| state | `cos(∇_θ L_key, ∇_θ L_trans)` |
|---|---|
| Z-space (D1.3's literal claim) | **0.000000** |
| parameter space, at init | **0.4458** |
| parameter space, after 200 steps | 0.1695 |
| parameter space, after 600 steps | 0.2845 |
| random-pair null in the same 173,209-dim space | 0.0017 |

100–260× the null. The terms are strongly *coupled* in the space where the
optimiser works. (Here the coupling is positive — cooperative — but the sign is
an empirical fact of this probe, not a structural guarantee, and it says
nothing about the CE/aux terms that also pull on the same parameters.)

**(b) At the acceptance boundary the two terms differ by ~10⁴ in magnitude.**
At `‖Zw‖ = 3` (the WIN edge) `L_transverse = 7.139`, against `L_key`'s WIN edge
of `3e-4` — a ratio of **23,796×** at `λ_t = 1.0`. The objective is the
transverse term until `‖Zw‖ ≲ 0.02`, i.e. ~150× past the requirement.

**(c) The instrument registered to choose `λ_t` cannot see any of this.**
Item 6 optimises a **free** `Z` (625 unconstrained dof per episode) with Adam.
Executed exactly as carded (`r4_5_item6.py`, compB geometry):

| `λ_t` | `L_key_final` | `‖Zw‖_final` | `retr24@61` | item-6 gate (`≤3`) |
|---|---|---|---|---|
| 0.0 | 4.257e-03 | 9.5400 | 0.3906 | fail |
| 0.1 | 4.989e-03 | 0.0012 | 0.7500 | **pass** |
| 1.0 | 1.784e-02 | 0.0060 | 0.5625 | **pass** |
| 3.0 | 9.478e-02 | 0.0098 | 0.3906 | **pass** |
| 10.0 | 4.526e-01 | 0.0101 | 0.2656 | **pass** |

Every `λ_t > 0` passes the registered gate by three orders of magnitude. **The
gate cannot fail, so it cannot choose.** This is the identical defect R3's
§STAGE-0 RULING used to BLOCK Stage 0.1 ("its gate is provably
non-diagnostic"), reintroduced in the replacement.

**(d) In the only parametrised probe available, `λ_t = 1.0` costs the WIN.**
Real encoder, easiest possible geometry (near-orthonormal keys, where
`Z_ideal = Σ vᵢ kᵢᵀ` is a plain sum of outer products), 8,000 steps, lr 1e-3,
single seed (`r4_9_easy.py`; held-out episodes):

| `λ_t` | `L_key` | `‖Z‖_F` | `‖Zw‖` | `‖Zw‖/‖Z‖_F` | `retr@1` | **`retr@61`** |
|---|---|---|---|---|---|---|
| 0.0 | 7.311e-03 | 4.949 | 0.9373 | 0.1894 | 1.0000 | **0.8906** |
| 1.0 | 1.012e-01 | 4.604 | 0.0946 | 0.0205 | 1.0000 | **0.1406** |

At equal budget the transverse term turned a WIN-shaped `0.8906` into
`0.1406`, and the `λ_t=0` arm **already satisfied the absolute `‖Zw‖ ≤ 3`
gate** without any transverse term. This does not prove `λ_t = 1.0` is wrong
for Stage 1 (synthetic keys, no CE/aux, one seed, 8k steps). It proves three
things that do bear on the launch: the "speed question" *is* the question at a
fixed 20,000-step budget; the free-`Z` instrument cannot see it; and the
"unconstrained transverse expectation ≈ 5" premise motivating D1's necessity is
itself scale-conditioned on `‖Z‖_F ≈ 25`, which the actual parametrisation does
not reach.

**Repair (binding).** (i) Restate D1.3's bonus with its scope ("orthogonal in
Z-space for fixed `(keys_v, w)`; **not** in parameter space — measured
`cos = 0.17–0.45` on the real encoder"). (ii) Replace item 6's free-`Z` sweep
with a **parametrised** achievability/`λ_t` probe (see the Stage-0′ ruling,
amendment A6). (iii) Either sweep `λ_t ∈ {0, 0.1, 1.0}` as a Stage-1 axis on
one recipe, or pre-register that the value is fixed by the *parametrised*
Stage-0′ curve and that a `λ_t = 0` arm is retained as the necessity control
for D1's own new term — currently **nothing in the wave tests whether the
transverse term is needed at all** in the parametrisation.

---

### F3 (FATAL — Stage 0′ launch card) — the carded script cannot run: three wrong checkpoint paths, one `AttributeError`, one non-existent function, and an item that has no way to score what it claims

The premise battery hit real signature/path surprises; the card re-introduces
five of them. Each is checked against the pinned runner
(`experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/ncr_lm_wave1_runner.py`,
md5 `9a93198b642242f512ff8489e32b0a53` — verified) and against the battery's own
archived scripts.

1. **All three checkpoint paths are wrong (job dies on the first line of the
   loop).** Card: `~/ncr_g3b31/results/mob_g3b31_{tag}_s0.ckpt.pt`. Actual, from
   the battery's own artifacts (`writecond_premise_REPL_compA.json`'s recorded
   `ckpt` field, and `pbe_supplement.py`):
   `~/ncr_g3b31_contrastive/results/mob_g3b31_{tag}_s0_ckpts/mob_g3b31_{tag}_s0.ckpt.pt`
   — wrong **directory** *and* missing the `_ckpts/` level. `R.load_checkpoint`
   returns `None` for a missing path (runner:1122-1125), so the card's
   `assert ckpt is not None` fires immediately on `primary`.
2. **`ncr_head.row_out` does not exist → `AttributeError` in item 3 (a STOP
   gate).** `NCREarlyLNModel` inherits `nm.NCRModel`, whose `__init__` sets
   `self.encoder = BindingEncoder(d, h, …)` (`ncr_models.py:165`). `row_out`
   lives on the *encoder*. Executed: `hasattr(NCRModel, 'row_out') = False`,
   `hasattr(NCRModel.encoder, 'row_out') = True`. Correct path:
   `ncr_head.encoder.row_out`. (Inherited verbatim from R3's own item-3 text —
   flagged now, not blamed on the reviser.)
3. **`R.score_operator_at_hops` does not exist.** A repo-wide grep finds the
   name in exactly one place: line 3798 of the design itself. The card's own
   comment claims the "exact call signature confirmed by the build agent
   against `ncr_lm_wave1_runner.py:480-537`" — lines 480–537 are
   `discriminability_metrics(integ, embed, o, entity_ids, tgt_slot)`, a
   different function with a different signature. The citation does not support
   the call.
4. **Item 6 has no way to score retrieval even if the function existed.**
   `retrieval24_acc` is computed from `o` against
   `entity_adapter(embed(entity_ids))` with `tgt_slot` (runner:513–521).
   `extract_real_kv` returns only `(Z_sgd, keys_v, values_v)` and discards the
   probe batch, so `entity_ids`, `tgt_slot`, and `query_key_col` are gone by the
   time item 6 runs.
5. **The multi-hop readout is incoherent with the single-hop extraction.**
   `extract_real_kv` builds ONE document set at `h=61`. `eval_arm_at_hops` draws
   a **fresh document per hop** (`manual_seed(base_seed + EVAL_SEED_OFFSET + h)`,
   runner:940), and `tgt_slot`/`answer_token`/`query` all depend on the hop.
   Scoring the `h=61`-fitted `Z` at `h ∈ {1,13,37}` therefore requires either
   different documents (invalidating the fit) or a fix-the-document-vary-the-hop
   instrument that does not exist in the pinned runner and is not written here.

**Ruling and repairs: see §STAGE-0′ RULING below.**

---

### F4 (FATAL — pre-registration contradicts itself) — Band 0 VOIDs CONTROL B by construction, with no self-correcting path

D5.3 Band 0: *"Every Stage-1 artifact (**PRIMARY, CONTROL A/B/C**) must show
`config.teacher_force_operator == false`, `teacher_force_check.active ==
false`, `teacher_force_check.ncr_zero_grad_checks_passed == 0`. FAIL ⇒ **VOID**
the cell … re-run, never scored."*

CONTROL B is, by its own definition (DRAFT-R3 §4, carried forward unmodified by
D7): *"Reuses `--teacher-force-operator` VERBATIM, unmodified: continue
training the compB checkpoint … for 2,000 further steps with
`teacher_force=True` … so `ncr_head`/the encoder receive EXACTLY zero gradient
(`teacher_force_check.ncr_zero_grad_checks_passed`, asserted every step)."*

So CONTROL B necessarily produces `config.teacher_force_operator = true`,
`active = true`, and `ncr_zero_grad_checks_passed > 0` — **it fails all three
Band-0 clauses by construction**, is VOIDed, and the prescribed remedy
("re-run") reproduces the identical artifact. 0.486 GPU-h is spent to produce a
cell the harvest is pre-registered to discard, and the one control that answers
"does read-side adaptation alone explain the gain" leaves no record.

**Repair (binding, one clause).** Band 0 applies to **PRIMARY and CONTROL
A/C**. For CONTROL B the gate INVERTS: `teacher_force_check.active == true` and
`ncr_zero_grad_checks_passed == steps_run` (the witness that the encoder was
untouched), plus the D7-mandated **separate** `teacher_force=False` eval
invocation whose artifact carries the clean readout. Both halves must be named
in the band, because D7's clean-eval script is exactly what makes CONTROL B
scoreable.

---

# MAJOR FINDINGS

### M1 — `L_transverse`'s normaliser is **not** "the same class" as `L_key`'s; `λ_t`'s effective weight drifts as `α⁻²` with the entity-adapter's free output scale

D1.2: *"`L_transverse` is normalized by the **same class** of quantity `L_key`
already divides by … this keeps the two terms' gradient magnitudes comparable
across episodes."*

`L_key = ‖Zkᵢ − vᵢ‖²/‖vᵢ‖²` is invariant under a joint rescale
`(k, v) → (αk, αv)`. `L_transverse = ‖Zw‖²/v̄²` is not: `Z_ideal` and `w` are
both invariant under that rescale, so the numerator is fixed while `v̄²` scales
as `α²`. Executed (`r4_10_norm.py`, fixed *relative* errors on both sides):

| `α` | `L_key` (rel-err 1e-2) | `‖Zw‖` (u fixed at 12% of `‖Z_ideal‖_F`) | `L_transverse` | effective `λ_t` ratio |
|---|---|---|---|---|
| 0.25 | 2.332e-03 | 2.8133 | 102.376 | 1.0000 |
| 0.50 | 2.355e-03 | 2.8133 | 25.594 | 0.2475 |
| 1.00 | 2.570e-03 | 2.8133 | 6.399 | 0.0567 |
| 2.00 | 2.333e-03 | 2.8133 | 1.600 | 0.0156 |
| 4.00 | 2.356e-03 | 2.8133 | 0.400 | 0.0039 |

A 16× swing in the ambient key/value scale moves the two terms' relative weight
by **256×**. `keys_v`/`values_v` are both `entity_adapter(embed(·))` outputs and
the adapter's output scale is trained (by CE and aux) and known to drift — this
program has an entire lane on adapter-geometry drift. A `λ_t` calibrated at
Stage 0′ against one adapter scale is a different effective `λ_t` 20,000 steps
later. The scale-free form `‖Zw‖² / (‖Z‖²_F / d)` is invariant (verified:
`0.354890` at `α ∈ {0.25, 1, 4}`) and is the same quantity F1's Band-2 repair
already needs.

### M2 — Band 2's `L_key ≤ 3e-4` has **no demonstrated achievability**; the best available pre-launch probe puts the real encoder ~10³ away, in exactly the pre-registered Band-4 depth-decay shape

The WIN calibration is a *sufficiency* table (perturb `Z_ideal`, watch
retrieval). Nothing anywhere establishes that the 173,209-param
`BindingEncoder` can *reach* `L_key ≤ 3e-4` on this task. Executed, in
write-side **isolation** (no CE, no aux, no ortho, no adapter drift, exact
targets, held-out episodes):

| probe | steps | best `L_key` | `retr@1` | `retr@61` |
|---|---|---|---|---|
| fresh ρ=0.219 geometry every step, lr 3e-4 (`r4_6`) | 3,000 | 9.48e-01 | 0.0000 | 0.0312 |
| same, lr 1e-3 (`r4_6b`) | 8,000 | 3.95e-01 | 0.9844 | 0.0312 |
| same, lr 3e-3 | 8,000 | 4.18e-01 | 0.9531 | 0.0469 |
| **finite entity pool (106 entities, the real construction), lr 1e-3** (`r4_6c`) | 12,000 | **2.23e-01** | **1.0000** | **0.0469** |
| near-orthonormal keys (easiest possible), lr 1e-3 (`r4_9`) | 8,000 | 7.31e-03 | 1.0000 | 0.8906 |

On the same held-out episodes `Z_ideal` itself reads `retr@61 = 0.9844`
(finite-pool) and `1.0000` (orthonormal) — **the target and the read are fine;
the encoder's achievable precision is the wall.** And the failure has exactly
the shape D5.3 Band 4 pre-registers as the PARTIAL signature: perfect at `h=1`,
chance at `h=61`.

**Caveats, stated plainly:** synthetic ρ-matched / pool-matched keys, not real
extracted `keys_v`; no CE/aux co-training and no `entity_adapter`
co-adaptation (the real run *can* reshape the key geometry toward the
near-orthonormal regime where the last row succeeds — a genuine escape my
probe blocks); ≤12k steps vs 20k; no LR schedule/warmup; one seed; CPU fp32.
This is **not** a prediction that Stage 1 fails. It is a demonstration that the
single most consequential quantity in the design — whether the pre-registered
mechanism band is reachable at all — is unmeasured, is cheap to measure before
launch, and the only measurement anyone has taken says it is far. Spending
24.94 GPU-h without it risks six cells that all read
`INCONCLUSIVE-BY-MECHANISM` with a Band-4 signature, i.e. the wave's most
likely outcome would be the one outcome the bands cannot cleanly interpret.

**Repair:** Stage 0′ amendment A6 (parametrised achievability probe), gating.

### M3 — CONTROL C is funded (4.861 GPU-h) but enters **no** adjudicating predicate; and it exists only for compB while all three recipes are scored against the ortho-ON `P0_ref`

D2 funds CONTROL C precisely to close the two-variable confound (supervision
added **and** ortho removed). Then D5.3's bands use it nowhere:

| band | uses CONTROL C? |
|---|---|
| Band 0 leak gate | no (config flags) |
| Band 1 TPC | "monitored against P0/CONTROL C trajectories" — no numeric predicate |
| Band 2 mechanism | no (PRIMARY quantities) |
| Band 3 retrieval (the verdict) | no — `fraction_closed_recipe` uses `P0_ref`, the **ortho-ON** baseline |
| Band 4 depth decay | no (per-hop `P0`) |
| Band 5 answer_accuracy | disclosure only |

Worked counter-example (`r4_8_bands.py` §D, compB anchors): a PRIMARY reading of
`0.7100` scores `fraction_closed = 0.707` ⇒ **WIN** — and it scores WIN
identically whether CONTROL C reads `0.0664` (supervision did everything) or
`0.7000` (ortho removal alone did everything, supervision-specific increment
`+0.0100`). The confound the cell was bought to close is not closed by the
scoring rule.

Second half: CONTROL C runs compB only, but the aggregation table (D5.4) can
declare **MAJORITY WIN** on compA + primary — two recipes with no matched
ortho-off baseline at all.

**Repair.** Add to the WIN predicate:
`x − CONTROL_C(recipe) > 0.15` for compB, and for compA/primary either fund the
matched cell or pre-register explicitly that their WINs are
ortho-confounded-and-disclosed (a legitimate, but *stated*, weaker claim). At
minimum, `fraction_closed` must be reported against **both** `P0_ref` and
CONTROL C, and D5.4's verdict labels must carry the C-conditioned reading.

### M4 — item 3 (the STOP gate) computes a different statistic than its own spec, rests on a false premise, and discards the one absolute check that Band 2 makes load-bearing

Three defects in one gate:

**(a) statistic mismatch.** The spec (R3, "folded in verbatim") requires the
per-row norm distribution's **within-episode max/min**. The code computes
`required_dynamic_range = row_norm_max / row_norm_med` — a *global* max over a
*global* median. Executed at compB geometry, n=256:

| statistic | value |
|---|---|
| global max / global median (**carded**) | 21.909 |
| global max / global min | 94.782 |
| **within-episode max/min (spec)**: median / p99 / max | **5.372 / 19.001 / 30.988** |

Three defensible-sounding numbers spanning 5×–95×, and the STOP/GO decision
depends entirely on which one is used. Nothing in the card says which episode
statistic the gate quantifies over.

**(b) false premise.** The docstring justifies "achievable range =
`cond(row_out)`" by asserting `row_out` is "applied to a **fixed-norm**
LayerNorm output". `nn.LayerNorm` here is `row_norm` with **learnable affine**
(`model_v4.py:51`): output is `γ ⊙ x̂ + β` with `‖x̂‖ = √h = 8` but `γ, β` free
after training, so the input norm is not fixed and the identity does not hold.

**(c) the dropped absolute check, and an internal contradiction.** The card
discards an absolute row-norm comparison as a "unit mismatch". But an absolute
ceiling exists and is computable from the same checkpoint the gate already
loads: `σ_max(W)·(‖γ‖_∞√h + ‖β‖) + ‖b‖`. At init that ceiling is **7.21**
against required per-row norms of median 4.05, **p99 24.56, max 88.64**. The
"unit mismatch" defence is only valid if the target is required *up to a global
scale* — which is exactly what F1 shows Band 2 does **not** allow, since it
gates on an absolute `L_key`. The design cannot have it both ways: either
Band 2 becomes scale-invariant (F1's repair) and item 3's ratio framing is
right, or Band 2 stays absolute and item 3 must add the absolute ceiling.

### M5 — the `pinv` truncation cliff: past `cond(keys_v) ≈ 3.4e5` the pinned `teacher_force_operator` silently stops being a zero-residual interpolant, and `L_key(Z_ideal)` lands **100× outside the WIN band** — while item 1 (`cond`) is explicitly NON-GATING

`torch.linalg.pinv` truncates singular values below `rtol·σ_max` with fp32
default `rtol = max(M,N)·eps = 2.98e-06`. Executed (`r4_4b_pinv_truncation.py`,
one key made progressively collinear):

| `cond(keys)` | `L_key(Z_ideal)` | `‖Z_ideal‖_F` | worst per-key rel. residual |
|---|---|---|---|
| 5.7e+01 | 1.73e-12 | 22.6 | 2.9e-05 |
| 1.7e+03 | 6.17e-10 | 574 | 2.6e-04 |
| 1.8e+04 | 6.53e-08 | 5,859 | 9.4e-03 |
| 5.9e+04 | 8.16e-07 | 20,597 | 5.3e-01 |
| 1.6e+05 | 5.74e-06 | 50,090 | 8.6e-01 |
| **1.7e+06** | **3.46e-02** | 16.7 | 9.3e-01 |

Fraction of episodes whose **target itself** violates the WIN band
(`L_key(Z_ideal) > 3e-4`): 0.000 at cond 1.8e4, **0.016** at 5.9e4, **0.141** at
1.6e5, **1.000** at 1.7e6.

Two consequences the design does not carry: (i) D1.3's proof premise
(`L_key(Z_ideal) = 0`) is conditional on `cond(keys_v)`, and the design states
it unconditionally; (ii) in the intermediate regime `‖Z_ideal‖_F` reaches
`5×10⁴`, which is what item 3's reachability question is actually about — the
required dynamic range is a function of the conditioning tail, not a fixed
property. Item 1 measures `cond` and is marked **"Informational, non-gating
(M2)"**. M2's finding was that *retrieval* is insensitive to `cond` **for an
exact write** — it does not license non-gating for the *validity of the
supervision target*.

**Repair.** Item 1 becomes gating on a registered statistic: report `cond`
median/p99/max **and** the fraction of episodes with `L_key(Z_ideal) > 3e-4`
(a two-line addition using tensors item 1 already has). If that fraction is
non-trivial, Band 2's `L_key` floor is distribution-limited and must be
re-derived, not inherited.

### M6 — the registered Tikhonov fallback does **not** yield "a small neighborhood of `Z_ideal`"; it re-creates the competing-minima situation D1.3 is built to avoid

D3 registers, as the contingency for ill-conditioned keys: *"project onto the
space spanned by the smallest **two or three** singular directions rather than
the single smallest … adopting it changes D1.3's zero-set proof (it would no
longer be exactly `{Z_ideal}` but a small neighborhood of it)."*

`w₂ = Vh[:,-2,:]` is the smallest **nonzero** singular direction — it is *in the
row space of `keys_v`*, i.e. in the very subspace `L_key` supervises. Executed:

| conditioning | `‖Z_ideal w₁‖` | `‖Z_ideal w₂‖` | as % of `‖Z_ideal‖_F` |
|---|---|---|---|
| benign (eps=1) | 1.60e-06 | **18.60** | 80.7% |
| mild (eps=1e-2) | 4.42e-05 | **625.6** | 100.0% |

Penalising `w₂` fights `L_key` head-on for 80–100% of the operator's norm. The
fallback is not a mild variant of the design — it is a different objective with
genuinely competing minima, exactly the F2-shaped situation D2 just spent a
control to escape. Keep it registered if you like, but the sentence describing
its consequence must be corrected, and if it is ever adopted the whole Band-2
calibration must be re-derived.

### M7 — Band 2 is a hard gate with **no emission point**: nothing in the pipeline logs `L_key` or `‖Z_sgd w‖` at eval

`eval_arm_at_hops` emits exactly `recovered_frac@0.9`, `mean_cos`,
`answer_accuracy`, plus `discriminability_metrics`'s four fields
(`offtarget_margin`, `retrieval24_acc`, `o_pairwise_cos`,
`target_pairwise_cos`). No `Z`-side diagnostic exists anywhere in the runner.
D8's provenance item (M8's repair) adds the two **config flags** and says
nothing about the two **measured quantities** the band gates on. Band 1
(TPC ✓), Band 3 (retrieval24 ✓), Band 4 (per-hop ✓), Band 5
(`answer_accuracy` ✓) and Band 0 (`teacher_force_check` ✓) all have homes;
Band 2 does not. As specified, the harvest cannot evaluate its own second gate.

**Repair.** Name the artifact field: extend `eval_arm_at_hops` (full_graft only)
to emit `write_diag = {L_key, L_key_cstar, Zw_norm, Z_fro, Zw_ratio,
keys_cond, keys_null_gap}` per hop, and register it in D8's build brief
alongside the config flags. Zero marginal cost — every tensor is already
in scope at that call site.

### M8 — item 5 is labelled gating with no predicate, drops the curve R3 actually asked for (so "items 1–5 fold in **verbatim**" is inaccurate), and its gradient norm is off by `1/B`

- R3's item 5 = `ortho_regularization_loss(Z_ideal)` + `‖∇_Z ortho‖`
  **"and the joint-minimisation `λ_w` curve of F2 re-run on real key
  geometry — the input to any honest `λ_w0`."** The carded item 5 keeps the
  first two and drops the joint-minimisation curve entirely. The section
  header claims items 1–5 fold in "**verbatim**" and flags only item 6 as a
  substitution. Two substitutions, one disclosed.
- The gating readout says *"Item 5 FAIL (ortho/`L_write` conflict does NOT
  reproduce on real geometry)"* — but the carded code returns no `gate_pass`
  and no threshold; `ortho_loss_at_Z_ideal > 0` with nonzero gradient is
  the only available signal and no number is registered as "reproduces".
- `ortho_regularization_loss` returns a **batch-mean** scalar (runner:743),
  so `torch.autograd.grad(...)` yields per-example gradients scaled by `1/B`.
  At `n=256` the reported `grad_norm_med` is `256×` smaller than the
  per-example quantity F2's synthetic evidence was expressed in — the two
  numbers are not comparable, which is the entire purpose of the item.
- Bookkeeping: the section header says "**six items, three gating**"; the item
  spec marks GATES on 3, 4, **and 6**, and then says item 5 "stays **gating**".
  That is four. The gating set must be stated once, unambiguously.

### M9 — the budget cap unilaterally revises the binding charter, the registered per-cell ceilings sum past the old cap, and the 5.5 ceiling is *below* the archive's own precedent for the identical recipe under a strictly heavier eval

- `§A3-ADJUDICATION`, which DRAFT-R4 opens by calling "adopted in full,
  binding", ends: *"wave-1 still **≤30 GPU-h hard cap** on GPUs 4/6/7
  post-CLEAR."* D6 registers **≤35**. It is disclosed, and the reasoning
  (Control C is D2-mandatory) is sound — but a design section cannot raise a
  cap its own charter set; this needs the coordinator's ratification in an
  adjudication block, not a paragraph in the revision.
- The registered per-cell ceilings sum to `6 × 5.5 + 0.6 = 33.6` GPU-h, so
  under contention the wave can consume 33.6 GPU-h with **no ceiling firing** —
  past the ≤30 charter, inside the new ≤35.
- The three `mob_g3b31_*` cells — the *identical* recipe — ran with
  `ceiling_gpuh = 6.0` and cost `0.8116/0.8293/0.8403` GPU-h at
  `0.146–0.151 s/step` with `eval_batch_size = 64`. This wave sets **5.5**
  (below precedent) while raising eval batch to **256** (τ's own requirement,
  4× the eval work, priced by m5 only *after* the ceiling was chosen). The
  ABORTED-BUDGET trap D6 exists to close is re-armed from the other side.
  Set `--ceiling-gpuh 6.0` (precedent) or defer the number to the
  pre-launch re-smoke that m5/m2 already require.

### M10 — item 4's `gate_pass` polarity is **inverted** relative to its own gating readout

Code: `gate_pass = (p90(‖Z_sgd w‖) <= 3.0)`. Gating readout: *"Item 4 shows
trained-checkpoint transverse gain `≫3` ⇒ **CONFIRMS** D1's mechanism is
necessary (proceed, with elevated confidence, not a blocker)."* So the outcome
the design *wants* (`≫3`) writes `gate_pass: false` into the JSON, and the
outcome that would undercut D1's necessity (`≤3`) writes `gate_pass: true`. A
harvest agent reading the artifact field gets the opposite of the registered
meaning, and there is no branch for the ambiguous middle. Rename to
`transverse_gain_exceeds_3` with the semantics spelled out in the same dict, or
drop the boolean and let the harvest apply the registered rule to the number.

### M11 — the transverse term is hard-wired to `d − K = 1` and will silently under-supervise on the spearhead's own K-ladder

`w = Vh[:, -1, :]` supervises **one** direction. That is exactly right at
`K=24, d=25`, and D1.3's proof depends on it (`N = {u wᵀ}` is 25-dimensional
*because* the null space is 1-dimensional). At any `K < d − 1` — the K-ladder
is the next rung of the spearhead, and `NCR_KLADDER_DESIGN.md` is live — the
null space is `(d−K)`-dimensional, the penalty covers `1` of `d−K` directions,
and the zero set silently reopens to a `25(d−K−1)`-dimensional family with the
same Lyapunov failure F1 diagnosed. One-line generalisation, worth registering
now while the proof is fresh: `W = Vh[:, K:, :]` `(B, d−K, d)`,
`L_transverse = ‖Z Wᵀ‖²_F / v̄²`, and the proof goes through verbatim with
`u wᵀ → U Wᵀ`.

---

# minor findings

**m1 — item 6's init perturbation is 25× its stated size.**
`Z_ideal + 0.05·‖Z_ideal‖_F·randn_like(Z)` applies per-element noise of scale
`0.05‖Z_ideal‖_F`, giving a Frobenius perturbation of
`0.05·√(d·d)·‖Z_ideal‖_F = 1.25‖Z_ideal‖_F` — measured **1.254×**, not 5%. The
intended 5% needs `/ (d*d)**0.5`. (Harmless in a free-`Z` fit; not harmless in
the parametrised replacement, where the init point matters.)

**m2 — Stage 0′ episodes are not the episodes any recorded number came from.**
`extract_real_kv` seeds `BASE_SEED + R.EVAL_SEED_OFFSET`; every eval in this
program seeds `base_seed + EVAL_SEED_OFFSET + h` (runner:940). Same
distribution, different draw — so item 4's transverse gains are not measured on
the episodes that produced `P0 = 0.0664`. One-token fix (`+ 61`), and it makes
the numbers directly comparable.

**m3 — `OUT_DIR` is never created, and `CEILING_S` is not a kill-switch.**
`open(os.path.join(OUT_DIR, ...))` with no `os.makedirs` (the battery's own
`_write` did it); the path survives only because the battery already created it.
`CEILING_S` prints a warning *after* the job finishes — it cannot stop a hung
job, and its comment ("`40*60 # 0.67 GPU-h < the ~0.1 GPU-h expected`") is
self-contradictory. Wrap the tmux command in `timeout 2400` for a real ceiling.

**m4 — `CUDA_VISIBLE_DEVICES=0` is hardcoded** while the GPU-class note says
"any single free H100 among 0–7". Pick one; the wave is placement-sensitive
(GPUs 4/6/7 reserved) and GPU 0's occupancy is not established in the card.

**m5 — item 6 reports stale finals.** `L_key`/`Zw` are read from the last loop
iteration, i.e. *before* the final `opt.step()`.

**m6 — D5.1's compA/primary anchors are rounded.** Raw: compA `P0 = 0.035156`,
primary `P0 = 0.039062`, both gaps `0.960938`; the design carries `0.0350/0.0390`
and gaps `0.9611/0.9610`. Threshold shift ≈ `1e-4` — immaterial, but the harvest
should recompute from the raw JSONs rather than the table.

**m7 — "structurally impossible for `L_write` to cause collapse" is one step too
strong.** The detach removes the *direct* gradient route to
`entity_adapter`/`embed`. `L_write` still changes `Z → o_raw → CE`, and CE
trains `entity_adapter` — a second-order route. Restate as "no direct gradient
route; any TPC movement is mediated by CE/aux", which is what D3's own
comparison discipline actually needs.

**m8 — `restore_arms_and_opts` builds Adam states in an eval-only job.** Matches
the battery's precedent; harmless, ~2× the parameter memory for nothing.

**m9 — record hygiene:** commit `8c665f7`'s message asserts "gradient subspaces
**provably orthogonal**" with no space qualifier. Per F2 that is true in
Z-space and false in parameter space (`cos = 0.17–0.45` measured). Same class
as the "provably fails → systematically fails" obligation the design body
otherwise discharges cleanly (V10).

**m10 — "the true floor is 127.8"** is the floor across the five *measured*
geometries; the R2 table's absolute floor including the orthonormal TOY row is
`0.000`. The design's wording is defensible; adding "measured" removes the last
ambiguity in a dead-mechanism number.

---

# §STAGE-0′ RULING

**BLOCKED AS CARDED. CLEARED TO RUN pre-CLEAR after the amendments below** —
which are cheap (one afternoon of build), leave the item structure intact, and
make Stage 0′ *more* load-bearing than the card claims: after A6 it becomes the
instrument that decides whether the 24.94 GPU-h is worth spending at all.

The authorisation in `§A3-ADJUDICATION` ("Stage 0′ MAY run pre-CLEAR once Rev-4
folds its spec") is conditioned on the spec being folded correctly. It was
not: as carded the script raises on line 1 of its own loop (F3.1), and would
raise twice more if it got past that (F3.2, F3.3).

**Amendments (all binding; A1–A5 and A7–A8 are one-liners, A6 is a
replacement):**

- **A1 — checkpoint paths.** Use the paths the premise battery actually used
  and recorded:
  `~/ncr_g3b31_contrastive/results/mob_g3b31_{tag}_s0_ckpts/mob_g3b31_{tag}_s0.ckpt.pt`
  for `tag ∈ {primary, compA, compB}`. Add a pre-flight `ls` step (the battery
  card's own step 1) before the run.
- **A2 — attribute path.** `arms["full_graft"]["ncr"].encoder.row_out`.
- **A3 — item 3 statistic.** Report all four: global max/median, global
  max/min, within-episode max/min (median, p99, max), and the absolute
  reachable ceiling `σ_max(W)(‖γ‖_∞√h + ‖β‖) + ‖b‖` vs `‖Z_ideal‖` row-norm
  p99/max. **Gate on the within-episode p99** (the spec statistic), and state
  the STOP direction in the emitted dict.
- **A4 — item 4 semantics.** Emit `transverse_gain_med`, `_p90`,
  **`transverse_ratio_med/p90` (`‖Zw‖/‖Z‖_F`)**, and replace `gate_pass` with a
  named field whose polarity matches the registered reading (M10).
- **A5 — item 5.** Either register a numeric predicate for "the conflict
  reproduces" (e.g. `ortho_loss(Z_ideal) > 10³` **and**
  `‖∇_Z ortho‖_per-example > 0`, with the `1/B` corrected by multiplying by
  `B` or by using per-example losses) **or** demote item 5 to informational and
  say so in the gating readout. Restore R3's joint-minimisation `λ_w` curve, or
  disclose its removal as a second substitution.
- **A6 — item 6 REPLACED (the substantive one).** Drop the free-`Z` sweep; it
  cannot fail (F2c). Replace with a **parametrised achievability + `λ_t`
  probe**, on real extracted `(keys_v, values_v)`:
  1. keep the probe batch from `extract_real_kv` (return it) so `entity_ids`,
     `tgt_slot`, `query_key_col` are available;
  2. instantiate a **fresh** `graft.build_ncr_head()` (`d=25, h=64`), train it
     on `L_key + λ_t·L_transverse` over the extracted episodes (or fresh
     forward passes) for ≥8,000 Adam steps at the runner's own `lr=3e-4` *and*
     at `1e-3`, for `λ_t ∈ {0, 0.1, 1.0, 3.0}`;
  3. score with the real path — `o = nm.binexp_read(Z, q_key.unsqueeze(1),
     h)["o"].squeeze(1)` then `R.discriminability_metrics(integ,
     backbone.embed, o, entity_ids, tgt_slot)` — on **held-out** episodes at
     `h ∈ {1, 61}`; if multi-hop scoring is wanted, build one batch **per hop**
     (`build_task1_document(cfg, pools, gen_h, n, h, device)`) and fit per hop,
     rather than scoring an `h=61` fit at other hops (F3.5);
  4. report the `L_key`, `‖Zw‖`, `‖Zw‖/‖Z‖_F` **learning curves** (not just
     finals) and the `retr@61` reached.
  **GATES:** if no `λ_t` reaches Band 2's (repaired, scale-invariant) targets
  within the probe budget, Stage 1 does not launch on the current band —
  either the band is re-derived from the probe's own achievable frontier
  (with the change disclosed as a post-hoc recalibration) or the mechanism is
  rescoped. There is no version of this wave worth 24.94 GPU-h if the
  mechanism band is unreachable, and this probe costs minutes.
- **A7 — item 1 becomes gating** on `cond` **and** on the fraction of episodes
  with `L_key(Z_ideal) > 3e-4` (M5). Keep `S[-1]/S[-2]`; also report `σ_min`
  relative to `σ_max` since that, not the ratio, is what sets `pinv`'s
  truncation.
- **A8 — housekeeping:** `os.makedirs(OUT_DIR, exist_ok=True)`; seed
  `+ EVAL_SEED_OFFSET + 61` (m2); `timeout 2400` in the tmux command (m3); fix
  the init scale `/ (d*d)**0.5` (m1); resolve `CUDA_VISIBLE_DEVICES` vs "any
  free GPU" (m4); read finals after the last `opt.step()` (m5).

**Cost after amendment:** still ≲0.2 GPU-h (A6 adds a few minutes of
single-GPU training on a 173K-param head). Archive to
`experiment-runs/2026-08-13_ncr_writecond_premise_battery/` as carded.

**Everything else about the card is fine** and should be kept verbatim: the
single-process/three-checkpoint structure, the tmux-by-name discipline, the
`tmux kill-session` instruction, the no-`pkill` rule, the output path, and the
honest ≈0.1 GPU-h pricing (V12).

---

# BINDING DISPOSITION PROPOSAL (for the coordinator)

**E1 (F1, mandatory).** Rewrite Band 2 scale-invariantly:
`L_key(c*·Z_sgd) ≤ 3e-4` **AND** `‖Z_sgd w‖/‖Z_sgd‖_F ≤ 0.12` (R3's own 12%
statement, restored with its conditioning), `c*` in closed form per episode,
reduction statistic named. Retract D4's "by construction" sentence — it is the
inverse of the truth, and the retraction should be as explicit as D1.4's own.

**E2 (F2, mandatory).** Scope D1.3's orthogonality bonus to Z-space with the
measured parameter-space cosine on the record. Replace item 6 with the
parametrised probe (A6). Either add `λ_t` as a Stage-1 axis on one recipe or
pre-register a `λ_t = 0` necessity arm; the wave currently cannot tell whether
D1's own new term helps, hurts, or is inert in the parametrisation.

**E3 (F3, mandatory).** Stage 0′ amendments A1–A8 before it runs.

**E4 (F4, mandatory, one clause).** Band 0 exempts CONTROL B and inverts for it;
name D7's clean-eval artifact as the scoreable output.

**E5 (M1).** Make `L_transverse` scale-free (`‖Zw‖²/(‖Z‖²_F/d)`, or the ratio)
so `λ_t`'s effective weight does not drift with the adapter's output scale.

**E6 (M2).** Achievability is gating, at Stage 0′, before any Stage-1 GPU-h.

**E7 (M3).** Wire CONTROL C into the WIN predicate, or state in D5.4 that
compA/primary WINs are ortho-confounded-and-disclosed.

**E8 (M4–M6, M8, M10, M11).** Item-3 statistic + absolute ceiling; item-1
gating on the target's own validity; correct D3's Tikhonov-consequence
sentence; item-5 predicate + `1/B`; item-4 polarity; register the `d−K > 1`
generalisation of the transverse term.

**E9 (M7).** Name Band 2's emission point in the build brief
(`write_diag` in `eval_arm_at_hops`, full_graft only).

**E10 (M9).** Take the ≤30 → ≤35 revision to the coordinator explicitly; set
per-cell `--ceiling-gpuh 6.0` (archive precedent for the identical recipe) or
defer to the re-smoke.

**Ceremony.** The mechanism cleared this round; the residue is instrument-level.
One further narrow adversarial round scoped to **E1/E2/E4 + the amended Stage 0′
script only** (not a full re-attack of D1–D8), then the pre-launch
resource/placement red-team, then launch. Stage 0′ (amended) runs in parallel
with Rev-5 drafting — its output is an input to E1/E2/E6, so running it first
shortens the loop.

---

# What the BUILD ceremony MAY prepare while the residue closes

Everything below is downstream of the *verified-clean* mechanism and cannot be
invalidated by E1–E10 (which touch bands, `λ_t`, and the Stage-0′ script):

- `write_supervision_loss(Z, keys_v, values_v, lambda_t)` with the **single
  top-of-function detach** (D3, V4), `L_key` verbatim from D1.1, and
  `L_transverse` written in the **scale-free** form (E5) with `λ_t` as a flag —
  the flag's *value* is not decided, its plumbing is.
- The `w` computation, with the `d−K` generalisation (M11) written now:
  `W = Vh[:, K:, :]`, which reduces to the carded single-`w` at `K = d−1`.
- `--write-supervision-weight` / `--write-transverse-weight` in
  `rec["config"]` and in `run_two_arm_cell`'s resume asserts (D8/M8).
- The m4 mutual-exclusion assert.
- The Band-0 harvest checker **with F4's CONTROL-B branch**.
- `write_diag` emission in `eval_arm_at_hops` (E9) — including
  `L_key_cstar`, `Zw_ratio`, `keys_cond`, so whichever band Rev-5 lands on has
  its number already logged.
- Per-step separated `‖∇_Z L_key‖` / `‖∇_Z L_transverse‖` logging for the real
  and placebo arms (D7 — and now doubly worth having, since F2 shows the
  parameter-space relationship is the open question).
- CONTROL B's separate `teacher_force=False` clean-eval script, on the
  `pbe_repl.py` pattern.
- The amended `stage0prime_eval.py` (A1–A8).

**May NOT be prepared or run:** any Stage-1 cell; any commitment to a `λ_t`
value; any band threshold (Rev-5 + Stage 0′ decide them jointly).

---

# Reproduction

All sims are pure-CPU, fp32, torch 2.8.0, no GPU, no box contact. They import
the **real** `nm.binexp_read` and the **real** encoder
(`ncr_models.NCRModel` → `chapter2/model_v4.BindingEncoder`, 173,209 params
= `NCR_PARAM_EXACT`), and reuse R3's own harness (`sim_common.py`:
`teacher_force_operator`, `l_write`, `retrieval24` reproduced verbatim from the
pinned runner) so this round's numbers are directly comparable to R3's.

```
scratchpad/r4_1_zeroset.py            V1-V3, D1.3 re-derived + executed at compB geometry
scratchpad/r4_2_gradortho.py          V5 + F2(a)(b): Z-space vs parameter-space gradients
scratchpad/r4_3_scale.py              F1: Band-2 scale sensitivity, c*-rescale, ratio form
scratchpad/r4_4_svd.py                V4 + M5/M6 setup: detach coverage, w jitter, grad norms
scratchpad/r4_4b_pinv_truncation.py   M5/M6: the pinv truncation cliff, 2-direction fallback
scratchpad/r4_5_item6.py              F2(c) + m1: item 6 as carded is non-diagnostic
scratchpad/r4_6_encoder_curve.py      M2: ||Zw|| learning curve, real encoder, fresh geometry
scratchpad/r4_6b_longer.py            M2: longer/hotter/easier-geometry controls
scratchpad/r4_6c_pool.py              M2: finite 106-entity pool (the real construction)
scratchpad/r4_7_reach.py              M4: item-3 statistics, LN premise, absolute ceiling
scratchpad/r4_8_bands.py              V6/V7 + M3: partition scan, raw-anchor check, Control C
scratchpad/r4_9_easy.py               F2(d): lambda_t contrast at the easiest geometry
scratchpad/r4_10_norm.py              M1: the alpha^-2 normalisation drift
```

(scratchpad root:
`/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad`)

**Raw artifacts read this round:**
`experiment-runs/2026-08-13_ncr_writecond_premise_battery/*.json` and
`*.py.txt` (checkpoint paths, `target_pairwise_cos`, `retrieval24_acc`,
`elapsed_s`, the `pbe_repl`/`pbe_supplement` launch pattern);
`experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/ncr_lm_wave1_runner.py`
(md5 `9a93198b642242f512ff8489e32b0a53`, confirmed = the pinned runner) and its
three `mob_g3b31_*.json` (timing, `config`, ceilings);
`experiment-runs/2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_smoke.py`
(`teacher_force_operator`, `build_task1_document`, `D_NCR/H_NCR/K_NCR`);
`matrix-thinking/ncr/ncr_models.py`, `ncr_earlyln_scale.py`,
`matrix-thinking/chapter2/model_v4.py` (the head's real class hierarchy);
`matrix-thinking/deltanet_rd/grammar_rd.py` (entity-pool size);
`NCR_WRITECOND_ATTACK_R2.md:105-118` (the `f(A*)` table),
`NCR_WRITECOND_ATTACK_R3.md` in full.

**Not re-litigated:** R3's V1–V13, the premise-battery facts, the novelty
verdicts. **No repo file other than this report was written; no STATE.md /
EXPERIMENT_LOG.md edit; no commit; no box contact; no GPU spend.**
