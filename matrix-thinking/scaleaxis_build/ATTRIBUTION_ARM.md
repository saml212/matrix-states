# THE STEP-EXTENSION ATTRIBUTION ARM — two priced options, one election

**Trigger:** EXPERIMENT_LOG 2026-08-22 **#21** (commit `6a06e1c`) — the scale
chapter's verdict of record contains **three SCALE-DEGRADES readings**, and
`NCR_SCALE_AXIS_DESIGN.md` DRAFT-R2 makes the token-budget attribution control
**mandatory before any of them is published**. Nothing here is queued; the
election is the coordinator's.

---

## 1. The governing design text, verbatim

§7.2, the block titled *"The step-extension attribution arm — now unconditional
on any DEGRADES verdict (MAJOR-4)"*:

> **No SCALE-DEGRADES verdict — on any curve, at any K, from calibration or
> from the sweep — is published without a step-extension attribution arm at
> the degrading K: 2 cells (frozen, seeds 0-1) at 40,000 steps, ≈+8.5 GPU-h
> at ×3.75 at K=40.** If the doubled-token cells recover to κ ≥ 0.90, the
> verdict is **TOKEN-BUDGET-LIMITED**, not scale-fragile, and is reported as
> such. If they do not, SCALE-DEGRADES stands and is *strengthened* by the
> control.

§7.1, which fixes its scope:

> **Adopted fix: the attribution arm is now conditional on ANY
> SCALE-DEGRADES verdict at ANY K, at harvest, with the pinned rule — "no
> SCALE-DEGRADES claim is published without it."** Priced in §8.2.

> **And the confound is one-directional, which is a strength (§1).**
> Under-training can only manufacture DEGRADES. It cannot manufacture STABLE
> and it cannot manufacture IMPROVES. So the attribution arm is needed for
> exactly one of the three outcomes, and the other two are *strengthened* by
> the very mismatch FROZEN_BIAS warns about.

§8.2's contingency line:

> | **Attribution arm (2 cells @ 40,000 steps at the degrading K)** | **ANY SCALE-DEGRADES verdict, any K — MANDATORY before publication** | **+8.5** (at K=40) |

§8.2's joint-contingency gate:

> **Pinned consequence:** the two contingencies are not jointly pre-authorized.
> The **first** to trigger (§7.2 branch (B)'s step extension, or the §7.2
> attribution arm) runs on the pre-registered rule. If the **second** would
> push the realized all-in total past **130 GPU-h**, it requires a **fresh
> gate** …

Branch (B) **never fired** — calibration cleared all three license legs (#19) —
so **the attribution arm is the FIRST contingency and runs on the pre-registered
rule.** No fresh gate is required by the letter of §8.2.

---

## 2. What must be discharged, and by which cells

From #21:

| | verdict | the degrading cells |
|---|---|---|
| **V1** | Curve 1 SCALE-DEGRADES at **K=32 trainable** (`Δ_scale = −0.1169`, also fails the κ gate) | (32, compB) |
| **V2** | Curve 1 SCALE-DEGRADES at **K=40 trainable** (`Δ_scale = −0.1442`, also fails the κ gate) | (40, compB) |
| **V3** | Curve 5b depth tail: SCALE-DEGRADES at 11sq (`T=10.5/72`) and `s*=13` (`T=6.5/72`), LOSO 8/8, exact p to 9.5e-07 — **BOTH arms**. Per-K at `s*=13`: **6/8 cells degrade incl. frozen K=32/40**; frozen K=16/24 stable. | (16,compB) (24,compB) (32,compB) (40,compB) **(32,primary) (40,primary)** |

**Union of degrading (K, recipe) cells = 6**: trainable at all four K, plus
frozen at K=32 and K=40.

---

## 3. Two design gaps, surfaced rather than papered over

### GAP 1 — the pinned cell shape says **frozen**; four of six degrading cells are **trainable**

The rule's shape — *"2 cells (**frozen**, seeds 0-1)"* — is inherited from
§7.2 branch **(C)**, where the arm originally lived and where the reading is on
the **frozen** calibration cells (all three §4.2 license legs are frozen).
MAJOR-4 then made the arm *"conditional on ANY SCALE-DEGRADES verdict at ANY
K"* but **did not re-derive the cell shape**.

Applied literally, the arm would extend **frozen** cells to control **trainable**
degradations. That cannot work: the doubled-token cell would differ from the
degrading cell in exactly the axis the chapter's own ORDERING result shows is
decisive at 392M (`T_W = 36.0/36`, perfect separation, exact p = 1.25e-05). A
literal frozen-only arm would spend real compute and **discharge one of three
verdicts**.

**The substantive reading — the one that serves the rule's stated purpose** —
is that the extension runs **in the recipe of the degrading cell**. The rule's
purpose is stated in §7.1: to separate *"the capability is scale-fragile"* from
*"at a fixed token budget a 392M model is 4× further from compute-optimal"*.
Only a same-recipe extension does that. **Both options below adopt the
recipe-matched reading; it is disclosed here rather than elected silently.**

### GAP 2 — `κ ≥ 0.90` is the wrong recovery bar for V3

The pinned criterion is *"recover to κ ≥ 0.90"*. That is the **capability** bar
and §5.5(ii) forbids applying it at depth, in its own words:

> (ii) κ at 13/15 squarings is **not** the CAPABILITY bar — that bar lives at
> `h_top` (5 squarings) and is untouched. A 98M κ of ~0.70 at 15 squarings is a
> numerical-depth reading, not a capability failure, and must never be reported
> as one.

**Pre-registered restatement, fixed here BEFORE the arm runs** (and written into
every spec's hypothesis):

* **V1/V2 (Curve 1, `h_top`)** — the pinned rule applies unchanged:
  `κ ≥ 0.90` on **2/2** extended seeds ⇒ **TOKEN-BUDGET-LIMITED** at that
  (K, recipe); otherwise **SCALE-DEGRADES stands and is strengthened**.
* **V3 (Curve 5b, `s*=13`)** — the extended cell is **TOKEN-BUDGET-LIMITED**
  iff its `Δ_scale` against the **same 98M twin** moves back inside
  **±δ_depth = 0.095**; otherwise the depth-tail SCALE-DEGRADES stands.
  (δ_depth = 0.095 is Rule R-δ's mechanical output, fixed before any 392M cell
  existed.)

Both readouts are reported for **every** extended cell, whichever verdict
recruited it.

---

## 4. Mechanism: resume-extension, which is the design's own pricing basis

§7.2 branch (B) prices *"extend the six calibration cells only to 40,000 steps"*
at **"+≈18.6 GPU-h at ×3.75"** = `6 × 3.10`, i.e. the **marginal** 20,000 steps
— not a 40,000-step re-run. §8.2's **"+8.5 (at K=40)"** = `2 × 4.24` is the same
arithmetic. So **the design pins resume-extension**, and its own contingency
prices are only correct under that reading.

The runner supports true step-level resume (`for step in range(start_step + 1,
steps + 1)`, with `start_step` **and** `cumulative_elapsed_s` restored from the
checkpoint and the seed and freeze flag **asserted** against it), and all 24
sweep checkpoints are on the box (211 GB).

Each spec **hardlinks** the parent checkpoint into a fresh attribution directory
under a fresh cell id, then resumes there. The hardlink costs **zero bytes** on
the same filesystem, and because `atomic_torch_save` does `os.replace` — which
swaps the *directory entry* while the old inode survives — **the 20,000-step
checkpoint of record is not overwritten**. `--out` points at a new json, so the
Stage-C-scored record is untouched and the runner's `already COMPLETED —
skipping` guard (which keys on the **out** path) cannot fire.

**Verified live** on the literal spec-0230 command line (3 marginal steps):

```
RESUMING from checkpoint at step 20000 (cumulative_elapsed_s=16515,
                    seed=0 verified, freeze_entity_adapter=False verified)
COMPLETED at step 20003/20003
PARENT ckpt of record: step = 20000 | cell_id = scaleaxis392m_K40_compB_s0
EXTENDED ckpt        : step = 20003 | cell_id = LIT3_attrib40k_K40_compB_s0
sweep record still at step 20000 COMPLETED
```

### Disclosed property: the LR schedule is RE-OPENED (a warm restart)

`get_lr` is linear-warmup + cosine to `0.1 × max_lr` over `total_steps`.
Resuming with `--steps 40000` recomputes the schedule over 40,000, so the LR at
the resume point goes from **3.00e-05** (the 20k floor) to **≈1.66e-04** — a
**5.5× warm restart**. That is what *"extend to 40,000 steps"* means in this
harness and it is what branch (B) pre-registers and prices, **but a κ recovery
could then be attributed to the warm restart rather than to the tokens.**

The confound-free alternative is a **fresh 40,000-step run** with a single
cosine, which costs **exactly 2×** and is priced in every row below. **Offered
as an ELECT-or-DECLINE; not chosen here.**

---

## 5. The two options, priced from MEASURED cost

Per-cell costs are the realized 20,000-step `gpu_h` from the sweep of record
(mean over 3 seeds), **not** a projection — §10's *"Measured-vs-projected
bookkeeping"* bullet requires the harvest to price from measured totals.

| (K, recipe) | measured 20k GPU-h | s/step |
|---|---|---|
| (16, trainable) | 2.2937 | 0.4129 |
| (24, trainable) | 2.8663 | 0.5159 |
| (32, frozen) | 3.7842 | 0.6812 |
| (32, trainable) | 3.8214 | 0.6879 |
| (40, frozen) | 4.6056 | 0.8290 |
| (40, trainable) | 4.5948 | 0.8271 |

| option | cells | ids | **resume (design's basis)** | fresh-40k alternative |
|---|---|---|---|---|
| **A — full "at the degrading K"** (all 6 degrading cells × seeds 0-1) | 12 | `0230-0241` | **43.93 GPU-h** | 87.86 GPU-h |
| **B — minimal probative subset** ({frozen, trainable} × {K=32, K=40} × seeds 0-1) | 8 | `0230-0237` | **33.61 GPU-h** | 67.22 GPU-h |
| *(the design's own §8.2 line, for reference: 2 frozen cells at K=40 only)* | 2 | `0232-0233` | 9.21 GPU-h | 18.42 GPU-h |

**The marginal cost of full compliance over the subset is +10.32 GPU-h.**

### Is Option B compliant?

**Partially, and precisely so.** Against the pinned text *"at the degrading K"*:

* **Compliant** for **V1** (32, compB), **V2** (40, compB), and for **V3's
  "both arms" headline claim** — the frozen leg of V3 degrades at exactly K=32
  and K=40, and Option B controls both.
* **NOT compliant** for two rows of V3's per-K table: **trainable K=16** and
  **trainable K=24** degrade at `s*=13` and would be published with no control.
  Under Option B those two rows must be reported as *"degrades; attribution arm
  not run"* or held back.

**An argument that cuts AGAINST Option B, stated because it is inconvenient:**
the token-budget confound is strongest where `D/N` is smallest, and `D/N` at
392M is **0.209 / 0.284 / 0.375 / 0.467** at K=16/24/32/40. So the two cells
Option B drops are the **two most token-starved cells in the chapter** — the
places a token-budget rescue is *most* likely to appear. Option B saves 10.32
GPU-h by declining to test the confound where it is worst.

**An argument that cuts FOR Option B:** the three verdicts the coordinator named
as publication-gated are V1, V2 and V3's both-arms claim, and Option B
discharges all three at the exact cells they are declared on, at the design's
own `n = 2`.

*(A 6-cell variant dropping frozen K=32 would cost 25.9 GPU-h and is **explicitly
non-compliant** — it would publish the "frozen K=32 degrades at `s*=13`" row
with no control. Named here so it is not chosen by accident.)*

### Ledger check against §8.2's 130 GPU-h envelope

Realized chapter to date, **measured**: 81.42 GPU-h of training (24 cells) +
≈0.5 A0 + 0.145 Stage C + 0.15 the 98M re-score + ≈0.1 gates/smokes ≈ **82.3
GPU-h**.

| | all-in after the arm | vs the 130 GPU-h line |
|---|---|---|
| **A, resume** | ≈126.2 | inside, by 3.8 |
| **B, resume** | ≈115.9 | inside |
| A, fresh-40k | ≈170.2 | **breaches** |
| B, fresh-40k | ≈149.5 | **breaches** |

The 130 gate governs the **second** contingency and this is the **first**, so it
does not bind by the letter of §8.2 — but **both fresh-40k variants land past
the envelope the rule exists to protect**, which is the coordinator's call to
make explicitly rather than inherit.

**Bookkeeping correction:** EXPERIMENT_LOG #20's headline *"~69 GPU-h"* for the
24-cell wave is **below the measured 81.415 GPU-h** summed from the cells' own
`gpu_h` fields. §10's bookkeeping bullet requires the harvest to report measured
totals; the ledger above uses 81.42.

---

## 6. The ceiling: why the flat 0.516 s/step is launch-losing

The build brief specified ceilings *"from the measured realized rate 0.516
s/step × the step multiple."* **0.516 s/step is the K=24 rate specifically**
(0.5165 frozen / 0.5159 trainable), not a chapter-wide rate — the realized rate
runs 0.4129 (K=16) to 0.8290 (K=40).

Applied flat, it yields `1.5 × (0.516 × 40000 / 3600) = 8.60 GPU-h`. **The K=40
cells actually need 9.21 GPU-h of cumulative wall clock at 40,000 steps**, so
every K=40 cell — and every K=32 cell — would hard-abort with `ABORTED-BUDGET`.
That is MAJOR-1(b)'s own failure mode: *"a breaker that fires on every cell is
worse than one that fires on none."*

Compounding it: `run_two_arm_cell` restores `cumulative_elapsed_s` from the
checkpoint and sets `t0 = time.time() − cumulative_elapsed_s`, so
`elapsed > ceiling_s` is checked against the **whole 40,000-step** wall clock,
not the marginal half. The literal smoke printed `cumulative_elapsed_s=16515`
and an `elapsed` of `16533s` at step 20003 — i.e. **4.59 GPU-h already spent
before the extension's first step.**

**Applied instead, per §3.6's primary rule:**
`ceiling = 1.5 × R₈ × (full 40,000-step projection at this cell's OWN measured
rate)`, with `R₈ = 1.0026` measured at Stage A0.4 (#18):

| cell | full-40k measured | **ceiling** |
|---|---|---|
| (16, trainable) | 4.587 | **6.899** |
| (24, trainable) | 5.733 | **8.622** |
| (32, frozen) | 7.568 | **11.382** |
| (32, trainable) | 7.643 | **11.494** |
| (40, frozen) | 9.211 | **13.853** |
| (40, trainable) | 9.190 | **13.821** |

---

## 7. Election surface

The subset occupies the **contiguous low block `0230-0237`**, so:

* **Elect A** ⇒ stage `0230-0241`.
* **Elect B** ⇒ stage `0230-0237` only.

Per audit process-finding **p2** and condition **C8**: `queue_worker.sh` claims
by `ls | sort`, which **orders** claims but does not **block** them — **staging
is the enforcement, naming is not.** Nothing is staged; all 12 specs are
CANDIDATE-marked and live in `matrix-thinking/scaleaxis_build/job_specs_attribution/`,
outside the queue tree.
