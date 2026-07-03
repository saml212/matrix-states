# Stage 0 Design — Why the Matrix-Write Fails at d≥32, and What Fixes It

> **STATUS: CLOSED (2026-07-03; one of STATE.md's five closed 2026-07-01→03
> programs; decision at §14.5, full diagnostic §12-14).** The original
> d≥32 "trainability wall" is substantially a **step-budget artifact**:
> every d=32 baseline seed tested (17/17, across Wave 0, an extended-budget
> arm, and a 100K-step probe) transitions reliably and effective rank
> recruits to K once budget suffices. But even at 100K steps (10× Task D's
> original budget), the formal pass bar (`recovered_frac@0.9 > 0.7`) still
> **FAILS** at a genuine converged plateau (best observed 0.65 at K=8,
> cos 0.83-0.91 — not undertraining). The honest frontier is therefore not
> trainability (transitions are reliable) but **exactness**, which degrades
> with `d`. *Why* the d≥32 write plateaus sub-exact rather than reaching
> 1.0 is open, named **Stage 0.5**, and is explicitly NOT answered here.

**Drafted 2026-07-01, before any code changes.** Status: design only, per
instruction — no model/training code is written here. This is the bounded,
gated precursor `NEXT_EXPERIMENT_DESIGN.md` §0/§8 already named ("Stage 0 —
d≥32 trainability precursor") and flagged as blocking two things at once:
Task E's d≥32 tranche (which reuses `model_v4.BindingEncoder` verbatim — see
`model_e.py`, confirmed unmodified) cannot run until this is fixed, and the
ICLR-2027-main-track submission needs the d≥32 failure **explained**, not
just reported as an "optimization/architecture limitation" hand-wave
(`TASK_D_WRITEUP.md` §5.3, `TASK_D_FINDINGS_DRAFT.md` §4 — both currently
say, in effect, "we don't know why").

> **CHANGELOG — 2026-07-01, revision 2 (post-attack-round).** An independent
> adversarial review of revision 1 returned BUILD WITH FIXES (1 FATAL,
> 6 MAJOR, 5 MINOR). All findings are addressed in this revision:
> **FATAL-1** — the diagnostic signature table could not discriminate
> dead-init from a *late, seed-stochastic phase transition* (observed today
> in Task E's own 20K-step calibration on the same architecture family:
> loss flat at ≈1.0 for 60–80% of budget, then a sharp transition, at a
> ≤20% per-seed rate — see `run_task_e_sweep.py` header note). Fixed: new
> H_late-transition hypothesis (§2.4, §3), Wave 0 budgets extended to 2.5×
> tier defaults at ≥3 seeds on the primary cells, an explicit
> extended-budget baseline arm ("Arm 0") in the Wave A comparison,
> mid-training eval checkpoints as a build requirement, and a re-done
> GPU-h budget (§6; core now 57.4 GPU-h ≤ 60).
> **MAJOR-2** — manifest naming must carry an intervention/variant field or
> runs silently collide on resume (§6.3 build requirements).
> **MAJOR-3** — Wave −1 now calibrates wall-clock at h=128 too (§6).
> **MAJOR-4** — candidate 3's triple confound is pre-registered (§4).
> **MAJOR-5** — Wave A screens at K=8 *and* K=16 at d=32 (K=8 is the
> friendlier probe per the data); candidates 5 and 6 demoted to the
> fallback pool to pay for it (§4, §6).
> **MAJOR-6** — H_cap's limited explanatory reach at d=128 stated in §2.1/§9.
> **MAJOR-7** — new candidate 7 (per-row un-shared `row_out`), the only
> H_cap isolator, held as a fallback/Wave-B arm (§4).
> **MINOR-8** — symbolic param formula corrected to `d·(4h+1)` (§5).
> **MINOR-9** — the d=64 orthogonal-init asymmetry re-derived as a
> training-dynamics (zero-slack drift) effect, not an init-time one (§2.2).
> **MINOR-10** — self-attack #10's "K=d at d=64" corrected: Stage 0 never
> tests a true K=d cell (§8).
> **MINOR-11** — the write path's `row_norm(q+read)` is Post-LN
> (warmup-sensitive per Xiong et al. 2020); candidate 1's leverage
> discussion corrected upward (§4).
> **MINOR-12** — pre/post-clip gradient-norm logging added to Wave 0
> instrumentation (§6.2).

This document differs from prior treatments in one respect: it does not
start from "try known fixes for training instability." It starts from
reading `model_v4.py` line by line and deriving concrete, falsifiable
mechanisms for the failure — one architectural (provable, training-
independent), one an order-statistics argument about the encoder's write
path, and one empirical (the late-transition dynamics observed today in the
same architecture family) — before proposing interventions. The
interventions map onto these mechanisms, not onto a generic
hyperparameter-search grid.

---

## 1. The hypothesis, in one sentence

> **H_S0.** The d≥32 write failure is caused by a combination of (a) a hard,
> provable architectural rank ceiling `rank(Z) ≤ h+1` from the shared
> `row_out` projection (binds only for `d > h`, i.e. only at `d=128` in the
> tested grid), (b) an optimization failure in which the `d` row-reader
> queries — sharing one small `h=64`-dim identity space and one shared
> attention/projection mechanism — fail to differentiate into `K` coherent
> write directions as `d` grows past `h`, even where the architectural cap
> does not bind (`d=32, 64`), and/or (c) a training-budget effect in which
> the transition to a working write exists but is late and seed-stochastic
> (as observed today at d=16 in Task E's calibration on the same encoder),
> so the fixed 8–10K-step budget systematically undersamples it at larger
> `d`; all three are testable and at least (b) and (c) are fixable without
> fundamentally changing the model.

- **CONFIRM** (partial or full): at least one intervention — including the
  cheapest of all, "same architecture, longer budget" (Arm 0, §6) — recovers
  `recovered_frac@0.9 > 0.7` at `d=32` (unconstrained) for `K ∈ {4,8,16}` —
  the write becomes learnable, and Task D's already-confirmed
  rank-recruitment story (Chapter 2's central finding) extends to larger
  matrices. Stage 2 of Task E's d≥32 tranche unblocks.
- **FALSIFY** (no intervention singly or combined crosses the bar): d≥32
  training failure is not fixable by any of the probed mechanisms at this
  parameter/step budget — a real, actionable negative that scopes Task E to
  `d=16` (as `NEXT_EXPERIMENT_DESIGN.md` §8 already provides for) and
  redirects future compute toward task-complexity scaling instead of matrix-
  size scaling.
- **Diagnostic pass condition (independent of CONFIRM/FALSIFY on the fix):**
  Stage 0 succeeds as a diagnostic regardless of whether a fix is found, if
  Wave 0 (§6) produces a mechanistically specific, falsifiable account —
  i.e., it either confirms or rejects each of H_cap, H_collision,
  H_dead-init, H_undertrained, and H_late-transition (§3) as *measured*
  claims, not asserted ones. A "we tried N things and nothing worked, here
  is why we believe that, with evidence" write-up is a legitimate,
  publishable outcome; "the encoder can't scale, cause unknown" (the
  current state of the docs) is not.

---

## 2. Why now, and why this shape (not a blind hyperparameter sweep)

`TASK_D_WRITEUP.md` §5.3 and `TASK_D_FINDINGS_DRAFT.md` §4 already establish
the *symptom* precisely (re-pulled this session from
`results/overnight_snapshots/AGGREGATE_latest.json`, 991 runs):

| d | effective rank (unconstrained), range across K | M3 ceiling (`recovered_frac@0.9`, force-rank = d) |
|---|---|---|
| 16 | 2.4 (K=1) → 15.1 (K=16), tracks K almost exactly | 0.94–0.999 (K=8, K=12); **0.045 at K=4 (anomaly, see §7)** |
| 32 | 1.23–4.68, erratic, **not monotone in K** (K=8: 4.68; K=16: 2.63; K=24: 1.23) | 0.0 at every K tested |
| 64 | 1.01–1.05, flat ≈1 for every K | 0.0 at every K tested |
| 128 | 1.02–1.02, flat ≈1 for every K | 0.0 at every K tested |

(The user's cited 1,234-run replication reports a slightly wider d=32 range,
1.5–5.8 — consistent with the 991-run pull above; more seeds, same
qualitative picture: erratic, non-monotone, well below K. Note for §6:
**K=8 is the healthiest d=32 cell** — effective rank 4.68 vs 2.63 at K=16 —
so K=8 is the friendlier screening probe, per the attack round's MAJOR-5.)

What is **not yet established** anywhere in the repo is *why*. The existing
candidate-fix list (`NEXT_EXPERIMENT_DESIGN.md` §8: "LR warmup, normalization/
residual-scaling, curriculum on K") is a reasonable first guess but was
written without reading the encoder's forward pass for a structural cause.
Reading it (`model_v4.py:34–64`) surfaces two concrete mechanisms, and
today's Task E calibration adds a third:

### 2.1 Mechanism (a) — a provable architectural rank ceiling (new finding)

`Z[b] = row_out(q[b])` where `q[b]` has shape `(d, h)` (one `h`-dim vector
per row) and `row_out` is a **single shared** `nn.Linear(h, d)` applied
identically to every row:

```
Z[b] = q[b] @ W_row_out^T + 1_d ⊗ bias_row_out        # (d,h)@(h,d) + rank-1 term
```

Since `rank(AB) ≤ min(rank A, rank B)` and `W_row_out^T ∈ R^{h×d}` has rank
`≤ h`:

> **rank(Z) ≤ h + 1**, independent of training, purely from the shared
> `row_out` projection's fan-in.

At the default `h=64`: the cap is 65. This **does not bind** for `d=32`
(natural ceiling is `d=32 < 65`) or `d=64` (`d=64 < 65`, exactly at the edge
with zero slack) — so it cannot be the explanation for the observed collapse
at those two sizes. It **does bind** at `d=128` (natural ceiling `128 > 65`),
where `K` values up to 128 are tested but the architecture cannot represent
rank above 65 *no matter how well it trains*. This is a real, separate,
previously-unstated design flaw, worth fixing regardless of the d=32/64
story, but it is not sufficient by itself to explain the d=32/64 failure —
flagged clearly so Stage 0 doesn't conflate two different bugs. **Scope
caveat (attack round, MAJOR-6):** even at d=128 itself, H_cap explains at
most the cells with `K > h+1 = 65` (K=96, 128 in the tested grid); the
observed collapse at d=128 for K ≤ 64 is *not* explained by the cap and
presumably shares whatever mechanism drives d=32/64 — see §9.

### 2.2 Mechanism (b) — row-query differentiation / "coordination problem"

The `d` row-reader queries (`row_queries`, shape `(d, h)`, init
`N(0, 0.02²)`) are the model's only handle for producing `d` *distinct* rows
of `Z` from the *same* `K`-token memory via the *same* shared
`nn.MultiheadAttention` reader. As `d` grows past `h=64`, two random
`h`-dim identity vectors are increasingly likely to have high pairwise
cosine similarity purely from the `d(d−1)/2` pairs available (order-
statistics: expected max pairwise cosine ≈ `(1/√h)·√(2·ln(d(d−1)/2))`).
Back-of-envelope at `h=64`: d=16 → ≈0.39, d=32 → ≈0.44, d=64 → ≈0.49. This
is a real, testable, but **gradual** effect — it does not predict a sharp
step between d=16 (near-perfect) and d=64 (fully collapsed) on its own.
**Honest caveat, stated up front:** the static init-time argument is likely
necessary but not sufficient; if training *amplifies* rather than resolves
near-duplicate row identities (a dynamic effect the static argument can't
predict), that could produce a much sharper collapse than the init-only
number suggests. Wave 0 (§6) measures this directly — pre- and post-training
— rather than resting on the estimate.

This reframes candidate interventions #2 ("muP scaling") and #5 ("attention-
pooling readout") from generic hyperparameter guesses into targeted fixes
for a specific, derived failure mode — with one diagnostic signature,
**corrected in this revision (attack round, MINOR-9):** QR-orthogonal init
on `row_queries` removes init-time pairwise collision *exactly* at both
d=32 (`d < h`: 32 orthonormal vectors in R⁶⁴, with 32 spare dimensions) and
d=64 (`d = h`: a complete orthonormal basis of R⁶⁴). The asymmetry between
the two is **training dynamics, not init**: at d=32 the rows can drift
under gradient updates while remaining mutually near-orthogonal (spare
dimensions to move into); at d=64 the basis is complete with zero slack, so
*any* training drift necessarily re-creates pairwise overlap. The
prediction is therefore: **if orthogonal init fixes d=32 but not d=64, that
pattern is consistent with H_collision-under-drift — but it must be
confirmed by the post-training collision measurement (Wave 0/B
instrumentation, §6.2), not read off init-time math alone**, since a
zero-slack drift story and a plain capacity story make the same binary
prediction and are only separated by the measured trajectory of the
pairwise-cosine distribution during training.

### 2.3 What the existing "h=256 diverged" data point does and doesn't tell us

`TASK_D_PREREGISTRATION.md` §6 and `TASK_D_WRITEUP.md` §4.2 report that a
naively larger encoder (`h=256, n_refine=3`) diverged. **This calibration
run was at `d=16`** (the fixed operating point for that section), where the
small `h=64` model already reaches unconstrained cosine ≈0.936 — i.e., the
divergence was observed while trying to push *past* an already-working
regime's ceiling, not while trying to fix a `d≥32` failure. It is suggestive
(evidence that naive width-scaling without LR/init compensation is fragile
in this codebase) but it is **not** a direct test of whether width-scaling
helps or hurts at `d=32/64`, and Stage 0 treats it as such — a motivating
anecdote for candidate 3 below, not a substitute for actually running the
test at the right `d`.

### 2.4 Mechanism (c) — late, seed-stochastic phase transitions (new evidence, FATAL-1)

Today's Task E calibration (20K-step round 0, same `BindingEncoder`
architecture family, d=16 — see the header note in `run_task_e_sweep.py`,
lines ~48–52) observed that training can sit at loss ≈ 1.0 (the cosine
loss's maximum plateau) for **60–80% of a 20K-step budget before a sharp
transition to convergence, with a per-seed transition rate of ≤20%** —
i.e., transition onset around ~12,500 steps, well past `run_overnight.py`'s
entire d=32 step budget (8,000) and most of the d=64 budget (10,000). This
has a direct, uncomfortable implication for every d≥32 number in the
existing 991-run sweep: **a loss curve that is flat at 1.0 at step 8,000 is
consistent with a model that would have transitioned at step 12–16K**, and
with per-seed rates ≤20%, even 4 seeds of flat-at-budget-end runs cannot
strongly exclude a stochastic-transition regime (P(0/4 transitions | 20%
rate) ≈ 0.41). Revision 1 of this document had no hypothesis category for
"flat now, transitions later" — its H_dead-init signature ("flat from
step 1") would have misclassified exactly this case. Fixed throughout: a
distinct H_late-transition hypothesis (§3), extended Wave 0 budgets at ≥3
seeds (§6), and an explicit extended-budget baseline arm in the Wave A
comparison (Arm 0) rather than an inference from curve shape.

---

## 3. Diagnosed candidate mechanisms (what Wave 0 will confirm/reject)

| ID | Hypothesis | Predicts |
|---|---|---|
| H_cap | Architectural `rank(Z)≤h+1` ceiling (§2.1) | Binds only at d=128, and there only for K>65; irrelevant to d=32/64 |
| H_collision | Row-query differentiation failure (§2.2) | Post-training row_queries pairwise-cosine distribution shifts toward higher correlation as d grows past h; orthogonal init helps at d=32 more than d=64, *and* the d=64 residual failure coincides with measured drift-induced re-collision during training |
| H_dead-init | Bad init region: loss saturates near its max (cosine loss → 1.0) from step 1 and **never** moves within an *extended* (2.5×) budget across ≥3 seeds | Loss flat at ≈1.0 through 20K+ steps in every seed, no transition |
| H_undertrained | Loss/effective-rank still *trending* (visibly decreasing, not plateaued) at budget's end | Monotone-decreasing loss curve truncated by the step budget |
| H_late-transition | The write is learnable but the transition is late and seed-stochastic (§2.4: flat at ≈1.0 for most of the budget, then a sharp drop; per-seed rate possibly ≪1) | ≥1 of ≥3 extended-budget seeds transitions after the standard 8–10K budget would have ended; the standard-budget sweep's 0.0s are budget artifacts |

**Discrimination note (FATAL-1):** H_dead-init and H_late-transition are
*indistinguishable* at the standard step budget and at low seed counts —
both look like "flat at 1.0." They are separated only by (i) budgets
extended well past the observed ~12.5K-step onset (Wave 0 runs 2.5× tier
defaults: 20K steps at d=32, 25K at d=64), (ii) ≥3 seeds per primary cell,
and (iii) mid-training eval checkpoints so a transition's onset step is
measured, not inferred. Even so, 0/3 flat seeds only disfavors — does not
exclude — a ≤20% stochastic rate (P(0/3 | 20%) ≈ 0.51); this residual
ambiguity is acknowledged in §8 (attack #11) rather than hidden.

Wave 0 (§6) is designed to produce a measured verdict on each of these
five, independent of whether any Wave A/B intervention succeeds.

---

## 4. Candidate interventions, ranked

Ranked by (mechanistic groundedness × cheapness), highest first. Arm 0 and
candidates 1–4 are probed singly in Wave A (§6) before any combination is
tried, so a result can be attributed to a specific cause. Candidates 5–7
are held in a pre-registered fallback pool (not screened in Wave A) to pay
for the FATAL-1 budget expansion — the demotion rationale is given per
candidate.

**0. (Arm 0) Same architecture, longer budget.** The cheapest possible
"intervention" and, post-§2.4, a genuinely live one: run the *unmodified*
baseline at 2.5× the tier step budget. This arm is satisfied by Wave 0's
extended-budget baseline runs (§6) — they end with the same
`recovered_frac` evaluation as every other run and enter the Wave A
comparison table as an explicit arm with its own life-criterion entry, not
as a curve-shape inference. If Arm 0 alone shows life, the d≥32 "failure"
was a budget artifact, and every other intervention is re-judged on
transition *onset step* and *per-seed reliability* relative to Arm 0
(pre-registered branch, §6.1).

**1. LR warmup + decay schedule.** *Currently absent* — verified:
`run_task_d.py::train()` uses a flat `torch.optim.Adam(..., lr=lr)` with no
scheduler at all, for every d. Cheapest code change.
Cite: Goyal et al. 2017 (arXiv:1706.02677, large-batch SGD warmup); Vaswani
et al. 2017 (original Transformer warmup). **Leverage assessment corrected
this revision (MINOR-11):** the encoder *stack* is Pre-LN
(`norm_first=True`), but the write path's row-reader refinement is
`q = row_norm(q + read)` (`model_v4.py:62`) — **Post-LN** (normalization
*after* the residual add), which is precisely the configuration Xiong et
al. 2020 (arXiv:2002.04745) show has large gradients near the output at
init and *needs* warmup. Since the write path is the exact locus of the
suspected failure, warmup's a priori leverage is higher than a "Pre-LN
model, warmup optional" reading would suggest — revision 1 had this
backwards.

**2. Orthogonal init on `row_queries`, collision-aware.** Targets
H_collision directly (§2.2). Initialize `row_queries` via QR (reusing
`task_d.py::_random_directions(orthogonal=True)`'s exact machinery — cheap
code reuse, no new numerical-stability surface) instead of `N(0, 0.02²)`.
Removes init-time pairwise collision exactly at both d=32 and d=64 (§2.2,
corrected); the d=32 vs d=64 asymmetry it probes is the *zero-slack
training-drift* effect, diagnosed via the §6.2 collision instrumentation.
Cite: Saxe, McClelland & Ganguli 2014 (arXiv:1312.6120, orthogonal init for
deep linear dynamics). Cheap (init-only change, zero extra params/FLOPs).

**3. Width scaling (`h`) with muP-style compensation.** Raising `h` toward
or past `d` removes the architectural ceiling (H_cap) and is the only
screened candidate that could ever fix `d=128`. Must be paired with LR/init
compensation or it risks reproducing the known-fragile "h=256 diverged"
failure mode (§2.3) — critically, **this time tested at d=32/64, not
d=16**, closing the gap the existing calibration anecdote leaves open.
Cite: Yang & Hu 2021 (arXiv:2011.14522, feature learning / muP); Yang et
al. 2022 (arXiv:2203.03466, Tensor Programs V, zero-shot HP transfer) — muP
is formally about *hidden width*, not output dimension `d`; this is an
adapted application, flagged as such, verify exact citation before external
use per house convention. Most expensive candidate (params scale ~12h²,
quadratically — §5).
**Pre-registered attribution limit (MAJOR-4): candidate 3 is
triple-confounded, and a candidate-3-only win CANNOT be attributed to a
named mechanism within Stage 0.** (i) H_cap does not bind at d=32, so a
d=32 win cannot be the cap being lifted; (ii) raising `h` also shrinks the
H_collision estimate (the §2.2 max-pairwise-cosine scales ∝1/√h), so
candidate 3 partially targets mechanism (b) as a side effect; (iii) it is
the largest parameter increase of any screened candidate (~3.9× at
h=64→128, §5), so raw capacity is a third live explanation. If candidate 3
is the only survivor, the Stage 0 verdict is "fix found, mechanism
unresolved," and mechanism attribution requires the Stage-0.5 follow-up
(param-matched control at h=64 + the §6.2 collision instrumentation read
on the h=128 runs) before any causal claim is published.

**4. Self-attention among row-queries (competitive differentiation).**
Targets H_collision's *dynamic* form: add one self-attention layer among the
`d` row-query outputs after cross-attending to the K-token memory, so
near-duplicate row identities can differentiate via competition rather than
independently reading the same shared memory in isolation. Cite: Locatello
et al. 2020 (arXiv:2006.15055, Slot Attention) — designed for exactly this
"independent output slots collapse onto each other" failure mode via
competitive (softmax-over-slots) attention; Lee et al. 2019 (arXiv:1810.00825,
Set Transformer / PMA) for the general "pooling by multi-head attention"
lineage this architecture already partially follows. Adds a modest number of
params (one more `MultiheadAttention(h,...)` layer, d-independent cost).

---

*Fallback pool — pre-registered, not screened in Wave A (demoted this
revision to fund FATAL-1's extended budgets; deployment conditions given
in §6.1 and below):*

**5. Curriculum / output-widening warm-start from a trained d=16 solution.**
Net2net-style, but **not off-the-shelf**: standard net2net widens *hidden*
layers; here `row_queries` and `row_out` are literally sized by `d`, the
*output* dimension, so there is no natural weight-preserving map (net2net's
function-preserving trick doesn't directly apply to growing the number of
independent output slots). A bespoke scheme (duplicate+noise the trained
`d=16` row_queries rows, zero-pad `row_out`'s new output columns) is needed
and is itself an unproven sub-design. Cite: Chen, Goodfellow & Shlens 2016
(arXiv:1511.05641, Net2Net); Bengio et al. 2009 (curriculum learning,
ICML). Demoted for exactly the engineering-risk-vs-payoff reason revision 1
already ranked it low on; the FATAL-1 budget expansion forces the cut.

**6. Loss/gradient-scale normalization.** Least mechanistically motivated
(a catch-all for "maybe the d²-sized write target changes gradient
statistics in a way flat Adam doesn't handle") but cheap: d-dependent loss
scaling, or a per-parameter-group LR multiplier for `row_out`/`row_queries`
vs. the shared encoder. Loose analogy to You et al. 2017 (arXiv:1708.03888,
LARS); no direct precedent. Demoted per the attack round's own MAJOR-5
budget clause ("cut candidate 6 if budget requires"). Note: Wave 0's new
pre/post-clip gradient-norm logging (§6.2, MINOR-12) will directly measure
whether the flat `clip_grad_norm_(1.0)` interaction with d is doing
anything pathological — so candidate 6's diagnostic content is partially
obtained for free, and it gets promoted out of the pool if those logs show
a d-dependent clipping pathology.

**7. Per-row (un-shared) `row_out` — the H_cap isolator (new this revision,
MAJOR-7).** Replace the single shared `nn.Linear(h, d)` with `d`
independent per-row projections (equivalently one `(d, h, d)` einsum
weight): row `i` of `Z` becomes `W_i q_i + b_i` with its own `W_i ∈ R^{d×h}`,
removing the `rank(Z) ≤ h+1` ceiling entirely (Z can reach rank d) while
leaving `row_queries` and the reader — the entire H_collision surface —
untouched. **This is the only intervention that isolates H_cap without
perturbing H_collision.** Capacity caveat, carried explicitly: that
submodule's params go from `d(h+1)` to `d²(h+1)` — ~32× at d=32/h=64
(2,080 → 66,560), so a win still needs the capacity-vs-mechanism
disclaimer (same family as candidates 3/4). Held as a fallback/Wave-B arm
rather than a Wave A screen because H_cap provably does not bind at
d=32/64 — its designed deployments are (i) resolving the d=64 zero-slack
ambiguity in Wave B if candidate 2 shows the fixes-32-not-64 signature
(§2.2), and (ii) as the mandatory prerequisite fix for any *future* d=128
run, regardless of Stage 0's outcome (at d=128 the cap binds and no
optimizer improvement can beat it above K=65).

---

## 5. FLOPs / param / memory estimate (CLAUDE.md pre-experiment checklist)

**Param count**, derived exactly from `model_v4.py` (h=64, n_layers=3,
n_heads=4, n_refine=1 baseline; verified by hand-deriving each submodule's
parameter count from `nn.TransformerEncoderLayer`'s standard layout).
*(Formula corrected this revision — MINOR-8: revision 1's bracket summed
the d-dependent terms to `d(3h+1)`; the correct expansion of
`in_proj [h(2d+1)] + row_queries [dh] + row_out [d(h+1)]` is
`d(4h+1) + h`. The numeric values below were already computed with the
correct coefficient — 257 = 4·64+1 — and are unchanged.)*

```
N(d; h) = N_fixed(h) + d·(4h + 1)
N_fixed(h) = n_layers·(12h² + 13h) + 4h² + 7h     (encoder stack + reader + row_norm + in_proj bias)
```
At `h=64`: `N_fixed = 166,784`, so `N(d) = 166,784 + 257d`.
- d=16: **170,896** (≈171K — matches the writeup's stated figure, confirms
  the derivation).
- d=32: **174,608** (≈175K).
- d=64: **183,232** (≈183K).
- d=128: **199,680** (≈200K).

Params barely grow with `d` (linear, +257/dim) while the **target** `Z` has
`d²` independent entries — d=128 is writing a 16,384-entry matrix from a
model that grew only 17% in parameter count over d=16. This asymmetry is
the quantitative version of "the write is hard," and motivates candidate 3
(the only screened intervention that changes this ratio, since `h`-scaling
grows `N_fixed` as `~12h²`, i.e., **quadratically**, unlike everything else
here) — as well as candidate 7, whose per-row `row_out` grows the write
head itself as `d²(h+1)`.

At `h=128` (candidate 3's representative test point):
`N_fixed(128) = 661,248`, `N(d;128) = 661,248 + 513d` → d=32: **677,664** —
**3.9×** the h=64 params at the same d (consistent with the ~12h² term
quadrupling when h doubles). This 3.9× is the capacity confound
pre-registered in §4 candidate 3, and the wall-clock risk motivating the
new Wave −1 h=128 calibration cell (§6, MAJOR-3).

**FLOPs.** All configs here are tiny by modern standards — dominant cost is
`O(n_layers·(K·h² + K²·h) + d·h² + d·K·h)` per sample, i.e., low tens of
MFLOPs/sample even at the largest config tested (h=128, d=128, K=128).
Training-FLOPs estimate (`≈ 6·N_params·batch·steps`): a d=32, h=64,
batch=256, steps=20,000 (extended) run is
`≈ 6 × 175K × 256 × 20,000 ≈ 5.4×10¹²` FLOPs — still trivial for an H100
(~10¹⁵ FLOPs/s peak). **Wall-clock time here is NOT FLOP-bound** — it is
dominated by kernel-launch overhead and small-batch GPU underutilization
(consistent with DEPLOY.md's observation that a single run uses only ~65%
of one H100 and Task D packed 4 runs/GPU). The budget below therefore uses
the **measured wall-clock unit (35–55 min/run at tier-default steps)** the
user specified, scaled linearly with the step multiplier for extended runs,
per CLAUDE.md's "10 minutes on paper" step — the FLOPs check here mainly
confirms these are not accidentally compute-bound runs that would
invalidate a wall-clock-based budget. The h=128 configs are the one place
this assumption is untested (3.9× params could shift the
launch-bound/compute-bound balance), which is exactly why Wave −1 now
times one (§6).

**Memory.** d=128 state alone is `128×128×4 bytes ≈ 64KB/sample`; even at
batch=4096 this is `~260MB` for `Z` alone, and the encoder's largest tensor
(`K` memory tokens at `h=128`) is smaller still. No config here approaches
H100 80GB memory limits — this sweep is GPU-*count*-bound (many small
jobs), not GPU-*memory*-bound, matching Task D's own experience.

---

## 6. Manifest (waves, run counts, budget)

**Budget units.** Standard run (tier-default steps, 8K at d≤32 / 10K at
d=64): **0.75 h** flat (midpoint of the user-specified measured 35–55
min/run). Extended runs scale linearly with the step multiplier:
**2.5× → 1.875 h/run** (Wave 0: 20K steps at d≤32, 25K at d=64);
**2× → 1.5 h/run** (Waves A/B: 16K steps at d=32, 20K at d=64). The 2.5×
Wave 0 budget clears the observed ~12.5K-step Task E transition onset with
≥60% margin; Waves A/B run at 2× for screening economy, with the
budget-matched comparison to Arm 0 taken at the 16K-step **checkpoint** of
Wave 0's 20K-step runs (checkpointed evals are a build requirement, §6.3 —
every extended run yields metrics at both the 2× and 2.5× marks). No
packing-factor discount is assumed (conservative; Task D's harness packs
4/GPU and would come in well under — a bonus, not assumed).

| Wave | Purpose | Runs | GPU-h |
|---|---|---|---|
| **−1** | Live calibration/timing (mandatory per CLAUDE.md) — confirm the wall-time unit and reproduce the d=32/d=64 collapse on THIS harness before committing the budget. Cells: baseline h=64 @ d=32 and d=64, **plus h=128 @ d=32 (MAJOR-3 — the riskiest wall-clock config, 3.9× params, must be timed before Wave A prices candidate 3)**. Standard steps, 1 seed. | 3 | 2.25 |
| **0** | Diagnostic instrumentation (§6.2) on the **unmodified** baseline at **2.5× steps**: d=32 K=8 ×3 seeds, d=32 K=16 ×3, d=64 K=32 ×3 (primary cells, ≥3 seeds per FATAL-1); d=16 K=4 ×2 (opportunistic anomaly probe, §7 — 2 seeds, a stated deviation from the ≥3 rule, non-gating); d=16 K=8 ×1 @ standard steps (healthy-regime instrumentation reference). **These runs double as Wave A's Arm 0.** | 12 | 21.4 |
| **A** | Cheap probe @ 2× steps, single seed, single factor: candidates 1–4 × {d=32 K=8, d=32 K=16} (8 runs); +1 standard-steps d=16 sanity probe of candidate 3's compensation recipe against the known h=256/d=16 divergence (attack #8); + sequential d=64 K=32 probes for candidates alive at d=32 (budget 2). | 11 | 15.75 |
| **B** | Full-seed confirmation @ 2× steps, **top-1 Wave-A survivor** (life criterion, §6.1), d=32: M1 unconstrained K∈{4,8,16} ×2 seeds (6 runs — these also yield the free eval-time M2 truncation curve `run_task_d.py` computes on every unconstrained run); M3 train-time force-rank k∈{1, K−1, K}={1,7,8} at K=8 ×2 seeds (6 runs; K+1 corroboration comes from the free M2 curves rather than extra training runs). | 12 | 18.0 |
| **Core total** | | **38** | **57.4** |

**Core is ≤60 GPU-h as required.** Pre-registered additions beyond core,
in priority order, if budget remains after Wave B (or is freed by a Wave A
wipeout — a FALSIFY path never spends Wave B's 18 GPU-h):

1. Combo probe (mandatory before declaring FALSIFY if Wave A has zero
   survivors; §8 attack #5): top-2-by-partial-effect combined, 1–2 runs
   @2×, ≤3.0 GPU-h — funded from the unspent Wave B allocation.
2. Candidate 7 deployment (if candidate 2 shows the fixes-32-not-64
   signature): d=64 K=32 ×2 @2×, +3.0 GPU-h.
3. Wave B second intervention (if two candidates clear §6.1): +18.0 GPU-h.
4. Wave B d=64 tranche, best intervention: M3 mini-straddle k∈{1,31,32,33}
   at K=32 ×2 seeds @2×, +12.0 GPU-h.

**If core overruns** (Wave −1 reveals the 0.75h unit or the linear step
scaling is optimistic), the pre-registered cut order is: (i) Wave B's M1
K=4 cell drops to 1 seed (−1.5), (ii) Wave A's K=16 probes for the two
*lowest-ranked* surviving candidates are dropped, keeping K=8 — the
friendlier probe per §2's data (−3.0), (iii) Wave 0's d=16 K=4 opportunistic
cell drops to 1 seed (−1.875), (iv) Wave B's M3 cells drop to 1 seed each
with a 2-seed re-confirm only at the knee (−3.0). Waves −1 and 0's primary
d≥32 cells are **never** cut — they carry the diagnostic pass condition
(§1) even if no budget remains for fixes.

### 6.1 "Life" criterion (Wave A → Wave B gate)

An intervention **shows life** if its d=32 probe reaches
`recovered_frac@0.9 ≥ 0.3` at **any eval checkpoint** (not only the final
one — a sharp transition followed by any late instability should still
count), at either K=8 or K=16 — a large, qualitative jump off the observed
0.0 floor, chosen conservatively so single-seed noise cannot manufacture a
false positive (every d=32/64/128 cell in the current 991-run snapshot is
*exactly* 0.0 with zero variance — any nonzero reading is already
informative). If more than one intervention shows life, rank by best d=32
checkpoint score and take the top 1 into core Wave B (top 2 only if the
stretch budget allows) — a pre-registered cap, not a post-hoc choice.

**Pre-registered branch — if Arm 0 shows life (FATAL-1):** if the
*unmodified baseline* at 2.5× steps transitions in ≥1/3 seeds at d=32, the
d≥32 failure is (at least partly) H_late-transition/H_undertrained, and
Wave A's question changes from "which intervention enables training at
all" to "which intervention makes the transition *early and reliable*."
Interventions are then judged against Arm 0 on (i) transition onset step
(from checkpoints) and (ii) per-seed transition rate in Wave B — and the
§1 success criterion's ">0.7" is read as requiring *reliable* (per-seed
majority) convergence, not one lucky seed.

**Known false-negative risk, accepted and stated (§8 attack #1):** with
per-seed transition rates possibly ≤20% (§2.4), a single-seed Wave A probe
can miss a genuinely-working intervention. Each candidate gets 2
quasi-independent draws (K=8 and K=16 probes) rather than 1, and the
fallback pool + combo probe provide a recovery path — but Wave A remains a
screen biased toward false negatives, not a verdict. Wave B's multi-seed
confirmation is the only reliability measurement in Stage 0.

### 6.2 Wave 0 instrumentation (the diagnostic payload)

Logged per run, at every checkpoint interval (build requirement §6.3):
1. **Training loss trajectory** (dense enough to date a sharp transition:
   every ≤200 steps) — separates H_dead-init / H_undertrained /
   H_late-transition per §3's table.
2. **Effective rank + stable rank of Z** on a fixed eval batch — does rank
   *grow then collapse*, never grow, or track the loss transition?
3. **Row-query collision metrics** (H_collision): pairwise-cosine
   distribution (max / mean / fraction above 0.5) of `row_queries` AND of
   the post-reader row representations `q` — at init, every checkpoint,
   and end of training. The *trajectory* (drift toward or away from
   collision) is the §2.2 signature, not just the endpoints.
4. **Pre- and post-clip gradient norms**, per parameter group
   (`row_out` / `row_queries` / encoder stack) (MINOR-12). The training
   loop clips at a flat `clip_grad_norm_(1.0)` for every d; if raw
   gradient norms grow with the d²-entry write target, clipping silently
   scales down the effective LR at large d — a cheap, previously
   unexamined suspect that this logging measures directly (and which
   would promote candidate 6 out of the fallback pool if confirmed).
5. **`recovered_frac@0.9` + mean cosine** at every checkpoint (feeds §6.1's
   any-checkpoint life criterion and Arm 0's budget-matched 16K-step read).

### 6.3 Build requirements (carried into the build phase, blocking)

- **Manifest naming must carry an intervention/variant field (MAJOR-2).**
  `run_overnight.py::_spec()`'s run tag is
  `t{tier}_d{d}_K{K}_fr{...}_s{seed}` — two different interventions at the
  same (d, K, fr, seed) would collide on the same output path and be
  silently skipped by the validity-checked resume (`is_done`). Stage 0's
  orchestrator must extend the tag and `out_path` with a `variant` field
  (the pattern already exists in `run_task_e_sweep.py::_spec`), or use one
  out-dir per intervention. This is a blocking build requirement, not a
  nicety — a collision here invalidates resume semantics for the whole
  sweep.
- **Mid-training eval checkpoints** (every ≤2K steps: the §6.2 metrics
  written incrementally to the run's JSON/log), so (i) transitions are
  dated, (ii) Arm 0 yields budget-matched 2× reads from 2.5× runs, and
  (iii) the any-checkpoint life criterion is computable.
- **Timeouts scale with the step multiplier.** `run_overnight.py`'s
  per-d `TIMEOUT` dict assumes tier-default steps; a 2.5× run at d=64
  would be killed at 5,400s. Scale timeouts by the same multiplier plus
  margin.
- **Any architecture-touching variant re-passes the full smoke gate**
  (including blank-out) before its results are trusted — §8 attack #7.

---

## 7. Secondary, opportunistic probe: the d=16, K=4 ceiling anomaly

Not primary scope (a different `d` regime than the trainability frontier
this document targets; the pre-registered success criterion in §1 is
d=32-only, per the brief). Included because it is essentially free: Wave 0's
instrumentation is identical code whether pointed at d=16,K=4 or d=32,K=16,
so two extended-budget d=16,K=4 runs (§6, already budgeted) answer a real
open question (`TASK_D_FINDINGS_DRAFT.md` §3.2: `d=16,K=4` ceiling is
0.045–0.134 across *every* force-rank, including unconstrained —
categorically different from K=8/K=12's 0.9+ ceilings at the same `d`) at
near-zero marginal cost.

Two candidate mechanisms, now explicitly separated: (i) **the late-
transition hypothesis (§2.4) applies here too** — today's Task E
observation was made at d=16, and K=4's low ceiling at an 8K-step budget is
exactly what a ~12.5K-step stochastic transition would produce; the
extended 2.5× budget in Wave 0 tests this directly. (ii) If the extended
runs *still* plateau low, the anomaly is a genuine few-binding pathology
(fewer effective gradient terms per batch at Q=K=4, or a local minimum
specific to small target rank) — note that at `d=16 ≤ h=64`, H_collision's
estimate (§2.2) puts d=16 comfortably in the working regime, and K=8/K=12
succeed at the same d, so a "d relative to h" story cannot explain it.
Stage 0 does **not** commit to fixing this — it is reported as an observed,
measured, opportunistic finding, explicitly flagged as open and non-gating
for the go/no-go decision in §1. Caveat: with 2 seeds and a ≤20% per-seed
transition rate, a null (no transition) is weak evidence
(P(0/2 | 20%) = 0.64) — reported with that arithmetic attached.

---

## 8. Attack-yourself: ways this design could produce misleading results

1. **Single-seed Wave A noise — in BOTH directions.** False positives are
   mitigated by the conservative `≥0.3` life threshold (§6.1) against an
   empirically-zero-variance baseline. False *negatives* are now the bigger
   risk (§2.4): at a ≤20% per-seed transition rate, one seed misses a
   working intervention 80% of the time. Partial mitigations: 2 probe
   points per candidate (K=8, K=16), any-checkpoint scoring, the combo
   probe, and the fallback pool. Residual risk accepted and stated: Wave A
   is a screen biased toward false negatives; only Wave B measures
   reliability.
2. **Confounded capacity vs. mechanism.** Interventions 3, 4, and 7 add
   parameters; a "win" could be raw capacity, not the hypothesized
   mechanism. Candidate 3 is the worst case and now carries a
   pre-registered attribution limit (§4, MAJOR-4): a candidate-3-only win
   is "fix found, mechanism unresolved" pending Stage-0.5 (param-matched
   control + the §6.2 collision instrumentation). Param counts are
   reported next to every result (§5's formula makes this free).
3. **"Needs more steps / a later transition," not "needs this
   intervention."** Rewritten post-FATAL-1: this is no longer mitigated by
   curve-shape inference alone. It is addressed structurally — Wave 0 runs
   the unmodified baseline at 2.5× steps with ≥3 seeds and checkpointed
   evals, and enters the Wave A comparison as an explicit Arm 0 with its
   own life-criterion entry and a pre-registered branch (§6.1) for the
   case where longer budget alone is the fix. No Wave A intervention
   conclusion is interpretable except *relative to Arm 0 at a
   budget-matched checkpoint*.
4. **Two-K probing in Wave A can still miss K-dependence.** The d=32 data
   is non-monotone in K (K=8: 4.68 eff. rank; K=16: 2.63; K=24: 1.23);
   Wave A now probes both K=8 (the friendlier cell, MAJOR-5) and K=16, but
   an intervention that only helps at K=4 or K=24 would still screen
   falsely negative. Mitigated for survivors by Wave B's K∈{4,8,16} sweep;
   documented as a residual risk for rejected candidates, which never
   reach Wave B.
5. **Interaction effects.** Wave A tests interventions singly; a true fix
   might require a combination (e.g., orthogonal init *and* warmup) that
   neither shows alone. If Wave A produces zero survivors (including
   Arm 0), before declaring FALSIFY, run one combined probe
   (top-2-by-partial-effect combined, single seed, d=32 only, @2× — ≤3
   GPU-h, funded from the unspent Wave B allocation, §6) rather than
   stopping at "nothing worked singly." Pre-registered here so it isn't a
   post-hoc rescue.
6. **Task-D-specific fix, doesn't transfer to Task E.** Any winning
   intervention is validated only on Task D's one-hop binding task. Task
   E's compositional multi-hop readout reuses the *same* `BindingEncoder`
   unmodified, so a fix here should transfer mechanically — but this is an
   assumption, not verified in Stage 0, and must be re-checked (cheaply,
   via Task E's own smoke gate) before trusting a Stage-0 fix at Task E's
   d≥32 tranche.
7. **New architecture variants must re-pass the blank-out test.**
   Interventions 4, 5, and 7 change the encoder's internal wiring. A
   careless implementation could accidentally leak information outside the
   `Z` bottleneck (the exact failure mode Task D's blank-out test exists
   to catch, `TASK_D_PREREGISTRATION.md` §4). Mandatory (§6.3): any
   surviving intervention re-passes `run_task_d.py`'s smoke gate
   (including blank-out) before its Wave B numbers are trusted.
8. **The "h=256 diverged" anecdote is not decoupled from the new test.**
   If candidate 3 (width scaling) also diverges at d=32/64, that alone
   doesn't distinguish "this specific compensation scheme is wrong" from
   "this codebase just can't train h>64 regardless of d" (the known d=16
   case). Mitigation (§4 candidate 3, budgeted in Wave A): one extra
   sanity probe — the *same* h-scaling + compensation recipe at d=16. If
   it resolves the known d=16 divergence, the recipe is validated as sound
   before judging it on d=32/64; if it *doesn't* even fix the known case,
   that's diagnostic on its own and cheap.
9. **GPU-h unit mismatch.** The 0.75h standard unit and the
   linear-with-steps extension are assumptions; the h=128 configs
   (3.9× params) are the likeliest violators, which is why Wave −1 now
   times an h=128 cell explicitly (MAJOR-3) alongside the two baseline
   cells. If Wave −1's timing busts the assumptions, the manifest is
   re-sized per §6's pre-registered cut order *before* Wave A launches,
   not discovered mid-sweep.
10. **QR-generator conditioning at K = d — corrected (MINOR-10).**
    Revision 1 claimed Wave B's d=64, K=32 cell was "exactly K=d"; it is
    not — K=32 = d/2 at d=64. **Stage 0 never tests a true K=d cell**
    (the manifest's largest K-to-d ratios are K=16 at d=32 and K=32 at
    d=64, both d/2), so the `_random_directions` QR edge case at n=d is
    out of scope here entirely. It remains a known edge to re-check if a
    future wave adds K=d cells; the existing 991-run sweep exercised
    d=64/K=32 with no NaN/Inf reports, consistent with low risk.
11. **0/3 flat seeds does not prove dead-init.** Even Wave 0's upgraded
    design (2.5× steps, 3 seeds) only *disfavors* a ≤20% stochastic
    transition rate if all seeds stay flat (P(0/3 | 20%) ≈ 0.51). The
    H_dead-init vs H_late-transition verdict therefore leans on the
    checkpointed *curve shape and collision/grad-norm trajectories*
    (§6.2) in addition to the binary transition count, and is reported
    with its confidence arithmetic attached, not as a certainty. If the
    verdict is genuinely ambiguous after Wave 0, the honest output is
    "ambiguous, here is the posterior" — pre-registered as an acceptable
    diagnostic outcome rather than a failure to be papered over.

---

## 9. What Stage 0 does and does NOT show

- **Shows, if CONFIRM:** a specific, falsifiable mechanistic account of why
  the matrix-write fails to train at d≥32 (not just "capacity/optimization
  limitation"), and — if an intervention crosses the bar — a validated fix
  that unblocks Task E's d≥32 tranche and the ICLR submission's need for an
  *explained* trainability frontier rather than a reported one.
- **Shows, if FALSIFY:** a decisive, budget-bounded negative — the screened
  interventions plus the pre-registered combo probe, all
  literature-grounded and mechanistically motivated, were tried and failed
  at this parameter/step budget — that legitimately scopes Task E to d=16
  and redirects compute toward task-complexity scaling. Both outcomes are
  decisive per §1.
- **Does NOT show:** that any winning fix transfers to Task E's multi-hop
  readout (attack 6, §8) or to real data — both are explicitly out of
  scope, sequenced after Stage 0 exactly as Task D → Task E was sequenced.
- **Does NOT show:** anything about whether rank is *used* once the model
  trains at d≥32 (that's Task D's own M1/M3 question, already gated on
  training succeeding at all — Stage 0 only targets "does it train").
- **Does NOT explain d=128, even in the best case (MAJOR-6).** H_cap
  accounts only for the d=128 cells with `K > h+1 = 65` (K=96, 128 in the
  tested grid); the observed collapse at d=128 for K ≤ 64 is *not*
  explained by the cap and presumably shares the d=32/64 mechanism —
  whatever Stage 0 concludes about d=32/64 is a *hypothesis* about
  d=128's K≤64 cells, not a finding, since d=128 is outside Stage 0's
  manifest. Any future d=128 work additionally requires the candidate-7
  fix (or h≥d) as a hard prerequisite, since above K=65 no optimizer
  improvement can beat the architectural ceiling.

---

## 10. Sequencing — what this unlocks

1. **Stage 0 (this doc)** — go/no-go on d≥32 trainability, with a
   mechanistic explanation regardless of outcome.
2. **If CONFIRM →** Task E's Stage 0-conditional d≥32/64 tranche
   (`NEXT_EXPERIMENT_DESIGN.md` §8, "conditional on Stage 0 unlocking d≥32")
   runs using the winning intervention; re-verify the fix survives Task E's
   own smoke gate (attack 6, §8) before trusting it there. If the win came
   from candidate 3 alone, Stage-0.5 (mechanism attribution, §4) runs
   before any published causal claim.
3. **If FALSIFY →** Task E proceeds at d=16 only (already provisioned as
   the fallback in `NEXT_EXPERIMENT_DESIGN.md` §8); the ICLR writeup
   reports the d≥32 negative as an honest, explained limitation (§9) rather
   than an unexplained one — itself a meaningful upgrade to the
   submission's rigor even without a fix.

---

## 11. Reproducibility pointers

- This design: `matrix-thinking/chapter2/STAGE0_DESIGN.md` (revision 2;
  revision-1 → revision-2 changes indexed in the header changelog against
  the attack round's finding IDs).
- Builds on (all reused, not modified, until Stage 0's build phase):
  `model_v4.py` (`BindingEncoder`, the object under diagnosis),
  `task_d.py` (generator, including the orthogonal-init machinery candidate
  2 reuses), `rank_utils.py` (effective/stable rank), `run_task_d.py`
  (train/eval/smoke — the harness Wave 0's instrumentation hooks into),
  `run_overnight.py` (orchestrator pattern Stage 0's manifest reuses, with
  the §6.3 variant-naming fix as a blocking build requirement), and
  `run_task_e_sweep.py::_spec` (the variant-field naming pattern to copy).
- Prior data this design is grounded in:
  `results/overnight_snapshots/AGGREGATE_latest.json` (991 runs, pulled this
  session for §2's table and §6.1's zero-variance baseline claim),
  `TASK_D_WRITEUP.md` §5.3, `TASK_D_FINDINGS_DRAFT.md` §3.2/§4,
  `TASK_D_PREREGISTRATION.md` §6 (the h=256/d=16 calibration note, §2.3),
  and `run_task_e_sweep.py`'s header calibration note (the late-transition
  observation behind §2.4/H_late-transition).
- Sibling design this precedes: `NEXT_EXPERIMENT_DESIGN.md` (Task E),
  whose §8 Stage 0 entry this document supersedes with a fully worked-out
  version — same bounded/gated philosophy, now with a mechanistic diagnosis
  behind the intervention list instead of a generic HP-search description.
- Next: build Wave 0's instrumentation (logging only, no architecture
  change) + the §6.3 orchestrator requirements → audit by a fresh-context
  agent (Build → Audit → Run, per `CLAUDE.md`) → Wave −1 calibration on the
  cluster → Waves 0/A/B per §6.

---

## 12. Results — Wave −1/0 (2026-07-02)

**Status: Wave −1 and Wave 0 COMPLETE (15/15 runs, 0 failed); Wave A
LAUNCHED (11 runs, running, not yet analyzed — deferred to a future
addendum).** Raw data: `experiment-runs/2026-07-02_stage0_waves/{wave-1,wave0}/`
(per-run JSON with dense checkpoint trajectories, `orchestrator.log`,
`SUMMARY.txt`/`PROGRESS.txt` per wave). All numbers below are pulled
directly from the per-run JSONs, not the `SUMMARY.txt` per-cell rollups,
which average over seeds and obscure the per-seed pattern that is the
point of this wave.

### 12.1 Wave −1 — timing calibration (§6, MAJOR-3)

| Cell | steps | wall_s | wall (min) | n_params |
|---|---|---|---|---|
| d=32, K=8, h=64 | 8,000 | 608.5 | 10.1 | 175,008 |
| d=32, K=8, h=128 | 8,000 | 602.9 | 10.0 | 677,664 |
| d=64, K=32, h=64 | 10,000 | 1,022.2 | 17.0 | 183,232 |

**Verdict: the "0.75h flat unit" budget assumption (§6) was conservative,
not violated.** h=128 (3.9× params, the riskiest cell per MAJOR-3) shows
*no* wall-clock penalty relative to h=64 at the same d/steps (602.9s vs
608.5s) — confirming §5's FLOPs argument that these runs are
launch/underutilization-bound, not compute-bound, even at the largest
screened width. No pre-launch re-sizing of the manifest (§9 attack #9's
contingency) was triggered.

*(Side note, not gating: `n_params=175,008` measured for d=32 matches the
design's own formula `N(d)=166,784+257d` exactly (166,784+257·32=175,008).
§5's printed table row for d=32, `174,608`, was an arithmetic slip in that
table — the formula and the code are both correct; d=16 (170,896) and
d=64 (183,232) match the printed table exactly.)*

### 12.2 Wave 0 — primary cells: the late-transition signature at d=32 (H_late-transition, §2.4/§3)

All 6 unmodified-baseline runs at d=32 (K=8 and K=16, 3 seeds each, 20K
steps = 2.5× Task D's 8K budget) show the same qualitative shape:
effective rank flat near 1.0 for 6,000–10,000 steps, then a sharp climb.

| Run | er@6K | er@8K | er@10K | er@12K | er final (20K) | cos final | recovered_frac@0.9 final |
|---|---|---|---|---|---|---|---|
| d32_K8_s0 | 1.018 | 1.589 | 8.850 | 12.242 | 8.775 | 0.841 | 0.1355 |
| d32_K8_s1 | 1.017 | 1.013 | 1.014 | 1.016 | 10.238 | 0.698 | 0.0006 |
| d32_K8_s2 | 1.022 | 1.028 | 2.259 | 8.546 | 8.977 | 0.788 | 0.0186 |
| d32_K16_s0 | 1.022 | 1.015 | 1.011 | 1.233 | 15.282 | 0.684 | 0.0005 |
| d32_K16_s1 | 1.860 | 13.922 | 17.422 | 17.753 | 16.832 | 0.771 | 0.0172 |
| d32_K16_s2 | 1.016 | 1.013 | 2.040 | 11.105 | 14.589 | 0.652 | 0.0001 |

Transition onset varies seed-to-seed (as early as 6–8K for K16_s1, as late
as 12–16K for K8_s1) but every one of the 6 seeds transitions somewhere in
the 6–16K window — none stay flat through 20K. **This directly falsifies
"the 8K-budget snapshot is a converged state"**: every run in this table
was flat or barely-moving at the exact step (8,000) Task D's original
sweep stopped at.

**But transitioned is not solved.** Final `recovered_frac@0.9` ranges
0.0001–0.1355 (min: K16_s2; max: K8_s0) — none approach the pre-registered
CONFIRM bar (`>0.7`, §1) or even the Wave A screening "life" bar (`≥0.3`,
§6.1). Final mean_cos ranges 0.652–0.841. K=8 seeds are still visibly
climbing at 20K (e.g. s0: `recovered_frac` 0.033→0.117→0.135 over the last
three checkpoints) — an H_undertrained signature layered on top of the
late transition, not a plateau.

### 12.3 d=64: mixed, not uniformly flat

3 seeds, 25,000 steps (2.5× the 10K budget):

| Run | er@18K | er@20K | er@22K | er final (25K) | cos final | recovered_frac@0.9 final |
|---|---|---|---|---|---|---|
| d64_K32_s0 | 1.011 | 1.010 | 1.007 | 1.006 | 0.0003 | 0.0 |
| d64_K32_s1 | 1.008 | 1.701 | 2.238 | 5.112 | 0.0853 | 0.0 |
| d64_K32_s2 | 1.017 | 1.577 | 2.550 | 6.021 | 0.1066 | 0.0 |

s0 is genuinely flat throughout (eff rank 1.006–1.075 across every one of
13 checkpoints, cos never exceeds 0.0025). s1 and s2 both begin a rank
climb at step 20,000 — 80% of budget — that mirrors the *shape* of the
d=32 transitions above but is truncated by the 25K cutoff before
`recovered_frac` moves off 0.0. Onset step therefore reads: d=16 < 2,000
(already at cos=0.48 by the first checkpoint); d=32: 6,000–16,000; d=64:
≥20,000, incomplete at 25,000 — consistent with onset scaling
superlinearly in `d`, itself worth tracking as future cells are run.

### 12.4 Row-query collision (H_collision, §2.2/§6.2): tracks the transition, causal test pending

d=32 baseline (no orthogonal init): `row_queries` pairwise-cosine mean
starts near-zero-to-slightly-negative at 2K (−0.0006 to −0.0107 across
the 6 seeds) and drifts more negative by 20K (−0.0216 to −0.0282) in
**every** seed — largest drift in `d32_K8_s0` (−0.0006 → −0.0282), the
seed with the earliest, cleanest transition.

d=64: none of the three seeds shows the same sustained, large-magnitude
negative drift. The flat seed (s0) drifts from +0.0015 (2K) to about
−0.0066 by 14K, then *plateaus* (no further increase in magnitude)
through 25K (final −0.0045) — small and self-limiting, not the ongoing
differentiation seen at d=32. Of the two onset seeds, s2's drift continues
in the negative direction coincident with its own rank rise (−0.0066 at
18K → −0.0087 at 25K, alongside eff rank 1.02→6.02); s1's is noisier and
partially reverses sign during its onset (−0.0064 at 20K → +0.0033 at
25K, alongside eff rank 1.70→5.11) — a weaker, more ambiguous version of
the same coincident pattern, not a clean confirmation.

This matches H_collision's predicted *trajectory* signature (§2.2: "drift
toward or away from collision," not just endpoints) at d=32, and a
partial/noisier version of it at d=64. It is **correlational, not yet
causal**: the orthogonal-init arm that tests causality (candidate 2) is
in Wave A, running, not analyzed as of this addendum.

### 12.5 d=16, K=4 opportunistic probe (§7)

2 seeds, 20,000 steps:

| Run | er@4K | er final | cos final | recovered_frac@0.9 final |
|---|---|---|---|---|
| d16_K4_s0 | 7.767 | 4.449 | 0.8798 | 0.3479 |
| d16_K4_s1 | 7.276 | 4.976 | 0.8864 | 0.4827 |

Both seeds show the "overshoot-then-compress" shape flagged in the
pre-registration: effective rank spikes to 7.3–7.8 at step 4,000
(mid-transition, above the K=4 target) then compresses to settle at
4.4–5.0 by 20,000. `recovered_frac@0.9` = 0.348/0.483 is a large jump off
the mega-replication's 0.0008–0.134 (full 8K-step budget, every
force-rank) ceiling for this cell — see `TASK_D_WRITEUP.md` §5.2 for the
correction this implies.

### 12.6 Hypothesis verdicts (§3 table) — results against pre-registration

| Hypothesis | Predicted signature (§3) | Verdict at d=32 | Verdict at d=64 |
|---|---|---|---|
| H_cap | Binds only at d=128, K>65 | N/A — not tested this wave (no d=128 cell); derivation (§2.1) unchanged | N/A |
| H_dead-init | Flat ≈1.0 through 20K+ in every seed, no transition | **REJECTED** — 6/6 seeds transition | **Disfavored, not resolved** — 1/3 (s0) matches the signature exactly; 2/3 (s1, s2) show late-onset movement inconsistent with true dead-init |
| H_late-transition | Flat for most of budget then a sharp drop, per-seed stochastic | **CONFIRMED, dominant** — 6/6 seeds show the flat-then-climb shape, onset 6–16K | **Consistent, incomplete** — 2/3 seeds show the same shape beginning ~20K, truncated by the 25K cutoff before completion |
| H_undertrained | Loss/rank still trending, truncated at budget's end | **Also live, layered on the transition** — K=8 seeds' `recovered_frac` still climbing at 20K | Cannot assess — onset seeds never approach a plateau within budget |
| H_collision | Pairwise-cosine trajectory shifts toward less collision as training proceeds, tracking the transition | **Supported (correlational)** — collision drift co-occurs with rank rise in all 6 seeds | **Supported (correlational), weaker** — present (one clean, one noisy) in the 2 onset seeds, self-limiting in the 1 flat seed |

**Diagnostic pass condition (§1) — met regardless of CONFIRM/FALSIFY on any
fix:** Wave 0 produced a measured, falsifiable verdict on each hypothesis
above (a mix of CONFIRMED / REJECTED / supported-but-not-yet-causal), not
an assertion. This satisfies §1's stated bar for Stage 0 as a diagnostic.

### 12.7 §1 CONFIRM/FALSIFY status and the §6.1 branch

Neither CONFIRM nor FALSIFY is reached yet — Wave 0 (Arm 0) does not, by
itself, cross the pre-registered `recovered_frac@0.9 > 0.7` bar at d=32
(max observed: 0.1355). It also does not cross the stricter Wave A
**screening** "life" threshold as originally defined (`≥0.3` at any
checkpoint, §6.1's first-stated criterion) — same 0.1355 ceiling.

It **does** trigger the **separately-defined pre-registered branch** in
§6.1 ("if Arm 0 shows life... transitions in ≥1/3 seeds at d=32"), which
uses a looser, curve-shape definition of "life" than the general `≥0.3`
bar defined earlier in the same subsection. This is an internal ambiguity
in the design's own §6.1 text (two different operational meanings of
"shows life" in the same subsection) that Wave 0's data forced a
resolution of: 6/6 seeds transition (far past the ≥1/3 threshold), so the
branch fires. Per its own text, this reframes Wave A's question from
"which intervention enables training at all" to **"which intervention
makes the transition early and reliable"** — the framing already adopted
in launching Wave A (§12.8).

### 12.8 Decision: Wave A launch (2026-07-02)

Wave A launched with 11 runs: candidates 1–4 (LR warmup, orthogonal init,
muP width, self-attention differentiation) × {d=32 K=8, d=32 K=16}, +1
d=16 sanity probe for candidate 3's compensation recipe (attack #8), +2
sequential d=64 K=32 probes reserved for "candidates alive at d=32" (§6's
Wave A row).

**Documented deviation:** the 2 d=64 probe slots were assigned to
`c2_orthogonal` and `c3_mup` at launch time, before the d=32 K=8/K=16
screen's own results were available. The design's intent (§6, "sequential
d=64 K=32 probes for candidates alive at d=32") presumes the d=64 probes
are chosen *after* seeing which d=32 candidates show life; the
orchestrator's manifest format requires the complete run list up front, so
a post-screen selection isn't mechanically available without a second
launch. The choice was made pre-data but mechanism-matched: candidate 2
(orthogonal init) directly targets the row-query-collision rate factor
measured in §12.4; candidate 3 (muP) targets width-driven scale transfer,
the mechanism most distinct from the others. If the d=32 screen (once
analyzed) contradicts this choice — i.e. a different candidate is the
actual d=32 survivor — the 2 d=64 probes are rerun against the real
survivor(s); at ~2 runs this is cheap relative to the wasted-compute risk
of waiting on a second launch cycle.

### 12.9 Compute

Wave −1: 3 runs, 608.5+602.9+1,022.2s = 2,233.6s = **0.62 GPU-h** actual
(serial-sum accounting) against 2.25 GPU-h budgeted. Wave 0: 12 runs,
summed wall_s = 13,501.2s = **3.75 GPU-h** actual (same accounting)
against 21.4 GPU-h budgeted. The orchestrator ran both waves at 4
runs/GPU in practice (`orchestrator.log`: `slots=16 (8x2)` for Wave −1,
`slots=32 (8x4)` for Wave 0), so true wall-clock-to-completion was well
under even the 4.37 GPU-h serial-sum figure. The flat-rate budget unit
(§6) was conservative by roughly 5–6× at these cell sizes — confirming,
not merely assuming, the packing bonus §6 flagged but did not quantify.
This leaves substantial headroom under the pre-registered 60 GPU-h core
budget for Wave A/B.

---

## 13. Results — Wave A (intervention screen) + extended-budget arm (2026-07-02)

**Status: Wave A COMPLETE (11/11 runs, 0 failed); a second, non-pre-registered
arm ("ext_budget," 17/17 runs, 0 failed) ran in parallel on the box to
directly test Arm 0 (unmodified baseline) at 2.5–6× the Wave A step count,
since Wave 0/A already showed transitions were budget-gated. A 100K-step
follow-up ("probe_100k," 5 runs) launched 2026-07-02 ~20:00 UTC and is
running, not yet analyzed.** Raw data: Wave A —
`experiment-runs/2026-07-02_stage0_waves/waveA/*.json`; ext_budget —
`/home/nvidia/chapter2/results/stage0/ext_budget/*.json` on the box
(17 runs; an archive copy had not fully landed locally as of this
addendum — only 1/17 was present under
`experiment-runs/2026-07-02_stage0_waves/ext_budget/` at write time). All
numbers below are pulled directly from the per-run JSONs (`trajectory`,
`checkpoints`, and top-level summary fields), not `SUMMARY.txt`/
`AGGREGATE.json` roll-ups, for the same reason §12 gives: per-seed
resolution is the point.

**One naming note, confirmed by inspection, worth flagging so it isn't
misread downstream:** every Stage 0 run JSON (Wave 0, Wave A, ext_budget
alike) carries a top-level `orthogonal: true` field. This is **not** the
candidate-2 row-query-init flag — it is the task-data generation setting
(orthogonal ground-truth keys, `task_d.py::_random_directions`), held fixed
across every Stage 0 run per the design's own scope. The actual candidate-2
signal is `init_row_query_collision` (`row_queries`' pairwise-cosine stats
*at initialization*): `max≈0.35, mean≈0.001` for c1/c3/c4 (standard
`N(0,0.02²)` init, matching §2.2's order-statistics estimate) versus
`max≈9e-8, mean≈-2e-9` for c2_orthogonal (QR-orthogonal init, effectively
exact zero collision at init, as designed). Verified per-run before trusting
any candidate-2 comparison below.

### 13.1 Wave A — intervention screen, 16K steps (20K for the d=64 riders), d=32 K∈{8,16}

| Candidate | Cell | final eff.rank | final mean_cos | final rec@0.9 | max rec@0.9 (any ckpt) | Wave A "life" (≥0.3, §6.1)? |
|---|---|---|---|---|---|---|
| c1_warmup | d32 K8 | 1.532 | 0.039 | 0.0 | 0.0 | No |
| c1_warmup | d32 K16 | 1.103 | 0.016 | 0.0 | 0.0 | No |
| c2_orthogonal | d32 K8 | 11.174 | 0.713 | 0.0020 | 0.0020 | No |
| c2_orthogonal | d32 K16 | 16.103 | 0.658 | 0.0002 | 0.0002 | No |
| c2_orthogonal | d64 K32 (20K) | 2.361 | 0.023 | 0.0 | 0.0 | No |
| c3_mup | d16 K8 sanity (8K, h=128) | 7.627 | 0.942 | 0.948 | 1.000 | Yes (known-good d16 regime — not a d32 screen cell) |
| c3_mup | d32 K8 (h=128) | 1.001 | 0.004 | 0.0 | 0.0 | No |
| c3_mup | d32 K16 (h=128) | 1.001 | −0.002 | 0.0 | 0.0 | No |
| c3_mup | d64 K32 (20K, h=128) | 2.536 | 0.027 | 0.0 | 0.0 | No |
| c4_selfattn | d32 K8 | 1.002 | 0.001 | 0.0 | 0.0 | No |
| c4_selfattn | d32 K16 | 1.002 | 0.0001 | 0.0 | 0.0 | No |

**c1_warmup (1,600-step linear warmup) kills the transition.** Both cells
stay near the loss plateau through all 16K steps (loss 0.966–0.978 at
final step, vs. c2_orthogonal's 0.29–0.34); effective rank never exceeds
1.53. This is the opposite of a null result — warmup measurably *delays*
whatever mechanism drives the plateau escape, consistent with the write
path's Post-LN configuration (§2's MINOR-11 leverage note) making warmup
load-bearing in the wrong direction here. Single seed; not multi-seed
confirmed.

**c2_orthogonal (QR-orthogonal `row_queries` init) is the only Wave A
candidate that transitions within budget — but at parity with, not ahead
of, the unmodified Wave 0 baseline.** Dense checkpoints pin onset (first
checkpoint where effective rank climbs off the ≈1.0–1.1 floor) at **K8:
step 10,000** (er 1.128 @ 8K → 5.989 @ 10K) and **K16: step 8,000** (er
1.048 @ 6K → 6.923 @ 8K), landing final mean_cos 0.713 (K8) / 0.658 (K16).
Wave 0's unmodified-baseline onsets at the same d=32 (§12.2, standard
init) span **6–16K** across 6 seeds — c2_orthogonal's onsets fall
squarely inside that range, not earlier than it. Zero-collision init does
not accelerate the transition in this data; it only reproduces it.

**c3_mup (h=128, muP-scaled LR) is completely flat at d=32** — effective
rank pinned at 1.001 in both K8 and K16 cells, loss never leaves the
plateau — **despite its own d=16 sanity probe passing cleanly**
(rec@0.9=0.948, cos=0.942, matching the known-good d=16/h=64 regime's
convergence shape). The compensation recipe is validated as sound at the
size it was designed for (§8 attack #8's sanity check does its job) and
still fails completely at d=32; the pre-registered attribution caveat (§4,
MAJOR-4) is moot here since there is no win to attribute. The most likely
reading, not directly measured this wave: the muP-lowered `row_out`
group's effective LR (the recipe deliberately shrinks it relative to
`row_queries`/encoder) works against whatever optimization step the
plateau escape requires — worth a pre/post-clip grad-norm read
(§6.2 instrumentation, already logged in these JSONs) if this candidate is
revisited.

**c4_selfattn (self-attention among row-queries) is dead** — effective
rank 1.002 in both cells, cosine ≈0, indistinguishable from random-guess
baseline. No signal that competitive differentiation is even beginning.

**Verdict: 0/10 screened d32/d64-probe cells cross the pre-registered
`≥0.3` Wave A "life" bar (§6.1), checked against every checkpoint, not
just the final one** (only the out-of-scope d16 sanity cross-check does,
as expected). No single-factor intervention beats the unmodified baseline
at matched budget.

### 13.2 Extended-budget arm (ext_budget) — d=32 @ 40,000 steps (5× Task D's original 8K, 2.5× Wave A's 16K)

3 seeds × K∈{4,8,16} baseline (Arm 0, unmodified) + 2-seed c2_orthogonal
riders at K∈{8,16}. Not pre-registered as a named wave in §6, but a direct,
mechanism-matched follow-on to §12's headline (the wall is a budget
artifact) and to Wave A's finding that no intervention beat baseline —
the natural next question is "how far does Arm 0 alone get with more
budget."

| Variant | K | seed | final eff.rank | final mean_cos | final rec@0.9 | max rec@0.9 (any ckpt) | life (≥0.3)? |
|---|---|---|---|---|---|---|---|
| baseline | 4 | 0 | 3.911 | 0.8202 | 0.0896 | 0.0896 | No |
| baseline | 4 | 1 | 4.121 | 0.8387 | 0.0127 | 0.0127 | No |
| baseline | 4 | 2 | 3.739 | 0.8605 | 0.0972 | 0.0972 | No |
| baseline | 8 | 0 | 7.939 | 0.8855 | 0.4454 | 0.4454 | **Yes** |
| baseline | 8 | 1 | 7.572 | 0.8612 | 0.2899 | 0.2899 | No (close) |
| baseline | 8 | 2 | 8.241 | 0.8839 | 0.4048 | 0.4048 | **Yes** |
| baseline | 16 | 0 | 14.562 | 0.8043 | 0.0551 | 0.0551 | No |
| baseline | 16 | 1 | 16.145 | 0.8488 | 0.2000 | 0.2000 | No |
| baseline | 16 | 2 | 14.459 | 0.6989 | 0.0017 | 0.0017 | No |
| c2_orthogonal | 8 | 0 | 8.078 | 0.8245 | 0.0891 | 0.0945 | No |
| c2_orthogonal | 8 | 1 | 8.362 | 0.8245 | 0.1027 | 0.1027 | No |
| c2_orthogonal | 16 | 0 | 14.298 | 0.6957 | 0.0012 | 0.0016 | No |
| c2_orthogonal | 16 | 1 | 15.014 | 0.7632 | 0.0138 | 0.0138 | No |

**Rank-tracks-K is restored at d=32 once budget suffices** — effective
rank settles near its target `K` in every cell (K=4: 3.7–4.1; K=8:
7.6–8.2; K=16: 14.3–16.1 across baseline+c2 riders), reproducing
`TASK_D_WRITEUP.md`'s M1 pattern (rank recruits to match the task's key
count) that the 8K-step sweep's mid-transition snapshots (§12.1's
"erratic, non-monotone" numbers) had obscured. mean_cos across all 13
d=32 cells ranges 0.696–0.886.

**But `recovered_frac@0.9` — the pre-registered pass metric — is still
well short of §1's `>0.7` CONFIRM bar everywhere, ranging 0.0012–0.4454.**
**Budget alone crosses Wave A's own `≥0.3` life bar in 2/3 K=8 baseline
seeds** (s0=0.4454, s2=0.4048; s1=0.2899 falls just short) — something
*zero* of the 10 Wave A single-factor-intervention cells did at any
checkpoint (§13.1). This is the clearest evidence in Stage 0 so far for
"step budget is the dominant variable": unmodified baseline at 2.5× more
steps than Wave A's screen beats every screened intervention.

**The trajectories are still climbing at budget end, not plateaued** —
e.g. `d32_K8_s0`: rec@0.9 = 0.338 (34K) → 0.430 (36K) → 0.445 (40K);
`d32_K8_s2`: rec@0.9 = 0.181 (38K) → 0.405 (40K), a jump inside the final
2,000 steps. Neither the CONFIRM bar nor a plateau has been reached by
40K — the honest read is "still improving, verdict undetermined," not
"converged short of the bar."

**c2_orthogonal riders do not beat, and on this data modestly
*underperform*, unmodified baseline at matched 40K budget** — K8:
c2's rec@0.9 range (0.089–0.103) sits below both of baseline's converged
seeds (0.405, 0.445) and its weaker one (0.290); K16: c2's range
(0.001–0.014) sits at or below baseline's own range (0.002–0.200).
Effective rank and mean_cos are comparable between arms (not a collapse),
so this reads as "orthogonal init doesn't help, and may cost a little
recovered_frac," not as a second dead candidate — but it is a second,
independent piece of evidence (after §13.1's onset-parity finding) against
H_collision's *row-query init* as the operative bottleneck at d=32,
now measured at extended budget rather than only at the 16K screen.

**K=4 anomaly (§7) persists in `recovered_frac`, not in cosine, at 40K.**
mean_cos 0.820–0.860 (healthy, comparable to K=8/K=16) but
`recovered_frac@0.9` only 0.013–0.097 — the same qualitative split
`TASK_D_FINDINGS_DRAFT.md` §3.2 first flagged, now confirmed to survive a
5× budget extension past the original 8K sweep. This is not a training-
budget artifact; whatever depresses K=4's exact-recovery rate while
leaving its cosine fidelity intact is a different, still-open mechanism
(§7's two candidate explanations remain live).

### 13.3 Extended-budget arm — d=64 @ 60,000 steps (6× the original 10K budget)

3 baseline seeds + 1 c2_orthogonal rider, K=32 (the tested cell throughout
Stage 0). Onset defined as the first checkpoint where effective rank
exceeds 3 (≈3× the ≈1.0–1.1 flat floor observed in every run before its
own transition) — a threshold chosen after inspecting the raw
checkpoints, not fit to the answer; the qualitative "flat-then-climb"
shape is unambiguous in every one of these 4 runs regardless of exact
threshold.

| Variant | seed | onset step | final eff.rank | final mean_cos | final rec@0.9 |
|---|---|---|---|---|---|
| baseline | 0 | 44,000 | 10.485 | 0.3356 | 0.0 |
| baseline | 1 | 24,000 | 16.864 | 0.4088 | 0.0 |
| baseline | 2 | 24,000 | 15.690 | 0.3929 | 0.0 |
| c2_orthogonal | 0 | 22,000 | 17.379 | 0.4290 | 0.0 |

**All 4 seeds transition within 60K steps — zero stay flat, unlike the
25K-step Wave 0 read where 1/3 baseline seeds (s0) never moved (§12.3).**
Onset ranges 22,000–44,000 (24K–44K for the 3 unmodified-baseline seeds);
final effective rank 10.5–17.4 of the ideal K=32, still well under target;
mean_cos 0.336–0.429 and **still rising at the 60K cutoff in every seed**
(e.g. baseline s0: 0.314 @58K → 0.336 @60K). `recovered_frac@0.9` is 0.0
in all 4 — the exact-recovery metric has not begun to move at d=64 even
where the transition itself is now clearly underway. This is deep, slow
convergence, not non-convergence: onset now confirmed to scale
superlinearly in `d` (d=16: <2K; d=32: 6–16K; d=64: 22–44K), and even the
earliest-onset d=64 seed (c2_orthogonal, 22K) is far from `recovered_frac`
movement 38K steps later. **The single c2_orthogonal seed transitions
earliest and reaches the best final numbers of the 4** — weak (n=1),
directionally consistent with the "fixes-64-more-than-32"
zero-slack-drift story floated in §2.2, but nowhere near strong enough to
act on without replication.

### 13.4 §1 CONFIRM/FALSIFY status, updated

Still neither. The best `recovered_frac@0.9` observed anywhere in Stage 0
to date (Wave 0 + Wave A + ext_budget combined) is **0.4454**
(`d32_K8_s0`, ext_budget, 40K steps) — well under the pre-registered
`>0.7` CONFIRM bar, and still climbing at the point measurement stopped.
FALSIFY is equally unwarranted: nothing has plateaued below the bar with
budget exhausted — every d=32 K=8/K=16 cell and every d=64 cell that has
transitioned is still moving at its respective cutoff. The diagnostic
pass condition (§1) remains met independent of this ambiguity (Wave 0 +
this wave together produce measured, falsifiable verdicts on every named
hypothesis — §13.1–13.3 above extend, not replace, §12.6's table: add
H_late-transition **CONFIRMED at d=64 too** now that all 4 d=64 runs in
this arm transition, and H_collision (row-query init specifically)
**DISFAVORED as an accelerant at d=32** by the two independent
onset-parity/underperformance findings in §13.1/13.2, though still open
at d=64 on n=1).

### 13.5 Decision: Wave B is moot; a 100K-step probe replaces it (pre-registration deviation, documented)

§6's Wave B was designed to run full-seed confirmation on "the top-1
Wave-A survivor" (§6.1). **Wave A produced zero survivors of its own
life criterion** (§13.1) — there is no intervention to confirm. Per §8
attack #5's pre-registered fallback, a combo probe would be the next
step if Wave A whiffed entirely; but §13.2's ext_budget data changes the
calculus the combo probe was designed for, because it already shows
*unmodified baseline* beating every screened intervention once given
2.5× the screening budget, with the trajectory still climbing at
40K/60K cutoffs. Running Wave B on a "winning" intervention that isn't
actually ahead of plain budget would not answer the live question, which
is now: **does the d=32 write cross `recovered_frac@0.9 > 0.7` with
enough budget, or does it plateau below it?** — a question about Arm 0's
asymptote, not about any candidate 1–4.

**Deviation from §6, documented per CLAUDE.md convention:** Wave B (full-
seed intervention confirmation) is deferred/skipped; a **100K-step probe**
(2.5× ext_budget's own 40K, 10× Task D's original budget) launched
2026-07-02 ~20:00 UTC instead — 5 runs, `variant=baseline`, d=32:
K=8 ×3 seeds (the cell closest to the bar, per §13.2), K=4 ×1, K=16 ×1
(single-seed opportunistic reads on the other two cells, budget-
conscious given K=8's priority). Running under `tmux` session
`probe100k` on the box, output to
`/home/nvidia/chapter2/results/stage0/probe_100k/`; as of this addendum
the 5 runs have completed 10,000–12,000 of their 100,000 steps
(~10–12%), not yet analyzable. This directly settles §13.4's open
CONFIRM/FALSIFY question for the highest-priority cell before any further
compute is committed to a Wave-B-style intervention confirmation that
Wave A no longer motivates.

### 13.6 Compute

Wave A: 11 runs, summed `wall_s` = 9,447.2s = **2.624 GPU-h** actual
(serial-sum accounting, consistent with §12.9's method). ext_budget: 17
runs, summed `wall_s` = 60,156.4s = **16.710 GPU-h** actual. Combined this
addendum: **19.33 GPU-h** serial-sum (Wave A + ext_budget), on top of
Wave −1/0's 4.37 GPU-h (§12.9) — **23.70 GPU-h serial-sum total for
Stage 0 to date**, against the pre-registered 60 GPU-h core budget (§6);
substantial headroom remains even before accounting for the packing bonus
(§12.9 measured Wave 0 packing at ~4 runs/GPU in practice). probe_100k's 5
runs (running, not yet complete) are additional, not yet reflected in this
sum. **Separately, the project's cumulative Brev grant burn as of this
addendum is reported at ~295 of the ~1,600 GPU-h grant** (a box-uptime/
billing figure tracked outside these run JSONs, per the same
compute-budget-audit convention as `EXPERIMENT_LOG.md`'s 2026-07-01 21:30
UTC check — not independently re-derivable from Stage 0's own run-level
`wall_s` sums alone, since it includes idle uptime and other concurrent
work on the box; recorded here for the budget trail, not re-verified line
by line).

---

## 14. Results — 100K probe; Stage 0 closes (2026-07-03)

**Status: probe_100k COMPLETE (5/5 runs, `complete=True`, 0 failed, 0
timed out, 0 `n_skipped_steps`). This is Stage 0's final wave — Stage 0 is
now CLOSED as a diagnostic.** Raw data:
`experiment-runs/2026-07-02_stage0_waves/probe_100k/*.json` (5 per-run
JSONs, 50 dense checkpoints each at every 2,000 steps, plus a 501-point
per-step `trajectory` field). All numbers below are pulled directly from
these JSONs (`checkpoints`, `trajectory`, top-level summary fields), per
the same per-seed-resolution convention as §12/§13, and independently
recomputed from the raw files for this addendum (not copied from
`SUMMARY.txt`/`PROGRESS.txt`).

### 14.1 Final and tail numbers, d=32, all K

5 runs: K=8 ×3 seeds (the cell closest to the bar per §13.2), K=4 ×1,
K=16 ×1 — as launched in §13.5.

| Run | K | seed | onset (er off ~1.0 floor) | final eff.rank | final mean_cos | final rec@0.9 | max rec@0.9 (any ckpt) | final rec@0.95 |
|---|---|---|---|---|---|---|---|---|
| K4_s0 | 4 | 0 | 6,000–8,000 | 3.730 | 0.8297 | 0.1479 | 0.1509 (82,000) | 0.0056 |
| K8_s0 | 8 | 0 | 6,000–8,000 | 7.762 | 0.9093 | 0.6324 | 0.6324 (100,000) | 0.1655 |
| K8_s1 | 8 | 1 | 12,000–14,000 | 7.296 | 0.8768 | 0.4131 | 0.4230 (98,000) | 0.0013 |
| K8_s2 | 8 | 2 | 8,000–10,000 | 7.680 | 0.9145 | 0.6533 | 0.6639 (96,000) | 0.2360 |
| K16_s0 | 16 | 0 | 10,000–12,000 | 14.643 | 0.8524 | 0.2482 | 0.2482 (100,000) | 0.0243 |

**Onset is reliable and lands in the same 6–16K window established at
16–40K budgets (§12.2, §13.2)** — all 5 runs show the same flat-then-climb
signature: effective rank pinned near 1.0–1.1 through the pre-onset
window, then a sharp rise that **overshoots its eventual settled value
before compressing down toward K** (the shape §12.5 first flagged for
d=16,K=4 reproduces at d=32 — e.g. `K8_s0`: er peaks 12.24 @12K, settles
7.76 @100K; `K16_s0`: er peaks 15.28 @20K, settles 14.64 @100K).
`recovered_frac@0.95` stays low everywhere except `K8_s2` (0.236) —
nowhere near a working `@0.95` standard, let alone `@0.99` (0.0–0.006 in
every run).

### 14.2 The tails are flat, not climbing — a converged plateau, not truncation

`recovered_frac@0.9` at the last 6 checkpoints (steps 90,000–100,000,
every 2,000):

- `K4_s0`: 0.148, 0.142, 0.146, 0.146, 0.135, 0.148 — oscillating in
  [0.135, 0.150], no trend.
- `K8_s0`: 0.617, 0.625, 0.615, 0.618, 0.596, 0.632 — oscillating in
  [0.596, 0.632], no trend.
- `K8_s1`: 0.406, 0.416, 0.409, 0.417, 0.423, 0.413 — oscillating in
  [0.406, 0.423], no trend.
- `K8_s2`: 0.649, 0.658, 0.659, 0.664, 0.661, 0.653 — oscillating in
  [0.649, 0.664], no trend.
- `K16_s0`: 0.233, 0.237, 0.244, 0.232, 0.244, 0.248 — oscillating in
  [0.232, 0.248], no trend.

This is qualitatively different from §13.2's ext_budget tails at 40K
(still climbing: `d32_K8_s0` 0.338→0.430→0.445 over its last 3
checkpoints) and from §13.3's d=64/60K tails (also still rising). At
100K, every one of these five runs has stopped moving — 60,000+ steps of
flat oscillation after the transition completes. This is the direct
evidence that what follows is a genuine converged read, not another
undertraining snapshot.

### 14.3 Formal pass criterion: FAILS. Diagnostic pass condition: SUCCEEDS.

**§1's pre-registered CONFIRM bar — `recovered_frac@0.9 > 0.7` at d=32,
unconstrained, for K∈{4,8,16} — FAILS in this probe**, at 10× Task D's
original 8K-step budget and with visibly converged (non-climbing)
trajectories (§14.2). Best observed anywhere in Stage 0 to date is now
**0.6639** (`K8_s2`, checkpoint 96,000) — up from ext_budget's 0.4454
ceiling (§13.4) but still short of 0.7, and, unlike that earlier number,
**genuinely plateaued rather than still-rising**. This settles §13.5's
open question ("does the d=32 write cross the bar with enough budget, or
does it plateau below it?") in favor of **plateau**: this is the
pre-registered "still climbing vs. converged short of the bar" branch
(§13.4's own wording) resolving to the latter, now that the trajectory
data supports it.

This is a decisive negative for **mechanism (c) alone** (§1: "a
training-budget effect... the fixed 8–10K-step budget systematically
undersamples it") as *sufficient* to reach the CONFIRM bar — step count
was the dominant lever through Wave A/ext_budget (§13.4), but 10× budget
with a flat tail is no longer an undersampling story. It does **not**
resolve mechanisms (a) (architectural cap, not tested this wave — H_cap
only derived to bind at d=128, K>65, §2.1/§9) or (b) (row-query
differentiation) as the cause of the residual gap; both remain viable,
unresolved explanations for why the plateau sits below 1.0 rather than at
it (§14.4).

**The diagnostic pass condition (§1, independent of CONFIRM/FALSIFY) was
already met as of §12.6 and remains met:** every named hypothesis
(H_cap, H_collision, H_dead-init, H_undertrained, H_late-transition) has
a measured, falsifiable verdict across Wave 0/A/ext_budget/probe_100k,
not an assertion. Stating both together, plainly, since they are easy to
conflate: **the write does not reach the pre-registered exact-recovery
bar at d=32 (formal pass criterion: FAIL) — and Stage 0 nonetheless
succeeds as a diagnostic (diagnostic pass condition: SUCCESS)**, because
it produced a mechanistic, evidenced account of why, not a shrug.

### 14.4 What closes: three findings, one open question

Putting §12–14 together, Stage 0's arc resolves into:

1. **Trainability at d=32 was substantially a step-budget artifact.**
   Task D's original 8K-step "wall" (rank≈1, recovery≈0) was a
   mid-transition snapshot. Transitions are reliable — 6/6 seeds in Wave
   0, 6/6 more in ext_budget, 5/5 in this probe (17/17 d=32 baseline
   seeds across Stage 0 transition somewhere in the 6–16K window) — but
   onset alone is not the pass bar.
2. **Rank-tracks-K is restored at d=32** once budget suffices (first
   shown at ext_budget's 40K, §13.2; confirmed again here): final
   effective rank 3.73 (K=4), 7.30–7.76 (K=8), 14.64 (K=16) — Task D's M1
   pattern (`TASK_D_WRITEUP.md`) holds at d=32, not only at d=8/16.
3. **Converged exactness degrades with `d` — this is the real frontier,
   not trainability.** d=16 reaches genuinely exact solutions under the
   harder multi-hop objective (`recovered_frac@0.9` = 1.00 at every
   tested hop including h=21, 4/5 K=8 seeds, Task E's 40K round —
   `EXPERIMENT_LOG.md` 2026-07-02). d=32 plateaus at cos 0.83–0.91 and
   `recovered_frac@0.9` 0.15–0.65 on the simpler single-hop task, at a
   budget where the trajectory has demonstrably stopped moving (§14.2).
   The "d≥32 trainability frontier" language used through §12/§13 and in
   `STATE.md` was imprecise: transitions are reliable, so trainability
   is not the frontier — **exactness is.**
4. **The K=4 anomaly is now a converged phenomenon, not a budget
   artifact.** mean_cos 0.830 (healthy, comparable to K=8's 0.877–0.914)
   but `recovered_frac@0.9` only 0.148 at 100K — the qualitative
   cos/recovery split `TASK_D_FINDINGS_DRAFT.md` §3.2 first flagged, and
   §13.2 showed surviving a 5× budget extension to 40K, is confirmed flat
   at 10× budget. **K=8 (d/4) outperforms both its neighbors in the
   tested grid** (K=4: 0.148; K=8: 0.413–0.653; K=16: 0.248) while all
   three sit near effective-rank≈K — this non-monotonicity in K is real
   and still unexplained.

**What remains open, explicitly named as Stage 0.5 and NOT answered by
Stage 0:** *why* the d≥32, h=64 write plateaus sub-exact rather than
reaching 1.0. Candidate mechanisms are unresolved, not ruled out:
architectural cap (H_cap/mechanism (a), only derived to bind at d=128,
K>65 in this grid — `rank(Z)≤h+1=65` does not numerically bind at
d=32,h=64, but a *soft* version of the same width constraint could still
be in play well below its hard ceiling); row-query differentiation
(H_collision/mechanism (b), disfavored as an *accelerant* by two
independent findings in §13.1/§13.2 but not ruled out as a *ceiling*
factor — the causal orthogonal-init test targeted onset speed, not
converged asymptote); or a mechanism not yet named. No Wave A
intervention was re-tested against the plateau specifically (all were
screened at 16–20K, before this probe's plateau was known to exist).

### 14.5 Decision: Stage 0 CLOSED

Stage 0 is complete as a diagnostic (§1's diagnostic pass condition,
§12.6/§13.4/§14.3) and has reached a stable, non-climbing read on its
formal pass criterion (§14.3: FAIL, not "undetermined" — the ambiguity
§13.4 left open is resolved by this wave's flat tails). No further Stage
0 waves are planned: Wave B was already skipped as moot (§13.5), and this
100K probe was its replacement per that same pre-registration deviation.

**Practical consequence for Chapter 2/3 scoping:** `d≥32` is usable for
downstream purposes only in the **approximate** regime (cos 0.83–0.91,
`recovered_frac@0.9` 0.15–0.65 depending on K) — not the exact regime
Task D validated at d≤16. Any downstream use of d≥32 that requires exact
recovery (the pinned-linear-unbind readout Task D's provable
`rank(Z)≥K` bound depends on; `CLAUDE.md`'s "readout must force exact
continuous recovery" rule) is not supported by this data. d≤16 remains
the only confirmed-exact regime. Stage 0.5 (§14.4's open question) is a
candidate future design, not scheduled as of this addendum; unrelated
work already in progress on the box per `STATE.md` (DeltaNet causal-rank
Wave −1, Stage G) continues independently.

### 14.6 Compute

probe_100k: 5 runs, summed `wall_s` = 14,257.2s = **3.96 GPU-h** actual
(serial-sum accounting, same method as §12.9/§13.6; per-run: K4_s0
3,207.7s, K8_s0 3,015.6s, K8_s1 2,726.8s, K8_s2 2,549.3s, K16_s0 2,757.8s).
This brings Stage 0's running total to **23.70 GPU-h (§13.6) + 3.96 GPU-h
= 27.66 GPU-h serial-sum**, against the pre-registered 60 GPU-h core
budget (§6) — Stage 0 closes having used under half its budgeted compute,
before any packing-bonus adjustment (§12.9 measured ~4 runs/GPU in
practice on earlier waves). **Separately, the project's cumulative Brev
grant burn is reported at ~320 of the ~1,600 GPU-h grant as of this
addendum** (a box-uptime/billing figure tracked outside these run JSONs,
per the same convention as §13.6 — not independently re-derivable from
Stage 0's own `wall_s` sums, since it includes idle uptime and other
concurrent work on the box).

---

## 15. Exactness frontier — d=64 point (2026-07-02)

**Status: `probe_d64_150k` COMPLETE (4/4 runs, `complete=True`, 0 failed).
Stage 0 itself remains CLOSED (§14.5) — this addendum does not reopen the
§1 CONFIRM/FALSIFY question, which was scoped to d=32 trainability and was
answered there. It extends §14.4 finding 3 ("converged exactness degrades
with `d` — this is the real frontier, not trainability") with a third
converged data point, using the same probe methodology, run under the same
`experiment-runs/2026-07-02_stage0_waves/` archive convention.** Raw data
archived this session from
`youthful-indigo-turkey:/home/nvidia/chapter2/results/stage0/probe_d64_150k/*.json`
to `experiment-runs/2026-07-02_stage0_waves/probe_d64_150k/*.json` (4
per-run JSONs, 75 checkpoints each at every 2,000 steps). All numbers below
are pulled directly from these JSONs (`checkpoints`, top-level summary
fields), per the same per-seed-resolution convention as §12–§14.

### 15.1 Final numbers, d=64, 150K steps

4 runs: `d=64`, `variant=baseline`, `h=64`, `n_params=183,232` (matches
§5's `N(d)=166,784+257d` formula exactly, same as every other d=64 cell in
this document) — K=16 ×1 seed, K=32 ×3 seeds.

| Run | K | seed | final eff.rank | final mean_cos | final rec@0.9 |
|---|---|---|---|---|---|
| K16_s0 | 16 | 0 | 3.367 | 0.3882 | 0.0 |
| K32_s0 | 32 | 0 | 10.429 | 0.4889 | 0.0 |
| K32_s1 | 32 | 1 | 18.231 | 0.4682 | 0.0 |
| K32_s2 | 32 | 2 | 18.098 | 0.4482 | 0.0 |

All 4 `complete=True`, `n_skipped_steps=0` (no eigh-backward instability in
this batch either).

### 15.2 Trajectories are flat at 150K — a converged plateau, same signature as §14.2

Last 8 checkpoints (steps 136,000–150,000, every 2,000) for each run:

- `K16_s0` mean_cos: 0.3870, 0.3874, 0.3874, 0.3887, 0.3875, 0.3875,
  0.3880, 0.3882 — oscillating in [0.387, 0.389], no trend.
- `K32_s0` mean_cos: 0.4789, 0.4802, 0.4830, 0.4846, 0.4858, 0.4854,
  0.4866, 0.4889 — a slight residual drift (+0.010 over 14K steps) but
  effective rank is flat (10.51 → 10.43) across the same window; the
  smallest, still-settling case of the four.
- `K32_s1` mean_cos: 0.4677–0.4685 across all 8 points — flat, no
  measurable trend.
- `K32_s2` mean_cos: 0.4480, 0.4487, 0.4482, 0.4476, 0.4479, 0.4580,
  0.4576, 0.4482 — a brief bump at step 146,000 (0.458) that returns to
  0.448 by 150,000; net change over the window is ≈0, read as noise
  around a plateau, not a resumed climb (effective rank shows the same
  pattern: 17.66 → 18.16 → 18.10).

This is qualitatively the same "flat tail, not truncation" signature
§14.2 established for the d=32/100K read (as distinct from §13.2/§13.3's
still-climbing 40K/60K tails) — the direct evidence that 150K is measuring
a genuine converged asymptote at d=64 rather than another mid-transition
snapshot.

### 15.3 The exactness frontier: three converged points at h=64 (fixed), monotone in `d`

| d | source | budget | cos range | rec@0.9 range | read |
|---|---|---|---|---|---|
| 16 | Task E 40K/80K rounds | 40K–80K steps | ≈1.0 | 1.00 at every tested hop, incl. h=21 (§14.4 pt. 3; `EXPERIMENT_LOG.md` 2026-07-02 "Task E K-wall resolution — 80K round") | **exact** |
| 32 | Stage 0 100K probe (§14) | 100K steps | 0.83–0.91 | 0.15–0.65 (K-dependent) | plateau, sub-exact |
| 64 | this addendum | 150K steps | 0.45–0.49 | 0.0 at every seed/K tested | plateau, far sub-exact |

Both exactness measures — cosine fidelity and exact-recovery fraction —
degrade **monotonically** as `d` increases from 16 to 32 to 64 at fixed
`h=64`. §14.4 named the exactness frontier from two converged endpoints
(d=16 exact, d=32 sub-exact); this addendum adds the third point and shows
the degradation is monotone across it, not merely a two-point contrast.
`recovered_frac@0.9` is not just lower at d=64 than d=32 — it is
**identically 0.0** in all 4 runs, a qualitatively harder floor than d=32's
0.15 minimum.

### 15.4 A new gradation: rank recruitment is itself now partial at d=64

At d=32 (§14.4 finding 2), converged effective rank tracked `K` closely:
K=4→3.73 (93%), K=8→7.30–7.76 (91–97%), K=16→14.64 (91%). At d=64 this
pattern **partially breaks**:

- K=32: effective rank 10.4–18.2 — **32–57% of target**, roughly half or
  less of what d=32's cells achieved as a fraction of their own `K`.
- K=16: effective rank 3.37 — **21% of target**, the same qualitative
  shortfall as K=32, and by a similar fraction.

This is a distinct phenomenon from d=32's "rank-tracks-K, but the exact
readout still falls short of full recovery" pattern (§14.4 pt. 2 and pt.
3 were separable there — rank was healthy, exactness was not). At d=64
the **rank recruitment itself is now the thing lagging**, not just the
final linear readout on top of a healthy rank. Naming this precisely and
without over-claiming a hard mechanism: call it **partial rank
recruitment**, not rank collapse (effective rank is nonzero and
K=32 > K=16 in every seed, so `K`-sensitivity survives) and not yet
identified as a distinct architectural ceiling (H_cap's derived bound,
§2.1, does not bind until `d=128`; nothing here contradicts that).

**Caution against over-interpreting, stated explicitly per instruction:**
the 3-seed K=32 spread (10.4, 18.2, 18.1 — a 1.75× min/max ratio) is
markedly wider than d=32's own K=16 baseline spread at the same seed count
(§13.2: 14.3–16.1, a 1.13× ratio). Combined with §12.3/§13.3's already-
established finding that d=64 onset scales superlinearly in `d` and was
**still incomplete at the 25K/60K budgets** tested there, the wide K=32
spread here is consistent with (but not proof of) per-seed variation in
how *close to done* rank recruitment is at 150K specifically — i.e., some
seeds may still be mid-recruitment on the rank axis even though their
cosine/rec@0.9 trajectories have visibly plateaued (§15.2). Rank and
exactness are not shown here to plateau on the same clock; this is flagged
as **open**, not resolved, by this single probe. The honest read is: rank
recruitment at d=64 is partial and more seed-variable than at d=32, a new
gradation worth tracking, not yet a characterized ceiling.

### 15.5 A d=48 interpolation wave is in-flight (not analyzed)

Launched 2026-07-02: 3 runs (`variant=baseline`, K=24 ×2 seeds, K=12 ×1
seed), 100,000 steps, output to
`/home/nvidia/chapter2/results/stage0/probe_d48_100k/` on the box —
confirmed present via remote listing this session. As of this addendum,
all 3 runs show `complete=False`, `steps_completed=4,000` of 100,000
(4%) — far too early to read. Noted here only to record that the
interpolation point between §14's d=32 and this addendum's d=64 is
running; results deferred to a future addendum.

### 15.6 Compute

4 runs, summed `wall_s` = 30,262.1s = **8.41 GPU-h** actual (serial-sum
accounting, same method as §12.9/§13.6/§14.6; per-run: K16_s0 7,572.8s,
K32_s0 5,776.7s, K32_s1 8,364.8s, K32_s2 8,547.8s — average 2.10h/run).

**This is well under a naive linear extrapolation from Wave −1's own d=64
calibration timing** (§12.1: 1,022.2s for 10,000 steps → 15,333s ≈ 4.26h
for 150,000 steps, ×4 runs ≈ 17 GPU-h) — the same conservative-estimate
pattern §12.9/§14.6 already flagged (flat-rate/linear-scaling budget
guesses running roughly 2× high here, 5–6× high at the smaller Wave 0/A
cells), reported for the record rather than assumed.

Stage 0's core (§14.6) is unchanged at 27.66 GPU-h serial-sum against its
60 GPU-h budget — it closed at §14.5 and this figure is not re-added to
it. This addendum's 8.41 GPU-h is post-closure, exactness-frontier
follow-on work, tracked separately for the same audit-trail convention.
The in-flight d=48 wave (§15.5) is additional and not yet reflected in any
compute total.

### 15.7 The d=48 interpolation wave — complete (2026-07-02)

**Status: `probe_d48_100k` COMPLETE (3/3 runs, `complete=True`,
`steps_completed=100,000/100,000`, `n_skipped_steps=0`).** Supersedes
§15.5's in-flight placeholder. Archived this session from
`youthful-indigo-turkey:/home/nvidia/chapter2/results/stage0/probe_d48_100k/*.json`
to `experiment-runs/2026-07-02_stage0_waves/probe_d48_100k/*.json` (3
per-run JSONs, 50 dense checkpoints each at every 2,000 steps). All
numbers below are pulled directly from these JSONs, same convention as
§12–§15.6.

**Final numbers, d=48, 100K steps, h=64, `variant=baseline`,
`n_params=179,120`:**

| Run | K | seed | final eff.rank | eff.rank / K | final mean_cos | final rec@0.9 | `wall_s` |
|---|---|---|---|---|---|---|---|
| K12_s0 | 12 | 0 | 10.816 | 90.1% | 0.7196 | 0.00195 | 3,416.06s |
| K24_s0 | 24 | 0 | 6.460 | 26.9% | 0.4926 | 0.0 | 3,610.33s |
| K24_s1 | 24 | 1 | 4.419 | 18.4% | 0.2530 | 0.0 | 3,602.30s |

These confirm the numbers this addendum was seeded with (`K12_s0` cos
0.72/rec 0.002/er 10.8; `K24_s0` cos 0.493/er 6.5; `K24_s1` cos 0.253/er
4.4) to 3-4 significant figures, and all 3 `complete=True` as stated.

**Correction to the "flat tails at 100K" framing this addendum was
seeded with: two of three runs are flat, but `K24_s1` is not — it is
still mid-transition at 100,000 steps, not converged.** Per-checkpoint
trajectories (pulled from the full 50-point `checkpoints` array, not
just the final value):

- `K12_s0`: onset ≈14,000 steps, effective rank overshoots to 12.70
  @20,000 then compresses toward 10.8; `mean_cos` climbs steadily
  through the run and is still drifting mildly at the end (0.694 @80,000
  → 0.7196 @100,000, +0.026 over the last 20K steps) — the same
  slow-residual-drift-on-an-otherwise-settled-trajectory pattern §15.2
  called "flat" for `K32_s0` (+0.010 over 14K steps there). Read as
  flat-ish/settling, consistent with the seeded framing.
- `K24_s0`: onset ≈40,000 steps (much later than K12_s0 — onset grows
  with K at fixed d, echoing the K-axis onset pattern
  `TASK_E_FINDINGS.md` §10 found on the K=12/16 axis at d=16). `mean_cos`
  rises from 0.461 @80,000 to 0.4926 @100,000 (+0.032 over 20K
  steps) — still climbing at a similar rate to `K32_s0`'s drift, larger
  in absolute size but effective rank is genuinely flat (6.58 → 6.46,
  a slight net *decline*, not a rise) over the same window. Read as
  flat-ish on rank, mildly still-climbing on cosine — borderline, not a
  clean plateau, but qualitatively closer to "settling" than to
  "transitioning."
- **`K24_s1`: NOT flat.** Effective rank sits pinned at ≈1.00–1.08
  (dead-init floor) through step 78,000 — 78% of the entire budget spent
  at the trivial-solution value — then transitions late: rank climbs
  1.08→4.42 and `mean_cos` climbs 0.003→0.253 between steps 78,000 and
  100,000. The per-2,000-step `mean_cos` delta peaks at +0.0735 around
  step 88,000–90,000 and has been decelerating since (+0.0565, +0.0327,
  +0.0195, +0.0197, +0.0121 through the last 5 checkpoints) — past its
  steepest point but still moving at roughly 6× the noise-floor step-to-
  step delta that §14.2/§15.2 call "flat" (≈±0.002) at the point the run
  was cut off, nowhere near a plateau. **This run's `cos 0.253 / er 4.4`
  reading is a mid-transition snapshot, not a converged value** — it is
  the exact same "late seed-stochastic phase transition, budget-starved"
  signature `TASK_E_FINDINGS.md` §10 named "the three-budget-artifacts pattern"
  for K=16 s1 at d=16 (which needed 120K+ steps; a completion wave for
  that specific case is running now, §15.7.3 below). The seeded framing
  ("all complete=True, flat tails at 100K") is accurate on `complete=True`
  but **wrong on flat tails** for this one seed — flagged here rather
  than silently propagated, per this project's verify-before-claiming
  rule.

**Practical consequence:** `K24_s1`'s 0.253/4.4 numbers should be read as
a lower bound on what K=24 eventually reaches at d=48, not as its
plateau — unlike `K24_s0`, which despite its own mild residual drift is
much further into its transition (onset 40K vs. 78K) and closer to a
genuine read.

#### 15.7.1 The exactness frontier: four points at h=64 (fixed), still monotone but noisier at the K=d/2 slice

Extending §15.3's three-point table with this addendum (`K=d/2` column;
`K24_s1`'s number flagged as a floor, not a plateau, per above):

| d | K (=d/2) | source | budget | cos range | read |
|---|---|---|---|---|---|
| 16 | 8 | Task E 40K/80K rounds | 40K–80K | ≈1.0 | exact |
| 32 | 16 | Stage 0 100K probe (§14, 1 seed) | 100K | 0.85 | plateau, sub-exact |
| 48 | 24 | this addendum (2 seeds) | 100K | 0.25–0.49 | plateau (s0) / still-climbing floor (s1) |
| 64 | 32 | §15.1–15.3 (3 seeds) | 150K | 0.45–0.49 | plateau, far sub-exact |

**d=48's K=24 range (0.25–0.49) overlaps d=64's K=32 range (0.45–0.49)
— the K=d/2 slice is not monotone-clean between d=48 and d=64.** At face
value this looks like the frontier stalling or even reversing between
48 and 64; two confounds make that reading premature rather than
disproving the frontier outright:

1. **Seed spread is large relative to the gap.** d=48 K=24 has only 2
   seeds and one of them (`s1`) is a confirmed still-climbing floor, not
   a plateau — its "true" eventual value is unknown and could land
   anywhere ≥0.253. If `s1` continues toward `s0`'s 0.493 (the two
   seeds' onset/transition dynamics are otherwise similar in kind, just
   offset in time), d=48's effective K=24 read moves toward 0.35–0.49,
   which no longer cleanly overlaps d=64's 0.45–0.49 range from below —
   it would sit adjacent to it, consistent with (not contradicting) a
   monotone-decreasing-in-d frontier. The current 0.25 floor is not
   strong evidence against monotonicity; it is under-converged data.
2. **Budget is not held fixed across `d`.** d=48 ran 100K steps; d=64 ran
   150K (§15.1). A 1.5× budget difference is not nothing given how late
   these transitions land (`K24_s1` transitioned at ~78K of its own
   100K budget — a d=64 run given only 100K might read comparably
   unconverged). The frontier table's cos-range comparison across d is
   confounded by this budget mismatch and should not be over-read as a
   clean apples-to-apples slice.

**The `K=d/4` slice is cleaner** — no overlap, monotone, and (for the d=48
point) drawn from a single seed but past a much longer completed
transition (onset ~14K, ~86K steps of subsequent training) than the
K24_s1 floor above: d=32 K=8 → 0.877–0.915 (3 seeds, §14.1); d=48 K=12
→ 0.7196 (this addendum, 1 seed). A clean, well-separated
monotone step at `K=d/4`, in contrast to the noisier, confound-prone
`K=d/2` read above.

#### 15.7.2 Partial rank recruitment continues its gradation

Effective-rank-as-fraction-of-K at d=48: K=12 → 90.1% (comparable to
d=32's K=16 91% from §14.4 finding 2 — d=48's K=d/4 point still
recruits rank almost fully); K=24 → 26.9% (s0, converged-ish) / 18.4%
(s1, floor, still rising). This sits **between** d=32's full-recruitment
regime (all tested K ≥91%) and d=64's most-degraded point (K=16 at
21%, §15.4) — consistent with §15.4's "partial rank recruitment" naming
extending smoothly rather than appearing abruptly at d=64. `s1`'s 18.4%
figure is, per §15.7 above, a floor that will likely rise with further
training, so the true d=48/K=24 recruitment fraction is bounded below by
18.4% and above by (at least) `s0`'s 26.9%, not yet pinned down more
tightly than that.

#### 15.7.3 Also running: Task E K=16 completion wave at 120K steps

Separately from Stage 0, a `run_task_e.py` completion wave for
`TASK_E_FINDINGS.md` §10's stuck `K=16 s1` case (d=16, "never
transitioned" at 80K) was confirmed running on the box this session:
3 seeds (`seed=1,3,4`, `--steps 120000`, `--save-z`, output to
`results/task_e_120k_k16/`), live processes confirmed via `ps aux` on
GPUs 0–2 (the other 5 GPUs were running unrelated concurrent work —
DeltaNet Wave −1/0 and Stage-G Wave 0 sweeps). Not part of this Stage 0
addendum's data or conclusions; noted here only as an in-flight pointer
for a future entry, same convention as §15.5.

#### 15.7.4 Compute

3 runs, summed `wall_s` = 10,628.7s = **2.95 GPU-h** actual (serial-sum;
K12_s0 3,416.1s, K24_s0 3,610.3s, K24_s1 3,602.3s — all three within 6%
of each other despite very different final effective ranks, consistent
with wall-clock being dominated by fixed per-step cost at this scale
rather than by convergence behavior). Post-closure exactness-frontier
running total: 8.41 (§15.6, d=64) + 2.95 (this addendum) = **11.36
GPU-h**, still tracked separately from Stage 0's own closed 27.66/60
GPU-h core total (§14.6, unchanged).

---
