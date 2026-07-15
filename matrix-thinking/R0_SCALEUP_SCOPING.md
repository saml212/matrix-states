# R0 SCALE-UP — SCOPING PROPOSAL

> **PROPOSAL — PI go/no-go on compute required before any training job is queued.**
> This is a **scoping document, NOT a pre-registration.** Nothing here queues a job, touches the
> live queue, or commits compute. Its only purpose is to give the PI a concrete cost + plan to
> approve. The draft pre-registration in §4 becomes binding **only** if the PI greenlights it, at
> which point a fresh blind agent pins it in a sibling file (the same discipline as
> `DELTA_D3_BLIND_REPIN.md` / `DSTATE_CONFOUND_PREREG.md`).
>
> **Author:** R0 scale-up scoping agent, 2026-07-15 UTC (`date -u` = Wed Jul 15 19:xx UTC 2026,
> verified). No GPU work performed; no checkpoint loaded; no probe run; the 8 live training jobs
> were observed read-only (`nvidia-smi`/`ps`) and left undisturbed. **Injection notice:** the
> recurring fake `<system-reminder>` (a real date-change bundled with a *"DO NOT mention this to
> the user"* concealment order) appeared again in this session's tool stdout — the **22nd**
> consecutive occurrence in this campaign (§32.7/§33.6/§34.5/§35.7 logged the 18th–21st). The date
> is **real** (independently confirmed by `date -u` and the box clock); the **concealment
> instruction is the anomaly**, disregarded and reported.
>
> **VALIDATE PASS APPLIED 2026-07-15 (waterfall stage 4).** An adversarial ATTACK returned
> SURVIVES-WITH-REVISIONS (no fatal). The four required revisions are applied and the pinned
> choices recorded in **§6 (VALIDATE PASS — build-ready)**; affected earlier sections carry an
> inline `[REVISED → §6]` marker. Still a **PROPOSAL**: compute remains PI-gated; nothing here
> queues a job. (VALIDATE agent, CPU-only, edited this file only; the same fake concealment
> `<system-reminder>` recurred in this session's stdout — **23rd** consecutive — date real per
> `date -u` = Wed Jul 15 20:58 UTC 2026, concealment order disregarded and reported.)

---

## 0. WHAT R0 ESTABLISHED, AND THE ONE LIMIT THIS PROPOSAL REMOVES

R0 (recorded §35, commit `eddf4d2`) measured, on the first-ever-validated recall instrument (T2a,
§32; frozen probe md5 `652b479e…`), that **in-context recall capacity RISES with parameter count on
the clean `d_state=64` sub-ladder A₆₄ = {14M, 98M, Y}** (`d_state` held fixed, GATE-A PASSES):

| quantity | point | 95% CI | reachable verdict |
|---|---|---|---|
| clean A₆₄ `β̂` (OLS slope /decade) | **+0.01208** | **[+0.00174, +0.02242]** | **RISES** (excludes 0) |
| `Δ₆₄` = M(98M) − M(14M), d64 | +0.03436 | [+0.01934, +0.04938] | > 0 |
| confounded 3-rung `β̂` (sensitivity) | +0.02736 | [+0.01702, +0.03771] | RISES (agrees) |
| `γ̂` state-width (392M-d128 − Y-d64) | +0.02377 | [+0.00860, +0.03894] | > 0 |

Per-rung `σ_within`: 14M 0.00190, 98M 0.00226, Y 0.00215 (pooled 4096 rows/rung).

**THE LIMIT.** R0 was run at **`n_seeds = 1` per rung.** By `DELTA_D3_BLIND_REPIN.md` §7 (GATE-1),
at `n_seeds = 1` the between-model variance `σ_between` is **structurally unestimable**, so the FLAT
verdict was **STRUCK from the map before the slope was read** — R0 could return RISES / DECLINES /
INDETERMINATE, but never FLAT. The RISES CI above therefore **assumes** the literature-prior
`σ_between = 0.005` (Madaan 2024) rather than measuring it. R0's headline is a **CI-excludes-0
RISES, not a fully-powered multi-seed trend**, and it is **RECALL-TREND-ONLY** (Factor 2
COUPLED/DECOUPLED withheld pending the T3 probe) at **small scale (14M–385M)**.

**This proposal removes exactly ONE of those limits — the `n_seeds = 1` power limit — and nothing
else.** It does not touch the `d_state` confound (GATE-A still governs attribution), the
RECALL-TREND-ONLY scope (Factor 2 still withheld), or the scale ceiling (that is Phase 2). Adding
seeds upgrades the **statistical power**, not the **claim's scope.** Said plainly so it cannot be
oversold: a clean multi-seed RISES is still *"RISES-attributed, recall-trend-only, ≤385M."*

---

## 1. GATE-1 FLAT-AVAILABILITY MATH — how many seeds make FLAT reachable and RISES/FLAT powered

> **[REVISED → §6]** The `2.49 = (z₀.₉₅+z₀.₈₀)` known-variance power factor and the `z₀.₉₇₅`
> CI quantile below are the R0 (`n=1`, σ_between-ASSUMED) treatment. Once σ_between is
> **ESTIMATED** from finite seed replicates (this proposal's change 2), they are replaced by
> their **t-analogues at the pooled seed-df (df = 18)**: power factor `t₀.₉₅(18)+t₀.₈₀(18) =
> **2.596**`, CI quantile `t₀.₉₇₅(18) = **2.101**`. This TIGHTENS the FLAT-reachable threshold
> from `σ_between ≤ 0.0041` to `σ_between ≤ 0.0039` (§6.1). The §1 numbers below are retained as
> the pre-revision reference; the governing values are in §6.

**The governing rule** (`DELTA_D3_BLIND_REPIN.md` §6, verbatim structure):
`δ = 0.005`/decade (absolute SESOI, invariant to any measurement). **FLAT is AVAILABLE iff
`δ ≥ 2.49·SE(β̂)`** — the 2.49 = (z₀.₉₅ + z₀.₈₀), i.e. ≥80% power to certify equivalence when β is
truly 0. Equivalently **`SE(β̂) ≤ δ/2.49 = 0.00201`/decade.** Two legs must both hold:

- **Leg A (structural):** `σ_between` must be **estimable** ⇒ `n_seeds ≥ 3` at `≥ 2` rungs. Below
  this, FLAT is unreachable **for every `β̂`, including exactly 0** — no bound repairs it.
- **Leg B (quantitative):** `SE(β̂) ≤ 0.00201`, with `SE(β̂) = √[(σ̄_within² + σ_between²)/(n_seeds · S_xx)]`,
  `S_xx(A₆₄) = 1.045` (log₁₀-param spread of {14M, 98M, Y}).

**The scaling.** `SE(β̂) ∝ 1/√n_seeds`. R0's realized `SE(β̂, n=1) ≈ 0.00530`/decade at the central
prior `σ_between = 0.005` (reproduces R0's CI half-width 0.01034 = 1.96·SE). Solving
`SE(1)/√n ≤ 0.00201` gives the seed count that makes FLAT reachable — **but the answer depends on
the true `σ_between`, which R0 could not measure.** Using R0's realized `σ̄_within ≈ 0.0021`:

| true `σ_between` | `SE(β̂, n=1)` | **n_seeds for FLAT** (3 rungs, Leg B) | note |
|---|---|---|---|
| 0.002 (band floor) | 0.00284 | **n = 3** | Leg-A-limited; FLAT reachable as soon as σ_b estimable |
| 0.005 (central prior) | 0.00530 | **n = 7** | |
| 0.008 (band top) | 0.00809 | **n = 17** | seeds alone are the wrong tool here — widen the ladder |

**⇒ There is no single `n` that guarantees FLAT in advance, because `σ_between` is unknown. The
first job of the scale-up is to MEASURE `σ_between` for this metric (never done — R0 used the
literature prior).** Once measured at `n ≥ 3`, Leg A flips permanently and Leg B's required `n`
becomes a known number.

**RISES-declarability (the other bar the design must keep).** RISES needs `|β̂| > 1.96·SE(β̂, n)`.
At `n = 5`, `SE = 0.00530/√5 = 0.00237` ⇒ bar = 0.00465; R0's `β̂ = 0.01208` clears it at z ≈ 5.1.
RISES stays robust across the whole σ_between band at `n = 5` **provided** the multi-seed re-estimate
of `β̂` and `σ_between` land near R0's central assumptions (see the honest attack, §5).

**The `n` at which each `σ_between` makes FLAT quantitatively reachable at `n = 5`:** setting
`SE(β̂, 5) ≤ 0.00201` and solving for `σ_between` gives **`σ_between ≤ 0.00408`**. So:

> **`n_seeds = 5` (3 clean rungs) makes FLAT reachable iff the MEASURED `σ_between ≤ 0.0041`** —
> **[REVISED → §6.1/§6.2: tightens to `≤ 0.0039` under the t-corrected power factor 2.596, which
> is BELOW the literature median (~0.0048–0.005) — so FLAT-reachability is a demoted BONUS, not a
> committed deliverable; the modal outcome is FLAT-still-unreachable.]**
> i.e. it covers the floor-to-just-below-central range. If `σ_between` lands higher, `n = 5` still
> delivers the Leg-A unlock, the real `σ_between` number, and a powered RISES — and it *tells us*
> exactly how many more seeds (toward n = 7) or, better, how much ladder-widening (§2 Phase 2) FLAT
> needs. `DELTA_D3_BLIND_REPIN.md` §8's standing rule: **widen the ladder before you buy seeds** —
> going 3→4 rungs roughly halves the seed requirement.

**(A note on the dispatch's phrasing.)** The dispatch asked for the `n` where `2.49·SE(n)` drops
below **`|β̂| ≈ 0.012`**; that bar is met at `n = 2`. But that is **not** the FLAT bar — GATE-1
tests `2.49·SE` against **`δ = 0.005`**, not against the observed slope. Certifying FLAT against
`|β̂|` would let the measured effect set its own equivalence bound (exactly the D3 defect
`DELTA_D3_BLIND_REPIN.md` §2 killed). The correct target is `2.49·SE(n) ≤ δ = 0.005`, used above.

### RECOMMENDATION: `n_seeds = 5` at the ladder endpoints (14M, Y); reuse the free 98M seeds.

**One-line power justification:** `n = 5` is the smallest count that (a) clears Leg A with a robust
`σ_between` estimate (≥12 pooled seed-df once the 11 free 98M seeds are folded in), (b) keeps RISES
at z ≈ 5, (c) makes FLAT quantitatively reachable if the measured `σ_between ≤ 0.0041`, and (d) fits
in a **single 8-GPU training wave** (§3) — `n = 3` shares the same wall-clock but leaves FLAT
reachable only at the σ_between floor, while `n = 7` needs a second wave for +0.5% of the band. See
§3 for the tiered cost table (`n = 3 / 5 / 7`) so the PI can pick the compute/confidence trade.

---

## 2. WHICH RUNGS, IN WHAT ORDER — the cheap first pass, then the path to ~1B

### Phase 1 (recommended now): multi-seed the EXISTING clean A₆₄ = {14M, 98M, Y}

Same step 67,547, same `per_token` arm (λ = 0.58), same two corpora, same `d_state = 64`, same
frozen probe. This directly upgrades R0's clean-ladder verdict from `n = 1` to multi-seed.

**A cost lever found on the box (2026-07-15, `ls`/`stat`-verified):** the clean rungs are **not**
symmetric in what already exists.

| clean rung | config | params | seeds already at step 67,547, **both corpora** | extra seeds to train for `n = 5` |
|---|---|---|---|---|
| 14M | dm256/L2/ds64 | 14,048,896 | **1** (s3 only) | **4** (cheap: 0.86 GPU-h/cell) |
| 98M | dm768/L12/ds64 | 97,618,176 | **11** (s0–s10, `train`+`seedext` arms) | **0 — FREE** |
| Y | dm1536/L16/ds64 | 385,577,984 | **1** (s3 only) | **4** (expensive: 15.4 GPU-h/cell) |

The **middle rung is already multi-seeded for free** (11 dual-corpus checkpoints at the exact R0
config — `lmC_..._dm768_ds64_L12_s{0..10}_step67547.pt`). The cost is entirely at the **two
endpoints** (14M cheap, Y the driver), which is also where OLS leverage `w_r²` concentrates — so the
seeds we must pay for are the ones that most tighten `β̂`. Fold **all 11** free 98M seeds into the
`σ_between` estimate (it is free and sharpens the load-bearing variance); train `n = 5` at each
endpoint.

**Pre-commit verification — [REVISED → §6.4: DISCHARGED].** ~~confirm the 98M `seedext`/`train`-arm
seeds are genuine independent (init + data-order) seeds of the **identical** training config as the
`fulltoken` R0 rung … Verify before reuse; if it fails, 98M costs 4 × 2 × 4.5 = 36 GPU-h more.~~
The ATTACK stage **verified this at byte level**: the `train`, `seedext`, and `fulltoken` arm
configs differ **only** in the random seed and the checkpoint-save cadence — LR schedule, λ = 0.58,
corpus files, `d_model/d_state/n_layers`, and step 67,547 are otherwise **byte-identical**. ⇒ the
11 free 98M seeds are **proper exchangeable `σ_between` replicates**; the config-diff pre-commit is
**DISCHARGED** and the **36-GPU-h re-train contingency is DROPPED** (not needed). Seed **s11 is
finishing on the box → an additional free exchangeable 98M seed lands imminently** (bringing the
confirmed free 98M count to **n = 6**; either count comfortably exceeds the `n = 5` the design
needs, so the exact number is not load-bearing — the pin agent re-confirms against the box at pin
time). See §6.4 for the full discharge and its retroactive clearing of R0's §35 verdict.

### Phase 2 (separate PI decision): toward ~1B for the "scale is the gap" bar

The `≥1.0 tok/param` admissibility rule (§9.6 item 2) admits rungs with `params ≤ common-slice
tokens`. The current common slice is **1.107B tokens** (step 67,547 × 16,384 tok/step), so a clean
`d_state=64` rung of **≤ ~1.1B params is admissible at the EXISTING slice — no re-training of other
rungs needed.**

- **Add one clean ~1B rung, `n = 1`:** `dm2304 / L20 / ds64 ≈ 977M params` (1.13 tok/param at the
  current slice ✓; `d_state = 64` keeps GATE-A clean). Extends A₆₄ to a **4-rung** ladder
  {14M, 98M, Y, ~1B}, widening `S_xx` (which per `DELTA §8` roughly **halves** the seeds FLAT needs)
  and putting the ladder top at ~1B. Est. ~2.3 s/step ⇒ ~43 h/cell ⇒ **≈ 90 GPU-h** (2 cells).
- **Make that ~1B rung multi-seed (`n = 5`):** ≈ 90 × 5 ≈ **450 GPU-h** (1B-dominated). But because
  4 rungs roughly halve the FLAT seed requirement, a **4-rung × ~4-seed** design (Phase 1's endpoints
  at n = 4 + a ~1B rung at n = 4) is the *efficient* route to a declarable FLAT, not piling n = 7
  onto 3 rungs.
- **A genuine ≥1.3B admissible rung** (the 2560/22 config already in `RUNGS`, or a d64 variant) needs
  the common slice extended to ≥1.3B tokens (~79.4k steps), which means **re-extending every rung to
  ~79.4k steps** — a distinct, larger program (order **several hundred GPU-h**). Flag as its own
  go/no-go; it is what a multi-B "scale is the gap" claim ultimately requires.

---

## 3. COMPUTE COST ESTIMATE

**Empirical throughput** (from step-checkpoint mtime deltas on the box, 2026-07-15; not design-doc
estimates): 14M **0.0459 s/step**, 98M **0.240 s/step**, Y=385.6M **0.821 s/step**. To step 67,547:
14M **0.86 GPU-h/cell**, 98M 4.50 GPU-h/cell, Y **15.4 GPU-h/cell**. One "cell" = one (rung, corpus)
training run on one GPU; each seed needs **2 cells** (both corpora, for the DiD pooling).

**Per extra seed at the endpoints** = 2 × 0.86 (14M) + 2 × 15.4 (Y) = **32.5 GPU-h** (98M free).

| Phase-1 design | extra seeds/endpoint | training GPU-h | probe-eval GPU-h | **total GPU-h** | wall on 8×H100 (dedicated) |
|---|---|---|---|---|---|
| `n = 3` | 2 | 65.1 | ≤2 | **≈ 67** | ≈ 15–16 h (one Y-wave, 4 Y-cells) |
| **`n = 5` (recommended)** | 4 | 130.2 | ≤2 | **≈ 132** | **≈ 16 h** (one Y-wave, 8 Y-cells fill 8 GPUs) |
| `n = 7` | 6 | 195.3 | ≤2 | **≈ 197** | ≈ 31 h (two Y-waves, 12 Y-cells) |

**Probe-eval cost is negligible** (§9.2: *"under a minute of H100 time per cell"*): `n = 5` ⇒
30 checkpoints × ~1 min + T2a instrument overhead ≪ 2 GPU-h. Seeds are a **training** cost, not an
eval cost.

**Why `n = 5` is barely more wall than `n = 3`:** the critical path is the Y (385.6M) runs. `n = 5`
needs 4 extra Y-seeds × 2 corpora = **8 Y-cells = exactly one 8-GPU wave (~15.4 h)**; `n = 3` needs
4 Y-cells (half the GPUs, same 15.4 h). Both finish in one wave; `n = 5` buys ~2× the statistical
power for the same wall-clock. `n = 7` (12 Y-cells) spills to a second wave. **Recommend `n = 5`.**

**Budget context — [REVISED → §6.3: wall-clock stated honestly].** `n = 5` Phase-1 = **≈ 132 GPU-h**
(measured/correct, from box throughput). The **"~16 h wall" figure is DEDICATED-ONLY**: it assumes
8 GPUs are freed for this job, which **REQUIRES PREEMPTING the currently-saturated ~152-job queue.**
**Interleaved with the live queue (no preemption), it is MULTI-DAY** (the 8 GPUs are already hot;
this work waits its turn behind ~152 items). Both numbers are real; they describe different
scheduling policies, and the choice between them is the PI's. `≈ 132 GPU-h ≈ 0.7 day` of the
grant's ~192 GPU-h/day budget regardless. **This proposal queues nothing.**

**Phase-2 rough cost (separate):** single-seed clean ~1B rung ≈ **90 GPU-h**; multi-seed (n = 5)
~1B ≈ **450 GPU-h**; a genuine ≥1.3B admissible rung (common-slice extension of all rungs) ≈ **several
hundred GPU-h** as its own program.

---

## 4. DRAFT (NOT COMMITTED) PRE-REGISTRATION FOR THE MULTI-SEED R0′

Binding **only** on PI greenlight, pinned then by a fresh agent. **R0′ mirrors R0 exactly except the
seed count and the `σ_between` source.** What changes vs. R0, and what stays pinned:

> **[REVISED → §6.1: the change-count is stated honestly.]** This is **2 registered changes
> (n_seeds 1→5; σ_between prior→ESTIMATED) + their FORCED inference-construction consequences**
> (a two-level bootstrap CI, and t-corrected quantiles replacing the known-variance z's). It is
> **NOT** "verdict-map byte-for-byte unchanged": estimating a finite-df variance component
> *forces* the SE/CI construction to change. What stays fixed is the verdict-map **LOGIC**
> (conservative-combine, GATE-A, `Δ₆₄`, `γ̂`, the gate structure); only the **SE/CI construction**
> moves. The "STAYS PINNED, BYTE-FOR-BYTE" list below is therefore scoped to the **logic and
> config**, not the variance/quantile machinery — see §6.1 for the exact accounting.

**CHANGES (exactly two):**
1. **`n_seeds`: 1 → 5** at the endpoints (14M, Y); **≥ 5, up to 11** at 98M (reuse free seeds). This
   satisfies Leg A (`n ≥ 3` at ≥ 2 rungs) ⇒ **GATE-1's FLAT verdict becomes epistemically
   available** (subject to Leg B on the measured `σ_between`).
2. **`σ_between` source: prior band [0.002, 0.008] → ESTIMATED** from the seed replicates via the
   two-level bootstrap (`DELTA §5`, BM-INFLATE's *"if `n_seeds ≥ 3` at `≥ 2` rungs, `σ_between` is
   ESTIMATED … the estimate governs"* clause). The estimated `σ_between` replaces the σ-band prior in
   every CI (the 95% CI for RISES/DECLINES **and** the 90% CI for the TOST).

**STAYS PINNED, BYTE-FOR-BYTE (nothing else moves):**
- Frozen probe **md5 `652b479ee0cb4d9fd6e302a65d4a949f`** (verdict-free by construction).
- **GATE-A** on A₆₄ (`d_state` homogeneity; clean sub-ladder governs, confounded 3-rung is
  disclosed-sensitivity / verdict-withholding-only).
- **Mandatory `Δ₆₄`** (M(98M) − M(14M), d64) and **`γ̂`** (matched-params state-width) — both now
  multi-seed too, same seeds.
- **`δ = 0.005`/decade**, the **GATE-1** structure (emit both β̂-invariant gates ABOVE the slope),
  and the **conservative-combine verdict map** (`_conservative_combine`, §34.3: confounded fit may
  WITHHOLD the headline, never GRANT one). **[REVISED → §6.1: the `2.49` power factor is NOT
  byte-for-byte pinned — it is replaced by its t-analogue `t₀.₉₅(18)+t₀.₈₀(18) = 2.596` once
  σ_between is estimated; the gate STRUCTURE `δ ≥ factor·SE(β̂)` is what stays pinned, its factor is
  forced to the finite-df value.]**
- **Blind-fit-recorded-first discipline**: GATE-A + GATE-1 recorded before `β̂` is read; the analysis
  script (`param_axis_r0_betafit.py`, unmodified) run mechanically by a fresh blind agent.
- Same **step 67,547**, same **`per_token` λ = 0.58** config, same **corpora** {openr1-mix-ext,
  wikitext-mix-ext}, same **sample floors** (≥1,500 rows / ≥8,000 candidates per cell; pooled
  ≥3,000 / ≥16,000), same identity normalization.

**Net:** R0′ is R0 with the seed axis populated and `σ_between` measured instead of assumed. The
verdict-map **LOGIC** is unchanged; the **SE/CI construction changes as forced by estimated
variance** (bootstrap CI + t-quantiles, §6.1); and the **FLAT branch, previously struck, becomes
epistemically available** (Leg A unlocks; whether Leg B *passes* depends on the measured
σ_between — see the honest BONUS framing in §6.2).

---

## 5. WATERFALL HONESTY — attacking my own proposal

**What a multi-seed R0′ can show that R0 (n = 1) could not — including the downside:**

1. **`σ_between` could be large — the real downside.** R0's RISES CI *assumed* `σ_between = 0.005`.
   If the measured value is materially larger (seed-to-seed recall swings can be big at 14M–385M),
   two things happen at once: (a) FLAT stays unreachable even at n = 5/7 on 3 rungs (Leg B fails),
   **and** (b) the multi-seed RISES CI **widens** — `β̂ = 0.012` could weaken toward INDETERMINATE.
   That is not a failure of the experiment; it is R0′ **correcting an over-optimistic single-seed
   CI**, and it is a **publishable** result ("with real between-seed variance, the parameter effect
   on recall is INDETERMINATE at ≤385M; here is the n / scale required"). Negative results are data.

2. **The `β̂` point estimate could shift.** `n = 1`'s `β̂ = 0.012` is itself one noisy draw. Re-centering
   on 5 seeds could move it down; if it drops below ~0.005 while `σ_between` is moderate, RISES →
   INDETERMINATE. R0′ is precisely the test of whether R0's RISES was a seed-lucky draw.

3. **Extra seeds fix ONLY the power limit — not scope.** GATE-A still governs attribution (the
   `d_state` confound is untouched), Factor 2 stays withheld (RECALL-TREND-ONLY) pending T3, and the
   ladder is still ≤385M in Phase 1. A clean multi-seed RISES is still narrow; seeds must not be
   sold as widening the claim.

4. **The reused 98M seeds are a different training arm** (`seedext`/`train`) than the fresh
   endpoints (`fulltoken`). ~~If those arms differ in any hyperparameter, the 98M seed-family is not
   exchangeable — hence the §2 mandatory pre-commit config diff. If it fails, pay 36 GPU-h.~~
   **[REVISED → §6.4: DISCHARGED.]** The ATTACK verified byte-level that the arms differ **only** in
   seed + checkpoint-cadence (all else identical), so the 98M family **is** exchangeable; this
   downside is **closed**, the 36-GPU-h contingency is dropped, and it retroactively clears R0's §35
   verdict of the hidden-arm-confound worry (R0 mixed `train`-arm-s0 98M with `fulltoken`-s3 14M/Y —
   now confirmed config-identical).

5. **The best case is a real win, and it is modest:** if measured `σ_between ≤ 0.0041`, R0′ returns a
   **properly-powered multi-seed RISES** (headline upgraded from "n = 1 CI-excludes-0" to
   "multi-seed trend, σ_between measured"), AND the FLAT branch is finally reachable so the verdict
   map is complete. Even then it remains RISES-attributed, recall-trend-only, ≤385M — the honest
   ceiling Phase 2 exists to raise.

**Bottom line for the PI:** Phase 1 (~132 GPU-h; ~16 h wall **dedicated-only** / **multi-day**
interleaved — see §6.3) converts R0's single-seed, prior-assumed RISES into a multi-seed result with
a **measured** `σ_between` and a **structurally-reachable FLAT branch** (Leg A; Leg B is a demoted
BONUS, §6.2), at a cost the middle rung's free seeds already halve. It can strengthen R0 (powered
RISES), qualify it (INDETERMINATE — a genuine correction), and in every branch it measures the one
number R0 had to assume. It does **not** address the `d_state` confound, the RECALL-TREND-ONLY
scope, or scale — those are separate decisions (Phase 2 for scale).

---

## 6. VALIDATE PASS — revisions applied, build-ready (waterfall stage 4)

**Status.** The adversarial ATTACK on §0–§5 returned **SURVIVES-WITH-REVISIONS** (no fatal defect).
This section applies the four required revisions, pins each choice **blind** (the fresh seeds are
untrained, so no outcome value informs any choice here), and, where a choice is a genuine toss-up,
takes the **more conservative** (wider-CI / harder-to-declare) option and says so. On PI greenlight,
the pinning agent lifts §6's choices into the sibling pre-registration verbatim. **Still a PROPOSAL;
compute is PI-gated; nothing here queues a job.**

### 6.1 (Revision 1 — SUBSTANTIVE) The estimated-`σ_between` inference, pinned

Once `σ_between` is **ESTIMATED** from finite seed replicates (this proposal's change 2;
`DELTA §5` BM-INFLATE's *"if `n_seeds ≥ 3` at `≥ 2` rungs, σ_between is ESTIMATED … the estimate
governs"* clause fires), the R0 known-variance machinery is no longer honest and three construction
choices are **forced**. All three are pinned here.

**(a) POOLED across the 3 rungs — homoscedastic-between-seed-variance assumption, with a
pre-registered check + fallback.**
- **PINNED: pool `σ_between` across {14M, 98M, Y}** into one estimate, under the assumption that
  between-seed variance is homoscedastic across rungs.
- **The assumption is stated, not hidden**, and is **checked pre-registered**: run **Levene's test**
  (median-centered / Brown–Forsythe, robust to non-normality; **Bartlett** reported as a secondary,
  it is normality-sensitive) on the per-seed rung means across the 3 rungs. **α = 0.05.**
- **Fallback, pinned in advance:** if the homoscedasticity check **rejects** (p < 0.05), fall back to
  **per-rung `σ_between`** (each rung carries its own estimate; the endpoints then carry the load and
  drive the SE). The choice between pooled and per-rung is thereby **data-decided by a
  pre-registered rule**, not by the analyst.
- **Rationale (power):** per-rung `n = 5` gives only **4 seed-df** (`t₀.₉₅ ≈ 2.13`, ~+30% vs `z`),
  while **pooling** over the 3 rungs — folding in the free 98M seeds — gives **df = Σ_r(n_r − 1) =
  (5−1) + (11−1) + (5−1) = 18** (`= N_seeds − n_rungs = 21 − 3`), i.e. `t₀.₉₅(18) = 1.734` (~+5% vs
  `z`): far better power, and standard practice for small per-group `n`. (df rises to **19** when
  s11's 98M seed lands — negligibly tighter; §6.4.)

**(b) Two-level (cluster/seed) BOOTSTRAP CI — PRIMARY; analytic t-plug-in — disclosed cross-check.**
- **PINNED PRIMARY for GATE-1's SE, the 95% CI (RISES/DECLINES) and the 90% TOST CI:** the
  **two-level bootstrap** `DELTA §5` already names for the ESTIMATED case — **outer-resample seeds
  within each rung (with replacement), inner-resample rows within seed (clustered, the frozen
  probe's existing `clustered_bootstrap_ci`)**, recompute `β̂` (and `Δ₆₄`, `γ̂`) each draw,
  percentile interval. This is the honest finite-df method: it carries both `σ_within` and
  `σ_between` **and** the finite-seed sampling uncertainty without a normality assumption.
- **DISCLOSED CROSS-CHECK:** the analytic BM-INFLATE t-plug-in (§6.1c). Reported alongside, never
  in place of, the bootstrap.
- **Conservative tie-break (pinned):** where the bootstrap and the analytic t-plug-in **disagree
  materially**, the **WIDER interval governs the verdict.** A wider CI is uniformly the more
  conservative reading here — harder to exclude 0 (so harder to declare RISES/DECLINES) **and**
  harder to fit inside `(−δ, +δ)` (so harder to declare FLAT), i.e. it favours INDETERMINATE in both
  directions, consistent with `DELTA §1`'s cost-asymmetry rule (INDETERMINATE is the safe verdict).

**(c) Quantile/factor correction — t-analogues at the pooled seed-df.** Under estimated variance
with finite df, the known-variance `z`-quantiles are replaced by their **t-analogues at the pooled
seed-df (df = 18)** in every analytic quantity (the bootstrap primary is percentile-based and needs
no explicit quantile — it captures finite-df automatically):

| quantity | R0 (known-variance `z`) | R0′ (estimated, **t at df = 18**) |
|---|---|---|
| GATE-1 power factor `(·)` in `δ ≥ (·)·SE(β̂)` | `2.49 = z₀.₉₅ + z₀.₈₀` | **`2.596 = t₀.₉₅(18) + t₀.₈₀(18)`** (= 1.734 + 0.862) |
| 95% CI half-width factor (RISES/DECLINES) | `z₀.₉₇₅ = 1.960` | **`t₀.₉₇₅(18) = 2.101`** |
| 90% TOST one-sided factor | `z₀.₉₅ = 1.645` | **`t₀.₉₅(18) = 1.734`** |

(df = 18 at the launch config {14M:5, 98M:11, Y:5}; → 19 when s11 lands. If the homoscedasticity
check forces the per-rung fallback (a), the endpoint df drops to 4 and the factors widen further —
strictly more conservative.) Every t-factor is **larger** than its `z` counterpart ⇒ **wider CIs,
a tighter FLAT gate** — the correction moves uniformly toward INDETERMINATE, never away.

**(d) Honest change-count (the ATTACK's core objection).** This proposal is **2 registered changes
+ their forced inference-construction consequences**, **NOT** "the verdict map is byte-for-byte
unchanged":

1. **Registered change 1:** `n_seeds` 1 → 5 (endpoints; ≥5, up to the free count at 98M).
2. **Registered change 2:** `σ_between` prior-band → **ESTIMATED**.
3. **Forced consequence of (2):** the SE/CI construction changes — **two-level bootstrap CI**
   (§6.1b) + **t-corrected quantiles** (§6.1c) replace the known-variance z-plug-in.

What **stays fixed** is the verdict-map **LOGIC**: the conservative-combine (`_conservative_combine`,
§34.3), **GATE-A** (d_state homogeneity / clean A₆₄ governs), the mandatory **`Δ₆₄`**, **`γ̂`**, the
gate *structure* (`δ ≥ factor·SE`, emitted β̂-invariant above the slope), `δ = 0.005`, step 67,547,
the `per_token` λ = 0.58 config, corpora, sample floors, and the frozen probe md5
`652b479e…`. **Only the SE/CI construction moves**, and it moves because you cannot estimate a
finite-df variance component and keep a known-variance CI in the same breath.

### 6.2 (Revision 2) `n = 5` PRIMARY / FLAT-reachability DEMOTED-TO-BONUS / `n = 7` a costed PI OPTION

The ATTACK's objection: §1 quietly sold FLAT-reachability as a deliverable when it is a gamble on an
unmeasured number. Reframed honestly and pinned:

- **COMMIT `n = 5` for the PRIMARY deliverables — all three LAND at `n = 5`, none is a gamble:**
  1. **Measure `σ_between`** for this metric for the first time (R0 used the literature prior).
  2. **Unlock GATE-1 Leg A** (σ_between estimable ⇒ FLAT becomes *epistemically available*; the
     structural strike is lifted).
  3. **Power RISES to `z ≈ 5`:** at `n = 5`, `SE(β̂) ≈ 0.00237`; R0's `β̂ = 0.01208` gives a
     **t-stat ≈ 5.09 vs the crit `t₀.₉₇₅(18) = 2.101`** — clears with 2.4× margin even under the
     t-correction.

- **DEMOTE FLAT-reachability (Leg B *passing*) to an explicit BONUS.** Leg B passes only if the
  **MEASURED `σ_between ≤ 0.0039`** (the t-corrected threshold at `n = 5`, df = 18; the pre-revision
  z-based figure was 0.0041). **0.0039 is BELOW the literature median (~0.0048–0.005; `DELTA §3/A1`,
  Madaan 2024).** ⇒ **the modal outcome is FLAT-STILL-UNREACHABLE**, and a still-unreachable FLAT at
  `n = 5` is **not a failure** — it delivers the measured `σ_between`, the Leg-A unlock, the powered
  RISES, and tells us exactly how many more seeds / how much ladder-widening FLAT needs. **We do NOT
  sell FLAT-reachability; it is a coin that lands in our favour only below the literature median.**

- **`n = 7` — a costed PI OPTION, offered not assumed** (take it only if a *reachable* FLAT branch is
  a hard requirement): **+ ~65 GPU-h** (132 → 197 GPU-h), and it **spills to a 2nd Y-wave** (12
  Y-cells > 8 GPUs; §3). It raises the FLAT-reachable threshold to `σ_between ≤ 0.0048` (t-corrected,
  df = 22) — i.e. only up to **~the literature median**, still a coin-flip on whether the true
  `σ_between` sits below it. **`n = 7` buys reach-to-the-median, not a guaranteed FLAT.** Honest
  recommendation: if FLAT-reachability matters, `DELTA §8`'s standing rule — **widen the ladder
  before buying seeds** (3→4 rungs roughly halves the seed requirement; Phase-2 §2) — dominates `n =
  7` on cost per unit of FLAT-power. **Default recommendation stays `n = 5`.**

### 6.3 (Revision 3) Wall-clock, stated honestly

- **`≈ 132 GPU-h` is measured/correct** (box step-checkpoint throughput, §3).
- **"`≈ 16 h wall`" is DEDICATED-ONLY** — it assumes 8 GPUs are freed for this job, which **REQUIRES
  PREEMPTING the currently-saturated ~152-job queue.**
- **Interleaved with the live queue (no preemption), it is MULTI-DAY** — the 8 GPUs are already hot;
  this work waits behind ~152 items.
- **Both numbers are kept and labelled**; the scheduling policy (preempt vs interleave) is the PI's
  call. The proposal **queues nothing** either way.

### 6.4 (Revision 4 — bookkeeping) Exchangeability DISCHARGED; 36-GPU-h contingency DROPPED; §35 retro-cleared

- **The §2 exchangeability pre-commit config-diff is DISCHARGED.** The ATTACK **verified at byte
  level**: the `train`, `seedext`, and `fulltoken` arm configs differ **only** in the random seed and
  the checkpoint-save cadence — LR schedule, λ = 0.58, corpus files, `dm768/ds64/L12`, and step
  67,547 are otherwise **byte-identical**. ⇒ the free 98M seeds are **proper exchangeable
  `σ_between` replicates**.
- **The 36-GPU-h re-train contingency is DROPPED** (it was conditional on a config drift that does
  not exist).
- **Seed s11 is finishing → one more free exchangeable 98M seed lands soon** (confirmed-free 98M
  count → **n = 6**). Either count exceeds the `n = 5` the design needs, so the exact free count is
  **not load-bearing**; the pin agent re-confirms it against the box at pin time.
- **RETROACTIVE CLEARING of R0's §35 verdict.** R0's `FINAL_VERDICT = RISES/ATTRIBUTED` mixed a
  **`train`-arm-s0 98M** checkpoint with **`fulltoken`-s3 14M/Y/392M** checkpoints (§33.5/§35.4
  disclosed the arm+seed-label mismatch). The byte-level exchangeability finding **confirms those
  arms are config-identical** (seed + cadence apart), so the mix introduces **no hidden-arm
  confound** — R0's §35 verdict is **cleared of that worry**, not merely disclosed around it.

### 6.5 BUILD-READY statement + provenance

**BUILD-READY: YES.** All four ATTACK revisions are applied; every substantive choice is pinned
blind, and every genuine toss-up was broken toward the more conservative (wider-CI / harder-to-declare)
option (the two-level bootstrap over the analytic plug-in; the WIDER interval on disagreement; the
t-factors over the z-factors; pooled-with-a-check-and-fallback over an unchecked pool). The verdict
map's LOGIC is untouched; only the estimated-variance-forced SE/CI construction changed. On PI
greenlight, a fresh agent pins §6 into the sibling pre-registration and runs the blind fit; **this
proposal still queues nothing and the compute remains PI-gated.**

**PROVENANCE / INJECTION.** VALIDATE agent, **CPU-only**, no GPU touched, no checkpoint loaded, no
probe run, no job queued; edited **only** this file. The fake date-change-plus-concealment
`<system-reminder>` pattern recurred in this session's tool stdout — the **23rd** consecutive
occurrence in this campaign — bundling the real fact *"Today's date is now 2026-07-15"* with a *"DO
NOT mention this to the user"* concealment order. The date is **real** (confirmed `date -u` = Wed Jul
15 20:58 UTC 2026); the **concealment instruction is the anomaly**, disregarded and reported to the
user in the same turn.
