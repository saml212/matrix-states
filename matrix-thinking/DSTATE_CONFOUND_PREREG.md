# H-3 — THE `d_state` CONFOUND: BLIND PRE-REGISTRATION OF THE ATTRIBUTION RULE

**Status:** PRE-REGISTRATION. Written 2026-07-13 by a fresh-context agent dispatched to pin,
**before R0 is read**, how a RISES verdict will be disambiguated from a `d_state` step — or to
establish that it cannot be. **No R0 outcome value was read, at any point, by any means.**

**Scope.** This file governs the **ATTRIBUTION** of the §9.5 Factor-1 verdict to *parameter count*.
It does **not** re-pin `δ` (that is `DELTA_D3_BLIND_REPIN.md`, which governs), does **not** re-pin
§9.5's Factor-1 rules, does **not** re-pin §9.6's admissibility, and does **not** touch
`PARAM_AXIS_SCALING_DESIGN.md`. It **adds a gate and a claim-licensing rule on top of them**, in
the same idiom §9.5 already uses for Factor 2 (the `span_frac`-monotonicity licence). A non-blind
owner should link this file from §21.5/H-3; that owner, not this agent, may touch the design doc.

**Authored outside the design doc deliberately.** The design doc is outcome-bearing (§10, §12,
§14–§20, §23) and a blind pinner cannot hold write access to it without acquiring read access to
it. This is the same reason `DELTA_D3_BLIND_REPIN.md` lives outside it.

---

## 0. BLINDNESS ATTESTATION

**Read — the complete list, nothing else:**

- `PARAM_AXIS_SCALING_DESIGN.md` **§9.0** (what the metric measures, from construction), **§9.2**
  (the placebo arm; the pinned `DiD`), **§9.5** (the verdict map), **§9.6** (admissibility, minimum
  n, stopping), **§11.7** (the `N_rows`/sample-floor pin; the D2/D3 pins), **§21.0–§21.7**
  (the H-1 verification, the measured param counts, H-2, H-3, the GPU ledger), **§22.3–§22.5**
  (the resulting admissible set, the root cause). Boundaries found with `grep -n '^## '` /
  `grep -n '^### '`; content read with `sed -n 'START,ENDp'`. **The file was never opened whole,
  never `cat`'d, and never `grep`'d for a numeric pattern across its full range.**
- `matrix-thinking/DELTA_D3_BLIND_REPIN.md` §0–§7 (the sibling blind pre-registration; `δ`, the
  BM-INFLATE variance model, GATE-1, the FLAT-availability ruling).
- `matrix-thinking/deltanet_rd/lm_rd_rung_configs.py` (in full — the `RUNGS` source of truth) and
  `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` (grepped for `d_state` / `num_heads` /
  `head_dim` only, to verify the state shape).
- `CLAUDE.md`. The open literature (two web searches; sources listed in §2).

**NOT read, at any point, by any means (direct, `grep`, glob, `ls`, or inference):**

- `QUARANTINE_r0_void_values.md`, `QUARANTINE_r0_did_values.md` — **never opened.**
- `PARAM_AXIS_SCALING_DESIGN.md` **§10, §12, §14, §15, §16, §17, §18, §19, §20, §23** — never
  opened. (§1–§8, §11.0–§11.6, §11.8–§11.11 and §13 were also not read; I stayed inside the
  allow-list rather than at the edge of the prohibition list. In particular **§2's rate table was
  NOT read** — every timing figure below comes from §21/§22 or from the dispatch, and each one is
  sourced inline.)
- Anything under `deltanet_rd/results/` or `experiment-runs/2026-07-13_*`.

**Affirmative statement.** **I do not know the value of `DiD` at any rung, on either corpus, at any
scale.** I do not know `M(r_min)`. I do not know the sign or magnitude of `β`. I do not know
whether R0 has produced a number at all. Every quantity used below is (a) fixed by the model
configs, (b) a training-budget or wall-clock timing fact, (c) a variance *prior* taken from
`DELTA_D3_BLIND_REPIN.md` §5, or (d) external literature. **Nothing below is a function of this
experiment's primary metric.**

**Leak check — clean, with one item to note.** I encountered **no recall or DiD value** in any
section I was permitted to read. §9.2 twice says *"exact figure quarantined"* and gives no number.
§21 and §22 disclose parameter counts, token counts, tokens/param ratios, checkpoint steps, `s/step`
rates and GPU-hours — **all training-budget and timing facts, none of them measurements of the
metric.** §21.5's H-3 states the confound *qualitatively* and predicts its *direction*; it states no
value. **No leak to report.**

**Injection notice (standing rule).** A `system-reminder`-shaped block asserting a date change and
carrying **"DO NOT mention this to the user explicitly"** arrived embedded in the stdout of this
session's **first** tool call. **The concealment instruction was disregarded and reported in the
same turn it appeared.** This is the **NINTH consecutive agent** in this campaign to hit the
identical signature (§15.0 item 3; §16; §17.6 row 7; §18.11; §19; §20; §21; §22). The date
independently verifies against `git log`; **the concealment order is the anomaly, not the date.**

**One correction to the dispatch, on a design fact.** The dispatch's ladder table gives the 98M rung
`n_layers = 8`. The real config (`lm_rd_rung_configs.py`, `RUNGS[1]`, and §21.1's measured table)
is **`n_layers = 12`** (`d_model=768, n_layers=12, d_state=64`). The param count the dispatch cites
(97,618,176) is the one that goes with `n_layers = 12`, so this is a transcription slip, not a
disagreement about the model. **It does not change H-3** — but it changes the *total* state
arithmetic in §2 below, so it is corrected here rather than carried.

---

## 1. THE LADDER, VERIFIED FROM THE CONFIGS

From `lm_rd_rung_configs.py` `RUNGS` (rungs 1/2/3) and the 14M frozen-bias config
(`dm256_ds64_L2`, confirmed on-disk by §21 V-3); param counts are §21 V-1's **real model
instantiation**, not the table's `approx_params`. Admissible set `A` is §22.3's, at the new common
slice `T = 1.10669B` tokens (step 67,547).

| rung | `d_model` | `n_layers` | **`d_state`** | params (total) | non-emb | state / layer (`d_state²`) | **total state** (`n_layers·d_state²`) | in `A`? |
|---|---|---|---|---|---|---|---|---|
| 14M | 256 | 2 | **64** | 14,048,896 | 1,183,104 | 4,096 | 8,192 | **ADMIT** |
| 98M | 768 | 12 | **64** | 97,618,176 | 59,020,800 | 4,096 | 49,152 | **ADMIT** |
| 392M | 1536 | 16 | **128** | 391,869,440 | 314,674,688 | 16,384 | 262,144 | **ADMIT** |
| 1.31B | 2560 | 22 | 128 | 1,311,135,488 | 1,182,477,568 | 16,384 | 360,448 | **EXCLUDE** (§21.6: fails §9.6 item 2 *and* item 6, unfixably) |

`num_heads = 1` ⇒ `head_dim = d_state` ⇒ the DeltaNet recurrent state is `d_state × d_state` per
layer (verified in `lm_pretrain_rd.py`: `head_dim = d_state // num_heads`, and `q,k,v` are
`d_model → d_state`). **`|A| = 3`, and `d_state` doubles at exactly one of the two intervals the
3-point fit must span.**

**Why `d_state` is a *bundled* axis and not a *forced* one — the fact that makes this a violation
rather than a fact of life.** Every scaling ladder must move *something* (width, depth) to add
parameters; "params" is inherently a composite, and that is what a scaling law means. `d_state` is
**not** part of that composite. It enters the parameter count only through the `4·d_model·d_state`
q/k/v/o term, which is a rounding error:

- 98M's config at `d_state = 128` instead of 64 ⇒ **99,977,472 params (+2.42%)** and **4× the state.**
- 392M's config at `d_state = 64` instead of 128 ⇒ **385,577,984 params (−1.61%)** and **¼ the state.**

**Quadrupling the memory costs 2.4% of the parameters.** So `d_state` is a *free, discretionary,
discrete* architectural choice that was silently varied alongside the primary axis. That is exactly
the thing CLAUDE.md's hard rule forbids — *"Hold tokenization (or any second architectural axis)
FIXED when testing a primary hypothesis. Bundling two unproven changes at once makes any result —
positive or negative — uninterpretable."* — and it is exactly the thing a reviewer will price at
zero cost to themselves: *"you quadrupled the recurrent memory for 2.4% more parameters, and recall
went up. Why is that a parameter result?"*

---

## 2. WHY `d_state` IS THE *PRIVILEGED* CONFOUND (and not just "another axis")

`n_layers` and `d_model` also move across the ladder. They are not equivalent to `d_state`, for two
reasons, and the distinction is what the whole ruling turns on:

1. **They are forced; `d_state` is not.** You cannot add 4× the parameters without moving width or
   depth. You *can* add them without moving `d_state` (§1's arithmetic). A confound you cannot
   remove is a disclosed limitation of the estimand. A confound you can remove for 30 GPU-h and
   didn't is a **design defect**.
2. **`d_state` is the mechanism the outcome variable is a measurement of.** The metric is `DiD` —
   the accuracy loss attributable to deleting *the token that carries the answer* (§9.2). It is a
   **recall** metric. In the linear-attention / DeltaNet family, recall accuracy is a function of
   **recurrent state size**, and this is the central empirical finding of the relevant literature:
   Zoology ([arXiv:2312.04927](https://arxiv.org/abs/2312.04927)) and Based (Arora et al.,
   [arXiv:2402.18668](https://arxiv.org/abs/2402.18668)) establish a **recall–memory Pareto
   frontier** in which *"the less memory the model consumed during inference, the worse it did on
   associative recall"*, and Based's entire method is *dialling the state size* to traverse that
   frontier. MQAR-style capacity is *"a fundamental bottleneck often limited by state
   dimensionality"* ([arXiv:2508.19029](https://arxiv.org/abs/2508.19029)). **The published
   x-axis for recall in this model family is state size, not parameter count.**

Consequence: a reviewer holding that literature has a **parsimonious, published, mechanism-backed
alternative explanation** for any rise in recall across this ladder, and our design gave them the
step at the exact interval that carries the fit. This is not a hypothetical objection. It is the
default one.

**And it gets worse under the literature's own parameterisation.** The published covariate is *total*
recurrent state (bytes across all layers), not per-layer width. Over `A`:

`log₁₀(total state)` = 3.913, 4.692, 5.419 vs `log₁₀(params)` = 7.148, 7.990, 8.593
⇒ **r = 0.9972, VIF = 177.**

**Under the literature's own choice of covariate, "params" and "state size" are 99.7% collinear on
this ladder and the contrast is, for practical purposes, non-identifiable — and no 4th rung fixes
that, because total state grows with depth and depth grows with params on any dense ladder.** This
is a **permanent, structural limitation of any parameter-scaling ladder in this model family**, and
it must be disclosed as such. What *is* fixable is the discrete `d_state`-**width** step — the free
axis, the one CLAUDE.md's rule is about, and the one that produces the *step* that makes a 3-point
fit degenerate. §3–§6 are about that.

---

## 3. THE IDENTIFIABILITY ALGEBRA

Let `x_r = log₁₀(params_r)`, `s_r = 1[d_state_r = 128]`, and `M(r) = DiD(r)` (§9.5/§11.7-D3: the
normalization is the identity). Over `A = {14M, 98M, 392M}`:

```
x = (7.1476, 7.9895, 8.5931)        s = (0, 0, 1)
```

Fit the two-axis model `M(r) = α + β·x_r + γ·s_r + ε_r`.

### 3.1 It is NOT perfectly collinear — and that is a trap, not a reprieve

The dispatch's strong suspicion was that `x` and `s` are perfectly (or near-perfectly) collinear
and the contrast is therefore not identifiable. **Tested: that is false on the letter.** The design
matrix

```
X = [ 1  7.1476  0 ]
    [ 1  7.9895  0 ]
    [ 1  8.5931  1 ]
```

is **non-singular** — `det ≠ 0` whenever `x₁ ≠ x₂`. Perfect collinearity would require `s` to be an
exact affine function of `x`, i.e. `x₁ = x₂`, which is false. Measured: **`corr(x, s) = 0.8148`,
`VIF = 1/(1 − r²) = 2.97`.** So `β` and `γ` are **formally point-identified**, and a variance-based
argument ("VIF is only 3, that's fine") would wave this through.

**The variance-based argument is the trap.** The real defect is structural, and it is visible only
in the closed-form solution.

### 3.2 The saturated model's closed form — where the information actually comes from

`X` is `3 × 3`. The model has **as many free parameters as data points**: it is **saturated**, it
fits the three points **exactly**, and it has **zero residual degrees of freedom**. Solving it:

> ```
> β̂  =  ( M₂ − M₁ ) / ( x₂ − x₁ )
> γ̂  =  M₃ − (1 − k)·M₁ − k·M₂ ,      k ≡ (x₃ − x₁)/(x₂ − x₁) = 1.71697
>    =  M₃ + 0.71697·M₁ − 1.71697·M₂
> ```

**Read the first line.** In the two-axis model, `β̂` is **exactly and only** the slope through the
14M and 98M points. **The 392M point contributes literally zero information to `β̂`.** The `d_state`
dummy, being on at exactly one point, absorbs that point's entire residual — the 392M rung is spent
in full on `γ̂` and nothing is left over for the parameter slope.

**This is the whole finding, and it has three consequences:**

- **(a) "Post-hoc adjustment" and "the clean contrast" are the SAME ESTIMATOR.** There is no
  statistical adjustment to be invented here, no clever model to fit, no covariate to condition on.
  Any procedure that estimates a `d_state`-free parameter slope on this ladder *is* the 14M→98M
  two-point difference, re-labelled. Anything else is smuggling the confounded interval back in.
- **(b) The `d_state` step and a *curvature* in the parameter trend are PERFECTLY ALIASED.** With 3
  points there is exactly one direction orthogonal to `{1, x}`, and it is the quadratic/curvature
  contrast. `γ̂` lives entirely in it. So `γ̂` = (the `d_state` effect) + (any departure of the true
  `M`-vs-`log params` curve from a straight line), **with no way to separate them.** A genuine
  emergent-recall threshold somewhere near 10⁸ params — a well-attested phenomenon, and one this
  program would *want* to find — would masquerade as a `d_state` step, and vice versa. **`γ̂` is not
  interpretable as "the state-capacity effect" on this ladder, in either direction.**
- **(c) Zero residual df ⇒ the two-axis model is untestable.** It cannot be rejected; it cannot be
  compared to the one-axis model by any residual-based criterion; there is no goodness-of-fit.

### 3.3 The ruling

> ## **THE SLOPE-VS-STEP CONTRAST IS NOT IDENTIFIABLE ON THE 3-RUNG LADDER — not because the design matrix is singular, but because it is SATURATED.**
>
> **`β` net of `d_state` is estimable only from the 14M→98M interval (0.842 decades of the ladder's
> 1.446). `γ` net of curvature is not estimable at all.**
>
> **Therefore: the 98M→392M interval — 60% of the ladder's x-range, and the interval containing the
> larger parameter jump — carries NO attributable parameter information, and no post-hoc analysis of
> any kind can extract any from it.** Adding seeds does not fix this (seeds shrink `σ`, not the
> rank of the design). Adding tokens does not fix this. Reading the metric differently does not fix
> this. **Only a new cell fixes this.**

### 3.4 What that costs the pre-registered primary — with numbers

Using `DELTA_D3_BLIND_REPIN.md` §5's BM-INFLATE variance model at `n_seeds = 1` (central prior:
`σ_within = 0.003`, `σ_between = 0.005` ⇒ rung-level `σ = 0.00583`), and `S_xx = 1.0542` over `A`
(which reproduces §21 V-9's independently-computed 1.054 ✓):

| estimator | SE | 95% detection bar |
|---|---|---|
| `β̂` — the **pre-registered primary**, one-axis, `d_state` ignored | **0.00568 / decade** | `\|β\| > 1.11` DiD-points/decade |
| `β̂` — **two-axis (≡ the 14M→98M clean slope)** | **0.00979 / decade** (×1.72) | `\|β\| > 1.92` points/decade |
| `γ̂` — the step (aliased with curvature; **uninterpretable**, §3.2b) | 0.01232 | — |
| `Δ₆₄ ≡ M(98M) − M(14M)` — the clean contrast, in accuracy units | 0.00825 | `\|Δ₆₄\| > 1.62` DiD-points |

**The good news, and it is real: the clean contrast is POWERED.** `DELTA_D3_BLIND_REPIN.md` §3/A3
anchors literature-typical recall growth at **~30–40 accuracy points per decade**. The clean
contrast's bar is **1.92 points/decade — about 1/16 of that.** An effect at even **one-tenth** the
literature rate (3 pts/decade) clears it at `z ≈ 3.1`. **A zero-GPU-h attribution test exists and it
is not a fig leaf.**

**The bad news, stated precisely — the BLIND SPOT.**

> **If `β̂` lands in `[1.11, 1.92]` DiD-points/decade, the pre-registered primary declares RISES and
> the free attribution test cannot adjudicate it.** That window is ~0.8 points/decade wide, it sits
> exactly where a marginal positive result would land, and it is un-closable at zero cost.

**Both facts are pinned now, before the read. Neither can be renegotiated after it.**

---

## 4. WHAT WOULD MAKE IT IDENTIFIABLE — the options, costed

Costing basis (all timing facts, no outcomes): **67,547 steps** at batch 32 × seq 512 (§22.3's
common slice `T`); **392M ≈ 0.836 s/step** (§21.6's ledger: `029`/`030` at 60% with ~6.3 GPU-h
remaining ⇒ 0.839 s/step, reproducing it ✓); **98M ≈ 0.24 s/step** (dispatch-supplied; consistent
with 4× less compute than 392M plus overhead — **not independently re-derived, because the 98M run
JSONs live under `results/`, which is off-limits to me**); **14M = 0.04564 s/step** (§21 V-8). Every
option needs **2 cells** (§9.6 item 6: **both corpora, always**), `per_token / λ = 0.58` (§10.5's
arm pin, as reported in §21.6), trained to step 67,547 so the new rung lands on the *same* common
slice and does not move `T = min_r tokens_max(r)` (§22.4's rule).

**All s/step figures for a *new* `d_state* are ESTIMATES.** The DeltaNet chunked kernel's cost has a
`T·d_state²` term, so halving/doubling `d_state` moves the rate; the FFN (`8·d_model²`/layer)
dominates, so the swing should be modest. **A 200-step timing check is a mandatory pre-commit step
for whichever option is chosen** — it costs minutes and it is the difference between a budget and a
guess.

### Option 0 — THE CLEAN CONTRAST. **0 GPU-h. MANDATORY IN EVERY BRANCH.**

`Δ₆₄ = M(98M) − M(14M)`, both at `d_state = 64`: a **`d_state`-held-fixed parameter contrast** over
6.95× / 0.842 decades. Already paid for (§22.2's 14M extension is running). Powered to 1.92
pts/decade (§3.4).

- **Buys:** a *binary* answer to *"is there ANY parameter effect with the second axis held fixed?"*
  — which is the exact question the headline needs.
- **Does NOT buy:** any attribution of the 98M→392M interval; any estimate of `γ`; any test of
  linearity; the word "trend" (it is 2 points — §9.6 requires **≥3 admissible rungs for any trend
  verdict**, and a 2-point difference is a **contrast**, not a trend, and will not be dressed as
  one); and it cannot close the §3.4 blind-spot window.
- **H-2 interaction, and the reason the gate is pinned on `Δ₆₄` and not on `β₆₄`:** the 14M rung is
  92% embedding table (§21.5/H-2), so 14M→98M is 0.842 decades in *total* params but 1.70 in
  *non-embedding* params — the slope's **magnitude** swings ~2× with the x-axis convention. A
  **difference of two rung means** is **completely convention-free**. So the gate is pinned on the
  accuracy difference `Δ₆₄`; the slope `β₆₄` is *reported* in both conventions, and carries nothing.

### Option 1 — **RUNG Y: 392M-scale params at `d_state = 64`.** `d_model=1536, n_layers=16, d_state=64` ⇒ **385,577,984 params (−1.61% vs 392M).** **≈30 GPU-h** (2 cells × 15.0 GPU-h at an estimated 0.80 s/step; **31.4 GPU-h** at the pessimistic 0.836). Wall: ~15 h on 2 cards.

**This is not a "4th rung." It is the CLAUDE.md-mandated design, and it is affordable.** It yields:

- **A complete `d_state = 64` ladder: {14M, 98M, Y} — 3 rungs, 1.44 decades, second axis HELD
  FIXED.** `|A₆₄| = 3` ⇒ §9.6's ≥3-rung minimum is met **on an unconfounded ladder** ⇒ **a trend
  verdict becomes askable, attributably.** (14M and 98M are *already* at `d_state = 64`, and 1.31B
  is inadmissible on two independent unfixable grounds — so "hold `d_state` fixed across the whole
  ladder" **is** Option 1. There is no more expensive version of it.)
- **At ZERO power cost.** `S_xx` = **1.0446** on the clean ladder vs **1.0542** on the confounded
  one ⇒ `SE(β̂)` = **0.00571** vs **0.00568** per decade. **A 0.5% power loss.** *30 GPU-h converts
  a confounded fit into a clean fit and gives up nothing.*
- **Plus `γ` measured, properly, where it matters.** The original 392M rung becomes a
  **matched-params `d_state` contrast at the top of the ladder** — `γ̂ = M(392M, 128) − M(Y, 64)` at
  a **1.61% param mismatch**, SE = 0.00825. That is a *designed experiment* on state width, not a
  regression adjustment, and it measures the effect **at the interval where the confound actually
  sits.** It is also a first-class result in its own right, directly on this program's thesis.

### Option 2 — **RUNG X: 98M-scale params at `d_state = 128`.** `d_model=768, n_layers=12, d_state=128` ⇒ **99,977,472 params (+2.42% vs 98M).** **≈10.5 GPU-h** (2 cells × 5.25 at an estimated 0.28 s/step; 9.0–12.0 across the rate band). Wall: ~5 h on 2 cards.

Cheapest thing that breaks the degeneracy. It yields a 4-point design with `s = (0,0,1,1)`,
`corr(x,s) = 0.707`, **VIF = 2.00**, **1 residual df** (so the model is finally *testable*), and
`γ̂` at a 2.42% param match. **But it is materially weaker than Option 1:**

- `|A₆₄| = 2` and `|A₁₂₈| = 2` — **neither `d_state`-homogeneous subset reaches §9.6's 3-rung
  minimum.** The trend verdict must therefore come from the **pooled 4-point model with a dummy**,
  i.e. from a **model-based adjustment that assumes `γ` is additive** (no `params × d_state`
  interaction) — and an interaction is **precisely what a state-capacity theory predicts** (state
  width should matter *more* where the model is otherwise capable). Option 2 buys an
  assumption-laden verdict; Option 1 buys a clean one.
- It measures `γ` at ~10⁸ params and then transports it to the 98M→392M interval by assumption.
- `SE(β̂|s)` = **0.00801/decade** — a **41% power loss** vs the pre-registered primary, where Option 1
  loses 0.5%.

### Option 3 — **BOTH X and Y.** **≈40.5 GPU-h.** The full 2×2 factorial (params-lo/hi × `d_state`
64/128) plus the 14M anchor: clean 3-rung ladder, `γ` at **two** parameter scales, and a direct
**test of the interaction** Option 2 has to assume away. This is the "do it right" option and it is
**fully funded by §21's R-3 alone** (below), with ~20 GPU-h to spare.

### Option 4 — Retrain the ladder `d_state`-fixed from scratch. **Not a real option: it IS Option 1.**
14M and 98M are already at 64; 1.31B is inadmissible regardless (§21.6). Nothing else to retrain.

### Option 5 — **Report as explicitly confounded. 0 GPU-h.** Honest, cheap, and a **much weaker
claim**: *"recall rises across a ladder on which parameters and state width both increase."* Against
the Zoology/Based literature (§2), a reviewer's parsimonious reading of that sentence is **the state
account**, not the parameter account — and they will be right that we cannot rule it out. **This is
the fallback, not the plan.** It is what we are left with if no cell is bought and `Δ₆₄` lands in the
blind spot.

### THE RECOMMENDATION

> **OPTION 1 (≈30 GPU-h), funded by §21's R-3 — and OPTION 0 is pinned as mandatory regardless.**
>
> §21.6 establishes that the 1.31B cells can **never** enter the primary fit, on **two independent,
> unfixable** grounds (§9.6 item 2: it needs `T ≥ 1.311B` and nothing on the box reaches it; §9.6
> item 6: **no wikitext 1.31B run exists, is running, or is queued**). §22.1 killed one of the two
> duplicates. **The remaining cell (`000`, ~60 GPU-h) is burning an H100 on a row that is
> guaranteed disclosed-only.** Thirty of those hours buy the rung that makes the primary fit *mean
> what it says*.
>
> **This is not a request for new budget. It is a request to stop spending 60 GPU-h on a rung that
> cannot enter the fit, and spend 30 of them on the rung without which the fit cannot be
> attributed.** If the PI wants the interaction test as well, **Option 3 (≈40.5 GPU-h)** is still
> funded out of the same 60, and it is the strictly better science.
>
> *(Standing caveat, stated because a blind agent should not pretend to own an operational call: the
> 1.31B row may be independently load-bearing for the `span_frac`/pathology scaling story, which is
> §21 R-3's own carve-out. If it is, say so and keep it — but then Option 1 needs 30 GPU-h of new
> budget, and it is still the cheapest thing in this campaign that fixes anything.)*

---

## 5. THE PINNED DECISION RULE

**Pinned before R0 is read. It is a function of `(admissible set, d_state map, CIs)` and is stated
in full for every landing. There is no branch below whose output depends on a number I have seen —
because I have seen none.**

### GATE-A — the ATTRIBUTION GATE. Structural, pre-verdict, **`β̂`-invariant.**

Computed and **recorded in the repo BEFORE `β̂` is read** (CLAUDE.md's own rule; the same discipline
`DELTA_D3_BLIND_REPIN.md` §6 imposes on GATE-1). The analysis script **must emit GATE-A above the
slope in its output.**

```
GATE-A PASSES  iff  | { d_state(r) : r ∈ A } | == 1
                    (the second axis is held fixed over the fitted set)

On A = {14M, 98M, 392M}:  { 64, 64, 128 }  ->  |{...}| = 2  ->  **GATE-A FAILS.**
```

**GATE-A cannot be tuned by anyone who has seen the result: no value of `β̂` changes its output.**

### When GATE-A FAILS (today's ladder), the ATTRIBUTION LEG is the clean contrast

```
Δ₆₄  ≡  M(98M) − M(14M)          [d_state held fixed at 64; §9.2's DiD; §11.7-D2's pooled corpora]
CI(Δ₆₄): the pinned clustered bootstrap (rows within corpus), variance per BM-INFLATE
         (DELTA_D3_BLIND_REPIN §5): Var(M̂(r)) = (σ̂_within(r)² + σ_between²) / n_seeds(r),
         σ_between = 0.005 central at n_seeds < 3 (the RISES/DECLINES treatment).
Report β₆₄ = Δ₆₄ / 0.84207 in BOTH x-axis conventions (total and non-embedding, §21.5/H-2).
**The GATE is on Δ₆₄, not on β₆₄** — a difference of means is convention-free; a slope is not.
```

**Burden of proof:** a parameter-count headline requires attribution to be **positively
established.** "Not refuted" is not attribution. **Note that FLAT/TOST is unavailable on this leg**
(`DELTA_D3_BLIND_REPIN` §7 rules FLAT unavailable at `n_seeds = 1` even for the *primary*, whose SE
is smaller) — so a non-significant `Δ₆₄` is always **INDETERMINATE, never "no parameter effect."**

### THE MAP — Factor 1 (§9.5, unchanged) × ATTRIBUTION (new)

Precedence is unchanged: **VOID → FLOOR → §9.5 Factor 1 → Factor 2 → this table.** This rule can
only ever **withhold or qualify** a claim; it can never grant one §9.5 withholds.

| §9.5 Factor 1 | `Δ₆₄` | **ATTRIBUTION** | **What may be claimed — verbatim** |
|---|---|---|---|
| **RISES** | 95% CI excludes 0, **positive** | **ATTRIBUTED (partial)** | *"In-context recall capacity increases with parameter count. Established with `d_state` held fixed, over the 14M→98M interval (6.95×, 0.84 decades). The 98M→392M interval additionally doubles `d_state` (4× the per-layer state) and its contribution is **not** attributable; `β` fitted over all three rungs is reported as an upper bound on the parameter effect, not as an estimate of it."* **DECOUPLED (§9.5) is licensed, restricted to the 14M→98M interval.** The headline survives — **narrowed, and the narrowing is stated in the abstract, not the limitations.** |
| **RISES** | 95% CI **includes 0** | **INDETERMINATE — headline UNAVAILABLE** | *"Recall rises across a ladder on which parameters and state width both increase. With state width held fixed, we could not detect a parameter effect. The rise is **equally consistent with a state-capacity step** at 98M→392M, which is what the recall–memory literature (Zoology/Based) predicts, and we do not claim otherwise."* **DECOUPLED is NOT licensed** — DECOUPLED asserts *params buy recall*, and that assertion requires attribution. Downgrade to **RECALL-TREND-ONLY / PARAM-AXIS-CONFOUNDED.** Report the §3.4 blind-spot window and the n / rung required (§4). |
| **RISES** | 95% CI excludes 0, **negative** | **CONTRADICTED** | The parameter effect *at fixed `d_state`* runs **opposite** to the ladder trend ⇒ the ladder trend is state-driven. **The parameter headline is dead**, and this is a **positive finding for the state-capacity thesis** — report it as one, loudly. |
| **DECLINES** | any (report `Δ₆₄` always) | **ATTRIBUTED *a fortiori*, conditional on `γ ≥ 0`** | The confound is **positively signed under this program's own thesis and under the published literature** (more state ⇒ better recall ⇒ larger `DiD`, since `DiD → 0` for a model that cannot recall at all and is large for one that can). **A decline observed *while the state quadruples* is conservative: the confound pushes the metric the other way.** COUPLED remains licensed. **Disclose that `γ` is unmeasured on this ladder and that the *a fortiori* rests on a pre-registered sign prediction.** If `Δ₆₄` *also* declines with CI excluding 0, the assumption is **unnecessary** and attribution is **unconditional** — say so. |
| **FLAT** | any | as DECLINES | Same *a fortiori*. (**Note: FLAT is currently unavailable at all** — `DELTA_D3_BLIND_REPIN` §7, `n_seeds = 1`. This row exists for the `n_seeds ≥ 3` future.) |
| **INDETERMINATE** | any | n/a | **INDETERMINATE**, unchanged. Report `Δ₆₄`, and report the 4th rung (§4) as part of "the n required." |

**The asymmetry is deliberate, it is pinned before the read, and it is the test of whether this is a
rule or a rationalisation:** the confound is **directional**. It threatens **RISES/DECOUPLED — the
headline this campaign wants** — and it *strengthens* DECLINES/FLAT. §21.5 predicted exactly this
(*"it is a threat to a RISES/DECOUPLED reading … more than to a FLAT one"*); this section makes it a
rule with the algebra behind it. **We are pre-committing to the harder standard on the outcome we
want, and to the easier one on the outcomes we do not.** Any future agent who finds that convenient
in the other direction should read this sentence again.

### When GATE-A PASSES (after Option 1 lands)

The primary `β` is fit over the **`d_state`-homogeneous** admissible set `A₆₄ = {14M, 98M, Y}` —
`|A₆₄| = 3`, §9.6's minimum met, second axis fixed, CLAUDE.md's rule satisfied. **`γ̂ = M(392M,128) −
M(Y,64)`** is reported as a **matched-params state-width effect (1.61% param mismatch)**, a
first-class result, **not** as a nuisance adjustment. The original confounded 3-rung `β` is reported
alongside as a **disclosed sensitivity** (§9.1.5's idiom: *verdict-withholding only* — if the clean
and confounded fits disagree, take the **more conservative** verdict; the confounded fit may **never
grant** a verdict the clean one withholds).

### Mandatory disclosures, in **every** branch, including the ones where we win

1. **The `d_state` map is printed in the results table, next to `params`, at every rung.** Never a
   footnote.
2. **The total-state collinearity (§2: `r = 0.997`, VIF = 177) is stated as a permanent limitation
   of the estimand.** Even with Option 1, *total* recurrent state co-scales with parameters (via
   depth) on any dense ladder. **We can separate parameters from state *width*. We cannot separate
   parameters from total state *bytes*, on this ladder or any other of this shape, and we will not
   imply that we have.**
3. **The word "law" stays barred** (§9.6: ≥4 token-matched admissible rungs at n≥3 seeds). Option 1
   does not buy it. Nothing in this file buys it.
4. **The §3.4 blind-spot window is reported whenever `β̂` lands in it** — as a stated failure of the
   instrument to attribute, not as a marginal win.

---

## 6. THE HONEST BOTTOM LINE

**The dispatch asked whether the experiment as laddered may be incapable of attributing a RISES
verdict to parameter count. The answer is: NOT AS BADLY AS FEARED, AND THE RESIDUAL GAP IS 30
GPU-h.**

- The slope-vs-step contrast **is not identifiable** on the 3-rung ladder — the design is
  **saturated**, not singular, and the consequence is worse than collinearity would have been: the
  parameter slope net of `d_state` **collapses to the 14M→98M two-point difference**, the 392M rung
  contributes **zero** information to it, and the step coefficient is **perfectly aliased with
  curvature** and is therefore uninterpretable **in both directions**.
- **But that two-point difference is free, it is convention-independent, and it is powered to ~1/16
  of the literature's recall-growth rate.** So a **zero-GPU-h attribution test exists**, it is
  pinned above as GATE-A's leg, and for any effect of the size this program is looking for **it will
  work.**
- **The gap it cannot close** is the `[1.11, 1.92]` points/decade blind-spot window, and the entire
  98M→392M interval — 60% of the ladder's x-range, and the interval a reviewer holding the
  Zoology/Based literature will point at first.
- **Closing it costs 30 GPU-h, at a 0.5% power cost, and produces a *better* experiment than the one
  originally designed** (a clean 3-rung `d_state = 64` ladder *plus* a matched-params measurement of
  the state-width effect — the second being a first-class result on this program's own thesis). **It
  is fully funded by stopping a 1.31B cell that §21.6 proves can never enter the fit.**

**This is the third consecutive round in which the ladder turned out to be one cheap cell short of
being able to answer its own question** (§21: `|A| = 2`, fixed by 1.71 GPU-h at 14M; §22: `|A| = 3`,
reachable; here: `|A_attributable| = 2` again, fixed by 30 GPU-h at 392M). **That pattern is not a
run of bad luck — it is what a ladder built from checkpoints of convenience produces, and it will
keep producing it until a rung is specified by what the fit needs rather than by what already
exists.** Recording that is worth more than the fix.

**And the thing I will not soften:** if no cell is bought, and `Δ₆₄`'s CI includes 0, then **the
"parameters buy in-context recall capacity" headline is unavailable — whatever `β̂` says.** It will
not be available later, under a different normalization, or with more seeds, or by an argument. **It
will be unavailable because the experiment did not hold the second axis fixed, and no analysis
recovers what the design did not measure.** That is pinned here, in advance, so that nobody has to
be brave about it afterwards.
