# δ (§11.7-D3) — BLIND RE-PIN OF THE EQUIVALENCE BOUND

**Status:** PRE-REGISTRATION. Written 2026-07-13 by a fresh-context agent dispatched to re-pin
`δ` **blind** — i.e. without reading, and without having been told, any R0 outcome value.
**This section supersedes §11.7-D3 and the `δ` clause of §9.5.** Where this file and
`PARAM_AXIS_SCALING_DESIGN.md` disagree about `δ`, **this file governs.**

This file is authored **outside** `PARAM_AXIS_SCALING_DESIGN.md` deliberately: the design doc is
outcome-bearing (§10, §12, §14-§20) and a blind pinner cannot hold write access to it without
acquiring read access to it. A non-blind owner should link this file from §11.7-D3; that owner,
not this agent, is the one permitted to touch the design doc.

---

## 0. BLINDNESS ATTESTATION

**Read (the complete list, nothing else):**

- `PARAM_AXIS_SCALING_DESIGN.md` **§9.0** (what the metric measures, from construction),
  **§9.2** (the placebo arm; the pinned `DiD`), **§9.5** (the verdict map), **§9.6**
  (admissibility, minimum n, stopping), **§9.7** (Rev-2's contamination ledger, opening lines),
  **§11.7** (the `N_rows`/sample-floor pin and the D2/D3 pins) — reached by `grep -n '^## '` for
  boundaries and `sed -n 'START,ENDp'` for content. The file was **never** opened whole, never
  `cat`'d, never `grep`'d for a numeric pattern across its full range.
- `matrix-thinking/deltanet_rd/lm_recall_gap_probe_v2_rd.py` — module header, and the estimator
  functions (`stat_did`, `stat_gap_true`, `stat_gap_placebo`, `clustered_bootstrap_ci`,
  `finalize_cell`, the `hit_*` record construction). **Source only. No result JSON, no log, no
  checkpoint.**
- `CLAUDE.md` (the project instructions injected into my context).
- The open literature (via a search subagent explicitly forbidden from reading this repo).

**NOT read, at any point, by any means (direct, grep, glob, `ls`, or inference):**

- `QUARANTINE_r0_void_values.md`, `QUARANTINE_r0_did_values.md` — **never opened.**
- `PARAM_AXIS_SCALING_DESIGN.md` **§10, §12, §13, §14, §15, §16, §17, §18, §19, §20** — never
  opened. (§13 was not on the prohibition list; I did not read it anyway, having no need.)
  **§1-§8 and §11.0-§11.6, §11.8-§11.11 were also not read** — I stayed inside the allow-list
  rather than at the edge of the prohibition list.
- Anything under `deltanet_rd/results/` or `experiment-runs/2026-07-13_param_axis_t2a_attempt2/`.
- Any training loss, checkpoint metric, or rung measurement.

**Affirmative statement.** **I do not know the value of `M(r_min)`.** I do not know the value of
`DiD` at any rung, on either corpus, at any scale. I do not know the sign of `β`. I do not know
whether R0 has produced a number at all. Nothing in the derivation below is a function of any
measured value of this experiment's primary metric; every quantity I use is either (a) fixed by
the design's own construction (sample floors, arm structure, the metric's algebraic range), (b)
taken from published external literature, or (c) an explicitly-labelled *prior* on a **variance
component**, which is orthogonal to the sign and size of the effect by construction.

**Leak check.** I encountered **no outcome value** inside the sections I was permitted to read.
§9.2 twice states that a figure is *"exact figure quarantined"* and gives no number — the
quarantine discipline held. §9.6 item 2 discloses **tokens-per-parameter** ratios (0.25 at the
1.31B rung, ~23 at 14M) and §9.6 item 1 discloses the **common token slice** (0.328B). These are
**training-budget facts, not recall measurements** — the section says so in terms (*"derived from
the training budget, not from any measured recall value"*) — and I use them below only for
arithmetic about *admissibility*, never about the metric. **No leak to report.**

**Premises I could not verify blind, flagged for a non-blind checker (do not let these pass
silently — see §8/H-1):** that the rung ladder is `{14M, 98M, 392M, 1.31B}` and that the
parameter-count convention used in §9.6's own tokens/param arithmetic is the nominal/total count.
Both are taken from `CLAUDE.md` plus §9.6's internal arithmetic (0.328/1.31 = 0.250 ✓;
0.328/0.014 = 23.4 ✓ — both reproduce §9.6's stated figures exactly, which is what licenses the
convention). If either premise is wrong, §8/H-1 changes; **§4-§7 do not.**

---

## 1. WHAT `δ` GOVERNS — the estimand, the units, the decision, the cost asymmetry

**The estimand.** From the probe source (`stat_did`, line 1048), stripped of prose:

```
d_i  =  hit_placebo_ablated(i)  −  hit_true_ablated(i)        ∈ {−1, 0, +1}
M(r) =  DiD(r)  =  mean_i d_i                                 (§9.1: normalization = identity)
```

`hit_*` is a **0/1 exact-match indicator** (`logits.argmax(-1) == target`). Both arms are scored
on the **same candidate** `i`, so `M(r)` is a **paired difference of two Bernoulli rates**. Three
properties follow *by construction*, and they are the whole basis of everything below:

- **P1 — `M` is an absolute quantity on a fixed, construction-given scale:** accuracy, in
  `[−1, +1]`, practically `[0, 1]`. **It already has a natural unit** — one accuracy point, `0.01`.
  It needs no external reference to be interpretable, and *therefore needs no measured reference
  to be thresholded.*
- **P2 — its sampling variance is bounded a priori:** `Var(d_i) = π_d − M²  ≤  π_d  ≤ 1`, where
  `π_d` = the discordance rate (the fraction of candidates on which exactly one of the two ablation
  arms hits). This is a construction bound, available before any read.
- **P3 — the independent unit is the ROW,** not the candidate (§9.2, §11.7): `clustered_bootstrap_ci`
  resamples rows. Sample floors are pinned per (rung, corpus): **≥1,500 contributing rows and
  ≥8,000 resolved candidates**, `N_rows` expected 2048, `C_max = 8`; §11.7-D2 pools the two corpora,
  so the pooled cell floor is **≥3,000 rows / ≥16,000 candidates**, expected **≈4,096 rows /
  ≈32,768 candidates**.

**The decision.** `β` ≡ the OLS slope of `M(r)` on `log10(params)` over the admissible rungs `A`.
**`β` is in DiD-accuracy-fraction per decade of parameters.** `δ` is the TOST equivalence bound on
`β`, in the same units. §9.5 Factor 1:

| Factor-1 outcome | Rule |
|---|---|
| RISES / DECLINES | 95% CI of `β` excludes 0 |
| **FLAT** | 95% CI includes 0 **and** TOST at 90% CI establishes `\|β\| < δ` |
| **INDETERMINATE** | 95% CI includes 0 **and** TOST fails |

So `δ`, and `δ` alone, is the switch between **FLAT** ("params buy *nothing*" — a publishable
positive finding, the FLAT-COUPLED headline) and **INDETERMINATE** ("we could not tell" — no
finding, report the n required).

**The cost asymmetry, stated so it cannot be forgotten.**

- Wrongly declaring **FLAT** publishes a **false negative result**: a ceiling claim about
  parameter scaling that is not true, into a literature that will cite it. It is not
  self-correcting — nobody re-runs a null.
- Wrongly declaring **INDETERMINATE** costs **GPU-hours**. §9.2 prices the eval at *"under a
  minute of H100 time per cell."*

The costs are not within an order of magnitude of each other. **INDETERMINATE is the safe
verdict, and every tie in what follows is broken toward it.**

---

## 2. WHY `δ = 0.125 × M(r_min)` CANNOT GATE THIS VERDICT

Three distinct failures. The first is fatal on its own; I rank them honestly, because the
statistical one is *not* the biggest and I will not inflate it.

**F-1 (FATAL) — the bound's LEVEL is set by an outcome, so its stringency is a free parameter
that the data chooses.** `δ = 0.125·M(r_min)` is not a bound on `β`; it is a bound on `β` *scaled
by a number we are about to measure*. Its magnitude is unbounded a priori: over the metric's
construction range `M(r_min) ∈ [0, 1]`, the pinned `δ` ranges over `[0, 0.125]` — **a factor of
infinity, spanned by the very experiment it adjudicates.** Two consequences, in opposite
directions, both wrong:

- If the smallest rung reads **high**, `δ` is **generous**, and FLAT becomes cheap — the false
  headline, purchased by a number nobody chose.
- If the smallest rung reads **low** (a 14M model on non-modal in-context bigrams is not
  obviously far from the floor), `δ → 0` and **no TOST can ever pass** — FLAT becomes unreachable
  regardless of the truth, and the *reason* is invisible in the verdict.

The bound's stringency is thereby made **inversely proportional to the instrument's own
headroom**, which is backwards, and it is **decided after the fact** by a quantity the author of
the pin, if contaminated, already knows. This is exactly the condition RULE T forbids: *a
threshold may gate a verdict only if its null is fixed by construction.* The null here — "the
smallest slope that counts as nothing" — is fixed by **measurement**. It has no construction-derived
value, and its arbitrariness is free to run wherever the incentive points. **It voids the FLAT
verdict on its face, independent of what the data say.** Lakens (2017, *Equivalence Tests*, and
the 2018 *AMPPS* tutorial) states the governing principle plainly: an equivalence bound must be
**independent of the data being tested**, and the endorsed data-independent construction is the
*resource-based* one — the bound set by the **design's** power, from N, not from the result.

**F-2 (REAL, MODEST) — the bound and the estimate are computed from the same data, and they are
anti-correlated in the FLAT-ward direction.** `β̂ = Σ_r w_r·M(r)` with `w_r = (x_r − x̄)/S_xx`. For
the *smallest* rung, `x_min − x̄ < 0`, so `w_min < 0`. At 3 rungs (below), `w_min = −0.723`. So an
upward sampling fluctuation `ε` in `M(r_min)`:
- **lowers** `β̂` by `0.723·ε`, and simultaneously
- **raises** `δ` by `0.125·ε`.

Both push toward FLAT. The TOST margin `(δ − β̂)` has sensitivity `+0.848` per unit of `ε`, versus
`+0.723` under a fixed bound — the FLAT-ward noise coupling is inflated by **≈17%**, and the
TOST's nominal α is correspondingly broken. I record the size honestly: **17%, not 200%.** F-2
alone would be a flaw, not a fatality. It is F-1 that is fatal. (The bound's own sampling error,
`SD(δ̂) = 0.125·σ_M`, is genuinely second-order and I do not lean on it.)

**F-3 (UNSUPPORTED PREMISE) — §9.5's justification is a power claim, and no power calculation was
done.** §9.5 asserts `δ` is *"the smallest change this instrument's power can meaningfully
claim."* Nothing in §9-§11 computes the instrument's power. §7 below computes it, and the
assertion does not survive: at the design's actual n, the smallest slope this design can certify
as negligible is **larger** than the largest slope that would be scientifically negligible. That
inversion is the finding of this document.

---

## 3. THE DERIVATION — three independent anchors, none of them our data

I set `δ` as a **SESOI** (smallest effect size of interest), in **absolute** units, and then check
attainability separately (§7). This ordering is not cosmetic: it is the only ordering in which
noise cannot buy a verdict. **A `δ` defined as a multiple of the realized SE would make a noisier
experiment easier to call FLAT** — the same defect as D3, wearing a lab coat. `δ` is therefore
fixed *before* any SE is known, and the SE enters only through the TOST, where more noise can only
ever push toward INDETERMINATE.

### A1 — the seed-noise floor: an effect smaller than *retraining the same model* is not an effect

**Principle (pinned).** Parameter scaling "buys nothing" if **scaling the model 10× moves the
metric less than re-running the same model with a different random seed.** If a decade of
parameters is worth less than a seed, "params buy nothing" is a fair English description of the
world. This criterion is scientific, absolute, and — critically — **fixed by external literature,
not by our rungs.**

**Madaan et al. 2024, "Quantifying Variance in Evaluation Benchmarks" (arXiv:2406.10229), Table 1.**
Ten Llama-2-7B models, identical but for the **pretraining seed** (init + data order). Std across
seeds of the final score, in accuracy points:

| Benchmark | seed std | | Benchmark | seed std |
|---|---|---|---|---|
| HellaSwag | **0.21** | | SIQA | **0.55** |
| MMLU-Cloze | **0.22** | | MMLU | **0.57** |
| PIQA | **0.41** | | AGIEval | **0.77** |
| GSM8k | **0.41** | | ARC-C | **0.80** |
| *(COPA, n=100)* | *2.15* | | *(HumanEval, n=164)* | *1.11* |

The two small-item-count benchmarks (italicised) are excluded from the band: their inflated std is
a finite-eval-set artefact, and our metric carries **16,000-32,768 candidates** per cell, placing
it with the large-N benchmarks. The band across the eight comparable benchmarks is
**0.21 - 0.80 accuracy points**, median **≈0.48**.

⇒ **σ_between (seed-to-seed SD of an accuracy-unit LM metric at fixed params) ∈ [0.002, 0.008],
central 0.005.** This is used **twice** below, and it is the load-bearing external number in this
document: once to set `δ` (A1), once to inflate the CI (§5). *Disclosed limitation:* Madaan is
7B and downstream-benchmark, not 14M-392M and not a DiD. It is the best available anchor; §8 pins
what replaces it (a direct measurement) and how.

⇒ **A1 gives `δ ≈ 0.005 per decade` (0.5 DiD-accuracy-points per decade).**

### A2 — the resource-based bound (Lakens' method), and why it *refutes* rather than sets `δ`

Lakens' data-independent rule is: set the bound to the smallest effect the design has **power** to
detect. Computed from construction-fixed N (§7): at 3 rungs × 1 seed, that is
**≈0.012 per decade** — *larger* than A1's meaningful bound. Lakens is explicit that the
resource-based bound must **also** be defensible as meaningful; when it is not, the correct
conclusion is **not to loosen the bound** but to declare the design underpowered for the claim.
**A2 does not set `δ`. A2 is the proof that this design cannot support FLAT** (§7).

### A3 — the literature's own effect size: what a *real* recall-vs-params effect looks like

`δ` must not be so tight that FLAT is meaningless, nor so loose that a real effect is laundered
into FLAT. Anchor it against the only in-range published measurements of **recall-intensive
accuracy vs. parameters**:

**Arora et al. 2024/2025, BASED (arXiv:2402.18668), Table 1**, Transformer++, **363M → 1.33B**
(0.564 decades — a range that *contains* our ladder's top):

| Task | 363M | 1.33B | slope |
|---|---|---|---|
| SWDE (acc) | 57.97 | 76.50 | **32.9 pts/decade** |
| FDA (acc) | 58.00 | 80.47 | **39.8 pts/decade** |
| SQuAD (F1) | 27.18 | 43.47 | **28.9 pts/decade** |

*(Disclosed: these arms also differ in tokens (10B vs 50B), so this is a compute-scaling slope, an
**upper** anchor on a pure-parameter slope. It is used as an order-of-magnitude reference only.)*

⇒ Recall-type accuracy, when it grows with scale in this parameter range, grows at
**~30-40 accuracy points per decade**. A `δ` of **0.5 points/decade** is **~1.5%** of that: a FLAT
verdict at this bound rules out **≥98%** of the literature-typical recall-growth rate. That is a
strong, meaningful, publishable claim — and it confirms `δ = 0.005` is **not absurdly tight
relative to what a real effect looks like**, which is the failure mode a too-small `δ` would have.
It also confirms (§7) that the design **retains ample power for RISES**: a true effect at even
one-tenth the literature rate (3 pts/decade) clears the 95% CI by 3×.

**A1 and A3 converge; A2 marks the design's ceiling. The three are independent and none of them is
our data.**

---

## 4. THE PIN

> ## `δ = 0.005` — five thousandths of a DiD accuracy-fraction, **per decade of parameters**.
>
> Equivalently: **0.5 DiD-accuracy-points per decade.** Absolute. Fixed by construction and
> external literature. **It does not move if `M(r_min)` is 0.02 or 0.40. It does not move if the
> corpora disagree. It does not move if the instrument is noisy. It does not move at all.**
>
> **Total-change reading (the honest gloss, replacing §9.5's):** over an admissible ladder
> spanning `L` decades, a FLAT verdict certifies `|ΔM| < 0.005·L` across the whole ladder — e.g.
> **< 0.72 accuracy points over 1.447 decades**, **< 1.0 point over 2.0 decades**. §9.5's gloss
> (*"less than 25% across the ladder's ~2 decades"*) is **retracted**: it is a relative statement
> anchored to an outcome, and it is arithmetically tied to a 2-decade span the admissible ladder
> may not have (§8/H-1).
>
> **Relation to the superseded pin, stated so no one can claim I reverse-engineered it:** the new
> `δ` coincides with the old one **iff `M(r_min) = 0.04`**. I do not know whether `M(r_min)` is
> above or below `0.04`, and therefore I do not know whether this re-pin is *stricter* or *looser*
> than the one it replaces. **That ignorance is the point, and it is the property the old pin could
> not have.**
>
> **The superseded `δ_old = 0.125·M(r_min)` may be REPORTED, and is VERDICT-WITHHOLDING ONLY**
> (the §9.1.5 idiom): if the two bounds disagree, take the **more conservative** verdict. `δ_old`
> may **never** grant a verdict `δ` withholds. It exists in the record so a reader can see the
> distance between the two, and for no other purpose.

---

## 5. THE VARIANCE MODEL — `δ` alone does not fix the verdict; the CI must carry the right components

The pinned CI (§9.5) is *"a bootstrap resampled over **rows** (clustered) and over **seeds where
n > 1**."* At **n = 1 seed**, the seed term is empty and the CI carries **exactly one** variance
component:

| Component | What it is | In the pinned CI at n=1? |
|---|---|---|
| **σ_within** | candidate/row sampling within a rung | **YES** (`clustered_bootstrap_ci`) |
| **σ_between** | model-to-model variation at fixed params (seed, init, data order) | **NO — omitted entirely** |
| **σ_curv** | departure of the true trend from linear-in-log-params | **NO — 1 residual df at 3 rungs; unestimable** |

Magnitudes, from construction and literature:

- **σ_within** `= σ_d·sqrt(DEFF / N_cand)`, with `σ_d = sqrt(π_d − M²) ≤ 1` (P2),
  `DEFF = 1 + (m−1)·ρ`, `m = C_max = 8`, and pooled `N_cand ∈ [16,000, 32,768]` (P3). Over the
  plausible ranges (`σ_d ∈ [0.22, 0.50]`, `ρ ∈ [0.05, 0.3]`) this gives
  **σ_within ∈ [0.002, 0.005], central ≈0.003.** *(The realized value is computed by the pinned
  bootstrap and is a noise-floor quantity, not an outcome — it is licensed input to the gate in
  §7, and it enters only in the INDETERMINATE-favouring direction.)*
- **σ_between ∈ [0.002, 0.008], central 0.005** — §3/A1, Madaan Table 1.

**σ_between is the same size as, or larger than, σ_within.** The pinned CI therefore omits a
variance component **of the same order as the one it keeps**. A TOST run against that CI does not
test the slope of `M` on parameters; it tests the slope **conditional on these three particular
checkpoints** — which is not the claim FLAT-COUPLED makes, and is not a claim anyone would publish.

> **BM-INFLATE — PINNED.** Every CI on `β` (the 95% CI for RISES/DECLINES **and** the 90% CI for
> the TOST) is computed with a rung-level variance of
>
> `Var(M̂(r)) = (σ̂_within(r)² + σ_between²) / n_seeds(r)`
>
> and `Var(β̂) = Σ_r w_r²·Var(M̂(r))`, `w_r = (x_r − x̄)/S_xx`.
>
> - **If `n_seeds ≥ 3` at `≥ 2` rungs:** `σ_between` is **ESTIMATED** from the seed replicates
>   (two-level bootstrap: resample seeds within rung, rows within seed). The estimate governs.
> - **Otherwise `σ_between` is UNESTIMABLE from our data** and the pinned **prior band
>   `[0.002, 0.008]`** is used. **A FLAT verdict requires the TOST to pass at the band's UPPER end
>   (`σ_between = 0.008`).** A verdict that flips inside the band is **INDETERMINATE.** *(Rationale
>   — and it is the asymmetry rule, not a taste: an unverifiable assumption may never be the thing
>   that purchases a positive verdict. RISES/DECLINES use the **central** 0.005, since inflating
>   the CI there is already the conservative direction.)*
> - The variance is treated as **known** (z-quantiles), **not** estimated from the OLS residual.
>   This is deliberate and it is the **FLAT-favourable** choice: estimating residual variance from
>   `|A|=3` gives 1 df and `t(0.95, 1) = 6.314` against `z = 1.645`, a **3.8× wider** CI. **We grant
>   the design the more favourable of the two treatments — and it still cannot reach FLAT (§7).**

---

## 6. THE DECISION RULE — and its behaviour under every possible outcome

`x_r = log10(params_r)`; `S_xx = Σ_{r∈A}(x_r − x̄)²`; `β̂` = OLS slope of `M(r)` on `x_r` over `A`;
`SE(β̂)` per BM-INFLATE (§5); `δ = 0.005/decade` (§4).

```
GATE-0 (structural).  |A| < 3            -> FLOOR (§9.5). No trend verdict. STOP.
GATE-1 (POWER, MANDATORY, PRE-VERDICT).
        FLAT is AVAILABLE  iff  δ  ≥  2.49 · SE(β̂)          [= (z.95 + z.80)·SE, i.e. ≥80%
                                                              power to certify equivalence
                                                              when β is truly 0]
        If GATE-1 fails, FLAT is struck from the verdict map BEFORE the slope is read, and the
        only non-significant verdict available is INDETERMINATE.

FACTOR 1 (only if GATE-0 passes):
    95% CI of β̂ excludes 0            -> RISES (β̂>0) / DECLINES (β̂<0)
    95% CI includes 0:
        GATE-1 passed AND 90% CI ⊂ (−δ, +δ)   -> FLAT
        otherwise                              -> INDETERMINATE (report n required, §8)
```

**GATE-1 is a function of `(SE, δ, n)` and is *identically invariant to `β̂`*.** It cannot be
tuned by anyone who has seen the result: there is no value of `β̂` that changes its output. It must
be **computed, and recorded in the repo, before `β̂` is read** — the project's own rule (CLAUDE.md:
*"a read-only audit round's verdict must be RECORDED in the repo BEFORE dispatching the dependent
stage"*). The analysis script must **emit GATE-1 above the slope** in its output.

**Behaviour under every landing — stated now, without the data, which is the test of a rule:**

Take the design's central variance (`σ_within = 0.003`, `σ_between = 0.005`), 3 rungs
`{14M, 98M, 392M}`, `n_seeds = 1` ⇒ `S_xx = 1.057`, `SE(β̂) = 0.0049/decade`. 95% half-width
`= 0.0096`; 90% half-width `= 0.0081`.

| The data land at… | 95% CI | TOST | **Verdict** |
|---|---|---|---|
| **A large effect** — say `β̂ = +0.030` (still <1/10 of A3's literature rate) | `[0.020, 0.040]`, excludes 0 | n/a | **RISES.** The design keeps its power for the positive result. |
| **Exactly zero** — `β̂ = 0.000` | `[−0.0096, +0.0096]`, includes 0 | 90% CI `[−0.0081, +0.0081]` ⊄ `(−0.005, +0.005)` — **fails** | **INDETERMINATE.** Not FLAT. We could not tell, and we say so. |
| **Exactly at the bound** — `β̂ = δ = 0.005` | `[−0.0046, +0.0146]`, includes 0 | upper 90% bound `= 0.0131 > δ` — **fails** | **INDETERMINATE.** An effect *at* the SESOI is by definition **not negligible** and must never be called FLAT. |
| **Just below significance** — `β̂ = +0.009` | includes 0 (barely) | fails by a mile | **INDETERMINATE.** |

**In no branch does noise buy FLAT.** The rule is fully specified without reference to the result;
that is the property D3 did not have.

---

## 7. THE FLAT-AVAILABILITY RULING — and I do not flinch from it

> ## **FLAT IS NOT AN EPISTEMICALLY AVAILABLE VERDICT FOR R0 AS DESIGNED.**
> **At `n_seeds = 1`, GATE-1 fails for every value of `β̂`, including `β̂ = 0` exactly.**
> **Under the design as it stands, the only verdicts reachable are RISES, DECLINES,
> INDETERMINATE, FLOOR, and VOID.**

**The proof (arithmetic, not opinion).** TOST at 90% requires the 90% CI inside `(−δ, +δ)`; the
*most* favourable case is `β̂ = 0` exactly, which still requires

```
δ  >  1.645 · SE(β̂)        ⇒  SE(β̂)  <  0.005 / 1.645  =  0.00304 / decade
```

and 80% power (GATE-1) requires `SE(β̂) ≤ 0.005/2.49 = 0.00201`. At `n_seeds = 1`, with
`S_xx = 1.057` (3 rungs) and BM-INFLATE:

| σ_between | σ_within | **SE(β̂)** | vs. TOST-possible (0.00304) | vs. GATE-1 (0.00201) |
|---|---|---|---|---|
| 0.008 (band top — **the pinned FLAT requirement**) | 0.003 | **0.0083** | ✗ **2.7× too large** | ✗ |
| 0.005 (central) | 0.003 | **0.0057** | ✗ 1.9× too large | ✗ |
| 0.002 (band **floor** — maximally FLAT-favourable) | 0.003 | **0.0035** | ✗ 1.2× too large | ✗ |
| 0.002 (floor) | 0.002 (floor) | **0.0028** | ~marginal | ✗ (power ≈ **57%** — a coin flip) |

**Across the entire literature-anchored band for `σ_between`, at central `σ_within`, the TOST
cannot pass at `n_seeds = 1` even if the true slope is exactly zero.** Only in the single corner
where *both* variance components sit at their floors does it become marginally possible — at ~57%
power, i.e. worse than a coin flip on a claim we would publish as a finding.

**And the corner cannot be checked.** `σ_between` is **not estimable at `n_seeds = 1`** — this is
a fact about the *design*, provable with no data at all. So we cannot know whether we are in the
corner. A verdict that depends on an assumption we structurally cannot test is not a verdict.

**The two independent legs, so that the ruling does not rest on one number:**

- **Leg A (unconditional, structural).** At `n_seeds = 1`, the CI required by a FLAT verdict
  **cannot be constructed**, because one of its two variance components is unmeasurable. Whatever
  the pinned bootstrap returns, it is a statement about *three specific checkpoints*, not about
  *parameters*. **No choice of `δ` repairs this** — the defect is in the variance, not the bound.
  This leg holds for **any** `δ`, including `δ_old`.
- **Leg B (quantitative).** Even granting the FLAT-favourable variance treatment (§5: known-variance
  z-quantiles rather than the 1-df t), and granting the *floor* of the literature band, GATE-1
  fails by 1.2×-4.1×.

**This is the honest possibility the dispatch named, and it is what the evidence says.** §9.5's
FLAT-COUPLED cell — *"The third outcome, and it is the one §5.2 could not express"* — was never
epistemically available at `n_seeds = 1`. It is not that we will fail to reach it; **it is that
there is no data this design could have produced that would license it.** Recording that **before**
the read is the entire value of pinning blind, and it kills the false headline before it is
written rather than after it is cited.

**The good news is real and should not be buried:** the design's power for **RISES/DECLINES is
intact** (§6, row 1). A true effect at even one-tenth the literature's recall-scaling rate clears
the BM-inflated 95% CI by 3×. **R0 can still return a positive result.** It cannot return a
negative one.

---

## 8. THE FALSIFIER — exactly what makes FLAT available, and the n required

**Falsifier of this ruling (one line):** *FLAT becomes available the moment `GATE-1` passes with
`σ_between` **estimated** rather than assumed — i.e. `δ ≥ 2.49·SE(β̂)` with `SE` computed from
`n_seeds ≥ 3` at `≥ 2` rungs.* Nothing else revokes it, and this revokes it completely. If a
future run measures `σ_between` for this metric and finds it at the band floor with `n_seeds ≥ 3`,
GATE-1 can pass at 3 rungs and FLAT is live. **The ruling is a statement about the current n, not
a metaphysical claim about the metric.**

**The n required (which §9.5's INDETERMINATE cell mandates we report).** Solving
`sqrt((σ_w² + σ_b²)/n_s) / sqrt(S_xx) ≤ δ/2.49`, at `σ_w = 0.003`:

| Design | `S_xx` | `n_seeds` for 80% power (σ_b = 0.005) | (σ_b = 0.008) |
|---|---|---|---|
| 3 rungs `{14M, 98M, 392M}` | 1.057 | **8 seeds/rung** | 20 |
| 4 rungs `{14M, 98M, 392M, 1.31B}` | 2.149 | **4 seeds/rung** | 9 |
| 4 rungs `{5M, 14M, 98M, 392M}` | 2.157 | **4 seeds/rung** | 9 |

⇒ **The minimum design under which FLAT is declarable at 80% power is ≈4 admissible rungs ×
≈4 seeds.** Two operational notes, offered because they change the price by an order of magnitude:

1. **Widen the ladder before you buy seeds.** `S_xx` doubles from 3 rungs to 4, halving the seed
   requirement. **Adding a cheap 5M rung buys the same `S_xx` as fixing the 1.31B rung** (2.157 vs
   2.149) at a tiny fraction of the cost — and a 5M model is also where seeds are cheapest. The
   OLS leverage `w_r²` is concentrated at the **endpoints**, so seeds at the *small* end are both
   the cheapest and among the most valuable.
2. **Seeds are a training cost, not an eval cost.** §9.2 prices the eval at *"under a minute of
   H100 time per cell"* — the probe is not the bottleneck; the pretraining runs are.

### H-1 — A STRUCTURAL HAZARD I FOUND WHILE DERIVING `S_xx`. **VERIFY BEFORE ANY READ.**

Not my remit, flagged rather than buried, because it may make everything above moot.

§9.6 item 1 pins the **common token slice = 0.328B tokens** (forced by the shortest-trained rung).
§9.6 item 2 pins: *"the primary trend fit is over rungs with **≥ 1.0 token/param** at the common
slice."* At a common slice of `T` tokens, a rung of `N` params has `T/N` tokens/param, so:

> **The ≥1.0 tok/param rule admits exactly the rungs with `N ≤ T = 0.328B` parameters.**

- 14M → 23.4 tok/param ✓ *(reproduces §9.6's own "~23" — the convention checks out)*
- 98M → 3.35 ✓
- **392M → 0.837 ✗ — 0.392B params > 0.328B tokens. EXCLUDED.**
- 1.31B → 0.250 ✗ *(reproduces §9.6's own "0.25")* **EXCLUDED** — as §9.6 anticipated.

§9.6 anticipated the rule removing the **1.31B** rung (*"If it removes the 1.31B rung, then the
ladder is not 2 orders of magnitude and we do not say that it is"*). **It does not appear to have
noticed that the same rule removes 392M**, which sits just 20% the wrong side of the line. If so,
`|A| = {14M, 98M} = 2 < 3` ⇒ **§9.5's FLOOR clause fires: no trend verdict of any kind is askable,
FLAT included**, and the ladder is 0.845 decades, not 2.

**This is CONTINGENT on the two premises I could not verify blind (§0): the rung ladder and the
param-count convention. A non-blind checker must resolve it against the actual model configs
before R0 is read.** The convention is strongly indicated — §9.6's *own* arithmetic (23.4 and
0.250) reproduces only under nominal/total params — but I will not assert it, and I note the
resolution is knife-edge: if the 392M rung's counting convention puts it at ≤328M, it survives.

**If H-1 holds, the cheapest repair is small:** train the 392M rung ~20% further (0.328B → ≥0.45B
tokens) to clear 1.0 tok/param, and add a 5M rung. That yields `{5M, 14M, 98M, 392M}` at a 0.45B
common slice — 4 admissible rungs, `S_xx ≈ 2.16`, 1.9 decades — which is also **exactly the
design that makes FLAT available at 4 seeds/rung** (table above). The two problems have one fix.

---

## 9. WHAT THIS PIN DOES NOT BUY

- **It does not make the result come out any particular way.** I do not know which way it came
  out, and the rule in §6 was written to be defensible in all four directions before I could.
- **It does not rescue a `σ_between` we never measure.** The prior band is an *anchor*, not a
  measurement; §5 uses it only in the INDETERMINATE-favouring direction, and §8 pins how to
  replace it with a real number.
- **It does not touch Factor 2** (the span_frac monotonicity licence), §9.6's admissibility, T1/T2,
  or the placebo identification. Those are pinned elsewhere and I did not read most of them.
- **It does not license a re-read.** §9.6's stop rule stands: this is a pre-registration written
  **before** the primary read, not a re-analysis after one. If R0 has already been read against
  `δ_old` by anyone, **this file is void as a pre-registration** and must be re-labelled as a
  post-hoc re-analysis, disclosed as such. *(I have no knowledge either way.)*

---

## 10. SUMMARY

| | |
|---|---|
| **Pinned** | **`δ = 0.005` DiD-accuracy-fraction per decade of `log10(params)`** — absolute, construction- and literature-derived, invariant to every measurement of this experiment. |
| **Supersedes** | §11.7-D3 (`δ = 0.125·M(r_min)`) and §9.5's `δ` clause + its "25% over 2 decades" gloss. |
| **Added** | **BM-INFLATE** (§5) — every CI on `β` must carry the between-model variance component, estimated where possible and bounded by the literature prior where not. **GATE-1** (§6) — a pre-verdict, `β̂`-invariant power gate that strikes FLAT from the map when the design cannot support it. |
| **Ruling** | **FLAT is NOT epistemically available at `n_seeds = 1`.** Not "hard to reach" — *unreachable, for every possible `β̂`, including exactly zero.* Any non-significant slope ⇒ **INDETERMINATE**. |
| **Price of FLAT** | ≈**4 admissible rungs × ≈4 seeds** (σ_b = 0.005). Widen the ladder before buying seeds. |
| **Still available** | **RISES / DECLINES.** The design retains full power for a positive result at ~1/10 the literature's recall-scaling rate. R0 can still find something. It just cannot find *nothing*. |
| **Flagged** | **H-1** — the ≥1.0 tok/param rule may exclude the 392M rung as well as 1.31B, leaving `\|A\| = 2` and firing FLOOR. Verify against the model configs **before** the read. |
