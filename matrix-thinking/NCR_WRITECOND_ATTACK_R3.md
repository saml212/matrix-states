# NCR WRITE-CONDITIONING — ATTACK ROUND 3

**Target:** `matrix-thinking/NCR_WRITE_CONDITIONING_DESIGN.md`, section
`## DRAFT-R3 — SUPERVISED WRITE LEARNING (2026-08-13)` (lines 2283-2886)
**ONLY.** Everything above it is dead-draft or harvested record; §A2's
settled-clean list and the PREMISE BATTERY HARVEST + REPLICATION facts are
treated as frozen evidence and are built on, not re-litigated.

**Repo state:** commit `56310ba`. Runner pin re-verified this round:
`md5(experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/ncr_lm_wave1_runner.py)
= 9a93198b642242f512ff8489e32b0a53`.

**Date:** 2026-08-13. **Round:** 3 (R1 = BLOCKED 5F/11M/7m; R2 = BLOCKED
5F/12M/8m; both ADOPTED).

---

## VERDICT: **BLOCKED**

**3 FATAL / 8 MAJOR / 8 minor.**

The mechanism is not dead — it is the first mechanism in this file whose
premise is *positively evidenced* (the harvest's BD=11), and its core algebra
is correct and independently reproduced here. But the **distance chosen in
§2(a) does not control the quantity the pre-registered metric measures.**
`L_write`'s global-minimum set is a 25-dimensional affine family, and I have
executed a walk along that family in which `L_write` stays pinned at its
numerical floor (`3.8e-12`), `retrieval24@h=1` stays at a perfect `1.0000`,
and `retrieval24@h*=61` falls to **0.0625 — chance**. The loss is provably
zero at the ideal write (§2(a) is right about that) and provably *also* zero
at operators that fail the claim. That is F1.

F2 and F3 are independent launch-losers of the same class: the objective the
wave would actually optimise is `CE + 0.5·aux + 0.1·ortho + λ_w·L_write`, and
the `0.1·ortho` term — which the draft cites as *evidence* (§A2 F5) but never
notices is still **switched on in all three recipes it plans to use** — has a
minimum provably disjoint from `L_write`'s, measured at `ortho_loss(Z_ideal)
= 15,147`. At the λ_w the draft's own Stage-0.3 derivation would produce, the
joint minimiser reads **chance at h=61**.

None of the three FATALs is a reason to abandon the mechanism. All three have
concrete, cheap repairs, and two of them are *already latent in the draft's
own text* (the rejected alternative 1 is the fix for F1; the runner's own
detach convention is the fix for F3). This is a REV-REQUIRED shape of
BLOCKED, not a KILL.

---

## What is VERIFIED CLEAN (recorded so R4 does not re-litigate)

| # | claim | status |
|---|---|---|
| V1 | `teacher_force_operator` solves `k zᵀ = v` and `Z = z_tᵀ` maps `k_i → v_i` | **VERIFIED**, re-derived + executed: `L_write(Z_ideal)` = `3.79e-12` (fp32), `2.43e-29` (fp64) |
| V2 | `pinv` returns the **zero-residual, minimum-norm** solution; `K=24 < d=25` ⇒ consistent | **VERIFIED**; and the min-norm choice gives `Z_ideal w = 0` exactly (`max_b ‖Z_ideal w‖ = 2.6e-05` fp32) — a fact the draft does not state and which F1 turns on |
| V3 | "full row rank — true for essentially every batch" | **VERIFIED, structurally**: `grammar_rd._assert_injective_entities` (`grammar_rd.py:289-300`, called at `:443` and `:516`) asserts `K` unique entity ids per row every batch, drawn **without replacement** (`:430`). §6's rank premise is safe by construction, not by luck |
| V4 | every code citation in DRAFT-R3's grounding paragraph | **VERIFIED**: runner md5, `ncr_lm_forward_ablatable:360-398`, `teacher_force_operator` (`ncr_lm_wave1_smoke.py:348-362`), `compute_arm_losses:768-852`, `ortho_regularization_loss:714-742`, `aux_read_supervision_loss:599-632`, `model_v4.py:25-64` |
| V5 | `ortho_reg_weight=0.1` in all three §G3-B31 baselines (§A2 F5) | **VERIFIED** in the raw JSONs — and see F2, which is what that fact actually implies for THIS design |
| V6 | budget arithmetic | **VERIFIED to the digit**: 0.875 s/step ⇒ 4.861 GPU-h/20k; S0.1 = 0.365; S1 = 19.981 (draft 19.977, rounding); total 20.345; low 19.301; high 21.389. Every number in §3's table reproduces |
| V7 | `τ = 0.09162` | **VERIFIED EXACT**: `1/24 + 4·sqrt(p(1−p)/256) = 0.091623` |
| V8 | `P0_ref = 0.07` "rounds up from 0.035-0.066, chosen conservatively" | **VERIFIED**: max over the three configs at `h=61` is `0.0664` (compB). Genuinely conservative |
| V9 | additive-only wiring; OFF ⇒ branch never runs; `is_full_graft` gate | **VERIFIED** against `compute_arm_losses:768-852`; the proposed insertion is structurally identical to the `ortho_reg_weight` branch at `runner.py:849-851` |
| V10 | "eval NEVER teacher-forces" for the intended wiring | **VERIFIED STRUCTURALLY**: `eval_both_arms → eval_arm_at_hops → ncr_lm_forward_ablatable` never calls `compute_arm_losses`; no loss term can reach eval. (But see M7 — the guarantee is one mis-wire from silently inverting, and nothing is pre-registered to catch it) |
| V11 | `h*=61` is produced by the standard in-loop eval | **VERIFIED**: `DEEP_LADDER = [5,12,20,29,40,61]`, and `build_attribution` emits `retrieval24_gap_deep["h=61"]` — §3.6's GAP clause is computable with no supplement |
| V12 | §3.6's PARTIAL `OR` does not double-fire; `fraction_closed(τ) = 0.0238` matches the stated `0.024` | **VERIFIED** by exhaustive grid (0 double-firings in 402 points). The *holes* are M5(a) |
| V13 | §2(b)'s "`L_write` never swaps `Z` in the forward pass, so the §R2.1 decode-mismatch cannot arise" | **SOUND as far as it goes.** Train and eval both read `Z_sgd = ncr_head.encode(...)` at the same parameters; the decode head co-adapts continuously; there is no second `Z`-distribution. The charter's suspicion (co-training against an evolving `Z`) is ordinary non-stationarity, not the R2.1 failure. See m3 for the one thing wrong with the section |

---

## FATAL FINDINGS

### F1 (FATAL, mechanism — §2(a)) — `L_write`'s zero set is a 25-dimensional family containing operators that read at CHANCE at `h*=61`; the "restricted to the K observed keys" choice leaves an *amplifying* direction unsupervised, and the draft's own argument for that choice is exactly inverted

**The algebra the draft did not do.** `d − K = 25 − 24 = 1`. Let `w` be the
unit vector orthogonal to every key (`k_i · w = 0 ∀i`, verified numerically:
`max|w·k_i| = 2.0e-07`). Then for **any** `γ ∈ ℝ` and **any** `u ∈ ℝ²⁵`:

```
(u wᵀ) k_i  =  u (w · k_i)  =  0        for every i = 1..K
```

so `L_write(Z_ideal + γ u wᵀ) = L_write(Z_ideal)` **identically**. The zero
set of `L_write` is therefore not the point `Z_ideal`; it is the affine family

```
{ Z : Z k_i = v_i ∀i }  =  Z_ideal + { u wᵀ : u ∈ ℝ^25 }        (25 free params)
```

`pinv`'s minimum-norm choice — the one §2(a) dismisses as *"an arbitrary
artifact… [that] carries no task information"* — is precisely the member of
that family with `Z w = 0`. It is not arbitrary. **It is the only member that
is dynamically stable under repeated application**, and it is the one the
harvest's BD=11 actually measured.

**Executed demonstration** (`scratchpad/sim_B_freedir.py`,
`sim_C_precision.py`; real `nm.binexp_read`, real
`discriminability_metrics` retrieval24 construction, n=256 episodes, key
geometry calibrated to compB's own measured `target_pairwise_cos ≈ 0.21`):

| `‖Z w‖` (transverse gain) | `L_write` | `retr24@1` | `retr24@13` | `retr24@37` | **`retr24@61`** |
|---|---|---|---|---|---|
| 0 (= `Z_ideal`) | 3.79e-12 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 1 | 3.81e-12 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 3 | 3.84e-12 | 1.0000 | 1.0000 | 0.9805 | **0.9492** |
| 5 | 3.91e-12 | 1.0000 | 1.0000 | 0.7891 | **0.7188** |
| 10 | 4.05e-12 | 1.0000 | 0.8867 | 0.4531 | **0.4062** |
| 25 | 5.14e-12 | 1.0000 | 0.4258 | 0.2266 | **0.1992** |
| 100 | 2.48e-11 | 1.0000 | 0.0898 | — | **0.0625 (chance)** |

`L_write` moves by **1.4×** across the whole table — it is at its floor
everywhere. `retrieval24@h=1` is a perfect `1.0000` everywhere. `h*=61` goes
from `1.0` to chance.

**It is not a precision artifact and not a binexp artifact.** Same collapse in
fp64 (transverse gain equal to `‖Z_ideal‖_F` ⇒ `0.9609 / 0.4336 / 0.3359` at
h=13/37/61, with `L_write = 3.7e-29`). Same collapse under `loop_read` — the
O(h) read that never forms a matrix power (`sim_E_mechanism.py`: at
`‖Zw‖=10`, binexp `0.4062` vs loop `0.4102`). The mechanism is a **transverse
Lyapunov exponent**: the key orbit is an exactly invariant subspace of measure
zero; float roundoff kicks the iterate off it at every step; the unsupervised
transverse gain then amplifies that roundoff by `(gain)^h`. At `h=61` and gain
10 that is `10^61`. No precision fixes it, and no read fixes it — only
constraining `Z` off-span does.

**Scale of the threat.** The threshold is `‖Z w‖ ≲ 3` for `h=61` (`0.9492`).
For an operator whose overall scale matches the target (`‖Z_ideal‖_F ≈ 25`),
an *unconstrained* transverse response is `≈ ‖Z‖_F/√d ≈ 5` in expectation —
**already past the threshold**. This is not an adversarial corner; it is the
generic case for any `Z` the loss has not been told to flatten there.

**The draft's own argument is the inverse of the truth.** §2(a) rejection 2
says: *"repeated squaring from a key with small residual stays close to the
discrete orbit at every depth precisely because the residual is bounded in the
FULL ambient space, not merely its in-span component (a projected loss could
leave complement-direction leakage unpenalised; this form cannot, by not
projecting at all)."* This conflates the **codomain** with the **domain**.
`L_write` does bound the residual in the full ambient space — of `Z`'s action
**at 24 points**. It says nothing whatever about `Z`'s action **on** the
complement. The form that closes the hole is rejection 1, full Frobenius to
`Z_ideal`, because

```
‖Z − Z_ideal‖²_F  =  ‖(Z − Z_ideal) P_span‖²_F  +  ‖Z (I − P_span)‖²_F
                                                   ^^^^^^^^^^^^^^^^^^
                                                   exactly the missing term
                                                   (Z_ideal (I − P_span) = 0)
```

**Compounding consequence — the design pre-registers a wrong diagnosis of its
own most likely failure.** §6 item 2 attributes any depth decay to *"`Z_sgd`'s
OWN condition number… `A_cond ≈ 9,959`"*, and §3.6 carries a "depth-decay
PARTIAL signature" for exactly the reading F1 produces (`WIN/PARTIAL at h≤20`,
`NULL by h=61`). If the wave reads that signature, the pre-registration
directs the assessor to the wrong cause, and the recorded verdict would be
"write supervision partially works" when the truth is "the loss does not
constrain the operator the read composes."

**And Stage 0.1's gate cannot see it** (see m7): the gate is `L_write`
descending **and** `retrieval24` not regressing. In the entire F1 family,
`L_write` is at its floor and `retrieval24@h=1 = 1.0000`. The canary passes at
every gain up to 100.

**Repair (binding proposal in the disposition):** supervise the complement.
Either the full Frobenius form to `Z_ideal` with a per-episode normaliser (the
draft's own §3 code snippet already computes `Z_ideal` — see m1), or keep
§2(a)'s restricted term and add `μ · ‖Z (I − P_span)‖²_F / (normaliser)` with
`P_span` built from the same `keys_v` the loss already has. Register the
transverse gain `‖Z_sgd w‖` as a logged diagnostic either way; the calibrated
requirement from the table above is `‖Z w‖ ≲ 3` at `‖Z‖_F ≈ 25`, i.e.
transverse gain ≤ ~12% of the operator norm.

---

### F2 (FATAL, §3 wiring / uninterpretable verdict) — `ortho_reg_weight = 0.1` is ON in all three recipes the design plans to use, its minimum is provably disjoint from `L_write`'s, and at the λ_w Stage 0.3 would produce the joint minimiser reads CHANCE at `h*=61`

The draft cites §A2's F5 as *evidence* (the runner already trains
`0.1·‖ZᵀZ−I‖²/d²` and it did not prevent collapse) and concludes the design
"exits that mechanism CLASS entirely" (§5). It never notices the operational
consequence: **§3 imports the three §G3-B31 recipes verbatim, and those
recipes carry `ortho_reg_weight = 0.1`.** Verified in the raw configs:

```
mob_g3b31_primary_s0.json : ortho_reg_weight 0.1, aux 0.5 contrastive+cosine, freeze=true
mob_g3b31_compA_s0.json   : ortho_reg_weight 0.1, aux 0.5 cosine,             freeze=true
mob_g3b31_compB_s0.json   : ortho_reg_weight 0.1, aux 0.5 contrastive+cosine, freeze=false
```

The objective actually optimised would be
`CE + 0.5·aux + 0.1·ortho + λ_w·L_write`. §3's wiring snippet, §3's cell
table, and §3.6's bands never mention `ortho`. Stage 0.3 — the cell that
*sets* `λ_w0` — derives it from "`A_cond ≈ 9,959`, `Z_ideal`'s own residual
`3.1e-16`, and §1.3's `1/√d_ncr ≈ 0.20` floor". There is no `ortho` term
anywhere in that derivation.

**R2's F1 already proved the two are incompatible in principle** (the correct
write is nowhere near orthonormal at this geometry: `f(A*) ≈ 149–195`). This
round measures it in the runner's own units
(`scratchpad/sim_D_ortho_placebo.py`, `ortho_regularization_loss` reproduced
verbatim):

```
ortho_loss(Z_ideal)                      = 15147.28      (⇒ 0.1 × that = 1514.7 loss units)
ortho_loss(norm-matched Z_ideal)         =     0.5075
‖∇_Z ortho_loss‖ at Z_ideal (median)     =     0.2178
‖∇_Z L_write‖   at Z_ideal (median)      =     3.0e-09   (zero, by construction)
```

At the target, `L_write`'s gradient is zero and `ortho`'s is not. The
stationary point of the sum is therefore displaced from `Z_ideal` by an amount
set entirely by `λ_w`. Joint minimisation (Adam, 4,000 steps, `Z` free, same
n=256 episodes):

| `λ_w` | `L_write*` | `retr24@1` | `retr24@13` | `retr24@37` | **`retr24@61`** | band |
|---|---|---|---|---|---|---|
| 0 | 2.26e+00 | 0.4922 | 0.0312 | 0.0273 | **0.0156** | NULL |
| 0.1 | 9.27e-01 | 0.9453 | 0.1367 | 0.0469 | **0.0391** | NULL |
| 1 | 5.27e-01 | 0.9766 | 0.3438 | 0.0586 | **0.0547** | NULL |
| 3 | 3.70e-01 | 0.9883 | 0.5039 | 0.0977 | **0.0547** | NULL |
| 10 | 2.39e-01 | 0.9961 | 0.7070 | 0.2539 | **0.1094** | PARTIAL |
| 100 | 7.09e-02 | 0.9961 | 0.8594 | 0.6445 | **0.4453** | PARTIAL |
| 1000 | 1.43e-02 | 1.0000 | 0.9297 | 0.8477 | **0.8008** | WIN |

`λ_w` must exceed the ortho weight by **~10⁴** before the combined objective
even permits the deep read. Every `λ_w` in the range a Stage-0.3 derivation
based on residual magnitude would produce (`O(0.1)`–`O(10)`, matching the
house's `aux 0.5 / ortho 0.1` scale) lands in the NULL/PARTIAL rows —
**for a reason that has nothing to do with the hypothesis under test.**

The caveat, stated: this minimises over `Z` as a free parameter, without CE
and aux. The direction and order of magnitude are robust (the displacement is
driven by a gradient that is nonzero at the target and a weight that is fixed
at 0.1); the exact λ_w floor in the real run is not. That is precisely why it
must be **measured before launch, not assumed** — and it is a zero-GPU
measurement on the existing checkpoints (see the Stage-0 ruling).

**Consequence if launched as written:** the most likely outcome is
`retr24@61 ≈ chance` on all three recipes, at which point §3.6 item 4 fires —
*"direct exact-operator write-supervision, as specified, is FALSIFIED at this
architecture/scale — a materially more serious finding than a simple NULL"* —
and the program records a false falsification of a mechanism that was never
given a chance to run. That is the single most expensive failure mode in this
design: 20 GPU-h spent to produce a wrong headline.

**Repair:** either (a) set `ortho_reg_weight = 0` in the write-supervised
arms — but then they are no longer the §G3-B31 recipes and P0 (measured **with**
ortho on) is no longer the matched baseline, so a fresh `ortho=0, λ_w=0`
no-supervision cell must be added (affordable — see M6); or (b) keep ortho at
0.1 and register a λ_w floor **derived from the measured joint curve on real
batches**, disclosing that the arm is a two-term hybrid and that §5's "exits
that mechanism class entirely" does not hold for what is actually run. (a) is
cleaner; either way it must be an explicit, pre-registered decision.

---

### F3 (FATAL, §2(a)/§3 wiring) — as specified the loss back-propagates into its own target, so its zero set contains total entity collapse — the §G3-B26 pathology this program already fought, and which the runner's own aux losses `detach()` to prevent

§3's wiring is `write_supervision_loss(Z, keys_v, values_v)` with
`keys_v`/`values_v` **undetached**. Both are outputs of
`integ.entity_adapter ∘ backbone.embed`. Gradient therefore flows from
`L_write` into the *target*.

The degenerate route is immediate: if the adapter/embedding maps every
in-document entity to one vector `c`, then `k_i = v_i = c` for all `i`, and
`Z = I` gives `L_write = 0` **exactly**. `retrieval24` at that point is
chance by construction (all 24 targets identical ⇒ the argmax is arbitrary).

This is not hypothetical in this program. It is the §G3-B26 finding, and the
runner documents the exact same hazard for the exact same reason
(`aux_read_supervision_loss`, `runner.py:614-630`, verbatim):

> *"withOUT detach, backward would ALSO flow through the target's own
> entity_adapter/embed call, training entity_adapter to chase a target that is
> itself moving under the same update (a self-referential optimum — e.g.
> entity_adapter could collapse all its outputs toward one point to trivially
> satisfy cos=1 everywhere). detach() removes that side entirely."*

Every aux target in this runner — the cosine target and all 24 contrastive
targets — is detached for this reason. `teacher_force_operator` itself
detaches `k` and `v` (`smoke:360`). `L_write` as specified is the only
supervision in the file that does not.

**The route is live in all three recipes, not just the trainable one.** The
"frozen" arms freeze `integ.entity_adapter` but **not** `backbone.embed`, and
the archive shows the target space collapsed anyway:

| arm, cell | measured `target_pairwise_cos` @ step 20,000, all hops |
|---|---|
| compA `full_graft` (frozen adapter, cosine aux) | **0.7987 – 0.8148** |
| primary `full_graft` (frozen adapter, ctr+cos) | **0.7196 – 0.7355** |
| compB `full_graft` (trainable adapter, ctr+cos) | 0.1956 – 0.2475 |
| **`backbone_only` (never trains — the frozen-at-init control)** | **0.0638 – 0.0926** |

The `backbone_only` arm is the clean control: its write/read pathway receives
exactly zero gradient (`assert_read_ablation_is_exact_zero`), and its target
space sits at `0.064–0.093`, i.e. the frozen-init basis the runner's own
§G3-B30 band-1 check names (`runner.py:652-653`, "0.065-0.081"). The trained
arms drifted to `0.72–0.81`. So the collapse is **training-induced and
measured**, the channel is open in every arm despite
`--freeze-entity-adapter` (because `backbone.embed` is not frozen), and it is
*worst* in the two arms the draft calls "frozen". Adding a loss whose global
minimum includes that collapse — with an undetached gradient path straight
into the collapsing parameters — is not a risk to monitor, it is an
invitation.

**Band interaction:** §3.6's Band 2 ("write-engagement: `L_write` must show
real descent") reads **PASS** on the collapse route. The design's own
engagement check cannot distinguish "the encoder learned to emit `Z_ideal`"
from "the target space collapsed and `Z → I`."

**Repair (one word):** `write_supervision_loss(Z, keys_v.detach(),
values_v.detach())` — matching `teacher_force_operator`'s own convention and
the house detach discipline, so the loss reaches parameters **only** through
`Z`, i.e. through `ncr_head` (the write encoder), which is what the mechanism
is about. Additionally register `target_pairwise_cos` as a Band-1 tripwire
(it is already emitted at every eval by `discriminability_metrics`, zero cost).

---

## MAJOR FINDINGS

### M1 — `L_write` spends most of its value and gradient on a degree of freedom the pre-registered metric provably ignores; §2(a)'s rejection of cosine conflates global scale with per-key scale

`binexp_read` renormalises at every squaring and every apply, so the read is
**exactly** invariant to a positive global scale on `Z`. Measured
(`sim_G_reachability.py`):

| `c` | `L_write(c·Z_ideal)` | `retr24@1` | `retr24@61` |
|---|---|---|---|
| 0.01 | 9.80e-01 | 1.0000 | **1.0000** |
| 1 | 3.79e-12 | 1.0000 | **1.0000** |
| 10 | 8.10e+01 | 1.0000 | **1.0000** |
| 100 | 9.80e+03 | 1.0000 | **1.0000** |

The loss ranges over 13 orders of magnitude across operators that are
*identical* to the metric. Decomposing a realistic error into "global scale"
and "everything else":

| `Z` | `L_write` | after optimal global rescale | share of loss that is read-invisible |
|---|---|---|---|
| `1.0·(Z_ideal + 0.1E)` | 5.87e-01 | 2.71e-01 | **53.9%** |
| `2.0·(Z_ideal + 0.1E)` | 3.35e+00 | 2.71e-01 | **91.9%** |
| `3.0·Z_ideal` (no directional error at all) | 4.00e+00 | 3.8e-12 | **100%** |

§2(a) rejects cosine because *"a `Z_sgd` that is directionally correct per key
but off by a per-key scale factor `cᵢ` still collapses under repeated
application."* That is true for a **per-key** `cᵢ` and false for a **global**
`c` — the two are different degrees of freedom, and the section treats them as
one. The correct object is a loss that is sensitive to per-key scale *ratios*
and invariant to the global scale (fit `c* = ⟨Zk, v⟩/‖Zk‖²` in closed form and
score the residual after rescaling — one extra einsum, no new machinery).

**Band consequence:** §3.6's Band 2 ("`L_write` must show real descent from
its own step-0 value") is satisfiable by pure global-scale fitting. As
written, the engagement check is not evidence of read-relevant progress.

### M2 — Stage 0.2 measures the wrong statistic: `cond(keys_v)` has *no* bearing on the read, and the quantity that does bear on reachability is not measured anywhere

Executed (`sim_A_keygeom.py`, n=256/geometry, real `binexp_read`, key
geometries calibrated to the archive's own measured `target_pairwise_cos`):

| geometry | `cond(keys)` med / p90 / max | `retr24@1/13/37/61` with the exact write |
|---|---|---|
| isotropic | 40 / 110 / 392 | 1.0000 / 1.0000 / 1.0000 / **1.0000** |
| ρ=0.21 (compB) | 58 / 172 / 1602 | 1.0000 / 1.0000 / 1.0000 / **1.0000** |
| ρ=0.73 (primary) | 170 / 478 / 1660 | 1.0000 / 1.0000 / 1.0000 / **1.0000** |
| ρ=0.80 (compA) | 207 / 586 / 2225 | 1.0000 / 1.0000 / 1.0000 / **1.0000** |
| ρ=0.99 (near-collapse) | 1023 / 2737 / **10262** | 1.0000 / 1.0000 / 1.0000 / **1.0000** |

Per-episode, the worst-conditioned decile (`cond ∈ [95, 420]`) reads
`retr24@61 = 1.0000`, identical to the best-conditioned half. **The exact write
is insensitive to key conditioning across four orders of magnitude** — because
the orbit is closed exactly. §6 item 3's stated worry (*"near that boundary,
`pinv` is numerically unstable and `Z_ideal` is a poorly-behaved function of
context"*) is therefore not a read-side risk at all, and the specified
gate condition (*"`cond(keys_v)` diverging/NaN for a large fraction of
episodes"*) will essentially never fire.

What *is* real is the **target-norm tail**, which Stage 0.2 does not measure:

| geometry | `‖Z_ideal‖_F` med / max | per-**row** norm med / max | row-norm max/med | within-episode row max/min (med, max) |
|---|---|---|---|---|
| ρ=0.21 | 25.3 / **730** | 4.15 / **303.7** | **73×** | 5.5, **89×** |
| ρ=0.80 | 25.1 / 256 | 4.13 / 157.5 | 38× | 5.6, 76× |

Against `BindingEncoder`'s actual output construction (`model_v4.py:59-63`) —
`Z = row_out(row_norm(q))`, i.e. **every row of `Z` is one shared
`Linear(64 → 25)` applied to a LayerNorm output of fixed norm** — that entire
dynamic range must be supplied by `cond(row_out.weight)`, the same matrix for
every row and every episode. That is the real reachability question, it is
checkable at **zero cost** on the three existing checkpoints
(`cond(ncr.row_out.weight)` vs the measured required range), and neither
Stage 0.2 nor §6 item 3 asks for it.

**Should Stage 0.2 gate the wave?** As specified: no — it gates on a
condition that cannot fire and would return "moderate, proceed" while the
actual obstruction is untouched. **Replaced** (see the disposition): yes, the
expanded version should gate, because `‖Z_ideal‖` tail + `cond(row_out)` +
`‖Z_sgd w‖` together decide whether the target is representable at all.

### M3 — the placebo's gradient budget is NOT matched (6.5e5× at convergence), and by construction it cannot detect the confound it is named for

§4 claims the wrong-operator placebo delivers the *"**SAME gradient budget**
by construction, not merely by claim."* Measured (`sim_D_ortho_placebo.py`,
median per-item `‖∇_Z‖`):

| point on the training path | `L_write` | `‖g_write‖` | `L_wrong` | `‖g_wrong‖` | ratio |
|---|---|---|---|---|---|
| random init (norm-matched) | 5.87e+01 | 1.53e-02 | 5.89e+01 | 1.51e-02 | 0.99 |
| `Z_ideal` + 30% err | 5.28e+00 | 4.59e-03 | 6.42e+00 | 4.81e-03 | 1.05 |
| `Z_ideal` + 10% err | 5.87e-01 | 1.53e-03 | 2.07e+00 | 2.42e-03 | 1.58 |
| `Z_ideal` + 1% err | 5.87e-03 | 1.53e-04 | 1.65e+00 | 1.96e-03 | **12.8** |
| converged | 3.79e-12 | 3.0e-09 | 1.66e+00 | 1.96e-03 | **6.5e5** |

The match holds only at initialisation. As `L_write` converges its gradient
vanishes and the placebo's does not, so over 20,000 steps the placebo receives
**vastly more** cumulative pressure on `Z`. This is R2's F3 failure mode
(mis-dosed null) with the sign flipped: over-dosed rather than under-dosed.
An over-dosed placebo that damages the model yields an uninformative "no
improvement." It also fights F2 harder: `ortho_loss(Z_wrong) = 34,946` vs
`ortho_loss(Z_ideal) = 15,147`.

**Worse, the placebo's reading is analytically pre-determined.**
`Z_wrong = tf_op(k, v[σ])` is *exactly attainable* (same `k`, consistent
system: `L_wrong(Z_wrong) = 3.96e-12`), and a perfectly-trained placebo scores,
against the TRUE answer:

```
retr24 @ 1/13/37/61  =  0.0000 / 0.0430 / 0.0508 / 0.0352      (chance or below)
```

Any successful learning of a wrong permutation **must** produce chance
retrieval. So the pre-registered reading *"placebo IMPROVES retrieval ⇒ generic
extra-gradient-pressure nuisance effect"* can never be observed, and the
control cannot discriminate the confound it names. It only tests "a wrong
target does not accidentally produce the right answer," which is true by
construction.

**Repair:** keep the cell (it is a real specificity check) but re-scope its
**readout** to the only thing it can discriminate — *conditioning transfer*:
does a well-formed-but-wrong target produce a well-conditioned `Z` (report
`A_cond`, eff-rank, transverse gain `‖Z w‖`, `o_pairwise_cos`) while reading at
chance? That result would genuinely separate "conditioning" from "correctness."
And log per-step `‖∇_Z‖` for both arms so the budget claim is measured rather
than asserted.

### M4 — Control B does not isolate readout adaptation, and cannot be run "VERBATIM, unmodified"

Two independent defects.

**(a) The required readout does not exist in the pinned path.** §4 specifies:
continue with `teacher_force=True` for 2,000 steps, *"then evaluate with
`teacher_force=False`"*. The runner's in-loop eval is

```python
eval_result = eval_both_arms(arms, pools, cfg, eval_batch_size, device, seed,
                             teacher_force=teacher_force_operator)   # runner.py:1408-1409 and 1432-1433
```

— the **same flag**. A `--teacher-force-operator` continuation produces
teacher-forced evals, i.e. P1b readings, at every eval point and at the final
write. The clean-`Z_sgd` readout the control is *for* requires a separate eval
invocation. This is repairable at zero cost — the premise battery already
built exactly such a script (`pbe_repl.py.txt` / `pbe_supplement.py.txt`,
archived) — but the design's claim that it *"Reuses `--teacher-force-operator`
VERBATIM, unmodified"* and that *"No existing function or code path is
modified"* is false as written, and a builder following it literally would
produce a control whose numbers are P1b.

**(b) `Z_sgd` is not unchanged.** §4 asserts *"`Z_sgd` restored, UNCHANGED
throughout this continuation by the zero-grad guarantee just cited."* The
guarantee (`runner.py:1344-1348`) is `ncr_untouched = all(p.grad is None for p
in arm["ncr"].parameters())` — it covers `ncr_head`'s **parameters**, not
`encode`'s **inputs**. Control B runs on compB, which has
`freeze_entity_adapter = false`. `teacher_force_operator` detaches `k, v`, but
`integ.query_key` (`runner.py:394`) is **not** detached and CE/aux flow through
`o` into `entity_adapter` every step. So over 2,000 steps `entity_adapter`
drifts ⇒ `keys_v`/`values_v` drift ⇒ the realised
`Z_sgd = ncr_head.encode(keys_v', values_v')` drifts, **and** the
`retrieval24` targets themselves drift (they are `entity_adapter(embed(...))`,
`runner.py:514`). The control confounds "readout adaptation" with write-input
drift and with target-space drift — and under teacher forcing the adapter is
being shaped by an objective that rewards making `pinv` work well, which is
write-side adaptation in disguise. Either freeze the adapter for this
continuation (and disclose that it is then no longer compB's recipe) or drop
the "isolates the read side" claim.

### M5 — the bands: a partition hole, 25 unadjudicated cross-recipe outcomes, recipe-variance sold as seed-variance, no `answer_accuracy`, and an inverted conservatism claim

**(a) Not a partition (R2's F4 recurring).** Exhaustive grid over readings
`x ∈ [0,1]` × GAP ∈ {0.05, 0.20}: **60 of 402 points fire no band at all** —
every reading in `[0.705, 1.000]` with the `full_graft − backbone_only` GAP
`≤ 0.15`. WIN needs `x>0.19167 ∧ fc≥0.70 ∧ GAP>0.15`; PARTIAL needs
`(τ<x≤0.19167) ∨ (0.024<fc<0.70)`; NULL needs `x≤τ`. A strong reading with a
failing GAP clause is **UNCLASSIFIED**. (No double-firings — that half of F4
was fixed.)

**(b) 25 of 27 cross-recipe outcomes are unadjudicated.** §3.6 item 4 covers
"NULL on ALL THREE"; all-WIN is implicit. Nothing says what
`WIN / NULL / NULL` or `PARTIAL / WIN / NULL` means. With three
claim-bearing runs this is not a corner case — it is the modal outcome shape
for a marginal effect.

**(c) Three recipes are not three seeds.** §3 justifies `n≥3` via *"this
document's OWN established replication convention — the harvest's ROBUST
verdict… was earned by three DIFFERENT recipes at one seed each."* The harvest
was a **unanimous, saturated** diagnostic: all three at chance vs all three at
0.977–1.0, effect size ~0.9. Recipe-vs-seed confounding is immaterial when the
separation is that large. §3.6's bands are **graded** (a 0.70-fraction
threshold, a τ-width NULL band), and the three runs differ on two
architectural axes at once (frozen/trainable adapter, cosine/contrastive aux)
with **zero within-recipe replicates**. There is no variance estimate that
separates recipe effect from seed noise, so no CI over the three means what a
pre-registered `n≥3` CI is supposed to mean. Either add same-recipe seeds
(affordable — M6) or state explicitly that the three runs are a
**robustness sweep across recipes**, not a seed replication, and drop the CI
language.

**(d) `answer_accuracy` is not co-scored — dropping an ADOPTED repair.**
§A2's X1 repair (i), adopted, required *"`answer_accuracy` co-scored"*. §3.6
scores `retrieval24` only. The archive shows why this matters: under the exact
write, `retrieval24 = 1.0` while `answer_accuracy` at `h=1/13/37/61` reads
`0.063 / 0.051 / 0.059 / 0.020` (compA), `0.031 / 0.051 / 0.039 / 0.027`
(primary), `0.047 / 0.055 / 0.047 / 0.035` (compB), and **`0.0` at every hop
in P1a** — chance (`1/24 = 0.042`) everywhere. A
retrieval-only WIN would therefore be entirely consistent with the LM answering
at chance — and the spearhead claim is a capability *inside a real LM*. The
runner already emits `answer_accuracy` in the same dict at the same eval, so
co-scoring costs nothing.

**(e) inverted conservatism.** §3.6 calls `P1b_ref = 0.977` *"the measured
range's lower end… the harder-to-reach ceiling"*. Using the lower ceiling
**shrinks** the gap (0.907 vs 0.930) and therefore **lowers** the WIN bar:
required reading `0.7049` instead of `0.7210`. The choice may be fine; the
stated rationale is backwards and should be corrected rather than repeated.

### M6 — the budget arithmetic is exactly right, but the rate attribution is wrong, the per-cell ceiling is missing, and the figure in the table is the one that already caused two ABORTED-BUDGET runs

Every number in §3 reproduces to the digit (V6). Three problems sit on top of
correct arithmetic.

**(a) The rate attribution is factually wrong.** The draft (inheriting R2's
M2) says `0.149 s/step` was *"compB's own anomalously fast regime…
unrepresentative of general training cells by ~6×."* The raw record says
otherwise — **all three** g3b31 cells, launched within 4 seconds of each other
on 2026-07-30 (i.e. run 3-concurrent, exactly the planned Stage-1
configuration):

| cell | steps | GPU-h | s/step |
|---|---|---|---|
| `mob_g3b31_primary_s0` | 20,000 | **0.840** | 0.1513 |
| `mob_g3b31_compA_s0` | 20,000 | **0.812** | 0.1461 |
| `mob_g3b31_compB_s0` | 20,000 | **0.829** | 0.1493 |
| `mob_g3b14_s0` | 20,000 | 1.101 | 0.1981 |
| `mob_g3b20_s0` | 20,000 | 4.997 | 0.8995 |
| `mob_g3b17_s0` | 19,677 | 5.002 | 0.9151 |
| `wave1_calib_K24_s0` | 19,026 | 4.867 | 0.9209 |
| `sanity_g3b12_tf_s0` | 3,000 | 0.695 | 0.8343 |

The split is **environmental, not configurational**: same host
(`brev-ukptqsu65`), same torch (2.12.1+cu130), same `batch_size=32`, same
backbone (97.6M), and `mob_g3b20_s0` (aux 3.0 + ortho 0.1) is 6× slower than
`mob_g3b31_compB_s0` (aux 0.5 + ortho 0.1). Pricing conservatively at 0.875
s/step is defensible; **the stated reason for it is false**, and the true
lesson (the rate is contention-determined with a 6× spread and is not
predictable from config) has a different implication.

**(b) The consequence: the wave was scoped against a likely ~6×-overstated
price.** The three PRIMARY cells' own measured cost is **2.481 GPU-h total**,
priced here at 14.58. On that basis the design cut the same-recipe seed arm
that M5(c) needs and the `ortho=0` control that F2 needs — both of which fit
comfortably inside the ≤20 GPU-h ceiling even at the pessimistic rate.

**(c) No per-cell `--ceiling-gpuh` is registered, and the table's own figure
is the trap.** The archive's two ABORTED-BUDGET cells:

```
wave1_calib_K24_s0 : ceiling 4.865 -> ABORTED-BUDGET at 19,026/20,000 (95.1%)
mob_g3b17_s0       : ceiling 5.000 -> ABORTED-BUDGET at 19,677/20,000 (98.4%)
```

§3's per-cell figure is **4.861 GPU-h** — within 0.1% of the ceiling that
already aborted a 20,000-step cell at 95%. At 0.92 s/step a 4.861 ceiling
completes **19,021 of 20,000 steps**. A builder reading the GPU-h column as
the ceiling reproduces the archive's own mistake, and an ABORTED-BUDGET
Stage-1 cell is not the recipe P0 was measured on. Register explicit per-cell
ceilings with headroom (≥5.5 for a 20,000-step cell) separately from the
expected-cost table.

### M7 — no leak-guard is pre-registered for the single highest-consequence build error, though the runner already emits a tamper-evident witness

The structural guarantee holds for the *intended* wiring (V10). But the new
flag is one plausible mis-wire from `teacher_force` — they are conceptually
adjacent ("use `teacher_force_operator`"), and the design's own §3 snippet
calls `arm["integ"].teacher_force_operator(...)` inside the loss. If
`teacher_force` were ever set for a Stage-1 PRIMARY cell, eval would read
`retrieval24 ≈ 1.0` at every hop — **visually indistinguishable from the WIN
the design hopes for**, and indistinguishable from a genuine success in a
blind harvest.

The runner already records everything needed to rule it out, in every
artifact: `config.teacher_force_operator`,
`teacher_force_check.active`, `teacher_force_check.ncr_zero_grad_checks_passed`.
§3.6 should carry a **Band-0** gate: every Stage-1 PRIMARY artifact must show
`teacher_force_operator == false`, `active == false`,
`ncr_zero_grad_checks_passed == 0`, checked before any band is read. Zero cost,
and it converts an un-detectable failure into an impossible one.

### M8 — provenance: the new flag will not be recorded, so this round's own verification method would be impossible for this wave

`rec["config"]` (`runner.py:1257-1264`) enumerates every flag **explicitly**;
a flag not added there is absent from the archived JSON. Every config claim I
verified this round (V5, M6) came from that dict. §3 specifies "one new CLI
flag" and never says to add it to `rec["config"]` — so `λ_w` would be
unrecoverable from the artifacts, and a future round could not check which
weight produced which reading. Also: `run_two_arm_cell`'s resume asserts cover
`seed` and `freeze_entity_adapter` only, so a resumed cell can silently change
`λ_w` mid-run. Both must be in the build brief.

---

## minor findings

- **m1 — dead code that is also a build ambiguity.** §3's snippet computes
  `Z_ideal` under `no_grad` and then never uses it: the loss as written,
  `‖Z k_i − v_i‖²`, does not need it. A builder cannot tell from §3 whether
  the intended loss is the restricted §2(a) form or a Frobenius-to-`Z_ideal`
  form. (Under F1's repair `Z_ideal` becomes load-bearing and the snippet
  becomes correct — resolve the ambiguity in that direction.)
- **m2 — "sub-microsecond" is off by ~3 orders.** Measured on CPU: the
  batched `pinv` + matmul at `B=32`, `(24,25)` is **1.23 ms/step**, 22× the
  `L_write` matmul's own 55 µs. Small against a 0.15–0.9 s step, but batched
  small SVDs are latency-bound on GPU and this is a per-step cost; if `Z_ideal`
  is kept, measure it in the pre-launch re-smoke rather than asserting it.
- **m3 — §2(b) defends a metric §3.6 does not score.** The decode-mismatch
  risk lives entirely in `answer_accuracy`; `retrieval24` never routes through
  the decode head (`discriminability_metrics` compares `o` directly against
  adapted entity embeddings). The section's argument is sound (V13) but
  aimed at a quantity the bands drop (M5(d)).
- **m4 — no mutual-exclusion assert.** With both `--teacher-force-operator`
  and `--write-supervision-weight > 0`, `Z` *is* `Z_ideal` and `L_write ≡ 0` —
  a silent no-op. Add the assert.
- **m5 — eval cost unpriced.** §3.6 pins `eval_batch_size = 256`, 4× the
  archived 64, across 9 hops × 2 arms at every eval point; `--eval-every` is
  unspecified. Fold into the pre-launch re-smoke §3 already calls for.
- **m6 — the ε-guard claim is too strong.** §2(a) says the guard *"cannot
  create a degenerate zero-loss escape route."* The guard itself is fine; the
  escape route exists anyway, through `keys_v`/`values_v` (F3). Restate.
- **m7 — Stage 0.1's gate is non-diagnostic for the wave's most likely
  failure.** The gate is "`L_write` decreasing AND `retrieval24` not
  regressing." Across the *entire* F1 failure family, `L_write` sits at its
  floor and `retrieval24@h=1 = 1.0000` (and the continuation's in-dist eval is
  `h ∈ {1,2,3}`). The canary passes at transverse gain 100, where `h=61`
  reads chance. If Stage 0.1 is kept, its gate must include a deep-hop reading
  and the transverse gain.
- **m8 — the depth-decay PARTIAL signature has no reference band at the hops
  it is read at.** §3.6 carries "clears WIN/PARTIAL at `h≤20` but decays toward
  NULL by `h*=61`" as a labelled outcome, but `P0_ref`/`P1b_ref`/`gap`/`τ` are
  defined at `h*=61` only. compB's own P0 reads `0.0469 / 0.0352 / 0.0312` at
  `h=5/12/20` — and `0.0742` at `h=40`, *above* `P0_ref = 0.07`. Register the
  per-hop references (they are already in the archive) or state that the
  signature is qualitative and non-adjudicating.

---

## §STAGE-0 RULING — may the zero-training-cost analytic checks run pre-CLEAR?

**Split ruling.**

**0.1 (warm-start sanity, 0.365 GPU-h): NO, blocked pre-CLEAR.** It is not
zero-cost, it trains a loss that F1/F3 say is mis-specified, it would run with
`ortho_reg_weight = 0.1` (F2) at a `λ_w` that Stage 0.3 has not yet been fixed
to produce, and its gate is provably non-diagnostic (m7). Any "engagement"
reading it produced would be uninterpretable and would create pressure to
proceed.

**0.2 (reachability/conditioning) and 0.3 (residual-compounding derivation):
YES, may run pre-CLEAR — but ONLY in the REPLACED form below.** Both are
read-only, zero-GPU, and their outputs are *inputs to the Rev-4 redesign*, so
running them ahead of CLEAR shortens the loop rather than pre-committing it.
**0.2 as specified must not be the version that runs**: M2 shows it gates on a
condition that cannot fire, and running it as written would return "cond is
moderate, proceed" and manufacture false assurance. **0.3 as specified must
not run at all**: a `λ_w0` derived from residual magnitude alone, with no
`ortho` term (F2) and no transverse constraint (F1), is an actively harmful
number — it is the number that sends the wave into the NULL rows of F2's
table.

**Stage 0′ — the replacement, authorised to run pre-CLEAR (eval-only forward
passes on the three existing checkpoints; no training, no box-config change.
Honest cost: not literally zero as §3's table claims — the premise battery's
own comparable read-only pass measured ≈0.1 GPU-h — but negligible, and it
should be authorised under the battery's own precedent, not smuggled in as
"0"):**

1. `cond(keys_v)` — keep it, informational only, **non-gating** (M2).
2. `‖Z_ideal‖_F` and its **per-row** norm distribution across ≥256 real
   episodes per recipe: median, p99, max, within-episode max/min.
3. `cond(ncr.row_out.weight)` for each checkpoint, compared against (2)'s
   required dynamic range — the actual reachability test (M2).
4. **`‖Z_sgd w‖`, the transverse gain**, on all three trained checkpoints,
   with `w` the right-null vector of `keys_v` — the F1 statistic. Report
   alongside the harvest's `A_cond ≈ 9,959`: this is the number that says
   whether the observed write failure is already a transverse-gain failure.
5. `ortho_regularization_loss(Z_ideal)` and `‖∇_Z ortho‖` at `Z_ideal` on
   **real** batches, and the joint-minimisation `λ_w` curve of F2 re-run on
   real key geometry — the input to any honest `λ_w0`.
6. The `L_write → retrieval24@61` calibration curve on real geometry
   (this round's synthetic version, C1, is below; redo it on real `keys_v`).

**GATING:** items 3, 4, and 5 **must gate** Stage 1. Specifically: if (4)
shows the trained checkpoints already sit at transverse gain ≫ 3, F1 is not
merely a theoretical hole but the diagnosis of the observed failure, and the
complement term is mandatory, not optional. If (3) shows `cond(row_out)` cannot
supply (2)'s range, the target is unreachable and the mechanism needs
rescoping before any GPU-h is spent.

---

## Calibration handed to Rev-4 (from this round's executed sims)

The number Stage 0.3 was supposed to produce, measured rather than argued
(`sim_C_precision.py`, ρ=0.21 geometry, n=256, real read; in-span
perturbations, so this is the *best case* with the transverse direction clean):

| relative error | `L_write` | RMS rel. per-key error | `retr24@61` | band |
|---|---|---|---|---|
| 1e-4 | 5.8e-07 | 7.6e-04 | 0.9961 | WIN |
| 1e-3 | 5.8e-05 | 7.6e-03 | 0.9062 | WIN |
| 3e-3 | 5.2e-04 | 2.3e-02 | 0.6211 | PARTIAL |
| 1e-2 | 5.8e-03 | 7.6e-02 | 0.1680 | PARTIAL |
| 3e-2 | 5.2e-02 | 2.3e-01 | 0.0352 | NULL |

⇒ **WIN (`retr24@61 ≥ 0.705`) requires `L_write ≲ 3e-4`** — a per-key
relative RMS error under ~1.8% — **and** transverse gain `‖Z w‖ ≲ 3` (F1).
Both must be pre-registered; either one alone is insufficient.

---

## BINDING DISPOSITION PROPOSAL (for the coordinator)

**Verdict: BLOCKED. Rev-4 required before any Stage-1 GPU-h. Stage 0′ (above)
authorised to run pre-CLEAR.**

**D1 (F1, mandatory).** Respecify the distance to constrain `Z` off-span.
Preferred: the full Frobenius form to `Z_ideal` with a per-episode normaliser
(which makes §3's own snippet correct and is decomposable into the §2(a) term
plus `‖Z(I − P_span)‖²_F`). Minimum acceptable: keep §2(a)'s term and add an
explicit complement penalty. Rewrite §2(a)'s rejection 1 and 2 — the current
text argues for the defect. Register `‖Z_sgd w‖` as a logged per-eval
diagnostic with the `≲ 3` calibration.

**D2 (F2, mandatory).** Resolve `ortho_reg_weight` explicitly. Preferred:
`ortho_reg_weight = 0` in all write-supervised arms, **plus** a fresh
`ortho=0, λ_w=0` no-supervision cell so the baseline is matched (P0 was
measured with ortho on and is no longer the right comparator). If ortho is
kept, register the λ_w floor from Stage 0′ item 5 and disclose the arm as a
two-term hybrid in §5. Either way §3 must state the setting; silence is not
an option.

**D3 (F3, mandatory, one word).** `keys_v.detach()`, `values_v.detach()` in
the loss call. Add `target_pairwise_cos` as a Band-1 tripwire.

**D4 (M1).** Quotient the global scale (closed-form `c*` rescale before
scoring), or state explicitly that `L_write`'s absolute value is not a
progress metric and re-define Band 2 on the rescaled residual.

**D5 (M5).** Rewrite §3.6 as an enumerated **partition** with the GAP clause
inside it (closing the `[0.705, 1.0] × GAP≤0.15` hole); add an explicit
cross-recipe aggregation rule covering all 27 outcomes; co-score
`answer_accuracy` at `h*` in every band row; correct the `P1b_ref`
conservatism sentence; and either fund same-recipe seeds or drop the `n≥3`/CI
framing in favour of "robustness sweep across three recipes."

**D6 (M6).** Re-attribute the rate honestly (the split is environmental, the
three target cells' own measured cost is 2.481 GPU-h total); register explicit
per-cell `--ceiling-gpuh` ≥ 5.5 for 20,000-step cells, separately from the
expected-cost table; and re-scope the wave with the headroom that buys — the
`ortho=0` control (D2) and same-recipe seeds (D5) both fit under ≤20 GPU-h.

**D7 (M3, M4).** Re-scope the placebo's readout to conditioning transfer and
log per-step `‖∇_Z‖` for both arms; fix Control B by naming the separate
clean-eval script (the battery's own `pbe_repl`/`pbe_supplement` pattern) and
either freezing the adapter for the continuation or dropping the "isolates the
read side" claim.

**D8 (M7, M8, m1, m2, m4, m5).** Build-brief items: Band-0 teacher-force
leak gate on every artifact; `--write-supervision-weight` added to
`rec["config"]` and to the resume asserts; mutual-exclusion assert; resolve the
`Z_ideal` dead-code ambiguity; price the eval at 256; measure the `pinv` cost
in the re-smoke.

**Ceremony.** The wave is 20–30 GPU-h and publication-adjacent (its verdict
would feed the spearhead's real-LM ladder), so it stays in the >10 GPU-h tier:
Rev-4 → one further narrow adversarial round scoped to D1/D2/D3 (the three
FATALs' repairs) → pre-launch resource/placement red-team → launch. §5's
novelty sweeps remain PENDING and independent of this verdict; note for the
gate that D1's repair (regression to a closed-form least-squares fast-weight
target, complement included) is *closer* to the TTT/state-distillation
literature the sweep must check than the restricted form was, so the sweep
should be run against the **repaired** loss, not the drafted one.

---

## Reproduction

All sims are pure-CPU, fp32/fp64 torch 2.8.0, no GPU, no box contact. They
import the **real** `nm.binexp_read` / `nm.loop_read`
(`matrix-thinking/ncr/ncr_models.py`) and reproduce `teacher_force_operator`,
`ortho_regularization_loss` and `discriminability_metrics`'s `retrieval24`
verbatim from the pinned runner.

```
scratchpad/sim_common.py            harness (keys/values from a single Hamiltonian
                                    K-cycle, Z_ideal, L_write, retrieval24)
scratchpad/sim_A_keygeom.py         M2  — key geometry, cond sweep, exact-write robustness
scratchpad/sim_B_freedir.py         F1  — the free-direction walk (fp32 + fp64)
scratchpad/sim_C_precision.py       F1/calibration — precision requirement, absolute
                                    transverse-gain threshold, o_pairwise_cos
scratchpad/sim_D_ortho_placebo.py   F2/M3 — ortho conflict + joint minimisation;
                                    placebo gradient budget
scratchpad/sim_E_mechanism.py       F1  — binexp vs loop_read (the mechanism is
                                    transverse amplification, not matrix powering)
scratchpad/sim_F_bands_budget.py    M5/M6 — band partition scan, τ, budget re-derivation
scratchpad/sim_G_reachability.py    M1/M2 — row-norm dynamic range, global-scale invariance
```

(scratchpad root:
`/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad`)

Raw artifacts read this round:
`experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/*.json` (configs,
elapsed_s), `experiment-runs/2026-08-13_ncr_writecond_premise_battery/*.json`
(P0/P1b/P1a/REPL/SUPP — `target_pairwise_cos`, `answer_accuracy`,
`retrieval24_acc`), and the archive's timing/ABORTED-BUDGET cells
(`wave1_calib_K24_s0`, `mob_g3b17_s0`, `mob_g3b20_s0`, `mob_g3b14_s0`,
`sanity_g3b12_tf_s0`).
