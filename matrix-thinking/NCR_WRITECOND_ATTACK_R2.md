# NCR write-conditioning — ADVERSARIAL ATTACK, ROUND 2

**Target:** `matrix-thinking/NCR_WRITE_CONDITIONING_DESIGN.md`, section
`## DRAFT-R1 (Rev-1, 2026-08-13)` ONLY (§1–§6 are the dead DRAFT-R0,
attacked in round 1 and not re-attacked here).
**Repo commit:** `0ff2ca8`. **Date:** 2026-08-13. **Round:** 2 of the
full multi-round gauntlet (`§A1-ADJUDICATION` W7).
**Prior round:** `NCR_WRITECOND_ATTACK_R1.md` = BLOCKED (5F/11M/7m),
adopted in full as the Rev-1 charter W1–W7.

---

## VERDICT: **BLOCKED**

**5 FATAL / 12 MAJOR / 8 minor.**

Every FATAL is demonstrated, not argued: four by executed code
(scratchpad `v1_mechanism.py`, `v2_stats_and_null.py`,
`v3_ideal_write.py`, `v4_tpc_anchored.py`, `v5_rank_blind.py`), one by a
raw artifact the design and attack R1 both missed, and one (F5) by the
pinned runner's own source plus a web-verified external sweep.

The three headline results:

1. **The replacement mechanism (W3) reproduces the exact failure mode
   that killed the mechanism it replaced.** `L_conf`'s claimed
   "zero at the ideal K-cycle write, proved, not asserted" is FALSE in
   this configuration. The proof conflates *similarity* with *orthogonal
   equivalence*; `f` is a singular-value functional and is not
   similarity-invariant. Measured against compB's OWN target geometry,
   the ideal write scores **`f(A*) ≈ 149–186`** where a random Gaussian
   scores **24.5** and `c·I` scores **0** — the true solution is ~6×
   MORE penalised than noise and infinitely more than the null operator,
   and the penalty's own descent direction monotonically destroys deep
   retrieval. This is F3, re-instantiated.

2. **W1's premise battery rests on a falsified archival claim.**
   "The teacher-force diagnostic has never been re-run on the
   architecture that fixed the bug that sank its last verdict"
   (`:851-852`) is false. It was re-run 3h06m after the fix landed, is
   written up as **§G3-B13**, and reads `answer_accuracy = 1.0000` at
   **every hop h=1…61** on the fixed architecture. §G3-B13 is cited
   **zero times** in this design and **zero times** in attack R1.

3. **The novelty wedge is occupied on both sides.** Internally: the
   pinned runner already trains `0.1·‖ZᵀZ − I_d‖²_F/d²` on `Z` in all
   three baseline cells — a soft-orthogonality/flatness penalty on the
   written operator, running inside the very baseline the design says
   needs one. Externally: "no external occupant" fails on three counts,
   including a functional identity (the loss is a reparameterisation of
   the matrix Rényi-2 entropy of the trace-normalised Gram, a named
   object that has been trained through). R1.4's "no gate re-entry is
   triggered" does not hold.

**Premise-battery launch ruling (W7's ≤2 GPU-h pre-authorization):**
**NOT AS SPECIFIED — four narrow, documentary repairs first.** See
§LAUNCH-RULING. The battery remains the right cheap cell; its bands,
its decision tree, its `n`, and its budget fallback are all unsound as
written.

---

## What is VERIFIED CLEAN (recorded so the next round does not re-litigate)

Adversarial checks that the draft **passed**:

| # | Claim | Verdict |
|---|---|---|
| V1 | `L_conf = λ‖G−tI‖²_F/t²` ⇔ `λ[K²tr(G²)/tr(G)² − K]` ⇔ `λ[K²Σsᵢ⁴/(Σsᵢ²)² − K]` | **VERIFIED** — all three agree to 1e-10 on random / spiked / permutation / `c·I` / rank-2 inputs (`v1` §1) |
| V2 | `∂f/∂A = (4K²/D³)·A·(D·G − N·I)`; `∂L/∂Z = U(λ∂f/∂A)Uᵀ` | **VERIFIED** independently vs central finite differences: max rel err **3.5e-8** (random), **1.3e-9** (spiked) (`v1` §2) |
| V3 | Scale-invariance `f(αA)=f(A)`, no `‖Z‖_F²` denominator needed (M4's fix) | **VERIFIED** exactly over α ∈ {1e-3, 1, 1e3} (`v1` §3) |
| V4 | Cauchy–Schwarz bound: `f ≥ 0`, `= 0` iff flat; closes M3's rank-2 hinge route | **VERIFIED** — `(1,1,ε…ε)` scores `f = 264`, not the exact 0 M3 found in DRAFT-R0's hinge. **The R1-MAJOR rank-2 route IS closed in this formulation.** |
| V5 | `‖A‖=‖A⁻¹‖⁻¹` bounds every `\|λᵢ\|` between the equalised singular values | **CORRECT** as stated; the non-normality residual is disclosed, not hidden, and `depart_normality` is co-scored — an honest carry of M2 |
| V6 | `0.8293 GPU-h` per 20,000-step cell | **VERIFIED** — it is literally `mob_g3b31_compB_s0.json:gpu_h = 0.8293428…` (`elapsed_s = 2985.63`). The number is honest; only its generalisation is not (M2 below) |
| V7 | Every §R1.6 Band-1 anchor | **VERIFIED to 5 decimals against the raw JSON** — see §W6-SPOTCHECK. compB h=40 violation and the PARTIAL signature both appear as recorded facts, correctly |
| V8 | `h∈{1,13,37,61}`: 13 ≡ 37 ≡ 61 ≡ 13 (mod 24) | **VERIFIED and worth keeping** — all three share effective distance 13 at different squaring counts. A genuine squaring-count discriminator, correctly motivated |
| V9 | `D_h` is not saturated at compB's operating point | **VERIFIED** — compA reaches `D = 3.5e-6`, so compB's 0.010 plateau has ≥3 decades of headroom. It is real signal, not an instrument floor. (But see m4: no VOID band is registered.) |
| V10 | The runner pin `9a93198b642242f512ff8489e32b0a53` | **VERIFIED** by `md5` on disk |

---

## FATAL FINDINGS

### F1 (FATAL, mechanism — W3) — `L_conf` is NOT zero at the ideal write in this configuration; it penalises the true solution ~6× harder than noise, and its descent direction destroys the capability

**The claim under attack** (`:1106-1113`):

> **Zero at the ideal K-cycle write, proved, not asserted:** the true
> `z_ideal`'s entity block is (isomorphic to) a `K×K` permutation matrix
> — orthogonal, all `sᵢ=1` — so `L_conf(A^*)=0` exactly.

**The defect.** "Isomorphic to" is a *similarity*: `A* = S·P·S⁻¹`, where
`S` is the matrix of the K key vectors expressed in the orthonormal
entity basis `U`. `f` is a function of **singular values**, which are
invariant under *orthogonal equivalence* (`A → QAR`, `Q,R` orthogonal),
**not** under similarity. `f(SPS⁻¹) = 0` **iff `S` is a scalar multiple
of an orthogonal matrix — i.e. iff the K adapted entity vectors are
orthonormal.**

That is true in the TOY program by construction: `NCR_ORTHO_WRITE.md`'s
`synthetic_keys_from_pi` (`analyze_zdump.py`, module DERIVATION) *builds*
orthonormal keys from Π's eigenbasis. It is false in the LM graft, where
`keys_v = entity_adapter(embed(token_id))` — 24 adapted GPT-2 embeddings
through a `Linear(768,25,bias=False)`.

**Executed demonstration** (`v3_ideal_write.py`, `v4_tpc_anchored.py`).
The ideal write is built exactly as the runner does it:
`Z* = integ.teacher_force_operator(keys_v, values_v) = (pinv(k)@v)ᵀ`,
`U = az.entity_subspace(Z*)`, `A* = UᵀZ*U`, single full 24-cycle.

| key geometry (source) | mean pairwise cos | `f(A*)` | `cond(A*)` |
|---|---|---|---|
| orthonormal keys (the TOY assumption) | 0.0000 | **0.000** | 1.00 |
| **compB measured `TPC_fg` = 0.2083 (h=1)** | 0.2083 | **148.7** | 292.0 |
| **compB measured `TPC_fg` = 0.2277 (h=29)** | 0.2277 | **170.8** | 249.1 |
| raw GPT-2 entity embeds (§G3-B26: 0.0837) | 0.0837 | **195.2** | 214.6 |
| primary measured `TPC_fg` = 0.7223 | 0.7223 | **167.7** | 223.8 |
| compA measured `TPC_fg` = 0.8138 | 0.8138 | **127.8** | 78.4 |
| — reference — random Gaussian `K×K` | — | **24.55** | — |
| — reference — permutation, and `c·I` | — | **0.00** | 1.00 |

Every measured target geometry in the archive puts the ideal write at
`f ≈ 128–195`, i.e. **~6× more penalised than a random matrix and
infinitely more penalised than `c·I`.**

**And the descent direction is destructive**, not merely misaimed
(`v4`, flattening `A*`'s spectrum toward its mean, retrieval24 in
float32 at h=1/13/37/61):

```
f(A*) = 186.0     retrieval24(A*)  = [1.000, 0.708, 0.250, 0.208]
flatten  25%  f=141.8   retrieval24 = [1.000, 0.042, 0.000, 0.000]
flatten  50%  f= 78.9   retrieval24 = [1.000, 0.042, 0.083, 0.000]
flatten  75%  f= 19.3   retrieval24 = [1.000, 0.083, 0.042, 0.000]
flatten 100%  f=  0.0   retrieval24 = [1.000, 0.083, 0.083, 0.000]
```

A **25%** move along the penalty's own gradient collapses h=13
retrieval from 0.708 to 0.042. The penalty's global minimum is a
configuration with **zero** deep retrieval.

**This is F3 verbatim, one mechanism later.** Attack R1 killed mechanism
(b) because "the IDEAL K-cycle write scores ≈max penalty… while
Band-2 reads 1.00000 down to zero capability." The replacement scores
the ideal write at 149–186 against 0 for `c·I`. The charter's own
justification for W3 — "zero at the ideal K-cycle write" — is the single
property the replacement was chosen for, and it does not hold.

**Why there is no cheap patch.** With the map fixed (`k_i ↦ k_{σ(i)}`),
`f(A*)=0` **⟺ the keys are orthonormal**. So the only
task-compatible way to satisfy `L_conf` is for `entity_adapter` to learn
near-orthonormal entity vectors. That is (i) not what the design says
the mechanism does, (ii) not instrumented (the design tracks
`depart_normality`/`A_cond`/`s1/s2`, never key-Gram orthonormality),
(iii) contradicted by compB's own converged `TPC_fg = 0.21`, (iv)
structurally impossible in the frozen-adapter cells, and (v) — decisively
— **a target-space-geometry intervention, i.e. a re-entry into the
§G3-B17–B32 contrastive/aux road this very document declares
EXHAUSTED** (`:1155-1160`). The repair route lands inside the closed
lane.

---

### F2 (FATAL, frame/archive — W1) — the premise battery's central archival claim is false; the post-fix teacher-force re-run EXISTS, is §G3-B13, and reads `answer_accuracy = 1.0` at every hop to h=61

**The claim** (`:851-852`, bolded in the draft):

> **The teacher-force diagnostic has never been re-run on the
> architecture that fixed the bug that sank its last verdict.**

**Falsified by direct raw-artifact read** (I read the JSONs myself, per
the coordinator-tiebreak rule; an agent's prose was not relied on):

`experiment-runs/2026-07-17_ncr_gate3_wave1/g3b12_smoke_results/sanity_g3b12_tf_s0.json`

| field | value |
|---|---|
| `cell_id` | `sanity_g3b12_tf_s0` |
| `started_utc` | `2026-07-18T07:36:45Z` (g3b9 ran `04:30:14Z` — **3h06m earlier**) |
| `config.teacher_force_operator` | `true` |
| `teacher_force_check` | `{active: true, ncr_zero_grad_checks_passed: 3000}` |
| `params.integ` | **38400** — the POST-fix single-`entity_adapter` count (§G3-B12: "38,400… was 57,600"). `g3b9_tf_diag.json` records **57600**. |
| `full_graft answer_accuracy` | **1.0000 at h = 1,2,3,5,12,20,29,40,61** |
| `full_graft mean_cos` | 1.0000 at every hop (0.9971 at h=61) |
| `full_graft recovered_frac@0.9` | 1.0000 (0.9844 at h=61) |
| `backbone_only` | null throughout (0.0–0.094) |

It is written up as **§G3-B13** (`NCR_REAL_LM_DESIGN.md:5591`),
verdict "DECODE PATH FULLY HEALTHY. The GATE-3 harness is proven
end-to-end," and is cited 11× elsewhere in that file, including
"**BANK the §G3-B13 result**" (`:6057`).

`grep -c "G3-B13"` → **`NCR_WRITE_CONDITIONING_DESIGN.md`: 0**,
**`NCR_WRITECOND_ATTACK_R1.md`: 0**. The design's own §R1 header lists
the sections it read: "§G3-B9/B10 … and §G3-B25/B26/B27/B28/B31/B32" —
it jumps over B11's remedy and B13's result.

**Three consequences, each independently damaging:**

**(a) P1a/P1b's stated value is void.** The draft sells them as closing
"the never-rerun-post-bugfix hole." There is no such hole. §G3-B13 is a
3,000-step post-training teacher-force cell; **P1b is a 5,000-step
post-training teacher-force cell** — a near-replica of a banked positive,
presented as an open question.

**(b) The g3b9 verdict is characterised as standing when the archive
records it as VOID.** The draft says g3b9 "returned READ/setup-broken,
NOT a WRITE-blocker — but on the OLD architecture," i.e. a live result
with an architecture caveat. §G3-B11 retracted it outright:
"**the §G3-B10 diagnostic is MIS-SPECIFIED; its own premise ('handed the
TRUE operator directly') is FALSE**" (`:5214-5215`). The draft also
enumerates three §G3-B11 defects; §G3-B11 found **four** ("Four distinct
defects, all demonstrated", `:5257`). The omitted **(3d)** — "the
recovery INSTRUMENT is mis-based → reads ~0 even for a perfect read…
a PERFECT read scores `mean_cos = 0.05`" — is precisely the defect that
kills the `mean_cos` evidence the draft leans on. The omission runs in a
self-serving direction.

**(c) The registered bands are mis-calibrated by an order of
magnitude.** P1a and P1b are gated at `τ = 0.0916`. The archived
reference for this configuration is **1.0**, and P1a's h=1 value is
**analytically determined**: `q_key ≡ keys_v[a_slot]` bit-identical
(the §G3-B12 fix, asserted every launch), and `pinv` fits 24 constraints
in 25 dimensions exactly, so `o = Z q_key = values_v[a_slot] = T_{tgt}`.
Simulated (`v3`): residual `3.1e-16`, **retrieval24 = 1.0000 at
h=1/13/37/61**, `‖Z^h‖₂` flat at 28.1. A P1a/P1b reading of, say, 0.12 —
a catastrophic 8× drop from the banked reference — scores **CLEARS** and
routes to "**AUTHORIZE STAGE 1 AS SPECIFIED**."

**What survives.** A genuinely narrower cell: §G3-B13 was never scored
with `retrieval24_acc`/`discriminability_metrics` (introduced at
§G3-B27, 2026-07-29 — confirmed: `grep -rl retrieval24 --include=*.json`
returns only the 2026-07-30 files), never at `n=256`, never at
`h∈{13,37}`. Re-scoring a banked-positive configuration with the new
primary instrument is worth ~0.2 GPU-h — but it must be pre-registered
against a reference of **1.0**, with `answer_accuracy` co-scored so the
two instruments are cross-checked on one config. A disagreement between
them would be a first-order instrument finding, and **the tree has no
branch for it.**

---

### F3 (FATAL, control — W5) — the "true null" delivers the active ingredient AND is 141× weaker in coherent pressure; F5's failure mode recurs

The draft's invertibility argument (`:1254-1260`):

> **Un-invertible, provably:** `ε_t` is isotropic and mean-zero, so for
> ANY fixed direction `Δ` (in particular the flattening direction
> `∂f/∂A` itself), `E[⟨ε_t,Δ⟩]=0` — the null carries no systematic
> component toward flatness… in expectation at every single step.

**`E[⟨ε,Δ⟩]=0` is a first-order statement about a nonlinear
functional.** What matters is `E[f(A+ε)]`, whose leading correction is
`½σ²·tr(∇²f)` — and for a spread spectrum that term is strongly
negative: isotropic perturbation lifts small singular values off the
floor and regresses the spectrum toward the Marchenko–Pastur bulk, i.e.
**toward flatness**.

**Executed** (`v2` §d, spiked `A`, `f(A) = 341.42`, 400 draws/row):

| `‖ε‖_F/‖A‖_F` | `E[f(A+ε)]` | ΔE |
|---|---|---|
| 0.178 | 322.72 | **−18.70** |
| 0.356 | 274.17 | **−67.24** |
| 0.712 | 164.69 | **−176.73** |
| 1.423 | 58.96 | **−282.46** |

And the accumulated random walk — which is what `Z_raw.grad += ε_t`
every step actually produces:

```
step  500:  f = 280.17    step 2000:  f = 195.47
step 1000:  f = 235.13    step 4000:  f = 139.91   (from 341.42)
```

**The null flattens the spectrum.** It administers the treatment's
active ingredient, at a dose set — by the draft's own construction — to
match the treatment's gradient norm. F5's charge ("the placebo delivers
the active ingredient") is not discharged; it is re-instantiated in a
form whose disclaimer ("provably un-invertible") is stronger than the
one W5 replaced.

**Second, independent defect: norm-matching is not pressure-matching.**
A coherent gradient of norm `g` displaces `Z` by `T·lr·g` over `T`
steps; an isotropic noise of the same per-step norm displaces it by
`√T·lr·g`. At `T = 20,000` that is a **141×** gap. So the null is
simultaneously *contaminated* (delivers flattening) and *under-dosed*
(141× less coherent displacement). "Placebo does not improve AND
`L_conf` does ⇒ structural evidence specific to spectral flatness"
(`:1269-1271`) does not follow from either direction: the generic
"extra coherent pressure on `Z_raw`" confound — the exact confound
§G3-B22–B25 diagnosed once already — remains completely unblocked.

**Adjudication of the reviser's own disclosed risk (§R1.8 item 1),
as instructed.** `g_t` is measured on the null arm's untouched `Z_raw`.
Because `f` is scale-invariant, `‖∂f/∂A‖ ~ f'/‖A‖`, so a *spiky* (null)
arm carries a systematically **larger** gradient norm than a *flattened*
(treatment) arm, whose `∂f/∂A → 0` as `f → 0`. The two mismatches are
opposite-signed and do not cancel: the null is **over**-dosed in norm
and **under**-dosed in coherence, both growing over training. **Verdict:
not "acceptable with monitoring" — the decoupling is real, directional,
and unmeasurable from the null arm alone.** The only sound fixes:
(i) run the treatment arm first at matched seed, log its realised
per-step `‖∇L_conf‖` series, and replay THAT schedule into the null
(paired by seed and step); AND (ii) add a second, *coherent* control —
a matched-gradient-norm penalty toward a fixed random NON-conformal,
NON-orthogonal target (D4's original), which the draft considered and
rejected (`:1260-1263`) for the very property that makes it the
necessary control.

---

### F4 (FATAL, decision tree — W1) — R-A…R-F is not a partition; 3/8 outcome cells fire ≥2 branches, two of them pitting "AUTHORIZE ≤25 GPU-h" against "KILL"; and `CLEARS(x)` has no depth quantifier

Enumerating all eight outcomes of `(CLEARS(P0), CLEARS(P1a),
CLEARS(P1b))` against the literal branch predicates (`v2` §f):

| P0 | P1a | P1b | branches that fire | |
|---|---|---|---|---|
| 0 | 0 | 0 | R-C RE-SCOPE(pipeline), **R-F KILL LANE** | **COLLISION** |
| 0 | 0 | 1 | **R-A AUTHORIZE**, **R-C RE-SCOPE/kill** | **COLLISION** |
| 0 | 1 | 0 | R-D RE-SCOPE(adapter) | ok |
| 0 | 1 | 1 | R-A AUTHORIZE | ok |
| 1 | 0 | 0 | R-C RE-SCOPE(pipeline) | mis-routed (below) |
| 1 | 0 | 1 | **R-B AUTHORIZE**, **R-C RE-SCOPE/kill** | **COLLISION** |
| 1 | 1 | 0 | R-D RE-SCOPE(adapter) | mis-routed (below) |
| 1 | 1 | 1 | R-B AUTHORIZE | ok |

**(a) Two collisions oppose AUTHORIZE against KILL.** Cells `(0,0,1)`
and `(1,0,1)` simultaneously satisfy an AUTHORIZE branch and R-C
("kills write-conditioning as specified"). A pre-registration that
permits the same data to be adjudicated either way is not a
pre-registration. This is the K-wall partition lesson exactly.

**(b) Two mis-routes discard the highest-value possible finding.**
Cells `(1,1,0)` and `(1,0,0)` have **`CLEARS(P0)` true** — i.e. the
actually-deployed SGD-learned `Z` retrieves above chance at h=1, which
**falsifies F1's premise** and re-founds the entire document. `(1,1,0)`
routes to R-D "out of this document's scope"; `(1,0,0)` routes to R-C,
whose stated rationale — "A perfect `Z` can't be the fix if a perfect
`Z` already fails" — is *self-contradicted in the cell where it fires*
(the imperfect `Z` succeeded). There is **no branch keyed on
`CLEARS(P0)` alone**, though R-B's own text concedes P0 clearing would
"contradict F1's archived 3/3 chance-at-h=1 finding."

**(c) `CLEARS(x)` has no depth quantifier.** The predicate is defined
(`:926`) as "`retrieval24_acc(x) > τ`" while every cell is measured at
`h∈{1,13,37,61}`. Is `CLEARS` ∃h, ∀h, or h=1? R-F's text ("at any
`h∈{1,13,37,61}`") implies ∃h; §R1.1's prose ("does ANY configuration
retrieve above chance **at h=1**") implies h=1. Under ∃h the tree reads
completely differently from under h=1 — and the draft supplies both
readings. The trigger is unresolved in the K-wall sense (bare-literal
trigger resolution).

**(d) With F2's reference value the bands are mis-set.** Both
teacher-force cells have an archived/analytic expectation of **1.0**
against a registered bar of **0.0916**. Nine-tenths of the informative
range sits inside "CLEARS."

**(e) No branch exists for instrument disagreement** — the outcome
§G3-B13 makes most likely if `retrieval24` and `answer_accuracy` diverge
on the same config.

---

### F5 (FATAL, novelty + frame + bundling — W3/R1.4) — the wedge is occupied on BOTH sides: internally by a penalty already running at weight 0.1 in the baseline being "fixed", externally on three independent counts

#### F5-A (internal) — the pinned runner ALREADY trains a soft-orthogonality penalty on `Z` at weight 0.1 in the baseline being "fixed"

`ncr_lm_wave1_runner.py:714-742`:

```python
def ortho_regularization_loss(Z):
    """ortho_loss = mean_B( ||Z^T Z - I_d||_F^2 ) / d^2 ..."""
```

applied in `compute_arm_losses` (`:855-857`) to `full_graft` only, and
**`ortho_reg_weight = 0.1` in all three §G3-B31 cells** (verified in
`mob_g3b31_{primary,compA,compB}_s0.json:config`). The design records
it once, as `ortho=0.1` inside the base-recipe string of the **dead**
§3.1 (`:347`), and once as a function it "read in full" (`:795`).
**DRAFT-R1 never confronts it.**

`‖ZᵀZ − I‖²_F` is zero iff every singular value of `Z` equals 1 — a
*flatness* penalty on the written operator, with the scale additionally
pinned. `L_conf` is the same functional family: soft
orthogonality/restricted-isometry on `Z`, minus the scale pin, plus a
subspace restriction. Four consequences:

**(a) R1.4's novelty verdict is falsified internally.** The memo's wedge
— "differentiable condition-number/restricted-isometry penalty on the
written state — **no external occupant**" — is occupied *internally*, by
a term that has been running in this program since §G3-B20. R1.4 asserts
"**no gate re-entry is triggered by this section**." Per CLAUDE.md's
novelty gate the internal-archive sweep exists precisely so we "don't
redo or contradict our own recorded work"; its CLEAN verdict does not
survive.

**(b) There is an unacknowledged NULL data point on the mechanism.**
compB carries `0.1·‖ZᵀZ−I‖²/d²` and still exhibits o-collapse and
retrieval at chance. A flatness pressure on `Z` at strength 0.1 is
already known **not** to fix the failure. The design presents its
mechanism family as untried.

**(c) The incremental content at K=24, d=25 is thin, and measurable.**
Gradient cosine between `∇L_conf` and `∇(ortho_reg)` (`v1` §6):

| `Z` regime | cos(∇L_conf, ∇ortho) | ‖∇L_conf‖ (λ=1) | 0.1·‖∇ortho‖ |
|---|---|---|---|
| random N(0,1) | **+0.412** | 3.918 | 0.875 |
| σ_max ≈ 13 (the §10.7 regime) | **+0.394** | 1.688 | **16.18** |
| near-orthogonal (ortho converged) | **+0.822** | 2.696 | 0.0004 |
| spiked | **+0.503** | 80.78 | 0.323 |

Substantially parallel everywhere, and at the measured operating
magnitude the *deployed* term's gradient is already **~10× larger** than
the new one at λ=1. The draft's own §R1.8 item 2 concedes the subspace
restriction "barely differs from global" at K=d−1 — with `ortho_reg`
in the picture, that concession becomes the whole story.

**(d) Undisclosed bundling.** Stage-1 cells are "compB's exact recipe
**PLUS exactly one new term**" — so every treatment cell runs **two**
near-collinear spectral penalties with **different scale conventions**
(one pins `sᵢ→1`, the other is scale-blind). The design never states
whether `ortho_reg` stays at 0.1, and either answer is a defect: keep it
and the arms are confounded and gradient-conflicting; drop it and the
treatment differs from the baseline on **two** axes at once.

**The honest reframe** this forces: at this config the proposal is
closer to *a scale-invariant re-parameterisation and strength sweep of
an already-deployed regulariser* than to a new mechanism. That is a
legitimate experiment — but it is a different claim, with a different
novelty posture and a different (cheaper) design, and per the standing
re-verification gate a reframed headline re-enters the novelty gate.

#### F5-B (external) — "no external occupant" is falsified on three counts, one of them a functional identity

An independent web-verified sweep (queries and links recorded below;
every item was retrieved, none cited from memory) puts the memo's
by-mechanism verdict — *"differentiable condition-number/restricted-
isometry penalty on the written state — no external occupant"*, which
R1.4 leans on to assert "**no gate re-entry is triggered by this
section**" — outside the defensible range:

1. **RIP framing is occupied.** Bansal, Chen & Wang, *Can We Gain More
   from Orthogonality Regularizations in Training Deep CNNs?*
   (arXiv **1810.09102**, NeurIPS 2018) defines **SRIP**,
   `λ·σ(WᵀW − I)`, explicitly as a differentiable restricted-isometry
   penalty, alongside SO `λ‖WᵀW−I‖²_F` — i.e. *exactly* the deployed
   `ortho_regularization_loss` of F5-A. Also Cissé et al., *Parseval
   Networks* (**1704.08847**, ICML 2017); Vorontsov et al.
   (**1702.00071**, ICML 2017) with an explicit singular-value margin
   around 1.
2. **The name collides.** Feng et al., *Lipschitz Constant Meets
   Condition Number* (**2503.20454**) proposes a
   "**Scale-Invariant Condition Number Constraint**" — the wedge
   sentence almost verbatim. Its mathematics differs (a log-Frobenius
   surrogate blind to spectral shape), which is precisely why it must be
   cited and distinguished rather than discovered by a reviewer.
3. **The functional itself is a named, standard object that has been
   backpropagated.** `f = K²Σsᵢ⁴/(Σsᵢ²)² − K` is an affine, strictly
   monotone reparameterisation of the **collision probability /
   matrix-based Rényi-2 entropy** of the trace-normalised Gram
   (Yu et al., **1808.07912**, IEEE TPAMI: `S_α = (1−α)⁻¹log tr(Aᵅ)` on
   a trace-normalised `A`; at α=2 this is `−log tr(A²)`). Its α=1
   sibling is a published differentiable regulariser (Hyung et al.,
   **2406.11672**, NeurIPS 2024, effective-rank regularization);
   scale-invariant spectrum-flatness *training losses* already exist at
   ICLR 2024 (Rudman & Eickhoff, I-STAR, **2305.19358**, on IsoScore
   **2108.07344**); and the coefficient-of-variation form —
   algebraically `√(f/K)` — is used as a "Spectral Isotropy
   Regularization" in **2605.29987**. Anchoring to a *scaled* identity
   also has precedent (Zhang et al., **2305.17326**, ICML 2024,
   uniformity to `(1/d)I`).

**What actually survives as novel, stated precisely:** not the penalty,
but the conjunction of (a) the **conformal anchor** `(tr(AᵀA)/K)·I`
rather than `I` — relaxing orthogonality to conformality, which the
sweep found absent from the entire soft-orthogonality line — and (b) the
**object**: an in-context-*written* fast-weight operator conditioned by a
**differentiable auxiliary loss**, where every located occupant
(MuonSSM **2606.30461**, DeltaProduct **2502.10297**, Lattice
**2504.05646**, Gated DeltaNet-2 **2605.22791**) conditions the state
**by construction** — an operator or parameterisation, never a loss.
That is a defensible, narrow claim. "The penalty is new" is not.

**The premise is inherited, not discovered.** "Spectral spread degrades
repeated composition" is standard: unitary RNNs (Arjovsky et al.,
**1511.06464**, ICML 2016) and dynamical isometry (Pennington et al.,
**1711.04735**, NeurIPS 2017; Xiao et al., ICML 2018) — "all singular
values of the Jacobian concentrated near 1" is the canonical statement
of §R1.2's own thesis. MuonSSM measures the fast-weight instance
directly (κ ≈ 2.2e6 → 1.2e5). §R1.2 presents this as its own derivation.

**Two citation-integrity items** the sweep flagged and could NOT
confirm — do not ship either until checked: the design calls MuonSSM
"**ICML 2026 Oral**" (`:1179`); that venue claim was not confirmed on
the arXiv record (the id and title DO resolve). And the α value used in
**2102.00533** was not confirmable from the abstract.

**Mandatory cite-and-distinguish set** (all verified to resolve):
1810.09102, 1704.08847, 1702.00071, 1911.12207, 1709.06079, 2503.20454,
Behrmann et al. (AISTATS 2021), 1808.07912, 2102.00533, 2406.11672,
2305.19358, 2108.07344, 2605.29987, 2210.11464, 2305.17326, 1511.06464,
1711.04735, Xiao et al. (ICML 2018), 2606.30461, 2502.10297, 2504.05646,
2605.22791.

---

## MAJOR FINDINGS

### M1 — `τ`'s `n=256` is unspecified in a runner whose eval batches are deterministic; the naive implementation silently yields `n_eff = 64` and a 2.0-SD bar

Arithmetic verified: `p=1/24`, `SD_256 = 0.024978/√4 = 0.012489`,
`τ = 0.041667 + 4(0.012489) = 0.091623` ✓ (`v2` §a).

But `eval_arm_at_hops` (`:940`) draws its batch from
`torch.Generator().manual_seed(base_seed + EVAL_SEED_OFFSET + h)` — a
**deterministic function of (seed, h)**. "`n=256` (4× `eval_batch_size=64`
pooled)" is never operationalised. Four calls at the same `base_seed`
return **four byte-identical batches**; the pooled accuracy equals the
single-batch accuracy and `n_eff = 64`:

| implementation | effective SD | `τ` in SD | exact one-sided P per test | familywise (8 tests) |
|---|---|---|---|---|
| 256 distinct docs (intended) | 0.012489 | 4.0 | **2.1e-4** | ≤1.7e-3 |
| 4× the same 64 docs (naive) | 0.024978 | **2.0** | **0.0502** | **≤0.402** |

Two further points: the draft's "`≈3.2×10⁻⁵`" is the **normal
approximation**; the **exact binomial** tail at `τ` with n=256 is
**2.1e-4**, 6.6× larger (right-skew at `p=1/24`). And raising the eval
draw to 256 quadruples eval-time activations while §R1.7 reuses the
§G3-B31 **6.86 GB** placement measurement verbatim — against this
project's own recorded rule: *"Smoke test batch size includes EVAL batch
size — eval can OOM even if training fits."*

**Fix:** pin the draw explicitly (one call at `eval_batch_size=256`, or
four calls at distinct, enumerated seed offsets), quote the exact
binomial tail, and re-smoke eval memory before launch.

### M2 — every cell is priced from a single point estimate drawn from the faster of two clearly separated regimes in the archive; the P0 fallback breaches the ≤2.0 GPU-h hard cap by 2.3× in the slow regime

All ten archived cells of this runner, same 97.6M backbone, same
`batch_size=32`:

| cell | steps | GPU-h | s/step |
|---|---|---|---|
| `wave1_calib_K24_s0` | 19,026 | 4.867 | **0.921** |
| `g3b9_tf_diag` (TF) | 8,000 | 1.865 | **0.839** |
| `sanity_g3b12_tf_s0` (TF) | 3,000 | 0.695 | **0.834** |
| `mob_g3b17_s0` | 19,677 | 5.002 | **0.915** |
| `mob_g3b20_s0` | 20,000 | 4.997 | **0.900** |
| `mob_g3b14_s0` | 20,000 | 1.101 | 0.198 |
| `mob_g3b31_primary/compA/compB` | 20,000 | 0.84/0.81/**0.829** | 0.151/0.146/**0.149** |

A **6.3× spread**, with five cells — including **both** teacher-force
cells, the closest analogues to P1a/P1b — in the slow regime. §R1.1.1
prices everything off `0.8293` (0.149 s/step) and discloses no spread.
In the slow regime: **P1b (5,000 steps) = 1.16 GPU-h**, **P0 fresh
retrain (20,000 steps) = 4.63 GPU-h** — the P0 fallback path alone is
**2.3× the ≤2.0 GPU-h hard cap**.

The failure is not a silent overrun (the runner enforces
`--ceiling-gpuh`, and `wave1_calib_K24_s0` is on disk as
`ABORTED-BUDGET`): it is worse. A P0 retrain launched under a 2.0 GPU-h
ceiling in the slow regime **aborts at ~8,600 of 20,000 steps**,
yielding an under-trained model that is *not* compB's recipe — and P0's
entire purpose is to be compB's recipe.

### M3 — `z_ideal` does not exist in the LM graft; three cells and the mechanism's own subspace depend on it

`spectral_diagnostics()` indexes `z_dump["z_ideal"]`
(`ncr_ortho_write.py:203`). The pinned runner contains **no** `z_ideal`,
**no** Z-dump, and **no** `o`/target dump (`grep z_ideal|z_dump|dump` →
one hit, `json.dump`). Yet:

- §R1.3 defines the mechanism's subspace as `U = az.entity_subspace(z_ideal)`;
- P0 is "Z-dump + spectral diagnostics … via `spectral_diagnostics()` verbatim";
- P2 is "a pure post-hoc re-analysis of stored `o`/target tensors, **zero additional GPU-h**";
- Stage-0 cell 0.4 runs `az.entity_subspace`/`match_eigenvalues` "on `z_ideal`".

None of these can execute as written. The only sensible LM-graft
definition is `z_ideal := integ.teacher_force_operator(keys_v, values_v)`
per batch — which adds a batched `pinv` (an SVD per item per step) to
every training step, absent from the §R1.3 FLOP budget, and which makes
`U` a function of the trainable `entity_adapter` recomputed each step
(the draft says `U` is "computed once from `z_ideal`, never
differentiated through" — "once" is ill-defined when `z_ideal` moves
every step). P2's "zero GPU-h" is contingent on a dump that must first
be built.

### M4 — "entity-adapter init EXHAUSTED" discards a live knob by mislabelling it; the axis is confounded; and the cell the draft calls "at chance" holds the grid's highest h=1 reading

The draft (`:812-817`): the three cells "already vary **entity-adapter
init** across their full tested range — 0992/0993 frozen-at-init, 0994
trainable… **This knob is exhausted.**"

Raw configs:

| cell | `freeze_entity_adapter` | `aux_loss_type` | h=1 `retrieval24` |
|---|---|---|---|
| primary (0992) | **True** | contrastive+cosine | **0.10938 (7/64)** |
| compA (0993) | **True** | **cosine** | 0.01562 |
| compB (0994) | False | contrastive+cosine | 0.03125 |

1. The varied flag is `freeze_entity_adapter` — **train-vs-freeze**, not
   *initialisation*. The initialisation itself (`Linear(768,25,bias=False)`,
   PyTorch default uniform) has **never been varied**: not scale, not
   orthogonal init, not tying to the target space. Given §G3-B25's open
   question is target-space geometry, and given F1 above shows the
   ideal write is flat **iff the entity vectors are orthonormal**, an
   orthogonal/whitened adapter init is a live, cheap, directly
   on-mechanism knob that this sentence discards.
2. The freeze axis is **confounded** with `aux_loss_type` (compA differs
   from primary on both), at **n=1 seed each**.
3. `primary`'s h=1 reading is **0.10938 = 7/64 = 2.71 SD above chance**
   — the highest h=1 value in the grid, from a **frozen-adapter** cell —
   and it **exceeds the design's own registered `τ = 0.0916`**. The
   draft lists it inside the phrase "all three read h=1 retrieval24 at
   chance." At n=64 that value is not 4-SD significant, so it is not a
   positive; but a document that gates a lane on `τ` cannot describe a
   super-`τ` archived value as "at chance" without a stated test.

### M5 — Band-1's t-CI treats nine hops of ONE run as nine iid draws; the diffs trend with depth and there is no seed-level variance in the statistic at all

Recomputed from the raw JSON (`v2` §b): mean paired diff **0.13855**,
SD **0.00657**, upper one-sided 95% CI `0.13855 + 1.8595(0.00657/3) =
0.14262` (draft: 0.14263 ✓), ceiling `0.15+3(0.00657) = 0.16970` ✓,
h=40 diff `0.15088 < 0.16970` ✓. **The arithmetic is exact.** The
inference is not:

- The nine hops are nine evals of **one** model, not nine independent
  samples. The paired diffs **trend with depth**: OLS slope
  **+1.96e-4/hop**, **Pearson r = +0.618**. A `t`-CI on a trending
  within-run series estimates nothing with a sampling interpretation.
- The rule therefore contains **zero seed-level variance** — which is
  exactly the quantity M5's "≈31% per-cell false-void probability"
  complaint was about. Replacing an every-hop rule with a
  within-run-across-hop CI does not control the across-seed false-void
  rate; it re-labels it.
- The pass margin is **0.00738** — 1.1 within-run SD. Combined with the
  upward trend, a cell evaluated on a deeper ladder fails.
- Internal inconsistency: SD is "per-cell re-measured, not assumed fixed
  at compB's 0.00657" for the CI, but **0.00657 is hard-coded** into the
  gross-outlier ceiling.

### M6 — the replacement decay law is inadmissible on the only data that exists, and the unconditioned failing baseline already scores the success value

compB's measured `D_h = 1 − o_pairwise_cos`:

```
h    :   1        2        3        5       12       20       29       40       61
D_h  : 0.20048  0.03164  0.01023  0.00846  0.01091  0.00962  0.01063  0.00977  0.01065
```

Two-phase: a fast transient h=1→3, then a **flat plateau** with ±12%
noise. Geometric fits (`v2` §c):

| fit window | `r = exp(slope)` | `R²(log D)` |
|---|---|---|
| all h=1…61 | 0.97963 | **0.172** |
| h≥3 (post-transient) | **1.00156** | **0.138** |

The design's own inherited gate is "if the fit's `R²` is poor (<0.5)…
escalated to the audit as an open question about whether the model is
the right frame at all." **The baseline's own data fails that gate by
3×, before a single treatment cell runs.**

Worse: for h≥3 the fitted `r = 1.0016 ≥ 1`, and §R1.2 states
"`r=1` ⇒ no decay ⇒ **perfect preservation** of whatever
discriminability exists at `h_ref`." **The unconditioned, retrieval-at-
chance baseline already scores the success value on the replacement
statistic.** This is F2's pathology — a success criterion satisfied by
the failure it is meant to detect — reproduced on the instrument that
replaced it. (`retrieval24` remains PRIMARY, which limits the blast
radius to the calibration/`s1/s2_eff` derivation — but that derivation
is what sets the Stage-1 mechanism target.)

### M7 — `sinθ_h ≈ sinθ_0` is neither sufficient nor necessary; the same degenerate operator maximises the success geometry AND the mechanism target

**Not sufficient** (`v2` §e): `Z = c·I` gives `sinθ_0 = sinθ_61 = 0.9806`,
**ratio = 1.00000 exactly** — a perfect score on §R1.2's stated target
(`sinθ_h/sinθ_0 ≈ 1`) — with **zero** key→value binding. And
`f(c·I) = 0`, a perfect score on §R1.3's target too. **The identity map
simultaneously maximises the success geometry and the mechanism
target.** §R1.3 acknowledges `L_conf(c·I)=0` and argues the true
solution is "**equally** at the minimum, not penalized relative to it" —
F1 shows that is false (true solution: 149–186; `c·I`: 0), so the
degenerate attractor is not merely tied, it **wins**.

**Not necessary** (`v2` §e): `Z = P + (η−1)uuᵀ` with `P` the 24-cycle
and `u` outside the entity subspace:

| η | sinθ ratio @ h=61 | retrieval24 @ h=1/13/37/61 (float32) |
|---|---|---|
| 1.05 | 2.95e-1 | **[1.0, 1.0, 1.0, 1.0]** |
| 1.30 | **6.77e-7** | **[1.0, 1.0, 1.0, 1.0]** |

A six-order-of-magnitude collapse of the criterion with **perfect**
retrieval at every depth. (Note this counterexample is *evidence for*
W3's entity-block restriction — `f(A)=0` there — while killing W2's
criterion. Recorded honestly in both directions.)

### M8 — the ε-guard creates a spurious global minimum at `−λK`, below the flat-spectrum minimum, in the direction that annihilates the task subspace

The guard (`:1142-1147`): "floor `D=tr(G)` at a small `ε` in the
denominator before the divide." Executed (`v1` §4, ε=1e-8):

```
||A||_F = 1e0 … 1e-4 :  f_guarded = 24.4517  (= f_unguarded)
||A||_F = 1e-5       :  f_guarded = -23.9952
||A||_F = 1e-6       :  f_guarded = -24.0000   -> -K
```

The **implemented** loss has a global minimum of `−λK = −24λ` at
`A → 0`, strictly below the flat-spectrum minimum of 0. The design's
proof ("`≥0`, `=0` iff every `sᵢ²` is equal") holds for the *unguarded*
loss only.

Reachability is not remote at this config: `f` is scale-invariant, so
its **radial gradient is exactly zero** — the aux term never opposes
shrinkage — and §G3-B25 records the read as "renormalized, **scale-free**,"
so the CE loss does not pin `‖Z‖` either. Nothing in the objective
opposes a drift in `‖Z_raw‖`; once inside the guard region the aux term
actively **rewards** further collapse of the entity block. **Fix:**
normalise `A` to unit Frobenius norm before the ratio, or gate the term
on `D > threshold` with the guard detached, and register a `‖A‖_F`
tripwire.

### M9 — P1b bundles write-path with training-length, using a probe length this same document retires two sections later

P1b = teacher-force **and** 5,000 steps; P0 = SGD write **and** 20,000
steps. R-A's inference ("closed-form `Z` retrieves; SGD-learned `Z` does
not ⇒ clean localization to write-QUALITY") and R-D's ("degrades once
`entity_adapter`/`embed` train") are both confounded with a 4× training
difference — R-D fatally so, since "P1b failed" and "5,000 steps is not
enough" are indistinguishable.

And §R1.2 (`:1048-1049`) retires exactly this probe length: "calibration
runs at **20,000 steps, the target config** (CLAUDE.md's calibration
rule; **the old 5,000-step cells are retired**)." The document retires
5,000-step probes as inadequate for calibration and then gates the whole
lane on one. Note the fix collides with M2: P1b at 20,000 steps costs
0.83 GPU-h (fast regime) or 4.6 (slow).

### M10 — live budget lines depend on the section this draft declares dead

§R1.7's Stage-1 table budgets "blank-out/localization battery (**§4**,
reused verbatim) — 0.05 GPU-h", and §R1.5 refers back to §2(b)'s
failure-mode 3 for its guard. §4 is inside the block DRAFT-R1 declares
"superseded for every downstream purpose… kept verbatim as the
historical record of **what attack R1 killed**" — and §4 is also where
the **F5-killed placebo** lives. A live deliverable cannot be
incorporated by reference from a dead section; the P=1 blank-out
invariant must be restated in DRAFT-R1's own text.

### M11 — `f(A)` is not a pure conditioning statistic; it also charges for non-invariance of the entity subspace, undisclosed

For a **perfectly orthogonal** `Z`, the entity-block compression already
reads `f(A) = 0.940` with `cond(A) = 4.53`, singular values spanning
`[0.221, 1.000]` (`v1` §7). Subspace compression alone manufactures
spectral spread. So `f(A)` conflates "`Z` is ill-conditioned" with "`Z`
does not preserve `span(U)`" — the second is a real and arguably
desirable target, but the design frames the mechanism purely as
flatness/conditioning and never separates the two. (This also means
`L_conf` and the deployed `ortho_reg` are not identical — the one point
where F5's redundancy argument is bounded.)

### M12 — `f` is NOT a condition-number penalty and is near-blind to rank deficiency; the §R1.2 → §R1.3 bridge and R1.4's wedge name both depend on it being one

R1.4 places `L_conf` in the "**differentiable condition-number**/
restricted-isometry penalty" wedge, and §R1.2 bridges its target to the
mechanism with "flattening all singular values of `A` forces
`‖A‖=‖A⁻¹‖⁻¹`, which bounds every eigenvalue's modulus between the
(now-equal) smallest and largest singular value."

For a rank-`r` block (r ones, K−r zeros) the closed form is
**`f = K(K−r)/r`** — derived and confirmed numerically (`v5_rank_blind.py`,
exact agreement):

| spectrum | `f` | `cond` |
|---|---|---|
| flat (ideal K-cycle) | 0.000 | 1 |
| **rank 23 — ONE entity direction annihilated** | **1.043** | **∞** |
| rank 22 | 2.182 | ∞ |
| rank 12 | 24.00 | ∞ |
| rank 2 | 264.0 | ∞ |
| full rank, `cond = 3` | **34.50** | 3 |
| full rank, `cond = 10` | **357.6** | 10 |

`f` scores an **infinitely ill-conditioned** rank-23 block at **1.04**
and a **well-conditioned** `cond=3` block at **34.5** — it rank-orders
these *backwards* relative to conditioning. `f` and `κ` share only a
zero set; `f` neither equals nor usefully bounds `κ`.

Two consequences. (i) The §R1.2 bridge's own parenthetical —
"`|λᵢ|≥s_min` **requires `A` invertible**, true here since the ideal
write is a permutation" — assumes the invertibility the penalty does not
enforce and is nearly indifferent to. (ii) At K=24 the failure mode
this blindness admits is exactly the task-relevant one: losing a single
entity direction breaks the K-cycle (that entity becomes unrecoverable
and `A^h` degrades progressively), for a penalty cost of 1.04 — versus
357 for a merely spiky but fully invertible operator. **The correct
framing is a scale-invariant spectral-flatness (conformality) penalty,
not a condition-number penalty** — which is also, per F5-B, the one
framing under which a narrow novelty claim survives. Under the standing
gate that reframe is a claim pivot and re-enters the novelty gate.

---

## minor findings

- **m1 — P1a's outcome is analytically determined.** `q_key ≡
  keys_v[a_slot]` bit-identical + a `pinv` fit of 24 constraints in 25
  dimensions ⇒ `o = values_v[a_slot] = T_{tgt}` exactly. Simulated
  residual `3.1e-16`, retrieval24 `1.0000` at h=1/13/37/61. P1a is a
  smoke test, not a diagnostic; register its analytic expectation.
- **m2 — the global `s1/s2` tracker is ill-posed here.** The ideal write
  is rank-K in `d=K+1`, so `s_25 ≈ 0` and the measured global condition
  number is `2.1e17`. Any global conditioning target at this config is
  meaningless; only the entity-block leg is interpretable.
- **m3 — no VOID/saturation band is registered for `D_h`**, despite two
  prior burns (§G3-B26, consolidation F3). `D_h` is *not* saturated at
  compB (compA reaches `3.5e-6`, ≥3 decades of headroom — verified, and
  a point in the instrument's favour), but `primary` and `compA` already
  sit at `3.9e-4` and `3.5e-6`; an arm landing there still yields a
  fitted `r` computed on numerical noise, with nothing to flag it.
- **m4 — the exact binomial tail (2.1e-4) is 6.6× the quoted normal
  approximation (3.2e-5)** at n=256. The union-bound familywise logic is
  valid regardless of dependence ✓, but the quoted per-test number is
  optimistic.
- **m5 — SD convention inconsistency in Band 1**: "re-measured per cell"
  for the CI, hard-coded 0.00657 for the outlier ceiling.
- **m6 — §G3-B29's rule is quoted correctly** (`NCR_REAL_LM_DESIGN.md:6861`,
  "retrieval24 MAX over ALL eval points/splits… ≤ 2×chance"), and M6's
  correction of the false "verbatim" provenance is properly carried ✓.
- **m7 — P1a is priced "~0" GPU-h** but still requires a 98M-backbone
  build plus 4 eval hops at n=256; small, but not zero, and it is inside
  a ≤2.0 GPU-h hard cap.
- **m8 — argmax/position-decomposition exposure is not stated.**
  `retrieval24` is a nearest-neighbour argmax over a 24-entity codebook
  (CLAUDE.md's standing caveat). No rank claim depends on it here, so
  the trap is **not triggered** for W1 — but (i) a positive P1a/P1b at
  h=1 is a single matrix-vector product with an exactly-fitted operator
  and demonstrates **no composition**, and (ii) the P=1 bottleneck check
  that closes position-decomposition is budgeted only via the dead §4
  (M10). Both should be stated in the reporting rules before launch.

---

## §W6-SPOTCHECK — band anchors vs the raw JSONs (all verified)

Source: `experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/mob_g3b31_*_s0.json`.

| # | design's claim | raw value | verdict |
|---|---|---|---|
| 1 | compB `TPC_fg` h=40 = 0.22725 vs bar 0.22637, `+0.00088` **VIOLATION** | `0.22724862` vs `0.07636934+0.15 = 0.22636934` → `+0.00087928` | **VERIFIED — recorded as a fact ✓** |
| 2 | paired-diff mean 0.13855, SD 0.00657 | `0.138554`, `0.006570` (9 hops) | **VERIFIED** |
| 3 | upper CI `0.13855+1.860(0.00657/3) = 0.14263 < 0.15` | `0.142623`; `t_{0.95,df=8} = 1.8595` ✓ | **VERIFIED** |
| 4 | h=40 paired difference `0.22725−0.07637 = 0.15088 < 0.1697` | `0.150879` vs `0.169710` | **VERIFIED** |
| 5 | compB PARTIAL signature `0.09375@h20 → 0.01562@h61` | `retrieval24 = 0.09375` (6/64) @h20, `0.01562` (1/64) @h61 | **VERIFIED — recorded as a fact, "never yet observed" correctly retracted ✓** |
| 6 | h=1 retrieval `0.031/0.109/0.016` vs chance 0.0417, n=64 | `0.03125 / 0.10938 / 0.01562`; granularity 1/64 ⇒ n=64 confirmed | **VERIFIED** (but see M4 on "at chance") |
| 7 | compB `o_pc` `0.7995 → 0.97 → 0.989–0.992`; `D_1 = 0.2005` | `0.79952, 0.96836, 0.98977–0.99154` | **VERIFIED** |
| 8 | `mean_cos` gap 0.31–0.38 at every hop incl. h=1; `offtarget_margin ≈ 0` | gaps `0.3148–0.3814`; margins `\|·\| ≤ 0.0079` | **VERIFIED** |
| 9 | 0.8293 GPU-h/20K-step cell | `gpu_h = 0.8293428717719185`, `elapsed_s = 2985.634` | **VERIFIED** (generalisation attacked in M2) |
| 10 | `same_op_check.verified = true` in all three JSONs | `true` in all three | **VERIFIED** |

**No transcription error found in DRAFT-R1's Band-1/Band-3 anchors.**
W6 is discharged on accuracy; it is the *inference* built on them (M5)
and the *omission* of compA's `retrieval24 = 0.10938 @ h=61` — a
baseline value above the design's own PARTIAL floor of 0.07917 at the
exact scoring depth, voided by Band 1 first but never mentioned — that
remain open.

---

## §LAUNCH-RULING — may the W1 premise battery (≤2 GPU-h, W7-pre-authorized) launch ahead of a full CLEAR?

**NO, not as specified. YES after four narrow repairs, none of which
requires GPU time or a new gauntlet round.**

Nothing found makes the *idea* of the battery unsound — it remains the
highest-value cheap cell in the queue, and F2 makes it **more**
interesting, not less (a banked `answer_accuracy = 1.0` config that has
never been scored with the current primary instrument is exactly the
sort of instrument cross-check this program has twice been burned for
skipping). What is unsound is every part of the apparatus that turns its
numbers into a decision:

**R-1 (from F2) — re-found and re-band the two teacher-force cells.**
Cite §G3-B13 and `sanity_g3b12_tf_s0.json`. Strike "never re-run
post-fix." Re-register P1a/P1b against the archived/analytic reference
of **1.0**, not chance: a PASS band near 1.0, an explicit
INSTRUMENT-DISAGREEMENT band, and `answer_accuracy` co-scored at every
sub-test so `retrieval24` is cross-checked against §G3-B13's own
headline metric on the same configuration.

**R-2 (from F4) — rewrite the tree as an exhaustive partition** over the
outcome space, with an explicit depth quantifier for `CLEARS(·)`, a
branch keyed on `CLEARS(P0)` alone (the premise-refuting outcome), and a
branch for instrument disagreement. Verify by enumeration that each of
the 8 (or 16, with P2) cells maps to exactly one disposition.

**R-3 (from M1) — pin the `n=256` draw and re-smoke eval memory.** State
the exact mechanism (one call at `eval_batch_size=256`, or four calls at
enumerated distinct seed offsets); a silent 4× duplicate of one
deterministic batch turns `τ` into a 2.0-SD bar with a ~5% per-test
false-positive rate. Quote the exact binomial tail. Re-measure eval VRAM
rather than reusing the 6.86 GB figure.

**R-4 (from M2) — remove the P0 fresh-retrain fallback from the ≤2 GPU-h
authorization.** Make P0 conditional on the retained checkpoint. If the
checkpoint is gone, P0 returns to the coordinator for separate
authorization priced at the archive's **slow** regime (0.83–0.92 s/step
⇒ 4.6–5.1 GPU-h), because a retrain launched under a 2.0 GPU-h ceiling
in that regime aborts at ~43% of target and yields a baseline that is
not compB's recipe.

With R-1…R-4 written into the draft, **the premise battery may launch at
≤2 GPU-h ahead of full CLEAR.** Without them it may not: as written it
can (i) authorize a ≤25 GPU-h wave off a reading 8× below its own
archived reference, (ii) be adjudicated post-hoc into either AUTHORIZE
or KILL on three of eight outcomes, (iii) run at `n_eff = 64` while
reporting `n = 256`, and (iv) breach its own hard cap.

---

## BINDING DISPOSITION PROPOSAL (for the coordinator)

Ordered. **X1–X5 are blocking on the wave; X6 gates the launch.**

**X1 (blocking, F1) — `L_conf` as specified is DEAD; the entity-block
flatness family is dead at this config unless re-founded.**
The mechanism's sole selection criterion ("zero at the ideal write") is
false here by a factor of ~150. Three admissible exits, in preference
order:
 (a) **Re-target the key geometry, not the operator.** `f(A*)=0` ⟺ the
 adapted entity vectors are orthonormal. A penalty on the key Gram
 (`‖KKᵀ − cI‖²_F`) is the mathematically correct object — but it is a
 target-space intervention inside the §G3-B17–B32 EXHAUSTED lane and
 must re-enter the novelty gate and confront that lane's null record
 explicitly. Do not adopt it silently as "the same mechanism."
 (b) **Re-found the flatness target on a measured `f(A*)`** rather than
 an assumed 0: penalize `‖f(A) − f(A*)‖` or the *departure from the
 ideal write's own spectrum*, with `A*` computed per batch from the
 teacher-force operator. This is a different loss with different
 optima and needs its own proof-at-optimum and its own gradient check.
 (c) **Abandon the write-conditioning frame** and record F1 as the
 finding: at this configuration the exact write is *necessarily*
 ill-conditioned, because K=24 non-orthonormal keys in d=25 force it,
 so "condition the write" is not a well-posed lever — the lever is the
 key geometry.

**X2 (blocking, F2) — re-found W1 on §G3-B13.** Per R-1. Also propagate
the correction: the falsified claim has reached `STATE.md:5-6` and
`EXPERIMENT_LOG.md:9809-9812` (where it is labelled "KEY ARCHIVE
DISCOVERY"). Both need correcting when this round is recorded.

**X3 (blocking, F3) — the null is re-specified again, with two arms.**
(i) Replay the **treatment** arm's own logged per-step `‖∇L_conf‖`
schedule into the null, paired by seed and step; (ii) **add** a coherent
matched-norm control toward a fixed random non-conformal, non-orthogonal
target (D4's original, wrongly rejected). Pre-register the reading with
the flattening contamination measured, not assumed absent: log `f(A)` in
the null arm every eval and require it **not** to decline relative to
the untreated baseline, or the control is void.

**X4 (blocking, F5) — confront `ortho_reg`, and re-enter the novelty
gate.** (a) State the deployed `0.1·‖ZᵀZ−I‖²/d²` in §R1.3 up front;
decide and justify whether it stays on in the treatment arms (and if it
does, add an `ortho_reg`-off control, because otherwise the arms are
confounded). (b) Re-run the **internal** sweep with
`ortho_regularization_loss` in scope — an internal sweep that does not
grep the pinned runner's own loss functions is not an internal sweep.
(c) **Re-enter the external gate under the corrected framing**: strike
"condition-number" (M12), strike "no external occupant," and re-register
the narrow surviving claim — *conformal (trace-normalised) anchor +
applied as a differentiable loss to an in-context-written operator*,
which the sweep found genuinely unoccupied — with the F5-B
cite-and-distinguish set discharged in §R1.3 text, and the two flagged
citation-integrity items (MuonSSM's "ICML 2026 Oral"; 2102.00533's α)
checked before either is asserted. If the honest reframe is "a
scale-invariant re-parameterisation and strength sweep of a deployed
regulariser," that is a smaller, cheaper, still-publishable experiment —
design it as that.

**X5 (blocking, F4 + M1 + M2 + M3 + M5 + M6 + M8 + M9 + M10) —
the instrument and budget repairs.** Partition the tree (R-2); pin `n`
(R-3); price from the measured *range*, not a point (R-4); define
`z_ideal` for the LM graft and add the Z/o dumps the cells need (M3);
replace the within-run t-CI with a rule carrying seed-level variance
(M5); retire or re-specify the `D_h` geometric law given `R² = 0.14–0.17`
and `r ≥ 1` on the baseline (M6); fix the ε-guard (M8); step-match P1b
to P0 (M9); restate the §4 blank-out invariant in DRAFT-R1's own text
(M10).

**X6 (gating) — premise-battery launch is authorized ONLY after R-1…R-4
are written into the draft** (documentary; no new audit round required
for those four alone). X1's mechanism re-founding is **not** a
precondition for the battery — the battery is upstream of the mechanism
and its result legitimately re-scopes X1.

**Ceremony.** Remain at full multi-round gauntlet (W7). Rev-2 must be
attacked again: two consecutive rounds have now killed the primary
mechanism on the *same* failure mode (a penalty whose optimum is not the
task solution), and this round found that failure mode a second time in
a construction whose proof looked airtight. A third mechanism proposal
must arrive with (i) an executed `f`-vs-retrieval sweep on the *measured*
key geometry, not an analytic claim, and (ii) an internal-archive sweep
that greps the pinned runner's own loss functions before claiming a
wedge is unoccupied.

---

## Reproduction

Scripts (CPU, numpy, no GPU, no training):
`/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad/`
 — `v1_mechanism.py` (V1–V4, F5-A(c), M8, M11)
 — `v2_stats_and_null.py` (M1, M5, M6, F3, M7, F4)
 — `v3_ideal_write.py` (F1 core, m1, m2)
 — `v4_tpc_anchored.py` (F1 anchored to compB's measured `TPC_fg`)
 — `v5_rank_blind.py` (M12)

Raw artifacts read directly (not via agent prose):
`mob_g3b31_{primary,compA,compB}_s0.json`,
`g3b12_smoke_results/sanity_g3b12_tf_s0.json`, `g3b9_tf_diag.json`,
`ncr_lm_wave1_runner.py` (md5 re-verified), plus all ten runner cells'
`gpu_h`/`step` fields for M2.

*Attack R2 complete. No repo file other than this report was written; no
STATE.md/EXPERIMENT_LOG.md edit, no commit, no box contact.*
