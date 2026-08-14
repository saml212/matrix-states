# NCR WRITE-CONDITIONING IN-LM — PRE-REGISTRATION (DRAFT R0)

**STATUS: DRAFT — NOT CLEARED FOR BUILD.** This is a CLAIM PIVOT
(`matrix-thinking/NCR_REAL_LM_DESIGN.md` §G3-B32, 2026-08-06: *"Any next
lever (write-conditioning in-LM: ortho/conformal-scaffold family per
the c*·I Z-dump finding + the free-write K=24 precedent) is a NEW
design = CLAIM-PIVOT territory → full novelty-gate re-entry before any
build (07-16 doctrine). No NCR job is queue-eligible until that design
round runs."*). Nothing in this document authorizes GPU spend. Required
before any launch: (1) triple novelty sweep (§5), (2) an independent
adversarial audit round, (3) the standard build ceremony (10–50 GPU-h
tier: audit + pre-launch resource/placement red-team, per CLAUDE.md
ceremony tiers). Written blind to any write-conditioning code — none
exists yet.

**Reused, unmodified inputs (grounding every number below):**
`NCR_REAL_LM_DESIGN.md` §G3-B24–B32 (the GATE-3 read-collapse diagnostic
arc); `NCR_ORTHO_WRITE.md` §9–§10 (toy K=24/K=32 hard-orthogonal-write
FAIL + mechanism audit); `NCR_ORTHO_FALLBACK_DESIGN.md` §3.1–§3.4, §B8
(the expm/Cayley/damped-polar fallback — Stage 1 never launched, Stage 0
unresolved); `NCR_KWALL_CHARACTERIZATION_DESIGN.md` §2 (the K=24/d=25
free-write config family); `NOVEL_ARCH_WATERFALL.md` §3 (the h\*=61/57
narrowed claim); `EXPERIMENT_LOG.md` 2026-07-09 Z-dump complement entry
(the c\*·I conformal-scaffold finding, commit 64c59d9);
`research/ortho_write_grounding.md` (2026-07-16 VERIFIED literature
sweep — MuonSSM, the orthogonal-RNN line, Muon/Kimi K2); `research/
consolidation-policy-novelty-2026-08-11.md` (ridge-baseline / reader-
nonlinearity discipline; confirms this lever is registered separately
from the consolidation-policy program).

---

## §0 One-paragraph mandate

`NCR_REAL_LM_DESIGN.md` §G3-B32 (harvest of 0992/0993/0994, the
retained-cosine-contrastive grid) closed the read-side road: even in
the arm with healthy target-space integrity (compB, 0994: TPC 0.196–
0.228, far below both the paired-drift bar and the 0.50 absolute
tripwire), the read output `o` still collapsed directionally with depth
(`o_pairwise_cos` 0.80 at h=1 → 0.989–0.992 at h≥3), and 24-way
retrieval stayed at chance throughout. The mechanism, diagnosed in
§G3-B26 and reconfirmed in §G3-B32: `binexp_read`'s repeated squaring
of the trained, in-context-written operator `Z` is power iteration
toward `Z`'s own dominant singular direction — measured `Z` s1/s2 ≈ 1.21
is already enough to erase query-discriminability by h=61 (§1 derives
why). This is a WRITE-side conditioning problem, not a read-side
target-space problem. This document designs the write-conditioning
fix.

---

## §1 The pivoted claim

> **A regularizer or reparametrization applied to NCR's write operator
> `Z` at write time — never touching the read path (`binexp_read`),
> the decode head, or the aux/target loss — holds `Z`'s dominant
> singular-value gap ratio close enough to 1 that `binexp_read`'s
> O(log h) repeated-squaring reads remain query-discriminative
> (24-way retrieval decisively above chance, §3 bands) at the
> pre-registered depth h\*=61, a capability the measured unconditioned
> write provably lacks (§G3-B26/B32: s1/s2=1.21 ⇒ retrieval24 at
> chance at every tested depth, in every one of three independently
> trained arms, 0992/0993/0994).**

### §1.1 Deriving "flat enough" — the exact endpoint (no approximation)

`binexp_read` computes `o = Z^h q` via repeated squaring — mathematically
identical to the naive power, just computed in `O(log h)` matmuls
(`NCR_REAL_LM_DESIGN.md` line 191, 279). Standard power-iteration theory
(e.g. Golub & Van Loan, *Matrix Computations*): for `Z` diagonalizable
with a unique dominant eigenpair `(λ_1, v_1)` and second eigenvalue
`λ_2`, and a generic query decomposed as `q = Σ c_i v_i`,

```
Z^h q = λ_1^h [ c_1 v_1 + Σ_{i≥2} c_i (λ_i/λ_1)^h v_i ]
```

so the *direction* of `Z^h q` converges to `v_1` with

```
sin θ_h ≈ C · ρ^h ,   ρ := |λ_2/λ_1| ,  C = f({c_i/c_1})
```

**The exact, assumption-free case.** If `Z = c·Q` with `Q` orthogonal
(`QᵀQ=I`), every singular value of `Z` equals `c` exactly, so
`ρ=s_2/s_1=1` — and this is not an asymptotic property: `(Q^h)ᵀ(Q^h) =
I` for every `h` by induction, so `Z^h` stays a scaled-orthogonal
matrix at **every** depth. `binexp_read` then preserves ALL pairwise
inner products among distinct queries exactly, up to the global scalar
`c^h` — invisible to cosine scoring, which is scale-invariant
(`NCR_ORTHO_WRITE.md` §10.7). **`sin θ_h = 0` for every h, by
construction, not by extrapolation.** This is the target every
mechanism below is measured against.

### §1.2 Calibrating the decay law against the one measured anchor

§G3-B26 (`mob_g3b24_s0`, the pre-contrastive checkpoint) is the only
cell with both a reported spectral ratio and a reported alignment
angle: `Z s1/s2 = 1.21`, `|cos(o_h61, Z's top singular direction)| =
0.9961`. Converting: `sin θ_61 = √(1−0.9961²) ≈ 0.088`. With `ρ =
s_2/s_1 = 1/1.21 ≈ 0.826`, `ρ^61 ≈ 8.9×10⁻⁶`, so the fitted constant is

```
C ≈ 0.088 / 8.9×10⁻⁶ ≈ 9,900   (≈ 10^4)
```

**Disclosed honestly: this is roughly four orders of magnitude larger
than the O(1) value the two-mode toy model predicts for a generic
query.** Two readings, both consistent with the rest of the record and
neither resolvable from n=1: (a) the trained `Z` has many singular
values clustered near `s_1` (not a clean top-1-vs-rest gap), so the
effective decay is driven by a much larger fraction of the spectrum
than "mode 2" alone; (b) `o_pairwise_cos` was already 0.999 by h=1 in
the original collapse checkpoint (§G3-B26), meaning most of the
"collapse" had already happened before h=61 — the fitted `C` is
absorbing an initial-condition effect the simple model doesn't
separate out. **This constant is not trustworthy as a universal
predictor and is not used as one below** — §3's calibration cell exists
specifically to re-fit it from ~40 controlled points before any Stage-1
cell is scored against a numeric bound.

### §1.3 A provisional numeric bound (order-of-magnitude, superseded by §3's calibration)

Using `d_ncr=25` ⇒ expected cosine SD between two independent random
unit vectors `≈ 1/√25 = 0.20` (already derived and cited in
`NCR_REAL_LM_DESIGN.md`, the Q2 mean_cos analysis) as the discriminability
floor — if `sin θ_{h*}` falls below this, `o`'s query-dependent
component is smaller than the ambient geometric noise floor, and 24-way
retrieval cannot exceed chance by construction. Solving `ρ_required` from
§1.2's fit at `ε=0.20`, `h*=61`:

```
ρ_required = (0.20 / 9,900)^(1/61) ≈ 0.838  ⇒  s1/s2 ≤ 1/0.838 ≈ 1.19
```

**This says the *needed* spectral improvement is small in ratio terms
(1.21 → ~1.19) but the exponential amplification at h=61 makes the
result extremely sensitive to exactly this constant, which §1.2 just
showed is poorly determined by n=1.** Treat 1.19 as a sanity check, not
a pass bar: it is barely tighter than the already-measured collapsed
value, which is not credible as an actionable target given the honest
uncertainty in `C`. The credible target is informed by an *existence
proof* already in the archive: the toy free-write K=24/d=25 arm
(identical `K`, `d` to the real-LM config, §2) reaches `cond(A)` (a
proxy upper bound on `s1/s2`, since `cond=s1/s_min≥s1/s2`) of **1.0–1.1**
across 4/4 seeds via *unconstrained* SGD (`NCR_ORTHO_WRITE.md` §9.1,
`free_K24_s0..s3`), with perfect recovery at every depth tested (h up to
61). **Provisional pre-registered target for the Stage-0 gate (§3):
`s1/s2 ≤ 1.10`** — comfortably inside the derived 1.19 bound, matched to
what an unconstrained write already achieves for free at this exact
`K,d` in an isolated setting. **This target is re-derived from the
calibration fit before Stage-1 is scored** (§3.1) — it is not frozen
here.

---

## §2 The mechanism family

Three mechanisms, organized by what they constrain and how hard:

| | constrains | how | reaches c\*·I specifically? |
|---|---|---|---|
| (a) structural orthogonal write | the whole matrix, exactly | hard reparametrization | no — any orthogonal `Q` |
| (b) conformal-scaffold anchoring | the whole matrix, softly | added loss term | yes — pulls toward the measured c\*·I attractor |
| (c) spectral-penalty-at-write | only the ratio `s1/s2` | added loss term | no — flatness only, no target point |

All three act on `Z_raw = Encoder(keys, values)` (the existing
write-adapter output, unchanged forward computation) and are additive
to the pinned `9a93198b…` v2/v3-instrumented runner (`NCR_REAL_LM_DESIGN.md`
§G3-B27/B31) — no existing code path is modified in place, matching the
program's standing additive-only discipline.

### §2(a) Structural orthogonal write — STRETCH, not primary

```
W  = Z_raw − Z_rawᵀ          # skew-symmetric part
Q  = torch.matrix_exp(W)      # exactly orthogonal for every finite W, no reflection
Z_write = Q                   # replaces Z_raw everywhere downstream (binexp_read, etc.)
```

This is `NCR_ORTHO_FALLBACK_DESIGN.md` §3.2's PRIMARY parametrization
(plain `expm`, no reflection — its own §3.1/§A1.1 shows the K-cycle
target's forced `−1` eigenvalue is reachable inside `SO(d)` without a
reflection factor), reused, NOT the NS-polar arm (`NCR_ORTHO_WRITE.md`
§2), which is EXCLUDED by name (see failure modes).

**FLOP cost.** `expm` via scaling-and-squaring Padé-13 ≈ 6 `d×d` matmuls
forward + ~2× for the Fréchet-derivative backward (`NCR_ORTHO_FALLBACK_
DESIGN.md` §3.2's own derivation) ≈ 12 matmul-equivalents ≈ `12·2d³ =
24d³` FLOPs/write. At `d=25`: `24×15,625 ≈ 3.75×10⁵` FLOPs per document
per step — negligible next to a 98M-parameter backbone's `~10⁸–10⁹`
FLOPs/token. Predicted wall-clock delta ≈ 0, **to be confirmed by the
canary (§3), not assumed** (mirrors the fallback design's own "confirm
via canary" discipline, itself grounded in the K-ladder's measured
finding that this regime is kernel-launch/overhead-bound at small `d`,
not compute-bound).

**What could make it fail (pre-attacked).**
1. **Direct toy-scale precedent at the IDENTICAL `K=24,d=25` geometry is
   a documented FAIL.** Hard NS-polar orthogonal projection at this
   exact tight-spare `d=K+1` configuration is Gate-0 dead 4/4 seeds
   ("too rigid to train through," an absorbing ill-conditioning-runaway
   trap — `NCR_ORTHO_WRITE.md` §9.1/§10). `expm` structurally eliminates
   the *specific* NS mechanism (no `(I+W)`-style inversion, no
   near-singular `σ_min` collapse point to trigger a backward explosion)
   — but a DIFFERENT rigidity failure remains live: forcing `Z` onto the
   `d(d−1)/2`-parameter orthogonal manifold (vs. the free `d²` params)
   could still starve the encoder of degrees of freedom needed to
   satisfy CE + target-integrity + composition jointly, now compounded
   by real-LM backbone co-training — an axis the toy Stage-1 design
   (`NCR_ORTHO_FALLBACK_DESIGN.md` §3) **never tested at all**: Stage 1
   (expm/Cayley) has never launched, gated behind a Stage-0 damped-polar
   smoke that itself returned **FAIL, different signature (flat/
   never-engaged loss, 39/39 zero-recovery eval points)**, with one
   pre-authorized retry (`eps_rel=1e-4`) never executed (`NCR_ORTHO_
   FALLBACK_DESIGN.md` §B8, `STATE.md:307`). **There is currently zero
   empirical evidence, toy or real-LM, that any hard orthogonal
   parametrization trains at this scale.** This is why (a) is STRETCH,
   gated behind its own canary (§3), not a co-primary arm.
2. **The target-structure assumption may not transfer.** `NCR_ORTHO_
   FALLBACK_DESIGN.md` §3.1's reachability analysis (why plain `expm`
   suffices, why Cayley needs a `D`-scaling fix) is derived for a
   K-cycle permutation target with a forced isolated `−1` eigenvalue.
   Whether the real-LM `grammar_rd` h-hop retrieval task has that same
   structure is UNVERIFIED — flagged as a required pre-build check
   (§6), not assumed.

### §2(b) Conformal-scaffold anchoring — PRIMARY

```
c_hat = tr(Z_raw) / d                                  # least-squares-optimal scalar:
                                                         # argmin_c ||Z_raw - cI||_F² ⇒ c=tr(Z_raw)/d
                                                         # (derivation: d/dc[||Z||²-2c·tr(Z)+c²d]=0)
L_conformal = λ_b · || Z_raw − c_hat·I_d ||_F² / d²     # normalized, added to the total loss
L_total = L_CE + λ_ortho·L_ortho + aux_total·(0.5 L_ctr + 0.5 L_cos) + L_conformal
```

`Z_raw` itself is UNCHANGED (no reparametrization) — this only reshapes
gradients toward the specific attractor unconstrained SGD already finds
*for free* in the isolated toy setting (`Z ≈ c*·I_d + rank-(K−1) task
correction`, Procrustes residual 0.003–0.018, τ_identity ≥0.9997 —
`EXPERIMENT_LOG.md` 2026-07-09 Z-dump complement finding, commit
64c59d9). The hypothesis: real-LM joint CE+backbone training prevents
the write from reaching that natural attractor on its own (the
isolated toy free-write does reach `cond→1.0`; the in-LM write measures
`s1/s2=1.21`), and an explicit soft nudge restores it. `c_hat` is
recomputed per-example from `Z_raw`'s own trace — matching the Z-dump
finding's own per-example scale-lock observation, not a fixed global
constant.

**FLOP cost.** Trace + Frobenius norm, `O(d²)` — no new matmuls, the
cheapest of the three. Negligible.

**What could make it fail (pre-attacked).**
1. **Wrong target if the task isn't cycle-structured.** `c*·I` encodes
   specifically a K-cycle permutation's identity/ambient split. If the
   real task's entity-relation structure differs, pulling toward `I`
   could actively suppress correct write content. Required pre-build
   check: fit `z_ideal`'s own eigenvalue structure via the existing
   `az.entity_subspace`/`match_eigenvalues` machinery (CPU-only,
   reused, not re-derived) before trusting this target.
2. **Gradient-starvation replay.** A soft penalty competing with CE+aux
   for gradient share is exactly the confound this program already
   diagnosed once and only partly ruled out (§G3-B22→B25's
   aux-starvation hypothesis, weakened but not closed). Contingency:
   the same decode-isolation-probe playbook (§G3-B25/B26) is
   pre-registered as the disambiguator if Stage-1 shows engagement but
   no gain.
3. **Degenerate `c_hat`.** If `Z_raw`→0, `c_hat`→0 and the penalty
   vacuizes/destabilizes. Required guard: floor `|c_hat|` at a small
   epsilon in the denominator-bearing normalization, or clip
   `‖Z_raw‖_F` from below before computing `c_hat` — specified at build
   time, not deferred.

### §2(c) Spectral-penalty-at-write — PRIMARY

```
s1, s2 = top_two_singular_values(Z_raw)     # power iteration, ~12 iters (matches the
                                              # program's existing n_power=12 convention),
                                              # DETACHED for the scale estimate only —
                                              # gradient flows through the Rayleigh
                                              # quotient terms, not the iteration count
L_spectral = λ_c · max(0, s1/s2 − ρ_target)²  / ||Z_raw||_F²   # hinge: zero once inside
                                                                  # the safe zone; normalized
                                                                  # so shrinking Z can't evade it
L_total = L_CE + λ_ortho·L_ortho + aux_total·(0.5 L_ctr + 0.5 L_cos) + L_spectral
```

`ρ_target = 1.10` provisionally (§1.3), re-derived from §3's calibration
fit before Stage-1 scoring.

**Why this evades §G3-B32's exhausted read-side road (stated sharply).**
The exhausted aux road (§G3-B22–B32) operates on `cos(o, target)` — a
comparison between the READ OUTPUT (after h-fold composition) and a
fixed external target embedding. By construction it cannot distinguish
"`Z` itself is ill-conditioned" from "the target/adapter space is
broken" — resolving exactly that ambiguity cost ~8 GPU-h (§G3-B25→B26).
`L_spectral` is a white-box regularizer computed **directly from `Z`'s
own measured singular values**, with no dependence on the read output,
the target embedding, or the adapter at all. It cannot inherit the
specific saturated-instrument pathology (a collapsed target space
reading `recovered_frac@0.9=1.0` for an information-free read) because
it never looks at the target space.

**FLOP cost.** Top-2 power iteration, `O(d²)` per iteration × ~12
iterations × 2 (deflation for the second mode) ≈ `24d² ≈ 1.5×10⁴`
FLOPs/write. Negligible.

**What could make it fail (pre-attacked).**
1. **A degenerate-minimum gaming route.** `Z→0` makes `s1=s2=0`
   (ratio undefined, or 1 by convention) — a trivial way to "satisfy"
   the penalty while destroying the write. **Guarded above** by
   normalizing the penalty by `‖Z_raw‖_F²`, so shrinking `Z` does not
   reduce the loss. This is the SAME class of degenerate-optimum bug
   §G3-B26 found the hard way (a non-contrastive cosine aux collapsing
   the target space to satisfy itself trivially) — applied here
   proactively, at design time, not discovered after a training run.
2. **The estimator is least reliable exactly where success lives.**
   Power iteration's convergence rate depends on the very gap `s2/s1`
   the penalty is trying to close to 1 — near `ρ_target`, the training-
   time estimate is least trustworthy. Mitigation: use the cheap
   iterative estimate ONLY for the training-time gradient signal; score
   all EVAL-time spectral checks via exact `torch.linalg.svd` on the
   full (small, `25×25`) matrix — cheap and exact off the training path.
3. **Penalizing the global ratio doesn't localize to the task-relevant
   (entity) subspace** — flattening could occur in an irrelevant
   direction while leaving the entity-block ill-conditioned. This is
   why the restricted/entity-block diagnostic (§3.2) is scored
   alongside the global ratio, not instead of it.

---

## §3 Experiment design

### §3.1 Config family — the K=24/d=25 recipe base

Extends the pinned v3-instrumented contrastive-grid runner (md5
`9a93198b642242f512ff8489e32b0a53`, `NCR_REAL_LM_DESIGN.md` §G3-B27/
B31) **unmodified except for an additive, flag-gated conditioning term**
— same discipline `NCR_KWALL_CHARACTERIZATION_DESIGN.md` §2 uses for its
`GRID_SHAPES` extension (add new keys, never mutate existing ones). Base
config = compB's exact recipe (0994): `K=24`, `d_ncr=25` (tight-spare,
`d=K+1` — the SAME config the toy free-write precedent, `NCR_ORTHO_
WRITE.md` `free_K24`, reaches `cond≈1.0` at, per `NCR_KWALL_
CHARACTERIZATION_DESIGN.md` §2's evidence-based convention), trainable
`entity_adapter = Linear(768,25,bias=False)`, `aux_total=0.5·(0.5·L_ctr
+ 0.5·L_cos)`, `ortho=0.1`, CE, 20,000 steps, 98M backbone (12-layer
DeltaNet, matching `NCR_REAL_LM_DESIGN.md` line 2569's tiering) —
**PLUS** exactly one new term (`L_conformal`, `L_spectral`, or the
`Z_write` reparametrization) per §2.

### §3.2 Instruments (all three §G3-B32 instruments carried forward)

1. **TPC (target_pairwise_cos)** — target-space integrity, the §G3-B31
   R1 paired within-run rule, reused verbatim (`TPC_fg ≤ TPC_bo + 0.15`
   at every eval point/hop, `TPC_fg < 0.50` absolute).
2. **o_pc (o_pairwise_cos) collapse watch** — pairwise cosine of the
   read output `o` across documents at each depth; the direct behavioral
   signature of directional collapse (§G3-B32: 0.80→0.989–0.992 in the
   unconditioned compB baseline).
3. **Restricted spectrum tracker** — TWO legs, both cheap:
   (i) **global**: exact `s1/s2` of `Z` via `torch.linalg.svd` at eval
   time (the quantity §1's derivation targets directly);
   (ii) **entity-block**: `spectral_diagnostics()` ported VERBATIM from
   `ncr_ortho_write.py:197` (`depart_normality`, `A_cond`, `A_eff_rank`,
   `min_mod_rel` on `A = UᵀZU`) — reused code, not reinvented, per
   house discipline.
4. **retrieval24** (`NCR_REAL_LM_DESIGN.md` §G3-B27's discriminability
   patch, the PRIMARY behavioral signal — chosen because it cannot
   saturate under target-space collapse the way `recovered_frac@0.9`
   did, §G3-B26's founding lesson).

### §3.3 Stage 0 — calibration, run BEFORE any Stage-1 cell (mandatory, CLAUDE.md pre-experiment rule)

| cell | purpose | seeds | steps | GPU-h |
|---|---|---|---|---|
| 0.0 baseline replication | compB's own o-collapse-under-healthy-TPC signature is currently **n=1** (0994 only) — confirm it is not a single-seed artifact before spending Stage 1 "fixing" it | 1 fresh seed (s4) | 20,000 | 0.83 |
| 0.1 decay-law fit, mech (b) | 3 conditioning strengths (`λ_b` low/med/high; `λ_b=0` reuses 0994, no re-run) | 1 | 5,000 | 3×0.21=0.63 |
| 0.2 decay-law fit, mech (c) | 3 strengths (`λ_c` low/med/high) | 1 | 5,000 | 3×0.21=0.63 |
| 0.3 mech (a) canary | Gate-0 only — does `expm` engage or reproduce the NS-polar absorbing-collapse signature (flat loss / loss dips then recollapses)? | 1 | 2,000 | 0.12 (1.4× disclosed multiplier) |
| 0.4 pre-build CPU check | fit `z_ideal`'s eigenvalue/group structure via `az.entity_subspace`/`match_eigenvalues` (reused) — verifies §2(a)/(b)'s cycle-structure assumption before either mechanism launches | — | — | 0 |

**Stage-0 subtotal: 2.21 GPU-h.**

**Stage-0 gate (frozen before any Stage-1 spend, per §G3-B31/NCR_ORTHO_
WRITE's own double-gate convention):**
- 0.0 must reproduce the o-collapse-under-healthy-TPC signature
  (`o_pairwise_cos` at h≥3 within the 0989-family noise band of the
  original 0.989–0.992, retrieval24 ≤ 2×chance) — if it does NOT
  replicate, the whole premise (§0) needs re-diagnosis before Stage 1,
  not a silent proceed.
- 0.1/0.2's 42 (strength, h, `sin θ`) points fit the log-linear decay
  law `log(sin θ) = log C + h·log ρ` (§1.1); the fitted `C` REPLACES
  §1.2's n=1 estimate, and the resulting `ρ_required` REPLACES §1.3's
  provisional 1.10 target for all Stage-1 scoring. If the fit's `R²` is
  poor (<0.5) or the sign of the slope disagrees with the model, that
  is itself a result — recorded, not silently discarded, and escalated
  to the audit as an open question about whether the 2-mode power-
  iteration model is the right frame at all.
- 0.3 must show ENGAGEMENT (a loss trajectory unlike the flat/
  never-engaged Stage-0 damped-polar signature, `NCR_ORTHO_FALLBACK_
  DESIGN.md` §B8) for mechanism (a) to proceed to Stage 1 at all; on
  FAIL with the SAME signature as either toy precedent (flat, or
  dip-then-recollapse), mechanism (a) is DROPPED from Stage 1 and (b)/
  (c) absorb its budget (extra seeds), not launched anyway.

### §3.4 Stage 1 — main grid (gated on §3.3)

| arm | seeds | steps | GPU-h |
|---|---|---|---|
| (b) conformal-scaffold anchoring, PRIMARY | 4 | 20,000 | 3.32 |
| (c) spectral-penalty-at-write, PRIMARY | 4 | 20,000 | 3.32 |
| (a) structural expm write, STRETCH (gated on 0.3) | 2 | 20,000 | 2.32 (1.4× multiplier) |
| shuffled/placebo-conditioning control (§4) | 3 | 20,000 | 2.49 |
| blank-out/localization battery (§4) | — (eval-only, bundled) | — | 0.05 |

**Stage-1 subtotal: 11.5 GPU-h.**

### §3.5 Total budget (K-wall-style derivation)

```
Stage 0            2.21 GPU-h
Stage 1            11.5  GPU-h
-----------------------------
Nominal total       13.7 GPU-h
× 1.4 contingency   19.2 GPU-h   (reruns, contention, split-result
                                   seed escalation — mirrors NCR_ORTHO_
                                   FALLBACK's own n→8 escalation
                                   convention, pre-authorized not
                                   ad hoc)
-----------------------------
Registered ceiling  ≤20 GPU-h nominal, hard cap ≤30 GPU-h (task ceiling)
```

Leaves ~10 GPU-h of explicit headroom under the 30 GPU-h wave-1 cap for
exactly one pre-registered contingency: a split-result seed bump on
whichever PRIMARY arm needs it (mirrors `NCR_ORTHO_FALLBACK_DESIGN.md`
§A1.3's frozen `1≤p≤3/4`-passing-seeds → `n→8` trigger). This sits in
the **10–50 GPU-h ceremony tier**: one audit round PLUS a pre-launch
resource/placement red-team (not the full multi-round gauntlet reserved
for >50 GPU-h or publication-bound spends) — flagged for the
coordinator's call given this is flagship-adjacent (§6 item 5).

**Placement.** Same runner class as 0992–0994, measured at 6.86 GB
VRAM / 73–80% SM util per cell (`NCR_REAL_LM_DESIGN.md` §G3-B31
PLACEMENT) — reusing that measurement (not re-deriving): one cell per
GPU on 4/6/7, no packing (packing would 3× critical-path wall time for
zero utilization gain at this VRAM/SM footprint, per the identical
prior ruling).

### §3.6 Success / kill bands, scored at h\*=61

Checked in order (mirrors §G3-B31/B32's Band-1-first convention):

1. **TPC integrity** (§3.2.1, verbatim §G3-B31 rule). Violation ⇒
   NULL-BY-COLLAPSE regardless of retrieval24.
2. **Mechanism check** — `s1/s2(Z) ≤ ρ_required` (from §3.3's fit, not
   §1.3's provisional 1.10) at h\*, both global and entity-block legs.
   FAIL here ⇒ INCONCLUSIVE-BY-MECHANISM (the intervention didn't hit
   its own target — informative for iterating, distinct from a clean
   behavioral NULL).
3. **retrieval24 @ h\*=61** (PRIMARY, matches the discriminability-proof
   convention `NCR_REAL_LM_DESIGN.md` §G3-B27 established): chance =
   1/24 ≈ 0.0417.
   - **NULL/COLLAPSE:** ≤ 2×chance = 0.0833 (verbatim §G3-B29 frozen
     rule — matches the measured unconditioned baseline exactly).
   - **PARTIAL:** (0.0833, 0.192].
   - **WIN:** > chance+0.15 = 0.192 (reuses the exact +0.15
     absolute-margin convention §G3-B31 R1 already established for TPC,
     applied here for internal consistency) **AND** the GAP metric
     (full_graft − backbone_only, §G3-B27's PRIMARY signal) also
     exceeds 0.15 at h\* — so a WIN cannot be produced by a
     backbone_only-side fluke.
4. **Depth-decay PARTIAL signature** (carried forward, never yet
   observed in any cell to date — §G3-B32: "no WIN, no depth-decay
   PARTIAL signature anywhere"): clears WIN/PARTIAL at shallow-mid
   depth (h≤20) but decays toward chance by h\*=61 — registered as its
   own labeled outcome, not silently folded into NULL.

**Mechanism-level FAIL (Gate-0, reused verbatim from `NCR_ORTHO_WRITE.md`
§9):** if the conditioning term prevents in-dist (`h∈{1,2,3}`)
`recovered_frac@0.9 ≥0.9` in ≥3/4 seeds — "too rigid to train through,"
the toy program's own registered failure mode, pre-registered as a LIVE
risk for mechanism (a) specifically given its direct precedent.

**If ALL Stage-1 arms FAIL/NULL:** the write-conditioning lever resolves
definitively to NULL/FAIL — the read-collapse mechanism would then be
established as robust to both read-side (§G3-B22–B32) and write-side
(this document) intervention at this architecture/scale, a materially
different and more serious finding requiring its own honest write-up,
not a quiet retry.

---

## §4 Baselines and controls

**Unconditioned baseline — the §G3-B32 recorded numbers ARE the
baseline, no re-run.** compB (0994): TPC 0.196–0.228 (healthy — isolates
the Z-conditioning mechanism from the target-collapse confound that
contaminates 0992/0993), `o_pairwise_cos` 0.80(h=1)→0.97(h=2)→0.989–
0.992(h=3..61), retrieval24 at chance throughout,
`retrieval24_acc_gap@h61 = −0.03125`. This is the null hypothesis every
Stage-1 arm must beat. (0992/0993 available as secondary reference but
both are already NULL-BY-COLLAPSE on TPC, so less clean as a
comparator — footnoted, not primary.)

**Shuffled/placebo-conditioning control.** Applies mechanism (b)'s exact
computational structure — an anchor penalty of identical form,
magnitude, and update frequency — but toward a RANDOM, per-example,
DETACHED target `R_rand` (a fresh random orthogonal matrix, not
`c_hat·I`) instead of the data-derived conformal target. Same gradient
perturbation/regularization pressure, zero information about
conformal/orthogonal structure relevant to composition. If Stage-1's
gains are real (structural), the placebo should reproduce compB's NULL;
if the placebo ALSO improves retrieval, the conclusion flips to "extra
regularization/gradient-budget reallocation helps, not specifically
flatness" — a control made necessary by this program's own documented
history of exactly this confound (§G3-B22–B25's aux-weight/gradient-share
ambiguity).

**Blank-out/localization battery.** Reuses the existing P=1 bottleneck
check verbatim (`NCR_ORTHO_WRITE.md` §9.0: raw-input corruption
post-encoding ⇒ bit-identical read, exactly-zero gradient w.r.t. raw
inputs — passed 24/24 in the toy program) as a sanity invariant: the
conditioning term only touches the encoder's `Z` output, so this
invariant must hold UNCHANGED under every mechanism — a cheap,
eval-time confirmation that no mechanism accidentally opens a side
channel around the state bottleneck. Bundled into existing eval, no
dedicated GPU-h.

**Ridge-baseline / index-coded-cache steelman discipline** (per
`research/consolidation-policy-novelty-2026-08-11.md`'s revival-condition
standard, applied "where applicable" per the task charter). Assessed and
scoped down, not skipped: the relevant steelman for a
repeated-squaring-composition claim is whether the SAME retrieval could
be solved without composition at all — this is already what the
`backbone_only` read-ablated arm (present in every 0992–0994 JSON, §G3-B27)
tests, and it is retained as the standing non-compositional control
(§3.6 item 3's GAP metric IS this comparison). A separate trained,
byte-matched direct-lookup baseline would test general retrievability,
not composability — the axis this design is not making a claim about
— so it is registered as **out of scope for wave-1**, with the reasoning
disclosed here rather than the instruction silently dropped.

---

## §5 Novelty-gate charter

Three sweeps, per standing 2026-07-16 doctrine (novelty checked at
design time AND re-checked before launch — this design is 4+ weeks
downstream of the most recent relevant sweep, `research/ortho_write_
grounding.md`, dated 2026-07-16, itself noting the field moves in
~2-week increments in this exact subarea).

**By-task sweep (external).** Search terms: "write-time spectral
conditioning fast-weight matrix read depth," "state conditioning for
long-horizon matrix-power composition," "orthogonality regularization
associative memory retrieval depth," "in-context operator conditioning
compositional read." Target: any paper testing whether conditioning a
WRITTEN (not recurrent-transition) matrix improves READ-time
composability at depth, on any task.

**By-mechanism sweep (external), starting from the ALREADY-VERIFIED
`research/ortho_write_grounding.md` inventory — refresh, don't redo:**
- **MuonSSM (Nguyen et al., arXiv:2606.30461, ICML 2026 Oral)** — the
  memo's own "most dangerous prior art": Newton–Schulz-orthogonalizes
  fast-weight WRITES (not the transition matrix), explicitly the same
  category distinction this design draws. Already-established
  differentiators to re-verify still hold: (1) MuonSSM's write is
  RANK-1 (a KV outer product; "orthogonalizing" it is single-value
  magnitude conditioning, not a genuine `Q∈O(d)`), this design's writes
  are full `d×d`; (2) different diagnostic — MuonSSM targets general
  SSM gradient/memory stability (condition number on the *accumulated
  recurrent state*), never discusses compositional-depth reads or
  matrix-power composition; (3) different regime — MuonSSM uses 1
  quintic NS iteration by design (5 iterations *underperforms* 1 in
  their own ablation), this design's (a) uses exact `expm`, and (b)/(c)
  are SOFT regularizers with no NS iteration at all — a mechanism class
  MuonSSM does not cover.
- **Muon / Kimi K2** (blog + arXiv:2507.20534) — optimizer-update
  orthogonalization, category (b) not (writes); cite-and-distinguish
  only.
- **Orthogonal/unitary RNN line** (Arjovsky arXiv:1511.06464, Wisdom
  arXiv:1611.00035, Helfrich/scoRNN arXiv:1707.09520, Mhammedi
  arXiv:1612.00188, Lezcano-Casado arXiv:1901.08428) — constrains the
  RECURRENT TRANSITION matrix, not in-context-written content; cited as
  the principle's lineage, not the mechanism's occupant.
- **DeltaNet/Gated DeltaNet/RWKV-7/DeltaProduct/Deep Delta Learning** —
  all category-(a) transition-matrix constraints; distinguish per the
  memo's existing table.
- **NEW for (b)/(c) specifically (not covered by the memo, which only
  checked hard-orthogonal writes):** search "soft spectral radius
  penalty recurrent state training," "spectral normalization fast
  weight matrix," "conformal/scaled-identity regularization associative
  memory," "singular value ratio penalty training loss." Miyato et al.
  spectral normalization (ICLR 2018, GAN discriminators) is the nearest
  known analog for a soft `s1`-constraining penalty and must be checked
  and distinguished (different target — normalizing Lipschitz constant
  of a weight matrix vs. flattening a fast-weight STATE's spectrum for
  read-depth robustness) if found to be close.

**Internal-archive sweep.** `NCR_ORTHO_WRITE.md` (hard-ortho FAIL at
this exact K/d — cite as the reason (a) is demoted), `NCR_ORTHO_
FALLBACK_DESIGN.md` (expm/Cayley never launched, damped-polar
unresolved — cite as why (a) needs its own canary), `NCR_KLADDER_
DESIGN.md`/`NCR_KLADDER_ATTACK_R2.md` (SPENT, K-ladder scale-up under
the ortho mechanism — different axis, K not conditioning, no overlap),
`research/consolidation-policy-novelty-2026-08-11.md` (explicitly
confirms this lever is registered separately from the S_mem
consolidation-policy program — "NOT the NCR claim pivot... different
lever, competes only for GPU-hours/coordinator bandwidth" — no overlap
to re-litigate), `KILL_LIST.md` (checked this round: no prior write-
conditioning or spectral-penalty entry found).

**What would count as scooped.** A paper that (i) conditions an
in-context-WRITTEN (not recurrent-transition, not optimizer-update)
matrix, (ii) motivates and diagnoses the fix via compositional-READ
failure under repeated matrix powers specifically (not general
long-horizon SSM stability), and (iii) either uses a soft
spectral-ratio/conformal-anchor penalty (mechanisms b/c — currently
unchecked territory per the memo) or a full-rank exact structural
parametrization at this task class (mechanism a — MuonSSM's rank-1 case
is close but distinguished). Any hit on all three collapses the claim
to a positioning-only contribution, cite-and-distinguish, not a kill —
unless it also shares the K=24/d=25/98M-backbone real-LM grafted-read
setting, which would be a genuine scoop requiring re-scoping.

---

## §6 Self-attack — what would kill this design at audit

1. **h\*=61 is borrowed, not derived for K=24.** It is the deepest
   already-instrumented, non-trivial-residue point in the EXISTING
   real-LM ladder (`h mod 24 = 13`, comfortably clear of the
   in-dist/identity residues {0,1,2,3} — checked, not assumed) — chosen
   for direct comparability to every already-measured checkpoint
   (0992–0994), not because it is K=24's own natural `h_star=8K−3=189`
   (`ncr_task.GRIDS[24]`). This is a deliberate continuity choice,
   disclosed here rather than left for an auditor to catch; it happens
   to numerically coincide with `NOVEL_ARCH_WATERFALL.md` §3's K=8
   h\*=61, which is coincidence of value, not shared derivation — stated
   explicitly to avoid implying more grounding than exists.
2. **§1's numeric bound (1.19, tightened to a provisional 1.10 target)
   rests on a two-mode approximation with a fitted constant four orders
   of magnitude off its own model's O(1) expectation.** This is the
   single biggest fragility in the document. Mitigated, not hidden: the
   whole point of §3.3's calibration cells is to replace this n=1
   estimate with a ~40-point fit BEFORE any Stage-1 cell is scored
   against a hard number. An auditor could reasonably demand the
   calibration cells run and the fit be inspected before Stage-1 is
   authorized at all (a stricter reading than this draft's "Stage 0
   gates Stage 1" sequencing) — flagged as an open sequencing question
   for the audit round, not resolved unilaterally here.
3. **compB's own headline finding (o collapses even with healthy TPC)
   is currently n=1.** §3.3 cell 0.0 exists specifically to close this,
   but until it runs, everything downstream is conditioned on an
   unreplicated single-seed result — an auditor could require 0.0 to
   clear BEFORE authorizing 0.1–0.3's spend, not in parallel.
4. **Mechanism (b)'s conformal target may be structurally wrong for the
   task.** If `grammar_rd`'s h-hop retrieval is not cycle/permutation-
   structured the way the toy Task E / opbank harness is, `c*·I` is not
   obviously the right attractor to nudge toward — §3.3 cell 0.4 (CPU,
   free) is registered to check this before either (a) or (b) launches,
   but the check itself could return an inconclusive answer given the
   real-LM's target space is adapter-mediated, not a clean synthetic
   `z_ideal`.
5. **Scale scope.** This design is K=24/98M ONLY. A WIN here does not
   establish the mechanism at K=32/392M (the flagship's own stated
   range, CLAUDE.md "98M–392M"). Explicitly NOT claimed here — a
   scale-up wave is a deferred, separately-gated follow-on, not
   silently assumed to transfer.
6. **Ceremony-tier tension.** Priced at ≤20 GPU-h nominal / ≤30 GPU-h
   ceiling — the 10–50 GPU-h tier (audit + pre-launch red-team) per
   CLAUDE.md's ceremony rule. But this IS the flagship spearhead's
   recorded next lever (CLAUDE.md "THE GOAL — one spearhead"), which
   could argue for the full multi-round gauntlet regardless of GPU-h.
   Flagged for the coordinator's explicit call, not pre-decided here.
7. **The placebo control (§4) only tests mechanism (b)'s structure.**
   If (c)'s spectral penalty wins but the placebo (built on (b)'s
   anchor-loss form) doesn't test an analogous "any extra regularizer"
   null for (c) specifically, a (c)-only WIN would be less rigorously
   controlled than a (b) WIN — an asymmetry an auditor should be asked
   to either accept (single placebo suffices, since both b/c add a
   loss term of similar character) or require a second, (c)-shaped
   placebo (add GPU-h: 3 seeds × 0.83 ≈ 2.5, still fits under the 30
   GPU-h ceiling with the existing headroom).

---

**NEXT:** novelty gate (three sweeps, §5) → independent adversarial
audit round → build ceremony (10–50 GPU-h tier, pending §6 item 6's
resolution) → Stage 0 launch. No GPU spend, no STATE.md/EXPERIMENT_LOG.md
update, no commit from this document — coordinator dispatches next.
