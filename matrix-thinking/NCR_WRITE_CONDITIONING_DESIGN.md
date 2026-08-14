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

---

## §A1-ADJUDICATION (coordinator, 2026-08-13) — attack R1 = BLOCKED (5F/11M/7m) ADOPTED; DRAFT-R0 dead as drafted; novelty gate discharged-with-obligations; Rev-1 dispatched; CEREMONY ESCALATED to full gauntlet

All four same-day verdicts adjudicated together. Novelty:
`research/writecond-novelty-2026-08-13.md` (by-task OPEN; by-mechanism
PARTIALLY-OCCUPIED with the (b)/(c) wedges open — MuonSSM
arXiv:2606.30461 and DeltaProduct arXiv:2502.10297 are mandatory
cite-and-distinguish anchors; internal CLEAN with the §N2 omission).
Attack: `NCR_WRITECOND_ATTACK_R1.md`, ADOPTED IN FULL.

**The five FATALs, accepted as charged:**
- **F1 (frame kill, governs everything):** the §G3-B32 cells read
  24-way retrieval AT CHANCE AT h=1 (0.031/0.109/0.016 vs 0.0417,
  n=64) — one application of Z, zero composition. There is no
  depth-destroyed capability for conditioning to restore until h=1
  retrieval exists. §3.6's all-NULL clause would have published a
  wrong structural conclusion off a wave that could not show
  otherwise.
- F2: §1.1's `sin θ_h = 0` success target is the COLLAPSE signature
  (verified: Z=cQ holds sin θ ≈ 0.99 constant; ρ=1 in §1.1's own
  formula yields C, not 0). The §1 analysis is rewritten, not
  patched.
- F3: mechanism (b)'s penalty is minimized by the identity map — the
  IDEAL K-cycle write scores ≈max penalty (3.99e-2 vs 0 at c·I;
  ‖Z*−ĉI‖²/‖Z*‖² = 0.998) and Band-2 reads 1.00000 down to zero
  capability (saturated instrument); the cited Z-dump entry itself
  records the scaffold as "dispensable… not load-bearing."
- F4: ρ_target=1.10 does not prevent the collapse it exists to
  prevent (o_pc@h61 = 0.877 at r=1.10 AND r=1.21; only r ≤ 1.02
  works).
- F5: the placebo anchors toward a random ORTHOGONAL matrix — the
  active ingredient class — inverting the control's reading.
Selected MAJORs: the fitted C≈9,900 is mathematically inadmissible
(sin θ > 1 for all h ≤ 48 — the MODEL is mis-specified, so
calibration would re-fit garbage, confirming the audit charter's
worry); the λ-vs-s substitution (s1/s2=1.21 with |λ₂/λ₁|=1 and zero
collapse is a live counterexample); a zero-cost rank-2 route through
(c)'s hinge; the ‖Z‖_F² normalization is a runaway escape hatch
(§10.7 measured σ_max 5→13); and **compB VIOLATES Band-1's paired
bar at h=40 (0.22725 vs 0.22637, raw-JSON-verified) and already
exhibits the "never yet observed" depth-decay PARTIAL signature
(0.094@h20 → 0.016@h61)** — the draft's own baseline anchors were
mis-transcribed from the raws. Verified clean: budget arithmetic,
the 9a93198b runner pin, expm/SO(d) citation, placement.

**Dispositions (binding Rev-1 charter):**
- **W1 (F1): PREMISE CELL FIRST.** Stage 0 becomes a cheap
  h=1-retrieval diagnostic on the real-LM graft (does ANY
  configuration retrieve above chance at h=1; if not, WHY — the wave
  is unauthorized until this cell reads positive or the claim is
  re-scoped to whatever it reveals). Design it with its own bands.
- W2 (F2 + MAJORs): §1 rewritten from scratch — correct success
  geometry, admissible decay model, s-vs-λ distinction carried
  honestly; the calibration sweep re-registered against the
  REWRITTEN model.
- **W3 (F3+F4 + attack D3): mechanisms (b) and (c) as drafted are
  DEAD; replaced by the attack's entity-block conformality penalty
  ‖AᵀA − (tr(AᵀA)/K)·I‖² on the written entity block** — zero at the
  ideal K-cycle write, smooth, SVD-free, scale-aware. Novelty: inside
  the swept-open identity/conformal-anchoring wedge (memo records the
  mapping); MuonSSM/DeltaProduct distinctions stated in §2 up front.
- **W4 (three-line convergence): mechanism (a) is CUT from wave-1**
  — external crowding (MuonSSM, DeltaProduct), the PI-ratified §N2
  demotion (idle-filler only, which the draft failed to cite), and
  zero empirical support. It stays exactly where §N2 put it; its
  seeds reallocate to the W3 mechanism.
- W5 (F5): the control is re-specified as a true null (e.g. a
  matched-magnitude penalty toward a random NON-conformal,
  NON-orthogonal target with the same gradient-norm budget), with
  its reading pre-registered un-invertibly.
- W6: every band re-anchored to the RAW artifacts (the compB
  Band-1/PARTIAL findings incorporated as recorded facts), the c*·I
  architecture-conditionality caveat carried forward explicitly, and
  the §N2 ruling cited wherever the ortho track is mentioned.
- **W7 (ceremony): ESCALATED to the full multi-round adversarial
  gauntlet** (publication-bound spearhead; the attack's own
  recommendation) — Rev-1 → attack R2 → iterate to CLEAR before any
  build ceremony; the premise cell (W1) is the only GPU spend
  authorized ahead of a CLEAR, as its own ≤2 GPU-h stage with 1
  audit round.

Rev-1 dispatched 2026-08-13. The novelty memo's re-entry conditions
apply: any headline reframe after the premise cell resolves re-enters
the gate.

---

## DRAFT-R1 (Rev-1, 2026-08-13)

**Binding charter:** `§A1-ADJUDICATION` above, adopted in full. This
section supersedes §1–§6 for every downstream purpose; §1–§6 are kept
verbatim as the historical record of what attack R1 killed (house
convention). Written against repo commit `59d03fb` plus fresh reads of
`matrix-thinking/ncr/ncr_ortho_write.py` (`spectral_diagnostics`,
`az.entity_subspace`/`block_decompose`), `matrix-thinking/chapter2/
analyze_zdump.py` (`entity_subspace`, `block_decompose`,
`effective_rank`), `experiment-runs/2026-07-30_ncr_g3b31_contrastive_
grid/ncr_lm_wave1_runner.py` (the pinned `9a93198b…` runner — read in
full for `ncr_lm_forward_ablatable`, `discriminability_metrics`,
`ortho_regularization_loss`, the teacher-force branch, and the
module-header disclosure that teacher-force is "a FAIL-diagnosis tool
for AFTER a result is read, not part of this calibration's own two
arms" — exactly the use W1 makes of it now that a result has been
read), the raw JSONs `mob_g3b31_{primary,compA,compB}_s0.json`, and
`NCR_REAL_LM_DESIGN.md` §G3-B9/B10 (old teacher-force diagnostic, PRE
the §G3-B12 single-adapter fix) and §G3-B25/B26/B27/B28/B31/B32 (the
fixed architecture, the collapse mechanism, and the frozen metric
definitions).

### R1.1 W1 — the premise cell

**What the archive already answers, so no GPU is spent re-asking it.**
`discriminability_metrics` (`ncr_lm_wave1_runner.py:480-537`) fixes
`retrieval24_acc` as `argmax_k cos(o, entity_adapter(embed(entity_k)))
== true_slot`, cosine-normalized both sides. The three §G3-B31 cells
already vary **entity-adapter init** across their full tested range —
0992/0993 frozen-at-init (§G3-B28's healthy-target-space control),
0994 trainable — and all three read h=1 retrieval24 at chance (0.03125
/ 0.10938 / 0.01562 vs chance 0.04167, n=64; attack R1 F1, re-verified
against the raw JSONs this round). **This knob is exhausted; W1 spends
no new GPU-h on it,** it is carried into the decision tree below as an
already-answered branch.

**The two knobs genuinely untested, and why each is diagnostic:**

1. **Write path — SGD-learned `Z` vs. closed-form teacher-forced `Z`.**
   `ncr_lm_forward_ablatable` (`runner.py:360-398`) already carries a
   `teacher_force` branch (`Z = integ.teacher_force_operator(keys_v,
   values_v)`, a closed-form least-squares operator fit that bypasses
   `ncr_head.encode` entirely) — pre-wired, smoke-verified
   (`NCR_REAL_LM_DESIGN.md:5033-5041`, "the audited closed-form op fit
   that bypasses the encoder... encoder zero-grad, residual 7.3e-6"),
   and explicitly scoped by the runner's own module docstring as **"a
   FAIL-diagnosis tool for AFTER a result is read, not part of this
   calibration's own two arms"** (`runner.py:29-32`) — i.e. exactly
   what W1 is now, on schedule, per that same docstring's own design
   intent. It was last exercised in §G3-B9/B10 (2026-07-18,
   `g3b9_tf_diag.json`) and returned **READ/setup-broken, NOT a
   WRITE-blocker** — but on the OLD architecture, which §G3-B11/§G3-B12
   (`NCR_REAL_LM_DESIGN.md:5256-5297`) then proved was itself defective
   in three ways load-bearing to that exact verdict: (3a) `q_key` and
   the bind-clause key were different vectors (separate `key_adapter`
   contexts), so the teacher-forced `Z` was fit to the wrong key and
   necessarily failed at read time regardless of `Z`'s quality; (3b)
   `value_adapter` received zero gradient under the old teacher-force
   detach, so the read's own output basis was frozen at random init;
   (3c) separate key/value adapter spaces made `h≥2` composition
   undefined by construction. §G3-B12's fix — the single shared
   `entity_adapter`, with `keys_v` and `q_key` BOTH built as
   `entity_adapter(embed(input_ids[pos]))`, a context-free function of
   token id — structurally repairs (3a): for the true answer entity,
   `q_key` and `keys_v[a_slot]` are now bit-identical by construction
   (this is exactly what `assert_read_target_write_key_same_op`,
   `runner.py:569-599`, checks every launch — `same_op_check.verified =
   true` in all three §G3-B31 JSONs, re-confirmed this round). **The
   teacher-force diagnostic has never been re-run on the architecture
   that fixed the bug that sank its last verdict.** §G3-B25's own open
   question — "a STRUCTURAL block in the decode path... candidates: the
   renormalized, scale-free `binexp_read` output may be
   information-complete but magnitude-degenerate... a stop-gradient/
   detach in the read→decode path" — is exactly what this isolates:
   hand the read a Z that is *provably* not an SGD-learning artifact,
   and see whether retrieval24 clears chance at h=1. §G3-B26 supplies
   the motivating alternative (the SGD-learned `Z`'s own s1/s2 is the
   fault); §G3-B32 supplies the trigger (retrieval24 at chance at h=1
   in the CURRENT pipeline, with a discriminability metric that did not
   exist when §G3-B9/B10 ran). Cost: near-zero (§R1.1.2).

2. **Read normalization / retrieval-metric variant — common-mode
   removal.** §G3-B25 flagged, unresolved: *"the renormalized,
   scale-free `binexp_read` output may be information-complete but
   magnitude-degenerate."* Attack R1's own M8 measured the mechanism
   directly: `mean_cos` (full_graft − backbone_only) sits at **0.31–
   0.38 at every hop including h=1** in compB — `o` is aligned to the
   target *cone* by a large, roughly depth-independent amount, and to
   the *correct* target by nothing (`offtarget_margin` ≈ 0 at every
   hop). `discriminability_metrics` computes `retrieval24_acc` from raw
   cosines (`cos_all`, `runner.py:517`) with **no common-mode removal**
   — a batch-wide additive/multiplicative bias that inflates every
   `cos(o, target_k)` roughly equally survives straight into the argmax
   and is invisible to it (an argmax over `k` is shift-covariant only
   if the shift is IDENTICAL across `k`, which a query-independent
   common mode is by definition, so a naive re-read of the SAME cosines
   changes nothing — the metric variant has to actually subtract the
   mode before re-scoring). **Variant:** re-score retrieval24 with the
   batch-mean direction removed from `o` (equivalently, mean-center
   `on = F.normalize(o)` across the eval batch, renormalize, then
   re-run the identical argmax) — a pure post-hoc re-analysis of stored
   `o`/target tensors, zero additional GPU-h (§R1.1.2, cell P2). This
   directly tests M8's own alternative reading: *"either the 1.21 does
   not describe compB's `Z`, or the h=1 collapse has a non-spectral
   cause (a query-independent additive component, which no spectral
   penalty touches)."*

#### R1.1.1 Cell battery (all on the pinned `9a93198b…` runner, additive flags only)

| cell | what | steps | seeds | GPU-h |
|---|---|---|---|---|
| **P0** | Z-dump + spectral diagnostics (global `torch.linalg.svd`, entity-block `A=UᵀZU` via `spectral_diagnostics()` verbatim) + `discriminability_metrics` at raw `h∈{1,13,37,61}` (13,37 ≡13 mod 24, matching the deep-ladder's own residue — different squaring counts, same effective distance, m6's cheap discriminator), `n=256` (4× `eval_batch_size=64` pooled, still eval-only). Run on the RETAINED `mob_g3b31_compB_s0` checkpoint if it survives on box; else one fresh 20,000-step retrain at compB's exact recipe (D1's own fallback, reused verbatim). | 0 (eval) / 20,000 (fallback) | 1 | 0 / 0.8293 |
| **P1a** | Teacher-force closed-form `Z` at **step 0** (fresh init, `--teacher-force-operator`, no training at all) — pure pipeline-sanity: does the closed-form fit + fixed read machinery retrieve above chance when `Z` is exact by construction and `q_key≡keys_v[a_slot]` by the §G3-B12 fix, decoupled from any learning question whatsoever. `discriminability_metrics` at `h∈{1,13,37,61}`, `n=256`. | 0 | 1 | ~0 |
| **P1b** | Teacher-force closed-form `Z` **after training** (`--teacher-force-operator`, backbone+entity_adapter+embed trained via CE as normal, `ncr_head`/encoder never enters the graph — `teacher_force_check.ncr_zero_grad_checks_passed` asserted every step, reused from §G3-B9's own construction). 5,000 steps (matches the §3.3-precedent per-cell length for a short probe, not a full 20K commitment). `discriminability_metrics` at `h∈{1,13,37,61}`, `n=256`. | 5,000 | 1 | 0.2073 |
| **P2** | Common-mode-centered re-score of P0/P1a/P1b's own stored `o`/target tensors (batch-mean-direction removal, re-normalize, re-argmax) at whichever `h∈{1,13,37,61}` sub-cell fails its raw margin. Pure post-hoc re-analysis. | — | — | 0 |

**GPU-h subtotal:** 0.21 GPU-h (best case, retained ckpt survives) to
1.04 GPU-h (worst case, P0 needs the fresh-retrain fallback). ×1.4
contingency (covers exactly one pre-authorized second-seed re-run of
P1b if its reading lands within 1 SD of the margin, not an open-ended
escalation) → **0.29–1.46 GPU-h. Registered ceiling ≤1.5 GPU-h
nominal, hard cap ≤2.0 GPU-h** — matches W7's own charter exactly
("its own ≤2 GPU-h stage with 1 audit round"). Placement: one cell at
a time is fine on a single free GPU (P0/P1a/P1b are sequential-cheap,
not a parallel sweep); no packing decision needed at this size.

#### R1.1.2 Statistical margin (one number, reused across every sub-test)

Chance `p=1/24=0.041667`, per-item SD `=√(p(1−p)/n)`. At the battery's
own `n=256` (4 pooled eval batches, eval-only so pooling is free): `SD
= 0.024978/√4 = 0.012489`. **Margin: `τ = chance + 4·SD = 0.041667 +
0.049956 = 0.09162`.** One-sided false-positive rate per test `≈3.2×
10⁻⁵`; even summed naively (not independence-corrected) over the
battery's ≤8 sub-tests (P0/P1a/P1b × raw+centered, minus the ones P2
never needs to touch), familywise `≤2.6×10⁻⁴` — no further multiplicity
correction needed on top of the 4-SD choice itself (D1's own proposed
route, "raise the eval n and re-derive," taken here). `retrieval24_acc`
stays PRIMARY/gating per house convention (§G3-B27); `offtarget_margin`
is recorded alongside every sub-test as a corroborating (non-gating)
signal.

#### R1.1.3 Pre-registered bands (decision tree)

Let **CLEARS(x)** mean `retrieval24_acc(x) > τ = 0.09162` at `n=256`.

- **R-A — AUTHORIZE STAGE 1 AS SPECIFIED (§R1.3's mechanism, cleanest
  reading).** `CLEARS(P1b)` **and NOT** `CLEARS(P0)`. A near-exact
  closed-form `Z` retrieves; the actually SGD-learned `Z` does not ⇒
  clean localization to write-QUALITY — exactly what a write-side
  conditioning penalty targets. This is the reading the whole document
  is betting on.
- **R-B — AUTHORIZE STAGE 1, BASELINE RE-ANCHORED.** `CLEARS(P1b)`
  **and** `CLEARS(P0)`. Surprising (contradicts F1's archived 3/3
  chance-at-h=1 finding at `n=64`) — before trusting it, note P0 IS
  itself a re-run of compB's exact recipe (or an eval of the retained
  checkpoint), so this reading is self-checking, not free-floating. If
  it holds, Stage 1 proceeds but every §3.6-successor band must be
  recomputed against P0's own (now non-chance) h=1 reading, not F1's
  archived numbers.
- **R-C — RE-SCOPE, PIPELINE PROBLEM (kills write-conditioning as
  specified).** **NOT** `CLEARS(P1a)`. Even a closed-form `Z` with
  provably-exact key/query matching fails at step 0 — no learned
  quantity is even in play yet. A perfect `Z` can't be the fix if a
  perfect `Z` already fails; redirect to a pipeline/target-space
  diagnosis, a different (cheaper) document, out of this design's
  scope.
- **R-D — RE-SCOPE, ADAPTER-TRAINING PROBLEM (kills write-conditioning
  as specified).** `CLEARS(P1a)` **and NOT** `CLEARS(P1b)`. The
  pipeline is sound at init but degrades once `entity_adapter`/`embed`
  train, even handed an exact `Z` throughout — implicates the
  adapter/embed's OWN discriminative training (already the subject of
  the exhausted §G3-B22–B32 contrastive-aux road, R2's §G3-B31 R2
  embed-factor finding is the live lead there), not `Z`'s conditioning.
  Out of this document's scope.
- **R-E — MECHANISM MUST ADDRESS THE COMMON MODE (a modifier, not a
  standalone verdict — compose with A/B/C/D above).** `CLEARS(P2)` at
  any sub-cell whose RAW counterpart did NOT clear. A pure
  rotation/scale-invariant conformality penalty (§R1.3) does not touch
  an additive common mode by construction; if removing one recovers
  signal, §R1.3's mechanism needs an explicit common-mode term (e.g.
  applying the penalty to the mean-centered entity block, or a
  separate centering loss) added before Stage 1 launches, regardless
  of which of R-A/B/C/D also fired.
- **R-F — KILL THE LANE.** None of P0/P1a/P1b nor their P2-centered
  variants clear `τ` at any `h∈{1,13,37,61}`. No configuration this
  battery can reach shows above-chance discriminability anywhere;
  write-conditioning (and §0's binding-lever frame itself) is
  FALSIFIED, not merely NULL — closes this design, requires its own
  write-up, exactly the "materially different and more serious
  finding" DRAFT-R0 §3.6 gestured at, now correctly gated on a cell
  cheap enough to actually justify saying so.

### R1.2 W2 — §1 rewritten: the correct discriminability geometry

**F2's fix, stated once and used everywhere below.** For `Z`
diagonalizable with dominant eigenpair `(λ_1,v_1)`, second `λ_2`, and
`q=Σc_iv_i`: `sinθ_h ≈ C·ρ^h`, `ρ=|λ_2/λ_1|`. **The success condition
is `sinθ_h ≈ sinθ_0` (the angle stays where it started — `ρ≈1` gives
`C`, not 0), not `sinθ_h→0`.** Verified numerically both ways this
round (attack R1 §F2, re-derivable from the same formula): a
scaled-orthogonal `Z=cQ` holds `sinθ_h` **constant** at `≈sinθ_0≈0.99`
for a generic random query in `d=25` — it does not collapse toward the
top direction (that IS the collapse signature), and it does not
collapse toward zero either. This document's target is **`sinθ_h/
sinθ_0 ≈ 1`** — no decay in the ratio, at any `h`.

**M1's fix: the fitted constant is inadmissible, so it is deleted, not
re-fit.** `C≈9,900` (DRAFT-R0 §1.2) implies `sinθ<1` only for `h≥48.2`
— for every shallower `h` the model asserts an impossible `sinθ>1`.
`C` is bounded, `C=tanθ_0∈(0,∞)`, generic value `√(d−1)=4.90` for a
random query in `d=25`. **Deleted:** the `C≈9,900` fit and the
`s1/s2≤1.19` bound built on it (DRAFT-R0 §1.3). They asserted a target
that describes no achievable configuration.

**M2's fix, carried honestly, not resolved.** `ρ=|λ_2/λ_1|` (eigenvalue
moduli) is what the derivation needs; `s1/s2` (singular values) is what
every downstream section measured. These are **not interchangeable**:
attack R1's own counterexample embeds `[[0,1.21],[1,0]]` in `d=25` —
`s1/s2=1.21` exactly, `|λ_2/λ_1|=1` exactly, **zero** directional
collapse at any depth. Singular-value flatness is neither necessary nor
sufficient for eigenvalue-ratio flatness in general. **This is why the
mechanism (§R1.3) targets flatness of the WHOLE singular spectrum of a
symmetric object (`AᵀA`), not a top-two ratio** — flattening all
singular values of `A` forces `‖A‖=‖A⁻¹‖⁻¹`, which bounds every
eigenvalue's modulus between the (now-equal) smallest and largest
singular value (`|λ_i|≤s_1` always; `|λ_i|≥s_min` requires `A`
invertible, true here since the ideal write is a permutation). **The
residual risk is disclosed, not hidden:** a matrix can have a
perfectly flat singular spectrum and still be far from normal (a
scaled orthogonal matrix, `Z=cQ`, is automatically normal — `QᵀQ=QQᵀ=I`
— so the TRUE task solution is not a counterexample to this claim, but
a training trajectory could in principle pass through a flat-but-
non-normal intermediate point). `depart_normality`
(`spectral_diagnostics`, reused verbatim) is co-scored specifically to
catch this residual gap — flagged as a live risk, not claimed solved
(§R1.8).

**D2's re-pre-registered instrument: `o_pairwise_cos(h)` directly, not
`sinθ_h`.** Bounded `[-1,1]` by construction (no admissibility failure
possible), already measured every eval call, directly coupled to
`retrieval24` (both are computed from the same cosine machinery,
`discriminability_metrics`). Define the **discriminability statistic**
`D_h := 1 − o_pairwise_cos(h) ∈[0,2]` (0 = fully collapsed to one
direction, larger = more spread among per-document reads — a
*necessary* precondition for `retrieval24` to clear chance, since
`argmax_k cos(o,T_k)` cannot discriminate `k` if every `o` is the same
vector, though NOT sufficient by itself — retrieval also needs the
spread to be aligned with the RIGHT `k`, which `D_h` alone cannot see;
`retrieval24` stays the PRIMARY arbiter everywhere in this document,
`D_h` is a diagnostic/interpretive aid, never a substitute pass
criterion — the exact lesson F2 already taught this program once).
compB's own measured trajectory: `D_1=0.2005 → D_2=0.03 → D_3..61≈
0.008–0.011` (from `o_pairwise_cos` 0.7995/0.97/0.989–0.992).

**Admissible decay law, fit directly on `D_h` (bounded by
construction):** `D_h ≈ D_{h_ref}·r^{h−h_ref}`, `r∈(0,1]`, `h_ref` set
by W1's premise-cell reading (the shallowest depth with a real,
above-chance signal to preserve — never assumed to be `h=1` by
default, per R-A/B/C/D above). `r=1` ⇒ no decay ⇒ perfect preservation
of whatever discriminability exists at `h_ref`; `r<1` ⇒ decay. Per-run
`(D_{h_ref}, r)` fit from the calibration sweep (§R1.6), THEN `r_fit`
regressed against the measured entity-block spectrum (`s1/s2`,
`depart_normality`) — M9's fix: **per-run fits, not one pooled (C,ρ)
regression across six different `Z`'s** — and calibration runs at
**20,000 steps, the target config** (CLAUDE.md's calibration rule; the
old 5,000-step cells are retired), giving **54 (strength, h) points**
(6 hops × 9... correction: the runner's own ladder is `train_hops
(1,2,3)` + `deep_ladder (5,12,20,29,40,61)` = 9 hops per run; 6 runs
(3 strengths × the mechanism, §R1.6) × 9 hops = **54 points**, m1's
count, not DRAFT-R0's mis-stated 42).

**The honest current numeric estimate (F4, admissibly backed out, not
re-derived from scratch — M1's own admissible-constant table):** using
`C=4.90` (generic query) or `C=1.00` (the document's own O(1)
convention) and requiring `D_{61}` above a discriminability floor,
the implied **effective whole-spectrum ratio is `s1/s2_eff ≤ 1.04–
1.07`**, and F4's own direct simulation (best-case hinge spectrum
`(r,1,…,1)`, queries at compB's measured TPC) shows `o_pc(h=61)`
saturates to the collapsed value's own level (`≈0.877`) for `r≥1.10`
and only **`r≤~1.02` keeps `o_pc(h=61)` near its `h=1` level**. Carried
forward as the working target: **`s1/s2_eff ≤ 1.02–1.05`**, explicitly
flagged as saturating sharply above `~1.02` — the target is stated on
the **effective, whole-spectrum ratio** (what `§R1.3`'s mechanism
actually controls), not the raw global top-two ratio M2 showed is
decoupled from the true decay rate. **Re-derived from the REWRITTEN
54-point calibration before any Stage-1 cell is scored against it** —
this document does not freeze a number it cannot yet measure honestly.

### R1.3 W3 — the mechanism: entity-block singular-value flatness

**Exact definition of `A`.** `A := UᵀZ_rawU`, `Z_raw =
ncr_head.encode(keys_v, values_v)` (the SAME tensor
`ncr_lm_forward_ablatable` already returns — no extra forward pass),
`U` the `d×K` orthonormal entity-subspace basis from
`az.entity_subspace(z_ideal)` (`chapter2/analyze_zdump.py:184-197`,
reused verbatim — SVD of the task's own ideal K-cycle operator,
`k_eff=K=24` at this config, `V` the `d×(d−K)=d×1` complement, the
SAME machinery `spectral_diagnostics()` already calls). `U` is treated
as a **fixed constant per training step** (computed once from
`z_ideal`, never differentiated through) — a design choice already
implicit in the reused `spectral_diagnostics` convention, stated
explicitly here since the loss now backpropagates through it.

**The loss, and its closed form.** Let `G=AᵀA` (`K×K`, symmetric
PSD), `t=tr(G)/K`:

```
L_conf = λ · ‖G − t·I_K‖²_F / t²                         (as specified, D3)
```

Expanding (`‖G−tI‖²_F = tr(G²) − 2t·tr(G) + t²K`, and `tr(G)=tK`):

```
L_conf = λ · [ K² · tr(G²) / tr(G)² − K ]                 (closed form — no SVD, no matrix inverse)
       = λ · [ K² · Σᵢsᵢ⁴ / (Σᵢsᵢ²)² − K ]                (sᵢ = singular values of A, i=1..K)
```

By Cauchy–Schwarz, `(Σsᵢ²)² ≤ K·Σsᵢ⁴`, so the bracket is **`≥0`,
`=0` iff every `sᵢ²` is equal** (i.e. `A=c·Q` for orthogonal
`Q∈O(K)`, some scalar `c`) — this is exactly the flatness condition
§R1.2 derived, now on the whole entity-block spectrum, not a top-two
ratio, closing M3's rank-2-collapse loophole by construction (a
`(1,1,ε,…,ε)` spectrum scores `f>0`, not the exact-zero M3 found for
DRAFT-R0's hinge). **Zero at the ideal K-cycle write, proved, not
asserted:** the true `z_ideal`'s entity block is (isomorphic to) a
`K×K` permutation matrix — orthogonal, all `sᵢ=1` — so `L_conf(A^*)=0`
exactly (matches attack R1's own numeric check, `L_correct(Z^*)=0`).
`L_conf(c·I)=0` too (flat spectrum trivially), but — unlike F3's killed
mechanism (b) — the TRUE solution is **equally** at the minimum, not
penalized relative to it; there is no attractor competing with the
task.

**A useful reading of the same quantity.** `f(A)=K²Σsᵢ⁴/(Σsᵢ²)²−K =
K·(K/PR−1)`, where `PR:=(Σsᵢ²)²/Σsᵢ⁴` is the participation ratio of
the squared-singular-value distribution (a Rényi-2/collision-entropy
flatness statistic — the same family as `effective_rank`
(`analyze_zdump.py:166-171`, Shannon-entropy flatness) and
`stable_rank` (`:174-178`, `Σsᵢ²/s₁²`) already computed elsewhere in
this program, but differentiable and used here as a training signal
rather than a diagnostic). `PR=K` (maximally flat) ⇒ `f=0`; `PR=1`
(all mass on one singular value) ⇒ `f=K(K−1)`, the maximum.

**Scale-invariance (M4's fix applies here too, verified the same way):**
`f(αA)=f(A)` for any `α≠0` — the `t²` normalization is not an add-on
guard, it falls out of the closed form. No separate `‖Z_raw‖_F²`
denominator is needed or present (M4's runaway escape hatch cannot
recur — there is nothing to normalize by that isn't already baked into
`f`'s own ratio structure).

**Gradient (verification derivation — PyTorch autodiff computes this
in practice; shown so the formula is checkable, matching house
convention for §2(b)'s `ĉ` derivation).** With `G=AᵀA`, `D=tr(G)`,
`N=tr(G²)`:

```
∂f/∂A = (4K²/D³) · A · (D·G − N·I_K)              [standard: ∂tr(G)/∂A=2A, ∂tr(G²)/∂A=4AG]
∂L_conf/∂Z_raw = U · (λ·∂f/∂A) · Uᵀ                [A=UᵀZU linear in Z, U fixed]
```

**Guard (carried from §2(b)'s failure-mode 3, applied to the new
self-referential scale `t`):** floor `D=tr(G)` at a small `ε` in the
denominator before the divide (`D` is the entity block's own squared
Frobenius norm — degenerates only if `Z_raw`'s entity block collapses
toward zero, the same class of degenerate-scale risk §2(b) already
flagged, guarded identically).

**Cost.** `A=UᵀZU`: two matmuls, `d²K+dK²` ≈ `25²·24+25·24²=29,400`
FLOPs. `G=AᵀA`: `K³=13,824`. Gradient: same order. **Total ≈3×10⁴
FLOPs/write** — negligible next to the 98M backbone, and cheaper than
DRAFT-R0's rejected top-2 power-iteration route (`24d²×12≈1.8×10⁵`)
while covering the WHOLE spectrum, not two singular values.

**Why this evades §G3-B32's exhausted read-side road, letter and
mechanism (stated here, not in an appendix, per charter).** The
exhausted aux road (§G3-B17–B32) operates on `cos(o, target)` —
comparisons between the READ OUTPUT after `h`-fold composition and a
target/adapter space that itself can (and did, in 0992/0993) collapse.
`L_conf` is computed **only from `Z_raw`'s own entity-block singular
values** — no dependence on `o`, `binexp_read`, the target embedding,
`entity_adapter`'s read-side role, or `h` at all. It **cannot** inherit
the saturated-instrument pathology (a collapsed target space reading
`recovered_frac@0.9=1.0` for an information-free read) because it
never looks at the target space — the letter argument. It also
attacks a **different mechanism**: §G3-B26/B32's collapse is a
depth-composition artifact (power iteration under repeated squaring);
`L_conf` shapes `Z_raw` once, at write time, before any squaring
happens — the mechanism argument. Additionally (new this round, closes
M3): it is **localized to the task-relevant subspace by construction**
— `U` is fixed from `z_ideal`, so gradient flows only into the
entity-block projection of `Z_raw`, not into whichever direction is
cheapest to flatten globally (DRAFT-R0's mechanism (c) rejected
exactly this loophole via the *global* ratio).

**MuonSSM / DeltaProduct distinctions (in-section, per charter — the
novelty memo's mandatory cite-and-distinguish anchors, `research/
writecond-novelty-2026-08-13.md`).** **MuonSSM** (arXiv:2606.30461,
ICML 2026 Oral) Newton–Schulz-orthogonalizes a **rank-1** fast-weight
WRITE (a KV outer product) **inside the forward recurrence**, motivated
by long-sequence gradient/memory stability of the *accumulated
recurrent state* — no repeated-squaring read, no `h`-depth
compositional-retrieval evaluation. `L_conf` acts on a **full `K×K`
entity block of a `d×d` WRITTEN OPERATOR** (`K=24` singular values, not
one), is a **soft Frobenius penalty**, not a hard NS-polar
reparametrization, and is scored against **read-time compositional
depth** (`retrieval24` at `h*=61`, `Z^h` applied at query time), a
target class MuonSSM's own paper never evaluates. **DeltaProduct**
(arXiv:2502.10297) constrains Householder-product TRANSITION matrices'
token-by-token state EVOLUTION (provably norm ≤1, governs how the
recurrent state updates step to step); `L_conf` conditions a WRITTEN
operator INSTANCE, self-composed `h` times at query time, never
touching how the state evolved to get there. Both distinctions match
the novelty memo's own recorded mapping exactly (§ below).

### R1.4 Novelty mapping (recorded per the memo's re-entry condition)

The memo's by-mechanism sweep names two OPEN wedges verbatim: *"soft
anchoring toward c·I for a written state — no external occupant"* and
*"differentiable condition-number/restricted-isometry penalty on the
written state — no external occupant."* `L_conf` is squarely the
**second** wedge — a differentiable restricted-isometry-type penalty
(flatness of `A`'s singular spectrum, i.e. `A` close to a scalar
multiple of an isometry) on the entity-restricted written state — and
explicitly **not** the first (no anchor point, no `c·I` target,
sidestepping F3 entirely by construction, §R1.3). This is the SAME
functional family attack R1's D3 disposition already placed inside the
swept-open wedge; §R1.3 is D3's mechanism carried to a full spec (exact
`A`, gradient, cost, proof-at-optimum), not a further mechanism change,
so **no gate re-entry is triggered by this section**. The W5 control
(§R1.5) is a control, not a claim, and does not trigger re-entry
either. Standing cite obligations (MuonSSM, DeltaProduct,
Preconditioned DeltaNet, Variational Linear Attention, Gated
DeltaNet-2, MeSH, Sanford/Wang, RWKV-7/Grazzi, uRNN/scoRNN/expRNN)
carry forward unchanged.

### R1.5 W5 — the true-null control

**Why the obvious fix (D4's spectrum-matched/structure-randomized
anchor, or an entity-complement-block penalty) doesn't work at THIS
config, disclosed rather than silently avoided.** The natural
"apply the same penalty to an irrelevant subspace" null degenerates
here: `V` (the complement of `U`) is `d×(d−K)=25×1` — **one-dimensional**
at this tight-spare config. `D:=VᵀZ_rawV` is a `1×1` scalar, and `f`
evaluated at `K'=1` is **identically zero for every value of `D`**
(Cauchy–Schwarz is tight by definition when there is only one term) —
a "complement-block flatness penalty" would be a no-op control, not a
null with real gradient pressure. (This is the same `d=K+1` spare-
direction fact `NCR_ORTHO_WRITE.md` §10.7 and DRAFT-R0 §2(c) failure-
mode 3 already flagged from a different angle — recorded here as the
reason the D-block route was considered and rejected, not silently
skipped.)

**Chosen construction: calibrated isotropic gradient-noise injection,
matched at EVERY step, not just at init.** At each training step,
compute `g_t := ‖∂L_conf/∂Z_raw‖_F` evaluated at the null arm's OWN
current `Z_raw` using `L_conf`'s exact formula (§R1.3) — this is a
calibration PROBE only (negligible added cost, same `≈3×10⁴` FLOPs as
§R1.3), its VALUE is used, its GRADIENT is discarded. Sample `ε_t ~
N(0, (g_t/d)²·I_{d×d})` fresh each step and add it **directly to
`Z_raw`'s gradient** post-backward (not as a loss term, avoiding any
interaction with the real loss's own backward graph):

```
Z_raw.grad += eps_t          # eps_t iid N(0, (g_t/d)^2), fresh per step
```

`E[‖ε_t‖_F²] = g_t²` by construction — the null injects a gradient
perturbation with the SAME Frobenius-norm budget `L_conf` would have
injected at that exact `Z_raw`, at every step (this also fixes a risk
this round's own drafting caught: a FIXED-σ noise schedule would drift
out of budget-match as `L_conf`'s own gradient naturally shrinks toward
convergence — the per-step recalibration closes that gap by
construction, not by assumption). **Un-invertible, provably:** `ε_t` is
isotropic and mean-zero, so for ANY fixed direction `Δ` (in particular
the flattening direction `∂f/∂A` itself), `E[⟨ε_t,Δ⟩]=0` — the null
carries no systematic component toward flatness, toward the entity
subspace, toward `Z_raw`'s own current value, or toward anything
task-structured, in expectation at every single step, not merely on
average over a run. This is a strictly cleaner invertibility argument
than D4's originally-suggested spectrum-matched/structure-randomized
anchor (which still anchors toward a specific, if randomized, target
point each step — considered, not chosen, for exactly this reason).

**Pre-registered interpretation (D4, restated under the corrected
control).** Under this null: placebo-improves-retrieval ⇒ evidence for
a generic "extra gradient pressure on `Z_raw`" nuisance effect,
independent of flatness (the exact confound §G3-B22–B25 already
diagnosed once for the aux term); placebo-does-not-improve AND
`L_conf` DOES ⇒ structural evidence specific to spectral flatness, not
merely "more regularization." The prior draft's control (F5) would
have read this backwards; this one cannot, by the argument above.

**Seeds:** `n=4`, `20,000` steps, `4×0.8293=3.32` GPU-h (§R1.6).

### R1.6 W6 — every band re-anchored to raw artifacts

**Band 1 (target-space integrity) — re-registered as a paired-mean CI,
not an every-hop hard rule.** M5's finding stands and is not erased:
under DRAFT-R0's literal rule, compB (0994) VIOLATES at `h=40`
(`TPC_fg=0.22725` vs bar `0.22637`, `+0.00088`) — `§G3-B32`'s own
letter-verdict already says so ("NULL-BY-COLLAPSE by the letter... the
margin is within eval noise"). M5 also showed the every-hop rule has
`≈31%` per-cell false-void probability from noise alone (paired-diff
mean `0.13855`, SD `0.00657`, bar `1.74` SD above the mean). **Fix:**
score the **one-sided 95% upper confidence bound on the mean paired
difference** across the 9 hops (`t`-distribution, `n=9`, per-cell SD
re-measured, not assumed fixed at compB's 0.00657): `INTEGRITY-OK` iff
this upper bound `<0.15` **AND** no single hop exceeds a gross-outlier
ceiling `0.15+3·SD_hop` **AND** `TPC_fg<0.50` absolute at every hop
(tripwire retained verbatim, unaffected by this fix — it discriminates
the true B26 catastrophic-collapse regime, `0.9925–0.9962`, by two
orders of magnitude of slack). **Under the corrected rule, compB's own
numbers: upper CI `=0.13855+1.860×(0.00657/3)=0.14263<0.15`** (passes),
**no hop exceeds `0.15+3×0.00657=0.1697`** (the paired DIFFERENCE at
h=40 is `0.22725−0.07637=0.15088`, which is `<0.1697`) — **compB reads
INTEGRITY-OK under this
document's rule while its own §G3-B32/attack-R1 letter-verdict under
the OLD rule was NULL-BY-COLLAPSE-BY-THE-LETTER.** Both readings are
recorded, per D5, rather than silently picking one; the old rule is
retired as statistically uncontrolled (M5), not as wrong about the raw
number.

**Band 3 (retrieval) — re-derived, false "verbatim §G3-B29" provenance
struck.** §G3-B29's own rule is `retrieval24 MAX over ALL eval points/
splits ≤2×chance` — a DIFFERENT form (max-over-points) that compB
itself fails (`0.09375@h=20>0.0833`), so it cannot be cited as matching
compB "exactly" (M6). This document's Band 3 is a fresh derivation,
scored ONLY at `h*=61`, seed-POOLED (not single-seed): with W3's `n=8`
and W5's `n=4` (§R1.7), pooled `n=8×64=512` (arm b/c-successor) or
`4×64=256` (null), SD`_{512}=0.024978/√8=0.008831`,
SD`_{256}=0.012489`. **PARTIAL floor** (3 SD above chance, on the
SMALLER pooled n so the floor is conservative for both arms):
`0.041667+3×0.012489=0.07917`. **WIN bar unchanged** (`chance+0.15=
0.19167`, M6's own "adequately conservative" verdict stands — `6.0`
SD even at single-seed `n=64`, more so pooled) **AND** the GAP metric
(`full_graft−backbone_only`) also `>0.15` at `h*` (unchanged, `≈8.5`
SD pooled). **Depth-decay PARTIAL signature — recorded as a fact, not
a target.** compB itself already shows `retrieval24_acc=0.09375@h=20 →
0.01562@h=61` — inside DRAFT-R0's own PARTIAL band definition at h=20,
decaying to chance by h*. This is `n=1`, single-seed; whether it
replicates is exactly what W3's `n=8` (§R1.7) will show, disclosed here
as the FIRST recorded instance of this signature (DRAFT-R0 §3.6 item 4
called it "never yet observed anywhere" — that claim no longer holds
against compB's own raw numbers, corrected per D5).

**M7's fix — WIN margin justified on its own statistics, not
transplanted.** The `+0.15` accuracy-margin WIN bar is `6.0` SD above
chance at `n=64` and more at pooled `n`; this is now stated as the
bar's OWN justification (a large-SD threshold on a binomial proportion)
rather than citing TPC's paired-drift-tolerance precedent (M7's
critique of the transplant stands and is not repeated).

**Architecture-conditionality caveat, carried forward per charter.**
The (now-retired) mechanism (b)'s `c*·I` motivation rested on the
Z-dump complement finding measured in a config where the complement
dimension is near-empty (`fD≤3×10⁻¹²` in DeltaNet-family states, per
`research/writecond-novelty-2026-08-13.md`'s internal-sweep note) —
**this caveat is largely MOOT for §R1.3's mechanism**, which never
targets the identity direction or relies on that finding at all
(F3's failure mode does not apply to a target-free flatness penalty).
Disclosed per W6's instruction regardless, since the caveat is a
standing obligation on any mention of the ortho/conformal track, not
conditional on which specific mechanism is live this round.

### R1.7 Wave-1 budget re-derivation

**W1 (pre-CLEAR, §R1.1.1):** 0.29–1.46 GPU-h, registered ceiling
**≤1.5 GPU-h nominal, hard cap ≤2.0 GPU-h**. This is the ONLY spend
authorized ahead of gauntlet CLEAR (W7).

**Stage 0 (calibration, gated on W1 AND full gauntlet CLEAR):**

| cell | purpose | steps | strengths | GPU-h |
|---|---|---|---|---|
| 0.1 | decay-law fit for §R1.3's single mechanism (`λ` low/med/high) | 20,000 | 3 | `3×0.8293=2.488` |
| 0.4 | pre-build CPU check — `az.entity_subspace`/`match_eigenvalues` on `z_ideal`, verifies the entity/K-cycle structure the flatness target's proof (§R1.3, "zero at the ideal write") assumes | — | — | 0 |

Cell "0.0" (baseline replication) is **subsumed into W1's P0** (D1's
own instruction), not double-spent here.

**Stage-0 subtotal: 2.488 GPU-h.**

**Stage 1 (main grid, gated on Stage 0's fit AND full gauntlet CLEAR)
— W4's reallocation applied: mechanism (a) is cut ENTIRELY from
wave-1 (no canary either — W4 supersedes attack R1's D7, which had
kept a 0.12 GPU-h canary; the coordinator's adjudication put (a)
fully at "exactly where §N2 put it," idle-filler-only), and the
former TWO primary arms (b)+(c), `4+4=8` seeds total, consolidate into
ONE mechanism (§R1.3) — freed budget reallocated to more seeds of the
surviving arm, not held back:**

| arm | seeds | steps | GPU-h |
|---|---|---|---|
| §R1.3 entity-block flatness (sole mechanism) | 8 | 20,000 | `8×0.8293=6.634` |
| §R1.5 true-null (calibrated noise injection) | 4 | 20,000 | `4×0.8293=3.317` |
| blank-out/localization battery (§4, reused verbatim) | bundled, eval-only | — | 0.05 |

**Stage-1 subtotal: 10.001 GPU-h.**

```
Stage 0                         2.488 GPU-h
Stage 1                        10.001 GPU-h
---------------------------------------------
Nominal total (post-CLEAR)     12.489 GPU-h
× 1.4 contingency              17.485 GPU-h
---------------------------------------------
Registered ceiling  ≤18 GPU-h nominal, hard cap ≤25 GPU-h
```

Down from DRAFT-R0's ≤20/≤30 — a leaner design (one mechanism, not
three) earns a smaller ceiling, not the same one carried by inertia.
Both budgets (W1's ≤2 GPU-h and Stage 0/1's ≤25 GPU-h) are reported
separately because they are authorized at different gates — W1 now,
Stage 0/1 only after a full multi-round CLEAR (W7) — and are not to be
summed into a single pre-authorization.

**Placement.** Unchanged reuse of the §G3-B31 measurement (6.86 GB
VRAM, 73–80% SM, one cell/GPU, no packing) for every Stage-0/1 cell;
W1's battery is small enough (3 short/zero-cost cells) to run
sequentially on one GPU without a placement decision.

### R1.8 What kills this at R2 — a fresh, honest self-attack

1. **The noise-injection null's per-step calibration could itself be
   gamed or drift.** `g_t` is measured from `L_conf`'s formula applied
   to the NULL arm's own (untouched) `Z_raw` — if the null arm's
   `Z_raw` drifts toward a very different regime than the real arm's
   (plausible: one arm is being actively flattened, the other is not),
   `g_t` tracks "how far THIS arm's `Z` currently is from flat," not
   "how much pressure the real arm is actually under" — the two could
   decouple over training in a way that is not obviously wrong-signed,
   but is not obviously innocuous either. Flagged as unresolved, not
   claimed fixed; an R2 attacker should try to construct a trajectory
   where this decoupling produces a systematically MIS-matched budget.
   **This is the top risk of this revision.**
2. **`K=24, d=25` makes the entity block ≈96% of the whole matrix** —
   at this exact tight-spare config, "entity-block-restricted" barely
   differs from "global" (M3's original localization concern), even
   though §R1.3's construction is genuinely subspace-restricted in
   general. The mechanism's specificity claim (evades M3 "by
   construction") is TRUE as a mathematical statement about the
   construction, but its PRACTICAL bite at K=d−1 is unverified and
   could be nearly indistinguishable from a global penalty in practice
   — an R2 attacker should check whether §R1.3's mechanism, AT THIS
   CONFIG, actually behaves differently from a naive global-Z flatness
   penalty, or whether the distinction is real only at smaller K/d
   ratios this document never tests.
3. **`D_h`'s admissible decay law (`D_h≈D_{h_ref}·r^{h-h_ref}`) is an
   empirical fit choice, not derived from first principles** the way
   the old (wrong) `sinθ_h≈Cρ^h` model at least tried to be — it could
   itself turn out to be the wrong functional form (e.g. a
   stretched-exponential or two-timescale decay), and §R1.2 does not
   prove otherwise, only that it is bounded and admissible where the
   old model was not. The calibration's own `R²` check (§R1.2,
   inherited from DRAFT-R0 §3.3's gate) is the only safeguard; an R2
   attacker should ask whether a single functional form fit across all
   `λ` strengths is defensible before the 54-point sweep exists to
   check it against.
4. **W1's decision tree (R-A through R-F) assumes the six outcomes are
   close to exhaustive and mutually informative** — a real result could
   land in a genuinely ambiguous zone the tree does not anticipate
   (e.g. `P1a` clears narrowly while `P1b` clears more narrowly still,
   neither cleanly CLEARS nor fails, both near `τ`) with no registered
   tie-break beyond the one pre-authorized P1b reseed. An R2 attacker
   should pressure-test whether `τ`'s 4-SD margin is tight enough that
   a real, non-degenerate signal could plausibly straddle it, which
   would leave the whole design without a clean gate despite the
   statistical care taken.

---

Rev-1 dispatched 2026-08-13. Attack R2 next; no build ceremony
authorized until the full multi-round gauntlet (W7) reaches CLEAR.

---

## §A2-ADJUDICATION (coordinator, 2026-08-13) — attack R2 = BLOCKED (5F/12M/8m) ADOPTED; MECHANISM DEAD-AS-SPECIFIED + RECORD CORRECTION; premise battery repaired-then-launch; mechanism track ON HOLD pending battery data

Report: `NCR_WRITECOND_ATTACK_R2.md`, four FATALs demonstrated by
executed scratchpad code, one by raw artifacts both prior rounds
missed. Settled-clean for R3 (do not re-litigate): the closed form +
gradient (independently reproduced, rel-err 3.5e-8), scale
invariance, the rank-2 route closed, the 0.8293 GPU-h figure, all
ten §W6 band anchors to five decimals.

**F1 — ADOPTED; L_conf is DEAD AS SPECIFIED.** "Zero at the ideal
K-cycle write" requires ORTHONORMAL adapted entity keys — true in
the toy by construction (`synthetic_keys_from_pi`), FALSE in the LM
graft. At compB's measured target geometry the ideal write scores
f ≈ 149–186 (random Gaussian 24.5; c·I = 0), and a 25% step along
the penalty's gradient collapses h=13 retrieval 0.708 → 0.042. This
re-instantiates the exact F3-class failure that killed mechanism (b)
— and the only key-geometry-compatible repair lands inside the
§G3-B17–B32 exhausted lane. No spectral-flatness penalty on the
written state survives non-orthonormal key geometry.

**F2 — ADOPTED, with a COORDINATOR RECORD CORRECTION.** DRAFT-R1's
"teacher-force never re-run post-fix" claim — which tick #3 and
EXPERIMENT_LOG 2026-08-13 #3 propagated as a "KEY ARCHIVE
DISCOVERY" — is FALSIFIED by `sanity_g3b12_tf_s0.json` (§G3-B13,
params.integ=38400): the POST-FIX teacher-forced pipeline reads
`answer_accuracy = 1.0000 at every hop to h=61`. Cited zero times
in DRAFT-R1 and zero times in attack R1. The correction is appended
to EXPERIMENT_LOG (2026-08-13 #4) and superseded in STATE.md —
never silently. Scientific import: the write/read MECHANISM is
perfect under teacher forcing; the failure is a JOINT-TRAINING
phenomenon. P1a/P1b are re-founded on §G3-B13 accordingly (bands
referenced to 1.0, not chance).

**F3 — ADOPTED:** the noise-injection null systematically FLATTENS
the spectrum (E[f] 341→275 at matched budget; →140 over a
4,000-step walk) while being 141× weaker in coherent displacement —
contaminated AND under-dosed. Dead. **F4 — ADOPTED:** the 6-branch
tree double-fires on 3/8 outcomes (twice AUTHORIZE-vs-KILL) and
`CLEARS(·)` lacks a depth quantifier. **F5 — ADOPTED:** the wedge
is OCCUPIED INTERNALLY — the pinned runner already trains
`0.1·‖ZᵀZ−I‖²/d²` on Z in all three baselines (so §G3-B32's
collapse happened WITH a spectral penalty active — direct evidence
AGAINST conditioning-strength as the missing lever) — and
externally on three counts (incl. functional identity with matrix
Rényi-2 entropy). The novelty memo gains this internal-occupancy
correction.

**DISPOSITIONS:**
- **X1 (premise battery: REPAIR THEN LAUNCH, pre-CLEAR, per the
  attack's own ruling).** The four documentary repairs, none
  needing GPU or a new audit round: (i) P1a/P1b re-founded on
  §G3-B13 with bands referenced to 1.0 and `answer_accuracy`
  co-scored; (ii) the decision tree rewritten as an enumerated
  PARTITION with a depth quantifier on CLEARS; (iii) the n=256
  pooling pinned against the runner's deterministic per-(seed,h)
  eval seeding (naive pooling gives n_eff=64 / a 2.0-SD bar at ~5%
  per-test FP — re-derive the bar honestly) + eval-VRAM re-smoke;
  (iv) P0's fresh-retrain fallback DROPPED from the ≤2 GPU-h
  authorization (measured 0.83–0.92 s/step ⇒ 4.6 GPU-h, aborts at
  ~43%). On completion + coordinator spot-check, the battery
  LAUNCHES on free GPUs — it is the highest-value cheap cell in
  the queue, made MORE interesting by F2.
- **X2 (mechanism track: ON HOLD).** L_conf as specified is dead
  (F1) and the conditioning-strength story is evidenced against
  (F5). No mechanism revision is dispatched until the premise
  battery's data lands — its branches (now including the
  teacher-force-perfect fact) determine whether the next lever is
  key-geometry conditioning, joint-training dynamics, or a
  re-scoped claim. Any new mechanism re-enters the novelty gate
  (the memo's re-entry clause), which now also carries F5's
  internal-occupancy fact.
- X3: the R2-settled-clean list is frozen for R3; the record
  correction stands in log #4.

Rev-2 (X1 repairs only) dispatched 2026-08-13.
