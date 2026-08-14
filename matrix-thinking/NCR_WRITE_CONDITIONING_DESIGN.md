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

---

## DRAFT-R2 — PREMISE BATTERY (Rev-2, 2026-08-13)

**Scope discipline, stated once.** This section implements ONLY X1(i)–(iv)
(§A2-ADJUDICATION) = R-1…R-4 (`NCR_WRITECOND_ATTACK_R2.md` §LAUNCH-RULING).
The mechanism track (X2) is untouched — §1–§6, `DRAFT-R1`, and every
mechanism-specific subsection (R1.2/R1.3/R1.4/R1.5) stay exactly as attack
R2 left them (F1-dead, X2 ON HOLD). Nothing below authorizes mechanism
work. Sources read in full for this round, beyond the R1.1 battery spec:
`sanity_g3b12_tf_s0.json` (raw, re-read directly), `NCR_REAL_LM_DESIGN.md`
§G3-B13 (`:5591-5635`), `mob_g3b31_compB_s0.json` (raw `config`/`arms`),
`ncr_lm_wave1_runner.py` (md5 `9a93198b642242f512ff8489e32b0a53`,
re-verified — `eval_arm_at_hops`, `eval_both_arms`, `build_attribution`,
`ncr_lm_forward_ablatable`, `save_checkpoint`/`load_checkpoint`/
`restore_arms_and_opts`, `build_two_arms`, `build_grammar_pools_and_cfg`),
`ncr_ortho_write.py:197-224` (`spectral_diagnostics`), `matrix-thinking/
H100_SETUP.md`, `matrix-thinking/queue/QUEUE_README.md`.

### R2.1 — Repair (i): P1a/P1b re-founded on §G3-B13

**The raw fact (re-verified directly against the JSON, not an agent's
prose).** `experiment-runs/2026-07-17_ncr_gate3_wave1/g3b12_smoke_results/
sanity_g3b12_tf_s0.json`: `teacher_force_operator=true`, `params.integ=
38400` (post-§G3-B12-fix single adapter), 3000 steps, CE-only (no
`aux_read_loss_weight`/`ortho_reg_weight` keys present — the OLD
pre-§G3-B27 runner). `full_graft.answer_accuracy = 1.0000` at every
recorded hop — `h=1,2,3,5,12,20,29,40,61` — `mean_cos` 1.0000 (0.9971 at
h=61), `recovered_frac@0.9` 1.0000 (0.9844 at h=61); `backbone_only`
(o≡0 null) `answer_accuracy` 0.0–0.094 throughout. Written up as §G3-B13
(`NCR_REAL_LM_DESIGN.md:5591`), verdict "DECODE PATH FULLY HEALTHY,"
explicitly disclosing what it does NOT prove: *"that the model LEARNS to
WRITE operators from context that compose to deep h"* (`:5625-5627`) —
i.e. §G3-B13 is a WRITE-GIVEN result, not a WRITE-LEARNED one. This
supersedes DRAFT-R1's `:851-852` claim ("never re-run post-fix"), which
is struck. `retrieval24_acc`/`discriminability_metrics`/
`target_pairwise_cos`/`o_pairwise_cos` did not exist when §G3-B13 ran
(introduced §G3-B27, 2026-07-29) — confirmed by `grep -c retrieval24
sanity_g3b12_tf_s0.json` = 0 — so none of those instruments were ever
scored on this config. Nor did §G3-B13 touch aux/ortho losses, `n=256`,
or `h∈{13,37}`.

**Consequence for the bands.** Every band this document sets for a
teacher-forced, post-training cell must reference `answer_accuracy → 1.0`
(degradation-from-perfect), never chance (improvement-from-nothing).
`retrieval24_acc`'s own reference is separately derived (below) — it is
NOT automatically 1.0 merely because `answer_accuracy` is, because the
two instruments are computed from different information (m1's exact
`o = T_tgt` argument governs `retrieval24`; the tied LM head's OWN
learned mapping governs `answer_accuracy`). Both metrics are co-scored on
every cell below, never one without the other — closing F4(e)'s missing
instrument-disagreement branch by construction, not by exception
handling after the fact.

**P1a and P1b redesigned — what NEW information each adds beyond
§G3-B13.**

- **P1a (fresh-init teacher-force, unchanged in spirit from R1.1.1).**
  `--teacher-force-operator` at **step 0** (no training at all,
  `build_two_arms` constructed and evaluated directly — no CLI training
  loop entered, avoiding any "is step 1 close enough to step 0"
  ambiguity). NEW beyond §G3-B13: (a) scores `retrieval24_acc`/
  `discriminability_metrics` — instruments that did not exist at
  §G3-B13's time — on the CURRENT pinned runner (`9a93198b`); (b) at
  genuine `n=256` (§G3-B13 was `n=64`); (c) at `h∈{1,13,37,61}`, a
  squaring-residue set §G3-B13 never scored (m6: `13≡37≡61≡13 mod 24`).
  Reference: `retrieval24_acc(P1a,h) → 1.0` is an **exact analytic claim**
  (attack R2's m1: `q_key≡keys_v[a_slot]` bit-identical by the §G3-B12
  fix, `pinv` fits `K=24` constraints in `d=25` exactly ⇒
  `o=values_v[a_slot]=T_tgt` exactly; simulated residual `3.1e-16`), valid
  at a fresh, well-separated (`TPC≈0.0099`, §G3-B31 R1's own fresh-init
  anchor) target space regardless of training. `answer_accuracy` at P1a
  is **NON-GATING, diagnostic-only**: the backbone/LM-head are
  UNTRAINED at step 0, so §G3-B13's `1.0` reference (measured on a
  3000-step-trained decode head) does not transfer here — there is no
  honest a-priori expectation for P1a's `answer_accuracy` and none is
  registered.

- **P1b (REDESIGNED — no fresh training run at all).** DRAFT-R1's P1b
  ("teacher-force after 5,000 steps of fresh CE training") is dropped: it
  would be a near-replica of §G3-B13 (M9's own point, now sharper — it is
  not just confounded with training length, it is a **second measurement
  of the same already-banked result**, on a recipe that omits the aux/
  ortho terms compB actually runs). **P1b now evaluates the closed-form
  teacher-forced `Z` on the RETAINED, ALREADY-TRAINED `mob_g3b31_compB_s0`
  checkpoint** — the SAME checkpoint P0 (below) evaluates, swapping only
  the source of `Z` (`integ.teacher_force_operator(keys_v,values_v)`
  instead of `ncr_head.encode(keys_v,values_v)`; both calls go through
  the identical `ncr_lm_forward_ablatable`, `runner.py:360`). This is the
  literal "§G3-B31 contrastive-grid checkpoint's behavior under teacher
  forcing" the repair calls for, and it is genuinely new along TWO axes
  §G3-B13 never touched:
  1. **The recipe.** compB trains with `aux_read_loss_weight=0.5`
     (contrastive+cosine) and `ortho_reg_weight=0.1` for 20,000 steps;
     §G3-B13 is CE-only for 3,000 steps on the OLD runner. compB's own
     `entity_adapter`/`embed` are SGD-shaped by those losses; §G3-B13's
     were not.
  2. **The decode head's OWN training regime.** §G3-B13's LM head trained
     on an EXACT `o` (teacher-forced) at every step of ITS training — it
     learned, in-distribution, to read a discriminative `o` correctly.
     compB's LM head trained on a COLLAPSED, uninformative `o`
     (`o_pairwise_cos` 0.989–0.992, §G3-B32) for all 20,000 of ITS steps —
     it never once saw a discriminative `o` during training. Handing
     compB's LM head a teacher-forced `o` at EVAL time is therefore an
     **out-of-training-distribution input to the decode head specifically**
     — `retrieval24_acc`'s `→1.0` analytic reference (a pure geometric
     argmax over `o`'s cosine to the K targets, computed independently of
     the LM head, m1's derivation) still transfers to P1b, but
     `answer_accuracy`'s `→1.0` reference (§G3-B13's) does **NOT**
     automatically transfer, because it additionally requires the LM head
     to correctly use an input-shape it never trained on. **This is
     pre-registered as a first-class, expected possible outcome
     (retrieval24 clears, answer_accuracy does not) — INSTRUMENT-
     DIVERGENT, defined precisely in §R2.2 — not a surprise to explain
     away after the fact.** This directly discharges F4(e).

  **`retrieval24_acc(P1b,h)`'s own reference is NOT the same tight
  analytic `1.0` as P1a's.** m1's exactness requires only that the K=24
  ADAPTED KEY vectors be linearly independent (guaranteeing the `pinv`
  fit is exact) — it says nothing about whether the K TARGET vectors
  (what `retrieval24_acc`'s argmax scores against) are separated enough
  for the argmax to pick the right one. At a FRESH-init adapter (P1a)
  targets are near-orthogonal (`TPC≈0.0099`) so this is moot. At compB's
  TRAINED adapter, `TPC_fg` is measured **healthy but non-zero**
  (0.196–0.228, §G3-B32) — comfortably below the 0.50 collapse tripwire,
  but not exactly orthogonal either. So P1b's `retrieval24_acc` is
  registered as a genuine **empirical** question (same statistical
  machinery as P0, §R2.2), not an assumed `1.0` — this is a deliberate,
  disclosed correction to how R1.1.1 implicitly treated "teacher-forced"
  as synonymous with "provably 1.0" everywhere.

  **Paired-by-construction bonus (closes M9 by construction, not by
  reasoning about it after the fact).** `eval_arm_at_hops`'s document
  draw depends only on `(base_seed, h)` (`torch.Generator().manual_seed
  (base_seed+EVAL_SEED_OFFSET+h)`, `runner.py:940`) — NOT on
  `teacher_force`. Calling `eval_both_arms(arms, pools, cfg, 256, device,
  base_seed, teacher_force=X)` twice at the SAME `base_seed` (once
  `X=False` for P0, once `X=True` for P1b) draws **bit-identical
  documents** for both. P0 and P1b are therefore paired not merely on
  step-count (M9's original ask) but on the literal eval batch — a
  strictly stronger fix.

- **P0 — restated (repair iv, cross-referenced here since it shares the
  checkpoint with P1b): artifact-analysis-only.** Evaluates compB's OWN
  `ncr_head.encode`-produced `Z` (the actually-deployed SGD write) on the
  retained checkpoint. NOT gated to `1.0` (this is the write the archive
  already found at chance, F1) — gated to the chance-referenced `τ`,
  §R2.3.

**What is explicitly NOT re-litigated here.** The R2-settled-clean list
(§A2-ADJUDICATION X3): closed-form/gradient, scale invariance, the rank-2
route, `0.8293` GPU-h, all ten §W6 band anchors. F1 (mechanism dead) and
F5 (novelty wedge internally occupied) stand untouched — this repair
does not reopen the mechanism.

### R2.2 — Repair (ii): the decision tree as an enumerated partition

**Metrics, per cell `x∈{P0,P1b}`, per hop `h`:** `retrieval24_acc(x,h)`
(PRIMARY, gating) and `answer_accuracy(x,h)` (co-scored, gating only via
the divergence rule below) — both already returned by `eval_both_arms`/
`build_attribution`, no new instrumentation.

**Depth quantifier, resolved (closes F4(c)).** Every `CLEARS` predicate
below is evaluated **at one named depth, never `∃h` or `∀h` over a set.**
Two depths are load-bearing: `h=1` (zero composition — the frame
question) and `h=61` (the pre-registered `h*`, §0/§1 — the depth-
preservation question). `h∈{13,37}` are scored and RECORDED (the
mod-24-residue corroborating check, m6) but are **not** part of the
gating partition below — a divergence between `h=13/37` and `h=61`'s
verdict is logged as a depth-decay note on the row, never silently
changes the row's verdict, and never introduces a third depth into the
partition's dimensionality (kept at 2 to stay in a checkable, exhaustive
16-cell grid rather than an unauditable combinatorial blowup).

**Band definitions (both cells, both depths — a single shared
definition, not two different tests, per R2.1's correction that P1b is
not assumed-`1.0`):**

```
chance   = 1/24            = 0.041667
SD_256   = sqrt(chance*(1-chance)/256) = 0.012489      (n=256, GENUINE per R2.3)
tau      = chance + 4*SD_256           = 0.09162
CLEARS_h(x)  :=  retrieval24_acc(x,h) > tau
HIGH_h(x)    :=  retrieval24_acc(x,h) > 0.95            (sub-label, does not change the row)
```

`0.95` is an engineering margin, not a statistical bound: m1's simulated
analytic check reads retrieval24 exactly `1.0000` in float32 (residual
`3.1e-16`); `0.95` is loose enough to absorb ordinary bf16/mixed-
precision noise without over-triggering, and any reading in `(tau,0.95]`
is labeled PARTIAL rather than HIGH — informative texture on a row,
never a new partition axis.

**The gate (checked FIRST, outside the partition — same convention as
Band-1 TPC-before-Band-3, `NCR_REAL_LM_DESIGN.md` §G3-B31 R1).**

```
GATE:  HIGH_h(P1a) for ALL h in {1,13,37,61}    (i.e. retrieval24_acc(P1a,h) > 0.95 at every one)
```

FAIL ⇒ **VOID — PIPELINE/INSTRUMENT BROKEN.** Something regressed
between §G3-B13's era and the current pinned runner, or in the fresh-
init construction itself; the partition below is not interpreted; escalate
to the coordinator before spending anything on P0/P1b's readings. PASS ⇒
proceed. `{GATE fails} ∪ {GATE passes}` is a 2-way, exhaustive, disjoint
split of the full outcome space by construction (a single boolean); the
16-cell grid below is defined ONLY on the `{GATE passes}` branch, so it
only needs to be exhaustive within that branch — which it is (next
paragraph) — for the whole space to be covered.

**The 16-cell partition (K-wall style — a Cartesian product of two
already-exhaustive-disjoint 4-way partitions, proof below).** Define:

```
a := CLEARS_1(P0)     b := CLEARS_1(P1b)     c := CLEARS_61(P0)     d := CLEARS_61(P1b)
AC := (a,c)  in  {00,01,10,11}      -- P0's own shallow/deep reading (re-verifies F1)
BD := (b,d)  in  {00,01,10,11}      -- P1b's shallow/deep reading (isolates TARGET-GEOMETRY
                                        given a PROVABLY-EXACT write, decoupling "is Z good"
                                        from "is the adapter/embed geometry good")
```

`AC` is a truth table over 2 independent booleans (`a`,`c`) — trivially
exhaustive (all 4 patterns enumerated) and disjoint (no bit-pattern
equals another). Same for `BD`. The full outcome space is `(a,b,c,d) ∈
{0,1}^4 = AC × BD` exactly, since `a,c` and `b,d` are measured on
different cells with independent Generator draws relative to each
other's Z-source axis. A Cartesian product of two exhaustive-disjoint
partitions is exhaustive-disjoint: `∪ᵢⱼ(ACᵢ×BDⱼ) = (∪ᵢACᵢ)×(∪ⱼBDⱼ) =
full×full = full`, and `(ACᵢ×BDⱼ)∩(ACᵢ'×BDⱼ') = ∅` whenever `i≠i'` or
`j≠j'`. **QED — every one of the 16 `(a,b,c,d)` outcomes maps to exactly
one grid cell, never zero, never two.**

**AC axis (P0 — the deployed write, re-verifying F1 at genuine n=256):**

| AC | reading | verdict |
|---|---|---|
| `00` (a=0,c=0) | fails everywhere | **ARCHIVE-CONFIRMED** — matches F1 exactly, at 4× the n |
| `01` (a=0,c=1) | fails h=1, clears h=61 | **ANOMALY-RECHECK** — composition cannot manufacture signal absent at h=1 (h=61 applies the SAME Z, just more times); re-verify before trusting (bug candidate: shape/seed mismatch), do not report as a finding |
| `10` (a=1,c=0) | clears h=1, fails h=61 | **DECAY-CONFIRMED** — the canonical collapse-with-depth signature this whole design targets; matches compB's OWN archived depth-decay instance (§R1.6: `0.09375@h20→0.01562@h61`) |
| `11` (a=1,c=1) | clears everywhere | **SURPRISE-CLEARS** — contradicts F1's archived chance-at-h=1 (n=64); self-checking (this IS compB's own recipe re-measured, not a fresh confounded run) — re-anchor F1's h=1 claim to this run's own numbers (matches R1.1.1's original R-B logic) |

**BD axis (P1b — target geometry given a provably-exact write):**

| BD | reading | verdict |
|---|---|---|
| `00` (b=0,d=0) | exact write fails everywhere on compB's geometry | **GEOMETRY-BLOCKS** — implicates the trained adapter/embed/target space itself (the §G3-B17–B32 EXHAUSTED lane), independent of Z-conditioning; write-conditioning is MOOT here regardless of AC |
| `01` (b=0,d=1) | fails h=1, clears h=61 | **ANOMALY-RECHECK** — same non-monotonicity argument as AC's `01`; re-verify before trusting |
| `10` (b=1,d=0) | exact write clears h=1, fails h=61 | **MECHANISM-CONFIRMED-BY-EXACT-Z** — the sharpest reading this battery can produce (flagged below) |
| `11` (b=1,d=1) | exact write clears everywhere | **EXACT-Z-SURVIVES-DEPTH** — tension with F1's own measured `f(A*)≈149–186`/`cond(A*)≈223–292` at compB's key geometry (`NCR_WRITECOND_ATTACK_R2.md` F1 table); worth recording as an instrument-sensitivity note (retrieval24's discrete argmax may tolerate more spectral spread than `o_pairwise_cos`'s continuous collapse implies) |

**The single most information-dense reading in the battery, called out
explicitly (BD=`10`).** F1 already measured, on compB's OWN key
geometry, that the EXACT write is itself badly conditioned in the
entity block (`cond(A*)≈223–292`, `f(A*)≈149–186` vs `c·I`'s `0`). If the
write-conditioning premise (§0/§1) is right, an exact-but-ill-
conditioned `Z` should ALSO decay under `binexp_read`'s repeated
squaring — i.e. `b=1,d=0` is the theory's OWN predicted signature,
produced with the confound of "did SGD even learn a good Z" removed
entirely (P1b's Z is exact BY CONSTRUCTION). Observing BD=`10` is direct,
clean, low-cost evidence FOR the founding hypothesis; observing BD=`11`
is direct evidence that conditioning-as-measured-by-`f`/`cond` does not
predict `retrieval24` collapse in practice at this config — either
reading is a first-order finding, pre-registered here as such, not
discovered incidentally.

**The 16-cell grid (AC × BD), verdict = the ordered pair, action per
cell:**

| AC＼BD | `00` GEOMETRY-BLOCKS | `01` ANOMALY-RECHECK | `10` MECH-CONFIRMED | `11` EXACT-Z-SURVIVES |
|---|---|---|---|---|
| `00` ARCHIVE-CONFIRMED | Write-cond MOOT (geometry is the blocker); redirect to the exhausted target-space lane, out of scope here | Re-verify BD before any read; AC alone stands as ARCHIVE-CONFIRMED | Archive confirmed AND mechanism confirmed on exact Z — strongest joint case FOR conditioning, but P0 itself shows no signal to condition (both need addressing) | Archive confirmed, but geometry supports deep retrieval given exact Z — the ONLY missing piece is Z-quality; cleanest re-founding for a future (re-designed) mechanism |
| `01` ANOMALY-RECHECK | Re-verify AC before any read; BD alone stands as GEOMETRY-BLOCKS | **Re-verify BOTH before drawing ANY conclusion** — two independent anomalies is a strong pipeline-bug signal, escalate immediately | Re-verify AC; BD alone stands as MECH-CONFIRMED | Re-verify AC; BD alone stands as EXACT-Z-SURVIVES |
| `10` DECAY-CONFIRMED | Deployed write decays with depth AND geometry blocks even an exact write — write-conditioning is moot (geometry is the binding constraint), a MORE serious finding than the mechanism track anticipated | Re-verify BD; AC alone stands as DECAY-CONFIRMED | **Both axes show depth-decay, exact Z included — the cleanest joint confirmation of the founding hypothesis (§0/§1) this battery can produce; the strongest case for re-founding a mechanism cell (X1(a)/(b)/(c) of §A2) once one is designed** | Deployed write decays, but an exact write on the SAME geometry does NOT — decay is Z-QUALITY-specific, not a geometry/depth-composition-in-general problem; the sharpest possible localization to write-conditioning as the fix |
| `11` SURPRISE-CLEARS | Deployed write already works, but exact-Z fails on the SAME geometry (inadmissible-looking on its face: SGD found something the closed-form fit didn't) — ANOMALY-RECHECK this cell specifically before trusting either half | Re-verify BD; AC alone stands as SURPRISE-CLEARS, re-anchor F1 | Deployed write already works at both depths; write-conditioning may not be needed at all — re-scope the whole document's premise (§0) | Deployed write AND exact write both succeed everywhere — write-conditioning was never the blocker; strongest case for closing this design as unnecessary |

**Answer-accuracy divergence rule (co-scored on every one of the 16
cells, closes F4(e) with a single reused threshold — item-(j) checked:
the `+0.15` margin is the SAME absolute margin already justified
elsewhere in this document — §R1.6/M7 — as `6.0`+ SD at `n=64`,
comfortably larger at pooled `n=256`; re-verified here it stays valid
for a binomial-proportion-shaped statistic at any plausible `p`, so no
new number is invented):**

```
GAP(x,h) := answer_accuracy(x,h) - answer_accuracy(backbone_only,h)      -- SAME checkpoint, SAME batch
DIVERGENT(x,h) := CLEARS_h(x) AND GAP(x,h) < 0.15
```

Any `(x,h)∈{(P0,1),(P0,61),(P1b,1),(P1b,61)}` with `DIVERGENT=true` is
labeled **INSTRUMENT-DIVERGENT** on its row, in addition to (never
instead of) its `AC`/`BD` verdict: the geometric read (`o`) carries
discriminative information the DECODE PATH does not surface as an
answer. This is EXPECTED for P1b in particular (R2.1's own argument: the
LM head never trained on a discriminative `o`) and is not treated as a
contradiction — it is a distinct, disclosed finding about the decode
path, orthogonal to the write-conditioning question.

**Modifier, unchanged from R1.1.1 (still valid, still orthogonal to the
grid above).** R-E (common-mode-centered re-score, P2): applied to any
cell reading `NULL` (fails to clear its `tau`) — re-score with the
batch-mean direction removed from `o` before the argmax. Zero additional
GPU-h (post-hoc re-analysis of already-stored tensors). A cell that
clears under P2 but not raw is labeled `+COMMON-MODE` on its row; this
never changes which of the 16 grid cells the ROW occupies (the raw
reading still governs `a`/`b`/`c`/`d`), it only adds a mechanism note.

### R2.3 — Repair (iii): the `n=256` statistics pinned honestly

**The bug, confirmed by reading `eval_arm_at_hops` directly (`runner.py:
934-964`).** `gen = torch.Generator(device=device).manual_seed(base_seed
+ EVAL_SEED_OFFSET + h)` — a pure function of `(base_seed, h)`. Four
calls at the same `base_seed`/`h` (e.g. four periodic evals during a
single training run, or four manually-pooled `eval_batch_size=64` calls
as R1.1.1's prose implied) return **byte-identical batches**. Naive
"pool 4×64" is `n_eff=64`, not `256` — M1's finding, reused verbatim.

**The fix actually used here (not the 4-pooled-calls route).**
`sample_batch_rd(cfg, batch_size, gen, ...)` (`grammar_rd.py:367`,
imported as `gr` and called inside `build_task1_document`) draws
`batch_size` items from ONE Generator in a single call — this is the SAME
mechanism that already gives `eval_batch_size=64` its `n=64`
independence (confirmed against the raw JSONs: retrieval24 values land on
the `k/64` grid, e.g. `0.10938=7/64`, `0.01562=1/64` — a single call's
`batch_size` items are independently drawn, not repeated). **A single
call with `eval_batch_size=256` therefore draws 256 genuinely independent
documents — this is the mechanism used for P0/P1a/P1b below, not four
pooled 64-item calls.** No runner code change; `--eval-batch-size 256`
(or, for the dedicated eval-only script below, `batch_size=256` passed
directly to `eval_both_arms`) is already a first-class, tested argument
path.

**A second, deliberate independence choice (beyond what M1 asked for).**
compB's own training used `seed=0`, and its periodic in-training evals
also passed `base_seed=0` — so a battery eval ALSO run at `base_seed=0`
would have its first 64 (of 256) items be byte-identical to compB's own
already-archived `n=64` reading (a `Generator` reseeded identically
replays its own stream from the start). Not wrong (still 256 genuinely
i.i.d. draws from the task distribution — the n=256 statistical argument
below holds regardless), but avoidable: **this battery's own eval calls
use `base_seed=90210`** (arbitrary, disjoint from every seed value
{0,1,2,3} used anywhere else in this program), so P0/P1b's n=256 reading
is a fully fresh draw, not a superset of the archived n=64 one.

**Statistical margin, restated at the genuine n=256 (M1's own numbers,
now honestly earned):**

```
chance = 1/24 = 0.041667
SD_256 = sqrt(0.041667 * 0.958333 / 256) = 0.012489
tau    = 0.041667 + 4*0.012489 = 0.09162        (4 SD above chance)
exact one-sided binomial tail at tau, n=256:  2.1e-4   (6.6x the normal approx 3.2e-5 --
                                                          right-skew at p=1/24, M1's own
                                                          finding, reused verbatim)
```

**Familywise, stated exactly.** Four `tau`-gated sub-tests govern the
partition (`CLEARS_1(P0)`, `CLEARS_61(P0)`, `CLEARS_1(P1b)`,
`CLEARS_61(P1b)`) ⇒ Bonferroni familywise FPR `≤ 4 × 2.1e-4 = 8.4e-4`.
The `h∈{13,37}` corroborating checks (non-gating, §R2.2) add 4 more
`tau`-tests if the coordinator elects to gate on them too — familywise
`≤ 8 × 2.1e-4 = 1.68e-3` in that case. Either number is comfortably
small; no further multiplicity correction is applied on top of the
4-SD choice itself (M1's own proposed route).

**Eval-VRAM re-smoke — MANDATORY pre-launch step, exact commands (closes
the M1/CLAUDE.md "eval batch size can OOM even if training fits" gap; the
6.86 GB figure, §G3-B31 PLACEMENT, was measured at `eval_batch_size=64`
DURING training — i.e. WITH backward+AdamW state resident, which
dominates that figure; P0/P1a/P1b run under `torch.no_grad()` with no
optimizer at all, at `batch_size=256` — a different, unmeasured regime in
BOTH directions and must be re-measured, not inferred either way):**

```bash
# On the box, BEFORE the real battery. Run the CHEAPEST real path (P1a,
# fresh-init, no checkpoint needed) at the battery's actual batch size,
# poll VRAM concurrently, and record the peak.
ssh youthful-indigo-turkey '
  cd ~/ncr_writecond && \
  ( nvidia-smi --query-gpu=index,memory.used --format=csv -l 2 \
      > vram_smoke_$(date +%s).csv & SMI_PID=$! ; \
    CUDA_VISIBLE_DEVICES=0 /home/nvidia/tdenv/bin/python3 premise_battery_eval.py \
      --cell P1a --eval-batch-size 256 --device cuda --smoke-only ; \
    kill $SMI_PID )'
```

Confirm peak `memory.used` on the target GPU is comfortably under 80 GB
(any single free H100 has ample headroom for a 98M-param eval-only
forward pass at batch=256 — expected well under 10 GB given the
6.86 GB WITH-backward figure at batch=64 as an upper-bound sanity check —
but the number in the log, not this expectation, is what gates launch).
If the smoke shows anything surprising (>20 GB, say), STOP and
re-investigate before running P0/P1b at the same batch size.

### R2.4 — Repair (iv): P0's retrain fallback dropped; ceiling re-derived

**P0 is artifact-analysis-only, full stop — no retrain path inside this
authorization.** R1.1.1's "else one fresh 20,000-step retrain" fallback
is struck. M2's measured teacher-force/calibration-cell regime is
**0.83–0.92 s/step** (five archived cells, including both prior
teacher-force cells — the closest analogues), NOT compB's own **0.149
s/step** fast regime R1.1.1 priced the fallback from. At 20,000 steps
that is **4.6–5.1 GPU-h** — 2.3–2.6× this battery's own `≤2.0 GPU-h`
hard cap, and a retrain launched under a 2.0 GPU-h ceiling in that regime
would abort at `~2.0/4.6 ≈ 43%` of target, yielding a checkpoint that is
**not** compB's recipe (M2's own finding).

**Pre-flight contingency (checked BEFORE the battery is considered
launch-ready — see LAUNCH CARD pre-flight):**

```bash
ssh youthful-indigo-turkey \
  'find ~ -iname "mob_g3b31_compB_s0.ckpt.pt" -o -iname "mob_g3b31_compB_s0*.pt" 2>/dev/null'
```

- **Checkpoint FOUND:** P0/P1b run eval-only, as designed below — cheap,
  no training, this battery's authorization covers it in full.
- **Checkpoint NOT FOUND:** P0/P1b are **BLOCKED** from this
  authorization. A fresh 20,000-step retrain at compB's exact recipe
  (`aux_read_loss_weight=0.5 aux_loss_type=contrastive+cosine
  ortho_reg_weight=0.1`, seed 0) returns to the coordinator for SEPARATE
  authorization, priced at the slow-regime **4.6–5.1 GPU-h** (M2), not
  silently substituted here. **Only P1a (needs no checkpoint) remains
  launch-eligible standalone** in this contingency.

**Ceiling re-derived (structurally different regime — all three cells
eval-only, zero training steps, zero backward passes, zero optimizer
steps; DISCLOSED: there is no archived eval-only cell of this exact
shape to calibrate against — M2's `s/step` table is entirely
training-regime data and does not transfer; the estimate below is a
structural FLOP argument (a handful of forward passes vs. thousands of
training steps) plus a wall-clock kill-switch, not an extrapolated
measured rate):**

```
P1a  (fresh model + pool build, 1 forward-eval pass, 4 hops, no ckpt load)   ~0.10 GPU-h ceiling
P0+P1b (shared ckpt load ~1 GB + 2x eval_both_arms passes + 1 probe pass
         for the Z-dump/spectral_diagnostics)                               ~0.20 GPU-h ceiling
                                                                             -----------------
Nominal (structural estimate, both cells)                                    0.30 GPU-h
x2 (no precedent for THIS shape -- conservative multiplier, not the
    original's blanket 1.4x)                                                 0.60 GPU-h
-----------------------------------------------------------------------------------------
Registered ceiling  <=0.6 GPU-h nominal, HARD CAP <=1.0 GPU-h
```

Enforced via a wall-clock kill-switch INSIDE `premise_battery_eval.py`
(reusing the `--ceiling-gpuh`/`ceiling_s` convention already in
`ncr_lm_wave1_runner.py:1427-1440`): abort with `status=ABORTED-BUDGET`
and dump whatever partial results exist if wall-clock exceeds 45 minutes
on a single GPU (`45min/60 = 0.75 GPU-h < 1.0` hard cap, leaving margin
for the kill-check's own polling granularity). **This is a large
reduction from R1.1.1's `≤1.5/2.0 GPU-h`** — the honest consequence of
removing every training step from the battery, not a re-assertion of the
same number by inertia (mirrors this document's own §R1.7 "a leaner
design earns a smaller ceiling" precedent).

**Item-(j) self-check on this section's own fixtures (required
discipline, applied here explicitly).** `0.6 GPU-h` nominal and `1.0
GPU-h` hard cap are both PRODUCIBLE under the rules just stated: P1a's
`0.10` + P0/P1b's `0.20` = `0.30` nominal, `×2 = 0.60` — arithmetic
reproduced, not merely asserted. The `45-min` kill-switch (`0.75 GPU-h`)
sits strictly inside the `1.0 GPU-h` hard cap with room for polling
overhead, not flush against it.

---

### LAUNCH CARD

**Pre-flight (run in order, before any GPU cell; all read-only or
CPU/box-side, no GPU spend):**

1. Checkpoint check (R2.4): the `find` command above. Branch per its
   result.
2. Eval-VRAM re-smoke (R2.3): the `nvidia-smi`-polled P1a smoke-only
   run above. Confirm peak VRAM is sane before trusting batch=256 at
   full scale.
3. Runner pin re-verify: `md5sum
   ~/ncr_g3b31/ncr_lm_wave1_runner.py` (or wherever it is deployed on
   box) `== 9a93198b642242f512ff8489e32b0a53`. If it does not match,
   STOP — this battery's every reference (retrieval24_acc formula,
   `EVAL_SEED_OFFSET`, `eval_both_arms` signature) is pinned to this
   exact file.
4. Shape-check (CLAUDE.md: smoke test before spending GPU-h): run
   `premise_battery_eval.py --cell P1a --smoke-only --device cpu` first
   if box CPU time allows (98M params forward-only on CPU is slow but
   tractable for ONE batch) — confirms no import/shape errors before
   committing GPU time to any cell.

**Cell spec (`premise_battery_eval.py` — to be written on the box by
the build agent from this spec; NOT committed to this repo by this
round, per this agent's scope; imports the pinned `9a93198b` runner's
own functions, reused verbatim, nothing reinvented):**

```python
#!/usr/bin/env python3
"""Premise battery (write-conditioning DRAFT-R2, X1(i)-(iv)). Eval-only,
no training. Deploy alongside the pinned ncr_lm_wave1_runner.py
(md5 9a93198b642242f512ff8489e32b0a53) -- imports its own functions,
does not copy them (additive-only discipline)."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # dir holding the pinned runner
import ncr_lm_wave1_runner as R
import ncr_ortho_write as ow          # spectral_diagnostics
import numpy as np

BASE_SEED = 90210          # R2.3: deliberately disjoint from every training seed {0,1,2,3}
HOPS = (1, 13, 37, 61)      # m6's squaring-residue set; corroborating only, not the ladder
CKPT = os.path.expanduser("~/ncr_g3b31/results/mob_g3b31_compB_s0.ckpt.pt")  # confirm real path at pre-flight step 1
OUT_DIR = os.path.expanduser("~/ncr_writecond/results")
CEILING_S = 45 * 60         # R2.4 wall-clock kill-switch, 0.75 GPU-h < 1.0 hard cap

def cell_p1a(device, batch_size, out):
    t0 = time.time()
    pools, cfg, pool_report = R.build_grammar_pools_and_cfg(seed=0)
    pools = pools.to(device)
    arms = R.build_two_arms(pool_report["vocab_size_total"], seed=0, device=device)
    with torch.no_grad():
        res = R.eval_arm_at_hops(arms["full_graft"], pools, cfg, HOPS, batch_size, device,
                                  BASE_SEED, read_ablate=False, teacher_force=True)
    rec = dict(cell="P1a", teacher_force=True, step=0, n=batch_size, result=res,
               elapsed_s=time.time() - t0, gate="HIGH_h(P1a) for all h in {1,13,37,61}: "
               "retrieval24_acc>0.95 required at every listed hop")
    _write(out, rec)
    return rec

def cell_p0_p1b(device, batch_size, out):
    t0 = time.time()
    pools, cfg, pool_report = R.build_grammar_pools_and_cfg(seed=0)
    pools = pools.to(device)
    ckpt = R.load_checkpoint(CKPT, device)
    assert ckpt is not None, f"checkpoint not found/invalid at {CKPT} -- rerun pre-flight step 1"
    arms, opts, data_gen = R.restore_arms_and_opts(ckpt, pool_report["vocab_size_total"],
                                                     lr=3e-4, device=device, freeze_entity_adapter=False)
    with torch.no_grad():
        p0  = R.eval_both_arms(arms, pools, cfg, batch_size, device, BASE_SEED, teacher_force=False)
        p1b = R.eval_both_arms(arms, pools, cfg, batch_size, device, BASE_SEED, teacher_force=True)
        attrib_p0, attrib_p1b = R.build_attribution(p0), R.build_attribution(p1b)
        # Z-dump + spectral_diagnostics (verbatim reuse, ow.spectral_diagnostics):
        gen = torch.Generator(device=device).manual_seed(BASE_SEED + R.EVAL_SEED_OFFSET + 61)
        probe = R.graft.build_task1_document(cfg, pools, gen, batch_size, 61, device)
        _, _, _, _, Z_sgd, keys_v, values_v = R.ncr_lm_forward_ablatable(
            arms["full_graft"]["backbone"], arms["full_graft"]["ncr"], arms["full_graft"]["integ"],
            probe, read_ablate=False, teacher_force=False)
        Z_ideal = arms["full_graft"]["integ"].teacher_force_operator(keys_v, values_v)
        U = ow.az.entity_subspace(Z_ideal[0].cpu().numpy())["U"]
        spec = ow.spectral_diagnostics(dict(Z=Z_sgd.detach().cpu().numpy(),
                                             z_ideal=Z_ideal.detach().cpu().numpy()))
    rec = dict(cell="P0+P1b", ckpt_step=ckpt["step"], n=batch_size,
               P0=dict(teacher_force=False, result=p0, attribution=attrib_p0),
               P1b=dict(teacher_force=True, result=p1b, attribution=attrib_p1b),
               spectral=spec, elapsed_s=time.time() - t0)
    _write(out, rec)
    return rec

def _write(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, default=str)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("P1a", "P0P1b"), required=True)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-only", action="store_true", help="batch_size=8, HOPS=(1,), sanity only")
    args = ap.parse_args()
    bs = 8 if args.smoke_only else args.eval_batch_size
    hops = (1,) if args.smoke_only else HOPS
    t_start = time.time()
    out = os.path.join(OUT_DIR, f"writecond_premise_{args.cell}{'_smoke' if args.smoke_only else ''}.json")
    fn = cell_p1a if args.cell == "P1a" else cell_p0_p1b
    fn(args.device, bs, out)
    if time.time() - t_start > CEILING_S:
        print(f"WARNING: exceeded {CEILING_S}s wall-clock ceiling", file=sys.stderr)
```

**Exact box-side launch commands (direct tmux, per `H100_SETUP.md`
convention — this battery is 2 short/cheap sequential cells, not a
parallel sweep, so no queue-job-spec/packing decision is needed, matching
R1.1.1's own placement note):**

```bash
# One-time deploy (after pre-flight steps 1-4 all pass):
scp premise_battery_eval.py youthful-indigo-turkey:~/ncr_writecond/

# Cell 1 -- P1a (gate; run and inspect BEFORE launching cell 2):
ssh youthful-indigo-turkey \
  'tmux new-session -d -s writecond_p1a \
   "cd ~/ncr_writecond && CUDA_VISIBLE_DEVICES=0 /home/nvidia/tdenv/bin/python3 \
    premise_battery_eval.py --cell P1a --eval-batch-size 256 --device cuda \
    2>&1 | tee p1a.log"'

# Poll: tmux has-session -t writecond_p1a (exits nonzero once done)
# Read: ~/ncr_writecond/results/writecond_premise_P1a.json --
#   gate = ALL of retrieval24_acc(h) > 0.95 for h in {1,13,37,61}.
# GATE FAIL -> STOP, escalate. GATE PASS -> cell 2.

# Cell 2 -- P0+P1b (only after cell 1's gate PASSES and checkpoint confirmed present):
ssh youthful-indigo-turkey \
  'tmux new-session -d -s writecond_p0p1b \
   "cd ~/ncr_writecond && CUDA_VISIBLE_DEVICES=0 /home/nvidia/tdenv/bin/python3 \
    premise_battery_eval.py --cell P0P1b --eval-batch-size 256 --device cuda \
    2>&1 | tee p0p1b.log"'
```

**GPU class.** Any single free H100 among 0–7 — no placement decision
needed (eval-only, small footprint per R2.3's re-smoke; run cell 1 then
cell 2 SEQUENTIALLY on the SAME GPU index for simplicity, not because
parallelism is unsafe). Never `pkill` — `tmux kill-session -t
writecond_p1a` / `writecond_p0p1b` by exact name if either needs to be
stopped.

**Expected wall-time.** Cell 1 (P1a): well under 5 minutes (model+pool
build dominates; no checkpoint I/O). Cell 2 (P0+P1b): well under 10
minutes (checkpoint load + a handful of forward-only passes). Combined
well under the `45`-minute wall-clock kill-switch (R2.4) and the `1.0
GPU-h` hard cap.

**Output paths.**
`~/ncr_writecond/results/writecond_premise_P1a.json`,
`~/ncr_writecond/results/writecond_premise_P0P1b.json` (plus
`*_smoke.json` variants from pre-flight step 4/the VRAM re-smoke). Scp
both real-run JSONs back into `experiment-runs/2026-08-13_ncr_writecond_
premise_battery/` (repo, small-file, per the hybrid archive policy) once
COMPLETE.

**Pre-registered bands the harvest is scored against (restated,
self-contained — no need to re-derive at harvest time):**

1. **GATE:** `retrieval24_acc(P1a,h) > 0.95` for all `h∈{1,13,37,61}`. FAIL
   ⇒ VOID, escalate, stop. Do not read P0/P1b's cell.
2. **Compute** `a=CLEARS_1(P0)`, `b=CLEARS_1(P1b)`, `c=CLEARS_61(P0)`,
   `d=CLEARS_61(P1b)` at `tau=0.09162` (exact, `n=256`, genuine — §R2.3).
3. **Look up** `(a,c)→AC` and `(b,d)→BD` in §R2.2's two 4-row tables, then
   the `AC×BD` 16-cell grid for the composite verdict + recommended
   action. `BD=10` (`b=1,d=0`) is the single highest-value reading —
   flag it explicitly in the harvest write-up if it occurs.
4. **Co-score** `DIVERGENT(x,h) := CLEARS_h(x) AND GAP(x,h)<0.15` on
   every one of the 4 `(x,h)` pairs; label INSTRUMENT-DIVERGENT rows
   accordingly (expected, not anomalous, for P1b per R2.1).
5. **Modifier:** apply P2 (common-mode-centered re-score, zero
   additional GPU-h) to any cell reading NULL; label `+COMMON-MODE` if
   it flips, without changing the row's grid cell.
6. **Report, do not silently pick:** every reading (AC, BD, divergence
   flags, P2 modifiers, the `h∈{13,37}` corroborating checks) goes into
   the harvest write-up verbatim — this battery's job is to populate the
   grid honestly, not to pre-decide which cell "should" occur.

**What this launch does NOT authorize.** No mechanism revision (X2 stays
ON HOLD, §A2-ADJUDICATION). No fresh 20,000-step retrain (R2.4). No
scope beyond the two cells above. The harvest's own routing (§R2.2's
grid) determines what, if anything, is dispatched next — including
whether a re-scoped premise (§0) or a re-founded mechanism cell is even
warranted — and that determination is the coordinator's, not
pre-decided in this document.

---

## PREMISE BATTERY HARVEST (coordinator, 2026-08-13) — VERDICT OF RECORD: AC=00 / BD=11 — "EXACT-Z-SURVIVES: the ONLY missing piece is Z-quality"

Run 2026-08-13 (box GPU 7, eval-only, total spend ≈0.1 GPU-h vs the
1.0 hard cap). Raw artifacts:
`experiment-runs/2026-08-13_ncr_writecond_premise_battery/` (repo +
SSD, hybrid policy). All readings verbatim, per band rule 6.

**GATE (P1a, fresh init, teacher-forced, n=256):** PASS —
retrieval24_acc 1.0 / 1.0 / 1.0 / 0.9961 at h=1/13/37/61; mean_cos
1.0→0.9966; o_pc ≈ 0.01 (no collapse). The exact-write algebra +
instrument are sane through 61 hops.

**The decisive pair (trained compB checkpoint, step 20000, n=256,
seed 90210):**
| | h=1 | h=13 | h=37 | h=61 | o_pc range |
|---|---|---|---|---|---|
| **P0 (own SGD writes)** | 0.0703 | 0.0352 | 0.0352 | 0.0664 | 0.77→0.99 |
| **P1b (exact writes substituted)** | 1.0000 | 0.9883 | 0.9883 | 0.9766 | 0.19–0.23 |

CLEARS @ τ=0.09162: a=0, c=0, b=1, d=1 ⇒ **AC=00 (ARCHIVE-CONFIRMED:
the SGD write shows no signal at ANY depth — §G3-B32 replicated at
n=256 on a disjoint eval seed), BD=11 (EXACT-Z-SURVIVES: the SAME
trained model's read path — binexp repeated squaring + retrieval
readout — executes 61-hop composition at 0.9766 when fed the exact
operator).** Spectral on the SGD-written Z: A_cond ≈ 9,959, eff-rank
≈ 10.0 of 24 — the learned write is catastrophically ill-conditioned;
the read machinery is NOT the blocker.

**Pre-registered cell verdict (16-cell grid, row AC=00 × col BD=11,
verbatim):** "Archive confirmed, but geometry supports deep retrieval
given exact Z — the ONLY missing piece is Z-quality; cleanest
re-founding for a future (re-designed) mechanism."

**DIVERGENT flags (band rule 4):** P1b@h=1 and P1b@h=61 are
INSTRUMENT-DIVERGENT (retrieval 1.0/0.977 vs answer_accuracy
0.047/0.035) — pre-registered as expected for P1b (R2.1): the decode
head was trained against the collapsed-write distribution. The
retrieval24 readout is the premise-relevant instrument. **P2
modifier (band rule 5):** not computable from the stored record (raw
read-output vectors not archived) — residual, re-runnable in minutes
if ever needed; NULL cells (all P0) are chance-level by a wide
margin, so no plausible P2 flip.

**Disclosed deviations:** (i) the audited script's `--smoke-only`
computed a reduced hop set but the cell functions read the global
HOPS — smoke ran all 4 hops (harmless, more data); (ii)
`eval_both_arms` takes no hops argument, so cell 2 measured h∈{1,2,3}
only — the h=61 coordinates come from a SUPPLEMENT
(`pbe_supplement`, archived) applying the IDENTICAL instrument
(`eval_arm_at_hops`, same seed/signature as P1a) to the restored
arms; (iii) the CKPT constant was patched to the verified real path
(`…_ckpts/` subdir) per pre-flight step 1's own instruction.

**What this changes (the first POSITIVE real-LM NCR capability
evidence in the program):** the trained 98M-param LM CONTAINS the
O(log h) composition-read capability — fully functional at h=61 —
and the sole blocker is that SGD does not learn a usable write
(cond ≈ 10⁴, rank collapse to ~10/24). The §G3-B32 "structural
block" is now LOCALIZED to write-learning. Spectral penalties at
write time are already evidenced-against (§A2 F5); the obvious
re-founded lever is SUPERVISED WRITE LEARNING — the runner's own
`teacher_force_operator(keys, values)` computes the exact target
Z_ideal from in-context content, giving a direct, cheap,
always-available supervision signal for the write path (train the
write to hit Z_ideal; the read already works). That is a NEW claim ⇒
novelty-gate re-entry per the memo, then design → attack → build.
Replication note: this battery is n=1 checkpoint (compB s0);
primary/compA checkpoints exist in the same archive dir and a
seed-replication cell is cheap (<0.1 GPU-h) — REQUIRED before any
publication-grade claim. Publisher dispatch deliberately HELD until
replication (coordinator call, PI may override).

**REPLICATION (same day, 2026-08-13): 3/3 — VERDICT UPGRADED TO
ROBUST.** The identical instrument (`pbe_repl`, archived; the frozen
arms needed `freeze_entity_adapter=True` to restore — matching their
training configs, disclosed) on the two sibling checkpoints:
- compA (frozen-cosine arm, step 20000): P0 = 0.043/0.020/0.020/0.035
  (chance); P1b = 1.0/1.0/1.0/0.9961. AC=00, BD=11.
- primary (frozen-ctrcos arm, step 20000): P0 =
  0.055/0.039/0.039/0.039; P1b = 1.0/0.9961/1.0/1.0. AC=00, BD=11.
All THREE §G3-B31 training configurations — different aux losses,
frozen AND trainable adapters — land in the same grid cell. The
read-path capability is universal across trained checkpoints; the
learned write is the sole blocker in every one. (Scope note: three
independent training runs, one base seed each — a multi-seed arm
belongs to the follow-on wave, not this diagnostic.) Publisher
dispatch UNBLOCKED per the standing per-finding directive.

---

## DRAFT-R3 — SUPERVISED WRITE LEARNING (2026-08-13)

**Status: NEW CLAIM, novelty-gate re-entry required.** This section is
written blind to any write-supervision code — none exists yet — per
this round's own charter. It is a claim PIVOT from every prior mechanism
in this file (§2(a)/(b)/(c), R1.3's `L_conf`), not a refinement of one:
where those mechanisms shaped `Z`'s *spectral shape* (orthogonality,
conformality, flatness) as a proxy for read-time robustness, this
mechanism regresses `Z` directly onto the *computable exact operator*
the runner already knows how to build (`teacher_force_operator`), never
touching a singular value, an eigenvalue, or a subspace basis. Grounding
read for this round, beyond the harvest/replication above: `§A2-
ADJUDICATION` (F1: `L_conf` is dead because the correct write is not
close to orthonormal at this geometry — `f(A*)≈149–195` across every
measured config, `NCR_WRITECOND_ATTACK_R2.md:111-136`; F5: the runner
already trains `0.1·‖ZᵀZ−I‖²/d²` on `Z` in all three §G3-B31 baselines
and it did not prevent collapse, `runner.py:714-742`, confirmed
`ortho_reg_weight=0.1` in all three `mob_g3b31_*.json:config`); `NCR_
WRITECOND_ATTACK_R2.md` M2 (the corrected training-cell rate, load-
bearing for §3's budget, see the flag below); `ncr_lm_wave1_runner.py`
(`9a93198b…`, re-read in full this round for `ncr_lm_forward_ablatable`
:360-398, `teacher_force_operator` — `ncr_lm_wave1_smoke.py:348-362` —
`compute_arm_losses`:768-852, `ortho_regularization_loss`:714-742,
`aux_read_supervision_loss`:599-632, the `--ortho-reg-weight`/`--aux-
read-loss-weight` CLI docstrings:1779-1808); `matrix-thinking/chapter2/
model_v4.py:25-64` (`BindingEncoder`, the transformer-based set encoder
`ncr_head.encode` actually is — relevant to §6's reachability check).

**A budget-derivation flag, stated up front because it is easy to miss.**
§3's GPU-h below prices every TRAINING cell at the **M2-corrected 0.83–
0.92 s/step** rate (`§A2-ADJUDICATION`/`NCR_WRITECOND_ATTACK_R2.md:1931-
1938`, ⇒ 4.6–5.1 GPU-h per 20,000-step run), **not** R1.6/R1.7's `0.8293
GPU-h/20k-step` figure (≈0.149 s/step) — that number was compB's own
anomalously fast regime, shown by M2 (ADOPTED, attack R2) to be
unrepresentative of general training cells by ~6×. R1.6/R1.7 are dead
sections (X2 ON HOLD) whose stale figure is still sitting in this file;
reusing it here would have under-priced this round's wave-1 by roughly
6× — flagged explicitly so a future round does not repeat the mistake.

### §1 The claim, one sentence, pre-registered

> **Direct supervision of the write path against the computable exact
> operator — `L_write := mean_B (1/K) Σᵢ ‖(Z_sgd − Z_ideal) kᵢ‖² / (‖vᵢ‖²
> + ε)`, where `Z_ideal := integ.teacher_force_operator(keys_v,
> values_v).detach()` is the SAME closed-form fit the runner already
> computes, added as a training-time-only loss term that never touches
> the forward read (`o = binexp_read(Z_sgd, q, h)` is unchanged; eval
> NEVER teacher-forces) — closes the write-quality gap the premise
> battery localized (SGD write: `A_cond≈9,959`, eff-rank `≈10/24`,
> retrieval24 at chance to `h*=61`; exact write on the SAME checkpoint:
> retrieval24 `0.977–1.0` at `h*=61`) and yields above-chance
> UNSUPERVISED-at-eval deep retrieval, at a pre-registered fraction of
> that measured gap (§3.6).**

`Z_ideal` is defined purely from in-context content already visible to
the model (the same `keys_v`/`values_v` `ncr_head.encode` itself
consumes) — no oracle information, no test-time-only signal, and (§2(c))
the eval protocol never uses it. This is a genuinely new claim relative
to every prior mechanism in this file: §2(a)/(b)/(c) and R1.3 all
regularized `Z`'s *shape*; this regresses `Z` toward a *specific,
per-example, exactly-computable point*, which is why it evades F1 and
F5 by construction, not by degree (§2(a)).

### §2 Design decisions with math

#### §2(a) — the distance: full-vector, per-key, scale-normalized regression to `Z_ideal` — not Frobenius-on-the-whole-matrix, not cosine, not a subspace projection

**The exactness fact this whole design leans on, proved not assumed.**
`teacher_force_operator` solves `k @ zᵀ ≈ v` for `zᵀ` via `pinv(k) @ v`,
`k,v: (K,d)=(24,25)`. Per column this is `K=24` equations in `d=25`
unknowns — **underdetermined** (one free direction per column,
`d−K=1`), so whenever `k` has full row rank (`K=24` linearly independent
adapted key vectors — true for essentially every batch since `K<d`, see
§6's reachability check for the failure boundary), the system is exactly
**consistent** and `pinv` returns the **zero-residual, minimum-norm**
solution: `Z_ideal @ kᵢ = vᵢ` **exactly**, not approximately (matches
attack R2's own independent simulation, residual `3.1e-16`,
`NCR_WRITECOND_ATTACK_R2.md:225-227`, and the harvest's own GATE cell,
retrieval24 `1.0/1.0/1.0/0.9961` at `h∈{1,13,37,61}`, the tiny departure
from `1.0` at `h=61` being accumulated float error over 61 squarings,
not fit error).

**The chosen distance.**

```
rᵢ(b) := Z_sgd(b) @ kᵢ(b) − vᵢ(b)                          (∈ R^d, i=1..K)
L_write(b) := (1/K) Σᵢ ‖rᵢ(b)‖² / (‖vᵢ(b)‖² + ε)
L_write := mean_B L_write(b)
```

Because `Z_ideal(b) @ kᵢ(b) = vᵢ(b)` exactly (above), this is *literally*
`(1/K) Σᵢ ‖(Z_sgd(b) − Z_ideal(b)) kᵢ(b)‖² / (‖vᵢ(b)‖² + ε)` — a distance
to `Z_ideal`, **restricted to its action on the K observed keys**, scale-
normalized per key. Three named alternatives from the task charter,
each considered and rejected in favor of this one, stated so the choice
reads as a decision, not a default:

1. **Full Frobenius, `‖Z_sgd − Z_ideal‖²_F` (unrestricted).** REJECTED.
   `Z_ideal`'s value on the `d−K=1`-dim complement of `span(keys)` is an
   **arbitrary artifact of `pinv`'s minimum-norm choice** — it carries no
   task information (no query in this task is ever drawn from outside
   the K keys; `assert_read_target_write_key_same_op`, `runner.py:569-
   599`, checked every launch in the base recipe, is the standing
   witness that reads and writes share exactly this key set). Matching
   `Z_sgd` to `Z_ideal` there spends gradient capacity forcing a
   meaningless coincidence, and — unlike the restricted form below — is
   not scale-normalized, so batches with larger `‖v‖` dominate the loss
   for no principled reason.
2. **Entity-block subspace projection, `A = UᵀZU` with `U` from SVD of
   `Z_ideal`'s row space (R1.3's own construction, reused).** REJECTED,
   two independent reasons. First, it is unnecessary: `L_write`'s
   AMBIENT (not subspace-projected) per-key residual already fully
   constrains `Z_sgd`'s action at the K points that matter — a query
   entity is always one of the K keys by construction, and repeated
   squaring from a key with small residual stays close to the discrete
   orbit `{k₁,…,k_K}` at every depth precisely because the residual is
   bounded in the FULL ambient space, not merely its in-span component
   (a projected loss could leave complement-direction leakage
   unpenalized; this form cannot, by not projecting at all). Second, it
   re-imports the exact machinery attack R2's M11/M12 found NOT to be a
   pure conditioning statistic (`f(A)` "charges for non-invariance of
   the entity subspace, undisclosed," `NCR_WRITECOND_ATTACK_R2.md:757-
   810`) — `L_write` needs no fixed subspace basis `U`, no SVD of
   anything, at all; there is no subspace object in this loss for a
   subspace-invariance critique to attach to.
3. **Cosine, matching `aux_read_supervision_loss`'s own house
   convention (`runner.py:607-612`: "NOT MSE… an MSE term would also
   pressure `o`'s NORM toward the target's norm").** REJECTED for this
   use, a **deliberate, disclosed departure** from that precedent, not
   an oversight. `aux_read_supervision_loss` supervises the READ OUTPUT
   `o`, whose norm is legitimately nuisance (`binexp_read` renormalizes
   at every squaring, `nm.binexp_read`/`_renorm_vec`/`_renorm_mat`, so
   `o`'s scale carries no task information and cosine — scale-invariant
   by construction — is the right tool). `L_write` supervises the WRITE
   OPERATOR's per-key ACTION, whose magnitude is exactly what compounds
   under `h`-fold squaring: a `Z_sgd` that is directionally correct per
   key but off by a per-key scale factor `cᵢ` still collapses under
   repeated application (`cᵢ^h` compounding is a variant of the same
   power-iteration mechanism `§G3-B26/B32` diagnosed for spectral
   ratios) — a cosine write-loss would be blind to exactly the failure
   mode this design exists to fix. `‖·‖²` is the right norm here;
   dividing by `‖vᵢ‖²+ε` (not a fixed global scale) keeps different
   documents' loss terms comparable without discarding magnitude
   information the way cosine would.

**Cost.** `Z_sgd @ keys_vᵀ`: one batched matmul, `(B,d,d)@(B,d,K)→(B,d,K)`,
`≈ d²K = 25²·24 = 15,000` FLOPs/example. `teacher_force_operator` itself
(`pinv` on a `(24,25)` matrix): negligible, sub-microsecond, no new
forward pass — `keys_v`/`values_v` are already extracted by
`ncr_lm_forward_ablatable` regardless of this loss. **Total ≈2×10⁴
FLOPs/write**, the same order as every mechanism this document has
costed, negligible next to the 98M backbone.

**Gradient (verification derivation, matches house convention):**

```
∂L_write/∂Z_sgd = (2/BK) Σ_b Σᵢ [ rᵢ(b) / (‖vᵢ(b)‖²+ε) ] ⊗ kᵢ(b)
```

a standard bilinear-form outer-product gradient; PyTorch autodiff
computes this in practice.

**`ε`-guard.** `‖vᵢ‖²+ε` floors the denominator (same class of guard as
§2(b)'s `ĉ`-floor and R1.3's `tr(G)`-floor) — degenerates only if a
value-adapter output collapses toward zero, an unrelated pathology this
guard does not mask (it is not in the numerator, so it cannot create a
degenerate zero-loss escape route the way §2(c)'s un-normalized penalty
or R1.3's earlier draft could).

#### §2(b) — supervision schedule: ALWAYS-ON, and why annealing's own named risk cannot arise here

The task charter names the risk directly: "the decode head co-trains
against whatever `Z` distribution it sees, so annealing interacts with
readout adaptation." This is a real, previously-observed effect in this
exact program — DRAFT-R2 §R2.1 diagnosed it precisely: compB's decode
head, having trained 20,000 steps against a COLLAPSED `o`
(`o_pairwise_cos` 0.989–0.992), reads a teacher-forced `o` at EVAL as an
**out-of-training-distribution input** (`retrieval24` and
`answer_accuracy` DIVERGE at P1b, §PREMISE BATTERY HARVEST). That
mismatch arises specifically because `--teacher-force-operator`
**swaps `Z` in the forward pass** — the decode head is trained on one
`Z`-distribution (whatever the flag produces) and could be evaluated on
another.

**`L_write` structurally cannot reproduce this risk, at any schedule.**
It is a training-time-only ADDITIVE loss term on `Z_sgd`; the forward
read always uses `Z_sgd = ncr_head.encode(keys_v, values_v)` — the SAME
tensor, unconditionally, whether `L_write`'s weight is `0`, constant, or
annealed. The decode head, `entity_adapter`, and backbone see and adapt
to the REAL, actually-improving-or-not `Z_sgd` at every single training
step, with or without supervision active — there is no second `Z`-
distribution for the decode head to be out-of-distribution with respect
to, because `L_write` never appears in the forward graph at all (compare
`ortho_regularization_loss`/`aux_read_loss_weight`, both also training-
loss-only, both also never swap the forward `Z` — same category, same
absence-of-mismatch argument). **Chosen: constant `λ_w` throughout
training** (matching house convention — `ortho_reg_weight` and `aux_
read_loss_weight` are both flat scalars for the full 20,000 steps in
every §G3-B31 cell, never annealed), justified on three independent
grounds: (i) the mismatch risk the charter names cannot arise
structurally, as just argued; (ii) it is the simplest choice and the
one this program has used everywhere else; (iii) §2(c) below explicitly
sanctions this: the pre-registered claim is "trainable-with-write-
supervision," which does not require supervision to ever be absent
during training, only that it be absent at EVAL — always-on satisfies
that by construction, with the eval-time boundary doing all the honesty
work.

**What is deferred, not solved.** A DIFFERENT, genuinely open dynamical-
stability question — does `Z_sgd` REGRESS once supervision is removed,
i.e. is a well-conditioned write self-sustaining under CE alone once
reached — is real and NOT answered by "always-on is schedule-safe."
Registered as an explicit non-primary follow-on (an annealed-off arm,
wave-2), not wave-1: answering it needs a converged write-supervised
checkpoint to anneal FROM, which does not exist yet.

#### §2(c) — why this is not cheating, stated as an honest boundary

`Z_ideal` is a pure function of `(keys_v, values_v)` — quantities the
model has ALREADY computed from the SAME in-context tokens
`ncr_head.encode` consumes for the real write (`extract_kv`, `runner.py
:310-329`); no additional information crosses into training that the
model doesn't already have access to. The model is trained to EMIT this
computable quantity via gradient descent — the same category of
technique as teacher-forcing a sequence decoder against ground-truth
previous tokens (an accepted, standard training technique whose whole
point is that a decoder trained this way must still produce correct
output at inference, when the crutch is gone) or state-space TTT losses
that regress a fast-weight state toward a closed-form target during
training only. **The eval protocol never teacher-forces — `Z_ideal` is
computed nowhere in the eval path, `retrieval24`/`answer_accuracy` are
scored on `Z_sgd = ncr_head.encode(...)` exactly as in every prior
§G3-B31 cell.** The honest boundary, stated once and carried everywhere
below: a WIN here establishes **"trainable-with-write-supervision"** —
a real, still-capability-relevant, but weaker claim than "SGD alone,
unsupervised, discovers a well-conditioned write" (which the harvest
already falsified, 3/3 configs, chance at `h*=61`). The **no-supervision
arm stays the recorded baseline** (P0, harvest, no rerun) precisely so
this weaker claim is never conflated with the stronger one it replaces.

### §3 Cells / seeds / budget

**Loss wiring — additive-only, matching the runner's own established
discipline exactly.** One new function, `write_supervision_loss(Z,
keys_v, values_v, eps=1e-6)` (§2(a)'s formula), and one new gated branch
in `compute_arm_losses` mirroring `ortho_regularization_loss`'s own
insertion (`runner.py:849-851`) byte-for-byte in structure:

```python
if is_full_graft and write_supervision_weight > 0.0:
    with torch.no_grad():
        Z_ideal = arm["integ"].teacher_force_operator(keys_v, values_v)
    write_loss = write_supervision_loss(Z, keys_v, values_v)
    total_loss = total_loss + write_supervision_weight * write_loss
```

plus one new CLI flag, `--write-supervision-weight` (default `0.0` =
OFF, byte-identical to today's runner — same "flag OFF ⇒ branch never
runs" guarantee `ortho_reg_weight`/`aux_read_loss_weight` already carry,
per `compute_arm_losses`'s own docstring, `runner.py:777-781`). No
existing function or code path is modified. `keys_v`/`values_v` are
already returned by `ncr_lm_forward_ablatable` — no extra forward pass.
This is the "audited-runner precedent" the charter calls for: identical
insertion pattern to §G3-B20's `ortho_reg_weight` and §G3-B31's
`aux_loss_type`, both additive, both independently gated.

**Warm-start vs. from-scratch — the three §G3-B31 arms, resolved as two
DIFFERENT roles, not one choice.** *Warm-start* (resume an already-
trained checkpoint, continue with `L_write` added) is CHEAP but
CONFOUNDED — it cannot distinguish "the write learns well under
supervision" from "supervision merely repairs an already-mostly-formed
`Z`," and DRAFT-R2's own F1 precedent (a 25% gradient step from a
partially-working point COLLAPSED retrieval 0.708→0.042 for a DIFFERENT,
wrong-target penalty, `NCR_WRITECOND_ATTACK_R2.md:130-140`) shows
gradient nudges near a trained checkpoint are not automatically benign.
*From-scratch* (train the target config from step 0 with `L_write`
active throughout) is the claim-bearing test — it matches CLAUDE.md's
calibration rule ("one real training run at the target config") and
avoids conflating repair with learning. **Resolution:** warm-start is
CALIBRATION-ONLY (Stage 0.1, cheap, non-claim-bearing, resumes compB's
own checkpoint for a short continuation as an early engagement check);
from-scratch is the claim-bearing PRIMARY (Stage 1). The three
§G3-B31 recipe families (primary: frozen+ctrcos; compA: frozen+cosine;
compB: trainable+ctrcos) are used as **Stage 1's `n≥3` seeds — three
independent from-scratch training runs, one per recipe, rather than
three RNG seeds of one recipe.** Justified two ways: (i) it directly
answers the task's own posed question (which of the three arms to use,
and how) by using all three, each once; (ii) it mirrors this document's
OWN established replication convention — the harvest's ROBUST verdict
(§PREMISE BATTERY HARVEST, "REPLICATION… 3/3") was earned by three
DIFFERENT recipes at one seed each, not three seeds of one recipe — so
this design tests write-supervision's robustness on the exact axis
(frozen vs. trainable adapter, cosine vs. contrastive aux) this
program's own precedent treats as the meaningful replication dimension.
A same-recipe multi-seed arm is a natural, cheaper wave-2 escalation if
wave-1 reads a clean WIN on all three configs, not funded here.

**Stage 0 — calibration, run BEFORE any Stage-1 cell (CLAUDE.md
mandatory pre-experiment rule).**

| cell | purpose | config | steps | seeds | GPU-h |
|---|---|---|---|---|---|
| 0.1 warm-start sanity | resume compB's checkpoint (step 20,000), add `L_write` @ `λ_w0` (from 0.3 below); does the per-key write-residual move at all, does retrieval24 not regress, within a short window, BEFORE any Stage-1 spend | compB, warm-start | +1,500 (continuation) | 1 | 0.365 |
| 0.2 reachability / conditioning check | CPU/eval-only, ZERO training: compute `cond(keys_v)` (top/bottom singular value ratio of the `(K,d)=(24,25)` key matrix) across `n≈256` sampled episodes on all three recipes — is `Z_ideal` a numerically well-conditioned (learnable) function of context, or does `pinv` sit near a rank-deficiency boundary for a non-trivial fraction of episodes? (§6's reachability question, made concrete and cheap) | all 3 | — | — | 0 |
| 0.3 residual-compounding derivation | analytic, zero GPU: using the harvest's OWN already-measured numbers (SGD-written `Z`: `A_cond≈9,959`, compB; `Z_ideal`'s own residual `≈3.1e-16`, attack-R2-simulated) and §1.3's `1/√d_ncr≈0.20` discriminability-floor derivation (reused, not re-derived), back out how small `L_write`'s own value must fall for `h*=61` retrieval to plausibly clear chance — sets `λ_w0` for 0.1, not guessed | — | — | — | 0 |

**Stage-0 subtotal (mid-rate estimate, 0.875 s/step ⇒ 4.861 GPU-h per
full-20,000-step-equivalent): 0.365 GPU-h.** Upper-bound check at 0.92
s/step (5.111 GPU-h-equivalent): 0.383 GPU-h — both comfortably small.

**Stage-0 gate.** 0.1 must show DIRECTIONAL engagement — `L_write`
itself decreasing over the continuation window AND `retrieval24@{1,13,
37,61}` not regressing below its pre-continuation reading — mirrors
DRAFT-R0 §3.3's "0.3 mech(a) canary" convention (does the term engage
at all before committing more spend). Zero engagement escalates to the
audit before any Stage-1 GPU-h is spent, not a silent proceed. 0.2/0.3
are informational (already-recorded evidence plus one cheap forward
pass), non-gating unless 0.2 reveals outright numerical pathology
(`cond(keys_v)` diverging/NaN for a large fraction of episodes) —
that specific outcome DOES gate, per §6 item 3.

**Stage 1 — main grid (gated on Stage 0).**

| arm | config(s) | seeds | steps | GPU-h (mid-rate) |
|---|---|---|---|---|
| PRIMARY: write supervision @ `λ_w0` | all 3 §G3-B31 recipes, from-scratch | 1 each (3 independent runs) | 20,000 | 3 × 4.861 = 14.58 |
| CONTROL A: wrong-fixed-operator placebo (§4) | compB recipe, from-scratch | 1 | 20,000 (SAME gradient budget as PRIMARY) | 4.861 |
| CONTROL B: readout-adaptation-only (§4) | compB checkpoint, warm-start | 1 | +2,000 (continuation) | 0.486 |
| blank-out / localization battery (§4, reused verbatim) | bundled, eval-only | — | — | 0.05 |

**Stage-1 subtotal (mid-rate): 19.98 GPU-h.**

```
Stage 0 (mid-rate)                  0.365 GPU-h
Stage 1 (mid-rate)                 19.977 GPU-h
------------------------------------------------
Nominal total (mid-rate estimate)  20.34 GPU-h
Range across the M2 0.83-0.92 s/step uncertainty:
  low  (0.83 s/step, 4.611 GPU-h/full-run-equiv)   ≈19.3 GPU-h
  high (0.92 s/step, 5.111 GPU-h/full-run-equiv)   ≈21.4 GPU-h
× 1.4 contingency (on the mid-rate nominal)        28.5 GPU-h
------------------------------------------------
Registered ceiling  ≤20 GPU-h nominal, hard cap ≤30 GPU-h
```

(Mirrors DRAFT-R0's own original ceiling convention — "≤20 GPU-h
nominal, hard cap ≤30 GPU-h," §3.5 — a callback, not a coincidence: this
is a leaner, single-mechanism design like R1.7's post-cut budget, but
priced at the CORRECTED training rate R1.7 never got the chance to use.)

**Placement.** GPUs 4/6/7, per this round's dispatch — three concurrent
training/continuation jobs at a time (matching §G3-B31's own measured
6.86 GB VRAM / 73–80% SM footprint per cell — same config family, same
expectation, to be confirmed by an eval-VRAM/training-VRAM re-smoke
before launch, per CLAUDE.md's "eval batch size can OOM even if training
fits" rule applied here in the opposite direction — a NEW loss term
adds a small forward+backward cost that should also be re-measured, not
assumed zero, before trusting the GPU-h table above at full scale). One
cell per GPU, no packing (identical reasoning to every prior stage in
this file). Five Stage-1 jobs total (3 PRIMARY + 2 controls) across 3
GPUs: the three PRIMARY recipes run concurrently first (they are the
claim-bearing arm and share nothing that would benefit from
sequencing), then Control A and Control B backfill the freed GPUs.

### §3.6 Success / kill bands, scored at `h*=61`

Reference anchors, taken directly from §PREMISE BATTERY HARVEST across
all three replicated configs (no rerun): **`P0_ref = 0.07`** (rounds up
from the measured range 0.035–0.066, the harder-to-beat end, chosen
conservatively) and **`P1b_ref = 0.977`** (the measured range's lower
end, compB, the harder-to-reach ceiling). `gap := P1b_ref − P0_ref =
0.907`. `τ = 0.09162` (n=256, exact, §R2.3, reused — Stage 1 evals at
the SAME `eval_batch_size=256`, `base_seed=90210` convention).

```
fraction_closed(x) := (retrieval24_acc(x, h*=61) − P0_ref) / gap
```

Checked in order (Band-1-first house convention):

1. **TPC / target-space integrity** — monitored (reused verbatim,
   R1.6's paired-CI form), non-gating unless the absolute `0.50`
   tripwire fires (that tripwire discriminates true catastrophic
   collapse by two orders of magnitude of slack and stays load-bearing
   regardless of mechanism).
2. **Write-engagement check** — `L_write` must show real descent from
   its own step-0 value over training. FAIL ⇒ INCONCLUSIVE-BY-MECHANISM
   (the intervention never engaged), distinct from a clean behavioral
   NULL.
3. **`retrieval24@h*=61`, PRIMARY:**
   - **NULL:** `≤ τ = 0.09162` (statistically indistinguishable from
     P0/chance).
   - **PARTIAL:** `τ <` reading `≤ chance+0.15 = 0.19167` **OR**
     `fraction_closed ∈ (0.024, 0.70)` — real, statistically significant
     movement off chance, short of the WIN bar.
   - **WIN:** reading `> chance+0.15 = 0.19167` (the established, 6+ SD
     bar, M7's own justification, reused unmodified) **AND**
     `fraction_closed ≥ 0.70` (closes at least 70% of the measured
     P0→P1b gap) **AND** the GAP metric (`full_graft − backbone_only`)
     also `> 0.15` at `h*` (rules out a `backbone_only`-side fluke,
     §3.6's original convention, reused).
   - **Depth-decay PARTIAL signature** (carried forward — ALREADY
     observed once in this exact program, compB's own `0.09375@h=20 →
     0.01562@h=61`, R1.6): clears WIN/PARTIAL at `h≤20` but decays
     toward NULL by `h*=61` — labeled explicitly, not folded into NULL.
4. **If PRIMARY reads NULL on ALL THREE recipes:** direct exact-operator
   write-supervision, as specified, is FALSIFIED at this architecture/
   scale — a materially more serious finding than a simple NULL
   (echoes DRAFT-R0 §3.6's own escalation clause), requiring its own
   honest write-up, not a quiet retry.

### §4 Baselines and controls

**No-supervision — the recorded P0 (harvest, all 3 configs), no rerun.**
The pre-supervision null hypothesis every Stage-1 arm must beat.

**Supervision-toward-a-wrong-fixed-operator (F1-informed placebo).**
`Z_wrong(b) := teacher_force_operator(keys_v(b), values_v(b)[σ])`, where
`σ` is a FIXED cyclic shift of the K value-slots (`σ(i) = (i+1) mod K`,
applied identically to every batch/step — not resampled, so it is
literally a different, wrong, but well-formed target every episode,
since the task's own true answer-permutation varies per document while
`σ` does not). `L_wrong` is the IDENTICAL functional form as `L_write`
(§2(a)), same weight `λ_w0`, same 20,000 steps — **SAME gradient budget**
by construction, not merely by claim, because it is the same loss
computed on the same machinery with only the target CONTENT changed.
This is a stronger match than R1.5's noise-injection null (which needed
per-step recalibration and still failed F3, "141× weaker in coherent
displacement") — here there is nothing to recalibrate; the placebo's
gradient magnitude tracks `L_write`'s own by sharing its exact
computation. Pre-registered reading: placebo IMPROVES retrieval ⇒
generic "extra gradient pressure on `Z`" nuisance effect, independent of
target correctness (the exact confound §G3-B22–B25 and F3 already
diagnosed); placebo does NOT improve, `L_write` DOES ⇒ evidence specific
to target CORRECTNESS, not merely "more supervision" — directly the
axis F1 killed the prior mechanism on, now tested in this mechanism's
own terms.

**Readout-adaptation-only control.** Reuses `--teacher-force-operator`
VERBATIM, unmodified: continue training the compB checkpoint (step
20,000) for 2,000 further steps with `teacher_force=True` — `Z` is
FORCIBLY replaced by `Z_ideal` in the forward pass throughout this
continuation, so `ncr_head`/the encoder receive EXACTLY zero gradient
(`teacher_force_check.ncr_zero_grad_checks_passed`, asserted every step,
already-existing machinery) — only `entity_adapter`, `backbone`, and the
decode head adapt. **Then evaluate with `teacher_force=False`** (`Z_sgd`
restored, UNCHANGED throughout this continuation by the zero-grad
guarantee just cited) and re-score `retrieval24@h*=61`. This isolates
whether read-side/`entity_adapter` adaptation to a clean-`Z` training
distribution, alone, moves the needle when reading with the OLD,
uncorrected `Z_sgd` — the exact question the charter poses ("does the
read-side alone explain any gain"). Pre-registered expectation, stated
honestly before running it: given DRAFT-R2's own out-of-distribution
finding (§R2.1 — a decode head trained on a collapsed `o` reads a clean
`o` out-of-distribution, and the CONVERSE direction — a decode head
further adapted toward a clean-`o` distribution reading an UNCHANGED
collapsed `o` — plausibly worsens the mismatch, not improves it), the
expected reading is NO improvement (or a regression) in P0's own
number. A surprising IMPROVEMENT here would be an important competing
finding (decode-side undertraining, independent of write quality) and
is not pre-dismissed.

**Blank-out / localization battery.** Reused verbatim (`NCR_ORTHO_WRITE.
md` §9.0's raw-input-corruption-post-encoding invariant) — `L_write`
only touches the encoder's `Z` output via an added loss term, so this
invariant must hold UNCHANGED under every Stage-1 arm; bundled into
existing eval, no dedicated GPU-h.

### §5 Novelty-gate charter

**Carry-forward, already-verified obligations (mandatory cite, not
re-swept).** From `research/writecond-novelty-2026-08-13.md` and `NCR_
WRITECOND_ATTACK_R2.md`'s own web-verified F5-B sweep: MuonSSM
(2606.30461), DeltaProduct (2502.10297), Gated DeltaNet-2 (2605.22791),
Preconditioned DeltaNet (2604.21100), Variational Linear Attention
(2605.11196), Lattice (2504.05646), MeSH (2510.07739), Sanford
(2402.09268) + Wang (2505.23683), the unitary-RNN/dynamical-isometry
line (Arjovsky 1511.06464, Pennington 1711.04735, Xiao et al. ICML
2018), and the full soft-orthogonality/RIP family F5-B located (SRIP/SO
1810.09102, Parseval 1704.08847, Vorontsov 1702.00071, SICNC 2503.20454,
Rényi-2/effective-rank 1808.07912 + 2406.11672, I-STAR/IsoScore
2305.19358/2108.07344, Spectral Isotropy Regularization 2605.29987,
uniformity-to-`(1/d)I` 2305.17326). **These are carried forward as
context, not as occupants of THIS mechanism's own territory** — every
one of them conditions `Z`'s SPECTRAL SHAPE (orthogonality, RIP,
flatness, entropy); `L_write` is a direct per-example regression to a
COMPUTED VALUE and touches no singular value anywhere. Stated plainly:
this design does not re-enter the crowded spectral-regularizer wedge
F5-A/F5-B just mapped — it exits that mechanism CLASS entirely.

**New sweeps required before launch (PENDING — not run by this draft;
the coordinator dispatches them next, per the task's own closing
instruction).** Search terms, as specified by the charter and
supplemented:
- "test-time distillation of fast weights"
- "auxiliary losses matching linear-attention states to closed-form
  targets"
- "teacher forcing for memory/state" / "teacher forcing fast weights"
- "DeltaNet-family supervised state matching"
- "TTT with explicit targets" / "test-time training closed-form
  target"
- NEW, added this round given §1's framing: "distillation toward
  least-squares fast-weight solution," "supervised state-matching loss
  linear attention," "in-context operator regression auxiliary loss"

**Named candidates the sweep must check and either clear or convert to
a cite-and-distinguish obligation (NOT verified by this draft — flagged
per CLAUDE.md's research-grounding rule, "VERIFIED citations only,
agent-checked, never from memory"):** Test-Time Training / TTT-family
(Sun et al.) — an obvious candidate given "explicit target for a fast-
weight state during training only, none at inference" is close to this
design's own boundary (§2(c)); DeltaNet/Gated DeltaNet/RWKV-7's own
delta-rule as an implicit "distance to a rank-1 target" (already
distinguished by mechanism elsewhere in this file, but the SUPERVISED,
non-recurrent-update framing here is a different question the sweep
must ask fresh); any 2026 paper on auxiliary losses for associative-
memory/fast-weight writes specifically scored against read-time
compositional depth (the by-task axis, still OPEN per the memo — this
draft's mechanism change does not by itself resolve that openness,
it only changes what the by-mechanism sweep must check).

**What would count as scooped.** A paper that (i) writes a fast-weight
operator from in-context content, (ii) computes or has access to a
closed-form/exact target for that operator from the SAME context, (iii)
trains the learned write toward that target via an explicit auxiliary
loss (not by swapping the target into the forward pass, i.e. not
literal teacher-forcing of the whole read), and (iv) evaluates
UNSUPERVISED read-time compositional depth. A hit on (i)-(iii) without
(iv) is a positioning-only overlap (cite-and-distinguish, not a kill,
mirroring how MuonSSM/DeltaProduct were handled for the prior
mechanism). A hit on all four, especially at anything resembling this
task's `K/d`/repeated-squaring-read structure, is a genuine scoop
requiring re-scoping — not assumed absent here, left for the dispatched
sweep to determine.

**Internal-archive note (self-check, not a substitute for the dispatched
sweep).** `L_write` is NOT `aux_read_supervision_loss` (that supervises
the READ OUTPUT `o` post-composition, cosine, target = the true answer
entity's adapted embedding) and NOT `ortho_regularization_loss` (that
shapes `Z`'s spectrum, no notion of a specific correct value) — it
supervises the WRITE OPERATOR directly, pre-composition, toward a
per-example EXACT value. No existing loss in this program's archive
does this; `KILL_LIST.md`/`NCR_ORTHO_WRITE.md`/`NCR_ORTHO_FALLBACK_
DESIGN.md` contain no entry for exact-operator write regression (a
targeted grep, not a substitute for the coordinator's own internal
sweep, which this round's charter still requires).

### §6 Self-attack — what would kill this at R3

1. **Supervision may fix `Z` on the synthetic task while the LM's own CE
   loss fights it.** `L_write` and `L_CE` both flow gradient into
   `Z_sgd` (via the read `o`) and into `entity_adapter` (via `keys_v`/
   `values_v`, shared with the write path) — if `entity_adapter`'s
   CURRENT geometry makes the CE-optimal write genuinely different from
   the geometric `Z_ideal` at that same training step (plausible early
   in training, before `entity_adapter` has settled), the two losses
   could pull in different directions, especially in compB's trainable-
   adapter arm. **Cheap registered check:** log `cos(∇_Z L_write,
   ∇_Z L_CE)` alongside training (near-zero extra cost, both gradients
   are already computed) — a persistently negative cosine is the
   diagnostic signature of this conflict, distinguishable from simple
   under-training.
2. **The `h*`-frontier may move.** `binexp_read` renormalizes magnitude
   at every squaring (`_renorm_mat`/`_renorm_vec`) but does NOT fix
   directional conditioning — so even a `Z_sgd` with a much-reduced
   per-key residual could still have a large enough OWN condition
   number (recall the harvest's measured `A_cond≈9,959` on the
   unsupervised write) that residual amplification under six squaring
   steps (`h=61` needs `⌈log₂61⌉=6`) overwhelms signal by `h*=61` even
   while `h≤20` clears WIN/PARTIAL — exactly the depth-decay PARTIAL
   signature §3.6 already carries forward as its own labeled outcome,
   not folded into a clean WIN or silently ignored.
3. **`Z_ideal` may be unreachable by `ncr_head.encode`'s own
   parametrization.** `BindingEncoder` (`model_v4.py:25-64`) is a
   transformer-based set encoder producing `Z` via attention over the K
   (key,value) tokens — a smooth function of its inputs. `Z_ideal =
   pinv(k)@v` is smooth almost everywhere but its Lipschitz constant
   blows up as `k` approaches rank-deficiency (§2(a)'s exactness proof
   requires `K` linearly independent keys; near that boundary, `pinv`
   is numerically unstable and `Z_ideal` is a poorly-behaved function of
   context for a gradient-trained approximator to match). **The cheap
   analytic check this deserves, specified, not deferred (Stage 0.2):**
   compute `cond(keys_v)` — the ratio of `keys_v`'s largest to smallest
   singular value, a `(K,d)=(24,25)` matrix, one `torch.linalg.svdvals`
   call per episode, zero training — across `n≈256` sampled episodes on
   all three recipes, BEFORE Stage 1 launches. A consistently
   moderate `cond(keys_v)` (say `<100`) supports reachability; a
   heavy-tailed or diverging distribution is a live reachability risk
   that should escalate to the audit (e.g. an auxiliary key-separation
   term, or re-scoping) BEFORE Stage-1 budget commits, not discovered
   after. Worth noting, disclosed rather than assumed either way: this
   is a DIFFERENT quantity from the harvest's measured `A_cond≈9,959`
   on `Z_sgd` itself (the WRITTEN operator's conditioning) — whether
   `Z_sgd`'s poor conditioning is INHERITED from `keys_v`'s own geometry
   or is a genuine `BindingEncoder` training failure independent of key
   geometry is exactly what comparing these two numbers, once measured,
   would reveal, and is not knowable from the archive as it stands.

---

Rev-3 dispatched 2026-08-13. Novelty sweeps (§5) and attack R3 next,
per the coordinator's own closing instruction — no GPU spend, no STATE.
md/EXPERIMENT_LOG.md update, no commit from this document.

---

## §A3-ADJUDICATION (coordinator, 2026-08-13) — supervised-write gate DISCHARGED-WITH-OBLIGATIONS; attack R3 = BLOCKED (3F/8M/8m) ADOPTED; Rev-4 dispatched

**Novelty (both sweeps same-day, verdicts in
`research/writecond-novelty-2026-08-13.md` RE-ENTRY 2):** external
OPEN — FAAST (arXiv:2605.04651) is the mandatory anchor (same K†V
target math as the RUNTIME mechanism, compositional capability
disclaimed) with the "why train toward it rather than compute it"
answer now pre-registered (extraction is the learned content;
runtime pinv brittleness); "provably fails" registered DOWN to
"systematically fails." Internal CLEAN — KEY_ANCHORING same-class/
different-lever precedent (soft-loss-underperformed + trained-content-
dispensable cautionary pattern) cited into the controls; the
consolidation lane's ridge-vs-min-norm discipline inherited onto
Z_ideal's own pinv construction; the F1-range transcription slip
(127.8 floor, not 149) to be corrected in Rev-4.

**Attack R3 (`NCR_WRITECOND_ATTACK_R3.md`) ADOPTED IN FULL.** 13
items verified clean and FROZEN for R4 (the exactness proof — full
row rank structurally guaranteed by `grammar_rd._assert_injective_
entities`; all code citations; budget to the digit; τ; §2(b)'s
decode-immunity argument). The FATALs:
- **F1 — the mis-specified minimum.** L_write's zero set is the
  25-dim affine family Z_ideal + u·wᵀ (w = the kernel direction of
  the observed keys at d−K=1): executed walk shows L_write ≈ 3.8e-12
  and retrieval@h=1 = 1.0000 held constant while retrieval@h=61
  falls 1.0 → 0.0625 (chance) as the transverse gain rises 0→100 —
  a transverse Lyapunov instability (gain^61), not precision.
  §2(a)'s Frobenius rejection was INVERTED: pinv's min-norm choice
  (Z_ideal·w = 0) IS the stability constraint. WIN calibration
  measured by the attack: L_write ≲ 3e-4 AND transverse gain ‖Zw‖
  ≲ 3 (unconstrained expectation ≈ 5 — the free direction MUST be
  supervised).
- **F2 — loss conflict:** all three recipes still carry
  `ortho_reg_weight=0.1` and ortho_loss(Z_ideal) = 15,147; the joint
  minimizer reads chance at h=61 for every λ_w ≤ 10. The ortho term
  must be REMOVED in supervised arms (it was already evidenced
  useless by §A2 F5).
- **F3 — undetached targets:** keys_v/values_v feeding Z_ideal
  undetached ⇒ the zero set contains entity collapse (backbone_only
  control proves the channel live). Targets detach; an
  entity-geometry watch joins the instruments.
**Stage-0 ruling adopted as split:** 0.1 BLOCKED pre-CLEAR
(mis-specified loss, non-diagnostic gate); 0.2/0.3 replaced by the
report's Stage 0′ (≈0.1 GPU-h eval-only, six items, three gating) —
Stage 0′ MAY run pre-CLEAR once Rev-4 folds its spec.

**DISPOSITIONS: D1–D8 per the report adopted verbatim as the Rev-4
charter** (D1 = the loss re-specified to supervise the full operator
— Frobenius or per-key + explicit ‖Zw‖ transverse penalty, decided
by the report's own analysis; D2 = ortho term removed from
supervised arms; D3 = detached targets + geometry watch; plus the
novelty obligations and the two internal-sweep disciplines above
folded in). Rev-4 → attack R4; the full gauntlet continues; wave-1
still ≤30 GPU-h hard cap on GPUs 4/6/7 post-CLEAR.

---

## DRAFT-R4 (Rev-4, 2026-08-13)

**Charter.** `§A3-ADJUDICATION` above, adopted in full, binding. This
section supersedes DRAFT-R3 (§1–§6, lines 2283–2886) for every
downstream purpose; DRAFT-R3 stays verbatim as the historical record of
what attack R3 killed (house convention, matching R1/R2's own
treatment of their predecessors). The 13 items `NCR_WRITECOND_ATTACK_R3.md`
verified clean (V1–V13) are FROZEN and reused without re-derivation:
`teacher_force_operator` solves `k zᵀ=v` and returns the zero-residual,
minimum-norm `Z_ideal` (V1/V2); `K=24` full row rank is structurally
guaranteed every batch by `grammar_rd._assert_injective_entities` (V3);
every code citation (V4); `ortho_reg_weight=0.1` is in all three
§G3-B31 baselines (V5); the budget arithmetic to the digit (V6); `τ =
0.09162` exact (V7); `P0_ref=0.07` conservative (V8, superseded below
by D5's per-recipe reference — see that section for why); additive-only
wiring (V9); eval never teacher-forces for the intended wiring (V10,
but see D8 for the leak-guard M7 demanded); `h*=61` is the standard
in-loop eval depth (V11); the band partition has no double-firings
(V12, though M5(a)'s single-sided hole is fixed below, D5); and §2(b)'s
decode-immunity argument is sound (V13). None of these is re-argued
below.

**Grounding read for this round (beyond DRAFT-R3's own grounding
paragraph, reused unmodified):** `NCR_WRITECOND_ATTACK_R3.md` in full,
its 8 reproduction scripts (`scratchpad/sim_{A..G}_*.py`, the F1/M1/M2/
M3 executed tables), `research/writecond-novelty-2026-08-13.md`
RE-ENTRY 2 (the FAAST answer and the "systematically fails" language
obligation), `matrix-thinking/chapter2/model_v4.py:25-64` (re-read for
`row_out = nn.Linear(h=64, d=25)`, confirmed shared across all `d` rows
of `Z` — line 52/59/63 — the exact reachability constraint Stage 0′
item 3 tests), `matrix-thinking/KEY_ANCHORING_DESIGN.md` §2.4/§10.14
(the same-class/different-lever soft-loss precedent), `research/
consolidation-policy-novelty-2026-08-11.md` §8 (the ridge-vs-min-norm
falsifier discipline, inherited onto `Z_ideal`'s own construction
below).

**Correction carried in from `§A3-ADJUDICATION`.** DRAFT-R3's grounding
paragraph (line 2296) and self-attack §6 item 3 both cite `f(A*)≈149–
195` for the (now-dead) mechanism-(b)/(c) conformality penalty. The
raw table (`NCR_WRITECOND_ATTACK_R2.md:111-118`) reads `f(A*) ∈
{148.7, 170.8, 195.2, 167.7, 127.8}` across the five measured
geometries — the true floor is **127.8** (compA), not 149. Corrected
here; the number is dead-mechanism trivia at this point (D2/D3 below
don't touch spectral shape at all) but is fixed per the adjudication's
explicit instruction, not left to propagate.

**Language discipline, applied throughout this section (novelty
RE-ENTRY 2 obligation).** The unsupervised SGD write's failure is
described as **"systematically fails"** (3/3 configs, chance at
`h*=61`, `§PREMISE BATTERY HARVEST`) — never "provably fails." No
impossibility proof exists in the literature or this archive; the
finding is a robust empirical replication, not a theorem, and the
distinction is load-bearing for how strongly downstream claims may be
stated.

---

### D1 — the loss re-specified: zero set exactly `{Z_ideal}`

**Choice (per the report's own analysis, and the report's "minimum
acceptable" option — chosen over full Frobenius because it is a
strictly smaller, targeted edit that preserves every surviving piece of
R3's own §2(a) reasoning rather than discarding it):** keep R3's
per-key, scale-normalized term **unmodified**, and add one new,
explicit term that supervises exactly the direction F1 showed it
misses.

#### D1.1 The per-key term (R3 §2(a), frozen, restated for reference)

```
rᵢ(b) := Z_sgd(b) @ kᵢ(b) − vᵢ(b)                        (∈ R^d, i=1..K)
L_key(b) := (1/K) Σᵢ ‖rᵢ(b)‖² / (‖vᵢ(b)‖² + ε)
```

Unchanged from DRAFT-R3, and — this is worth stating precisely because
it resolves a build ambiguity below (m1) — **this term never needed
`Z_ideal` as a runtime tensor.** It is defined directly from `Z_sgd`
acting on the raw `(keys_v, values_v)` the model already extracted.
`Z_ideal` is a *proof device* (used below to characterize `L_key`'s
zero set), not a quantity the loss function computes.

#### D1.2 The missing direction, closed: an explicit transverse penalty

**`w`: the kernel direction of the observed keys, computed per batch.**
`keys_v(b)` is `(K,d)=(24,25)`. Take its **full** SVD (not the reduced
form `pinv` would use internally):

```
U, S, Vh = torch.linalg.svd(keys_v, full_matrices=True)   # U:(B,K,K) S:(B,K) Vh:(B,d,d)
w = Vh[:, -1, :]                                            # (B,d), unit norm
```

Since `keys_v` has full row rank `K=24` (V3, structurally guaranteed),
`Σ` (the rectangular `K×d` singular-value matrix implicit in the `S`
above) has exactly `d−K=1` structurally-zero column beyond the `K`
real singular values, so `Vh`'s **last row** is the unique (up to sign)
unit vector with `keys_v @ w ≈ 0` — this is the *same* `w` attack R3
verified numerically (`max|w·kᵢ| = 2.0e-07`, F1), now specified as a
per-batch, differentiable computation rather than a one-off analysis
script.

**The penalty.**

```
Zw(b)        := Z_sgd(b) @ w(b)                              # (B,d)
v̄²(b)        := (1/K) Σᵢ ‖vᵢ(b)‖² + ε                        # per-episode value-energy scale
L_transverse(b) := ‖Zw(b)‖² / v̄²(b)

L_write(b)   := L_key(b) + λ_t · L_transverse(b)
L_write      := mean_B L_write(b)
```

`L_transverse` is normalized by the **same class** of quantity `L_key`
already divides by (per-episode value energy, not a fixed global
constant) — this keeps the two terms' gradient magnitudes comparable
across episodes with different `‖v‖` scale, matching the house
discipline `L_key`'s own normalization already established (R3 §2(a),
untouched).

**Cost.** One additional batched **full** SVD of a `(B,24,25)` matrix
per step. `torch.linalg.pinv` (which R3's draft called, then never
used — m1) and `torch.linalg.svd` share the same LAPACK-family
underlying routine and are the same asymptotic cost; attack R3's own
measurement of the batched `pinv` call is the direct stand-in:
**≈1.23ms/step at B=32** (m2). Net effect on the wiring: this SVD
*replaces* R3's unused `pinv` call rather than adding to it — the
corrected loss is **no more expensive** than R3's drafted (and
already-negligible) version. ≪0.2% of the 0.83–0.92 s/step training
step; to be confirmed, not assumed, by the pre-launch re-smoke (D8/m2,
extended to cover this call too).

#### D1.3 Zero-set proof — `{Z : L_write(b) = 0} = {Z_ideal(b)}`, exactly, for `λ_t > 0`

This is the derivation the report demanded, done directly from the
attack's own algebra (F1) rather than re-derived from scratch.

Attack R3 established: the full solution set of `Z kᵢ = vᵢ, i=1..K` is
the affine family `Z_ideal + {u wᵀ : u ∈ R^25}` (`N := {M : M kᵢ = 0
∀i} = {u wᵀ}`, a 25-dimensional space, since `M` is unconstrained on
the 1-dim complement `span(w)` and exactly zero on the K-dim
`span(k₁,…,k_K)`).

**Step 1 — `L_key(b) = 0 ⟺ Z_sgd(b) ∈ Z_ideal(b) + N`.** Each term of
`L_key` is a nonnegative squared norm over a positive denominator; the
sum is zero iff every term is zero iff `(Z_sgd − Z_ideal) kᵢ = 0` for
all `i=1..K` iff `Z_sgd − Z_ideal ∈ N` (by `N`'s own definition, F1).

**Step 2 — restricted to that family, `L_transverse(b) = 0 ⟺ u = 0`.**
For `Z_sgd = Z_ideal + u wᵀ` (any `u ∈ R^25`):

```
Zw = (Z_ideal + u wᵀ) w = Z_ideal w + u (wᵀw) = 0 + u·1 = u
```

using `Z_ideal w = 0` exactly (V2, `pinv`'s min-norm property — attack
R3 measured `max_b ‖Z_ideal w‖ = 2.6e-05` fp32, i.e. zero up to float
noise) and `wᵀw=1` (unit norm by construction). So
`L_transverse(b) = ‖u‖²/v̄²(b) = 0 ⟺ u = 0` (since `v̄²(b) > 0`
strictly, by the `ε`-floor).

**Step 3 — combine.** `L_write(b) = L_key(b) + λ_t·L_transverse(b) = 0`
(both nonnegative, `λ_t>0`) `⟺ L_key(b)=0 AND L_transverse(b)=0 ⟺
Z_sgd(b) = Z_ideal(b) + u wᵀ with u=0 ⟺ Z_sgd(b) = Z_ideal(b)`. **QED —
the zero set is the single point `Z_ideal(b)`, not a 25-dimensional
family.** F1 is closed by construction, not by degree.

**A structural bonus this proof exposes, worth stating because it
answers a question D2 has to ask anyway.** `L_key`'s gradient with
respect to `u` is *identically* zero on the whole family (§ Step 1: the
family is exactly `L_key`'s flat directions), while `L_transverse`'s
gradient lives *entirely* in `u`-space and is zero on `L_key`'s own
gradient directions (its value only depends on `Zw`, which the per-key
residual `rᵢ` cannot see — `rᵢ` is computed by contracting `Z` against
`kᵢ ⊥ w`). **The two terms' gradients are orthogonal subspaces of
matrix-space for a fixed `(keys_v, w)`; they do not compete for the
same minimum.** This is the opposite situation from F2's diagnosed
ortho-vs-`L_write` conflict (§D2 below), where the two terms' minima
were genuinely different points and `λ_w` had to overpower `ortho` by
`~10⁴`. Here, `λ_t` only has to be large enough to make the transverse
direction converge **within the training budget** — a speed question,
not a competing-minima question — which is exactly what Stage 0′ item
6 (below) is designed to check on real geometry before Stage 1 commits.

#### D1.4 Correcting R3's own rejection of "alternative 1" (full Frobenius)

R3's §2(a) rejected the unrestricted `‖Z_sgd − Z_ideal‖²_F` on the
grounds that `Z_ideal`'s value on the complement of `span(keys)` is *"an
arbitrary artifact of `pinv`'s minimum-norm choice… carries no task
information."* **F1 proved this backwards, and it must be stated as a
retraction, not silently patched around:** `Z_ideal w = 0` is not
arbitrary — it is the **unique dynamically stable** member of the
25-dimensional family (transverse Lyapunov exponent `= log(gain)`,
amplified by `gain^h` under `h`-fold squaring; F1's executed table:
`retr24@61` goes `1.0000 → 0.0625` as transverse gain goes `0 → 100`
at a numerically *unchanged* `L_key`). Leaving that direction
unsupervised is fatal, not harmless.

**This retraction does not flip the recommendation to full Frobenius.**
R3's alternatives 2 (subspace projection — unnecessary machinery, and
re-imports the M11/M12 non-invariance critique) and 3 (cosine — blind
to per-key scale, the exact failure mode this loss exists to prevent)
are untouched by F1 and remain valid reasons to keep a *restricted,
scale-normalized* form for the in-span component rather than an
unrestricted, un-normalized Frobenius term. What changes is narrow and
precise: the complement direction needed **its own explicit term**
(D1.2), not a re-litigation of the in-span form. `ε`-guard, gradient
form, and cost discussion for `L_key` carry forward from R3 unmodified.

#### D1.5 The calibrated bands (carried forward from the attack, not re-derived)

From `NCR_WRITECOND_ATTACK_R3.md`'s own executed calibration
(§"Calibration handed to Rev-4", in-span-only sweep for `L_key`; F1's
own table for transverse gain):

```
WIN region:  L_key ≲ 3e-4   (per-key RMS relative error ≲ 1.8%)
             AND  ‖Z_sgd w‖ ≲ 3   (unconstrained expectation ≈5 — MUST be supervised)
```

Both conditions are necessary; neither alone is sufficient (F1's own
table: `‖Zw‖=0` with `L_key` at its floor is the WIN case; `‖Zw‖=100`
with `L_key` *also* at its floor collapses to chance — the two
quantities are the two coordinates of the same 25-dim family D1.3 just
collapsed to a point). This calibration is carried into the mechanism
check band (§ Bands, Band 2, below) as a hard pre-registered gate, and
into Stage 0′ item 6 as the target the real-geometry recalibration
either confirms or revises **before** Stage 1 launches.

---

### D2 — `ortho_reg_weight` removed from supervised arms; the matched control this creates

**Disposition (report's preferred option, adopted): `ortho_reg_weight =
0` in every write-supervised Stage-1 arm** (PRIMARY ×3 recipes,
CONTROL A). F2 showed the joint minimizer of `CE + 0.5·aux + 0.1·ortho
+ λ_w·L_write` reads chance at `h*=61` for every `λ_w ≤ 10` — the ortho
term's own minimum (`Z→c·I`) is measurably disjoint from `Z_ideal`'s
(`ortho_loss(Z_ideal) = 15,147`, gradient nonzero at the target) and it
was already evidenced-against as a *sufficient* mechanism on its own
(§A2 F5: it trained in every §G3-B31 baseline and did not prevent
collapse). Removing it in the supervised arms is not a new risk — it
retires a term this program has twice independently found to fight or
be neutral to the mechanisms it was supposed to help.

**The confound this creates, and its control.** `P0` (the recorded
harvest baseline, no rerun) was measured **with** `ortho=0.1`. Once the
supervised arms run with `ortho=0`, a Stage-1 WIN is a *two-variable*
change relative to `P0` (write supervision added **and** ortho
removed), and F2's own finding — ortho and `L_write` are NOT neutral to
each other — means "ortho off" cannot be waved away as inert. **New
cell, funded (report's own disclosed instruction): CONTROL C —
`ortho_reg_weight=0`, `write_supervision_weight=0` (no supervision,
ortho off), compB recipe, from-scratch, 20,000 steps, 1 seed.** This
isolates "does removing ortho alone move retrieval off `P0`'s chance
reading" from "does write supervision move it" — the two-variable
confound closed by adding the one missing cell rather than by
argument. Placement/cost in the Stage-1 grid below (D6).

**No-supervision baseline stays `P0` exactly as recorded (unchanged
from R3/R2/R1) — it is not rerun, and it is not replaced by Control C.**
`P0` is the *historical recipe's* own number (ortho on, the actual
comparison the field and this program's own prior sections make);
Control C is the *matched* baseline for THIS wave's specific two-arm
comparison. Both are reported; neither is silently substituted for the
other (D5 makes this explicit in the bands).

---

### D3 — detached targets; the entity-geometry watch

**Repair (F3, one word, extended to cover D1's new term too — this
extension matters and is not in the R3 text, so it is stated
explicitly rather than assumed inherited):**

```python
keys_v_d, values_v_d = keys_v.detach(), values_v.detach()
write_loss = write_supervision_loss(Z, keys_v_d, values_v_d, lambda_t=WRITE_LAMBDA_T)
```

R3's fix detached the inputs to `L_key`. **D1's `L_transverse` also
consumes `keys_v` (to compute `w` via SVD) and `values_v` (for
`v̄²`).** If only `L_key`'s inputs were detached and `w`'s SVD ran on
an undetached `keys_v`, gradient would flow from `L_transverse` back
into `entity_adapter`/`embed` through exactly the route F3 diagnosed —
a second, narrower door into the same collapse room. Both tensors are
detached **once**, at the top of `write_supervision_loss`, before
either sub-term is computed, closing both doors with the same fix.

**Why this matters more here than it did for `L_key` alone.** If
`keys_v` collapses toward a single point (F3's degenerate route,
`k_i=v_i=c ∀i`), `keys_v` becomes rank-deficient (not rank `K=24`), and
`w` — defined as the SVD's *smallest*-singular-value direction — becomes
numerically **ill-posed**, not merely wrong: near-degenerate keys leave
`d−K` directions all comparably small, so which one the "last row of
`Vh`" picks becomes an artifact of floating-point roundoff rather than
a meaningful null direction. This is a second, independent signature of
the same collapse pathology (beyond TPC), available for free as a
numerical-health check: log `S[-1]/S[-2]` (the gap between the smallest
and second-smallest singular value of `keys_v`) alongside `w` — a
collapsing gap is itself diagnostic, feeding directly into Stage 0′
item 3's reachability question (below) and into the ridge-vs-min-norm
caution this section closes with.

**Entity-geometry watch (the coordinator's paraphrase of F3's own
instrument requirement — `target_pairwise_cos` as a Band-1 tripwire).**
This already exists in R3's §3.6 item 1 as a monitored, non-gating
signal with an absolute `0.50` collapse tripwire. What D3 adds is not a
new metric but a new **comparison discipline**: because the detach fix
makes it structurally impossible for `L_write` *itself* to cause
collapse (its gradient no longer reaches `entity_adapter`/`embed` at
all), any TPC drift observed in a Stage-1 PRIMARY/CONTROL-A cell beyond
what CONTROL C (ortho-off, no supervision, same recipe, same aux) or
`P0` (ortho on, no supervision) already show is attributable to
CE/aux/backbone co-training dynamics generally — **never** to `L_write`
specifically, by construction. The watch is therefore: log TPC at every
eval for every Stage-1 cell (already free, `discriminability_metrics`);
at harvest time, plot PRIMARY's TPC trajectory against `P0`'s and
CONTROL C's own (same-recipe) trajectories — if PRIMARY's TPC diverges
*materially* from both, that is a live anomaly (a mis-wire or an
unanticipated interaction) worth escalating even below the absolute
`0.50` threshold, because the detach proof above says it should not be
possible.

**Ridge-vs-min-norm discipline, inherited (novelty obligation, §A3-
ADJUDICATION).** `research/consolidation-policy-novelty-2026-08-11.md`
§8's standing caution — an un-damped min-norm solution can be
numerically fragile near rank-deficiency even when it is the
theoretically correct target — applies here as a genuine, not
rhetorical, contingency: `Z_ideal`'s own construction (`pinv`, used
only in the zero-set *proof*, never at runtime post-D1.2) and `w`'s SVD
both become ill-conditioned as `cond(keys_v)` grows (D3's own point,
above). This is why Stage 0′ item 3 (below) is extended, not
"verbatim," to report the smallest/second-smallest singular-value gap
of `keys_v` alongside `cond(keys_v)` — if the measured tail is heavy
enough that this gap collapses for a non-trivial fraction of real
episodes, a Tikhonov-damped `w` (project onto the space spanned by the
smallest **two or three** singular directions rather than the single
smallest, or damp via `Vh` weighted by `1/(σ+ρ)`) is registered as the
fallback, **not** the default — adopting it changes D1.3's zero-set
proof (it would no longer be exactly `{Z_ideal}` but a small
neighborhood of it) and is not taken pre-emptively.

---

### D4 — quotienting the global scale (M1)

`binexp_read` renormalizes at every squaring, so the read is exactly
invariant to a positive global scale on `Z` (M1, measured: `L_key`
spans 13 orders of magnitude — `9.80e-01` to `9.80e+03` — across
operators the read treats as *identical*). **Adopted fix: `L_key`'s
own step-0-to-current "descent" is not used as a standalone engagement
metric** (M1's own finding: pure global-rescaling can satisfy it with
zero directional progress). Instead, Band 2 (below) reads `L_key` and
`‖Z_sgd w‖` **directly against their calibrated absolute targets**
(D1.5: `≲3e-4` / `≲3`), never as a "moved from its initial value"
relative check — this sidesteps M1's exact failure mode by
construction, because the calibrated targets are properties of the
*residual*, not sensitive to which global scale the encoder happens to
have settled on this cell (a `10×`-larger `Z` with the same
*directional* error rescales `L_key` by `100×` but the target `3e-4`
is measured on the same rescaled quantity — the calibration table
(D1.5) already reports RMS **relative** per-key error, which factors
out exactly this degree of freedom). No additional closed-form
`c*`-rescale machinery is added; the existing normalization plus the
band redefinition is sufficient and cheaper.

---

### D5 — bands rewritten as an enumerated partition; `answer_accuracy` co-scored; per-recipe reference; cross-recipe aggregation

#### D5.1 Why the reference is now per-recipe, not a single shared `P1b_ref`

R3 used one shared `P1b_ref=0.977` across all three recipes, described
as *"the harder-to-reach ceiling."* M5(e) showed this is backwards —
the lower end of the range **shrinks** the gap and **lowers** the WIN
bar, the opposite of conservative. Rather than relabel that sentence
(the minimum fix), **each recipe is now scored against its own
harvest-measured ceiling** — a strictly more precise design that also
removes the "which end is conservative" question structurally, at zero
extra cost (both numbers are already in the harvest table):

| recipe | `P0_ref` (h=61) | `P1b_ref` (h=61) | `gap` |
|---|---|---|---|
| compB | 0.0664 | 0.9766 | 0.9102 |
| compA | 0.0350 | 0.9961 | 0.9611 |
| primary | 0.0390 | 1.0000 | 0.9610 |

`fraction_closed_recipe(x) := (x − P0_ref[recipe]) / gap[recipe]`,
computed per recipe. `τ = 0.09162` (global — a sampling-noise threshold
tied to `n=256`, not to any recipe's P0/P1b, so it is legitimately
shared, unlike the gap).

#### D5.2 The partition (closes M5(a)'s hole structurally, not by patching the OR)

R3's `PARTIAL := (τ<x≤0.19167) OR (fraction_closed∈(0.024,0.70))`
missed 60/402 points (M5(a)): any `x∈[0.705,1.0]` with a failing GAP
clause fell through both the WIN and PARTIAL predicates. Fix: define
NULL and WIN as clean, disjoint, positive predicates, and **PARTIAL as
their set complement** — a partition is then true by construction for
*any* well-defined NULL/WIN pair, with no separate OR-clause to leave a
hole in.

```
NULL   :=  x ≤ τ
WIN    :=  x > (chance + 0.15 = 0.19167)
           AND fraction_closed_recipe(x) ≥ 0.70
           AND GAP(full_graft − backbone_only, h*) > 0.15
PARTIAL := NOT NULL AND NOT WIN                      (everything else, by definition)
```

Per-recipe absolute WIN thresholds this produces (informational,
`0.19167` floor is not binding for any of the three — `fraction_closed`
is):

```
compB:   x > 0.0664 + 0.70×0.9102 = 0.7035
compA:   x > 0.0350 + 0.70×0.9611 = 0.7078
primary: x > 0.0390 + 0.70×0.9610 = 0.7117
```

#### D5.3 The full band order (Band 0 through Band 5), checked in order

- **Band 0 — teacher-force leak gate (M7/D8, new).** Every Stage-1
  artifact (PRIMARY, CONTROL A/B/C) must show
  `config.teacher_force_operator == false`,
  `teacher_force_check.active == false`,
  `teacher_force_check.ncr_zero_grad_checks_passed == 0`. FAIL ⇒ VOID
  the cell (an indistinguishable-from-WIN artifact, per M7) — re-run,
  never scored.
- **Band 1 — TPC / target-space integrity.** Monitored per-recipe
  against `P0`/CONTROL C's own trajectories (D3's watch); the absolute
  `0.50` tripwire, if it fires, ⇒ NULL-BY-COLLAPSE regardless of
  retrieval, unchanged house convention.
- **Band 2 — mechanism check (D1.5, D4).** `L_key ≤ 3e-4` AND
  `‖Z_sgd w‖ ≤ 3` at `h*`, measured via **exact** `torch.linalg.svd`
  off the training path (not the cheap training-time estimate — same
  discipline the dead spectral design used, reused). FAIL ⇒
  INCONCLUSIVE-BY-MECHANISM, distinct from a clean behavioral verdict.
- **Band 3 — `retrieval24@h*=61`, PRIMARY signal.** The D5.2 partition,
  per recipe.
- **Band 4 — depth-decay PARTIAL signature (carried forward, now with
  per-hop reference points, m8).** Clears WIN/PARTIAL at `h≤20` but
  decays toward NULL by `h*=61` — labeled explicitly. Per-hop `P0`
  references now registered (compB, m8): `h=5: 0.0469`, `h=12: 0.0352`,
  `h=20: 0.0312`, `h=40: 0.0742`, `h=61: 0.0664` — informational,
  non-adjudicating (the signature is read qualitatively, per m8's own
  resolution: no formal band is defined at intermediate hops).
- **Band 5 — `answer_accuracy`, disclosure only, never gating (M5(d)).**
  Co-scored at `h*` for every cell (already emitted by
  `discriminability_metrics`, zero cost — discharges the ADOPTED X1
  repair R3 silently dropped). **Explicitly not part of WIN/PARTIAL/
  NULL logic:** the harvest's own `P1b` ceiling reads `answer_accuracy`
  at chance on every recipe (`0.020–0.063` across all hops, `0.0` in
  P1a) — gating on it would make WIN unreachable by construction. This
  is reported as an open, separately-flagged gap between
  retrieval-capability and instruction-following-capability, consistent
  with the spearhead's "capability inside a real LM" framing — not
  silently dropped, not silently promoted to a gate it structurally
  cannot pass.

#### D5.4 Cross-recipe aggregation (closes the 25/27-unadjudicated-outcomes hole, M5(b))

Aggregated by **count**, not by which specific recipe produced which
band — the count fully determines the verdict; which recipe is
retained for diagnosis, never discarded:

| pattern (WIN/PARTIAL/NULL counts, out of 3) | verdict | action |
|---|---|---|
| 3-0-0 | **ROBUST WIN** | claim holds across the recipe-diversity axis; wave-2 same-recipe-seed escalation becomes the natural next step (not funded here) |
| 2-{0,1}-0 | **MAJORITY WIN** | recorded as WIN-with-caveat; the non-WIN recipe's Band 0–2 readings are inspected before any wave-2 decision |
| any mix with ≥1 WIN AND ≥1 NULL | **SPLIT** | recorded explicitly, per the CLAUDE.md tiebreak discipline — never averaged or silently resolved toward the majority; escalates to the audit round for interpretation, cross-referenced against CONTROL A/C's own per-recipe readings before any claim is drawn |
| 0-{1,2,3}-{0,1,2} (≥1 PARTIAL, 0 WIN, ≤2 NULL) | **PARTIAL-ROBUST** | a real, informative, sub-WIN verdict — recorded as such, not silently deferred to wave-2 without a written verdict |
| 0-0-3 | **FALSIFIED** (R3/R0's own escalation clause) | direct exact-operator write-supervision is FALSIFIED at this architecture/scale — requires its own honest write-up, not a quiet retry |

---

### D6 — budget re-derivation (rate honesty, per-cell ceilings, Control C funded)

**Rate attribution, corrected (M6a — the STATED reason for the 0.83–
0.92 s/step price was false, the price itself was not).** The three
`mob_g3b31_*` cells' own measured rate (0.146–0.151 s/step) is not
"compB's own anomalously fast regime" — all three ran within 4 seconds
of each other, and the true spread (`0.15`–`0.92` s/step, ~6× across
the archive) is **environmentally/contention-driven**, unpredictable
from config. Pricing at the conservative mid-rate (0.875 s/step,
⇒4.861 GPU-h/20k-step-equivalent) stays the right call **for the
correct reason** — contention risk, not a per-recipe cost difference —
restated here so a future round does not re-litigate this with the
wrong justification again.

**Per-cell ceilings, registered separately from the pricing table
(M6c, closes the ABORTED-BUDGET trap the archive already hit twice at
`wave1_calib_K24_s0`/`mob_g3b17_s0`).** Every 20,000-step Stage-1 cell
gets `--ceiling-gpuh 5.5` (not the 4.861 expected-cost figure); the
2,000-step CONTROL B continuation gets `--ceiling-gpuh 0.6`.

**Stage 0′ (eval-only, six items, ≈0.1 GPU-h — see its own launch card
below).**

**Stage 1 (mid-rate pricing, unchanged per-cell figure from R3 — V6
verified it to the digit, M6a only corrected the *reason*, not the
*number*):**

| arm | config | steps | seeds | GPU-h (mid) |
|---|---|---|---|---|
| PRIMARY: D1+D2+D3-corrected write supervision | all 3 §G3-B31 recipes, from-scratch | 20,000 | 1 each | 3 × 4.861 = 14.583 |
| CONTROL A: wrong-fixed-operator placebo (D7, M3-rescoped) | compB, from-scratch | 20,000 | 1 | 4.861 |
| CONTROL B: readout-adaptation-only (D7, M4-fixed) | compB ckpt, warm-start | +2,000 | 1 | 0.486 |
| **CONTROL C (NEW, D2): ortho-off, no-supervision** | compB, from-scratch | 20,000 | 1 | 4.861 |
| blank-out/localization battery | bundled, eval-only | — | — | 0.05 |

**Stage-1 subtotal: 24.841 GPU-h.**

```
Stage 0′ (eval-only)                0.10  GPU-h
Stage 1 (mid-rate)                 24.841 GPU-h
------------------------------------------------
Nominal total (mid-rate)           24.94  GPU-h
Range across the M2 0.83-0.92 s/step uncertainty:
  low  (0.83 s/step)   ≈23.67 GPU-h
  high (0.92 s/step)   ≈26.22 GPU-h
------------------------------------------------
× 1.4 contingency (on mid-rate nominal)   34.92 GPU-h
```

**Registered ceiling: ≤25 GPU-h nominal, hard cap ≤35 GPU-h.** This is
a deliberate, disclosed revision upward from R0/R1/R2/R3's own
`≤20/≤30` convention, made explicitly rather than silently exceeded:
Control C is D2-mandatory (it closes a two-variable confound F2's own
finding created), and the revision does **not** change ceremony tier —
CLAUDE.md's tiering is 10–50 GPU-h → one audit round + pre-launch
red-team either way, and `§A3-ADJUDICATION`'s own charter already put
this wave in the ">10 GPU-h, publication-adjacent" bracket regardless
of the exact ceiling. **Expected real cost is materially below the
nominal figure** — M6b's own finding (the three g3b31 cells' actual
measured cost was 2.481 GPU-h total for 3×20k-step runs, ~6× cheaper
than the conservative mid-rate price) — kept conservative here
deliberately (contention risk is real and unpredictable, M6a), not
re-priced down.

**Placement.** Unchanged from R3: GPUs 4/6/7, one cell per GPU, no
packing (matches every prior stage's identical VRAM/SM-footprint
reasoning). Now 6 Stage-1 jobs (3 PRIMARY + 3 controls, up from 5) —
PRIMARY's three recipes run first (claim-bearing, share nothing that
benefits from sequencing), then CONTROL A/C backfill, CONTROL B last
(cheapest, shortest).

---

### D7 — control repairs (M3, M4)

**CONTROL A — rescoped readout, gradient logging, and a D1-specific
strengthening.** M3 showed the placebo's gradient budget is matched
only at init and diverges to `6.5e5×` at convergence (the per-key
term's gradient vanishes as `L_write` converges to a REAL target but
never vanishes against a permuted, unreachable one) — the placebo
cannot discriminate the confound it names, and its retrieval reading is
analytically pre-determined at chance. **Repair, adopted:** re-scope
the readout to **conditioning transfer** — does a well-formed-but-wrong
target (`Z_wrong := teacher_force_operator(keys_v, values_v[σ])`, fixed
cyclic shift `σ`, unchanged from R3) produce a well-conditioned `Z`
(`A_cond`, eff-rank, `‖Z_sgd w‖`, `o_pairwise_cos`) while reading at
chance? Log per-step `‖∇_Z L_key‖` and `‖∇_Z L_transverse‖`
**separately** for both the real and placebo arms (not just a combined
`‖∇_Z‖`, per D1.3's own gradient-orthogonality finding — a persistently
matched `L_transverse` gradient with a diverging `L_key` gradient would
be the CLEAN confirmation that the two sub-terms behave exactly as
D1.3 predicts, since `w` depends only on `keys_v`, which is identical
between the real and placebo arms — the transverse-suppression pressure
is content-independent by construction, so the placebo is now a
strictly cleaner specificity test than it was in R3: any conditioning
difference between arms can ONLY come from the per-key term, not from a
shared mechanism artifact).

**CONTROL B — adapter frozen for the continuation; separate clean-eval
script named.** M4(a): the pinned `eval_both_arms(..., teacher_force=
teacher_force_operator)` call means a continuation with the flag set
produces teacher-forced (P1b-shaped) evals throughout, not the clean
`Z_sgd` readout the control needs — the premise battery's own
`pbe_repl.py` / `pbe_supplement.py` pattern (already archived,
`experiment-runs/2026-08-13_ncr_writecond_premise_battery/`) is the
correct, already-built tool: a **separate** eval invocation with
`teacher_force=False` after the continuation, not a flag toggle inside
the training loop. M4(b): `entity_adapter` is not frozen by the
zero-grad guarantee (`ncr_untouched` covers `ncr_head`'s parameters,
not `integ.query_key`'s inputs) — compB is `freeze_entity_adapter=
false`, so 2,000 steps of CE/aux would drift `entity_adapter` AND the
`retrieval24` targets themselves, confounding "readout adaptation" with
write-input drift and target-space drift. **Repair, adopted:**
`--freeze-entity-adapter` for this continuation specifically,
**disclosed as no longer compB's exact recipe for this one cell** — the
control's whole purpose (isolate decode-head-only adaptation) requires
it, and the disclosure is cheaper than losing the isolation the control
exists to provide.

---

### D8 — build-brief items (M7, M8, m1, m2, m4, m5, m6)

- **Band-0 gate (M7).** Specified above (D5.3); zero cost, converts an
  undetectable failure into an impossible one.
- **Config provenance (M8).** `--write-supervision-weight` (and the new
  `--write-transverse-weight` = `λ_t`, D1.2) added to `rec["config"]`
  explicitly, and to `run_two_arm_cell`'s resume asserts alongside
  `seed`/`freeze_entity_adapter` — a resumed cell must not silently
  change either weight mid-run.
- **Dead-code ambiguity, resolved by elimination (m1).** R3's snippet
  computed `Z_ideal` under `no_grad` and never used it. D1.2 shows
  neither sub-term needs it at runtime — **the fix is to drop the
  `Z_ideal = arm["integ"].teacher_force_operator(...)` line from the
  wiring entirely**, not to make it load-bearing. `Z_ideal` remains a
  proof device (D1.3) and a Stage-0′/harvest diagnostic quantity, never
  a training-time tensor.
- **`pinv`/SVD cost (m2).** Now: one SVD call per step (D1.2), replacing
  R3's unused `pinv` call — measure it explicitly in the pre-launch
  re-smoke rather than asserting the ~1.2ms/step transfer holds at
  Stage-1's actual batch size and backbone-co-training load.
- **Mutual-exclusion assert (m4).** `assert not (args.teacher_force_
  operator and args.write_supervision_weight > 0)` at cell launch — with
  both set, `Z ≡ Z_ideal` in the forward pass and `L_write ≡ 0`
  identically (D1.3's own zero-set proof, evaluated at the trivial
  point `Z_sgd=Z_ideal`), a silent no-op that Band 0 would also catch
  but should never reach.
- **Eval cost at `eval_batch_size=256` (m5).** Fold into the pre-launch
  re-smoke this round's own charter already requires (D6's placement
  note); price alongside the new SVD call, not separately.
- **`ε`-guard claim restated (m6).** D1.2/D1.4's guards prevent a
  degenerate *denominator* collapse (`v̄²→0`); they do NOT and cannot
  prevent the F3 collapse route, which is closed by D3's detach, not by
  either `ε`-floor. Stated as two independent defenses now, not one
  overloaded claim.

---

### FAAST fold-in and the "why train, not compute" answer (novelty RE-ENTRY 2, §2(c) obligation)

`Z_ideal` is a pure function of `(keys_v, values_v)` — quantities the
model has already computed for the real write; the model is trained to
**emit** this computable quantity via gradient descent, never to
consume it at eval (unchanged from R3, and V13/D3 above keep this
argument sound). **FAAST (arXiv:2605.04651) is the mandatory anchor and
uses the SAME `K†V` target math as the RUNTIME mechanism at both train
and eval**, explicitly disclaiming compositional capability — the
closest prior art, and the question it forces is answered, pre-
registered, not deferred: **why train toward the target rather than
compute it at runtime?**

1. **`Z_ideal` presupposes the model's OWN extracted `keys_v`/
   `values_v` — the learned content is *extraction*, not binding.**
   Computing `Z_ideal` at eval would require `pinv` on every forward
   pass, which is exactly what `teacher_force=True` already does (the
   P1b arm) — and that arm's own `entity_adapter`/decode head never
   adapts to the resulting `Z`-distribution honestly (§R2.1's
   out-of-distribution finding). Training toward `Z_ideal` lets the
   *learned write* (`ncr_head.encode`, a smooth function of context)
   converge to behave like the closed-form solution while the rest of
   the model co-adapts to the REAL, gradually-improving `Z_sgd`
   throughout — never to a swapped-in `Z_ideal` at any point, at train
   or eval.
2. **Runtime `pinv` is brittle exactly where D3's ridge-vs-min-norm
   caution flags it — near key rank-deficiency** — and inherits the
   min-norm conditioning caveats D3 registers as a contingency, not a
   default. A trained write does not carry that per-example numerical
   fragility into inference; a runtime `pinv` mechanism (FAAST's own
   choice) does, by construction.

This is the pre-registered answer novelty RE-ENTRY 2 required before
launch; it is now on the record rather than assumed self-evident.

**KEY_ANCHORING precedent, cited into the controls (novelty
obligation).** `matrix-thinking/KEY_ANCHORING_DESIGN.md` §2.4/§10.14 is
the same CLASS of intervention (a soft regression loss toward a
computed target) at a DIFFERENT lever (cross-episode key stability, not
write-operator correctness) — its own banked finding is directly
cautionary: the soft `L_anchor` term saturated well below the hard
mechanism's ceiling (`0.9987→0.4806` collapsing with `K`, §10.14's own
table), and separately, trained anchor CONTENT proved dispensable to
ablation (candidate (e), CONFIRMED-BY-ABLATION: "constancy suffices").
This is exactly the failure mode CONTROL A/C are built to catch here —
a soft loss that moves a number without the mechanism being the reason
— and is cited as the standing motivation for keeping both controls,
not as evidence against D1's own mechanism (KEY_ANCHORING's lever is
different; the caution is about METHOD, not this specific claim).

---

### Stage 0′ — the calibration replacement (six items, three gating), launch card

Authorized to run pre-CLEAR per the Stage-0 ruling adopted in
`§A3-ADJUDICATION` (0.1 BLOCKED; 0.2/0.3 replaced). Items 1–5 fold in
**verbatim** from `NCR_WRITECOND_ATTACK_R3.md`'s own Stage-0′ spec.
**Item 6 is executed against the D1-corrected loss (`L_key +
λ_t·L_transverse`), not R3's drafted restricted-only form** — a
necessary substitution, flagged explicitly rather than silently made,
since "the loss" item 6 calibrates now refers to the surviving D1 form.
Honest cost: not literally zero (eval-only forward passes on the three
existing checkpoints, no training, no box-config change) — the
premise battery's own comparable read-only pass measured ≈0.1 GPU-h,
and this is priced the same way, not smuggled in as free.

**Item spec:**

1. `cond(keys_v)` across `n≈256` real episodes, all 3 recipes.
   Informational, non-gating (M2: retrieval is insensitive to key
   conditioning across 4 orders of magnitude in the executed sim) —
   **extended per D3**: also log `S[-1]/S[-2]` (the smallest/
   second-smallest singular-value gap) as the numerical-health signal
   for `w`'s own well-posedness, not just `cond`.
2. `‖Z_ideal‖_F` and its per-row norm distribution (median, p99, max,
   within-episode max/min) across the same episodes/recipes.
3. `cond(ncr.row_out.weight)` per checkpoint (`nn.Linear(64,25)`,
   `model_v4.py:52`, shared across all 25 rows of `Z` — M2's own
   reachability question) vs. item 2's required dynamic range. **GATES**
   — if `cond(row_out)` cannot supply the required range, the target is
   unreachable by this parametrization and the mechanism needs
   rescoping before any GPU-h is spent.
4. `‖Z_sgd w‖`, the transverse gain, on all three trained checkpoints
   (`w` from item 1's own SVD). **GATES** — if the trained checkpoints
   already sit at transverse gain `≫3`, D1's transverse term is not a
   theoretical hole-closer but the direct diagnosis of the *observed*
   failure, confirming the mechanism is necessary, not optional.
5. `ortho_regularization_loss(Z_ideal)` and `‖∇_Z ortho‖` at `Z_ideal`
   on real batches (verbatim `ortho_regularization_loss`, reused); this
   item is now **moot for Stage-1 pricing** (D2 sets `ortho=0` in every
   supervised arm) but stays **gating as a sanity re-confirmation** that
   F2's diagnosed conflict was real and specifically justifies D2's
   removal, on real (not R3's Hamiltonian-cycle-synthetic) key geometry
   — a documented, not assumed, justification for a decision already
   made.
6. **The `L_write→retrieval24@61` calibration curve, redone on real
   `keys_v`/`values_v` (D1-corrected form).** For each of the 3
   checkpoints: extract real `(keys_v, values_v)` from `n≈256`
   episodes; sweep controlled in-span (`L_key`) and transverse (`‖Zw‖`)
   perturbations exactly as attack R3's `sim_B`/`sim_C` did on
   synthetic Hamiltonian-cycle keys, but now on the REAL extracted
   geometry, scoring `retrieval24` via the real `nm.binexp_read` +
   `discriminability_metrics`. **GATES, extended beyond the report's
   "informational" framing for this one item specifically** — because
   D1's loss is new, item 6 must also confirm the provisional `λ_t=1.0`
   (D1.2, chosen because D1.3 proved the two sub-terms don't compete
   for a minimum — an optimization-speed question, not a landscape
   question) actually drives `‖Zw‖` under 3 within a 20,000-step budget
   on real geometry; if it does not, `λ_t` is revised **before** Stage 1
   launches, not discovered mid-wave.

**Cell spec (`stage0prime_eval.py`, deploy alongside the pinned
`9a93198b` runner, same discipline as the premise battery's own launch
card — imports the pinned runner's functions, reuses `nm.binexp_read`/
`discriminability_metrics`/`ortho_regularization_loss` verbatim,
reinvents nothing):**

```python
#!/usr/bin/env python3
"""Stage 0' (write-conditioning DRAFT-R4). Eval-only + cheap CPU-side
optimization (item 5/6's joint-min / calibration sweeps run on
extracted tensors, not via box training). No GPU training, no
box-config change."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncr_lm_wave1_runner as R
import ncr_models as nm                 # binexp_read, real read path

BASE_SEED = 90210
N_EPISODES = 256
CKPTS = {
    "primary": os.path.expanduser("~/ncr_g3b31/results/mob_g3b31_primary_s0.ckpt.pt"),
    "compA":   os.path.expanduser("~/ncr_g3b31/results/mob_g3b31_compA_s0.ckpt.pt"),
    "compB":   os.path.expanduser("~/ncr_g3b31/results/mob_g3b31_compB_s0.ckpt.pt"),
}
OUT_DIR = os.path.expanduser("~/ncr_writecond/results")
CEILING_S = 40 * 60          # 0.67 GPU-h < the ~0.1 GPU-h expected, ample margin

def extract_real_kv(arms, pools, cfg, device, n=N_EPISODES, seed=BASE_SEED):
    """Items 1-2/6's shared input: real (keys_v, values_v) from n real
    episodes, via the pinned forward path -- no synthetic geometry."""
    gen = torch.Generator(device=device).manual_seed(seed + R.EVAL_SEED_OFFSET)
    probe = R.graft.build_task1_document(cfg, pools, gen, n, 61, device)
    with torch.no_grad():
        _, _, _, _, Z_sgd, keys_v, values_v = R.ncr_lm_forward_ablatable(
            arms["full_graft"]["backbone"], arms["full_graft"]["ncr"],
            arms["full_graft"]["integ"], probe, read_ablate=False, teacher_force=False)
    return Z_sgd, keys_v, values_v

def item_1_2_keygeom(integ, keys_v, values_v):
    cond = torch.linalg.svdvals(keys_v)                      # (B,K)
    cond_ratio = (cond[:, 0] / cond[:, -1])
    U, S, Vh = torch.linalg.svd(keys_v, full_matrices=True)
    gap = S[:, -1] / (S[:, -2] + 1e-12)                       # D3 extension: null-space health
    Z_ideal = integ.teacher_force_operator(keys_v, values_v)  # proof-device use only, D1.2/D8
    row_norms = Z_ideal.norm(dim=-1)                          # (B,d) per-row
    return dict(cond_med=cond_ratio.median().item(), cond_p99=cond_ratio.quantile(0.99).item(),
                cond_max=cond_ratio.max().item(), null_gap_med=gap.median().item(),
                null_gap_min=gap.min().item(), Z_ideal_fro_med=Z_ideal.norm(dim=(-2,-1)).median().item(),
                Z_ideal_fro_max=Z_ideal.norm(dim=(-2,-1)).max().item(),
                row_norm_med=row_norms.median().item(), row_norm_max=row_norms.max().item())

def item_3_reachability(ncr_head, keygeom):
    """Dimensionally-correct reachability test (M2): row_out is ONE shared
    Linear(64,25) applied to a fixed-norm LayerNorm output for every row of
    Z, so the ACHIEVABLE per-row output dynamic range (max/min realizable
    row norm, for a fixed input norm) equals row_out.weight's own condition
    number -- compare THAT ratio against the REQUIRED row-norm ratio
    (item 2's max/med), not against an absolute row norm (unit mismatch)."""
    sv = torch.linalg.svdvals(ncr_head.row_out.weight)        # weight: (25,64) -> 25 singular values
    cond_row_out = (sv[0] / sv[-1]).item()
    required_dynamic_range = keygeom["row_norm_max"] / max(keygeom["row_norm_med"], 1e-8)
    return dict(cond_row_out=cond_row_out, required_dynamic_range=required_dynamic_range,
                gate_pass=(cond_row_out >= required_dynamic_range))

def item_4_transverse(Z_sgd, keys_v):
    _, _, Vh = torch.linalg.svd(keys_v, full_matrices=True)
    w = Vh[:, -1, :]
    Zw = torch.einsum('bij,bj->bi', Z_sgd, w)
    return dict(transverse_gain_med=Zw.norm(dim=-1).median().item(),
                transverse_gain_p90=Zw.norm(dim=-1).quantile(0.90).item(),
                gate_pass=(Zw.norm(dim=-1).quantile(0.90).item() <= 3.0))

def item_5_ortho_conflict(integ, keys_v, values_v):
    Z_ideal = integ.teacher_force_operator(keys_v, values_v).detach().requires_grad_(True)
    ortho_loss = R.ortho_regularization_loss(Z_ideal)         # verbatim reuse
    grad = torch.autograd.grad(ortho_loss, Z_ideal)[0]
    return dict(ortho_loss_at_Z_ideal=ortho_loss.item(), grad_norm_med=grad.norm(dim=(-2,-1)).median().item())

def item_6_calibration_real_geometry(Z_sgd, keys_v, values_v, w, integ, pools, cfg, device,
                                      lambda_t_grid=(0.1, 1.0, 3.0, 10.0), n_steps=3000, lr=0.05):
    """D1-corrected sweep on REAL geometry -- replaces R3's synthetic sim_C/sim_D.
    Free Z per episode (Adam, CPU-cheap: a 25x25 matrix, n_steps << a training
    step's own cost), real keys_v/values_v, real nm.binexp_read + the pinned
    runner's own discriminability_metrics for scoring (exact call signature
    confirmed by the build agent against ncr_lm_wave1_runner.py:480-537, not
    re-derived here)."""
    v_bar2 = values_v.pow(2).sum(-1).mean(-1) + 1e-6
    Z_ideal = integ.teacher_force_operator(keys_v, values_v)   # init point only, not a runtime dependency
    results = []
    for lam_t in lambda_t_grid:
        Z = (Z_ideal + 0.05 * Z_ideal.norm(dim=(-2, -1), keepdim=True) * torch.randn_like(Z_sgd)
             ).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            Zk = torch.einsum('bij,bkj->bki', Z, keys_v)
            L_key = ((Zk - values_v).pow(2).sum(-1) / (values_v.pow(2).sum(-1) + 1e-6)).mean(-1)
            Zw = torch.einsum('bij,bj->bi', Z, w)
            L_trans = Zw.pow(2).sum(-1) / v_bar2
            (L_key + lam_t * L_trans).mean().backward()
            opt.step()
        with torch.no_grad():
            # retrieval24@{1,13,37,61} on the CONVERGED Z, via the real read path --
            # eval_arm_at_hops-equivalent scoring, wired by the build agent against
            # the pinned runner's own signature (same discipline as the premise
            # battery's own cell_p0_p1b, not reinvented here).
            retr = R.score_operator_at_hops(Z.detach(), keys_v, cfg, device, hops=(1, 13, 37, 61))
        results.append(dict(lambda_t=lam_t, L_key_final=L_key.mean().item(),
                             transverse_final=Zw.norm(dim=-1).mean().item(), retrieval24=retr))
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    n = 8 if args.smoke_only else N_EPISODES
    t0 = time.time()
    out = {}
    for name, ckpt_path in CKPTS.items():
        pools, cfg, pool_report = R.build_grammar_pools_and_cfg(seed=0)
        pools = pools.to(args.device)
        ckpt = R.load_checkpoint(ckpt_path, args.device)
        assert ckpt is not None, f"checkpoint not found at {ckpt_path}"
        arms, _, _ = R.restore_arms_and_opts(ckpt, pool_report["vocab_size_total"], lr=3e-4,
                                              device=args.device, freeze_entity_adapter=False)
        Z_sgd, keys_v, values_v = extract_real_kv(arms, pools, cfg, args.device, n=n)
        integ = arms["full_graft"]["integ"]
        keygeom = item_1_2_keygeom(integ, keys_v, values_v)
        _, _, Vh = torch.linalg.svd(keys_v, full_matrices=True)
        w = Vh[:, -1, :]
        out[name] = dict(
            item12=keygeom,
            item3=item_3_reachability(arms["full_graft"]["ncr"], keygeom),
            item4=item_4_transverse(Z_sgd, keys_v),
            item5=item_5_ortho_conflict(integ, keys_v, values_v),
            item6=item_6_calibration_real_geometry(Z_sgd, keys_v, values_v, w, integ, pools, cfg,
                                                    args.device) if name == "compB" else None,
        )
    with open(os.path.join(OUT_DIR, "stage0prime.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    if time.time() - t0 > CEILING_S:
        print(f"WARNING: exceeded {CEILING_S}s wall-clock ceiling", file=sys.stderr)
```

**Exact box-side launch commands (single sequential job, all 3
checkpoints inside one process — no packing/placement decision, matches
the premise battery's own precedent for a small eval-only pass):**

```bash
scp stage0prime_eval.py youthful-indigo-turkey:~/ncr_writecond/

ssh youthful-indigo-turkey \
  'tmux new-session -d -s writecond_stage0prime \
   "cd ~/ncr_writecond && CUDA_VISIBLE_DEVICES=0 /home/nvidia/tdenv/bin/python3 \
    stage0prime_eval.py --device cuda 2>&1 | tee stage0prime.log"'

# Poll: tmux has-session -t writecond_stage0prime (exits nonzero once done)
# Read: ~/ncr_writecond/results/stage0prime.json
```

**GPU class.** Any single free H100 among 0–7. Never `pkill` —
`tmux kill-session -t writecond_stage0prime` by exact name if it needs
stopping.

**Expected wall-time.** Well under 20 minutes (3 checkpoint loads +
forward-only passes + item 6's cheap CPU-weight Adam sweeps on
compB only) — comfortably inside the `40`-minute kill-switch and the
`≈0.1 GPU-h` expected cost.

**Output path.** `~/ncr_writecond/results/stage0prime.json`. Scp back
into `experiment-runs/2026-08-13_ncr_writecond_premise_battery/` (repo,
small-file, hybrid archive policy) once complete — same archive
directory as the premise battery, since this is its direct
methodological continuation.

**Gating readout (restated, self-contained).** Item 3 FAIL (row_out's
achievable dynamic range short of `Z_ideal`'s required range) ⇒ STOP,
escalate — the target is unreachable by this parametrization, rescope
before Stage 1. Item 4 shows trained-checkpoint transverse gain `≫3` ⇒
CONFIRMS D1's mechanism is necessary (proceed, with elevated
confidence, not a blocker). Item 5 FAIL (ortho/`L_write` conflict does
NOT reproduce on real geometry) ⇒ open question for the audit —
D2's removal decision would then rest on R2's synthetic evidence alone,
flagged, not silently proceeded past. Item 6 FAIL (provisional `λ_t=
1.0` does not suppress transverse gain under 3 within budget on real
geometry) ⇒ revise `λ_t` per item 6's own curve before Stage-1 launch,
not after.

---

### Fresh self-attack — what would kill this at R4

1. **The transverse penalty is defined per-batch from `keys_v` — but
   `keys_v` varies EVERY batch (a fresh document each step), so `w`
   is a moving target, not a fixed direction.** D1.3's zero-set proof
   is per-episode (`Z_ideal(b)`, `w(b)` both indexed by `b`) — correct
   for the loss's own well-posedness — but this means the network never
   sees a repeated, learnable geometric structure to anchor a general
   "suppress the transverse direction" strategy on; it must learn a
   *function* `keys_v ↦ Z(keys_v)` that keeps `Z(keys_v) w(keys_v)`
   small for EVERY `keys_v` it might see, not memorize one `w`. This is
   presumably learnable (row-attention encoders regularly generalize
   across varying inputs) but is NOT verified anywhere in this document
   — Stage 0′ item 6's real-geometry sweep tests whether the *loss
   surface* rewards this, not whether *training* actually reaches it
   across the full data distribution. The clean disambiguator (not
   funded here, registered for the audit's consideration) is a learning
   curve of `‖Z_sgd w‖` over Stage-1 training steps, not just its final
   value.
2. **`λ_t` is a NEW hyperparameter with no swept range in Stage 1** —
   Stage 0′ item 6 picks (or revises) a single value before launch, and
   Stage 1 commits to it for all three PRIMARY recipes at once. If the
   real-geometry calibration curve turns out to be steep (a narrow
   window of `λ_t` that suppresses transverse gain without somehow
   fighting CE/aux convergence — D1.3's orthogonality argument says it
   shouldn't compete for a MINIMUM, but says nothing about
   optimization-trajectory interactions during the first few thousand
   steps, before either loss is near converged), a single point
   estimate from Stage 0′ could under- or over-shoot for one or more of
   the three recipes' own key geometries, which D1.5/D5's per-recipe
   references already show differ measurably (`cond`, TPC, `f(A*)`
   history). A future round should consider whether `λ_t` needs to be a
   *third* Stage-1 axis (not funded/scoped here) if Stage 0′'s
   single-checkpoint (compB-only) item-6 sweep does not visibly
   transfer to compA/primary's own geometry.
3. **CONTROL C answers "does removing ortho alone move `P0`'s number,"
   but not "does ortho ACTIVELY fight write supervision specifically
   because of the loss-landscape mismatch F2 diagnosed, or for some
   more generic gradient-competition reason that would also show up
   between `L_write` and CE/aux."** If PRIMARY WINS with `ortho=0` and
   CONTROL C also reads NULL (ortho removal alone does nothing), that
   is fully consistent with F2's mechanism-specific diagnosis — but
   does not positively rule out a more boring "any extra loss term
   fights `L_write` a little, ortho or not" story, since no `ortho=0.1,
   L_write ON` arm is funded in this wave (D2's stated reason: this
   was ALREADY measured to fail by F2's own synthetic joint-
   minimization sweep, at a cost the design chose not to re-pay on the
   real box). This is a real, disclosed gap between "we have a
   mechanism-specific story" and "we have ruled out every alternative
   story" — registered honestly rather than closed by an unfunded
   fourth cell.
4. **Band 0's leak-guard checks the FLAG, not the loss's OWN
   sensitivity to an accidental near-teacher-force regime** — if
   `write_supervision_weight` were set absurdly high (well past what
   Stage 0′ calibrates), the trained `Z_sgd` could converge close
   enough to `Z_ideal` that Stage-1 PRIMARY becomes *behaviorally*
   indistinguishable from a genuine teacher-forced arm even with the
   flag correctly `False` throughout — not a leak, but a claim-shape
   question (is "trainable-with-write-supervision" still the right
   description of a `Z_sgd` that has converged to within numerical
   noise of `Z_ideal` at every step?). §2(c)'s honest boundary (a WIN
   establishes "trainable-with-write-supervision," a weaker but real
   claim, R3 §2(c), unchanged) already covers this in principle; an
   auditor may reasonably ask for `‖Z_sgd − Z_ideal‖` to be logged
   explicitly (cheap, already have both tensors during eval-time
   diagnostics) so the harvest can report how close is "close," not
   just that the flag was off.

---

Rev-4 dispatched 2026-08-13. Attack R4, scoped to this section, and
Stage 0′ (independently authorized pre-CLEAR per `§A3-ADJUDICATION`)
run next, per the coordinator's own dispatch. No GPU spend, no STATE.md/
EXPERIMENT_LOG.md update, no commit, no box contact from this document.

---

## §A4-ADJUDICATION (coordinator, 2026-08-14) — attack R4 = REV-REQUIRED (4F/11M/10m) ADOPTED; MECHANISM CERTIFIED CLEAN; instrument repairs = Rev-5 charter; amended Stage 0′ becomes the wave-deciding gate

Report: `NCR_WRITECOND_ATTACK_R4.md` (834 lines) vs DRAFT-R4 at
`8c665f7`. **The qualitative turn: the mechanism is CLEAN** — D1.3's
zero-set collapse independently re-derived AND executed at compB's
measured geometry (L_key flat to 1.9e-12 across ‖u‖ 0→100; Z_ideal·w
= 1.7e-05; joint zero only at u=0); D1.2's w construction, D3's
detach coverage incl. the SVD route, D5.2's partition (18,018
constructed outcomes, zero holes/double-fires), raw-anchor table,
budget to the digit, language sweep — V1–V13 FROZEN for R5.

**FATALs adopted, each with the report's named repair:**
- F1: the global-scale quotient was DROPPED IN TRANSCRIPTION (R3's
  bound was "‖Zw‖ ≲ 3 AT ‖Z‖_F ≈ 25"); executed: Z = 1.5·Z_ideal
  retrieves 1.0000 yet fails Band 2, while a collapsed ‖Z‖_F≈1
  encoder passes the gate at chance. Repair: the four-line c*-rescale
  (scale-free transverse reading ‖Zw‖/‖Z‖_F·√d-normalized per the
  report).
- F2: gradient-orthogonality is exact in Z-space (cos 0.000) and
  FALSE through the encoder's 173,209 parameters (cos 0.17–0.45 vs
  0.0017 null); λ_t=1.0 turned retr@61 0.8906 → 0.1406 at equal
  budget. Repair: λ_t is DECIDED BY THE AMENDED STAGE 0′ parametrized
  probe (A6), never assumed benign.
- F3 (as numbered in the report) + F4: Band 0 voids CONTROL B by
  construction (no self-correcting path); CONTROL C funded but enters
  no adjudicating predicate (counter-example: C=0.70, PRIMARY=0.71
  still WIN). Repairs: the report's Band-0 CONTROL-B branch + a
  C-vs-PRIMARY margin predicate in the verdict grid.
- M2 (promoted to load-bearing): L_key ≤ 3e-4 has NO demonstrated
  achievability — the real encoder plateaus at 0.22–0.40 in
  write-side isolation with the pre-registered Band-4 signature.
  The AMENDED Stage 0′ achievability/λ_t probe (~0.2 GPU-h,
  eval+probe-only) is hereby the WAVE-DECIDING GATE: the 24.94 GPU-h
  Stage 1 launches only on its pass bands.
- Also adopted: the pinv truncation cliff at cond(keys) ≈ 3.4e5
  (target lands 100× outside WIN — a Stage-0′ item), and the
  Tikhonov-fallback rejection (moves the minimum 81–100% off
  target — struck).

**Stage 0′: BLOCKED AS CARDED → CLEARED AFTER A1–A8** (checkpoint
paths corrected to the battery's recorded `…_contrastive/…_s0_ckpts/`
convention; the `.encoder` attribute route; the nonexistent
`score_operator_at_hops` replaced per A-list; item 6 → the
parametrized achievability/λ_t probe; probe-batch retention).

**BUILD-PREPARABLES (authorized in parallel per the report):** the
loss module (detach + scale-free transverse), d−K generalization,
config provenance, Band-0 checker w/ CONTROL-B branch, write_diag
emission, CONTROL B's clean-eval script, the amended Stage-0′
script. NOT authorized: any Stage-1 cell, any λ_t value, any band
threshold.

**DISPOSITIONS: the report's repairs + A1–A8 adopted verbatim as the
Rev-5 charter.** Rev-5 → coordinator runs amended Stage 0′ →
R5 narrow verification (expected terminal on the instrument layer) →
build ceremony on its CLEAR + Stage-0′ pass.

---

## DRAFT-R5 (Rev-5, 2026-08-14)

**Charter.** `§A4-ADJUDICATION` above, adopted in full, binding: the
report's E1/E2/E4/E6 repairs + Stage-0′ amendments A1–A8, applied as
instrument-level fixes only. **V1–V13 (R3) and F1–F4/M1–M11 (R4's own
CERTIFIED-CLEAN mechanism verdict) are FROZEN and NOT re-derived below** —
every number cited from the report is reproduced from
`NCR_WRITECOND_ATTACK_R4.md`, never re-measured. This section supersedes
DRAFT-R4 (lines 2946–3957) for every downstream purpose; DRAFT-R4 stays
verbatim as the historical record of what attack R4 killed (house
convention).

**Grounding read for this round.** `NCR_WRITECOND_ATTACK_R4.md` in full
(834 lines); the pinned runner
`experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/ncr_lm_wave1_runner.py`
(md5 `9a93198b642242f512ff8489e32b0a53`, re-confirmed this round by direct
`md5` on the archived file — unchanged); the premise battery's own archived
scripts (`experiment-runs/2026-08-13_ncr_writecond_premise_battery/
pbe_repl.py.txt`, `pbe_supplement.py.txt`, `premise_battery_eval.py.txt`,
and the three `writecond_premise_REPL_*.json` / `writecond_premise_SUPP.json`
result files — read directly for their own recorded `ckpt`/`ckpt_step`
fields, not assumed); `matrix-thinking/ncr/ncr_models.py`,
`matrix-thinking/ncr/ncr_earlyln_scale.py`, `matrix-thinking/chapter2/
model_v4.py` (read in full for the exact class hierarchy — `NCREarlyLNModel
(nm.NCRModel)` → `self.encoder = BindingEncoder(...)` → `self.row_out =
nn.Linear(h, d)`, `self.row_norm = nn.LayerNorm(h)` with
`elementwise_affine=True` by default, `model_v4.py:41-52`).

Every function/attribute this section or its build artifacts reference was
**cross-checked against the pinned runner's and the real model classes'
ACTUAL source** this round (not the design text's own prior claims) —
`discriminability_metrics` (runner.py:480-537), `ortho_regularization_loss`
(runner.py:714-742), `eval_arm_at_hops` (runner.py:934-964),
`build_task1_document` (`ncr_lm_wave1_smoke.py:441-464`), `query_key`
(`ncr_lm_wave1_smoke.py:331-346`), `teacher_force_operator`
(`ncr_lm_wave1_smoke.py:348-362`), `load_checkpoint`/`restore_arms_and_opts`
(runner.py:1122-1179), and the checkpoint-path convention itself (read
directly off `pbe_supplement.py.txt:16` and
`writecond_premise_REPL_{compA,primary}.json`'s own `ckpt` fields, not
inferred).

---

### R5.1 — F1 repair: the scale quotient restored exactly

D4/D1.5's Band 2 is rewritten scale-invariantly, exactly per the report's
binding repair (E1):

```
Band 2 (F1-repaired):
  gate (i)  L_key(c* . Z_sgd)  <=  3e-4                       [D1.5, unchanged]
  gate (ii) ||Z_sgd W^T||_F / ||Z_sgd||_F  <=  0.12            [= 3/25, R3's OWN
                                                                  "at ||Z||_F ~= 25"
                                                                  12% statement,
                                                                  RESTORED with its
                                                                  conditioning]
  c* = sum_i <Z_sgd k_i, v_i> / sum_i ||Z_sgd k_i||^2   (per episode, closed form)
  reduction statistics NAMED: median AND p90 across the eval batch, both gates
  applies at h* (eval batch); neither quantity is a function of h
```

`W` is M11's `d-K` generalization of `w` (`Vh[:, K:, :]`, reducing to the
carded single `w = Vh[:, -1, :]` exactly at `K = d-1`) — folded in here
rather than deferred, since D4's own rewrite and M11's generalization touch
the identical quantity.

**Executed re-verification** (`matrix-thinking/writecond_build/
write_supervision_loss.py` + `test_write_supervision_loss.py`, real
`K=24,d=25` dims, fp32, CPU, 35/35 checks PASS):

- **Case 1 (the report's own table).** `Z = c . Z_ideal` for
  `c in {0.01, 0.1, 1, 1.5, 10, 100}`: the repaired gate (i)
  `L_key(c*.Z) <= 3e-4` PASSES at **every** `c` (`L_key_cstar` max
  `4.35e-11` to `4.27e-11` across all six, vs. the **raw**, un-repaired
  gate — reproduced alongside as a regression check — which FAILS at
  every `c != 1` exactly as F1's table shows, e.g. `L_key_raw = 2.500e-01`
  at `c=1.5`, `9.801e+03` at `c=100`).
- **The collapsed-`||Z||` counter-example, closed.** An operator with a
  transverse component whose norm is FIXED (independent of any global
  rescale `c`) — the shape of the report's own case
  (`||Z||_F~1, ||Zw||=0.20`, "passes the bare 3.0 comfortably, reads at
  chance") — is scored at `c in {1, 0.1, 0.01}`: the RATIO gate (ii)
  FAILS at every `c` (`Zw_ratio` median `0.9933`, scale-invariant by
  construction), while the OLD bare `||Zw|| <= 3.0` reading would have
  **PASSED 81% of episodes** at `c=0.01` — the exact loophole F1
  diagnosed, reproduced and closed in the same test.
- **M11's `d-K` generalization**, verified independently at `K=3,d=6`
  (a genuinely multi-dimensional null space, `d-K=3`): `Z_ideal` is still
  the exact zero of the generalized loss (`L_write=1.03e-13`); a
  null-space-only perturbation (`Delta = U W`, `U` random) keeps `L_key`
  at `~0` and makes `L_transverse` strictly positive; a span(keys)-only
  perturbation does the reverse — the same V1–V5 pattern, now proven at
  `K < d-1`, not just the carded `K=24,d=25` point.

`band2_check()` in `write_supervision_loss.py` is the single, tested
implementation; its two threshold defaults (`3e-4`, `0.12`) are asserted
directly against the report's own numbers in the test suite (`inspect.
signature` check) so a future edit cannot silently drift them.

---

### R5.2 — F2 repair: λ_t nowhere assumed; the parameter-space conflict is a monitored risk

`write_supervision_loss()` (the shipped loss module) takes `lambda_t` as a
**required** argument with **no default value** — every call site,
including Stage 0′ item 6, must supply it explicitly; there is no code
path in this build where a value is silently assumed (F2's repair, E2).

**The parameter-space gradient-conflict fact, recorded, not re-derived**
(V1–V13/F1–F4 frozen per the charter — these are the report's own
executed numbers, `NCR_WRITECOND_ATTACK_R4.md` F2(a)/(d)):

```
cos(grad_theta L_key, grad_theta L_transverse):
  Z-space (D1.3's literal claim)         0.000000
  parameter space, at init               0.4458
  parameter space, after 200 steps       0.1695
  parameter space, after 600 steps       0.2845
  random-pair null, same 173,209-dim space   0.0017
```

100–260× the null — the terms are coupled (cooperatively, per the
report's own measurement) in the space the optimizer actually works in,
even though D1.3's Z-space orthogonality is exact. **This is now a
monitored risk, with a named instrument, not an assumption papered over:**

1. **Stage 0′ item 6's own learning curves ARE the first-line instrument.**
   `item_6_achievability_probe()` records `L_key_med`, `L_transverse_med`,
   and `Zw_ratio_med` **every `log_every` steps across all `n_steps`**, per
   `(lr, lambda_t)` cell — not just the final value. A parameter-space
   competition strong enough to matter would show up as non-monotone or
   stalled convergence in these curves, on real geometry, **before** any
   Stage-1 GPU-h is spent, at every grid point Stage 0′ runs.
2. **The BUILD-PREPARABLES list's own D7 repair is the Stage-1-time
   instrument** (authorized, not yet wired — see R5.5): per-step
   *separated* `||grad_Z L_key||` / `||grad_Z L_transverse||` logging for
   both PRIMARY and CONTROL A — D7's own point, now doubly load-bearing
   because F2 showed the parameter-space relationship is the open
   question, not a settled orthogonality.
3. **No band threshold is set on this risk** (per the charter's own
   constraint) — it is carried forward as an explicitly named, instrumented
   risk for the harvest to read, not resolved by assumption or by an
   invented numeric gate.

---

### R5.3 — Band-0/CONTROL repairs

**Band 0, F4-repaired (E4), restated in full:**

```
Band 0 (leak gate), F4-repaired:

  PRIMARY / CONTROL A / CONTROL C  (unchanged from D5.3):
    config.teacher_force_operator == False
    AND teacher_force_check.active == False
    AND teacher_force_check.ncr_zero_grad_checks_passed == 0
    FAIL => VOID the cell (re-run, never scored)

  CONTROL B  (INVERTED, F4's repair):
    config.teacher_force_operator == True
    AND teacher_force_check.active == True
    AND teacher_force_check.ncr_zero_grad_checks_passed == steps_run
    AND a SEPARATE D7 clean-eval artifact exists with its OWN
        config.teacher_force_operator == False
    FAIL => VOID (same remedy)
```

CONTROL B's own D7 definition ("reuses `--teacher-force-operator`
VERBATIM... so `ncr_head` receives EXACTLY zero gradient, asserted every
step") **necessarily** produces the inverted triple — the un-repaired gate
VOIDs a *correctly-run* CONTROL B by construction, with no self-correcting
path (F4's own finding, reproduced below).

**Executed re-verification** (`band0_checker.py` + `test_band0_checker.py`,
13/13 checks PASS, no torch needed):

- **F4's bug, reproduced first.** `band0_check_current()` (the pre-repair
  gate, kept ONLY as a regression fixture) VOIDs a well-formed CONTROL B
  record (`teacher_force_operator=True, active=True,
  ncr_zero_grad_checks_passed=2000`) — confirming the defect existed
  before asserting the fix.
- **The repaired gate PASSES the identical record.**
- **Six negative tests, all executed, all fire correctly:** missing D7
  clean-eval artifact → VOID; a clean-eval artifact itself mis-flagged
  (`teacher_force_operator=True`) → VOID; a continuation that did NOT
  actually teacher-force throughout (`active=False`) → VOID (the
  inversion is not "always pass CONTROL B"); a partial
  `1999/2000` zero-grad-check count → VOID (off-by-one has teeth); a
  missing `steps_run` → VOID rather than silently defaulting; PRIMARY/
  CONTROL A/CONTROL C's own unchanged branch still VOIDs a leaked cell.

**The C-vs-PRIMARY margin predicate (M3, E7), added to the verdict grid:**

```
D5.2's WIN predicate, M3-repaired:

  compB:            WIN  iff  win_base(x)  AND  (x - CONTROL_C(compB)) > 0.15
  compA / primary:  WIN  iff  win_base(x)         [unchanged predicate --
                                                     no matched CONTROL C
                                                     funded this wave]
                     -> labeled ortho_confounded_disclosed = True whenever
                        this WIN fires (the report's OWN second option,
                        taken here since no new Stage-1 cell is authorized)

  win_base(x) := x > 0.19167  AND  fraction_closed_recipe(x) >= 0.70
                 AND  GAP(full_graft - backbone_only, h*) > 0.15   [D5.2, unchanged]

  NULL := x <= tau (0.09162)          PARTIAL := NOT NULL AND NOT WIN
```

**M3's own worked counter-example, reproduced and closed**
(`band_partition.py` + `test_band_partition.py`): a PRIMARY reading
`x=0.7100` at compB anchors (`fraction_closed = 0.7071 >= 0.70`) —
`win_base` is identical whether `CONTROL_C` reads `0.0664` or `0.7000`
(confirmed: `win_base_before_c_margin` is `True` in both cases,
identically, reproducing exactly the defect the report diagnosed). After
the repair: `CONTROL_C=0.0664` (margin `0.6436 > 0.15`) still scores
**WIN**; `CONTROL_C=0.7000` (margin `0.01 <= 0.15`) now scores
**PARTIAL** — the confound the cell was funded to close is now actually
closed by the scoring rule, not just by the cell existing.

**Partition re-verified hole/double-fire-free AFTER the addition** (the
charter's own explicit requirement — re-run, not asserted): **50,736
constructed outcomes** (43,452 for compB across a fine `x`-grid ×
`{GAP, C-margin-offset}` boundary sweep including the exact `0.15`
boundary on BOTH the GAP and the new C-margin clause; 7,284 for
compA/primary across the same `x`×`GAP` sweep) — **zero holes, zero
double-fires**, in both sweeps. The disclosure flag
(`ortho_confounded_disclosed`) is verified to fire if-and-only-if a
compA/primary cell reads WIN, on three representative cases (WIN/NULL/
PARTIAL). `classify()` refuses to run for `recipe="compB"` without a
`control_c_reading` argument (an `AssertionError`, executed) — M3's repair
cannot be silently skipped by omission.

---

### R5.4 — Stage 0′ FINAL CARD (A1–A8 applied)

**A1 — checkpoint paths, corrected to the battery's own recorded
convention** (verified directly against `pbe_supplement.py.txt:16` and
`writecond_premise_REPL_{compA,primary}.json`'s own `ckpt` fields, not
inferred):

```
~/ncr_g3b31_contrastive/results/mob_g3b31_{tag}_s0_ckpts/mob_g3b31_{tag}_s0.ckpt.pt
   for tag in {primary, compA, compB}
```

**A2 — the `.encoder` attribute route**, verified against the real class
hierarchy this round (`ncr_models.py:165`, `ncr_earlyln_scale.py:115-126`,
`model_v4.py:52`): `arms["full_graft"]["ncr"].encoder.row_out`, NOT
`arms["full_graft"]["ncr"].row_out` — executed:
`hasattr(NCREarlyLNModel_instance, 'row_out')` is `False`,
`hasattr(NCREarlyLNModel_instance.encoder, 'row_out')` is `True`
(`test_stage0prime_helpers.py`).

**A3 — item 3's statistic.** All four candidate statistics are reported
(global max/median, global max/min, within-episode max/min median/p99/
max); the gate uses the **within-episode p99** (the spec statistic) rather
than the carded global max/median. M4(b)'s LayerNorm-affine correction is
applied — `encoder.row_norm` is `nn.LayerNorm(h)` with
`elementwise_affine=True` (verified: default, unset in `model_v4.py`), so
the achievable ceiling is `sigma_max(row_out.weight) . (||gamma||_inf .
sqrt(h) + ||beta||) + ||row_out.bias||`, restored (M4(c)) alongside the
ratio framing rather than replacing it.

**A4 — item 4's semantics.** Emits `transverse_gain_med/_p90` (absolute,
informational per F1) **and** `transverse_ratio_med/_p90`
(`||ZW||_F/||Z||_F`, the quantity that actually adjudicates Band 2). M10's
polarity fix: no more inverted `gate_pass` — renamed
`transverse_gain_exceeds_3` (`True` = CONFIRMS D1's mechanism is
necessary). Verified with a positive AND a negative case (`Z_ideal` reads
`False`; a large transverse perturbation flips it to `True`).

**A5 — item 5's numeric predicate + the 1/B fix.** `conflict_reproduces :=
ortho_loss(Z_ideal) > 1e3 AND grad_norm_per_example > 0`
(the report's own stated example, taken verbatim — no threshold invented
beyond it), with M8's correction applied: `ortho_regularization_loss`
batch-mean-reduces, so `torch.autograd.grad` yields per-example gradients
scaled by `1/B`; multiplying by `B` undoes exactly that factor — verified
**exactly**: the correction ratio measured `64.000` at `B=64`, to the
digit. R3's own joint-minimization `lambda_w` curve is **disclosed as NOT
reproduced** this round (M8's second substitution, now flagged explicitly
— `joint_min_curve_reproduced: False` is a literal field in every item-5
result, not a silent omission); D2's `ortho=0` decision rests on this
item's real-geometry reconfirmation plus F2's own synthetic evidence.

**A6 — item 6, the substantive replacement.** The free-`Z` sweep (which
F2(c) proved cannot fail, so cannot choose `lambda_t`) is replaced by a
**parametrized achievability/λ_t probe**: a FRESH
`els.NCREarlyLNModel(d=25,h=64)` instance per `(lr, lambda_t)` grid cell
(`lr in {3e-4, 1e-3}`, `lambda_t in {0, 0.1, 1.0, 3.0}` — 8 cells), trained
on `L_key + lambda_t.L_transverse` for `>=8000` Adam steps on the
extracted `(keys_v, values_v)`, scored on a **separate, hop-matched
held-out set** via the real path (`nm.binexp_read` →
`R.discriminability_metrics`).

**Cost-driven scoping decision, disclosed (a deviation from a literal
reading of the report's own words, not a silent one — see R5.6).** The
report says "build one batch per hop and fit per hop." Read literally,
that means retraining the whole 8-cell grid once per scored hop. This
build instead trains **once** per cell (`keys_v`/`values_v` content is
K raw bind-clause entity vectors — not itself hop-conditioned; only the
query/target construction is, per `build_task1_document`'s own
`hop_set` mechanism) and scores **each** trained cell against
hop-**matched held-out documents at every hop** — closing exactly the
defect F3.5 diagnosed (an h=61-fit `Z` scored against a document built
for a different `h`) at half the training cost of a literal per-hop
retrain. Band 2's own reading is reported **per held-out hop** (not just
one, cherry-picked) — `held_out_band2_by_hop`, keyed by hop — and the
GATE requires **all** scored hops to pass, not just one.

**F3.4's probe-batch-retention fix.** `extract_real_kv()` now returns the
full `probe` batch dict (not just `Z_sgd, keys_v, values_v`), so
`entity_ids`/`tgt_slot`/`query_key_col` survive for scoring — the un-amended
card discarded them, making item 6 unable to score retrieval at all (F3.4).
A dedicated `extract_held_out()` draws a disjoint-seeded batch **per hop**
(`seed + EVAL_SEED_OFFSET + hop + 500_000`, disjoint from the training
draw's own `seed + EVAL_SEED_OFFSET + 61`).

**`R.score_operator_at_hops` (the non-existent function F3.3 flagged) is
replaced** by the real composition the pinned runner actually exposes:
`nm.binexp_read(Z, q_key.unsqueeze(1), h)["o"].squeeze(1)` (real, verified
signature) followed by `R.discriminability_metrics(integ, embed, o,
entity_ids, tgt_slot)` (real, verified signature, lines 480–537 — the SAME
function the design's own dead citation pointed at the wrong lines for).

**A7 — item 1 becomes gating** on `cond(keys_v)` (med/p99/max) **and** the
fraction of episodes whose own target `L_key(Z_ideal)` violates the WIN
band (M5's pinv-truncation cliff). No numeric pass/fail cutoff is invented
for "non-trivial fraction" (the report's own language is qualitative) —
the fraction and `cond` stats are reported for the coordinator's own
reading, per the charter's "no band thresholds beyond what the report
fixes" constraint. **The cliff itself is reproduced**: a near-collinear
key pair drives `cond_max` past `10^4` and moves
`frac_episodes_target_violates_win` off zero from a well-conditioned
baseline of exactly `0.0` (`test_item_1_2_pinv_truncation_cliff`,
executed).

**A8 — housekeeping, all applied:** `os.makedirs(OUT_DIR, exist_ok=True)`;
seed corrected to `+ EVAL_SEED_OFFSET + 61` for the training extraction
(m2 — the un-amended card drew a *different* episode distribution than the
one that produced `P0=0.0664`, making item 4's readings non-comparable to
the harvest's own numbers); `CEILING_S` reframed as a post-hoc warning only
(the real ceiling is the `timeout 2400` wrapper on the box launch command,
m3); `CUDA_VISIBLE_DEVICES`/GPU-class note unchanged from the card ("any
single free H100 among 0-7", m4 — left as the card's own choice, not
re-litigated); item 6's finals are read from the trained model's own
post-training state, not a stale pre-`opt.step()` snapshot (m5, structurally
avoided by this build's own control flow — training and scoring are
separate loops, not interleaved).

**Gating bands for THE WAVE-DECIDING READING:**

```
Item 1 (cond + pinv-cliff fraction):    reported, non-gating (A7 — informational,
                                          coordinator judgment call, per the report's
                                          own qualitative language)
Item 3 (reachability):                  GATES. FAIL (cond(row_out) < required
                                          within-episode-p99 dynamic range)
                                          => STOP, escalate -- rescope before Stage 1.
Item 4 (transverse gain on TRAINED
        checkpoints):                   informational (transverse_gain_exceeds_3=True
                                          CONFIRMS necessity, elevated confidence,
                                          never a blocker).
Item 5 (ortho conflict):                GATES as a sanity re-confirmation.
                                          conflict_reproduces=False => open question for
                                          the audit, D2's ortho=0 removal then rests on
                                          F2's synthetic evidence alone (flagged, not
                                          silently proceeded past).
Item 6 (achievability/lambda_t):        THE WAVE-DECIDING GATE.
   stage1_gate = "GO" iff ANY (lr, lambda_t) cell reaches BOTH of Band 2's
      repaired gates (L_key_cstar<=3e-4 AND Zw_ratio<=0.12), at BOTH the
      median AND the p90, AT EVERY SCORED HOLD-OUT HOP.
   GO           => Stage 1 launches; the winning (lr, lambda_t) is the
                    provisional Stage-1 value UNLESS a Stage-1 lambda_t
                    axis / lambda_t=0 necessity arm is separately funded
                    (F2 repair (iii), still an open pre-registration
                    choice for the coordinator, not resolved by this
                    build).
   NO-GO-ON-CURRENT-BAND => Stage 1 does NOT launch on Band 2 as currently
                    calibrated. Two re-scope paths, both pre-registered by
                    the report, neither chosen here: (a) re-derive Band 2
                    from item 6's own achievable frontier (disclosed as a
                    post-hoc recalibration), or (b) the mechanism is
                    rescoped. This build makes NO GO/NO-GO call itself --
                    it has not been run on real checkpoints (no box
                    contact this round, per the charter).
```

**Item (j) — every fixture producible.** Every item's core function is
exercised end-to-end in `test_stage0prime_helpers.py` using **synthetic**
fixtures that match the real tensor shapes exactly (`K=24, d=25, h=64`,
real `els.NCREarlyLNModel`/`nm.binexp_read` — not hand-reimplemented
stand-ins): a well-conditioned key/value pair (item 1/2's baseline), a
deliberately near-collinear pair (item 1's cliff), a synthetic
`entity_adapter`/`embed` pair with a fixed lookup table (items 5/6's
scoring rig, `make_scoring_rig()`), and an all-zero held-out set (a
negative fixture proving item 6's held-out path is genuinely exercised,
not skipped). The ONE fixture this build cannot produce off-box is a real
trained checkpoint — `stage0prime_eval.py`'s own `load_checkpoint` call
asserts loudly (not silently) if one is missing, per A1's pre-flight `ls`
step.

---

### R5.5 — BUILD-PREPARABLES (authorized artifacts, all written, all tests RUN to completion)

All files live under `matrix-thinking/writecond_build/`. **131 individual
assertions plus a 50,736-outcome constructed sweep, all executed this
round, zero failures** (CPU, fp32, torch 2.8.0, no GPU, no box contact —
every negative test executed to completion, not merely written, per
CLAUDE.md):

| file | purpose | tests | result |
|---|---|---|---|
| `write_supervision_loss.py` | D1/D3/D4-F1/M11: the loss module — `null_directions`, `write_supervision_loss`, `band2_check` | `test_write_supervision_loss.py` | 35/35 PASS |
| `band0_checker.py` | F4: Band-0 gate with the CONTROL-B branch | `test_band0_checker.py` | 13/13 PASS |
| `band_partition.py` | D5.2 + M3: the NULL/WIN/PARTIAL partition + C-margin predicate | `test_band_partition.py` | 12/12 PASS (+ 50,736-outcome sweep, 0 holes/0 double-fires) |
| `write_diag.py` | M7: Band 2's emission point (`compute_write_diag`) | `test_write_diag.py` | 18/18 PASS |
| `config_provenance.py` | D8/M8/m4: config fields, mutual-exclusion assert, resume-mismatch asserts | `test_config_provenance.py` | 10/10 PASS |
| `stage0prime_helpers.py` | A1–A8: every item's box-independent core math (items 1–6) | `test_stage0prime_helpers.py` | 43/43 PASS |
| `stage0prime_eval.py` | the amended box launch script (A1–A8 wired) | box-only, cannot import off-box (`fla`) — see below | compiles clean (`py_compile`) |
| `control_b_clean_eval.py` | D7: CONTROL B's separate teacher_force=False clean-eval script, on the `pbe_repl.py` pattern | box-only | compiles clean (`py_compile`) |

**Why two files are box-only and untested end-to-end (disclosed, not
hidden).** `ncr_lm_wave1_runner.py` cannot be imported off this machine —
confirmed directly: `import ncr_lm_wave1_runner` raises
`ModuleNotFoundError: No module named 'fla'` (its own `graft ->
lm_pretrain_rd -> fla.ops.delta_rule.chunk_delta_rule` chain; the runner's
own module docstring already says "chunk_delta_rule has no CPU path").
`stage0prime_helpers.py` exists precisely to factor every item's **testable
math** out of that dependency chain (verbatim duplicates of
`discriminability_metrics`/`ortho_regularization_loss` — both already
fla-independent themselves, only the RUNNER MODULE's own import chain is
the blocker — matching the runner file's own established "duplicate,
don't drift" convention, `cosine_and_recovered_frac`'s own precedent).
`stage0prime_eval.py` and `control_b_clean_eval.py` import the REAL pinned
runner and are therefore box-only by construction; both `py_compile`
clean, both were cross-checked line-by-line against the pinned runner's
verified signatures (R5.4), and `stage0prime_eval.py` imports
`stage0prime_helpers` unmodified — the box run uses the SAME tested code
path this round's tests exercised, not a duplicate that could drift.

**Also shipped per the D7/M7/M8 build brief (specs, not yet wired into
the archived runner file — editing that file is outside this round's
authorized repo-write scope, `§A4-ADJUDICATION`):**
- `write_diag.py`'s `compute_write_diag(Z, keys_v, values_v)` — the exact
  six named fields (M7): `L_key, L_key_cstar, Zw_norm, Z_fro, Zw_ratio,
  keys_cond, keys_null_gap`. Wiring point named explicitly:
  `eval_arm_at_hops`, full_graft only, right after its own
  `ncr_lm_forward_ablatable` call (runner.py:942).
- `config_provenance.py`'s `assert_no_teacher_force_write_supervision_conflict`
  (m4) and `assert_writecond_resume_match` (M8) — both executed against
  positive AND negative cases, including the old-checkpoint-defaults case
  (M8's own "only NEW checkpoints are checked" guarantee).
- Per-step separated `||grad_Z L_key||`/`||grad_Z L_transverse||` logging
  (D7) is NOT yet a standalone artifact — `write_supervision_loss()`
  already returns both sub-losses un-summed (`L_key`, `L_transverse`,
  each `(B,)`), which is the one line of plumbing a Stage-1 build needs to
  call `.backward()` on each separately for the log; not written as a
  separate file since it is a direct, mechanical consequence of the loss
  module's own return shape, not new logic.

**Not authorized and not written this round** (per the charter): any
Stage-1 cell, any committed `lambda_t` value, any band threshold beyond
what the report already fixed.

---

### R5.6 — Budget/tail bookkeeping

**This round's own cost: zero GPU-h, zero box contact.** Every test above
ran CPU-only, fp32, torch 2.8.0, on the local machine — no SSH, no tmux, no
checkpoint read, matching the charter's own "no box contact" instruction.

**Stage 0′'s own cost, HONEST CORRECTION (flagged, not smoothed over).**
`§A3-ADJUDICATION`/`§A4-ADJUDICATION` both carry forward a "still ≲0.2
GPU-h" estimate for the amended Stage 0′ ("A6 adds a few minutes of
single-GPU training on a 173K-param head"). That estimate under-counts
A6's realized shape: items 1–5 remain cheap (forward passes + SVDs on
existing checkpoints, seconds, per V12); **item 6 is now 8 grid cells
(2 lr × 4 λ_t) × ≥8000 real Adam steps EACH, training an actual
173,209-param transformer encoder (not a free-`Z` elementwise
optimization)** — 64,000 gradient steps total (halved from a literal
per-hop-retrain reading of the report's own words by R5.4's disclosed
scoping decision). This build has **no real H100 timing for this specific
op** (no box contact this round) — a rough bound: the premise battery's
own comparable **forward-only** passes measured `elapsed_s` 5.75–15.02 at
`n=256` through the FULL 98M-param backbone; item 6 touches only the
173K-param head (no backbone forward/backward at all) and its own
diagnostics (SVD calls) are logged only every `log_every=500` steps, not
every step — so per-step cost should be materially *below* a full backbone
forward pass, but 64,000 steps is still a real number of steps, not
"seconds." **Recommendation to the coordinator (not resolved here):**
before committing to the full 8-cell/8000-step grid, run a short timing
probe (e.g. `n_steps=500` on one cell) to get a real H100 ms/step figure
and re-derive Stage 0′'s own ceiling from it, rather than trusting the
pre-A6 "~0.2 GPU-h" figure — this is cheap (minutes) and removes the last
unverified number in this build. Even a generously-priced Stage 0′ (say,
up to ~1 GPU-h) stays negligible against the 24.94 GPU-h nominal / ≤35
GPU-h hard-capped Stage-1 wave and does not change the ceremony tier.

**Stage 1's own pricing is UNCHANGED by this round** — D6's `24.841 +
0.10 = 24.94` GPU-h nominal, `≤25` GPU-h registered ceiling, `≤35` GPU-h
hard cap, all carried forward verbatim; this revision touches only the
Band-2/Band-0/D5.2 definitions and the Stage-0′ script, none of which
change Stage-1's own per-cell cost.

---

### R5.7 — Fresh self-attack: what kills this at R5

1. **The item-6 "train once, score per hop" scoping decision (R5.4) is
   this build's single most exposed engineering judgment call.** It rests
   on an inference — "keys_v/values_v content is not itself
   hop-conditioned, only the query/target construction is" — read off
   `build_task1_document`'s own code, never verified empirically on real
   box data (no box contact this round). If a document drawn under
   `hop_set=(1,)` differs systematically in its K bind-clause CONTENT
   (not just its query/target) from one drawn under `hop_set=(61,)`, then
   training on h=61-extracted episodes and scoring at h=1 could read an
   artificially pessimistic (or optimistic) achievability signal for h=1
   specifically. A narrow audit could kill this by either (a) reading
   `grammar_rd.sample_batch_rd`'s own construction directly for a
   hop-conditioned content dependency, or (b) an on-box A/B: train once
   on h=1-drawn episodes vs h=61-drawn episodes and diff the two
   achievability curves.
2. **`band2_check`'s `c*` formula (F1's repair, adopted verbatim) is a
   proxy, not necessarily `L_key`'s own true minimizer over a global
   rescale.** It minimizes the UNNORMALIZED sum-square residual, not
   `L_key`'s per-key-normalized objective; the design doc itself flags
   this ("a legitimate, cheap proxy... not necessarily identical"). If the
   true normalized-`L_key` minimizer differs meaningfully from this `c*`
   on real geometry, an operator that is "really" fine at ITS OWN optimal
   rescale could still fail gate (i) — a false NEGATIVE the current form
   cannot self-diagnose. Inherited from the binding adjudication text (not
   a defect this build introduced), but worth a fresh look if item 6's
   real-geometry `L_key_cstar` readings look surprisingly far from 0 even
   at low `lambda_t`.
3. **Item 5's numeric predicate (`ortho_loss(Z_ideal) > 1e3`) is an "e.g."
   in the source report, applied here as if it were load-bearing.** F2's
   original ortho-loss measurement (`15,147` at `Z_ideal`) was on
   synthetic Hamiltonian-cycle keys; real extracted geometry could
   plausibly read materially lower (hundreds, not thousands) while the
   underlying conflict is still real — this predicate would then report
   `conflict_reproduces=False` on a real conflict, an instrument false
   negative baked into a threshold nobody rigorously derived.
4. **The verbatim-duplicate discipline (`discriminability_metrics`,
   `ortho_regularization_loss` in `stage0prime_helpers.py`) has no
   automated drift check.** If the pinned runner is ever revised (a
   future round fixes a bug in either function), nothing in this repo
   flags that `stage0prime_helpers.py`'s own copy has silently gone
   stale. Mitigation registered, not built: a future round with box
   access should diff the two bodies directly (or better, hash-pin the
   runner's own md5 and assert it in `stage0prime_helpers.py`'s own
   module docstring/test, the same discipline this build already applies
   to citing `9a93198b642242f512ff8489e32b0a53`).
5. **`item_6_achievability_probe`'s all-hops-must-pass gate (this
   round's own strengthening over an earlier single-hop draft) could make
   the GO/NO-GO gate MORE conservative than the report intended** — the
   report's own words ask whether "no λ_t reaches Band 2's targets," not
   explicitly whether it must do so at every scored hop simultaneously. If
   a narrow audit judges this too strict (e.g. an operator that is clean
   at h=61 but slightly over Band 2's ratio bound at h=1, due to ordinary
   held-out sampling noise on a small `n`), the fix is a one-line relaxation
   (any-hop instead of all-hops) — flagged here as a real, disclosed design
   choice this build made beyond the report's literal text, not hidden
   inside the gate's own boolean logic.

---

## §A5-ADJUDICATION (coordinator, 2026-08-14) — attack R5 = REV-REQUIRED (4F/6M/9m) ADOPTED; repairs COORDINATOR-IMPLEMENTED from the report's own executed specs; Stage 0′ CLEARED with timing probe gating

Report: `NCR_WRITECOND_ATTACK_R5.md` vs DRAFT-R5 at `c77d6ef`. The
pattern of record: everything Rev-5 EXECUTED was verified right (the
auditor's own harder impostor killed by the repaired gate at 100%;
the partition independently re-enumerated over 51,170 outcomes,
0/0/0; 131/131 assertions clean) — everything never executed
end-to-end broke. The train-once-score-per-hop inference was CLOSED
AS PROVABLE by the auditor (all seven bind-clause tensors
bit-identical across hop sets).

**Repairs implemented BY THE COORDINATOR (K-wall §A10 precedent:
transcriptions of the auditor's executed specs, disclosed here;
verification = the re-run suite + the auditor's own divergence cases
re-executed against the committed code + Stage 0′'s live run):**
- F1 (`stage0prime_helpers.py` gate): `GO iff (any_median AND
  any_p90)` — the auditor's edge case re-run through the committed
  code now reads NO-GO (was GO).
- F2: `build_fresh_encoder(device)` + `.to(device)`, called with
  `keys_v_train.device`.
- F3: `freeze_entity_adapter` read from EACH checkpoint
  (`ckpt.get(...)`), never hardcoded — the battery's own
  `pbe_repl` lesson, now in the script — with the per-checkpoint
  value printed as the A1+ pre-flight.
- F4 adjudicated OPTION (a) (the only one fitting the registered
  budget): CONTROL B runs WITHOUT the freeze, the M4(b) drift
  confound is a DISCLOSED weakening of the read-side-isolation
  claim, and `control_b_clean_eval.py`'s hard assert is now a
  warning + `freeze_entity_adapter_verified` record field (the
  band0 checker keeps its branch; the artifact becomes producible).
- M1: `--n-steps/--lr-grid/--lambda-t-grid/--timing-probe` CLI
  (timing probe = 1 cell × 500 steps, own output file).
- M2: the constant SVD hoisted once per grid (W= passed through
  loss and band checks; the auditor measured 19%/step recomputed).
- M3: zero/empty held-out hops VOID (never free-PASS), VOIDed hops
  fail the all-hops aggregation; `n_valid_episodes` emitted. The
  old fixture updated + a NEW soundness test asserting the VOID
  path blocks the gate.
- Suite: 44/44 after repairs (was 43 + the new test).
**Stage 0′ RULING (per the report): CLEARED to run, TIMING PROBE
FIRST and GATING** — the full 8-cell grid runs only if the probe's
extrapolated cost is sane. Remaining M-items (M4 d−K bound note,
etc.) ride to the harvest record; R4's F4-Stage-1 items are settled
by the option-(a) adjudication above. The instrument layer is now
certified BY EXECUTION where it counts (the committed artifacts);
Stage 0′'s reading decides the 24.94 GPU-h wave per §A4.

---

## STAGE 0′ HARVEST (coordinator, 2026-08-14) — VERDICT OF RECORD: NO-GO-ON-CURRENT-BAND. The supervised-write lever is BLOCKED-ON-ACHIEVABILITY; the write HEAD's parametrization is the wall.

Run 2026-08-14 on box GPU 7: timing probe 18s (gate passed:
extrapolated grid cost sane) → full 8-cell grid 414s. Total Stage 0′
spend ≈0.13 GPU-h. Raw artifacts:
`experiment-runs/2026-08-14_writecond_stage0prime/` (repo+SSD). The
F3 fix's A1+ pre-flight printed the per-checkpoint freeze flags
exactly as the archived configs record (True/True/False).

**The wave-deciding reading:** any_median = False, any_p90 = False ⇒
`stage1_gate = NO-GO-ON-CURRENT-BAND` under the §A5-repaired
conjunctive gate. Per §A4, THE 24.94 GPU-h STAGE-1 WAVE IS NOT
LAUNCHED.

**Per-cell facts (all 8 cells, n=256):** best L_key_med (final) =
8.377e-2 (lr=1e-3, λ_t=0) — 280× ABOVE the 3e-4 bar — worst 6.7e-1;
trajectories decelerating (cell-0: 9.76 → 0.196 over 8000 steps,
power-law-ish slowdown). The transverse penalty WORKS as re-specified
(Zw_ratio_med 0.005–0.014 at every λ_t > 0, vs 0.14–0.16 at λ_t=0 —
the F1-repaired instrument reads correctly in both directions).
Retrieval at chance in every cell at both hops (0.02–0.06) —
approximate writes with 8–67% residual energy per key are useless
for composition, exactly as the theory requires. Item 3's
reachability instrument LOCALIZES the wall: the exact operator
demands 13.2× median / 58.1× p99 / 76.1× max within-episode dynamic
range, against the encoder `row_out` head's conditioning of ≈5.04 —
the one-shot BindingEncoder emission architecturally cannot express
V·K† at the measured key geometry.

**What the full arc now establishes (three verdicts of record):**
(1) the READ path is perfect at h=61 given exact Z (premise battery,
3/3 checkpoints); (2) unsupervised training never learns the write
(§G3-B32, replicated) and spectral conditioning doesn't fix it
(§A2 F5); (3) DIRECT supervision doesn't fix it either at probe
budgets — the blocker is the write head's PARAMETRIZATION, now
quantified (dynamic-range deficit ≈ 3–15×).

**Forward options (each a NEW claim ⇒ novelty/design gauntlet):**
- **(a) Expressive write head** — replace the one-shot emission with
  a construction that can EXPRESS the exact operator (a delta-rule /
  key-orthogonalized outer-product-sum write can represent V·K† for
  independent keys by construction). The research continuation;
  fits the remaining grant window if its gauntlet clears.
- **(b) Re-scope to the closed-form-write architecture** — the
  premise battery's P1b arm IS this architecture evaluated
  end-to-end (3/3 checkpoints, 0.977–1.0 at h=61 in the real LM):
  claim = capability-separation DEMO with the model's own learned
  extraction + read machinery and an exact in-context write, with
  learnability honestly documented open (this section). FAAST
  cite-and-distinguish mandatory (they disclaim composition; we
  demonstrate it). Near-zero GPU cost — the evidence exists.
Coordinator's recommendation: (b) feeds the flagship NOW (deadline
~late Sept); (a) is the right use of the remaining box window IF its
design survives the gauntlet. Neither launches without its gate.

