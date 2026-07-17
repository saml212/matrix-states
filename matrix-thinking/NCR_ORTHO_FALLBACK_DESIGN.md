# NCR ORTHOGONAL-WRITE FALLBACK — PRE-REGISTRATION

**Status: DRAFT, pre-attack.** Written to survive an adversarial Opus attack
round before any freeze. No GPU spend is authorized by this document alone.
Successor to `NCR_ORTHO_WRITE.md` (committed `cffc209`, amended `62a6fb6`,
built `3086dfa`), whose Newton–Schulz-polar (NS-polar) primary arm returned
**§9 VERDICT OF RECORD: FAIL** (Part A, K=32, Gate-0 dead 4/4 seeds) and whose
**§10 POST-FAIL CODE RE-AUDIT** (independent, read-only, 2026-07-17) found
**MECHANISM-CONFIRMED, not a bug**: a tight-spare (`d=K+1`) ill-conditioning
runaway — the scale-invariant cosine read exerts zero gradient pressure on
the write's overall scale or its one unconstrained spare direction, both
random-walk freely, and once the spare direction's singular value crosses
~1e-7 the NS forward stops orthogonalizing while the polar backward
Jacobian explodes ~1/σ_min, converting into a task-destroying clipped unit
step — an absorbing near-singular basin at K≥24 (escapable at K=8, per
§10.7's local repro, but not at target scale over 320K steps).

This document is **THE RETRY-OF-RECORD FOR GATE 1** — the ortho-write
verdict gate that both `NCR_KLADDER_DESIGN.md` §9 and
`NCR_REAL_LM_DESIGN.md` §9.1 are double/triple-gated on. Stated explicitly,
because it is load-bearing for the whole downstream program:

> `NCR_ORTHO_WRITE.md` §9's FAIL, taken alone, would resolve GATE 1 to its
> pre-registered NULL/FAIL branch **today** — both K-ladder and Real-LM fall
> back to K=15/d≈16 (the pre-ortho-write free-write "SCALES" regime,
> `NOVEL_ARCH_WATERFALL.md` §11.2), abandoning the K≥24 orthogonal-write
> mechanism and the h*≈40 realistic-depth headline. §10's own recommendation
> ("proceed to the §2 skew-symmetric parametrization... not more budget")
> is why that resolution has not yet been recorded: it is *this* experiment,
> not a shrug, and it must run before GATE 1 is closed out either way. **A
> WIN or PARTIAL verdict from Stage 1 below (at K=32, from any surviving
> arm) DIRECTLY DISCHARGES GATE 1** exactly as if the original NS-polar Part
> A had won — no separate re-approval needed; a build agent then substitutes
> the winning arm's name for "Newton–Schulz" in `NCR_KLADDER_DESIGN.md`
> §9's WIN/PARTIAL branch text and `NCR_REAL_LM_DESIGN.md` §9.1's WIN/PARTIAL
> branch text (a disclosed terminology edit, not a re-gauntlet). **If every
> arm here ALSO returns FAIL/NULL, GATE 1 resolves definitively to NULL/FAIL**
> and both downstream documents execute their *already pre-registered* K=15
> fallback branches — the honest terminal state, not a new decision made in
> the moment.

**Ceremony tier (CLAUDE.md doctrine).** Stage 0 alone (≤1.0 GPU-h worst
case) qualifies for the light <10 GPU-h / 1-audit-round tier and could be
fast-tracked. Stage 1 (≈70–104 GPU-h worst case, publication-bound — it is
the GATE-1 retry for the flagship ICLR-2027 spearhead) requires the full
multi-round adversarial gauntlet; this document is round 1 of that gauntlet.

**Coordinator steer incorporated (2026-07-17, verified against source before
drafting — see §3.1/§7/§8).** A parallel novelty-delta sweep
(`research/ncr_separation_grounding.md`, commit `ae41a15`) landed *during*
this draft and is incorporated live rather than left as a `[PENDING]` slot.
Its load-bearing, independently-corroborated ruling: **unscaled Cayley
`(I−W)(I+W)⁻¹` is not a safe primary arm** (Lezcano-Casado & Martínez-Rubio,
ICML 2019, arXiv:1901.08428, Prop. 3.2 — full-text verified by the sweep,
spot-checked here by a fresh `WebSearch`: skew weights blow up when the
target rotation has eigenvalue −1, and diagonal ±1 scaling "just relocates"
the singularity, does not remove it); **`expm(W−Wᵀ)` is the theoretically
preferred parametrization**, proven free of that specific blow-up. This
document promotes `expm` to **PRIMARY** and demotes Cayley to a **DEMOTED
comparison arm**, carrying the singularity analysis and an honest
expected-failure prediction (§3.1, §3.3, §9). This is a genuine mid-draft
revision, recorded here rather than silently absorbed.

---

## §1 HYPOTHESIS + FALSIFIABLE PREDICTION

**Hypothesis (one sentence).** A skew-symmetric-group orthogonal
parametrization — primarily the matrix exponential `Q=expm(W−Wᵀ)`, with
Cayley `Q=(I−W)(I+W)⁻¹` run as a demoted comparison arm — will train
through Gate-0 and recover realistic-depth (`h*≈40`) composition at
`K∈{24,32}` where the hard Newton–Schulz-polar projection failed, because
neither route ever inverts (or otherwise numerically requires) a matrix
whose smallest singular value can random-walk to zero, structurally
eliminating the *specific* §10 ill-conditioning-runaway mechanism — while
Cayley (only) remains independently exposed to a *different*, literature-
established blow-up when the target has an eigenvalue exactly −1, which
this task's K-cycle structure is shown below (§3.1) to require exactly.

**Falsifiable prediction.** An arm passing Gate-0 must ALSO show its
trained operator's spectral signature move toward orthogonal
(departure-from-normality → ~0, cond# → ~1) — mirroring §9.1's own use of
mechanistic corroboration as a check on the behavioral verdict. A WIN/PARTIAL
with a non-orthogonal spectral signature would itself be an anomaly requiring
explanation before being trusted.

**No calibrated odds are offered (unlike `NCR_ORTHO_WRITE.md` §7, which had
an in-silico post-hoc polar preview as a numerical anchor).** Cayley/expm
have never been run even once in-silico against this task. The closest
available anchor is §10.7's K=8 local CPU repro of the *original* NS-polar
mechanism escaping the trap at small scale — informative about the general
shape of ill-conditioning basins, uninformative about expm/Cayley's own
basins, which are structurally different objects. This is a disclosed
evidence gap, not a claim of confidence — flagged again in §10.

---

## §2 STAGE 0 — THE DECISIVE CHEAP GATE (§10.8's recommendation)

**Purpose.** §10.8 names, as the cheap confirmatory test *before* committing
to a full reparametrization, a **raw-Z conditioning fix** on the *existing*
NS-polar path: re-run one ortho K=24 cell with a σ_min floor. This isolates
whether the FAIL is fully explained by the spare-direction random walk
(§10.7), independent of which parametrization family Stage 1 ultimately
uses — Stage 0 patches the diagnosed mechanism directly rather than
sidestepping it via a different group parametrization.

**Exact intervention.** Insert one line before the existing pre-scale step,
leave everything else in `newton_schulz_polar` / `NCROrthoWriteModel.encode`
byte-identical:

```
Z_raw    = self.encoder(keys, values)          # UNCHANGED
Z_damped = Z_raw + eps * I                     # NEW — raw-Z conditioning floor
sigma_hat = detached_power_iteration(Z_damped, n_power=12)   # UNCHANGED call, damped input
X0 = Z_damped / sigma_hat
Q  = newton_schulz(X0, n_iter=40)              # UNCHANGED — do NOT bump n_iter (§10.8)
```

`eps = 1e-3` (fixed constant, not learned, not annealed — minimal-change
principle for a decisive attribution test). `n_iter=40`, `n_power=12` stay
at `NS_ITER_DEFAULT` / `NS_POWER_DEFAULT` exactly (`ncr_ortho_write.py`) —
§10.8 explicitly warns bumping `n_iter` *worsens* the backward explosion;
this intervention must not be confounded with that axis.

**Why `eps=1e-3` (grounded in §10's own measured numbers, not chosen
freely).** By Weyl's inequality for singular values, `|σ_i(A+E)−σ_i(A)| ≤
‖E‖₂`; for `E=εI`, `‖E‖₂=ε`, so adding `εI` perturbs every one of the K
signal-bearing singular values (empirically O(1) at convergence, §5's
healthy K24-free case: cond≈1.0) by at most `1e-3` — a ≤0.1% relative
disturbance, negligible. Along the near-null direction specifically, the
`εv` term dominates once the true component drops below `ε` (a standard
Tikhonov/ridge damping effect), giving an *effective* floor of order `ε` on
that direction. Repro 2 (§10.2) measured `‖dL/dZ‖_F≈24` (benign) for
`σ_min≥1e-6`, exploding to `4.6e7` only at `σ_min=1e-8` — i.e. the danger
zone is `≲1e-7`. `eps=1e-3` sits **~4 orders of magnitude above** that
threshold, and Repro 1b (§10.1) showed NS-40 orthogonalizes cleanly for
input cond up to `1e6`, degrading only at `1e7`+ — flooring σ_min at `1e-3`
against an O(1) σ_max keeps cond well inside that convergent regime with
large margin in both directions.

**Scope: n=1 seed, K=24, reduced step budget, ≤0.5 GPU-h.** At the measured
ortho_K24 rate (~3.3 h / 320K steps, §9.0), 0.5 GPU-h buys **~48,000 steps**
(≈0.6× the standard 1× 80K-step convention) — pinned exactly, not
"as many as fit." This is a directional smoke, not a statistically powered
Gate-0 verdict (Stage 1's n=4 delivers that for whichever arm(s) survive).

**Branch logic (pre-specified, both directions).**

- **PASS** (Gate-0 clears within 48K steps — `min_{h∈{1,2,3}} recovered_frac@0.9
  ≥ 0.9` AND `A_eff_rank ≥ 0.9·24=21.6`): confirms the ill-conditioning trap
  as the sole blocker for the original mechanism; the damped-polar
  construction is promoted to a **third Stage-1 arm** (full 4×-budget,
  K∈{24,32}, n=4 seeds — §3.4), using this exact `eps` as the pinned
  hyperparameter, no further tuning.
- **FAIL, same signature** (loss dips then re-collapses to ~1.0, matching
  §9.1's own dip-then-collapse curve): the floor was insufficient — SGD
  still pushes the spare direction's *effective* magnitude below the
  danger zone despite the additive floor (e.g. because the encoder can
  learn to counter-rotate the offending direction away from the identity
  basis the fixed `εI` protects, or because `1e-3` undersells the actual
  random-walk rate at K=24 over 48K steps). **One pre-authorized retry**:
  `eps=1e-2` (10× up), same 48K-step budget, ≤0.5 GPU-h.
- **FAIL, different signature** (flat/never-engaged loss from step 0, no
  dip at all — unlike any §9.1 curve): the damping itself broke something
  unrelated to the trap (`eps` too large, swamping the K signal-bearing
  modes despite the Weyl bound being small in the worst case — worth
  confirming empirically rather than trusting the bound alone). **One
  pre-authorized retry**: `eps=1e-4` (10× down), same budget.
- **If the pre-authorized retry ALSO fails** (either branch): conclude the
  §10 ill-conditioning-trap diagnosis, while mechanistically well-evidenced
  by the static/dynamic audit, is **not sufficient alone** to explain the
  FAIL at K≥24 — an incomplete diagnosis, recorded as such. Stage 1 proceeds
  **without** a damped-polar arm (2-arm grid: expm + Cayley only) — this was
  always going to run regardless of Stage 0's outcome (§3), so Stage 0's
  failure costs only its own ≤1.0 GPU-h, not schedule.

**Stage-0 total: ≤1.0 GPU-h worst case (initial attempt + one diagnostic
retry). Runs strictly BEFORE any Stage-1 spend.**

---

## §3 STAGE 1 — THE PARAMETRIZATION ARMS

### §3.1 The det-parity requirement — shared math, both structural arms

**Before pricing anything: does either candidate parametrization even reach
the target?** A single Hamiltonian K-cycle permutation matrix has
determinant `(−1)^(K−1)`. For **K=24 and K=32 (both even)**, `K−1` is odd,
so **det = −1** — the target entity operator is an *improper* orthogonal
matrix (reflection-type, in `O(d)\SO(d)`), not a rotation. Concretely: the
K-cycle's real block-diagonal (Schur) form is one trivial `+1` block
(`j=0`), `K/2−1` genuine `SO(2)` rotation blocks (`j=1..K/2−1`, eigenvalues
`e^{±2πij/K}`), and — because K is even — **exactly one isolated real
eigenvalue `−1`** (`j=K/2`, `e^{iπ}=−1`), living in a 1-dimensional
invariant subspace.

**Claim:** for *any* real skew-symmetric `W`, both
`Q=(I−W)(I+W)⁻¹` (Cayley) and `Q=expm(W)` (skew-exp) have **det(Q)=+1 for
every W** — both parametrizations, as literally specified, are confined to
`SO(d)` and can **never** represent the target's det=−1 exactly, at any
training budget.

- *Cayley:* `W` skew ⇒ eigenvalues are `0` or conjugate pairs `±iθ`. Cayley
  maps `λ↦(1−λ)/(1+λ)`: `0↦1`; each pair `±iθ↦` a conjugate pair whose
  product is `1`. `det(Q)=∏λ_i=+1` always.
- *skew-exp:* `det(expm(W))=e^{tr(W)}`, and skew-symmetric matrices are
  traceless (zero diagonal) ⇒ `det(Q)=e^0=1` always. (Cleaner proof, one
  line.)

**Consequence:** BOTH candidate arms, as specified in `NCR_ORTHO_WRITE.md`
§2's original fallback naming, are mathematically **incapable** of reaching
the K-cycle target at K∈{24,32} — not a trainability risk, a *reachability*
impossibility. Left unfixed this guarantees FAIL/NULL for both arms
regardless of training quality, wasting the entire Stage-1 budget on an
experiment that could never have succeeded. This is exactly the "compute it
on paper, 10 minutes, no exceptions" checklist item, and it is being
discharged HERE, at pre-registration time, not left for a build agent to
discover mid-run.

**Required, cheap, no-new-learned-parameters fix (build-blocking, applies
to both arms):** compose a single FIXED (non-learned) odd-determinant
reflection with the learned `SO(d)` factor.

- `Q = R · expm(W)`, `R` any fixed diagonal `±1` matrix with `det(R)=−1`
  (e.g. `R=diag(−1,1,…,1)`). Since `expm: so(d)→SO(d)` is **surjective**
  (standard Lie-theory fact for the compact connected group `SO(d)` — also
  the theoretical content of Lezcano-Casado & Martínez-Rubio 2019's own
  "trivialization," §7), `R·Q_target ∈ SO(d)` (verify: `det(R·Q_target) =
  det(R)·det(Q_target) = (−1)(−1) = +1`) is *exactly* reachable by some `W`
  — surjectivity guarantees existence, not just asymptotic approach.
  `Q=R·expm(W)=R·R·Q_target=Q_target` (`R²=I`). **Complete, exact closure,
  no residual gap.**
- `Q = D · (I−W)(I+W)⁻¹`, `D=diag(d_1,…,d_d)`, `d_i∈{±1}`. Per Helfrich et
  al. 2018's scoRNN theorem (`research/ortho_write_grounding.md` §1,
  VERIFIED arXiv:1707.09520), this parametrizes *exactly* the orthogonal
  matrices with `|{i:d_i=−1}|` eigenvalues equal to −1. The target has
  **exactly one** such eigenvalue ⇒ `D` with exactly one `−1` entry (e.g.
  `diag(−1,1,…,1)`) is the theorem-backed, general (not lucky-alignment)
  fix for *reachability*.

**But reachability is not the whole story for Cayley — this is where §3.1's
math and the verified literature (§7/§8) diverge sharply between the two
arms.** `σ_min(I+W)≥1` for **any** real skew `W`, unconditionally (proof:
`W` skew ⇒ normal ⇒ `I+W` normal ⇒ singular values of `I+W` equal
`|eigenvalues| = |1+iθ_j| = √(1+θ_j²) ≥ 1`). **The `(I+W)` inversion inside
Cayley can therefore never become singular — the *exact* §10 mechanism
(NS forward failing to converge on a near-singular raw matrix, backward
exploding ~1/σ_min) cannot recur through Cayley's inversion route.** This is
a genuine, correct, and distinct finding from the one below — keep them
separate, an attack round will probe exactly this distinction.

**A separate, independently-verified, and more serious problem exists for
Cayley specifically.** Lezcano-Casado & Martínez-Rubio (ICML 2019,
arXiv:1901.08428, Prop. 3.2 — full-text verified by the 2026-07-17 novelty
sweep, `research/ncr_separation_grounding.md`, spot-checked here via a fresh
`WebSearch` confirming the paper's identity/topic): **the Cayley map has a
hard singularity as `θ→∞` when approximating a target with eigenvalue −1**
— reaching the boundary requires `‖W‖→∞`, i.e. gradient descent pursuing
that exact target drives the PARAMETER itself to diverge, even though
`I+W` stays technically invertible at every finite step. Diagonal ±1
scaling (Helfrich, above) **"just relocates" this singularity — it does
not remove it** (per the sweep's characterization of Prop. 3.2). The
target this task trains toward has EXACTLY one eigenvalue −1 (this
document's own §3.1 derivation) — the K-cycle structure is precisely the
"order-2 composed state" the sweep's ruling flags as triggering the
blow-up. **This is a mechanistically distinct pathology from §10's**:
not a matrix-inversion collapse, but unbounded parameter growth chasing an
asymptotically-approached, never-exactly-reached boundary of the Cayley
map's image. Both are real; they are not the same trap, and Cayley is
vulnerable only to the second.

**Honesty boundary (do not overclaim).** The `D`-reflection fix closes
*reachability* (Helfrich's theorem is exact, general, not alignment-lucky).
It does **not** by itself establish whether the *residual* `SO(d)` target
`D·Q_target` sits far from or near Cayley's own −1-eigenvalue boundary —
that is an open, embedding-specific question. **Pre-launch audit action
item:** compute `D·Q_target`'s eigenvalues in closed form for the actual
entity embedding (CPU-only, minutes) before launching Cayley cells, rather
than assuming either direction.

**skew-exp has no analogous blow-up.** Per the sweep, expm's own singular
set is "eigenvalue gaps at nonzero multiples of `2πi`" — a
periodicity/gauge-redundancy issue (multiple `W`s map to the same `Q`, e.g.
`θ` and `θ+2π`), not a norm-divergence-to-reach-a-boundary issue. Confirmed
independently by this document's own eigenvalue-mapping proof: `expm`
reaches a real eigenvalue exactly −1 at **finite** `θ=π` per 2D rotation
block (a genuine `SO(2)` rotation by `π`, giving a *paired* `(−1,−1)`
diagonalizable block, consistent with `det=+1` preservation) — no
parameter ever needs to diverge. **This is why `expm` is promoted to
PRIMARY and Cayley demoted to a comparison arm carrying this analysis.**

### §3.2 PRIMARY arm — skew-exp / matrix exponential

```
W  = raw_skew_param(encoder(keys, values))   # any d×d, then W ← W − Wᵀ
Q  = R @ torch.matrix_exp(W)                 # R = fixed diag(-1,1,...,1)
```

- **Structural elimination of the §10 trap:** no matrix inversion anywhere
  in the forward path; `expm` is defined and well-conditioned for *every*
  real `W`, however large its norm (unlike Cayley, no boundary to
  asymptotically chase for THIS task's target — §3.1).
- **Cost/gradient path.** `torch.matrix_exp` is natively differentiable
  (Fréchet-derivative backward, available since PyTorch 1.7 — framework
  capability, not a literature claim). Implemented via scaling-and-squaring
  + Padé approximation; at `d∈{24,25,32,33}` (tiny matrices) this is a
  handful of `d×d` matmuls per forward/backward, the same order of magnitude
  as NS-polar's 40-iteration loop — expect wall-clock **comparable to or
  slightly above** the measured NS-polar rate (~3.3 h/cell), not
  qualitatively different, especially given the established finding
  (`NCR_KLADDER_DESIGN.md`, FLOP-ratio scaling refuted at small K) that this
  regime is **kernel-launch/small-batch overhead-bound, not compute-bound**
  — differences between parametrizations at this tiny `d` are predicted to
  compress toward the overhead floor, to be CONFIRMED (not assumed) by an
  early canary cell (§3.5).
- **Param/FLOP delta vs. the failed NS-polar arm:** `W` has the same
  `d(d−1)/2` free parameters as any `d×d` skew generator (comparable to
  NS-polar's raw `Z` encoder output count); `R` is a fixed buffer, zero
  learned parameters. No material param-count change from the original
  `NCROrthoWriteModel` (`≤~185K/cell`, §6 of `NCR_ORTHO_WRITE.md`).

### §3.3 DEMOTED comparison arm — Cayley

```
W  = raw_skew_param(encoder(keys, values))   # any d×d, then W ← W − Wᵀ
Q  = D @ torch.linalg.solve(I + W, I - W)    # D = diag(-1,1,...,1), one -1 entry
```

- Runs at the **same grid** as the primary arm (§3.5) — not dropped, per the
  coordinator's ruling — but with an explicitly lowered PRIOR (§9) and an
  extra required diagnostic: **log `‖W‖_F` over training** (cheap, one
  scalar/step) so a FAIL can be characterized. A Cayley FAIL that shows
  `‖W‖_F` diverging (correlating with the trained operator approaching the
  target's −1-eigenvalue component) is a **publishable confirming instance**
  of Lezcano-Casado & Martínez-Rubio's Prop. 3.2 in a genuinely new
  application (in-context-written fast-weight operators, never their
  setting) — not a wasted run. A Cayley FAIL *without* that signature would
  be the more alarming outcome (an unexplained failure mode, requiring its
  own re-audit before trusting Stage 1's other arms).
- **Cost.** One `d×d` solve per forward (`torch.linalg.solve`, LU-based,
  `O(d³)`, cheaper than NS-polar's 40-iteration loop in raw FLOPs) — predict
  wall-clock at or below the free-write baseline rate (~2.3–2.8 h/cell),
  though (as with expm) the overhead-bound regime may compress this;
  confirm via canary, do not assume.
- If Cayley unexpectedly WINS despite the predicted elevated risk: report it
  fully and honestly — the demotion sets *expected priority and sequencing*
  (§5), not a pre-judged verdict.

### §3.4 Conditional third arm — damped-polar (from Stage 0)

Promoted only if Stage 0 (§2) PASSES. Identical construction to the
original `NCROrthoWriteModel.encode` plus the `+eps*I` floor at the pinned
`eps` that cleared Gate-0 in Stage 0. Full 4×-budget run, K∈{24,32}, n=4
seeds — mirrors Part A exactly, same as the other two arms. No det-parity
fix needed (still a raw-matrix polar factor, `det(Q)=sign(det Z)` is free
to flip through training, unlike Cayley/expm's structural `SO(d)`
confinement — §3.1).

### §3.5 Grid, cost, saturation packing

**Grid:** `{arm} × K∈{24,32} × n=4 seeds`, mirroring Part A's structure
exactly. `arm ∈ {expm, cayley}` always (16 cells); `+damped-polar` if Stage
0 passes (24 cells). **Part B (the structured-operator discriminator) is
NOT re-run this wave** — its own free-bank baseline never trained in the
original run (§9.2, compounded null unrelated to the ortho-constraint), so
it needs its own calibration fix before it can serve its mod-K-trap-safe
role again; deferred, not abandoned (flagged as an open item, §10).

**Per-cell cost pin:** `4.3 GPU-h` (conservative — the upper end of the
measured `2.3–4.3 h/cell` precedent, `NCR_ORTHO_WRITE.md` §9.0/CEILING
AMENDMENT: free ~2.35 h, NS-polar-ortho ~3.3 h, discriminator ~3.4–4.3 h).
Pinning conservative after the CEILING AMENDMENT episode (a first per-cell
estimate was ~2× optimistic against the measured on-box rate) — this
history is exactly why the pin here is deliberately pessimistic rather than
reusing the lower end.

**Saturation packing (doctrine: 100% utilization, not occupancy — measured
precedent these cells drew ~68–77% SM on H100).** Memory is never the
constraint: `≤185K` params/cell, order-of-magnitude estimate `<100 MB/cell`
fully loaded (params + Adam states + `TRAIN_BATCH=256` activations at
`d≤33`) against 80 GB/GPU — inherits §6's "never the constraint" finding.
Propose **N=2 cells/GPU** (8 GPUs → 16 concurrent slots): single-cell SM
draw (~70–77%) doesn't saturate solo, and 2-packed pushes aggregate demand
to ~140–154% of one GPU's capacity — the natural "fills it without wild
oversubscription" choice, matching the doctrine's packing intent without
guessing at a more aggressive factor the memory math doesn't require.
**Contention-priced abort ceiling: 12.0 h** (2× the amended 6.0 h solo
runaway guard) — generous headroom given the solo utilization has slack
before a true 2× slowdown, but explicitly a **guard, not a prediction**.
**Mandatory canary, per the "saturation-packing is a pre-launch design gate"
doctrine:** launch ONE packed pair (2 cells, 1 GPU) first, measure actual
wall-clock and realized per-process SM, BEFORE committing the full grid —
if realized aggregate SM leaves headroom, N may be raised to 3/GPU
opportunistically (measured, never assumed); if contention is worse than
predicted, fall back toward solo execution for the remaining cells.

**Predicted wave structure.** 16-cell (2-arm) grid: one wave, all cells fit
16 packed slots simultaneously. 24-cell (3-arm) grid: wave 1 fills all 16
slots, wave 2 (8 remaining cells) runs **solo** across the 8 GPUs
(unpacked — faster per-cell than a second packed wave for a
sub-GPU-count remainder).

---

## §4 VERDICT BANDS — re-pinned, structure inherited verbatim from §4

Same Gate-0 precondition, same thresholds, same band names as
`NCR_ORTHO_WRITE.md` §4 — only the `{arm}` label generalizes. Evaluated
independently per `arm × K` cell (n=4 seeds).

**Gate-0 precondition (unchanged):** `min_{h∈{1,2,3}} recovered_frac@0.9 ≥
0.9` AND `mean A_eff_rank ≥ 0.9·K`. A cell failing Gate-0 is DEAD and
cannot be scored on far depth.

- **WIN** (K=32 primary): arm's median `rec@0.9` at `h*=40 ≥ 0.9` across
  Gate-0-passing seeds, AND the free-write baseline reads **< 0.5 at
  h=40**. **Re-pinned to the FRESH, in-harness §9.1 measurement** (an
  upgrade over the original §4's archived-only option): `free_K32_s{0..3}`
  all read exactly **0.000** at h=40/h=20/h=29/h=61 — stronger, zero-extra-
  cost evidence than the archived baseline alone. Mechanistic corroboration
  (unchanged, arm-agnostic — measures the trained operator's own spectral
  properties regardless of which parametrization produced it): departure-
  from-normality ≤0.02, cond# ≤~2, min|λ|/c* ≥0.9.
- **PARTIAL:** median `rec@0.9` ≥0.9 at `h∈{20 OR 29}` but <0.9 at h=40.
- **NULL:** median `rec@0.9` <0.9 at every depth ≥12, Gate-0 still passes.
- **FAIL:** Gate-0 DEAD (in-dist <0.5) in ≥3/4 seeds.

**K=24 cells: same bands, SECONDARY/robustness role** — mirrors
`NCR_ORTHO_WRITE.md` §4's own K=32-primary framing. K=24 alone does not
drive the GATE-1 discharge decision; K=32 is the decisive cell.

**Arm-selection rule (NEW — multiple arms now compete for one downstream
decision).** If ≥1 arm clears WIN or PARTIAL at K=32: select the single
BEST arm (highest far-depth median `rec@0.9` at h=40; tie-break by lower
measured per-cell cost; tie-break by tightest mechanistic corroboration) as
"the" winning parametrization forwarded to `NCR_KLADDER_DESIGN.md` /
`NCR_REAL_LM_DESIGN.md` GATE 1. Record which arm won; a build agent
substitutes that arm's name for "Newton–Schulz" in both documents' GATE-1
WIN-branch text (disclosed edit, not a re-gauntlet). **Expected priority
under a tie is `expm` > damped-polar > Cayley** (per §3's risk ordering),
but this is a tie-break only — an unambiguously better Cayley result is
reported and used, not suppressed.

**If ALL arms FAIL/NULL at K=32:** GATE 1 resolves definitively NULL/FAIL —
proceed to the already-pre-registered K=15 fallback in both downstream
documents (§9.1 of each). No new design work required at that point.

---

## §5 LEDGER — GPU-h, ABORT-ON-COST, sequencing

| Stage | Worst-case GPU-h | Gate |
|---|---|---|
| Stage 0 (raw-Z conditioning smoke, K=24, n=1, ≤2 attempts) | ≤1.0 | Runs first, always |
| Stage 1, 2-arm (expm + Cayley, K∈{24,32}, n=4) | 16 cells × 4.3 h = **68.8** | Runs regardless of Stage 0 |
| Stage 1, 3-arm (+ damped-polar) | 24 cells × 4.3 h = **103.2** | Only if Stage 0 PASSES |
| **Total worst case, 2-arm branch** | **≈69.8 GPU-h** | |
| **Total worst case, 3-arm branch** | **≈104.2 GPU-h** | |

Both totals sit comfortably inside a single ~192 GPU-h/day operative budget
window on the 8-GPU cluster (CLAUDE.md Hardware), especially wall-clock-
reduced via §3.5's packing.

**ABORT-ON-COST discipline (mirroring the CEILING AMENDMENT precedent —
being ~2× optimistic on a first estimate is the documented failure mode to
guard against here):**

1. Stage 0 runs to completion or its one pre-authorized retry — never more
   (§2). No "just a bit more budget" extension on Stage 0 regardless of how
   close it looks to clearing.
2. **Early canary, mandatory before either the packing plan or the 4.3 h/
   cell pin are trusted at scale:** launch cell 1 (first arm, K=24, seed 0)
   SOLO; compare measured wall-clock against the 4.3 h pin. If measured >
   6.0 h (the amended solo runaway ceiling) for that single cell, **HALT and
   re-price before continuing** — exactly the CEILING AMENDMENT's own
   correction pattern, not a hypothetical.
3. **Sequencing for early information, not just cost control:** run all
   arms' K=24 groups (4 seeds each) BEFORE any K=32 cells. K=24 is the
   cheaper, smaller-scale regime; per §9.1's own secondary finding, if an
   arm fails at the *easier* K=24 it is unlikely (though not certain — run
   K=32 regardless, it remains the primary/decisive cell) to succeed at
   K=32. This lets the coordinator triage compute toward the surviving
   arm(s) before committing the full K=32 spend, an informed and disclosed
   abort-on-cost decision, never a blind one.
4. **Priority under compute pressure:** if the cluster cannot run the full
   grid concurrently, `expm` cells are prioritized over Cayley's (§3's risk
   ordering) — Cayley may be delayed, never silently dropped without
   recording the decision.
5. Packed-cell wall-clock is measured via the mandatory 1-pair canary
   (§3.5) before the full packed grid launches; the 12.0 h packed ceiling is
   a guard, re-priced if the canary shows something different.

---

## §6 INHERITED INFRASTRUCTURE — reused verbatim, not re-derived

`NCR_ORTHO_WRITE.md` §9.0's integrity check found the HARNESS clean: frozen
bands intact against `cffc209`, run-script md5 match, 0 tracebacks / 0 OOM /
0 aborted-budget across all logs, all 24 cells `COMPLETED`, all 24
`blank_out.passed=True` (P=1 bottleneck holds — read is bit-identical under
raw-input corruption, grad w.r.t. raw inputs exactly zero), all 16 Part-A
cells carry `axis_c_lock_sha256`, both archive mirrors md5-verified. None of
this is specific to the NS-polar arm — it instruments `encode()`'s *output*,
not its internals. This fallback reuses the SAME inheritance pattern
`NCR_ORTHO_WRITE.md` §2 established (`NCROrthoWriteModel(NCREarlyLNModel)`
overriding ONLY `encode()`): a new `NCROrthoFallbackModel` (or an `arm=`
dispatch inside the existing class) overriding ONLY `encode()` per §3.2/
§3.3/§3.4, with `forward()`, `eval_read()`, `z_dump`, `ncr_spectral`,
`axis_c_lock`, `blank_out_check`, `eval_cell`, and the mod-K discipline
(`ncr_task._gen_grid`, novel-residue enforcement) ALL INHERITED UNCHANGED.

- **Mod-K discipline:** unchanged — the realistic ladder `{5,12,20,29,40,61}`
  (`§3` of `NCR_ORTHO_WRITE.md`) and its novel-residue enforcement carry over
  verbatim; depth semantics don't depend on which orthogonal parametrization
  produced the operator.
- **Blank-out test, axis_c_lock:** unchanged instrumentation, attaches to
  the projected `Q` exactly as it attached to the NS-polar `Q` before.
- **Blind-run / record-before-read protocol:** THIS document's WIN/PARTIAL/
  NULL/FAIL bands (§4) are frozen before any Stage-1 GPU spend; no band
  moves after seeing results. The runner emits raw per-cell metrics
  (`recovered@h`, spectral diagnostics, `‖W‖_F` for Cayley) and **NO
  verdict** — a separate blind-assess step applies §4, exactly mirroring
  `NCR_ORTHO_WRITE.md` §6/§9's own discipline.
- **Resume-safety, tmux+supervisor pattern:** new runner script (e.g.
  `ncr_ortho_fallback.py`) reuses the existing `ncr_ortho_write.py` runner
  scaffolding (`--ceiling-gpuh` guard, per-cell JSON output, skip-if-
  COMPLETED resume logic), launched via `tmux new-session -d` with a
  self-healing supervisor loop (CLAUDE.md hard rule) — differs from the
  original runner ONLY in `encode()`'s arm dispatch.

---

## §7 PRIOR ART

- **Arjovsky, Shah & Bengio (2016), "Unitary Evolution RNNs."**
  arXiv:1511.06464, ICML 2016. VERIFIED (`research/ortho_write_grounding.md`
  §1). Constrains the *recurrent transition matrix*, not written content —
  category (a) in that memo's novelty framing, motivation-precedent only.
- **Helfrich, Willmott & Ye (2018), "Orthogonal RNNs with Scaled Cayley
  Transform" (scoRNN).** arXiv:1707.09520, ICML 2018. VERIFIED. Established
  that unscaled Cayley cannot represent eigenvalue −1 and proposed the
  diagonal ±1 scaling this document's §3.1 uses for *reachability* — but
  Lezcano-Casado & Martínez-Rubio (below) show that scaling does not remove
  the training-time blow-up, only relocates it. Cite both, precisely
  differentiated: Helfrich fixes *where you can end up*; Lezcano-Casado
  shows scaling doesn't fix *how you get there*.
- **Lezcano-Casado & Martínez-Rubio (2019), "Cheap Orthogonal Constraints in
  Neural Networks" (expRNN).** arXiv:1901.08428, ICML 2019 (PMLR v97).
  **VERIFIED twice independently**: full-text by the 2026-07-17 novelty
  sweep (`research/ncr_separation_grounding.md`, Prop. 3.2 content) and by
  a fresh `WebSearch` in this session (paper identity, authors, venue, expRNN
  construction all confirmed). The load-bearing citation for §3.1's Cayley-
  vs-expm asymmetry: Cayley has a proven −1-eigenvalue blow-up; the
  exponential-map trivialization does not.
- **MuonSSM (Nguyen, Vo, Vo, Nguyen & Pham), ICML 2026 Oral.**
  arXiv:2606.30461. VERIFIED (full-text, `research/ortho_write_grounding.md`
  §4 and `research/ncr_separation_grounding.md`'s 2026-07-16 re-sweep).
  Closest prior art to the orthogonalized-*fast-weight-write* strategic
  move — but orthogonalizes RANK-1 KV injections with a single quintic NS
  step for stability, not full-rank `d×d` operators for composition-
  exactness; its "O(log L)" is Blelloch associative-scan TRAINING
  parallelism, not query-time O(log h) reads — a conflation to pre-empt
  explicitly in any writeup.
- **Orthogonal Self-Attention (Zhang & Martens), Feb 2026.**
  arXiv:2602.05996. VERIFIED (2026-07-17 sweep + independent `WebSearch`
  this session: confirmed authors, date, mechanism — matrix exponential of
  a skew-symmetric matrix computed from query-key content, giving an
  orthogonal attention matrix, to fix rank collapse in skipless
  Transformers). **The closest mechanism-level neighbor to this document's
  Stage-1 primary arm**: in-context content → skew-symmetric generator →
  `expm` → orthogonal matrix — but single-use per forward pass (token
  mixing within one attention layer), **never persisted, no matrix-power
  read**. Cite alongside MuonSSM as complementary nearest neighbors (one
  has the persistence/fast-weight axis, the other has the expm-
  parametrization axis; NCR's combination of both, plus O(log h) reads, is
  the searched-unclaimed combination — §8).
- **Low-Rank Orthogonalization for Large-Scale Matrix Optimization** (He,
  Deng & Lu). arXiv:2509.11983. VERIFIED (2026-07-17 sweep + independent
  `WebSearch`: confirmed real, on-topic — Newton–Schulz orthogonalization
  stability and singular-value behavior in the Muon-optimizer context).
  Forward-only optimizer-update instability, not a differentiated-through
  write-path — but corroborates §10's diagnosis as a **confirming instance
  of established NS small-σ instability**, not a novel discovery.
- **projUNN (App. G.3).** arXiv:2203.05483. VERIFIED (2026-07-17 sweep,
  full-text). Hard projection differentiated-through a training loop
  accumulates drift; a tangent-space variant is stable. Further
  corroboration of the projection-vs-parametrization framing (§8) and a
  noted **future fallback-fallback** if both Stage-1 structural arms fail —
  out of scope for this document's Stage 1, flagged for the record (§10).
- **EΔ-MHC-Geo** (arXiv:2605.06729), built on **Deep Delta Learning**
  (Zhang, Liu, Wang, Gu; arXiv:2601.00417). VERIFIED-as-real (2026-07-17
  sweep). Data-dependent Cayley parametrization on the *depth* axis
  (hyper-connections) — a depth-wise mirror of this document's Stage-1
  Cayley arm, differentiated (different axis: network depth, not
  fast-weight write content).
- **RotRNN** (arXiv:2407.07239) — **withdrawn; in-context-ness UNVERIFIED,
  do not cite as in-context** (explicit flag from the 2026-07-17 sweep,
  carried forward here verbatim rather than re-litigated).
- **Path Development Network** (arXiv:2204.00740) — O(h) Lie-group
  composition; "covered bucket," no new differentiator beyond what
  DeltaProduct/RWKV-7 already establish about O(h)-vs-O(log h).
- **Explicitly excluded as false positives** (2026-07-17 sweep): arXiv
  2605.10970 and 2601.15313 — "orthogonality" in both papers means key
  *separation*, not operator parametrization; not relevant here.

---

## §8 NOVELTY-DELTA SWEEP — LANDED, incorporated (not [PENDING])

The parallel novelty-delta sweep this document was drafted to leave a slot
for **landed during drafting** (`research/ncr_separation_grounding.md`,
"2026-07-17 novelty-delta sweep" section, commit `ae41a15`) and is
incorporated directly into §3.1/§7 above rather than left open. Its two
headline rulings, both already threaded through this document:

1. **Q1 (combination novelty): UNCLAIMED.** The combination of
   (a) in-context-written operator content, (b) Cayley/expm orthogonal
   parametrization, (c) persisted for O(log h) repeated-squaring reads is
   not found anywhere in the searched literature. Closest neighbors: OSA
   (parametrization facet, unpersisted) and MuonSSM (persistence facet,
   rank-1/magnitude-only) — genuinely complementary, neither is a scoop.
2. **Q2 (projection-vs-parametrization trainability): ESTABLISHED THEORY.**
   §10's FAIL/MECHANISM-CONFIRMED finding is a **confirming instance** of
   already-published theory (Lezcano-Casado & Martínez-Rubio 2019;
   arXiv:2509.11983; projUNN App G.3; classical polar-Fréchet-derivative
   `~1/(σ_i+σ_j)` results, Higham) applied in a genuinely new setting
   (differentiated-through fast-weight writes, never studied by any of
   those papers) — **frame this way in every paper/report that cites §10,
   never as a novel discovery of the instability itself.**

**Design ruling adopted (load-bearing, already applied throughout §3):**
Cayley is NOT a safe primary arm; `expm(W−Wᵀ)` is the theoretically
preferred parametrization. Cayley remains as a demoted comparison arm
carrying the singularity analysis (§3.1) and the mitigation/expected-
failure framing (§3.3).

**Pre-submission flags carried forward from the sweep, unresolved, noted
for whoever writes the eventual paper:** body-level re-reads of OSA
(2602.05996) and EΔ-MHC-Geo (2605.06729) — both too recent in the sweep's
own assessment for abstract-only characterization to be fully trusted for
a submitted claim.

---

## §9 RISK REGISTER

**What kills each arm:**

- **expm (primary):** (1) The det-parity fix (`R`, fixed reflection) is
  necessary but its cost/gradient-path interaction with the rest of the
  training signal is untested at this task — a first-of-its-kind
  application, not yet run even once; (2) `torch.matrix_exp`'s
  scaling-and-squaring backward, while bounded (no divergence risk per
  §3.1), could still be slower or noisier than expected at the measured
  overhead-bound regime — confirm via canary, don't assume the FLOP
  argument transfers to wall-clock; (3) if Gate-0 fails here too, that
  would be the most informative possible negative result (the parametrization
  literature's own preferred choice failing would undercut the entire §10
  "parametrization, not projection" framing) — worth a fast, dedicated
  re-audit before accepting a FAIL at face value, mirroring §9.4's own
  "a code bug cannot be excluded from behavioral data alone" caution.
- **Cayley (demoted comparison):** the Lezcano-Casado & Martínez-Rubio
  blow-up (§3.1/§3.3) is the headline risk, specifically likely to trigger
  here because the target provably has an eigenvalue exactly −1 (this
  document's own derivation). A FAIL with a diverging `‖W‖_F` trace is a
  publishable confirming instance, not wasted compute — but a FAIL
  *without* that signature would be unexplained and should trigger its own
  mini re-audit before the arm is written off as "the predicted mechanism."
- **damped-polar (conditional third arm):** inherits the ORIGINAL NS-polar
  arm's own risk profile minus the specific spare-direction pathology §10
  diagnosed — i.e. it is a patch, not a structural fix, so any *other*
  route to the same ill-conditioned basin (not mediated by the spare
  direction specifically) would still be live. Stage 0's pre-authorized
  retries (§2) are the cheap test of whether the single floor generalizes;
  if it only works at the specific `eps`/K=24 combination tested in Stage
  0 and not at K=32's full budget, that is itself informative (the trap's
  severity scales with K, consistent with §9.1's own K24-vs-K32 secondary
  observation) and should be reported as a scale-dependent partial fix, not
  silently extrapolated.

**Seed-variance precedent (the §2.35-style pattern, `CAPABILITY_SEPARATION_
DESIGN.md` §2.35, `NCR_REAL_LM_DESIGN.md` §N1 R3).** The S₅ bridge-cell
precedent found a ~1-in-5 catastrophic (rank-deficient-basin) seed rate,
measurable only at n≥5 — n=3 or n=4 draws can read a genuinely-mixed
population as falsely clean OR falsely dead. **This document's Stage-1 n=4
per cell is the SAME n as the original (failed) `NCR_ORTHO_WRITE.md` Part A
run** — adequate for a clean WIN/FAIL (§9.1's original result was 4/4
unanimous in both directions, no ambiguity), but if a NEW arm returns a
SPLIT result (e.g. 2/4 or 3/4 seeds clearing Gate-0), that is exactly the
regime §2.35 warns is under-sampled at n=4 — the coordinator should not
treat a split result as a clean verdict without at least considering the
§R3 "raise to n=5 + catastrophic-seed disposition clause" pattern before
scoring it, rather than mechanically applying the ≥3/4 FAIL threshold to a
population that may need more seeds to characterize honestly. Flagged as an
open question for the attack round (§10) rather than pre-committing extra
budget speculatively.

**What NULL/FAIL (all arms) means for the mid-Aug flagship schedule
(honest).** If Stage 1 returns FAIL/NULL for every arm run, GATE 1 resolves
definitively to its NULL/FAIL branch and BOTH `NCR_KLADDER_DESIGN.md` and
`NCR_REAL_LM_DESIGN.md` execute their *already fully specified* K=15/d≈16
fallback (no new design work needed — §9.1 of each document is written and
ready). This is **not** a program-ending result: (1) `NCR_REAL_LM_DESIGN.md`
§N1's own claim-restructure ruling (R1, ~line 3401) already demoted the
far-depth K=32 crack from the sole flagship headline to supporting
evidence — the exactness-by-construction + O(log h) access-complexity
claims remain defensible at K=15's lower ceiling; (2) Axis A / Task 2 (the
S₅-generator bridge cell, GATE 2) is **explicitly independent of GATE 1**
(`NCR_REAL_LM_DESIGN.md` §9.2) and is entirely unaffected by any outcome
here. The cost of a total FAIL is **schedule and ceiling** (≈1–2 days
wall-clock spent here, likely on the box in parallel with other queued
work per the on-box queue directive; K=15's lower `h*` instead of K=32's
40), not the mid-Aug deadline itself. **Recommended timebox (new,
doctrine-consistent — "dead directions stay dead" applied to calendar
time, not just GPU-h):** if Stage 1 has not resolved to a WIN/PARTIAL for
at least one arm at K=32 within **~3 days wall-clock** (generous headroom
for the packed grid plus one contingency canary-driven re-price), the
coordinator should trigger the pre-registered K=15 default rather than open
a third fallback design — this fallback gets one well-resourced attempt,
not an open-ended pursuit.

---

## §10 STATUS + OPEN QUESTIONS FOR THE ATTACK ROUND

- [x] §1–§9 pre-registered (this commit, record-first, before any tunable
      code for this fallback).
- [ ] Independent code audit (fresh agent) of the `expm`/Cayley `encode()`
      overrides + the Stage-0 `eps`-floor patch, BEFORE any GPU spend —
      matching `NCR_ORTHO_WRITE.md`'s own gauntlet shape (pre-registration →
      build → independent audit → pre-launch resource/placement audit →
      runner).
- [ ] Pre-launch audit: (a) the §3.5 packing canary; (b) the §3.1
      closed-form eigenvalue check on `D·Q_target`/`R·Q_target` for the
      ACTUAL entity embedding (cheap, CPU-only — do not launch Cayley/expm
      cells on the unverified assumption that the reflection fix's
      reachability proof transfers cleanly to the specific code path).
- [ ] Stage 0 runner + smoke.
- [ ] Stage 1 runner (arm dispatch), smoke, then the gauntlet.

**Open questions this draft flags for the attack round, rather than
resolving unilaterally:**

1. Is `eps=1e-3` actually the right order of magnitude, or should Stage 0
   itself run a small `eps` sweep (e.g. `{1e-4, 1e-3, 1e-2}` all at once,
   still ≤1.5 GPU-h total) rather than a single value + one reactive retry?
   The single-value choice was made to keep Stage 0 minimal-change and
   cheap; a sweep would cost more but resolve the "wrong eps" ambiguity in
   one pass instead of two sequential ones.
2. Does the `D·Q_target`/`R·Q_target` closed-form check (flagged above as a
   pre-launch item) actually require reading code this draft did not read
   (the specific entity-embedding construction)? This document reasons from
   the ARCHITECTURAL description in `NCR_ORTHO_WRITE.md` (K-cycle over a
   `d=K+1` tight-spare embedding) but has not inspected
   `ncr_earlyln_scale.py` / `ncr_task.py`'s literal embedding code — the
   independent code audit should confirm the eigenvalue-parity argument
   transfers exactly, not just architecturally.
3. Is n=4 per Stage-1 cell adequate given the §2.35 seed-variance
   precedent (§9), or should the arm-selection logic pre-commit to n=5 for
   any arm that returns a split (non-unanimous) Gate-0 result, priced now
   rather than decided ad hoc mid-run?
4. Is the K=24-before-K=32 sequencing (§5) actually informative, or could
   K=24's smaller `d=K+1=25` behave qualitatively differently from K=32's
   `d=33` in ways that make an early K=24 FAIL a poor predictor for K=32 —
   worth pressure-testing against §9.1's own secondary K24 data (ortho_K24
   failed identically to ortho_K32 in the original run, some support for
   the sequencing's informativeness, but only n=1 prior data point).
5. Should the damped-polar Stage-0 PASS/FAIL threshold (Gate-0 at 48K
   reduced steps) be validated against how many steps the *original*
   free-write K24 arm needed to first clear Gate-0 (not just how it looked
   at the full 320K-step endpoint) — if free-write K24 itself needed close
   to 48K steps to converge, Stage 0's reduced budget may be systematically
   underpowered regardless of whether the eps-floor fix works, confounding
   PASS/FAIL with "ran out of steps." This data point was not available
   during drafting and should be pulled from the archived free_K24 z-dumps
   before Stage 0 launches.

---

## §A1 ATTACK ROUND 1 (2026-07-17)

Independent adversarial review, READ-ONLY except this appended section. Every
arithmetic claim recomputed; the det-parity math re-derived against the ACTUAL
task/embedding code (`ncr_task.py`, `chapter2/task_e.py`, `analyze_zdump.py`)
and confirmed numerically (script logged in session archive). The σ_min(I+W)
proof, the ledger, and the packing plan survive; the load-bearing NEW math
(§3.1 det-parity reachability) does NOT.

### FATAL

**A1.1 — FATAL. The §3.1 "reachability impossibility" for `expm` is FALSE.
The spare dimension's determinant sign is a FREE choice the loss is
indifferent to, so plain `expm(W)` (confined to SO(d), NO reflection) reaches
an EXACT solution at even K. The reflection fix `R` is unnecessary and rests
on a false premise — and the false premise directly contradicts this design's
own inherited §10 mechanism.**

*Evidence (code, not architecture).* The learned operator's behavioral target
is NOT "the K-cycle permutation matrix embedded as identity-on-spare." The
task's ideal write is `z_ideal = K_mat · P · K_matᵀ` (verbatim,
`analyze_zdump.py` DERIVATION; `K_mat` = K random orthonormal key columns, `P`
= canonical K-cycle) — this is **rank K in ambient d=K+1, with the (d−K)=1
spare direction mapped to ZERO**. The read (`binexp_read`, then
`recovery_cosine` to `pool[π^h(i)]`) constrains `Q` ONLY on the K entity keys;
the spare direction is entirely unconstrained. §10.7 of the parent doc states
this as the FAIL mechanism itself: *"zero pressure on the (d−K)=1 spare
direction's magnitude."*

*The math §3.1 got wrong.* A solving orthogonal `Q` must act on the entity
subspace `E` as the K-cycle (forced) and, being orthogonal with `E` invariant,
acts on the 1-dim spare as `s = ±1` (free). `det(Q) = det(P|_E)·s =
(−1)^(K−1)·s`. For even K this is `(−1)·s`: **choosing `s=−1` gives det=+1,
i.e. `Q ∈ SO(d)`, reachable by plain `expm(W)`.** The design's own §3.1 even
supplies the refutation and fails to connect it: it notes `expm` reaches a
`(−1,−1)` PAIRED block at θ=π — the cycle's single −1 eigenvalue (j=K/2) pairs
with the free spare's −1 to make −1 multiplicity 2 (even), det +1, expm-
reachable. Numerically confirmed (K=4, d=5, random key frame): the `s=−1`
operator is orthogonal to 2e-15, solves the task exactly at h∈{1,2,3,5,40},
det=+1.000, reachable by SO(d); the `s=+1` operator solves it too (det=−1).
**BOTH determinant components contain an exact solution.** §3.1's claim that
both arms are "mathematically INCAPABLE of reaching the target… wasting the
entire Stage-1 budget on an experiment that could never have succeeded" is
wrong.

*Does the reflection break anything?* No — `R·expm(W)` still contains a
solution (the `s=+1` one), and the read (`(R·expm(W))^h` by repeated squaring
of the stored `Q`) is computed correctly and lands on the entity targets for
the block-diagonal solution. So the reflection is HARMLESS to solution
existence and to the O(log h) read — it is merely UNNECESSARY, adds a fixed
mis-motivated buffer, and (small) risks perturbing trainability by forcing the
opposite spare sign. The damage is scientific: a false "verified 10-minute
checklist finding" that the design would carry into the paper (§8's
differentiator-sentence style), plus a "build-blocking" fix that blocks
nothing real.

*Cayley is different — right conclusion, wrong reason.* Unscaled Cayley's
image EXCLUDES eigenvalue −1 entirely (`(1−iθ)/(1+iθ)=−1` is unsolvable). EVERY
solution has ≥1 eigenvalue −1 (the cycle's forced j=K/2 mode, independent of
the spare), so unscaled Cayley genuinely cannot reach any solution and DOES
need the `D`-scaling. The design reaches this conclusion via the wrong
invariant (det parity); the correct invariant is the FORCED −1 eigenvalue on
the entity subspace. Keep Cayley's demotion; fix its stated reason. (The
Helfrich "exactly one −1 entry in D" count is likewise spare-dependent and
should be re-derived from the entity-subspace spectrum, not from det.)

*Required fix (blocking).* (1) Delete the "reachability impossibility"
framing and the claim that plain expm/Cayley waste the budget. (2) Make
**plain `expm(W)` (no `R`) the PRIMARY arm** — it reaches an exact SO(d)
solution via the free spare, is simpler, and drops the whole reflection
apparatus + the §3.1/§10 pre-launch eigenvalue-parity check for expm. (3)
Re-derive the Cayley `D`-scaling justification from the forced entity-subspace
−1 eigenvalue. (4) Optionally keep `R·expm` as a disclosed comparison, not a
necessity. The rationale for choosing expm over NS-polar (avoids the 1/σ_min
backward blow-up) is untouched and still valid — only the det-parity sub-
argument dies.

### MAJOR

**A1.2 — MAJOR. The Stage-0 `Z+εI` patch is NOT a sound σ_min floor for the
actual (non-normal) encoder output; the Weyl justification proves the wrong
thing, and the 10× eps retry ladder addresses the wrong axis.**

The design (§2) justifies the floor via Weyl: `|σ_i(A+εI)−σ_i(A)| ≤ ε`. That
bounds the CHANGE in each singular value, not a LOWER FLOOR. For a NON-NORMAL
matrix, εI shifts EIGENVALUES, not singular values; σ_min(Z+εI) can be ≪ ε.
Concrete: `Z=[[0,M],[0,0]]` (nilpotent, σ={M,0}); `Z+εI=[[ε,M],[0,ε]]` has
`σ_min·σ_max = ε²`, so `σ_min ≈ ε²/M ≪ ε` for large M=σ_max. The ortho cells'
measured departure-from-normality was 0.17–0.69 (parent §9.1) and the drift
drove encoder cond≈1e8 — i.e. exactly the non-normal, large-σ_max regime where
`ε²/M` collapses the additive floor. The "Tikhonov/ridge, effective floor of
order ε on that direction" argument is a normal/PSD intuition that does not
transfer. Consequence: a Stage-0 FAIL "same signature" would be
MIS-ATTRIBUTED to "SGD pushed the effective magnitude below the floor" when in
fact no floor ever existed; the pre-authorized `eps=1e-2` (10×) retry chases
the wrong axis (bigger ε does not fix a structural `ε²/M` collapse). *Fix:*
either (a) replace additive `εI` with a genuine σ_min floor — SVD singular-
value clamp of the raw encoder output, or damping along the polar direction
`Z(ZᵀZ+εI)^{-1/2}` — or (b) explicitly downgrade Stage-0's soundness claim,
correct the Weyl reasoning to "additive-εI probe, floor not guaranteed for
non-normal Z," and state that a Stage-0 FAIL is INCONCLUSIVE about the trap.
Bounded blast radius (Stage 0 is ≤1 GPU-h and Stage 1's 2-arm grid runs
regardless), so MAJOR not FATAL — but the math is wrong and the recorded
conclusion would be too.

**A1.3 — MAJOR. The split-result seed-escalation must be pre-registered as a
frozen decision rule NOW, not left to mid-run coordinator discretion — a
blind-assess protocol cannot have a discretionary hole at its decision point.**

§9 / open-Q3 correctly invoke the §2.35 1-in-5 catastrophic-seed precedent
(measurable only at n≥5) but then punt: *"the coordinator should not treat a
split result as a clean verdict without at least considering…"* That is
exactly the ad-hoc, post-hoc-discretion move the record-before-read discipline
(§6, inherited) forbids. The escalation trigger and its budget must be frozen
before launch. *Fix:* pin a crisp rule, e.g. **"IF the selected arm's decisive
K=32 cell has 1 ≤ (Gate-0-passing seeds) ≤ 3 of 4, n is raised to 8 for that
arm×K cell (pre-priced contingency +4 cells × 4.3 h = +17.2 GPU-h) and the §R3
catastrophic-seed disposition clause applies; the ≥3/4 FAIL threshold is NOT
mechanically applied to a split population."** Freeze the number; add the
contingency line to the §5 ledger.

**A1.4 — MAJOR. "A build agent substitutes the winning arm's name for
'Newton–Schulz'… a disclosed terminology edit, not a re-gauntlet" (preamble,
§4) understates the downstream work for a non-NS winner.**

At the GATE level both downstream docs phrase GATE 1 on the OUTCOME
(`NCR_KLADDER_DESIGN.md` §9: "rec@0.9 at h*=40 ≥0.9 at K=32";
`NCR_REAL_LM_DESIGN.md` §9: "ortho-write verdict WIN/PARTIAL"), so the gate
DISCHARGE is legitimately arm-agnostic — good. BUT both BUILDS hard-wire
NS-polar mechanism specifics: the K-ladder's §2 FLOP model has an NS-specific
"Newton–Schulz term" that grows with K and its own §A1.1 FATAL rests on the
NS/encoder rank pipeline; the real-LM build bakes in the "Newton–Schulz
orthogonal-write pipeline" (§2, lines ~252/361) and an NS-specific stability
claim ("‖QᵀQ−I‖=1 unfixable by any amount of Newton–Schulz iteration", ~line
679). Swapping in `expm` changes the per-write cost scaling and the stability
argument, so it is NOT a pure rename. *Fix:* state that a non-NS winner
requires the K-ladder and real-LM COST/STABILITY models (not the science gate)
to be re-derived for the new parametrization before their builds execute —
one bounded re-derivation, but flag it, don't call it a terminology edit.
(The encoder rank-reachability issue is arm-independent and unaffected.)

### MINOR

**A1.5 — MINOR. The §1 falsifiable prediction / §4 WIN mechanistic
corroboration is near-vacuous for the structural arms.** expm/Cayley produce a
GLOBALLY orthogonal `Q` by construction (departure→0, cond→1 trivially,
independent of whether the task is learned), so "the spectral signature moves
toward orthogonal" cannot be falsified for these arms. The teeth that remain
live in the entity-BLOCK quantities (`A=UᵀQU`: min|λ|/c*, block departure,
block cond detect E-invariance / spare leakage, which are NOT guaranteed by
global orthogonality). *Fix:* re-scope the WIN mechanistic leg explicitly to
the entity-block diagnostics for structural arms and drop the global-
orthogonality framing (which was the meaningful independent check only for
NS-polar).

**A1.6 — MINOR. Checklist item 2 ("compute FLOPs on paper, no exceptions") is
only qualitatively discharged for expm.** §3.2 gives "a handful of d×d
matmuls… same order as NS-40" but no per-write-step count for
`matrix_exp`+backward at d=25/33. Because the cost pin (4.3 h/cell) is
empirical and the regime is overhead-bound, this does not drive a wrong
budget, but the explicit number (scaling-and-squaring ≈ Padé-13 ~6 matmuls +
~2–4 squarings per forward, doubled-ish for the Fréchet backward, × batch 256)
should be written down per the checklist. (torch.matrix_exp backward IS
implemented/stable in torch 2.8; the target's minimal generator has eigenvalue
gaps < 2π so it sits away from the expm singular set — both CLEAR.)

**A1.7 — MINOR. eps is carried K=24→K=32 without re-validation (instrument-
relative hard rule).** If Stage 0 PASSES, §3.4 pins the K=24 eps as the K=32
damped-polar hyperparameter "no further tuning." The σ_min danger threshold is
K/d-relative (parent hard rule: the n_iter-sufficiency frontier MOVES with
K/d). §9's damped-polar bullet half-acknowledges this ("scale-dependent
partial fix"). *Fix:* either re-validate the floor at K=32 or explicitly scope
the K=32 damped-polar cell's result as provisional. (Compounds with A1.2: if
the floor is unsound at K=24 it is more so at the larger d of K=32.)

**A1.8 — MINOR. The grid is all-even-K (24, 32); there is no odd-K control.**
The (now-refuted) parity hypothesis predicted plain-expm success at odd K
(det=(−1)^(K−1)=+1) and failure at even K without R. An odd-K cell would have
been the direct falsification test — moot given A1.1's refutation, but noted:
the current grid cannot distinguish "parity story" from "spare-freedom story"
because both K are even.

### CLEAR (probed, survived)

- **σ_min(I+W) ≥ 1 (§3.1):** CORRECT. W skew ⇒ W normal ⇒ I+W normal (both
  `(I±W)` products equal `I−W²`); singular values = |eigenvalues| = |1+iθ| =
  √(1+θ²) ≥ 1. Correctly and cleanly SCOPED — used only against §10-trap
  re-entry through Cayley's inversion, kept explicitly separate from the
  distinct ‖W‖→∞ parameter-blowup. Good.
- **Ledger arithmetic (§5):** CLEAN. 2-arm 2×2×4=16 cells ×4.3=68.8 (+1.0
  Stage-0 = 69.8 ✓); 3-arm 3×2×4=24 ×4.3=103.2 (+1.0=104.2 ✓); packing
  N=2×8=16 slots = one wave for 16 cells, 16+8-solo for 24 ✓; 4.3 h pin is the
  conservative discriminator-rate upper bound applied to cheaper single-
  relation cells; 12 h contention ceiling generous (measured 68–77% SM single
  → ~1.4–1.5× packed slowdown → ~6.5 h, well under 12 h). No double-count of
  the canary.
- **Saturation packing (§3.5):** realistic and canary-gated; memory trivial;
  achieves the 100%-utilization intent.
- **Open-Q5 (Stage-0 48K budget) — DISCHARGED by the data the draft said was
  unavailable.** Pulled the archived `free_K24_s{0..3}` loss histories: healthy
  cells reach loss <0.01 by step 7K–12K and <0.002 by 10K–22K (worst seed
  22K). Stage-0's 48K budget carries ≥2× margin over healthy convergence, so a
  Stage-0 FAIL is NOT confoundable with "ran out of steps." (Caveat: this is
  the FREE arm; damped-polar-through-constraint could be somewhat slower, but
  the 2× margin covers it.)
- **Mod-K / blank-out / blind-record / resume-safety / novelty (OSA+MuonSSM
  complementary, confirming-instance framing):** faithfully inherited; ladder
  residues {5,12,20,29,40,61} all novel mod 24 AND mod 32 (asserted in
  `realistic_ladder_eval`).

### VERDICT: **REVISE**

Blocking items before freeze:
- **A1.1 (FATAL)** — correct the det-parity math; make plain `expm` primary;
  drop the "reachability impossibility"/"wasted budget" framing; re-derive
  Cayley's `D`-scaling reason from the forced entity-subspace −1 eigenvalue.
- **A1.2 (MAJOR)** — fix or downgrade the `Z+εI` σ_min-floor claim (Weyl
  bounds change, not floor; non-normal collapse `ε²/M`); the 10× retry chases
  the wrong axis.
- **A1.3 (MAJOR)** — pre-register the split-result n=5→8 escalation as a
  frozen rule + priced ledger line.
- **A1.4 (MAJOR)** — restate the downstream substitution as a bounded cost/
  stability-model re-derivation for a non-NS winner, not a terminology edit.

MINORs (A1.5–A1.8) fold into the same revision. The experiment is salvageable
and worth running — expm is a sound choice on its real merit (no 1/σ_min
backward blow-up); it is the design's central NEW argument, not the
experiment, that fails the attack.
