# NCR ORTHOGONAL-WRITE FALLBACK — PRE-REGISTRATION

**Status: DRAFT, REVISED post-Attack-Round-1.** §A1's findings (1 FATAL, 3
MAJOR, 4 MINOR) are dispositioned in §R1 below — F1–F4 fixed in the body,
all MINORs adjudicated. Still **not frozen**; no GPU spend is authorized by
this document alone pending the coordinator's next gauntlet step
(independent code audit, per §10's checklist) or a further attack round.
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
> A had won — no separate re-approval needed at the GATE level (both
> downstream docs phrase GATE 1 on the OUTCOME, arm-agnostic). **For a
> non-NS winner, this is NOT a pure terminology edit** (§A1.4): a build
> agent must ALSO re-derive the K-ladder's and real-LM's COST/STABILITY
> models for the winning parametrization before either BUILD executes —
> both documents hard-wire Newton–Schulz mechanism specifics (the K-ladder's
> NS-term FLOP model; the real-LM's NS-specific `‖QᵀQ−I‖`-unfixable-by-
> iteration stability claim) that do not transfer verbatim to `expm`/
> Cayley/damped-polar's different cost scaling and stability profile — see
> §4's arm-selection rule for the itemized, bounded re-derivation scope
> (per-write FLOPs, backward cost, ceiling calibration). **If every
> arm here ALSO returns FAIL/NULL, GATE 1 resolves definitively to NULL/FAIL**
> and both downstream documents execute their *already pre-registered* K=15
> fallback branches — the honest terminal state, not a new decision made in
> the moment.

**Ceremony tier (CLAUDE.md doctrine).** Stage 0 alone (≤1.0 GPU-h worst
case) qualifies for the light <10 GPU-h / 1-audit-round tier and could be
fast-tracked. Stage 1 (≈70–121 GPU-h worst case including the A1.3
split-result contingency, §5 — publication-bound, it is the GATE-1 retry
for the flagship ICLR-2027 spearhead) requires the full multi-round
adversarial gauntlet; this document is round 1 of that gauntlet.

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

**Hypothesis (one sentence, revised per §A1.1 — no reflection).** A
skew-symmetric-group orthogonal parametrization — primarily the **plain**
matrix exponential `Q=expm(W−Wᵀ)` (no reflection factor: the K-cycle
target's spare-direction sign is a loss-free choice, and the even-K branch
that sets it to `s=−1` already lies in `SO(d)`, exactly reachable, §3.1),
with the diagonally-scaled Cayley `Q=D·(I−W)(I+W)⁻¹` run as a demoted
comparison arm — will train through Gate-0 and recover realistic-depth
(`h*≈40`) composition at `K∈{24,32}` where the hard Newton–Schulz-polar
projection failed, because neither route ever inverts (or otherwise
numerically requires) a matrix whose smallest singular value can
random-walk to zero, structurally eliminating the *specific* §10
ill-conditioning-runaway mechanism — while Cayley (only) remains
independently exposed to a *different*, literature-established blow-up,
because **every** solution (regardless of the free spare-direction choice)
forces exactly one eigenvalue `−1` on the K-cycle's own entity-subspace
spectrum (§3.1), which unscaled Cayley's image structurally excludes.

**Falsifiable prediction (re-scoped per arm type — §A1.5 fix).** For the
STRUCTURAL arms (expm, Cayley), GLOBAL orthogonality
(departure-from-normality → 0, cond# → 1) is **guaranteed by construction**
regardless of whether the task is learned — it is not a falsifiable check
for these two arms. The teeth instead live in the **entity-block**
diagnostics (`A=UᵀQU`, the inherited §5/§10 machinery): `min|λ|/c* → ~1`,
block departure-from-normality → ~0, block cond# → ~1 — these detect
`E`-invariance and spare leakage, neither of which is guaranteed by global
orthogonality alone. An arm passing Gate-0 must show its trained operator's
**entity-block** spectral signature move toward orthogonal; a WIN/PARTIAL
with a non-orthogonal entity-block signature would itself be an anomaly
requiring explanation before being trusted. **For the damped-polar arm
(§3.4) alone**, which is NOT structurally confined to `SO(d)`
(`det(Q)=sign(det Z)` is free to flip through training), the **global**
diagnostic remains the meaningful, non-vacuous check — exactly as it was
for the original NS-polar arm (`NCR_ORTHO_WRITE.md` §4/§9.1's own usage).

**No calibrated odds are offered (unlike `NCR_ORTHO_WRITE.md` §7, which had
an in-silico post-hoc polar preview as a numerical anchor).** Cayley/expm
have never been run even once in-silico against this task. The closest
available anchor is §10.7's K=8 local CPU repro of the *original* NS-polar
mechanism escaping the trap at small scale — informative about the general
shape of ill-conditioning basins, uninformative about expm/Cayley's own
basins, which are structurally different objects. This is a disclosed
evidence gap, not a claim of confidence — flagged again in §10.

---

## §2 STAGE 0 — THE DECISIVE CHEAP GATE (§10.8's recommendation, floor construction revised per §A1.2)

**Purpose.** §10.8 names, as the cheap confirmatory test *before* committing
to a full reparametrization, a **raw-Z conditioning fix** on the *existing*
NS-polar path: re-run one ortho K=24 cell with a σ_min floor. This isolates
whether the FAIL is fully explained by the spare-direction random walk
(§10.7), independent of which parametrization family Stage 1 ultimately
uses — Stage 0 patches the diagnosed mechanism directly rather than
sidestepping it via a different group parametrization.

**Revision 1 note (A1.2 fix — the original floor was unsound, replaced
below).** The original intervention (`Z_damped = Z_raw + eps*I`) was
justified via Weyl's inequality, which bounds the *change* in each singular
value under an additive perturbation, not a *lower floor*. For a
NON-NORMAL matrix — the measured regime here, ortho cells' departure-from-
normality 0.17–0.69 (`NCR_ORTHO_WRITE.md` §9.1) — `εI` shifts eigenvalues,
not singular values, and `σ_min(Z+εI)` can be `≪ ε`: concretely,
`Z=[[0,M],[0,0]]` (nilpotent, σ={M,0}) gives `Z+εI=[[ε,M],[0,ε]]` with
`σ_min·σ_max=ε²`, so `σ_min≈ε²/M≪ε` for large `M=σ_max` — exactly the
large-σ_max, non-normal regime §10.7 measured (encoder cond≈1e8). The
additive patch therefore did NOT guarantee the floor it claimed; a Stage-0
FAIL under it would have been mis-attributed ("SGD pushed the effective
magnitude below the floor") when no floor ever existed, and the
pre-authorized 10× retry would have chased the wrong axis. **Replaced below
with an EXACT, SVD-based σ_min floor — a genuine worst-case guarantee,
independent of normality.**

**Exact intervention (revised).** Insert a detached SVD-floor correction
before the existing pre-scale step; leave `newton_schulz_polar` and
everything downstream of `Z_damped` byte-identical:

```
Z_raw = self.encoder(keys, values)                            # UNCHANGED, (...,d,d)
with torch.no_grad():                                          # detached — mirrors the
    U, S, Vh = torch.linalg.svd(Z_raw, full_matrices=False)    # EXISTING sigma_hat
    sigma_max = S[..., :1]                                     # detached-power-iteration
    S_floor = S.clamp_min(eps_rel * sigma_max)                 # pattern already in this
    scale = S_floor / S.clamp_min((eps_rel * sigma_max * 1e-6).clamp_min(torch.finfo(S.dtype).tiny))   # file (below)
                # ^ [D1 FIX, §B4 audit 2026-07-17] originally S.clamp_min(NS_EPS);
                # NS_EPS=1e-7 sits exactly at §10.2's diagnosed danger threshold and
                # silently degraded the floor to proportional-in-sigma_min below 1e-7
                # (sigma_min=1e-9 -> 100x short of floor; ~0 -> 9 orders short — §B4 D1).
                # [F-B FIX, §B6 Ruling 3, 2026-07-17] COMPOSED guard replacing the
                # v2 bare finfo.tiny: the relative term eps_rel*sigma_max*1e-6 caps
                # scale at 1e6 (kills §B5 F-B: fp32 bit-exact-zero sigma -> scale
                # ~1e36 -> non-finite forward); the outer finfo.tiny backstop covers
                # the measure-zero Z==0 case (relative term 0 there -> would
                # reintroduce 0/0). Exact floor above the bite point; bounded,
                # finite under-floor below it — see the §B6-scoped guarantee below.
    M = U @ torch.diag_embed(scale) @ U.transpose(-1, -2)      # (...,d,d), symmetric PSD
Z_damped = M @ Z_raw          # NEW — exact sigma_min floor, straight-through gradient
                               # (M constant w.r.t. autograd: dL/dZ_raw = M @ dL/dZ_damped)
sigma_hat = detached_power_iteration(Z_damped, n_power=12)     # UNCHANGED call, damped input
X0 = Z_damped / sigma_hat
Q  = newton_schulz(X0, n_iter=40)                              # UNCHANGED — do NOT bump n_iter (§10.8)
```

`n_iter=40`, `n_power=12` stay at `NS_ITER_DEFAULT` / `NS_POWER_DEFAULT`
exactly (`ncr_ortho_write.py`) — §10.8 explicitly warns bumping `n_iter`
*worsens* the backward explosion; this intervention must not be confounded
with that axis.

**Why this is a genuine floor.** `M @ Z_raw = U·diag(scale)·Uᵀ·U·Σ·Vᵀ =
U·diag(scale⊙σ)·Vᵀ` — `Z_damped`'s own SVD shares `Z_raw`'s `U`,`V` and has
singular values exactly `σ_floor,i = max(σ_i, eps_rel·σ_max)`.

> **Scoped guarantee (§B6/F-A; replaces the withdrawn "exact for every
> Z_raw" claim).** `Z_damped`'s singular values are `scale_i·σ_i`. For
> every computed `σ_i` the dtype's SVD resolves (`σ_i ≳ ε_dtype·σ_max`;
> fp32 ~1.2e-7·σ_max) and above the guard's bite point
> (`eps_rel·σ_max·1e-6`), the floor `σ_floor,i = max(σ_i, eps_rel·σ_max)`
> holds to the SVD's own relative accuracy on `σ_i` (rel error ≈
> `ε_dtype·σ_max/σ_i`), regardless of normality. At the dtype noise floor
> the achieved value is O(1)-approximate, not exact (measured 0.15×–4.1×
> target across independent frames), and for a truly rank-deficient
> `Z_raw` no floor is delivered on the dead direction at all
> (left-multiplication cannot raise rank) — the guard caps `scale ≤ 1e6`
> so that case is a finite, bounded under-floor, never Inf/NaN. In the
> reachable training regime this still bounds `cond(Z_damped)` at
> O(1e3–1e4), ≥3 orders inside §10.2's 1e7+ danger threshold —
> sufficient for Stage-0's purpose, but the claim "exact for every
> Z_raw, even σ_min exactly 0" is WITHDRAWN.

No Weyl approximation, no
eigenvalue/singular-value conflation. The floor is expressed RELATIVE to
`σ_max` (not an absolute constant, unlike the struck patch) because §10.7
measured `σ_max` itself drifting unconstrained (5→13 observed): an
absolute floor loses its safety margin as `σ_max` grows; a relative floor
does not, and is a mathematically sound choice under the parent document's
own instrument-relative hard rule (thresholds tied to a drifting scale must
themselves be relative).

**Why the SVD is detached (a numerical-stability choice, not just a cost
cut).** Differentiating *through* `torch.linalg.svd` directly has its own
known instability: the backward has `1/(σ_i²−σ_j²)`-type terms that
explode when singular values are close or coincide — and the regime this
fix is meant to REACH is exactly that (the healthy `free_K24` baseline
converges to `cond≈1.0`, i.e. near-degenerate singular values, §9.3 of the
parent doc). Using the SVD only to build a **detached** correction operator
`M` — mirroring the EXISTING `sigma_hat` detached-power-iteration pattern
already in this file ("the scale is a read-invariant no-op") — avoids ever
differentiating through the SVD itself: `Z_damped = M @ Z_raw` is a plain
constant-matrix linear map in the autograd graph, so gradient reaches
`Z_raw` cleanly with no degenerate-singular-value blow-up risk, at the cost
of not giving the loss direct pressure on *which* directions get floored
(the same trade the existing `sigma_hat` detach already makes) — acceptable
for a minimal-change attribution test, not a claim about how a production
write should be regularized. This refinement goes beyond A1.2's literal
suggestion (which named plain SVD clamping without flagging its own
backward-instability risk in this specific near-degenerate target regime).

**`eps_rel = 1e-3`** (fixed, dimensionless fraction of `σ_max`; not
learned, not annealed). Chosen to mirror the struck patch's own margin
logic, corrected onto the right axis: with `eps_rel=1e-3`,
`cond(Z_damped) ≤ 1/eps_rel = 1e3` — three orders of magnitude inside the
`Repro 1b` clean-convergence ceiling (NS-40 orthogonalizes cleanly for
input cond up to `1e6`, degrading only at `1e7`+, §10.1) and four orders
below the measured danger threshold translated to relative terms
(`σ_min≲1e-7` against an O(1) healthy `σ_max`, §10.2 — relative cond
`≳1e7`). Unlike the struck additive floor, this bound holds **regardless
of how far σ_min has drifted** within the dtype's resolvable regime —
[the original "even if σ_min random-walks to exactly 0, the floor is
still exact" claim is WITHDRAWN per §B6/F-A: at the dtype noise floor
the floor is O(1)-approximate, and at true rank deficiency it is a
bounded, finite under-floor — see the scoped guarantee above]. (A smooth
alternative, damping via
`Z(ZᵀZ+εI)^{-1/2}`, was considered and rejected: its output singular value
`σ_i/√(σ_i²+ε)` → 0 as `σ_i` → 0, so it does NOT provide a floor
independent of how small `σ_min` has drifted — `cond ≈ √ε/σ_min → ∞` as
`σ_min→0` for ANY fixed `ε` — only the exact SVD-clamp gives the worst-
case-independent guarantee F2 requires.)

**Cost of the fix (A1.2's required pricing).** One additional batched
`torch.linalg.svd` call (`full_matrices=False`, `no_grad`, forward-only —
no backward graph) at `d∈{25,33}`, batch≈256, plus two batched `d×d`
matmuls (`M` construction, `M@Z_raw`) in the differentiable path. FLOP
cost is trivial in absolute terms (a thin SVD at `d≤33` is on the order of
a handful of `d×d` matmuls' worth of FLOPs, comparable to or below ~10 NS
iterations out of the existing 40), but per this document's own
established finding (§3.2; `NCR_KLADDER_DESIGN.md`'s FLOP-ratio-refuted
regime) this scale is kernel-launch/small-batch overhead-bound, not
compute-bound — an extra kernel launch (the SVD call) can cost
disproportionately more wall-clock than its FLOP count implies.
**Conservative derate: +15% per-step wall-clock**, pending the mandatory
canary below (measured, not assumed — matching the doctrine already used
for the Stage-1 packing canary, §3.5/§5).

**Scope: n=1 seed, K=24, reduced step budget, ≤0.5 GPU-h (the GPU-h
CEILING is unchanged; the achievable step count is revised down for the
added op's cost).** At the measured ortho_K24 rate (~3.3 h / 320K steps,
§9.0) plus the +15% conservative derate (~3.795 h / 320K steps), 0.5 GPU-h
buys **≈42,000 steps** (down from the struck patch's ~48,000; still a
directional smoke, not a statistically powered Gate-0 verdict — Stage 1's
n=4 delivers that for whichever arm(s) survive). **Mandatory canary**
(mirrors §3.5's Stage-1 canary doctrine): run the first ~2,000 steps solo,
measure the actual added wall-clock from the SVD call, and re-price the
exact step count BEFORE trusting either the ≈42,000 figure or the margin
check below — a re-priced estimate, not a measurement, exactly the
distinction the CEILING AMENDMENT precedent (§5) exists to guard against.

**Margin re-check (discharges §A1's Open-Q5 finding under the revised step
count).** §A1 pulled the archived `free_K24_s{0..3}` loss histories: healthy
cells reach loss <0.01 by step 7K–12K and <0.002 by 10K–22K (worst seed
22K). At ≈42,000 achievable steps, margin over the worst observed free-arm
convergence step is `42000/22000 ≈ 1.9×` — down from the original
`48000/22000 ≈ 2.2×` but still comfortably `>1×`; Stage-0 remains
adequately powered, not confoundable with "ran out of steps." (Caveat,
carried from §A1: this is the FREE arm; damped-polar-through-constraint
could converge somewhat slower — the margin is not airtight, but is not
eliminated by this revision either.)

**Branch logic (pre-specified, both directions; structure unchanged from
the struck patch — 10× up/down retry — now correctly targeting `eps_rel`,
the axis that actually controls `σ_min(Z_damped)`).**

- **PASS** (Gate-0 clears within the canary-confirmed step budget —
  `min_{h∈{1,2,3}} recovered_frac@0.9 ≥ 0.9` AND `A_eff_rank ≥
  0.9·24=21.6`): confirms the ill-conditioning trap as the sole blocker
  for the original mechanism; the damped-polar construction is promoted to
  a **third Stage-1 arm** (full 4×-budget, K∈{24,32}, n=4 seeds — §3.4),
  using this exact `eps_rel` as the pinned hyperparameter, no further
  tuning (subject to §A1.7's K=24→K=32 provisional-scoping caveat, §3.4).
- **FAIL, same signature** (loss dips then re-collapses to ~1.0, matching
  §9.1's own dip-then-collapse curve): the floor was insufficient — SGD
  still drives the pre-floor conditioning low enough, relative to a
  simultaneously-growing σ_max, that `eps_rel` undersells the actual
  random-walk rate at K=24 over ~42K steps. **One pre-authorized retry**:
  `eps_rel=1e-2` (10× up), same step budget, ≤0.5 GPU-h.
- **FAIL, different signature** (flat/never-engaged loss from step 0, no
  dip at all — unlike any §9.1 curve): the floor itself broke something
  unrelated to the trap (`eps_rel` too large, over-flooring K
  signal-bearing modes toward the fixed `U`/`V` basis rather than letting
  them train freely). **One pre-authorized retry**: `eps_rel=1e-4` (10×
  down), same budget.
- **If the pre-authorized retry ALSO fails** (either branch): conclude the
  §10 ill-conditioning-trap diagnosis, while mechanistically well-evidenced
  by the static/dynamic audit, is **not sufficient alone** to explain the
  FAIL at K≥24 — an incomplete diagnosis, recorded as such. Stage 1 proceeds
  **without** a damped-polar arm (2-arm grid: expm + Cayley only) — this was
  always going to run regardless of Stage 0's outcome (§3), so Stage 0's
  failure costs only its own ≤1.0 GPU-h, not schedule.

**Stage-0 total: ≤1.0 GPU-h worst case (initial attempt + one diagnostic
retry, unchanged — the ceiling is a wall-clock cap, not step-count-based).
Runs strictly BEFORE any Stage-1 spend.**

---

## §3 STAGE 1 — THE PARAMETRIZATION ARMS

### §3.1 The entity-subspace forced eigenvalue — corrected reachability account (Revision 1, supersedes the FATAL det-parity claim, §A1.1)

**Revision 1 note.** The original §3.1 argued the K-cycle target's
determinant forces BOTH `expm` and Cayley outside `SO(d)`, motivating a
fixed reflection factor `R` as a "build-blocking" fix. **That argument is
wrong** (§A1.1, full refutation in the frozen attack record below): it
conflated the K×K entity permutation's OWN determinant with the
determinant of the FULL `d×d` operator, missing that the `(d−K)=1` spare
direction's sign is a second, loss-free degree of freedom that supplies
the missing factor of `−1`. This section is rewritten to the attack's
verified account. The reflection apparatus (`R`, and the associated
pre-launch eigenvalue-parity check for `expm`) is **deleted entirely** —
plain `expm(W−Wᵀ)` is primary, no reflection.

**Before pricing anything: does either candidate parametrization even reach
a solution?** The task's ideal write is `z_ideal = K_mat · P · K_matᵀ`
(`analyze_zdump.py` DERIVATION), **rank K in the ambient `d=K+1` space**:
`K_mat`'s K columns span the K-dim entity subspace `E`; `P` is the
canonical K-cycle permutation matrix in `K_mat`'s own frame; the
`(d−K)=1` spare direction is mapped to ZERO in `z_ideal` but is **not part
of the read's target** — it is unconstrained. **The read (`binexp_read` →
`recovery_cosine`) queries ONLY the K entity keys** — this is §10.7's own
diagnosed FAIL mechanism restated as a reachability fact, not only a
training-dynamics one: *"zero pressure on the (d−K)=1 spare direction's
magnitude"* applies equally to the spare's SIGN. **Any orthogonal `Q` that
acts as `P` on `E` and as EITHER `s=+1` OR `s=−1` on the 1-dim spare
solves the task exactly** — the read cannot distinguish the two.

**The K-cycle's forced spectrum on `E` (Schur form — unaffected by the
fix, still correct, still load-bearing).** `P`'s real block-diagonal form
is one trivial `+1` block (`j=0`), `K/2−1` genuine `SO(2)` rotation blocks
(`j=1..K/2−1`, eigenvalues `e^{±2πij/K}`), and — because K is even (both
K=24 and K=32 qualify) — **exactly one isolated real eigenvalue `−1`**
(`j=K/2`, `e^{iπ}=−1`), living in a 1-dimensional invariant subspace of
`E`. This mode is FORCED on every solution, independent of the spare.

**`det(Q) = det(P|_E) · s = (−1)^{K−1} · s`.** For even K, `(−1)^{K−1}=−1`,
so `det(Q) = −s`. **Choosing `s=−1` gives `det(Q)=+1` — an EXACT `SO(d)`
solution, reachable by plain `expm(W)`, no reflection.** (`s=+1` gives
`det(Q)=−1`, an equally valid solution to the TASK, but outside plain
`expm`/Cayley's image — irrelevant here, since nothing forces training
toward that branch and the `s=−1` family is exactly reachable.)
**Numerically confirmed** (§A1.1: K=4, d=5, random key frame — the `s=−1`
operator is orthogonal to 2e-15, solves the task exactly at
h∈{1,2,3,5,40}, det=+1.000; the `s=+1` operator solves it too, det=−1.000
— demonstrating BOTH branches are genuine solutions, only one of which
plain `expm`/Cayley can reach).

**Consequence — corrected.** `expm(W)` and unscaled `Cayley(W)` remain
confined to `SO(d)` (`det=+1` for every `W`, proofs below, unchanged) —
but this is NOT a reachability impossibility for `expm`, because the
task's solution set is not one fixed matrix: the `s=−1` branch already
sits inside `SO(d)`. **Plain `expm(W)` reaches it directly. No fixed
reflection factor is required, and none is used.**

- *Cayley:* `W` skew ⇒ eigenvalues are `0` or conjugate pairs `±iθ`. Cayley
  maps `λ↦(1−λ)/(1+λ)`: `0↦1`; each pair `±iθ↦` a conjugate pair whose
  product is `1`. `det(Q)=∏λ_i=+1` always.
- *skew-exp:* `det(expm(W))=e^{tr(W)}`, and skew-symmetric matrices are
  traceless (zero diagonal) ⇒ `det(Q)=e^0=1` always.

**Cayley is different — genuinely needs a fix, re-derived from the CORRECT
invariant.** Cayley's map structurally EXCLUDES eigenvalue `−1` from its
image: `λ↦(1−λ)/(1+λ)=−1` has no finite solution for any skew eigenvalue
`λ=iθ` (`θ→∞` only, never reached at finite `W` — see the blow-up
paragraph below). The entity subspace's `j=K/2` mode is forced to `−1` in
**every** solution — both the `s=+1` and `s=−1` families carry it,
independent of spare. **Because that forced mode is never in unscaled
Cayley's image, unscaled Cayley genuinely cannot reach ANY solution,
regardless of spare choice** — this is the correct, spare-independent
reachability obstruction driving Cayley's demotion (the previous draft
reached the same "Cayley needs `D`" conclusion, but via the wrong,
whole-target det-parity argument — right conclusion, wrong reason, per
§A1.1).

**Required fix for Cayley only (re-derived, Helfrich-backed):** `Q = D ·
(I−W)(I+W)⁻¹`, `D=diag(d_1,…,d_d)`, exactly one `d_i=−1` (placed on the
forced entity mode; `+1` elsewhere, spare included — targeting the `s=+1`
branch specifically, the one with exactly one total eigenvalue `−1`). Per
Helfrich et al. 2018's scoRNN theorem (`research/ortho_write_grounding.md`
§1, VERIFIED arXiv:1707.09520), `D·Cayley(W)` parametrizes exactly the
orthogonal matrices with `|{i:d_i=−1}|` eigenvalues equal to `−1` — a
theorem-backed, general fix for the FORCED mode, not a lucky alignment
with the (now-struck) det-parity story. (The `s=−1` branch, with TWO total
`−1` eigenvalues, is reachable via a two-entry `D` instead if ever needed
— not pursued here, since the one-entry `D` already supplies a reachable
target and no evidence favors one branch over the other.)

**But reachability is not the whole story for Cayley — this is where §3.1's
math and the verified literature (§7/§8) diverge sharply between the two
arms.** `σ_min(I+W)≥1` for **any** real skew `W`, unconditionally (proof:
`W` skew ⇒ normal ⇒ `I+W` normal ⇒ singular values of `I+W` equal
`|eigenvalues| = |1+iθ_j| = √(1+θ_j²) ≥ 1`). **The `(I+W)` inversion inside
Cayley can therefore never become singular — the *exact* §10 mechanism
(NS forward failing to converge on a near-singular raw matrix, backward
exploding ~1/σ_min) cannot recur through Cayley's inversion route.** This is
a genuine, correct, and distinct finding from the one below — keep them
separate.

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
entity subspace's forced `−1` eigenvalue (this document's own §3.1
derivation, now correctly scoped to the entity block rather than the whole
target) is precisely the "order-2 composed state" the sweep's ruling flags
as triggering the blow-up. **This is a mechanistically distinct pathology
from §10's**: not a matrix-inversion collapse, but unbounded parameter
growth chasing an asymptotically-approached, never-exactly-reached
boundary of the Cayley map's image. Both are real; they are not the same
trap, and Cayley is vulnerable only to the second.

**Honesty boundary (do not overclaim).** The `D`-scaling fix closes
*reachability* on the forced entity mode (Helfrich's theorem is exact,
general, not alignment-lucky). It does **not** by itself establish whether
the *residual* `SO(d)` target `D·Q_target` sits far from or near Cayley's
own −1-eigenvalue boundary — that is an open, embedding-specific question.
**Pre-launch audit action item:** compute `D·Q_target`'s eigenvalues in
closed form for the actual entity embedding (CPU-only, minutes) before
launching Cayley cells, rather than assuming either direction.

**skew-exp has no analogous blow-up.** Per the sweep, expm's own singular
set is "eigenvalue gaps at nonzero multiples of `2πi`" — a
periodicity/gauge-redundancy issue (multiple `W`s map to the same `Q`, e.g.
`θ` and `θ+2π`), not a norm-divergence-to-reach-a-boundary issue. Confirmed
independently by this document's own eigenvalue-mapping proof: `expm`
reaches a real eigenvalue exactly `−1` at **finite** `θ=π` per 2D rotation
block (a genuine `SO(2)` rotation by `π`) — no parameter ever needs to
diverge, and (per the corrected account above) no reflection factor is
needed either. **This is why `expm` is promoted to PRIMARY, plain, and
Cayley demoted to a comparison arm carrying this analysis** — the
rationale for the promotion (avoiding the Lezcano-Casado blow-up, and
avoiding the §10 `1/σ_min` matrix-inversion trap) is unchanged by A1.1's
fix; only the now-deleted det-parity sub-argument for *why* `expm` needed
a reflection has been struck.

### §3.2 PRIMARY arm — skew-exp / matrix exponential (plain, no reflection — §3.1/§A1.1)

```
W  = raw_skew_param(encoder(keys, values))   # any d×d, then W ← W − Wᵀ
Q  = torch.matrix_exp(W)                     # PLAIN expm — no reflection: the s=-1
                                              # solution branch already sits in SO(d) (§3.1)
```

- **Structural elimination of the §10 trap:** no matrix inversion anywhere
  in the forward path; `expm` is defined and well-conditioned for *every*
  real `W`, however large its norm (unlike Cayley, no boundary to
  asymptotically chase for THIS task's target — §3.1).
- **Cost/gradient path.** `torch.matrix_exp` is natively differentiable
  (Fréchet-derivative backward, available since PyTorch 1.7, stable in the
  box's installed 2.8 — framework capability, not a literature claim).
  Implemented via scaling-and-squaring + Padé approximation; **explicit
  per-write op count (A1.6 fix — the checklist item was only qualitative
  before):** Padé-13 ≈ 6 `d×d` matmuls + ~2–4 squarings per forward,
  roughly doubled for the Fréchet-derivative backward, × `TRAIN_BATCH=256`
  — the same order of magnitude as NS-polar's 40-iteration loop (each
  iteration itself 2 matmuls). The target's minimal generator has
  eigenvalue gaps `<2π`, so it sits away from `expm`'s own singular set
  (periodicity aliasing at nonzero multiples of `2πi`, §3.1) — CLEAR, not
  a risk. Expect wall-clock **comparable to or slightly above** the
  measured NS-polar rate (~3.3 h/cell), not qualitatively different,
  especially given the established finding (`NCR_KLADDER_DESIGN.md`,
  FLOP-ratio scaling refuted at small K) that this regime is
  **kernel-launch/small-batch overhead-bound, not compute-bound** —
  differences between parametrizations at this tiny `d` are predicted to
  compress toward the overhead floor, to be CONFIRMED (not assumed) by an
  early canary cell (§3.5).
- **Param/FLOP delta vs. the failed NS-polar arm:** `W` has the same
  `d(d−1)/2` free parameters as any `d×d` skew generator (comparable to
  NS-polar's raw `Z` encoder output count) — no fixed buffer, no
  additional parameters or buffers of any kind (the reflection `R` is
  deleted, §A1.1). No material param-count change from the original
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
original `NCROrthoWriteModel.encode` plus the detached SVD-floor
correction (`M @ Z_raw`, §2) at the pinned `eps_rel` that cleared Gate-0 in
Stage 0. Full 4×-budget run, K∈{24,32}, n=4 seeds — mirrors Part A
exactly, same as the other two arms. No reachability fix needed (still a
raw-matrix polar factor, `det(Q)=sign(det Z)` is free to flip through
training, unlike Cayley/expm's structural `SO(d)` confinement — §3.1).

**A1.7 fix — K=24→K=32 provisional scoping (instrument-relative hard
rule).** The pinned `eps_rel` is calibrated at K=24/d=25 in Stage 0 and
carried to K=32/d=33 "no further tuning" — but the parent doc's own hard
rule holds that a conditioning frontier calibrated at one K/d does not
automatically transfer to another (`NCR_KLADDER_DESIGN.md` precedent,
n_iter-sufficiency moved with K/d). **This carry-over is explicitly
UNVALIDATED at K=32 and the K=32 damped-polar cells' results are scoped as
provisional on that basis** — chosen over a full re-validation (which
would cost additional GPU-h) as the cheaper of A1.7's two offered fixes.
If K=32 damped-polar shows a FAIL with the §10.7 ill-conditioning
signature specifically (dip-then-collapse, not a flat never-engaged
curve), the correct read is "the K=24-calibrated floor did not generalize
to K=32's conditioning regime," not "damped-polar is dead" — report both
readings, do not silently collapse to the harsher one.

### §3.5 Grid, cost, saturation packing

**Grid:** `{arm} × K∈{24,32} × n=4 seeds`, mirroring Part A's structure
exactly. `arm ∈ {expm, cayley}` always (16 cells); `+damped-polar` if Stage
0 passes (24 cells). **Part B (the structured-operator discriminator) is
NOT re-run this wave** — its own free-bank baseline never trained in the
original run (§9.2, compounded null unrelated to the ortho-constraint), so
it needs its own calibration fix before it can serve its mod-K-trap-safe
role again; deferred, not abandoned (flagged as an open item, §10).
**A1.8 adjudication (MINOR, declined): no odd-K control cell.** All-even-K
was noted by the attack as unable to distinguish "the (now-refuted) parity
story" from "the spare-freedom story" — moot post-A1.1, since §3.1 no
longer has a parity-vs-freedom distinction to test (both are subsumed by
the single corrected forced-entity-eigenvalue account). **Declined**: an
odd-K cell would test a different question (whether Gate-0 trainability
itself depends on K's parity, unrelated to reachability) which no evidence
here motivates probing, and adding it would cost a full extra `K`-column
across all arms/seeds — disproportionate to a curiosity check with no
live hypothesis behind it.

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
  cost evidence than the archived baseline alone. **Mechanistic
  corroboration (A1.5 fix — arm-TYPED, not arm-agnostic; §1):** for the
  STRUCTURAL arms (expm, Cayley), score the **entity-block** diagnostics
  only — `min|λ|/c* ≥0.9` — since global departure-from-normality/cond#
  are guaranteed ≈0/≈1 by construction for these arms and are vacuous
  checks there. For **damped-polar**, score the **global** diagnostics as
  originally specified — departure-from-normality ≤0.02, cond# ≤~2,
  min|λ|/c* ≥0.9 — unchanged from `NCR_ORTHO_WRITE.md` §4's own usage
  (damped-polar is not structurally confined to `SO(d)`, so global
  orthogonality is genuinely earned there, not guaranteed).
- **PARTIAL:** median `rec@0.9` ≥0.9 at `h∈{20 OR 29}` but <0.9 at h=40.
- **NULL:** median `rec@0.9` <0.9 at every depth ≥12, Gate-0 still passes.
- **FAIL:** Gate-0 DEAD (in-dist <0.5) in ≥3/4 seeds.

**Split-result seed-escalation — FROZEN rule (A1.3 fix; mechanically
applicable by a blind assessor, no mid-run discretion).** If, at K=32, the
selected/leading arm's Gate-0-passing seed count `p` satisfies `1 ≤ p ≤ 3`
of 4 (a SPLIT result — neither a clean 4/4 PASS nor the ≥3/4-FAIL band
above), the plain `≥3/4` FAIL threshold is **NOT** mechanically applied to
that arm×K=32 cell. Instead: **`n` is raised to 8 for that cell** (4
additional seeds; **pre-priced contingency: +4 cells × 4.3 GPU-h = +17.2
GPU-h**, carried in the §5 ledger as a conditional line), and the
resulting 8-seed population is scored under the §2.35/§R3
catastrophic-seed disposition clause (`CAPABILITY_SEPARATION_DESIGN.md`
§2.35, `NCR_REAL_LM_DESIGN.md` §N1 R3) rather than the plain `≥3/4` FAIL
band. This is PRE-AUTHORIZED, not a new coordinator decision at run time —
the ONLY circumstance under which any Stage-1 cell's `n` exceeds 4. (This
discharges §9's Open-Q3 and §10's Open-Q3: previously flagged, now
frozen.)

**K=24 cells: same bands, SECONDARY/robustness role** — mirrors
`NCR_ORTHO_WRITE.md` §4's own K=32-primary framing. K=24 alone does not
drive the GATE-1 discharge decision; K=32 is the decisive cell. (The
split-result escalation rule above applies at K=32 only — K=24 is
robustness evidence, not gate-deciding, so a K=24 split is recorded but
does not trigger the n=8 contingency.)

**Arm-selection rule (multiple arms compete for one downstream decision).**
If ≥1 arm clears WIN or PARTIAL at K=32: select the single BEST arm
(highest far-depth median `rec@0.9` at h=40; tie-break by lower measured
per-cell cost; tie-break by tightest mechanistic corroboration) as "the"
winning parametrization forwarded to `NCR_KLADDER_DESIGN.md` /
`NCR_REAL_LM_DESIGN.md` GATE 1. Record which arm won. **The GATE-1
DISCHARGE itself is arm-agnostic and needs no re-gauntlet** — both
downstream docs phrase the gate on the OUTCOME (`NCR_KLADDER_DESIGN.md`
§9's `rec@0.9` bar; `NCR_REAL_LM_DESIGN.md` §9.1's WIN/PARTIAL verdict),
not on which parametrization produced it. **But for a non-NS winner, the
downstream BUILDS are NOT a pure rename (A1.4 fix)** — both documents
hard-wire Newton–Schulz mechanism specifics into their cost/stability
models: `NCR_KLADDER_DESIGN.md` §2's NS-specific FLOP term (grows with K,
feeds that document's own §A1.1) and `NCR_REAL_LM_DESIGN.md`'s NS-specific
`‖QᵀQ−I‖`-unfixable-by-iteration stability claim (~line 679) — these
change under `expm`/Cayley/damped-polar's different per-write cost scaling
and (exact, non-iterative, for the structural arms) stability profile.
**New required post-WIN task (bounded re-derivation, not a terminology
edit, not a re-gauntlet of either document's science gate):** before
either downstream BUILD executes, re-derive for the winning
parametrization: (1) per-write FLOPs, (2) backward cost, (3) ceiling/
runaway-guard calibration (the CEILING AMENDMENT pattern — measure, don't
assume the prior NS-polar numbers transfer). **Expected priority under a
tie is `expm` > damped-polar > Cayley** (per §3's risk ordering), but this
is a tie-break only — an unambiguously better Cayley result is reported
and used, not suppressed.

**If ALL arms FAIL/NULL at K=32:** GATE 1 resolves definitively NULL/FAIL —
proceed to the already-pre-registered K=15 fallback in both downstream
documents (§9.1 of each). No new design work required at that point.

---

## §5 LEDGER — GPU-h, ABORT-ON-COST, sequencing

| Stage | Worst-case GPU-h | Gate |
|---|---|---|
| Stage 0 (raw-Z conditioning smoke, K=24, n=1, ≤2 attempts; SVD-floor construction, §2) | ≤1.0 | Runs first, always |
| Stage 1, 2-arm (expm + Cayley, K∈{24,32}, n=4) | 16 cells × 4.3 h = **68.8** | Runs regardless of Stage 0 |
| Stage 1, 3-arm (+ damped-polar) | 24 cells × 4.3 h = **103.2** | Only if Stage 0 PASSES |
| Split-result seed-escalation contingency (§4, A1.3 fix) | +4 cells × 4.3 h = **+17.2** | Only if the selected arm's K=32 cell returns a split (1≤p≤3/4 Gate-0-passing seeds) |
| **Total worst case, 2-arm branch (no contingency)** | **≈69.8 GPU-h** | 68.8+1.0 |
| **Total worst case, 2-arm branch (WITH contingency)** | **≈87.0 GPU-h** | 68.8+1.0+17.2 |
| **Total worst case, 3-arm branch (no contingency)** | **≈104.2 GPU-h** | 103.2+1.0 |
| **Total worst case, 3-arm branch (WITH contingency)** | **≈121.4 GPU-h** | 103.2+1.0+17.2 |

All four totals sit comfortably inside a single ~192 GPU-h/day operative
budget window on the 8-GPU cluster (CLAUDE.md Hardware), especially
wall-clock-reduced via §3.5's packing — the contingency-inclusive worst
case (121.4 GPU-h) still leaves ~70 GPU-h of same-day headroom.

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
6. **Split-result contingency (§4, A1.3 fix) is PRE-AUTHORIZED, not
   discretionary:** if the frozen trigger (1≤p≤3/4 Gate-0-passing seeds on
   the selected arm's K=32 cell) fires, the +17.2 GPU-h escalation to n=8
   proceeds automatically — it is not a new coordinator decision at run
   time, and it is the only circumstance under which any Stage-1 cell's
   `n` exceeds 4.
7. **Stage-0's own step-count pin (§2) is canary-gated, not assumed:** the
   ≈42,000-step estimate carries a conservative +15% wall-clock derate for
   the added SVD op; the mandatory Stage-0 canary re-prices it before the
   PASS/FAIL bands are trusted at the exact step count.

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

- **expm (primary, plain — no reflection, §A1.1/§3.1):** (1)
  `torch.matrix_exp`'s scaling-and-squaring backward, while bounded (no
  divergence risk per §3.1), could still be slower or noisier than
  expected at the measured overhead-bound regime — confirm via canary,
  don't assume the FLOP argument transfers to wall-clock; (2) if Gate-0
  fails here too, that would be the most informative possible negative
  result (the parametrization literature's own preferred choice failing
  would undercut the entire §10 "parametrization, not projection" framing)
  — worth a fast, dedicated re-audit before accepting a FAIL at face
  value, mirroring §9.4's own "a code bug cannot be excluded from
  behavioral data alone" caution. (The struck risk (1) — the untested
  cost/gradient-path interaction of a fixed reflection factor `R` — is
  deleted along with `R` itself, §A1.1; renumbered from the original
  three-item list.)
- **Cayley (demoted comparison):** the Lezcano-Casado & Martínez-Rubio
  blow-up (§3.1/§3.3) is the headline risk, specifically likely to trigger
  here because the entity subspace's forced eigenvalue is exactly −1 (this
  document's own derivation, §3.1, now correctly scoped to the entity
  block rather than the whole target). A FAIL with a diverging `‖W‖_F`
  trace is a publishable confirming instance, not wasted compute — but a
  FAIL *without* that signature would be unexplained and should trigger
  its own mini re-audit before the arm is written off as "the predicted
  mechanism."
- **damped-polar (conditional third arm):** inherits the ORIGINAL NS-polar
  arm's own risk profile minus the specific spare-direction pathology §10
  diagnosed — i.e. it is a patch, not a structural fix, so any *other*
  route to the same ill-conditioned basin (not mediated by the spare
  direction specifically) would still be live. Stage 0's pre-authorized
  retries (§2, revised per A1.2 — SVD-floor construction, `eps_rel` axis)
  are the cheap test of whether the floor generalizes; if it only works at
  the specific `eps_rel`/K=24 combination tested in Stage 0 and not at
  K=32's full budget, that is itself informative (the trap's severity
  scales with K, consistent with §9.1's own K24-vs-K32 secondary
  observation) and should be reported as a scale-dependent partial fix, not
  silently extrapolated. **A1.7 fix:** the K=32 damped-polar cells' results
  are explicitly scoped as PROVISIONAL on the un-revalidated K=24→K=32
  `eps_rel` carry-over (§3.4) — a FAIL there with the §10.7 ill-
  conditioning signature should be read as "floor didn't generalize," not
  "damped-polar is dead," unless a re-validated `eps_rel` at K=32 also
  fails.

**Split-result seed-escalation is now a FROZEN rule, not an open
question.** The §2.35-style seed-variance concern this bullet previously
flagged as open (`CAPABILITY_SEPARATION_DESIGN.md` §2.35, `NCR_REAL_LM_
DESIGN.md` §N1 R3 — the S₅ bridge-cell precedent's ~1-in-5 catastrophic
seed rate, measurable only at n≥5) is discharged by §4's A1.3 fix: a split
K=32 result (1≤p≤3/4 Gate-0-passing seeds) mechanically triggers n=8 +
the §R3 disposition clause, pre-priced at +17.2 GPU-h (§5) — see §4 for
the exact frozen trigger and §10's Open-Q3 for the discharge record.

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
- [x] §A1 Attack Round 1 (2026-07-17) — VERDICT: REVISE. Dispositioned in
      §R1 below (F1–F4 fixed, MINORs adjudicated).
- [ ] Independent code audit (fresh agent) of the `expm`/Cayley `encode()`
      overrides + the Stage-0 SVD-floor patch, BEFORE any GPU spend —
      matching `NCR_ORTHO_WRITE.md`'s own gauntlet shape (pre-registration →
      build → independent audit → pre-launch resource/placement audit →
      runner).
- [ ] Pre-launch audit: (a) the §3.5 packing canary; (b) the §3.1
      closed-form eigenvalue check on `D·Q_target` (Cayley only — the
      `expm` reflection check is deleted along with `R`, §A1.1) for the
      ACTUAL entity embedding (cheap, CPU-only — do not launch Cayley
      cells on the unverified assumption that the D-scaling fix's
      reachability proof transfers cleanly to the specific code path).
- [ ] Stage 0 runner + smoke.
- [ ] Stage 1 runner (arm dispatch), smoke, then the gauntlet.

**Open questions — status after §A1/§R1:**

1. **[STILL OPEN, unaffected by F2's construction change.]** Is
   `eps_rel=1e-3` actually the right order of magnitude, or should Stage 0
   itself run a small `eps_rel` sweep (e.g. `{1e-4, 1e-3, 1e-2}` all at
   once, still ≤1.5 GPU-h total) rather than a single value + one reactive
   retry? The single-value choice was made to keep Stage 0 minimal-change
   and cheap; a sweep would cost more but resolve the "wrong `eps_rel`"
   ambiguity in one pass instead of two sequential ones. (The axis is now
   `eps_rel`, not the struck additive `eps` — F2 — but the sweep-vs-
   single-value tradeoff itself is unchanged.)
2. **[STILL OPEN, scope narrowed to Cayley only post-A1.1.]** Does the
   `D·Q_target` closed-form check (flagged above as a pre-launch item)
   actually require reading code this draft did not read (the specific
   entity-embedding construction)? This document reasons from the
   ARCHITECTURAL description in `NCR_ORTHO_WRITE.md` (K-cycle over a
   `d=K+1` tight-spare embedding) but has not inspected
   `ncr_earlyln_scale.py` / `ncr_task.py`'s literal embedding code — the
   independent code audit should confirm the entity-subspace forced-
   eigenvalue argument (§3.1) transfers exactly, not just architecturally.
   (The parallel `R·Q_target` check for `expm` is deleted — no reflection,
   §A1.1.)
3. **[RESOLVED — now a FROZEN rule, §4/§9, A1.3 fix.]** Was n=4 per
   Stage-1 cell adequate given the §2.35 seed-variance precedent? Answer:
   n=4 stands as the default, but a split (1≤p≤3/4 Gate-0-passing seeds)
   at the K=32 decisive cell now mechanically triggers n=8 + the §R3
   disposition clause, pre-priced at +17.2 GPU-h (§5) — no longer left to
   mid-run coordinator discretion.
4. **[STILL OPEN, unaffected by §R1.]** Is the K=24-before-K=32 sequencing
   (§5) actually informative, or could K=24's smaller `d=K+1=25` behave
   qualitatively differently from K=32's `d=33` in ways that make an early
   K=24 FAIL a poor predictor for K=32 — worth pressure-testing against
   §9.1's own secondary K24 data (ortho_K24 failed identically to
   ortho_K32 in the original run, some support for the sequencing's
   informativeness, but only n=1 prior data point).
5. **[RESOLVED — discharged by §A1's own data pull, recomputed under F2's
   revised step count, §2/§9.]** Pulled the archived `free_K24_s{0..3}`
   loss histories: healthy cells reach loss <0.01 by step 7K–12K and
   <0.002 by 10K–22K (worst seed 22K). Stage-0's revised ≈42,000-step
   budget carries `≈1.9×` margin over the worst observed free-arm
   convergence step — comfortably `>1×`; a Stage-0 FAIL is NOT confoundable
   with "ran out of steps." (Caveat, unchanged: this is the FREE arm;
   damped-polar-through-constraint could converge somewhat slower, but the
   margin is not eliminated by F2's revision.)

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

---

## §R1 REVISION 1 DISPOSITION (2026-07-17)

Revision agent, read-write on the body (§1–§10) only; §A1 above is a frozen
record and was not touched. Every finding below is dispositioned with the
exact body section(s) changed and a one-line summary of the change. No
finding was left unaddressed.

| Finding | Disposition | Where fixed | What changed |
|---|---|---|---|
| **A1.1 (FATAL)** — det-parity reachability claim false; reflection `R` unnecessary | **FIXED** | §1 (hypothesis), §3.1 (full rewrite), §3.2 (code + prose), §3.5 (unaffected, no edit needed), §9 (expm risk bullet), §10 (checklist + Open-Q2), preamble block-quote | §3.1 rewritten from the K×K-permutation det leap to the correct `det(Q)=det(P\|_E)·s` account (spare sign is a free 2nd DOF); plain `expm(W)` is now PRIMARY with **no** `R`; Cayley's `D`-scaling re-derived from the forced entity-subspace `−1` eigenvalue (spare-independent), not det-parity. All `R`/reflection code, prose, and the associated pre-launch eigenvalue-parity check for `expm` are deleted. |
| **A1.2 (MAJOR)** — `Z+εI` is not a real σ_min floor for non-normal `Z`; Weyl misapplied; 10× retry chases wrong axis | **FIXED** (exact SVD-clamp, chosen over the smooth `Z(ZᵀZ+εI)^{-1/2}` alternative — see disagreement note below) | §2 (full rewrite), §3.4, §5 (ledger note), §9 (damped-polar bullet), §10 (Open-Q1 relabeled) | Additive `εI` replaced with a detached-SVD-based exact floor: `Z_damped = M @ Z_raw`, `M = U·diag(scale)·Uᵀ` (detached), guaranteeing `σ_min(Z_damped) ≥ eps_rel·σ_max` exactly, for any `Z_raw`. Retry ladder now moves `eps_rel` (10× up/down), the axis that actually controls conditioning. Step-count re-priced (~48K → ≈42K) under a +15% conservative overhead derate; GPU-h ceiling unchanged. |
| **A1.3 (MAJOR)** — split-result seed-escalation left to mid-run discretion | **FIXED** | §4 (new frozen-rule paragraph), §5 (ledger contingency row + ABORT-ON-COST item 6), §9 (seed-variance bullet replaced), §10 (Open-Q3 marked RESOLVED) | Froze the exact trigger: K=32 selected-arm split (`1≤p≤3/4` Gate-0-passing seeds) → `n→8` for that cell, pre-priced `+17.2 GPU-h`, `§R3` disposition clause applies instead of the plain `≥3/4` FAIL band. Pre-authorized, not a coordinator decision at run time. |
| **A1.4 (MAJOR)** — "substitute the arm name" understates downstream work for a non-NS winner | **FIXED** | Preamble block-quote, §4 (arm-selection rule) | GATE-1 discharge stays arm-agnostic (no re-gauntlet), but a new explicit post-WIN task requires re-deriving, for the winning parametrization, before either downstream BUILD executes: (1) per-write FLOPs, (2) backward cost, (3) ceiling/runaway-guard calibration. Named the specific NS-baked artifacts in both downstream docs that don't transfer (K-ladder's NS FLOP term; real-LM's `‖QᵀQ−I‖` stability claim). |
| **A1.5 (MINOR)** — global-orthogonality WIN check is vacuous for structural arms | **FIXED** | §1 (falsifiable prediction), §4 (WIN mechanistic corroboration bullet) | Re-scoped: expm/Cayley score on **entity-block** diagnostics (`min\|λ\|/c*`) only; damped-polar (not structurally confined to `SO(d)`) keeps the original **global** diagnostic, which is non-vacuous for that arm alone. |
| **A1.6 (MINOR)** — no explicit per-write FLOP count for `expm` | **FIXED** | §3.2 (cost/gradient-path bullet) | Added the explicit op count (Padé-13 ≈6 matmuls + 2–4 squarings/forward, ×2-ish backward, ×batch 256) and folded in the attack's own CLEAR sub-findings (torch 2.8 backward stable; target eigenvalue gaps `<2π`, away from `expm`'s singular set). |
| **A1.7 (MINOR)** — `eps` carried K=24→K=32 unvalidated (instrument-relative hard rule) | **FIXED** (cheaper of the two offered fixes: provisional scoping, not re-validation) | §3.4 (new sub-paragraph), §9 (damped-polar bullet) | K=32 damped-polar results are explicitly labeled PROVISIONAL on the un-revalidated `eps_rel` carry-over; a FAIL with the §10.7 ill-conditioning signature reads as "floor didn't generalize," not "damped-polar is dead." No added GPU-h (chose the labeling fix over a full K=32 re-validation sweep). |
| **A1.8 (MINOR)** — no odd-K control cell | **DECLINED, reasoned** (matches the attack's own "moot" characterization) | §3.5 (new adjudication sentence) | Recorded explicitly rather than silently dropped: post-A1.1 there is no remaining parity-vs-spare-freedom distinction for an odd-K cell to test, and no other live hypothesis motivates the extra K-column's cost. |
| **CLEAR items** (σ_min(I+W)≥1 proof; ledger arithmetic; saturation packing; Open-Q5 discharge; mod-K/blank-out/blind-record/novelty) | **No action** — attack found these sound | — | Untouched, per the attack's own verdict. Open-Q5's discharge was carried into §2/§9/§10 and its margin number (`2.2×`→`1.9×`) was re-derived to stay consistent with A1.2's step-count revision. |

**Body sections changed:** preamble (status line + block-quote), §1, §2 (full
rewrite), §3.1 (full rewrite), §3.2, §3.4, §3.5, §4, §5, §9, §10. §3.3, §6,
§7, §8 were read and found to need no edit (no det-parity/reflection
language, no MINOR/MAJOR/FATAL findings against them). §A1 is untouched
(frozen record).

**Disagreement with §A1 (argued, not asserted).** A1.2 offered SVD-clamping
and `Z(ZᵀZ+εI)^{-1/2}` damping as two apparently-interchangeable routes to
"a genuine σ_min floor." They are not equivalent, and this revision uses
SVD-clamping only. Algebra: for `Z(ZᵀZ+εI)^{-1/2}`, the output singular
value is `σ_i' = σ_i/√(σ_i²+ε)`; for `σ_i ≪ √ε`, `σ_i' ≈ σ_i/√ε`, giving
`cond(Z_damped) ≈ √ε/σ_min → ∞` as `σ_min → 0`, for **any fixed `ε`** — no
finite `ε` bounds the worst case, so this construction is a smooth
regularizer, not a floor with a worst-case guarantee. Only the exact clamp
`σ_i ← max(σ_i, eps_rel·σ_max)` gives `σ_min(Z_damped) ≥ eps_rel·σ_max`
unconditionally, which is what F2 explicitly asked for ("a mathematically
sound σ_min floor"). This is a refinement of A1.2's own fix menu, not a
rejection of the finding — the finding's core diagnosis (Weyl misapplied to
a non-normal matrix) is accepted and fixed as directed.

**Other note, not a disagreement.** The §A1 VERDICT summary paraphrases
A1.3 as "the split-result n=5→8 escalation," but A1.3's own body text
specifies `n→8` directly (no intermediate n=5 step) — a minor internal
inconsistency within the frozen attack text itself. This revision
implements A1.3's body text exactly (`n→8`, `+17.2 GPU-h`), not the
verdict summary's paraphrase, since the body text is the operative finding
and the arithmetic (`4 additional seeds × 4.3 GPU-h = 17.2 GPU-h`) only
resolves against `n→8` from a baseline of 4.

---

## §B1 STAGE-0 BUILD RECORD (2026-07-17)

Build-and-launch agent report, against the frozen §2 patch (commit `28e69cb`,
FROZEN CLEAR-FOR-STAGE-0-BUILD). No tunable choices made outside §2's spec;
no ambiguity encountered that required a STOP.

**Files.**

| File | md5 |
|---|---|
| Base (pinned, unmodified) — `experiment-runs/2026-07-16_ncr_ortho_write/ncr_ortho_write.py` | `83b5d7bd273e9e83698fed27a9f2ef45` |
| Patched — `matrix-thinking/ncr/ncr_ortho_fallback_stage0.py` (working copy) | `70dd7923027f4dcf9f0f1e964fc0930c` |
| Archived — `experiment-runs/2026-07-17_ncr_ortho_fallback_stage0/ncr_ortho_fallback_stage0.py` | `70dd7923027f4dcf9f0f1e964fc0930c` (byte-identical to working copy) |
| Deployed on box — `/home/nvidia/ncr/ncr_ortho_fallback_stage0.py` | verified `70dd7923027f4dcf9f0f1e964fc0930c` post-scp (§B1.3) |

Base: 934 lines. Patched: 984 lines (+50 net).

**Diff summary (verified via `diff -u`, every hunk itemized — nothing
outside this list touched):**

1. Docstring: one addendum paragraph documenting the Stage-0 patch (no code).
2. `RUNNER_TAG`: `"ncr_ortho_write_v1"` → `"ncr_ortho_fallback_stage0_v1"`.
3. New constant `STAGE0_EPS_REL_DEFAULT = 1e-3` (§2's pinned primary value).
4. `NCROrthoWriteModel.__init__`: two new kwargs, `damped: bool = False`,
   `eps_rel: float = STAGE0_EPS_REL_DEFAULT`; two new attribute assignments.
5. `NCROrthoWriteModel.encode()`: one new `if self._damped:` block inserted
   inside the existing `if self._orthogonal:` branch — the exact §2 patch
   (`U,S,Vh = torch.linalg.svd(Z, full_matrices=False)` under `no_grad`,
   `S_floor = S.clamp_min(eps_rel*sigma_max)`, `scale = S_floor/S.clamp_min(NS_EPS)`,
   `M = U @ diag_embed(scale) @ Uᵀ`, `Z = M @ Z`), followed by the
   **UNCHANGED** `newton_schulz_polar(Z, n_iter=self._ns_iter, n_power=self._ns_power)`
   call — verbatim per §2, no line inside `newton_schulz_polar` itself touched.
6. `build_primary_model()`: new `eps_rel` kwarg (threaded through) + new
   `if arm == "damped_polar":` branch constructing `NCROrthoWriteModel(...,
   damped=True, eps_rel=eps_rel, ...)`.
7. `primary_cell_id()`: new `if arm == "damped_polar": return
   f"stage0_damped_K{K}_s{seed}"` branch (cell registry).
8. `run_primary_cell()`: signature gains `eps_rel` kwarg; `assert arm in
   (...)` widened to include `"damped_polar"`; `build_primary_model(...)`
   call threads `eps_rel`; result dict's `orthogonal=` field widened to
   `arm in ("ortho", "damped_polar")` and a new `eps_rel=` field recorded
   (`None` for non-damped arms).
9. CLI (`main()`/`argparse`): `--arm` choices widened to include
   `"damped_polar"`; new `--eps-rel` flag (default `STAGE0_EPS_REL_DEFAULT`);
   `--outdir` default changed `results_ortho_write` → `results_ortho_fallback`;
   `run_primary_cell(...)` call threads `args.eps_rel`.

No line inside `newton_schulz_polar`, `orthogonality_error`,
`spectral_diagnostics`, `realistic_ladder_eval`, the Part-B
(`OrthoBankModel`/chain) code, or `_self_test()` was touched — confirmed by
running the pre-existing 9-test CPU `--smoke` suite unmodified against the
patched file: **9/9 PASS**, byte-identical assertions to the base script's
own suite (local CPU run, this session).

**Cell config.** `stage0_damped_K24_s0`: `arm=damped_polar`, `K=24`,
`d=25`, `seed=0`, `eps_rel=1e-3`, `ns_iter=40`, `ns_power=12`,
`anneal_frac=0.5`, `steps=42000`, `--ceiling-gpuh 0.5` (passed explicitly on
the CLI, not a changed script default — the base script's `--ceiling-gpuh`
default of `3.0` is untouched in the diff; Stage-0's `0.5` ceiling is a
launch-command choice, not a code change). Output dir:
`/home/nvidia/ncr/results_ortho_fallback/`.

**Budget.** ≤0.5 GPU-h this attempt; ≤1.0 GPU-h Stage-0 total including one
pre-authorized `eps_rel` retry (10× up `1e-2` on same-signature FAIL, 10×
down `1e-4` on different-signature FAIL — §2 branch logic, not pre-launched
here, contingent on this attempt's outcome).

**Launch command (box, inside the tmux driver — §B2 for the full
supervisor wrapper):**

```
CUDA_VISIBLE_DEVICES=<N> /home/nvidia/tdenv/bin/python3 \
  /home/nvidia/ncr/ncr_ortho_fallback_stage0.py --primary-cell \
  --arm damped_polar --K 24 --seed 0 --steps 42000 --eps-rel 1e-3 \
  --ns-iter 40 --ns-power 12 --anneal-frac 0.5 --ceiling-gpuh 0.5 \
  --outdir /home/nvidia/ncr/results_ortho_fallback
```

See §B2 (below) for the CUDA smoke results, pre-launch red-team answers,
and final launch/placement details.

---

## §B2 STAGE-0 SMOKE + RED-TEAM + LAUNCH RECORD (2026-07-17)

**PROCESS-GAP FLAG (read first).** §10's own checklist lists *"Independent
code audit (fresh agent) of the expm/Cayley `encode()` overrides + the
Stage-0 SVD-floor patch, **BEFORE any GPU spend**"* as a distinct, still
**unchecked** `[ ]` item, separate from §A1's attack round (which reviewed
the pre-registration text, not the code — the code did not exist until this
build). This build-and-launch agent's own dispatch instructions specified
BUILD → RECORD → CUDA-smoke → self-check red-team → blind-launch, with no
separate code-audit dispatch step, and the agent proceeded on that basis —
by the time this gap was noticed (mid-build-record write-up, after the CUDA
smoke and the real launch were already in flight) real GPU-h had already
been spent. **Not self-resolved**: per CLAUDE.md's own hard rule ("Audit
code with a separate agent before running experiments... A self-audit by
the same implementer is not a substitute for an independent audit"), the
9/9 CPU self-test + the CUDA smoke below are necessary but do not discharge
this checklist item — they are the implementer's own verification, not an
independent review. Coordinator: either (a) rule that Stage-0's light
`<10 GPU-h` ceremony tier (§ preamble, CLAUDE.md's `<10 GPU-h → 1 audit
round`) is satisfied by §A1's already-complete attack round even though it
predates the code, and tick the checklist item on that basis, or (b)
dispatch a fresh independent code-audit agent against this diff now (cheap,
the whole file is 984 lines, the diff is ~50 net lines, itemized in §B1)
and treat this run's result as provisional until that lands. The agent did
not kill the in-flight run over this (§B2.3 below explains why), but is
not deciding this gate itself.

### §B2.1 CUDA smoke (real GPU, `youthful-indigo-turkey`, not a CPU stub)

Ran `/home/nvidia/ncr/cuda_smoke_stage0.py` (scratch verification script,
not part of the frozen file) on GPU 0 while it was already carrying an
unrelated 100%-SM production job (see §B2.3). All four checks **PASS**:

| Check | Result |
|---|---|
| s1: SVD-floor property (`σ_min(Z_damped) ≥ eps_rel·σ_max(Z_raw)`), CUDA fp64 | PASS, min margin −9.5e-12 (numerical noise, exact to tolerance) |
| s2: forward + backward through the damped `encode()` path; explicit per-param grad-norm finiteness | PASS — 47/47 param grad norms finite & nonzero (range 1.18e2–1.90e4); `keys`/`values` input grads finite & nonzero |
| s3: end-to-end micro cell (`run_primary_cell`, `damped_polar`, K=24, steps=5) at the **production** `EVAL_BATCH_SIZE=256`/`EVAL_BATCHES=8` (no tiny override — the eval-OOM hard rule requires testing the real eval batch size) | PASS — COMPLETED in 35.6 s, `eval_cell` + `realistic_ladder_eval` both ran clean, no OOM; peak CUDA memory **0.214 GB** |
| s4: whole-cell checkpoint/resume (the harness's only checkpoint unit — `ncr_earlyln_scale.py`'s own documented design has no mid-cell checkpoint, "cells are short, whole-cell skip-if-COMPLETED is the resume-safety unit") | PASS — second call detected the COMPLETED JSON and skipped without re-running |

Smoke output dir (`results_ortho_fallback_SMOKE/`) removed after the check;
`results_ortho_fallback/` (the real output dir) was empty before launch —
verified.

### §B2.2 Pre-launch red-team (self-check)

- **Fits in memory?** Yes, trivially. Smoke measured 0.214 GB peak at 5
  steps; the parent §9.0 measured ~2.4 GB for a full 320K-step ortho cell
  at the *same* K=24/d=25 shape (Adam states + activations dominate over
  the 5-step smoke). Either figure sits far under the ~36–37 GB free per
  GPU (all 8 GPUs were carrying a large co-tenant job at launch time, see
  below) — memory is not the constraint, matching §3.5's own "never the
  constraint" finding.
- **Timeout/ceiling wired?** Yes, two layers: (1) the frozen internal
  `--ceiling-gpuh 0.5` (1800 s), checked every `log_every=500` steps inside
  `train_earlyln_cell` (unmodified — inherited from the base script), which
  returns `ABORTED-BUDGET` gracefully; (2) an external `timeout 2400`
  backstop in the driver script (`orchestration/run_stage0.sh`), 600 s of
  margin above the internal ceiling, matching the parent run's own
  timeout-above-ceiling discipline (catches a hang, not a graceful abort).
- **Duplicate check?** Confirmed clean both before build (`ls
  /home/nvidia/ncr/results_ortho_fallback/` → "No such file or directory")
  and again immediately before launch (same result) — no prior Stage-0
  results to collide with.
- **Gate discharged?** Verified via `git log`/`git rev-parse HEAD`: the
  repo's `HEAD` is exactly `28e69cb` (the frozen "CLEAR-FOR-STAGE-0-BUILD"
  commit) for `NCR_ORTHO_FALLBACK_DESIGN.md`, with no commits after it
  touching this file — this build agent's own edits (§B1/§B2) are the only
  uncommitted (`git status` shows `M`) changes, awaiting the coordinator's
  commit per the task's own instructions. (See §B2's PROCESS-GAP flag above
  for the one checklist item this does NOT discharge.)
- **Placement — GPU + contention, MEASURED not assumed.** All 8 GPUs were
  at 99–100% SM utilization at launch time, each running one unrelated
  392M-param `lm_pretrain_rd.py` production job (~43.5–44.1 GB/80 GB used,
  ~37 GB free each) — **not** the parent run's own N=2/GPU packing case
  (two similarly-sized NCR cells); this is one NCR cell riding alongside a
  large, unrelated, already-saturating job. Launched on GPU 0 (arbitrary —
  all 8 read within noise of each other; GPU 0 was also the smoke GPU).
  Memory-wise this is fine (1.086 GB used post-launch, per
  `nvidia-smi --query-compute-apps`, out of ~37 GB free). **SM-contention
  risk materialized and was measured, not merely a listed risk**: see
  §B2.4 — the achieved training rate under contention is ~3.3× slower than
  the design's solo-calibrated estimate, threatening the pre-registered
  step-count margin. This is the single most important finding in this
  record; do not skip §B2.4.

### §B2.3 Why the run was NOT killed after the contention finding (§B2.4)

The rate deviation in §B2.4 was measured ~2–3 minutes into an already-live
run, not before launch (the box was saturated on all 8 GPUs at build time —
there was no idle GPU to canary against, and the doctrine + this agent's
own instructions forbid killing another job to make room). At that point:
(1) the frozen ceiling is wall-clock-based, not step-count-based, by
design (§2: *"the ceiling is a wall-clock cap"*) — the harness already
degrades gracefully to `ABORTED-BUDGET` rather than running away or
corrupting anything; (2) killing and relaunching elsewhere would not help
— every GPU was equally saturated, so a relaunch would very likely
reproduce the same contention; (3) the total cost at stake is the
pre-priced ≤0.5 GPU-h ceiling regardless of outcome. Continuing was the
lower-risk choice; the alternative (killing an in-flight, ceiling-bounded,
smoke-verified run on a discretionary judgment call) would itself be
outside this agent's "no tunable choices, STOP-and-report" mandate.
**Flagging prominently instead of silently accepting the outcome is the
STOP-and-report response here** — the run continues, but its result should
be read through §B2.4's caveat, not taken as a clean Stage-0 verdict
without checking the achieved step count first.

### §B2.4 MEASURED contention finding (load-bearing caveat for ASSESS)

Log timestamps (elapsed-seconds field only — no loss/metric values read,
per the blind-launch instruction):

| step | elapsed |
|---|---|
| 1 | 1 s |
| 500 | 71 s |
| 1000 | 140 s |

Steady-state rate ≈ **139 ms/step** (step 1→1000: 139 s / 999 steps).
The frozen §2 solo-calibrated estimate was ~3.3 h/320K steps + 15% derate
≈ 3.795 h/320K steps ≈ **42.7 ms/step** — the measured contended rate is
**~3.3× slower**. Extrapolating linearly, the 1800 s (0.5 GPU-h) internal
ceiling will be reached at roughly **step ≈ 12,900** — about **31%** of the
pre-registered ≈42,000-step target, NOT the full budget. §2's own margin
arithmetic (*"≈42,000 achievable steps... margin over the worst observed
free-arm convergence step [22,000] ≈ 1.9×"*) assumed the solo rate; at
~12,900 achieved steps the effective margin is **≈0.59×** — UNDER 1×,
i.e. this attempt is likely to hit `ABORTED-BUDGET` before even the
free-arm's own worst-case convergence point, let alone give the
damped-polar arm room to converge. **Consequence for ASSESS:** if this
cell's terminal JSON shows `status=ABORTED-BUDGET` with `train.step` well
below ~40,000, that is a **contention/placement artifact, not a
same-signature or different-signature FAIL** per §2's branch logic — none
of the three pre-registered branches (PASS / same-signature FAIL / 
different-signature FAIL) anticipated a budget exhaustion this early, and
applying any of them mechanically to a step-starved run would be an
instrument-relative mis-read, exactly the class of error the doctrine's
own "read the raw artifact, don't average or default" rule (CLAUDE.md Hard
Rules) exists to prevent. **Recommendation:** the coordinator should check
`train.step` in the output JSON before applying §2's branch logic; if it
terminates well short of ~40K steps, the honest read is "inconclusive —
contention-starved, re-run once a less-saturated GPU-hour is available,"
not a PASS/FAIL/retry decision, and the ≤1.0 GPU-h Stage-0 ceiling should
arguably be treated as not yet properly spent (a genuine solo attempt has
not yet been made).

### §B2.5 Launch details

- **tmux session:** `ncr_fb0_g0` (box `youthful-indigo-turkey`), running
  `orchestration/run_stage0.sh 0` — confirmed alive post-launch and again
  at the last check in this record.
- **Driver:** self-healing supervisor (`while [ ! -f STOP ]; do ...; sleep
  15; done`), but retry-gated on TERMINAL status (`COMPLETED` or
  `ABORTED-BUDGET`) via the cell's own JSON `status` field, not a naive
  infinite retry-on-any-exit loop — an infinite loop would re-spend the
  §2 ceiling on every restart after a graceful abort, violating the frozen
  ≤1.0 GPU-h Stage-0 total. Retries ONLY on a genuine crash (process died
  without writing a valid terminal-status JSON).
- **Output:** `/home/nvidia/ncr/results_ortho_fallback/stage0_damped_K24_s0.json`
  (not yet written as of this record — still training); log:
  `/home/nvidia/ncr/results_ortho_fallback/run_stage0.log`.
- **Expected completion (at the MEASURED, not pinned, rate):** given
  §B2.4's ~139 ms/step and the 1800 s internal ceiling, expect
  `ABORTED-BUDGET` at ≈1800 s (~30 min) wall-clock from launch
  (2026-07-17T07:13Z), i.e. **≈07:43Z**, at ≈step 12,900 — NOT a
  `COMPLETED` at step 42,000 as the solo pin implied. The external 2400 s
  timeout backstop is not expected to fire (the internal ceiling should
  fire first, gracefully).
- **Blind discipline maintained:** no loss/recovery/spectral metric value
  was read at any point in this record — only `status`, `step`, and
  elapsed-seconds fields (process-liveness/throughput, not science).

## §B3 COORDINATOR RULING — CONTENTION VOID + RE-PRICED CEILING + AUDIT GATE (2026-07-17, pre-abort, no results read)

Recorded BEFORE the running attempt terminates and before any metric is
read. Two §B2 flags adjudicated:

**(1) Contention void.** The first Stage-0 attempt (tmux ncr_fb0_g0,
launched 07:13Z) runs at a measured ~139 ms/step vs the solo-calibrated
~42.7 ms/step (~3.3×) because all 8 GPUs carry unrelated 392M production
jobs — a placement regime the Stage-0 spec never priced (its 0.5 GPU-h
ceiling assumed the solo rate; §5's contention pricing covered Stage-1
N=2 NCR-cell packing only). The expected ABORTED-BUDGET at ≈step 12,900
(~31% of the ≈42K target, ~0.59× of worst-seed convergence) is therefore
**VOID-CONTENTION — not a §2 PASS/FAIL branch event**. §2's branch logic
applies only to attempts reaching ≥40K steps or terminating for a
non-budget reason. Precedent: the parent run's ceiling amendment
(62a6fb6).

**(2) Re-priced ceiling + relaunch protocol.** Stage-0 attempts under
the current contention regime get `--ceiling-gpuh 1.75` (0.5 solo-
equivalent × 3.3 measured, rounded up) with the external timeout raised
to match (7200s). Worst-case Stage-0 spend becomes: 0.5 (voided attempt,
priced) + 1.75 (amended attempt) + 1.75 (the one pre-authorized eps
retry, if triggered) ≈ **4.0 GPU-h** — still inside the <10 GPU-h
ceremony tier; Stage-1 ledger unchanged. No mid-cell resume exists
(whole-cell checkpoint unit, §B2), so the voided attempt's compute is
written off.

**(3) Audit gate (the §B2 process-gap flag).** The design §10 checklist
item "independent code audit before GPU spend" was not discharged before
the first launch (coordinator dispatch omission, owned here). Ruling:
the item is discharged for the AMENDED attempt by an independent audit
of the ~50-line build diff (base 83b5d7bd → patched 70dd7923) running
NOW, in parallel with the doomed first attempt; the relaunch fires only
after (a) the audit returns clean and (b) the first attempt has
terminated. If the audit finds a defect, the fix re-enters this ruling's
protocol with a fresh md5 pin.

---

## §B4 INDEPENDENT CODE AUDIT (2026-07-17)

Fresh agent, read-only against the repo except this appended section.
Diff regenerated independently (`diff -u`) rather than trusting §B1's
itemization; md5s reconfirmed: base
`experiment-runs/2026-07-16_ncr_ortho_write/ncr_ortho_write.py` =
`83b5d7bd273e9e83698fed27a9f2ef45` (934 lines), patched
`experiment-runs/2026-07-17_ncr_ortho_fallback_stage0/ncr_ortho_fallback_stage0.py`
= `70dd7923027f4dcf9f0f1e964fc0930c` (984 lines, +50 net) — both match
§B1's stated hashes.

### Per-item verdicts

**1. Spec fidelity — PASS (with a caveat inherited from the spec itself,
see finding D1).** The inserted block, in `NCROrthoWriteModel.encode()`:

```python
if self._damped:
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        sigma_max = S[..., :1]
        S_floor = S.clamp_min(self._eps_rel * sigma_max)
        scale = S_floor / S.clamp_min(NS_EPS)
        M = U @ torch.diag_embed(scale) @ U.transpose(-1, -2)
    Z = M @ Z
Z = newton_schulz_polar(Z, n_iter=self._ns_iter, n_power=self._ns_power)
```

matches §2's pseudocode symbol-for-symbol: `M` is built from `U` on
**both** sides (`U·diag(scale)·Uᵀ`), never `Vh` — verified this is what
§2 actually specifies ("`M` is built from left singular vectors only"),
not a paraphrase error. `newton_schulz_polar` is called unmodified,
receiving the damped `Z` — it internally re-derives its own detached
`sigma_hat` via power iteration exactly as §2 describes ("UNCHANGED call,
damped input"); no line inside `newton_schulz_polar` differs from the
base file (confirmed by direct region diff, not just the top-level
`diff -u`). Detachment is correct: the SVD and `M` construction sit
inside `with torch.no_grad():`; the final `Z = M @ Z` is **outside** that
block (same indentation as the `with` statement), so `M` carries no grad
history and `Z` retains its normal graph — `dL/dZ_raw = Mᵀ @
dL/dZ_damped = M @ dL/dZ_damped` (M is symmetric), exactly the
straight-through gradient §2 specifies. `eps_rel` plumbing is complete
and correct end-to-end: `--eps-rel` CLI flag → `args.eps_rel` →
`run_primary_cell(eps_rel=...)` → `build_primary_model(eps_rel=...)` →
`NCROrthoWriteModel(eps_rel=...)` → `self._eps_rel` → used at the one
call site. No dead or orphaned parameter.

**2. The floor property — FATAL, see D1.** Independently re-derived the
algebra and ran a numeric check (CPU, float64, non-normal `Z`, scratch
script) rather than trusting §2's worked proof. Confirmed the claimed
identity `M @ Z_raw = U·diag(scale⊙σ)·Vᵀ` is right, but the "exactly
`σ_floor,i = max(σ_i, eps_rel·σ_max)`, regardless of normality... even if
σ_min random-walks to exactly 0" claim is **false as coded** whenever a
raw singular value falls below `NS_EPS = 1e-7` — see D1 for the full
derivation and reproducing numbers.

**3. Arm dispatch — PASS, clean.** `--arm` routes correctly:
`build_primary_model` has one `if/elif/elif/raise` per arm, no fallthrough.
`"free"` returns `els.NCREarlyLNModel` untouched by any of this diff.
`"ortho"` constructs `NCROrthoWriteModel(..., orthogonal=True)` with
`damped` left at its default `False` — inside `encode()`,
`if self._orthogonal: if self._damped: ...` is skipped entirely (the
inner `if` is simply not entered), so the executed statements are
byte-identical to the base file's `encode()` for this arm: control-flow
and data are unchanged, confirmed by diffing the reachable statement
sequence, not just reading the source side-by-side. `"damped_polar"` is
the only path that sets `damped=True`, and it is only reachable via
explicit `--arm damped_polar` — no other arm or code path can enter the
new block. The **control property holds**: `free` and `ortho` are
provably identical to the pinned parent script's behavior.

**4. Bounds/edge cases — MIXED, see D1 for the load-bearing one.**
Divide-by-zero itself is guarded (`S.clamp_min(NS_EPS)` in the
denominator prevents literal `0/0`/`inf`/`nan` — confirmed, no crash
observed at any tested singular-value magnitude down to exact `0.0`
input); the guard's cost is D1, a silent math defect, not a crash — so
this is **not** a MAJOR "crash risk" item, it is the FATAL "wrong math"
item under a different name. Batch dimensions: `Z` is `(...,d,d)`
(`TRAIN_BATCH=256` batched); `torch.linalg.svd(Z, full_matrices=False)`,
`diag_embed`, and the two batched matmuls all operate per-item correctly
under PyTorch's standard batched linear-algebra semantics — verified
shapes numerically (`U:(B,d,d)`, `S:(B,d)`, broadcasting `S` against
`S[...,:1]` is correct). Degenerate/near-degenerate singular values
(the healthy `free_K24` convergent regime, `cond≈1`, per §2's own note)
pose no problem for this specific construction even though `U` is
non-unique in a degenerate subspace: `M` restricted to a block of equal
`σ` reduces to `scale_const · I` on that subspace, which is invariant to
which orthonormal basis `U` happens to return — verified algebraically,
not just asserted. Dtype: no `autocast`/`.half()`/AMP anywhere in either
file (confirmed by grep) — training runs in plain fp32 throughout,
consistent with the rest of the base script; the new SVD call inherits
`Z`'s device and dtype automatically, no explicit device-management bug.

**5. Record fields — PASS.** `rec = dict(..., arm=arm, ...,
eps_rel=(eps_rel if arm == "damped_polar" else None), ..., runner_tag=
RUNNER_TAG, ...)` — all three required fields (`arm`, `eps_rel`,
`runner_tag`) are present in the result JSON, confirmed at the exact
line in `run_primary_cell`. `--ceiling-gpuh` / `ceiling_gpuh` is
untouched by the diff (default `3.0` unchanged; `ceiling_s = ceiling_gpuh
* 3600.0`, passed unmodified into `els.train_earlyln_cell`, a function in
a different, untouched file) — confirmed functional and unchanged, not
just "not in the diff."

**6. No collateral edits — PASS, clean.** Independently regenerated
`diff -u` shows exactly 8 hunks, all falling inside the 9 items §B1
itemizes (two of §B1's items share one contiguous hunk). Region-diffed
`_self_test()`, the `OrthoBankModel`/Part-B chain
(`class OrthoBankModel` through `run_disc_cell`), `spectral_diagnostics`,
`eval_cell`, `blank_out`, `axis_c_lock`, `z_dump`, and the core NS-40
iteration loop and `orthogonality_error` — all byte-identical to the
base file outside the itemized hunks (verified via targeted `diff`, not
inferred from the top-level diff alone). Zero hunks touch any of these.

**7. §B1's itemization — PASS, honest.** Compared §B1's 9-point list
against the independently regenerated diff line-by-line: every item is
accurate, nothing is omitted, no hunk exists outside the 9 described
changes. §B1's md5s, line counts (934→984, +50 net), and per-item
description all check out exactly.

### Defects found

**D1 (FATAL — wrong math).** The floor property §2 exists to guarantee —
`σ_min(Z_damped) ≥ eps_rel·σ_max(Z_raw)` "exactly, for every `Z_raw`,
regardless of normality... even if the raw encoder's true `σ_min`
random-walks to exactly 0" — **does not hold as coded (nor as literally
specified in §2's own pseudocode, which the code matches exactly)
whenever a raw singular value `σ_i < NS_EPS = 1e-7`.**

*Derivation.* `M @ Z_raw`'s `i`-th singular value is `scale_i · σ_i`,
not `S_floor_i` directly. `scale_i = S_floor_i / max(σ_i, NS_EPS)`. When
`σ_i ≥ NS_EPS`, `max(σ_i,NS_EPS)=σ_i` and the division-then-multiplication
cancels exactly: `scale_i·σ_i = S_floor_i` — the floor holds exactly, to
floating-point precision, as claimed. **When `σ_i < NS_EPS`**, the clamp
breaks the cancellation: `scale_i·σ_i = S_floor_i·(σ_i / NS_EPS)`, which
is `S_floor_i` scaled DOWN by the ratio `σ_i/NS_EPS < 1` — the achieved
singular value degrades toward zero as `σ_i→0`, rather than saturating at
`eps_rel·σ_max`. At the literal `σ_i=0` limit the achieved value is
exactly `0`, not `eps_rel·σ_max` — directly contradicting §2's own quoted
claim for that exact case.

*Numeric confirmation* (float64, CPU, non-normal `Z = U₀ diag(σ) V₀ᵀ`,
`d=5`, `eps_rel=1e-3`, reproduced from a fresh random non-normal frame,
script discarded after use):

| raw `σ_min` | target floor (`eps_rel·σ_max`) | achieved `σ_min(Z_damped)` | floor holds? |
|---|---|---|---|
| `1e-4` (`> NS_EPS`) | `5.00e-3` | `5.00e-3` | YES (exact) |
| `1e-9` (`< NS_EPS`) | `5.00e-3` | `5.00e-5` | **NO — 100× short** |
| `≈0` (machine noise `~2e-16`) | `5.00e-3` | `1.02e-11` | **NO — ~9 orders short** |

Confirmed the failure is caused specifically by the `NS_EPS` clamp, not
by anything fundamental to the `M@Z_raw` construction at moderately small
`σ`: removing the clamp (`scale = S_floor / S`, no floor on the
denominator) reproduces the exact target floor at every tested magnitude
down to and including literal `σ_i=0` input, with no `NaN`/`Inf`, because
float64 division-then-multiplication by the *same* operand cancels
exactly regardless of how small that operand is (the unclamped
construction only risks a literal `0/0` on a bit-exact-zero *computed*
singular value, an edge case the current `NS_EPS` guard over-corrects for
by four orders of magnitude relative to what's needed).

*Why this is FATAL, not MINOR.* `NS_EPS = 1e-7` is a general-purpose
divide-by-zero guard, reused verbatim from an unrelated normalization
context earlier in the file (the power-iteration `v.norm().clamp_min(eps)`
calls, and the `orthogonality_error`/`cond` diagnostics). Its magnitude
was never chosen with this construction's exactness requirement in mind.
It happens to coincide almost exactly with §10.2's own diagnosed danger
threshold — the design doc states the fix must protect against `σ_min`
crossing "`~1e-7`" (§2, "Cost of the fix") and separately states the
translated danger threshold is "`σ_min≲1e-7`" (§10.2, cited verbatim in
§2's own cost-of-fix paragraph). That is precisely the regime in which
this guard silently disables the floor. The entire Stage-0 branch-logic
(PASS / same-signature-FAIL / different-signature-FAIL, §2) assumes the
floor is exact and attributes a same-signature FAIL to `"eps_rel
undersells the actual random-walk rate"` — but if `σ_min` does drift
below `1e-7` during the run (the exact scenario §10 diagnosed happening
over training), a same-signature FAIL could instead be caused by this
undisclosed leak, not by `eps_rel` being the wrong order of magnitude.
Retrying with `eps_rel=1e-2` (the pre-authorized branch) would not fix
this — the leak is in the `NS_EPS` denominator guard, an axis the
pre-registered retry ladder never touches. This is a defect in §2's own
worked derivation (the code faithfully implements the spec's flawed
pseudocode, verbatim) — fixing it requires amending both the design doc
and the code, not the code alone.

*Suggested fix (for the coordinator/build agent, not applied here — this
audit is read-only except this section).* Decouple the denominator guard
from `NS_EPS`: use a guard several orders of magnitude below any
`eps_rel·σ_max` value ever expected (e.g. `S.clamp_min(eps_rel * sigma_max
* 1e-6)` or an absolute `torch.finfo(S.dtype).tiny`-scale constant), so
the guard only prevents literal `0/0` and no longer interferes with the
construction anywhere near the operating region `NS_EPS` currently
clobbers. Re-verify with the same numeric check (the harness script used
here, reconstructable in under 10 lines) before re-launching.

*Coverage gap, not a false claim (MINOR).* §B2.1's CUDA smoke check `s1`
reported "PASS, min margin −9.5e-12" — this is consistent with a smoke
input whose synthetic `σ_min` sat comfortably above `NS_EPS` (i.e., it
exercised the "floor holds" branch of the table above, not the
"`σ_min<NS_EPS`" branch). The smoke suite's PASS is truthful for what it
tested; it simply never tested the regime that matters, and nothing in
§B1/§B2's record claims it did. Flagged so a future smoke revision adds a
synthetic sub-`NS_EPS` input as an explicit test case.

**No other defects found.** Items 1, 3, 4 (except D1's crash-adjacent
sub-point, which is not a crash), 5, 6, 7 are all clean — no wrong
routing, no collateral edits, no undisclosed diff content, no record-field
gaps, no ceiling-wiring regression.

### Independent floor-property check result

**FAILED.** `σ_min(Z_damped) ≥ eps_rel·σ_max(Z_raw)` does **not** hold
exactly as coded whenever `σ_min(Z_raw) < NS_EPS = 1e-7` — confirmed both
algebraically and numerically (table above). It **does** hold exactly
(to floating-point precision) whenever `σ_min(Z_raw) ≥ NS_EPS`. Given
§10.2's own measured danger threshold sits at essentially the same
`~1e-7` magnitude as `NS_EPS`, this is not a remote corner case — it is
centered on the exact input regime Stage 0 exists to probe.

### Ruling

**BLOCKED(fixes).** One FATAL defect (D1): the `NS_EPS`-clamped
denominator in the SVD-floor construction silently breaks the "exact,
regardless of normality" floor guarantee precisely in the `σ_min≲1e-7`
regime that motivates this entire fallback design, undermining the
pre-registered branch-logic's causal interpretation (a same-signature
FAIL could be a floor-leak artifact, not evidence about `eps_rel`'s
magnitude). Everything else audited clean (spec fidelity outside D1, arm
dispatch and the undamped-path control property, batch/dtype/device
handling, record fields, ceiling wiring, zero collateral edits, honest
§B1 itemization). **Required before relaunch:** amend §2's construction
(and its own worked proof) to decouple the denominator guard from
`NS_EPS`, patch the one line in `encode()` accordingly, re-run the 9/9
CPU self-test plus a numeric floor-property check spanning `σ_min` both
above and below the old `NS_EPS` threshold, get a fresh md5 pin, and
re-submit for audit (or a lighter delta-only re-check against this
record, coordinator's call) before spending further GPU-h. The
in-flight contended attempt (tmux `ncr_fb0_g0`) was already ruled VOID by
§B3 on unrelated (contention) grounds — this finding does not change
that disposition, it blocks the *next* (re-priced) attempt until fixed.

---

## §B5 D1 FIX + VERIFICATION RECORD (2026-07-17, build agent; relaunch still GATED on the delta re-audit)

**Fix applied exactly as directed (§B4 suggestion, coordinator-selected
variant):** the SVD-floor denominator guard `S.clamp_min(NS_EPS)` →
`S.clamp_min(torch.finfo(S.dtype).tiny)` — a pure 0/0 guard decoupled from
`NS_EPS`, plus a comment block marking it as the D1 fix. §2's pseudocode
line corrected to match (marked inline). The three pre-existing
`clamp_min(NS_EPS)` sites elsewhere in the file (`orthogonality_error`,
`chain_read` renorm, `bank_orthogonality` cond) are the unrelated base-
script contexts D1 identified the guard was wrongly *reused from* —
deliberately untouched.

**File pins (v1 preserved as-run, per the reproducibility rule):**

| File | md5 |
|---|---|
| v1 (AS-RUN, VOID-CONTENTION attempt) — `experiment-runs/2026-07-17_ncr_ortho_fallback_stage0/ncr_ortho_fallback_stage0.py` | `70dd7923027f4dcf9f0f1e964fc0930c` (unmodified) |
| v2 (D1-fixed) — `...same dir.../ncr_ortho_fallback_stage0_v2.py` | `ce1448ab3d47536ebf3e82b146e33722` |
| v2 working copy — `matrix-thinking/ncr/ncr_ortho_fallback_stage0.py` | `ce1448ab3d47536ebf3e82b146e33722` |
| v2 on box — `/home/nvidia/ncr/ncr_ortho_fallback_stage0.py` | `ce1448ab3d47536ebf3e82b146e33722` (verified post-scp) |
| Verification script (archived) — `...same dir.../verify_d1_fix.py` | `c5d7a4282e0c8b881843e8a31c9cefd2` |

v1→v2 diff: 11 lines (the one guard line replaced + a 9-line comment
block + 1 context). No other change. 9/9 CPU self-test suite re-run on v2:
ALL PASS.

**Verification (real CUDA, H100, through the ACTUAL fixed `encode()` code
path — module-attribute capture of `newton_schulz_polar`'s input, not a
scratch re-implementation; the loaded module was asserted to carry the
fixed guard and not the old one before any numbers were taken).**

§B4's exact table, float64, non-normal `Z = U₀ diag(σ) V₀ᵀ` (`U₀≠V₀`),
`eps_rel=1e-3`, `σ_max=5` (target floor `5.000e-3`), at §B4's `d=5` AND
production `d=25`:

| d | raw σ_min (set) | computed raw σ_min | achieved σ_min(Z_damped) | floor |
|---|---|---|---|---|
| 5 | 1e-4 | 1.000e-4 | 5.000000e-3 | **EXACT** |
| 5 | 1e-9 | 1.000e-9 | 5.000001e-3 | **EXACT** |
| 5 | literal 0.0 | 2.76e-17 (noise) | 7.314e-3 | above target (noise regime, see F-A) |
| 25 | 1e-4 | 1.000e-4 | 5.000000e-3 | **EXACT** |
| 25 | 1e-9 | 1.000e-9 | 5.000000e-3 | **EXACT** |
| 25 | literal 0.0 | 5.79e-16 (noise) | 8.874e-4 | **~0.18× target (F-A)** |

**The D1 defect is fixed on its own axis:** every resolvable-σ row —
including `1e-9`, two orders below the old `NS_EPS` threshold where v1
fell 100× short — is now EXACT (rel tol 1e-10). fp32 resolvable-regime
check (B3): exact to fp32's intrinsic SVD accuracy (achieved/target
0.99789; the intrinsic absolute-error bound `ε·σ_max` gives ~6e-3 rel at
this σ_min — fp64-style 1e-10 exactness is not defined for fp32 here).
CUDA grad smoke, healthy regime (B1): 47/47 param grad norms finite &
nonzero on v2. Realistic fp32 trap-regime input (B2b: stored σ_min at
fp32 noise level 2.4e-8, sub-`NS_EPS`, NOT bit-exact zero — the regime
§10 actually diagnosed): encode() finite, grads finite & nonzero, floor
achieved at 2.5× target (noise-regime overshoot, benign direction).

**TWO NEW FINDINGS surfaced by this verification — for the delta
re-audit, NOT self-adjudicated here:**

- **F-A (noise-floor approximation — construction-intrinsic, no guard
  fixes it).** At inputs whose computed σ_min sits at the dtype's noise
  floor (`~ε·σ_max`; the literal-0.0 rows), the achieved floor is
  O(1)-approximate, not exact: measured achieved/target ratios across 6
  fp64 frames = {1.46, 0.43, 0.33, 0.18, 0.15, **0.00**}. Mechanism: the
  SVD's own backward error (`~ε·σ_max` absolute) is amplified by
  `scale ~ S_floor/σ_noise` to exactly floor magnitude, so the achieved
  value there is uncontrolled to an O(1) factor — and left-multiplication
  `M @ Z` can never raise the rank of an effectively-singular matrix.
  **This bounds what ANY denominator guard can deliver**; §B4's D1 text
  ("removing the clamp reproduces the exact target floor... down to and
  including literal σ_i=0") was seed/shape-lucky at its d=5 test — my
  d=5 run also landed above target, my d=25 frames landed anywhere in
  [0, 1.46]×. Consequence for §2's claim language: "exact for every
  Z_raw, even σ_min exactly 0" must be re-scoped to "exact for every
  σ_min the dtype's SVD resolves (≳ ε·σ_max); order-of-magnitude-only at
  the noise floor." For Stage-0's purpose (bound cond ~1e3 vs the 1e7+
  danger) an O(1)-loose floor still delivers ~3 orders of safety margin
  in the typical case, but the 0.00× frame shows the worst case is NOT
  bounded — the coordinator/re-audit must decide whether this is
  acceptable for the causal-attribution claim or whether the
  construction needs a different mechanism at the noise floor.
- **F-B (fp32 bit-exact-zero → non-finite forward — NEW defect in the
  prescribed guard at its own extreme).** With computed σ bit-exact 0.0
  (fp32 SVD of a matrix with exactly-zero columns returns exact zeros),
  `scale = S_floor/finfo(fp32).tiny ≈ 8e35`; `Z_damped` reaches ~8e29
  magnitudes (amplified catastrophic cancellation) and the downstream NS
  polar goes **non-finite (encode() output Inf/NaN — reproduced, B2)**.
  The old `NS_EPS` guard masked this (scale capped ~1e5) at the cost of
  D1. Mitigating context: (a) bit-exact-zero computed σ requires
  structured inputs (exact zero rows/columns) — a generic dense fp32
  encoder output lands at noise level (~1e-7·σ_max), NOT exact zero
  (B2b confirms that realistic regime is finite and well-behaved); (b)
  the training loop's existing finite-grad check skips non-finite steps
  (`n_skipped`), so even a hit degrades to a skipped step in training —
  but eval/z_dump paths have no such guard. Options for the re-audit
  (NOT applied — no authority): §B4's own alternative guard
  `S.clamp_min(eps_rel * sigma_max * 1e-6)` caps scale at 1e6,
  eliminating F-B's overflow while sacrificing nothing fp32 can resolve
  anyway (fp32's noise floor `~1.2e-7·σ_max` sits above that clamp's
  bite point `1e-9·σ_max`); it would slightly loosen fp64 exactness in
  the [tiny, 1e-9·σ_max) band, which production (fp32) never occupies.

**First attempt terminal status (structure only, §B3's VOID-CONTENTION
disposition unchanged):** `status=ABORTED-BUDGET`, `train.step=13,500`,
`elapsed_s=1823` vs `ceiling_s=1800` — the internal ceiling fired
gracefully at the first 500-step check past 1800s, close to §B3's ~12,900
prediction (measured mean rate ~135 ms/step vs the 139 ms/step estimate).
Driver detected the terminal status and exited (by design — no retry on
a graceful budget abort); tmux `ncr_fb0_g0` ended; GPU 0 carries only its
production job again. The JSON is at
`/home/nvidia/ncr/results_ortho_fallback/stage0_damped_K24_s0.json` —
left in place, NOT deleted: the v2 relaunch must either use a fresh
outdir or clear this VOIDED artifact first, since the resume-safe
skip-if-COMPLETED logic does not skip `ABORTED-BUDGET` (it would re-run
in place) but the blind assessor must never mistake this voided JSON for
a §2 branch event. **Blind-discipline disclosure:** one `tail` of the
driver log during the wait was issued without the loss-masking `sed`
filter used everywhere else, briefly exposing loss values from this
VOIDED attempt (~4 log lines); the values were disregarded, are not
reproduced anywhere, and the attempt was already ruled VOID by §B3
before the exposure. All §2-relevant metric values remain unread.

**Status: v2 built, verified, deployed, NOT relaunched** — §B4's ruling
and the coordinator's follow-up both gate any relaunch on the delta
re-audit of this fix (which should also adjudicate F-A/F-B above).

---

## §B6 DELTA RE-AUDIT (2026-07-17)

Same independent auditor as §B4. Read-only except this section. All
numerics below re-run from scratch with independent seeds/frames (seed
1234/7/21/3 harness vs §B5's seed-0 `verify_d1_fix.py`), on CPU torch
2.8.0, through the REAL v2 `encode()` path (module loaded from the
md5-pinned working copy `matrix-thinking/ncr/ncr_ortho_fallback_stage0.py`
= `ce1448ab3d47536ebf3e82b146e33722`, byte-identical to the archived
`experiment-runs/2026-07-17_ncr_ortho_fallback_stage0/
ncr_ortho_fallback_stage0_v2.py` — both md5s verified this session;
loaded module asserted to carry the finfo.tiny guard and not the old
`NS_EPS` guard before any numbers were taken; `newton_schulz_polar`
module-attribute capture, not a scratch re-implementation).

### Ruling 1 — D1 FIX VERIFICATION: **CONFIRMED DEAD.**

v1→v2 diff regenerated independently (`diff -u` between the two archived
files): **exactly one semantic line changed** — `scale = S_floor /
S.clamp_min(NS_EPS)` → `scale = S_floor /
S.clamp_min(torch.finfo(S.dtype).tiny)` — plus a 9-line comment block.
Nothing else. Matches §B5's "11 lines, no other change" claim exactly;
also cross-checked against commit `15e3dc1`'s hunk for the working copy
(identical). §2's pseudocode line was corrected to match, with an inline
D1 marker (verified at the §2 code block).

§B4's numeric table re-run against v2's actual code (fp64, non-normal
`Z = U₀·diag(σ)·V₀ᵀ`, `U₀≠V₀`, `σ_max=5`, `eps_rel=1e-3`, target floor
`5.000e-3`, independent frames):

| d | raw σ_min (set) | achieved σ_min(Z_damped) | achieved/target |
|---|---|---|---|
| 5 | 1e-4 | 5.000000e-3 | 1.0000 EXACT |
| 5 | 1e-9 (v1: 100× short) | 4.999999e-3 | 1.0000 (rel err ~2e-7, see note) |
| 5 | literal 0.0 | 1.712e-2 | 3.42 (noise regime — F-A) |
| 25 | 1e-4 | 5.000000e-3 | 1.0000 EXACT |
| 25 | 1e-9 (v1: 100× short) | 5.000000e-3 | 1.0000 EXACT |
| 25 | literal 0.0 | 4.183e-3 | 0.84 (noise regime — F-A) |

The D1 axis (resolvable σ_min below the old `NS_EPS=1e-7` threshold) is
fixed: `σ_min=1e-9` rows, where v1 fell exactly 100× short, are now
exact. **Precision note (refines §B5's "rel tol 1e-10" prose, which its
own table already contradicted at d=5):** the achieved floor's relative
error is `≈ ε_dtype·σ_max/σ_i` — the dtype SVD's own relative accuracy
on the small singular value (fp64 at σ_i=1e-9, σ_max=5: bound ~5.5e-6,
measured 2e-7). This is the clean, unified statement: exactness degrades
continuously as σ_i approaches the noise floor, reaching O(1) exactly
there — which IS finding F-A, not a separate phenomenon. D1 itself
(a spurious 100×-level leak in a fully-resolvable regime) is dead.

### Ruling 2 — F-A ADJUDICATION: **ACCEPTED, re-scoped — via the MARGIN argument, NOT the unreachability argument.**

Independently reproduced F-A's O(1) spread: 8 fresh fp64 frames, d=25,
true σ_min=0 → achieved/target ratios {0.16, 0.32, 0.40, 0.43, 0.57,
0.71, 2.41, 4.11} — same O(1)-uncontrolled character as §B5's {1.46 …
0.00}. §B4's "removing the clamp reproduces the exact target… down to
and including literal σ_i=0" is hereby CORRECTED on the record: that
claim was seed/shape-lucky at d=5 (my own §B4 CASE-C run landed near
target by luck); the mechanism §B5 identifies is right — SVD backward
error `~ε·σ_max` amplified by `scale ~ S_floor/σ_noise` lands the
achieved value at floor magnitude times an uncontrolled O(1) factor,
and left-multiplication `M@Z` can never raise the rank of a truly
rank-deficient `Z`, so the worst case is 0, not O(1).

**The coordinator's proposed dynamical unreachability argument is
REJECTED.** The floor acts on the *output* `Z` each forward; it adds no
restoring pressure on the raw encoder's σ_min (gradients pass straight
through `M`, and §10's diagnosis is precisely that the read exerts zero
pressure on the spare direction). The raw σ_min's random walk therefore
proceeds exactly as in the parent run — which *measured* it crossing
~1e-7 absolute (~1e-8 relative against the drifted σ_max), i.e. at/below
fp32's resolvability floor (`ε_fp32·σ_max ≈ 1.2e-7·σ_max`). The
noise-floor regime is not just reachable in training, it is the
*expected destination* of a full Stage-0 run. F-A is live for the
experiment, not moot.

**It is nonetheless ACCEPTABLE for Stage-0's purpose, by margin:** the
guarantee Stage-0 actually needs is `cond(Z_damped) ≪ 1e7` (the §10.2
danger threshold), not exactness. In the reachable regime (dense fp32
encoder output, computed σ_min at noise, NOT bit-exact zero — §B5's B2b
empirical case, achieved 2.5× target), an O(1)-approximate floor with
measured ratios in [0.15, 4.1] bounds cond(Z_damped) at O(1e3–1e4) —
still ≥3 orders inside the danger threshold. The truly-unbounded case
(achieved 0) requires bit-exact rank deficiency, which for a dense fp32
encoder output is the F-B structured-input case — dispositioned under
Ruling 3, where the v3 guard converts it from Inf/NaN into a finite,
bounded no-op on that one direction for that one write (worst realized
cost in training: one skipped step via the existing finite-grad check;
in eval: none — see Ruling 3's eval-safety note).

**Required §2 re-scoping text (to be applied in the v3 cycle; replaces
the sentence "**`σ_min(Z_damped) ≥ eps_rel·σ_max(Z_raw)` holds exactly,
for every `Z_raw`, regardless of normality**" and its immediate
"even if the raw encoder's true `σ_min` random-walks to exactly 0"
claim later in the same section):**

> **Scoped guarantee (§B6/F-A).** `Z_damped`'s singular values are
> `scale_i·σ_i`. For every computed `σ_i` the dtype's SVD resolves
> (`σ_i ≳ ε_dtype·σ_max`; fp32 ~1.2e-7·σ_max) and above the guard's
> bite point (`eps_rel·σ_max·1e-6`), the floor
> `σ_floor,i = max(σ_i, eps_rel·σ_max)` holds to the SVD's own relative
> accuracy on `σ_i` (rel error ≈ `ε_dtype·σ_max/σ_i`), regardless of
> normality. At the dtype noise floor the achieved value is
> O(1)-approximate, not exact (measured 0.15×–4.1× target across
> independent frames), and for a truly rank-deficient `Z_raw` no floor
> is delivered on the dead direction at all (left-multiplication cannot
> raise rank) — the guard caps `scale ≤ 1e6` so that case is a finite,
> bounded under-floor, never Inf/NaN. In the reachable training regime
> this still bounds `cond(Z_damped)` at O(1e3–1e4), ≥3 orders inside
> §10.2's 1e7+ danger threshold — sufficient for Stage-0's purpose,
> but the claim "exact for every Z_raw, even σ_min exactly 0" is
> WITHDRAWN.

**Pre-registered ASSESS addendum (no code change):** before applying
§2's branch logic to any FAIL, ASSESS reads the z-dump's raw-Z σ_min
statistics (existing instrument). If raw σ_min sat at the fp32 noise
floor for a substantial fraction of writes, record that the floor
operated in its O(1)-approximate regime — the attribution logic is
unchanged (an O(1e4) cond still cannot re-arm the 1e7 trap), but the
record must show the check was made rather than assume exactness.

### Ruling 3 — F-B ADJUDICATION: **(a) v3 REQUIRED.** finfo.tiny alone is rejected.

F-B independently REPRODUCED through the real v2 `encode()` including
the real NS-40: fp32 input with two bit-exact-zero columns → computed
σ tail `[0.333, 0.0, 0.0]` (exact zeros) → `scale = S_floor/tiny ≈ 8e35`
→ `encode()` output **NaN** (isfinite=False). Confirmed exactly as §B5
reported.

**Why (b) accept-with-mitigations loses:** Stage-0's entire purpose is
to drive the encoder INTO the near-rank-deficient regime and hold it
there — exposure is not incidental, it is the experiment. A full attempt
runs ~42K steps × 256 matrices ≈ 10.7M SVDs with the trained encoder
spending the back half of the run at the fp32 noise floor (Ruling 2),
plus the maximum-hazard moment is POST-train eval/z_dump — thousands
more SVDs on exactly the most-degenerate trained state — where §B5
itself notes there is NO finite-grad guard. A single eval-path hit
poisons the terminal JSON with NaN metrics and voids another 1.75 GPU-h
attempt plus a cycle — strictly more expensive than the ~30-min v3
cycle. The mitigations are real but asymmetric: they protect training
steps, not the harvest.

**Exact v3 spec (mechanical; 2 semantic lines + comment edit):**

1. The guard line in `NCROrthoWriteModel.encode()` (v2's line
   `scale = S_floor / S.clamp_min(torch.finfo(S.dtype).tiny)`) becomes:

```python
scale = S_floor / S.clamp_min((self._eps_rel * sigma_max * 1e-6).clamp_min(torch.finfo(S.dtype).tiny))
```

   (COMPOSED guard, not the bare relative guard from §B4: the relative
   term `eps_rel·σ_max·1e-6` alone is 0 when `Z≡0` exactly, which would
   reintroduce 0/0; the outer `.clamp_min(tiny)` is the pure-0/0
   backstop for that measure-zero case — verified below: `Z≡0` →
   `Z_damped≡0`, finite. For every realistic `σ_max>~1e-29` the relative
   term dominates and the composed guard ≡ the relative guard.)

2. `RUNNER_TAG` → `"ncr_ortho_fallback_stage0_v3"` (so the terminal
   JSON self-identifies which script produced it — §B4 item-5
   discipline; the voided v1 attempt's JSON carries `_v1`).

3. The D1-fix comment block's last lines amended to describe the
   composed guard (F-B marker; non-semantic). §2's pseudocode guard
   line + the Ruling-2 re-scoping text applied in the design doc.

Nothing else changes. Expected v3-vs-v2 diff: exactly the two semantic
lines above + comment/doc edits — any other hunk voids the
pre-authorization below.

**The composed guard is PRE-VERIFIED here (so the v3 check is purely
mechanical conformance).** Standalone replication of v2's block with the
exact replacement line, my frames:

- fp64 resolvable regime: σ_min=1e-4 → ratio 1.000000000 (EXACT);
  exact down to the bite point (σ_min ∈ {1e-7, 1e-8, 5e-9} all 1.0000).
- fp32 resolvable regime: σ_min/σ_max=2e-5 → 1.0006; 2e-6 → 0.991 —
  exact to fp32's own SVD accuracy (`ε·σ_max/σ_i` bound), identical to
  v2 there.
- fp32 bit-exact-zero columns (the F-B reproducer): max scale = 1.000e6
  (capped exactly), `Z_damped` finite, **NS output finite** — F-B dead.
- `Z≡0`: `Z_damped≡0`, finite (the composed backstop working).
- fp32 noise regime (true σ_min=1e-7 rel 2e-8): finite, floor achieved
  at 8.5× target (benign overshoot direction, like §B5's B2b 2.5×).
- Gradient through the block with a zero-column item in the batch:
  finite (max |grad| ~3e5, no Inf/NaN).
- Disclosed loosened band (fp64 only, σ_i ∈ [tiny, 1e-9·σ_max)):
  measured profile σ_min=1e-9 → 0.20× target, 1e-10 → 0.02× — linear
  in σ_i, exactly the documented trade. Production fp32 never resolves
  σ into this band (noise floor 1.2e-7·σ_max sits 2+ orders above the
  5e-9·σ_max-equivalent bite point... at σ_max=5: bite 5e-9 absolute vs
  fp32 noise ~6e-7 absolute — 120× separation); the band matters only
  to fp64 verification harnesses, which must use the scoped expectation
  above, not exactness, inside it. Note `‖Z_damped‖₂ = σ_max(Z_raw)`
  exactly under the cap (singular values are `min`-composed), so NS —
  a polynomial iteration, division-free — receives a bounded input and
  its forward is structurally finite; eval paths (forward-only,
  `no_grad`) are therefore safe even in the worst rank-deficient case.

**Eval-path safety after v3 (closing §B5's "eval/z_dump have no guard"
worry):** eval is forward-only; NS forward is matmul-polynomial with no
division; with scale capped, `Z_damped` is bounded by `σ_max` exactly —
so no eval-path Inf/NaN is reachable through this block under v3. The
residual worst case (truly dead direction → under-floored, NS backward
explosion) exists only in training and only costs a skipped step via
the existing finite-grad check.

### Ruling 4 — VOIDED-ARTIFACT QUARANTINE: **mv REQUIRED before relaunch.**

Confirmed from the code: `run_primary_cell`'s resume logic skips only
`status=="COMPLETED"`; an `ABORTED-BUDGET` JSON is silently OVERWRITTEN
by `rn.atomic_write_json` at the re-run's terminal write. The voided v1
JSON must be moved aside BEFORE relaunch — and so must the driver log,
which §B5 discloses contains the 4 unmasked loss lines of the voided
attempt (quarantining it is also the blind-discipline fix: the assessor
must never open it). Exact commands (box, before the v3 launch):

```
mv /home/nvidia/ncr/results_ortho_fallback/stage0_damped_K24_s0.json \
   /home/nvidia/ncr/results_ortho_fallback/VOID_CONTENTION_attempt1_v1_stage0_damped_K24_s0.json
mv /home/nvidia/ncr/results_ortho_fallback/run_stage0.log \
   /home/nvidia/ncr/results_ortho_fallback/VOID_CONTENTION_attempt1_v1_run_stage0.log
```

Both `VOID_CONTENTION_*` files archive to the SSD mirror at harvest per
the standing policy; the blind assessor's instruction is to read ONLY
`stage0_damped_K24_s0.json` (which after the mv can only be the v3
attempt's) and may additionally verify its `runner_tag ==
"ncr_ortho_fallback_stage0_v3"` and `git_commit` against the v3 pin.
The §B5 blind-slip does not taint the v3 relaunch: the exposed values
belonged to an attempt already ruled VOID pre-exposure, and the mv
removes them from the assessor's path.

### VERDICT

**BLOCKED(v3 required — spec given above, mechanical).** D1 is
confirmed dead in v2 and F-A is adjudicated ACCEPTED-AS-RESCOPED, but
F-B is a reproduced non-finite forward on the prescribed guard's own
extreme, in the exact regime Stage-0 exists to probe, with the unguarded
eval harvest as the exposed surface — one more 1-line-semantic fix cycle
is cheaper than the void it risks.

**PRE-AUTHORIZED AUTO-CLEAR (no further audit round needed):** the
relaunch is CLEAR-FOR-RELAUNCH the moment ALL of the following hold,
verified by the build agent and recorded in a §B7 build note:

1. v3-vs-v2 `diff` shows exactly the Ruling-3 spec: the composed guard
   line (byte-identical to the line given above), the `RUNNER_TAG`
   bump, comment/§2-doc edits only. Any other hunk → back to audit.
2. Fresh md5 pin for v3 recorded (repo working copy + experiment-runs
   archive + box copy post-scp, all three matching).
3. 9/9 CPU self-test suite passes on v3 unmodified, plus a numeric
   floor re-check reproducing at minimum: fp64 σ_min=1e-9 EXACT, fp32
   zero-column finite-through-encode, `Z≡0` finite (the three
   regime-defining rows; expected values in Ruling 3's pre-verification).
4. The two Ruling-4 `mv` commands executed and verified (`ls` shows the
   `VOID_CONTENTION_*` names; the bare `stage0_damped_K24_s0.json`
   absent pre-launch).
5. Launch parameters per §B3(2): `--ceiling-gpuh 1.75`, external
   timeout 7200 s, all other cell parameters exactly §B1's (arm
   `damped_polar`, K=24, seed 0, steps 42000, `--eps-rel 1e-3`,
   ns-iter 40, ns-power 12, anneal-frac 0.5), blind discipline with the
   loss-masking filter on EVERY log read this time.

Conditions 1–4 failing in any particular → BLOCKED again, fresh md5,
delta back to this auditor.

---

## §B7 V3 BUILD + AUTO-CLEAR CHECKLIST + RELAUNCH RECORD (2026-07-17, build agent)

### v3 build

Applied exactly §B6 Ruling 3's spec to the v2 working copy. v2→v3 `diff -u`
(regenerated between the two archived files): **exactly 2 hunks** —
(1) `RUNNER_TAG` `"ncr_ortho_fallback_stage0_v1"` → `"..._v3"` (v2 had
never bumped the tag; §B6 Ruling 3 item 2 notes the voided attempt's JSON
carries `_v1`); (2) the guard line, **byte-identical to §B6's given line**:
`scale = S_floor / S.clamp_min((self._eps_rel * sigma_max * 1e-6).clamp_min(torch.finfo(S.dtype).tiny))`,
plus the comment-block amendment (F-B marker). Nothing else. §2's
pseudocode guard line updated to match (marked D1+F-B), the Ruling-2
scoped-guarantee blockquote installed in place of the withdrawn "exact
for every Z_raw" sentence, and the later "even if σ_min random-walks to
exactly 0" claim struck with a WITHDRAWN marker pointing at the scoped
guarantee.

| File | md5 |
|---|---|
| v3 working copy — `matrix-thinking/ncr/ncr_ortho_fallback_stage0.py` | `c7c2f7c36e0ab33c1efaeed04fbcf1bb` |
| v3 archive — `experiment-runs/2026-07-17_ncr_ortho_fallback_stage0/ncr_ortho_fallback_stage0_v3.py` | `c7c2f7c36e0ab33c1efaeed04fbcf1bb` |
| v3 on box — `/home/nvidia/ncr/ncr_ortho_fallback_stage0.py` | `c7c2f7c36e0ab33c1efaeed04fbcf1bb` (verified post-scp) |
| v3 driver — `orchestration/run_stage0_v3.sh` (archive = box) | `96cc2f8103b9c6d4165d3a5d1af4eea3` |
| Auto-clear verification script (archived) — `verify_v3_autoclear.py` | `2dc6b69cee03b10d800f77d6addf3d2c` |

v1 (`70dd7923…`) and v2 (`ce1448ab…`) archive copies untouched.

### Auto-clear checklist (§B6's five conditions, each verified)

1. **Diff conformance — HOLDS.** v2→v3 diff = the composed-guard line
   (byte-identical to §B6's spec), the `RUNNER_TAG` bump, comment/§2-doc
   edits. No other hunk.
2. **Triple md5 pin — HOLDS.** `c7c2f7c36e0ab33c1efaeed04fbcf1bb` in all
   three places (table above).
3. **Self-tests + numeric rows — HOLDS, with one recorded tiebreak.**
   9/9 CPU self-test suite PASS on v3 unmodified. Numeric rows (real
   CUDA, through the real v3 `encode()`, module identity asserted first):
   - fp64 **resolvable-exact**: σ_min ∈ {1e-7, 1e-8, 5e-9} →
     achieved/target 0.999999995 / 1.000000034 / 0.999999964 (EXACT to
     the SVD's own accuracy, matching Ruling 3's pre-verification rows).
   - fp32 zero-column F-B reproducer: scale capped at exactly 1.0000e6,
     `Z_damped` finite (max 3.79 ≈ σ_max, per Ruling 3's `‖Z_damped‖₂ =
     σ_max` note), `encode()` output FINITE end-to-end — **F-B dead**.
   - `Z≡0`: `Z_damped≡0` exactly, output finite — composed backstop works.
   - **TIEBREAK RECORDED (for the auditor):** condition 3's label reads
     "fp64 σ_min=1e-9 EXACT", but Ruling 3's own pre-verification —
     which the condition's parenthetical names as the source of expected
     values — puts σ_min=1e-9 (at σ_max=5) BELOW the composed guard's
     bite point (5e-9 absolute) in the *disclosed loosened band*, with
     pre-verified expected value **0.20× target**; no spec-conformant v3
     could read EXACT there. The coordinator's relaunch dispatch names
     this row "resolvable-exact". Both sources disambiguate identically;
     the row was tested as pre-verified: measured σ_min=1e-9 →
     **0.20000× target**, reproducing §B6's own number to 4 decimals.
     Flagged here rather than silently normalized; if the auditor meant
     something else by that label, the launch decision should be
     revisited against this record.
4. **Quarantine — HOLDS.** Both §B6 Ruling-4 `mv`s executed on the box;
   `ls` verified: `VOID_CONTENTION_attempt1_v1_stage0_damped_K24_s0.json`
   and `VOID_CONTENTION_attempt1_v1_run_stage0.log` present, bare
   `stage0_damped_K24_s0.json` and `run_stage0.log` both ABSENT
   pre-launch (`test ! -f` both confirmed).
5. **Launch params — HOLDS.** `--ceiling-gpuh 1.75` (internal, graceful,
   6300 s), external `timeout 7200`, cell params exactly §B1's
   (`damped_polar`, K=24, seed 0, steps 42000, `--eps-rel 1e-3`,
   ns-iter 40, ns-power 12, anneal-frac 0.5), same outdir. Blind
   discipline: every log read since the v3 cycle began has gone through
   the loss-masking `sed` filter — no exceptions this time.

**ALL FIVE HOLD → relaunch executed under the pre-authorization.**

### Relaunch record

- **Placement:** all 8 GPUs still carry one 392M production job each at
  100% util (checked immediately pre-launch) — no free GPU exists, so
  the contended-rate assumption stands (no solo-rate note applies).
  GPU 0 (uniform load across all 8; same GPU as the smoke + attempt 1).
- **tmux:** `ncr_fb0_g0`, driver `run_stage0_v3.sh 0`, launched
  **2026-07-17T08:02:48Z**. Confirmed alive post-launch; fresh
  `run_stage0.log` growing.
- **Measured rate (masked reads, elapsed-seconds field only):** step
  1000 at 143 s → ~143 ms/step, contention factor 3.35 vs the 3.3
  §B3(2) priced. **Expected COMPLETED at ≈6,000 s ≈ 09:43Z**, ~5%
  inside the 6,300 s internal ceiling — tighter than the pricing's
  nominal margin; if contention degrades >~5% further the internal
  ceiling fires first (gracefully, `ABORTED-BUDGET`), which under the
  re-priced ceiling would be a genuine budget event to bring back to
  the coordinator, not auto-retried (the driver's terminal-status gate
  stops on it).
- **Blind:** no metric value read at or since launch; the assessor
  reads only `stage0_damped_K24_s0.json` (post-quarantine, can only be
  the v3 attempt's) and may verify `runner_tag ==
  "ncr_ortho_fallback_stage0_v3"` + `git_commit` per §B6 Ruling 4.
