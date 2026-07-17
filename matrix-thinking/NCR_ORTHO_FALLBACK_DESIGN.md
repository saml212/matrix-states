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
    scale = S_floor / S.clamp_min(NS_EPS)                      # file (below)
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
**`σ_min(Z_damped) ≥ eps_rel·σ_max(Z_raw)` holds exactly, for every
`Z_raw`, regardless of normality** — no Weyl approximation, no
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
of how far σ_min has drifted** — even if the raw encoder's true `σ_min`
random-walks to exactly 0, `Z_damped`'s floored `σ_min` is still
`eps_rel·σ_max` exactly. (A smooth alternative, damping via
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
