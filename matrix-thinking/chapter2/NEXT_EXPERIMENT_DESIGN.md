# Task E Design Spec — Compositional Multi-Hop Relational Recall (Reasoning-Transfer)

> **STATUS: CLOSED — Task E ran and its gate PASSED / CONFIRMED (2026-07-04
> consolidation header). Results: `TASK_E_FINDINGS.md`; STATE.md "Chapter 2
> — STATUS".** This is the design/pre-registration this project's naming
> convention would otherwise call `TASK_E_DESIGN.md` — kept under its
> original filename because ~10 living documents (STATE.md, EXPERIMENT_LOG.md,
> TASK_E_FINDINGS.md, TASK_D_WRITEUP.md, STAGE0_DESIGN.md, both DeltaNet
> design docs) cite it by this exact name; renaming would break every
> inbound reference for no benefit.

**Drafted 2026-07-01, before any build.** Status: ready for a build+audit pass, not
yet locked/pre-registered (locking happens once the audit gauntlet has attacked
this document the way `archive/chapter2-gauntlet/gauntlet/ATTACK_*.md` attacked Task D, and once the Stage-0
trainability precursor below has actually run). Supersedes nothing — it is the
`TASK_D_PREREGISTRATION.md` §8 sequencing item ("reasoning-transfer: does the
mechanism survive on an actual reasoning task") made concrete, and it is the
exact future-work gap the ICML MI-workshop paper names explicitly: *"A fully
disambiguating test requires a reasoning task whose ground-truth solution
provably requires k>1 independent quantities at the answer position; we do not
have such a task at this scale and flag it as future work"*
(`submissions/icml-mi-workshop-2026/sections/05_positive_control.tex`).

---

## 0. Why this, not the other two candidates

Three candidates were weighed (research + attack pass on each, see §11 for the
attack that was actually run against this design):

1. **Fix d≥32 trainability** (Task D's fixed h=64 encoder fails to train at
   d≥32; calibration also showed a naively-bigger encoder diverging). Compute-
   hungry, but it is an **optimization/engineering question**, not a test of
   matrix-thinking's core claim — the pre-registration itself scopes d≥32 as
   "a generality probe, not the primary test" (`TASK_D_PREREGISTRATION.md` §6),
   and d=16 is the already-passed decisive gate. Made the primary experiment,
   it risks burning 8×H100 budget on an open-ended hyperparameter hunt with no
   falsifiable hypothesis and no guaranteed payoff. **Verdict: fold in as a
   bounded, gated precursor sub-study (Stage 0 below), not the flagship.**
2. **Reasoning-transfer** (this doc). This is the literal next item in
   `TASK_D_PREREGISTRATION.md` §8's own sequencing and the paper's stated
   future work. It is also compute-hungry in the way the user asked for —
   see §9 — because it needs a hop-depth curriculum, larger entity pools, and
   (per Stage 0) genuinely bigger matrices, not more tiny packed runs.
3. **C2 param-matched vector-state control.** Highest rigor value, lowest
   compute value — it is a cheap, mostly eval-time control. `CLAUDE.md`'s hard
   rules say "the param-matched flat-vector ablation blocks ALL downstream
   decisions. Run it first," which is real tension with not making it the
   primary experiment here. Resolution: `TASK_D_PREREGISTRATION.md` already
   made a considered, documented decision to descope C2 to a parallel
   Stage-1b track rather than block the Stage-1 gate, and `TASK_D_FINDINGS_DRAFT.md`
   §6 confirms H_primary was evaluated without it. This design continues that
   precedent explicitly rather than silently: **C2, and a new cheap analog for
   this task (C_MLP, §6), run in parallel on idle capacity and are reported
   alongside the headline result — required before any publication claim, but
   not gating Stage 2's launch.** This is flagged, not swept under the rug.

**Bottom line: reasoning-transfer is the right flagship** because it is (a)
the pre-registered, already-promised next step, (b) the actual thesis payoff
("does rank help reasoning," not just memory), and (c) can be designed to
naturally absorb genuine compute (§9) rather than manufacturing an excuse to
use 8×H100.

---

## 1. The sharpened question

Task D (`TASK_D_FINDINGS_DRAFT.md`) confirmed: a from-scratch matrix-native
model, trained under a hard single-matrix bottleneck on K one-hop key→value
bindings, develops effective rank ≈ K and depends on that rank causally. That
is associative memory — one lookup, no composition. The open question, stated
by the project twice now (ICML paper §5's future-work line, and
`TASK_D_FINDINGS_DRAFT.md` §6's closing sentence), is whether this extends to
**reasoning**: does the SAME rank-K matrix Z compose correctly under repeated
self-application, i.e., does gradient descent discover an operator with real
algebraic structure, or does it merely memorize a lookup table that happens to
satisfy the one-hop rank bound?

> **H_E (primary).** A matrix-native model trained under Task D's bottleneck
> on one-hop *and* short multi-hop (h ≤ H_train) relational queries, where the
> multi-hop readout is pinned to literal iterated matrix action
> (`pred = Zʰ · key`, no learned per-hop weights), generalizes zero-shot to
> hop depths h > H_train at accuracy approaching the **idealized linear-
> associative-memory ceiling** (§3) — not chance, and not merely the
> unconstrained-MLP shortcut baseline's floor (§6, C_MLP).

- **CONFIRM** ⇒ SGD's rank-K matrix acquires genuine composable operator
  semantics beyond what one-hop training requires; matrix-thinking's
  compositional-reasoning premise survives its first reasoning test.
- **FALSIFY** (held-out-hop accuracy ≈ chance ≈ C_MLP floor at h=H_train+1,
  despite high in-distribution and one-hop accuracy) ⇒ rank-sufficiency for
  memory does **not** entail eigenstructure-sufficiency for reasoning; a
  clean, decisive, still-publishable negative that completes the picture
  the ProsQA-MULTI position-decomposition finding started (rank can be
  causally necessary for recall and still be functionally inert for
  reasoning).

Both outcomes are decisive and publishable, matching Task D's bar.

**A secondary, honest reframe (from the attack pass, §11): this is not
straightforwardly a test of "does composition emerge from nothing."** For an
injective functional graph with orthonormal keys, the *classical* minimum-norm
solution `Z_ideal = Σᵢ e_π(i) eᵢᵀ` composes exactly under repeated
application — this is textbook linear-associative-memory chaining (Kohonen
1972/73; Anderson 1972), not novel, and not claimed as novel here (same
discipline Task D applied to the rank≥K fact itself). The genuinely open,
novel question is narrower and sharper: **does SGD's trained Z, constrained
only by a rank-K bottleneck and a modest multi-hop training signal, converge
close enough to `Z_ideal` in eigenstructure (not just Frobenius norm or rank
magnitude) to compose correctly out to unseen depths** — see §3, §7.

---

## 2. Method — Task E generator

Entities are `d`-dimensional vectors drawn fresh per episode from an
orthonormal pool of size `N ≤ d` (QR-orthogonalized, reusing
`task_d.py::_random_directions(orthogonal=True)` verbatim). `K ≤ N` of them
have a defined outgoing edge, given by an **injective** partial function `π`
on the K-subset (in-degree ≤ 1 **and** out-degree ≤ 1 — a disjoint union of
simple paths and/or cycles). Two variants, both required:

- **Primary variant — cyclic permutation.** `π` is a **single Hamiltonian
  K-cycle** over the K-subset (not a general random permutation — see
  `archive/chapter2-gauntlet/gauntlet/AUDIT_task_e_validity.md` Finding B, folded in below). Every
  entity in the subset has a well-defined successor for *any* h (chains never
  dead-end), so held-out-depth queries at arbitrarily large h are always
  valid by construction — this closes the MNNS/B-4 "frontier collapses at
  unspecified depth" trap (`KILL_LIST.md` item 1; `archive/chapter2-gauntlet/gauntlet/ATTACK_task_shortcuts.md`
  §B-4) outright: there is no ambiguity about whether a query is well-formed.
  **Audit correction (2026-07-01):** the first build sampled `π` as a
  *general* uniform random permutation, which typically decomposes into
  several independent, generally-short disjoint cycles — since `π^h` is
  periodic with the period of whichever cycle an entity belongs to, this
  silently collapsed most nominal "held-out-hop" queries into in-distribution
  or trivial-identity ones (measured: 100% collapse at every held-out hop for
  K=4; the `H_extra=8` probe was 100% collapsed at the K=8 primary operating
  point, since `H_extra` literally equaled `K`). **Fix:** `π` is forced to a
  single K-cycle (`task_e.py::_permutation_graph`), so `π^h` is periodic
  *only* with period K, uniformly across every entity — and
  `TaskEConfig.__post_init__` now asserts, at config-construction time, that
  every `H_test`/`H_extra` hop has `h % K != 0` and `h % K` not equal to any
  training hop's residue mod K, so a periodicity-confounded config can never
  be constructed silently again.
- **Secondary/robustness variant — chain-into-sink.** `π` is a genuinely
  partial injection into a larger pool (`K < N`); unbound entities are
  absorbing terminals. Chains are finite; a query `(a, h)` is valid only if
  `a`'s chain has length ≥ h from `a` (checked and asserted at generation
  time, not assumed — the exact fix the MNNS post-mortem demands). Run as a
  generality check on the primary variant's result, not the headline.

**Grammar (reuses Task D's BIND/QUERY tokens exactly):**
`BIND k₁ v₁ ... BIND k_K v_K` where `(kᵢ, vᵢ) = (eᵢ, e_π(i))`, followed by
multi-hop queries `QUERY a h` for `(a, h)` pairs sampled per batch,
`h ~ Uniform{1..H_train}` during training.

**Readout, architecturally pinned (load-bearing, mirrors Task D §2.1):**
```
v_0 = key_a
v_t = Z @ v_{t-1}      for t = 1..h        # literal iterated matmul
pred(a, h) = v_h
```
No MLP, no h-conditioned parameters, no learned per-hop head. `Z` itself is
produced by Task D's unmodified `BindingEncoder` from the K one-hop
`(key, value)` tokens only — the encoder never sees `h` or the query.
Scored by the same absolute cosine criterion as Task D
(`cos(pred(a,h), e_π^h(a)) > τ`), never a K-way argmax.

**Why rank(Z) ≥ K still holds (unchanged from Task D, premises re-verified):**
the one-hop proof (`TASK_D_PREREGISTRATION.md` §3) requires linearly
independent keys **and** linearly independent values. Keys are the K
entities themselves (orthonormal, hence independent by construction). Values
are `{e_π(i)}` — independent **only because `π` is injective** (in-degree
≤ 1). This is the exact attack the red-team pass landed (§11): if merges were
allowed (two sources mapping to the same target), `rank(V_mat) < K` and the
bound silently weakens — the same "K nominal edges ≠ K rank constraints"
miscounting trap that killed MNNS. **Fix, mandatory and load-bearing:**
injectivity of `π` is enforced by the generator and asserted at
generation-time via a `task_d.py::_self_test`-style check (stacked values
have rank K, not just K entries) — this is a required smoke-gate addition,
not an optional nicety.

---

## 3. The idealized-Z ceiling (a designed-in, classical, zero-training reference)

Because keys are exactly orthonormal and `π` is injective, the hand-built
`Z_ideal = Σᵢ₌₁ᴷ e_π(i) eᵢᵀ` (never trained, computed analytically per
episode) satisfies `Z_ideal @ eᵢ = e_π(i)` exactly, and therefore
`Z_ideal^h @ e_a = e_π^h(a)` **exactly, cosine ≡ 1.0, for every valid h** —
this is classical chaining (Kohonen/Anderson), not a novel claim. It gives an
unusually clean interpretive ceiling: because the ideal reference has *zero*
decay with h, **any decay observed in the SGD-trained model's compositional
accuracy vs. h is entirely attributable to how far the trained Z deviates
from the classical construction — not to generic floating-point error
accumulation or task-inherent ambiguity.** Report `Z_ideal`'s curve alongside
every M3_E number (C7, §6) as the ceiling, not just "unconstrained-model
accuracy," which is what Task D used.

---

## 4. Architecture

Identical to Task D's `MatrixMemoryModel` / `BindingEncoder`
(`model_v4.py`) for the encoder and one-hop path — reused verbatim, not
reimplemented, to keep this a clean extension rather than a confound. New:
a `compose(Z, key, h)` function implementing the pinned iterated-matmul
readout (§2), and a `force_rank_k` variant that projects Z once
(`rank_utils.truncate_to_rank`) and then iterates the *same* rank-k-projected
matrix h times (tests whether a rank sufficient for one-hop recall stays
sufficient under repeated self-application, or whether composition needs
strictly more rank — an open, pre-registered secondary measurement, §6 M4_E).

---

## 5. New mandatory controls (in addition to Task D's C1–C5, reused verbatim)

| # | Control | Closes |
|---|---------|--------|
| C6 (mandatory, load-bearing) | **Injective functional graph** (in-degree ≤ 1 AND out-degree ≤ 1), asserted at generation time via a value-rank check. | The merge/miscounting attack in §2 and §11 — without this, "K edges" does not imply "rank(Z) ≥ K." |
| C7 | **Idealized-Z reference ceiling** (§3), computed analytically, reported alongside every compositional-accuracy number. | Distinguishes "SGD-specific compositional failure" from "generic small-model numerical error" — was missing from the naive design and was the attack pass's second-most-important finding. |
| C8 | **Eigenstructure fidelity metric** — in addition to Task D's effective/stable rank (magnitude-only, sign-blind), report the distance (in the complex plane, optimally matched) between the trained Z's eigenvalues and `Z_ideal`'s eigenvalues (roots of unity for pure cycles). Motivated directly by Grazzi et al. (arXiv:2411.12537, ICLR 2025) and DeltaProduct (arXiv:2502.10297): composability/state-tracking is gated by eigenvalue *sign and phase*, not rank magnitude alone — a rank-K matrix with the wrong eigenvalue signs can satisfy one-hop recall while failing composition. | Turns the attack pass's reframe ("rank ≠ eigenstructure") into a measured, falsifiable quantity instead of an unmeasured caveat. |
| C_MLP | **Unconstrained MLP-flatten shortcut baseline** — `pred(a,h) = MLP([flatten(Z), one-hot(h)])`, trained on the same in-distribution hops, evaluated zero-shot at held-out h (where `one-hot(h)` is out-of-vocabulary). Rank-blind by `KILL_LIST.md` Lesson 1 by construction. | Establishes an honest floor: if *any* architecture — not just the pinned-composition one — could shortcut hop-extrapolation, H_E's positive result would be uninterpretable. Expected to fail at held-out h; if it doesn't, the task is broken. |
| C_composition-purity | **Extended blank-out test** (mandatory pre-training smoke gate item, extends Task D §2.3's blank-out test): verify `pred(a,h)` is a pure deterministic function of `(Z, key_a, h)` computed via exactly h sequential matmuls, with no learned per-h parameter and no other input pathway — corrupt anything but `Z` and `h` post-encoding and confirm bit-for-bit invariance across the full decode sequence. | Closes the "model secretly conditions on h via a side channel" attack (§11) — the exact failure mode Task D's original blank-out test was built to catch, extended to the iteration pathway. |
| C2 (parallel, not gating) | Param-matched **vector-state control**, run cheaply alongside Stage 2 on idle capacity. A vector state has no natural "apply the same operator again" primitive without a bespoke learned hop-MLP — this breaks architectural parity with the matrix arm's literal-iteration readout, which is itself informative and should be reported, not skipped. | The `CLAUDE.md` hard-rule tension (§0) — addressed explicitly, not silently. |

---

## 6. Pre-registered measurements & decision criteria

Fixed operating point for the causal test: `d=16`, K=8 (permutation variant),
`H_train = {1,2,3}`, `H_test = {4,5,6}` (and one further-out probe at
`h ∈ {7,21}` — **not** `{8,10}`, corrected 2026-07-01 — to check whether
degradation is graceful or a hard cliff), orthonormal keys (gate default, per
Task D's own audit finding that Gaussian near-orthogonal vectors smear the
knee). `{7,21}` is chosen so `h mod K` avoids `{0} ∪ (H_train mod K)` for
every K in the (K=4-dropped) Stage-2 sweep grid `{8,12,16}` — the original
`{8,10}` was self-defeating at K=8 specifically (`8 mod 8 = 0` ⇒ identity;
`10 mod 8 = 2` ⇒ coincides with the training hop `h=2`), per
`archive/chapter2-gauntlet/gauntlet/AUDIT_task_e_validity.md` Finding B.

> **ADDENDUM (2026-07-01, post-launch — a second, narrower residue collision
> the periodicity fix above did not check).** The stated criterion (`h mod K`
> avoids `{0} ∪ H_train_residues`) was validated against `H_train`'s residues
> only — it was never checked against `H_test`'s residues, and it should have
> been. `21 mod 8 = 5` and `21 mod 16 = 5`, and `5 ∈ H_test = {4,5,6}` at
> **both** K=8 and K=16 — so at those two K values the `h=21` probe is
> degenerate with the already-tested `h=5` query in *effective-hop* terms.
> Only `K=12` (`21 mod 12 = 9`, not in `H_test ∪ H_train ∪ {0}`) actually
> probes novel ground. This is an audit escape: the config-time periodicity
> guard (§2, §11) checks `h mod K` against `{0} ∪ H_train_residues` but not
> `H_test_residues`, so a collision with `H_test` — as opposed to `H_train` or
> the identity — passes silently.
>
> **Reinterpretation, not a retraction.** The first Task E sweep (20K-step
> round, `EXPERIMENT_LOG.md` 2026-07-01) turned this into an accidentally
> informative result. At the one seed that converged at K=8, `h=5` reached
> `recovered_frac@0.9 = 0.915`, while `h=21` — the *same effective hop* via
> the collision above — reached only `0.006`. Same effective hop, wildly
> different outcome. This shows **raw iteration depth** (the literal count of
> sequential matmul self-applications of `Z`, i.e. compounding
> spectral/numerical error across more matrix powers) drives decay in a way
> `effective_hop = h mod K` alone does not capture. `h=21` should therefore be
> read as a **depth-decay probe**, not a held-out-hop generalization probe —
> its effective hop is already in `H_test`, so it cannot validly stand in for
> "novel ground" at K=8 or K=16. Its M3_E entries should be reported/labeled
> as such rather than folded into the held-out-hop generalization claim.
>
> **Standing requirement for future variants.** Any `H_extra`/`H_test` choice
> must state per-K validity explicitly (which residues it probes, and against
> which set — `H_train`, `H_test`, or genuinely novel) rather than relying on
> a single global `{0} ∪ H_train_residues` guard. `TaskEConfig.__post_init__`
> should be extended to also flag (not necessarily block) an `H_extra` hop
> whose residue collides with `H_test`, so this specific escape cannot recur
> silently.

**M1_E — learned effective rank vs K, under multi-hop training** (sanity
re-confirmation of Task D's M1 under the new loss mixture; not the interesting
part, but not guaranteed to survive the objective change unexamined).
CONFIRM: same criteria as Task D M1 (ρ ≥ 0.8, band). If this fails where
Task D's M1 passed, that alone is informative (multi-hop supervision
disrupts rank recruitment) and blocks interpretation of M3_E/M4_E below.

**M2_E — in-distribution multi-hop accuracy** (h ≤ H_train, unconstrained
rank). Basic capacity sanity check: should be high. Not a decision gate by
itself.

**M3_E — compositional generalization (PRIMARY, tests H_E).** Accuracy at
held-out hops h ∈ H_test, unconstrained rank, reported against three
references: chance (~0 at high τ), C_MLP's floor, and C7's idealized-Z
ceiling (§3).
- CONFIRM: mean cosine (or `recovered_frac@0.9`) at held-out hops is (i)
  significantly above both chance and the C_MLP floor (non-overlapping CIs
  across ≥5 seeds) AND (ii) degrades gracefully with hop distance from
  H_train (not a hard cliff at H_train+1) AND (iii) tracked against C7 so the
  gap-to-ideal is reported as a number, not asserted qualitatively.
- **Stratification requirement (added 2026-07-01, `archive/chapter2-gauntlet/gauntlet/AUDIT_task_e_validity.md`
  Finding B, fix item 3):** report every M3_E number alongside `effective_hop`
  — `h mod K` for the permutation variant (now a single K-cycle, so this is a
  single deterministic value per nominal `h`, not a per-query distribution to
  split), and raw `h` for the acyclic chain variant. PASS/FAIL must be read
  off `effective_hop`, not raw nominal `h`, so a decision is never
  attributable to periodicity rather than genuine composition depth.
  `run_task_e.py::evaluate_at_hops` emits this field automatically per hop
  entry.
- FALSIFY: held-out accuracy is statistically indistinguishable from both
  chance and C_MLP already at h=H_train+1, despite M2_E being high. Decisive
  negative (§1).

**M4_E — causal rank–composition link (extends Task D's M3).** Force-rank-k
training (straddle grid around K), measured at both M2_E (in-distribution)
and M3_E (held-out) hops.
- CONFIRM: in-distribution reproduces Task D's M3 step at k≈K; held-out
  compositional accuracy is *at least as rank-dependent* (fails at k<K).
- Secondary, exploratory (not a strict pass/fail gate — pre-registered as
  open, not assumed): does held-out composition require **strictly more**
  rank than one-hop recall (`k*_compose > k*_memory`)? If so, report it as a
  novel, nuanced finding (rank requirement scales with reasoning depth); if
  not, report that plainly too. No claim is made a priori about which way
  this goes.

**Overall gate:** PASS if M1_E CONFIRM and M3_E CONFIRM and M4_E CONFIRM.
M3_E FALSIFY alone is a decisive, publishable negative regardless of the
others (matches Task D's decisiveness bar, §1).

---

## 7. What this experiment does and does NOT claim (scope discipline)

- **Shows, if CONFIRM:** gradient descent, under Task D's rank-forcing
  bottleneck plus a modest multi-hop training signal, converges close enough
  to the classical minimum-norm operator (in eigenstructure, not just rank)
  to compose correctly at unseen depths — i.e., rank-sufficiency for memory
  measurably extends toward eigenstructure-sufficiency for reasoning.
- **Does NOT show, even under CONFIRM:** that transformers in general
  compositionally generalize (Wang et al. 2405.15071's "grokked but doesn't
  generalize to novel entity combinations" and Dziri et al. 2305.18654's
  "Faith and Fate" results are the standing counter-evidence in the
  literature for compositionality claims broadly — this experiment's claim
  is narrower and mechanistic: specifically about the rank-K matrix state
  Task D already showed is causally used, not about transformers or
  compositionality in general).
- **Does NOT show:** that this transfers to real-language reasoning data
  (OpenR1-Math, ProsQA). That is a later, separate step, deliberately
  sequenced after this synthetic gate exactly as Task D was sequenced after
  the workshop paper — see §10.

---

## 8. Compute plan (8×H100)

Task D used only ~40–60 GPU-h of a ~1.6k GPU-h budget (`DEPLOY.md`), packing
4 tiny (~1GB, ~65%-of-one-H100) runs per GPU — genuinely underused. This plan
is designed to be **compute-hungry in kind, not just in GPU-hours**: Stage 2
at d≥32 runs **one job per GPU at full batch/model size** (not packed),
so each run is compute-bound, not orchestration-bound.

**Stage 0 — d≥32 trainability precursor (bounded, gated, ~40–60 GPU-h).**
On the *original* Task D one-hop harness (cheap reuse, no Task E changes
needed yet): a focused ablation of standard fixes for "write into a large
d×d state from a small representation" — LR warmup, attention-pooling
readout in place of the fixed learned row-reader queries, orthogonal/scaled
init on `row_out`, a deeper pre-LN residual encoder, `n_refine` 1→3,
curriculum pretraining at d=16 then growing to d=32/64. ~20–30 configs ×
{d=32, d=64} × 3 seeds. **Decision gate:** at least one config must reach
`recovered_frac@0.9` ceiling > 0.7 at d=32 (unconstrained) before Stage 2
runs at that d. **If none do:** d≥32 stays out of scope for Task E; Stage 2
runs at d=16 only, scaled instead via larger K/entity-pool/hop-depth/seed
count — still genuinely compute-hungry, just via task complexity rather than
matrix size. This bounded fallback is what prevents Stage 0 from becoming an
open-ended, low-value HP search (the risk that disqualified Candidate 1 as
the flagship, §0).

**Stage 1 — Task E harness build + smoke (<1 GPU-h).** Multi-hop generator
with the injective-graph constraint (C6), pinned iterated-Z readout,
extended blank-out/composition-purity test (C_composition-purity), C_MLP
baseline, C7 idealized-Z reference, C8 eigenvalue-fidelity metric,
force-rank-k extension. Must pass smoke before any training, per
`CLAUDE.md`'s smoke-test rule.

**Stage 2 — main compositional sweep (~80–120 GPU-h at d=16; +~50–75 GPU-h
conditional on Stage 0 unlocking d=32/64).**
- d=16 (guaranteed-trainable): K ∈ {8,12,16} — **K=4 dropped 2026-07-01**
  (`archive/chapter2-gauntlet/gauntlet/AUDIT_task_e_validity.md` Finding B: even under the single-K-cycle
  fix, K=4 leaves only residues `{0,1,2,3}` mod 4 available, i.e. every
  integer hop collides with either the identity or a training residue — no
  H_test/H_extra choice exists that isn't fully confounded at K=4; the code
  now refuses to construct such a config at all, see `TaskEConfig.__post_init__`) —
  H_train ∈ {1,2,3} × H_test-generalization eval, force-rank-k straddle grid
  at the primary operating point (K=8, H_train=2), ≥5 seeds throughout, both
  permutation and chain-into-sink variants. Longer runs than Task D
  (multi-hop curriculum needs more steps to converge — plan 20–30K steps vs.
  Task D's 8–10K).
- d=32/64 (conditional on Stage 0 PASS): reduced grid (K ∈ {8,16,32},
  H_train ∈ {1,2}) at the Stage-0-selected architecture, 1 job/GPU, larger
  batch (1024–4096 vs. Task D's 256), longer training (30–50K steps) — this
  is the genuinely-saturating-8×H100 tranche.
- C2 (vector-state) and C_MLP controls run in parallel on idle capacity
  throughout, <15 GPU-h combined.

**Total: ~170–260 GPU-h base plan, up to ~300–350 GPU-h if Stage 0 unlocks
d≥32** — roughly 3–6× Task D's usage, ~10–20% of the ~1.6k GPU-h budget,
leaving headroom for reruns and seed refills, and (critically) using
genuinely larger models where it unlocks (Stage 2 d≥32), not just more tiny
packed jobs.

---

## 9. Sequencing — what this unlocks

1. **Task E (this doc)** — go/no-go on "does the rank-K matrix compose,"
   the reasoning-transfer step `TASK_D_PREREGISTRATION.md` §8 named.
2. **If PASS →** real-data reasoning transfer (matrix-CODI bolt-on readout
   on ProsQA/OpenR1-Math, or a from-scratch matrix-native LM on OpenR1-Math
   using the H100's already-tokenized 43.7M-token corpus). This is
   deliberately sequenced *after* Task E, not instead of it — jumping
   straight to real text would reintroduce every confound Task D was built
   to eliminate (position-decomposition, per the ProsQA-MULTI gauntlet,
   `.team-runs/20260430-235745-position-decomp-gauntlet/thread.md`; argmax
   decoding; unconstrained MLP readouts).
3. **If FAIL (M3_E FALSIFY) →** publish as a clean companion negative to the
   ICML workshop paper and Task D: "rank is causally necessary for
   associative recall but does not, by itself, confer compositional
   reasoning" — a real, citable, falsifiable finding, not a dead end to hide.

---

## 10. Related work (new citations beyond `research/task-d-novelty-july2026.md`)

From the research+attack pass (§11); confidence noted, verify exact arXiv IDs
before external citation per this project's house convention:

1. **Sinha et al., CLUTRR** (EMNLP 2019) — multi-hop kinship relational
   reasoning with depth generalization (train ≤4 hops, test to 10). Same
   generalization *spirit* as M3_E, but composition is via a trained
   LSTM/GNN over symbolic relation labels, not a literal matrix power, and
   there is no rank-necessity theorem or bottleneck. Closest benchmark-family
   precedent; verify exact arXiv ID before citing.
2. **Wang et al., "Grokked Transformers are Implicit Reasoners"**,
   arXiv:2405.15071 (NeurIPS 2024) — the key cautionary prior: transformers
   *do* grok in-distribution 2-hop composition, but the resulting circuit
   fails to systematically generalize to novel entity combinations. Different
   generalization axis than M3_E (novel entities at fixed depth vs. deeper
   hops at fixed entities), but establishes "looks compositional, is
   pattern-matching" as the default failure mode to watch for.
3. **Dziri et al., "Faith and Fate"**, arXiv:2305.18654 — transformer
   compositionality often reduces to linearized subgraph matching over seen
   computation graphs. Standing tension to address explicitly in any writeup
   (§7).
4. **"Loop, Think, & Generalize"**, arXiv:2604.07822 (2026) — positive
   precedent for the underlying mechanism (iterating a fixed operator beyond
   training depth generalizes), at the level of looped transformer layers
   rather than a literal d×d matrix power. Very recent; single-pass
   verification only, re-check before citing.
5. **Grazzi et al., "Unlocking State-Tracking in Linear RNNs via Negative
   Eigenvalues"**, arXiv:2411.12537 (ICLR 2025), and **DeltaProduct**,
   arXiv:2502.10297 — composability/state-tracking is gated by eigenvalue
   sign/phase structure, not rank magnitude alone. Directly motivates C8
   (§6) and the honest reframe in §1/§3: this experiment's real contribution
   is measuring eigenstructure fidelity, not just rank.
6. **Merrill, Petty & Sabharwal, "The Illusion of State in State-Space
   Models"**, arXiv:2404.08819 (ICML 2024) — TC⁰ limits on streaming linear
   recurrence over unbounded sequences. A different regime (Task E uses a
   bounded encoder and small fixed h ∈ {1..10}), so it does not scoop this
   design, but explains why eigenstructure (not just rank) is the right
   thing to check.
7. **Kohonen (1972/73); Anderson (1972)** — classical linear-associative-
   memory chaining, the source of §3's idealized-Z ceiling fact. Not claimed
   as novel (same discipline Task D applied to the rank≥K fact).

---

## 11. Attacks this design must survive (research + attack pass already run)

A dedicated research+attack agent (novelty check + adversarial pass, general-
purpose agent, ~15 tool uses) was run against this exact design before
writing this document. Findings folded in above; summarized here for the
audit gauntlet that comes next:

- **Novelty: clear.** No paper combines a provable rank≥K necessity theorem,
  a causal train-time force-rank-k ablation, a literal matrix-power readout
  (no learned per-hop weights), and zero-shot hop-depth extrapolation. See
  §10 for the closest lines and how each differs.
- **Landed, FIXABLE — the merge/miscounting attack (strongest attack found).**
  The one-hop rank≥K proof requires linearly independent values, which
  requires `π` to be injective (in-degree ≤ 1), not just out-degree ≤ 1.
  Without this, "K edges" does not imply "K rank constraints" — the exact
  `KILL_LIST.md`/B-4 miscounting trap that killed MNNS, reincarnated.
  **Fix applied in §2/§5 (C6): injectivity is enforced by the generator and
  asserted at generation time, mandatory and load-bearing, not optional.**
- **Landed, reframe not fatal — "composition is classical, not novel,
  under an injective graph."** The hand-built minimum-norm `Z_ideal`
  composes exactly (§3) with zero extra training pressure — this is textbook
  chaining (Kohonen/Anderson), not new. **Fix applied: H_E and §1/§7 are
  reframed around eigenstructure fidelity to `Z_ideal` (measured via C8, not
  asserted), not "genuine relational semantics" — avoids overclaiming a
  positive result beyond what it actually shows.**
- **Landed, FIXABLE — hop-index side-channel.** If `h` (or any query-time
  signal) reaches anything that helps *compute* `Z`, or if a per-h learned
  parameter exists, the model can memorize `(start, h) → answer` pairs
  without composing. **Fix applied (C_composition-purity, §5): extended
  blank-out test, mandatory smoke-gate item, verifies `pred(a,h)` is a pure
  function of `(Z, key_a, h)` via exactly h matmuls with no other pathway.**
- **Not landed / addressed by design, not new fixes:** the master
  rank-1-in-costume shortcut (`archive/chapter2-gauntlet/gauntlet/ATTACK_task_shortcuts.md` §0) is
  closed the same way Task D closes it — continuous pinned readout, no
  argmax, no unconstrained MLP on the primary arm (C_MLP is a deliberately
  separate, labeled shortcut-permitting control, not the headline model).
  Position-decomposition (the ProsQA-MULTI failure mode, §9) cannot occur
  here because there is still exactly one global matrix state (P=1), no
  multiple latent slots to route information into.
- **Open, pre-registered as such (not resolved by fiat):** whether held-out
  generalization is graceful or a hard cliff, and whether composition needs
  strictly more rank than recall (M4_E secondary) are genuinely unknown a
  priori — both directions are pre-registered as informative, per §6.

**Post-build audit gauntlet (2026-07-01, `archive/chapter2-gauntlet/gauntlet/AUDIT_task_e_correctness.md` +
`archive/chapter2-gauntlet/gauntlet/AUDIT_task_e_validity.md`), findings applied:**
- **FATAL (both audits):** `_assert_injective`'s rank threshold had a `-1`
  slack that could not detect a single-pair merge — the codebase's own
  negative unit test (`_test_injectivity_guard_detects_merge`) proved this by
  failing to raise. Fixed to an exact `vrank >= K_eff` threshold; the smoke
  gate's step [1] (`task_e._self_test()`) now actually reaches its PASSED
  line.
- **FATAL (validity):** cycle-length periodicity confound described above
  (§2) — fixed via the single-K-cycle generator, the config-time periodicity
  guard, the corrected `H_extra=(7,21)` default, dropping K=4 from the Stage-2
  sweep (§8), and per-hop `effective_hop` stratification in `evaluate_at_hops`
  (§6). **Update (2026-07-01 addendum, §6):** the `H_extra=(7,21)` fix was
  itself incomplete — `h=21` collides with an `H_test` residue (not just
  `H_train`) at K=8 and K=16, so it validly probes novel ground only at
  K=12. See §6's addendum for the full accounting and the reinterpretation
  of `h=21` as a depth-decay probe.
- **MAJOR:** the `--out` JSON now records the full run config (K, d, variant,
  H_train/H_test/H_extra, force_rank_k, n_params), matching Task D's
  `evaluate()` convention. The smoke gate gained a new step [5]: the model's
  own (post a few real SGD steps) learned Z is forwarded through
  `hop_set=cfg.H_extra` and asserted finite, closing the "numerical explosion
  at held-out h is only tested against Z_ideal/random, never the real model"
  gap.
- **MINOR:** `eigen_utils.eigenvalue_fidelity`'s output tensor now inherits
  `Z`'s device; `MLPShortcutModel` now accepts the actual `H_train` set and
  uses exact membership (not a `[1, h_train_max]` range) for its OOV
  one-hot(h) check, closing the non-contiguous-H_train fragility at the class
  level, not just the CLI level.

---

## 12. Reproducibility (to be filled in as Stage 1 lands)

- This design: `matrix-thinking/chapter2/NEXT_EXPERIMENT_DESIGN.md`.
- Builds on: `matrix-thinking/chapter2/task_d.py`, `model_v4.py`,
  `rank_utils.py`, `run_task_d.py` (reused, not reimplemented, for the
  one-hop path and encoder).
- Prior art for the reasoning-transfer attempts this supersedes/avoids:
  `matrix-thinking/KILL_LIST.md` (MNNS), `.team-runs/20260430-235745-position-decomp-gauntlet/`
  (ProsQA-MULTI position-decomposition).
- Novelty + attack pass backing §10–§11: run inline for this design (no
  separate file written; findings folded into this document directly).
- Next: build Stage 1 (task_e.py, model changes, extended smoke gate) →
  audit by a fresh-context agent (not the implementer) → Stage 0 → Stage 2,
  per `CLAUDE.md`'s Build → Audit → Run workflow.
