# Task E Findings — 40K-Step Calibration Round (Compositional Multi-Hop Relational Recall)

**Dated 2026-07-02 UTC. Status: INTERIM / mid-round.** n=24/33 planned runs
complete on the Brev 8×H100 cluster (K=12 and K=16 unconstrained have 2-3
stragglers each still running; one Tier-2 `C_MLP` run is still draining). All
numbers below are read directly from per-run JSONs on the cluster — cite as
"40K calibration round, n=24/33" and re-pull before citing externally, per
this project's standing convention for mid-sweep snapshots
(`TASK_D_FINDINGS_DRAFT.md`). Design/pre-registration:
`NEXT_EXPERIMENT_DESIGN.md`, including its 2026-07-01 `H_extra` addendum
(the `h=21` residue-collision reinterpretation, load-bearing for §3 below).
Prior round: `EXPERIMENT_LOG.md`'s "Task E 20K-step round — calibration
finding" entry (2026-07-01), which this round directly supersedes.

---

## 1. What ran

The 20K-step round (`EXPERIMENT_LOG.md`, 2026-07-01) was stopped for
violating `CLAUDE.md`'s mandatory-calibration-run rule: training showed a
late, seed-stochastic phase transition (loss flat at ~1.0 for 60-80% of
budget, then sharp collapse), so a 20K-step budget mostly measured "did this
seed transition in time" rather than the underlying hypothesis. This round
relaunches at **40K steps** (`run_task_e_sweep.py::build_manifest(calibrate=True)`,
a fresh `--out-dir` so run names don't collide with the 20K results) to give
the transition room to occur.

Grid (`d=16`, permutation variant, `H_train={1,2,3}`, `H_test={4,5,6}`,
`H_extra={7,21}`, orthonormal keys):

- **Tier 1, M1_E/M2_E/M3_E** (unconstrained rank, tag `frN`): `K ∈ {8,12,16}`
  × 5 seeds = 15 runs.
- **Tier 1, M4_E calibration probe** (not the full straddle grid — see §6):
  force-rank `k ∈ {7,8,9}` at `K=8` (the primary M4_E operating point) × 3
  seeds = 9 runs.
- **Tier 2, C_MLP floor**: `K ∈ {8,12,16}` × 3 seeds = 9 runs.

33 runs total; 24 complete as of this writing.

---

## 2. Headline results — K=8, unconstrained rank (M1_E/M2_E/M3_E)

`recovered_frac@0.9` by hop, plus whole-`Z` stable rank (of `d=16`) and the
raw (unrestricted) eigenvalue-fidelity metric from `eigen_utils`, 5/5 seeds
complete:

| Seeds | h=1 (ID) | h=2 (ID) | h=3 (ID) | h=4 (held-out) | h=5 (held-out) | h=6 (held-out) | h=7 (held-out) | h=21 (depth probe) | stable_rank | eig_fidelity_mean | n_skipped_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **s1, s2, s3, s4** (4/5, converged) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 14.7–15.6 | 0.60–2.30 | 0 |
| **s0** (1/5, partial/stuck) | 0.93 | 0.76 | 0.58 | 0.43 | 0.30 | 0.21 | 0.16 | **0.00** | 5.96 | — | 0 |

4 of 5 seeds are **exact to depth 21** — cosine-threshold recovery of 1.00 at
every tested hop, in-distribution and held-out alike, with zero non-finite
training steps. This is a qualitative change from the 20K round, where only
1/5 seeds converged at all and that one seed's held-out recovery **decayed**
with hop distance (0.964 → 0.774 at h=4→7) rather than holding flat at 1.0.
Seed s0 shows the same decaying-with-hop signature the one 20K-round
convergent seed showed, at roughly half the eventual recovery — read as a
seed still mid-transition or converged to an inferior local solution, not a
new failure mode.

**Force-rank probe at K=8** (`fr ∈ {7,8,9}` × seeds 0-2, 9 runs — the
calibration-only subset of the M4_E straddle grid, not the full grid):

| Config | Result |
|---|---|
| **fr=8, fr=9** (rank ≥ K=8, provably sufficient), all 6 seeds | **All dead.** stable_rank ≈ 1.0–1.15, recovery 0.00 at every hop, `n_skipped_steps` 3–10 per run (eigh-backward instability events during training). |
| **fr=7, seed s2** (rank K−1=7, provably *insufficient* for exact recovery) | **Converged**, but with a distinct signature: ID recovery 1.00/0.99/0.99 (h=1-3), held-out 0.97/0.95/0.92/0.88 (h=4-7), **h=21 = 0.06.** |
| **fr=7, seeds s0, s1**; remaining 8/9 probe runs not itemized above | Consistent with the fr=8/fr=9 dead pattern (not separately confirmed convergent). |

Only 1 of 9 force-rank calibration runs converged at all.

---

## 3. The depth-amplification mechanism finding

The fr=7, s2 result is the most mechanistically informative single number in
this round. At the `@0.9` cosine tolerance, a rank-7 approximation of an
8-cycle is good enough to pass at shallow depth: dropping one of the
8 Fourier/eigenmodes that make up the exact cyclic permutation operator
leaves per-item cosine similarity at roughly `sqrt(1 - 1/8) ≈ 0.94 > 0.9` —
comfortably above threshold at h=1 through h=7. But 21-fold self-application
of the same deficient operator (`Z^21`) amplifies that missing-mode error
past the threshold: h=21 collapses to 0.06.

Contrast this directly with the K=8 unconstrained result (§2): seeds s1-s4,
whose whole-matrix rank is provably sufficient (stable_rank 14.7-15.6, well
above K=8), hold cosine 1.00 at h=21 with **zero** decay. Same task, same K,
same hop, two different rank regimes — one exact through depth 21, one
exposed exactly there.

**Mechanism finding: raw iteration depth (the literal count of sequential
matmul self-applications of `Z`) is a spectral-exactness amplifier, and a
sharper rank test than any single-hop cosine tolerance.** A rank-deficient
operator can look behaviorally correct at shallow, single-digit hop depths
under a fixed cosine bar and only reveal its deficiency once composed enough
times. This is the converse of the depth-decay discovery from the 20K round
(`EXPERIMENT_LOG.md`, 2026-07-01): there, the same `h=21` probe — which,
per the design doc's `H_extra` addendum, collides with the `H_test` residue
`effective_hop = 21 mod 8 = 5` at K=8 (and at K=16, but *not* at K=12,
where `21 mod 12 = 9` is genuinely novel ground) — separated a converging-
but-undertrained seed's h=5 recovery (0.915) from its h=21 recovery (0.006)
despite identical effective hop. This round's fr=7 result reproduces the
same signature under controlled conditions (a *known*-deficient rank
regime, not just an undertrained seed) and adds the clean positive
comparison: exact operators (rank ≥ K, converged) survive depth 21
perfectly; deficient operators (rank = K−1, converged) are exposed by it.
`h=21` at K=8/K=16 should continue to be read as a **depth-decay probe**,
not a held-out-hop generalization probe, per the design doc's standing
guidance — this round's data are consistent with, and sharpen, that
reinterpretation rather than overturning it.

---

## 4. The rank-inflation (M1_E) complication, and what it means for the thesis

This is the most important honest caveat in this round.

**Whole-`Z` effective/stable rank of the converged K=8 solutions (s1-s4) is
≈ 14.7-15.6 — essentially the full ambient dimension `d=16` — not ≈ K=8.**
Task D's M1 finding ("learned effective rank tracks binding count K almost
exactly") does **not** straightforwardly transfer to multi-hop training.
Under the one-hop objective, SGD's minimum-norm incentive keeps `Z` close to
rank K; under the multi-hop objective, SGD appears to implement the exact
K-cycle correctly on the entity subspace but also fills the orthogonal
complement — there is no training pressure to stay minimal-rank once the
compositional loss is satisfied, and it is possible the late-phase
transition itself passes through (and retains) full-rank structure on its
way to a solution.

This also explains the eigenvalue-fidelity numbers in §2: `eig_fidelity_mean`
of 0.60-2.30 on operators with **perfect** behavioral recovery (1.00 at
every hop through h=21) looks like a contradiction until read correctly —
the saved metric does a Hungarian match of **all d=16** eigenvalues of the
trained `Z` against the K=8 roots of unity of the ideal cyclic operator, so
it is measuring distance on 8 eigenvalue directions the model was never
constrained to control at all. The metric is polluted by the free
directions in the orthogonal complement, exactly as an internal review
flagged as a risk before this round's data came in. It needs subspace
restriction to be interpretable, not this whole-matrix form.

**What this means for the thesis:** the naive "rank tracks K" reading from
Task D does not transfer to Task E as-is. What we can currently say is
weaker but still real: a rank-sufficient trained `Z` composes exactly to
depth 21 (§2), and a rank-K−1 trained `Z` provably cannot (§3) — so rank
*restricted to the entity subspace* is very likely still the causally
load-bearing quantity, but this round's whole-matrix M1_E measurement
cannot itself demonstrate that, because whole-matrix rank in the converged
solutions is uninformative (pinned near `d`, not `K`, regardless of whether
composition works). The decisive check — project the trained `Z` onto the
K=8-dimensional entity subspace (span of the K keys), verify the restricted
operator is the exact permutation matrix and the restricted rank is exactly
K — is queued as an instrumented rerun (§6), not yet run. Until it lands,
"SGD discovers a genuine, minimal-rank composable K-cycle" should be read as
"SGD discovers *some* operator that composes exactly on the tested subspace
and hop range, of unconfirmed minimality" — a real but narrower claim.

---

## 5. Trainability walls

**Superseded 2026-07-02 — see §10.** The "K-axis wall" claimed below was a
step-budget artifact: at 80K steps (2× this round's 40K budget), K=12 is
3/3 seeds exact and K=16 is 2/3 converged. Left in place verbatim below for
the historical record; do not cite the "dead"/"wall" language past this
point without reading §10 first.

**K-axis wall at d=16 under multi-hop supervision.** K=8 unconstrained
converges well (4/5 seeds, §2). K=12 and K=16 unconstrained are **dead** at
40K steps in every run completed so far:

- K=12, seed s3: effective rank 1.21 (essentially rank-1), all-zero
  recovery.
- K=12, seeds s0, s1: training finished flat at loss ≈ 0.983 — never
  transitioned within the 40K budget.
- K=16, seeds s0, s2, s4: effective rank 1.0-1.5, all-zero recovery.
- K=12 (2 seeds) and K=16 (2 seeds) are still draining and not yet counted
  above; this section will need a final check once they land.

Doubling the step budget from 20K to 40K did **not** rescue K≥12 — every
completed run at K=12/16 shows the same collapsed-rank, all-zero-recovery,
or never-transitioned signature the 20K round showed, not a slower version
of the K=8 convergence pattern. This reads as a genuine trainability wall in
K at this fixed operating point, not an undertraining artifact. Notably,
Task D trained K=16 at d=16 without difficulty under **single-hop**
supervision (M1 effective rank K=16 → 15.09, M3 causal step confirmed,
`TASK_D_WRITEUP.md` §5.1-5.2) — so this wall is specific to the added
multi-hop compositional objective, not to K=16 model capacity in general.

This mirrors the d-axis trainability wall Task D already found (the same
fixed `h=64` encoder fails to train at all at `d≥32`, `TASK_D_WRITEUP.md`
§5.3). The emerging pattern across both experiments is that **the
trainability frontier moves inward whenever task complexity increases** —
larger K under a harder (multi-hop) objective here, larger d under the
original objective there. This is now the program's central diagnosed
phenomenon; Stage 0 (queued next, §6) is the scheduled attack on the d-axis
version of it.

**C_MLP floor.** All 9 completed runs: 0.00 recovery at every hop, including
in-distribution, mean cosine ≈ 0.26. The floor is confirmed in the narrow
sense the control was designed for (the rank-blind, unconstrained
flatten+one-hot(h) shortcut model does not beat chance at held-out hops).
Honest caveat, not new to this round but worth restating plainly: this
baseline is architecturally unable to extrapolate to held-out h by
construction (`one-hot(h)` is out-of-vocabulary at held-out h), so failing
there was expected regardless of any rank story. What is more notable is
that it apparently fails to fit even the **in-distribution** hops at this
model size — this could reflect a genuine architectural limitation
(flatten-then-MLP cannot learn K=8/12/16-way binding+composition at all at
this scale) or simply undertraining of the control arm; the pre-registered
continuous-h variant (`NEXT_EXPERIMENT_DESIGN.md` §5, C_MLP) is the intended
follow-up to distinguish these, and has not been run.

---

## 6. Decisions taken

1. **Calibration gate passes.** The pre-registered gate (≥3/5 seeds converge
   at K=8, unconstrained rank, by the target step budget) requires 3/5;
   this round shows 4/5. **PASS.**
2. **The full M4_E force-rank straddle grid (`FR_GRID_M4 = {2,4,6,7,8,9,10}`
   × 5 seeds = 35 runs) is NOT being launched at 40K**, despite the gate
   passing. This is a deviation from the pre-registered green-light
   condition, documented here with reasoning rather than silently skipped.
   Reasoning: the 9-run force-rank calibration probe (§2) already shows
   1/9 converged, and that one convergence (fr=7, s2) is itself only
   degenerate-informative — it demonstrates the depth-amplification
   mechanism (§3) rather than giving a clean M4_E in-distribution-vs-
   held-out causal step. Separately, fr=8 and fr=9 (rank ≥ K, provably
   *sufficient*) show a distinct new failure mode — eigh-backward numerical
   instability (`n_skipped_steps` 3-10 per run) — that appears to interact
   badly with the multi-hop objective's late phase transition, not simply
   "hasn't transitioned yet." Together this predicts a near-certain
   all-zero outcome from the full grid at an estimated ~20 GPU-h cost. Per
   the project's decision-sufficient-compute practice (run what changes a
   decision; don't burn compute on a predictably uninformative result),
   the full grid launch is skipped. **The M4_E causal staircase is
   therefore not measurable at this operating point with the current
   training recipe** — an honest gap, not a result.
3. **A Z-dump instrumented rerun is queued** to resolve the M1_E
   rank-inflation complication (§4): dump the trained `Z` for the converged
   K=8 seeds, project onto the K=8-dimensional entity subspace, and verify
   the restricted operator is the exact permutation matrix with restricted
   rank exactly K. Not yet run.
4. **Stage 0 (d≥32 trainability precursor)** is next on the box, per the
   original Task D-derived plan (`NEXT_EXPERIMENT_DESIGN.md` §8). Given
   §5's finding that this round's K-axis wall and Task D's d-axis wall look
   like the same underlying phenomenon, Stage 0's outcome may be directly
   informative for the K-axis wall too, not just the originally-scoped
   d-axis question.

---

## 7. What would change these conclusions

- **The K=12/K=16 stragglers.** 4 of the 8 completed K=12/K=16 unconstrained
  runs are in; the remaining 4 (plus the draining C_MLP run) could in
  principle include a late convergent seed. A single late convergence would
  not by itself overturn "genuine wall" (it would still be 1-for-8 or
  worse), but multiple would, and this section should be re-checked once
  the round finishes.
- **The Z-dump subspace-restricted check (§4, §6).** If the restricted rank
  on the entity subspace is *not* exactly K even for the perfect-recovery
  seeds, the "SGD discovers a genuine composable K-cycle" reading weakens
  further, to something like "SGD discovers a solution that composes
  correctly on the tested subspace/hop range without confirmed minimal-rank
  structure." If it *is* exactly K, the M1_E complication (§4) is resolved
  in the thesis's favor and the whole-matrix metric can be retired in favor
  of the subspace-restricted one going forward.
- **If the full M4_E grid were run anyway and did not produce all-zero
  results** — i.e., if the eigh-backward instability at fr=8/9 turned out
  to be seed-specific rather than a structural property of forcing rank
  under this objective — the "not measurable at this operating point"
  verdict (§6, item 2) would need to be withdrawn. The calibration evidence
  argues against this, but it is a prediction, not a certainty, since the
  full grid has not been run.
- **A continuous-h C_MLP variant that fits in-distribution hops well** would
  reframe the current C_MLP floor from "genuine rank-blind floor" to
  "architectural artifact of the one-hot(h) baseline," changing how much
  weight the C_MLP comparison in §5 and the design doc's M3_E criterion can
  bear.

---

## 8. Reproducibility

- Pre-registration + addenda: `matrix-thinking/chapter2/NEXT_EXPERIMENT_DESIGN.md`.
- Prior round (stopped, superseded by this one):
  `EXPERIMENT_LOG.md`, "Task E 20K-step round — calibration finding"
  (2026-07-01).
- Code: `matrix-thinking/chapter2/{task_e,model_v4,rank_utils,eigen_utils,
  run_task_e,run_task_e_sweep}.py` (reuses Task D's encoder/rank utilities
  verbatim).
- Orchestrator: `matrix-thinking/chapter2/run_task_e_sweep.py`, this round
  launched with `build_manifest(calibrate=True)` and `--steps 40000`.
- Audit trail: `matrix-thinking/chapter2/gauntlet/AUDIT_task_e_correctness.md`,
  `AUDIT_task_e_validity.md`.
- This snapshot: per-run JSONs on the Brev 8×H100 cluster (n=24/33); no
  local aggregate file has been pulled yet for this round (unlike Task D's
  `results/overnight_snapshots/`) — a follow-up pull/aggregate is needed
  before any figure is generated from this data.

---

## 9. Z-dump subspace analysis (2026-07-02) — resolves the §4 rank-inflation complication

**Data.** 8 run JSONs (`t1_matrix_permutation_K8_{frN,fr7}_s*.json`) pulled
from `/home/nvidia/chapter2/results/task_e_40k_zdump/` on the cluster to
`experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump/`, each carrying an
embedded `Z_dump` (4 eval examples: the trained 16×16 operator `Z` and the
analytic K=8-cycle target `z_ideal`). Covers `frN` (unconstrained rank)
seeds 0–4 and `fr7` (rank-7-forced) seeds 0–2. **Reproducibility check
(required before trusting this dump): every metric field in every hop entry
of `M2_in_distribution`/`M3_held_out` across all 8 JSONs is exact float64
equality against the corresponding value in the original 40K calibration
round** (`experiment-runs/2026-07-02_task_e_40k/task_e_40k/`) — verified
programmatically over the full field set (`mean_cos`, `recovered_frac@0.9`
etc. at every hop), not a rounded/spot-check comparison. This is
consistent with the Z-dump job having loaded the calibration round's already-
trained checkpoints and added the subspace-projection/dump instrumentation at
eval time only, rather than an independent retrain — so there is no GPU-
nondeterminism seed-flip question to resolve (no reruns; same checkpoints,
same numbers), and the dumped `Z` matrices are exactly the operators §2–§4
already reported on, not new ones.

**Method.** `matrix-thinking/chapter2/analyze_zdump.py` (rerunnable,
numpy+stdlib only). For each (run, example): recover the K-dim entity
subspace `E` from `z_ideal`'s SVD (`U`, top-K left singular vectors,
threshold `1e-3 × σ_max`); block-decompose the trained `Z` in the `[E, E⊥]`
basis as `A = UᵀZU` (restricted operator), `B = UᵀZV` (E⊥→E leakage),
`C = VᵀZU` (E→E⊥ leakage), `D = VᵀZV` (complement); compare `A` to
`Π := Uᵀz_idealU` (the exact cycle expressed in the same basis), both raw and
after fitting the single least-squares isotropic scale
`c* = argmin_c‖A − cΠ‖_F` (cosine similarity — what every recovery number in
this file is scored on — is invariant to a uniform scalar rescaling of `Z`,
so a raw, scale-naive comparison conflates a cosine-irrelevant magnitude DOF
with genuine structural error; this was caught empirically, not assumed —
the raw `‖A−Π‖_F/‖Π‖_F` numbers below are large even for the seeds with
*exact* h=21 recovery, and only make sense once `c*` is fit and removed).
Eigenvalues of `A` are Hungarian-matched (same optimal-assignment algorithm
as `eigen_utils.py::_hungarian_min_cost`, ported) against the K-th roots of
unity **after** normalizing each eigenvalue to the unit circle — matching on
raw (magnitude-inclusive) distance was tried first and produced incoherent
assignments for exactly the reason above. A synthetic-key construction lets
the depth-decay curve be predicted purely from `A`'s own spectrum with **no
access to the raw (unrecorded) key vectors**: because the true keys are a
cyclically-permuted orthonormal set, they decompose into `Π`'s own
eigenbasis with *exactly* equal `1/√K` magnitude on every mode regardless of
which orthonormal basis was chosen for `E` (derived in the script's module
docstring; self-consistency `‖Π·aᵢ − aᵢ₊₁‖ = 0.000000` on every run, so the
construction is exact, not approximate).

### Headline table (means over the 4 eval examples per run)

| run | k_eff | c* | ‖A−c\*Π‖/‖c\*Π‖ (phase/structure residual) | eig(A) phase-resid max | ‖B‖ | ‖C‖ | ‖D‖ | ρ(D) | eff_rank(D) | eff_rank(A) | cond(D) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frN s1 (exact, h=21 cos 1.00) | 8 | 1.375 | 0.0074 | 0.0020 | 0.023 | 0.024 | 3.89 | 1.38 | 8.000 | 8.000 | 1.010 |
| frN s2 (exact) | 8 | 1.250 | 0.0189 | 0.0052 | 0.061 | 0.053 | 3.55 | 1.27 | 8.000 | 7.999 | 1.032 |
| frN s3 (exact) | 8 | 2.843 | 0.0152 | 0.0031 | 0.080 | 0.099 | 8.03 | 2.86 | 8.000 | 7.999 | 1.027 |
| frN s4 (exact) | 8 | 1.011 | 0.0238 | 0.0117 | 0.056 | 0.059 | 2.85 | 1.02 | 7.999 | 7.999 | 1.043 |
| frN s0 (partial) | 8 | 1.696 | 0.3209 | 0.0638 | 3.274 | 0.636 | 4.13 | 1.65 | 7.899 | 7.735 | 1.753 |
| fr7 s2 (converged, tolerance slack) | 8 | 1.898 | 0.3926 | 2.000 | 0.470 | 0.341 | 0.029 | 0.013 | 4.611 | 6.985 | 1.86×10⁶ |
| fr7 s0 (dead) | 8 | −1.149 | 12.17 | 1.090 | 23.81 | 26.87 | 22.91 | 8.79 | 1.004 | 1.003 | 1.09×10¹⁰ |
| fr7 s1 (dead) | 8 | −1.039 | 6.04 | 1.507 | 24.14 | 28.05 | 22.37 | 7.03 | 1.001 | 1.001 | 5.11×10⁸ |

`k_eff` = 8 = K exactly on every run/example (SVD threshold `1e-3σ_max`
against a spectrum that is cleanly `{1,1,1,1,1,1,1,1,~1e-8,...}` in every
case, the residual ~1e-8 values being pure fp32 storage noise) — the
row-space/column-space principal angle between `z_ideal`'s left and right
singular-vector bases is `0.000°` on every one of the 32 (run × example)
checks, confirming row space = column space for the cycle, as expected.

### Verdicts

**(a) Restricted operator == exact cycle?** **Yes, up to a cosine-invisible
isotropic scale**, for every converged `frN` seed (s1–s4): scale-corrected
residual `‖A − c*Π‖_F/‖c*Π‖_F` is 0.7–2.4%, and the max eigenvalue
phase-only residual (chord distance after normalizing each eigenvalue to the
unit circle; max possible value 2.0 for diametrically opposite points) is
0.002–0.012 per seed. The **raw** (magnitude-inclusive) comparison is badly misleading —
`‖A−Π‖_F/‖Π‖_F` runs 0.05–1.84 across these same seeds, which would read as
"nowhere near the exact cycle" — entirely explained by the fitted scale `c*`
ranging 1.0–2.8 across seeds: SGD finds an operator on `E` that is the exact
8-cycle times an arbitrary positive scalar, and cosine similarity (the only
thing the training/eval loss ever measures) cannot see that scalar at all.
This is the single biggest resolution in this round: **the "eigenvalue
fidelity" story only makes sense once the scale DOF is removed**, matching
in spirit (not mechanism) the earlier §4 observation that the *whole-matrix*
`eig_fidelity_mean` metric was polluted by directions the model was never
constrained on — here the pollution is a scalar, not a subspace, but the
same lesson applies: raw eigenvalue/Frobenius distance to `Z_ideal` is not
directly interpretable without first identifying which degrees of freedom
the readout actually scores.

**(b) Restricted rank == K?** **Yes, cleanly**, for the converged `frN`
seeds: `effective_rank(A)` = 7.999–8.000 (out of a maximum possible 8) for
s1–s4, essentially exact. This is the direct resolution of §4's headline
complication: **restricted to the entity subspace, the "rank tracks K"
reading from Task D's M1 does hold** — it was the *whole-matrix* rank
(14.7–15.6 out of d=16) that was uninformative, not because SGD ignores rank
discipline on the task-relevant subspace, but because §4's measurement was
on the wrong matrix. Once you restrict to `E`, K=8 recruitment is exact.

**(c) Leakage ≈ 0 — which condition, and does the data satisfy it?**
Derivation (see script docstring): for a query `u ∈ E`, `Zu`'s component
back in `E` is `Uᵀ(Zu) = Aa`, exactly, and its component leaked into `E⊥` is
`Ca`. By induction, `Z^h u` stays *entirely* within `E` — so `U^TZ^hu = A^h a`
exactly, independent of `B` and `D` entirely — **if and only if `C ≈ 0`**;
`B` (the reverse E⊥→E leakage) is irrelevant to any query that starts in
`E`, because such a trajectory never visits `E⊥` in the first place when
`C=0`, regardless of what `B` or `D` look like. **The data satisfies this
asymmetrically, exactly as predicted**: for the converged `frN` seeds, `‖C‖`
(0.024–0.099) and `‖B‖` (0.023–0.080) are both small relative to `‖Z‖`
(4.0–11.4, i.e. <2.5% relative leakage either way) — but for the partial
seed s0, they diverge sharply: `‖B‖ = 3.27` is **5× larger** than
`‖C‖ = 0.64`, exactly the pattern the derivation predicts is harmless (large
B, small-ish C) for a solution whose behavioral defects, per (e)-adjacent
analysis below, come from `A`'s own imperfect phase structure rather than
E⊥ leakage. Direct empirical confirmation that `C≈0` is the operative
condition, not merely a plausible derivation: the depth-decay curves
predicted **purely from `A` and `Π`, with `B`, `C`, `D` never used at all**,
match every run's actual measured `mean_cos` at every hop (h=1…7, 21) to
within 0.001–0.02 for the `frN` seeds and 0.001–0.007 through h=7 for fr7
s2 (h=21 gap 0.067, still same order and same direction — see (e)). If `C`
were not negligible, ignoring it entirely could not have produced predictions
this accurate.

**(d) What is D, and why is it invisible?** `D` is the single biggest
resolution of the "where did the extra whole-matrix rank go" question:
**`effective_rank(D)` is 7.9–8.0 (essentially FULL rank on its own 8
dimensions) for every converged `frN` seed** — this is exactly the missing
piece of §4's whole-matrix rank of 14.7–15.6 ≈ K(=8, in A) + (d−K)(=8, in D,
nearly saturated). `D` is not a decaying, benign leftover: `spectral_radius(D)`
is **1.0–2.9 across seeds — at or above 1 in every case** — meaning if a
query trajectory ever entered `E⊥`, repeated self-application would not
attenuate it, and in most seeds would actively grow it. It is invisible only
because `C≈0` means no real query trajectory (which always starts in `E`,
since keys are among the K entities) ever visits it — an unconstrained,
potentially-unstable subspace that exists purely because nothing in the loss
ever asks `Z` to be small or contractive off the entity subspace, not because
SGD "chose" to put anything useful there. Is `D` structured or unconstrained
noise? **Neither cleanly "noise" in the i.i.d.-Gaussian sense nor
"structured" in a task-relevant sense — its singular-value spectrum is
almost perfectly flat**: `condition_number(D)` (top/bottom singular value)
is 1.01–1.75 for every converged/partial `frN` seed, i.e. `D` is close to a
scaled near-isometry (an ad hoc reference check: a random 8×8 Gaussian
matrix scaled to the same Frobenius norm as `frN s1`'s `D` has condition
number ≈18, an order of magnitude less flat). Reading: `D` looks like
whatever the encoder's natural (untrained-in-this-direction) output
distribution defaults to — plausibly inheriting the same near-isotropic bias
that makes `A` a scaled rotation on `E` — rather than either meaningful
task structure or literal noise.

**(e) fr7 s2 mode analysis + depth-decay curve match.** This is the
clearest single confirmation of the depth-amplification mechanism (§3).
Phase-only eigenvalue residuals for `A` (after normalizing each eigenvalue
to the unit circle, then Hungarian-matching): **exactly one mode per example
has residual 2.0 (the maximum possible — diametrically opposite phase — in
practice because that mode's raw magnitude is ≈ 0.0000, i.e. genuinely
absent, not merely mis-rotated) while the other 7 modes match to residual
0.001–0.039**, with those 7 surviving modes sharing a common magnitude
≈1.9–2.35 (the fitted `c* ≈ 1.90`). This is exactly the "one clean
Fourier/eigenmode dropped, the other 7 intact" story §3 above
predicted from the `sqrt(1−1/8)≈0.94` heuristic — now confirmed as an actual
eigenvalue, not inferred from a single cosine number. Predicted vs. measured
depth curve (predicted from `A`'s literal matrix powers via the no-raw-keys
synthetic-key construction, no fitting):

| h | eff. hop | predicted cos(h) | measured mean_cos | measured recovered_frac@0.9 | Δ (meas−pred) |
|---|---|---|---|---|---|
| 1 | 1 | 0.9317 | 0.9303 | 0.996 | −0.0015 |
| 2 | 2 | 0.9286 | 0.9280 | 0.989 | −0.0007 |
| 3 | 3 | 0.9258 | 0.9259 | 0.986 | +0.0002 |
| 4 | 4 | 0.9227 | 0.9237 | 0.970 | +0.0009 |
| 5 | 5 | 0.9181 | 0.9212 | 0.947 | +0.0031 |
| 6 | 6 | 0.9136 | 0.9185 | 0.917 | +0.0049 |
| 7 | 7 | 0.9089 | 0.9163 | 0.881 | +0.0075 |
| 21 | 5 | 0.7536 | 0.8206 | 0.060 | +0.0670 |

Both curves are essentially flat 0.91–0.93 through h=7 (matching §3's
`sqrt(7/8)≈0.935` one-shot estimate almost exactly) and both drop
sharply by h=21 — a **quantitative, not just qualitative, match on the
depth-amplification mechanism**: dropping one eigenmode barely moves
single-digit-hop cosine (well above the 0.9 threshold) but the same
deficiency compounds under `A^21` enough to collapse `recovered_frac@0.9`
from ~0.88 (h=7) to 0.06 (h=21) even though `mean_cos` itself only falls
from 0.93 to ~0.82/0.75 — the *fraction* of items crossing the 0.9 threshold
is far more sensitive than the mean, because different items have different
phase alignment with the missing mode and the per-item distribution widens
under repeated composition, not just shifts. The h=21 gap (predicted 0.754
vs. measured 0.821, Δ=0.067) is the largest single discrepancy in the whole
table but is in the same direction and same order of magnitude as every
other (much smaller) gap at shorter hops, consistent with 21-fold
self-application compounding the residual (non-exactly-zero) leakage and
scale-fitting error that are negligible at h≤7 — not a sign the mechanism is
wrong.

**Dead runs, contrast.** `fr7` s0/s1 (the eigh-backward-instability dead
runs, `n_skipped_steps` 8 and 10) collapse `A` to `effective_rank(A) ≈ 1.00`
— **even restricted to the entity subspace**, not just in the whole matrix —
confirming the §2/§5 "dead" pattern is a genuine collapse of the task-
relevant operator, not an artifact of measuring the wrong matrix. The single
surviving mode does not align coherently with any root of unity (phase
residuals scattered across the full 0–2.0 range across examples and modes,
no consistent winner) — consistent with a rank-collapsed operator pointing in an arbitrary direction
rather than a degenerate-but-still-partially-correct solution. `cond(D)` for
these runs is enormous (5×10⁸–10¹⁰) because `D` itself is also rank-collapsed
(near-singular), not because it is more "structured" than the healthy runs.

**Partial seed frN s0 — how does the late-phase transition build the
solution?** No forcing of a story here — reporting what is actually seen.
Unlike fr7 s2's clean single-outlier pattern, **s0 shows no discrete missing
mode at all**: all 8 phase-only residuals are small-but-nonzero and
*graded* (example 0: `0.0, 0.0, 0.018, 0.018, 0.053, 0.053, 0.071, 0.071`,
sorted ascending, complex-conjugate pairs repeating each value — every mode
carries some error, none is cleanly "correct" or cleanly "absent"). This is
a genuinely different signature from fr7 s2's "K−1 modes exact, 1 mode
gone": it looks like **s0 is still
globally converging across all modes simultaneously** (diffuse, graded
imperfection) rather than having locked in a subset of modes and left the
rest untouched. This is consistent with — though this dump alone cannot
fully confirm — a late-phase transition that improves all K modes'
alignment together rather than snapping them into place one at a time.
`‖B‖=3.27` also stands out as by far the largest of any `frN` seed (~41–142×
the converged seeds' `‖B‖` of 0.023–0.080), while `‖C‖=0.64` stays comparable to the
converged seeds — since `B` is provably irrelevant to E-originating query
behavior (see (c)), this is a real but behaviorally-inert difference, not
part of the explanation for s0's decaying-with-hop recovery.

**Bottom line for the thesis (updates §4).** The whole-matrix M1_E
measurement in §4 was the wrong instrument, not a wrong result: restricted
to the K=8-dimensional entity subspace, the converged solutions are (i) rank
exactly K, (ii) the exact classical K-cycle up to a cosine-invisible
isotropic scale, and (iii) functionally decoupled from a full-rank,
occasionally-unstable orthogonal complement that the loss never constrains
and no real query ever visits. **"SGD discovers a genuine, minimal-rank
composable K-cycle" is now confirmed on the entity subspace, not merely
consistent with the behavioral evidence** — the queued instrumented check
from §6 item 3 is complete and resolves in the thesis's favor. The whole-
matrix rank/eigenvalue-fidelity metrics from §2/§4 should be retired in favor
of the subspace-restricted ones (`analyze_zdump.py`) for any future Task E
reporting.

Reproducibility: `matrix-thinking/chapter2/analyze_zdump.py --dir
experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump` (numpy + stdlib
only) regenerates every number in this section.

---

## 10. K-wall resolution — 80K round (2026-07-02)

**Status: RESOLVED. §5's "K-axis trainability wall" claim is retracted.**
It was a step-budget artifact, not a genuine capacity limit: K=12 and K=16
unconstrained rank, which were 0/5 and 0/5 "dead" at 40K steps (§5), are
3/3 and 2/3 converged at 80K steps — double the budget, same grid
otherwise.

**Data.** 6 runs, `K ∈ {12, 16}` unconstrained rank (`frN`) × seeds
`{0, 1, 2}`, `d=16`, 80,000 steps, same task config as the 40K round
(permutation variant, `H_train={1,2,3}`, `H_test={4,5,6}`,
`H_extra={7,21}`, orthonormal keys). Pulled from
`/home/nvidia/chapter2/results/task_e_80k_kwall/*.json` on the Brev
8×H100 cluster to `experiment-runs/2026-07-02_task_e_80k_kwall/`
(archived here, 6 JSONs; training logs referenced below remain on the
cluster at `.../task_e_80k_kwall/logs/`, not copied locally).
`n_skipped_steps = 0` in every run (no eigh-backward instability this
round, unlike the 40K round's force-rank arms).

### Results

`recovered_frac@0.9` by hop, whole-matrix effective rank (of `d=16`, mean
over eval examples):

| K | seed | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=21 | eff. rank | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | s0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.99996 | 15.99 | exact |
| 12 | s1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 16.00 | exact |
| 12 | s2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 16.00 | exact |
| 16 | s0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9960 | 15.99 | near-exact (h=21 depth-decay) |
| 16 | s1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.00 | never transitioned |
| 16 | s2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.99997 | 0.9997 | 0.2617 | 15.96 | near-exact through h=7, depth-decay at h=21 |

**K=12: 3/3 exact.** `recovered_frac@0.9` = 1.00 at every hop 1–7 and at
h=21 for s1/s2; s0's h=21 is 0.999959 (one fp epsilon below exact, not a
distinct failure mode — cosine `mean_cos` at h=21 for s0 is 0.9720 vs.
0.9976–0.9976 for s1/s2, a small but real spread, all still far above the
0.9 recovery bar). **K=16: 2/3 converged.** s0 is essentially perfect
(h=1–7 exact, h=21 = 0.9960 — the depth-amplification signature from §3,
here mild); s2 shows the same signature more sharply (exact through h=6,
0.9997 at h=7, collapsing to 0.2617 at h=21 — quantitatively the same
"one-mode-off, amplified by 21-fold self-application" pattern §3/§9
already characterized in detail for the fr=7 force-rank case, now seen in
an *unconstrained*-rank seed instead). s1 never transitioned: training log
`cosine_loss` oscillates in [0.99, 1.00] for the full 80,000 steps with no
downward trend at any point (confirmed by sampling every 500-step log
line), effective rank pinned at 1.003, all-zero recovery at every hop —
this is the same "stuck seed," not "slow seed," signature as the 40K
round's K=8 s0 partial case, not evidence against the budget-artifact
finding for the other 5 seeds.

**Whole-matrix effective rank ≈ 16.0 = d in every converged run, K=12 and
K=16 alike** (15.96–16.00 across all 5 converged seeds; the dead K=16 s1
sits at 1.00). This is the same rank-inflation pattern §4 found and §9
resolved for K=8: the whole-matrix number is not informative about
task-relevant rank on its own — §9 showed, for K=8, that once you restrict
to the entity subspace the *restricted* rank is exactly K and the
restricted operator is the exact cycle. The K=12/K=16 runs in this round
were launched with `--save-z`, so their `Z_dump` blocks are available for
the same subspace-restricted check; **this analysis is queued, not yet
run** — §9's K=8 result should not be assumed to transfer to K=12/16
without rerunning `analyze_zdump.py` against this round's dumps.

### Why 40K looked dead: transition onset

Sampling `cosine_loss` from the training logs at 10K-step intervals for
the 5 converged seeds:

| run | loss @ 39.5K | loss @ 49.5K | loss @ 59.5K | loss @ 69.5K | loss @ 79.5K |
|---|---|---|---|---|---|
| K12 s0 | 0.988 | 0.963 | 0.445 | 0.010 | 0.001 |
| K12 s1 | 0.995 | 0.838 | 0.004 | 0.001 | 0.000 |
| K12 s2 | 0.994 | 0.970 | 0.173 | 0.004 | 0.001 |
| K16 s0 | 0.995 | 0.962 | 0.609 | 0.026 | 0.004 |
| K16 s2 | 0.989 | 0.985 | 0.746 | 0.064 | 0.013 |

Every one of these seeds is still at loss ≈0.96–1.00 — indistinguishable
from "dead" — at the 40K mark, and transitions somewhere in the
**45K–75K** window. The 40K round's K=12/K=16 runs were killed by the
clock, not by the task: they were sampled at exactly the step count where
every one of these seeds still looks flat. **Updated transition-onset
scaling: K=8 transitions in ~12–40K steps (40K round, §2); K=12/K=16
transition in ~45–75K steps at d=16 (this round) — onset grows with K, on
top of the onset-grows-with-`d` pattern Stage 0 found on the other axis.**

### The three-budget-artifacts pattern (program-level finding)

This is the **third** time this project has declared a cell "dead" at a
fixed step budget and had that verdict overturned by a late,
seed-stochastic phase transition once the budget was extended, all via
the same mechanism (loss flat near the trivial-solution value for the
large majority of the budget, then a sharp late collapse):

1. **Task E, K=8, 20K→40K** (2026-07-01→07-02): 1/5 seeds converged at
   20K; 4/5 at 40K. (`EXPERIMENT_LOG.md`, "Task E 40K-step round —
   calibration finding.")
2. **Stage 0, d≥32** (Task D's original 8K-step budget → Stage 0's 100K-step
   probe): the original d≥32 "wall" (effective rank ≈1, recovery ≈0) was
   substantially a step-budget artifact — 17/17 d=32 baseline seeds tested
   across Stage 0 transition reliably, onset 6–16K steps.
   (`STAGE0_DESIGN.md` §12–14; `EXPERIMENT_LOG.md`, "Stage 0 — 100K-step
   probe.")
3. **Task E, K=12/K=16, 40K→80K** (this section): 0/5 and 0/5 "dead" at
   40K; 3/3 and 2/3 converged at 80K, onset 45–75K.

**This is now a finding of the program, not a per-experiment footnote:
late seed-stochastic phase transitions make fixed-budget negatives
unreliable. Every "dead" cell must be re-tested at 2–2.5× budget before
being called dead** — three independent axes (K=8's seed count, d≥32's
architecture, K=12/16's binding count) have now each individually
produced a false "wall" at their first-tested budget and a real,
often-clean convergence at 2–2.5× that budget. The converse caution
applies too: Stage 0's 100K-step probe (2.5× its own 40K precursor) *did*
find a genuine, budget-independent plateau (sub-exact recovery, flat
tails) — so "re-test at 2–2.5×" is a necessary check before calling
something dead, not a guarantee that more budget always rescues a result.

### What this changes

- §5's K-axis wall is retracted as stated; the corrected picture is: K=8
  (12–40K onset), K=12 (3/3 exact by 80K, onset 45–65K), K=16 (2/3
  converged by 80K, onset 50–75K, 1/3 still stuck at 80K with no
  downward trend). K=16's remaining stuck seed (s1) is an open item, not
  resolved by this round — per the pattern above, the correct next step
  for that seed specifically is a further budget extension (e.g.
  120–160K), not a "dead" verdict; this has not been run.
- §4/§9's rank-inflation story (whole-matrix rank ≈ d, uninformative;
  restricted-to-entity-subspace rank ≈ K, exact) is echoed again in this
  round's whole-matrix numbers (K=12/K=16 converged seeds all sit at
  ≈15.96–16.00, not ≈K) but not yet confirmed at the restricted level for
  K=12/K=16 specifically — the `--save-z` dumps from this round are
  queued for the same `analyze_zdump.py` check §9 already ran for K=8.
- The depth-amplification mechanism (§3, quantitatively confirmed in §9)
  reproduces again here in an unconstrained-rank seed (K=16 s2) rather
  than only the deliberately-forced fr=7 case — further evidence this is
  a general property of near-but-not-exactly-converged operators under
  repeated self-application, not an artifact specific to force-ranking.

### Compute

6 runs × ~2.4 GPU-h ≈ **14.5 GPU-h serial-sum** (80K steps/run, d=16,
K∈{12,16}, unconstrained rank).

Reproducibility: raw JSONs archived at
`experiment-runs/2026-07-02_task_e_80k_kwall/` (training logs remain on
the cluster, see above); source config
`matrix-thinking/chapter2/run_task_e.py` / `run_task_e_sweep.py`
(same code as the 40K round, `--steps 80000`, K restricted to `{12,16}`).

### Z-dump subspace analysis — K=12/K=16 (2026-07-02): the queued check above is now run

**Runs this queued check.** Same method as §9, same script
(`analyze_zdump.py --dir experiment-runs/2026-07-02_task_e_80k_kwall`,
numpy+stdlib only, no GPU/torch needed since the dump is already on
disk), applied to this section's 6 `--save-z` dumps instead of §9's K=8
dumps. No new training runs, no additional GPU-hours — the `### Compute`
figure above is unchanged.

**Methodology note:** running the script locally (numpy 2.0.2, Apple
Accelerate BLAS backend) emits `RuntimeWarning: divide/overflow/invalid
value encountered in matmul` for every run, including the healthy ones.
Traced to `principal_angles_deg`'s `U1.T @ U2` call specifically — a
known Accelerate-BLAS spurious-warning issue (denormal-handling
artifact), not a real numerical fault: the script's own `block_decompose`
asserts `np.isfinite` on every `A/B/C/D` block and would raise
`FloatingPointError` on a genuine non-finite value (it did not, exit
code 0), and every number below cross-checks exactly against this
section's already-published, independently-derived values (e.g. `K16
s2`'s `recovered_frac@0.9`=0.2617 at h=21 in the table above reproduces
to 5 decimal places from the dump's own stored eval fields). Numbers
below are trusted; the warning is cosmetic.

**Headline table** (means over the 4 eval examples per run; same columns
as §9's table):

| run | k_eff | c* | ‖A−c\*Π‖/‖c\*Π‖ (phase/structure residual) | eig(A) phase-resid max | ‖B‖ | ‖C‖ | ‖D‖ | ρ(D) | eff_rank(D) | eff_rank(A) | cond(D) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K12 s0 (exact) | 12 | 1.508 | 0.0360 | 0.0125 | 0.094 | 0.128 | 2.994 | 1.516 | 3.999/4 | 11.996/12 | 1.045 |
| K12 s1 (exact) | 12 | 1.228 | 0.0098 | 0.0044 | 0.015 | 0.019 | 2.459 | 1.234 | 4.000/4 | 12.000/12 | 1.008 |
| K12 s2 (exact) | 12 | 1.177 | 0.0260 | 0.0099 | 0.051 | 0.047 | 2.352 | 1.184 | 4.000/4 | 11.998/12 | 1.023 |
| K16 s0 (near-exact) | 16 | 1.591 | 0.0577 | 0.0129 | 0.000 | 0.000 | 0.000 | 0.000 | n/a (0/0) | 15.985/16 | n/a |
| K16 s1 (dead) | 16 | 0.447* | 152.3 | 1.476 | 0.000 | 0.000 | 0.000 | 0.000 | n/a (0/0) | 1.002/16 | n/a |
| K16 s2 (converged, h=21 decay) | 16 | 1.877 | 0.0937 | 0.0334 | 0.000 | 0.000 | 0.000 | 0.000 | n/a (0/0) | 15.961/16 | n/a |

`*K16 s1`'s `c*` is not a meaningful scale — the 4 per-example values are
`1.65, 1.71, −1.61, 0.03` (sign-flipping, no consistent fit), the mean is
reported only for column completeness. `k_eff` = K exactly on every
run/example, same as §9; principal angle `U(col)` vs `Vrow(row)` is
`0.000°` everywhere, same as §9.

**Answering the key question: do K=12 and K=16 (s0, s2) show the same
invariant-subspace structure as K=8? Yes, for the converged seeds, with
one structural caveat at K=16=d that changes what "same" can mean.**

**(a) Restricted operator == exact cycle, up to isotropic scale? Yes for
5/6 runs.** K=12's scale-corrected residual is 0.98–3.60% (s1 best,
comparable to §9's tightest K=8 seeds at 0.74–1.52%; s0 at 3.60% is
somewhat looser but still single-digit-percent, same order of magnitude
as K8's own loosest converged seed, s4 at 2.38%). K=16 s0/s2 sit a bit
higher still — 5.77% and 9.37% — the largest scale-corrected residuals
of any converged seed across §9 and this table, but still in the same
"few-percent, not qualitatively different" regime, not the 100s–1000s%
range the dead seeds show. **K=16 s1 fails this test outright**: raw
residual 16.8×, scale-corrected residual 152×, and (per the footnote) no
consistent `c*` even exists across the 4 eval examples — this is not
"a worse fit," it is no fit, matching §9's `fr7` dead-seed signature
(`‖A−c*Π‖/‖c*Π‖` = 6.0–12.2× there) both in kind and rough magnitude.

**(b) Restricted rank == K? Yes, cleanly, for 5/6 runs.**
`effective_rank(A)` is 99.97–100.0% of `K` for all three K=12 seeds and
99.76–99.9% of `K` for K=16 s0/s2 — as exact as §9's K=8 result
(99.99–100.0%). `K16 s1` collapses to `effective_rank(A)`=1.002 (6.3% of
its K=16 target) — the same rank-1 collapse §9 found for the dead `fr7`
seeds, and consistent with this run's own whole-matrix number in the
table above (eff. rank 1.00, "never transitioned").

**(c) Leakage ≈ 0? Yes for K=12, structurally-guaranteed (not
demonstrated) for K=16.** For K=12, `‖B‖` and `‖C‖` are 0.3–2.1% of
`‖Z‖` in every seed (`‖Z‖`=4.71–6.03) — same magnitude and same
asymmetric pattern (leakage small in both directions for converged
seeds) as §9's K=8 result (<2.5%). **For K=16, `B=C=D=0` exactly in
every example, but this is not evidence about what SGD learned** — per
the task's own framing, `K=16=d` means the entity subspace `E` *is* the
full ambient space, so `E⊥` is the zero-dimensional space and `B`, `C`,
`D` are literally empty (`d−K=0`) matrices by construction, for *any*
`Z`, trained or not. The script correctly reports this (`eff_rank(D)
"out of 0" = 0.000`, no crash, no division error) rather than silently
producing a misleading `0.000` that looks like a *learned* zero-leakage
result. **The "leakage≈0" verdict at K=16 is a tautology of the
K=d geometry, not a finding about the trained operator** — it should not
be cited alongside K=8/K=12's genuine (non-tautological) low-leakage
results as if it were the same kind of evidence.

**(d) What is D, and is it the same "full-rank invisible complement"
story as K=8? Confirmed for K=12; inapplicable (no D exists) at K=16.**
For K=12, `effective_rank(D)` is 3.999–4.000 out of a maximum possible
4 (`d−K`=4) — full rank on its own dimensions, matching §9's K=8 pattern
(7.9–8.0/8) exactly. `spectral_radius(D)` is 1.18–1.52 (at-or-above-1,
non-contractive, same qualitative reading as §9(d)'s 1.0–2.9 range) and
`condition_number(D)` is 1.008–1.045 — even flatter (closer to a scaled
near-isometry) than §9's already-flat K=8 range of 1.01–1.75. **At
K=16, `D` is a 0×0 matrix — there is no complement, so the question "is
the complement full-rank and unconstrained" does not apply.** This is
the sense in which K=16 is genuinely special, not merely a boundary case
of the same story: at `K=d`, "restricted rank = K" and "whole-matrix
rank = K" become the same statement, and the entire D/leakage apparatus
that explained §4's whole-matrix rank-inflation puzzle for K<d is
vacuous by construction.

**(e) K=16 s2 (h=21 `recovered_frac@0.9`=0.2617): is the depth decay
explained by slightly-off eigenvalues, and does it match `fr7` s2's
missing-mode signature?** The depth-decay curve predicted purely from
`A`'s own spectrum (no raw keys, same construction as §9(e)) matches the
measured curve closely at every hop, including h=21 (predicted `cos`
0.8610 vs. measured 0.8686, Δ=+0.0076; predicted-vs-measured were within
0.0002–0.0014 through h=1–7) — **confirming, as in §9(e), that the
depth-amplification mechanism (§3) fully explains this run's h=21
collapse purely from `A`'s phase structure, no other mechanism needed.**
**But the micro-pattern is different from `fr7` s2's signature, not the
same.** `fr7` s2 had *exactly one* mode per example pinned at the
maximum possible phase residual (2.0, i.e. genuinely absent) with the
other 7 intact at 0.001–0.039 — a discrete single-dropped-mode story.
K=16 s2's phase residuals are **graded across all 8 mode-pairs, with no
outlier**: per-example max residual is only 0.032–0.055 (example-mean
0.0334), roughly two orders of magnitude below `fr7`'s dropped-mode
value of 2.0, and the full sorted list decays smoothly (e.g. one example:
0.055, 0.020, 0.018, 0.017, 0.012, 0.010, 0.003, 0.000 — no gap
separating a "bad" mode from "good" ones). This is the same *qualitative*
signature §9 already described for the *partial* `frN s0` seed at K=8
("no discrete missing mode... every mode carries some error, none is
cleanly correct or cleanly absent") — **K=16 s2's depth decay is real,
mechanistically the same depth-amplification phenomenon, and fully
predicted by `A`'s spectrum, but its root cause looks like diffuse
under-convergence across all K modes, not a single cleanly-dropped
Fourier mode like `fr7`.** Consistent with K=16 s2 being a converged-but-
imperfect *unconstrained*-rank seed (no artificial rank forcing) rather
than the deliberately rank-starved `fr7` construction — the two are
different routes to "one seed's `A` deviates from the exact cycle,"
and this dump lets that difference be seen directly rather than assumed.
For contrast, `K16 s0` (h=21=0.9960, near-exact) has an even smaller
mean phase residual (0.0129) with the same graded, no-outlier shape —
consistent with it simply being further along the same diffuse-
convergence axis than s2, not a different regime.

**Bottom line.** §9's thesis — SGD finds a genuine, minimal-rank
composable K-cycle on the entity subspace, invisible-but-inert full-rank
junk in the complement — **replicates at K=12 (3/3 seeds) and at K=16
for its 2 converged seeds**, with residuals in the same single-digit-
percent regime as K=8 (slightly larger at K=16, plausibly because there
is less "room" — a smaller/no complement — to absorb whatever SGD
doesn't perfectly constrain). The one qualitative change is not a
degradation but a geometric fact: **at K=16=d, the leakage/complement
apparatus (`B`, `C`, `D`) that made §4/§9's whole-matrix-vs-restricted-
rank distinction meaningful for K<d collapses to the empty matrix by
construction** — "restricted rank = K" and "whole-matrix rank = K"
coincide, so this section's whole-matrix numbers (§10's table:
eff. rank ≈15.96–16.00 for K=16's converged seeds) were *already* the
restricted number, not a separate rank-inflation puzzle needing this
script to resolve. `K16 s1` (dead) fails every converged-seed test in
this table, matching §9's dead-seed signature at K=8 — the "genuine
K-cycle on the entity subspace" story is specific to converged
solutions, not a universal feature of this task's loss landscape.

Reproducibility: `matrix-thinking/chapter2/analyze_zdump.py --dir
experiment-runs/2026-07-02_task_e_80k_kwall` (numpy + stdlib only)
regenerates every number in this section.

### K=16 s1's stuck-seed completion wave, extended to s3/s4 (120K steps, 2026-07-02): 2/5 total — a boundary-case scaling observation, not a budget artifact this time

The "open item" flagged above (K=16 s1's 80K-step non-transition, "not yet
resolved either way... the correct next step is a further budget
extension") was run: 3 seeds — s1 (re-run at higher budget) plus 2 new
seeds, s3 and s4 — at **120,000 steps** (1.5× the 80K round's budget),
same task config (`d=16, K=16`, permutation variant, orthonormal keys,
unconstrained rank, `H_train={1,2,3}`, `H_test={4,5,6}`, `H_extra={7,21}`).
`n_skipped_steps=0` in all 3 (no eigh-backward instability). Pulled from
`youthful-indigo-turkey:/home/nvidia/chapter2/results/task_e_120k_k16/`
(marked `ALL_DONE`) to
`experiment-runs/2026-07-02_task_e_120k_k16/` (3 result JSONs + training
logs).

**Result: all 3 seeds are dead — zero recovery at every hop, every seed.**

| seed | `recovered_frac@0.9` (all 8 hops) | `mean_cos` range (all 8 hops) | effective rank range | stable rank range |
|---|---|---|---|---|
| s1 | 0.0 everywhere | −0.0015 to 0.0017 | 1.494–1.549 | 1.052–1.060 |
| s3 | 0.0 everywhere | −0.0025 to 0.0627 | 2.475–2.513 | 1.081–1.089 |
| s4 | 0.0 everywhere | −0.0032 to 0.0662 | 2.966–2.987 | 1.342–1.358 |

`mean_cos` is noise-level (near 0, the value a random operator would score)
at every one of the 24 (seed × hop) readings — none of these are partial
or near-miss transitions. Training-log `cosine_loss`, sampled every 25,000
steps, confirms no downward trend for any of the 3 seeds through the full
budget: s1 stays in `[0.992, 1.002]` from step 10,000 to 110,000; s3 drifts
mildly from 1.0045 (10K) to 0.9771 (110K); s4 drifts mildly from 0.9970
(10K) to 0.9844 (110K) — real but small movement, nowhere near the sharp
late collapse (loss dropping to ≈0.01–0.001 within a few thousand steps)
that characterizes every seed that actually transitions elsewhere in this
document (§10's own K=12/K=16 table above, §2's K=8 rounds). Effective
rank is graded across the 3 seeds (s1 1.5 < s3 2.5 < s4 3.0, vs. the
dead-init floor of ≈1.0 and the target of 16) — consistent with s3/s4
being marginally "less stuck" than s1, but none within striking distance
of a transition signature (converged seeds elsewhere in this document jump
from ≈1 to ≈16 within a few thousand steps once they move at all).

**Combined with the 80K round's K=16 table (§10 above: s0 near-exact, s2
near-exact, s1 dead), K=16's total tally across every seed tested at any
budget (80K or 120K) is 2/5 converged (s0, s2) and 3/5 dead (s1, s3, s4).**
K=16=d transitions are genuinely rare at this operating point, not merely
slow.

**This is explicitly flagged as a boundary-case scaling observation, not a
fourth instance of the project's three-budget-artifacts pattern (§10's own
"three budget artifacts" section above, `EXPERIMENT_LOG.md`'s matching
`[LEARN]` entry): 1.5× more budget (80K→120K) converted zero additional
seeds this time**, in contrast to every prior instance of that pattern
(Task E K=8 20K→40K, Stage 0 d≥32 8K→100K, Task E K=12/K=16 40K→80K), each
of which rescued at least one previously-dead seed at the next budget tier.
The prior pattern's own stated caveat already anticipated this outcome
("re-testing at 2–2.5× is a necessary check, not a guarantee... more budget
does not always rescue a result" — echoing Stage 0's own 100K-probe
plateau). `K=16=d` is the one cell in this document where the entity
subspace *is* the full ambient space (§10's Z-dump addendum above, item
(d)) — there is no orthogonal complement, and correspondingly no "slack"
capacity for a marginal write to grow into before the exact solution is
required; whether that structural fact is *why* K=16 transitions less
reliably than K=8/K=12 is not established by this round (n=5 total seeds
across two budgets is not enough to separate "K=16 is intrinsically harder
to transition" from "these particular 3 dead seeds have unlucky
initializations"), but the boundary-case framing (`K=d`, not `K<d`) is the
most concrete structural candidate available and is recorded here rather
than left as an unexplained residual.

**No further budget extension is planned for K=16 s1/s3/s4** absent a
specific reason to expect a fourth budget tier would behave differently
from the third — this is a genuine, if narrow, negative result for this
one cell, not a wall to be re-tested indefinitely.

Reproducibility: `experiment-runs/2026-07-02_task_e_120k_k16/` (3 result
JSONs + training logs; `ALL_DONE` sentinel present); source config
`matrix-thinking/chapter2/run_task_e.py` (same code as the 80K round,
`--steps 120000`, seeds `{1,3,4}`).
