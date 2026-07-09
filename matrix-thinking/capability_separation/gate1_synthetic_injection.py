"""CAPABILITY_SEPARATION_DESIGN.md S1.7 gate 1(b) -- the synthetic-injection
acceptance test, run BEFORE any REAL trained checkpoint's degauging output is
trusted (gate 1 duty (c), which requires a real GPU-trained model and is out
of BUILD scope here). This script exercises the PRODUCTION harness's FULL
degauge pipeline end-to-end (query-coverage guard -> fit/eval split with
diversity-floor retry -> AMBIENT SVD-subspace derivation -> restriction ->
Procrustes/scale degauging -> disjoint-eval scoring) on a KNOWN (Q_true,
c_true)-conjugated S4 trajectory, catching wiring/shape/dtype bugs a
standalone numpy re-run of verify_option_a_readout.py cannot (S1.7 gate 1).

AMBIENT injection (S1.25 pinned item 1 / S1.7 gate 1(b), Rev 5 -- closes
DEFECT 1's exact blind spot). Rev 0-4's injection constructed Z_synth
directly at rho_G's OWN d_min dimension, exercising ONLY the degauging step
(fit_scale/fit_orthogonal_intertwiner/score_eval) in isolation --
`entity_subspace_from_words` (the SVD-subspace-derivation step,
S1.4.1 step 2) was never exercised by gate 1(b) at all, so the
mathematically-degenerate UNCENTERED covariance there went undetected while
a perfect real model's degauged output read mean_cos=-0.02..0.17 in
production (S1.25's diagnosis). Fixed here: the injected trajectory is now
constructed at AMBIENT `d_state = d_min+2`, block-embedded per Option A's
own target shape --

    Z_synth(w) = c_true * Q_true @ rho_G_embedded(w) @ Q_true.T + noise(std=0.03)

with `Q_true` an ambient-`d_state`-orthogonal matrix (not a bare d_min one)
and `rho_G_embedded` (S1.4, groups.py) in place of bare `rho_G` -- so the
acceptance test now runs the FULL production pipeline end-to-end:
`entity_subspace_from_words` (the CENTERED-covariance SVD, S1.4.1 step 2's
Rev-5 fix, readout.py) -> `restrict` -> degauge -> score, on the SAME
on-disk dump round-trip discipline this script already had.

NEGATIVE CONTROL (required alongside the positive acceptance run, per this
dispatch's build item (c)): a LOCAL, uncentered reproduction of the
pre-Rev-5 covariance statistic is run through the SAME honest ambient
Z-dump, to prove that centering is load-bearing IN THIS HARNESS, not just
asserted from S1.25's own prose -- the uncentered path must FAIL the same
acceptance bars the centered (production) path PASSES, on IDENTICAL input
data. Both paths run to completion; neither is skipped.

"Written into the harness's real on-disk dump format/shapes/dtypes" (S1.7's
exact wording): the injected trajectory is WRITTEN to and READ BACK from an
on-disk .npz dump (the same array names/shapes/dtypes a real training run's
held-out word dump would use), round-tripping through disk before being fed
to the production degauging code -- this is what actually exercises
wiring/shape/dtype bugs a pure in-memory call cannot.

The words themselves (rho_G(w) targets) are drawn via group_task.py's REAL
query-coverage guard + sample_eval_words (S4, N=50, with repeats -- a random
walk revisits S4's 24 elements many times over 50 words of length 1-8,
S1.25/S1.3.5's re-pinned TRAIN-support regime, group_task.py's new default),
NOT a bare one-shot enumeration of the 24 group elements -- this is what
"the production harness" actually draws at eval time, distinct from
verify_option_a_readout.py's own toy setup (14 fit / 10 eval, one-shot
enumeration of all 24 elements, no repeats) which this design's Rev 1
explicitly superseded with the pinned 60/40-of-N=50 split (S1.4.1 step 4).

Run: DRY_RUN_BYPASS=1 .venv/bin/python gate1_synthetic_injection.py
CPU-only, numpy + stdlib (no GPU needed, per S1.6's cost table:
"Degauging-pipeline validation on a REAL trained checkpoint (S1.7 gate 1) |
CPU-only, numpy | ~0.0" -- this script is the (b) half of that gate).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groups import D_MIN, D_STATE, rho_G_embedded
from group_task import (
    check_coverage_with_retry, sample_eval_words, COVERAGE_BAR, N_EVAL_WORDS,
)
from readout import _split_with_diversity_retry, degauge_and_score, entity_subspace_from_words, restrict

RNG_SEED = 20260712     # distinct from every seed already pinned elsewhere in this repo
NOISE_STD = 0.03        # S1.3.2's exact noisy-condition noise level
C_TRUE = 1.7             # S1.3.2's exact ground-truth scale
GROUP = "S4"
BASE_CELL_SEED = 7001    # a plausible real cell seed (exercises the retry-once machinery's
                         # normal, first-try-passes path -- S4 clears its bar comfortably, S1.3.3)


def _entity_subspace_from_words_UNCENTERED(Z_words: np.ndarray, d_min: int) -> np.ndarray:
    """S1.25 DEFECT 1's exact PRE-Rev-5 statistic, reproduced HERE ONLY for
    this script's own negative-control assertion (never used for production
    scoring -- readout.entity_subspace_from_words is always centered). SVD
    of the UNCENTERED empirical covariance sum_w Z(w)Z(w)^T -- for Option
    A's orthogonal targets this is provably isotropic (~= c^2 * I_{d_state}),
    carrying zero subspace information, and empirically lets the constant
    identity-complement block outrank the rho-block's 2 weakest directions.
    See readout.entity_subspace_from_words' own docstring for the full
    load-bearing argument this negatively controls for."""
    d_state = Z_words.shape[-1]
    cov = np.zeros((d_state, d_state))
    for Z in Z_words:
        cov += Z @ Z.T
    U_full, _, _ = np.linalg.svd(cov)
    return U_full[:, :d_min]


def _make_Q_true(d: int, rng: np.random.Generator) -> np.ndarray:
    """S1.3.2's exact construction: Haar-uniform orthogonal via QR +
    Mezzadri (2007) sign correction."""
    X = rng.standard_normal((d, d))
    Qraw, Rraw = np.linalg.qr(X)
    return Qraw * np.sign(np.diag(Rraw))


def _write_zdump(path: str, Z_list: list, rho_list: list, group: str, condition: str) -> None:
    """Write a .npz dump matching a real held-out-word Z-dump's array
    names/shapes/dtypes (float64, (N,d_state,d_state) for Z -- AMBIENT
    scope, Rev 5; (N,d_min,d_min) for rho) -- the on-disk round-trip S1.7's
    wording requires."""
    Z_arr = np.stack(Z_list).astype(np.float64)
    rho_arr = np.stack(rho_list).astype(np.float64)
    np.savez(path, Z=Z_arr, rho=rho_arr, group=group, condition=condition)


def _read_zdump(path: str):
    d = np.load(path, allow_pickle=False)
    return list(d["Z"]), list(d["rho"]), str(d["group"]), str(d["condition"])


def run_condition(label: str, Q: np.ndarray, rho_list: list, fit_idx: list, eval_idx: list,
                  rng: np.random.Generator, d_min: int, d_state: int, dump_dir: str,
                  q_is_ambient: bool = True) -> dict:
    """AMBIENT injection (S1.25 pinned item 1, Rev 5): Z_synth(w) = c_true *
    Q @ rho_G_embedded(w, d_state) @ Q.T + noise, Q ambient d_state x d_state
    orthogonal -- runs the FULL production pipeline (entity_subspace_from_words
    -> restrict -> degauge_and_score), not just the degauging step.

    `q_is_ambient` (build item (c) design note): the HONEST condition's Q is
    genuinely ambient-d_state-orthogonal (S1.7 gate 1(b)'s own wording).
    The CORRUPT negative control's Q_def, per the SAME gate 1(b) text, is
    "S1.3.2's Q_def (rank-2-of-3) corruption" -- S1.3.2's ORIGINAL object,
    at d_min scale, NOT an ambient rank-(d_state-1)-of-d_state matrix (a
    rank-(d_state-1) ambient corruption was tried and found to NOT reliably
    survive restriction -- the missing ambient direction can fall outside
    the SVD-derived d_min-dim subspace, letting Procrustes rescue it; this
    would test a different, not-intended failure mode of the SUBSPACE step,
    not the DEGAUGING step this negative control targets, S1.3.2's own Part
    2). `q_is_ambient=False` instead conjugates `rho(w)` DIRECTLY at d_min
    scale (Q here is d_min x d_min) BEFORE block-embedding -- the identity
    complement stays intact, so entity_subspace_from_words still finds the
    correct rho-supported subspace (as it would for a healthy model), and
    the rank deficiency lands exactly where S1.3.2 always intended it: in
    the restricted d_min x d_min operator the degauging step must fail to
    rescue."""
    if q_is_ambient:
        Z_list = [C_TRUE * (Q @ rho_G_embedded(rho, d_state) @ Q.T) + rng.normal(0, NOISE_STD, (d_state, d_state))
                  for rho in rho_list]
    else:
        Z_list = [rho_G_embedded(C_TRUE * (Q @ rho @ Q.T), d_state) + rng.normal(0, NOISE_STD, (d_state, d_state))
                  for rho in rho_list]

    # on-disk round-trip (S1.7's "written into the harness's real on-disk
    # dump format/shapes/dtypes" requirement).
    dump_path = os.path.join(dump_dir, f"zdump_{label.split(':')[0].strip()}.npz")
    _write_zdump(dump_path, Z_list, rho_list, GROUP, label)
    Z_read, rho_read, group_read, cond_read = _read_zdump(dump_path)
    assert group_read == GROUP and cond_read == label, "on-disk dump round-trip metadata mismatch"
    assert len(Z_read) == len(rho_read) == len(rho_list), "on-disk dump round-trip length mismatch"
    for a, b in zip(Z_read, Z_list):
        assert np.allclose(a, b), "on-disk dump round-trip VALUE mismatch (shape/dtype bug)"

    # FULL production pipeline (S1.4.1 steps 2-3): CENTERED ambient SVD ->
    # restriction to d_min, THEN fit/eval split + degauging, mirroring
    # readout.run_subspace_restriction_pipeline's own step order exactly.
    Z_read_arr = np.stack(Z_read)
    U = entity_subspace_from_words(Z_read_arr, d_min)
    A_words = np.stack([restrict(Z, U) for Z in Z_read_arr])

    A_fit = [A_words[i] for i in fit_idx]
    A_eval = [A_words[i] for i in eval_idx]
    rho_fit = [rho_read[i] for i in fit_idx]
    rho_eval = [rho_read[i] for i in eval_idx]

    scores = degauge_and_score(A_fit, A_eval, rho_fit, rho_eval, d_min)
    print(f"\n=== {label} ===")
    print(f"  on-disk round-trip: {dump_path}  (Z ambient d_state={d_state}, rho d_min={d_min}, "
          f"shapes/dtypes/values verified identical)")
    print(f"  entity_subspace_from_words (CENTERED, production) -> restrict -> degauge -> score")
    print(f"  EVAL SET (n={len(A_eval)}, never used to fit c_hat/Q_hat): "
          f"mean_cos={scores['mean_cos']:.6f}  mean_rel_err={scores['mean_rel_err']:.6f}  "
          f"recovered_frac@0.9={scores['recovered_frac_90']:.4f}  "
          f"crosscheck(full-Q) mean_cos={scores['crosscheck_mean_cos']:.6f}")
    return scores, Z_read_arr


def run_uncentered_negative_control(label: str, Z_read_arr: np.ndarray, rho_read: list,
                                    fit_idx: list, eval_idx: list, d_min: int) -> dict:
    """S1.25 DEFECT 1 negative control (this dispatch's build item (c)):
    re-derive U from the IDENTICAL honest ambient Z-dump via the OLD,
    UNCENTERED covariance statistic, restrict, degauge, score -- must FAIL
    the SAME acceptance bars the centered (production) path PASSES, on
    IDENTICAL input data. Proves centering is load-bearing IN THIS HARNESS,
    not just asserted from S1.25's own prose."""
    U_uncentered = _entity_subspace_from_words_UNCENTERED(Z_read_arr, d_min)
    A_words_u = np.stack([restrict(Z, U_uncentered) for Z in Z_read_arr])
    A_fit_u = [A_words_u[i] for i in fit_idx]
    A_eval_u = [A_words_u[i] for i in eval_idx]
    rho_fit = [rho_read[i] for i in fit_idx]
    rho_eval = [rho_read[i] for i in eval_idx]
    scores = degauge_and_score(A_fit_u, A_eval_u, rho_fit, rho_eval, d_min)
    print(f"\n=== {label} (NEGATIVE CONTROL -- OLD uncentered entity_subspace_from_words) ===")
    print(f"  same honest ambient Z-dump, uncentered SVD -> restrict -> degauge -> score")
    print(f"  EVAL SET (n={len(A_eval_u)}): mean_cos={scores['mean_cos']:.6f}  "
          f"mean_rel_err={scores['mean_rel_err']:.6f}  "
          f"recovered_frac@0.9={scores['recovered_frac_90']:.4f}")
    return scores


def main():
    print("=" * 100)
    print("GATE 1(b) -- synthetic-injection acceptance test on the PRODUCTION harness (S4)")
    print("=" * 100)
    rng = np.random.default_rng(RNG_SEED)
    d_min = D_MIN[GROUP]

    print(f"\n[step 1] query-coverage guard (REAL, group_task.check_coverage_with_retry, "
          f"S4, bar={COVERAGE_BAR[GROUP]}, N={N_EVAL_WORDS}):")
    cov_log = check_coverage_with_retry(GROUP, BASE_CELL_SEED, bar=COVERAGE_BAR[GROUP],
                                        n_words=N_EVAL_WORDS, label="query-coverage", offset=10_000)
    print(f"  result={cov_log['result']}  attempts={cov_log['attempts']}")
    assert cov_log["result"] == "pass"

    print(f"\n[step 2] draw the REAL N={N_EVAL_WORDS} held-out S4 word sample "
          f"(seed={cov_log['final_seed']}):")
    idx_list, rho_list = sample_eval_words(GROUP, cov_log["final_seed"], N_EVAL_WORDS)
    print(f"  drew {len(rho_list)} words; distinct S4 elements reached = "
          f"{len({tuple(np.round(m,6).flatten()) for m in rho_list})} (of |S4|=24)")

    print(f"\n[step 3] REAL fit/eval split with diversity-floor retry "
          f"(readout._split_with_diversity_retry):")
    fit_idx, eval_idx, split_log = _split_with_diversity_retry(GROUP, rho_list, seed=cov_log["final_seed"] + 20_000)
    print(f"  result={split_log['result']}  n_fit={len(fit_idx)}  n_eval={len(eval_idx)}  "
          f"disjoint={set(fit_idx).isdisjoint(eval_idx)}")
    assert split_log["result"] == "pass"

    d_state = D_STATE[GROUP]
    Q_true = _make_Q_true(d_state, rng)
    print(f"\n[ground truth] c_true={C_TRUE}, Q_true AMBIENT d_state={d_state} orthogonal "
          f"(||QQ^T-I||={np.linalg.norm(Q_true @ Q_true.T - np.eye(d_state)):.2e}), "
          f"det(Q_true)={np.linalg.det(Q_true):.4f}  (Rev 5: was d_min={d_min}-dimensional pre-Rev-5)")

    with tempfile.TemporaryDirectory() as dump_dir:
        honest, Z_honest_arr = run_condition("honest: exact conjugation + noise(std=0.03)", Q_true, rho_list,
                                             fit_idx, eval_idx, rng, d_min, d_state, dump_dir,
                                             q_is_ambient=True)

        # S1.3.2's OWN Q_def object -- d_min scale, last row zeroed (rank
        # d_min-1, "rank-2-of-3" for S4 exactly per S1.7 gate 1(b)'s
        # wording), conjugating rho(w) BEFORE block-embedding -- see
        # run_condition's own docstring for why this, not an ambient
        # rank-(d_state-1) corruption, is what this negative control targets.
        Q_true_dmin = _make_Q_true(d_min, rng)
        Q_def = Q_true_dmin.copy()
        Q_def[-1, :] = 0.0
        rank_qdef = np.linalg.matrix_rank(Q_def)
        print(f"\n[negative control] rank(Q_def)={rank_qdef} (of {d_min}) -- d_min-scale Q_true's last "
              f"row zeroed (S1.3.2's own construction), embedded into the ambient target with an "
              f"UNTOUCHED identity complement")
        corrupt, _ = run_condition("corrupt: rank-deficient Q_def (NEGATIVE CONTROL)", Q_def, rho_list,
                                   fit_idx, eval_idx, rng, d_min, d_state, dump_dir,
                                   q_is_ambient=False)

    # rho_read for the uncentered negative control below: the fit/eval split
    # indexes into the SAME rho_list drawn in step 2 (rho is never touched by
    # the ambient injection -- only Z gains the ambient/block-embedded shape).
    rho_read_for_uncentered = rho_list

    print(f"\n[negative control -- build item (c)] re-deriving U from the IDENTICAL honest "
          f"ambient Z-dump via the OLD, UNCENTERED covariance (S1.25 DEFECT 1's exact statistic):")
    uncentered = run_uncentered_negative_control(
        "honest data, UNCENTERED entity_subspace_from_words", Z_honest_arr, rho_read_for_uncentered,
        fit_idx, eval_idx, d_min,
    )

    # NOTE on WHICH score gate 1(b) grades (crosscheck_* = full-Q Procrustes,
    # not the new mean_cos/recovered_frac_90 PRIMARY fields): this synthetic
    # injection's Q_true is an ARBITRARY Haar-random ambient rotation (by
    # construction, to prove Procrustes recovers ANY gauge, S1.3.2's own
    # Part-2 intent) -- it is NOT close to I_{d_state}. Scale-only degauging
    # (Q_hat=I, this dispatch's new PRIMARY per S1.5/S1.25 pinned item 5)
    # is only a valid PRIMARY metric for REAL trained checkpoints, where
    # S1.25 separately measured Q_hat~=I empirically; grading THIS synthetic
    # test's arbitrary-Q injection against Q_hat=I would fail regardless of
    # whether centering is correct (confirmed below: crosscheck_mean_cos is
    # near-oracle while the scale-only mean_cos is not, on the SAME centered
    # run). Gate 1(b) therefore grades `crosscheck_*` (the full-Q fit) --
    # matching S1.25's own cited figures (0.9996/1.00 centered,
    # 0.711/0.15 uncentered), which were themselves full-Q numbers.
    print(f"\n[gate 1(b) scoring note] honest scale-only (PRIMARY, real-checkpoint metric) "
          f"mean_cos={honest['mean_cos']:.4f} vs honest crosscheck (full-Q, THIS test's metric) "
          f"mean_cos={honest['crosscheck_mean_cos']:.4f} -- the gap confirms Q_true is genuinely "
          f"non-identity here (as intended), so gate 1(b) grades crosscheck_*, not mean_cos.")

    print("\n" + "=" * 100)
    print("GATE 1(b) ACCEPTANCE CHECKS (graded on crosscheck_* / full-Q Procrustes -- see note above)")
    print("=" * 100)
    print(f"  (1) recovery:            mean_cos={honest['crosscheck_mean_cos']:.6f} > 0.95 ?  "
          f"{'PASS' if honest['crosscheck_mean_cos'] > 0.95 else 'FAIL'}")
    print(f"      recovered_frac@0.9={honest['crosscheck_recovered_frac_90']:.4f} >= 0.9 ?  "
          f"{'PASS' if honest['crosscheck_recovered_frac_90'] >= 0.9 else 'FAIL'}")
    gap = honest["crosscheck_mean_cos"] - corrupt["crosscheck_mean_cos"]
    print(f"  (2) negative NOT rescued: mean_cos gap = {gap:.6f} >= 0.3 ?  "
          f"{'PASS' if gap >= 0.3 else 'FAIL'}")
    print(f"      corrupt recovered_frac@0.9={corrupt['crosscheck_recovered_frac_90']:.4f} < 0.5 ?  "
          f"{'PASS' if corrupt['crosscheck_recovered_frac_90'] < 0.5 else 'FAIL'}")
    print(f"  (3) UNCENTERED negative control MUST FAIL the honest-recovery bar (S1.25 DEFECT 1, "
          f"build item (c)): uncentered mean_cos={uncentered['crosscheck_mean_cos']:.6f} <= 0.95 ?  "
          f"{'PASS (correctly fails)' if uncentered['crosscheck_mean_cos'] <= 0.95 else 'FAIL (defect not reproduced -- INVESTIGATE)'}")
    print(f"      centered-vs-uncentered gap on IDENTICAL data: "
          f"{honest['crosscheck_mean_cos'] - uncentered['crosscheck_mean_cos']:.6f} (S1.25 cites "
          f"0.9996-0.711=0.2886 on real-checkpoint-derived synthetic data; this instance's exact "
          f"gap will differ with the RNG draw, sign/direction is what's load-bearing)")

    assert honest["crosscheck_mean_cos"] > 0.95, \
        f"gate 1(b) FAIL: honest mean_cos {honest['crosscheck_mean_cos']:.4f} <= 0.95"
    assert honest["crosscheck_recovered_frac_90"] >= 0.9, \
        f"gate 1(b) FAIL: honest recovered_frac@0.9 {honest['crosscheck_recovered_frac_90']:.4f} < 0.9"
    assert gap >= 0.3, f"gate 1(b) FAIL: rank-deficient corruption RESCUED (gap {gap:.4f} < 0.3)"
    assert corrupt["crosscheck_recovered_frac_90"] < 0.5, (
        f"gate 1(b) FAIL: rank-deficient corruption RESCUED "
        f"(recovered_frac@0.9 {corrupt['crosscheck_recovered_frac_90']:.4f} >= 0.5)"
    )
    assert uncentered["crosscheck_mean_cos"] <= 0.95, (
        f"gate 1(b) build-item-(c) FAIL: the UNCENTERED entity_subspace_from_words negative "
        f"control was expected to FAIL the honest-recovery bar (S1.25 DEFECT 1) on IDENTICAL "
        f"ambient data the centered path passed, but scored "
        f"mean_cos={uncentered['crosscheck_mean_cos']:.4f} > 0.95 -- either the defect does not "
        f"reproduce in this harness's exact construction, or the centered fix is not actually "
        f"load-bearing here; investigate before trusting gate 1(b)."
    )
    assert honest["crosscheck_mean_cos"] > uncentered["crosscheck_mean_cos"], (
        "gate 1(b) build-item-(c) FAIL: centered (production) path did not outperform the "
        "uncentered negative control on identical data -- centering is not demonstrated load-bearing"
    )

    print("\n" + "=" * 100)
    print("GATE 1(b) VERDICT: ALL ACCEPTANCE CHECKS PASSED (incl. build item (c)'s uncentered")
    print("negative control). The production degauging harness (query-coverage guard -> fit/eval")
    print("split w/ diversity-floor retry -> on-disk AMBIENT dump round-trip -> CENTERED")
    print("entity_subspace_from_words -> restrict -> Procrustes/scale degauging -> disjoint-eval")
    print("scoring) correctly recovers a known gauge under noise, correctly FAILS to rescue a")
    print("rank-deficient corruption, AND correctly demonstrates the OLD uncentered subspace")
    print("derivation fails on the SAME data the centered fix passes -- wired through the REAL")
    print("production code path, not a standalone re-run of the toy script.")
    print("=" * 100)


if __name__ == "__main__":
    main()
