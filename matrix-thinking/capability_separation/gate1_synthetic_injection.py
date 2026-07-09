"""CAPABILITY_SEPARATION_DESIGN.md S1.7 gate 1(b) -- the synthetic-injection
acceptance test, run BEFORE any REAL trained checkpoint's degauging output is
trusted (gate 1 duty (c), which requires a real GPU-trained model and is out
of BUILD scope here). This script exercises the PRODUCTION harness's degauge
step end-to-end (query-coverage guard -> fit/eval split with diversity-floor
retry -> Procrustes/scale degauging -> disjoint-eval scoring) on a KNOWN
(Q_true, c_true)-conjugated S4 trajectory, catching wiring/shape/dtype bugs a
standalone numpy re-run of verify_option_a_readout.py cannot (S1.7 gate 1).

Per S1.7's exact spec: the injected values are §1.3.2's OWN ground truth
(Z_synth(w) = c_true * Q_true @ rho_G(w) @ Q_true.T + noise(std=0.03)) at
rho_G's OWN dimension (d_min=3 for S4) -- this tests the DEGAUGING step in
isolation (fit_scale/fit_orthogonal_intertwiner/score_eval, applied to
whatever the "restricted operator" A(w) turns out to be), NOT the
SVD-subspace-derivation step (entity_subspace_from_words), which requires a
real ambient-dimension Z from a trained checkpoint and is gate 1 duty (c),
not (b) -- disclosed explicitly, not silently conflated.

"Written into the harness's real on-disk dump format/shapes/dtypes" (S1.7's
exact wording): the injected trajectory is WRITTEN to and READ BACK from an
on-disk .npz dump (the same array names/shapes/dtypes a real training run's
held-out word dump would use), round-tripping through disk before being fed
to the production degauging code -- this is what actually exercises
wiring/shape/dtype bugs a pure in-memory call cannot.

The words themselves (rho_G(w) targets) are drawn via group_task.py's REAL
query-coverage guard + sample_eval_words (S4, N=50, with repeats -- a random
walk revisits S4's 24 elements many times over 50 words of length 9-16),
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
from groups import D_MIN
from group_task import (
    check_coverage_with_retry, sample_eval_words, COVERAGE_BAR, N_EVAL_WORDS,
)
from readout import _split_with_diversity_retry, degauge_and_score

RNG_SEED = 20260712     # distinct from every seed already pinned elsewhere in this repo
NOISE_STD = 0.03        # S1.3.2's exact noisy-condition noise level
C_TRUE = 1.7             # S1.3.2's exact ground-truth scale
GROUP = "S4"
BASE_CELL_SEED = 7001    # a plausible real cell seed (exercises the retry-once machinery's
                         # normal, first-try-passes path -- S4 clears its bar comfortably, S1.3.3)


def _make_Q_true(d: int, rng: np.random.Generator) -> np.ndarray:
    """S1.3.2's exact construction: Haar-uniform orthogonal via QR +
    Mezzadri (2007) sign correction."""
    X = rng.standard_normal((d, d))
    Qraw, Rraw = np.linalg.qr(X)
    return Qraw * np.sign(np.diag(Rraw))


def _write_zdump(path: str, Z_list: list, rho_list: list, group: str, condition: str) -> None:
    """Write a .npz dump matching a real held-out-word Z-dump's array
    names/shapes/dtypes (float64, (N,d,d)) -- the on-disk round-trip S1.7's
    wording requires."""
    Z_arr = np.stack(Z_list).astype(np.float64)
    rho_arr = np.stack(rho_list).astype(np.float64)
    np.savez(path, Z=Z_arr, rho=rho_arr, group=group, condition=condition)


def _read_zdump(path: str):
    d = np.load(path, allow_pickle=False)
    return list(d["Z"]), list(d["rho"]), str(d["group"]), str(d["condition"])


def run_condition(label: str, Q: np.ndarray, rho_list: list, fit_idx: list, eval_idx: list,
                  rng: np.random.Generator, d_min: int, dump_dir: str) -> dict:
    Z_list = [C_TRUE * (Q @ rho @ Q.T) + rng.normal(0, NOISE_STD, (d_min, d_min)) for rho in rho_list]

    # on-disk round-trip (S1.7's "written into the harness's real on-disk
    # dump format/shapes/dtypes" requirement).
    dump_path = os.path.join(dump_dir, f"zdump_{label.split(':')[0].strip()}.npz")
    _write_zdump(dump_path, Z_list, rho_list, GROUP, label)
    Z_read, rho_read, group_read, cond_read = _read_zdump(dump_path)
    assert group_read == GROUP and cond_read == label, "on-disk dump round-trip metadata mismatch"
    assert len(Z_read) == len(rho_read) == len(rho_list), "on-disk dump round-trip length mismatch"
    for a, b in zip(Z_read, Z_list):
        assert np.allclose(a, b), "on-disk dump round-trip VALUE mismatch (shape/dtype bug)"

    A_fit = [Z_read[i] for i in fit_idx]
    A_eval = [Z_read[i] for i in eval_idx]
    rho_fit = [rho_read[i] for i in fit_idx]
    rho_eval = [rho_read[i] for i in eval_idx]

    scores = degauge_and_score(A_fit, A_eval, rho_fit, rho_eval, d_min)
    print(f"\n=== {label} ===")
    print(f"  on-disk round-trip: {dump_path}  (Z/rho shapes/dtypes/values verified identical)")
    print(f"  EVAL SET (n={len(A_eval)}, never used to fit c_hat/Q_hat): "
          f"mean_cos={scores['mean_cos']:.6f}  mean_rel_err={scores['mean_rel_err']:.6f}  "
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

    Q_true = _make_Q_true(d_min, rng)
    print(f"\n[ground truth] c_true={C_TRUE}, Q_true orthogonal "
          f"(||QQ^T-I||={np.linalg.norm(Q_true @ Q_true.T - np.eye(d_min)):.2e}), "
          f"det(Q_true)={np.linalg.det(Q_true):.4f}")

    with tempfile.TemporaryDirectory() as dump_dir:
        honest = run_condition("honest: exact conjugation + noise(std=0.03)", Q_true, rho_list,
                               fit_idx, eval_idx, rng, d_min, dump_dir)

        Q_def = Q_true.copy()
        Q_def[-1, :] = 0.0
        rank_qdef = np.linalg.matrix_rank(Q_def)
        print(f"\n[negative control] rank(Q_def)={rank_qdef} (of {d_min}) -- Q_true's last row zeroed")
        corrupt = run_condition("corrupt: rank-deficient Q_def (NEGATIVE CONTROL)", Q_def, rho_list,
                                fit_idx, eval_idx, rng, d_min, dump_dir)

    print("\n" + "=" * 100)
    print("GATE 1(b) ACCEPTANCE CHECKS")
    print("=" * 100)
    print(f"  (1) recovery:            mean_cos={honest['mean_cos']:.6f} > 0.95 ?  "
          f"{'PASS' if honest['mean_cos'] > 0.95 else 'FAIL'}")
    print(f"      recovered_frac@0.9={honest['recovered_frac_90']:.4f} >= 0.9 ?  "
          f"{'PASS' if honest['recovered_frac_90'] >= 0.9 else 'FAIL'}")
    gap = honest["mean_cos"] - corrupt["mean_cos"]
    print(f"  (2) negative NOT rescued: mean_cos gap = {gap:.6f} >= 0.3 ?  "
          f"{'PASS' if gap >= 0.3 else 'FAIL'}")
    print(f"      corrupt recovered_frac@0.9={corrupt['recovered_frac_90']:.4f} < 0.5 ?  "
          f"{'PASS' if corrupt['recovered_frac_90'] < 0.5 else 'FAIL'}")

    assert honest["mean_cos"] > 0.95, f"gate 1(b) FAIL: honest mean_cos {honest['mean_cos']:.4f} <= 0.95"
    assert honest["recovered_frac_90"] >= 0.9, \
        f"gate 1(b) FAIL: honest recovered_frac@0.9 {honest['recovered_frac_90']:.4f} < 0.9"
    assert gap >= 0.3, f"gate 1(b) FAIL: rank-deficient corruption RESCUED (gap {gap:.4f} < 0.3)"
    assert corrupt["recovered_frac_90"] < 0.5, \
        f"gate 1(b) FAIL: rank-deficient corruption RESCUED (recovered_frac@0.9 {corrupt['recovered_frac_90']:.4f} >= 0.5)"

    print("\n" + "=" * 100)
    print("GATE 1(b) VERDICT: ALL ACCEPTANCE CHECKS PASSED. The production degauging harness")
    print("(query-coverage guard -> fit/eval split w/ diversity-floor retry -> on-disk dump ")
    print("round-trip -> Procrustes/scale degauging -> disjoint-eval scoring) correctly recovers")
    print("a known gauge under noise and correctly FAILS to rescue a rank-deficient corruption,")
    print("wired through the REAL production code path, not a standalone re-run of the toy script.")
    print("=" * 100)


if __name__ == "__main__":
    main()
