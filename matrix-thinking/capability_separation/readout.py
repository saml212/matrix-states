"""CAPABILITY_SEPARATION_DESIGN.md S1.3.2/S1.4.1 -- the production readout
harness: block-diagonal rho_G target embedding (Option A, groups.py),
continuous Frobenius/cosine scoring (never argmax), and the Procrustes/scale
degauging pipeline, PORTED from verify_option_a_readout.py's validated
toy pipeline -- `fit_scale`/`fit_orthogonal_intertwiner`/`score_eval` are
imported DIRECTLY (S1.11: "verify_option_a_readout.py's functions imported
directly into the eval harness, not re-implemented"), not rewritten.

Also implements S1.4.1's subspace-restriction analysis, generalizing
analyze_zdump.py's `entity_subspace`/`block_decompose` precedent to a task
with no single fixed target operator (S1.4.1's own framing: "this design
generalizes the precedent rather than reusing it literally"):

  1. dump Z(w) for a coverage-guarded N=50 held-out word sample per group.
  2. derive the model's OWN dominant d_min(G)-dim subspace U via SVD of the
     empirical covariance sum_w Z(w) Z(w)^T (data-driven, non-circular).
  3. restrict A(w) = U^T Z(w) U.
  4. 60/40 fit/eval split (index-disjoint), with fit-set/eval-set diversity
     floors enforced via group_task.py's retry-once machinery (a RESHUFFLE
     of the already-drawn N=50 sample, distinct from the top-level
     query-coverage guard's fresh-redraw retry -- S1.4.1 step 4: "redraw the
     fit/eval split from the same N=50 sample").
  5. degauge + score on the DISJOINT eval subset; also report whole-matrix
     effective rank (Task D/E convention), flagged non-decisive by default
     (S1.4.1 step 5).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chapter2"))

from verify_option_a_readout import fit_scale, fit_orthogonal_intertwiner, score_eval
from groups import D_MIN, rho_G_embedded
from group_task import (
    GROUP_NAMES, sample_eval_words, check_coverage_with_retry, CoverageGuardError,
    COVERAGE_BAR, FIT_FLOOR, EVAL_FLOOR, N_EVAL_WORDS, N_FIT, N_EVAL_SPLIT, _distinct_count,
)
from rank_utils import effective_rank as torch_effective_rank, stable_rank as torch_stable_rank


# ---------------------------------------------------------------------------
# Subspace-restriction (S1.4.1 steps 1-3)
# ---------------------------------------------------------------------------

def entity_subspace_from_words(Z_words: np.ndarray, d_min: int) -> np.ndarray:
    """S1.4.1 step 2: the model's OWN dominant d_min-dim subspace U of
    d_state x d_state, via SVD of the empirical covariance sum_w Z(w)Z(w)^T
    over a held-out word sample -- data-driven, NOT derived from rho_G
    (non-circular). Z_words: (N, d_state, d_state). Returns U: (d_state, d_min)."""
    d_state = Z_words.shape[-1]
    cov = np.zeros((d_state, d_state))
    for Z in Z_words:
        cov += Z @ Z.T
    U_full, _, _ = np.linalg.svd(cov)
    return U_full[:, :d_min]


def restrict(Z: np.ndarray, U: np.ndarray) -> np.ndarray:
    """S1.4.1 step 3: A(w) = U^T Z(w) U."""
    return U.T @ Z @ U


def dump_Z(model, gens_words_idx, device, force_rank_k=None) -> np.ndarray:
    """Run model.encode on a list of (L,)-shaped generator-index int arrays
    (per-SAMPLE-varying L -- forwarded one word at a time, since
    GroupWordEncoder's positional embedding needs one consistent L per
    forward call; batched TRAINING data uses per-batch-fixed-L instead,
    group_task.sample_train_batch). Returns (N, d_state, d_state) numpy."""
    model.eval()
    outs = []
    with torch.no_grad():
        for idx in gens_words_idx:
            t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device).unsqueeze(0)
            Z = model.encode(t, force_rank_k=force_rank_k)
            outs.append(Z.squeeze(0).detach().cpu().numpy())
    return np.stack(outs)


# ---------------------------------------------------------------------------
# Degauging + scoring (S1.3.2, ported not reimplemented)
# ---------------------------------------------------------------------------

def degauge_and_score(A_fit: list, A_eval: list, rho_fit: list, rho_eval: list, d_min: int) -> dict:
    """S1.3.2's pipeline, applied to the RESTRICTED operator A(w) (S1.4.1).
    fit_scale/fit_orthogonal_intertwiner/score_eval are the LITERAL
    verify_option_a_readout.py functions -- no reimplementation."""
    c_hat, ratios = fit_scale(A_fit, rho_fit)
    Q_hat, sv_min, sv_2nd = fit_orthogonal_intertwiner(A_fit, rho_fit, c_hat, d_min)
    coses, rel_errs = score_eval(A_eval, rho_eval, Q_hat, c_hat)
    rec90 = float(np.mean([c > 0.9 for c in coses]))
    return dict(c_hat=c_hat, Q_hat=Q_hat, sv_min=sv_min, sv_2nd=sv_2nd,
                mean_cos=float(np.mean(coses)), mean_rel_err=float(np.mean(rel_errs)),
                recovered_frac_90=rec90, coses=coses, rel_errs=rel_errs)


def _split_with_diversity_retry(name: str, prod_list: list, seed: int):
    """S1.4.1 step 4's fit/eval diversity floors, retry-once-then-fail via a
    RESHUFFLE of the already-drawn N=50 sample (distinct from the top-level
    query-coverage guard's fresh-external-redraw retry). Returns
    (fit_idx, eval_idx), index-disjoint."""
    n = len(prod_list)
    fit_bar, eval_bar = FIT_FLOOR[name], EVAL_FLOOR[name]
    log = {"group": name, "fit_bar": fit_bar, "eval_bar": eval_bar, "attempts": []}
    for attempt, shuffle_seed in enumerate((seed, seed + 1)):
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(n)
        fit_idx = perm[:N_FIT].tolist()
        eval_idx = perm[N_FIT:N_FIT + N_EVAL_SPLIT].tolist()
        fit_count = _distinct_count([prod_list[i] for i in fit_idx])
        eval_count = _distinct_count([prod_list[i] for i in eval_idx])
        passed = fit_count >= fit_bar and eval_count >= eval_bar
        log["attempts"].append({"attempt": attempt, "shuffle_seed": shuffle_seed,
                                "fit_count": fit_count, "eval_count": eval_count, "passed": passed})
        if passed:
            log["result"] = "pass"
            return fit_idx, eval_idx, log
    log["result"] = "hard_fail"
    raise CoverageGuardError(
        f"{name}: fit/eval diversity floor HARD-FAILED after 2 reshuffle attempts. Log: {log}"
    )


# ---------------------------------------------------------------------------
# Full per-cell pipeline (S1.4.1 steps 1-5)
# ---------------------------------------------------------------------------

def run_subspace_restriction_pipeline(model, name: str, base_seed: int, device,
                                      force_rank_k: int | None = None,
                                      n_words: int = N_EVAL_WORDS) -> dict:
    """One cell's full S1.4.1 pipeline. Returns a dict with the degauged
    eval-set scores (the Stage-1 decision metric, M1/M3), the whole-matrix
    effective/stable rank (non-decisive safety net, step 5), and the
    coverage/diversity-floor logs (for the harvest's provenance trail)."""
    d_min = D_MIN[name]

    cov_log = check_coverage_with_retry(name, base_seed, bar=COVERAGE_BAR[name], n_words=n_words,
                                        label="query-coverage", offset=10_000)
    seed = cov_log["final_seed"]
    idx_list, rho_list = sample_eval_words(name, seed, n_words)

    Z_words = dump_Z(model, idx_list, device, force_rank_k=force_rank_k)
    U = entity_subspace_from_words(Z_words, d_min)
    A_words = np.stack([restrict(Z, U) for Z in Z_words])

    fit_idx, eval_idx, split_log = _split_with_diversity_retry(name, rho_list, seed=seed + 20_000)
    assert set(fit_idx).isdisjoint(eval_idx), "fit/eval split not disjoint"

    A_fit = [A_words[i] for i in fit_idx]
    A_eval = [A_words[i] for i in eval_idx]
    rho_fit = [rho_list[i] for i in fit_idx]
    rho_eval = [rho_list[i] for i in eval_idx]

    scores = degauge_and_score(A_fit, A_eval, rho_fit, rho_eval, d_min)

    # restricted effective/stable rank on the disjoint eval subset (torch
    # rank_utils applied per-matrix, numpy->torch bridge).
    A_eval_t = torch.tensor(np.stack(A_eval), dtype=torch.float32)
    Z_eval_t = torch.tensor(Z_words[eval_idx], dtype=torch.float32)
    scores["restricted_effective_rank"] = torch_effective_rank(A_eval_t).mean().item()
    scores["restricted_stable_rank"] = torch_stable_rank(A_eval_t).mean().item()
    # step 5: whole-matrix rank too, non-decisive safety net.
    scores["whole_matrix_effective_rank"] = torch_effective_rank(Z_eval_t).mean().item()
    scores["whole_matrix_stable_rank"] = torch_stable_rank(Z_eval_t).mean().item()

    scores["coverage_log"] = cov_log
    scores["split_log"] = split_log
    scores["n_fit"], scores["n_eval"] = len(fit_idx), len(eval_idx)
    scores["group"] = name
    scores["force_rank_k"] = force_rank_k
    return scores


# ---------------------------------------------------------------------------
# NEGATIVE TEST -- the fit/eval diversity-floor retry (S1.4.1 step 4's
# "redraw the fit/eval split from the same N=50 sample" rule) is a
# DIFFERENT retry mechanism from group_task.py's top-level query-coverage
# guard (which redraws FRESH words from a shifted seed) -- it RESHUFFLES
# the already-drawn sample instead. CLAUDE.md's hard rule ("never trust a
# 'proves the check has teeth' test without running it to completion")
# applies to this mechanism too, distinctly from group_task.py's own
# undersampling/second-miss tests.
# ---------------------------------------------------------------------------

def _test_diversity_floor_second_miss_hard_fails():
    """NEGATIVE TEST: an artificially collapsed word sample (nearly every
    word resolves to the SAME group element, so no reshuffle can clear the
    fit/eval diversity floor) must hard-fail with CoverageGuardError after
    exactly 2 reshuffle attempts -- not silently pass, not retry a third
    time."""
    print("=" * 88)
    print("NEGATIVE TEST -- fit/eval diversity-floor retry HARD-FAILS on a collapsed sample")
    print("=" * 88)
    name = "A5"
    n = N_FIT + N_EVAL_SPLIT   # 50
    d_min = D_MIN[name]
    identity = np.eye(d_min)
    # 49 identical (identity) elements + 1 distinct element -- far below
    # BOTH the fit floor (S1.4.1: min(3*d_min, floor(0.8*|G|))=9 for A5)
    # and the eval floor (min(2*d_min,floor(0.6*|G|))=6 for A5) regardless
    # of which 30/20 reshuffle split is drawn.
    collapsed = [identity.copy() for _ in range(n - 1)]
    other = identity.copy()
    other[0, 0] = -1.0   # a single genuinely-different "element" (not a real group element,
                         # doesn't need to be -- _distinct_count only cares about matrix identity)
    collapsed.append(other)

    raised = False
    try:
        _split_with_diversity_retry(name, collapsed, seed=555_000)
    except CoverageGuardError as e:
        raised = True
        print(f"  CoverageGuardError raised as expected:\n    {str(e)[:260]}...")
    assert raised, "collapsed sample did NOT hard-fail the diversity-floor retry (no teeth)"

    # sanity: a HEALTHY real sample (drawn fresh) clears the SAME check normally.
    idx_list, rho_list = sample_eval_words(name, seed=555_001, n_words=n)
    fit_idx, eval_idx, log = _split_with_diversity_retry(name, rho_list, seed=555_002)
    assert log["result"] == "pass"
    print(f"  post-check sanity: a healthy real A5 sample clears the SAME diversity-floor "
          f"check normally (n_fit={len(fit_idx)}, n_eval={len(eval_idx)})")
    print("\nRESULT: fit/eval diversity-floor retry correctly HARD-FAILS on a collapsed sample "
          "after 2 reshuffle attempts, and passes normally on healthy data.\n")
    return raised


if __name__ == "__main__":
    _test_diversity_floor_second_miss_hard_fails()
    print("readout.py is otherwise a library module (imported by gate1_synthetic_injection.py, "
          "force_rank_arms.py, run_capability_sep.py). Run those scripts to exercise the full "
          "S1.4.1 pipeline end-to-end.")
