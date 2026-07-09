"""CAPABILITY_SEPARATION_DESIGN.md S2.3/S2.4/S2.6 (Rev 3, DESIGN-CLEARED-
FOR-BUILD per S2.18) -- the Stage-2 task harness: depth entry, the D_test
grid evaluation pipeline, per-depth query-coverage bars, the TARGET RANK
UNIT TEST (S1.33's [LEARN], see below), the prefix-state-fidelity
diagnostic, and the last-K-window / Arm-1 evaluation helpers.

Training regime (S2.3, reused verbatim): FINAL-STATE-ONLY supervision,
per-batch-fixed-D drawn from D~Uniform{1,D_TRAIN_MAX=8} -- this is BYTE-
IDENTICAL in form to Stage 1's own `group_task.sample_train_batch` at its
DEFAULT `l_hi=L_TRAIN_HI=8` (Stage 1's own pinned range), so this module
reuses that function directly (via `l_hi=D_TRAIN_MAX` for explicitness,
even though it's the existing default) rather than re-implementing an
identical sampler -- "Batching, reused verbatim... zero new padding/mask
code" (S2.3).

Per-depth eval-word sampling ALSO needs zero new code: `group_task.
sample_eval_words(name, seed, n_words, L_lo=D, L_hi=D)` already draws
`rng.integers(L_lo, L_hi+1)` per word, so pinning `L_lo=L_hi=D` degenerates
that RANGE sampler into an exact-fixed-depth sampler for free.

What IS new here: (1) per-fixed-depth query-coverage BARS (S2.4's
D-in-{4..64} grid uses a FIXED SINGLE DEPTH per calibration point, not
Stage 1's L-RANGE bars in `group_task.COVERAGE_BAR*`, so those tables don't
apply past D=16) -- computed by generalizing `coverage_calibration.py`'s
Monte Carlo (`sample_word_result`/`GROUPS`) to a fixed depth and re-using
`pick_bars` UNMODIFIED (imported, not reimplemented) on a single-row table;
(2) the D_test grid evaluation pipeline (batched, since every probe word in
a fixed-depth sample shares the SAME D -- simpler than Stage 1's per-word
loop, `readout.dump_Z`, which exists only because Stage 1's samples mix
lengths); (3) the TARGET RANK UNIT TEST; (4) the prefix-fidelity diagnostic;
(5) the Arm-1 checkpoint-reuse / L_max-ceiling handling.

TARGET RANK UNIT TEST -- the S1.33 [LEARN] this dispatch names explicitly.
S1.33 diagnosed Stage 1's own M3 arms as an undelivered causal test (not a
real negative) because `groups.py::rho_G_embedded`'s block-embedding is
`torch.eye(d_state)` overwritten in the top-left d_min x d_min block, so
the AS-BUILT target's rank is d_state (ALL singular values 1, an
ambient-identity capacity tax), not d_min -- capacity-capped arms bought
the free ambient-identity payoff first and never delivered a genuine
rank-vs-recovery test. Stage 2 reuses this EXACT target construction
verbatim (S2.3: "Target = rho_G(product(w))... reused VERBATIM"), AND its
own architecture has an independent, ANALOGOUS low-rank risk at low D
(rank(S_D) <= min(32, n_h*D), `stage2_composer.py`) -- so a low-D M-D0
convergence deficit could be misread as "the composer failed to learn
depth-1 composition" when it is actually "the target demands rank d_state
but the architecture caps state rank below that at this depth," the exact
D-AMB pattern one level down. `_test_target_rank_matches_necessity` below
proves, by direct computation (not assumption), that the as-built target's
rank IS d_state for every group, with a planted eye-padding-style mutant
(zero-padding, not identity-padding, giving rank d_min) that the SAME
rank-equality check correctly flags as wrong -- the negative unit test
CLAUDE.md's hard rule requires, run to completion, not merely written.
"""
from __future__ import annotations

import functools
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chapter2"))

import coverage_calibration as cc
import group_task as gt
import readout
from groups import GROUP_NAMES, D_MIN, D_STATE, generating_set, word_product, rho_G_embedded
from group_word_encoder import GroupWordModel
from rank_utils import effective_rank as torch_effective_rank, stable_rank as torch_stable_rank

# ---------------------------------------------------------------------------
# Depth entry (S2.3, reused).
# ---------------------------------------------------------------------------

D_TRAIN_MAX = 8   # S2.3: preserves Stage 1's own D_train_max, byte-identical to L_TRAIN_HI.

# S2.6's example D_test grid: dense within the near-plateau band (S2.4),
# out to the decisive ~8x/D=64 checkpoint (S2.6 M-D3).
D_TEST_GRID = (9, 10, 12, 14, 16, 20, 24, 32, 48, 64)
NEAR_PLATEAU_BAND = tuple(D for D in D_TEST_GRID if D <= 2 * D_TRAIN_MAX)     # S2.4: ~D_train_max+1..2x
FAR_DEPTH_BAND = tuple(D for D in D_TEST_GRID if D >= 4 * D_TRAIN_MAX)        # S2.4: ~4x and beyond


def sample_train_batch_stage2(name: str, batch_size: int, gen: torch.Generator,
                              device="cpu", dtype=torch.float32) -> dict:
    """S2.3: final-state-only, per-batch-fixed-D ~ Uniform{1,D_TRAIN_MAX}.
    Reuses `group_task.sample_train_batch` verbatim (identical sampler in
    form; `l_hi` made explicit here for documentation, not because the
    default differs)."""
    return gt.sample_train_batch(name, batch_size, gen, device=device, dtype=dtype, l_hi=D_TRAIN_MAX)


# ---------------------------------------------------------------------------
# Per-fixed-depth query-coverage bars (S2.4's single-fixed-depth grid,
# generalizing coverage_calibration.py's Monte Carlo WITHOUT editing that
# file -- FILE OWNERSHIP forbids it; `pick_bars`/`sample_word_result`/
# `undersampled_trial`/`GROUPS` are imported and reused verbatim, only the
# fixed-depth healthy-trial draw is new (3 lines, mirroring `healthy_trial`
# with L=D instead of L~Uniform(9,16))).
# ---------------------------------------------------------------------------

def _healthy_trial_fixed_depth(rng, info: dict, D: int, n_words: int = cc.N_EVAL_WORDS) -> int:
    """coverage_calibration.py::healthy_trial's exact structure, L fixed at
    D instead of drawn from Uniform{9,16} -- S2.4's own executed-artifact
    convention ("a grid of FIXED single depths... rather than a length
    RANGE")."""
    results = set()
    for _ in range(n_words):
        results.add(cc.sample_word_result(rng, info["gens"], D))
    return len(results)


@functools.lru_cache(maxsize=None)
def stage2_depth_coverage_bar(name: str, D: int, n_trials: int = 20000,
                              base_seed: int = 20260714) -> dict:
    """Returns {'frac':..., 'count':...} via coverage_calibration.py's own
    `pick_bars` algorithm (imported, unmodified), applied to a single-row
    table built from a FIXED-DEPTH Monte Carlo draw. `n_trials=20000`
    matches S2.4's own executed design-round grid; smoke/CPU-fast callers
    should pass a smaller n_trials explicitly. `lru_cache`d: the same
    (group, D) point is looked up repeatedly across many cells/seeds during
    a real run, and the 20000-trial Monte Carlo is deterministic given its
    inputs -- caching avoids redoing it every call."""
    info = cc.GROUPS[name]
    rng = np.random.default_rng(base_seed + D)
    healthy = np.array([_healthy_trial_fixed_depth(rng, info, D, n_words=n_trials and cc.N_EVAL_WORDS)
                        for _ in range(n_trials)])
    undersamp = np.array([cc.undersampled_trial(rng, info) for _ in range(n_trials)])
    row = dict(order=info["order"], h_p1=float(np.percentile(healthy, 1)),
              u_max=float(undersamp.max()))
    bars = cc.pick_bars({name: row})
    return bars[name]


def check_depth_coverage_with_retry(name: str, base_seed: int, D: int, bar: int | None = None,
                                    n_words: int = gt.N_EVAL_WORDS, offset: int = 10_000) -> dict:
    """`group_task.check_coverage_with_retry`'s exact retry-once-then-hard-
    fail structure, generalized to a fixed depth via
    `group_task.sample_eval_words(..., L_lo=D, L_hi=D)` (already supported,
    zero new sampler code) and `group_task._distinct_count`/
    `CoverageGuardError` (reused, not reimplemented). `bar` defaults to
    `stage2_depth_coverage_bar(name, D)`'s pinned count (rounded up)."""
    if bar is None:
        bar = math.ceil(stage2_depth_coverage_bar(name, D)["count"])
    log = {"label": "stage2-depth-coverage", "group": name, "D": D, "bar": bar, "attempts": []}
    for attempt, seed_shift in enumerate((0, 1)):
        seed = base_seed + seed_shift + offset
        idx_list, prod_list = gt.sample_eval_words(name, seed, n_words, L_lo=D, L_hi=D)
        count = gt._distinct_count(prod_list)
        passed = count >= bar
        log["attempts"].append({"attempt": attempt, "seed": seed, "distinct_count": count, "passed": passed})
        if passed:
            log["result"] = "pass"
            log["final_seed"] = seed
            return log
    log["result"] = "hard_fail"
    raise gt.CoverageGuardError(
        f"stage2-depth-coverage guard HARD-FAILED for {name} at D={D}: two independent "
        f"N={n_words} draws both scored below bar={bar}. Full log: {log}"
    )


# ---------------------------------------------------------------------------
# TARGET RANK UNIT TEST (S1.33 [LEARN], the dispatch-named must-have).
# ---------------------------------------------------------------------------

def _mutant_zero_pad_target(rho: np.ndarray, d_state: int) -> np.ndarray:
    """A planted eye-padding-style DEFECT: zero-pads the ambient complement
    instead of identity-padding it. rank = d_min (not d_state) -- exactly
    the WRONG value the necessity argument needs, constructed here only to
    prove the rank check below has teeth."""
    d_min = rho.shape[0]
    out = np.zeros((d_state, d_state))
    out[:d_min, :d_min] = rho
    return out


def _test_target_rank_matches_necessity():
    print("=" * 88)
    print("TARGET RANK UNIT TEST (S1.33 [LEARN] -- D-AMB eye-padding rank tax)")
    print("=" * 88)
    rng = np.random.default_rng(20260716)
    for name in GROUP_NAMES:
        d_min, d_state = D_MIN[name], D_STATE[name]
        gens = generating_set(name)
        idx = rng.integers(0, len(gens), size=5)
        rho = word_product(gens, idx)

        honest = rho_G_embedded(rho, d_state)
        honest_rank = int(np.linalg.matrix_rank(honest))
        assert honest_rank == d_state, (
            f"{name}: as-built target rank {honest_rank} != necessity value d_state={d_state} "
            f"-- D-AMB (S1.33) REGRESSION: the block-embedded target is no longer full ambient "
            f"rank, so a rank-capped composer state could recover it without actually delivering "
            f"the rho-block, silently re-introducing the ambient-identity capacity tax"
        )

        mutant = _mutant_zero_pad_target(rho, d_state)
        mutant_rank = int(np.linalg.matrix_rank(mutant))
        assert mutant_rank == d_min, f"{name}: mutant construction sanity check itself failed"
        # THE TEETH CHECK: does the SAME necessity assertion (rank == d_state)
        # correctly flag the mutant as wrong? If mutant_rank == honest_rank,
        # the check would silently PASS a broken construction -- no teeth.
        mutant_would_pass = (mutant_rank == d_state)
        assert not mutant_would_pass, (
            f"{name}: TARGET RANK UNIT TEST HAS NO TEETH -- the planted eye-padding mutant "
            f"(rank={mutant_rank}) is indistinguishable from the honest target (rank={d_state}) "
            f"under this check"
        )
        print(f"  [{name}] honest as-built target rank={honest_rank} (== d_state={d_state})  PASS  |  "
              f"planted zero-pad mutant rank={mutant_rank} (== d_min={d_min}) -- correctly "
              f"DETECTED as violating the necessity value (would FAIL the same check)")
    print("\nRESULT: the as-built block-embedded target has rank EXACTLY d_state (full ambient "
          "rank, all singular values 1) for all 5 groups -- this is what any rank-vs-depth "
          "necessity argument in Stage 2's own M-D2 must use, NOT d_min (S1.33's D-AMB lesson) "
          "-- and the planted eye-padding-style mutant is correctly caught by the same check "
          "(negative test run to completion, not merely written).\n")


# ---------------------------------------------------------------------------
# D_test grid evaluation pipeline (Arms 2-3): batched, since every probe
# word at a fixed D shares the same length -- reuses readout.py's subspace-
# restriction/degauging machinery verbatim (entity_subspace_from_words,
# restrict, degauge_and_score, _split_with_diversity_retry), only the
# per-depth SAMPLING is new.
# ---------------------------------------------------------------------------

def evaluate_composer_at_depth(composer, name: str, D: int, seed: int, device="cpu",
                                reset_every: int | None = None,
                                n_words: int = gt.N_EVAL_WORDS, bar: int | None = None) -> dict:
    """One (cell, D) point of the M-D1/M-D2 accuracy-vs-depth /
    rank-vs-depth curves (S2.6). `reset_every`: the last-K-window control
    (S2.9 item 4), eval-time-only truncation of an already-trained Arm-3
    checkpoint -- zero new training cells."""
    d_min = D_MIN[name]
    cov_log = check_depth_coverage_with_retry(name, seed, D, bar=bar, n_words=n_words)
    idx_list, rho_list = gt.sample_eval_words(name, cov_log["final_seed"], n_words, L_lo=D, L_hi=D)

    idx_batch = torch.tensor(np.stack(idx_list), dtype=torch.long, device=device)
    composer.eval()
    with torch.no_grad():
        Z = composer(idx_batch, reset_every=reset_every)
    Z_np = Z.detach().cpu().numpy().astype(np.float64)

    U = readout.entity_subspace_from_words(Z_np, d_min)
    A_words = np.stack([readout.restrict(Z_np[i], U) for i in range(len(rho_list))])

    fit_idx, eval_idx, split_log = readout._split_with_diversity_retry(
        name, rho_list, seed=cov_log["final_seed"] + 20_000)
    A_fit = [A_words[i] for i in fit_idx]
    A_eval = [A_words[i] for i in eval_idx]
    rho_fit = [rho_list[i] for i in fit_idx]
    rho_eval = [rho_list[i] for i in eval_idx]

    scores = readout.degauge_and_score(A_fit, A_eval, rho_fit, rho_eval, d_min)
    A_eval_t = torch.tensor(np.stack(A_eval), dtype=torch.float32)
    Z_eval_t = torch.tensor(Z_np[eval_idx], dtype=torch.float32)
    scores["restricted_effective_rank"] = torch_effective_rank(A_eval_t).mean().item()
    scores["restricted_stable_rank"] = torch_stable_rank(A_eval_t).mean().item()
    scores["whole_matrix_effective_rank"] = torch_effective_rank(Z_eval_t).mean().item()
    scores["whole_matrix_stable_rank"] = torch_stable_rank(Z_eval_t).mean().item()
    scores["group"], scores["D"], scores["n_fit"], scores["n_eval"] = name, D, len(fit_idx), len(eval_idx)
    scores["coverage_log"], scores["split_log"] = cov_log, split_log
    scores["reset_every"] = reset_every
    return scores


# ---------------------------------------------------------------------------
# Arm-1 (GroupWordEncoder) reuse path -- DISCLOSED GAP: S1.33's own harvest
# record states Stage 1's 58-cell sweep persisted NO checkpoints
# (`run_capability_sep.py::train_and_eval_cell` trains then discards the
# model, only json metrics survive) -- so, as of this build, there is
# NOTHING on box for `load_arm1_checkpoint` to load. This loader is
# launch-ready for whenever Arm-1 checkpoints exist (a dedicated retrain-
# and-save pass, or a future Stage-1 re-run that persists them) and fails
# LOUDLY rather than fabricating a result. Separately, ANY Arm-1 checkpoint
# is hard-capped at D<=ARM1_L_MAX=16 by construction
# (`GroupWordEncoder.embed_tokens` asserts `L <= self.L_max`, and Stage 1's
# own harness always built `L_max=16`) -- D_TEST_GRID values beyond 16
# CANNOT be forward-passed through Arm 1 at all, which is not a harness bug
# but S1.25's proven untrained-positional-row defect manifesting exactly as
# S2.2.1 predicts ("Arm 1... collapses at or before D_test requires an
# untrained positional row"). `evaluate_arm1_at_depth` reports this as an
# explicit architecturally-excluded result rather than crashing the grid.
# ---------------------------------------------------------------------------

ARM1_L_MAX = 16


class Arm1CheckpointUnavailable(RuntimeError):
    pass


def load_arm1_checkpoint(path: str, d_state: int, n_gens: int, device="cpu") -> GroupWordModel:
    if not os.path.exists(path):
        raise Arm1CheckpointUnavailable(
            f"No Arm-1 checkpoint at {path!r}. Stage 1's own 58-cell sweep never persisted "
            f"checkpoints (S1.33 M2 build-gap disclosure, CAPABILITY_SEPARATION_DESIGN.md) -- "
            f"Arm-1 cells for Stage 2's D_test grid need a dedicated retrain-and-save pass (or a "
            f"future Stage-1 re-run that saves state_dicts) before this loader can be used for "
            f"a real evaluation. This is a disclosed launch-time gap, not a build defect."
        )
    model = GroupWordModel(d_state, n_gens, L_max=ARM1_L_MAX, h=32)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def evaluate_arm1_at_depth(model: GroupWordModel, name: str, D: int, seed: int, device="cpu",
                           n_words: int = gt.N_EVAL_WORDS, bar: int | None = None) -> dict:
    if D > ARM1_L_MAX:
        return dict(group=name, D=D, out_of_range=True, mean_cos=None, recovered_frac_90=None,
                   note=f"D={D} exceeds Arm-1's trained L_max={ARM1_L_MAX} positional-embedding "
                        f"table -- GroupWordEncoder.embed_tokens hard-asserts L<=L_max. This IS "
                        f"S1.25's proven untrained-row defect manifesting exactly as S2.2.1 "
                        f"predicts, not a harness bug. Reported as architecturally excluded.")
    result = readout.run_subspace_restriction_pipeline(model, name, base_seed=seed, device=device)
    result["group"], result["D"], result["out_of_range"] = name, D, False
    return result


# ---------------------------------------------------------------------------
# Prefix-state-fidelity diagnostic (S2.3: free post-hoc, zero additional
# GPU cost -- every intermediate S_t is a byproduct of one forward pass).
# Derives the subspace U ONCE (at t=D_TRAIN_MAX, the decisive/best-trained
# depth) and reuses it + the SAME fit-set c_hat across every earlier t, so
# the per-t scores are directly comparable (one fixed gauge, not refit at
# every t, which would launder a length-conditioned shortcut into looking
# like genuine composition).
# ---------------------------------------------------------------------------

def prefix_fidelity(composer, name: str, seed: int, device="cpu",
                    n_words: int = gt.N_EVAL_WORDS) -> dict:
    d_min = D_MIN[name]
    cov_log = check_depth_coverage_with_retry(name, seed, D_TRAIN_MAX, n_words=n_words)
    idx_list, rho_list = gt.sample_eval_words(name, cov_log["final_seed"], n_words,
                                              L_lo=D_TRAIN_MAX, L_hi=D_TRAIN_MAX)
    idx_batch = torch.tensor(np.stack(idx_list), dtype=torch.long, device=device)

    composer.eval()
    with torch.no_grad():
        all_Z = composer.forward_all(idx_batch)     # list of D_TRAIN_MAX tensors, each (N, d_state, d_state)
    Z_terminal = all_Z[-1].detach().cpu().numpy().astype(np.float64)

    U = readout.entity_subspace_from_words(Z_terminal, d_min)
    fit_idx, eval_idx, split_log = readout._split_with_diversity_retry(
        name, rho_list, seed=cov_log["final_seed"] + 20_000)
    A_fit_terminal = [readout.restrict(Z_terminal[i], U) for i in fit_idx]
    rho_fit = [rho_list[i] for i in fit_idx]
    c_hat, _ratios = readout.fit_scale(A_fit_terminal, rho_fit) if hasattr(readout, "fit_scale") \
        else _fit_scale_fallback(A_fit_terminal, rho_fit)

    gens = generating_set(name)
    per_t = []
    for t, Z_t in enumerate(all_Z, start=1):
        Z_t_np = Z_t.detach().cpu().numpy().astype(np.float64)
        A_t_eval = np.stack([readout.restrict(Z_t_np[i], U) for i in eval_idx])
        rho_prefix_eval = [word_product(gens, idx_list[i][:t]) for i in eval_idx]
        coses = []
        for A, rho_p in zip(A_t_eval, rho_prefix_eval):
            target = c_hat * rho_p
            num = float((A * target).sum())
            den = np.linalg.norm(A) * np.linalg.norm(target)
            coses.append(num / den if den > 1e-12 else 0.0)
        per_t.append(dict(t=t, mean_cos=float(np.mean(coses)),
                          recovered_frac_90=float(np.mean([c > 0.9 for c in coses]))))
    return dict(group=name, D_train_max=D_TRAIN_MAX, per_t=per_t, n_eval=len(eval_idx),
               coverage_log=cov_log, split_log=split_log)


def _fit_scale_fallback(A_fit: np.ndarray, rho_fit: list) -> tuple:
    """Only reached if verify_option_a_readout's fit_scale is somehow not
    exposed via readout.py -- least-squares scalar fit c minimizing
    sum||A - c*rho||^2, the same closed form verify_option_a_readout.py
    uses (num/den of the same inner products), kept as a defensive fallback
    so prefix_fidelity never silently no-ops if an import shape changes."""
    num = sum(float((A * rho).sum()) for A, rho in zip(A_fit, rho_fit))
    den = sum(float((rho * rho).sum()) for rho in rho_fit)
    c_hat = num / den if den > 1e-12 else 1.0
    return c_hat, None


# ---------------------------------------------------------------------------
# Smoke.
# ---------------------------------------------------------------------------

def smoke():
    print("=" * 88)
    print("  stage2_task.py SMOKE")
    print("=" * 88)

    _test_target_rank_matches_necessity()

    print("  per-fixed-depth coverage bar (fast n_trials=200 for CPU smoke; real gate uses 20000):")
    bar = stage2_depth_coverage_bar("S4", D=20, n_trials=200, base_seed=999_000)
    print(f"    S4 @ D=20: {bar}")
    assert bar["count"] > 0

    print("\n  check_depth_coverage_with_retry (S4 @ D=8, real 50-word draw):")
    log = check_depth_coverage_with_retry("S4", base_seed=42, D=8, bar=5)
    assert log["result"] == "pass"
    print(f"    result={log['result']} final_seed={log['final_seed']}")

    print("\n  evaluate_composer_at_depth (untrained composer, S4, D=20, tiny bar override for speed):")
    import stage2_composer as sc
    torch.manual_seed(0)
    composer = sc.GroupWordDeltaComposer(D_STATE["S4"], len(generating_set("S4")), h=32, n_h=2, beta_max=2.0)
    scores = evaluate_composer_at_depth(composer, "S4", D=20, seed=42, bar=5, n_words=50)
    assert "mean_cos" in scores and "restricted_effective_rank" in scores
    print(f"    mean_cos={scores['mean_cos']:.4f}  recovered_frac_90={scores['recovered_frac_90']:.4f}  "
          f"restricted_eff_rank={scores['restricted_effective_rank']:.3f}")

    print("\n  evaluate_composer_at_depth with reset_every=4 (last-K-window control):")
    scores_k = evaluate_composer_at_depth(composer, "S4", D=20, seed=42, bar=5, n_words=50, reset_every=4)
    assert scores_k["reset_every"] == 4
    print(f"    reset_every=4: mean_cos={scores_k['mean_cos']:.4f}")

    print("\n  evaluate_arm1_at_depth out-of-range handling (D=32 > ARM1_L_MAX=16):")
    from group_word_encoder import GroupWordModel as GWM
    dummy_arm1 = GWM(D_STATE["S4"], len(generating_set("S4")), L_max=ARM1_L_MAX, h=32)
    r = evaluate_arm1_at_depth(dummy_arm1, "S4", D=32, seed=42)
    assert r["out_of_range"] is True and r["mean_cos"] is None
    print(f"    D=32: out_of_range={r['out_of_range']}  note={r['note'][:70]}...")
    r_inrange = evaluate_arm1_at_depth(dummy_arm1, "S4", D=8, seed=42, bar=5, n_words=50)
    assert r_inrange["out_of_range"] is False
    print(f"    D=8 (in range): out_of_range={r_inrange['out_of_range']}  mean_cos={r_inrange['mean_cos']:.4f}")

    print("\n  load_arm1_checkpoint on a missing path -- must fail loudly, not silently:")
    raised = False
    try:
        load_arm1_checkpoint("/nonexistent/path/checkpoint.pt", D_STATE["S4"], len(generating_set("S4")))
    except Arm1CheckpointUnavailable as e:
        raised = True
        print(f"    Arm1CheckpointUnavailable raised as expected: {str(e)[:100]}...")
    assert raised, "load_arm1_checkpoint did not raise on a missing checkpoint -- silent gap risk"

    print("\n  prefix_fidelity (untrained composer, S4, D_train_max=8, tiny n_words for CPU speed):")
    pf = prefix_fidelity(composer, "S4", seed=7, n_words=50)
    assert len(pf["per_t"]) == D_TRAIN_MAX
    print(f"    per_t mean_cos trajectory: {[round(p['mean_cos'], 3) for p in pf['per_t']]}")

    print("\n" + "=" * 88 + "\n  stage2_task.py SMOKE PASSED\n" + "=" * 88)


if __name__ == "__main__":
    smoke()
