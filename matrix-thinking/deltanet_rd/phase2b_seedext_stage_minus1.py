"""phase2b_seedext_stage_minus1.py -- REASONING_LINK_DESIGN.md sec 16.19
(Rev 3, DESIGN-CLEARED-FOR-BUILD after the round-4 verify) Phase-2b SEED
EXTENSION Stage -1 self-tests: every item is an EXECUTABLE test with its
own registered pass criterion, mirroring phase2_stage_minus1.py's house
pattern exactly -- each item imports and exercises the REAL functions in
reasoning_link_probe.py / phase2_familiarization_train.py /
phase2_trajectory_analysis.py / frozen_bias_lm_sweep.py /
phase2b_off_cache.py, never a reimplemented copy. A SEPARATE suite (not an
extension of the audited 23-item phase2_stage_minus1.py) so the existing
suite stays byte-identical; the chain runs BOTH.

CPU-runnable throughout (the repo's CPU fla-stub, installed as an
`import reasoning_link_probe` side effect). Items SE2/SE7 read the REAL
archived n=3 artifacts (repo archive under experiment-runs/, or the
box-local results/phase2/ copies -- first that exists); everything else is
synthetic fixtures.

Run: REASONING_LINK_FORCE_CPU_STUB=1 python phase2b_seedext_stage_minus1.py
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reasoning_link_probe as rlp  # noqa: E402  (installs the CPU fla stub as an import side effect)
import phase2_familiarization_train as pft  # noqa: E402
import phase2_hexachotomy as phx  # noqa: E402
import phase2_trajectory_analysis as pta  # noqa: E402
import phase2b_off_cache as poc  # noqa: E402
import frozen_bias_lm_sweep as fbs  # noqa: E402

RESULTS = []


def _report(item: str, name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append({"item": item, "name": name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[Phase-2b seedext Stage-1 item {item}] {name}: {status}"
          + (f" -- {detail}" if detail else ""))
    assert passed, f"Phase-2b seedext Stage -1 item {item} ({name}) FAILED: {detail}"


def _find_archived(name: str) -> str:
    """Locates a REAL archived Phase-2b artifact: the repo's committed archive first (local dev
    box), then the box-local results/phase2 copy (H100). Loud failure if neither exists."""
    candidates = [
        os.path.join(HERE, "..", "..", "experiment-runs", "2026-07-08_phase2b", "results", name),
        os.path.join(HERE, "results", "phase2", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"archived artifact {name!r} found at NEITHER {candidates[0]!r} nor "
                             f"{candidates[1]!r} -- SE2/SE7 need the real archived n=3 artifacts")


# ---------------------------------------------------------------------------
# SE1 -- CI_T_975_BY_DF scipy cross-check (sec 16.19.5 item 2: pin AND verify).
# ---------------------------------------------------------------------------

def test_se1_ci_t_pinned_dict_scipy_crosscheck():
    from scipy.stats import t as t_dist
    expected_entries = {2: 4.303, 5: 2.571, 8: 2.306, 11: 2.201}
    assert rlp.CI_T_975_BY_DF == expected_entries, (
        f"CI_T_975_BY_DF drifted from the registered dict: {rlp.CI_T_975_BY_DF}")
    assert rlp.CI_T_975_BY_DF[2] == rlp.CI_T_975_DF2, (
        "the df=2 entry must equal the standing CI_T_975_DF2 constant (one truth, two names)")
    for df, pinned in rlp.CI_T_975_BY_DF.items():
        computed = float(t_dist.ppf(0.975, df))
        assert abs(computed - pinned) < 5e-4, (
            f"t.ppf(0.975, {df})={computed:.6f} disagrees with the pinned {pinned} at 3 decimals")
    _report("SE1", "CI_T_975_BY_DF: all 4 pinned t(df,.975) entries agree with "
                   "scipy.stats.t.ppf(0.975, df) to 3 decimals (pin AND verify)", True,
            f"entries={sorted(rlp.CI_T_975_BY_DF)}")


# ---------------------------------------------------------------------------
# SE2 -- delta_ci_n: archived-cell regression (sec 16.19.5 item 4(a)) + wrapper/assert negatives.
# ---------------------------------------------------------------------------

def test_se2_delta_ci_n_archived_regression_and_negatives():
    # (a) hard regression on EVERY already-archived n=3 cell (both corpora, both arms, primary
    # raw + secondary_ood, both K's, all 5 checkpoints): feeding the archived deltas against
    # zeros reproduces the stored mean and deltas BYTE-EXACTLY and the CI bounds to <= 2e-15.
    # The small slack on the BOUNDS ONLY is a measured, disclosed producer-platform artifact:
    # CPython changed builtins.sum to Neumaier compensated summation for floats in 3.12, so the
    # archive producer's interpreter and this one can differ by 1 ULP inside the variance sum,
    # amplified to a few ULP through sqrt/multiply in the half-width (measured max 8.9e-16
    # across all 480 bound readings on this box; means and deltas byte-exact everywhere).
    n_cells = 0
    max_bound_diff = 0.0
    for corpus in ("openr1-mix-ext", "wikitext-mix-ext"):
        with open(_find_archived(f"trajectory_{corpus}_phase2b.json")) as f:
            traj = json.load(f)
        for arm in ("global", "per_token"):
            for block in (traj["per_arm"][arm]["raw"], traj["secondary_ood"][arm]):
                for c_str, cell in block.items():
                    for K in (32, 20):
                        d = cell[f"delta_k{K}"]
                        re_ci = rlp.delta_ci_n(d["deltas"], [0.0, 0.0, 0.0])
                        assert re_ci["deltas"] == d["deltas"], f"deltas drift at {corpus}/{arm}/{c_str}/K{K}"
                        assert re_ci["mean"] == d["mean"], f"mean drift at {corpus}/{arm}/{c_str}/K{K}"
                        for k in ("ci_low", "ci_high"):
                            diff = abs(re_ci[k] - d[k])
                            max_bound_diff = max(max_bound_diff, diff)
                            assert diff <= 2e-15, (
                                f"{k} drift {diff} at {corpus}/{arm}/{c_str}/K{K} exceeds the "
                                f"measured producer-platform envelope (2e-15)")
                        n_cells += 1
    # (b) delta_ci_n3 is a byte-identical thin wrapper.
    a, b = [3.1, 3.3, 2.9], [2.8, 3.0, 2.7]
    assert rlp.delta_ci_n3(a, b) == rlp.delta_ci_n(a, b), "delta_ci_n3 wrapper output drifted"
    # (c) negatives, run to completion: n=1, unequal lengths, unpinned df, and the wrapper's own
    # retained len==3 contract.
    for bad_call, desc in (
            (lambda: rlp.delta_ci_n([1.0], [2.0]), "n=1"),
            (lambda: rlp.delta_ci_n([1.0, 2.0], [1.0]), "unequal lengths"),
            (lambda: rlp.delta_ci_n([1.0] * 4, [0.0] * 4), "df=3 unpinned"),
            (lambda: rlp.delta_ci_n3([1.0] * 12, [0.0] * 12), "wrapper len!=3")):
        try:
            bad_call()
            raise RuntimeError(f"NEGATIVE FAILED TO FAIL: {desc} did not raise")
        except AssertionError:
            pass
    _report("SE2", "delta_ci_n reproduces every archived n=3 cell (deltas/mean byte-exact, CI "
                   "bounds <= 2e-15, producer-platform sum() difference disclosed); delta_ci_n3 "
                   "is a byte-identical wrapper; all 4 negatives raise", True,
            f"n_cells={n_cells}, max_bound_diff={max_bound_diff:.3g}")


# ---------------------------------------------------------------------------
# SE3 -- synthetic n=12 (and n=9) fixtures with HAND-COMPUTED expected CIs (sec 16.19.5 item
# 4(b), done before any real n=12 data exists so the test has real teeth).
# ---------------------------------------------------------------------------

def test_se3_hand_computed_n12_and_n9_fixtures():
    # n=12: deltas 0..11 (values_a = 10+i, values_b = 10 -- exact float subtraction).
    # HAND: mean = 66/12 = 5.5; var = sum((i-5.5)^2)/11 = (506 - 12*30.25)/11 = 143/11 = 13.0
    # (integer arithmetic, exact); se = sqrt(13/12); hw = 2.201*sqrt(13/12) = 2.290873432412479.
    r = rlp.delta_ci_n([10.0 + i for i in range(12)], [10.0] * 12)
    assert r["mean"] == 5.5, r["mean"]
    assert abs(r["ci_high"] - r["mean"] - 2.290873432412479) < 1e-12, r
    assert abs(r["mean"] - r["ci_low"] - 2.290873432412479) < 1e-12, r
    # n=9 (exercises the df=8 pinned entry): deltas 0..8.
    # HAND: mean = 36/9 = 4.0; var = (204 - 9*16)/8 = 60/8 = 7.5; se = sqrt(7.5/9);
    # hw = 2.306*sqrt(5/6) = 2.1050803626781885.
    r9 = rlp.delta_ci_n([float(i) for i in range(9)], [0.0] * 9)
    assert r9["mean"] == 4.0, r9["mean"]
    assert abs(r9["ci_high"] - r9["mean"] - 2.1050803626781885) < 1e-12, r9
    _report("SE3", "hand-computed synthetic fixtures: n=12 (mean 5.5, hw 2.201*sqrt(13/12)) and "
                   "n=9/df=8 (mean 4.0, hw 2.306*sqrt(5/6)) reproduce to <1e-12", True)


# ---------------------------------------------------------------------------
# SE4 -- episode_seed: exhaustive collision-freedom over the FULL widened keyspace at the NEW
# STRIDE_SEED=8_000 (sec 16.19.5 item 3(a)), run to completion.
# ---------------------------------------------------------------------------

def test_se4_episode_seed_widened_enumeration():
    assert rlp.STRIDE_SEED == 8_000, f"STRIDE_SEED must be the re-pinned 8_000, got {rlp.STRIDE_SEED}"
    # NOTE (registered, sec 16.19.5 item 3(a)): the stride change means episode_seed's NUMERIC
    # values for ckpt_seed_idx in {1,2} DIFFER from the original n=3 instrument's. Old-seed
    # bit-identity is deliberately NOT asserted here -- the design's protection is NON-INVOCATION
    # (archived results pool already-computed floats; episode_seed is never re-derived for
    # archived seeds), so this item asserts COLLISION-FREEDOM ONLY over the widened space.
    seen = {}
    n = 0
    for purpose in rlp.PURPOSE_BASE:
        for leg in rlp.LEG_BASE:
            for cond in range(4):
                for corpus_idx in range(2):
                    for seed_idx in range(12):
                        for k_idx in range(6):
                            s = rlp.episode_seed(purpose, leg, cond, corpus_idx, seed_idx, k_idx)
                            key = (purpose, leg, cond, corpus_idx, seed_idx, k_idx)
                            assert s not in seen, f"collision: {key} vs {seen[s]} -> {s}"
                            seen[s] = key
                            n += 1
    assert n == 3 * 2 * 4 * 2 * 12 * 6 == 3456, n   # purposes x legs x conditions x corpora x seeds x k's
    # Exact-threshold negatives (CLAUDE.md's integer/structural rule): idx 11 passes, 12 raises.
    rlp.episode_seed("eval", "leg_a", 0, 0, 11, 0)
    try:
        rlp.episode_seed("eval", "leg_a", 0, 0, 12, 0)
        raise RuntimeError("NEGATIVE FAILED TO FAIL: ckpt_seed_idx=12 did not raise")
    except AssertionError:
        pass
    _report("SE4", "episode_seed: zero collisions over the FULL widened keyspace (purposes x legs "
                   "x conditions x corpora x ckpt_seed_idx{0..11} x k_idx) at STRIDE_SEED=8000; "
                   "idx=12 hard-rejects; old-seed bit-identity NOT claimed (non-invocation is the "
                   "protection)", True, f"n_checked={n}")


# ---------------------------------------------------------------------------
# SE5 -- phase2_seed: exhaustive collision-freedom at the widened _MAX_CKPT_SEED=12 (sec 16.19.5
# item 3(b)), mirroring phase2_stage_minus1 item 9's method over the widened seed range.
# ---------------------------------------------------------------------------

def test_se5_phase2_seed_widened_enumeration():
    assert pft._MAX_CKPT_SEED == 12, f"_MAX_CKPT_SEED must be 12, got {pft._MAX_CKPT_SEED}"
    seen = {}
    n = 0
    for kind in pft._KIND_OFFSET:
        for arm in pft._ARM_INDEX:
            for corpus in ("openr1-mix-ext", "wikitext-mix-ext"):
                for ckpt_seed in range(12):
                    for k in pft.K_SWEEP:
                        for c in (0, *pft.CKPT_STEPS):
                            s = pft.phase2_seed(kind, arm, corpus, ckpt_seed, k, c)
                            key = (kind, arm, corpus, ckpt_seed, k, c)
                            assert s not in seen, f"collision: {key} vs {seen[s]} -> {s}"
                            seen[s] = key
                            n += 1
    assert n == len(pft._KIND_OFFSET) * 3 * 2 * 12 * len(pft.K_SWEEP) * 6, n
    # Disjointness from Phase-1's own episode_seed combinations (mirrors item 9's check, now
    # against the widened-and-restrided Phase-1 formula too).
    for combo in rlp.enumerate_registered_seed_combinations():
        assert rlp.episode_seed(*combo) not in seen, f"Phase-1 collision at {combo}"
    # Exact-threshold negatives: seed 11 passes, 12 hard-rejects.
    pft.phase2_seed("eval_lquery_heldout", "off", "wikitext-mix-ext", 11, 32, 2500)
    try:
        pft.phase2_seed("eval_lquery_heldout", "off", "wikitext-mix-ext", 12, 32, 2500)
        raise RuntimeError("NEGATIVE FAILED TO FAIL: ckpt_seed=12 did not raise")
    except AssertionError:
        pass
    _report("SE5", "phase2_seed: zero collisions over the FULL registered combination space at "
                   "_MAX_CKPT_SEED=12 (all kinds x arms x corpora x seeds{0..11} x K_SWEEP x "
                   "checkpoints), disjoint from every Phase-1 episode_seed combination; seed=12 "
                   "hard-rejects", True, f"n_checked={n}")


# ---------------------------------------------------------------------------
# SE6 -- contains_point + the sec 16.19.8 4-outcome MECE partition: boundary + totality tests.
# ---------------------------------------------------------------------------

def test_se6_contains_point_and_mece_partition():
    P = pta.SEEDEXT_ARCHIVED_POINT
    assert P == -0.4999, P
    # contains_point: NON-STRICT both sides.
    assert pta.contains_point(-0.5, -0.4, -0.4999)
    assert pta.contains_point(-0.4999, -0.1, -0.4999), "left endpoint exactly at point must count"
    assert pta.contains_point(-0.9, -0.4999, -0.4999), "right endpoint exactly at point must count"
    assert not pta.contains_point(-0.49, -0.1, -0.4999)
    # Bucket routing, each fixture maps to EXACTLY the registered bucket:
    cases = [
        ((-0.6, -0.4), "(i)"),                 # excludes 0, negative, contains -0.4999
        ((-0.4999, -0.1), "(ii)"),             # hmm -- see endpoint pin below
        ((-0.35, -0.05), "(ii)"),              # round-3 MAJOR-A's own counter-example CI
        ((-0.9, -0.6), "(ii)"),                # entirely MORE negative -- the other sub-direction
        ((-0.1, 0.2), "(iii)"),                # straddles zero
        ((0.0, 0.5), "(iii)"),                 # ci_low exactly 0: det() strict -> contains zero
        ((-0.5, 0.0), "(iii)"),                # ci_high exactly 0: same
        ((0.1, 0.4), "(iv)"),                  # positive side
    ]
    # The round-4 verify pins the endpoint-exactly-at--0.4999 boundary to bucket (i) via the
    # NON-STRICT contains_point -- correct the second fixture accordingly (a CI whose LEFT
    # endpoint is exactly -0.4999 CONTAINS the point, so it routes to (i), never (ii)).
    cases[1] = ((-0.4999, -0.1), "(i)")
    for (lo, hi), want in cases:
        got = pta.classify_seedext_outcome(lo, hi)["bucket"]
        assert got == want, f"CI [{lo},{hi}] routed to {got}, expected {want}"
    # Totality sweep: a dense grid of (lo, hi) pairs each maps to EXACTLY ONE bucket (the MECE
    # walk, re-enumerated mechanically -- every condition evaluated, exactly one fires).
    grid = [x / 8.0 for x in range(-10, 11)] + [P, P - 1e-9, P + 1e-9, 0.0]
    n_swept = 0
    for lo in grid:
        for hi in grid:
            if lo > hi:
                continue
            r = pta.classify_seedext_outcome(lo, hi)
            assert r["bucket"] in ("(i)", "(ii)", "(iii)", "(iv)"), r
            # independent re-derivation of the region, from the primitives directly:
            if lo <= 0.0 <= hi:
                want = "(iii)"
            elif hi < 0.0:
                want = "(i)" if (lo <= P <= hi) else "(ii)"
            else:
                want = "(iv)"
            assert r["bucket"] == want, f"[{lo},{hi}]: {r['bucket']} != independent {want}"
            n_swept += 1
    # Malformed CI refuses.
    try:
        pta.classify_seedext_outcome(0.5, -0.5)
        raise RuntimeError("NEGATIVE FAILED TO FAIL: ci_low > ci_high did not raise")
    except AssertionError:
        pass
    _report("SE6", "contains_point non-strict at both endpoints; all boundary fixtures (incl. the "
                   "round-3 [-0.35,-0.05] counter-example and endpoint-exactly-at--0.4999 -> (i)) "
                   "route to exactly the registered bucket; dense-grid totality sweep matches an "
                   "independent primitive-level re-derivation", True, f"n_swept={n_swept}")


# ---------------------------------------------------------------------------
# SE7 -- load_archived_arm_val: positive against the REAL archived artifacts (both hop_sets) +
# KeyError-on-miss negatives (sec 16.19.5 item 5; round-3 MAJOR-B symmetry).
# ---------------------------------------------------------------------------

def test_se7_archived_loader_real_artifacts_and_negatives():
    cache = poc.load_off_lquery_cache(_find_archived("off_lquery_cache-Phase2b.json"))
    with open(_find_archived("trajectory_wikitext-mix-ext_phase2b.json")) as f:
        traj = json.load(f)
    corpus, arm = "wikitext-mix-ext", "per_token"
    n_checked = 0
    max_diff = 0.0
    for hop_set, block in ((pft.H_TRAIN, traj["per_arm"][arm]["raw"]),
                            (pft.H_TEST_HELD_OUT, traj["secondary_ood"][arm])):
        for c in phx.CHECKPOINTS:
            for K in (32, 20):
                d = block[str(c)][f"delta_k{K}"]
                off_vals = [cache[pta.off_cache_key(corpus, s, K, c, hop_set)] for s in range(3)]
                arm_vals = [pta.load_archived_arm_val(corpus, arm, s, K, c, hop_set,
                                                        off_cache=cache, trajectory_json=traj)
                            for s in range(3)]
                # Driver-level reconstruction regression (sec 16.19.9 item 10): re-running the CI
                # over the loader's reconstruction reproduces the ARCHIVED CI. deltas re-derive as
                # off - (off - delta); IEEE subtraction round-trip plus the producer's sum() give
                # a measured worst-case 1.11e-16 -- pinned at <= 5e-16, disclosed (SE2's note).
                re_ci = rlp.delta_ci_n(off_vals, arm_vals)
                for k in ("mean", "ci_low", "ci_high"):
                    diff = abs(re_ci[k] - d[k])
                    max_diff = max(max_diff, diff)
                    assert diff <= 5e-16, f"{k} drift {diff} at c={c} K={K} hop={hop_set}"
                n_checked += 1
    # Negatives, run to completion -- KeyError on ANY miss, never a silent fallback:
    for bad_call, desc, exc in (
            (lambda: pta.load_archived_arm_val(corpus, arm, 3, 32, 2500, pft.H_TRAIN,
                                                 off_cache=cache, trajectory_json=traj),
             "ckpt_seed=3 (a NEW seed) through the archived loader", KeyError),
            (lambda: pta.load_archived_arm_val(corpus, arm, 0, 32, 2500, pft.H_TRAIN,
                                                 off_cache={}, trajectory_json=traj),
             "missing off-cache key", KeyError),
            (lambda: pta.load_archived_arm_val(corpus, arm, 0, 32, 2500, pft.H_TRAIN,
                                                 off_cache=cache,
                                                 trajectory_json={"per_arm": {arm: {"raw": {}}}}),
             "missing trajectory block key", KeyError),
            (lambda: pta.load_archived_arm_val(corpus, arm, 0, 32, 2500, (2, 3),
                                                 off_cache=cache, trajectory_json=traj),
             "unregistered hop_set", ValueError)):
        try:
            bad_call()
            raise RuntimeError(f"NEGATIVE FAILED TO FAIL: {desc} did not raise")
        except exc:
            pass
    _report("SE7", "load_archived_arm_val reconstructs all 20 archived wikitext/per_token cells "
                   "(BOTH hop_sets) such that delta_ci_n reproduces the archived CIs to <= 5e-16; "
                   "KeyError on new-seed/missing-key misuse, ValueError on an unregistered "
                   "hop_set -- never a silent live fallback", True,
            f"n_checked={n_checked}, max_diff={max_diff:.3g}")


# ---------------------------------------------------------------------------
# SE8 -- the whole-harvest-runtime live-eval guard (sec 16.19.5 item 5, round-3 MAJOR-B scope):
# ckpt_seed=0 raises through the SAME guard instance for BOTH hop_sets; ckpt_seed>=3 passes.
# ---------------------------------------------------------------------------

def test_se8_whole_runtime_guard_both_hop_sets():
    original = pta.eval_query_loss_heldout
    calls = []

    def _stub_inner(model, K, hop_set, corpus, ckpt_seed, checkpoint_step, batch_size=16,
                     device="cpu"):
        calls.append((ckpt_seed, tuple(hop_set)))
        return 3.0

    try:
        pta.eval_query_loss_heldout = _stub_inner
        pta.install_seedext_live_eval_guard()
        guarded = pta.eval_query_loss_heldout
        assert getattr(guarded, "_seedext_whole_runtime_guard", False), "guard marker missing"
        # Idempotence: a second install must NOT double-wrap.
        pta.install_seedext_live_eval_guard()
        assert pta.eval_query_loss_heldout is guarded, "second install double-wrapped the guard"
        # Negative, BOTH hop_sets through the SAME guard instance (round-3 MAJOR-B's extended
        # negative test), run to completion:
        for hop_set in (pft.H_TRAIN, pft.H_TEST_HELD_OUT):
            try:
                pta.eval_query_loss_heldout(object(), 32, hop_set, "wikitext-mix-ext", 0, 2500)
                raise RuntimeError(f"NEGATIVE FAILED TO FAIL: ckpt_seed=0 at hop_set={hop_set} "
                                    f"did not raise through the guard")
            except AssertionError:
                pass
        assert not calls, "a ckpt_seed=0 call REACHED the inner eval function -- guard is a no-op"
        # Positive: ckpt_seed=3 flows through to the inner function, both hop_sets.
        for hop_set in (pft.H_TRAIN, pft.H_TEST_HELD_OUT):
            assert pta.eval_query_loss_heldout(object(), 32, hop_set, "wikitext-mix-ext", 3, 2500) == 3.0
        assert calls == [(3, tuple(pft.H_TRAIN)), (3, tuple(pft.H_TEST_HELD_OUT))], calls
    finally:
        # Test hygiene ONLY (disclosed): production never uninstalls the guard -- this suite
        # restores the module global so later items (and the pre-existing 23-item suite, which
        # legitimately scores archived seeds) are not poisoned by a guard wrapping a stub.
        pta.eval_query_loss_heldout = original
    _report("SE8", "whole-harvest-runtime guard: ckpt_seed=0 raises through the SAME installed "
                   "guard instance for BOTH hop_sets (primary AND OOD, round-3 MAJOR-B), "
                   "ckpt_seed=3 passes through, install is idempotent", True)


# ---------------------------------------------------------------------------
# SE9 -- batch-effect pre-pooling gate: pinned pooled_SE formula + both flag axes + clear case.
# ---------------------------------------------------------------------------

def test_se9_batch_effect_gate():
    # Hand-computed fixture: old=[3.0, 3.02, 2.98] (mean 3.0, var 0.0004, SE 0.02/sqrt(3));
    # new = 9 values mean 3.0, var 0.000111 (SE = sqrt(0.000111/9)).
    old = [3.0, 3.02, 2.98]
    new = [3.01, 2.99, 3.015, 2.985, 3.005, 2.995, 3.012, 2.988, 3.0]
    g = pta.batch_effect_gate(old, new)
    se_old = math.sqrt(0.0004 / 3)
    var_new = sum((v - 3.0) ** 2 for v in new) / 8
    se_new = math.sqrt(var_new / 9)
    assert abs(g["pooled_se"] - math.sqrt(se_old ** 2 + se_new ** 2)) < 1e-15, (
        "pooled_SE must be the Welch SE-of-a-difference form sqrt(SE_old^2 + SE_new^2)")
    assert not g["flagged"], f"clear fixture wrongly flagged: {g}"
    # Mean-shift flag: shift new by +1.0 -- far beyond 2 x pooled_SE.
    g2 = pta.batch_effect_gate(old, [v + 1.0 for v in new])
    assert g2["mean_shift_flag"] and g2["flagged"], g2
    # Variance-ratio flag (either direction, larger/smaller > 4): blow up the NEW cohort's spread.
    g3 = pta.batch_effect_gate(old, [3.0 + (v - 3.0) * 10 for v in new])
    assert g3["var_ratio_flag"] and g3["flagged"], g3
    g4 = pta.batch_effect_gate([3.0, 3.5, 2.5], new)   # OLD side blown up -- same flag
    assert g4["var_ratio_flag"] and g4["flagged"], g4
    _report("SE9", "batch-effect gate: pooled_SE matches the hand-computed Welch form; mean-shift "
                   "and variance-ratio (both directions) fixtures flag; the clear fixture pools",
            True, f"clear var_ratio={g['var_ratio']:.2f}")


# ---------------------------------------------------------------------------
# SE10 -- the FOUR-BUCKET behavioral test THROUGH THE REAL analyze_corpus_seedext chain
# (synthetic on-disk cache/floor/trajectory artifacts + stubbed model loader/live eval): each of
# the 4 sec 16.19.8 buckets is exercised end-to-end, never a hand-built CI fed to the classifier.
# ---------------------------------------------------------------------------

def _se10_run_scenario(td: str, scenario_mean: float) -> dict:
    corpus, arm = pta.SEEDEXT_CORPUS, pta.SEEDEXT_ARM
    # Per-seed engineered structure (shared across every (K, c, hop_set) cell):
    off_jit = [0.0, 0.02, -0.02, 0.01, -0.01, 0.015, -0.015, 0.005, -0.005, 0.012, -0.012, 0.0]
    spread = [-0.055, -0.045, -0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035, 0.045, 0.055]
    off = [3.0 + j for j in off_jit]                       # batch gate stays CLEAR (SE9's shape)
    delta = [scenario_mean + p for p in spread]            # hw ~= 0.0229 around scenario_mean

    # Synthetic on-disk artifacts, REAL writer/reader machinery end-to-end:
    cache = {}
    for s in range(12):
        for K in (32, 20):
            for c in phx.CHECKPOINTS:
                for hop_set in (pft.H_TRAIN, pft.H_TEST_HELD_OUT):
                    cache[pta.off_cache_key(corpus, s, K, c, hop_set)] = off[s]
    cache_path = os.path.join(td, "off_cache.json")
    poc.write_off_lquery_cache(cache_path, cache)
    floor_path = os.path.join(td, "floor_n12.json")
    poc.write_floor_pinned(floor_path, {}, cache_path)     # pins the synthetic cache's sha256

    ci3 = {"deltas": delta[:3]}
    per_c = {str(c): {"delta_k32": ci3, "delta_k20": ci3} for c in phx.CHECKPOINTS}
    traj = {"per_arm": {arm: {"raw": per_c}}, "secondary_ood": {arm: per_c}}
    traj_path = os.path.join(td, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(traj, f)

    def _fake_load_eval_model(ckpt_path, device):
        m = re.search(r"phase2fam_(\w+)_.+_s(\d+)_step(\d+)\.pt$", ckpt_path)
        assert m, ckpt_path
        return types.SimpleNamespace(arm=m.group(1), ckpt_seed=int(m.group(2)),
                                      checkpoint_step=int(m.group(3)))

    live_seeds = []

    def _fake_eval(model, K, hop_set, corpus_, ckpt_seed, checkpoint_step, batch_size=16,
                    device="cpu"):
        assert model.arm == arm and model.ckpt_seed == ckpt_seed
        live_seeds.append(ckpt_seed)
        return off[ckpt_seed] - delta[ckpt_seed]

    orig_load, orig_eval = pta.phase2b_load_eval_model, pta.eval_query_loss_heldout
    pta.phase2b_load_eval_model = _fake_load_eval_model
    pta.eval_query_loss_heldout = _fake_eval
    try:
        result = pta.analyze_corpus_seedext("/fake/ckpts", cache_path, floor_path, traj_path,
                                              device="cpu")
    finally:
        pta.phase2b_load_eval_model, pta.eval_query_loss_heldout = orig_load, orig_eval

    # The guard was live INSIDE the driver: every live call carried a NEW seed only.
    assert live_seeds and min(live_seeds) >= 3, f"live path touched an archived seed: {sorted(set(live_seeds))}"
    # Both readouts fully populated, 12 deltas per pooled cell, gates clear, expected pooled CI.
    for table in (result["primary"], result["secondary_ood"]):
        assert set(table) == set(phx.CHECKPOINTS)
        for c in phx.CHECKPOINTS:
            for kk in ("delta_k32", "delta_k20"):
                entry = table[c][kk]
                assert entry["flagged"] is False, entry["batch_gate"]
                pooled = entry["pooled"]
                assert len(pooled["deltas"]) == 12
                for got, want in zip(pooled["deltas"], delta):
                    assert abs(got - want) < 1e-12, (got, want)
    return result


def test_se10_four_bucket_behavioral():
    scenarios = [
        (-0.4999, "(i)", "TRANSIENT-CONFIRMED-AT-MAGNITUDE"),   # CI contains -0.4999, excludes 0
        (-0.20, "(ii)", "TRANSIENT-CONFIRMED-SMALLER"),          # CI ~[-0.223,-0.177] ~ the round-3
                                                                  # [-0.35,-0.05]-class reading
        (0.0, "(iii)", "TRANSIENT-REFUTED"),                     # CI straddles zero
        (0.40, "(iv)", "NEW-PATTERN(SIGN-FLIP)"),                # CI positive
    ]
    outcomes = []
    for mean, want_bucket, want_outcome in scenarios:
        with tempfile.TemporaryDirectory() as td:
            result = _se10_run_scenario(td, mean)
        anchor = result["anchor"]
        assert anchor["K"] == 32 and anchor["checkpoint_step"] == 2500
        assert anchor["bucket"] == want_bucket, (
            f"scenario mean={mean}: anchor routed to {anchor['bucket']} ({anchor['outcome']}), "
            f"expected {want_bucket} ({want_outcome}); CI=[{anchor['ci_low']}, {anchor['ci_high']}]")
        assert anchor["outcome"] == want_outcome
        outcomes.append((mean, anchor["bucket"],
                          round(anchor["ci_low"], 4), round(anchor["ci_high"], 4)))
    _report("SE10", "FOUR-BUCKET behavioral test through the REAL analyze_corpus_seedext chain "
                    "(real cache/floor/trajectory files + verified loader + archived-values "
                    "loader + guarded live path): all four sec 16.19.8 buckets fire on their "
                    "engineered anchor CIs; live path touched seeds >=3 only; 12 deltas per cell "
                    "on BOTH readouts", True, f"outcomes={outcomes}")


# ---------------------------------------------------------------------------
# SE11 -- rung1-seedext manifest: composition + disjointness (positive AND injected-overlap
# negative, sec 16.19.7.1's registered Stage -1 item).
# ---------------------------------------------------------------------------

def test_se11_seedext_manifest_disjointness():
    steps = 20000
    ext = fbs.rung1_seedext_manifest(steps)
    base = fbs.rung1_manifest(steps)
    assert len(ext) == 18, f"seedext manifest must be 18 cells, got {len(ext)}"
    assert {s["arm"] for s in ext} == {"off", "per_token"}, "arms must be off/per_token ONLY"
    assert {s["corpus"] for s in ext} == {"wikitext-mix-ext"}
    assert sorted({s["seed"] for s in ext}) == list(range(3, 12))
    for s in ext:
        assert s["lam"] == (0.58 if s["arm"] == "per_token" else 0.0), s
    calib = fbs.calibration_cell_seedext(steps)
    assert calib["name"] in {s["name"] for s in ext} and calib["seed"] == 3 and calib["arm"] == "per_token"
    fbs.assert_manifests_disjoint(ext, base)   # positive: must NOT raise
    # NEGATIVE (run to completion): inject an overlapping seed -- a seed-0 per_token/openr1 spec
    # whose cell name IS in the rung-1 manifest -- the assert must fire.
    overlapping = ext + [fbs._spec("per_token", 0.58, "openr1-mix-ext", 0, steps, 1000)]
    try:
        fbs.assert_manifests_disjoint(overlapping, base)
        raise RuntimeError("NEGATIVE FAILED TO FAIL: injected overlapping seed did not fire the "
                            "disjointness assert")
    except AssertionError:
        pass
    _report("SE11", "rung1-seedext manifest: 18 cells, off/per_token x wikitext x seeds 3-11, "
                    "lambda 0.58/0.00, calibration=(per_token,wikitext,s3); disjoint from rung-1 "
                    "(positive) AND the injected-overlap negative fires", True)


# ---------------------------------------------------------------------------
# SE12 -- the val-loss sanity-band gate (sec 16.19.7.1's PI-signoff-style fixtures): in-band
# passes, below-band AND above-band both refuse -- function-level AND at the real CLI boundary.
# ---------------------------------------------------------------------------

def _write_calib_fixture(td: str, val_loss: float, wall_s: float = 908.7872) -> str:
    p = os.path.join(td, "calib.json")
    with open(p, "w") as f:
        json.dump({"complete": True, "wall_s": wall_s, "steps": 20000,
                    "run_name": "fixture",
                    "checkpoints": [
                        {"step": 10000, "val_loss": {"wikitext-mix-ext": 5.0}},
                        {"step": 20000, "val_loss": {"wikitext-mix-ext": val_loss}},
                    ]}, f)
    return p


def test_se12_valloss_band_gate():
    lo, hi = fbs.SEEDEXT_CALIB_BAND
    assert (lo, hi) == (4.2426, 4.4426), (lo, hi)
    assert fbs.seedext_valloss_band_gate(4.35)          # sec 16.19.7.1's own inside fixture
    assert fbs.seedext_valloss_band_gate(lo) and fbs.seedext_valloss_band_gate(hi)  # inclusive ends
    assert not fbs.seedext_valloss_band_gate(4.0)       # below-band refused
    assert not fbs.seedext_valloss_band_gate(5.0)       # above-band refused (sec 16.19.7.1's own)
    # Real CLI boundary (the exact seam the chain invokes), run to completion:
    sweep = os.path.join(HERE, "frozen_bias_lm_sweep.py")
    with tempfile.TemporaryDirectory() as td:
        rc_pass = subprocess.call([sys.executable, sweep, "--seedext-band-check",
                                    _write_calib_fixture(td, 4.35)], cwd=HERE)
        assert rc_pass == 0, f"in-band CLI check exited {rc_pass}"
        for bad in (4.0, 5.0):
            with tempfile.TemporaryDirectory() as td2:
                rc = subprocess.call([sys.executable, sweep, "--seedext-band-check",
                                       _write_calib_fixture(td2, bad)], cwd=HERE)
                assert rc == 6, f"out-of-band {bad} CLI check exited {rc}, expected 6"
    _report("SE12", "val-loss band gate [4.2426, 4.4426]: 4.35 passes, 4.0 (below) and 5.0 "
                    "(above) both refuse -- at the pure function AND the real CLI boundary "
                    "(exit 6), the exact seam frozen_bias_seedext_chain.sh invokes", True)


# ---------------------------------------------------------------------------
# SE13 -- the Leg-A timing gate (sec 16.19.6's calibration re-check): banked rate passes, a
# ceiling-busting rate refuses -- function-level AND at the real CLI boundary.
# ---------------------------------------------------------------------------

def test_se13_timing_gate():
    ok = fbs.seedext_timing_gate(fbs.SEEDEXT_BANKED_LEGA_CELL_S)
    assert ok["ok"] and abs(ok["ratio_vs_banked"] - 1.0) < 1e-12, ok
    # 18 x wall_s/3600 + 2.107 > 66.5 requires wall_s > 12878.6s (~14.2x the banked rate).
    bad = fbs.seedext_timing_gate(13000.0)
    assert not bad["ok"] and bad["projected_wave_gpu_h"] > fbs.SEEDEXT_WAVE_CEILING_GPUH, bad
    sweep = os.path.join(HERE, "frozen_bias_lm_sweep.py")
    with tempfile.TemporaryDirectory() as td:
        rc = subprocess.call([sys.executable, sweep, "--seedext-timing-check",
                               _write_calib_fixture(td, 4.35, wall_s=908.7872)], cwd=HERE)
        assert rc == 0, f"banked-rate CLI timing check exited {rc}"
    with tempfile.TemporaryDirectory() as td:
        rc = subprocess.call([sys.executable, sweep, "--seedext-timing-check",
                               _write_calib_fixture(td, 4.35, wall_s=13000.0)], cwd=HERE)
        assert rc == 8, f"ceiling-busting CLI timing check exited {rc}, expected 8"
    _report("SE13", "timing gate: banked 908.7872s/cell projects 6.65 GPU-h (passes); a "
                    "13000s/cell calibration projects past the 66.5 ceiling and refuses (exit 8) "
                    "-- function AND CLI boundary", True)


# ---------------------------------------------------------------------------
# SE14 -- the SECOND (sec 15.26) contention gate is INDEPENDENTLY wired (sec 16.19.7.1's
# registered build-time wiring check): a present Stage-1 sentinel does NOT satisfy it.
# ---------------------------------------------------------------------------

def test_se14_second_contention_gate_independent_wiring():
    sweep = os.path.join(HERE, "frozen_bias_lm_sweep.py")
    with tempfile.TemporaryDirectory() as td:
        stage1 = os.path.join(td, "STAGE1_RATES_OK")
        with open(stage1, "w") as f:
            f.write("fixture\n")
        missing_1526 = os.path.join(td, "SEC1526_WAVE_DONE")   # deliberately NOT created
        # Stage-1 PRESENT + sec1526 MISSING -> the calibration launch must refuse (exit 4) and
        # the refusal must name the sec1526 path (proving gate 2 is wired with its OWN sentinel,
        # not silently satisfied by gate 1's).
        r = subprocess.run(
            [sys.executable, sweep, "--wave", "rung1-seedext", "--calibration-only",
             "--rung1-steps", "20000", "--out-dir", td, "--gpus", "1", "--gpu-offset", "0",
             "--stage1-sentinel", stage1, "--sec1526-sentinel", missing_1526],
            cwd=HERE, capture_output=True, text=True)
        assert r.returncode == 4, f"expected exit 4, got {r.returncode}\n{r.stdout}\n{r.stderr}"
        assert missing_1526 in (r.stdout + r.stderr), (
            "the refusal does not name the sec1526 sentinel path -- the second gate may not be "
            "independently wired")
        assert "FOUND" in r.stdout and stage1 in r.stdout, "gate 1's own PASS was not logged first"
        # BOTH missing -> refusal happens at gate 1 (names the stage1 path) -- ordering check.
        r2 = subprocess.run(
            [sys.executable, sweep, "--wave", "rung1-seedext", "--calibration-only",
             "--rung1-steps", "20000", "--out-dir", td, "--gpus", "1", "--gpu-offset", "0",
             "--stage1-sentinel", os.path.join(td, "nope"), "--sec1526-sentinel", missing_1526],
            cwd=HERE, capture_output=True, text=True)
        assert r2.returncode == 4 and os.path.join(td, "nope") in (r2.stdout + r2.stderr)
    _report("SE14", "second contention gate: with the K-anchoring Stage-1 sentinel PRESENT and "
                    "the sec-15.26 sentinel MISSING the calibration launch refuses (exit 4) "
                    "naming the sec-15.26 path -- the two gates are independently enforced, "
                    "neither a stand-in for the other", True)


# ---------------------------------------------------------------------------
# SE15 -- frozen_bias_seedext_chain.sh's PI-signoff-style refusal without its env precondition.
# ---------------------------------------------------------------------------

def test_se15_chain_env_precondition_refusal():
    chain = os.path.join(HERE, "frozen_bias_seedext_chain.sh")
    env = {k: v for k, v in os.environ.items() if k != "FROZENBIAS_RUNG1_STEPS"}
    r = subprocess.run(["bash", chain], cwd=HERE, env=env, capture_output=True, text=True)
    assert r.returncode != 0, "chain did NOT refuse without FROZENBIAS_RUNG1_STEPS"
    assert "FROZENBIAS_RUNG1_STEPS" in (r.stdout + r.stderr), (
        f"refusal does not name the missing env var: {r.stderr[:400]}")
    _report("SE15", "frozen_bias_seedext_chain.sh refuses (nonzero, names the var) when "
                    "FROZENBIAS_RUNG1_STEPS is unset -- the PI-signoff precondition has teeth "
                    "(checked before cd, so the refusal fires on any box)", True)


# ---------------------------------------------------------------------------
# SE16 -- the OFF-eval cache EXTENSION + n=12 re-pin machinery (sec 16.19.7): positive merge,
# idempotent re-run, injected-overlap negative, tampered-cache negative -- via a synthetic
# builder (the registered build_fn test seam), real writer/reader/verifier machinery throughout.
# ---------------------------------------------------------------------------

def _se16_setup(td: str) -> tuple:
    corpus = poc.SEEDEXT_CORPUS
    cache = {}
    for s in (0, 1, 2):
        for K in poc.K_VALUES:
            for c in phx.CHECKPOINTS:
                for hop_set in poc.HOP_SETS:
                    cache[pta.off_cache_key(corpus, s, K, c, hop_set)] = 3.0 + 0.01 * s
    cache_path = os.path.join(td, "cache.json")
    poc.write_off_lquery_cache(cache_path, cache)
    orig_floor = os.path.join(td, "floor_orig.json")
    poc.write_floor_pinned(orig_floor, {}, cache_path)
    n12_floor = os.path.join(td, "floor_n12.json")

    def _synthetic_builder(ckpt_dir, device, corpora, ckpt_seeds, **kw):
        out = {}
        for corpus_ in corpora:
            for s in ckpt_seeds:
                for K in poc.K_VALUES:
                    for c in phx.CHECKPOINTS:
                        for hop_set in poc.HOP_SETS:
                            out[pta.off_cache_key(corpus_, s, K, c, hop_set)] = 3.0 + 0.001 * s
        return out

    return cache, cache_path, orig_floor, n12_floor, _synthetic_builder


def test_se16_cache_extension_and_n12_repin():
    with tempfile.TemporaryDirectory() as td:
        cache, cache_path, orig_floor, n12_floor, builder = _se16_setup(td)
        r = poc.extend_off_lquery_cache_seedext("/fake", "cpu", cache_path, orig_floor, n12_floor,
                                                  build_fn=builder)
        assert not r["skipped"] and r["n_new_keys"] == 180, r
        merged = poc.load_off_lquery_cache(cache_path)
        assert len(merged) == 60 + 180, len(merged)
        for k, v in cache.items():
            assert merged[k] == v, f"archived key {k!r} changed"   # byte-identical existing keys
        doc = poc.validate_floor_pinned(n12_floor)
        assert doc is not None, "n12 pin failed tamper-evidence validation against the extended cache"
        fd = doc["floor_by_corpus"][poc.SEEDEXT_CORPUS]
        assert len(fd["per_seed_ratios"]) == 12, "n12 pin must pool 12 per-seed ratios"
        assert "sec 16.19.7" in doc["formula_version"], "n12 pin must carry its own formula prose"
        # Idempotent re-run: everything already in place -> skip, cache byte-identical.
        sha_before = poc.sha256_of_file(cache_path)
        r2 = poc.extend_off_lquery_cache_seedext("/fake", "cpu", cache_path, orig_floor, n12_floor,
                                                   build_fn=builder)
        assert r2["skipped"] and poc.sha256_of_file(cache_path) == sha_before
    # NEGATIVE 1 (run to completion): one seedext key ALREADY present in the cache -> the overlap
    # assert fires before anything is written.
    with tempfile.TemporaryDirectory() as td:
        cache, cache_path, orig_floor, n12_floor, builder = _se16_setup(td)
        poisoned = dict(cache)
        poisoned[pta.off_cache_key(poc.SEEDEXT_CORPUS, 3, 32, 250, pft.H_TRAIN)] = 9.9
        poc.write_off_lquery_cache(cache_path, poisoned)
        poc.write_floor_pinned(orig_floor, {}, cache_path)   # re-pin so step-1 verification passes
        try:
            poc.extend_off_lquery_cache_seedext("/fake", "cpu", cache_path, orig_floor, n12_floor,
                                                  build_fn=builder)
            raise RuntimeError("NEGATIVE FAILED TO FAIL: pre-existing seedext key did not fire "
                                "the overlap assert")
        except AssertionError:
            pass
    # NEGATIVE 2 (run to completion): cache tampered after the original pin -> CacheIntegrityFailure.
    with tempfile.TemporaryDirectory() as td:
        cache, cache_path, orig_floor, n12_floor, builder = _se16_setup(td)
        with open(cache_path) as f:
            doc = json.load(f)
        first_key = next(iter(doc["cache"]))
        doc["cache"][first_key] += 1.0
        with open(cache_path, "w") as f:
            json.dump(doc, f, indent=2)
        try:
            poc.extend_off_lquery_cache_seedext("/fake", "cpu", cache_path, orig_floor, n12_floor,
                                                  build_fn=builder)
            raise RuntimeError("NEGATIVE FAILED TO FAIL: tampered cache did not raise "
                                "CacheIntegrityFailure")
        except poc.CacheIntegrityFailure:
            pass
    _report("SE16", "cache extension + n12 re-pin: 180 additive keys, existing 60 byte-identical, "
                    "n12 pin validates (12 per-seed ratios, its own sec 16.19.7 prose), re-run is "
                    "an idempotent no-op; injected-overlap and tampered-cache negatives both "
                    "refuse", True)


# ---------------------------------------------------------------------------
# SE17 -- the seedext eval timing-pilot projection (sec 16.19.7 m3): nominal warm rate passes,
# a ceiling-busting rate aborts; pass count pinned at 360.
# ---------------------------------------------------------------------------

def test_se17_seedext_timing_pilot_projection():
    assert poc.SEEDEXT_N_CACHED_PASSES == 360, poc.SEEDEXT_N_CACHED_PASSES
    warm = poc.project_and_gate_timing_pilot_seedext(2.1488)   # the prior wave's realized warm rate
    assert warm["ok"], warm
    assert abs(warm["projected_eval_gpu_h"] - 2.1488 * 360 / 3600.0) < 1e-12
    cold = poc.project_and_gate_timing_pilot_seedext(13.7339)  # the realized COLD rate precedent
    assert cold["ok"], ("the cold-cache rate is affordable at THIS wave's cost structure "
                         "(+1.16 GPU-h vs a 66.5 ceiling) -- it must pass, unlike the original "
                         f"wave's tight 26.4 ceiling: {cold}")
    busted = poc.project_and_gate_timing_pilot_seedext(700.0)
    assert not busted["ok"] and busted["projected_total_gpu_h"] > 66.5, busted
    _report("SE17", "seedext timing pilot: 360 passes pinned; warm 2.1488 s/pass and cold "
                    "13.7339 s/pass both project inside the 66.5 GPU-h ceiling (raw-projection "
                    "gate per m3's literal wording, 10x bracket reported as disclosure); a "
                    "700 s/pass rate aborts", True)


ALL_ITEMS = [
    test_se1_ci_t_pinned_dict_scipy_crosscheck,
    test_se2_delta_ci_n_archived_regression_and_negatives,
    test_se3_hand_computed_n12_and_n9_fixtures,
    test_se4_episode_seed_widened_enumeration,
    test_se5_phase2_seed_widened_enumeration,
    test_se6_contains_point_and_mece_partition,
    test_se7_archived_loader_real_artifacts_and_negatives,
    test_se8_whole_runtime_guard_both_hop_sets,
    test_se9_batch_effect_gate,
    test_se10_four_bucket_behavioral,
    test_se11_seedext_manifest_disjointness,
    test_se12_valloss_band_gate,
    test_se13_timing_gate,
    test_se14_second_contention_gate_independent_wiring,
    test_se15_chain_env_precondition_refusal,
    test_se16_cache_extension_and_n12_repin,
    test_se17_seedext_timing_pilot_projection,
]


def run_all_selftests() -> bool:
    print("=" * 70)
    print("PHASE-2B SEED-EXTENSION Stage -1 SELF-TESTS -- 17 items, REASONING_LINK_DESIGN.md "
          "sec 16.19 (Rev 3, DESIGN-CLEARED-FOR-BUILD, round-4 verify)")
    print(f"fla_stub_installed={rlp.FLA_STUB_INSTALLED}")
    print("=" * 70)
    t0 = time.time()
    failures = []
    for fn in ALL_ITEMS:
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"  ** FAILURE in {fn.__name__}: {e}")
        except Exception as e:   # noqa: BLE001 -- a raised RuntimeError IS a test failure here
            failures.append((fn.__name__, repr(e)))
            print(f"  ** FAILURE in {fn.__name__}: {e!r}")
    wall = time.time() - t0
    print("=" * 70)
    if failures:
        print(f"PHASE-2B SEEDEXT Stage -1: {len(failures)} FAILURE(S) in {wall:.1f}s")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        print("=" * 70)
        return False
    print(f"PHASE-2B SEEDEXT Stage -1: ALL {len(ALL_ITEMS)} ITEMS PASSED in {wall:.1f}s")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = run_all_selftests()
    sys.exit(0 if ok else 1)
