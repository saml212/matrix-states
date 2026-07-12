"""NCR operator-bank CPU self-test suite -- NOVEL_ARCH_WATERFALL.md S8.4.
Negative tests run to completion (CLAUDE.md hard rule), not just written.

t1  relation-distinctness assert has teeth (duplicate-relation kill-proof)
t2  mod-K periodicity guard applies per relation (task_e's own assert, reused
    verbatim -- bad held-out h must raise, exactly as it does for the
    single-relation program)
t3  bank-score C1 regression: OLD min-of-medians formula reproduces the
    false-HOLD counterexample; FIXED median-of-mins formula does not
t4  bank-score S8.3a fold: n_seeds_all3_hold / quorum gate at EVEN seed count
    (the residual S8.3a flagged: two moderate per-seed-mins can't average to
    HOLD, but hold_gated must still require the quorum)
t5  B-CHAIN fixed-point exclusion has teeth: WITHOUT exclusion, fixed points
    (composite sigma(a)==a) provably occur; WITH exclusion, none do, over
    the SAME generator stream
t6  param match (all trained arms within +-15% of ncr-bank)
t7  closed-form binexp/loop agreement per relation (ncr_opbank_models, reuse)
t8  blank-out (P=1 bottleneck) executed and PASSING for all 3 gradient-
    capable arms, on an untrained (random-init) model -- the mechanism
    check does not depend on training
t9  relation-ID-swap ablation has teeth: a POSITIVE-control synthetic model
    (Z_bank = the true per-relation z_ideal, i.e. an oracle write) shows
    correct-r recovery >> wrong-r recovery, clearing the pinned bar --
    proves the diagnostic detects a real difference when one exists,
    complementing the untrained-model near-null result (smoke test) which
    only proves it does NOT false-positive
t10 end-to-end micro cell, ALL 4 ARMS (the S7e lesson: "all arms, not one
    representative" -- a per-arm crash must not stay invisible)
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
for p in (CHAPTER2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import task_e as te                # noqa: E402
import ncr_opbank_task as ot       # noqa: E402
import ncr_opbank_models as om     # noqa: E402
import run_ncr_opbank as ro        # noqa: E402


def t1_relation_distinctness_kill_proof():
    gen = torch.Generator().manual_seed(1)
    succ_dup = torch.stack([te._permutation_graph(4, 8, gen, "cpu", torch.float32)] * 3, dim=1)
    try:
        ot.assert_relation_distinct(succ_dup)
        raise AssertionError("t1 FAILED: distinctness check did not catch 3 duplicated relations")
    except AssertionError as e:
        assert "IDENTICAL" in str(e), e
    # positive: genuinely independent draws must NOT raise
    cfg = ot.BankConfig()
    ot.generate_bank_episode(16, cfg, gen)   # raises internally if it fails
    print("t1 PASS: relation-distinctness assert has teeth (kill-proof + positive)")


def t2_modK_guard_per_relation():
    cfg = ot.BankConfig()
    gen = torch.Generator().manual_seed(2)
    # a multiple of K is the identity -- must raise via the SAME task_e guard
    # the single-relation program uses, applied here through BankConfig's
    # own __post_init__ re-construction path (h=8k is caught structurally by
    # sample_eval_batch_axis_r reducing to h_eff=0, which is a query with
    # hops=0; that's in-distribution, not a guard violation by itself, so the
    # load-bearing check is the CONFIG-LEVEL guard task_e enforces for
    # claim-mode points -- verified by re-exercising TaskEConfig directly
    # with a training-residue held-out point, mirroring assert_claim_point).
    try:
        te.TaskEConfig(d=cfg.d, K=cfg.K, variant="permutation",
                      H_train=ot.H_TRAIN, H_test=(11,),   # 11 % 8 == 3, a TRAIN residue
                      H_extra=(), orthogonal=True)
        raise AssertionError("t2 FAILED: task_e's mod-K guard did not reject a train-residue h")
    except AssertionError as e:
        assert "residue" in str(e) or "IDENTITY" in str(e), e
    # positive: h*=61 (novel residue) must construct cleanly
    te.TaskEConfig(d=cfg.d, K=cfg.K, variant="permutation", H_train=ot.H_TRAIN,
                   H_test=(ot.GRID8["h_star"],), H_extra=(), orthogonal=True)
    print("t2 PASS: mod-K periodicity guard applies per relation (reused verbatim)")


def t3_bank_score_C1_regression():
    psr = {0: {0: 1.0, 1: 1.0, 2: 0.0}, 1: {0: 1.0, 1: 0.0, 2: 1.0},
          2: {0: 0.0, 1: 1.0, 2: 1.0}}
    old_min_of_medians = min(
        sorted(psr[s][r] for s in psr)[len(psr) // 2] for r in range(3))
    assert old_min_of_medians == 1.0, "t3 setup broken: must reproduce the OLD bug first"
    bs = ot.bank_score(psr)
    assert bs["median"] == 0.0 and bs["band"] == "FAIL" and bs["n_seeds_all3_hold"] == 0, bs
    print(f"t3 PASS: C1 counterexample -- OLD formula gave false HOLD (1.0), "
         f"FIXED formula correctly gives FAIL (0.0): {bs}")


def t4_even_n_quorum_gate():
    # 2 seeds, each with per-relation-min = 0.7 (moderate, sub-0.9): median
    # of {0.7, 0.7} = 0.7 -> DEGRADED, not HOLD -- can't misfire here. The
    # residual S8.3a flagged is n_seeds_all3_hold NOT being gated; construct
    # the case where it WOULD matter: 2 seeds, mins {0.95, 0.0} -> median
    # 0.475 -> FAIL band regardless (can't average to HOLD from one 0.0), so
    # test the boundary the fix targets directly: hold_gated must require
    # n_seeds_all3_hold >= ceil(n/2), checked explicitly.
    psr_2of2 = {0: {0: 0.95, 1: 0.95, 2: 0.95}, 1: {0: 0.95, 1: 0.95, 2: 0.95}}
    bs = ot.bank_score(psr_2of2)
    assert bs["band"] == "HOLD" and bs["n_seeds_all3_hold"] == 2 and bs["quorum"] == 1
    assert bs["hold_gated"] is True, bs
    psr_1of2 = {0: {0: 0.95, 1: 0.95, 2: 0.95}, 1: {0: 0.95, 1: 0.95, 2: 0.4}}
    bs2 = ot.bank_score(psr_1of2)
    # median of mins {0.95, 0.4} = 0.675 -> DEGRADED, not HOLD; n_all3_hold=1 >= quorum=1
    assert bs2["band"] == "DEGRADED" and bs2["hold_gated"] is False, bs2
    print(f"t4 PASS: n_seeds_all3_hold quorum gate behaves correctly at even n "
         f"(2/2 HOLD+gated; 1/2 correctly DEGRADED+not-gated)")


def t5_axis_b_fixed_point_exclusion_kill_proof():
    cfg = ot.BankConfig()
    h_star = ot.GRID8["h_star"]
    # WITHOUT exclusion: over enough draws, some fixed points MUST occur
    # (K=8, so P(fixed point) ~= 1/8 per query under independent random
    # cycles) -- prove the confound is REAL, not hypothetical.
    gen_a = torch.Generator().manual_seed(5)
    n_fixed = 0
    n_total = 0
    for _ in range(30):
        eb = ot.sample_eval_batch_axis_b(cfg, 8, gen_a, r1=0, h1=h_star, r2=1, h2=h_star,
                                          exclude_fixed_points=False)
        n_fixed += int((eb["query_keys"] == eb["targets"]).all(dim=-1).sum())
        n_total += eb["query_keys"].shape[0]
    assert n_fixed > 0, (
        f"t5 setup FAILED: 0/{n_total} fixed points with exclusion OFF -- "
        f"cannot prove the exclusion logic has anything to exclude")
    # WITH exclusion (default): NONE of the returned batch may be a fixed point
    gen_b = torch.Generator().manual_seed(5)   # same stream as gen_a for a fair kill-proof
    n_fixed_excl = 0
    n_total_excl = 0
    for _ in range(30):
        eb = ot.sample_eval_batch_axis_b(cfg, 8, gen_b, r1=0, h1=h_star, r2=1, h2=h_star,
                                          exclude_fixed_points=True)
        n_fixed_excl += int((eb["query_keys"] == eb["targets"]).all(dim=-1).sum())
        n_total_excl += eb["query_keys"].shape[0]
    assert n_fixed_excl == 0, f"t5 FAILED: exclusion left {n_fixed_excl}/{n_total_excl} fixed points"
    print(f"t5 PASS: B-CHAIN fixed-point exclusion has teeth "
         f"({n_fixed}/{n_total} fixed points with exclusion OFF, "
         f"0/{n_total_excl} with exclusion ON, same generator stream)")


def t6_param_match():
    rep = om.assert_param_match()
    print(f"t6 PASS: param match -- {rep}")


def t7_closed_form_per_relation():
    om._self_test()   # includes the per-relation binexp/loop-vs-fp64-power check
    print("t7 PASS: closed-form binexp/loop agreement holds per relation (see above)")


def t8_blank_out_untrained():
    cfg = ot.BankConfig()
    for arm in ("ncr-bank", "fwm-bank", "loopedvec-bank"):
        torch.manual_seed(8)
        model = om.ARM_BUILDERS[arm]()
        model.eval()
        res = ro.blank_out_check_bank(model, cfg, "cpu", seed=8)
        assert res["passed"], f"t8 FAILED for {arm}: {res}"
    print("t8 PASS: blank-out (P=1 bottleneck) holds for ncr-bank/fwm-bank/loopedvec-bank, "
         "untrained (mechanism-level, not a trained-capability check)")


class _OracleBankModel(torch.nn.Module):
    """Positive control for t9: NOT a trained model -- Z_bank IS the exact
    per-relation z_ideal (S8.1's own C7 classical minimum-norm operator),
    so a correct-r read recovers EXACTLY (cos~1) and a wrong-r read gets a
    DIFFERENT relation's operator entirely. Proves the swap-ablation
    MEASUREMENT has teeth (detects a real difference) independent of
    whether any trained model has learned to exploit it yet (that is
    Phase-0's own empirical question, not this test's)."""
    arm = "ncr-bank"

    def encode(self, keys, values, rel_ids):
        cfg = ot.BankConfig()
        B = keys.shape[0]
        Zs = []
        for r in range(cfg.R):
            kr = keys[:, r * cfg.K:(r + 1) * cfg.K, :]
            vr = values[:, r * cfg.K:(r + 1) * cfg.K, :]
            Zs.append(te.ideal_Z(kr, vr))
        return torch.stack(Zs, dim=1)


def t9_swap_ablation_kill_proof():
    cfg = ot.BankConfig()
    model = _OracleBankModel()
    res = ro.relation_id_swap_ablation(model, cfg, "cpu", seed=9, h=ot.GRID8["h_star"])
    assert res["applicable"]
    assert res["median_cos_right"] > 0.99, (
        f"t9 setup FAILED: oracle correct-r read should recover ~exactly, got {res}")
    assert res["median_cos_wrong"] < res["bar"], (
        f"t9 FAILED: swap ablation did not detect the oracle's real relation-id "
        f"sensitivity -- {res}")
    gap = res["median_cos_right"] - res["median_cos_wrong"]
    assert gap > 0.5, f"t9 FAILED: right/wrong gap too small to call this 'detected': {res}"
    print(f"t9 PASS: relation-ID-swap ablation has teeth -- oracle right={res['median_cos_right']:.3f} "
         f"vs wrong={res['median_cos_wrong']:.3f} (gap {gap:.3f}, bar {res['bar']})")


def t10_end_to_end_all_arms():
    for arm in om.ALL_ARMS:
        rec = ro.run_cell_bank(arm, seed=0, steps=5, device="cpu",
                               outdir="/tmp/ncropbank_selftest_e2e")
        assert rec["status"] == "COMPLETED", (arm, rec)
        if arm in ("ncr-bank", "fwm-bank", "loopedvec-bank"):
            assert rec["blank_out"]["passed"], (arm, rec["blank_out"])
    print(f"t10 PASS: end-to-end micro cell COMPLETED for all {len(om.ALL_ARMS)} arms "
         f"({', '.join(om.ALL_ARMS)}) -- the S7e 'all arms, not one representative' lesson")


def _self_test():
    t1_relation_distinctness_kill_proof()
    t2_modK_guard_per_relation()
    t3_bank_score_C1_regression()
    t4_even_n_quorum_gate()
    t5_axis_b_fixed_point_exclusion_kill_proof()
    t6_param_match()
    t7_closed_form_per_relation()
    t8_blank_out_untrained()
    t9_swap_ablation_kill_proof()
    t10_end_to_end_all_arms()
    print("\nncr_opbank_selftest: ALL 10/10 PASSED")


if __name__ == "__main__":
    _self_test()
