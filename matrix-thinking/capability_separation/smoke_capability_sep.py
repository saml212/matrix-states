"""CAPABILITY_SEPARATION_DESIGN.md -- the master pre-launch smoke gate,
mirroring `chapter2/run_task_d.py`'s `python run_task_d.py --smoke`
convention: runs EVERY build-stage self-test/smoke/negative-test in this
directory, in dependency order (S1.19's own sequencing discipline: a
downstream component's smoke should not be trusted until its dependencies'
smokes have already passed), and fails loudly on the first failure rather
than continuing past a broken foundation.

Run FIRST, before any GPU cell launches (S1.7 gate 7's mandatory pre-training
smoke list, this design's S1.4 delta-3 L=1/L=16 item, and every other
build-time obligation S1.12 flags as "not settled evidence until executed"):

    DRY_RUN_BYPASS=1 .venv/bin/python smoke_capability_sep.py

This does NOT run the real 58-cell/8000-step sweep (that is a separate,
PI-signed-off GPU launch, run_capability_sep.py without --smoke) or the
real DeltaProduct Fig.5 reproduction (box-only, beta_fla_smoke.py
--reproduce-fig5, needs real fla+GPU) -- both are explicitly out of this
BUILD agent's scope (build, not launch; independent audit follows).
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _section(title):
    print("\n" + "#" * 100)
    print(f"# {title}")
    print("#" * 100)


def main():
    failures = []

    def run(label, fn):
        _section(label)
        try:
            fn()
            print(f"\n[OK] {label}")
        except Exception:
            print(f"\n[FAIL] {label}")
            traceback.print_exc()
            failures.append(label)

    def _groups():
        import groups
        groups._self_test()
    run("1. groups.py -- generating-set closure + numpy/torch product agreement", _groups)

    def _encoder():
        import group_word_encoder as gwe
        gwe.smoke("cpu")
    run("2. group_word_encoder.py -- GroupWordEncoder L=1/L=16 forward+backward, force_rank_k", _encoder)

    def _blankout():
        import torch
        from blank_out import run_blank_out_test, run_blank_out_planted_leak_test
        from group_word_encoder import GroupWordModel
        torch.manual_seed(0)
        m = GroupWordModel(d_state=5, n_gens=4, L_max=16, h=32)
        run_blank_out_test(m, device="cpu")
        run_blank_out_planted_leak_test(m, device="cpu")
    run("3. blank_out.py -- the P=1 bottleneck gradient-based blank-out test "
        "+ BA-F6 planted-leak NEGATIVE TEST", _blankout)

    def _task():
        import group_task as gt
        r1 = gt._test_coverage_guard_detects_undersampling()
        r2 = gt._test_coverage_guard_second_miss_hard_fails()
        r3 = gt._test_group_salted_seeds_differ_across_groups_same_within()
        r4 = gt._test_sample_eval_words_defaults_respect_train_support()
        assert r1 and r2 and r3 and r4
    run("4. group_task.py -- query-coverage guard NEGATIVE TESTS (undersampling + second-miss "
        "hard-fail) + BA-F3 group-salted-seed NEGATIVE-then-POSITIVE TEST + F1 "
        "sample_eval_words()-defaults train-support NEGATIVE TEST", _task)

    def _readout_smoke():
        import torch
        from group_word_encoder import GroupWordModel
        from groups import D_STATE
        from group_task import generating_set
        import readout
        torch.manual_seed(0)
        name = "S4"
        model = GroupWordModel(D_STATE[name], len(generating_set(name)), L_max=16, h=32)
        res = readout.run_subspace_restriction_pipeline(model, name, base_seed=42, device="cpu")
        assert "mean_cos" in res and "restricted_effective_rank" in res
        assert readout._test_diversity_floor_second_miss_hard_fails()
    run("5. readout.py -- subspace-restriction + degauging pipeline (untrained-model smoke) "
        "+ fit/eval diversity-floor NEGATIVE TEST", _readout_smoke)

    def _gate1b():
        import gate1_synthetic_injection as g1
        g1.main()
    run("6. gate1_synthetic_injection.py -- Gate-1(b) synthetic-injection acceptance test (EXECUTED)", _gate1b)

    def _force_rank():
        import force_rank_arms as fra
        fra.smoke("cpu")
    run("7. force_rank_arms.py -- the 4-arm force-rank grid, 58-cell family total", _force_rank)

    def _truncation_curve():
        import truncation_curve as tc
        tc._test_truncation_curve_knees_at_planted_rank()
        tc.smoke("cpu")
    run("8. truncation_curve.py -- BA-F2 M2 post-hoc rank-k truncation curve: planted-rank "
        "UNIT TEST + real-pipeline wiring smoke", _truncation_curve)

    def _truncation_curve_nontied():
        import truncation_curve as tc
        tc._test_knee_threshold_falsifiable_on_nontied_spectrum()
    run("9. truncation_curve.py -- RA-1 COMPANION TEST: non-tied singular spectrum makes the "
        "0.9 knee-threshold constant falsifiable (a 0.5 mutation flips the knee 5->3)", _truncation_curve_nontied)

    def _budget():
        import budget_guard as bgm
        bgm._self_test()
        bgm._test_near_cap_denies_in_pinned_order()
        bgm._test_a5_to_s5_hard_stop_window()
    run("10. budget_guard.py -- worst-case arithmetic self-test + NEGATIVE TESTS (near-cap denial "
        "ordering + BA-F4 A5-to-S5 hard-stop window)", _budget)

    def _tost():
        import tost_analysis as ta
        ta._test_confirm_path()
        ta._test_falsify_path()
        ta._test_inconclusive_no_guard()
        ta._test_reject_path()
        ta._test_budget_denial_path()
        ta._test_budget_granted_path()
    run("11. tost_analysis.py -- TOST unit tests (CONFIRM/FALSIFY/INCONCLUSIVE/REJECT/denial/granted)", _tost)

    def _beta_fla():
        import torch
        import beta_fla_smoke as bfs
        # device-aware (deploy-stage box-smoke fix, 2026-07-09): real fla's
        # Triton kernel is CUDA-only + bf16-only -- on the box this section
        # must exercise the REAL kernel on GPU (piece 2 runs for real there);
        # locally (no fla, no CUDA) it stays the CPU stub and piece 2
        # correctly self-skips.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        is_stub = bfs.smoke_forward_backward(dev)
        bfs._test_fig5_l_max_cap_applied()           # BA-F5: L_max=4 cap NEGATIVE-then-POSITIVE test
        bfs.reproduce_fig5(dev, is_stub=is_stub)     # self-skips (box-only) under the CPU stub
    run("12. beta_fla_smoke.py -- forward/backward/grad-check (piece 1) + BA-F5 L_max cap test "
        "+ Fig.5 reproduction (real-fla) / box-only-skip (CPU stub) (piece 2)", _beta_fla)

    def _runner():
        import run_capability_sep as rcs
        rcs.smoke()
    run("13. run_capability_sep.py -- manifest/resume-safety/escalation wiring + BA-F1 "
        "calibration-report/--sweep gate smoke + F2/F3 independent-literal pins "
        "(GATE1A_L_RANGE, STEP_BUDGET)", _runner)

    print("\n" + "=" * 100)
    if failures:
        print(f"CAPABILITY-SEPARATION SMOKE GATE: {len(failures)} FAILURE(S): {failures}")
        print("=" * 100)
        sys.exit(1)
    print("CAPABILITY-SEPARATION SMOKE GATE: ALL 13 SECTIONS PASSED.")
    print("=" * 100)


if __name__ == "__main__":
    main()
