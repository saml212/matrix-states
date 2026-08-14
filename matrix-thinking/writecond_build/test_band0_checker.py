"""Tests for band0_checker.py -- run to completion, no torch needed."""
from __future__ import annotations

from band0_checker import band0_check, band0_check_current

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_primary_rec(leak=False, ncr_grad_leak=False):
    return dict(config=dict(teacher_force_operator=leak),
                teacher_force_check=dict(active=leak, ncr_zero_grad_checks_passed=(5 if ncr_grad_leak else 0)))


def make_control_b_rec(correct=True, steps_run=2000):
    """The record D7's own continuation ACTUALLY produces: teacher-force
    reused VERBATIM, so config/active are True and
    ncr_zero_grad_checks_passed counts every step (asserted per-step in
    the training loop, runner:1336-1349)."""
    if correct:
        return dict(config=dict(teacher_force_operator=True),
                     teacher_force_check=dict(active=True, ncr_zero_grad_checks_passed=steps_run))
    return dict(config=dict(teacher_force_operator=False),
                teacher_force_check=dict(active=False, ncr_zero_grad_checks_passed=0))


def make_clean_eval_rec(clean=True):
    return dict(config=dict(teacher_force_operator=(not clean)))


def test_primary_pass():
    rec = make_primary_rec(leak=False)
    v = band0_check(rec, "PRIMARY")
    check("PRIMARY: clean run PASSES Band 0", v["verdict"] == "PASS", v)


def test_primary_leak_voided():
    rec = make_primary_rec(leak=True, ncr_grad_leak=True)
    v = band0_check(rec, "PRIMARY")
    check("PRIMARY: a leaked teacher-force signal is VOIDed", v["verdict"] == "VOID", v)


def test_control_a_and_c_same_branch():
    for kind in ("CONTROL_A", "CONTROL_C"):
        v_pass = band0_check(make_primary_rec(leak=False), kind)
        v_fail = band0_check(make_primary_rec(leak=True), kind)
        check(f"{kind}: clean PASSES", v_pass["verdict"] == "PASS")
        check(f"{kind}: leaked VOIDs", v_fail["verdict"] == "VOID")


def test_control_b_F4_bug_reproduced_on_old_gate():
    """F4's own finding: the WELL-FORMED CONTROL_B record (which by D7's
    own definition MUST show teacher_force_operator=True throughout) is
    VOIDed by the un-repaired gate -- reproduced here to prove the bug
    existed before asserting the repair closes it."""
    rec = make_control_b_rec(correct=True, steps_run=2000)
    v_old = band0_check_current(rec, "CONTROL_B")
    check("F4 bug reproduction: the OLD (un-repaired) gate VOIDs a well-formed CONTROL_B record",
          v_old["verdict"] == "VOID", v_old)


def test_control_b_repaired_gate_passes_the_correct_record():
    rec = make_control_b_rec(correct=True, steps_run=2000)
    clean = make_clean_eval_rec(clean=True)
    v = band0_check(rec, "CONTROL_B", clean_eval_rec=clean, steps_run=2000)
    check("F4 repair: the REPAIRED gate PASSES the same well-formed CONTROL_B record",
          v["verdict"] == "PASS", v)


def test_control_b_missing_clean_eval_voids():
    rec = make_control_b_rec(correct=True, steps_run=2000)
    v = band0_check(rec, "CONTROL_B", clean_eval_rec=None, steps_run=2000)
    check("CONTROL_B: missing D7 clean-eval artifact still VOIDs (has teeth)",
          v["verdict"] == "VOID" and v["clean_eval_present"] is False, v)


def test_control_b_bad_clean_eval_voids():
    rec = make_control_b_rec(correct=True, steps_run=2000)
    dirty_clean = make_clean_eval_rec(clean=False)   # clean-eval itself mis-flagged as teacher_force=True
    v = band0_check(rec, "CONTROL_B", clean_eval_rec=dirty_clean, steps_run=2000)
    check("CONTROL_B: a mis-flagged clean-eval artifact still VOIDs (has teeth)",
          v["verdict"] == "VOID" and v["clean_eval_ok"] is False, v)


def test_control_b_actually_not_teacher_forced_voids():
    """Negative test: if the continuation somehow did NOT teacher-force
    throughout (a real bug -- e.g. the flag silently dropped mid-resume),
    the repaired gate must still VOID it -- the inversion doesn't mean
    'always pass CONTROL_B'."""
    rec = make_control_b_rec(correct=False)   # active=False, i.e. did NOT teacher-force
    clean = make_clean_eval_rec(clean=True)
    v = band0_check(rec, "CONTROL_B", clean_eval_rec=clean, steps_run=2000)
    check("CONTROL_B: a continuation that did NOT teacher-force throughout is VOIDed (inversion has teeth)",
          v["verdict"] == "VOID", v)


def test_control_b_partial_grad_leak_voids():
    """ncr_zero_grad_checks_passed < steps_run means SOME step leaked
    gradient to ncr_head -- must VOID, not silently pass."""
    rec = dict(config=dict(teacher_force_operator=True),
               teacher_force_check=dict(active=True, ncr_zero_grad_checks_passed=1999))
    clean = make_clean_eval_rec(clean=True)
    v = band0_check(rec, "CONTROL_B", clean_eval_rec=clean, steps_run=2000)
    check("CONTROL_B: a partial (1999/2000) zero-grad-check count VOIDs (off-by-one has teeth)",
          v["verdict"] == "VOID", v)


def test_control_b_missing_steps_run_voids():
    rec = make_control_b_rec(correct=True, steps_run=2000)
    clean = make_clean_eval_rec(clean=True)
    v = band0_check(rec, "CONTROL_B", clean_eval_rec=clean, steps_run=None)
    check("CONTROL_B: missing steps_run VOIDs rather than silently defaulting", v["verdict"] == "VOID", v)


if __name__ == "__main__":
    test_primary_pass()
    test_primary_leak_voided()
    test_control_a_and_c_same_branch()
    test_control_b_F4_bug_reproduced_on_old_gate()
    test_control_b_repaired_gate_passes_the_correct_record()
    test_control_b_missing_clean_eval_voids()
    test_control_b_bad_clean_eval_voids()
    test_control_b_actually_not_teacher_forced_voids()
    test_control_b_partial_grad_leak_voids()
    test_control_b_missing_steps_run_voids()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")
