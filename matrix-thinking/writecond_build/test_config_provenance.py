"""Tests for config_provenance.py -- run to completion, no torch needed."""
from __future__ import annotations

from config_provenance import (
    assert_no_teacher_force_write_supervision_conflict,
    assert_writecond_resume_match,
    build_writecond_config_fields,
)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_config_fields():
    d = build_writecond_config_fields(0.5, 1.0)
    check("config fields dict has write_supervision_weight", d["write_supervision_weight"] == 0.5)
    check("config fields dict has write_transverse_weight", d["write_transverse_weight"] == 1.0)


def test_mutual_exclusion_negative():
    """m4's own assert must actually FIRE on the exact combination it
    exists to catch -- executed, not merely written."""
    fired = False
    try:
        assert_no_teacher_force_write_supervision_conflict(teacher_force_operator=True, write_supervision_weight=0.5)
    except AssertionError:
        fired = True
    check("m4: teacher_force=True + write_supervision_weight>0 FIRES the assert", fired)


def test_mutual_exclusion_positive_cases_do_not_fire():
    for tf, wsw in [(False, 0.5), (True, 0.0), (False, 0.0)]:
        try:
            assert_no_teacher_force_write_supervision_conflict(teacher_force_operator=tf, write_supervision_weight=wsw)
            ok = True
        except AssertionError:
            ok = False
        check(f"m4: (teacher_force={tf}, weight={wsw}) does NOT fire", ok)


def test_resume_match_negative_weight():
    ckpt = dict(write_supervision_weight=0.5, write_transverse_weight=1.0)
    fired = False
    try:
        assert_writecond_resume_match(ckpt, write_supervision_weight=1.0, write_transverse_weight=1.0, cell_id="x")
    except AssertionError:
        fired = True
    check("M8: write_supervision_weight resume mismatch FIRES", fired)


def test_resume_match_negative_lambda_t():
    ckpt = dict(write_supervision_weight=0.5, write_transverse_weight=1.0)
    fired = False
    try:
        assert_writecond_resume_match(ckpt, write_supervision_weight=0.5, write_transverse_weight=3.0, cell_id="x")
    except AssertionError:
        fired = True
    check("M8: write_transverse_weight (lambda_t) resume mismatch FIRES", fired)


def test_resume_match_positive():
    ckpt = dict(write_supervision_weight=0.5, write_transverse_weight=1.0)
    try:
        assert_writecond_resume_match(ckpt, write_supervision_weight=0.5, write_transverse_weight=1.0, cell_id="x")
        ok = True
    except AssertionError:
        ok = False
    check("M8: matching resume does NOT fire", ok)


def test_resume_match_old_checkpoint_defaults():
    """Old checkpoints (pre-patch) have neither field -- must resume
    exactly as before (trivially matching the CURRENT CLI value)."""
    ckpt_old = dict()   # no write_supervision_weight/write_transverse_weight key at all
    try:
        assert_writecond_resume_match(ckpt_old, write_supervision_weight=0.7, write_transverse_weight=2.0, cell_id="x")
        ok = True
    except AssertionError:
        ok = False
    check("M8: an old (pre-patch) checkpoint with neither field resumes without firing", ok)


if __name__ == "__main__":
    test_config_fields()
    test_mutual_exclusion_negative()
    test_mutual_exclusion_positive_cases_do_not_fire()
    test_resume_match_negative_weight()
    test_resume_match_negative_lambda_t()
    test_resume_match_positive()
    test_resume_match_old_checkpoint_defaults()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")
