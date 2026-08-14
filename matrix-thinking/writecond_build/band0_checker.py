"""Band-0 (teacher-force leak gate) checker, F4-repaired: the CONTROL-B
branch (adopted verbatim by the coordinator adjudication).

Source: NCR_WRITE_CONDITIONING_DESIGN.md D5.3 (Band 0) + D7 (CONTROL B's
own definition) + NCR_WRITECOND_ATTACK_R4.md F4 (the un-repaired gate VOIDs
CONTROL B by construction, with no self-correcting path -- CONTROL B, by
D7's own definition, reuses --teacher-force-operator VERBATIM for its
2,000-step continuation, so it necessarily produces
config.teacher_force_operator=True / teacher_force_check.active=True /
ncr_zero_grad_checks_passed>0 -- the OPPOSITE of what the un-repaired gate
required of every Stage-1 artifact).

No box dependency: operates on plain dicts (the shape `run_two_arm_cell`'s
own `rec` JSON already has -- `config`, `teacher_force_check`), pure
Python. CPU-testable, no torch needed at all.
"""
from __future__ import annotations


def band0_check_current(rec: dict, cell_kind: str) -> dict:
    """The UN-REPAIRED Band 0 gate (D5.3, before F4), reproduced here ONLY
    so test_band0_checker.py can demonstrate the bug it fixes (F4's own
    finding: this VOIDs a well-formed CONTROL_B record by construction).
    NOT used by any real call site -- band0_check() below is the one and
    only gate any harvest/build code should call.
    """
    cfg_ok = rec["config"]["teacher_force_operator"] is False
    tfc = rec["teacher_force_check"]
    active_ok = tfc["active"] is False
    grad_ok = tfc["ncr_zero_grad_checks_passed"] == 0
    passed = cfg_ok and active_ok and grad_ok
    return dict(verdict="PASS" if passed else "VOID", cfg_ok=cfg_ok, active_ok=active_ok, grad_ok=grad_ok)


def band0_check(rec: dict, cell_kind: str, clean_eval_rec: dict | None = None,
                 steps_run: int | None = None) -> dict:
    """D5.3 Band 0, F4-repaired.

    cell_kind: one of "PRIMARY", "CONTROL_A", "CONTROL_C" (unchanged
    branch) or "CONTROL_B" (the inverted branch).

    PRIMARY / CONTROL_A / CONTROL_C -- unchanged from D5.3: every artifact
    must show config.teacher_force_operator == False,
    teacher_force_check.active == False,
    teacher_force_check.ncr_zero_grad_checks_passed == 0. FAIL => VOID the
    cell (re-run, never scored) -- an indistinguishable-from-a-genuine-
    teacher-forced-WIN artifact, per M7.

    CONTROL_B -- the gate INVERTS (F4's repair, adopted). CONTROL_B, by
    D7's own definition, reuses --teacher-force-operator VERBATIM for the
    2,000-step continuation, so it MUST show teacher_force_check.active ==
    True and ncr_zero_grad_checks_passed == steps_run (the per-step
    witness that ncr_head received EXACTLY zero gradient throughout the
    continuation -- the property this control's entire purpose depends
    on: "readout-adaptation-only", D7). A FAIL here means the continuation
    did NOT actually teacher-force throughout, so CONTROL_B is not
    measuring what it claims to.

    CONTROL_B's SCOREABLE output additionally requires D7's SEPARATE
    teacher_force=False clean-eval artifact (the pbe_repl.py pattern --
    the CONTINUATION's own `rec` is teacher-forced throughout by the
    branch above, so it carries no clean Z_sgd readout by construction;
    the training-time record and the clean-eval record are two DIFFERENT
    files). `clean_eval_rec` must be supplied and itself carry
    config.teacher_force_operator == False (that SEPARATE run's own flag,
    not the continuation's).
    """
    if cell_kind == "CONTROL_B":
        if steps_run is None:
            return dict(verdict="VOID", reason="CONTROL_B branch requires steps_run to check "
                                                 "ncr_zero_grad_checks_passed == steps_run")
        cfg_ok = rec["config"]["teacher_force_operator"] is True
        tfc = rec["teacher_force_check"]
        active_ok = tfc["active"] is True
        grad_ok = tfc["ncr_zero_grad_checks_passed"] == steps_run
        clean_present = clean_eval_rec is not None
        clean_ok = clean_present and clean_eval_rec.get("config", {}).get("teacher_force_operator") is False
        passed = cfg_ok and active_ok and grad_ok and clean_ok
        reason = None if passed else (
            "training-continuation did not teacher-force throughout (cfg/active/grad mismatch)"
            if not (cfg_ok and active_ok and grad_ok) else
            "D7's separate teacher_force=False clean-eval artifact is missing or itself mis-flagged")
        return dict(verdict="PASS" if passed else "VOID", cfg_ok=cfg_ok, active_ok=active_ok,
                     grad_ok=grad_ok, clean_eval_present=clean_present, clean_eval_ok=clean_ok, reason=reason)

    assert cell_kind in ("PRIMARY", "CONTROL_A", "CONTROL_C"), f"unknown cell_kind {cell_kind!r}"
    cfg_ok = rec["config"]["teacher_force_operator"] is False
    tfc = rec["teacher_force_check"]
    active_ok = tfc["active"] is False
    grad_ok = tfc["ncr_zero_grad_checks_passed"] == 0
    passed = cfg_ok and active_ok and grad_ok
    reason = None if passed else "a leaked teacher-force signal was detected on a non-CONTROL_B cell"
    return dict(verdict="PASS" if passed else "VOID", cfg_ok=cfg_ok, active_ok=active_ok,
                 grad_ok=grad_ok, reason=reason)
