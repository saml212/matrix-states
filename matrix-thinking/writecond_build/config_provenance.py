"""D8's config-provenance build-brief items, as a self-contained,
CPU-testable module -- NOT a wiring edit into the pinned (archived)
`ncr_lm_wave1_runner.py` itself (out of this round's authorized repo-write
scope, `§A4-ADJUDICATION`: "Repo writes allowed: the design-doc append +
matrix-thinking/writecond_build/**"). Stage-1's build agent wires these
into `rec["config"]`, `save_checkpoint`, and `run_two_arm_cell`'s resume
path exactly as D8 specifies; this module ships the tested LOGIC those call
sites need, so the wiring is a mechanical copy-in, not a fresh derivation.

Covers:
  - M8: `--write-supervision-weight` / `--write-transverse-weight` (=
    lambda_t, D1.2) added to `rec["config"]`, and to the resume-time
    mismatch asserts alongside `seed`/`freeze_entity_adapter` (the SAME
    pattern `save_checkpoint`'s own docstring already establishes for
    those two fields, runner:1088-1105 -- a resumed cell must not
    silently change either weight mid-run).
  - m4: the mutual-exclusion assert -- `teacher_force_operator` AND
    `write_supervision_weight > 0` together make `Z == Z_ideal` in the
    forward pass identically (D1.3's own zero-set proof, evaluated at the
    trivial point Z_sgd=Z_ideal: L_write=0 identically, a silent no-op).
"""
from __future__ import annotations


WRITECOND_CONFIG_FIELDS = ("write_supervision_weight", "write_transverse_weight")


def build_writecond_config_fields(write_supervision_weight: float, write_transverse_weight: float) -> dict:
    """M8: the two new provenance fields, in the SAME dict-literal style
    `run_two_arm_cell`'s own `rec["config"]` already uses (runner:1257-1264)
    -- a Stage-1 build agent folds this dict's items into that literal
    alongside the existing `teacher_force_operator`/`aux_read_loss_weight`/
    `ortho_reg_weight` entries."""
    return dict(write_supervision_weight=float(write_supervision_weight),
                write_transverse_weight=float(write_transverse_weight))


def assert_no_teacher_force_write_supervision_conflict(teacher_force_operator: bool,
                                                          write_supervision_weight: float) -> None:
    """m4: the mutual-exclusion assert, called at cell launch (mirrors the
    runner's OWN launch-time discipline -- e.g. `_assert_ladder_sound`,
    called at import time, runner:320). With both set, `Z ident Z_ideal` in
    the forward pass (teacher_force_operator routes Z through
    integ.teacher_force_operator instead of ncr_head.encode -- runner:390-
    393) and `L_write ident 0` identically (D1.3's own zero-set proof,
    evaluated AT the trivial point Z_sgd=Z_ideal) -- a silent no-op that
    Band 0 would ALSO catch (teacher_force_check.active=True on a
    non-CONTROL_B cell VOIDs, band0_checker.py) but should never reach in
    the first place."""
    assert not (teacher_force_operator and write_supervision_weight > 0.0), (
        "assert_no_teacher_force_write_supervision_conflict: --teacher-force-operator and "
        "--write-supervision-weight > 0 are set together -- Z would be IDENTICALLY Z_ideal every "
        "step (teacher_force_operator bypasses ncr_head.encode entirely) and L_write would be "
        "IDENTICALLY zero (D1.3's own zero-set proof, evaluated at the trivial point Z_sgd=Z_ideal) "
        "-- a silent no-op. This is m4's own gate; Band 0 would also VOID the resulting cell, but "
        "this assert stops the launch before any GPU-h is spent on it.")


def assert_writecond_resume_match(ckpt: dict, write_supervision_weight: float,
                                   write_transverse_weight: float, cell_id: str) -> None:
    """M8: the resume-mismatch assert, SAME pattern as `run_two_arm_cell`'s
    own `seed`/`freeze_entity_adapter` asserts (runner:1214-1235) --
    `ckpt.get(field, current_value)` defaults to the CURRENT CLI value for
    checkpoints saved BEFORE this patch (old checkpoints resume exactly as
    before, trivially matching), so only NEW checkpoints (which now always
    carry both fields) are actually checked."""
    ckpt_wsw = ckpt.get("write_supervision_weight", write_supervision_weight)
    ckpt_wtw = ckpt.get("write_transverse_weight", write_transverse_weight)
    assert ckpt_wsw == write_supervision_weight, (
        f"[{cell_id}] --write-supervision-weight MISMATCH on resume: checkpoint was built with "
        f"write_supervision_weight={ckpt_wsw} but this launch passed {write_supervision_weight}. "
        f"Resuming would silently change L_write's own weighting mid-run.")
    assert ckpt_wtw == write_transverse_weight, (
        f"[{cell_id}] --write-transverse-weight (lambda_t) MISMATCH on resume: checkpoint was built "
        f"with write_transverse_weight={ckpt_wtw} but this launch passed {write_transverse_weight}. "
        f"Resuming would silently change lambda_t mid-run (F2's own point: lambda_t's effect is "
        f"NOT scale/trajectory-invariant -- a mid-run change is not a harmless no-op).")
