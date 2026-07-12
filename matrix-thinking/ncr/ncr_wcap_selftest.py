"""NCR write-capacity diagnostic CPU self-test suite --
NOVEL_ARCH_WATERFALL.md S9.7 (the cheap, <=10 GPU-h leg of S9's scaling
ladder pre-registration). Negative tests run to completion (CLAUDE.md
hard rule), not just written. Kept as its own NEW file, mirroring the
`ncr_opbank_selftest.py` precedent, so this diagnostic's build-validation
gate is independently reviewable without touching the wave-1 suite
(`ncr_selftest.py`, exercised unmodified end-to-end above this file's
tests, t01 below).

t01 GRIDS[8]/GRIDS[12] byte-identity regression -- the S9.5 prereq #1
    hard requirement ("must verify nt.GRIDS[8] resolves byte-identically
    before and after the change"); a snapshot of the pre-S9.7 values,
    independently re-typed here (not re-derived from the module under
    test), so a bug that corrupted GRIDS[8]/[12] in place cannot pass by
    comparing the module against itself.
t02 existing wave-1 CPU smoke suite (ncr_selftest.run_all) still passes
    end-to-end after this diagnostic's edits -- the regression floor.
t03 new-grid residue/identity invariants at K=14/15/16 (mirrors
    ncr_task's own _self_test spot checks, executed here too as an
    independent caller).
t04 closed-form layout smoke generalized to the Condition A/B ambient
    dimension (d=32, K=16) -- the "fla-transpose lesson" (CLAUDE.md):
    catches any bug the d=16-only anchor check structurally cannot see.
t05 per-arm micro end-to-end cell, ALL FOUR ARMS, at EVERY target
    (K, d, h) -- the S9.5 hard-rule bakein ("a K=16 Condition-A micro
    test does not license skipping Condition-B's own micro test"),
    executed at all 4 diagnostic cells: spare-probe K=14 (d=16,h=64),
    spare-probe K=15 (d=16,h=64), Condition A K=16 (d=32,h=64),
    Condition B K=16 (d=32,h=128).
t06 blank-out (P=1 bottleneck) executed and PASSING for the ncr arm at
    the largest (most capacity-different) diagnostic cell, Condition B
    K=16 -- an untrained (random-init) model, the mechanism check does
    not depend on training.
t07 param-count formula cross-check: build_arm('ncr', d, h) params match
    the S9.3-corrected closed form P(d,h)=40h^2+4dh+46h+d exactly at
    every diagnostic cell (independent arithmetic check, not just "it
    built without crashing").
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

import task_e as te            # noqa: E402
import ncr_task as nt          # noqa: E402
import ncr_models as nm        # noqa: E402


# Independently re-typed snapshot of the pre-S9.7 GRIDS[8]/GRIDS[12]
# values (copied from the committed ncr_task.py BEFORE this diagnostic's
# edit landed) -- t01 compares the live module against THIS, not against
# itself, so an in-place corruption cannot pass silently.
_GRIDS8_SNAPSHOT = dict(
    ladder=(5, 13, 21, 29, 61, 125, 253, 509, 1021),
    h_star=61,
    sweep=tuple(range(57, 65)),
    cost_probe=(2**10 + 5, 2**14 + 5, 2**17 + 5, 2**20 + 5),
    ladder_residue=5,
)
_GRIDS12_SNAPSHOT = dict(
    ladder=(9, 21, 45, 93, 189, 381, 765, 1533),
    h_star=57,
    sweep=tuple(range(49, 61)),
    cost_probe=(),
    ladder_residue=9,
)

# The 4 diagnostic cells (S9.7): (label, K, d, h)
WCAP_CELLS = [
    ("spare-probe K=14", 14, 16, 64),
    ("spare-probe K=15", 15, 16, 64),
    ("Condition A K=16", 16, 32, 64),
    ("Condition B K=16", 16, 32, 128),
]


def t01_grids_byte_identity_regression():
    assert nt.GRIDS[8] == _GRIDS8_SNAPSHOT, (
        f"GRIDS[8] CHANGED by the S9.7 edit -- this breaks "
        f"ncr_opbank_task.py's live `nt.GRIDS[8]` import (S9.5 prereq #1). "
        f"got={nt.GRIDS[8]} want={_GRIDS8_SNAPSHOT}")
    assert nt.GRIDS[12] == _GRIDS12_SNAPSHOT, (
        f"GRIDS[12] CHANGED by the S9.7 edit. "
        f"got={nt.GRIDS[12]} want={_GRIDS12_SNAPSHOT}")
    assert set(nt.GRIDS.keys()) >= {8, 12, 14, 15, 16}, nt.GRIDS.keys()
    print("t01 PASS: GRIDS[8]/GRIDS[12] byte-identical to the pre-S9.7 "
          "snapshot; GRIDS extended (not replaced) with 14/15/16")


def t02_existing_suite_still_passes(device):
    import ncr_selftest
    ncr_selftest.run_all(device)
    print("t02 PASS: existing 14-section wave-1 CPU smoke suite passes "
          "unmodified end-to-end after the S9.7 edits")


def t03_new_grid_invariants():
    for K in (14, 15, 16):
        g = nt.GRIDS[K]
        assert g["ladder_residue"] == K - 3, K
        for h in g["ladder"]:
            assert h % K == g["ladder_residue"], (K, h)
        assert g["h_star"] % K == g["ladder_residue"], K
        assert (g["h_star"] + 3) % K == 0, (K, "h_star+3 must be the identity point")
        assert nt.residue_label(g["h_star"] + 3, K) == "identity", K
        assert nt.residue_label(g["ladder_residue"], K) == "novel", K
        assert sorted(g["sweep"])[0] == g["h_star"] - K + 4, K
        assert len(g["sweep"]) == K, K
    print("t03 PASS: GRIDS[14]/[15]/[16] residue/identity/sweep invariants hold")


def t04_closed_form_generalized(device):
    import run_ncr as rn
    rn.closed_form_checks(device)                 # anchor (d=16,K=8), unchanged
    rn.closed_form_checks(device, d=32, K=16)      # S9.7: Condition A/B ambient dim
    print("t04 PASS: closed-form shift-matrix / transpose-detection checks hold "
          "at the anchor (d=16,K=8) AND the diagnostic's ambient dim (d=32,K=16)")


def t05_per_arm_micro_end_to_end(device):
    import tempfile
    import run_ncr as rn
    for label, K, d, h in WCAP_CELLS:
        with tempfile.TemporaryDirectory() as td_:
            real_pts = nt.eval_points
            real_batches, real_bs = rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE

            def tiny_points(K_, d=nt.D_PIN, _real=real_pts):
                keep_lo = {1, 2, 3}
                g = nt.GRIDS[K_]
                keep = keep_lo | {g["ladder"][0], g["h_star"]} | set(g["sweep"][:4])
                return [p for p in _real(K_, d=d)
                        if p.h in keep and p.component != "cost_probe"]
            try:
                nt.eval_points = tiny_points
                rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE = 2, 16
                for arm in nm.ALL_ARMS:
                    rec = rn.run_cell(arm, K, seed=0, steps=10, device=device,
                                      outdir=td_, d=d, h=h)
                    assert rec["status"] == "COMPLETED", (label, arm, rec["status"])
                    assert rec["eval"]["points"], (label, arm, "no eval points")
                    assert rec["params"] > 0, (label, arm)
                    if arm in ("ncr", "fwm"):
                        assert rec["axis_c_lock_sha256"], (label, arm)
                    if arm in ("ncr", "fwm", "loopedvec"):
                        assert rec["blank_out"]["passed"], (label, arm, rec["blank_out"])
            finally:
                nt.eval_points = real_pts
                rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE = real_batches, real_bs
        print(f"  [{label}] (K={K},d={d},h={h}) all 4 arms: "
              f"train->instruments->eval->JSON, blank-out PASSED")
    print("t05 PASS: per-arm micro end-to-end at all 4 diagnostic cells, all 4 arms")


def t06_blank_out_condition_b(device):
    """Untrained-model mechanism check at the largest capacity diagnostic
    cell (Condition B K=16, d=32, h=128) -- does not depend on training."""
    import run_ncr as rn
    torch.manual_seed(0)
    cfg = nt.claim_config(16, d=32)
    model = nm.build_arm("ncr", d=32, h=128).to(device)
    bo = rn.blank_out_check(model, cfg, device, seed=0)
    assert bo["passed"], bo
    print(f"t06 PASS: blank-out (P=1 bottleneck) at Condition B K=16 (d=32,h=128), "
          f"untrained model: {bo}")


def t07_param_formula_cross_check():
    for label, K, d, h in WCAP_CELLS:
        m = nm.build_arm("ncr", d=d, h=h)
        got = nm.n_params(m)
        want = 40 * h * h + 4 * d * h + 46 * h + d   # S9.3-corrected P(d,h)
        assert got == want, (label, K, d, h, got, want)
    print("t07 PASS: build_arm('ncr',d,h) params match the S9.3-corrected "
          "closed form P(d,h)=40h^2+4dh+46h+d exactly at every diagnostic cell")


ALL_TESTS = [
    ("t01_grids_byte_identity_regression", lambda device: t01_grids_byte_identity_regression()),
    ("t02_existing_suite_still_passes", lambda device: t02_existing_suite_still_passes(device)),
    ("t03_new_grid_invariants", lambda device: t03_new_grid_invariants()),
    ("t04_closed_form_generalized", t04_closed_form_generalized),
    ("t05_per_arm_micro_end_to_end", t05_per_arm_micro_end_to_end),
    ("t06_blank_out_condition_b", t06_blank_out_condition_b),
    ("t07_param_formula_cross_check", lambda device: t07_param_formula_cross_check()),
]


def run_all(device="cpu"):
    for i, (name, t) in enumerate(ALL_TESTS, 1):
        print(f"\n[{i}/{len(ALL_TESTS)}] {name}")
        t(device)
    print(f"\nncr_wcap_selftest: {len(ALL_TESTS)}/{len(ALL_TESTS)} sections PASSED")


if __name__ == "__main__":
    run_all("cpu")
