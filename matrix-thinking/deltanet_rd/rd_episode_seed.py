"""rd_episode_seed.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1 "F1b" /
M-NEW-1 / R3-F5's episode-seed schedule for Tasks 1/2's on-the-fly
`grammar_rd.sample_batch_rd` generation.

Precedent this generalizes (NOT re-derived): `reasoning_link_probe.py`'s
`episode_seed`/`STRIDE_SEED` mixed-radix-stride pattern (collision-free by
construction: each digit's stride strictly exceeds the previous digit's own
max reach). `rd_episode_seed` deliberately EXCLUDES architecture from its
inputs -- for a given `(task, seed_idx, ckpt_idx)`, all three H2H arms
(contender/ablation/Transformer) draw from
`torch.Generator().manual_seed(rd_episode_seed(...))` and therefore see the
byte-identical stream of `sample_batch_rd` episodes -- the on-the-fly-
generation analogue of `CLAUDE.md`'s "use the SAME dataset for ALL
experiments in a comparison," operationalized for a generator that has no
file to share.

Design history (sec 1.15/1.17/1.18, retained here as the build's own
record): Rev 1's original 3-key `TASK_BASE` had no `task1_stress`/
`task2_calib` slot (M-NEW-1, Rev 2's fix: widened to 5 keys). `seed_idx` had
no runtime bound while `ckpt_idx` did (R3-F5, Rev 3's fix: added the
matching assert + its own negative test, mirroring `ckpt_idx`'s).

Run the smoke gate: python rd_episode_seed.py
Exit code 0 = every item PASSED.
"""
from __future__ import annotations

import sys

# sec 1.3.1 F1b (M-NEW-1, Rev 2): 5 keys, each spaced 500,000 apart -- a build agent facing a
# missing key (Rev 1's 3-key table lacked task1_stress/task2_calib, yet Wave -1 items C/D need
# them) would either invent one or silently reuse task2_sweep's own base, colliding calibration
# with sweep seeds -- exactly the bug class this widening closes.
TASK_BASE = {
    "task1_calib": 0,
    "task1_stress": 500_000,
    "task1_sweep": 1_000_000,
    "task2_calib": 1_500_000,
    "task2_sweep": 2_000_000,
}
# Per-seed-index stride (n=3 seeds standing convention, extendable to n=12 per the sec 1.8
# seed-extension contingency): 49 * 10_000 + 9_999 = 499_999 < 500_000 -- the seed_idx<50 runtime
# bound below is EXACTLY what keeps the widest possible (seed_idx, ckpt_idx) pair strictly inside
# one TASK_BASE key's own 500,000-wide range, by construction (verified arithmetically here, and
# mechanically by smoke_collision_freedom() below).
STRIDE_SEED_RD = 10_000
STRIDE_STEP_RD = 1        # per-checkpoint-index stride (raw checkpoint index, not raw step count)

# sec 1.8's seed-extension contingency ceiling (n=3 -> n=9 -> n=12): the collision-freedom smoke
# enumerates seed_idx over this practical range, not the full runtime-legal [0,50) -- mirrors the
# design doc's own text ("no two distinct (task,seed_idx,ckpt_idx) triples ... seed_idx in [0,11]
# ... mapping to the same seed"). The runtime assert below stays the wider, defensive [0,50) bound;
# see the arithmetic note above for why that wider bound is ALSO collision-free, even though the
# smoke only exercises the practically-reachable sub-range exhaustively.
SEED_IDX_SMOKE_MAX = 11    # inclusive


def rd_episode_seed(task: str, seed_idx: int, ckpt_idx: int) -> int:
    """Deliberately excludes architecture/arm from the formula -- the load-bearing property (see
    module docstring). Returns an int suitable for `torch.Generator().manual_seed(...)`."""
    assert task in TASK_BASE, f"unknown task key {task!r}"
    assert 0 <= seed_idx < 50, (
        f"seed_idx={seed_idx} out of bounds [0,50) -- seed_idx*STRIDE_SEED_RD would overflow "
        f"into the next TASK_BASE key's own 500,000-wide range (R3-F5, Rev 3 runtime assert -- "
        f"the ckpt_idx bound below had one from M-NEW-1/Rev 2; seed_idx did not, an asymmetry "
        f"attack round 3 caught: seed_idx=50 on task1_calib collides with task1_stress's own seed_idx=0)")
    assert 0 <= ckpt_idx < STRIDE_SEED_RD, (
        f"ckpt_idx={ckpt_idx} >= STRIDE_SEED_RD={STRIDE_SEED_RD} -- would overflow into the "
        f"next seed_idx's own seed range (M-NEW-1, Rev 2 runtime assert)")
    return TASK_BASE[task] + seed_idx * STRIDE_SEED_RD + ckpt_idx * STRIDE_STEP_RD


# ---------------------------------------------------------------------------
# Smoke gate -- mirrors this codebase's house style (smoke_frozen_bias_lm.py's numbered items,
# FAILURES list, _report() PASS/FAIL banner) AND phase2b_seedext_stage_minus1.py's own SE4/SE5
# exhaustive-enumeration + manual try/except negative-test convention (NOT pytest.raises -- this
# suite has zero pytest dependency, matching that precedent).
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def smoke_1_collision_freedom() -> None:
    """Exhaustive collision-freedom over ALL FIVE TASK_BASE keys x seed_idx in
    [0, SEED_IDX_SMOKE_MAX] x the FULL ckpt_idx range [0, STRIDE_SEED_RD) --
    mirrors SE4's own "run to completion" exhaustive-enumeration discipline
    (phase2b_seedext_stage_minus1.py:164-196), extended here to the FULL
    ckpt_idx range (not merely a sample) since 5*12*10,000=600,000 int
    comparisons is cheap in pure Python (~1s), and CLAUDE.md's own
    structural-check rule prefers exact/exhaustive over a sampled proxy
    wherever affordable."""
    seen: dict[int, tuple] = {}
    n = 0
    for task in TASK_BASE:
        for seed_idx in range(SEED_IDX_SMOKE_MAX + 1):
            for ckpt_idx in range(STRIDE_SEED_RD):
                s = rd_episode_seed(task, seed_idx, ckpt_idx)
                key = (task, seed_idx, ckpt_idx)
                if s in seen:
                    _report("smoke 1: collision-freedom (5 keys x seed_idx[0..11] x full ckpt_idx)",
                            False, f"collision: {key} vs {seen[s]} -> seed {s}")
                    return
                seen[s] = key
                n += 1
    expected_n = len(TASK_BASE) * (SEED_IDX_SMOKE_MAX + 1) * STRIDE_SEED_RD
    ok = (n == expected_n == 600_000)
    _report("smoke 1: collision-freedom (5 keys x seed_idx[0..11] x full ckpt_idx[0..9999])", ok,
            f"n_checked={n} (expect {expected_n})")


def smoke_2_seed_idx_negative_test() -> None:
    """R3-F5's own dedicated negative test: seed_idx=49 (max legal) passes;
    seed_idx=50 must raise. Run to completion (CLAUDE.md: never merely
    write a negative test, actually execute it)."""
    boundary_ok = True
    try:
        rd_episode_seed("task1_calib", 49, 0)
    except AssertionError:
        boundary_ok = False

    over_raised = False
    try:
        rd_episode_seed("task1_calib", 50, 0)
        raise RuntimeError("NEGATIVE FAILED TO FAIL: seed_idx=50 did not raise")
    except AssertionError:
        over_raised = True

    ok = boundary_ok and over_raised
    _report("smoke 2: seed_idx exact-threshold negative test (49 passes, 50 raises)", ok,
            f"boundary_ok={boundary_ok} over_raised={over_raised}")


def smoke_3_ckpt_idx_negative_test() -> None:
    """M-NEW-1's ckpt_idx bound: STRIDE_SEED_RD-1 (max legal) passes;
    ckpt_idx=STRIDE_SEED_RD must raise. Run to completion."""
    boundary_ok = True
    try:
        rd_episode_seed("task1_calib", 0, STRIDE_SEED_RD - 1)
    except AssertionError:
        boundary_ok = False

    over_raised = False
    try:
        rd_episode_seed("task1_calib", 0, STRIDE_SEED_RD)
        raise RuntimeError(f"NEGATIVE FAILED TO FAIL: ckpt_idx={STRIDE_SEED_RD} did not raise")
    except AssertionError:
        over_raised = True

    ok = boundary_ok and over_raised
    _report(f"smoke 3: ckpt_idx exact-threshold negative test "
            f"({STRIDE_SEED_RD - 1} passes, {STRIDE_SEED_RD} raises)", ok,
            f"boundary_ok={boundary_ok} over_raised={over_raised}")


def smoke_4_unknown_task_negative_test() -> None:
    """Dict-membership guard: an unregistered task key must raise (mirrors
    episode_seed's own "unknown X" assert-message convention)."""
    raised = False
    try:
        rd_episode_seed("task3_bogus", 0, 0)
        raise RuntimeError("NEGATIVE FAILED TO FAIL: unknown task key did not raise")
    except AssertionError:
        raised = True
    _report("smoke 4: unknown task key negative test", raised)


def smoke_5_cross_task_margin() -> None:
    """Arithmetic sanity: the widest legal (seed_idx, ckpt_idx) pair under the RUNTIME bound
    (seed_idx<50, not merely the smoke's practical seed_idx<=11) still lands strictly below the
    next TASK_BASE key's own base -- the load-bearing arithmetic fact the module docstring claims,
    checked directly rather than merely asserted in prose."""
    widest = 49 * STRIDE_SEED_RD + (STRIDE_SEED_RD - 1)
    ok = widest < 500_000
    _report("smoke 5: widest runtime-legal (seed_idx=49, ckpt_idx=9999) offset stays < 500,000 "
            "(the TASK_BASE key spacing)", ok, f"widest_offset={widest}")


def main() -> int:
    print("=" * 70)
    print("rd_episode_seed.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1 (F1b) smoke suite")
    print("=" * 70)
    smoke_1_collision_freedom()
    smoke_2_seed_idx_negative_test()
    smoke_3_ckpt_idx_negative_test()
    smoke_4_unknown_task_negative_test()
    smoke_5_cross_task_margin()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
