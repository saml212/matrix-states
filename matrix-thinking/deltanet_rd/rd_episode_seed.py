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
    wherever affordable. This is the practically-reachable sub-range (no
    current call site exceeds seed_idx=11) -- see smoke_1b below for the
    FULL runtime-legal seed_idx range (AUD-F2 fix)."""
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


# AUD-F2 fix: sample of ckpt_idx values used by smoke_1b below, deliberately including BOTH runtime
# boundaries (0 and STRIDE_SEED_RD-1=9999) plus interior points spread across the range -- a SAMPLE,
# not the full [0,10000) enumeration (audit §1.20's own instruction: "a ckpt_idx sample incl. 0 and
# 9_999"), because smoke_1b's OWN novelty is widening seed_idx to the full runtime-legal [0,50)
# range (5 x 50 x this sample = manageable), while ckpt_idx's full-range exhaustiveness is already
# covered by smoke_1 above at the practically-reached seed_idx<=11 sub-range.
CKPT_IDX_SAMPLE = sorted({0, 1, 2, STRIDE_SEED_RD // 2, STRIDE_SEED_RD - 2, STRIDE_SEED_RD - 1})


def smoke_1b_collision_freedom_full_seed_idx_range() -> None:
    """AUD-F2 fix: the audit's mutation (c) found smoke_1's ORIGINAL
    seed_idx<=11 sub-range blind to any collision only reachable at
    seed_idx in [12,49] -- e.g. a TASK_BASE key shrunk enough that only a
    seed_idx>=12 offset from the PRECEDING key's block reaches into it.
    Extends the exhaustive enumeration to the FULL runtime-legal seed_idx
    range (rd_episode_seed's own `assert 0 <= seed_idx < 50`), crossed
    with `CKPT_IDX_SAMPLE` (a deliberate sample, not the full ckpt_idx
    range, at this wider seed_idx span -- see that constant's own
    docstring) x all 5 TASK_BASE keys."""
    seen: dict[int, tuple] = {}
    n = 0
    for task in TASK_BASE:
        for seed_idx in range(50):
            for ckpt_idx in CKPT_IDX_SAMPLE:
                s = rd_episode_seed(task, seed_idx, ckpt_idx)
                key = (task, seed_idx, ckpt_idx)
                if s in seen:
                    _report("smoke 1b: collision-freedom (5 keys x FULL seed_idx[0..49] x "
                            "ckpt_idx sample)", False, f"collision: {key} vs {seen[s]} -> seed {s}")
                    return
                seen[s] = key
                n += 1
    expected_n = len(TASK_BASE) * 50 * len(CKPT_IDX_SAMPLE)
    ok = (n == expected_n)
    _report("smoke 1b: collision-freedom (5 keys x FULL runtime-legal seed_idx[0..49] x ckpt_idx "
            f"sample {CKPT_IDX_SAMPLE})", ok, f"n_checked={n} (expect {expected_n})")


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
    """AUD-F2 fix: derived from the LIVE TASK_BASE dict, never a hardcoded
    500_000. The audit's own mutation test (shrink `task1_stress` to
    400_000 in a scratchpad copy) found the PRE-FIX version of this check
    compared against a hardcoded 500_000 constant -- so it kept PASSING
    even when the real dict's own spacing had shrunk below the widest
    legal offset, i.e. it was checking the WRONG number, INERT to the
    exact class of bug it exists to catch. Fix: compute the MINIMUM
    pairwise spacing between any two TASK_BASE bases directly from the
    dict, and assert the widest runtime-legal (seed_idx, ckpt_idx) offset
    stays strictly below THAT (not a constant that can silently drift out
    of sync with the dict it's supposed to describe)."""
    bases = sorted(TASK_BASE.values())
    min_spacing = min(b - a for a, b in zip(bases, bases[1:]))
    widest = 49 * STRIDE_SEED_RD + (STRIDE_SEED_RD - 1)
    ok = widest < min_spacing
    _report("smoke 5: widest runtime-legal (seed_idx=49, ckpt_idx=9999) offset stays < the LIVE "
            "TASK_BASE dict's own minimum pairwise base spacing (not a hardcoded constant)", ok,
            f"widest_offset={widest} min_TASK_BASE_spacing={min_spacing} bases={bases}")


def main() -> int:
    print("=" * 70)
    print("rd_episode_seed.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1 (F1b) smoke suite")
    print("=" * 70)
    smoke_1_collision_freedom()
    smoke_1b_collision_freedom_full_seed_idx_range()
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
