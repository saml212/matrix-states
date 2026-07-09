"""CAPABILITY_SEPARATION_DESIGN.md S2.5/S2.7/S2.8/S2.11 (Rev 3,
DESIGN-CLEARED-FOR-BUILD per S2.18) -- the Stage-2 calibration-first CLI:
cell-grid construction (68 new training cells: 50 primary + 18 n_h-grid),
the 11-cell calibration-first set, per-cell and ledger-level budget guards,
resume-safe cell running, and the orchestration glue tying
`stage2_composer.py` + `stage2_task.py` + `stage2_instrument.py` together.

This module builds and self-tests the ORCHESTRATION LOGIC on CPU with tiny
synthetic cells; it does NOT launch any real GPU training (out of this
BUILD agent's scope -- build, not launch; an independent audit follows,
S2.11's own sequencing). `main()`'s `--calibration-only`/real-launch paths
mirror `run_capability_sep.py`'s own PI-signoff-gated convention but are
NOT exercised end-to-end here.

Grid (S2.5): primary = 5 groups x {Arm 2 (beta in [0,1]), Arm 3 (beta in
[0,2])} x n=5 seeds, n_h=2 default = 50 cells. n_h force-arm grid = {S5,A6}
x n_h in {1,2,4} x n=3 seeds, Arm 3 only = 18 cells. Total 68. Calibration
set (S2.8 item 2, 11 cells): the 10 base (group,arm) cells at n_h=2, seed
index 0, PLUS the promoted (S5, Arm 3, n_h=4, seed index 0) cell -- drawn
from, not added to, the 68 (S2.5 Rev 2 pin: the base (S5,Arm3,n_h=2) cell
and the promoted (S5,Arm3,n_h=4) cell are DISTINCT, not a near-duplicate).

Budget (S2.7 Rev 2): per-cell abort ceiling = 1.5x the pricier end of the
0.018-0.054 GPU-h/cell planning band = 0.081 GPU-h/cell (this dispatch's
own "per-cell budget guard at 1.5x the S2.7 band" instruction). Ledger
circuit breaker mirrors `budget_guard.py`'s `check_base_sweep_projection`
MECHANISM, re-keyed to Stage 2's own 25 GPU-h cap and 68-cell grid (Stage
1's own module constants -- BASE_SEEDS, GENERAL_ESCALATION_ELIGIBLE, the
58-cell table -- are specific to Stage 1's own arm/cell-type structure and
do not apply here; re-implementing the MECHANISM against Stage 2's own
numbers, rather than importing Stage-1-specific constants, is what "reuses
the exact mechanism, re-keyed" means, S2.8 item 3).
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stage2_composer as sc
import stage2_instrument as si
import stage2_task as st
from groups import GROUP_NAMES, D_STATE, generating_set

PI_SIGNOFF_VAR = "CAPABILITY_SEP_STAGE2_PI_SIGNOFF"

# ---------------------------------------------------------------------------
# Grid construction (S2.5).
# ---------------------------------------------------------------------------

PRIMARY_SEEDS = 5
NH_GRID_SEEDS = 3
NH_GRID_GROUPS = ("S5", "A6")
NH_GRID_VALUES = (1, 2, 4)
ARM_BETA_MAX = {"arm2_beta01": 1.0, "arm3_beta02": 2.0}
ARMS = tuple(ARM_BETA_MAX)


def build_primary_grid() -> list[dict]:
    """5 groups x 2 beta-arms x n=5 seeds, n_h=2 default -- 50 cells."""
    cells = []
    for name in GROUP_NAMES:
        for arm in ARMS:
            for seed in range(PRIMARY_SEEDS):
                cells.append(dict(cell_id=f"{name}__{arm}__nh2__seed{seed}", group=name, arm=arm,
                                  n_h=2, seed=seed))
    assert len(cells) == 50, f"primary grid: expected 50 cells, got {len(cells)}"
    return cells


def build_nh_grid() -> list[dict]:
    """{S5,A6} x n_h in {1,2,4} x n=3 seeds, Arm 3 ONLY (S2.5 Rev 2 pin:
    the n_h axis tests DeltaProduct's published sufficiency claims, measured
    at allow_neg_eigval=True -- running it on Arm 2 would confound n_h with
    the beta-range exclusion Theorems 1/3 already predict binds Arm 2) --
    18 cells."""
    cells = []
    for name in NH_GRID_GROUPS:
        for n_h in NH_GRID_VALUES:
            for seed in range(NH_GRID_SEEDS):
                cells.append(dict(cell_id=f"{name}__arm3_beta02__nh{n_h}__seed{seed}", group=name,
                                  arm="arm3_beta02", n_h=n_h, seed=seed))
    assert len(cells) == 18, f"n_h grid: expected 18 cells, got {len(cells)}"
    return cells


def build_calibration_set(primary: list[dict], nh_grid: list[dict]) -> list[dict]:
    """11 cells: the 10 base (group,arm) cells at n_h=2/seed0, PLUS the
    promoted (S5, Arm3, n_h=4, seed0) cell -- drawn from the two grids
    above, zero incremental cells (S2.5/S2.8 item 2)."""
    base = [c for c in primary if c["seed"] == 0]
    assert len(base) == 10, f"base calibration set: expected 10, got {len(base)}"
    promoted = [c for c in nh_grid
               if c["group"] == "S5" and c["arm"] == "arm3_beta02" and c["n_h"] == 4 and c["seed"] == 0]
    assert len(promoted) == 1, f"promoted (S5,Arm3,n_h=4) calibration cell: expected 1, got {len(promoted)}"
    calib = base + promoted
    assert len(calib) == 11
    # S2.5 Rev 2 pin: the base (S5,Arm3,n_h=2) cell and the promoted
    # (S5,Arm3,n_h=4) cell must be DISTINCT (not a near-duplicate).
    s5_arm3_calib = [c for c in calib if c["group"] == "S5" and c["arm"] == "arm3_beta02"]
    assert len(s5_arm3_calib) == 2 and {c["n_h"] for c in s5_arm3_calib} == {2, 4}, (
        f"S5/Arm3 calibration cells must be the DISTINCT (n_h=2 base) and (n_h=4 promoted) pair, "
        f"got {s5_arm3_calib}"
    )
    return calib


# ---------------------------------------------------------------------------
# Budget guards (S2.7/S2.8 item 3).
# ---------------------------------------------------------------------------

PLANNING_RATE_LO, PLANNING_RATE_HI = 0.018, 0.054   # GPU-h/cell, S2.7 Rev 1 band
PER_CELL_ABORT_CEILING = round(PLANNING_RATE_HI * 1.5, 4)   # 0.081 GPU-h/cell
STAGE2_CAP = 25.0
N_CALIBRATION_CELLS = 11
N_TOTAL_CELLS = 68
N_REMAINING_AFTER_CALIBRATION = N_TOTAL_CELLS - N_CALIBRATION_CELLS   # 57


class PerCellBudgetAbort(RuntimeError):
    pass


class Stage2BudgetAbort(RuntimeError):
    pass


def check_per_cell_projection(elapsed_h: float, steps_done: int, steps_total: int) -> dict:
    """Per-cell circuit breaker: if the IN-PROGRESS cell's wall-clock rate
    projects past 1.5x the pricier end of the S2.7 planning band, hard-
    abort THIS cell before it burns further budget."""
    if steps_done <= 0:
        return dict(ok=True, projected=0.0)
    projected = round(elapsed_h * (steps_total / steps_done), 6)
    ok = projected <= PER_CELL_ABORT_CEILING
    result = dict(elapsed_h=elapsed_h, steps_done=steps_done, steps_total=steps_total,
                 projected=projected, ceiling=PER_CELL_ABORT_CEILING, ok=ok)
    if not ok:
        raise PerCellBudgetAbort(
            f"cell projected {projected:.4f} GPU-h exceeds the per-cell abort ceiling "
            f"{PER_CELL_ABORT_CEILING} GPU-h (1.5x the {PLANNING_RATE_HI} GPU-h/cell planning-"
            f"band ceiling, S2.7) at {steps_done}/{steps_total} steps -- HARD ABORT this cell. {result}"
        )
    return result


def check_stage2_sweep_projection(real_rate_per_cell: float,
                                  n_remaining: int = N_REMAINING_AFTER_CALIBRATION,
                                  spend_so_far: float = 0.0, cap: float = STAGE2_CAP) -> dict:
    """S2.8 item 3's timing pilot / circuit breaker: mirrors
    `budget_guard.py::check_base_sweep_projection`'s MECHANISM (project the
    remaining grid from the calibration cells' REAL measured rate,
    hard-abort before launching it if the projection exceeds the cap),
    re-keyed to Stage 2's own 25 GPU-h cap and 57-cell remainder (Stage 1's
    own module constants are specific to its 58-cell/arm-type structure and
    do not transfer, S2.0's file-ownership constraint)."""
    projected = round(spend_so_far + n_remaining * real_rate_per_cell, 4)
    ok = projected <= cap
    result = dict(real_rate_per_cell=real_rate_per_cell, n_remaining=n_remaining,
                 spend_so_far=spend_so_far, projected_total=projected, cap=cap, ok=ok)
    if not ok:
        raise Stage2BudgetAbort(
            f"Stage-2 sweep projection {projected:.2f} GPU-h exceeds the {cap:.1f} GPU-h cap at "
            f"real rate {real_rate_per_cell:.4f} GPU-h/cell -- HARD ABORT before spending the "
            f"remaining {n_remaining}-cell grid. Re-scope before relaunch. {result}"
        )
    return result


# ---------------------------------------------------------------------------
# Resume-safe cell running (mirrors run_capability_sep.py's exact
# atomic-write / output-validity convention, CLAUDE.md's "skip already-
# completed work by checking output validity, not just existence" rule).
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_KEYS = {"cell_id", "group", "arm", "n_h", "seed", "status"}


def cell_output_path(results_dir: str, cell_id: str) -> str:
    return os.path.join(results_dir, f"{cell_id}.json")


def is_valid_output(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not REQUIRED_OUTPUT_KEYS.issubset(d.keys()):
        return False
    if d["status"] == "completed":
        for key in ("D_test_results",):
            if key in d and d[key] is None:
                return False
    return True


def run_cell_resume_safe(cell: dict, results_dir: str, run_fn) -> dict:
    """`run_fn(cell) -> dict` performs the actual (build-time: synthetic;
    launch-time: real GPU) work. Atomic write via a .tmp + os.replace, the
    SAME crash-safety convention `run_capability_sep.py::run_cell_resume_safe`
    uses."""
    path = cell_output_path(results_dir, cell["cell_id"])
    if is_valid_output(path):
        with open(path) as f:
            print(f"  [{cell['cell_id']}] SKIP (valid output already on disk)")
            return json.load(f)
    result = run_fn(cell)
    os.makedirs(results_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp_path, path)
    return result


# ---------------------------------------------------------------------------
# One-cell orchestration: build the composer, train (build-time: a tiny
# synthetic loop; real launches pass a real step budget), run the
# query-dependence gate at calibration time, evaluate the D_test grid.
# NOT a real training loop -- this wires the pieces together and is
# exercised end-to-end only with `steps_override` tiny values in smoke.
# ---------------------------------------------------------------------------

def build_cell_composer(cell: dict, device="cpu") -> sc.GroupWordDeltaComposer:
    name = cell["group"]
    d_state, n_gens = D_STATE[name], len(generating_set(name))
    torch.manual_seed(cell["seed"])
    return sc.GroupWordDeltaComposer(d_state, n_gens, h=32, n_h=cell["n_h"],
                                     beta_max=ARM_BETA_MAX[cell["arm"]]).to(device)


def checkpoint_path(results_dir: str, cell_id: str) -> str:
    return os.path.join(results_dir, f"{cell_id}.pt")


def train_cell_tiny(cell: dict, results_dir: str, device="cpu", steps: int = 20,
                    budget_guard: bool = True) -> tuple[dict, sc.GroupWordDeltaComposer]:
    """A SMOKE-ONLY training loop (tiny step count, tiny batch) exercising
    the full wire-up (composer -> loss -> backward -> per-cell budget
    guard -> checkpoint persistence); NOT the real 8K+-step Stage-2
    training regime (out of BUILD scope). Real launches replace this
    function's body while keeping the same cell_output_path/checkpoint_path
    /is_valid_output/run_cell_resume_safe wrapper.

    UNLIKE Stage 1's own convention (`run_capability_sep.py::
    train_and_eval_cell` trains then discards the model, S1.33 M2's own
    disclosed build gap), Arms 2-3 DO need their trained state persisted:
    the last-K-window shortcut control (S2.9 item 4) is pinned as
    EVAL-TIME truncation of the EXISTING trained Arm-3 checkpoint, which
    requires the checkpoint to actually exist on disk after training
    completes. Returns (json-safe result dict, the trained composer) --
    the composer is ALSO what the calibration gate (S2.8 item 2(e)) must
    run against, never a freshly-reinitialized stand-in."""
    name = cell["group"]
    composer = build_cell_composer(cell, device=device)
    opt = torch.optim.Adam(composer.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(cell["seed"] + 1)

    t0 = time.time()
    final_loss = None
    for step in range(1, steps + 1):
        batch = st.sample_train_batch_stage2(name, 32, gen, device=device)
        Z = composer(batch["token_idx"])
        loss = torch.nn.functional.mse_loss(Z, batch["target"])  # any finite differentiable loss suffices for the wiring smoke
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
        if budget_guard and step % 5 == 0:
            elapsed_h = (time.time() - t0) / 3600.0
            check_per_cell_projection(elapsed_h, step, steps)
    wall_s = time.time() - t0

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = checkpoint_path(results_dir, cell["cell_id"])
    torch.save(composer.state_dict(), ckpt_path)

    result = dict(cell_id=cell["cell_id"], group=cell["group"], arm=cell["arm"], n_h=cell["n_h"],
                 seed=cell["seed"], status="completed", steps_completed=steps,
                 final_loss=final_loss, wall_clock_s=wall_s, checkpoint_path=ckpt_path)
    return result, composer


def load_cell_composer(cell: dict, results_dir: str, device="cpu") -> sc.GroupWordDeltaComposer:
    """Round-trip loader for the resume-skip path: rebuild the composer's
    ARCHITECTURE (deterministic from `cell`) and load its trained weights
    from the checkpoint `train_cell_tiny` persisted -- the SAME on-disk
    round-trip discipline `gate1_synthetic_injection.py` uses for its own
    dump/reload (write, then read back, before trusting downstream code)."""
    composer = build_cell_composer(cell, device=device)
    ckpt_path = checkpoint_path(results_dir, cell["cell_id"])
    composer.load_state_dict(torch.load(ckpt_path, map_location=device))
    return composer.to(device)


def run_calibration_gate_for_cell(cell: dict, composer: sc.GroupWordDeltaComposer,
                                  depths=si.PROBE_DEPTHS, device="cpu") -> dict:
    """S2.8 item 2(e): the query-dependence diagnostic on a trained (or, in
    smoke, an untrained-but-forward-capable) composer's own reader."""
    n_gens = composer.n_gens

    def real_state_fn(D):
        tok = si.build_probe_tokens(n_gens, D, device=device)
        composer.eval()
        with torch.no_grad():
            tok_embed = composer.embed_tokens(tok)
            states = composer.states_from_embedding(tok_embed)
        return states[-1]

    report = si.run_query_dependence_gate(
        composer.readout.row_queries, composer.readout.reader, cell["n_h"], real_state_fn,
        prepare_mem=composer.readout.prepare_mem, depths=depths, seed=si.PROBE_SEED, h=composer.h,
    )
    route = si.route_gate_result(report, bos_already_applied=composer.use_bos_row)
    return dict(cell_id=cell["cell_id"], report=report, route=route)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def run_calibration_wave(results_dir: str, device="cpu", steps: int = 20) -> list[dict]:
    primary, nh_grid = build_primary_grid(), build_nh_grid()
    calib = build_calibration_set(primary, nh_grid)
    results = []
    for cell in calib:
        def _run(c, _results_dir=results_dir):
            train_result, composer = train_cell_tiny(c, _results_dir, device=device, steps=steps)
            # gate runs on the ACTUALLY TRAINED composer (never a fresh
            # reinit -- that was a build-time bug caught and fixed before
            # this file was committed, see the git history / final report).
            gate_result = run_calibration_gate_for_cell(c, composer, depths=(1, 8, 64), device=device)
            train_result["gate_route"] = gate_result["route"]["route"]
            return train_result
        result = run_cell_resume_safe(cell, results_dir, _run)
        if "gate_route" not in result:
            # resumed from disk: the checkpoint exists, but the gate wasn't
            # re-run this process -- re-run it against the LOADED (not
            # fresh) composer so a resumed run still reports a gate route.
            composer = load_cell_composer(cell, results_dir, device=device)
            gate_result = run_calibration_gate_for_cell(cell, composer, depths=(1, 8, 64), device=device)
            result["gate_route"] = gate_result["route"]["route"]
        results.append(result)
    return results


def smoke():
    print("=" * 88)
    print("  stage2_run.py SMOKE -- grid construction, budget guards, resume-safety, calibration wave")
    print("=" * 88)

    primary, nh_grid = build_primary_grid(), build_nh_grid()
    calib = build_calibration_set(primary, nh_grid)
    total = len(primary) + len(nh_grid)
    print(f"  primary grid: {len(primary)} cells (expect 50)")
    print(f"  n_h grid: {len(nh_grid)} cells (expect 18)")
    print(f"  total new training cells: {total} (expect 68)")
    print(f"  calibration set: {len(calib)} cells (expect 11)")
    assert len(primary) == 50 and len(nh_grid) == 18 and total == 68 and len(calib) == 11

    print(f"\n  budget: PER_CELL_ABORT_CEILING={PER_CELL_ABORT_CEILING} GPU-h/cell "
          f"(1.5x {PLANNING_RATE_HI})")
    assert abs(PER_CELL_ABORT_CEILING - 0.081) < 1e-9

    print("\n  Stage-2 ledger projection (NEGATIVE-then-POSITIVE):")
    ok_proj = check_stage2_sweep_projection(real_rate_per_cell=0.03)
    print(f"    real_rate=0.03 GPU-h/cell -> projected={ok_proj['projected_total']} <= 25.0: PASS")
    raised = False
    try:
        check_stage2_sweep_projection(real_rate_per_cell=1.0)
    except Stage2BudgetAbort as e:
        raised = True
        print(f"    real_rate=1.0 GPU-h/cell -> Stage2BudgetAbort raised as expected: {str(e)[:90]}...")
    assert raised, "Stage2BudgetAbort has no teeth -- an absurd rate did not trip the ledger breaker"

    print("\n  per-cell budget guard (NEGATIVE-then-POSITIVE):")
    ok = check_per_cell_projection(elapsed_h=0.001, steps_done=10, steps_total=20)
    print(f"    small elapsed_h -> ok={ok['ok']} projected={ok['projected']}")
    assert ok["ok"]
    raised = False
    try:
        check_per_cell_projection(elapsed_h=1.0, steps_done=1, steps_total=20)
    except PerCellBudgetAbort as e:
        raised = True
        print(f"    huge elapsed_h at 1/20 steps -> PerCellBudgetAbort raised as expected: {str(e)[:90]}...")
    assert raised, "PerCellBudgetAbort has no teeth"

    print("\n  resume-safety (tiny synthetic cell, CPU, steps=6):")
    import tempfile
    with tempfile.TemporaryDirectory() as results_dir:
        cell = calib[0]

        def _run_and_ckpt(c, _rd=results_dir):
            result, _composer = train_cell_tiny(c, _rd, steps=6)
            return result

        t0 = time.time()
        r1 = run_cell_resume_safe(cell, results_dir, _run_and_ckpt)
        dt1 = time.time() - t0
        t0 = time.time()
        r2 = run_cell_resume_safe(cell, results_dir, _run_and_ckpt)
        dt2 = time.time() - t0
        print(f"    first run: {dt1:.3f}s  second run (resume/skip): {dt2:.3f}s")
        assert dt2 < dt1, "resume pass took as long as the first -- resume-safety broken"
        assert r1["cell_id"] == r2["cell_id"] == cell["cell_id"]
        assert os.path.exists(checkpoint_path(results_dir, cell["cell_id"])), \
            "training completed but no checkpoint was persisted -- the last-K-window control " \
            "(S2.9 item 4) needs this on disk"

        # corrupted-output re-run: truncated JSON must NOT be treated as done.
        path = cell_output_path(results_dir, cell["cell_id"])
        with open(path, "w") as f:
            f.write('{"cell_id": "' + cell["cell_id"] + '", "status": "completed"')  # truncated, invalid JSON
        assert not is_valid_output(path), "corrupted output was accepted as valid -- resume-safety has no teeth"
        r3 = run_cell_resume_safe(cell, results_dir, _run_and_ckpt)
        assert r3["status"] == "completed", "did not properly re-run after a corrupted output"
        print("    corrupted (truncated) output correctly triggers a re-run, not a false-skip  OK")

        print("\n  last-K-window eval-time reload from the JUST-PERSISTED checkpoint (S2.9 item 4):")
        reloaded = load_cell_composer(cell, results_dir)
        d_state = D_STATE[cell["group"]]
        tok = torch.randint(0, reloaded.n_gens, (4, 8))
        Z_full = reloaded(tok, reset_every=None)
        Z_trunc = reloaded(tok, reset_every=4)
        assert Z_full.shape == Z_trunc.shape == (4, d_state, d_state)
        assert not torch.allclose(Z_full, Z_trunc), \
            "reloaded checkpoint's truncated-eval read is identical to the full read -- suspicious"
        print("    reloaded composer forward()s correctly with and without reset_every  OK")

    print("\n  one full calibration-cell pipeline (composer train + query-dependence gate), CPU, tiny:")
    torch.manual_seed(0)
    cell = dict(cell_id="smoke__S3__arm3_beta02__nh2__seed0", group="S3", arm="arm3_beta02", n_h=2, seed=0)
    with tempfile.TemporaryDirectory() as results_dir2:
        train_result, trained_composer = train_cell_tiny(cell, results_dir2, steps=6)
        gate_result = run_calibration_gate_for_cell(cell, trained_composer, depths=(1, 8))
    print(f"    train status={train_result['status']}  gate route={gate_result['route']['route']}")
    assert train_result["status"] == "completed"
    assert gate_result["route"]["route"] in ("pass", "apply_bos_fix_rerun_all_11",
                                             "instrument_defect", "mechanism_diagnostic_required")

    print("\n  target-rank unit test (S1.33 [LEARN], invoked as a stage2_run smoke section):")
    st._test_target_rank_matches_necessity()

    print("\n" + "=" * 88 + "\n  stage2_run.py SMOKE PASSED\n" + "=" * 88)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibration-only", action="store_true")
    ap.add_argument("--results-dir", default="stage2_results")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    if os.environ.get(PI_SIGNOFF_VAR) != "1":
        raise RuntimeError(
            f"{PI_SIGNOFF_VAR}=1 required before any GPU cell (mirrors run_capability_sep.py's "
            f"own gate). Not set automatically by this script. This build agent does not launch "
            f"real cells -- an independent audit follows before any real dispatch."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_calibration_wave(args.results_dir, device=device, steps=20)
    print(f"calibration wave: {len(results)} cells run/skipped -> {args.results_dir}")
    if not args.calibration_only:
        print("real 57-cell remainder launch is NOT wired in this build -- gated on the "
              "calibration wave's own query-dependence PASS + real-rate re-derivation, "
              "S2.8 items 2-3.")


if __name__ == "__main__":
    main()
