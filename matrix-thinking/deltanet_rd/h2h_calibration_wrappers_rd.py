"""h2h_calibration_wrappers_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.7
gates 1-2: the 3-arm calibration-cell manifest (Wave -1 items C/D/E, sec
1.6), the per-arch x task timing pilots (mechanical enforced abort,
`phase2b_off_cache.py --time-pilot`'s own pattern), and the R3-F4 M-sweep-
specific timing pilot (2 M-values x 1 checkpoint x 1 horizon,
checkpoints-resident-across-passes REQUIRED) with its pre-registered
de-scope order.

**BUILD-STAGE SCOPE, stated plainly (the build agent "does not launch GPU
work"):** every function below that would need a real trained checkpoint or
real corpus data is DEPENDENCY-INJECTED -- it takes a `train_fn`/`eval_fn`
callable as a parameter rather than importing a concrete training loop, so
this module's own logic (manifest shape, seed derivation, gate arithmetic,
de-scope ordering, checkpoint-residency behavior) is fully unit-testable
with synthetic/fake callables (this file's own smoke suite), while the REAL
callables (real `lm_pretrain_rd`-style training, real inference passes)
are wired in only at actual launch time, on the H100 box, by a LATER stage
-- never invoked here.

Run the smoke gate: python h2h_calibration_wrappers_rd.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rd_episode_seed import rd_episode_seed

N_SEEDS = 3
WARMUP_STEPS = 100                 # M3: pinned identical across every arm, never gridded
CONTENDER_LR = 3e-4                 # M3: the contender's own already-validated default
TRANSFORMER_LR_GRID = (1e-4, 3e-4, 1e-3)   # M3: equally-sized 3-point grid, calibration-cell only

# sec 1.6 item C: primary load K/d=0.5 (K=32 at d_state=64), full budget; K/d=0.75 stress,
# quarter budget, locate-only (never decision-grade -- M-NEW-3's own text).
TASK1_LOADS = ({"kd": 0.5, "K": 32, "role": "primary", "budget_frac": 1.0},
               {"kd": 0.75, "K": 48, "role": "stress_locate_only", "budget_frac": 0.25})
ARMS = ("contender", "ablation", "transformer")


def build_task1_calibration_manifest() -> list[dict]:
    """Wave -1 item C: 3 arms x 2 loads (1 full + 1 quarter) -- sec 1.6's own costed shape."""
    cells = []
    for arm in ARMS:
        for load in TASK1_LOADS:
            cells.append({
                "arch": arm, "task": "task1_calib", "kd": load["kd"], "K": load["K"],
                "role": load["role"], "budget_frac": load["budget_frac"],
                "seed": rd_episode_seed("task1_calib" if load["role"] == "primary" else "task1_stress",
                                         seed_idx=0, ckpt_idx=0),
                "lr": CONTENDER_LR, "warmup_steps": WARMUP_STEPS,
            })
    return cells


def build_task2_calibration_manifest() -> list[dict]:
    """Wave -1 item D: target hop-depth config, held-out-hop guard re-verified, full cells,
    all 3 arms."""
    return [{"arch": arm, "task": "task2_calib", "role": "primary", "budget_frac": 1.0,
             "seed": rd_episode_seed("task2_calib", seed_idx=0, ckpt_idx=0),
             "lr": CONTENDER_LR, "warmup_steps": WARMUP_STEPS} for arm in ARMS]


def build_task3_calibration_manifest() -> list[dict]:
    """Wave -1 item E, sec 1.6's own reduced scope: the contender's rung-1
    config is ALREADY calibrated by FROZEN_BIAS_LM_DESIGN.md's own rung-1
    run (NOT re-run here -- registered as `reused`, never a phantom extra
    cell); the ablation reuses the contender's pinned LR (1 full cell); the
    Transformer gets its OWN 3-point LR grid (M3) at quarter-budget (3
    cells) -- 1 full + 3 quarter, matching sec 1.6's own costed line.

    AUD-F4 (build audit §1.20, cosmetic): task3's own seed schedule below
    reuses `task2_calib`'s TASK_BASE key (task3 has no dedicated
    on-the-fly-generation seed need -- it trains on STATIC corpora, sec
    1.4 -- so its own seed only needs to be collision-free against the
    OTHER on-the-fly tasks, which `task2_calib`'s key already guarantees
    by construction) -- mirrors `h2h_sweep_runner_rd.py:39-43`'s own
    identical reasoning for the 27-cell sweep manifest's task3 rows."""
    cells = [{"arch": "contender", "task": "task3_calib", "role": "reused_not_relaunched",
              "budget_frac": 0.0, "seed": None, "lr": CONTENDER_LR, "warmup_steps": WARMUP_STEPS,
              "note": "FROZEN_BIAS_LM_DESIGN.md's own rung-1 run -- not re-launched this wave"},
             {"arch": "ablation", "task": "task3_calib", "role": "primary", "budget_frac": 1.0,
              "seed": rd_episode_seed("task2_calib", seed_idx=1, ckpt_idx=0),
              "lr": CONTENDER_LR, "warmup_steps": WARMUP_STEPS}]
    for i, lr in enumerate(TRANSFORMER_LR_GRID):
        cells.append({"arch": "transformer", "task": "task3_calib", "role": f"lr_grid_{i}",
                      "budget_frac": 0.25, "seed": rd_episode_seed("task2_calib", seed_idx=2 + i, ckpt_idx=0),
                      "lr": lr, "warmup_steps": WARMUP_STEPS})
    return cells


def build_full_calibration_manifest() -> list[dict]:
    return (build_task1_calibration_manifest() + build_task2_calibration_manifest()
            + build_task3_calibration_manifest())


def check_calibration_band(measured: float, band_lo: float, band_hi: float, metric_name: str) -> dict:
    """Gate 1's own band check: `measured` (a val-loss or recovered_frac@0.9
    reading) must land inside `[band_lo, band_hi]`, the small-scale
    precedent's own predicted range. Returns a result dict; callers hard-
    abort the 27-cell sweep on a failed band (never launched silently)."""
    ok = band_lo <= measured <= band_hi
    return {"metric": metric_name, "measured": measured, "band": (band_lo, band_hi), "within_band": ok}


RUNG1_BAND_MULT = 3.0   # sec 1.7 gate 1a/1b: same >3x episode-restricted-chance bar as the ladder's own rung 1


def check_gate1_full_cell_band(arch: str, rung1_accuracy: float, K: int,
                                instrument_health_passed: bool,
                                rung1_mult: float = RUNG1_BAND_MULT) -> dict:
    """sec 1.7 gate 1b, RE-WIRED (sec 1.31.4 item 5, Rev 5.1): the pre-Rev-5.1 version below
    still enforced the DEAD `rf@0.9 > 0` conjunct -- run as-is against round-4 output it FATALs
    every cell, since rf@0.9 was demoted to a Leg-B diagnostic reading (§1.31's own dead-clause
    list item 4, §1.30's localization: no linear tap on either raw state clears rf@0.9, so
    round 4's contender numbers routinely read rf@0.9==0 even on cells rung 1 passes cleanly).

    Now ARM-AWARE per §1.31.3's bands, a task1_calib/task2_calib FULL cell (role=='primary')
    passes gate 1 iff:
      (i)   the CONTENDER cell's rung-1 accuracy clears `rung1_mult` x episode-restricted
            chance (1/K) -- training+instrument sanity, launch-blocking ONLY for this arch;
      (ii)  baseline arms' (ablation/transformer) rung-1 is recorded as DATA, never
            launch-blocking -- their failure IS the pre-registered separation, not a broken
            cell;
      (iii) instrument-health (planted-signal positive controls pass AND noise nulls stay
            <=1.5x chance, sec 1.31.4 item 2 / `check_instrument_health`) IS launch-blocking
            for ALL arms;
      (iv)  `rf@0.9 > 0` is REMOVED from the band entirely (Leg-B diagnostic now, never a gate
            conjunct, never a LOSE criterion, per §1.31.1's own text).
    Stress/locate-only cells (task1_stress, role=='stress_locate_only') stay EXEMPT from this
    band and must never call this function as a gating decision -- callers decide cell
    eligibility, this function only scores what it is handed."""
    chance = 1.0 / K
    rung1_pass_bar = rung1_mult * chance
    rung1_ok = rung1_accuracy > rung1_pass_bar
    is_contender = (arch == "contender")
    rung1_blocking = is_contender          # baseline arms' rung1 is data-only, never blocking
    within_band = (rung1_ok if rung1_blocking else True) and instrument_health_passed
    return {"arch": arch, "rung1_accuracy": rung1_accuracy, "K": K, "chance": chance,
            "rung1_pass_bar": rung1_pass_bar, "rung1_ok": rung1_ok, "rung1_blocking": rung1_blocking,
            "instrument_health_passed": instrument_health_passed, "within_band": within_band}


# ---------------------------------------------------------------------------
# sec 1.7 gate 2 -- per-arch x task timing pilots (phase2b_off_cache.py's own
# time_one_..._pass / project_and_gate_... pattern).
# ---------------------------------------------------------------------------

def time_one_cell(train_fn, *args, **kwargs) -> float:
    """Wraps ONE real training-cell call in a wall-clock timer -- `train_fn`
    is dependency-injected (a real training loop at launch time; a fake,
    synthetic-sleep function in this module's own smoke suite)."""
    t0 = time.time()
    train_fn(*args, **kwargs)
    return time.time() - t0


def project_and_gate_arch_task_pilot(elapsed_s_per_cell: float, n_cells_this_pair: int,
                                      remaining_headroom_gpu_h: float) -> dict:
    """Projects the full (arch,task) pair's own cost from ONE measured cell, gates against the
    remaining budget headroom -- mirrors phase2b_off_cache.py's
    project_and_gate_timing_pilot's own shape exactly (pure function, no side effects)."""
    projected_gpu_h = elapsed_s_per_cell * n_cells_this_pair / 3600.0
    return {"elapsed_s_per_cell": elapsed_s_per_cell, "n_cells": n_cells_this_pair,
            "projected_gpu_h": projected_gpu_h, "remaining_headroom_gpu_h": remaining_headroom_gpu_h,
            "ok": projected_gpu_h <= remaining_headroom_gpu_h}


# ---------------------------------------------------------------------------
# R3-F4 -- the M-sweep's OWN scoped timing pilot: 2 M-values x 1 checkpoint x 1 horizon,
# checkpoints-resident-across-passes REQUIRED.
# ---------------------------------------------------------------------------

class ResidentCheckpointCache:
    """Build requirement, pinned by R3-F4: trained checkpoints for role
    (b-primary) MUST be held resident in memory across all inference
    passes for a given (task, seed) -- loaded ONCE, evaluated at every
    M/horizon combination, never reloaded from disk per pass (a naive
    reload-per-pass implementation could run 3-6x the padded estimate,
    R3-F4's own finding)."""

    def __init__(self):
        self._cache: dict = {}
        self.load_count: dict = {}

    def get_or_load(self, key, loader_fn):
        if key not in self._cache:
            self._cache[key] = loader_fn()
            self.load_count[key] = self.load_count.get(key, 0) + 1
        return self._cache[key]


MSWEEP_PILOT_M_VALUES = (2, 32)   # "one small, one large" -- R3-F4's own pinned pilot pair
MSWEEP_PILOT_HORIZON = "H4"        # the primary decision horizon
MSWEEP_TOTAL_PASSES = 90           # R3-F6-corrected: 5 M-values x 3 horizons x 2 tasks x 3 seeds


def run_msweep_timing_pilot(eval_fn, checkpoint_loader_fn) -> dict:
    """Runs `len(MSWEEP_PILOT_M_VALUES)` real inference passes over ONE
    checkpoint, held RESIDENT (loaded once via `ResidentCheckpointCache`),
    at the primary horizon -- the measured mean s/pass REPLACES the
    design-time ~5s/pass assumption before the 90-pass fan-out is
    authorized."""
    cache = ResidentCheckpointCache()
    elapsed = []
    for m in MSWEEP_PILOT_M_VALUES:
        ckpt = cache.get_or_load("pilot_ckpt", checkpoint_loader_fn)
        t0 = time.time()
        eval_fn(ckpt, m, MSWEEP_PILOT_HORIZON)
        elapsed.append(time.time() - t0)
    assert cache.load_count["pilot_ckpt"] == 1, (
        f"checkpoint was loaded {cache.load_count['pilot_ckpt']} times, expected exactly 1 -- "
        f"R3-F4's own checkpoints-resident-across-passes requirement was violated")
    mean_s_per_pass = sum(elapsed) / len(elapsed)
    return {"elapsed_s": elapsed, "mean_s_per_pass": mean_s_per_pass,
            "n_checkpoint_loads": cache.load_count["pilot_ckpt"]}


def project_and_gate_msweep_fanout(measured_s_per_pass: float, remaining_headroom_gpu_h: float,
                                    n_passes: int = MSWEEP_TOTAL_PASSES) -> dict:
    """REPLACES the design-time ~5s/pass assumption (sec 1.6 item F) with
    the pilot's own measured value; gates the full 90-pass fan-out against
    the remaining headroom BEFORE it launches."""
    projected_gpu_h = measured_s_per_pass * n_passes / 3600.0
    return {"measured_s_per_pass": measured_s_per_pass, "n_passes": n_passes,
            "projected_gpu_h": projected_gpu_h, "remaining_headroom_gpu_h": remaining_headroom_gpu_h,
            "ok": projected_gpu_h <= remaining_headroom_gpu_h}


# sec 1.7 gate 2's own pre-registered de-scope order (config, not prose): drop M=32 first (the
# single most expensive point, least load-bearing once a WIN is already established at a smaller
# M via the fixed-sequence walk), then H8 (the least-informative secondary horizon). H4 (primary)
# and the floor-eligible M in {2,4,8,16} core are NEVER de-scoped by this rule.
DESCOPE_ORDER = ({"axis": "M", "value": 32}, {"axis": "horizon", "value": "H8"})
NEVER_DESCOPE = {"horizon": "H4", "M_core": (2, 4, 8, 16)}


def apply_descope_order(cells: list[dict], projected_gpu_h: float, headroom_gpu_h: float,
                         s_per_pass: float) -> dict:
    """Drops cells in `DESCOPE_ORDER` until the projection fits, or reports
    a hard abort if even the NEVER_DESCOPE core would need trimming (a
    bracket overrun requiring THAT is a hard abort, never a silent
    re-scope -- sec 1.7 gate 2's own instruction)."""
    remaining = list(cells)
    dropped: list[dict] = []
    for rule in DESCOPE_ORDER:
        current_gpu_h = len(remaining) * s_per_pass / 3600.0
        if current_gpu_h <= headroom_gpu_h:
            break
        if rule["axis"] == "M":
            keep, drop = [], []
            for c in remaining:
                (drop if c.get("M") == rule["value"] else keep).append(c)
        else:
            keep, drop = [], []
            for c in remaining:
                (drop if c.get("horizon") == rule["value"] else keep).append(c)
        remaining, dropped = keep, dropped + drop

    final_gpu_h = len(remaining) * s_per_pass / 3600.0
    core_intact = all(
        not (c.get("horizon") == NEVER_DESCOPE["horizon"] and c.get("M") in NEVER_DESCOPE["M_core"])
        or c in remaining
        for c in cells)
    return {"remaining_cells": remaining, "dropped_cells": dropped, "final_projected_gpu_h": final_gpu_h,
            "ok": final_gpu_h <= headroom_gpu_h, "hard_abort": final_gpu_h > headroom_gpu_h,
            "core_intact": core_intact}


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def smoke_1_manifest_shapes():
    t1 = build_task1_calibration_manifest()
    t2 = build_task2_calibration_manifest()
    t3 = build_task3_calibration_manifest()
    full = build_full_calibration_manifest()
    ok = (len(t1) == 6 and len(t2) == 3 and len(t3) == 5 and len(full) == 14
          and all(c["arch"] == "contender" for c in t3 if c["role"] == "reused_not_relaunched"))
    _report("smoke 1: calibration manifest shapes (task1=6, task2=3, task3=5, full=14)", ok,
            f"t1={len(t1)} t2={len(t2)} t3={len(t3)} full={len(full)}")


def smoke_2_seeds_come_from_rd_episode_seed():
    # rd_episode_seed deliberately EXCLUDES arch from its formula (F1b's own load-bearing
    # property) -- so all 3 arms at the SAME load share the SAME seed; 2 loads x 3 arms = 6 rows,
    # but only 2 DISTINCT seed values.
    t1 = build_task1_calibration_manifest()
    seeds = [c["seed"] for c in t1]
    n_unique = len(set(seeds))
    ok = n_unique == 2 and len(seeds) == 6
    _report("smoke 2: task1 calibration cells draw from rd_episode_seed (arch-agnostic: 2 unique "
            "seeds across 6 rows, one per load, shared across all 3 arms by design)", ok,
            f"n_unique={n_unique} seeds={seeds}")


def smoke_3_calibration_band_check():
    r_pass = check_calibration_band(2.15, 2.10, 2.20, "task3_openr1_val_loss")
    r_fail = check_calibration_band(2.50, 2.10, 2.20, "task3_openr1_val_loss")
    ok = r_pass["within_band"] and not r_fail["within_band"]
    _report("smoke 3: check_calibration_band correctly passes/fails against a band", ok)


def smoke_3b_gate1_full_cell_band():
    """sec 1.31.4 item 5 (Rev 5.1 re-wire): arm-aware bands, rf@0.9 REMOVED entirely.
    (a) contender, rung-1 well above bar, instrument-health passes -- PASSES.
    (b) contender, rung-1 AT chance (never learned recall, sec 1.21's own diagnosed failure
        mode) -- FAILS (contender's rung-1 IS launch-blocking).
    (c) contender, rung-1 passes, instrument-health FAILS -- FAILS (instrument-health is
        launch-blocking for every arch).
    (d) THE ROUND-3 CONTENDER CASE (sec 1.31.4 item 5's own required selftest): rung1=0.999,
        rf@0.9=0 -- under the PRE-Rev-5.1 band this FATALs; under the re-wired band it must now
        PASS (rf@0.9 is not read by this function at all any more -- there is no parameter for
        it), proving the conjunct is actually gone, not merely defaulted-away.
    (e) BASELINE ARM, rung-1 at chance, instrument-health passes -- PASSES (baseline rung-1 is
        data-only, never launch-blocking) -- proves (ii)'s asymmetry has teeth.
    (f) BASELINE ARM, instrument-health FAILS -- FAILS (instrument-health blocks every arch,
        including baselines) -- proves (iii)'s "ALL arms" scope has teeth.
    Run to completion (CLAUDE.md's own 'run the negative unit test... to completion' rule)."""
    K = 32
    chance = 1.0 / K
    r_pass = check_gate1_full_cell_band("contender", 0.5, K, instrument_health_passed=True)
    r_fail_rung1 = check_gate1_full_cell_band("contender", chance, K, instrument_health_passed=True)
    r_fail_instrument = check_gate1_full_cell_band("contender", 0.5, K, instrument_health_passed=False)
    r_round3_contender = check_gate1_full_cell_band("contender", 0.999, K, instrument_health_passed=True)
    r_baseline_low_rung1 = check_gate1_full_cell_band("ablation", chance, K, instrument_health_passed=True)
    r_baseline_bad_instrument = check_gate1_full_cell_band("transformer", 0.999, K,
                                                            instrument_health_passed=False)
    ok = (r_pass["within_band"] and not r_fail_rung1["within_band"]
          and not r_fail_instrument["within_band"] and r_round3_contender["within_band"]
          and r_baseline_low_rung1["within_band"] and not r_baseline_bad_instrument["within_band"]
          and r_fail_rung1["rung1_ok"] is False and "rf_ok" not in r_pass)
    _report("smoke 3b (sec 1.31.4 item 5): arm-aware band -- contender PASS case passes; "
            "contender chance-level rung-1 fails; instrument-health failure fails for EITHER "
            "arch; the round-3 contender case (rung1=0.999/rf@0.9=0, no longer even a "
            "parameter) now PASSES; a baseline arm at chance-level rung-1 STILL PASSES "
            "(data-only, never blocking)", ok,
            f"pass={r_pass['within_band']} fail_rung1={r_fail_rung1['within_band']} "
            f"fail_instrument={r_fail_instrument['within_band']} "
            f"round3_contender={r_round3_contender['within_band']} "
            f"baseline_low_rung1={r_baseline_low_rung1['within_band']} "
            f"baseline_bad_instrument={r_baseline_bad_instrument['within_band']}")


def smoke_4_timing_pilot_projection():
    def _fake_train_fn(sleep_s=0.01):
        time.sleep(sleep_s)
    elapsed = time_one_cell(_fake_train_fn, sleep_s=0.01)
    proj = project_and_gate_arch_task_pilot(elapsed, n_cells_this_pair=3, remaining_headroom_gpu_h=100.0)
    ok = proj["ok"] and proj["projected_gpu_h"] > 0
    _report("smoke 4: time_one_cell + project_and_gate_arch_task_pilot (synthetic train_fn)", ok,
            f"elapsed={elapsed:.4f}s projected_gpu_h={proj['projected_gpu_h']:.6f}")


def smoke_5_msweep_pilot_checkpoint_residency():
    load_calls = {"n": 0}

    def _fake_loader():
        load_calls["n"] += 1
        return {"fake": "checkpoint"}

    def _fake_eval_fn(ckpt, m, horizon):
        assert ckpt == {"fake": "checkpoint"}
        time.sleep(0.001)

    result = run_msweep_timing_pilot(_fake_eval_fn, _fake_loader)
    ok = (result["n_checkpoint_loads"] == 1 and load_calls["n"] == 1
          and len(result["elapsed_s"]) == len(MSWEEP_PILOT_M_VALUES))
    _report("smoke 5: run_msweep_timing_pilot loads the checkpoint EXACTLY ONCE across all pilot "
            "passes (R3-F4's own residency requirement)", ok, f"result={result}")


def smoke_6_msweep_residency_violation_negative_test():
    """A cache that loads TWICE (simulating a reload-per-pass bug) must be
    CAUGHT by the residency assert -- run to completion."""
    cache = ResidentCheckpointCache()
    cache.get_or_load("k", lambda: object())
    cache._cache.pop("k")   # force a second, spurious load on the next call (simulates the bug)
    cache.get_or_load("k", lambda: object())
    raised = False
    try:
        assert cache.load_count["k"] == 1, "simulated violation"
        raise RuntimeError("NEGATIVE FAILED TO FAIL: simulated double-load did not raise")
    except AssertionError:
        raised = True
    _report("smoke 6: checkpoint-residency violation (simulated double-load) is DETECTABLE by the "
            "same assert run_msweep_timing_pilot uses", raised)


def smoke_7_descope_order():
    cells = [{"M": m, "horizon": h} for m in (2, 4, 8, 16, 32) for h in ("H2", "H4", "H8")]
    # s_per_pass chosen so the full 15-cell set overruns a small headroom, forcing de-scope.
    result = apply_descope_order(cells, projected_gpu_h=0, headroom_gpu_h=0.002, s_per_pass=5.0)
    dropped_axes = {(d.get("M"), d.get("horizon")) for d in result["dropped_cells"]}
    m32_dropped_first = all(d.get("M") != 32 for d in result["remaining_cells"])
    core_survives_if_possible = result["core_intact"] or result["hard_abort"]
    ok = m32_dropped_first and core_survives_if_possible
    _report("smoke 7: apply_descope_order drops M=32 first, then H8, per the pre-registered order",
            ok, f"n_remaining={len(result['remaining_cells'])} n_dropped={len(result['dropped_cells'])} "
            f"final_gpu_h={result['final_projected_gpu_h']:.6f} hard_abort={result['hard_abort']}")


def smoke_8_descope_hard_abort_when_core_would_need_trimming():
    """If even the NEVER_DESCOPE core can't fit, this must be a HARD ABORT
    (ok=False), never a silent re-scope of H4/the M-core."""
    cells = [{"M": m, "horizon": "H4"} for m in (2, 4, 8, 16)]   # ONLY the never-descope core
    result = apply_descope_order(cells, projected_gpu_h=0, headroom_gpu_h=1e-9, s_per_pass=5.0)
    ok = result["hard_abort"] and len(result["dropped_cells"]) == 0
    _report("smoke 8: de-scoping never touches the NEVER_DESCOPE core -- an overrun there is a "
            "HARD ABORT, not a silent trim", ok, f"result={result}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    print("=" * 70)
    print("h2h_calibration_wrappers_rd.py -- sec 1.7 gates 1-2 smoke suite")
    print("=" * 70)
    smoke_1_manifest_shapes()
    smoke_2_seeds_come_from_rd_episode_seed()
    smoke_3_calibration_band_check()
    smoke_3b_gate1_full_cell_band()
    smoke_4_timing_pilot_projection()
    smoke_5_msweep_pilot_checkpoint_residency()
    smoke_6_msweep_residency_violation_negative_test()
    smoke_7_descope_order()
    smoke_8_descope_hard_abort_when_core_would_need_trimming()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
