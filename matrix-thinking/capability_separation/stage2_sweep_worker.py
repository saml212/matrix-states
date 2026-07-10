"""S2.30: the 57-cell-remainder SWEEP WORKER -- the minimal launch wiring the
§2.24/§2.26 chain deferred until the calibration wave's own readout existed
(stage2_run.py::main's own text: "real 57-cell remainder launch is NOT wired
in this build -- gated on the calibration wave's own query-dependence PASS +
real-rate re-derivation, S2.8 items 2-3"). Both gates are now discharged on
the record (§2.30: 2(e) route=pass 11/11 at all 7 depths; measured rate
0.0433 GPU-h/cell, in-band, 57-cell projection 2.47 GPU-h vs the 25 GPU-h
cap) -- this worker still RE-DERIVES the rate from the calibration cell JSONs
on disk and re-runs `check_stage2_sweep_projection` at startup (the code's
own breaker, never bypassed), plus the same PI-signoff env gate as every
other real path.

Composition, not new mechanics: every load-bearing component is the audited
one -- `build_primary_grid`/`build_nh_grid`/`build_calibration_set` (§2.19+),
`run_cell_resume_safe(strict_real=True)` (§2.22 N1), `run_real_cell`
(§2.20 F4, with the §2.28 M-D0 exclusions and the §2.29 anchor-scaled
per-cell breaker), `check_stage2_sweep_projection` (§2.8 item 3).

Sharding: cells are deduped by cell_id (FIRST occurrence kept -- the §2.29
audit's incidental finding: `build_primary_grid` and `build_nh_grid` emit 6
IDENTICAL (S5/A6, arm3, n_h=2, seed 0-2) configs under colliding cell_ids;
they are the same experiment, run once -- 68 objects == 62 distinct ids ==
11 calibration + 51 remainder; the ledger projection conservatively still
prices 57), sorted by cell_id, then round-robin sharded: shard k of N runs
`cells[k::N]`. Shards are disjoint by cell_id, outputs are per-cell files
written atomically (tmp + os.replace), checkpoints are per-cell names -- a
shared results_dir across concurrent workers is race-free.

Failure policy (CLAUDE.md sweep rule: one crash must not kill the remaining
configs): per-cell try/except logs the failure and continues the shard; the
worker exits nonzero iff any cell failed, so the supervisor loop retries
(resume-safe: completed cells SKIP) with its own bail cap.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import stage2_run as sr


def measured_calibration_rate(results_dir: str, calib_ids: list[str]) -> float:
    """Re-derive the real GPU-h/cell rate from the 11 calibration cells'
    own wall_clock_s on disk (S2.8 item 3's 'REAL measured rate') -- fails
    loudly if any calibration output is missing/invalid rather than
    silently proceeding on a partial basis."""
    total_s, n = 0.0, 0
    for cid in calib_ids:
        path = sr.cell_output_path(results_dir, cid)
        if not sr.is_valid_output(path, strict_real=True):
            raise RuntimeError(
                f"S2.30 rate re-derivation: calibration cell {cid!r} has no valid real output at "
                f"{path} -- the sweep may not launch without the full 11-cell calibration basis"
            )
        with open(path) as f:
            total_s += float(json.load(f)["wall_clock_s"])
        n += 1
    return (total_s / 3600.0) / n


def build_remainder(calib_ids: set[str]) -> list[dict]:
    """The full grid, deduped by cell_id (first occurrence), minus the 11
    calibration cells, deterministically ordered."""
    seen: dict[str, dict] = {}
    for cell in sr.build_primary_grid() + sr.build_nh_grid():
        seen.setdefault(cell["cell_id"], cell)
    remainder = [c for cid, c in sorted(seen.items()) if cid not in calib_ids]
    return remainder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--results-dir", default="stage2_results")
    args = ap.parse_args()
    assert 0 <= args.shard < args.nshards

    if os.environ.get(sr.PI_SIGNOFF_VAR) != "1":
        raise RuntimeError(f"{sr.PI_SIGNOFF_VAR}=1 required (S2.30 sweep launch, citing "
                           f"S2.24/S2.26/S2.28/S2.29/S2.30)")

    primary, nh_grid = sr.build_primary_grid(), sr.build_nh_grid()
    calib_ids = [c["cell_id"] for c in sr.build_calibration_set(primary, nh_grid)]

    # S2.8 item 3: the ledger breaker at the REAL re-derived rate, every worker,
    # before any cell runs. n_remaining stays the registry's conservative 57.
    rate = measured_calibration_rate(args.results_dir, calib_ids)
    proj = sr.check_stage2_sweep_projection(real_rate_per_cell=rate)
    print(f"[shard {args.shard}/{args.nshards}] measured rate {rate:.4f} GPU-h/cell -> "
          f"projection {proj['projected_total']} GPU-h (cap {sr.STAGE2_CAP}) OK")

    remainder = build_remainder(set(calib_ids))
    shard_cells = remainder[args.shard::args.nshards]
    print(f"[shard {args.shard}/{args.nshards}] {len(remainder)} distinct remainder cells total; "
          f"this shard: {len(shard_cells)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    for cell in shard_cells:
        try:
            result = sr.run_cell_resume_safe(
                cell, args.results_dir,
                lambda c: sr.run_real_cell(c, args.results_dir, device=device, steps=None),
                strict_real=True,
            )
            print(f"[shard {args.shard}] {cell['cell_id']}: {result.get('status')} "
                  f"gate_route={result.get('gate_route')}")
        except Exception:
            failures.append(cell["cell_id"])
            print(f"[shard {args.shard}] {cell['cell_id']}: FAILED (continuing shard; "
                  f"supervisor will retry)")
            traceback.print_exc()

    if failures:
        print(f"[shard {args.shard}] {len(failures)} cell(s) failed: {failures}")
        sys.exit(1)
    print(f"[shard {args.shard}] SHARD COMPLETE ({len(shard_cells)} cells valid on disk)")


if __name__ == "__main__":
    main()
