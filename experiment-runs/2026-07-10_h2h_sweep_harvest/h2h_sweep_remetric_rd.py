#!/usr/bin/env python
"""h2h_sweep_remetric_rd.py -- harvest-side Leg-A/rung-2/Leg-B/S0 re-metric over the
27-cell sweep's 18 grammar checkpoints (task1/task2 x 3 arms x 3 seeds): sec 1.31.4
item 6's `run_cell_round4` applied VERBATIM to the sweep's `_r4.pt` checkpoints.
task3 (LM) cells carry no Leg-A metric (their verdict input is the anchored val-loss
band, already in the training JSONs) and are excluded by construction.

Why this exists: `h2h_sweep_stage_d.sh` ran "the 27 training cells ONLY" (its own
header); the sweep training JSONs carry only the demoted online-probe metrics, never
`acc_A`. Sec 1.31.1 pins "verdicts land at the sweep" on n=3 acc_A reads -- this
script produces exactly those reads, through the SAME audited round-4 driver, with
loader-side provenance pinning (md5 computed at manifest-build, checked at load) and
the pinned EVAL_SEED episode protocol (identical across arms/seeds by construction).

Run (box, from /home/nvidia/chapter2/deltanet_rd, tdenv python, GPU 0):
  CUDA_VISIBLE_DEVICES=0 python h2h_sweep_remetric_rd.py \
      --only h2h_contender_task1_sweep_s0        # pilot, one cell
  CUDA_VISIBLE_DEVICES=0 python h2h_sweep_remetric_rd.py  # all 18, resume-safe
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_round4_driver_rd import run_cell_round4, _md5_of_file        # noqa: E402
from h2h_sweep_runner_rd import build_27_cell_manifest                # noqa: E402


def build_sweep_remetric_specs(ckpt_dir: str):
    """18 grammar cells from the SAME 27-cell manifest the sweep trained from
    (one source of truth for names/seeds), each pointed at its _r4 checkpoint."""
    specs, manifest = [], {}
    for c in build_27_cell_manifest():
        if c["task"] == "task3_sweep":
            continue  # LM cell: no grammar Leg-A eval applies
        path = os.path.join(ckpt_dir, f"{c['name']}_r4.pt")
        assert os.path.isfile(path), f"missing sweep checkpoint: {path}"
        cell_id = c["name"]
        manifest[cell_id] = {"path": path, "md5": _md5_of_file(path),
                             "mtime": os.path.getmtime(path),
                             "arch": c["arch"], "task": c["task"]}
        # K=32 for BOTH tasks: task1_sweep trained at K=32 (sweep manifest);
        # task_cfg pins task2 to K=32 internally; matches ROUND4_CELL_SPEC's
        # task2_calib row (K=32) so rung2/instrument-health read identically.
        specs.append({"cell_id": cell_id, "arch": c["arch"], "task": c["task"],
                      "K": 32, "role": "sweep_remetric", "fresh": False,
                      "seed": c["seed"]})
    assert len(specs) == 18, f"expected 18 grammar sweep cells, got {len(specs)}"
    return specs, manifest


def _is_valid_out(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return "leg_a" in doc and "leg_b_ridge" in doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/data/h2h_rung1_ckpts")
    ap.add_argument("--out-dir", default="results/h2h_rung1/sweep_remetric")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="run just one cell_id (pilot)")
    args = ap.parse_args()

    specs, manifest = build_sweep_remetric_specs(args.ckpt_dir)
    if args.only:
        specs = [s for s in specs if s["cell_id"] == args.only]
        assert specs, f"--only {args.only!r} matched no sweep cell"
    os.makedirs(args.out_dir, exist_ok=True)

    for spec in specs:
        out_path = os.path.join(args.out_dir, f"{spec['cell_id']}_round4.json")
        if _is_valid_out(out_path):
            print(f"SKIP (already valid): {out_path}")
            continue
        r = run_cell_round4(spec, manifest, args.device, args.out_dir)
        s0 = r.get("s0_necessity_check")
        print(f"DONE {spec['cell_id']}: acc_A={r['leg_a']['acc_A']:.4f} "
              f"rung2={r['rung2_identity_classifier']['accuracy']:.4f} "
              f"rf@0.9={r['leg_b_ridge']['rf@0.9']:.4f} "
              f"s0_fires={s0['hard_stop_fires'] if s0 else 'n/a'} "
              f"instr={r['instrument_health']['passed']} wall={r['wall_s']:.1f}s",
              flush=True)
    print("SWEEP RE-METRIC COMPLETE")


if __name__ == "__main__":
    main()
