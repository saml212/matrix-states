#!/usr/bin/env python
"""h2h_fix5_lrgrid_rd.py -- flagship rebuttal FIX-5 runner
(papers/flagship/gauntlet/round-1/04_rebuttal_report.md, FIX-5): a
learning-rate search for the transformer baseline on Task 1 (episodic
recall) under the identical frozen H2H protocol -- the pre-registered fix
for A1's residual (the round-3/4 sweep's transformer arm on the recall
task trained at the untuned shared default lr=3e-4; it was never
LR-searched there, only on the LM control task, sec 1.31's task3_calib
TRANSFORMER_LR_GRID).

Pre-registration: HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.44 (written and
committed BEFORE this script ran on real GPU cells).

Grid: {1e-4, 3e-4, 1e-3, 3e-3} x seed_idx {0,1,2} x 20,000 steps
(FULL_STEPS, the frozen sweep's own full budget). The 3e-4 x {0,1,2}
column is REUSED verbatim from the already-completed round-4 27-cell
sweep (transformer x task1_sweep x s{0,1,2}) -- byte-identical protocol,
task, K, steps, seed schedule -- never re-launched; only the other 3 LR
columns (9 cells) train fresh here.

Two-stage protocol per fresh cell, IDENTICAL to how every other h2h
number in this campaign was produced (sec 1.31.4 item 6 / sec 1.42's own
task2diag precedent): (1) TRAIN via h2h_cell_train_rd.train_grammar_cell
with role="sweep" (the AUD2-F4 structural dial guard -- role="sweep" is
the ONLY role that skips the mid-training three-loss dial check,
avoiding a spurious DialExhausted abort on a plain LR-grid cell),
persisting the raw JSON WITH its training curve; (2) RE-METRIC via
h2h_round4_driver_rd.run_cell_round4 (fresh=False, loading the
just-trained, provenance-pinned checkpoint) for the audited acc_A -- the
SAME function that produced every other acc_A number in this design doc,
including the reused 3e-4 column.

Box (from /home/nvidia/chapter2/deltanet_rd, tdenv python, GPUs 0-1 via
the stage script h2h_fix5_stage.sh -- tmux h2h_fix5_grid). --harvest is
torch-free (pure JSON math) and is meant to also run locally off
box-downloaded artifacts.

Run the manifest/decision-rule selftest: python h2h_fix5_lrgrid_rd.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rd_episode_seed import rd_episode_seed                            # noqa: E402

FIX5_LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3)     # rebuttal FIX-5's own pinned span
FIX5_REUSED_LR = 3e-4                        # already covered by the round-4 sweep, verbatim
FIX5_SEED_IDXS = (0, 1, 2)                   # n=3, sec 1.8's standing convention
FIX5_TASK = "task1_sweep"                    # SAME task string as the reused cells (K=32 via task_cfg)
FIX5_K = 32
BAR_K32, CHANCE_K32 = 0.09375, 0.03125       # frozen sec 1.31.1 demonstration bar (verbatim)

# The already-completed round-4 sweep's own artifacts for the reused 3e-4 column (cited, never
# copied/duplicated -- sec 1.44's own provenance record).
REUSED_3E4_CELLS = {
    0: {"raw": "experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s0.json",
        "remetric": "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/"
                     "h2h_transformer_task1_sweep_s0_round4.json"},
    1: {"raw": "experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s1.json",
        "remetric": "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/"
                     "h2h_transformer_task1_sweep_s1_round4.json"},
    2: {"raw": "experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s2.json",
        "remetric": "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/"
                     "h2h_transformer_task1_sweep_s2_round4.json"},
}


def _cell_name(lr: float, seed_idx: int) -> str:
    return f"h2h_fix5_transformer_task1_lr{lr:.0e}_s{seed_idx}".replace("+", "")


def fix5_cells() -> list[dict]:
    """9 fresh cells: the 3 LR points NOT already covered by the round-4 sweep, x 3 seeds.
    Deliberately reuses the SAME seed per seed_idx across every LR point (rd_episode_seed's
    `task1_sweep` key, matching the reused 3e-4 cells' own seed schedule EXACTLY) -- holds
    model init + data stream fixed and isolates learning rate as the sole treatment variable,
    the correct design for a paired LR grid (stronger than an independently-reseeded grid)."""
    cells = []
    for lr in FIX5_LR_GRID:
        if lr == FIX5_REUSED_LR:
            continue
        for seed_idx in FIX5_SEED_IDXS:
            cells.append({
                "arch": "transformer", "task": FIX5_TASK, "K": FIX5_K, "seed_idx": seed_idx,
                "seed": rd_episode_seed(FIX5_TASK, seed_idx=seed_idx, ckpt_idx=0),
                "name": _cell_name(lr, seed_idx),
                "budget_frac": 1.0, "role": "sweep", "lr": lr,
            })
    assert len(cells) == (len(FIX5_LR_GRID) - 1) * len(FIX5_SEED_IDXS) == 9, \
        f"expected 9 fresh FIX-5 cells, got {len(cells)}"
    names = [c["name"] for c in cells]
    assert len(names) == len(set(names)), "FIX-5 cell name collision"
    # never silently collide with the sweep's own 27 names
    from h2h_sweep_runner_rd import build_27_cell_manifest
    sweep_names = {c["name"] for c in build_27_cell_manifest()}
    assert not (sweep_names & set(names)), "FIX-5 cell name collides with the 27-cell sweep"
    return cells


def _valid_remetric(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return "leg_a" in doc


def mode_list_cells(args) -> int:
    for c in fix5_cells():
        print(c["name"])
    return 0


def mode_run_cell(args) -> int:
    """Stage 1: TRAIN one fresh cell (mirrors h2h_cell_train_rd.mode_run_cell /
    h2h_task2diag_rd.mode_run_cell's identical gate discipline)."""
    import h2h_cell_train_rd as ct
    from h2h_sweep_runner_rd import is_valid_result, REQUIRED_RESULT_KEYS
    ct.require_launch_tokens(args.gates_dir)
    ct.require_margins_frozen(args.margins_token)
    cells = {c["name"]: c for c in fix5_cells()}
    assert args.run_cell in cells, f"unknown FIX-5 cell {args.run_cell!r}"
    cell = cells[args.run_cell]
    if is_valid_result(args.out):
        print(f"SKIP (already valid): {args.out}")
        return 0
    result = ct.run_one_cell(cell, args.device, args.ckpt_dir, steps_override=args.steps_override)
    result = {**cell, **result}
    assert all(k in result for k in REQUIRED_RESULT_KEYS)
    ct._atomic_dump(args.out, result)
    print(f"CELL COMPLETE: {args.run_cell} final_metric={result['final_metric']} "
          f"wall_s={result['wall_s']:.1f}")
    return 0


def mode_remetric(args) -> int:
    """Stage 2: audited acc_A re-metric of the just-trained checkpoint, via the SAME
    `run_cell_round4` every other h2h number in this campaign goes through."""
    from h2h_round4_driver_rd import run_cell_round4, _md5_of_file
    for c in fix5_cells():
        ckpt_path = os.path.join(args.ckpt_dir, f"{c['name']}_r{args.dial_round}.pt")
        assert os.path.isfile(ckpt_path), f"missing FIX-5 checkpoint: {ckpt_path}"
        manifest = {c["name"]: {"path": ckpt_path, "md5": _md5_of_file(ckpt_path),
                                "mtime": os.path.getmtime(ckpt_path)}}
        out_path = os.path.join(args.remetric_dir, f"{c['name']}_round4.json")
        if _valid_remetric(out_path):
            print(f"SKIP (already valid): {out_path}")
            continue
        spec = {"cell_id": c["name"], "arch": c["arch"], "task": c["task"], "K": c["K"],
                "role": "fix5_lr_grid_remetric", "fresh": False, "seed": c["seed"]}
        r = run_cell_round4(spec, manifest, args.device, args.remetric_dir)
        print(f"REMETRIC {c['name']}: acc_A={r['leg_a']['acc_A']:.4f}")
    return 0


def _repo_root() -> str:
    # matrix-thinking/deltanet_rd/h2h_fix5_lrgrid_rd.py -> repo root is 2 up from matrix-thinking
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def mode_harvest(args) -> int:
    """Builds the 12-cell (4 LR x 3 seed) acc_A + training-curve table and applies the
    sec 1.44 pre-registered decision rule. Never recomputes the reused 3e-4 column -- reads
    it verbatim from the already-completed round-4 sweep's own artifacts. Torch-free (pure
    JSON math); safe to run off box-downloaded artifacts on a dev machine."""
    rows = []
    for lr in FIX5_LR_GRID:
        for seed_idx in FIX5_SEED_IDXS:
            name = _cell_name(lr, seed_idx)
            if lr == FIX5_REUSED_LR:
                raw_path = os.path.join(_repo_root(), REUSED_3E4_CELLS[seed_idx]["raw"])
                remetric_path = os.path.join(_repo_root(), REUSED_3E4_CELLS[seed_idx]["remetric"])
                reused = True
            else:
                raw_path = os.path.join(args.raw_dir, f"{name}.json")
                remetric_path = os.path.join(args.remetric_dir, f"{name}_round4.json")
                reused = False
            with open(raw_path) as f:
                raw = json.load(f)
            with open(remetric_path) as f:
                rem = json.load(f)
            rows.append({
                "lr": lr, "seed_idx": seed_idx, "reused": reused, "name": name,
                "acc_A": rem["leg_a"]["acc_A"], "chance": rem["leg_a"]["chance"],
                "pass_bar": rem["leg_a"]["pass_bar"], "passed": rem["leg_a"]["passed"],
                "loss_first": raw["loss_first"], "loss_final_mean5": raw["loss_final_mean5"],
                "curve": raw["curve"], "raw_path": raw_path, "remetric_path": remetric_path,
            })
    assert len(rows) == 12, f"expected 12 (lr,seed) rows, got {len(rows)}"

    by_lr = {}
    for lr in FIX5_LR_GRID:
        lr_rows = [r for r in rows if r["lr"] == lr]
        accs = [r["acc_A"] for r in lr_rows]
        mean_acc = sum(accs) / len(accs)
        by_lr[lr] = {"mean_acc_A": mean_acc, "per_seed_acc_A": accs,
                    "n_seeds_clearing_bar": sum(1 for a in accs if a > BAR_K32),
                    "mean_clears_bar": mean_acc > BAR_K32}

    any_lr_clears = any(v["mean_clears_bar"] for v in by_lr.values())
    best_lr = max(by_lr, key=lambda lr: by_lr[lr]["mean_acc_A"])
    clearing_lrs = [lr for lr, v in by_lr.items() if v["mean_clears_bar"]]
    if any_lr_clears:
        outcome = "TUNED_TRANSFORMER_CLEARS"
        verdict = ("At least one LR clears the sec 1.31.1 demonstration bar "
                   f"(clearing LRs={clearing_lrs}, best mean acc_A={by_lr[best_lr]['mean_acc_A']:.4f} "
                   f"at lr={best_lr:g} > {BAR_K32}). Per FIX-5's own pre-stated branch: the "
                   "transformer datum is retracted as evidence of a chance-level baseline; the "
                   "capability separation is confined to the vector-state ablation comparison, "
                   "which the frozen sec 1.31.1 registration already designates as the verdict "
                   "carrier.")
    else:
        outcome = "TUNED_TRANSFORMER_STILL_BELOW_BAR"
        verdict = ("No LR in the grid clears the sec 1.31.1 demonstration bar (best mean "
                  f"acc_A={by_lr[best_lr]['mean_acc_A']:.4f} at lr={best_lr:g}, bar={BAR_K32}). "
                  "Per FIX-5's own pre-stated branch: the separation strengthens to a "
                  "two-baseline result (the transformer joins the vector-state ablation as a "
                  "second baseline that fails to recall even after a 4-point LR search).")

    doc = {"grid": list(FIX5_LR_GRID), "reused_lr": FIX5_REUSED_LR, "seed_idxs": list(FIX5_SEED_IDXS),
          "bar": BAR_K32, "chance": CHANCE_K32, "rows": rows, "by_lr": by_lr,
          "best_lr": best_lr, "clearing_lrs": clearing_lrs, "outcome": outcome, "verdict": verdict}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "FIX5_LRGRID_VERDICT.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, out)
    print(f"FIX5 VERDICT written: {out}\n  outcome={outcome}\n  best_lr={best_lr:g} "
          f"mean_acc_A={by_lr[best_lr]['mean_acc_A']:.4f} bar={BAR_K32}")
    for lr in FIX5_LR_GRID:
        v = by_lr[lr]
        per_seed = ", ".join(f"{a:.4f}" for a in v["per_seed_acc_A"])
        print(f"  lr={lr:g}: mean_acc_A={v['mean_acc_A']:.4f} per_seed=[{per_seed}] "
              f"n_clearing={v['n_seeds_clearing_bar']}/3")
    return 0


def mode_selftest() -> int:
    ok_all = True

    def rep(item, ok, detail=""):
        nonlocal ok_all
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {item}" + (f" -- {detail}" if detail else ""))

    # 1) manifest invariants
    cells = fix5_cells()
    rep("selftest 1: 9 fresh cells (3 LRs x 3 seeds), seed schedule matches the reused 3e-4 "
        "cells' own task1_sweep key, uniform K/role/budget",
        len(cells) == 9
        and all(c["seed"] == 1_000_000 + 10_000 * c["seed_idx"] for c in cells)
        and all(c["role"] == "sweep" and c["budget_frac"] == 1.0 and c["K"] == 32 for c in cells)
        and {c["lr"] for c in cells} == {1e-4, 1e-3, 3e-3})

    # 2) reused-LR skip: 3e-4 never appears in the fresh manifest (negative test)
    rep("selftest 2: 3e-4 (the reused LR) is never in the fresh cell manifest",
        all(c["lr"] != FIX5_REUSED_LR for c in cells))

    # 3) harvest decision-rule math on synthetic by_lr tables (both branches + tie-break)
    def fake_by_lr(accs_per_lr):
        by_lr = {}
        for lr, accs in accs_per_lr.items():
            mean_acc = sum(accs) / len(accs)
            by_lr[lr] = {"mean_acc_A": mean_acc, "per_seed_acc_A": accs,
                        "n_seeds_clearing_bar": sum(1 for a in accs if a > BAR_K32),
                        "mean_clears_bar": mean_acc > BAR_K32}
        return by_lr

    below = fake_by_lr({1e-4: [0.03, 0.03, 0.04], 3e-4: [0.027, 0.029, 0.029],
                        1e-3: [0.05, 0.04, 0.06], 3e-3: [0.02, 0.03, 0.02]})
    ok3a = not any(v["mean_clears_bar"] for v in below.values())
    above = fake_by_lr({1e-4: [0.03, 0.03, 0.04], 3e-4: [0.027, 0.029, 0.029],
                        1e-3: [0.95, 0.90, 0.92], 3e-3: [0.02, 0.03, 0.02]})
    ok3b = (any(v["mean_clears_bar"] for v in above.values())
           and max(above, key=lambda lr: above[lr]["mean_acc_A"]) == 1e-3)
    multi = fake_by_lr({1e-4: [0.03, 0.03, 0.04], 3e-4: [0.5, 0.5, 0.5],
                        1e-3: [0.95, 0.90, 0.92], 3e-3: [0.02, 0.03, 0.02]})
    clearing = [lr for lr, v in multi.items() if v["mean_clears_bar"]]
    ok3c = clearing == [3e-4, 1e-3]   # both flagged, never just the best silently swallowing the other
    rep("selftest 3: harvest decision-rule branches (all-below-bar / one-LR-clears+argmax / "
        "multi-LR-clears discloses ALL clearing LRs, not just the best)", ok3a and ok3b and ok3c)

    # 4) exact-threshold boundary on the demonstration bar itself (never fuzzy-tolerance)
    exact = fake_by_lr({1e-4: [BAR_K32, BAR_K32, BAR_K32]})   # mean == bar exactly -> must NOT clear
    ok4 = not exact[1e-4]["mean_clears_bar"]
    rep("selftest 4: mean_acc_A exactly AT the bar does not clear it (strict >, not >=)", ok4)

    print("SELFTEST:", "ALL PASS" if ok_all else "FAILURES PRESENT")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--run-cell", type=str)
    ap.add_argument("--remetric", action="store_true")
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--steps-override", type=int, default=None,
                    help="ONLY for a real-CUDA smoke cell; a missing/default value trains the "
                         "full pinned 20,000-step schedule.")
    ap.add_argument("--out", type=str)
    ap.add_argument("--out-dir", type=str, default="results/h2h_rung1/fix5_lrgrid")
    ap.add_argument("--raw-dir", type=str, default="results/h2h_rung1/fix5_lrgrid")
    ap.add_argument("--remetric-dir", type=str, default="results/h2h_rung1/fix5_lrgrid/remetric")
    ap.add_argument("--ckpt-dir", type=str, default="/data/h2h_rung1_ckpts")
    ap.add_argument("--dial-round", type=int, default=4)
    ap.add_argument("--gates-dir", type=str, default="results/h2h_rung1/gates")
    ap.add_argument("--margins-token", type=str, default="results/h2h_rung1/MARGINS_FROZEN.token")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.selftest:
        return mode_selftest()
    if args.list_cells:
        return mode_list_cells(args)
    if args.run_cell:
        return mode_run_cell(args)
    if args.remetric:
        return mode_remetric(args)
    if args.harvest:
        return mode_harvest(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
