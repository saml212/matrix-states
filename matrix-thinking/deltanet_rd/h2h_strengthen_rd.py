#!/usr/bin/env python
"""h2h_strengthen_rd.py -- BASELINE-STRENGTHENING SWEEP for the H2H
transformer arm (Task 1, episodic recall).

Pre-registration: HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.46 (written and
committed BEFORE this script's cells run on real GPU cells).

WHY: after sec 1.44/1.45's 4-point LR grid (h2h_fix5_lrgrid_rd.py), the
transformer baseline still reads chance at 20,000 steps / 14.44M params /
every searched LR (`TUNED_TRANSFORMER_STILL_BELOW_BAR`). The PI wants a
STRONGER, legitimately-trained baseline before the paper reports the
capability separation against it -- widen capacity AND training length
(never just LR again) so the paper can either (a) report a ratio against
a baseline that actually trained, or (b) state "non-competitive after
capacity x training-length x LR search" with a search that actually
covered capacity and length, not just LR. BOTH outcomes are pre-registered
as publishable (sec 1.46).

GRID (pinned; do not widen without a new pre-registration):
  arch=transformer, task="task1_sweep" (K=32 via task_cfg), role="sweep"
  (the AUD2-F4 structural dial guard -- see h2h_fix5_lrgrid_rd.py's own
  docstring for why "sweep" is mandatory here, not incidental).
  seeds: idx {0,1,2}, `rd_episode_seed("task1_sweep", seed_idx, ckpt_idx=0)`
  -- the SAME schedule fix5/the round-4 sweep used (1,000,000 / 1,010,000 /
  1,020,000), so every capacity's seed lane is the byte-identical episode
  stream to every other arm/round in this campaign that also used this key.

  Capacity (`CAPACITIES`, TRANSFORMER_KW overrides -- see
  `apply_capacity_override` below for the full reader audit):
    C0 = {n_layers:2, d_model:256, n_heads:4, ffn_mult:4}  (the pinned R3-F3
         reference config; ALREADY MEASURED at 20,000 steps for both
         lr=1e-3 and lr=3e-4 by h2h_fix5_lrgrid_rd.py / the round-4 sweep --
         REUSED verbatim below, NEVER relaunched).
    C1 = {n_layers:4, d_model:256, n_heads:4, ffn_mult:4}   (16.01M params)
    C2 = {n_layers:6, d_model:512, n_heads:8, ffn_mult:4}   (44.61M params)
  lr: {1e-3 (fix5's own best-reading LR), 3e-4 (the frozen shared default)}.
  steps: {20,000 (FULL_STEPS, the frozen sweep's own full budget),
          60,000 (3x -- "training-length" as its own independent axis)}.

  Fresh cells = {C1,C2} x 2 lr x 2 steps x 3 seeds = 24, plus
                C0 x 2 lr x 60,000 steps x 3 seeds = 6 (the ONLY C0 cells
                that are not already-measured reuse)
              = 30 fresh cells total. `strengthen_cells()` is the single
  source of truth for this count (asserted).

Two-stage protocol per fresh cell, IDENTICAL to every other h2h acc_A
number in this campaign (sec 1.31.4 item 6 / h2h_fix5_lrgrid_rd.py's own
precedent, mirrored here almost line-for-line): (1) TRAIN via
`h2h_cell_train_rd.train_grammar_cell` (`run_one_cell`) with
`steps_override=cell["steps"]`, persisting the raw JSON with its training
curve; (2) RE-METRIC via `h2h_round4_driver_rd.run_cell_round4`
(`fresh=False`, loading the just-trained, provenance-md5-pinned
checkpoint) for the audited `acc_A`.

CAPACITY OVERRIDE MECHANISM (the part fix5 never needed -- every fix5 cell
shared ONE capacity): `apply_capacity_override(arch_kw)` mutates
`h2h_cell_train_rd.TRANSFORMER_KW` **and** `h2h_cell_train_rd.TAP_DIM
["transformer"]` IN PLACE, at process start, before ANY model is built --
applied identically in the train stage (`mode_run_cell`) and the re-metric
stage (`mode_remetric`, per-cell, since a single `--remetric` invocation
without `--run-cell` walks cells of THREE different capacities and must
re-apply the override before each one). Every reader of `TRANSFORMER_KW`
in `h2h_cell_train_rd.py` was enumerated by grep before this script was
written (see `apply_capacity_override`'s own docstring for the full list);
`TAP_DIM["transformer"]` is a SEPARATE module global (hardcoded 256, never
derived from `TRANSFORMER_KW["d_model"]` in the base file) that must be
overridden IN TANDEM or the probe adapter's input dim silently stays 256
while the transformer's native tap dim tracks the real (overridden)
`d_model` -- verified this fails LOUDLY (a `mat1/mat2 shapes cannot be
multiplied` RuntimeError at the very first forward pass) if forgotten,
never silently, in this script's own selftest (item 5) and in a live
CPU-torch probe run during this script's build (documented in the sec
1.46 design record's "reader audit" table).

ONE KNOWN NON-TRANSFERRING DETAIL (disclosed, not fixed here -- flagged
for coordinator ruling in the sec 1.46 record): `train_grammar_cell`'s own
`capped_mask_fn` (h2h_cell_train_rd.py, the `task1`+transformer "M2
capped-cache" REPORT-ONLY training-curve diagnostic) calls
`cap_length_tokens(2, 2, 256)` with LITERAL hardcoded ints, not a
`TRANSFORMER_KW` dict lookup -- it does NOT track a capacity override.
This means `recovered_frac_capped_M2` / `probe_cos_mean_capped_M2` in
every fresh cell's training curve are computed against the C0 cap-length
regardless of the cell's real capacity. This is NEVER the decision metric
(acc_A, from `run_cell_round4`, unaffected) and is disclosed here rather
than patched -- editing `h2h_cell_train_rd.py` itself is out of this
script's scope and would touch code shared by every other h2h round.

Run the selftest (CPU, tiny steps, torch required -- same convention as
every other h2h_*_rd.py module):
  python h2h_strengthen_rd.py --selftest
--list-cells / --run-cell <name> / --remetric [--run-cell <name>] /
--harvest as documented on each mode function below.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rd_episode_seed import rd_episode_seed                            # noqa: E402

# ---------------------------------------------------------------------------
# Pinned grid (sec 1.46)
# ---------------------------------------------------------------------------

STRENGTHEN_TASK = "task1_sweep"          # SAME task string as fix5/the round-4 sweep -> K=32
STRENGTHEN_K = 32
SEED_IDXS = (0, 1, 2)                    # n=3, sec 1.8's standing convention
FULL_STEPS_C0 = 20_000                   # the frozen sweep's own full budget (already measured)
LONG_STEPS = 60_000                      # 3x -- the "training-length" axis
STEPS_GRID = (FULL_STEPS_C0, LONG_STEPS)
LR_GRID = (1e-3, 3e-4)                   # 1e-3 = fix5's own best-reading LR; 3e-4 = frozen default
BAR_K32 = 0.09375                        # frozen sec 1.31.1 demonstration bar (3x chance), verbatim
COMPETITIVE_BAR = 0.50                   # sec 1.46's own "competitive" threshold, pre-registered
CHANCE_K32 = 1.0 / STRENGTHEN_K

# Capacity grid: TRANSFORMER_KW overrides. C0 is the pinned R3-F3 reference
# (sec 1.3(b) of the design doc); C1/C2 widen depth/width. n_heads is NOT
# read by count_transformer_params (transformer_baseline_rd.py) -- doubling
# it at C2 changes attention-score FLOPs by zero (head_dim shrinks in
# lockstep, n_heads*head_dim=d_model is invariant) but keeps head_dim a
# clean divisor of d_model (512/8=64, matching every other arm's own
# per-head/d_state=64 convention in this campaign -- not required by the
# code, chosen for consistency).
CAPACITIES = {
    "C0": dict(d_model=256, n_layers=2, n_heads=4, ffn_mult=4),   # 14.44M params (measured)
    "C1": dict(d_model=256, n_layers=4, n_heads=4, ffn_mult=4),   # 16.01M params
    "C2": dict(d_model=512, n_layers=6, n_heads=8, ffn_mult=4),   # 44.61M params
}

# The already-completed C0 x 20,000-step cells' own artifacts (cited, never
# copied/duplicated -- sec 1.46's own provenance record, mirroring
# h2h_fix5_lrgrid_rd.REUSED_3E4_CELLS): lr=1e-3 comes from fix5's own 9
# fresh cells; lr=3e-4 comes from the original round-4 27-cell sweep (the
# SAME artifacts fix5 itself reused). Keyed by (lr, seed_idx).
REUSED_C0_20K_CELLS = {
    (1e-3, 0): {"raw": "experiment-runs/2026-07-11_h2h_fix5_lrgrid/results/"
                       "h2h_fix5_transformer_task1_lr1e-03_s0.json",
               "remetric": "experiment-runs/2026-07-11_h2h_fix5_lrgrid/results/remetric/"
                           "h2h_fix5_transformer_task1_lr1e-03_s0_round4.json"},
    (1e-3, 1): {"raw": "experiment-runs/2026-07-11_h2h_fix5_lrgrid/results/"
                       "h2h_fix5_transformer_task1_lr1e-03_s1.json",
               "remetric": "experiment-runs/2026-07-11_h2h_fix5_lrgrid/results/remetric/"
                           "h2h_fix5_transformer_task1_lr1e-03_s1_round4.json"},
    (1e-3, 2): {"raw": "experiment-runs/2026-07-11_h2h_fix5_lrgrid/results/"
                       "h2h_fix5_transformer_task1_lr1e-03_s2.json",
               "remetric": "experiment-runs/2026-07-11_h2h_fix5_lrgrid/results/remetric/"
                           "h2h_fix5_transformer_task1_lr1e-03_s2_round4.json"},
    (3e-4, 0): {"raw": "experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s0.json",
               "remetric": "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/"
                           "h2h_transformer_task1_sweep_s0_round4.json"},
    (3e-4, 1): {"raw": "experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s1.json",
               "remetric": "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/"
                           "h2h_transformer_task1_sweep_s1_round4.json"},
    (3e-4, 2): {"raw": "experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s2.json",
               "remetric": "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/"
                           "h2h_transformer_task1_sweep_s2_round4.json"},
}

# The contender arm's OWN already-measured task1_sweep numbers (sec 1.40's axis-1 WIN verdict,
# n=3, acc_A 0.9995-1.0) -- cited for Outcome B/C's ratio report (never retrained here; the
# contender is unaffected by this transformer-only strengthening round).
CONTENDER_REFERENCE_REMETRIC = {
    0: "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s0_round4.json",
    1: "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s1_round4.json",
    2: "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s2_round4.json",
}


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


# ---------------------------------------------------------------------------
# Capacity override -- the mechanism fix5 never needed
# ---------------------------------------------------------------------------

def apply_capacity_override(arch_kw: dict) -> None:
    """Mutates `h2h_cell_train_rd.TRANSFORMER_KW` and
    `h2h_cell_train_rd.TAP_DIM["transformer"]` IN PLACE, at process start,
    before any model/rig is built in EITHER the train or the re-metric
    stage. Full reader audit (grep of h2h_cell_train_rd.py, done before
    this function was written):

      1. `build_arm_model` (line ~324): `TransformerLM(vocab_size,
         **TRANSFORMER_KW)` -- a dict-unpack lookup at CALL time, so
         in-place mutation of the SAME dict object is picked up correctly
         (no rebinding of the name `TRANSFORMER_KW` needed or done here).
      2. `ProbeRig.__init__` (line ~339): `ph.build_adapter_arm(
         TAP_DIM[arch], VALUE_DIM)` -- `TAP_DIM["transformer"]` is a
         SEPARATE global, hardcoded to 256 in the base file, NEVER derived
         from `TRANSFORMER_KW["d_model"]`. Overriding `TRANSFORMER_KW`
         alone silently leaves the probe adapter's input dim at 256 while
         `transformer_native_tap`'s real output dim tracks the overridden
         d_model -- confirmed this fails LOUDLY (shape-mismatch
         RuntimeError at the very first `assert_fused_tap_matches_audited`
         forward pass in training, or the first re-metric forward), never
         silently, if this line is skipped. Both must move together.
      3. `_transformer_episode_chunks` (the VRAM guard, line ~622): reads
         `TRANSFORMER_KW["n_heads"]` at CALL time -- safe under mutation.
      4. `cap_length_tokens(M, TRANSFORMER_KW["n_layers"],
         TRANSFORMER_KW["d_model"])` call sites (capped_eval_pass, lines
         ~1257/~1297/~2151, all axis-2 M-sweep machinery this sweep never
         calls) -- dict lookups at CALL time, safe under mutation, but
         irrelevant here (this sweep never passes M).
      5. `train_grammar_cell`'s `capped_mask_fn` (line ~876): calls
         `cap_length_tokens(2, 2, 256)` with LITERAL hardcoded ints, NOT a
         TRANSFORMER_KW lookup. Does NOT track this override. Affects only
         the report-only `recovered_frac_capped_M2` / `probe_cos_mean_
         capped_M2` training-curve diagnostic columns (never the decision
         metric acc_A). Disclosed in this module's own docstring and the
         sec 1.46 design record; NOT patched here (out of this script's
         scope -- it is shared code, not owned by this sweep).

    `TAP_DIM` for "contender"/"ablation" is untouched (this sweep is
    transformer-only)."""
    import h2h_cell_train_rd as ct
    ct.TRANSFORMER_KW.update(arch_kw)
    ct.TAP_DIM["transformer"] = arch_kw["d_model"]


# ---------------------------------------------------------------------------
# Cell manifest
# ---------------------------------------------------------------------------

def _lr_str(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "")


def _cell_name(cap_id: str, lr: float, steps: int, seed_idx: int) -> str:
    return f"h2h_strengthen_{cap_id}_lr{_lr_str(lr)}_st{steps}_s{seed_idx}"


def strengthen_cells() -> list[dict]:
    """30 fresh cells, in the SAME order the sec 1.46 queue specs
    (0600-0629) are numbered: C1 then C2 (each lr x steps x seed), then
    the 6 C0 x 60,000-step cells. Never emits a (C0, 20_000) cell -- those
    6 (lr, seed) points are pinned REUSE (`REUSED_C0_20K_CELLS`)."""
    cells = []
    for cap_id in ("C1", "C2"):
        for lr in LR_GRID:
            for steps in STEPS_GRID:
                for seed_idx in SEED_IDXS:
                    cells.append({
                        "arch": "transformer", "task": STRENGTHEN_TASK, "K": STRENGTHEN_K,
                        "seed_idx": seed_idx,
                        "seed": rd_episode_seed(STRENGTHEN_TASK, seed_idx=seed_idx, ckpt_idx=0),
                        "name": _cell_name(cap_id, lr, steps, seed_idx),
                        "role": "sweep", "lr": lr, "steps": steps,
                        "capacity": cap_id, "arch_kw": dict(CAPACITIES[cap_id]),
                    })
    for lr in LR_GRID:
        for seed_idx in SEED_IDXS:
            cells.append({
                "arch": "transformer", "task": STRENGTHEN_TASK, "K": STRENGTHEN_K,
                "seed_idx": seed_idx,
                "seed": rd_episode_seed(STRENGTHEN_TASK, seed_idx=seed_idx, ckpt_idx=0),
                "name": _cell_name("C0", lr, LONG_STEPS, seed_idx),
                "role": "sweep", "lr": lr, "steps": LONG_STEPS,
                "capacity": "C0", "arch_kw": dict(CAPACITIES["C0"]),
            })
    assert len(cells) == 30, f"expected 30 fresh strengthening cells, got {len(cells)}"
    names = [c["name"] for c in cells]
    assert len(names) == len(set(names)), "strengthen cell name collision"
    # never silently collide with the 27-cell sweep or fix5's own 9 cells
    from h2h_sweep_runner_rd import build_27_cell_manifest
    from h2h_fix5_lrgrid_rd import fix5_cells
    other_names = {c["name"] for c in build_27_cell_manifest()} | {c["name"] for c in fix5_cells()}
    assert not (other_names & set(names)), "strengthen cell name collides with an existing manifest"
    return cells


def all_configs() -> list[dict]:
    """The 12 (capacity, lr, steps) CONFIGS the harvest's decision rule is applied to (each
    backed by 3 seeds -- either 3 fresh cells or the 3 reused C0@20k points). Order matches
    `strengthen_cells()` plus the 2 reused C0@20k configs appended last."""
    configs = []
    for cap_id in ("C1", "C2"):
        for lr in LR_GRID:
            for steps in STEPS_GRID:
                configs.append({"capacity": cap_id, "lr": lr, "steps": steps, "reused": False})
    for lr in LR_GRID:
        configs.append({"capacity": "C0", "lr": lr, "steps": LONG_STEPS, "reused": False})
    for lr in LR_GRID:
        configs.append({"capacity": "C0", "lr": lr, "steps": FULL_STEPS_C0, "reused": True})
    assert len(configs) == 12, f"expected 12 (capacity,lr,steps) configs, got {len(configs)}"
    return configs


def _valid_remetric(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return "leg_a" in doc and math.isfinite(doc["leg_a"].get("acc_A", float("nan")))


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def mode_list_cells(args) -> int:
    for c in strengthen_cells():
        print(c["name"])
    return 0


def mode_run_cell(args) -> int:
    """Stage 1: TRAIN one fresh cell. Applies the cell's OWN capacity
    override BEFORE importing/building anything model-shaped (mirrors
    h2h_fix5_lrgrid_rd.mode_run_cell's gate discipline otherwise
    verbatim)."""
    cells = {c["name"]: c for c in strengthen_cells()}
    assert args.run_cell in cells, f"unknown strengthen cell {args.run_cell!r}"
    cell = cells[args.run_cell]
    apply_capacity_override(cell["arch_kw"])

    import h2h_cell_train_rd as ct
    from h2h_sweep_runner_rd import is_valid_result, REQUIRED_RESULT_KEYS
    ct.require_launch_tokens(args.gates_dir)
    ct.require_margins_frozen(args.margins_token)
    if is_valid_result(args.out):
        print(f"SKIP (already valid): {args.out}")
        return 0
    steps_override = args.steps_override if args.steps_override is not None else cell["steps"]
    result = ct.run_one_cell(cell, args.device, args.ckpt_dir, steps_override=steps_override)
    result = {**cell, **result}
    assert all(k in result for k in REQUIRED_RESULT_KEYS)
    ct._atomic_dump(args.out, result)
    print(f"CELL COMPLETE: {args.run_cell} capacity={cell['capacity']} "
          f"final_metric={result['final_metric']} wall_s={result['wall_s']:.1f}")
    return 0


def _remetric_one(cell: dict, args) -> None:
    from h2h_round4_driver_rd import run_cell_round4, _md5_of_file
    apply_capacity_override(cell["arch_kw"])
    ckpt_path = os.path.join(args.ckpt_dir, f"{cell['name']}_r{args.dial_round}.pt")
    assert os.path.isfile(ckpt_path), f"missing strengthen checkpoint: {ckpt_path}"
    manifest = {cell["name"]: {"path": ckpt_path, "md5": _md5_of_file(ckpt_path),
                               "mtime": os.path.getmtime(ckpt_path)}}
    out_path = os.path.join(args.remetric_dir, f"{cell['name']}_round4.json")
    if _valid_remetric(out_path):
        print(f"SKIP (already valid): {out_path}")
        return
    spec = {"cell_id": cell["name"], "arch": cell["arch"], "task": cell["task"], "K": cell["K"],
            "role": "strengthen_remetric", "fresh": False, "seed": cell["seed"]}
    r = run_cell_round4(spec, manifest, args.device, args.remetric_dir)
    # inject the capacity/step-target provenance the base run_cell_round4 result dict does not
    # carry -- REQUIRED by the sec 1.46 validity_check ("arch_kw recorded in the JSON equals the
    # spec's") and by the harvest, which reads this field back rather than re-deriving it.
    r["arch_kw"] = cell["arch_kw"]
    r["capacity"] = cell["capacity"]
    r["lr"] = cell["lr"]
    r["steps_target"] = cell["steps"]
    from h2h_cell_train_rd import _atomic_dump
    _atomic_dump(out_path, r)
    print(f"REMETRIC {cell['name']}: acc_A={r['leg_a']['acc_A']:.4f}")


def mode_remetric(args) -> int:
    """Stage 2: audited acc_A re-metric of the just-trained checkpoint(s),
    via the SAME `run_cell_round4` every other h2h number in this campaign
    goes through. With `--run-cell NAME` (the queue-spec convention -- one
    job, one cell, both stages chained in its `cmd`): re-metrics ONLY that
    cell, re-applying its own capacity override first. Without it: loops
    over all 30 fresh cells, re-applying each cell's OWN override before
    that cell's checkpoint is touched (capacities differ across cells in
    this sweep, unlike fix5's single-capacity 9-cell round -- the override
    must be re-applied per iteration, not once)."""
    cells = {c["name"]: c for c in strengthen_cells()}
    if args.run_cell:
        assert args.run_cell in cells, f"unknown strengthen cell {args.run_cell!r}"
        _remetric_one(cells[args.run_cell], args)
        return 0
    for cell in cells.values():
        _remetric_one(cell, args)
    return 0


# ---------------------------------------------------------------------------
# Harvest -- pre-registered decision rule (sec 1.46)
# ---------------------------------------------------------------------------

def _delta_ci3(values_a: list[float], values_b: list[float]) -> dict:
    """Paired delta=(a-b) CI at n=3, t(2,.975)=4.303 -- a deliberate, small,
    STANDALONE copy (not an import) of `reasoning_link_probe.delta_ci_n3`'s
    formula (same file's own house convention for paired n=3 CIs
    throughout this campaign, e.g. sec 1.40's "paired CIs exclude the
    margin by 3x+"). Copied rather than imported so `--harvest` stays
    torch-free (`reasoning_link_probe.py` imports torch at module level;
    fix5_lrgrid's own harvest is torch-free by the same discipline)."""
    assert len(values_a) == 3 and len(values_b) == 3
    T_975_DF2 = 4.303
    deltas = [a - b for a, b in zip(values_a, values_b)]
    mean = sum(deltas) / 3
    var = sum((d - mean) ** 2 for d in deltas) / 2
    se = math.sqrt(var / 3)
    hw = T_975_DF2 * se
    return {"deltas": deltas, "mean": mean, "ci_low": mean - hw, "ci_high": mean + hw}


def _config_key(cfg: dict) -> str:
    return f"{cfg['capacity']}_lr{_lr_str(cfg['lr'])}_st{cfg['steps']}"


def _load_config_rows(cfg: dict, args) -> list[dict]:
    rows = []
    for seed_idx in SEED_IDXS:
        if cfg["reused"]:
            paths = REUSED_C0_20K_CELLS[(cfg["lr"], seed_idx)]
            raw_path = os.path.join(_repo_root(), paths["raw"])
            remetric_path = os.path.join(_repo_root(), paths["remetric"])
        else:
            name = _cell_name(cfg["capacity"], cfg["lr"], cfg["steps"], seed_idx)
            raw_path = os.path.join(args.raw_dir, f"{name}.json")
            remetric_path = os.path.join(args.remetric_dir, f"{name}_round4.json")
        with open(raw_path) as f:
            raw = json.load(f)
        with open(remetric_path) as f:
            rem = json.load(f)
        rows.append({"seed_idx": seed_idx, "acc_A": rem["leg_a"]["acc_A"],
                    "chance": rem["leg_a"]["chance"], "curve": raw["curve"],
                    "loss_first": raw["loss_first"], "loss_final_mean5": raw["loss_final_mean5"],
                    "raw_path": raw_path, "remetric_path": remetric_path})
    return rows


def mode_harvest(args) -> int:
    """Builds the 12-config x 3-seed table and applies the sec 1.46
    pre-registered decision rule:
      CLEARS       iff >= 2/3 seeds' acc_A >= BAR_K32 (0.09375).
      COMPETITIVE  iff >= 2/3 seeds' acc_A >= COMPETITIVE_BAR (0.50).
      Outcome A: no config clears -> non-competitive after capacity x
                 training-length x LR search (strengthens the separation).
      Outcome B: some config clears, none competitive -> report the best
                 clearing config's ratio against the contender (paired
                 delta-CI, sec 1.46 -- NOT a naive ratio+CI, which is
                 unstable near-zero; a difference-CI is this campaign's
                 own standing statistic, e.g. sec 1.40), disclosed as an
                 over-budget baseline (more params and/or more tokens than
                 the contender).
      Outcome C: any config competitive -> the ratio at the BEST
                 competitive config becomes the headline AND triggers the
                 pre-registered horizon fan-out (mstar's H2/H4/H8, via
                 h2h_mstar_harvest_rd.py's protocol) as a FOLLOW-ON, never
                 run by this script itself.
    Torch-free (pure JSON math); safe to run off box-downloaded
    artifacts."""
    configs = all_configs()
    by_config = {}
    for cfg in configs:
        rows = _load_config_rows(cfg, args)
        accs = [r["acc_A"] for r in rows]
        n_clear = sum(1 for a in accs if a >= BAR_K32)
        n_competitive = sum(1 for a in accs if a >= COMPETITIVE_BAR)
        by_config[_config_key(cfg)] = {
            **cfg, "per_seed_acc_A": accs, "mean_acc_A": sum(accs) / len(accs),
            "n_clearing_bar": n_clear, "clears": n_clear >= 2,
            "n_competitive": n_competitive, "competitive": n_competitive >= 2,
            "rows": rows,
        }

    any_clears = any(v["clears"] for v in by_config.values())
    any_competitive = any(v["competitive"] for v in by_config.values())
    clearing_keys = [k for k, v in by_config.items() if v["clears"]]
    competitive_keys = [k for k, v in by_config.items() if v["competitive"]]

    contender_accs = {}
    for seed_idx, path in CONTENDER_REFERENCE_REMETRIC.items():
        with open(os.path.join(_repo_root(), path)) as f:
            contender_accs[seed_idx] = json.load(f)["leg_a"]["acc_A"]

    if not any_clears:
        outcome = "OUTCOME_A_NON_COMPETITIVE"
        best_key = max(by_config, key=lambda k: by_config[k]["mean_acc_A"])
        verdict = ("No (capacity, lr, steps) config clears the frozen sec 1.31.1 demonstration "
                  f"bar (0.09375) on >=2/3 seeds (best mean acc_A={by_config[best_key]['mean_acc_A']:.4f} "
                  f"at {best_key}). The transformer baseline is NON-COMPETITIVE after an explicit "
                  "capacity x training-length x LR search -- the capability separation strengthens "
                  "to a search that covered all three axes, not LR alone.")
        ratio_report = None
    else:
        best_key = max(clearing_keys, key=lambda k: by_config[k]["mean_acc_A"])
        best = by_config[best_key]
        ratio_seeds = [best["rows"][i]["seed_idx"] for i in range(3)]
        contender_paired = [contender_accs[s] for s in ratio_seeds]
        transformer_paired = best["per_seed_acc_A"]
        delta = _delta_ci3(contender_paired, transformer_paired)
        mean_ratio = ((sum(contender_paired) / 3) / (sum(transformer_paired) / 3)
                     if sum(transformer_paired) > 0 else float("inf"))
        ratio_report = {"best_config": best_key, "contender_acc_A": contender_paired,
                        "transformer_acc_A": transformer_paired, "mean_ratio_contender_over_transformer": mean_ratio,
                        "paired_delta_contender_minus_transformer": delta}
        if not any_competitive:
            outcome = "OUTCOME_B_CLEARS_NOT_COMPETITIVE"
            verdict = (f"{best_key} clears the bar on >=2/3 seeds (mean acc_A="
                      f"{best['mean_acc_A']:.4f}) but no config reaches the 0.50 competitive "
                      "threshold. Reporting the ratio against the contender at this best "
                      f"clearing config as an OVER-BUDGET baseline (capacity={best['capacity']} "
                      f"vs C0's reference, steps={best['steps']} vs the frozen 20,000) -- "
                      "disclosed, not silently normalized away.")
        else:
            outcome = "OUTCOME_C_COMPETITIVE"
            best_key = max(competitive_keys, key=lambda k: by_config[k]["mean_acc_A"])
            best = by_config[best_key]
            ratio_seeds = [best["rows"][i]["seed_idx"] for i in range(3)]
            contender_paired = [contender_accs[s] for s in ratio_seeds]
            delta = _delta_ci3(contender_paired, best["per_seed_acc_A"])
            mean_ratio = ((sum(contender_paired) / 3) / (sum(best["per_seed_acc_A"]) / 3)
                         if sum(best["per_seed_acc_A"]) > 0 else float("inf"))
            ratio_report = {"best_config": best_key, "contender_acc_A": contender_paired,
                            "transformer_acc_A": best["per_seed_acc_A"],
                            "mean_ratio_contender_over_transformer": mean_ratio,
                            "paired_delta_contender_minus_transformer": delta}
            verdict = (f"{best_key} reaches the 0.50 competitive threshold (mean acc_A="
                      f"{best['mean_acc_A']:.4f}). This becomes the headline ratio config AND "
                      "triggers the pre-registered horizon fan-out (mstar H2/H4/H8, via "
                      "h2h_mstar_harvest_rd.py's protocol) as a FOLLOW-ON -- NOT run by this "
                      "script; a separate dispatch applies once this verdict is recorded.")

    doc = {"configs": [_config_key(c) for c in configs], "bar": BAR_K32,
          "competitive_bar": COMPETITIVE_BAR, "chance": CHANCE_K32,
          "by_config": by_config, "any_clears": any_clears, "any_competitive": any_competitive,
          "clearing_configs": clearing_keys, "competitive_configs": competitive_keys,
          "outcome": outcome, "verdict": verdict, "ratio_report": ratio_report}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "STRENGTHEN_VERDICT.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, out)
    print(f"STRENGTHEN VERDICT written: {out}\n  outcome={outcome}")
    for key, v in sorted(by_config.items()):
        per_seed = ", ".join(f"{a:.4f}" for a in v["per_seed_acc_A"])
        print(f"  {key}: mean_acc_A={v['mean_acc_A']:.4f} per_seed=[{per_seed}] "
              f"n_clearing={v['n_clearing_bar']}/3 n_competitive={v['n_competitive']}/3")
    print(f"  verdict: {verdict}")
    return 0


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def mode_selftest() -> int:
    ok_all = True

    def rep(item, ok, detail=""):
        nonlocal ok_all
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {item}" + (f" -- {detail}" if detail else ""))

    # 1) manifest invariants
    cells = strengthen_cells()
    caps_seen = {c["capacity"] for c in cells}
    steps_by_cap = {cap: sorted({c["steps"] for c in cells if c["capacity"] == cap})
                    for cap in caps_seen}
    rep("selftest 1: 30 fresh cells (C1/C2 x 2lr x 2steps x 3seeds=24, C0 x 2lr x 60k x 3seeds=6), "
        "seed schedule matches task1_sweep's own key, uniform K/role, C0 never emits a 20k cell",
        len(cells) == 30 and caps_seen == {"C0", "C1", "C2"}
        and steps_by_cap["C0"] == [60_000] and steps_by_cap["C1"] == [20_000, 60_000]
        and steps_by_cap["C2"] == [20_000, 60_000]
        and all(c["seed"] == 1_000_000 + 10_000 * c["seed_idx"] for c in cells)
        and all(c["role"] == "sweep" and c["K"] == 32 for c in cells)
        and all(c["lr"] in LR_GRID for c in cells))

    # 2) capacity override -- POSITIVE round trip (build, save, reload under the SAME override,
    #    forward pass through both the model and the probe adapter without a shape error)
    import torch
    import h2h_cell_train_rd as ct
    import probe_head_rd as ph
    tiny_c1 = dict(d_model=32, n_layers=3, n_heads=2, ffn_mult=2)
    apply_capacity_override(tiny_c1)
    m = ct.build_arm_model("transformer", 500, seed=1, device="cpu")
    rig = ct.ProbeRig("transformer", 500, "cpu")
    ok2a = (len(m.blocks) == 3 and m.d_model == 32 and rig.adapter.in_features == 32
           and ct.TRANSFORMER_KW["d_model"] == 32 and ct.TAP_DIM["transformer"] == 32)
    tmp_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_strengthen_selftest_ckpt.pt")
    torch.save({"arch": "transformer", "vocab_size_total": 500, "model": m.state_dict(),
               "rig": rig.state_dict()}, tmp_ckpt)
    m2, rig2, _ = ct.load_h2h_checkpoint(tmp_ckpt, "cpu")
    ctx = torch.randint(0, 500, (2, 20))
    q = torch.randint(0, 500, (2, 3, 4))
    tap = ph.transformer_native_tap(m2, ctx, q)
    pred = rig2.pred(tap)
    ok2b = (len(m2.blocks) == 3 and m2.d_model == 32 and tuple(tap.shape) == (2, 3, 32)
           and tuple(pred.shape) == (2, 3, 64))
    rep("selftest 2: apply_capacity_override moves TRANSFORMER_KW + TAP_DIM together; checkpoint "
        "ROUND-TRIPS into the overridden shape and a real forward pass (tap -> adapter) succeeds",
        ok2a and ok2b, f"d_model={ct.TRANSFORMER_KW['d_model']} tap_dim={ct.TAP_DIM['transformer']}")

    # 3) NEGATIVE test: a MISMATCHED override before load must fail LOUDLY (state_dict size
    #    mismatch), never silently -- proves the mechanism has teeth, run to completion.
    apply_capacity_override(dict(d_model=64, n_layers=3, n_heads=2, ffn_mult=2))
    raised3 = False
    try:
        ct.load_h2h_checkpoint(tmp_ckpt, "cpu")
        print("NEGATIVE FAILED TO FAIL: mismatched capacity override did not raise on load",
              file=sys.stderr)
    except RuntimeError:
        raised3 = True
    rep("selftest 3: a MISMATCHED capacity override (d_model 64 vs the checkpoint's 32) "
        "raises RuntimeError on load_h2h_checkpoint -- negative test run to completion", raised3)
    os.remove(tmp_ckpt)

    # 4) TAP_DIM forgotten (mutate TRANSFORMER_KW only) also fails loudly, at the FIRST forward
    #    pass (a distinct failure mode from item 3 -- proves BOTH globals are load-bearing).
    ct.TRANSFORMER_KW.update(dict(d_model=48, n_layers=2, n_heads=4, ffn_mult=2))
    # deliberately do NOT update TAP_DIM["transformer"] here
    ct.TAP_DIM["transformer"] = 256   # force it back to the stale base-file default
    m4 = ct.build_arm_model("transformer", 500, seed=1, device="cpu")
    rig4 = ct.ProbeRig("transformer", 500, "cpu")
    ctx4 = torch.randint(0, 500, (2, 20))
    q4 = torch.randint(0, 500, (2, 3, 4))
    tap4 = ph.transformer_native_tap(m4, ctx4, q4)
    raised4 = False
    try:
        rig4.pred(tap4)
        print("NEGATIVE FAILED TO FAIL: forgetting TAP_DIM did not raise", file=sys.stderr)
    except RuntimeError:
        raised4 = True
    rep("selftest 4: forgetting TAP_DIM (TRANSFORMER_KW overridden alone) raises RuntimeError "
        "at the first tap->adapter forward pass -- confirms TAP_DIM is independently load-bearing",
        raised4)

    # 5) harvest decision-rule branches on synthetic by_config tables (>= thresholds, exact
    #    boundary, all three outcomes)
    def fake_cfg(accs):
        n_clear = sum(1 for a in accs if a >= BAR_K32)
        n_comp = sum(1 for a in accs if a >= COMPETITIVE_BAR)
        return {"per_seed_acc_A": accs, "mean_acc_A": sum(accs) / len(accs),
               "n_clearing_bar": n_clear, "clears": n_clear >= 2,
               "n_competitive": n_comp, "competitive": n_comp >= 2}

    a_cfg = fake_cfg([0.03, 0.04, 0.035])
    ok5a = not a_cfg["clears"]                                    # Outcome A shape
    b_cfg = fake_cfg([0.12, 0.10, 0.04])
    ok5b = b_cfg["clears"] and not b_cfg["competitive"]            # Outcome B shape
    c_cfg = fake_cfg([0.55, 0.60, 0.30])
    ok5c = c_cfg["clears"] and c_cfg["competitive"]                # Outcome C shape (2/3 >= 0.50)
    exact_bar = fake_cfg([BAR_K32, BAR_K32, 0.02])
    ok5d = exact_bar["clears"]                                     # >= is inclusive at the bar
    exact_comp = fake_cfg([COMPETITIVE_BAR, COMPETITIVE_BAR, 0.02])
    ok5e = exact_comp["competitive"]
    one_only = fake_cfg([BAR_K32, 0.02, 0.02])
    ok5f = not one_only["clears"]                                  # 1/3 is not enough (need >=2/3)
    rep("selftest 5: harvest decision rule -- Outcome A/B/C shapes, inclusive >= at both "
        "thresholds, and 1/3-clearing is NOT sufficient (need >=2/3)",
        ok5a and ok5b and ok5c and ok5d and ok5e and ok5f)

    # 6) queue-spec schema self-check against the archive's own precedent schema
    example_spec = {
        "id": "0600_h2h_strengthen_C1_lr1e-03_st20000_s0",
        "lane": "strengthen", "hypothesis": "smoke", "cmd": "true",
        "gpu_h_estimate": 0.29, "output_dir": "/tmp/x",
        "validity_check": "true", "notes": "smoke",
    }
    required_spec_keys = ("id", "lane", "hypothesis", "cmd", "gpu_h_estimate", "output_dir",
                          "validity_check", "notes")
    rep("selftest 6: queue-spec schema matches the archive precedent's own field set "
        "(experiment-runs/2026-08-29_box_final_archive/queue/completed/005_..._K128_s0.json)",
        all(k in example_spec for k in required_spec_keys))

    # 7) GPU-h ledger arithmetic sanity (catches silent drift if CAPACITIES/STEPS_GRID/LR_GRID
    #    change without the design-doc ledger being re-derived)
    from h2h_strengthen_specs_gen import estimate_gpu_h, C0_TRAIN_RATE_20K, RUN_METRIC_RATE_20K
    ledger = 0.0
    for c in strengthen_cells():
        ledger += estimate_gpu_h(c["capacity"], c["steps"])
    rep("selftest 7: recomputed 30-cell GPU-h ledger from the params-ratio estimator matches "
        "the design-doc-cited total to 2 decimals (see the module for the method)",
        24.0 < ledger < 36.0, f"ledger={ledger:.3f} GPU-h "
        f"(C0 rate={C0_TRAIN_RATE_20K}, remetric ref={RUN_METRIC_RATE_20K})")

    print("=" * 70)
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
                         "cell's own pinned steps (20,000 or 60,000 per the grid).")
    ap.add_argument("--out", type=str)
    ap.add_argument("--out-dir", type=str, default="results/h2h_rung1/strengthen")
    ap.add_argument("--raw-dir", type=str, default="results/h2h_rung1/strengthen")
    ap.add_argument("--remetric-dir", type=str, default="results/h2h_rung1/strengthen/remetric")
    ap.add_argument("--ckpt-dir", type=str, default="/data/h2h_strengthen_ckpts")
    ap.add_argument("--dial-round", type=int, default=4)
    ap.add_argument("--gates-dir", type=str, default="results/h2h_rung1/gates")
    ap.add_argument("--margins-token", type=str, default="results/h2h_rung1/MARGINS_FROZEN.token")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.selftest:
        return mode_selftest()
    if args.list_cells:
        return mode_list_cells(args)
    if args.run_cell and not args.remetric:
        return mode_run_cell(args)
    if args.remetric:
        return mode_remetric(args)
    if args.harvest:
        return mode_harvest(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
