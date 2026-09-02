#!/usr/bin/env python
"""h2h_strengthen_rd.py -- BASELINE-STRENGTHENING SWEEP for the H2H
transformer arm (Task 1, episodic recall).

Pre-registration: HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.46 (written and
committed BEFORE this script's cells run on real GPU cells).

REV-narrow AUDIT (2026-09-01, 4 MAJOR, applied here -- no launch/staging/
commit happened before or during this revision):
  M1 every spec's train stage is wrapped `timeout -k 120 <N>h` (see
     `h2h_strengthen_specs_gen.TIMEOUT_HOURS`); on kill, `train_grammar_cell`
     never reaches its atomic-dump-at-the-end, so no raw JSON is written,
     `is_valid_result`/this module's own step_count check both read "not
     valid", and the GPU frees for the next queue claim.
  M2 the GPU-h cost model was re-derived as BLOCK-FLOP-scaled (not naive
     total-param-ratio) with a corrected anchor rate -- see
     `h2h_strengthen_specs_gen.py`'s own module docstring for the full
     derivation; both the pre-audit and audit-corrected totals are
     disclosed in sec 1.46. The pre-audit record's "/data ~95% full, 139
     GiB free" premise was ALSO wrong (corrected: 914 GB free) -- moot
     regardless, since checkpoints move to `/ephemeral/` per STATE.md's
     own disk policy either way.
  M3 provenance hardening: `_remetric_one` now records the LOADED model's
     own `n_params_loaded`/`d_model_loaded`/`n_layers_loaded` (read back
     from the checkpoint after `run_cell_round4` returns, not merely
     copied from the spec/cell dict); `_valid_remetric` now REQUIRES the
     remetric record's `provenance.md5` to match the checkpoint file's
     CURRENT md5 on disk (a stale record from an overwritten/mismatched
     checkpoint is no longer trusted); `mode_run_cell`'s skip-if-valid
     path now also requires the existing raw JSON's `step_count` to equal
     the cell's real target (a stale smoke output sitting at the real
     `--out` path can no longer be mistaken for a completed real cell);
     every spec's `validity_check` now also asserts the raw JSON's own
     `n_params` matches the capacity's formula-derived count. Together
     these close the "double-dump race" (a probe/smoke and a real run
     sharing a path) at the SOURCE, not just by convention.
  M4 a gating probe (`strengthen_specs_probe/0599_h2h_strengthen_probe_C2.json`,
     300 steps on the SAME cell identity as the real, most-expensive
     0621 spec, DISTINCT paths) MUST validate and re-price the ledger
     from its own measured s/step BEFORE any of 0600-0629 are staged.
  minor: the `capped_M2` training-curve diagnostic's true M-multiplier is
     capacity-dependent (M~=2 at C0, M~=4 at C1, M~=12 at C2 -- see the
     "KNOWN NON-TRANSFERRING DETAIL" section below); the md5-provenance
     rule (M3) closes the double-dump race the minor note also flagged.

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
for coordinator ruling in the sec 1.46 record; RELABELED per the audit's
own minor note): `train_grammar_cell`'s own `capped_mask_fn`
(h2h_cell_train_rd.py, the `task1`+transformer "M2 capped-cache"
REPORT-ONLY training-curve diagnostic) calls `cap_length_tokens(2, 2,
256)` with LITERAL hardcoded ints, not a `TRANSFORMER_KW` dict lookup --
it does NOT track a capacity override. `cap_length_tokens(M, n_layers,
d_model) = M * CONTENDER_TOTAL_STATE_BYTES / (2*n_layers*d_model*
bytes_per_elt)` (transformer_baseline_rd.py's own pinned formula) is
proportional to `M/(n_layers*d_model)`; holding the computed cap-length
value fixed at C0's own (n_layers=2, d_model=256) but applying it to a
DIFFERENT capacity's real (n_layers, d_model) is therefore equivalent to
a DIFFERENT effective M at that capacity: solving `M_eff = 2 *
(n_layers*d_model)_real / (n_layers*d_model)_C0` gives **M_eff~=4 at C1**
(product 1024 vs C0's 512, exactly 2x) and **M_eff~=12 at C2** (product
3072 vs C0's 512, exactly 6x) -- verified by exact arithmetic against the
pinned formula, not approximated. So `recovered_frac_capped_M2` /
`probe_cos_mean_capped_M2` in a C1/C2 fresh cell's training curve are
NOT "the M=2 capped-cache read" their name implies -- they are the M~=4
(C1) / M~=12 (C2) read, respectively; only at C0 does the column's name
match its content. This is NEVER the decision metric (`acc_A`, from
`run_cell_round4`, unaffected) and is RELABELED here (in prose, for any
downstream consumer of the training curve) rather than patched in code
-- editing `h2h_cell_train_rd.py` itself is out of this script's scope
and would touch code shared by every other h2h round.

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


def _md5_of_file(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_remetric(path: str, ckpt_path: str | None = None) -> bool:
    """AUDIT FIX (M3, 2026-09-01): when `ckpt_path` is given, ALSO requires the record's own
    `provenance.md5` to match the checkpoint file's CURRENT md5 on disk -- closes the
    "double-dump race" (a stale remetric JSON left behind after its checkpoint was later
    overwritten, e.g. by a probe/smoke sharing a path, is no longer silently trusted). Without
    `ckpt_path` this check is skipped (this module's only call site always passes it; the
    parameter stays optional so a caller with no checkpoint to check against -- there is none
    today -- is not forced to fabricate one)."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if "leg_a" not in doc or not math.isfinite(doc["leg_a"].get("acc_A", float("nan"))):
        return False
    if ckpt_path is not None:
        prov = doc.get("provenance") or {}
        if not os.path.isfile(ckpt_path) or prov.get("md5") != _md5_of_file(ckpt_path):
            return False
    return True


def _existing_train_result_is_current(path: str, cell: dict) -> bool:
    """AUDIT FIX (M3, 2026-09-01): True only if `path` is a valid result (REQUIRED_RESULT_KEYS
    present, per `is_valid_result`) AND its recorded `step_count` equals `cell`'s REAL target.
    `is_valid_result` alone would accept a stale smoke output (e.g. a `--steps-override 300`
    probe run) sitting at a cell's real `--out` path as "already done", forever. Factored out of
    `mode_run_cell` so it is independently selftest-able without spawning a real training run."""
    from h2h_sweep_runner_rd import is_valid_result
    if not is_valid_result(path):
        return False
    with open(path) as f:
        existing = json.load(f)
    return existing.get("step_count") == cell["steps"]


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
    from h2h_sweep_runner_rd import REQUIRED_RESULT_KEYS
    ct.require_launch_tokens(args.gates_dir)
    ct.require_margins_frozen(args.margins_token)
    # AUDIT FIX (M3, 2026-09-01): skip requires step_count == target, not merely "a valid-shaped
    # JSON exists" -- see _existing_train_result_is_current's own docstring.
    if _existing_train_result_is_current(args.out, cell):
        print(f"SKIP (already valid, step_count matches target {cell['steps']}): {args.out}")
        return 0
    if os.path.isfile(args.out):
        print(f"NOT SKIPPING: {args.out} exists but is stale/under-trained for target "
              f"{cell['steps']} -- retraining")
    steps_override = args.steps_override if args.steps_override is not None else cell["steps"]
    result = ct.run_one_cell(cell, args.device, args.ckpt_dir, steps_override=steps_override)
    result = {**cell, **result}
    assert all(k in result for k in REQUIRED_RESULT_KEYS)
    ct._atomic_dump(args.out, result)
    print(f"CELL COMPLETE: {args.run_cell} capacity={cell['capacity']} "
          f"final_metric={result['final_metric']} wall_s={result['wall_s']:.1f}")
    return 0


def _remetric_one(cell: dict, args) -> None:
    from h2h_round4_driver_rd import run_cell_round4, _md5_of_file as _round4_md5_of_file
    apply_capacity_override(cell["arch_kw"])
    ckpt_path = os.path.join(args.ckpt_dir, f"{cell['name']}_r{args.dial_round}.pt")
    assert os.path.isfile(ckpt_path), f"missing strengthen checkpoint: {ckpt_path}"
    manifest = {cell["name"]: {"path": ckpt_path, "md5": _round4_md5_of_file(ckpt_path),
                               "mtime": os.path.getmtime(ckpt_path)}}
    out_path = os.path.join(args.remetric_dir, f"{cell['name']}_round4.json")
    # AUDIT FIX (M3, 2026-09-01): pass ckpt_path so the skip check ALSO verifies the record's own
    # provenance.md5 still matches the checkpoint currently on disk (closes the double-dump race).
    if _valid_remetric(out_path, ckpt_path=ckpt_path):
        print(f"SKIP (already valid, provenance md5 matches): {out_path}")
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
    # AUDIT FIX (M3, 2026-09-01): record the LOADED model's OWN shape/param-count, read back from
    # the checkpoint AFTER run_cell_round4 has already loaded+evaluated it under the SAME
    # capacity override -- never merely copied from the spec/cell dict. A capacity-override bug
    # that silently built the WRONG shape would otherwise pass every check that only compares the
    # spec's OWN arch_kw against itself (a check with no independent teeth).
    import h2h_cell_train_rd as ct
    loaded_model, _, _ = ct.load_h2h_checkpoint(ckpt_path, args.device)
    r["n_params_loaded"] = sum(p.numel() for p in loaded_model.parameters())
    r["d_model_loaded"] = loaded_model.d_model
    r["n_layers_loaded"] = len(loaded_model.blocks)
    from h2h_cell_train_rd import _atomic_dump
    _atomic_dump(out_path, r)
    print(f"REMETRIC {cell['name']}: acc_A={r['leg_a']['acc_A']:.4f} "
          f"n_params_loaded={r['n_params_loaded']}")


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


def _harvest_scenario(tmp_root: str, accs_by_config: dict, contender_accs: dict) -> dict:
    """Test-only harness for the selftest's harvest SCENARIO BATTERY (item 13): writes synthetic
    raw+remetric JSON pairs for every one of the 12 `all_configs()` configs (keyed by
    `_config_key`) under `tmp_root`, monkeypatches `_repo_root`/`REUSED_C0_20K_CELLS`/
    `CONTENDER_REFERENCE_REMETRIC` (module globals, restored in `finally` regardless of outcome)
    so the "reused" configs and the contender reference ALSO resolve to synthetic data instead of
    the real repo artifacts, then calls the REAL `mode_harvest()` end to end and returns its
    written `STRENGTHEN_VERDICT.json`. `accs_by_config`: `{config_key: [acc_s0, acc_s1, acc_s2]}`
    for all 12 keys. `contender_accs`: `{seed_idx: acc_A}` for the 3 contender reference seeds.

    Uses `sys.modules[__name__]` (never `import h2h_strengthen_rd`) to reach this module's OWN
    globals regardless of whether it is running as `__main__` (invoked directly) or as an
    imported module -- `import h2h_strengthen_rd` from inside itself would silently create a
    SECOND, separate module object when run as `__main__`, and patching that second copy's
    globals would have no effect on the `mode_harvest`/`_load_config_rows` functions actually
    executing."""
    self_mod = sys.modules[__name__]
    raw_dir = os.path.join(tmp_root, "raw")
    remetric_dir = os.path.join(tmp_root, "remetric")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(remetric_dir, exist_ok=True)

    def _write(name: str, acc: float, steps: int, arch_kw: dict) -> None:
        with open(os.path.join(raw_dir, f"{name}.json"), "w") as f:
            json.dump({"curve": [], "loss_first": 1.0, "loss_final_mean5": 1.0,
                      "step_count": steps}, f)
        with open(os.path.join(remetric_dir, f"{name}_round4.json"), "w") as f:
            json.dump({"leg_a": {"acc_A": acc, "chance": CHANCE_K32}, "arch_kw": arch_kw,
                      "steps_target": steps}, f)

    reused_override: dict = {}
    for cfg in all_configs():
        accs = accs_by_config[_config_key(cfg)]
        assert len(accs) == 3, f"need exactly 3 seed accs for {_config_key(cfg)}"
        for seed_idx, acc in zip(SEED_IDXS, accs):
            name = _cell_name(cfg["capacity"], cfg["lr"], cfg["steps"], seed_idx)
            _write(name, acc, cfg["steps"], CAPACITIES[cfg["capacity"]])
            if cfg["reused"]:
                reused_override[(cfg["lr"], seed_idx)] = {
                    "raw": os.path.relpath(os.path.join(raw_dir, f"{name}.json"), tmp_root),
                    "remetric": os.path.relpath(
                        os.path.join(remetric_dir, f"{name}_round4.json"), tmp_root),
                }

    contender_dir = os.path.join(tmp_root, "contender")
    os.makedirs(contender_dir, exist_ok=True)
    contender_override: dict = {}
    for seed_idx, acc in contender_accs.items():
        p = os.path.join(contender_dir, f"contender_s{seed_idx}.json")
        with open(p, "w") as f:
            json.dump({"leg_a": {"acc_A": acc}}, f)
        contender_override[seed_idx] = os.path.relpath(p, tmp_root)

    orig_repo_root, orig_reused, orig_contender = (
        self_mod._repo_root, dict(self_mod.REUSED_C0_20K_CELLS), dict(self_mod.CONTENDER_REFERENCE_REMETRIC))
    self_mod._repo_root = lambda: tmp_root
    self_mod.REUSED_C0_20K_CELLS.clear()
    self_mod.REUSED_C0_20K_CELLS.update(reused_override)
    self_mod.CONTENDER_REFERENCE_REMETRIC.clear()
    self_mod.CONTENDER_REFERENCE_REMETRIC.update(contender_override)
    try:
        out_dir = os.path.join(tmp_root, "out")
        args = argparse.Namespace(raw_dir=raw_dir, remetric_dir=remetric_dir, out_dir=out_dir)
        mode_harvest(args)
        with open(os.path.join(out_dir, "STRENGTHEN_VERDICT.json")) as f:
            return json.load(f)
    finally:
        self_mod._repo_root = orig_repo_root
        self_mod.REUSED_C0_20K_CELLS.clear()
        self_mod.REUSED_C0_20K_CELLS.update(orig_reused)
        self_mod.CONTENDER_REFERENCE_REMETRIC.clear()
        self_mod.CONTENDER_REFERENCE_REMETRIC.update(orig_contender)


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
    #    change without the design-doc ledger being re-derived). AUDIT FIX (M2): the ledger is
    #    now BLOCK-FLOP-scaled (~60.7 GPU-h), not the pre-audit naive param-ratio number
    #    (~30.35 GPU-h) -- both are exposed and checked so a future regression to the wrong
    #    model is caught either way.
    from h2h_strengthen_specs_gen import (estimate_gpu_h, estimate_gpu_h_naive,   # noqa: E402
                                          block_flop_ratio, param_ratio,
                                          C0_TRAIN_RATE_20K, RUN_METRIC_RATE_20K)
    ledger = sum(estimate_gpu_h(c["capacity"], c["steps"]) for c in strengthen_cells())
    ledger_naive = sum(estimate_gpu_h_naive(c["capacity"], c["steps"]) for c in strengthen_cells())
    ok7a = 55.0 < ledger < 65.0            # audit's own stated envelope: "~40-60 GPU-h" total;
                                            # this script's own block-FLOP formula lands at ~60.7,
                                            # at/near the top of that range -- disclosed, not hidden
    ok7b = 28.0 < ledger_naive < 33.0       # the pre-audit naive number, unchanged, for comparison
    ok7c = abs(C0_TRAIN_RATE_20K - 945.0 / 3600.0) < 1e-9   # M2's corrected anchor, exact
    rep("selftest 7: block-FLOP-scaled ledger (M2 fix) and the retained pre-audit naive ledger "
        "both compute to their expected envelopes; the anchor rate is the corrected 945s/20k "
        "figure, not the borrowed 908.79s", ok7a and ok7b and ok7c,
        f"block_flop_ledger={ledger:.3f} naive_ledger={ledger_naive:.3f} "
        f"anchor={C0_TRAIN_RATE_20K:.4f}")

    # 8) M2: block_flop_ratio reproduces the audit's own headline figures exactly (block params
    #    12x, head params 2x, combined ~7.3x at C2) -- proves this script's re-derivation is not
    #    merely SOME formula that happens to total roughly the right ballpark, but the SAME
    #    mechanism the audit itself described.
    from h2h_strengthen_specs_gen import _block_params, _head_params
    block_ratio_c2 = _block_params("C2") / _block_params("C0")
    head_ratio_c2 = _head_params("C2") / _head_params("C0")
    combined_ratio_c2 = block_flop_ratio("C2")
    rep("selftest 8: block_flop_ratio reproduces the audit's own headline figures -- block 12x, "
        "head 2x, combined ~7.3x at C2 (vs param_ratio's naive 3.09x)",
        11.9 < block_ratio_c2 < 12.1 and abs(head_ratio_c2 - 2.0) < 1e-9
        and 7.2 < combined_ratio_c2 < 7.3 and 3.0 < param_ratio("C2") < 3.2,
        f"block={block_ratio_c2:.4f} head={head_ratio_c2:.4f} combined={combined_ratio_c2:.4f} "
        f"naive_param_ratio={param_ratio('C2'):.4f}")

    # 9) M3: _valid_remetric requires provenance.md5 to match the checkpoint CURRENTLY on disk --
    #    closes the double-dump race. POSITIVE (matching md5 -> valid) and NEGATIVE (checkpoint
    #    bytes changed after the record was written -> md5 drift -> invalid, never silently
    #    trusted) both run to completion.
    tmp_ckpt2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_strengthen_selftest_ckpt2.pt")
    with open(tmp_ckpt2, "wb") as f:
        f.write(b"fake checkpoint bytes v1")
    tmp_rem = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_strengthen_selftest_rem.json")
    real_md5 = _md5_of_file(tmp_ckpt2)
    with open(tmp_rem, "w") as f:
        json.dump({"leg_a": {"acc_A": 0.5}, "provenance": {"path": tmp_ckpt2, "md5": real_md5}}, f)
    ok9a = _valid_remetric(tmp_rem, ckpt_path=tmp_ckpt2) is True
    with open(tmp_ckpt2, "wb") as f:
        f.write(b"DIFFERENT checkpoint bytes v2 -- simulates an overwritten/re-dumped checkpoint")
    ok9b = _valid_remetric(tmp_rem, ckpt_path=tmp_ckpt2) is False
    rep("selftest 9: _valid_remetric requires provenance.md5 == the checkpoint's CURRENT md5 -- "
        "PASSES when they match, FAILS (stale, not trusted) once the checkpoint changes underneath "
        "an unchanged remetric record -- the double-dump race, closed", ok9a and ok9b)
    os.remove(tmp_ckpt2)
    os.remove(tmp_rem)

    # 10) M3: mode_run_cell's skip-if-valid path now requires step_count == the cell's real
    #     target, factored into `_existing_train_result_is_current` for direct testability
    #     (spawning a real training run inside the selftest would defeat the point of a fast CPU
    #     selftest). A stale smoke output (step_count=300) sitting at a cell's real --out path
    #     (target 60,000) must NOT be treated as "already done".
    tmp_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_strengthen_selftest_out.json")
    stale_cell = {"name": "x", "steps": 60_000}
    with open(tmp_out, "w") as f:
        json.dump({"arch": "transformer", "task": "task1_sweep", "seed_idx": 0,
                  "final_metric": 0.1, "step_count": 300}, f)   # a 300-step SMOKE's own output
    ok10a = _existing_train_result_is_current(tmp_out, stale_cell) is False
    with open(tmp_out, "w") as f:
        json.dump({"arch": "transformer", "task": "task1_sweep", "seed_idx": 0,
                  "final_metric": 0.1, "step_count": 60_000}, f)   # the REAL target, trained fully
    ok10b = _existing_train_result_is_current(tmp_out, stale_cell) is True
    rep("selftest 10: a stale/under-trained raw JSON (step_count=300) at a cell's real --out path "
        "(target 60,000) is correctly rejected as NOT current; the same path with the real "
        "step_count IS accepted", ok10a and ok10b)
    os.remove(tmp_out)

    # 11) M3: the loaded-model provenance fields (n_params_loaded/d_model_loaded/n_layers_loaded)
    #     -- simulates exactly what _remetric_one does AFTER run_cell_round4 returns: reload the
    #     checkpoint under the SAME override and read the LOADED model's own shape, never merely
    #     copy the spec's own arch_kw back at itself (a check with no independent teeth).
    tiny_c2 = dict(d_model=40, n_layers=5, n_heads=4, ffn_mult=2)
    apply_capacity_override(tiny_c2)
    m11 = ct.build_arm_model("transformer", 500, seed=1, device="cpu")
    tmp_ckpt3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_strengthen_selftest_ckpt3.pt")
    torch.save({"arch": "transformer", "vocab_size_total": 500, "model": m11.state_dict()}, tmp_ckpt3)
    loaded11, _, _ = ct.load_h2h_checkpoint(tmp_ckpt3, "cpu")
    n_params_loaded = sum(p.numel() for p in loaded11.parameters())
    ok11 = (n_params_loaded == sum(p.numel() for p in m11.parameters())
           and loaded11.d_model == 40 and len(loaded11.blocks) == 5)
    rep("selftest 11: the loaded-model provenance fields (n_params_loaded/d_model_loaded/"
        "n_layers_loaded) read back the ACTUAL loaded model's own shape, matching what was saved",
        ok11, f"n_params_loaded={n_params_loaded} d_model={loaded11.d_model} "
        f"n_layers={len(loaded11.blocks)}")
    os.remove(tmp_ckpt3)

    # 12) M4: the probe spec (strengthen_specs_probe/0599_...) exists, targets the correct (most
    #     expensive) cell identity, checks step_count==300 (never steps_target, which correctly
    #     stays 60,000 -- the cell's real identity is unchanged), n_params==44,613,632, and uses
    #     DISTINCT out/remetric/ckpt paths from every main spec.
    import h2h_strengthen_specs_gen as gen
    probe_path = os.path.join(gen.PROBE_SPECS_DIR, f"{gen.PROBE_SPEC_ID}.json")
    ok12 = False
    detail12 = "probe spec not found -- run h2h_strengthen_specs_gen.py first"
    if os.path.isfile(probe_path):
        with open(probe_path) as f:
            probe_spec = json.load(f)
        vcheck = probe_spec["validity_check"]
        main_spec_example = gen._spec_for_cell(strengthen_cells()[0], gen.SPEC_ID_START)
        ok12 = (f"--steps-override {gen.PROBE_STEPS_OVERRIDE}" in probe_spec["cmd"]
               and gen.PROBE_CELL_NAME in probe_spec["cmd"]
               and "step_count') == 300" in vcheck and "n_params') == 44613632" in vcheck
               and "steps_target" not in vcheck
               and probe_spec["output_dir"] != main_spec_example["output_dir"]
               and gen.PROBE_CKPT_DIR != gen.BOX_CKPT_DIR)
        detail12 = f"cmd_has_override={'--steps-override 300' in probe_spec['cmd']}"
    rep("selftest 12: the M4 gating probe spec exists, targets steps_override=300 on the most "
        "expensive cell identity, checks step_count==300 (not steps_target) + n_params==C2's "
        "44,613,632, and uses paths DISTINCT from every main spec", ok12, detail12)

    # 13) harvest SCENARIO BATTERY -- exercises the REAL mode_harvest() end to end (not just the
    #     dict-math replica in item 5) across synthetic Outcome A/B/C data, via a monkeypatched
    #     _repo_root + REUSED_C0_20K_CELLS + CONTENDER_REFERENCE_REMETRIC (restored in `finally`).
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_root:
        battery_ok = True
        target_key = "C2_lr1e-03_st60000"   # a FRESH (non-reused) config -- the most expensive cell
        assert target_key in [_config_key(c) for c in all_configs() if not c["reused"]]

        def scenario(target_accs, contender_accs=None):
            accs_by_config = {_config_key(c): [0.02, 0.025, 0.03] for c in all_configs()}
            accs_by_config[target_key] = target_accs
            return _harvest_scenario(tmp_root, accs_by_config,
                                     contender_accs or {0: 1.0, 1: 1.0, 2: 1.0})

        doc_a = scenario([0.03, 0.04, 0.035])
        battery_ok &= doc_a["outcome"] == "OUTCOME_A_NON_COMPETITIVE" and doc_a["ratio_report"] is None
        rep("selftest 13a: harvest battery Outcome A (every config below bar) -> "
            "OUTCOME_A_NON_COMPETITIVE, no ratio_report", battery_ok, f"outcome={doc_a['outcome']}")

        doc_b = scenario([0.15, 0.10, 0.20])   # clears (>=2/3 >= 0.09375), not competitive (<0.50)
        delta_b = doc_b["ratio_report"]["paired_delta_contender_minus_transformer"]["mean"] if doc_b["ratio_report"] else None
        ok13b = (doc_b["outcome"] == "OUTCOME_B_CLEARS_NOT_COMPETITIVE"
                and doc_b["ratio_report"] is not None
                and doc_b["ratio_report"]["best_config"] == target_key
                and delta_b is not None and abs(delta_b - 0.85) < 1e-9)   # (1-.15)+(1-.10)+(1-.20) / 3
        rep("selftest 13b: harvest battery Outcome B (one config clears, none competitive) -> "
            "OUTCOME_B_CLEARS_NOT_COMPETITIVE, ratio_report's paired delta-CI mean is EXACTLY "
            "0.85 against a known contender_acc_A=1.0 reference (hand-verified)", ok13b,
            f"outcome={doc_b['outcome']} delta_mean={delta_b}")

        doc_c = scenario([0.60, 0.55, 0.70])   # competitive (>=2/3 >= 0.50)
        ok13c = (doc_c["outcome"] == "OUTCOME_C_COMPETITIVE"
                and doc_c["ratio_report"] is not None
                and doc_c["ratio_report"]["best_config"] == target_key)
        rep("selftest 13c: harvest battery Outcome C (one config competitive) -> "
            "OUTCOME_C_COMPETITIVE, ratio_report points at that config", ok13c,
            f"outcome={doc_c['outcome']}")

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
    ap.add_argument("--ckpt-dir", type=str, default="/ephemeral/h2h_strengthen_ckpts")   # M2 fix
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
