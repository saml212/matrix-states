#!/usr/bin/env python
"""h2h_strengthen_specs_gen.py -- generates the box queue-spec JSONs for
the sec 1.46 baseline-strengthening sweep (h2h_strengthen_rd.py) and holds
the GPU-h cost model both the design record and h2h_strengthen_rd.py's own
selftest (items 7-8) cite.

ROUND-1 AUDIT (2026-09-01, 4 MAJOR): naive total-param-count ratio and a
cross-program anchor rate, both corrected to a block-FLOP model with an
in-lineage anchor -- see `block_flop_ratio`'s own docstring.

ROUND-2 AUDIT (2026-09-01, PASS-STAGE-30-AFTER-PROBE, 0 FATAL/0 MAJOR, 4
minor, applied here):
  m1 the round-1 fix's own anchor citation was ITSELF wrong: "941-952
     s/cell" and "~24 s/cell" are NOT literal `fix5 MANIFEST.md` text --
     they are `HEAD_TO_HEAD_DEMO_DESIGN.md` sec 1.44's own PRE-fix5 cost
     PROJECTION (based on the REUSED round-4-sweep cells' rate, a related
     but DIFFERENT measurement than fix5's own 9 fresh cells). Corrected
     below to cite the actual fix5 raw/remetric JSON `wall_s` fields
     directly, with TWO disclosed anchors (fast-cluster / realized) per
     the audit's own instruction.
  m2 the re-metric stage now ALSO gets a `timeout -k 60 30m` wrapper (the
     round-1 fix only wrapped the train stage).
  m3 `_validity_check_lines` now ALSO asserts the remetric JSON's own
     `n_params_loaded` (the LOADED-model provenance field h2h_strengthen_rd
     writes) matches the capacity's formula-derived count -- independent
     of the pre-existing raw-JSON `n_params` check (a different artifact,
     a different bug class: this one catches a capacity mixup that
     survived training but was never actually re-loaded correctly at
     re-metric time).
  minor(TRIM) `C2 x lr=3e-4 x 60,000 steps` (specs 0621-0623, the most
     expensive 3 cells) is DEFERRED to `strengthen_specs_deferred/`, a
     pre-registered narrowing (see `is_deferred`/`strengthen_specs_deferred/
     README.md`) BEFORE staging, with a pre-registered conditional re-add.

COST MODEL (m1, corrected citations). The fix5 round (`h2h_fix5_lrgrid_rd.py`,
sec 1.44/1.45) trained 9 transformer x task1_sweep cells at 20,000 steps
each. Reading their OWN raw JSONs directly
(`experiment-runs/2026-07-11_h2h_fix5_lrgrid/results/*.json`'s own
`wall_s` field, re-verified by this script's own build, not merely cited
from a secondary summary):
  6/9 cells, no contention: 941.7734, 943.0457, 944.8535, 948.2702,
    948.5031, 952.6023 s -- mean 946.5080 s -> **the FAST-CLUSTER anchor**
  3/9 cells, co-tenant contention (fix5's own MANIFEST.md calls these
    "the two outliers," undercounting by one -- there are THREE):
    992.3003, 1080.8125, 1162.6606 s
  all 9 cells: mean 990.5357 s -> **the REALIZED anchor** (includes the
    contention outliers -- the more conservative of the two)
Re-metric (`experiment-runs/2026-07-11_h2h_fix5_lrgrid/results/remetric/
*.json`'s own `wall_s`, all 9 cells): sum 186.1601 s (matches the
MANIFEST's own stated "186.2 s" total to the first decimal -- an
independent cross-check that these ARE the artifacts the MANIFEST
summarized), mean 20.6845 s/cell = 0.0057457 GPU-h/cell. Used for BOTH
anchors (remetric shows no comparable contention-outlier pattern in
these 9 cells).

`estimate_gpu_h` (THE per-spec ledger figure) uses the FAST-CLUSTER
anchor, stated first per the audit's own "60.67 vs ~63.4" framing;
`estimate_gpu_h_realized` uses the REALIZED anchor, disclosed alongside
it; `estimate_gpu_h_naive` is the round-1-superseded naive param-ratio
number, retained for disclosure only. ALL THREE are superseded by the
0599 probe's own measured rate (`strengthen_reprice.py`) before any
staged spec runs for real -- neither anchor here is a substitute for
that on-box measurement, both are design-time estimates for sizing the
pre-launch ceremony tier and the (separately pinned) TIMEOUT_HOURS.

  block_flop_ratio(cap): see that function's own docstring (unchanged
  from round 1 -- reproduces the audit's own 12x-block/2x-head/~7.3x-
  combined figures at C2 exactly).

Run standalone to (re)write the 27 staged main specs, the 3 deferred
specs, and the 1 probe spec:
  python h2h_strengthen_specs_gen.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_strengthen_rd import (CAPACITIES, LR_GRID, STEPS_GRID, FULL_STEPS_C0,   # noqa: E402
                               LONG_STEPS, strengthen_cells, _lr_str)

VOCAB_SIZE_TOTAL = 50259   # DEPLOY-PIN-3 (h2h_cell_train_rd.py): GPT-2 base 50257 + <Q> + BUFFER
N_QUERY_TRAIN = 8          # DEPLOY-PIN-1 (h2h_cell_train_rd.py): the transformer's tap/answer
                            # route replicates the FULL block stack once per query -- this many
                            # extra (1+N_QUERY_TRAIN)-pass multiplier on block (not head) compute

# --- AUDIT FIX (round 2, m1, 2026-09-01): re-cited directly from the fix5 raw/remetric JSONs'
# own `wall_s` fields (module docstring above has the full per-cell breakdown and provenance). ---
FAST_CLUSTER_TRAIN_RATE_20K = 946.5080379645029 / 3600.0   # = 0.262919 GPU-h/20k-cell (6/9 cells)
REALIZED_TRAIN_RATE_20K = 990.5357400046455 / 3600.0        # = 0.275149 GPU-h/20k-cell (all 9)
RUN_METRIC_RATE_20K = 186.1600570678711 / 9 / 3600.0        # = 0.0057457 GPU-h/cell (all 9;
                                                              # corrected from the round-1 fix's
                                                              # own wrong "~24s"/0.0067 citation)

# THE primary per-spec ledger anchor (fast-cluster, stated first per the audit's own framing).
# Kept under this historical name for h2h_strengthen_rd.py's own selftest imports.
C0_TRAIN_RATE_20K = FAST_CLUSTER_TRAIN_RATE_20K

# --- AUDIT FIX (M1, round 1, 2026-09-01): per-spec train-stage wall-clock timeout, `timeout -k
# 120 <N>h`. N = 2x the auditor's own corrected upper estimate per (capacity, steps) cell -- a
# SEPARATELY pinned table (not re-derived from the anchors above); superseded by the RE-PRICE
# RULE (strengthen_reprice.py) if the 0599 probe's own measured rate says otherwise. ---
TIMEOUT_HOURS = {
    ("C0", 60_000): 3.0,
    ("C1", 20_000): 1.5,
    ("C1", 60_000): 4.0,
    ("C2", 20_000): 5.0,
    ("C2", 60_000): 12.0,
}
TIMEOUT_KILL_GRACE_S = 120   # `timeout -k 120` -- SIGKILL 120s after SIGTERM if still alive

# --- AUDIT FIX (m2, round 2, 2026-09-01): the re-metric stage ALSO gets a wall-clock ceiling --
# round 1 only wrapped the train stage. 30 minutes is generous vs the fix5 precedent's own
# realized ~19-32s/cell re-metric time at C0; re-metric cost scales with capacity via
# block_flop_ratio same as training, so a much larger C2 model's re-metric could plausibly run
# into minutes, not seconds -- 30m stays a safe multiple even then. ---
REMETRIC_TIMEOUT_MIN = 30
REMETRIC_TIMEOUT_KILL_GRACE_S = 60

SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strengthen_specs")
DEFERRED_SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strengthen_specs_deferred")
PROBE_SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strengthen_specs_probe")
SPEC_ID_START = 600   # sorts AFTER the running 1.31B K=16 grace wave (0478-0485)
PROBE_SPEC_ID = "0599_h2h_strengthen_probe_C2"   # sorts BEFORE 0600 -- the probe runs FIRST

BOX_CWD = "/home/nvidia/chapter2/deltanet_rd"
BOX_PY = "/home/nvidia/tdenv/bin/python3"
# AUDIT FIX (M2, round 1, 2026-09-01): checkpoints on /ephemeral per STATE.md's disk policy.
BOX_CKPT_DIR = "/ephemeral/h2h_strengthen_ckpts"
BOX_OUT_DIR = f"{BOX_CWD}/results/h2h_rung1/strengthen"
BOX_REMETRIC_DIR = f"{BOX_OUT_DIR}/remetric"
BOX_GATES_DIR = f"{BOX_CWD}/results/h2h_rung1/gates"
BOX_MARGINS_TOKEN = f"{BOX_CWD}/results/h2h_rung1/MARGINS_FROZEN.token"
DIAL_ROUND = 4

# --- Probe (M4, round 1, 2026-09-01): DISTINCT paths from the main sweep's own. ---
PROBE_OUT_DIR = f"{BOX_CWD}/results/h2h_rung1/strengthen_probe"
PROBE_REMETRIC_DIR = f"{PROBE_OUT_DIR}/remetric"
PROBE_CKPT_DIR = "/ephemeral/h2h_strengthen_probe_ckpts"
PROBE_CELL_NAME = "h2h_strengthen_C2_lr1e-03_st60000_s0"
PROBE_STEPS_OVERRIDE = 300
PROBE_TRAIN_TIMEOUT_MIN = 20   # generous vs the auditor's own "~2 min" expectation
# The EXACT filenames the 0599 probe spec writes (queue watcher interface contract):
PROBE_RAW_FILENAME = f"{PROBE_CELL_NAME}.json"                        # directly in PROBE_OUT_DIR
PROBE_REMETRIC_FILENAME = f"{PROBE_CELL_NAME}_round4.json"            # in PROBE_OUT_DIR/remetric/


def is_deferred(cell: dict) -> bool:
    """TRIM (round-2 audit, coordinator election, 2026-09-01): C2 x lr=3e-4 x 60,000 steps (3
    seeds, specs 0621-0623 -- the three single most expensive cells in the whole sweep) is
    DEFERRED, not staged, pending the pre-registered conditional re-add written into sec 1.46
    and `strengthen_specs_deferred/README.md`."""
    return cell["capacity"] == "C2" and cell["lr"] == 3e-4 and cell["steps"] == LONG_STEPS


def staged_cells() -> list[dict]:
    return [c for c in strengthen_cells() if not is_deferred(c)]


def deferred_cells() -> list[dict]:
    return [c for c in strengthen_cells() if is_deferred(c)]


def _params(cap_id: str) -> int:
    from transformer_baseline_rd import count_transformer_params
    kw = CAPACITIES[cap_id]
    return count_transformer_params(VOCAB_SIZE_TOTAL, kw["d_model"], kw["n_layers"], kw["ffn_mult"])


def _block_params(cap_id: str) -> int:
    """attn+ffn+norm params only (n_layers * per_layer), EXCLUDING the vocab embedding/tied head
    -- the part of compute incurred on EVERY forward pass through the stack, including the
    N_QUERY_TRAIN query-replicated tap passes."""
    kw = CAPACITIES[cap_id]
    d, nl, ffn = kw["d_model"], kw["n_layers"], kw["ffn_mult"]
    per_layer = 2 * d + 4 * d ** 2 + 2 * ffn * d ** 2
    return nl * per_layer


def _head_params(cap_id: str) -> int:
    """the tied vocab-embedding/output-projection term (vocab*d_model) -- incurred ONCE per step
    (the main next-token CE forward over the un-replicated context), never on the
    N_QUERY_TRAIN-replicated tap passes (`transformer_native_tap`'s `return_hidden=True` call
    explicitly skips the LM-head matmul)."""
    kw = CAPACITIES[cap_id]
    return VOCAB_SIZE_TOTAL * kw["d_model"]


def param_ratio(cap_id: str) -> float:
    """PRE-AUDIT (round 1, naive) total-param-count ratio. Retained for disclosure only -- see
    `estimate_gpu_h_naive`."""
    return _params(cap_id) / _params("C0")


def block_flop_ratio(cap_id: str) -> float:
    """AUDIT-CORRECTED (round 1, M2) per-step FLOP proxy: C = (1+N_QUERY_TRAIN)*block_params +
    head_params (proportional units -- the constant backward-pass multiplier cancels in a ratio
    to C0)."""
    passes = 1 + N_QUERY_TRAIN
    c = passes * _block_params(cap_id) + _head_params(cap_id)
    c0 = passes * _block_params("C0") + _head_params("C0")
    return c / c0


def estimate_gpu_h_naive(cap_id: str, steps: int) -> float:
    """PRE-AUDIT (round 1) total-param-ratio estimate. Superseded -- kept only for disclosure."""
    ratio = param_ratio(cap_id)
    train = C0_TRAIN_RATE_20K * ratio * (steps / FULL_STEPS_C0)
    remetric = RUN_METRIC_RATE_20K * ratio
    return train + remetric


def estimate_gpu_h(cap_id: str, steps: int) -> float:
    """THE per-spec ledger figure: block-FLOP-scaled, FAST-CLUSTER anchor (m1)."""
    ratio = block_flop_ratio(cap_id)
    train = FAST_CLUSTER_TRAIN_RATE_20K * ratio * (steps / FULL_STEPS_C0)
    remetric = RUN_METRIC_RATE_20K * ratio
    return train + remetric


def estimate_gpu_h_realized(cap_id: str, steps: int) -> float:
    """DISCLOSED alongside `estimate_gpu_h` (m1): block-FLOP-scaled, REALIZED anchor (includes
    the fix5 round's own observed co-tenant-contention outliers -- the more conservative read)."""
    ratio = block_flop_ratio(cap_id)
    train = REALIZED_TRAIN_RATE_20K * ratio * (steps / FULL_STEPS_C0)
    remetric = RUN_METRIC_RATE_20K * ratio
    return train + remetric


def checkpoint_mb(cap_id: str) -> float:
    """fp32 model-only checkpoint size (no optimizer state saved -- verified against the C0
    reference: 14.44M params -> 57.8MB matches the fix5 MANIFEST's own measured checkpoint
    size exactly)."""
    return _params(cap_id) * 4 / 1e6


def _validity_check_lines(remetric_json: str, out_json: str, arch_kw: dict, steps: int,
                          n_params: int, check_steps_target: bool = True) -> str:
    """Every python string literal below is deliberately SINGLE-quoted, with zero embedded
    double quotes -- the whole one-liner is wrapped in DOUBLE quotes for the outer
    `bash -c "$vcheck"` (queue_worker.sh's own invocation); a double-quoted literal (e.g. from
    json.dumps) embedded inside that outer double-quoted argument would prematurely close the
    shell's quoting and truncate the command (caught live during round 1's build -- simulated
    queue_worker.sh's exact invocation pattern against a real spec). `arch_kw` is rendered via
    repr() (single-quoted), never json.dumps() (double-quoted), for the same reason.

    AUDIT FIX (round 1, M3): the checkpoint-md5-vs-provenance-record check (closes the
    'double-dump race') and the raw JSON's own recorded `n_params` matching the capacity's
    formula-derived count.

    AUDIT FIX (round 2, m3, 2026-09-01): ALSO asserts the remetric JSON's own `n_params_loaded`
    (the LOADED-model provenance field `h2h_strengthen_rd._remetric_one` writes, read back from
    the checkpoint AFTER `run_cell_round4` returns) matches the SAME capacity's formula-derived
    count -- independent of, and a different bug class from, the pre-existing raw-JSON `n_params`
    check: that one catches a mixup that happened at TRAIN time; this one catches a mixup that
    happened at RE-METRIC time (e.g. a capacity override left stale between the two stages of a
    chained cmd, or a wrong checkpoint loaded under the right override)."""
    arch_kw_repr = repr(arch_kw)
    steps_target_check = f"assert d.get('steps_target') == {steps}; " if check_steps_target else ""
    return (
        "import json, math, hashlib; "
        f"d = json.load(open('{remetric_json}')); "
        "assert 'leg_a' in d and math.isfinite(d['leg_a']['acc_A']); "
        f"assert d.get('arch_kw') == {arch_kw_repr}, ('arch_kw mismatch: ' + str(d.get('arch_kw'))); "
        f"assert d.get('n_params_loaded') == {n_params}, ('n_params_loaded mismatch: ' + str(d.get('n_params_loaded'))); "
        f"{steps_target_check}"
        "prov = d.get('provenance', {}); "
        "ckpt_md5 = hashlib.md5(open(prov['path'], 'rb').read()).hexdigest(); "
        "assert ckpt_md5 == prov['md5'], 'checkpoint md5 drift vs the remetric record -- stale/overwritten checkpoint'; "
        f"raw = json.load(open('{out_json}')); "
        f"assert raw.get('step_count') == {steps}, ('step_count mismatch: ' + str(raw.get('step_count'))); "
        f"assert raw.get('n_params') == {n_params}, ('n_params mismatch: ' + str(raw.get('n_params')))"
    )


def _spec_for_cell(cell: dict, spec_id: int) -> dict:
    name = cell["name"]
    out_json = f"{BOX_OUT_DIR}/{name}.json"
    remetric_json = f"{BOX_REMETRIC_DIR}/{name}_round4.json"
    train_timeout_h = TIMEOUT_HOURS[(cell["capacity"], cell["steps"])]
    train_cmd = (f"timeout -k {TIMEOUT_KILL_GRACE_S} {train_timeout_h:g}h "
                f"{BOX_PY} h2h_strengthen_rd.py --run-cell {name} "
                f"--out {out_json} --ckpt-dir {BOX_CKPT_DIR} "
                f"--gates-dir {BOX_GATES_DIR} --margins-token {BOX_MARGINS_TOKEN} --device cuda")
    # AUDIT FIX (m2, round 2): re-metric stage now ALSO wrapped in a timeout.
    remetric_cmd = (f"timeout -k {REMETRIC_TIMEOUT_KILL_GRACE_S} {REMETRIC_TIMEOUT_MIN}m "
                    f"{BOX_PY} h2h_strengthen_rd.py --remetric --run-cell {name} "
                    f"--ckpt-dir {BOX_CKPT_DIR} --remetric-dir {BOX_REMETRIC_DIR} "
                    f"--dial-round {DIAL_ROUND} --gates-dir {BOX_GATES_DIR} "
                    f"--margins-token {BOX_MARGINS_TOKEN} --device cuda")
    cmd = (f"export HEADTOHEAD_PI_SIGNOFF=1 HEADTOHEAD_MATCH_GATE_SIGNOFF=1 "
          f"H2H_DIAL_ROUND={DIAL_ROUND} && cd {BOX_CWD} && {train_cmd} && {remetric_cmd}")
    n_params = _params(cell["capacity"])
    vcheck_body = _validity_check_lines(remetric_json, out_json, cell["arch_kw"], cell["steps"],
                                        n_params)
    validity_check = f'{BOX_PY} -c "{vcheck_body}"'
    gpu_h = round(estimate_gpu_h(cell["capacity"], cell["steps"]), 4)
    gpu_h_realized = round(estimate_gpu_h_realized(cell["capacity"], cell["steps"]), 4)
    gpu_h_naive = round(estimate_gpu_h_naive(cell["capacity"], cell["steps"]), 4)
    deferred_note = (" DEFERRED per sec 1.46's TRIM -- see strengthen_specs_deferred/README.md; "
                     "this file is NOT staged into strengthen_specs/." if is_deferred(cell) else "")
    return {
        "id": f"{spec_id:04d}_h2h_strengthen_{cell['capacity']}_lr{_lr_str(cell['lr'])}_st{cell['steps']}_s{cell['seed_idx']}",
        "lane": "strengthen",
        "hypothesis": (f"H2H transformer baseline (Task 1 recall) at capacity {cell['capacity']} "
                      f"({CAPACITIES[cell['capacity']]}), lr={cell['lr']:g}, "
                      f"{cell['steps']} steps, seed_idx={cell['seed_idx']}: does capacity + "
                      "training-length + the fix5 LR grid together clear the frozen 0.09375 "
                      "demonstration bar (sec 1.46)? STAGED ONLY AFTER 0599's probe validates "
                      f"(sec 1.46, M4).{deferred_note}"),
        "cmd": cmd,
        "gpu_h_estimate": gpu_h,
        "output_dir": BOX_OUT_DIR,
        "validity_check": validity_check,
        "notes": (f"AUDIT-CORRECTED (round 2, m1) block-FLOP estimate at the FAST-CLUSTER anchor "
                 f"(block_flop_ratio={block_flop_ratio(cell['capacity']):.4f}, anchor="
                 f"{FAST_CLUSTER_TRAIN_RATE_20K:.6f} GPU-h/20k-cell, mean of the fix5 round's own "
                 f"6/9 non-contended cells); at the REALIZED anchor (all 9 fix5 cells incl. "
                 f"contention, {REALIZED_TRAIN_RATE_20K:.6f} GPU-h/20k-cell): {gpu_h_realized:.4f} "
                 f"GPU-h. Round-1 (superseded) naive param-ratio estimate was {gpu_h_naive:.4f} "
                 f"GPU-h (param_ratio={param_ratio(cell['capacity']):.4f}) -- all three disclosed "
                 f"in sec 1.46. Train stage wrapped `timeout -k {TIMEOUT_KILL_GRACE_S} "
                 f"{train_timeout_h:g}h`; re-metric stage wrapped `timeout -k "
                 f"{REMETRIC_TIMEOUT_KILL_GRACE_S} {REMETRIC_TIMEOUT_MIN}m` (m2, round 2) -- on "
                 "kill, no result JSON is written, validity fails, the GPU frees. Checkpoint "
                 f"~{checkpoint_mb(cell['capacity']):.1f} MB fp32 model-only at "
                 f"{BOX_CKPT_DIR} (/ephemeral, per STATE.md's disk policy, NOT /data). ALL THREE "
                 "estimates are superseded by the 0599 probe's own measured rate "
                 "(strengthen_reprice.py) before staging."),
    }


def build_all_specs() -> list[dict]:
    """ALL 30 cells (staged + deferred), in `strengthen_cells()`'s own order -- ids
    SPEC_ID_START..SPEC_ID_START+29 are assigned here UNCHANGED by the TRIM (the deferred cells
    keep their original 0621-0623 identity, they are just written to a different directory by
    `write_all_specs`, never renumbered)."""
    specs = []
    for i, cell in enumerate(strengthen_cells()):
        specs.append(_spec_for_cell(cell, SPEC_ID_START + i))
    assert len(specs) == 30
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids)), "spec id collision"
    return specs


def build_probe_spec() -> dict:
    """M4 (round 1): a 300-step probe of the single MOST EXPENSIVE cell (C2, 60,000-step target,
    the real spec 0615) -- gates the 27-cell staged wave. DISTINCT out/remetric/ckpt paths from
    the real cell's own; validity checks step_count==300 (never steps_target, which correctly
    stays 60,000), a finite acc_A, and the exact C2 param count (44,613,632) -- catching a
    capacity-override mixup before any staged cell is dispatched. Re-metric stage also timeout-
    wrapped (m2, round 2)."""
    n_params_c2 = _params("C2")
    out_json = f"{PROBE_OUT_DIR}/{PROBE_RAW_FILENAME}"
    remetric_json = f"{PROBE_REMETRIC_DIR}/{PROBE_REMETRIC_FILENAME}"
    train_cmd = (f"timeout -k 60 {PROBE_TRAIN_TIMEOUT_MIN}m {BOX_PY} h2h_strengthen_rd.py "
                f"--run-cell {PROBE_CELL_NAME} --steps-override {PROBE_STEPS_OVERRIDE} "
                f"--out {out_json} --ckpt-dir {PROBE_CKPT_DIR} "
                f"--gates-dir {BOX_GATES_DIR} --margins-token {BOX_MARGINS_TOKEN} --device cuda")
    remetric_cmd = (f"timeout -k {REMETRIC_TIMEOUT_KILL_GRACE_S} {REMETRIC_TIMEOUT_MIN}m "
                    f"{BOX_PY} h2h_strengthen_rd.py --remetric --run-cell {PROBE_CELL_NAME} "
                    f"--ckpt-dir {PROBE_CKPT_DIR} --remetric-dir {PROBE_REMETRIC_DIR} "
                    f"--dial-round {DIAL_ROUND} --gates-dir {BOX_GATES_DIR} "
                    f"--margins-token {BOX_MARGINS_TOKEN} --device cuda")
    cmd = (f"export HEADTOHEAD_PI_SIGNOFF=1 HEADTOHEAD_MATCH_GATE_SIGNOFF=1 "
          f"H2H_DIAL_ROUND={DIAL_ROUND} && cd {BOX_CWD} && {train_cmd} && {remetric_cmd}")
    vcheck_body = _validity_check_lines(remetric_json, out_json, CAPACITIES["C2"],
                                        PROBE_STEPS_OVERRIDE, n_params_c2,
                                        check_steps_target=False)
    validity_check = f'{BOX_PY} -c "{vcheck_body}"'
    ratio = block_flop_ratio("C2")
    probe_gpu_h = round(FAST_CLUSTER_TRAIN_RATE_20K * ratio * (PROBE_STEPS_OVERRIDE / FULL_STEPS_C0)
                        + RUN_METRIC_RATE_20K * ratio, 4)
    real_spec_id = next(f"{SPEC_ID_START + i:04d}" for i, c in enumerate(strengthen_cells())
                        if c["name"] == PROBE_CELL_NAME)
    return {
        "id": PROBE_SPEC_ID,
        "lane": "strengthen_probe",
        "hypothesis": (f"CALIBRATION PROBE (sec 1.46, M4): {PROBE_STEPS_OVERRIDE} steps of the "
                      "single most expensive STAGED strengthening cell (C2, 44.61M params, the "
                      f"60,000-step target of the real spec {real_spec_id}) -- measures real box "
                      "s/step before any of the 27 staged main specs (0600-0629 minus the "
                      "0621-0623 TRIM) are staged. The staged wave is gated on this probe "
                      "validating; strengthen_reprice.py consumes its raw+remetric JSONs."),
        "cmd": cmd,
        "gpu_h_estimate": probe_gpu_h,
        "output_dir": PROBE_OUT_DIR,
        "validity_check": validity_check,
        "notes": (f"Auditor's own estimate: ~2 min. This script's own block-FLOP formula (fast-"
                 f"cluster anchor) gives ~{probe_gpu_h * 3600:.0f}s ({probe_gpu_h:.4f} GPU-h) for "
                 "train+remetric combined -- same order of magnitude. DISTINCT paths from the "
                 f"real 0615 spec's own ({BOX_OUT_DIR} vs {PROBE_OUT_DIR}; {BOX_CKPT_DIR} vs "
                 f"{PROBE_CKPT_DIR}) even though this probe trains the IDENTICAL cell identity. "
                 f"Raw JSON: {PROBE_RAW_FILENAME} (directly in output_dir); remetric JSON: "
                 f"remetric/{PROBE_REMETRIC_FILENAME}. Its measured s/step feeds "
                 "strengthen_reprice.py's RE-PRICE RULE (sec 1.46) before 0600-0629 (minus the "
                 "TRIM) move into the pending queue."),
    }


def _dump(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


_DEFERRED_README = """# strengthen_specs_deferred/ -- TRIM (round-2 audit, coordinator election, 2026-09-01)

`0621_h2h_strengthen_C2_lr3e-04_st60000_s0.json`,
`0622_h2h_strengthen_C2_lr3e-04_st60000_s1.json`,
`0623_h2h_strengthen_C2_lr3e-04_st60000_s2.json`
(C2 x lr=3e-4 x 60,000 steps, 3 seeds -- the three single most expensive
cells in the whole sweep, ~17.25 GPU-h at the fast-cluster anchor / ~18.05
GPU-h at the realized anchor) are DEFERRED, NOT staged into
`strengthen_specs/`, pending a pre-registered conditional re-add.

**Why:** with C2 x lr=1e-3 already covering C2's own best-reading LR at
both step counts (20k and 60k), the marginal information from ALSO
running C2's frozen-default LR (3e-4) at the longest, most expensive step
count is lower than for the other 24 staged cells, and dropping it moves
the design-time ledger from ~60.67/~63.4 GPU-h (30 cells, fast-cluster/
realized anchors) to ~43.43/~45.4 GPU-h (27 cells) -- narrowing the
pre-launch ceremony gap.

**Pre-registered conditional re-add (decided NOW, before any cell's
result exists -- see HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.46):** once the
27 staged cells' harvest reports `mean_acc_A` for both `C2 x lr=3e-4 x
20,000` and `C2 x lr=1e-3 x 20,000`, IF the former's mean exceeds the
latter's (the frozen-default LR outperforming C2's own currently-best LR
at the SAME, cheaper step count -- inverting the fix5-established
LR ranking specifically at this capacity), these 3 deferred cells run as
a SEPARATE follow-on, budgeted at the SAME ceiling this directory's own
files were priced at (<=17.25 GPU-h, fast-cluster anchor; re-price before
that follow-on launches, exactly as sec 1.46's RE-PRICE RULE requires for
the staged wave). If the condition does NOT fire, these files stay
deferred indefinitely -- never launched, never silently re-added.

**Outcomes A/B/C (sec 1.46's own decision rule) are UNCHANGED by this
TRIM** -- these 3 cells were never load-bearing for any of the three
outcomes (C2 x lr=1e-3 already covers both step counts; C2 x lr=3e-4 at
20,000 steps is unaffected, only its 60,000-step sibling is deferred).
"""


def write_all_specs() -> None:
    os.makedirs(SPECS_DIR, exist_ok=True)
    os.makedirs(DEFERRED_SPECS_DIR, exist_ok=True)
    os.makedirs(PROBE_SPECS_DIR, exist_ok=True)
    probe = build_probe_spec()
    _dump(os.path.join(PROBE_SPECS_DIR, f"{probe['id']}.json"), probe)
    print(f"wrote {os.path.join(PROBE_SPECS_DIR, probe['id'] + '.json')}")
    cells = strengthen_cells()
    all_specs = build_all_specs()
    for cell, spec in zip(cells, all_specs):
        target_dir = DEFERRED_SPECS_DIR if is_deferred(cell) else SPECS_DIR
        path = os.path.join(target_dir, f"{spec['id']}.json")
        _dump(path, spec)
        print(f"wrote {path}")
    readme_path = os.path.join(DEFERRED_SPECS_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(_DEFERRED_README)
    print(f"wrote {readme_path}")


def total_gpu_h() -> dict:
    staged, deferred = staged_cells(), deferred_cells()
    assert len(staged) == 27 and len(deferred) == 3
    def _sum(cells, fn):
        return sum(fn(c["capacity"], c["steps"]) for c in cells)
    return {
        "n_staged": len(staged), "n_deferred": len(deferred),
        "staged_fast": _sum(staged, estimate_gpu_h),
        "staged_realized": _sum(staged, estimate_gpu_h_realized),
        "staged_naive": _sum(staged, estimate_gpu_h_naive),
        "deferred_fast": _sum(deferred, estimate_gpu_h),
        "deferred_realized": _sum(deferred, estimate_gpu_h_realized),
        "all30_fast": _sum(staged + deferred, estimate_gpu_h),
        "all30_realized": _sum(staged + deferred, estimate_gpu_h_realized),
    }


if __name__ == "__main__":
    write_all_specs()
    s = total_gpu_h()
    print(f"\n{s['n_staged']} staged main specs written to {SPECS_DIR}")
    print(f"{s['n_deferred']} deferred specs (+README) written to {DEFERRED_SPECS_DIR}")
    print("1 probe spec written to", PROBE_SPECS_DIR)
    print(f"\nStaged (27) ledger: fast-cluster={s['staged_fast']:.3f} GPU-h, "
         f"realized={s['staged_realized']:.3f} GPU-h "
         f"(pre-audit naive={s['staged_naive']:.3f} GPU-h)")
    print(f"Deferred (3) ledger: fast-cluster={s['deferred_fast']:.3f} GPU-h, "
         f"realized={s['deferred_realized']:.3f} GPU-h")
    print(f"All 30 (staged+deferred): fast-cluster={s['all30_fast']:.3f} GPU-h, "
         f"realized={s['all30_realized']:.3f} GPU-h")
    if s["staged_fast"] > 50.0 or s["staged_realized"] > 50.0:
        print(f"\n*** staged total >50 GPU-h -- CEREMONY ESCALATES to the full multi-round "
             "adversarial gauntlet per CLAUDE.md. STOP AND REPORT. ***")
