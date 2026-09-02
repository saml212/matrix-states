#!/usr/bin/env python
"""h2h_strengthen_specs_gen.py -- generates the box queue-spec JSONs for
the sec 1.46 baseline-strengthening sweep (h2h_strengthen_rd.py) and holds
the GPU-h cost model both the design record and h2h_strengthen_rd.py's own
selftest (item 7) cite.

REV-narrow AUDIT (2026-09-01, 4 MAJOR, applied here): the original
(2026-09-01, pre-audit) version of this module used a NAIVE total-param-
count ratio and an anchor rate borrowed from a DIFFERENT program. Both are
corrected below; `estimate_gpu_h_naive` is RETAINED (not deleted) so the
design record can disclose both numbers per the audit's own instruction.

COST MODEL, CORRECTED (M2). Two independent bugs in the pre-audit model:

  1. WRONG ANCHOR: `C0_TRAIN_RATE_20K` was 0.2524 GPU-h/cell, cited to
     HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.6 -- but that figure is
     FROZEN_BIAS_LM_DESIGN.md's own 20-cell rate, a DIFFERENT program/task
     entirely. The transformer x task1_sweep cells' OWN measured rate
     (`experiment-runs/2026-07-11_h2h_fix5_lrgrid/MANIFEST.md`: "941-952
     s/cell") is the correct anchor -- midpoint ~945s = 0.2625 GPU-h/cell.

  2. WRONG SCALING LAW: total-param-count ratio is not the right FLOP
     proxy for THIS training loop, because the transformer's own tap route
     (`transformer_native_tap`) replicates the FULL block stack (attention
     + FFN, every layer) once per query -- `N_QUERY_TRAIN=8` extra passes,
     DEPLOY-PIN-1 -- but explicitly SKIPS the tied vocab-embedding/output
     matmul on every one of those replicated passes (`model(...,
     return_hidden=True)`, sec 1.31.4 item 3's OOM fix: "no LM-head matmul
     at all"). The vocab-projection matmul -- the term that DOMINATES raw
     param count (vocab=50,259 vs d_model<=512) -- therefore runs on 1x
     the tokens per step, while the attention+FFN "block" compute runs on
     (1 + N_QUERY_TRAIN) = 9x the tokens per step. A capacity that grows
     d_model (which grows BOTH the block AND the head) looks cheaper, in
     compute terms, than its raw param count implies, because a smaller
     share of its growth sees the 9x multiplier. `block_flop_ratio()`
     implements this split; it reproduces the audit's own headline
     figures exactly:
       - block-param ratio C2/C0 = 18,880,512 / 1,573,888 = 11.996 (~12x,
         "block params scale 12x")
       - head-param ratio  C2/C0 = 25,732,608 / 12,866,304 = 2.0 (exactly
         "head 2x")
       - combined (9x-weighted) ratio C2/C0 = 7.238 (~7.3x, the audit's
         own headline number, matched to 2 sig figs)

  GPU-h_train(cap, steps)    = C0_TRAIN_RATE_20K * block_flop_ratio(cap) * (steps/20000)
  GPU-h_remetric(cap)        = RUN_METRIC_RATE_20K * block_flop_ratio(cap)   [re-metric is
                                                                                forward-only over
                                                                                a FIXED episode
                                                                                set -- independent
                                                                                of training steps]

This is STILL a design-time ESTIMATE (block-FLOP-scaled, not measured) --
the sec 1.46 record states this total ALONGSIDE the pre-audit naive number
and the auditor's own independently-stated range, and gates the real
30-cell wave behind spec 0599 (`strengthen_specs_probe/`), a 300-step
probe on the single most expensive cell (C2, 60,000-step target) whose
MEASURED s/step re-prices this entire ledger before 0600-0629 are staged.

Run standalone to (re)write the 30 main spec JSONs + the 1 probe spec:
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

# --- AUDIT FIX (M2, 2026-09-01): anchor rate corrected to the transformer x task1_sweep cells'
# OWN measured rate (fix5 MANIFEST.md, "941-952 s/cell"), not the borrowed FROZEN_BIAS_LM_DESIGN
# sec-1.6 figure (908.79s, a DIFFERENT program/task). ---
C0_TRAIN_RATE_20K = 945.0 / 3600.0   # = 0.2625 GPU-h/cell (945s midpoint of 941-952s)
RUN_METRIC_RATE_20K = 0.0067         # measured, experiment-runs/2026-07-11_h2h_fix5_lrgrid/MANIFEST.md
                                      # ("~24s/cell") -- unaffected by the anchor fix (already the
                                      # correct task's own artifact, not borrowed cross-program)

# --- AUDIT FIX (M1, 2026-09-01): per-spec train-stage wall-clock timeout, `timeout -k 120 <N>h`.
# N = 2x the auditor's own corrected upper estimate per (capacity, steps) cell. On kill, no raw
# JSON is written (train_grammar_cell's own atomic-dump-at-the-end discipline), so
# is_valid_result/mode_run_cell's own skip logic sees "not valid" and the GPU frees for the next
# queue claim -- never a silent partial-write, never a wedged worker. ---
TIMEOUT_HOURS = {
    ("C0", 60_000): 3.0,
    ("C1", 20_000): 1.5,
    ("C1", 60_000): 4.0,
    ("C2", 20_000): 5.0,
    ("C2", 60_000): 12.0,
}
TIMEOUT_KILL_GRACE_S = 120   # `timeout -k 120` -- SIGKILL 120s after SIGTERM if still alive

SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strengthen_specs")
PROBE_SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strengthen_specs_probe")
SPEC_ID_START = 600   # sorts AFTER the running 1.31B K=16 grace wave (0478-0485)
PROBE_SPEC_ID = "0599_h2h_strengthen_probe_C2"   # sorts BEFORE 0600 -- the probe runs FIRST

BOX_CWD = "/home/nvidia/chapter2/deltanet_rd"
BOX_PY = "/home/nvidia/tdenv/bin/python3"
# AUDIT FIX (M2, 2026-09-01): checkpoints move to /ephemeral per STATE.md's own disk policy
# ("Training checkpoints go to /ephemeral/, NEVER to [root/`/data`]") -- the pre-audit record's
# own "/data at ~95% full, 139 GiB free" premise was ALSO wrong (corrected: /data has 914 GB
# free), but /ephemeral is the campaign's actual standing convention regardless of /data's real
# headroom, so this fix stands independent of the disk-space correction.
BOX_CKPT_DIR = "/ephemeral/h2h_strengthen_ckpts"
BOX_OUT_DIR = f"{BOX_CWD}/results/h2h_rung1/strengthen"
BOX_REMETRIC_DIR = f"{BOX_OUT_DIR}/remetric"
BOX_GATES_DIR = f"{BOX_CWD}/results/h2h_rung1/gates"
BOX_MARGINS_TOKEN = f"{BOX_CWD}/results/h2h_rung1/MARGINS_FROZEN.token"
DIAL_ROUND = 4

# --- Probe (M4, 2026-09-01): DISTINCT paths from the main sweep's own -- "any smoke must use
# distinct --out/--remetric-dir/--ckpt-dir" (audit minor note) -- so a 300-step probe run under
# the SAME cell identity as the real 0621 spec can never be mistaken for, or silently overwritten
# by/into, that real cell's own artifacts. ---
PROBE_OUT_DIR = f"{BOX_CWD}/results/h2h_rung1/strengthen_probe"
PROBE_REMETRIC_DIR = f"{PROBE_OUT_DIR}/remetric"
PROBE_CKPT_DIR = "/ephemeral/h2h_strengthen_probe_ckpts"
PROBE_CELL_NAME = "h2h_strengthen_C2_lr1e-03_st60000_s0"
PROBE_STEPS_OVERRIDE = 300
PROBE_TIMEOUT_MIN = 20   # generous vs the auditor's own "~2 min" expectation


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
    """PRE-AUDIT (naive) total-param-count ratio. Retained for disclosure only -- see
    `estimate_gpu_h_naive`."""
    return _params(cap_id) / _params("C0")


def block_flop_ratio(cap_id: str) -> float:
    """AUDIT-CORRECTED (M2) per-step FLOP proxy: C = (1+N_QUERY_TRAIN)*block_params + head_params
    (proportional units -- the constant backward-pass multiplier cancels in a ratio to C0)."""
    passes = 1 + N_QUERY_TRAIN
    c = passes * _block_params(cap_id) + _head_params(cap_id)
    c0 = passes * _block_params("C0") + _head_params("C0")
    return c / c0


def estimate_gpu_h_naive(cap_id: str, steps: int) -> float:
    """PRE-AUDIT total-param-ratio estimate. Superseded by `estimate_gpu_h` as the ledger figure
    -- kept only so sec 1.46 can disclose both numbers per the audit's own instruction."""
    ratio = param_ratio(cap_id)
    train = C0_TRAIN_RATE_20K * ratio * (steps / FULL_STEPS_C0)
    remetric = RUN_METRIC_RATE_20K * ratio
    return train + remetric


def estimate_gpu_h(cap_id: str, steps: int) -> float:
    """THE ledger figure (audit-corrected, M2): block-FLOP-scaled with the corrected anchor."""
    ratio = block_flop_ratio(cap_id)
    train = C0_TRAIN_RATE_20K * ratio * (steps / FULL_STEPS_C0)
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
    shell's quoting and truncate the command (caught live during this script's own build --
    simulated queue_worker.sh's exact invocation pattern against a real spec). `arch_kw` is
    rendered via repr() (single-quoted), never json.dumps() (double-quoted), for the same reason.

    AUDIT FIX (M3, 2026-09-01): adds (a) the checkpoint-md5-vs-provenance-record check (closes
    the 'double-dump race' the audit's minor note flags -- a remetric JSON whose recorded
    provenance md5 no longer matches the checkpoint CURRENTLY on disk at that path is treated as
    stale, not trusted) and (b) the raw JSON's own recorded `n_params` must match the capacity's
    formula-derived count (catches a capacity-override mixup at the SOURCE, before even checking
    acc_A)."""
    arch_kw_repr = repr(arch_kw)
    steps_target_check = f"assert d.get('steps_target') == {steps}; " if check_steps_target else ""
    return (
        "import json, math, hashlib; "
        f"d = json.load(open('{remetric_json}')); "
        "assert 'leg_a' in d and math.isfinite(d['leg_a']['acc_A']); "
        f"assert d.get('arch_kw') == {arch_kw_repr}, ('arch_kw mismatch: ' + str(d.get('arch_kw'))); "
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
    remetric_cmd = (f"{BOX_PY} h2h_strengthen_rd.py --remetric --run-cell {name} "
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
    gpu_h_naive = round(estimate_gpu_h_naive(cell["capacity"], cell["steps"]), 4)
    return {
        "id": f"{spec_id:04d}_h2h_strengthen_{cell['capacity']}_lr{_lr_str(cell['lr'])}_st{cell['steps']}_s{cell['seed_idx']}",
        "lane": "strengthen",
        "hypothesis": (f"H2H transformer baseline (Task 1 recall) at capacity {cell['capacity']} "
                      f"({CAPACITIES[cell['capacity']]}), lr={cell['lr']:g}, "
                      f"{cell['steps']} steps, seed_idx={cell['seed_idx']}: does capacity + "
                      "training-length + the fix5 LR grid together clear the frozen 0.09375 "
                      "demonstration bar (sec 1.46)? STAGED ONLY AFTER 0599's probe validates "
                      "(sec 1.46, M4)."),
        "cmd": cmd,
        "gpu_h_estimate": gpu_h,
        "output_dir": BOX_OUT_DIR,
        "validity_check": validity_check,
        "notes": (f"AUDIT-CORRECTED (2026-09-01) block-FLOP estimate (block_flop_ratio="
                 f"{block_flop_ratio(cell['capacity']):.4f}, anchor={C0_TRAIN_RATE_20K:.4f} "
                 f"GPU-h/20k-cell from fix5's own measured 941-952s/cell); PRE-AUDIT naive "
                 f"param-ratio estimate was {gpu_h_naive:.4f} GPU-h (param_ratio="
                 f"{param_ratio(cell['capacity']):.4f}) -- both disclosed in sec 1.46. Train "
                 f"stage wrapped `timeout -k {TIMEOUT_KILL_GRACE_S} {train_timeout_h:g}h` "
                 f"(no raw JSON on kill -> validity fails -> GPU frees). Checkpoint "
                 f"~{checkpoint_mb(cell['capacity']):.1f} MB fp32 model-only at "
                 f"{BOX_CKPT_DIR} (/ephemeral, per STATE.md's disk policy, NOT /data)."),
    }


def build_all_specs() -> list[dict]:
    specs = []
    for i, cell in enumerate(strengthen_cells()):
        specs.append(_spec_for_cell(cell, SPEC_ID_START + i))
    assert len(specs) == 30
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids)), "spec id collision"
    return specs


def build_probe_spec() -> dict:
    """M4 (2026-09-01): a 300-step probe of the single MOST EXPENSIVE cell (C2, 60,000-step
    target, the same cell identity 0621 will later train for real) -- gates the 30-cell wave.
    DISTINCT out/remetric/ckpt paths from the real cell's own (audit minor note); validity checks
    step_count==300 (never steps_target, which correctly stays 60,000 -- this cell's real
    identity/target is unchanged, only how many of those steps THIS probe run executed), a
    finite acc_A, and the exact C2 param count (44,613,632) -- catching a capacity-override
    mixup before any of the 30 real cells are staged."""
    n_params_c2 = _params("C2")
    out_json = f"{PROBE_OUT_DIR}/{PROBE_CELL_NAME}.json"
    remetric_json = f"{PROBE_REMETRIC_DIR}/{PROBE_CELL_NAME}_round4.json"
    train_cmd = (f"timeout -k 60 {PROBE_TIMEOUT_MIN}m {BOX_PY} h2h_strengthen_rd.py "
                f"--run-cell {PROBE_CELL_NAME} --steps-override {PROBE_STEPS_OVERRIDE} "
                f"--out {out_json} --ckpt-dir {PROBE_CKPT_DIR} "
                f"--gates-dir {BOX_GATES_DIR} --margins-token {BOX_MARGINS_TOKEN} --device cuda")
    remetric_cmd = (f"{BOX_PY} h2h_strengthen_rd.py --remetric --run-cell {PROBE_CELL_NAME} "
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
    probe_gpu_h = round(C0_TRAIN_RATE_20K * ratio * (PROBE_STEPS_OVERRIDE / FULL_STEPS_C0)
                        + RUN_METRIC_RATE_20K * ratio, 4)
    # the matching REAL spec's id, computed from strengthen_cells()'s own order -- NEVER a
    # hardcoded offset (a hand-counted "SPEC_ID_START + 20" was wrong by 5 during this script's
    # own build: the real match is index 15/id 0615, not index 20/id 0620 -- caught by
    # cross-checking the actually-written spec filename, not by re-counting harder by hand).
    real_spec_id = next(f"{SPEC_ID_START + i:04d}" for i, c in enumerate(strengthen_cells())
                        if c["name"] == PROBE_CELL_NAME)
    return {
        "id": PROBE_SPEC_ID,
        "lane": "strengthen_probe",
        "hypothesis": (f"CALIBRATION PROBE (sec 1.46, M4): {PROBE_STEPS_OVERRIDE} steps of the "
                      "single most expensive strengthening cell (C2, 44.61M params, the "
                      f"60,000-step target of the real spec {real_spec_id}) -- measures "
                      "real box s/step before any of the 30 main specs (0600-0629) are staged. "
                      "The 30-cell wave is gated on this probe validating."),
        "cmd": cmd,
        "gpu_h_estimate": probe_gpu_h,
        "output_dir": PROBE_OUT_DIR,
        "validity_check": validity_check,
        "notes": (f"Auditor's own estimate: ~2 min. This script's own block-FLOP formula gives "
                 f"~{probe_gpu_h * 3600:.0f}s ({probe_gpu_h:.4f} GPU-h) for train+remetric "
                 "combined -- same order of magnitude. DISTINCT paths from the real 0621 spec's "
                 f"own ({BOX_OUT_DIR} vs {PROBE_OUT_DIR}; {BOX_CKPT_DIR} vs {PROBE_CKPT_DIR}) "
                 "even though this probe trains the IDENTICAL cell identity, so the probe run "
                 "can never be mistaken for, or silently clobber, the real cell's own artifacts. "
                 "Its measured s/step MUST re-price the sec 1.46 ledger before 0600-0629 move "
                 "into the pending queue; if the re-priced 30-cell total exceeds 50 GPU-h, STOP "
                 "and report (the ceremony tier escalates to the full multi-round gauntlet)."),
    }


def write_all_specs() -> None:
    os.makedirs(SPECS_DIR, exist_ok=True)
    os.makedirs(PROBE_SPECS_DIR, exist_ok=True)
    probe = build_probe_spec()
    path = os.path.join(PROBE_SPECS_DIR, f"{probe['id']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(probe, f, indent=2)
    os.replace(tmp, path)
    print(f"wrote {path}")
    for spec in build_all_specs():
        path = os.path.join(SPECS_DIR, f"{spec['id']}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(spec, f, indent=2)
        os.replace(tmp, path)
        print(f"wrote {path}")


def total_gpu_h() -> dict:
    specs = build_all_specs()
    total = sum(s["gpu_h_estimate"] for s in specs)
    total_naive = sum(estimate_gpu_h_naive(c["capacity"], c["steps"]) for c in strengthen_cells())
    by_cap = {}
    for cell, spec in zip(strengthen_cells(), specs):
        by_cap.setdefault(cell["capacity"], 0.0)
        by_cap[cell["capacity"]] += spec["gpu_h_estimate"]
    return {"total": total, "total_naive": total_naive, "by_capacity": by_cap, "n_specs": len(specs)}


if __name__ == "__main__":
    write_all_specs()
    summary = total_gpu_h()
    print(f"\n{summary['n_specs']} main specs written to {SPECS_DIR}")
    print(f"1 probe spec written to {PROBE_SPECS_DIR}")
    print(f"Total GPU-h estimate (block-FLOP, audit-corrected): {summary['total']:.3f}")
    print(f"Total GPU-h estimate (naive param-ratio, pre-audit): {summary['total_naive']:.3f}")
    for cap, gh in sorted(summary["by_capacity"].items()):
        print(f"  {cap}: {gh:.3f} GPU-h")
    if summary["total"] > 50.0:
        print(f"\n*** >50 GPU-h ({summary['total']:.2f}) -- CEREMONY ESCALATES to the full "
             "multi-round adversarial gauntlet per CLAUDE.md. STOP AND REPORT. ***")
