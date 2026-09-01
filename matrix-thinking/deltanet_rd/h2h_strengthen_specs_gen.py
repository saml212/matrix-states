#!/usr/bin/env python
"""h2h_strengthen_specs_gen.py -- generates the 30 box queue-spec JSONs for
the sec 1.46 baseline-strengthening sweep (h2h_strengthen_rd.py) and holds
the GPU-h cost model both the design record and h2h_strengthen_rd.py's own
selftest (item 7) cite.

COST MODEL (stated, not assumed): the only DIRECTLY MEASURED rate in this
design lineage is C0 (d_model=256, n_layers=2, n_heads=4, ffn_mult=4) at
20,000 steps -- `0.2524 GPU-h/cell` (HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.6,
18,175.744s/20 cells) and re-metric `~0.0067 GPU-h/cell`
(`experiment-runs/2026-07-11_h2h_fix5_lrgrid/MANIFEST.md`, "~24s/cell").
C1/C2 were never run, so their rate is an ESTIMATE, extrapolated by the
PARAMETER-COUNT ratio to C0:

  GPU-h_train(cap, steps) ~= 0.2524 * (params(cap)/params(C0)) * (steps/20000)
  GPU-h_remetric(cap)     ~= 0.0067 * (params(cap)/params(C0))     [remetric is
                                                                     forward-only
                                                                     eval passes
                                                                     over a FIXED
                                                                     episode set --
                                                                     independent of
                                                                     how many steps
                                                                     the checkpoint
                                                                     trained for]

Method justification: batch (GRAMMAR_BATCH=32), context length (T, from
K=32's own T_bind=K*clause_len), and queries/episode (N_QUERY_TRAIN=8) are
IDENTICAL across every capacity in this grid -- only d_model/n_layers/
n_heads/ffn_mult vary. Under a fixed batch/sequence-length regime,
per-token training compute for a transformer LM is well-approximated as
proportional to total parameter count (Kaplan et al., "Scaling Laws for
Neural Language Models," 2020 -- the ~6N FLOPs/token heuristic, forward
2N + backward 4N). `n_heads` does not appear in
`transformer_baseline_rd.count_transformer_params`'s closed form at all,
and the one FLOPs term that DOES depend on head count -- attention scores,
n_heads * T^2 * head_dim -- is n_heads-INVARIANT once head_dim=d_model/
n_heads is substituted back in (= T^2 * d_model), so it is already
captured by the d_model-driven params ratio; doubling n_heads at C2
(4->8) changes zero terms in this estimate by construction, not by
omission. This is a DESIGN-TIME ESTIMATE, not a measurement -- exactly
the same status Rev 3 of the design doc gave its own axis-2 ~5s/pass
figure ("a design-time ASSUMPTION, to be REPLACED by a real measured
value" once a real cell runs); the FIRST strengthen cell to complete on
box should be treated as that pilot and can revise this ledger, per
CLAUDE.md's own "a calibration run before a big sweep is mandatory" rule.

Run standalone to (re)write the 30 spec JSONs:
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

C0_TRAIN_RATE_20K = 0.2524      # measured, HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.6
RUN_METRIC_RATE_20K = 0.0067    # measured, experiment-runs/2026-07-11_h2h_fix5_lrgrid/MANIFEST.md

SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strengthen_specs")
SPEC_ID_START = 600   # sorts AFTER the running 1.31B K=16 grace wave (0478-0485)

BOX_CWD = "/home/nvidia/chapter2/deltanet_rd"
BOX_PY = "/home/nvidia/tdenv/bin/python3"
BOX_CKPT_DIR = "/data/h2h_strengthen_ckpts"
BOX_OUT_DIR = f"{BOX_CWD}/results/h2h_rung1/strengthen"
BOX_REMETRIC_DIR = f"{BOX_OUT_DIR}/remetric"
BOX_GATES_DIR = f"{BOX_CWD}/results/h2h_rung1/gates"
BOX_MARGINS_TOKEN = f"{BOX_CWD}/results/h2h_rung1/MARGINS_FROZEN.token"
DIAL_ROUND = 4


def _params(cap_id: str) -> int:
    from transformer_baseline_rd import count_transformer_params
    kw = CAPACITIES[cap_id]
    return count_transformer_params(VOCAB_SIZE_TOTAL, kw["d_model"], kw["n_layers"], kw["ffn_mult"])


def param_ratio(cap_id: str) -> float:
    return _params(cap_id) / _params("C0")


def estimate_gpu_h(cap_id: str, steps: int) -> float:
    ratio = param_ratio(cap_id)
    train = C0_TRAIN_RATE_20K * ratio * (steps / FULL_STEPS_C0)
    remetric = RUN_METRIC_RATE_20K * ratio
    return train + remetric


def checkpoint_mb(cap_id: str) -> float:
    """fp32 model-only checkpoint size (no optimizer state saved -- verified against the C0
    reference: 14.44M params -> 57.8MB matches the fix5 MANIFEST's own measured checkpoint
    size exactly)."""
    return _params(cap_id) * 4 / 1e6


def _spec_for_cell(cell: dict, spec_id: int) -> dict:
    name = cell["name"]
    out_json = f"{BOX_OUT_DIR}/{name}.json"
    remetric_json = f"{BOX_REMETRIC_DIR}/{name}_round4.json"
    train_cmd = (f"{BOX_PY} h2h_strengthen_rd.py --run-cell {name} "
                f"--out {out_json} --ckpt-dir {BOX_CKPT_DIR} "
                f"--gates-dir {BOX_GATES_DIR} --margins-token {BOX_MARGINS_TOKEN} --device cuda")
    remetric_cmd = (f"{BOX_PY} h2h_strengthen_rd.py --remetric --run-cell {name} "
                    f"--ckpt-dir {BOX_CKPT_DIR} --remetric-dir {BOX_REMETRIC_DIR} "
                    f"--dial-round {DIAL_ROUND} --gates-dir {BOX_GATES_DIR} "
                    f"--margins-token {BOX_MARGINS_TOKEN} --device cuda")
    cmd = (f"export HEADTOHEAD_PI_SIGNOFF=1 HEADTOHEAD_MATCH_GATE_SIGNOFF=1 "
          f"H2H_DIAL_ROUND={DIAL_ROUND} && cd {BOX_CWD} && {train_cmd} && {remetric_cmd}")
    # NOTE: the arch_kw comparison uses repr() (single-quoted Python dict literal), never
    # json.dumps() (double-quoted) -- the whole one-liner is wrapped in DOUBLE quotes for the
    # outer `bash -c "$vcheck"` (queue_worker.sh's own invocation, see its script for the exact
    # form); a double-quoted dict literal embedded inside that outer double-quoted argument
    # would prematurely close the shell's quoting and truncate the command. Every python string
    # literal below is therefore deliberately single-quoted, with zero embedded double quotes.
    arch_kw_repr = repr(cell["arch_kw"])
    validity_check = (
        f"{BOX_PY} -c \""
        "import json, math; "
        f"d = json.load(open('{remetric_json}')); "
        "assert 'leg_a' in d and math.isfinite(d['leg_a']['acc_A']); "
        f"assert d.get('arch_kw') == {arch_kw_repr}, ('arch_kw mismatch: ' + str(d.get('arch_kw'))); "
        f"assert d.get('steps_target') == {cell['steps']}; "
        f"raw = json.load(open('{out_json}')); "
        f"assert raw.get('step_count') == {cell['steps']}, ('step_count mismatch: ' + str(raw.get('step_count')))"
        "\""
    )
    gpu_h = round(estimate_gpu_h(cell["capacity"], cell["steps"]), 4)
    return {
        "id": f"{spec_id:04d}_h2h_strengthen_{cell['capacity']}_lr{_lr_str(cell['lr'])}_st{cell['steps']}_s{cell['seed_idx']}",
        "lane": "strengthen",
        "hypothesis": (f"H2H transformer baseline (Task 1 recall) at capacity {cell['capacity']} "
                      f"({CAPACITIES[cell['capacity']]}), lr={cell['lr']:g}, "
                      f"{cell['steps']} steps, seed_idx={cell['seed_idx']}: does capacity + "
                      "training-length + the fix5 LR grid together clear the frozen 0.09375 "
                      "demonstration bar (sec 1.46)?"),
        "cmd": cmd,
        "gpu_h_estimate": gpu_h,
        "output_dir": BOX_OUT_DIR,
        "validity_check": validity_check,
        "notes": (f"params-ratio-extrapolated estimate from the measured C0/20k rate "
                 f"(0.2524 GPU-h/cell train, 0.0067 GPU-h/cell remetric; param_ratio="
                 f"{param_ratio(cell['capacity']):.4f}); checkpoint ~{checkpoint_mb(cell['capacity']):.1f} "
                 f"MB fp32 model-only at /data/h2h_strengthen_ckpts, NOT the SSD archive path "
                 "(disk-pressure discipline, sec 1.46)."),
    }


def build_all_specs() -> list[dict]:
    specs = []
    for i, cell in enumerate(strengthen_cells()):
        specs.append(_spec_for_cell(cell, SPEC_ID_START + i))
    assert len(specs) == 30
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids)), "spec id collision"
    return specs


def write_all_specs() -> None:
    os.makedirs(SPECS_DIR, exist_ok=True)
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
    by_cap = {}
    for cell, spec in zip(strengthen_cells(), specs):
        by_cap.setdefault(cell["capacity"], 0.0)
        by_cap[cell["capacity"]] += spec["gpu_h_estimate"]
    return {"total": total, "by_capacity": by_cap, "n_specs": len(specs)}


if __name__ == "__main__":
    write_all_specs()
    summary = total_gpu_h()
    print(f"\n{summary['n_specs']} specs written to {SPECS_DIR}")
    print(f"Total GPU-h estimate: {summary['total']:.3f}")
    for cap, gh in sorted(summary["by_capacity"].items()):
        print(f"  {cap}: {gh:.3f} GPU-h")
