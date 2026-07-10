"""S2.33 route item (2) -- the S5 decisive-config seed extension, box side.

Licensed by S1.4.2's pre-registered escalation-to-5 trigger, imported into
Stage 2 by S2.5's flanking-seed pin ("3 (flanking-cell economization,
mirrors Stage 1's own force-rank-grid convention, S1.4.2)"): S5's decisive
(arm3_beta02, n_h=4) triad is AMBIGUOUS at the far-depth bar under the
S2.31a-decisional crosscheck lens (far64 X = {0.80, 0.00, 0.65}, mean
0.4833, ddof=1 sigma 0.4253; 95% CI straddles the 0.735 bar) -> "extend
THAT cell type to 5 seeds before drawing a conclusion".

Composition, not new mechanics (the S2.30 sweep-worker convention):
`run_cell_resume_safe(strict_real=True)` + `run_real_cell` (audited,
S2.20 F4 + S2.28 exclusions + S2.29 anchor-scaled breaker), cell dicts
shaped exactly as `build_nh_grid` emits them with seed in {3,4} threaded
into BOTH the `seed` field and the `cell_id` f-string (the S1.36a
seed-aliasing lesson -- fresh cell_ids, no resume-collision possible).

After training, the D=8 crosscheck ceiling is computed for the 2 new cells
by the EXACT S2.32 recompute convention (remetric_2p32_box.py::
ceiling_crosscheck_at_d8, replicated inline because that module hard-caps
CUDA at import time): `stage2_task.evaluate_composer_at_depth` at
D=D_TRAIN_MAX with seed=cell_seed*1000+8, CPU, primary field asserted
bit-identical against the just-written m_d0_profile D=8 row.

GPU: CUDA device 0 ONLY (set via CUDA_VISIBLE_DEVICES=0 by the supervisor;
GPUs 6/7 -- NCR Phase-0, task2-diagnosis -- are never visible to this
process). Ledger: 2 cells x ~105 s at the measured S5-nh4 sweep rate
~= 0.06 GPU-h.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import stage2_run as sr
import stage2_task as st

RESULTS_DIR = "stage2_results"
EXT_SEEDS = (3, 4)
OUT_PATH = "route_2p33_s5ext_output.json"


def build_ext_cells() -> list[dict]:
    """The (S5, arm3_beta02, n_h=4) row of `build_nh_grid`, extended to
    seeds 3/4 -- byte-identical field shape to the grid's own cells, seed
    threaded into BOTH `seed` and the `cell_id` f-string (S1.36a lesson)."""
    template = [c for c in sr.build_nh_grid()
                if c["group"] == "S5" and c["arm"] == "arm3_beta02" and c["n_h"] == 4]
    assert len(template) == 3 and {c["seed"] for c in template} == {0, 1, 2}
    cells = []
    for seed in EXT_SEEDS:
        cells.append(dict(cell_id=f"S5__arm3_beta02__nh{4}__seed{seed}", group="S5",
                          arm="arm3_beta02", n_h=4, seed=seed))
    for c in cells:
        assert c["cell_id"].endswith(f"__seed{c['seed']}"), c
        assert c["cell_id"] not in {t["cell_id"] for t in template}, f"seed-aliasing: {c['cell_id']}"
    return cells


def ceiling_crosscheck_at_d8(cell_json: dict, cell: dict) -> dict:
    """EXACT replica of remetric_2p32_box.py::ceiling_crosscheck_at_d8 (which
    cannot be imported here without its module-level CUDA kill): re-run
    `evaluate_composer_at_depth` at D=D_TRAIN_MAX on CPU with the
    m_d0_convergence_profile seed convention, keep the crosscheck fields,
    assert the primary field reproduces the just-written m_d0 D=8 row."""
    composer = sr.load_cell_composer(cell, RESULTS_DIR, device="cpu")
    seed_arg = cell["seed"] * 1000 + st.D_TRAIN_MAX
    s = st.evaluate_composer_at_depth(composer, cell["group"], st.D_TRAIN_MAX,
                                      seed=seed_arg, device="cpu")
    committed_row = next(r for r in cell_json["m_d0_profile"] if r["D"] == st.D_TRAIN_MAX)
    reproduced = s["recovered_frac_90"]
    committed = committed_row["recovered_frac_90"]
    bit_identical = (committed is not None) and abs(reproduced - committed) < 1e-9
    return dict(cell_id=cell["cell_id"],
                reproduced_primary_rf90_d8=reproduced,
                committed_primary_rf90_d8=committed,
                reproduction_bit_identical=bit_identical,
                crosscheck_rf90_d8=s["crosscheck_recovered_frac_90"],
                crosscheck_mean_cos_d8=s["crosscheck_mean_cos"])


def main():
    if os.environ.get(sr.PI_SIGNOFF_VAR) != "1":
        raise RuntimeError(f"{sr.PI_SIGNOFF_VAR}=1 required (S2.33 routed extension, citing "
                           f"S2.30's sweep authorization + S1.4.2's escalation-to-5 trigger)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[route_2p33] device={device}")

    results, ceilings, failures = {}, {}, []
    for cell in build_ext_cells():
        try:
            r = sr.run_cell_resume_safe(
                cell, RESULTS_DIR,
                lambda c: sr.run_real_cell(c, RESULTS_DIR, device=device, steps=None),
                strict_real=True)
            results[cell["cell_id"]] = dict(status=r.get("status"), final_loss=r.get("final_loss"),
                                            steps=r.get("steps_completed"), wall_s=r.get("wall_clock_s"),
                                            gate_route=r.get("gate_route"))
            print(f"[route_2p33] {cell['cell_id']}: {r.get('status')} loss={r.get('final_loss'):.6f} "
                  f"gate_route={r.get('gate_route')}")
            ceilings[cell["cell_id"]] = ceiling_crosscheck_at_d8(r, cell)
            print(f"[route_2p33]   D=8 ceiling: primary={ceilings[cell['cell_id']]['reproduced_primary_rf90_d8']} "
                  f"(bit_identical={ceilings[cell['cell_id']]['reproduction_bit_identical']}) "
                  f"xcheck={ceilings[cell['cell_id']]['crosscheck_rf90_d8']}")
        except Exception:
            import traceback
            failures.append(cell["cell_id"])
            traceback.print_exc()

    with open(OUT_PATH, "w") as f:
        json.dump(dict(train_summaries=results, ceiling_crosscheck_d8=ceilings,
                       failures=failures), f, indent=2, default=str)
    print(f"[route_2p33] wrote {OUT_PATH}")
    if failures:
        sys.exit(1)
    print("[route_2p33] ROUTE_2P33_S5EXT_DONE")


if __name__ == "__main__":
    main()
