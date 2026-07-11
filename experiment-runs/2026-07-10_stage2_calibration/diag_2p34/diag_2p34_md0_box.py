"""S2.34 DIAGNOSIS (routed by S2.33's closing per S2.6's INCONCLUSIVE clause)
-- box-side crosscheck-lens M-D0 re-read.

The committed `m_d0_profile` persisted ONLY the primary-lens fields
(`recovered_frac_90`/`mean_cos`); the primary lens is the documented broken
instrument on converged composer cells (S2.31a ground 3, reproduced 4x), so
the committed per-depth TRAIN-support profile is uninformative exactly where
the S2.33-routed diagnosis questions live. This script re-reads the
TRAIN-support depths D in {2..8} (each cell's own committed evaluable set --
rows with excluded=False; the S2.28 pinned exclusions are honored by
construction) under the S2.31a-DECISIONAL crosscheck lens, using the EXACT
S2.32 recompute convention (`remetric_2p32_box.py::ceiling_crosscheck_at_d8`,
generalized over D): `stage2_task.evaluate_composer_at_depth` UNMODIFIED,
seed = cell_seed*1000 + D (m_d0_convergence_profile's own convention), CPU
only, per-row bit-identical primary reproduction asserted against the
committed m_d0_profile row (harness-fidelity teeth, the S2.32/S2.33
convention).

Scope: the 26 cells the three routed questions name --
  (a) A6 arm3 n_h in {1,2,4}, all seeds (11 cells);
  (b) S5 arm3 n_h=4, seeds 0-4 incl. the S2.33 extension cells (5 cells);
  (c) A5 arm2 + arm3 n_h=2, seeds 0-4 (10 cells).

No training, eval-only forward passes, CUDA never visible ("~0.0 GPU-h
free eval-only forwards" precedent, S2.6/S2.7/S2.8-2(e), S2.32).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only; GPUs 6/7 (NCR) never visible

import torch

torch.cuda.is_available = lambda: False  # hard local override, S2.32 convention

import stage2_run as sr
import stage2_task as st

RESULTS_DIR = "stage2_results"
DEVICE = "cpu"
D_TRAIN_MAX = st.D_TRAIN_MAX  # 8

FOCUS_CELL_IDS = (
    [f"A6__arm3_beta02__nh1__seed{s}" for s in range(3)]
    + [f"A6__arm3_beta02__nh2__seed{s}" for s in range(5)]
    + [f"A6__arm3_beta02__nh4__seed{s}" for s in range(3)]
    + [f"S5__arm3_beta02__nh4__seed{s}" for s in range(5)]
    + [f"A5__arm2_beta01__nh2__seed{s}" for s in range(5)]
    + [f"A5__arm3_beta02__nh2__seed{s}" for s in range(5)]
)
assert len(FOCUS_CELL_IDS) == 26


def load_cell_json(cid: str) -> dict:
    with open(os.path.join(RESULTS_DIR, f"{cid}.json")) as f:
        return json.load(f)


def crosscheck_profile(cell: dict) -> list[dict]:
    """The crosscheck-lens M-D0 profile: evaluate_composer_at_depth at every
    committed-evaluable train depth, keeping the crosscheck fields the
    production m_d0_convergence_profile discarded; primary asserted
    bit-identical per row."""
    composer = sr.load_cell_composer(cell, RESULTS_DIR, device=DEVICE)
    rows = []
    for committed_row in cell["m_d0_profile"]:
        D = committed_row["D"]
        if committed_row.get("excluded"):
            rows.append(dict(D=D, excluded=True, gating=committed_row.get("gating")))
            continue
        seed_arg = cell["seed"] * 1000 + D  # m_d0_convergence_profile's own convention
        s = st.evaluate_composer_at_depth(composer, cell["group"], D, seed=seed_arg, device=DEVICE)
        reproduced = s["recovered_frac_90"]
        committed = committed_row["recovered_frac_90"]
        bit_identical = (committed is not None) and abs(reproduced - committed) < 1e-9
        rows.append(dict(
            D=D, excluded=False, gating=committed_row.get("gating"),
            committed_primary_rf90=committed,
            reproduced_primary_rf90=reproduced,
            reproduction_bit_identical=bit_identical,
            crosscheck_rf90=s["crosscheck_recovered_frac_90"],
            crosscheck_mean_cos=s["crosscheck_mean_cos"],
        ))
    return rows


def main():
    out, n_rows, n_bit = {}, 0, 0
    for i, cid in enumerate(FOCUS_CELL_IDS):
        cell = load_cell_json(cid)
        try:
            rows = crosscheck_profile(cell)
            out[cid] = dict(cell_id=cid, final_loss=cell["final_loss"],
                            steps_completed=cell["steps_completed"], profile=rows)
            for r in rows:
                if not r.get("excluded"):
                    n_rows += 1
                    n_bit += int(bool(r["reproduction_bit_identical"]))
        except Exception as e:  # noqa: BLE001 -- report, don't crash the loop
            out[cid] = dict(cell_id=cid, error=f"{type(e).__name__}: {e}")
        evald = [f"D{r['D']}={r['crosscheck_rf90']:.2f}" for r in out[cid].get("profile", [])
                 if not r.get("excluded")]
        print(f"[{i+1}/26] {cid}: " + (" ".join(evald) if evald else str(out[cid].get("error"))),
              flush=True)

    print(f"bit-identical primary reproduction: {n_bit}/{n_rows} evaluated rows")
    with open("diag_2p34_md0_xcheck_output.json", "w") as f:
        json.dump(dict(focus_cells=out,
                       teeth=dict(n_rows_evaluated=n_rows, n_bit_identical=n_bit)),
                  f, indent=2, default=str)
    print("wrote diag_2p34_md0_xcheck_output.json")
    print("DIAG_2P34_MD0_DONE")


if __name__ == "__main__":
    main()
