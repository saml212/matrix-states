"""S2.32 -- crosscheck-lens M-D3 endpoint, computed by feeding SHADOW cell
dicts (primary field overwritten by the crosscheck field, D=8 ceiling
patched from the box recompute) into the UNMODIFIED
`stage2_harvest.m_d3_verdict` / `verify_manifest` / `build_cells_by_group_arm_nh`.
No new metric logic -- same pre-registered machinery, different field read,
per S2.31a's tiebreak instruction ("a crosscheck-lens re-metric of the full
committed grid ... using the existing degauge_and_score / harvest machinery,
no new metric").

Run with numpy+the stage2_harvest.py module available (CPU-only, no torch
needed for this stage -- pure JSON+numpy post-processing per that module's
own docstring).
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stage2_harvest as h

RESULTS_DIR = "stage2_results"
CEILING_RECOMPUTE_PATH = "remetric_2p32_box_output.json"
MID_DEPTH = h.MID_DEPTH
FAR_DEPTH = h.FAR_DEPTH
D_TRAIN_MAX = h.D_TRAIN_MAX


def build_shadow_cell(cell: dict, ceiling_xcheck_d8: float | None) -> dict:
    """Deep-copies `cell`, then (a) overwrites every D_test_results row's
    `recovered_frac_90` with its own `crosscheck_recovered_frac_90` (already
    present in the committed JSON, S2.31's own 6-config table), and (b)
    patches the m_d0_profile D=D_TRAIN_MAX row's `recovered_frac_90` with
    the box-recomputed crosscheck ceiling (absent from the committed JSON --
    the ONE field this re-metric required a checkpoint forward pass for)."""
    shadow = copy.deepcopy(cell)
    for row in shadow["D_test_results"]:
        row["recovered_frac_90"] = row["crosscheck_recovered_frac_90"]
    if ceiling_xcheck_d8 is not None:
        for row in shadow["m_d0_profile"]:
            if row["D"] == D_TRAIN_MAX and not row.get("excluded"):
                row["recovered_frac_90"] = ceiling_xcheck_d8
    return shadow


def main():
    loaded = h.load_cell_results(RESULTS_DIR, cell_ids=h.expected_full_grid_cell_ids())
    manifest = h.verify_manifest(loaded, h.expected_full_grid_cell_ids())
    assert manifest["clean"], f"manifest not clean: {manifest}"

    with open(CEILING_RECOMPUTE_PATH) as f:
        box_output = json.load(f)
    ceiling_map = box_output["ceiling_crosscheck_d8"]
    teeth = box_output["teeth_shuffled_control"]

    n_ceiling_ok = sum(1 for v in ceiling_map.values() if "error" not in v)
    n_bit_identical = sum(1 for v in ceiling_map.values() if v.get("reproduction_bit_identical"))
    assert n_ceiling_ok == len(loaded), f"ceiling recompute incomplete: {n_ceiling_ok}/{len(loaded)}"
    assert n_bit_identical == len(loaded), (
        f"PRIMARY reproduction not bit-identical on {len(loaded) - n_bit_identical} cells -- "
        f"the recompute harness does not faithfully reproduce the production pipeline, STOP"
    )

    # --- teeth verdict (S2.31a's pinned falsifier: shuffled crosscheck < 0.5 everywhere) ---
    teeth_ok = True
    teeth_rows = []
    for cid, t in teeth.items():
        shuf = t["shuffled_crosscheck_recovered_frac_90"]
        fires = shuf >= 0.5
        teeth_ok = teeth_ok and not fires
        teeth_rows.append(dict(cell_id=cid, real_xcheck_rf90=t["real_crosscheck_recovered_frac_90"],
                               committed_xcheck_rf90=t["committed_crosscheck_rf90_d64"],
                               real_bit_identical=t["reproduction_bit_identical_crosscheck"],
                               shuffled_xcheck_rf90=shuf, n_items=t["n_items"],
                               n_fixed_points=t["n_fixed_points_in_permutation"], falsifier_fires=fires))

    # --- shadow cells + crosscheck-lens M-D3 ---
    shadow_cells = {cid: build_shadow_cell(cell, ceiling_map[cid].get("crosscheck_rf90_d8"))
                    for cid, cell in loaded.items()}
    grouped_primary = h.build_cells_by_group_arm_nh(loaded)
    grouped_xcheck = h.build_cells_by_group_arm_nh(shadow_cells)

    verdict_primary = h.m_d3_verdict(grouped_primary)   # reproduces the committed S2.31 FALSIFY, as a check
    verdict_xcheck = h.m_d3_verdict(grouped_xcheck)      # THE S2.32 crosscheck-lens endpoint

    # --- full per-cell flat table ---
    flat = []
    for cid, cell in sorted(loaded.items()):
        dtr = {row["D"]: row for row in cell["D_test_results"]}
        mid, far = dtr[MID_DEPTH], dtr[FAR_DEPTH]
        m_d0_d8 = next(r for r in cell["m_d0_profile"] if r["D"] == D_TRAIN_MAX)
        flat.append(dict(
            cell_id=cid, group=cell["group"], arm=cell["arm"], n_h=cell["n_h"], seed=cell["seed"],
            final_loss=cell["final_loss"],
            primary_ceiling_d8=m_d0_d8["recovered_frac_90"],
            crosscheck_ceiling_d8=ceiling_map[cid].get("crosscheck_rf90_d8"),
            primary_rf90_mid32=mid["recovered_frac_90"], crosscheck_rf90_mid32=mid["crosscheck_recovered_frac_90"],
            primary_rf90_far64=far["recovered_frac_90"], crosscheck_rf90_far64=far["crosscheck_recovered_frac_90"],
            crosscheck_mean_cos_far64=far["crosscheck_mean_cos"],
        ))

    out = dict(
        manifest=manifest,
        teeth_control=dict(rows=teeth_rows, all_pass=teeth_ok,
                           rule="falsifier fires if any shuffled crosscheck rf90 >= 0.5 (S2.31a pinned)"),
        verdict_primary_reproduction=verdict_primary,
        verdict_crosscheck_lens=verdict_xcheck,
        flat_per_cell_table=flat,
        ceiling_recompute_summary=dict(n_cells=len(loaded), n_ok=n_ceiling_ok, n_bit_identical=n_bit_identical),
    )
    with open("crosscheck_lens_verdict_output.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("=" * 100)
    print(f"TEETH: all_pass={teeth_ok}")
    for r in teeth_rows:
        print(f"  {r['cell_id']:40s} real_xcheck={r['real_xcheck_rf90']:.4f} "
              f"shuffled_xcheck={r['shuffled_xcheck_rf90']:.4f} fires={r['falsifier_fires']} "
              f"(n_fixed_points={r['n_fixed_points']}/{r['n_items']})")
    print("=" * 100)
    print(f"PRIMARY-LENS reproduction verdict: {verdict_primary['verdict']}")
    print(f"CROSSCHECK-LENS S2.32 verdict: {verdict_xcheck['verdict']}")
    print(f"  reason: {verdict_xcheck['reason']}")
    for g, r in verdict_xcheck["per_group"].items():
        print(f"  {g}: {r}")
    print("=" * 100)
    print("wrote crosscheck_lens_verdict_output.json")


if __name__ == "__main__":
    main()
