"""S2.33 -- routed-items verdict harness (local, CPU, pure JSON+numpy).

Computes, from committed raws + the route_2p33 box outputs, with
`stage2_harvest.m_d3_verdict` UNMODIFIED (the S2.32 shadow-cell convention):

  ITEM 1 (A6 n_h-sufficiency): discharges S2.2.4's "calibration-pending for
  A6 (S2.5's force-arm grid)" decisive-config pin to n_h=4 -- the ONE
  constant `DECISIVE_CONTENDER_NH["A6"]` is overridden 2->4 for the
  discharged-pin verdicts, everything else untouched. The mechanism
  adjudication (n_h staircase at the identical pinned budget) is emitted as
  a table from the committed cell JSONs (0 GPU-h, inside the S1.30 rule-(4)
  <=0.1 GPU-h diagnostic cap).

  ITEM 2 (S5 seed extension): pools seeds {0,1,2} + {3,4} for the
  (S5, arm3_beta02, n_h=4) decisive leg, gated by S1.4.2's variance-ratio
  check (var_ratio > 4.0 -> flag, don't silently pool; both pooled and
  unpooled rows reported either way).

  ITEM 3 (the 3/62 A5 2(e) deferrals): the S1.30-style mechanism diagnostic
  from committed gate_reports -- per-failing-leg bar decomposition (which of
  the two co-decisional bars failed), full-grid loss/2(e) co-location, and
  the disclosure-only A5 sensitivity row (decisive aggregation is NEVER
  re-run without the flagged cell for verdict purposes -- the pre-registered
  aggregation is over ALL seeds; the sensitivity row exists to bound what
  the flag could possibly be hiding).

Verdicts emitted: (a) as-built pins / primary lens [S2.31 reproduction],
(b) as-built pins / crosscheck lens on the 62-cell grid [S2.32
reproduction], (c) DISCHARGED pins (A6 decisive n_h=4, S5 n=5) / crosscheck
lens -- THE S2.33 endpoint, decisional per S2.31a -- and (d) discharged
pins / primary lens (disclosure only, the broken-on-converged-cells lens).
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "matrix-thinking", "capability_separation"))

import stage2_harvest as h

SWEEP_RESULTS = os.path.join(HERE, "..", "sweep_results")
REMETRIC = os.path.join(HERE, "..", "remetric_2p32", "remetric_2p32_box_output.json")
ROUTE_BOX_OUT = os.path.join(HERE, "route_2p33_s5ext_output.json")
EXT_CELL_JSONS = [os.path.join(HERE, f"S5__arm3_beta02__nh4__seed{s}.json") for s in (3, 4)]
OUT = os.path.join(HERE, "route_2p33_verdict_output.json")

FAR, MID, D8 = h.FAR_DEPTH, h.MID_DEPTH, h.D_TRAIN_MAX
VAR_RATIO_MAX = 4.0  # S1.4.2 / REASONING_LINK_DESIGN S16.19.5 pin


def build_shadow_cell(cell: dict, ceiling_xcheck_d8: float | None) -> dict:
    """Byte-for-byte the S2.32 shadow construction (crosscheck_lens_verdict.py)."""
    shadow = copy.deepcopy(cell)
    for row in shadow["D_test_results"]:
        row["recovered_frac_90"] = row["crosscheck_recovered_frac_90"]
    if ceiling_xcheck_d8 is not None:
        for row in shadow["m_d0_profile"]:
            if row["D"] == D8 and not row.get("excluded"):
                row["recovered_frac_90"] = ceiling_xcheck_d8
    return shadow


def far64(cell):
    return next(r for r in cell["D_test_results"] if r["D"] == FAR)


def verdict_with_pins(cells: dict, a6_nh: int) -> dict:
    old = dict(h.DECISIVE_CONTENDER_NH)
    try:
        h.DECISIVE_CONTENDER_NH["A6"] = a6_nh
        return h.m_d3_verdict(h.build_cells_by_group_arm_nh(cells))
    finally:
        h.DECISIVE_CONTENDER_NH.update(old)


def main():
    # --- load the committed 62 + manifest (S2.31's own discipline) ---
    loaded62 = h.load_cell_results(SWEEP_RESULTS, cell_ids=h.expected_full_grid_cell_ids())
    manifest = h.verify_manifest(loaded62, h.expected_full_grid_cell_ids())
    assert manifest["clean"], manifest

    ceiling_map = json.load(open(REMETRIC))["ceiling_crosscheck_d8"]

    # --- load the 2 extension cells + their box-side ceilings ---
    route_out = json.load(open(ROUTE_BOX_OUT))
    ext_cells = {}
    for p in EXT_CELL_JSONS:
        with open(p) as f:
            c = json.load(f)
        problems = h.verify_config_match(c)
        assert not problems, (c["cell_id"], problems)
        assert c["cell_id"] not in loaded62, f"aliasing: {c['cell_id']}"
        ext_cells[c["cell_id"]] = c
    assert len(ext_cells) == 2
    ext_ceilings = route_out["ceiling_crosscheck_d8"]
    for cid, r in ext_ceilings.items():
        assert r["reproduction_bit_identical"], (cid, r)

    all64 = dict(loaded62) | ext_cells
    full_ceiling = dict(ceiling_map) | ext_ceilings

    # --- shadow (crosscheck-lens) cells ---
    shadow62 = {cid: build_shadow_cell(c, ceiling_map[cid].get("crosscheck_rf90_d8"))
                for cid, c in loaded62.items()}
    shadow64 = {cid: build_shadow_cell(c, full_ceiling[cid].get("crosscheck_rf90_d8"))
                for cid, c in all64.items()}

    # =====================================================================
    # ITEM 2 -- S1.4.2 trigger arithmetic + var-ratio pooling gate
    # =====================================================================
    s5_ids_old = [f"S5__arm3_beta02__nh4__seed{s}" for s in (0, 1, 2)]
    s5_ids_new = sorted(ext_cells)
    old_far = [far64(shadow64[c])["recovered_frac_90"] for c in s5_ids_old]
    new_far = [far64(shadow64[c])["recovered_frac_90"] for c in s5_ids_new]
    old_ceil = [full_ceiling[c]["crosscheck_rf90_d8"] for c in s5_ids_old]
    new_ceil = [full_ceiling[c]["crosscheck_rf90_d8"] for c in s5_ids_new]

    def _ambiguity(vals, bar):
        arr = np.asarray(vals, float)
        m, s = float(arr.mean()), float(arr.std(ddof=1))
        half = 1.96 * s / np.sqrt(len(arr))
        return dict(mean=m, sigma_ddof1=s, ci95=[m - half, m + half], bar=bar,
                    ci_straddles_bar=bool(m - half < bar < m + half))

    trigger = _ambiguity(old_far, 0.9 * float(np.mean(old_ceil)))
    v_old = float(np.asarray(old_far).var(ddof=1))
    v_new = float(np.asarray(new_far).var(ddof=1))
    if v_old == 0 and v_new == 0:
        var_ratio = 1.0
    elif min(v_old, v_new) == 0:
        var_ratio = float("inf")
    else:
        var_ratio = max(v_old, v_new) / min(v_old, v_new)
    var_flag = bool(var_ratio > VAR_RATIO_MAX)
    item2 = dict(
        trigger_arithmetic_pre_extension=trigger,
        old_cohort=dict(ids=s5_ids_old, far64_xcheck=old_far, ceil_xcheck=old_ceil),
        new_cohort=dict(ids=s5_ids_new, far64_xcheck=new_far, ceil_xcheck=new_ceil),
        var_ratio=var_ratio, var_ratio_max=VAR_RATIO_MAX, var_ratio_flag=var_flag,
        pooled_far64_mean=float(np.mean(old_far + new_far)),
        pooled_ceiling_mean=float(np.mean(old_ceil + new_ceil)),
        unpooled_far64_mean_old=float(np.mean(old_far)),
    )

    # =====================================================================
    # ITEM 1 -- the A6 n_h staircase (the mechanism table, 0 GPU-h)
    # =====================================================================
    stair = []
    for nh, seeds in ((1, range(3)), (2, range(5)), (4, range(3))):
        for s in seeds:
            cid = f"A6__arm3_beta02__nh{nh}__seed{s}"
            c = loaded62[cid]
            stair.append(dict(n_h=nh, seed=s, final_loss=c["final_loss"],
                              steps=c["steps_completed"],
                              far64_xcheck=far64(c)["crosscheck_recovered_frac_90"],
                              ceil_xcheck_d8=ceiling_map[cid]["crosscheck_rf90_d8"]))
    a6_nh4_far = [r["far64_xcheck"] for r in stair if r["n_h"] == 4]
    a6_nh4_ceil = [r["ceil_xcheck_d8"] for r in stair if r["n_h"] == 4]
    item1 = dict(
        staircase=stair,
        nh4_triad_far64_xcheck=a6_nh4_far,
        nh4_triad_ceiling_xcheck=a6_nh4_ceil,
        nh4_triad_ambiguity=_ambiguity(a6_nh4_far, 0.9 * float(np.mean(a6_nh4_ceil))),
    )

    # =====================================================================
    # ITEM 3 -- the 2(e) deferral mechanism diagnostic (0 GPU-h)
    # =====================================================================
    flagged = ["A5__arm2_beta01__nh2__seed3", "A5__arm2_beta01__nh2__seed4",
               "A5__arm3_beta02__nh2__seed3"]
    legs = []
    for cid in flagged:
        c = loaded62[cid]
        for d in c["gate_report"]["per_depth"]:
            if not d["depth_pass"]:
                legs.append(dict(cell_id=cid, D=d["D"], final_loss=c["final_loss"],
                                 T_over_Ta=d["T"] / d["T_anchor"], R_over_Ra=d["R"] / d["R_anchor"],
                                 bar_T_ok=d["bar_T_ok"], bar_R_ok=d["bar_R_ok"],
                                 anchor_floor_ok=d["anchor_floor_ok"], T_anchor=d["T_anchor"]))
    # full-grid co-location: every 2(e)-non-pass cell's loss vs the converged set
    coloc = [dict(cell_id=cid, final_loss=c["final_loss"], gate_route=c["gate_route"])
             for cid, c in sorted(loaded62.items()) if c["gate_route"] != "pass"]
    converged_fail = [r for r in coloc if r["final_loss"] < 0.01]
    # disclosure-only A5 sensitivity row (never decisional): drop the flagged arm3 cell
    a5_ids = [f"A5__arm3_beta02__nh2__seed{s}" for s in range(5)]
    a5_far_all = [far64(shadow62[c])["recovered_frac_90"] for c in a5_ids]
    a5_ceil_all = [ceiling_map[c]["crosscheck_rf90_d8"] for c in a5_ids]
    keep = [i for i, c in enumerate(a5_ids) if c != "A5__arm3_beta02__nh2__seed3"]
    a5_far_x = [a5_far_all[i] for i in keep]
    a5_ceil_x = [a5_ceil_all[i] for i in keep]
    item3 = dict(
        failing_legs=legs,
        all_non_pass_cells=coloc,
        n_converged_cells_failing_2e=len(converged_fail),
        a5_sensitivity_disclosure_only=dict(
            all5=dict(far_mean=float(np.mean(a5_far_all)), ceil_mean=float(np.mean(a5_ceil_all)),
                      far_bar=0.9 * float(np.mean(a5_ceil_all)),
                      far_clears=bool(np.mean(a5_far_all) >= 0.9 * np.mean(a5_ceil_all))),
            drop_flagged_seed3=dict(far_mean=float(np.mean(a5_far_x)), ceil_mean=float(np.mean(a5_ceil_x)),
                                    far_bar=0.9 * float(np.mean(a5_ceil_x)),
                                    far_clears=bool(np.mean(a5_far_x) >= 0.9 * np.mean(a5_ceil_x))),
        ),
    )

    # =====================================================================
    # VERDICTS
    # =====================================================================
    v_asbuilt_primary = verdict_with_pins(loaded62, a6_nh=2)     # S2.31 reproduction
    v_asbuilt_xcheck = verdict_with_pins(shadow62, a6_nh=2)      # S2.32 reproduction
    v_discharged_xcheck = verdict_with_pins(shadow64, a6_nh=4)   # THE S2.33 endpoint
    v_discharged_primary = verdict_with_pins(all64, a6_nh=4)     # disclosure only

    assert v_asbuilt_primary["verdict"] == "FALSIFY", v_asbuilt_primary["verdict"]
    assert v_asbuilt_xcheck["verdict"] == "FALSIFY", v_asbuilt_xcheck["verdict"]

    out = dict(manifest62=manifest, item1_a6=item1, item2_s5=item2, item3_a5_2e=item3,
               verdict_asbuilt_primary=v_asbuilt_primary["verdict"],
               verdict_asbuilt_xcheck=v_asbuilt_xcheck["verdict"],
               verdict_discharged_pins_xcheck=v_discharged_xcheck,
               verdict_discharged_pins_primary_disclosure=v_discharged_primary)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("=" * 100)
    print(f"reproductions: as-built primary={v_asbuilt_primary['verdict']} "
          f"as-built xcheck={v_asbuilt_xcheck['verdict']} (both must be FALSIFY)")
    print(f"S2.33 endpoint (discharged pins, xcheck decisional): {v_discharged_xcheck['verdict']}")
    print(f"  reason: {v_discharged_xcheck['reason'][:200]}")
    for g, r in v_discharged_xcheck["per_group"].items():
        print(f"  {g}: nh={r['decisive_n_h']} ceil={r['ceiling']:.3f} far={r['far_recovered_frac_90']:.3f} "
              f"bar={r['far_bar']:.3f} clears={r['far_clears']} arm2_far={r['arm2_far_recovered_frac_90']:.3f} "
              f"collapses={r['arm2_collapses']} separates={r['separates_from_arm2']}")
    print(f"item2: var_ratio={item2['var_ratio']:.3f} flag={item2['var_ratio_flag']} "
          f"pooled_far={item2['pooled_far64_mean']:.4f} (old-only {item2['unpooled_far64_mean_old']:.4f})")
    print(f"item3: converged cells failing 2(e): {item3['n_converged_cells_failing_2e']} (must be 0 for "
          f"the co-location claim); failing legs all T-bar-only: "
          f"{all(l['bar_R_ok'] and not l['bar_T_ok'] for l in item3['failing_legs'])}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
