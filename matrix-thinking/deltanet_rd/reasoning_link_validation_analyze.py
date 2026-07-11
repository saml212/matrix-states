"""reasoning_link_validation_analyze.py -- REASONING_LINK_DESIGN.md sec 17.6
analysis. Pure post-processing over `reasoning_link_validation.py`'s 80
per-cell JSONs (items 1/2: label-shuffle null + seed-bimodality resample)
and the sec 17.4 `04_remetric/` archive (item 3: per_token h>=2
concentration) -- no GPU, no checkpoints, runs anywhere.

Produces:
  - item 1 table: per-(cell,h) real/null/NULL-CLEARS verdict, aggregate counts
  - item 2 table: per-cell ON/OFF at original vs resampled seed, STABLE/FLIPPED
  - item 3 table: per (arm,corpus,surgery) h>=2 count/magnitude clearing the
    registered 0.10 floor
  - a plain-text AGGREGATE_SUMMARY-style report + a JSON dump of everything

Run standalone:
    python reasoning_link_validation_analyze.py \
        --validation-dir results/reasoning_link_validation \
        --remetric-dir results/reasoning_link_remetric \
        --out-json results/reasoning_link_validation/ANALYSIS.json \
        --out-txt results/reasoning_link_validation/ANALYSIS_SUMMARY.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reasoning_link_validation import null_clears, coherently_on, ABSOLUTE_FLOOR  # noqa: E402

H_LIST = (1, 2, 3, 4)


def load_validation_cells(validation_dir: str) -> dict:
    cells = {}
    for path in sorted(glob.glob(os.path.join(validation_dir, "*.json"))):
        base = os.path.basename(path)
        if base in ("ANALYSIS.json",):
            continue
        with open(path) as f:
            d = json.load(f)
        if "cell" not in d:
            continue
        cells[d["cell"]["cell_name"]] = d
    return cells


def load_remetric_cells(remetric_dir: str) -> dict:
    cells = {}
    for path in sorted(glob.glob(os.path.join(remetric_dir, "*.json"))):
        base = os.path.basename(path).replace(".json", "")
        with open(path) as f:
            d = json.load(f)
        cells[base] = d
    return cells


# ---------------------------------------------------------------------------
# Item 1 -- label-shuffle null.
# ---------------------------------------------------------------------------

def analyze_item1(validation_cells: dict) -> dict:
    rows = []
    for cell_name, d in validation_cells.items():
        per_h = d["per_h_original"]
        for h in H_LIST:
            rec = per_h[str(h)] if str(h) in per_h else per_h[h]
            real = rec["recovered_frac"]
            null = rec.get("null_recovered_frac_mean")
            null_ci_upper = rec.get("null_ci_upper")
            null_width = rec.get("null_width")
            if null is None:
                continue
            verdict = null_clears(real, null, null_ci_upper, null_width)
            rows.append({
                "cell_name": cell_name, "h": h, "real_recovered_frac": real,
                "null_recovered_frac": null, **verdict,
            })

    n_real_nonzero = sum(1 for r in rows if r["real_recovered_frac"] > 0)
    n_real_ge_floor = sum(1 for r in rows if r["real_recovered_frac"] >= ABSOLUTE_FLOOR)
    n_null_clears = sum(1 for r in rows if r["null_clears"])

    # Cell-level: NULL-CLEARS if ANY tested h individually clears.
    cell_verdict = {}
    for r in rows:
        cell_verdict.setdefault(r["cell_name"], False)
        if r["null_clears"]:
            cell_verdict[r["cell_name"]] = True
    n_cells_clear = sum(1 for v in cell_verdict.values() if v)

    # Restricted to the sec 17.4-identified 38 signal-bearing cells (>=1 nonzero h at real grading).
    signal_bearing_cells = {c for c, v in
                             {cn: any(rr["real_recovered_frac"] > 0 for rr in rows if rr["cell_name"] == cn)
                              for cn in validation_cells}.items() if v}
    n_signal_bearing = len(signal_bearing_cells)
    n_signal_bearing_clear = sum(1 for c in signal_bearing_cells if cell_verdict.get(c, False))
    frac_signal_bearing_clear = (n_signal_bearing_clear / n_signal_bearing) if n_signal_bearing else None

    real_vals = [r["real_recovered_frac"] for r in rows if r["real_recovered_frac"] >= ABSOLUTE_FLOOR]
    null_vals_at_those = [r["null_recovered_frac"] for r in rows if r["real_recovered_frac"] >= ABSOLUTE_FLOOR]

    return {
        "rows": rows, "n_readings": len(rows),
        "n_real_nonzero": n_real_nonzero, "n_real_ge_floor_0p10": n_real_ge_floor,
        "n_readings_null_clears": n_null_clears,
        "n_cells_total": len(validation_cells), "n_cells_null_clears": n_cells_clear,
        "n_signal_bearing_cells_sec17_4": n_signal_bearing,
        "n_signal_bearing_cells_null_clears": n_signal_bearing_clear,
        "frac_signal_bearing_cells_null_clears": frac_signal_bearing_clear,
        "real_recovered_frac_mean_at_ge_floor_readings": (sum(real_vals) / len(real_vals)) if real_vals else None,
        "null_recovered_frac_mean_at_those_same_readings": (sum(null_vals_at_those) / len(null_vals_at_those))
                                                             if null_vals_at_those else None,
        "outcome_per_sec17_6": (
            "SIGNAL-NOT-EXPLAINED-BY-NULL" if (frac_signal_bearing_clear is not None and frac_signal_bearing_clear >= 0.5)
            else "TRIVIAL-ARTIFACT" if (frac_signal_bearing_clear is not None)
            else "NO-SIGNAL-BEARING-CELLS-TO-TEST"
        ),
    }


# ---------------------------------------------------------------------------
# Item 2 -- seed-bimodality / episode-resampling stability.
# ---------------------------------------------------------------------------

def cell_on_status(per_h: dict) -> dict:
    on_hs = []
    for h in H_LIST:
        rec = per_h[str(h)] if str(h) in per_h else per_h[h]
        if coherently_on(rec["recovered_frac"]):
            on_hs.append(h)
    return {"on_hs": on_hs, "coherently_on": len(on_hs) > 0}


def analyze_item2(validation_cells: dict) -> dict:
    rows = []
    for cell_name, d in validation_cells.items():
        orig = cell_on_status(d["per_h_original"])
        resamp = cell_on_status(d["per_h_resample"])
        stable = orig["coherently_on"] == resamp["coherently_on"]
        rows.append({
            "cell_name": cell_name, "cell": d["cell"],
            "on_original": orig["coherently_on"], "on_hs_original": orig["on_hs"],
            "on_resample": resamp["coherently_on"], "on_hs_resample": resamp["on_hs"],
            "stable": stable,
        })

    # sec 17.4's 38-cell definition uses raw >0 nonzero, not the 0.10 floor -- report BOTH the
    # 0.10-floor "coherently ON" set (this function's own primary criterion, matching item 1's
    # ABSOLUTE_FLOOR reuse) and, separately, the raw->0.10 downgrade count for transparency.
    any_on_original = [r for r in rows if r["on_original"]]
    n_stable_among_on = sum(1 for r in any_on_original if r["stable"])
    n_flipped_among_on = sum(1 for r in any_on_original if not r["stable"])
    frac_flipped = (n_flipped_among_on / len(any_on_original)) if any_on_original else None

    return {
        "rows": rows, "n_cells_total": len(rows),
        "n_coherently_on_original_ge_floor": len(any_on_original),
        "n_stable_among_on": n_stable_among_on, "n_flipped_among_on": n_flipped_among_on,
        "frac_flipped_among_on": frac_flipped,
        "outcome_per_sec17_6": (
            "EPISODE-SPECIFICITY-DOMINATES" if (frac_flipped is not None and frac_flipped > 0.5)
            else "STATE-PROPERTY-DOMINATES" if (frac_flipped is not None)
            else "NO-ON-CELLS-TO-TEST"
        ),
    }


# ---------------------------------------------------------------------------
# Item 3 -- per_token h>=2 concentration, pure analysis of the sec 17.4 04_remetric archive.
# ---------------------------------------------------------------------------

import re as _re

# sec 17.4 archive's own naming convention: leg_a_<arm>_<corpus>_s<seed>_k<k>_<surgery>. `arm` can
# itself contain an underscore (per_token), so parse with an anchored regex over the three legal
# arm literals rather than a bare split (the remetric JSONs carry no structured cell dict -- run_cell
# writes ckpt_path/K/surgery but not arm/corpus as fields).
_LEG_A_NAME_RE = _re.compile(r"^leg_a_(off|per_token|global)_(.+)_s(\d+)_k(\d+)_(native|off)$")


def analyze_item3(remetric_cells: dict) -> dict:
    table = {}  # (arm, corpus, surgery) -> {"n_cells":.., "n_h_ge2_readings":.., "n_h_ge2_clearing":.., "max":..}
    for cell_name, d in remetric_cells.items():
        if not cell_name.startswith("leg_a_"):
            continue
        m = _LEG_A_NAME_RE.match(cell_name)
        if m is None:
            raise ValueError(f"analyze_item3: unparseable Leg-A archive filename {cell_name!r} -- "
                              f"the sec 17.4 naming convention drifted; refusing to silently skip.")
        arm, corpus, surgery = m.group(1), m.group(2), m.group(5)
        key = (arm, corpus, surgery)
        per_h = d.get("per_h", {})
        entry = table.setdefault(key, {"n_cells": 0, "n_h_ge2_readings": 0,
                                        "n_h_ge2_clearing_floor": 0, "max_h_ge2_recovered_frac": 0.0,
                                        "sum_h_ge2_recovered_frac": 0.0})
        entry["n_cells"] += 1
        for h in (2, 3, 4):
            rec = per_h.get(str(h), per_h.get(h))
            if rec is None:
                continue
            rf = rec["recovered_frac"]
            entry["n_h_ge2_readings"] += 1
            entry["sum_h_ge2_recovered_frac"] += rf
            entry["max_h_ge2_recovered_frac"] = max(entry["max_h_ge2_recovered_frac"], rf)
            if rf >= ABSOLUTE_FLOOR:
                entry["n_h_ge2_clearing_floor"] += 1

    rows = []
    for (arm, corpus, surgery), entry in sorted(table.items(), key=lambda kv: str(kv[0])):
        mean_rf = entry["sum_h_ge2_recovered_frac"] / entry["n_h_ge2_readings"] if entry["n_h_ge2_readings"] else 0.0
        rows.append({
            "arm": arm, "corpus": corpus, "surgery": surgery, "n_cells": entry["n_cells"],
            "n_h_ge2_readings": entry["n_h_ge2_readings"],
            "n_h_ge2_clearing_floor_0p10": entry["n_h_ge2_clearing_floor"],
            "frac_h_ge2_clearing_floor": (entry["n_h_ge2_clearing_floor"] / entry["n_h_ge2_readings"])
                                          if entry["n_h_ge2_readings"] else None,
            "mean_h_ge2_recovered_frac": mean_rf, "max_h_ge2_recovered_frac": entry["max_h_ge2_recovered_frac"],
        })

    by_arm = {}
    for r in rows:
        a = by_arm.setdefault(r["arm"], {"n_h_ge2_readings": 0, "n_h_ge2_clearing_floor_0p10": 0})
        a["n_h_ge2_readings"] += r["n_h_ge2_readings"]
        a["n_h_ge2_clearing_floor_0p10"] += r["n_h_ge2_clearing_floor_0p10"]
    for a, v in by_arm.items():
        v["frac_clearing"] = (v["n_h_ge2_clearing_floor_0p10"] / v["n_h_ge2_readings"]) if v["n_h_ge2_readings"] else None

    return {"rows": rows, "by_arm_summary": by_arm}


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------

def render_txt(item1: dict, item2: dict, item3: dict) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("REASONING-LINK sec 17.6 VALIDATION -- ANALYSIS SUMMARY")
    lines.append("=" * 100)
    lines.append("")
    lines.append("ITEM 1 -- LABEL-SHUFFLE NULL")
    lines.append(f"  {item1['n_readings']} (cell,h) readings; real>0: {item1['n_real_nonzero']}; "
                  f"real>=0.10 floor: {item1['n_real_ge_floor_0p10']}")
    lines.append(f"  readings that NULL-CLEAR: {item1['n_readings_null_clears']}")
    lines.append(f"  cells (of {item1['n_cells_total']}) with >=1 NULL-CLEARing h: {item1['n_cells_null_clears']}")
    lines.append(f"  sec 17.4 signal-bearing cells (38 expected): {item1['n_signal_bearing_cells_sec17_4']}, "
                  f"of which NULL-CLEAR: {item1['n_signal_bearing_cells_null_clears']} "
                  f"({item1['frac_signal_bearing_cells_null_clears']})")
    lines.append(f"  mean real recovered_frac (readings >=0.10 floor): "
                  f"{item1['real_recovered_frac_mean_at_ge_floor_readings']}")
    lines.append(f"  mean null recovered_frac (SAME readings): "
                  f"{item1['null_recovered_frac_mean_at_those_same_readings']}")
    lines.append(f"  OUTCOME: {item1['outcome_per_sec17_6']}")
    lines.append("")
    lines.append("ITEM 2 -- SEED-BIMODALITY / EPISODE-RESAMPLING STABILITY")
    lines.append(f"  cells coherently ON at original seed (>=0.10 floor, any h): "
                  f"{item2['n_coherently_on_original_ge_floor']} / {item2['n_cells_total']}")
    lines.append(f"  of those: STABLE under resample = {item2['n_stable_among_on']}, "
                  f"FLIPPED = {item2['n_flipped_among_on']} (frac flipped = {item2['frac_flipped_among_on']})")
    lines.append(f"  OUTCOME: {item2['outcome_per_sec17_6']}")
    lines.append("")
    lines.append("ITEM 3 -- PER_TOKEN h>=2 CONCENTRATION (by arm, corpus, surgery)")
    for r in item3["rows"]:
        lines.append(f"  arm={r['arm']:<10} corpus={r['corpus']:<20} surgery={r['surgery']:<7} "
                      f"n_cells={r['n_cells']} h>=2 clearing 0.10: {r['n_h_ge2_clearing_floor_0p10']}/"
                      f"{r['n_h_ge2_readings']} ({r['frac_h_ge2_clearing_floor']}) "
                      f"mean={r['mean_h_ge2_recovered_frac']:.4f} max={r['max_h_ge2_recovered_frac']:.4f}")
    lines.append("")
    lines.append("  by-arm summary (all corpora/surgeries pooled):")
    for arm, v in item3["by_arm_summary"].items():
        lines.append(f"    {arm}: {v['n_h_ge2_clearing_floor_0p10']}/{v['n_h_ge2_readings']} "
                      f"clearing (frac={v['frac_clearing']})")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-dir", default="results/reasoning_link_validation")
    ap.add_argument("--remetric-dir", default="results/reasoning_link_remetric")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-txt", default=None)
    args = ap.parse_args()

    validation_cells = load_validation_cells(args.validation_dir)
    remetric_cells = load_remetric_cells(args.remetric_dir)
    print(f"Loaded {len(validation_cells)} validation cells, {len(remetric_cells)} remetric cells.")

    item1 = analyze_item1(validation_cells)
    item2 = analyze_item2(validation_cells)
    item3 = analyze_item3(remetric_cells)

    txt = render_txt(item1, item2, item3)
    print(txt)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"item1": item1, "item2": item2, "item3": item3}, f, indent=2, default=str)
        print(f"wrote {args.out_json}")
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
        with open(args.out_txt, "w") as f:
            f.write(txt)
        print(f"wrote {args.out_txt}")


if __name__ == "__main__":
    main()
