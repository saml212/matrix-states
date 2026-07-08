"""sec 15.25.4 harvest artifact: build the compact recomputed-admissibility
table (NOT a full copy of the 15 multi-MB wide-grid raw JSONs, which stay
solely at their existing archive location,
experiment-runs/2026-07-07_keyanchor_scaling_wide/) documenting, per cell:
the pre-flip admission legs (independently re-read from the archived raw
JSON this session), whether it was flipped, and its own h4 value (which
this unlock never alters).
"""
import json
import os

SRC_DIR = "experiment-runs/2026-07-07_keyanchor_scaling_wide/results/deltanet_rd_exactness/wavekeyanchor-scaling-wide"

UNLOCK_CELLS = {
    (72, 1740), (72, 1742),
    (78, 1840), (78, 1841), (78, 1842),
    (84, 1940), (84, 1941), (84, 1942),
    (90, 2040), (90, 2041), (90, 2042),
}
EXCLUDED_SAME_SIGNATURE = {(69, 1730)}


def main():
    rows = []
    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(SRC_DIR, fname)) as f:
            d = json.load(f)
        K, seed = d.get("K"), d.get("seed")
        admission = d.get("geo3_admission", {})
        h4 = d["checkpoints"][-1]["M3_held_out"]["4"]["recovered_frac@0.9"]
        pre = admission.get("admissible")
        if (K, seed) in UNLOCK_CELLS:
            post, action = True, "FLIPPED (sec 15.24.6 outcome-(c) unlock)"
        elif (K, seed) in EXCLUDED_SAME_SIGNATURE:
            post, action = pre, "NOT flipped (out of declared 11-cell scope, disclosed)"
        else:
            post, action = pre, "unchanged (already admissible pre-unlock)"
        rows.append({
            "K": K, "seed": seed, "file": fname,
            "admissible_pre_unlock": pre, "admissible_post_unlock": post,
            "action": action,
            "n_geo3_fallback_train_steps": admission.get("n_geo3_fallback_train_steps"),
            "checkpoint_fallback_seen": admission.get("checkpoint_fallback_seen"),
            "ns_converged_no_fallback": admission.get("ns_converged_no_fallback"),
            "value_salvage_tier_pass": admission.get("value_salvage_tier_pass"),
            "finite_loss_no_divergence": admission.get("finite_loss_no_divergence"),
            "task_performance_floor_pass": admission.get("task_performance_floor_pass"),
            "h4_M3_hop4_recovered_frac_at_0.9": h4,
        })
    out = {
        "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.25.4 (C17 TOLERANCE-MISCALIBRATION unlock)",
        "source_raws": SRC_DIR,
        "n_flipped": sum(1 for r in rows if "FLIPPED" in r["action"]),
        "cells": rows,
    }
    print(json.dumps(out, indent=2))
    assert out["n_flipped"] == 11


if __name__ == "__main__":
    main()
