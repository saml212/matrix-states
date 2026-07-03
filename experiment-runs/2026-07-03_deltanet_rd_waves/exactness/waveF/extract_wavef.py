#!/usr/bin/env python3
"""Extract the 18-cell Wave F (soft-arm) table from result JSONs.

Per run: final-checkpoint key_gram_deviation_mean, value_gram_deviation_mean,
M2 (in-distribution) rec@0.9 at h=1,2,3, M3 (held-out) rec@0.9 at h=4,5,6,7,21,
and the premise/alignment gate fields (salvage-tier + alignment).
"""
import json
import glob
import os
import statistics as st

RESULTS_DIR = "/home/nvidia/chapter2/deltanet_rd/results/deltanet_rd_exactness/waveF"

ARMS = ["wF_geo1_l01", "wF_geo1_l10", "wF_geo2_zca"]
KS = [16, 32]
SEEDS = [0, 1, 2]

rows = []
for arm in ARMS:
    for K in KS:
        for seed in SEEDS:
            fn = os.path.join(RESULTS_DIR, f"{arm}_K{K}_s{seed}.json")
            if not os.path.exists(fn):
                print(f"MISSING: {fn}")
                continue
            d = json.load(open(fn))
            ck = d["checkpoints"][-1]
            assert ck["step"] == d["steps"], f"{fn}: final ckpt step {ck['step']} != steps {d['steps']}"
            m2 = ck["M2_in_distribution"]
            m3 = ck["M3_held_out"]

            def g(dct, h, field):
                return dct[str(h)][field]

            row = {
                "arm": arm,
                "K": K,
                "seed": seed,
                "complete": d["complete"],
                "steps": d["steps"],
                "exactness_config": d["exactness_config"],
                # gram deviation (mean over h=1..3 in M2 — near-constant across h at fixed ckpt)
                "key_gram_dev": st.mean(g(m2, h, "key_gram_deviation_mean") for h in (1, 2, 3)),
                "value_gram_dev": st.mean(g(m2, h, "value_gram_deviation_mean") for h in (1, 2, 3)),
                "rec_h1": g(m2, 1, "recovered_frac@0.9"),
                "rec_h2": g(m2, 2, "recovered_frac@0.9"),
                "rec_h3": g(m2, 3, "recovered_frac@0.9"),
                "rec_h4": g(m3, 4, "recovered_frac@0.9"),
                "rec_h5": g(m3, 5, "recovered_frac@0.9"),
                "rec_h6": g(m3, 6, "recovered_frac@0.9"),
                "rec_h7": g(m3, 7, "recovered_frac@0.9"),
                "rec_h21": g(m3, 21, "recovered_frac@0.9"),
                # premise / alignment gate — check across all h1-3 (M2) that they hold
                "premise_salvage_tier_all": all(
                    g(m2, h, "premise_valid_salvage_tier") for h in (1, 2, 3)
                ),
                "premise_value_salvage_tier_all": all(
                    g(m2, h, "premise_valid_value_salvage_tier") for h in (1, 2, 3)
                ),
                "alignment_valid_all": all(g(m2, h, "alignment_valid") for h in (1, 2, 3)),
                "alignment_cos_mean": st.mean(g(m2, h, "alignment_cos_mean") for h in (1, 2, 3)),
                "salvage_ratio_mean": st.mean(g(m2, h, "salvage_ratio_mean") for h in (1, 2, 3)),
                "value_salvage_ratio_mean": st.mean(
                    g(m2, h, "value_salvage_ratio_mean") for h in (1, 2, 3)
                ),
            }
            rows.append(row)

print(json.dumps(rows, indent=2))
print(f"\n\nTOTAL ROWS: {len(rows)}", flush=True)
