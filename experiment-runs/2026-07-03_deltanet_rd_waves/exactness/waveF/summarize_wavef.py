import json
import statistics as st

with open("wavef_table.json") as f:
    text = f.read()
# array ends at the matching closing bracket before the trailing "TOTAL ROWS" line
arr_end = text.rindex("]") + 1
rows = json.loads(text[:arr_end])
print(f"parsed {len(rows)} rows")

ARMS = ["wF_geo1_l01", "wF_geo1_l10", "wF_geo2_zca"]
KS = [16, 32]
METRICS = [
    "key_gram_dev", "value_gram_dev",
    "rec_h1", "rec_h2", "rec_h3", "rec_h4", "rec_h5", "rec_h6", "rec_h7", "rec_h21",
]

def fmt(vals):
    lo, hi = min(vals), max(vals)
    mean = st.mean(vals)
    return f"{mean:.4f} [{lo:.4f}-{hi:.4f}]"

print("\n=== Per-cell rows ===")
for r in sorted(rows, key=lambda r: (r["arm"], r["K"], r["seed"])):
    print(r["arm"], "K", r["K"], "seed", r["seed"],
          "gram_dev", round(r["key_gram_dev"], 4),
          "vgram_dev", round(r["value_gram_dev"], 4),
          "h1", round(r["rec_h1"], 4), "h2", round(r["rec_h2"], 4), "h3", round(r["rec_h3"], 4),
          "h4", round(r["rec_h4"], 4), "h5", round(r["rec_h5"], 4), "h6", round(r["rec_h6"], 4),
          "h7", round(r["rec_h7"], 4), "h21", round(r["rec_h21"], 4),
          "premise", r["premise_salvage_tier_all"], r["premise_value_salvage_tier_all"],
          "align", r["alignment_valid_all"], round(r["alignment_cos_mean"], 4))

print("\n=== Per arm x K aggregates (mean [min-max] across 3 seeds) ===")
for arm in ARMS:
    for K in KS:
        cell = [r for r in rows if r["arm"] == arm and r["K"] == K]
        assert len(cell) == 3, (arm, K, len(cell))
        print(f"\n-- {arm} K={K} (n={len(cell)}) --")
        for m in METRICS:
            vals = [r[m] for r in cell]
            print(f"  {m:15s} {fmt(vals)}")
        premise = all(r["premise_salvage_tier_all"] for r in cell)
        premise_v = all(r["premise_value_salvage_tier_all"] for r in cell)
        align = all(r["alignment_valid_all"] for r in cell)
        print(f"  premise_salvage_tier(all 3 seeds)={premise} value={premise_v} alignment={align}")
        cfg = cell[0]["exactness_config"]
        print(f"  exactness_config={cfg}")
