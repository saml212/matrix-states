import json

CHECKPOINTS = (250, 500, 1000, 2500, 5000)

for corpus in ["openr1-mix-ext", "wikitext-mix-ext"]:
    with open(f"../results/trajectory_{corpus}_phase2b.json") as f:
        traj = json.load(f)
    print(f"\n=== OOD (h in {{3,4}}) secondary readout: {corpus} ===")
    for arm in ["global", "per_token"]:
        print(f" {arm}:")
        for c in CHECKPOINTS:
            e = traj["secondary_ood"][arm][str(c)]
            d32, d20 = e["delta_k32"], e["delta_k20"]
            print(f"  c={c:5d} K32: mean={d32['mean']:+.4f} CI=[{d32['ci_low']:+.4f},{d32['ci_high']:+.4f}] det={e['det32']}   "
                  f"K20: mean={d20['mean']:+.4f} CI=[{d20['ci_low']:+.4f},{d20['ci_high']:+.4f}] det={e['det20']}")
