import glob, json, os
BASE = "/Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-07-10_stage2_calibration"
cells = {}
for sub in ("sweep_results", "route_2p33"):
    for path in sorted(glob.glob(os.path.join(BASE, sub, "*.json"))):
        fn = os.path.basename(path)
        if "harvest_report" in fn or "verdict" in fn or "s5ext_output" in fn: continue
        d = json.load(open(path))
        cells[d.get("cell_id") or os.path.splitext(fn)[0]] = d

def row(c):
    dtr = {r["D"]: r for r in c["D_test_results"]}
    return (f"{c['cell_id']:38s} loss={c['final_loss']:.2e} "
            f"xck@9={dtr[9]['crosscheck_recovered_frac_90']:.2f} "
            f"xck@16={dtr[16]['crosscheck_recovered_frac_90']:.2f} "
            f"xck@32={dtr[32]['crosscheck_recovered_frac_90']:.2f} "
            f"xck@64={dtr[64]['crosscheck_recovered_frac_90']:.2f} "
            f"rank@9={dtr[9]['restricted_effective_rank']:.2f} "
            f"rank@64={dtr[64]['restricted_effective_rank']:.2f}")

for cfg in (("A6","arm3_beta02",1),("A6","arm3_beta02",2),("A6","arm3_beta02",4),
            ("S5","arm3_beta02",4),("A5","arm3_beta02",2),("A5","arm2_beta01",2),
            ("A6","arm2_beta01",2),("S5","arm2_beta01",2)):
    g,a,n = cfg
    print(f"\n== {g} {a} nh{n} ==")
    for cid in sorted(cells):
        c = cells[cid]
        if (c["group"],c["arm"],c["n_h"]) == cfg:
            print(row(c))
