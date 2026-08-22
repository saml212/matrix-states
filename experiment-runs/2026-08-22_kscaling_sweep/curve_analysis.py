import json, glob
D = "experiment-runs/2026-08-22_kscaling_sweep/"
CFG = {12:42, 16:40, 20:50, 24:36, 28:42, 32:48}
FIX = {12:40, 16:36, 20:44, 24:52, 28:32, 32:36}
cells = {}
for f in glob.glob(D+"sweep_*_kscaling.json") + glob.glob(D+"anchor_*_kscaling.json") + glob.glob("experiment-runs/2026-08-22_kscaling_wave0/k32_wave0_*_kscaling.json"):
    d = json.load(open(f))
    K = d["kscaling"]["K"]
    arm = "frozen" if d["freeze_entity_adapter"] else "trainable"
    seed = d["ckpt_seed"]
    cells[(K, arm, seed)] = d
print("total cells:", len(cells))
def acc(d, reg, h): return d["matched"][reg]["per_hop"]["h=%d" % h]["acc"]
def kap(a, K): ch = 1.0/K; return (a-ch)/(1-ch)

print("\n=== CURVE 1+3: kappa at h_top (frozen | trainable) and h_fix ===")
T = 0; nstrata = 0
for K in sorted(CFG):
    fro = [acc(cells[(K,"frozen",s)], "P1b", CFG[K]) for s in (0,1,2)]
    tra = [acc(cells[(K,"trainable",s)], "P1b", CFG[K]) for s in (0,1,2)]
    fx_f = [acc(cells[(K,"frozen",s)], "P1b", FIX[K]) for s in (0,1,2)]
    fx_t = [acc(cells[(K,"trainable",s)], "P1b", FIX[K]) for s in (0,1,2)]
    U = sum(1 if a>b else (0.5 if a==b else 0) for a in fro for b in tra)
    T += U; nstrata += 1
    print("K=%2d: frozen k=%s trainable k=%s U_K=%.1f | h_fix k: f=%s t=%s" % (
        K, ["%.4f"%kap(a,K) for a in fro], ["%.4f"%kap(a,K) for a in tra], U,
        ["%.3f"%kap(a,K) for a in fx_f], ["%.3f"%kap(a,K) for a in fx_t]))
print("\nStratified T = %.1f / %d  (6-strata threshold: T>=42 for p<0.01; lower tail <=12)" % (T, nstrata*9))

print("\n=== CURVE 2: P0 per-hop breaches per K ===")
for K in sorted(CFG):
    d0 = cells[(K,"frozen",0)]
    band_hi = d0["wall_band"][1]
    ch = 1.0/K
    hops = d0["hops"]
    over_lines = []
    for h in hops:
        vals = [acc(cells[(K,a,s)], "P0", h) for a in ("frozen","trainable") for s in (0,1,2)]
        n_over = sum(1 for v in vals if v > band_hi)
        if n_over:
            over_lines.append("   h=%3d (%s): %s OVER=%d/6" % (
                h, "train" if h <= 3 else "deep", ["%.3f"%v for v in vals], n_over))
    print("K=%2d chance=%.4f band_hi=%.4f %s" % (K, ch, band_hi, "CLEAN" if not over_lines else ""))
    for L in over_lines: print(L)
