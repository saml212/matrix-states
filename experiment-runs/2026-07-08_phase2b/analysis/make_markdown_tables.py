import json

CHECKPOINTS = (250, 500, 1000, 2500, 5000)

def classify(holds_by_c, det_arm_global_5000, det_arm_per_token_5000, agree_5000):
    NON_TERM = (250, 500, 1000, 2500)
    h = holds_by_c
    c1_raw = None
    for c in NON_TERM:
        if h[c]:
            c1_raw = c
            break
    if c1_raw is not None:
        idx = CHECKPOINTS.index(c1_raw)
        monotone_run = all(h[c] for c in CHECKPOINTS[idx:])
        if monotone_run:
            return {"outcome": "PERSISTENT", "c1": c1_raw}
    if h[5000] is False and any(h[c] for c in NON_TERM):
        return {"outcome": "TRANSIENT", "c1": None}
    if h[5000] is True and not any(h[c] for c in NON_TERM):
        return {"outcome": "LATE-EMERGENT", "c1": None}
    if not any(h[c] for c in CHECKPOINTS):
        if det_arm_global_5000 and det_arm_per_token_5000 and agree_5000:
            return {"outcome": "CONVERGED-EQUIVALENT", "c1": None}
        return {"outcome": "UNRESOLVED", "c1": None}
    return {"outcome": "NON-MONOTONE", "c1": None}

out = []
for corpus in ["openr1-mix-ext", "wikitext-mix-ext"]:
    with open(f"../results/trajectory_{corpus}_phase2b.json") as f:
        traj = json.load(f)
    out.append(f"\n**{corpus}** (registered pipeline corpus-level verdict: `{traj['classification']['outcome']}`, "
                f"global-arm-representative holds + both-arm terminal det_arm/agree)\n")
    for arm in ["global", "per_token"]:
        pa = traj["per_arm"][arm]
        holds_by_c = {int(k): v for k, v in pa["holds_by_c"].items()}
        det_arm_by_c = {int(k): v for k, v in pa["det_arm_by_c"].items()}
        g5 = {int(k): v for k, v in traj["per_arm"]["global"]["det_arm_by_c"].items()}[5000]
        p5 = {int(k): v for k, v in traj["per_arm"]["per_token"]["det_arm_by_c"].items()}[5000]
        a5 = traj["agree_by_c"]["5000"]
        indep = classify(holds_by_c, g5, p5, a5)
        out.append(f"\n*{corpus} × {arm}* — independent per-arm classification: **{indep['outcome']}**"
                    + (f" (c1={indep['c1']})" if indep['c1'] else "") + "\n")
        out.append("| c | K=32 mean Δ | K=32 CI | det32 | K=20 mean Δ | K=20 CI | det20 | holds(c) |")
        out.append("|---|---|---|---|---|---|---|---|")
        raw = {int(k): v for k, v in pa["raw"].items()}
        for c in CHECKPOINTS:
            d32 = raw[c]["delta_k32"]
            d20 = raw[c]["delta_k20"]
            out.append(f"| {c} | {d32['mean']:+.4f} | [{d32['ci_low']:+.4f}, {d32['ci_high']:+.4f}] | "
                        f"{raw[c]['det32']} | {d20['mean']:+.4f} | [{d20['ci_low']:+.4f}, {d20['ci_high']:+.4f}] | "
                        f"{raw[c]['det20']} | {holds_by_c[c]} |")

print("\n".join(out))

# OOD table
print("\n\n### OOD (h in {3,4}) secondary readout — det(K,c) table\n")
for corpus in ["openr1-mix-ext", "wikitext-mix-ext"]:
    with open(f"../results/trajectory_{corpus}_phase2b.json") as f:
        traj = json.load(f)
    for arm in ["global", "per_token"]:
        print(f"\n*{corpus} × {arm} (OOD, h∈{{3,4}})*\n")
        print("| c | K=32 mean Δ | K=32 CI | det32 | K=20 mean Δ | K=20 CI | det20 |")
        print("|---|---|---|---|---|---|---|")
        for c in CHECKPOINTS:
            e = traj["secondary_ood"][arm][str(c)]
            d32, d20 = e["delta_k32"], e["delta_k20"]
            print(f"| {c} | {d32['mean']:+.4f} | [{d32['ci_low']:+.4f}, {d32['ci_high']:+.4f}] | {e['det32']} | "
                  f"{d20['mean']:+.4f} | [{d20['ci_low']:+.4f}, {d20['ci_high']:+.4f}] | {e['det20']} |")
