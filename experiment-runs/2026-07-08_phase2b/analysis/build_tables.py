"""Independent harvest-time re-derivation of Phase-2b hexachotomy verdicts,
per (corpus, arm) -- reusing the box's exact det/holds/classify_trajectory
primitives (reimplemented verbatim from phase2_hexachotomy.py, verified
byte-identical against the box copy) applied to the ALREADY-COMPUTED raw CIs
in trajectory_{corpus}_phase2b.json (per_arm.<arm>.raw), rather than trusting
PHASE2B_SUMMARY.json's own corpus-level (global-arm-representative) verdict.
"""
import json, math

CHECKPOINTS = (250, 500, 1000, 2500, 5000)
NON_TERMINAL_CHECKPOINTS = (250, 500, 1000, 2500)
TERMINAL_CHECKPOINT = 5000
CI_T = 4.303

def det(ci_low, ci_high):
    return (ci_low > 0.0) or (ci_high < 0.0)

def holds(det32, det20, abs_d32, abs_d20):
    return bool(det32 and (not det20) and (abs_d32 > abs_d20))

def agree(a_lo, a_hi, b_lo, b_hi):
    return a_lo <= b_hi and b_lo <= a_hi

def classify(holds_by_c, det_arm_global_5000, det_arm_per_token_5000, agree_5000):
    h = holds_by_c
    c1_raw = None
    for c in NON_TERMINAL_CHECKPOINTS:
        if h[c]:
            c1_raw = c
            break
    if c1_raw is not None:
        idx = CHECKPOINTS.index(c1_raw)
        monotone_run = all(h[c] for c in CHECKPOINTS[idx:])
        if monotone_run:
            # stage05_pass_by_c always True in Phase-2b -> c1_final = c1_raw always
            return {"outcome": "PERSISTENT", "c1": c1_raw}
    if h[TERMINAL_CHECKPOINT] is False and any(h[c] for c in NON_TERMINAL_CHECKPOINTS):
        return {"outcome": "TRANSIENT", "c1": None}
    if h[TERMINAL_CHECKPOINT] is True and not any(h[c] for c in NON_TERMINAL_CHECKPOINTS):
        return {"outcome": "LATE-EMERGENT", "c1": None}
    if not any(h[c] for c in CHECKPOINTS):
        if det_arm_global_5000 and det_arm_per_token_5000 and agree_5000:
            return {"outcome": "CONVERGED-EQUIVALENT", "c1": None}
        return {"outcome": "UNRESOLVED", "c1": None}
    return {"outcome": "NON-MONOTONE", "c1": None}

results = {}
for corpus in ["openr1-mix-ext", "wikitext-mix-ext"]:
    with open(f"../results/trajectory_{corpus}_phase2b.json") as f:
        traj = json.load(f)
    results[corpus] = {"registered_pipeline_classification": traj["classification"], "per_arm": {}}
    for arm in ["global", "per_token"]:
        pa = traj["per_arm"][arm]
        holds_by_c = {int(k): v for k, v in pa["holds_by_c"].items()}
        det_arm_by_c = {int(k): v for k, v in pa["det_arm_by_c"].items()}
        results[corpus]["per_arm"][arm] = {
            "holds_by_c": holds_by_c,
            "det_arm_by_c": det_arm_by_c,
            "raw": {int(k): v for k, v in pa["raw"].items()},
        }

# terminal det_arm/agree shared quantities (as computed by the pipeline)
for corpus in results:
    g_det5000 = results[corpus]["per_arm"]["global"]["det_arm_by_c"][5000]
    p_det5000 = results[corpus]["per_arm"]["per_token"]["det_arm_by_c"][5000]
    with open(f"../results/trajectory_{corpus}_phase2b.json") as f:
        traj = json.load(f)
    agree5000 = traj["agree_by_c"]["5000"]
    for arm in ["global", "per_token"]:
        c = classify(results[corpus]["per_arm"][arm]["holds_by_c"], g_det5000, p_det5000, agree5000)
        results[corpus]["per_arm"][arm]["independent_classification"] = c

# ---- Print full report ----
print("=" * 100)
print("PER-(corpus,arm) INDEPENDENT RE-DERIVATION (4 primary contrasts)")
print("=" * 100)
for corpus in results:
    print(f"\n--- {corpus} ---")
    print(f"  REGISTERED PIPELINE verdict (global-arm-representative, both arms' terminal det_arm/agree): "
          f"{results[corpus]['registered_pipeline_classification']['outcome']}")
    for arm in ["global", "per_token"]:
        pa = results[corpus]["per_arm"][arm]
        print(f"  [{arm}] holds_by_c = {pa['holds_by_c']}")
        print(f"  [{arm}] det_arm_by_c = {pa['det_arm_by_c']}")
        print(f"  [{arm}] INDEPENDENT per-arm classification = {pa['independent_classification']}")

print("\n" + "=" * 100)
print("PER-CHECKPOINT DELTA TABLE (K=32), sign: positive = arm LOWER loss than off = arm HELPS")
print("=" * 100)
for corpus in results:
    for arm in ["global", "per_token"]:
        print(f"\n{corpus} x {arm} (K=32):")
        raw = results[corpus]["per_arm"][arm]["raw"]
        for c in CHECKPOINTS:
            d = raw[c]["delta_k32"]
            det32 = raw[c]["det32"]
            det20 = raw[c]["det20"]
            holds_c = results[corpus]["per_arm"][arm]["holds_by_c"][c]
            print(f"  c={c:5d}  mean_Delta={d['mean']:+.4f}  CI=[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
                  f"det32={det32}  det20={det20}  holds={holds_c}")

with open("full_report.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nwrote full_report.json")
