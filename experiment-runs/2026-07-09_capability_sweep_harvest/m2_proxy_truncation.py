"""M2 proxy: post-hoc rank-k truncation curves (S1.5 M2) on the ARCHIVED
round-7 pinned-budget diagnosis checkpoints (results/gate1_diagnosis/ on box,
md5-verified copies in ./m2_proxy_ckpts/).

DISCLOSED DEVIATION from the pre-registration: the 58-cell sweep saved NO
checkpoints and never invoked truncation_curve.py, so M2 "knee in >=2/3
seeds" is NOT computable from the sweep as-run (build gap, recorded in
S1.33). This proxy is n=1 per group, on the round-7 gate-1 diagnosis
checkpoints trained at the same Rev-7 pinned budgets (S3=8K, S4=20K,
A5=20K, S5=8K, A6=40K), unconstrained arm. Corroborating-only, exactly as
M2's pre-registered role.

Run with the repo venv: .venv/bin/python m2_proxy_truncation.py
"""
import sys, json, os, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "matrix-thinking", "capability_separation"))
from groups import D_STATE, D_MIN
from group_task import generating_set
from group_word_encoder import GroupWordModel
from truncation_curve import truncation_curve

here = os.path.dirname(os.path.abspath(__file__))
CKPTS = {"S3": "ckpt_S3_8000.pt", "S4": "ckpt_S4_20000.pt", "A5": "ckpt_A5_20000.pt",
         "S5": "ckpt_S5_8000.pt", "A6": "ckpt_A6_40000.pt"}

out = {}
for g, fname in CKPTS.items():
    sd = torch.load(os.path.join(here, "m2_proxy_ckpts", fname), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model = GroupWordModel(D_STATE[g], len(generating_set(g)), L_max=16, h=32)
    model.load_state_dict(sd)
    model.eval()
    res = truncation_curve(model, g, base_seed=10_000, device="cpu")
    out[g] = {"knee": res["knee"], "ckpt": fname,
              "confirm_band": [D_MIN[g] - 1, D_MIN[g] + 1],
              "curve": {str(k): {kk: vv for kk, vv in res["curve"][k].items()
                                 if isinstance(vv, (int, float))} for k in res["curve"]}}
    print(f"{g} (d_min={D_MIN[g]}, d_state={D_STATE[g]}): knee k*={res['knee']} "
          f"[CONFIRM band {D_MIN[g]-1}..{D_MIN[g]+1}]")
    for k in sorted(res["curve"]):
        c = res["curve"][k]
        print(f"    k={k}: rec90={c['recovered_frac_90']:.4f}  cos={c['mean_cos']:.4f}")

json.dump(out, open(os.path.join(here, "m2_proxy_truncation_curves.json"), "w"), indent=2)
print("\nwritten m2_proxy_truncation_curves.json")
