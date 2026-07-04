"""geo3_recheck.py -- 2026-07-04 GPU re-measurement of the K=32 launch-read
simulator discrepancy flagged by KEY_ANCHORING_ATTACK_R2.md (Priority Target 3).

Question: recorded gate artifact (GEO3_DRIFT_DIAGNOSTIC.json, produced on
GPU by geo3_drift_diagnostic.py) shows K=32 predicted rec@0.9 h=4 = 0.7734375
(mean mapping); attacker's independent CPU rerun at "the recorded inputs"
(K=32, c=0.9037153...) gets ~0.0664. Before assuming a GPU-vs-CPU platform
artifact (e.g. TF32), we checked the actual archived JSON's own
`launch_read.inputs` field: drift_mean = 0.9416046142578125. That is K=16's
own measured drift, not K=32's (0.9037153124809265) -- because
geo3_drift_diagnostic.py's main() calls
`g3sim.launch_read(drift_mean=lr16["mean"], ...)` using ONLY the K=16
diagnostic result for BOTH K's predictions in launch_read's internal loop
(`for label, c in (("mean", drift_mean), ("p10", drift_p10)): out[label] =
{K: simulate_recovery(K, gram_resid, c, ...) for K in (16, 32)}`).

So the two candidate explanations to disambiguate here are:
  (A) INPUT MISMATCH: 0.7734 (recorded) used c=0.9416046 (K16's drift);
      0.0664 (attacker's CPU) used c=0.9037153 (K32's OWN drift) -- two
      different inputs to the same deterministic function, no platform
      story needed at all.
  (B) PLATFORM ARTIFACT: same c on GPU vs CPU gives different answers
      (TF32 or similar).

This script tests both directly, on GPU (device='cuda', CUDA_VISIBLE_DEVICES
pinned to one idle GPU by the caller) AND on CPU (device='cpu', same
process), for all three combinations, 3 seeds each:
  1. K=32, c=DRIFT_K32_MEAN (0.9037153124809265)   -- "K32's own drift"
  2. K=32, c=DRIFT_K16_MEAN (0.9416046142578125)   -- "as literally recorded"
  3. K=16, c=DRIFT_K16_MEAN (0.9416046142578125)   -- control, expect ~1.0
gram_resid=0.01 (matches launch_read's fixed default) for every cell.

Run on youthful-indigo-turkey, GPU 6 only (CUDA_VISIBLE_DEVICES=6), venv
/home/nvidia/tdenv/bin/python, against the repo copy at
/home/nvidia/chapter2/deltanet_rd/geo3_simulator.py. Invocation:
  DRY_RUN_BYPASS=1 CUDA_VISIBLE_DEVICES=6 /home/nvidia/tdenv/bin/python geo3_recheck.py
"""
import sys, os, json, time

sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")
import torch
from geo3_simulator import simulate_recovery

DRIFT_K16_MEAN = 0.9416046142578125
DRIFT_K32_MEAN = 0.9037153124809265
GRAM_RESID = 0.01

CELLS = [
    ("K32_own_drift_0.9037", 32, DRIFT_K32_MEAN),
    ("K32_as_recorded_using_K16_drift_0.9416", 32, DRIFT_K16_MEAN),
    ("K16_control_own_drift_0.9416", 16, DRIFT_K16_MEAN),
]

def main():
    out = {"gram_resid": GRAM_RESID, "cuda_available": torch.cuda.is_available(),
           "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
           "results": {}}
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        for label, K, c in CELLS:
            key = f"{device}__{label}"
            out["results"][key] = []
            for seed in (0, 1, 2):
                t0 = time.time()
                r = simulate_recovery(K, GRAM_RESID, c, seed=seed, device=device)
                dt = time.time() - t0
                row = {"seed": seed, "K": K, "align_cos": c, "device": device,
                       "h4_rec": r["rec"][4], "h3_rec": r["rec"][3],
                       "h1_rec": r["rec"][1], "h2_rec": r["rec"][2],
                       "actual_gram_resid": r["actual_gram_resid"],
                       "mean_cos_h4": r["mean_cos"][4], "wall_s": dt}
                out["results"][key].append(row)
                print(f"{key} seed={seed}: h4_rec={row['h4_rec']:.4f} "
                      f"mean_cos_h4={row['mean_cos_h4']:.4f} actual_gram_resid={row['actual_gram_resid']:.6f} "
                      f"({dt:.2f}s)", flush=True)
    with open("/home/nvidia/geo3_recheck_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote /home/nvidia/geo3_recheck_results.json")

if __name__ == "__main__":
    main()
