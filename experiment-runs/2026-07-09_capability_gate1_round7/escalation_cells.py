"""ADJUDICATION ROUND 7 — escalation-evidence training cells (coordinator-
authorized extension, PI saturation directive STATE.md@a9eda38).

Usage: python3 escalation_cells.py <GROUP> <STEPS>
Cells: S4@20000, S5@20000 (pre-registered first 2-2.5x escalation, gate 1(a)
rule (c) standard path); A5@40000, A6@40000 (trajectory-extension evidence
for the HARD-STOP lift-or-keep adjudication).

Reuses gate1_diag.py's train_cell / diagnose_checkpoint byte-identically
(same seed=0, same config) — only the step budget differs. Writes
ckpt_{name}_{steps}.pt + escalation_{name}_{steps}.json under
results/gate1_diagnosis/.
"""
import json
import os
import sys

import torch

DIAG_DIR = "/home/nvidia/chapter2/capability_separation/results/gate1_diagnosis"
sys.path.insert(0, DIAG_DIR)
import gate1_diag as gd  # noqa: E402  (pulls in production imports itself)

name, steps = sys.argv[1], int(sys.argv[2])
assert name in ("S3", "S4", "A5", "S5", "A6")

print(f"=== ESCALATION CELL {name}@{steps} (seed 0, gate1_diag train path) ===", flush=True)
model = gd.train_cell(name, seed=0, steps=steps)
torch.save(model.state_dict(), os.path.join(gd.OUT_DIR, f"ckpt_{name}_{steps}.pt"))
out = gd.diagnose_checkpoint(model, name, f"{steps} steps")
out["gpu_seconds_training"] = gd.GPU_SECONDS
with open(os.path.join(gd.OUT_DIR, f"escalation_{name}_{steps}.json"), "w") as f:
    json.dump(out, f, indent=2)

min15 = min(out["per_L"][L]["mean_cos"] for L in range(1, 6))
bar = "CLEARS" if min15 >= 0.9 else "MISSES"
print(f"=== DONE {name}@{steps}: min L1-5 mean_cos = {min15:.4f} -> {bar} the 0.9 bar "
      f"(gpu_s={gd.GPU_SECONDS:.1f}) ===", flush=True)
