"""Quick test: can we reconstruct entity_adapter init via build_arm() on CPU,
and is it (a) deterministic across two calls, (b) different from a final ckpt."""
import os, sys, time
sys.path.insert(0, "/home/nvidia/ncr_g3b31_contrastive")
os.chdir("/home/nvidia/ncr_g3b31_contrastive")

import torch
import ncr_lm_wave1_runner as R

VOCAB = 50259
SEED = 1
DEVICE = "cpu"

t0 = time.time()
arm1 = R.build_arm(VOCAB, SEED, DEVICE)
w1 = arm1["integ"].entity_adapter.weight.detach().clone()
print("build 1 took", time.time() - t0, "s, shape", tuple(w1.shape))

t0 = time.time()
arm2 = R.build_arm(VOCAB, SEED, DEVICE)
w2 = arm2["integ"].entity_adapter.weight.detach().clone()
print("build 2 took", time.time() - t0, "s")

print("bit-identical across two reconstructions:", torch.equal(w1, w2))

# different seed should differ
arm3 = R.build_arm(VOCAB, SEED + 1, DEVICE)
w3 = arm3["integ"].entity_adapter.weight.detach().clone()
print("differs for seed+1:", not torch.equal(w1, w3))

# compare against final checkpoint weight
ckpt_path = "/home/nvidia/ncr_g3b31_contrastive/results/mob_g3b31_compB_s1_ckpts/mob_g3b31_compB_s1.ckpt.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
print("ckpt seed field:", ckpt.get("seed"), "step:", ckpt.get("step"))
w_final = ckpt["full_graft"]["integ_state"]["entity_adapter.weight"]
print("final shape", tuple(w_final.shape))
print("init == final (should be False):", torch.equal(w1, w_final))
print("init frob norm:", w1.norm().item(), "final frob norm:", w_final.norm().item())
print("drift frob norm:", (w_final - w1).norm().item())
print("relative drift:", (w_final - w1).norm().item() / w1.norm().item())
