#!/usr/bin/env python3
"""EMBED-PATH BUILD -- clean steady-state throughput measurement (no eval/
checkpoint overhead mid-run) for BOTH close_target choices, to confirm/
correct the D-F2 budget re-derivation (R1.3: 9.84-13.20 GPU-h for 12 cells
at 20,000 steps each, batch_size=32). 300 steps, eval_every=1000 (never
fires mid-run), ckpt_every=1000 (one ckpt at the end only) -- isolates the
per-step training cost from logging/eval overhead."""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/ncr_embedpath"))
import torch
import ncr_lm_wave1_runner as R

DEVICE = "cuda"
VERIFY_DIR = "/tmp/embedpath_verify"
os.makedirs(VERIFY_DIR, exist_ok=True)

for close_target in ("embed", "entity_adapter"):
    out_path = f"{VERIFY_DIR}/throughput_{close_target}.json"
    ckpt_dir = f"{VERIFY_DIR}/throughput_{close_target}_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rec = R.run_two_arm_cell(
        cell_id=f"throughput_{close_target}", steps=300, batch_size=32, eval_batch_size=64,
        lr=3e-4, warmup_steps=200, ceiling_gpuh=1.0, seed=31337, device=DEVICE,
        out_path=out_path, ckpt_path=f"{ckpt_dir}/throughput_{close_target}.ckpt.pt",
        stop_file=f"{ckpt_dir}/STOP", ckpt_every=1000, eval_every=1000,
        aux_read_loss_weight=0.5, ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
        contrastive_temperature=0.07, close_target=close_target, min_conduit_ratio=1e-4)
    elapsed = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    per_step = elapsed / 300
    gpuh_20k = per_step * 20000 / 3600
    print(f"=== throughput[{close_target}] ===")
    print(f"  status={rec['status']} step={rec['step']} elapsed_s={elapsed:.2f} "
          f"per_step_s={per_step:.4f} peak_vram_gb={peak_vram_gb:.2f} "
          f"extrapolated_gpuh_at_20000steps={gpuh_20k:.4f}")

# --- flag-OFF (compB itself, unpatched behavior) for direct comparison ---
out_path = f"{VERIFY_DIR}/throughput_off.json"
ckpt_dir = f"{VERIFY_DIR}/throughput_off_ckpts"
os.makedirs(ckpt_dir, exist_ok=True)
if os.path.exists(out_path):
    os.remove(out_path)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
rec = R.run_two_arm_cell(
    cell_id="throughput_off", steps=300, batch_size=32, eval_batch_size=64,
    lr=3e-4, warmup_steps=200, ceiling_gpuh=1.0, seed=31337, device=DEVICE,
    out_path=out_path, ckpt_path=f"{ckpt_dir}/throughput_off.ckpt.pt",
    stop_file=f"{ckpt_dir}/STOP", ckpt_every=1000, eval_every=1000,
    aux_read_loss_weight=0.5, ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
    contrastive_temperature=0.07)
elapsed = time.time() - t0
per_step = elapsed / 300
gpuh_20k = per_step * 20000 / 3600
print(f"=== throughput[OFF/compB-equivalent] ===")
print(f"  status={rec['status']} step={rec['step']} elapsed_s={elapsed:.2f} "
      f"per_step_s={per_step:.4f} extrapolated_gpuh_at_20000steps={gpuh_20k:.4f}")
print("THROUGHPUT_DONE")
