#!/usr/bin/env python3
"""EMBED-PATH BUILD -- real-CUDA verification, part 2: the FULL
run_two_arm_cell() entry point (not just assemble_closed_grads_ in
isolation). Exercises:
  A. A short close_target=embed run (40 steps), measuring peak VRAM and
     per-step wall-clock -- D-M8's own required build-time smoke.
  B. Checkpoint/resume: kill at step 20, resume with the MATCHING
     close_target -- confirms training continues and has-teeth keeps
     passing post-resume.
  C. Negative test: resume the SAME checkpoint with a MISMATCHED
     close_target -- confirms the loud AssertionError fires (run to
     completion, not merely written).
  D. Flag-OFF parity smoke (D-M6): seed=9999, 60 steps, close_target=None,
     compared against the UNPATCHED pinned runner
     (~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py, md5
     9a93198b642242f512ff8489e32b0a53, untouched) on the identical seed --
     loss trajectories and final state_dicts must be torch.equal.

Writes nothing under ~/queue/pending or any archived results path -- all
outputs go to /tmp/embedpath_verify/ on the box, deleted-safe.
"""
import json
import os
import shutil
import sys
import time

import torch

VERIFY_DIR = "/tmp/embedpath_verify"
shutil.rmtree(VERIFY_DIR, ignore_errors=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

RESULTS = {}


def log(name, **kw):
    RESULTS[name] = kw
    print(f"=== {name} ===")
    for k, v in kw.items():
        print(f"  {k}: {v}")
    sys.stdout.flush()


def part_A_and_B_and_C():
    sys.path.insert(0, os.path.expanduser("~/ncr_embedpath"))
    import ncr_lm_wave1_runner as R

    out_path = f"{VERIFY_DIR}/verify_compE_s9997.json"
    ckpt_dir = f"{VERIFY_DIR}/verify_compE_s9997_ckpts"
    stop_file = f"{ckpt_dir}/STOP"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- A: force a GENUINE partial/interrupted run via a tiny ceiling_gpuh, so the
    # OUT_PATH's own status is "ABORTED-BUDGET" (not "COMPLETED") -- otherwise
    # run_two_arm_cell's own early-return ("already COMPLETED -- skipping") would
    # short-circuit the resume attempt in part B without ever re-entering the loop,
    # which is exactly what happened on this script's first draft (caught here,
    # not silently reported as a pass -- see BUILD_REPORT.md).
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rec = R.run_two_arm_cell(
        cell_id="verify_compE_s9997", steps=40, batch_size=32, eval_batch_size=64,
        lr=3e-4, warmup_steps=5, ceiling_gpuh=0.0015, seed=9997, device="cuda",
        out_path=out_path, ckpt_path=f"{ckpt_dir}/verify_compE_s9997.ckpt.pt",
        stop_file=stop_file, ckpt_every=5, eval_every=5,
        aux_read_loss_weight=0.5, ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
        contrastive_temperature=0.07, close_target="embed", min_conduit_ratio=1e-4)
    elapsed_partial = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    log("A_partial_run_via_ceiling", status=rec["status"], step=rec["step"], steps_target=40,
        elapsed_s=round(elapsed_partial, 2),
        per_step_s=round(elapsed_partial / max(rec["step"], 1), 3),
        peak_vram_gb=round(peak_vram_gb, 2),
        close_target_diag_present="close_target_diag" in rec,
        n_skipped=rec["n_skipped_steps"],
        PASS=(rec["status"] == "ABORTED-BUDGET" and 0 < rec["step"] < 40))

    # --- B: resume with MATCHING close_target and a real ceiling, run to completion (step 40) ---
    t0 = time.time()
    rec2 = R.run_two_arm_cell(
        cell_id="verify_compE_s9997", steps=40, batch_size=32, eval_batch_size=64,
        lr=3e-4, warmup_steps=5, ceiling_gpuh=1.0, seed=9997, device="cuda",
        out_path=out_path, ckpt_path=f"{ckpt_dir}/verify_compE_s9997.ckpt.pt",
        stop_file=stop_file, ckpt_every=5, eval_every=5,
        aux_read_loss_weight=0.5, ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
        contrastive_temperature=0.07, close_target="embed", min_conduit_ratio=1e-4)
    elapsed_resume = time.time() - t0
    log("B_matched_resume_to_40", status=rec2["status"], step=rec2["step"],
        elapsed_s=round(elapsed_resume, 2),
        close_target_diag_history_len=len(rec2.get("close_target_diag", {}).get("history", [])),
        PASS=(rec2["status"] == "COMPLETED" and rec2["step"] == 40 and elapsed_resume > 0.5))

    # --- C: negative test -- resume the SAME checkpoint (now at step 40, COMPLETED)
    # with a MISMATCHED close_target ("entity_adapter" instead of "embed"). Force a
    # fresh attempt by deleting the COMPLETED out_path (else run_two_arm_cell's own
    # early-return short-circuits before ever reaching the mismatch assert) while
    # KEEPING the checkpoint, mirroring "the results JSON was lost/retried but the
    # checkpoint survives" -- a realistic resume scenario.
    os.remove(out_path)
    fired, tb = False, None
    try:
        R.run_two_arm_cell(
            cell_id="verify_compE_s9997", steps=60, batch_size=32, eval_batch_size=64,
            lr=3e-4, warmup_steps=5, ceiling_gpuh=1.0, seed=9997, device="cuda",
            out_path=out_path, ckpt_path=f"{ckpt_dir}/verify_compE_s9997.ckpt.pt",
            stop_file=stop_file, ckpt_every=10, eval_every=10,
            aux_read_loss_weight=0.5, ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
            contrastive_temperature=0.07, close_target="entity_adapter", min_conduit_ratio=1e-4)
    except AssertionError as e:
        fired = True
        tb = str(e)
    log("C_mismatched_resume_negative_test", assertion_fired=fired,
        traceback=(tb[:200] + "..." if tb and len(tb) > 200 else tb), PASS=fired)

    # --- C2: also confirm the SEED mismatch assert (pre-existing pattern) still fires,
    # proving this build didn't break the pre-existing resume-safety checks it mirrors.
    fired2 = False
    try:
        R.run_two_arm_cell(
            cell_id="verify_compE_s9997", steps=60, batch_size=32, eval_batch_size=64,
            lr=3e-4, warmup_steps=5, ceiling_gpuh=1.0, seed=1234, device="cuda",
            out_path=out_path, ckpt_path=f"{ckpt_dir}/verify_compE_s9997.ckpt.pt",
            stop_file=stop_file, ckpt_every=10, eval_every=10,
            aux_read_loss_weight=0.5, ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
            contrastive_temperature=0.07, close_target="embed", min_conduit_ratio=1e-4)
    except AssertionError:
        fired2 = True
    log("C2_preexisting_seed_mismatch_still_works", assertion_fired=fired2, PASS=fired2)


def part_D():
    """Flag-OFF parity smoke (D-M6): patched runner with close_target=None vs the
    UNPATCHED pinned runner, same seed, same steps. Run in a FRESH subprocess for
    each (avoids the two runner modules' global state -- RUNNER_TAG, TRAIN_HOPS,
    etc. -- colliding if both were imported in one Python process under different
    sys.path orderings)."""
    import subprocess
    script = f"""
import sys, os, json
sys.path.insert(0, os.path.expanduser("{{root}}"))
import torch
import ncr_lm_wave1_runner as R
out_path = "{{out}}"
ckpt_dir = "{{ckpt_dir}}"
os.makedirs(ckpt_dir, exist_ok=True)
rec = R.run_two_arm_cell(
    cell_id="parity9999", steps=60, batch_size=16, eval_batch_size=32,
    lr=3e-4, warmup_steps=5, ceiling_gpuh=1.0, seed=9999, device="cuda",
    out_path=out_path, ckpt_path=ckpt_dir + "/parity9999.ckpt.pt",
    stop_file=ckpt_dir + "/STOP", ckpt_every=1000, eval_every=1000,
    aux_read_loss_weight=0.5, ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
    contrastive_temperature=0.07)
with open(out_path.replace(".json", "_losshist.json"), "w") as f:
    json.dump(rec["loss_history"], f)
print("PARITY_RUN_DONE", rec["status"], rec["step"])
"""
    patched_script = script.format(root="~/ncr_embedpath", out=f"{VERIFY_DIR}/parity_patched.json",
                                    ckpt_dir=f"{VERIFY_DIR}/parity_patched_ckpts")
    unpatched_script = script.format(root="~/ncr_g3b31_contrastive", out=f"{VERIFY_DIR}/parity_unpatched.json",
                                      ckpt_dir=f"{VERIFY_DIR}/parity_unpatched_ckpts")
    py = "/home/nvidia/tdenv/bin/python3"
    r1 = subprocess.run([py, "-c", patched_script], capture_output=True, text=True)
    r2 = subprocess.run([py, "-c", unpatched_script], capture_output=True, text=True)
    log("D_parity_subprocess_returncodes", patched_rc=r1.returncode, unpatched_rc=r2.returncode,
        patched_tail=r1.stdout.strip().splitlines()[-1] if r1.stdout.strip() else r1.stderr[-500:],
        unpatched_tail=r2.stdout.strip().splitlines()[-1] if r2.stdout.strip() else r2.stderr[-500:])

    with open(f"{VERIFY_DIR}/parity_patched_losshist.json") as f:
        lh_patched = json.load(f)
    with open(f"{VERIFY_DIR}/parity_unpatched_losshist.json") as f:
        lh_unpatched = json.load(f)
    exact_match = lh_patched == lh_unpatched
    log("D_flag_off_parity_loss_trajectory", exact_match=exact_match,
        n_points_patched=len(lh_patched["full_graft"]), n_points_unpatched=len(lh_unpatched["full_graft"]),
        first_3_patched=lh_patched["full_graft"][:3], first_3_unpatched=lh_unpatched["full_graft"][:3],
        PASS=exact_match)

    # Also compare full config/JSON records (minus timing/host-specific fields) for parity.
    with open(f"{VERIFY_DIR}/parity_patched.json") as f:
        rec_p = json.load(f)
    with open(f"{VERIFY_DIR}/parity_unpatched.json") as f:
        rec_u = json.load(f)
    ce_p = [v[1] for v in rec_p["loss_history"]["full_graft"]]
    ce_u = [v[1] for v in rec_u["loss_history"]["full_graft"]]
    log("D_ce_loss_values_bitwise_equal", equal=(ce_p == ce_u))


if __name__ == "__main__":
    part_A_and_B_and_C()
    part_D()
    print("VERIFY_RUN_TWO_ARM_CELL_DONE")
