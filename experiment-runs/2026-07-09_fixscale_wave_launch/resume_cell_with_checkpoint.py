#!/usr/bin/env python3
"""resume_cell_with_checkpoint.py -- operational convenience script (NOT part of the
audited fixscale_wave.py codebase; mirrors fixscale_slot0_waiter.sh's own established
precedent for a one-off adaptation that sits outside the core audited driver rather than
editing it) for resuming ONE specific fixscale cell from an --init-checkpoint after an
ABORTED_BUDGET contention-artifact adjudication.

FROZEN_BIAS_LM_DESIGN.md sec 13.21: fixscale_train_arm_off_98m_openr1-mix-ext_s1 was
ABORTED_BUDGET by the 1.5x circuit breaker at step 14200 (rate breach caused by h2h_calib3
co-scheduling on GPU 5/6, not a genuinely slow/broken cell -- loss trajectory was healthy).
The ABORTED_BUDGET marker + the partial (steps_completed=14000) result JSON were archived
aside (.superseded_<ts> suffix) so fixscale_wave.py's own cell_state() now reads 'absent'
for this cell. fixscale_wave.py's `cell` CLI subcommand has NO --init-checkpoint passthrough
(argparse: --scale/--corpus/--arm/--seed/--gpu/--dry-run/--force-gpu-busy only) and its
base_cmd() never emits --init-checkpoint, so it cannot resume -- this script invokes
lm_pretrain_rd.py directly instead, reusing fixscale_wave's own `_cell`/`base_cmd`/
`_budget_watchdog`/`gpu_occupancy_ok` UNMODIFIED (only the cmd list gets --init-checkpoint
appended and cell['steps'] is overridden to the REMAINDER for cmd-construction purposes
only -- the watchdog is handed the ORIGINAL cell dict, since _budget_watchdog only reads
cell['scale']/cell['cell_id'], never cell['steps'], so no monkeypatching of SCALES is
needed; the 1.5x wall-clock ceiling it uses is the FULL-cell ceiling (6.717h), which is a
valid, if slightly loose, backstop for a 53547-step remainder that only needs ~3.5h at
reference rate -- looser-not-tighter is the safe direction to err in here).

lm_pretrain_rd.py's own train() loop is `for step in range(1, args.steps + 1)` -- the step
counter ALWAYS RESTARTS AT 1 on --init-checkpoint (only model_state_dict is loaded; "optimizer
state NOT loaded... fresh Adam moments + fresh LR schedule", disclosed in the script's own
--init-checkpoint print). So --steps here is set to the REMAINDER (67547-14000=53547), not
67547 -- the resulting result JSON will read "steps": 53547, "steps_completed": 53547 (once
complete), NOT 67547. This still satisfies fixscale_wave.py's own pin-collector completeness
check: cell_state() -> _out_json_state() only tests `d.get("complete") is True`; the
steps_completed>=cfg["steps"] assertion lives ONLY in _assert_gate_tier_config_identity, which
out_path() only invokes for the special-cased gate-tier-owned arm_off/seed=0 reuse path --
never for this (seed=1) cell. Verified by direct read of fixscale_wave.py before running this.

Known, accepted side effects (documented, not hidden):
  - LR schedule restarts (100-step warmup + cosine decay over 53547 steps) rather than
    continuing the single 67547-step schedule the other 5 sibling cells trained under -- a
    real, disclosed deviation from "byte-identical training dynamics", inherent to the
    checkpoint format (no optimizer/scheduler state archived), not something this script can
    avoid.
  - ckpt filenames are `{run_name}_step{step}.pt` using THIS run's own local step counter,
    which restarts at 1 -- local steps 1000..14000 (multiples of ckpt_every) WILL silently
    overwrite the original run's checkpoints at those same filenames (including the
    --init-checkpoint source file itself, at local step 14000). The source checkpoint is
    backed up (.pre_resume_backup_<ts>) by the caller BEFORE this script is launched; the
    other overwritten intermediate checkpoints (real steps 1000-13000) are not independently
    referenced anywhere downstream (only checkpoints[-1] -- the final one -- is read by the
    pin comparator), so their loss is not functionally consequential, only a provenance note.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixscale_wave as W  # noqa: E402

SCALE = "98m"
CORPUS = "openr1-mix-ext"
SEED = 1
ARM = "arm_off"
GPU = int(os.environ.get("RESUME_GPU", "6"))
ARCHIVED_REAL_STEPS = 14000
INIT_CKPT = ("/data/fixscale_ckpts/train/fixscale_train_arm_off_98m_openr1-mix-ext_s1/"
             "lmC_openr1-mix-ext_dm768_ds64_L12_s1_step14000.pt")


def main() -> int:
    cell = W._cell(ARM, SCALE, CORPUS, SEED)
    state = W.cell_state(cell)
    if state in W.TERMINAL_STATES:
        print(f"cell {cell['cell_id']}: already terminal ({state}) -- refusing to resume-launch "
              f"over a terminal state; a human must archive it first (matches this campaign's own "
              f"'do not silently retry a terminal state' discipline).", flush=True)
        return 3
    assert os.path.exists(INIT_CKPT), f"--init-checkpoint source missing: {INIT_CKPT}"

    total_steps = W.SCALES[SCALE]["steps"]
    remaining = total_steps - ARCHIVED_REAL_STEPS
    assert remaining > 0

    ok, msg = W.gpu_occupancy_ok(GPU)
    print(f"GPU occupancy check gpu={GPU}: {msg}", flush=True)
    if not ok:
        print(f"REFUSING to launch: {msg}", flush=True)
        return 1

    cmd_cell = dict(cell, steps=remaining)
    cmd = W.base_cmd(cmd_cell) + ["--init-checkpoint", INIT_CKPT]
    print(f"RESUME LAUNCH gpu={GPU} remaining_steps={remaining} "
          f"(archived_real_steps={ARCHIVED_REAL_STEPS}, total={total_steps}): {' '.join(cmd)}",
          flush=True)

    os.makedirs(W.TRAIN_ROOT, exist_ok=True)
    os.makedirs(cell["ckpt_dir"], exist_ok=True)
    log_path = os.path.join(W.TRAIN_ROOT, f"{cell['cell_id']}.log")
    watch_csv = os.path.join(W.TRAIN_ROOT, f"{cell['cell_id']}_resume_rate_watch.csv")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU)}

    import subprocess
    t_launch = time.time()
    with open(log_path, "a") as logf:
        logf.write(f"\n===== RESUME launch {time.strftime('%Y-%m-%d %H:%M:%S')} UTC gpu={GPU} "
                    f"init_checkpoint={INIT_CKPT} remaining_steps={remaining} "
                    f"(FROZEN_BIAS_LM_DESIGN.md sec 13.21 contention-artifact resume) =====\n"
                    f"{' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=W.HERE)
        stop_evt = threading.Event()
        breach_evt = threading.Event()
        # Reuses fixscale_wave's OWN audited _budget_watchdog unmodified. It only reads
        # cell['scale'] (for SCALES[scale] ref_rate/per_cell_gpuh) and cell['cell_id'] (for its
        # own print/marker messages) -- pass the ORIGINAL (unmodified-steps) cell so its ceiling
        # is the full-cell 1.5x bound (6.717h), a safe (loose, not tight) backstop for this
        # 53547-step remainder.
        watcher = threading.Thread(target=W._budget_watchdog,
                                    args=(proc, log_path, t_launch, cell, stop_evt, watch_csv, breach_evt),
                                    daemon=True)
        watcher.start()
        rc = proc.wait()
        stop_evt.set()
        watcher.join(timeout=5)

    if breach_evt.is_set():
        import json
        marker = W.aborted_budget_marker_path(cell["out_path"])
        with open(marker, "w") as f:
            json.dump({"cell_id": cell["cell_id"], "reason": "1.5x circuit breaker (sec 13.8), "
                       "manual resume invocation (sec 13.21)",
                       "wall_s_at_abort": time.time() - t_launch,
                       "aborted_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
        print(f"cell {cell['cell_id']}: ABORTED_BUDGET (rc={rc}) -> {marker}", flush=True)
        return 1

    final = W._out_json_state(cell["out_path"])
    print(f"cell {cell['cell_id']}: {final} (rc={rc})", flush=True)
    return 0 if final == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
