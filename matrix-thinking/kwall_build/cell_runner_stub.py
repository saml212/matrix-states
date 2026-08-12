#!/usr/bin/env python3
"""CPU-STUB emulating `ncr_earlyln_scale.py --cell ...`'s OBSERVABLE
CONTRACT (exit codes, JSON schema on disk, `--stop-file` handling) WITHOUT
any torch/CUDA training. Used ONLY by
`tests/test_orchestrator_integration.py` to drive the REAL
`orchestrator.py` dispatch loop through real `subprocess.run` calls on a
machine with no GPU — flagged per CLAUDE.md's CPU-stub hard rule
("CPU-stub where CUDA is unavailable, flagged as such").

Behavior is selected via the `KWALL_STUB_BEHAVIOR` env var, read once per
invocation (so a test can make attempt 1 crash and attempt 2 complete by
changing the env var between the two `dispatch_cell` calls the real
orchestrator issues internally):
  - "complete"      : writes a COMPLETED JSON, exits 0. (default)
  - "abort_budget"  : writes an ABORTED-BUDGET JSON, exits 0.
  - "crash"         : exits 1, writes no JSON at all.
  - "crash_after_dir": creates the outdir (so `os.makedirs` has already
                        run, matching the real script's own ordering) then
                        exits 1 with no JSON -- the "present, JSON absent"
                        recovery-table state.
  - "hang"          : creates the outdir, then sleeps (for SIGKILL-based
                        mid-attempt-crash recovery tests). The caller kills
                        this process externally.
Honors `--stop-file`: if the file exists at invocation time, exits 3
immediately (mirrors the real script's `sys.exit(3)` on a stop request,
simplified to a single up-front check since the stub never loops)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--d-override", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--ceiling-gpuh", type=float, default=1.20)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--stop-file", type=str, default="")
    args = ap.parse_args()

    if args.stop_file and os.path.exists(args.stop_file):
        sys.exit(3)

    behavior = os.environ.get("KWALL_STUB_BEHAVIOR", "complete")

    if behavior == "crash":
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    if behavior == "crash_after_dir":
        sys.exit(1)
    if behavior == "hang":
        time.sleep(3600)
        return

    out_path = os.path.join(args.outdir, f"earlyln_K{args.K}_s{args.seed}.json")
    elapsed_s = float(os.environ.get("KWALL_STUB_ELAPSED_S", "1.0"))
    if behavior == "abort_budget":
        rec = {"status": "ABORTED-BUDGET", "K": args.K, "d": args.d_override,
               "d_override": args.d_override, "seed": args.seed,
               "elapsed_s": elapsed_s, "step": 1}
    else:  # "complete"
        # Fake gate1 inputs matching `_cell_gate1`'s exact expected shape
        # (`ncr_earlyln_scale.py:317-329`) so `harvest()` can be called
        # against stub-produced cells without KeyError -- controllable via
        # KWALL_STUB_CONVERGED so orchestrator-integration tests can drive
        # the trigger/band computation deterministically.
        converged = os.environ.get("KWALL_STUB_CONVERGED", "1") == "1"
        indist = 1.0 if converged else 0.0
        aer_val = float(args.K) if converged else 0.0
        rec = {"status": "COMPLETED", "K": args.K, "d": args.d_override,
               "d_override": args.d_override, "seed": args.seed,
               "elapsed_s": elapsed_s,
               "train": {"status": "COMPLETED", "step": args.steps,
                        "loss_history": [[1, 1.0], [args.steps, 0.01]]},
               "gpu_h": elapsed_s / 3600.0,
               "eval": {"points": [
                   {"component": "train_support",
                    "reads": {"binexp": {"recovered_frac@0.9": indist}}}
                   for _ in range(3)]},
               "deep_probe": {"A_eff_rank": [aer_val, aer_val]}}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, out_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
