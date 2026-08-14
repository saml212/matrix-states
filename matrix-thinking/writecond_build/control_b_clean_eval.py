#!/usr/bin/env python3
"""CONTROL B's separate clean-eval script (D7, M4(a) repair) -- on the
premise battery's own `pbe_repl.py` pattern
(experiment-runs/2026-08-13_ncr_writecond_premise_battery/pbe_repl.py.txt),
reused, not reinvented.

WHY a separate script: the pinned `eval_both_arms(..., teacher_force=
teacher_force_operator)` call means a CONTROL_B continuation (trained WITH
`--teacher-force-operator` throughout, D7's own definition -- 2,000 further
steps on the compB checkpoint, so `ncr_head`/the encoder receive EXACTLY
zero gradient the whole continuation) produces teacher-forced (P1b-shaped)
evals for the ENTIRE run -- there is no clean `Z_sgd` readout inside the
training-time `rec` at all (M4(a)'s finding). This script is the SEPARATE,
`teacher_force=False` invocation D7 names as CONTROL_B's actual scoreable
output; `band0_checker.py`'s CONTROL_B branch REQUIRES this script's own
output as `clean_eval_rec` before CONTROL_B can PASS Band 0.

Always restores with `freeze_entity_adapter=True` (D7, M4(b)): CONTROL_B's
own continuation was launched with `--freeze-entity-adapter` (M4(b)'s
repair -- `entity_adapter` is NOT frozen by the teacher-force zero-grad
guarantee alone; compB's own base recipe is `freeze_entity_adapter=False`),
so the checkpoint's own optimizer/param-group shape and `requires_grad`
state assume it; restoring WITHOUT it would be exactly the class of
mismatch `save_checkpoint`'s own docstring warns about for `seed` (runner
:1088-1105) and `freeze_entity_adapter` (runner:1097-1105) -- asserted
here explicitly, not silently assumed.

Box-only (imports `ncr_lm_wave1_runner`, which needs `fla`) -- cannot be
run/imported off the box; matches every other script in this build's own
box-only tier (`stage0prime_eval.py`, the premise battery's own scripts).

Usage (matches pbe_repl.py's own CLI exactly):
  python3 control_b_clean_eval.py <ckpt_path> <tag>
e.g.
  python3 control_b_clean_eval.py \\
      ~/ncr_writecond/results/writecond_controlB_s0_ckpts/writecond_controlB_s0.ckpt.pt controlB
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncr_lm_wave1_runner as R          # noqa: E402

BASE_SEED = 90210
HOPS = (1, 13, 37, 61)          # the premise battery's own eval-ladder convention, reused verbatim


def main():
    ckpt_path, tag = sys.argv[1], sys.argv[2]
    t0 = time.time()
    device = "cuda"
    pools, cfg, pool_report = R.build_grammar_pools_and_cfg(seed=0)
    pools = pools.to(device)
    ckpt = R.load_checkpoint(os.path.expanduser(ckpt_path), device)
    assert ckpt is not None, f"checkpoint missing at {ckpt_path}"

    # R5 F4 repair, coordinator-adjudicated option (a): CONTROL B is a
    # warm-start continuation of compB (freeze_entity_adapter=False in the
    # checkpoint), and the pinned runner's resume assert (runner:1228-1232)
    # ABORTS a freeze-flag mismatch -- so the D7 hard-assert here made the
    # cell unproducible either way (R4 F4's void-by-construction, one layer
    # down). Per §A5: run the continuation WITHOUT the freeze, disclose the
    # R3 M4(b) drift confound as a weakening of the "isolates the read
    # side" claim, and record verification as a WARNING FIELD, never abort.
    fe = bool(ckpt.get("freeze_entity_adapter", False))
    freeze_entity_adapter_verified = fe is True
    if not freeze_entity_adapter_verified:
        print(f"WARNING: CONTROL_B checkpoint records freeze_entity_adapter={fe!r} "
              f"(compB's own recipe); the M4(b) adapter-drift confound applies and is "
              f"DISCLOSED in the output record (R5 F4 option (a)).", file=sys.stderr)

    arms, opts, data_gen = R.restore_arms_and_opts(ckpt, pool_report["vocab_size_total"],
                                                    lr=3e-4, device=device, freeze_entity_adapter=fe)
    with torch.no_grad():
        # P0: the CLEAN readout -- teacher_force=False. THIS is the
        # artifact band0_checker.py's CONTROL_B branch requires as
        # `clean_eval_rec`, and the ONLY scoreable output CONTROL_B has
        # (D7's whole reason for this separate script existing).
        p0 = R.eval_arm_at_hops(arms["full_graft"], pools, cfg, HOPS, 256, device,
                                 BASE_SEED, read_ablate=False, teacher_force=False)
        # P1b: informational only (the checkpoint's own teacher-forced
        # ceiling, for reference against the continuation's OWN
        # training-time rec, which is teacher-forced throughout by
        # construction, band0_checker.py's CONTROL_B branch) -- NEVER
        # gating, never CONTROL_B's own scoreable output.
        p1b = R.eval_arm_at_hops(arms["full_graft"], pools, cfg, HOPS, 256, device,
                                  BASE_SEED, read_ablate=False, teacher_force=True)

    out = os.path.expanduser(f"~/ncr_writecond/results/writecond_controlB_clean_eval_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = dict(
        cell=f"CONTROL_B_CLEAN_{tag}", ckpt=ckpt_path, ckpt_step=ckpt["step"], n=256,
        config=dict(teacher_force_operator=False),   # THIS run's own flag -- what band0_checker.py's clean_eval_ok reads
        freeze_entity_adapter_verified=freeze_entity_adapter_verified,   # R5 F4(a): warning field, never an abort
        P0=dict(teacher_force=False, result=p0),
        P1b_informational=dict(teacher_force=True, result=p1b),
        elapsed_s=time.time() - t0,
    )
    with open(out, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"CONTROL_B_CLEAN_{tag}_DONE -> {out}")


if __name__ == "__main__":
    main()
