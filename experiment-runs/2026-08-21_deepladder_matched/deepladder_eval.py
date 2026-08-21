#!/usr/bin/env python3
"""DEEP-LADDER MATCHED-POOL EVAL (residual-ordering question).

Pre-registration: EXPERIMENT_LOG.md "2026-08-21 #7" (commit ad52dcf), BINDING.
Question: the matched-pool frozen-vs-trainable gap at h=61 was +0.0098 and
ceiling-compressed (medians 0.978-1.000). Does the residual ordering become
measurable OFF-CEILING at greater physical depth?

  hops  = {61, 253, 1021, 4093}  -- ALL == 13 (mod 24). K=24 is a single
          Hamiltonian cycle, so these four depths have the IDENTICAL GROUND
          TRUTH BY CONSTRUCTION. Any spread across them is the numerical cost
          of 5/7/9/11 binexp squarings, NOT new compositional information.
          Every output record repeats this label.
  pools = MATCHED to the checkpoint's own seed (the #6 retraction: scoring a
          trained entity_adapter against a different entity train/heldout
          split manufactures a depth-amplified freeze effect).
  P1b   = teacher_force=True, EXACT-WRITE -- all four hops.
  P0    = teacher_force=False, LEARNED-WRITE -- h=61 ONLY (cost-capped
          spot-check per #7; chance = 1/24).

Bands (#7, applied by the aggregator, not here):
  RESIDUAL-CONFIRMED  = frozen-vs-trainable median gap > 0.05 AND
                        Mann-Whitney p < 0.01 at ANY h >= 253
  RESIDUAL-NEGLIGIBLE = gap <= 0.05 at EVERY h

Audited instrument imported as a library, unmodified. New files only.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncr_lm_wave1_runner as R  # noqa: E402  (AUDITED, library use only)

BASE_SEED = 90210
N_EVAL = 256
LR = 3e-4
REQUIRED_CKPT_STEP = 20000
METRIC = "retrieval24_acc"
CHANCE = 1.0 / 24.0
K_CYCLE = 24
HOPS_P1B = (61, 253, 1021, 4093)
HOPS_P0 = (61,)
FIXED_RESIDUE = 13
SAME_GT_LABEL = ("ALL hops in this ladder are == 13 (mod 24); the K=24 ground truth is a single "
                 "Hamiltonian cycle, so all four depths have the IDENTICAL correct answer by "
                 "construction. Spread across depths is NUMERICAL (5/7/9/11 squarings), not "
                 "compositional.")


class LoudFailure(RuntimeError):
    pass


def preflight():
    res = sorted({h % K_CYCLE for h in HOPS_P1B})
    if res != [FIXED_RESIDUE]:
        raise LoudFailure(f"PREFLIGHT FAIL: ladder {list(HOPS_P1B)} residues {res} != "
                          f"[{FIXED_RESIDUE}] -- the same-ground-truth label would be FALSE")
    # The instrument's own ladder-soundness rule, volunteered (it is a module-level
    # check on DEEP_LADDER only and is never invoked by eval_arm_at_hops).
    R._assert_ladder_sound(HOPS_P1B, K_CYCLE, R.TRAIN_HOPS)
    for h in HOPS_P1B:
        if not (isinstance(h, int) and h >= 1):
            raise LoudFailure(f"PREFLIGHT FAIL: binexp_read needs int h>=1, got {h!r}")
    return {f"h={h}": max(0, h.bit_length() - 1) for h in HOPS_P1B}


def cell_config(p):
    if not p or not os.path.exists(p):
        return None
    d = json.load(open(p))
    c = d.get("config") or {}
    return dict(cfg_aux_loss_type=c.get("aux_loss_type"),
                cfg_freeze_entity_adapter=c.get("freeze_entity_adapter"),
                cfg_seed=c.get("seed"), cfg_status=d.get("status"), cfg_step=d.get("step"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cellcfg", default=None)
    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_writecond/results"))
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda"
    squarings = preflight()

    path = os.path.expanduser(args.ckpt)
    ckpt = R.load_checkpoint(path, device)
    if ckpt is None:
        raise LoudFailure(f"CHECKPOINT FAIL [{args.tag}]: {path} missing or failed validation")
    cfg_rec = cell_config(args.cellcfg)

    step = int(ckpt["step"])
    if step != REQUIRED_CKPT_STEP:
        print(f"!!! SKIP+FLAG [{args.tag}]: ckpt_step={step} != {REQUIRED_CKPT_STEP}. NOT SCORED.",
              flush=True)
        return 3

    if "seed" in ckpt:
        ckpt_seed, seed_src = int(ckpt["seed"]), "checkpoint"
    elif cfg_rec and cfg_rec.get("cfg_seed") is not None:
        ckpt_seed, seed_src = int(cfg_rec["cfg_seed"]), "training-results-JSON (absent from ckpt)"
    else:
        raise LoudFailure(f"SEED UNRESOLVABLE [{args.tag}] -- refusing to guess the pool seed")

    if "freeze_entity_adapter" in ckpt:
        freeze, fz_src = bool(ckpt["freeze_entity_adapter"]), "checkpoint"
    elif cfg_rec and cfg_rec.get("cfg_freeze_entity_adapter") is not None:
        freeze, fz_src = bool(cfg_rec["cfg_freeze_entity_adapter"]), "training-results-JSON"
    else:
        freeze, fz_src = False, ("DEFAULTED to False -- absent from BOTH checkpoint and training "
                                 "record (predates sec G3-B31). FLAGGED, not trusted.")
        print(f"### ADVISORY [{args.tag}]: {fz_src}", flush=True)

    # MATCHED pools, per the #6 retraction.
    pools, cfg, report = R.build_grammar_pools_and_cfg(seed=ckpt_seed)
    pools = pools.to(device)
    arms, _o, _g = R.restore_arms_and_opts(ckpt, report["vocab_size_total"], lr=LR,
                                           device=device, freeze_entity_adapter=freeze)
    arm = arms["full_graft"]

    out_blocks = {}
    t = time.time()
    with torch.no_grad():
        p1b = R.eval_arm_at_hops(arm, pools, cfg, HOPS_P1B, N_EVAL, device,
                                 BASE_SEED, read_ablate=False, teacher_force=True)
    out_blocks["P1b"] = dict(regime="P1b", teacher_force=True,
                             regime_meaning="EXACT-WRITE (operator teacher-forced from the true binding)",
                             hops=list(HOPS_P1B), elapsed_s=time.time() - t, result=p1b)
    t = time.time()
    with torch.no_grad():
        p0 = R.eval_arm_at_hops(arm, pools, cfg, HOPS_P0, N_EVAL, device,
                                BASE_SEED, read_ablate=False, teacher_force=False)
    out_blocks["P0"] = dict(regime="P0", teacher_force=False,
                            regime_meaning="LEARNED-WRITE (model's own encoder writes); chance = 1/24",
                            hops=list(HOPS_P0), note="cost-capped spot-check at h=61 only, per #7",
                            elapsed_s=time.time() - t, result=p0)

    record = dict(
        script="deepladder_eval.py", cell=f"DEEPLADDER_{args.tag}", tag=args.tag, ckpt=path,
        prereg="EXPERIMENT_LOG.md 2026-08-21 #7 (commit ad52dcf)",
        instrument="ncr_lm_wave1_runner.eval_arm_at_hops (audited, unmodified)",
        metric_of_record=METRIC, chance=CHANCE, K_cycle=K_CYCLE,
        hops_P1b=list(HOPS_P1B), hops_P0=list(HOPS_P0),
        residue_mod_K=FIXED_RESIDUE, same_ground_truth_by_construction=True,
        same_ground_truth_label=SAME_GT_LABEL, n_squarings=squarings,
        hop_residues={f"h={h}": h % K_CYCLE for h in HOPS_P1B},
        n=N_EVAL, base_seed=BASE_SEED,
        pool_seed=ckpt_seed, pools_matched=True,
        ckpt_step=step, ckpt_seed=ckpt_seed, ckpt_seed_source=seed_src,
        ckpt_cell_id=ckpt.get("cell_id"), runner_tag=ckpt.get("runner_tag"),
        freeze_entity_adapter=freeze, freeze_flag_source=fz_src,
        cell_config=cfg_rec, **out_blocks, elapsed_s=time.time() - t0)

    defects = []
    for regime, hops in (("P1b", HOPS_P1B), ("P0", HOPS_P0)):
        res = record[regime]["result"]
        for h in hops:
            cell = res.get(f"h={h}")
            if cell is None:
                defects.append(f"{regime}/h={h}: NO OUTPUT (hop never scored)")
                continue
            v = cell.get(METRIC)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                defects.append(f"{regime}/h={h}: {METRIC}={v!r} not finite")
            elif not (0.0 <= v <= 1.0):
                defects.append(f"{regime}/h={h}: {METRIC}={v} out of [0,1]")
            if cell.get("n") != N_EVAL:
                defects.append(f"{regime}/h={h}: n={cell.get('n')!r} != {N_EVAL}")
    record["self_check_defects"] = defects
    record["self_check"] = "PASS" if not defects else "FAIL"

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}_deepladder.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp, out)

    if defects:
        print(f"!!! SELF-CHECK FAIL [{args.tag}]: {len(defects)} defect(s) -- NOT VALID", flush=True)
        for d in defects:
            print(f"    - {d}", flush=True)
        return 4

    print(f"[{args.tag}] freeze={freeze} pool_seed={ckpt_seed} (MATCHED) | P1b "
          + " ".join(f"h{h}={p1b[f'h={h}'][METRIC]:.4f}" for h in HOPS_P1B)
          + f" | P0 h61={p0['h=61'][METRIC]:.4f}", flush=True)
    print(f"DEEPLADDER_{args.tag}_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LoudFailure as e:
        print(f"!!! LOUD FAILURE: {e}", flush=True)
        sys.exit(5)
