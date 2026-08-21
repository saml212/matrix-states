#!/usr/bin/env python3
"""FULL PREMISE BATTERY UNDER MATCHED ENTITY-POOL SEEDS (quarantine resolution).

Context: the audited `pbe_repl` instrument hardcodes
build_grammar_pools_and_cfg(seed=0), but cell mob_*_sN was trained with
--seed N, and that seed drives build_entity_pools(..., heldout_frac=0.5,
seed=...) -- the entity train/heldout SPLIT. So every premise-battery
number of record on a seed!=0 checkpoint was scored against a partially
DIFFERENT entity vocabulary than the cell trained on. A frozen-at-init
entity_adapter is indifferent to which entities those are; a TRAINED one
is not. That asymmetry lands exactly on the freeze axis, so it can
manufacture a freeze effect. This script re-scores every cell under the
checkpoint's OWN pool seed and, in the same process, under pool seed 0,
so the two are directly comparable on identical batches.

NOTHING of record is overwritten: output goes to *_poolmatched.json.
The audited instrument is imported as a library and used unmodified.

Regime naming is mandatory on every number:
  P1b = teacher_force=True  -- EXACT-WRITE (the read is under test)
  P0  = teacher_force=False -- the model's own learned encoder writes
        (the "wall" claim: P0 should sit at chance = 1/24 ~= 0.0417)

Usage: poolmatched_battery.py --ckpt PATH --tag TAG [--cellcfg PATH] [--outdir DIR]
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncr_lm_wave1_runner as R  # noqa: E402  (AUDITED instrument, library use only)

BASE_SEED = 90210          # eval seed, verbatim from pbe_repl
N_EVAL = 256               # n=256, verbatim
LR = 3e-4
REQUIRED_CKPT_STEP = 20000
METRIC = "retrieval24_acc"
CHANCE = 1.0 / 24.0
HOPS = (1, 3, 61)          # h=1 and h=61 are the hops of record; h=3 added (train-hop, cheap)
REFERENCE_POOL_SEED = 0    # the seed the instrument of record hardcoded


class LoudFailure(RuntimeError):
    pass


def cell_config(cellcfg_path):
    """aux_loss_type / freeze_entity_adapter as RECORDED by the training run.
    The checkpoint does not carry aux_loss_type, so the 2x2 arm label cannot be
    read from it -- refuse to guess if the training record is unavailable."""
    if not cellcfg_path or not os.path.exists(cellcfg_path):
        return None
    d = json.load(open(cellcfg_path))
    c = d.get("config") or {}
    return dict(cfg_aux_loss_type=c.get("aux_loss_type"),
                cfg_freeze_entity_adapter=c.get("freeze_entity_adapter"),
                cfg_seed=c.get("seed"),
                cfg_aux_read_loss_weight=c.get("aux_read_loss_weight"),
                cfg_ortho_reg_weight=c.get("ortho_reg_weight"),
                cfg_status=d.get("status"), cfg_step=d.get("step"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cellcfg", default=None,
                    help="the cell's training results JSON (source of aux_loss_type)")
    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_writecond/results"))
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda"
    path = os.path.expanduser(args.ckpt)
    ckpt = R.load_checkpoint(path, device)
    if ckpt is None:
        raise LoudFailure(f"CHECKPOINT FAIL [{args.tag}]: {path} missing or failed "
                          "load_checkpoint validation (runner_tag/step/arm keys)")

    cfg_rec = cell_config(args.cellcfg)

    # --- ckpt_step guard, on EVERY cell -----------------------------------
    step = int(ckpt["step"])
    if step != REQUIRED_CKPT_STEP:
        print(f"!!! SKIP+FLAG [{args.tag}]: ckpt_step={step} != {REQUIRED_CKPT_STEP}. NOT SCORED.",
              flush=True)
        return 3

    # --- seed: from the checkpoint; fall back to the training record ------
    if "seed" in ckpt:
        ckpt_seed, seed_src = int(ckpt["seed"]), "checkpoint"
    elif cfg_rec and cfg_rec.get("cfg_seed") is not None:
        ckpt_seed, seed_src = int(cfg_rec["cfg_seed"]), "training-results-JSON (key absent from ckpt: cell predates sec G3-B26)"
    else:
        raise LoudFailure(f"SEED UNRESOLVABLE [{args.tag}]: neither the checkpoint nor "
                          f"{args.cellcfg!r} records a seed -- refusing to guess the pool seed.")

    # --- freeze flag: from the checkpoint; fall back to the training record
    if "freeze_entity_adapter" in ckpt:
        freeze, fz_src = bool(ckpt["freeze_entity_adapter"]), "checkpoint"
    elif cfg_rec and cfg_rec.get("cfg_freeze_entity_adapter") is not None:
        freeze, fz_src = bool(cfg_rec["cfg_freeze_entity_adapter"]), "training-results-JSON"
    else:
        freeze, fz_src = False, ("DEFAULTED to False -- flag absent from BOTH checkpoint and "
                                 "training record (cell predates sec G3-B31). FLAGGED, not trusted.")
        print(f"### ADVISORY [{args.tag}]: {fz_src}", flush=True)

    # --- restore ONCE; the arms do not depend on the entity pools ---------
    pools_m, cfg_m, report_m = R.build_grammar_pools_and_cfg(seed=ckpt_seed)
    arms, _opts, _gen = R.restore_arms_and_opts(
        ckpt, report_m["vocab_size_total"], lr=LR, device=device, freeze_entity_adapter=freeze)
    arm = arms["full_graft"]

    pool_seeds = [ckpt_seed] if ckpt_seed == REFERENCE_POOL_SEED else [ckpt_seed, REFERENCE_POOL_SEED]
    by_pool = {}
    for ps in pool_seeds:
        if ps == ckpt_seed:
            pools, cfg, report = pools_m, cfg_m, report_m
        else:
            pools, cfg, report = R.build_grammar_pools_and_cfg(seed=ps)
        if report["vocab_size_total"] != report_m["vocab_size_total"]:
            raise LoudFailure(
                f"VOCAB SIZE DIVERGES [{args.tag}]: pool seed {ps} gives "
                f"{report['vocab_size_total']} vs {report_m['vocab_size_total']} at seed "
                f"{ckpt_seed} -- the two readings would not be comparable.")
        pools = pools.to(device)
        block = dict(pool_seed=ps, matched=(ps == ckpt_seed),
                     is_instrument_of_record_seed=(ps == REFERENCE_POOL_SEED))
        for regime, tf in (("P1b", True), ("P0", False)):
            t = time.time()
            with torch.no_grad():
                res = R.eval_arm_at_hops(arm, pools, cfg, HOPS, N_EVAL, device,
                                         BASE_SEED, read_ablate=False, teacher_force=tf)
            block[regime] = dict(
                regime=regime, teacher_force=tf,
                regime_meaning=("EXACT-WRITE (operator teacher-forced from the true binding)"
                                if tf else
                                "LEARNED-WRITE (the model's own encoder writes); chance = 1/24"),
                elapsed_s=time.time() - t, result=res)
        by_pool[f"pool_seed={ps}"] = block

    record = dict(
        script="poolmatched_battery.py",
        cell=f"POOLMATCHED_{args.tag}", tag=args.tag, ckpt=path,
        instrument="ncr_lm_wave1_runner.eval_arm_at_hops via the pbe_repl call pattern (audited, unmodified)",
        metric_of_record=METRIC, chance=CHANCE, hops=list(HOPS),
        n=N_EVAL, base_seed=BASE_SEED,
        ckpt_step=step, ckpt_seed=ckpt_seed, ckpt_seed_source=seed_src,
        ckpt_cell_id=ckpt.get("cell_id"), runner_tag=ckpt.get("runner_tag"),
        freeze_entity_adapter=freeze, freeze_flag_source=fz_src,
        matched_pool_seed=ckpt_seed, reference_pool_seed=REFERENCE_POOL_SEED,
        trivially_matched=(ckpt_seed == REFERENCE_POOL_SEED),
        cell_config=cfg_rec, by_pool=by_pool, elapsed_s=time.time() - t0)

    # --- self-check: fail loudly on the silent-zero-scoring bug class -----
    defects = []
    for ps in pool_seeds:
        blk = by_pool.get(f"pool_seed={ps}")
        if not blk:
            defects.append(f"pool_seed={ps}: entire block missing")
            continue
        for regime in ("P1b", "P0"):
            res = blk.get(regime, {}).get("result", {})
            for h in HOPS:
                v = res.get(f"h={h}", {}).get(METRIC)
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    defects.append(f"pool_seed={ps}/{regime}/h={h}: {METRIC}={v!r} not finite")
                elif not (0.0 <= v <= 1.0):
                    defects.append(f"pool_seed={ps}/{regime}/h={h}: {METRIC}={v} out of [0,1]")
                if res.get(f"h={h}", {}).get("n") != N_EVAL:
                    defects.append(f"pool_seed={ps}/{regime}/h={h}: n != {N_EVAL}")
    record["self_check_defects"] = defects
    record["self_check"] = "PASS" if not defects else "FAIL"

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}_poolmatched.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp, out)

    if defects:
        print(f"!!! SELF-CHECK FAIL [{args.tag}]: {len(defects)} defect(s) -- NOT VALID", flush=True)
        for d in defects:
            print(f"    - {d}", flush=True)
        return 4

    m = by_pool[f"pool_seed={ckpt_seed}"]
    parts = []
    for ps in pool_seeds:
        b = by_pool[f"pool_seed={ps}"]
        tagps = "matched" if ps == ckpt_seed else "seed0"
        parts.append(f"{tagps}: P1b h1={b['P1b']['result']['h=1'][METRIC]:.4f} "
                     f"h61={b['P1b']['result']['h=61'][METRIC]:.4f} "
                     f"P0max={max(b['P0']['result'][f'h={h}'][METRIC] for h in HOPS):.4f}")
    print(f"[{args.tag}] freeze={freeze} ckpt_seed={ckpt_seed} | " + " | ".join(parts), flush=True)
    print(f"POOLMATCHED_{args.tag}_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LoudFailure as e:
        print(f"!!! LOUD FAILURE: {e}", flush=True)
        sys.exit(5)
