#!/usr/bin/env python3
"""K-SCALING MATCHED-POOL SCORER (P1b + P0) -- NCR_KSCALING_DESIGN.md sec 6.

Descends from ncr_writecond/poolmatched_battery.py (md5 7c3610cc09303627234b0cd6c9977014,
the instrument that resolved the 2026-08-21 #3 pool-seed quarantine and produced
the #6 adjudication of record). Three things change, all forced by K-scaling:

  1. MATCHED POOLS ARE THE ONLY MODE. The ancestor scored under BOTH the
     checkpoint's own seed and the legacy seed-0 pools, because its job was to
     adjudicate a retraction. That question is settled (#6): trained adapters
     specialise to their training entities and the mismatch compounds with
     composition depth. Every number here is built with the checkpoint's OWN
     recorded seed -- gate memo carried requirement 4, "matched pool seeds in
     ALL scoring from day one". pool_seed / ckpt_seed / matched are recorded on
     every block so the property is auditable, never assumed.

  2. CHANCE IS 1/K, NOT 1/24. The ancestor hard-codes CHANCE = 1.0/24.0. Across
     K in {12..32} chance spans 0.0833 -> 0.0312, so a raw accuracy curve would
     drift with K for a purely arithmetic reason. Every accuracy is reported
     ALONGSIDE margin_over_chance = acc - 1/K, and the P0 wall test uses the
     per-K binomial band chance +/- 3*sqrt(p(1-p)/n) (gate memo requirement 2).

  3. THE HOP GRID IS THIS K'S OWN LADDER. Carried ladders are what the gate
     memo forbids; the ancestor's HOPS=(1,3,61) is meaningless at K=12
     (61 == 1 mod 12, a train residue). Hops come from kscaling_config.

REGIME NAMING (mandatory on every number, inherited verbatim):
  P1b = teacher_force=True  -- EXACT-WRITE: the operator is the closed-form
        pseudoinverse fit, so the READ is what is under test. This is the
        capability curve.
  P0  = teacher_force=False -- the model's own learned encoder writes. This is
        the WALL curve; it should sit at chance = 1/K.

Usage:
  NCR_K=32 kscaling_battery.py --k 32 --ckpt PATH --tag TAG [--cellcfg PATH] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kscaling_config as KS               # noqa: E402
import ncr_lm_wave1_runner as R            # noqa: E402  (patched instrument, library use only)

BASE_SEED = 90210          # eval seed, verbatim from pbe_repl / poolmatched_battery
N_EVAL = 256               # n=256, verbatim
LR = 3e-4
REQUIRED_CKPT_STEP = 20000
METRIC = "retrieval24_acc"     # NAME is historical (argmax over the K slots); it is
                                # K-generic in the code -- eval_arm_at_hops computes
                                # cos_all.argmax(1) == tgt_slot over exactly K slots.
                                # Renaming it would break comparability with the 55
                                # K=24 cells of record, so it is kept and annotated.
WALL_SD_MULT = 3.0


class LoudFailure(RuntimeError):
    pass


def chance_band(k: int, n: int, mult: float = WALL_SD_MULT) -> tuple[float, float, float, float]:
    """Binomial band around chance for the P0 wall test, per K."""
    p = 1.0 / k
    sd = math.sqrt(p * (1.0 - p) / n)
    return p, sd, max(0.0, p - mult * sd), min(1.0, p + mult * sd)


def cell_config(path):
    """aux_loss_type / freeze_entity_adapter as RECORDED by the training run --
    never inferred from the cell name (2026-08-21 #6 instrument discipline)."""
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    c = d.get("config") or {}
    return dict(cfg_aux_loss_type=c.get("aux_loss_type"),
                cfg_freeze_entity_adapter=c.get("freeze_entity_adapter"),
                cfg_seed=c.get("seed"), cfg_K=c.get("K"), cfg_d_ncr=c.get("d_ncr"),
                cfg_aux_read_loss_weight=c.get("aux_read_loss_weight"),
                cfg_ortho_reg_weight=c.get("ortho_reg_weight"),
                cfg_status=d.get("status"), cfg_step=d.get("step"),
                cfg_kscaling=d.get("kscaling"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True,
                    help="mandatory restatement of NCR_K; asserted against kscaling_config "
                         "AND against the checkpoint's own recorded d_ncr")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cellcfg", default=None)
    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_kscaling/results"))
    ap.add_argument("--required-step", type=int, default=REQUIRED_CKPT_STEP)
    # F1 fix (KSCALING_AUDIT_R1): read-only scoring of the K=24 anchor cells of
    # record, which carry the pinned gate3 tag. Allowlisted values only; the
    # runner's own resume guard stays strict.
    ap.add_argument("--anchor-runner-tag", default=None,
                    choices=["ncr_gate3_wave1_runner_v1", "ncr_kscaling_runner_v1",
                             "ncr_scaleaxis_runner_v1"])
    # SCALE-AXIS PORT PATCH (sec 3.2 item 21). The allowlist is EXTENDED and
    # PAIRED WITH B5's scale guard, exactly as the design words it. Extending it
    # alone would be the hazard m5 found; it is extended so that a cross-harness
    # read fails on the SCALE GUARD (the right reason) rather than on the tag
    # (the wrong reason), which is what makes B5's negative test meaningful.
    # Re-measure extension (2026-08-22): independent eval draw for the
    # pre-registered single-seed-excursion re-measure license only.
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--n", type=int, default=N_EVAL)
    args = ap.parse_args()

    assert args.k == KS.K_NCR, (
        f"--k {args.k} != NCR_K-resolved K_NCR={KS.K_NCR}: refusing to score, the numbers "
        f"would be labelled with the wrong K and the chance baseline would be wrong.")

    t0 = time.time()
    device = "cuda"
    path = os.path.expanduser(args.ckpt)
    if args.anchor_runner_tag:
        R.RUNNER_TAG = args.anchor_runner_tag
    ckpt = R.load_checkpoint(path, device)
    if ckpt is None:
        raise LoudFailure(f"CHECKPOINT FAIL [{args.tag}]: {path} missing or failed "
                          "load_checkpoint validation (runner_tag/step/arm keys)")

    # --- THE K GUARD: the checkpoint itself records d_ncr; d must be K+1 -----
    # This is what makes it impossible to score a K=24 checkpoint as if it were
    # K=32 (which would produce a plausible-looking but meaningless curve point).
    ck_d = int(ckpt["full_graft"]["ncr_config"]["d"])
    ck_integ_d = int(ckpt["full_graft"]["integ_config"]["d_ncr"])
    if ck_d != KS.D_NCR or ck_integ_d != KS.D_NCR:
        raise LoudFailure(
            f"K MISMATCH [{args.tag}]: checkpoint records ncr d={ck_d} / integ d_ncr={ck_integ_d}, "
            f"but this process is configured for K={KS.K_NCR} -> d={KS.D_NCR}. The checkpoint was "
            f"trained at K={ck_d - 1}. Refusing to score.")


    # ---- BUILD REQUIREMENT B5 (NCR_SCALE_AXIS_DESIGN.md sec 3.5 m5) ---------
    # restore_arms_and_opts rebuilds the backbone from ckpt[arm]["backbone_config"],
    # NOT from RUNG1_BACKBONE, and the ONLY structural check that existed was the
    # K guard above. So a wrong-SCALE checkpoint at the RIGHT K would load and
    # score SUCCESSFULLY AND SILENTLY the moment the runner-tag allowlist is
    # extended -- which this design requires. In a design whose entire purpose is
    # a cross-scale comparison, that is the one guard that must exist.
    _bb = ckpt["full_graft"].get("backbone_config") or {}
    _want = {kk: R.RUNG1_BACKBONE[kk] for kk in ("d_model", "n_layers", "d_state")}
    _got = {kk: _bb.get(kk) for kk in ("d_model", "n_layers", "d_state")}
    if _got != _want:
        raise LoudFailure(
            f"SCALE MISMATCH [{args.tag}]: checkpoint backbone_config {_got} != this process's "
            f"resolved backbone {_want} (NCR_SCALE={os.environ.get('NCR_SCALE')!r}, "
            f"scale={KS.SCALE!r}). The checkpoint was trained at a DIFFERENT SCALE. Scoring it "
            f"here would produce a plausible-looking cross-scale number that is neither scale. "
            f"Refusing (B5).")

    cfg_rec = cell_config(args.cellcfg)

    step = int(ckpt["step"])
    if step != args.required_step:
        print(f"!!! SKIP+FLAG [{args.tag}]: ckpt_step={step} != {args.required_step}. NOT SCORED.",
              flush=True)
        return 3

    if "seed" in ckpt:
        ckpt_seed, seed_src = int(ckpt["seed"]), "checkpoint"
    elif cfg_rec and cfg_rec.get("cfg_seed") is not None:
        ckpt_seed, seed_src = int(cfg_rec["cfg_seed"]), "training-results-JSON"
    else:
        raise LoudFailure(f"SEED UNRESOLVABLE [{args.tag}]: neither the checkpoint nor "
                          f"{args.cellcfg!r} records a seed -- refusing to guess the pool seed.")

    if "freeze_entity_adapter" in ckpt:
        freeze, fz_src = bool(ckpt["freeze_entity_adapter"]), "checkpoint"
    elif cfg_rec and cfg_rec.get("cfg_freeze_entity_adapter") is not None:
        freeze, fz_src = bool(cfg_rec["cfg_freeze_entity_adapter"]), "training-results-JSON"
    else:
        raise LoudFailure(
            f"ARM LABEL UNRESOLVABLE [{args.tag}]: freeze_entity_adapter absent from BOTH the "
            f"checkpoint and {args.cellcfg!r}. The frozen-vs-trainable ordering is THE secondary "
            f"question of this sweep -- refusing to default the arm label. (The ancestor "
            f"defaulted-and-flagged for 3 pre-flag legacy cells; no cell in this sweep predates "
            f"the flag, so a missing label here means something is wrong.)")

    # --- MATCHED POOLS: the checkpoint's OWN recorded seed, always -----------
    pools, cfg, report = R.build_grammar_pools_and_cfg(seed=ckpt_seed)
    arms, _opts, _gen = R.restore_arms_and_opts(
        ckpt, report["vocab_size_total"], lr=LR, device=device, freeze_entity_adapter=freeze)
    arm = arms["full_graft"]
    pools = pools.to(device)

    hops = tuple(KS.TRAIN_HOPS) + tuple(KS.DEEP_LADDER) + (KS.FIXED_DIST_PROBE,)
    p, sd, lo, hi = chance_band(KS.K_NCR, args.n)

    block = dict(pool_seed=ckpt_seed, ckpt_seed=ckpt_seed, matched=True,
                 matched_pool_policy="checkpoint's own recorded seed (gate memo requirement 4)")
    for regime, tf in (("P1b", True), ("P0", False)):
        t = time.time()
        with torch.no_grad():
            res = R.eval_arm_at_hops(arm, pools, cfg, hops, args.n, device,
                                     args.base_seed, read_ablate=False, teacher_force=tf)
        per_hop = {}
        for h in hops:
            acc = res[f"h={h}"][METRIC]
            per_hop[f"h={h}"] = dict(
                h=h, residue=h % KS.K_NCR, n_squarings=KS.n_squarings(h),
                n_applies=KS.n_applies(h),
                role=("train" if h in KS.TRAIN_HOPS else
                      "fixed_dist_control" if h == KS.FIXED_DIST_PROBE else
                      "ladder_top" if h == KS.H_TOP else "ladder"),
                acc=acc, margin_over_chance=acc - p,
                within_chance_3sd=bool(lo <= acc <= hi))
        block[regime] = dict(
            regime=regime, teacher_force=tf,
            regime_meaning=("EXACT-WRITE (operator teacher-forced from the true binding); "
                            "the READ is under test -- this is the CAPABILITY curve"
                            if tf else
                            f"LEARNED-WRITE (the model's own encoder writes); this is the WALL "
                            f"curve -- chance = 1/K = {p:.4f}"),
            elapsed_s=time.time() - t, per_hop=per_hop, result=res)

    record = dict(
        script="kscaling_battery.py",
        ancestor="ncr_writecond/poolmatched_battery.py md5 7c3610cc09303627234b0cd6c9977014",
        cell=f"KSCALING_{args.tag}", tag=args.tag, ckpt=path,
        instrument="ncr_lm_wave1_runner.eval_arm_at_hops (K-scaling patched copy), matched pools",
        metric_of_record=METRIC,
        metric_note="name is historical; the metric is argmax over exactly K slots, K-generic",
        chance=p, chance_sd_at_n=sd, wall_band=[lo, hi], wall_band_mult=WALL_SD_MULT,
        hops=list(hops), n=args.n, base_seed=args.base_seed,
        ckpt_step=step, ckpt_seed=ckpt_seed, ckpt_seed_source=seed_src,
        ckpt_cell_id=ckpt.get("cell_id"), runner_tag=ckpt.get("runner_tag"),
        ckpt_recorded_d_ncr=ck_d,
        freeze_entity_adapter=freeze, freeze_flag_source=fz_src,
        recipe=("FROZEN-contrastive (primary)" if freeze else "TRAINABLE-contrastive (compB)"),
        # sec 3.2 item 22 (B1's own find): this was `KS.provenance(64, 768)` -- a
        # hard-coded d_model that would have recorded HALF the true integ param
        # count in every 392M score record. Re-pointed at the resolved backbone.
        kscaling=KS.provenance(R.H_NCR, R.RUNG1_BACKBONE["d_model"]),
        scale=KS.SCALE, backbone_config_scored=ckpt["full_graft"].get("backbone_config"),
        cell_config=cfg_rec, matched=block, elapsed_s=time.time() - t0)

    # --- self-check: fail loudly on the silent-zero-scoring bug class --------
    defects = []
    for regime in ("P1b", "P0"):
        blk = record["matched"].get(regime)
        if not blk:
            defects.append(f"{regime}: block missing")
            continue
        for h in hops:
            e = blk["per_hop"].get(f"h={h}")
            if e is None:
                defects.append(f"{regime}/h={h}: missing")
                continue
            v = e["acc"]
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                defects.append(f"{regime}/h={h}: acc={v!r} not finite")
            elif not (0.0 <= v <= 1.0):
                defects.append(f"{regime}/h={h}: acc={v} out of [0,1]")
            if blk["result"].get(f"h={h}", {}).get("n") != args.n:
                defects.append(f"{regime}/h={h}: n != {args.n}")
        residues = [blk["per_hop"][f"h={h}"]["residue"] for h in KS.DEEP_LADDER]
        if len(set(residues)) != len(residues):
            defects.append(f"{regime}: ladder residues {residues} are not distinct")
    record["self_check_defects"] = defects
    record["self_check"] = "PASS" if not defects else "FAIL"

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}_kscaling.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, default=str)
    os.replace(tmp, out)

    if defects:
        print(f"!!! SELF-CHECK FAIL [{args.tag}]: {len(defects)} defect(s) -- NOT VALID", flush=True)
        for d in defects:
            print(f"    - {d}", flush=True)
        return 4

    # Liveness only -- no verdict, no band adjudication printed (blind discipline,
    # matching the runner's own convention).
    top = record["matched"]["P1b"]["per_hop"][f"h={KS.H_TOP}"]
    p0max = max(record["matched"]["P0"]["per_hop"][f"h={h}"]["acc"] for h in hops)
    print(f"[{args.tag}] K={KS.K_NCR} d={KS.D_NCR} freeze={freeze} ckpt_seed={ckpt_seed} "
          f"| P1b@h_top={KS.H_TOP} acc={top['acc']:.4f} margin={top['margin_over_chance']:+.4f} "
          f"| P0max={p0max:.4f} (chance {p:.4f}, band [{lo:.4f},{hi:.4f}])", flush=True)
    print(f"KSCALING_{args.tag}_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LoudFailure as e:
        print(f"!!! LOUD FAILURE: {e}", flush=True)
        sys.exit(5)
