#!/usr/bin/env python3
"""DEPTH EXTENSION ACROSS K -- thin wrapper over the audited eval path.

Pre-registration: EXPERIMENT_LOG.md "2026-08-22 #2" final block (commit
9bf2f40), BINDING.

WHY A WRAPPER AND NOT kscaling_battery.py: the battery's hop grid is fixed
(`hops = TRAIN_HOPS + DEEP_LADDER + (FIXED_DIST_PROBE,)`) and it exposes no
custom-hop flag, so it cannot express this ladder. Nothing audited is
modified: this file imports kscaling_config and ncr_lm_wave1_runner and calls
the SAME `eval_arm_at_hops` path the battery calls, with the battery's own
guards (K/d_ncr guard, ckpt_step guard, seed + freeze read from the
checkpoint, matched pools, anchor-runner-tag allowlist) reproduced.
Battery of record: md5 5735c788563d9a21f2198c9f5b4793d5 (unmodified).

THE LADDER (derived per K, never hand-entered):
  r_fix(K) = FIXED_DIST_PROBE(K) mod K  -- which is FIXED_DIST_RESIDUE = 4 for
  every K in the grid, i.e. effective composition distance 4 at every K.
  For each squaring count s in {5, 7, 9, 11}, the rung is the SMALLEST
  h in [2^s, 2^(s+1)-1] with h == r_fix (mod K), so n_squarings(h) == s
  exactly. Rung 1 therefore reproduces the config's own FIXED_DIST_TABLE[K],
  which is asserted at run time as a cross-check against the design §4.2 table.

  All four rungs of a given K share ONE residue, so within a K they have the
  IDENTICAL GROUND TRUTH BY CONSTRUCTION -- spread across them is the
  numerical cost of 5/7/9/11 binexp squarings, not compositional information.
  Labelled as such on every record. (Across K the ground truth differs: K
  indexes a different cycle and chance is 1/K.)

REGIMES (mandatory naming):
  P1b = teacher_force=True  -- EXACT-WRITE, the capability curve. All 4 rungs.
  P0  = teacher_force=False -- LEARNED-WRITE, the wall curve. DEEPEST RUNG
        ONLY (11 squarings), per the pre-registration's cost cap.
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
import kscaling_config as KS       # noqa: E402
import ncr_lm_wave1_runner as R    # noqa: E402  (audited instrument, library use only)

BASE_SEED = 90210
N_EVAL = 256
LR = 3e-4
REQUIRED_CKPT_STEP = 20000
METRIC = "retrieval24_acc"   # historical name; argmax over exactly K slots, K-generic
WALL_SD_MULT = 3.0
SQUARING_PROFILE = (5, 7, 9, 11)
SAME_GT = ("All rungs of this K share residue r_fix (mod K): IDENTICAL GROUND TRUTH BY "
           "CONSTRUCTION per K. Spread across rungs is numerical squaring-depth cost, not "
           "compositional information.")


class LoudFailure(RuntimeError):
    pass


def band_for(s: int) -> tuple[int, int]:
    return (1 << s), (1 << (s + 1)) - 1


def derive_depth_ladder(k: int) -> tuple[int, ...]:
    """Smallest admissible h at each pinned squaring count, all at r_fix(k)."""
    r_fix = KS.FIXED_DIST_TABLE[k] % k
    if r_fix not in KS.admissible_residues(k):
        raise LoudFailure(f"K={k}: r_fix={r_fix} is not admissible (identity or a train "
                          f"residue) -- the ladder would not be held-out")
    ladder = []
    for s in SQUARING_PROFILE:
        lo, hi = band_for(s)
        cands = [h for h in range(lo, hi + 1) if h % k == r_fix]
        if not cands:
            raise LoudFailure(f"K={k}: no h in [{lo},{hi}] with residue {r_fix} -- "
                              f"squaring count {s} is unreachable at this K")
        ladder.append(min(cands))
    return tuple(ladder)


def verify_ladder(k: int, ladder: tuple[int, ...]) -> dict:
    """Mechanical verification of every property the pre-registration relies on."""
    r_fix = KS.FIXED_DIST_TABLE[k] % k
    residues = [h % k for h in ladder]
    squarings = [KS.n_squarings(h) for h in ladder]
    if len(set(residues)) != 1 or residues[0] != r_fix:
        raise LoudFailure(f"K={k}: ladder {ladder} residues {residues} are not all r_fix={r_fix} "
                          f"-- the same-ground-truth-per-K label would be FALSE")
    if tuple(squarings) != SQUARING_PROFILE:
        raise LoudFailure(f"K={k}: ladder {ladder} squarings {squarings} != {SQUARING_PROFILE}")
    if list(ladder) != sorted(ladder) or len(set(ladder)) != len(ladder):
        raise LoudFailure(f"K={k}: ladder {ladder} not strictly increasing")
    if ladder[0] != KS.FIXED_DIST_TABLE[k]:
        raise LoudFailure(f"K={k}: rung 1 {ladder[0]} != config FIXED_DIST_TABLE {KS.FIXED_DIST_TABLE[k]} "
                          f"-- wrapper and design table disagree")
    for h in ladder:
        if not (isinstance(h, int) and h >= 1):
            raise LoudFailure(f"K={k}: binexp_read requires int h>=1, got {h!r}")

    # ---- WHICH GUARD APPLIES, AND WHY (disclosed judgment call) -------------
    # kscaling_config.assert_ladder_sound (which the K-scaling copy of
    # R._assert_ladder_sound delegates to) enforces PAIRWISE RESIDUE
    # DISTINCTNESS. That is the correct guard for the BREADTH ladder
    # (DEEP_LADDER), whose rungs must each measure a DIFFERENT ground truth.
    # It is structurally inapplicable to THIS object, a fixed-effective-
    # distance probe family, whose rungs must all share ONE residue -- that is
    # what the pre-registration mandates ("hops h == r_fix mod K") and what
    # makes "same ground truth by construction per K" true. Calling it here
    # would be asserting the negation of the design.
    #
    # This is the config's OWN established convention, not a new exemption:
    # kscaling_config's module docstring says FIXED_DIST_PROBE "DELIBERATELY
    # shares residue 4 with ladder rung 1 ... so it is kept OUT of DEEP_LADDER
    # and carried as its own labelled probe rather than being smuggled past the
    # distinctness assert."  Our rung 1 IS FIXED_DIST_PROBE (asserted above).
    #
    # So we enforce the two legs that DO apply -- the original pinned
    # _assert_ladder_sound's checks -- plus the inverse of distinctness, which
    # is mandatory here and which no existing guard checks.
    train_residues = {h % k for h in R.TRAIN_HOPS}
    for h in ladder:
        r = h % k
        if r == 0:
            raise LoudFailure(f"K={k}: rung h={h} is IDENTITY mod K -- confounded, not held-out")
        if r in train_residues:
            raise LoudFailure(f"K={k}: rung h={h} residue {r} collides with a train residue "
                              f"{sorted(train_residues)} -- secretly in-distribution")
    if len(set(residues)) != 1:
        raise LoudFailure(f"K={k}: rungs span residues {residues}; this ladder MUST be "
                          f"single-residue or the same-ground-truth label is false")

    return dict(r_fix=r_fix, residues=residues, n_squarings=squarings,
                n_applies=[KS.n_applies(h) for h in ladder],
                effective_distance=r_fix, rung1_matches_fixed_dist_table=True,
                ladder_guard=("identity + train-residue legs of the pinned _assert_ladder_sound, "
                              "PLUS single-residue enforcement. The kscaling pairwise-DISTINCTNESS "
                              "leg is deliberately NOT applied: it is the breadth-ladder guard and "
                              "asserts the negation of this fixed-distance construction (see "
                              "kscaling_config's FIXED_DIST_PROBE docstring, same convention). "
                              "DISCLOSED for adjudication."))


def chance_band(k: int, n: int, mult: float = WALL_SD_MULT):
    p = 1.0 / k
    sd = math.sqrt(p * (1.0 - p) / n)
    return p, sd, max(0.0, p - mult * sd), min(1.0, p + mult * sd)


def cell_config(path):
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    c = d.get("config") or {}
    return dict(cfg_aux_loss_type=c.get("aux_loss_type"),
                cfg_freeze_entity_adapter=c.get("freeze_entity_adapter"),
                cfg_seed=c.get("seed"), cfg_K=c.get("K"), cfg_d_ncr=c.get("d_ncr"),
                cfg_status=d.get("status"), cfg_step=d.get("step"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cellcfg", default=None)
    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_scaleaxis/results"))
    # SCALE-AXIS PORT PATCH S4 == AUDIT-R1 MAJOR-5 / condition C4. The pinned
    # default pointed at ~/ncr_kscaling/results -- the 98M tree, which today
    # holds 103 records of record plus results_depthext6/. A 392M record landing
    # there COLLIDES ON-KEY with its 98M twin (rdelta_aggregate keys on
    # K/recipe/seed, not scale) and whichever path sorts later silently wins,
    # then reads as a 98M number in Rule R-delta AND in the exact-reproduction
    # cross-check. B5 does NOT close this: B5 guards the checkpoint INPUT, this
    # is the output DESTINATION. Latent-not-executing today (kappa_reader always
    # passes --outdir explicitly and no spec invokes a scorer), but sec 4.6's
    # Stage C harvest is a manual, unscripted path -- precisely where a bare
    # invocation happens. Closed at the write end here and at the read end by
    # rdelta_aggregate.load()'s scale assert.
    ap.add_argument("--required-step", type=int, default=REQUIRED_CKPT_STEP)
    ap.add_argument("--anchor-runner-tag", default=None,
                    choices=["ncr_gate3_wave1_runner_v1", "ncr_kscaling_runner_v1",
                             "ncr_scaleaxis_runner_v1"])
    # SCALE-AXIS PORT PATCH (sec 3.2 item 21). The allowlist is EXTENDED and
    # PAIRED WITH B5's scale guard, exactly as the design words it. Extending it
    # alone would be the hazard m5 found; it is extended so that a cross-harness
    # read fails on the SCALE GUARD (the right reason) rather than on the tag
    # (the wrong reason), which is what makes B5's negative test meaningful.
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--n", type=int, default=N_EVAL)
    args = ap.parse_args()

    if args.k != KS.K_NCR:
        raise LoudFailure(f"--k {args.k} != NCR_K-resolved K_NCR={KS.K_NCR}: refusing to score, "
                          f"the numbers would carry the wrong K and chance baseline.")

    ladder = derive_depth_ladder(KS.K_NCR)
    lad_diag = verify_ladder(KS.K_NCR, ladder)
    h_deepest = ladder[-1]
    print(f"[{args.tag}] K={KS.K_NCR} ladder={list(ladder)} residues={lad_diag['residues']} "
          f"squarings={lad_diag['n_squarings']} (r_fix={lad_diag['r_fix']}, "
          f"effective distance {lad_diag['effective_distance']} at every rung)", flush=True)

    t0 = time.time()
    device = "cuda"
    path = os.path.expanduser(args.ckpt)
    if args.anchor_runner_tag:
        R.RUNNER_TAG = args.anchor_runner_tag
    ckpt = R.load_checkpoint(path, device)
    if ckpt is None:
        raise LoudFailure(f"CHECKPOINT FAIL [{args.tag}]: {path} missing or failed validation")

    ck_d = int(ckpt["full_graft"]["ncr_config"]["d"])
    ck_integ_d = int(ckpt["full_graft"]["integ_config"]["d_ncr"])
    if ck_d != KS.D_NCR or ck_integ_d != KS.D_NCR:
        raise LoudFailure(
            f"K MISMATCH [{args.tag}]: checkpoint records ncr d={ck_d} / integ d_ncr={ck_integ_d}, "
            f"but configured for K={KS.K_NCR} -> d={KS.D_NCR}. Trained at K={ck_d-1}. Refusing.")


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
        raise LoudFailure(f"SEED UNRESOLVABLE [{args.tag}] -- refusing to guess the pool seed")

    if "freeze_entity_adapter" in ckpt:
        freeze, fz_src = bool(ckpt["freeze_entity_adapter"]), "checkpoint"
    elif cfg_rec and cfg_rec.get("cfg_freeze_entity_adapter") is not None:
        freeze, fz_src = bool(cfg_rec["cfg_freeze_entity_adapter"]), "training-results-JSON"
    else:
        raise LoudFailure(
            f"ARM LABEL UNRESOLVABLE [{args.tag}]: freeze_entity_adapter absent from BOTH the "
            f"checkpoint and {args.cellcfg!r}. The frozen-vs-trainable ordering IS the question "
            f"-- refusing to default the arm label.")

    # MATCHED POOLS: the checkpoint's own recorded seed, always.
    pools, cfg, report = R.build_grammar_pools_and_cfg(seed=ckpt_seed)
    arms, _o, _g = R.restore_arms_and_opts(ckpt, report["vocab_size_total"], lr=LR,
                                           device=device, freeze_entity_adapter=freeze)
    arm = arms["full_graft"]
    pools = pools.to(device)

    p, sd, lo, hi = chance_band(KS.K_NCR, args.n)
    blocks = {}
    for regime, tf, hops in (("P1b", True, ladder), ("P0", False, (h_deepest,))):
        t = time.time()
        with torch.no_grad():
            res = R.eval_arm_at_hops(arm, pools, cfg, tuple(hops), args.n, device,
                                     args.base_seed, read_ablate=False, teacher_force=tf)
        per_hop = {}
        for h in hops:
            acc = res[f"h={h}"][METRIC]
            per_hop[f"h={h}"] = dict(h=h, residue=h % KS.K_NCR,
                                     n_squarings=KS.n_squarings(h), n_applies=KS.n_applies(h),
                                     acc=acc, margin_over_chance=acc - p,
                                     within_chance_3sd=bool(lo <= acc <= hi))
        blocks[regime] = dict(
            regime=regime, teacher_force=tf, hops=list(hops),
            regime_meaning=("EXACT-WRITE (operator teacher-forced from the true binding); "
                            "the READ is under test -- CAPABILITY curve"
                            if tf else
                            f"LEARNED-WRITE (model's own encoder writes) -- WALL curve, "
                            f"chance = 1/K = {p:.4f}"),
            note=None if tf else "deepest rung only (11 squarings), per the #2 cost cap",
            elapsed_s=time.time() - t, per_hop=per_hop, result=res)

    record = dict(
        script="depthext_eval.py",
        prereg="EXPERIMENT_LOG.md 2026-08-22 #2 final block (commit 9bf2f40)",
        wrapper_rationale=("kscaling_battery.py md5 5735c788563d9a21f2198c9f5b4793d5 has a fixed "
                           "hop grid and no custom-hop flag; this wrapper calls the SAME "
                           "eval_arm_at_hops path with the same guards. Nothing audited modified."),
        cell=f"DEPTHEXT_{args.tag}", tag=args.tag, ckpt=path,
        K=KS.K_NCR, d_ncr=KS.D_NCR, ckpt_recorded_d_ncr=ck_d,
        scale=KS.SCALE, backbone_config_scored=ckpt["full_graft"].get("backbone_config"),
        # AUDIT-R1 m2 / condition C11: was a bare literal 64. INVARIANT by
        # sec 3.3's code proof (ENC_H is backbone-independent), but a bare
        # size-bearing literal B1 now greps for -- read from the runner instead.
        kscaling=KS.provenance(R.H_NCR, R.RUNG1_BACKBONE["d_model"]),
        metric_of_record=METRIC,
        metric_note="historical name; argmax over exactly K slots, K-generic",
        chance=p, chance_sd_at_n=sd, wall_band=[lo, hi], wall_band_mult=WALL_SD_MULT,
        depth_ladder=list(ladder), squaring_profile=list(SQUARING_PROFILE),
        ladder_diagnostics=lad_diag,
        same_ground_truth_by_construction_per_K=True, same_ground_truth_label=SAME_GT,
        n=args.n, base_seed=args.base_seed,
        ckpt_step=step, ckpt_seed=ckpt_seed, ckpt_seed_source=seed_src,
        pool_seed=ckpt_seed, pools_matched=True,
        ckpt_cell_id=ckpt.get("cell_id"), runner_tag=ckpt.get("runner_tag"),
        anchor_runner_tag_override=args.anchor_runner_tag,
        freeze_entity_adapter=freeze, freeze_flag_source=fz_src,
        recipe=("FROZEN-contrastive (primary)" if freeze else "TRAINABLE-contrastive (compB)"),
        cell_config=cfg_rec, matched=blocks, elapsed_s=time.time() - t0)

    defects = []
    for regime, hops in (("P1b", ladder), ("P0", (h_deepest,))):
        blk = blocks.get(regime)
        if not blk:
            defects.append(f"{regime}: block missing")
            continue
        for h in hops:
            e = blk["per_hop"].get(f"h={h}")
            if e is None:
                defects.append(f"{regime}/h={h}: NO OUTPUT")
                continue
            v = e["acc"]
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                defects.append(f"{regime}/h={h}: acc={v!r} not finite")
            elif not (0.0 <= v <= 1.0):
                defects.append(f"{regime}/h={h}: acc={v} out of [0,1]")
            if blk["result"].get(f"h={h}", {}).get("n") != args.n:
                defects.append(f"{regime}/h={h}: n != {args.n}")
        if regime == "P1b":
            rs = [blk["per_hop"][f"h={h}"]["residue"] for h in hops]
            if len(set(rs)) != 1:
                defects.append(f"P1b: rungs span residues {rs}, not a single ground truth")
            sq = [blk["per_hop"][f"h={h}"]["n_squarings"] for h in hops]
            if tuple(sq) != SQUARING_PROFILE:
                defects.append(f"P1b: squaring profile {sq} != {list(SQUARING_PROFILE)}")
    record["self_check_defects"] = defects
    record["self_check"] = "PASS" if not defects else "FAIL"

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}_depthext.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, default=str)
    os.replace(tmp, out)

    if defects:
        print(f"!!! SELF-CHECK FAIL [{args.tag}]: {len(defects)} defect(s) -- NOT VALID", flush=True)
        for d in defects:
            print(f"    - {d}", flush=True)
        return 4

    ph = blocks["P1b"]["per_hop"]
    print(f"[{args.tag}] K={KS.K_NCR} freeze={freeze} seed={ckpt_seed} | P1b "
          + " ".join(f"{KS.n_squarings(h)}sq(h{h})={ph[f'h={h}']['acc']:.4f}" for h in ladder)
          + f" | P0@{SQUARING_PROFILE[-1]}sq={blocks['P0']['per_hop'][f'h={h_deepest}']['acc']:.4f} "
            f"(chance {p:.4f}, band [{lo:.4f},{hi:.4f}])", flush=True)
    print(f"DEPTHEXT_{args.tag}_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LoudFailure as e:
        print(f"!!! LOUD FAILURE: {e}", flush=True)
        sys.exit(5)
