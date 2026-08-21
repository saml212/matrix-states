#!/usr/bin/env python3
"""RESIDUE-SPACE COMPLETION (A) + PHYSICAL-DEPTH ROBUSTNESS (B).

Pre-registration: EXPERIMENT_LOG.md entry "2026-08-21 #2" (repo commit
ee602df), BINDING -- hop sets, arms, metric, n, seeds, and framing are
fixed there and are reproduced verbatim below, never re-derived by taste.

Eval-only. NOTHING in this file modifies the audited instrument: it
imports ncr_lm_wave1_runner as a library and reuses, unmodified,
build_grammar_pools_and_cfg / load_checkpoint / restore_arms_and_opts /
eval_arm_at_hops -- the identical call pattern as the audited
`pbe_repl` premise-battery replication script (2026-08-13), which is
the instrument of record for retrieval24_acc on these checkpoints.
The ONLY differences from pbe_repl are (i) the hop sets and (ii) the
per-checkpoint freeze flag being READ FROM THE CHECKPOINT rather than
passed on argv.

Experiment A -- RESIDUE-SPACE COMPLETION.
  h in {4,6,7,8,9,10,11,14,15,17,18,19,21,22,23}: the 15 residues mod
  K=24 that are neither the identity (0), nor a train residue (1,2,3),
  nor one of the residues the DEEP_LADDER already measured
  ({5,12,20,16,13}). Claim under test: the exact-write capability
  covers the ENTIRE reachable outcome space. This is COVERAGE, not
  depth scaling -- the task's ground truth is a single Hamiltonian
  24-cycle, so h and h mod 24 are the same question.

Experiment B -- PHYSICAL-DEPTH ROBUSTNESS AT FIXED RESIDUE.
  h in {13, 61, 253, 1021, 4093}: ALL of these are == 13 (mod 24) and
  therefore have the *SAME GROUND TRUTH BY CONSTRUCTION*. Any variation
  across them is numerical (binexp_read performs 3..12 squarings), NOT
  new compositional information. Every output record repeats this
  label.

REGIME NAMING IS MANDATORY on every number (the same checkpoint reads
~0.99 under P1b and ~0.05 under P0):
  P1b = teacher_force=True  -- EXACT-WRITE regime (the operator is
        constructed from the true key/value binding; the read machinery
        is what is under test). The pre-registered regime for A and B.
  P0  = teacher_force=False -- the model's own learned encoder writes
        the operator. Recorded as INFORMATIONAL ONLY.

Usage:
  residue_depth_eval.py --ckpt PATH --tag TAG [--probe]
    --probe : load, print checkpoint metadata, time ONE h=4093 P1b eval
              cell, write nothing, exit 0. (Pre-registration guard: do
              not commit the full grid before the deepest cell's
              wall-clock is known to be sane.)
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncr_lm_wave1_runner as R  # noqa: E402  (the AUDITED instrument, imported as a library)

# --- frozen constants, from the pre-registration -------------------------
BASE_SEED = 90210          # eval seed, verbatim from pbe_repl / the premise battery
N_EVAL = 256               # n=256, verbatim
POOL_SEED = 0              # build_grammar_pools_and_cfg(seed=0), verbatim from pbe_repl
LR = 3e-4                  # only needed so restore_arms_and_opts can rebuild the optimizer
K_CYCLE = 24               # K_NCR: the single Hamiltonian cycle length (grammar_rd.py:262-286)
REQUIRED_CKPT_STEP = 20000
METRIC_OF_RECORD = "retrieval24_acc"

HOPS_A = (4, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19, 21, 22, 23)
HOPS_B = (13, 61, 253, 1021, 4093)
B_RESIDUE = 13

A_FRAMING = ("COVERAGE of the reachable outcome space (15 previously-unmeasured "
             "residues mod K=24), NOT depth scaling. Regime of record: P1b "
             "(teacher_force=True, exact-write).")
B_FRAMING = ("NUMERICAL ROBUSTNESS of the O(log h) binexp read to physical "
             "squaring depth. EVERY h in this set is == 13 (mod 24) and so has "
             "the IDENTICAL ground truth by construction -- these five readings "
             "carry NO new compositional information. Regime of record: P1b "
             "(teacher_force=True, exact-write).")


class LoudFailure(RuntimeError):
    pass


def preflight() -> dict:
    """Every structural precondition the pre-registration depends on, checked
    with EXACT integer arithmetic (CLAUDE.md: no float slack in a structural
    check). Raises LoudFailure -- never warns and continues."""
    train_residues = {h % K_CYCLE for h in R.TRAIN_HOPS}
    measured_residues = {h % K_CYCLE for h in R.DEEP_LADDER}
    already_used = {0} | train_residues | measured_residues
    expected_A = tuple(r for r in range(K_CYCLE) if r not in already_used)

    if tuple(sorted(HOPS_A)) != expected_A:
        raise LoudFailure(
            f"PREFLIGHT FAIL (A): the pre-registered hop set {sorted(HOPS_A)} is not the set of "
            f"unused non-identity residues {list(expected_A)} derived from TRAIN_HOPS="
            f"{list(R.TRAIN_HOPS)} and DEEP_LADDER={list(R.DEEP_LADDER)} at K={K_CYCLE}. "
            "The instrument's own ladder constants must have changed -- STOP and re-adjudicate.")
    if len(HOPS_A) != 15:
        raise LoudFailure(f"PREFLIGHT FAIL (A): expected 15 residues, got {len(HOPS_A)}")
    if any(h <= 3 or h >= K_CYCLE for h in HOPS_A):
        raise LoudFailure(f"PREFLIGHT FAIL (A): every h must satisfy 3 < h < {K_CYCLE}: {HOPS_A}")

    residues_B = sorted({h % K_CYCLE for h in HOPS_B})
    if residues_B != [B_RESIDUE]:
        raise LoudFailure(
            f"PREFLIGHT FAIL (B): hop set {list(HOPS_B)} does NOT sit at a single residue -- "
            f"residues={residues_B}, expected [{B_RESIDUE}]. The 'same ground truth by "
            "construction' label would be FALSE -- STOP.")
    if len(set(HOPS_B)) != len(HOPS_B):
        raise LoudFailure(f"PREFLIGHT FAIL (B): duplicate physical depths in {list(HOPS_B)}")

    # The instrument's own ladder-soundness rule, applied EXPLICITLY to both of
    # our hop sets. It is a module-level check on DEEP_LADDER only (line 320) and
    # is NOT invoked by eval_arm_at_hops, so this is an ADDITIONAL guard we
    # volunteer -- not a bypass. Both sets pass it: no residue is 0 (identity)
    # and none collides with a train residue.
    R._assert_ladder_sound(tuple(HOPS_A), K_CYCLE, R.TRAIN_HOPS)
    R._assert_ladder_sound(tuple(HOPS_B), K_CYCLE, R.TRAIN_HOPS)

    # binexp_read's own precondition (ncr_models.py:107): h must be a python int >= 1.
    for h in HOPS_A + HOPS_B:
        if not (isinstance(h, int) and h >= 1):
            raise LoudFailure(f"PREFLIGHT FAIL: binexp_read requires int h>=1, got {h!r}")

    return dict(train_residues=sorted(train_residues),
                already_measured_residues=sorted(measured_residues),
                derived_unused_residues=list(expected_A),
                n_squarings_B={f"h={h}": max(0, h.bit_length() - 1) for h in HOPS_B})


def open_checkpoint(ckpt_path: str, device: str) -> dict:
    path = os.path.expanduser(ckpt_path)
    ckpt = R.load_checkpoint(path, device)
    if ckpt is None:
        raise LoudFailure(f"CHECKPOINT FAIL: {path} missing or failed load_checkpoint validation")
    return ckpt


def checkpoint_metadata(ckpt: dict, ckpt_path: str) -> dict:
    if "freeze_entity_adapter" not in ckpt:
        raise LoudFailure(
            f"CHECKPOINT FAIL: {ckpt_path} has no 'freeze_entity_adapter' key. The freeze flag "
            "MUST be read from the checkpoint itself (sec G3-B31: a mismatch rebuilds the "
            "optimizer with the wrong param-group shape) -- refusing to guess.")
    return dict(ckpt=ckpt_path,
                ckpt_step=int(ckpt["step"]),
                ckpt_seed=ckpt.get("seed", None),
                ckpt_cell_id=ckpt.get("cell_id", None),
                freeze_entity_adapter=bool(ckpt["freeze_entity_adapter"]),
                runner_tag=ckpt.get("runner_tag", None))


def eval_both_regimes(arm, pools, cfg, hops, device) -> dict:
    """P1b (exact-write, teacher_force=True) and P0 (learned write,
    teacher_force=False), timed per regime. Same call signature as pbe_repl."""
    out = {}
    for regime, tf in (("P1b", True), ("P0", False)):
        t = time.time()
        with torch.no_grad():
            res = R.eval_arm_at_hops(arm, pools, cfg, tuple(hops), N_EVAL, device,
                                     BASE_SEED, read_ablate=False, teacher_force=tf)
        out[regime] = dict(regime=regime, teacher_force=tf,
                           regime_meaning=("EXACT-WRITE (operator teacher-forced from the true "
                                           "binding); the pre-registered regime of record"
                                           if tf else
                                           "LEARNED-WRITE (the model's own encoder writes the "
                                           "operator); INFORMATIONAL ONLY"),
                           elapsed_s=time.time() - t, result=res)
    return out


def self_check(record: dict) -> list:
    """FAIL LOUDLY on the silent-zero-scoring bug class (recurred 6 times):
    every expected experiment x regime x hop MUST have produced a finite,
    in-range metric on a batch of exactly N_EVAL. Returns a list of defect
    strings; empty list == clean."""
    defects = []
    for exp_key, hops in (("experiment_A", HOPS_A), ("experiment_B", HOPS_B)):
        if exp_key not in record:
            defects.append(f"{exp_key}: ENTIRE EXPERIMENT MISSING from the record")
            continue
        for regime in ("P1b", "P0"):
            block = record[exp_key].get(regime)
            if not block or "result" not in block:
                defects.append(f"{exp_key}/{regime}: missing result block")
                continue
            res = block["result"]
            for h in hops:
                key = f"h={h}"
                if key not in res:
                    defects.append(f"{exp_key}/{regime}/{key}: NO OUTPUT (hop never scored)")
                    continue
                cell = res[key]
                if METRIC_OF_RECORD not in cell:
                    defects.append(f"{exp_key}/{regime}/{key}: metric {METRIC_OF_RECORD} absent")
                    continue
                v = cell[METRIC_OF_RECORD]
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    defects.append(f"{exp_key}/{regime}/{key}: {METRIC_OF_RECORD}={v!r} not finite")
                elif not (0.0 <= v <= 1.0):
                    defects.append(f"{exp_key}/{regime}/{key}: {METRIC_OF_RECORD}={v} out of [0,1]")
                if cell.get("n") != N_EVAL:
                    defects.append(f"{exp_key}/{regime}/{key}: n={cell.get('n')!r}, expected {N_EVAL}")
    return defects


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probe", action="store_true",
                    help="metadata + a single timed h=4093 P1b cell; writes nothing")
    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_writecond/results"))
    ap.add_argument("--pool-seed", type=int, default=POOL_SEED,
                    help="entity-pool seed. Default 0 == the audited pbe_repl instrument's own "
                         "hardcoded value (the regime every existing premise-battery number of "
                         "record was produced under). Pass the checkpoint's own seed for the "
                         "matched-seed sensitivity reading, which is written to a separate file.")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda"
    diag = preflight()
    print(f"[{args.tag}] PREFLIGHT PASS: A={list(HOPS_A)} (derived unused residues match), "
          f"B={list(HOPS_B)} all == {B_RESIDUE} (mod {K_CYCLE}), squarings={diag['n_squarings_B']}",
          flush=True)

    pools, cfg, pool_report = R.build_grammar_pools_and_cfg(seed=args.pool_seed)
    pools = pools.to(device)

    ckpt = open_checkpoint(args.ckpt, device)
    meta = checkpoint_metadata(ckpt, args.ckpt)
    print(f"[{args.tag}] checkpoint metadata: {meta}", flush=True)

    if meta["ckpt_step"] != REQUIRED_CKPT_STEP:
        print(f"!!! SKIP+FLAG [{args.tag}]: ckpt_step={meta['ckpt_step']} != "
              f"{REQUIRED_CKPT_STEP}. The pre-registration admits ckpt_step==20000 ONLY. "
              "NOT SCORED.", flush=True)
        return 3

    if meta["ckpt_seed"] is not None and meta["ckpt_seed"] != args.pool_seed:
        # DISCLOSED, NOT SILENT, and NOT "fixed" here. The audited instrument
        # (pbe_repl, and therefore every existing premise-battery number of
        # record on these checkpoints) hardcodes
        # build_grammar_pools_and_cfg(seed=0), while cell mob_g3b31_*_sN was
        # trained with --seed N. build_entity_pools(..., heldout_frac=0.5,
        # seed=seed) uses that seed for the entity train/heldout SPLIT, so the
        # two pools are not the same set of entity tokens. Reproducing the
        # instrument's own choice keeps these new residues directly comparable
        # to the already-recorded h in {1,13,37,61} readings; changing it
        # unilaterally would make them incomparable. Run with --pool-seed N for
        # the matched-seed sensitivity reading (recorded separately).
        print(f"### ADVISORY [{args.tag}]: pool_seed={args.pool_seed} but ckpt_seed="
              f"{meta['ckpt_seed']}. This REPRODUCES the audited pbe_repl instrument exactly "
              "(it hardcodes seed=0); flagged for coordinator adjudication, not silently "
              "altered.", flush=True)

    arms, _opts, _gen = R.restore_arms_and_opts(
        ckpt, pool_report["vocab_size_total"], lr=LR, device=device,
        freeze_entity_adapter=meta["freeze_entity_adapter"])
    arm = arms["full_graft"]

    if args.probe:
        t = time.time()
        with torch.no_grad():
            res = R.eval_arm_at_hops(arm, pools, cfg, (max(HOPS_B),), N_EVAL, device,
                                     BASE_SEED, read_ablate=False, teacher_force=True)
        dt = time.time() - t
        print(f"[{args.tag}] PROBE h={max(HOPS_B)} P1b: {METRIC_OF_RECORD}="
              f"{res[f'h={max(HOPS_B)}'][METRIC_OF_RECORD]:.6f} in {dt:.2f}s "
              f"(total incl. load {time.time()-t0:.1f}s)", flush=True)
        print("PROBE_DONE", flush=True)
        return 0

    record = dict(
        script="residue_depth_eval.py",
        cell=f"RESIDUE_DEPTH_{args.tag}",
        tag=args.tag,
        prereg="EXPERIMENT_LOG.md 2026-08-21 #2 (repo commit ee602df)",
        instrument="ncr_lm_wave1_runner.eval_arm_at_hops via the pbe_repl call pattern (audited)",
        metric_of_record=METRIC_OF_RECORD,
        n=N_EVAL, base_seed=BASE_SEED, pool_seed=args.pool_seed, K_cycle=K_CYCLE,
        pool_seed_is_instrument_default=(args.pool_seed == POOL_SEED),
        residue_diagnostics=diag,
        **meta,
    )
    record["pool_seed_matches_ckpt_seed"] = (meta["ckpt_seed"] == args.pool_seed)

    per_hop_residue = {f"h={h}": h % K_CYCLE for h in HOPS_A + HOPS_B}

    record["experiment_A"] = dict(name="RESIDUE-SPACE COMPLETION", hop_set=list(HOPS_A),
                                  framing=A_FRAMING,
                                  hop_residues={f"h={h}": h % K_CYCLE for h in HOPS_A})
    record["experiment_A"].update(eval_both_regimes(arm, pools, cfg, HOPS_A, device))

    record["experiment_B"] = dict(name="PHYSICAL-DEPTH ROBUSTNESS AT FIXED RESIDUE",
                                  hop_set=list(HOPS_B), residue_mod_K=B_RESIDUE,
                                  framing=B_FRAMING,
                                  same_ground_truth_by_construction=True,
                                  n_squarings=diag["n_squarings_B"],
                                  hop_residues={f"h={h}": h % K_CYCLE for h in HOPS_B})
    record["experiment_B"].update(eval_both_regimes(arm, pools, cfg, HOPS_B, device))

    record["per_hop_residue"] = per_hop_residue
    record["elapsed_s"] = time.time() - t0

    defects = self_check(record)
    record["self_check_defects"] = defects
    record["self_check"] = "PASS" if not defects else "FAIL"

    os.makedirs(args.outdir, exist_ok=True)
    suffix = "" if args.pool_seed == POOL_SEED else f"_poolseed{args.pool_seed}"
    out = os.path.join(args.outdir, f"residue_depth_{args.tag}{suffix}.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp, out)

    if defects:
        print(f"!!! SELF-CHECK FAIL [{args.tag}] -- {len(defects)} defect(s); "
              f"record written to {out} for forensics but the cell is NOT VALID:", flush=True)
        for d in defects:
            print(f"    - {d}", flush=True)
        return 4

    a1b = record["experiment_A"]["P1b"]["result"]
    b1b = record["experiment_B"]["P1b"]["result"]
    print(f"[{args.tag}] A P1b per-residue {METRIC_OF_RECORD}: "
          + ", ".join(f"h={h}:{a1b[f'h={h}'][METRIC_OF_RECORD]:.4f}" for h in HOPS_A), flush=True)
    print(f"[{args.tag}] B P1b (ALL h == {B_RESIDUE} mod {K_CYCLE}, same ground truth): "
          + ", ".join(f"h={h}:{b1b[f'h={h}'][METRIC_OF_RECORD]:.4f}" for h in HOPS_B), flush=True)
    a0_mean = record["experiment_A"]["P0"]["result"]["mean_" + METRIC_OF_RECORD]
    b0_mean = record["experiment_B"]["P0"]["result"]["mean_" + METRIC_OF_RECORD]
    print(f"[{args.tag}] P0 means (INFORMATIONAL ONLY, learned-write regime): "
          f"A={a0_mean:.4f} B={b0_mean:.4f}", flush=True)
    print(f"[{args.tag}] SELF-CHECK PASS -> {out} ({record['elapsed_s']:.1f}s)", flush=True)
    print(f"RESIDUE_DEPTH_{args.tag}_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LoudFailure as e:
        print(f"!!! LOUD FAILURE: {e}", flush=True)
        sys.exit(5)
