#!/usr/bin/env python3
"""V2' -- THE CONSTANT-LR ATTRIBUTION CONTROL AT K=40 TRAINABLE.
EXPERIMENT_LOG 2026-08-23 #3's V2 adjudication. Two specs, 0250-0251.

WHY V2' EXISTS. The attribution arm's resume RE-OPENED the cosine (get_lr is
warmup + cosine to 0.1*max_lr over `total_steps`, so resuming a 20,000-step
parent with --steps 40000 recomputes the schedule over 40,000). That was
disclosed BEFORE the arm ran -- 3.00e-05 -> 1.66e-04, a 5.535x warm restart --
and it then ACTIVELY DAMAGED the K=40 trainable cells: kappa 0.8438 -> 0.5513,
the largest movement anywhere in the arm. sec 7.2's licensing logic ("under-
training can only manufacture DEGRADES") assumes a NON-HARMFUL extension, so a
damaged cell cannot strengthen the verdict. #3 therefore adjudicated V2
SCALE-DEGRADES STANDS but UNSTRENGTHENED, and dispatched this control.

WHAT V2' CHANGES: the schedule variable, and nothing else. Same parents, same
seeds, same recipe, same pools, same ladder, same token multiple. The LR is held
at the parent's OWN FINAL VALUE (3.0e-05, derived by evaluating the parent's
completed schedule at its own last step) for the entire marginal segment.

    step        BEFORE (re-opened cosine)    AFTER (V2' constant)
    20001              1.660549e-04                3.000000e-05
    20025              1.657992e-04                3.000000e-05
    20050              1.655328e-04                3.000000e-05
    30000              6.991808e-05                3.000000e-05
    40000              3.000000e-05                3.000000e-05

PARENTS ARE THE ORIGINAL 20k CELLS, NOT THE DAMAGED 40k ONES. Verified on the
box before generation: scaleaxis392m_K40_compB_s{0,1} are at step 20000 with
freeze=False and their own seeds; the 40k attribution cells sit in a separate
tree and are NOT touched.

PRE-REGISTRATION, pinned in #3 and restated in every spec:
  * TOKEN-BUDGET-LIMITED  iff kappa >= 0.90 at h_top on 2/2 seeds.
  * UNRECOVERED-AND-UNDAMAGED (kappa back near the parent's 0.84 band)
        => SCALE-DEGRADES STANDS, CLEANLY STRENGTHENED.
  * COLLAPSES AGAIN => the damage is NOT the schedule; a new instability
        finding, and the schedule is exonerated.

TWO CHECKER LESSONS FROM THE ARM, BOTH APPLIED HERE:
  #1 the field-path bug -- `steps_target` is recorded at the TOP LEVEL of the
     runner's record, NOT under `config`. The arm's checker read
     config.steps_target -> None -> "not the extended budget", flagging four
     flawless cells as failed. Fixed here, and the smoke exercises the clause.
  #2 the Gate-0-marginal clause is MIS-SCOPED for resumed segments: at plateau
     the marginal CE is flat (+0.026/+0.030 over 20,000 steps against +-0.03
     plateau noise) and `h[-1] < h[0]` fails on cells that are perfectly fine.
     Flat CE is attribution EVIDENCE, not a defect. Scoped here to
     PLATEAU-TOLERANT, FINITE-CE ONLY.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NCR_SCALE", "392m")
os.environ.setdefault("NCR_K", "40")
_TREE = os.environ.get("SCALEAXIS_TREE", _HERE)
sys.path.insert(0, _TREE)
import kscaling_config as KS   # noqa: E402
assert KS.SCALE == "392m" and KS.RUNG == 2, (KS.SCALE, KS.RUNG)

OUT = os.path.join(_HERE, "job_specs_v2prime")
SWEEP_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                             "experiment-runs", "2026-08-22_scaleaxis_sweep")
PY = "/home/nvidia/tdenv/bin/python3"
WORKDIR = "/home/nvidia/ncr_scaleaxis"
EPH = "/ephemeral/scaleaxis"
ROOT = f"{EPH}/v2prime"
SCALE, K, RECIPE = "392m", 40, "compB"
BASE_STEPS, EXT_STEPS = 20_000, 40_000
SEEDS = (0, 1)
R8 = 1.0026                 # MEASURED at Stage A0.4 (EXPERIMENT_LOG #18)
CEILING_MULT = 1.5          # design sec 3.6 breaker 1
PARENT_FINAL_LR = 3.0e-5    # get_lr(20000, 3e-4, 200, 20000) = 0.1 * max_lr
PARENT_KAPPA = 0.8438       # #21/#3: the K=40 trainable parent median at h_top
DAMAGED_KAPPA = 0.5513      # what the warm-restart extension produced

FLAGS = ("--aux-loss-type contrastive+cosine --contrastive-temperature 0.07")
COMMON = ("--mode calibration --device cuda --batch-size 32 --eval-batch-size 64 "
          "--warmup-steps 200 --lr 3e-4 --aux-read-loss-weight 0.5 --ortho-reg-weight 0.1 "
          "--eval-every 1000 --ckpt-every 5000")


def realized(sweep_dir: str | None = None) -> float:
    """MEASURED 20,000-step cost of the K=40 trainable cells, from the sweep of
    record. Priced from measurement, never a projection (design sec 10)."""
    d0 = sweep_dir or SWEEP_DEFAULT
    if not os.path.isdir(d0):
        raise SystemExit(f"--sweep-dir {d0!r} does not exist; V2' is priced from the MEASURED "
                         f"20,000-step cost of its own parents.")
    vals = []
    for p in sorted(glob.glob(os.path.join(d0, "*.json"))):
        d = json.load(open(p))
        if d.get("mode") != "calibration" or d.get("status") != "COMPLETED":
            continue
        ks = d.get("kscaling") or {}
        if int(ks.get("K", 0)) != K or d["config"]["freeze_entity_adapter"]:
            continue
        vals.append(d.get("gpu_h") or (d.get("elapsed_s") or 0) / 3600.0)
    assert len(vals) == 3, f"expected the 3 K={K} trainable sweep cells in {d0}, got {len(vals)}"
    return sum(vals) / len(vals)


HYP = (
    "V2' -- THE CONSTANT-LR ATTRIBUTION CONTROL. EXPERIMENT_LOG 2026-08-23 #3's V2 adjudication. "
    "Cell: K={K} (d={d}), TRAINABLE-contrastive (compB), seed {seed}, RESUMED from its ORIGINAL "
    "20,000-step parent {parent} and extended to {ext} steps at a CONSTANT learning rate. "
    "WHAT THIS CONTROLS. The attribution arm's resume re-opened the cosine (3.00e-05 -> "
    "1.66e-04, a 5.535x warm restart, disclosed before that arm ran) and ACTIVELY DAMAGED these "
    "cells: kappa {damaged:.4f} against a parent {parent_kappa:.4f}, the largest movement in the "
    "arm. Design sec 7.2's licensing logic -- \\\"under-training can only manufacture DEGRADES\\\" "
    "-- assumes a NON-HARMFUL extension, so a damaged cell cannot strengthen a verdict; #3 "
    "adjudicated V2 SCALE-DEGRADES STANDS but UNSTRENGTHENED and dispatched this. V2' removes "
    "THE SCHEDULE VARIABLE AND NOTHING ELSE: same parent, same seed, same recipe, same pools, "
    "same ladder, same token multiple (2.0x), with the LR held at the parent's OWN FINAL VALUE "
    "({const_lr:.3e}) for the entire marginal segment. That constant is DERIVED, not hand-set: "
    "resume_const_lr(start_step) = get_lr(start_step, lr, warmup, start_step), i.e. the parent's "
    "completed schedule evaluated at its own last step. The extended cell therefore differs from "
    "its parent in TOKENS ONLY, which is what makes it a token-budget control. "
    "PRE-REGISTERED DECISION RULE, pinned in #3 BEFORE this runs: "
    "(a) kappa >= 0.90 at h_top={htop} on 2/2 seeds => TOKEN-BUDGET-LIMITED, and the Curve-1 "
    "SCALE-DEGRADES at K={K} trainable is withdrawn as a scale claim; "
    "(b) UNRECOVERED AND UNDAMAGED -- kappa back in the parent's {parent_kappa:.2f} band, i.e. "
    "the doubled budget neither rescues nor harms => SCALE-DEGRADES STANDS, CLEANLY STRENGTHENED, "
    "which is the outcome the compromised V2 could not deliver; "
    "(c) COLLAPSES AGAIN toward {damaged:.2f} => the damage is NOT the schedule, the warm restart "
    "is EXONERATED, and this becomes a new instability finding at K={K} trainable. "
    "READOUT: P1b kappa at h_top={htop} (residue {htopres} = K/2, antipodal), matched pools, "
    "n=256, base seed 90210, ckpt_step == {ext}; the s*=13 depth rung is reported alongside for "
    "continuity with V3 but carries no V2' verdict. 98M reference kappa@h_top for this cell: "
    "0.9880. THE CONFOUND BEING CONTROLLED is unchanged (design sec 1/sec 7.1): steps, batch and "
    "t_in were held fixed across scales, so this cell saw the SAME tokens as its 98M twin while "
    "carrying 4.008x the parameters -- D/N = {dn:.3f} tokens/param at 392M against 1.87 at 98M."
)

NOTES = (
    "CANDIDATE -- NOT queue-eligible; the coordinator stages on the build report. THE LAST "
    "TRAINING PAYLOAD OF THE GRANT. The 130-GPU-h gate FIRED on this third contingency and is "
    "LICENSED BY RECORDED COORDINATOR ADJUDICATION (#3): grounds on record -- ~900 GPU-h "
    "unallocated in a closing window, the verdict materially decides the flagship's central "
    "scale claim, and the marginal cost is ~7% of remaining budget. "
    "RUNNER PATCH: this wave is the first user of --const-lr-on-resume, a MINIMAL, DISCLOSED, "
    "OFF-BY-DEFAULT patch to the scaleaxis runner COPY (patch_scaleaxis.py entries R6-R12; the "
    "PINNED originals in ~/ncr_g3b31_contrastive are untouched and re-verified, runner "
    "9a93198b642242f512ff8489e32b0a53 / graft bc105af69661e488ff95f5046e2bcd8a). ONE behavioural "
    "line changes: `cur_lr = (const_lr if const_lr is not None else get_lr(step, ...))`. With the "
    "flag absent the expression is byte-equivalent to the pinned call, so every existing cell is "
    "bit-unaffected. The resolved constant is printed at startup and RECORDED at the TOP LEVEL of "
    "the results record (const_lr_on_resume / resume_const_lr / resume_start_step), so this "
    "spec's validity_check asserts the mechanism actually fired rather than trusting the flag. "
    "PARENTS ARE THE ORIGINAL 20k CELLS, NOT THE DAMAGED 40k ONES -- verified on the box before "
    "generation (step=20000, freeze=False, seed matched); the 40k attribution cells live in a "
    "separate tree and are not read. The parent ckpt is HARDLINKED under a fresh cell id, so "
    "atomic_torch_save's os.replace leaves the 20,000-step CHECKPOINT OF RECORD intact, and "
    "--out points at a NEW json so the record of record is untouched and the runner's "
    "\\\"already COMPLETED -- skipping\\\" guard (which keys on the OUT path) cannot fire. "
    "CHECKER LESSONS APPLIED: (#1) `steps_target` is asserted at the TOP LEVEL, which is where "
    "the runner records it -- the arm's checker read config.steps_target, got None, and flagged "
    "four flawless cells as failed; the smoke here exercises EVERY clause by making each one "
    "first-to-fail in turn, which is what assert-order masking defeated last time. (#2) The "
    "Gate-0-marginal clause is SCOPED to PLATEAU-TOLERANT, FINITE-CE ONLY: a resumed segment at "
    "plateau moves CE by ~+0.03 against +-0.03 noise, and that flat CE is attribution EVIDENCE, "
    "not a defect -- the recovery verdict rests on the pre-registered kappa bar from the battery, "
    "never on CE. CEILING: {ceilprov} "
    "Scored afterwards by kscaling_battery.py from the scaleaxis tree (B5 scale guard active). "
    "One cell per GPU per the standing declined-packing ruling; /ephemeral only, never root fs."
)


def validity(out_json: str, params: int) -> str:
    """Every clause is INDEPENDENTLY first-to-fail-able, and the smoke proves it
    (EXPERIMENT_LOG 2026-08-23 #1: assert-order masking hid the broken clause)."""
    lad = list(KS.LADDER_TABLE[K])
    return (
        f"{PY} -c \"import json, math; d=json.load(open('{out_json}')); "
        f"assert d.get('status')=='COMPLETED', ('status', d.get('status')); "
        f"assert d.get('step',0)>={EXT_STEPS}, ('step', d.get('step')); "
        # #1's FIX: steps_target is TOP-LEVEL, not under config.
        f"assert d.get('steps_target')=={EXT_STEPS}, ('steps_target', d.get('steps_target')); "
        f"assert d.get('runner_tag')=='ncr_scaleaxis_runner_v1', ('runner_tag', d.get('runner_tag')); "
        # V2's whole point: prove the constant-LR mechanism FIRED.
        f"assert d.get('const_lr_on_resume') is True, ('const_lr_on_resume', d.get('const_lr_on_resume')); "
        f"assert d.get('resume_start_step')=={BASE_STEPS}, ('resume_start_step', d.get('resume_start_step')); "
        f"rc=d.get('resume_const_lr'); "
        f"assert rc is not None and abs(rc-{PARENT_FINAL_LR!r})<1e-12, ('resume_const_lr', rc); "
        f"ks=d.get('kscaling') or {{}}; "
        f"assert ks.get('K')=={K}, ('K', ks.get('K')); "
        f"assert ks.get('d_ncr')=={K+1}, ('d_ncr', ks.get('d_ncr')); "
        f"assert ks.get('h_top')=={lad[-1]}, ('h_top', ks.get('h_top')); "
        f"assert ks.get('deep_ladder')=={lad}, ('deep_ladder', ks.get('deep_ladder')); "
        f"assert ks.get('scale')=='{SCALE}', ('scale', ks.get('scale')); "
        f"bb=ks.get('backbone') or {{}}; "
        f"assert (bb.get('d_model'), bb.get('d_state'), bb.get('n_layers'))==(1536,128,16), ('backbone', bb); "
        f"cfg=d.get('config') or {{}}; "
        f"assert cfg.get('freeze_entity_adapter') is False, ('RECIPE MISMATCH -- V2 prime is the TRAINABLE arm', cfg.get('freeze_entity_adapter')); "
        f"pp=d.get('params') or {{}}; "
        f"assert pp.get('per_arm')=={params}, ('params.per_arm vs design sec 3.4', pp.get('per_arm')); "
        f"lh=d.get('loss_history') or {{}}; "
        f"assert set(lh)=={{'full_graft','backbone_only'}}, ('loss_history arms', sorted(lh)); "
        f"assert len(lh['backbone_only'])>=100, ('backbone_only len', len(lh['backbone_only'])); "
        f"h=lh['full_graft']; assert len(h)>=100, ('full_graft len', len(h)); "
        # #2's SCOPING: finite-CE ONLY. A resumed segment at plateau moves CE by
        # ~+0.03 against +-0.03 noise; that flatness is EVIDENCE, not a defect.
        f"assert all(math.isfinite(r[1]) for r in h), 'non-finite CE in loss_history (Gate-0-marginal, plateau-tolerant: finiteness only)'\"")


def spec(job_id: str, seed: int, marginal: float, params: int) -> dict:
    parent = f"scaleaxis392m_K{K}_{RECIPE}_s{seed}"
    cell = f"v2prime_K{K}_{RECIPE}_s{seed}"
    outdir, ckdir = f"{ROOT}/results", f"{ROOT}/ckpts/{cell}"
    out_json = f"{outdir}/{cell}.json"
    lad = KS.LADDER_TABLE[K]
    full40k = 2.0 * marginal
    ceil_v = round(CEILING_MULT * R8 * full40k, 3)
    ceilprov = (
        f"{CEILING_MULT} x R_8 x the FULL {EXT_STEPS}-step projection at this cell's OWN MEASURED "
        f"rate ({marginal:.4f} GPU-h per 20,000 steps => {full40k:.4f} at 40,000; R_8 = {R8} "
        f"MEASURED at Stage A0.4) = {ceil_v} GPU-h. The ceiling MUST cover the FULL CUMULATIVE "
        f"cost: run_two_arm_cell restores cumulative_elapsed_s from the checkpoint and sets "
        f"t0 = time.time() - cumulative_elapsed_s, so `elapsed > ceiling_s` is checked against "
        f"the WHOLE 40,000-step wall clock (~4.6 GPU-h is already spent before the first marginal "
        f"step). A ceiling sized on the marginal half fires ABORTED-BUDGET immediately.")
    dn = (BASE_STEPS * 32 * KS.t_in(K)) / params
    cmd = (f"mkdir -p {outdir} {ckdir} && "
           f"[ -f {ckdir}/{cell}.ckpt.pt ] || ln {EPH}/ckpts/{parent}/{parent}.ckpt.pt "
           f"{ckdir}/{cell}.ckpt.pt && cd {WORKDIR} && "
           f"NCR_SCALE={SCALE} NCR_K={K} {PY} ncr_lm_wave1_runner.py "
           f"--k {K} --scale {SCALE} {COMMON} --steps {EXT_STEPS} --const-lr-on-resume "
           f"--ceiling-gpuh {ceil_v} {FLAGS} "
           f"--seed {seed} --cell-id {cell} --out {out_json} "
           f"--ckpt-dir {ckdir} 2>&1 | tee -a {outdir}/{cell}.log")
    return {
        "id": job_id, "lane": "NCR", "tier": "scaleaxis_v2prime",
        "hypothesis": HYP.format(K=K, d=K + 1, seed=seed, parent=parent, ext=EXT_STEPS,
                                 damaged=DAMAGED_KAPPA, parent_kappa=PARENT_KAPPA,
                                 const_lr=PARENT_FINAL_LR, htop=lad[-1], htopres=lad[-1] % K,
                                 dn=dn),
        "cmd": cmd,
        "gpu_h_estimate": round(marginal, 3),
        "output_dir": outdir,
        "validity_check": validity(out_json, params),
        "notes": NOTES.format(ceilprov=ceilprov),
        "v2prime": {
            "supersedes": "V2 (K=40 trainable) -- SCALE-DEGRADES STANDS but UNSTRENGTHENED",
            "adjudication": "EXPERIMENT_LOG 2026-08-23 #3",
            "gate_130_gpuh": "FIRED on this third contingency; LICENSED by recorded "
                             "coordinator adjudication in #3",
            "variable_removed": "the LR schedule (5.535x warm restart at the resume point)",
            "parent_cell": parent,
            "parent_ckpt": f"{EPH}/ckpts/{parent}/{parent}.ckpt.pt",
            "parent_is_original_20k_not_damaged_40k": True,
            "parent_kappa_h_top": PARENT_KAPPA,
            "warm_restart_kappa_h_top": DAMAGED_KAPPA,
            "resume_const_lr_expected": PARENT_FINAL_LR,
            "lr_before_at_20001": 1.660549e-04,
            "lr_after_at_20001": PARENT_FINAL_LR,
            "warm_restart_factor_removed": 5.535,
            "base_steps": BASE_STEPS, "extended_steps": EXT_STEPS, "token_multiple": 2.0,
            "recipe": RECIPE, "frozen": False, "seed": seed,
            "marginal_gpu_h_measured": round(marginal, 4),
            "full_40k_gpu_h_measured": round(full40k, 4),
            "ceiling_gpuh": ceil_v, "ceiling_provenance": ceilprov,
            "prereg": {
                "TOKEN-BUDGET-LIMITED": "kappa >= 0.90 at h_top on 2/2 seeds",
                "SCALE-DEGRADES-STANDS-CLEANLY-STRENGTHENED":
                    f"unrecovered AND undamaged (kappa back near the parent's "
                    f"{PARENT_KAPPA:.2f} band)",
                "INSTABILITY-FINDING-SCHEDULE-EXONERATED":
                    f"collapses again toward {DAMAGED_KAPPA:.2f}",
            },
            "queue_eligible": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-ids", default=None)
    ap.add_argument("--sweep-dir", default=None)
    args = ap.parse_args()
    marginal = realized(args.sweep_dir)
    params = KS.TOTAL_PARAM_TABLE_392M[K]
    os.makedirs(OUT, exist_ok=True)

    written = [spec(f"{250 + i:04d}_ncr_scaleaxis_v2prime_K{K}_{RECIPE}_s{s}", s, marginal, params)
               for i, s in enumerate(SEEDS)]
    ids = [s["id"][:4] for s in written]
    assert ids == ["0250", "0251"], ids
    if args.queue_ids:
        existing = {ln.strip()[:4] for ln in open(args.queue_ids) if ln.strip()}
        clash = sorted(set(ids) & existing)
        assert not clash, f"ID COLLISION with queue history: {clash}"
    for s in written:
        with open(os.path.join(OUT, s["id"] + ".json"), "w") as f:
            json.dump(s, f, indent=1)

    tot = sum(s["v2prime"]["marginal_gpu_h_measured"] for s in written)
    print(f"wrote {len(written)} CANDIDATE V2' specs to {OUT}  ({ids[0]}-{ids[-1]})")
    print(f"  K={K} {RECIPE}, seeds {list(SEEDS)}, resume {BASE_STEPS} -> {EXT_STEPS} at a "
          f"CONSTANT lr = {PARENT_FINAL_LR:.3e}")
    print(f"  marginal spend: {tot:.2f} GPU-h  (ceiling {written[0]['v2prime']['ceiling_gpuh']} "
          f"GPU-h/cell, from the FULL-40k measured cost)")
    print("QUEUE-ELIGIBLE: NO. The coordinator stages on the build report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
