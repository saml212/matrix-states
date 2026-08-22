#!/usr/bin/env python3
"""THE STEP-EXTENSION ATTRIBUTION ARM -- the token-budget control that
NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 makes MANDATORY before any SCALE-DEGRADES
verdict is published.

THE GOVERNING TEXT, QUOTED VERBATIM (design sec 7.2, the block titled "The
step-extension attribution arm -- now unconditional on any DEGRADES verdict
(MAJOR-4)"):

    > **No SCALE-DEGRADES verdict -- on any curve, at any K, from calibration or
    > from the sweep -- is published without a step-extension attribution arm at
    > the degrading K: 2 cells (frozen, seeds 0-1) at 40,000 steps, ~+8.5 GPU-h
    > at x3.75 at K=40.** If the doubled-token cells recover to kappa >= 0.90, the
    > verdict is **TOKEN-BUDGET-LIMITED**, not scale-fragile, and is reported as
    > such. If they do not, SCALE-DEGRADES stands and is *strengthened* by the
    > control.

and its shape/justification from sec 7.1 (the token-budget confound):

    > **Adopted fix: the attribution arm is now conditional on ANY
    > SCALE-DEGRADES verdict at ANY K, at harvest, with the pinned rule -- "no
    > SCALE-DEGRADES claim is published without it."** Priced in sec 8.2.

    > **And the confound is one-directional, which is a strength (sec 1).**
    > Under-training can only manufacture DEGRADES. It cannot manufacture STABLE
    > and it cannot manufacture IMPROVES. So the attribution arm is needed for
    > exactly one of the three outcomes, and the other two are *strengthened* by
    > the very mismatch FROZEN_BIAS warns about.

sec 8.2 prices it as one contingency line:

    > | **Attribution arm (2 cells @ 40,000 steps at the degrading K)** | **ANY
    > SCALE-DEGRADES verdict, any K -- MANDATORY before publication** | **+8.5**
    > (at K=40) |

THE VERDICTS IT MUST DISCHARGE (EXPERIMENT_LOG 2026-08-22 #21, commit 6a06e1c):
  V1  Curve 1 SCALE-DEGRADES at K=32 TRAINABLE  (Delta_scale -0.1169)
  V2  Curve 1 SCALE-DEGRADES at K=40 TRAINABLE  (Delta_scale -0.1442)
  V3  Curve 5b depth-tail SCALE-DEGRADES in BOTH ARMS at 11sq (T=10.5/72) and
      s*=13 (T=6.5/72), LOSO 8/8; per-K at s*=13, 6/8 cells degrade INCLUDING
      frozen K=32/40, with frozen K=16/24 stable.

  => the degrading (K, recipe) cells of record are exactly:
       trainable K=16, K=24, K=32, K=40   and   frozen K=32, K=40.

TWO DESIGN GAPS, SURFACED RATHER THAN PAPERED OVER (see sec "GAPS" below and
ATTRIBUTION_ARM.md): the pinned cell shape says **frozen**, but four of the six
degrading cells -- and two of the three headline verdicts -- are TRAINABLE; and
the pinned recovery criterion `kappa >= 0.90` is explicitly NOT the right bar at
s*=13 by the design's own sec 5.5(ii).

MECHANISM: RESUME-EXTENSION, which is the design's own pricing basis.
sec 7.2 branch (B) prices "extend the six calibration cells only to 40,000
steps" at "+~18.6 GPU-h at x3.75" = 6 x 3.10, i.e. the MARGINAL 20,000 steps,
not a 40,000-step re-run; sec 8.2's "+8.5 (at K=40)" = 2 x 4.24 is the same
arithmetic. The runner supports TRUE step-level resume
(`for step in range(start_step + 1, steps + 1)`, with `start_step` and
`cumulative_elapsed_s` restored from the checkpoint and the seed and
freeze-flag asserted against it), and all 24 sweep checkpoints are on the box.

  Each spec HARDLINKS the parent cell's checkpoint into a fresh attribution
  directory under a fresh cell id, then resumes there. The hardlink costs zero
  bytes on the same filesystem and -- because `atomic_torch_save` does
  `os.replace`, which swaps the DIRECTORY ENTRY while the old inode survives --
  the 20,000-step CHECKPOINT OF RECORD IS NOT OVERWRITTEN. `--out` likewise
  points at a NEW json, so the Stage-C-scored record of record is untouched and
  the runner's `already COMPLETED -- skipping` guard (which keys on the OUT
  path) cannot fire.
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
os.environ.setdefault("NCR_K", "24")
_TREE = os.environ.get("SCALEAXIS_TREE", _HERE)
sys.path.insert(0, _TREE)
import kscaling_config as KS   # noqa: E402
assert KS.SCALE == "392m" and KS.RUNG == 2, (KS.SCALE, KS.RUNG)

OUT = os.path.join(_HERE, "job_specs_attribution")
# The arm is priced from MEASURED cost, so the generator needs the sweep
# records. Repo-side that is the committed archive; on the box it is the live
# results dir. --sweep-dir makes the source explicit instead of implicit.
SWEEP = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                     "experiment-runs", "2026-08-22_scaleaxis_sweep")
PY = "/home/nvidia/tdenv/bin/python3"
WORKDIR = "/home/nvidia/ncr_scaleaxis"
EPH = "/ephemeral/scaleaxis"
ATTRIB = f"{EPH}/attribution"
SCALE = "392m"
EXT_STEPS = 40_000
BASE_STEPS = 20_000
SEEDS = (0, 1)                    # "seeds 0-1", verbatim from the pinned rule
R8 = 1.0026                       # MEASURED at Stage A0.4 (EXPERIMENT_LOG #18)
CEILING_MULT = 1.5                # sec 3.6 breaker 1

# The degrading (K, recipe) cells, from #21. `subset` = the minimal probative
# set for the three named verdicts; `extra` = the rest of "at the degrading K".
V1_V2_CURVE1 = [(32, "compB"), (40, "compB")]
V3_CURVE5B = [(16, "compB"), (24, "compB"), (32, "compB"), (40, "compB"),
              (32, "primary"), (40, "primary")]
SUBSET = [(40, "compB"), (40, "primary"), (32, "compB"), (32, "primary")]
EXTRA = [(24, "compB"), (16, "compB")]
assert set(SUBSET) | set(EXTRA) == set(V1_V2_CURVE1) | set(V3_CURVE5B)

RECIPES = {
    "primary": dict(label="FROZEN-contrastive (primary recipe)",
                    flags="--aux-loss-type contrastive+cosine --freeze-entity-adapter "
                          "--contrastive-temperature 0.07", frozen=True),
    "compB": dict(label="TRAINABLE-contrastive (compB recipe)",
                  flags="--aux-loss-type contrastive+cosine --contrastive-temperature 0.07",
                  frozen=False),
}
COMMON = ("--mode calibration --device cuda --batch-size 32 --eval-batch-size 64 "
          "--warmup-steps 200 --lr 3e-4 --aux-read-loss-weight 0.5 --ortho-reg-weight 0.1 "
          "--eval-every 1000 --ckpt-every 5000")

# 98M per-K reference kappa at h_top, and the #21 Delta_scale of record.
REF98_HTOP = {(16, "compB"): 0.9958, (24, "compB"): 0.9878,
              (32, "compB"): 0.9919, (40, "compB"): 0.9880,
              (32, "primary"): 0.9960, (40, "primary"): 0.9920}
DELTA_C1 = {(32, "compB"): -0.1169, (40, "compB"): -0.1442}


def realized(sweep_dir: str | None = None) -> dict:
    """MEASURED per-(K, recipe) 20,000-step cost from the sweep of record.
    Not a projection: sec 10's "Measured-vs-projected bookkeeping" bullet
    requires the harvest to price from measured, not projected, totals."""
    d0 = sweep_dir or SWEEP
    if not os.path.isdir(d0):
        raise SystemExit(
            f"--sweep-dir {d0!r} does not exist. This generator prices the arm from the "
            f"MEASURED 20,000-step cost of the sweep of record, so it needs those records: "
            f"the committed archive (experiment-runs/2026-08-22_scaleaxis_sweep) in the repo, "
            f"or /ephemeral/scaleaxis/results on the box.")
    agg = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d0, "*.json"))):
        d = json.load(open(p))
        if d.get("mode") != "calibration" or d.get("status") != "COMPLETED":
            continue
        ks = d.get("kscaling") or {}
        r = "primary" if d["config"]["freeze_entity_adapter"] else "compB"
        g = d.get("gpu_h") or (d.get("elapsed_s") or 0) / 3600.0
        agg[(int(ks["K"]), r)].append(g)
    assert len(agg) == 8, (
        f"expected all 8 (K, recipe) cells in {d0}, got {sorted(agg)} -- the arm cannot be "
        f"priced from a partial sweep.")
    return {k: sum(v) / len(v) for k, v in agg.items()}


HYP = (
    "NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 7.2 -- THE MANDATORY STEP-EXTENSION ATTRIBUTION ARM, "
    "the token-budget control that gates publication of every SCALE-DEGRADES verdict in "
    "EXPERIMENT_LOG 2026-08-22 #21. Cell: K={K} (d={d}), {label}, seed {seed}, RESUMED from its "
    "own 20,000-step checkpoint of record and extended to {ext} steps -- DOUBLE THE TOKEN BUDGET "
    "at identical params, identical recipe, identical seed, identical pools, identical ladder. "
    "PINNED RULE, verbatim: \\\"No SCALE-DEGRADES verdict -- on any curve, at any K, from "
    "calibration or from the sweep -- is published without a step-extension attribution arm at "
    "the degrading K: 2 cells (frozen, seeds 0-1) at 40,000 steps... If the doubled-token cells "
    "recover to kappa >= 0.90, the verdict is TOKEN-BUDGET-LIMITED, not scale-fragile, and is "
    "reported as such. If they do not, SCALE-DEGRADES stands and is strengthened by the "
    "control.\\\" WHY THIS CELL IS IN THE ARM: {why} "
    "THE CONFOUND BEING CONTROLLED (sec 1, sec 7.1): steps, batch and t_in were held fixed across "
    "scales, so this cell saw the SAME tokens as its 98M twin while carrying 4.008x the "
    "parameters -- D/N = {dn:.3f} tokens/param at 392M against 1.87 at 98M, i.e. 4x further from "
    "compute-optimal. Under-training can ONLY manufacture DEGRADES; it cannot manufacture STABLE "
    "and it cannot manufacture IMPROVES, so this arm is needed for exactly one of the three "
    "outcomes and the other two are STRENGTHENED by the mismatch. Without it, "
    "\\\"the capability is scale-fragile\\\" cannot be separated from \\\"at a fixed token budget a "
    "392M model is 4x further from compute-optimal\\\" -- sec 10 R3's \\\"publishable and wrong\\\" "
    "outcome. READOUT: P1b kappa at h_top={htop} (residue {htopres} = K/2, antipodal), matched "
    "pools, n=256, base seed 90210, ckpt_step == {ext}; plus the depth ladder at s*=13 for the "
    "Curve-5b leg. 98M reference kappa@h_top for this cell: {ref98:.4f}. {delta_note} "
    "PRE-REGISTERED DECISION RULE, per verdict, fixed BEFORE this arm runs: "
    "(a) CURVE 1 -- kappa@h_top >= 0.90 on >= 2/2 extended seeds => TOKEN-BUDGET-LIMITED at this "
    "(K, recipe); otherwise SCALE-DEGRADES STANDS and is strengthened. "
    "(b) CURVE 5b at s*=13 -- the kappa >= 0.90 bar is NOT applicable (sec 5.5(ii): \\\"kappa at "
    "13/15 squarings is not the CAPABILITY bar -- that bar lives at h_top (5 squarings)... A 98M "
    "kappa of ~0.70 at 15 squarings is a numerical-depth reading, not a capability failure, and "
    "must never be reported as one\\\"), so the criterion is restated ONCE, here, before data: "
    "the extended cell is TOKEN-BUDGET-LIMITED at s*=13 iff its Delta_scale vs the SAME 98M twin "
    "moves back inside +-delta_depth = 0.095; otherwise the depth-tail SCALE-DEGRADES STANDS. "
    "Both readouts are reported for every cell regardless of which verdict it was recruited for."
)

WHY = {
    "V1V2": ("this is one of the two cells carrying a Curve-1 SCALE-DEGRADES verdict "
             "(Delta_scale {dc1:+.4f}, also failing the kappa gate) -- V1/V2 of the three "
             "publication-gated verdicts."),
    "V3F": ("this is a FROZEN cell that degrades on the Curve-5b depth tail at s*=13 -- it is "
            "what makes #21's \\\"BOTH arms lose very-deep-read robustness\\\" claim true, and no "
            "trainable cell can control it."),
    "V3T": ("this cell degrades on the Curve-5b depth tail at s*=13. It carries no Curve-1 "
            "verdict, so it is OUTSIDE the minimal probative subset and is present only under "
            "the full \\\"at the degrading K\\\" reading of the pinned rule."),
}

NOTES = (
    "CANDIDATE -- NOT queue-eligible. Gated on (a) an audit of THIS build and (b) an explicit "
    "coordinator election between OPTION A (all 12 specs, 0230-0241 -- the full \\\"at the "
    "degrading K\\\" reading) and OPTION B (0230-0237 only -- the minimal probative subset for "
    "#21's three named verdicts). THE ELECTION IS A STAGING DECISION, NOT A NAMING ONE: "
    "queue_worker.sh claims by `ls | sort`, which ORDERS claims but does not BLOCK them, so the "
    "subset occupies the contiguous low block 0230-0237 and electing B means staging only those. "
    "{option_note} "
    "MECHANISM: RESUME-EXTENSION, the design's own pricing basis (sec 7.2 branch (B) prices six "
    "cells to 40,000 steps at +18.6 GPU-h = 6 x the K=24 x3.75 per-cell figure, i.e. the MARGINAL "
    "20,000 steps; sec 8.2's +8.5 at K=40 is the same arithmetic). The parent 20,000-step "
    "checkpoint is HARDLINKED into {attrib}/ckpts under a fresh cell id -- zero bytes on the same "
    "filesystem -- so the CHECKPOINT OF RECORD IS NOT OVERWRITTEN (atomic_torch_save does "
    "os.replace, which swaps the directory entry while the old inode survives), and --out points "
    "at a NEW json so the Stage-C-scored record is untouched and the runner's "
    "\\\"already COMPLETED -- skipping\\\" guard (which keys on the OUT path) cannot fire. The "
    "runner asserts the checkpoint's recorded seed AND freeze flag against this spec's, so a "
    "mis-paired resume dies before any GPU work. "
    "DISCLOSED PROPERTY OF THE MECHANISM: get_lr is linear-warmup + cosine to 0.1*max_lr over "
    "total_steps, so resuming with --steps 40000 RE-OPENS the schedule -- LR at the resume point "
    "goes from 3.00e-05 (the 20k floor) to ~1.66e-04, a 5.5x warm restart. That is what "
    "\\\"extend to 40,000 steps\\\" means in this harness and it is what branch (B) pre-registers "
    "and prices, but a kappa recovery could then be attributed to the warm restart rather than to "
    "the tokens. The confound-free alternative -- a FRESH 40,000-step run with a single cosine -- "
    "costs EXACTLY 2x and is priced in ATTRIBUTION_ARM.md; it is offered as an ELECT-or-DECLINE, "
    "not chosen here. "
    "CEILING: {ceilprov} "
    "Built by the ATTRIBUTION-ARM BUILD agent 2026-08-22 against design DRAFT-R2 as amended "
    "(A1/A2/A3) and the scaleaxis tree of record (runner c16e5ccf51794347a62bca62bb702ab5, graft "
    "e47b17cd4e25f66153bb71cf9df3adf7, config e44916b6685ffd602b7d2e0434041bee, battery "
    "f70bc1c1141eda5b043eda5c4ee1b459). NCR_K, --k and --scale 392m all present and mutually "
    "asserted against the RESOLVED backbone dict. Checkpoints to /ephemeral, never the root fs. "
    "One cell per GPU per the standing declined-packing ruling. Scored afterwards by "
    "kscaling_battery.py and depthext_eval.py from the scaleaxis tree (B5 scale guard active; "
    "C4 --outdir defaults now point at ~/ncr_scaleaxis/results)."
)


def validity(out_json: str, k: int, recipe: str, params: int) -> str:
    lad = list(KS.LADDER_TABLE[k])
    frozen = str(RECIPES[recipe]["frozen"])
    return (
        f"{PY} -c \"import json, math; d=json.load(open('{out_json}')); "
        f"assert d.get('status')=='COMPLETED', d.get('status'); "
        f"assert d.get('step',0)>={EXT_STEPS}, d.get('step'); "
        f"assert (d.get('config') or {{}}).get('steps_target')=={EXT_STEPS}, 'not the extended budget'; "
        f"assert d.get('runner_tag')=='ncr_scaleaxis_runner_v1', d.get('runner_tag'); "
        f"ks=d.get('kscaling') or {{}}; "
        f"assert ks.get('K')=={k}, ('K', ks.get('K')); "
        f"assert ks.get('d_ncr')=={k+1}, ('d_ncr', ks.get('d_ncr')); "
        f"assert ks.get('h_top')=={lad[-1]}, ('h_top', ks.get('h_top')); "
        f"assert ks.get('deep_ladder')=={lad}, ks.get('deep_ladder'); "
        f"assert ks.get('scale')=='{SCALE}', ('scale', ks.get('scale')); "
        f"bb=ks.get('backbone') or {{}}; "
        f"assert (bb.get('d_model'), bb.get('d_state'), bb.get('n_layers'))==(1536,128,16), bb; "
        f"cfg=d.get('config') or {{}}; "
        f"assert cfg.get('freeze_entity_adapter') is {frozen}, "
        f"('RECIPE MISMATCH -- the arm must match the DEGRADING cell', cfg.get('freeze_entity_adapter')); "
        f"pp=d.get('params') or {{}}; "
        f"assert pp.get('per_arm')=={params}, ('PARAM COUNT vs design sec 3.4', pp.get('per_arm')); "
        f"lh=d.get('loss_history') or {{}}; "
        f"assert set(lh)=={{'full_graft','backbone_only'}}, sorted(lh); "
        f"assert len(lh['backbone_only'])>=100, len(lh['backbone_only']); "
        f"h=lh['full_graft']; assert len(h)>=100, len(h); "
        f"assert all(math.isfinite(r[1]) for r in h), 'non-finite CE in loss_history'; "
        f"assert h[-1][1] < h[0][1], ('GATE-0 NOT CONVERGED', h[0], h[-1])\"")


def spec(job_id: str, k: int, recipe: str, seed: int, in_subset: bool,
         why_key: str, real: dict) -> dict:
    d, r = k + 1, RECIPES[recipe]
    parent = f"scaleaxis392m_K{k}_{recipe}_s{seed}"
    cell = f"attrib40k_K{k}_{recipe}_s{seed}"
    outdir = f"{ATTRIB}/results"
    out_json = f"{outdir}/{cell}.json"
    ckdir = f"{ATTRIB}/ckpts/{cell}"
    lad = KS.LADDER_TABLE[k]
    params = KS.TOTAL_PARAM_TABLE_392M[k]
    marginal = real[(k, recipe)]
    full40k = 2.0 * marginal
    ceil_v = round(CEILING_MULT * R8 * full40k, 3)
    ceilprov = (
        f"{CEILING_MULT} x R_8 x the FULL {EXT_STEPS}-step projection at this cell's OWN MEASURED "
        f"rate ({marginal:.4f} GPU-h per 20,000 steps => {full40k:.4f} GPU-h at 40,000; "
        f"R_8 = {R8} MEASURED at Stage A0.4) = {ceil_v} GPU-h. THE CEILING MUST COVER THE FULL "
        f"CUMULATIVE COST, NOT THE MARGINAL 20,000 STEPS: run_two_arm_cell restores "
        f"cumulative_elapsed_s from the checkpoint and sets t0 = time.time() - "
        f"cumulative_elapsed_s, so `elapsed > ceiling_s` is checked against the WHOLE 40,000-step "
        f"wall clock. A ceiling sized on the marginal half would fire ABORTED-BUDGET on step 1 of "
        f"the resume.")
    dn = (BASE_STEPS * 32 * KS.t_in(k)) / params
    dnote = (f"#21 records Delta_scale = {DELTA_C1[(k, recipe)]:+.4f} at h_top for this cell."
             if (k, recipe) in DELTA_C1 else
             "This cell carries no Curve-1 verdict; it is recruited for the Curve-5b depth tail.")
    why = WHY[why_key].format(dc1=DELTA_C1.get((k, recipe), 0.0))
    onote = ("IN THE MINIMAL PROBATIVE SUBSET (Option B) and in Option A."
             if in_subset else
             "OPTION A ONLY -- outside the minimal probative subset, because this cell carries no "
             "Curve-1 verdict and #21's \\\"both arms\\\" claim is already controlled by the frozen "
             "K=32/40 cells. Including it is what makes the arm fully compliant with \\\"at the "
             "degrading K\\\" for every row of the Curve-5b per-K table.")

    # The hardlink is idempotent: `ln -f` re-points, and a re-run after a
    # partial extension keeps the RESUMED checkpoint (which is what we want).
    cmd = (f"mkdir -p {outdir} {ckdir} && "
           f"[ -f {ckdir}/{cell}.ckpt.pt ] || ln {EPH}/ckpts/{parent}/{parent}.ckpt.pt "
           f"{ckdir}/{cell}.ckpt.pt && cd {WORKDIR} && "
           f"NCR_SCALE={SCALE} NCR_K={k} {PY} ncr_lm_wave1_runner.py "
           f"--k {k} --scale {SCALE} {COMMON} --steps {EXT_STEPS} "
           f"--ceiling-gpuh {ceil_v} {r['flags']} "
           f"--seed {seed} --cell-id {cell} --out {out_json} "
           f"--ckpt-dir {ckdir} 2>&1 | tee -a {outdir}/{cell}.log")

    return {
        "id": job_id, "lane": "NCR", "tier": "scaleaxis_attribution",
        "hypothesis": HYP.format(K=k, d=d, label=r["label"], seed=seed, ext=EXT_STEPS,
                                 why=why, dn=dn, htop=lad[-1], htopres=lad[-1] % k,
                                 ref98=REF98_HTOP[(k, recipe)], delta_note=dnote),
        "cmd": cmd,
        "gpu_h_estimate": round(marginal, 3),
        "output_dir": outdir,
        "validity_check": validity(out_json, k, recipe, params),
        "notes": NOTES.format(option_note=onote, attrib=ATTRIB, ceilprov=ceilprov),
        "attribution": {
            "design_section": "NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 7.2 (the pinned rule)",
            "gates_verdicts": ([f"V1/V2 Curve-1 SCALE-DEGRADES at K={k} trainable"]
                               if (k, recipe) in DELTA_C1 else [])
                              + [f"V3 Curve-5b depth-tail SCALE-DEGRADES at s*=13 "
                                 f"(K={k}, {recipe})"],
            "parent_cell": parent,
            "parent_ckpt": f"{EPH}/ckpts/{parent}/{parent}.ckpt.pt",
            "mechanism": "resume-extension (hardlinked parent ckpt; record of record preserved)",
            "base_steps": BASE_STEPS, "extended_steps": EXT_STEPS,
            "token_multiple": 2.0,
            "D_over_N_at_20k": round(dn, 4),
            "recipe": recipe, "frozen": r["frozen"], "seed": seed,
            "in_minimal_probative_subset": in_subset,
            "option": "A+B" if in_subset else "A only",
            "marginal_gpu_h_measured": round(marginal, 4),
            "full_40k_gpu_h_measured": round(full40k, 4),
            "ceiling_gpuh": ceil_v,
            "ceiling_provenance": ceilprov,
            "queue_eligible": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-ids", default=None)
    ap.add_argument("--sweep-dir", default=None,
                    help="dir of the 24 sweep training records (repo archive, or "
                         "/ephemeral/scaleaxis/results on the box)")
    args = ap.parse_args()
    real = realized(args.sweep_dir)
    os.makedirs(OUT, exist_ok=True)

    written, n = [], 230
    # Subset first (longest-first within it), so Option B is the contiguous
    # low block 0230-0237 and electing B is "stage only those".
    for k, recipe in SUBSET:
        for seed in SEEDS:
            why = "V1V2" if (k, recipe) in DELTA_C1 else "V3F"
            written.append(spec(f"{n:04d}_ncr_scaleaxis_attrib40k_K{k}_{recipe}_s{seed}",
                                k, recipe, seed, True, why, real))
            n += 1
    for k, recipe in EXTRA:
        for seed in SEEDS:
            written.append(spec(f"{n:04d}_ncr_scaleaxis_attrib40k_K{k}_{recipe}_s{seed}",
                                k, recipe, seed, False, "V3T", real))
            n += 1

    ids = [s["id"][:4] for s in written]
    assert len(written) == 12 and len(set(ids)) == 12, ids
    assert ids == [f"{i:04d}" for i in range(230, 242)], ids
    assert [s["id"][:4] for s in written if s["attribution"]["in_minimal_probative_subset"]] \
        == [f"{i:04d}" for i in range(230, 238)]
    if args.queue_ids:
        existing = {ln.strip()[:4] for ln in open(args.queue_ids) if ln.strip()}
        clash = sorted(set(ids) & existing)
        assert not clash, f"ID COLLISION with queue history: {clash}"

    for s in written:
        with open(os.path.join(OUT, s["id"] + ".json"), "w") as f:
            json.dump(s, f, indent=1)

    sub = [s for s in written if s["attribution"]["in_minimal_probative_subset"]]
    a_resume = sum(s["attribution"]["marginal_gpu_h_measured"] for s in written)
    b_resume = sum(s["attribution"]["marginal_gpu_h_measured"] for s in sub)
    print(f"wrote {len(written)} CANDIDATE attribution specs to {OUT}")
    print(f"  OPTION A (full 'at the degrading K')  ids 0230-0241, {len(written)} cells: "
          f"{a_resume:6.2f} GPU-h resume / {2*a_resume:6.2f} GPU-h fresh-40k")
    print(f"  OPTION B (minimal probative subset)   ids 0230-0237, {len(sub)} cells: "
          f"{b_resume:6.2f} GPU-h resume / {2*b_resume:6.2f} GPU-h fresh-40k")
    print(f"  marginal cost of FULL compliance over the subset: {a_resume - b_resume:+.2f} GPU-h")
    print("QUEUE-ELIGIBLE: NO. Every spec is CANDIDATE-marked. Election is a STAGING decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
