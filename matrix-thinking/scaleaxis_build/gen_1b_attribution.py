#!/usr/bin/env python3
"""1.31B ATTRIBUTION ARM -- constant-LR resume-to-40k at the degrading trainable
cells. PI 2026-08-23. Ids 0430-0435. GATED, and CONDITIONAL.

GOVERNING RULE, design sec 7.2, verbatim:
  "No SCALE-DEGRADES verdict -- on any curve, at any K, from calibration or
   from the sweep -- is published without a step-extension attribution arm at
   the degrading K: 2 cells (frozen, seeds 0-1) at 40,000 steps... If the
   doubled-token cells recover to kappa >= 0.90, the verdict is
   TOKEN-BUDGET-LIMITED, not scale-fragile, and is reported as such. If they do
   not, SCALE-DEGRADES stands and is strengthened by the control."

RECIPE-MATCHED, per the ruling already recorded in DEVIATIONS.md GAP 1: the
pinned "frozen" cell shape is an artifact of branch (C)'s origin and cannot
control a TRAINABLE degradation. These specs are trainable-arm, matching the
cells they control.

CONSTANT-LR RESUME IS NOW THE HOUSE MECHANISM (V2' precedent, EXPERIMENT_LOG
2026-08-23 #3): the warm-restart variant re-opened the cosine (3.0e-5 ->
1.66e-4, 5.535x) and ACTIVELY DAMAGED the 392M K=40 trainable cells
(kappa 0.8438 -> 0.5513), making that control uninformative in the harmful
direction. Every cell here carries --const-lr-on-resume; the warm-restart
variant is DEAD and is not offered.

TWO PRE-REGISTERED BARS, per the ratified conventions:
  * CURVE 1 (h_top): kappa >= 0.90 on >= 2/3 extended seeds => TOKEN-BUDGET-
    LIMITED at that (K, recipe); else SCALE-DEGRADES stands, strengthened.
  * V3 ANALOG (Curve 5b at s*=13): the kappa >= 0.90 bar is NOT applicable
    (sec 5.5(ii)); the cell is TOKEN-BUDGET-LIMITED at depth iff its
    Delta_scale vs the matched twin returns inside +-delta_depth = 0.095.

CONDITIONALITY, STATED PLAINLY: these specs presuppose that the 1B trainable
cells at K=32/K=40 DEGRADE. NO 1B CELL HAS RUN. If the 1B sweep returns
SCALE-STABLE at those cells there is nothing to attribute and this wave is NOT
run -- sec 7.1's one-directionality means STABLE and IMPROVES need no control.
They are specced now so the machinery is ready, not because the verdict is
assumed.
"""
from __future__ import annotations
import argparse, json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NCR_SCALE", "1310m"); os.environ.setdefault("NCR_K", "40")
sys.path.insert(0, os.environ.get("SCALEAXIS_TREE", _HERE))
import kscaling_config as KS                                    # noqa: E402
assert KS.SCALE == "1310m" and KS.RUNG == 3
OUT = os.path.join(_HERE, "job_specs_1b_attribution")
PY, WORKDIR, EPH = "/home/nvidia/tdenv/bin/python3", "/home/nvidia/ncr_scaleaxis", "/ephemeral/scaleaxis1b"
ROOT, SCALE, BASE, EXT = f"{EPH}/attribution", "1310m", 20_000, 40_000
RATE = {32: 2.2451, 40: 2.7445}; SEEDS = (0, 1, 2); CELLS = ((40, "compB"), (32, "compB"))
CONST_LR = 3.0e-5; CEIL_MULT = 1.5
COMMON = ("--mode calibration --device cuda --batch-size 32 --eval-batch-size 64 "
          "--warmup-steps 200 --lr 3e-4 --aux-read-loss-weight 0.5 --ortho-reg-weight 0.1 "
          "--eval-every 1000 --ckpt-every 5000")
FLAGS = "--aux-loss-type contrastive+cosine --contrastive-temperature 0.07"


def gpu_h(k, steps=BASE): return round(RATE[k] * steps / 3600, 3)


def spec(jid, k, recipe, seed):
    parent = f"scaleaxis1310m_K{k}_{recipe}_s{seed}"
    cell = f"attrib1b40k_K{k}_{recipe}_s{seed}"
    outdir, ckdir = f"{ROOT}/results", f"{ROOT}/ckpts/{cell}"
    oj = f"{outdir}/{cell}.json"; lad = KS.LADDER_TABLE[k]; params = KS.total_param_table()[k]
    marginal = gpu_h(k); full = 2 * marginal; ceil_v = round(CEIL_MULT * full, 3)
    cmd = (f"mkdir -p {outdir} {ckdir} && [ -f {ckdir}/{cell}.ckpt.pt ] || "
           f"ln {EPH}/ckpts/{parent}/{parent}.ckpt.pt {ckdir}/{cell}.ckpt.pt && cd {WORKDIR} && "
           f"NCR_SCALE={SCALE} NCR_K={k} {PY} ncr_lm_wave1_runner.py --k {k} --scale {SCALE} "
           f"{COMMON} --steps {EXT} --const-lr-on-resume --ceiling-gpuh {ceil_v} {FLAGS} "
           f"--seed {seed} --cell-id {cell} --out {oj} --ckpt-dir {ckdir} "
           f"2>&1 | tee -a {outdir}/{cell}.log")
    vc = (f"{PY} -c \"import json, math; d=json.load(open('{oj}')); "
          f"assert d.get('status')=='COMPLETED', ('status', d.get('status')); "
          f"assert d.get('step',0)>={EXT}, ('step', d.get('step')); "
          f"assert d.get('steps_target')=={EXT}, ('steps_target (TOP-LEVEL, #1)', d.get('steps_target')); "
          f"assert d.get('runner_tag')=='ncr_scaleaxis_runner_v1', ('runner_tag', d.get('runner_tag')); "
          f"assert d.get('const_lr_on_resume') is True, ('const_lr_on_resume', d.get('const_lr_on_resume')); "
          f"assert d.get('resume_start_step')=={BASE}, ('resume_start_step', d.get('resume_start_step')); "
          f"rc=d.get('resume_const_lr'); assert rc is not None and abs(rc-{CONST_LR!r})<1e-12, ('resume_const_lr', rc); "
          f"ks=d.get('kscaling') or {{}}; assert ks.get('K')=={k}, ('K', ks.get('K')); "
          f"assert ks.get('rung')==3, ('rung', ks.get('rung')); "
          f"assert ks.get('h_top')=={lad[-1]}, ('h_top', ks.get('h_top')); "
          f"assert ks.get('scale')=='{SCALE}', ('scale', ks.get('scale')); "
          f"bb=ks.get('backbone') or {{}}; assert (bb.get('d_model'), bb.get('d_state'), bb.get('n_layers'))==(2560,128,22), ('backbone', bb); "
          f"cfg=d.get('config') or {{}}; assert cfg.get('freeze_entity_adapter') is False, ('RECIPE MISMATCH -- the arm must match the DEGRADING cell', cfg.get('freeze_entity_adapter')); "
          f"pp=d.get('params') or {{}}; assert pp.get('per_arm')=={params}, ('params.per_arm', pp.get('per_arm')); "
          f"lh=d.get('loss_history') or {{}}; assert set(lh)=={{'full_graft','backbone_only'}}, ('loss_history arms', sorted(lh)); "
          f"assert len(lh['backbone_only'])>=100, ('backbone_only len', len(lh['backbone_only'])); "
          f"h=lh['full_graft']; assert len(h)>=100, ('full_graft len', len(h)); "
          f"assert all(math.isfinite(r[1]) for r in h), 'non-finite CE (Gate-0-marginal, plateau-tolerant per #2)'\"")
    return {"id": jid, "lane": "NCR", "tier": "scaleaxis1b_attribution",
            "hypothesis": (
                f"1.31B ATTRIBUTION ARM -- the token-budget control design sec 7.2 makes MANDATORY "
                f"before any SCALE-DEGRADES verdict is published. Cell: K={k} (d={k+1}), "
                f"TRAINABLE-contrastive (compB), seed {seed}, rung 3, RESUMED from its own "
                f"20,000-step parent {parent} and extended to {EXT} steps at a CONSTANT lr "
                f"({CONST_LR:.3e}) -- DOUBLE the token budget at identical params, recipe, seed, "
                f"pools and ladder. RECIPE-MATCHED per the ruling in DEVIATIONS.md GAP 1: sec "
                f"7.2's pinned 'frozen' shape is an artifact of branch (C)'s origin and cannot "
                f"control a TRAINABLE degradation. CONSTANT-LR IS THE HOUSE MECHANISM (V2', "
                f"EXPERIMENT_LOG 2026-08-23 #3): the warm-restart variant damaged the 392M K=40 "
                f"trainable cells (kappa 0.8438 -> 0.5513) and is DEAD. D/N = "
                f"{(BASE*32*KS.t_in(k))/params:.3f} tokens/param at 20k -- the most token-starved "
                f"point in the whole three-rung curve, so under-training is at its most plausible "
                f"here and the control is at its most necessary. PRE-REGISTERED BARS: (a) CURVE 1 "
                f"-- kappa >= 0.90 at h_top={lad[-1]} on >= 2/3 extended seeds => "
                f"TOKEN-BUDGET-LIMITED, else SCALE-DEGRADES stands and is STRENGTHENED; (b) V3 "
                f"ANALOG at s*=13 -- the kappa>=0.90 bar is NOT applicable (sec 5.5(ii): kappa at "
                f"13/15 squarings is a numerical-depth reading, not a capability bar), so the "
                f"criterion is Delta_scale returning inside +-delta_depth = 0.095 vs the matched "
                f"twin. Both readouts reported for every cell."),
            "cmd": cmd, "gpu_h_estimate": marginal, "output_dir": outdir, "validity_check": vc,
            "notes": (
                f"CANDIDATE -- NOT queue-eligible, and TRIPLY gated: (a) the 1B sweep must have "
                f"COMPLETED and produced parent {parent}; (b) the pinned cross-scale tests must "
                f"have ADJUDICATED -- and this wave runs ONLY IF they return SCALE-DEGRADES at "
                f"this (K, recipe); (c) an envelope ruling, since the 1B ledger already sits far "
                f"outside sec 8.2's numbers. CONDITIONALITY STATED PLAINLY: these specs presuppose "
                f"a degradation that NO 1B CELL HAS YET MEASURED. If the 1B trainable cells read "
                f"SCALE-STABLE there is nothing to attribute and this wave is NOT run -- sec 7.1's "
                f"one-directionality means STABLE and IMPROVES need no control. They are specced "
                f"now so the machinery is ready, not because the verdict is assumed. "
                f"MECHANISM: resume-extension, the design's own pricing basis (sec 7.2 branch (B) "
                f"and sec 8.2 both price extensions as the MARGINAL 20,000 steps). The parent "
                f"ckpt is HARDLINKED under a fresh cell id, so atomic_torch_save's os.replace "
                f"leaves the 20,000-step CHECKPOINT OF RECORD intact and --out points at a NEW "
                f"json, so the runner's 'already COMPLETED -- skipping' guard cannot fire. "
                f"CEILING {ceil_v} GPU-h = 1.5 x the FULL {EXT}-step projection ({full:.3f} GPU-h "
                f"at this cell's measured/interpolated {RATE[k]} s/step). IT MUST COVER THE FULL "
                f"CUMULATIVE COST: run_two_arm_cell restores cumulative_elapsed_s and checks "
                f"elapsed against the WHOLE 40,000-step clock, so a ceiling sized on the marginal "
                f"half fires ABORTED-BUDGET on step 1 of the resume. "
                f"Gate-0-marginal is SCOPED plateau-tolerant/finite-CE-only per #2; steps_target "
                f"is asserted at the TOP LEVEL per #1."),
            "attribution1b": {
                "rule": "NCR_SCALE_AXIS_DESIGN.md sec 7.2, recipe-matched per DEVIATIONS.md GAP 1",
                "mechanism": "constant-LR resume (V2' precedent; warm-restart variant DEAD)",
                "parent_cell": parent, "parent_spec_must_complete_first": True,
                "conditional_on": "a SCALE-DEGRADES verdict at this (K, recipe) -- NOT yet measured",
                "K": k, "recipe": recipe, "seed": seed, "rung": 3,
                "base_steps": BASE, "extended_steps": EXT, "token_multiple": 2.0,
                "resume_const_lr": CONST_LR,
                "marginal_gpu_h": marginal, "full_40k_gpu_h": round(full, 3),
                "ceiling_gpuh": ceil_v,
                "prereg_curve1": "kappa >= 0.90 at h_top on >= 2/3 seeds => TOKEN-BUDGET-LIMITED",
                "prereg_v3_analog": "Delta_scale at s*=13 inside +-0.095 => TOKEN-BUDGET-LIMITED at depth",
                "queue_eligible": False}}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--queue-ids", default=None)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    w, n = [], 430
    for k, r in CELLS:
        for s in SEEDS:
            w.append(spec(f"{n:04d}_ncr_scaleaxis_1b_attrib40k_K{k}_{r}_s{s}", k, r, s)); n += 1
    ids = [x["id"][:4] for x in w]
    assert len(w) == 6 and ids == [f"{i:04d}" for i in range(430, 436)], ids
    if a.queue_ids:
        ex = {l.strip()[:4] for l in open(a.queue_ids) if l.strip()}
        cl = sorted(set(ids) & ex); assert not cl, f"ID COLLISION: {cl}"
    for x in w:
        json.dump(x, open(os.path.join(OUT, x["id"] + ".json"), "w"), indent=1)
    tot = sum(x["gpu_h_estimate"] for x in w)
    print(f"wrote {len(w)} CANDIDATE 1B attribution specs to {OUT} ({ids[0]}-{ids[-1]})")
    print(f"  K=40 compB s0-s2 + K=32 compB s0-s2, constant-LR resume 20k -> 40k")
    print(f"  MARGINAL ledger: {tot:.2f} GPU-h; ceilings "
          f"{sorted({x['attribution1b']['ceiling_gpuh'] for x in w})}")
    print("QUEUE-ELIGIBLE: NO. Triply gated AND conditional on a verdict not yet measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
