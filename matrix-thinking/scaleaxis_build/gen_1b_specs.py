#!/usr/bin/env python3
"""THE 1.31B SCALE POINT -- rung 3, the parameter curve's THIRD point.
PI directive 2026-08-23. Calibration pair 0360-0361 + 24-cell sweep 0370-0393.

CEREMONY BASIS. This is a DESIGN AMENDMENT to the audited scale-axis design,
not a new design: NCR_SCALE_AXIS_DESIGN.md sec 3 parameterizes the port BY RUNG
("RUNG1_BACKBONE ... resolved from kscaling_config's RUNGS table via the
mandatory NCR_SCALE env var"), sec 3.4's param formula is rung-generic and was
validated against four independently-measured endpoints at rungs 1 and 2, and
sec 3.5's invariants (ladders, chance, bands, document geometry, the battery,
the recipe) are functions of K alone and are UNCHANGED by any rung. Adding rung
3 therefore needs NO new literal in the graft -- the G1 asserts are now
table-driven -- and the same bands and instruments apply.

THE CONFIG IS THE REPO'S OWN, VERBATIM. lm_rd_rung_configs.py on the box:
    3: {"d_model": 2560, "n_layers": 22, "d_state": 128,
        "approx_params": 1_310_000_000, "target": "~1.3B"}
This is the architecture the 14M->1.31B attractor span study used. NOT derived
here, so nothing is flagged as new on the architecture axis.

PARAM COUNT, sec 3.4's formula at rung 3, VERIFIED AGAINST HARDWARE:
    backbone(vocab 50259) = 1,311,140,608
    K=16 1,311,398,801 | K=24 1,311,441,817 | K=32 1,311,484,833 | K=40 1,311,527,849
The VRAM probe measured the nn.Module count at K=16 and K=40 and both match the
formula EXACTLY, so rung 3's derivation is validated on real CUDA rather than
trusted (the same standard rungs 1/2 met against archived measurements).

MEASURED COST AND MEMORY (this build's own probe, real CUDA, batch 32, two arms
+ eval pass; NO OOM at either end of the K range):
    K=16  t_in 128  1.3355 s/step  49.91 GB   ->  7.419 GPU-h/cell
    K=40  t_in 286  2.7445 s/step  64.17 GB   -> 15.247 GPU-h/cell
K=24/K=32 are INTERPOLATED IN t_in and FLAGGED as interpolations (the same
convention design sec 4.4 Rule P3 uses, and audit m5's lesson that a step
assignment must not be labelled an interpolation -- this one genuinely is one).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NCR_SCALE", "1310m")
os.environ.setdefault("NCR_K", "24")
_TREE = os.environ.get("SCALEAXIS_TREE", _HERE)
sys.path.insert(0, _TREE)
import kscaling_config as KS   # noqa: E402
assert KS.SCALE == "1310m" and KS.RUNG == 3, (KS.SCALE, KS.RUNG)

OUT = os.path.join(_HERE, "job_specs_1b")
PY = "/home/nvidia/tdenv/bin/python3"
WORKDIR = "/home/nvidia/ncr_scaleaxis"
EPH = "/ephemeral/scaleaxis1b"
SCALE = "1310m"
CALIB_K, SWEEP_K = 24, (40, 32, 24, 16)     # longest-first dispatch (sec 8.3)
SEEDS = (0, 1, 2)
CEILING_MULT = 1.5

# MEASURED at K=16/40 by this build's probe; K=24/32 interpolated in t_in.
RATE = {16: 1.3355, 24: 1.7457, 32: 2.2451, 40: 2.7445}
MEASURED_K = (16, 40)
PEAK_GB = {16: 49.913, 40: 64.17}

RECIPES = {
    "primary": dict(label="FROZEN-contrastive (primary recipe)",
                    flags="--aux-loss-type contrastive+cosine --freeze-entity-adapter "
                          "--contrastive-temperature 0.07"),
    "compB": dict(label="TRAINABLE-contrastive (compB recipe)",
                  flags="--aux-loss-type contrastive+cosine --contrastive-temperature 0.07"),
}
COMMON = ("--mode calibration --device cuda --steps 20000 --batch-size 32 "
          "--eval-batch-size 64 --warmup-steps 200 --lr 3e-4 "
          "--aux-read-loss-weight 0.5 --ortho-reg-weight 0.1 --eval-every 1000")

REF = {  # 98M / 392M kappa@h_top medians of record, the two existing points
    (16, "primary"): (1.0000, 1.0000), (16, "compB"): (0.9958, 0.9958),
    (24, "primary"): (1.0000, 1.0000), (24, "compB"): (0.9878, 0.9878),
    (32, "primary"): (0.9960, 0.9960), (32, "compB"): (0.9919, 0.8750),
    (40, "primary"): (0.9920, 0.9880), (40, "compB"): (0.9880, 0.8438),
}


def gpu_h(k): return round(RATE[k] * 20000 / 3600, 3)


def ceiling(k): return round(CEILING_MULT * gpu_h(k), 3)


def validity(out_json, k, params):
    lad = list(KS.LADDER_TABLE[k])
    return (
        f"{PY} -c \"import json, math; d=json.load(open('{out_json}')); "
        f"assert d.get('status')=='COMPLETED', ('status', d.get('status')); "
        f"assert d.get('step',0)>=20000, ('step', d.get('step')); "
        f"assert d.get('steps_target')==20000, ('steps_target (TOP-LEVEL, EXPERIMENT_LOG "
        f"2026-08-23 #1)', d.get('steps_target')); "
        f"assert d.get('runner_tag')=='ncr_scaleaxis_runner_v1', ('runner_tag', d.get('runner_tag')); "
        f"ks=d.get('kscaling') or {{}}; "
        f"assert ks.get('K')=={k}, ('K', ks.get('K')); "
        f"assert ks.get('d_ncr')=={k+1}, ('d_ncr', ks.get('d_ncr')); "
        f"assert ks.get('h_top')=={lad[-1]}, ('h_top', ks.get('h_top')); "
        f"assert ks.get('deep_ladder')=={lad}, ('deep_ladder', ks.get('deep_ladder')); "
        f"assert ks.get('scale')=='{SCALE}', ('scale', ks.get('scale')); "
        f"assert ks.get('rung')==3, ('rung', ks.get('rung')); "
        f"bb=ks.get('backbone') or {{}}; "
        f"assert (bb.get('d_model'), bb.get('d_state'), bb.get('n_layers'))==(2560,128,22), ('backbone', bb); "
        f"pp=d.get('params') or {{}}; "
        f"assert pp.get('per_arm')=={params}, ('params.per_arm vs sec 3.4 at rung 3', pp.get('per_arm')); "
        f"lh=d.get('loss_history') or {{}}; "
        f"assert set(lh)=={{'full_graft','backbone_only'}}, ('loss_history arms', sorted(lh)); "
        f"assert len(lh['backbone_only'])>=100, ('backbone_only len', len(lh['backbone_only'])); "
        f"h=lh['full_graft']; assert len(h)>=100, ('full_graft len', len(h)); "
        f"assert all(math.isfinite(r[1]) for r in h), 'non-finite CE in loss_history'; "
        f"assert h[-1][1] < h[0][1], ('GATE-0 NOT CONVERGED', h[0], h[-1])\"")


HYP = (
    "THE 1.31B SCALE POINT -- rung 3, the parameter curve's THIRD point (PI 2026-08-23). "
    "Cell: K={K} (d={d}), {label}, seed {seed}, backbone RUNG 3 (d_model=2560, d_state=128, "
    "n_layers=22) -- the repo's OWN 1.31B architecture, taken VERBATIM from "
    "lm_rd_rung_configs.py RUNGS[3] (the config the 14M->1.31B attractor span study used), at "
    "MATCHED (K, d), MATCHED recipe and MATCHED 20,000 steps. This is a DESIGN AMENDMENT to the "
    "audited scale axis, not a new design: sec 3 parameterizes the port BY RUNG, sec 3.4's param "
    "formula is rung-generic, and sec 3.5's invariants (ladders, chance, bands, document "
    "geometry, battery, recipe) are functions of K alone. Same bands, same instruments. "
    "PARAM COUNT {params:,}/arm from sec 3.4's formula, VERIFIED AGAINST THE MEASURED nn.Module "
    "COUNT on real CUDA at K=16 and K=40 (both exact) -- 3.347x the 392M rung and 13.41x the 98M "
    "rung, so the three-point curve spans 13.4x in parameters at fixed (K, d, recipe, steps). "
    "THE TOKEN-BUDGET CONFOUND DEEPENS AND STAYS ONE-DIRECTIONAL: steps, batch and t_in are held "
    "fixed, so this cell sees the SAME tokens as its 98M and 392M twins while carrying 13.4x/3.3x "
    "the parameters -- D/N = {dn:.3f} tokens/param here against 0.47 at 392M and 1.87 at 98M. "
    "Under-training can ONLY manufacture DEGRADES; it cannot manufacture STABLE and it cannot "
    "manufacture IMPROVES, so a SCALE-STABLE or SCALE-IMPROVES reading at rung 3 is STRONGER than "
    "it looks, and any SCALE-DEGRADES reading inherits sec 7.2's MANDATORY attribution arm before "
    "publication. Primary readout: P1b kappa at h_top={htop} (residue {htopres} = K/2, "
    "antipodal), matched pools, n=256, base seed 90210, ckpt_step == 20000. Both regimes scored: "
    "P1b = EXACT-WRITE (the CAPABILITY curve); P0 = LEARNED-WRITE (the WALL curve, predicted "
    "inside the per-K binomial band around chance = 1/{K} = {chance:.4f}). Ladder {ladder}, "
    "residues {res} -- UNCHANGED by the rung (a number-theoretic function of K alone). "
    "REFERENCES for this cell, both frozen before any 1.31B number existed: 98M kappa@h_top "
    "median {ref98:.4f}, 392M {ref392:.4f}. MEASURED COST: {sps} s/step, {gh} GPU-h/cell, peak "
    "{peak} at batch 32 ({measured}). {tier_note}")

CAL_NOTE = (
    "THIS IS ONE OF THE TWO CALIBRATION CELLS (K=24, both recipes, seed 0), run FIRST AND ALONE. "
    "LICENSE_SWEEP_1B requires all three legs on the FROZEN cell, the same legs sec 4.2 pins: "
    "(1) Gate-0 convergence (final CE < initial CE on full_graft, finite throughout, reaches "
    "step 20000 COMPLETED); (2) P1b kappa >= 0.90 at the train hops h in {1,2,3}; (3) P1b kappa "
    ">= 0.90 at h_top(24)=36. K=24 is elected for the same three pre-stated reasons as at 392M: "
    "it is the CENTRE of the ported range, it carries the largest evidence base at both existing "
    "rungs, and its t_in=174/pad=0 make it byte-identical to the pinned document construction so "
    "a calibration failure cannot be a padding artifact. --ckpt-every 5000 so the P1b kappa "
    "trajectory instrument (kappa_reader.py, B7) is available if branch (B) is needed. Failure "
    "of any leg routes to sec 7.2's branches, never to the sweep.")
SWEEP_NOTE = (
    "THIS IS ONE OF THE 24 SWEEP CELLS, TRIPLE-GATED: (a) the build audit, (b) the measured-cost "
    "and memory adjudication below, and (c) the LICENSE_SWEEP_1B sentinel dropped by the "
    "calibration pair. Dispatch order is longest-first (K=40 -> 32 -> 24 -> 16) per sec 8.3, "
    "where FILENAME ORDER is the schedule control.")

NOTES = (
    "CANDIDATE -- NOT queue-eligible. {gate} "
    "DESIGN AMENDMENT, ceremony basis stated: sec 3's rung-parameterization licenses a third "
    "rung without a new design -- the graft's backbone asserts are now TABLE-DRIVEN, so rung 3 "
    "introduced NO new literal, and sec 3.4's formula (validated at rungs 1/2 against four "
    "independently-measured endpoints) reproduces the MEASURED nn.Module count at rung 3 exactly "
    "at both K probed. Produced by patch_scaleaxis.py from md5-PINNED sources; the pinned "
    "originals in ~/ncr_g3b31_contrastive are UNTOUCHED and re-verified (runner "
    "9a93198b642242f512ff8489e32b0a53, graft bc105af69661e488ff95f5046e2bcd8a). NCR_K, --k and "
    "--scale 1310m are all present and mutually asserted against the RESOLVED backbone dict "
    "before any GPU work. "
    "THREE THINGS THAT DO NOT TRANSFER CLEANLY, FLAGGED FOR RULING RATHER THAN ASSUMED: "
    "(i) design sec 4.4 Rule P4's 40 GB gate FIRES -- the measured peak is {peak} at K={K} "
    "(64.17 GB at K=40), so sec 8.3's placement assumption RE-OPENS and must be adjudicated "
    "before this wave queues. It is NOT a blocker (it fits on an 80 GB H100 at one cell per GPU) "
    "but it forecloses any packing and leaves ~16 GB of headroom at K=40. "
    "(ii) sec 5's CROSS-SCALE INSTRUMENTS ARE PAIRWISE (TEST-X is 98M-vs-392M over 8 strata; "
    "delta = 0.05 is a two-point equivalence margin). A THIRD rung needs either a second pairwise "
    "test (392M-vs-1310M, or 98M-vs-1310M) or a genuine three-point trend statistic. NEITHER IS "
    "PRE-REGISTERED. This spec asserts no cross-scale verdict; the readout is the within-rung "
    "band table plus the descriptive three-point curve, and the trend instrument must be pinned "
    "BEFORE the harvest reads it. "
    "(iii) THE LEDGER IS A NEW ENVELOPE. The 24-cell sweep is 269.0 GPU-h at measured rates -- "
    "2.4x the entire 392M chapter and well past sec 8.2's ~112 GPU-h tier-(c) boundary and its "
    "130 GPU-h contingency gate. Those numbers were written for a 4x port, not a 13.4x one; this "
    "wave needs its own explicit envelope ruling, not an inherited one. "
    "MIN_KERNEL_T: rung 3 is d_state=128, the same value A0.2 validated the 128 floor at for "
    "rung 2, so the finding transfers -- but d_model and n_layers moved, so the gate is RE-FIRED "
    "at this rung by scaleaxis_gates (K=16's t_in is exactly 128, zero margin). "
    "CEILING: {ceilprov} "
    "Checkpoints to {eph} (NEVER the root fs); one cell per GPU per the standing declined-packing "
    "ruling, which the 64 GB peak now makes mandatory rather than elective. Scored afterwards by "
    "kscaling_battery.py from the scaleaxis tree, whose B5 scale guard refuses any checkpoint "
    "whose recorded backbone is not this rung's.")


def spec(job_id, k, recipe, seed, tier):
    d, r = k + 1, RECIPES[recipe]
    cell = f"scaleaxis1310m_K{k}_{recipe}_s{seed}"
    outdir, ckdir = f"{EPH}/results", f"{EPH}/ckpts/{cell}"
    out_json = f"{outdir}/{cell}.json"
    lad = KS.LADDER_TABLE[k]
    params = KS.total_param_table()[k]
    ck_every = 5000 if tier == "calibration" else 10000
    ceil_v = ceiling(k)
    meas = ("MEASURED at this K" if k in MEASURED_K else
            "INTERPOLATED in t_in from the K=16/K=40 measurements, FLAGGED as an interpolation")
    peak = (f"{PEAK_GB[k]} GB" if k in PEAK_GB else
            f"~{round(PEAK_GB[16] + (PEAK_GB[40]-PEAK_GB[16])*(KS.t_in(k)-128)/158, 1)} GB "
            f"(interpolated)")
    ceilprov = (f"{CEILING_MULT} x this cell's own {'MEASURED' if k in MEASURED_K else 'interpolated'} "
                f"20,000-step projection ({gpu_h(k)} GPU-h at {RATE[k]} s/step) = {ceil_v} GPU-h. "
                f"Priced from THIS BUILD'S OWN real-CUDA probe, not from any 98M/392M multiplier.")
    cmd = (f"mkdir -p {outdir} {ckdir} && cd {WORKDIR} && "
           f"NCR_SCALE={SCALE} NCR_K={k} {PY} ncr_lm_wave1_runner.py "
           f"--k {k} --scale {SCALE} {COMMON} --ckpt-every {ck_every} "
           f"--ceiling-gpuh {ceil_v} {r['flags']} "
           f"--seed {seed} --cell-id {cell} --out {out_json} "
           f"--ckpt-dir {ckdir} 2>&1 | tee -a {outdir}/{cell}.log")
    ref98, ref392 = REF[(k, recipe)]
    return {
        "id": job_id, "lane": "NCR", "tier": f"scaleaxis1b_{tier}",
        "hypothesis": HYP.format(K=k, d=d, label=r["label"], seed=seed, params=params,
                                 dn=(20000 * 32 * KS.t_in(k)) / params, htop=lad[-1],
                                 htopres=lad[-1] % k, chance=1.0 / k, ladder=list(lad),
                                 res=[h % k for h in lad], ref98=ref98, ref392=ref392,
                                 sps=RATE[k], gh=gpu_h(k), peak=peak, measured=meas,
                                 tier_note=CAL_NOTE if tier == "calibration" else SWEEP_NOTE),
        "cmd": cmd, "gpu_h_estimate": gpu_h(k), "output_dir": outdir,
        "validity_check": validity(out_json, k, params),
        "notes": NOTES.format(
            gate=("DOUBLE-GATED: the build audit, and the memory/ledger adjudication below."
                  if tier == "calibration" else
                  "TRIPLE-GATED: the build audit, the memory/ledger adjudication below, and the "
                  "LICENSE_SWEEP_1B sentinel from the calibration pair."),
            peak=peak, K=k, ceilprov=ceilprov, eph=EPH),
        "scaleaxis1b": {
            "scale": SCALE, "rung": 3, "K": k, "d_ncr": d, "recipe": recipe, "seed": seed,
            "backbone": dict(KS.BACKBONE), "config_source": "lm_rd_rung_configs.py RUNGS[3], verbatim",
            "params_per_arm": params,
            "params_verified_against_measured_nn_module_count": k in MEASURED_K,
            "ratio_to_392m": round(params / KS.TOTAL_PARAM_TABLE_392M[k], 4),
            "ratio_to_98m": round(params / KS.TOTAL_PARAM_TABLE_98M[k], 4),
            "s_per_step": RATE[k], "rate_basis": meas,
            "peak_gb_batch32": PEAK_GB.get(k), "gpu_h_estimate": gpu_h(k),
            "ceiling_gpuh": ceil_v, "ceiling_provenance": ceilprov,
            "P4_40GB_gate": "FIRES -- placement assumption re-opens, adjudicate before queueing",
            "cross_scale_instrument": "NOT PRE-REGISTERED for three rungs (sec 5 is pairwise)",
            "ledger_note": "24-cell sweep = 269.0 GPU-h, 2.4x the 392M chapter; new envelope",
            "queue_eligible": False,
            "gate": ("build audit + memory/ledger ruling" if tier == "calibration" else
                     "build audit + memory/ledger ruling + LICENSE_SWEEP_1B"),
        },
    }


# ==========================================================================
# 1B SEED THICKENING (PI 2026-08-23, six-day window). Seeds 3,4,5 at every
# (K, recipe), taking the THIRD scale point to n=6 to match 98M and 392M.
# SEPARATE entry point so it can never rewrite or renumber 0360-0393.
# ==========================================================================
THICKEN_SEEDS = (3, 4, 5)
THICKEN_ID_START = 400
OUT_THICKEN = os.path.join(_HERE, "job_specs_1b_thicken")


def main_thicken() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-ids", default=None)
    args, _ = ap.parse_known_args([a for a in sys.argv[1:] if a != "thicken"])
    os.makedirs(OUT_THICKEN, exist_ok=True)
    written, n = [], THICKEN_ID_START
    for k in SWEEP_K:                       # longest-first, sec 8.3
        for recipe in ("primary", "compB"):
            for seed in THICKEN_SEEDS:
                sp = spec(f"{n:04d}_ncr_scaleaxis_1b_thicken_K{k}_{recipe}_s{seed}",
                          k, recipe, seed, "sweep")
                old = f"scaleaxis1310m_K{k}_{recipe}_s{seed}"
                new = f"scaleaxis1310m_thicken_K{k}_{recipe}_s{seed}"
                for f in ("cmd", "validity_check"):
                    sp[f] = sp[f].replace(old, new)
                sp["tier"] = "scaleaxis1b_thicken"
                sp["scaleaxis1b"]["thickening"] = {
                    "directive": "PI 2026-08-23 -- take the THIRD scale point to n=6, matching "
                                 "98M and 392M",
                    "takes_n_from": 3, "takes_n_to": 6,
                    "within_K_pairs_from": 9, "within_K_pairs_to": 36,
                    "sibling_seeds_of_record": [0, 1, 2],
                    "changes_no_verdict": ("the 1B bands are the SAME bands, pinned before any "
                                           "1B number existed; this wave adds precision. A "
                                           "pinned verdict FLIPPING at n=6 is a FINDING with "
                                           "its own entry, never a silent update."),
                }
                sp["notes"] = ("CANDIDATE -- NOT queue-eligible. 1B SEED-THICKENING (PI "
                               "2026-08-23): byte-pattern-identical to the audited 1B sweep "
                               "specs 0370-0393 modulo the seed integer, the id and the cell "
                               "id. Same measured ceilings, same rung-3 assertions. Gated on "
                               "the SAME LICENSE_SWEEP_1B sentinel as its n=3 siblings. "
                               + sp["notes"])
                written.append(sp)
                n += 1
    ids = [x["id"][:4] for x in written]
    assert len(written) == 24 and ids == [f"{i:04d}" for i in range(THICKEN_ID_START,
                                                                    THICKEN_ID_START + 24)], ids
    if args.queue_ids:
        existing = {ln.strip()[:4] for ln in open(args.queue_ids) if ln.strip()}
        clash = sorted(set(ids) & existing)
        assert not clash, f"ID COLLISION: {clash}"
    for x in written:
        with open(os.path.join(OUT_THICKEN, x["id"] + ".json"), "w") as f:
            json.dump(x, f, indent=1)
    tot = sum(x["gpu_h_estimate"] for x in written)
    print(f"wrote {len(written)} CANDIDATE 1B thickening specs to {OUT_THICKEN} "
          f"({ids[0]}-{ids[-1]})")
    print(f"  MEASURED ledger: {tot:.2f} GPU-h  (per-cell mean {tot/24:.3f}); "
          f"wall on 8 GPUs ~{tot/8:.1f} h")
    print("QUEUE-ELIGIBLE: NO.")
    return 0


# ==========================================================================
# n=9 SURPLUS WAVE (PI 2026-08-29). Seeds 6,7,8, ids 0440-0463. Priced from the
# MEASURED gpu_h of the n=6 thickening cells (0400-0423, 24/24 COMPLETED, 0
# failures) -- NOT from the probe projection, and not from any estimate. The
# three prior directives under-priced by 46%, 39% and 28%; measured-only from
# here. Separate entry point so it cannot renumber 0400-0423.
# ==========================================================================
N9_SEEDS = (6, 7, 8)
N9_ID_START = 440
OUT_N9 = os.path.join(_HERE, "job_specs_1b_n9")
# MEASURED means over the three n=6 seeds, per (K, recipe). Source:
# /ephemeral/scaleaxis1b/results/*thicken*.json, read 2026-08-29.
MEASURED_N6 = {(16, "compB"): 7.8129, (16, "primary"): 7.8150,
               (24, "compB"): 10.0759, (24, "primary"): 10.0904,
               (32, "compB"): 12.8763, (32, "primary"): 12.8930,
               (40, "compB"): 15.7868, (40, "primary"): 15.7647}


def main_n9() -> int:
    import re as _re
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-ids", default=None)
    ap.add_argument("--only-k", default=None, help="comma list, e.g. 16,24 -- the runway cut")
    args, _ = ap.parse_known_args([a for a in sys.argv[1:] if a != "n9"])
    ks_allowed = ([int(x) for x in args.only_k.split(",")] if args.only_k else list(SWEEP_K))
    os.makedirs(OUT_N9, exist_ok=True)
    written, n = [], N9_ID_START
    for k in SWEEP_K:                                   # longest-first (sec 8.3)
        if k not in ks_allowed:
            n += 6
            continue
        for recipe in ("primary", "compB"):
            for seed in N9_SEEDS:
                sp = spec(f"{n:04d}_ncr_scaleaxis_1b_n9_K{k}_{recipe}_s{seed}",
                          k, recipe, seed, "sweep")
                old = f"scaleaxis1310m_K{k}_{recipe}_s{seed}"
                new = f"scaleaxis1310m_n9_K{k}_{recipe}_s{seed}"
                for f in ("cmd", "validity_check"):
                    sp[f] = sp[f].replace(old, new)
                meas = MEASURED_N6[(k, recipe)]
                ceil_v = round(CEILING_MULT * meas, 3)
                sp["cmd"] = _re.sub(r"--ceiling-gpuh [0-9.]+", f"--ceiling-gpuh {ceil_v}", sp["cmd"])
                sp["gpu_h_estimate"] = round(meas, 3)
                sp["tier"] = "scaleaxis1b_n9"
                sp["scaleaxis1b"]["gpu_h_estimate"] = round(meas, 3)
                sp["scaleaxis1b"]["ceiling_gpuh"] = ceil_v
                sp["scaleaxis1b"]["ceiling_provenance"] = (
                    f"{CEILING_MULT} x the MEASURED mean gpu_h of this exact (K, recipe) cell "
                    f"over the three n=6 thickening seeds ({meas:.4f} GPU-h, spread "
                    f"<0.06 GPU-h) = {ceil_v}. Measured, not projected, not estimated.")
                sp["scaleaxis1b"]["n9"] = {
                    "directive": "PI 2026-08-29 -- the grant's last training payload",
                    "takes_n_from": 6, "takes_n_to": 9,
                    "within_K_pairs_from": 36, "within_K_pairs_to": 81,
                    "sibling_seeds_of_record": [0, 1, 2, 3, 4, 5],
                    "priced_from": "MEASURED gpu_h of cells 0400-0423 (24/24 COMPLETED)",
                    "changes_no_verdict": ("same bands, pinned before any 1B number existed; "
                                           "this wave adds precision only. A pinned verdict "
                                           "FLIPPING at n=9 is a FINDING with its own entry."),
                }
                sp["notes"] = ("CANDIDATE -- NOT queue-eligible. 1B n=9 SURPLUS WAVE (PI "
                               "2026-08-29), the grant's LAST TRAINING PAYLOAD. "
                               "Byte-pattern-identical to the audited 0370-0393 / 0400-0423 "
                               "specs modulo the seed integer, the id and the cell id; the "
                               "ONLY other change is the ceiling, now priced from MEASURED "
                               "sibling cost rather than the probe projection. " + sp["notes"])
                written.append(sp)
                n += 1
    ids = [x["id"][:4] for x in written]
    assert len(set(ids)) == len(ids)
    if args.queue_ids:
        ex = {l.strip()[:4] for l in open(args.queue_ids) if l.strip()}
        cl = sorted(set(ids) & ex)
        assert not cl, f"ID COLLISION: {cl}"
    for x in written:
        json.dump(x, open(os.path.join(OUT_N9, x["id"] + ".json"), "w"), indent=1)
    tot = sum(x["gpu_h_estimate"] for x in written)
    print(f"wrote {len(written)} CANDIDATE 1B n=9 specs to {OUT_N9} ({ids[0]}-{ids[-1]})")
    print(f"  MEASURED ledger: {tot:.2f} GPU-h over {len(written)} cells")
    print("QUEUE-ELIGIBLE: NO.")
    return 0


# ==========================================================================
# GRACE WAVE (PI 2026-08-29). K=16 ONLY at 1.31B, seeds 9-16, both recipes.
# PRIMARY SORTS FIRST BY CONSTRUCTION: the breach lives on the FROZEN arm and
# the box's remaining life is unknown, so primary takes the lower id block and
# whatever completes, completes. New entry point -- nothing renumbers.
# ==========================================================================
GRACE_SEEDS = tuple(range(9, 17))          # 9..16 inclusive = 8 seeds
GRACE_K = 16
GRACE_ID_START = 470                       # 0470-0477 primary, 0478-0485 compB
OUT_GRACE = os.path.join(_HERE, "job_specs_1b_grace")
GRACE_PREREG = (
    "PRE-REGISTRATION, carried in every cell of this wave and fixed BEFORE any of it runs: "
    "THIS WAVE CHANGES NO ADJUDICATED VERDICT. Every band it touches was pinned and adjudicated "
    "at n<=9 and none is reopened here. Its ONE question is the K=16 / 1.31B h=1 WALL-BREACH "
    "RATE on the FROZEN (primary) arm -- the fraction of scored seeds whose P0 reading at h=1 "
    "sits above the per-K binomial band [0.0171, 0.1079] around chance = 1/16 = 0.0625. That "
    "rate currently stands at 2 of the scored primary seeds. DELIVERABLE: the breach rate with "
    "an EXACT (Clopper-Pearson) binomial confidence interval at WHATEVER n LANDS -- the box may "
    "die mid-wave and that is priced in, so the estimator must be valid at any n, which an exact "
    "interval is and a normal approximation is not. The SAME read is reported DESCRIPTIVELY on "
    "compB. Single-seed excursions remain subject to KSCALING sec 7.2's re-measure clause (base "
    "seed 31337) before any individual cell is called a breach; the RATE is over cells that "
    "survive that clause. If this wave's rate moves a previously adjudicated WALL verdict, that "
    "is a FINDING requiring its own EXPERIMENT_LOG entry, never a silent update.")


def main_grace() -> int:
    import re as _re
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-ids", default=None)
    args, _ = ap.parse_known_args([a for a in sys.argv[1:] if a != "grace"])
    os.makedirs(OUT_GRACE, exist_ok=True)
    written, n = [], GRACE_ID_START
    for recipe in ("primary", "compB"):          # primary FIRST -> lower ids -> claimed first
        for seed in GRACE_SEEDS:
            sp = spec(f"{n:04d}_ncr_scaleaxis_1b_grace_K{GRACE_K}_{recipe}_s{seed}",
                      GRACE_K, recipe, seed, "sweep")
            old = f"scaleaxis1310m_K{GRACE_K}_{recipe}_s{seed}"
            new = f"scaleaxis1310m_grace_K{GRACE_K}_{recipe}_s{seed}"
            for f in ("cmd", "validity_check"):
                sp[f] = sp[f].replace(old, new)
            meas = MEASURED_N6[(GRACE_K, recipe)]
            ceil_v = round(CEILING_MULT * meas, 3)
            sp["cmd"] = _re.sub(r"--ceiling-gpuh [0-9.]+", f"--ceiling-gpuh {ceil_v}", sp["cmd"])
            sp["gpu_h_estimate"] = round(meas, 3)
            sp["tier"] = "scaleaxis1b_grace"
            sp["scaleaxis1b"]["gpu_h_estimate"] = round(meas, 3)
            sp["scaleaxis1b"]["ceiling_gpuh"] = ceil_v
            sp["scaleaxis1b"]["ceiling_provenance"] = (
                f"{CEILING_MULT} x the MEASURED mean gpu_h of this exact (K, recipe) cell over "
                f"the n=6 thickening seeds ({meas:.4f} GPU-h) = {ceil_v}. Measured, not projected.")
            sp["scaleaxis1b"]["grace_wave"] = {
                "directive": "PI 2026-08-29 -- grace-period uptime is free; the final wave",
                "question": "K=16 / 1.31B h=1 wall-breach RATE on the frozen arm",
                "band": [0.0171, 0.1079], "chance": 0.0625,
                "breaches_at_dispatch": "2 of the scored primary seeds",
                "deliverable": "breach rate + EXACT (Clopper-Pearson) binomial CI at whatever n lands",
                "compB_role": "same read, DESCRIPTIVE only",
                "primary_sorts_first": True,
                "box_may_die_mid_wave": "priced in; the estimator is valid at any n",
                "changes_no_adjudicated_verdict": True,
                "remeasure_clause": "KSCALING sec 7.2, base seed 31337, before any single cell "
                                    "is called a breach",
            }
            sp["hypothesis"] = sp["hypothesis"] + " " + GRACE_PREREG
            sp["notes"] = ("CANDIDATE -- NOT queue-eligible; the coordinator stages on receipt. "
                           "1.31B GRACE WAVE (PI 2026-08-29), K=16 ONLY, the wall-breach "
                           "stratum. PRIMARY TAKES THE LOWER ID BLOCK BY CONSTRUCTION "
                           "(0470-0477 vs compB 0478-0485) because queue_worker.sh claims by "
                           "`ls | sort`, so filename order is the only lever that decides what "
                           "runs first if the box dies mid-wave -- and the breach lives on the "
                           "frozen arm. Byte-pattern-identical to the audited 0370-0393 / "
                           "0400-0423 / 0440-0463 specs modulo the seed integer, the id and the "
                           "cell id; ceiling priced from MEASURED sibling cost. " + sp["notes"])
            written.append(sp)
            n += 1
    ids = [x["id"][:4] for x in written]
    assert len(written) == 16 and len(set(ids)) == 16, ids
    prim = [x["id"] for x in written if "_primary_" in x["id"]]
    comp = [x["id"] for x in written if "_compB_" in x["id"]]
    assert max(prim) < min(comp), ("primary must sort before compB", prim, comp)
    if args.queue_ids:
        ex = {l.strip()[:4] for l in open(args.queue_ids) if l.strip()}
        cl = sorted(set(ids) & ex)
        assert not cl, f"ID COLLISION: {cl}"
    for x in written:
        json.dump(x, open(os.path.join(OUT_GRACE, x["id"] + ".json"), "w"), indent=1)
    tot = sum(x["gpu_h_estimate"] for x in written)
    print(f"wrote {len(written)} CANDIDATE 1B GRACE specs to {OUT_GRACE} ({ids[0]}-{ids[-1]})")
    print(f"  K=16 only, seeds {GRACE_SEEDS[0]}-{GRACE_SEEDS[-1]}, primary {prim[0]}-{prim[-1]} "
          f"THEN compB {comp[0]}-{comp[-1]}")
    print(f"  MEASURED ledger: {tot:.2f} GPU-h; makespan on 8 GPUs ~{2*7.815:.2f} h")
    print("QUEUE-ELIGIBLE: NO.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-ids", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    written, n = [], 360
    for recipe in ("primary", "compB"):
        written.append(spec(f"{n:04d}_ncr_scaleaxis_1b_calib_K{CALIB_K}_{recipe}_s0",
                            CALIB_K, recipe, 0, "calibration"))
        n += 1
    n = 370
    for k in SWEEP_K:
        for recipe in ("primary", "compB"):
            for seed in SEEDS:
                written.append(spec(f"{n:04d}_ncr_scaleaxis_1b_K{k}_{recipe}_s{seed}",
                                    k, recipe, seed, "sweep"))
                n += 1
    ids = [s["id"][:4] for s in written]
    assert len(written) == 26 and len(set(ids)) == 26, ids
    assert ids[:2] == ["0360", "0361"] and ids[2:] == [f"{i:04d}" for i in range(370, 394)]
    if args.queue_ids:
        existing = {ln.strip()[:4] for ln in open(args.queue_ids) if ln.strip()}
        clash = sorted(set(ids) & existing)
        assert not clash, f"ID COLLISION: {clash}"
    for s in written:
        with open(os.path.join(OUT, s["id"] + ".json"), "w") as f:
            json.dump(s, f, indent=1)
    cal = sum(s["gpu_h_estimate"] for s in written if "calib" in s["id"])
    swp = sum(s["gpu_h_estimate"] for s in written if "calib" not in s["id"])
    print(f"wrote {len(written)} CANDIDATE 1.31B specs to {OUT}")
    print(f"  calibration pair 0360-0361: {cal:.2f} GPU-h")
    print(f"  24-cell sweep    0370-0393: {swp:.2f} GPU-h  (longest-first K=40->32->24->16)")
    print(f"  TOTAL {cal+swp:.2f} GPU-h; wall on 8 GPUs ~{(cal+swp)/8:.1f} h")
    print("QUEUE-ELIGIBLE: NO.")
    return 0


if __name__ == "__main__":
    sys.exit(main_grace() if "grace" in sys.argv[1:] else
             main_n9() if "n9" in sys.argv[1:] else
             main_thicken() if "thicken" in sys.argv[1:] else main())
