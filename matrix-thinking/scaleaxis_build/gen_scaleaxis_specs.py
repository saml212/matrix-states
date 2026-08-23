#!/usr/bin/env python3
"""Generate the 24 SCALE-AXIS job specs -- NCR_SCALE_AXIS_DESIGN.md DRAFT-R2
sec 4.1 (the calibration SEXTET), sec 4.5 (the sweep) and sec 8.3 (SPEC NAMING
IS THE SCHEDULE CONTROL).

  calibration sextet   K=24 x {primary, compB} x seeds 0,1,2   ids 0190-0195
  sweep                K in {40, 32, 16} x 2 x 3               ids 0200-0217

NOTHING HERE IS QUEUE-ELIGIBLE.  Every spec is CANDIDATE-marked. The sextet is
double-gated on (a) the build audit and (b) Stage A0's Rules P1-P4 having been
applied to MEASURED numbers; the 18 sweep cells are triple-gated, adding (c)
the LICENSE_SWEEP_SCALEAXIS sentinel from sec 7.2 branch (D).

SPEC NAMING IS A LOAD-BEARING SCHEDULING DECISION (sec 8.3, verify-R2 MAJOR-3).
queue_worker.sh:119 claims by `for f in $(ls "$PENDING" | sort)` -- dispatch
order is LEXICOGRAPHIC FILENAME ORDER, so filenames ARE the control surface.
R1's pinned shortest-first was the WORST natural order (10.842 h); the design
pins LONGEST-FIRST -- 0200-0205 = K=40, 0206-0211 = K=32, 0212-0217 = K=16 --
for 10.194 h. The 9.021 h mixed order is an ELECT-or-DECLINE the build round
DECLINES here (it buys 1.17 h of wall at the cost of a numbering scheme nobody
can read at a glance, and the worker's own 60 s busy-poll makes 9.03-vs-9.02
spurious precision anyway); `--elect-mixed-order` implements it if elected.

The sextet takes 0190-0195 so it sorts BEFORE every sweep spec: Stage A runs
FIRST AND ALONE and its verdict licenses Stage B, so a sweep spec must never
be claimable while a calibration spec is pending.

ID COLLISION.  Verified on the box 2026-08-22 against ~/queue/{completed,
pending,failed,parked_k24plus}: zero ids in 0190-0217 (the K-scaling wave used
0100-0151). Asserted again at generation time from --queue-ids if supplied.

--ceiling-gpuh (sec 3.6, MAJOR-1(b,c)).  R0's 1.5x-solo rule is REPLACED: it
contradicts the runner's own CONTENDED_MULTIPLIER = 3.3 and would hard-abort
every cell under any contention above 1.5x. The pinned value is
1.5 x (per-cell projection at Stage A0's MEASURED contended rate), falling back
to the runner's own suggested_ceiling_gpuh (3.795 x solo) -- NEVER a hand-picked
number, and never the as-run 6.0 (a landmine above ~5.3x at 392M).
Until Stage A0 has run, specs generated with the default carry
ceiling_provenance = "PROJECTED-NOT-LAUNCH-READY" and are explicitly not
queue-eligible on that ground alone. --ceilings-from <dir of phase0 JSONs>
re-derives them from measurement.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NCR_SCALE", "392m")
os.environ.setdefault("NCR_K", "24")
# The generator reads the SCALEAXIS TREE's own kscaling_config -- never the
# kscaling tree's -- so the ladder table, the sec 3.4 param formulas and the
# scale resolution a spec asserts are the ones the cell will actually run
# under. On the box the tree IS this directory; from the repo mirror, point
# SCALEAXIS_TREE at a freshly patched copy.
_TREE = os.environ.get("SCALEAXIS_TREE", _HERE)
sys.path.insert(0, _TREE)
import kscaling_config as KS   # noqa: E402  (the scaleaxis tree's own copy)
assert KS.SCALE == "392m" and KS.RUNG == 2, (KS.SCALE, KS.RUNG)
assert os.path.dirname(os.path.abspath(KS.__file__)) == os.path.abspath(_TREE), KS.__file__

OUT = os.path.join(_HERE, "job_specs")
PY = "/home/nvidia/tdenv/bin/python3"
WORKDIR = "/home/nvidia/ncr_scaleaxis"
EPH = "/ephemeral/scaleaxis"
SCALE = "392m"

CALIB_K = 24
SWEEP_K_LONGEST_FIRST = (40, 32, 16)      # sec 8.3's pinned dispatch order
SEEDS = (0, 1, 2)

RECIPES = {
    "primary": dict(
        label="FROZEN-contrastive (primary recipe)",
        flags="--aux-loss-type contrastive+cosine --freeze-entity-adapter "
              "--contrastive-temperature 0.07"),
    "compB": dict(
        label="TRAINABLE-contrastive (compB recipe)",
        flags="--aux-loss-type contrastive+cosine --contrastive-temperature 0.07"),
}

# sec 3.5: THE RECIPE IS NOT A VARIABLE HERE EITHER; THE BACKBONE IS.
# Held at the audited K-scaling values, verbatim, EXCEPT --ckpt-every (sec 4.3.2)
# and --ceiling-gpuh (sec 3.6), both of which are stated per tier below.
COMMON = ("--mode calibration --device cuda --steps 20000 --batch-size 32 "
          "--eval-batch-size 64 --warmup-steps 200 --lr 3e-4 "
          "--aux-read-loss-weight 0.5 --ortho-reg-weight 0.1 --eval-every 1000")

# sec 8.2's MEASURED 98M per-cell gpu_h (means over the archived cells' own
# gpu_h fields -- NOT the EXPERIMENT_LOG headline projections).
GPU_H_98M_MEASURED = {16: 0.8019, 24: 0.8271, 32: 0.9583, 40: 1.1309}
PROJECTION_MULT = 3.75          # sec 8.2's central column; UNVERIFIED FOR THE GRAFT
CEILING_MULT = 1.5              # sec 3.6 breaker 1: 1.5 x the CONTENDED projection


def gpu_h(k: int, r: float = PROJECTION_MULT) -> float:
    return round(GPU_H_98M_MEASURED[k] * r, 3)


def ceiling(k: int, contended_gpuh: dict | None, r8: float | None = None,
            solo_gpuh: dict | None = None) -> tuple[float, str]:
    # AUDIT-R1 MAJOR-3 / condition C3. sec 3.6 breaker 1 pins
    #   ceiling = 1.5 x (per-cell projection at the MEASURED CONTENDED rate R_8),
    # with 3.795 x solo as the fallback ONLY "if R_8 cannot be measured". The
    # previous "measured" branch used contended_gpuh_for_target_steps, which is
    # phase0-solo x CONTENDED_MULTIPLIER = 3.3 -- the runner's PROJECTION
    # CONSTANT substituted for R_8, shipping a backstop ~2.9x LOOSER than pinned
    # and 32% looser than the PROJECTED placeholder it replaced, so the A0.5
    # "re-price" WEAKENED the breaker. Primary rule first, fallback labelled.
    if r8 is not None and solo_gpuh and k in solo_gpuh:
        return (round(CEILING_MULT * r8 * solo_gpuh[k], 3),
                f"MEASURED (sec 3.6 PRIMARY): 1.5 x R_8 x Stage-A0 solo projection "
                f"(R_8={r8:.4f}, solo={solo_gpuh[k]:.3f} GPU-h)")
    if contended_gpuh and k in contended_gpuh:
        return (round(CEILING_MULT * contended_gpuh[k], 3),
                f"MEASURED-FALLBACK: 1.5 x the runner's own contended projection "
                f"({contended_gpuh[k]:.3f} GPU-h = solo x CONTENDED_MULTIPLIER 3.3). Used "
                f"because R_8 was NOT measured -- run `run_stage_a0.sh contended` for "
                f"sec 3.6's primary rule.")
    # Fallback, sec 3.6: the runner's OWN convention, 3.3 (CONTENDED_MULTIPLIER)
    # x 1.15 = 3.795 x solo. Applied to the PROJECTED solo cost, so it is a
    # projection of a projection and is labelled as such.
    return (round(3.795 * gpu_h(k), 3), "PROJECTED-NOT-LAUNCH-READY: 3.795 x projected solo "
                                        "(the runner's own suggested_ceiling_gpuh convention); "
                                        "MUST be re-derived from Stage A0 before queueing")


HYP = (
    "NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 -- THE 392M SCALE AXIS. Cell: K={K} (d={d}, the PAIR is "
    "the variable), {label}, seed {seed}, backbone RUNG 2 (d_model=1536, d_state=128, "
    "n_layers=16) -- 4.008x the 98M rung-1 cells at MATCHED (K, d), MATCHED recipe and MATCHED "
    "20,000 steps. HYPOTHESIS (sec 1): the capability separation established at 98M -- exact-write "
    "composition reads at ceiling (P1b kappa >= 0.90 at h_top), the model's OWN learned writes "
    "pinned at chance (P0 in the per-K binomial band), and the frozen-over-trainable ordering "
    "emerging at depth -- is SCALE-STABLE at 392M across K in {{16,24,32,40}}. Falsifiable in "
    "BOTH directions and both directions are the same paper (sec 1): SCALE-DEGRADES is a measured "
    "negative slope in a scaling law for an exact-composition capability; SCALE-IMPROVES is a "
    "positive one; SCALE-STABLE says the separation is a property of the mechanism, not of the "
    "operating point. THE TOKEN-BUDGET CONFOUND IS ONE-DIRECTIONAL AND THAT IS A STRENGTH: steps, "
    "batch and t_in are held fixed, so this cell sees the SAME tokens as its 98M twin while "
    "carrying 4x the parameters (D/N falls 1.87 -> 0.47 tokens/param at K=40). Under-training can "
    "only manufacture DEGRADES; it cannot manufacture STABLE and it cannot manufacture IMPROVES. "
    "Accordingly the step-extension attribution arm (2 cells @ 40,000 steps at the degrading K) is "
    "MANDATORY BEFORE ANY PUBLISHED SCALE-DEGRADES VERDICT AT ANY K (sec 7.2, MAJOR-4). "
    "Primary readout: P1b kappa at h_top={htop} (residue {htopres} = K/2, the ANTIPODAL point of "
    "the K-cycle), matched pools, n=256, base seed 90210, ckpt_step == 20000. Both regimes scored "
    "on every cell: P1b = EXACT-WRITE (teacher-forced operator; the READ is under test) = the "
    "CAPABILITY curve; P0 = LEARNED-WRITE = the WALL curve, predicted inside the per-K binomial "
    "band around chance = 1/{K} = {chance:.4f} (band {band}). Ladder {ladder}, residues {res} -- "
    "6 distinct, none 0, none in the train residues {{1,2,3}}, squaring profile (2,3,4,4,5,5) "
    "IDENTICAL to 98M, so the cross-scale comparison is instrument-matched; the ladder is a "
    "number-theoretic function of K alone and is UNCHANGED by the port (sec 3.5). "
    "Fixed-effective-distance control h_fix={hfix} (residue 4 at every K, same squaring count as "
    "h_top). PARAM COUNT {params:,}/arm, from sec 3.4's formula VALIDATED AGAINST FOUR MEASURED "
    "ENDPOINTS; the K=16..40 spread at 392M is 0.0204% (vs 0.051% at 98M) and the scale ratio is "
    "4.008x uniform across K to four significant figures -- so this is a param-matched curve at "
    "both scales AND at matched parameter RATIO, not a capacity curve in disguise. t_in={tin} "
    "(doc_left_pad={pad}). 98M REFERENCE FOR THIS CELL, frozen before any 392M cell existed "
    "(sec 2.1): P1b kappa@h_top median {ref98:.4f} across seeds. Equivalence margin delta=0.05 "
    "(sec 5.2), which exceeds the largest within-(K,recipe) seed range of record (0.0292). "
    "{tier_note}")

CAL_TIER_NOTE = (
    "THIS CELL IS ONE OF THE SIX CALIBRATION CELLS (sec 4.1, election 2: the FULL K=24 SEXTET, "
    "adopted). K=24 is elected for three pre-stated reasons: it is the CENTRE of the ported range "
    "so the license generalizes in both directions; it has the largest 98M evidence base (the "
    "55-cell g3b31 family plus the 6-cell anchor re-score) so a 392M anomaly there is maximally "
    "diagnosable; and its t_in=174/pad=0 make it BYTE-IDENTICAL to the pinned document "
    "construction, so a calibration failure cannot be a padding artifact. The six cells ARE six "
    "of the 24 -- the whole K=24 stratum -- so they are NOT extra cost and there is no "
    "conditioned-vs-unconditioned split to report (sec 4.1 m8, dissolved). LICENSE-SWEEP requires "
    "all three legs on the FROZEN cells: (1) Gate-0 convergence on ALL THREE frozen seeds; "
    "(2) P1b kappa >= 0.90 at train hops h in {1,2,3} on >= 2/3 frozen seeds; (3) P1b kappa >= "
    "0.90 at h_top(24)=36 on >= 2/3 frozen seeds. The TRAINABLE cells do NOT gate -- they price "
    "the trainable recipe at 392M and give sec 5.3's tests their K=24 stratum early. "
    "--ckpt-every 5000 (not the as-run 10000) is LOAD-BEARING: it is the ONLY instrument for the "
    "P1b kappa trajectory branch (B) reads (FATAL-3: there is NO in-run P1b kappa at any step, at "
    "any scale, in this harness -- the eval record is overwritten, withheld from stdout, and P0-"
    "regime anyway). kappa_reader.py (B7) hardlinks the single ckpt path at each write, reads its "
    "step, and invokes the battery at that --required-step. Cost of the extra writes: ~0.7% of a "
    "cell, disclosed so the re-price is not read as graft overhead.")

SWEEP_TIER_NOTE = (
    "THIS CELL IS ONE OF THE 18 SWEEP CELLS (sec 4.5). K in {12,20,28,36} are deliberately NOT "
    "ported: given FATAL-2 the binding constraint on this design is READOUT HEADROOM, not K "
    "resolution (election 4, ratified), and sec 4.6's depth extension is what buys the headroom. "
    "The four chosen K include the smallest ported (16, the only one with a non-zero pad) and the "
    "largest of record (40 -- K=44 is construction-impossible: an antipodal top rung needs "
    "3K/2 <= 63). HARD PRE-SWEEP GATE ON K=16 (sec 4.5, sec 3.2 item 19): no K=16 spec may be "
    "queued until Stage A0.2 has shown, at the 392M mixer config, that chunk_delta_rule's backward "
    "floor is still <= 128 -- K=16's t_in is EXACTLY 128, zero margin, and MIN_KERNEL_T = 128 was "
    "MEASURED AT d_state = 64 ONLY.")

NOTES = (
    "CANDIDATE -- NOT queue-eligible. {gates} "
    "Built by the 392M SCALE-AXIS BUILD agent 2026-08-22 against repo commit 331f8d7 and design "
    "NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 (commit 2aca91d, DESIGN PHASE CLOSED after 5 gauntlet "
    "rounds). PATCHED COPIES in /home/nvidia/ncr_scaleaxis, produced by patch_scaleaxis.py from "
    "md5-PINNED sources (K-scaling patched graft 74ee84fc920b024901d11add66cc5c2d, K-scaling "
    "patched runner ee5833743049e1bb1864124ad5d3fbf6, kscaling_config "
    "eaddd0411fd1cdaaa6028735023c1b99, battery of record 5735c788563d9a21f2198c9f5b4793d5, "
    "depthext_eval e95ffe192c66d3e054f3febc37fe4a91). BUILD REQUIREMENT B4 is discharged: the "
    "GRAFT md5 is now a hard-pinned constant (patch_kscaling.py pinned only the runner and the "
    "graft md5 lived in gen_job_specs.py PROSE), with a proven-teeth negative test. The ULTIMATE "
    "pinned originals in ~/ncr_g3b31_contrastive are UNTOUCHED and re-verified (runner "
    "9a93198b642242f512ff8489e32b0a53, graft bc105af69661e488ff95f5046e2bcd8a). "
    "THE PORT IS ONE DICT (sec 3.1): RUNG1_BACKBONE 768/64/12 -> 1536/128/16, resolved from "
    "kscaling_config's RUNGS table via the mandatory NCR_SCALE env var; BACKBONE_PARAM_TARGET "
    "moves with it (98e6 -> 392e6, sec 3.2 item 17) or the 15% gate fires on a CORRECT build. "
    "NCR_K, --k and --scale are all present and the runner asserts all three against the RESOLVED "
    "backbone dict before any GPU work. RUNNER_TAG = ncr_scaleaxis_runner_v1, so a 392M checkpoint "
    "can never be silently resumed by or confused with a 98M cell. B5's scale guard is in BOTH "
    "scorers with a proven-teeth negative test (a 98M checkpoint under the 392M config is "
    "REFUSED), because restore_arms_and_opts rebuilds from ckpt['backbone_config'], NOT from "
    "RUNG1_BACKBONE, so a wrong-SCALE checkpoint at the right K would otherwise load and score "
    "SILENTLY. Per-cell isolation is the queue's own -- one cell per spec, a failure routes to "
    "failed/ and is not auto-retried. Checkpoints go to /ephemeral (NEVER the root fs). Placement "
    "1 cell/GPU per the standing declined-packing ruling. Scored afterwards by kscaling_battery.py "
    "(matched pools from the checkpoint's OWN recorded seed; refuses unless recorded d_ncr == K+1 "
    "AND the recorded backbone matches this scale). CEILING PROVENANCE: {ceilprov}")

GATES_CAL = ("DOUBLE-GATED: (a) the build audit passes, AND (b) Stage A0's Rules P1-P4 (sec 4.4) "
             "have been applied to MEASURED numbers -- R = phase0(392M)/phase0(98M) at K=24 AND "
             "K=40 (like-for-like: verify-R2 FATAL-1 showed R1's cross-instrument ratio was "
             "1.5500x inflated and would have aborted the wave across its ENTIRE predicted "
             "range), R_8 from 8 concurrent probes, the P4 memory reading and the SM sampler. "
             "R > 4.5 => COST-OUT to sec 4.4.1's publishable K=24 FROZEN TRIO. R_8 > 1.25 => do "
             "not queue wave 1.")
GATES_SWEEP = (GATES_CAL[:-1] + ", AND (c) the calibration sextet returns LICENSE-SWEEP "
               "(sec 7.2 branch (D), the LICENSE_SWEEP_SCALEAXIS sentinel).")


def validity(out_json: str, k: int, params: int) -> str:
    """status/step + K identity + SCALE identity + the sec 3.4 PARAM COUNT +
    the Gate-0 loss_history clause (the AUDIT_R2 L1 form: control arm logged,
    UNGATED). A mislabelled, wrong-scale or non-converged cell fails its OWN
    validity check and routes to failed/ instead of entering a curve."""
    lad = list(KS.LADDER_TABLE[k])
    return (
        f"{PY} -c \"import json, math; d=json.load(open('{out_json}')); "
        f"assert d.get('status')=='COMPLETED', d.get('status'); "
        f"assert d.get('step',0)>=20000, d.get('step'); "
        f"assert d.get('runner_tag')=='ncr_scaleaxis_runner_v1', d.get('runner_tag'); "
        f"ks=d.get('kscaling') or {{}}; "
        f"assert ks.get('K')=={k}, ('K', ks.get('K')); "
        f"assert ks.get('d_ncr')=={k+1}, ('d_ncr', ks.get('d_ncr')); "
        f"assert ks.get('d_equals_k_plus_1') is True; "
        f"assert ks.get('h_top')=={lad[-1]}, ('h_top', ks.get('h_top')); "
        f"assert ks.get('deep_ladder')=={lad}, ks.get('deep_ladder'); "
        f"assert ks.get('scale')=='{SCALE}', ('scale', ks.get('scale')); "
        f"bb=ks.get('backbone') or {{}}; "
        f"assert (bb.get('d_model'), bb.get('d_state'), bb.get('n_layers'))==(1536,128,16), bb; "
        f"pp=d.get('params') or {{}}; "
        f"assert pp.get('per_arm')=={params}, ('PARAM COUNT vs design sec 3.4', pp.get('per_arm')); "
        f"lh=d.get('loss_history') or {{}}; "
        f"assert set(lh)=={{'full_graft','backbone_only'}}, sorted(lh); "
        f"assert len(lh['backbone_only'])>=100, len(lh['backbone_only']); "
        f"h=lh['full_graft']; assert len(h)>=100, len(h); "
        f"assert all(math.isfinite(r[1]) for r in h), 'non-finite CE in loss_history'; "
        f"assert h[-1][1] < h[0][1], ('GATE-0 NOT CONVERGED', h[0], h[-1])\"")


REF98_KAPPA_HTOP = {          # sec 2.1 CURVE 1 medians, frozen before any 392M cell existed
    (16, "primary"): 1.0000, (16, "compB"): 0.9958,
    (24, "primary"): 1.0000, (24, "compB"): 0.9878,
    (32, "primary"): 0.9960, (32, "compB"): 0.9919,
    (40, "primary"): 0.9920, (40, "compB"): 0.9880,
}
WALL_BAND = {16: [0.0171, 0.1079], 24: [0.0042, 0.0791],
             32: [0.0000, 0.0639], 40: [0.0000, 0.0543]}


def spec(job_id: str, k: int, recipe: str, seed: int, tier: str,
         contended: dict | None, r8: float | None = None,
         solo_gpuh: dict | None = None) -> dict:
    d, r = k + 1, RECIPES[recipe]
    cell = f"scaleaxis392m_K{k}_{recipe}_s{seed}"
    outdir = f"{EPH}/results"
    out_json = f"{outdir}/{cell}.json"
    lad = KS.LADDER_TABLE[k]
    params = KS.TOTAL_PARAM_TABLE_392M[k]
    ceil_v, ceil_prov = ceiling(k, contended, r8, solo_gpuh)
    ckpt_every = 5000 if tier == "calibration" else 10000
    fields = dict(K=k, d=d, label=r["label"], seed=seed, chance=1.0 / k,
                  htop=lad[-1], htopres=lad[-1] % k, hfix=KS.FIXED_DIST_TABLE[k],
                  ladder=list(lad), res=[h % k for h in lad], band=WALL_BAND[k],
                  tin=KS.t_in(k), pad=KS.doc_left_pad(k), params=params,
                  ref98=REF98_KAPPA_HTOP[(k, recipe)],
                  tier_note=CAL_TIER_NOTE if tier == "calibration" else SWEEP_TIER_NOTE)
    cmd = (f"mkdir -p {outdir} {EPH}/ckpts && cd {WORKDIR} && "
           f"NCR_SCALE={SCALE} NCR_K={k} {PY} ncr_lm_wave1_runner.py "
           f"--k {k} --scale {SCALE} {COMMON} --ckpt-every {ckpt_every} "
           f"--ceiling-gpuh {ceil_v} {r['flags']} "
           f"--seed {seed} --cell-id {cell} --out {out_json} "
           f"--ckpt-dir {EPH}/ckpts/{cell} 2>&1 | tee -a {outdir}/{cell}.log")
    return {
        "id": job_id, "lane": "NCR", "tier": f"scaleaxis_{tier}",
        "hypothesis": HYP.format(**fields), "cmd": cmd,
        "gpu_h_estimate": gpu_h(k), "output_dir": outdir,
        "validity_check": validity(out_json, k, params),
        "notes": NOTES.format(gates=(GATES_CAL if tier == "calibration" else GATES_SWEEP),
                              ceilprov=ceil_prov),
        "scaleaxis": {
            "scale": SCALE, "rung": 2, "K": k, "d_ncr": d, "recipe": recipe, "seed": seed,
            "params_per_arm_design_sec3_4": params,
            "params_ratio_to_98m": round(params / KS.TOTAL_PARAM_TABLE_98M[k], 6),
            "ckpt_every": ckpt_every, "ceiling_gpuh": ceil_v,
            "ceiling_provenance": ceil_prov,
            "gpu_h_projection_basis": f"sec 8.2 measured 98M {GPU_H_98M_MEASURED[k]} x "
                                      f"{PROJECTION_MULT} (UNVERIFIED FOR THE GRAFT -- "
                                      f"Stage A0 re-prices from measured R)",
            "queue_eligible": False,
            "gate": ("build audit + Stage A0 Rules P1-P4" if tier == "calibration" else
                     "build audit + Stage A0 Rules P1-P4 + LICENSE_SWEEP_SCALEAXIS"),
        },
    }


# ==========================================================================
# SEED THICKENING (PI directive 2026-08-23): seeds 3,4,5 at every (K, recipe).
# A SEPARATE entry point, exactly as gen_job_specs.py separates main_frontier()
# -- "so that generating the frontier wave cannot rewrite, resurrect or
# renumber a spec belonging to the completed wave". main() and validity() are
# UNTOUCHED, so the as-run 0190-0217 still regenerate byte-identically.
#
# NO NEW DESIGN. Same generator, same recipe, same ladders, same bands, same
# param assertions -- only the seed integer, the id block, the output dir and
# the (now MEASURED) ceiling differ. This wave changes NO verdict: the bands
# were pinned and adjudicated at n=3. It adds PRECISION -- n=3 -> n=6 per cell,
# and the TEST-W ordering strata from 9 within-K pairs to 36. If any pinned
# verdict FLIPS at n=6 that is a FINDING requiring its own log entry, never a
# silent update.
# ==========================================================================
THICKEN_SEEDS = (3, 4, 5)
THICKEN_ID_START = 300
OUT_THICKEN = os.path.join(_HERE, "job_specs_thicken")


def measured_20k(sweep_dir: str) -> dict:
    """MEASURED per-(K, recipe) 20,000-step cost from the sweep of record.
    The thickening cells are the SAME configuration as their n=3 siblings, so
    their own siblings are the right price -- no projection is involved."""
    import collections
    if not os.path.isdir(sweep_dir):
        raise SystemExit(f"--sweep-dir {sweep_dir!r} does not exist; the thickening wave is "
                         f"priced from the MEASURED cost of its own n=3 siblings.")
    agg = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(sweep_dir, "*.json"))):
        d = json.load(open(p))
        if d.get("mode") != "calibration" or d.get("status") != "COMPLETED":
            continue
        ks = d.get("kscaling") or {}
        r = "primary" if d["config"]["freeze_entity_adapter"] else "compB"
        agg[(int(ks["K"]), r)].append(d.get("gpu_h") or (d.get("elapsed_s") or 0) / 3600.0)
    assert len(agg) == 8, f"expected all 8 (K, recipe) cells in {sweep_dir}, got {sorted(agg)}"
    return {k: sum(v) / len(v) for k, v in agg.items()}


def validity_thicken(out_json: str, k: int, params: int) -> str:
    """validity() plus the TOP-LEVEL steps_target clause. EXPERIMENT_LOG
    2026-08-23 #1: the runner records steps_target at the TOP LEVEL of its
    record, not under `config`; a checker that looked under config read None
    and flagged four flawless cells as failed."""
    base = validity(out_json, k, params)
    return base[:-len('"')] + (
        f"; assert d.get('steps_target')==20000, ('steps_target (TOP-LEVEL, EXPERIMENT_LOG "
        f"2026-08-23 #1)', d.get('steps_target'))\"")


def main_thicken() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--queue-ids", default=None)
    args, _ = ap.parse_known_args([a for a in sys.argv[1:] if a != "thicken"])
    real = measured_20k(args.sweep_dir)
    os.makedirs(OUT_THICKEN, exist_ok=True)

    written, n = [], THICKEN_ID_START
    # sec 8.3's pinned dispatch order: longest-first, K=40 -> 32 -> 24 -> 16.
    for k in (40, 32, 24, 16):
        for recipe in ("primary", "compB"):
            for seed in THICKEN_SEEDS:
                sp = spec(f"{n:04d}_ncr_scaleaxis_392m_thicken_K{k}_{recipe}_s{seed}",
                          k, recipe, seed, "sweep", None)
                marg = real[(k, recipe)]
                ceil_v = round(CEILING_MULT * 1.0026 * marg, 3)
                ceilprov = (f"MEASURED (sec 3.6 PRIMARY): 1.5 x R_8 x this cell's OWN measured "
                            f"20,000-step cost ({marg:.4f} GPU-h, mean of its three n=3 "
                            f"siblings; R_8 = 1.0026 measured at Stage A0.4) = {ceil_v} GPU-h")
                import re as _re
                sp["cmd"] = _re.sub(r"--ceiling-gpuh [0-9.]+", f"--ceiling-gpuh {ceil_v}",
                                    sp["cmd"])
                sp["cmd"] = sp["cmd"].replace(f"scaleaxis392m_K{k}_{recipe}_s{seed}",
                                              f"scaleaxis392m_thicken_K{k}_{recipe}_s{seed}")
                cell = f"scaleaxis392m_thicken_K{k}_{recipe}_s{seed}"
                sp["validity_check"] = validity_thicken(
                    f"{EPH}/results/{cell}.json", k, KS.TOTAL_PARAM_TABLE_392M[k])
                sp["gpu_h_estimate"] = round(marg, 3)
                sp["tier"] = "scaleaxis_thicken"
                sp["scaleaxis"]["ceiling_gpuh"] = ceil_v
                sp["scaleaxis"]["ceiling_provenance"] = ceilprov
                sp["scaleaxis"]["gpu_h_projection_basis"] = (
                    f"MEASURED sibling cost {marg:.4f} GPU-h (not a projection)")
                sp["scaleaxis"]["thickening"] = {
                    "directive": "PI 2026-08-23 -- seed thickening of the flagship's central "
                                 "tables; no new design, audited generator, new seeds only",
                    "takes_n_from": 3, "takes_n_to": 6,
                    "TEST_W_pairs_from": 9, "TEST_W_pairs_to": 36,
                    "changes_no_verdict": ("bands were pinned and adjudicated at n=3; this wave "
                                           "adds precision (CIs, LOSO robustness) and is reported "
                                           "as n=6 updates to the SAME tables. A pinned verdict "
                                           "FLIPPING at n=6 is a FINDING requiring its own "
                                           "EXPERIMENT_LOG entry, never a silent update."),
                    "sibling_seeds_of_record": [0, 1, 2],
                }
                sp["notes"] = ("CANDIDATE -- NOT queue-eligible; the coordinator stages on the "
                               "build receipt. SEED-THICKENING wave (PI 2026-08-23): "
                               "byte-pattern-identical to the audited sweep specs 0200-0217 "
                               "modulo the seed integer, the id, the cell id and the MEASURED "
                               "ceiling. No new design; the audited generator IS the machinery. "
                               + sp["notes"].split("Built by the 392M SCALE-AXIS BUILD agent", 1)[1]
                               if "Built by the 392M SCALE-AXIS BUILD agent" in sp["notes"]
                               else sp["notes"])
                written.append(sp)
                n += 1

    ids = [s["id"][:4] for s in written]
    assert len(written) == 24 and len(set(ids)) == 24, ids
    assert ids == [f"{i:04d}" for i in range(THICKEN_ID_START, THICKEN_ID_START + 24)], ids
    if args.queue_ids:
        existing = {ln.strip()[:4] for ln in open(args.queue_ids) if ln.strip()}
        clash = sorted(set(ids) & existing)
        assert not clash, f"ID COLLISION with queue history: {clash}"
    for s in written:
        with open(os.path.join(OUT_THICKEN, s["id"] + ".json"), "w") as f:
            json.dump(s, f, indent=1)
    tot = sum(s["gpu_h_estimate"] for s in written)
    print(f"wrote {len(written)} CANDIDATE thickening specs to {OUT_THICKEN} "
          f"({ids[0]}-{ids[-1]})")
    print(f"  K in (40,32,24,16) x (primary, compB) x seeds {list(THICKEN_SEEDS)}")
    print(f"  MEASURED ledger: {tot:.2f} GPU-h; dispatch order longest-first")
    print("QUEUE-ELIGIBLE: NO.")
    return 0


def main() -> int:
    """WARNING: writes to job_specs/, which holds the AS-RUN artifacts of the
    0190-0217 wave -- including their Stage-A0 RE-PRICED ceilings, applied via
    --ceilings-from at launch (condition C9). Re-running this WITHOUT
    --ceilings-from reverts those specs to the PROJECTED placeholder and
    destroys the as-run record. The seed-thickening wave therefore uses
    main_thicken(), which writes to its own directory and never touches this
    one -- the same separation gen_job_specs.py enforces for its frontier
    wave."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceilings-from", default=None,
                    help="dir of Stage-A0 phase0-timing JSONs; re-derives every --ceiling-gpuh "
                         "from the MEASURED contended projection")
    ap.add_argument("--queue-ids", default=None,
                    help="file of existing queue ids (one per line) for the collision assert")
    ap.add_argument("--elect-mixed-order", action="store_true",
                    help="sec 8.3's 9.02 h ELECT-or-DECLINE; DECLINED by default")
    args = ap.parse_args()

    contended, r8, solo_gpuh = None, None, None
    if args.ceilings_from:
        contended, solo_gpuh, eight, solo_s = {}, {}, [], {}
        for p in sorted(glob.glob(os.path.join(args.ceilings_from, "*.json"))):
            d = json.load(open(p))
            if d.get("mode") != "phase0-timing":
                continue
            if (d.get("kscaling") or {}).get("scale") != SCALE:
                continue
            k = int(d["config"]["K"])
            v = float(d["measured"]["mean_s_per_step_both_arms_combined"])
            if "8way" in os.path.basename(p):
                eight.append(v)
                continue
            solo_s[k] = v
            contended[k] = float(d["projected"]["contended_gpuh_for_target_steps"])
            solo_gpuh[k] = float(d["projected"]["uncontended_gpuh_for_target_steps"])
        assert contended, f"no 392M solo phase0-timing records found in {args.ceilings_from}"
        # C3: R_8 = 8-way / solo at the SAME K (24), the like-for-like ratio.
        if eight and 24 in solo_s:
            r8 = (sum(eight) / len(eight)) / solo_s[24]
            print(f"C3: measured R_8 = {r8:.4f} from {len(eight)} 8-way probes "
                  f"-- using sec 3.6's PRIMARY ceiling rule")
        else:
            print("C3: R_8 NOT measured -- falling back to the runner's 3.3x contended "
                  "projection, and every spec says so in ceiling_provenance")
        # Rule P3's per-K price is the ledger basis; solo_gpuh keeps the ceiling on
        # the SAME base the projection uses, so the two cannot drift apart.
        for k in SWEEP_K_LONGEST_FIRST + (CALIB_K,):
            if k not in solo_gpuh and 24 in solo_gpuh:
                solo_gpuh[k] = solo_gpuh[24] * GPU_H_98M_MEASURED[k] / GPU_H_98M_MEASURED[24]

    os.makedirs(OUT, exist_ok=True)
    written = []

    # --- Stage A: the calibration SEXTET, ids 0190-0195 (sorts FIRST) -------
    n = 190
    for recipe in ("primary", "compB"):
        for seed in SEEDS:
            written.append(spec(f"{n:04d}_ncr_scaleaxis_392m_calib_K{CALIB_K}_{recipe}_s{seed}",
                                CALIB_K, recipe, seed, "calibration", contended, r8, solo_gpuh))
            n += 1

    # --- Stage B: 18 cells, LONGEST-FIRST naming (sec 8.3) ------------------
    n = 200
    for k in SWEEP_K_LONGEST_FIRST:                 # 40, 32, 16
        for recipe in ("primary", "compB"):
            for seed in SEEDS:
                written.append(spec(f"{n:04d}_ncr_scaleaxis_392m_K{k}_{recipe}_s{seed}",
                                    k, recipe, seed, "sweep", contended, r8, solo_gpuh))
                n += 1

    ids = [s["id"][:4] for s in written]
    assert len(written) == 24 and len(set(ids)) == 24, ids
    assert ids[:6] == ["0190", "0191", "0192", "0193", "0194", "0195"], ids[:6]
    assert ids[6:] == [f"{i:04d}" for i in range(200, 218)], ids[6:]
    # sec 8.3: `ls | sort` must yield K=40 first, then K=32, then K=16.
    order = [s["scaleaxis"]["K"] for s in sorted(written, key=lambda s: s["id"])[6:]]
    assert order == [40] * 6 + [32] * 6 + [16] * 6, order
    if args.queue_ids:
        existing = {ln.strip()[:4] for ln in open(args.queue_ids) if ln.strip()}
        clash = sorted(set(ids) & existing)
        assert not clash, f"ID COLLISION with queue history: {clash}"

    for s in written:
        with open(os.path.join(OUT, s["id"] + ".json"), "w") as f:
            json.dump(s, f, indent=1)

    cal = [s for s in written if s["tier"].endswith("calibration")]
    swp = [s for s in written if s["tier"].endswith("sweep")]
    print(f"wrote {len(written)} CANDIDATE specs to {OUT} "
          f"({len(cal)} calibration sextet + {len(swp)} sweep)")
    print(f"dispatch order (ls | sort), sweep tier: "
          f"{[s['scaleaxis']['K'] for s in sorted(swp, key=lambda x: x['id'])]}")
    print(f"ledger @x{PROJECTION_MULT}: calibration "
          f"{sum(s['gpu_h_estimate'] for s in cal):.2f} + sweep "
          f"{sum(s['gpu_h_estimate'] for s in swp):.2f} = "
          f"{sum(s['gpu_h_estimate'] for s in written):.2f} GPU-h (design sec 8.2: 83.7)")
    print(f"ceiling provenance: {written[0]['scaleaxis']['ceiling_provenance']}")
    print("QUEUE-ELIGIBLE: NO. Every spec is CANDIDATE-marked.")
    if args.elect_mixed_order:
        print("NOTE: --elect-mixed-order requested but sec 8.3's mixed order is DECLINED by this "
              "build (1.17 h of wall against an unreadable numbering scheme; the worker's 60 s "
              "busy-poll makes the difference spurious). Longest-first stands.")
    return 0


if __name__ == "__main__":
    sys.exit(main_thicken() if "thicken" in sys.argv[1:] else main())
