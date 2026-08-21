#!/usr/bin/env python3
"""Generate the 32 K-scaling job specs (2 calibration + 30 sweep).

NCR_KSCALING_DESIGN.md sec 12. Specs are written to job_specs/ and are NOT
queued. Both tiers are marked CANDIDATE; the sweep tier is DOUBLE-gated
(audit AND the calibration gate's LICENSE-SWEEP verdict).

Deliberately generated rather than hand-written: 30 near-identical specs, each
carrying a K in four places (env var, --k flag, paths, validity check), is
exactly the shape of job that produces a silently mislabelled cell. The
generator derives all four from one loop variable, and the runner's own
--k/NCR_K tripwire (sec 5.4 R6b) plus each spec's validity_check catch any
residual drift at runtime.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import kscaling_config as KS   # noqa: E402  (imported for the ladder table + geometry)

OUT = os.path.join(_HERE, "job_specs")
PY = "/home/nvidia/tdenv/bin/python3"
WORKDIR = "/home/nvidia/ncr_kscaling"
EPH = "/ephemeral/kscaling"

SWEEP_K = (12, 16, 20, 28, 32)      # K=24 exists (55 cells); it is the anchor, not a sweep point
SEEDS = (0, 1, 2)
CALIB_K = 32                        # the riskiest K -- see design sec 10

RECIPES = {
    "primary": dict(
        label="FROZEN-contrastive (primary recipe)",
        flags="--aux-loss-type contrastive+cosine --freeze-entity-adapter --contrastive-temperature 0.07"),
    "compB": dict(
        label="TRAINABLE-contrastive (compB recipe)",
        flags="--aux-loss-type contrastive+cosine --contrastive-temperature 0.07"),
}

# Held at the audited G3-B31 values -- the recipe is NOT a variable in this sweep.
COMMON = ("--mode calibration --device cuda --steps 20000 --batch-size 32 --eval-batch-size 64 "
          "--warmup-steps 200 --lr 3e-4 --aux-read-loss-weight 0.5 --ortho-reg-weight 0.1 "
          "--ckpt-every 10000 --eval-every 1000 --ceiling-gpuh 6.0")

# sec 11.1, calibrated against the measured 0.82-0.83 GPU-h of the K=24 cells of record.
GPU_H = {12: 0.824, 16: 0.783, 20: 0.777, 24: 0.838, 28: 0.892, 32: 0.995}

# MEASURED total parameters per arm (kscaling_smoke item F, all six K, 2026-08-21) --
# quoted rather than re-derived so the spec states a counted number, not a formula.
PARAMS_PER_ARM = {12: 97_809_805, 16: 97_816_977, 20: 97_824_149,
                  24: 97_831_321, 28: 97_838_493, 32: 97_845_665}

CAL_HYP = (
    "NCR_KSCALING_DESIGN.md sec 10 CALIBRATION GATE. Two cells at K={K} (d={d}) -- the "
    "RISKIEST K in the grid -- run first and ALONE; their readout LICENSES (or re-scopes) the "
    "30-cell sweep. Prior cited, not rediscovered (research/kscaling-novelty-2026-08-21.md leg 1): "
    "the toy harness recorded K=32 far-depth death at h~5-6 and FRONTIER-AT-K*=30. That is a "
    "FREE-WRITE result, so it does not directly bar this cell's claim, which is about EXACT-WRITE "
    "(P1b) reads -- but P0 IS the free-write analogue, so the toy prior PREDICTS P0 at chance here, "
    "making it a positive control on the wall curve rather than a threat to the capability curve. "
    "LICENSE-SWEEP requires all three: (1) Gate-0 convergence (final CE < initial CE on full_graft, "
    "finite throughout, reaches step 20000 COMPLETED); (2) P1b margin-over-chance >= 0.90 at train "
    "hops h in {{1,2,3}} on the FROZEN cell; (3) P1b margin-over-chance >= 0.90 at h_top({K})={htop} "
    "on the FROZEN cell. If (3) fails but (1) and (2) pass, K={K} is the FRONTIER -- re-scope the "
    "sweep to K<=28 and report FRONTIER-AT-K*={K}; do NOT launch the 30 blindly. If (1) or (2) "
    "fails it is an instrument/convergence problem, not a science result. This cell is the "
    "{label}. Chance = 1/{K} = {chance:.4f}. Ladder {ladder}, residues {res} (6 distinct, none 0, "
    "none in the train residues {{1,2,3}}). t_in={tin}, doc_left_pad={pad}.")

SWEEP_HYP = (
    "NCR_KSCALING_DESIGN.md sec 7 -- the CAPABILITY-BREADTH SCALING CURVE. Cell: K={K} (d={d}, "
    "the PAIR is the variable -- house precedent: K=24 is dead at d=48 and healthy at d=25), "
    "{label}, seed {seed}. Tests whether exact in-context-written operator composition "
    "(closed-form V.K-pseudoinverse writes, O(log h) repeated-squaring reads) inside a 98M "
    "DeltaNet LM retains its capability as binding breadth grows, while the model's OWN learned "
    "writes stay at chance. Both regimes are scored on every cell (gate memo carried requirement "
    "1): P1b = EXACT-WRITE (teacher-forced operator; the READ is under test) is the CAPABILITY "
    "curve; P0 = LEARNED-WRITE is the WALL curve, predicted at chance = 1/{K} = {chance:.4f}. "
    "Primary readout: P1b margin-over-chance at h_top={htop} (residue {htopres} = K/2, the "
    "ANTIPODAL point of the K-cycle -- the maximum reachable effective distance), matched pools, "
    "n=256. Fresh residue-verified ladder for this K {ladder}, residues {res}: 6 DISTINCT residues, "
    "none 0 (identity), none in the train residues {{1,2,3}}, squaring profile (2,3,4,4,5,5) held "
    "IDENTICAL across every K so the fp-depth axis (EXPERIMENT_LOG 2026-08-21 #3 result B: a real "
    "monotone drift, 1.0000->0.9219 over 3->11 squarings) is matched and cannot confound the K "
    "axis. The carried pinned ladder (5,12,20,29,40,61) is NOT usable here -- it crashes the "
    "soundness guard at K in {{12,20,28}} and silently collides residues at K in {{16,24,32}}. "
    "Fixed-effective-distance control h_fix={hfix} (residue 4 at EVERY K, same squaring count as "
    "h_top) separates breadth from depth, since h_top's effective distance K/2={halfk} itself grows "
    "with K. Param count {params:,}/arm -- the whole K range spans 0.037%, so this is not a "
    "capacity curve in disguise. t_in={tin} (doc_left_pad={pad}: the 7K+6 backbone input is below "
    "lm_pretrain_rd's hard T>=128 kernel floor at K in {{12,16}} -- MEASURED AssertionError, not "
    "assumed -- so those K are left-padded with inert BUFFER tokens and all four position fields "
    "shifted; pad is 0 for K>=20, keeping them byte-identical to the pinned construction). "
    "Secondary pre-registered question (EXPERIMENT_LOG 2026-08-21 #6): does the residual "
    "frozen-vs-trainable ordering (+0.0098 and ceiling-compressed at K=24 h=61) open at larger K? "
    "See design sec 7.3 -- per-K inference is UNREACHABLE at n=3 seeds (min two-sided "
    "Mann-Whitney p = 0.10); the pre-registered test is on the POOLED 15-vs-15.")

NOTES_CAL = (
    "CANDIDATE -- queue-eligible after audit (the CALIBRATION tier; the 30 sweep specs are NOT "
    "queue-eligible until this gate returns LICENSE-SWEEP). Built by the K-scaling DESIGN+BUILD "
    "agent 2026-08-21 on PATCHED COPIES in /home/nvidia/ncr_kscaling (runner md5 "
    "ee5833743049e1bb1864124ad5d3fbf6, graft md5 74ee84fc920b024901d11add66cc5c2d, config md5 "
    "6eaf8384a3ef6e9e43b3947720291024). The PINNED originals in ~/ncr_g3b31_contrastive are "
    "UNTOUCHED and md5-verified by the patch generator on every run (runner "
    "9a93198b642242f512ff8489e32b0a53, graft bc105af69661e488ff95f5046e2bcd8a). Real-CUDA smoke "
    "PASSED at all six K (12 PASS/0 FAIL at K=12,16; 11 PASS/0 FAIL/1 N-A at K=20,24,28,32) with "
    "all eight negative-test instances FIRED -- including the one that proves the new pairwise "
    "residue-distinctness assert rejects a ladder the PINNED guard accepts. An END-TO-END 3-step "
    "run through this exact cmd shape ALSO passed, and is what caught the second launch-losing "
    "FATAL in this build (build_attribution indexed deep_gap['h=61'], a K=24 ladder literal that "
    "would have raised KeyError at the first eval of every cell at every K -- design sec 5.5). NCR_K and --k are "
    "both present and the runner asserts they agree before any GPU work.")

NOTES_SWEEP = (
    "CANDIDATE -- NOT queue-eligible until BOTH (a) the build audit passes AND (b) the "
    "calibration gate (specs 0100/0101) returns LICENSE-SWEEP. DOUBLE-GATED by design: if the "
    "K=32 frozen calibration cell fails its deep-capability leg, this sweep is re-scoped to "
    "K<=28 rather than launched. Built by the K-scaling DESIGN+BUILD agent 2026-08-21 on the same "
    "patched copies and same verified smoke as 0100/0101. Per-cell isolation is provided by the "
    "queue itself -- one cell per spec, a failure routes to failed/ and is not auto-retried, and "
    "the other 29 cells are unaffected; there is no shared driver whose crash could take down the "
    "wave. Checkpoints go to /ephemeral (NEVER the root fs). Scored afterwards by "
    "kscaling_battery.py, which builds every eval pool with the CHECKPOINT'S OWN recorded seed "
    "(matched pools from day one, gate memo carried requirement 4) and refuses to run unless the "
    "checkpoint's recorded d_ncr equals K+1.")


def validity(out_json: str, k: int) -> str:
    """status/step PLUS a K-identity check, so a mislabelled cell fails its own
    validity check instead of silently entering the curve as another K."""
    return (
        f"{PY} -c \"import json; d=json.load(open('{out_json}')); "
        f"assert d.get('status')=='COMPLETED', d.get('status'); "
        f"assert d.get('step',0)>=20000, d.get('step'); "
        f"ks=d.get('kscaling') or {{}}; "
        f"assert ks.get('K')=={k}, ('K', ks.get('K')); "
        f"assert ks.get('d_ncr')=={k+1}, ('d_ncr', ks.get('d_ncr')); "
        f"assert ks.get('d_equals_k_plus_1') is True; "
        f"assert ks.get('h_top')=={KS.LADDER_TABLE[k][-1]}, ('h_top', ks.get('h_top')); "
        f"assert ks.get('deep_ladder')=={list(KS.LADDER_TABLE[k])}, ks.get('deep_ladder')\"")


def spec(job_id: str, k: int, recipe: str, seed: int, tier: str) -> dict:
    d, r = k + 1, RECIPES[recipe]
    cell = f"kscaling_K{k}_{recipe}_s{seed}"
    outdir = f"{EPH}/results"
    out_json = f"{outdir}/{cell}.json"
    ladder = KS.LADDER_TABLE[k]
    fields = dict(
        K=k, d=d, label=r["label"], seed=seed, chance=1.0 / k,
        htop=ladder[-1], htopres=ladder[-1] % k, halfk=k // 2,
        hfix=KS.FIXED_DIST_TABLE[k], ladder=list(ladder), res=[h % k for h in ladder],
        tin=KS.t_in(k), pad=KS.doc_left_pad(k), params=PARAMS_PER_ARM[k])
    hyp = (CAL_HYP if tier == "calibration" else SWEEP_HYP).format(**fields)
    cmd = (f"mkdir -p {outdir} {EPH}/ckpts && cd {WORKDIR} && "
           f"NCR_K={k} {PY} ncr_lm_wave1_runner.py --k {k} {COMMON} {r['flags']} "
           f"--seed {seed} --cell-id {cell} --out {out_json} "
           f"--ckpt-dir {EPH}/ckpts/{cell} 2>&1 | tee -a {outdir}/{cell}.log")
    return {
        "id": job_id, "lane": "NCR", "tier": tier,
        "hypothesis": hyp, "cmd": cmd,
        "gpu_h_estimate": GPU_H[k], "output_dir": outdir,
        "validity_check": validity(out_json, k),
        "notes": NOTES_CAL if tier == "calibration" else NOTES_SWEEP,
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    written = []

    for i, recipe in enumerate(("primary", "compB")):
        s = spec(f"{100 + i:04d}_ncr_kscaling_calib_K{CALIB_K}_{recipe}_s0",
                 CALIB_K, recipe, 0, "calibration")
        written.append(s)

    n = 110
    for k in SWEEP_K:
        for recipe in ("primary", "compB"):
            for seed in SEEDS:
                s = spec(f"{n:04d}_ncr_kscaling_K{k}_{recipe}_s{seed}", k, recipe, seed, "sweep")
                written.append(s)
                n += 1

    for s in written:
        with open(os.path.join(OUT, s["id"] + ".json"), "w") as f:
            json.dump(s, f, indent=1)

    ncal = sum(1 for s in written if s["tier"] == "calibration")
    nsw = sum(1 for s in written if s["tier"] == "sweep")
    tot = sum(s["gpu_h_estimate"] for s in written)
    print(f"wrote {len(written)} specs to {OUT}  ({ncal} calibration + {nsw} sweep)")
    print(f"ledger: calibration {sum(s['gpu_h_estimate'] for s in written if s['tier']=='calibration'):.2f} "
          f"+ sweep {sum(s['gpu_h_estimate'] for s in written if s['tier']=='sweep'):.2f} "
          f"= {tot:.2f} GPU-h")
    assert len(written) == 32, len(written)
    assert ncal == 2 and nsw == 30
    return 0


if __name__ == "__main__":
    sys.exit(main())
