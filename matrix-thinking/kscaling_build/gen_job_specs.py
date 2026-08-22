#!/usr/bin/env python3
"""Generate the K-scaling job specs.

  (no argument)  the ORIGINAL 32 (2 calibration + 30 sweep), sec 12
  frontier       the 12 K=36/40 FRONTIER EXTENSION specs, 0140-0151, sec 14

NCR_KSCALING_DESIGN.md sec 12/14. Specs are written to job_specs/ and are NOT
queued. Both original tiers are marked CANDIDATE; the sweep tier is DOUBLE-gated
(audit AND the calibration gate's LICENSE-SWEEP verdict). The frontier tier is
SINGLE-gated (audit only) -- it is not calibration-gated, because the sweep it
extends already ran to completion (EXPERIMENT_LOG 2026-08-22 #2).

The two modes are separate so that generating the frontier wave cannot rewrite,
resurrect or renumber a spec belonging to the completed wave (0134 and 0137 in
particular were deliberately retired -- those two cells ran as the calibration
pair 0100/0101).

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

FRONTIER_K = (36, 40)               # sec 14; K=44 is construction-impossible (see sec 14)

# sec 11.1, calibrated against the measured 0.82-0.83 GPU-h of the K=24 cells of record.
# Same rule for the frontier K: smoke item L's projected train-only GPU-h at 20000
# steps x the 1.17 eval/ckpt overhead multiplier (0.901 -> 1.054; 1.003 -> 1.174).
GPU_H = {12: 0.824, 16: 0.783, 20: 0.777, 24: 0.838, 28: 0.892, 32: 0.995,
         36: 1.054, 40: 1.174}

# MEASURED total parameters per arm (kscaling_smoke item F, all six K, 2026-08-21;
# K=36/40 measured 2026-08-22 on the same box) -- quoted rather than re-derived so
# the spec states a counted number, not a formula.
PARAMS_PER_ARM = {12: 97_809_805, 16: 97_816_977, 20: 97_824_149,
                  24: 97_831_321, 28: 97_838_493, 32: 97_845_665,
                  36: 97_852_837, 40: 97_860_009}

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

FRONTIER_HYP = (
    "NCR_KSCALING_DESIGN.md sec 14 -- the K=36/40 FRONTIER EXTENSION of the capability-breadth "
    "scaling curve. Cell: K={K} (d={d}, the PAIR is the variable), {label}, seed {seed}. The curve "
    "of record (EXPERIMENT_LOG 2026-08-22 #2 as corrected by #5) is FLAT and AT CEILING over "
    "K=12..32: P1b kappa >= 0.9708 in 36/36 cells at h_top; at h_fix 36/36 clear the 0.90 bar, "
    "35/36 >= 0.97, floor 0.9470 (anchor compB_s0). Both recipes. The model's OWN learned "
    "writes (P0) sit in the chance band at every K >= 16. This wave asks whether that flatness "
    "survives a further 1.25x of binding breadth, at the two largest K whose ANTIPODAL top rung is "
    "still constructible inside the frozen squaring-5 band [32,63]. (K=44 is not: an antipodal "
    "rung needs 3K/2 = 66 <= 63, so derive_ladder(44) raises -- EXPERIMENT_LOG 2026-08-22 #3. It "
    "is DROPPED this wave, not deferred silently.) "
    "PRE-REGISTERED NULL EXPECTATIONS, locked in #3 BEFORE this build and carried verbatim by "
    "reference: (a) CAPABILITY-HOLDS continues at K=36/40 (kappa >= 0.90, likely >= 0.95, both "
    "recipes) -- a frontier here would be a FINDING, not the expectation; (b) WALL-HOLDS 0/6 at "
    "both K, extrapolating the h=1 toehold trend 5/6 -> 1/6 -> 1/6 -> 0 by K=28; (c) the LIVE "
    "RISK to watch and NOT assume away is trainable/contrastive Gate-0 convergence (CE finite and "
    "falling) in the K~36-40 regime -- the toy prior's mechanism was trainability collapse, and it "
    "is monitored per cell by the existing Gate-0 check (this spec's validity_check asserts it "
    "directly on loss_history). "
    "Both regimes are scored on every cell: P1b = EXACT-WRITE (teacher-forced operator; the READ "
    "is under test) is the CAPABILITY curve; P0 = LEARNED-WRITE is the WALL curve, predicted at "
    "chance = 1/{K} = {chance:.4f}. Primary readout: P1b kappa at h_top={htop} (residue {htopres} "
    "= K/2, the ANTIPODAL point of the K-cycle), matched pools, n=256, ckpt_step == 20000. Fresh "
    "residue-verified ladder for this K {ladder}, residues {res}: 6 DISTINCT residues, none 0 "
    "(identity), none in the train residues {{1,2,3}}, squaring profile (2,3,4,4,5,5) held "
    "IDENTICAL to every other K so the fp-depth axis (EXPERIMENT_LOG 2026-08-21 #3 result B: a "
    "real monotone drift, 1.0000->0.9219 over 3->11 squarings) is matched and cannot confound the "
    "K axis. DISCLOSED RESIDUAL: popcount at h_top is 4 here (54 = 0b110110, 60 = 0b111100) vs "
    "2-3 across K=12..32 -- n_applies was never matchable across K (design sec 4.2) and is "
    "recorded per rung; n_squarings, the axis the DRIFT finding implicates, is 5 at every K. "
    "Fixed-effective-distance control h_fix={hfix} (residue 4 at EVERY K, same squaring count as "
    "h_top) separates breadth from depth, since h_top's effective distance K/2={halfk} itself "
    "grows with K. Param count {params:,}/arm (MEASURED, smoke item F) -- the K=12..40 range now "
    "spans 0.051%, so this is still not a capacity curve in disguise. t_in={tin} "
    "(doc_left_pad={pad}: both frontier K are far above lm_pretrain_rd's hard T>=128 kernel "
    "floor, so they are byte-identical to the pinned construction, like every K >= 20). Bands "
    "otherwise identical to the main sweep (sec 7 as amended): kappa bar 0.90 on >= 2/3 seeds for "
    "CAPABILITY-HOLDS(K); the sec 7.3 stratified within-K exact permutation test extends to 8 "
    "strata x 9 pairs, threshold recomputed at build by the SAME construction -- T >= 53/72 "
    "(exact two-sided p = 0.0099; T >= 52 gives 0.0156 and does NOT clear), symmetric lower "
    "threshold T <= 19/72.")

NOTES_FRONTIER = (
    "CANDIDATE -- NOT queue-eligible until audited. SINGLE-gated (no LICENSE sentinel): this wave "
    "is NOT calibration-gated, because the sweep it extends already completed 36/36 cells with 0 "
    "failures (EXPERIMENT_LOG 2026-08-22 #2) and its internal gate returned CLEAR-WITH-CONSTRAINTS "
    "(#3). Ceremony tier: 10-50 GPU-h => one audit round on the extension delta, per #3. Built by "
    "the K=36/40 frontier BUILD agent 2026-08-22 against repo commit add3239, on the SAME patched "
    "copies and the SAME battery as the completed wave -- runner md5 "
    "ee5833743049e1bb1864124ad5d3fbf6, graft md5 74ee84fc920b024901d11add66cc5c2d, battery md5 "
    "5735c788563d9a21f2198c9f5b4793d5 (battery UNCHANGED by this extension: it reads the ladder "
    "from kscaling_config, so admitting two more K needed no edit to the scorer). The PINNED "
    "originals in ~/ncr_g3b31_contrastive are UNTOUCHED and re-verified at deploy time (runner "
    "9a93198b642242f512ff8489e32b0a53, graft bc105af69661e488ff95f5046e2bcd8a). Real-CUDA smoke "
    "PASSED at BOTH new K on an H100 (11 PASS / 0 FAIL / 1 N-A each) with every applicable "
    "negative test FIRED. DISCLOSED SMOKE DEVIATION at K=36 ONLY: negative item B (the pinned "
    "reference ladder (5,12,20,29,40,61) must be REJECTED) has nothing to fire on, because at "
    "K=36 that ladder is genuinely sound (residues 5,12,20,29,4,25 -- six distinct, none 0, none "
    "in {1,2,3}, profile intact); it is recorded as a POSITIVE soundness proof instead of a "
    "vacuous 'did not fire' FAIL, and the pairwise-distinctness guard is still exercised at K=36 "
    "by item C, which builds a residue-colliding fixture on purpose. At K=40 item B fires normally "
    "(h=40 is IDENTITY mod 40). An END-TO-END 3-step run through this exact cmd shape ALSO passed "
    "at K=40 -- the audit's own instrument, since module smokes were proven blind to spec-level "
    "defects on the previous build. NCR_K and --k are both present and the runner asserts they "
    "agree before any GPU work. Per-cell isolation is provided by the queue itself -- one cell per "
    "spec, a failure routes to failed/ and is not auto-retried. Checkpoints go to /ephemeral "
    "(NEVER the root fs). Placement 1 cell/GPU per the standing declined-packing ruling; measured "
    "SM util 97% median / 100% max at both K, so no packing is warranted. Scored afterwards by "
    "kscaling_battery.py, which builds every eval pool with the CHECKPOINT'S OWN recorded seed and "
    "refuses to run unless the checkpoint's recorded d_ncr equals K+1.")

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


def validity(out_json: str, k: int, gate0: bool = False) -> str:
    """status/step PLUS a K-identity check, so a mislabelled cell fails its own
    validity check instead of silently entering the curve as another K.

    gate0=True additionally asserts the #3 null-expectation (c) LIVE RISK -- the
    Gate-0 convergence leg -- directly on the run's own loss_history: BOTH arms
    logged, every logged CE finite, and final CE strictly below initial CE. Same
    shape as the embed-path specs' close_target history assert. A cell whose
    trainable/contrastive optimisation collapsed in the K~36-40 regime then
    fails its OWN validity check and routes to failed/ instead of entering the
    curve as a spurious frontier point."""
    # `math` is imported ONLY in the gate0 form, so the non-gate0 string stays
    # byte-identical to the 30 specs of the completed wave.
    head = (
        f"{PY} -c \"import json{', math' if gate0 else ''}; d=json.load(open('{out_json}')); "
        f"assert d.get('status')=='COMPLETED', d.get('status'); "
        f"assert d.get('step',0)>=20000, d.get('step'); "
        f"ks=d.get('kscaling') or {{}}; "
        f"assert ks.get('K')=={k}, ('K', ks.get('K')); "
        f"assert ks.get('d_ncr')=={k+1}, ('d_ncr', ks.get('d_ncr')); "
        f"assert ks.get('d_equals_k_plus_1') is True; "
        f"assert ks.get('h_top')=={KS.LADDER_TABLE[k][-1]}, ('h_top', ks.get('h_top')); "
        f"assert ks.get('deep_ladder')=={list(KS.LADDER_TABLE[k])}, ks.get('deep_ladder')")
    if not gate0:
        return head + "\""
    return head + (
        f"; lh=d.get('loss_history') or {{}}; "
        f"assert set(lh)=={{'full_graft','backbone_only'}}, sorted(lh); "
        f"h=lh['full_graft']; assert len(h)>=100, len(h); "
        f"assert all(math.isfinite(r[1]) for r in h), 'non-finite CE in loss_history'; "
        f"assert h[-1][1] < h[0][1], ('GATE-0 NOT CONVERGED', h[0], h[-1])\"")


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
    hyp = {"calibration": CAL_HYP, "sweep": SWEEP_HYP, "frontier": FRONTIER_HYP}[tier].format(**fields)
    notes = {"calibration": NOTES_CAL, "sweep": NOTES_SWEEP, "frontier": NOTES_FRONTIER}[tier]
    cmd = (f"mkdir -p {outdir} {EPH}/ckpts && cd {WORKDIR} && "
           f"NCR_K={k} {PY} ncr_lm_wave1_runner.py --k {k} {COMMON} {r['flags']} "
           f"--seed {seed} --cell-id {cell} --out {out_json} "
           f"--ckpt-dir {EPH}/ckpts/{cell} 2>&1 | tee -a {outdir}/{cell}.log")
    return {
        "id": job_id, "lane": "NCR", "tier": tier,
        "hypothesis": hyp, "cmd": cmd,
        "gpu_h_estimate": GPU_H[k], "output_dir": outdir,
        "validity_check": validity(out_json, k, gate0=(tier == "frontier")),
        "notes": notes,
    }


def _write(written: list[dict]) -> None:
    for s in written:
        with open(os.path.join(OUT, s["id"] + ".json"), "w") as f:
            json.dump(s, f, indent=1)


def main_frontier() -> int:
    """sec 14: 12 specs, ids 0140-0151, K in {36,40} x {primary,compB} x seeds 0-2.

    IDs start at 0140 so they cannot collide with 0100-0139 (the completed wave,
    including the retired 0134/0137) or with anything in the queue's history."""
    os.makedirs(OUT, exist_ok=True)
    written, n = [], 140
    for k in FRONTIER_K:
        for recipe in ("primary", "compB"):
            for seed in SEEDS:
                written.append(spec(f"{n:04d}_ncr_kscaling_K{k}_{recipe}_s{seed}",
                                    k, recipe, seed, "frontier"))
                n += 1
    ids = [s["id"][:4] for s in written]
    assert len(written) == 12 and len(set(ids)) == 12, ids
    assert min(ids) == "0140" and max(ids) == "0151", ids
    _write(written)
    print(f"wrote {len(written)} FRONTIER specs to {OUT}  ({ids[0]}-{ids[-1]})")
    print(f"ledger: frontier {sum(s['gpu_h_estimate'] for s in written):.2f} GPU-h "
          f"({6 * GPU_H[36]:.2f} at K=36 + {6 * GPU_H[40]:.2f} at K=40)")
    return 0


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

    _write(written)

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
    sys.exit(main_frontier() if "frontier" in sys.argv[1:] else main())
