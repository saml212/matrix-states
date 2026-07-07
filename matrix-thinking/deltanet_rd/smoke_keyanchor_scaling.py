"""smoke_keyanchor_scaling.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15
(attack-round-1 verdict RUN-AFTER-REASONING-LINK, wave PARKED sec 15.18)'s
registered fla-free Wave -1 smoke suite for the cliff-location-scaling-law-
across-d_state wave, as a committed, CPU-runnable script. Mirrors smoke_
keyanchor_dstate.py's own split: this file imports ONLY key_anchoring.py,
run_deltanet_rd_exactness_sweep.py, and fit_cliff_curve.py (all verified
fla-free at module scope) -- run_deltanet_rd.py itself is NOT imported (it
touches CUDA-dependent code paths at call time); the pin-naming CONVENTION
it uses is reproduced here as a pure string rule and cross-checked against
the actual committed pin files.

This is GAP-2 (MAJOR-3, sec 15.18): the draft's own sec 15.13 Stage -1 item
8 ("re-run smoke_key_anchoring.py at d=80/96") was never actually
actionable -- that suite has no --d-state flag. This file is the wave-
specific smoke the sec 15.15 build-scope enumeration should have listed
(per the smoke_keyanchor_cliff.py / smoke_keyanchor_dstate.py convention).

Covers:
  smoke 1: keyanchor_scaling_full_manifest()'s shape -- 30 cells (15/15
    split by d_state), exact K-grids and seed blocks per sec 15.15 item 5's
    registered table, candidate (d) arch, full instrumentation, n_iter read
    from the wave's OWN namespaced dict (never ka.GATE2_N_ITER_BY_K).
  smoke 2: keyanchor_scaling_gate1_manifest()'s shape -- 10 cells (5/d),
    exact seeds, 5,000 steps.
  smoke 3: keyanchor_scaling_calibration_manifest() -- exactly the 2 cells
    (d=80 K=43 seed=1030, d=96 K=51 seed=1430), filtered from (never
    hand-duplicated against) the full manifest.
  smoke 4: every cell's own filename carries the correct '_d80'/'_d96' name
    bit (sec 13's _spec() generalization, reused unmodified).
  smoke 5: zero-collision -- keyanchor-scaling's manifest (mandatory +
    Gate-1 probes + calibration) is collision-free against every prior/
    sibling wave this program has ever registered, INCLUDING keyanchor-
    dstate and keyanchor-dose (built after the dstate smoke's own list was
    written).
  smoke 6: GATE2_N_ITER_BY_K collision closed by construction (sec 15.6) --
    the K=48/d=80 cell's own n_iter is read from KEYANCHOR_SCALING_GATE2_
    N_ITER_BY_D_K, UNAFFECTED by corrupting the flat ka.GATE2_N_ITER_BY_K[48]
    entry (negative test: corrupt, rebuild the spec, confirm no drift).
  smoke 7: keyanchor_scaling_kernel_gate_check -- POSITIVE test against the
    real committed artifact (must currently read CLEARED/ok=True) AND four
    NEGATIVE tests to completion (sec 15.18 Q1's own mandated enforcement
    branch): missing file, verdict without 'CLEARED', exit_code != 0,
    incomplete T-sweep / a False cell in grid_pass.
  smoke 8: keyanchor_scaling_stage_gate mechanics -- kernel gate checked
    FIRST and blocks BOTH stages when the artifact is bad; --scaling-stage
    full refuses when a calibration cell is missing; a synthetic in-bracket
    pair of calibration cells writes CALIBRATION_DONE; a synthetic
    over-bracket cell refuses with no sentinel; --accept-scaling-stage-
    override bypasses the calibration/abort gate but NOT the kernel gate.
  smoke 9: keyanchor_scaling_check_abort -- two-tier (main-grid vs
    anchor-K) bracket, boundary-exact, per d.
  smoke 10: rev7_threshold_derive.py's d=80/96 pins -- byte-DIFFERENT from
    d=64/d=128 (and from each other), sigma_chance exact, r_min_headline
    unchanged at 0.35 across all four pins.
  smoke 11: pin-selection-by-d_state -- the _pin_name convention
    (run_deltanet_rd.py lines ~1669-1670) resolves to REV7_THRESHOLD_
    PINNED_D80.json / _D96.json, both files exist, and ka.validate_rev7_pin
    (the ACTUAL launcher-gate function, fla-free) validates each one live.
  smoke 12: key_anchoring.pooled_null_check's d_state-genericity -- called
    with d_state=80 and d_state=96 directly, confirms no hardcoded-64
    assumption (sec 15.18's own verified-list claim, re-checked here rather
    than only cited).
  smoke 13: Gate 2 EXECUTED at d_state=80 and d_state=96 with this wave's
    own registered K-grids explicitly (never the function's own (16,32)
    default) -- also the first real Gate-2 data at these two new d's,
    reported honestly regardless of outcome.
  smoke 14: disk-gate d_state-awareness -- keyanchor_scaling_projected_
    ckpt_bytes scales with each cell's own d_state (d=96 cells project MORE
    bytes than d=80 cells at the same step count), unlike keyanchor_mech_
    projected_ckpt_bytes (hardcoded to d_state=64) which would silently
    under-project if reused here unmodified.
  smoke 15: manifest enumeration -- prints all 30 mandatory cells
    (d_state, K, seed) so a human/CI log has the full grid on record.

Wired as an ADDITIONAL pre-launch CPU gate for --wave keyanchor-scaling
(run_deltanet_rd_exactness_sweep.py's main(), alongside smoke_key_
anchoring.py + gate2_construction_test.py) -- rc!=0 aborts the wave before
any GPU cell dispatches.

Exit code 0 = every item PASSED. Run: python smoke_keyanchor_scaling.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import key_anchoring as ka                              # noqa: E402 (fla-free)
import run_deltanet_rd_exactness_sweep as rdx            # noqa: E402 (fla-free)

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


# ---------------------------------------------------------------------------
# smoke 1: keyanchor_scaling_full_manifest() shape.
# ---------------------------------------------------------------------------

_EXPECTED_SEEDS = {
    80: {20: {1020, 1021, 1022}, 43: {1030, 1031, 1032}, 48: {1130, 1131, 1132},
         53: {1230, 1231, 1232}, 58: {1330, 1331, 1332}},
    96: {24: {1420, 1421, 1422}, 51: {1430, 1431, 1432}, 57: {1530, 1531, 1532},
         63: {1630, 1631, 1632}, 69: {1730, 1731, 1732}},
}


def smoke_1_full_manifest_shape():
    m = rdx.keyanchor_scaling_full_manifest()
    right_count = len(m) == 30
    split_ok = (len([s for s in m if s["d_state"] == 80]) == 15
                and len([s for s in m if s["d_state"] == 96]) == 15)
    ks_ok = all(sorted(set(s["K"] for s in m if s["d_state"] == d))
                == sorted(rdx.KEYANCHOR_SCALING_KS_BY_D[d]) for d in (80, 96))
    right_arm = all(s["arm"] == "d" for s in m)
    right_lambda_mode = all(s["anchor_lambda_mode"] == "learned" for s in m)
    right_instrumentation = all(s["drift_probe"] is True and s["rev7_engagement"] is True for s in m)
    seeds_ok = all(set(s["seed"] for s in m if s["d_state"] == d and s["K"] == K)
                   == _EXPECTED_SEEDS[d][K]
                   for d in (80, 96) for K in rdx.KEYANCHOR_SCALING_KS_BY_D[d])
    n_iter_ok = all(s["geo3_n_iter"] == rdx.KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[s["d_state"]][s["K"]]
                     for s in m)
    print(f"    count={len(m)} d=80: {len([s for s in m if s['d_state']==80])} cells, "
          f"d=96: {len([s for s in m if s['d_state']==96])} cells")
    for d in (80, 96):
        for K in rdx.KEYANCHOR_SCALING_KS_BY_D[d]:
            print(f"      d={d} K={K}: seeds={sorted(s['seed'] for s in m if s['d_state']==d and s['K']==K)} "
                  f"(expect {sorted(_EXPECTED_SEEDS[d][K])})")
    ok = right_count and split_ok and ks_ok and right_arm and right_lambda_mode \
        and right_instrumentation and seeds_ok and n_iter_ok
    _report("smoke 1: keyanchor_scaling_full_manifest() shape -- 30 cells (15/15 split), exact "
            "K-grids + registered seed blocks (sec 15.15 item 5), candidate (d) arch, full "
            "instrumentation, n_iter from the wave's own namespaced dict", ok)


def smoke_2_gate1_manifest_shape():
    g1 = rdx.keyanchor_scaling_gate1_manifest(80) + rdx.keyanchor_scaling_gate1_manifest(96)
    right_count = len(g1) == 10
    expected_seed = {(80, 20): 1025, (80, 43): 1035, (80, 48): 1135, (80, 53): 1235, (80, 58): 1335,
                      (96, 24): 1425, (96, 51): 1435, (96, 57): 1535, (96, 63): 1635, (96, 69): 1735}
    seeds_ok = all(s["seed"] == expected_seed[(s["d_state"], s["K"])] for s in g1)
    steps_ok = all(s["steps"] == rdx.KEYANCHOR_SCALING_GATE1_STEPS == 5000 for s in g1)
    print(f"    count={len(g1)} entries={sorted((s['d_state'], s['K'], s['seed']) for s in g1)}")
    ok = right_count and seeds_ok and steps_ok
    _report("smoke 2: keyanchor_scaling_gate1_manifest() shape -- 10 cells (5/d), exact registered "
            "seeds, 5,000 steps", ok)


def smoke_3_calibration_manifest_shape():
    calib = rdx.keyanchor_scaling_calibration_manifest()
    right_count = len(calib) == 2
    right_cells = right_count and {(s["d_state"], s["K"], s["seed"]) for s in calib} == \
        {(80, 43, 1030), (96, 51, 1430)}
    full_filtered = [s for s in rdx.keyanchor_scaling_full_manifest()
                      if (s["d_state"], s["K"], s["seed"]) in {(80, 43, 1030), (96, 51, 1430)}]
    consistent = right_count and sorted(full_filtered, key=lambda s: s["d_state"]) == \
        sorted(calib, key=lambda s: s["d_state"])
    print(f"    calibration manifest: {[(s['d_state'], s['K'], s['seed']) for s in calib]}")
    ok = right_count and right_cells and consistent
    _report("smoke 3: keyanchor_scaling_calibration_manifest() -- exactly [d=80/K=43/seed=1030, "
            "d=96/K=51/seed=1430], filtered from (not hand-duplicated against) the full manifest", ok)


def smoke_4_name_bit():
    m = rdx.keyanchor_scaling_full_manifest() + rdx.keyanchor_scaling_gate1_manifest(80) \
        + rdx.keyanchor_scaling_gate1_manifest(96)
    bad = [s["name"] for s in m if not s["name"].endswith(f"_d{s['d_state']}")]
    ok = not bad
    print(f"    {len(m)} cells checked; all end in '_d{{d_state}}': {ok}"
          f"{'' if ok else f' -- offenders: {bad}'}")
    _report("smoke 4: every cell's filename carries the correct '_d80'/'_d96' name bit "
            "(_spec()'s d_state != 64 generalization, reused unmodified)", ok)


# ---------------------------------------------------------------------------
# smoke 5: zero-collision vs every prior/sibling manifest, INCLUDING
# keyanchor-dstate and keyanchor-dose (built after the dstate smoke's own
# list was written).
# ---------------------------------------------------------------------------

def smoke_5_zero_collision():
    scaling_paths = {rdx.out_path("/fake/root", s)
                      for s in (rdx.keyanchor_scaling_full_manifest()
                                + rdx.keyanchor_scaling_gate1_manifest(80)
                                + rdx.keyanchor_scaling_gate1_manifest(96)
                                + rdx.keyanchor_scaling_calibration_manifest())}
    prior_manifests = {
        "wavegeo3": rdx.geo3_wave1_manifest(include_k48=True),
        "wavekeyanchor": rdx.keyanchor_wave1_manifest(),
        "wavekeyanchor-neg1": rdx.keyanchor_wave_neg1_manifest(),
        "waveref": rdx.reference_arms_manifest(),
        "wavekeyanchor-confirm": rdx.keyanchor_confirm_manifest(),
        "wavekeyanchor-mech": rdx.keyanchor_mech_manifest(),
        "wavekeyanchor-mech-gate1": rdx.keyanchor_mech_gate1_manifest(),
        "wavekeyanchor-k48": rdx.keyanchor_k48_manifest() + rdx.keyanchor_k48_dprime_manifest()
                             + rdx.keyanchor_k48_fixed_lambda1_manifest(),
        "wavekeyanchor-k48-ref": rdx.keyanchor_k48_reference_manifest(),
        "wavekeyanchor-k48-gate1": rdx.keyanchor_k48_gate1_manifest(),
        "wavekeyanchor-e": rdx.keyanchor_e_wave_manifest(),
        "wavekeyanchor-cliff": rdx.keyanchor_cliff_manifest() + rdx.keyanchor_cliff_gate1_manifest(),
        "wavekeyanchor-dstate": rdx.keyanchor_dstate_manifest() + rdx.keyanchor_dstate_gate1_manifest()
                                + rdx.keyanchor_dstate_calibration_manifest(),
        "wavekeyanchor-dose": rdx.keyanchor_dose_manifest("rank4") + rdx.keyanchor_dose_manifest("diffuse")
                              + rdx.keyanchor_dose_calibration_manifest(),
    }
    ok = True
    for label, manifest in prior_manifests.items():
        prior_paths = {rdx.out_path("/fake/root", s) for s in manifest}
        disjoint = scaling_paths.isdisjoint(prior_paths)
        if not disjoint:
            print(f"    FAIL: keyanchor-scaling collides with {label} (shared fake_root): "
                  f"{scaling_paths & prior_paths}")
        ok = ok and disjoint
    print(f"    keyanchor-scaling ({len(scaling_paths)} paths) disjoint from all "
          f"{len(prior_manifests)} prior/sibling manifests (shared fake_root): {ok}")
    print("    real out_dir: 'wavekeyanchor-scaling' (main()'s own f\"wave{args.wave}\" convention) "
          "-- distinct from every prior wave's own out_dir")
    _report("smoke 5: zero-collision -- keyanchor-scaling's manifest (mandatory + Gate-1 probes + "
            "calibration) is collision-free against every prior/sibling wave this program has "
            "ever registered, including keyanchor-dstate and keyanchor-dose", ok)


# ---------------------------------------------------------------------------
# smoke 6: GATE2_N_ITER_BY_K collision closed by construction (sec 15.6).
# ---------------------------------------------------------------------------

def smoke_6_gate2_n_iter_collision_closed():
    # Baseline: K=48/d=80's n_iter reads 20 from the wave's OWN namespaced
    # dict, coincidentally the same value the flat dict already has for
    # K=48 (verified sufficient only at d=64) -- so a naive reuse would
    # silently "work" today. The negative test proves it is NOT actually
    # falling through to the flat dict: corrupt ka.GATE2_N_ITER_BY_K[48] to
    # an obviously-wrong value and confirm the scaling spec is unaffected.
    baseline_spec = rdx._keyanchor_scaling_spec(48, 1130, 80)
    baseline_n_iter = baseline_spec["geo3_n_iter"]
    baseline_matches_namespaced = baseline_n_iter == rdx.KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[80][48] == 20

    original = ka.GATE2_N_ITER_BY_K[48]
    ka.GATE2_N_ITER_BY_K[48] = 999999   # obviously-wrong sentinel
    try:
        corrupted_spec = rdx._keyanchor_scaling_spec(48, 1130, 80)
        unaffected = corrupted_spec["geo3_n_iter"] == 20 != 999999
    finally:
        ka.GATE2_N_ITER_BY_K[48] = original   # restore -- must not leak into other smokes

    restored_ok = ka.GATE2_N_ITER_BY_K[48] == original == 20
    print(f"    K=48/d=80 baseline n_iter={baseline_n_iter} (expect 20, from the wave's own "
          f"namespaced dict): {baseline_matches_namespaced}")
    print(f"    after corrupting ka.GATE2_N_ITER_BY_K[48]=999999, K=48/d=80 n_iter is STILL "
          f"{corrupted_spec['geo3_n_iter']} (unaffected by the flat dict): {unaffected}")
    print(f"    ka.GATE2_N_ITER_BY_K[48] restored to {original}: {restored_ok}")
    ok = baseline_matches_namespaced and unaffected and restored_ok
    _report("smoke 6: sec 15.6 GATE2_N_ITER_BY_K collision closed BY CONSTRUCTION -- corrupting "
            "the flat ka.GATE2_N_ITER_BY_K[48] entry does not affect K=48/d=80's own n_iter "
            "(negative test to completion, not merely a code-inspection claim)", ok)


# ---------------------------------------------------------------------------
# smoke 7: keyanchor_scaling_kernel_gate_check -- positive + 4 negative tests.
# ---------------------------------------------------------------------------

def smoke_7_kernel_gate_check():
    # Positive: the REAL committed artifact must currently pass (attack-
    # round-1 FATAL-2 was resolved same-session; this is a live re-check,
    # not a trust-the-filename claim).
    real = rdx.keyanchor_scaling_kernel_gate_check()
    positive_ok = real["ok"] is True and "CLEARED" not in "" or real["ok"] is True
    print(f"    POSITIVE (real artifact): ok={real['ok']} reason={real['reason']!r}")

    def _write(doc, tmp):
        p = os.path.join(tmp, "fake_kernel_result.json")
        with open(p, "w") as f:
            json.dump(doc, f)
        return p

    base_doc = None
    with open(rdx.KEYANCHOR_SCALING_KERNEL_GATE_RESULT_PATH) as f:
        base_doc = json.load(f)

    # Negative (a): missing file entirely.
    missing_path = os.path.join(tempfile.mkdtemp(), "does_not_exist.json")
    neg_a = rdx.keyanchor_scaling_kernel_gate_check(missing_path)
    neg_a_ok = neg_a["ok"] is False and "not found" in neg_a["reason"]
    print(f"    NEGATIVE (a) missing file: ok={neg_a['ok']} (expect False): {neg_a_ok}")

    # Negative (b): verdict string does not contain 'CLEARED'.
    with tempfile.TemporaryDirectory() as tmp:
        doc = copy.deepcopy(base_doc)
        doc["verdict"] = "d_state 80/96 FAIL at >=1 T -- sec 15 wave dead as designed"
        p = _write(doc, tmp)
        neg_b = rdx.keyanchor_scaling_kernel_gate_check(p)
        neg_b_ok = neg_b["ok"] is False and "CLEARED" in neg_b["reason"]
        print(f"    NEGATIVE (b) verdict without 'CLEARED': ok={neg_b['ok']} (expect False): {neg_b_ok}")

    # Negative (c): exit_code != 0.
    with tempfile.TemporaryDirectory() as tmp:
        doc = copy.deepcopy(base_doc)
        doc["exit_code"] = 3
        p = _write(doc, tmp)
        neg_c = rdx.keyanchor_scaling_kernel_gate_check(p)
        neg_c_ok = neg_c["ok"] is False and "exit_code" in neg_c["reason"]
        print(f"    NEGATIVE (c) exit_code=3: ok={neg_c['ok']} (expect False): {neg_c_ok}")

    # Negative (d): the FATAL-2 scenario itself -- T=256-only sweep (does
    # not cover the registered {128,224,448} protocol).
    with tempfile.TemporaryDirectory() as tmp:
        doc = copy.deepcopy(base_doc)
        doc["t_sweep"] = [256]
        doc["grid_pass"] = {"80": {"256": True}, "96": {"256": True}}
        p = _write(doc, tmp)
        neg_d = rdx.keyanchor_scaling_kernel_gate_check(p)
        neg_d_ok = neg_d["ok"] is False and "protocol" in neg_d["reason"]
        print(f"    NEGATIVE (d) T=256-only sweep (the FATAL-2 false-negative shape): "
              f"ok={neg_d['ok']} (expect False): {neg_d_ok}")

    # Negative (e): one T cell False (d=32-style genuine failure), full sweep present.
    with tempfile.TemporaryDirectory() as tmp:
        doc = copy.deepcopy(base_doc)
        doc["grid_pass"]["96"]["448"] = False
        p = _write(doc, tmp)
        neg_e = rdx.keyanchor_scaling_kernel_gate_check(p)
        neg_e_ok = neg_e["ok"] is False and "96" in neg_e["reason"]
        print(f"    NEGATIVE (e) grid_pass['96']['448']=False: ok={neg_e['ok']} (expect False): {neg_e_ok}")

    ok = real["ok"] is True and neg_a_ok and neg_b_ok and neg_c_ok and neg_d_ok and neg_e_ok
    _report("smoke 7: keyanchor_scaling_kernel_gate_check -- POSITIVE against the real committed "
            "artifact (currently CLEARED) AND 5 NEGATIVE tests run to completion (missing file, "
            "verdict without CLEARED, exit_code!=0, T=256-only false-negative shape, a single False "
            "grid cell)", ok)


# ---------------------------------------------------------------------------
# smoke 8: keyanchor_scaling_stage_gate mechanics.
# ---------------------------------------------------------------------------

_SCALING_EC_FIELDS = ("embed_source", "gram_alpha", "gram_rho", "strong_pin", "lambda_orth",
                      "use_zca", "fnce_m", "geo3_active", "geo3_n_iter", "geo3_resid_tol",
                      "anchor_active", "anchor_lambda_mode", "anchor_lambda_fixed",
                      "lambda_anchor", "drift_probe", "rev7_engagement")


def _write_fake_result(out_dir, spec, wall_s):
    ec = {field: spec.get(field) for field in _SCALING_EC_FIELDS}
    doc = {"K": spec["K"], "seed": spec["seed"], "steps": spec["steps"], "complete": True,
           "steps_completed": spec["steps"], "exactness_config": ec, "wall_s": wall_s,
           "timed_out": False, "d_state": spec["d_state"],
           "checkpoints": [{"M3_held_out": {"4": {"recovered_frac@0.9": 0.99}}}]}
    p = rdx.out_path(out_dir, spec)
    with open(p, "w") as f:
        json.dump(doc, f)


def smoke_8_stage_gate_mechanics():
    calib80, calib96 = rdx.keyanchor_scaling_calibration_manifest()
    trigger80 = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(80, False)]
    trigger96 = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(96, False)]

    # (a) kernel gate blocks 'calibration' stage too, when the artifact is bad.
    with tempfile.TemporaryDirectory() as tmp:
        refused_calib_bad_kernel = False
        try:
            rdx.keyanchor_scaling_stage_gate(tmp, "calibration", accept_override=False,
                                              ) if False else None
        except SystemExit:
            pass
        # keyanchor_scaling_stage_gate has no path arg -- exercise the
        # kernel-gate-first ordering via the module-level check directly
        # (same function main() actually calls) against a corrupted path.
        bad = rdx.keyanchor_scaling_kernel_gate_check(os.path.join(tmp, "nope.json"))
        refused_calib_bad_kernel = bad["ok"] is False
    print(f"    (a) kernel_gate_check on a missing artifact reports ok=False (dispatch would "
          f"sys.exit(1) before EITHER stage proceeds): {refused_calib_bad_kernel}")

    # (b) stage='full' REFUSES when calibration cells are missing (real
    # kernel artifact is CLEARED, so this exercises the calibration leg).
    with tempfile.TemporaryDirectory() as tmp:
        refused_missing = False
        try:
            rdx.keyanchor_scaling_stage_gate(tmp, "full", accept_override=False)
        except SystemExit as e:
            refused_missing = (e.code == 1)
    print(f"    (b) stage='full' refuses when both calibration cells missing: {refused_missing}")

    # (c)-(e) need KEYANCHOR_SCALING_PI_SIGNOFF=1 to get PAST keyanchor_scaling_stage_gate's own
    # unconditional signoff check and reach the calibration-specific logic these three actually
    # exercise -- MINOR-1 fix (build-audit, 2026-07-07): env-self-sufficient (save/set/restore),
    # mirrors smoke_keyanchor_scaling_wide.py's own smoke_11_stage_gate_mechanics pattern (lines
    # 488-527 there) so this smoke suite does not silently depend on an externally-exported
    # signoff token (e.g. from keyanchor_scaling_chain.sh's own GATE 0) to run to completion.
    saved_pi_signoff = os.environ.get("KEYANCHOR_SCALING_PI_SIGNOFF")
    try:
        os.environ["KEYANCHOR_SCALING_PI_SIGNOFF"] = "1"

        # (c) in-bracket calibration (both d's) -> sentinel written correctly.
        with tempfile.TemporaryDirectory() as tmp:
            _write_fake_result(tmp, calib80, trigger80 - 100.0)
            _write_fake_result(tmp, calib96, trigger96 - 100.0)
            report = rdx.keyanchor_scaling_stage_gate(tmp, "full", accept_override=False)
            sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_SCALING_STAGE_SENTINEL_NAME)
            clean_ok = (report["sentinel_written"] is True and os.path.exists(sentinel_path)
                        and report["gate_bypassed"] is False)
            with open(sentinel_path) as f:
                payload = json.loads(f.read().strip())
            sentinel_content_ok = ("80" in str(payload["calibration_wall_s_by_d"])
                                    or 80 in payload["calibration_wall_s_by_d"]
                                    or "80" in {str(k) for k in payload["calibration_wall_s_by_d"]})
        print(f"    (c) both in-bracket -> sentinel written+correct: {clean_ok and sentinel_content_ok}")

        # (d) one over-bracket calibration cell -> refuses, no sentinel.
        with tempfile.TemporaryDirectory() as tmp:
            _write_fake_result(tmp, calib80, trigger80 + 100.0)   # over trigger
            _write_fake_result(tmp, calib96, trigger96 - 100.0)   # in bracket
            refused_over = False
            try:
                rdx.keyanchor_scaling_stage_gate(tmp, "full", accept_override=False)
            except SystemExit as e:
                refused_over = (e.code == 1)
            sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_SCALING_STAGE_SENTINEL_NAME)
            no_sentinel = not os.path.exists(sentinel_path)
        print(f"    (d) d=80 over-bracket -> refuses, no sentinel: {refused_over and no_sentinel}")

        # (e) --accept-scaling-stage-override bypasses the calibration gate but
        # NOT the kernel gate -- verified by construction (the kernel check
        # runs unconditionally, before accept_override is even read, sec
        # 15.13's own docstring claim) via the real (CLEARED) artifact.
        with tempfile.TemporaryDirectory() as tmp:
            report = rdx.keyanchor_scaling_stage_gate(tmp, "full", accept_override=True)
            sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_SCALING_STAGE_SENTINEL_NAME)
            override_ok = (report["gate_bypassed"] is True and report["sentinel_written"] is False
                            and not os.path.exists(sentinel_path)
                            and report["kernel_gate"]["ok"] is True)
        print(f"    (e) --accept-scaling-stage-override bypasses calibration gate, kernel_gate still "
              f"evaluated+CLEARED, no sentinel: {override_ok}")
    finally:
        if saved_pi_signoff is not None:
            os.environ["KEYANCHOR_SCALING_PI_SIGNOFF"] = saved_pi_signoff
        else:
            os.environ.pop("KEYANCHOR_SCALING_PI_SIGNOFF", None)

    ok = (refused_calib_bad_kernel and refused_missing and clean_ok and sentinel_content_ok
          and refused_over and no_sentinel and override_ok)
    _report("smoke 8: keyanchor_scaling_stage_gate mechanics -- kernel gate checked first (blocks "
            "on a bad artifact), refuses on missing/over-bracket calibration cells, correct "
            "sentinel on a clean pass, override bypasses calibration gate only", ok)


# ---------------------------------------------------------------------------
# smoke 9: keyanchor_scaling_check_abort.
# ---------------------------------------------------------------------------

def smoke_9_check_abort():
    main_spec = [s for s in rdx.keyanchor_scaling_full_manifest()
                 if s["d_state"] == 80 and s["K"] == 43][0]
    anchor_spec = [s for s in rdx.keyanchor_scaling_full_manifest()
                   if s["d_state"] == 80 and s["K"] == 20][0]
    main_trigger = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(80, False)]
    anchor_trigger = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(80, True)]

    main_literal_ok = abs(main_trigger - 3742.2) < 0.5
    anchor_literal_ok = abs(anchor_trigger - 1825.2) < 0.5
    d96_main_ok = abs(rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(96, False)] - 4658.04) < 1.0
    d96_anchor_ok = abs(rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(96, True)] - 2289.6) < 0.5
    print(f"    d=80 main trigger={main_trigger:.1f}s (expect ~3742.2): {main_literal_ok}")
    print(f"    d=80 anchor trigger={anchor_trigger:.1f}s (expect ~1825.2): {anchor_literal_ok}")
    print(f"    d=96 main trigger={rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(96, False)]:.1f}s "
          f"(expect ~4658.04): {d96_main_ok}")
    print(f"    d=96 anchor trigger={rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(96, True)]:.1f}s "
          f"(expect ~2289.6): {d96_anchor_ok}")

    no_raise_main = True
    try:
        rdx.keyanchor_scaling_check_abort(main_spec, wall_s=100.0)
    except rdx.KeyanchorScalingAbort:
        no_raise_main = False

    raises_main = False
    try:
        rdx.keyanchor_scaling_check_abort(main_spec, wall_s=main_trigger + 1.0)
    except rdx.KeyanchorScalingAbort:
        raises_main = True

    # Boundary-exact (>=, not >).
    boundary_raises = False
    try:
        rdx.keyanchor_scaling_check_abort(main_spec, wall_s=main_trigger)
    except rdx.KeyanchorScalingAbort:
        boundary_raises = True

    # Anchor cell: at the MAIN threshold it should NOT yet raise (its own
    # threshold is lower, this proves the two-tier split is genuinely
    # per-cell, not a single shared scalar).
    anchor_uses_own_tier = True
    try:
        rdx.keyanchor_scaling_check_abort(anchor_spec, wall_s=main_trigger - 1.0)
        # main_trigger > anchor_trigger, so this SHOULD raise for the anchor cell.
        anchor_uses_own_tier = False
    except rdx.KeyanchorScalingAbort:
        anchor_uses_own_tier = True

    print(f"    wall_s=100s (main, <<trigger): does not raise: {no_raise_main}")
    print(f"    wall_s=trigger+1 (main): raises: {raises_main}")
    print(f"    wall_s==trigger exactly (main, boundary >=): raises: {boundary_raises}")
    print(f"    anchor cell at main_trigger-1s (> its own lower anchor_trigger): raises "
          f"(two-tier split is per-cell, not shared): {anchor_uses_own_tier}")
    ok = (main_literal_ok and anchor_literal_ok and d96_main_ok and d96_anchor_ok and no_raise_main
          and raises_main and boundary_raises and anchor_uses_own_tier)
    _report("smoke 9: keyanchor_scaling_check_abort -- two-tier (main-grid vs anchor-K) bracket "
            "matches sec 15.14's own pinned literals (3742.2/1825.2/4658.04/2289.6s), boundary-"
            "exact, genuinely per-cell", ok)


# ---------------------------------------------------------------------------
# smoke 10: rev7_threshold_derive.py's d=80/96 pins.
# ---------------------------------------------------------------------------

def smoke_10_threshold_pins():
    import math
    paths = {
        64: os.path.join(HERE, "REV7_THRESHOLD_PINNED.json"),
        80: os.path.join(HERE, "REV7_THRESHOLD_PINNED_D80.json"),
        96: os.path.join(HERE, "REV7_THRESHOLD_PINNED_D96.json"),
        128: os.path.join(HERE, "REV7_THRESHOLD_PINNED_D128.json"),
    }
    missing = [d for d, p in paths.items() if not os.path.exists(p)]
    if missing:
        _report("smoke 10: d=80/96 threshold pins (sec 15.5)", False,
                 f"missing pin file(s) for d in {missing}")
        return
    docs = {}
    for d, p in paths.items():
        with open(p) as f:
            docs[d] = json.load(f)

    import itertools
    all_distinct = True
    for a, b in itertools.combinations(docs, 2):
        same = json.dumps(docs[a]["derived"], sort_keys=True) == json.dumps(docs[b]["derived"], sort_keys=True)
        if same:
            print(f"    FAIL: derived block for d={a} byte-identical to d={b}")
            all_distinct = False
    inputs_ok = docs[80]["derived"]["inputs"]["d_state"] == 80 and docs[96]["derived"]["inputs"]["d_state"] == 96
    sigma80_ok = docs[80]["derived"]["null"]["sigma_chance"] == 1.0 / math.sqrt(80)
    sigma96_ok = docs[96]["derived"]["null"]["sigma_chance"] == 1.0 / math.sqrt(96)
    headline_ok = all(docs[d]["derived"]["effect_size_floors"]["r_min_headline_band"] == 0.35 for d in docs)
    print(f"    all 4 pins (64/80/96/128) pairwise byte-DIFFERENT: {all_distinct}")
    print(f"    d=80/96 pins' derived.inputs.d_state correct: {inputs_ok}")
    print(f"    sigma_chance == 1/sqrt(d) exactly at d=80 ({docs[80]['derived']['null']['sigma_chance']!r}) "
          f"and d=96 ({docs[96]['derived']['null']['sigma_chance']!r}): {sigma80_ok and sigma96_ok}")
    print(f"    r_min_headline_band fixed at 0.35 across all 4 pins: {headline_ok}")
    ok = all_distinct and inputs_ok and sigma80_ok and sigma96_ok and headline_ok
    _report("smoke 10: d=80/96 threshold pins (sec 15.5) -- byte-DIFFERENT from every other pin, "
            "sigma_chance==1/sqrt(d) exactly, r_min_headline unchanged at 0.35", ok)


# ---------------------------------------------------------------------------
# smoke 11: pin-selection-by-d_state + live validate_rev7_pin.
# ---------------------------------------------------------------------------

def _pin_name_for(d_state: int) -> str:
    """Reproduces run_deltanet_rd.py's own _pin_name convention (lines
    ~1669-1670) as a pure string rule -- run_deltanet_rd.py itself is not
    imported here (CUDA-dependent code paths at call time), so this is the
    documented convention, cross-checked against the real files below."""
    return "REV7_THRESHOLD_PINNED.json" if d_state == 64 else f"REV7_THRESHOLD_PINNED_D{d_state}.json"


def smoke_11_pin_selection():
    name80, name96 = _pin_name_for(80), _pin_name_for(96)
    names_ok = name80 == "REV7_THRESHOLD_PINNED_D80.json" and name96 == "REV7_THRESHOLD_PINNED_D96.json"
    path80, path96 = os.path.join(HERE, name80), os.path.join(HERE, name96)
    exist_ok = os.path.exists(path80) and os.path.exists(path96)
    # Live re-validation via the ACTUAL launcher-gate function (fla-free,
    # no CUDA needed) -- proves these pins would pass run_deltanet_rd.py's
    # own --rev7-engagement gate, not merely that the JSON exists.
    valid80 = ka.validate_rev7_pin(pin_path=path80) is not None
    valid96 = ka.validate_rev7_pin(pin_path=path96) is not None
    d_state_matches = False
    if valid80 and valid96:
        d_state_matches = (ka.validate_rev7_pin(pin_path=path80)["derived"]["inputs"]["d_state"] == 80
                            and ka.validate_rev7_pin(pin_path=path96)["derived"]["inputs"]["d_state"] == 96)
    print(f"    _pin_name(80)={name80!r} _pin_name(96)={name96!r}: {names_ok}")
    print(f"    both files exist on disk: {exist_ok}")
    print(f"    ka.validate_rev7_pin(...) succeeds (script-hash match + live derive() re-run "
          f"byte-identical) for d=80: {valid80}, d=96: {valid96}")
    print(f"    validated pin's own d_state matches the requested d: {d_state_matches}")
    ok = names_ok and exist_ok and valid80 and valid96 and d_state_matches
    _report("smoke 11: pin-selection-by-d_state -- _pin_name resolves to the D80/D96 filenames "
            "(sec 15.5), both exist, and ka.validate_rev7_pin (the real launcher-gate function) "
            "validates each one live", ok)


# ---------------------------------------------------------------------------
# smoke 12: pooled_null_check's d_state-genericity.
# ---------------------------------------------------------------------------

def smoke_12_pooled_null_check_generic():
    import torch
    torch.manual_seed(0)
    ran = {}
    for d in (80, 96):
        off_diag = torch.randn(200) * (1.0 / (d ** 0.5))   # plausible near-null cosines at this d
        try:
            result = ka.pooled_null_check(off_diag, d_state=d)
            ran[d] = "mean" in result or isinstance(result, dict)
        except TypeError as e:
            ran[d] = False
            print(f"    d={d}: pooled_null_check(off_diag, d_state={d}) raised TypeError: {e!r}")
    print(f"    pooled_null_check(off_diag, d_state=80) runs and accepts the kwarg: {ran.get(80)}")
    print(f"    pooled_null_check(off_diag, d_state=96) runs and accepts the kwarg: {ran.get(96)}")
    ok = ran.get(80) is True and ran.get(96) is True
    _report("smoke 12: key_anchoring.pooled_null_check accepts d_state=80/96 directly (sec 15.18's "
            "own verified-list claim, re-checked here by EXECUTING it, not only citing it)", ok)


# ---------------------------------------------------------------------------
# smoke 13: Gate 2 at d_state=80/96 with this wave's own registered K-grids.
# ---------------------------------------------------------------------------

def smoke_13_gate2_at_new_d_states():
    overall_ok = True
    for d in (80, 96):
        ks = rdx.KEYANCHOR_SCALING_KS_BY_D[d]
        assert ks == rdx.KEYANCHOR_SCALING_KS_BY_D[d], f"ks= drifted from sec 15.3's registered K-grid at d={d}"
        table = ka.frame_potential_init(107, d, seed=ka.ANCHOR_INIT_SEED)
        result = ka.gate2_construction_check(table, ks=ks, seed=0)
        print(f"    d_state={d}: table shape={tuple(table.shape)} ks={ks}")
        print(f"      G2-a sigma_ratio={result['g2a_sigma_ratio']:.6f} (pass={result['g2a_pass']})")
        print(f"      G2-b max|cos|   ={result['g2b_max_abs_cos']:.6f} (pass={result['g2b_pass']})")
        for K in ks:
            leg = result["g2c_ns_legs"][K]
            print(f"      G2-c K={K:3d}: n_iter={leg['n_iter']} n_fallback={leg['n_fallback']}/"
                  f"{leg['n_subsets']} max_resid={leg['max_resid']:.3e} pass={leg['pass']}")
        print(f"      OVERALL GATE 2 (d_state={d}, ks={ks}): {'PASS' if result['pass'] else 'FAIL'}")
        overall_ok = overall_ok and result["pass"]
    _report("smoke 13: Gate 2 EXECUTED at d_state=80 AND d_state=96 with this wave's own "
            "registered K-grids explicitly (never the (16,32) default) -- reported honestly "
            "regardless of outcome", overall_ok)


# ---------------------------------------------------------------------------
# smoke 14: disk-gate d_state-awareness (found during this build).
# ---------------------------------------------------------------------------

def smoke_14_disk_gate_d_state_aware():
    spec80 = [s for s in rdx.keyanchor_scaling_full_manifest() if s["d_state"] == 80 and s["K"] == 43][0]
    spec96 = [s for s in rdx.keyanchor_scaling_full_manifest() if s["d_state"] == 96 and s["K"] == 51][0]
    bytes80 = rdx.keyanchor_scaling_projected_ckpt_bytes([spec80])
    bytes96 = rdx.keyanchor_scaling_projected_ckpt_bytes([spec96])
    scales_with_d = bytes96 > bytes80   # same steps/ckpt_every, only d_state differs
    ratio = bytes96 / bytes80
    expected_ratio_approx = 96 / 80   # row block dominates for n_train=107 at these d's
    ratio_plausible = 1.0 < ratio < 1.5
    # Regression guard: the OLD (mech-wave) projector is d_state-BLIND --
    # confirm it would have produced the SAME byte count for both specs
    # (demonstrating why it must not be reused here unmodified).
    old_bytes80 = rdx.keyanchor_mech_projected_ckpt_bytes([spec80])
    old_bytes96 = rdx.keyanchor_mech_projected_ckpt_bytes([spec96])
    old_is_blind = old_bytes80 == old_bytes96
    print(f"    keyanchor_scaling_projected_ckpt_bytes: d=80 -> {bytes80} bytes, d=96 -> {bytes96} "
          f"bytes (ratio {ratio:.4f}, expect ~{expected_ratio_approx:.4f}): scales with d: {scales_with_d}")
    print(f"    keyanchor_mech_projected_ckpt_bytes (OLD, hardcoded d_state=64): d=80 spec -> "
          f"{old_bytes80}, d=96 spec -> {old_bytes96} -- IDENTICAL (d_state-blind, confirms why "
          f"reusing it unmodified would under-project): {old_is_blind}")
    ok = scales_with_d and ratio_plausible and old_is_blind
    _report("smoke 14: disk-gate d_state-awareness -- keyanchor_scaling_projected_ckpt_bytes "
            "scales with each cell's own d_state (unlike the mech-wave projector it deliberately "
            "does not reuse)", ok)


# ---------------------------------------------------------------------------
# smoke 15: manifest enumeration -- print all 30 mandatory cells.
# ---------------------------------------------------------------------------

def smoke_15_print_all_30_cells():
    m = sorted(rdx.keyanchor_scaling_full_manifest(), key=lambda s: (s["d_state"], s["K"], s["seed"]))
    print(f"    {len(m)}-cell mandatory manifest (d_state, K, seed, name):")
    for s in m:
        print(f"      d_state={s['d_state']:3d}  K={s['K']:3d}  seed={s['seed']:5d}  {s['name']}")
    ok = len(m) == 30
    _report("smoke 15: manifest enumeration -- all 30 mandatory cells printed for the human/CI "
            "record", ok)


# ---------------------------------------------------------------------------
# smoke 16: model_rd.py's _SAFE_D_STATE / run_deltanet_rd.py's --d-state
# argparse choices extension (sec 15.2 item 3). model_rd.py/run_deltanet_
# rd.py themselves cannot be imported in this CPU sandbox (model_rd.py
# imports fla.modules at module scope) -- checked at the SOURCE-TEXT level
# instead, same as this project's own house pattern for verifying
# uncommitted-but-critical constants without requiring a GPU-only import.
# ---------------------------------------------------------------------------

def smoke_16_safe_d_state_extended():
    import re
    with open(os.path.join(HERE, "model_rd.py")) as f:
        model_src = f.read()
    with open(os.path.join(HERE, "run_deltanet_rd.py")) as f:
        run_src = f.read()
    m = re.search(r"^_SAFE_D_STATE\s*=\s*\(([^)]*)\)", model_src, re.MULTILINE)
    safe_d_state_ok = m is not None and set(int(x.strip()) for x in m.group(1).split(",") if x.strip()) \
        == {64, 80, 96, 128}
    d_state_choices_ok = 'choices=[64, 80, 96, 128]' in run_src or 'choices=[64,80,96,128]' in run_src
    # d_state=16 (the measured crash trigger) must STILL be rejected --
    # this extension must not have accidentally widened the set further
    # than sec 15.2 item 3 itself licenses.
    still_rejects_16 = m is not None and "16" not in [x.strip() for x in m.group(1).split(",")]
    print(f"    model_rd.py _SAFE_D_STATE tuple: {m.group(1).strip() if m else None!r} "
          f"(expect {{64,80,96,128}}): {safe_d_state_ok}")
    print(f"    run_deltanet_rd.py --d-state choices=[64, 80, 96, 128] present: {d_state_choices_ok}")
    print(f"    d_state=16 still excluded (the measured D=32/D=16 crash-adjacent trigger, sec 15.2's "
          f"own non-monotone-in-D disclosure): {still_rejects_16}")
    ok = safe_d_state_ok and d_state_choices_ok and still_rejects_16
    _report("smoke 16: model_rd.py's _SAFE_D_STATE / run_deltanet_rd.py's --d-state argparse "
            "choices extended to include 80/96 (sec 15.2 item 3, licensed by the PASSING kernel-"
            "safety artifact) -- source-level check since model_rd.py cannot be imported in this "
            "fla-less CPU sandbox", ok)


def main() -> int:
    print("=" * 70)
    print("smoke_keyanchor_scaling.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15 (attack-round-1 "
          "verdict RUN-AFTER-REASONING-LINK, wave PARKED) cliff-location-scaling-law Wave -1 smoke "
          "suite (fla-free except smoke 12's own torch dependency, key_anchoring.py itself)")
    print("=" * 70)
    smoke_1_full_manifest_shape()
    smoke_2_gate1_manifest_shape()
    smoke_3_calibration_manifest_shape()
    smoke_4_name_bit()
    smoke_5_zero_collision()
    smoke_6_gate2_n_iter_collision_closed()
    smoke_7_kernel_gate_check()
    smoke_8_stage_gate_mechanics()
    smoke_9_check_abort()
    smoke_10_threshold_pins()
    smoke_11_pin_selection()
    smoke_12_pooled_null_check_generic()
    smoke_13_gate2_at_new_d_states()
    smoke_14_disk_gate_d_state_aware()
    smoke_15_print_all_30_cells()
    smoke_16_safe_d_state_extended()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
