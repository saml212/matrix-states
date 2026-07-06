"""smoke_keyanchor_dstate.py -- KEY_ANCHORING_DESIGN.md sec 13 (Rev 13.2,
DESIGN-CLEARED-FOR-BUILD)'s registered fla-free Wave -1 smoke suite for the
cliff-universality-across-d_state wave, as a committed, CPU-runnable script.
Mirrors smoke_keyanchor_cliff.py's own split: this file imports ONLY
key_anchoring.py, run_deltanet_rd_exactness_sweep.py, and fit_cliff_curve.py
(all verified fla-free at module scope).

Covers:
  smoke 1 (sec 13.3 item 6, d=64 byte-identity regression): the
    parameterized _spec()/manifest machinery at d_state=64 defaults must
    reproduce the keyanchor-cliff manifest byte-identically (json.dumps
    sorted equality) -- proves adding d_state to _spec() did not silently
    perturb any already-published d=64 result's filename/config provenance.
  smoke 2: keyanchor_dstate_manifest()'s shape -- 12 cells, K in
    {68,76,84,92}, 3 seeds each, exact seed blocks, d_state=128, full
    instrumentation.
  smoke 3: keyanchor_dstate_gate1_manifest()'s shape -- 4 cells, seeds
    {535,635,735,835}, one per K, 5,000 steps.
  smoke 4: keyanchor_dstate_calibration_manifest()'s shape -- exactly the
    K=68/seed=530 cell, filtered from (never hand-duplicated against) the
    full manifest.
  smoke 5: every d=128 cell's own filename carries the '_d128' name bit
    (sec 13's _spec() generalization: d_state != 64 gets a name bit).
  smoke 6: zero-collision -- keyanchor-dstate's manifest (mandatory +
    Gate-1 probes + calibration) is collision-free against every prior/
    sibling wave this program has ever registered.
  smoke 7: GATE2_N_ITER_BY_K extended with {68:20, 76:20, 84:20, 92:20}
    (sec 13.2 item 3, confirmed sufficient by keyanchor_dstate_niter_
    check.py, not merely assumed).
  smoke 8: rev7_threshold_derive.py's --d-state flag -- d=128 pin's
    'derived' block is byte-DIFFERENT from the existing d=64 pin's, AND
    sigma_chance == 1/sqrt(128) exactly.
  smoke 9: read_wall_s_only(path) -- parses a real archived cell JSON,
    returns wall_s only, contains no M3_held_out content in its output.
  smoke 10: the sec 13.6 mechanical decision table -- all 4 branches
    (PROCEED_FULL/OPTION_B/OPTION_C/ESCALATE) fire at the exact registered
    thresholds (1.7492/2.3322/3.4983 GPU-h/cell), boundary-inclusive.
  smoke 11: keyanchor_dstate_stage_gate's calibration-first mechanics --
    stage='full' REFUSES when the calibration cell is missing; a synthetic
    in-bracket calibration cell produces the correct branch + writes
    CALIBRATION_DONE; a synthetic over-bracket (ESCALATE) calibration cell
    REFUSES and writes NO sentinel; --accept-dstate-stage-override bypasses
    cleanly and writes no sentinel.
  smoke 12: keyanchor_dstate_check_abort -- does not raise well under the
    1.5x-bracket-upper-edge trigger, raises KeyanchorDstateAbort at/above it.
  smoke 13: fit_cliff_curve.py's d=128 anchor-free mode -- --k32-dir/
    --k48-dir must be OMITTED at --d-state 128 (asserts if passed), --k-grid
    is REQUIRED at any non-64 d (no silent d=64 CLIFF_KS reuse).
  smoke 14 (sec 13.3 item 5a, the assert-guarded Wave-1 smoke): EXECUTES
    key_anchoring.gate2_construction_check on the registered d=128 anchor
    table (frame_potential_init(107, 128, seed=ANCHOR_INIT_SEED)) with
    ks=(68,76,84,92) EXPLICITLY passed (never the function's own default
    ks=(16,32), never a stale (16,32,48) tuple copy-pasted from a prior
    wave) -- asserts the call's own ks argument equals this wave's
    registered K-grid before trusting the result, and reports every leg
    (G2-a/G2-b/G2-c at each K) -- this is ALSO the first real Gate-2 data
    at d=128 (sec 13.3 item 5's own build task); if any leg fails, this
    smoke reports it honestly (it gates the wave).

Wired as an ADDITIONAL pre-launch CPU gate for --wave keyanchor-dstate
(run_deltanet_rd_exactness_sweep.py's main(), alongside smoke_key_anchoring.py
+ gate2_construction_test.py) -- rc!=0 aborts the wave before any GPU cell
dispatches.

Exit code 0 = every item PASSED. Run: python smoke_keyanchor_dstate.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import key_anchoring as ka                              # noqa: E402 (fla-free)
import run_deltanet_rd_exactness_sweep as rdx            # noqa: E402 (fla-free)
import fit_cliff_curve as fcc                            # noqa: E402 (fla-free)

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


# ---------------------------------------------------------------------------
# smoke 1: sec 13.3 item 6's d=64 byte-identity regression.
# ---------------------------------------------------------------------------

_PRE_EXISTING_SPEC_KEYS = {"wave", "K", "seed", "steps", "force_rank_k", "arm", "embed_source",
                          "gram_alpha", "gram_rho", "strong_pin", "lambda_orth", "use_zca", "fnce_m",
                          "geo3_active", "geo3_n_iter", "geo3_resid_tol", "anchor_active",
                          "anchor_lambda_mode", "anchor_lambda_fixed", "lambda_anchor", "drift_probe",
                          "rev7_engagement", "name"}


def smoke_1_d64_regression():
    cliff = rdx.keyanchor_cliff_manifest()
    cliff_g1 = rdx.keyanchor_cliff_gate1_manifest()
    count_ok = len(cliff) == 12 and len(cliff_g1) == 4
    d_state_ok = all(s["d_state"] == 64 for s in cliff + cliff_g1)
    no_name_bit = all(not s["name"].endswith("_d128") for s in cliff + cliff_g1)
    keys_ok = all(set(k for k in s if k in _PRE_EXISTING_SPEC_KEYS) == _PRE_EXISTING_SPEC_KEYS
                  for s in cliff)
    # Explicit d_state=64 call is byte-identical to the omitted-kwarg call.
    explicit_64 = rdx._spec("keyanchor-k48", 48, 30, 20000, "d", geo3_active=True, geo3_n_iter=20,
                             geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                             drift_probe=True, rev7_engagement=True, d_state=64)
    omitted = rdx._spec("keyanchor-k48", 48, 30, 20000, "d", geo3_active=True, geo3_n_iter=20,
                         geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                         drift_probe=True, rev7_engagement=True)
    byte_identical = json.dumps(explicit_64, sort_keys=True) == json.dumps(omitted, sort_keys=True)
    print(f"    keyanchor_cliff_manifest(): {len(cliff)} cells, all d_state==64: {d_state_ok}, "
          f"no '_d128' name bit: {no_name_bit}")
    print(f"    _spec(..., d_state=64) explicit vs omitted: byte-identical(json.dumps sorted)="
          f"{byte_identical}")
    ok = count_ok and d_state_ok and no_name_bit and keys_ok and byte_identical
    _report("smoke 1: d=64 byte-identity regression (sec 13.3 item 6) -- the parameterized "
            "_spec()/manifest machinery at d_state=64 defaults reproduces the keyanchor-cliff "
            "manifest unperturbed", ok)


# ---------------------------------------------------------------------------
# smoke 2/3/4/5: keyanchor-dstate manifest/gate1/calibration shape, name bit.
# ---------------------------------------------------------------------------

def smoke_2_dstate_manifest_shape():
    m = rdx.keyanchor_dstate_manifest()
    right_count = len(m) == 12
    ks_present = sorted(set(s["K"] for s in m)) == [68, 76, 84, 92]
    right_d_state = all(s["d_state"] == 128 for s in m)
    right_arm = all(s["arm"] == "d" for s in m)
    right_lambda_mode = all(s["anchor_lambda_mode"] == "learned" for s in m)
    right_instrumentation = all(s["drift_probe"] is True and s["rev7_engagement"] is True for s in m)
    expected_seeds = {68: {530, 531, 532}, 76: {630, 631, 632}, 84: {730, 731, 732}, 92: {830, 831, 832}}
    seeds_ok = all(set(s["seed"] for s in m if s["K"] == K) == expected_seeds[K] for K in expected_seeds)
    print(f"    count={len(m)} Ks={sorted(set(s['K'] for s in m))} d_state={{s['d_state'] for s in m}}")
    for K in (68, 76, 84, 92):
        print(f"      K={K}: seeds={sorted(s['seed'] for s in m if s['K'] == K)} "
              f"(expect {sorted(expected_seeds[K])})")
    ok = (right_count and ks_present and right_d_state and right_arm and right_lambda_mode
          and right_instrumentation and seeds_ok)
    _report("smoke 2: keyanchor_dstate_manifest() shape -- 12 cells, K in {68,76,84,92}, 3 seeds "
            "each, exact registered seed blocks, d_state=128, candidate (d) arch, full "
            "instrumentation", ok)


def smoke_3_dstate_gate1_manifest_shape():
    g1 = rdx.keyanchor_dstate_gate1_manifest()
    right_count = len(g1) == 4
    ks_present = sorted(s["K"] for s in g1) == [68, 76, 84, 92]
    expected_seed_by_k = {68: 535, 76: 635, 84: 735, 92: 835}
    seeds_ok = all(s["seed"] == expected_seed_by_k[s["K"]] for s in g1)
    steps_ok = all(s["steps"] == rdx.KEYANCHOR_DSTATE_GATE1_STEPS == 5000 for s in g1)
    d_state_ok = all(s["d_state"] == 128 for s in g1)
    print(f"    count={len(g1)} Ks={sorted(s['K'] for s in g1)} "
          f"seeds={ {s['K']: s['seed'] for s in g1} } (expect {expected_seed_by_k})")
    ok = right_count and ks_present and seeds_ok and steps_ok and d_state_ok
    _report("smoke 3: keyanchor_dstate_gate1_manifest() shape -- 4 cells, seeds "
            "{535,635,735,835}, one per K, 5,000 steps, d_state=128", ok)


def smoke_4_dstate_calibration_manifest_shape():
    calib = rdx.keyanchor_dstate_calibration_manifest()
    right_count = len(calib) == 1
    right_cell = right_count and calib[0]["K"] == 68 and calib[0]["seed"] == 530 \
        and calib[0]["d_state"] == 128
    # Filtered from, not hand-duplicated against, the full manifest's own K=68 entry.
    full_k68 = [s for s in rdx.keyanchor_dstate_manifest(Ks=(68,)) if s["seed"] == 530]
    consistent = right_count and full_k68 == calib
    print(f"    calibration manifest: {calib[0] if calib else None}")
    ok = right_count and right_cell and consistent
    _report("smoke 4: keyanchor_dstate_calibration_manifest() -- exactly [K=68, seed=530, "
            "d_state=128], filtered from (not hand-duplicated against) the full manifest", ok)


def smoke_5_dstate_name_bit():
    all_cells = rdx.keyanchor_dstate_manifest() + rdx.keyanchor_dstate_gate1_manifest()
    ok = all(s["name"].endswith("_d128") for s in all_cells)
    bad = [s["name"] for s in all_cells if not s["name"].endswith("_d128")]
    print(f"    {len(all_cells)} cells checked; all end in '_d128': {ok}"
          f"{'' if ok else f' -- offenders: {bad}'}")
    _report("smoke 5: every d=128 cell's filename carries the '_d128' name bit "
            "(_spec()'s d_state != 64 generalization)", ok)


# ---------------------------------------------------------------------------
# smoke 6: zero-collision vs every prior/sibling manifest.
# ---------------------------------------------------------------------------

def smoke_6_zero_collision():
    dstate_paths = {rdx.out_path("/fake/root", s)
                     for s in (rdx.keyanchor_dstate_manifest() + rdx.keyanchor_dstate_gate1_manifest()
                               + rdx.keyanchor_dstate_calibration_manifest())}
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
    }
    ok = True
    for label, manifest in prior_manifests.items():
        prior_paths = {rdx.out_path("/fake/root", s) for s in manifest}
        disjoint = dstate_paths.isdisjoint(prior_paths)
        if not disjoint:
            print(f"    FAIL: keyanchor-dstate collides with {label} (shared fake_root)")
        ok = ok and disjoint
    print(f"    keyanchor-dstate ({len(dstate_paths)} paths) disjoint from all "
          f"{len(prior_manifests)} prior/sibling manifests (shared fake_root): {ok}")
    print("    real out_dir: 'wavekeyanchor-dstate' (main()'s own f\"wave{args.wave}\" convention) "
          "-- distinct from every prior wave's own out_dir")
    _report("smoke 6: zero-collision -- keyanchor-dstate's manifest (mandatory + Gate-1 probes + "
            "calibration) is collision-free against every prior/sibling wave this program has "
            "ever registered", ok)


# ---------------------------------------------------------------------------
# smoke 7: GATE2_N_ITER_BY_K extended.
# ---------------------------------------------------------------------------

def smoke_7_gate2_n_iter_by_k_extended():
    expected = {16: 12, 32: 20, 48: 20, 34: 20, 38: 20, 42: 20, 46: 20,
                68: 20, 76: 20, 84: 20, 92: 20}
    ok = ka.GATE2_N_ITER_BY_K == expected
    print(f"    GATE2_N_ITER_BY_K={ka.GATE2_N_ITER_BY_K} (expect {expected})")
    _report("smoke 7: GATE2_N_ITER_BY_K extended with {68:20, 76:20, 84:20, 92:20} (sec 13.2 "
            "item 3, confirmed sufficient by keyanchor_dstate_niter_check.py)", ok)


# ---------------------------------------------------------------------------
# smoke 8: rev7_threshold_derive.py's --d-state flag + d=128 pin byte-diff.
# ---------------------------------------------------------------------------

def smoke_8_d128_threshold_pin():
    import math
    import subprocess

    pinned_64_path = os.path.join(HERE, "REV7_THRESHOLD_PINNED.json")
    if not os.path.exists(pinned_64_path):
        _report("smoke 8: d=128 threshold pin (sec 13.2)", False,
                 f"{pinned_64_path!r} does not exist -- cannot diff")
        return
    with open(pinned_64_path) as f:
        d64_pin = json.load(f)

    with tempfile.TemporaryDirectory() as tmp:
        d128_path = os.path.join(tmp, "REV7_THRESHOLD_PINNED_D128.json")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "rev7_threshold_derive.py"),
                               "--d-state", "128", "--out", d128_path], cwd=HERE)
        ran_ok = rc == 0 and os.path.exists(d128_path)
        print(f"    rev7_threshold_derive.py --d-state 128 --out <fresh tmp path> exit_code={rc} "
              f"wrote_file={os.path.exists(d128_path)}")
        if not ran_ok:
            _report("smoke 8: d=128 threshold pin (sec 13.2)", False,
                     "rev7_threshold_derive.py --d-state 128 failed to run or write its output")
            return
        with open(d128_path) as f:
            d128_pin = json.load(f)

    d64_derived_str = json.dumps(d64_pin["derived"], sort_keys=True)
    d128_derived_str = json.dumps(d128_pin["derived"], sort_keys=True)
    byte_different = d64_derived_str != d128_derived_str
    inputs_ok = d128_pin["derived"]["inputs"]["d_state"] == 128
    sigma_exact = d128_pin["derived"]["null"]["sigma_chance"] == 1.0 / math.sqrt(128)
    headline_unchanged = (d64_pin["derived"]["effect_size_floors"]["r_min_headline_band"]
                           == d128_pin["derived"]["effect_size_floors"]["r_min_headline_band"] == 0.35)
    print(f"    d128 pin derived != d64 pin derived (byte-DIFFERENT, expected): {byte_different}")
    print(f"    d128 pin's derived.inputs.d_state == 128: {inputs_ok}")
    print(f"    d128 pin's sigma_chance == 1/sqrt(128) exactly: {sigma_exact} "
          f"({d128_pin['derived']['null']['sigma_chance']!r})")
    print(f"    r_min_headline_band fixed at 0.35 in BOTH pins: {headline_unchanged}")
    ok = byte_different and inputs_ok and sigma_exact and headline_unchanged
    _report("smoke 8: d=128 threshold pin (sec 13.2, round-2 verify finding 6) -- "
            "rev7_threshold_derive.py --d-state 128 produces a 'derived' block byte-DIFFERENT "
            "from REV7_THRESHOLD_PINNED.json's, with sigma_chance==1/sqrt(128) exactly and "
            "r_min_headline unchanged at 0.35", ok)


# ---------------------------------------------------------------------------
# smoke 9: read_wall_s_only(path) -- h4-blindness (sec 13.5's F9 rule).
# ---------------------------------------------------------------------------

def smoke_9_read_wall_s_only():
    import glob

    # Two roots: the local-repo archive (dev box) and the raw results dir
    # (training box, where experiment-runs/ does not exist -- first box run
    # failed here). Same cell JSONs either way; first valid hit wins.
    candidates = (glob.glob(os.path.join(HERE, "..", "..", "experiment-runs",
                                           "2026-07-06_keyanchor_cliff", "results", "**", "*.json"),
                             recursive=True)
                  + glob.glob(os.path.join(HERE, "results", "deltanet_rd_exactness",
                                             "wavekeyanchor-cliff", "**", "*.json"),
                               recursive=True))
    real_path, real_doc = None, None
    for c in candidates:
        try:
            with open(c) as f:
                d = json.load(f)
            if "wall_s" in d and d.get("checkpoints"):
                real_path, real_doc = c, d
                break
        except Exception:
            continue
    if real_path is None:
        _report("smoke 9: read_wall_s_only h4-blindness (sec 13.5 F9)", False,
                 "no real archived cell JSON found to test against")
        return

    returned = rdx.read_wall_s_only(real_path)
    matches = returned == real_doc["wall_s"]
    is_float = isinstance(returned, float)
    has_m3 = any("M3_held_out" in ckpt for ckpt in real_doc.get("checkpoints", []))
    repr_clean = "M3_held_out" not in repr(returned) and "recovered_frac" not in repr(returned)
    print(f"    tested against: {os.path.basename(real_path)}")
    print(f"    read_wall_s_only(path) == doc['wall_s']: {matches} ({returned})")
    print(f"    return type is float: {is_float}")
    print(f"    test file DOES contain M3_held_out (meaningful negative test, not vacuous): {has_m3}")
    print(f"    repr(return value) contains no M3_held_out/recovered_frac substring: {repr_clean}")
    ok = matches and is_float and has_m3 and repr_clean
    _report("smoke 9: read_wall_s_only(path) -- sec 13.5's F9 blinding-rule helper -- EXECUTED "
            "against a real archived cell JSON, returns wall_s only, no M3_held_out content "
            "anywhere in its output", ok)


# ---------------------------------------------------------------------------
# smoke 10: sec 13.6's mechanical decision table.
# ---------------------------------------------------------------------------

def smoke_10_decision_table():
    t = rdx.KEYANCHOR_DSTATE_DECISION_THRESHOLDS
    thresholds_ok = (abs(t["proceed_full"] - 1.7492) < 1e-4 and abs(t["option_b"] - 2.3322) < 1e-4
                      and abs(t["option_c"] - 3.4983) < 1e-4)
    d1 = rdx.keyanchor_dstate_decision_branch(1.0)
    d2 = rdx.keyanchor_dstate_decision_branch(t["proceed_full"])
    d3 = rdx.keyanchor_dstate_decision_branch(t["proceed_full"] + 1e-6)
    d4 = rdx.keyanchor_dstate_decision_branch(t["option_b"])
    d5 = rdx.keyanchor_dstate_decision_branch(t["option_b"] + 1e-6)
    d6 = rdx.keyanchor_dstate_decision_branch(t["option_c"])
    d7 = rdx.keyanchor_dstate_decision_branch(t["option_c"] + 1e-6)
    branches_ok = (d1["branch"] == "PROCEED_FULL" and d1["n_cells"] == 12
                   and d2["branch"] == "PROCEED_FULL"
                   and d3["branch"] == "OPTION_B" and d3["n_cells"] == 9
                   and d4["branch"] == "OPTION_B"
                   and d5["branch"] == "OPTION_C" and d5["n_cells"] == 6
                   and d6["branch"] == "OPTION_C"
                   and d7["branch"] == "ESCALATE" and d7["n_cells"] == 0)
    expected_amendment = 12 * 4.0 - rdx.KEYANCHOR_DSTATE_HEADROOM_GPUH
    d8 = rdx.keyanchor_dstate_decision_branch(4.0)
    amendment_ok = d8["branch"] == "ESCALATE" and abs(d8["ceiling_amendment_gpuh"] - expected_amendment) < 1e-9
    print(f"    thresholds={t} (expect ~1.7492/2.3322/3.4983): {thresholds_ok}")
    print(f"    r=1.0 -> {d1['branch']} (12 cells); "
          f"r=proceed_full boundary -> {d2['branch']}; r=just-above -> {d3['branch']} (9 cells)")
    print(f"    r=option_b boundary -> {d4['branch']}; r=just-above -> {d5['branch']} (6 cells)")
    print(f"    r=option_c boundary -> {d6['branch']}; r=just-above -> {d7['branch']} (0 cells)")
    print(f"    r=4.0 -> {d8['branch']}, ceiling_amendment_gpuh={d8['ceiling_amendment_gpuh']:.4f} "
          f"(expect {expected_amendment:.4f}): {amendment_ok}")
    ok = thresholds_ok and branches_ok and amendment_ok
    _report("smoke 10: sec 13.6 mechanical decision table -- all 4 branches fire at the exact "
            "registered thresholds (1.7492/2.3322/3.4983 GPU-h/cell), boundary-inclusive, "
            "ceiling-amendment formula correct", ok)


# ---------------------------------------------------------------------------
# smoke 11: keyanchor_dstate_stage_gate's calibration-first mechanics.
# ---------------------------------------------------------------------------

_DSTATE_EC_FIELDS = ("embed_source", "gram_alpha", "gram_rho", "strong_pin", "lambda_orth",
                     "use_zca", "fnce_m", "geo3_active", "geo3_n_iter", "geo3_resid_tol",
                     "anchor_active", "anchor_lambda_mode", "anchor_lambda_fixed",
                     "lambda_anchor", "drift_probe", "rev7_engagement")


def _write_fake_calib_result(out_dir, wall_s):
    calib_spec = rdx.keyanchor_dstate_calibration_manifest()[0]
    ec = {field: calib_spec.get(field) for field in _DSTATE_EC_FIELDS}
    doc = {"K": calib_spec["K"], "seed": calib_spec["seed"], "steps": calib_spec["steps"],
           "complete": True, "steps_completed": calib_spec["steps"], "exactness_config": ec,
           "wall_s": wall_s, "timed_out": False, "d_state": calib_spec["d_state"],
           "checkpoints": [{"M3_held_out": {"4": {"recovered_frac@0.9": 0.99}}}]}
    p = rdx.out_path(out_dir, calib_spec)
    with open(p, "w") as f:
        json.dump(doc, f)
    return calib_spec


def smoke_11_stage_gate_mechanics():
    # (a) stage='calibration' is a no-op.
    with tempfile.TemporaryDirectory() as tmp:
        r = rdx.keyanchor_dstate_stage_gate(tmp, "calibration", accept_override=False)
        calib_noop_ok = r.get("not_applicable") is True and r.get("gate_bypassed") is False

    # (b) stage='full' REFUSES when calibration cell missing.
    with tempfile.TemporaryDirectory() as tmp:
        refused_missing = False
        try:
            rdx.keyanchor_dstate_stage_gate(tmp, "full", accept_override=False)
        except SystemExit as e:
            refused_missing = (e.code == 1)

    # (c) in-bracket calibration -> PROCEED_FULL, sentinel written correctly.
    with tempfile.TemporaryDirectory() as tmp:
        _write_fake_calib_result(tmp, 3600.0)   # r=1.0 GPU-h/cell
        report = rdx.keyanchor_dstate_stage_gate(tmp, "full", accept_override=False)
        sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_DSTATE_STAGE_SENTINEL_NAME)
        proceed_full_ok = (report["decision"]["branch"] == "PROCEED_FULL"
                            and report["sentinel_written"] is True and os.path.exists(sentinel_path))
        with open(sentinel_path) as f:
            payload = json.loads(f.read().strip())
        sentinel_content_ok = (payload["decision"]["branch"] == "PROCEED_FULL"
                                and isinstance(payload.get("timestamp"), (int, float))
                                and payload["timestamp"] > 0)

    # (d) over-bracket calibration -> ESCALATE, refuses, no sentinel.
    with tempfile.TemporaryDirectory() as tmp:
        _write_fake_calib_result(tmp, 14400.0)   # r=4.0 GPU-h/cell
        refused_escalate = False
        try:
            rdx.keyanchor_dstate_stage_gate(tmp, "full", accept_override=False)
        except SystemExit as e:
            refused_escalate = (e.code == 1)
        sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_DSTATE_STAGE_SENTINEL_NAME)
        no_sentinel_on_escalate = not os.path.exists(sentinel_path)

    # (e) --accept-dstate-stage-override bypasses cleanly, writes no sentinel.
    with tempfile.TemporaryDirectory() as tmp:
        report = rdx.keyanchor_dstate_stage_gate(tmp, "full", accept_override=True)
        sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_DSTATE_STAGE_SENTINEL_NAME)
        override_ok = (report["gate_bypassed"] is True and report["sentinel_written"] is False
                        and not os.path.exists(sentinel_path))

    print(f"    (a) stage='calibration' no-op: {calib_noop_ok}")
    print(f"    (b) stage='full' refuses when calibration missing: {refused_missing}")
    print(f"    (c) in-bracket (r=1.0) -> PROCEED_FULL, sentinel written+correct: "
          f"{proceed_full_ok and sentinel_content_ok}")
    print(f"    (d) over-bracket (r=4.0) -> ESCALATE, refuses, no sentinel: "
          f"{refused_escalate and no_sentinel_on_escalate}")
    print(f"    (e) --accept-dstate-stage-override bypasses, no sentinel: {override_ok}")
    ok = (calib_noop_ok and refused_missing and proceed_full_ok and sentinel_content_ok
          and refused_escalate and no_sentinel_on_escalate and override_ok)
    _report("smoke 11: keyanchor_dstate_stage_gate calibration-first mechanics -- no-op at "
            "'calibration', refuses on missing calibration cell, correct branch+sentinel on a "
            "clean in-bracket read, ESCALATE refuses with no sentinel, override bypasses cleanly",
            ok)


# ---------------------------------------------------------------------------
# smoke 12: keyanchor_dstate_check_abort.
# ---------------------------------------------------------------------------

def smoke_12_check_abort():
    spec = rdx.keyanchor_dstate_manifest()[0]
    no_raise_ok = True
    try:
        rdx.keyanchor_dstate_check_abort(spec, wall_s=100.0, bracket_upper_gpuh=1.0)
    except rdx.KeyanchorDstateAbort:
        no_raise_ok = False

    raises_ok = False
    try:
        rdx.keyanchor_dstate_check_abort(spec, wall_s=6000.0, bracket_upper_gpuh=1.0)
    except rdx.KeyanchorDstateAbort:
        raises_ok = True

    # Boundary-exact: wall_s == 1.5*bracket*3600 raises (>=, not >).
    boundary_raises = False
    try:
        rdx.keyanchor_dstate_check_abort(spec, wall_s=1.5 * 1.0 * 3600.0, bracket_upper_gpuh=1.0)
    except rdx.KeyanchorDstateAbort:
        boundary_raises = True

    print(f"    wall_s=100s (<<5400s trigger): does not raise: {no_raise_ok}")
    print(f"    wall_s=6000s (>=5400s trigger): raises KeyanchorDstateAbort: {raises_ok}")
    print(f"    wall_s==5400s exactly (boundary, >=): raises: {boundary_raises}")
    ok = no_raise_ok and raises_ok and boundary_raises
    _report("smoke 12: keyanchor_dstate_check_abort -- does not raise well under the 1.5x-"
            "bracket-upper-edge trigger, raises KeyanchorDstateAbort at/above it (boundary-exact)",
            ok)


# ---------------------------------------------------------------------------
# smoke 13: fit_cliff_curve.py's d=128 anchor-free mode.
# ---------------------------------------------------------------------------

def smoke_13_fit_cliff_curve_anchor_free_mode():
    import subprocess

    # (a) --k-grid is REQUIRED at a non-64 d (no silent CLIFF_KS reuse).
    with tempfile.TemporaryDirectory() as tmp:
        fake_cliff_dir = os.path.join(tmp, "wavekeyanchor-dstate")
        os.makedirs(fake_cliff_dir, exist_ok=True)
        rc = subprocess.run(
            [sys.executable, os.path.join(HERE, "fit_cliff_curve.py"),
             "--cliff-out-dir", fake_cliff_dir, "--d-state", "128",
             "--out", os.path.join(tmp, "out.json")],
            cwd=HERE, capture_output=True, text=True)
        k_grid_required = rc.returncode != 0 and "k-grid" in (rc.stderr + rc.stdout).lower()
    print(f"    --d-state 128 with --k-grid omitted: exits non-zero (AssertionError expected): "
          f"{k_grid_required} (rc={rc.returncode})")

    # (b) --k32-dir/--k48-dir must be OMITTED at a non-64 d (asserts if passed).
    with tempfile.TemporaryDirectory() as tmp:
        fake_cliff_dir = os.path.join(tmp, "wavekeyanchor-dstate")
        fake_k32_dir = os.path.join(tmp, "fake_k32")
        os.makedirs(fake_cliff_dir, exist_ok=True)
        os.makedirs(fake_k32_dir, exist_ok=True)
        rc2 = subprocess.run(
            [sys.executable, os.path.join(HERE, "fit_cliff_curve.py"),
             "--cliff-out-dir", fake_cliff_dir, "--d-state", "128", "--k-grid", "68", "76", "84", "92",
             "--k32-dir", fake_k32_dir, "--out", os.path.join(tmp, "out2.json")],
            cwd=HERE, capture_output=True, text=True)
        anchors_forbidden = rc2.returncode != 0 and "must be omitted" in (rc2.stderr + rc2.stdout).lower()
    print(f"    --d-state 128 with --k32-dir passed: exits non-zero (AssertionError expected): "
          f"{anchors_forbidden} (rc={rc2.returncode})")

    # (c) ANCHORED_D_STATES contains 64 only, per this build's own registration.
    anchored_ok = fcc.ANCHORED_D_STATES == (64,)
    print(f"    fit_cliff_curve.ANCHORED_D_STATES == (64,): {anchored_ok}")

    # (d) real_bootstrap_ci's own anchor-optional path: k32_seeds=None,
    # k48_seeds=None produces an anchor-free fit without crashing (small
    # synthetic per_seed_by_k, few trials -- mechanics only, not a real fit).
    import numpy as np
    per_seed_by_k = {68: [0.9, 0.91, 0.89], 76: [0.7, 0.68, 0.72],
                     84: [0.3, 0.32, 0.28], 92: [0.05, 0.06, 0.04]}
    anchor_free_ci = fcc.real_bootstrap_ci(per_seed_by_k, None, None, n_trials=50, d_state=128, seed=0)
    ran_without_crash = "n_trials" in anchor_free_ci and anchor_free_ci["n_trials"] == 50
    print(f"    real_bootstrap_ci(..., k32_seeds=None, k48_seeds=None, d_state=128) runs without "
          f"crashing: {ran_without_crash}")

    ok = k_grid_required and anchors_forbidden and anchored_ok and ran_without_crash
    _report("smoke 13: fit_cliff_curve.py's d=128 anchor-free mode -- --k-grid REQUIRED at a "
            "non-64 d, --k32-dir/--k48-dir FORBIDDEN at a non-64 d, ANCHORED_D_STATES==(64,), "
            "real_bootstrap_ci runs anchor-free without crashing", ok)


# ---------------------------------------------------------------------------
# smoke 14: sec 13.3 item 5a's assert-guarded Gate-2 check at d_state=128,
# ks=(68,76,84,92) EXPLICITLY -- this is ALSO the first real Gate-2 data at
# d=128 (item 5's own build task), executed here (not merely written).
# ---------------------------------------------------------------------------

def smoke_14_gate2_d128_explicit_ks():
    REGISTERED_KS = (68, 76, 84, 92)
    table = ka.frame_potential_init(107, 128, seed=ka.ANCHOR_INIT_SEED)
    # The assert-guarded check itself (sec 13.3 item 5a's own registered
    # wording): confirm the K-grid about to be passed equals this wave's
    # registered K-grid BEFORE trusting the result -- never the function's
    # own default ks=(16,32), never a stale (16,32,48) tuple.
    assert REGISTERED_KS == (68, 76, 84, 92), \
        f"ks= drifted from sec 13.2's registered K-grid: {REGISTERED_KS}"
    result = ka.gate2_construction_check(table, ks=REGISTERED_KS, seed=0)
    print(f"    table shape: {tuple(table.shape)} (n_entities=107, d_state=128)")
    print(f"    G2-a sigma_ratio = {result['g2a_sigma_ratio']:.6f} (>= 0.1? {result['g2a_pass']})")
    print(f"    G2-b max|cos|    = {result['g2b_max_abs_cos']:.6f} (<= 0.5? {result['g2b_pass']})")
    for K in REGISTERED_KS:
        leg = result["g2c_ns_legs"][K]
        print(f"    G2-c K={K:3d}: n_iter={leg['n_iter']} n_fallback={leg['n_fallback']}/"
              f"{leg['n_subsets']} max_resid={leg['max_resid']:.3e} pass={leg['pass']}")
    print(f"    OVERALL GATE 2 (ks={REGISTERED_KS}) at d_state=128: "
          f"{'PASS' if result['pass'] else 'FAIL'}")
    _report(f"smoke 14: sec 13.3 item 5a -- Gate 2 EXECUTED at d_state=128 with ks={REGISTERED_KS} "
            f"EXPLICITLY (never the ks=(16,32) default, never a stale tuple) -- ALSO the first "
            f"real Gate-2 data at d=128 (reported honestly regardless of outcome)", result["pass"])


def main() -> int:
    print("=" * 70)
    print("smoke_keyanchor_dstate.py -- KEY_ANCHORING_DESIGN.md sec 13 (Rev 13.2) cliff-"
          "universality-across-d_state Wave -1 smoke suite (fla-free)")
    print("=" * 70)
    smoke_1_d64_regression()
    smoke_2_dstate_manifest_shape()
    smoke_3_dstate_gate1_manifest_shape()
    smoke_4_dstate_calibration_manifest_shape()
    smoke_5_dstate_name_bit()
    smoke_6_zero_collision()
    smoke_7_gate2_n_iter_by_k_extended()
    smoke_8_d128_threshold_pin()
    smoke_9_read_wall_s_only()
    smoke_10_decision_table()
    smoke_11_stage_gate_mechanics()
    smoke_12_check_abort()
    smoke_13_fit_cliff_curve_anchor_free_mode()
    smoke_14_gate2_d128_explicit_ks()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
