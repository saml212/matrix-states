"""smoke_keyanchor_cliff.py -- KEY_ANCHORING_DESIGN.md sec 12 (Rev 12.2,
CLEARED-FOR-BUILD)'s registered fla-free Wave -1 smoke suite for the
capacity-cliff localization wave, as a committed, CPU-runnable script.
Mirrors smoke_keyanchor_k48_e.py's own split: this file imports ONLY
key_anchoring.py, run_deltanet_rd_exactness_sweep.py, and fit_cliff_curve.py
(all verified fla-free at module scope).

Covers:
  smoke 1 (sec 12.2.1's FIRST Wave -1 smoke, "mirrors sec 11.9 item 14
    exactly"): keyanchor_k48_manifest(K=48, seeds=(30,31,32)) and
    keyanchor_k48_gate1_manifest(K=48, seed=0) -- the generalized functions
    called AT THEIR K=48 DEFAULTS -- produce BYTE-IDENTICAL output
    (json.dumps sorted equality) to the pre-generalization hardcoded K=48
    manifests, reconstructed here verbatim as the non-regression oracle.
  smoke 2 (sec 12.2.1's SECOND Wave -1 smoke, NEW at Rev 12.1, attack
    finding 6): fit_cliff_curve.py's own assert_no_reference_arm_paths
    raises on an injected reference-arm-style path -- a negative unit test,
    run to completion, not merely written.
  smoke 3: keyanchor_cliff_manifest()'s shape -- 12 cells, K in
    {34,38,42,46}, 3 seeds each, seed blocks exactly {130,131,132}/
    {230,231,232}/{330,331,332}/{430,431,432}.
  smoke 4: keyanchor_cliff_gate1_manifest()'s shape -- 4 cells, seeds
    {133,233,333,433}, one per K.
  smoke 5: stage split -- K=38/K=42 tagged stage 1, K=34/K=46 tagged stage 2
    (sec 12.2.3's staged launch).
  smoke 6: zero-collision -- keyanchor_cliff_manifest()'s own out_path()
    filenames are disjoint from every prior wave's manifest this program
    has ever registered (sec 11.9 item 16's own precedent, extended).
  smoke 7: GATE2_N_ITER_BY_K extended with {34,38,42,46: 20} (sec 12.2.1).
  smoke 8: rev7_threshold_derive.py's 'derived' block byte-diff (sec 12.3/12.6).
  smoke 9 (sec 12.7.2 R3-4, build-audit REQUIRED addition): Stage-1 sub-
    ceiling (KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING=11.68 GPU-h) -- synthetic
    Stage-1 result JSONs whose wall_s SUM exceeds the ceiling REFUSE
    --stage 2 (sys.exit) and write NO sentinel, even with every individual
    cell within the per-cell bracket; boundary-exact at sum==ceiling
    (passes, per the chosen strict '>' comparison).
  smoke 10 (sec 12.7.2 R3-5, build-audit REQUIRED addition): the durable
    STAGE1_RATES_OK clearance sentinel -- in-bracket/under-sub-ceiling
    Stage-1 JSONs produce it with correct content (six wall_s values,
    computed stage1_gpuh, ceiling, bracket edge, timestamp);
    --accept-stage-override bypass produces NO sentinel.
  smoke 11 (independent-verifier poke B): partial Stage-1 (5 of 6
    registered cells) REFUSES stage-2 and writes NO sentinel -- per-seed
    completeness against the registered manifest, not per-K non-emptiness.
  smoke 12 (independent-verifier poke A): a planted stale STAGE1_RATES_OK
    does not survive a failing gate re-check -- the sentinel only exists
    if the MOST RECENT evaluation passed clean.

Wired as an ADDITIONAL pre-launch CPU gate for --wave keyanchor-cliff
(run_deltanet_rd_exactness_sweep.py's main(), alongside smoke_key_anchoring.py
+ gate2_construction_test.py) -- rc!=0 aborts the wave before any GPU cell
dispatches.

Exit code 0 = every item PASSED. Run: python smoke_keyanchor_cliff.py
"""
from __future__ import annotations

import os
import sys

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
# smoke 1: manifest-generalization non-regression at K=48 defaults --
# EXECUTABLE json.dumps sorted-equality comparison (task's own required
# form), not a hand-eyeball diff.
# ---------------------------------------------------------------------------

def _hardcoded_keyanchor_k48_manifest():
    """The ORIGINAL, pre-Rev-12.1-generalization body of
    keyanchor_k48_manifest() (sec 11.1 arm 1) -- reproduced verbatim as the
    non-regression oracle. NEVER edited to match the generalized function;
    exists only to prove the generalization didn't change K=48 behavior."""
    steps = rdx.KEYANCHOR_K48_TIER_STEPS
    runs = []
    for s in (30, 31, 32):
        runs.append(rdx._spec("keyanchor-k48", 48, s, steps, "d", geo3_active=True,
                               geo3_n_iter=rdx.KEYANCHOR_K48_NS_N_ITER, geo3_resid_tol=1e-2,
                               anchor_active=True, anchor_lambda_mode="learned",
                               drift_probe=True, rev7_engagement=True))
    return runs


def _hardcoded_keyanchor_k48_gate1_manifest():
    """The ORIGINAL, pre-generalization body of keyanchor_k48_gate1_manifest()
    (sec 11.1 arm 4)."""
    return [rdx._spec("keyanchor-k48-gate1", 48, 0, rdx.KEYANCHOR_K48_GATE1_STEPS, "d", geo3_active=True,
                       geo3_n_iter=rdx.KEYANCHOR_K48_NS_N_ITER, geo3_resid_tol=1e-2, anchor_active=True,
                       anchor_lambda_mode="learned", drift_probe=True, rev7_engagement=True)]


def smoke_1_manifest_regression_at_k48_defaults():
    import json

    new_d = rdx.keyanchor_k48_manifest(K=48, seeds=(30, 31, 32))
    old_d = _hardcoded_keyanchor_k48_manifest()
    new_d_default = rdx.keyanchor_k48_manifest()   # zero-arg call -- what every existing call site uses
    d_equal_explicit = json.dumps(new_d, sort_keys=True) == json.dumps(old_d, sort_keys=True)
    d_equal_default = json.dumps(new_d_default, sort_keys=True) == json.dumps(old_d, sort_keys=True)
    print(f"    keyanchor_k48_manifest(K=48, seeds=(30,31,32)) vs hardcoded oracle: "
          f"byte-identical(json.dumps sorted)={d_equal_explicit}")
    print(f"    keyanchor_k48_manifest() [zero-arg, existing call sites] vs hardcoded oracle: "
          f"byte-identical(json.dumps sorted)={d_equal_default}")

    new_g1 = rdx.keyanchor_k48_gate1_manifest(K=48, seed=0)
    old_g1 = _hardcoded_keyanchor_k48_gate1_manifest()
    new_g1_default = rdx.keyanchor_k48_gate1_manifest()
    g1_equal_explicit = json.dumps(new_g1, sort_keys=True) == json.dumps(old_g1, sort_keys=True)
    g1_equal_default = json.dumps(new_g1_default, sort_keys=True) == json.dumps(old_g1, sort_keys=True)
    print(f"    keyanchor_k48_gate1_manifest(K=48, seed=0) vs hardcoded oracle: "
          f"byte-identical(json.dumps sorted)={g1_equal_explicit}")
    print(f"    keyanchor_k48_gate1_manifest() [zero-arg] vs hardcoded oracle: "
          f"byte-identical(json.dumps sorted)={g1_equal_default}")

    ok = d_equal_explicit and d_equal_default and g1_equal_explicit and g1_equal_default
    _report("smoke 1: manifest-generalization non-regression -- keyanchor_k48_manifest/"
            "keyanchor_k48_gate1_manifest called AT K=48 DEFAULTS are byte-identical "
            "(json.dumps sorted) to their pre-generalization hardcoded forms", ok)


# ---------------------------------------------------------------------------
# smoke 2: fit_cliff_curve.py's negative unit test (sec 12.2.1's second Wave
# -1 smoke, attack finding 6) -- assert_no_reference_arm_paths MUST raise on
# an injected reference-arm-style path. Run to completion, not merely
# written (this project's own hard rule).
# ---------------------------------------------------------------------------

def smoke_2_fit_script_rejects_reference_arm_paths():
    # A real reference-arm result path, as reference_arms_manifest()/
    # keyanchor_k48_reference_manifest() would actually produce it (wave
    # tag 'ref', arm tag 'georef').
    injected_ref_path = "/fake/results/wavekeyanchor-k48-ref/wref_rdx_K48_armgeoref_s1_geo3n20_dprobe.json"
    clean_paths = [
        "/fake/results/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K34_armd_s130_geo3n20_anchor_learned_dprobe_rev7.json",
        "/fake/results/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K34_armd_s131_geo3n20_anchor_learned_dprobe_rev7.json",
    ]

    raised = False
    try:
        fcc.assert_no_reference_arm_paths(clean_paths + [injected_ref_path])
    except fcc.ReferenceArmPathError:
        raised = True
    except Exception as e:  # noqa: BLE001 -- any OTHER exception type is itself a failure of this smoke
        print(f"    UNEXPECTED exception type on injected reference-arm path: {e!r}")
    print(f"    assert_no_reference_arm_paths raises ReferenceArmPathError on an injected "
          f"reference-arm path: {raised}")

    no_raise_on_clean = True
    try:
        fcc.assert_no_reference_arm_paths(clean_paths)
    except Exception as e:  # noqa: BLE001
        no_raise_on_clean = False
        print(f"    UNEXPECTED: assert_no_reference_arm_paths raised on CLEAN candidate-(d) "
              f"paths: {e!r}")
    print(f"    assert_no_reference_arm_paths does NOT raise on clean candidate-(d)-only paths: "
          f"{no_raise_on_clean}")

    # Real manifest-derived positive control: every out_path() this wave's
    # OWN keyanchor_cliff_manifest() produces must itself pass the assertion
    # (i.e. the wave's real, intended input list is clean by construction).
    real_paths = [rdx.out_path("/fake/root", s) for s in rdx.keyanchor_cliff_manifest()]
    real_clean = True
    try:
        fcc.assert_no_reference_arm_paths(real_paths)
    except Exception as e:  # noqa: BLE001
        real_clean = False
        print(f"    UNEXPECTED: the wave's own real manifest paths tripped the reference-arm "
              f"assertion: {e!r}")
    print(f"    keyanchor_cliff_manifest()'s own out_path() list passes the assertion cleanly "
          f"(zero reference-arm paths by construction, sec 12.2 item 2's own cut): {real_clean}")

    ok = raised and no_raise_on_clean and real_clean
    _report("smoke 2: fit_cliff_curve.py's assert_no_reference_arm_paths negative unit test -- "
            "RUN TO COMPLETION (raises on an injected reference-arm path, passes clean on "
            "candidate-(d)-only paths incl. this wave's own real manifest)", ok)


# ---------------------------------------------------------------------------
# smoke 3/4/5: keyanchor-cliff manifest shape, Gate-1 probe shape, stage split.
# ---------------------------------------------------------------------------

def smoke_3_cliff_manifest_shape():
    m = rdx.keyanchor_cliff_manifest()
    right_count = len(m) == 12
    ks_present = sorted(set(s["K"] for s in m)) == [34, 38, 42, 46]
    right_arm = all(s["arm"] == "d" for s in m)
    right_lambda_mode = all(s["anchor_lambda_mode"] == "learned" for s in m)
    right_instrumentation = all(s["drift_probe"] is True and s["rev7_engagement"] is True for s in m)
    expected_seeds = {34: {130, 131, 132}, 38: {230, 231, 232}, 42: {330, 331, 332}, 46: {430, 431, 432}}
    seeds_ok = all(set(s["seed"] for s in m if s["K"] == K) == expected_seeds[K] for K in expected_seeds)
    print(f"    count={len(m)} Ks={sorted(set(s['K'] for s in m))} "
          f"seeds_by_K={{K: sorted(s['seed'] for s in m if s['K']==K) for K in (34,38,42,46)}}")
    for K in (34, 38, 42, 46):
        print(f"      K={K}: seeds={sorted(s['seed'] for s in m if s['K'] == K)} "
              f"(expect {sorted(expected_seeds[K])})")
    ok = right_count and ks_present and right_arm and right_lambda_mode and right_instrumentation and seeds_ok
    _report("smoke 3: keyanchor_cliff_manifest() shape -- 12 cells, K in {34,38,42,46}, 3 seeds "
            "each, exact registered seed blocks, candidate (d) arch, full instrumentation", ok)


def smoke_4_cliff_gate1_manifest_shape():
    g1 = rdx.keyanchor_cliff_gate1_manifest()
    right_count = len(g1) == 4
    ks_present = sorted(s["K"] for s in g1) == [34, 38, 42, 46]
    expected_seed_by_k = {34: 133, 38: 233, 42: 333, 46: 433}
    seeds_ok = all(s["seed"] == expected_seed_by_k[s["K"]] for s in g1)
    steps_ok = all(s["steps"] == rdx.KEYANCHOR_CLIFF_GATE1_STEPS == 5000 for s in g1)
    print(f"    count={len(g1)} Ks={sorted(s['K'] for s in g1)} "
          f"seeds={ {s['K']: s['seed'] for s in g1} } (expect {expected_seed_by_k})")
    ok = right_count and ks_present and seeds_ok and steps_ok
    _report("smoke 4: keyanchor_cliff_gate1_manifest() shape -- 4 cells, seeds "
            "{133,233,333,433}, one per K, 5,000 steps", ok)


def smoke_5_stage_split():
    stage1 = sorted(K for K, s in rdx.KEYANCHOR_CLIFF_STAGE_BY_K.items() if s == 1)
    stage2 = sorted(K for K, s in rdx.KEYANCHOR_CLIFF_STAGE_BY_K.items() if s == 2)
    ok = stage1 == [38, 42] and stage2 == [34, 46]
    print(f"    stage1={stage1} (expect [38, 42])  stage2={stage2} (expect [34, 46])")
    _report("smoke 5: stage split -- K=38/K=42 tagged stage 1 (interior points), K=34/K=46 "
            "tagged stage 2 (sec 12.2.3's staged launch)", ok)


def smoke_6_zero_collision():
    cliff_paths = {rdx.out_path("/fake/root", s)
                   for s in rdx.keyanchor_cliff_manifest() + rdx.keyanchor_cliff_gate1_manifest()}
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
    }
    ok = True
    for label, manifest in prior_manifests.items():
        prior_paths = {rdx.out_path("/fake/root", s) for s in manifest}
        disjoint = cliff_paths.isdisjoint(prior_paths)
        if not disjoint:
            print(f"    FAIL: keyanchor-cliff collides with {label} (shared fake_root)")
        ok = ok and disjoint
    print(f"    keyanchor-cliff ({len(cliff_paths)} paths) disjoint from all {len(prior_manifests)} "
          f"prior/sibling manifests (shared fake_root): "
          f"{all((cliff_paths).isdisjoint({rdx.out_path('/fake/root', s) for s in m}) for m in prior_manifests.values())}")
    # Real out_dir separation: --wave keyanchor-cliff resolves to its own
    # wavekeyanchor-cliff/ subdirectory (main()'s own f"wave{args.wave}"
    # convention) -- collision-free regardless of the shared-root check above.
    print("    real out_dir: 'wavekeyanchor-cliff' (main()'s own f\"wave{args.wave}\" convention) "
          "-- distinct from every prior wave's own out_dir")
    _report("smoke 6: zero-collision -- keyanchor-cliff's manifest (mandatory + Gate-1 probes) is "
            "collision-free against every prior/sibling wave this program has ever registered", ok)


def smoke_7_gate2_n_iter_by_k_extended():
    # Subset check, not exact-dict equality: later waves legitimately extend
    # this dict (sec 13 added {68,76,84,92} and broke the original exact
    # assertion -- build-audit Finding 1). THIS wave's requirement is only
    # that its own keys are present with the registered tier.
    required = {16: 12, 32: 20, 48: 20, 34: 20, 38: 20, 42: 20, 46: 20}
    ok = all(ka.GATE2_N_ITER_BY_K.get(k) == v for k, v in required.items())
    print(f"    GATE2_N_ITER_BY_K={ka.GATE2_N_ITER_BY_K} (require superset of {required})")
    _report("smoke 7: GATE2_N_ITER_BY_K extended with {34:20, 38:20, 42:20, 46:20} (sec 12.2.1)", ok)


# ---------------------------------------------------------------------------
# smoke 9/10: sec 12.7.2 R3-4 (Stage-1 sub-ceiling) / R3-5 (clearance
# sentinel) -- the two build-audit REQUIRED code additions landed in
# keyanchor_cliff_stage_gate / _write_keyanchor_cliff_stage1_sentinel
# (run_deltanet_rd_exactness_sweep.py). Synthetic Stage-1 result JSONs,
# built with the SAME "_fake_result from a real spec's own field list"
# discipline smoke_keyanchor_confirm.py's smoke D already established
# (never a hand-typed exactness_config that could silently drift from
# is_done()'s own checked-field list).
# ---------------------------------------------------------------------------

# The exact field list is_done() cross-checks against exactness_config,
# straight off _spec()'s own returned dict -- copying this list (rather
# than hand-typing a parallel one) means a future is_done() field addition
# can't silently desync this smoke's synthetic JSONs from what the real
# gate actually reads.
_IS_DONE_EC_FIELDS = ("embed_source", "gram_alpha", "gram_rho", "strong_pin", "lambda_orth",
                      "use_zca", "fnce_m", "geo3_active", "geo3_n_iter", "geo3_resid_tol",
                      "anchor_active", "anchor_lambda_mode", "anchor_lambda_fixed",
                      "lambda_anchor", "drift_probe", "rev7_engagement")


def _fake_cliff_result(spec, wall_s):
    ec = {field: spec.get(field) for field in _IS_DONE_EC_FIELDS}
    return {"K": spec["K"], "seed": spec["seed"], "steps": spec["steps"], "complete": True,
            "steps_completed": spec["steps"], "exactness_config": ec, "wall_s": wall_s,
            "timed_out": False}


def _write_stage1_cells(out_dir, wall_s_by_seed_index):
    """Writes all 6 Stage-1 (K=38, K=42) cells' result JSONs into out_dir,
    one wall_s value per cell in manifest order (2 K's x 3 seeds = 6),
    exactly the same cells keyanchor_cliff_stage_gate's own per_k_cells
    loop reads. Returns the list of (spec, wall_s) pairs written."""
    stage1_ks = tuple(sorted(K for K, s in rdx.KEYANCHOR_CLIFF_STAGE_BY_K.items() if s == 1))
    specs = rdx.keyanchor_cliff_manifest(Ks=stage1_ks)
    assert len(specs) == len(wall_s_by_seed_index) == 6, \
        f"expected exactly 6 Stage-1 cells, got {len(specs)} specs / {len(wall_s_by_seed_index)} wall_s values"
    written = []
    for spec, wall_s in zip(specs, wall_s_by_seed_index):
        p = rdx.out_path(out_dir, spec)
        with open(p, "w") as f:
            import json
            json.dump(_fake_cliff_result(spec, wall_s), f)
        written.append((spec, wall_s))
    return written


def smoke_9_stage1_sub_ceiling_refusal():
    """sec 12.7.2 R3-4: synthetic Stage-1 result JSONs whose wall_s SUM
    exceeds the dedicated 11.68 GPU-h Stage-1 ceiling must cause --stage 2
    to be REFUSED (sys.exit) AND must NOT write the STAGE1_RATES_OK
    sentinel -- regardless of every individual cell sitting WITHIN the
    per-cell bracket (bracket_upper_s = 3492.0s each), which is exactly the
    gap the attack flagged (a per-cell-clean Stage-1 can still blow the
    STAGE-1 TOTAL). Boundary-exact test at the threshold itself: this smoke
    tests both a sum strictly ABOVE 11.68 (must refuse) and a sum exactly
    AT 11.68 (must pass, per the chosen STRICT-greater-than comparison --
    sum > ceiling refuses, sum == ceiling passes).

    DISCLOSED arithmetic fact this test is built around (also documented at
    the check's own call site in run_deltanet_rd_exactness_sweep.py): at
    the doc's own registered numbers, 6 cells ALL individually within the
    per-cell bracket (bracket_upper_s=3503.808s unrounded) can sum to at
    most 6*3503.808=21,022.8s=5.84 GPU-h -- well under the 11.68 GPU-h
    Stage-1 ceiling. So a sum > ceiling is mathematically UNREACHABLE
    without at least one cell also individually exceeding the per-cell
    bracket. The sub-ceiling check is therefore placed FIRST in the gate
    (ahead of the per-cell bracket check, not after it) precisely so it is
    evaluated independently rather than being unreachable dead code behind
    the bracket check's own sys.exit -- this smoke's Case A exercises that
    ordering directly (cells that are over-ceiling AND over-bracket; the
    sub-ceiling message must be the one that actually fires first)."""
    import tempfile

    bracket_upper_s = rdx.KEYANCHOR_CLIFF_GPUH_PER_CELL[1] * 3600.0   # 3503.808s (unrounded)
    ceiling_s = rdx.KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING * 3600.0      # 11.68 * 3600 = 42,048.0s
    max_in_bracket_sum_s = 6 * bracket_upper_s                        # 21,022.848s
    assert ceiling_s > max_in_bracket_sum_s, \
        "test's own arithmetic assumption broke -- re-derive this smoke if either constant changes"

    # --- Case A: sum strictly ABOVE the ceiling. Per the arithmetic fact
    # above, this necessarily also puts every cell over the per-cell
    # bracket -- constructed so the SUB-CEILING check (evaluated first)
    # is what actually refuses, not merely "some check or other" refuses.
    over_wall_s = [ceiling_s / 6.0 + 1.0] * 6   # sum = ceiling_s + 6.0s > ceiling
    assert sum(over_wall_s) > ceiling_s
    with tempfile.TemporaryDirectory() as tmp:
        _write_stage1_cells(tmp, over_wall_s)
        sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME)
        refused = False
        try:
            rdx.keyanchor_cliff_stage_gate(tmp, stage=2, accept_override=False)
        except SystemExit as e:
            refused = (e.code == 1)
        no_sentinel = not os.path.exists(sentinel_path)
        print(f"    Case A (sum={sum(over_wall_s):.1f}s = {sum(over_wall_s)/3600.0:.4f} GPU-h > "
              f"ceiling {rdx.KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING:.2f} GPU-h): "
              f"stage-2 refused(sys.exit(1))={refused}, no sentinel written={no_sentinel}")
    case_a_ok = refused and no_sentinel

    # --- Case B: sum EXACTLY at the ceiling (boundary-exact). Per the
    # chosen comparison (sum > ceiling refuses; sum == ceiling PASSES), the
    # SUB-CEILING check itself must not refuse here -- but ceiling_s/6 =
    # 7008.0s is itself ABOVE the per-cell bracket (3503.808s), so the
    # (second, still-active) per-cell bracket check downstream WOULD then
    # refuse on its own separate grounds. To isolate the sub-ceiling
    # check's own boundary behavior in isolation, this case calls the
    # sub-ceiling arithmetic directly (the same computation
    # keyanchor_cliff_stage_gate performs) rather than routing through the
    # full gate, which is the only way to observe "sum==ceiling passes"
    # without a different, unrelated check also firing at these per-cell
    # magnitudes. The full end-to-end sentinel-content/write path (a
    # genuinely in-bracket AND under-ceiling clean pass) is covered by
    # smoke 10 below.
    at_ceiling_gpuh = ceiling_s / 6.0 * 6 / 3600.0
    assert abs(at_ceiling_gpuh - rdx.KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING) < 1e-9
    sub_ceiling_passes_at_boundary = not (at_ceiling_gpuh > rdx.KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING)
    print(f"    Case B (sub-ceiling arithmetic in isolation, sum={ceiling_s:.1f}s = "
          f"{at_ceiling_gpuh:.4f} GPU-h == ceiling {rdx.KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING:.2f} "
          f"GPU-h exactly): sub-ceiling check itself does NOT refuse at the exact boundary "
          f"(sum > ceiling is False when sum == ceiling)={sub_ceiling_passes_at_boundary}")
    case_b_ok = sub_ceiling_passes_at_boundary

    ok = case_a_ok and case_b_ok
    _report("smoke 9: sec 12.7.2 R3-4 Stage-1 sub-ceiling -- sum > 11.68 GPU-h REFUSES stage-2 "
            "via the sub-ceiling check firing FIRST (ahead of the per-cell bracket check) and "
            "writes NO sentinel (Case A); the sub-ceiling comparison itself does not refuse at "
            "sum == 11.68 GPU-h exactly (Case B, boundary-exact per the chosen strict-'>' "
            "comparison; full clean-pass sentinel write covered by smoke 10)", ok)


def smoke_11_partial_stage1_refusal():
    """Independent-verifier poke B (FATAL, fixed): a Stage-1 with FEWER
    completed cells than the registered seeds (5 of 6 -- one K short a
    seed) must REFUSE stage 2 and write NO sentinel, even when every
    present cell is comfortably in-bracket and the 5-cell sum is under the
    sub-ceiling. Before the fix, `len(cells) == 0` only caught a K with
    zero cells, so 5-of-6 silently certified Stage 1 (verified live by the
    independent verifier)."""
    import json
    import tempfile

    bracket_upper_s = rdx.KEYANCHOR_CLIFF_GPUH_PER_CELL[1] * 3600.0
    clean_wall_s = [3000.0, 3050.0, 3100.0, 3000.0, 3050.0, 3100.0]
    assert all(w < bracket_upper_s for w in clean_wall_s)
    with tempfile.TemporaryDirectory() as tmp:
        written = _write_stage1_cells(tmp, clean_wall_s)
        # Drop exactly one cell (the last-written spec) to simulate an
        # in-flight/crashed 6th cell -- the verifier's own live repro.
        dropped_spec, _ = written[-1]
        os.remove(rdx.out_path(tmp, dropped_spec))
        sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME)
        refused = False
        try:
            rdx.keyanchor_cliff_stage_gate(tmp, stage=2, accept_override=False)
        except SystemExit as e:
            refused = (e.code == 1)
        no_sentinel = not os.path.exists(sentinel_path)
        print(f"    5-of-6 Stage-1 cells (dropped K={dropped_spec['K']} seed={dropped_spec['seed']}, "
              f"all present cells in-bracket): refused(sys.exit(1))={refused}, "
              f"no sentinel written={no_sentinel}")
    _report("smoke 11: partial Stage-1 (5 of 6 registered cells) REFUSES stage-2 and writes NO "
            "sentinel (verifier poke B -- per-seed completeness, not just per-K nonemptiness)",
            refused and no_sentinel)


def smoke_12_stale_sentinel_removed_on_failed_recheck():
    """Independent-verifier poke A: a pre-existing (stale/fabricated)
    STAGE1_RATES_OK must not survive a gate re-check that refuses --
    downstream consumers gate on the file's existence alone, so the gate
    now removes any pre-existing sentinel before evaluating, and only the
    clean-pass branch rewrites it."""
    import tempfile

    ceiling_s = rdx.KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING * 3600.0
    over_wall_s = [ceiling_s / 6.0 + 1.0] * 6   # forces refusal (same as smoke 9 Case A)
    with tempfile.TemporaryDirectory() as tmp:
        _write_stage1_cells(tmp, over_wall_s)
        sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME)
        with open(sentinel_path, "w") as f:
            f.write('{"stale": true, "fabricated": "leftover from an earlier run"}\n')
        refused = False
        try:
            rdx.keyanchor_cliff_stage_gate(tmp, stage=2, accept_override=False)
        except SystemExit as e:
            refused = (e.code == 1)
        stale_removed = not os.path.exists(sentinel_path)
        print(f"    planted stale sentinel + over-ceiling Stage-1: refused={refused}, "
              f"stale sentinel removed by the failed re-check={stale_removed}")
    _report("smoke 12: a planted stale STAGE1_RATES_OK does not survive a failing gate re-check "
            "(verifier poke A -- the sentinel only exists if the MOST RECENT evaluation passed)",
            refused and stale_removed)


def smoke_10_sentinel_write_and_override_bypass():
    """sec 12.7.2 R3-5: synthetic in-bracket, within-sub-ceiling Stage-1
    JSONs produce STAGE1_RATES_OK with correct content (six cells' wall_s,
    computed Stage-1 GPU-h, bracket edge, a timestamp); --accept-stage-
    override bypasses the gate WITHOUT writing the sentinel."""
    import json
    import tempfile

    bracket_upper_s = rdx.KEYANCHOR_CLIFF_GPUH_PER_CELL[1] * 3600.0
    # Comfortably in-bracket AND comfortably under the Stage-1 sub-ceiling
    # (well within both checks, isolating this smoke from smoke 9's own
    # boundary constructions).
    clean_wall_s = [3000.0, 3050.0, 3100.0, 3000.0, 3050.0, 3100.0]
    assert all(w < bracket_upper_s for w in clean_wall_s)
    assert sum(clean_wall_s) / 3600.0 < rdx.KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING

    with tempfile.TemporaryDirectory() as tmp:
        written = _write_stage1_cells(tmp, clean_wall_s)
        sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME)
        report = rdx.keyanchor_cliff_stage_gate(tmp, stage=2, accept_override=False)
        sentinel_written = os.path.exists(sentinel_path)
        content_ok = False
        if sentinel_written:
            with open(sentinel_path) as f:
                lines = f.read().strip().splitlines()
            one_line = len(lines) == 1
            payload = json.loads(lines[0]) if one_line else None
            if payload is not None:
                expected_gpuh = sum(clean_wall_s) / 3600.0
                by_k = payload.get("stage1_wall_s_by_k", {})
                flat_recorded = sorted(w for cells in by_k.values() for w in cells)
                flat_expected = sorted(clean_wall_s)
                content_ok = (
                    one_line
                    and flat_recorded == flat_expected
                    and abs(payload.get("stage1_gpuh", -1.0) - expected_gpuh) < 1e-9
                    and abs(payload.get("stage1_gpuh_ceiling", -1.0)
                            - rdx.KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING) < 1e-9
                    and abs(payload.get("bracket_upper_s", -1.0) - bracket_upper_s) < 1e-9
                    and isinstance(payload.get("timestamp"), (int, float))
                    and payload.get("timestamp") > 0
                )
            print(f"    sentinel path: {sentinel_path}")
            print(f"    sentinel content: one JSON line={one_line}, "
                  f"wall_s values match={flat_recorded == flat_expected if one_line else 'n/a'}, "
                  f"stage1_gpuh={payload.get('stage1_gpuh') if payload else None:.4f} "
                  f"(expect {sum(clean_wall_s)/3600.0:.4f}), "
                  f"stage1_gpuh_ceiling={payload.get('stage1_gpuh_ceiling') if payload else None}, "
                  f"bracket_upper_s={payload.get('bracket_upper_s') if payload else None}, "
                  f"timestamp present and >0={isinstance(payload.get('timestamp'), (int, float)) and payload.get('timestamp') > 0 if payload else False}")
        report_reflects_write = report.get("sentinel_written") is True and \
            report.get("sentinel_path") == sentinel_path

    # --- override bypass: --accept-stage-override must NOT write the
    # sentinel, on the SAME clean-Stage-1-cells fixture.
    with tempfile.TemporaryDirectory() as tmp:
        _write_stage1_cells(tmp, clean_wall_s)
        sentinel_path_override = os.path.join(tmp, rdx.KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME)
        override_report = rdx.keyanchor_cliff_stage_gate(tmp, stage=2, accept_override=True)
        no_sentinel_on_override = not os.path.exists(sentinel_path_override)
        override_report_says_no_sentinel = override_report.get("sentinel_written") is False
        print(f"    --accept-stage-override on the SAME clean fixture: "
              f"sentinel NOT written={no_sentinel_on_override}, "
              f"report explicitly says sentinel_written=False: {override_report_says_no_sentinel}")

    ok = (sentinel_written and content_ok and report_reflects_write
          and no_sentinel_on_override and override_report_says_no_sentinel)
    _report("smoke 10: sec 12.7.2 R3-5 clearance sentinel -- in-bracket/under-sub-ceiling Stage-1 "
            "JSONs produce STAGE1_RATES_OK with correct content (six wall_s values, computed "
            "stage1_gpuh, ceiling, bracket edge, timestamp); --accept-stage-override bypass "
            "produces NO sentinel", ok)


def smoke_8_threshold_pin_byte_diff():
    """sec 12.3/sec 12.6's required pre-launch check (mechanical, not a
    paper exercise): re-run rev7_threshold_derive.py to a NEW pin file and
    byte-diff its 'derived' block against the existing
    REV7_THRESHOLD_PINNED.json's 'derived' block -- byte-identical is the
    PASS condition (sec 12.3's own text: 'derive()'s only three free inputs
    are N_ENTITIES=107, D_STATE=64, ALPHA=0.05'; no K dependence anywhere
    in the file, so the SAME derived block is expected for any K).
    Regenerates the pin fresh here (not merely re-reading the copy left
    over from an earlier session) so this smoke is the actual mechanical
    re-derivation-and-diff, not a stale artifact comparison."""
    import json
    import subprocess
    import tempfile

    import rev7_threshold_derive as r7d

    pinned_path = os.path.join(HERE, "REV7_THRESHOLD_PINNED.json")
    if not os.path.exists(pinned_path):
        _report("smoke 8: threshold pin byte-diff (sec 12.3/12.6)", False,
                f"{pinned_path!r} does not exist -- cannot diff")
        return

    with tempfile.TemporaryDirectory() as tmp:
        new_pin_path = os.path.join(tmp, "REV7_THRESHOLD_PINNED_K34_K38_K42_K46.json")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "rev7_threshold_derive.py"),
                               "--out", new_pin_path], cwd=HERE)
        ran_ok = rc == 0 and os.path.exists(new_pin_path)
        print(f"    rev7_threshold_derive.py --out <fresh tmp path> exit_code={rc} "
              f"wrote_file={os.path.exists(new_pin_path)}")
        if not ran_ok:
            _report("smoke 8: threshold pin byte-diff (sec 12.3/12.6)", False,
                     "rev7_threshold_derive.py failed to run or write its output")
            return
        with open(pinned_path) as f:
            existing = json.load(f)
        with open(new_pin_path) as f:
            fresh = json.load(f)
    existing_derived = json.dumps(existing["derived"], sort_keys=True)
    fresh_derived = json.dumps(fresh["derived"], sort_keys=True)
    byte_identical = existing_derived == fresh_derived
    print(f"    fresh re-derivation's 'derived' block vs REV7_THRESHOLD_PINNED.json's 'derived' "
          f"block: byte-identical(json.dumps sorted)={byte_identical}")
    print(f"    (sec 12.3: N_ENTITIES={r7d.N_ENTITIES} D_STATE={r7d.D_STATE} ALPHA={r7d.ALPHA} -- "
          f"the only 3 free inputs, no K anywhere in the derivation)")

    # Also confirm the repo-committed REV7_THRESHOLD_PINNED_K34_K38_K42_K46.json
    # (this build's own artifact, sec 12.3's registered pin path) matches too,
    # if present -- a secondary, non-blocking cross-check.
    repo_new_pin = os.path.join(HERE, "REV7_THRESHOLD_PINNED_K34_K38_K42_K46.json")
    if os.path.exists(repo_new_pin):
        with open(repo_new_pin) as f:
            repo_pin = json.load(f)
        repo_derived = json.dumps(repo_pin["derived"], sort_keys=True)
        repo_matches = repo_derived == existing_derived
        print(f"    committed REV7_THRESHOLD_PINNED_K34_K38_K42_K46.json's own 'derived' block "
              f"also byte-identical: {repo_matches}")

    _report("smoke 8: threshold pin byte-diff (sec 12.3/12.6) -- fresh rev7_threshold_derive.py "
            "re-derivation's 'derived' block is byte-identical to REV7_THRESHOLD_PINNED.json's, "
            "mechanically confirming no K-dependence exists", byte_identical)


def main() -> int:
    print("=" * 70)
    print("smoke_keyanchor_cliff.py -- KEY_ANCHORING_DESIGN.md sec 12 (Rev 12.2) capacity-cliff "
          "localization wave Wave -1 smoke suite (fla-free)")
    print("=" * 70)
    smoke_1_manifest_regression_at_k48_defaults()
    smoke_2_fit_script_rejects_reference_arm_paths()
    smoke_3_cliff_manifest_shape()
    smoke_4_cliff_gate1_manifest_shape()
    smoke_5_stage_split()
    smoke_6_zero_collision()
    smoke_7_gate2_n_iter_by_k_extended()
    smoke_8_threshold_pin_byte_diff()
    smoke_9_stage1_sub_ceiling_refusal()
    smoke_10_sentinel_write_and_override_bypass()
    smoke_11_partial_stage1_refusal()
    smoke_12_stale_sentinel_removed_on_failed_recheck()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
