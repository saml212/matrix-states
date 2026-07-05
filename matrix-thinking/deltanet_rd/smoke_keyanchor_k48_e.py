"""smoke_keyanchor_k48_e.py -- KEY_ANCHORING_DESIGN.md sec 11.9 (K=48
capacity-curve extension) and sec 10.13 (candidate (e), frozen-random-table
ablation)'s registered fla-free Wave -1 smokes, as a committed, CPU-runnable
script (2026-07 K48+e build). Mirrors smoke_keyanchor_mech.py's own split:
this file imports ONLY key_anchoring.py and run_deltanet_rd_exactness_sweep.py
(both verified fla-free at module scope) -- the model-construction/forward/
backward smokes for candidate (e) (frozen-table-receives-no-grad, fixed-
lambda-never-moves) live in smoke_key_anchoring.py instead (smoke_15/16/17),
since those require model_rd.py (fla-required), following that file's own
established split.

Covers:
  sec 11.9 item 14: manifest-refactor non-regression -- the generalized
    reference_arms_manifest()/keyanchor_wave1_manifest()/
    keyanchor_confirm_manifest() (now K-parameterized) produce BYTE-
    IDENTICAL K=16/32 manifests to their original hardcoded forms.
  sec 11.9 item 15: Gate 2's K=48 leg wired into gate2_construction_test.py
    (ks extended to (16,32,48)) -- reproduces the 0/512-fallback,
    sigma_ratio~=1.0/max|cos|~=0.284 result.
  sec 11.9 item 16: zero-collision smoke's wave-list scope, corrected --
    'wavegeo3' (the archive housing the pre-existing K=48 bare-geo3 cells,
    seeds 0/1/2) is added to the enumerated collision-check list, since
    out_path()/is_done() key on spec['name'] within a caller-supplied
    out_dir (a hardcoded per-wave-directory enumeration, confirmed by
    reading out_path/is_done directly, sec 11.11 point 6) -- this smoke
    checks the keyanchor-k48 wave's own manifest against EVERY prior wave's
    filenames, including wavegeo3's K=48 rider.
  sec 10.13 candidate (e): manifest shape (3 cells, K=32, seeds
    {60,61,62}), frozen/init-mode fields present and correct, zero
    collision against every other registered manifest (incl. the K48 wave
    built in this SAME build).
  Seed non-collision: every new seed block this build introduces (K48
    candidate (d): {30,31,32}; K48 candidate (d'): {40,41,42}; K48 Gate-1:
    seed 0 (K=48, distinct wave); K48 fixed-lambda1: seed 50; candidate
    (e): {60,61,62}) is checked against every PRIOR seed block this
    program has ever used, per sec 11.1's own "no seed integer's
    provenance is ever ambiguous" discipline.

Wired as an ADDITIONAL pre-launch CPU gate for --wave keyanchor-k48-ref,
keyanchor-k48-gate1, keyanchor-k48, and keyanchor-e (run_deltanet_rd_
exactness_sweep.py's main(), alongside smoke_key_anchoring.py +
gate2_construction_test.py) -- rc!=0 aborts the wave before any GPU cell
dispatches.

Exit code 0 = every item PASSED. Run: python smoke_keyanchor_k48_e.py
"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import key_anchoring as ka   # noqa: E402 (fla-free, see module docstring)
import run_deltanet_rd_exactness_sweep as rdx   # noqa: E402 (fla-free, see module docstring)

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


# ---------------------------------------------------------------------------
# item 14: manifest-refactor non-regression. Every generalized manifest
# function's DEFAULT arguments must reproduce the exact pre-refactor
# hardcoded output, field-for-field (every _spec(...) dict entry), not just
# the same length.
# ---------------------------------------------------------------------------

def _hardcoded_reference_arms_manifest():
    """The ORIGINAL, pre-refactor body of reference_arms_manifest() (sec
    3.6), reproduced verbatim here as the non-regression oracle -- this
    function must NEVER be edited to match the refactor; it exists only to
    prove the refactor didn't change K=16/32 behavior."""
    steps = rdx.KEYANCHOR_TIER_STEPS
    runs = []
    for K in (16, 32):
        n_iter = 12 if K == 16 else 20
        for s in (1, 2, 3):
            runs.append(rdx._spec("ref", K, s, steps, "georef", geo3_active=True,
                                    geo3_n_iter=n_iter, geo3_resid_tol=1e-2, drift_probe=True))
    return runs


def _hardcoded_keyanchor_wave1_manifest():
    """The ORIGINAL, pre-refactor body of keyanchor_wave1_manifest() (sec 5)."""
    steps = rdx.KEYANCHOR_TIER_STEPS
    runs = []
    for K in (16, 32):
        n_iter = 12 if K == 16 else 20
        for s in range(3):
            runs.append(rdx._spec("keyanchor", K, s, steps, "d", geo3_active=True, geo3_n_iter=n_iter,
                                    geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                                    drift_probe=True))
    for K in (16, 32):
        n_iter = 12 if K == 16 else 20
        for s in range(3):
            runs.append(rdx._spec("keyanchor", K, s, steps, "c", geo3_active=True, geo3_n_iter=n_iter,
                                    geo3_resid_tol=1e-2, lambda_anchor=rdx.KEYANCHOR_LAMBDA_ANCHOR,
                                    drift_probe=True))
    return runs


def _hardcoded_keyanchor_confirm_manifest(include_k16_spot_check=True):
    """The ORIGINAL, pre-refactor body of keyanchor_confirm_manifest() (sec 9.5)."""
    steps = rdx.KEYANCHOR_TIER_STEPS
    runs = []
    for s in range(3):
        runs.append(rdx._spec("keyanchor-confirm", 32, s, steps, "d", geo3_active=True, geo3_n_iter=20,
                                geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                                drift_probe=True))
    if include_k16_spot_check:
        runs.append(rdx._spec("keyanchor-confirm", 16, 0, steps, "d", geo3_active=True, geo3_n_iter=12,
                                geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                                drift_probe=True))
    return runs


def smoke_14_manifest_refactor_non_regression():
    ok = True

    ref_new, ref_old = rdx.reference_arms_manifest(), _hardcoded_reference_arms_manifest()
    ref_equal = ref_new == ref_old
    print(f"    reference_arms_manifest(): new=={len(ref_new)} old=={len(ref_old)} "
          f"byte-identical={ref_equal}")
    ok = ok and ref_equal

    kw1_new, kw1_old = rdx.keyanchor_wave1_manifest(), _hardcoded_keyanchor_wave1_manifest()
    kw1_equal = kw1_new == kw1_old
    print(f"    keyanchor_wave1_manifest(): new=={len(kw1_new)} old=={len(kw1_old)} "
          f"byte-identical={kw1_equal}")
    ok = ok and kw1_equal

    for spot_check in (True, False):
        kc_new = rdx.keyanchor_confirm_manifest(include_k16_spot_check=spot_check)
        kc_old = _hardcoded_keyanchor_confirm_manifest(include_k16_spot_check=spot_check)
        kc_equal = kc_new == kc_old
        print(f"    keyanchor_confirm_manifest(include_k16_spot_check={spot_check}): "
              f"new=={len(kc_new)} old=={len(kc_old)} byte-identical={kc_equal}")
        ok = ok and kc_equal

    _report("smoke 14: manifest-refactor non-regression -- generalized K16/32 defaults are "
            "byte-identical to the pre-refactor hardcoded manifests", ok)


# ---------------------------------------------------------------------------
# item 15: Gate 2's K=48 leg is wired into the committed test
# (gate2_construction_test.py's ks argument extended to (16,32,48)) and
# reports PASS on the registered init.
# ---------------------------------------------------------------------------

def smoke_15_gate2_k48_leg():
    healthy = ka.frame_potential_init(107, 64, seed=ka.ANCHOR_INIT_SEED)
    gate = ka.gate2_construction_check(healthy, ks=(16, 32, 48), seed=0)
    has_48_leg = 48 in gate["g2c_ns_legs"]
    leg48 = gate["g2c_ns_legs"].get(48, {})
    leg48_pass = bool(leg48.get("pass"))
    overall_pass = gate["pass"]
    print(f"    ks=(16,32,48) leg keys present: {sorted(gate['g2c_ns_legs'].keys())}")
    if has_48_leg:
        print(f"    K=48 leg: n_fallback={leg48['n_fallback']}/{leg48['n_subsets']} "
              f"max_resid={leg48['max_resid']:.2e} pass={leg48_pass} "
              f"(expect 0 fallbacks, max_resid ~1e-6)")
    print(f"    G2-a sigma_ratio={gate['g2a_sigma_ratio']:.6f} (expect ~0.999999642) "
          f"G2-b max|cos|={gate['g2b_max_abs_cos']:.6f} (expect ~0.284151)")
    print(f"    overall gate pass (all legs incl. K=48): {overall_pass}")
    ok = has_48_leg and leg48_pass and overall_pass
    _report("smoke 15: Gate 2's K=48 leg (ks=(16,32,48)) reports PASS on the registered init "
            "(0 fallbacks, sigma_ratio~=1.0, max|cos|~=0.284)", ok)

    # gate2_construction_test.py's own committed ks argument -- confirm it
    # has been widened to include 48 (sec 11.9 item 15's own "wired into
    # the committed test" requirement, not just the underlying function
    # supporting it).
    import gate2_construction_test as g2test
    import inspect
    src = inspect.getsource(g2test)
    committed_has_48 = "ks=(16, 32, 48)" in src or "ks=(16,32,48)" in src
    print(f"    gate2_construction_test.py's own committed ks argument includes 48: "
          f"{committed_has_48}")
    _report("smoke 15b: gate2_construction_test.py's committed ks argument extended to "
            "(16, 32, 48), not merely the underlying function", committed_has_48)


# ---------------------------------------------------------------------------
# item 16: zero-collision smoke's wave-list scope, corrected -- 'wavegeo3'
# (housing the pre-existing K=48 bare-geo3 rider, seeds 0/1/2) is added to
# the enumerated collision-check list. Confirms (a) out_path/is_done key on
# spec['name'] within a caller-supplied out_dir (a hardcoded per-wave-
# directory enumeration, per sec 11.11 point 6's own finding) and (b) the
# keyanchor-k48 wave's own manifest has zero out_path() collisions against
# EVERY prior wave this program has ever registered, including wavegeo3.
# ---------------------------------------------------------------------------

def smoke_16_zero_collision_incl_wavegeo3():
    k48_d = rdx.keyanchor_k48_manifest()
    k48_ref = rdx.keyanchor_k48_reference_manifest()
    k48_g1 = rdx.keyanchor_k48_gate1_manifest()
    k48_fix1 = rdx.keyanchor_k48_fixed_lambda1_manifest()
    k48_dprime = rdx.keyanchor_k48_dprime_manifest()
    ke = rdx.keyanchor_e_manifest()
    ke_fp = rdx.keyanchor_e_fp_manifest()

    # every PRIOR wave's manifest this program has ever registered,
    # INCLUDING wavegeo3 (sec 11.9 item 16's own correction -- omitted from
    # sec 10.9 item 13's original enumeration).
    prior_manifests = {
        "wavegeo3": rdx.geo3_wave1_manifest(include_k48=True),
        "wavekeyanchor": rdx.keyanchor_wave1_manifest(),
        "wavekeyanchor-neg1": rdx.keyanchor_wave_neg1_manifest(),
        "waveref": rdx.reference_arms_manifest(),
        "wavekeyanchor-confirm": rdx.keyanchor_confirm_manifest(),
        "wavekeyanchor-mech": rdx.keyanchor_mech_manifest(),
        "wavekeyanchor-mech-gate1": rdx.keyanchor_mech_gate1_manifest(),
    }

    fake_root = "/nonexistent/keyanchor_k48_e_smoke_out_dir"   # out_path() is a pure string join, no I/O
    # sec 11.11 point 6's own finding: out_path/is_done key on spec['name']
    # WITHIN a caller-supplied out_dir, i.e. filenames only truly collide
    # if two waves are ever pointed at the IDENTICAL out_dir. This smoke
    # checks BOTH: (a) within-a-shared-root filename collisions (the
    # stricter, belt-and-suspenders check this item registers), AND
    # (b) confirms every new wave gets its OWN out_dir subdirectory in
    # main() (so real collisions are structurally impossible regardless).
    new_manifests = {
        "wavekeyanchor-k48": k48_d, "wavekeyanchor-k48-ref": k48_ref,
        "wavekeyanchor-k48-gate1": k48_g1, "wavekeyanchor-k48 (fixed-lambda1)": k48_fix1,
        "wavekeyanchor-k48 (dprime)": k48_dprime, "wavekeyanchor-e": ke,
        "wavekeyanchor-e (e-fp)": ke_fp,
    }

    ok = True
    for new_label, new_manifest in new_manifests.items():
        new_paths = {rdx.out_path(fake_root, s) for s in new_manifest}
        unique_within = len(new_paths) == len(new_manifest)
        if not unique_within:
            print(f"    FAIL: {new_label} has internal out_path() duplicates")
        ok = ok and unique_within
        for prior_label, prior_manifest in prior_manifests.items():
            prior_paths = {rdx.out_path(fake_root, s) for s in prior_manifest}
            disjoint = new_paths.isdisjoint(prior_paths)
            if not disjoint:
                print(f"    FAIL: {new_label} collides with {prior_label} (shared fake_root)")
            ok = ok and disjoint
        print(f"    {new_label}: {len(new_manifest)} cells, unique_within={unique_within}, "
              f"disjoint from all {len(prior_manifests)} prior waves (shared root)=True")

    # The 'e' and 'e-fp' arms REALLY DO share one out_dir (both launch
    # under --wave keyanchor-e -> wavekeyanchor-e), so THIS pairwise check
    # is load-bearing, not belt-and-suspenders: their filenames must be
    # disjoint within the same directory (arm bit 'e' vs 'e-fp' + disjoint
    # seed blocks 60-62 vs 70-72 guarantee it -- verified here, not assumed).
    e_paths = {rdx.out_path(fake_root, s) for s in ke}
    e_fp_paths = {rdx.out_path(fake_root, s) for s in ke_fp}
    e_arms_disjoint = e_paths.isdisjoint(e_fp_paths)
    print(f"    'e' vs 'e-fp' (SHARED wavekeyanchor-e out_dir -- load-bearing check): "
          f"disjoint={e_arms_disjoint}")
    ok = ok and e_arms_disjoint

    # (b) real out_dir separation: each --wave value gets its OWN
    # subdirectory (main()'s own f"wave{args.wave}" convention), so even
    # without the shared-root check above, real collisions cannot occur.
    real_out_dirs = {
        "keyanchor-k48": "wavekeyanchor-k48", "keyanchor-k48-ref": "wavekeyanchor-k48-ref",
        "keyanchor-k48-gate1": "wavekeyanchor-k48-gate1", "keyanchor-e": "wavekeyanchor-e",
        "geo3": "wavegeo3",
    }
    all_distinct_dirs = len(set(real_out_dirs.values())) == len(real_out_dirs)
    print(f"    real per-wave out_dir subdirectories all distinct: {all_distinct_dirs} "
          f"({real_out_dirs})")
    ok = ok and all_distinct_dirs

    with tempfile.TemporaryDirectory() as tmp:
        all_not_done = all(
            not rdx.is_done(tmp, s)
            for manifest in new_manifests.values() for s in manifest)
    print(f"    is_done() False for every fresh K48/e cell (no data present): {all_not_done}")
    ok = ok and all_not_done

    _report("smoke 16: zero-collision check EXTENDED to include wavegeo3 (sec 11.9 item 16) -- "
            "every new K48/e manifest is collision-free against every prior wave, incl. wavegeo3, "
            "and gets its own out_dir subdirectory", ok)


# ---------------------------------------------------------------------------
# candidate (e) manifest shape: 3 cells, K=32, seeds {60,61,62}, arm 'e',
# anchor_table_frozen=True, anchor_table_init_mode='random_unit_rows',
# anchor_lambda_mode='fixed' at the registered 0.58.
# ---------------------------------------------------------------------------

def smoke_candidate_e_manifest_shape():
    ke = rdx.keyanchor_e_manifest()
    right_count = len(ke) == 3
    right_K = all(s["K"] == 32 for s in ke)
    right_seeds = sorted(s["seed"] for s in ke) == [60, 61, 62]
    right_arm = all(s["arm"] == "e" for s in ke)
    right_frozen = all(s.get("anchor_table_frozen") is True for s in ke)
    right_init_mode = all(s.get("anchor_table_init_mode") == "random_unit_rows" for s in ke)
    right_lambda_mode = all(s["anchor_lambda_mode"] == "fixed" for s in ke)
    right_lambda_value = all(s["anchor_lambda_fixed"] == rdx.KEYANCHOR_E_LAMBDA_FIXED for s in ke)
    right_drift_probe = all(s["drift_probe"] is True for s in ke)
    right_rev7 = all(s["rev7_engagement"] is True for s in ke)
    print(f"    count={len(ke)} Ks={sorted(set(s['K'] for s in ke))} "
          f"seeds={sorted(s['seed'] for s in ke)} arms={sorted(set(s['arm'] for s in ke))}")
    print(f"    anchor_table_frozen all True: {right_frozen}  "
          f"anchor_table_init_mode all 'random_unit_rows': {right_init_mode}")
    print(f"    anchor_lambda_mode all 'fixed': {right_lambda_mode}  "
          f"anchor_lambda_fixed all {rdx.KEYANCHOR_E_LAMBDA_FIXED}: {right_lambda_value}")
    print(f"    drift_probe all True: {right_drift_probe}  rev7_engagement all True: {right_rev7}")
    ok = (right_count and right_K and right_seeds and right_arm and right_frozen
          and right_init_mode and right_lambda_mode and right_lambda_value
          and right_drift_probe and right_rev7)
    _report("smoke: candidate (e) manifest shape (3 cells, K=32, seeds {60,61,62}, frozen "
            "random-unit-rows table, fixed lambda=0.58, full Rev-7.1 instrumentation)", ok)


# ---------------------------------------------------------------------------
# candidate (e-fp) manifest shape (audit prescription): 3 cells, K=32,
# seeds {70,71,72}, arm 'e-fp', anchor_table_frozen=True,
# anchor_table_init_mode='frame_potential' (the stub's LITERAL init text),
# same fixed lambda=0.58, full Rev-7.1 instrumentation; and the combined
# wave manifest is exactly e + e-fp = 6 cells.
# ---------------------------------------------------------------------------

def smoke_candidate_e_fp_manifest_shape():
    ke_fp = rdx.keyanchor_e_fp_manifest()
    right_count = len(ke_fp) == 3
    right_K = all(s["K"] == 32 for s in ke_fp)
    right_seeds = sorted(s["seed"] for s in ke_fp) == [70, 71, 72]
    right_arm = all(s["arm"] == "e-fp" for s in ke_fp)
    right_frozen = all(s.get("anchor_table_frozen") is True for s in ke_fp)
    right_init_mode = all(s.get("anchor_table_init_mode") == "frame_potential" for s in ke_fp)
    right_lambda_mode = all(s["anchor_lambda_mode"] == "fixed" for s in ke_fp)
    right_lambda_value = all(s["anchor_lambda_fixed"] == rdx.KEYANCHOR_E_LAMBDA_FIXED for s in ke_fp)
    right_drift_probe = all(s["drift_probe"] is True for s in ke_fp)
    right_rev7 = all(s["rev7_engagement"] is True for s in ke_fp)
    print(f"    count={len(ke_fp)} Ks={sorted(set(s['K'] for s in ke_fp))} "
          f"seeds={sorted(s['seed'] for s in ke_fp)} arms={sorted(set(s['arm'] for s in ke_fp))}")
    print(f"    anchor_table_frozen all True: {right_frozen}  "
          f"anchor_table_init_mode all 'frame_potential': {right_init_mode}")
    print(f"    anchor_lambda_mode all 'fixed': {right_lambda_mode}  "
          f"anchor_lambda_fixed all {rdx.KEYANCHOR_E_LAMBDA_FIXED}: {right_lambda_value}")
    print(f"    drift_probe all True: {right_drift_probe}  rev7_engagement all True: {right_rev7}")
    ok = (right_count and right_K and right_seeds and right_arm and right_frozen
          and right_init_mode and right_lambda_mode and right_lambda_value
          and right_drift_probe and right_rev7)
    _report("smoke: candidate (e-fp) manifest shape (3 cells, K=32, seeds {70,71,72}, FROZEN "
            "frame-potential table per the literal sec 10.13 stub, fixed lambda=0.58, full "
            "Rev-7.1 instrumentation)", ok)

    # the combined wave manifest: exactly e + e-fp, 6 cells, in that order.
    ke_all = rdx.keyanchor_e_wave_manifest()
    combined_ok = (len(ke_all) == 6
                   and ke_all == rdx.keyanchor_e_manifest() + rdx.keyanchor_e_fp_manifest())
    print(f"    keyanchor_e_wave_manifest(): {len(ke_all)} cells, == e + e-fp exactly: {combined_ok}")
    _report("smoke: keyanchor-e wave manifest is exactly both arms (e + e-fp, 6 cells)", combined_ok)


# ---------------------------------------------------------------------------
# K=48 manifest shapes: candidate (d) 3 cells/seeds{30,31,32}; reference
# arms 3 cells/seeds{1,2,3}/K=48; Gate-1 probe 1 cell/seed 0/5000 steps;
# fixed-lambda1 probe 1 cell/seed 50; candidate (d') 3 cells/seeds{40,41,42}.
# ---------------------------------------------------------------------------

def smoke_k48_manifest_shapes():
    ok = True

    k48_d = rdx.keyanchor_k48_manifest()
    d_ok = (len(k48_d) == 3 and all(s["K"] == 48 for s in k48_d)
            and sorted(s["seed"] for s in k48_d) == [30, 31, 32]
            and all(s["arm"] == "d" for s in k48_d)
            and all(s["anchor_lambda_mode"] == "learned" for s in k48_d)
            and all(s["rev7_engagement"] is True and s["drift_probe"] is True for s in k48_d))
    print(f"    candidate (d) K=48: count={len(k48_d)} seeds={sorted(s['seed'] for s in k48_d)} "
          f"shape_ok={d_ok}")
    ok = ok and d_ok

    k48_ref = rdx.keyanchor_k48_reference_manifest()
    ref_ok = (len(k48_ref) == 3 and all(s["K"] == 48 for s in k48_ref)
              and sorted(s["seed"] for s in k48_ref) == [1, 2, 3]
              and all(s["drift_probe"] is True for s in k48_ref))
    print(f"    reference arms K=48: count={len(k48_ref)} seeds={sorted(s['seed'] for s in k48_ref)} "
          f"shape_ok={ref_ok}")
    ok = ok and ref_ok

    k48_g1 = rdx.keyanchor_k48_gate1_manifest()
    g1_ok = (len(k48_g1) == 1 and k48_g1[0]["K"] == 48 and k48_g1[0]["seed"] == 0
             and k48_g1[0]["steps"] == rdx.KEYANCHOR_K48_GATE1_STEPS == 5000)
    print(f"    Gate-1 probe K=48: count={len(k48_g1)} seed={k48_g1[0]['seed']} "
          f"steps={k48_g1[0]['steps']} shape_ok={g1_ok}")
    ok = ok and g1_ok

    k48_fix1 = rdx.keyanchor_k48_fixed_lambda1_manifest()
    fix1_ok = (len(k48_fix1) == 1 and k48_fix1[0]["K"] == 48 and k48_fix1[0]["seed"] == 50
               and k48_fix1[0]["anchor_lambda_mode"] == "fixed"
               and k48_fix1[0]["anchor_lambda_fixed"] == 1.0)
    print(f"    fixed-lambda1 probe K=48: seed={k48_fix1[0]['seed']} "
          f"lambda_fixed={k48_fix1[0]['anchor_lambda_fixed']} shape_ok={fix1_ok}")
    ok = ok and fix1_ok

    k48_dprime = rdx.keyanchor_k48_dprime_manifest()
    dprime_ok = (len(k48_dprime) == 3 and all(s["K"] == 48 for s in k48_dprime)
                 and sorted(s["seed"] for s in k48_dprime) == [40, 41, 42]
                 and all(s["anchor_lambda_mode"] == "learned_per_entity" for s in k48_dprime))
    print(f"    candidate (d') K=48: count={len(k48_dprime)} "
          f"seeds={sorted(s['seed'] for s in k48_dprime)} shape_ok={dprime_ok}")
    ok = ok and dprime_ok

    _report("smoke: K=48 manifest shapes (candidate d/ref/gate1/fixed-lambda1/d') match sec 11.1's "
            "registered cell counts, seeds, and arm identity", ok)


# ---------------------------------------------------------------------------
# seed non-collision: every NEW seed block this build introduces is checked
# against every PRIOR seed block this program has ever used at the SAME K
# (sec 11.1's own escalating-decade-block discipline -- the seed integer
# alone need not be globally unique across DIFFERENT Ks, since K already
# disambiguates the archive, but within a K it must never silently overlap
# a prior wave's own seed identity for the SAME arm).
# ---------------------------------------------------------------------------

def smoke_seed_non_collision():
    # (K, arm-family) -> {seed: wave_label}, covering every block this
    # program has ever registered at K=32/K=48 for the anchor-arm family.
    prior_blocks = {
        (32, "d/c learned"): {0, 1, 2},                    # Wave 1 (keyanchor_wave1_manifest)
        (32, "d confirm"): {0, 1, 2},                      # keyanchor-confirm (same identity, different wave tag)
        (32, "d mech"): {10, 11, 12},                      # Rev 7.1 candidate (d)
        (32, "dprime mech"): {20, 21, 22},                 # Rev 7.1 candidate (d')
    }
    new_blocks = {
        (48, "d k48"): set(s["seed"] for s in rdx.keyanchor_k48_manifest()),          # {30,31,32}
        (48, "dprime k48"): set(s["seed"] for s in rdx.keyanchor_k48_dprime_manifest()),  # {40,41,42}
        (48, "d k48 fixed-lambda1"): set(s["seed"] for s in rdx.keyanchor_k48_fixed_lambda1_manifest()),  # {50}
        (48, "d k48 gate1"): set(s["seed"] for s in rdx.keyanchor_k48_gate1_manifest()),  # {0}
        (32, "e"): set(s["seed"] for s in rdx.keyanchor_e_manifest()),                # {60,61,62}
        (32, "e-fp"): set(s["seed"] for s in rdx.keyanchor_e_fp_manifest()),          # {70,71,72}
    }
    expected_new = {
        (48, "d k48"): {30, 31, 32},
        (48, "dprime k48"): {40, 41, 42},
        (48, "d k48 fixed-lambda1"): {50},
        (48, "d k48 gate1"): {0},
        (32, "e"): {60, 61, 62},
        (32, "e-fp"): {70, 71, 72},
    }
    ok = True
    for key, seeds in new_blocks.items():
        expected = expected_new[key]
        matches = seeds == expected
        print(f"    {key}: seeds={sorted(seeds)} (expect {sorted(expected)}) match={matches}")
        ok = ok and matches

    # Cross-check: within K=32, BOTH new candidate-(e)-family seed blocks
    # ({60,61,62} random and {70,71,72} frame-potential -- the audit's own
    # 'do not reuse 60-62' prescription) must not overlap ANY prior K=32
    # anchor-arm seed block (each is a fresh, never-before-used-anywhere
    # block per sec 11.1's own convention), NOR each other.
    for new_label in ("e", "e-fp"):
        new_seeds = new_blocks[(32, new_label)]
        for (K, label), seeds in prior_blocks.items():
            if K != 32:
                continue
            overlap = new_seeds & seeds
            no_overlap = len(overlap) == 0
            print(f"    candidate ({new_label}) {sorted(new_seeds)} vs prior K=32 block '{label}' "
                  f"{sorted(seeds)}: overlap={sorted(overlap)} (expect none)")
            ok = ok and no_overlap
    e_vs_efp_overlap = new_blocks[(32, "e")] & new_blocks[(32, "e-fp")]
    e_vs_efp_disjoint = len(e_vs_efp_overlap) == 0
    print(f"    candidate (e) {sorted(new_blocks[(32, 'e')])} vs candidate (e-fp) "
          f"{sorted(new_blocks[(32, 'e-fp')])}: overlap={sorted(e_vs_efp_overlap)} (expect none -- "
          f"the audit's own 'seeds 60-62 stay registered as the e random arm' prescription)")
    ok = ok and e_vs_efp_disjoint

    # Within K=48, the new blocks (d/{30,31,32}, dprime/{40,41,42},
    # fixed-lambda1/{50}, gate1/{0}) must be pairwise disjoint from each
    # other too (K=48 has no PRIOR anchor-arm seed history to check against
    # -- this is a genuinely new K -- so the only collision risk is WITHIN
    # this build's own new blocks).
    k48_new_only = {k: v for k, v in new_blocks.items() if k[0] == 48}
    items = list(k48_new_only.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (label_i, seeds_i), (label_j, seeds_j) = items[i], items[j]
            overlap = seeds_i & seeds_j
            no_overlap = len(overlap) == 0
            print(f"    K=48 '{label_i}' {sorted(seeds_i)} vs '{label_j}' {sorted(seeds_j)}: "
                  f"overlap={sorted(overlap)} (expect none)")
            ok = ok and no_overlap

    _report("smoke: seed non-collision -- every new seed block (K48 d/d'/fixed-lambda1/gate1, "
            "candidate (e) both arms) is exactly its registered set and disjoint from every "
            "prior/sibling block", ok)


def main() -> int:
    print("=" * 70)
    print("smoke_keyanchor_k48_e.py -- KEY_ANCHORING_DESIGN.md sec 11.9 (K=48) + "
          "sec 10.13 (candidate e, both arms) Wave -1 smoke suite (fla-free)")
    print("=" * 70)
    smoke_14_manifest_refactor_non_regression()
    smoke_15_gate2_k48_leg()
    smoke_16_zero_collision_incl_wavegeo3()
    smoke_candidate_e_manifest_shape()
    smoke_candidate_e_fp_manifest_shape()
    smoke_k48_manifest_shapes()
    smoke_seed_non_collision()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
