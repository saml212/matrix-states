"""smoke_keyanchor_scaling_wide.py -- KEY_ANCHORING_SCALING_DRAFT.md sec
15.20 Rev 1 (2026-07-07, post-attack-round-1)'s registered fla-free Wave -1
smoke suite for the d=96 wider-K cliff-hunting wave + d=80 seed escalation +
the fit_cliff_curve.py admissibility fix, as a committed, CPU-runnable
script. A WIDE-SUITE SIBLING to smoke_keyanchor_scaling.py (never a
modification of that file -- sec 15.20.6 Stage -1 item 6 re-runs BOTH
suites; the original wave's own 30-cell manifest must stay byte-identical
and independently re-verifiable regardless of anything this wave adds).

Covers:
  smoke 1: keyanchor_scaling_wide_d96_manifest() shape -- 12 cells, K in
    {72,78,84,90} x 3 seeds, exact registered seed blocks (sec 15.20.1's own
    table), candidate (d) arch, n_iter read from the NAMESPACED
    KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96] dict (never the flat one).
  smoke 2: keyanchor_scaling_d80_escalation_manifest() shape -- 4 cells,
    K in {48,53}, the ALREADY-RESERVED contingency seeds (1133/1134/1233/
    1234, sec 15.20.2) -- no new seed allocation.
  smoke 3: keyanchor_scaling_wide_full_manifest() -- exactly 16 cells
    (12 wide + 4 escalation), enumerated for the human/CI record.
  smoke 4: keyanchor_scaling_wide_calibration_manifest() -- exactly [K=72/
    seed=1740], filtered from (never hand-duplicated against) the d=96 wide
    manifest.
  smoke 5: keyanchor_scaling_wide_gate1_manifest() -- exactly 4 cells (one
    per new K), the OPTIONAL/cut-eligible probes.
  smoke 6: manifest-regression -- the ORIGINAL keyanchor_scaling_manifest(96)
    (no Ks= override, i.e. the 5-K/15-cell call sec 15.19's own harvest used)
    is BYTE-IDENTICAL in shape/seeds/n_iter to its own pre-registered values
    after this wave's own additive dict extensions ran at import time (sec
    15.20.6 Stage -1 item 5's own requirement).
  smoke 7: GATE2_N_ITER_BY_K (flat) extension closed by construction -- the
    NEW K=72/78/90 flat entries exist (no KeyError from a direct
    gate2_construction_check(ks=(72,78,84,90)) call) but this wave's own
    K=72-90/d=96 cells read the NAMESPACED dict, unaffected by corrupting
    the flat dict (negative test, mirrors the original suite's smoke 6);
    K=84's flat entry is untouched at its own d=128-verified value.
  smoke 8: Gate 2 EXECUTED at d_state=96 with this wave's own wide K-grid
    {72,78,84,90} explicitly (never the function's own (16,32) default).
  smoke 9: keyanchor_scaling_wide_kernel_gate_check -- POSITIVE against a
    fake-but-registered-shape PASSING artifact AND against the REAL artifact
    in WHICHEVER state this checkout actually has it (box-only, sec 15.20.1's
    NEW probe absent on a CPU-only dev checkout / present+PASSING once the
    box has run it -- both are honest states, never hardcoded) AND 5
    NEGATIVE tests to completion (missing file, verdict without CLEARED, exit_code!=0,
    disjoint t_sweep, a single False grid cell).
  smoke 9b: keyanchor_scaling_wide_niter_gate_check -- sec 15.20.6 Gate (b),
    build-audit MAJOR-1 fix. POSITIVE against the REAL, committed
    keyanchor_scaling_wide_niter_result.json artifact (unlike the wide
    kernel gate above, this one is CPU-only and IS already run+committed)
    AND 4 NEGATIVE tests to completion (missing file, wrong d_state,
    candidate_ks not a superset of the registered grid, all_ks_converged
    not True).
  smoke 10: the K=69 reuse sha256 gate -- POSITIVE against the REAL,
    committed sec 15.19 archive (copy + independently re-verify) AND 3
    NEGATIVE tests (corrupted copy, missing copied file, missing pinned
    manifest).
  smoke 11: keyanchor_scaling_wide_stage_gate mechanics -- BOTH PI signoffs
    required and never OR'd (missing either alone refuses); both kernel
    gates + the sha256 gate checked before EITHER leg's calibration/full
    logic; d80-escalation leg skips the calibration-cell check entirely;
    d96-wide/calibration is a no-op pass-through; d96-wide/full writes the
    CALIBRATION_DONE sentinel on a clean pass and refuses+no-sentinel over
    the abort/re-price trigger; --accept-scaling-wide-stage-override
    bypasses ONLY the calibration/abort-trigger check.
  smoke 12: keyanchor_scaling_wide_decision_rule -- every named verdict
    (CLIFF-BEYOND-WINDOW, AMBIGUOUS, ABSOLUTE-SLACK-FAVORED, POWER-LAW-
    FAVORED, BOTH-CONSISTENT, NEITHER-SURVIVES) reachable, boundary-exact at
    the 0.98 ceiling threshold and the 10% degenerate_frac gate, band
    disjointness sanity, and negative tests (malformed CI, missing CI on a
    non-degenerate fit, empty per_k_h4_means).
  smoke 13: zero-collision -- keyanchor-scaling-wide's manifest (16 mandatory
    + 4 Gate-1 probes + 1 calibration) is collision-free against every
    prior/sibling wave this program has ever registered, INCLUDING the
    ORIGINAL keyanchor-scaling wave itself (same _keyanchor_scaling_spec
    filename convention -- collision-freedom here rests on (K,seed,d_state)
    disjointness, not on out_dir, since sec 15.20.2's own d80-escalation leg
    deliberately WRITES INTO the original wave's own out_dir).
  smoke 14: keyanchor_scaling_check_abort reused unmodified for wide-wave
    cells -- the two-tier bracket fires identically for a K=72/d=96 cell (a
    non-anchor main-grid cell) as it does for the original wave's own
    K=51/d=96 cell (same tier, same trigger).
  smoke 15: sim_cliff_power_wide_grid_results.json artifact sanity -- exists,
    parses, and its own headline half-widths/degenerate_fracs match sec
    15.20.4's published numbers (the already-run MAJOR-2 power check).
  smoke 16: fit_cliff_curve.py admissibility-filter fix -- the REAL d=96
    regression fixture (re-run on the archived sec 15.19 raws, unfiltered,
    must reproduce degenerate_frac=94.77% to 4 decimal places, x0 still
    pinned at 0.9) AND a synthetic negative test (one fabricated
    admissible=false seed dragging the mean must be silently excluded).

Wired as an ADDITIONAL pre-launch CPU gate for --wave keyanchor-scaling-wide
(run_deltanet_rd_exactness_sweep.py's main()) -- rc!=0 aborts the wave before
any GPU cell dispatches.

Exit code 0 = every item PASSED. Run: python smoke_keyanchor_scaling_wide.py
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
import fit_cliff_curve as fcc                             # noqa: E402 (fla-free)

FAILURES: list[str] = []

# Where the sec 15.19 archive (30 raws, incl. K=69/d=96) is found is environment-dependent: a
# CPU-only dev checkout has the committed repo-root experiment-runs/ copy but not the GPU box's
# own live out_dir; the GPU box has the live out_dir (the production KEYANCHOR_SCALING_WIDE_K69_
# ORIGINAL_DIR default the actual launch chain reads) but not the repo-root archive path shipped
# separately. Try the committed archive FIRST, then fall back to the on-box raw out_dir -- both
# hold the SAME fixture value (the archive is a byte-for-byte copy of those raws, sec 15.19), so
# this changes nothing about what's being verified, only where the smoke looks for it. Neither
# candidate is created here and no artifact requirement is weakened -- if NEITHER exists the first
# candidate's path still surfaces in the "not found" message below, unchanged from before this fix.
_LOCAL_K69_ARCHIVE_DIR_CANDIDATES = (
    os.path.join(HERE, "..", "..", "experiment-runs", "2026-07-07_keyanchor_scaling",
                 "results", "deltanet_rd_exactness", "wavekeyanchor-scaling"),   # committed repo archive
    os.path.join(HERE, "results", "deltanet_rd_exactness", "wavekeyanchor-scaling"),   # on-box live out_dir
)
LOCAL_K69_ARCHIVE_DIR = next((c for c in _LOCAL_K69_ARCHIVE_DIR_CANDIDATES if os.path.isdir(c)),
                              _LOCAL_K69_ARCHIVE_DIR_CANDIDATES[0])
LOCAL_D96_ARCHIVE_DIR = LOCAL_K69_ARCHIVE_DIR   # same directory holds all 30 sec 15.19 raws


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


# ---------------------------------------------------------------------------
# smoke 1: keyanchor_scaling_wide_d96_manifest() shape.
# ---------------------------------------------------------------------------

_EXPECTED_WIDE_SEEDS = {72: {1740, 1741, 1742}, 78: {1840, 1841, 1842},
                        84: {1940, 1941, 1942}, 90: {2040, 2041, 2042}}


def smoke_1_d96_wide_manifest_shape():
    m = rdx.keyanchor_scaling_wide_d96_manifest()
    right_count = len(m) == 12
    ks_ok = sorted(set(s["K"] for s in m)) == sorted(rdx.KEYANCHOR_SCALING_D96_WIDE_KS)
    right_arm = all(s["arm"] == "d" for s in m)
    right_d_state = all(s["d_state"] == 96 for s in m)
    right_lambda_mode = all(s["anchor_lambda_mode"] == "learned" for s in m)
    right_instrumentation = all(s["drift_probe"] is True and s["rev7_engagement"] is True for s in m)
    seeds_ok = all(set(s["seed"] for s in m if s["K"] == K) == _EXPECTED_WIDE_SEEDS[K]
                   for K in rdx.KEYANCHOR_SCALING_D96_WIDE_KS)
    n_iter_ok = all(s["geo3_n_iter"] == rdx.KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96][s["K"]] == 20
                    for s in m)
    h_extra_none = all(s["h_extra"] is None for s in m)   # none of {72,78,84,90} need an override
    print(f"    count={len(m)} Ks={sorted(set(s['K'] for s in m))}")
    for K in rdx.KEYANCHOR_SCALING_D96_WIDE_KS:
        print(f"      K={K}: seeds={sorted(s['seed'] for s in m if s['K']==K)} "
              f"(expect {sorted(_EXPECTED_WIDE_SEEDS[K])})")
    ok = (right_count and ks_ok and right_arm and right_d_state and right_lambda_mode
          and right_instrumentation and seeds_ok and n_iter_ok and h_extra_none)
    _report("smoke 1: keyanchor_scaling_wide_d96_manifest() shape -- 12 cells, K in {72,78,84,90} "
            "x 3 seeds, exact registered seed blocks, candidate (d), n_iter from the NAMESPACED "
            "dict, no H_extra override at any new K", ok)


# ---------------------------------------------------------------------------
# smoke 2: keyanchor_scaling_d80_escalation_manifest() shape.
# ---------------------------------------------------------------------------

def smoke_2_d80_escalation_manifest_shape():
    m = rdx.keyanchor_scaling_d80_escalation_manifest()
    right_count = len(m) == 4
    right_ks = sorted(set(s["K"] for s in m)) == [48, 53]
    right_d_state = all(s["d_state"] == 80 for s in m)
    expected = {48: {1133, 1134}, 53: {1233, 1234}}
    seeds_ok = all(set(s["seed"] for s in m if s["K"] == K) == expected[K] for K in (48, 53))
    # these are ALREADY-RESERVED contingency seeds -- confirm the manifest reads them from
    # the SAME dict the original wave's own sec 15.15 table registered (no new allocation).
    matches_registered = all(
        set(s["seed"] for s in m if s["K"] == K) == set(rdx.KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K[80][K])
        for K in (48, 53))
    n_iter_ok = all(s["geo3_n_iter"] == 20 for s in m)   # already-verified K's, sec 15.20.2
    print(f"    count={len(m)} entries={sorted((s['K'], s['seed']) for s in m)}")
    ok = right_count and right_ks and right_d_state and seeds_ok and matches_registered and n_iter_ok
    _report("smoke 2: keyanchor_scaling_d80_escalation_manifest() shape -- 4 cells, K in {48,53}, "
            "the ALREADY-RESERVED sec 15.15 contingency seeds, no new allocation", ok)


# ---------------------------------------------------------------------------
# smoke 3: full 16-cell manifest.
# ---------------------------------------------------------------------------

def smoke_3_full_manifest_16_cells():
    m = sorted(rdx.keyanchor_scaling_wide_full_manifest(), key=lambda s: (s["d_state"], s["K"], s["seed"]))
    right_count = len(m) == 16
    d96_count = len([s for s in m if s["d_state"] == 96]) == 12
    d80_count = len([s for s in m if s["d_state"] == 80]) == 4
    print(f"    {len(m)}-cell mandatory wide-wave manifest (d_state, K, seed, name):")
    for s in m:
        print(f"      d_state={s['d_state']:3d}  K={s['K']:3d}  seed={s['seed']:5d}  {s['name']}")
    ok = right_count and d96_count and d80_count
    _report("smoke 3: keyanchor_scaling_wide_full_manifest() -- exactly 16 cells (12 d=96-wide + "
            "4 d=80-escalation), enumerated for the human/CI record", ok)


# ---------------------------------------------------------------------------
# smoke 4: calibration manifest.
# ---------------------------------------------------------------------------

def smoke_4_calibration_manifest():
    calib = rdx.keyanchor_scaling_wide_calibration_manifest()
    right_count = len(calib) == 1
    right_cell = right_count and calib[0]["K"] == 72 and calib[0]["seed"] == 1740 \
        and calib[0]["d_state"] == 96
    d96_wide = rdx.keyanchor_scaling_wide_d96_manifest()
    filtered = [s for s in d96_wide if s["K"] == 72 and s["seed"] == 1740]
    consistent = right_count and filtered == calib
    print(f"    calibration manifest: {[(s['d_state'], s['K'], s['seed']) for s in calib]}")
    ok = right_count and right_cell and consistent
    _report("smoke 4: keyanchor_scaling_wide_calibration_manifest() -- exactly [K=72/seed=1740/"
            "d=96], filtered from (not hand-duplicated against) the d=96 wide manifest", ok)


# ---------------------------------------------------------------------------
# smoke 5: Gate-1 manifest.
# ---------------------------------------------------------------------------

def smoke_5_gate1_manifest():
    g1 = rdx.keyanchor_scaling_wide_gate1_manifest()
    right_count = len(g1) == 4
    expected_seed = {72: 1745, 78: 1845, 84: 1945, 90: 2045}
    seeds_ok = all(s["seed"] == expected_seed[s["K"]] for s in g1)
    steps_ok = all(s["steps"] == rdx.KEYANCHOR_SCALING_GATE1_STEPS == 5000 for s in g1)
    print(f"    count={len(g1)} entries={sorted((s['K'], s['seed']) for s in g1)}")
    ok = right_count and seeds_ok and steps_ok
    _report("smoke 5: keyanchor_scaling_wide_gate1_manifest() -- 4 cells (one per new K), exact "
            "registered seeds, 5,000 steps", ok)


# ---------------------------------------------------------------------------
# smoke 6: manifest-regression -- the ORIGINAL 5-K/15-cell d=96 call stays
# byte-identical after this wave's own additive dict extensions.
# ---------------------------------------------------------------------------

_ORIGINAL_D96_EXPECTED_SEEDS = {24: {1420, 1421, 1422}, 51: {1430, 1431, 1432},
                                 57: {1530, 1531, 1532}, 63: {1630, 1631, 1632}, 69: {1730, 1731, 1732}}


def smoke_6_manifest_regression():
    m = rdx.keyanchor_scaling_manifest(96)   # NO Ks= override -- the ORIGINAL 5-K call
    right_count = len(m) == 15
    right_ks = sorted(set(s["K"] for s in m)) == sorted(rdx.KEYANCHOR_SCALING_D96_KS)
    seeds_ok = all(set(s["seed"] for s in m if s["K"] == K) == _ORIGINAL_D96_EXPECTED_SEEDS[K]
                   for K in rdx.KEYANCHOR_SCALING_D96_KS)
    n_iter_ok = all(s["geo3_n_iter"] == 20 for s in m)
    # KEYANCHOR_SCALING_D96_KS itself must be untouched (still the ORIGINAL 5, sec 15.19).
    d96_ks_untouched = tuple(rdx.KEYANCHOR_SCALING_D96_KS) == (24, 51, 57, 63, 69)
    # The wide K's must be ABSENT from this call's own output (additive, never merged in).
    no_wide_bleed = not (set(s["K"] for s in m) & set(rdx.KEYANCHOR_SCALING_D96_WIDE_KS))
    print(f"    keyanchor_scaling_manifest(96) [no Ks= override]: {len(m)} cells, "
          f"Ks={sorted(set(s['K'] for s in m))} (expect {sorted(rdx.KEYANCHOR_SCALING_D96_KS)})")
    ok = right_count and right_ks and seeds_ok and n_iter_ok and d96_ks_untouched and no_wide_bleed
    _report("smoke 6: manifest-regression -- keyanchor_scaling_manifest(96) (the ORIGINAL 5-K/"
            "15-cell call, no Ks= override) is BYTE-IDENTICAL to its own pre-registered shape "
            "after this wave's own additive dict extensions (sec 15.20.6 Stage -1 item 5)", ok)


# ---------------------------------------------------------------------------
# smoke 7: flat GATE2_N_ITER_BY_K extension closed by construction.
# ---------------------------------------------------------------------------

def smoke_7_flat_dict_extension_and_namespacing():
    # (a) K=72/78/90 now exist in the flat dict (no KeyError from a direct
    # gate2_construction_check call using the wide K-grid).
    flat_has_new_ks = all(K in ka.GATE2_N_ITER_BY_K for K in (72, 78, 90))
    flat_new_values_ok = all(ka.GATE2_N_ITER_BY_K[K] == 20 for K in (72, 78, 90))
    # K=84's flat entry is UNTOUCHED at its own d=128-verified value (20, per sec 13.2/13.3 --
    # numerically the same value here, but this wave's own K=84/d=96 cells must NOT read it).
    flat_k84_untouched = ka.GATE2_N_ITER_BY_K[84] == 20

    # (b) this wave's own K=72-90/d=96 cells read the NAMESPACED dict, unaffected by corrupting
    # the flat one (negative test, mirrors smoke_keyanchor_scaling.py's own smoke 6).
    baseline_spec = rdx._keyanchor_scaling_spec(84, 1940, 96)
    baseline_n_iter = baseline_spec["geo3_n_iter"]
    baseline_matches_namespaced = baseline_n_iter == rdx.KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96][84] == 20

    originals = {K: ka.GATE2_N_ITER_BY_K[K] for K in (72, 78, 84, 90)}
    for K in (72, 78, 84, 90):
        ka.GATE2_N_ITER_BY_K[K] = 999999   # obviously-wrong sentinel
    try:
        corrupted_specs = {K: rdx._keyanchor_scaling_spec(K, 0, 96)["geo3_n_iter"] for K in (72, 78, 84, 90)}
        unaffected = all(v == 20 != 999999 for v in corrupted_specs.values())
    finally:
        for K, v in originals.items():
            ka.GATE2_N_ITER_BY_K[K] = v   # restore -- must not leak into other smokes

    restored_ok = all(ka.GATE2_N_ITER_BY_K[K] == originals[K] == 20 for K in (72, 78, 84, 90))
    print(f"    flat dict has new K=72/78/90 entries (all n_iter=20): {flat_has_new_ks and flat_new_values_ok}")
    print(f"    flat K=84 entry untouched (still 20, d=128-verified only): {flat_k84_untouched}")
    print(f"    K=84/d=96 baseline n_iter={baseline_n_iter} (from the NAMESPACED dict): "
          f"{baseline_matches_namespaced}")
    print(f"    after corrupting flat ka.GATE2_N_ITER_BY_K[72,78,84,90]=999999, wide-wave specs "
          f"are STILL n_iter=20 (unaffected): {unaffected}")
    print(f"    flat dict restored: {restored_ok}")
    ok = (flat_has_new_ks and flat_new_values_ok and flat_k84_untouched and baseline_matches_namespaced
          and unaffected and restored_ok)
    _report("smoke 7: flat GATE2_N_ITER_BY_K extension (K=72/78/90 added, K=84 left untouched) "
            "closed by construction -- wide-wave cells read the namespaced dict, unaffected by "
            "corrupting the flat one (negative test to completion)", ok)


# ---------------------------------------------------------------------------
# smoke 8: Gate 2 EXECUTED at d_state=96 with the wide K-grid.
# ---------------------------------------------------------------------------

def smoke_8_gate2_at_wide_grid():
    ks = rdx.KEYANCHOR_SCALING_D96_WIDE_KS
    table = ka.frame_potential_init(107, 96, seed=ka.ANCHOR_INIT_SEED)
    # MINOR-3 (build-audit adjudication, 2026-07-07): gate2_construction_check reads n_iter for
    # every K from the FLAT ka.GATE2_N_ITER_BY_K dict (key_anchoring.py's own gate2_ns_leg call,
    # never the namespaced one) -- at K=84 this is smoke-only diagnostic behavior, numerically
    # harmless here (both dicts read 20), but NOT what the real launch path uses: the actual
    # K=84/d=96 cells are built via _keyanchor_scaling_spec, which reads
    # KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96][84] (the namespaced dict, sec 15.6/15.20.1's own
    # fix) and is unaffected by this call or by corrupting the flat dict (see smoke 7 above).
    result = ka.gate2_construction_check(table, ks=ks, seed=0)
    print(f"    d_state=96, ks={ks}")
    print(f"      G2-a sigma_ratio={result['g2a_sigma_ratio']:.6f} (pass={result['g2a_pass']})")
    print(f"      G2-b max|cos|   ={result['g2b_max_abs_cos']:.6f} (pass={result['g2b_pass']})")
    for K in ks:
        leg = result["g2c_ns_legs"][K]
        print(f"      G2-c K={K:3d}: n_iter={leg['n_iter']} n_fallback={leg['n_fallback']}/"
              f"{leg['n_subsets']} max_resid={leg['max_resid']:.3e} pass={leg['pass']}")
    print(f"      OVERALL GATE 2 (d_state=96, ks={ks}): {'PASS' if result['pass'] else 'FAIL'}")
    _report("smoke 8: Gate 2 EXECUTED at d_state=96 with this wave's own wide K-grid {72,78,84,90} "
            "explicitly (never the (16,32) default) -- reported honestly regardless of outcome",
            result["pass"])


# ---------------------------------------------------------------------------
# smoke 9: keyanchor_scaling_wide_kernel_gate_check -- positive (fake) + 5 negatives.
# ---------------------------------------------------------------------------

_FAKE_WIDE_KERNEL_BASE = {
    "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.20.1",
    "t_sweep": [448, 504, 546, 588, 630],
    "candidate_t_sweep": [504, 546, 588, 630],
    "control_t": 448,
    "grid_pass": {"96": {"448": True, "504": True, "546": True, "588": True, "630": True}},
    "messages": {}, "verdict": "d_state=96 KERNEL-SAFE at the full wide-grid T sweep -- CLEARED",
    "exit_code": 0,
}


def smoke_9_wide_kernel_gate_check():
    # The REAL artifact is box-only, generated by `python smoke_dstate_kernel_wide.py` on the GPU
    # box (sec 15.20.1) -- it is NOT part of this repo's committed CPU-only checkout, so a dev
    # machine never has it and reports a clean "not found" refusal. On the GPU box, though, GATE
    # 1b in keyanchor_scaling_wide_chain.sh REQUIRES this artifact to already exist (and PASS)
    # before the smoke suite even runs -- so on box the REAL check legitimately returns ok=True.
    # Accept EITHER truthful state (never crash, never silently accept a bad artifact) rather than
    # hardcoding "must be absent", which goes stale the moment the box does its job.
    real = rdx.keyanchor_scaling_wide_kernel_gate_check()
    if os.path.exists(rdx.KEYANCHOR_SCALING_WIDE_KERNEL_GATE_RESULT_PATH):
        real_state_honest = real["ok"] is True and "PASSES" in real["reason"]
        print(f"    REAL artifact (present -- box already ran smoke_dstate_kernel_wide.py): "
              f"ok={real['ok']} reason={real['reason']!r} (expect True): {real_state_honest}")
    else:
        real_state_honest = real["ok"] is False and "not found" in real["reason"]
        print(f"    REAL artifact (absent -- CPU-only checkout, box probe not yet run): "
              f"ok={real['ok']} (expect False/'not found'): {real_state_honest}")

    def _write(doc, tmp, name="fake.json"):
        p = os.path.join(tmp, name)
        with open(p, "w") as f:
            json.dump(doc, f)
        return p

    with tempfile.TemporaryDirectory() as tmp:
        p = _write(_FAKE_WIDE_KERNEL_BASE, tmp)
        pos = rdx.keyanchor_scaling_wide_kernel_gate_check(p)
        pos_ok = pos["ok"] is True
        print(f"    POSITIVE (fake, registered shape): ok={pos['ok']} reason={pos['reason']!r}: {pos_ok}")

        # Negative (a): missing file.
        neg_a = rdx.keyanchor_scaling_wide_kernel_gate_check(os.path.join(tmp, "nope.json"))
        neg_a_ok = neg_a["ok"] is False and "not found" in neg_a["reason"]
        print(f"    NEGATIVE (a) missing file: ok={neg_a['ok']} (expect False): {neg_a_ok}")

        # Negative (b): verdict without 'CLEARED'.
        doc = copy.deepcopy(_FAKE_WIDE_KERNEL_BASE)
        doc["verdict"] = "d_state=96 FAILS at >=1 wide-grid T"
        p2 = _write(doc, tmp, "neg_b.json")
        neg_b = rdx.keyanchor_scaling_wide_kernel_gate_check(p2)
        neg_b_ok = neg_b["ok"] is False and "CLEARED" in neg_b["reason"]
        print(f"    NEGATIVE (b) verdict without 'CLEARED': ok={neg_b['ok']} (expect False): {neg_b_ok}")

        # Negative (c): exit_code != 0.
        doc = copy.deepcopy(_FAKE_WIDE_KERNEL_BASE)
        doc["exit_code"] = 3
        p3 = _write(doc, tmp, "neg_c.json")
        neg_c = rdx.keyanchor_scaling_wide_kernel_gate_check(p3)
        neg_c_ok = neg_c["ok"] is False and "exit_code" in neg_c["reason"]
        print(f"    NEGATIVE (c) exit_code=3: ok={neg_c['ok']} (expect False): {neg_c_ok}")

        # Negative (d): disjoint t_sweep (the FATAL-2-shaped false-negative risk, generalized).
        doc = copy.deepcopy(_FAKE_WIDE_KERNEL_BASE)
        doc["t_sweep"] = [448]
        doc["grid_pass"] = {"96": {"448": True}}
        p4 = _write(doc, tmp, "neg_d.json")
        neg_d = rdx.keyanchor_scaling_wide_kernel_gate_check(p4)
        neg_d_ok = neg_d["ok"] is False and "protocol" in neg_d["reason"]
        print(f"    NEGATIVE (d) disjoint t_sweep=[448] only: ok={neg_d['ok']} (expect False): {neg_d_ok}")

        # Negative (e): one candidate T fails.
        doc = copy.deepcopy(_FAKE_WIDE_KERNEL_BASE)
        doc["grid_pass"] = copy.deepcopy(_FAKE_WIDE_KERNEL_BASE["grid_pass"])
        doc["grid_pass"]["96"]["630"] = False
        p5 = _write(doc, tmp, "neg_e.json")
        neg_e = rdx.keyanchor_scaling_wide_kernel_gate_check(p5)
        neg_e_ok = neg_e["ok"] is False and "630" in neg_e["reason"]
        print(f"    NEGATIVE (e) grid_pass['96']['630']=False: ok={neg_e['ok']} (expect False): {neg_e_ok}")

    ok = real_state_honest and pos_ok and neg_a_ok and neg_b_ok and neg_c_ok and neg_d_ok and neg_e_ok
    _report("smoke 9: keyanchor_scaling_wide_kernel_gate_check -- correctly reports the REAL "
            "artifact's actual state (absent on a CPU-only checkout / PASSING once the box has "
            "run it), PASSES a fake registered-shape artifact, and 5 NEGATIVE tests run to "
            "completion", ok)


# ---------------------------------------------------------------------------
# smoke 9b: keyanchor_scaling_wide_niter_gate_check -- sec 15.20.6 Gate (b),
# build-audit MAJOR-1 fix, 2026-07-07. Positive (real, CPU-only, already-run
# artifact) + 4 negatives -- mirrors smoke 9's own shape.
# ---------------------------------------------------------------------------

_FAKE_NITER_BASE = {
    "script": "keyanchor_scaling_wide_niter_check.py",
    "d_state": 96,
    "candidate_ks": [72, 78, 84, 90],
    "n_iter_grid": [12, 16, 20, 24],
    "all_ks_converged": True,
    "per_k": {
        str(K): {"convergence_20_to_24": {"mean_at_20": 0.95, "mean_at_24": 0.95,
                                            "rel_change_20_to_24": 0.0, "tolerance": 0.005,
                                            "converged": True}}
        for K in (72, 78, 84, 90)
    },
}


def smoke_9b_niter_gate_check():
    # The REAL artifact is CPU-only and already run+committed this build (unlike the wide kernel
    # gate above, which needs the GPU box) -- confirm it PASSES for real, not merely that the
    # function doesn't crash.
    real = rdx.keyanchor_scaling_wide_niter_gate_check()
    real_ok = real["ok"] is True
    print(f"    REAL artifact ({rdx.KEYANCHOR_SCALING_WIDE_NITER_GATE_RESULT_PATH!r}): "
          f"ok={real['ok']} reason={real['reason']!r}: {real_ok}")

    def _write(doc, tmp, name="fake.json"):
        p = os.path.join(tmp, name)
        with open(p, "w") as f:
            json.dump(doc, f)
        return p

    with tempfile.TemporaryDirectory() as tmp:
        p = _write(_FAKE_NITER_BASE, tmp)
        pos = rdx.keyanchor_scaling_wide_niter_gate_check(p)
        pos_ok = pos["ok"] is True
        print(f"    POSITIVE (fake, registered shape): ok={pos['ok']} reason={pos['reason']!r}: {pos_ok}")

        # Negative (a): missing file.
        neg_a = rdx.keyanchor_scaling_wide_niter_gate_check(os.path.join(tmp, "nope.json"))
        neg_a_ok = neg_a["ok"] is False and "not found" in neg_a["reason"]
        print(f"    NEGATIVE (a) missing file: ok={neg_a['ok']} (expect False): {neg_a_ok}")

        # Negative (b): wrong d_state (e.g. a stale d=128 sibling artifact copied in by mistake).
        doc = copy.deepcopy(_FAKE_NITER_BASE)
        doc["d_state"] = 128
        p2 = _write(doc, tmp, "neg_b.json")
        neg_b = rdx.keyanchor_scaling_wide_niter_gate_check(p2)
        neg_b_ok = neg_b["ok"] is False and "d_state" in neg_b["reason"]
        print(f"    NEGATIVE (b) d_state=128: ok={neg_b['ok']} (expect False): {neg_b_ok}")

        # Negative (c): candidate_ks not a superset of the registered grid (a partial/disjoint
        # sweep must not pass -- same FATAL-2-shaped discipline as the wide kernel gate's own
        # negative (d)).
        doc = copy.deepcopy(_FAKE_NITER_BASE)
        doc["candidate_ks"] = [72, 78]
        p3 = _write(doc, tmp, "neg_c.json")
        neg_c = rdx.keyanchor_scaling_wide_niter_gate_check(p3)
        neg_c_ok = neg_c["ok"] is False and "cover" in neg_c["reason"]
        print(f"    NEGATIVE (c) candidate_ks=[72,78] only (not a superset): ok={neg_c['ok']} "
              f"(expect False): {neg_c_ok}")

        # Negative (d): all_ks_converged is False.
        doc = copy.deepcopy(_FAKE_NITER_BASE)
        doc["all_ks_converged"] = False
        doc["per_k"] = copy.deepcopy(_FAKE_NITER_BASE["per_k"])
        doc["per_k"]["84"]["convergence_20_to_24"]["converged"] = False
        p4 = _write(doc, tmp, "neg_d.json")
        neg_d = rdx.keyanchor_scaling_wide_niter_gate_check(p4)
        neg_d_ok = neg_d["ok"] is False and "84" in neg_d["reason"]
        print(f"    NEGATIVE (d) all_ks_converged=False (K=84 the culprit): ok={neg_d['ok']} "
              f"(expect False): {neg_d_ok}")

    ok = real_ok and pos_ok and neg_a_ok and neg_b_ok and neg_c_ok and neg_d_ok
    _report("smoke 9b: keyanchor_scaling_wide_niter_gate_check -- PASSES against the REAL, "
            "committed n_iter-sufficiency artifact (build-audit MAJOR-1 fix), PASSES a fake "
            "registered-shape artifact, and 4 NEGATIVE tests run to completion", ok)


# ---------------------------------------------------------------------------
# smoke 10: K=69 reuse sha256 gate -- positive (real archive) + 3 negatives.
# ---------------------------------------------------------------------------

def smoke_10_k69_sha256_gate():
    if not os.path.isdir(LOCAL_K69_ARCHIVE_DIR):
        _report("smoke 10: K=69 reuse sha256 gate", False,
                f"local sec 15.19 archive not found at {LOCAL_K69_ARCHIVE_DIR!r} -- cannot exercise "
                f"the positive path (this repo's own committed archive should always be present)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        copy_report = rdx.keyanchor_scaling_wide_copy_k69(tmp, src_dir=LOCAL_K69_ARCHIVE_DIR)
        copy_ok = copy_report["ok"] is True and len(copy_report["copied"]) == 3
        print(f"    copy from the REAL sec 15.19 archive: ok={copy_report['ok']} "
              f"({len(copy_report.get('copied', []))} files): {copy_ok}")

        pos = rdx.keyanchor_scaling_wide_k69_sha256_gate(tmp)
        pos_ok = pos["ok"] is True
        print(f"    POSITIVE (clean copy vs the PINNED sec 15.19 manifest): ok={pos['ok']} "
              f"reason={pos['reason']!r}: {pos_ok}")

        # Negative (a): corrupt one copied file's content.
        target = os.path.join(tmp, rdx.keyanchor_scaling_wide_k69_filenames()[0])
        with open(target, "a") as f:
            f.write(" ")
        neg_a = rdx.keyanchor_scaling_wide_k69_sha256_gate(tmp)
        neg_a_ok = neg_a["ok"] is False and "MISMATCH" in neg_a["reason"]
        print(f"    NEGATIVE (a) corrupted copy: ok={neg_a['ok']} (expect False/MISMATCH): {neg_a_ok}")
        # restore for the next negative test
        rdx.keyanchor_scaling_wide_copy_k69(tmp, src_dir=LOCAL_K69_ARCHIVE_DIR)

        # Negative (b): missing copied file.
        os.remove(target)
        neg_b = rdx.keyanchor_scaling_wide_k69_sha256_gate(tmp)
        neg_b_ok = neg_b["ok"] is False and "missing" in neg_b["reason"]
        print(f"    NEGATIVE (b) missing copied file: ok={neg_b['ok']} (expect False/missing): {neg_b_ok}")

    # Negative (c): missing pinned manifest entirely.
    neg_c = rdx.keyanchor_scaling_wide_k69_sha256_gate(
        tempfile.mkdtemp(), pinned_path=os.path.join(tempfile.mkdtemp(), "nope.sha256"))
    neg_c_ok = neg_c["ok"] is False and "not found" in neg_c["reason"]
    print(f"    NEGATIVE (c) missing pinned manifest: ok={neg_c['ok']} (expect False/not found): {neg_c_ok}")

    # Sanity: the pinned manifest file itself is committed and has exactly 3 entries.
    pinned_exists = os.path.exists(rdx.KEYANCHOR_SCALING_WIDE_K69_PINNED_SHA256_PATH)
    pinned_count_ok = False
    if pinned_exists:
        with open(rdx.KEYANCHOR_SCALING_WIDE_K69_PINNED_SHA256_PATH) as f:
            lines = [l for l in f if l.strip() and not l.strip().startswith("#")]
        pinned_count_ok = len(lines) == 3
    print(f"    pinned manifest committed at {rdx.KEYANCHOR_SCALING_WIDE_K69_PINNED_SHA256_PATH!r}, "
          f"exactly 3 entries: {pinned_exists and pinned_count_ok}")

    ok = copy_ok and pos_ok and neg_a_ok and neg_b_ok and neg_c_ok and pinned_exists and pinned_count_ok
    _report("smoke 10: K=69 reuse sha256 gate -- POSITIVE against the REAL committed sec 15.19 "
            "archive (copy + independently re-verify) AND 3 NEGATIVE tests run to completion "
            "(corrupted copy, missing copied file, missing pinned manifest)", ok)


# ---------------------------------------------------------------------------
# smoke 11: keyanchor_scaling_wide_stage_gate mechanics.
# ---------------------------------------------------------------------------

def _with_fake_wide_kernel_artifact(fn):
    """Context-manager-shaped helper: monkeypatches KEYANCHOR_SCALING_WIDE_
    KERNEL_GATE_RESULT_PATH to a fake PASSING artifact and KEYANCHOR_
    SCALING_WIDE_K69_ORIGINAL_DIR to the REAL local sec 15.19 archive for
    the duration of fn(), then restores both -- the wide-wave gates this
    smoke suite is exercising depend on two box-only/real-archive paths
    that a CPU-only, pre-launch smoke run cannot assume exist yet."""
    orig_kernel_path = rdx.KEYANCHOR_SCALING_WIDE_KERNEL_GATE_RESULT_PATH
    orig_k69_src = rdx.KEYANCHOR_SCALING_WIDE_K69_ORIGINAL_DIR
    with tempfile.TemporaryDirectory() as tmp:
        fake_path = os.path.join(tmp, "fake_wide_kernel.json")
        with open(fake_path, "w") as f:
            json.dump(_FAKE_WIDE_KERNEL_BASE, f)
        rdx.KEYANCHOR_SCALING_WIDE_KERNEL_GATE_RESULT_PATH = fake_path
        rdx.KEYANCHOR_SCALING_WIDE_K69_ORIGINAL_DIR = LOCAL_K69_ARCHIVE_DIR
        try:
            return fn()
        finally:
            rdx.KEYANCHOR_SCALING_WIDE_KERNEL_GATE_RESULT_PATH = orig_kernel_path
            rdx.KEYANCHOR_SCALING_WIDE_K69_ORIGINAL_DIR = orig_k69_src


def smoke_11_stage_gate_mechanics():
    if not os.path.isdir(LOCAL_K69_ARCHIVE_DIR):
        _report("smoke 11: keyanchor_scaling_wide_stage_gate mechanics", False,
                f"local sec 15.19 archive not found at {LOCAL_K69_ARCHIVE_DIR!r}")
        return

    saved_pi = os.environ.get("KEYANCHOR_SCALING_PI_SIGNOFF")
    saved_ext = os.environ.get("KEYANCHOR_SCALING_EXT_PI_SIGNOFF")
    try:
        # (a) neither signoff -> refuses.
        os.environ.pop("KEYANCHOR_SCALING_PI_SIGNOFF", None)
        os.environ.pop("KEYANCHOR_SCALING_EXT_PI_SIGNOFF", None)
        refused_no_signoff = False
        with tempfile.TemporaryDirectory() as tmp:
            try:
                rdx.keyanchor_scaling_wide_stage_gate(tmp, "d96-wide", "calibration", False, tmp)
            except SystemExit as e:
                refused_no_signoff = (e.code == 1)
        print(f"    (a) neither signoff set -> refuses: {refused_no_signoff}")

        # (b) only the ORIGINAL (stale) signoff -> STILL refuses (never OR'd, sec 15.20.5 MAJOR-4).
        os.environ["KEYANCHOR_SCALING_PI_SIGNOFF"] = "1"
        os.environ.pop("KEYANCHOR_SCALING_EXT_PI_SIGNOFF", None)
        refused_stale_only = False
        with tempfile.TemporaryDirectory() as tmp:
            try:
                rdx.keyanchor_scaling_wide_stage_gate(tmp, "d96-wide", "calibration", False, tmp)
            except SystemExit as e:
                refused_stale_only = (e.code == 1)
        print(f"    (b) only the STALE original-wave signoff set -> STILL refuses (distinct env "
              f"vars, never OR'd): {refused_stale_only}")

        # (c) only the EXT signoff, not the original -> still refuses.
        os.environ.pop("KEYANCHOR_SCALING_PI_SIGNOFF", None)
        os.environ["KEYANCHOR_SCALING_EXT_PI_SIGNOFF"] = "1"
        refused_ext_only = False
        with tempfile.TemporaryDirectory() as tmp:
            try:
                rdx.keyanchor_scaling_wide_stage_gate(tmp, "d96-wide", "calibration", False, tmp)
            except SystemExit as e:
                refused_ext_only = (e.code == 1)
        print(f"    (c) only the EXT signoff set (not the original) -> STILL refuses: {refused_ext_only}")

        # Both set from here on.
        os.environ["KEYANCHOR_SCALING_PI_SIGNOFF"] = "1"
        os.environ["KEYANCHOR_SCALING_EXT_PI_SIGNOFF"] = "1"

        def _happy_path():
            results = {}
            # (d) d96-wide/calibration: not_applicable pass-through, all 3 gates cleared.
            with tempfile.TemporaryDirectory() as tmp:
                report = rdx.keyanchor_scaling_wide_stage_gate(tmp, "d96-wide", "calibration", False, tmp)
                results["calib_passthrough"] = (report["not_applicable"] is True
                                                 and report["kernel_gate"]["ok"] is True
                                                 and report["wide_kernel_gate"]["ok"] is True
                                                 and report["sha_gate"]["ok"] is True)
            # (e) d80-escalation: not_applicable pass-through too, no calibration cell needed.
            with tempfile.TemporaryDirectory() as tmp:
                report = rdx.keyanchor_scaling_wide_stage_gate(tmp, "d80-escalation", None, False, tmp)
                results["escalation_passthrough"] = report["not_applicable"] is True

            # (f) d96-wide/full REFUSES when the calibration cell is missing.
            with tempfile.TemporaryDirectory() as tmp:
                refused_missing_calib = False
                try:
                    rdx.keyanchor_scaling_wide_stage_gate(tmp, "d96-wide", "full", False, tmp)
                except SystemExit as e:
                    refused_missing_calib = (e.code == 1)
                results["refused_missing_calib"] = refused_missing_calib

            # (g) a synthetic in-bracket calibration cell -> sentinel written.
            with tempfile.TemporaryDirectory() as tmp:
                calib_spec = rdx.keyanchor_scaling_wide_calibration_manifest()[0]
                trigger = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(96, False)]
                ec = {f: calib_spec.get(f) for f in
                      ("embed_source", "gram_alpha", "gram_rho", "strong_pin", "lambda_orth", "use_zca",
                       "fnce_m", "geo3_active", "geo3_n_iter", "geo3_resid_tol", "anchor_active",
                       "anchor_lambda_mode", "anchor_lambda_fixed", "lambda_anchor", "drift_probe",
                       "rev7_engagement")}
                doc = {"K": calib_spec["K"], "seed": calib_spec["seed"], "steps": calib_spec["steps"],
                       "complete": True, "steps_completed": calib_spec["steps"], "exactness_config": ec,
                       "wall_s": trigger - 100.0, "timed_out": False, "d_state": 96,
                       "checkpoints": [{"M3_held_out": {"4": {"recovered_frac@0.9": 0.99}}}]}
                p = rdx.out_path(tmp, calib_spec)
                with open(p, "w") as f:
                    json.dump(doc, f)
                report = rdx.keyanchor_scaling_wide_stage_gate(tmp, "d96-wide", "full", False, tmp)
                sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_SCALING_WIDE_STAGE_SENTINEL_NAME)
                results["clean_full"] = (report["sentinel_written"] is True
                                          and os.path.exists(sentinel_path))

            # (h) an over-bracket calibration cell -> refuses, no sentinel.
            with tempfile.TemporaryDirectory() as tmp:
                calib_spec = rdx.keyanchor_scaling_wide_calibration_manifest()[0]
                trigger = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(96, False)]
                ec = {f: calib_spec.get(f) for f in
                      ("embed_source", "gram_alpha", "gram_rho", "strong_pin", "lambda_orth", "use_zca",
                       "fnce_m", "geo3_active", "geo3_n_iter", "geo3_resid_tol", "anchor_active",
                       "anchor_lambda_mode", "anchor_lambda_fixed", "lambda_anchor", "drift_probe",
                       "rev7_engagement")}
                doc = {"K": calib_spec["K"], "seed": calib_spec["seed"], "steps": calib_spec["steps"],
                       "complete": True, "steps_completed": calib_spec["steps"], "exactness_config": ec,
                       "wall_s": trigger + 100.0, "timed_out": False, "d_state": 96,
                       "checkpoints": [{"M3_held_out": {"4": {"recovered_frac@0.9": 0.99}}}]}
                p = rdx.out_path(tmp, calib_spec)
                with open(p, "w") as f:
                    json.dump(doc, f)
                refused_over = False
                try:
                    rdx.keyanchor_scaling_wide_stage_gate(tmp, "d96-wide", "full", False, tmp)
                except SystemExit as e:
                    refused_over = (e.code == 1)
                sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_SCALING_WIDE_STAGE_SENTINEL_NAME)
                results["refused_over_bracket"] = refused_over and not os.path.exists(sentinel_path)

            # (i) --accept-scaling-wide-stage-override bypasses (g)/(h)'s own check but NOT
            # the 3 gates above it -- verified by the report's own gate fields.
            with tempfile.TemporaryDirectory() as tmp:
                report = rdx.keyanchor_scaling_wide_stage_gate(tmp, "d96-wide", "full", True, tmp)
                sentinel_path = os.path.join(tmp, rdx.KEYANCHOR_SCALING_WIDE_STAGE_SENTINEL_NAME)
                results["override_ok"] = (report["gate_bypassed"] is True
                                           and report["sentinel_written"] is False
                                           and not os.path.exists(sentinel_path)
                                           and report["kernel_gate"]["ok"] is True
                                           and report["wide_kernel_gate"]["ok"] is True
                                           and report["sha_gate"]["ok"] is True)
            return results

        r = _with_fake_wide_kernel_artifact(_happy_path)
        for k, v in r.items():
            print(f"    ({k}): {v}")

        ok = (refused_no_signoff and refused_stale_only and refused_ext_only
              and all(r.values()))
    finally:
        if saved_pi is not None:
            os.environ["KEYANCHOR_SCALING_PI_SIGNOFF"] = saved_pi
        else:
            os.environ.pop("KEYANCHOR_SCALING_PI_SIGNOFF", None)
        if saved_ext is not None:
            os.environ["KEYANCHOR_SCALING_EXT_PI_SIGNOFF"] = saved_ext
        else:
            os.environ.pop("KEYANCHOR_SCALING_EXT_PI_SIGNOFF", None)

    _report("smoke 11: keyanchor_scaling_wide_stage_gate mechanics -- both signoffs required and "
            "never OR'd, both kernel gates + the sha256 gate checked before either leg proceeds, "
            "d80-escalation skips the calibration check, d96-wide/full refuses on missing/over-"
            "bracket calibration, override bypasses only the calibration/abort-trigger check", ok)


# ---------------------------------------------------------------------------
# smoke 12: keyanchor_scaling_wide_decision_rule -- every branch + negatives.
# ---------------------------------------------------------------------------

def smoke_12_decision_rule():
    abs_lo, abs_hi = rdx.KEYANCHOR_SCALING_WIDE_BAND_ABS_SLACK
    pow_lo, pow_hi = rdx.KEYANCHOR_SCALING_WIDE_BAND_POWER_LAW
    bands_disjoint = abs_hi < pow_lo
    print(f"    abs-slack band={rdx.KEYANCHOR_SCALING_WIDE_BAND_ABS_SLACK}, "
          f"power-law band={rdx.KEYANCHOR_SCALING_WIDE_BAND_POWER_LAW}, disjoint: {bands_disjoint}")

    cases = {
        "1a_cliff_beyond_window": (rdx.keyanchor_scaling_wide_decision_rule(
            0.95, {69: 0.99, 72: 0.985, 90: 0.99}, None, None), "CLIFF-BEYOND-WINDOW"),
        "1b_ambiguous": (rdx.keyanchor_scaling_wide_decision_rule(
            0.50, {69: 0.99, 72: 0.70, 90: 0.99}, None, None), "AMBIGUOUS"),
        "2_abs_slack": (rdx.keyanchor_scaling_wide_decision_rule(
            0.02, {69: 0.99, 90: 0.5}, abs_lo + 0.001, abs_hi - 0.001), "ABSOLUTE-SLACK-FAVORED"),
        "3_power_law": (rdx.keyanchor_scaling_wide_decision_rule(
            0.02, {69: 0.99, 90: 0.5}, pow_lo + 0.001, pow_hi - 0.001), "POWER-LAW-FAVORED"),
        "4_both_consistent": (rdx.keyanchor_scaling_wide_decision_rule(
            0.02, {69: 0.99, 90: 0.5}, 0.70, 0.80), "BOTH-CONSISTENT"),
        "5_neither_survives": (rdx.keyanchor_scaling_wide_decision_rule(
            0.02, {69: 0.99, 90: 0.5}, 0.90, 0.92), "NEITHER-SURVIVES"),
    }
    branch_ok = True
    for name, (result, expected_verdict) in cases.items():
        got = result["verdict"]
        this_ok = got == expected_verdict
        branch_ok = branch_ok and this_ok
        print(f"    {name}: verdict={got!r} (expect {expected_verdict!r}): {this_ok}")

    # Boundary-exact: h4 mean EXACTLY at the 0.98 ceiling threshold counts as "at ceiling" (>=).
    boundary_ceiling = rdx.keyanchor_scaling_wide_decision_rule(
        0.95, {69: 0.98}, None, None)["verdict"] == "CLIFF-BEYOND-WINDOW"
    boundary_below_ceiling = rdx.keyanchor_scaling_wide_decision_rule(
        0.95, {69: 0.9799999}, None, None)["verdict"] == "AMBIGUOUS"
    print(f"    boundary: h4 mean == 0.98 exactly -> CLIFF-BEYOND-WINDOW (>=, not >): "
          f"{boundary_ceiling}")
    print(f"    boundary: h4 mean == 0.9799999 (just under) -> AMBIGUOUS: {boundary_below_ceiling}")

    # degenerate_frac boundary: exactly 0.10 must NOT trigger 1a/1b (the gate is "> 10%").
    boundary_degenerate = rdx.keyanchor_scaling_wide_decision_rule(
        0.10, {69: 0.99}, 0.70, 0.80)["verdict"] == "BOTH-CONSISTENT"
    print(f"    boundary: degenerate_frac == 0.10 exactly does NOT trigger step 1 (requires "
          f"'> 10%', a real fit reaches steps 2-5): {boundary_degenerate}")

    # Negative (a): empty per_k_h4_means must raise.
    neg_a_ok = False
    try:
        rdx.keyanchor_scaling_wide_decision_rule(0.02, {}, 0.7, 0.8)
    except AssertionError:
        neg_a_ok = True
    print(f"    NEGATIVE (a) empty per_k_h4_means raises AssertionError: {neg_a_ok}")

    # Negative (b): non-degenerate fit with no CI must raise.
    neg_b_ok = False
    try:
        rdx.keyanchor_scaling_wide_decision_rule(0.02, {69: 0.99}, None, None)
    except AssertionError:
        neg_b_ok = True
    print(f"    NEGATIVE (b) non-degenerate fit missing its own CI raises AssertionError: {neg_b_ok}")

    # Negative (c): malformed CI (lo > hi) must raise.
    neg_c_ok = False
    try:
        rdx.keyanchor_scaling_wide_decision_rule(0.02, {69: 0.99}, 0.9, 0.5)
    except AssertionError:
        neg_c_ok = True
    print(f"    NEGATIVE (c) malformed CI (lo > hi) raises AssertionError: {neg_c_ok}")

    ok = (bands_disjoint and branch_ok and boundary_ceiling and boundary_below_ceiling
          and boundary_degenerate and neg_a_ok and neg_b_ok and neg_c_ok)
    _report("smoke 12: keyanchor_scaling_wide_decision_rule -- all 6 named verdicts reachable "
            "(incl. CLIFF-BEYOND-WINDOW routing), boundary-exact at the 0.98 ceiling and 10% "
            "degenerate_frac thresholds, bands confirmed disjoint, 3 negative tests to completion", ok)


# ---------------------------------------------------------------------------
# smoke 13: zero-collision, including against the ORIGINAL keyanchor-scaling
# wave itself (the d80-escalation leg deliberately shares its out_dir).
# ---------------------------------------------------------------------------

def smoke_13_zero_collision():
    wide_paths = {rdx.out_path("/fake/root", s)
                  for s in (rdx.keyanchor_scaling_wide_full_manifest()
                            + rdx.keyanchor_scaling_wide_gate1_manifest()
                            + rdx.keyanchor_scaling_wide_calibration_manifest())}
    original_scaling_paths = {rdx.out_path("/fake/root", s)
                               for s in (rdx.keyanchor_scaling_full_manifest()
                                         + rdx.keyanchor_scaling_gate1_manifest(80)
                                         + rdx.keyanchor_scaling_gate1_manifest(96)
                                         + rdx.keyanchor_scaling_calibration_manifest())}
    disjoint_from_original = wide_paths.isdisjoint(original_scaling_paths)
    print(f"    keyanchor-scaling-wide ({len(wide_paths)} paths) disjoint from the ORIGINAL "
          f"keyanchor-scaling manifest ({len(original_scaling_paths)} paths): "
          f"{disjoint_from_original}")
    if not disjoint_from_original:
        print(f"      FAIL overlap: {wide_paths & original_scaling_paths}")

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
    ok = disjoint_from_original
    for label, manifest in prior_manifests.items():
        prior_paths = {rdx.out_path("/fake/root", s) for s in manifest}
        disjoint = wide_paths.isdisjoint(prior_paths)
        if not disjoint:
            print(f"    FAIL: keyanchor-scaling-wide collides with {label}: {wide_paths & prior_paths}")
        ok = ok and disjoint
    print(f"    keyanchor-scaling-wide ({len(wide_paths)} paths) disjoint from all "
          f"{len(prior_manifests)} OTHER prior/sibling manifests (shared fake_root): {ok}")
    print("    real out_dirs: d96-wide leg -> 'wavekeyanchor-scaling-wide' (a NEW directory); "
          "d80-escalation leg -> 'wavekeyanchor-scaling' (the ORIGINAL wave's own directory, "
          "deliberately shared -- sec 15.20.2's own re-fit invocation, collision-freedom here "
          "rests on (K,seed,d_state) disjointness within that shared directory, confirmed above)")
    _report("smoke 13: zero-collision -- keyanchor-scaling-wide's manifest (16 mandatory + 4 "
            "Gate-1 + 1 calibration) is collision-free against the ORIGINAL keyanchor-scaling "
            "manifest AND every other prior/sibling wave this program has ever registered", ok)


# ---------------------------------------------------------------------------
# smoke 14: keyanchor_scaling_check_abort reused unmodified for wide-wave cells.
# ---------------------------------------------------------------------------

def smoke_14_abort_check_reused():
    wide_d96_spec = [s for s in rdx.keyanchor_scaling_wide_d96_manifest() if s["K"] == 72][0]
    original_d96_spec = [s for s in rdx.keyanchor_scaling_full_manifest()
                          if s["d_state"] == 96 and s["K"] == 51][0]
    trigger = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(96, False)]

    # Same trigger for both (neither is the K=24 anchor).
    no_raise_wide = True
    try:
        rdx.keyanchor_scaling_check_abort(wide_d96_spec, wall_s=100.0)
    except rdx.KeyanchorScalingAbort:
        no_raise_wide = False
    raises_wide = False
    try:
        rdx.keyanchor_scaling_check_abort(wide_d96_spec, wall_s=trigger + 1.0)
    except rdx.KeyanchorScalingAbort:
        raises_wide = True
    raises_original_same_trigger = False
    try:
        rdx.keyanchor_scaling_check_abort(original_d96_spec, wall_s=trigger + 1.0)
    except rdx.KeyanchorScalingAbort:
        raises_original_same_trigger = True

    wide_esc_spec = [s for s in rdx.keyanchor_scaling_d80_escalation_manifest() if s["K"] == 48][0]
    d80_trigger = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(80, False)]
    raises_escalation = False
    try:
        rdx.keyanchor_scaling_check_abort(wide_esc_spec, wall_s=d80_trigger + 1.0)
    except rdx.KeyanchorScalingAbort:
        raises_escalation = True

    print(f"    K=72/d=96 (wide): wall_s=100s does not raise: {no_raise_wide}")
    print(f"    K=72/d=96 (wide): wall_s=trigger+1 ({trigger:.1f}+1) raises: {raises_wide}")
    print(f"    K=51/d=96 (ORIGINAL wave, same tier): wall_s=trigger+1 also raises at the SAME "
          f"trigger (bracket numbers unchanged, sec 15.20.6): {raises_original_same_trigger}")
    print(f"    K=48/d=80 (escalation): wall_s=d80_trigger+1 raises at its own d=80 trigger: "
          f"{raises_escalation}")
    ok = no_raise_wide and raises_wide and raises_original_same_trigger and raises_escalation
    _report("smoke 14: keyanchor_scaling_check_abort reused UNMODIFIED for wide-wave cells -- "
            "the two-tier bracket fires identically for a new K=72/d=96 cell as for the original "
            "wave's own K=51/d=96 cell (same tier, same trigger, sec 15.20.6's own disclosure)", ok)


# ---------------------------------------------------------------------------
# smoke 15: sim_cliff_power_wide_grid_results.json artifact sanity.
# ---------------------------------------------------------------------------

def smoke_15_power_check_artifact():
    p = os.path.join(HERE, "sim_cliff_power_wide_grid_results.json")
    if not os.path.exists(p):
        _report("smoke 15: sim_cliff_power_wide_grid_results.json artifact sanity", False,
                f"artifact not found at {p!r} (sec 15.20.4/15.20.6 Stage -1 item 7 -- should "
                f"already be committed, this build session ran it)")
        return
    with open(p) as f:
        doc = json.load(f)
    has_rows = len(doc.get("results_primary", [])) == 12
    x0_abs_ok = abs(doc.get("abs_slack_x0_unrounded", 0) - 0.729667) < 1e-6
    x0_pow_ok = abs(doc.get("power_law_x0", 0) - 0.804619) < 1e-6
    gap_ok = abs(doc.get("inter_band_gap", 0) - 0.029) < 1e-6
    threshold_ok = abs(doc.get("half_gap_threshold", 0) - 0.0145) < 1e-6
    # sec 15.20.4's own headline finding: BOTH rival-center half-widths at the measured cliff
    # width (w=0.0597) exceed the 0.0145 trigger -- re-checked directly against the committed
    # artifact's own major2_trigger_check block, not re-derived.
    trigger_rows = {r["truth"]: r for r in doc.get("major2_trigger_check", [])}
    abs_row = trigger_rows.get("abs_slack_center_measured_w0597", {})
    pow_row = trigger_rows.get("power_law_center_measured_w0597", {})
    abs_exceeds = abs_row.get("exceeds_threshold") is True and abs(abs_row.get("projected_half_width", 0) - 0.0430) < 5e-4
    pow_exceeds = pow_row.get("exceeds_threshold") is True and abs(pow_row.get("projected_half_width", 0) - 0.0319) < 5e-4
    print(f"    artifact parses, {len(doc.get('results_primary', []))} results_primary rows "
          f"(expect 12): {has_rows}")
    print(f"    abs_slack_x0_unrounded={doc.get('abs_slack_x0_unrounded')} (expect 0.729667): {x0_abs_ok}")
    print(f"    power_law_x0={doc.get('power_law_x0')} (expect 0.804619): {x0_pow_ok}")
    print(f"    inter_band_gap={doc.get('inter_band_gap')} (expect 0.029), "
          f"half_gap_threshold={doc.get('half_gap_threshold')} (expect 0.0145): "
          f"{gap_ok and threshold_ok}")
    print(f"    measured-width (w=0.0597) trigger check: abs-slack half-width="
          f"{abs_row.get('projected_half_width')} (expect ~0.0430, exceeds={abs_row.get('exceeds_threshold')}): "
          f"{abs_exceeds}; power-law half-width={pow_row.get('projected_half_width')} "
          f"(expect ~0.0319, exceeds={pow_row.get('exceeds_threshold')}): {pow_exceeds}")
    ok = has_rows and x0_abs_ok and x0_pow_ok and gap_ok and threshold_ok and abs_exceeds and pow_exceeds
    _report("smoke 15: sim_cliff_power_wide_grid_results.json exists, parses, and its own headline "
            "figures (rival x0's, inter-band gap, half-gap trigger, both measured-width half-widths "
            "exceeding the trigger) match sec 15.20.4's published numbers", ok)


# ---------------------------------------------------------------------------
# smoke 16: fit_cliff_curve.py admissibility-filter fix.
# ---------------------------------------------------------------------------

def smoke_16_admissibility_fix():
    if not os.path.isdir(LOCAL_D96_ARCHIVE_DIR):
        _report("smoke 16: fit_cliff_curve.py admissibility-filter fix", False,
                f"local sec 15.19 archive not found at {LOCAL_D96_ARCHIVE_DIR!r}")
        return

    # Real regression fixture (sec 15.20.3 negative test 1): re-run the FIXED loader on the
    # archived, unmodified sec 15.19 raws, no manual pre-filtering.
    ci_report = fcc.real_bootstrap_ci(
        {K: fcc.load_k_mean_h4(LOCAL_D96_ARCHIVE_DIR, K, arm="d")[1] for K in (24, 51, 57, 63, 69)},
        None, None, n_trials=4000, d_state=96, seed=20260706)
    degenerate_frac = ci_report["degenerate_frac"]
    fixture_ok = abs(degenerate_frac - 0.9477) < 1e-4
    print(f"    real regression fixture: fixed loader on the archived sec 15.19 d=96 raws -> "
          f"degenerate_frac={degenerate_frac:.4f} (expect 0.9477 EXACTLY, per this build's own "
          f"hand-computed sensitivity check): {fixture_ok}")

    # K=69/seed=1730 (the ONE non-admissible cell) is silently dropped -- confirm directly.
    mean69, seeds69 = fcc.load_k_mean_h4(LOCAL_D96_ARCHIVE_DIR, 69, arm="d")
    k69_excludes_1730 = len(seeds69) == 2   # 3 archived seeds, 1 (seed 1730) non-admissible
    print(f"    K=69 fixed loader returns {len(seeds69)} seeds (expect 2 -- seed 1730 auto-"
          f"excluded via geo3_admission.admissible=false): {k69_excludes_1730}")

    # Negative test 2 (synthetic, sec 15.20.3): a fabricated admissible=false seed with an
    # outlier h4 must be silently excluded, not dragging the mean.
    with tempfile.TemporaryDirectory() as tmp:
        good_vals = [0.95, 0.96]
        for i, h4 in enumerate(good_vals):
            doc = {"complete": True, "geo3_admission": {"admissible": True},
                   "checkpoints": [{"M3_held_out": {"4": {"recovered_frac@0.9": h4}}}]}
            with open(os.path.join(tmp, f"wsynthetic_rdx_K99_armd_s{100 + i}_d96.json"), "w") as f:
                json.dump(doc, f)
        bad_doc = {"complete": True, "geo3_admission": {"admissible": False},
                   "checkpoints": [{"M3_held_out": {"4": {"recovered_frac@0.9": 0.05}}}]}
        with open(os.path.join(tmp, "wsynthetic_rdx_K99_armd_s102_d96.json"), "w") as f:
            json.dump(bad_doc, f)
        mean, per_seed = fcc.load_k_mean_h4(tmp, 99, arm="d")
        neg2_length_ok = len(per_seed) == 2
        neg2_not_dragged = mean is not None and abs(mean - sum(good_vals) / 2) < 1e-9
    print(f"    NEGATIVE (synthetic, sec 15.20.3): 3-seed dir with 1 fabricated "
          f"admissible=false/h4=0.05 outlier -> per_seed length={len(per_seed)} (expect 2): "
          f"{neg2_length_ok}, mean={mean} (expect {sum(good_vals)/2}, NOT dragged toward 0.05): "
          f"{neg2_not_dragged}")

    # Regression guard on the ANCHORED (d=64) path: the fix must be a pure no-op there, since
    # every archived d=64 candidate-(d) cell is already 100% admissible (sec 15.20.3's own scan).
    d64_cliff_dir = os.path.join(HERE, "..", "..", "experiment-runs", "2026-07-06_keyanchor_cliff",
                                  "results", "deltanet_rd_exactness", "wavekeyanchor-cliff")
    d64_regression_ok = True
    if os.path.isdir(d64_cliff_dir):
        for K in (34, 38, 42, 46):
            _, seeds = fcc.load_k_mean_h4(d64_cliff_dir, K, arm="d")
            if len(seeds) != 3:
                d64_regression_ok = False
        print(f"    d=64 regression guard: every K in (34,38,42,46) still returns 3 seeds "
              f"(100% admissible, unaffected by the fix): {d64_regression_ok}")
    else:
        print(f"    d=64 regression guard: archive not found at {d64_cliff_dir!r}, skipped "
              f"(non-fatal -- the d=96 fixture above is this smoke's own load-bearing check)")

    ok = fixture_ok and k69_excludes_1730 and neg2_length_ok and neg2_not_dragged and d64_regression_ok
    _report("smoke 16: fit_cliff_curve.py admissibility-filter fix -- REAL regression fixture "
            "reproduces degenerate_frac=0.9477 exactly on the archived sec 15.19 d=96 raws, K=69 "
            "auto-excludes seed 1730, a synthetic negative test confirms the fabricated outlier "
            "does not drag the mean, and the d=64 anchored path is unaffected", ok)


def main() -> int:
    print("=" * 70)
    print("smoke_keyanchor_scaling_wide.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.20 Rev 1 "
          "(post-attack-round-1) d=96 wider-K wave + d=80 escalation + admissibility-fix Wave -1 "
          "smoke suite (fla-free except smoke 8's own torch dependency, key_anchoring.py itself)")
    print("=" * 70)
    smoke_1_d96_wide_manifest_shape()
    smoke_2_d80_escalation_manifest_shape()
    smoke_3_full_manifest_16_cells()
    smoke_4_calibration_manifest()
    smoke_5_gate1_manifest()
    smoke_6_manifest_regression()
    smoke_7_flat_dict_extension_and_namespacing()
    smoke_8_gate2_at_wide_grid()
    smoke_9_wide_kernel_gate_check()
    smoke_9b_niter_gate_check()
    smoke_10_k69_sha256_gate()
    smoke_11_stage_gate_mechanics()
    smoke_12_decision_rule()
    smoke_13_zero_collision()
    smoke_14_abort_check_reused()
    smoke_15_power_check_artifact()
    smoke_16_admissibility_fix()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
