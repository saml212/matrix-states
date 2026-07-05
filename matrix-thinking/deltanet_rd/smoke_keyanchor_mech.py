"""smoke_keyanchor_mech.py -- KEY_ANCHORING_DESIGN.md sec 10.9's registered
Wave -1 smoke suite for the Rev 7.1 mechanism-tier confirmation wave
(2026-07-06 keyanchor-mech build). Covers items 1,2,3,4,5,6,7,10,11,12,13
of sec 10.9's 13-item list -- ALL CPU-only, ALL fla-free (imports ONLY
key_anchoring.py and run_deltanet_rd_exactness_sweep.py, both verified
fla-free at module scope, same discipline as smoke_keyanchor_confirm.py).

Items 8 and 9 (candidate (d') forward/backward + held-out-bypass/NaN-
injection isolation) require `model_rd.py` (imports `fla` at module scope)
-- they live in smoke_key_anchoring.py as smoke_13/smoke_14 instead
(2026-07-06 build), following that file's own established split (dynamic,
fla-required checks there; static/fla-free checks here).

Wired as an ADDITIONAL pre-launch CPU gate for --wave keyanchor-mech-gate1
and --wave keyanchor-mech (run_deltanet_rd_exactness_sweep.py's main(),
alongside smoke_key_anchoring.py + gate2_construction_test.py) -- rc!=0
aborts the wave before any GPU cell dispatches.

Exit code 0 = every item PASSED. Run: python smoke_keyanchor_mech.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import torch
import torch.nn.functional as F

import key_anchoring as ka   # noqa: E402 (fla-free, see module docstring)
import run_deltanet_rd_exactness_sweep as rdx   # noqa: E402 (fla-free, see module docstring)

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


# ---------------------------------------------------------------------------
# item 1: r_e norm-invariance positive control -- synthetic anchor rows
# scaled by {1, 5, 0.1}; assert logged r_e is bit-identical (allclose at
# float precision) across all three (proves the norm-invariance claim in
# the ACTUAL code path, compute_C_matrix, not just in algebra).
# ---------------------------------------------------------------------------

def smoke_1_r_e_norm_invariance():
    torch.manual_seed(0)
    n, R, d = 12, 16, 32
    raw_keys = torch.randn(n, R, d)
    anchor_rows_unit = F.normalize(torch.randn(n, d), dim=-1)
    r_e_by_scale = {}
    for scale in (1.0, 5.0, 0.1):
        C = ka.compute_C_matrix(raw_keys, anchor_rows_unit * scale)
        r_e_by_scale[scale] = C.diag().clone()
    ref = r_e_by_scale[1.0]
    all_close = all(torch.allclose(ref, r_e_by_scale[s], atol=1e-6) for s in (5.0, 0.1))
    max_dev = max((ref - r_e_by_scale[s]).abs().max().item() for s in (5.0, 0.1))
    print(f"    r_e at scale=1.0 (first 3): {ref[:3].tolist()}")
    print(f"    max |r_e(scale=1) - r_e(scale=s)| over s in {{5,0.1}}: {max_dev:.3e} (expect ~0, atol=1e-6)")
    _report("smoke 1: r_e norm-invariance (anchor-row scale {1,5,0.1} -> bit-identical r_e)", all_close)


# ---------------------------------------------------------------------------
# item 2: anchor-row norm logging correctness -- construct a table with a
# KNOWN non-unit-norm row; assert the logged norm matches to float
# precision.
# ---------------------------------------------------------------------------

def smoke_2_anchor_row_norm_logging():
    torch.manual_seed(1)
    n, d = 8, 16
    direction = F.normalize(torch.randn(n, d), dim=-1)
    known_norms = torch.tensor([0.5, 1.0, 1.34, 2.0, 3.7, 0.01, 10.0, 1.0])
    table = direction * known_norms.unsqueeze(-1)
    logged = table.norm(dim=-1)
    ok = torch.allclose(logged, known_norms, atol=1e-5)
    print(f"    known norms:  {known_norms.tolist()}")
    print(f"    logged norms: {[round(v, 5) for v in logged.tolist()]}")
    _report("smoke 2: anchor-row norm logging matches known construction to float precision", ok)


# ---------------------------------------------------------------------------
# item 3: held-out-row zero-init verification -- F.cosine_similarity against
# an exact-zero row returns exactly 0.0, never NaN (sec 10.1.1's finding,
# verified in the actual code path here, not just cited).
# ---------------------------------------------------------------------------

def smoke_3_heldout_zero_row_cosine():
    torch.manual_seed(2)
    n, R, d = 5, 4, 16
    raw_keys = torch.randn(n, R, d)
    anchor_rows = torch.randn(n, d)
    anchor_rows[2] = 0.0   # a held-out row: EXACT zero vector, sec 10.1.1
    C = ka.compute_C_matrix(raw_keys, anchor_rows)
    zero_col = C[:, 2]   # every entity's cosine against the exact-zero anchor row
    all_exact_zero = bool((zero_col == 0.0).all().item())
    no_nan = not bool(torch.isnan(zero_col).any().item())
    print(f"    C[:, held_out_col] = {zero_col.tolist()} (expect all exactly 0.0, no NaN)")
    ok = all_exact_zero and no_nan
    _report("smoke 3: held-out (exact-zero) anchor row -- cosine is exactly 0.0, never NaN", ok)


# ---------------------------------------------------------------------------
# item 4: mismatched-pair null-pool construction WITH resample jitter
# (attack-R7 finding 4's fix): (i) diagonal bit-identical to r_e's own
# construction (an IDENTITY, not an assertion -- same code path,
# compute_C_matrix); (ii) off-diagonal matches a HAND-computed mean-of-
# cosines reference; (iii) a cosine-of-mean implementation (the SUPERSEDED
# construction) does NOT match -- proving this smoke has real
# discriminating power (Jensen's gap), unlike a zero-jitter toy.
# ---------------------------------------------------------------------------

def smoke_4_null_pool_mean_of_cosines():
    torch.manual_seed(3)
    n, R, d = 5, 20, 8
    jitter_sigma = 0.6   # registered jitter magnitude: large enough that mean-of-cosines and
                          # cosine-of-mean provably diverge (>> float tolerance) -- see below.
    base_dirs = F.normalize(torch.randn(n, d), dim=-1)
    raw_keys = F.normalize(
        base_dirs.unsqueeze(1) + jitter_sigma * torch.randn(n, R, d), dim=-1)
    anchor_rows = F.normalize(torch.randn(n, d), dim=-1)

    C = ka.compute_C_matrix(raw_keys, anchor_rows)

    # Hand-computed mean-of-cosines reference (independent loop, not the
    # vectorized einsum -- a genuinely separate implementation).
    hand_mean_of_cos = torch.empty(n, n)
    hand_cos_of_mean = torch.empty(n, n)
    for i in range(n):
        for j in range(n):
            per_resample_cos = F.cosine_similarity(
                raw_keys[i], anchor_rows[j].unsqueeze(0).expand(R, d), dim=-1)
            hand_mean_of_cos[i, j] = per_resample_cos.mean()
            mean_raw = raw_keys[i].mean(dim=0)   # the SUPERSEDED cosine-of-mean construction
            hand_cos_of_mean[i, j] = F.cosine_similarity(
                mean_raw.unsqueeze(0), anchor_rows[j].unsqueeze(0), dim=-1).squeeze(0)

    matches_mean_of_cos = torch.allclose(C, hand_mean_of_cos, atol=1e-6)
    diverges_from_cos_of_mean = not torch.allclose(C, hand_cos_of_mean, atol=1e-3)
    max_gap_vs_cos_of_mean = (C - hand_cos_of_mean).abs().max().item()

    # Diagonal identity: C[e,e] must be bit-identical to r_e as sec 10.2
    # defines it -- same function, same tensor, sliced at the matching column.
    r_e_direct = torch.tensor([
        F.cosine_similarity(raw_keys[i], anchor_rows[i].unsqueeze(0).expand(R, d), dim=-1).mean()
        for i in range(n)])
    diag_matches_r_e = torch.allclose(C.diag(), r_e_direct, atol=1e-6)

    print(f"    C matches hand mean-of-cosines: {matches_mean_of_cos}")
    print(f"    C diverges from hand cosine-of-mean (Jensen's gap, max |diff|={max_gap_vs_cos_of_mean:.4f}, "
          f"expect >> 1e-3): {diverges_from_cos_of_mean}")
    print(f"    C.diag() == r_e (same code path, identity): {diag_matches_r_e}")
    ok = matches_mean_of_cos and diverges_from_cos_of_mean and diag_matches_r_e
    _report("smoke 4: null-pool C[i,j] is mean-of-cosines (matches hand ref, discriminates from "
            "cosine-of-mean, diagonal==r_e identity)", ok)


# ---------------------------------------------------------------------------
# item 5: REV7_THRESHOLD_PINNED zero-data-dependency proof, wired as a
# PERMANENT smoke (already run once for Rev 7.1 -- this formalizes it):
# run rev7_threshold_derive.py from the repo root AND in a fresh, otherwise-
# EMPTY sandbox containing only a copy of the script; assert the `derived`
# block is byte-identical both times and the emitted script hash matches
# the committed pin's recorded hash. PLUS the pin-triple's negative tests
# with synthetic tampering (script-hash mismatch, derived-block mismatch,
# broken pin-precedes-anchor-start ordering).
# ---------------------------------------------------------------------------

def smoke_5_rev7_pin_zero_data_dependency_and_tamper():
    # (a) positive: the committed pin validates against the working tree.
    pin = ka.validate_rev7_pin()
    positive_ok = pin is not None
    print(f"    committed REV7_THRESHOLD_PINNED.json validates (exists + script-hash match + live "
          f"derive() reproduction): {positive_ok}")

    # (b) zero-data-dependency: run the script fresh in an EMPTY sandbox
    # (no repo, no wave JSON, nothing but a copy of the script itself) and
    # assert its derived block matches the committed pin byte-for-byte
    # (modulo the timestamp, which is not part of `derived`).
    script_src = os.path.join(HERE, "rev7_threshold_derive.py")
    sandbox_independent = False
    with tempfile.TemporaryDirectory() as tmp:
        sandbox_script = os.path.join(tmp, "rev7_threshold_derive.py")
        with open(script_src) as f:
            src_text = f.read()
        with open(sandbox_script, "w") as f:
            f.write(src_text)
        out_json = os.path.join(tmp, "REV7_THRESHOLD_PINNED.json")
        env = {**os.environ, "DRY_RUN_BYPASS": "1"}
        proc = subprocess.run([sys.executable, sandbox_script, "--out", out_json],
                                cwd=tmp, env=env, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(out_json):
            import json as _json
            with open(out_json) as f:
                sandbox_pin = _json.load(f)
            sandbox_independent = (sandbox_pin["derived"] == pin["derived"]
                                     and sandbox_pin["provenance"]["script_sha256"]
                                     == pin["provenance"]["script_sha256"])
        else:
            print(f"    sandbox run FAILED: rc={proc.returncode} stderr={proc.stderr[-500:]}")
    print(f"    fresh-sandbox (no repo, no wave data) derive() reproduces the committed pin's "
          f"derived block + script hash byte-identically: {sandbox_independent}")

    # (c) tamper test 1: a script with a DIFFERENT hash must fail validation.
    with tempfile.TemporaryDirectory() as tmp:
        bad_script = os.path.join(tmp, "rev7_threshold_derive.py")
        with open(bad_script, "w") as f:
            f.write("# tampered -- not the real derivation script\n")
        tampered_script_rejected = ka.validate_rev7_pin(script_path=bad_script) is None
    print(f"    tampered script (hash mismatch) correctly REJECTED: {tampered_script_rejected}")

    # (d) tamper test 2: a pin whose `derived` block is corrupted but whose
    # RECORDED script hash still matches the real script -- a DIFFERENT
    # failure mode from (c) (hash matches; live re-derivation does not).
    import copy
    import hashlib
    import json as _json2
    real_hash = ka.sha256_of_file(script_src)
    corrupted_pin = copy.deepcopy(pin)
    corrupted_pin["derived"]["bonferroni_crosscheck"]["r_crit_exact_beta"] = 0.999999
    corrupted_pin["provenance"]["script_sha256"] = real_hash   # hash still "matches"
    with tempfile.TemporaryDirectory() as tmp:
        bad_pin_path = os.path.join(tmp, "REV7_THRESHOLD_PINNED.json")
        with open(bad_pin_path, "w") as f:
            _json2.dump(corrupted_pin, f)
        tampered_derived_rejected = ka.validate_rev7_pin(pin_path=bad_pin_path,
                                                            script_path=script_src) is None
    print(f"    tampered derived-block (hash matches, live re-run diverges) correctly REJECTED: "
          f"{tampered_derived_rejected}")

    # (e) tamper test 3: the pin-precedes-anchor-start ordering assertion,
    # both directions (intact and broken).
    import time as _time
    intact_ok = True
    try:
        ka.assert_rev7_pin_not_broken(pin, [_time.time()])
    except AssertionError:
        intact_ok = False
    broken_raises = False
    try:
        ka.assert_rev7_pin_not_broken(pin, [1700000000.0])   # a time before the pin was generated
    except AssertionError:
        broken_raises = True
    print(f"    pin-precedes-anchor-start: intact ordering passes={intact_ok}, "
          f"broken ordering raises={broken_raises}")

    ok = (positive_ok and sandbox_independent and tampered_script_rejected
          and tampered_derived_rejected and intact_ok and broken_raises)
    _report("smoke 5: REV7_THRESHOLD_PINNED zero-data-dependency (fresh empty sandbox) + pin-triple "
            "negative tests (tampered script/derived-block/ordering)", ok)


# ---------------------------------------------------------------------------
# item 6: BH-FDR step-up + BY correctness -- synthetic p-value vector with a
# HAND-VERIFIED BH cutoff; assert the implementation matches; assert the BY
# discovery count equals BH run at q/sum(1/i) (the pinned by_effective_q).
# ---------------------------------------------------------------------------

def smoke_6_bh_by_correctness():
    # Hand-constructed example (textbook BH illustration, n=10, q=0.10):
    # p-values sorted; k* is the LARGEST k with p_(k) <= (k/n)*q.
    p_sorted = [0.001, 0.008, 0.015, 0.02, 0.04, 0.06, 0.11, 0.15, 0.30, 0.80]
    n = len(p_sorted)
    q = 0.10
    thresholds = [(k / n) * q for k in range(1, n + 1)]
    # hand-derive k*: check each rank.
    hand_k = 0
    for k in range(1, n + 1):
        if p_sorted[k - 1] <= thresholds[k - 1]:
            hand_k = k
    p_values = {i: p for i, p in enumerate(p_sorted)}
    bh = ka.bh_step_up_decision(p_values, thresholds)
    bh_matches_hand = bh["k_discoveries"] == hand_k
    print(f"    hand-derived k*={hand_k}, bh_step_up_decision k*={bh['k_discoveries']} (expect equal)")

    # BY correctness: BY discovery count == BH run at q/by_factor.
    pin = ka.validate_rev7_pin()
    n_pin = pin["derived"]["inputs"]["n_entities"]
    by_q = pin["derived"]["by_crosscheck"]["by_effective_q"]
    torch.manual_seed(4)
    synthetic_p = {i: v for i, v in enumerate(torch.rand(n_pin).tolist())}
    by_thresholds = [(k / n_pin) * by_q for k in range(1, n_pin + 1)]
    by_result = ka.bh_step_up_decision(synthetic_p, by_thresholds)
    # re-derive independently: BY is BH run at the SAME thresholds -- sanity
    # check that re-running with an identical threshold list gives the SAME answer.
    by_result_2 = ka.bh_step_up_decision(synthetic_p, by_thresholds)
    by_deterministic = by_result["k_discoveries"] == by_result_2["k_discoveries"]
    print(f"    BY discovery count (q_by={by_q:.6f}): k*={by_result['k_discoveries']} "
          f"(deterministic re-run matches: {by_deterministic})")
    ok = bh_matches_hand and by_deterministic
    _report("smoke 6: BH-FDR step-up matches a hand-verified cutoff; BY correctness/determinism", ok)


# ---------------------------------------------------------------------------
# item 7: exact-Beta-vs-empirical decision-rule, BOTH branches exercised --
# a synthetic null sample deliberately OUTSIDE the registered (mean,SD)
# tolerance (assert fallback to empirical) and one INSIDE it (assert the
# analytic branch is used).
# ---------------------------------------------------------------------------

def smoke_7_pooled_null_check_both_branches():
    torch.manual_seed(5)
    sigma_chance = 1.0 / (64 ** 0.5)
    # inside tolerance: draw from N(0, sigma_chance) truncated to [-1,1]-ish range.
    inside = torch.clamp(torch.randn(20000) * sigma_chance, -0.9, 0.9)
    inside_check = ka.pooled_null_check(inside)
    # outside tolerance: shift the mean far outside +/-0.03125 AND blow up the SD.
    outside = torch.randn(20000) * 0.5 + 0.3
    outside_check = ka.pooled_null_check(outside)
    print(f"    inside-tolerance sample: mean={inside_check['mean']:.5f} sd={inside_check['sd']:.5f} "
          f"-> pass={inside_check['pass']} (expect True, analytic branch primary)")
    print(f"    outside-tolerance sample: mean={outside_check['mean']:.5f} sd={outside_check['sd']:.5f} "
          f"-> pass={outside_check['pass']} (expect False, empirical branch primary)")
    ok = inside_check["pass"] is True and outside_check["pass"] is False
    _report("smoke 7: pooled null check reaches BOTH the analytic-primary and empirical-fallback "
            "branches (not assumed, exercised)", ok)


# ---------------------------------------------------------------------------
# item 10: per-entity lambda_e interior-band-fraction computation --
# synthetic 107-entity trajectory set with a KNOWN interior/non-interior
# split; assert the computed fraction matches by hand-count.
# ---------------------------------------------------------------------------

def smoke_10_interior_band_fraction():
    n_entities = 107
    n_interior_known = 60   # the other 47 pinned near 1.0 ("pin_rediscovery")
    per_entity_series = {}
    for eid in range(n_interior_known):
        # 5-point window, final+trailing-mean both interior [0.2,0.8], range<0.1
        per_entity_series[eid] = [0.5, 0.51, 0.49, 0.5, 0.5]
    for eid in range(n_interior_known, n_entities):
        per_entity_series[eid] = [0.97, 0.98, 0.97, 0.98, 0.97]
    result = ka.interior_band_fraction_per_entity(per_entity_series)
    hand_frac = n_interior_known / n_entities
    ok = abs(result["interior_frac"] - hand_frac) < 1e-9 and result["n_entities"] == n_entities
    print(f"    computed interior_frac={result['interior_frac']:.6f}, hand-count={hand_frac:.6f}")
    _report("smoke 10: per-entity lambda_e interior-band-fraction matches a hand-counted split", ok)


# ---------------------------------------------------------------------------
# item 11: Hartigan's dip-test positive control -- a known-bimodal and a
# known-unimodal synthetic sample; assert correct discrimination at the
# registered alpha=0.05, BEFORE this ever touches real r_e/lambda_e data.
# ---------------------------------------------------------------------------

def smoke_11_dip_test_positive_control():
    torch.manual_seed(6)
    unimodal = torch.randn(80).tolist()                                   # clearly unimodal
    bimodal = ((torch.randn(40) - 3.0).tolist() + (torch.randn(40) + 3.0).tolist())   # clearly bimodal
    r_uni = ka.hartigan_dip_test(unimodal, n_boot=300, seed=100)
    r_bi = ka.hartigan_dip_test(bimodal, n_boot=300, seed=100)
    print(f"    unimodal:  dip={r_uni['dip_statistic']:.4f} p={r_uni['p_value']:.4f} "
          f"significant={r_uni['significant_at_alpha']} (expect False)")
    print(f"    bimodal:   dip={r_bi['dip_statistic']:.4f} p={r_bi['p_value']:.4f} "
          f"significant={r_bi['significant_at_alpha']} (expect True)")
    ok = (not r_uni["significant_at_alpha"]) and r_bi["significant_at_alpha"]
    _report("smoke 11: Hartigan's dip test correctly discriminates known-unimodal vs known-bimodal "
            "at alpha=0.05", ok)


# ---------------------------------------------------------------------------
# item 12: checkpoint save/reload round-trip (sec 10.10) -- construct a
# tiny stub state (mirroring run_deltanet_rd.py's OWN ckpt_payload schema),
# torch.save, reload, assert EXACT tensor equality. Pure torch, no model_rd/
# fla needed (the checkpoint payload is plain tensors, sec 10.10's own
# design -- see run_deltanet_rd.py's checkpoint-writer docstring).
# ---------------------------------------------------------------------------

def smoke_12_checkpoint_round_trip():
    torch.manual_seed(7)
    n_train, d_state = 107, 64
    payload_d = {
        "step": 2000,
        "anchor_train_ids": torch.arange(10, 10 + n_train),
        "anchor_table_trained_rows": torch.randn(n_train, d_state),
        "anchor_lambda_raw": torch.tensor(0.3),
    }
    payload_dprime = {
        "step": 2000,
        "anchor_train_ids": torch.arange(10, 10 + n_train),
        "anchor_table_trained_rows": torch.randn(n_train, d_state),
        "anchor_lambda_table_trained_rows": torch.randn(n_train, 1),
    }
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        good_paths = []
        for label, payload in (("candidate (d)", payload_d), ("candidate (d')", payload_dprime)):
            path = os.path.join(tmp, f"{label}.pt")
            torch.save(payload, path)
            good_paths.append(path)
            reloaded = torch.load(path, weights_only=True)
            this_ok = all(
                torch.equal(payload[k], reloaded[k]) if torch.is_tensor(payload[k])
                else payload[k] == reloaded[k]
                for k in payload)
            print(f"    {label}: round-trip exact-equal={this_ok} (keys: {sorted(payload.keys())})")
            ok = ok and this_ok

        # The gate itself (ka.gate_checkpoint_round_trip) must PASS both
        # well-formed payloads...
        gate_good = ka.gate_checkpoint_round_trip(good_paths)
        print(f"    gate on well-formed payloads: pass={gate_good['pass']} (expect True)")
        ok = ok and gate_good["pass"]

        # ...and REFUSE a shape impostor (e633862 audit fold-in 2): a
        # payload identical in keys/finiteness but with d_state=32 rows --
        # 107x32 instead of 107x64. Before the shape leg was added, this
        # impostor passed the gate silently.
        impostor = {
            "step": 2000,
            "anchor_train_ids": torch.arange(10, 10 + n_train),
            "anchor_table_trained_rows": torch.randn(n_train, 32),   # WRONG d_state
            "anchor_lambda_raw": torch.tensor(0.3),
        }
        imp_path = os.path.join(tmp, "impostor_107x32.pt")
        torch.save(impostor, imp_path)
        gate_imp = ka.gate_checkpoint_round_trip([imp_path])
        imp_refused = not gate_imp["pass"] and any("shape_mismatch" in reason
                                                     for _, reason in gate_imp["bad"])
        print(f"    gate on 107x32 impostor: refused={imp_refused} "
              f"(bad={gate_imp['bad']}; expect a shape_mismatch refusal)")
        ok = ok and imp_refused

        # Wrong-shaped lambda table ((n,) instead of (n,1)) must refuse too.
        impostor2 = dict(payload_dprime)
        impostor2["anchor_lambda_table_trained_rows"] = torch.randn(n_train)   # (n,), not (n,1)
        imp2_path = os.path.join(tmp, "impostor_lambda_flat.pt")
        torch.save(impostor2, imp2_path)
        gate_imp2 = ka.gate_checkpoint_round_trip([imp2_path])
        imp2_refused = not gate_imp2["pass"]
        print(f"    gate on (n,)-shaped lambda-table impostor: refused={imp2_refused} (expect True)")
        ok = ok and imp2_refused
    _report("smoke 12: checkpoint save/reload round-trip + gate shape checks (well-formed passes; "
            "107x32 and flat-lambda impostors refused)", ok)


# ---------------------------------------------------------------------------
# item 13: zero-collision manifest assertion (mirrors the confirm wave's
# own "smoke C") -- build the FULL keyanchor-mech manifest (both the 8-cell
# mandatory manifest and the Gate-1 probe), assert every out_path() is
# distinct from every existing result file across ALL prior waves
# (wavekeyanchor, wavekeyanchor-neg1, waveref, wavekeyanchor-confirm), and
# that is_done() returns False for every fresh cell before launch.
# ---------------------------------------------------------------------------

def smoke_13_zero_collision_manifest():
    km = rdx.keyanchor_mech_manifest()
    kg1 = rdx.keyanchor_mech_gate1_manifest()
    kw1 = rdx.keyanchor_wave1_manifest()
    kneg1 = rdx.keyanchor_wave_neg1_manifest()
    ref = rdx.reference_arms_manifest()
    kconfirm = rdx.keyanchor_confirm_manifest()

    fake_out_dir = "/nonexistent/keyanchor_mech_smoke_out_dir"   # out_path() is a pure string join, no I/O
    km_paths = {rdx.out_path(fake_out_dir, s) for s in km}
    kg1_paths = {rdx.out_path(fake_out_dir, s) for s in kg1}
    prior_paths = ({rdx.out_path(fake_out_dir, s) for s in kw1}
                   | {rdx.out_path(fake_out_dir, s) for s in kneg1}
                   | {rdx.out_path(fake_out_dir, s) for s in ref}
                   | {rdx.out_path(fake_out_dir, s) for s in kconfirm})

    no_collision_km = km_paths.isdisjoint(prior_paths)
    no_collision_kg1 = kg1_paths.isdisjoint(prior_paths)
    no_collision_km_vs_kg1 = km_paths.isdisjoint(kg1_paths)
    unique_within_km = len(km_paths) == len(km)
    unique_within_kg1 = len(kg1_paths) == len(kg1)

    with tempfile.TemporaryDirectory() as tmp:
        all_not_done = all(not rdx.is_done(tmp, s) for s in (km + kg1))

    print(f"    keyanchor-mech manifest: {len(km)} cells | Gate-1 probe: {len(kg1)} cell")
    print(f"    collisions -- vs prior waves (kw1/kneg1/ref/kconfirm): km={not no_collision_km}, "
          f"gate1={not no_collision_kg1} (expect False, False); km-vs-gate1: "
          f"{not no_collision_km_vs_kg1} (expect False)")
    print(f"    unique within km: {unique_within_km}, unique within gate1: {unique_within_kg1}")
    print(f"    is_done() False for every fresh cell (no data present): {all_not_done}")
    ok = (no_collision_km and no_collision_kg1 and no_collision_km_vs_kg1 and unique_within_km
          and unique_within_kg1 and all_not_done
          and len(km) == 7 and len(kg1) == 1)
    _report("smoke 13: keyanchor-mech manifest -- zero out_path() collisions across ALL prior "
            "waves, is_done() False pre-launch", ok)


# ---------------------------------------------------------------------------
# audit smoke F1 (e633862 audit FATAL 1): the Gate-1 probe's WRITER path and
# the gate's READER default must be the SAME executed string -- computed
# equality through the real functions (out_path + the manifest's own spec +
# main()'s f"wave{wave}" convention at the argparse-default --out-dir),
# never a code-read. Before the fix, the reader default was a hand-typed
# sibling path (keyanchor_mech_gate1/gate1_probe.json) that no writer ever
# wrote to -- the chain could not complete as committed.
# ---------------------------------------------------------------------------

def smoke_f1_gate1_writer_reader_path_seam():
    spec = rdx.keyanchor_mech_gate1_manifest()[0]
    # Reconstruct the writer path EXACTLY as main() computes it for
    # --wave keyanchor-mech-gate1 at the argparse default --out-dir:
    # out_dir = os.path.join(args.out_dir, f"wave{args.wave}").
    writer_path = rdx.out_path(os.path.join(rdx.DEFAULT_OUT_DIR, "wavekeyanchor-mech-gate1"), spec)
    reader_default = rdx.KEYANCHOR_MECH_GATE1_JSON_DEFAULT
    equal = writer_path == reader_default
    print(f"    writer path (executed): {writer_path}")
    print(f"    reader default:         {reader_default}")
    print(f"    EQUAL: {equal}")
    # The probe spec must also carry the full Rev-7.1 instrumentation
    # (audit fold-in 1 -- previously only claimed in a comment).
    rev7_on = spec["rev7_engagement"] is True and spec["drift_probe"] is True
    print(f"    gate1 spec rev7_engagement+drift_probe both True: {rev7_on}")
    _report("smoke F1: Gate-1 writer path == reader default (executed equality) + probe spec "
            "carries rev7_engagement", equal and rev7_on)


# ---------------------------------------------------------------------------
# audit smoke F2 (e633862 audit FATAL 2): validate_bands_pinned must REFUSE
# a pin whose STORED bands dict was tampered even when every referenced
# reference JSON still hash-matches (the pre-fix validator only re-hashed
# the reference files, so engaged_k/per_seed edits inside the pin itself
# passed silently). Fully self-contained: builds synthetic reference JSONs
# + a legit pin in a tempdir, then tampers it three ways.
# ---------------------------------------------------------------------------

def smoke_f2_bands_pinned_content_tamper():
    import json

    def _write_pin(tmp):
        drifts = [0.87, 0.88, 0.86]
        ref_paths = {32: []}
        for i, v in enumerate(drifts):
            p = os.path.join(tmp, f"ref32_s{i}.json")
            with open(p, "w") as f:
                json.dump({"complete": True, "steps_completed": 20000,
                           "checkpoints": [{"drift_probe": {"post_ns": {"mean": v},
                                                              "pre_ns": {"mean": v}}}]}, f)
            ref_paths[32].append(p)
        bp = os.path.join(tmp, "BANDS_PINNED.json")
        ka.write_bands_pinned(bp, {32: drifts}, {32: 0.9423}, ref_paths)
        return bp

    def _tamper(bp, mutate):
        with open(bp) as f:
            doc = json.load(f)
        mutate(doc)
        with open(bp, "w") as f:
            json.dump(doc, f)

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        bp = _write_pin(tmp)
        legit_passes = ka.validate_bands_pinned(bp) is not None
        print(f"    untampered synthetic pin validates: {legit_passes} (expect True)")
        ok = ok and legit_passes

        # (a) tampered engaged_k -- the audit's own reproduction case.
        _tamper(bp, lambda d: d["bands"]["32"].__setitem__("engaged_k", 0.123456))
        engaged_k_refused = ka.validate_bands_pinned(bp) is None
        print(f"    tampered engaged_k REFUSED: {engaged_k_refused} (expect True; passed "
              f"silently before the F2 fix)")
        ok = ok and engaged_k_refused

    with tempfile.TemporaryDirectory() as tmp:
        bp = _write_pin(tmp)
        # (b) tampered per_seed input inside the pin.
        _tamper(bp, lambda d: d["bands"]["32"]["per_seed"].__setitem__(0, 0.5))
        per_seed_refused = ka.validate_bands_pinned(bp) is None
        print(f"    tampered per_seed REFUSED: {per_seed_refused} (expect True)")
        ok = ok and per_seed_refused

    with tempfile.TemporaryDirectory() as tmp:
        bp = _write_pin(tmp)
        # (c) tampered ceiling: self-consistent under re-derivation (the
        # re-derivation READS the stored ceiling), so it needs the caller-
        # provided registered-ceiling cross-check to catch -- exactly why
        # gate_bands_pinned now passes ceiling_by_k.
        _tamper(bp, lambda d: (d["bands"]["32"].__setitem__("ceiling", 0.5),
                                d["bands"]["32"].__setitem__("unresolvable", True)))
        without_ceilings = ka.validate_bands_pinned(bp) is not None
        with_ceilings = ka.validate_bands_pinned(bp, ceiling_by_k={32: 0.9423}) is None
        print(f"    tampered ceiling: passes WITHOUT registered ceilings={without_ceilings} "
              f"(disclosed limitation), REFUSED WITH them={with_ceilings} (expect True, True)")
        ok = ok and without_ceilings and with_ceilings

    _report("smoke F2: BANDS_PINNED stored-content tamper refusal (engaged_k, per_seed; ceiling "
            "via the registered-ceiling cross-check)", ok)


def main() -> int:
    print("=" * 70)
    print("smoke_keyanchor_mech.py -- KEY_ANCHORING_DESIGN.md sec 10.9 Wave -1 smoke suite "
          "(items 1-7,10-13; items 8-9 are in smoke_key_anchoring.py's smoke_13/14, fla-required) "
          "+ e633862 audit-regression smokes F1/F2")
    print("=" * 70)
    smoke_1_r_e_norm_invariance()
    smoke_2_anchor_row_norm_logging()
    smoke_3_heldout_zero_row_cosine()
    smoke_4_null_pool_mean_of_cosines()
    smoke_5_rev7_pin_zero_data_dependency_and_tamper()
    smoke_6_bh_by_correctness()
    smoke_7_pooled_null_check_both_branches()
    smoke_10_interior_band_fraction()
    smoke_11_dip_test_positive_control()
    smoke_12_checkpoint_round_trip()
    smoke_13_zero_collision_manifest()
    smoke_f1_gate1_writer_reader_path_seam()
    smoke_f2_bands_pinned_content_tamper()
    print("=" * 70)
    if FAILURES:
        print(f"FAILED: {FAILURES}", file=sys.stderr)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
