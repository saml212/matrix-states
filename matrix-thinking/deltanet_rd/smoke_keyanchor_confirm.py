"""smoke_keyanchor_confirm.py -- KEY_ANCHORING_DESIGN.md sec 9.5's required
follow-up (2026-07-06 keyanchor-confirm build): regression + manifest-wiring
coverage for the two documented bugs in the wave-1 verdict (sec 9.3) and the
new --wave keyanchor-confirm machinery, WITHOUT requiring fla/CUDA.

FLA-FREE BY DESIGN (unlike smoke_key_anchoring.py, which imports model_rd --
and therefore fla -- at module scope): every check here imports ONLY
key_anchoring.py and run_deltanet_rd_exactness_sweep.py, both verified
fla-free at module scope (geo3_simulator.py, which key_anchoring.py itself
imports, is "fla-free... pure torch" by its own docstring; run_deltanet_rd_
exactness_sweep.py imports only stdlib + key_anchoring + run_deltanet_rd_sweep,
which is stdlib-only). This means this script runs in the EXACT throwaway
CPU-torch-only venv KEY_ANCHORING_ATTACK_R1/R2/R3.md used for every round --
no fla, no GPU, no box required. Two checks below that would otherwise need
to IMPORT keyanchor_drift_diagnostic.py (which pulls in model_rd -> fla) are
instead done as STATIC SOURCE-TEXT checks -- reading the .py file as plain
text, never importing it -- so their coverage is available everywhere too;
the corresponding DYNAMIC (import + monkeypatch + call) versions of those
same two checks live in smoke_key_anchoring.py (already fla-required, already
wired as a box-side launch gate).

Wired as an ADDITIONAL pre-launch CPU gate for --wave keyanchor-confirm
(run_deltanet_rd_exactness_sweep.py's main(), alongside smoke_key_anchoring.py
+ gate2_construction_test.py) -- rc!=0 aborts the wave before any GPU cell
dispatches.

Exit code 0 = every item PASSED. Run: python smoke_keyanchor_confirm.py
"""
from __future__ import annotations

import os
import subprocess
import sys

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


def _read(relpath: str) -> str:
    with open(os.path.join(HERE, relpath)) as f:
        return f.read()


# ---------------------------------------------------------------------------
# smoke A: keyanchor_drift_diagnostic.py's log_every regression (sec 9.3
# item 1 -- the crash). Live check against key_anchoring.py alone (fully
# fla-free: the diagnostic's FIXED value must equal the registered cadence
# and pass the assertion; the RETIRED formula, reconstructed here rather
# than imported, must reproducibly fail it) + a static source-text check
# that the retired formula is actually gone from the diagnostic's own file.
# ---------------------------------------------------------------------------

def smoke_a_log_every_regression():
    fixed_value = ka.LAMBDA_LOG_CADENCE_STEPS   # the diagnostic's own PROBE_LOG_EVERY == this, by construction
    try:
        ka.assert_lambda_log_cadence(fixed_value)
        fixed_passes = True
    except AssertionError:
        fixed_passes = False

    default_probe_steps = 5000   # the diagnostic's own DEFAULT_PROBE_STEPS
    retired_formula_value = max(1, default_probe_steps // 5)   # == 1000, the bug
    try:
        ka.assert_lambda_log_cadence(retired_formula_value)
        retired_would_have_crashed = False
    except AssertionError:
        retired_would_have_crashed = True   # expected -- confirms the bug was real

    src = _read("keyanchor_drift_diagnostic.py")
    # Check the ACTIVE code (run_one_k's own body), not the module
    # docstring -- which deliberately quotes the retired formula verbatim
    # as historical documentation of the bug (see the module docstring's
    # own "item 1" paragraph). A whole-file substring search would false-
    # positive against that documentation.
    fn_start = src.index("def run_one_k(")
    fn_end = src.index("\ndef ", fn_start + 1)
    run_one_k_body = src[fn_start:fn_end]
    retired_formula_gone = "probe_steps // 5" not in run_one_k_body
    fixed_constant_used = "log_every=PROBE_LOG_EVERY" in run_one_k_body
    diag_matches_registered_cadence = "PROBE_LOG_EVERY = LAMBDA_LOG_CADENCE_STEPS" in src

    print(f"    fixed_value={fixed_value} passes_assertion={fixed_passes}")
    print(f"    retired_formula_value={retired_formula_value} "
          f"would_have_crashed={retired_would_have_crashed} (expect True)")
    print(f"    source: retired formula gone={retired_formula_gone} "
          f"fixed constant used at call site={fixed_constant_used} "
          f"constant pinned to registered cadence={diag_matches_registered_cadence}")
    ok = (fixed_passes and retired_would_have_crashed and retired_formula_gone
          and fixed_constant_used and diag_matches_registered_cadence)
    _report("smoke A: keyanchor_drift_diagnostic.py log_every regression "
            "(fixed value passes; retired formula reproducibly crashes; source confirms the fix)", ok)


# ---------------------------------------------------------------------------
# smoke B: the swallowed-failure bug (sec 9.3 item 2) -- generic bash proof
# that `cmd | tee log` without `pipefail` discards a failing command's exit
# code, contrasted with `set -o pipefail` fixing it, PLUS a static check
# that keyanchor_confirm_chain.sh (this build's forward-looking chain)
# actually sets pipefail, and that keyanchor_drift_diagnostic.py's main()
# is a defensive try/except wrapper (source-text check; the DYNAMIC
# monkeypatch-based version of this same check lives in
# smoke_key_anchoring.py, which requires fla to even import).
# ---------------------------------------------------------------------------

def smoke_b_pipefail_regression():
    without_pipefail = subprocess.run(
        ["bash", "-c", "false | tee /dev/null; echo $?"],
        capture_output=True, text=True, check=False).stdout.strip()
    with_pipefail = subprocess.run(
        ["bash", "-c", "set -o pipefail; false | tee /dev/null; echo $?"],
        capture_output=True, text=True, check=False).stdout.strip()
    bug_reproduced = without_pipefail == "0"
    fix_verified = with_pipefail != "0"

    chain_src = _read("keyanchor_confirm_chain.sh")
    chain_sets_pipefail = "set -euo pipefail" in chain_src or "set -o pipefail" in chain_src

    diag_src = _read("keyanchor_drift_diagnostic.py")
    has_run_main_split = "def _run():" in diag_src and "def main():" in diag_src
    main_wraps_run = ("try:" in diag_src and "_run()" in diag_src
                        and "except SystemExit:" in diag_src
                        and "except Exception:" in diag_src and "sys.exit(1)" in diag_src)

    print(f"    without 'set -o pipefail': cmd|tee $?={without_pipefail!r} (expect '0' -- the bug)")
    print(f"    with    'set -o pipefail': cmd|tee $?={with_pipefail!r} (expect non-'0' -- the fix)")
    print(f"    keyanchor_confirm_chain.sh sets pipefail: {chain_sets_pipefail}")
    print(f"    keyanchor_drift_diagnostic.py has _run()/main() split: {has_run_main_split}, "
          f"main() wraps _run() in try/except->sys.exit(1): {main_wraps_run}")
    ok = bug_reproduced and fix_verified and chain_sets_pipefail and has_run_main_split and main_wraps_run
    _report("smoke B: chain-script swallowed-exit (tee/pipefail) regression "
            "+ diagnostic's defensive exit-code wrapper (static check)", ok)


# ---------------------------------------------------------------------------
# smoke C: keyanchor_wave1_manifest's drift_probe fix (sec 9.3 item on the
# manifest side: the docstring claimed instrumentation that was never
# threaded) + keyanchor_confirm_manifest's own wiring + zero is_done()/
# out_path() collision between the confirm manifest and the original
# Wave-1 / reference-arm manifests (deliverable 4, task-directed).
# ---------------------------------------------------------------------------

def smoke_c_manifest_wiring():
    kw1 = rdx.keyanchor_wave1_manifest()
    kw1_anchor_specs = [s for s in kw1 if s["arm"] in ("d", "c")]
    kw1_all_probed = len(kw1_anchor_specs) > 0 and all(s.get("drift_probe") is True for s in kw1_anchor_specs)

    kc = rdx.keyanchor_confirm_manifest()
    kc_all_probed = len(kc) > 0 and all(s.get("drift_probe") is True for s in kc)
    kc_len_ok = len(kc) in (3, 4)
    kc_all_arm_d = all(s["arm"] == "d" for s in kc)
    kc_k32_seeds = sorted(s["seed"] for s in kc if s["K"] == 32)

    ref = rdx.reference_arms_manifest()
    fake_out_dir = "/nonexistent/keyanchor_smoke_out_dir"   # out_path() is a pure string join, no I/O
    kw1_paths = {rdx.out_path(fake_out_dir, s) for s in kw1}
    ref_paths = {rdx.out_path(fake_out_dir, s) for s in ref}
    kc_paths = {rdx.out_path(fake_out_dir, s) for s in kc}
    no_collision_kw1 = kc_paths.isdisjoint(kw1_paths)
    no_collision_ref = kc_paths.isdisjoint(ref_paths)
    unique_within_kc = len(kc_paths) == len(kc)

    print(f"    keyanchor_wave1_manifest anchor-arm (d+c) specs: {len(kw1_anchor_specs)}, "
          f"all drift_probe=True: {kw1_all_probed}")
    print(f"    keyanchor_confirm_manifest: {len(kc)} specs (K=32 seeds={kc_k32_seeds}), "
          f"all arm='d': {kc_all_arm_d}, all drift_probe=True: {kc_all_probed}")
    print(f"    out_path() collisions -- vs original keyanchor: {not no_collision_kw1}, "
          f"vs reference arms: {not no_collision_ref} (expect False, False); "
          f"unique within confirm manifest: {unique_within_kc}")
    ok = (kw1_all_probed and kc_all_probed and kc_len_ok and kc_all_arm_d
          and kc_k32_seeds == [0, 1, 2] and no_collision_kw1 and no_collision_ref and unique_within_kc)
    _report("smoke C: keyanchor_wave1_manifest drift_probe fix + keyanchor_confirm_manifest "
            "wiring + zero out_path() collisions", ok)


# ---------------------------------------------------------------------------
# smoke D: is_done()'s own drift_probe identity check (this build's fix to
# run_deltanet_rd.py's exactness_config + is_done() -- deliverable 3's
# "is_done identity must NOT collide with the originals", verified
# end-to-end here rather than by filename-encoding alone). A spec with
# drift_probe=True must NOT be satisfied by a legacy result JSON that lacks
# the field (simulating an archived pre-fix file), even if every other
# field matches.
# ---------------------------------------------------------------------------

def smoke_d_is_done_drift_probe_identity():
    import json
    import tempfile

    spec_probed = rdx._spec("keyanchor-confirm", 32, 0, 20000, "d", geo3_active=True, geo3_n_iter=20,
                              geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                              drift_probe=True)
    spec_unprobed = rdx._spec("keyanchor", 32, 0, 20000, "d", geo3_active=True, geo3_n_iter=20,
                                geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                                drift_probe=False)

    def _fake_result(spec, exactness_config_overrides=None):
        ec = {"embed_source": "learned", "gram_alpha": None, "gram_rho": None, "strong_pin": False,
              "lambda_orth": 0.0, "use_zca": False, "fnce_m": None,
              "geo3_active": True, "geo3_n_iter": 20, "geo3_resid_tol": 1e-2,
              "anchor_active": True, "anchor_lambda_mode": "learned", "anchor_lambda_fixed": None,
              "lambda_anchor": 0.0}
        if exactness_config_overrides:
            ec.update(exactness_config_overrides)
        return {"K": spec["K"], "seed": spec["seed"], "steps": spec["steps"], "complete": True,
                "steps_completed": spec["steps"], "exactness_config": ec}

    with tempfile.TemporaryDirectory() as tmp:
        # Legacy-shaped JSON (drift_probe key absent entirely -- an
        # archived pre-fix file) at the PROBED spec's own path: must NOT
        # be recognized as done for a drift_probe=True spec.
        p_probed = rdx.out_path(tmp, spec_probed)
        with open(p_probed, "w") as f:
            json.dump(_fake_result(spec_probed), f)   # no "drift_probe" key -- legacy shape
        legacy_json_does_not_satisfy_probed_spec = not rdx.is_done(tmp, spec_probed)

        # A result JSON that DOES carry drift_probe=True correctly
        # satisfies the probed spec.
        with open(p_probed, "w") as f:
            json.dump(_fake_result(spec_probed, {"drift_probe": True}), f)
        real_probed_json_satisfies_probed_spec = rdx.is_done(tmp, spec_probed)

        # The unprobed spec's own legacy-shaped JSON still validates fine
        # (drift_probe defaults to False on both sides) -- no false
        # negative introduced for every pre-existing archived result.
        p_unprobed = rdx.out_path(tmp, spec_unprobed)
        with open(p_unprobed, "w") as f:
            json.dump(_fake_result(spec_unprobed), f)   # no "drift_probe" key
        legacy_json_satisfies_unprobed_spec = rdx.is_done(tmp, spec_unprobed)

    print(f"    legacy (no drift_probe key) JSON satisfies drift_probe=True spec: "
          f"{not legacy_json_does_not_satisfy_probed_spec} (expect False)")
    print(f"    real drift_probe=True JSON satisfies drift_probe=True spec: "
          f"{real_probed_json_satisfies_probed_spec} (expect True)")
    print(f"    legacy (no drift_probe key) JSON satisfies drift_probe=False spec: "
          f"{legacy_json_satisfies_unprobed_spec} (expect True -- no regression for old archives)")
    ok = (legacy_json_does_not_satisfy_probed_spec and real_probed_json_satisfies_probed_spec
          and legacy_json_satisfies_unprobed_spec)
    _report("smoke D: is_done() drift_probe field identity (not filename-encoding alone)", ok)


def main() -> int:
    print("=" * 70)
    print("smoke_keyanchor_confirm.py -- fla-free regression + manifest-wiring suite")
    print("=" * 70)
    smoke_a_log_every_regression()
    smoke_b_pipefail_regression()
    smoke_c_manifest_wiring()
    smoke_d_is_done_drift_probe_identity()
    print("=" * 70)
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
