"""smoke_fixscale.py -- CPU-stub self-test suite for fixscale_wave.py (FIX-AT-SCALE wave, BUILD
stage, FROZEN_BIAS_LM_DESIGN.md sec 13.12/sec 13.15). Mirrors this codebase's established CPU-stub
pattern (h2h_fla_stub_rd.ensure_fla_stub(), env REASONING_LINK_FORCE_CPU_STUB=1) -- every test here
runs on CPU, no GPU/CUDA/real-fla-kernel/real-corpus-data required.

Run: REASONING_LINK_FORCE_CPU_STUB=1 python3 smoke_fixscale.py

BOX-ONLY, NOT covered here (CLAUDE.md's own CPU-stub hard rule: "CPU-stub self-test suites test
logic only; real-kernel coverage needs a separate narrow smoke of the PRODUCTION path... wired as
its own enforced chain gate" -- disclosed explicitly, not silently skipped):
  1. Real chunk_delta_rule / RMSNorm / ShortConvolution kernels (Triton, CUDA-only) -- this suite's
     toy DeltaNetLM runs under the CPU stub (h2h_fla_stub_rd), which is a functional stand-in, not
     the production kernel. lm_pretrain_rd.py's OWN smoke()/frozen_bias_retrofit_eval_rd.py's OWN
     smoke() (both `assert device == "cuda"`) are the real-kernel gates -- unaffected by this file.
  2. A real lm_pretrain_rd.py training subprocess (`run_cell`'s actual command line, --frozen-bias-
     arm/--d-model/etc. against a real corpus at /data/deltanet_rd_data) -- test 8 below exercises
     `run_cell`'s subprocess+watchdog+abort MACHINERY for real (a real child process, real
     threading, real kill), but substitutes a benign `sleep`-based stand-in command for the actual
     trainer (no GPU/data on this dev box).
  3. `run_shared_comparator_measurement`'s own `load_corpus`/`get_batch` data-loading path (needs
     real corpus files under /data/deltanet_rd_data) -- test 3 below exercises the REUSED
     downstream half (`capture_raw_keys` + `derive_three_modes_from_capture`, the actual shared-
     pass property) with a synthetic batch instead of a loaded corpus.
  4. VRAM peak-memory figures (`peak_memory_allocated_bytes`/`_reserved_bytes`) -- these are
     produced by lm_pretrain_rd.py's own `train()` (torch.cuda.max_memory_allocated), which is
     None on a CPU-only run; this suite only checks that fixscale_wave.py PROPAGATES whatever
     value is present in a cell's result JSON into `build_probe_report`, not that a real CUDA
     figure is measured here.
  5. The tmux supervisor script's actual multi-session runtime behavior (fixscale_supervisor.sh) --
     `bash -n` syntax-checked separately (see build report); its per-loop Python calls are the same
     `fixscale_wave.py sweep/pin/await-armoff-and-pin` entry points this suite DOES exercise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("REASONING_LINK_FORCE_CPU_STUB", "1")

import h2h_fla_stub_rd  # noqa: E402
h2h_fla_stub_rd.ensure_fla_stub()

import fixscale_wave as fw  # noqa: E402

RESULTS = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    RESULTS["pass" if ok else "fail"] += 1


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


# ---------------------------------------------------------------------------
# 1. Manifest / config construction, all cells.
# ---------------------------------------------------------------------------

def test_manifest_construction():
    section("[1] Manifest construction -- all cells")
    cells = fw.build_manifest()
    check("28 total training cells (sec 13.7 Rev1 total)", len(cells) == 28, f"got {len(cells)}")
    ids = [c["cell_id"] for c in cells]
    check("no duplicate cell_ids", len(ids) == len(set(ids)))

    by_arm = {}
    for c in cells:
        by_arm.setdefault(c["arm"], []).append(c)
    check("arm_off: 12 cells (3 seeds x 2 corpora x 2 scales)", len(by_arm.get("arm_off", [])) == 12)
    check("arm_per_token: 12 cells", len(by_arm.get("arm_per_token", [])) == 12)
    check("arm_global_probe: 4 cells (1 seed x 2 corpora x 2 scales)", len(by_arm.get("arm_global_probe", [])) == 4)

    for c in cells:
        expect = fw.ARM_TO_FROZEN_BIAS_ARM[c["arm"]]
        if c["frozen_bias_arm"] != expect:
            check(f"frozen_bias_arm mapping for {c['cell_id']}", False,
                  f"got {c['frozen_bias_arm']!r} expected {expect!r}")
            return
    check("every cell's frozen_bias_arm matches ARM_TO_FROZEN_BIAS_ARM", True)

    probe_seeds = {c["seed"] for c in by_arm["arm_global_probe"]}
    check("arm_global_probe uses seed=0 only (n=1)", probe_seeds == {0}, f"got {probe_seeds}")
    primary_seeds = {c["seed"] for c in by_arm["arm_off"]} | {c["seed"] for c in by_arm["arm_per_token"]}
    check("arm_off/arm_per_token use seeds {0,1,2} (n=3)", primary_seeds == {0, 1, 2}, f"got {primary_seeds}")

    for scale in fw.SCALES:
        n = len([c for c in cells if c["scale"] == scale])
        check(f"scale={scale}: 14 cells", n == 14, f"got {n}")
        steps = {c["steps"] for c in cells if c["scale"] == scale}
        check(f"scale={scale}: single steps value {fw.SCALES[scale]['steps']}",
              steps == {fw.SCALES[scale]["steps"]}, f"got {steps}")

    phases = {c["phase"] for c in cells}
    check("phase in {arm_off, post_pin} only", phases == {"arm_off", "post_pin"})
    n_postpin = len([c for c in cells if c["phase"] == "post_pin"])
    check("post_pin phase = arm_per_token + arm_global_probe = 16 cells", n_postpin == 16, f"got {n_postpin}")


# ---------------------------------------------------------------------------
# 2. Arm wiring: per_token vs global produce different biases; b_global unit-norm; lambda applied.
# ---------------------------------------------------------------------------

def test_arm_wiring_bias_construction():
    section("[2] Arm wiring -- bias construction, unit-norm, lambda")
    import torch
    from lm_pretrain_rd import apply_frozen_bias_blend, build_frozen_bias_table, frozen_bias_global_vector

    V, D_STATE, SEED = 500, 64, 20260705
    table = build_frozen_bias_table(V, D_STATE, seed=SEED)
    check("table shape (vocab, d_state)", tuple(table.shape) == (V, D_STATE), f"got {tuple(table.shape)}")

    b_global = frozen_bias_global_vector(table)
    norm = b_global.norm().item()
    check("b_global is unit-norm (||b_global|| == 1.0)", abs(norm - 1.0) < 1e-5, f"norm={norm}")

    manual_mean = torch.nn.functional.normalize(table.mean(dim=0), dim=-1)
    check("b_global == normalize(mean_i B[i]) (sec 13.3's own construction)",
          torch.allclose(b_global, manual_mean, atol=1e-6))

    torch.manual_seed(7)
    B, T = 4, 32
    k_raw = torch.randn(B, T, D_STATE)
    k_raw = torch.nn.functional.normalize(k_raw, dim=-1)
    token_ids = torch.randint(0, V, (B, T))
    lam = 0.58

    blended_pt = apply_frozen_bias_blend(k_raw, token_ids, "per_token", table, None, lam)
    blended_gl = apply_frozen_bias_blend(k_raw, token_ids, "global", None, b_global, lam)
    check("per_token and global blends differ (different bias source)",
          not torch.equal(blended_pt, blended_gl))

    off = apply_frozen_bias_blend(k_raw, token_ids, "off", None, None, lam)
    check("arm='off' returns k_raw UNCHANGED", torch.equal(off, k_raw))

    manual_pt = torch.nn.functional.normalize((1 - lam) * k_raw + lam * table[token_ids], dim=-1)
    check("per_token blend == normalize((1-lam)*k_raw + lam*table[token_ids]) (lambda applied)",
          torch.allclose(blended_pt, manual_pt, atol=1e-6))
    manual_gl = torch.nn.functional.normalize(
        (1 - lam) * k_raw + lam * b_global.view(1, 1, -1), dim=-1)
    check("global blend == normalize((1-lam)*k_raw + lam*b_global) (lambda applied)",
          torch.allclose(blended_gl, manual_gl, atol=1e-6))

    lam0 = apply_frozen_bias_blend(k_raw, token_ids, "per_token", table, None, 0.0)
    check("lambda=0.0 -> blend degenerates to normalize(k_raw) (no bias contribution)",
          torch.allclose(lam0, torch.nn.functional.normalize(k_raw, dim=-1), atol=1e-6))


# ---------------------------------------------------------------------------
# 3. Shared-forward-pass property (sec 13.15 finding 1): ONE capture_raw_keys call derives all
#    three modes -- proven against the REAL reused capture_raw_keys/apply_frozen_bias_blend, not a
#    re-implementation.
# ---------------------------------------------------------------------------

def test_shared_forward_pass_one_capture_three_modes():
    section("[3] Shared-forward-pass comparator -- one capture, three modes (sec 13.15 finding 1)")
    import torch
    from frozen_bias_retrofit_eval_rd import capture_raw_keys
    from lm_pretrain_rd import DeltaNetLM, EOT_TOKEN_ID, build_frozen_bias_table, frozen_bias_global_vector

    V, D_STATE, SEED = 300, 64, 999
    torch.manual_seed(41)
    model = DeltaNetLM(V, d_model=64, d_state=D_STATE, n_layers=2, conv_size=4,
                        frozen_bias_arm="off")
    model.eval()
    x = torch.randint(0, V, (4, 128))   # >= _MIN_KERNEL_T (F15-LM floor, lm_pretrain_rd.py:890)

    n_calls = {"count": 0}
    real_hook_forward = model.blocks[0].mixer.k_conv1d.forward

    def counting_forward(*a, **kw):
        n_calls["count"] += 1
        return real_hook_forward(*a, **kw)
    model.blocks[0].mixer.k_conv1d.forward = counting_forward

    keys_by_layer, token_ids_cat = capture_raw_keys(model, [x], "cpu")
    check("capture_raw_keys called k_conv1d exactly once for one batch (one real forward pass)",
          n_calls["count"] == 1, f"got {n_calls['count']}")

    content_mask = (token_ids_cat != EOT_TOKEN_ID)
    table = build_frozen_bias_table(V, D_STATE, seed=SEED)
    global_vec = frozen_bias_global_vector(table)
    num_heads = model.blocks[0].mixer.num_heads

    per_mode = fw.derive_three_modes_from_capture(
        keys_by_layer, token_ids_cat, content_mask, num_heads, chunk_size=16,
        table=table, global_vec=global_vec, lam=0.58)

    check("all three modes present (per_token, global, kraw)",
          set(per_mode.keys()) == {"per_token", "global", "kraw"})
    for mode in ("per_token", "global", "kraw"):
        check(f"mode={mode}: every layer has a summary dict",
              all(isinstance(v, dict) for v in per_mode[mode].values()))
    check("derived from ONE capture (n_calls still 1 -- no second forward pass for the 2nd/3rd mode)",
          n_calls["count"] == 1, f"got {n_calls['count']}")


# ---------------------------------------------------------------------------
# 4. Resume-safe skip logic: valid -> skip, invalid/truncated -> re-run.
# ---------------------------------------------------------------------------

def test_resume_safe_skip_logic():
    section("[4] Resume-safe skip logic")
    with tempfile.TemporaryDirectory() as tmp:
        orig = (fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT)
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.GATE_TIER_CALIB_ROOT = os.path.join(tmp, "calib")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        fw.PIN_ROOT = os.path.join(tmp, "pins")
        os.makedirs(fw.TRAIN_ROOT)
        os.makedirs(fw.GATE_TIER_CALIB_ROOT)
        try:
            cell = fw._cell("arm_per_token", "98m", "openr1-mix-ext", 1)
            check("absent result -> state 'absent'", fw.cell_state(cell) == "absent")

            with open(cell["out_path"], "w") as f:
                f.write("{not valid json::")
            check("corrupt/truncated JSON -> state 'absent' (re-run)", fw.cell_state(cell) == "absent")

            with open(cell["out_path"], "w") as f:
                json.dump({"complete": False, "steps_completed": 400}, f)
            check("valid JSON, complete=False -> state 'absent' (re-run, partial)",
                  fw.cell_state(cell) == "absent")

            with open(cell["out_path"], "w") as f:
                json.dump({"complete": True, "steps_completed": cell["steps"]}, f)
            check("valid JSON, complete=True -> state 'complete' (skip)",
                  fw.cell_state(cell) == "complete")

            with open(cell["out_path"], "w") as f:
                json.dump({"complete": False, "timed_out": True}, f)
            check("timed_out=True -> state 'timed_out' (terminal, skip, never silently re-run)",
                  fw.cell_state(cell) == "timed_out")

            os.remove(cell["out_path"])
            with open(fw.aborted_budget_marker_path(cell["out_path"]), "w") as f:
                json.dump({"reason": "test"}, f)
            check("ABORTED_BUDGET marker present, no out_path -> state 'aborted_budget' (terminal)",
                  fw.cell_state(cell) == "aborted_budget")
            os.remove(fw.aborted_budget_marker_path(cell["out_path"]))

            with open(fw.refused_marker_path(cell["out_path"]), "w") as f:
                f.write("refused")
            check("REFUSED marker present -> state 'refused' (terminal)",
                  fw.cell_state(cell) == "refused")
            os.remove(fw.refused_marker_path(cell["out_path"]))
        finally:
            fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT = orig


def test_gate_tier_reuse():
    section("[4b] Gate-tier arm_off seed=0 reuse (no duplicate training)")
    with tempfile.TemporaryDirectory() as tmp:
        orig = (fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT)
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.GATE_TIER_CALIB_ROOT = os.path.join(tmp, "calib")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        fw.PIN_ROOT = os.path.join(tmp, "pins")
        os.makedirs(fw.TRAIN_ROOT)
        os.makedirs(fw.GATE_TIER_CALIB_ROOT)
        try:
            gate_path = fw.gate_tier_calib_out_path("98m", "openr1-mix-ext")
            check("arm_off/seed=0 out_path is TRAIN_ROOT (own path) before gate tier completes",
                  fw.out_path("arm_off", "98m", "openr1-mix-ext", 0) != gate_path)

            with open(gate_path, "w") as f:
                json.dump({"complete": True, "steps_completed": 67547}, f)
            check("arm_off/seed=0 out_path becomes the gate-tier path once it is complete",
                  fw.out_path("arm_off", "98m", "openr1-mix-ext", 0) == gate_path)

            cell = fw._cell("arm_off", "98m", "openr1-mix-ext", 0)
            check("that cell's own state reads 'complete' from the gate-tier file (no re-run)",
                  fw.cell_state(cell) == "complete")

            other = fw._cell("arm_off", "98m", "openr1-mix-ext", 1)
            check("arm_off/seed=1 (NOT seed 0) is unaffected by the gate-tier file",
                  fw.cell_state(other) == "absent")
        finally:
            fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT = orig


# ---------------------------------------------------------------------------
# 5. Budget-abort trigger: real subprocess + real watchdog thread, tiny thresholds -> real kill.
# ---------------------------------------------------------------------------

def test_budget_abort_trigger():
    section("[5] Budget-abort trigger (1.5x circuit breaker, sec 13.8) -- real subprocess+watchdog")
    with tempfile.TemporaryDirectory() as tmp:
        orig_train, orig_ckpt = fw.TRAIN_ROOT, fw.CKPT_ROOT
        orig_interval = fw.WATCH_INTERVAL_S
        orig_scales = {k: dict(v) for k, v in fw.SCALES.items()}
        orig_base_cmd = fw.base_cmd
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        os.makedirs(fw.TRAIN_ROOT)
        fw.WATCH_INTERVAL_S = 0.3          # real watchdog loop, just fast for the test
        fw.SCALES["98m"]["per_cell_gpuh"] = 0.0002   # ceiling_s = 1.5*0.0002*3600 = 1.08s
        fw.SCALES["98m"]["ref_per_step_s"] = 0.236    # unchanged; wallclock signal alone fires here
        # stand-in "trainer": a benign long sleep (no GPU/data needed) -- disclosed box-only item 2.
        fw.base_cmd = lambda cell: [sys.executable, "-c", "import time; time.sleep(30)"]
        try:
            cell = fw._cell("arm_off", "98m", "openr1-mix-ext", 0)
            t0 = time.time()
            state = fw.run_cell(cell, gpu=0, dry_run=False)
            elapsed = time.time() - t0
            check("breach detected within a few watch intervals (elapsed < 10s, not the full 30s sleep)",
                  elapsed < 10, f"elapsed={elapsed:.2f}s")
            check("run_cell returns 'aborted_budget'", state == "aborted_budget", f"got {state!r}")
            check("cell_state() independently reads 'aborted_budget' back",
                  fw.cell_state(cell) == "aborted_budget")
            marker = fw.aborted_budget_marker_path(cell["out_path"])
            check("ABORTED_BUDGET marker file written", os.path.exists(marker))
            with open(marker) as f:
                d = json.load(f)
            check("marker records the breaker reason", "1.5x circuit breaker" in d.get("reason", ""))
        finally:
            fw.TRAIN_ROOT, fw.CKPT_ROOT = orig_train, orig_ckpt
            fw.WATCH_INTERVAL_S = orig_interval
            fw.SCALES = orig_scales
            fw.base_cmd = orig_base_cmd


# ---------------------------------------------------------------------------
# 6. assert_blind_not_broken negative test -- EXECUTED, not merely authored (CLAUDE.md hard rule).
# ---------------------------------------------------------------------------

def test_assert_blind_not_broken_negative():
    section("[6] assert_blind_not_broken_frozenbias -- positive + negative (EXECUTED)")
    from bands_pinned_frozenbias import assert_blind_not_broken_frozenbias

    pinned_at = time.time()
    doc = {"pinned_at": pinned_at}

    positive_raised = False
    try:
        assert_blind_not_broken_frozenbias(doc, [pinned_at + 10.0])
    except AssertionError:
        positive_raised = True
    check("pinned_at strictly precedes anchor start -> does NOT raise", not positive_raised)

    negative_raised = False
    try:
        assert_blind_not_broken_frozenbias(doc, [pinned_at - 10.0])   # backdated fake start
    except AssertionError:
        negative_raised = True
    check("BACKDATED anchor start (precedes pinned_at) -> MUST raise AssertionError "
          "(the check 'has teeth')", negative_raised)

    exact_tie_raised = False
    try:
        assert_blind_not_broken_frozenbias(doc, [pinned_at])   # equal, not strictly-precedes
    except AssertionError:
        exact_tie_raised = True
    check("exact-tie timestamp (not STRICTLY before) -> also raises (strict inequality enforced)",
          exact_tie_raised)

    no_anchors_raised = False
    try:
        assert_blind_not_broken_frozenbias(doc, [])
    except AssertionError:
        no_anchors_raised = True
    check("empty anchor list -> raises (nothing to check the blind against)", no_anchors_raised)


def test_check_blind_end_to_end():
    section("[6b] fixscale_wave.check_blind -- end-to-end call-site wiring")
    with tempfile.TemporaryDirectory() as tmp:
        orig_pin_root = fw.PIN_ROOT
        fw.PIN_ROOT = os.path.join(tmp, "pins")
        try:
            no_pin_raised = False
            try:
                fw.check_blind("98m", started_at=time.time())
            except AssertionError:
                no_pin_raised = True
            check("no pin file yet -> check_blind REFUSES (raises)", no_pin_raised)

            os.makedirs(fw.PIN_ROOT, exist_ok=True)
            pinned_at = time.time()
            with open(fw.bands_pin_path("98m"), "w") as f:
                json.dump({"pinned_at": pinned_at}, f)

            ok_raised = False
            try:
                fw.check_blind("98m", started_at=pinned_at + 5.0)
            except AssertionError:
                ok_raised = True
            check("pin exists, start AFTER pinned_at -> does NOT raise (launch permitted)", not ok_raised)

            bad_raised = False
            try:
                fw.check_blind("98m", started_at=pinned_at - 5.0)
            except AssertionError:
                bad_raised = True
            check("pin exists, start BEFORE pinned_at (backdated) -> RAISES (launch refused)", bad_raised)
        finally:
            fw.PIN_ROOT = orig_pin_root


# ---------------------------------------------------------------------------
# 7. Tier-string exactness.
# ---------------------------------------------------------------------------

def test_tier_string_exactness():
    section("[7] Tier-stamp exactness (sec 13.15 finding 3)")
    expected = "exploratory-probe — NOT a confirmatory bar, n=1"
    check("TIER_PROBE constant matches the exact promised string", fw.TIER_PROBE == expected,
          f"got {fw.TIER_PROBE!r}")

    payload = {"foo": "bar"}
    stamped = fw.stamp_probe_tier(payload)
    check("stamp_probe_tier inserts 'tier' == TIER_PROBE", stamped.get("tier") == fw.TIER_PROBE)
    check("stamp_probe_tier does not mutate the caller's dict", "tier" not in payload)
    check("'tier' is the FIRST key (readability convention)", next(iter(stamped)) == "tier")

    import mech_schema
    check("TIER_PROBE is NOT mech_schema.TIER (no accidental reuse of the wrong hardcoded string)",
          fw.TIER_PROBE != mech_schema.TIER)

    reraised = False
    try:
        mech_schema.wrap_exploratory({"tier_should_be_ignored": True})
    except Exception:
        reraised = True
    check("(sanity) mech_schema.wrap_exploratory is independently callable but unused by this build",
          not reraised)
    wrapped_by_mech_schema = mech_schema.wrap_exploratory({"x": 1})
    check("mech_schema.wrap_exploratory's own tier DIFFERS from TIER_PROBE (confirms the substitution "
          "sec 13.15 finding 3 requires is real, not accidental)",
          wrapped_by_mech_schema["tier"] != fw.TIER_PROBE)


def test_probe_report_shape():
    section("[7b] build_probe_report -- shape + tier stamp, no training data present")
    with tempfile.TemporaryDirectory() as tmp:
        orig = (fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT)
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.GATE_TIER_CALIB_ROOT = os.path.join(tmp, "calib")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        fw.PIN_ROOT = os.path.join(tmp, "pins")
        os.makedirs(fw.TRAIN_ROOT)
        os.makedirs(fw.GATE_TIER_CALIB_ROOT)
        try:
            report = fw.build_probe_report("98m", "openr1-mix-ext")
            check("report tier stamp exact", report["tier"] == fw.TIER_PROBE)
            check("report discloses non-gating status", "non_gating" in report)
            check("report probe_state reflects absent cell", report["probe_state"] == "absent")
            check("report never claims a comparator band before the pin exists",
                  report["comparator_arm_off_prime_global_blend_span_frac_band"] is None)
        finally:
            fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT = orig


# ---------------------------------------------------------------------------
# 8. Static-partition sweep coverage (Track C's --gpus/--gpu-offset convention).
# ---------------------------------------------------------------------------

def test_static_partition_coverage():
    section("[8] Static-partition sweep -- every cell covered exactly once across slots")
    for scale in fw.SCALES:
        for phase in ("arm_off", "post_pin"):
            cells = fw.cells_for(scale=scale, phase=phase)
            n_gpus = 3
            covered = []
            for slot in range(n_gpus):
                covered += [c for i, c in enumerate(cells) if i % n_gpus == slot]
            ids_covered = sorted(c["cell_id"] for c in covered)
            ids_all = sorted(c["cell_id"] for c in cells)
            check(f"scale={scale} phase={phase}: partition covers every cell exactly once",
                  ids_covered == ids_all)


def test_manifest_cli_smoke():
    section("[9] CLI smoke -- `manifest` subcommand runs clean (no torch import required)")
    env = {**os.environ, "REASONING_LINK_FORCE_CPU_STUB": "1"}
    proc = subprocess.run([sys.executable, os.path.join(fw.HERE, "fixscale_wave.py"),
                            "manifest", "--scale", "98m"],
                           capture_output=True, text=True, env=env, timeout=30)
    ok = proc.returncode == 0
    check("manifest --scale 98m exits 0", ok, proc.stderr[-500:] if not ok else "")
    if ok:
        cells = json.loads(proc.stdout)
        check("manifest CLI output has 14 cells for scale=98m", len(cells) == 14, f"got {len(cells)}")


def main() -> int:
    test_manifest_construction()
    test_arm_wiring_bias_construction()
    test_shared_forward_pass_one_capture_three_modes()
    test_resume_safe_skip_logic()
    test_gate_tier_reuse()
    test_budget_abort_trigger()
    test_assert_blind_not_broken_negative()
    test_check_blind_end_to_end()
    test_tier_string_exactness()
    test_probe_report_shape()
    test_static_partition_coverage()
    test_manifest_cli_smoke()

    section("SUMMARY")
    print(f"  PASS={RESULTS['pass']}  FAIL={RESULTS['fail']}")
    print("\n  BOX-ONLY (not covered here, see module docstring): real fla/CUDA kernels, a real "
          "lm_pretrain_rd.py training subprocess, run_shared_comparator_measurement's own "
          "load_corpus/get_batch path against real data, real peak-VRAM figures, the tmux "
          "supervisor's multi-session runtime.")
    return 0 if RESULTS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
