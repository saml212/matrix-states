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

            cfg98m = fw.SCALES["98m"]
            with open(gate_path, "w") as f:
                json.dump({"complete": True, "steps_completed": cfg98m["steps"],
                           "frozen_bias": {"arm": "off"}, "d_model": cfg98m["d_model"],
                           "d_state": cfg98m["d_state"], "n_layers": cfg98m["n_layers"]}, f)
            check("arm_off/seed=0 out_path becomes the gate-tier path once it is complete "
                  "(config-identity fields present -- sec 13.17 l10)",
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

    # sec 13.17 l11: --with-state was a silent no-op (mutated a local copy, never wrote it back).
    proc2 = subprocess.run([sys.executable, os.path.join(fw.HERE, "fixscale_wave.py"),
                             "manifest", "--scale", "98m", "--with-state"],
                            capture_output=True, text=True, env=env, timeout=30)
    ok2 = proc2.returncode == 0
    check("manifest --scale 98m --with-state exits 0", ok2, proc2.stderr[-500:] if not ok2 else "")
    if ok2:
        cells2 = json.loads(proc2.stdout)
        check("l11 fix: every cell in --with-state output carries a 'state' key",
              all("state" in c for c in cells2), f"missing on {[c['cell_id'] for c in cells2 if 'state' not in c]}")
        check("l11 fix: state values are one of TERMINAL_STATES|'absent' (real cell_state, not a stub)",
              all(c["state"] in (fw.TERMINAL_STATES | {"absent"}) for c in cells2))


# ---------------------------------------------------------------------------
# 10. do_sweep blind-check gate (sec 13.17 F1/M1): pre-pin REFUSES retryable with NO terminal
#     marker; post-pin launches every cell. Monkeypatched run_cell (do_sweep-level, not run_cell-
#     level) -- this is the audit's own M1 ask, and it regression-tests F1 directly.
# ---------------------------------------------------------------------------

def test_do_sweep_blind_gate():
    section("[10] do_sweep blind-check gate (sec 13.17 F1/M1) -- pre-pin retryable/no-terminal-marker, post-pin launches")
    with tempfile.TemporaryDirectory() as tmp:
        orig = (fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT)
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.GATE_TIER_CALIB_ROOT = os.path.join(tmp, "calib")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        fw.PIN_ROOT = os.path.join(tmp, "pins")
        os.makedirs(fw.TRAIN_ROOT)
        os.makedirs(fw.GATE_TIER_CALIB_ROOT)
        orig_run_cell = fw.run_cell
        launched = []

        def stub_run_cell(cell, gpu, dry_run=False, **kwargs):
            st = fw.cell_state(cell)
            if st in fw.TERMINAL_STATES:
                return st
            launched.append(cell["cell_id"])
            with open(cell["out_path"], "w") as f:
                json.dump({"complete": True}, f)
            return "complete"
        fw.run_cell = stub_run_cell
        try:
            slot_cells = [c for i, c in enumerate(fw.cells_for(scale="98m", phase="post_pin")) if i % 3 == 0]

            rc1 = fw.do_sweep("98m", "post_pin", gpus=3, gpu_offset=4, slot=0, dry_run=False, stop_path=None)
            check("pre-pin sweep returns rc=1 (retryable)", rc1 == 1, f"got {rc1}")
            check("pre-pin sweep launches NOTHING", launched == [], f"launched={launched}")
            states_prepin = {c["cell_id"]: fw.cell_state(c) for c in slot_cells}
            check("F1 regression: pre-pin leaves every slot-0 cell in a NON-terminal state ('absent')",
                  all(s == "absent" for s in states_prepin.values()), f"states={states_prepin}")
            check("F1 regression: no .REFUSED marker file exists for any slot-0 cell",
                  not any(os.path.exists(fw.refused_marker_path(c["out_path"])) for c in slot_cells))

            os.makedirs(fw.PIN_ROOT, exist_ok=True)
            with open(fw.bands_pin_path("98m"), "w") as f:
                json.dump({"pinned_at": time.time() - 5}, f)

            rc2 = fw.do_sweep("98m", "post_pin", gpus=3, gpu_offset=4, slot=0, dry_run=False, stop_path=None)
            check("post-pin sweep returns rc=0", rc2 == 0, f"got {rc2}")
            expected = [c["cell_id"] for c in slot_cells]
            check("M1: post-pin sweep launches EVERY slot-0 cell", launched == expected,
                  f"expected={expected} got={launched}")
        finally:
            fw.run_cell = orig_run_cell
            fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT = orig


def test_cell_subcommand_post_pin_gate():
    section("[10b] `cell` subcommand -- same post_pin blind gate as `sweep` (sec 13.17 m5)")
    with tempfile.TemporaryDirectory() as tmp:
        orig = (fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT)
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.GATE_TIER_CALIB_ROOT = os.path.join(tmp, "calib")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        fw.PIN_ROOT = os.path.join(tmp, "pins")
        os.makedirs(fw.TRAIN_ROOT)
        os.makedirs(fw.GATE_TIER_CALIB_ROOT)
        try:
            cell = fw._cell("arm_per_token", "98m", "openr1-mix-ext", 0)
            check("m5: post_pin cell gate REFUSES (retryable) before the pin exists",
                  not fw._post_pin_blind_gate(cell, "98m"))

            os.makedirs(fw.PIN_ROOT, exist_ok=True)
            with open(fw.bands_pin_path("98m"), "w") as f:
                json.dump({"pinned_at": time.time() - 5}, f)
            check("m5: post_pin cell gate PASSES once the pin exists",
                  fw._post_pin_blind_gate(cell, "98m"))

            arm_off_cell = fw._cell("arm_off", "98m", "openr1-mix-ext", 0)
            check("(sanity) arm_off cells are phase='arm_off', never post_pin-gated",
                  arm_off_cell["phase"] == "arm_off")
        finally:
            fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT = orig


# ---------------------------------------------------------------------------
# 11. ARM_TO_FROZEN_BIAS_ARM literal + probe-cell base_cmd (sec 13.17 M2): kills a probe-arm-swap
#     mutant that the pre-fix self-referential smoke check could not.
# ---------------------------------------------------------------------------

def test_arm_to_frozen_bias_arm_literal_and_probe_cmd():
    section("[11] ARM_TO_FROZEN_BIAS_ARM independent literal + probe/off/per_token base_cmd flags (sec 13.17 M2)")
    check("ARM_TO_FROZEN_BIAS_ARM == independent literal dict (not self-referential)",
          fw.ARM_TO_FROZEN_BIAS_ARM == {"arm_off": "off", "arm_per_token": "per_token",
                                         "arm_global_probe": "global"},
          f"got {fw.ARM_TO_FROZEN_BIAS_ARM}")

    def _arm_flag(cmd):
        return cmd[cmd.index("--frozen-bias-arm") + 1] if "--frozen-bias-arm" in cmd else None

    probe_cell = fw._cell("arm_global_probe", "98m", "openr1-mix-ext", fw.SEED_PROBE)
    check("probe cell base_cmd carries --frozen-bias-arm global",
          _arm_flag(fw.base_cmd(probe_cell)) == "global", f"cmd={fw.base_cmd(probe_cell)}")

    off_cell = fw._cell("arm_off", "98m", "openr1-mix-ext", 0)
    check("arm_off cell base_cmd carries --frozen-bias-arm off",
          _arm_flag(fw.base_cmd(off_cell)) == "off")

    pt_cell = fw._cell("arm_per_token", "98m", "openr1-mix-ext", 0)
    check("arm_per_token cell base_cmd carries --frozen-bias-arm per_token",
          _arm_flag(fw.base_cmd(pt_cell)) == "per_token")


# ---------------------------------------------------------------------------
# 12. Wave -1 pre/post-blend bit-identity at d_state=128 (sec 13.17 M3) -- the shape this wave
#     newly introduces; lm_pretrain_rd.py's own smoke item [17] hardcodes d_state=64.
# ---------------------------------------------------------------------------

def test_wave_minus1_bit_identity_d128():
    section("[12] Wave -1 pre/post-blend bit-identity at d_state=128 (sec 13.17 M3, CPU-stub leg)")
    raised = False
    try:
        fw.check_off_path_bit_identity(d_state=128, device="cpu")
    except AssertionError:
        raised = True
    check("d_state=128: frozen_bias_arm='off' bit-identical to no-frozen_bias-kwargs (CPU stub)",
          not raised)
    check("also holds at d_state=64 (sanity, matches lm_pretrain_rd.py item [17]'s own shape)",
          _no_raise(lambda: fw.check_off_path_bit_identity(d_state=64, device="cpu")))


def _no_raise(fn) -> bool:
    try:
        fn()
        return True
    except AssertionError:
        return False


# ---------------------------------------------------------------------------
# 13. GPU-occupancy guard (sec 13.17 M4) -- injectable _run_fn, no real nvidia-smi/GPU required.
# ---------------------------------------------------------------------------

def test_gpu_occupancy_guard():
    section("[13] GPU-occupancy guard (sec 13.17 M4)")

    class _FakeProc:
        def __init__(self, rc, stdout):
            self.returncode = rc
            self.stdout = stdout

    busy = lambda *a, **kw: _FakeProc(0, "5000\n")
    free = lambda *a, **kw: _FakeProc(0, "100\n")
    missing = lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())

    ok, msg = fw.gpu_occupancy_ok(0, _run_fn=busy)
    check("busy GPU (5000 MiB > 2048 threshold) -> refused", not ok, msg)

    ok, msg = fw.gpu_occupancy_ok(0, _run_fn=free)
    check("free GPU (100 MiB) -> clear to launch", ok, msg)

    ok, msg = fw.gpu_occupancy_ok(0, force=True, _run_fn=busy)
    check("--force-gpu-busy bypasses a busy reading entirely", ok, msg)

    ok, msg = fw.gpu_occupancy_ok(0, _run_fn=missing)
    check("nvidia-smi absent -> fail-OPEN (permits launch, CPU/dev box)", ok, msg)

    with tempfile.TemporaryDirectory() as tmp:
        orig_train, orig_ckpt = fw.TRAIN_ROOT, fw.CKPT_ROOT
        orig_base_cmd = fw.base_cmd
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        os.makedirs(fw.TRAIN_ROOT)
        fw.base_cmd = lambda cell: [sys.executable, "-c", "print('should not run')"]
        try:
            import unittest.mock as mock
            with mock.patch.object(fw, "gpu_occupancy_ok", return_value=(False, "busy (test)")):
                cell = fw._cell("arm_off", "98m", "openr1-mix-ext", 0)
                state = fw.run_cell(cell, gpu=0, dry_run=False)
                check("run_cell REFUSES (returns 'absent', retryable) when the GPU guard fails",
                      state == "absent", f"got {state!r}")
                check("run_cell wrote NO output when the GPU guard refused (never launched)",
                      not os.path.exists(cell["out_path"]))
        finally:
            fw.TRAIN_ROOT, fw.CKPT_ROOT = orig_train, orig_ckpt
            fw.base_cmd = orig_base_cmd


# ---------------------------------------------------------------------------
# 14. F2 fix -- rate-signal gated on step>=1000: a healthy cell stuck logging only 'step 1' for
#     several watch ticks must NOT be aborted by the rate signal (repro_watchdog_false_abort.py's
#     own scenario, baked into the permanent suite so it is never only an ephemeral scratchpad proof).
# ---------------------------------------------------------------------------

def test_f2_rate_signal_step_gate():
    section("[14] F2 fix -- budget-watchdog rate signal gated on step>=1000 (sec 13.17 F2)")
    with tempfile.TemporaryDirectory() as tmp:
        orig_train, orig_ckpt = fw.TRAIN_ROOT, fw.CKPT_ROOT
        orig_interval = fw.WATCH_INTERVAL_S
        orig_scales = {k: dict(v) for k, v in fw.SCALES.items()}
        orig_base_cmd = fw.base_cmd
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        os.makedirs(fw.TRAIN_ROOT)
        fw.WATCH_INTERVAL_S = 0.3
        # tiny ref rate -- if the step>=1000 gate were REMOVED, cum_rate=elapsed/1 at the very first
        # tick would be many orders of magnitude over 1.5x this ref, false-aborting immediately.
        fw.SCALES["98m"]["ref_per_step_s"] = 0.001
        fw.SCALES["98m"]["per_cell_gpuh"] = 10.0   # wallclock ceiling unreachable -- isolates signal (a)
        trainer = ("import sys, time, json\n"
                   "print('  step      1  loss 5.0000  lr 1.0e-04', flush=True)\n"
                   "time.sleep(1.5)\n"
                   "json.dump({'complete': True}, open(sys.argv[1], 'w'))\n")
        fw.base_cmd = lambda cell: [sys.executable, "-c", trainer, cell["out_path"]]
        try:
            cell = fw._cell("arm_off", "98m", "openr1-mix-ext", 0)
            state = fw.run_cell(cell, gpu=0, dry_run=False)
            check("F2 fix: cell stuck at step=1 for multiple watch ticks completes (NOT aborted)",
                  state == "complete", f"got {state!r}")
            check("no ABORTED_BUDGET marker written", not os.path.exists(fw.aborted_budget_marker_path(cell["out_path"])))
        finally:
            fw.TRAIN_ROOT, fw.CKPT_ROOT = orig_train, orig_ckpt
            fw.WATCH_INTERVAL_S = orig_interval
            fw.SCALES = orig_scales
            fw.base_cmd = orig_base_cmd


# ---------------------------------------------------------------------------
# 15. do_sweep STOP-file check between cells (sec 13.17 m6).
# ---------------------------------------------------------------------------

def test_do_sweep_stop_between_cells():
    section("[15] do_sweep honors STOP file BETWEEN cells (sec 13.17 m6)")
    with tempfile.TemporaryDirectory() as tmp:
        orig = (fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT)
        fw.TRAIN_ROOT = os.path.join(tmp, "train")
        fw.GATE_TIER_CALIB_ROOT = os.path.join(tmp, "calib")
        fw.CKPT_ROOT = os.path.join(tmp, "ckpt")
        fw.PIN_ROOT = os.path.join(tmp, "pins")
        os.makedirs(fw.TRAIN_ROOT)
        os.makedirs(fw.GATE_TIER_CALIB_ROOT)
        orig_run_cell = fw.run_cell
        launched = []
        stop_path = os.path.join(tmp, "STOP_test")

        def stub_run_cell(cell, gpu, dry_run=False, **kwargs):
            launched.append(cell["cell_id"])
            # drop the STOP file after the FIRST cell so the loop must halt before the second
            if len(launched) == 1:
                with open(stop_path, "w") as f:
                    f.write("stop")
            with open(cell["out_path"], "w") as f:
                json.dump({"complete": True}, f)
            return "complete"
        fw.run_cell = stub_run_cell
        try:
            slot_cells = [c for i, c in enumerate(fw.cells_for(scale="98m", phase="arm_off")) if i % 3 == 0]
            check("(setup) slot-0 arm_off has >= 2 cells for this test to be meaningful",
                  len(slot_cells) >= 2, f"got {len(slot_cells)}")

            rc = fw.do_sweep("98m", "arm_off", gpus=3, gpu_offset=0, slot=0, dry_run=False,
                              stop_path=stop_path)
            check("m6: do_sweep returns non-zero when it halts on STOP", rc != 0, f"got {rc}")
            check("m6: do_sweep launched exactly ONE cell before honoring STOP (not the whole slot)",
                  len(launched) == 1, f"launched={launched}")
        finally:
            fw.run_cell = orig_run_cell
            fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT = orig


# ---------------------------------------------------------------------------
# 16. Gate-tier reuse config-identity assert (sec 13.17 l10) -- a mismatched reused JSON is REFUSED
#     (raises), not silently trusted.
# ---------------------------------------------------------------------------

def test_gate_tier_config_identity_mismatch():
    section("[16] Gate-tier reuse REFUSES a config-identity mismatch (sec 13.17 l10)")
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
            cfg = fw.SCALES["98m"]
            with open(gate_path, "w") as f:
                json.dump({"complete": True, "steps_completed": cfg["steps"],
                           "frozen_bias": {"arm": "per_token"},  # WRONG arm -- mutant
                           "d_model": cfg["d_model"], "d_state": cfg["d_state"],
                           "n_layers": cfg["n_layers"]}, f)
            raised = False
            try:
                fw.out_path("arm_off", "98m", "openr1-mix-ext", 0)
            except AssertionError:
                raised = True
            check("l10: reused JSON with frozen_bias.arm != 'off' is REFUSED (raises)", raised)

            with open(gate_path, "w") as f:
                json.dump({"complete": True, "steps_completed": cfg["steps"] - 1,  # short by 1 -- mutant
                           "frozen_bias": {"arm": "off"}, "d_model": cfg["d_model"],
                           "d_state": cfg["d_state"], "n_layers": cfg["n_layers"]}, f)
            raised2 = False
            try:
                fw.out_path("arm_off", "98m", "openr1-mix-ext", 0)
            except AssertionError:
                raised2 = True
            check("l10: reused JSON with steps_completed < required is REFUSED (raises)", raised2)

            with open(gate_path, "w") as f:
                json.dump({"complete": True, "steps_completed": cfg["steps"],
                           "frozen_bias": {"arm": "off"}, "d_model": cfg["d_model"],
                           "d_state": cfg["d_state"], "n_layers": cfg["n_layers"]}, f)
            ok_raised = False
            try:
                got = fw.out_path("arm_off", "98m", "openr1-mix-ext", 0)
            except AssertionError:
                ok_raised = True
                got = None
            check("l10: a genuinely matching reused JSON is accepted (no false-positive refusal)",
                  not ok_raised and got == gate_path)
        finally:
            fw.TRAIN_ROOT, fw.GATE_TIER_CALIB_ROOT, fw.CKPT_ROOT, fw.PIN_ROOT = orig


# ---------------------------------------------------------------------------
# 17. Atomic pin write (sec 13.17 l12) -- write_wave_pin/write_bands_pinned_frozenbias uses
#     tmp+rename; no leftover tmp file, content is exactly what was written.
# ---------------------------------------------------------------------------

def test_atomic_pin_write():
    section("[17] Atomic pin write -- tmp+rename, no leftover partial file (sec 13.17 l12)")
    from bands_pinned_frozenbias import write_bands_pinned_frozenbias
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "BANDS_PINNED-FrozenBias-TEST.json")
        per_corpus = {"openr1-mix-ext": [1.0, 1.1, 0.9], "wikitext-mix-ext": [1.2, 1.3, 1.1]}
        result_paths = {"openr1-mix-ext": [], "wikitext-mix-ext": []}
        # write_bands_pinned_frozenbias hashes referenced result paths -- pass empty per-corpus
        # path lists (this test only exercises the WRITE PATH's atomicity, not hashing) by
        # constructing the doc directly with zero-length path lists (sha256_of_file never called).
        doc = write_bands_pinned_frozenbias(
            path, per_corpus, per_corpus, per_corpus, per_corpus,
            result_paths, result_paths, result_paths, result_paths)
        check("write returns the doc AND the file exists at the final path", os.path.exists(path))
        leftover = [f for f in os.listdir(tmp) if f != os.path.basename(path)]
        check("no leftover .tmp* file after a successful write", leftover == [], f"leftover={leftover}")
        with open(path) as f:
            on_disk = json.load(f)
        check("on-disk content matches the returned doc (json round-trip identity)",
              json.dumps(on_disk, sort_keys=True) == json.dumps(json.loads(json.dumps(doc)), sort_keys=True))


# ---------------------------------------------------------------------------
# 18. run_shared_comparator_measurement -- monkeypatched-loader wiring test (sec 13.17 m8, optional).
#     Proves the PRODUCTION function's own branching/threading (not just the pure-tensor helper
#     test 3 already covers) without real checkpoint/corpus files.
# ---------------------------------------------------------------------------

def test_run_shared_comparator_measurement_wiring():
    section("[18] run_shared_comparator_measurement -- monkeypatched-loader wiring test (sec 13.17 m8, optional)")
    import torch
    import frozen_bias_retrofit_eval_rd as fbre
    import lm_pretrain_rd as lmp
    from lm_pretrain_rd import DeltaNetLM

    V, D_STATE, D_MODEL, SEED = 300, 64, 64, 12345
    torch.manual_seed(SEED)
    model = DeltaNetLM(V, d_model=D_MODEL, d_state=D_STATE, n_layers=2, conv_size=4, frozen_bias_arm="off")
    model.eval()

    orig_load_checkpoint = fbre.load_checkpoint
    orig_load_corpus = lmp.load_corpus
    try:
        fbre.load_checkpoint = lambda path, device: (model, {"run_name": "fake"})
        val_tokens = torch.randint(0, V, (4096,))
        lmp.load_corpus = lambda data_dir, corpus, device: (None, val_tokens, {}, None, None)

        result = fw.run_shared_comparator_measurement(
            checkpoint_path="/fake/ckpt.pt", corpus="openr1-mix-ext", lam=0.58,
            frozen_bias_seed=999, data_dir="/fake/data", chunk_size=16, n_windows=8,
            batch_size=4, seq_len=128, device="cpu")

        check("wiring test: forward_passes == 1 (production path also derives 3 modes from ONE pass)",
              result.get("forward_passes") == 1, f"got {result.get('forward_passes')}")
        check("wiring test: modes_derived_from_that_pass == [per_token, global, kraw]",
              result.get("modes_derived_from_that_pass") == ["per_token", "global", "kraw"])
        check("wiring test: per_mode_per_layer has all three modes",
              set(result.get("per_mode_per_layer", {}).keys()) == {"per_token", "global", "kraw"})
    finally:
        fbre.load_checkpoint = orig_load_checkpoint
        lmp.load_corpus = orig_load_corpus


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
    test_do_sweep_blind_gate()
    test_cell_subcommand_post_pin_gate()
    test_arm_to_frozen_bias_arm_literal_and_probe_cmd()
    test_wave_minus1_bit_identity_d128()
    test_gpu_occupancy_guard()
    test_f2_rate_signal_step_gate()
    test_do_sweep_stop_between_cells()
    test_gate_tier_config_identity_mismatch()
    test_atomic_pin_write()
    test_run_shared_comparator_measurement_wiring()

    section("SUMMARY")
    print(f"  PASS={RESULTS['pass']}  FAIL={RESULTS['fail']}")
    print("\n  BOX-ONLY (not covered here, see module docstring): real fla/CUDA kernels, a real "
          "lm_pretrain_rd.py training subprocess, run_shared_comparator_measurement's own "
          "load_corpus/get_batch path against real data, real peak-VRAM figures, the tmux "
          "supervisor's multi-session runtime.")
    return 0 if RESULTS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
