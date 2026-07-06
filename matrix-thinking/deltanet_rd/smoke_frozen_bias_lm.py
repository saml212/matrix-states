"""smoke_frozen_bias_lm.py -- FROZEN_BIAS_LM_DESIGN.md build-level smoke
suite, mirroring smoke_keyanchor_cliff.py's own house style (numbered
items, a FAILURES list, `_report()` PASS/FAIL banner, every negative test
run to completion, `main()` aggregates and returns a non-zero exit code on
any failure).

Covers (this build's own REQUIRED smoke-suite scope):
  smoke 1: hook forward/backward/grad-finite on CPU tiny model, ALL THREE
    arms (off/per_token/global) -- delegates to lm_pretrain_rd.py's own
    smoke() item [17] logic, re-run here standalone (CPU, via the fla stub)
    so this suite is independently runnable without a full lm_pretrain_rd
    --smoke invocation.
  smoke 2: B-buffer determinism from seed -- build_frozen_bias_table(seed=X)
    called twice gives torch.equal tables; different seeds give DIFFERENT
    tables (a degenerate always-same-table bug would silently defeat every
    seed-dependent claim in this design).
  smoke 3: Arm 1' retrofit equality (item 3b's own sec 8.0b check, reused
    directly from smoke_frozen_bias_wave_neg1.py -- not re-derived).
  smoke 4: manifest shape/seeds/zero-collision vs archives.
  smoke 5: budget guard refusal at over-ceiling.
  smoke 6: sec 8.2a sentinel-gate refusal without the file (+ override bypass).
  smoke 7: blind-pin writer/validator round-trip + tamper detection.

Run: python smoke_frozen_bias_lm.py
Exit code 0 = every item PASSED.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


# ---------------------------------------------------------------------------
# fla stub (SAME functionally-real stub smoke_frozen_bias_wave_neg1.py installs -- this suite
# needs real forward/backward through q/k/v_conv1d + RMSNorm, never the real Triton kernel).
# ---------------------------------------------------------------------------

def _ensure_fla_stub() -> bool:
    try:
        import fla  # noqa: F401
        return False
    except ImportError:
        pass

    class _StubShortConvolution(nn.Module):
        def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False,
                     activation: str | None = "silu"):
            super().__init__()
            self.activation = activation
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                                   padding=kernel_size - 1, bias=bias)

        def forward(self, x: torch.Tensor, cache=None):
            B, T, D = x.shape
            out = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
            if self.activation == "silu":
                out = F.silu(out)
            return out, None

    class _StubRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    def _stub_chunk_delta_rule(q, k, v, beta, initial_state=None, output_final_state=True,
                                use_qk_l2norm_in_kernel=True):
        B, T, H, Dh = q.shape
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        o = v * torch.sigmoid(beta).unsqueeze(-1)
        final_state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype, device=q.device)
        return o, final_state

    fla_mod = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_delta_rule = types.ModuleType("fla.ops.delta_rule")
    fla_modules.ShortConvolution = _StubShortConvolution
    fla_modules.RMSNorm = _StubRMSNorm
    fla_ops_delta_rule.chunk_delta_rule = _stub_chunk_delta_rule
    fla_mod.modules = fla_modules
    fla_mod.ops = fla_ops
    fla_ops.delta_rule = fla_ops_delta_rule
    sys.modules["fla"] = fla_mod
    sys.modules["fla.modules"] = fla_modules
    sys.modules["fla.ops"] = fla_ops
    sys.modules["fla.ops.delta_rule"] = fla_ops_delta_rule
    return True


_STUB_INSTALLED = _ensure_fla_stub()

from lm_pretrain_rd import (DeltaNetLM, FROZEN_BIAS_ARM_MODES,             # noqa: E402
                             build_frozen_bias_table)
import bands_pinned_frozenbias as bpf                                       # noqa: E402
import frozen_bias_lm_sweep as sweep                                        # noqa: E402
import smoke_frozen_bias_wave_neg1 as wave_neg1                             # noqa: E402

TINY_VOCAB, TINY_D_MODEL, TINY_D_STATE, TINY_N_LAYERS = 200, 32, 64, 1


# ---------------------------------------------------------------------------
# smoke 1: forward/backward/grad-finite, ALL THREE arms, standalone (re-verifies lm_pretrain_rd.py
# smoke() item [17]'s own logic here so this suite does not require a full lm_pretrain_rd --smoke
# invocation to catch a regression).
# ---------------------------------------------------------------------------

def smoke_1_forward_backward_grad_finite_all_arms():
    ok = True
    for arm in FROZEN_BIAS_ARM_MODES:
        torch.manual_seed(101)
        m = DeltaNetLM(TINY_VOCAB, d_model=TINY_D_MODEL, d_state=TINY_D_STATE, n_layers=TINY_N_LAYERS,
                       conv_size=4, frozen_bias_arm=arm, frozen_bias_lambda=0.58,
                       frozen_bias_vocab_size=TINY_VOCAB, frozen_bias_seed=42)
        x = torch.randint(0, TINY_VOCAB, (4, 128))
        logits = m(x)
        finite_fwd = torch.isfinite(logits).all().item()
        loss = F.cross_entropy(logits.reshape(-1, TINY_VOCAB), x.reshape(-1))
        loss.backward()
        finite_bwd = all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        arm_ok = finite_fwd and finite_bwd
        print(f"    arm={arm:>9s}: forward finite={finite_fwd} backward finite={finite_bwd}")
        ok = ok and arm_ok
    _report("smoke 1: forward/backward/grad-finite, all 3 arms (off/per_token/global)", ok)


# ---------------------------------------------------------------------------
# smoke 2: B-buffer determinism from seed.
# ---------------------------------------------------------------------------

def smoke_2_table_determinism_from_seed():
    t1 = build_frozen_bias_table(500, 64, seed=20260705)
    t2 = build_frozen_bias_table(500, 64, seed=20260705)
    same_seed_identical = torch.equal(t1, t2)
    t3 = build_frozen_bias_table(500, 64, seed=999)
    different_seed_different = not torch.equal(t1, t3)
    # unit-row property: every row must be unit norm (random_unit_rows_init's own contract).
    norms = t1.norm(dim=-1)
    unit_rows = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"    same seed (20260705) twice -> identical table: {same_seed_identical}")
    print(f"    different seed (999) -> different table: {different_seed_different}")
    print(f"    every row unit-norm: {unit_rows}")
    ok = same_seed_identical and different_seed_different and unit_rows
    _report("smoke 2: B-buffer determinism from seed (build_frozen_bias_table)", ok)


# ---------------------------------------------------------------------------
# smoke 3: Arm 1' retrofit equality -- reuses smoke_frozen_bias_wave_neg1.py's own sec 8.0b check
# directly (not re-derived) -- this smoke suite's OWN item 3b requirement.
# ---------------------------------------------------------------------------

def smoke_3_arm1prime_retrofit_equality():
    ok = wave_neg1.smoke_b_code_path_equality()
    _report("smoke 3: Arm 1' retrofit equality (sec 8.0b, reused from smoke_frozen_bias_wave_neg1.py)", ok)


# ---------------------------------------------------------------------------
# smoke 4: manifest shape/seeds/zero-collision vs archives.
# ---------------------------------------------------------------------------

def smoke_4_manifest_shape():
    steps = 20000
    ckpt_every = 1000
    full = sweep.rung1_manifest(steps, ckpt_every)
    core = sweep.core_manifest(steps, ckpt_every)
    mini = sweep.mini_sweep_manifest(steps, ckpt_every)
    calib = sweep.calibration_cell(steps, ckpt_every)

    count_ok = len(full) == 20
    core_count_ok = len(core) == 18
    mini_count_ok = len(mini) == 2
    arms = sorted(set(s["arm"] for s in core))
    arms_ok = arms == ["global", "off", "per_token"]
    seeds_ok = sorted(set(s["seed"] for s in full)) == [0, 1, 2]
    calib_in_core = calib["name"] in {s["name"] for s in core}
    calib_matches_lambda_primary = abs(calib["lam"] - sweep.LAMBDA_PRIMARY) < 1e-9

    # mini-sweep is per_token-only, at the registered brackets, seed 0, openr1-mix-ext only.
    mini_arms_ok = all(s["arm"] == "per_token" for s in mini)
    mini_lams_ok = sorted(s["lam"] for s in mini) == sorted(sweep.LAMBDA_MINI_SWEEP)
    mini_seed_ok = all(s["seed"] == 0 for s in mini)
    mini_corpus_ok = all(s["corpus"] == "openr1-mix-ext" for s in mini)

    # zero-collision: every cell name in the full manifest is unique (no accidental overwrite).
    names = [s["name"] for s in full]
    no_collision = len(names) == len(set(names))

    # zero-collision against archives: this program's own cell names must never match any name
    # pattern already used by run_lm_rd_trackc_sweep.py's own Wave-C archives (a DIFFERENT naming
    # prefix, "frozenbias_lm_" vs "w1_rung1"/"w1_control"/"calib_*" -- checked directly, not assumed).
    archive_prefixes = ("w1_rung1", "w1_control", "calib_rung", "calib_control", "mixcontrol")
    no_archive_collision = not any(n.startswith(p) for n in names for p in archive_prefixes)

    print(f"    total mandatory training cells: {len(full)} (expect 20)")
    print(f"    core 3-arm comparison: {len(core)} (expect 18), arms present: {arms}")
    print(f"    lambda mini-sweep: {len(mini)} (expect 2), lambdas: "
          f"{sorted(s['lam'] for s in mini)} (expect {sorted(sweep.LAMBDA_MINI_SWEEP)})")
    print(f"    seeds present across full manifest: {sorted(set(s['seed'] for s in full))}")
    print(f"    calibration cell ({calib['name']}) is one of the core 18: {calib_in_core}, "
          f"lambda={calib['lam']} (expect {sweep.LAMBDA_PRIMARY})")
    print(f"    zero name-collisions within manifest: {no_collision}")
    print(f"    zero collisions against Wave-C/Track-C archive naming prefixes: {no_archive_collision}")

    ok = (count_ok and core_count_ok and mini_count_ok and arms_ok and seeds_ok and calib_in_core
          and calib_matches_lambda_primary and mini_arms_ok and mini_lams_ok and mini_seed_ok
          and mini_corpus_ok and no_collision and no_archive_collision)
    _report("smoke 4: manifest shape (20 cells: 18 core + 2 mini-sweep), seeds {0,1,2}, "
            "zero-collision vs archives", ok)


# ---------------------------------------------------------------------------
# smoke 5: budget guard refusal at over-ceiling.
# ---------------------------------------------------------------------------

def smoke_5_budget_guard_refusal():
    # Comfortably under ceiling -- must NOT raise/exit.
    under_ok = True
    try:
        cum = sweep.budget_guard(5.0, "smoke-under", accept_override=False)
        under_ok = cum <= sweep.GPU_H_PROGRAM_CEILING
    except SystemExit:
        under_ok = False
    print(f"    projected 5.0 GPU-h (well under {sweep.GPU_H_PROGRAM_CEILING} ceiling): "
          f"proceeds without exit: {under_ok}")

    # Deliberately over ceiling -- must sys.exit(5) without override.
    over_refused = False
    try:
        sweep.budget_guard(sweep.GPU_H_PROGRAM_CEILING + 50.0, "smoke-over", accept_override=False)
    except SystemExit as e:
        over_refused = (e.code == 5)
    print(f"    projected {sweep.GPU_H_PROGRAM_CEILING + 50.0} GPU-h (over ceiling), no override: "
          f"refused (sys.exit(5)): {over_refused}")

    # Same over-ceiling projection WITH --accept-budget-override -- must proceed.
    override_ok = True
    try:
        sweep.budget_guard(sweep.GPU_H_PROGRAM_CEILING + 50.0, "smoke-over-override", accept_override=True)
    except SystemExit:
        override_ok = False
    print(f"    SAME over-ceiling projection WITH accept_override=True: proceeds: {override_ok}")

    ok = under_ok and over_refused and override_ok
    _report("smoke 5: budget guard refusal at over-ceiling (+ override bypass)", ok)


# ---------------------------------------------------------------------------
# smoke 6: sec 8.2a sentinel-gate refusal without the file.
# ---------------------------------------------------------------------------

def smoke_6_contention_gate_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        missing_sentinel = os.path.join(tmp, "STAGE1_RATES_OK")
        refused = False
        try:
            sweep.contention_gate(missing_sentinel, accept_override=False)
        except SystemExit as e:
            refused = (e.code == 4)
        print(f"    missing sentinel, no override: refused (sys.exit(4)): {refused}")

        # override bypass -- must proceed (loud warning printed, verified via stdout capture is
        # unnecessary here; behavioral check is "does not raise").
        override_ok = True
        try:
            sweep.contention_gate(missing_sentinel, accept_override=True)
        except SystemExit:
            override_ok = False
        print(f"    missing sentinel, WITH --accept-contention-override: proceeds: {override_ok}")

        # sentinel present -- must proceed without needing the override.
        with open(missing_sentinel, "w") as f:
            f.write('{"stage1_gpuh": 5.0}\n')
        present_ok = True
        try:
            sweep.contention_gate(missing_sentinel, accept_override=False)
        except SystemExit:
            present_ok = False
        print(f"    sentinel PRESENT, no override needed: proceeds: {present_ok}")

    ok = refused and override_ok and present_ok
    _report("smoke 6: sec 8.2a contention-gate refusal without the K-anchoring Stage-1 sentinel "
            "(+ override bypass, + proceeds when present)", ok)


# ---------------------------------------------------------------------------
# smoke 7: blind-pin writer/validator round-trip + tamper detection.
# ---------------------------------------------------------------------------

def smoke_7_blind_pin_round_trip_and_tamper():
    with tempfile.TemporaryDirectory() as tmp:
        # Build 3 tiny "result JSON" files per group to hash (content is irrelevant to the pin
        # itself -- only their BYTES are hashed; the actual span_frac/val_loss VALUES are passed
        # separately, mirroring the real writer's own separation of "what got hashed" from "what
        # got derived").
        def _write_fake_results(prefix, n=3):
            paths = []
            for i in range(n):
                p = os.path.join(tmp, f"{prefix}_s{i}.json")
                with open(p, "w") as f:
                    json.dump({"seed": i, "fake": True, "tag": prefix}, f)
                paths.append(p)
            return paths

        arm1_paths = {"openr1-mix-ext": _write_fake_results("arm1_openr1"),
                      "wikitext-mix-ext": _write_fake_results("arm1_wikitext")}
        arm1prime_paths = {"openr1-mix-ext": _write_fake_results("arm1p_openr1"),
                            "wikitext-mix-ext": _write_fake_results("arm1p_wikitext")}
        arm1double_paths = {"openr1-mix-ext": _write_fake_results("arm1d_openr1"),
                             "wikitext-mix-ext": _write_fake_results("arm1d_wikitext")}
        arm1_kraw_paths = {"openr1-mix-ext": _write_fake_results("arm1k_openr1"),
                            "wikitext-mix-ext": _write_fake_results("arm1k_wikitext")}

        val_loss = {"openr1-mix-ext": [3.1, 3.15, 3.05], "wikitext-mix-ext": [3.4, 3.35, 3.45]}
        arm1prime_sf = {"openr1-mix-ext": [0.24, 0.26, 0.22], "wikitext-mix-ext": [0.30, 0.28, 0.32]}
        arm1double_sf = {"openr1-mix-ext": [0.65, 0.63, 0.67], "wikitext-mix-ext": [0.70, 0.68, 0.72]}
        arm1_kraw_sf = {"openr1-mix-ext": [0.22, 0.24, 0.20], "wikitext-mix-ext": [0.27, 0.25, 0.29]}

        pin_path = os.path.join(tmp, "BANDS_PINNED-FrozenBias.json")
        doc = bpf.write_bands_pinned_frozenbias(
            pin_path, val_loss, arm1prime_sf, arm1double_sf, arm1_kraw_sf,
            arm1_paths, arm1prime_paths, arm1double_paths, arm1_kraw_paths)
        wrote_ok = os.path.exists(pin_path) and "pinned_at" in doc
        print(f"    writer produced {pin_path}: {wrote_ok}")

        # Round-trip: validate with the SAME inputs -> must succeed (doc returned, not None).
        validated = bpf.validate_bands_pinned_frozenbias(
            pin_path, val_loss, arm1prime_sf, arm1double_sf, arm1_kraw_sf)
        round_trip_ok = validated is not None
        print(f"    clean round-trip validation succeeds: {round_trip_ok}")

        # Tamper detection 1: mutate one of the referenced result JSON files on disk (bytes change)
        # -- hash mismatch must cause validation to fail.
        with open(arm1_paths["openr1-mix-ext"][0], "w") as f:
            json.dump({"seed": 0, "fake": True, "tampered": True}, f)
        tampered_file_detected = bpf.validate_bands_pinned_frozenbias(
            pin_path, val_loss, arm1prime_sf, arm1double_sf, arm1_kraw_sf) is None
        print(f"    tampered referenced result-file (hash mismatch) DETECTED (validation fails): "
              f"{tampered_file_detected}")
        # restore for the next check
        with open(arm1_paths["openr1-mix-ext"][0], "w") as f:
            json.dump({"seed": 0, "fake": True, "tag": "arm1_openr1"}, f)

        # Tamper detection 2: pass DIFFERENT span_frac values than what was pinned (files unchanged,
        # but the caller's own re-extracted numbers differ) -- content re-derivation must catch this
        # (the e633862-audit F2 fix's own load-bearing property, mirrored here).
        tampered_values = dict(arm1prime_sf)
        tampered_values["openr1-mix-ext"] = [0.99, 0.98, 0.97]   # wildly different from what was pinned
        tampered_value_detected = bpf.validate_bands_pinned_frozenbias(
            pin_path, val_loss, tampered_values, arm1double_sf, arm1_kraw_sf) is None
        print(f"    tampered PINNED VALUE (re-derivation mismatch) DETECTED (validation fails): "
              f"{tampered_value_detected}")

        # assert_blind_not_broken: pinned_at must strictly precede Arm-2/Arm-2' start times.
        import time
        blind_ok_case = True
        try:
            bpf.assert_blind_not_broken_frozenbias(doc, [doc["pinned_at"] + 10.0])
        except AssertionError:
            blind_ok_case = False
        print(f"    assert_blind_not_broken_frozenbias PASSES when Arm-2 starts AFTER pinning: "
              f"{blind_ok_case}")

        blind_broken_case = False
        try:
            bpf.assert_blind_not_broken_frozenbias(doc, [doc["pinned_at"] - 10.0])
        except AssertionError:
            blind_broken_case = True
        print(f"    assert_blind_not_broken_frozenbias RAISES when Arm-2 started BEFORE pinning "
              f"(blind broken): {blind_broken_case}")

    ok = (wrote_ok and round_trip_ok and tampered_file_detected and tampered_value_detected
          and blind_ok_case and blind_broken_case)
    _report("smoke 7: blind-pin writer/validator round-trip + tamper detection (file-hash AND "
            "pinned-value re-derivation) + assert_blind_not_broken", ok)


def main() -> int:
    print("=" * 70)
    print("smoke_frozen_bias_lm.py -- FROZEN_BIAS_LM_DESIGN.md build-level smoke suite")
    print(f"fla stub installed (real fla absent): {_STUB_INSTALLED}")
    print("=" * 70)
    smoke_1_forward_backward_grad_finite_all_arms()
    smoke_2_table_determinism_from_seed()
    smoke_3_arm1prime_retrofit_equality()
    smoke_4_manifest_shape()
    smoke_5_budget_guard_refusal()
    smoke_6_contention_gate_refusal()
    smoke_7_blind_pin_round_trip_and_tamper()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
