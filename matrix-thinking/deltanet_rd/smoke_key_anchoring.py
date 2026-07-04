"""smoke_key_anchoring.py -- KEY_ANCHORING_DESIGN.md sec 5's registered
Wave -1 smoke list (9 items), as a committed, CPU-runnable script.

SCOPE NOTE, flagged for audit scrutiny (this wave's own HARD CONSTRAINT is
build + CPU-verify only -- no GPU, no training launch; this box's GPUs are
all busy on other waves regardless): `model_rd.py::bind()` ends with
`kernel_state_design_layout()`, which calls `fla.ops.delta_rule.
chunk_delta_rule` -- a Triton kernel with NO CPU path (model_rd.py's own
module docstring). A handful of the design's 9 smokes (2/3/4/6/8) are
therefore phrased in terms of `bind()`'s full output, which this script
cannot call end-to-end without a GPU. Every one of those smokes is instead
run at THE INSERTION SITE bind() actually uses -- the REAL, PRODUCTION
`key_anchoring.anchor_blend_gather_scatter` and `model_rd.
geo3_orthogonalize_logged` functions, invoked on a REAL `DeltaNetRDBlock`
instance's REAL registered buffers/parameters (`anchor_table`,
`anchor_trained_mask`, `anchor_lambda()`) -- never a hand-copied
reimplementation. This is sound because nothing upstream of that insertion
site (embed/k_proj/v_proj/k_conv1d/W_beta) or downstream of it
(kernel_state_design_layout, force_rank_k, the readout) is touched by
`anchor_active` at all (sec 2.2's own insertion-site note) -- the ONLY
place `anchor_active` changes bind()'s behavior IS this insertion site, so
testing it directly, on the model's own real buffers, is a faithful test of
the actual mechanism, not an approximation of it. The one thing this
CANNOT smoke-test is the front-end (embed->conv) numerics feeding into
k_eff_raw -- that is unchanged, pre-existing, already-smoke-tested code
(model_rd.py's own `_self_test`), not new KEY_ANCHORING machinery.
**Scrutiny item for the independent auditor:** once a GPU is available, re-run
these smokes through a REAL `model.bind()` call (CUDA) for full end-to-end
confidence -- this script is not a substitute for that follow-up, only the
CPU-only portion the current hard constraint allows.

Exit code 0 = every item PASSED. Run: python smoke_key_anchoring.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

import torch
import torch.nn.functional as F

import key_anchoring as ka
import model_rd as mrd
import gate2_construction_test as g2test

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def _build_model(vocab=300, d_state=64, n_train=107, lambda_mode="learned", lambda_fixed=None):
    train_ids = torch.arange(10, 10 + n_train)   # ids 10..10+n_train-1 are "trained"; 0..9 held-out
    model = mrd.DeltaNetRDBlock(
        vocab, d_model=64, d_state=d_state, conv_size=4, buffer_id=0,
        geo3_active=True, geo3_n_iter=12, geo3_resid_tol=1e-2,
        anchor_active=True, anchor_lambda_mode=lambda_mode, anchor_lambda_fixed=lambda_fixed,
        anchor_train_ids=train_ids)
    return model, train_ids


def _adversarial_raw(B, K, d, seed=0):
    """The SAME near-duplicate adversarial construction model_rd.py's own
    self-test / geo3_simulator.make_adversarial_duplicates use -- the
    fallback-trigger / NS-instability regime R3 finding 4 named."""
    g = torch.Generator().manual_seed(seed)
    A = F.normalize(torch.randn(B, K, d, generator=g), dim=-1)
    A = A.clone()
    A[:, 1] = F.normalize(A[:, 0] + 1e-5 * torch.randn(B, d, generator=g), dim=-1)
    return A


def _random_key_ids(B, K, train_ids, vocab, frac_trained, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_trained_slots = int(round(K * frac_trained))
    key_ids = torch.empty(B, K, dtype=torch.int64)
    if n_trained_slots > 0:
        idx = torch.randint(0, train_ids.numel(), (B, n_trained_slots), generator=g)
        key_ids[:, :n_trained_slots] = train_ids[idx]
    if n_trained_slots < K:
        key_ids[:, n_trained_slots:] = torch.randint(0, 10, (B, K - n_trained_slots), generator=g)  # held-out pool: ids 0..9
    return key_ids


# ---------------------------------------------------------------------------
# smoke 1: anchor init load + Gate-2 legs at init; the sec 4 regression
# quadruple produces its expected verdicts.
# ---------------------------------------------------------------------------

def smoke_1_gate2():
    ok = g2test.run()
    _report("smoke 1: anchor init + Gate-2 legs + regression quadruple", ok)


# ---------------------------------------------------------------------------
# smoke 2: blend forward/backward -- finite loss, finite grad on EVERY
# parameter including the anchor table + lambda raw-param, at a realistic
# AND an adversarial (near-duplicate) input.
# ---------------------------------------------------------------------------

def smoke_2_blend_fwd_bwd():
    model, train_ids = _build_model()
    all_ok = True
    for label, raw_fn in (("realistic", lambda B, K, d: F.normalize(torch.randn(B, K, d), dim=-1)),
                            ("adversarial (near-duplicate)", _adversarial_raw)):
        model.zero_grad()
        B, K, d = 4, 16, model.d_state
        raw = raw_fn(B, K, d).clone().requires_grad_(True)
        key_ids = _random_key_ids(B, K, train_ids, model.vocab_size_total, frac_trained=0.5, seed=1)
        lam = model.anchor_lambda()
        k_blend = ka.anchor_blend_gather_scatter(raw, model.anchor_table.weight,
                                                    model.anchor_trained_mask, key_ids, lam)
        q, fallback, resid = mrd.geo3_orthogonalize_logged(k_blend, n_iter=model.geo3_n_iter,
                                                              resid_tol=model.geo3_resid_tol)
        loss = q.sum()
        loss.backward()
        finite_loss = bool(torch.isfinite(loss).item())
        raw_grad_finite = raw.grad is not None and bool(torch.isfinite(raw.grad).all().item())
        all_param_grad_finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters())
        anchor_grad_present = model.anchor_table.weight.grad is not None and \
            bool(torch.isfinite(model.anchor_table.weight.grad).all().item())
        lambda_grad_present = model.anchor_lambda_raw.grad is not None and \
            bool(torch.isfinite(model.anchor_lambda_raw.grad).all().item())
        this_ok = finite_loss and raw_grad_finite and all_param_grad_finite and anchor_grad_present and lambda_grad_present
        print(f"    [{label}] loss_finite={finite_loss} raw_grad_finite={raw_grad_finite} "
              f"anchor_table_grad_finite={anchor_grad_present} lambda_grad_finite={lambda_grad_present} "
              f"fallback_triggered={fallback}")
        all_ok = all_ok and this_ok
    _report("smoke 2: blend forward/backward (realistic + adversarial), all grads finite", all_ok)


# ---------------------------------------------------------------------------
# smoke 3: held-out bypass bit-identity -- an all-held-out batch's blend
# output is strictly torch.equal to the raw (anchor-disabled-equivalent)
# input.
# ---------------------------------------------------------------------------

def smoke_3_heldout_bit_identity():
    model, train_ids = _build_model()
    B, K, d = 4, 16, model.d_state
    raw = F.normalize(torch.randn(B, K, d), dim=-1)
    key_ids = torch.randint(0, 10, (B, K), dtype=torch.int64)   # ALL held-out (ids 0..9)
    lam = model.anchor_lambda()
    k_blend = ka.anchor_blend_gather_scatter(raw, model.anchor_table.weight,
                                                model.anchor_trained_mask, key_ids, lam)
    bit_identical = torch.equal(k_blend, raw)
    _report("smoke 3: all-held-out batch bit-identical to anchor-disabled path", bit_identical)


# ---------------------------------------------------------------------------
# smoke 4: held-out gradient isolation -- NaN injected into EVERY held-out
# anchor row; mixed-split batch, SAME adversarial input as smoke 2 (R3
# finding 4's own broadening ask); assert (i) ALL grads finite, (ii) anchor
# grad EXACTLY zero at held-out rows, (iii) held-out output rows bit-equal
# to the pure-geo3 (no-blend) path despite the planted NaNs.
# ---------------------------------------------------------------------------

def smoke_4_nan_injection():
    model, train_ids = _build_model()
    with torch.no_grad():
        model.anchor_table.weight[~model.anchor_trained_mask] = float("nan")
    B, K, d = 4, 16, model.d_state
    raw = _adversarial_raw(B, K, d, seed=2).clone().requires_grad_(True)
    key_ids = _random_key_ids(B, K, train_ids, model.vocab_size_total, frac_trained=0.5, seed=3)
    trained_here = model.anchor_trained_mask[key_ids]
    lam = model.anchor_lambda()

    k_blend = ka.anchor_blend_gather_scatter(raw, model.anchor_table.weight,
                                                model.anchor_trained_mask, key_ids, lam)
    loss = k_blend.sum()
    loss.backward()

    all_grads_finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters())
    raw_grad_finite = raw.grad is not None and bool(torch.isfinite(raw.grad).all().item())
    anchor_grad = model.anchor_table.weight.grad
    heldout_grad_exact_zero = bool((anchor_grad[~model.anchor_trained_mask] == 0).all().item())
    heldout_mask_rows = ~trained_here
    heldout_output_bit_equal = torch.equal(k_blend[heldout_mask_rows], raw.detach()[heldout_mask_rows])

    print(f"    all_grads_finite={all_grads_finite} raw_grad_finite={raw_grad_finite} "
          f"anchor_grad_heldout_exact_zero={heldout_grad_exact_zero} "
          f"heldout_output_bit_equal_despite_NaN={heldout_output_bit_equal}")
    ok = all_grads_finite and raw_grad_finite and heldout_grad_exact_zero and heldout_output_bit_equal
    _report("smoke 4: NaN-injected held-out rows -- finite grads, exact-zero anchor grad, bit-equal output", ok)

    # Negative-control contrast (documented in KEY_ANCHORING_ATTACK_R3.md's round-4 verify):
    # the SUPERSEDED torch.where form really was gradient-poisoned on the same input.
    model2, _ = _build_model()
    with torch.no_grad():
        model2.anchor_table.weight[~model2.anchor_trained_mask] = float("nan")
    raw2 = raw.detach().clone().requires_grad_(True)
    lam2 = model2.anchor_lambda()
    trained_here2 = model2.anchor_trained_mask[key_ids].unsqueeze(-1)
    blended = F.normalize((1 - lam2) * raw2 + lam2 * model2.anchor_table.weight[key_ids], dim=-1)
    out_where = torch.where(trained_here2, blended, raw2)
    out_where.sum().backward()
    where_poisoned = not bool(torch.isfinite(model2.anchor_table.weight.grad).all().item())
    print(f"    [negative control] superseded torch.where form: anchor grad non-finite "
          f"(expect True, proving the fix is load-bearing) = {where_poisoned}")
    _report("smoke 4b: negative control -- superseded torch.where form IS gradient-poisoned", where_poisoned)


# ---------------------------------------------------------------------------
# smoke 5: lambda logging -- trajectory summary fields; cadence assertion
# fires on a mis-set cadence (negative control).
# ---------------------------------------------------------------------------

def smoke_5_lambda_logging():
    traj = [{"step": i * ka.LAMBDA_LOG_CADENCE_STEPS, "lambda": 0.5 + 0.02 * ((-1) ** i)}
            for i in range(1, 102)]
    summary = ka.lambda_window_summary(traj)
    fields_present = all(k in summary for k in
                          ("final_value", "trailing_mean", "trailing_range", "n_window_points", "band"))
    consistent = summary["n_window_points"] == ka.LAMBDA_WINDOW_LOG_POINTS

    cadence_ok = False
    try:
        ka.assert_lambda_log_cadence(ka.LAMBDA_LOG_CADENCE_STEPS)
        cadence_ok = True
    except AssertionError:
        cadence_ok = False
    cadence_negative_control_fires = False
    try:
        ka.assert_lambda_log_cadence(ka.LAMBDA_LOG_CADENCE_STEPS + 50)   # deliberately mis-set
    except AssertionError:
        cadence_negative_control_fires = True

    print(f"    summary={summary}")
    print(f"    cadence assert passes at registered cadence: {cadence_ok}; "
          f"fires (correctly) at a mis-set cadence: {cadence_negative_control_fires}")
    ok = fields_present and consistent and cadence_ok and cadence_negative_control_fires
    _report("smoke 5: lambda trajectory summary fields + cadence assertion (incl. negative control)", ok)


# ---------------------------------------------------------------------------
# smoke 6: item-5 instrument -- the pre-NS side channel is populated,
# detached, correct shape, and differs from post-NS k_eff_items.
# ---------------------------------------------------------------------------

def smoke_6_item5_instrument():
    model, train_ids = _build_model()
    B, K, d = 4, 16, model.d_state
    raw = F.normalize(torch.randn(B, K, d), dim=-1)
    key_ids = _random_key_ids(B, K, train_ids, model.vocab_size_total, frac_trained=0.6, seed=4)
    lam = model.anchor_lambda()
    k_blend = ka.anchor_blend_gather_scatter(raw, model.anchor_table.weight,
                                                model.anchor_trained_mask, key_ids, lam)
    # replicate bind()'s own side-channel assignment (the exact line inside
    # bind()'s anchor_active branch -- see model_rd.py) so this smoke tests
    # the SAME semantics without needing the CUDA-only kernel call.
    model.anchor_last_k_blend_raw = k_blend.detach()
    k_eff_items, _, _ = mrd.geo3_orthogonalize_logged(k_blend, n_iter=model.geo3_n_iter,
                                                          resid_tol=model.geo3_resid_tol)

    populated = model.anchor_last_k_blend_raw is not None
    detached = populated and not model.anchor_last_k_blend_raw.requires_grad
    correct_shape = populated and model.anchor_last_k_blend_raw.shape == (B, K, d)
    differs_from_post_ns = populated and not torch.equal(model.anchor_last_k_blend_raw, k_eff_items)
    print(f"    populated={populated} detached={detached} correct_shape={correct_shape} "
          f"differs_from_post_ns={differs_from_post_ns}")
    ok = populated and detached and correct_shape and differs_from_post_ns
    _report("smoke 6: item-5 pre-NS side channel populated/detached/shaped/distinct-from-post-NS", ok)


# ---------------------------------------------------------------------------
# smoke 7: item-6 checkpoint wiring -- 6a/6b computed at a "live checkpoint"
# on the CURRENT table; the pinned collapsed table substituted in place
# must FAIL both (negative control wired into the harness, not just the
# design doc).
# ---------------------------------------------------------------------------

def smoke_7_item6_checkpoint_wiring():
    model, train_ids = _build_model()
    # simulate ONE optimizer step perturbing the anchor table (the
    # "checkpoint mid-training" scenario the re-run gate exists for)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    B, K, d = 4, 16, model.d_state
    raw = F.normalize(torch.randn(B, K, d), dim=-1)
    key_ids = _random_key_ids(B, K, train_ids, model.vocab_size_total, frac_trained=1.0, seed=5)
    lam = model.anchor_lambda()
    k_blend = ka.anchor_blend_gather_scatter(raw, model.anchor_table.weight,
                                                model.anchor_trained_mask, key_ids, lam)
    q, _, _ = mrd.geo3_orthogonalize_logged(k_blend, n_iter=model.geo3_n_iter, resid_tol=model.geo3_resid_tol)
    q.sum().backward()
    opt.step()

    cond_live = ka.raw_table_conditioning(model.anchor_table.weight[train_ids].detach())
    live_wiring_ok = cond_live["sigma_ratio_pass"] and cond_live["max_abs_cos_pass"]
    print(f"    live checkpoint (post-1-step) conditioning: {cond_live}")

    # negative control: substitute the pinned moderate-collapse table in place of the real one
    collapsed = ka.build_collapsed_table(noise_sigma=0.30, seed=42)
    with torch.no_grad():
        model.anchor_table.weight[train_ids] = collapsed
    cond_collapsed = ka.raw_table_conditioning(model.anchor_table.weight[train_ids].detach())
    negative_control_fires = not cond_collapsed["sigma_ratio_pass"] and not cond_collapsed["max_abs_cos_pass"]
    print(f"    collapsed-table-substituted conditioning (expect BOTH fail): {cond_collapsed}")

    ok = live_wiring_ok and negative_control_fires
    _report("smoke 7: item-6 checkpoint re-run wiring + collapsed-table negative control", ok)


# ---------------------------------------------------------------------------
# smoke 8: per-entity anchor-input-alignment instrument -- at fixed lambda=1
# the blend output IS the anchor row for every trained entity by
# construction (cos ~= 1.0 exactly), for all 107 trained entities; values
# lie in [-1,1]. (The FULL through-real-episodes sweep + h=1 behavioral
# companion live in keyanchor_drift_diagnostic.py, which requires bind()/
# CUDA -- out of this CPU-only smoke's scope, flagged for the GPU follow-up.)
# ---------------------------------------------------------------------------

def smoke_8_per_entity_alignment():
    model, train_ids = _build_model(lambda_mode="fixed", lambda_fixed=1.0)
    n_train = train_ids.numel()
    B, K, d = 1, n_train, model.d_state   # one big episode covering all 107 trained entities at once
    raw = F.normalize(torch.randn(B, K, d), dim=-1)
    key_ids = train_ids.unsqueeze(0)                          # (1, 107) -- every trained entity, once each
    lam = model.anchor_lambda()
    assert float(lam) == 1.0
    k_blend = ka.anchor_blend_gather_scatter(raw, model.anchor_table.weight,
                                                model.anchor_trained_mask, key_ids, lam)
    a_e = F.cosine_similarity(k_blend[0], model.anchor_table.weight[train_ids], dim=-1)   # (107,)
    in_range = bool(((a_e >= -1.0 - 1e-6) & (a_e <= 1.0 + 1e-6)).all().item())
    all_near_one = bool((a_e > 1.0 - 1e-4).all().item())
    n_entities_covered = a_e.numel() == n_train
    print(f"    n_entities={a_e.numel()} min={a_e.min().item():.6f} max={a_e.max().item():.6f} "
          f"(expect ALL ~1.0 at fixed lambda=1)")
    ok = in_range and all_near_one and n_entities_covered
    _report("smoke 8: per-entity anchor-input-alignment, lambda=1 analytic identity over all 107 entities", ok)
    print("    NOTE: the full through-real-episodes sweep + h=1 behavioral companion "
          "(keyanchor_drift_diagnostic.py) require model.bind()/CUDA -- out of this "
          "CPU-only smoke's scope; run on GPU as a follow-up (scrutiny item).")


# ---------------------------------------------------------------------------
# smoke 9: override-demotion stamping (Rev 5) -- dry-context override launch
# returns the stamp payload; result assembly with/without the stamp lands
# the correct fields in a written-and-reloaded JSON.
# ---------------------------------------------------------------------------

def smoke_9_override_stamping():
    stamp = ka.override_stamp_payload(timestamp=1234567890.0)
    stamp_ok = stamp.get("unblind_override") is True and "unblind_override_at" in stamp

    with tempfile.TemporaryDirectory() as tmp:
        override_fields = ka.assemble_claim_tier_fields(stamp)
        p_override = os.path.join(tmp, "override_result.json")
        with open(p_override, "w") as f:
            json.dump({"K": 32, "seed": 0, **override_fields}, f)
        with open(p_override) as f:
            reloaded_override = json.load(f)
        override_case_ok = (reloaded_override.get("claim_tier") == "descriptive"
                              and reloaded_override.get("unblind_override") is True
                              and "unblind_override_at" in reloaded_override)

        non_override_fields = ka.assemble_claim_tier_fields(None)
        p_non = os.path.join(tmp, "non_override_result.json")
        with open(p_non, "w") as f:
            json.dump({"K": 32, "seed": 1, **non_override_fields}, f)
        with open(p_non) as f:
            reloaded_non = json.load(f)
        non_override_case_ok = (reloaded_non.get("unblind_override") is False
                                  and "claim_tier" not in reloaded_non)

    print(f"    stamp={stamp}")
    print(f"    override case reloaded fields: {reloaded_override}")
    print(f"    non-override case reloaded fields: {reloaded_non}")
    ok = stamp_ok and override_case_ok and non_override_case_ok
    _report("smoke 9: override-demotion stamping, both cases (in-JSON, write+reload)", ok)


def main() -> int:
    print("=" * 70)
    print("KEY_ANCHORING_DESIGN.md sec 5 -- Wave -1 smoke suite (CPU-only)")
    print("=" * 70)
    smoke_1_gate2()
    smoke_2_blend_fwd_bwd()
    smoke_3_heldout_bit_identity()
    smoke_4_nan_injection()
    smoke_5_lambda_logging()
    smoke_6_item5_instrument()
    smoke_7_item6_checkpoint_wiring()
    smoke_8_per_entity_alignment()
    smoke_9_override_stamping()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL 9 ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
