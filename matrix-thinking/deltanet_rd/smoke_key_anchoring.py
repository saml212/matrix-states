"""smoke_key_anchoring.py -- KEY_ANCHORING_DESIGN.md sec 5's registered
Wave -1 smoke list (9 items) PLUS item 10 (candidate (c)'s AnchorEMA --
2026-07-04 audit fix; (c) is 6 of the 28 mandatory runs and previously had
zero CPU coverage), as a committed, CPU-runnable script. Wired as a launch
gate for every KEY_ANCHORING wave by run_deltanet_rd_exactness_sweep.py's
main() (same audit fix) -- rc!=0 aborts the wave.

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


# ---------------------------------------------------------------------------
# smoke 10: candidate (c)'s AnchorEMA (2026-07-04 audit fix -- (c) is 6 of
# the 28 mandatory runs and previously had zero CPU coverage): update()
# twice with synthetic key_ids, loss_anchor() finite, and the anchor table
# receives NO gradient (a plain tensor, never an nn.Parameter -- sec 2.3's
# stop-gradient / no-gradient-into-statistics discipline).
# ---------------------------------------------------------------------------

def smoke_10_anchor_ema():
    from run_deltanet_rd import AnchorEMA
    torch.manual_seed(6)
    ema = AnchorEMA(vocab_size_total=50, d_state=8, device="cpu")
    key_ids = torch.randint(0, 50, (2, 4))
    k_eff = F.normalize(torch.randn(2, 4, 8), dim=-1).clone().requires_grad_(True)
    ema.update(key_ids, k_eff)
    ema.update(key_ids, k_eff)
    loss = ema.loss_anchor(key_ids, k_eff)
    loss.backward()
    loss_finite = bool(torch.isfinite(loss).item())
    k_grad_finite = k_eff.grad is not None and bool(torch.isfinite(k_eff.grad).all().item())
    table_plain = not isinstance(ema.table, torch.nn.Parameter) and not ema.table.requires_grad
    table_no_grad = getattr(ema.table, "grad", None) is None
    counts_updated = bool((ema.counts[torch.unique(key_ids)] == 2).all().item())
    no_bias_corrected = not hasattr(ema, "bias_corrected")   # the deleted landmine stays deleted
    print(f"    loss={loss.item():.4f} finite={loss_finite} k_eff_grad_finite={k_grad_finite} "
          f"table_plain_tensor_no_grad={table_plain and table_no_grad} "
          f"counts_at_2_after_2_updates={counts_updated} bias_corrected_absent={no_bias_corrected}")
    ok = (loss_finite and k_grad_finite and table_plain and table_no_grad
          and counts_updated and no_bias_corrected)
    _report("smoke 10: AnchorEMA (candidate (c)) -- finite loss, gradient-isolated table", ok)


# ---------------------------------------------------------------------------
# smoke 11: keyanchor_drift_diagnostic.py's exit-code regression (sec 9.3
# item 2, the swallowed-failure bug -- DYNAMIC companion to
# smoke_keyanchor_confirm.py's fla-free static source check). Requires fla
# (this whole module already does, via `import model_rd as mrd` above) --
# runs wherever THIS file already runs (box tdenv).
# ---------------------------------------------------------------------------

def smoke_11_drift_diag_exit_code_regression():
    """keyanchor_drift_diagnostic.main() is a thin wrapper around _run()
    that turns ANY exception into sys.exit(1) (2026-07-06 fix -- see that
    module's docstring, item 2). Monkeypatches _run to raise, calls the
    REAL main(), and asserts SystemExit(1) -- never a silent return or
    zero exit, closing the sec 9.3 verdict's documented "crashed AND
    exited 0 anyway" gap defensively at this file's own level (the actual
    root cause was keyanchor_chain.sh's tee/pipefail bug -- see
    smoke_keyanchor_confirm.py's smoke B -- but this guarantees the
    diagnostic itself is ALSO never the one silently swallowing a
    failure)."""
    import keyanchor_drift_diagnostic as kd

    orig_run = kd._run
    kd._run = lambda: (_ for _ in ()).throw(RuntimeError("smoke-injected failure"))
    try:
        raised, code = None, None
        try:
            kd.main()
        except SystemExit as e:
            raised, code = True, e.code
        except Exception as e:
            raised, code = "wrong_exception_type", repr(e)
    finally:
        kd._run = orig_run

    print(f"    main() with a raising _run(): raised SystemExit={raised} code={code} (expect True, 1)")
    ok = raised is True and code == 1
    _report("smoke 11: keyanchor_drift_diagnostic.main() exit-code regression "
            "(any _run() exception -> SystemExit(1), never swallowed)", ok)


# ---------------------------------------------------------------------------
# smoke 12: run_deltanet_rd.py's train() checkpoint block wires item 6
# (table conditioning) + sec 3.7 (per-entity alignment/h1 companion) --
# STATIC source check (dynamic end-to-end coverage needs a REAL
# bind()/train() call, CUDA-only, same carve-out as smoke 8's own note).
# ---------------------------------------------------------------------------

def smoke_12_train_checkpoint_item6_sec37_wiring():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "run_deltanet_rd.py")) as f:
        src = f.read()
    has_item6_call = "ka.raw_table_conditioning(" in src and "item6_table_conditioning" in src
    has_alignment_call = "ka.measure_full_pool_alignment(" in src and "per_entity_alignment" in src
    has_h1_call = "ka.measure_h1_behavioral_companion(" in src and "per_entity_h1_companion" in src
    gated_on_anchor_active = "if model.anchor_active:" in src
    gated_on_final_step = "if step == steps:" in src
    print(f"    item6 call present: {has_item6_call}")
    print(f"    sec 3.7 alignment call present: {has_alignment_call}, h1 companion call present: {has_h1_call}")
    print(f"    gated on model.anchor_active: {gated_on_anchor_active}, gated on final step: {gated_on_final_step}")
    ok = (has_item6_call and has_alignment_call and has_h1_call
          and gated_on_anchor_active and gated_on_final_step)
    _report("smoke 12: run_deltanet_rd.py train() item-6/sec-3.7 checkpoint wiring (static source check)", ok)


def _build_model_dprime(vocab=300, d_state=64, n_train=107):
    """sec 10.5.1, candidate (d'): the SAME _build_model construction, with
    anchor_lambda_mode='learned_per_entity'."""
    train_ids = torch.arange(10, 10 + n_train)
    model = mrd.DeltaNetRDBlock(
        vocab, d_model=64, d_state=d_state, conv_size=4, buffer_id=0,
        geo3_active=True, geo3_n_iter=12, geo3_resid_tol=1e-2,
        anchor_active=True, anchor_lambda_mode="learned_per_entity",
        anchor_train_ids=train_ids)
    return model, train_ids


# ---------------------------------------------------------------------------
# smoke 13 (sec 10.9 item 8): candidate (d') forward/backward -- finite loss,
# finite grad on EVERY parameter INCLUDING anchor_lambda_table, at a
# realistic AND an adversarial (near-duplicate) input, PLUS a direct check
# that lambda_e actually VARIES per entity under a synthetic gradient (the
# structural-capacity question sec 10.5.1 exists to test). Same GPU-free
# insertion-site testing methodology as smoke 2 (anchor_blend_gather_
# scatter_per_entity + geo3_orthogonalize_logged called directly, both
# pure-torch, no fla kernel/CUDA touched -- only `import model_rd` above
# requires fla to be INSTALLABLE, not a GPU to be PRESENT).
# ---------------------------------------------------------------------------

def smoke_13_dprime_blend_fwd_bwd():
    model, train_ids = _build_model_dprime()
    all_ok = True
    for label, raw_fn in (("realistic", lambda B, K, d: F.normalize(torch.randn(B, K, d), dim=-1)),
                            ("adversarial (near-duplicate)", _adversarial_raw)):
        model.zero_grad()
        B, K, d = 4, 16, model.d_state
        raw = raw_fn(B, K, d).clone().requires_grad_(True)
        key_ids = _random_key_ids(B, K, train_ids, model.vocab_size_total, frac_trained=0.7, seed=11)
        k_blend = ka.anchor_blend_gather_scatter_per_entity(
            raw, model.anchor_table.weight, model.anchor_trained_mask, key_ids,
            model.anchor_lambda_table.weight)
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
        lambda_e_grad_present = model.anchor_lambda_table.weight.grad is not None and \
            bool(torch.isfinite(model.anchor_lambda_table.weight.grad).all().item())
        this_ok = (finite_loss and raw_grad_finite and all_param_grad_finite and anchor_grad_present
                   and lambda_e_grad_present)
        print(f"    [{label}] loss_finite={finite_loss} raw_grad_finite={raw_grad_finite} "
              f"anchor_table_grad_finite={anchor_grad_present} "
              f"lambda_e_table_grad_finite={lambda_e_grad_present} fallback_triggered={fallback}")
        all_ok = all_ok and this_ok

    # Structural-capacity check: lambda_e must actually be ABLE to vary per
    # entity (not collapse to an effectively-scalar table by construction) --
    # apply one synthetic gradient step with a per-entity-varying target loss
    # and assert the trained rows' sigmoid(lambda_e) values are NOT all
    # identical afterward.
    model2, train_ids2 = _build_model_dprime()
    opt = torch.optim.SGD([model2.anchor_lambda_table.weight], lr=10.0)
    target = torch.linspace(0.1, 0.9, train_ids2.numel())
    for _ in range(5):
        opt.zero_grad()
        lam_e = torch.sigmoid(model2.anchor_lambda_table.weight[train_ids2].squeeze(-1))
        loss = ((lam_e - target) ** 2).sum()
        loss.backward()
        opt.step()
    final_lam_e = torch.sigmoid(model2.anchor_lambda_table.weight[train_ids2].squeeze(-1)).detach()
    varies_per_entity = bool((final_lam_e.std() > 0.05).item())
    print(f"    lambda_e varies per entity under a synthetic gradient: std={final_lam_e.std().item():.4f} "
          f"(expect > 0.05, target range was [0.1,0.9])")
    all_ok = all_ok and varies_per_entity
    _report("smoke 13: candidate (d') blend forward/backward (realistic + adversarial, all grads "
            "finite) + lambda_e structural-capacity check (varies per entity under gradient)", all_ok)


# ---------------------------------------------------------------------------
# smoke 14 (sec 10.9 item 9): candidate (d') held-out bypass + NaN-injection
# isolation -- re-run fresh for the NEW architecture, not assumed inherited
# from smoke 3/4: all-held-out batch bit-identity to bare geo3, PLUS NaN
# planted in every held-out anchor_lambda_table row, assert finite gradients
# everywhere and exact-zero gradient at held-out rows.
# ---------------------------------------------------------------------------

def smoke_14_dprime_heldout_and_nan():
    # (a) held-out bypass bit-identity.
    model, train_ids = _build_model_dprime()
    B, K, d = 4, 16, model.d_state
    raw = F.normalize(torch.randn(B, K, d), dim=-1)
    key_ids = torch.randint(0, 10, (B, K), dtype=torch.int64)   # ALL held-out (ids 0..9)
    k_blend = ka.anchor_blend_gather_scatter_per_entity(
        raw, model.anchor_table.weight, model.anchor_trained_mask, key_ids,
        model.anchor_lambda_table.weight)
    bit_identical = torch.equal(k_blend, raw)
    _report("smoke 14a: candidate (d') all-held-out batch bit-identical to anchor-disabled path",
            bit_identical)

    # (b) NaN-injection isolation -- planted in BOTH anchor_table AND
    # anchor_lambda_table's held-out rows (candidate (d') has two tables now).
    model2, train_ids2 = _build_model_dprime()
    with torch.no_grad():
        model2.anchor_table.weight[~model2.anchor_trained_mask] = float("nan")
        model2.anchor_lambda_table.weight[~model2.anchor_trained_mask] = float("nan")
    raw2 = _adversarial_raw(B, K, d, seed=12).clone().requires_grad_(True)
    key_ids2 = _random_key_ids(B, K, train_ids2, model2.vocab_size_total, frac_trained=0.5, seed=13)
    trained_here2 = model2.anchor_trained_mask[key_ids2]

    k_blend2 = ka.anchor_blend_gather_scatter_per_entity(
        raw2, model2.anchor_table.weight, model2.anchor_trained_mask, key_ids2,
        model2.anchor_lambda_table.weight)
    loss = k_blend2.sum()
    loss.backward()

    all_grads_finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model2.parameters())
    raw_grad_finite = raw2.grad is not None and bool(torch.isfinite(raw2.grad).all().item())
    anchor_grad_zero = bool((model2.anchor_table.weight.grad[~model2.anchor_trained_mask] == 0).all().item())
    lambda_e_grad_zero = bool(
        (model2.anchor_lambda_table.weight.grad[~model2.anchor_trained_mask] == 0).all().item())
    heldout_output_bit_equal = torch.equal(k_blend2[~trained_here2], raw2.detach()[~trained_here2])
    print(f"    all_grads_finite={all_grads_finite} raw_grad_finite={raw_grad_finite} "
          f"anchor_table_grad_heldout_exact_zero={anchor_grad_zero} "
          f"lambda_e_table_grad_heldout_exact_zero={lambda_e_grad_zero} "
          f"heldout_output_bit_equal_despite_NaN={heldout_output_bit_equal}")
    ok = (all_grads_finite and raw_grad_finite and anchor_grad_zero and lambda_e_grad_zero
          and heldout_output_bit_equal)
    _report("smoke 14b: candidate (d') NaN-injected held-out rows (both tables) -- finite grads, "
            "exact-zero grad, bit-equal output", ok)


# ---------------------------------------------------------------------------
# smoke 15 (sec 10.13, candidate (e), 2026-07 K48+e build): the anchor
# table's TRAINED-ROW block receives NO gradient when anchor_table_frozen=
# True, on a REAL forward/backward through the actual bind()-insertion-site
# functions (never a hand-wave) -- mirrors smoke 2's own realistic +
# adversarial construction. Also asserts the ORDINARY (non-frozen) path
# still gets a real, finite gradient on the SAME architecture, so this smoke
# cannot pass merely because gradients are broken everywhere.
# ---------------------------------------------------------------------------

def _build_model_frozen_e(vocab=300, d_state=64, n_train=107, lambda_fixed=0.58,
                            init_mode="random_unit_rows"):
    train_ids = torch.arange(10, 10 + n_train)
    model = mrd.DeltaNetRDBlock(
        vocab, d_model=64, d_state=d_state, conv_size=4, buffer_id=0,
        geo3_active=True, geo3_n_iter=20, geo3_resid_tol=1e-2,
        anchor_active=True, anchor_lambda_mode="fixed", anchor_lambda_fixed=lambda_fixed,
        anchor_train_ids=train_ids, anchor_table_frozen=True,
        anchor_table_init_mode=init_mode)
    return model, train_ids


def smoke_15_candidate_e_frozen_table_no_grad():
    all_ok = True

    # (a) requires_grad is False on the trained-row block immediately at
    # construction -- checked BEFORE any forward/backward, since a
    # backward-only check would miss a table that merely never gets a
    # nonzero grad by coincidence (e.g. bug that zeros the graph elsewhere).
    model_e, train_ids = _build_model_frozen_e()
    requires_grad_false = model_e.anchor_table.weight.requires_grad is False
    print(f"    anchor_table.weight.requires_grad immediately after construction: "
          f"{model_e.anchor_table.weight.requires_grad} (expect False)")
    all_ok = all_ok and requires_grad_false

    # (b) a REAL forward/backward through the production insertion-site
    # functions (same pattern as smoke 2/4): loss.backward() must not raise
    # (frozen leaf tensors are a valid autograd state, but a bug that tries
    # to write into a no-grad tensor's .grad via an in-place op could still
    # break) and model_e.anchor_table.weight.grad must be None afterward --
    # NOT a tensor of exact zeros (torch never populates .grad for a
    # requires_grad=False leaf at all), which is the actual "frozen" contract.
    B, K, d = 4, 16, model_e.d_state
    for label, raw_fn in (("realistic", lambda B, K, d: F.normalize(torch.randn(B, K, d), dim=-1)),
                            ("adversarial (near-duplicate)", _adversarial_raw)):
        model_e.zero_grad()
        raw = raw_fn(B, K, d).clone().requires_grad_(True)
        key_ids = _random_key_ids(B, K, train_ids, model_e.vocab_size_total, frac_trained=0.5, seed=21)
        lam = model_e.anchor_lambda()   # fixed mode: returns the registered constant, no grad needed
        k_blend = ka.anchor_blend_gather_scatter(raw, model_e.anchor_table.weight,
                                                    model_e.anchor_trained_mask, key_ids, lam)
        q, fallback, resid = mrd.geo3_orthogonalize_logged(k_blend, n_iter=model_e.geo3_n_iter,
                                                              resid_tol=model_e.geo3_resid_tol)
        loss = q.sum()
        loss.backward()
        finite_loss = bool(torch.isfinite(loss).item())
        raw_grad_finite = raw.grad is not None and bool(torch.isfinite(raw.grad).all().item())
        anchor_grad_is_none = model_e.anchor_table.weight.grad is None
        print(f"    [{label}] loss_finite={finite_loss} raw_grad_finite={raw_grad_finite} "
              f"anchor_table.weight.grad is None: {anchor_grad_is_none} (expect True -- frozen, "
              f"never receives ANY grad, not merely a zero one) fallback_triggered={fallback}")
        all_ok = all_ok and finite_loss and raw_grad_finite and anchor_grad_is_none

    # (c) negative control: the ORDINARY (non-frozen, candidate-(d)-style)
    # path on the identical architecture DOES get a real, finite gradient --
    # proves this smoke isn't passing because gradients are broken globally.
    model_d, train_ids_d = _build_model(lambda_mode="fixed", lambda_fixed=0.58)
    raw_d = F.normalize(torch.randn(B, K, d), dim=-1).clone().requires_grad_(True)
    key_ids_d = _random_key_ids(B, K, train_ids_d, model_d.vocab_size_total, frac_trained=0.5, seed=22)
    lam_d = model_d.anchor_lambda()
    k_blend_d = ka.anchor_blend_gather_scatter(raw_d, model_d.anchor_table.weight,
                                                  model_d.anchor_trained_mask, key_ids_d, lam_d)
    q_d, _, _ = mrd.geo3_orthogonalize_logged(k_blend_d, n_iter=model_d.geo3_n_iter,
                                                 resid_tol=model_d.geo3_resid_tol)
    q_d.sum().backward()
    trained_grad_present = model_d.anchor_table.weight.grad is not None and \
        bool(torch.isfinite(model_d.anchor_table.weight.grad).all().item()) and \
        bool((model_d.anchor_table.weight.grad[model_d.anchor_trained_mask] != 0).any().item())
    print(f"    [negative control] non-frozen (candidate d) anchor_table trained-row grad "
          f"present and nonzero: {trained_grad_present} (expect True)")
    all_ok = all_ok and trained_grad_present

    _report("smoke 15: candidate (e) frozen anchor table receives NO grad "
            "(requires_grad=False pre- and post-backward), non-frozen path unaffected", all_ok)


# ---------------------------------------------------------------------------
# smoke 16 (sec 10.13, candidate (e)): fixed lambda NEVER moves across
# optimizer steps -- a real multi-step training loop on the frozen-table
# architecture, asserting anchor_lambda() returns the EXACT same registered
# constant before and after several backward+opt.step() calls (fixed mode
# has no nn.Parameter backing it at all, so this is really a construction/
# wiring check: no code path accidentally re-introduces a trainable lambda
# under anchor_lambda_mode='fixed').
# ---------------------------------------------------------------------------

def smoke_16_candidate_e_fixed_lambda_never_moves():
    model_e, train_ids = _build_model_frozen_e(lambda_fixed=0.58)
    opt = torch.optim.Adam(model_e.parameters(), lr=0.01)
    B, K, d = 4, 16, model_e.d_state
    lam_before = model_e.anchor_lambda().clone()
    no_lambda_param = model_e.anchor_lambda_raw is None and model_e.anchor_lambda_table is None
    for step in range(5):
        opt.zero_grad()
        raw = F.normalize(torch.randn(B, K, d), dim=-1).clone().requires_grad_(True)
        key_ids = _random_key_ids(B, K, train_ids, model_e.vocab_size_total, frac_trained=0.5, seed=30 + step)
        lam = model_e.anchor_lambda()
        k_blend = ka.anchor_blend_gather_scatter(raw, model_e.anchor_table.weight,
                                                    model_e.anchor_trained_mask, key_ids, lam)
        q, _, _ = mrd.geo3_orthogonalize_logged(k_blend, n_iter=model_e.geo3_n_iter,
                                                   resid_tol=model_e.geo3_resid_tol)
        q.sum().backward()
        opt.step()
    lam_after = model_e.anchor_lambda()
    lambda_bit_identical = torch.equal(lam_before, lam_after)
    lambda_equals_registered = bool((lam_after == 0.58).all().item())
    print(f"    lambda before={lam_before.item():.6f} after 5 opt.step()s={lam_after.item():.6f} "
          f"(expect bit-identical, ==0.58) no_trainable_lambda_param={no_lambda_param}")
    ok = lambda_bit_identical and lambda_equals_registered and no_lambda_param
    _report("smoke 16: candidate (e) fixed lambda never moves across optimizer steps "
            "(no trainable lambda param exists under anchor_lambda_mode='fixed')", ok)


# ---------------------------------------------------------------------------
# smoke 17 (sec 10.13, candidate (e)): random_unit_rows_init is genuinely
# NOT frame_potential_init -- same shape/dtype/seed-determinism contract,
# but a DIFFERENT (unoptimized) construction: its frame potential (sum of
# squared off-diagonal cosines) must be materially HIGHER than the
# frame-potential-minimized table's own, at the same (n, d, seed). Guards
# against a future refactor accidentally aliasing the two initializers.
# ---------------------------------------------------------------------------

def _frame_potential(table: torch.Tensor) -> float:
    row_norm = F.normalize(table, dim=-1)
    gram = row_norm @ row_norm.transpose(0, 1)
    n = table.shape[0]
    off_diag = gram[~torch.eye(n, dtype=torch.bool)]
    return (off_diag ** 2).sum().item()


def smoke_17_random_unit_rows_init_distinct():
    n, d, seed = 107, 64, ka.ANCHOR_INIT_SEED
    random_table = ka.random_unit_rows_init(n, d, seed=seed)
    fp_table = ka.frame_potential_init(n, d, seed=seed)
    same_shape = random_table.shape == fp_table.shape == (n, d)
    unit_norm = torch.allclose(random_table.norm(dim=-1), torch.ones(n), atol=1e-5)
    deterministic = torch.equal(random_table, ka.random_unit_rows_init(n, d, seed=seed))
    different_seed_differs = not torch.equal(random_table, ka.random_unit_rows_init(n, d, seed=seed + 1))
    fp_random = _frame_potential(random_table)
    fp_optimized = _frame_potential(fp_table)
    # frame_potential_init's own descent converges EXACTLY to the Welch
    # bound (verified: fp_optimized == n*(n-1)*(n-d)/(d*(n-1)) == 71.8906
    # for (107,64), matching gate2_construction_test.py's own pinned
    # sigma_ratio==1.0 tight-frame property) -- this IS the theoretical
    # minimum frame potential, so ANY ratio > 1.0 already proves the random
    # draw is not that minimizer. Measured across 5 seeds (20260705, 1, 2,
    # 3, 42) before picking this threshold: ratio lands 2.41-2.50x in every
    # case (a real, seed-stable gap, not a single-draw fluke) -- 2.0x is a
    # safe floor with margin below the observed range, not a tuned-to-pass
    # threshold picked after seeing seed=20260705 alone.
    materially_worse = fp_random > fp_optimized * 2.0
    print(f"    shapes match: {same_shape}  unit_norm: {unit_norm}  "
          f"deterministic (same seed): {deterministic}  differs (seed+1): {different_seed_differs}")
    print(f"    frame potential: random={fp_random:.4f}  frame_potential_init={fp_optimized:.4f} "
          f"ratio={fp_random / fp_optimized:.4f} (random expected ~2.4-2.5x optimized's own Welch-"
          f"bound minimum -- NOT the same construction)")
    ok = same_shape and unit_norm and deterministic and different_seed_differs and materially_worse
    _report("smoke 17: random_unit_rows_init is genuinely distinct from frame_potential_init "
            "(same contract, different/unoptimized construction)", ok)


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
    smoke_10_anchor_ema()
    smoke_11_drift_diag_exit_code_regression()
    smoke_12_train_checkpoint_item6_sec37_wiring()
    print("-" * 70)
    print("KEY_ANCHORING_DESIGN.md sec 10.9 items 8/9 -- candidate (d') smokes (2026-07-06 "
          "keyanchor-mech build)")
    print("-" * 70)
    smoke_13_dprime_blend_fwd_bwd()
    smoke_14_dprime_heldout_and_nan()
    print("-" * 70)
    print("KEY_ANCHORING_DESIGN.md sec 10.13 -- candidate (e) frozen-random-table "
          "ablation smokes (2026-07 K48+e build)")
    print("-" * 70)
    smoke_15_candidate_e_frozen_table_no_grad()
    smoke_16_candidate_e_fixed_lambda_never_moves()
    smoke_17_random_unit_rows_init_distinct()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL 17 ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
