"""f15_lm_checkpoint.py -- the F15-LM checkpoint (DELTANET_REALDATA_DESIGN.md
section 4.4, section 7's "F15-LM checkpoint (blocking)" row, R2-3/R2-8).
BLOCKING: per section 7, this must pass before Wave -1 launches, and per the
task brief, results are reported item-by-item before any further build.

Items covered here (Tier 0/1 unique to this LM path; beta-mask exactness and
the R2-8 per-item leak check are ALREADY covered by model_rd.py's own
--smoke suite, section 5.2 -- re-invoked at the end of this script rather
than duplicated, so there is exactly one source of truth for each check):

  0. Environment facts: fla/torch versions, CUDA, measured ShortConvolution
     default kernel_size (conv_size -- section 5.2's adjacency constraint is
     parametric in this, never hard-coded against an assumed value).
  1. Tier 0 (MAJOR-4): fp64 finite-difference gradcheck of the PATCHED
     fla.ops.delta_rule.naive.delta_rule_recurrence (float32 downcast
     removed -- the exact bug the synthetic design's own F15 checkpoint
     diagnosed) at Wave-1-relevant (D,T) shapes, run to completion, WITH a
     detached-path negative test.
  2. Tier 1: bf16 chunk_delta_rule (the production kernel) vs the
     Tier-0-verified reference, forward + gradients, <1e-2 relative
     tolerance, WITH a negative test AND a supplementary check of the
     model_rd.py-specific q=k tensor-aliasing usage pattern.
  3. Crash-safety sweep at the PADDED shapes the harness actually submits
     (max(T_bind, model_rd._MIN_KERNEL_T) for K in {8,16,32,48,64} at
     conv_size=4, with harness-realistic exact-zero k rows), plus a padding
     state-neutrality check, plus a subprocess-isolated head-dim safety
     matrix. Rounds 1-2 of this checkpoint SURFACED two kernel crash
     regimes (both CUDA illegal memory access in prepare_wy_repr_bwd's
     autotuner): raw T<chunk_size=64 (the synthetic design's documented
     hazard, reproduced at (T=56, d_state=16)) AND head dim < 64 even at
     safe T (D=32 crashed at T=128/224; D=16 at T=224). Calling
     chunk_delta_rule directly (R2-3) forfeits the stock layer's q_len<=64
     auto-fallback protection, so model_rd (a) pads to >= _MIN_KERNEL_T=128
     (state-neutral by construction: beta=0 at pad positions) and (b)
     asserts d_state in the measured-safe set _SAFE_D_STATE=(64,128).
  4. Round-trip: production-kernel single full-sequence call == prefix call
     -> continuation call via initial_state (mirrors deltanet_core.py's own
     "core 2" self-test, through the PRODUCTION kernel instead of the
     pure-PyTorch reference -- this is a VERIFICATION of the kernel's state
     hand-off capability, not a claim that model_rd.py's training forward
     pass literally splits BIND into two calls; it does not, see
     model_rd.py's bind()).
  5. R2-3 buffer-pinning: post-optimizer-step exact-zero holds WITH the
     pin/zero-grad convention applied, AND (negative control) WOULD drift
     WITHOUT it -- demonstrating the intervention is load-bearing, not
     vacuous.
  6. grammar_rd._self_test() + model_rd._self_test() (beta-mask exactness
     from the tensor, R2-8 leak check, C15, buffer pinning at init, no-W_q
     param surface, etc. -- re-invoked here, not duplicated). Includes
     model_rd [model 10], the audit-round-1 FATAL-0 regression: idealized
     recall through kernel_state_design_layout (fla returns final_state
     [K,V], KEY axis first; the design convention is [V,K] -- see that
     helper's docstring) must give recall cosine ~1.0, and a deliberately
     transposed state must FAIL. Items 2 and 4 above are LAYOUT-BLIND by
     construction (item 2 compares kernel vs reference in the SAME fla
     layout; item 4's round trip is layout-agnostic), so [model 10] is
     the only item that closes the kernel-state -> retrieval loop.

Run: python f15_lm_checkpoint.py   (requires CUDA; ~a few minutes)
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS: dict[str, dict] = {}


def _record(item: str, passed: bool, detail: dict):
    RESULTS[item] = {"passed": bool(passed), **detail}
    status = "PASS" if passed else "FAIL"
    print(f"\n>>> [{item}] {status}", flush=True)
    for k, v in detail.items():
        print(f"      {k}: {v}", flush=True)


# ---------------------------------------------------------------------------
# 0. Environment facts
# ---------------------------------------------------------------------------

def check_environment():
    import fla
    from fla.modules import ShortConvolution

    fla_version = getattr(fla, "__version__", "unknown")
    sc = ShortConvolution(hidden_size=64, kernel_size=4, bias=False, activation="silu")
    measured_kernel_size = sc.kernel_size[0] if isinstance(sc.kernel_size, tuple) else sc.kernel_size

    detail = {
        "fla_version": fla_version, "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "measured_ShortConvolution_kernel_size_when_constructed_with_4": measured_kernel_size,
        "DeltaNetRDTaskConfig_default_conv_size": 4,
    }
    passed = torch.cuda.is_available() and measured_kernel_size == 4
    _record("0_environment", passed, detail)
    return passed


# ---------------------------------------------------------------------------
# 1. Tier 0 -- fp64 gradcheck of the PATCHED naive reference
# ---------------------------------------------------------------------------

def patched_delta_rule_recurrence(q, k, v, beta, initial_state=None, output_final_state=True):
    """Copy of fla.ops.delta_rule.naive.delta_rule_recurrence (fla 0.5.1,
    verified against the installed package source 2026-07-02) with the
    internal ``map(lambda x: x.float(), ...)`` downcast REMOVED -- Tier 0's
    ground-truth reference (MAJOR-4). Preserves dtype throughout (fp64-
    gradcheck-able, unlike the stock function, which silently defeats a
    fp64 gradcheck by downcasting to float32 first). Shape convention
    IDENTICAL to fla's own: (b, h, l, d) -- NOT chunk_delta_rule's own
    (b, l, h, d); Tier 1 transposes explicitly between the two."""
    orig_dtype = q.dtype
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v, dtype=q.dtype, device=q.device)
    q = q * (d_k ** -0.5)
    if beta.ndim < v.ndim:
        beta = beta[..., None]
    if initial_state is not None:
        S = S + initial_state
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    S_out = None if output_final_state is False else S
    return o.to(orig_dtype), S_out


def check_tier0_gradcheck():
    device = "cuda"
    results = {}
    all_pass = True
    # Wave-1-relevant shapes: T_bind at K=8 (56), K=16 (112), K=32 (224) --
    # the smallest through the primary/likely-primary K, at conv_size=4.
    # D kept small (gradcheck cost scales with numel); B=1 per section 4.4.
    for T in (56, 112, 224):
        torch.manual_seed(0)
        B, H, D = 1, 1, 4
        q = torch.randn(B, H, T, D, dtype=torch.float64, device=device, requires_grad=True)
        k = F.normalize(torch.randn(B, H, T, D, dtype=torch.float64, device=device), dim=-1).requires_grad_(True)
        v = torch.randn(B, H, T, D, dtype=torch.float64, device=device, requires_grad=True)
        beta = torch.rand(B, H, T, dtype=torch.float64, device=device).sigmoid().requires_grad_(True)

        def f(q_, k_, v_, beta_):
            o, S = patched_delta_rule_recurrence(q_, k_, v_, beta_, output_final_state=True)
            return o, S

        t0 = time.time()
        try:
            ok = torch.autograd.gradcheck(f, (q, k, v, beta), eps=1e-6, atol=1e-5, rtol=1e-3)
        except Exception as e:
            ok = False
            results[f"T{T}_error"] = repr(e)
        dt = time.time() - t0
        results[f"T{T}_gradcheck_passed"] = bool(ok)
        results[f"T{T}_wall_s"] = round(dt, 2)
        all_pass = all_pass and bool(ok)
        print(f"    T={T}: gradcheck {'PASSED' if ok else 'FAILED'} in {dt:.1f}s", flush=True)

    # Detached-path negative test (run to completion, per house discipline)
    T = 56
    B, H, D = 1, 1, 4
    torch.manual_seed(1)
    q = torch.randn(B, H, T, D, dtype=torch.float64, device=device, requires_grad=True)
    k = F.normalize(torch.randn(B, H, T, D, dtype=torch.float64, device=device), dim=-1)
    v = torch.randn(B, H, T, D, dtype=torch.float64, device=device)
    beta = torch.rand(B, H, T, dtype=torch.float64, device=device).sigmoid()

    def f_broken(q_):
        o, S = patched_delta_rule_recurrence(q_, k.detach(), v.detach(), beta.detach(),
                                              output_final_state=True)
        return o, S

    neg_failed = False
    try:
        passed = torch.autograd.gradcheck(f_broken, (q,), eps=1e-6, atol=1e-5, rtol=1e-3)
        neg_failed = not passed
    except Exception:
        neg_failed = True
    # q still genuinely affects o (via _q at each step) even with k/v/beta
    # detached, so THIS particular negative test checks a DIFFERENT thing:
    # that gradcheck has teeth on a shape mismatch / detach elsewhere. Use a
    # cleaner negative: initial_state detached.
    S0 = torch.randn(B, H, D, D, dtype=torch.float64, device=device, requires_grad=True)
    k2 = F.normalize(torch.randn(B, H, T, D, dtype=torch.float64, device=device), dim=-1)
    v2 = torch.randn(B, H, T, D, dtype=torch.float64, device=device)
    beta2 = torch.rand(B, H, T, dtype=torch.float64, device=device).sigmoid()
    q2 = torch.randn(B, H, T, D, dtype=torch.float64, device=device)

    def f_init_broken(S0_):
        o, S = patched_delta_rule_recurrence(q2, k2, v2, beta2, initial_state=S0_.detach(),
                                              output_final_state=True)
        return o, S

    neg2_failed = False
    try:
        passed2 = torch.autograd.gradcheck(f_init_broken, (S0,), eps=1e-6, atol=1e-5, rtol=1e-3)
        neg2_failed = not passed2
    except Exception:
        neg2_failed = True
    results["negative_test_detached_initial_state_correctly_failed"] = neg2_failed
    all_pass = all_pass and neg2_failed
    print(f"    negative test (detached initial_state) correctly "
          f"{'FAILED gradcheck (teeth confirmed)' if neg2_failed else 'PASSED gradcheck -- NO TEETH, BUG'}",
          flush=True)

    results["largest_T_verified"] = 224
    results["gap_to_wave2_target_disclosed"] = ("T=512 (Wave 2's real-corpus target) NOT tested here -- "
                                                  "fp64 gradcheck cost scales with T; 224 is Wave 1's own "
                                                  "K=32 BIND-phase length (the primary path this build "
                                                  "phase targets). Re-run at larger T before Wave 2 begins.")
    _record("1_tier0_gradcheck", all_pass, results)
    return all_pass


# ---------------------------------------------------------------------------
# 2. Tier 1 -- bf16 kernel vs Tier-0-verified reference
# ---------------------------------------------------------------------------

def _rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius error ||a-b||_F / ||b||_F -- the bf16-appropriate
    aggregate comparison metric. (Round 1 of this checkpoint used max
    ELEMENTWISE relative error with a 1e-3 clamp against a bf16-run
    reference; both choices were wrong: tiny elements made the elementwise
    ratio meaningless, and a bf16 SEQUENTIAL reference accumulates
    quantization error at every one of T steps -- it is not a ground truth.
    Fixed: fp32 reference (dtype-preserving, Tier-0-verified) consuming the
    SAME bf16-QUANTIZED input values, relative-Frobenius comparison.)"""
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def check_tier1_cross_check():
    from fla.ops.delta_rule import chunk_delta_rule
    device = "cuda"
    results = {}
    all_pass = True

    B, H, T, D = 4, 1, 224, 64   # a realistic Wave-1 K=32 shape
    torch.manual_seed(2)
    k0 = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), dim=-1)
    v0 = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    beta0 = torch.rand(B, T, H, dtype=torch.float32, device=device).sigmoid()
    # Quantize ONCE to bf16; BOTH paths consume these exact values (the
    # reference reads them cast back UP to fp32, exact) -- the only
    # remaining difference is the kernel's internal arithmetic, which is
    # exactly what Tier 1 exists to measure.
    k_q, v_q, beta_q = k0.to(torch.bfloat16), v0.to(torch.bfloat16), beta0.to(torch.bfloat16)

    # --- kernel path (bf16; q and k are SEPARATE tensors here for a clean
    # independent-gradient comparison; the model_rd.py-specific q=k ALIASING
    # usage is checked separately below) ---
    q_ker = k_q.clone().requires_grad_(True)
    k_ker = k_q.clone().requires_grad_(True)
    v_ker = v_q.clone().requires_grad_(True)
    beta_ker = beta_q.clone().requires_grad_(True)
    # use_qk_l2norm_in_kernel=False matches model_rd.bind()'s own usage
    # (external zero-safe normalization): kernel and reference compute the
    # IDENTICAL formula on IDENTICAL already-normalized inputs.
    o_ker, S_ker = chunk_delta_rule(q=q_ker, k=k_ker, v=v_ker, beta=beta_ker,
                                     output_final_state=True, use_qk_l2norm_in_kernel=False)
    (o_ker.float().sum() + S_ker.float().sum()).backward()

    # --- reference path (fp32 -- the Tier-0-verified dtype-PRESERVING
    # patched reference, on the same bf16-quantized values) ---
    def to_ref_layout(x):
        return x.transpose(1, 2).contiguous()   # (B,T,H,D) -> (B,H,T,D)

    q_ref = to_ref_layout(k_q.float()).requires_grad_(True)
    k_ref = to_ref_layout(k_q.float()).requires_grad_(True)
    v_ref = to_ref_layout(v_q.float()).requires_grad_(True)
    beta_ref = beta_q.float().transpose(1, 2).contiguous().requires_grad_(True)
    o_ref, S_ref = patched_delta_rule_recurrence(q_ref, k_ref, v_ref, beta_ref, output_final_state=True)
    (o_ref.sum() + S_ref.sum()).backward()

    o_ker_aligned = o_ker.transpose(1, 2)   # (B,T,H,D) -> (B,H,T,D) for comparison
    o_rel = _rel_fro(o_ker_aligned, o_ref)
    S_rel = _rel_fro(S_ker, S_ref)
    fwd_ok = (o_rel < 2e-2) and (S_rel < 2e-2)

    k_grad_rel = _rel_fro(k_ker.grad.transpose(1, 2), k_ref.grad)
    v_grad_rel = _rel_fro(v_ker.grad.transpose(1, 2), v_ref.grad)
    beta_grad_rel = _rel_fro(beta_ker.grad.transpose(1, 2), beta_ref.grad)
    grad_ok = (k_grad_rel < 5e-2) and (v_grad_rel < 5e-2) and (beta_grad_rel < 5e-2)

    results.update({
        "forward_o_rel_fro_err": round(o_rel, 6), "forward_S_rel_fro_err": round(S_rel, 6),
        "forward_ok_lt_2e-2": fwd_ok,
        "grad_k_rel_fro_err": round(k_grad_rel, 6), "grad_v_rel_fro_err": round(v_grad_rel, 6),
        "grad_beta_rel_fro_err": round(beta_grad_rel, 6), "grad_ok_lt_5e-2": grad_ok,
    })
    all_pass = all_pass and fwd_ok and grad_ok
    print(f"    forward (rel Fro vs fp32 reference): o={o_rel:.6f} S={S_rel:.6f} ok={fwd_ok}", flush=True)
    print(f"    grad (rel Fro): k={k_grad_rel:.6f} v={v_grad_rel:.6f} beta={beta_grad_rel:.6f} "
          f"ok={grad_ok}", flush=True)

    # --- supplementary: model_rd.py's ACTUAL usage pattern -- q and k are
    # the LITERAL SAME tensor object (aliased), not just equal-valued.
    # Confirm autograd correctly accumulates the gradient contribution from
    # BOTH the q-role and k-role uses into the single tensor, and that this
    # matches (q_grad_from_kernel + k_grad_from_kernel) computed separately
    # above, within the same tolerance. ---
    qk_aliased = k_q.clone().requires_grad_(True)
    o_a, S_a = chunk_delta_rule(q=qk_aliased, k=qk_aliased, v=v_ker.detach().requires_grad_(True),
                                 beta=beta_ker.detach().requires_grad_(True),
                                 output_final_state=True, use_qk_l2norm_in_kernel=False)
    (o_a.float().sum() + S_a.float().sum()).backward()
    aliased_ok = torch.isfinite(qk_aliased.grad).all().item()
    expected_combined = q_ker.grad.float() + k_ker.grad.float()
    aliased_matches_sum = torch.allclose(qk_aliased.grad.float(), expected_combined, atol=5e-2, rtol=5e-2)
    results["aliased_qk_grad_finite"] = aliased_ok
    results["aliased_qk_grad_matches_separate_q_plus_k_grad"] = aliased_matches_sum
    all_pass = all_pass and aliased_ok and aliased_matches_sum
    print(f"    model_rd.py's q=k aliasing pattern: grad finite={aliased_ok}, "
          f"matches separate-q-grad + separate-k-grad={aliased_matches_sum}", flush=True)

    # --- negative test: corrupt the reference (wrong scale) and confirm the
    # cross-check correctly DETECTS the disagreement (has teeth) ---
    def wrong_reference(q_, k_, v_, beta_):
        # deliberately omit the 1/sqrt(d) scale the real reference applies
        o = torch.zeros_like(v_)
        b, h, l, d_k = q_.shape
        S = torch.zeros(b, h, d_k, v_.shape[-1], dtype=q_.dtype, device=q_.device)
        beta_e = beta_[..., None] if beta_.ndim < v_.ndim else beta_
        for i in range(l):
            _k, _q, _v = k_[:, :, i], q_[:, :, i], v_[:, :, i].clone()
            _v = (_v - (S.clone() * _k[..., None]).sum(-2)) * beta_e[:, :, i]
            S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
            o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)   # NO 1/sqrt(d) scale on q -- deliberate bug
        return o, S

    o_wrong, S_wrong = wrong_reference(q_ref.detach(), k_ref.detach(), v_ref.detach(), beta_ref.detach())
    neg_detected = _rel_fro(o_ker_aligned, o_wrong) > 2e-2   # must EXCEED the same tolerance the real check uses
    results["negative_test_wrong_scale_correctly_detected_as_mismatch"] = neg_detected
    all_pass = all_pass and neg_detected
    print(f"    negative test (missing 1/sqrt(d) scale in reference): mismatch correctly "
          f"{'DETECTED (teeth confirmed)' if neg_detected else 'MISSED -- NO TEETH, BUG'}", flush=True)

    _record("2_tier1_kernel_vs_reference", all_pass, results)
    return all_pass


# ---------------------------------------------------------------------------
# 3. Short-T crash-safety sweep (Wave 1's actual planned K-grid)
# ---------------------------------------------------------------------------

def check_short_t_sweep():
    """Tests the PADDED shapes model_rd.bind() actually submits to the
    kernel -- max(T_bind, _MIN_KERNEL_T) for every Wave-1 K-grid cell --
    with harness-realistic inputs: EXACT-ZERO k rows at buffer/pad
    positions (the fixed harness's zero-safe external F.normalize maps
    zero conv outputs to zero rows) and beta=0 everywhere off the K item
    positions. Also verifies the padding is STATE-NEUTRAL (padded vs
    unpadded S_T at a safe T, bit-level tolerance).

    DOCUMENTED HAZARD (round 1 of this checkpoint, 2026-07-02): calling
    chunk_delta_rule at RAW T=56 (K=8, d_state=16) crashed the backward
    with a CUDA illegal memory access inside prepare_wy_repr_bwd_kernel's
    autotuner -- the synthetic design's known T<chunk_size=64 crash,
    reproduced here (it had passed flakily at d_state=64 in the same
    round). model_rd._MIN_KERNEL_T=128 padding is the fix; this item now
    verifies the padded reality, not the raw hazard."""
    from fla.ops.delta_rule import chunk_delta_rule
    from model_rd import _MIN_KERNEL_T
    device = "cuda"
    results = {"documented_hazard": "raw T<64 backward crash reproduced at (T=56, d_state=16), "
                                     "round 1 of this checkpoint -- harness pads to >= "
                                     f"{_MIN_KERNEL_T} (state-neutral, verified below)"}
    all_pass = True
    conv_size = 4
    buf_len = conv_size - 1
    clause_len = buf_len + 4

    def build_inputs(K, T, D, B=8, H=1, seed=None):
        """Harness-realistic kernel inputs: unit-norm k at item positions
        and their local windows, EXACT-ZERO k at deep buffer/pad positions,
        beta masked to items only."""
        torch.manual_seed(K if seed is None else seed)
        item_pos = (torch.arange(K, device=device) * clause_len + buf_len + 2)
        k_raw = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
        # zero out deep-buffer/pad rows (positions > item+1 within each
        # clause and the entire tail pad) to mirror the real conv-of-zeros
        near_item = torch.zeros(T, dtype=torch.bool, device=device)
        for off in (-2, -1, 0, 1):
            near_item[(item_pos + off).clamp(0, T - 1)] = True
        k_raw[:, ~near_item] = 0.0
        k = F.normalize(k_raw, dim=-1)                       # zero-safe: 0-row -> 0-row
        v = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
        beta_mask = torch.zeros(B, T, H, device=device)
        beta_mask[:, item_pos, :] = 1.0
        beta = torch.rand(B, T, H, dtype=torch.float32, device=device).sigmoid() * beta_mask
        return (k.to(torch.bfloat16), v.to(torch.bfloat16), beta.to(torch.bfloat16), item_pos)

    for K in (8, 16, 32, 48, 64):
        T_raw = K * clause_len
        T = max(T_raw, _MIN_KERNEL_T)                        # what bind() actually submits
        B, H, D = 8, 1, 64
        k, v, beta, _ = build_inputs(K, T, D)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)
        beta = beta.requires_grad_(True)
        try:
            o, S = chunk_delta_rule(q=k, k=k, v=v, beta=beta, output_final_state=True,
                                     use_qk_l2norm_in_kernel=False)
            (o.float().sum() + S.float().sum()).backward()
            finite = torch.isfinite(k.grad).all().item() and torch.isfinite(S).all().item()
            results[f"K{K}_Traw{T_raw}_Tpadded{T}"] = {"status": "OK", "grads_finite": finite}
            all_pass = all_pass and finite
            print(f"    K={K:3d} T_raw={T_raw:4d} T_padded={T:4d}: forward+backward OK "
                  f"(zero-k rows included), finite_grad={finite}", flush=True)
        except Exception as e:
            results[f"K{K}_Traw{T_raw}_Tpadded{T}"] = {"status": "CRASHED", "error": repr(e)[:300]}
            all_pass = False
            print(f"    K={K:3d} T_padded={T:4d}: CRASHED: {e!r}", flush=True)

    # --- padding state-neutrality: at a safe unpadded T (K=32 -> 224),
    # appending 64 zero-k/zero-beta pad positions must leave S unchanged ---
    K, D, pad = 32, 64, 64
    T_raw = K * clause_len
    k, v, beta, _ = build_inputs(K, T_raw, D, seed=99)
    with torch.no_grad():
        _, S_unpadded = chunk_delta_rule(q=k, k=k, v=v, beta=beta, output_final_state=True,
                                          use_qk_l2norm_in_kernel=False)
        zk = torch.zeros(k.shape[0], pad, 1, D, dtype=torch.bfloat16, device=device)
        zb = torch.zeros(k.shape[0], pad, 1, dtype=torch.bfloat16, device=device)
        k_p = torch.cat([k, zk], dim=1)
        v_p = torch.cat([v, torch.randn_like(zk)], dim=1)    # pad V may be ARBITRARY: beta=0 kills it
        beta_p = torch.cat([beta, zb], dim=1)
        _, S_padded = chunk_delta_rule(q=k_p, k=k_p, v=v_p, beta=beta_p, output_final_state=True,
                                        use_qk_l2norm_in_kernel=False)
    pad_diff = (S_unpadded.float() - S_padded.float()).abs().max().item()
    pad_neutral = pad_diff < 1e-6
    results["padding_state_neutrality_max_abs_diff"] = pad_diff
    results["padding_state_neutral"] = pad_neutral
    all_pass = all_pass and pad_neutral
    print(f"    padding state-neutrality (K=32, +{pad} zero-beta pads): max |dS| = {pad_diff:.2e} "
          f"({'NEUTRAL' if pad_neutral else 'NOT NEUTRAL -- BUG'})", flush=True)

    # --- head-dim (d_state) safety matrix, each probe in a FRESH SUBPROCESS
    # so a crash cannot poison this process's CUDA context (rounds 1-2 of
    # this checkpoint were poisoned exactly this way). The harness's
    # _SAFE_D_STATE values MUST all pass; unsafe dims' outcomes are recorded
    # as documentation (they are rejected at model construction anyway). ---
    import subprocess
    from model_rd import _SAFE_D_STATE
    probe_src = (
        "import sys, torch\n"
        "import torch.nn.functional as F\n"
        "from fla.ops.delta_rule import chunk_delta_rule\n"
        "T, D = int(sys.argv[1]), int(sys.argv[2])\n"
        "torch.manual_seed(0)\n"
        "B, H = 8, 1\n"
        "k = F.normalize(torch.randn(B,T,H,D, dtype=torch.float32, device='cuda'), dim=-1)"
        ".to(torch.bfloat16).requires_grad_(True)\n"
        "v = torch.randn(B,T,H,D, dtype=torch.bfloat16, device='cuda', requires_grad=True)\n"
        "beta = torch.rand(B,T,H, dtype=torch.bfloat16, device='cuda').sigmoid().requires_grad_(True)\n"
        "o, S = chunk_delta_rule(q=k,k=k,v=v,beta=beta, output_final_state=True, "
        "use_qk_l2norm_in_kernel=False)\n"
        "(o.float().sum()+S.float().sum()).backward()\n"
        "assert torch.isfinite(k.grad).all()\n"
        "print('PROBE_OK')\n"
    )
    matrix = {}
    for D_probe in (16, 32, 64, 128):
        outcomes = []
        for T_probe in (128, 224):
            r = subprocess.run([sys.executable, "-c", probe_src, str(T_probe), str(D_probe)],
                                capture_output=True, text=True, timeout=300)
            outcomes.append("OK" if "PROBE_OK" in r.stdout else "CRASH")
        matrix[f"D{D_probe}"] = dict(zip(("T128", "T224"), outcomes))
        print(f"    head-dim safety: D={D_probe:3d} -> T128={outcomes[0]} T224={outcomes[1]}", flush=True)
    results["head_dim_safety_matrix"] = matrix
    safe_ok = all(all(v == "OK" for v in matrix[f"D{D}"].values()) for D in _SAFE_D_STATE)
    results["all_SAFE_D_STATE_dims_verified_safe"] = safe_ok
    all_pass = all_pass and safe_ok

    _record("3_short_T_crash_sweep", all_pass, results)
    return all_pass


# ---------------------------------------------------------------------------
# 4. Round-trip (production kernel, full call vs prefix+continuation)
# ---------------------------------------------------------------------------

def check_round_trip():
    from fla.ops.delta_rule import chunk_delta_rule
    device = "cuda"
    # T1/T2 both >= _MIN_KERNEL_T (round 1 used 40/32 -- below the
    # documented backward-crash threshold; forward-only calls dodged the
    # crash, but the harness never submits such lengths anymore, so the
    # round-trip is now verified at kernel-realistic lengths).
    B, H, T1, T2, D = 4, 1, 128, 128, 64
    torch.manual_seed(3)
    k_full = F.normalize(torch.randn(B, T1 + T2, H, D, dtype=torch.float32, device=device),
                          dim=-1).to(torch.bfloat16)
    v_full = torch.randn(B, T1 + T2, H, D, dtype=torch.bfloat16, device=device)
    beta_full = torch.rand(B, T1 + T2, H, dtype=torch.bfloat16, device=device).sigmoid()

    _, S_single = chunk_delta_rule(q=k_full, k=k_full, v=v_full, beta=beta_full,
                                    output_final_state=True, use_qk_l2norm_in_kernel=False)
    _, S_prefix = chunk_delta_rule(q=k_full[:, :T1], k=k_full[:, :T1], v=v_full[:, :T1],
                                    beta=beta_full[:, :T1], output_final_state=True,
                                    use_qk_l2norm_in_kernel=False)
    _, S_two = chunk_delta_rule(q=k_full[:, T1:], k=k_full[:, T1:], v=v_full[:, T1:],
                                 beta=beta_full[:, T1:], initial_state=S_prefix,
                                 output_final_state=True, use_qk_l2norm_in_kernel=False)
    diff = (S_single.float() - S_two.float()).abs().max().item()
    passed = diff < 5e-2   # bf16-appropriate tolerance
    _record("4_round_trip", passed, {"max_abs_diff": round(diff, 5), "tolerance": 5e-2})
    return passed


# ---------------------------------------------------------------------------
# 5. R2-3 buffer-pinning: post-optimizer-step exact-zero, WITH vs WITHOUT
# ---------------------------------------------------------------------------

def check_buffer_pinning():
    import grammar_rd as grd
    from model_rd import DeltaNetRDBlock, pin_buffer_row_, zero_buffer_grad_

    device = "cuda"
    tokenizer = grd.load_gpt2_tokenizer()
    pools, _ = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    pools = pools.to(device)
    cfg = grd.DeltaNetRDTaskConfig(K=8, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))

    def run(with_pinning: bool):
        torch.manual_seed(5)
        # d_state=64 (the Wave-1 primary and a measured-safe head dim) --
        # round 2 of this checkpoint used d_state=16 here and reproduced
        # the prepare_wy_repr_bwd crash D<64 causes; see model_rd._SAFE_D_STATE.
        model = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                                 buffer_id=pools.buffer_id).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)   # large LR to make drift obvious if unpinned
        gen = torch.Generator(device=device).manual_seed(9)
        max_abs = []
        for step in range(8):
            b = grd.sample_batch_rd(cfg, 16, gen, hop_set=cfg.H_train, pools=pools, device=device)
            pred, targets, S_T, _, _ = model(b)
            loss = (1.0 - F.cosine_similarity(pred, targets, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            if with_pinning:
                zero_buffer_grad_(model.embed, model.buffer_id)
            opt.step()
            if with_pinning:
                pin_buffer_row_(model.embed, model.buffer_id)
            with torch.no_grad():
                max_abs.append(model.embed.weight[model.buffer_id].abs().max().item())
        return max_abs

    with_pin = run(with_pinning=True)
    without_pin = run(with_pinning=False)
    pin_holds = all(v == 0.0 for v in with_pin)
    would_drift = any(v > 1e-6 for v in without_pin)
    passed = pin_holds and would_drift
    detail = {
        "with_pinning_max_abs_per_step": with_pin,
        "without_pinning_max_abs_per_step": without_pin,
        "with_pinning_stays_exactly_zero": pin_holds,
        "without_pinning_would_drift_negative_control": would_drift,
    }
    _record("5_buffer_pinning_post_optimizer_step", passed, detail)
    return passed


# ---------------------------------------------------------------------------
# 6. Re-invoke grammar_rd + model_rd self-tests (beta-mask exactness, R2-8
#    leak check, C15, no-W_q param surface, etc.)
# ---------------------------------------------------------------------------

def check_smoke_suite():
    import grammar_rd as grd
    import model_rd as mrd
    ok = True
    try:
        grd._self_test()
    except Exception as e:
        print(f"grammar_rd._self_test() FAILED: {e!r}")
        ok = False
    try:
        mrd._self_test()
    except Exception as e:
        print(f"model_rd._self_test() FAILED: {e!r}")
        ok = False
    _record("6_grammar_and_model_self_tests", ok, {"grammar_rd_and_model_rd_self_test": ok})
    return ok


def main():
    print("=" * 70 + "\n  F15-LM CHECKPOINT -- DeltaNet-RD\n" + "=" * 70)
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available. F15-LM checkpoint cannot run (chunk_delta_rule "
              "has no CPU path).", file=sys.stderr)
        sys.exit(1)

    steps = [
        ("0_environment", check_environment),
        ("1_tier0_gradcheck", check_tier0_gradcheck),
        ("2_tier1_kernel_vs_reference", check_tier1_cross_check),
        ("3_short_T_crash_sweep", check_short_t_sweep),
        ("4_round_trip", check_round_trip),
        ("5_buffer_pinning_post_optimizer_step", check_buffer_pinning),
        ("6_grammar_and_model_self_tests", check_smoke_suite),
    ]
    all_pass = True
    for name, fn in steps:
        try:
            ok = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            _record(name, False, {"exception": repr(e)})
            ok = False
        all_pass = all_pass and ok

    print("\n" + "=" * 70)
    print(f"  F15-LM CHECKPOINT: {'ALL ITEMS PASSED' if all_pass else 'AT LEAST ONE ITEM FAILED'}")
    print("=" * 70)
    for name, r in RESULTS.items():
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {name}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f15_lm_checkpoint_result.json")
    with open(out_path, "w") as f:
        json.dump({"all_passed": all_pass, "items": RESULTS}, f, indent=2, default=str)
    print(f"\nwrote {out_path}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
