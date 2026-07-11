"""reasoning_link_poscontrol.py -- the pre-submission real-kernel positive
control for reasoning-null-moss's Bound 1 (gauntlet round-1 rebuttal FIX-6
tier 2 / final-review decision (b), 2026-07-11).

WHY THIS EXISTS (read this before the code): Bound 1's "instrument verdict,
not a refutation" reading rests entirely on the INTERNAL consistency of a
366/366 zero pattern from `reasoning_link_probe.py`'s own readout
(`squeeze_state_head` + `apply_state_power` + `cosine_and_recovered`) run
against real trained checkpoints. That readout's recovery math was validated
ONLY at unit level -- hand-built matrices, CPU stub, no real kernel, no real
checkpoint (`reasoning_link_stage_minus1.py` items 1-19; module docstring:
"Items needing 'a real checkpoint' ... use ... the CPU fla-stub ... What it
does NOT verify: the real CUDA Triton chunk_delta_rule kernel's own
behavior, or any property of a REAL trained checkpoint's weights"). A
systematic extraction/state-layout bug in the multi-layer path would be
INDISTINGUISHABLE, from the outside, from a genuine construct-invalidity
null -- and this project has ALREADY found exactly this bug class once
before in a sibling file (model_rd.py's kernel_state_design_layout helper,
f15_lm_checkpoint.py item 6 / "[model 10]": "fla returns final_state [K,V],
KEY axis first; the design convention is [V,K] ... a deliberately
transposed state must FAIL").

DESIGN: run ONE real trained checkpoint's REAL forward pass (real fla
Triton kernel, real embed/conv/multi-layer stack, real learned beta gate --
nothing stubbed) through this project's OWN unmodified production
extraction functions (`reasoning_link_probe.register_kqv_hooks`,
`.forward_body`, `.readout_layer_index`, `.squeeze_state_head`,
`.apply_state_power`, `.cosine_and_recovered` -- imported and called
verbatim, never reimplemented, per this codebase's own Stage-1
convention), on a SYNTHETIC bind sequence whose h-hop answer is knowable in
closed form BY CONSTRUCTION, independent of whatever the checkpoint's own
embedding table happens to encode:

  1. Draw K mutually ORTHONORMAL synthetic "keys" k_0..k_{K-1} in R^d_state
     (via a real QR decomposition -- one per batch row, so B independent
     trials run in one forward pass). Orthonormality is what makes the
     delta-rule's own self-correction term (S_{t-1}^T @ k_t) exactly ZERO
     at every step for a NEW item and exactly RECOVER the correct value for
     a REPEATED query -- no reliance on whatever geometry the checkpoint's
     trained embedding table happens to have (which the paper's own premise
     iii/iv findings show is generally NOT well-conditioned).
  2. Assign each item's synthetic "value" to be the NEXT item's key along a
     SINGLE K-cycle: v_i := k_{(i+1) mod K} (CLAUDE.md's own rule: hop-depth
     constructions need a single full K-cycle, never a general permutation,
     or held-out hops silently collapse mod cycle length). This makes the
     correct h-hop answer for querying item a exactly k_{(a+h) mod K} -- a
     closed form that holds EXACTLY (mod bf16 kernel rounding) for every h,
     not just h=1, since orthogonality kills cross-item interference at
     every step of the chain.
  3. Inject these synthetic (k, v) pairs at K token positions of a REAL
     forward pass through a REAL checkpoint by REPLACING (not adding to)
     the readout layer's own k_conv1d / v_conv1d outputs via a forward
     hook -- registered to fire BEFORE reasoning_link_probe's own capture
     hooks (PyTorch chains forward-hook return values in registration
     order), so the capture hooks are ALSO exercised against real,
     verifiable content. All non-item ("filler") positions are forced to
     k=0, which is state-neutral by construction under the delta rule
     (zero key writes nothing, regardless of value or beta -- the same
     zero-safe convention f15_lm_checkpoint.py's own crash-sweep uses) --
     this is what lets a synthetic, non-grammar_rd sequence safely satisfy
     the kernel's own _MIN_KERNEL_T floor without contaminating the
     analytic closed form. Everything else in the forward pass (layer 0's
     real processing, the real embed table, the real learned beta gate,
     the real bf16 kernel boundary, real multi-layer state threading) is
     completely UNTOUCHED.
  4. Read S_T back out via the PRODUCTION `squeeze_state_head` (no
     modification), apply `apply_state_power` for h in {1,2,3,4} against
     the K synthetic queries at once, and score with the PRODUCTION
     `cosine_and_recovered` against the closed-form target. Because v_i is
     an INDEPENDENT key (not v_i=k_i), this construction is NOT symmetric
     under S -> S^T -- a real state-layout/transpose bug would give a
     CLEAN, exactly-derivable wrong answer (predecessor instead of
     successor), not an accidental pass. A second, explicit
     "deliberately-transposed" pass (test 3 below) proves the check has
     teeth for exactly that failure mode, mirroring f15_lm_checkpoint.py's
     own "[model 10]" precedent.

PASS criterion (rebuttal FIX-6 tier 2, verbatim): recovery at cosine >= 0.9
(reuses reasoning_link_probe.RECOVERY_COS_THRESHOLD, never hardcoded) for
the production (unmodified) extraction path, for every h in {1,2,3,4}.
FAIL: STOP -- do not revise Bound 1's framing here; report to the
coordinator per the review's decision (b).

Run (box only -- chunk_delta_rule has no CPU path):
  CUDA_VISIBLE_DEVICES=<idle gpu> /home/nvidia/tdenv/bin/python \
      reasoning_link_poscontrol.py 2>&1 | tee poscontrol_run.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Force the REAL fla package -- this control is worthless under the CPU stub.
os.environ.pop("REASONING_LINK_FORCE_CPU_STUB", None)

import reasoning_link_probe as rlp  # noqa: E402
from lm_pretrain_rd import DeltaNetLM  # noqa: E402 (re-exported by rlp too; imported directly for clarity)

DEFAULT_CKPT = ("/data/deltanet_rd_frozenbias_ckpts/"
                 "frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0/"
                 "lmC_openr1-mix-ext_dm256_ds64_L2_s0_step20000.pt")
K_ITEMS = 32          # matches the paper's own primary K=32 anchor cell
T_SEQ = 128            # == model_rd._MIN_KERNEL_T, the kernel's own floor -- cheapest valid length
B_TRIALS = 8           # independent random draws (batch rows), one forward pass
H_LIST = (1, 2, 3, 4)  # matches reasoning_link_probe.H_TEST exactly
SEED = 20260711


# ---------------------------------------------------------------------------
# Independent reference: verbatim copy of f15_lm_checkpoint.py's own
# Tier-0-gradchecked, dtype-preserving patched delta-rule recurrence
# (fla.ops.delta_rule.naive.delta_rule_recurrence with the float32 downcast
# removed). Reused here as a SECOND, independently-implemented ground truth
# for S_T, cross-checking the production kernel's own output -- not just my
# closed-form derivation, in case that derivation itself has an error.
# ---------------------------------------------------------------------------

def patched_delta_rule_recurrence(q, k, v, beta, initial_state=None, output_final_state=True):
    """Verbatim copy of f15_lm_checkpoint.py's patched_delta_rule_recurrence
    (see that file's own docstring for provenance/audit history). Shape
    convention (b,h,l,d), NOT chunk_delta_rule's own (b,l,h,d)."""
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


# ---------------------------------------------------------------------------
# The override hooks -- fire BEFORE rlp.register_kqv_hooks (registration
# order), so the production capture hooks see exactly what we injected.
# ---------------------------------------------------------------------------

def _make_override_hook(injected: torch.Tensor, item_positions: torch.Tensor):
    """injected: (B,K,d_state). Returns a forward hook that REPLACES the
    conv1d submodule's raw output with `injected` at `item_positions` and
    EXACT ZERO everywhere else (state-neutral filler, f15_lm_checkpoint.py's
    own zero-safe convention)."""
    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        raw = out[0] if is_tuple else out
        new = torch.zeros_like(raw)
        new[:, item_positions, :] = injected.to(raw.dtype)
        return (new,) + out[1:] if is_tuple else new
    return hook


def run_positive_control(ckpt_path: str, device: str = "cuda") -> dict:
    report: dict = {"ckpt_path": ckpt_path, "device": device, "seed": SEED,
                     "K": K_ITEMS, "T_seq": T_SEQ, "B_trials": B_TRIALS, "H_list": list(H_LIST)}

    assert rlp.FLA_STUB_INSTALLED is False, (
        "REAL fla is required for this control -- FLA_STUB_INSTALLED is True, which means "
        "either REASONING_LINK_FORCE_CPU_STUB leaked into this process's env, or the real fla "
        "package is not importable on this box/venv. STOP -- this run would prove nothing.")
    report["fla_stub_installed"] = False
    import fla
    report["fla_version"] = getattr(fla, "__version__", "unknown")
    report["torch_version"] = torch.__version__
    report["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    assert torch.cuda.is_available(), "chunk_delta_rule has no CPU path -- CUDA required."

    torch.manual_seed(SEED)

    model, ckpt = rlp.load_checkpoint(ckpt_path, device)
    cfg = ckpt["config"]
    report["ckpt_config"] = {k: v for k, v in cfg.items()}

    # Guard: this construction's analytic closed form assumes the PLAIN kernel path --
    # no frozen-bias blend, no geo3 selection/orthogonalization, no hard-select masking,
    # no gated-delta variant. Fail loudly (not silently) if the chosen checkpoint doesn't
    # satisfy this, per this project's own "a specification that has not been executed is
    # not a passed gate" / never-silently-assume discipline.
    assert cfg.get("frozen_bias_arm", "off") == "off", (
        f"checkpoint frozen_bias_arm={cfg.get('frozen_bias_arm')!r} != 'off' -- the frozen-bias "
        f"blend is applied to k BETWEEN this control's override hook and the kernel call, which "
        f"would corrupt the injected orthonormal keys. Pick an arm='off' checkpoint.")
    assert cfg.get("geo3_active", False) is False, "geo3_active must be False (would re-select/orthogonalize k)"
    assert cfg.get("hard_select_active", False) is False, "hard_select_active must be False (would mask beta)"
    assert cfg.get("gated_delta_active", False) is False, "gated_delta_active must be False (different kernel call)"
    assert cfg.get("num_heads", 1) == 1, "squeeze_state_head hard-asserts num_heads==1"
    report["config_guard_passed"] = True

    L = rlp.readout_layer_index(model)
    d_state = model.d_state
    vocab_size = model.vocab_size
    report["readout_layer_index"] = L
    report["n_layers"] = model.n_layers
    report["d_state"] = d_state
    assert K_ITEMS <= d_state, f"K_ITEMS={K_ITEMS} must be <= d_state={d_state} for K mutually orthonormal keys"

    device_t = torch.device(device)

    # --- synthetic orthonormal keys, per batch row (B independent trials in one forward pass) ---
    gen = torch.Generator(device=device_t).manual_seed(SEED)
    raw = torch.randn(B_TRIALS, d_state, d_state, generator=gen, device=device_t, dtype=torch.float32)
    Q, _ = torch.linalg.qr(raw)                                    # (B,d_state,d_state), orthonormal columns
    k_items = Q[:, :, :K_ITEMS].transpose(-1, -2).contiguous()      # (B,K,d_state), rows mutually orthonormal
    # single K-cycle successor assignment (CLAUDE.md: never a general permutation for hop-depth constructions)
    v_items = k_items.roll(shifts=-1, dims=1)                       # v_i := k_{(i+1) mod K}

    # sanity: verify orthonormality actually holds before trusting the closed form
    gram = torch.einsum('bkd,bjd->bkj', k_items, k_items)           # (B,K,K)
    eye = torch.eye(K_ITEMS, device=device_t).unsqueeze(0).expand(B_TRIALS, -1, -1)
    ortho_max_dev = (gram - eye).abs().max().item()
    report["key_orthonormality_max_abs_dev"] = ortho_max_dev
    assert ortho_max_dev < 1e-4, f"synthetic keys not orthonormal (max dev {ortho_max_dev}) -- QR construction bug"

    item_positions = torch.arange(K_ITEMS, device=device_t)
    token_ids = torch.randint(0, vocab_size, (B_TRIALS, T_SEQ), device=device_t)

    # --- override hooks (registered FIRST) + production capture hooks (registered SECOND) ---
    h_override_k = model.blocks[L].mixer.k_conv1d.register_forward_hook(
        _make_override_hook(k_items, item_positions))
    h_override_v = model.blocks[L].mixer.v_conv1d.register_forward_hook(
        _make_override_hook(v_items, item_positions))
    h_beta_capture = {}

    def _beta_hook(module, inp, out):
        h_beta_capture["pre_sigmoid"] = out.detach()
    h_beta = model.blocks[L].mixer.b_proj.register_forward_hook(_beta_hook)

    capture_handles, captured = rlp.register_kqv_hooks(model)

    t0 = time.time()
    with torch.no_grad(), rlp.frozen_bias_surgery(model, force_off=True):
        _, final_states = rlp.forward_body(model, token_ids, initial_states=None, need_hidden=False)
    fwd_wall_s = time.time() - t0
    report["forward_wall_s"] = round(fwd_wall_s, 3)

    h_override_k.remove()
    h_override_v.remove()
    h_beta.remove()
    rlp.remove_hooks(capture_handles)

    # --- consistency check: the production capture hooks must have seen exactly what we injected ---
    k_captured_at_L = captured["k"][L]
    v_captured_at_L = captured["v"][L]
    k_capture_dev = (k_captured_at_L[:, item_positions, :].float() - k_items).abs().max().item()
    v_capture_dev = (v_captured_at_L[:, item_positions, :].float() - v_items).abs().max().item()
    k_filler_max = k_captured_at_L[:, K_ITEMS:, :].abs().max().item()
    report["hook_consistency"] = {
        "k_capture_vs_injected_max_abs_dev": k_capture_dev,
        "v_capture_vs_injected_max_abs_dev": v_capture_dev,
        "k_filler_positions_max_abs_value": k_filler_max,
    }
    assert k_capture_dev < 1e-3 and v_capture_dev < 1e-3, (
        "register_kqv_hooks did not see the injected override -- hook ordering/consistency bug "
        "in this control script, not a finding about the production readout.")

    beta_at_L = torch.sigmoid(h_beta_capture["pre_sigmoid"])        # (B,T,1) plain learned gate
    beta_items = beta_at_L[:, item_positions, 0]                    # (B,K)
    report["beta_items_min"] = beta_items.min().item()
    report["beta_items_max"] = beta_items.max().item()
    report["beta_items_mean"] = beta_items.mean().item()
    n_beta_near_zero = int((beta_items < 1e-3).sum().item())
    report["n_beta_near_zero_of"] = f"{n_beta_near_zero} / {beta_items.numel()}"

    # --- PRODUCTION readout: squeeze_state_head + apply_state_power + cosine_and_recovered, verbatim ---
    S_T_production = rlp.squeeze_state_head(final_states[L])        # (B,d,d), UNMODIFIED production code
    S_T_wrong_layout = S_T_production.transpose(-1, -2)              # explicit teeth check (see module docstring)

    per_h = {}
    for h in H_LIST:
        target_h = k_items.roll(shifts=-h, dims=1)                   # (B,K,d): item a's h-hop target = k_{a+h mod K}

        pred_production = rlp.apply_state_power(S_T_production, k_items, h)
        cos_prod, rec_prod = rlp.cosine_and_recovered(pred_production, target_h)

        pred_wrong = rlp.apply_state_power(S_T_wrong_layout, k_items, h)
        cos_wrong, rec_wrong = rlp.cosine_and_recovered(pred_wrong, target_h)

        # negative control: query against the WRONG hop target (h+1's answer) using the
        # production S_T -- must NOT recover, proving the scorer discriminates (has teeth).
        target_wrong_hop = k_items.roll(shifts=-(h + 1), dims=1)
        cos_wronghop, rec_wronghop = rlp.cosine_and_recovered(pred_production, target_wrong_hop)

        per_h[h] = {
            "production_cos_mean": cos_prod.mean().item(),
            "production_cos_min": cos_prod.min().item(),
            "production_recovered_frac": rec_prod.float().mean().item(),
            "deliberately_transposed_cos_mean": cos_wrong.mean().item(),
            "deliberately_transposed_recovered_frac": rec_wrong.float().mean().item(),
            "wrong_hop_target_cos_mean_negative_control": cos_wronghop.mean().item(),
            "wrong_hop_target_recovered_frac_negative_control": rec_wronghop.float().mean().item(),
            "n_scored": int(rec_prod.numel()),
        }
        print(f"  h={h}: production recovered_frac={per_h[h]['production_recovered_frac']:.4f} "
              f"(cos_mean={per_h[h]['production_cos_mean']:.4f}, cos_min={per_h[h]['production_cos_min']:.4f}) | "
              f"deliberately-transposed recovered_frac={per_h[h]['deliberately_transposed_recovered_frac']:.4f} | "
              f"wrong-hop-target (negative control) recovered_frac="
              f"{per_h[h]['wrong_hop_target_recovered_frac_negative_control']:.4f}", flush=True)

    report["per_h"] = per_h

    # --- independent reference cross-check of S_T itself (second, independently-implemented ground truth) ---
    # Reconstruct the EXACT bf16-quantized (k,v,beta) sequence the kernel actually consumed (K items +
    # zero filler), transpose to the reference's (b,h,l,d) layout, run the audited reference recurrence.
    k_full = torch.zeros(B_TRIALS, T_SEQ, d_state, device=device_t, dtype=torch.float32)
    v_full = torch.zeros(B_TRIALS, T_SEQ, d_state, device=device_t, dtype=torch.float32)
    beta_full = torch.zeros(B_TRIALS, T_SEQ, device=device_t, dtype=torch.float32)
    k_full[:, item_positions, :] = k_items
    v_full[:, item_positions, :] = v_items
    beta_full[:, item_positions] = beta_items
    # match the kernel's own bf16 boundary + use_qk_l2norm_in_kernel=True (l2norm is a no-op on
    # already-unit-norm rows up to bf16 rounding, applied for exactness against the real call).
    k_bf = F.normalize(k_full, dim=-1).to(torch.bfloat16).float()
    v_bf = v_full.to(torch.bfloat16).float()
    beta_bf = beta_full.to(torch.bfloat16).float()
    # (b,t,d) -> (b,h=1,t,d)
    to_ref_layout = lambda x: x.unsqueeze(1)
    _, S_ref = patched_delta_rule_recurrence(
        to_ref_layout(k_bf), to_ref_layout(k_bf), to_ref_layout(v_bf), to_ref_layout(beta_bf),
        output_final_state=True)
    S_ref = S_ref[:, 0, :, :]                                          # (B,d,d), fla's own [K,V] layout
    # NOTE: patched_delta_rule_recurrence pre-scales q by d_k**-0.5 (fla's own convention) -- irrelevant
    # here since we only use its own S output, never its o output, and S's recurrence does not use q at all.
    kernel_vs_reference_rel_fro = ((S_T_production - S_ref).norm() / S_ref.norm().clamp(min=1e-12)).item()
    report["kernel_vs_independent_reference_S_rel_fro_err"] = kernel_vs_reference_rel_fro
    report["kernel_vs_independent_reference_note"] = (
        "diagnostic only, NOT part of overall_pass -- a second, independently-implemented ground "
        "truth for S_T (f15_lm_checkpoint.py's own audited reference), cross-checked here for extra "
        "rigor; excluded from the blocking verdict since a mismatch here could also reflect this "
        "script's own normalize/cast-order choices rather than the production readout under test. "
        "reported for transparency; expect < 2e-2 relative Frobenius error per F15's own precedent.")

    # --- verdict (matches the rebuttal's literal ask: "confirm the production extraction path "
    # "recovers it at cosine >= 0.9" -- gated on the SAME recovered_frac/cosine convention the paper
    # itself reports, aggregated over the K*B synthetic cells per h) ---
    threshold = rlp.RECOVERY_COS_THRESHOLD
    report["recovery_cos_threshold"] = threshold
    production_pass = all(per_h[h]["production_recovered_frac"] >= threshold for h in H_LIST)
    negative_controls_ok = all(
        per_h[h]["wrong_hop_target_recovered_frac_negative_control"] < 0.05 for h in H_LIST)
    report["verdict"] = {
        "production_readout_recovers_known_signal": bool(production_pass),
        "negative_controls_have_teeth": bool(negative_controls_ok),
        "overall_pass": bool(production_pass and negative_controls_ok),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "reasoning_link_poscontrol_result.json"))
    args = ap.parse_args()

    print("=" * 78)
    print("  reasoning_link_poscontrol -- pre-submission real-kernel positive control")
    print("  (gauntlet round-1 FIX-6 tier 2 / final-review decision (b))")
    print("=" * 78)
    print(f"  checkpoint: {args.ckpt}")

    report = run_positive_control(args.ckpt, device=args.device)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 78)
    v = report["verdict"]
    overall = "PASS" if v["overall_pass"] else "FAIL"
    print(f"  VERDICT: {overall}")
    for k, val in v.items():
        print(f"    {k}: {val}")
    print(f"  wrote {args.out}")
    print("=" * 78)
    sys.exit(0 if v["overall_pass"] else 1)


if __name__ == "__main__":
    main()
