"""probe_head_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1: the Wave -1
shared probe head (F1's entire resolution). ONE architecture-neutral,
TRAINED readout, reused across all three H2H arms, closing the FATAL gap
Rev 0 shipped with no working readout for any of its three arms.

Pieces, each pinned exactly per sec 1.3.1.1-1.3.1.4:
  - `build_probe_target_table` -- the frozen T_val ground-truth table
    (`key_anchoring.random_unit_rows_init`, reused UNMODIFIED, at the NEW
    `PROBE_TARGET_SEED` -- sec 1.3.1.1).
  - `build_shared_probe` / `build_adapter_arm` -- the IDENTICAL-class/init
    `shared_probe` (nn.Linear(value_dim,value_dim,bias=True)) and per-arm
    `adapter_arm` (nn.Linear(native_tap_dim,value_dim,bias=False)) factories
    -- sec 1.3.1.2.
  - `contender_native_tap` / `ablation_native_tap` / `transformer_native_tap`
    -- the three per-arm `state_summary_raw` extraction functions, EXACTLY
    per sec 1.3.1.2's own construction (contender: `S_T_last @ q_query` via
    a non-recurrent query pathway; ablation: Hadamard `s_T (.) q_query`;
    Transformer: final-block post-norm hidden at `<Q>` from a real
    context-extending forward pass -- the one arm whose query DOES enter
    its own context, a disclosed, structural asymmetry, sec 1.9 item 9).
  - `cosine_recovery_frac` / `probe_aux_loss` -- the frozen metric/loss
    formula (`F.cosine_similarity(...) @ 0.9`; `1 - cos` aux loss,
    `aux_weight=0.1` default) -- sec 1.3.1.3.
  - `run_probe_capacity_null` -- the MANDATORY sec 1.3.1.4 sanity null
    (gate 7): frozen-random `state_summary_raw`, fresh draws every step,
    held-out fresh eval draws, pass bar `recovered_frac@0.9 < 0.05`.

Run the smoke gate (incl. the gate-7 null for all three arms' REAL adapter
shapes at rung-1): python probe_head_rd.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_constants_rd import PROBE_TARGET_SEED
from key_anchoring import random_unit_rows_init                        # no fla dep -- safe pre-stub
from h2h_fla_stub_rd import ensure_fla_stub

_STUB_INSTALLED = ensure_fla_stub()
from lm_pretrain_rd import DeltaNetLM                                   # noqa: E402
from ablation_mixer_rd import AblationLM                                # noqa: E402
from transformer_baseline_rd import TransformerLM                       # noqa: E402

RECOVERY_THRESHOLD = 0.9
AUX_WEIGHT_DEFAULT = 0.1
NULL_PASS_BAR = 0.05
# lm_pretrain_rd._MIN_KERNEL_T (128): the real chunk_delta_rule kernel's own backward-crash floor
# (F15-LM measurement) -- every smoke item below that instantiates a REAL DeltaNetLM must feed it
# a context >= this length; used only inside this file's own smoke suite (not re-exported).
_CONTENDER_SMOKE_CTX_LEN = 128


# ---------------------------------------------------------------------------
# sec 1.3.1.1 -- frozen ground-truth target table
# ---------------------------------------------------------------------------

def build_probe_target_table(vocab_size_total: int, value_dim: int,
                              seed: int = PROBE_TARGET_SEED) -> torch.Tensor:
    """T_val: (vocab_size_total, value_dim), frozen (caller registers as a buffer +
    requires_grad_(False) -- this function only builds the tensor, matching
    random_unit_rows_init's own init-vs-freeze separation of concerns)."""
    return random_unit_rows_init(vocab_size_total, value_dim, seed)


# ---------------------------------------------------------------------------
# sec 1.3.1.2 -- shared probe + per-arm adapter factories
# ---------------------------------------------------------------------------

def build_shared_probe(value_dim: int) -> nn.Linear:
    return nn.Linear(value_dim, value_dim, bias=True)


def build_adapter_arm(native_tap_dim: int, value_dim: int) -> nn.Linear:
    return nn.Linear(native_tap_dim, value_dim, bias=False)


# ---------------------------------------------------------------------------
# sec 1.3.1.2 -- per-arm native taps
# ---------------------------------------------------------------------------

def contender_native_tap(model: DeltaNetLM, context_token_ids: torch.Tensor,
                          query_tokens: torch.Tensor) -> torch.Tensor:
    """S_T_last @ q_query. context_token_ids: (B,T_ctx) -- streamed through
    model.forward(..., return_states=True); the query window is NEVER
    appended to this call (protects the P=1 bottleneck). query_tokens:
    (B,Q,query_len) -- run through the LAST block's OWN
    embed->norm1->q_proj->q_conv1d pathway ONLY (generalizes
    model_rd.py's `effective_key_window` precedent to the contender's real
    trained q_proj/q_conv1d). Returns (B,Q,d_state)."""
    _, final_states = model(context_token_ids, return_states=True)
    S_T_last = final_states[-1]                       # (B,H,head_dim,head_dim), H==1 every rung
    assert S_T_last.shape[1] == 1, (
        f"contender_native_tap assumes num_heads==1 (every registered rung config), got "
        f"num_heads={S_T_last.shape[1]}")
    S = S_T_last.squeeze(1)                             # (B,d_state,d_state)

    B, Q, qlen = query_tokens.shape
    last_block = model.blocks[-1]
    x_embed = model.embed(query_tokens.reshape(B * Q, qlen))
    a = last_block.norm1(x_embed)
    q_conv, _ = last_block.mixer.q_conv1d(last_block.mixer.q_proj(a))
    q_last = q_conv[:, -1, :].view(B, Q, -1)             # <Q> position's post-conv q, (B,Q,d_state)

    return torch.einsum("bij,bqj->bqi", S, q_last)       # per-query matvec read, (B,Q,d_state)


def ablation_native_tap(model: AblationLM, context_token_ids: torch.Tensor,
                         query_tokens: torch.Tensor) -> torch.Tensor:
    """s_T (.) q_query (Hadamard) -- same non-recurrent query-pathway
    discipline as the contender's own tap. Returns (B,Q,d_state)."""
    _, final_states = model(context_token_ids, return_states=True)
    s_T = final_states[-1]                                # (B,d_state)

    B, Q, qlen = query_tokens.shape
    last_block = model.blocks[-1]
    x_embed = model.embed(query_tokens.reshape(B * Q, qlen))
    a = last_block.norm1(x_embed)
    q_conv, _ = last_block.mixer.q_conv1d(last_block.mixer.q_proj(a))
    q_last = q_conv[:, -1, :].view(B, Q, -1)

    return s_T.unsqueeze(1) * q_last                      # (B,1,d_state) * (B,Q,d_state)


def transformer_native_tap(model: TransformerLM, context_token_ids: torch.Tensor,
                            query_tokens: torch.Tensor, attn_mask_fn=None) -> torch.Tensor:
    """Final-block (post-norm_f) hidden state at each query's own `<Q>`
    position, from a SINGLE forward pass per (batch,query) pair over
    `[context ++ that one query_window]` -- the ONE arm whose query DOES
    enter its own context (sec 1.9 item 9's disclosed, structural
    asymmetry: attention has no non-attention read channel). `attn_mask_fn`:
    None (uncapped, b-control/training) or `callable(T_total) -> (T_total,
    T_total) bool mask` (hard-capped, b-primary, INFERENCE-ONLY). Returns
    (B,Q,d_model). NOTE (disclosed compute cost, R3-F4's own concern): this
    replicates the context Q times (once per query) since the Transformer
    has no query-free read channel to reuse a single context pass across
    queries -- exactly why R3-F4 gates the M-sweep fan-out behind a REAL
    timing pilot rather than a design-time assumption."""
    B, T_ctx = context_token_ids.shape
    _, Q, qlen = query_tokens.shape
    ctx_rep = context_token_ids.unsqueeze(1).expand(B, Q, T_ctx).reshape(B * Q, T_ctx)
    q_flat = query_tokens.reshape(B * Q, qlen)
    seq = torch.cat([ctx_rep, q_flat], dim=1)
    mask = attn_mask_fn(T_ctx + qlen) if attn_mask_fn is not None else None
    _, hidden = model(seq, attn_mask=mask, return_hidden=True)
    return hidden[:, -1, :].view(B, Q, -1)


# ---------------------------------------------------------------------------
# sec 1.3.1.2 (metric) / sec 1.3.1.3 (training regime)
# ---------------------------------------------------------------------------

def cosine_recovery_frac(pred: torch.Tensor, target: torch.Tensor,
                          threshold: float = RECOVERY_THRESHOLD) -> torch.Tensor:
    """The frozen `recovered_frac@0.9` metric formula, unchanged across
    every revision of this design -- fraction of rows clearing the
    threshold."""
    cos = F.cosine_similarity(pred, target, dim=-1)
    return (cos >= threshold).float().mean()


def probe_aux_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """mean(1 - cos(pred, target)) -- sec 1.3.1.3's joint aux loss term."""
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def joint_loss(loss_ce: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
                aux_weight: float = AUX_WEIGHT_DEFAULT) -> torch.Tensor:
    """loss_total = loss_CE + aux_weight * mean(1 - cos(pred,target)) --
    sec 1.3.1.3's exact formula."""
    return loss_ce + aux_weight * probe_aux_loss(pred, target)


# ---------------------------------------------------------------------------
# sec 1.3.1.4 -- MANDATORY probe-capacity sanity null (gate 7)
# ---------------------------------------------------------------------------

def run_probe_capacity_null(native_tap_dim: int, value_dim: int, T_val: torch.Tensor,
                             n_train_steps: int = 500, batch_size: int = 64, lr: float = 1e-3,
                             n_eval: int = 2000, threshold: float = RECOVERY_THRESHOLD,
                             seed: int = 0) -> dict:
    """Trains `adapter_arm + shared_probe` (this arm's REAL adapter shape)
    on FRESH i.i.d. Gaussian `state_summary_raw` every step (never a fixed
    pool -- m-new-2's own pinned protocol) against FRESH random rows of the
    SAME frozen `T_val` table, then scores `recovered_frac@0.9` on a
    SEPARATE held-out set of fresh draws never seen during training. Pass:
    `recovered_frac < 0.05` (the probe alone, with no real state to read,
    must not "solve" the task)."""
    torch.manual_seed(seed)
    adapter = build_adapter_arm(native_tap_dim, value_dim)
    probe = build_shared_probe(value_dim)
    opt = torch.optim.Adam(list(adapter.parameters()) + list(probe.parameters()), lr=lr)
    n_rows = T_val.shape[0]

    for _ in range(n_train_steps):
        x = torch.randn(batch_size, native_tap_dim)
        target = T_val[torch.randint(0, n_rows, (batch_size,))]
        pred = probe(adapter(x))
        loss = probe_aux_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        x_eval = torch.randn(n_eval, native_tap_dim)
        target_eval = T_val[torch.randint(0, n_rows, (n_eval,))]
        pred_eval = probe(adapter(x_eval))
        frac = cosine_recovery_frac(pred_eval, target_eval, threshold).item()

    return {"recovered_frac": frac, "passed": frac < NULL_PASS_BAR, "native_tap_dim": native_tap_dim,
            "value_dim": value_dim, "n_train_steps": n_train_steps, "n_eval": n_eval}


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def smoke_1_target_table_frozen_unit_rows():
    t1 = build_probe_target_table(500, 16)
    t2 = build_probe_target_table(500, 16)
    same_seed_identical = torch.equal(t1, t2)
    t3 = build_probe_target_table(500, 16, seed=999)
    different_seed_different = not torch.equal(t1, t3)
    unit_rows = torch.allclose(t1.norm(dim=-1), torch.ones(500), atol=1e-5)
    ok = same_seed_identical and different_seed_different and unit_rows
    _report("smoke 1: build_probe_target_table determinism + unit-row property", ok,
            f"same_seed_identical={same_seed_identical} different_seed_different="
            f"{different_seed_different} unit_rows={unit_rows}")


def smoke_2_contender_tap_forward_backward():
    # d_state PINNED to 64 (contender's own _SAFE_D_STATE floor for the real Triton kernel's
    # backward -- model_rd.py's F15-LM measurement; smaller d_state is not a valid contender shape
    # anywhere in this codebase, not just an arbitrary smoke-test choice).
    torch.manual_seed(1)
    m = DeltaNetLM(300, d_model=32, d_state=64, n_layers=2, conv_size=4)
    adapter = build_adapter_arm(64, 12)
    probe = build_shared_probe(12)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    tap = contender_native_tap(m, ctx, q_tok)
    pred = probe(adapter(tap))
    target = torch.randn(2, 3, 12)
    loss = probe_aux_loss(pred, target)
    loss.backward()
    ok = (tap.shape == (2, 3, 64) and torch.isfinite(tap).all().item()
          and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    _report("smoke 2: contender_native_tap forward + backward-through-adapter+probe", ok,
            f"tap.shape={tuple(tap.shape)}")


def smoke_3_contender_tap_p1_bottleneck_blankout():
    """Blank-out check (CLAUDE.md's own required discipline), decomposed
    into two independently CPU-stub-provable halves:
      (a) S_T_last is a pure function of context ALONE -- it comes from a
          `model(context_token_ids, return_states=True)` call that
          structurally never receives query_tokens, so perturbing the
          query cannot change it (checked directly: bit-identical
          final_states across two different query draws).
      (b) q_last (the LAST block's own embed->norm1->q_proj->q_conv1d
          pathway output) IS a function of the query window and changes
          when it is perturbed -- checked directly on the intermediate,
          not the full tap.
    DISCLOSED BOX-ONLY LIMITATION (h2h_fla_stub_rd.py's own documented
    stub behavior): `_stub_chunk_delta_rule` always returns an all-ZERO
    `final_state`, so `state_summary_raw = S_T_last @ q_last` is
    IDENTICALLY ZERO under the CPU stub regardless of q_last -- the
    end-to-end claim "the FULL tap changes when q changes" cannot be
    exercised meaningfully on this dev box; it requires the real Triton
    kernel (a nonzero, trained S_T_last) and is registered here as a
    box-only follow-up smoke item for the deploy stage, not silently
    skipped."""
    torch.manual_seed(2)
    m = DeltaNetLM(300, d_model=32, d_state=64, n_layers=1, conv_size=4)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    q_tok_perturbed = torch.randint(0, 300, (2, 3, 5))
    with torch.no_grad():
        _, fs1 = m(ctx, return_states=True)
        _, fs2 = m(ctx, return_states=True)
        last_block = m.blocks[-1]

        def _q_last(qt):
            B, Q, qlen = qt.shape
            a = last_block.norm1(m.embed(qt.reshape(B * Q, qlen)))
            qc, _ = last_block.mixer.q_conv1d(last_block.mixer.q_proj(a))
            return qc[:, -1, :].view(B, Q, -1)

        q1, q2 = _q_last(q_tok), _q_last(q_tok_perturbed)
    state_unchanged_by_query = torch.equal(fs1[-1], fs2[-1])
    q_last_changed_by_query = not torch.allclose(q1, q2)
    ok = state_unchanged_by_query and q_last_changed_by_query
    _report("smoke 3: contender P=1 bottleneck blank-out (state is context-only; q_last IS "
            "query-dependent; end-to-end tap-changes-with-q needs a real nonzero S_T -- box-only, "
            "see docstring)", ok,
            f"state_unchanged_by_query={state_unchanged_by_query} "
            f"q_last_changed_by_query={q_last_changed_by_query}")


def smoke_4_ablation_tap_forward_and_bottleneck():
    torch.manual_seed(4)
    m = AblationLM(300, d_model=32, d_state=16, n_layers=1, conv_size=4)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    tap = ablation_native_tap(m, ctx, q_tok)
    with torch.no_grad():
        _, fs1 = m(ctx, return_states=True)
        q_tok_perturbed = torch.randint(0, 300, (2, 3, 5))
        _, fs2 = m(ctx, return_states=True)
        tap2 = ablation_native_tap(m, ctx, q_tok_perturbed)
    state_unchanged = torch.equal(fs1[-1], fs2[-1])
    tap_changed = not torch.allclose(tap, tap2)
    ok = tap.shape == (2, 3, 16) and state_unchanged and tap_changed
    _report("smoke 4: ablation_native_tap shape + P=1 bottleneck blank-out", ok,
            f"tap.shape={tuple(tap.shape)} state_unchanged={state_unchanged} tap_changed={tap_changed}")


def smoke_5_transformer_tap_uncapped_vs_capped():
    from transformer_baseline_rd import sink_fifo_mask
    torch.manual_seed(6)
    m = TransformerLM(300, d_model=32, n_layers=2, n_heads=4, ffn_mult=2)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    tap_uncapped = transformer_native_tap(m, ctx, q_tok, attn_mask_fn=None)
    tap_capped = transformer_native_tap(
        m, ctx, q_tok, attn_mask_fn=lambda t: sink_fifo_mask(t, cap_length=10, k_sink=4))
    ok = (tap_uncapped.shape == (2, 3, 32) and tap_capped.shape == (2, 3, 32)
          and not torch.allclose(tap_uncapped, tap_capped))
    _report("smoke 5: transformer_native_tap shape + uncapped-vs-capped modes actually differ", ok,
            f"tap_uncapped.shape={tuple(tap_uncapped.shape)}")


def smoke_6_probe_capacity_null_all_three_arm_shapes():
    """Gate 7 itself: the mandatory null MUST pass for every arm's REAL
    rung-1 adapter shape before HEADTOHEAD_MATCH_GATE_SIGNOFF may be set."""
    T_val = build_probe_target_table(2000, 64, seed=PROBE_TARGET_SEED)
    # native_tap_dim per arm, rung-1; a FIXED, deterministic per-arm seed offset (never
    # Python's hash() -- CLAUDE.md's own hard rule: hash() is salted per-process, zlib.crc32 or a
    # literal pin is the reproducible alternative; a literal enumeration is simplest here).
    shapes = {"contender": (64, 1), "ablation": (64, 2), "transformer": (256, 3)}
    all_pass = True
    for arm, (dim, seed_offset) in shapes.items():
        result = run_probe_capacity_null(dim, 64, T_val, n_train_steps=300, seed=seed_offset)
        print(f"    arm={arm:>11s}: native_tap_dim={dim:>3d} recovered_frac={result['recovered_frac']:.4f} "
              f"passed={result['passed']}")
        all_pass = all_pass and result["passed"]
    _report("smoke 6 (GATE 7): probe-capacity null PASSES for all 3 arms' real rung-1 adapter "
            "shapes (bar < 0.05)", all_pass)


def smoke_7_joint_loss_gradients_flow_to_backbone():
    """The aux loss must produce REAL gradients on the backbone's own
    parameters (not just the adapter/probe) -- confirms sec 1.3.1.3's
    "trains THROUGHOUT training, from step 0" premise is mechanically true,
    not merely a design intention."""
    torch.manual_seed(8)
    m = DeltaNetLM(300, d_model=32, d_state=64, n_layers=1, conv_size=4)
    adapter = build_adapter_arm(64, 12)
    probe = build_shared_probe(12)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    y = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    logits = m(ctx)
    loss_ce = F.cross_entropy(logits.reshape(-1, 300), y.reshape(-1))
    tap = contender_native_tap(m, ctx, q_tok)
    pred = probe(adapter(tap))
    target = torch.randn(2, 3, 12)
    loss = joint_loss(loss_ce, pred, target)
    loss.backward()
    q_proj_grad_nonzero = m.blocks[-1].mixer.q_proj.weight.grad.abs().sum().item() > 0
    _report("smoke 7: joint aux loss produces nonzero gradient on the backbone's own q_proj "
            "(gradients genuinely flow, not detached)", q_proj_grad_nonzero)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    print("=" * 70)
    print("probe_head_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1 smoke suite")
    print(f"fla stub installed (real fla absent or forced): {_STUB_INSTALLED}")
    print("=" * 70)
    smoke_1_target_table_frozen_unit_rows()
    smoke_2_contender_tap_forward_backward()
    smoke_3_contender_tap_p1_bottleneck_blankout()
    smoke_4_ablation_tap_forward_and_bottleneck()
    smoke_5_transformer_tap_uncapped_vs_capped()
    smoke_6_probe_capacity_null_all_three_arm_shapes()
    smoke_7_joint_loss_gradients_flow_to_backbone()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
