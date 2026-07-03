"""FLOP accounting — STAGE_G_DESIGN.md section 5.1 (analytic table, verified
by V1) and build requirement (iii): "no by-hand FLOP number in this program
is load-bearing until independently re-derived or instrumented."

Two independent counts, meant to be compared against each other (Wave -1
item 1), never trusted alone:

  analytic_matrix_flops / analytic_loopformer_flops
      Reproduce section 5.1's per-component formulas (2*MACs convention)
      as functions of the actual config, so they can be evaluated at BOTH
      the original Regime-1 config (d=32, vocab=50257, L=512) AND Stage
      G's own Regime-2 config (d=32, vocab=256) -- section 5.1's own
      numbers are reproduced at the Regime-1 config as a self-check (see
      test_stageg.py).

  instrumented_flops_per_token
      A live count via torch.utils.flop_counter.FlopCounterMode (stdlib
      torch, no hand arithmetic) over one real forward pass -- the
      "profiler-based, not by-hand" corroboration Wave -1's manifest item 1
      requires. FlopCounterMode only sees FLOPs from ATen ops it has
      registered formulas for (matmul/addmm/bmm/conv/etc.); einsum lowers
      to bmm internally so is captured automatically.
"""
from __future__ import annotations

import torch


# ═══════════════════════════════════════════════════════════════════════════
# Analytic tables (section 5.1)
# ═══════════════════════════════════════════════════════════════════════════

def analytic_matrix_flops(d, L, n_layers, n_iterations, vocab, K, causal_exact=False,
                           n_probes_extra=True, proj_flops=None):
    """Returns (total_flops_per_token, breakdown_dict).

    causal_exact=False: "non-causal convention" (unmasked L) -- matches
    section 5.1's headline 230.6M-equivalent number.
    causal_exact=True: halves the attention-score L-dependent term (the
    footnote's 2x over-count correction under causal masking) -- matches
    the 11.8x causal-exact ratio.
    proj_flops: per-projection FLOP override (H_b's factored family:
    4*d^2*r per projection); default None -> RowThenColProjection's 4d^3.
    """
    eff_L = L / 2 if causal_exact else L
    proj = proj_flops if proj_flops is not None else 4 * d ** 3   # RowThenCol: 2 matmuls d^3 MACs -> 4d^3 FLOPs
    attn = 4 * proj + 4 * eff_L * d ** 2                 # 4 projections (Q,K,V,O) + SDPA score+softmax@V
    mult = 6 * proj + 4 * d ** 3                          # 6 projections + 2 matmuls for M_mult
    per_block = attn + mult
    backbone = per_block * n_layers * n_iterations

    # MultiProbeHead-equivalent extraction cost (probes exist for
    # multiprobe AND both tied forms; tied forms additionally reuse or
    # replace the final vocab GEMM -- callers pass K=0 for heads with no
    # probe-extraction stage, e.g. a pure bilinear tie).
    probe_extract = (2 * d ** 2 * K + 2 * d * K) if n_probes_extra else 0
    out_gemm = 2 * K * vocab                              # Linear(K, vocab); 0 for tied heads (see below)
    head = probe_extract + out_gemm

    embed = 4 * d ** 2                                    # outer product + pos-outer-product, both ~2d^2

    total = backbone + head + embed
    return total, {"attn_per_block": attn, "mult_per_block": mult, "per_block": per_block,
                    "backbone": backbone, "head": head, "embed": embed, "total": total,
                    "eff_L": eff_L}


def analytic_matrix_tied_bilinear_head_flops(d, vocab):
    """H_f form (i): logit(w) = tau * (u_w^T M v_w) for every vocab row --
    O(vocab*d^2) MACs per token (the design's own item-4 cost note:
    "cost O(B*L*vocab*d^2) per naive einsum is trivial at vocab=256")."""
    return 2 * vocab * d ** 2   # (B,L,d,d)x(vocab,d)->(B,L,vocab,d) contraction, then reduce over d


def analytic_loopformer_flops(n_embd, n_head, n_layer, n_loops, intermediate_dim, L, vocab,
                               freq_dim=256, causal_exact=False):
    eff_L = L / 2 if causal_exact else L
    qkvo = 8 * n_embd ** 2                                # 2*(3*n_embd^2 + n_embd^2) MACs*2 for FLOPs
    score = 4 * eff_L * n_embd
    mlp = 4 * n_embd * intermediate_dim
    per_block = qkvo + score + mlp
    backbone = per_block * n_layer * n_loops

    adaLN_per_block = 2 * n_embd * (4 * n_embd)           # Linear(n_embd, 4n_embd), 2*MACs
    timestep_embedder = 2 * (2 * freq_dim * n_embd + 2 * n_embd * n_embd)  # 2 linears x 2 embedders
    cond_total_per_step = adaLN_per_block * n_layer + timestep_embedder
    amortized_cond = (cond_total_per_step * n_loops) / L   # amortized over the L tokens in the batch item

    head = 2 * n_embd * vocab                              # tied lm_head
    total = backbone + amortized_cond + head
    return total, {"per_block": per_block, "backbone": backbone,
                    "amortized_cond": amortized_cond, "head": head, "total": total, "eff_L": eff_L}


def analytic_flops_for_cell(family, variant, mat_dim, seq_len, n_iterations, vocab_size,
                             n_embd=80, n_layer=2, intermediate_dim=128):
    """Per-cell analytic FLOPs/token for a (family, variant) training cell --
    design build requirement (ii) via audit MAJOR-3(c): every result JSON
    carries an analytic_flops_per_token field. Returns
    {"non_causal": float, "causal_exact": float}.

    Variant adjustments (matrix side) are derived from models.VARIANT_AXES
    -- not name-matching -- so newly registered variants (the N3
    follow-up, the section-9 combined probe) inherit correct counts
    automatically: n_layers from the variant's override (depth-matched
    cells -> 2); a tied_bilinear output head replaces MultiProbeHead's
    cost with the exact-bilinear head cost (init scale changes no FLOPs).
    H_i's per-iteration conditioning is computed once per (batch,
    iteration) and amortizes over L to near-zero per token (same argument
    as section 5.1's AdaLN correction, V1) -- not counted. H_a's full-rank
    embed changes only the embedding lookup term (~2d^2 either way) --
    not adjusted."""
    if family == "matrix":
        from models import VARIANT_AXES   # deferred import: keeps flops.py usable standalone
        axes = VARIANT_AXES.get(variant, {})
        n_layers = axes.get("n_layers", 8)
        tied_head = axes.get("output_head") == "tied_bilinear"
        proj_rank = axes.get("proj_rank")
        proj_flops = 4 * mat_dim ** 2 * proj_rank if proj_rank else None   # H_b factored: 2 GEMMs d^2<->r
        out = {}
        for key, causal in (("non_causal", False), ("causal_exact", True)):
            total, bd = analytic_matrix_flops(mat_dim, seq_len, n_layers, n_iterations,
                                               vocab_size, K=mat_dim, causal_exact=causal,
                                               proj_flops=proj_flops)
            if tied_head:
                total = (bd["backbone"] + bd["embed"]
                          + analytic_matrix_tied_bilinear_head_flops(mat_dim, vocab_size))
            out[key] = total
        return out
    from models import VECTOR_VARIANT_AXES   # deferred import (see matrix branch)
    n_embd = VECTOR_VARIANT_AXES.get(variant, {}).get("n_embd", n_embd)   # capacity_782k override
    out = {}
    for key, causal in (("non_causal", False), ("causal_exact", True)):
        total, _ = analytic_loopformer_flops(n_embd, 4, n_layer, n_iterations,
                                              intermediate_dim, seq_len, vocab_size,
                                              causal_exact=causal)
        out[key] = total
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Instrumented count (Wave -1's final corroboration)
# ═══════════════════════════════════════════════════════════════════════════

def instrumented_flops_per_token(model, token_ids, forward_kwargs=None, backward=False):
    """FlopCounterMode-based live count (torch stdlib). Returns
    (flops_per_token, raw_total_flops, breakdown_dict).

    backward=False: forward-only (matches section 5.1's convention, which
    is stated as "forward pass, per token").

    Deliberately does NOT call FlopCounterMode.get_table() -- that method
    imports the third-party `tabulate` package, violating this build's
    "Torch+stdlib only" constraint (and not guaranteed present on the
    training box). The per-op breakdown is read directly off
    `fcm.flop_counts` (a plain dict of Counters, no extra dependency)."""
    from torch.utils.flop_counter import FlopCounterMode

    forward_kwargs = forward_kwargs or {}
    B, L = token_ids.shape
    model.eval() if not backward else model.train()

    with FlopCounterMode(display=False) as fcm:
        out = model(token_ids, **forward_kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        if backward:
            logits.float().sum().backward()
    total_flops = fcm.get_total_flops()
    per_token = total_flops / (B * L)

    breakdown = {}
    for module_name, op_counts in fcm.flop_counts.items():
        breakdown[module_name] = {str(op): int(v) for op, v in op_counts.items()}
    return per_token, total_flops, breakdown
