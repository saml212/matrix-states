"""transformer_baseline_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3(b): the
standard, "modern-strong" Transformer baseline, used in TWO inference roles
from ONE trained arch (no extra training cost): role (b-control), full/
uncapped attention (the disclosed FLOP-matched control); role (b-primary),
axis 2's baseline, a hard-capped KV cache (sink+FIFO, k_sink=4) implemented
as a windowed-attention MASK over an otherwise-identical forward pass
(sec 1.17's own "verified clean" list: "sink+FIFO spec buildable as a
windowed-attention mask" -- the eviction policy is INFERENCE-ONLY config;
training is ALWAYS uncapped, per M-NEW-4's binding budget constraint).

Recipe, pinned exactly (sec 1.3(b), M3):
  - RoPE (rotary), not learned-absolute -- "modern-strong" AND avoids axis
    2's 8x-cap_length horizons exceeding a learned-absolute position ceiling.
  - Pre-norm, matching the contender's own DeltaNetLMBlock convention.
  - FFN: `lm_pretrain_rd.FFN`, reused VERBATIM (not reimplemented) -- same
    plain 2-matrix GELU MLP, mult=4, removing one entire arch-vs-arch
    asymmetry axis.
  - GPT-2-convention embedding init (std=0.02), tied input/output embedding.
  - `n_layers_transformer` PINNED to 2 (R3-F3, `PINNED_N_LAYERS_TRANSFORMER`
    below) -- matches the contender's own n_layers=2 at rung-1.

`d_model=256` is this build's own pinned DEFAULT (not re-derived from
scratch here -- verify_match_gate.py's Pass 1/Pass 2 are the actual sec 1.7
gate 6 authority): at `d_model=256, n_layers=2, mult=4`, this build's own
closed-form check (mirrored independently in `verify_match_gate.py`) gives
FLOPs/token ~2.78% off the contender's own measured 8.43e7/token figure
(sec 4.2's head-dominated method) -- inside the <=5% band -- and reproduces
the design doc's own worked `cap_length(M)=M*8` table EXACTLY
(cap_length(1)=8, matching sec 1.4.2's own arithmetic verbatim), so this
value is not an independent guess, it is the SAME `d_model` the design
doc's own prose derivation already assumed throughout sec 1.4.2.

Run the smoke gate: python transformer_baseline_rd.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_fla_stub_rd import ensure_fla_stub

_STUB_INSTALLED = ensure_fla_stub()
from fla.modules import RMSNorm  # noqa: E402

PINNED_N_LAYERS_TRANSFORMER = 2      # R3-F3
PINNED_K_SINK = 4                    # F-NEW-2
CONTENDER_TOTAL_STATE_BYTES_RUNG1 = 32_768   # 2 layers x 64x64 x 4 bytes fp32 (sec 1.2/1.3(b), M2)
BYTES_PER_ELT_FP32 = 4
ROPE_BASE_DEFAULT = 10_000.0


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """Standard rotary cache: cos/sin, each (seq_len, head_dim) (the
    "duplicated-half" LLaMA-style layout, paired with `rotate_half`
    below)."""
    assert head_dim % 2 == 0, f"head_dim={head_dim} must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                      # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B,H,T,head_dim); cos/sin: (T,head_dim), broadcast over B,H."""
    return x * cos.view(1, 1, *cos.shape) + _rotate_half(x) * sin.view(1, 1, *sin.shape)


# ---------------------------------------------------------------------------
# Sink+FIFO capped-attention mask (F-NEW-2) -- INFERENCE-ONLY config, never used in training.
# ---------------------------------------------------------------------------

def sink_fifo_mask(seq_len: int, cap_length: int, k_sink: int = PINNED_K_SINK,
                    device=None) -> torch.Tensor:
    """Boolean (T,T) attention-allowed mask, True == query i MAY attend to
    key j. Causal (j<=i) AND (j is one of the first k_sink "sink" tokens OR
    j is within the most-recent (cap_length-k_sink) tokens before/at i) --
    the exact byte-budget-equivalent of a real deployed sink+FIFO KV cache
    evaluated via ONE forward pass (sec 1.17's own "verified clean" note).
    Requires cap_length > k_sink (FIFO window >= 1, so every query can
    always see AT LEAST itself) -- the pinned rung-1 grid never needs
    cap_length <= k_sink (cap_length(M=1)=8 > k_sink=4), so this is a
    disclosed, asserted precondition, not a silently-handled edge case."""
    assert cap_length > k_sink, (
        f"cap_length={cap_length} must be > k_sink={k_sink} (FIFO window must be >= 1 token so "
        f"every query position can see at least itself) -- this design's own pinned grid never "
        f"needs cap_length <= k_sink; a caller reaching this is out of the design's own scope.")
    fifo_window = cap_length - k_sink
    idx = torch.arange(seq_len, device=device)
    i = idx.view(seq_len, 1)
    j = idx.view(1, seq_len)
    causal = j <= i
    is_sink = j < k_sink
    is_recent = j > (i - fifo_window)
    return causal & (is_sink | is_recent)


def cap_length_tokens(m_multiplier: int, n_layers_transformer: int, d_model: int,
                       bytes_per_elt: int = BYTES_PER_ELT_FP32,
                       contender_state_bytes: int = CONTENDER_TOTAL_STATE_BYTES_RUNG1) -> int:
    """sec 1.4.2's own formula: cap_length(M) = M * contender_total_state_bytes /
    (2 (K&V) * n_layers_transformer * d_model * bytes_per_elt). Integer
    division is EXACT for every pinned grid point at this build's own
    config (verified by the smoke suite against the design doc's own
    worked table, not merely computed and trusted)."""
    numerator = m_multiplier * contender_state_bytes
    denominator = 2 * n_layers_transformer * d_model * bytes_per_elt
    assert numerator % denominator == 0, (
        f"cap_length(M={m_multiplier}) is not an exact integer token count at "
        f"n_layers={n_layers_transformer}, d_model={d_model}, bytes_per_elt={bytes_per_elt} -- "
        f"{numerator}/{denominator} -- MATCH-GATE requires an exact byte-match, not a rounded "
        f"approximation (sec 1.3's own tolerance table).")
    return numerator // denominator


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TransformerAttention(nn.Module):
    """Standard causal multi-head self-attention with RoPE, via
    `F.scaled_dot_product_attention` -- NEVER passes both `attn_mask` and
    `is_causal=True` together (CLAUDE.md's own PyTorch-2.4 hard rule):
    `attn_mask=None` routes through `is_causal=True` (pure causal, training
    -- always uncapped); an explicit boolean mask routes through
    `is_causal=False` with that mask supplying causality itself."""

    def __init__(self, d_model: int, n_heads: int, rope_base: float = ROPE_BASE_DEFAULT):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = build_rope_cache(T, self.head_dim, self.rope_base, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if attn_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        o = o.transpose(1, 2).reshape(B, T, self.d_model)
        return self.o_proj(o)


class TransformerBlock(nn.Module):
    """Pre-norm attn + pre-norm FFN -- the contender's own `FFN` class
    reused verbatim (M3)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 rope_base: float = ROPE_BASE_DEFAULT):
        super().__init__()
        from lm_pretrain_rd import FFN  # noqa: local import, after ensure_fla_stub()
        self.norm1 = RMSNorm(d_model, eps=1e-5)
        self.attn = TransformerAttention(d_model, n_heads, rope_base=rope_base)
        self.norm2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = FFN(d_model, mult=ffn_mult)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Top-level model. `attn_mask=None` (training, role b-control) is
    pure/uncapped causal attention; an explicit boolean `(T,T)` mask
    (`sink_fifo_mask(...)`, role b-primary, INFERENCE-ONLY) hard-caps the
    KV cache. `return_hidden=True` additionally returns the POST-`norm_f`
    hidden states (sec 1.3.1.2's Transformer native tap: "the final-block
    hidden state... from a SINGLE forward pass" -- this build's own
    disclosed choice is post-final-norm, the same representation that
    directly feeds the tied head, not a pre-norm intermediate)."""

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 2, n_heads: int = 4,
                 ffn_mult: int = 4, rope_base: float = ROPE_BASE_DEFAULT):
        super().__init__()
        assert n_layers >= 1, f"n_layers={n_layers} must be >= 1"
        self.vocab_size, self.d_model, self.n_layers, self.n_heads = vocab_size, d_model, n_layers, n_heads
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult=ffn_mult, rope_base=rope_base)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model, eps=1e-5)

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor | None = None,
                return_hidden: bool = False):
        x = self.embed(token_ids)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm_f(x)
        logits = F.linear(x, self.embed.weight)
        return (logits, x) if return_hidden else logits


def count_transformer_params(vocab_size: int, d_model: int, n_layers: int, ffn_mult: int = 4) -> int:
    """Closed-form param-count formula (independent of `count_mixer_params`-
    style helpers) -- verify_match_gate.py's Pass 2 re-derives this
    formula from scratch itself rather than importing this function, per
    the 2-pass MATCH-GATE's independence requirement; kept here as Pass
    1's own quick-check / this file's own smoke item 6."""
    per_layer = 2 * d_model + 4 * d_model ** 2 + 2 * ffn_mult * d_model ** 2
    return vocab_size * d_model + n_layers * per_layer + d_model


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def smoke_1_forward_backward_grad_finite_both_modes():
    torch.manual_seed(11)
    m = TransformerLM(500, d_model=32, n_layers=2, n_heads=4, ffn_mult=4)
    x = torch.randint(0, 500, (3, 24))
    y = torch.randint(0, 500, (3, 24))
    ok = True
    for name, mask in (("uncapped", None), ("capped", sink_fifo_mask(24, cap_length=10, k_sink=4))):
        m.zero_grad()
        logits = m(x, attn_mask=mask)
        finite_fwd = torch.isfinite(logits).all().item()
        loss = F.cross_entropy(logits.reshape(-1, 500), y.reshape(-1))
        loss.backward()
        finite_bwd = all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        print(f"    mode={name}: forward finite={finite_fwd} backward finite={finite_bwd}")
        ok = ok and finite_fwd and finite_bwd
    _report("smoke 1: forward/backward/grad-finite, uncapped AND sink+FIFO-capped modes", ok)


def smoke_2_rope_is_isometry():
    torch.manual_seed(5)
    x = torch.randn(1, 2, 8, 16)
    cos, sin = build_rope_cache(8, 16, ROPE_BASE_DEFAULT, x.device, x.dtype)
    x_rot = apply_rope(x, cos, sin)
    ok = torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-5)
    _report("smoke 2: RoPE preserves per-vector norm (isometry sanity)", ok,
            f"max_norm_diff={(x.norm(dim=-1) - x_rot.norm(dim=-1)).abs().max().item():.2e}")


def smoke_3_sink_fifo_mask_structure():
    """Hand-computed small case: T=10, cap_length=6, k_sink=2 -> fifo_window=4.
    Query i=8 should see: sink {0,1} union recent {5,6,7,8} (j>8-4=4)."""
    mask = sink_fifo_mask(10, cap_length=6, k_sink=2)
    row8 = mask[8].nonzero(as_tuple=True)[0].tolist()
    expected = sorted({0, 1, 5, 6, 7, 8})
    ok = row8 == expected
    causal_respected = bool((mask.triu(diagonal=1) == False).all())  # noqa: E712
    ok = ok and causal_respected
    _report("smoke 3: sink_fifo_mask structural check (hand-computed row + causality)", ok,
            f"row8={row8} expected={expected} causal_respected={causal_respected}")


def smoke_4_uncapped_none_matches_explicit_full_causal_mask():
    """attn_mask=None (is_causal=True path) must be numerically equivalent
    to an EXPLICIT full-lower-triangular boolean mask (is_causal=False
    path) -- guards against the two SDPA code paths silently diverging."""
    torch.manual_seed(21)
    m = TransformerLM(200, d_model=16, n_layers=1, n_heads=2, ffn_mult=2)
    m.eval()
    x = torch.randint(0, 200, (2, 12))
    full_causal = torch.tril(torch.ones(12, 12, dtype=torch.bool))
    with torch.no_grad():
        out_none = m(x, attn_mask=None)
        out_explicit = m(x, attn_mask=full_causal)
    ok = torch.allclose(out_none, out_explicit, atol=1e-4)
    _report("smoke 4: attn_mask=None (is_causal=True) matches an explicit full-causal boolean mask",
            ok, f"max_abs_diff={(out_none - out_explicit).abs().max().item():.2e}")


def smoke_5_capped_mode_has_teeth():
    """The capped mask must actually CHANGE the model's output relative to
    uncapped, when cap_length < T (else the cap is silently a no-op)."""
    torch.manual_seed(31)
    m = TransformerLM(200, d_model=16, n_layers=1, n_heads=2, ffn_mult=2)
    m.eval()
    x = torch.randint(0, 200, (2, 20))
    mask = sink_fifo_mask(20, cap_length=8, k_sink=4)
    with torch.no_grad():
        out_uncapped = m(x, attn_mask=None)
        out_capped = m(x, attn_mask=mask)
    ok = not torch.allclose(out_uncapped, out_capped, atol=1e-4)
    _report("smoke 5: capped mode changes output vs. uncapped (the cap has teeth)", ok)


def smoke_6_cap_length_table_matches_design_doc():
    """sec 1.4.2's own worked table, n_layers=2/d_model=256/fp32:
    M in {1,2,4,8,16,32} -> cap_length in {8,16,32,64,128,256}."""
    expected = {1: 8, 2: 16, 4: 32, 8: 64, 16: 128, 32: 256}
    got = {m: cap_length_tokens(m, PINNED_N_LAYERS_TRANSFORMER, 256) for m in expected}
    ok = got == expected
    _report("smoke 6: cap_length(M) table matches sec 1.4.2's own worked table exactly", ok,
            f"got={got} expected={expected}")


def smoke_7_param_count_formula_matches_real_instantiation():
    vocab, d_model, n_layers, ffn_mult = 50257, 256, 2, 4
    m = TransformerLM(vocab, d_model=d_model, n_layers=n_layers, n_heads=4, ffn_mult=ffn_mult)
    real = sum(p.numel() for p in m.parameters())
    formula = count_transformer_params(vocab, d_model, n_layers, ffn_mult)
    ok = real == formula
    _report("smoke 7: count_transformer_params formula matches a REAL instantiated model's numel()",
            ok, f"real={real:,} formula={formula:,}")


def smoke_8_cap_length_floor_exclusion_negative_test():
    """cap_length=k_sink (fifo_window=0) must raise (documented degenerate
    case never reached by the pinned grid) -- run to completion."""
    raised = False
    try:
        sink_fifo_mask(10, cap_length=4, k_sink=4)
        raise RuntimeError("NEGATIVE FAILED TO FAIL: cap_length==k_sink did not raise")
    except AssertionError:
        raised = True
    _report("smoke 8: sink_fifo_mask(cap_length==k_sink) negative test (fifo_window=0 refused)",
            raised)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    print("=" * 70)
    print("transformer_baseline_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3(b) smoke suite")
    print(f"fla stub installed (real fla absent or forced): {_STUB_INSTALLED}")
    print("=" * 70)
    smoke_1_forward_backward_grad_finite_both_modes()
    smoke_2_rope_is_isometry()
    smoke_3_sink_fifo_mask_structure()
    smoke_4_uncapped_none_matches_explicit_full_causal_mask()
    smoke_5_capped_mode_has_teeth()
    smoke_6_cap_length_table_matches_design_doc()
    smoke_7_param_count_formula_matches_real_instantiation()
    smoke_8_cap_length_floor_exclusion_negative_test()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
