"""Stage G shared components — see STAGE_G_DESIGN.md (frozen, revision 2.1).

These classes are VERBATIM copies (not reimplementations) of the two exact
architectures Stage G diagnoses:

  - MatrixThinker's backbone/head components, copied from
    experiment-runs/8xh100-session1/round2_matrix_script.py (the model
    actually trained for Runs 12/14/15).
  - LoopFormer's components, copied from
    experiment-runs/8xh100-session1/loopformer_96K_script.py (Runs 13/14).

Copying verbatim (rather than importing those scripts, which are DDP/argparse
entry points not designed as libraries, and rather than re-deriving from
memory) is what makes the design's requirement 2 ("baseline construction must
be behaviorally identical to the round2 script's model... verify param
shapes/counts") checkable by direct diff instead of by trust. See
test_stageg.py's `test_baseline_identical_to_round2_script` for the
shape/count/identity verification.

Only intervention-relevant knobs (documented per class) are threaded through;
everything else is unchanged from the source scripts.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ═══════════════════════════════════════════════════════════════════════════
# MatrixThinker components — verbatim from round2_matrix_script.py
# ═══════════════════════════════════════════════════════════════════════════

class MatrixRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, d))
        self.eps = eps

    def forward(self, M):
        rms = torch.sqrt(M.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps)
        return M / rms * self.weight


class RowThenColProjection(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))

    def forward(self, M):
        return torch.einsum('...ij,jk->...ik',
                             F.silu(torch.einsum('ij,...jk->...ik', self.A, M)), self.B)


class FactoredDenseProjection(nn.Module):
    """H_b's gated rank-swept family (design section 4 item 8; gate FIRED
    2026-07-02 -- the depth-matched control failed to close the gap):
    "a rank-r factored Linear(d^2,r) -> Linear(r,d^2) swept over a few r
    values as a parametric family bridging Kronecker-restricted and
    increasingly expressive" (section 6.1 pins r in {1,4,16}).

    vec(M) -> Linear(d^2, r) -> silu -> Linear(r, d^2) -> reshape(d,d).
    The silu sits between the two maps, mirroring RowThenColProjection's
    own silu placement (silu(A@M)@B), so the swap changes ONLY the linear
    family, not the nonlinearity structure.

    Disclosed asymmetries (the design's own "no cheap clean control
    exists" ambiguity, section 1 H_b, carried forward not resolved):
    (i) params: 2*d^2*r vs RowThenCol's 2d^2 -- r=1 is the param-matched
    point ("matching params forces rank <=1"); r=16 costs 16x per
    projection. (ii) init: RowThenCol starts near-identity (eye + 0.02
    noise), which a rank r<d^2 map cannot represent -- torch Linear
    default init is used, so early dynamics differ for init reasons as
    well as expressiveness reasons."""

    def __init__(self, d, r):
        super().__init__()
        self.d = d
        self.down = nn.Linear(d * d, r, bias=False)
        self.up = nn.Linear(r, d * d, bias=False)

    def forward(self, M):
        shape = M.shape
        v = M.reshape(*shape[:-2], self.d * self.d)
        return self.up(F.silu(self.down(v))).reshape(shape)


class MatrixFrobeniusAttention(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1, proj_factory=None):
        super().__init__()
        # proj_factory (H_b rider): callable d -> projection module. Default
        # None -> RowThenColProjection, leaving the baseline construction
        # path (and its RNG consumption order -- bit-identity test [3])
        # untouched.
        proj_factory = proj_factory or RowThenColProjection
        self.d, self.n_heads, self.head_dim = d, n_heads, d // n_heads
        self.norm = MatrixRMSNorm(d)
        self.q_proj = proj_factory(d)
        self.k_proj = proj_factory(d)
        self.v_proj = proj_factory(d)
        self.o_proj = proj_factory(d)
        self.dropout_p = dropout

    def forward(self, M):
        B, L, d, _ = M.shape
        H, hd = self.n_heads, self.head_dim
        M_n = self.norm(M)
        Q = self.q_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        K = self.k_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        V = self.v_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True,
                                              dropout_p=self.dropout_p if self.training else 0.0)
        out = out.reshape(B, H, L, hd, d).permute(0, 2, 1, 3, 4).reshape(B, L, d, d)
        return M + self.o_proj(out)


class MatrixMultiplicativeLayer(nn.Module):
    def __init__(self, d, dropout=0.1, proj_factory=None):
        super().__init__()
        proj_factory = proj_factory or RowThenColProjection   # H_b rider; None -> baseline path unchanged
        self.norm = MatrixRMSNorm(d)
        self.delta_gate = proj_factory(d)
        self.delta_value = proj_factory(d)
        self.delta_up = proj_factory(d)
        self.gamma_gate = proj_factory(d)
        self.gamma_value = proj_factory(d)
        self.gamma_up = proj_factory(d)
        self.key_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.val_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.gate_mult_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_mult_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_write_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_write_bias = nn.Parameter(torch.tensor(-2.0))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('I', torch.eye(d))

    def forward(self, M):
        M_n = self.norm(M)
        s = self.scale.clamp(0.01, 0.5)
        delta = self.delta_up(F.silu(self.delta_gate(M_n)) * self.delta_value(M_n)) * s
        gamma = self.gamma_up(F.silu(self.gamma_gate(M_n)) * self.gamma_value(M_n)) * s
        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)
        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)
        g_m = torch.sigmoid((self.gate_mult_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_mult_bias)
        g_w = torch.sigmoid((self.gate_write_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_write_bias)
        return M + self.dropout(g_m * (M_mult - M_n) + g_w * M_write)


class ThinkingBlock(nn.Module):
    """H_i rider (revision 2, F5): optional per-iteration AdaLN-style
    conditioning, applied ONLY when `cond_dim` is not None. With
    cond_dim=None this is bit-identical to round2_matrix_script.py's
    ThinkingBlock (no new params, no new forward args used)."""

    def __init__(self, d, n_heads=8, dropout=0.1, cond_dim=None, proj_factory=None):
        super().__init__()
        self.attn = MatrixFrobeniusAttention(d, n_heads, dropout, proj_factory=proj_factory)
        self.think = MatrixMultiplicativeLayer(d, dropout, proj_factory=proj_factory)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            # Per-iteration scale/shift applied to the matrix (broadcast over
            # both matrix axes) — the minimal analog of LoopFormer's AdaLN
            # modulation (design section 4, item 6): "a small timestep
            # embedder ... producing a per-iteration scale/shift applied at
            # each ThinkingBlock's norm."
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2, bias=True))
            nn.init.zeros_(self.adaLN[-1].weight)
            nn.init.zeros_(self.adaLN[-1].bias)  # start as a no-op (scale=0,shift=0 -> 1+scale=1)

    def forward(self, M, cond=None):
        if self.cond_dim is not None and cond is not None:
            scale, shift = self.adaLN(cond).chunk(2, dim=-1)          # (B, 1) each
            scale = scale.view(-1, 1, 1, 1)
            shift = shift.view(-1, 1, 1, 1)
            M = M * (1 + scale) + shift
        return self.think(self.attn(M))


class MultiProbeHead(nn.Module):
    """Matrix-native output: K bilinear probes -> Linear -> vocab logits.
    Untied (F2's diagnosed default state). Verbatim from round2_matrix_script.py."""

    def __init__(self, d, vocab_size, n_probes=None):
        super().__init__()
        K = n_probes or d
        self.U = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.V = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.out = nn.Linear(K, vocab_size, bias=False)

    def forward(self, M):
        MV = torch.einsum('blij, kj -> blik', M, self.V)
        probes = torch.einsum('ki, blik -> blk', self.U, MV)
        return self.out(probes)


# ═══════════════════════════════════════════════════════════════════════════
# LoopFormer components — verbatim from loopformer_96K_script.py
# (needed for: Wave -1's Regime-1 step-0 entry-std check (H_j), and as the
#  source pattern H_i's ThinkingBlock rider above is modeled on)
# ═══════════════════════════════════════════════════════════════════════════

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class LoopFormerBlock(nn.Module):
    def __init__(self, n_embd, n_head, intermediate_dim, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.RMSNorm(n_embd, elementwise_affine=False)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.norm_2 = nn.RMSNorm(n_embd, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, n_embd),
            nn.Dropout(dropout),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_embd, 4 * n_embd, bias=True),
        )

    def forward(self, x, c):
        shift_attn, scale_attn, shift_mlp, scale_mlp = self.adaLN(c).chunk(4, dim=-1)
        h = self.norm_1(x)
        h = h * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        x = x + self.attn(h)
        h = self.norm_2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(h)
        return x


class LoopFormer(nn.Module):
    """ICLR 2026 (arxiv 2602.11451). Verbatim from loopformer_96K_script.py."""

    def __init__(self, n_embd=128, n_head=4, n_layer=2, n_loops=8,
                 intermediate_dim=320, max_len=2048, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.n_loops = n_loops
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            LoopFormerBlock(n_embd, n_head, intermediate_dim, dropout)
            for _ in range(n_layer)
        ])
        self.time_embedder = TimestepEmbedder(n_embd)
        self.dt_embedder = TimestepEmbedder(n_embd)
        self.norm_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _one_loop(self, x, c):
        for block in self.blocks:
            x = block(x, c)
        return x

    def forward(self, token_ids, n_loops=None, **kwargs):
        n_loops = n_loops or self.n_loops
        B, L = token_ids.shape
        device = token_ids.device
        tok_emb = self.wte(token_ids)
        pos_emb = self.wpe(torch.arange(L, device=device))
        x = self.drop(tok_emb + pos_emb)
        steps = [1.0 / n_loops] * n_loops
        ti = torch.zeros(B, dtype=x.dtype, device=device)
        for dt in steps:
            dt_base = torch.ones_like(ti) * dt
            c = self.time_embedder(ti) + self.dt_embedder(dt_base)
            if self.training and n_loops > 1:
                x = torch.utils.checkpoint.checkpoint(self._one_loop, x, c, use_reentrant=False)
            else:
                x = self._one_loop(x, c)
            ti = ti + dt
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, {'ranks': [], 'model_type': 'loopformer'}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
