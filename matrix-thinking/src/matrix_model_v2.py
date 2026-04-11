"""
Matrix-Thinking Model V2: Multiplicative Composition.

Each token is a d×d matrix. The core update per layer is:

    M_new = M + gate_m * ((I + Δ) · M · (I + Γ) - M) + gate_w * (v · kᵀ)

Where Δ, Γ are small data-dependent perturbations computed through bottlenecked
projections, gates are learned sigmoids, and v·kᵀ is a rank-1 additive write.

Key design decisions:
  - Multiplicative (not bilinear): products of perturbations are exponentially expressive
  - Bottlenecked projections: d² → r → d² to control parameter count
  - Lightweight attention: simple linear Q/K/V, not full multiplicative layers
  - Separate gating for multiplicative and additive paths
  - Pre-norm architecture for stable deep training
  - Dropout on attention and MLP paths
  - Clamp on scale parameter to prevent numerical instability
  - Two-layer MLP collapse head for output expressiveness
  - n_heads must divide d² (validated in __init__)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MatrixEmbedding(nn.Module):
    """Embed bytes as rank-1 matrices via outer product."""

    def __init__(self, vocab_size: int, mat_dim: int):
        super().__init__()
        self.mat_dim = mat_dim
        self.embed_u = nn.Embedding(vocab_size, mat_dim)
        self.embed_v = nn.Embedding(vocab_size, mat_dim)

    def forward(self, byte_ids):
        u = self.embed_u(byte_ids)
        v = self.embed_v(byte_ids)
        return torch.einsum('...i,...j->...ij', u, v)


class MatrixRMSNorm(nn.Module):
    """RMSNorm adapted for matrices. Normalizes by RMS of all d² entries."""

    def __init__(self, mat_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(mat_dim, mat_dim))
        self.eps = eps

    def forward(self, M):
        rms = torch.sqrt(M.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps)
        return M / rms * self.weight


class MultiplicativeMatrixLayer(nn.Module):
    """Core thinking operation: multiplicative composition.

    M_new = norm(M) → Δ, Γ, k, v, gates
    M_out = M + gate_m * ((I + Δ)·M_normed·(I + Γ) - M_normed) + gate_w * (v·kᵀ)

    Pre-norm: normalization happens BEFORE the computation, not after.
    Scale is clamped to prevent blowup.
    """

    def __init__(self, mat_dim: int, bottleneck: int = None, dropout: float = 0.1):
        super().__init__()
        self.d = mat_dim
        d2 = mat_dim * mat_dim
        if bottleneck is None:
            bottleneck = max(mat_dim, d2 // 4)

        # Pre-norm
        self.norm = MatrixRMSNorm(mat_dim)

        # SwiGLU bottlenecked projections for Δ and Γ
        # Gate path and value path, multiplied together (SwiGLU pattern)
        self.delta_gate = nn.Linear(d2, bottleneck, bias=False)
        self.delta_value = nn.Linear(d2, bottleneck, bias=False)
        self.delta_up = nn.Linear(bottleneck, d2, bias=False)
        self.gamma_gate = nn.Linear(d2, bottleneck, bias=False)
        self.gamma_value = nn.Linear(d2, bottleneck, bias=False)
        self.gamma_up = nn.Linear(bottleneck, d2, bias=False)

        # Low-rank additive write
        self.to_key = nn.Linear(d2, mat_dim, bias=False)
        self.to_value = nn.Linear(d2, mat_dim, bias=False)

        # Separate gates
        self.gate_mult = nn.Linear(d2, 1)
        self.gate_write = nn.Linear(d2, 1)

        # Scale for perturbations — clamped to [0.01, 0.5] to prevent instability
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Cache identity matrix
        self.register_buffer('I', torch.eye(mat_dim).reshape(1, 1, mat_dim, mat_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.delta_up.weight, std=0.01)
        nn.init.normal_(self.gamma_up.weight, std=0.01)
        nn.init.normal_(self.delta_gate.weight, std=0.02)
        nn.init.normal_(self.delta_value.weight, std=0.02)
        nn.init.normal_(self.gamma_gate.weight, std=0.02)
        nn.init.normal_(self.gamma_value.weight, std=0.02)
        nn.init.constant_(self.gate_mult.bias, -2.0)
        nn.init.constant_(self.gate_write.bias, -2.0)

    def forward(self, M):
        B, L, d, _ = M.shape

        # Pre-norm
        M_normed = self.norm(M)
        M_flat = M_normed.reshape(B, L, d * d)

        # Clamp scale
        scale = self.scale.clamp(0.01, 0.5)

        # Compute perturbations through SwiGLU bottleneck
        delta = self.delta_up(F.silu(self.delta_gate(M_flat)) * self.delta_value(M_flat)).reshape(B, L, d, d) * scale
        gamma = self.gamma_up(F.silu(self.gamma_gate(M_flat)) * self.gamma_value(M_flat)).reshape(B, L, d, d) * scale

        # Multiplicative: (I + Δ) · M_normed · (I + Γ)
        M_mult = torch.matmul(torch.matmul(self.I + delta, M_normed), self.I + gamma)

        # Additive write: v · kᵀ
        k = self.to_key(M_flat)
        v = self.to_value(M_flat)
        M_write = torch.einsum('...i,...j->...ij', v, k)

        # Separate gates
        g_m = torch.sigmoid(self.gate_mult(M_flat)).unsqueeze(-1)  # (B, L, 1, 1)
        g_w = torch.sigmoid(self.gate_write(M_flat)).unsqueeze(-1)

        # Residual with gated updates + dropout
        update = g_m * (M_mult - M_normed) + g_w * M_write
        M_new = M + self.dropout(update)

        return M_new


class MatrixAttention(nn.Module):
    """Lightweight pre-norm attention over matrix-valued tokens.

    Flattens d×d matrices to d² vectors for Q/K/V projections,
    computes multi-head attention, then reshapes back to matrices.
    """

    def __init__(self, mat_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d = mat_dim
        self.n_heads = n_heads
        d2 = mat_dim * mat_dim

        assert d2 % n_heads == 0, (
            f"mat_dim²={d2} must be divisible by n_heads={n_heads}. "
            f"Try mat_dim={mat_dim} with n_heads in {[h for h in [1,2,4,8] if d2 % h == 0]}"
        )

        self.head_dim = d2 // n_heads

        # Pre-norm
        self.norm = MatrixRMSNorm(mat_dim)

        # Projections
        self.q_proj = nn.Linear(d2, d2, bias=False)
        self.k_proj = nn.Linear(d2, d2, bias=False)
        self.v_proj = nn.Linear(d2, d2, bias=False)
        self.out_proj = nn.Linear(d2, d2, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, M):
        B, L, d, _ = M.shape
        d2 = d * d

        # Pre-norm
        M_normed = self.norm(M)
        M_flat = M_normed.reshape(B, L, d2)

        Q = self.q_proj(M_flat).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(M_flat).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(M_flat).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5

        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=M.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Aggregate and project
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, d2)
        out = self.out_proj(out).reshape(B, L, d, d)

        # Residual
        return M + self.out_dropout(out)


class MatrixThinkingBlock(nn.Module):
    """One layer: pre-norm attention then pre-norm multiplicative thinking."""

    def __init__(self, mat_dim: int, n_heads: int = 4,
                 bottleneck: int = None, dropout: float = 0.1):
        super().__init__()
        self.attn = MatrixAttention(mat_dim, n_heads, dropout)
        self.think = MultiplicativeMatrixLayer(mat_dim, bottleneck, dropout)

    def forward(self, M):
        M = self.attn(M)
        M = self.think(M)
        return M


class MatrixThinkingModelV2(nn.Module):
    """V2: Multiplicative composition, pre-norm, dropout, MLP output head."""

    def __init__(self, mat_dim=16, n_layers=6, n_heads=4,
                 bottleneck=None, max_len=512, vocab_size=256,
                 dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim
        self.n_layers = n_layers
        d2 = mat_dim * mat_dim

        assert d2 % n_heads == 0, (
            f"mat_dim²={d2} must be divisible by n_heads={n_heads}"
        )

        self.embed = MatrixEmbedding(vocab_size, mat_dim)
        self.pos_u = nn.Embedding(max_len, mat_dim)
        self.pos_v = nn.Embedding(max_len, mat_dim)

        self.layers = nn.ModuleList([
            MatrixThinkingBlock(mat_dim, n_heads, bottleneck, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = MatrixRMSNorm(mat_dim)

        # SwiGLU output head
        self.collapse_gate = nn.Linear(d2, d2)
        self.collapse_value = nn.Linear(d2, d2)
        self.collapse_out = nn.Linear(d2, vocab_size)

    def forward(self, byte_ids, return_ranks=False):
        B, L = byte_ids.shape
        device = byte_ids.device

        M = self.embed(byte_ids)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pu = self.pos_u(positions)
        pv = self.pos_v(positions)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

        ranks = []
        if return_ranks:
            ranks.append(self._measure_ranks(M))

        for layer in self.layers:
            M = layer(M)
            if return_ranks:
                ranks.append(self._measure_ranks(M))

        M = self.final_norm(M)
        M_flat = M.reshape(B, L, self.mat_dim ** 2)
        logits = self.collapse_out(F.silu(self.collapse_gate(M_flat)) * self.collapse_value(M_flat))

        if return_ranks:
            return logits, torch.stack(ranks)
        return logits

    def _measure_ranks(self, M):
        """Effective rank = exp(entropy of singular values). Diagnostic only."""
        B, L, d, _ = M.shape
        M_cpu = M.detach().cpu().reshape(B * L, d, d)
        try:
            S = torch.linalg.svdvals(M_cpu)
        except Exception:
            return torch.ones(B, L)
        S = S.clamp(min=1e-10)
        S_norm = S / S.sum(dim=-1, keepdim=True)
        entropy = -(S_norm * S_norm.log()).sum(dim=-1)
        return entropy.exp().reshape(B, L)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
