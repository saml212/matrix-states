"""
Experiment 1a: Matrix V3 — Kronecker projections, no flattening, SwiGLU.

This is the main model. Matrix-valued tokens with matrix-native operations throughout.
The core thinking operation is (I+Δ)·M·(I+Γ) + v·kᵀ where Δ and Γ are computed
via sum-of-Kroneckers projections that never flatten the matrix.

H100 version: FA3 attention, DDP, torch.compile, bfloat16.
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent dirs to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, train


# ─── Matrix-Native Operations ─────────────────────────────────

class MatrixRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, d))
        self.eps = eps

    def forward(self, M):
        rms = torch.sqrt(M.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps)
        return M / rms * self.weight


class KroneckerProjection(nn.Module):
    """Sum-of-K bilinear terms: output = Σ_k A_k @ M @ B_k. Never flattens."""

    def __init__(self, d, K=4):
        super().__init__()
        self.d = d
        self.K = K
        self.As = nn.Parameter(torch.randn(K, d, d) * 0.02)
        self.Bs = nn.Parameter(torch.randn(K, d, d) * 0.02)
        # Init first component near identity
        nn.init.eye_(self.As[0])
        nn.init.eye_(self.Bs[0])
        self.As.data[0] += torch.randn(d, d) * 0.01
        self.Bs.data[0] += torch.randn(d, d) * 0.01

    def forward(self, M):
        out = torch.einsum('ij,bsjk,kl->bsil', self.As[0], M, self.Bs[0])
        for k in range(1, self.K):
            out = out + torch.einsum('ij,bsjk,kl->bsil', self.As[k], M, self.Bs[k])
        return out


class MatrixEmbedding(nn.Module):
    def __init__(self, vocab_size, d):
        super().__init__()
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)

    def forward(self, ids):
        u = self.embed_u(ids)
        v = self.embed_v(ids)
        return torch.einsum('...i,...j->...ij', u, v)


def make_projection(proj_type, d, K):
    """Factory for matrix-native projection modules."""
    if proj_type == 'kronecker':
        return KroneckerProjection(d, K)
    elif proj_type == 'householder':
        from exp2b_householder import HouseholderProjection
        return HouseholderProjection(d, K)
    else:
        return KroneckerProjection(d, K)  # default


class MatrixNativeMultiplicativeLayer(nn.Module):
    """Core thinking: (I+Δ)·M·(I+Γ) + v·kᵀ. All projections are matrix-native, no flatten."""

    def __init__(self, d, K=4, dropout=0.1, proj_type='kronecker'):
        super().__init__()
        self.d = d
        self.norm = MatrixRMSNorm(d)

        # SwiGLU in matrix space via configurable projections
        self.delta_gate = make_projection(proj_type, d, K)
        self.delta_value = make_projection(proj_type, d, K)
        self.delta_up = make_projection(proj_type, d, K)
        self.gamma_gate = make_projection(proj_type, d, K)
        self.gamma_value = make_projection(proj_type, d, K)
        self.gamma_up = make_projection(proj_type, d, K)

        # Key/value for additive write — extract vectors via column projection
        self.key_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.val_col = nn.Parameter(torch.randn(d, 1) * 0.02)

        # Gates via Frobenius inner product (scalar from matrix, no flatten)
        self.gate_mult_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_mult_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_write_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_write_bias = nn.Parameter(torch.tensor(-2.0))

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('I', torch.eye(d))

        # Small init for up projections if they have Kronecker structure
        for proj in [self.delta_up, self.gamma_up]:
            if hasattr(proj, 'As') and hasattr(proj, 'Bs'):
                proj.As.data[1:] *= 0.1
                proj.Bs.data[1:] *= 0.1

    def forward(self, M):
        B, L, d, _ = M.shape
        M_n = self.norm(M)
        scale = self.scale.clamp(0.01, 0.5)

        # SwiGLU Delta — entirely in matrix space
        gate_d = F.silu(self.delta_gate(M_n))
        val_d = self.delta_value(M_n)
        delta = self.delta_up(gate_d * val_d) * scale  # Hadamard product of matrices

        # SwiGLU Gamma
        gate_g = F.silu(self.gamma_gate(M_n))
        val_g = self.gamma_value(M_n)
        gamma = self.gamma_up(gate_g * val_g) * scale

        # Multiplicative composition
        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)

        # Additive write
        k = torch.matmul(M_n, self.key_col).squeeze(-1)  # (B,L,d)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)

        # Scalar gates from Frobenius inner product
        g_m = torch.sigmoid(
            (self.gate_mult_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_mult_bias
        )
        g_w = torch.sigmoid(
            (self.gate_write_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_write_bias
        )

        update = g_m * (M_mult - M_n) + g_w * M_write
        return M + self.dropout(update)


class MatrixNativeAttention(nn.Module):
    """Multi-head attention over matrix tokens. Matrix-native Q/K/V, Frobenius scores."""

    def __init__(self, d, n_heads=4, K=4, dropout=0.1, proj_type='kronecker'):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        assert d % n_heads == 0

        self.norm = MatrixRMSNorm(d)
        self.q_proj = make_projection(proj_type, d, K)
        self.k_proj = make_projection(proj_type, d, K)
        self.v_proj = make_projection(proj_type, d, K)
        self.o_proj = make_projection(proj_type, d, K)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, M):
        B, L, d, _ = M.shape
        H = self.n_heads
        hd = self.head_dim

        M_n = self.norm(M)
        Q = self.q_proj(M_n)  # (B, L, d, d)
        K = self.k_proj(M_n)
        V = self.v_proj(M_n)

        # Split rows into heads: (B, L, d, d) → (B, H, L, hd, d)
        Q = Q.reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)
        K = K.reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)
        V = V.reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)

        # Frobenius inner product for attention scores
        scores = torch.einsum('bhlij,bhmij->bhlm', Q, K) / math.sqrt(hd * d)

        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=M.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Value aggregation
        out = torch.einsum('bhlm,bhmij->bhlij', attn, V)
        out = out.permute(0, 2, 1, 3, 4).reshape(B, L, d, d)
        out = self.o_proj(out)

        return M + self.out_dropout(out)


class MatrixThinkingBlock(nn.Module):
    def __init__(self, d, n_heads=4, K=4, dropout=0.1, proj_type='kronecker'):
        super().__init__()
        self.attn = MatrixNativeAttention(d, n_heads, K, dropout, proj_type)
        self.think = MatrixNativeMultiplicativeLayer(d, K, dropout, proj_type)

    def forward(self, M):
        M = self.attn(M)
        M = self.think(M)
        return M


class MatrixV3(nn.Module):
    """Matrix V3: Kronecker projections, no flattening, SwiGLU."""

    def __init__(self, mat_dim=16, n_layers=6, n_heads=4, K=4,
                 max_len=512, vocab_size=50257, dropout=0.1, proj_type='kronecker'):
        super().__init__()
        self.mat_dim = mat_dim
        d2 = mat_dim * mat_dim

        self.embed = MatrixEmbedding(vocab_size, mat_dim)
        self.pos_u = nn.Embedding(max_len, mat_dim)
        self.pos_v = nn.Embedding(max_len, mat_dim)

        self.layers = nn.ModuleList([
            MatrixThinkingBlock(mat_dim, n_heads, K, dropout, proj_type)
            for _ in range(n_layers)
        ])

        self.final_norm = MatrixRMSNorm(mat_dim)

        # Output head: weighted Frobenius readout (uses ALL matrix entries, no flatten)
        # collapse_W weights each entry of the matrix, sum over rows gives d-dim vector
        self.collapse_W = nn.Parameter(torch.randn(mat_dim, mat_dim) * 0.02)
        self.collapse_out = nn.Linear(mat_dim, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        device = x.device

        M = self.embed(x)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pu = self.pos_u(positions)
        pv = self.pos_v(positions)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

        for layer in self.layers:
            M = layer(M)

        M = self.final_norm(M)

        # Collapse: matrix → vector via weighted sum over columns
        # (collapse_W * M) sums element-wise, then sum over last dim gives d-dim vector
        # This uses ALL d×d entries of M, not just one column
        out_vec = (self.collapse_W * M).sum(dim=-1)  # (B, L, d)
        logits = self.collapse_out(out_vec)  # (B, L, vocab_size)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── H100 Config ───────────────────────────────────────────────

def get_h100_config():
    return TrainConfig(
        data_dir="/data/wikitext103_tokenized",  # adjust for H100 filesystem
        seq_len=512,
        batch_size=64,
        max_steps=50000,
        lr=3e-4,
        warmup_steps=1000,
        dropout=0.1,
        eval_interval=1000,
        log_interval=100,
        save_dir="/results",
        experiment_name="exp1a_matrix_v3",
        device="cuda",
        compile_model=True,
        use_bfloat16=True,
    )


# ─── Mac Mini Config (for testing) ────────────────────────────

def get_mini_config():
    return TrainConfig(
        data_dir="/Volumes/1TB_SSD/learned-representations/data/wikitext103_tokenized",
        seq_len=128,
        batch_size=8,
        max_steps=200,
        lr=1e-3,
        warmup_steps=20,
        dropout=0.1,
        eval_interval=50,
        log_interval=25,
        save_dir="/Volumes/1TB_SSD/learned-representations/results/h100_test",
        experiment_name="exp1a_matrix_v3_mini",
        device="mps",
        compile_model=False,
        use_bfloat16=False,
    )


# ─── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="Run Mac Mini test version")
    args = parser.parse_args()

    if args.mini:
        config = get_mini_config()
        # Small model for testing
        model = MatrixV3(mat_dim=8, n_layers=2, n_heads=2, K=2,
                         max_len=256, vocab_size=50257, dropout=0.1)
    else:
        config = get_h100_config()
        # Full model for H100
        model = MatrixV3(mat_dim=16, n_layers=6, n_heads=4, K=4,
                         max_len=1024, vocab_size=50257, dropout=0.1)

    print(f"Model: {model.count_parameters():,} params")
    train(model, config)
