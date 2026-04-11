"""
Experiment 3b: GELU activation instead of SwiGLU in the multiplicative layer.

Tests whether SwiGLU's gate*value pattern matters in matrix space,
or if simple GELU is sufficient.
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, train
from exp1a_matrix_v3 import (
    MatrixV3, MatrixRMSNorm, MatrixEmbedding,
    MatrixNativeAttention, MatrixThinkingBlock,
    KroneckerProjection, make_projection
)


class GELUMultiplicativeLayer(nn.Module):
    """Like MatrixNativeMultiplicativeLayer but uses GELU instead of SwiGLU.
    Delta = up(gelu(down(M))) instead of up(silu(gate(M)) * value(M))."""

    def __init__(self, d, K=4, dropout=0.1, proj_type='kronecker'):
        super().__init__()
        self.d = d
        self.norm = MatrixRMSNorm(d)

        # GELU bottleneck (not SwiGLU — single path, not gate*value)
        self.delta_down = make_projection(proj_type, d, K)
        self.delta_up = make_projection(proj_type, d, K)
        self.gamma_down = make_projection(proj_type, d, K)
        self.gamma_up = make_projection(proj_type, d, K)

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
        B, L, d, _ = M.shape
        M_n = self.norm(M)
        scale = self.scale.clamp(0.01, 0.5)

        # GELU bottleneck (not SwiGLU)
        delta = self.delta_up(F.gelu(self.delta_down(M_n))) * scale
        gamma = self.gamma_up(F.gelu(self.gamma_down(M_n))) * scale

        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)

        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)

        g_m = torch.sigmoid(
            (self.gate_mult_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_mult_bias
        )
        g_w = torch.sigmoid(
            (self.gate_write_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_write_bias
        )

        update = g_m * (M_mult - M_n) + g_w * M_write
        return M + self.dropout(update)


class GELUThinkingBlock(nn.Module):
    def __init__(self, d, n_heads=4, K=4, dropout=0.1, proj_type='kronecker'):
        super().__init__()
        self.attn = MatrixNativeAttention(d, n_heads, K, dropout, proj_type)
        self.think = GELUMultiplicativeLayer(d, K, dropout, proj_type)

    def forward(self, M):
        M = self.attn(M)
        M = self.think(M)
        return M


class MatrixV3GELU(nn.Module):
    """MatrixV3 with GELU instead of SwiGLU in the multiplicative layer."""

    def __init__(self, mat_dim=16, n_layers=6, n_heads=4, K=4,
                 max_len=512, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim
        self.embed = MatrixEmbedding(vocab_size, mat_dim)
        self.pos_u = nn.Embedding(max_len, mat_dim)
        self.pos_v = nn.Embedding(max_len, mat_dim)

        self.layers = nn.ModuleList([
            GELUThinkingBlock(mat_dim, n_heads, K, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = MatrixRMSNorm(mat_dim)
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
        out_vec = (self.collapse_W * M).sum(dim=-1)
        logits = self.collapse_out(out_vec)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_mini_config():
    return TrainConfig(
        data_dir="/Volumes/1TB_SSD/learned-representations/data/wikitext103_tokenized",
        seq_len=128, batch_size=8, max_steps=200, lr=1e-3, warmup_steps=20,
        dropout=0.1, eval_interval=50, log_interval=25,
        save_dir="/Volumes/1TB_SSD/learned-representations/results/h100_test",
        experiment_name="exp3b_gelu_mini",
        device="mps", compile_model=False, use_bfloat16=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    config = get_mini_config() if args.mini else None

    if args.mini:
        model = MatrixV3GELU(mat_dim=8, n_layers=2, n_heads=2, K=2,
                              max_len=256, vocab_size=50257, dropout=0.1)
    else:
        model = MatrixV3GELU(mat_dim=16, n_layers=6, n_heads=4, K=4,
                              max_len=1024, vocab_size=50257, dropout=0.1)

    print(f"Model: {model.count_parameters():,} params")
    train(model, config)
