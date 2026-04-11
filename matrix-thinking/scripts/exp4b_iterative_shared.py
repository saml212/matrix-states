"""
Experiment 4b: Iterative refinement with shared weights.

2 shared-weight layers applied 6 times (6 iterations of 2-layer block).
Uses PonderNet-style weighted loss across all iterations.
This is the prerequisite for adaptive matrix thinking — proving that
iterating the same block multiple times helps.
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, train as base_train
from exp1a_matrix_v3 import (
    MatrixV3, MatrixRMSNorm, MatrixEmbedding,
    MatrixNativeAttention, MatrixNativeMultiplicativeLayer,
    MatrixThinkingBlock, KroneckerProjection
)


class IterativeMatrixV3(nn.Module):
    """Matrix V3 with shared weights applied iteratively."""

    def __init__(self, mat_dim=16, n_block_layers=2, n_iterations=6,
                 n_heads=4, K=4, max_len=512, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim
        self.n_iterations = n_iterations

        self.embed = MatrixEmbedding(vocab_size, mat_dim)
        self.pos_u = nn.Embedding(max_len, mat_dim)
        self.pos_v = nn.Embedding(max_len, mat_dim)

        # Shared block — same weights reused across iterations
        self.shared_layers = nn.ModuleList([
            MatrixThinkingBlock(mat_dim, n_heads, K, dropout)
            for _ in range(n_block_layers)
        ])

        self.final_norm = MatrixRMSNorm(mat_dim)
        self.collapse_W = nn.Parameter(torch.randn(mat_dim, mat_dim) * 0.02)
        self.collapse_out = nn.Linear(mat_dim, vocab_size, bias=False)

        # Iteration embedding — tells the model which iteration it's on
        self.iter_embed = nn.Embedding(n_iterations, mat_dim * mat_dim)

    def forward(self, x):
        B, L = x.shape
        device = x.device

        M = self.embed(x)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pu = self.pos_u(positions)
        pv = self.pos_v(positions)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

        # Apply shared block iteratively
        for t in range(self.n_iterations):
            # Add iteration embedding (so the model knows which pass it's on)
            iter_emb = self.iter_embed(torch.tensor(t, device=device))
            iter_mat = iter_emb.reshape(self.mat_dim, self.mat_dim) * 0.01
            M = M + iter_mat  # broadcast over batch and seq

            for layer in self.shared_layers:
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
        experiment_name="exp4b_iterative_shared_mini",
        device="mps", compile_model=False, use_bfloat16=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    config = get_mini_config() if args.mini else None

    if args.mini:
        model = IterativeMatrixV3(mat_dim=8, n_block_layers=1, n_iterations=4,
                                  n_heads=2, K=2, max_len=256, vocab_size=50257, dropout=0.1)
    else:
        model = IterativeMatrixV3(mat_dim=16, n_block_layers=2, n_iterations=6,
                                  n_heads=4, K=4, max_len=1024, vocab_size=50257, dropout=0.1)

    print(f"Model: {model.count_parameters():,} params")
    print(f"Shared block applied {model.n_iterations} times")
    base_train(model, config)
