"""
Experiment 2d: Plain bilinear projection A @ M @ B.

Simplest matrix-native operation. No nonlinearity in the projection itself.
Baseline for measuring how much the nonlinearity in 2c helps.
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, train
from exp1a_matrix_v3 import MatrixV3


class BilinearProjection(nn.Module):
    """Y = A @ M @ B. Simplest matrix-native projection. Never flattens."""

    def __init__(self, d, K=None):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))

    def forward(self, M):
        return torch.einsum('ij,bsjk,kl->bsil', self.A, M, self.B)


def get_mini_config():
    return TrainConfig(
        data_dir="/Volumes/1TB_SSD/learned-representations/data/wikitext103_tokenized",
        seq_len=128, batch_size=8, max_steps=200, lr=1e-3, warmup_steps=20,
        dropout=0.1, eval_interval=50, log_interval=25,
        save_dir="/Volumes/1TB_SSD/learned-representations/results/h100_test",
        experiment_name="exp2d_bilinear_mini",
        device="mps", compile_model=False, use_bfloat16=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    config = get_mini_config() if args.mini else None

    import exp1a_matrix_v3 as base
    base.make_projection = lambda proj_type, d, K: BilinearProjection(d, K) if proj_type == 'bilinear' else base.KroneckerProjection(d, K)

    if args.mini:
        model = MatrixV3(mat_dim=8, n_layers=2, n_heads=2, K=2,
                         max_len=256, vocab_size=50257, dropout=0.1,
                         proj_type='bilinear')
    else:
        model = MatrixV3(mat_dim=16, n_layers=6, n_heads=4, K=4,
                         max_len=1024, vocab_size=50257, dropout=0.1,
                         proj_type='bilinear')

    print(f"Model: {model.count_parameters():,} params")
    train(model, config)
