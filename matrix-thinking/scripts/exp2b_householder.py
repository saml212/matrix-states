"""
Experiment 2b: Householder reflections for matrix-native projections.

Same architecture as 1a but proj_type='householder'.
Fewest parameters of any matrix-native operation (d+1 per reflection).
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, train
from exp1a_matrix_v3 import MatrixV3


class HouseholderProjection(nn.Module):
    """Product of K Householder reflections. Never flattens. Fewest params."""

    def __init__(self, d, K=4):
        super().__init__()
        self.d = d
        self.K = K
        self.v = nn.Parameter(torch.randn(K, d))
        self.beta_raw = nn.Parameter(torch.zeros(K))

    def forward(self, M):
        result = M
        for k in range(self.K):
            v = self.v[k]
            v = v / (v.norm() + 1e-8)
            beta = 2.0 * torch.sigmoid(self.beta_raw[k])
            vtM = torch.einsum('i,bsij->bsj', v, result)
            outer = torch.einsum('i,bsj->bsij', v, vtM)
            result = result - beta * outer
        return result


def get_mini_config():
    return TrainConfig(
        data_dir="/Volumes/1TB_SSD/learned-representations/data/wikitext103_tokenized",
        seq_len=128, batch_size=8, max_steps=200, lr=1e-3, warmup_steps=20,
        dropout=0.1, eval_interval=50, log_interval=25,
        save_dir="/Volumes/1TB_SSD/learned-representations/results/h100_test",
        experiment_name="exp2b_householder_mini",
        device="mps", compile_model=False, use_bfloat16=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    config = get_mini_config() if args.mini else None  # TODO: H100 config

    # Register the Householder projection so make_projection can find it
    import exp1a_matrix_v3 as base
    base.make_projection = lambda proj_type, d, K: HouseholderProjection(d, K) if proj_type == 'householder' else base.KroneckerProjection(d, K)

    if args.mini:
        model = MatrixV3(mat_dim=8, n_layers=2, n_heads=2, K=2,
                         max_len=256, vocab_size=50257, dropout=0.1,
                         proj_type='householder')
    else:
        model = MatrixV3(mat_dim=16, n_layers=6, n_heads=4, K=4,
                         max_len=1024, vocab_size=50257, dropout=0.1,
                         proj_type='householder')

    print(f"Model: {model.count_parameters():,} params")
    train(model, config)
