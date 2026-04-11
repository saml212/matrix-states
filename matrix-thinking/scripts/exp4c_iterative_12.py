"""
Experiment 4c: More iterations — 1 shared block × 12 iterations.

Tests whether more thinking steps (more iterations of the same block)
continue to improve results. Compared to 4b (4 iterations).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, train
from exp4b_iterative_shared import IterativeMatrixV3


def get_mini_config():
    return TrainConfig(
        data_dir="/Volumes/1TB_SSD/learned-representations/data/wikitext103_tokenized",
        seq_len=128, batch_size=8, max_steps=200, lr=1e-3, warmup_steps=20,
        dropout=0.1, eval_interval=50, log_interval=25,
        save_dir="/Volumes/1TB_SSD/learned-representations/results/h100_test",
        experiment_name="exp4c_iterative_12_mini",
        device="mps", compile_model=False, use_bfloat16=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    config = get_mini_config() if args.mini else None

    if args.mini:
        model = IterativeMatrixV3(mat_dim=8, n_block_layers=1, n_iterations=8,
                                  n_heads=2, K=2, max_len=256, vocab_size=50257, dropout=0.1)
    else:
        model = IterativeMatrixV3(mat_dim=16, n_block_layers=2, n_iterations=12,
                                  n_heads=4, K=4, max_len=1024, vocab_size=50257, dropout=0.1)

    print(f"Model: {model.count_parameters():,} params")
    print(f"Shared block applied {model.n_iterations} times")
    train(model, config)
