"""
Experiment 1c: Matrix V2 — flatten→linear projections (the current code).

Same matrix thinking but with flattening for projections.
Comparison against V3 (Kronecker, no flatten) to measure the cost of flattening.
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

# Import the existing V2 model
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from matrix_model_v2 import MatrixThinkingModelV2


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
        experiment_name="exp1c_matrix_v2_flatten_mini",
        device="mps",
        compile_model=False,
        use_bfloat16=False,
    )


def get_h100_config():
    return TrainConfig(
        data_dir="/data/wikitext103_tokenized",
        seq_len=512,
        batch_size=64,
        max_steps=50000,
        lr=3e-4,
        warmup_steps=1000,
        dropout=0.1,
        eval_interval=1000,
        log_interval=100,
        save_dir="/results",
        experiment_name="exp1c_matrix_v2_flatten",
        device="cuda",
        compile_model=True,
        use_bfloat16=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    if args.mini:
        config = get_mini_config()
        model = MatrixThinkingModelV2(mat_dim=8, n_layers=2, n_heads=2,
                                      max_len=256, vocab_size=50257, dropout=0.1)
    else:
        config = get_h100_config()
        model = MatrixThinkingModelV2(mat_dim=16, n_layers=6, n_heads=4,
                                      max_len=1024, vocab_size=50257, dropout=0.1)

    print(f"Model: {model.count_parameters():,} params")
    train(model, config)
