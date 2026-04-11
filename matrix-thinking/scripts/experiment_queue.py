#!/usr/bin/env python3
"""
Experiment Queue: Runs continuously on 8x H100.
After Round 1 (matrix vs vector), this chains experiments back-to-back.
Each experiment saves results then the next one launches immediately.

The queue explores the architecture space systematically:
- Round 2: 3D attention vs Frobenius (does structured attention help?)
- Round 3: Expansion factor sweep (how much thinking capacity matters?)
- Round 4: mat_dim sweep (how big should the matrices be?)
- Round 5: Loss modifications (rank convergence pressure)
- Round 6: Best config on big reasoning data
"""

# This file documents the queue. Each round is a separate launch.
# The launcher script chains them.

ROUNDS = {
    "round2_3d_attention": {
        "description": "Bring back 3D matrix product attention. Does structured coupling help?",
        "gpus_0_3": {"model": "matrix", "attention": "3d_product", "mat_dim": 32, "n_layers": 8, "n_iterations": 8},
        "gpus_4_7": {"model": "matrix", "attention": "frobenius", "mat_dim": 32, "n_layers": 8, "n_iterations": 8},
        "question": "Does 3D attention (rows couple with cols) beat Frobenius (scalar scores)?",
    },
    "round3_expansion": {
        "description": "How much capacity do thinking layers need?",
        "gpus_0_3": {"model": "matrix", "expansion": 2, "mat_dim": 32, "n_layers": 8, "n_iterations": 8},
        "gpus_4_7": {"model": "matrix", "expansion": 4, "mat_dim": 32, "n_layers": 8, "n_iterations": 8},
        "question": "Does 4x expansion beat 2x? How much thinking capacity is enough?",
    },
    "round4_mat_dim": {
        "description": "Scale up the matrix dimension",
        "gpus_0_3": {"model": "matrix", "mat_dim": 48, "n_layers": 8, "n_iterations": 8},
        "gpus_4_7": {"model": "matrix", "mat_dim": 64, "n_layers": 6, "n_iterations": 8},
        "question": "Bigger matrices = richer structure. Where's the sweet spot?",
    },
    "round5_rank_loss": {
        "description": "Add rank convergence pressure to the loss",
        "gpus_0_3": {"model": "matrix", "rank_loss_weight": 0.1, "n_iterations": 8},
        "gpus_4_7": {"model": "matrix", "rank_loss_weight": 0.0, "n_iterations": 8},
        "question": "Does pressuring later iterations toward low rank improve predictions?",
    },
    "round6_best_on_reasoning": {
        "description": "Best architecture on big reasoning data",
        "all_8_gpus": True,
        "data_dir": "/toy_story_slam/data",  # Full reasoning corpus
        "config": "best from rounds 2-5",
        "question": "Does the winner scale to harder data?",
    },
}
