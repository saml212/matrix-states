"""
Main H100 Experiment: Autoregressive Matrix Thinker.

The full novel architecture:
- 16×16 matrix tokens
- 3D matrix product attention (rows at pos s couple with cols at pos t)
- SiLU activation (SwiGLU pattern with gate*value)
- Multiplicative composition (I+Δ)·M·(I+Γ)
- PonderNet halting with rank-biased convergence
- Shared thinking layers applied iteratively

This is the model we're testing. Multiple configs run in parallel.
"""

import sys
import os
import math
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, get_dataloaders, get_device

# Import the TRUE autoregressive thinker (v2 — appends thoughts, doesn't refine in place)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from matrix_thinker import AutoregressiveMatrixThinker


def train_thinker(model, config):
    """Training loop for the matrix thinker (handles the info dict output)."""
    device = get_device(config)
    model = model.to(device)

    meta = json.load(open(os.path.join(config.data_dir, "meta.json")))
    vocab_size = meta["vocab_size"]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f" {config.experiment_name}")
    print(f" Params: {n_params:,} | Device: {device}")
    print(f" Vocab: {vocab_size:,} | Seq len: {config.seq_len}")
    print(f" Max thoughts: {model.max_thoughts}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr,
        weight_decay=config.weight_decay, betas=(0.9, 0.98)
    )

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_loader, val_loader = get_dataloaders(config)
    save_dir = os.path.join(config.save_dir, config.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    training_curve = []
    data_iter = iter(train_loader)

    model.train()
    while step < config.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        use_amp = config.use_bfloat16 and device.type == "cuda"
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, info = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            ppl = math.exp(min(loss.item(), 20))
            print(f"[{elapsed:7.0f}s] Step {step:6d} | Loss {loss.item():.4f} | PPL {ppl:.1f} "
                  f"| Steps {info['expected_steps']:.1f} "
                  f"| Ranks {[f'{r:.1f}' for r in info['mean_ranks_per_step']]}")

        if step % config.eval_interval == 0:
            model.eval()
            val_loss = 0; val_n = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    if val_n >= config.eval_batches: break
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits, _ = model(vx)
                    val_loss += F.cross_entropy(vlogits.reshape(-1, vocab_size), vy.reshape(-1)).item()
                    val_n += 1

            val_loss /= max(val_n, 1)
            val_ppl = math.exp(min(val_loss, 20))
            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                marker = " *"

            print(f"  → Val Loss {val_loss:.4f} | Val PPL {val_ppl:.1f}{marker}")
            training_curve.append({
                "step": step, "train_loss": loss.item(),
                "val_loss": val_loss, "val_ppl": val_ppl,
            })
            model.train()

    elapsed = time.time() - start_time
    results = {
        "experiment": config.experiment_name,
        "params": n_params,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "training_time_s": elapsed,
        "steps": step,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        "training_curve": training_curve,
    }
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n Done: {config.experiment_name} | Best PPL: {results['best_val_ppl']:.1f} | {elapsed/60:.0f}min")
    return results


# ─── Configs for 8 GPUs ────────────────────────────────────────

CONFIGS = {
    # GPU 0: Main architecture — mat_dim=16, 2 layers, 200 thoughts
    "thinker_d16_deep": {
        "mat_dim": 16, "n_thinking_layers": 2, "max_thoughts": 200,
        "n_heads": 4, "experiment_name": "thinker_d16_deep",
    },
    # GPU 1: Wider matrices
    "thinker_d24_wide": {
        "mat_dim": 24, "n_thinking_layers": 2, "max_thoughts": 200,
        "n_heads": 4, "experiment_name": "thinker_d24_wide",
    },
    # GPU 2: Widest matrices
    "thinker_d32": {
        "mat_dim": 32, "n_thinking_layers": 1, "max_thoughts": 200,
        "n_heads": 8, "experiment_name": "thinker_d32",
    },
    # GPU 3: Deeper thinking per step (3 layers)
    "thinker_d16_3layer": {
        "mat_dim": 16, "n_thinking_layers": 3, "max_thoughts": 200,
        "n_heads": 4, "experiment_name": "thinker_d16_3layer",
    },
    # GPU 4: Minimal layers, pure iteration
    "thinker_d16_1layer": {
        "mat_dim": 16, "n_thinking_layers": 1, "max_thoughts": 200,
        "n_heads": 4, "experiment_name": "thinker_d16_1layer",
    },
    # GPU 5: Frobenius attention (no 3D — ablation)
    "thinker_d16_frobenius": {
        "mat_dim": 16, "n_thinking_layers": 2, "max_thoughts": 200,
        "n_heads": 4, "experiment_name": "thinker_d16_frobenius",
        # TODO: pass flag to switch attention type
    },
    # GPU 6: Vector transformer baseline
    "vector_baseline": None,
    # GPU 7: Matrix with flattening baseline
    "matrix_flatten_baseline": None,
}


if __name__ == "__main__":
    from matrix_thinker import AutoregressiveMatrixThinker

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    if cfg is None:
        print(f"Config {args.config} uses a separate script.")
        sys.exit(0)

    if args.mini:
        train_config = TrainConfig(
            data_dir="/Volumes/1TB_SSD/learned-representations/data/reasoning",
            seq_len=64, batch_size=4, max_steps=100, lr=1e-3, warmup_steps=10,
            dropout=0.1, eval_interval=50, log_interval=25,
            save_dir="/Volumes/1TB_SSD/learned-representations/results/h100_test",
            experiment_name=cfg["experiment_name"] + "_mini",
            device="mps", compile_model=False, use_bfloat16=False,
        )
        model = AutoregressiveMatrixThinker(
            mat_dim=min(cfg["mat_dim"], 8),
            n_thinking_layers=min(cfg["n_thinking_layers"], 1),
            max_thoughts=min(cfg["max_thoughts"], 4),
            n_heads=min(cfg["n_heads"], 2),
            max_len=128, vocab_size=50257, dropout=0.1
        )
    else:
        train_config = TrainConfig(
            data_dir="/data/reasoning",  # OpenR1-Math reasoning data
            seq_len=512, batch_size=16, max_steps=20000, lr=3e-4, warmup_steps=500,
            dropout=0.1, eval_interval=500, log_interval=50,
            save_dir="/results",
            experiment_name=cfg["experiment_name"],
            device="cuda", compile_model=False,
            use_bfloat16=True,
        )
        model = AutoregressiveMatrixThinker(
            mat_dim=cfg["mat_dim"],
            n_thinking_layers=cfg["n_thinking_layers"],
            max_thoughts=cfg["max_thoughts"],  # up to 200
            n_heads=cfg["n_heads"],
            max_len=2048, vocab_size=50257, dropout=0.1
        )

    print(f"Model: {model.count_parameters():,} params")
    train_thinker(model, train_config)
