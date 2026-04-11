"""
Common utilities for all H100 experiments.
Data loading, training loop, evaluation, logging.

Works on both H100 (CUDA + DDP) and Mac Mini (MPS or CPU).
"""

import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict


# ─── Config ────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    data_dir: str = "/Volumes/1TB_SSD/learned-representations/data/wikitext103_tokenized"
    seq_len: int = 512
    # Training
    batch_size: int = 64
    max_steps: int = 50000
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.1
    # Eval
    eval_interval: int = 1000
    eval_batches: int = 50
    # Logging
    log_interval: int = 100
    save_dir: str = "results"
    experiment_name: str = "unnamed"
    # Hardware
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    compile_model: bool = False  # True for H100
    use_ddp: bool = False
    use_fa3: bool = False
    use_bfloat16: bool = False
    seed: int = 42


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config.device)


# ─── Data ──────────────────────────────────────────────────────

class TokenDataset(torch.utils.data.Dataset):
    """Load pre-tokenized data from .pt files."""

    def __init__(self, data_path: str, seq_len: int):
        self.data = torch.load(data_path, weights_only=True)
        self.seq_len = seq_len
        self.n_sequences = len(self.data) // seq_len - 1

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def get_dataloaders(config: TrainConfig):
    train_ds = TokenDataset(os.path.join(config.data_dir, "train.pt"), config.seq_len)
    val_ds = TokenDataset(os.path.join(config.data_dir, "val.pt"), config.seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        drop_last=True, num_workers=0
    )
    return train_loader, val_loader


# ─── Training Loop ─────────────────────────────────────────────

def train(model, config: TrainConfig):
    device = get_device(config)
    model = model.to(device)

    # bfloat16 handled via autocast in training loop, not model.to()
    # Keeps optimizer states in float32 for stability

    if config.compile_model and device.type == "cuda":
        model = torch.compile(model, dynamic=False, fullgraph=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load vocab size from meta
    meta = json.load(open(os.path.join(config.data_dir, "meta.json")))
    vocab_size = meta["vocab_size"]

    print(f"\n{'='*60}")
    print(f" Experiment: {config.experiment_name}")
    print(f" Params: {n_params:,} | Device: {device}")
    print(f" Vocab: {vocab_size:,} | Seq len: {config.seq_len}")
    print(f" Batch: {config.batch_size} | Steps: {config.max_steps:,}")
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
    save_dir = Path(config.save_dir) / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
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

        # Mixed precision for CUDA, normal for MPS/CPU
        use_amp = config.use_bfloat16 and device.type == "cuda"
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            # Forward — handle both matrix models (return tuple) and vector models
            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        # Log
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            ppl = math.exp(min(loss.item(), 20))  # cap to avoid overflow
            print(f"[{elapsed:7.0f}s] Step {step:6d} | Loss {loss.item():.4f} | PPL {ppl:.1f} | LR {scheduler.get_last_lr()[0]:.2e}")

        # Eval
        if step % config.eval_interval == 0:
            model.eval()
            val_loss = 0
            val_n = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    if val_n >= config.eval_batches:
                        break
                    vx, vy = vx.to(device), vy.to(device)
                    vout = model(vx)
                    vlogits = vout[0] if isinstance(vout, tuple) else vout
                    val_loss += F.cross_entropy(vlogits.reshape(-1, vocab_size), vy.reshape(-1)).item()
                    val_n += 1

            val_loss /= max(val_n, 1)
            val_ppl = math.exp(min(val_loss, 20))
            marker = ""

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "best.pt")
                marker = " *"

            print(f"  → Val Loss {val_loss:.4f} | Val PPL {val_ppl:.1f}{marker}")

            training_curve.append({
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            })

            model.train()

    # Save final results
    elapsed = time.time() - start_time
    results = {
        "experiment": config.experiment_name,
        "params": n_params,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "training_time_s": elapsed,
        "steps": step,
        "config": asdict(config),
        "training_curve": training_curve,
    }

    with open(save_dir / "result.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print(f" Done: {config.experiment_name}")
    print(f" Best Val PPL: {results['best_val_ppl']:.1f}")
    print(f" Time: {elapsed/60:.0f} min")
    print(f" Saved to: {save_dir}")
    print(f"{'='*60}\n")

    return results
