"""
Training loop for BytePHMTransformer V3.

V3: Forced segmentation (top-K boundaries). No collapse possible.
Loss = prediction + variance regularization + boundary entropy.
"""

import os
import sys
import time
import json
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .model import BytePHMTransformer
from .data import prepare_text_data, ByteDataset


def compute_loss(model, x, y, var_weight=0.01, entropy_weight=0.001):
    """Compute losses for V3 model."""
    logits, boundary_probs, segment_counts = model(x)

    # Next-byte prediction loss
    pred_loss = F.cross_entropy(
        logits.reshape(-1, 256),
        y.reshape(-1),
        reduction='mean'
    )

    # Segment length variance loss (encourage uniform-ish segments)
    var_loss = model.segmenter.boundary_variance_loss(segment_counts)

    # Boundary entropy loss (encourage sharp boundary decisions)
    ent_loss = model.segmenter.boundary_entropy_loss(boundary_probs)

    total_loss = pred_loss + var_weight * var_loss + entropy_weight * ent_loss

    with torch.no_grad():
        boundary_rate = (segment_counts > 0).float().sum(dim=-1).mean().item() / x.shape[1]
        n_segs = segment_counts.shape[1]
        # Segment length stats
        lengths = segment_counts[segment_counts > 0]
        mean_len = lengths.mean().item() if len(lengths) > 0 else 0
        std_len = lengths.std().item() if len(lengths) > 1 else 0

    diag = {
        'pred_loss': pred_loss.item(),
        'var_loss': var_loss.item(),
        'entropy_loss': ent_loss.item(),
        'n_segments': n_segs,
        'mean_seg_length': mean_len,
        'std_seg_length': std_len,
    }
    return total_loss, diag


def evaluate(model, dataloader, device, max_batches=50):
    model.eval()
    total_pred = 0
    total_mean_len = 0
    total_std_len = 0
    n = 0
    with torch.no_grad():
        for x, y in dataloader:
            if n >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, _, segment_counts = model(x)
            pred = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            total_pred += pred.item()
            lengths = segment_counts[segment_counts > 0]
            total_mean_len += lengths.mean().item() if len(lengths) > 0 else 0
            total_std_len += lengths.std().item() if len(lengths) > 1 else 0
            n += 1
    model.train()
    n = max(n, 1)
    return {
        'pred_loss': total_pred / n,
        'bpc': total_pred / n / math.log(2),
        'mean_seg_length': total_mean_len / n,
        'std_seg_length': total_std_len / n,
    }


def train(config_path: str = None, config: dict = None):
    if config is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

    device = config.get('device', 'mps')
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    ssd_root = Path(config.get('ssd_root', '/Volumes/1TB_SSD/learned-representations'))
    results_dir = Path(config.get('results_dir', 'results/phase1'))
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = prepare_text_data(
        output_path=str(ssd_root / 'data' / 'text.bin'),
        size_mb=config.get('data_size_mb', 10),
        cache_dir=str(ssd_root / 'data' / 'cache'),
    )

    seq_len = config.get('seq_len', 512)
    batch_size = config.get('batch_size', 8)
    target_compression = config.get('target_compression', 4.0)

    dataset = ByteDataset(data_path, seq_len)
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.05))
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=True
    )

    model = BytePHMTransformer(
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 4),
        d_ff=config.get('d_ff', None),
        phm_n=config.get('phm_n', 4),
        dropout=config.get('dropout', 0.1),
        max_len=config.get('max_len', 2048),
        seg_layers=config.get('seg_layers', 1),
        seg_d_model=config.get('seg_d_model', None),
        target_compression=target_compression,
    ).to(device)

    n_params = model.count_parameters()
    n_segs = seq_len // int(target_compression)
    print(f"\nModel: {n_params:,} params | Device: {device}")
    print(f"Compression: {target_compression}x → {n_segs} segments per {seq_len}-byte sequence")
    print(f"Train: {n_train:,} seqs | Val: {n_val:,} seqs | Batch: {batch_size}")
    print(f"Steps/epoch: {len(train_loader):,}")

    lr = config.get('lr', 3e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.98),
    )

    n_epochs = config.get('n_epochs', 20)
    total_steps = n_epochs * len(train_loader)
    warmup_steps = config.get('warmup_steps', min(500, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    tau_start = config.get('tau_start', 1.5)
    tau_end = config.get('tau_end', 0.2)
    var_weight = config.get('var_weight', 0.01)
    entropy_weight = config.get('entropy_weight', 0.001)
    log_interval = config.get('log_interval', 25)
    eval_interval = config.get('eval_interval', 100)

    checkpoint_dir = ssd_root / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_pred_loss': [], 'var_loss': [], 'entropy_loss': [],
        'val_pred_loss': [], 'val_bpc': [],
        'mean_seg_length': [], 'std_seg_length': [],
        'lr': [], 'tau': [], 'step': [],
    }

    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()

    print(f"\nTraining {n_epochs} epochs ({total_steps:,} steps)")
    print(f"LR: {lr} | Warmup: {warmup_steps} | τ: {tau_start}→{tau_end}")
    print(f"Var weight: {var_weight} | Entropy weight: {entropy_weight}\n")

    for epoch in range(n_epochs):
        model.train()
        epoch_pred = 0
        epoch_steps = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            progress = global_step / max(total_steps, 1)
            tau = tau_start * (tau_end / tau_start) ** progress
            model.segmenter.set_tau(tau)

            optimizer.zero_grad()
            total_loss, diag = compute_loss(model, x, y, var_weight, entropy_weight)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_pred += diag['pred_loss']
            epoch_steps += 1
            global_step += 1

            if global_step % log_interval == 0:
                avg_pred = epoch_pred / epoch_steps
                bpc = avg_pred / math.log(2)
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]

                history['train_pred_loss'].append(avg_pred)
                history['var_loss'].append(diag['var_loss'])
                history['entropy_loss'].append(diag['entropy_loss'])
                history['mean_seg_length'].append(diag['mean_seg_length'])
                history['std_seg_length'].append(diag['std_seg_length'])
                history['lr'].append(current_lr)
                history['tau'].append(tau)
                history['step'].append(global_step)

                print(f"[{elapsed:6.0f}s] Step {global_step:5d} | "
                      f"Pred {avg_pred:.4f} BPC {bpc:.3f} | "
                      f"Seg μ={diag['mean_seg_length']:.1f} σ={diag['std_seg_length']:.1f} | "
                      f"τ {tau:.3f} | LR {current_lr:.2e} | Grad {grad_norm:.2f}")

            if global_step % eval_interval == 0:
                val = evaluate(model, val_loader, device)
                history['val_pred_loss'].append(val['pred_loss'])
                history['val_bpc'].append(val['bpc'])

                print(f"  → Val BPC {val['bpc']:.3f} | "
                      f"Seg μ={val['mean_seg_length']:.1f} σ={val['std_seg_length']:.1f}")

                if val['pred_loss'] < best_val_loss:
                    best_val_loss = val['pred_loss']
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': global_step,
                        'val_loss': best_val_loss,
                        'config': config,
                    }, checkpoint_dir / 'best.pt')
                    print(f"  → Saved best (val_loss={best_val_loss:.4f})")

        print(f"\n--- Epoch {epoch+1}/{n_epochs} complete ---\n")

    torch.save({
        'model_state_dict': model.state_dict(),
        'step': global_step,
        'config': config,
    }, checkpoint_dir / 'final.pt')

    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDone in {total_time/3600:.1f}h | Best val loss: {best_val_loss:.4f}")
    return model, history


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/phase1.yaml'
    train(config_path)
