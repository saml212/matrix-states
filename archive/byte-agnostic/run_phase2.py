#!/usr/bin/env python3
"""
Phase 2: Multi-modal training on text + images.
The model sees raw bytes from both domains with NO domain labels.
The key question: does the segmenter learn different boundary patterns?
"""

import sys
import os
import time
import json
import math
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import BytePHMTransformer
from src.data import ByteDataset, MultiDomainDataset


def compute_loss(model, x, y, var_weight=0.01, entropy_weight=0.001):
    logits, boundary_probs, segment_counts = model(x)
    pred_loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    var_loss = model.segmenter.boundary_variance_loss(segment_counts)
    ent_loss = model.segmenter.boundary_entropy_loss(boundary_probs)
    total = pred_loss + var_weight * var_loss + entropy_weight * ent_loss

    with torch.no_grad():
        n_segs = segment_counts.shape[1]
        lengths = segment_counts[segment_counts > 0]
        mean_len = lengths.mean().item() if len(lengths) > 0 else 0
        std_len = lengths.std().item() if len(lengths) > 1 else 0

    return total, {
        'pred_loss': pred_loss.item(),
        'mean_seg_length': mean_len,
        'std_seg_length': std_len,
    }


def analyze_domain_boundaries(model, text_path, image_path, device, n_samples=200, seq_len=1024):
    """Analyze boundary patterns per domain."""
    model.eval()
    results = {}

    for domain, path in [('text', text_path), ('images', image_path)]:
        data = np.memmap(path, dtype=np.uint8, mode='r')
        seg_lengths_all = []

        for i in range(n_samples):
            start = i * seq_len
            if start + seq_len >= len(data):
                break
            chunk = data[start:start + seq_len]
            x = torch.tensor([chunk.tolist()], dtype=torch.long, device=device)

            with torch.no_grad():
                _, bp, sc, _ = model.segmenter(x, hard=True)

            lengths = sc[sc > 0]
            seg_lengths_all.extend(lengths.cpu().tolist())

        results[domain] = {
            'mean_seg_length': np.mean(seg_lengths_all) if seg_lengths_all else 0,
            'std_seg_length': np.std(seg_lengths_all) if seg_lengths_all else 0,
            'median_seg_length': np.median(seg_lengths_all) if seg_lengths_all else 0,
        }
        print(f"  {domain}: μ={results[domain]['mean_seg_length']:.1f} "
              f"σ={results[domain]['std_seg_length']:.1f} "
              f"median={results[domain]['median_seg_length']:.1f}")

    model.train()
    return results


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/phase2.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = config.get('device', 'mps')
    ssd_root = Path(config.get('ssd_root'))
    results_dir = Path(config.get('results_dir', 'results/phase2'))
    results_dir.mkdir(parents=True, exist_ok=True)

    text_path = config['text_path']
    image_path = config['image_path']
    seq_len = config.get('seq_len', 1024)
    batch_size = config.get('batch_size', 16)

    # Multi-domain dataset — interleaved, no labels
    dataset = MultiDomainDataset([text_path, image_path], seq_len)
    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val

    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0
    )

    model = BytePHMTransformer(
        d_model=config.get('d_model', 640),
        n_heads=config.get('n_heads', 10),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 2560),
        phm_n=config.get('phm_n', 4),
        dropout=config.get('dropout', 0.1),
        max_len=config.get('max_len', 4096),
        seg_layers=config.get('seg_layers', 1),
        seg_d_model=config.get('seg_d_model', 320),
        target_compression=config.get('target_compression', 4.0),
    ).to(device)

    print(f"\n{'='*60}")
    print(f"PHASE 2: MULTI-MODAL (text + images)")
    print(f"{'='*60}")
    print(f"Model: {model.count_parameters():,} params | Device: {device}")
    print(f"Text: {os.path.getsize(text_path)/1e6:.0f}MB | Images: {os.path.getsize(image_path)/1e6:.0f}MB")
    print(f"Train: {n_train:,} seqs | Val: {n_val:,} seqs | Batch: {batch_size}")
    print(f"Steps/epoch: {len(train_loader):,}\n")

    lr = config.get('lr', 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    n_epochs = config.get('n_epochs', 10)
    total_steps = n_epochs * len(train_loader)
    warmup = config.get('warmup_steps', 500)

    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    tau_start = config.get('tau_start', 1.5)
    tau_end = config.get('tau_end', 0.2)
    var_weight = config.get('var_weight', 0.01)
    entropy_weight = config.get('entropy_weight', 0.001)
    log_interval = config.get('log_interval', 50)
    eval_interval = config.get('eval_interval', 200)

    checkpoint_dir = ssd_root / 'checkpoints' / 'phase2'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {'train_bpc': [], 'val_bpc': [], 'seg_mu': [], 'seg_sigma': [], 'step': []}
    global_step = 0
    best_val = float('inf')
    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_n = 0

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

            epoch_loss += diag['pred_loss']
            epoch_n += 1
            global_step += 1

            if global_step % log_interval == 0:
                bpc = epoch_loss / epoch_n / math.log(2)
                elapsed = time.time() - start_time
                print(f"[{elapsed:6.0f}s] Step {global_step:5d} | "
                      f"BPC {bpc:.3f} | Seg μ={diag['mean_seg_length']:.1f} σ={diag['std_seg_length']:.1f} | "
                      f"Grad {grad_norm:.2f}")

                history['train_bpc'].append(bpc)
                history['seg_mu'].append(diag['mean_seg_length'])
                history['seg_sigma'].append(diag['std_seg_length'])
                history['step'].append(global_step)

            if global_step % eval_interval == 0:
                model.eval()
                val_loss = 0
                val_n = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        if val_n >= 50: break
                        vx, vy = vx.to(device), vy.to(device)
                        logits, _, _ = model(vx)
                        val_loss += F.cross_entropy(logits.reshape(-1, 256), vy.reshape(-1)).item()
                        val_n += 1
                val_bpc = val_loss / val_n / math.log(2)
                history['val_bpc'].append(val_bpc)
                print(f"  → Val BPC {val_bpc:.3f}")

                if val_loss / val_n < best_val:
                    best_val = val_loss / val_n
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'step': global_step,
                        'val_loss': best_val,
                        'config': config,
                    }, checkpoint_dir / 'best.pt')
                    print(f"  → Saved best (val_loss={best_val:.4f})")

                # Domain-specific boundary analysis every 1000 steps
                if global_step % 1000 == 0:
                    print("\n  Domain boundary analysis:")
                    domain_results = analyze_domain_boundaries(
                        model, text_path, image_path, device, n_samples=100, seq_len=seq_len
                    )
                    with open(results_dir / f'domain_boundaries_step{global_step}.json', 'w') as f:
                        json.dump(domain_results, f, indent=2)
                    print()

                model.train()

        print(f"\n--- Epoch {epoch+1}/{n_epochs} complete ---\n")

    # Final analysis
    print("\n" + "="*60)
    print("FINAL DOMAIN BOUNDARY ANALYSIS")
    print("="*60)
    model.eval()
    domain_results = analyze_domain_boundaries(
        model, text_path, image_path, device, n_samples=500, seq_len=seq_len
    )

    with open(results_dir / 'final_domain_analysis.json', 'w') as f:
        json.dump(domain_results, f, indent=2)

    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'step': global_step,
        'config': config,
    }, checkpoint_dir / 'final.pt')

    elapsed = time.time() - start_time
    print(f"\nPhase 2 complete in {elapsed/3600:.1f}h | Best val loss: {best_val:.4f}")


if __name__ == '__main__':
    main()
