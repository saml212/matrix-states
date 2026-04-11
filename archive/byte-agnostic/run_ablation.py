#!/usr/bin/env python3
"""
Quick ablation: fixed-stride segmentation vs learned segmentation.
Uses the same data as Run 3 (10MB, seq_len=512) for fast comparison.
"""

import sys
import os
import time
import json
import math
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ablations import AblationModel
from src.data import ByteDataset


def train_ablation(name, model, train_loader, val_loader, device,
                   n_epochs=5, lr=3e-4, results_dir='results/ablations'):
    """Train an ablation model and return final metrics."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = n_epochs * len(train_loader)
    warmup = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"Params: {n_params:,} | Epochs: {n_epochs} | Steps: {total_steps:,}")
    print(f"{'='*60}\n")

    best_val = float('inf')
    history = {'train_bpc': [], 'val_bpc': [], 'step': []}
    global_step = 0
    start = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, bp, sc = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))

            # Add variance loss if segmented
            if model.segmenter is not None and hasattr(model.segmenter, 'boundary_variance_loss'):
                loss = loss + 0.01 * model.segmenter.boundary_variance_loss(sc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_n += 1
            global_step += 1

            if global_step % 100 == 0:
                bpc = epoch_loss / epoch_n / math.log(2)
                elapsed = time.time() - start
                print(f"  [{elapsed:5.0f}s] Step {global_step:5d} | BPC {bpc:.3f}")

        # Eval
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                if val_n >= 50:
                    break
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x)
                val_loss += F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
                val_n += 1

        val_bpc = val_loss / val_n / math.log(2)
        train_bpc = epoch_loss / epoch_n / math.log(2)
        best_val = min(best_val, val_bpc)

        history['train_bpc'].append(train_bpc)
        history['val_bpc'].append(val_bpc)
        history['step'].append(global_step)

        print(f"  Epoch {epoch+1}: Train BPC {train_bpc:.3f} | Val BPC {val_bpc:.3f} | Best {best_val:.3f}")

    elapsed = time.time() - start
    print(f"\n  {name} done in {elapsed:.0f}s | Best Val BPC: {best_val:.3f}")

    with open(results_dir / f'{name}_history.json', 'w') as f:
        json.dump({'name': name, 'params': n_params, 'best_val_bpc': best_val,
                   'history': history, 'elapsed_s': elapsed}, f, indent=2)

    return best_val, n_params


def main():
    device = 'mps'
    data_path = '/Volumes/1TB_SSD/learned-representations/data/text.bin'
    seq_len = 512
    batch_size = 8
    n_epochs = 5  # Quick — 5 epochs for comparison

    dataset = ByteDataset(data_path, seq_len)
    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    results = {}

    # 1. Learned segmentation + PHM (our approach)
    m1 = AblationModel(
        d_model=640, n_heads=10, n_layers=6, d_ff=2560, phm_n=4,
        max_len=2048, seg_d_model=320, target_compression=4.0,
        use_phm=True, segmenter_type='learned'
    ).to(device)
    bpc, params = train_ablation('learned_seg_phm', m1, train_loader, val_loader, device, n_epochs)
    results['learned_seg_phm'] = {'bpc': bpc, 'params': params}
    del m1; torch.mps.empty_cache()

    # 2. Fixed-stride segmentation + PHM
    m2 = AblationModel(
        d_model=640, n_heads=10, n_layers=6, d_ff=2560, phm_n=4,
        max_len=2048, target_compression=4.0,
        use_phm=True, segmenter_type='fixed'
    ).to(device)
    bpc, params = train_ablation('fixed_seg_phm', m2, train_loader, val_loader, device, n_epochs)
    results['fixed_seg_phm'] = {'bpc': bpc, 'params': params}
    del m2; torch.mps.empty_cache()

    # 3. Learned segmentation + Standard Linear (no PHM)
    m3 = AblationModel(
        d_model=640, n_heads=10, n_layers=6, d_ff=2560,
        max_len=2048, seg_d_model=320, target_compression=4.0,
        use_phm=False, segmenter_type='learned'
    ).to(device)
    bpc, params = train_ablation('learned_seg_standard', m3, train_loader, val_loader, device, n_epochs)
    results['learned_seg_standard'] = {'bpc': bpc, 'params': params}
    del m3; torch.mps.empty_cache()

    # Print summary
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Params':>12} {'Val BPC':>10}")
    print("-"*52)
    for name, r in sorted(results.items(), key=lambda x: x[1]['bpc']):
        print(f"{name:<30} {r['params']:>12,} {r['bpc']:>10.3f}")
    print("="*60)

    with open('results/ablations/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/ablations/summary.json")


if __name__ == '__main__':
    main()
