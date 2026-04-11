#!/usr/bin/env python3
"""
Critical ablations for publishability.

1. Fixed-stride vs learned segmentation — is the benefit from LEARNING or just shorter context?
2. Random (untrained) segmenter vs trained — does domain differentiation come from learning?
3. PHM vs standard linear at matched params — does PHM structure help?

These are the experiments reviewers will demand.
"""

import sys, os, time, json, math, torch, copy
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import BytePHMTransformer
from src.ablations import AblationModel
from src.data import ByteDataset, MultiDomainDataset


def train_and_eval(name, model, train_loader, text_val_loader, image_val_loader,
                   device, max_steps=10000, lr=3e-4):
    """Train and return per-domain results + segment stats."""
    results_dir = Path(f'results/critical_ablations/{name}')
    results_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warmup = min(300, max_steps // 10)
    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        p = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f" {name} | {n_params:,} params | {max_steps} steps")
    print(f"{'='*60}")

    global_step = 0; start = time.time()
    data_iter = iter(train_loader)
    model.train()

    while global_step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        if hasattr(model, 'segmenter') and model.segmenter is not None:
            progress = global_step / max(max_steps, 1)
            model.segmenter.set_tau(1.5 * (0.2 / 1.5) ** progress)

        optimizer.zero_grad()
        logits, bp, sc = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if hasattr(model, 'segmenter') and model.segmenter is not None:
            if hasattr(model.segmenter, 'boundary_variance_loss'):
                loss = loss + 0.01 * model.segmenter.boundary_variance_loss(sc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % 2000 == 0:
            bpc = loss.item() / math.log(2)
            print(f"  [{time.time()-start:5.0f}s] Step {global_step:5d} | BPC {bpc:.3f}")

    # Per-domain eval
    model.eval()
    domain_results = {}
    for dname, loader in [('text', text_val_loader), ('images', image_val_loader)]:
        total_loss = 0; n = 0
        seg_lengths = []
        with torch.no_grad():
            for vx, vy in loader:
                if n >= 100: break
                vx, vy = vx.to(device), vy.to(device)
                logits, bp, sc = model(vx)
                total_loss += F.cross_entropy(logits.reshape(-1, 256), vy.reshape(-1)).item()
                if sc is not None:
                    lengths = sc[sc > 0]
                    seg_lengths.extend(lengths.cpu().tolist())
                n += 1
        bpc = total_loss / n / math.log(2)
        seg_std = np.std(seg_lengths) if seg_lengths else 0
        domain_results[dname] = {'bpc': bpc, 'seg_std': seg_std}
        print(f"  {dname}: BPC {bpc:.3f} | seg_std {seg_std:.2f}")

    elapsed = time.time() - start
    result = {
        'name': name, 'params': n_params, 'elapsed_s': elapsed,
        'text_bpc': domain_results['text']['bpc'],
        'image_bpc': domain_results['images']['bpc'],
        'text_seg_std': domain_results['text']['seg_std'],
        'image_seg_std': domain_results['images']['seg_std'],
        'seg_std_gap': abs(domain_results['images']['seg_std'] - domain_results['text']['seg_std']),
    }

    with open(results_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, default=float)
    return result


def main():
    device = 'mps'
    seq_len = 512
    batch_size = 16
    max_steps = 10000

    text_path = '/Volumes/1TB_SSD/learned-representations/data/text.bin'
    image_path = '/Volumes/1TB_SSD/learned-representations/data/images.bin'

    multi_ds = MultiDomainDataset([text_path, image_path], seq_len)
    n_val = max(1, int(len(multi_ds) * 0.05))
    train_ds, _ = torch.utils.data.random_split(multi_ds, [len(multi_ds) - n_val, n_val])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    text_ds = ByteDataset(text_path, seq_len)
    image_ds = ByteDataset(image_path, seq_len)
    text_val = torch.utils.data.Subset(text_ds, range(len(text_ds)-500, len(text_ds)))
    image_val = torch.utils.data.Subset(image_ds, range(len(image_ds)-500, len(image_ds)))
    text_val_loader = torch.utils.data.DataLoader(text_val, batch_size=batch_size, drop_last=True)
    image_val_loader = torch.utils.data.DataLoader(image_val, batch_size=batch_size, drop_last=True)

    cfg = dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_len=2048,
               seg_d_model=128, target_compression=4.0, dropout=0.1)

    results = {}

    # 1. LEARNED segmentation + PHM (our full model)
    print("\n### ABLATION 1: Learned Segmentation + PHM (full model)")
    m1 = BytePHMTransformer(**cfg, phm_n=4, algebra_mode='learned').to(device)
    results['learned_seg_phm'] = train_and_eval(
        'learned_seg_phm', m1, train_loader, text_val_loader, image_val_loader, device, max_steps)
    del m1; torch.mps.empty_cache()

    # 2. FIXED-STRIDE segmentation + PHM (is learning necessary?)
    print("\n### ABLATION 2: Fixed-Stride Segmentation + PHM")
    m2 = AblationModel(**cfg, phm_n=4, use_phm=True, segmenter_type='fixed').to(device)
    results['fixed_seg_phm'] = train_and_eval(
        'fixed_seg_phm', m2, train_loader, text_val_loader, image_val_loader, device, max_steps)
    del m2; torch.mps.empty_cache()

    # 3. Learned segmentation + STANDARD LINEAR (does PHM help?)
    print("\n### ABLATION 3: Learned Segmentation + Standard Linear")
    m3 = AblationModel(**cfg, use_phm=False, segmenter_type='learned').to(device)
    results['learned_seg_standard'] = train_and_eval(
        'learned_seg_standard', m3, train_loader, text_val_loader, image_val_loader, device, max_steps)
    del m3; torch.mps.empty_cache()

    # 4. NO segmentation (flat byte transformer)
    print("\n### ABLATION 4: No Segmentation (flat bytes)")
    m4 = AblationModel(**cfg, use_phm=True, segmenter_type='none').to(device)
    results['no_seg_phm'] = train_and_eval(
        'no_seg_phm', m4, train_loader, text_val_loader, image_val_loader, device, max_steps)
    del m4; torch.mps.empty_cache()

    # Summary
    print("\n" + "="*85)
    print("CRITICAL ABLATION RESULTS")
    print("="*85)
    print(f"{'Condition':<25} {'Params':>10} {'Text':>8} {'Image':>8} {'Seg σ gap':>10} {'Key Question'}")
    print("-"*85)
    for name, r in sorted(results.items(), key=lambda x: x[1].get('text_bpc', 99)):
        gap = r.get('seg_std_gap', 0)
        question = {
            'learned_seg_phm': 'Full model',
            'fixed_seg_phm': 'Learning necessary?',
            'learned_seg_standard': 'PHM necessary?',
            'no_seg_phm': 'Segmentation necessary?',
        }.get(name, '')
        print(f"{name:<25} {r['params']:>10,} {r['text_bpc']:>8.3f} {r['image_bpc']:>8.3f} "
              f"{gap:>10.2f} {question}")

    # Answer key questions
    print("\n" + "="*85)
    full = results['learned_seg_phm']
    fixed = results['fixed_seg_phm']
    std = results['learned_seg_standard']
    noseg = results['no_seg_phm']

    print(f"\nQ1: Does LEARNED segmentation beat FIXED-STRIDE?")
    diff = fixed['text_bpc'] - full['text_bpc']
    if diff > 0.05:
        print(f"  YES! Learned is {diff:.3f} BPC better on text")
    elif diff < -0.05:
        print(f"  NO. Fixed-stride is {-diff:.3f} BPC better")
    else:
        print(f"  INCONCLUSIVE. Difference is only {abs(diff):.3f} BPC")

    print(f"\nQ2: Does learned seg produce domain differentiation that fixed doesn't?")
    print(f"  Learned seg σ gap: {full['seg_std_gap']:.2f}")
    print(f"  Fixed seg σ gap:   {fixed['seg_std_gap']:.2f}")

    print(f"\nQ3: Does PHM help over standard linear?")
    diff = std['text_bpc'] - full['text_bpc']
    print(f"  PHM advantage: {diff:.3f} BPC on text")

    print(f"\nQ4: Does ANY segmentation help over no segmentation?")
    diff = noseg['text_bpc'] - full['text_bpc']
    print(f"  Segmentation advantage: {diff:.3f} BPC on text")

    Path('results/critical_ablations').mkdir(parents=True, exist_ok=True)
    with open('results/critical_ablations/summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results/critical_ablations/summary.json")


if __name__ == '__main__':
    main()
