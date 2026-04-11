#!/usr/bin/env python3
"""
Path B: Algebraic Regularization Experiment.

Novel contribution — first attempt to force PHM layers toward genuine
algebraic structure through regularization losses.

Conditions:
1. quaternion_init + algebra_reg (Path B — our novel approach)
2. quaternion_init + no reg (does the algebra drift without reg?)
3. quaternion_fixed (upper bound — best possible quaternion performance)
4. learned + no reg (baseline — converges to nilpotent)

All on multi-modal data (text + images).
"""

import sys, os, time, json, math, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import BytePHMTransformer
from src.data import ByteDataset, MultiDomainDataset
from src.algebra_reg import full_algebra_regularization
from src.phm import PHMLinear


def measure_algebra(model):
    """Quick algebra diagnostics."""
    traces = []
    comms = []
    for name, mod in model.named_modules():
        if isinstance(mod, PHMLinear):
            A = mod.get_algebra()
            sq_tr = [(A[i] @ A[i]).trace().item() for i in range(4)]
            traces.append(sq_tr)
            # Commutativity of generators
            c = 0
            for i in range(1, 4):
                for j in range(i+1, 4):
                    c += torch.norm(A[i]@A[j] - A[j]@A[i]).item()
            comms.append(c / 3)
    avg_tr = np.mean(traces, axis=0).tolist() if traces else [0]*4
    avg_comm = np.mean(comms) if comms else 0
    return avg_tr, avg_comm


def train_condition(name, algebra_mode, use_algebra_reg, train_loader, val_loader,
                    text_val_loader, image_val_loader, device, max_steps=10000, lr=3e-4):
    """Train one condition."""
    results_dir = Path(f'results/algebra_reg/{name}')
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024, phm_n=4,
               max_len=2048, seg_d_model=128, target_compression=4.0, dropout=0.1)

    model = BytePHMTransformer(**cfg, algebra_mode=algebra_mode).to(device)
    n_params = model.count_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warmup = min(300, max_steps // 10)

    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        p = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*65}")
    print(f" {name} | mode={algebra_mode} | alg_reg={use_algebra_reg} | {n_params:,} params")
    print(f"{'='*65}")

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
        progress = global_step / max(max_steps, 1)
        model.segmenter.set_tau(1.5 * (0.2 / 1.5) ** progress)

        optimizer.zero_grad()
        logits, bp, sc = model(x)
        pred_loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        var_loss = model.segmenter.boundary_variance_loss(sc)

        total_loss = pred_loss + 0.01 * var_loss

        # Algebraic regularization (Path B)
        alg_diag = {}
        if use_algebra_reg:
            alg_loss, alg_diag = full_algebra_regularization(
                model, closure_weight=0.1, division_weight=0.05,
                anticomm_weight=0.1, trace_weight=0.05
            )
            total_loss = total_loss + alg_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % 1000 == 0:
            elapsed = time.time() - start
            bpc = pred_loss.item() / math.log(2)
            avg_tr, avg_comm = measure_algebra(model)
            alg_str = f"| AlgReg {alg_diag.get('total_algebra_reg', 0):.4f}" if use_algebra_reg else ""
            print(f"  [{elapsed:5.0f}s] Step {global_step:5d} | BPC {bpc:.3f} "
                  f"| tr(A²)=[{avg_tr[0]:.1f},{avg_tr[1]:.1f},{avg_tr[2]:.1f},{avg_tr[3]:.1f}] "
                  f"| comm={avg_comm:.3f} {alg_str}")

    # Final eval
    model.eval()
    domain_bpc = {}
    for dname, loader in [('text', text_val_loader), ('images', image_val_loader)]:
        total = 0; n = 0
        with torch.no_grad():
            for vx, vy in loader:
                if n >= 50: break
                vx, vy = vx.to(device), vy.to(device)
                logits, _, _ = model(vx)
                total += F.cross_entropy(logits.reshape(-1, 256), vy.reshape(-1)).item()
                n += 1
        domain_bpc[dname] = total / n / math.log(2)

    avg_tr, avg_comm = measure_algebra(model)
    elapsed = time.time() - start

    print(f"\n  RESULT: Text {domain_bpc['text']:.3f} | Image {domain_bpc['images']:.3f}")
    print(f"  Algebra: tr(A²)={[f'{t:.1f}' for t in avg_tr]} | comm={avg_comm:.3f}")
    print(f"  Time: {elapsed:.0f}s")

    # Quaternion distance
    quat_ref = sorted([4.0, -4.0, -4.0, -4.0])
    quat_dist = np.linalg.norm(np.array(sorted(avg_tr)) - np.array(quat_ref))

    result = {
        'name': name, 'algebra_mode': algebra_mode, 'use_algebra_reg': use_algebra_reg,
        'params': n_params, 'text_bpc': domain_bpc['text'], 'image_bpc': domain_bpc['images'],
        'avg_squared_traces': avg_tr, 'avg_commutativity': avg_comm,
        'quaternion_distance': quat_dist, 'elapsed_s': elapsed,
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
    train_ds, val_ds = torch.utils.data.random_split(multi_ds, [len(multi_ds) - n_val, n_val])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    text_ds = ByteDataset(text_path, seq_len)
    image_ds = ByteDataset(image_path, seq_len)
    text_val = torch.utils.data.Subset(text_ds, range(len(text_ds)-500, len(text_ds)))
    image_val = torch.utils.data.Subset(image_ds, range(len(image_ds)-500, len(image_ds)))
    text_val_loader = torch.utils.data.DataLoader(text_val, batch_size=batch_size, drop_last=True)
    image_val_loader = torch.utils.data.DataLoader(image_val, batch_size=batch_size, drop_last=True)

    loaders = (train_loader, val_loader, text_val_loader, image_val_loader)
    results = {}

    # Condition 1: PATH B — quaternion init + algebraic regularization (NOVEL)
    results['pathB_quat_init_reg'] = train_condition(
        'pathB_quat_init_reg', 'quaternion_init', True, *loaders, device, max_steps)
    torch.mps.empty_cache()

    # Condition 2: quaternion init, no reg (does algebra drift?)
    results['quat_init_noreg'] = train_condition(
        'quat_init_noreg', 'quaternion_init', False, *loaders, device, max_steps)
    torch.mps.empty_cache()

    # Condition 3: fixed quaternion (upper bound)
    results['quat_fixed'] = train_condition(
        'quat_fixed', 'quaternion', False, *loaders, device, max_steps)
    torch.mps.empty_cache()

    # Condition 4: fully learned, no reg (baseline → nilpotent)
    results['learned_noreg'] = train_condition(
        'learned_noreg', 'learned', False, *loaders, device, max_steps)
    torch.mps.empty_cache()

    # Summary
    print("\n" + "="*80)
    print("ALGEBRAIC REGULARIZATION EXPERIMENT — RESULTS")
    print("="*80)
    print(f"{'Condition':<25} {'Text':>8} {'Image':>8} {'Quat Dist':>10} {'Comm':>8} {'tr(A²)':>30}")
    print("-"*89)
    for name, r in sorted(results.items(), key=lambda x: x[1]['text_bpc']):
        tr_str = ','.join(f'{t:.1f}' for t in r['avg_squared_traces'])
        print(f"{name:<25} {r['text_bpc']:>8.3f} {r['image_bpc']:>8.3f} "
              f"{r['quaternion_distance']:>10.2f} {r['avg_commutativity']:>8.3f} [{tr_str:>28}]")

    print("\n" + "="*80)
    print("KEY QUESTIONS")
    print("="*80)

    pb = results['pathB_quat_init_reg']
    qn = results['quat_init_noreg']
    qf = results['quat_fixed']
    ln = results['learned_noreg']

    # Does algebra reg prevent drift?
    print(f"\n1. Does algebra reg prevent drift from quaternion structure?")
    print(f"   With reg:    quat_dist = {pb['quaternion_distance']:.2f}")
    print(f"   Without reg: quat_dist = {qn['quaternion_distance']:.2f}")
    if pb['quaternion_distance'] < qn['quaternion_distance'] * 0.5:
        print(f"   → YES! Regularization preserves algebraic structure")
    else:
        print(f"   → No significant difference")

    # Does preserved algebra help performance?
    print(f"\n2. Does preserving quaternion algebra help multi-modal performance?")
    print(f"   Path B (reg):  Text={pb['text_bpc']:.3f}, Image={pb['image_bpc']:.3f}")
    print(f"   Fixed quat:    Text={qf['text_bpc']:.3f}, Image={qf['image_bpc']:.3f}")
    print(f"   Learned (nil):  Text={ln['text_bpc']:.3f}, Image={ln['image_bpc']:.3f}")

    best = min(results.items(), key=lambda x: x[1]['text_bpc'] + x[1]['image_bpc'])
    print(f"\n   Best overall: {best[0]} (text={best[1]['text_bpc']:.3f}, image={best[1]['image_bpc']:.3f})")

    Path('results/algebra_reg').mkdir(parents=True, exist_ok=True)
    with open('results/algebra_reg/summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results/algebra_reg/summary.json")


if __name__ == '__main__':
    main()
