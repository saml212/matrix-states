#!/usr/bin/env python3
"""
Algebra Ablation: Does quaternion structure help multi-modal processing?

Compares 4 conditions on interleaved text+image bytes:
1. 'quaternion'      — A fixed to Hamilton basis (not learnable)
2. 'quaternion_init' — A starts as Hamilton, can learn/drift
3. 'learned'         — A fully learnable (original PHM, our baseline)
4. 'standard'        — No PHM, standard nn.Linear

If quaternion structure genuinely helps multi-modal processing, conditions 1-2
should beat condition 3. If PHM is just parameter-efficient factorization,
condition 3 should match 1-2 (all beat 4 on parameter efficiency).
"""

import sys, os, time, json, math, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import BytePHMTransformer
from src.ablations import AblationModel
from src.data import ByteDataset, MultiDomainDataset


def train_and_eval(name, model, train_loader, val_loader, text_val_loader,
                   image_val_loader, device, max_steps=5000, lr=3e-4):
    """Train model for fixed step budget and return per-domain results."""
    results_dir = Path(f'results/algebra_ablation/{name}')
    results_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = max_steps
    warmup = min(300, total_steps // 10)

    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"CONDITION: {name}")
    print(f"Params: {n_params:,} | Steps: {total_steps:,}")
    print(f"{'='*60}")

    tau_start, tau_end = 1.5, 0.2
    global_step = 0
    start = time.time()
    epoch_loss = 0; epoch_n = 0

    model.train()
    data_iter = iter(train_loader)
    while global_step < total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        progress = global_step / max(total_steps, 1)
        if hasattr(model, 'segmenter') and model.segmenter is not None:
            model.segmenter.set_tau(tau_start * (tau_end / tau_start) ** progress)

        optimizer.zero_grad()
        logits, bp, sc = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if hasattr(model, 'segmenter') and model.segmenter is not None and hasattr(model.segmenter, 'boundary_variance_loss'):
            loss = loss + 0.01 * model.segmenter.boundary_variance_loss(sc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item(); epoch_n += 1; global_step += 1

        if global_step % 500 == 0:
            bpc = epoch_loss / epoch_n / math.log(2)
            print(f"  [{time.time()-start:5.0f}s] Step {global_step:5d} | BPC {bpc:.3f}")

    # Per-domain eval
        model.eval()
        domain_bpc = {}
        for dname, loader in [('combined', val_loader), ('text', text_val_loader), ('images', image_val_loader)]:
            total = 0; n = 0
            with torch.no_grad():
                for vx, vy in loader:
                    if n >= 50: break
                    vx, vy = vx.to(device), vy.to(device)
                    logits, _, _ = model(vx)
                    total += F.cross_entropy(logits.reshape(-1, 256), vy.reshape(-1)).item()
                    n += 1
            domain_bpc[dname] = total / n / math.log(2)

        print(f"  Final: Combined {domain_bpc['combined']:.3f} | "
              f"Text {domain_bpc['text']:.3f} | Image {domain_bpc['images']:.3f}")

    elapsed = time.time() - start
    result = {
        'name': name, 'params': n_params,
        'text_bpc': domain_bpc['text'],
        'image_bpc': domain_bpc['images'],
        'combined_bpc': domain_bpc['combined'],
        'elapsed_s': elapsed,
    }

    # PHM algebra analysis (if applicable)
    if name != 'standard':
        from src.phm import PHMLinear
        traces = []
        for mod_name, module in model.named_modules():
            if isinstance(module, PHMLinear):
                A = module.get_algebra()
                sq_traces = [(A[i] @ A[i]).trace().item() for i in range(A.shape[0])]
                traces.append(sq_traces)
        if traces:
            avg_traces = np.mean(traces, axis=0).tolist()
            result['avg_squared_traces'] = avg_traces
            print(f"  Avg tr(A_i^2): {[f'{t:.2f}' for t in avg_traces]}")

    with open(results_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, default=float)

    return result


def main():
    device = 'mps'
    seq_len = 512  # Shorter for faster ablation
    batch_size = 16
    max_steps = 10000  # ~20min per condition

    text_path = '/Volumes/1TB_SSD/learned-representations/data/text.bin'
    image_path = '/Volumes/1TB_SSD/learned-representations/data/images.bin'

    # Multi-domain dataset
    multi_ds = MultiDomainDataset([text_path, image_path], seq_len)
    n_val = max(1, int(len(multi_ds) * 0.05))
    n_train = len(multi_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(multi_ds, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # Per-domain val loaders
    text_ds = ByteDataset(text_path, seq_len)
    image_ds = ByteDataset(image_path, seq_len)
    text_val = torch.utils.data.Subset(text_ds, range(len(text_ds) - 500, len(text_ds)))
    image_val = torch.utils.data.Subset(image_ds, range(len(image_ds) - 500, len(image_ds)))
    text_val_loader = torch.utils.data.DataLoader(text_val, batch_size=batch_size, shuffle=False, drop_last=True)
    image_val_loader = torch.utils.data.DataLoader(image_val, batch_size=batch_size, shuffle=False, drop_last=True)

    # Common model config (smaller for speed)
    cfg = dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_len=2048,
               seg_d_model=128, target_compression=4.0, dropout=0.1)

    results = {}

    # 1. Fixed quaternion
    print("\n" + "#"*60)
    print("# CONDITION 1: Fixed Quaternion Algebra")
    print("#"*60)
    m1 = BytePHMTransformer(**cfg, phm_n=4, algebra_mode='quaternion').to(device)
    results['quaternion_fixed'] = train_and_eval(
        'quaternion_fixed', m1, train_loader, val_loader,
        text_val_loader, image_val_loader, device, max_steps)
    del m1; torch.mps.empty_cache()

    # 2. Quaternion-initialized, learnable
    print("\n" + "#"*60)
    print("# CONDITION 2: Quaternion-Initialized (Learnable)")
    print("#"*60)
    m2 = BytePHMTransformer(**cfg, phm_n=4, algebra_mode='quaternion_init').to(device)
    results['quaternion_init'] = train_and_eval(
        'quaternion_init', m2, train_loader, val_loader,
        text_val_loader, image_val_loader, device, max_steps)
    del m2; torch.mps.empty_cache()

    # 3. Fully learned (original PHM, our baseline)
    print("\n" + "#"*60)
    print("# CONDITION 3: Fully Learned PHM (Baseline)")
    print("#"*60)
    m3 = BytePHMTransformer(**cfg, phm_n=4, algebra_mode='learned').to(device)
    results['learned_phm'] = train_and_eval(
        'learned_phm', m3, train_loader, val_loader,
        text_val_loader, image_val_loader, device, max_steps)
    del m3; torch.mps.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("ALGEBRA ABLATION RESULTS")
    print("="*70)
    print(f"{'Condition':<25} {'Params':>10} {'Text BPC':>10} {'Image BPC':>10} {'Combined':>10}")
    print("-"*65)
    for name, r in sorted(results.items(), key=lambda x: x[1]['combined_bpc']):
        print(f"{name:<25} {r['params']:>10,} {r['text_bpc']:>10.3f} {r['image_bpc']:>10.3f} {r['combined_bpc']:>10.3f}")

    # Key question answers
    print("\n" + "="*70)
    print("KEY QUESTIONS")
    print("="*70)

    qf = results['quaternion_fixed']['combined_bpc']
    qi = results['quaternion_init']['combined_bpc']
    lp = results['learned_phm']['combined_bpc']

    if qf < lp - 0.05:
        print("✓ Quaternion structure HELPS: fixed quaternion beats learned PHM")
        print(f"  (quaternion {qf:.3f} vs learned {lp:.3f}, delta={lp-qf:.3f})")
    elif lp < qf - 0.05:
        print("✗ Quaternion structure HURTS: learned PHM beats fixed quaternion")
        print(f"  (learned {lp:.3f} vs quaternion {qf:.3f}, delta={qf-lp:.3f})")
    else:
        print("— Quaternion structure is NEUTRAL: similar to learned PHM")
        print(f"  (quaternion {qf:.3f} vs learned {lp:.3f}, delta={abs(qf-lp):.3f})")

    if qi < min(qf, lp) - 0.05:
        print("✓ Quaternion INIT + learning is BEST: warm-start helps")
    elif qi > max(qf, lp) + 0.05:
        print("✗ Quaternion init hurts when combined with learning")
    else:
        print("— Quaternion init provides similar results to other conditions")

    # Check algebra drift for quaternion_init
    if 'avg_squared_traces' in results.get('quaternion_init', {}):
        traces = results['quaternion_init']['avg_squared_traces']
        quat_ref = [4.0, -4.0, -4.0, -4.0]
        drift = np.linalg.norm(np.array(sorted(traces)) - np.array(sorted(quat_ref)))
        print(f"\n  Quaternion-init algebra drift: {drift:.2f}")
        print(f"  Learned traces: {[f'{t:.1f}' for t in traces]}")
        print(f"  Reference:      {[f'{t:.1f}' for t in quat_ref]}")
        if drift < 2.0:
            print("  → Stayed near quaternion structure!")
        else:
            print("  → Drifted away from quaternion structure")

    Path('results/algebra_ablation').mkdir(parents=True, exist_ok=True)
    with open('results/algebra_ablation/summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results/algebra_ablation/summary.json")


if __name__ == '__main__':
    main()
