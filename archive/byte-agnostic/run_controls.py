#!/usr/bin/env python3
"""
Control experiments: single-domain models for cross-domain transfer measurement.

Train text-only and image-only models with the SAME architecture and compute
budget as the multi-modal Phase 2 model. Compare per-domain BPC to measure
whether multi-modal training helps (positive transfer) or hurts (interference).
"""

import sys, os, time, json, math, yaml, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import BytePHMTransformer
from src.data import ByteDataset


def train_single_domain(name, data_path, config, device='mps'):
    """Train a single-domain model and return results."""
    results_dir = Path(f'results/controls/{name}')
    results_dir.mkdir(parents=True, exist_ok=True)

    seq_len = config.get('seq_len', 1024)
    batch_size = config.get('batch_size', 16)

    dataset = ByteDataset(data_path, seq_len)
    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

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

    n_params = model.count_parameters()
    n_epochs = config.get('control_epochs', 3)  # Fewer epochs for controls
    total_steps = n_epochs * len(train_loader)
    lr = config.get('lr', 3e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warmup = min(300, total_steps // 10)

    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*50}")
    print(f"CONTROL: {name}")
    print(f"Params: {n_params:,} | Data: {os.path.getsize(data_path)/1e6:.0f}MB")
    print(f"Train: {n_train:,} | Steps/epoch: {len(train_loader):,} | Epochs: {n_epochs}")
    print(f"{'='*50}")

    tau_start = config.get('tau_start', 1.5)
    tau_end = config.get('tau_end', 0.2)
    best_val = float('inf')
    global_step = 0
    start = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            progress = global_step / max(total_steps, 1)
            model.segmenter.set_tau(tau_start * (tau_end / tau_start) ** progress)

            optimizer.zero_grad()
            logits, bp, sc = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            var_loss = model.segmenter.boundary_variance_loss(sc)
            (loss + 0.01 * var_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_n += 1
            global_step += 1

            if global_step % 100 == 0:
                bpc = epoch_loss / epoch_n / math.log(2)
                print(f"  [{time.time()-start:5.0f}s] Step {global_step:5d} | BPC {bpc:.3f}")

        # Eval
        model.eval()
        val_loss = 0; val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                if val_n >= 100: break
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x)
                val_loss += F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
                val_n += 1
        val_bpc = val_loss / val_n / math.log(2)
        best_val = min(best_val, val_bpc)
        train_bpc = epoch_loss / epoch_n / math.log(2)
        print(f"  Epoch {epoch+1}: Train BPC {train_bpc:.3f} | Val BPC {val_bpc:.3f}")

    elapsed = time.time() - start
    print(f"  {name} done in {elapsed/60:.1f}min | Best Val BPC: {best_val:.3f}")

    result = {
        'name': name, 'params': n_params, 'best_val_bpc': best_val,
        'final_train_bpc': train_bpc, 'elapsed_s': elapsed, 'steps': global_step,
    }

    torch.save({
        'model_state_dict': model.state_dict(),
        'step': global_step, 'config': config,
    }, results_dir / 'model.pt')

    with open(results_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/phase2.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['control_epochs'] = 3  # Quick controls

    text_path = config['text_path']
    image_path = config['image_path']

    results = {}

    # Text-only control
    results['text_only'] = train_single_domain('text_only', text_path, config)

    # Image-only control
    results['image_only'] = train_single_domain('image_only', image_path, config)

    # Compare with multi-modal
    print("\n" + "="*60)
    print("CROSS-DOMAIN TRANSFER ANALYSIS")
    print("="*60)
    print(f"{'Model':<20} {'Text BPC':>10} {'Image BPC':>10}")
    print("-"*40)
    print(f"{'Text-only':<20} {results['text_only']['best_val_bpc']:>10.3f} {'N/A':>10}")
    print(f"{'Image-only':<20} {'N/A':>10} {results['image_only']['best_val_bpc']:>10.3f}")
    print(f"\nCompare with Phase 2 multi-modal per-domain eval:")
    print(f"  Multi-modal text BPC: see results/phase2/domain_eval.json")
    print(f"  Multi-modal image BPC: see results/phase2/domain_eval.json")
    print(f"\n  Positive transfer = multi-modal BPC < single-domain BPC")

    with open('results/controls/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/controls/summary.json")


if __name__ == '__main__':
    main()
