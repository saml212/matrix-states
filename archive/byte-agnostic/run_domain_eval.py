#!/usr/bin/env python3
"""
Per-domain evaluation: measure BPC separately for text and images.
Runs on saved checkpoint without interrupting training.
"""

import sys, os, math, json, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import BytePHMTransformer
from src.data import ByteDataset


def eval_domain(model, data_path, device, seq_len=1024, n_samples=500):
    """Evaluate model on a single domain."""
    dataset = ByteDataset(data_path, seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True)

    model.eval()
    total_loss = 0
    n = 0
    seg_lengths = []

    with torch.no_grad():
        for x, y in loader:
            if n >= n_samples:
                break
            x, y = x.to(device), y.to(device)
            logits, bp, sc = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            total_loss += loss.item()

            # Collect segment length stats
            lengths = sc[sc > 0]
            seg_lengths.extend(lengths.cpu().tolist())
            n += 1

    bpc = total_loss / n / math.log(2)
    return {
        'bpc': bpc,
        'loss': total_loss / n,
        'n_batches': n,
        'seg_length_mean': np.mean(seg_lengths),
        'seg_length_std': np.std(seg_lengths),
        'seg_length_median': np.median(seg_lengths),
        'seg_length_p10': np.percentile(seg_lengths, 10),
        'seg_length_p90': np.percentile(seg_lengths, 90),
    }


def main():
    device = 'mps'
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else '/Volumes/1TB_SSD/learned-representations/checkpoints/phase2/best.pt'

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    model = BytePHMTransformer(
        d_model=config.get('d_model', 640),
        n_heads=config.get('n_heads', 10),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 2560),
        phm_n=config.get('phm_n', 4),
        dropout=0.0,  # no dropout for eval
        max_len=config.get('max_len', 4096),
        seg_layers=config.get('seg_layers', 1),
        seg_d_model=config.get('seg_d_model', 320),
        target_compression=config.get('target_compression', 4.0),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    step = ckpt.get('step', '?')
    print(f"Model from step {step} | Params: {model.count_parameters():,}")

    text_path = '/Volumes/1TB_SSD/learned-representations/data/text.bin'
    image_path = '/Volumes/1TB_SSD/learned-representations/data/images.bin'

    results = {}
    for domain, path in [('text', text_path), ('images', image_path)]:
        if os.path.exists(path):
            print(f"\nEvaluating {domain}...")
            r = eval_domain(model, path, device, seq_len=1024, n_samples=200)
            results[domain] = r
            print(f"  BPC: {r['bpc']:.3f}")
            print(f"  Seg length: μ={r['seg_length_mean']:.1f} σ={r['seg_length_std']:.1f} "
                  f"median={r['seg_length_median']:.1f}")
            print(f"  Seg length range: p10={r['seg_length_p10']:.0f} p90={r['seg_length_p90']:.0f}")

    # PHM algebra analysis
    print("\n" + "="*60)
    print("PHM ALGEBRA ANALYSIS")
    print("="*60)
    from src.phm import PHMLinear
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, PHMLinear):
            A = module.get_algebra()  # (n, n, n)
            # Commutativity
            comm_scores = []
            for i in range(A.shape[0]):
                for j in range(i+1, A.shape[0]):
                    comm = torch.norm(A[i] @ A[j] - A[j] @ A[i]).item()
                    comm_scores.append(comm)
            avg_comm = np.mean(comm_scores)

            # Frobenius norm of each factor (MPS-compatible)
            norms = [torch.norm(A[i]).item() for i in range(A.shape[0])]

            if layer_idx < 5 or layer_idx % 10 == 0:  # print first 5 and every 10th
                print(f"  Layer {layer_idx} ({name}): "
                      f"commutativity={avg_comm:.4f} "
                      f"norms={[f'{n:.2f}' for n in norms]}")
            layer_idx += 1

    print(f"\n  Total PHM layers: {layer_idx}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if 'text' in results and 'images' in results:
        print(f"  Text BPC:  {results['text']['bpc']:.3f}")
        print(f"  Image BPC: {results['images']['bpc']:.3f}")
        print(f"  Text seg σ:  {results['text']['seg_length_std']:.2f}")
        print(f"  Image seg σ: {results['images']['seg_length_std']:.2f}")
        print(f"  Domain differentiation (σ gap): "
              f"{abs(results['images']['seg_length_std'] - results['text']['seg_length_std']):.2f}")

    out_path = 'results/phase2/domain_eval.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'step': step, 'results': results}, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
