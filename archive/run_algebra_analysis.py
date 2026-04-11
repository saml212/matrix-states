#!/usr/bin/env python3
"""Deep PHM algebra analysis on trained model checkpoint."""

import sys, os, torch, json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import BytePHMTransformer
from src.phm import PHMLinear


def quaternion_basis():
    A0 = torch.eye(4)
    A1 = torch.tensor([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,-1.],[0.,0.,1.,0.]])
    A2 = torch.tensor([[0.,0.,-1.,0.],[0.,0.,0.,1.],[1.,0.,0.,0.],[0.,-1.,0.,0.]])
    A3 = torch.tensor([[0.,0.,0.,-1.],[0.,0.,-1.,0.],[0.,1.,0.,0.],[1.,0.,0.,0.]])
    return torch.stack([A0, A1, A2, A3])


def extract_structure_constants(A):
    """Extract structure constants c_{ijk} where A_i @ A_j = sum_k c_{ijk} A_k."""
    n = A.shape[0]
    basis_flat = A.reshape(n, -1)  # (n, n^2)
    c = torch.zeros(n, n, n)
    residuals = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            product_flat = (A[i] @ A[j]).reshape(-1)
            result = torch.linalg.lstsq(basis_flat.T, product_flat)
            c[i, j] = result.solution
            reconstruction = basis_flat.T @ result.solution
            residuals[i, j] = (product_flat - reconstruction).norm() / (product_flat.norm() + 1e-8)
    return c, residuals


def compute_invariants(A):
    """Compute basis-invariant algebraic properties."""
    n = A.shape[0]
    eye = torch.eye(n)
    inv = {}

    # Structure constants
    c, residuals = extract_structure_constants(A)
    inv['closure_residual'] = residuals.mean().item()

    # Commutativity
    commutator = c - c.permute(1, 0, 2)
    inv['commutativity'] = commutator.norm().item() / (c.norm().item() + 1e-8)

    # Associativity
    assoc = 0.0; count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                lhs = (A[i] @ A[j]) @ A[k]
                rhs = A[i] @ (A[j] @ A[k])
                assoc += (lhs - rhs).norm().item()
                count += 1
    inv['associativity_violation'] = assoc / count

    # Squared traces (most discriminative)
    inv['squared_traces'] = [(A[i] @ A[i]).trace().item() for i in range(n)]

    # Identity detection
    identity_dists = [(A[i] / (A[i].norm() + 1e-8) - eye / eye.norm()).norm().item() for i in range(n)]
    inv['identity_index'] = int(np.argmin(identity_dists))
    inv['identity_distance'] = min(identity_dists)

    # Nilpotent elements (A_i^2 ≈ 0)
    inv['nilpotent_scores'] = [(A[i] @ A[i]).norm().item() / (A[i].norm().item()**2 + 1e-8) for i in range(n)]

    # Determinants
    inv['determinants'] = [torch.det(A[i]).item() for i in range(n)]

    # Killing form rank
    killing = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            trace_sum = 0.0
            for k_idx in range(n):
                ad_i_k = A[i] @ A[k_idx] - A[k_idx] @ A[i]
                ad_j_k = A[j] @ A[k_idx] - A[k_idx] @ A[j]
                trace_sum += (ad_i_k * ad_j_k).sum().item()
            killing[i, j] = trace_sum
    inv['killing_rank'] = int(torch.linalg.matrix_rank(killing).item())

    return inv


def classify_algebra(A):
    """Classify learned algebra by comparing invariants to known algebras."""
    inv = compute_invariants(A)

    # Compare squared trace signatures
    learned_traces = sorted(inv['squared_traces'])

    known = {
        'quaternion': sorted([4, -4, -4, -4]),
        'split_quaternion': sorted([4, -4, 4, 4]),
        'tessarine': sorted([4, -4, -4, 4]),
        'dual_numbers': sorted([4, 0, 0, 0]),
    }

    distances = {}
    for name, ref_traces in known.items():
        trace_dist = np.linalg.norm(np.array(learned_traces) - np.array(ref_traces))
        distances[name] = trace_dist

    best = min(distances, key=distances.get)
    return best, distances, inv


def main():
    device = 'cpu'  # Use CPU for algebra analysis (MPS lacks some linalg ops)
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else '/Volumes/1TB_SSD/learned-representations/checkpoints/phase2/best.pt'

    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']

    model = BytePHMTransformer(
        d_model=config.get('d_model', 640), n_heads=config.get('n_heads', 10),
        n_layers=config.get('n_layers', 6), d_ff=config.get('d_ff', 2560),
        phm_n=config.get('phm_n', 4), dropout=0.0,
        max_len=config.get('max_len', 4096), seg_layers=config.get('seg_layers', 1),
        seg_d_model=config.get('seg_d_model', 320),
        target_compression=config.get('target_compression', 4.0),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    step = ckpt.get('step', '?')
    print(f"Step {step} | {model.count_parameters():,} params\n")

    # Analyze each PHM layer
    results = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, PHMLinear):
            A = module.get_algebra()  # (4, 4, 4)
            best, distances, inv = classify_algebra(A)

            layer_type = 'attn' if 'attn' in name else 'mlp'
            proj_type = name.split('.')[-1] if 'attn' in name else 'fc'

            results[f'layer_{layer_idx}'] = {
                'name': name,
                'type': layer_type,
                'proj': proj_type,
                'best_match': best,
                'distances': distances,
                'invariants': {k: v for k, v in inv.items() if k != 'killing_form'},
            }

            if layer_idx < 10 or layer_idx % 6 == 0:
                print(f"Layer {layer_idx:2d} ({name})")
                print(f"  Best match: {best} (dist={distances[best]:.2f})")
                print(f"  tr(A_i^2): {[f'{t:.1f}' for t in inv['squared_traces']]}")
                print(f"  Commutativity: {inv['commutativity']:.4f}")
                print(f"  Associativity: {inv['associativity_violation']:.6f}")
                print(f"  Closure: {inv['closure_residual']:.4f}")
                print(f"  Killing rank: {inv['killing_rank']}")
                print(f"  Dets: {[f'{d:.3f}' for d in inv['determinants']]}")
                print()
            layer_idx += 1

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)

    # Count algebra types
    from collections import Counter
    type_counts = Counter(r['best_match'] for r in results.values())
    print(f"\nAlgebra type distribution across {layer_idx} PHM layers:")
    for alg, count in type_counts.most_common():
        print(f"  {alg}: {count} layers ({100*count/layer_idx:.0f}%)")

    # Attention vs MLP
    print(f"\nBy module type:")
    for mod_type in ['attn', 'mlp']:
        layers = [r for r in results.values() if r['type'] == mod_type]
        if layers:
            avg_comm = np.mean([r['invariants']['commutativity'] for r in layers])
            types = Counter(r['best_match'] for r in layers)
            print(f"  {mod_type}: avg_comm={avg_comm:.4f}, types={dict(types)}")

    # By attention projection type
    print(f"\nBy attention projection:")
    for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        layers = [r for r in results.values() if r['proj'] == proj]
        if layers:
            avg_comm = np.mean([r['invariants']['commutativity'] for r in layers])
            avg_closure = np.mean([r['invariants']['closure_residual'] for r in layers])
            print(f"  {proj}: avg_comm={avg_comm:.4f}, avg_closure={avg_closure:.4f}")

    # Save
    out_path = 'results/phase2/algebra_analysis.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump({'step': step, 'results': results}, f, indent=2, default=convert)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
