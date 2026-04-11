"""
Analysis tools for trained BytePHMTransformer.

- Boundary visualization across domains
- PHM algebra analysis (Kronecker factor structure)
- Training curve plotting
"""

import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .model import BytePHMTransformer
from .phm import PHMLinear


def load_model(checkpoint_path: str, device: str = 'mps'):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
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
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def visualize_boundaries(model, text: str, output_path: str = None, device: str = 'mps'):
    """Visualize boundary predictions on a text string."""
    model.eval()
    byte_ids = torch.tensor(
        [list(text.encode('utf-8'))],
        dtype=torch.long, device=device
    )

    with torch.no_grad():
        _, boundary_probs, _, _ = model.segmenter(byte_ids, hard=True)

    probs = boundary_probs[0].cpu().numpy()
    chars = list(text.encode('utf-8'))

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), gridspec_kw={'height_ratios': [1, 3]})

    # Top: text with boundary markers
    ax = axes[0]
    ax.set_xlim(0, len(chars))
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Show characters with color coding
    for i, (byte_val, p) in enumerate(zip(chars, probs)):
        try:
            c = chr(byte_val) if 32 <= byte_val < 127 else '·'
        except:
            c = '·'
        color = 'red' if p > 0.5 else 'black'
        fontweight = 'bold' if p > 0.5 else 'normal'
        ax.text(i + 0.5, 0.5, c, ha='center', va='center',
                fontsize=8, color=color, fontweight=fontweight,
                fontfamily='monospace')
        if p > 0.5 and i > 0:
            ax.axvline(x=i, color='red', alpha=0.3, linewidth=1)

    ax.set_title('Learned Boundaries (red = segment start)')

    # Bottom: boundary probability heatmap
    ax = axes[1]
    ax.bar(range(len(probs)), probs, width=1.0, color='steelblue', alpha=0.7)
    ax.set_xlabel('Byte position')
    ax.set_ylabel('Boundary probability')
    ax.set_xlim(0, len(chars))
    ax.set_ylim(0, 1)

    # Mark word boundaries for reference
    for i, byte_val in enumerate(chars):
        if byte_val == ord(' ') or byte_val == ord('\n'):
            ax.axvline(x=i, color='green', alpha=0.2, linewidth=0.5, linestyle='--')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved boundary visualization to {output_path}")
    plt.close()
    return probs


def analyze_boundaries_by_domain(model, data_paths: dict, seq_len: int = 512,
                                  n_samples: int = 100, output_dir: str = None,
                                  device: str = 'mps'):
    """Analyze boundary patterns across domains."""
    model.eval()
    results = {}

    for domain, path in data_paths.items():
        data = np.memmap(path, dtype=np.uint8, mode='r')
        seg_lengths = []
        boundary_rates = []

        for i in range(n_samples):
            start = i * seq_len
            if start + seq_len >= len(data):
                break
            chunk = data[start:start + seq_len]
            x = torch.tensor([chunk.tolist()], dtype=torch.long, device=device)

            with torch.no_grad():
                _, boundary_probs, _, n_segs = model.segmenter(x, hard=True)

            probs = boundary_probs[0].cpu().numpy()
            boundaries = np.where(probs > 0.5)[0]
            boundary_rates.append(probs.mean())

            if len(boundaries) > 1:
                lengths = np.diff(boundaries)
                seg_lengths.extend(lengths.tolist())

        results[domain] = {
            'mean_seg_length': np.mean(seg_lengths) if seg_lengths else 0,
            'std_seg_length': np.std(seg_lengths) if seg_lengths else 0,
            'median_seg_length': np.median(seg_lengths) if seg_lengths else 0,
            'mean_boundary_rate': np.mean(boundary_rates),
            'seg_lengths': seg_lengths,
        }

        print(f"\n{domain}:")
        print(f"  Mean segment length: {results[domain]['mean_seg_length']:.1f} bytes")
        print(f"  Std segment length: {results[domain]['std_seg_length']:.1f}")
        print(f"  Median segment length: {results[domain]['median_seg_length']:.1f}")
        print(f"  Boundary rate: {results[domain]['mean_boundary_rate']:.3f}")

    # Plot comparison
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        if len(results) == 1:
            axes = [axes]

        for ax, (domain, res) in zip(axes, results.items()):
            if res['seg_lengths']:
                ax.hist(res['seg_lengths'], bins=50, alpha=0.7, color='steelblue')
            ax.set_title(f'{domain}\nμ={res["mean_seg_length"]:.1f}, σ={res["std_seg_length"]:.1f}')
            ax.set_xlabel('Segment length (bytes)')
            ax.set_ylabel('Count')

        plt.suptitle('Segment Length Distributions by Domain')
        plt.tight_layout()
        plt.savefig(output_dir / 'segment_lengths.png', dpi=150, bbox_inches='tight')
        plt.close()

    return results


def analyze_phm_algebra(model, output_dir: str = None):
    """
    Extract and analyze learned Kronecker factors (the "algebra").

    For each PHM layer, extracts the A matrices and computes:
    - Effective multiplication table
    - Frobenius distance to known algebras (quaternions, etc.)
    - Singular value decomposition of A factors
    """
    results = {}

    # Known algebra references
    # Quaternion: i²=j²=k²=ijk=-1
    quaternion_A = torch.zeros(4, 4, 4)
    # Identity
    quaternion_A[0] = torch.eye(4)
    # i
    quaternion_A[1, 0, 1] = 1; quaternion_A[1, 1, 0] = -1
    quaternion_A[1, 2, 3] = 1; quaternion_A[1, 3, 2] = -1
    # j
    quaternion_A[2, 0, 2] = 1; quaternion_A[2, 2, 0] = -1
    quaternion_A[2, 1, 3] = -1; quaternion_A[2, 3, 1] = 1
    # k
    quaternion_A[3, 0, 3] = 1; quaternion_A[3, 3, 0] = -1
    quaternion_A[3, 1, 2] = 1; quaternion_A[3, 2, 1] = -1

    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, PHMLinear):
            A = module.get_algebra()  # (n, n, n)
            n = A.shape[0]

            # SVD of each A factor
            svs = []
            for i in range(n):
                _, s, _ = torch.svd(A[i])
                svs.append(s.numpy())

            # Distance to quaternion algebra (if n=4)
            quat_dist = None
            if n == 4:
                quat_dist = torch.norm(A.cpu() - quaternion_A).item()

            # Commutativity measure: ||A[i]@A[j] - A[j]@A[i]|| for all i,j
            comm_scores = []
            for i in range(n):
                for j in range(i + 1, n):
                    comm = torch.norm(A[i] @ A[j] - A[j] @ A[i]).item()
                    comm_scores.append(comm)
            avg_comm = np.mean(comm_scores) if comm_scores else 0

            results[f'layer_{layer_idx}_{name}'] = {
                'singular_values': svs,
                'quaternion_distance': quat_dist,
                'avg_commutativity': avg_comm,
                'A_matrices': A.numpy(),
            }
            layer_idx += 1

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot singular values across layers
        fig, ax = plt.subplots(figsize=(12, 4))
        for lname, res in results.items():
            for i, sv in enumerate(res['singular_values']):
                ax.plot(sv, 'o-', alpha=0.5, markersize=3,
                        label=f'{lname} factor {i}' if i == 0 else None)

        ax.set_xlabel('Singular value index')
        ax.set_ylabel('Singular value')
        ax.set_title('PHM Algebra: Singular Values of Learned A Matrices')
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'phm_algebra_svd.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Plot quaternion distances
        if any(r['quaternion_distance'] is not None for r in results.values()):
            fig, ax = plt.subplots(figsize=(10, 4))
            names = [k for k, v in results.items() if v['quaternion_distance'] is not None]
            dists = [v['quaternion_distance'] for v in results.values() if v['quaternion_distance'] is not None]
            ax.bar(range(len(names)), dists, color='steelblue')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
            ax.set_ylabel('Frobenius distance to quaternion algebra')
            ax.set_title('How Close Are Learned Algebras to Quaternions?')
            plt.tight_layout()
            plt.savefig(output_dir / 'phm_quaternion_distance.png', dpi=150, bbox_inches='tight')
            plt.close()

    return results


def plot_training_curves(history_path: str, output_dir: str = None):
    """Plot training curves from saved history."""
    with open(history_path) as f:
        history = json.load(f)

    output_dir = Path(output_dir) if output_dir else Path(history_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    if history.get('train_pred_loss'):
        ax.plot(history['step'][:len(history['train_pred_loss'])],
                history['train_pred_loss'], label='Train', alpha=0.7)
    if history.get('val_pred_loss'):
        val_steps = history['step'][::len(history['step']) // max(len(history['val_pred_loss']), 1)]
        ax.plot(val_steps[:len(history['val_pred_loss'])],
                history['val_pred_loss'], label='Val', alpha=0.9, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Prediction Loss (CE)')
    ax.set_title('Next-Byte Prediction Loss')
    ax.legend()

    # BPC
    ax = axes[0, 1]
    if history.get('train_pred_loss'):
        bpc = [l / math.log(2) for l in history['train_pred_loss']]
        ax.plot(history['step'][:len(bpc)], bpc, label='Train BPC', alpha=0.7)
    if history.get('val_bpc'):
        val_steps = history['step'][::len(history['step']) // max(len(history['val_bpc']), 1)]
        ax.plot(val_steps[:len(history['val_bpc'])],
                history['val_bpc'], label='Val BPC', alpha=0.9, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Bits per Character')
    ax.set_title('Bits per Character')
    ax.legend()

    # Boundary rate
    ax = axes[1, 0]
    if history.get('boundary_rate'):
        ax.plot(history['step'][:len(history['boundary_rate'])],
                history['boundary_rate'], color='green', alpha=0.7)
        ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Target rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('Boundary Rate')
    ax.set_title('Segmenter Boundary Rate')
    ax.legend()

    # LR and Tau
    ax = axes[1, 1]
    if history.get('lr'):
        ax2 = ax.twinx()
        ax.plot(history['step'][:len(history['lr'])],
                history['lr'], color='blue', alpha=0.7, label='LR')
        ax2.plot(history['step'][:len(history['tau'])],
                 history['tau'], color='orange', alpha=0.7, label='τ (Gumbel)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate', color='blue')
        ax2.set_ylabel('Gumbel τ', color='orange')
        ax.set_title('Learning Rate & Gumbel Temperature')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)

    plt.suptitle('Training Progress', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")
    plt.close()
