# Matrix-Valued Hidden States: The 2025-2026 Landscape

## The field is converging: matrix states beat vector states

| Architecture | State Type | Result |
|-------------|-----------|--------|
| Mamba-1 → Mamba-2 | diagonal → full matrix | Matches Transformer at 2x speed |
| RWKV-4 → RWKV-5/6 | vector → matrix | Significant quality gain |
| TTT Layers (Stanford) | Weight matrices AS states | Outperforms Mamba + Transformer |
| Titans (Google, NeurIPS 2025) | Matrix memory, surprise-update | 2M+ context |
| GLA/DeltaNet/Based | Matrix state linear attention | Matches softmax quality |

## Most important for us:

### TTT Layers (arXiv: 2407.04620)
Hidden state IS a weight matrix. Updated via gradient descent at each token.
TTT-MLP outperforms both Mamba and Transformers at matched compute.

### Titans (arXiv: 2501.00663, NeurIPS 2025)
Matrix memory updated by SURPRISE — writes more aggressively on unexpected input.
Parallels our segmenter: both detect "interesting" points in the byte stream.

### COCONUT (Meta, arXiv: 2412.06769)
Chain of CONTINUOUS thought — thinks in latent space, not token space.
Iterative refinement without decoding. Outperforms chain-of-thought.

## The gap nobody has filled:
All matrix-valued states are UNSTRUCTURED. Nobody uses algebraically
structured matrix states (Kronecker/Clifford). Our experiment is the
first data point on what happens when you try (→ nilpotent collapse).
