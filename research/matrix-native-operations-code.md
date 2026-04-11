# Matrix-Native Operations — Working Code and Repo Findings

## What Each Major Repo Does (NO FLATTENING)

**GATr (Qualcomm):** 16-dim multivectors. Projection = weighted sum of basis
sandwich products: `output = Σ_k w_k * L_k @ x @ R_k`. Attention = Frobenius
inner product. Never flattens.

**TTT (Stanford):** Weight-matrix hidden state. Read = `W @ q` (left multiply).
Write = outer product gradient step. Never flattens.

**DeltaProduct (AutoML):** Householder products: `H_k @ M = M - β * v @ (vᵀ @ M)`.
Each reflection costs d+1 params. Never forms the full matrix. Never flattens.

**CliffordNet:** Geometric product IS the bilinear map. Replaces both linear
layers AND FFNs with one algebraic operation. Never flattens.

## Parameter Efficiency Summary

| Operation | Params | FLOPs/tok | vs Flatten |
|-----------|--------|-----------|------------|
| Householder (K=4) | 68 | 2,112 | 964x fewer params |
| Kronecker (K=4, n=4) | 128 | varies | 512x fewer params |
| Bilinear (A@M@B) | 512 | 8,192 | 128x fewer params |
| RowThenCol (gelu(A@M)@B) | 512 | 8,448 | 128x fewer params |
| MultiHead bilinear | 512 | 8,192 | 128x fewer params |
| **Flatten→Linear (current)** | **65,536** | **65,536** | **baseline** |

## Key einsum patterns (copy-paste ready)

```python
# Sandwich: A @ M @ B
torch.einsum("ij, bsjk, kl -> bsil", A, M, B)

# Multi-head sandwich
torch.einsum("hij, bsjk, hkl -> bhsil", As, M, Bs)

# Frobenius attention score (no flatten)
torch.einsum("bhqij, bhkij -> bhqk", Q, K)

# Attention value aggregation
torch.einsum("bhqk, bhkij -> bhqij", attn_weights, V)

# Householder: v^T @ M then outer product
vtM = torch.einsum("i, bsij -> bsj", v, M)
update = torch.einsum("i, bsj -> bsij", v, vtM)
```

## Full runnable code in agent output — includes:
- BilinearProjection
- BilinearWithTranspose
- RowThenColumnProjection (gelu between left and right multiply)
- MultiHeadBilinear
- KroneckerBilinear
- HouseholderProduct
- MatrixAttention (full multi-head, no flatten)
- MatrixTransformerBlock (complete pre-norm block)
- TTTStyleMatrixUpdate
- Einsum reference card
