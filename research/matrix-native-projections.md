# Matrix-Native Projections — Research Results

## The Answer: Sum-of-Kroneckers (K=4)

Replace flatten→Linear→reshape with: `output = Σ_k A_k @ M @ B_k`

This keeps data as a matrix throughout. Each term A_k @ M @ B_k mixes
rows (via A_k) and columns (via B_k) independently. Multiple terms (K>1)
allow cross-interactions between row-mixing and column-mixing.

## Parameter and FLOP comparison (d=16)

| Method | Params | FLOPs | vs Linear |
|--------|--------|-------|-----------|
| Flatten→Linear | 65,536 | 131,072 | 1.0x |
| Bilinear (K=1) | 512 | 16,384 | 128x fewer params |
| **Kronecker K=4** | **2,048** | **65,536** | **32x fewer params** |
| Kronecker K=8 | 4,096 | 131,072 | 16x fewer params |

K=4 is the sweet spot: 32x fewer params, half the FLOPs.

## Key code patterns

Projection: `torch.matmul(torch.matmul(A, M), B)` — never flattens
Attention scores: `torch.einsum('bhlij,bhmij->bhlm', Q, K)` — Frobenius inner product
SwiGLU in matrix space: `gate * value` where both are d×d — Hadamard product is valid
Output head: `torch.einsum('cij,blij->blc', output_matrices, M)` — direct, no flatten

## The full model at K=4 would use ~200K backbone params vs current ~2.2M
That's 10x fewer parameters — could add more layers/heads instead.

## Working code provided by research agent — see full agent output for
copy-pasteable MatrixNativeAttention and MatrixNativeMultiplicativeLayer classes.
