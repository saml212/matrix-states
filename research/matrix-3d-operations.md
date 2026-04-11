# 3D Operations: Coupling Sequence with Matrix Structure

## The Core Answer
Matrix multiplication along the sequence dimension IS the 3D operation.

`(M_s @ M_t)[i,j] = Σ_k M_s[i,k] * M_t[k,j]`

Row i of M_s couples with column j of M_t through contraction index k.
This is fundamentally different from Frobenius inner product (scalar per pair).

## Three Concrete Operations

### 1. Pairwise Matrix Product Attention
```python
scores = torch.einsum('bsik,btkj->bstij', Q, K)  # (B,L,L,d,d) matrix per pair
scalar = torch.einsum('bstii->bst', scores) / d    # trace for weighting
weights = softmax(scalar, dim=-1)
out = torch.einsum('bst,bstij->bsij', weights, scores)  # structured aggregation
```
The score matrix S[s,t] both determines the weight AND transforms the value.

### 2. Sequential Matrix Product Scan (Transfer Matrix)
```python
S_t = S_{t-1} @ T_t    # cumulative product
# T_t is derived from the token matrix at position t
# Parallelizable via associative scan in O(log L) depth
```
Each position's matrix transforms the running state. Position t's internal
structure directly determines how information from t-1 flows to t+1.

### 3. Row-Column Cross-Attention
```python
# Row vectors of M_s attend to column vectors of M_t
Q = M @ W_q     # preserves row structure
K = W_k @ M     # preserves column structure
scores = torch.einsum('bsik,btkj->bstij', Q, K)  # (d,d) score per pair
# Each (row_i, col_j) combination has its own attention weight
```

## Which to use in our architecture

For the autoregressive matrix thinker, option 1 (pairwise matrix product attention)
is the most natural replacement for standard Frobenius attention. It gives each
position pair a STRUCTURED interaction instead of a scalar weight.

The key change: instead of `einsum('bhlij,bhmij->bhlm')` (Frobenius, scalar score),
use `einsum('bhlik,bhmkj->bhlmij')` (matrix product, structured score).

This IS the 3D operation. Rows of Q at position l couple with columns of K
at position m through the shared k dimension.
