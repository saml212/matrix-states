# Brainstorm: Cheap Matrix Operations for Matrix-Valued Token Representations

**Date:** 2026-03-26
**Context:** Matrix d=16 RowThenCol projections (`silu(A @ M) @ B`) are 130x more parameter-efficient per layer but ~10-30x slower per FLOP than standard vector operations. At matched compute budgets, LoopFormer (standard vectors + looping) beats us 2x on BPB. This document brainstorms every possible way to close that gap.

**The bottleneck:** 10 RowThenCol projections per layer (6 in MultiplicativeThinkingLayer + 4 in attention), each doing two d x d matmuls per position. At d=16, each projection is 2 x 16^3 = 8,192 FLOPs. With 10 projections x 12-48 layers x 1-8 iterations, this dominates the compute budget.

---

## Idea 1: Fused RowThenCol Triton Kernel

**What:** Write a single Triton kernel that computes `silu(A @ M) @ B` in one pass without materializing the intermediate `silu(A @ M)`. Fuse the left matmul, activation, and right matmul into a single tiled computation. Also fuse across all 10 projections in a layer so you do one kernel launch instead of 10.

**Speedup estimate:** 2-4x wall-clock on GPU. The current implementation does 3 separate operations (einsum, silu, einsum) with 3 memory round-trips through HBM. A fused kernel eliminates 2 of those round-trips. At d=16 the matrices are tiny (256 elements), so the bottleneck is memory bandwidth, not arithmetic — fusion should help enormously.

**Quality tradeoff:** None. Identical computation.

**Feasibility:** Medium. Triton can handle this. The 16x16 tile size is awkward for standard GEMM tiling (typically 64 or 128), but you could pack multiple positions into a single tile. The key challenge is that each position's A and B are shared (same projection) while M varies per position — this is a batched matmul pattern that Triton handles well.

**Implementation sketch:**
```python
@triton.jit
def fused_rowthencolumn(M_ptr, A_ptr, B_ptr, out_ptr, B_size, L, d):
    # Each program instance handles one (batch, seq_pos) pair
    pid = tl.program_id(0)
    # Load A (shared), M[pid] (per-position), B (shared)
    # Compute AM = A @ M, apply silu in registers, compute (silu(AM)) @ B
    # Store to out[pid]
```

---

## Idea 2: Diagonal + Low-Rank (DLR) Projections

**What:** Replace the full d x d matrices A and B with `A = diag(a) + U_A @ V_A^T` where a is d-dimensional, U_A is (d, r), V_A is (d, r). The diagonal captures per-dimension scaling (cheap) and the low-rank term captures cross-dimension interactions (expressive).

**Cost analysis:** Full A @ M costs d^3 = 4,096 FLOPs. DLR costs: diagonal part d^2 = 256, low-rank part 2 * d^2 * r FLOPs. At r=2: 256 + 1,024 = 1,280 FLOPs. That is a 3.2x reduction per matmul, 3.2x per RowThenCol, ~3x overall.

**Params:** Full A has d^2 = 256 params. DLR at r=2 has d + 2*d*r = 16 + 64 = 80 params. 3.2x reduction.

**Speedup estimate:** 3x FLOPs reduction. Wall-clock improvement depends on whether the diagonal and low-rank can be fused. With a Triton kernel: ~2-3x wall-clock.

**Quality tradeoff:** Low-rank r=2 means A can only mix 2 directions across rows. At d=16, this is quite restrictive. However, the diagonal provides per-row independent scaling which preserves row independence. The key question: does RowThenCol need full d x d mixing, or is "mostly diagonal with a bit of cross-talk" sufficient? The answer likely depends on how much row-column independence the model actually exploits.

**Risk:** You tested "identity + low-rank" and got "minimal speedup." But that was I + low-rank (no diagonal learning). Diagonal + low-rank is strictly more expressive because the diagonal can learn non-trivial per-dimension scales. Worth retesting.

---

## Idea 3: Block-Diagonal Projections

**What:** Replace A and B with block-diagonal matrices. At d=16 with block size b=4, you get 4 blocks of 4x4. Each block operates on 4 rows (or columns) independently.

**Cost:** Full d^3 = 4,096 FLOPs. Block-diagonal: 4 * 4^3 = 256 FLOPs. That is a 16x reduction per matmul.

**Params:** 4 * 16 = 64 params vs 256. 4x reduction.

**Speedup estimate:** 16x FLOPs per matmul, ~8x per RowThenCol (two matmuls), ~6x overall accounting for other costs.

**Quality tradeoff:** HIGH. Block-diagonal means rows 0-3 never interact with rows 4-7, etc. The entire point of RowThenCol is mixing across all rows. Block-diagonal with blocks of size 4 means only local mixing within groups of 4 rows.

**Mitigation:** Alternate between two different block partitions across layers (like shuffle in ShuffleNet). Even layers use blocks {0-3, 4-7, 8-11, 12-15}, odd layers use blocks {0,4,8,12}, {1,5,9,13}, etc. This gives global mixing over 2 layers while keeping each layer cheap.

**Assessment:** The shuffled variant is promising. 8x cheaper per layer, global mixing over 2 layers. The quality hit from delayed global mixing could be significant for rank dynamics though — the multiplicative composition (I+delta)*M*(I+gamma) needs delta and gamma to have full rank to build rank effectively. Block-diagonal delta/gamma have rank at most 4 per block. Need to test.

---

## Idea 4: Monarch Factorization of A and B

**What:** Factor A = P_1 @ D_1 @ P_2 @ D_2 where P_i are fixed permutation matrices and D_i are block-diagonal. This is the Monarch matrix structure (Dao, ICML 2022). At d=16 with sqrt(d)=4 blocks of size 4:

**Cost:** Two block-diagonal matmuls + two permutations. 2 * (4 * 4^3) + 2 * 0 = 512 FLOPs per Monarch matmul (permutations are free). Vs 4,096 for full. 8x cheaper.

**Total per RowThenCol:** 2 * 512 = 1,024 FLOPs + activation. Vs 2 * 4,096 = 8,192. ~8x cheaper.

**Params:** 2 * 64 = 128 per A. Vs 256 full. 2x reduction.

**Quality tradeoff:** Monarch matrices can approximate any matrix (universal approximation with O(d log d) params and FLOPs). At d=16, the approximation quality with sqrt(d)=4 blocks should be quite good. The architecture-research-march2026.md already identified Monarch as "16x FLOP reduction on 256-dim; naturally decomposes into row/col operations on our 16x16 matrices."

**Assessment:** This is one of the strongest candidates. 8x cheaper FLOPs, well-characterized approximation quality, existing PyTorch code (github.com/HazyResearch/m2). The row/column decomposition of Monarch aligns naturally with the row-then-column structure of our projections. A Monarch A followed by a Monarch B means: permute rows, block-diagonal mix rows, permute rows, block-diagonal mix rows, then same for columns. This is essentially a factored version of what RowThenCol already does.

---

## Idea 5: Depthwise-Separable Matrix Operations

**What:** Borrow the insight from depthwise separable convolutions. Instead of mixing rows and columns jointly, do:
1. **Row-wise:** Apply a shared 1D transform to each row independently. Cost: d * d = d^2 per position.
2. **Column-wise:** Apply a shared 1D transform to each column independently. Cost: d * d = d^2 per position.

Concretely: `M_new = diag(w_row) @ M @ diag(w_col)` — pure diagonal scaling. Or: `M_new = (I + a @ b^T) @ M @ (I + c @ e^T)` — rank-1 perturbation.

**Cost:** Diagonal version: 2 * d^2 = 512 FLOPs. Rank-1 perturbation: 2 * (2 * d^2) = 2,048 FLOPs. Vs 8,192 for RowThenCol. 4-16x cheaper.

**Quality tradeoff:** VARIES. Pure diagonal is extremely restrictive — it can only scale rows/columns, never mix them. Rank-1 perturbation (I + a b^T) is more interesting: it mixes all rows through a single direction, which is exactly what the multiplicative composition (I + delta) * M * (I + gamma) was designed to do. The question is whether the delta and gamma need to be full-rank (RowThenCol) or rank-1 (this idea).

**Key insight:** The proven multiplicative composition already uses (I + delta) structure. If delta is computed cheaply (e.g., as a rank-1 or rank-2 outer product from M itself), rather than via a full RowThenCol projection, you preserve the mathematical structure while dropping the cost dramatically.

**Assessment:** Rank-1 perturbations for delta and gamma are 4x cheaper and preserve the (I + delta) * M * (I + gamma) structure exactly. This is architecturally identical to what we have, just with a cheaper way to compute delta and gamma. Rank-2 gives 2x cheaper. This directly attacks the bottleneck.

---

## Idea 6: Spectral Operations (SVD-Based)

**What:** Decompose M = U @ S @ V^T (thin SVD, keep top-k singular values). Then operate only on S: `S_new = f(S)` where f is a learned nonlinear function. Reconstruct: `M_new = U @ S_new @ V^T`.

**Cost:** Thin SVD for top-k on a 16x16 matrix: ~O(d^2 * k) with power iteration. At k=4: ~1,024 FLOPs. The function f on a 4-element vector: negligible. Reconstruction: ~1,024 FLOPs. Total: ~2,048 FLOPs. Vs 8,192 per RowThenCol: 4x cheaper.

**Params:** f can be a small MLP on k values: k * hidden * 2 = 4 * 16 * 2 = 128 params. Tiny.

**Quality tradeoff:** This ONLY modifies singular values, not singular vectors. The row and column spaces of M are preserved. This means spectral operations cannot change WHICH directions the matrix encodes — only how strongly it encodes them. For rank dynamics (our key metric), this is exactly what we want: we want to modulate how many directions are active (effective rank), not which directions they are (that is attention's job).

**Risk:** SVD is not differentiable at degenerate singular values (when two are equal). At d=16 with learned representations, this might happen. Mitigation: use soft-SVD or add epsilon to singular values before backprop. Or use power iteration (differentiable) instead of full SVD.

**Major risk:** SVD on the forward pass of every position at every layer is serial and hard to parallelize across positions. Even with batched SVD, the constant factor is high. This might be slower in wall-clock despite fewer FLOPs.

**Assessment:** Theoretically elegant but practically risky. SVD is expensive in wall-clock even if cheap in FLOPs, and gradient issues at degenerate points could destabilize training. Worth trying for the scientific insight (does the model need to rotate singular vectors or just modulate singular values?) but not the most practical path.

---

## Idea 7: Element-Wise Gating with Learned Masks

**What:** Instead of `silu(A @ M) @ B`, do `sigma(W1) * M * sigma(W2)` where W1 and W2 are d x d learned parameter matrices. The sigma functions produce element-wise gates. The `*` is Hadamard (element-wise) product.

**Cost:** Two Hadamard products: 2 * d^2 = 512 FLOPs. Vs 8,192 for RowThenCol: 16x cheaper.

**Params:** 2 * d^2 = 512. Same as RowThenCol (512 = 2 * 256).

**Quality tradeoff:** FUNDAMENTAL CHANGE. This is not a matrix operation — it is a position-independent element-wise scaling. It cannot mix rows or columns at all. It is strictly less expressive than RowThenCol. However, it IS data-dependent if you make W1 and W2 functions of M: `W1 = g(M), W2 = h(M)` for some cheap functions g, h (e.g., average pooling along rows/columns + broadcast).

**Data-dependent variant:** `gate_row = sigmoid(M.mean(dim=-1, keepdim=True))` (average each row, broadcast) then `M_new = gate_row * M * gate_col`. Cost: 2 * d^2 + 2 * d = 544 FLOPs. This is data-dependent but only along rows and columns independently — no cross-row-column interaction.

**Assessment:** Too weak on its own. But could be a CHEAP SUPPLEMENT to a stronger operation. E.g., one Monarch-factored RowThenCol (8x cheaper) + one element-wise gating (16x cheaper) = still much cheaper than two full RowThenCols, potentially similar expressiveness.

---

## Idea 8: Polynomial Matrix Transform

**What:** `M_new = c0 * I + c1 * M + c2 * M^2 + c3 * M^3`. This is a matrix polynomial. M^2 = M @ M, M^3 = M^2 @ M.

**Cost:** M^2 costs d^3 = 4,096 FLOPs. M^3 costs another d^3 = 4,096. Total: ~8,192 FLOPs for degree-3. Same as RowThenCol. BUT: if you precompute M^2 once and reuse it across all projections in the layer, the amortized cost drops. With 10 projections sharing M^2, the matmul cost is 4,096 / 10 + per-projection linear combo = ~600 FLOPs per projection. ~13x cheaper amortized.

**Params:** 4 scalar coefficients per projection. Or 4 * d^2 for element-wise coefficients (position-dependent polynomial). Extremely compact.

**Quality tradeoff:** Matrix polynomials can approximate any analytic function of the matrix (Cayley-Hamilton says degree d-1 suffices for d x d). At d=16 with degree 3, this is quite restrictive. But the key question is whether the RowThenCol projections actually need more than what a degree-3 polynomial provides.

**Deep issue:** The Cayley-Hamilton theorem says M^16 is a linear combination of I, M, ..., M^15 for any 16x16 matrix M. So a degree-15 polynomial is COMPLETE for d=16. A degree-3 polynomial is approximate but very cheap.

**Data-dependent variant:** Make the coefficients c_k data-dependent: `c_k = f_k(trace(M), ||M||_F, ...)`. Tiny cost, makes the polynomial adaptive.

**Assessment:** Interesting, especially with shared M^2 precomputation. The amortized cost is very attractive. The quality concern is that degree-3 may be too restrictive, and Cayley-Hamilton completeness at degree-15 is too expensive (15 matmuls = 60K FLOPs). A practical middle ground: degree-2 polynomial with data-dependent coefficients. Cost: one matmul (M^2) shared across 10 projections + cheap linear combos. ~1,000 FLOPs/projection amortized. Worth testing.

---

## Idea 9: Batched Reshape for Tensor Core Utilization

**What:** The problem with 16x16 matmuls is that they are too small for GPU tensor cores (which want at least 16x16x16, but are optimized for much larger). Instead of doing per-position 16x16 matmuls, reshape to do a single LARGE matmul across all positions simultaneously.

**Concretely:** For `A @ M` where A is (d, d) and M is (batch, seq, d, d):
- Reshape M to (batch * seq * d, d) — a tall skinny matrix.
- Compute A @ M^T as a (d, batch*seq*d) matmul — this is ONE large matmul.
- Reshape back.

With batch=64, seq=1024, d=16: M becomes (1,048,576, 16). The matmul is (16, 16) @ (16, 1,048,576) = a (16, 1,048,576) output. This is a standard GEMM with M=16, N=1,048,576, K=16.

**Speedup estimate:** On H100 tensor cores, this large matmul can achieve close to peak TFLOPS. The 16x16 per-position version has massive kernel launch overhead and cannot utilize tensor cores. Estimated 5-20x wall-clock improvement.

**BUT:** This is what PyTorch already does with `torch.einsum('ij,bsjk->bsik', A, M)` — einsum broadcasts A across the batch/seq dimensions. The question is whether the current einsum implementation is already doing the optimal reshaping. It may not be — einsum sometimes generates suboptimal loops for broadcast dimensions.

**What to actually try:**
```python
# Current (potentially suboptimal):
out = torch.einsum('ij,bsjk->bsik', A, M)

# Explicit reshape (force single GEMM):
B, S, d, _ = M.shape
out = (A @ M.reshape(B*S, d, d).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, S, d, d)

# Or using torch.matmul (handles batch broadcasting):
out = torch.matmul(A.unsqueeze(0).unsqueeze(0), M)
```

**Assessment:** This is a PURE IMPLEMENTATION optimization — same math, potentially much faster. Should be the FIRST thing to try before any algorithmic changes. Profile the existing einsum vs explicit reshape to see if there is a gap. On MPS (Mac), einsum might be especially suboptimal.

---

## Idea 10: Replace MultiplicativeThinkingLayer with a Single Geometric Product

**What:** CliffordNet replaces both attention AND FFN with a single geometric (Clifford) product. The geometric product on multivectors is a single bilinear operation that simultaneously captures:
- Scalar products (grades collapse)
- Cross products / rotations (grades change)
- Rank building (outer product component)

For our 16x16 matrices interpreted as elements of a Clifford algebra Cl(16), the geometric product `M_new = M1 * M2` (Clifford product) does all the mixing that RowThenCol does, but in one operation.

**Cost:** Clifford product for Cl(p,q) with p+q=n has 2^n components. For d=16, that is 2^16 = 65,536 components. WAY too expensive. But if we interpret our 16x16 matrix as Cl(4,0) (4D Clifford algebra, which has 2^4 = 16 components), the geometric product is 16^2 * 16 = 4,096 FLOPs — same as one matmul.

**The problem:** Our matrix is 16x16 = 256 elements, but Cl(4,0) only has 16 basis elements. We would need Cl(8,0) (256 basis elements, 2^8 = 256) but the geometric product for Cl(8,0) costs ~256^2 = 65,536 FLOPs. Back to the same cost.

**Alternative:** Use the geometric product only on a smaller Clifford algebra (e.g., Cl(4,0)) applied to ROWS of the matrix independently. Each row is 16-dimensional, interpreted as a Cl(4,0) multivector (16 components). The geometric product between two rows costs 16^2 = 256 FLOPs. For all 16 rows: 16 * 256 = 4,096 FLOPs. Same as one matmul but with richer algebraic structure.

**Assessment:** Theoretically elegant but practically unclear. The Clifford interpretation of our matrices is a stretch — our matrices arise from outer products, not from Clifford algebra generators. The geometric product has nice properties (associativity, invertibility) but whether they help for language modeling is unproven. CliffordNet (2601.06793) showed promising results but on different tasks and with different architecture. Worth exploring if other ideas fail, but not the first priority.

---

## Idea 11: Remove the Thinking Layer Entirely — Attention-Only Architecture

**What:** Eliminate the MultiplicativeThinkingLayer (6 RowThenCol projections). Keep only the attention layer (4 RowThenCol projections, or replace with standard vector attention on flattened matrices). The iterative refinement comes purely from repeated attention — just like a standard deep transformer.

**Specifically:**
1. Embed bytes as 16x16 matrices (outer product — keep this, it is proven).
2. Flatten to 256-dim vectors.
3. Standard transformer attention (flash-compatible, highly optimized).
4. Reshape to 16x16 for output head (or keep flat and use a learned output projection).
5. Loop T times (LoopFormer-style shared weights).

**Cost:** Standard attention on 256-dim vectors: ~2*256^2 = 131,072 FLOPs per position per layer for Q/K/V/O projections. This is LESS than our current matrix attention (4 * 8,192 = 32,768) but WAIT — the standard linear projections at d=256 are 4 * 256^2 = 262,144 FLOPs. Actually MORE. But standard attention uses Flash Attention which makes the memory-bound attention computation extremely fast.

**Hmm. Let me re-examine.** Our full model has:
- Matrix attention: 4 RowThenCol x 8,192 = 32,768 FLOPs for projections + attention computation
- Thinking layer: 6 RowThenCol x 8,192 = 49,152 FLOPs
- Total: ~82,000 FLOPs per position per layer

A flat d=256 transformer:
- Attention projections (Q/K/V/O): 4 * 256 * 256 = 262,144 FLOPs
- FFN (4x expansion): 2 * 256 * 1024 = 524,288 FLOPs
- Total: ~786,000 FLOPs per position per layer

So our matrix model is actually 10x CHEAPER per layer in FLOPs. The problem is that our layers are SLOWER in wall-clock because the 16x16 matmuls don't utilize hardware well.

**The REAL idea:** Keep matrix representation but use Frobenius attention (which IS just standard SDPA on flattened 256-dim vectors) and replace the MultiplicativeThinkingLayer with a standard FFN on flattened 256-dim vectors. This gives:
- Outer product embedding (proven advantage)
- Standard flash attention on 256-dim "vectors" (actually flattened matrices)
- Standard FFN on 256-dim (highly optimized, tensor core friendly)
- Reshape to 16x16 only at the output head
- LoopFormer-style looping for iterative refinement

**Cost:** Same as a standard d=256 transformer. ~786K FLOPs/position/layer. This is ~10x MORE FLOPs than our matrix model but MUCH faster in wall-clock (optimized kernels, tensor cores, flash attention).

**Quality tradeoff:** The matrix structure is preserved in the representation (embedding and output head) but lost in the computation. The Run 18 ablation showed this works: flat-vector at 24M params got BPB 1.011 vs matrix ops at 2.4M getting BPB 1.91. The key question from the project analysis: does it work at MATCHED PARAMS? If yes, matrix operations are dead for good.

**Assessment:** This is the "nuclear option" — abandon matrix operations entirely, keep only the embedding and output head. It is the most practical path to a working system. But it concedes the thesis that matrix operations are useful. Run the matched-param ablation first.

---

## Idea 12: Gated DeltaNet-Style Matrix Update

**What:** DeltaNet (and Gated DeltaNet) uses the delta rule to update a matrix state: `S_new = S - beta * S @ k @ k^T + beta * v @ k^T`. This is a rank-1 update that costs O(d^2) per step. Gated DeltaNet adds a forget gate: `S_new = alpha * S + beta * v @ k^T`.

**Apply to our setting:** Instead of the full MultiplicativeThinkingLayer with 6 RowThenCol projections, use a delta-rule update on each position's matrix:
```python
k = linear_k(M.flatten())  # d-dim key from flattened matrix, d^2 FLOPs
v = linear_v(M.flatten())  # d-dim value, d^2 FLOPs
beta = sigmoid(linear_beta(M.flatten()))  # scalar gate, d^2 FLOPs
M_new = M - beta * M @ k.unsqueeze(-1) @ k.unsqueeze(-2) + beta * v.unsqueeze(-1) @ k.unsqueeze(-2)
```

**Cost:** 3 linear projections from d^2 to d: 3 * d^2 * d = 3 * 4,096 = 12,288 FLOPs. The delta update: M @ k = d^2 FLOPs, outer products = 2 * d^2 FLOPs. Total: ~13,000 FLOPs. Vs 49,152 for the full thinking layer. ~3.8x cheaper.

**BUT:** We could compute k and v WITHOUT flattening: `k = M @ w_k` (d^2 FLOPs), `v = M @ w_v` (d^2 FLOPs). This keeps the matrix structure and is even cheaper (no flatten+unflatten).

**Quality tradeoff:** The delta rule update is a rank-1 perturbation to M. One delta-rule step can increase the rank by at most 1. This is actually EXACTLY what we want for rank enrichment — gradual rank building across iterations. The current (I+delta)*M*(I+gamma) can increase rank by up to min(rank(delta), rank(gamma)), which is potentially higher per step but also noisier.

**Assessment:** Very promising. The delta rule is computationally cheap, well-studied (Mamba-2, DeltaNet, Gated DeltaNet all use it), naturally builds rank by 1 per step, and preserves the matrix structure. The key concern is whether rank-1 updates per iteration are sufficient — with T=8 iterations, you can build from rank 1 to rank 9, which matches the observed effective ranks of 5-6. This is a strong candidate.

---

## Idea 13: Householder Reflections (Revisited, Non-Orthogonal Variant)

**What:** You analyzed Householder reflections and correctly identified the problem: they are orthogonal transformations, which preserve singular values and cannot change rank. But what about GENERALIZED Householder transformations?

A generalized Householder matrix is: `H = I - beta * v @ w^T` where v and w are NOT the same vector, and beta is NOT constrained to 2/(v^T v). This is NOT orthogonal. It CAN change singular values and rank.

**Cost:** `H @ M = M - beta * v @ (w^T @ M)` = one vector-matrix product (d^2 FLOPs) + one outer product (d^2 FLOPs) + one subtraction (d^2 FLOPs) = 3 * d^2 = 768 FLOPs. For K=4 generalized reflections: 4 * 768 = 3,072 FLOPs. Vs 4,096 for one matmul. 25% cheaper per side (A or B), ~25% cheaper overall for RowThenCol.

**Params:** K * (2d + 1) per side. K=4: 4 * 33 = 132 params. Vs 256 for full A. 2x fewer.

**Quality:** Generalized Householder with K=4 reflections can represent any matrix with rank deficiency at most 4 (for the product H_4 @ H_3 @ H_2 @ H_1). At d=16 with K=4, this means the projection cannot have rank less than 12. For K=d=16, it is EXACTLY a full matrix (this is the Householder QR factorization generalized).

**The key advantage over standard Householder:** Each generalized reflection H = I - beta * v @ w^T has 2d+1 = 33 params and can modify BOTH the direction and magnitude of the transformation. Standard Householder (v=w, beta=2/v^Tv) is a rigid reflection that preserves norms. Generalized Householder can stretch, shrink, and shear — exactly what rank dynamics need.

**Assessment:** Moderately promising. The savings are modest (25% per matmul) compared to other ideas, but the parameterization is clean and the approximation quality is well-characterized. Could be combined with Idea 2 (diagonal + generalized Householder) for better results.

---

## Idea 14: TTT-Style Self-Supervised Matrix Update

**What:** TTT (Test-Time Training) treats the hidden state as a weight matrix and updates it via gradient descent on a self-supervised loss at each token. Apply this to our matrix tokens: instead of a learned (I+delta)*M*(I+gamma) update, compute delta and gamma as GRADIENTS of a local self-supervised objective.

**Concretely:**
```python
# Self-supervised loss: predict some property of M from a compressed version
z = M.mean(dim=-1)  # compress to d-dim vector
M_reconstructed = z.unsqueeze(-1) @ z.unsqueeze(-2)  # rank-1 reconstruction
loss = (M - M_reconstructed).pow(2).sum()  # reconstruction error
# Gradient gives the direction that increases M's rank
delta = torch.autograd.grad(loss, M)  # shape (d, d), costs ~d^2
M_new = M + lr * delta
```

**Cost:** The reconstruction and loss are O(d^2). The gradient computation is also O(d^2) (chain rule through simple operations). Total: ~4 * d^2 = 1,024 FLOPs. Vs 49,152 for the thinking layer. ~48x cheaper.

**Quality tradeoff:** UNKNOWN. This replaces LEARNED matrix updates with a FIXED self-supervised objective. The model cannot learn what updates are useful — the update rule is hardcoded. However, the specific objective (increase rank = increase reconstruction error from rank-1 projection) is EXACTLY aligned with what we observe the model wanting to do (rank enrichment).

**Risk:** This is speculative. No one has tried TTT-style updates for matrix-valued token representations. The gradient might point in useless directions. The learning rate is a sensitive hyperparameter.

**Assessment:** Highly speculative but conceptually compelling. The self-supervised gradient gives a "free" rank-enriching update direction. If it works, it eliminates the need for learned delta/gamma projections entirely. Worth a small-scale test, but not the primary approach.

---

## Idea 15: Factored RowThenCol via Kronecker Structure

**What:** Our matrices are 16x16. We can view them as 4x4 blocks of 4x4 sub-matrices (a Kronecker-compatible view). Then replace A and B with Kronecker products: `A = A1 (x) A2` where A1 is 4x4 and A2 is 4x4. The Kronecker product A1 (x) A2 acting on a 16x16 matrix M (viewed as a 4x4 array of 4x4 blocks) costs: A1 mixes between blocks (4^3 = 64 FLOPs per block column, 4 block columns = 256), A2 mixes within blocks (4^3 = 64 per block, 16 blocks = 1,024). Total: ~1,280 FLOPs. Vs 4,096 for full. 3.2x cheaper.

**Params:** 2 * 16 = 32 per Kronecker factor. Vs 256 for full A. 8x fewer.

**Quality:** Kronecker products are rank-1 in the "block matrix" sense. A sum of K Kronecker products (as in our existing Sum-of-Kroneckers K=4 analysis) gives rank-K in block-matrix space. K=4 with 128 params each is 512 total params — same as RowThenCol but with richer structure.

**Assessment:** This was already analyzed in the project (matrix-native-projections.md) and found to be promising at K=4 with 32x fewer params. The question is whether it was ever IMPLEMENTED AND TESTED. If not, this should be tested. It is a well-understood factorization with predictable quality.

---

## Idea 16: Attention-Only with Matrix-Preserving Value Mixing

**What:** Instead of separate attention + thinking layer, do ATTENTION with matrix-structured value aggregation. Standard attention computes `out = softmax(Q @ K^T / sqrt(d)) @ V`. For matrix tokens, V is a matrix. The weighted sum of matrices IS a matrix. The attention already mixes positions. The missing piece is per-position nonlinear transformation — but what if we just use a simple nonlinearity?

```python
# Standard Frobenius attention (already fast via SDPA)
attn_out = frobenius_attention(M)  # linear combination of matrix values

# Simple nonlinear per-position transform (NO RowThenCol)
M_new = M + silu(attn_out) * scale  # element-wise nonlinearity on matrix
```

**Cost:** Frobenius attention: same as standard SDPA on 256-dim vectors. Per-position nonlinearity: d^2 = 256 FLOPs. Total: essentially just the attention cost, which is O(L^2 * d^2) for sequence length L and is ALREADY optimized via Flash Attention.

**Quality tradeoff:** This eliminates ALL RowThenCol projections from the thinking layer (saving 49,152 FLOPs/position/layer) and also from the attention Q/K/V/O projections (saving 32,768 FLOPs/position/layer). Instead, attention scores are computed via raw Frobenius inner products (no learned projections — M IS Q, K, and V). This is extremely simple but the question is whether learned Q/K/V projections are needed when the matrices already have rich structure from the outer-product embedding.

**Projection-free attention variant:**
```python
# No Q/K/V projections at all
scores = einsum('bqij,bkij->bqk', M, M) / sqrt(d*d)  # Frobenius inner product
weights = softmax(scores)
attn_out = einsum('bqk,bkij->bqij', weights, M)
M_new = M + silu(attn_out)
```

**Assessment:** This is the maximally cheap version. It eliminates ALL learned matrix operations. The matrix structure exists purely in the representation (embedding + output), and all computation is standard scalar attention + element-wise nonlinearities. If this works, it proves that the matrix REPRESENTATION is the contribution, and the matrix OPERATIONS are unnecessary. This is actually the key experiment the project-analysis-march2026.md recommends: "the paper should be about the embedding, not the operations."

---

## Idea 17: Interleave Cheap and Expensive Layers

**What:** Not all layers need full RowThenCol projections. Use the expensive multiplicative layer only every K-th layer, and use a cheap operation (diagonal gating, element-wise SiLU, or delta-rule update) for the others.

**Example with K=4 (1 expensive per 4 layers):**
- Layer 1: Frobenius attention (cheap) + diagonal gating (cheap)
- Layer 2: Frobenius attention (cheap) + delta-rule update (cheap)
- Layer 3: Frobenius attention (cheap) + diagonal gating (cheap)
- Layer 4: Frobenius attention (cheap) + FULL multiplicative thinking (expensive)

**Cost:** If 3/4 of thinking layers are 10x cheaper, the average cost per layer drops by ~7x. At 12 layers: 3 full thinking + 9 cheap = ~20% of the original cost.

**Quality tradeoff:** The expensive layers provide the full row-column mixing needed for rank building. The cheap layers provide refinement and gating without the heavy cross-dimension mixing. This mirrors MoE architectures where only a fraction of capacity is activated per token.

**Assessment:** Pragmatic and easily testable. The key question is which layers should be expensive. Based on the rank dynamics observation (rank increases most in early iterations), the expensive layers should probably be early in the layer stack. Late layers may only need fine-tuning of established structure — cheap ops suffice.

---

## Idea 18: Mamba-2 Style: Matrix State with Scalar Gating

**What:** Mamba-2 uses matrix-valued states but keeps the per-step update cheap: `S_new = alpha * S + beta * B @ x @ C^T` where alpha, beta are scalars, B and C are input-dependent vectors, and x is the input. This is a gated rank-1 update to the matrix state.

**Apply to our per-position matrices:** Instead of (I+delta)*M*(I+gamma), do:
```python
# Compute input-dependent vectors from M
b = M.mean(dim=-1)  # row-wise average, d-dim
c = M.mean(dim=-2)  # col-wise average, d-dim
alpha = sigmoid(linear_alpha(M.flatten()))  # scalar forget gate
beta = sigmoid(linear_beta(M.flatten()))  # scalar update gate

# Rank-1 update
M_new = alpha * M + beta * b.unsqueeze(-1) @ c.unsqueeze(-2)
```

**Cost:** Two averages: 2 * d^2 FLOPs. Two scalar gates: 2 * d^2 FLOPs (linear from d^2 to 1). One outer product: d^2 FLOPs. One scaling + add: 2 * d^2 FLOPs. Total: ~7 * d^2 = 1,792 FLOPs. Vs 49,152 for full thinking layer. ~27x cheaper.

**Quality:** Rank-1 update per iteration. Very limited per step, but compounding over T=8 iterations can build rank up to 9 (initial 1 + 8 updates). The scalar gating (alpha, beta) controls the trade-off between preserving existing structure and adding new directions.

**Key advantage:** This matches the Mamba-2 recipe which is proven at scale. The scalar gating is critical — it prevents the matrix from growing unboundedly.

**Assessment:** Strong candidate. 27x cheaper, well-aligned with proven SSM architectures, naturally builds rank. The main risk is that the update vectors (b, c) derived from row/column averages may be too correlated with M's existing structure to add genuinely new rank directions. Could mitigate by projecting b, c through small learned transforms before the outer product.

---

## Idea 19: Shared Projections Across All 10 RowThenCols

**What:** Currently each of the 10 RowThenCol projections has its own A and B (10 * 512 = 5,120 params per layer). What if they SHARED A and B, with only a cheap per-projection modulation?

```python
# Shared A, B (512 params, computed once)
shared_intermediate = silu(A @ M)  # one left matmul for ALL projections
shared_output = shared_intermediate @ B  # one right matmul

# Per-projection: cheap diagonal modulation (d params each)
proj_k = diag(scale_k) @ shared_output + bias_k  # 2d params per projection
```

**Cost:** 2 matmuls shared (8,192 FLOPs) + 10 * (d^2 + d^2) diagonal ops = 8,192 + 5,120 = 13,312 FLOPs. Vs 10 * 8,192 = 81,920 original. ~6x cheaper.

**Params:** 512 shared + 10 * 32 = 832. Vs 5,120 original. ~6x fewer.

**Quality tradeoff:** All 10 projections see the same "base" transformation. The diagonal modulation can only scale the output per-element, not mix it. This means the Q, K, V, O, delta_gate, delta_value, delta_up, gamma_gate, gamma_value, gamma_up projections all start from the same nonlinear mixture. The diversity comes only from diagonal rescaling.

**Risk:** This was partially tested (Run 15: "shared gates + low-rank r=8" got 3% worse BPB at 12% faster). That was sharing only the gate projections. Sharing ALL projections is more aggressive and may hurt more.

**Mitigation:** Share A but not B. Each projection has its own B but they share the expensive silu(A @ M) computation. Cost: 1 matmul shared + 10 matmuls independent = 11 * 4,096 = 45,056. Still ~1.8x cheaper. More modest but safer.

**Assessment:** Moderate improvement (1.8-6x) with moderate risk. The "share A, independent B" variant is the safest bet.

---

## Idea 20: Replace RowThenCol with a Single Data-Dependent Scaling

**What:** The MOST radical simplification. Replace ALL matrix operations with:
```python
# Per-position learned scaling
scale = sigmoid(linear(M.flatten()))  # (batch, seq, d, d) -> (batch, seq, 1)
M_new = scale * M + (1 - scale) * M_mean  # interpolate toward sequence average
```

This is a single linear projection (d^2 params) followed by an interpolation. No matmuls on M at all.

**Cost:** One linear: d^2 * 1 = 256 FLOPs. One interpolation: 2 * d^2 = 512 FLOPs. Total: 768 FLOPs. Vs 81,920. ~107x cheaper.

**Quality:** TERRIBLE. This cannot mix rows, columns, or build rank. It can only interpolate each position's matrix toward the mean. It is essentially a soft average.

**Why include it:** As a LOWER BOUND. If even this trivial operation, combined with the outer-product embedding and LoopFormer-style iteration, achieves reasonable BPB, then matrix operations are definitively unnecessary. This is the ultimate ablation.

---

## Idea 21: FFT-Based Circular Matrix Multiply

**What:** Circulant matrices can be diagonalized by the DFT: `C = F^(-1) @ diag(F @ c) @ F` where c is the first column and F is the DFT matrix. So `C @ M = ifft(fft(c) * fft(M_cols))` (applied column-wise). Cost: d * d * log(d) = 16 * 16 * 4 = 1,024 FLOPs. Vs d^3 = 4,096 for full matmul. 4x cheaper.

**Params:** Circulant A needs only d = 16 params (its first column). Vs d^2 = 256 for full. 16x fewer.

**Quality:** Circulant matrices have a very specific structure — they are diagonalizable by the DFT, which means they act as independent filters on each frequency component. This is great for signal processing but might be too structured for general matrix mixing.

**Extension:** Use a block-circulant + diagonal correction: `A = circ(c) + diag(a)`. Cost: 1,024 + 256 = 1,280 FLOPs. Params: 32. Much more expressive than pure circulant.

**Assessment:** Interesting for the FLOPs savings but circulant structure is a very strong inductive bias. Whether it helps or hurts depends on whether the row/column mixing patterns in our learned projections have any periodicity. Probably not — this is a long shot. But if combined with random permutations (changing which "frequency" maps to which row), it could be more general.

---

## Idea 22: Mixed-Precision Batched Matmul

**What:** Use FP8 (8-bit floating point) for the RowThenCol matmuls. On H100, FP8 tensor cores give 2x the TFLOPS of FP16. Our 16x16 matrices are small, so the concern about precision loss is amplified (fewer elements to absorb rounding error).

**Speedup:** 2x wall-clock on H100 (from 1979 TFLOPS FP8 vs 990 TFLOPS FP16). On MPS (Apple Silicon): no FP8 support, so no benefit on current dev hardware.

**Quality tradeoff:** FP8 has only 3 bits of mantissa. For 16x16 matmuls, this means effective precision of ~1 part in 8 per element. Accumulation in FP16 or FP32 helps, but the intermediate silu activation on FP8 values will be noisy. Need to measure the impact on training stability.

**Practical concern:** As noted in the problem statement, 16x16 matmuls are too small for tensor core speedup — you need the matrices to be aligned to at least 16-element boundaries, and the overhead of FP8 quantization/dequantization may dominate for such small tiles. The batched reshape approach (Idea 9) would help here: reshape to a large matmul THEN apply FP8.

**Assessment:** Only useful on H100+, only with the batched reshape, and the quality impact is uncertain. 2x speedup is nice but not transformative. Lower priority.

---

## Summary Ranking

| # | Idea | Est. Speedup | Quality Risk | Priority |
|---|------|-------------|-------------|----------|
| 9 | Batched reshape for tensor cores | 5-20x wall-clock | None | **HIGHEST** |
| 1 | Fused Triton kernel | 2-4x wall-clock | None | **HIGH** |
| 12 | DeltaNet-style delta-rule update | 3.8x FLOPs | Low | **HIGH** |
| 4 | Monarch factorization | 8x FLOPs | Low-medium | **HIGH** |
| 18 | Mamba-2 style scalar gating | 27x FLOPs | Medium | **HIGH** |
| 5 | Depthwise-separable (rank-1 perturbation) | 4-16x FLOPs | Medium | **HIGH** |
| 17 | Interleave cheap/expensive layers | 4-7x FLOPs | Low | **MEDIUM** |
| 19 | Shared projections (share A) | 1.8-6x FLOPs | Low-medium | **MEDIUM** |
| 3 | Block-diagonal + shuffle | 8x FLOPs | Medium | **MEDIUM** |
| 2 | Diagonal + low-rank | 2-3x FLOPs | Low | **MEDIUM** |
| 8 | Polynomial (shared M^2) | 4-13x amortized | Medium | **MEDIUM** |
| 15 | Kronecker factorization | 3.2x FLOPs | Low | **MEDIUM** |
| 16 | Attention-only, no thinking layer | removes thinking cost | High | **TEST** |
| 11 | Full flatten to vectors | removes all matrix ops | Very high | **TEST** |
| 13 | Generalized Householder | 1.3x FLOPs | Low | LOW |
| 6 | Spectral (SVD-based) | 4x FLOPs | High (SVD cost) | LOW |
| 7 | Element-wise gating | 16x FLOPs | Very high | LOW |
| 14 | TTT-style self-supervised | 48x FLOPs | Unknown | SPECULATIVE |
| 10 | Clifford geometric product | ~1x FLOPs | Unknown | SPECULATIVE |
| 21 | FFT-based circulant | 4x FLOPs | High | LOW |
| 22 | Mixed FP8 precision | 2x wall-clock (H100 only) | Medium | LOW |
| 20 | Trivial interpolation (ablation) | 107x FLOPs | Very high | ABLATION |

---

## Recommended Experiment Plan

### Phase A: Implementation optimizations (no quality tradeoff)
1. **Idea 9:** Profile current einsum vs batched reshape. If there is a gap, fix it. This is free performance.
2. **Idea 1:** Write fused Triton kernel for RowThenCol. Combine with Idea 9.

### Phase B: Cheap drop-in replacements (small quality risk)
3. **Idea 12 (DeltaNet delta-rule):** Replace MultiplicativeThinkingLayer with delta-rule updates. Preserve the attention RowThenCols. 3.8x cheaper thinking layer.
4. **Idea 4 (Monarch):** Replace A, B in RowThenCol with Monarch-factored matrices. 8x cheaper per RowThenCol.
5. **Idea 5 (rank-1 perturbation for delta/gamma):** In the multiplicative composition (I+delta)*M*(I+gamma), compute delta and gamma as rank-1 or rank-2 outer products from M instead of via RowThenCol. Preserves the proven (I+delta) structure.

### Phase C: Architectural changes (moderate quality risk)
6. **Idea 17 (interleave):** Use full thinking layer every 4th layer, cheap ops otherwise.
7. **Idea 18 (Mamba-2 style):** Replace thinking layer with scalar-gated rank-1 updates.
8. **Idea 19 (shared A):** Share the left projection across all RowThenCols in a layer.

### Phase D: Key ablations
9. **Idea 16 (attention-only):** Drop the thinking layer entirely. Does attention + nonlinearity suffice with the matrix embedding?
10. **Idea 11 (full flatten):** The nuclear option. Flatten to vectors, standard transformer, reshape at output. If this matches matrix ops at matched params, the operations thesis is dead.
11. **Idea 20 (trivial interpolation):** Lower bound. How much does any nontrivial per-position operation matter?

---

## The Highest-Leverage Combination

If I had to pick ONE compound approach to implement first, it would be:

**Monarch-factored rank-2 multiplicative composition with shared left projection and fused kernel.**

Concretely:
1. Compute `silu(A_monarch @ M)` ONCE using Monarch-factored A (8x cheaper, shared across all projections in the layer).
2. For each of the 10 projections, apply an independent Monarch-factored B_k: `proj_k = silu(shared_intermediate) @ B_k_monarch`.
3. For delta and gamma in (I+delta)*M*(I+gamma), use rank-2 outer products derived from the shared intermediate instead of separate RowThenCol projections.
4. Fuse steps 1-3 into a single Triton kernel.

**Estimated total cost:** ~8,000 FLOPs/position/layer. Vs ~82,000 original. ~10x cheaper. If combined with the batched reshape for tensor core utilization, wall-clock could improve by 20-50x, putting us within 2-3x of LoopFormer's speed while preserving the matrix structure.

**The remaining gap to LoopFormer:** LoopFormer at matched FLOPs got BPB 0.87 vs our 1.67. A 10x FLOPs reduction means we can run 10x more iterations or 10x more layers at the same budget. If our BPB scales with compute (more iterations help), we might close the gap. If it does not (the learning algorithm is the bottleneck, not compute), then we need the architectural changes from Phase D.
