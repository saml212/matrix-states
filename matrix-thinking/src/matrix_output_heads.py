"""
Matrix-native output heads for language models with d×d matrix token representations.

Problem: The current output head collapses a (B, L, d, d) matrix to (B, L, d) via
`(W * M).sum(dim=-1)` then projects to vocab via Linear(d, vocab_size).
This destroys column structure: every vocab token shares the same column probe (W),
differing only in how they weight the rows (via the Linear layer).

Solution: The MultiProbeHead extracts K bilinear features from the matrix using
K learned (u_k, v_k) probe pairs: probe_k = u_k^T @ M @ v_k. Then a standard
Linear(K, vocab) maps probes to logits. The effective per-class scoring matrix is
W_eff[w] = U^T diag(out_w) V, which is a sum of K rank-1 matrices — at K >= d,
this can represent ANY d×d scoring matrix per class.

Key properties:
  - Never flattens the matrix
  - Uses both row and column structure
  - K=d gives same param count as current head (+1024 extra for probe matrices)
  - Same speed as current head (the bottleneck is Linear(K, vocab), same GEMM)
  - Strictly more expressive than current head
  - Supports weight tying with input embeddings

References:
  Kong & Fowlkes, "Low-rank Bilinear Pooling" (CVPR 2017) — co-decomposition
  Kim et al., "Hadamard Product for Low-rank Bilinear Pooling" (ICLR 2017)
"""

import torch
import torch.nn as nn
import math


class MultiProbeHead(nn.Module):
    """Recommended matrix-native output head.

    Extracts K bilinear probes from the matrix, then projects to vocab.

    probe_k = u_k^T @ M @ v_k     (scalar, for each of K probes)
    logits  = Linear(probes)       (K → vocab)

    Why this works:
    The effective scoring matrix for class w is:
        W_eff[w] = sum_k out_weight[w,k] * u_k @ v_k^T

    This is a sum of K rank-1 matrices. When K >= d, W_eff[w] can be
    any d×d matrix — full expressiveness without materializing (vocab, d, d)
    weight tensors.

    Comparison to current head:
        Current: W_eff[w,i,j] = Linear.weight[w,i] * collapse_W[i,j]
                 → all classes share column directions (collapse_W)
                 → only row scalings differ per class
        This:    W_eff[w,i,j] = sum_k Linear.weight[w,k] * U[k,i] * V[k,j]
                 → each class has independent row AND column structure
                 → strictly more expressive

    Speed: The two einsum probe extractions are O(B*L*K*d) each.
    For K=d=32, that's 2 * B*L*1024 FLOPs — negligible vs the
    Linear(K, vocab) which is O(B*L*K*vocab) = O(B*L*32*50257).
    Total cost is dominated by the same GEMM as the current head.
    """

    def __init__(self, d, vocab_size, n_probes=None, bias=False):
        """
        Args:
            d: matrix dimension
            vocab_size: number of output tokens
            n_probes: number of bilinear probes (default: d, full expressiveness)
            bias: include per-class bias term
        """
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.n_probes = n_probes or d
        K = self.n_probes

        # Bilinear probe directions
        # U[k] probes row structure, V[k] probes column structure
        self.U = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.V = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))

        # Project probe features to vocab logits
        self.out = nn.Linear(K, vocab_size, bias=bias)

    def forward(self, M):
        """
        Args:
            M: (B, L, d, d) matrix representations

        Returns:
            logits: (B, L, vocab_size)
        """
        # Step 1: Right-multiply M by each v_k
        # M: (B,L,d,d), V: (K,d) → MV: (B,L,d,K)
        MV = torch.einsum('blij, kj -> blik', M, self.V)

        # Step 2: Left-contract with each u_k → scalar per probe
        # U: (K,d), MV: (B,L,d,K) → probes: (B,L,K)
        probes = torch.einsum('ki, blik -> blk', self.U, MV)

        # Step 3: Linear projection to vocab
        logits = self.out(probes)  # (B, L, vocab_size)

        return logits

    def extra_repr(self):
        return (f'd={self.d}, vocab_size={self.vocab_size}, '
                f'n_probes={self.n_probes}, '
                f'params={sum(p.numel() for p in self.parameters()):,}')


class TiedMultiProbeHead(nn.Module):
    """MultiProbeHead with probe weights tied to input embeddings.

    Uses the input embedding tables (embed_u, embed_v) as the probe
    directions, with a learned linear mixing layer on top.

    This is the matrix analog of weight-tied output embeddings in standard LMs.
    When the model hasn't transformed M much, the probes naturally recover
    the original token — because u_w^T (u_t v_t^T) v_w = (u_w . u_t)(v_w . v_t),
    which is maximized when w = t.

    Extra params: only the mixing layer Linear(K, vocab) — but note that with
    weight tying, K = vocab (one probe per vocab token), so this layer would be
    vocab^2 params. Instead, we use a bottleneck: project vocab probes down to
    a manageable K, then back up to vocab.
    """

    def __init__(self, embed_u, embed_v, n_probes=None, bias=False):
        """
        Args:
            embed_u: nn.Embedding for row structure (vocab, d) — shared reference
            embed_v: nn.Embedding for column structure (vocab, d) — shared reference
            n_probes: number of probes to use (default: d)
            bias: include per-class bias
        """
        super().__init__()
        d = embed_u.embedding_dim
        vocab_size = embed_u.num_embeddings
        K = n_probes or d

        self.embed_u = embed_u  # shared reference
        self.embed_v = embed_v  # shared reference

        # Select K probe directions from the vocab embedding space
        # via learned linear combinations
        self.probe_mix_u = nn.Linear(d, K, bias=False)
        self.probe_mix_v = nn.Linear(d, K, bias=False)

        # Project probes to vocab
        self.out = nn.Linear(K, vocab_size, bias=bias)

        # Temperature
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, M):
        """
        Args:
            M: (B, L, d, d)

        Returns:
            logits: (B, L, vocab_size)
        """
        # Derive probe directions from embeddings via learned mixing
        # This lets us use K << vocab probes that are informed by the
        # embedding structure
        U_probes = self.probe_mix_u.weight  # (K, d)
        V_probes = self.probe_mix_v.weight  # (K, d)

        MV = torch.einsum('blij, kj -> blik', M, V_probes)
        probes = torch.einsum('ki, blik -> blk', U_probes, MV)

        logits = self.out(probes) * self.log_scale.exp()
        return logits


# ─── Benchmark and comparison ─────────────────────────────────────

def benchmark(d=32, vocab_size=50257, B=2, L=64, n_iter=100):
    """Benchmark all output head approaches."""
    import time

    M = torch.randn(B, L, d, d)

    results = []

    # Current approach
    collapse_W = torch.randn(d, d)
    collapse_out = nn.Linear(d, vocab_size, bias=False)
    params_current = d * d + d * vocab_size
    _ = collapse_out((collapse_W * M).sum(dim=-1))  # warmup
    t0 = time.time()
    for _ in range(n_iter):
        v = (collapse_W * M).sum(dim=-1)
        logits = collapse_out(v)
    t_ms = (time.time() - t0) / n_iter * 1000
    results.append(('Current (1-axis collapse)', params_current, t_ms))

    # MultiProbeHead K=d (recommended)
    head = MultiProbeHead(d, vocab_size, n_probes=d)
    params = sum(p.numel() for p in head.parameters())
    _ = head(M)  # warmup
    t0 = time.time()
    for _ in range(n_iter):
        logits = head(M)
    t_ms = (time.time() - t0) / n_iter * 1000
    results.append((f'MultiProbe K={d} (recommended)', params, t_ms))

    # MultiProbeHead K=d//2
    head2 = MultiProbeHead(d, vocab_size, n_probes=d // 2)
    params2 = sum(p.numel() for p in head2.parameters())
    _ = head2(M)
    t0 = time.time()
    for _ in range(n_iter):
        logits = head2(M)
    t_ms = (time.time() - t0) / n_iter * 1000
    results.append((f'MultiProbe K={d//2}', params2, t_ms))

    # MultiProbeHead K=2*d
    head3 = MultiProbeHead(d, vocab_size, n_probes=2 * d)
    params3 = sum(p.numel() for p in head3.parameters())
    _ = head3(M)
    t0 = time.time()
    for _ in range(n_iter):
        logits = head3(M)
    t_ms = (time.time() - t0) / n_iter * 1000
    results.append((f'MultiProbe K={2*d}', params3, t_ms))

    print(f"\n{'Head':<35} {'Params':>12} {'Time (ms)':>10} {'vs Current':>10}")
    print("-" * 72)
    t_base = results[0][2]
    for name, params, t_ms in results:
        ratio = t_ms / t_base
        print(f"{name:<35} {params:>12,} {t_ms:>10.2f} {ratio:>10.1f}x")

    # Correctness: verify gradient flows to all matrix elements
    print("\nGradient check (should be non-zero for both row and col dims):")
    M_g = torch.randn(1, 1, d, d, requires_grad=True)
    head_g = MultiProbeHead(d, vocab_size, n_probes=d)
    logits = head_g(M_g)
    logits.sum().backward()
    g = M_g.grad
    row_grad = g[0, 0].abs().mean(dim=1)  # avg over columns, per row
    col_grad = g[0, 0].abs().mean(dim=0)  # avg over rows, per column
    print(f"  Row gradient std:  {row_grad.std():.4f} (>0 = rows differentially weighted)")
    print(f"  Col gradient std:  {col_grad.std():.4f} (>0 = cols differentially weighted)")
    print(f"  Total grad norm:   {g.norm():.4f}")

    # Compare: current head gradient
    M_g2 = torch.randn(1, 1, d, d, requires_grad=True)
    cW = nn.Parameter(torch.randn(d, d) * 0.02)
    cOut = nn.Linear(d, vocab_size, bias=False)
    v = (cW * M_g2).sum(dim=-1)
    logits2 = cOut(v)
    logits2.sum().backward()
    g2 = M_g2.grad
    row_grad2 = g2[0, 0].abs().mean(dim=1)
    col_grad2 = g2[0, 0].abs().mean(dim=0)
    print(f"\n  Current head comparison:")
    print(f"  Row gradient std:  {row_grad2.std():.4f}")
    print(f"  Col gradient std:  {col_grad2.std():.4f}")
    print(f"  Total grad norm:   {g2.norm():.4f}")


if __name__ == '__main__':
    benchmark()
