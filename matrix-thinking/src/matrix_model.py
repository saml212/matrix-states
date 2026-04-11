"""
Matrix-Thinking Model: tokens are matrices, operations are matrix-matrix.

The fundamental experiment: does a model that thinks in matrices develop
meaningful rank dynamics? Do hard-to-predict tokens have higher rank
representations than easy-to-predict ones?

Architecture:
  - Each byte is embedded as a rank-1 matrix (outer product of two vectors)
  - "Thinking" layers perform matrix-to-matrix operations via bilinear maps
  - Output projects matrix → vector only at the final step
  - Rank is measured at every layer for analysis

This is a from-the-math-up design, not a modification of transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MatrixEmbedding(nn.Module):
    """Embed bytes as rank-1 matrices via outer product.

    Each byte b gets two vectors u_b and v_b, and the embedding is u_b ⊗ v_b
    (the outer product), which is a rank-1 matrix.
    """

    def __init__(self, vocab_size: int, mat_dim: int):
        super().__init__()
        self.mat_dim = mat_dim
        self.embed_u = nn.Embedding(vocab_size, mat_dim)
        self.embed_v = nn.Embedding(vocab_size, mat_dim)
        nn.init.normal_(self.embed_u.weight, std=0.5)
        nn.init.normal_(self.embed_v.weight, std=0.5)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_ids: (B, L) long tensor
        Returns:
            matrices: (B, L, mat_dim, mat_dim) — each token is a matrix
        """
        u = self.embed_u(byte_ids)  # (B, L, d)
        v = self.embed_v(byte_ids)  # (B, L, d)
        # Outer product: u ⊗ v = u * v^T, giving a rank-1 matrix
        return torch.einsum('...i,...j->...ij', u, v)  # (B, L, d, d)


class MatrixInteraction(nn.Module):
    """Core thinking operation: matrix-to-matrix transformation.

    Given a token matrix M (d×d), produce a new matrix M' (d×d) by:
    M' = A @ M @ B + C @ M^T @ D

    Where A, B, C, D are learned (small) matrices. The first term captures
    forward structure, the second captures transpose structure (relationships
    between rows and columns swap — a genuinely matrix-native operation that
    has no vector analog).

    This IS the "thinking." Not attention-then-MLP. Just matrices
    interacting with matrices through multiplication.
    """

    def __init__(self, mat_dim: int):
        super().__init__()
        self.mat_dim = mat_dim
        # Forward path: A @ M @ B
        self.A = nn.Parameter(torch.randn(mat_dim, mat_dim) * 0.02)
        self.B = nn.Parameter(torch.randn(mat_dim, mat_dim) * 0.02)
        # Transpose path: C @ M^T @ D
        self.C = nn.Parameter(torch.randn(mat_dim, mat_dim) * 0.02)
        self.D = nn.Parameter(torch.randn(mat_dim, mat_dim) * 0.02)
        # Bias (matrix-valued)
        self.bias = nn.Parameter(torch.zeros(mat_dim, mat_dim))
        # Layer norm (normalize Frobenius norm)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: (B, L, d, d) input matrices
        Returns:
            M_new: (B, L, d, d) output matrices
        """
        # Forward: A @ M @ B
        fwd = torch.einsum('ij,...jk,kl->...il', self.A, M, self.B)
        # Transpose: C @ M^T @ D
        M_t = M.transpose(-2, -1)
        trans = torch.einsum('ij,...jk,kl->...il', self.C, M_t, self.D)
        # Combine with residual
        out = M + fwd + trans + self.bias
        # Normalize (matrix analog of layer norm: normalize Frobenius norm)
        norm = torch.norm(out, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        out = out * (self.scale * math.sqrt(self.mat_dim) / norm)
        return out


class MatrixAttention(nn.Module):
    """Attention where Q, K, V are all matrices.

    Instead of dot product similarity (scalar), we use trace(Q_i^T @ K_j)
    which measures the Frobenius inner product between two matrices.
    This captures "in what structural ways are these two tokens related"
    rather than just "how similar are they."

    Value aggregation: sum of V_j matrices weighted by attention scores.
    """

    def __init__(self, mat_dim: int, n_heads: int = 4):
        super().__init__()
        self.mat_dim = mat_dim
        self.n_heads = n_heads
        assert mat_dim % n_heads == 0
        self.head_dim = mat_dim // n_heads

        # Projections: matrix → matrix (learned bilinear maps)
        self.q_proj = MatrixInteraction(mat_dim)
        self.k_proj = MatrixInteraction(mat_dim)
        self.v_proj = MatrixInteraction(mat_dim)
        self.out_proj = MatrixInteraction(mat_dim)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: (B, L, d, d) matrix representations
        Returns:
            M_out: (B, L, d, d) attended matrix representations
        """
        B, L, d, _ = M.shape
        Q = self.q_proj(M)  # (B, L, d, d)
        K = self.k_proj(M)
        V = self.v_proj(M)

        # Attention scores via Frobenius inner product: trace(Q_i^T @ K_j)
        # For each pair (i, j), compute trace(Q[i]^T @ K[j])
        # = sum of element-wise Q[i] * K[j]
        # Reshape for multi-head: split the d×d matrix into n_heads blocks
        # Each head attends over a (head_dim × d) slice of the matrix

        # Simple single-head for now: Frobenius inner product
        # Q_flat: (B, L, d*d), K_flat: (B, L, d*d)
        Q_flat = Q.reshape(B, L, d * d)
        K_flat = K.reshape(B, L, d * d)

        # Attention: (B, L, L)
        scores = torch.bmm(Q_flat, K_flat.transpose(1, 2)) / math.sqrt(d * d)

        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=M.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)

        # Aggregate values: weighted sum of V matrices
        V_flat = V.reshape(B, L, d * d)  # (B, L, d*d)
        out_flat = torch.bmm(attn, V_flat)  # (B, L, d*d)
        out = out_flat.reshape(B, L, d, d)

        return self.out_proj(out)


class MatrixThinkingBlock(nn.Module):
    """One layer of matrix thinking: attention + matrix interaction."""

    def __init__(self, mat_dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = MatrixAttention(mat_dim, n_heads)
        self.think = MatrixInteraction(mat_dim)
        self.act = nn.GELU()

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        # Attention (inter-token matrix interaction)
        M = M + self.attn(M)
        # Thinking (intra-token matrix transformation)
        # Apply interaction then GELU element-wise
        residual = M
        M = self.think(M)
        M = residual + self.act(M)
        return M


class MatrixThinkingModel(nn.Module):
    """
    Complete model: bytes → matrices → think → collapse → predict.

    The key measurement: rank of each token's matrix at each layer.
    """

    def __init__(self, mat_dim: int = 16, n_layers: int = 4, n_heads: int = 4,
                 max_len: int = 512, vocab_size: int = 256):
        super().__init__()
        self.mat_dim = mat_dim
        self.n_layers = n_layers

        # Embed bytes as rank-1 matrices
        self.embed = MatrixEmbedding(vocab_size, mat_dim)

        # Positional embedding (also a matrix per position)
        self.pos_embed_u = nn.Embedding(max_len, mat_dim)
        self.pos_embed_v = nn.Embedding(max_len, mat_dim)

        # Thinking layers
        self.layers = nn.ModuleList([
            MatrixThinkingBlock(mat_dim, n_heads)
            for _ in range(n_layers)
        ])

        # Collapse: matrix → vector via learned projection
        # Take the matrix M (d×d), multiply by a learned vector to get d-dim vector
        self.collapse = nn.Linear(mat_dim * mat_dim, vocab_size)

    def forward(self, byte_ids: torch.Tensor, return_ranks: bool = False):
        """
        Args:
            byte_ids: (B, L) byte values
            return_ranks: if True, also return per-layer rank measurements
        Returns:
            logits: (B, L, 256) next-byte prediction
            ranks: (n_layers+1, B, L) effective rank at each layer (if return_ranks)
        """
        B, L = byte_ids.shape
        device = byte_ids.device

        # Embed as matrices
        M = self.embed(byte_ids)  # (B, L, d, d)

        # Add positional matrix
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_u = self.pos_embed_u(positions)  # (B, L, d)
        pos_v = self.pos_embed_v(positions)  # (B, L, d)
        pos_M = torch.einsum('...i,...j->...ij', pos_u, pos_v)
        M = M + pos_M * 0.1  # Small positional signal

        ranks = []
        if return_ranks:
            ranks.append(self._measure_ranks(M))

        # Think
        for layer in self.layers:
            M = layer(M)
            if return_ranks:
                ranks.append(self._measure_ranks(M))

        # Collapse to vector and predict
        M_flat = M.reshape(B, L, self.mat_dim * self.mat_dim)
        logits = self.collapse(M_flat)  # (B, L, 256)

        if return_ranks:
            return logits, torch.stack(ranks)  # (n_layers+1, B, L)
        return logits

    def _measure_ranks(self, M: torch.Tensor) -> torch.Tensor:
        """Measure effective rank of each token's matrix.

        Effective rank = exp(entropy of singular values).
        Rank 1 = all energy in one singular value.
        Rank d = energy spread equally across all singular values.
        """
        B, L, d, _ = M.shape
        # Compute singular values (on CPU to avoid MPS limitations)
        M_cpu = M.detach().cpu().reshape(B * L, d, d)
        try:
            S = torch.linalg.svdvals(M_cpu)  # (B*L, d)
        except:
            return torch.ones(B, L)

        # Normalize singular values to form a probability distribution
        S = S.clamp(min=1e-10)
        S_norm = S / S.sum(dim=-1, keepdim=True)

        # Effective rank = exp(entropy)
        entropy = -(S_norm * S_norm.log()).sum(dim=-1)
        eff_rank = entropy.exp().reshape(B, L)
        return eff_rank

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
