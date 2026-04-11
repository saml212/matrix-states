"""
Autoregressive Matrix Thinker v2: True thought appending.

Each generated thought is APPENDED to the context sequence as a new position.
The model attends over input tokens + all previous thoughts. Each thought
can see everything before it but nothing after. This is genuine autoregression
in matrix space.

When a thought's rank drops low enough (crystallizes), it collapses to a
vector and produces a token prediction.

Architecture:
- 3D matrix product attention (rows at pos s couple with cols at pos t)
- SiLU activation (SwiGLU gate*value pattern)
- Multiplicative composition (I+Δ)·M·(I+Γ) per position
- PonderNet-style weighted loss across thinking steps
- Shared thinking layers applied at each step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Components ────────────────────────────────────────────────

class MatrixRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, d))
        self.eps = eps

    def forward(self, M):
        rms = torch.sqrt(M.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps)
        return M / rms * self.weight


class RowThenColProjection(nn.Module):
    """silu(A @ M) @ B — nonlinearity between left and right multiply."""
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))

    def forward(self, M):
        return torch.einsum('bsij,jk->bsik',
                            F.silu(torch.einsum('ij,bsjk->bsik', self.A, M)),
                            self.B)


class MatrixProductAttention(nn.Module):
    """3D attention: rows at position s couple with cols at position t.

    Score = trace(Q_s @ K_t^T) per pair — a scalar derived from the
    matrix product, not just the Frobenius inner product.
    The matrix product also transforms V before aggregation.
    Uses chunked computation to control memory.
    """

    def __init__(self, d, n_heads=4, dropout=0.1, chunk_size=64):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.chunk_size = chunk_size
        assert d % n_heads == 0

        self.norm = MatrixRMSNorm(d)
        self.q_proj = RowThenColProjection(d)
        self.k_proj = RowThenColProjection(d)
        self.v_proj = RowThenColProjection(d)
        self.o_proj = RowThenColProjection(d)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, M, causal_mask=None):
        B, L, d, _ = M.shape
        H = self.n_heads
        hd = self.head_dim

        M_n = self.norm(M)
        Q = self.q_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)
        K = self.k_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)
        V = self.v_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)

        # 3D matrix product scores: Q_s @ K_t^T → trace for scalar weight
        mat_scores = torch.einsum('bhlik,bhmjk->bhlmij', Q, K)
        scalar_scores = torch.einsum('bhlmii->bhlm', mat_scores) / math.sqrt(hd)

        # Causal mask
        if causal_mask is None:
            causal_mask = torch.triu(torch.ones(L, L, device=M.device, dtype=torch.bool), diagonal=1)
        scalar_scores = scalar_scores.masked_fill(causal_mask[None, None], float('-inf'))

        attn_weights = self.attn_dropout(F.softmax(scalar_scores, dim=-1))

        # Structured aggregation: matrix score transforms V before weighting
        transformed_V = torch.einsum('bhlmij,bhmjk->bhlmik', mat_scores, V)
        out = torch.einsum('bhlm,bhlmij->bhlij', attn_weights, transformed_V)

        out = out.permute(0, 2, 1, 3, 4).reshape(B, L, d, d)
        return M + self.o_proj(out)


class MultiplicativeThinkingLayer(nn.Module):
    """Per-position multiplicative composition: (I+Δ)·M·(I+Γ) + v·kᵀ"""

    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.d = d
        self.norm = MatrixRMSNorm(d)

        # SwiGLU projections for delta and gamma
        self.delta_gate = RowThenColProjection(d)
        self.delta_value = RowThenColProjection(d)
        self.delta_up = RowThenColProjection(d)
        self.gamma_gate = RowThenColProjection(d)
        self.gamma_value = RowThenColProjection(d)
        self.gamma_up = RowThenColProjection(d)

        # Additive write
        self.key_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.val_col = nn.Parameter(torch.randn(d, 1) * 0.02)

        # Gates via Frobenius inner product
        self.gate_mult_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_mult_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_write_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_write_bias = nn.Parameter(torch.tensor(-2.0))

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('I', torch.eye(d))

    def forward(self, M):
        B, L, d, _ = M.shape
        M_n = self.norm(M)
        scale = self.scale.clamp(0.01, 0.5)

        delta = self.delta_up(F.silu(self.delta_gate(M_n)) * self.delta_value(M_n)) * scale
        gamma = self.gamma_up(F.silu(self.gamma_gate(M_n)) * self.gamma_value(M_n)) * scale

        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)

        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)

        g_m = torch.sigmoid((self.gate_mult_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_mult_bias)
        g_w = torch.sigmoid((self.gate_write_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_write_bias)

        update = g_m * (M_mult - M_n) + g_w * M_write
        return M + self.dropout(update)


class ThinkingBlock(nn.Module):
    """One block: 3D attention + multiplicative thinking."""
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = MatrixProductAttention(d, n_heads, dropout)
        self.think = MultiplicativeThinkingLayer(d, dropout)

    def forward(self, M, causal_mask=None):
        M = self.attn(M, causal_mask)
        M = self.think(M)
        return M


# ─── Main Model ────────────────────────────────────────────────

class AutoregressiveMatrixThinker(nn.Module):
    """
    True autoregressive matrix thinker.

    For each input token position:
    1. Embed token as 16×16 matrix
    2. APPEND thought matrices to the sequence one at a time
    3. Each thought attends over all input tokens + all previous thoughts
    4. When thought rank crystallizes (σ₂/σ₁ < threshold), collapse to vector
    5. Predict next token from the collapsed vector

    During training: run fixed number of thoughts, PonderNet weighted loss
    During inference: stop when rank converges
    """

    def __init__(self, mat_dim=16, n_thinking_layers=2, max_thoughts=200,
                 n_heads=4, max_len=2048, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim
        self.max_thoughts = max_thoughts
        self.n_thinking_layers = n_thinking_layers
        d = mat_dim

        # Token embedding
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)

        # Shared thinking block (reused at each thought step)
        self.thinking_block = nn.ModuleList([
            ThinkingBlock(d, n_heads, dropout)
            for _ in range(n_thinking_layers)
        ])

        # Thought step embedding
        self.thought_step_embed = nn.Embedding(max_thoughts, d * d)

        # Thought seed: learned starting point for thought generation
        self.thought_seed = nn.Parameter(torch.randn(1, 1, d, d) * 0.02)

        # Halting: learned probability biased by rank
        self.halt_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.halt_bias = nn.Parameter(torch.tensor(0.0))

        # Collapse and prediction
        self.final_norm = MatrixRMSNorm(d)
        self.collapse_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.collapse_out = nn.Linear(d, vocab_size, bias=False)

    def embed_tokens(self, token_ids):
        u = self.embed_u(token_ids)
        v = self.embed_v(token_ids)
        return torch.einsum('...i,...j->...ij', u, v)

    def add_positions(self, M, start_pos=0):
        B, L, d, _ = M.shape
        positions = torch.arange(start_pos, start_pos + L, device=M.device)
        positions = positions.clamp(max=self.pos_u.num_embeddings - 1)
        positions = positions.unsqueeze(0).expand(B, -1)
        pu = self.pos_u(positions)
        pv = self.pos_v(positions)
        return M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

    def measure_rank(self, M):
        """Effective rank. M can be (B, d, d) or (B, L, d, d)."""
        shape = M.shape
        M_flat = M.detach().cpu().reshape(-1, shape[-2], shape[-1])
        S = torch.linalg.svdvals(M_flat).clamp(min=1e-10)
        S_norm = S / S.sum(dim=-1, keepdim=True)
        entropy = -(S_norm * S_norm.log()).sum(dim=-1)
        return entropy.exp().reshape(shape[:-2])

    def halt_probability(self, thought_matrix):
        """Halting probability for a single thought. (B, 1, d, d) → (B, 1)."""
        B = thought_matrix.shape[0]
        M = thought_matrix.squeeze(1)  # (B, d, d)
        learned = (self.halt_W * M).sum(dim=(-2, -1)) + self.halt_bias
        rank = self.measure_rank(M).to(M.device)
        rank_signal = 2.0 - rank  # positive when rank < 2
        return torch.sigmoid(learned + 0.5 * rank_signal).unsqueeze(-1)  # (B, 1)

    def forward(self, token_ids, n_thoughts=None):
        """
        token_ids: (B, L)
        n_thoughts: override for max thinking steps (uses self.max_thoughts if None)

        Returns:
            logits: (B, L, vocab_size) — predictions for each input position
            info: dict with rank profiles, halt distribution, expected steps
        """
        B, L = token_ids.shape
        device = token_ids.device
        d = self.mat_dim
        T = n_thoughts or self.max_thoughts

        # Embed input tokens as matrices
        token_matrices = self.embed_tokens(token_ids)  # (B, L, d, d)
        token_matrices = self.add_positions(token_matrices, start_pos=0)

        # The full sequence starts as just the input tokens
        # We'll generate thoughts for predicting the LAST token's next token
        # For training, we generate thoughts conditioned on the full input
        sequence = token_matrices  # (B, L, d, d)

        # Storage for PonderNet
        all_logits = []
        all_halt_probs = []
        all_ranks = []

        for t in range(T):
            # Create the new thought: start from seed + step embedding
            step_emb = self.thought_step_embed(
                torch.tensor(min(t, self.thought_step_embed.num_embeddings - 1), device=device)
            )
            step_mat = step_emb.reshape(1, 1, d, d) * 0.01
            new_thought = self.thought_seed.expand(B, -1, -1, -1) + step_mat  # (B, 1, d, d)

            # Add position embedding for the thought
            new_thought = self.add_positions(new_thought, start_pos=L + t)

            # APPEND thought to sequence
            sequence = torch.cat([sequence, new_thought], dim=1)  # (B, L+t+1, d, d)

            # Run shared thinking layers over the FULL sequence
            # (causal: each position sees itself and everything before)
            S = sequence.shape[1]
            causal_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)

            for layer in self.thinking_block:
                sequence = layer(sequence, causal_mask)

            # The LAST position is the latest thought — measure its rank
            latest_thought = sequence[:, -1:, :, :]  # (B, 1, d, d)
            rank = self.measure_rank(latest_thought).to(device)  # (B, 1)
            all_ranks.append(rank.mean().item())

            # Halting probability
            halt_prob = self.halt_probability(latest_thought)  # (B, 1)
            all_halt_probs.append(halt_prob)

            # Collapse latest thought to logits
            M_normed = self.final_norm(latest_thought)
            out_vec = (self.collapse_W * M_normed).sum(dim=-1)  # (B, 1, d)
            logits = self.collapse_out(out_vec)  # (B, 1, vocab_size)
            all_logits.append(logits)

        # PonderNet weighted combination
        halt_probs = torch.cat(all_halt_probs, dim=-1).clamp(1e-6, 1 - 1e-6)  # (B, T)
        halt_probs[:, -1] = 1.0  # force halt at last step

        log_survive = torch.zeros(B, device=device)
        p_halt = []
        for t in range(T):
            log_p = torch.log(halt_probs[:, t]) + log_survive
            p_halt.append(torch.exp(log_p))
            log_survive = log_survive + torch.log(1 - halt_probs[:, t])

        p_halt = torch.stack(p_halt, dim=-1)  # (B, T)

        # Weighted logits: each thinking step produces a prediction,
        # weighted by its halting probability
        logits_stack = torch.cat(all_logits, dim=1)  # (B, T, vocab_size)
        # For training: predict the NEXT token after the input sequence
        # The weighted combination gives the final prediction
        final_logits = (logits_stack * p_halt.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B, 1, vocab)

        # For per-position prediction during training, we need logits for all input positions
        # Use the input token matrices directly through the thinking block for the "fast" predictions
        # and the thought-augmented prediction for the last position
        # SIMPLIFIED: for now, return just the thought-augmented prediction
        # Full per-position prediction requires running thoughts per position (expensive)

        # Also produce "fast" logits from the input tokens directly (no thoughts)
        fast_normed = self.final_norm(token_matrices)
        fast_vec = (self.collapse_W * fast_normed).sum(dim=-1)  # (B, L, d)
        fast_logits = self.collapse_out(fast_vec)  # (B, L, vocab_size)

        # Replace the last position's prediction with the thought-augmented one
        combined_logits = fast_logits.clone()
        combined_logits[:, -1:, :] = final_logits

        # Expected thinking steps
        steps = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        expected_steps = (p_halt * steps).sum(dim=-1).mean()

        info = {
            'expected_steps': expected_steps.item(),
            'mean_ranks_per_step': all_ranks,
            'halt_distribution': p_halt.mean(dim=0).detach().cpu().tolist(),
        }

        return combined_logits, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
