"""
Full model: Learned Segmenter + PHM Transformer for next-byte prediction.

V2: Updated segmenter interface returns both hard and soft boundaries.

Architecture:
  1. Raw bytes → LearnedSegmenter → variable-length segments
  2. Segments → PHM Transformer (attention + PHM-MLP) → contextualized
  3. Contextualized → expand back to byte-level → predict next byte (256-way)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .phm import PHMLinear, PHMLP
from .segmenter import LearnedSegmenter


class PHMAttention(nn.Module):
    """Multi-head attention with PHM projections for Q, K, V, and output."""

    def __init__(self, d_model: int, n_heads: int, n: int = 4, dropout: float = 0.1,
                 algebra_mode: str = 'learned'):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = PHMLinear(d_model, d_model, n=n, algebra_mode=algebra_mode)
        self.k_proj = PHMLinear(d_model, d_model, n=n, algebra_mode=algebra_mode)
        self.v_proj = PHMLinear(d_model, d_model, n=n, algebra_mode=algebra_mode)
        self.out_proj = PHMLinear(d_model, d_model, n=n, algebra_mode=algebra_mode)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Causal attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        else:
            causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class PHMTransformerBlock(nn.Module):
    """Pre-norm transformer block with PHM attention and PHM MLP."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n: int = 4, dropout: float = 0.1,
                 algebra_mode: str = 'learned'):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = PHMAttention(d_model, n_heads, n=n, dropout=dropout, algebra_mode=algebra_mode)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = PHMLP(d_model, d_ff, n=n, dropout=dropout, algebra_mode=algebra_mode)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class BytePHMTransformer(nn.Module):
    """
    Full model for next-byte prediction.

    Pipeline:
      bytes → segmenter → segments → PHM transformer → expand to bytes → predict
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 4,
                 d_ff: int = None, phm_n: int = 4, dropout: float = 0.1,
                 max_len: int = 1024, seg_layers: int = 1, seg_d_model: int = None,
                 target_compression: float = 4.0, algebra_mode: str = 'learned'):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        if seg_d_model is None:
            seg_d_model = d_model // 2

        self.d_model = d_model
        self.max_len = max_len

        # Segmenter
        self.segmenter = LearnedSegmenter(
            d_model=d_model, seg_d_model=seg_d_model,
            n_heads=min(4, seg_d_model // 16), n_layers=seg_layers,
            max_len=max_len, dropout=dropout,
            target_compression=target_compression
        )

        # Positional encoding for segments
        self.seg_pos_embed = nn.Embedding(max_len, d_model)

        # PHM Transformer backbone
        self.layers = nn.ModuleList([
            PHMTransformerBlock(d_model, n_heads, d_ff, n=phm_n, dropout=dropout,
                                algebra_mode=algebra_mode)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: predict next byte (256 classes)
        self.output_head = nn.Linear(d_model, 256)

        # For byte-level expansion
        self.byte_embed_for_expand = nn.Embedding(256, d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, byte_ids: torch.Tensor):
        """
        Args:
            byte_ids: (batch, seq_len) byte values [0, 255]
        Returns:
            logits: (batch, seq_len, 256) next-byte prediction logits
            boundary_probs: (batch, seq_len) soft boundary probabilities
            segment_counts: (batch, n_segs) bytes per segment
        """
        B, L = byte_ids.shape
        device = byte_ids.device

        # Step 1: Segment (guaranteed n_segs segments via top-K)
        segment_reprs, boundary_probs, segment_counts, n_segments = self.segmenter(byte_ids)
        n_segs = segment_reprs.shape[1]

        # Add segment positional embeddings
        seg_positions = torch.arange(n_segs, device=device).unsqueeze(0).expand(B, -1)
        seg_positions = seg_positions.clamp(max=self.max_len - 1)
        segment_reprs = segment_reprs + self.seg_pos_embed(seg_positions)

        # Step 2: PHM Transformer over segments (4x shorter than full byte sequence)
        x = segment_reprs
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)  # (B, n_segs, d_model)

        # Step 3: Expand back to byte level using segment assignments
        # Recompute segment assignments from boundary_probs
        # Use top-K hard boundaries for assignment
        scores = self.segmenter.predictor(byte_ids)
        scores[:, 0] = scores[:, 0] + 100.0
        _, top_indices = scores.topk(n_segs, dim=-1)
        top_indices, _ = top_indices.sort(dim=-1)
        boundary_hard = torch.zeros(B, L, device=device)
        boundary_hard.scatter_(1, top_indices, 1.0)
        seg_assignments = (torch.cumsum(boundary_hard, dim=-1).long() - 1).clamp(0, n_segs - 1)

        seg_assignments_expanded = seg_assignments.unsqueeze(-1).expand(-1, -1, self.d_model)
        byte_level_reprs = torch.gather(x, 1, seg_assignments_expanded)

        # Add byte-level embeddings for fine-grained prediction
        byte_embeds = self.byte_embed_for_expand(byte_ids)
        byte_level_reprs = byte_level_reprs + byte_embeds

        # Predict next byte
        logits = self.output_head(byte_level_reprs)

        return logits, boundary_probs, segment_counts

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
