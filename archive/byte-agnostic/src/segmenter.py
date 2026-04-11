"""
Learned byte segmenter V3: Forced segmentation with learned boundary placement.

The key insight from MANTa: don't ask the model WHETHER to segment,
force it to produce exactly K segments and let it learn WHERE to place boundaries.

This version uses top-k boundary selection: the segmenter predicts boundary scores
for each position, and we take the top-K scores as boundaries. This guarantees
exactly K segments per sequence, eliminating collapse entirely.

During training, we use a differentiable soft top-k approximation.
During inference, we use hard top-k.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SegmenterTransformer(nn.Module):
    """Small transformer that predicts boundary scores per byte."""

    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 1,
                 d_ff: int = None, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 2

        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.boundary_head = nn.Linear(d_model, 1)

        nn.init.normal_(self.boundary_head.weight, std=0.02)
        nn.init.zeros_(self.boundary_head.bias)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        B, L = byte_ids.shape
        positions = torch.arange(L, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        x = self.byte_embed(byte_ids) + self.pos_embed(positions)
        x = self.encoder(x)
        scores = self.boundary_head(x).squeeze(-1)  # (B, L)
        return scores


class LearnedSegmenter(nn.Module):
    """
    Forced-rate segmentation with learned boundary placement.

    The model MUST produce n_segments segments per sequence.
    It learns WHERE to place boundaries, not WHETHER to segment.

    Uses differentiable soft top-k during training:
    - Compute boundary scores for each position
    - Apply temperature-scaled softmax to get soft boundary weights
    - The top-K positions get the highest weights

    This is inspired by MANTa's architectural bottleneck but simpler.
    """

    def __init__(self, d_model: int, seg_d_model: int = None, n_heads: int = 4,
                 n_layers: int = 1, max_len: int = 2048, dropout: float = 0.1,
                 tau_init: float = 1.0, tau_min: float = 0.1,
                 target_compression: float = 4.0):
        super().__init__()
        if seg_d_model is None:
            seg_d_model = d_model // 2

        self.d_model = d_model
        self.target_compression = target_compression
        self.predictor = SegmenterTransformer(
            d_model=seg_d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, max_len=max_len
        )
        self.byte_proj = nn.Linear(seg_d_model, d_model)
        self.byte_embed_shared = self.predictor.byte_embed

        self.tau = tau_init
        self.tau_min = tau_min

    def set_tau(self, tau: float):
        self.tau = max(tau, self.tau_min)

    def forward(self, byte_ids: torch.Tensor, hard: bool = False):
        """
        Returns:
            segment_reprs: (B, n_segs, d_model) pooled segment representations
            boundary_probs: (B, L) soft boundary probabilities
            segment_counts: (B, n_segs) bytes per segment
            n_segments: (B,) number of segments (constant = L // target_compression)
        """
        B, L = byte_ids.shape
        device = byte_ids.device
        n_segs = max(2, L // int(self.target_compression))

        # Get boundary scores
        scores = self.predictor(byte_ids)  # (B, L)

        # Position 0 is always a boundary — give it a very high score
        scores = scores.clone()
        scores[:, 0] = scores[:, 0] + 100.0

        # Select top-K boundary positions
        if hard or not self.training:
            # Hard: pick top-K positions
            _, top_indices = scores.topk(n_segs, dim=-1)  # (B, n_segs)
            top_indices, _ = top_indices.sort(dim=-1)  # sort by position

            # Create boundary mask
            boundary_hard = torch.zeros(B, L, device=device)
            boundary_hard.scatter_(1, top_indices, 1.0)
            boundary_probs = torch.sigmoid(scores)
        else:
            # Soft: use temperature-scaled softmax as differentiable top-K
            # High scores → high boundary probability
            boundary_probs = torch.sigmoid(scores / self.tau)

            # Hard top-K with straight-through estimator
            _, top_indices = scores.topk(n_segs, dim=-1)
            top_indices, _ = top_indices.sort(dim=-1)

            boundary_hard = torch.zeros(B, L, device=device)
            boundary_hard.scatter_(1, top_indices, 1.0)

            # STE: forward uses hard, backward uses soft sigmoid
            boundary_for_pooling = boundary_hard - boundary_probs.detach() + boundary_probs

        # Create segment assignments from boundaries
        if hard or not self.training:
            seg_ids = torch.cumsum(boundary_hard, dim=-1).long() - 1
        else:
            seg_ids = torch.cumsum(boundary_hard, dim=-1).long() - 1

        seg_ids = seg_ids.clamp(0, n_segs - 1)

        # Get byte representations
        byte_reprs = self.byte_embed_shared(byte_ids)
        byte_reprs = self.byte_proj(byte_reprs)  # (B, L, d_model)

        # Pool bytes into segments (scatter-mean)
        segment_reprs = torch.zeros(B, n_segs, self.d_model, device=device)
        segment_counts = torch.zeros(B, n_segs, device=device)

        seg_ids_expanded = seg_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
        segment_reprs.scatter_add_(1, seg_ids_expanded, byte_reprs)

        ones = torch.ones(B, L, device=device)
        segment_counts.scatter_add_(1, seg_ids, ones)

        # Mean pooling
        mask = segment_counts > 0
        segment_reprs[mask] = segment_reprs[mask] / segment_counts[mask].unsqueeze(-1)

        n_segments_tensor = torch.full((B,), n_segs, device=device, dtype=torch.long)

        return segment_reprs, boundary_probs, segment_counts, n_segments_tensor

    def boundary_variance_loss(self, segment_counts: torch.Tensor) -> torch.Tensor:
        """Encourage roughly equal segment lengths (penalize extreme variance).

        Without this, the model might put all boundaries in one area.
        """
        # Target: each segment has target_compression bytes
        target_len = self.target_compression
        # Penalize deviation from target
        return ((segment_counts - target_len) ** 2).mean()

    def boundary_entropy_loss(self, boundary_probs: torch.Tensor) -> torch.Tensor:
        """Encourage the boundary predictor to be confident (sharp scores)."""
        p = boundary_probs.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        return entropy.mean()
