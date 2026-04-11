"""
Ablation models for isolating the contributions of:
1. Segmentation (learned vs fixed-stride vs none)
2. PHM layers (PHM vs standard nn.Linear)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .phm import PHMLinear, PHMLP


class FixedStrideSegmenter(nn.Module):
    """Fixed-stride segmenter — no learning, just chop every K bytes.

    This ablation isolates: does the LEARNING of boundaries matter,
    or is it just having shorter attention context that helps?
    """

    def __init__(self, d_model: int, stride: int = 4, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.stride = stride
        self.byte_embed = nn.Embedding(256, d_model)

    def set_tau(self, tau): pass  # no-op for compatibility

    def forward(self, byte_ids, hard=False):
        B, L = byte_ids.shape
        device = byte_ids.device
        n_segs = L // self.stride

        byte_reprs = self.byte_embed(byte_ids)  # (B, L, d_model)

        # Fixed-stride pooling: average every `stride` bytes
        # Reshape to (B, n_segs, stride, d_model) then mean over stride dim
        trimmed = byte_reprs[:, :n_segs * self.stride]
        segment_reprs = trimmed.reshape(B, n_segs, self.stride, self.d_model).mean(dim=2)

        # Dummy boundary probs (uniform — no learning)
        boundary_probs = torch.zeros(B, L, device=device)
        boundary_probs[:, ::self.stride] = 1.0

        # Segment counts (all equal to stride)
        segment_counts = torch.full((B, n_segs), float(self.stride), device=device)
        n_segments = torch.full((B,), n_segs, device=device, dtype=torch.long)

        return segment_reprs, boundary_probs, segment_counts, n_segments

    def boundary_variance_loss(self, segment_counts):
        return torch.tensor(0.0, device=segment_counts.device)

    def boundary_entropy_loss(self, boundary_probs):
        return torch.tensor(0.0, device=boundary_probs.device)


class StandardAttention(nn.Module):
    """Standard multi-head attention with nn.Linear (no PHM).

    This ablation isolates: do PHM layers help, or would standard linear be the same?
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        else:
            causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class StandardMLP(nn.Module):
    """Standard two-layer MLP with nn.Linear (no PHM)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class StandardTransformerBlock(nn.Module):
    """Pre-norm transformer block with standard nn.Linear (no PHM)."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = StandardMLP(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class AblationModel(nn.Module):
    """Configurable model for ablation studies.

    Args:
        use_phm: If True, use PHM layers. If False, use standard nn.Linear.
        segmenter_type: 'learned' (top-K), 'fixed' (fixed-stride), or 'none' (no segmentation).
    """

    def __init__(self, d_model=640, n_heads=10, n_layers=6, d_ff=2560,
                 phm_n=4, dropout=0.1, max_len=4096, seg_layers=1,
                 seg_d_model=320, target_compression=4.0,
                 use_phm=True, segmenter_type='learned'):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.segmenter_type = segmenter_type
        self.use_phm = use_phm
        self.target_compression = target_compression

        # Segmenter
        if segmenter_type == 'learned':
            from .segmenter import LearnedSegmenter
            self.segmenter = LearnedSegmenter(
                d_model=d_model, seg_d_model=seg_d_model,
                n_heads=min(4, seg_d_model // 16), n_layers=seg_layers,
                max_len=max_len, dropout=dropout,
                target_compression=target_compression
            )
        elif segmenter_type == 'fixed':
            self.segmenter = FixedStrideSegmenter(
                d_model=d_model, stride=int(target_compression), max_len=max_len
            )
        else:  # 'none' — process raw bytes
            self.segmenter = None
            self.byte_embed = nn.Embedding(256, d_model)

        # Positional encoding
        self.pos_embed = nn.Embedding(max_len, d_model)

        # Transformer backbone
        if use_phm:
            from .model import PHMTransformerBlock
            self.layers = nn.ModuleList([
                PHMTransformerBlock(d_model, n_heads, d_ff, n=phm_n, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                StandardTransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 256)

        if segmenter_type != 'none':
            self.byte_embed_for_expand = nn.Embedding(256, d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, byte_ids):
        B, L = byte_ids.shape
        device = byte_ids.device

        if self.segmenter is not None:
            # Segmented path
            segment_reprs, boundary_probs, segment_counts, n_segments = self.segmenter(byte_ids)
            n_segs = segment_reprs.shape[1]

            seg_positions = torch.arange(n_segs, device=device).unsqueeze(0).expand(B, -1)
            seg_positions = seg_positions.clamp(max=self.max_len - 1)
            segment_reprs = segment_reprs + self.pos_embed(seg_positions)

            x = segment_reprs
            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)

            # Expand back to byte level
            if self.segmenter_type == 'learned':
                scores = self.segmenter.predictor(byte_ids)
                scores[:, 0] = scores[:, 0] + 100.0
                _, top_indices = scores.topk(n_segs, dim=-1)
                top_indices, _ = top_indices.sort(dim=-1)
                boundary_hard = torch.zeros(B, L, device=device)
                boundary_hard.scatter_(1, top_indices, 1.0)
                seg_assignments = (torch.cumsum(boundary_hard, dim=-1).long() - 1).clamp(0, n_segs - 1)
            else:
                # Fixed stride
                stride = int(self.target_compression)
                seg_assignments = torch.arange(L, device=device).unsqueeze(0).expand(B, -1) // stride
                seg_assignments = seg_assignments.clamp(max=n_segs - 1)

            seg_exp = seg_assignments.unsqueeze(-1).expand(-1, -1, self.d_model)
            byte_level = torch.gather(x, 1, seg_exp)
            byte_level = byte_level + self.byte_embed_for_expand(byte_ids)
            logits = self.output_head(byte_level)

            return logits, boundary_probs, segment_counts
        else:
            # No segmentation — raw byte transformer
            x = self.byte_embed(byte_ids)
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embed(positions)

            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)
            logits = self.output_head(x)

            # Return dummy boundary info for compatibility
            dummy_bp = torch.zeros(B, L, device=device)
            dummy_sc = torch.ones(B, 1, device=device) * L
            return logits, dummy_bp, dummy_sc

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
