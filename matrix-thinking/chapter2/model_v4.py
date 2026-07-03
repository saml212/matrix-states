"""v4 — matrix-native associative-memory model for Task D.

Architecture (see TASK_D_PREREGISTRATION.md §4):

    (K bindings) --encoder--> Z in R^{d x d}  --unbind--> predicted value
                              ^^^^^^^^^^^^^^^^
                              the ONLY thing the decoder sees (hard bottleneck)

The decoder is a PURE FUNCTION of (Z, query_key): `pred = Z @ key`. It never
touches the raw binding tokens — the bottleneck is enforced by construction (the
decode function's signature), and verified at runtime by the blank-out test in
run_task_d.py. The readout is the pinned linear unbind (NOT an MLP, NOT argmax) —
required for the rank(Z) >= K necessity (Nichani et al. 2412.06538).

Self-contained apart from rank_utils (same directory). No src/ imports.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rank_utils import truncate_to_rank


class BindingEncoder(nn.Module):
    """Set of K (key, value) pairs -> a single d x d matrix Z.

    Permutation-invariant over bindings (a set encoder), and able to express any
    Z of rank up to d (via d independent learned row-reader queries) -- it does
    NOT hardcode the Sigma v_j k_j^T outer-product solution, so "does SGD discover
    rank ~ K" is a genuine question, not a built-in answer.
    """

    def __init__(self, d: int, h: int = 64, n_layers: int = 3, n_heads: int = 4,
                 n_refine: int = 1):
        super().__init__()
        self.d = d
        self.h = h
        self.n_refine = n_refine
        # One token per binding, carrying [key ; value].
        self.in_proj = nn.Linear(2 * d, h)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=4 * h,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        # d learned "row-reader" latents; each reads one row of Z from the bindings.
        self.row_queries = nn.Parameter(torch.randn(d, h) * 0.02)
        self.reader = nn.MultiheadAttention(h, n_heads, batch_first=True, dropout=0.0)
        self.row_norm = nn.LayerNorm(h)
        self.row_out = nn.Linear(h, d)   # h -> a d-vector row of Z

    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # keys, values: (B, K, d)
        B = keys.shape[0]
        tok = self.in_proj(torch.cat([keys, values], dim=-1))       # (B, K, h)
        mem = self.encoder(tok)                                     # (B, K, h)
        q = self.row_queries.unsqueeze(0).expand(B, self.d, self.h)  # (B, d, h)
        for _ in range(self.n_refine):
            read, _ = self.reader(q, mem, mem, need_weights=False)   # (B, d, h)
            q = self.row_norm(q + read)                              # residual refine
        Z = self.row_out(q)                                         # (B, d, d): rows of Z
        return Z


class MatrixMemoryModel(nn.Module):
    """Encoder -> Z (optional rank forcing) -> linear unbind readout."""

    def __init__(self, d: int, h: int = 64, n_layers: int = 3, n_heads: int = 4,
                 n_refine: int = 1):
        super().__init__()
        self.d = d
        self.encoder = BindingEncoder(d, h, n_layers, n_heads, n_refine)

    def encode(self, keys, values, force_rank_k: int | None = None) -> torch.Tensor:
        Z = self.encoder(keys, values)               # (B, d, d)
        if force_rank_k is not None and force_rank_k > 0:
            Z = truncate_to_rank(Z, force_rank_k)     # C1: train-time rank constraint
        return Z

    @staticmethod
    def unbind(Z: torch.Tensor, query_keys: torch.Tensor) -> torch.Tensor:
        """Pinned linear unbind. PURE function of (Z, query_keys) -- the decoder
        cannot see the raw bindings. pred[b,q,:] = Z[b] @ query_keys[b,q,:].
        Z: (B, d, d); query_keys: (B, Q, d) -> pred: (B, Q, d).
        """
        return torch.einsum("bij,bqj->bqi", Z, query_keys)

    def forward(self, batch: dict, force_rank_k: int | None = None):
        Z = self.encode(batch["keys"], batch["values"], force_rank_k=force_rank_k)
        pred = self.unbind(Z, batch["query_keys"])
        return pred, Z


# ---------------------------------------------------------------------------
# Vector-state baseline (C2). NOT PART OF THE STAGE-1 GATE.
# ---------------------------------------------------------------------------
# Round-1 audit (AUDIT_validity.md) verdict: this is NOT param/state-matched --
# ~4.3K params & a 16-dim state vs. the matrix encoder's ~171K params & d^2=256-dim
# state. As-is, M4 is uninterpretable, so it is DESCOPED from the Stage-1 gate.
# The Stage-1 go/no-go is M1+M2+M3 on the matrix model only.
#
# A proper C2 control (param-matched, d^2-dim state, reshape-INequivalent to the
# matrix so a win can't be "just capacity") is a Stage-1b task with its own design
# + audit. HRRVectorMemory is retained only for development/smoke; do NOT report
# it as the C2 baseline.

class HRRVectorMemory(nn.Module):
    """Holographic Reduced Representation associative memory over a d-vector state.

    Bundle:  s = sum_j  circconv(enc_k(k_j), enc_v(v_j))
    Unbind:  pred_q = circconv(involution(enc_k(k_q)), s)   (approx v_q)
    enc_k/enc_v are learned per-vector maps so the model is trained (matched-ish
    capacity to the matrix encoder), but binding itself is the fixed HRR operator.
    """

    def __init__(self, d: int, h: int = 64):
        super().__init__()
        self.d = d
        self.enc_k = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        self.enc_v = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    @staticmethod
    def _circconv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # circular convolution via FFT along the last dim
        return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1),
                               n=a.shape[-1], dim=-1)

    @staticmethod
    def _involution(a: torch.Tensor) -> torch.Tensor:
        # HRR approximate inverse: reverse all but the first component
        return torch.cat([a[..., :1], a[..., 1:].flip(-1)], dim=-1)

    def forward(self, batch: dict, force_rank_k: int | None = None):
        # force_rank_k is ignored (a vector state has no matrix rank) -- accepted
        # only so the training loop can call matrix and vector models uniformly.
        kk = self.enc_k(batch["keys"])                     # (B, K, d)
        vv = self.enc_v(batch["values"])                   # (B, K, d)
        s = self._circconv(kk, vv).sum(dim=1)              # (B, d) bundled memory
        qk = self.enc_k(batch["query_keys"])               # (B, Q, d)
        pred = self._circconv(self._involution(qk), s.unsqueeze(1))  # (B, Q, d)
        return pred, s
