"""CAPABILITY_SEPARATION_DESIGN.md S1.4 -- GroupWordEncoder: BindingEncoder
(matrix-thinking/chapter2/model_v4.py) extended by the THREE disclosed
architectural deltas (S1.4, CA1-m1/CA2-M2 fixes), and nothing else:

  (1) a learned absolute positional embedding (order-sensitivity: group
      composition is non-abelian, and BindingEncoder is otherwise
      permutation-EQUIVARIANT by construction -- correct for Task D's
      order-independent set-of-bindings task, silently wrong here).
  (2) a single generator-index input embedding replacing Task D's [key;
      value] concatenation (no separate value to bind -- the WORD is the
      whole composition).
  (3) per-batch-fixed-L batching (S1.4's pinned scheme: one L per batch,
      shared by every episode in that batch; L varies across batches, not
      within one) -- reused verbatim by group_task.py's sampler, and
      requires NO new padding/mask code in this file (fixed-shape (B,L,h)
      tensors only).

Same core module stack as BindingEncoder (nn.TransformerEncoder + learned
row_queries + MultiheadAttention reader + row_norm + row_out), same
n_layers/n_heads/n_refine defaults, EXTENDED not replaced (S1.4's "same
core module stack, EXTENDED by the three deltas" wording, CA3-m5 fix).

The P=1 hard bottleneck (CLAUDE.md hard rule) holds by construction: there
is no `unbind`/query step in this task (S1.4 -- "no separate query/unbind
step... the whole word maps to one element"), so the ONLY channel from raw
input to any downstream scoring function is `Z = encoder(...)`. The
blank-out test (blank_out.py) verifies this at the embedding layer, the
first CONTINUOUS tensor derived from the discrete generator-index tokens
(gradients cannot be defined w.r.t. integer LongTensor indices directly --
see blank_out.py's module docstring for the exact, disclosed adaptation of
run_task_d.py::smoke() step [5]'s method to this discrete-token task).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chapter2"))
from rank_utils import truncate_to_rank


class GroupWordEncoder(nn.Module):
    """Word of generator indices (B, L) -> a single d_state x d_state matrix Z.

    NOT permutation-invariant over the word (delta 1, positional embedding) --
    deliberately, since group composition is order-sensitive. Otherwise the
    same row-reader-query architecture as BindingEncoder: d_state learned
    "row-reader" latents each read one row of Z from the encoded word.
    """

    def __init__(self, d_state: int, n_gens: int, L_max: int, h: int = 32,
                 n_layers: int = 3, n_heads: int = 4, n_refine: int = 1):
        super().__init__()
        self.d_state = d_state
        self.n_gens = n_gens
        self.L_max = L_max
        self.h = h
        self.n_refine = n_refine
        # Delta (2): single generator-index embedding, replaces Task D's
        # [key; value] concat -- there is no separate value here.
        self.tok_embed = nn.Embedding(n_gens, h)
        # Delta (1): learned absolute positional embedding, tags word
        # position 1..L (order-sensitivity; BindingEncoder has no analog).
        self.pos_embed = nn.Embedding(L_max, h)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=4 * h,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        # d_state learned "row-reader" latents; each reads one row of Z.
        self.row_queries = nn.Parameter(torch.randn(d_state, h) * 0.02)
        self.reader = nn.MultiheadAttention(h, n_heads, batch_first=True, dropout=0.0)
        self.row_norm = nn.LayerNorm(h)
        self.row_out = nn.Linear(h, d_state)   # h -> a d_state-vector row of Z

    def embed_tokens(self, token_idx: torch.Tensor) -> torch.Tensor:
        """token_idx: (B, L) long generator indices -> (B, L, h) continuous
        embedding (token + position). This is the FIRST continuous tensor
        derived from the raw discrete word -- the blank-out test's leaf
        (blank_out.py) is taken here, not at token_idx itself (integer
        tensors cannot carry gradients)."""
        B, L = token_idx.shape
        assert L <= self.L_max, f"word length {L} exceeds L_max={self.L_max}"
        pos = torch.arange(L, device=token_idx.device).unsqueeze(0).expand(B, L)
        return self.tok_embed(token_idx) + self.pos_embed(pos)

    def encode_from_embedding(self, tok_embed: torch.Tensor) -> torch.Tensor:
        """tok_embed: (B, L, h) -> Z: (B, d_state, d_state). Factored out of
        forward() so the blank-out test can hand this function a DETACHED
        leaf tensor directly (mirrors run_task_d.py smoke() step [5]'s
        Z-as-independent-leaf construction)."""
        B = tok_embed.shape[0]
        mem = self.encoder(tok_embed)                                   # (B, L, h)
        q = self.row_queries.unsqueeze(0).expand(B, self.d_state, self.h)  # (B, d_state, h)
        for _ in range(self.n_refine):
            read, _ = self.reader(q, mem, mem, need_weights=False)       # (B, d_state, h)
            q = self.row_norm(q + read)                                   # residual refine
        Z = self.row_out(q)                                              # (B, d_state, d_state)
        return Z

    def forward(self, token_idx: torch.Tensor) -> torch.Tensor:
        return self.encode_from_embedding(self.embed_tokens(token_idx))


class GroupWordModel(nn.Module):
    """Encoder -> Z (optional rank forcing). No unbind/query step (S1.4:
    "there is nothing to query here, the whole word maps to one element")."""

    def __init__(self, d_state: int, n_gens: int, L_max: int, h: int = 32,
                 n_layers: int = 3, n_heads: int = 4, n_refine: int = 1):
        super().__init__()
        self.d_state = d_state
        self.encoder = GroupWordEncoder(d_state, n_gens, L_max, h, n_layers, n_heads, n_refine)

    def encode(self, token_idx: torch.Tensor, force_rank_k: int | None = None) -> torch.Tensor:
        Z = self.encoder(token_idx)
        if force_rank_k is not None and force_rank_k > 0:
            Z = truncate_to_rank(Z, force_rank_k)     # C1: train-time rank constraint
        return Z

    def forward(self, batch: dict, force_rank_k: int | None = None):
        Z = self.encode(batch["token_idx"], force_rank_k=force_rank_k)
        return Z, batch["target"]


def cosine_loss(Z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Task D's exact loss primitive (task_d.py's recovery_cosine, applied
    to FLATTENED matrices rather than vectors, S1.4): 1 - mean cosine
    similarity between flattened Z and flattened block-embedded target."""
    zf = Z.flatten(start_dim=-2)
    tf = target.flatten(start_dim=-2)
    cos = (zf * tf).sum(-1) / (zf.norm(dim=-1).clamp(min=1e-8) * tf.norm(dim=-1).clamp(min=1e-8))
    return (1.0 - cos).mean()


def recovery_cosine(Z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-episode cosine similarity between flattened Z and flattened
    target (not reduced) -- the success-metric primitive (tau=0.9)."""
    zf = Z.flatten(start_dim=-2)
    tf = target.flatten(start_dim=-2)
    return (zf * tf).sum(-1) / (zf.norm(dim=-1).clamp(min=1e-8) * tf.norm(dim=-1).clamp(min=1e-8))


# ---------------------------------------------------------------------------
# Smoke: forward/backward at L=1 and L=16 (the pinned distribution's two
# extremes, S1.4's delta-3 smoke item), plus force_rank_k train-time path.
# ---------------------------------------------------------------------------

def smoke(device="cpu"):
    print("=" * 60)
    print("  GroupWordEncoder SMOKE (L=1 / L=16 forward+backward, force_rank_k)")
    print("=" * 60)
    torch.manual_seed(0)
    d_state, n_gens, L_max = 5, 4, 16     # e.g. an S4/A5-shaped instance (d_min=3, d_state=5)
    model = GroupWordModel(d_state, n_gens, L_max, h=32).to(device)

    for L in (1, 16):
        token_idx = torch.randint(0, n_gens, (8, L), device=device)
        target = torch.randn(8, d_state, d_state, device=device)
        Z = model.encode(token_idx)
        assert Z.shape == (8, d_state, d_state), Z.shape
        assert not torch.isnan(Z).any() and not torch.isinf(Z).any()
        loss = cosine_loss(Z, target)
        model.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"L={L}: no grad for {name}"
            assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), \
                f"L={L}: bad grad for {name}"
        print(f"  L={L:>2}: forward {tuple(Z.shape)}, loss={loss.item():.4f}, "
              f"all params finite grad  OK")

    print("\n  force_rank_k train-time path (C1):")
    token_idx = torch.randint(0, n_gens, (8, 4), device=device)
    target = torch.randn(8, d_state, d_state, device=device)
    model.zero_grad()
    Zk = model.encode(token_idx, force_rank_k=2)
    from rank_utils import effective_rank
    er = effective_rank(Zk).mean().item()
    assert er <= 2 + 1e-2, f"force_rank_k=2 didn't constrain rank (eff rank {er:.3f})"
    cosine_loss(Zk, target).backward()
    for p in model.parameters():
        assert not (p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()))
    print(f"  force_rank_k=2: eff_rank={er:.3f} <= 2, backprops cleanly  OK")

    print("\n" + "=" * 60 + "\n  GroupWordEncoder SMOKE PASSED\n" + "=" * 60)


if __name__ == "__main__":
    smoke("cuda" if torch.cuda.is_available() else "cpu")
