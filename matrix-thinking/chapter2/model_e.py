"""model_e -- the compositional (multi-hop) readout for Task E, built on top of
Task D's UNMODIFIED `BindingEncoder` (see NEXT_EXPERIMENT_DESIGN.md section 4).

    (K one-hop bindings) --BindingEncoder (model_v4, verbatim)--> Z in R^{d x d}
                                                                      |
                                                          compose() -- literal
                                                          iterated matmul, h steps
                                                                      |
                                                             pred = Z^h @ key_a

`BindingEncoder` never sees h or the query -- it is reused verbatim, not
reimplemented, so this stays a clean extension of Task D rather than a
confound. The ONLY new learnable thing in `MatrixCompositionModel` relative to
Task D is nothing: `compose()` is a PURE function of (Z, query_keys, hops),
with zero learned parameters (no per-hop weights, no h-conditioned pathway) --
verified structurally in run_task_e.py's smoke gate (C_composition-purity).

Also here: `MLPShortcutModel`, the C_MLP baseline -- an UNCONSTRAINED,
rank-blind-by-construction shortcut readout (`KILL_LIST.md` Lesson 1) used
only as an honest floor for hop-extrapolation, never as the headline model.

Self-contained apart from model_v4 (BindingEncoder) and rank_utils
(truncate_to_rank), both same-directory. No src/ imports.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model_v4 import BindingEncoder
from rank_utils import truncate_to_rank


class MatrixCompositionModel(nn.Module):
    """Task D's BindingEncoder + a pinned iterated-matmul multi-hop readout.

    `encode()` mirrors MatrixMemoryModel.encode (model_v4.py) exactly,
    including the force_rank_k train-time bottleneck (C1): Z is projected to
    rank k ONCE, and `compose()` then iterates that SAME rank-k-projected
    matrix h times -- testing whether a rank sufficient for one-hop recall
    stays sufficient under repeated self-application (M4_E), not re-truncating
    at every hop.
    """

    def __init__(self, d: int, h: int = 64, n_layers: int = 3, n_heads: int = 4,
                 n_refine: int = 1):
        super().__init__()
        self.d = d
        self.encoder = BindingEncoder(d, h, n_layers, n_heads, n_refine)

    def encode(self, keys, values, force_rank_k: int | None = None) -> torch.Tensor:
        Z = self.encoder(keys, values)                # (B, d, d)
        if force_rank_k is not None and force_rank_k > 0:
            Z = truncate_to_rank(Z, force_rank_k)      # projected ONCE; compose() reuses this Z
        return Z

    @staticmethod
    def compose(Z: torch.Tensor, query_keys: torch.Tensor, hops: torch.Tensor) -> torch.Tensor:
        """Pinned multi-hop readout: pred[b,q] = Z[b]^{hops[b,q]} @ query_keys[b,q].

        Literal iterated matrix-vector product -- NOT re-attention over raw
        inputs, no learned per-hop parameter, no h-conditioned pathway into Z
        (C_composition-purity, NEXT_EXPERIMENT_DESIGN.md section 5). The
        signature is intentionally restricted to (Z, query_keys, hops): the
        run_task_e.py smoke gate asserts this via `inspect.signature` (mirrors
        Task D's `unbind(Z, query_keys)` signature pin).

        Z: (B, d, d); query_keys: (B, Q, d); hops: (B, Q) int64, hops >= 0
        (h=0 is the identity, permitted for eval-time sanity but not part of
        the training distribution). Loops for exactly `max(hops)` steps and
        selects each query's own hop count via masking -- so every query gets
        EXACTLY its own h sequential matmuls, no more, no fewer, and no cross-
        query leakage (verified in the smoke gate's h-purity check).
        """
        cur = query_keys
        result = torch.where(hops.unsqueeze(-1) == 0, cur, torch.zeros_like(cur))
        max_h = int(hops.max().item()) if hops.numel() > 0 else 0
        for t in range(1, max_h + 1):
            cur = torch.einsum("bij,bqj->bqi", Z, cur)     # exactly one matmul per step, all queries
            mask = (hops == t).unsqueeze(-1)
            result = torch.where(mask, cur, result)
        return result

    def forward(self, batch: dict, force_rank_k: int | None = None):
        Z = self.encode(batch["keys"], batch["values"], force_rank_k=force_rank_k)
        pred = self.compose(Z, batch["query_keys"], batch["hops"])
        return pred, Z


# ---------------------------------------------------------------------------
# C_MLP -- unconstrained MLP-flatten shortcut baseline (NOT the headline model)
# ---------------------------------------------------------------------------
# Establishes the honest floor for hop-extrapolation: if THIS architecture
# (which throws away rank structure via flatten(Z) and gets an explicit h
# signal via one-hot(h)) could shortcut zero-shot generalization to held-out
# hops, H_E's positive result on the pinned-composition model would be
# uninterpretable. At held-out h, one-hot(h) is OUT-OF-VOCABULARY (all-zero --
# the embedding only has columns for h in H_train), so the model has no
# explicit hop signal there; whether it still shortcuts is an empirical
# question this control answers, not assumed.

class MLPShortcutModel(nn.Module):
    """pred(a, h) = MLP([flatten(Z), key_a, one_hot(h)]).

    Uses the SAME BindingEncoder architecture/capacity as the matrix arm (for
    a fair comparison of the READOUT, not the encoder), but flattens Z (rank-
    blind by construction, KILL_LIST.md Lesson 1) and feeds an explicit
    h-signal to an unconstrained MLP. `force_rank_k` is accepted (for a
    uniform calling convention with MatrixCompositionModel) but is a no-op:
    C_MLP is rank-UNCONSTRAINED by design.
    """

    def __init__(self, d: int, h: int = 64, n_layers: int = 3, n_heads: int = 4,
                 n_refine: int = 1, h_train_max: int = 3, mlp_hidden: int = 256,
                 h_train: tuple[int, ...] | None = None):
        super().__init__()
        self.d = d
        self.h_train_max = h_train_max
        # Defense-in-depth (gauntlet/AUDIT_task_e_correctness.md MINOR item 2):
        # if the caller supplies the ACTUAL H_train set, `_one_hot_h` checks
        # exact set MEMBERSHIP instead of the range [1, h_train_max]. A range
        # check silently mislabels a non-contiguous, skipped, untrained hop
        # (e.g. h=2 when H_train={1,3}) as in-vocabulary. When h_train is
        # omitted (back-compat), falls back to the old range check -- correct
        # only for a contiguous H_train, which run_task_e.py's CLI already
        # enforces for --model mlp as a first line of defense.
        self.h_train_set = frozenset(h_train) if h_train is not None else None
        if h_train is not None:
            assert max(h_train) == h_train_max, \
                f"h_train_max={h_train_max} must equal max(h_train)={max(h_train)}"
        self.encoder = BindingEncoder(d, h, n_layers, n_heads, n_refine)
        in_dim = d * d + d + h_train_max     # flatten(Z) + key_a + one-hot(h)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, d),
        )

    def _one_hot_h(self, hops: torch.Tensor) -> torch.Tensor:
        """hops: (B, Q) int64. Values in H_train are in-vocabulary; anything
        else (in particular every held-out h > h_train_max, AND any skipped
        hop inside [1, h_train_max] for a non-contiguous H_train) maps to the
        all-zero OUT-OF-VOCABULARY vector -- the deliberate, honest
        floor-closing property this control exists for."""
        B, Q = hops.shape
        oh = torch.zeros(B, Q, self.h_train_max, device=hops.device, dtype=torch.float32)
        if self.h_train_set is not None:
            # Exact SET membership -- correct for a non-contiguous H_train
            # (e.g. {1, 3}), unlike a range check (see __init__ docstring).
            in_vocab = torch.zeros_like(hops, dtype=torch.bool)
            for v in self.h_train_set:
                in_vocab |= (hops == v)
        else:
            in_vocab = (hops >= 1) & (hops <= self.h_train_max)
        idx = (hops - 1).clamp(min=0, max=self.h_train_max - 1)
        oh.scatter_(-1, idx.unsqueeze(-1), in_vocab.to(torch.float32).unsqueeze(-1))
        return oh

    def encode(self, keys, values, force_rank_k: int | None = None) -> torch.Tensor:
        return self.encoder(keys, values)     # force_rank_k is a no-op here (see class docstring)

    def forward(self, batch: dict, force_rank_k: int | None = None):
        Z = self.encode(batch["keys"], batch["values"])
        B, Q, d = batch["query_keys"].shape
        zflat = Z.reshape(B, 1, d * d).expand(B, Q, d * d)
        oh = self._one_hot_h(batch["hops"])
        feat = torch.cat([zflat, batch["query_keys"], oh], dim=-1)
        pred = self.mlp(feat)
        return pred, Z
