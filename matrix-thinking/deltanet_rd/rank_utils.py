"""Rank utilities -- copied VERBATIM (with attribution) from
``matrix-thinking/deltanet/rank_utils.py``, which is itself copied verbatim
(with attribution) from ``matrix-thinking/chapter2/rank_utils.py`` /
``matrix-thinking/scripts/_smoke_rank_aware.py``.

Why copied, not imported: pod-safe self-contained convention (see the
upstream copy's own docstring for the 2026-04-29 rank_aware_v1 import-
fragility incident this rule exists to prevent).

DELTANET_REALDATA_DESIGN.md section 5.1 mandates reusing ``truncate_to_rank``
verbatim as the post-BIND-phase rank-forcing mechanism (Wave B's causal arm,
transplanted from DELTANET_CAUSAL_RANK_DESIGN.md section 3.5), and C15's
post-truncation direct-SVD rank assertion uses this same eigh(ZZ^T) path.

DO NOT MODIFY the function bodies -- verbatim-identical to
matrix-thinking/deltanet/rank_utils.py. Fix upstream first, then re-copy.
"""
from __future__ import annotations

import torch


def truncate_to_rank(Z: torch.Tensor, k: int) -> torch.Tensor:
    """Project each (d, d) matrix in a batch to its best rank-k approximation.

    Uses eigh(ZZ^T) rather than full SVD: eigh backward is numerically stable
    even when singular values coincide (sigma_i = sigma_j), avoiding the
    1/(sigma_i^2 - sigma_j^2) -> inf that full-SVD backward produces at
    degenerate spectra.

    Z: (..., d, d).  Returns same shape, rank <= k along the last two dims.
    """
    orig_dtype = Z.dtype
    with torch.autocast(device_type=Z.device.type, enabled=False):
        Zf = Z.float()
        ZZt = Zf @ Zf.transpose(-1, -2)              # (..., d, d) symmetric PSD
        eigvals, eigvecs = torch.linalg.eigh(ZZt)    # ascending; stable backward
        k = min(k, eigvecs.shape[-1])
        U_k = eigvecs[..., -k:]                       # top-k eigenvectors of ZZ^T
        Zk = U_k @ (U_k.transpose(-1, -2) @ Zf)       # project columns onto span(U_k)
    return Zk.to(orig_dtype)


def effective_rank(Z: torch.Tensor) -> torch.Tensor:
    """Effective rank = exp(entropy of the normalized singular-value spectrum).

    Returns one value per leading batch element: Z of shape (..., d, d) ->
    (...). Range [1, d]; equals the number of "effectively active" singular
    directions.
    """
    Zf = Z.float()
    sigma = torch.linalg.svdvals(Zf).clamp(min=1e-10)          # (..., d)
    p = sigma / sigma.sum(dim=-1, keepdim=True)
    H = -(p * torch.log(p)).sum(dim=-1)
    return torch.exp(H)


def stable_rank(Z: torch.Tensor) -> torch.Tensor:
    """Stable rank = ||Z||_F^2 / ||Z||_2^2. Range [1, rank(Z)]; robust to
    small singular values."""
    Zf = Z.float()
    sigma = torch.linalg.svdvals(Zf)                           # (..., d)
    fro_sq = (sigma ** 2).sum(dim=-1)
    spec_sq = (sigma[..., 0] ** 2).clamp(min=1e-20)            # svdvals is descending
    return fro_sq / spec_sq
