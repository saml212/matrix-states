"""Rank utilities -- copied VERBATIM (with attribution) from
``matrix-thinking/chapter2/rank_utils.py``, which is itself copied verbatim
(with attribution) from ``matrix-thinking/scripts/_smoke_rank_aware.py`` /
``run_matrix_codi.py``.

Why copied, not imported: DeltaNet causal-rank scripts are self-contained for
pod deployment (no cross-directory imports) -- same convention chapter2's own
scripts follow (task_d.py, task_e.py, rank_utils.py all self-contained),
established after the 2026-04-29 rank_aware_v1 run lost 24/27 experiments to
an import fragility bug after a pod resume.

DELTANET_CAUSAL_RANK_DESIGN.md section 3.5 mandates reusing
``truncate_to_rank`` verbatim as the single post-BIND-phase truncation
mechanism (the two-kernel-call split's rank-forcing step) and section 6.2
(C15) mandates the post-truncation direct-SVD rank assertion use this same
battle-tested ``eigh(ZZ^T)`` path, not a re-derivation.

DO NOT MODIFY the function bodies -- they are verbatim-identical to the
ones in matrix-thinking/chapter2/rank_utils.py (only THIS module docstring
differs: it replaces chapter2's Task-D-specific header with this
DeltaNet-specific attribution note, so the two FILES are not byte-identical
even though every function body IS). If a fix is ever needed, fix it
upstream in chapter2/rank_utils.py first, per that file's own re-audit
history, then re-copy the function bodies here.
"""
from __future__ import annotations

import torch


def truncate_to_rank(Z: torch.Tensor, k: int) -> torch.Tensor:
    """Project each (d, d) matrix in a batch to its best rank-k approximation.

    Uses eigh(ZZ^T) rather than full SVD: eigh backward is numerically stable
    even when singular values coincide (σ_i = σ_j), avoiding the
    1/(σ_i^2 - σ_j^2) -> inf that full-SVD backward produces at degenerate
    spectra. Verified NaN-free on a constructed [3,1,1,0] spectrum.

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

    Task D's pre-registered PRIMARY rank metric (C3). Returns one value per
    leading batch element: Z of shape (..., d, d) -> (...).
    Range [1, d]; equals the number of "effectively active" singular directions.
    """
    Zf = Z.float()
    sigma = torch.linalg.svdvals(Zf).clamp(min=1e-10)          # (..., d)
    p = sigma / sigma.sum(dim=-1, keepdim=True)
    H = -(p * torch.log(p)).sum(dim=-1)
    return torch.exp(H)


def stable_rank(Z: torch.Tensor) -> torch.Tensor:
    """Stable rank = ||Z||_F^2 / ||Z||_2^2. Task D's pre-registered SECONDARY
    rank metric (C3). Range [1, rank(Z)]; robust to small singular values.
    """
    Zf = Z.float()
    sigma = torch.linalg.svdvals(Zf)                           # (..., d)
    fro_sq = (sigma ** 2).sum(dim=-1)
    spec_sq = (sigma[..., 0] ** 2).clamp(min=1e-20)            # svdvals is descending
    return fro_sq / spec_sq
