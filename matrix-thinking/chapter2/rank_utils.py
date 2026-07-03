"""Rank utilities for Task D — copied verbatim (with attribution) from the
battle-tested implementation in ``matrix-thinking/scripts/_smoke_rank_aware.py``
/ ``run_matrix_codi.py``.

Why copied, not imported: Task D scripts are self-contained for pod deployment
(no cross-directory imports) — this avoids the env/import fragility that cost
the 2026-04-29 rank_aware_v1 run 24 of 27 experiments after a pod resume.

These functions were smoke-tested (incl. a degenerate-spectrum NaN-free backward
test, [3,1,1,0]) in the rank_aware_v1 era, BUT they are RE-AUDITED for Task D:
  - Task D uses truncate_to_rank both at train time (force_rank_k) and eval time
    (M2/M3). Gradient flow through eigh under Task D's conditioning must be
    re-checked by the audit gauntlet.
  - effective_rank is Task D's pre-registered PRIMARY rank metric (C3).
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
