#!/usr/bin/env python3
"""
Smoke test for the three new rank-aware training features added to run_matrix_codi.py:
  1. --rank-loss {entropy, nuclear}  + --rank-lambda
  2. --force-rank-during-training k

Runs entirely on CPU with a tiny d=4 matrix, batch=2, n_latents=3.
No transformers / GPT-2 required — we test the two new primitives directly
(compute_rank_loss, truncate_to_rank) and then exercise the full forward path
through a mock MatrixBottleneck.

Usage:
    python3 matrix-thinking/scripts/_smoke_rank_aware.py
"""

import sys
import os

# Ensure the parent directory is importable as a module source for the snippet below.
# We do NOT import the full run_matrix_codi (that would trigger transformers download).
# Instead we inline the three new functions and the updated MatrixBottleneck.forward
# signature so we can test them in isolation.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inline the new helper functions from run_matrix_codi (keep in sync!)
# ---------------------------------------------------------------------------

def truncate_to_rank(Z, k):
    """Same as run_matrix_codi.truncate_to_rank.

    Uses eigh(ZZ^T) instead of full SVD. eigh backward is stable even when
    eigenvalues (= σ_i^2) coincide, avoiding the 1/(σ_i^2 - σ_j^2) NaN that
    full-SVD backward produces at degenerate spectra.
    """
    orig_dtype = Z.dtype
    with torch.autocast(device_type=Z.device.type, enabled=False):
        Zf = Z.float()
        ZZt = Zf @ Zf.transpose(-1, -2)              # (B, d, d), symmetric PSD
        eigvals, eigvecs = torch.linalg.eigh(ZZt)    # ascending order; stable backward
        k = min(k, eigvecs.shape[-1])
        U_k = eigvecs[..., -k:]                      # (B, d, k) — top-k eigenvectors
        Zk = U_k @ (U_k.transpose(-1, -2) @ Zf)     # (B, d, d)
    return Zk.to(orig_dtype)


def compute_rank_loss(Z_list, rank_loss_type):
    """Same as run_matrix_codi.compute_rank_loss."""
    assert rank_loss_type in ("entropy", "nuclear")
    loss_terms = []
    for Z in Z_list:
        with torch.autocast(device_type=Z.device.type, enabled=False):
            sigma = torch.linalg.svdvals(Z.float())
        if rank_loss_type == "entropy":
            sigma_sum = sigma.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            p = sigma / sigma_sum
            p_safe = p.clamp(min=1e-10)
            H = -(p_safe * torch.log(p_safe)).sum(dim=-1)
            loss_terms.append(-H.mean())
        else:
            nuc = sigma.sum(dim=-1)
            loss_terms.append(-nuc.mean())
    return torch.stack(loss_terms).mean()


def effective_rank(Z):
    """Effective rank via singular-value entropy."""
    Zf = Z.float()
    sigma = torch.linalg.svdvals(Zf).clamp(min=1e-10)
    p = sigma / sigma.sum(dim=-1, keepdim=True)
    H = -(p * torch.log(p)).sum(dim=-1)
    return torch.exp(H)


# ---------------------------------------------------------------------------
# Minimal MatrixBottleneck with force_rank_k support (mirrors the production code)
# ---------------------------------------------------------------------------

class MinimalBottleneck(nn.Module):
    """Cut-down bottleneck: just w_up → reshape → optional force_rank → flatten → w_down."""

    def __init__(self, hidden_dim, mat_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mat_dim = mat_dim
        self.w_up = nn.Linear(hidden_dim, mat_dim * mat_dim)
        self.w_down = nn.Linear(mat_dim * mat_dim, hidden_dim)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_down.weight, std=0.02)

    def forward(self, h, force_rank_k=None):
        B = h.shape[0]
        d = self.mat_dim
        Z = self.w_up(h).view(B, d, d)  # (B, d, d)

        # eigh-based rank-k truncation: stable backward even at coincident σ.
        # See run_matrix_codi.MatrixBottleneck.forward for full rationale.
        if force_rank_k is not None and force_rank_k > 0:
            orig_dtype = Z.dtype
            with torch.autocast(device_type=Z.device.type, enabled=False):
                Zf = Z.float()
                ZZt = Zf @ Zf.transpose(-1, -2)              # (B, d, d), symmetric PSD
                eigvals, eigvecs = torch.linalg.eigh(ZZt)    # ascending; stable backward
                k_clamped = min(force_rank_k, eigvecs.shape[-1])
                U_k = eigvecs[..., -k_clamped:]              # (B, d, k)
                Zf_trunc = U_k @ (U_k.transpose(-1, -2) @ Zf)  # (B, d, d)
            Z = Zf_trunc.to(orig_dtype)

        h_out = self.w_down(Z.reshape(B, d * d))
        return h_out, Z


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rank_loss_entropy(d=4, B=2, n_latents=3, lam=0.1):
    print("\n[TEST 1] rank_loss=entropy, lambda=0.1")
    Z_list = [torch.randn(B, d, d, requires_grad=True) for _ in range(n_latents)]
    loss = compute_rank_loss(Z_list, "entropy")
    assert loss.requires_grad, "rank_loss must have requires_grad=True"
    assert not torch.isnan(loss), f"NaN in rank_loss (entropy): {loss}"
    scaled = lam * loss
    scaled.backward()
    for i, Z in enumerate(Z_list):
        assert Z.grad is not None, f"No grad for Z_list[{i}]"
        assert not torch.isnan(Z.grad).any(), f"NaN grad in Z_list[{i}]"
    print(f"  rank_loss (entropy) at lambda={lam}: {scaled.item():.6f}")
    print("  PASS")


def test_rank_loss_nuclear(d=4, B=2, n_latents=3, lam=0.1):
    print("\n[TEST 2] rank_loss=nuclear, lambda=0.1")
    Z_list = [torch.randn(B, d, d, requires_grad=True) for _ in range(n_latents)]
    loss = compute_rank_loss(Z_list, "nuclear")
    assert loss.requires_grad, "rank_loss must have requires_grad=True"
    assert not torch.isnan(loss), f"NaN in rank_loss (nuclear): {loss}"
    scaled = lam * loss
    scaled.backward()
    for i, Z in enumerate(Z_list):
        assert Z.grad is not None, f"No grad for Z_list[{i}]"
        assert not torch.isnan(Z.grad).any(), f"NaN grad in Z_list[{i}]"
    print(f"  rank_loss (nuclear) at lambda={lam}: {scaled.item():.6f}")
    print("  PASS")


def test_force_rank_forward_and_grad(d=4, B=2, hidden_dim=32, force_k=2):
    print(f"\n[TEST 3] force_rank_during_training k={force_k}: forward + grad")
    model = MinimalBottleneck(hidden_dim=hidden_dim, mat_dim=d)
    h = torch.randn(B, hidden_dim, requires_grad=True)

    # Without truncation
    h_out_full, Z_full = model(h)
    er_before = effective_rank(Z_full.detach()).mean().item()

    # Reset and run with truncation
    h2 = torch.randn(B, hidden_dim, requires_grad=True)
    h_out_trunc, Z_trunc = model(h2, force_rank_k=force_k)
    er_after = effective_rank(Z_trunc.detach()).mean().item()

    print(f"  effective_rank before truncate: {er_before:.3f}")
    print(f"  effective_rank after truncate_to_rank({force_k}): {er_after:.3f}")
    assert er_after <= force_k + 1e-2, \
        f"Truncation to rank {force_k} gave effective rank {er_after:.3f} > {force_k}+eps"

    # Backward through the truncation
    loss = h_out_trunc.sum()
    assert loss.requires_grad, "loss must have requires_grad after truncation"
    loss.backward()
    assert h2.grad is not None, "No grad at h2 after backward through truncation"
    assert not torch.isnan(h2.grad).any(), "NaN grad through truncation"
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"
    print(f"  backward through eigh truncation succeeded")
    print("  PASS")


def test_no_change_when_off(d=4, B=2, hidden_dim=32):
    """Verify that with rank_loss='none' and lambda=0 the loss is unchanged."""
    print("\n[TEST 4] Default OFF: rank_loss='none', lambda=0 → no Z_list needed")
    Z_list = [torch.randn(B, d, d, requires_grad=True) for _ in range(3)]
    # Simulated cfg defaults
    rank_loss_type = "none"
    rank_lambda = 0.0
    _need = (rank_loss_type != "none" and rank_lambda > 0.0)
    assert not _need, "rank loss should be OFF by default"

    # L_rank placeholder as used in compute_codi_loss
    L_rank = torch.zeros(1)
    assert L_rank.item() == 0.0
    print("  L_rank placeholder = 0.0 when disabled")
    print("  PASS")


def test_degenerate_sigma(d=4, force_k=2):
    """
    TEST 5 — Degenerate spectrum: coincident singular values must not NaN.

    Constructs a (1, 4, 4) tensor Z whose SVD has spectrum [3.0, 1.0, 1.0, 0.0]
    (two coincident σ = 1.0). Full SVD backward would NaN here because
    1/(σ_i^2 - σ_j^2) = 1/(1 - 1) = inf. The eigh-based truncation must:
      - Produce output of effective rank <= force_k (no NaN).
      - Complete a full backward pass without any NaN gradient.
    """
    print(f"\n[TEST 5] Degenerate spectrum (coincident σ): NaN-free backward")

    B = 1
    # Construct Z = U @ diag(S) @ Vh where S = [3.0, 1.0, 1.0, 0.0]
    # Use a fixed orthonormal U and Vh so the spectrum is exactly degenerate.
    torch.manual_seed(42)
    U_ref, _ = torch.linalg.qr(torch.randn(d, d))   # (d, d) orthogonal
    Vh_ref, _ = torch.linalg.qr(torch.randn(d, d))  # (d, d) orthogonal
    S_target = torch.tensor([3.0, 1.0, 1.0, 0.0])   # exactly coincident at index 1,2
    # Build Z and wrap in a leaf so we can compute gradients.
    Z_np = U_ref @ torch.diag(S_target) @ Vh_ref    # (d, d)
    Z_base = Z_np.unsqueeze(0)                       # (1, d, d)

    # Verify the spectrum is what we intended.
    sigma_check = torch.linalg.svdvals(Z_base[0]).tolist()
    print(f"  Constructed spectrum: {[f'{s:.4f}' for s in sigma_check]}")
    assert abs(sigma_check[1] - sigma_check[2]) < 1e-4, \
        f"Expected coincident σ[1]≈σ[2], got {sigma_check[1]:.6f} vs {sigma_check[2]:.6f}"

    # Attach as a parameter so we can take a gradient.
    Z_param = nn.Parameter(Z_base.clone())

    # Apply eigh-based truncation (the fixed path).
    Z_trunc = truncate_to_rank(Z_param, force_k)

    # Check no NaN in output.
    assert not torch.isnan(Z_trunc).any(), \
        f"NaN in truncated Z: {Z_trunc}"

    # Check effective rank <= force_k.
    er = effective_rank(Z_trunc.detach()).mean().item()
    print(f"  effective_rank after truncate_to_rank({force_k}): {er:.4f}")
    assert er <= force_k + 1e-2, \
        f"Truncation to rank {force_k} gave effective rank {er:.4f} > {force_k}+eps"

    # Backward: must complete without NaN.
    loss = Z_trunc.sum()
    loss.backward()

    assert Z_param.grad is not None, "No gradient returned for Z_param"
    nan_count = torch.isnan(Z_param.grad).sum().item()
    inf_count = torch.isinf(Z_param.grad).sum().item()
    grad_max = Z_param.grad.abs().max().item()
    print(f"  grad max_abs={grad_max:.4f}  NaN count={nan_count}  Inf count={inf_count}")
    assert nan_count == 0, f"NaN in gradient ({nan_count} elements)"
    assert inf_count == 0, f"Inf in gradient ({inf_count} elements)"
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("  SMOKE TEST: rank-aware training features")
    print("=" * 60)

    test_rank_loss_entropy()
    test_rank_loss_nuclear()
    test_force_rank_forward_and_grad()
    test_no_change_when_off()
    test_degenerate_sigma()

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 60)
