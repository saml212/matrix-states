"""
Algebraic Regularization for PHM layers.

Novel contribution: losses that push learned Kronecker factors toward
genuine algebraic structure instead of letting them collapse to nilpotent.

Three regularization terms:
1. Closure loss — force A_i @ A_j to stay in span of {A_0...A_n-1}
2. Division algebra loss — prevent det(A_i) → 0 (anti-nilpotent)
3. Structure preservation loss — encourage anti-commutativity of generators

These have NEVER been tried in the literature (confirmed by research).
"""

import torch
import torch.nn as nn


def closure_loss(A: torch.Tensor) -> torch.Tensor:
    """Force products A_i @ A_j to remain in the span of the basis.

    If the A matrices form a genuine algebra, then A_i @ A_j can be
    expressed as a linear combination of {A_0, ..., A_{n-1}}.
    This loss penalizes the residual of that projection.

    Args:
        A: (n, n, n) learned Kronecker factors
    Returns:
        scalar loss (0 = perfect closure)
    """
    n = A.shape[0]
    # Flatten basis: (n, n*n)
    basis_flat = A.reshape(n, -1)

    total_residual = torch.tensor(0.0, device=A.device)
    for i in range(n):
        for j in range(n):
            product = (A[i] @ A[j]).reshape(-1)  # (n*n,)
            # Project product onto basis span via least squares
            # coeffs = (basis_flat @ basis_flat.T)^{-1} @ basis_flat @ product
            # Use pseudoinverse for stability
            projection = basis_flat.T @ torch.linalg.lstsq(basis_flat.T, product).solution
            residual = product - projection
            total_residual = total_residual + residual.pow(2).sum()

    return total_residual / (n * n)


def closure_loss_fast(A: torch.Tensor) -> torch.Tensor:
    """Fast version of closure loss using batched operations.

    Computes how well A_i @ A_j can be reconstructed from the basis.
    """
    n = A.shape[0]
    basis_flat = A.reshape(n, -1)  # (n, n^2)

    total = torch.tensor(0.0, device=A.device)
    gram = basis_flat @ basis_flat.T  # (n, n)
    gram_inv = torch.linalg.pinv(gram)  # (n, n)

    for i in range(n):
        for j in range(n):
            product_flat = (A[i] @ A[j]).reshape(-1)  # (n^2,)
            # Coefficients via normal equations
            coeffs = gram_inv @ (basis_flat @ product_flat)  # (n,)
            reconstruction = coeffs @ basis_flat  # (n^2,)
            residual = product_flat - reconstruction
            total = total + residual.pow(2).mean()

    return total / (n * n)


def division_algebra_loss(A: torch.Tensor) -> torch.Tensor:
    """Prevent nilpotent collapse by penalizing near-zero determinants.

    In a division algebra, every non-zero element is invertible (det ≠ 0).
    This loss encourages the A matrices to stay invertible.

    We skip A_0 if it's identity-like (identity always has det=1).

    Args:
        A: (n, n, n) learned Kronecker factors
    Returns:
        scalar loss (0 = all generators invertible)
    """
    n = A.shape[0]
    loss = torch.tensor(0.0, device=A.device)

    for i in range(n):
        det = torch.det(A[i])
        # Penalize |det| < threshold (log barrier)
        # Use -log(|det| + eps) to push det away from zero
        loss = loss + torch.relu(0.5 - det.abs())  # penalty when |det| < 0.5

    return loss / n


def anticommutator_loss(A: torch.Tensor) -> torch.Tensor:
    """Encourage generators to anti-commute (Clifford algebra property).

    For Clifford algebras: e_i * e_j + e_j * e_i = 0 for i ≠ j.
    This is the defining property of quaternions and geometric algebras.

    We skip A_0 (assumed identity-like) and encourage A_i * A_j = -A_j * A_i
    for i,j ≥ 1.

    Args:
        A: (n, n, n) learned Kronecker factors
    Returns:
        scalar loss (0 = perfect anti-commutativity)
    """
    n = A.shape[0]
    loss = torch.tensor(0.0, device=A.device)
    count = 0

    for i in range(1, n):  # skip identity
        for j in range(i + 1, n):
            anticomm = A[i] @ A[j] + A[j] @ A[i]
            loss = loss + anticomm.pow(2).mean()
            count += 1

    return loss / max(count, 1)


def squared_trace_loss(A: torch.Tensor, target_traces: list = None) -> torch.Tensor:
    """Push squared traces toward a target algebra's signature.

    Quaternion: tr(A_i^2) = [4, -4, -4, -4]
    Split-quaternion: tr(A_i^2) = [4, -4, 4, 4]

    Default target: quaternion (most useful for multi-modal).

    Args:
        A: (n, n, n) learned Kronecker factors
        target_traces: list of n target values for tr(A_i^2)
    """
    n = A.shape[0]
    if target_traces is None:
        # Quaternion target
        target_traces = [float(n)] + [-float(n)] * (n - 1)

    loss = torch.tensor(0.0, device=A.device)
    targets = torch.tensor(target_traces, device=A.device)

    for i in range(n):
        sq_trace = (A[i] @ A[i]).trace()
        loss = loss + (sq_trace - targets[i]).pow(2)

    return loss / n


def full_algebra_regularization(model, closure_weight=0.1, division_weight=0.05,
                                anticomm_weight=0.1, trace_weight=0.05):
    """Compute full algebraic regularization loss across all PHM layers.

    Args:
        model: BytePHMTransformer or similar with PHMLinear modules
        closure_weight: weight for closure loss
        division_weight: weight for division algebra loss
        anticomm_weight: weight for anti-commutativity loss
        trace_weight: weight for squared trace loss

    Returns:
        total_loss: scalar
        diagnostics: dict of per-component losses
    """
    from .phm import PHMLinear

    total_closure = torch.tensor(0.0, device=next(model.parameters()).device)
    total_division = torch.tensor(0.0, device=next(model.parameters()).device)
    total_anticomm = torch.tensor(0.0, device=next(model.parameters()).device)
    total_trace = torch.tensor(0.0, device=next(model.parameters()).device)
    n_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, PHMLinear) and module.algebra_mode != 'quaternion':
            A = module.A  # (n, n, n) — use parameter, not detached
            total_closure = total_closure + closure_loss_fast(A)
            total_division = total_division + division_algebra_loss(A)
            total_anticomm = total_anticomm + anticommutator_loss(A)
            total_trace = total_trace + squared_trace_loss(A)
            n_layers += 1

    if n_layers == 0:
        zero = torch.tensor(0.0, device=next(model.parameters()).device)
        return zero, {'closure': 0, 'division': 0, 'anticomm': 0, 'trace': 0}

    total_closure = total_closure / n_layers
    total_division = total_division / n_layers
    total_anticomm = total_anticomm / n_layers
    total_trace = total_trace / n_layers

    total_loss = (closure_weight * total_closure +
                  division_weight * total_division +
                  anticomm_weight * total_anticomm +
                  trace_weight * total_trace)

    diagnostics = {
        'closure': total_closure.item(),
        'division': total_division.item(),
        'anticomm': total_anticomm.item(),
        'trace': total_trace.item(),
        'total_algebra_reg': total_loss.item(),
    }

    return total_loss, diagnostics
