"""NCR write-conditioning loss module (Rev-5 BUILD-PREPARABLE).

Implements D1 (loss re-specified, zero set exactly {Z_ideal}), D3 (detached
targets, single top-of-function detach covering both sub-terms' SVD route),
D4/F1-repaired Band 2 (scale-free reading, c*-rescale + the ratio form, R3's
own "at ||Z||_F ~= 25" 12% conditioning restored), and M11 (d-K
generalization of the transverse term: W = Vh[:, K:, :] rather than the
single w = Vh[:, -1, :], which reduces to the carded single-w form exactly
at K = d-1).

Source: matrix-thinking/NCR_WRITE_CONDITIONING_DESIGN.md DRAFT-R4 (D1-D4,
D8/m1) + NCR_WRITECOND_ATTACK_R4.md F1/M1/M11 + the coordinator adjudication
at the design doc's tail (`8c665f7`..`6b3d750`, `§A4-ADJUDICATION`), adopted
as the Rev-5 charter.

No box dependency: pure torch (torch.linalg.svd/svdvals + einsum). CPU-
testable -- see test_write_supervision_loss.py (run to completion as part
of this Rev-5 revision, not merely written).

This module NEVER assumes a value for `lambda_t` (F2's repair -- every
call site must supply it; Stage 0' amendment A6 is what decides it) and
never sets its own Band-2 thresholds beyond the two the report already
fixed (L_key <= 3e-4, ||ZW||_F/||Z||_F <= 0.12) -- no new numeric
threshold is introduced anywhere in this file.
"""
from __future__ import annotations

import torch


def null_directions(keys_v: torch.Tensor, K: int | None = None) -> torch.Tensor:
    """D1.2 + M11's generalization. keys_v: (B, K, d), assumed DETACHED by
    the caller (D3 -- write_supervision_loss's own top-of-function detach
    covers the common call path; this function does not detach on its own
    so it can also be called on Z_ideal-only proof-device tensors where
    detach is the caller's concern, e.g. Stage 0' item 1/4).

    Returns W: (B, d-K, d) -- the (d-K) smallest right-singular directions
    of keys_v (its structural null space; keys_v has full row rank K,
    V3, structurally guaranteed every batch by
    grammar_rd._assert_injective_entities). At K = d-1 this is exactly the
    single `w` D1.2 defines (Vh[:, -1:, :], attack R3's own verified
    `max|w.k_i| = 2.0e-07`); M11's generalization is W = Vh[:, K:, :] for
    any K < d-1, and D1.3's zero-set proof goes through verbatim with
    `u w^T -> U W^T` (u: (B,d) -> U: (B, d, d-K)).
    """
    Bsz, Kdim, d = keys_v.shape
    if K is None:
        K = Kdim
    assert Kdim == K, f"null_directions: keys_v's own K dim ({Kdim}) must match K={K}"
    assert K < d, f"null_directions requires K < d (a nontrivial null space); got K={K}, d={d}"
    _, _, Vh = torch.linalg.svd(keys_v, full_matrices=True)   # Vh: (B, d, d), orthogonal
    W = Vh[:, K:, :]                                            # (B, d-K, d), rows orthonormal (W W^T = I_{d-K})
    return W


def write_supervision_loss(Z: torch.Tensor, keys_v: torch.Tensor, values_v: torch.Tensor,
                            lambda_t: float, K: int | None = None, eps: float = 1e-6,
                            W: torch.Tensor | None = None) -> dict:
    """D1's re-specified write-conditioning loss. D3's single
    top-of-function detach covers BOTH sub-terms' inputs -- L_key's direct
    (keys_v, values_v) AND L_transverse's SVD route (attack R4's F3/D3
    point: detaching only L_key's inputs leaves a second, narrower door
    into the entity_adapter/embed collapse route open via an undetached
    `w`'s SVD backward).

    Z: (B, d, d) -- the SGD-trained write operator (ncr_head.encode's own
       output). NOT detached: gradient must flow here, into the encoder.
    keys_v, values_v: (B, K, d) -- the extracted, UNDETACHED adapter
       outputs exactly as ncr_lm_forward_ablatable returns them. Detached
       HERE, once, before either sub-term is computed (D3).
    lambda_t: the transverse term's weight. NEVER defaulted or assumed --
       Stage 0' amendment A6 is the instrument that decides it (F2's
       repair); every caller must supply a value explicitly.
    K: bind-clause count. Defaults to keys_v's own K dim; pass explicitly
       only to exercise the d-K generalization at K < d-1 (M11).
    W: precomputed null directions (B, d-K, d). If None, computed via
       null_directions(keys_v.detach(), K) internally. Exposed so a caller
       computing W once (e.g. Stage 0' items 1/4/6, which all need the SAME
       SVD of the SAME keys_v) can reuse it rather than re-running SVD.

    Returns dict(L_key, L_transverse: (B,) per-episode; L_write: scalar,
    batch-mean; ZW: (B, d-K, d) the transverse projection pre-square,
    diagnostic; v_bar2: (B,) the per-episode value-energy scale;
    W: the null-direction basis actually used).
    """
    assert Z.dim() == 3 and Z.shape[-1] == Z.shape[-2], f"Z must be (B,d,d), got {tuple(Z.shape)}"
    Bsz, K_actual, d = keys_v.shape
    if K is None:
        K = K_actual
    keys_v_d, values_v_d = keys_v.detach(), values_v.detach()      # D3: THE single top-of-function detach

    # --- L_key (D1.1, R3 sec2(a), frozen, unmodified) ---
    r = torch.einsum('bij,bkj->bki', Z, keys_v_d) - values_v_d      # (B,K,d), r_i = Z k_i - v_i
    L_key = (r.pow(2).sum(-1) / (values_v_d.pow(2).sum(-1) + eps)).mean(-1)   # (B,)

    # --- L_transverse (D1.2, M11's d-K generalization) ---
    if W is None:
        W = null_directions(keys_v_d, K=K)                          # (B, d-K, d) -- off the DETACHED keys (D3)
    ZW = torch.einsum('bij,bkj->bki', Z, W)                          # (B, d-K, d): Z applied to each null direction
    v_bar2 = values_v_d.pow(2).sum(-1).mean(-1) + eps                # (B,), D1.2's per-episode value-energy scale
    L_transverse = ZW.pow(2).sum(dim=(-2, -1)) / v_bar2               # (B,), generalizes ||Zw||^2/v_bar2 to d-K > 1

    L_write = (L_key + lambda_t * L_transverse).mean()                # scalar, batch-mean (D1.2)

    return dict(L_key=L_key, L_transverse=L_transverse, L_write=L_write,
                ZW=ZW, v_bar2=v_bar2, W=W)


def band2_check(Z: torch.Tensor, keys_v: torch.Tensor, values_v: torch.Tensor,
                 W: torch.Tensor | None = None, K: int | None = None,
                 l_key_bound: float = 3e-4, zw_ratio_bound: float = 0.12,
                 eps: float = 1e-6) -> dict:
    """F1-repaired Band 2 (D4, attack R4's F1, adopted verbatim by the
    coordinator adjudication). Two hard gates, BOTH exactly scale-invariant
    to a positive global rescale of Z (F1's diagnosed failure mode -- a
    bare, un-quotiented ||Z_sgd w|| <~ 3 made a 1.5x-globally-rescaled
    PERFECT operator FAIL and a collapsed-||Z||~1 operator PASS at chance
    -- cannot recur here BY CONSTRUCTION, not by re-calibration):

      (i)  L_key(c* . Z) <= l_key_bound          [3e-4, D1.5's calibration]
      (ii) ||Z W^T||_F / ||Z||_F <= zw_ratio_bound  [0.12 = 3/25, R3's OWN
           "at ||Z||_F ~= 25" statement, restored WITH its conditioning --
           this is exactly what DRAFT-R4's D1.5 dropped in transcription]

    c* = sum_i <Z k_i, v_i> / sum_i ||Z k_i||^2 (per episode, closed form --
    the report's own stated rescale, F1's repair paragraph, verbatim).
    This is the minimizer of the UNNORMALIZED sum-square residual over a
    single positive global scale c on Z; it is a legitimate, cheap proxy
    for L_key's own per-key-normalized minimizer (not necessarily
    identical to it), and is the exact form the adjudication binds to --
    this module does not substitute a different c*.

    The ratio gate (ii) needs no c*: ||Z W^T||_F and ||Z||_F both scale
    linearly with any positive c on Z, so their ratio is scale-free by
    construction, independent of whatever c* gate (i) computes.

    Returns dict with both gates' raw per-episode values, the two NAMED
    reduction statistics (median AND p90 -- F1's repair explicitly
    required this to be named, "the current text names none"), and
    band2_pass (bool AND of both gates, per-episode).
    """
    keys_v_d, values_v_d = keys_v.detach(), values_v.detach()
    Bsz, K_actual, d = keys_v_d.shape
    if K is None:
        K = K_actual

    Zk = torch.einsum('bij,bkj->bki', Z, keys_v_d)                   # (B,K,d)
    num = (Zk * values_v_d).sum(dim=(-2, -1))                        # (B,) sum_i <Zk_i, v_i>
    den = Zk.pow(2).sum(dim=(-2, -1)) + eps                          # (B,) sum_i ||Zk_i||^2
    c_star = num / den                                               # (B,)

    Z_rescaled = Z * c_star.view(-1, 1, 1)
    r_star = torch.einsum('bij,bkj->bki', Z_rescaled, keys_v_d) - values_v_d
    L_key_cstar = (r_star.pow(2).sum(-1) / (values_v_d.pow(2).sum(-1) + eps)).mean(-1)   # (B,)

    r_raw = torch.einsum('bij,bkj->bki', Z, keys_v_d) - values_v_d
    L_key_raw = (r_raw.pow(2).sum(-1) / (values_v_d.pow(2).sum(-1) + eps)).mean(-1)      # diagnostic only, never gated

    if W is None:
        W = null_directions(keys_v_d, K=K)
    ZW = torch.einsum('bij,bkj->bki', Z, W)                           # (B, d-K, d)
    Zw_fro = ZW.norm(dim=(-2, -1))                                    # (B,)
    Z_fro = Z.norm(dim=(-2, -1))                                      # (B,)
    Zw_ratio = Zw_fro / (Z_fro + eps)                                 # (B,) -- scale-free BY CONSTRUCTION

    gate1 = L_key_cstar <= l_key_bound
    gate2 = Zw_ratio <= zw_ratio_bound
    band2_pass = gate1 & gate2

    return dict(
        c_star=c_star, L_key_cstar=L_key_cstar, L_key_raw=L_key_raw,
        Zw_fro=Zw_fro, Z_fro=Z_fro, Zw_ratio=Zw_ratio,
        gate1_Lkey_pass=gate1, gate2_ratio_pass=gate2, band2_pass=band2_pass,
        # Reduction statistics, named explicitly (F1's own repair requirement):
        L_key_cstar_median=L_key_cstar.median().item(), L_key_cstar_p90=L_key_cstar.quantile(0.90).item(),
        Zw_ratio_median=Zw_ratio.median().item(), Zw_ratio_p90=Zw_ratio.quantile(0.90).item(),
        band2_pass_frac=band2_pass.float().mean().item(),
        # The population-level Band-2 reading at the two NAMED statistics
        # (median and p90) against the two fixed thresholds -- the form
        # Stage 0' item 6 (A6) uses to decide GO/NO-GO, never a new
        # invented threshold:
        band2_pass_at_median=bool(L_key_cstar.median().item() <= l_key_bound
                                   and Zw_ratio.median().item() <= zw_ratio_bound),
        band2_pass_at_p90=bool(L_key_cstar.quantile(0.90).item() <= l_key_bound
                                and Zw_ratio.quantile(0.90).item() <= zw_ratio_bound),
    )
