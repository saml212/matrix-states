"""deltanet_core.py -- copied VERBATIM (with attribution) from
``matrix-thinking/deltanet/deltanet_core.py``, itself the F15-checkpoint-
verified dtype-clean pure-PyTorch delta-rule STATE recurrence.

Why copied, not imported: deltanet_rd/ is self-contained for pod deployment
(no cross-directory imports), matching this repo's established pod-safety
convention -- see matrix-thinking/deltanet/rank_utils.py's own docstring for
the origin of this rule (the 2026-04-29 rank_aware_v1 import-fragility
incident).

**What deltanet_rd/ actually uses from this module: ONLY ``apply_state_power``**
(the pinned multi-hop external readout primitive, DELTANET_REALDATA_DESIGN.md
section 5.1: "pinned iterated-matmul composition readout pred(a,h) =
S_T^h . key_a"). ``delta_rule_state`` (the pure-PyTorch recurrence) is kept
here for reference/self-test parity with the synthetic design's own module,
but is NOT used on model_rd.py's training hot path -- DELTANET_REALDATA_DESIGN.md
section 4.3's R2-3 decision calls ``fla.ops.delta_rule.chunk_delta_rule``
(the production Triton kernel) directly instead, because (a) it rejects
float32 categorically (making it useless for an LM-scale fp64 gradcheck
regardless) and (b) LM-scale training needs the chunked-parallel kernel's
throughput, not a per-token Python loop (design section 4.3, reversing the
synthetic design's own choice for exactly the opposite reason it made it).
See f15_lm_checkpoint.py for the LM-path verification chain that replaces
this module's own _self_test gradcheck for the production-kernel path.

DO NOT MODIFY the function bodies -- verbatim-identical to
matrix-thinking/deltanet/deltanet_core.py (only this docstring differs). If a
fix is ever needed, fix it upstream there first, then re-copy.
"""
from __future__ import annotations

import torch


def delta_rule_state(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                      initial_state: torch.Tensor | None = None,
                      return_trajectory: bool = False):
    """k, v: (B, T, D) (D_k == D_v == D, the design's square d x d state);
    beta: (B, T). initial_state: (B, D, D) or None (-> zeros). Dtype- and
    device-preserving throughout -- NO internal float32/float64 cast (see
    module docstring for why that matters: it is what makes this module
    gradcheck-able at float64, unlike fla's own naive.py reference).

    Returns S_T: (B, D, D). If return_trajectory=True, also returns a list
    of T per-step states S_1..S_T (diagnostic use only, e.g. confirming the
    state stays finite mid-BIND-phase; never used on the training hot path).
    """
    B, T, D = k.shape
    assert v.shape == (B, T, D), f"k {tuple(k.shape)} vs v {tuple(v.shape)}: D_k must == D_v"
    assert beta.shape == (B, T), beta.shape
    S = torch.zeros(B, D, D, dtype=k.dtype, device=k.device)
    if initial_state is not None:
        assert initial_state.shape == (B, D, D), tuple(initial_state.shape)
        S = S + initial_state.to(dtype=k.dtype)
    traj = [] if return_trajectory else None
    for t in range(T):
        k_t = k[:, t, :]                                    # (B, D)
        v_t = v[:, t, :]                                     # (B, D)
        beta_t = beta[:, t]                                   # (B,)
        pred_t = torch.einsum("bij,bj->bi", S, k_t)           # S_{t-1} @ k_t
        delta_t = beta_t.unsqueeze(-1) * (v_t - pred_t)       # beta_t (v_t - S_{t-1} k_t)
        S = S + torch.einsum("bi,bj->bij", delta_t, k_t)      # + delta_t (outer) k_t
        if return_trajectory:
            traj.append(S)
    return (S, traj) if return_trajectory else S


def apply_state_power(S: torch.Tensor, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Pinned multi-hop readout primitive (design section 5.4):
    pred[b,q] = S[b]^{h[b,q]} @ x[b,q] -- literal iterated matrix-vector
    product, per-query hop depth, NO learned per-hop weights. S: (B, D, D);
    x: (B, Q, D); h: (B, Q) int64, h >= 0 (h=0 is the identity)."""
    cur = x
    result = torch.where(h.unsqueeze(-1) == 0, cur, torch.zeros_like(cur))
    max_h = int(h.max().item()) if h.numel() > 0 else 0
    for t in range(1, max_h + 1):
        cur = torch.einsum("bij,bqj->bqi", S, cur)
        mask = (h == t).unsqueeze(-1)
        result = torch.where(mask, cur, result)
    return result


# ---------------------------------------------------------------------------
# Self-test (part of run_deltanet_rd.py --smoke)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    print("[core 1] architecture-native ideal: orthonormal keys + beta=1 -> "
          "S_K == sum v_j k_j^T exactly (mechanically verified through the "
          "ACTUAL delta_rule_state code path)")
    B, K, D = 3, 6, 6
    keys, _ = torch.linalg.qr(torch.randn(B, D, K, device=DEV))
    keys = keys.transpose(-1, -2)
    values = torch.nn.functional.normalize(torch.randn(B, K, D, device=DEV), dim=-1)
    beta_ones = torch.ones(B, K, device=DEV)
    S_T = delta_rule_state(keys, values, beta_ones)
    S_ideal = torch.einsum("bki,bkj->bij", values, keys)
    diff = (S_T - S_ideal).abs().max().item()
    assert diff < 1e-4, f"architecture-native ideal mismatch: max abs diff {diff:.6f}"
    print(f"  max abs diff vs analytic ideal = {diff:.2e}")

    print("\n[core 2] round-trip: delta_rule_state(BIND) -> initial_state into a "
          "continuation call == a single continuous call")
    T1, T2 = 5, 4
    k_full = torch.nn.functional.normalize(torch.randn(2, T1 + T2, D, device=DEV), dim=-1)
    v_full = torch.randn(2, T1 + T2, D, device=DEV)
    beta_full = torch.rand(2, T1 + T2, device=DEV).sigmoid()
    S_single = delta_rule_state(k_full, v_full, beta_full)
    S_bind = delta_rule_state(k_full[:, :T1], v_full[:, :T1], beta_full[:, :T1])
    S_two = delta_rule_state(k_full[:, T1:], v_full[:, T1:], beta_full[:, T1:], initial_state=S_bind)
    rt_diff = (S_single - S_two).abs().max().item()
    assert rt_diff < 1e-5, f"round-trip mismatch: max abs diff {rt_diff:.6f}"
    print(f"  round-trip max abs diff = {rt_diff:.2e}")

    print("\n[core 3] apply_state_power: h=0 is identity; composes correctly at arbitrary h")
    Sp = torch.randn(2, 4, 4, device=DEV)
    x = torch.randn(2, 3, 4, device=DEV)
    h0 = torch.zeros(2, 3, dtype=torch.int64, device=DEV)
    assert torch.allclose(apply_state_power(Sp, x, h0), x), "h=0 must be the identity"
    h3 = torch.full((2, 3), 3, dtype=torch.int64, device=DEV)
    manual = x.clone()
    for _ in range(3):
        manual = torch.einsum("bij,bqj->bqi", Sp, manual)
    assert torch.allclose(apply_state_power(Sp, x, h3), manual, atol=1e-5)
    print("  h=0 identity + manual h=3 composition both match")

    print("\ndeltanet_core (deltanet_rd copy) self-test PASSED")


if __name__ == "__main__":
    _self_test()
