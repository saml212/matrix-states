"""deltanet_core.py -- dtype-clean pure-PyTorch delta-rule STATE recurrence,
the numerical primitive the primary bespoke DeltaNet causal-rank harness
(model_dn.py) is built on. See DELTANET_CAUSAL_RANK_DESIGN.md sections 3.1,
3.5, and the F15 checkpoint (section 6.4).

**F15 checkpoint finding (2026-07-01, verified on the Brev box against
fla-org flash-linear-attention 0.5.1 / torch 2.12.1+cu130 / triton 3.7.1) --
why this module exists instead of calling fla's kernels directly:**
  - ``fla.ops.delta_rule.chunk_delta_rule`` (the production Triton kernel)
    explicitly REJECTS float32 inputs (raises
    "ChunkDeltaRuleFunction does not support float32. Please use
    bfloat16."), which makes a float64 finite-difference gradcheck through
    it categorically impossible.
  - ``fla.ops.delta_rule.naive.delta_rule_recurrence`` (fla's own
    pure-PyTorch reference) internally hard-casts every input to float32
    (``q, k, v, beta = map(lambda x: x.float(), ...)``), which silently
    defeats a float64 gradcheck too -- observed Jacobian mismatches of
    10-20% against the numerical gradient (a precision artifact of the
    downcast, not a real autograd bug: removing the downcast and re-running
    the IDENTICAL gradcheck passes cleanly, confirmed at the F15
    checkpoint).
  - ``chunk_delta_rule``'s BACKWARD additionally crashes with a CUDA
    "illegal memory access" (inside ``prepare_wy_repr_bwd_kernel``'s Triton
    autotuner) for sequence lengths below its internal ``chunk_size=64`` --
    exactly the short-sequence regime this design's BIND phases live in.
    Reproduced independently twice; confirmed ABSENT at T=128 (>=
    chunk_size). The ``fla.layers.delta_net.DeltaNet`` nn.Module
    conveniently self-protects against this by auto-switching to
    ``fused_recurrent`` mode whenever ``q_len <= 64`` regardless of the
    configured ``mode=`` -- a fact worth knowing if/when the
    production-block secondary check (design section 4.3) is built, but
    irrelevant here since this module never calls the Triton kernels at all.

This is precisely the design's pre-registered NEW-1 contingency ("fp64 if
the kernel supports it, else ... use the pure-PyTorch recurrent fallback
path", section 6.4). The primary bespoke harness uses ONLY this module --
never fla's Triton kernels -- for training AND the mandatory gradcheck.

**Math** (design section 3.1; Schlag, Irie & Schmidhuber arXiv:2102.11174;
Yang et al. arXiv:2406.06484). Scalar gate beta_t in [0,1], key k_t in R^d
(unit-normalized), value v_t in R^d, state S_t in R^{d x d}:

    S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T

Indexing convention (matches the design doc's own matrix notation literally,
NOT fla's internal tensor layout -- this module never interoperates
tensor-for-tensor with fla's kernels, so there is no cross-convention risk):
S is indexed [value-dim row i, key-dim col j], so ``S @ k`` (matrix-vector,
contracting the key/col axis) is well-defined AND ``S`` is square
(d_k == d_v == d, design section 3.1), so repeated self-application
``S^h`` (design section 5.4's readout) is well-defined.

**No q, no per-token output.** A real DeltaNet layer also computes a
per-token output ``o_t = q_t^T S_t`` for the residual stream. This design's
primary harness readout is EXTERNAL and PINNED
(``pred(a,h) = S_T^h @ key_a``, section 5.4) and never consults any
per-token output -- so this module computes ONLY the state trajectory /
final state. This is a deliberate simplification specific to this design's
readout, not a different recurrence: the state-update equation above is
unchanged and is exactly what a real DeltaNet layer computes internally.
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
    product, per-query hop depth, NO learned per-hop weights (mirrors
    chapter2/model_e.py::MatrixCompositionModel.compose exactly, reused
    here rather than reimplemented ad hoc so the two lineages' composition
    semantics stay identical). S: (B, D, D); x: (B, Q, D); h: (B, Q) int64,
    h >= 0 (h=0 is the identity)."""
    cur = x
    result = torch.where(h.unsqueeze(-1) == 0, cur, torch.zeros_like(cur))
    max_h = int(h.max().item()) if h.numel() > 0 else 0
    for t in range(1, max_h + 1):
        cur = torch.einsum("bij,bqj->bqi", S, cur)
        mask = (h == t).unsqueeze(-1)
        result = torch.where(mask, cur, result)
    return result


# ---------------------------------------------------------------------------
# Self-test (part of run_deltanet.py --smoke)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    print("[core 1] architecture-native ideal (design section 3.6 proposition), "
          "mechanically verified through the ACTUAL delta_rule_state code path "
          "(not a hand derivation): orthonormal keys + beta=1 -> S_K == sum v_j k_j^T exactly")
    B, K, D = 3, 6, 6
    keys, _ = torch.linalg.qr(torch.randn(B, D, K, device=DEV))
    keys = keys.transpose(-1, -2)                       # (B, K, D) orthonormal rows
    values = torch.nn.functional.normalize(torch.randn(B, K, D, device=DEV), dim=-1)
    beta_ones = torch.ones(B, K, device=DEV)
    S_T = delta_rule_state(keys, values, beta_ones)
    S_ideal = torch.einsum("bki,bkj->bij", values, keys)   # sum_j v_j k_j^T
    diff = (S_T - S_ideal).abs().max().item()
    assert diff < 1e-4, f"architecture-native ideal mismatch: max abs diff {diff:.6f}"
    # order-invariance corollary (design section 5.3): a random permutation of
    # the SAME bindings reaches the identical S_T (orthonormal-key regime only).
    perm = torch.randperm(K)
    S_T_perm = delta_rule_state(keys[:, perm], values[:, perm], beta_ones)
    diff_perm = (S_T_perm - S_T).abs().max().item()
    assert diff_perm < 1e-4, f"order-invariance violated: max abs diff {diff_perm:.6f}"
    print(f"  max abs diff vs analytic ideal = {diff:.2e}; order-invariance diff = {diff_perm:.2e}")

    print("\n[core 2] round-trip: delta_rule_state(BIND) -> initial_state into a "
          "continuation call == a single continuous call over the concatenated sequence")
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

    print("\n[core 3] gradcheck (float64, small D) through BOTH kernel hooks, per NEW-1")
    Bc, Tc, Dc = 1, 4, 3
    kq = torch.randn(Bc, Tc, Dc, dtype=torch.float64, device=DEV, requires_grad=True)
    vq = torch.randn(Bc, Tc, Dc, dtype=torch.float64, device=DEV, requires_grad=True)
    bq = torch.rand(Bc, Tc, dtype=torch.float64, device=DEV, requires_grad=True)

    def f_ofs(k_, v_, b_):
        return delta_rule_state(k_, v_, b_)

    assert torch.autograd.gradcheck(f_ofs, (kq, vq, bq), eps=1e-6, atol=1e-5, rtol=1e-3), \
        "gradcheck FAILED: output_final_state backward"
    print("  output_final_state backward: gradcheck PASSED")

    S0 = torch.randn(Bc, Dc, Dc, dtype=torch.float64, device=DEV, requires_grad=True)
    k2 = torch.randn(Bc, Tc, Dc, dtype=torch.float64, device=DEV)
    v2 = torch.randn(Bc, Tc, Dc, dtype=torch.float64, device=DEV)
    b2 = torch.rand(Bc, Tc, dtype=torch.float64, device=DEV)

    def f_init(S0_):
        return delta_rule_state(k2, v2, b2, initial_state=S0_)

    assert torch.autograd.gradcheck(f_init, (S0,), eps=1e-6, atol=1e-5, rtol=1e-3), \
        "gradcheck FAILED: initial_state backward"
    print("  initial_state backward: gradcheck PASSED")

    print("\n[core 4] NEGATIVE TEST (run to completion, per the standing house rule on "
          "structural checks): a deliberately DETACHED path must FAIL gradcheck")

    def f_init_broken(S0_):
        return delta_rule_state(k2, v2, b2, initial_state=S0_.detach())

    neg1_failed = False
    try:
        passed = torch.autograd.gradcheck(f_init_broken, (S0,), eps=1e-6, atol=1e-5, rtol=1e-3)
        neg1_failed = not passed
    except Exception:
        neg1_failed = True
    assert neg1_failed, "negative test did NOT fail -- the gradcheck has no teeth"
    print("  detached initial_state correctly FAILS gradcheck (teeth confirmed)")

    def f_ofs_broken(k_, v_, b_):
        return delta_rule_state(k_, v_.detach(), b_.detach())

    neg2_failed = False
    try:
        passed2 = torch.autograd.gradcheck(f_ofs_broken, (kq, vq, bq), eps=1e-6, atol=1e-5, rtol=1e-3)
        neg2_failed = not passed2
    except Exception:
        neg2_failed = True
    assert neg2_failed, "negative test 2 did NOT fail -- the gradcheck has no teeth"
    print("  detached v/beta correctly FAILS gradcheck (teeth confirmed)")

    print("\n[core 5] apply_state_power: h=0 is identity; composes correctly at arbitrary h")
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

    print("\ndeltanet_core self-test PASSED")


if __name__ == "__main__":
    _self_test()
