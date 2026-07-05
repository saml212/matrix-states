"""model_rd.py -- the R2-3 custom DeltaNet-RD block: calls
``fla.ops.delta_rule.chunk_delta_rule`` (the production Triton kernel)
DIRECTLY, not through the stock ``fla.layers.delta_net.DeltaNet`` nn.Module,
because the stock layer computes beta internally via its own ``b_proj`` and
exposes NO hook for an externally-supplied hard mask -- C9's beta-mask
requires one (DELTANET_REALDATA_DESIGN.md section 5.2, R2-3).

Architecture (H=1 single head, single layer -- C11, transplanted verbatim
from DELTANET_CAUSAL_RANK_DESIGN.md, section 5.1: "full transplant, not
re-derived here"):

  BIND phase (per grammar_rd.py's streamed token sequence):
    x_t          = embed(token_id_t)                        # (B,T,d_model), LEARNED table
    k_pre_t      = W_k(x_t)                                  # (B,T,d_state), LEARNED, no bias
    v_pre_t      = W_v(x_t)                                  # (B,T,d_state), LEARNED, no bias
    k_eff_t      = k_conv1d(k_pre)_t   (SiLU-activated)       # CAUSAL short conv, kernel=conv_size
    v_eff_t      = v_conv1d(v_pre)_t   (SiLU-activated)       # matches fla.layers.delta_net's own
                                                                # conv+activation convention exactly
                                                                # (production-block fidelity, section
                                                                # 4.3/5.2: "the fla-block conv path is
                                                                # part of the mechanism under test")
    beta_logit_t = sigmoid(W_beta(x_t))                       # LEARNED, from the RAW pre-conv
                                                                # embedding (matches stock DeltaNet's
                                                                # own b_proj(hidden_states) convention)
    beta_t       = beta_logit_t * beta_mask_t                 # C9/R2-3: HARD 0 off the K VALUE
                                                                # (write) positions -- the mask is
                                                                # architectural; the sigmoid GATE
                                                                # VALUE at real write positions is
                                                                # genuinely learned
    k_eff_t      = normalize(k_eff_t)                          # EXTERNAL zero-safe l2norm (see
                                                                # bind()'s docstring: deep buffer
                                                                # positions are exact-zero rows)
    S_T = kernel_state_design_layout(k_eff, v_eff, beta)       # PRODUCTION kernel via the ONE
                                                                # layout-reconciling helper: fla
                                                                # returns [K,V] (key axis FIRST);
                                                                # the design convention is [V,K]
                                                                # (audit FATAL-0 -- see the
                                                                # helper's docstring); bf16-cast
                                                                # at that boundary only; sequences
                                                                # padded to >= _MIN_KERNEL_T first
    S_T = truncate_to_rank(S_T, force_rank_k)  if forced       # section 5.1's two-kernel-call split,
                                                                # ONE truncation per forward pass;
                                                                # C15 asserted immediately after

  QUERY phase (external, pinned readout -- section 5.4 of the synthetic
  design, transplanted verbatim; "the query span passes through the feature
  path only, never the recurrence", section 5.2):
    q_eff_a = normalize(k_conv1d(W_k(embed([buf...,KEY_a,REL,<Q>])))[-1])
    pred(a,h) = S_T^h @ q_eff_a                                # apply_state_power, zero new params

**Build-time decision, flagged for audit scrutiny: no W_q / q_conv1d.** The
stock DeltaNet layer's own per-token output ``o_t = q_t^T S_t`` is UNUSED
here (Wave 1's readout is external and pinned, never the kernel's own
per-token output) and mathematically ``q`` does NOT affect ``S_T`` at all --
verified directly against fla's own naive-recurrence reference (its per-step
update ``S = S + k_t (v_t) ^T`` after the beta/value adjustment never
consults ``q``; ``q`` only computes the discarded ``o``). So this block
reuses ``k_eff`` as the kernel's ``q`` argument rather than building a
functionally-inert extra projection+conv module -- mirrors
DELTANET_CAUSAL_RANK_DESIGN.md's model_dn.py's own
"readout adds ZERO new parameters beyond W_k" discipline (C_composition-
purity), extended one step further (no W_q at all, not just no NEW readout
params). Empirically verified harmless on this exact fla 0.5.1 build,
2026-07-02 (forward+backward finite at H=1, D=64, T in {24,56,64,112,224}
with q=k reuse) -- re-verified formally in f15_lm_checkpoint.py.

**Build-time decision: W_v is genuinely learned (unlike
DELTANET_CAUSAL_RANK_DESIGN.md's model_dn.py, where W_v=identity).** Real
tokens have no pre-made R^d "value vector" the way the synthetic harness's
hand-built values did; section 5.2 requires "the learned W_k/W_v
projections must extract the key component into k_eff and the value
component into v_eff -- learned, not hand-wired." Consequently the scoring
TARGET for a query is not a fixed generator vector (task_dn.py's
convention) but ``v_eff`` computed for the TARGET entity's own bind-time
clause -- gathered via ``tgt_slot`` from the SAME forward pass's
``v_eff_items``, since each entity appears as VALUE in EXACTLY ONE clause
(the K-cycle is a bijection). See ``forward()``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule

from deltanet_core import apply_state_power
from rank_utils import effective_rank, stable_rank, truncate_to_rank
from key_anchoring import (ANCHOR_INIT_SEED, LAMBDA_LOG_CADENCE_STEPS, LAMBDA_WINDOW_LOG_POINTS,
                            anchor_blend_gather_scatter, anchor_blend_gather_scatter_per_entity,
                            frame_potential_init, random_unit_rows_init, raw_table_conditioning)

# chunk_delta_rule's BACKWARD crashes with a CUDA illegal memory access
# (inside prepare_wy_repr_bwd_kernel's Triton autotuner) in two measured
# regimes on this box's build (fla 0.5.1 / torch 2.12.1+cu130 / triton
# 3.7.1, F15-LM checkpoint rounds 1-2, 2026-07-02):
#   (a) sequence length below the kernel's internal chunk_size=64 -- the
#       synthetic design's own F15 finding ("confirmed ABSENT at T>=128"),
#       reproduced here at raw (T_bind=56, d_state=16);
#   (b) HEAD DIM below 64, even at safe T -- measured crash matrix (fresh
#       process per probe): D=16 crashed at T=224 (passed flakily at 128),
#       D=32 crashed at T=128 AND 224; D=64 and D=128 clean at every
#       T in {128, 224, 448} probed.
# The stock fla.layers.delta_net.DeltaNet self-protects against (a) by
# auto-switching to fused_recurrent at q_len <= 64; calling the kernel
# directly (R2-3) forfeits that protection. This harness closes (a) by
# padding every BIND sequence to at least _MIN_KERNEL_T with trailing
# BUFFER tokens (STATE-NEUTRAL BY CONSTRUCTION: pad positions carry the
# zero-pinned buffer embedding and a hard beta=0, and the delta rule
# writes nothing where beta=0 -- verified bit-identical S_T padded-vs-
# unpadded in f15_lm_checkpoint.py item 3), and closes (b) by asserting
# d_state is in the MEASURED-SAFE set below at construction time. Wave 1's
# primary d_state=64 (section 4.1's continuity decision) is in the safe
# set; the mini self-test configs were moved from d_state=16 to 64 for
# exactly this reason, not for capacity.
_MIN_KERNEL_T = 128
_SAFE_D_STATE = (64, 128)


def truncate_to_rank_svd_lowrank(Z: torch.Tensor, k: int, n_oversample: int = 6,
                                   niter: int = 4) -> torch.Tensor:
    """Randomized-SVD rank-k truncation -- ported VERBATIM (function body)
    from matrix-thinking/deltanet/model_dn.py::truncate_to_rank_svd_lowrank,
    where it was added as the audited FINDING-1 stability fallback to
    rank_utils.truncate_to_rank's eigh path (Halko, Martinsson & Tropp 2011:
    Gaussian sketch + matmuls + QR + one SMALL (q x d) SVD).

    Why it is now IN the RD harness (build deviation #6's pre-registered
    re-add condition FIRED, Wave -1 2026-07-02): all 3 force-rank Wave -1
    cells crashed with torch._C._LinAlgError ("linalg.eigh failed to
    converge") in truncate_to_rank's FORWARD eigh on real-embedding states
    -- a harder failure mode than the synthetic harness's measured BACKWARD
    skip-rate instability (a forward raise cannot be caught by the
    gradient-hygiene skip). Guarantees preserved vs the eigh path: output
    rank <= k BY CONSTRUCTION (exactly k components kept, regardless of
    oversampling) -- C15's assert_rank_le applies unchanged; exactness on
    well-separated spectra verified in _self_test [model 4b], run to
    completion. Semantics caveat (unchanged from the synthetic harness):
    torch.svd_lowrank draws a FRESH Gaussian test matrix per call (global
    RNG, no generator arg), so the truncation is stochastic at
    ~subspace-error magnitude and gradcheck through it is not meaningful;
    selection is recorded per-run in the result JSON ("trunc_impl") and
    suffixed into sweep run names, never a silent default swap.

    Z: (..., d, d). Returns same shape, rank <= k along the last two dims.
    """
    orig_dtype = Z.dtype
    with torch.autocast(device_type=Z.device.type, enabled=False):
        Zf = Z.float()
        d_last = Zf.shape[-1]
        q = min(k + n_oversample, d_last)
        U, S, V = torch.svd_lowrank(Zf, q=q, niter=niter)
        # keep EXACTLY k components -- rank <= k by construction even with q > k
        Uk, Sk, Vk = U[..., :, :k], S[..., :k], V[..., :, :k]
        Zk = Uk @ (Sk.unsqueeze(-1) * Vk.transpose(-1, -2))
    return Zk.to(orig_dtype)


# Selectable truncation implementations (same pattern as the synthetic
# harness's model_dn.py). "eigh" stays the design default; "svd_lowrank" is
# the pre-registered stability fallback, activated for the force-rank arm
# by Wave -1's measured eigh forward-convergence failures (see
# truncate_to_rank_svd_lowrank's docstring).
TRUNC_IMPLS = {
    "eigh": truncate_to_rank,
    "svd_lowrank": truncate_to_rank_svd_lowrank,
}

# torch.linalg.LinAlgError is the public alias of torch._C._LinAlgError
# (the exact exception Wave -1's crashes raised); never fall back to a
# broad RuntimeError -- that would silently swallow unrelated failures.
_LinAlgError = getattr(torch.linalg, "LinAlgError", None) or torch._C._LinAlgError


class TruncationError(RuntimeError):
    """Raised by bind() when the rank truncation fails numerically even
    after the one-shot jitter retry. train() converts this into a SKIPPED
    step (n_skipped accounting) and evaluate_pool into a skipped eval
    batch: a numerical non-convergence must never kill a run (the
    harness-robustness rule; Wave -1's eigh forward raise was exactly such
    a kill)."""


def assert_rank_le(S: torch.Tensor, k: int) -> None:
    """C15: post-truncation rank(S_T) <= k, asserted by DIRECT SVD. The
    threshold comparison is EXACT (rank <= k, never k+1); the SVD-to-integer
    numerical-zero cutoff itself is the ordinary, necessary tolerance for
    turning floating-point singular values into an integer rank (not the
    "-1"-style slack the house rule on structural checks prohibits)."""
    r = torch.linalg.matrix_rank(S.float())
    assert torch.all(r <= k), f"C15 VIOLATED: post-truncation rank {r.tolist()} > k={k}"


def entity_subspace_rank(S_T: torch.Tensor, s_ideal: torch.Tensor, K: int):
    """PRIMARY M1-equivalent metric, transplanted verbatim from
    DELTANET_CAUSAL_RANK_DESIGN.md's model_dn.py (TASK_E_FINDINGS.md's
    subspace-restriction lesson): project S_T onto the K-dim entity subspace
    recovered from the architecture-native ideal's own SVD. s_ideal: (B,d,d)
    = sum_j v_eff_j @ k_eff_j^T. Returns (effective_rank, stable_rank) of the
    K x K restricted operator, each (B,)."""
    U, Sig, Vh = torch.linalg.svd(s_ideal.float())
    K = min(K, U.shape[-1])
    Uk = U[..., :, :K]
    Vk = Vh[..., :K, :].transpose(-1, -2)
    S_restricted = torch.einsum("bdk,bde,bel->bkl", Uk, S_T.float(), Vk)
    return effective_rank(S_restricted), stable_rank(S_restricted)


def gram_deviation(eff: torch.Tensor) -> torch.Tensor:
    """C16's instrument (section 5.2): ||Eff^T Eff - I||_F per batch row.
    Applied to the (already L2-normalized) key side for premise (i); applied
    to an L2-normalized COPY of the value side for premise (ii) ("checked
    with the SAME instrument applied to the value side", section 5.2) --
    the copy does not change what v_eff feeds into the kernel, it is
    diagnostic-only. eff: (B, K, d)."""
    gram = eff @ eff.transpose(-1, -2)
    n = eff.shape[1]
    eye = torch.eye(n, device=eff.device, dtype=gram.dtype).expand_as(gram)
    return (gram - eye).norm(dim=(-2, -1))


def orth_penalty(eff: torch.Tensor) -> torch.Tensor:
    """F-geo-1 (DELTANET_RD_EXACTNESS_DESIGN.md sec 5.5): the per-episode
    differentiable orthogonality penalty's own object, BEFORE the
    lambda_orth scale -- ||K_eff^T K_eff - I_K||_F^2 over the K in-episode
    k_eff_items (the EXACT tensor C16's gram_deviation already gathers;
    this is that same Gram-minus-identity matrix, squared-Frobenius
    instead of gram_deviation's plain Frobenius norm, per sec 5.5's
    literal L_orth formula). Training-loss use only -- run_deltanet_rd.py
    multiplies by lambda_orth and adds to the total loss; this diagnostic/
    penalty is NEVER part of eval scoring (sec 14.3's no-eval-leak rule).
    eff: (B,K,d) -- the same k_eff_items gram_deviation consumes.
    Returns (B,) squared-Frobenius-norm per batch row (mean over batch is
    the caller's job, matching cosine_loss's own per-row-then-mean
    convention)."""
    gram = eff @ eff.transpose(-1, -2)
    n = eff.shape[1]
    eye = torch.eye(n, device=eff.device, dtype=gram.dtype).expand_as(gram)
    return (gram - eye).pow(2).sum(dim=(-2, -1))


def salvage_ratio(eff: torch.Tensor) -> torch.Tensor:
    """R2-1(ii)'s causal-arm headline-validity instrument: sigma_K / sigma_1
    of the (K,d) effective-key matrix (linear independence with bounded
    conditioning -- the bound's actual minimal premise). eff: (B, K, d)."""
    s = torch.linalg.svdvals(eff.float())          # (B, min(K,d)) descending
    return s[..., -1] / s[..., 0].clamp(min=1e-12)


def pin_buffer_row_(embedding: nn.Embedding, buffer_id: int) -> None:
    """R2-3's zero-pinning mechanism: force the reserved BUFFER row to EXACT
    zero. Called at init AND after every optimizer step in the training loop
    (run_deltanet_rd.py) -- re-zeroing post-step (rather than only masking
    the gradient pre-step) is belt-and-suspenders against optimizer-state
    drift (e.g. stale Adam moments) and is what the F15-LM checkpoint's
    'buffer embedding rows are exactly zero AFTER optimizer steps' item
    actually verifies."""
    with torch.no_grad():
        embedding.weight[buffer_id].zero_()


def zero_buffer_grad_(embedding: nn.Embedding, buffer_id: int) -> None:
    """Companion to pin_buffer_row_: zero the buffer row's GRADIENT before
    the optimizer step, so no optimizer state (Adam moments, etc.) for that
    row ever accumulates real signal in the first place -- "excluded from
    the optimizer" (section 5.2), not merely corrected after the fact."""
    if embedding.weight.grad is not None:
        with torch.no_grad():
            embedding.weight.grad[buffer_id].zero_()


def pin_rows_(embedding: nn.Embedding, row_ids: torch.Tensor, values: torch.Tensor) -> None:
    """DELTANET_RD_EXACTNESS_DESIGN.md sec 4 -- generalizes pin_buffer_row_
    to an ARBITRARY SET of frozen rows (the embed-source arms' frozen
    identity rows: arm (i) frozen_orthonormal, arm (ii) frozen_gpt2_span,
    arm (iv) frozen_gram_matched, sec 4.2/4.3/4.5; also reused verbatim by
    arm (i-strong)'s strong_pin_table, sec 4.4). Force
    embedding.weight[row_ids] = values (exact), called at init AND after
    every optimizer step -- same belt-and-suspenders discipline as
    pin_buffer_row_ (re-asserting post-step guards against optimizer-state
    drift). row_ids: (N,) int64 on the SAME device as embedding.weight;
    values: (N, d_model)."""
    with torch.no_grad():
        embedding.weight[row_ids] = values.to(dtype=embedding.weight.dtype,
                                                device=embedding.weight.device)


def zero_rows_grad_(embedding: nn.Embedding, row_ids: torch.Tensor) -> None:
    """Generalizes zero_buffer_grad_ to an arbitrary SET of frozen rows
    (companion to pin_rows_ -- zero the gradient BEFORE the optimizer step
    so no optimizer state accumulates real signal for a frozen row)."""
    if embedding.weight.grad is not None:
        with torch.no_grad():
            embedding.weight.grad[row_ids] = 0.0


def _gather_at(seq: torch.Tensor, item_pos: torch.Tensor) -> torch.Tensor:
    d = seq.shape[-1]
    idx = item_pos.unsqueeze(-1).expand(-1, -1, d)
    return torch.gather(seq, 1, idx)


def target_clause_index(succ: torch.Tensor, tgt_slot: torch.Tensor) -> torch.Tensor:
    """Mini-audit FATAL fix (2026-07-03): map the answer entity's SLOT
    (tgt_slot = pi^h(a)) to the CLAUSE whose VALUE token is that entity.
    Clause i binds KEY=entity[i] -> VALUE=entity[pi(i)], so the clause
    holding entity s as its VALUE is pi^{-1}(s): one argsort (the inverse
    permutation) + one gather. Design section 14.4's tgt(j) ("clause index
    of pi^{h_j}(a_j) -- the correct entity's bind clause") is exactly this.
    ONE shared site: forward()'s cosine/eval targets AND train()'s NCE
    class labels both call it -- they can never disagree.
    succ: (B,K) the K-cycle bijection; tgt_slot: (B,Q). Returns (B,Q)."""
    inv_succ = torch.argsort(succ, dim=1)                 # pi^{-1} as a table
    return torch.gather(inv_succ, 1, tgt_slot)


def kernel_state_design_layout(k_norm: torch.Tensor, v: torch.Tensor,
                                beta: torch.Tensor) -> torch.Tensor:
    """The ONE place this harness calls chunk_delta_rule for a final state,
    and the ONE place the fla-vs-design axis convention is reconciled
    (audit round-1 FATAL-0 -- this transpose was MISSING in the original
    build and every downstream consumer silently inherited the wrong axis
    order).

    LAYOUT FACTS (verified against fla 0.5.1 source + a single-write probe:
    k=e0, v=e1, beta=1 gives final_state[0,1]=1.0 -- KEY axis FIRST):
      - chunk_delta_rule returns final_state of shape [N, H, K, V]: the
        state is S_fla[key_dim, value_dim], retrieval is k^T @ S_fla.
      - This design's convention (deltanet_core.py's docstring, the
        DELTANET_CAUSAL_RANK_DESIGN.md section 3.1 notation, and
        apply_state_power's einsum) is S[value_dim, key_dim]: S = sum_j
        v_j k_j^T, retrieval is S @ k. deltanet_core.delta_rule_state
        produces THIS layout natively (S += delta outer k -> rows=value).
      - Measured consequence of omitting the transpose (audit round 1, an
        unmodified real run): recovered_frac saturated at 1.000 while
        entity-subspace rank collapsed to ~1.0, key Gram deviation grew to
        7.36, and C19 zero-shot stayed 0.000 -- a degenerate shortcut
        producing a FALSE CONFIRM; the transpose-only control run
        converged ~40x faster and reached C19 0.79-1.00.

    Inputs (design side, fp32): k_norm/v (B,T,d) with k_norm already
    L2-normalized (zero-safe), beta (B,T) already hard-masked. Handles the
    bf16 cast (chunk_delta_rule rejects float32 categorically -- F15-LM
    fact), the H=1 head axis, and the [K,V]->[V,K] transpose. Returns S_T
    (B,d,d) in the DESIGN layout. Gradients flow (transpose and casts are
    autograd-transparent). Covered by _self_test [model 10]'s idealized-
    recall regression (orthonormal keys, beta=1 -> recall cosine ~1.0;
    deliberately-transposed negative control must FAIL) -- the pre-fix
    test suite was transpose-blind, that test is the audit-prescribed
    regression closing it."""
    k_bf = k_norm.unsqueeze(2).to(torch.bfloat16)                    # (B,T,1,d), H=1
    v_bf = v.unsqueeze(2).to(torch.bfloat16)
    beta_bf = beta.unsqueeze(-1).to(torch.bfloat16)                  # (B,T,1)
    _, S_kv = chunk_delta_rule(q=k_bf, k=k_bf, v=v_bf, beta=beta_bf,
                                output_final_state=True, use_qk_l2norm_in_kernel=False)
    # fla [N,H,K,V] -> squeeze H -> [B,K,V] -> transpose -> DESIGN [B,V,K]
    return S_kv.squeeze(1).float().transpose(-1, -2)


def newton_schulz_orthogonalize(A: torch.Tensor, n_iter: int = 12
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """DELTANET_RD_EXACTNESS_DESIGN.md sec 14.1/14.2 -- F-geo-3's primary
    orthogonalizer. A: (B,K,d), K<=d, rows ALREADY L2-normalized (bind()'s
    existing zero-safe F.normalize -- the pre-scaling proof below relies on
    this). Returns (Q, resid): Q (B,K,d) with Q@Q^T ~= I_K per batch element;
    resid (B,) = ||Q Q^T - I_K||_F AFTER n_iter, detached (sec 14.1's Wave -1
    fallback trigger, no_grad -- does not affect Q's own gradient path).

    Pre-scaling (sec 14.1): every row of A has unit norm, so
    sigma_max(A) <= ||A||_F = sqrt(K) unconditionally (A A^T's trace is K;
    its top eigenvalue is at most the trace). X_0 := A / sqrt(K) therefore
    has sigma_max(X_0) <= 1 ALWAYS, inside Newton-Schulz's convergence basin,
    with no data-dependent norm estimate computed at all.

    Order-equivariant by construction (sec 14.1's rejection of Gram-Schmidt/
    QR): every step involves only X_t and X_t X_t^T (K x K) -- permuting A's
    rows permutes X_t X_t^T's rows/columns identically and permutes every
    iterate the same way, no row is privileged."""
    B, K, d = A.shape
    X = A / (K ** 0.5)                            # sec 14.1 pre-scale: sigma_max(X)<=1 always
    I_K = torch.eye(K, device=A.device, dtype=A.dtype)
    for _ in range(n_iter):
        G = X @ X.transpose(-1, -2)               # (B,K,K)
        X = 1.5 * X - 0.5 * (G @ X)               # cubic Newton-Schulz / Bjorck-Bowie
    with torch.no_grad():
        resid = (X @ X.transpose(-1, -2) - I_K).norm(dim=(-2, -1))
    return X, resid


def _polar_via_eigh(A: torch.Tensor, eps_scale: float = 1e-4) -> torch.Tensor:
    """Fallback path (sec 14.4). REFACTOR NOTE: mathematically and
    procedurally the SAME jitter-retry-then-clamp discipline as
    ZCAWhiten._recompute_transform -- applied here to the per-episode
    (B,K,K) key Gram matrix instead of ZCA's (d,d) population feature
    covariance. Same math, same failure class (eigh non-convergence on a
    near-singular Gram matrix), same fix -- a shared refactor is flagged as
    a follow-on build item, not done here (sec 14.2's build spec keeps this
    a standalone function to match the design's literal code block)."""
    B, K, d = A.shape
    Sigma = A @ A.transpose(-1, -2)                            # (B,K,K)
    eye = torch.eye(K, device=A.device, dtype=A.dtype)
    eps = eps_scale * Sigma.diagonal(dim1=-2, dim2=-1).sum(-1) / K       # (B,)
    try:
        eigvals, eigvecs = torch.linalg.eigh(Sigma + eps.view(B, 1, 1) * eye)
    except _LinAlgError:
        jitter = eps + 1e-6 * Sigma.diagonal(dim1=-2, dim2=-1).sum(-1) / K
        eigvals, eigvecs = torch.linalg.eigh(Sigma + jitter.view(B, 1, 1) * eye)  # ONE retry,
        # matching bind()'s own force_rank_k retry-once-then-raise discipline (L771-780);
        # a second failure propagates a LinAlgError to geo3_orthogonalize's caller (bind()),
        # which does NOT catch it -- an eigh failure on BOTH the primary and the fallback path
        # is a genuine numerical emergency, not a routine skip-step event (unlike force_rank_k's
        # TruncationError conversion), and is deliberately left to surface loudly.
    eigvals = eigvals.clamp(min=eps.view(B, 1).expand(-1, K))
    inv_sqrt = eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2)
    return inv_sqrt @ A


def geo3_orthogonalize(k_eff_items: torch.Tensor, n_iter: int = 12,
                        resid_tol: float = 1e-2) -> torch.Tensor:
    """bind()'s F-geo-3 call site (sec 14.2). Primary: batched,
    differentiable Newton-Schulz. Fallback: WHOLE-BATCH retry via the shared
    eigh-polar route if ANY episode's residual exceeds resid_tol after
    n_iter -- batch-level, not episode-level, granularity (sec 14.4's
    documented scoping tradeoff -- matches bind()'s own retry-the-whole-
    computation convention for force_rank_k rather than introducing a new
    per-element dynamic-branching primitive). Returns Q only; callers that
    need the fallback-triggered flag for per-checkpoint logging (sec 14.4)
    should call newton_schulz_orthogonalize directly, or use
    geo3_orthogonalize_logged below."""
    Q, resid = newton_schulz_orthogonalize(k_eff_items, n_iter=n_iter)
    if bool((resid > resid_tol).any()):
        Q = _polar_via_eigh(k_eff_items)
    return Q


def geo3_orthogonalize_logged(k_eff_items: torch.Tensor, n_iter: int = 12,
                               resid_tol: float = 1e-2) -> tuple[torch.Tensor, bool, torch.Tensor]:
    """Sec 14.4's per-step/per-checkpoint fallback-triggered logging build
    item: identical math to geo3_orthogonalize, but ALSO returns
    (fallback_triggered: bool, resid_after_ns: (B,) detached) so callers
    (train()/evaluate_pool()) can log which numerics a given step used --
    "log a per-step fallback-triggered flag so any resulting loss-curve
    noise is diagnosable, not mysterious" (sec 14.4)."""
    Q, resid = newton_schulz_orthogonalize(k_eff_items, n_iter=n_iter)
    fallback_triggered = bool((resid > resid_tol).any())
    if fallback_triggered:
        Q = _polar_via_eigh(k_eff_items)
    return Q, fallback_triggered, resid


class ZCAWhiten(nn.Module):
    """F-geo-2 (DELTANET_RD_EXACTNESS_DESIGN.md sec 5.5, BLOCK-3's
    buildable spec; sec 13's build-time clarification #2 pins the
    insertion point): a ZCA-style whitening module on the post-conv KEY
    features, inserted IMMEDIATELY POST-CONV, BEFORE the existing
    ``F.normalize(k_conv)`` step in both bind()'s BIND-phase path and
    effective_key_window()'s query-phase path -- i.e. ``k_conv -> ZCA ->
    L2-normalize``, in both places that currently each compute k_conv then
    normalize. Population statistics are causal-safe (EMA over TRAINING
    steps only, never within-episode) and CONTENT-POSITION-ONLY (BUFFER
    positions -- exact-zero conv output by construction -- are masked out
    of every statistics update; ~75% of tokens at K=8's padded T=128, so
    pooling them would degenerate the covariance, per BLOCK-3(a)).

    UNCENTERED second moment, a DELIBERATE BUILD DECISION not spelled out
    verbatim in the design text (flagged for audit scrutiny): Sigma =
    E[x x^T] over content positions, NO mean subtraction, so
    W_zca = (Sigma + eps*I)^{-1/2} and y = W_zca @ x satisfy x=0 -> y=0
    EXACTLY. A CENTERED whitening transform (subtract a running mean mu
    before applying W_zca) would map the architecture's exact-zero BUFFER
    conv output to a NONZERO vector (-W_zca @ mu), silently breaking every
    zero-buffer invariant this harness depends on elsewhere: R2-8's
    cross-clause leak isolation, the padding state-neutrality smoke ("pad
    positions carry... a hard beta=0, and the delta rule writes nothing
    where beta=0" -- model_rd.py's own _MIN_KERNEL_T comment), and
    F.normalize's zero-safe convention. Uncentered ZCA preserves ALL of
    these unchanged while still decorrelating the CONTENT-position
    population (buffer rows contribute zero mass to Sigma either way, so
    centered vs. uncentered is indistinguishable for THEM regardless --
    the difference only matters for what happens to a zero INPUT, which
    only buffer rows ever are).

    Numerics (BLOCK-3(b), pre-registered): EMA running second moment
    (momentum 0.99, bias-corrected -- the BN convention), fp32 buffers,
    NO gradient into the statistics (stop-grad, BN-style: only the applied
    linear map gets a gradient path, via x itself); eps = 1e-4 * tr(Sigma)/d
    (scale-invariant, strictly-PD by construction); transform recomputed
    every 100 EMA UPDATES from the cached bias-corrected EMA (not every
    forward call -- see the query-path scoping note below); the
    deviation-#6 jitter-retry (+1e-6*tr/d on LinAlgError, one retry) then
    an eigenvalue-clamping fallback (never an unhandled crash); a 500-
    update identity warm-up with a hard switch at exactly update 500 (500
    is itself a multiple of the 100-update recompute cadence, so the
    switch and the first live recompute coincide by construction, not by
    a separate special case).

    SCOPING DEVIATION (documented, not silent): the design's phrase
    "content positions... KEY/REL/VALUE/PERIOD content and query-window
    content" nominally pools BOTH the BIND stream and every query-window
    call into the same running statistics. This module instead accumulates
    statistics ONLY from bind()'s BIND-phase call (content positions =
    token_ids != buffer_id there); effective_key_window()'s query-path
    calls TRANSFORM using the current frozen W_zca but never update the
    EMA or the step counter. Rationale: bind() and effective_key_window()
    are each called exactly once per training step (via forward()/
    readout()), so counting BOTH would double the update cadence against
    a single per-step step counter with no natural single point to
    reconcile a shared counter between two independent call sites without
    a second piece of cross-module state; and the BIND stream already
    contains the SAME KEY/REL token distribution a query window would
    contribute (identical weights, identical token types) -- only VALUE
    (bind-only, already pooled) and the <Q> marker (query-only, structural)
    differ. This is a deliberate scope-narrowing, flagged for audit
    scrutiny, not a silent simplification.
    """

    def __init__(self, d_state: int, momentum: float = 0.99, eps_scale: float = 1e-4,
                 warmup_updates: int = 500, recompute_every: int = 100):
        super().__init__()
        assert warmup_updates % recompute_every == 0, \
            "warmup_updates must be a multiple of recompute_every so the hard warm-up " \
            "switch coincides with a scheduled recompute (BLOCK-3(b))"
        self.d_state = d_state
        self.momentum = momentum
        self.eps_scale = eps_scale
        self.warmup_updates = warmup_updates
        self.recompute_every = recompute_every
        self.register_buffer("running_cov", torch.zeros(d_state, d_state))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("W_zca", torch.eye(d_state))

    @torch.no_grad()
    def _update_stats(self, x_content: torch.Tensor) -> None:
        if x_content.shape[0] == 0:
            return
        x32 = x_content.float()
        batch_cov = (x32.transpose(0, 1) @ x32) / x32.shape[0]
        self.running_cov.mul_(self.momentum).add_(batch_cov, alpha=1.0 - self.momentum)
        self.num_updates += 1

    @torch.no_grad()
    def _bias_corrected_cov(self) -> torch.Tensor:
        t = int(self.num_updates.item())
        if t == 0:
            return self.running_cov.clone()
        bc = 1.0 - self.momentum ** t
        return self.running_cov / max(bc, 1e-8)

    @torch.no_grad()
    def _recompute_transform(self) -> None:
        Sigma = self._bias_corrected_cov()
        d = Sigma.shape[-1]
        eps = (self.eps_scale * torch.diagonal(Sigma).sum() / d).clamp(min=1e-12)
        eye = torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
        try:
            eigvals, eigvecs = torch.linalg.eigh(Sigma + eps * eye)
        except _LinAlgError:
            try:
                jitter = eps + 1e-6 * torch.diagonal(Sigma).sum() / d
                eigvals, eigvecs = torch.linalg.eigh(Sigma + jitter * eye)
            except _LinAlgError:
                # eigenvalue-clamping fallback (svd_lowrank-style spirit --
                # never an unhandled crash mid-run, matching bind()'s own
                # TruncationError discipline elsewhere in this file).
                eigvals, eigvecs = torch.linalg.eigh(Sigma.double())
                eigvals = eigvals.float()
                eigvecs = eigvecs.float()
        eigvals = eigvals.clamp(min=float(eps.item()))
        inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2)
        self.W_zca.copy_(inv_sqrt)

    def cond_number(self) -> float:
        """Wave -1 smoke instrument (BLOCK-3(c)): cond(Sigma + eps*I)."""
        Sigma = self._bias_corrected_cov()
        d = Sigma.shape[-1]
        eps = (self.eps_scale * torch.diagonal(Sigma).sum() / d).clamp(min=1e-12)
        vals = torch.linalg.eigvalsh(Sigma + eps * torch.eye(d, device=Sigma.device))
        return float((vals.max() / vals.clamp(min=1e-20).min()).item())

    def forward(self, x: torch.Tensor, content_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B,T,d_state) post-conv key features. content_mask: (B,T)
        bool, True at positions that should feed the running statistics
        (required, and used, only while self.training AND update_stats is
        requested by the caller -- see bind()'s single-call-site
        discipline above). Returns whitened features, same shape as x;
        identity for the first warmup_updates EMA updates (a training-time
        notion; eval calls always apply the current cached W_zca)."""
        if self.training and content_mask is not None:
            with torch.no_grad():
                flat = x.reshape(-1, x.shape[-1])
                mask_flat = content_mask.reshape(-1)
                self._update_stats(flat[mask_flat])
                t = int(self.num_updates.item())
                if t >= self.warmup_updates and t % self.recompute_every == 0:
                    self._recompute_transform()
        if int(self.num_updates.item()) < self.warmup_updates:
            return x
        W = self.W_zca.to(dtype=x.dtype)
        return x @ W.transpose(-1, -2)


class DeltaNetRDBlock(nn.Module):
    """The primary R2-3 custom block. H=1 single head, single layer (C11).
    No W_q/q_conv1d (see module docstring), no output gate, no o_proj -- the
    readout is external and pinned (section 5.4), so nothing downstream of
    S_T is a learned parameter beyond what BIND already used."""

    def __init__(self, vocab_size_total: int, d_model: int = 256, d_state: int = 64,
                 conv_size: int = 4, buffer_id: int = None, trunc_impl: str = "eigh",
                 embed_source: str = "learned", frozen_row_ids: torch.Tensor | None = None,
                 frozen_row_values: torch.Tensor | None = None,
                 strong_pin_ids: torch.Tensor | None = None,
                 strong_pin_values: torch.Tensor | None = None,
                 use_zca: bool = False, geo3_active: bool = False,
                 geo3_n_iter: int = 12, geo3_resid_tol: float = 1e-2,
                 anchor_active: bool = False, anchor_lambda_mode: str = "learned",
                 anchor_lambda_fixed: float | None = None,
                 anchor_train_ids: torch.Tensor | None = None,
                 anchor_init_seed: int | None = None,
                 anchor_table_frozen: bool = False,
                 anchor_table_init_mode: str = "frame_potential"):
        """DELTANET_RD_EXACTNESS_DESIGN.md sec 4/4.4/5.5/14 extensions, ALL
        ADDITIVE and OFF BY DEFAULT (embed_source="learned",
        frozen_row_ids=None, strong_pin_ids=None, use_zca=False,
        geo3_active=False) -- run_deltanet_rd.py's default path is
        byte-identical to the pre-extension code when none of these are
        passed (regression-checked, see run_deltanet_rd.py's smoke).

        anchor_active/anchor_lambda_mode/anchor_lambda_fixed/anchor_train_ids
          (KEY_ANCHORING_DESIGN.md sec 2.2, candidate (d), PRIMARY): the
          trainable per-entity anchor table + masked gather/scatter blend,
          inserted PRE-Newton-Schulz at the existing geo3_active site (a
          modification to geo3's OWN write path, never a replacement for
          it -- asserted below: anchor_active implies geo3_active).
          anchor_train_ids: (n_train,) int64 real vocab ids (typically
          pools.train_name_ids) -- the ONLY rows the frame-potential init
          populates and the ONLY rows anchor_trained_mask marks True (the
          C17/M1 held-out bypass, sec 3.3: held-out rows carry NO anchor
          arithmetic in either direction). anchor_lambda_mode="learned":
          a single learned scalar nn.Parameter, sigmoid(raw_param), init
          raw_param=0.0 -> lambda=0.5 (sec 2.2). "fixed": a non-trainable
          scalar from the pre-registered grid {0.3,0.6,0.9} (sec 2.2's
          fallback diagnostic), REQUIRES anchor_lambda_fixed.
          "learned_per_entity" (KEY_ANCHORING_DESIGN.md sec 10.5.1,
          candidate (d')): a per-entity nn.Embedding(vocab_size_total, 1)
          replacing the single scalar -- SAME sigmoid/init-0.5 convention,
          gathered per t_idx entity inside
          key_anchoring.anchor_blend_gather_scatter_per_entity. anchor_
          lambda() is NOT valid in this mode (asserts); bind() dispatches
          to the per-entity blend function directly.

        anchor_table_frozen/anchor_table_init_mode (KEY_ANCHORING_DESIGN.md
          sec 10.13's registered candidate (e), "frozen-random-table
          ablation"): anchor_table_frozen=True calls
          self.anchor_table.weight.requires_grad_(False) immediately after
          construction (the trained ROWS never receive a gradient; held-out
          rows already never do, sec 3.3's C17 bypass, unaffected either
          way) -- a minimal, one-line addition to the existing masked
          gather/scatter/held-out-bypass path, no new mechanism.
          anchor_table_init_mode selects which construction populates the
          trained-row block: "frame_potential" (default, candidate (d)'s
          own tight-frame-minimized init, UNCHANGED) or "random_unit_rows"
          (candidate (e)'s own registered init -- key_anchoring.
          random_unit_rows_init, frame_potential_init's own pre-optimization
          starting point, i.e. seeded random unit rows with NO frame-
          potential descent -- matching sec 10.13's own candidate-(e) name
          and sec 10.13.4's motivating text, "a random, FROZEN anchor
          table," rather than a trained-but-frozen tight frame). Orthogonal
          to anchor_lambda_mode: candidate (e) is registered at
          anchor_lambda_mode="fixed" (matched-lambda comparison), but
          anchor_table_frozen composes with any anchor_lambda_mode in
          principle (this class does not couple the two).

        embed_source: informational only at this layer (recorded into the
          result JSON by run_deltanet_rd.py); the CALLER (embed_arms.py +
          run_deltanet_rd.py) builds frozen_row_ids/values via the correct
          arm-specific construction and passes them in here uniformly --
          this class does not know or care WHICH arm produced them.
        frozen_row_ids/(N,) int64, frozen_row_values (N,d_model): arms (i)/
          (ii)/(iv) -- rows of self.embed pinned via pin_rows_ at init AND
          after every optimizer step (run_deltanet_rd.py's train() loop),
          same discipline as the existing buffer-row pinning.
        strong_pin_ids/(P,) int64, strong_pin_values (P,d_state): arm
          (i-strong) sec 4.4 -- a SEPARATE, dedicated per-entity lookup
          (keyed by entity token id, valued in d_state directly, NOT
          d_model) that bypasses W_k/k_conv1d entirely at bind()'s write
          positions and at effective_key_window()'s query positions.
          Orthogonal to embed_source (can combine with any of them; the
          design only exercises it at embed_source="learned").
        use_zca: F-geo-2 (sec 5.5) -- inserts a shared ZCAWhiten instance
          immediately post-conv, before L2-normalize, in both bind() and
          effective_key_window() (sec 13's build-time clarification #2).
        geo3_active/geo3_n_iter/geo3_resid_tol: F-geo-3 (sec 14) -- per-
          episode differentiable Newton-Schulz orthogonalization of the K
          keys actually written into S_T, inserted at bind()'s gather site
          (a THIRD branch alongside strong_pin_active, sec 14.2). Mutually
          exclusive with strong_pin_active (asserted below) -- both arms
          replace k_eff_items at write time by construction and combining
          them has no defined meaning (sec 14.2). Composes with use_zca in
          principle (ZCA transforms the population distribution BEFORE
          F-geo-3's per-episode gather+orthogonalize) but that combination
          is explicitly out of THIS design's scope (sec 14.2).
        """
        super().__init__()
        assert buffer_id is not None, "buffer_id (the reserved zero-pinned token id) is required"
        assert not (strong_pin_ids is not None and geo3_active), \
            "arm (i-strong) and F-geo-3 (geo3_active) are MUTUALLY EXCLUSIVE (sec 14.2): both " \
            "replace k_eff_items at write time by construction; combining them has no defined meaning"
        assert trunc_impl in TRUNC_IMPLS, \
            f"trunc_impl must be one of {sorted(TRUNC_IMPLS)}, got {trunc_impl!r}"
        assert d_state in _SAFE_D_STATE, \
            f"d_state={d_state} is NOT in the measured-safe head-dim set {_SAFE_D_STATE} for " \
            f"chunk_delta_rule's backward on this box's build -- see the _SAFE_D_STATE comment " \
            f"(D=16/32 crash prepare_wy_repr_bwd's autotuner; measured 2026-07-02). Re-measure " \
            f"before widening this set."
        self.vocab_size_total = vocab_size_total
        self.d_model = d_model
        self.d_state = d_state
        self.conv_size = conv_size
        self.buffer_id = buffer_id
        self.trunc_impl = trunc_impl
        self.embed_source = embed_source

        self.embed = nn.Embedding(vocab_size_total, d_model)
        self.k_proj = nn.Linear(d_model, d_state, bias=False)
        self.v_proj = nn.Linear(d_model, d_state, bias=False)
        self.k_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size,
                                          bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size,
                                          bias=False, activation="silu")
        self.W_beta = nn.Linear(d_model, 1, bias=True)

        pin_buffer_row_(self.embed, self.buffer_id)

        # -- arms (i)/(ii)/(iv), sec 4.2/4.3/4.5: frozen identity rows --
        # NOTE: every tensor accepted here is normalized to CPU FIRST,
        # regardless of what device the caller happened to construct it
        # on -- self.embed (like every nn.Module submodule) starts life on
        # CPU until the STANDARD `DeltaNetRDBlock(...).to(device)` chain
        # this codebase uses everywhere moves it; indexed-assignment into
        # a CPU embedding.weight with a CUDA index tensor raises a device-
        # mismatch RuntimeError, so doing this indexing pre-.to() with
        # CUDA-resident constructor args (a real, caught-in-review bug in
        # an earlier draft of this build) would crash. CPU-normalizing
        # here makes __init__ robust to either calling convention.
        if frozen_row_ids is not None:
            assert frozen_row_values is not None
            frozen_row_ids = frozen_row_ids.detach().cpu()
            frozen_row_values = frozen_row_values.detach().cpu()
            self.register_buffer("frozen_row_ids", frozen_row_ids.to(dtype=torch.int64))
            # frozen_row_values is ALSO kept on the module (not just applied
            # once at init) so callers (run_deltanet_rd.py's train() loop)
            # can re-pin post-optimizer-step via model.frozen_row_ids/
            # model.frozen_row_values alone, without threading the values
            # tensor through train()'s own signature -- same
            # self-sufficiency the buffer-row pinning already has.
            self.register_buffer("frozen_row_values", frozen_row_values.to(dtype=torch.float32))
            pin_rows_(self.embed, self.frozen_row_ids, self.frozen_row_values)
        else:
            self.frozen_row_ids = None
            self.frozen_row_values = None

        # -- arm (i-strong), sec 4.4: surgical k_eff pinning (bypasses
        # W_k/k_conv1d entirely at write + query positions for the
        # entities this lookup covers). Table is DENSE (vocab_size_total x
        # d_state) for a trivial gather-by-id; cheap at this vocab/d_state
        # scale (~12MB at vocab=50259, d_state=64, fp32).
        if strong_pin_ids is not None:
            assert strong_pin_values is not None and strong_pin_values.shape[-1] == d_state
            strong_pin_ids = strong_pin_ids.detach().cpu()
            strong_pin_values = strong_pin_values.detach().cpu()
            table = torch.zeros(vocab_size_total, d_state)
            table[strong_pin_ids] = F.normalize(strong_pin_values, dim=-1)
            self.register_buffer("strong_pin_table", table)
            self.register_buffer("strong_pin_ids", strong_pin_ids.to(dtype=torch.int64))
            self.strong_pin_active = True
        else:
            self.strong_pin_active = False

        # -- F-geo-2, sec 5.5/BLOCK-3: shared ZCA whitening on the key path --
        self.zca = ZCAWhiten(d_state) if use_zca else None

        # -- F-geo-3, sec 14.2: per-episode differentiable key orthogonalization --
        self.geo3_active = geo3_active
        self.geo3_n_iter = geo3_n_iter
        self.geo3_resid_tol = geo3_resid_tol
        # sec 14.4's per-step fallback-triggered logging build item: bind()
        # writes these two attributes every time geo3_active is True (side
        # channel, NOT part of bind()'s return tuple -- sec 14.2 pins bind()
        # to the SAME 3/4-tuple contract every existing caller expects, so
        # the fallback flag/residual cannot be a 5th return value without
        # breaking that contract; train()/evaluate_pool() read these
        # attributes off the model right after each bind() call instead).
        self.geo3_last_fallback_triggered: bool | None = None
        self.geo3_last_resid: torch.Tensor | None = None
        # KEY_ANCHORING_DESIGN.md sec 3.1/sec 4's item-5 instrument symmetry:
        # the PRE-Newton-Schulz raw gathered key, stashed EVERY time
        # geo3_active fires (regardless of anchor_active) -- the evidentiary
        # object for bare-geo3/candidate-(c) drift measurement
        # (key_anchoring.measure_drift's pre_ns_attr="geo3_last_k_eff_raw"),
        # a passive read-only side channel added under the SAME "side
        # channel, not part of bind()'s return tuple" discipline as the two
        # attributes just above -- purely additive, no behavior change.
        self.geo3_last_k_eff_raw: torch.Tensor | None = None

        # -- KEY_ANCHORING_DESIGN.md sec 2.2, candidate (d): trainable
        # per-entity anchor table + masked gather/scatter blend --
        assert not (anchor_active and self.strong_pin_active), \
            "anchor_active and strong_pin_active are MUTUALLY EXCLUSIVE (sec 2.2, mirroring the " \
            "geo3/strong_pin exclusivity above): both replace k_eff_items at write time"
        assert not anchor_active or geo3_active, \
            "anchor_active REQUIRES geo3_active (sec 2.2): anchoring is defined as a modification " \
            "to geo3's own write path, not a replacement for it"
        self.anchor_active = anchor_active
        self.anchor_lambda_mode = anchor_lambda_mode
        # sec 2.2/sec 3.1: the PRE-Newton-Schulz blended key, stashed EVERY
        # bind() call while anchor_active -- item 5's evidentiary quantity
        # for candidate (d) (key_anchoring.measure_drift's
        # pre_ns_attr="anchor_last_k_blend_raw"). Same side-channel
        # discipline as geo3_last_resid/geo3_last_k_eff_raw above.
        self.anchor_last_k_blend_raw: torch.Tensor | None = None
        if anchor_active:
            assert anchor_lambda_mode in ("learned", "fixed", "learned_per_entity"), \
                f"anchor_lambda_mode must be 'learned', 'fixed', or 'learned_per_entity', " \
                f"got {anchor_lambda_mode!r}"
            assert anchor_train_ids is not None and anchor_train_ids.numel() > 0, \
                "anchor_active requires anchor_train_ids (the trained-entity vocab ids, e.g. " \
                "pools.train_name_ids) -- the frame-potential init and anchor_trained_mask are " \
                "both built over exactly this id set"
            anchor_train_ids = anchor_train_ids.detach().cpu().to(dtype=torch.int64)
            n_train = anchor_train_ids.numel()
            assert n_train <= d_state * 4, \
                f"sanity guard: {n_train} trained entities is implausibly large relative to " \
                f"d_state={d_state} for the frame-potential init (expected ~107 at d_state=64, " \
                f"sec 2.2) -- check anchor_train_ids before proceeding"
            assert anchor_table_init_mode in ("frame_potential", "random_unit_rows"), \
                f"anchor_table_init_mode must be 'frame_potential' or 'random_unit_rows', " \
                f"got {anchor_table_init_mode!r}"
            seed = anchor_init_seed if anchor_init_seed is not None else ANCHOR_INIT_SEED
            if anchor_table_init_mode == "frame_potential":
                init_table = frame_potential_init(n_train, d_state, seed=seed)   # (n_train, d_state)
                cond = raw_table_conditioning(init_table)
                assert cond["sigma_ratio_pass"] and cond["max_abs_cos_pass"], (
                    f"anchor init FAILED its own Gate-2 construction legs at model-build time "
                    f"(sigma_ratio={cond['sigma_ratio']:.4f}, max_abs_cos={cond['max_abs_cos']:.4f}) -- "
                    f"this should never happen for the registered frame-potential recipe; re-run "
                    f"gate2_construction_test.py before proceeding")
            else:
                # candidate (e), sec 10.13: NO frame-potential descent, NO Gate-2
                # construction check -- the whole point of this arm is a table
                # that carries no optimized geometric structure at all; gating
                # it on Gate 2 (a check FOR the frame-potential property) would
                # be a category error the same way sec 11.4.3 already flagged
                # for i-strong-as-ceiling-validator.
                init_table = random_unit_rows_init(n_train, d_state, seed=seed)
            anchor_table_full = torch.zeros(vocab_size_total, d_state)
            anchor_table_full[anchor_train_ids] = init_table
            self.anchor_table = nn.Embedding(vocab_size_total, d_state)
            with torch.no_grad():
                self.anchor_table.weight.copy_(anchor_table_full)
            self.anchor_table_frozen = bool(anchor_table_frozen)
            self.anchor_table_init_mode = anchor_table_init_mode
            if self.anchor_table_frozen:
                # candidate (e)'s "frozen" half: the trained-row block never
                # receives a gradient (held-out rows already never do, sec
                # 3.3's C17 bypass -- this flag only changes the TRAINED rows'
                # own status). Smoke-checked directly (no grad reaches
                # anchor_table.weight after a real backward pass) rather than
                # merely asserted here.
                self.anchor_table.weight.requires_grad_(False)
            trained_mask = torch.zeros(vocab_size_total, dtype=torch.bool)
            trained_mask[anchor_train_ids] = True
            self.register_buffer("anchor_trained_mask", trained_mask)
            self.register_buffer("anchor_train_ids_buf", anchor_train_ids)
            if anchor_lambda_mode == "learned":
                # sigmoid(raw_param), init raw_param=0.0 -> lambda=0.5 (sec 2.2)
                self.anchor_lambda_raw = nn.Parameter(torch.tensor(0.0))
                self.anchor_lambda_fixed_value = None
                self.anchor_lambda_table = None
            elif anchor_lambda_mode == "learned_per_entity":
                # KEY_ANCHORING_DESIGN.md sec 10.5.1, candidate (d'): a
                # per-entity table replacing the single scalar
                # anchor_lambda_raw -- SAME sigmoid(raw)->[0,1] parameterization,
                # SAME init (raw=0 -> lambda_e=0.5 for every entity, so any
                # divergence in the final lambda_e distribution is attributable
                # to training, not a different starting condition). Dense over
                # the whole vocab (mirrors anchor_table's own dense-but-masked
                # convention) -- held-out rows are never read in either
                # direction by the t_idx-gathered blend (key_anchoring.
                # anchor_blend_gather_scatter_per_entity), same M1 bypass.
                self.anchor_lambda_raw = None
                self.anchor_lambda_fixed_value = None
                self.anchor_lambda_table = nn.Embedding(vocab_size_total, 1)
                with torch.no_grad():
                    self.anchor_lambda_table.weight.zero_()
            else:
                assert anchor_lambda_fixed is not None, \
                    "anchor_lambda_mode='fixed' requires anchor_lambda_fixed (the pre-registered " \
                    "grid value, e.g. one of {0.3, 0.6, 0.9}, sec 2.2)"
                self.anchor_lambda_raw = None
                self.anchor_lambda_table = None
                self.register_buffer("anchor_lambda_fixed_value",
                                       torch.tensor(float(anchor_lambda_fixed)))
        else:
            self.anchor_table = None
            self.anchor_trained_mask = None
            self.anchor_train_ids_buf = None
            self.anchor_lambda_raw = None
            self.anchor_lambda_fixed_value = None
            self.anchor_lambda_table = None
            self.anchor_table_frozen = False
            self.anchor_table_init_mode = anchor_table_init_mode

    def anchor_lambda(self) -> torch.Tensor:
        """Current (scalar) lambda value: sigmoid(raw_param) for 'learned'
        mode, the fixed grid value for 'fixed' mode (sec 2.2). NOT valid
        for 'learned_per_entity' (candidate (d'), sec 10.5.1) -- there is
        no single scalar; bind()'s anchor branch calls
        key_anchoring.anchor_blend_gather_scatter_per_entity directly with
        self.anchor_lambda_table.weight instead."""
        assert self.anchor_active, "anchor_lambda() called but anchor_active is False"
        assert self.anchor_lambda_mode != "learned_per_entity", \
            "anchor_lambda() has no single scalar value under anchor_lambda_mode='learned_per_entity' " \
            "(candidate (d'), sec 10.5.1) -- use anchor_lambda_table.weight directly"
        if self.anchor_lambda_mode == "learned":
            return torch.sigmoid(self.anchor_lambda_raw)
        return self.anchor_lambda_fixed_value

    # -- shared feature path (embedding -> proj -> causal conv), no recurrence --

    def _zca_apply(self, k_conv: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """sec 13 clarification #2: ZCA sits POST-CONV, BEFORE L2-normalize.
        content_mask excludes BUFFER positions only (BLOCK-3(a)); the EMA
        update happens ONLY from this (bind()'s) call site -- see
        ZCAWhiten's own docstring for the documented single-call-site
        scoping deviation."""
        if self.zca is None:
            return k_conv
        content_mask = token_ids != self.buffer_id
        return self.zca(k_conv, content_mask=content_mask)

    def _kv_over_sequence(self, token_ids: torch.Tensor):
        """Full-sequence k_eff/v_eff PRE-l2norm (l2norm is applied to k
        separately where needed -- see bind()/effective_key_window() -- v is
        never l2-normalized, matching use_qk_l2norm_in_kernel's own scope:
        q/k only). Returns (k_conv, v_conv, x) each (B,T,*). k_conv passes
        through the ZCA whitening layer here (if active, sec 5.5) -- this
        is the SOLE site that drives the ZCA module's running statistics
        (see _zca_apply's docstring)."""
        x = self.embed(token_ids)
        k_conv, _ = self.k_conv1d(self.k_proj(x))
        k_conv = self._zca_apply(k_conv, token_ids)
        v_conv, _ = self.v_conv1d(self.v_proj(x))
        return k_conv, v_conv, x

    def effective_key_window(self, window_token_ids: torch.Tensor) -> torch.Tensor:
        """QUERY-time / self-query k_eff extraction (section 5.2: "q_eff is
        extracted... through the model's own embedding -> conv -> W_k
        path"). window_token_ids: (N, L), L = query_len = buf_len+3,
        structured [BUFFER..., KEY, REL, <Q-or-VALUE>]. Returns the
        L2-normalized k_eff at the LAST position only, (N, d_state). NEVER
        calls chunk_delta_rule -- "the query span passes through the
        feature path only, never the recurrence" (section 5.2).

        Arm (i-strong) override (sec 4.4): when strong_pin_active, bypasses
        embed/k_proj/k_conv1d entirely and returns the fixed per-entity
        lookup keyed by the window's KEY token (index -3 in every window
        this harness builds -- grammar_rd.py's query_tokens AND
        self_query_tokens both share the [buf...,KEY,REL,<Q>] shape).
        Entities outside the pinned pool would silently read an
        all-zero table row; the caller (run_deltanet_rd.py) is responsible
        for restricting the active entity pools to strong_pin_ids' own
        coverage when this arm is active (sec 4.4's "reduced, dedicated
        pool" -- enforced structurally there, not re-checked per call
        here for hot-path cost reasons).

        Arm F-geo-2 (ZCA, sec 5.5): applied post-conv, pre-normalize, same
        insertion point as bind()'s -- this call site TRANSFORMS with the
        current cached W_zca but never updates its running statistics
        (ZCAWhiten's documented single-call-site scoping)."""
        if self.strong_pin_active:
            key_tok = window_token_ids[:, -3]
            return F.normalize(self.strong_pin_table[key_tok], dim=-1)
        x = self.embed(window_token_ids)
        k_conv, _ = self.k_conv1d(self.k_proj(x))
        if self.zca is not None:
            k_conv = self.zca(k_conv, content_mask=None)     # transform-only, no stats update
        return F.normalize(k_conv[:, -1, :], dim=-1)

    # -- BIND (two-kernel-call split step 1-2, section 5.1) --

    def bind(self, batch: dict, force_rank_k: int | None = None, return_beta: bool = False):
        """Runs chunk_delta_rule over the BIND phase ONLY (queries never
        enter this call -- the state-freeze is structural by construction,
        matching model_dn.py's own discipline), then truncates ONCE if
        force_rank_k is given, then C15-asserts. Returns
        (S_T (B,d_state,d_state), k_eff_items (B,K,d_state) L2-normalized,
        v_eff_items (B,K,d_state) raw) -- or a 4-tuple with beta_items
        (B,K) appended when return_beta=True (see below).

        Two kernel-boundary conventions, both F15-LM-driven:
          (1) sequences shorter than _MIN_KERNEL_T are padded with trailing
              BUFFER tokens (state-neutral: beta=0 there -- see the
              _MIN_KERNEL_T comment for the reproduced short-T backward
              crash this avoids);
          (2) k is L2-normalized HERE (zero-safe F.normalize -- deep buffer/
              pad positions have exactly-zero conv output, and the harness
              must not rely on the kernel-internal l2norm's behavior on
              zero rows), with use_qk_l2norm_in_kernel=False. This also
              makes k_eff_items EXACTLY the rows the kernel consumed --
              C16's object is the literal kernel input, not an
              approximation of it.

        return_beta (DELTANET_RD_EXACTNESS_DESIGN.md sec 3.2's beta-dump
        build item): when True, ALSO returns beta_items -- beta gathered
        at item_pos, EXACTLY like k_eff_items/v_eff_items (a 4th return
        value). Default False preserves the EXACT pre-extension 3-tuple
        return unchanged -- every pre-existing call site (forward(),
        evaluate_pool via forward(), train() via forward()) never passes
        this flag, so the training hot path's compute graph is untouched
        byte-for-byte (sec 13/CLAUDE.md's regression-check discipline;
        Wave -1 smoke: bitwise-identical loss with the flag on vs off on
        a fixed seed/batch, run_deltanet_rd.py --smoke)."""
        token_ids = batch["token_ids"]
        beta_mask = batch["beta_mask"]
        item_pos = batch["item_pos"]

        B, T = token_ids.shape
        if T < _MIN_KERNEL_T:
            pad_len = _MIN_KERNEL_T - T
            token_ids = torch.cat([token_ids, torch.full((B, pad_len), self.buffer_id,
                                                          dtype=token_ids.dtype,
                                                          device=token_ids.device)], dim=1)
            beta_mask = torch.cat([beta_mask, torch.zeros(B, pad_len, dtype=beta_mask.dtype,
                                                           device=beta_mask.device)], dim=1)

        k_conv, v_conv, x = self._kv_over_sequence(token_ids)
        beta_logit = torch.sigmoid(self.W_beta(x)).squeeze(-1)          # (B,T), from RAW pre-conv x
        beta = beta_logit * beta_mask                                    # C9/R2-3: hard 0 off VALUE positions

        if self.strong_pin_active:
            # Arm (i-strong), sec 4.4: bypass k_proj/k_conv1d entirely at
            # the K write positions -- k_eff_items[j] := u_{key_ids[j]}
            # (the FIXED per-entity lookup), scattered into a full-length
            # k_norm tensor for the kernel-call API. Off-write positions
            # are architecturally irrelevant regardless of their k value:
            # beta=0 there zeroes the ENTIRE delta-rule update term
            # (kernel_state_design_layout's own math), so leaving them
            # zero is EXACT, not an approximation.
            key_ids = batch["key_ids"]
            k_eff_items = F.normalize(self.strong_pin_table[key_ids], dim=-1)   # (B,K,d_state)
            k_norm = torch.zeros(B, token_ids.shape[1], self.d_state,
                                  device=token_ids.device, dtype=v_conv.dtype)
            k_norm.scatter_(1, item_pos.unsqueeze(-1).expand(-1, -1, self.d_state),
                             k_eff_items.to(k_norm.dtype))
        elif self.geo3_active:
            # F-geo-3, sec 14.2: THIRD branch. k_norm_raw is the existing
            # learned path, UNCHANGED up to here -- F-geo-3 does NOT bypass
            # k_proj/k_conv1d/embed (unlike i-strong); it operates strictly
            # downstream of the model's OWN learned k_conv.
            k_norm_raw = F.normalize(k_conv, dim=-1)
            k_eff_raw = _gather_at(k_norm_raw, item_pos)     # (B,K,d) -- this episode's raw item keys
            self.geo3_last_k_eff_raw = k_eff_raw.detach()    # sec 3.1 item-5 instrument symmetry (bare
                                                               # geo3 / candidate (c)'s pre-NS evidentiary
                                                               # object) -- side channel, always stashed
                                                               # while geo3_active, independent of anchor_active
            ns_input = k_eff_raw
            if self.anchor_active:
                # KEY_ANCHORING_DESIGN.md sec 2.2, candidate (d): masked
                # gather/scatter blend, INSERTED PRE-NEWTON-SCHULZ -- the
                # masked blend touches TRAINED-entity rows ONLY; held-out
                # rows are a bit-exact clone (sec 3.3's C17 bypass). The
                # downstream geo3_orthogonalize_logged call below is
                # UNCHANGED (sec 2.2: "the same single Newton-Schulz pass
                # geo3 already pays for -- no second orthogonalization").
                key_ids = batch["key_ids"]
                if self.anchor_lambda_mode == "learned_per_entity":
                    # KEY_ANCHORING_DESIGN.md sec 10.5.1, candidate (d'): the
                    # ONLY architectural difference from candidate (d) below --
                    # a per-entity lambda_e gathered inside the blend function
                    # itself, replacing the single scalar. Same masked
                    # gather/scatter/held-out-bypass pattern, same insertion
                    # site, same downstream Newton-Schulz call.
                    k_blend_raw = anchor_blend_gather_scatter_per_entity(
                        k_eff_raw, self.anchor_table.weight, self.anchor_trained_mask, key_ids,
                        self.anchor_lambda_table.weight)
                else:
                    lam = self.anchor_lambda()
                    k_blend_raw = anchor_blend_gather_scatter(
                        k_eff_raw, self.anchor_table.weight, self.anchor_trained_mask, key_ids, lam)
                self.anchor_last_k_blend_raw = k_blend_raw.detach()   # sec 3.1 item-5 pre-NS side channel
                ns_input = k_blend_raw
            k_eff_items, fallback_triggered, resid = geo3_orthogonalize_logged(
                ns_input, n_iter=self.geo3_n_iter, resid_tol=self.geo3_resid_tol)  # (B,K,d), Q Q^T ~= I_K
            self.geo3_last_fallback_triggered = fallback_triggered           # sec 14.4 logging side channel
            self.geo3_last_resid = resid.detach()
            k_norm = k_norm_raw.clone()                      # non-write positions: the model's own raw
                                                               # conv output, architecturally INERT
                                                               # (beta=0 there -- kernel_state_design_layout's
                                                               # own math zeroes the entire update term)
            k_norm.scatter_(1, item_pos.unsqueeze(-1).expand(-1, -1, self.d_state),
                             k_eff_items.to(k_norm.dtype))    # substitute AT item_pos ONLY -- IDENTICAL
                                                               # scatter pattern to arm (i-strong), above
        else:
            k_norm = F.normalize(k_conv, dim=-1)                             # zero-safe (0-row -> 0-row)
            k_eff_items = _gather_at(k_norm, item_pos)                       # EXACTLY the kernel's input rows

        # Kernel call + fla[K,V] -> design[V,K] layout reconciliation live
        # in ONE helper (kernel_state_design_layout -- audit FATAL-0; see
        # its docstring for the layout facts and the measured consequence
        # of getting this wrong). S_T is in the DESIGN convention here:
        # S @ k retrieves, matching apply_state_power/effective_ideal_S.
        S_T = kernel_state_design_layout(k_norm, v_conv, beta)           # (B,d_state,d_state)

        if force_rank_k is not None and force_rank_k > 0:
            # ONE truncation per forward pass, impl selectable (deviation
            # #6's fallback, activated by Wave -1's eigh forward crashes).
            # A LinAlgError (eigh non-convergence -- the Wave -1 failure
            # mode) gets ONE retry with a +1e-6 diagonal jitter; if that
            # also fails, TruncationError propagates to the caller's
            # skip-step / skip-batch accounting rather than killing the run.
            trunc_fn = TRUNC_IMPLS[self.trunc_impl]
            try:
                S_T = trunc_fn(S_T, force_rank_k)
            except _LinAlgError as e:
                try:
                    eye = torch.eye(S_T.shape[-1], device=S_T.device, dtype=S_T.dtype)
                    S_T = trunc_fn(S_T + 1e-6 * eye, force_rank_k)
                except _LinAlgError as e2:
                    raise TruncationError(
                        f"{self.trunc_impl} truncation failed to converge even after the "
                        f"one-shot +1e-6 diagonal-jitter retry") from e2
            assert_rank_le(S_T, force_rank_k)                            # C15, exact threshold, both impls

        v_eff_items = _gather_at(v_conv, item_pos)                       # raw (v is never l2-normed)
        if return_beta:
            beta_items = _gather_at(beta.unsqueeze(-1), item_pos).squeeze(-1)   # (B,K), EXACTLY like k_eff_items
            return S_T, k_eff_items, v_eff_items, beta_items
        return S_T, k_eff_items, v_eff_items

    def readout(self, S_T: torch.Tensor, query_tokens: torch.Tensor,
                hops: torch.Tensor, k_eff_items: torch.Tensor | None = None,
                a_slot: torch.Tensor | None = None) -> torch.Tensor:
        """Pinned external readout (section 5.4): pure function of (S_T,
        query_tokens, hops) plus the SAME k_proj/k_conv1d/embed weights BIND
        already used -- introduces zero parameters beyond what bind() uses.

        k_eff_items/a_slot (DELTANET_RD_EXACTNESS_DESIGN.md sec 14.2, F-geo-3
        ONLY): optional, default None -- default behavior (both None) is
        BYTE-IDENTICAL to the pre-extension code above. When self.geo3_active,
        BOTH are required: effective_key_window() CANNOT recompute the
        correct (per-episode-JOINT) orthogonalized value from a query window
        in isolation (sec 14.2) -- the orthogonalizing map is a function of
        the K keys that co-occur in THIS episode, not of any one key alone.
        The query-time value is instead a DIRECT gather from bind()'s own
        k_eff_items at a_slot -- the literal tensor bind() computed, NEVER
        recomputed via embed/k_proj/k_conv1d."""
        if self.geo3_active:
            assert k_eff_items is not None and a_slot is not None, (
                "F-geo-3 requires bind()'s own k_eff_items and the batch's "
                "a_slot at query time -- effective_key_window() CANNOT "
                "recompute the correct (per-episode-joint) value from a "
                "window in isolation (sec 14.2)")
            d = k_eff_items.shape[-1]
            idx = a_slot.unsqueeze(-1).expand(-1, -1, d)
            q_eff = torch.gather(k_eff_items, 1, idx)     # (B,Q,d) -- DIRECT reuse of bind()'s own
                                                            # orthogonalized value; NEVER recomputed
                                                            # via embed/k_proj/k_conv1d
        else:
            B, Q, L = query_tokens.shape                  # unchanged, existing path
            q_eff = self.effective_key_window(query_tokens.reshape(B * Q, L)).reshape(B, Q, -1)
        return apply_state_power(S_T, q_eff, hops)

    def effective_ideal_S(self, k_eff_items: torch.Tensor, v_eff_items: torch.Tensor) -> torch.Tensor:
        """Architecture-native ideal (section 3.6 of the synthetic design,
        transplanted): sum_j v_eff_j @ k_eff_j^T, built from THIS model's
        CURRENT effective (post-conv) keys/values -- the correct reference
        for entity_subspace_rank's SVD-derived K-dim subspace."""
        return torch.einsum("bki,bkj->bij", v_eff_items, k_eff_items)

    def forward(self, batch: dict, force_rank_k: int | None = None):
        """Returns (pred, targets, S_T, k_eff_items, v_eff_items). targets
        are gathered from v_eff_items at the CLAUSE whose VALUE token is the
        answer entity (target_clause_index -- mini-audit FATAL fix,
        2026-07-03): each entity appears as VALUE in exactly one clause (the
        K-cycle is a bijection), so v_eff_items[pi^{-1}(tgt_slot)] IS the
        answer entity's canonical value-side representation under the
        model's OWN current W_v. The PRE-FIX code gathered
        v_eff_items[tgt_slot] directly -- but clause i's VALUE token is
        entity pi(i), so that scored entity pi^{h+1}(a), ONE HOP PAST the
        queried answer pi^h(a): the mini-audit verified a 100% grammar-level
        mismatch at every hop as-built (binding was UNSATISFIABLE as
        specced, making learned-target collapse the accessible optimum) and
        100% match with this fix. Corrects train loss AND eval scoring in
        this one site; grammar_rd._self_test carries the scored-token ==
        answer-entity assert with a negative control on the old index."""
        S_T, k_eff_items, v_eff_items = self.bind(batch, force_rank_k=force_rank_k)
        pred = self.readout(S_T, batch["query_tokens"], batch["hops"],
                             k_eff_items=k_eff_items if self.geo3_active else None,
                             a_slot=batch.get("a_slot") if self.geo3_active else None)
        d = v_eff_items.shape[-1]
        tgt_clause = target_clause_index(batch["succ"], batch["tgt_slot"])
        tgt_idx = tgt_clause.unsqueeze(-1).expand(-1, -1, d)
        targets = torch.gather(v_eff_items, 1, tgt_idx)
        return pred, targets, S_T, k_eff_items, v_eff_items


# ---------------------------------------------------------------------------
# Self-test (part of run_deltanet_rd.py --smoke). REQUIRES CUDA (chunk_delta_rule
# is a Triton kernel; there is no CPU fallback -- unlike deltanet_core.py's
# pure-PyTorch recurrence, which R2-3 deliberately does NOT use on the
# training hot path here). Also requires a real GPT-2 tokenizer (grammar_rd).
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import grammar_rd as grd
    import geo3_simulator as g3sim   # sec 14.6: reuse the registered simulator's synthetic-input
                                       # generators (make_near_collinear/make_adversarial_duplicates)
                                       # for [model 15] below -- "already committed, import, do NOT
                                       # reimplement"

    assert torch.cuda.is_available(), \
        "model_rd's self-test requires CUDA -- chunk_delta_rule is a Triton kernel with no CPU path"
    torch.manual_seed(0)
    DEV = "cuda"

    tokenizer = grd.load_gpt2_tokenizer()
    pools, _ = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    pools = pools.to(DEV)

    print("[model 1] forward / backward / grad-finite, H=1 single layer (C11), no W_q")
    cfg = grd.DeltaNetRDTaskConfig(K=8, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    gen = torch.Generator(device=DEV).manual_seed(1)
    # d_state=64, NOT a smaller "mini" value: d_state must be in the
    # measured-safe head-dim set (_SAFE_D_STATE) -- d_state=16 was the
    # F15-LM round-1/2 crash trigger. d_model=64 keeps the test light.
    model = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64,
                             conv_size=cfg.conv_size, buffer_id=pools.buffer_id).to(DEV)
    b = grd.sample_batch_rd(cfg, 16, gen, hop_set=cfg.H_train, pools=pools, device=DEV)
    pred, targets, S_T, k_eff_items, v_eff_items = model(b)
    assert pred.shape == targets.shape == (16, cfg.queries, 64)
    assert not torch.isnan(pred).any() and not torch.isnan(targets).any()
    loss = (1.0 - F.cosine_similarity(pred, targets, dim=-1)).mean()
    loss.backward()
    n_grad_none = [name for name, p in model.named_parameters() if p.grad is None]
    assert not n_grad_none, f"no grad for: {n_grad_none}"
    for name, p in model.named_parameters():
        assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), f"bad grad {name}"
    print(f"  forward {tuple(pred.shape)}, loss {loss.item():.4f}, ALL params (incl. embed) have finite grad")

    print("\n[model 2] buffer row pinning: exact zero at init, embed reachable at other ids")
    assert torch.equal(model.embed.weight[pools.buffer_id],
                        torch.zeros(model.d_model, device=DEV))
    assert model.embed.weight[pools.query_id].abs().sum().item() > 0, \
        "query_id row should be a normal LEARNED (nonzero-init) row, not pinned"
    print("  buffer row exactly zero; query-marker row is a normal learned row")

    print("\n[model 3] param surface: no W_q / q_conv1d exist on the module at all; "
          "unsafe-d_state guard has teeth")
    param_names = {n for n, _ in model.named_parameters()}
    assert not any("q_proj" in n or "q_conv" in n for n in param_names), \
        f"unexpected q-side parameters found: {param_names}"
    guard_raised = False
    try:
        DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=16,
                         conv_size=cfg.conv_size, buffer_id=pools.buffer_id)
    except AssertionError:
        guard_raised = True
    assert guard_raised, "_SAFE_D_STATE guard FAILED to reject d_state=16 (the measured crash trigger)"
    print(f"  confirmed no q-side parameters; params = {sorted(n.split('.')[0] for n in param_names)}; "
          f"d_state=16 correctly REJECTED by the safe-head-dim guard")

    print("\n[model 4] C15 + force_rank_k, BOTH --trunc-impl variants (deviation #6's fallback, "
          "Wave -1 eigh crash): constrains AND asserts rank; grads finite through each; "
          "negative test has teeth")
    for impl in ("eigh", "svd_lowrank"):
        m_i = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64,
                               conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                               trunc_impl=impl).to(DEV)
        m_i.load_state_dict(model.state_dict())              # identical weights, only impl differs
        m_i.zero_grad()
        pred_k, tgt_k, S_k, _, _ = m_i(b, force_rank_k=3)
        assert_rank_le(S_k, 3)
        (1.0 - F.cosine_similarity(pred_k, tgt_k, dim=-1)).mean().backward()
        for name, p in m_i.named_parameters():
            assert p.grad is not None and not torch.isnan(p.grad).any(), \
                f"[{impl}] bad/missing grad {name} through the force-rank path"
        print(f"  [{impl}] force_rank_k=3 -> rank<=3 (asserted); every param has a finite grad")
    neg_raised = False
    try:
        rank4 = (torch.randn(1, 5, 2, device=DEV) @ torch.randn(1, 2, 5, device=DEV)
                 + torch.randn(1, 5, 2, device=DEV) @ torch.randn(1, 2, 5, device=DEV))
        assert_rank_le(rank4, 2)
    except AssertionError:
        neg_raised = True
    assert neg_raised, "C15 negative test FAILED to fire on a known higher-rank matrix"
    print("  C15 negative test on a rank-4 matrix vs k=2 correctly fires")

    print("\n[model 4b] trunc-impl exactness (ported from model_dn.py's own smoke discipline): "
          "eigh and svd_lowrank agree on a WELL-SEPARATED spectrum (10x gap at the truncation "
          "index); C15 holds on the svd_lowrank output")
    torch.manual_seed(7)
    Bx, dx, kx = 6, 12, 4
    Ux, _ = torch.linalg.qr(torch.randn(Bx, dx, dx, device=DEV))
    Vx, _ = torch.linalg.qr(torch.randn(Bx, dx, dx, device=DEV))
    sig = torch.tensor([64., 32., 16., 8.,                       # kept (k=4)
                        .8, .4, .2, .1, .05, .02, .01, .005],    # dropped; 10x gap at the boundary
                       device=DEV)
    Zx = Ux @ torch.diag(sig) @ Vx.transpose(-1, -2)
    Z_eigh = truncate_to_rank(Zx, kx)
    Z_slr = truncate_to_rank_svd_lowrank(Zx, kx)
    max_impl_diff = (Z_eigh - Z_slr).abs().max().item()
    assert torch.allclose(Z_eigh, Z_slr, atol=1e-2), \
        f"impl disagreement on a well-separated spectrum: max abs diff {max_impl_diff:.2e}"
    assert_rank_le(Z_slr, kx)
    print(f"  eigh vs svd_lowrank max abs diff = {max_impl_diff:.2e} (values up to 64); "
          f"C15 holds on the svd_lowrank output")

    print("\n[model 5] blank-out DIRECTION 1 (post-S_T): with S_T detached, the readout's "
          "gradient reaches ONLY the query feature path -- BIND-only embedding rows (the "
          "'.' period token, which never appears in a query window) and the entire VALUE "
          "path (v_proj/v_conv1d) get ZERO gradient; the <Q> marker row gets NONZERO "
          "(sensitivity control). NOTE: a whole-table zero-gradient probe (model_dn.py's "
          "form) is WRONG here by design -- the readout legitimately re-embeds QUERY "
          "tokens through the same shared table (section 5.2's embedding->conv->W_k "
          "query path), so the leak probe must be row/module-level, not table-level.")
    S_g = model.bind(b)[0]
    S_leaf = S_g.detach().clone().requires_grad_(True)
    pred_leaf = model.readout(S_leaf, b["query_tokens"], b["hops"])
    g_embed, g_vproj, g_vconv = torch.autograd.grad(
        pred_leaf.sum(), [model.embed.weight, model.v_proj.weight, model.v_conv1d.weight],
        allow_unused=True)
    assert g_vproj is None or g_vproj.abs().sum().item() == 0.0, \
        "LEAK: the readout's graph touches v_proj -- it must be key-path-only"
    assert g_vconv is None or g_vconv.abs().sum().item() == 0.0, \
        "LEAK: the readout's graph touches v_conv1d -- it must be key-path-only"
    assert g_embed is not None, "readout unexpectedly has NO gradient path to the embedding at all"
    assert g_embed[pools.period_id].abs().sum().item() == 0.0, \
        "LEAK: the BIND-only '.' token's embedding row receives readout gradient outside S_T"
    assert g_embed[pools.query_id].abs().sum().item() > 0.0, \
        "sensitivity control failed: the <Q> marker row gets no gradient -- the probe is vacuous"
    print("  v-path grad: None/zero; BIND-only '.' row grad: zero; <Q> row grad: nonzero "
          "(readout is key-path-only and reaches bindings only through S_T)")

    print("\n[model 6] beta-mask exactness FROM THE TENSOR (R2-3 F15-LM item, checked here too): "
          "beta is EXACTLY zero at every non-VALUE position")
    with torch.no_grad():
        x_full = model.embed(b["token_ids"])
        beta_logit_full = torch.sigmoid(model.W_beta(x_full)).squeeze(-1)
        beta_full = beta_logit_full * b["beta_mask"]
        is_value = torch.zeros_like(b["beta_mask"], dtype=torch.bool)
        is_value.scatter_(1, b["item_pos"], True)
        assert torch.equal(beta_full[~is_value], torch.zeros_like(beta_full[~is_value])), \
            "beta is not EXACTLY zero at a non-VALUE position"
        assert (beta_full[is_value] >= 0).all() and (beta_full[is_value] <= 1).all()
    print("  beta tensor is exactly 0 off VALUE positions; in [0,1] on VALUE positions")

    print("\n[model 7] R2-8 per-item cross-clause leak check: corrupting clause j's real tokens "
          "leaves clause j+1's k_eff/v_eff BIT-IDENTICAL (buf_len >= conv_size-1 isolates)")
    with torch.no_grad():
        k_ref, v_ref, _ = model._kv_over_sequence(b["token_ids"])
        corrupt = b["token_ids"].clone()
        j = 0
        j_key_pos = int(b["item_pos"][0, j].item()) - 2
        for pos in (j_key_pos, j_key_pos + 1, j_key_pos + 2, j_key_pos + 3):   # KEY,REL,VALUE,PERIOD
            corrupt[:, pos] = pools.train_name_ids[0].item()  # some other valid id (structurally arbitrary here)
        k_c, v_c, _ = model._kv_over_sequence(corrupt)
        jp1 = 1
        pos_next = int(b["item_pos"][0, jp1].item())
        assert torch.equal(k_ref[:, pos_next, :], k_c[:, pos_next, :]), \
            "LEAK: corrupting clause 0 changed clause 1's k_eff -- buffer isolation is broken"
        assert torch.equal(v_ref[:, pos_next, :], v_c[:, pos_next, :]), \
            "LEAK: corrupting clause 0 changed clause 1's v_eff -- buffer isolation is broken"
        # sensitivity control: corrupting clause j's OWN value position must change clause j's OWN k_eff/v_eff
        pos_j = int(b["item_pos"][0, j].item())
        assert not torch.equal(k_ref[:, pos_j, :], k_c[:, pos_j, :]), \
            "corrupting clause 0's own tokens did NOT change its own k_eff -- sensitivity control failed"
    print("  clause j+1 k_eff/v_eff BIT-IDENTICAL after corrupting clause j; clause j's own "
          "k_eff DOES change (sensitivity confirmed, the leak check has teeth)")

    print("\n[model 8] entity_subspace_rank + effective_ideal_S: finite, in-range on a real batch")
    with torch.no_grad():
        _, k_eff_items, v_eff_items = model.bind(b)
        s_ideal = model.effective_ideal_S(k_eff_items, v_eff_items)
        S_T2, _, _ = model.bind(b)
        er, sr = entity_subspace_rank(S_T2, s_ideal, cfg.K)
    assert torch.isfinite(er).all() and torch.isfinite(sr).all()
    assert (er >= 1 - 1e-2).all() and (er <= cfg.K + 1e-2).all()
    print(f"  entity_subspace effective_rank mean={er.mean().item():.3f} (K={cfg.K}), finite")

    print("\n[model 9] gram_deviation / salvage_ratio: finite, sane range on a real batch")
    gd = gram_deviation(k_eff_items)
    sv = salvage_ratio(k_eff_items)
    assert torch.isfinite(gd).all() and torch.isfinite(sv).all()
    assert (sv >= 0).all() and (sv <= 1 + 1e-4).all()
    print(f"  key-side gram_deviation mean={gd.mean().item():.4f}, salvage_ratio mean={sv.mean().item():.4f}")

    print("\n[model 10] AUDIT FATAL-0 REGRESSION -- idealized recall through the ACTUAL "
          "kernel+layout code path (kernel_state_design_layout, the same helper bind() "
          "calls): orthonormal keys, distinct values, beta=1 -> apply_state_power "
          "retrieval cosine must be ~1.0; a DELIBERATELY-TRANSPOSED state (fla's raw "
          "[K,V] layout, i.e. the pre-fix bug) must FAIL the same retrieval. The pre-fix "
          "test suite was transpose-blind -- kernel-vs-reference compared like layouts "
          "and no test ever closed the loop from kernel state to retrieval; this one does.")
    with torch.no_grad():
        Br, Dr, Kr = 4, 64, 8
        Tr = _MIN_KERNEL_T                                   # kernel-realistic length
        gen_r = torch.Generator(device=DEV).manual_seed(77)
        # orthonormal keys (QR), unit-norm values from a permutation of an
        # independent orthonormal set (distinct, exactly recoverable)
        keys_r, _ = torch.linalg.qr(torch.randn(Br, Dr, Kr, generator=gen_r, device=DEV))
        keys_r = keys_r.transpose(-1, -2).contiguous()        # (B,K,D) orthonormal rows
        vals_r, _ = torch.linalg.qr(torch.randn(Br, Dr, Kr, generator=gen_r, device=DEV))
        vals_r = vals_r.transpose(-1, -2).contiguous()        # (B,K,D)
        # scatter the K writes into a T=128 sequence: zero k rows / beta=0 elsewhere
        item_pos_r = torch.arange(Kr, device=DEV) * (Tr // Kr)
        k_seq = torch.zeros(Br, Tr, Dr, device=DEV)
        v_seq = torch.zeros(Br, Tr, Dr, device=DEV)
        beta_seq = torch.zeros(Br, Tr, device=DEV)
        k_seq[:, item_pos_r] = keys_r
        v_seq[:, item_pos_r] = vals_r
        beta_seq[:, item_pos_r] = 1.0
        S_design = kernel_state_design_layout(k_seq, v_seq, beta_seq)     # (B,D,D) design layout
        h1 = torch.ones(Br, Kr, dtype=torch.int64, device=DEV)
        recall = apply_state_power(S_design, keys_r, h1)                   # S @ k_j -> v_j
        cos_ok = F.cosine_similarity(recall, vals_r, dim=-1)
        # negative control: the raw fla [K,V] layout (== transposing back) must FAIL
        recall_bad = apply_state_power(S_design.transpose(-1, -2), keys_r, h1)
        cos_bad = F.cosine_similarity(recall_bad, vals_r, dim=-1)
    assert cos_ok.mean().item() > 0.98, \
        f"FATAL-0 REGRESSION: idealized recall through the fixed layout gives mean cos " \
        f"{cos_ok.mean().item():.4f} (expected ~1.0) -- the kernel-state axis convention is wrong AGAIN"
    assert cos_bad.mean().item() < 0.5, \
        f"FATAL-0 regression negative control is VACUOUS: transposed-state recall gives mean cos " \
        f"{cos_bad.mean().item():.4f} (expected far below 1) -- the test cannot detect the bug it exists for"
    print(f"  idealized recall (fixed layout): mean cos = {cos_ok.mean().item():.4f} (~1.0 required); "
          f"transposed negative control: mean cos = {cos_bad.mean().item():.4f} (must be low) -- "
          f"the regression test detects the pre-fix bug in both directions")
    # deviation #6 follow-through: the same idealized recall must survive an
    # svd_lowrank truncation at k=K (the state's true rank is K, so a rank-K
    # truncation is near-lossless) -- verifies [model 10] holds under the
    # fallback impl, not just unforced.
    with torch.no_grad():
        S_trunc = truncate_to_rank_svd_lowrank(S_design, Kr)
        assert_rank_le(S_trunc, Kr)
        cos_trunc = F.cosine_similarity(apply_state_power(S_trunc, keys_r, h1), vals_r, dim=-1)
    assert cos_trunc.mean().item() > 0.98, \
        f"[model 10 / svd_lowrank] recall after rank-K svd_lowrank truncation gives mean cos " \
        f"{cos_trunc.mean().item():.4f} (expected ~1.0)"
    print(f"  recall after svd_lowrank truncation at k=K: mean cos = {cos_trunc.mean().item():.4f} "
          f"(fallback impl preserves the layout/retrieval contract)")

    print("\n[model 11] beta-dump (return_beta): flag on vs off is BITWISE IDENTICAL for "
          "S_T/k_eff_items/v_eff_items on the SAME batch (sec 3.2's Wave -1 smoke: the added "
          "beta gather must not touch the compute graph); beta_items shape/range and manual-"
          "gather agreement")
    with torch.no_grad():
        S_a, k_a, v_a = model.bind(b, force_rank_k=None, return_beta=False)
        S_b, k_b, v_b, beta_items = model.bind(b, force_rank_k=None, return_beta=True)
    assert torch.equal(S_a, S_b), "return_beta=True changed S_T vs return_beta=False"
    assert torch.equal(k_a, k_b), "return_beta=True changed k_eff_items vs return_beta=False"
    assert torch.equal(v_a, v_b), "return_beta=True changed v_eff_items vs return_beta=False"
    assert beta_items.shape == (b["token_ids"].shape[0], cfg.K)
    assert (beta_items >= 0).all() and (beta_items <= 1).all()
    # Manual re-derivation MUST replicate bind()'s own _MIN_KERNEL_T padding
    # before recomputing beta -- embed()/W_beta are position-independent
    # (no conv, no cross-position mixing) so the padded-vs-unpadded VALUES
    # are mathematically identical at every position < T_original, but
    # cuBLAS is not guaranteed BIT-exact across differently-shaped matmuls
    # (T=56 folded into the batch dim vs T=128) -- recomputing on the SAME
    # (B,128) padded shape bind() itself used keeps this an exact, not an
    # approximate, regression check.
    T_orig = b["token_ids"].shape[1]
    if T_orig < _MIN_KERNEL_T:
        pad_len = _MIN_KERNEL_T - T_orig
        tok_padded = torch.cat([b["token_ids"], torch.full((b["token_ids"].shape[0], pad_len),
                                                             model.buffer_id, dtype=b["token_ids"].dtype,
                                                             device=b["token_ids"].device)], dim=1)
        mask_padded = torch.cat([b["beta_mask"], torch.zeros(b["beta_mask"].shape[0], pad_len,
                                                               dtype=b["beta_mask"].dtype,
                                                               device=b["beta_mask"].device)], dim=1)
    else:
        tok_padded, mask_padded = b["token_ids"], b["beta_mask"]
    with torch.no_grad():
        x_full = model.embed(tok_padded)
        beta_logit_full = torch.sigmoid(model.W_beta(x_full)).squeeze(-1)
        beta_full = beta_logit_full * mask_padded
        beta_manual = torch.gather(beta_full, 1, b["item_pos"])
    assert torch.equal(beta_items, beta_manual), "beta_items does not match a manual gather at item_pos"
    print(f"  S_T/k_eff/v_eff bitwise identical with return_beta True vs False; "
          f"beta_items shape {tuple(beta_items.shape)}, range [{beta_items.min().item():.4f}, "
          f"{beta_items.max().item():.4f}], matches a manual gather exactly")

    print("\n[model 12] generalized frozen-row pinning (pin_rows_/zero_rows_grad_, arms "
          "(i)/(ii)/(iv)'s shared mechanism): pinned rows survive a real optimizer step exactly; "
          "a non-frozen row DOES move (sensitivity control)")
    torch.manual_seed(3)
    used_ids = b["entity_ids"].unique()
    assert used_ids.numel() >= 3, "batch needs >=3 distinct entities for this probe"
    frozen_ids = used_ids[:2].to(DEV)
    other_id = used_ids[2].item()
    frozen_vals = F.normalize(torch.randn(2, 64, device=DEV), dim=-1)
    m_frz = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64,
                             conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                             frozen_row_ids=frozen_ids, frozen_row_values=frozen_vals).to(DEV)
    assert torch.allclose(m_frz.embed.weight[frozen_ids], frozen_vals, atol=1e-6), \
        "frozen rows not pinned to the requested values at init"
    before_frozen = m_frz.embed.weight[frozen_ids].clone()
    before_other = m_frz.embed.weight[other_id].clone()
    opt = torch.optim.Adam(m_frz.parameters(), lr=0.1)
    pred_f, tgt_f, _, _, _ = m_frz(b)
    (1.0 - F.cosine_similarity(pred_f, tgt_f, dim=-1)).mean().backward()
    zero_rows_grad_(m_frz.embed, m_frz.frozen_row_ids)
    opt.step()
    pin_rows_(m_frz.embed, m_frz.frozen_row_ids, frozen_vals)
    after_frozen = m_frz.embed.weight[frozen_ids]
    after_other = m_frz.embed.weight[other_id]
    assert torch.equal(after_frozen, before_frozen), "frozen rows moved after an optimizer step"
    assert not torch.allclose(after_other, before_other), \
        "sensitivity control failed: a non-frozen row did not move -- the probe is vacuous"
    print("  frozen rows bit-identical after a real optimizer step; a non-frozen row moved "
          "(sensitivity control confirms the probe has teeth)")

    print("\n[model 13] arm (i-strong) surgical k_eff pin: bind()'s k_eff_items EXACTLY equals "
          "the normalized lookup for the pinned entities; effective_key_window (query path) "
          "agrees; k_proj/k_conv1d receive NO gradient (documented dead-parameter deviation) "
          "while v_proj/W_beta still do")
    torch.manual_seed(4)
    strong_names_train = pools.train_name_ids[:cfg.K].to(DEV)     # a RESTRICTED, K-name pool
    strong_pools = grd.EntityPools(
        vocab_size_base=pools.vocab_size_base, buffer_id=pools.buffer_id, query_id=pools.query_id,
        period_id=pools.period_id, train_name_ids=strong_names_train,
        heldout_name_ids=pools.heldout_name_ids, rel_a_ids=pools.rel_a_ids, rel_b_ids=pools.rel_b_ids,
        vocab_size_total=pools.vocab_size_total)
    pin_vals = F.normalize(torch.randn(cfg.K, 64, generator=gen, device=DEV), dim=-1)
    m_sp = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64,
                            conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                            strong_pin_ids=strong_names_train, strong_pin_values=pin_vals).to(DEV)
    gen_sp = torch.Generator(device=DEV).manual_seed(5)
    b_sp = grd.sample_batch_rd(cfg, 8, gen_sp, hop_set=cfg.H_train, pools=strong_pools, device=DEV)
    with torch.no_grad():
        S_sp, k_sp, v_sp = m_sp.bind(b_sp)
        expected_k = F.normalize(m_sp.strong_pin_table[b_sp["key_ids"]], dim=-1)
    assert torch.allclose(k_sp, expected_k, atol=1e-6), \
        "arm (i-strong) k_eff_items does not exactly match the pinned lookup"
    sq_sp = grd.self_query_tokens(cfg, strong_pools, b_sp["key_ids"], b_sp["rel_id"])
    with torch.no_grad():
        q_sp = m_sp.effective_key_window(sq_sp.reshape(-1, cfg.query_len)).reshape(
            b_sp["key_ids"].shape[0], cfg.K, -1)
    assert torch.allclose(q_sp, expected_k, atol=1e-6), \
        "arm (i-strong) effective_key_window does not match the pinned lookup at the query path"
    m_sp.zero_grad()
    pred_sp, tgt_sp, _, _, _ = m_sp(b_sp)
    (1.0 - F.cosine_similarity(pred_sp, tgt_sp, dim=-1)).mean().backward()
    assert m_sp.k_proj.weight.grad is None or m_sp.k_proj.weight.grad.abs().sum().item() == 0.0, \
        "k_proj unexpectedly received gradient under arm (i-strong) -- the bypass is leaking"
    assert m_sp.v_proj.weight.grad is not None and m_sp.v_proj.weight.grad.abs().sum().item() > 0.0, \
        "v_proj received NO gradient -- the v-path should remain learned under arm (i-strong)"
    assert m_sp.W_beta.weight.grad is not None and m_sp.W_beta.weight.grad.abs().sum().item() > 0.0, \
        "W_beta received NO gradient -- beta should remain learned under arm (i-strong)"
    print(f"  k_eff_items/effective_key_window both EXACTLY match the pinned lookup; "
          f"k_proj/k_conv1d correctly receive zero/no gradient (dead under this arm, documented); "
          f"v_proj/W_beta gradients confirmed nonzero (still learned)")

    print("\n[model 14] F-geo-2 ZCA whitening (Wave -1 smoke, BLOCK-3(c)): finite outputs, "
          "cond(Sigma+eps*I) < 1e6, post-warmup whitened covariance ~I on content positions, "
          "buffer(zero-input) positions stay EXACTLY zero through the transform (the uncentered-"
          "whitening invariant sec 5.5's ZCAWhiten docstring argues for), buffer-only batch "
          "contributes ZERO statistics mass (mask verification)")
    torch.manual_seed(6)
    zca = ZCAWhiten(d_state=64, warmup_updates=500, recompute_every=100).to(DEV)
    zca.train()
    Bz, Tz, Dz = 8, 128, 64
    A = torch.randn(Dz, Dz, device=DEV) * 0.5 + torch.eye(Dz, device=DEV)   # correlated features
    is_buffer = torch.zeros(Bz, Tz, dtype=torch.bool, device=DEV)
    is_buffer[:, ::2] = True
    for _ in range(520):
        raw = torch.randn(Bz, Tz, Dz, device=DEV) @ A.T
        raw = raw.masked_fill(is_buffer.unsqueeze(-1), 0.0)
        out = zca(raw, content_mask=~is_buffer)
        assert torch.isfinite(out).all(), "ZCA output not finite"
        assert torch.equal(out[is_buffer], torch.zeros_like(out[is_buffer])), \
            "buffer (zero-input) positions did NOT map to exact zero through ZCA -- centering leaked"
    assert zca.num_updates.item() == 520, f"expected 520 EMA updates, got {zca.num_updates.item()}"
    cond = zca.cond_number()
    assert cond < 1e6, f"ZCA cond(Sigma+eps*I) = {cond:.3e} >= 1e6"
    with torch.no_grad():
        raw = torch.randn(4096, Dz, device=DEV) @ A.T
        whitened = raw @ zca.W_zca.T
        cov_w = (whitened.T @ whitened) / whitened.shape[0]
        dev = (cov_w - torch.eye(Dz, device=DEV)).norm().item()
    assert dev < 0.1 * Dz, f"post-warmup whitened covariance deviation {dev:.4f} >= 0.1*d_state={0.1*Dz}"
    before_stats = (zca.running_cov.clone(), zca.num_updates.clone())
    zca(torch.zeros(Bz, Tz, Dz, device=DEV),
        content_mask=torch.zeros(Bz, Tz, dtype=torch.bool, device=DEV))
    assert torch.equal(zca.running_cov, before_stats[0]) and zca.num_updates.item() == before_stats[1].item(), \
        "a buffer-only (all-masked-out) batch changed the running statistics -- mask has no teeth"
    print(f"  finite outputs over 520 updates; buffer(zero) positions exactly zero post-transform; "
          f"cond(Sigma+eps*I)={cond:.2f} < 1e6; post-warmup ||Sigma_w-I||_F={dev:.4f} < "
          f"0.1*d_state={0.1*Dz:.1f}; buffer-only batch contributes zero statistics mass (mask verified)")

    print("\n[model 15] F-geo-3 Newton-Schulz core (DELTANET_RD_EXACTNESS_DESIGN.md sec 14.6 Wave -1 "
          "smoke items 1/2/3/6): convergence on realistic (near-collinear, early-training-like) AND "
          "adversarial (near-duplicate-row) inputs, gradient finiteness on both, order-equivariance, "
          "and a QR row-space cross-check -- reuses geo3_simulator's synthetic-input generators "
          "(already committed, imported above, NOT reimplemented -- sec 14.6's own instruction)")
    for K in (16, 32, 48):
        gen15 = torch.Generator(device=DEV).manual_seed(100 + K)
        # item 1 (realistic): pure Newton-Schulz must converge within n_iter=12 on a near-collinear
        # probe (same construction the attack round's own GPU probe already verified clean --
        # geo3_simulator.py's __main__ part (A)). K=48 is the design's own documented hardest cell
        # (sec 14.9 item 3: "least spare dimensionality... the regime Newton-Schulz converges
        # slowest in") -- logged, not hard-asserted, at K=48; hard-asserted at the two MANDATORY
        # Wave-1 cells K=16/32.
        A_real = g3sim.make_near_collinear(64, K, 64, max(2, K // 6), 0.15, gen15, DEV).requires_grad_(True)
        Q_real, resid_real = newton_schulz_orthogonalize(A_real, n_iter=12)
        if K in (16, 32):
            assert (resid_real <= 1e-2).all(), \
                f"K={K}: realistic (near-collinear) input did NOT converge within n_iter=12 " \
                f"(max resid {resid_real.max().item():.4f} > resid_tol=1e-2) -- a MANDATORY Wave-1 cell"
        # item 2: gradient finiteness through the realistic case
        Q_real.sum().backward()
        assert torch.isfinite(A_real.grad).all(), f"K={K}: non-finite gradient on the realistic input"

        # item 1 (adversarial): near-duplicate rows -- geo3_orthogonalize's fallback
        # (_polar_via_eigh, engaged automatically if NS alone doesn't converge) must NEVER crash
        # and must MEASURABLY IMPROVE over doing nothing. MEASURED FINDING (calibrated here, not
        # assumed -- flagged for audit scrutiny): at geo3_simulator's DEFAULT adversarial
        # construction (eps=1e-5, i.e. near-EXACT duplicate pairs, Sigma's corresponding
        # eigenvalues ~1e-7), NEITHER Newton-Schulz(n_iter=12) NOR the eigh fallback
        # (eps_scale=1e-4, ~1000x looser than the true ~1e-7 eigenvalue) drives Gram deviation to
        # near-zero -- both land at the SAME reproducible, K-INVARIANT plateau (~sqrt(3) =~1.732
        # for n_dup_pairs=3, confirmed by direct probe: the 3 duplicate-pair rows land at norm
        # 1/sqrt(2) and mutual Gram ~0.5 instead of 1/0, a genuine "split the difference" fixed
        # point, not a crash/explosion/NaN). This is NOT a code bug -- it is the mathematically
        # expected behavior of ANY fixed, non-adaptive regularization confronting a near-singular
        # (not just near-collinear) input: eps=1e-4 fixed on this file's own analogy to ZCA's
        # population-covariance eps is ~1000x too loose to resolve a ~1e-7-eigenvalue direction,
        # while a much smaller eps (probed separately: 1e-8..1e-12) makes it WORSE (blows up:
        # gram_dev >>1 as eps undershoots) -- there is no single fixed eps that is both safe
        # against blow-up AND resolves an arbitrarily-close duplicate. The safety net: sec 14.10's
        # substitute admission stack already excludes any REAL training run that leans on the
        # fallback at all ("ns_converged_no_fallback") from admissible evidence -- so a step this
        # degraded, if it ever occurs on real data, self-excludes rather than silently
        # contaminating a reported result. Assert only what sec 14.6 item 1 actually requires
        # (triggers, converges to a FINITE fixed point, never crashes, and is a genuine
        # improvement over no correction at all) -- NOT a specific tightness bound, which this
        # measurement shows is not achievable with the current fixed-eps fallback on a
        # sufficiently adversarial input.
        A_adv = g3sim.make_adversarial_duplicates(64, K, 64, gen15, DEV).requires_grad_(True)
        with torch.no_grad():
            _, resid_adv_ns = newton_schulz_orthogonalize(A_adv, n_iter=12)
            gd_raw = gram_deviation(A_adv.detach())
        fallback_would_trigger = bool((resid_adv_ns > 1e-2).any())
        assert fallback_would_trigger, \
            f"K={K}: the default adversarial construction (near-duplicate rows) unexpectedly did " \
            f"NOT trip the fallback trigger -- the probe is vacuous"
        Q_final = geo3_orthogonalize(A_adv, n_iter=12, resid_tol=1e-2)
        final_gd = gram_deviation(Q_final)
        assert torch.isfinite(Q_final).all(), f"K={K}: adversarial case produced non-finite output"
        assert (final_gd < gd_raw).all(), \
            f"K={K}: geo3_orthogonalize's adversarial-case output ({final_gd.max().item():.4f}) is " \
            f"NOT an improvement over the uncorrected raw input ({gd_raw.max().item():.4f})"
        # item 2: gradient finiteness through the (fallback-engaged) adversarial case
        Q_final.sum().backward()
        assert torch.isfinite(A_adv.grad).all(), f"K={K}: non-finite gradient on the adversarial input"
        print(f"  K={K}: realistic resid@12={resid_real.max().item():.2e}"
              f"{' (<=1e-2, mandatory cell)' if K in (16, 32) else ' (logged only, K=48 stretch cell)'}; "
              f"adversarial fallback_triggered={fallback_would_trigger}, raw Gram dev="
              f"{gd_raw.max().item():.3f} -> geo3_orthogonalize Gram dev={final_gd.max().item():.3f} "
              f"(improved, never near-zero for this near-EXACT-duplicate probe -- see comment "
              f"above, flagged for audit); gradients finite on both")

    print("  item 3 (order-equivariance): permuting A's K rows permutes geo3_orthogonalize's output "
          "IDENTICALLY -- closes sec 14.1's equivariance claim (argued-not-measured until this smoke)")
    gen_oe = torch.Generator(device=DEV).manual_seed(999)
    K_oe, d_oe = 24, 64
    A_oe = g3sim.make_near_collinear(32, K_oe, d_oe, 5, 0.2, gen_oe, DEV)
    perm = torch.randperm(K_oe, device=DEV)
    Q_oe = geo3_orthogonalize(A_oe, n_iter=12, resid_tol=1e-2)
    Q_oe_perm_input = geo3_orthogonalize(A_oe[:, perm, :], n_iter=12, resid_tol=1e-2)
    max_oe_diff = (Q_oe[:, perm, :] - Q_oe_perm_input).abs().max().item()
    assert torch.allclose(Q_oe[:, perm, :], Q_oe_perm_input, atol=1e-5), \
        f"geo3_orthogonalize is NOT order-equivariant: permuting input rows did not permute output " \
        f"rows identically (max abs diff {max_oe_diff:.2e})"
    print(f"    max abs diff (permute-then-orth vs orth-then-permute) = {max_oe_diff:.2e} (< 1e-5 required)")

    print("  item 6 (QR row-space cross-check, non-differentiable, torch.no_grad only): "
          "geo3_orthogonalize's converged output spans the SAME row space as torch.linalg.qr(A.T).Q.T "
          "(projector distance, not raw value comparison -- the two methods return DIFFERENT rotations "
          "by design, sec 14.1's Gram-Schmidt/QR rejection)")
    with torch.no_grad():
        gen_qr = torch.Generator(device=DEV).manual_seed(555)
        K_qr, d_qr = 16, 64
        A_qr = F.normalize(torch.randn(8, K_qr, d_qr, generator=gen_qr, device=DEV), dim=-1)
        Q_ns = geo3_orthogonalize(A_qr, n_iter=12, resid_tol=1e-2)
        Q_qr, _ = torch.linalg.qr(A_qr.transpose(-1, -2))     # (B,d,K)
        Q_qr = Q_qr.transpose(-1, -2)                          # (B,K,d): orthonormal rows spanning A's row space
        # projector distance: ||Q_ns^T Q_ns - Q_qr^T Q_qr||_F -- both (d,d) projectors onto the SAME
        # K-dim row space iff this is ~0, invariant to which orthonormal BASIS either method returns
        P_ns = Q_ns.transpose(-1, -2) @ Q_ns
        P_qr = Q_qr.transpose(-1, -2) @ Q_qr
        proj_dist = (P_ns - P_qr).norm(dim=(-2, -1))
    assert (proj_dist < 0.1).all(), \
        f"QR cross-check FAILED: max projector distance {proj_dist.max().item():.4f} >= 0.1 -- " \
        f"Newton-Schulz is not solving the row-space problem QR solves"
    print(f"    max projector distance (Newton-Schulz vs QR row-space) = {proj_dist.max().item():.2e} (< 0.1)")

    print("\n[model 16] F-geo-3 end-to-end (DELTANET_RD_EXACTNESS_DESIGN.md sec 14.6 Wave -1 smoke "
          "items 2/4/8/9/10): bind()-level gradient finiteness (well-conditioned + forced-near-"
          "duplicate THROUGH a real bind() call), self-consistency of the query-path gather, K=16 "
          "padding-boundary scatter, the R2-8 leak invariant RE-SCOPED to raw pre-orthogonalization "
          "k_conv (+ a positive control confirming the post-orth global coupling IS live, by design), "
          "v-path bit-identity/blank-out/gradient-pattern under the new readout mechanics, and the "
          "strong_pin/geo3 mutual-exclusivity guard")
    torch.manual_seed(8)
    m_g3 = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                            buffer_id=pools.buffer_id, geo3_active=True, geo3_n_iter=12,
                            geo3_resid_tol=1e-2).to(DEV)
    assert m_g3.geo3_active and not m_g3.strong_pin_active

    guard_raised_g3 = False
    try:
        DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                         buffer_id=pools.buffer_id, geo3_active=True,
                         strong_pin_ids=pools.train_name_ids[:cfg.K].to(DEV),
                         strong_pin_values=F.normalize(torch.randn(cfg.K, 64, device=DEV), dim=-1))
    except AssertionError:
        guard_raised_g3 = True
    assert guard_raised_g3, \
        "geo3_active + strong_pin_ids simultaneously did NOT raise -- sec 14.2's mutual-exclusivity assert has no teeth"
    print("  strong_pin/geo3 mutual-exclusivity guard has teeth")

    gen_g3 = torch.Generator(device=DEV).manual_seed(9)
    b_g3 = grd.sample_batch_rd(cfg, 16, gen_g3, hop_set=cfg.H_train, pools=pools, device=DEV)

    print("  item 2a: well-conditioned (mid-training-like, 5 real SGD steps) input through bind() -- "
          "all gradients finite, fallback NOT triggered")
    opt_g3 = torch.optim.Adam(m_g3.parameters(), lr=1e-2)
    for _ in range(5):
        pred_wc, tgt_wc, _, _, _ = m_g3(b_g3)
        loss_wc = (1.0 - F.cosine_similarity(pred_wc, tgt_wc, dim=-1)).mean()
        opt_g3.zero_grad()
        loss_wc.backward()
        opt_g3.step()
    m_g3.zero_grad()
    pred_wc, tgt_wc, S_wc, k_wc, v_wc = m_g3(b_g3)
    (1.0 - F.cosine_similarity(pred_wc, tgt_wc, dim=-1)).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m_g3.parameters() if p.grad is not None), \
        "non-finite gradient through bind()/readout() at a well-conditioned (mid-training-like) input"
    assert m_g3.geo3_last_fallback_triggered is False, \
        "well-conditioned input unexpectedly triggered the eigh fallback"
    print(f"    resid={m_g3.geo3_last_resid.max().item():.2e} (fallback NOT triggered, as expected)")

    print("  item 2b: forced-near-duplicate (adversarial) input THROUGH a real bind() call -- "
          "k_proj rank-collapsed to 2 output dims, guaranteeing K>2 raw keys cannot be mutually "
          "orthonormal (a physically-motivated, deterministic way to trip the fallback via the REAL "
          "forward path, not a synthetic stand-in)")
    torch.manual_seed(11)
    m_g3_adv = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                                buffer_id=pools.buffer_id, geo3_active=True, geo3_n_iter=12,
                                geo3_resid_tol=1e-2).to(DEV)
    with torch.no_grad():
        m_g3_adv.k_proj.weight[2:, :] = 0.0
    pred_adv, tgt_adv, S_adv, k_adv, v_adv = m_g3_adv(b_g3)
    (1.0 - F.cosine_similarity(pred_adv, tgt_adv, dim=-1)).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m_g3_adv.parameters()), \
        "non-finite gradient through bind()/readout() at a forced-near-duplicate (adversarial) input"
    print(f"    fallback_triggered={m_g3_adv.geo3_last_fallback_triggered}, "
          f"resid={m_g3_adv.geo3_last_resid.max().item():.2e}; all gradients finite "
          f"(rank-collapsed k_proj forces K={cfg.K} raw keys into a 2-dim subspace -- cannot be "
          f"mutually orthonormal for K>2, so this cell is EXPECTED to trip the fallback)")

    print("  item 4 (self-consistency, the i-strong model-13 analog): readout() with a_slot=arange(K) "
          "reproduces k_eff_items EXACTLY (a triviality check on the indexing, sec 14.6 item 4 -- both "
          "sides are the identical underlying tensor by construction) and the full readout() call "
          "matches apply_state_power(S_T, k_eff_items, hops) directly")
    with torch.no_grad():
        S_T4, k_eff4, v_eff4 = m_g3.bind(b_g3)
        a_slot_self = torch.arange(cfg.K, device=DEV).unsqueeze(0).expand(b_g3["token_ids"].shape[0], -1)
        idx4 = a_slot_self.unsqueeze(-1).expand(-1, -1, m_g3.d_state)
        q_eff_direct = torch.gather(k_eff4, 1, idx4)
        assert torch.equal(q_eff_direct, k_eff4), \
            "self a_slot=arange(K) gather does not reproduce k_eff_items exactly"
        hops_self = torch.ones_like(a_slot_self)
        pred_self = m_g3.readout(S_T4, b_g3["query_tokens"], hops_self, k_eff_items=k_eff4, a_slot=a_slot_self)
        pred_expected = apply_state_power(S_T4, k_eff4, hops_self)
        assert torch.equal(pred_self, pred_expected), \
            "readout(a_slot=arange(K)) does not match apply_state_power(S_T,k_eff_items,hops) directly"
        # tie to grammar_rd's own slot-order invariant ([grammar 3b]): self_query_tokens' slot j
        # window's KEY token is key_ids[:,j] EXACTLY -- confirming self a_slot=arange(K) is the
        # SEMANTICALLY correct "each slot queries its own entity" mapping, not just a vacuous gather
        sq_g3 = grd.self_query_tokens(cfg, pools, b_g3["key_ids"], b_g3["rel_id"])
        assert torch.equal(sq_g3[:, :, -3], b_g3["key_ids"]), \
            "self_query_tokens' slot order does not match key_ids' slot order (semantics broken)"
    print("  self a_slot gather == k_eff_items exactly; readout(a_slot=arange(K)) == "
          "apply_state_power(S_T,k_eff_items,hops) exactly; self_query_tokens slot order confirmed")

    print("  item 8 (padding boundary, K=16 -> T_bind=112, padded to 128): the geo3 write-side "
          "scatter lands at the correct PRE-padding item_pos indices")
    cfg16 = grd.DeltaNetRDTaskConfig(K=16, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    assert cfg16.T_bind == 112 and cfg16.T_bind < _MIN_KERNEL_T
    torch.manual_seed(12)
    m_g3_16 = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg16.conv_size,
                               buffer_id=pools.buffer_id, geo3_active=True).to(DEV)
    gen16 = torch.Generator(device=DEV).manual_seed(13)
    b16 = grd.sample_batch_rd(cfg16, 8, gen16, hop_set=cfg16.H_train, pools=pools, device=DEV)
    assert (b16["item_pos"] < cfg16.T_bind).all(), "item_pos already reaches into the would-be-padded region pre-cat"
    with torch.no_grad():
        S16, k_eff16, v_eff16 = m_g3_16.bind(b16)
        # independent re-derivation: pad token_ids to 128 EXACTLY as bind() does internally, recompute
        # k_norm_raw/k_eff_raw/orthogonalize/scatter by hand, and confirm gathering the HAND-BUILT
        # padded k_norm at item_pos reproduces bind()'s own k_eff_items exactly
        pad_len16 = _MIN_KERNEL_T - cfg16.T_bind
        tok16_padded = torch.cat([b16["token_ids"], torch.full((8, pad_len16), pools.buffer_id,
                                                                 dtype=b16["token_ids"].dtype, device=DEV)], dim=1)
        k_conv16, _, _ = m_g3_16._kv_over_sequence(tok16_padded)
        k_raw16 = F.normalize(k_conv16, dim=-1)
        k_eff_raw16 = _gather_at(k_raw16, b16["item_pos"])
        k_eff_manual16 = geo3_orthogonalize(k_eff_raw16, n_iter=m_g3_16.geo3_n_iter, resid_tol=m_g3_16.geo3_resid_tol)
        k_norm_manual16 = k_raw16.clone()
        k_norm_manual16.scatter_(1, b16["item_pos"].unsqueeze(-1).expand(-1, -1, 64), k_eff_manual16)
        gathered_manual16 = _gather_at(k_norm_manual16, b16["item_pos"])
    assert k_eff16.shape == (8, 16, 64)
    assert torch.allclose(gathered_manual16, k_eff16, atol=1e-5), \
        "geo3's write-side scatter at K=16 (padded T_bind) does NOT land at the correct pre-padding item_pos"
    assert torch.isfinite(S16).all() and torch.isfinite(k_eff16).all()
    print(f"    T_bind=112 padded to 128; item_pos max={b16['item_pos'].max().item()} < 112 (pre-pad range); "
          f"hand-rederived padded scatter matches bind()'s own k_eff_items exactly")

    print("  item 9 (R2-8 RE-SCOPED, sec 14.6): raw pre-orthogonalization k_conv stays LEAK-FREE "
          "(identical to [model 7]'s check, now on a geo3_active model) + a POSITIVE CONTROL: the "
          "POST-orth k_eff_items[j] for j!=i MUST change when clause i is corrupted (global coupling "
          "is intended, not a bug, under F-geo-3)")
    with torch.no_grad():
        k_raw_ref, v_raw_ref, _ = m_g3._kv_over_sequence(b_g3["token_ids"])
        corrupt_g3 = b_g3["token_ids"].clone()
        j0 = 0
        j0_key_pos = int(b_g3["item_pos"][0, j0].item()) - 2
        for pos in (j0_key_pos, j0_key_pos + 1, j0_key_pos + 2, j0_key_pos + 3):
            corrupt_g3[:, pos] = pools.train_name_ids[0].item()
        k_raw_c, v_raw_c, _ = m_g3._kv_over_sequence(corrupt_g3)
        j1 = 1
        pos_j1 = int(b_g3["item_pos"][0, j1].item())
        assert torch.equal(k_raw_ref[:, pos_j1, :], k_raw_c[:, pos_j1, :]), \
            "LEAK (raw k_conv level): corrupting clause 0 changed clause 1's RAW k_conv under geo3_active"
        assert torch.equal(v_raw_ref[:, pos_j1, :], v_raw_c[:, pos_j1, :]), \
            "LEAK (v path): corrupting clause 0 changed clause 1's v_eff under geo3_active (item 10b)"
        pos_j0 = int(b_g3["item_pos"][0, j0].item())
        assert not torch.equal(k_raw_ref[:, pos_j0, :], k_raw_c[:, pos_j0, :]), \
            "sensitivity control failed: corrupting clause 0's own tokens did not change its own raw k_conv"

        # positive control: build FULL batches (ref vs corrupted) and run bind() end-to-end; the
        # POST-orthogonalization k_eff_items row for clause 1 (j1 != corrupted clause 0) MUST differ,
        # since it is now a JOINT function of all K episode keys (sec 14.2) -- the R2-8 invariant is
        # violated BY DESIGN at this level and this asserts that violation is actually live, not vacuous
        b_g3_corrupt = dict(b_g3)
        b_g3_corrupt["token_ids"] = corrupt_g3
        _, k_eff_ref_full, _ = m_g3.bind(b_g3)
        _, k_eff_corrupt_full, _ = m_g3.bind(b_g3_corrupt)
    assert not torch.equal(k_eff_ref_full[:, j1, :], k_eff_corrupt_full[:, j1, :]), \
        "POSITIVE CONTROL FAILED: post-orth k_eff_items[j!=corrupted] did NOT change -- the joint " \
        "transform is not actually live (or this probe is vacuous)"
    print("    raw k_conv/v_eff: clause-1 BIT-IDENTICAL after corrupting clause 0 (leak-free, as "
          "required); POST-orth k_eff_items[1]: changes after corrupting clause 0 (global coupling "
          "confirmed live, the positive control has teeth)")

    print("  item 10 (v-path bit-identity + gradient pattern under the new readout mechanics):")
    torch.manual_seed(14)
    m_plain_fixed = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                                     buffer_id=pools.buffer_id, geo3_active=False).to(DEV)
    m_plain_fixed.load_state_dict(m_g3.state_dict())
    with torch.no_grad():
        _, k_eff_g3_fixed, v_eff_g3_fixed = m_g3.bind(b_g3)
        _, k_eff_plain_fixed, v_eff_plain_fixed = m_plain_fixed.bind(b_g3)
    assert torch.equal(v_eff_g3_fixed, v_eff_plain_fixed), \
        "item 10a FAILED: v_eff_items differs geo3_active=True vs False at IDENTICAL weights/batch " \
        "(v_conv is supposed to be untouched by F-geo-3)"
    assert not torch.equal(k_eff_g3_fixed, k_eff_plain_fixed), \
        "sensitivity control failed: k_eff_items is IDENTICAL geo3 on/off (the orthogonalization has no effect)"
    print("    (a) v_eff_items BIT-IDENTICAL geo3_active True vs False at fixed weights/batch; "
          "k_eff_items correctly DIFFERS (sensitivity control)")

    m_g3.zero_grad()
    S_T10, k_eff10, v_eff10 = m_g3.bind(b_g3)
    S_leaf10 = S_T10.detach().clone().requires_grad_(True)
    a_slot10 = b_g3["a_slot"]
    pred_leaf10 = m_g3.readout(S_leaf10, b_g3["query_tokens"], b_g3["hops"], k_eff_items=k_eff10, a_slot=a_slot10)
    g_embed10, g_vproj10, g_vconv10, g_kproj10, g_kconv10 = torch.autograd.grad(
        pred_leaf10.sum(), [m_g3.embed.weight, m_g3.v_proj.weight, m_g3.v_conv1d.weight,
                             m_g3.k_proj.weight, m_g3.k_conv1d.weight], allow_unused=True)
    assert g_vproj10 is None or g_vproj10.abs().sum().item() == 0.0, \
        "LEAK: geo3's readout graph touches v_proj -- it must remain key-path-only"
    assert g_vconv10 is None or g_vconv10.abs().sum().item() == 0.0, \
        "LEAK: geo3's readout graph touches v_conv1d -- it must remain key-path-only"
    assert g_embed10[pools.period_id].abs().sum().item() == 0.0, \
        "LEAK: the BIND-only '.' token's embedding row receives readout gradient outside S_T under geo3"
    assert g_kproj10 is not None and torch.isfinite(g_kproj10).all() and g_kproj10.abs().sum().item() > 0.0, \
        "the NEW query-side gradient route (readout -> k_eff_items -> k_proj) is missing or non-finite"
    assert g_kconv10 is not None and torch.isfinite(g_kconv10).all() and g_kconv10.abs().sum().item() > 0.0, \
        "the NEW query-side gradient route (readout -> k_eff_items -> k_conv1d) is missing or non-finite"
    print("    (b/c) with S_T detached: v-path grad None/zero (key-path-only preserved); BIND-only "
          "'.' row grad zero; k_proj/k_conv1d DO receive a FINITE, NONZERO gradient via the NEW "
          "query-side route (readout -> k_eff_items -> embed/k_proj/k_conv1d) -- confirmed live, not exploding")

    # full (non-isolated) forward+backward: v_proj/W_beta remain genuinely learned under geo3
    m_g3.zero_grad()
    pred_full10, tgt_full10, _, _, _ = m_g3(b_g3)
    (1.0 - F.cosine_similarity(pred_full10, tgt_full10, dim=-1)).mean().backward()
    assert m_g3.v_proj.weight.grad is not None and m_g3.v_proj.weight.grad.abs().sum().item() > 0.0, \
        "v_proj received NO gradient under geo3_active -- the v-path should remain learned"
    assert m_g3.W_beta.weight.grad is not None and m_g3.W_beta.weight.grad.abs().sum().item() > 0.0, \
        "W_beta received NO gradient under geo3_active -- beta should remain learned"
    print("    v_proj/W_beta confirmed nonzero-gradient (still learned) in the full forward+backward")

    print("\n[model 17] KEY_ANCHORING_DESIGN.md sec 2.2/5, candidate (d) -- REAL bind()-level "
          "smokes 2/3/4 (the GPU-required complement to smoke_key_anchoring.py's CPU-only "
          "insertion-site tests): forward/backward through the ACTUAL chunk_delta_rule kernel "
          "with anchor_active=True, held-out bypass bit-identity through a real mixed-split "
          "batch, and the NaN-injection held-out-gradient-isolation test through the real bind() "
          "path.")
    torch.manual_seed(15)
    m_anchor = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                                buffer_id=pools.buffer_id, geo3_active=True, geo3_n_iter=12,
                                geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                                anchor_train_ids=pools.train_name_ids).to(DEV)
    assert m_anchor.anchor_active and m_anchor.geo3_active

    guard_raised_anchor = False
    try:
        DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                         buffer_id=pools.buffer_id, geo3_active=False, anchor_active=True,
                         anchor_train_ids=pools.train_name_ids)
    except AssertionError:
        guard_raised_anchor = True
    assert guard_raised_anchor, "anchor_active without geo3_active did NOT raise -- sec 2.2's assert has no teeth"
    print("  anchor_active-requires-geo3_active guard has teeth")

    gen_anchor = torch.Generator(device=DEV).manual_seed(16)
    b_anchor = grd.sample_batch_rd(cfg, 16, gen_anchor, hop_set=cfg.H_train, pools=pools, device=DEV)

    print("  item 2: forward/backward through the REAL bind() (chunk_delta_rule kernel), "
          "anchor_active=True -- finite loss, finite grad on EVERY parameter incl. anchor_table "
          "and the lambda raw-param")
    m_anchor.zero_grad()
    pred_a, tgt_a, S_a, k_eff_a, v_eff_a = m_anchor(b_anchor)
    loss_a = (1.0 - F.cosine_similarity(pred_a, tgt_a, dim=-1)).mean()
    loss_a.backward()
    assert torch.isfinite(loss_a).item()
    for name, p in m_anchor.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), f"non-finite grad through real bind(): {name}"
    assert m_anchor.anchor_table.weight.grad is not None and \
        torch.isfinite(m_anchor.anchor_table.weight.grad).all()
    assert m_anchor.anchor_lambda_raw.grad is not None and torch.isfinite(m_anchor.anchor_lambda_raw.grad).all()
    print(f"    loss={loss_a.item():.4f} finite; ALL params incl. anchor_table/anchor_lambda_raw "
          f"have finite grad through the REAL kernel call")

    print("  item 3: held-out bypass bit-identity through a REAL bind() call -- an all-held-out "
          "batch (heldout entity pool) must be bit-identical to the same weights with anchor "
          "disabled")
    m_plain_anchor = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                                       buffer_id=pools.buffer_id, geo3_active=True, geo3_n_iter=12,
                                       geo3_resid_tol=1e-2, anchor_active=False).to(DEV)
    m_plain_anchor.load_state_dict(m_anchor.state_dict(), strict=False)   # anchor_table/lambda absent on m_plain_anchor
    gen_ho = torch.Generator(device=DEV).manual_seed(17)
    b_heldout = grd.sample_batch_rd(cfg, 16, gen_ho, hop_set=cfg.H_train, pools=pools, device=DEV,
                                      use_heldout_entities=True)
    with torch.no_grad():
        _, k_eff_anchor_ho, _ = m_anchor.bind(b_heldout)
        _, k_eff_plain_ho, _ = m_plain_anchor.bind(b_heldout)
    assert torch.equal(k_eff_anchor_ho, k_eff_plain_ho), \
        "sec 3.3 VIOLATED: an all-held-out batch's k_eff_items differs anchor_active=True vs False " \
        "through the REAL bind() call"
    print("    all-held-out batch: k_eff_items BIT-IDENTICAL anchor_active True vs False "
          "through the real kernel call")

    print("  item 4: NaN-injected held-out anchor rows, mixed-split batch, THROUGH the real "
          "bind() call -- finite grads, exact-zero held-out anchor grad, bit-equal held-out "
          "output rows despite the planted NaNs")
    torch.manual_seed(18)
    m_anchor_nan = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                                     buffer_id=pools.buffer_id, geo3_active=True, geo3_n_iter=12,
                                     geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                                     anchor_train_ids=pools.train_name_ids).to(DEV)
    with torch.no_grad():
        m_anchor_nan.anchor_table.weight[~m_anchor_nan.anchor_trained_mask] = float("nan")
    gen_mixed = torch.Generator(device=DEV).manual_seed(19)
    b_mixed = grd.sample_batch_rd(cfg, 16, gen_mixed, hop_set=cfg.H_train, pools=pools, device=DEV)
    m_anchor_nan.zero_grad()
    pred_n, tgt_n, S_n, k_eff_n, v_eff_n = m_anchor_nan(b_mixed)
    (1.0 - F.cosine_similarity(pred_n, tgt_n, dim=-1)).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m_anchor_nan.parameters()), \
        "NaN-injected held-out anchor rows produced a non-finite gradient SOMEWHERE through the real bind()"
    heldout_rows_mask = ~m_anchor_nan.anchor_trained_mask[b_mixed["key_ids"]]
    assert (m_anchor_nan.anchor_table.weight.grad[~m_anchor_nan.anchor_trained_mask] == 0).all(), \
        "anchor_table gradient at held-out rows is not EXACTLY zero through the real bind() call"
    with torch.no_grad():
        _, k_eff_ref_mixed, _ = m_plain_anchor.bind(b_mixed)   # pure-geo3 reference at the same weights
    assert torch.equal(k_eff_n[heldout_rows_mask], k_eff_ref_mixed[heldout_rows_mask]), \
        "held-out rows are NOT bit-equal to pure-geo3 despite the planted NaNs (real bind() call)"
    print("    ALL grads finite; anchor_table grad EXACTLY zero at held-out rows; held-out "
          "k_eff_items rows bit-equal to pure-geo3 -- confirmed through the REAL kernel call")

    print("\nmodel_rd self-test PASSED")


if __name__ == "__main__":
    _self_test()
