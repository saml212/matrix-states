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


class DeltaNetRDBlock(nn.Module):
    """The primary R2-3 custom block. H=1 single head, single layer (C11).
    No W_q/q_conv1d (see module docstring), no output gate, no o_proj -- the
    readout is external and pinned (section 5.4), so nothing downstream of
    S_T is a learned parameter beyond what BIND already used."""

    def __init__(self, vocab_size_total: int, d_model: int = 256, d_state: int = 64,
                 conv_size: int = 4, buffer_id: int = None, trunc_impl: str = "eigh"):
        super().__init__()
        assert buffer_id is not None, "buffer_id (the reserved zero-pinned token id) is required"
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

        self.embed = nn.Embedding(vocab_size_total, d_model)
        self.k_proj = nn.Linear(d_model, d_state, bias=False)
        self.v_proj = nn.Linear(d_model, d_state, bias=False)
        self.k_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size,
                                          bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size,
                                          bias=False, activation="silu")
        self.W_beta = nn.Linear(d_model, 1, bias=True)

        pin_buffer_row_(self.embed, self.buffer_id)

    # -- shared feature path (embedding -> proj -> causal conv), no recurrence --

    def _kv_over_sequence(self, token_ids: torch.Tensor):
        """Full-sequence k_eff/v_eff PRE-l2norm (l2norm is applied to k
        separately where needed -- see bind()/effective_key_window() -- v is
        never l2-normalized, matching use_qk_l2norm_in_kernel's own scope:
        q/k only). Returns (k_conv, v_conv, x) each (B,T,*)."""
        x = self.embed(token_ids)
        k_conv, _ = self.k_conv1d(self.k_proj(x))
        v_conv, _ = self.v_conv1d(self.v_proj(x))
        return k_conv, v_conv, x

    def effective_key_window(self, window_token_ids: torch.Tensor) -> torch.Tensor:
        """QUERY-time / self-query k_eff extraction (section 5.2: "q_eff is
        extracted... through the model's own embedding -> conv -> W_k
        path"). window_token_ids: (N, L), L = query_len = buf_len+3,
        structured [BUFFER..., KEY, REL, <Q-or-VALUE>]. Returns the
        L2-normalized k_eff at the LAST position only, (N, d_state). NEVER
        calls chunk_delta_rule -- "the query span passes through the
        feature path only, never the recurrence" (section 5.2)."""
        x = self.embed(window_token_ids)
        k_conv, _ = self.k_conv1d(self.k_proj(x))
        return F.normalize(k_conv[:, -1, :], dim=-1)

    # -- BIND (two-kernel-call split step 1-2, section 5.1) --

    def bind(self, batch: dict, force_rank_k: int | None = None):
        """Runs chunk_delta_rule over the BIND phase ONLY (queries never
        enter this call -- the state-freeze is structural by construction,
        matching model_dn.py's own discipline), then truncates ONCE if
        force_rank_k is given, then C15-asserts. Returns
        (S_T (B,d_state,d_state), k_eff_items (B,K,d_state) L2-normalized,
        v_eff_items (B,K,d_state) raw).

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
              approximation of it."""
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
        k_norm = F.normalize(k_conv, dim=-1)                             # zero-safe (0-row -> 0-row)
        beta_logit = torch.sigmoid(self.W_beta(x)).squeeze(-1)          # (B,T), from RAW pre-conv x
        beta = beta_logit * beta_mask                                    # C9/R2-3: hard 0 off VALUE positions

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

        k_eff_items = _gather_at(k_norm, item_pos)                       # EXACTLY the kernel's input rows
        v_eff_items = _gather_at(v_conv, item_pos)                       # raw (v is never l2-normed)
        return S_T, k_eff_items, v_eff_items

    def readout(self, S_T: torch.Tensor, query_tokens: torch.Tensor,
                hops: torch.Tensor) -> torch.Tensor:
        """Pinned external readout (section 5.4): pure function of (S_T,
        query_tokens, hops) plus the SAME k_proj/k_conv1d/embed weights BIND
        already used -- introduces zero parameters beyond what bind() uses."""
        B, Q, L = query_tokens.shape
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
        pred = self.readout(S_T, batch["query_tokens"], batch["hops"])
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

    print("\nmodel_rd self-test PASSED")


if __name__ == "__main__":
    _self_test()
