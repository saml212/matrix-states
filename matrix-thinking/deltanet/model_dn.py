"""model_dn.py -- the primary bespoke DeltaNet causal-rank model wrapper.
See DELTANET_CAUSAL_RANK_DESIGN.md sections 3.5, 4.2, 4.3, 5.2, 6.2 (frozen
design, revision 2.1).

Architecture (design section 4.3's recommended build sequencing item (i)):
"a minimal, bespoke single-layer, single-head DeltaNet-only harness (delta
rule + linear input/output projections, no conv, no surrounding transformer
block)". Concretely:

  BIND phase (per design section 4.1's grammar):
    raw_key_t, raw_val_t = split(bind_embed_t)            # (B,T,d) each
    k_eff_t = normalize(W_k(raw_key_t))                    # LEARNED (C_MLP-
                                                             # equivalent freedom;
                                                             # section 3.6 item 2's
                                                             # open question)
    v_eff_t = raw_val_t                                     # W_v == identity
                                                             # (build-time scoping
                                                             # decision, see below)
    beta_t  = sigmoid(W_beta([raw_key_t; raw_val_t])) * beta_mask_t   # HARD mask
                                                             # (C9/NEW-3): mask is
                                                             # architectural, the
                                                             # sigmoid GATE VALUE on
                                                             # real BIND positions is
                                                             # genuinely learned
                                                             # (section 7 item 7)
    S_T = delta_rule_state(k_eff, v_eff, beta)              # deltanet_core.py
    S_T = truncate_to_rank(S_T, force_rank_k)  if forced    # section 3.5's ONE
                                                             # truncation per forward
                                                             # pass; C15 asserted after

  QUERY phase (external, pinned readout -- design section 5.4):
    q_eff_a = normalize(W_k(query_key_a))                   # SAME W_k as BIND (a
                                                             # query is "a lookup
                                                             # key with no known
                                                             # value", read through
                                                             # the identical
                                                             # key-projection)
    pred(a,h) = S_T^h @ q_eff_a                              # apply_state_power;
                                                             # zero new parameters
                                                             # (verified in the
                                                             # smoke gate, mirroring
                                                             # model_e.py's own
                                                             # C_composition-purity
                                                             # check)

**Build-time interpretive decision, flagged for audit scrutiny (design text
says "projected by the layer's own W_k, W_v" without pinning the exact
input/output space of each):** W_v is the identity (raw value passed
straight into the delta rule and compared directly against the raw target
at scoring time), while W_k is a genuine learned Linear(d,d). Two reasons:
(1) it keeps W_k's input distribution IDENTICAL at bind-time and query-time
(always a raw d-dim key vector -- no bind/query embedding-distribution
mismatch for W_k to additionally have to learn around, which would confound
"did rank-forcing break recovery" with "did the bind/query encoding
mismatch break recovery"); (2) it keeps the value channel exactly matching
task_e.py/model_e.py's own convention (cosine-scored against the RAW
generator target), so C6's injective-graph guarantee (linear independence
of {value_j}) transfers to the rank>=K bound's premise (section 4.1)
without an extra invertibility argument about a second learned map. W_k
is genuinely learned and NOT architecturally forced toward orthonormal
(only the QUERY-phase beta hard-mask is architectural, per section 7 item
7) -- so section 3.6 item 2's open question ("does SGD converge toward the
clean orthonormal/beta~1 regime") is live, not begged.

**Truncation implementation is selectable (audit round-1 FINDING 1):**
``trunc_impl="eigh"`` (default -- rank_utils.truncate_to_rank, verbatim per
the design) or ``trunc_impl="svd_lowrank"`` (randomized-SVD fallback; see
``truncate_to_rank_svd_lowrank``'s docstring for the measured eigh
instability that motivates it and its semantics caveats). Both go through
the identical single-truncation-per-forward-pass mechanism and the
identical C15 assertion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deltanet_core import delta_rule_state, apply_state_power
from rank_utils import truncate_to_rank, effective_rank, stable_rank


def truncate_to_rank_svd_lowrank(Z: torch.Tensor, k: int, n_oversample: int = 6,
                                   niter: int = 4) -> torch.Tensor:
    """Randomized-SVD rank-k truncation -- the audit-round-1 FINDING-1
    numerical-stability fallback to rank_utils.truncate_to_rank's eigh path.

    Why it exists: the independent audit measured the eigh path's per-step
    non-finite-gradient skip rate under force-rank at fr=K as ~2% at d=64
    and ~37% at d=128 (75% by step 400) vs 0% unconstrained -- i.e. the
    design's section 6.4 F13 stop branch (skip rate > 0.1%) fires before
    Wave -1 even launches, exactly the Task-E-M4_E instability the design's
    section 7 item 2 pre-registered as a live risk. This variant replaces
    the d x d eigh(ZZ^T) (whose backward has 1/(lambda_i - lambda_j) terms
    that blow up on the near-degenerate spectra this task's orthonormal-key
    solutions produce BY DESIGN) with torch.svd_lowrank: randomized subspace
    iteration (Halko, Martinsson & Tropp 2011) -- Gaussian sketch + matmuls
    + QR + one SMALL (q x d, q = k + n_oversample) SVD. The backward path is
    mostly plain matmul/QR; the residual degenerate-spectrum risk lives only
    in the small SVD's backward and is measured EMPIRICALLY (the skip-rate
    probe whose results are recorded in run_deltanet_sweep.py's
    timing-constants comment), not assumed away.

    Guarantees preserved vs the eigh path:
      - output rank <= k BY CONSTRUCTION (exactly k components are kept from
        the small SVD, regardless of oversampling) -- C15's assert_rank_le
        applies to the output unchanged;
      - exactness on well-separated spectra: agrees with the eigh path to
        allclose tolerance (verified in _self_test with a run-to-completion
        check, not assumed) -- with a spectral gap at index k and niter
        power iterations, subspace error ~ (sigma_{k+1}/sigma_k)^(2*niter+1).

    Semantics caveat (why "eigh" stays the default): torch.svd_lowrank draws
    a FRESH Gaussian test matrix per call (global RNG, no generator arg), so
    this truncation is stochastic -- two calls on the same input differ at
    ~subspace-error magnitude, and gradcheck through it is not meaningful
    (each re-evaluation re-randomizes). Selecting it is a Wave -1 / F13
    decision recorded per-run in the result JSON ("trunc_impl"), never a
    silent default swap. Soft/penalty-based rank surrogates remain OUT of
    scope without design sign-off (they change the causal semantics).

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


# Selectable truncation implementations (audit round-1 FINDING 1). "eigh" is
# the design default (section 3.5, rank_utils.truncate_to_rank verbatim);
# "svd_lowrank" is the pre-registered stability fallback. The C15 assertion
# below applies identically to both -- the MECHANISM (one truncation per
# forward pass, at the BIND/QUERY boundary, inside the training graph) is
# impl-agnostic; only the numerical route to the rank-k projection differs.
TRUNC_IMPLS = {
    "eigh": truncate_to_rank,
    "svd_lowrank": truncate_to_rank_svd_lowrank,
}


def assert_rank_le(S: torch.Tensor, k: int) -> None:
    """C15 (design section 3.5/6.2): post-truncation rank(S_T) <= k asserted
    by DIRECT SVD (torch.linalg.matrix_rank, which uses a standard
    numerical-zero singular-value cutoff to define numerical rank -- this is
    the ordinary, necessary tolerance for turning floating-point singular
    values into an integer rank, NOT the kind of "-1"-style slack the
    standing house rule on structural/integer checks prohibits). The
    THRESHOLD comparison itself is EXACT: rank <= k, never k+1.
    S: (..., d, d)."""
    r = torch.linalg.matrix_rank(S.float())
    assert torch.all(r <= k), f"C15 VIOLATED: post-truncation rank {r.tolist()} > k={k}"


def entity_subspace_rank(S_T: torch.Tensor, s_ideal: torch.Tensor, K: int):
    """PRIMARY M1-equivalent metric (design section 3.6/6.3): project S_T
    onto the K-dimensional entity subspace recovered from the
    architecture-native ideal's own SVD (analyze_zdump.py-style method,
    TASK_E_FINDINGS.md sections 4/9's subspace-restriction lesson, pulled
    forward pre-emptively per section 3.6 rather than re-discovered the hard
    way). Whole-matrix rank is polluted by an unconstrained orthogonal
    complement the loss never touches; the K-dim-restricted rank is what
    actually tracks K.

    s_ideal: (B,d,d) = sum_j value_j @ effective_key_j^T (this module's
      s_ideal, NOT task_dn.py's raw-key s_ideal -- see run_deltanet.py's
      effective_ideal_S). K <= d required (asserted by the caller's config).
    Returns (effective_rank, stable_rank) of the K x K restricted operator,
    each (B,)."""
    U, Sig, Vh = torch.linalg.svd(s_ideal.float())
    K = min(K, U.shape[-1])
    Uk = U[..., :, :K]                            # (B, d, K): value-side span
    Vk = Vh[..., :K, :].transpose(-1, -2)          # (B, d, K): key-side span
    S_restricted = torch.einsum("bdk,bde,bel->bkl", Uk, S_T.float(), Vk)   # (B,K,K)
    return effective_rank(S_restricted), stable_rank(S_restricted)


class DeltaNetGateModel(nn.Module):
    """The primary bespoke, single-layer, single-head (C11) DeltaNet
    causal-rank harness. Zero surrounding transformer block, zero conv
    (design section 4.3 item (i)); the two-kernel-call split (section 3.5)
    is realized trivially here because the BIND-only pass IS the model's
    entire forward computation -- no QUERY token ever enters the recurrence
    (section 5.2's revision-2 note: the freeze is structural by
    construction in this harness, not enforced via the beta-mask, which is
    load-bearing only for the buffer positions WITHIN the BIND phase)."""

    def __init__(self, d: int, trunc_impl: str = "eigh"):
        super().__init__()
        assert trunc_impl in TRUNC_IMPLS, \
            f"trunc_impl must be one of {sorted(TRUNC_IMPLS)}, got {trunc_impl!r}"
        self.d = d
        self.trunc_impl = trunc_impl
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_beta = nn.Linear(2 * d, 1, bias=True)

    def effective_key(self, raw_key: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.W_k(raw_key), dim=-1)

    def bind(self, batch: dict, force_rank_k: int | None = None,
              return_trajectory: bool = False):
        """The BIND-only kernel call (design section 3.5 step 1-2): runs the
        delta rule over the BIND phase ONLY, then applies
        truncate_to_rank ONCE if force_rank_k is given, then C15-asserts
        the result. Returns S_T (and optionally the per-step trajectory,
        diagnostic use only)."""
        bind_embed = batch["bind_embed"]                  # (B, T_bind, 2d)
        beta_mask = batch["beta_mask"]                     # (B, T_bind), hard 0/1
        d = self.d
        raw_key = bind_embed[..., :d]
        raw_val = bind_embed[..., d:]
        k_eff = self.effective_key(raw_key)
        beta_logit = torch.sigmoid(self.W_beta(bind_embed)).squeeze(-1)
        beta = beta_logit * beta_mask                       # C9/NEW-3: hard 0 off BIND items
        out = delta_rule_state(k_eff, raw_val, beta, return_trajectory=return_trajectory)
        S_T, traj = out if return_trajectory else (out, None)
        if force_rank_k is not None and force_rank_k > 0:
            trunc_fn = TRUNC_IMPLS[self.trunc_impl]          # eigh (default) or svd_lowrank (FINDING 1)
            S_T = trunc_fn(S_T, force_rank_k)                # ONE truncation per forward pass
            assert_rank_le(S_T, force_rank_k)                # C15, exact threshold, every call, both impls
        return (S_T, traj) if return_trajectory else S_T

    def readout(self, S_T: torch.Tensor, query_keys: torch.Tensor,
                hops: torch.Tensor) -> torch.Tensor:
        """Pinned external readout (design section 5.4): pure function of
        (S_T, query, hops) -- introduces ZERO new learned parameters beyond
        the SAME W_k used at bind time (verified in the smoke gate by a
        param-count check mirroring model_e.py's C_composition-purity)."""
        q_eff = self.effective_key(query_keys)
        return apply_state_power(S_T, q_eff, hops)

    def effective_ideal_S(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Architecture-native ideal (design section 3.6), built from THIS
        model's CURRENT effective (post-W_k) keys, not the raw generator
        keys -- the correct reference for entity_subspace_rank's SVD-derived
        K-dim subspace, since that is the space S_T actually operates in."""
        k_eff = self.effective_key(keys)
        return torch.einsum("bki,bkj->bij", values, k_eff)

    def forward(self, batch: dict, force_rank_k: int | None = None):
        S_T = self.bind(batch, force_rank_k=force_rank_k)
        pred = self.readout(S_T, batch["query_keys"], batch["hops"])
        return pred, S_T


def per_head_entity_subspace_rank(S_T: torch.Tensor, s_ideal: torch.Tensor, K: int):
    """Per-head entity_subspace_rank (F11's primary Reserve-wave metric),
    looped over H (H is small, <=4 in this build's manifest) -- reuses
    entity_subspace_rank VERBATIM per head, never on a stacked/concatenated
    tensor (concatenating heads before the SVD would silently mix per-head
    subspaces and produce a meaningless number). S_T, s_ideal: (B, H, d, d).
    Returns (ers, srs): each a length-H list of (B,) tensors -- the caller
    sums/means across heads as the joint-bound metric (design section
    6.4/10: "Sigma_head rank(S_T^(head)) >= K")."""
    H = S_T.shape[1]
    ers, srs = [], []
    for h in range(H):
        er, sr = entity_subspace_rank(S_T[:, h], s_ideal[:, h], K)
        ers.append(er)
        srs.append(sr)
    return ers, srs


def assert_rank_le_per_head(S_T: torch.Tensor, ks) -> None:
    """C15, PER HEAD (never on a stacked tensor -- see
    DeltaNetMultiHeadGateModel's docstring). S_T: (B, H, d, d). ks: int
    (same k asserted on every head) or a length-H sequence (per-head k)."""
    H = S_T.shape[1]
    ks = ks if isinstance(ks, (list, tuple)) else [ks] * H
    assert len(ks) == H, f"per-head ks must have length H={H}, got {len(ks)}"
    for h in range(H):
        assert_rank_le(S_T[:, h], ks[h])


class DeltaNetMultiHeadGateModel(nn.Module):
    """H-head generalization of DeltaNetGateModel -- the F11 / Reserve-wave
    pre-registration (DELTANET_CAUSAL_RANK_DESIGN.md section 6.4/section 8's
    explicit H=1 qualifier, section 4.2's "multi-head escape", section 9's
    "H>1 sweep ... deferred (section 9)"). This class is NOT the primary
    gate -- DeltaNetGateModel (H=1, C11) above is untouched and stays
    primary. This class exists ONLY for the pre-registered Reserve-wave
    generality probe (H in {2,4}, measuring per-head entity-subspace-
    restricted rank against the joint bound Sigma_head rank(S_T^(head)) >= K).

    ARCHITECTURAL CHOICE (build-time interpretive decision, FLAGGED FOR
    AUDIT SCRUTINY -- the design text (section 4.2) describes the escape as
    "H heads, each with its own d_head x d_head state" WITHOUT pinning
    d_head's value relative to d; the F11 pre-registration (section 6.4/10)
    cites the joint bound "Sigma_head rank(S_T^(head)) >= K" but leaves the
    per-head dimension unstated):

      PER-HEAD STATE DIM = d (the SAME d as the H=1 primary gate), so TOTAL
      CAPACITY SCALES WITH H (H independent d x d states, NOT a d/H x d/H
      split of one fixed total budget -- NOT the standard transformer
      multi-head convention).

    Why this reading, not the standard d/H split: section 4.2's own framing
    calls this "an escape" -- "H heads ... could jointly store K items ...
    WITHOUT any single head ever reaching rank K". An escape is only live
    if concentrating in ONE head is actually POSSIBLE (a single head needs
    capacity d_head >= K to hold everything alone) -- otherwise "spreading"
    would be a capacity NECESSITY, not an SGD choice, and section 8's own
    open question ("does the joint bound distribute by concentration or by
    spreading") would be begged by construction rather than tested. At the
    primary cell (d=64, K=32), the standard d/H split would force
    d_head=16 at H=4 -- STRICTLY LESS than K=32, making concentration
    IMPOSSIBLE regardless of what SGD does (spreading would be forced, not
    discovered). The full-d-per-head reading keeps every head capacity-
    equal to the H=1 primary gate (d_head=d=64 >= K=32 at every H in
    {2,4}), so a concentrated solution and a spread solution are BOTH
    representable, and which one SGD finds is the actual empirical
    question. This is the task brief's own stated default when the design
    is ambiguous; documented here (and echoed in every run's result JSON
    via `multihead_dhead_convention`) per its "if ambiguous, say so"
    instruction.

    RAW VALUE CHANNEL: shared (identity) across heads, exactly matching
    DeltaNetGateModel's own W_v=identity choice (module docstring above) --
    every head sees the SAME raw_val stream; only W_k^(h) and W_beta^(h)
    are per-head-learned, so heads can only specialize via WHICH keys they
    respond to (effective_key) and WHEN they write (beta-gate), never via a
    head-private value transform. Minimal generalization of the H=1 model's
    own choice, not re-litigated for H>1.

    K BINDINGS ACROSS HEADS: every head streams the SAME K bind items
    (task_dn.py's batch is head-agnostic, UNCHANGED) -- "heads free to
    divide labor" via their own learned W_k/W_beta, never via a hard-coded
    item-to-head routing. Matches the task brief's stated default ("same K
    bindings, heads free to divide labor -- the question is whether the
    rank bound distributes").

    READOUT COMBINATION: SUMMED across heads (pred = sum_h S_T^(h)^hop @
    q_eff^(h)) -- zero new parameters beyond H copies of W_k/W_beta, the
    natural parameter-free extension of C_composition-purity to H heads.
    (Concatenation is not dimensionally available here: under the
    full-d-per-head choice above every head's own readout is already a
    full d-dim vector in the SAME target space, not a d/H-wide slot to
    concatenate into -- summation is the only zero-param combination that
    produces a d-dim prediction comparable to the single d-dim target.)

    VECTORIZATION: heads are folded into the batch dimension of the
    UNCHANGED deltanet_core.delta_rule_state: (B,H,T,d) -> (B*H,T,d) ->
    delta_rule_state -> (B*H,d,d) -> (B,H,d,d). deltanet_core.py itself is
    NOT modified (its F15/NEW-1/NEW-2 findings and gradcheck coverage all
    still apply verbatim to every per-head call); only this class's
    reshape at the call site is new.
    """

    def __init__(self, d: int, H: int, trunc_impl: str = "eigh"):
        super().__init__()
        assert trunc_impl in TRUNC_IMPLS, \
            f"trunc_impl must be one of {sorted(TRUNC_IMPLS)}, got {trunc_impl!r}"
        assert H >= 2, \
            f"DeltaNetMultiHeadGateModel is for H>=2 (the Reserve-wave probe); use " \
            f"DeltaNetGateModel (H=1, C11, the primary gate) for H=1"
        self.d, self.H = d, H
        self.trunc_impl = trunc_impl
        self.W_k = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(H)])
        self.W_beta = nn.ModuleList([nn.Linear(2 * d, 1, bias=True) for _ in range(H)])

    def effective_key(self, raw_key: torch.Tensor) -> torch.Tensor:
        """raw_key: (..., d) -> (..., H, d), per-head normalized effective key."""
        return torch.stack([F.normalize(wk(raw_key), dim=-1) for wk in self.W_k], dim=-2)

    def bind(self, batch: dict, force_rank_k=None, return_trajectory: bool = False):
        """Per-head BIND-only pass, vectorized via the batch-fold (class
        docstring). force_rank_k: None (unconstrained -- the ONLY arm this
        build's Reserve-wave launch manifest uses), a single int (applied
        identically to every head), or a length-H sequence (PER-HEAD
        force-rank bounds -- generality support per the task brief,
        exercised by the smoke gate, not by the launch manifest). Returns
        S_T: (B, H, d, d) [, per-head trajectory list if requested]."""
        bind_embed = batch["bind_embed"]                  # (B, T, 2d)
        beta_mask = batch["beta_mask"]                      # (B, T)
        B, T, _ = bind_embed.shape
        d, H = self.d, self.H
        raw_key = bind_embed[..., :d]
        raw_val = bind_embed[..., d:]
        k_eff = self.effective_key(raw_key)                  # (B, T, H, d)
        beta_logits = torch.stack(
            [torch.sigmoid(wb(bind_embed)).squeeze(-1) for wb in self.W_beta], dim=-1)  # (B, T, H)
        beta = beta_logits * beta_mask.unsqueeze(-1)          # C9/NEW-3, per head

        # Fold H into the batch dim -- delta_rule_state (deltanet_core.py)
        # is UNCHANGED; only this reshape is new.
        k_flat = k_eff.permute(0, 2, 1, 3).reshape(B * H, T, d)
        v_flat = raw_val.unsqueeze(1).expand(B, H, T, d).reshape(B * H, T, d)   # shared identity value channel
        beta_flat = beta.permute(0, 2, 1).reshape(B * H, T)

        out = delta_rule_state(k_flat, v_flat, beta_flat, return_trajectory=return_trajectory)
        S_flat, traj = out if return_trajectory else (out, None)
        S_T = S_flat.reshape(B, H, d, d)

        if force_rank_k is not None:
            ks = force_rank_k if isinstance(force_rank_k, (list, tuple)) else [force_rank_k] * H
            assert len(ks) == H, f"per-head force_rank_k must have length H={H}, got {len(ks)}"
            trunc_fn = TRUNC_IMPLS[self.trunc_impl]
            heads_out = []
            for h in range(H):
                if ks[h] is not None and ks[h] > 0:
                    Sh = trunc_fn(S_T[:, h], ks[h])              # ONE truncation per forward pass, per head
                    assert_rank_le(Sh, ks[h])                     # C15, PER HEAD (never on a stacked tensor)
                else:
                    Sh = S_T[:, h]
                heads_out.append(Sh)
            S_T = torch.stack(heads_out, dim=1)
        return (S_T, traj) if return_trajectory else S_T

    def readout(self, S_T: torch.Tensor, query_keys: torch.Tensor, hops: torch.Tensor) -> torch.Tensor:
        """SUMMED multi-head readout (class docstring). S_T: (B,H,d,d);
        query_keys: (B,Q,d); hops: (B,Q). Returns (B,Q,d)."""
        B, H, d, _ = S_T.shape
        Q = query_keys.shape[1]
        q_eff = self.effective_key(query_keys)                       # (B, Q, H, d)
        S_flat = S_T.reshape(B * H, d, d)
        q_flat = q_eff.permute(0, 2, 1, 3).reshape(B * H, Q, d)
        hops_flat = hops.unsqueeze(1).expand(B, H, Q).reshape(B * H, Q)
        pred_flat = apply_state_power(S_flat, q_flat, hops_flat)      # (B*H, Q, d)
        pred_per_head = pred_flat.reshape(B, H, Q, d)
        return pred_per_head.sum(dim=1)                                # zero-new-param combine

    def effective_ideal_S(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Per-head architecture-native ideal (section 3.6), each head's
        OWN post-W_k^(h) effective keys against the SHARED raw values.
        Returns (B, H, d, d)."""
        k_eff = self.effective_key(keys)                    # (B, K, H, d)
        return torch.einsum("bki,bkhj->bhij", values, k_eff)

    def forward(self, batch: dict, force_rank_k=None):
        S_T = self.bind(batch, force_rank_k=force_rank_k)
        pred = self.readout(S_T, batch["query_keys"], batch["hops"])
        return pred, S_T


# ---------------------------------------------------------------------------
# Self-test (part of run_deltanet.py --smoke)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import task_dn as tdn

    torch.manual_seed(0)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    print("[model 1] forward / backward / grad-finite, H=1 single layer (C11)")
    cfg = tdn.DeltaNetTaskConfig(d=16, K=8, conv_size=4,
                                  H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    gen = torch.Generator(device=DEV).manual_seed(1)
    model = DeltaNetGateModel(d=16).to(DEV)
    b = tdn.sample_batch(cfg, 32, gen, hop_set=cfg.H_train, device=DEV)
    pred, S_T = model(b)
    assert pred.shape == b["targets"].shape
    assert not torch.isnan(pred).any()
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(b["targets"], dim=-1)
    loss = (1.0 - (pred_n * tgt_n).sum(-1)).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), f"bad grad {name}"
    print(f"  forward {tuple(pred.shape)}, loss {loss.item():.4f}, all params finite grad")

    print("\n[model 2] C15 + trunc impls: force_rank_k constrains AND asserts rank for BOTH "
          "--trunc-impl variants; grads exist AND are finite through each (explicit "
          "p.grad-is-not-None asserts -- audit round-1 MINOR-2 closed the vacuous-pass "
          "gap the old combined check had); negative test has teeth")
    for impl in ("eigh", "svd_lowrank"):
        m_i = DeltaNetGateModel(d=16, trunc_impl=impl).to(DEV)
        m_i.load_state_dict(model.state_dict())              # identical weights, only impl differs
        m_i.zero_grad()
        pred_k, S_k = m_i(b, force_rank_k=3)
        assert_rank_le(S_k, 3)   # re-assert externally too (redundant w/ internal, that's the point)
        (1.0 - F.cosine_similarity(pred_k, b["targets"], dim=-1)).mean().backward()
        for name, p in m_i.named_parameters():
            assert p.grad is not None, f"[{impl}] no grad for {name} through the force-rank path"
            assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), \
                f"[{impl}] non-finite grad for {name} through the force-rank path"
        print(f"  [{impl}] force_rank_k=3 -> rank<=3 (asserted); every param has a finite grad")

    # exactness check (audit FINDING-1 requirement (a)): on a WELL-SEPARATED
    # spectrum (10x gap at the truncation index) the two impls must agree.
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
    assert_rank_le(Z_slr, kx)    # FINDING-1 requirement (b): C15 holds on the svd_lowrank output too

    neg_raised = False
    try:
        # construct a genuine rank-4 matrix (sum of two independent rank-2
        # factorizations, generically rank 4) and assert against k=2 (must fire)
        rank4 = torch.randn(1, 5, 2) @ torch.randn(1, 2, 5) + torch.randn(1, 5, 2) @ torch.randn(1, 2, 5)
        assert_rank_le(rank4, 2)
    except AssertionError:
        neg_raised = True
    assert neg_raised, "C15 negative test FAILED to fire on a known higher-rank matrix"
    print(f"  eigh vs svd_lowrank agree on a well-separated spectrum (max abs diff "
          f"{max_impl_diff:.2e} on values up to 64); C15 holds on svd_lowrank output; "
          f"negative test on a rank-4 matrix vs k=2 correctly fires")

    print("\n[model 3] C_composition-purity: readout adds ZERO new parameters beyond W_k")
    import inspect
    sig = set(inspect.signature(DeltaNetGateModel.readout).parameters) - {"self"}
    assert sig <= {"S_T", "query_keys", "hops"}, f"readout takes inputs beyond (S_T,query_keys,hops): {sig}"
    n_total = sum(p.numel() for p in model.parameters())
    n_wk = sum(p.numel() for p in model.W_k.parameters())
    n_wbeta = sum(p.numel() for p in model.W_beta.parameters())
    assert n_total == n_wk + n_wbeta, "unexpected extra parameters beyond W_k, W_beta"
    print(f"  readout signature pinned to (S_T, query_keys, hops); n_params = W_k({n_wk}) + "
          f"W_beta({n_wbeta}) = {n_total}, nothing else")

    print("\n[model 4] blank-out, DIRECTION 1 (post-S_T, structural-by-construction): "
          "detaching S_T before readout severs ALL gradient to bind_embed")
    bind_embed_g = b["bind_embed"].clone().requires_grad_(True)
    S_g = model.bind({**b, "bind_embed": bind_embed_g})
    pred_g = model.readout(S_g, b["query_keys"], b["hops"])
    gk = torch.autograd.grad(pred_g.sum(), bind_embed_g, retain_graph=True, allow_unused=True)[0]
    assert gk is not None and gk.abs().sum() > 0, "bindings don't affect pred at all?!"
    S_leaf = S_g.detach().clone().requires_grad_(True)
    pred_leaf = model.readout(S_leaf, b["query_keys"], b["hops"])
    g_leak = torch.autograd.grad(pred_leaf.sum(), bind_embed_g, allow_unused=True)[0]
    assert g_leak is None, "LEAK: bindings reach the readout outside S_T"
    print("  bindings affect pred only through S_T (gradient present through S_T, "
          "None once S_T is detached)")

    print("\n[model 5] blank-out, DIRECTION 2 (buffer positions, C14/NEW-3): corrupting "
          "the LAST conv_size-1 positions before the BIND/QUERY boundary (the boundary "
          "buffer) leaves S_T BIT-IDENTICAL; corrupting a REAL bind item changes it "
          "(sensitivity control -- the test has teeth)")
    with torch.no_grad():
        S_ref = model.bind(b)
        buf_len = cfg.buf_len
        T_bind = cfg.T_bind
        boundary_buf_idx = torch.arange(T_bind - buf_len, T_bind)   # last buf_len positions
        is_item = torch.zeros(T_bind, dtype=torch.bool)
        is_item[b["item_pos"][0].cpu()] = True
        assert not is_item[boundary_buf_idx].any(), "boundary positions unexpectedly overlap a BIND item"

        b_corrupt_buf = {**b, "bind_embed": b["bind_embed"].clone()}
        b_corrupt_buf["bind_embed"][:, boundary_buf_idx, :] = torch.randn(
            b["bind_embed"].shape[0], buf_len, 2 * cfg.d, device=DEV) * 10.0
        S_corrupt_buf = model.bind(b_corrupt_buf)
        assert torch.equal(S_ref, S_corrupt_buf), \
            "corrupting boundary BUFFER positions changed S_T -- the beta-mask isn't isolating buffers"

        item0_pos = int(b["item_pos"][0, 0].item())
        b_corrupt_item = {**b, "bind_embed": b["bind_embed"].clone()}
        b_corrupt_item["bind_embed"][:, item0_pos, :] += 10.0
        S_corrupt_item = model.bind(b_corrupt_item)
        assert not torch.equal(S_ref, S_corrupt_item), \
            "corrupting a REAL bind item did NOT change S_T -- sensitivity control failed " \
            "(the blank-out test above would be vacuous without this)"
    print(f"  boundary buffer (last {buf_len} of {T_bind} positions) corruption -> S_T unchanged; "
          f"a real BIND item corruption -> S_T DOES change (sensitivity confirmed)")

    print("\n[model 6] two-kernel-call round trip at the MODEL level (not just deltanet_core): "
          "bind() on the full BIND sequence == bind() on a PREFIX then continued via "
          "initial_state, given the C14 zero-buffer convention")
    with torch.no_grad():
        k_eff_full = model.effective_key(b["bind_embed"][..., :cfg.d])
        v_full = b["bind_embed"][..., cfg.d:]
        beta_full = torch.sigmoid(model.W_beta(b["bind_embed"])).squeeze(-1) * b["beta_mask"]
        split_at = cfg.T_bind // 2
        S_full = delta_rule_state(k_eff_full, v_full, beta_full)
        S_prefix = delta_rule_state(k_eff_full[:, :split_at], v_full[:, :split_at], beta_full[:, :split_at])
        S_two = delta_rule_state(k_eff_full[:, split_at:], v_full[:, split_at:], beta_full[:, split_at:],
                                  initial_state=S_prefix)
        rt_diff = (S_full - S_two).abs().max().item()
        assert rt_diff < 1e-5, f"model-level round trip mismatch: {rt_diff:.6f}"
    print(f"  model-level round trip max abs diff = {rt_diff:.2e}")

    print("\n[model 7] entity_subspace_rank: identity-regime sanity (orthonormal keys, "
          "beta=1, UNTRAINED W_k=I) -- restricted rank == K exactly, matching section 3.6")
    d0, K0 = 12, 5
    m0 = DeltaNetGateModel(d=d0).to(DEV)
    with torch.no_grad():
        m0.W_k.weight.copy_(torch.eye(d0, device=DEV))       # force W_k = identity for this unit test
    cfg0 = tdn.DeltaNetTaskConfig(d=d0, K=K0, conv_size=1, H_train=(1,), H_test=(2,), H_extra=())
    gen0 = torch.Generator(device=DEV).manual_seed(3)
    b0 = tdn.sample_batch(cfg0, 4, gen0, hop_set=(1,), device=DEV)
    with torch.no_grad():
        # force beta=1 on BIND items by monkey-patching W_beta's bias hugely positive
        m0.W_beta.bias.fill_(20.0)
        m0.W_beta.weight.zero_()
        S_T0 = m0.bind(b0)
        s_ideal0 = m0.effective_ideal_S(b0["keys"], b0["values"])
        er, sr = entity_subspace_rank(S_T0, s_ideal0, K0)
    assert torch.all(er <= K0 + 1e-2) and torch.all(er >= K0 - 1e-2), \
        f"entity-subspace rank should == K={K0} in the identity/beta=1 regime, got {er.tolist()}"
    print(f"  entity_subspace_rank (effective_rank) = {er.tolist()}, expected ~{K0}")

    print("\n[model 8] multi-head (H=2): forward/backward/grad-finite (C11's H=1 default "
          "stays the primary gate; this is the F11 Reserve-wave probe's harness ONLY)")
    cfg_mh = tdn.DeltaNetTaskConfig(d=16, K=8, conv_size=4,
                                     H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    gen_mh = torch.Generator(device=DEV).manual_seed(2)
    model_mh = DeltaNetMultiHeadGateModel(d=16, H=2).to(DEV)
    b_mh = tdn.sample_batch(cfg_mh, 32, gen_mh, hop_set=cfg_mh.H_train, device=DEV)
    pred_mh, S_mh = model_mh(b_mh)
    assert S_mh.shape == (32, 2, 16, 16), S_mh.shape
    assert pred_mh.shape == b_mh["targets"].shape
    assert not torch.isnan(pred_mh).any()
    loss_mh = (1.0 - F.cosine_similarity(pred_mh, b_mh["targets"], dim=-1)).mean()
    loss_mh.backward()
    for name, p in model_mh.named_parameters():
        assert p.grad is not None, f"no grad for {name} (H=2)"
        assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), f"bad grad {name} (H=2)"
    n_wk_mh = sum(p.numel() for p in model_mh.W_k.parameters())
    n_wb_mh = sum(p.numel() for p in model_mh.W_beta.parameters())
    n_total_mh = sum(p.numel() for p in model_mh.parameters())
    assert n_total_mh == n_wk_mh + n_wb_mh, "unexpected extra multi-head params"
    print(f"  H=2: S_T {tuple(S_mh.shape)}, pred {tuple(pred_mh.shape)}, loss {loss_mh.item():.4f}, "
          f"all params finite grad, n_params={n_total_mh} (2x an H=1 model's per-head count)")

    print("\n[model 9] multi-head C15: per-head force-rank constrains AND asserts PER HEAD "
          "(not on a stacked tensor); uniform AND per-head-list force_rank_k both work; "
          "per-head negative test has teeth")
    S_k_mh = model_mh.bind(b_mh, force_rank_k=3)
    assert_rank_le_per_head(S_k_mh, 3)
    for h in range(2):
        assert_rank_le(S_k_mh[:, h], 3)
    S_k_list = model_mh.bind(b_mh, force_rank_k=[2, 5])         # per-head bounds, generality support
    assert_rank_le(S_k_list[:, 0], 2)
    assert_rank_le(S_k_list[:, 1], 5)
    neg_raised_mh = False
    try:
        rank4_head = (torch.randn(4, 8, 2, device=DEV) @ torch.randn(4, 2, 8, device=DEV)
                      + torch.randn(4, 8, 2, device=DEV) @ torch.randn(4, 2, 8, device=DEV))
        rank1_head = torch.randn(4, 8, 1, device=DEV) @ torch.randn(4, 1, 8, device=DEV)
        bad_stack = torch.stack([rank4_head, rank1_head], dim=1)   # (4,2,8,8): head0 rank4, head1 rank1
        assert_rank_le_per_head(bad_stack, 2)                       # head 0 (rank 4) must trip this
    except AssertionError:
        neg_raised_mh = True
    assert neg_raised_mh, "per-head C15 negative test FAILED to fire on a known higher-rank head"
    print("  per-head force_rank_k (uniform=3 AND per-head list [2,5]) constrains + asserts "
          "correctly; per-head negative test (head0 rank4 vs k=2, head1 rank1 fine) correctly fires")

    print("\n[model 10] multi-head IDEALIZED recall (H=2, CONCENTRATED regime): forcing "
          "W_k=I and beta~=1 on EVERY head makes every head independently reach the SAME "
          "architecture-native ideal (heads see identical (raw_key,raw_val,beta_mask) input, "
          "section 3.6's proposition applied per head) -- the SUMMED readout must still "
          "recover the target EXACTLY (cos~=1). This is the harness-mechanics sanity check "
          "the task brief asks for ('idealized recall must pass at H=2'), not yet the SGD "
          "'does it distribute' question -- that is the actual Reserve-wave empirical "
          "question, answered only by the real training runs.")
    d0h, K0h, H0h = 12, 5, 2
    m_ideal = DeltaNetMultiHeadGateModel(d=d0h, H=H0h).to(DEV)
    with torch.no_grad():
        for wk in m_ideal.W_k:
            wk.weight.copy_(torch.eye(d0h, device=DEV))
        for wb in m_ideal.W_beta:
            wb.bias.fill_(20.0)
            wb.weight.zero_()
    cfg0h = tdn.DeltaNetTaskConfig(d=d0h, K=K0h, conv_size=1, H_train=(1,), H_test=(2,), H_extra=())
    gen0h = torch.Generator(device=DEV).manual_seed(4)
    b0h = tdn.sample_batch(cfg0h, 8, gen0h, hop_set=(1,), device=DEV)
    with torch.no_grad():
        pred0h, S0h = m_ideal(b0h)
    cos0h = F.cosine_similarity(pred0h, b0h["targets"], dim=-1)
    assert cos0h.mean().item() > 1 - 1e-3, \
        f"multi-head idealized recall FAILED: mean cos {cos0h.mean().item():.6f} (expected ~1.0)"
    s_ideal0h = m_ideal.effective_ideal_S(b0h["keys"], b0h["values"])
    for h in range(H0h):
        diff_h = (S0h[:, h] - s_ideal0h[:, h]).abs().max().item()
        assert diff_h < 1e-2, f"head {h} did not reach its own architecture-native ideal: {diff_h:.4f}"
    print(f"  H=2 idealized recall: mean cos={cos0h.mean().item():.6f} (expect ~1.0); both "
          f"heads independently reached their own ideal S (max per-head diff < 1e-2)")

    print("\n[model 11] multi-head DISTRIBUTED positive control (H=2): constructing S_T "
          "DIRECTLY (bypassing bind/training) so head 0 stores ONLY entities {0..K/2-1} and "
          "head 1 stores ONLY the rest -- confirms the SUMMED readout mechanism can represent "
          "a genuinely DISTRIBUTED Sigma_head rank(S_T^(head)) >= K solution correctly (each "
          "head individually rank K/2 < K, joint sum == K), not just the redundant/"
          "concentrated case in [model 10]. A MECHANICS check (the harness can measure this "
          "case), not a claim about what SGD actually finds -- that is the real run's job.")
    d1h, K1h, H1h = 16, 8, 2
    torch.manual_seed(5)
    Q1h, _ = torch.linalg.qr(torch.randn(d1h, d1h, device=DEV))
    keys1h = Q1h[:, :K1h].T.unsqueeze(0)                          # (1, K1h, d1h) orthonormal rows
    values1h = F.normalize(torch.randn(1, K1h, d1h, device=DEV), dim=-1)
    half = K1h // 2
    S_head0 = torch.einsum("ki,kj->ij", values1h[0, :half], keys1h[0, :half]).unsqueeze(0)   # entities 0..half-1
    S_head1 = torch.einsum("ki,kj->ij", values1h[0, half:], keys1h[0, half:]).unsqueeze(0)   # entities half..K1h-1
    S_dist = torch.stack([S_head0, S_head1], dim=1)                # (1, 2, d1h, d1h)
    m_dist = DeltaNetMultiHeadGateModel(d=d1h, H=H1h).to(DEV)
    with torch.no_grad():
        for wk in m_dist.W_k:
            wk.weight.copy_(torch.eye(d1h, device=DEV))            # W_k=I on both heads -- query reads the raw key
    hops1h = torch.ones(1, K1h, dtype=torch.int64, device=DEV)      # h=1: one direct application, S@k_j == v_j
    pred_dist = m_dist.readout(S_dist, keys1h, hops1h)
    cos_dist = F.cosine_similarity(pred_dist[0], values1h[0], dim=-1)
    assert cos_dist.mean().item() > 1 - 1e-3, \
        f"distributed positive control FAILED: mean cos {cos_dist.mean().item():.6f}"
    er_h0 = effective_rank(S_dist[:, 0]).item()
    er_h1 = effective_rank(S_dist[:, 1]).item()
    print(f"  H=2 distributed control: mean cos={cos_dist.mean().item():.6f} (expect ~1.0) "
          f"recovering ALL {K1h} entities from two K/2={half}-item heads via the summed "
          f"readout; per-head whole-state eff rank ~= {er_h0:.2f}/{er_h1:.2f} "
          f"(each ~= {half}, jointly sufficient for K1h={K1h})")

    print("\n[model 12] multi-head H=4 quick smoke (generality beyond H=2): forward/backward/"
          "grad-finite + per-head shapes + per_head_entity_subspace_rank plumbing")
    model_h4 = DeltaNetMultiHeadGateModel(d=16, H=4).to(DEV)
    gen_h4 = torch.Generator(device=DEV).manual_seed(6)
    b_h4 = tdn.sample_batch(cfg_mh, 16, gen_h4, hop_set=cfg_mh.H_train, device=DEV)
    pred_h4, S_h4 = model_h4(b_h4)
    assert S_h4.shape == (16, 4, 16, 16), S_h4.shape
    loss_h4 = (1.0 - F.cosine_similarity(pred_h4, b_h4["targets"], dim=-1)).mean()
    loss_h4.backward()
    for name, p in model_h4.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"H=4 bad/missing grad {name}"
    s_ideal_h4 = model_h4.effective_ideal_S(b_h4["keys"], b_h4["values"])
    ers4, srs4 = per_head_entity_subspace_rank(S_h4, s_ideal_h4, cfg_mh.K)
    assert len(ers4) == 4 and len(srs4) == 4
    print(f"  H=4: S_T {tuple(S_h4.shape)}, loss {loss_h4.item():.4f}, all grads finite, "
          f"per_head_entity_subspace_rank returns {len(ers4)} entries as expected")

    print("\nmodel_dn self-test PASSED")


if __name__ == "__main__":
    _self_test()
