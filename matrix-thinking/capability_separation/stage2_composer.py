"""CAPABILITY_SEPARATION_DESIGN.md S2.2.2 (Rev 3, DESIGN-CLEARED-FOR-BUILD
per S2.18) -- GroupWordDeltaComposer: the bespoke fp32 per-step torch
recurrence

    S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

implemented as an explicit Python-level scan over the D word positions (and,
for n_h>=2, an inner loop over the n_h Householder factors per position),
run with a SINGLE head (H=1, d_head=h=32), so the per-step state IS a single
32x32 matrix S_t -- no reshape across heads needed. NO positional table
(order comes from the recurrence itself, not a lookup index, S2.2.1).

KERNEL ADJUDICATION (S2.2.2 Rev 1): torch, not `fla`, on every decisive
path -- no efficiency need at this scale (D<=64, d_head=32, tiny batches),
the whole fla envelope-risk class (bf16-only, head_dim>=32 crash floor,
qk-L2-norm coupling) does not bind a plain torch loop, and fp32 removes the
bf16-compounding confound at the decisive far-depth (~8x) checkpoint. `fla`
is retained in exactly ONE non-decisive role: `fla_cross_check` below, a
one-cell numerical cross-check against `beta_fla_smoke.py`'s box-verified
reference kernel -- CUDA+real-fla only, self-skips loudly on CPU/stub,
mirroring `beta_fla_smoke.py::reproduce_fig5`'s own box-only-skip
convention.

READOUT REUSE, disclosed adaptation (S2.2.2: "reuse GroupWordEncoder's OWN
readout head UNMODIFIED -- row_queries/reader/row_norm/row_out"). Literal
Python-level reuse (importing the four submodules out of an existing
`GroupWordEncoder` instance) is not available without ALSO instantiating
that class's `tok_embed`/`pos_embed`/`TransformerEncoder` (encoder-specific
machinery this design explicitly excludes, S2.2.1) as dead, ungraded
parameters -- which would (a) corrupt the +-15% param-matching tolerance
(S2.2.2/S2.9 item 7) with unused weights, and (b) break the standard "every
trainable parameter receives a gradient" smoke convention every other
module in this directory follows (group_word_encoder.py's own `smoke()`).
`RowReadoutHead` below is therefore NEW CODE that replicates the exact
submodule types, hyperparameters, and refine-loop formula of
`group_word_encoder.py`'s `GroupWordEncoder.__init__` (row_queries/
reader/row_norm/row_out, lines 74-78) and `encode_from_embedding` (lines
96-102) verbatim -- "UNMODIFIED" is honored at the level of architecture/
formula, not literal object identity, which the class bundling makes
infeasible here without editing group_word_encoder.py (out of this build's
FILE OWNERSHIP, another wave owns it).

q_proj is DELIBERATELY OMITTED from the trained composer (a disclosed build
decision, not an oversight): the pinned recurrence above has no
q-dependence -- q only affects a per-token OUTPUT read in the standard
DeltaNet/DeltaProduct formalism (matches `beta_fla_smoke.py`'s own
`DeltaProductLayer`, whose q enters `chunk_delta_rule`'s per-token `o` but
not its returned `final_state`), and Stage 2's readout consumes ONLY the
state S_t (via `RowReadoutHead`), never a per-token output. A live q_proj
would therefore be a dead (never-graded) parameter -- corrupting the
+-15% param-matching tolerance and failing the "every parameter gets a
gradient" smoke convention. `fla_cross_check` below constructs its own
throwaway q tensor purely to satisfy `chunk_delta_rule`'s input signature;
this does not reintroduce a trained q_proj into the composer.

REGISTERED RISK (S2.2.2 Rev 1, the round's single highest-value finding):
rank(S_D) <= min(32, n_h*D) -- PROVEN below as an exact, always-true
structural fact (not a statistical tendency): each Householder micro-step
is S' = S(I - beta k k^T) + beta v k^T = S + beta(v - Sk)k^T, a RANK-1
update to S regardless of k/v/beta, so rank(S') <= rank(S) + 1 for every
micro-step, hence rank(S_D) <= n_h*D (and <= 32 trivially, the ambient
dimension) -- with the ENTIRE training range (D<=8, n_h=2 default) feeding
the reader a state whose 32 rows span a subspace of dimension << 32
(near-collinear rows). This is the exact mechanism `stage2_instrument.py`'s
query-dependence diagnostic (S2.8 item 2(e)) is built to detect. Mandatory
detection there, not here; `use_bos_row` below is the pre-validated fix
(S1.30 item 6, transposed to this reader's memory axis), wired as a build-
time toggle so the calibration gate's routing can apply it without a
second build cycle if it fires.
"""
from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_word_encoder import cosine_loss  # reused verbatim, not reimplemented (S1.4's loss primitive)

# S2.2.2/S2.9 item 7 param-matching, S2.20 F6 -- see GroupWordDeltaComposer's
# docstring. Computed exactly (not guessed): d=530 was the smallest tested
# value landing EVERY (group, n_h in {1,2,4}) combination within the
# design's pinned +-15% tolerance of Arm 1's own per-group trainable-param
# count (verified directly via count_params/check_param_match below, not
# estimated) -- centers closest on n_h=2 (the primary grid's default,
# delta +0.05%), with n_h=1 at -4.7..-4.8% and n_h=4 at +9.6..+9.7%, both
# comfortably inside tolerance for all 5 groups.
WIDEN_HIDDEN = 530


def count_params(model: nn.Module) -> int:
    """Total trainable parameter count (S2.2.2/S2.9 item 7 param-matching;
    S2.20 F6)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


PARAM_MATCH_TOLERANCE = 0.15   # S2.2.2 Rev 1 pin: arms count as matched within +-15% of Arm 1.


def check_param_match(composer_params: int, arm1_params: int,
                      tol: float = PARAM_MATCH_TOLERANCE) -> dict:
    """S2.2.2's pinned +-15%-of-Arm-1 param-matching check (S2.20 F6):
    reports the exact counts and signed delta fraction UNCONDITIONALLY (the
    design's own "exact per-arm counts reported regardless" instruction,
    S2.2.2/S2.9 item 7) -- the CALLER decides whether to assert on
    `within_tolerance` (e.g. `run_real_cell`, where the design pins it;
    Arm 1 vs itself, or a build-time exploratory config, are not required
    to assert)."""
    delta = (composer_params - arm1_params) / arm1_params
    ok = abs(delta) <= tol
    return dict(composer_params=composer_params, arm1_params=arm1_params,
               delta_frac=delta, tolerance=tol, within_tolerance=ok)


class RowReadoutHead(nn.Module):
    """d_state learned "row-reader" latents, each reading one row of Z from
    a (B, rows, h) memory -- replicates GroupWordEncoder's row_queries/
    reader/row_norm/row_out block (group_word_encoder.py:74-78,96-102)
    verbatim (same nn.MultiheadAttention config, same LayerNorm/Linear
    shapes, same random-init scale, same residual-refine formula). See this
    module's docstring for why literal cross-file reuse isn't available.

    `use_bos_row`: the pre-validated S1.30-item-6 fix, transposed to this
    reader's memory axis -- a learned, always-present extra memory row
    prepended to the `rows`-row state memory before the reader, guaranteeing
    at least one key not derived from (and generically not collinear with)
    the state rows. OFF by default; the calibration gate's routing turns it
    on for ALL Arm-2/Arm-3 cells at once if the query-dependence diagnostic
    fires (S2.8 item 2(e)'s two-level FAIL routing, "all-arms-or-none").
    """

    def __init__(self, d_state: int, h: int = 32, n_heads: int = 4, n_refine: int = 1,
                 use_bos_row: bool = False):
        super().__init__()
        self.d_state = d_state
        self.h = h
        self.n_refine = n_refine
        self.use_bos_row = use_bos_row
        self.row_queries = nn.Parameter(torch.randn(d_state, h) * 0.02)
        self.reader = nn.MultiheadAttention(h, n_heads, batch_first=True, dropout=0.0)
        self.row_norm = nn.LayerNorm(h)
        self.row_out = nn.Linear(h, d_state)
        if use_bos_row:
            self.bos_row = nn.Parameter(torch.randn(h) * 0.02)

    def prepare_mem(self, mem: torch.Tensor) -> torch.Tensor:
        """mem: (B, rows, h) -> (B, rows[+1], h), prepending the learned BOS
        row iff use_bos_row. Exposed separately (not inlined into forward)
        so stage2_instrument.py's query-dependence diagnostic can apply the
        IDENTICAL transform to both the real probe memory and the synthetic
        healthy anchor before comparing them (apples-to-apples)."""
        if not self.use_bos_row:
            return mem
        B = mem.shape[0]
        bos = self.bos_row.view(1, 1, self.h).expand(B, 1, self.h)
        return torch.cat([bos, mem], dim=1)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        """mem: (B, rows, h) -> Z: (B, d_state, d_state)."""
        mem = self.prepare_mem(mem)
        B = mem.shape[0]
        q = self.row_queries.unsqueeze(0).expand(B, self.d_state, self.h)
        for _ in range(self.n_refine):
            read, _ = self.reader(q, mem, mem, need_weights=False)
            q = self.row_norm(q + read)
        return self.row_out(q)


class GroupWordDeltaComposer(nn.Module):
    """Word of generator indices (B, D) -> a single d_state x d_state matrix
    Z, via a genuine step-wise recurrence (no positional table, S2.2.1).

    `beta_max`: 1.0 for Arm 2 (beta in [0,1], plain sigmoid, Grazzi et al.'s
    baseline gate -- interesting eigenvalue 1-beta in [0,1], never negative);
    2.0 for Arm 3 (beta in [0,2], `allow_neg_eigval`-style, this repo's
    `beta_fla_smoke.py::DeltaProductLayer` pattern -- eigenvalue 1-beta in
    [-1,1]).

    `n_h`: DeltaProduct Householder-product count per macro-token (S2.2.3);
    default 2, force-arm-gridded to {1,2,4} for S5/A6 (S2.5).

    `widen_hidden` (S2.2.2/S2.9 item 7 param-matching, S2.20 F6): a
    residual MLP applied to `tok_embed` (inside `states_from_embedding`,
    BEFORE the k/v/beta projections) purely to add trainable capacity --
    the design's own "widening projections" option (S2.2.2), computed
    exactly at build/fix time (`WIDEN_HIDDEN` below), not asserted. A bare
    single-layer delta-rule composer sits ~70-84% BELOW Arm 1
    (GroupWordEncoder)'s ~43K trainable params at every (group, n_h) in
    the pinned grid (verified by direct count, not estimated); this residual
    widen closes that gap to within the design's own pinned +-15% tolerance
    for every (group, n_h in {1,2,4}) combination (`check_param_match`
    below, exercised in smoke). Deliberately does NOT touch `h` (the
    ambient/state dimension every rank-bound/instrument proof is pinned
    against) or `n_h` (the experimental Householder-count axis, S2.5) --
    only adds a differentiable detour between the leaf `tok_embed` and the
    k/v/beta projections, so it changes NEITHER the recurrence formula,
    NOR the (h,h) state shape, NOR the blank-out test's leaf-detach
    contract (the leaf is still exactly `embed_tokens`'s output; gradient
    flows through `widen` like any other differentiable layer in that path).
    """

    def __init__(self, d_state: int, n_gens: int, h: int = 32, n_h: int = 2,
                 beta_max: float = 2.0, n_heads_reader: int = 4, n_refine: int = 1,
                 use_bos_row: bool = False, widen_hidden: int = WIDEN_HIDDEN):
        super().__init__()
        assert beta_max in (1.0, 2.0), f"beta_max must be 1.0 (Arm 2) or 2.0 (Arm 3), got {beta_max}"
        self.d_state = d_state
        self.n_gens = n_gens
        self.h = h
        self.n_h = n_h
        self.beta_max = beta_max
        self.allow_neg_eigval = beta_max > 1.0
        # Delta 2 (S1.4, reused): single generator-index embedding, no
        # separate value to bind -- the WORD is the whole composition.
        self.tok_embed = nn.Embedding(n_gens, h)
        # S2.20 F6: param-matching residual widen (see class docstring).
        # Shape-preserving ((B,D,h) -> (B,D,h)), so nothing downstream
        # needs to change.
        self.widen = nn.Sequential(nn.Linear(h, widen_hidden), nn.GELU(),
                                   nn.Linear(widen_hidden, h)) if widen_hidden > 0 else None
        # k_proj/v_proj/beta_proj: beta_fla_smoke.py's DeltaProductLayer
        # projection pattern (q_proj deliberately omitted, see module
        # docstring). Each macro-token contributes n_h independent
        # (k,v,beta) micro-step triples (Householder-product expansion,
        # Siems et al. 2502.10297).
        self.k_proj = nn.Linear(h, h * n_h, bias=False)
        self.v_proj = nn.Linear(h, h * n_h, bias=False)
        self.beta_proj = nn.Linear(h, n_h)
        self.readout = RowReadoutHead(d_state, h=h, n_heads=n_heads_reader,
                                      n_refine=n_refine, use_bos_row=use_bos_row)

    @property
    def use_bos_row(self) -> bool:
        return self.readout.use_bos_row

    @staticmethod
    def rank_bound(D: int, n_h: int, h: int = 32) -> int:
        """S2.2.2's registered-risk bound, exact (proven in the module
        docstring): rank(S_D) <= min(h, n_h*D)."""
        return min(h, n_h * D)

    # -------------------------------------------------------------------
    # embed_tokens / states_from_embedding split, mirroring
    # group_word_encoder.py's embed_tokens/encode_from_embedding split --
    # the blank-out test's leaf is taken at embed_tokens' OUTPUT (the first
    # continuous tensor derived from the discrete word), same convention.
    # -------------------------------------------------------------------

    def embed_tokens(self, token_idx: torch.Tensor) -> torch.Tensor:
        """token_idx: (B, D) long generator indices -> (B, D, h). NO
        positional embedding added anywhere in this pipeline (S2.2.1/S2.2.2:
        order is a structural property of the recurrence, not a lookup)."""
        return self.tok_embed(token_idx)

    def states_from_embedding(self, tok_embed: torch.Tensor,
                              reset_every: int | None = None) -> list[torch.Tensor]:
        """tok_embed: (B, D, h) -> list of D+1 states [S_0, S_1, ..., S_D],
        each (B, h, h). S_0 is the pinned zero state (S2.2.2 Rev 2 minor 8:
        "no learned or nonzero S_0 is used anywhere in this design").

        `reset_every` (S2.9 item 4's last-K-window shortcut control, PINNED
        as EVAL-TIME truncation, S2.15 MODERATE-4): if set to K, the state
        is reset to S_0=0 immediately before processing every K-th macro
        position (1-indexed positions K+1, 2K+1, ... -- i.e. right before
        starting a new K-sized block), so the terminal read provably
        depends on at most the last K generators. None (default) = no
        truncation, the normal Arm 2/3 forward pass."""
        if self.widen is not None:
            # S2.20 F6: param-matching residual widen -- applied to the SAME
            # `tok_embed` leaf the blank-out test detaches (this function's
            # own parameter, not a separate re-derivation), BEFORE the
            # k/v/beta projections. Shape-preserving; does not touch the
            # recurrence formula or the (h,h) state shape below.
            tok_embed = tok_embed + self.widen(tok_embed)
        B, D, h = tok_embed.shape
        n_h = self.n_h
        k = self.k_proj(tok_embed).view(B, D, n_h, h)
        v = self.v_proj(tok_embed).view(B, D, n_h, h)
        # Householder maps require unit-norm keys (DeltaProduct's own
        # semantics, Siems et al. 2502.10297; matches beta_fla_smoke.py's
        # disclosed use_qk_l2norm_in_kernel=True convention).
        k = F.normalize(k, dim=-1, eps=1e-8)
        beta_raw = self.beta_proj(tok_embed).view(B, D, n_h)
        beta = self.beta_max * torch.sigmoid(beta_raw)

        eye = torch.eye(h, device=tok_embed.device, dtype=tok_embed.dtype).unsqueeze(0)
        S = tok_embed.new_zeros(B, h, h)
        states = [S]
        for t in range(D):
            if reset_every is not None and t > 0 and (t % reset_every) == 0:
                S = tok_embed.new_zeros(B, h, h)
            for j in range(n_h):
                kk = k[:, t, j, :]                                  # (B, h)
                vv = v[:, t, j, :]                                  # (B, h)
                bb = beta[:, t, j].view(B, 1, 1)                    # (B, 1, 1)
                kkT = kk.unsqueeze(-1) @ kk.unsqueeze(-2)            # (B, h, h)
                S = S @ (eye - bb * kkT) + bb * (vv.unsqueeze(-1) @ kk.unsqueeze(-2))
            states.append(S)
        return states

    def read_state(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, h, h) -- already the (B, 32, 32) memory shape (a lossless
        relabeling, not a projection, S2.2.2) -> Z: (B, d_state, d_state)."""
        return self.readout(S)

    def forward(self, token_idx: torch.Tensor, reset_every: int | None = None) -> torch.Tensor:
        """Terminal-state read only (the decisive scoring path, S2.9 item 3
        -- every arm scored on ONE terminal-state query per D_test point)."""
        tok_embed = self.embed_tokens(token_idx)
        states = self.states_from_embedding(tok_embed, reset_every=reset_every)
        return self.read_state(states[-1])

    def forward_all(self, token_idx: torch.Tensor, reset_every: int | None = None) -> list[torch.Tensor]:
        """Every intermediate Z_t, t=1..D -- the FREE per-step trajectory
        read (S2.3/S2.9 item 3), diagnostic-only (M-D0's convergence
        profile, the prefix-fidelity check), never folded into the decisive
        metric."""
        tok_embed = self.embed_tokens(token_idx)
        states = self.states_from_embedding(tok_embed, reset_every=reset_every)
        return [self.read_state(S) for S in states[1:]]


# ---------------------------------------------------------------------------
# Blank-out / P=1 bottleneck test -- MUST be re-verified for this new
# forward pass, not inherited from GroupWordEncoder's own result (a
# genuinely different computation graph, S2.2.2/S2.9 item 6). Mirrors
# blank_out.py's exact 3-part mechanism (sanity nonzero grad / leaf-detach
# None grad / signature check), adapted here because blank_out.py's own
# functions are typed against GroupWordModel specifically and this design's
# FILE OWNERSHIP forbids editing blank_out.py to generalize them.
# ---------------------------------------------------------------------------

def composer_scoring_fn(Z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The production scoring function's signature for this composer --
    a PURE function of (Z, target), mirroring blank_out.py::scoring_fn."""
    return cosine_loss(Z, target)


def run_composer_blank_out_test(composer: GroupWordDeltaComposer, device="cpu", B: int = 8, D: int = 4) -> None:
    print("=" * 60)
    print("  GroupWordDeltaComposer BLANK-OUT test: scoring path touches the raw word ONLY via Z")
    print("=" * 60)
    torch.manual_seed(0)
    d_state = composer.d_state
    token_idx = torch.randint(0, composer.n_gens, (B, D), device=device)
    target = torch.randn(B, d_state, d_state, device=device)

    # (a) sanity: with Z NOT detached, gradient DOES flow from a downstream
    # function of Z back to tok_embed (a real signal exists).
    tok_embed = composer.embed_tokens(token_idx).clone().detach().requires_grad_(True)
    states = composer.states_from_embedding(tok_embed)
    Zg = composer.read_state(states[-1])
    loss_g = composer_scoring_fn(Zg, target)
    g_embed = torch.autograd.grad(loss_g, tok_embed, retain_graph=True, allow_unused=True)[0]
    assert g_embed is not None and g_embed.abs().sum() > 0, \
        "the word embedding doesn't affect the score at all?!"
    print(f"  (a) sanity: grad(score, tok_embed) nonzero (sum|grad|={g_embed.abs().sum().item():.4f}) OK")

    # (b) meaningful check: with Z detached to an INDEPENDENT leaf, the SAME
    # tok_embed must have NO remaining path to the score.
    Z_leaf = Zg.detach().clone().requires_grad_(True)
    loss_leaf = composer_scoring_fn(Z_leaf, target)
    g_leak = torch.autograd.grad(loss_leaf, tok_embed, allow_unused=True)[0]
    assert g_leak is None, "LEAK: the raw word reaches the score outside Z"
    print("  (b) leaf-detach: grad(score(Z_leaf), tok_embed) is None -- no path around Z  OK")

    # (c) signature check.
    sig = set(inspect.signature(composer_scoring_fn).parameters)
    assert sig <= {"Z", "target"}, f"composer_scoring_fn takes inputs beyond (Z, target): {sig}"
    print(f"  (c) inspect.signature(composer_scoring_fn) = {sorted(sig)} subset of {{'Z','target'}}  OK")

    print("\n" + "=" * 60 + "\n  COMPOSER BLANK-OUT TEST PASSED\n" + "=" * 60)


def _make_leaky_composer_scoring_fn(tok_embed_ref: torch.Tensor):
    """Planted-leak negative test fixture, mirrors blank_out.py's
    `_make_leaky_scoring_fn` exactly (nonzero coefficient -- a zero
    coefficient has zero LOCAL gradient too and would silently defeat this
    exact test, not just its forward value)."""
    def leaky_scoring_fn(Z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cosine_loss(Z, target) + 1e-3 * tok_embed_ref.mean()
    return leaky_scoring_fn


def run_composer_blank_out_planted_leak_test(composer: GroupWordDeltaComposer, device="cpu",
                                             B: int = 8, D: int = 4) -> None:
    print("=" * 60)
    print("  NEGATIVE TEST -- composer blank-out CATCHES a planted tok_embed leak")
    print("=" * 60)
    torch.manual_seed(1)
    token_idx = torch.randint(0, composer.n_gens, (B, D), device=device)
    target = torch.randn(B, composer.d_state, composer.d_state, device=device)

    tok_embed = composer.embed_tokens(token_idx).clone().detach().requires_grad_(True)
    states = composer.states_from_embedding(tok_embed)
    Zg = composer.read_state(states[-1])
    Z_leaf = Zg.detach().clone().requires_grad_(True)

    leaky_fn = _make_leaky_composer_scoring_fn(tok_embed)
    loss_leak = leaky_fn(Z_leaf, target)
    g_leak = torch.autograd.grad(loss_leak, tok_embed, allow_unused=True)[0]
    assert g_leak is not None and g_leak.abs().sum() > 0, \
        "blank-out FAILED to catch a planted tok_embed leak (no teeth)"
    print(f"  planted leak CAUGHT: grad(leaky_score(Z_leaf), tok_embed) nonzero "
          f"(sum|grad|={g_leak.abs().sum().item():.4f})  OK")

    loss_honest = composer_scoring_fn(Z_leaf, target)
    g_honest = torch.autograd.grad(loss_honest, tok_embed, allow_unused=True)[0]
    assert g_honest is None, "honest composer_scoring_fn path unexpectedly leaks under the same mechanism"
    print("  honest composer_scoring_fn path (same mechanism, same tensors): grad is still None  OK")
    print("\n" + "=" * 60 + "\n  PLANTED-LEAK NEGATIVE TEST PASSED\n" + "=" * 60)


# ---------------------------------------------------------------------------
# One-cell fla numerical cross-check (S2.2.2 Rev 1's kernel adjudication) --
# CUDA+real-fla only, box-only-verifiable (fla's Triton kernel is CUDA/
# bf16-only, and the CPU stub's simplified recurrence has no genuine
# negative-eigenvalue dynamics to compare against -- disclosed, matching
# beta_fla_smoke.py's own convention). Self-skips loudly here.
# ---------------------------------------------------------------------------

FLA_CROSS_CHECK_CONFIGS = ((1, 1), (1, 8), (2, 8))   # (n_h, D), S2.2.2 pinned
FLA_CROSS_CHECK_TOL_ALL = 5e-2
FLA_CROSS_CHECK_TOL_SINGLE_STEP = 1e-2


def fla_cross_check(device: str = "cpu", h: int = 32, B: int = 4) -> dict:
    """PASS iff rel-Frobenius ||S_torch - S_fla||_F / ||S_torch||_F <= 5e-2
    at every config in FLA_CROSS_CHECK_CONFIGS AND <= 1e-2 for the
    single-step (n_h=1, D=1) config (S2.2.2 Rev 1, pinned tolerances)."""
    print("=" * 88)
    print("  stage2_composer.py -- ONE-CELL fla NUMERICAL CROSS-CHECK (S2.2.2 Rev 1)")
    print("=" * 88)
    import beta_fla_smoke as bfs
    is_stub = bfs.ensure_fla_stub()
    if is_stub or not torch.cuda.is_available():
        print("  *** SKIPPED (CPU stub and/or no CUDA) ***")
        print("  fla's Triton kernel is CUDA/bf16-only and the CPU stub's recurrence has no")
        print("  genuine negative-eigenvalue dynamics -- a comparison against it would not be")
        print("  evidence about the real kernel (mirrors beta_fla_smoke.py's own box-only-skip")
        print("  convention). REGISTERED FOR THE BOX: re-run with real fla + CUDA before the")
        print("  Stage-2 calibration gate trusts the bespoke recurrence (S2.11 build item).")
        print("=" * 88)
        return {"status": "skipped_cpu_or_stub", "box_only": True}

    from fla.ops.delta_rule import chunk_delta_rule
    results = []
    torch.manual_seed(0)
    for n_h, D in FLA_CROSS_CHECK_CONFIGS:
        composer = GroupWordDeltaComposer(d_state=5, n_gens=4, h=h, n_h=n_h, beta_max=2.0).to(device)
        token_idx = torch.randint(0, 4, (B, D), device=device)
        tok_embed = composer.embed_tokens(token_idx)
        states = composer.states_from_embedding(tok_embed)
        S_torch = states[-1]                                        # (B, h, h), fp32

        # S2.20 F6: `states_from_embedding` now widens `tok_embed` internally
        # (param-matching residual, GroupWordDeltaComposer docstring) --
        # this manual re-derivation of k/v/beta for the fla comparison must
        # apply the SAME widen step, or S_torch (widened) and this
        # recomputation (unwidened) would silently diverge, making the
        # cross-check compare two different computations, not a genuine
        # torch-vs-fla numerical check of the SAME recurrence.
        tok_embed_widened = tok_embed + composer.widen(tok_embed) if composer.widen is not None else tok_embed
        k = composer.k_proj(tok_embed_widened).view(B, D, n_h, h)
        v = composer.v_proj(tok_embed_widened).view(B, D, n_h, h)
        k = F.normalize(k, dim=-1, eps=1e-8)
        beta = composer.beta_max * torch.sigmoid(composer.beta_proj(tok_embed_widened).view(B, D, n_h))
        q = torch.zeros_like(k)   # q does not affect final_state (module docstring); a throwaway
                                  # tensor purely to satisfy chunk_delta_rule's input signature.
        q_exp = q.reshape(B, D * n_h, 1, h).to(torch.bfloat16)
        k_exp = k.reshape(B, D * n_h, 1, h).to(torch.bfloat16)
        v_exp = v.reshape(B, D * n_h, 1, h).to(torch.bfloat16)
        beta_exp = beta.reshape(B, D * n_h, 1).to(torch.bfloat16)
        _o, final_state = chunk_delta_rule(q_exp, k_exp, v_exp, beta_exp,
                                           use_qk_l2norm_in_kernel=True, allow_neg_eigval=True)
        S_fla = final_state.squeeze(1).to(torch.float32)             # (B, h, h)

        rel_fro = (S_torch - S_fla).norm() / S_torch.norm().clamp(min=1e-12)
        tol = FLA_CROSS_CHECK_TOL_SINGLE_STEP if (n_h, D) == (1, 1) else FLA_CROSS_CHECK_TOL_ALL
        ok = rel_fro.item() <= tol
        results.append(dict(n_h=n_h, D=D, rel_fro=rel_fro.item(), tol=tol, ok=ok))
        print(f"  (n_h={n_h}, D={D}): rel-Frobenius={rel_fro.item():.4e} <= {tol}: {'PASS' if ok else 'FAIL'}")

    all_ok = all(r["ok"] for r in results)
    print(f"\nRESULT: fla cross-check {'PASSED' if all_ok else 'FAILED'} at all {len(results)} configs.")
    print("=" * 88)
    return {"status": "ran_real_fla", "box_only": False, "results": results, "all_ok": all_ok}


# ---------------------------------------------------------------------------
# Smoke: forward/backward at D=1 and D=64 (train/decisive extremes), rank
# bound (proven exact, checked structurally), n_h in {1,2,4}, use_bos_row,
# last-K-window truncation, blank-out (+ planted-leak negative), fla
# cross-check (box-only, self-skips here).
# ---------------------------------------------------------------------------

def smoke(device="cpu"):
    print("=" * 88)
    print("  GroupWordDeltaComposer SMOKE")
    print("=" * 88)
    torch.manual_seed(0)
    d_state, n_gens = 5, 4   # S4/A5-shaped (d_min=3, d_state=5)

    for n_h in (1, 2, 4):
        for beta_max in (1.0, 2.0):
            composer = GroupWordDeltaComposer(d_state, n_gens, h=32, n_h=n_h, beta_max=beta_max).to(device)
            for D in (1, 8, 64):
                token_idx = torch.randint(0, n_gens, (4, D), device=device)
                target = torch.randn(4, d_state, d_state, device=device)
                Z = composer(token_idx)
                assert Z.shape == (4, d_state, d_state), Z.shape
                assert not torch.isnan(Z).any() and not torch.isinf(Z).any()
                loss = cosine_loss(Z, target)
                composer.zero_grad()
                loss.backward()
                for name, p in composer.named_parameters():
                    assert p.grad is not None, f"n_h={n_h} beta_max={beta_max} D={D}: no grad for {name}"
                    assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), \
                        f"n_h={n_h} beta_max={beta_max} D={D}: bad grad for {name}"
                # rank bound (exact structural fact, module docstring proof) --
                # torch.linalg.matrix_rank uses a numerically robust default
                # tolerance, so this is a real, always-true assertion, not a
                # statistical tendency.
                with torch.no_grad():
                    tok_embed = composer.embed_tokens(token_idx)
                    S_D = composer.states_from_embedding(tok_embed)[-1]
                    bound = GroupWordDeltaComposer.rank_bound(D, n_h, h=32)
                    ranks = torch.linalg.matrix_rank(S_D)
                    assert (ranks <= bound).all(), \
                        f"n_h={n_h} D={D}: rank(S_D)={ranks.tolist()} exceeds the PROVEN bound {bound}"
                print(f"  n_h={n_h} beta_max={beta_max} D={D:>2}: forward {tuple(Z.shape)}, "
                      f"loss={loss.item():.4f}, rank(S_D)<={bound} (observed {ranks.tolist()}), "
                      f"all params finite grad  OK")

    print("\n  S2.20 F6 -- param-match vs Arm 1 (GroupWordModel), all 5 groups x n_h in {1,2,4}:")
    from groups import GROUP_NAMES, D_STATE, generating_set
    from group_word_encoder import GroupWordModel
    for g_name in GROUP_NAMES:
        g_d_state, g_n_gens = D_STATE[g_name], len(generating_set(g_name))
        arm1_ref = GroupWordModel(g_d_state, g_n_gens, L_max=16, h=32)
        arm1_params = count_params(arm1_ref)
        for n_h in (1, 2, 4):
            comp = GroupWordDeltaComposer(g_d_state, g_n_gens, h=32, n_h=n_h, beta_max=2.0)
            pm = check_param_match(count_params(comp), arm1_params)
            assert pm["within_tolerance"], (
                f"S2.20 F6 param-match FAILED for {g_name} n_h={n_h}: composer="
                f"{pm['composer_params']} Arm1={pm['arm1_params']} delta={pm['delta_frac'] * 100:+.1f}% "
                f"(tolerance +-{pm['tolerance'] * 100:.0f}%)"
            )
            print(f"    {g_name} n_h={n_h}: composer={pm['composer_params']:6d}  Arm1={pm['arm1_params']:6d}  "
                  f"delta={pm['delta_frac'] * 100:+5.1f}%  within_tolerance={pm['within_tolerance']}")

    print("\n  use_bos_row=True path:")
    composer_bos = GroupWordDeltaComposer(d_state, n_gens, h=32, n_h=2, beta_max=2.0,
                                          use_bos_row=True).to(device)
    token_idx = torch.randint(0, n_gens, (4, 5), device=device)
    target = torch.randn(4, d_state, d_state, device=device)
    Z = composer_bos(token_idx)
    assert Z.shape == (4, d_state, d_state)
    cosine_loss(Z, target).backward()
    assert composer_bos.readout.bos_row.grad is not None and \
        not torch.isnan(composer_bos.readout.bos_row.grad).any()
    print("  bos_row present, receives finite gradient  OK")

    print("\n  last-K-window truncation (reset_every):")
    composer_k = GroupWordDeltaComposer(d_state, n_gens, h=32, n_h=2, beta_max=2.0).to(device)
    token_idx = torch.randint(0, n_gens, (4, 8), device=device)
    tok_embed = composer_k.embed_tokens(token_idx)
    full_states = composer_k.states_from_embedding(tok_embed, reset_every=None)
    trunc_states = composer_k.states_from_embedding(tok_embed, reset_every=4)
    # after a reset at position 5 (t=4, 0-indexed loop var), the terminal
    # state (t=8, last 4 positions only) must differ from the untruncated one.
    assert not torch.allclose(full_states[-1], trunc_states[-1]), \
        "reset_every=4 did not change the terminal state -- truncation control has no teeth"
    # a FRESH 4-position word run standalone must match the truncated tail
    # exactly (the truncation control genuinely forgets everything before
    # the last reset boundary).
    tail_embed = tok_embed[:, 4:]
    standalone_states = composer_k.states_from_embedding(tail_embed, reset_every=None)
    assert torch.allclose(trunc_states[-1], standalone_states[-1], atol=1e-5), \
        "truncated tail does not match a standalone last-K-length forward -- truncation is leaking"
    print("  reset_every=4 differs from the untruncated terminal state, and exactly matches a")
    print("  standalone last-4-position forward (no leakage across the reset boundary)  OK")

    print("\n  forward_all trajectory read (every intermediate Z_t):")
    all_Z = composer_k.forward_all(token_idx)
    assert len(all_Z) == 8
    assert all(z.shape == (4, d_state, d_state) for z in all_Z)
    print(f"  forward_all returned {len(all_Z)} intermediate states, all correctly shaped  OK")

    print("\n  fla cross-check (box-only -- expect self-skip in this CPU build):")
    fla_result = fla_cross_check(device=device)
    print(f"  fla_cross_check status={fla_result['status']}")

    print("\n" + "=" * 88 + "\n  GroupWordDeltaComposer SMOKE PASSED\n" + "=" * 88)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smoke(dev)
    m = GroupWordDeltaComposer(d_state=5, n_gens=4, h=32, n_h=2, beta_max=2.0).to(dev)
    run_composer_blank_out_test(m, device=dev)
    run_composer_blank_out_planted_leak_test(m, device=dev)
