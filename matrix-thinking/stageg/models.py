"""Stage G Regime-2 models — see STAGE_G_DESIGN.md section 4/6.1 (frozen,
revision 2.1).

Two model families:

  MatrixThinkerG   -- the "matrix-all baseline" plus every Wave-A OFAT
                       intervention axis (H_j, H_a, H_f, H_g, H_i), all
                       CLI-selectable. With every intervention at its
                       matrix-native default (embed_rank='rank1',
                       embed_init='default', output_head='multiprobe',
                       n_layers=8, iter_cond=False) this is behaviorally
                       identical to round2_matrix_script.py's MatrixThinker
                       (see test_stageg.py's
                       test_baseline_identical_to_round2_script) except for
                       the vocab_size/max_len constructor args (Regime 2 is
                       byte-vocab, design section 5.4) -- no wiring
                       differs.

  VectorReferenceModel -- the "vector-all baseline": a byte-vocab-scaled
                       LoopFormer clone (design section 12: LoopFormer is
                       the exact vector-side architecture Stage G
                       diagnoses; its own defaults -- dense tied embedding,
                       AdaLN per-loop conditioning -- are already the
                       "alternative" state of the H_f/H_i axes, so no
                       intervention is needed to reach them). A few
                       mirror-relevant knobs are exposed for Wave B's
                       bidirectional top-1 confirmation (design section
                       6.1: "Wave B mirrors ... from the vector-all
                       baseline direction (flip toward matrix)"); NOT every
                       axis has a natural vector-side mirror built yet (see
                       the module-level NOTE below) -- this is a documented,
                       deliberate scope cut (Wave B's manifest only needs
                       ONE winning axis, chosen after Wave A finishes; the
                       design itself defers this choice to a human gate).

Both share the identical byte-vocab embedding/ops components (`common.py`)
and the identical tied-head classes (`tied_heads.py`) so an H_f flip means
literally the same class on both sides.

NOTE on Wave-B mirror completeness (flag for the auditor): VectorReferenceModel
supports tie/untie (H_f), iter_cond on/off (H_i), and an embed-init-std
override (H_j) directly, because these map onto existing LoopFormer knobs.
H_a's mirror ("rank-1-constrain the vector model's embedding") has no
natural LoopFormer analog (LoopFormer's wte is already a dense full-rank
table with no outer-product structure to remove) -- VectorReferenceModel's
`embed_rank='rank1'` option below implements one *defensible* construction
(a d-wide outer-product bottleneck up-projected to n_embd via a shared
Linear(d, n_embd)) but this is a genuine judgment call, not a design-
specified construction, and is called out again in the build report.
H_g's depth-match axis has no vector-side mirror at all (LoopFormer's
2-shared-blocks-times-8-loops structure IS already what H_g asks the
matrix side to imitate -- there is nothing to flip on the vector side).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from common import (
    MatrixRMSNorm, ThinkingBlock, MultiProbeHead, TimestepEmbedder,
    CausalSelfAttention, FactoredDenseProjection,
)
from tied_heads import TiedBilinearHead, TiedFactoredHead

OUTPUT_HEADS = ("multiprobe", "tied_bilinear", "tied_factored")


# ═══════════════════════════════════════════════════════════════════════════
# Matrix-side embedding — H_j (init scale) and H_a (rank) live here
# ═══════════════════════════════════════════════════════════════════════════

class MatrixEmbedding(nn.Module):
    """rank='rank1' (matrix-native default): M = u (x) v, outer product.
    rank='full' (H_a alternative): direct (vocab,d,d) table, no rank
    constraint at embedding time -- the "reshape equivalence" construction
    CLAUDE.md's hard rule names ("any d^2-dim vector can be reshaped to
    d x d matrix ... structure only matters if OPERATIONS preserve it").

    init_scale='default': PyTorch's nn.Embedding default init (std=1.0) --
      the diagnosed F6/H_j bug: for rank1 this makes the OUTER PRODUCT
      enter the residual stream at entry-std ~= 1 (1.0^2), NOT 1.0, because
      products have std=sigma^2 (CLAUDE.md hard rule). For 'full' rank
      there is no product, so 'default' targets the SAME ~1.0 entry-std
      directly, to isolate the rank axis alone from the init-scale axis.
    init_scale='matched': std=sqrt(0.02) on the rank1 FACTORS (so the
      product enters at std=0.02, matching LoopFormer's explicit init) --
      or std=0.02 DIRECTLY on the full-rank table (no squaring needed,
      there is no product).
    """

    def __init__(self, d, vocab_size, max_len, rank="rank1", init_scale="default"):
        super().__init__()
        if rank not in ("rank1", "full"):
            raise ValueError(f"rank must be 'rank1' or 'full', got {rank!r}")
        if init_scale not in ("default", "matched"):
            raise ValueError(f"init_scale must be 'default' or 'matched', got {init_scale!r}")
        self.d = d
        self.rank = rank
        self.init_scale = init_scale
        if rank == "rank1":
            self.embed_u = nn.Embedding(vocab_size, d)
            self.embed_v = nn.Embedding(vocab_size, d)
            self.pos_u = nn.Embedding(max_len, d)
            self.pos_v = nn.Embedding(max_len, d)
            if init_scale == "matched":
                s = math.sqrt(0.02)   # CLAUDE.md hard rule: factor std = sqrt(target product std)
                for emb in (self.embed_u, self.embed_v, self.pos_u, self.pos_v):
                    nn.init.normal_(emb.weight, mean=0.0, std=s)
            # 'default': leave PyTorch's std=1.0 default (the diagnosed bug, F6)
        else:
            self.embed_full = nn.Embedding(vocab_size, d * d)
            self.pos_full = nn.Embedding(max_len, d * d)
            s = 0.02 if init_scale == "matched" else 1.0   # direct entry std, no product involved
            nn.init.normal_(self.embed_full.weight, mean=0.0, std=s)
            nn.init.normal_(self.pos_full.weight, mean=0.0, std=s)

    def forward(self, token_ids):
        B, L = token_ids.shape
        d = self.d
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        if self.rank == "rank1":
            u, v = self.embed_u(token_ids), self.embed_v(token_ids)
            M = torch.einsum('...i,...j->...ij', u, v)
            pu, pv = self.pos_u(pos), self.pos_v(pos)
            M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1
        else:
            M = self.embed_full(token_ids).reshape(B, L, d, d)
            P = self.pos_full(pos).reshape(B, L, d, d)
            M = M + P * 0.1
        return M

    def entry_std(self, token_ids=None, n_sample=4096, device="cpu"):
        """Free H_j diagnostic (design section 4 item 1 / 6.2 item 2): the
        step-0 std of the raw embedding contribution to the residual
        stream, BEFORE any position term. Instrumented (real forward pass
        over sampled ids), not hand-derived."""
        with torch.no_grad():
            if token_ids is None:
                token_ids = torch.randint(
                    0, self.embed_u.num_embeddings if self.rank == "rank1" else self.embed_full.num_embeddings,
                    (n_sample,), device=device)
            if self.rank == "rank1":
                u, v = self.embed_u(token_ids), self.embed_v(token_ids)
                M = torch.einsum('...i,...j->...ij', u, v)
            else:
                M = self.embed_full(token_ids).reshape(-1, self.d, self.d)
            return M.float().std().item()


def _make_output_head(output_head, d, vocab_size, K, embed_u=None, embed_v=None):
    if output_head == "multiprobe":
        return MultiProbeHead(d, vocab_size, K)
    if output_head == "tied_bilinear":
        if embed_u is None or embed_v is None:
            raise ValueError("tied_bilinear (H_f form i) requires rank-1 embed_u/embed_v")
        return TiedBilinearHead(embed_u, embed_v)
    if output_head == "tied_factored":
        if embed_u is None or embed_v is None:
            raise ValueError("tied_factored (H_f form ii) requires rank-1 embed_u/embed_v")
        return TiedFactoredHead(embed_u, embed_v, n_probes=K)
    raise ValueError(f"output_head must be one of {OUTPUT_HEADS}, got {output_head!r}")


# ═══════════════════════════════════════════════════════════════════════════
# Matrix-all baseline + interventions
# ═══════════════════════════════════════════════════════════════════════════

class MatrixThinkerG(nn.Module):
    def __init__(self, mat_dim=32, n_layers=8, n_heads=8, n_probes=None,
                 max_len=1024, vocab_size=256, dropout=0.1, n_iterations=8,
                 embed_rank="rank1", embed_init="default",
                 output_head="multiprobe", iter_cond=False, proj_rank=None):
        super().__init__()
        self.mat_dim = d = mat_dim
        self.n_iterations_default = n_iterations
        self.iter_cond = iter_cond
        self.output_head_type = output_head
        self.proj_rank = proj_rank

        self.embed = MatrixEmbedding(d, vocab_size, max_len, rank=embed_rank, init_scale=embed_init)

        cond_dim = d if iter_cond else None
        if iter_cond:
            self.time_embedder = TimestepEmbedder(d)
            self.dt_embedder = TimestepEmbedder(d)

        # H_b (gated, design section 4 item 8): proj_rank=None (default)
        # keeps RowThenColProjection -- the baseline path, bit-identical to
        # round2 (test [3]); an integer r swaps EVERY projection in every
        # block for the rank-r factored dense family.
        proj_factory = (lambda dd: FactoredDenseProjection(dd, proj_rank)) if proj_rank else None

        self.layers = nn.ModuleList([ThinkingBlock(d, n_heads, dropout, cond_dim=cond_dim,
                                                    proj_factory=proj_factory)
                                      for _ in range(n_layers)])
        self.final_norm = MatrixRMSNorm(d)

        K = n_probes or d
        eu = self.embed.embed_u if embed_rank == "rank1" else None
        ev = self.embed.embed_v if embed_rank == "rank1" else None
        self.output_head = _make_output_head(output_head, d, vocab_size, K, eu, ev)

    def _cond_at(self, t, n_iterations, B, device, dtype):
        if not self.iter_cond:
            return None
        ti = torch.full((B,), t / n_iterations, dtype=dtype, device=device)
        dt = torch.full((B,), 1.0 / n_iterations, dtype=dtype, device=device)
        return self.time_embedder(ti) + self.dt_embedder(dt)

    def _one_iteration(self, M, cond):
        for layer in self.layers:
            if self.training:
                if cond is not None:
                    M = torch.utils.checkpoint.checkpoint(layer, M, cond, use_reentrant=False)
                else:
                    M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else:
                M = layer(M, cond) if cond is not None else layer(M)
        return M

    def backbone(self, token_ids, n_iterations=None, measure_ranks=False):
        """Everything forward() does BEFORE the output head: embed ->
        n_iterations of the (possibly conditioned) ThinkingBlock stack ->
        final_norm. Factored out so N3's temperature calibration
        (train_stageg.py's `calibrate_tied_bilinear_if_needed`) can obtain
        a REAL step-0 M_n sample without duplicating this logic (and
        risking drift between the two)."""
        n_iterations = n_iterations or self.n_iterations_default
        B, L = token_ids.shape
        d = self.mat_dim
        M = self.embed(token_ids)

        ranks = []
        for t in range(n_iterations):
            cond = self._cond_at(t, n_iterations, B, M.device, M.dtype)
            M = self._one_iteration(M, cond)
            if measure_ranks:
                with torch.no_grad():
                    Mf = M.detach().float().reshape(-1, d, d)
                    if Mf.shape[0] > 512:
                        Mf = Mf[torch.randperm(Mf.shape[0], device=Mf.device)[:512]]
                    S = torch.linalg.svdvals(Mf).clamp(min=1e-10)
                    Sn = S / S.sum(-1, keepdim=True)
                    ranks.append((-(Sn * Sn.log()).sum(-1)).exp().mean().item())

        return self.final_norm(M), ranks

    def forward(self, token_ids, n_iterations=None, measure_ranks=False):
        M_n, ranks = self.backbone(token_ids, n_iterations=n_iterations, measure_ranks=measure_ranks)
        logits = self.output_head(M_n)
        return logits, {'ranks': ranks}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self):
        """Mirrors round2_matrix_script.py's logged (embed/think/head) split
        (design build requirement ii: every run logs param count
        independently). Relies on nn.Module.named_parameters()'s default
        de-duplication of shared tensors (tied heads' embed_u/embed_v
        report ONLY under 'embed.*', since self.embed is registered before
        self.output_head) -- verified explicitly in test_stageg.py's tie
        test, not just assumed here."""
        embed_p = sum(p.numel() for n, p in self.named_parameters() if n.startswith('embed.'))
        think_p = sum(p.numel() for n, p in self.named_parameters() if n.startswith('layers.'))
        head_p = sum(p.numel() for n, p in self.named_parameters()
                     if n.startswith('output_head.') or n.startswith('final_norm.'))
        cond_p = sum(p.numel() for n, p in self.named_parameters() if 'embedder' in n)
        return {"embed": embed_p, "think": think_p, "head": head_p, "cond": cond_p}


# ═══════════════════════════════════════════════════════════════════════════
# Vector-all baseline: byte-vocab LoopFormer clone
# ═══════════════════════════════════════════════════════════════════════════

class VectorEmbedding(nn.Module):
    """Vector-side embedding. 'dense' (default, LoopFormer's own state):
    standard full-rank nn.Embedding(vocab, n_embd). 'rank1' (H_a Wave-B
    mirror, judgment-call construction -- see module docstring NOTE): a
    d-wide outer-product bottleneck up-projected to n_embd via a SHARED
    Linear(d, n_embd, bias=False)."""

    def __init__(self, n_embd, vocab_size, max_len, rank="dense", d_bottleneck=32,
                 init_std=0.02):
        super().__init__()
        self.rank = rank
        if rank == "dense":
            self.wte = nn.Embedding(vocab_size, n_embd)
            self.wpe = nn.Embedding(max_len, n_embd)
            nn.init.normal_(self.wte.weight, mean=0.0, std=init_std)
            nn.init.normal_(self.wpe.weight, mean=0.0, std=init_std)
        elif rank == "rank1":
            d = d_bottleneck
            self.embed_u = nn.Embedding(vocab_size, d)
            self.embed_v = nn.Embedding(vocab_size, d)
            self.pos_u = nn.Embedding(max_len, d)
            self.pos_v = nn.Embedding(max_len, d)
            s = math.sqrt(init_std) if init_std < 1 else init_std
            for emb in (self.embed_u, self.embed_v, self.pos_u, self.pos_v):
                nn.init.normal_(emb.weight, mean=0.0, std=s)
            self.up_proj = nn.Linear(d, n_embd, bias=False)
        else:
            raise ValueError(rank)

    def forward(self, token_ids):
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        if self.rank == "dense":
            return self.wte(token_ids) + self.wpe(pos)
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1
        # collapse the rank-1 (d,d) matrix to a d-vector before up-projecting
        # (row-sum -- the cheapest, most literal analog of "the outer
        # product's factor information", not a claim of equivalence).
        flat = M.sum(dim=-1)
        return self.up_proj(flat)


class VectorLoopBlock(nn.Module):
    """Verbatim LoopFormerBlock behavior, with iter_cond OPTIONAL (default
    True, matching LoopFormer's own always-on AdaLN) so the H_i Wave-B
    mirror ("strip the vector side's conditioning") is a real, cheap flip."""

    def __init__(self, n_embd, n_head, intermediate_dim, dropout=0.1, iter_cond=True):
        super().__init__()
        self.iter_cond = iter_cond
        self.norm_1 = nn.RMSNorm(n_embd, elementwise_affine=False)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.norm_2 = nn.RMSNorm(n_embd, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, n_embd),
            nn.Dropout(dropout),
        )
        if iter_cond:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, 4 * n_embd, bias=True))

    def forward(self, x, c=None):
        if self.iter_cond and c is not None:
            shift_attn, scale_attn, shift_mlp, scale_mlp = self.adaLN(c).chunk(4, dim=-1)
        else:
            shift_attn = scale_attn = shift_mlp = scale_mlp = 0.0
        h = self.norm_1(x)
        h = h * (1 + (scale_attn if isinstance(scale_attn, float) else scale_attn.unsqueeze(1))) \
            + (shift_attn if isinstance(shift_attn, float) else shift_attn.unsqueeze(1))
        x = x + self.attn(h)
        h = self.norm_2(x)
        h = h * (1 + (scale_mlp if isinstance(scale_mlp, float) else scale_mlp.unsqueeze(1))) \
            + (shift_mlp if isinstance(shift_mlp, float) else shift_mlp.unsqueeze(1))
        x = x + self.mlp(h)
        return x


class VectorReferenceModel(nn.Module):
    """Byte-vocab LoopFormer clone -- the Regime-2 vector-all baseline.
    Defaults (embed='dense' tied, iter_cond=True) reproduce LoopFormer's
    OWN architecture exactly (design section 12: the exact vector-side
    model Stage G diagnoses), re-sized for byte vocab at a MEASURED
    d=32-matched param budget (audit MAJOR-2): at the Regime-2 standard
    config (vocab=256, max_len=1024, n_layer=2, intermediate=128),
    MatrixThinkerG baseline = 290,328 params; this model at the n_embd=80
    default = 300,976 (matrix/vector ratio 0.965). The build's original
    n_embd=64 guess was 222,400 (a 30.5% mismatch) -- corrected here, and
    continuously re-verified by test_stageg.py's param-budget check."""

    def __init__(self, n_embd=80, n_head=4, n_layer=2, n_loops=8, intermediate_dim=128,
                 max_len=1024, vocab_size=256, dropout=0.1,
                 embed_rank="dense", embed_init_std=0.02, tie_head=True, iter_cond=True):
        super().__init__()
        self.n_loops_default = n_loops
        self.iter_cond = iter_cond
        self.tie_head = tie_head

        self.embed = VectorEmbedding(n_embd, vocab_size, max_len, rank=embed_rank,
                                      init_std=embed_init_std)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            VectorLoopBlock(n_embd, n_head, intermediate_dim, dropout, iter_cond=iter_cond)
            for _ in range(n_layer)
        ])
        if iter_cond:
            self.time_embedder = TimestepEmbedder(n_embd)
            self.dt_embedder = TimestepEmbedder(n_embd)
        self.norm_f = nn.RMSNorm(n_embd)

        if tie_head and embed_rank == "dense":
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.embed.wte.weight = self.lm_head.weight   # weight tying, H_f "matrix-native"->"alt" flip target
        else:
            # untied (H_f Wave-B mirror: "does UNtying hurt the vector side
            # symmetrically?"), or embed_rank='rank1' where there is no
            # single dense table to tie to.
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=embed_init_std)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not getattr(self, 'lm_head', None):
                pass  # projection layers keep their own component-local init (matches LoopFormerBlock/CausalSelfAttention/TimestepEmbedder defaults)

    def _one_loop(self, x, c):
        for block in self.blocks:
            x = block(x, c)
        return x

    def forward(self, token_ids, n_loops=None, n_iterations=None, measure_ranks=False, **kwargs):
        # n_iterations accepted as an alias so callers can share one
        # training loop across MatrixThinkerG and VectorReferenceModel.
        n_loops = n_loops or n_iterations or self.n_loops_default
        B, L = token_ids.shape
        device = token_ids.device

        x = self.drop(self.embed(token_ids))

        steps = [1.0 / n_loops] * n_loops
        ti = torch.zeros(B, dtype=x.dtype, device=device)
        for dt in steps:
            c = None
            if self.iter_cond:
                dt_base = torch.ones_like(ti) * dt
                c = self.time_embedder(ti) + self.dt_embedder(dt_base)
            if self.training and n_loops > 1:
                x = torch.utils.checkpoint.checkpoint(self._one_loop, x, c, use_reentrant=False)
            else:
                x = self._one_loop(x, c)
            ti = ti + dt

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, {'ranks': [], 'model_type': 'vector_reference'}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self):
        embed_p = sum(p.numel() for n, p in self.named_parameters()
                      if n.startswith('embed.') or n.startswith('lm_head.'))
        think_p = sum(p.numel() for n, p in self.named_parameters() if n.startswith('blocks.'))
        cond_p = sum(p.numel() for n, p in self.named_parameters() if 'embedder' in n)
        head_p = sum(p.numel() for n, p in self.named_parameters() if n.startswith('norm_f.'))
        return {"embed": embed_p, "think": think_p, "head": head_p, "cond": cond_p}


# ═══════════════════════════════════════════════════════════════════════════
# CLI-facing intervention spec + factory (single source of truth, shared by
# train_stageg.py, wave_neg1.py, run_stageg_sweep.py, test_stageg.py)
# ═══════════════════════════════════════════════════════════════════════════

MATRIX_ALL_BASELINE = dict(embed_rank="rank1", embed_init="default",
                            output_head="multiprobe", n_layers=8, iter_cond=False)

VARIANT_AXES = {
    # variant name -> dict of MatrixThinkerG kwargs overriding MATRIX_ALL_BASELINE
    "baseline": {},
    "h_j_init_matched": dict(embed_init="matched"),
    "h_a_full_rank": dict(embed_rank="full"),
    "h_f_tied_bilinear": dict(output_head="tied_bilinear"),
    "h_g_depth_matched": dict(n_layers=2),
    "h_i_iter_cond": dict(iter_cond=True),
    # Pre-registered N3 follow-up (design section 4 item 4): "if the
    # primary H_f cell still lands far below the matrix-all baseline, H_f
    # is re-run once on the matched-init (H_j-fixed) variant before any
    # verdict is recorded" -- registered here so triggering that branch is
    # a --variant flag, never a mid-experiment code edit. Deliberately a
    # TWO-axis flip, so it is NOT part of the Wave-A OFAT screen (see
    # WAVE_A_SCREEN_AXES).
    "h_f_tied_matched_init": dict(output_head="tied_bilinear", embed_init="matched"),
    # Pre-registered COMBINED PROBE (design section 9 attack #4): "if Wave
    # A produces zero survivors singly, a combined probe (top-2 factors
    # together) is pre-registered before declaring a distributed-tax
    # verdict." Triggered 2026-07-02: zero single-axis survivors
    # (recovered_frac: h_j -0.003, h_a -0.003, h_g -0.05, h_f -0.163).
    # Axes combined, and why these three:
    #   H_g (n_layers=2) + H_i (iter_cond=True) -- the interacting pair
    #     attack #4 itself names ("a depth-matched model has fewer
    #     independent weight sets, which changes ... the value of a
    #     conditioning signal"); section 4 item 5 already scopes H_g's
    #     verdict as conditional on H_i. H_b, the attack's third named
    #     factor, is gated out of Wave A by design.
    #   H_j (embed_init="matched") -- section 1: "upstream of every other
    #     hypothesis"; section 4 item 3: the ~50x step-0 scale mismatch
    #     "changes the optimization landscape every other component
    #     operates in"; a zero-param/zero-FLOP flip, so including it
    #     cannot confound the param match.
    # Reading note (documented deviation): read strictly off measured
    # single-axis recovered_frac, "top-2" would be h_j/h_a (each ~0
    # alone); the mechanistic reading -- the attack's own named
    # interaction, which is what the probe exists to expose -- is
    # H_g x H_i. This cell takes the union of both readings minus h_a (a
    # pure representation axis, ~0 alone, named in no interaction).
    # Together the three flips give the matrix model LoopFormer's
    # training-setup structure (matched init scale, 2-layer shared-depth
    # shape, per-iteration conditioning) while leaving its matrix-ness
    # (outer-product embedding, matrix ops, MultiProbeHead) untouched.
    "combined_hj_hg_hi": dict(embed_init="matched", n_layers=2, iter_cond=True),
    # H_b's GATED rank-swept family (design section 4 item 8 / section 6.1
    # row 6; gate FIRED 2026-07-02 -- "deployed in Wave B only if item 5's
    # depth-matched control doesn't close the gap", which it didn't,
    # recovered_frac -0.050): "a rank-r factored Linear(d^2,r) ->
    # Linear(r,d^2) swept over a few r values as a parametric family
    # bridging Kronecker-restricted and increasingly expressive, rather
    # than a single point comparison"; section 6.1 pins r in {1,4,16}.
    # r=1 is the param-matched point ("matching RowThenColProjection's
    # exact ~2d^2 params ... forces rank <=1"); r=16 the expressiveness
    # direction. See common.FactoredDenseProjection for the disclosed
    # init/param asymmetries. NOT Wave-A screen cells (gated pool).
    "h_b_factored_r1": dict(proj_rank=1),
    "h_b_factored_r4": dict(proj_rank=4),
    "h_b_factored_r16": dict(proj_rank=16),
    # Wave-B capacity-confound control (coordinator direction, 2026-07-02):
    # the r4 winner carries 2.7x baseline params (781,848 vs 290,328) at
    # matched tokens-seen, so its +2.025 recovered_frac confounds
    # "less-restricted projection family" with "more capacity" (exactly
    # section 9 attack #1's disclosed convention risk). This cell is the
    # factored-family point param-matched to BOTH anchors: 289,993 params
    # = 96.3% of the 300,976 vector reference (inside the required +/-5%
    # band) and 99.9% of the 290,328 matrix baseline. DOCUMENTED DEVIATION
    # from the requested "param-matched r=4": the measured sweep shows r=4
    # cannot reach the band by narrowing layer count alone (n_layers
    # 2/3/4 -> 265,350/351,433/437,516); the two in-band family points are
    # (nl=3, r=3)=289,993 and (nl=2, r=5)=306,310 -- (3,3) chosen for
    # depth closer to the 8-layer winner plus the exact baseline match.
    # What was narrowed, exactly: n_layers 8 -> 3 AND proj_rank 4 -> 3;
    # nothing else differs from h_b_factored_r4.
    "h_b_factored_r3_nl3": dict(proj_rank=3, n_layers=3),
}

# The five Wave-A OFAT screen cells (design section 6.1/7): exactly one
# axis flipped from the matrix-all baseline per cell. run_stageg_sweep.py's
# Wave-A manifest is built from THIS list, not from VARIANT_AXES, so
# registering conditional follow-up variants above can never silently
# change the screen's shape.
WAVE_A_SCREEN_AXES = ("h_j_init_matched", "h_a_full_rank", "h_f_tied_bilinear",
                       "h_g_depth_matched", "h_i_iter_cond")


@dataclass
class MatrixModelSpec:
    mat_dim: int = 32
    n_heads: int = 8
    n_iterations: int = 8
    max_len: int = 1024
    vocab_size: int = 256
    dropout: int = 0.1
    variant: str = "baseline"

    def build(self):
        if self.variant not in VARIANT_AXES:
            raise ValueError(f"unknown variant {self.variant!r}; choices: {list(VARIANT_AXES)}")
        kwargs = dict(MATRIX_ALL_BASELINE)
        kwargs.update(VARIANT_AXES[self.variant])
        return MatrixThinkerG(mat_dim=self.mat_dim, n_heads=self.n_heads,
                               n_iterations=self.n_iterations, max_len=self.max_len,
                               vocab_size=self.vocab_size, dropout=self.dropout, **kwargs)


# Vector-all baseline (design section 6.1: LoopFormer's OWN defaults --
# dense tied embedding, AdaLN conditioning -- ARE the "alternative" state
# of H_f/H_i; no intervention needed to reach them). Wave-B mirror axes
# (design section 6.1: "bidirectional vector-all-baseline mirror for the
# winning axis") are exposed as separate named variants, built ONLY for
# the single top-1/2 axis Wave A identifies -- see module docstring NOTE
# on incomplete axis coverage (H_g has no vector-side mirror at all;
# H_a's mirror is a documented judgment call).
VECTOR_ALL_BASELINE = dict(embed_rank="dense", embed_init_std=0.02, tie_head=True, iter_cond=True)

VECTOR_VARIANT_AXES = {
    "baseline": {},
    "mirror_h_f_untied": dict(tie_head=False),
    "mirror_h_i_no_cond": dict(iter_cond=False),
    "mirror_h_j_init1": dict(embed_init_std=1.0),
    "mirror_h_a_rank1": dict(embed_rank="rank1", tie_head=False),  # rank1 has no single dense table to tie
    # Wave-B bidirectional capacity control (coordinator direction,
    # 2026-07-02): the vector reference scaled to the r4 winner's param
    # count -- does the VECTOR side also improve to ~2.94 BPB at 782K
    # params, or does the factored-matrix model hold an advantage at
    # matched capacity? Single documented knob: n_embd 80 -> 152 (768,616
    # params, -1.7% of the 781,848 target, inside +/-5%; n_embd=156 gives
    # 800,068, +2.3% -- 152 is closer). n_layer/intermediate_dim/n_loops
    # unchanged. Handled via the n_embd-override pop in
    # VectorModelSpec.build.
    "capacity_782k": dict(n_embd=152),
}

# Matrix Wave-A variant -> its Wave-B vector-side bidirectional mirror
# (design section 6.1/section 7's "bidirectional vector-all-baseline
# mirror for the winning axis"). None = no natural mirror exists (H_g's
# depth-match axis: LoopFormer's 2-shared-blocks-times-8-loops structure
# IS what H_g asks the matrix side to imitate; there is nothing to flip
# on the vector side, per models.py's module docstring NOTE).
WAVE_B_MIRROR = {
    "h_j_init_matched": "mirror_h_j_init1",
    "h_a_full_rank": "mirror_h_a_rank1",
    "h_f_tied_bilinear": "mirror_h_f_untied",
    "h_g_depth_matched": None,
    "h_i_iter_cond": "mirror_h_i_no_cond",
    "h_f_tied_matched_init": "mirror_h_f_untied",   # N3 follow-up shares H_f's tie/untie mirror axis
    "combined_hj_hg_hi": None,   # multi-axis probe; no single vector-side mirror exists
    "h_b_factored_r1": None,     # H_b family is a matrix-side projection swap; no vector mirror pre-registered
    "h_b_factored_r4": None,     # (Wave-B capacity control uses vector 'capacity_782k' instead, launched directly)
    "h_b_factored_r16": None,
    "h_b_factored_r3_nl3": None,
}


@dataclass
class VectorModelSpec:
    n_embd: int = 80   # MEASURED param match to MatrixThinkerG d=32 (audit MAJOR-2; see VectorReferenceModel docstring)
    n_head: int = 4
    n_layer: int = 2
    n_loops: int = 8
    intermediate_dim: int = 128
    max_len: int = 1024
    vocab_size: int = 256
    dropout: int = 0.1
    variant: str = "baseline"

    def build(self):
        if self.variant not in VECTOR_VARIANT_AXES:
            raise ValueError(f"unknown vector variant {self.variant!r}; choices: {list(VECTOR_VARIANT_AXES)}")
        kwargs = dict(VECTOR_ALL_BASELINE)
        kwargs.update(VECTOR_VARIANT_AXES[self.variant])
        # A variant may pin n_embd (capacity_782k) -- pop it out of the
        # VectorReferenceModel kwargs so it overrides the spec field
        # instead of colliding with the explicit n_embd= argument below.
        n_embd = kwargs.pop("n_embd", self.n_embd)
        return VectorReferenceModel(n_embd=n_embd, n_head=self.n_head, n_layer=self.n_layer,
                                     n_loops=self.n_loops, intermediate_dim=self.intermediate_dim,
                                     max_len=self.max_len, vocab_size=self.vocab_size,
                                     dropout=self.dropout, **kwargs)
