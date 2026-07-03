"""H_f: genuinely-tied output heads — STAGE_G_DESIGN.md section 4, item 4
(F2's blocking pre-Wave-A build task).

`matrix-thinking/src/matrix_output_heads.py::TiedMultiProbeHead` is
explicitly NOT reused: its `forward()` derives probes from fresh
`probe_mix_u/v` Linear layers and never reads the `embed_u`/`embed_v`
tensors it stores, and its `out` projection is a full untied
`Linear(K, vocab)` — it is dead code that does not tie (F2). Both forms
below are new, and both are covered by the mandatory tie-verification unit
test in test_stageg.py (design section 12, build requirement (v)): in-place
perturbation of the embedding weights must change logits, tensor `is`
identity must hold between the head's embedding references and the model's
owned embedding parameters, and neither form may own a vocab-sized
parameter.

Two forms, per design section 4 item 4:

  (i) Regime 2 (byte vocab) — exact bilinear tied logits with a learned
      scalar temperature (N3): logit(w) = tau * (u_w^T M v_w). The
      temperature exists because at default std-1.0 embedding init the raw
      bilinear form has scale ~O(d) (sum of d^2 std-1 x std-1 terms through
      a unit-RMS-normalized M has variance ~d^2 => std~d), which saturates
      softmax at step 0 -- an H_f x H_j confound baked into the swap itself
      if left untreated. `calibrate_temperature` sets tau empirically from
      a real forward pass (instrumented, not hand-derived, per CLAUDE.md's
      "no by-hand number is load-bearing" discipline) so step-0 logit std
      is ~1 regardless of the exact embedding-init cell it's run under.

  (ii) Regime 1 (BPE vocab) — factored tied head:
      logits = (probes @ W_mix) @ [E_u, E_v]^T
      where [E_u, E_v] in R^{vocab x 2d} is the FEATURE-CONCATENATION (N4:
      not a row-stack) of the embedding tables themselves (E_u =
      embed_u.weight, E_v = embed_v.weight -- the SAME tensors, not
      copies), and W_mix in R^{K x 2d} is the new mixing weight.
      `probes` reuses the same K learned bilinear-probe-direction machinery
      (U, V in R^{K x d}) already established by MultiProbeHead in this
      codebase -- these are NOT vocab-sized (2*K*d params, tiny at K=d=32:
      2,048 params) and are explicitly not "the vocab-sized untied
      parameter" the tie-verification test checks for; the untied,
      vocab-sized piece the design calls out as removed is MultiProbeHead's
      `out: Linear(K, vocab)` (K*vocab params, e.g. 1.6M at K=32,
      vocab=50257), which this head does not have -- its only vocab-facing
      weight is the tied, shared [E_u, E_v]^T. This reading of the design's
      "W_mix is the only new parameter" (i.e. the only new parameter beyond
      the pre-existing K-probe-extraction convention) is a documented
      INTERPRETATION -- flagged for the auditor, see the report.

References: Press & Wolf 2017 (arXiv:1608.05859); Inan et al. 2016
(arXiv:1611.01462) -- exact citations not re-verified here (design section
4 item 4's own caveat: "verify exact citations before external use").
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TiedBilinearHead(nn.Module):
    """Form (i) -- Regime 2 (byte vocab), exact bilinear tie with learned
    temperature (N3). No copies of the embedding tables are made; `embed_u`
    and `embed_v` are stored as direct module references so
    `head.embed_u is model.embed_u` holds by construction (mandatory tie
    test)."""

    def __init__(self, embed_u: nn.Embedding, embed_v: nn.Embedding, tau_init: float | None = None):
        super().__init__()
        assert embed_u.embedding_dim == embed_v.embedding_dim
        assert embed_u.num_embeddings == embed_v.num_embeddings
        self.embed_u = embed_u  # shared reference, NOT copied -- owns no vocab-sized param here
        self.embed_v = embed_v
        d = embed_u.embedding_dim
        # Analytic fallback init (design section 4 item 4: "raw bilinear
        # form has scale ~O(d)" at default std-1 embedding init) --
        # `calibrate_temperature` below overrides this with an empirical
        # measurement once the full model exists (mandatory per N3).
        #
        # Reparametrized as log_tau (audit MAJOR-1): tau's scalar gradient
        # is a sum over B*L*vocab correlated terms and was measured at
        # 99.6% of the model's total grad-norm (|grad(tau)|=81.03 of
        # 81.26), collapsing the SHARED clip_grad_norm_ coefficient to
        # ~0.012 and shrinking backbone updates to ~7.4% of baseline at
        # identical nominal clip. log_tau (i) keeps tau positive by
        # construction and (ii) is excluded from the shared clip group by
        # train_stageg.py's `split_tau_param_group` (own param group, 10x
        # lower LR, clipped separately). Exposed as a read-only `tau`
        # property so calibration/logging call sites are unchanged.
        init = tau_init if tau_init is not None else 1.0 / math.sqrt(d)
        self.log_tau = nn.Parameter(torch.tensor(float(init)).log())

    @property
    def tau(self):
        return self.log_tau.exp()

    def forward(self, M):
        # logit(w) = tau * (u_w^T M v_w) for every vocab row w, every token.
        # M: (B,L,d,d); embed weights: (vocab,d).
        Eu = self.embed_u.weight   # (vocab, d)
        Ev = self.embed_v.weight   # (vocab, d)
        MV = torch.einsum('blij,wj->blwi', M, Ev)          # (B,L,vocab,d)
        logits = torch.einsum('wi,blwi->blw', Eu, MV)      # (B,L,vocab)
        return self.tau * logits

    @torch.no_grad()
    def calibrate_temperature(self, M_sample: torch.Tensor):
        """N3: set tau so step-0 logit std ~= 1, measured empirically from a
        real (B,L,d,d) sample of the backbone's step-0 output -- not hand-
        derived. Call once, right after model construction, before training.

        Caller must put the model in eval() mode first (audit MAJOR-1
        rider: dropout active during calibration added ~4% std variance) --
        train_stageg.py's calibrate_tied_bilinear_if_needed does the
        eval()/train() bracketing."""
        Eu = self.embed_u.weight.detach()
        Ev = self.embed_v.weight.detach()
        MV = torch.einsum('blij,wj->blwi', M_sample, Ev)
        raw = torch.einsum('wi,blwi->blw', Eu, MV)
        std = raw.float().std().clamp_min(1e-8)
        self.log_tau.data = -std.log().to(self.log_tau.dtype)   # tau = 1/std
        return std.item()


class TiedFactoredHead(nn.Module):
    """Form (ii) -- Regime 1 (BPE vocab), factored tied head.

    logits = (probes @ W_mix) @ [E_u, E_v]^T

    `probes` (B,L,K) comes from the SAME bilinear-probe-extraction pattern
    MultiProbeHead already uses in this codebase (U,V in R^{K,d}, tiny,
    not vocab-sized). [E_u, E_v] in R^{vocab, 2d} is the feature-
    concatenation of the two embedding tables (N4) -- built via
    torch.cat, NOT copied weight data (see the tie test: perturbing
    embed_u.weight in-place must change logits through this path)."""

    def __init__(self, embed_u: nn.Embedding, embed_v: nn.Embedding, n_probes: int | None = None):
        super().__init__()
        assert embed_u.embedding_dim == embed_v.embedding_dim
        d = embed_u.embedding_dim
        K = n_probes or d
        self.embed_u = embed_u
        self.embed_v = embed_v
        self.U = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.V = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.W_mix = nn.Parameter(torch.randn(K, 2 * d) * (1.0 / math.sqrt(K)))

    def forward(self, M):
        # probes: identical construction to MultiProbeHead.forward
        MV = torch.einsum('blij,kj->blik', M, self.V)          # (B,L,d,K)
        probes = torch.einsum('ki,blik->blk', self.U, MV)      # (B,L,K)
        mixed = probes @ self.W_mix                              # (B,L,2d)
        # [E_u, E_v] feature-concat (N4) built fresh each call from the
        # LIVE weight tensors -- not cached/copied, so grad + perturbation
        # flow through the real embedding parameters.
        E = torch.cat([self.embed_u.weight, self.embed_v.weight], dim=-1)  # (vocab, 2d)
        return mixed @ E.t()                                     # (B,L,vocab)
