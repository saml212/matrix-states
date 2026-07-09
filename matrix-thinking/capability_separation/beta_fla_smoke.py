"""CAPABILITY_SEPARATION_DESIGN.md S1.6 -- the beta in [0,2] `fla`
positive-control smoke, REGISTERED for the box, non-load-bearing for
Stage 1's CONFIRM/FALSIFY/INCONCLUSIVE verdict (Stage 1 uses the bespoke
`GroupWordEncoder` throughout, never this layer). Purpose: validate the
`fla` `allow_neg_eigval`-style negative-eigenvalue state-tracking code path
on THIS repo's stack, on a minimal DeltaProduct-style layer (Siems et al.
2502.10297; Grazzi et al. arXiv:2411.12537's `allow_neg_eigval` mechanism),
BEFORE Stage 2 or any later wave wants a production delta-rule kernel
instead of the bespoke encoder.

Two pieces, per S1.6's cost-table itemization:
  (1) forward/backward/grad-check (~0.05 GPU-h nominal on the box; runs
      instantly, CPU-safe via a stub when real `fla` is unavailable --
      exercises WIRING, not the real Triton kernel's numerics).
  (2) one reproduced DeltaProduct Fig.5 qualitative-split point on a
      minimal S4 (or A5) word-problem instance (L<=4, n_h in {1,2}):
      n_h=1 should FAIL state tracking on this group family, n_h=2 should
      SUCCEED (matches STATE.md's verified citation: DeltaProduct's own
      published n_h numbers, S4 n_h=2). This piece needs a REAL GPU + real
      `fla` (the CPU stub's simplified recurrence has no genuine negative-
      eigenvalue dynamics to reproduce a real qualitative split with) --
      CPU-stub-SKIPPED locally with a loud message, mirroring
      deltanet_rd/h2h_fla_stub_rd.py's own disclosed-stub-limitation
      convention (box-only-verifiable items depending on a NONZERO, trained
      state cannot be given meaningful numbers by a CPU stub).

Self-contained (no cross-directory import of deltanet_rd's stub -- this
repo's own rank_utils.py precedent: "Task D scripts are self-contained for
pod deployment, no cross-directory imports"), but mirrors the SAME
CPU-stub-installer pattern deltanet_rd/h2h_fla_stub_rd.py already uses.

Run: DRY_RUN_BYPASS=1 .venv/bin/python beta_fla_smoke.py [--reproduce-fig5]
"""
from __future__ import annotations

import argparse
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_STUB_MARKER = "_CAPABILITY_SEP_CPU_STUB"


def ensure_fla_stub() -> bool:
    """Installs a CPU-safe stand-in for `fla.ops.delta_rule.chunk_delta_rule`
    (with an `allow_neg_eigval` kwarg accepted, matching the real API's
    surface) UNLESS real `fla` is importable and
    `CAPABILITY_SEP_FORCE_CPU_STUB` is not set to '1'. Returns True iff a
    stub is in effect (this call OR an earlier call in this process
    installed it), False iff real fla is in use."""
    if os.environ.get("CAPABILITY_SEP_FORCE_CPU_STUB", "0") != "1":
        try:
            import fla  # noqa: F401
            return bool(getattr(fla, _STUB_MARKER, False))
        except ImportError:
            pass

    def _stub_chunk_delta_rule(q, k, v, beta, initial_state=None, output_final_state=True,
                               use_qk_l2norm_in_kernel=True, allow_neg_eigval=False):
        """CPU-safe stand-in for the CUDA-only Triton kernel. `allow_neg_eigval`
        is ACCEPTED (matches the real API surface this smoke validates
        wiring against) but the stub's recurrence has no genuine
        negative-eigenvalue dynamics -- disclosed, not silently papered
        over (see module docstring: box-only-verifiable numerics)."""
        B, T, H, Dh = q.shape
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        beta_eff = beta * (2.0 if allow_neg_eigval else 1.0)   # crude signal that the flag was honored
        qk_gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        o = v * torch.sigmoid(beta_eff).unsqueeze(-1) * qk_gate
        final_state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype, device=q.device)
        return o, final_state

    fla_mod = types.ModuleType("fla")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_delta_rule = types.ModuleType("fla.ops.delta_rule")
    fla_ops_delta_rule.chunk_delta_rule = _stub_chunk_delta_rule
    fla_mod.ops = fla_ops
    fla_ops.delta_rule = fla_ops_delta_rule
    setattr(fla_mod, _STUB_MARKER, True)
    sys.modules["fla"] = fla_mod
    sys.modules["fla.ops"] = fla_ops
    sys.modules["fla.ops.delta_rule"] = fla_ops_delta_rule
    return True


class DeltaProductLayer(nn.Module):
    """Minimal DeltaProduct-style layer (Siems et al. 2502.10297): each
    macro-token contributes `n_h` micro-steps (Householder-product
    expansion) to the underlying delta-rule recurrence, using a beta-GATE
    range [0,2] (`beta = 2*sigmoid(raw)`, NOT this repo's baseline plain
    sigmoid beta in [0,1]) and `allow_neg_eigval=True` (Grazzi et al.
    2411.12537's negative-eigenvalue state-tracking mechanism) -- the
    combination DeltaProduct's Fig.5 shows is necessary for state tracking
    on non-abelian groups at n_h>=2 Householder products."""

    def __init__(self, d_model: int, n_heads: int, n_h: int = 2, allow_neg_eigval: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_h = n_h
        self.allow_neg_eigval = allow_neg_eigval
        # n_h independent (q,k,v,beta) projections -- one Householder-style
        # micro-step per macro-token per h.
        self.q_proj = nn.Linear(d_model, d_model * n_h, bias=False)
        self.k_proj = nn.Linear(d_model, d_model * n_h, bias=False)
        self.v_proj = nn.Linear(d_model, d_model * n_h, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads * n_h)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from fla.ops.delta_rule import chunk_delta_rule
        B, T, D = x.shape
        H, Dh, n_h = self.n_heads, self.d_head, self.n_h

        def _split_heads(t):
            return t.view(B, T, n_h, H, Dh)

        q = _split_heads(self.q_proj(x))    # (B, T, n_h, H, Dh)
        k = _split_heads(self.k_proj(x))
        v = _split_heads(self.v_proj(x))
        beta_raw = self.beta_proj(x).view(B, T, n_h, H)
        beta = 2.0 * torch.sigmoid(beta_raw)   # DeltaProduct's beta-gate range [0,2]

        # Expand each macro-token into n_h consecutive micro-steps: (B, T*n_h, H, Dh).
        q_exp = q.permute(0, 1, 2, 3, 4).reshape(B, T * n_h, H, Dh)
        k_exp = k.reshape(B, T * n_h, H, Dh)
        v_exp = v.reshape(B, T * n_h, H, Dh)
        beta_exp = beta.reshape(B, T * n_h, H)

        o, _final_state = chunk_delta_rule(q_exp, k_exp, v_exp, beta_exp,
                                           allow_neg_eigval=self.allow_neg_eigval)
        # collapse back to (B, T, ...): the LAST micro-step of each macro-token
        # is that token's output (standard DeltaProduct convention).
        o = o.view(B, T, n_h, H, Dh)[:, :, -1, :, :].reshape(B, T, D)
        return self.out_proj(o)


# ---------------------------------------------------------------------------
# Piece (1): forward/backward/grad-check smoke (CPU-safe via stub).
# ---------------------------------------------------------------------------

def smoke_forward_backward(device="cpu"):
    print("=" * 88)
    print("  beta_fla_smoke.py PIECE (1) -- forward/backward/grad-check "
          "(allow_neg_eigval, beta in [0,2], n_h in {1,2})")
    print("=" * 88)
    is_stub = ensure_fla_stub()
    print(f"  fla backend: {'CPU STUB (real fla unavailable or CAPABILITY_SEP_FORCE_CPU_STUB=1)' if is_stub else 'REAL fla'}")

    torch.manual_seed(0)
    d_model, n_heads = 32, 4
    for n_h in (1, 2):
        layer = DeltaProductLayer(d_model, n_heads, n_h=n_h, allow_neg_eigval=True).to(device)
        x = torch.randn(4, 6, d_model, device=device, requires_grad=True)
        out = layer(x)
        assert out.shape == x.shape, (out.shape, x.shape)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()
        loss = out.pow(2).mean()
        loss.backward()
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"n_h={n_h}: no grad for {name}"
            assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), \
                f"n_h={n_h}: bad grad for {name}"
        assert x.grad is not None and not torch.isnan(x.grad).any()
        print(f"  n_h={n_h}: forward {tuple(out.shape)}, loss={loss.item():.4f}, "
              f"all params + input finite grad  OK")

    print("\n" + "=" * 88)
    print("  PIECE (1) PASSED (wiring verified; stub numerics are NOT the real kernel's --")
    print("  box-only-verifiable, per module docstring)")
    print("=" * 88)
    return is_stub


# ---------------------------------------------------------------------------
# Piece (2): DeltaProduct Fig.5 qualitative-split reproduction (BOX-ONLY).
# ---------------------------------------------------------------------------

def reproduce_fig5(device="cpu", is_stub=True):
    print("\n" + "=" * 88)
    print("  beta_fla_smoke.py PIECE (2) -- DeltaProduct Fig.5 reproduction "
          "(minimal S4 instance, L<=4, n_h in {1,2})")
    print("=" * 88)
    if is_stub:
        print("  *** SKIPPED (CPU-stub backend) ***")
        print("  This piece requires a REAL fla `chunk_delta_rule` kernel and GPU training")
        print("  (~0.85 GPU-h, S1.6's cost table) to produce a genuine qualitative split --")
        print("  the CPU stub's recurrence has no real negative-eigenvalue dynamics, so any")
        print("  'result' computed against it would NOT be evidence about DeltaProduct's real")
        print("  behavior. REGISTERED FOR THE BOX: re-run this function with real fla installed")
        print("  on the H100 cluster (unset CAPABILITY_SEP_FORCE_CPU_STUB, ensure `import fla`")
        print("  succeeds) before Stage 1 launch, per S1.6's cost table. Non-load-bearing for")
        print("  Stage 1's CONFIRM/FALSIFY/INCONCLUSIVE verdict either way.")
        print("=" * 88)
        return {"status": "skipped_cpu_stub", "box_only": True}

    # Real-fla path (registered, runs on the box): train n_h in {1,2} DeltaProduct
    # layers on a minimal S4 word-problem instance and compare recovery.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from groups import D_STATE
    from group_task import generating_set
    from group_word_encoder import cosine_loss

    torch.manual_seed(0)
    d_model, n_heads = 32, 4
    L_max = 4
    n_gens = len(generating_set("S4"))
    d_state = D_STATE["S4"]
    results = {}
    for n_h in (1, 2):
        torch.manual_seed(0)
        embed = nn.Embedding(n_gens, d_model).to(device)
        layer = DeltaProductLayer(d_model, n_heads, n_h=n_h, allow_neg_eigval=True).to(device)
        readout = nn.Linear(d_model, d_state * d_state).to(device)
        params = list(embed.parameters()) + list(layer.parameters()) + list(readout.parameters())
        opt = torch.optim.Adam(params, lr=3e-4)
        from group_task import sample_train_batch
        gen = torch.Generator().manual_seed(n_h)
        final_loss = None
        for step in range(200):
            batch = sample_train_batch("S4", 64, gen, device=device)
            tok = embed(batch["token_idx"])
            enc = layer(tok)
            Z = readout(enc.mean(dim=1)).view(-1, d_state, d_state)
            loss = cosine_loss(Z, batch["target"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = loss.item()
        results[n_h] = final_loss
        print(f"  n_h={n_h}: final cosine_loss (200 steps, minimal instance) = {final_loss:.4f}")

    print(f"\n  qualitative split: n_h=1 loss={results[1]:.4f}  n_h=2 loss={results[2]:.4f}  "
          f"n_h=2 lower (expected DeltaProduct split)? {results[2] < results[1]}")
    print("=" * 88)
    return {"status": "ran_real_fla", "box_only": False, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce-fig5", action="store_true",
                    help="also attempt piece (2) (box-only unless real fla + GPU available)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_stub = smoke_forward_backward(device=device)
    if args.reproduce_fig5:
        reproduce_fig5(device=device, is_stub=is_stub)
    else:
        print("\n(pass --reproduce-fig5 to also attempt piece (2); box-only unless real fla+GPU)")


if __name__ == "__main__":
    main()
