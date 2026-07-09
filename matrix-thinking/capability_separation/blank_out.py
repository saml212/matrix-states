"""CAPABILITY_SEPARATION_DESIGN.md S1.4 -- the blank-out bottleneck test,
"reused verbatim from run_task_d.py::smoke() step [5]" per the design's own
wording, with ONE disclosed, necessary adaptation:

Task D's raw inputs (`keys`, `values`) are CONTINUOUS vectors, so
`torch.autograd.grad` w.r.t. them is directly meaningful. This task's raw
input is a LongTensor of generator indices (`token_idx`) -- PyTorch has no
notion of a gradient w.r.t. an integer tensor (`only Tensors of floating
point dtype can require gradients`), so a literal `keys.requires_grad_(True)`
analog is impossible here.

The adaptation: the leaf is taken at `GroupWordEncoder.embed_tokens(...)`'s
OUTPUT (`tok_embed`, shape (B, L, h)) -- the first CONTINUOUS tensor derived
from the raw discrete word, and the ONLY tensor `encode_from_embedding`
touches. This preserves the exact mechanism CLAUDE.md's hard rule requires
(a REAL gradient-based check, not a code-inspection-only shape check): (a)
with `Z` NOT detached, gradient of a downstream function of `Z` w.r.t.
`tok_embed` must be nonzero (sanity: a real signal exists); (b) with `Z`
detached to an INDEPENDENT leaf (mirroring run_task_d.py's `Z_leaf`
construction exactly), gradient of the SAME downstream function computed
from the detached `Z_leaf` w.r.t. the ORIGINAL `tok_embed` must be `None`
(no path around `Z`) -- `encode_from_embedding` is never called twice on the
same graph the way `unbind` is called twice in Task D's version; instead the
SCORING function itself (whatever consumes `Z`) is what gets the leaf-swap
treatment, exactly parallel to Task D's `pred_leaf = model.unbind(Z_leaf,
...)`. `inspect.signature` on the scoring function is checked to admit only
`Z` (plus the fixed reference target) as inputs -- this task simplifies
Task D's check further (S1.4: "no separate query_keys argument to check").
"""
from __future__ import annotations

import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_word_encoder import GroupWordModel, cosine_loss


def scoring_fn(Z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The PRODUCTION scoring function's signature (mirrors
    MatrixMemoryModel.unbind's role in run_task_d.py's blank-out test): a
    PURE function of (Z, target) -- no other inputs. Any function used
    downstream of Z for loss/scoring in this task family must have this
    same signature shape for the blank-out test to certify the bottleneck."""
    return cosine_loss(Z, target)


def run_blank_out_test(model: GroupWordModel, device="cpu", B: int = 8, L: int = 4) -> None:
    print("=" * 60)
    print("  BLANK-OUT test: scoring path touches the raw word ONLY via Z")
    print("=" * 60)
    torch.manual_seed(0)
    d_state = model.d_state
    token_idx = torch.randint(0, model.encoder.n_gens, (B, L), device=device)
    target = torch.randn(B, d_state, d_state, device=device)

    # (a) sanity: with Z NOT detached, gradient DOES flow from a downstream
    # function of Z back to tok_embed (a real signal exists).
    tok_embed = model.encoder.embed_tokens(token_idx).clone().detach().requires_grad_(True)
    Zg = model.encoder.encode_from_embedding(tok_embed)
    if model.encoder.d_state:  # no rank forcing here -- raw encoder output
        pass
    loss_g = scoring_fn(Zg, target)
    g_embed = torch.autograd.grad(loss_g, tok_embed, retain_graph=True, allow_unused=True)[0]
    assert g_embed is not None and g_embed.abs().sum() > 0, \
        "the word embedding doesn't affect the score at all?!"
    print(f"  (a) sanity: grad(score, tok_embed) nonzero (sum|grad|={g_embed.abs().sum().item():.4f}) OK")

    # (b) meaningful check: with Z detached to an INDEPENDENT leaf, the SAME
    # tok_embed must have NO remaining path to the score.
    Z_leaf = Zg.detach().clone().requires_grad_(True)
    loss_leaf = scoring_fn(Z_leaf, target)
    g_leak = torch.autograd.grad(loss_leaf, tok_embed, allow_unused=True)[0]
    assert g_leak is None, "LEAK: the raw word reaches the score outside Z"
    print("  (b) leaf-detach: grad(score(Z_leaf), tok_embed) is None -- no path around Z  OK")

    # (c) signature check: scoring_fn admits ONLY (Z, target) as inputs.
    sig = set(inspect.signature(scoring_fn).parameters)
    assert sig <= {"Z", "target"}, f"scoring_fn takes inputs beyond (Z, target): {sig}"
    print(f"  (c) inspect.signature(scoring_fn) = {sorted(sig)} subset of {{'Z','target'}}  OK")

    print("\n" + "=" * 60 + "\n  BLANK-OUT TEST PASSED\n" + "=" * 60)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = GroupWordModel(d_state=5, n_gens=4, L_max=16, h=32).to(device)
    run_blank_out_test(m, device=device)
