"""Step 1 of the S2.25 analytic adjudication (LOCAL, CPU, fp32).

Closed form at D=1, n_h=1, S_0 = 0:
    S_1 = S_0 (I - beta k k^T) + beta v k^T  =  beta v k^T          (composer/S2.2 convention: rows=value dim, cols=key dim)
The fla 0.5.1 convention (verified from the installed naive.py + chunk.py
docstring on the box: state [N, H, K, V], update S = (I - beta k k^T) S + beta k v^T):
    S_1^fla = beta k v^T = (S_1)^T

This script computes S_1 analytic BY HAND (explicit scalar-index loops --
independent literal, no matmul, no recurrence code) for the EXACT test
inputs fla_cross_check uses (seed 0, same construction order), and compares
the composer's states_from_embedding output against it.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/Users/samuellarson/Experiments/learned-representations/matrix-thinking/capability_separation")
from stage2_composer import GroupWordDeltaComposer, FLA_CROSS_CHECK_CONFIGS

B, h = 4, 32
device = "cpu"

torch.manual_seed(0)
# EXACT reproduction of fla_cross_check's loop-iteration-1 construction order:
# (n_h=1, D=1) is the first config in FLA_CROSS_CHECK_CONFIGS.
n_h, D = FLA_CROSS_CHECK_CONFIGS[0]
assert (n_h, D) == (1, 1), FLA_CROSS_CHECK_CONFIGS
composer = GroupWordDeltaComposer(d_state=5, n_gens=4, h=h, n_h=n_h, beta_max=2.0).to(device)
token_idx = torch.randint(0, 4, (B, D), device=device)

tok_embed = composer.embed_tokens(token_idx)
states = composer.states_from_embedding(tok_embed)
S_torch = states[-1]  # (B, h, h) fp32 -- the composer's S_1

# Manual re-derivation of (k, v, beta), identical to fla_cross_check's own
# (and to states_from_embedding's internals):
tok_embed_w = tok_embed + composer.widen(tok_embed) if composer.widen is not None else tok_embed
k = composer.k_proj(tok_embed_w).view(B, D, n_h, h)
v = composer.v_proj(tok_embed_w).view(B, D, n_h, h)
k = F.normalize(k, dim=-1, eps=1e-8)
beta = composer.beta_max * torch.sigmoid(composer.beta_proj(tok_embed_w).view(B, D, n_h))

# ---- ANALYTIC TRUTH, BY HAND: S_1[b][i][j] = beta[b] * v[b][i] * k[b][j] ----
# explicit scalar loops; no tensor ops beyond indexing.
S_analytic = torch.zeros(B, h, h)
with torch.no_grad():
    kb = k[:, 0, 0, :]      # (B, h)
    vb = v[:, 0, 0, :]      # (B, h)
    bb = beta[:, 0, 0]      # (B,)
    for b in range(B):
        for i in range(h):
            for j in range(h):
                S_analytic[b, i, j] = bb[b].item() * vb[b, i].item() * kb[b, j].item()

def relfro(A, Bm):
    return ((A - Bm).norm() / Bm.norm().clamp(min=1e-12)).item()

with torch.no_grad():
    r_comp = relfro(S_torch, S_analytic)
    r_comp_T = relfro(S_torch, S_analytic.transpose(-1, -2))
    print(f"composer  vs analytic (v k^T, S2.2 convention):        rel-Fro = {r_comp:.3e}")
    print(f"composer  vs analytic^T (k v^T, fla convention):       rel-Fro = {r_comp_T:.3e}")
    print(f"beta range: [{bb.min().item():.4f}, {bb.max().item():.4f}]  (beta_max=2.0 arm)")
    print(f"|k^T v| per batch: {[abs((kb[b] @ vb[b]).item()) for b in range(B)]}")
    print(f"||k|| per batch:   {[kb[b].norm().item() for b in range(B)]}")

assert r_comp < 1e-5, f"COMPOSER DEVIATES from the pinned closed form: {r_comp}"
print("\nVERDICT (Step 1): the composer MATCHES the pinned S2.2 closed form S_1 = beta v k^T on CPU fp32.")
print(f"Expected fla-vs-composer rel-Fro if fla returns the transpose: {r_comp_T:.4f} (compare to the observed 1.4008)")
