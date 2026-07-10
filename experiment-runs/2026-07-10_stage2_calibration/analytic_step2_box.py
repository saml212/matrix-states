"""Step 2 of the S2.25 analytic adjudication (BOX, GPU, eval-only, seconds).

Same seed-0 (n_h=1, D=1) inputs as fla_cross_check's first config, built on
CPU fp32 (identical to Step 1 local). Analytic truth by explicit scalar
loops. Then the REAL fla 0.5.1 chunk_delta_rule (correct kwargs:
output_final_state=True, NO nonexistent allow_neg_eigval) on cuda:0,
compared against the analytic truth in BOTH conventions, plus a
use_qk_l2norm_in_kernel True-vs-False probe (q=0 zero-vector hazard check),
plus the D=8 configs to confirm the transpose explains ALL three failures.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nvidia/chapter2/capability_separation")
from stage2_composer import GroupWordDeltaComposer, FLA_CROSS_CHECK_CONFIGS

from fla.ops.delta_rule import chunk_delta_rule

B, h = 4, 32


def relfro(A, Bm):
    return ((A - Bm).norm() / Bm.norm().clamp(min=1e-12)).item()


torch.manual_seed(0)
for cfg_i, (n_h, D) in enumerate(FLA_CROSS_CHECK_CONFIGS):
    # cpu construction => identical weights/inputs to the local Step-1 run for cfg 0
    composer = GroupWordDeltaComposer(d_state=5, n_gens=4, h=h, n_h=n_h, beta_max=2.0)
    token_idx = torch.randint(0, 4, (B, D))
    with torch.no_grad():
        tok_embed = composer.embed_tokens(token_idx)
        S_torch = composer.states_from_embedding(tok_embed)[-1]        # (B,h,h) fp32 CPU
        tok_w = tok_embed + composer.widen(tok_embed) if composer.widen is not None else tok_embed
        k = F.normalize(composer.k_proj(tok_w).view(B, D, n_h, h), dim=-1, eps=1e-8)
        v = composer.v_proj(tok_w).view(B, D, n_h, h)
        beta = composer.beta_max * torch.sigmoid(composer.beta_proj(tok_w).view(B, D, n_h))

    if (n_h, D) == (1, 1):
        # ---- ANALYTIC TRUTH BY HAND (scalar loops): S1[b,i,j] = beta[b]*v[b,i]*k[b,j]
        S_an = torch.zeros(B, h, h)
        kb, vb, bb = k[:, 0, 0, :], v[:, 0, 0, :], beta[:, 0, 0]
        for b in range(B):
            for i in range(h):
                for j in range(h):
                    S_an[b, i, j] = bb[b].item() * vb[b, i].item() * kb[b, j].item()
        print(f"[cfg (1,1)] composer vs analytic vk^T:  {relfro(S_torch, S_an):.3e}  (expect ~1e-7)")

    # ---- REAL fla, correct kwargs ----
    dev = "cuda:0"
    q_exp = torch.zeros(B, D * n_h, 1, h).to(dev, torch.bfloat16)
    k_exp = k.reshape(B, D * n_h, 1, h).to(dev, torch.bfloat16)
    v_exp = v.reshape(B, D * n_h, 1, h).to(dev, torch.bfloat16)
    beta_exp = beta.reshape(B, D * n_h, 1).to(dev, torch.bfloat16)

    for l2flag in (False, True):
        _o, fs = chunk_delta_rule(q_exp, k_exp, v_exp, beta_exp,
                                  output_final_state=True,
                                  use_qk_l2norm_in_kernel=l2flag)
        assert fs is not None, "output_final_state=True must return a state"
        S_fla = fs.squeeze(1).to(torch.float32).cpu()                 # (B,h,h) [K,V] layout
        r_raw = relfro(S_fla, S_torch)
        r_T = relfro(S_fla.transpose(-1, -2), S_torch)
        o_finite = torch.isfinite(_o.float()).all().item()
        line = (f"[cfg ({n_h},{D})] l2norm_in_kernel={l2flag}: "
                f"fla-vs-composer RAW={r_raw:.4e}  TRANSPOSED={r_T:.4e}  o finite={o_finite}")
        if (n_h, D) == (1, 1):
            line += (f"  | fla vs analytic vk^T={relfro(S_fla, S_an):.4e}"
                     f"  fla vs analytic (vk^T)^T=kv^T={relfro(S_fla, S_an.transpose(-1, -2)):.4e}")
        print(line)

print("\nbeta stats across configs were in (0,2) (beta_max=2.0 arm); no in-op sigmoid/clamp per installed source.")
