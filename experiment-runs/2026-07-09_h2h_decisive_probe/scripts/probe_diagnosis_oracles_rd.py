"""probe_diagnosis_oracles_rd.py -- follow-up oracles for the sec 1.3.1 probe
diagnosis (READ-ONLY; writes results/h2h_rung1/probe_diagnosis/oracles.json).

Tests the EPISODE-MEMBERSHIP hypothesis quantitatively:
  the plateau ~0.18 == cos(mean of the episode's 32 T_val rows, T_val[answer])
  ~= 1/sqrt(32) = 0.1768 -- i.e. the probe learned WHICH entities are in the
  episode, not WHICH one answers the query.

Per arm:
  o1. membership oracle (T_val space): cos(episode-mean T_val row, target).
  o2. cos(ridge-probe pred, episode-mean T_val direction) -- is the probe's
      output essentially the membership direction?
  o3. embed-space CONSTANT floor: cos(fit-set mean answer embedding, target
      embedding) -- adjudicates how much of the tied-embed fit is the generic
      name-direction artifact.
  o4. embed-space MEMBERSHIP oracle: cos(episode-mean embedding, answer emb).
  o5. pairwise-answer embed cos (how non-orthogonal the embed "codebook" is)
      vs pairwise T_val cos (~0 by construction).
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_cell_train_rd import (load_h2h_checkpoint, get_pools, task_cfg,     # noqa: E402
                               make_eval_episodes, GRAMMAR_BATCH)
from grammar_rd import sample_batch_rd                                        # noqa: E402
from probe_diagnosis_rd import (extract_taps, ridge_fit, ridge_pred,          # noqa: E402
                                DIAG_FIT_SEED, N_FIT_BATCHES)

DEVICE = "cuda:0"
OUT = "results/h2h_rung1/probe_diagnosis/oracles.json"


def episode_mean_rows(table, ep_entities):
    """mean over the episode's K entity rows of `table` -- (N, dim)."""
    return table[ep_entities].mean(dim=1)


def main():
    out = {}
    pools = get_pools(DEVICE)
    cfg_eval = task_cfg("task1_calib", 32, n_query=None)
    hop_set = tuple(cfg_eval.H_train)
    for arch in ("contender", "ablation", "transformer"):
        ck = f"/data/h2h_rung1_ckpts/h2h_calib_{arch}_task1_calib_primary_K32_auxrev2.pt"
        model, rig, doc = load_h2h_checkpoint(ck, DEVICE)
        eval_batches = make_eval_episodes(cfg_eval, pools, DEVICE, hop_set)
        gen = torch.Generator(device=DEVICE); gen.manual_seed(DIAG_FIT_SEED)
        fit_batches = [sample_batch_rd(cfg_eval, GRAMMAR_BATCH, gen, hop_set, pools, device=DEVICE)
                       for _ in range(N_FIT_BATCHES)]
        fit = extract_taps(arch, model, fit_batches)
        ev = extract_taps(arch, model, eval_batches)
        Xf, Xe = fit["tap"].to(DEVICE), ev["tap"].to(DEVICE)
        af, ae = fit["answer_id"].to(DEVICE), ev["answer_id"].to(DEVICE)
        epf, epe = fit["ep_entities"].to(DEVICE), ev["ep_entities"].to(DEVICE)
        T_val = rig.T_val
        E = model.embed.weight.detach()

        r = {}
        # o1: membership oracle in T_val space
        mem = episode_mean_rows(T_val, epe)
        r["o1_membership_oracle_Tval_cos"] = F.cosine_similarity(mem, T_val[ae], dim=-1).mean().item()
        # o2: ridge probe pred vs membership direction
        W = ridge_fit(Xf, T_val[af], lambdas=(100.0,))[100.0]
        pred = ridge_pred(W, Xe)
        r["o2_pred_vs_membership_cos"] = F.cosine_similarity(pred, mem, dim=-1).mean().item()
        r["o2_pred_vs_target_cos"] = F.cosine_similarity(pred, T_val[ae], dim=-1).mean().item()
        # o3: embed-space constant floor (fit-set mean answer embedding)
        c = E[af].mean(0, keepdim=True)
        r["o3_embed_const_floor_cos"] = F.cosine_similarity(c.expand(ae.shape[0], -1), E[ae], dim=-1).mean().item()
        # o4: embed-space membership oracle
        mem_e = episode_mean_rows(E, epe)
        r["o4_embed_membership_oracle_cos"] = F.cosine_similarity(mem_e, E[ae], dim=-1).mean().item()
        # o5: pairwise geometry of the two "codebooks" over the train pool
        pool = pools.train_name_ids.to(DEVICE)
        for name, tab in (("Tval", T_val), ("embed", E)):
            rows = F.normalize(tab[pool].float(), dim=-1)
            G = rows @ rows.T
            off = G[~torch.eye(G.shape[0], dtype=torch.bool, device=DEVICE)]
            r[f"o5_pool_pairwise_cos_{name}_mean"] = off.mean().item()
            r[f"o5_pool_pairwise_cos_{name}_std"] = off.std().item()
        out[arch] = r
        print(arch, json.dumps(r, indent=1), flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("theoretical 1/sqrt(32) =", 1 / 32 ** 0.5)


if __name__ == "__main__":
    main()
