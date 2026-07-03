"""geo3_drift_diagnostic.py -- DELTANET_RD_EXACTNESS_DESIGN.md sec 14.6's
Wave -1 GATING DIAGNOSTIC (build item 4, F-geo-3/Rev B). Measures per-entity
CROSS-EPISODE orthogonalized-key drift, feeds it through
geo3_simulator.launch_read (already committed -- imported, NOT
reimplemented), and writes the launch-read verdict to JSON. This is the
diagnostic sec 14.12's sequencing gates Wave 1 on: "Wave 1 mandatory cells
ONLY IF the launch read passes."

Measurement (pinned, sec 14.6, "the measurement (statistic + sampling spec,
pinned)"): >=8 entities per K, drawn randomly WITHOUT replacement from the
TRAIN name pool; >=32 episode-context resamples per entity; per entity,
orthogonalize each resampled episode's own K-cycle and collect that
entity's OWN output row; aggregation = pairwise cosines pooled WITHIN
entity across resamples, then pooled ACROSS entities; report mean and p10
of the pooled distribution. Run per K in {16,32} (48 optional stretch), AT
INIT and AFTER a 5,000-step probe train (sec 14.5: "does training shrink
the drift" is exactly what a probe-trained measurement answers).

The launch read (sec 14.6): keys at the post-orthogonalization guaranteed
Gram residual (resid_tol=1e-2), values tilted to the MEASURED mean drift
(from the TRAINED checkpoint, per sec 14.5's pinned drift-band definition
-- "measured on the sec 14.6 F1 diagnostic's pooled pairwise-cosine
statistic AT THE TRAINED FINAL CHECKPOINT"); GATE: predicted K=16 h=4
rec@0.9 >= 0.8 under the MEAN mapping. The p10 run and both K=32
predictions are reported alongside, non-gating.

Usage (single GPU per process, matching this harness's own convention --
set CUDA_VISIBLE_DEVICES, do not pass a --gpu index):
  python geo3_drift_diagnostic.py --k 16 32 --out results/geo3_drift/wave_neg1_drift.json
  python geo3_drift_diagnostic.py --k 16 32 48 --out results/geo3_drift/wave_neg1_drift_k48.json --probe-steps 5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
import geo3_simulator as g3sim
import grammar_rd as grd
from model_rd import DeltaNetRDBlock
from run_deltanet_rd import train as train_geo3


def sample_batch_fixed_entity(cfg, batch_size, fixed_entity_id, gen, pools, device):
    """Mirrors grammar_rd.sample_batch_rd's construction, except entity slot
    0 is PINNED to fixed_entity_id in EVERY row (so every row is one
    "episode-context resample" of the SAME fixed entity) and only the other
    K-1 slots are freshly, independently drawn without replacement from the
    train pool minus the fixed entity -- sec 14.6's "fixing one entity's raw
    key and resampling its K-1 episode-mates" construction, verbatim.

    bind() only reads token_ids/beta_mask/item_pos (verified against
    model_rd.py's own bind() signature -- key_ids is read too but ONLY on
    the strong_pin_active branch, which is mutually exclusive with
    geo3_active and therefore dead here) -- key_ids/succ/value_ids are still
    built with FULL grammar fidelity (a real single K-cycle, real per-clause
    VALUE tokens occupying the conv window that feeds item_pos) so the
    measured raw key reflects a REALISTIC episode, not a stripped-down one.
    """
    B, K = batch_size, cfg.K
    buf_len, clause_len, T_bind = cfg.buf_len, cfg.clause_len, cfg.T_bind
    other_pool = pools.train_name_ids[pools.train_name_ids != fixed_entity_id].to(device)
    assert other_pool.numel() >= K - 1, \
        f"train pool minus the fixed entity ({other_pool.numel()}) < K-1={K - 1}"
    scores = torch.rand(B, other_pool.numel(), generator=gen, device=device)
    pool_idx = scores.argsort(dim=-1)[:, :K - 1]
    other_entities = other_pool[pool_idx]                                    # (B,K-1), distinct per row
    fixed_col = torch.full((B, 1), int(fixed_entity_id), dtype=torch.int64, device=device)
    entity_ids = torch.cat([fixed_col, other_entities], dim=1)               # (B,K), slot 0 PINNED

    succ = grd._permutation_graph(B, K, gen, device, dtype=torch.float32)    # single K-cycle (task-faithful)
    key_ids = entity_ids
    value_ids = torch.gather(entity_ids, 1, succ)

    rel_pool = pools.rel_a_ids.to(device)
    rel_idx = torch.randint(0, rel_pool.numel(), (B,), generator=gen, device=device)
    rel_id = rel_pool[rel_idx]

    token_ids = torch.full((B, T_bind), int(pools.buffer_id), dtype=torch.int64, device=device)
    item_pos = (torch.arange(K, device=device) * clause_len + buf_len + 2).unsqueeze(0).expand(B, K).contiguous()
    key_pos = item_pos - 2
    rel_pos = item_pos - 1
    period_pos = item_pos + 1
    token_ids.scatter_(1, key_pos, key_ids)
    token_ids.scatter_(1, rel_pos, rel_id.unsqueeze(1).expand(B, K))
    token_ids.scatter_(1, item_pos, value_ids)
    token_ids.scatter_(1, period_pos,
                        torch.full((B, K), int(pools.period_id), dtype=torch.int64, device=device))
    beta_mask = torch.zeros(B, T_bind, device=device, dtype=torch.float32)
    beta_mask.scatter_(1, item_pos, torch.ones(B, K, device=device))
    return {"token_ids": token_ids, "beta_mask": beta_mask, "item_pos": item_pos}


@torch.no_grad()
def measure_entity_rows(model, cfg, pools, fixed_entity_id, n_resamples, gen, device):
    """One entity's n_resamples orthogonalized-key rows (one per independent
    episode context) -- the raw material geo3_simulator.pairwise_drift_stats
    (and this script's own pooled-aggregation, sec 14.6) consumes."""
    model.eval()
    b = sample_batch_fixed_entity(cfg, n_resamples, fixed_entity_id, gen, pools, device)
    _, k_eff_items, _ = model.bind(b)
    return k_eff_items[:, 0, :].detach()   # (n_resamples, d_state) -- slot 0 = the fixed entity


def measure_drift(model, cfg, pools, n_entities, n_resamples, gen, device):
    """sec 14.6's full pinned statistic: pairwise cosines pooled WITHIN
    entity across resamples, then pooled ACROSS entities; report mean+p10
    of that pooled distribution."""
    entity_pool = pools.train_name_ids.to(device)
    assert entity_pool.numel() >= n_entities
    perm = torch.randperm(entity_pool.numel(), generator=gen, device=device)[:n_entities]
    entities = entity_pool[perm]
    per_entity_stats = []
    all_off_diag = []
    for eid in entities.tolist():
        rows = measure_entity_rows(model, cfg, pools, eid, n_resamples, gen, device)
        mean_c, p10_c = g3sim.pairwise_drift_stats(rows)
        per_entity_stats.append({"entity_id": eid, "mean": mean_c, "p10": p10_c})
        n = rows.shape[0]
        pw = F.cosine_similarity(rows.unsqueeze(0), rows.unsqueeze(1), dim=-1)
        all_off_diag.append(pw[~torch.eye(n, dtype=torch.bool, device=rows.device)])
    pooled = torch.cat(all_off_diag)
    return {
        "mean": pooled.mean().item(), "p10": torch.quantile(pooled, 0.10).item(),
        "n_entities": n_entities, "n_resamples": n_resamples,
        "n_pooled_pairs": pooled.numel(), "per_entity": per_entity_stats,
    }


def run_one_k(K, tokenizer_pools, device, seed, n_entities, n_resamples, probe_steps, probe_batch_size):
    pools, pool_report = tokenizer_pools
    cfg = grd.DeltaNetRDTaskConfig(K=K, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    assert pool_report["n_train_names"] >= max(n_entities, K), \
        f"K={K}: train pool ({pool_report['n_train_names']}) too small for n_entities={n_entities} or K={K}"

    torch.manual_seed(seed)
    model = DeltaNetRDBlock(pools.vocab_size_total, d_model=256, d_state=64, conv_size=cfg.conv_size,
                             buffer_id=pools.buffer_id, geo3_active=True, geo3_n_iter=12,
                             geo3_resid_tol=1e-2).to(device)

    gen_init = torch.Generator(device=device).manual_seed(seed + 1)
    t0 = time.time()
    at_init = measure_drift(model, cfg, pools, n_entities, n_resamples, gen_init, device)
    print(f"  K={K} AT INIT: mean={at_init['mean']:.4f} p10={at_init['p10']:.4f} "
          f"({at_init['n_pooled_pairs']} pooled pairs, {time.time() - t0:.1f}s)", flush=True)

    print(f"  K={K}: probe-training {probe_steps} steps (batch_size={probe_batch_size})...", flush=True)
    t1 = time.time()
    probe_result = train_geo3(model, cfg, pools, pool_report, device, d_model=256, d_state=64,
                               steps=probe_steps, batch_size=probe_batch_size, lr=3e-4, seed=seed,
                               log_every=max(1, probe_steps // 5), ckpt_every=probe_steps + 1,
                               exactness_config={"geo3_active": True, "geo3_n_iter": 12,
                                                  "geo3_resid_tol": 1e-2})
    print(f"  K={K}: probe train done in {time.time() - t1:.1f}s, "
          f"final loss={probe_result['trajectory'][-1]['loss']:.4f}, "
          f"skip_rate={probe_result['skip_rate']:.4%}", flush=True)

    gen_trained = torch.Generator(device=device).manual_seed(seed + 2)
    t2 = time.time()
    after_probe = measure_drift(model, cfg, pools, n_entities, n_resamples, gen_trained, device)
    print(f"  K={K} AFTER {probe_steps}-STEP PROBE: mean={after_probe['mean']:.4f} "
          f"p10={after_probe['p10']:.4f} ({after_probe['n_pooled_pairs']} pooled pairs, "
          f"{time.time() - t2:.1f}s)", flush=True)

    return {
        "K": K, "at_init": at_init, "after_probe": after_probe,
        "probe_steps": probe_steps, "probe_batch_size": probe_batch_size,
        "probe_final_loss": probe_result["trajectory"][-1]["loss"],
        "probe_skip_rate": probe_result["skip_rate"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[16, 32],
                     help="K values to run (sec 14.7's mandatory cells are 16,32; 48 is the "
                          "optional Reserve-eligible stretch -- pass --k 16 32 48 explicitly to include it).")
    ap.add_argument("--n-entities", type=int, default=8, help="sec 14.6: >=8 entities per K.")
    ap.add_argument("--n-resamples", type=int, default=32, help="sec 14.6: >=32 context resamples per entity.")
    ap.add_argument("--probe-steps", type=int, default=5000, help="sec 14.6: 'a 5,000-step probe train'.")
    ap.add_argument("--probe-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--heldout-frac", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "F-geo-3's bind() requires CUDA (chunk_delta_rule has no CPU path)"
    assert args.n_entities >= 8 and args.n_resamples >= 32, \
        "sec 14.6 pins n_entities>=8 and n_resamples>=32 -- pass --n-entities/--n-resamples " \
        "only to INCREASE them, never below the pinned minimum"

    tokenizer = grd.load_gpt2_tokenizer()
    pools, pool_report = grd.build_entity_pools(tokenizer, heldout_frac=args.heldout_frac, seed=0)
    pools = pools.to(device)

    per_k = {}
    for K in args.k:
        print(f"=== K={K} ===", flush=True)
        per_k[K] = run_one_k(K, (pools, pool_report), device, args.seed, args.n_entities,
                              args.n_resamples, args.probe_steps, args.probe_batch_size)

    # sec 14.6's registered launch read: gated on K=16's TRAINED-checkpoint mean/p10
    # (sec 14.5's pinned drift-band definition reads the TRAINED final checkpoint,
    # not the at-init measurement -- "does training shrink the drift" is the live question).
    assert 16 in per_k, "the sec 14.6 GATE requires K=16 to have been measured -- pass --k with 16 included"
    lr16 = per_k[16]["after_probe"]
    launch = g3sim.launch_read(drift_mean=lr16["mean"], drift_p10=lr16["p10"],
                                gram_resid=1e-2, seed=args.seed, device=device)

    out = {
        "design_ref": "DELTANET_RD_EXACTNESS_DESIGN.md sec 14.6 Wave -1 gating diagnostic",
        "sampling_spec": {"n_entities": args.n_entities, "n_resamples": args.n_resamples,
                            "probe_steps": args.probe_steps, "seed": args.seed},
        "per_k": per_k,
        "launch_read": launch,
        "launch": launch["launch"],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"LAUNCH READ (gate: K=16 h=4 rec@0.9 >= 0.8 under MEAN drift mapping):")
    print(f"  K=16 trained-checkpoint drift: mean={lr16['mean']:.4f} p10={lr16['p10']:.4f}")
    print(f"  predicted K=16 h=4 rec@0.9 (mean mapping) = {launch['predicted_gate_value']:.4f}")
    print(f"  LAUNCH = {launch['launch']}")
    if 32 in per_k:
        lr32 = per_k[32]["after_probe"]
        print(f"  [non-gating] K=32 trained-checkpoint drift: mean={lr32['mean']:.4f} p10={lr32['p10']:.4f}, "
              f"predicted K=32 h=4 rec@0.9 (mean mapping) = {launch['mean'][32]['rec'][4]:.4f}")
    print(f"wrote {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
