"""READ-ONLY mechanistic probe -- sec G3-B23, PART 2 (mechanism-faithful
cyclicity + full singular-value spectrum). The v1 probe's RAW matrix_power
test (Z^24 vs I, unnormalized) found Z is NOT orthogonal in the classic
sense (singular values 0.004..1.31, huge spread) and RAW Z^24 explodes away
from I -- but the model's OWN read mechanism (nm.binexp_read) renormalizes
the running matrix/vector after every squaring/application step (see
ncr_models.py _renorm_mat/_renorm_vec), so cosine-based composition can
still be exact even though the RAW matrix powers are not literally the
identity. This second pass tests cyclicity THE WAY THE MODEL ACTUALLY USES
Z: does nm.binexp_read(Z, entity_i, h=24) point back at entity_i (cos), for
every h in a small ladder, using the REAL K-cycle succ mapping from
grammar_rd (not build_task1_document's stripped-down dict, which drops
`succ`) so multi-hop ground truth is available. Also reports the FULL
per-rank singular-value spectrum (mean over batch, sorted descending) to
see whether a small number of directions carry the non-orthogonality or
whether it is spread across all 25.

Still fully read-only: torch.no_grad() throughout, no writes outside this
scratch dir, no optimizer, no backward.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/home/nvidia/ncr_g3b12_fix")

import torch
import torch.nn.functional as F

import ncr_lm_wave1_runner as runner  # noqa: E402

DEVICE = "cuda"
CKPT_PATH = "/home/nvidia/ncr_g3b12_fix/results/mob_g3b20_s0_ckpts/mob_g3b20_s0.ckpt.pt"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "z_ortho_probe_v2_results.json")
PROBE_SEED = 20260721
BATCH_SIZE = 64
H_LADDER = (1, 2, 3, 5, 12, 20, 24, 25, 48, 61)  # 24/48 = full cycle order(s); 61 mod 24 == 13


@torch.no_grad()
def iterate_succ(succ: torch.Tensor, h: int) -> torch.Tensor:
    """succ: (B,K) int64 bijection on slots. Returns (B,K): succ^h applied
    starting from EVERY slot simultaneously (cur[b,i] = succ^h(i))."""
    B, K = succ.shape
    cur = torch.arange(K, device=succ.device).unsqueeze(0).expand(B, K).clone()
    for _ in range(h):
        cur = torch.gather(succ, 1, cur)
    return cur


@torch.no_grad()
def spectrum(Zf: torch.Tensor) -> list:
    sv = torch.linalg.svdvals(Zf)  # (B,d), ascending order per torch convention
    sv_sorted_desc, _ = torch.sort(sv, dim=-1, descending=True)
    return sv_sorted_desc.mean(dim=0).tolist()  # (d,) mean per rank across batch


@torch.no_grad()
def cyclicity_via_read(nm_mod, Z: torch.Tensor, ent_vecs: torch.Tensor, succ: torch.Tensor, label: str) -> dict:
    """Mechanism-faithful test: apply the model's OWN nm.binexp_read (which
    renormalizes at every step, exactly as the real read pathway does) with
    q = every entity vector simultaneously, at each h in H_LADDER, and
    compare (cosine) the result to entity_adapter(embed(entity_at_succ^h(i)))
    -- the TRUE h-hop target for starting slot i. h=24 (and h=48=2*24)
    starting from ANY slot must return to the SAME slot (a single K=24-cycle
    has order exactly 24 as a permutation, guaranteed by construction, not
    assumed) -- verified directly below as `succ24_is_identity`."""
    B, K, d = ent_vecs.shape
    K24 = iterate_succ(succ, 24)
    ident_check = (K24 == torch.arange(K, device=succ.device).unsqueeze(0)).float().mean().item()

    out = {"label": label, "succ24_is_identity_frac": ident_check, "per_h": {}}
    for h in H_LADDER:
        tgt_slot_h = iterate_succ(succ, h)  # (B,K)
        tgt_vec_h = torch.gather(ent_vecs, 1, tgt_slot_h.unsqueeze(-1).expand(B, K, d))  # (B,K,d)
        o = nm_mod.binexp_read(Z, ent_vecs, h=h)["o"]  # (B,K,d) -- q=ent_vecs applied per-entity-as-query
        cos = F.cosine_similarity(o, tgt_vec_h, dim=-1)  # (B,K)
        finite = torch.isfinite(cos)
        cos_f = cos[finite]
        out["per_h"][str(h)] = {
            "mean_cos_to_true_h_hop_target": cos_f.mean().item() if cos_f.numel() else None,
            "min_cos": cos_f.min().item() if cos_f.numel() else None,
            "frac_cos_ge_0.9": (cos_f >= 0.9).float().mean().item() if cos_f.numel() else None,
            "finite_frac": finite.float().mean().item(),
        }
    return out


def main():
    t0 = time.time()
    ckpt = runner.load_checkpoint(CKPT_PATH, DEVICE)
    assert ckpt is not None, "checkpoint failed to load/validate -- STOP"
    pools, cfg, pool_report = runner.build_grammar_pools_and_cfg(seed=0)
    vocab_size_total = pool_report["vocab_size_total"]
    pools = pools.to(DEVICE)
    arms, opts, data_gen = runner.restore_arms_and_opts(ckpt, vocab_size_total, 3e-4, DEVICE)
    del opts, data_gen
    for arm in arms.values():
        arm["backbone"].eval(); arm["ncr"].eval(); arm["integ"].eval()

    # Direct grammar_rd sample (NOT build_task1_document, which drops `succ`)
    # -- same pools/cfg, same construction, ONE extra returned field (succ).
    gr = runner.graft.gr
    gen = torch.Generator(device=DEVICE).manual_seed(PROBE_SEED)
    b = gr.sample_batch_rd(cfg, BATCH_SIZE, gen, hop_set=(1,), pools=pools, device=DEVICE)
    entity_ids = b["entity_ids"]  # (B,K) token ids, slot order
    succ = b["succ"]              # (B,K) the K-cycle bijection

    results = {}
    spectra = {}
    for arm_name in ("full_graft", "backbone_only"):
        arm = arms[arm_name]
        with torch.no_grad():
            ent_vecs = arm["integ"].entity_adapter(arm["backbone"].embed(entity_ids).float())  # (B,K,d)
            # Z from the SAME K entities as keys/values (matches extract_kv's own
            # key/value construction: keys=values=entity_adapter(embed(.)) pairs
            # defined by succ, i.e. keys_v=ent_vecs, values_v=gather(ent_vecs,succ)).
            values_v = torch.gather(ent_vecs, 1, succ.unsqueeze(-1).expand(-1, -1, ent_vecs.shape[-1]))
            Z = arm["ncr"].encode(ent_vecs, values_v)
        results[arm_name] = cyclicity_via_read(runner.nm, Z, ent_vecs, succ, arm_name)
        spectra[arm_name] = spectrum(Z.float())
        print(f"[probe2] {arm_name}: succ24_is_identity_frac={results[arm_name]['succ24_is_identity_frac']}", flush=True)
        for h in H_LADDER:
            r = results[arm_name]["per_h"][str(h)]
            print(f"  h={h:3d}  mean_cos={r['mean_cos_to_true_h_hop_target']:.4f}  "
                  f"frac>=0.9={r['frac_cos_ge_0.9']:.4f}", flush=True)

    out = {
        "probe": "sec_G3_B23_z_orthogonality_probe_part2_mechanism_faithful_cyclicity",
        "ckpt_step": ckpt["step"], "ckpt_cell_id": ckpt["cell_id"],
        "K": cfg.K, "H_LADDER": list(H_LADDER),
        "singular_value_spectrum_mean_per_rank_desc": spectra,
        "cyclicity_via_binexp_read": results,
        "elapsed_s": time.time() - t0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[probe2] wrote {OUT_PATH}", flush=True)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
