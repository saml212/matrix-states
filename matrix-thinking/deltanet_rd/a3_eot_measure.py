#!/usr/bin/env python3
"""A-3(ii) FAITHFUL MEASUREMENT: does a wrong (non-collider) probe-time
eot_override drive a REAL model's acc_copy down?

Runs the DEPLOYED run_t2_repaired_probe END-TO-END on gpt2-large (W2, real
induction transformer, via the DEPLOYED HFLogitsWrapper) over the real openr1
corpus, under two eot_override values:

  correct : eot = 50256 (GPT2 <|endoftext|>, the real doc boundary)
  wrong   : eot = a rare NON-COLLIDER token  => the REAL 50256 boundaries are
            NOT excluded from candidate/placebo/pool logic (the concerning
            direction: real boundaries leak in), and the wrong id (essentially
            absent) excludes almost nothing.

The eot_override feeds build_key_value_pools(eot_token_id=), and (via the module
global) detect_candidates_and_baseline / assign_placebo_positions /
draw_exclusive_replacement. It does NOT touch the plant-window sampling
(get_batch) or the arm-1 argmax readout. So the prediction is: acc_copy and the
KS-based T2a-1 verdict are UNCHANGED. This measures it directly.

Same seed => same window population; the ONLY variable is the eot id.
"""
import os, sys, json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_recall_gap_probe_v2_rd as probe
import t2a_reference_driver_v2_rd as driver

DEVICE = "cuda"
SEQ = 512
NW = 512                 # n_windows == n_plants (kept small; pools use the FULL train corpus)
VOCAB = 50257
CORRECT_EOT = 50256
WRONG_EOT = 50255        # rare non-collider: real 50256 boundaries now leak into candidate/pool logic
DATA = "/data/deltanet_rd_data"
CORPUS = "openr1-mix-ext"

print(f"[load] gpt2-large bf16 + DEPLOYED HFLogitsWrapper on {DEVICE}", flush=True)
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.bfloat16).to(DEVICE).eval()
model = driver.HFLogitsWrapper(hf).to(DEVICE).eval()

print(f"[load] FULL corpus {CORPUS} (pools need the full train split)", flush=True)
train, val, meta, _, _ = probe.load_corpus(DATA, CORPUS, DEVICE)
print(f"  train={train.numel():,} val={val.numel():,}", flush=True)
mode_next = probe.build_bigram_mode_table(train, VOCAB, DEVICE)
counts = torch.bincount(train, minlength=VOCAB).tolist()
seed = probe.corpus_fixed_seed(CORPUS) + driver.RUNG_MATCHING_SEED_OFFSET


def run_one(eot):
    with driver.eot_override(eot):
        did = probe.run_did_eval(model, val, 32, SEQ, NW, DEVICE, mode_next, seed, vocab_size=VOCAB)
        delta_pool = [r["delta"] for r in did["records"]]
        pools = probe.build_key_value_pools(train, VOCAB, DEVICE, eot_token_id=eot)
        floors_ok = bool(pools["floor_pool_key"] and pools["floor_pool_val"] and pools["floor_licensed_b"])
        if not (delta_pool and floors_ok):
            return {"eot": eot, "void": True, "reason": "empty delta / pool floor",
                    "delta_pool_n": len(delta_pool),
                    "pools": {k: pools.get(k) for k in ("n_p_key", "median_p_val", "n_licensed_b_ge2")}}
        t2 = probe.run_t2_repaired_probe(model, val, SEQ, DEVICE, CORPUS, delta_pool, pools,
                                          counts, NW, vocab_size=VOCAB)
    if t2.get("void"):
        return {"eot": eot, "void": True, "reason": t2.get("void_reason")}
    recs = t2["records"]
    t2b1 = probe.check_t2b1_mechanism_exists(recs)
    t2b1b = probe.check_t2b1b_key_conditioned(recs)
    t2a1 = probe.check_t2a1_ceiling(recs, t2b1, t2b1b)
    return {
        "eot": eot, "void": False,
        "n_plants": t2["n_plants"], "n_dropped": t2["n_dropped"],
        "acc_copy": round(t2["acc_copy"], 4),
        "delta_pool_n": len(delta_pool),
        "aiming_argmax_changed_keyswap": round(t2["logit_liveness"]["argmax_changed_frac_keyswap"], 4),
        "ks_point": round(t2a1.get("leg_iv_ks_point", float("nan")), 4),
        "ks_ci": [round(x, 4) if x is not None else None for x in (t2a1.get("leg_iv_ks_ci") or [None, None])],
        "prior": round(t2a1.get("prior", float("nan")), 4),
        "leg_iv_ks_ci_excl0": t2a1.get("leg_iv_ks_ci_excludes_zero_and_t2b1b"),
        "leg_vi_aiming": t2a1.get("leg_vi_aiming_keyswap_argmax_changed"),
        "t2a1_passes": t2a1.get("passes"),
        "t2b1_passes": t2b1.get("passes"), "t2b1b_passes": t2b1b.get("passes"),
    }


out = {}
for tag, eot in (("correct", CORRECT_EOT), ("wrong", WRONG_EOT)):
    print(f"\n[A-3(ii) {tag}] eot_override={eot}", flush=True)
    r = run_one(eot)
    out[tag] = r
    print(json.dumps(r, indent=2), flush=True)

print("\n=== A-3(ii) VERDICT COMPARISON (correct eot vs wrong non-collider eot) ===")
if not out["correct"].get("void") and not out["wrong"].get("void"):
    dc = out["wrong"]["acc_copy"] - out["correct"]["acc_copy"]
    print(f"  acc_copy: correct={out['correct']['acc_copy']:.4f}  wrong={out['wrong']['acc_copy']:.4f}  "
          f"(delta = {dc:+.4f})")
    print(f"  KS:       correct={out['correct']['ks_point']:.4f} ci={out['correct']['ks_ci']}  "
          f"wrong={out['wrong']['ks_point']:.4f} ci={out['wrong']['ks_ci']}")
    print(f"  T2a-1 verdict:  correct passes={out['correct']['t2a1_passes']}  "
          f"wrong passes={out['wrong']['t2a1_passes']}")
print(json.dumps(out, indent=2))
