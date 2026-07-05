import json, statistics

FILES = {
    "wave1ext": [
        "results/lm_rd_trackc/wave1ext/w1ext_rung1_lm_openr1-mix-ext_dm768_ds64_L12_s0.json",
        "results/lm_rd_trackc/wave1ext/w1ext_rung1_lm_openr1-mix-ext_dm768_ds64_L12_s1.json",
        "results/lm_rd_trackc/wave1ext/w1ext_rung1_lm_openr1-mix-ext_dm768_ds64_L12_s2.json",
        "results/lm_rd_trackc/wave1ext/w1ext_rung1_lm_wikitext-mix-ext_dm768_ds64_L12_s0.json",
        "results/lm_rd_trackc/wave1ext/w1ext_rung1_lm_wikitext-mix-ext_dm768_ds64_L12_s1.json",
        "results/lm_rd_trackc/wave1ext/w1ext_rung1_lm_wikitext-mix-ext_dm768_ds64_L12_s2.json",
    ],
}


def eff_rank_frac1_mean(rank_stats_summary_corpus):
    vals = [v for k, v in rank_stats_summary_corpus.items() if k.endswith("_f1.0")]
    return statistics.mean(vals) if vals else None


out = {}
for wave, paths in FILES.items():
    out[wave] = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        cks = d.get("checkpoints", [])
        last_ck = cks[-1] if cks else {}
        first_ck = cks[0] if cks else {}
        rec = {
            "path": p,
            "run_name": d.get("run_name"),
            "corpus": d.get("corpus"),
            "other_corpus": d.get("other_corpus"),
            "seed": d.get("seed"),
            "d_model": d.get("d_model"),
            "d_state": d.get("d_state"),
            "n_layers": d.get("n_layers"),
            "num_heads": d.get("num_heads"),
            "steps": d.get("steps"),
            "steps_completed": d.get("steps_completed"),
            "complete": d.get("complete"),
            "n_skipped_steps": d.get("n_skipped_steps"),
            "skip_rate": d.get("skip_rate"),
            "timed_out": d.get("timed_out"),
            "n_params": d.get("n_params"),
            "wall_s": d.get("wall_s"),
            "final_step": d.get("final_step"),
            "final_checkpoint_path": d.get("final_checkpoint_path"),
            "n_checkpoints": len(cks),
            "first_ckpt_step": first_ck.get("step"),
            "final_ckpt_step": last_ck.get("step"),
            "final_val_loss": last_ck.get("val_loss"),
            "final_eff_rank_frac1_by_corpus": {
                c: eff_rank_frac1_mean(v) for c, v in (last_ck.get("rank_stats_summary") or {}).items()
            },
            "peak_memory_allocated_bytes": d.get("peak_memory_allocated_bytes"),
            "peak_memory_reserved_bytes": d.get("peak_memory_reserved_bytes"),
        }
        out[wave].append(rec)

with open("/tmp/run_summaries_wave1ext.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("wrote /tmp/run_summaries_wave1ext.json")
for wave, recs in out.items():
    for r in recs:
        print(wave, r["run_name"], "final_step=", r["final_ckpt_step"], "val_loss=", r["final_val_loss"],
              "eff_rank_f1=", r["final_eff_rank_frac1_by_corpus"], "n_params=", r["n_params"],
              "skip_rate=", r["skip_rate"], "timed_out=", r["timed_out"], "complete=", r["complete"])
