"""Harvest analysis for Track C Wave 2 (rung-2) + mixcontrol attractor probes.

Computes, from the probe output JSONs' per_checkpoint/per_corpus/per_layer
summaries, pooled gram-deviation stats restricted to arbitrary eval-corpus
subsets (exact pooling: combined mean via n-weighted average, combined std via
E[x^2] = var + mean^2 -- valid because summarize_gram_records reports
population std, unbiased=False).

Why: the archived rung-1 harvest (2026-07-04) pooled over the 4 eval corpora
that existed on-box then ({openr1, openr1-mix, wikitext, wikitext-mix});
today's box has 7 (adds {openr1-mix-ext, wikitext-mix-ext, openr1-stress}),
so the new probes' top-level 'pooled' blocks are NOT corpus-matched to the
archived numbers. This script recomputes matched subsets.
"""
import json
import math
import sys

ARCHIVED_4 = ["openr1", "openr1-mix", "wikitext", "wikitext-mix"]
EXT_2 = ["openr1-mix-ext", "wikitext-mix-ext"]


def pooled_subset(probe_json, corpora):
    """Exact pooled mean/std/eff_rank/stable_rank over the given eval corpora,
    across every checkpoint and layer in the probe output."""
    tot_n = 0
    sum_x = 0.0
    sum_x2 = 0.0
    sum_er = 0.0
    sum_sr = 0.0
    n_excluded = 0
    for ckpt, cinfo in probe_json["per_checkpoint"].items():
        for corpus in corpora:
            if corpus not in cinfo["per_corpus"]:
                continue
            for layer, s in cinfo["per_corpus"][corpus]["per_layer"].items():
                n = s["n_scored"]
                if n == 0:
                    n_excluded += s.get("n_excluded_below_min_valid", 0)
                    continue
                m = s["gram_deviation_mean"]
                sd = s["gram_deviation_std"]
                tot_n += n
                sum_x += n * m
                sum_x2 += n * (sd * sd + m * m)
                sum_er += n * s["effective_rank_mean"]
                sum_sr += n * s["stable_rank_mean"]
                n_excluded += s.get("n_excluded_below_min_valid", 0)
    if tot_n == 0:
        return None
    mean = sum_x / tot_n
    var = sum_x2 / tot_n - mean * mean
    return {
        "n_scored": tot_n,
        "n_excluded": n_excluded,
        "gram_deviation_mean": mean,
        "gram_deviation_std": math.sqrt(max(var, 0.0)),
        "effective_rank_mean": sum_er / tot_n,
        "stable_rank_mean": sum_sr / tot_n,
    }


def per_corpus_table(probe_json):
    """Per-eval-corpus pooled stats (across checkpoints and layers)."""
    corpora = set()
    for cinfo in probe_json["per_checkpoint"].values():
        corpora.update(cinfo["per_corpus"].keys())
    return {c: pooled_subset(probe_json, [c]) for c in sorted(corpora)}


def per_checkpoint_pooled(probe_json, corpora):
    """Pooled stats per individual checkpoint (for per-run / per-seed rows)."""
    out = {}
    for ckpt, cinfo in probe_json["per_checkpoint"].items():
        sub = {"per_checkpoint": {ckpt: cinfo}}
        out[ckpt] = pooled_subset(sub, corpora)
    return out


def anchors(K, d):
    return {"random": math.sqrt(K * (K - 1) / d), "collapse": math.sqrt(K * (K - 1))}


def span_frac(gd, K, d):
    a = anchors(K, d)
    return (gd - a["random"]) / (a["collapse"] - a["random"])


def main():
    files = sys.argv[1:]
    result = {}
    for path in files:
        with open(path) as f:
            pj = json.load(f)
        # infer (K,d): K = chunk_size, d = head_dim = d_state/num_heads
        any_ck = next(iter(pj["per_checkpoint"].values()))
        cfg = any_ck["config"]
        K = pj["chunk_size"]
        d = cfg["d_state"] // cfg.get("num_heads", 1)
        a = anchors(K, d)
        entry = {
            "config": {k: cfg[k] for k in ("d_model", "d_state", "n_layers", "num_heads")},
            "n_params": any_ck["n_params"],
            "K": K, "d": d, "anchors": a,
            "pooled_all_corpora_as_written": pj["pooled"],
            "pooled_archived4": pooled_subset(pj, ARCHIVED_4),
            "pooled_ext2": pooled_subset(pj, EXT_2),
            "per_corpus": per_corpus_table(pj),
            "per_checkpoint_pooled_all": per_checkpoint_pooled(
                pj, ARCHIVED_4 + EXT_2 + ["openr1-stress"]),
            "per_checkpoint_pooled_ext2": per_checkpoint_pooled(pj, EXT_2),
        }
        for key in ("pooled_all_corpora_as_written", "pooled_archived4", "pooled_ext2"):
            if entry[key] and "gram_deviation_mean" in entry[key]:
                entry[key]["span_frac"] = span_frac(entry[key]["gram_deviation_mean"], K, d)
        result[path] = entry
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
