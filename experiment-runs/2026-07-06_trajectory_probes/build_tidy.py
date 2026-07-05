"""Build a tidy per-(cell, seed, checkpoint) trajectory table from the
lm_attractor_probe_rd.py per-run JSON outputs.

This is a pure post-processing / re-analysis tool (no GPU, no torch) --
reuses the exact archived-4 n-weighted pooling convention from
analyze_probe_wave2.py (2026-07-05 harvests) so every row's span_frac is
computed the SAME way as the already-published final-checkpoint numbers.

Input: one or more probe-output JSON files, each holding one or more
checkpoints' `per_checkpoint` blocks (as written by lm_attractor_probe_rd.py).
Each checkpoint entry already carries its own corpus_trained_on/seed/step/
n_params/config -- no filename parsing needed for that metadata (only the
family label, since the probe doesn't record which "wave" it came from).

Output: tidy.json (list of records) + tidy.csv, one row per (checkpoint).
"""
import csv
import json
import math
import sys

ARCHIVED_4 = ["openr1", "openr1-mix", "wikitext", "wikitext-mix"]


def pooled_subset_one_checkpoint(cinfo, corpora):
    """Same exact-pooling arithmetic as analyze_probe_wave2.py's
    pooled_subset, restricted to a single checkpoint's per_corpus block."""
    tot_n = 0
    sum_x = 0.0
    sum_x2 = 0.0
    sum_er = 0.0
    sum_sr = 0.0
    n_excluded = 0
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


def anchors(K, d):
    return {"random": math.sqrt(K * (K - 1) / d), "collapse": math.sqrt(K * (K - 1))}


def span_frac(gd, K, d):
    a = anchors(K, d)
    return (gd - a["random"]) / (a["collapse"] - a["random"])


def build_rows(probe_json_path, family_label):
    with open(probe_json_path) as f:
        pj = json.load(f)
    K = pj["chunk_size"]
    rows = []
    for ckpt_path, cinfo in pj["per_checkpoint"].items():
        cfg = cinfo["config"]
        d = cfg["d_state"] // cfg.get("num_heads", 1)
        a4 = pooled_subset_one_checkpoint(cinfo, ARCHIVED_4)
        all7 = pooled_subset_one_checkpoint(cinfo, list(cinfo["per_corpus"].keys()))
        row = {
            "family": family_label,
            "checkpoint_path": ckpt_path,
            "corpus_trained_on": cinfo.get("corpus_trained_on"),
            "seed": cinfo.get("seed"),
            "step": cinfo.get("step"),
            "n_params": cinfo.get("n_params"),
            "d_model": cfg.get("d_model"),
            "d_state": cfg.get("d_state"),
            "n_layers": cfg.get("n_layers"),
            "K": K,
            "d_head": d,
            "anchor_random": anchors(K, d)["random"],
            "anchor_collapse": anchors(K, d)["collapse"],
        }
        if a4 is not None:
            row["archived4_gram_deviation_mean"] = a4["gram_deviation_mean"]
            row["archived4_gram_deviation_std"] = a4["gram_deviation_std"]
            row["archived4_span_frac"] = span_frac(a4["gram_deviation_mean"], K, d)
            row["archived4_effective_rank_mean"] = a4["effective_rank_mean"]
            row["archived4_stable_rank_mean"] = a4["stable_rank_mean"]
            row["archived4_n_scored"] = a4["n_scored"]
        if all7 is not None:
            row["all7_gram_deviation_mean"] = all7["gram_deviation_mean"]
            row["all7_span_frac"] = span_frac(all7["gram_deviation_mean"], K, d)
            row["all7_n_scored"] = all7["n_scored"]
        rows.append(row)
    return rows


def main():
    # args: pairs of (family_label, json_path)
    args = sys.argv[1:]
    out_prefix = args[-1]
    pairs = args[:-1]
    assert len(pairs) % 2 == 0, "usage: build_tidy.py <family1> <path1> [<family2> <path2> ...] <out_prefix>"
    all_rows = []
    for i in range(0, len(pairs), 2):
        family, path = pairs[i], pairs[i + 1]
        all_rows.extend(build_rows(path, family))

    all_rows.sort(key=lambda r: (r["family"], r["corpus_trained_on"] or "", r["seed"] if r["seed"] is not None else -1, r["step"] or 0))

    with open(f"{out_prefix}.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(f"{out_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"wrote {len(all_rows)} rows to {out_prefix}.json / {out_prefix}.csv")


if __name__ == "__main__":
    main()
