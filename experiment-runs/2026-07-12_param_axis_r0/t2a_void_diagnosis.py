#!/usr/bin/env python3
"""VOID DIAGNOSIS (PARAM_AXIS_SCALING_DESIGN.md sec 9.5's VOID row: "HALT. No
verdict. Diagnose.").

T2a failed on BOTH reference models (RWKV7-Goose-1.5B acc_copy=0.113,
falcon-mamba-7b acc_copy=0.234, both vs a >=0.90 bar) while T2b-1
(mechanism-exists) fired decisively on both (n_minus=0, p=1e-17 / 7e-37).
That combination says the plumbing WORKS and the BAR is unreachable. This
script tests the mechanistic hypothesis for WHY, with zero model forwards --
pure token statistics on the exact plant construction the pinned instrument
uses.

HYPOTHESIS (H-KEYFREQ): `pick_t2_marker_tokens` selects a key token `tok_a`
that is individually HIGH-FREQUENCY (drawn from the top-`entropy_pool`=400
tokens BY COUNT, with min_freq=200). A high-frequency key recurs MANY TIMES
inside the same 512-token window. `run_t2_planted_copy` plants ONE (tok_a ->
tok_b) pair and then queries a LATER occurrence of tok_a. But every OTHER
naturally-occurring tok_a in that window carries its OWN natural continuation.
The planted association is therefore one of many competing (tok_a -> ?)
associations in context, and an argmax read at the query position follows the
aggregate/natural prior rather than the single plant. If true, acc_copy is a
severe UNDER-estimate of one-shot copy ability for EVERY model -- ours and the
references alike -- and the absolute >=0.90 bar is unreachable BY
CONSTRUCTION, not by model deficiency.

MEASUREMENT: the distribution of `count(tok_a in window)` over the exact
windows the T2a run used. A well-posed one-shot copy probe needs that count
to be ~1 (the plant) + 1 (the query) = 2. If it is much larger, H-KEYFREQ is
confirmed.
"""
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

import random  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, GPT2Tokenizer  # noqa: E402

from lm_recall_gap_probe_v2_rd import (  # noqa: E402
    pick_t2_marker_tokens, _combine_seed, _make_window, VOCAB_SIZE,
)
from lm_pretrain_rd import load_corpus, get_batch, CORPUS_DIRS, DEFAULT_DATA_DIR, corpus_fixed_seed  # noqa: E402

EOT_GPT2 = 50256
SEQ_LEN = 512
N_PLANTS = 512
DEVICE = "cuda"


def keycount_stats(windows_tokens, tok_a):
    """count(tok_a) per window, over the same windows the plant run used."""
    counts = [int((w == tok_a).sum().item()) for w in windows_tokens]
    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    return {
        "n_windows": n,
        "mean_count_tok_a_per_window": sum(counts) / n,
        "median": counts_sorted[n // 2],
        "p10": counts_sorted[int(0.10 * n)],
        "p90": counts_sorted[int(0.90 * n)],
        "max": counts_sorted[-1],
        "frac_windows_with_ge_5_occurrences": sum(1 for c in counts if c >= 5) / n,
    }


def main():
    out = {"hypothesis": "H-KEYFREQ", "seq_len": SEQ_LEN, "cases": []}

    # ---- Case 1+2: the two REFERENCE models' own tokenizations (the exact T2a setup) ----
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    chunks = []
    for name in ["openr1-mix-ext", "wikitext-mix-ext"]:
        val = torch.load(os.path.join(DEFAULT_DATA_DIR, CORPUS_DIRS[name], "val.pt"),
                          map_location="cpu")
        ids = [i for i in val[:400_000].tolist() if i != EOT_GPT2]
        chunks.append(gpt2_tok.decode(ids))
    text = "\n\n".join(chunks)

    for model_id in ["RWKV/RWKV7-Goose-World3-1.5B-HF", "tiiuae/falcon-mamba-7b"]:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        enc = tok(text, return_tensors=None)["input_ids"]
        vocab_size = 65536 if "RWKV" in model_id else 65024
        ids_t = torch.tensor(enc, dtype=torch.long)
        half = len(ids_t) // 2
        train_ids, val_ids = ids_t[:half], ids_t[half:].to(DEVICE)
        tok_a, tok_b = pick_t2_marker_tokens(train_ids, vocab_size, DEVICE)

        # reproduce run_t2_planted_copy's OWN window draw, seed=0 (the T2a call)
        gen = torch.Generator(device=DEVICE).manual_seed(0 + 555)
        x0, y0 = get_batch(val_ids, N_PLANTS, SEQ_LEN, gen)
        window = _make_window(x0, y0)
        stats = keycount_stats([window[b] for b in range(window.shape[0])], tok_a)
        stats.update({"model": model_id, "tok_a": tok_a, "tok_b": tok_b,
                       "tok_a_decoded": tok.decode([tok_a]), "tok_b_decoded": tok.decode([tok_b])})
        out["cases"].append(stats)
        print(json.dumps(stats, indent=2), flush=True)

    # ---- Case 3+4: OUR OWN rungs' GPT-2 tokenization, both corpora ----
    for corpus in ["openr1-mix-ext", "wikitext-mix-ext"]:
        train_tokens, val_tokens, _, _, _ = load_corpus(DEFAULT_DATA_DIR, corpus, DEVICE)
        tok_a, tok_b = pick_t2_marker_tokens(train_tokens, VOCAB_SIZE, DEVICE)
        seed = corpus_fixed_seed(corpus) + 424242 + 777    # the driver's t2_seed
        gen = torch.Generator(device=DEVICE).manual_seed(seed + 555)
        x0, y0 = get_batch(val_tokens, N_PLANTS, SEQ_LEN, gen)
        window = _make_window(x0, y0)
        stats = keycount_stats([window[b] for b in range(window.shape[0])], tok_a)
        stats.update({"model": f"OUR RUNGS (GPT-2 tokenizer) / {corpus}",
                       "tok_a": tok_a, "tok_b": tok_b,
                       "tok_a_decoded": gpt2_tok.decode([tok_a]),
                       "tok_b_decoded": gpt2_tok.decode([tok_b])})
        out["cases"].append(stats)
        print(json.dumps(stats, indent=2), flush=True)

    with open("/home/nvidia/chapter2/deltanet_rd/results/param_axis_r0/t2a_void_diagnosis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote t2a_void_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
