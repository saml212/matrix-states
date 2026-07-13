#!/usr/bin/env python3
"""T2a -- instrument-teeth gate (PARAM_AXIS_SCALING_DESIGN.md sec 9.3 T1c /
sec 9.4 T2a), executed against RWKV7-Goose-World3-1.5B-HF and
falcon-mamba-7b.

WHY THIS DRIVER EXISTS SEPARATELY FROM param_axis_r0_driver_v2.py: the main
instrument's `pick_t2_marker_tokens` / `run_t2_planted_copy` are generic over
any (train_tokens, val_tokens, vocab_size, model) tuple -- they are NOT
GPT-2-specific -- but the reference models' native vocabularies (RWKV7-Goose:
65536, falcon-mamba-7b: 65024) do not match our GPT-2/50257 pipeline. Planting
raw GPT-2 token ids into a foreign embedding table would not construct the
intended key->value bigram in that model's input space at all. This driver
bridges that gap the SAME way this repo's own wave_neg1_hf_reference_smoke.py
already does for the (now-retired) Wave -1 check: decode our own GPT-2-
tokenized val text back to raw English text, then RE-TOKENIZE it with the
reference model's OWN tokenizer, and run the PINNED v2 T2 machinery
(pick_t2_marker_tokens / run_t2_planted_copy / check_t2b1_mechanism_exists /
check_t2b2-style bar) entirely in that tokenizer's id space. No statistical
logic is reimplemented; only the corpus/tokenizer plumbing changes.

Delta pool: sec 9.4 pins plant distances to "the main metric's own empirical
Delta distribution" -- this driver reads `all_deltas_pooled` out of the main
driver's r0_v2_result.json (a plain list of integer token-distances, which is
tokenizer-independent) rather than recomputing it.

DISCLOSED LIMITATION: `draw_exclusive_replacement` / `assign_placebo_positions`
(reused verbatim from the v2 module) exclude replacement/placebo positions
landing on `EOT_TOKEN_ID`, a GPT-2-specific module-level constant (50256) --
NOT this driver's actual reference-model EOS id. This under-protects against
landing a placebo on a genuine document boundary in the re-tokenized text; the
effect is a small, disclosed statistical-hygiene gap, not a validity-destroying
one (it can only ever fail to exclude a position it should have, on a small
fraction of candidates near doc boundaries -- it never fabricates a hit).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer  # noqa: E402

from lm_recall_gap_probe_v2_rd import (  # noqa: E402
    pick_t2_marker_tokens, run_t2_planted_copy, check_t2b1_mechanism_exists,
)
from lm_pretrain_rd import CORPUS_DIRS  # noqa: E402 -- maps "openr1-mix-ext" -> "reasoning_mix_eot_extended" etc.

EOT_GPT2 = 50256


class HFLogitsWrapper(torch.nn.Module):
    """Adapts a transformers CausalLM (which returns a ModelOutput with
    `.logits`) to this module's `model(x) -> logits_tensor` calling
    convention, used verbatim by run_ablation_arm / the intact pass inside
    run_t2_planted_copy."""
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        return self.hf_model(input_ids=x).logits


def decode_val_text(data_dir: str, corpus_names: list, n_tokens_each: int) -> str:
    """`corpus_names` are this repo's corpus KEYS (e.g. "openr1-mix-ext"),
    mapped through CORPUS_DIRS to the actual on-disk directory name (e.g.
    "reasoning_mix_eot_extended") -- NOT used as directory names directly."""
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    chunks = []
    for name in corpus_names:
        corpus_dir = CORPUS_DIRS[name]
        val = torch.load(os.path.join(data_dir, corpus_dir, "val.pt"), map_location="cpu")
        ids = val[:n_tokens_each].tolist()
        ids = [i for i in ids if i != EOT_GPT2]
        chunks.append(gpt2_tok.decode(ids))
    return "\n\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--corpus-dirs", nargs="+", default=["openr1-mix-ext", "wikitext-mix-ext"])
    ap.add_argument("--n-source-tokens-each", type=int, default=400_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--delta-pool-file", required=True,
                     help="path to r0_v2_result.json (reads its all_deltas_pooled field)")
    ap.add_argument("--n-plants", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    with open(args.delta_pool_file) as f:
        r0 = json.load(f)
    delta_pool = r0["all_deltas_pooled"]
    assert delta_pool, "empty delta pool -- main driver must run first"

    text = decode_val_text(args.data_dir, args.corpus_dirs, args.n_source_tokens_each)
    print(f"decoded {len(text)} chars of source text from {args.corpus_dirs}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16).to(args.device)
    hf_model.eval()
    model = HFLogitsWrapper(hf_model).to(args.device)
    model.eval()
    vocab_size = getattr(hf_model.config, "vocab_size", None) or len(tok)

    enc = tok(text, return_tensors=None)["input_ids"]
    print(f"re-tokenized to {len(enc)} tokens with {args.model}'s own tokenizer "
          f"(vocab_size={vocab_size})", flush=True)
    ids_t = torch.tensor(enc, dtype=torch.long)
    half = len(ids_t) // 2
    train_ids, val_ids = ids_t[:half], ids_t[half:]
    if len(val_ids) < args.seq_len + 4:
        val_ids = ids_t  # not enough re-tokenized text for a held-out half; reuse all of it
    # run_t2_planted_copy calls get_batch(val_tokens, ..., gen) with a CUDA generator --
    # the token tensor must live on the same device (same contract as load_corpus's return).
    val_ids = val_ids.to(args.device)

    tok_a, tok_b = pick_t2_marker_tokens(train_ids, vocab_size, args.device)
    print(f"marker tokens (this tokenizer's own ids): tok_a={tok_a} ({tok.decode([tok_a])!r}), "
          f"tok_b={tok_b} ({tok.decode([tok_b])!r})", flush=True)

    t2_result = run_t2_planted_copy(model, val_ids, batch_size=1, seq_len=args.seq_len,
                                     device=args.device, seed=0, delta_pool=delta_pool,
                                     tok_a=tok_a, tok_b=tok_b, n_plants=args.n_plants,
                                     vocab_size=vocab_size, eval_micro_batch=32)
    t2b1 = check_t2b1_mechanism_exists(t2_result["records"])

    # sec 9.4 T2a bar: acc_copy >= 0.90 at the Delta-median, and >= 0.75 in every
    # Delta-decile carrying >=10% of the candidate mass.
    deltas_sorted = sorted(r["delta"] for r in t2_result["records"])
    n = len(deltas_sorted)
    median_delta = deltas_sorted[n // 2] if n else None
    near_median = [r for r in t2_result["records"] if median_delta is not None
                   and abs(r["delta"] - median_delta) <= max(1, int(0.05 * median_delta))]
    acc_at_median = (sum(r["hit_intact"] for r in near_median) / len(near_median)
                     if near_median else None)

    # decile pass
    decile_pass = True
    decile_report = []
    if n >= 10:
        deciles = sorted(set(deltas_sorted))
        import math
        edges = [deltas_sorted[int(q * (n - 1))] for q in [i / 10 for i in range(11)]]
        for lo, hi in zip(edges[:-1], edges[1:]):
            bucket = [r for r in t2_result["records"] if lo <= r["delta"] <= hi]
            frac_mass = len(bucket) / n
            if frac_mass < 0.10 or not bucket:
                continue
            acc_bucket = sum(r["hit_intact"] for r in bucket) / len(bucket)
            decile_report.append({"lo": lo, "hi": hi, "frac_mass": frac_mass, "acc": acc_bucket})
            if acc_bucket < 0.75:
                decile_pass = False

    t1c_pass = (acc_at_median is not None and acc_at_median >= 0.90) and decile_pass

    result = {
        "model": args.model, "vocab_size": vocab_size,
        "n_retokenized_tokens": len(enc), "n_plants": t2_result["n_plants"],
        "tok_a": tok_a, "tok_b": tok_b,
        "acc_copy_overall": t2_result["acc_copy"], "acc_copy_se": t2_result["acc_copy_se"],
        "median_delta": median_delta, "acc_at_median_delta": acc_at_median,
        "decile_report": decile_report, "decile_pass": decile_pass,
        "t2b1_mechanism_exists": t2b1,
        "t2a_pass": bool(t1c_pass and t2b1["passes"]),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0 if result["t2a_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
