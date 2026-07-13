#!/usr/bin/env python3
"""wave_neg1_hf_reference_smoke.py -- PARAM_AXIS_SCALING_DESIGN.md R0's
"Wave -1" validity smoke (design sec 3-A instruments list / F4 disposition):
"if the instrument cannot read AR on a model known to have it, it has no
teeth." Runs the SAME bigram-ablation-gap ALGORITHM as
lm_recall_gap_probe_rd.py, generically, against a real HF causal LM known
to have associative-recall capability (RWKV7-Goose or falcon-mamba, both
already cached at /data/hf_cache), on real text (decoded from this
project's own GPT-2-tokenized val corpus, then re-tokenized with the
reference model's own tokenizer -- same underlying English text, different
tokenizer, so this is a genuine independent read, not a copy of the
project's own DeltaNet-checkpoint pipeline).

Does NOT touch DeltaNetLM / lm_pretrain_rd.py at all -- this is a clean-room
re-implementation against transformers' generic AutoModelForCausalLM API,
by design (the point is an INDEPENDENT confirmation the algorithm itself
detects real AR, not a confirmation that this project's own plumbing agrees
with itself).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer  # noqa: E402

EOT_GPT2 = 50256


def decode_val_text(data_dir: str, corpus_dir: str, n_tokens: int) -> str:
    """Decodes the first n_tokens of this project's own GPT-2-tokenized val
    split back into raw text via the (offline-cached) GPT2 tokenizer."""
    val = torch.load(os.path.join(data_dir, corpus_dir, "val.pt"), map_location="cpu")
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    ids = val[:n_tokens].tolist()
    ids = [i for i in ids if i != EOT_GPT2]
    return gpt2_tok.decode(ids)


def build_bigram_mode_table_generic(train_ids: list, vocab_size: int) -> dict:
    from collections import Counter, defaultdict
    nxt = defaultdict(Counter)
    for a, b in zip(train_ids[:-1], train_ids[1:]):
        nxt[a][b] += 1
    mode = {}
    for a, c in nxt.items():
        mode[a] = min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return mode


def make_window(ids: list, seq_len: int, start: int) -> list:
    return ids[start:start + seq_len + 1]


def run_generic_ar_hit_gap(model, ids: list, seq_len: int, n_windows: int, device: str,
                            mode_next: dict, vocab_size: int, eos_id: int, seed: int,
                            shuffle_positions: bool = False):
    rng = random.Random(seed)
    ablate_rng = random.Random(seed * 7919 + 1)
    n = len(ids)
    n_cand = hit_intact = hit_ablated = 0
    n_base = hit_base = 0
    tried = 0
    while tried < n_windows and n > seq_len + 2:
        start = rng.randrange(0, n - seq_len - 1)
        window = make_window(ids, seq_len, start)
        tried += 1
        if shuffle_positions:
            perm = list(range(len(window)))
            rng.shuffle(perm)
            window = [window[i] for i in perm]
        x = window[:-1]; y = window[1:]
        T = len(x)
        seen = {}
        cands, baseline = [], []
        for k in range(T):
            a, b = x[k], y[k]
            if a == eos_id or b == eos_id:
                continue
            key = (a, b)
            if key in seen:
                j = seen[key]
                if k - j > 2 and mode_next.get(a) != b:
                    cands.append((k, j))
                continue
            seen[key] = k
            if k % 11 == 0:
                baseline.append(k)
        if not cands and not baseline:
            continue
        x_t = torch.tensor([x], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x_t).logits[0]
        pred = logits.argmax(dim=-1)
        for k in baseline:
            hit_base += int(pred[k].item() == y[k])
            n_base += 1
        if cands:
            window_ab = list(window)
            for k, j in cands:
                true_next = window[j + 1]
                target = y[k]
                repl = true_next
                while repl == true_next or repl == target or repl == eos_id:
                    repl = ablate_rng.randrange(vocab_size)
                window_ab[j + 1] = repl
            x_ab = torch.tensor([window_ab[:-1]], dtype=torch.long, device=device)
            with torch.no_grad():
                pred_ab = model(x_ab).logits[0].argmax(dim=-1)
            for k, j in cands:
                hit_intact += int(pred[k].item() == y[k])
                hit_ablated += int(pred_ab[k].item() == y[k])
                n_cand += 1
    return {
        "n_candidates": n_cand, "n_baseline": n_base,
        "acc_intact": hit_intact / n_cand if n_cand else None,
        "acc_ablated": hit_ablated / n_cand if n_cand else None,
        "gap": (hit_intact - hit_ablated) / n_cand if n_cand else None,
        "acc_baseline_nonAR": hit_base / n_base if n_base else None,
        "shuffle_positions": shuffle_positions, "n_windows_tried": tried,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     help="HF repo id, e.g. RWKV/RWKV7-Goose-World3-1.5B-HF")
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--corpus-dir", default="wikitext103_mix_eot_extended")
    ap.add_argument("--n-source-tokens", type=int, default=200_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-windows", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    text = decode_val_text(args.data_dir, args.corpus_dir, args.n_source_tokens)
    print(f"decoded {len(text)} chars of source text", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16).to(args.device)
    model.eval()
    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else len(tok)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1

    enc = tok(text, return_tensors=None)["input_ids"]
    print(f"re-tokenized to {len(enc)} tokens with {args.model}'s own tokenizer "
          f"(vocab_size={vocab_size})", flush=True)

    mode_next = build_bigram_mode_table_generic(enc, vocab_size)

    real = run_generic_ar_hit_gap(model, enc, args.seq_len, args.n_windows, args.device,
                                   mode_next, vocab_size, eos_id, seed=0, shuffle_positions=False)
    shuf = run_generic_ar_hit_gap(model, enc, args.seq_len, args.n_windows, args.device,
                                   mode_next, vocab_size, eos_id, seed=0, shuffle_positions=True)

    result = {
        "model": args.model, "n_source_tokens": len(enc),
        "real": real, "t1_shuffled_control": shuf,
        "t1_pass": (real["gap"] is not None and shuf["gap"] is not None and abs(shuf["gap"]) < 0.10),
        "instrument_has_teeth": (real["gap"] is not None and real["gap"] > 0.05),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
