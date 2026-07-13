#!/usr/bin/env python3
"""Real-corpus scan for BUG 2 (t2a_reference_driver_v2_rd.py bridge boundary
collision, RWKV7-Goose-World3-1.5B-HF).

Proves, by actually re-tokenizing the FULL real bridged corpus (both
REQUIRED corpora, both train+val splits) through the exact production
pipeline (decode_corpus_to_documents -> per-document ref_tokenizer encode
with add_special_tokens=False, exactly matching _retokenize_documents),
whether candidate separator token ids collide with real in-document text.

Candidates checked:
  - 65530 (current eos_token_id, string '\\n\\n')  -- the KNOWN collider,
    counted for confirmation/contrast, not as a candidate fix.
  - 0     (bos_token_id == pad_token_id == unk_token_id, string
    '<|rwkv_tokenizer_end_of_text|>') -- the tokenizer's own HF
    "added special token" (see _added_tokens_decoder in
    hf_rwkv_tokenizer.py), never part of the raw byte-level trie vocab.

Does NOT stop at the first collision (unlike the production assertion) --
counts every occurrence across every document, so both a "zero" and a
"nonzero, here's exactly how many" answer are informative.
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

from transformers import AutoTokenizer, GPT2TokenizerFast  # noqa: E402
from t2a_reference_driver_v2_rd import decode_corpus_to_documents, REQUIRED_CORPORA  # noqa: E402
from lm_pretrain_rd import load_corpus, EOT_TOKEN_ID as GPT2_EOT, DEFAULT_DATA_DIR  # noqa: E402

CANDIDATES = {65530: "eos_token ('\\n\\n', current/broken)", 0: "bos/pad/unk_token ('<|rwkv_tokenizer_end_of_text|>', proposed)"}

def scan():
    print("Loading RWKV7-Goose-World3-1.5B-HF tokenizer...", flush=True)
    ref_tok = AutoTokenizer.from_pretrained("RWKV/RWKV7-Goose-World3-1.5B-HF", trust_remote_code=True)
    print("Loading GPT2 fast tokenizer (for decode)...", flush=True)
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")

    grand_totals = {cid: 0 for cid in CANDIDATES}
    grand_docs_with_hit = {cid: 0 for cid in CANDIDATES}
    grand_tokens = 0
    grand_docs = 0

    for corpus_name in REQUIRED_CORPORA:
        for split in ("train", "val"):
            t_load0 = time.time()
            train, val, meta, train_offs, val_offs = load_corpus(DEFAULT_DATA_DIR, corpus_name, "cpu")
            tokens, offs = (train, train_offs) if split == "train" else (val, val_offs)
            docs = decode_corpus_to_documents(tokens, offs, gpt2_tok, None, eot_id=GPT2_EOT)
            print(f"[{corpus_name}/{split}] decoded {len(docs):,} documents "
                  f"({time.time() - t_load0:.1f}s)", flush=True)

            cell_totals = {cid: 0 for cid in CANDIDATES}
            cell_docs_with_hit = {cid: 0 for cid in CANDIDATES}
            n_tokens_this_split = 0
            t0 = time.time()
            for i, doc_text in enumerate(docs):
                if not doc_text:
                    continue
                try:
                    enc = ref_tok(doc_text, return_tensors=None, add_special_tokens=False)["input_ids"]
                except TypeError:
                    enc = ref_tok(doc_text, return_tensors=None)["input_ids"]
                if not enc:
                    continue
                n_tokens_this_split += len(enc)
                for cid in CANDIDATES:
                    c = enc.count(cid)
                    if c:
                        cell_totals[cid] += c
                        cell_docs_with_hit[cid] += 1
                if (i + 1) % 20000 == 0:
                    elapsed = time.time() - t0
                    print(f"  ...[{corpus_name}/{split}] {i+1:,}/{len(docs):,} docs, "
                          f"{n_tokens_this_split:,} tokens so far, {elapsed:.0f}s elapsed, "
                          f"running totals={cell_totals}", flush=True)

            elapsed = time.time() - t0
            print(f"[{corpus_name}/{split}] DONE: {len(docs):,} docs, {n_tokens_this_split:,} "
                  f"ref-tokenizer tokens, {elapsed:.1f}s", flush=True)
            for cid, label in CANDIDATES.items():
                print(f"    id={cid} ({label}): total occurrences={cell_totals[cid]:,}, "
                      f"in {cell_docs_with_hit[cid]:,}/{len(docs):,} documents", flush=True)
                grand_totals[cid] += cell_totals[cid]
                grand_docs_with_hit[cid] += cell_docs_with_hit[cid]
            grand_tokens += n_tokens_this_split
            grand_docs += len(docs)

    print("\n" + "=" * 78)
    print("GRAND TOTAL across BOTH required corpora x BOTH splits "
          f"({grand_docs:,} documents, {grand_tokens:,} ref-tokenizer tokens):")
    for cid, label in CANDIDATES.items():
        verdict = "ZERO -- SAFE AS SEPARATOR" if grand_totals[cid] == 0 else "NONZERO -- COLLIDES, DO NOT USE"
        print(f"  id={cid} ({label}): {grand_totals[cid]:,} occurrences in "
              f"{grand_docs_with_hit[cid]:,} documents -> {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    scan()
