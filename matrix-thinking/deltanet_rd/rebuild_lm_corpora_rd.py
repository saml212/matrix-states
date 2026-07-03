"""rebuild_lm_corpora_rd.py -- AUDIT FIX-3 (2026-07-03): rebuild both LM
corpora WITH <|endoftext|> document separators + doc-offset arrays.

Why (the audit's finding, verified): the original tokenized corpora
(/data/deltanet_rd_data/{reasoning,wikitext103_tokenized}) contain ZERO
<|endoftext|> (50256) tokens, and OpenR1's average document length (466
tokens) is BELOW the probe tier's seq_len=512 -- so a majority of training
windows span unrelated concatenated problems with no boundary signal,
contaminating the reasoning-vs-narrative contrast Wave C/D exists to
measure. The original prep script did not survive (committed empty,
verified in git history 2026-07-03); this script re-derives the recipe
from the raw HF sources and verifies it BIT-EXACTLY against the original
tensors before writing anything.

Reverse-engineered recipes (each verified in verify_*, run-to-completion,
against the original streams -- never assumed):
  - OpenR1  (open-r1/OpenR1-Math-220k, config 'default', split 'train',
    93,733 rows == the original meta.json's n_examples): the original
    stream is the CONCATENATION OF PER-DOCUMENT TOKENIZATIONS of
    `problem + "\\n" + solution + "\\n"`, in dataset order -- verified
    BIT-EXACT over all 46,026,934 tokens (2026-07-03). NOT a single
    full-string tokenizer pass: at boundaries where a document ends with
    "\\n", per-chunk tokenization keeps "\\n\\n" as one merged token (628)
    while a full-string pass pre-tokenizes it as 198+198 (GPT-2's
    `\\s+(?!\\S)` pre-token regex) -- a full-string reconstruction runs
    570 tokens long and diverges first at token 90,619 (measured; the
    diagnostic that identified the real recipe). Original train/val
    split: a FLAT TOKEN SLICE at 43,725,587 (val starts mid-document --
    verified by decoding val.pt's head).
  - WikiText-103 (Salesforce/wikitext, config 'wikitext-103-raw-v1'):
    stream = one tokenizer pass over `"".join(rows)` per split (each raw
    row already carries its trailing "\\n"; blank rows are '' and vanish;
    no "\\n\\n" runs exist, so chunking is irrelevant here -- full-string
    and per-row concat verified IDENTICAL and bit-exact on
    validation/test). Documents = articles, segmented at top-level
    heading rows (' = Title = \\n' -- starts with ' = ', not ' = = ');
    the official article counts (60 validation / 60 test) are asserted
    as the heuristic's own verification.

Rebuild output (per corpus, NEW directories -- the originals are never
touched, pure additive):
  /data/deltanet_rd_data/reasoning_eot/      {train,val}.pt  {train,val}_doc_offsets.pt  meta.json
  /data/deltanet_rd_data/wikitext103_eot/    {train,val,test}.pt  + offsets + meta.json

  - every document is tokenized SEPARATELY and followed by one EOT token
    (50256), so `token == 50256 <=> document boundary` holds by
    construction (asserted: no in-document token is 50256);
  - doc_offsets = int64 tensor of each document's START index;
  - OpenR1's split is document-aligned 95/5 by token mass (closest doc
    boundary to the original 95.0% train fraction) -- a DISCLOSED
    deviation from the original mid-document token slice (recorded in
    meta.json); WikiText keeps the official HF splits (already
    document-aligned).

Run ON THE BOX (needs the HF cache + /data):
  HF_HOME=/data/hf_cache python rebuild_lm_corpora_rd.py \
      --data-dir /data/deltanet_rd_data [--skip-verify]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

EOT = 50256
OPENR1_TRAIN_TOKENS = 43_725_587       # original meta.json, verified on-box
OPENR1_VAL_TOKENS = 2_301_347
WIKITEXT_TOKENS = {"train": 117_920_140, "validation": 247_289, "test": 283_287}


def load_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------

def openr1_docs():
    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default")["train"]
    assert len(ds) == 93_733, f"expected 93,733 rows (original n_examples), got {len(ds)}"
    return [p + "\n" + s for p, s in zip(ds["problem"], ds["solution"])]


def wikitext_docs(split: str):
    """Segment a split's rows into articles. Heading heuristic (verified
    against the official 60/60 validation/test article counts by the
    caller): a TOP-LEVEL heading row starts with ' = ' (not ' = = ') AND is
    surrounded by blank rows -- the surround condition kills two measured
    false-positive classes the prefix test alone admits (2026-07-03, test
    split: content rows like ' = 1 rad / s and a nominal impedance k = \\n'
    and legend rows like ' = Qualified for the next round \\n', all with
    non-blank neighbors). Rows before the first heading (blank preamble)
    are attached to no article iff they are all blank -- a non-blank
    preamble would be silently lost, so it is asserted against instead."""
    from datasets import load_dataset
    # list(...) is LOAD-BEARING: datasets 5.0 returns a LAZY Column here,
    # and is_heading's per-row neighbor indexing through it goes via a
    # per-access Arrow table select -- measured as an effective hang
    # (py-spy: __getitem__ -> _fast_select_column dominating, 2026-07-03)
    # on the 1.8M-row train split. A materialized Python list makes the
    # same loop take seconds.
    rows = list(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")[split]["text"])
    n = len(rows)

    def is_heading(i):
        r = rows[i]
        if not (r.startswith(" = ") and not r.startswith(" = = ")):
            return False
        prev_blank = (i == 0) or rows[i - 1] == ""
        next_blank = (i + 1 == n) or rows[i + 1] == ""
        return prev_blank and next_blank

    docs, cur = [], []
    seen_heading = False
    for i, row in enumerate(rows):
        if is_heading(i):
            if cur and seen_heading:
                docs.append("".join(cur))
            elif cur:
                assert all(r == "" for r in cur), (
                    f"{split}: non-blank content before the first article heading would be "
                    f"dropped -- inspect rows[:{i}]")
            seen_heading = True
            cur = [row]
        else:
            cur.append(row)
    if cur and seen_heading:
        docs.append("".join(cur))
    # content preservation: segmentation must lose ZERO characters (blank
    # preamble rows are '' and contribute none)
    assert sum(len(d) for d in docs) == sum(len(r) for r in rows), \
        f"{split}: article segmentation lost characters -- docs are not a partition of the stream"
    return docs


# ---------------------------------------------------------------------------
# Recipe verification against the ORIGINAL tensors (bit-exact, run first)
# ---------------------------------------------------------------------------

def verify_openr1(tok, docs, data_dir) -> dict:
    """Bit-exact reconstruction of the original stream: concat of PER-DOC
    tokenizations of doc + '\\n' (see module docstring for the measured
    evidence that this -- and not a full-string pass -- is the original
    recipe)."""
    print("[verify openr1] reconstructing per-doc-chunk stream (bit-exact check vs original)...",
          flush=True)
    t0 = time.time()
    ids = []
    B = 2000
    for i in range(0, len(docs), B):
        enc = tok([d + "\n" for d in docs[i:i + B]], add_special_tokens=False)["input_ids"]
        for e in enc:
            ids.extend(e)
    dt = time.time() - t0
    old_train = torch.load(os.path.join(data_dir, "reasoning", "train.pt"), map_location="cpu")
    old_val = torch.load(os.path.join(data_dir, "reasoning", "val.pt"), map_location="cpu")
    n_total = OPENR1_TRAIN_TOKENS + OPENR1_VAL_TOKENS
    assert len(ids) == n_total, \
        f"recipe FAILED: reconstructed stream {len(ids)} tokens != original train+val {n_total}"
    stream_t = torch.tensor(ids, dtype=torch.int64)
    assert torch.equal(stream_t[:OPENR1_TRAIN_TOKENS], old_train), \
        "recipe FAILED: reconstructed stream's train slice is not bit-identical to reasoning/train.pt"
    assert torch.equal(stream_t[OPENR1_TRAIN_TOKENS:], old_val), \
        "recipe FAILED: reconstructed stream's val slice is not bit-identical to reasoning/val.pt"
    print(f"[verify openr1] PASSED: {n_total:,} tokens reconstructed BIT-IDENTICAL to the original "
          f"train.pt+val.pt (recipe: per-doc tok(problem+'\\n'+solution+'\\n'), concatenated, "
          f"dataset order; {dt:.0f}s)", flush=True)
    return {"bit_exact_vs_original": True, "n_tokens_reconstructed": n_total,
            "recipe": "concat of per-doc tok(problem+'\\n'+solution+'\\n')"}


def verify_wikitext(tok, data_dir) -> dict:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    report = {}
    name_map = {"train": "train.pt", "validation": "val.pt", "test": "test.pt"}
    for split, fname in name_map.items():
        print(f"[verify wikitext/{split}] tokenizing ''.join(rows) (bit-exact check)...", flush=True)
        stream = tok("".join(ds[split]["text"]), add_special_tokens=False)["input_ids"]
        old = torch.load(os.path.join(data_dir, "wikitext103_tokenized", fname), map_location="cpu")
        assert len(stream) == WIKITEXT_TOKENS[split] == old.numel(), \
            f"recipe FAILED [{split}]: {len(stream)} vs meta {WIKITEXT_TOKENS[split]} vs file {old.numel()}"
        assert torch.equal(torch.tensor(stream, dtype=torch.int64), old), \
            f"recipe FAILED [{split}]: not bit-identical to the original {fname}"
        print(f"[verify wikitext/{split}] PASSED: {len(stream):,} tokens BIT-IDENTICAL", flush=True)
        report[split] = {"bit_exact_vs_original": True, "n_tokens": len(stream)}
    # article-segmentation heuristic verification: official WikiText-103
    # article counts are 60 validation / 60 test
    n_val_docs = len(wikitext_docs("validation"))
    n_test_docs = len(wikitext_docs("test"))
    assert n_val_docs == 60 and n_test_docs == 60, (
        f"article-heading heuristic FAILED its own verification: got {n_val_docs} validation / "
        f"{n_test_docs} test articles, official counts are 60/60")
    report["heading_heuristic"] = {"validation_docs": n_val_docs, "test_docs": n_test_docs,
                                    "official_counts_matched": True}
    print(f"[verify wikitext] heading heuristic PASSED: 60/60 articles on validation/test "
          f"(official counts)", flush=True)
    return report


# ---------------------------------------------------------------------------
# EOT-separated build
# ---------------------------------------------------------------------------

def tokenize_docs_with_eot(tok, docs, batch_size: int = 1000):
    """Per-document tokenization (batched through the fast tokenizer), one
    EOT appended after every document. Returns (flat int64 tensor,
    doc_offsets int64 tensor of document START indices, per_doc_lens list
    [token count INCLUDING the trailing EOT])."""
    all_ids, offsets, lens = [], [], []
    pos = 0
    for i in range(0, len(docs), batch_size):
        enc = tok(docs[i:i + batch_size], add_special_tokens=False)["input_ids"]
        for ids in enc:
            assert EOT not in ids, (
                f"a document tokenizes to include EOT id {EOT} -- the 'EOT <=> boundary' "
                f"invariant would break; inspect doc index ~{len(offsets)}")
            offsets.append(pos)
            all_ids.extend(ids)
            all_ids.append(EOT)
            pos += len(ids) + 1
            lens.append(len(ids) + 1)
        if (i // batch_size) % 20 == 0:
            print(f"  tokenized {min(i + batch_size, len(docs)):,}/{len(docs):,} docs "
                  f"({pos:,} tokens)", flush=True)
    return (torch.tensor(all_ids, dtype=torch.int64),
            torch.tensor(offsets, dtype=torch.int64), lens)


def split_by_token_mass(doc_offsets: torch.Tensor, n_total: int, train_frac: float):
    """Document-aligned split: the first doc whose START crosses
    train_frac*n_total begins the val split. Returns the split doc index."""
    target = int(round(train_frac * n_total))
    idx = int(torch.searchsorted(doc_offsets, target, right=True).item())
    return max(1, min(idx, doc_offsets.numel() - 1))


def write_corpus(out_dir: str, splits: dict, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    for split, (toks, offs) in splits.items():
        torch.save(toks, os.path.join(out_dir, f"{split}.pt"))
        torch.save(offs, os.path.join(out_dir, f"{split}_doc_offsets.pt"))
        meta[f"{split}_tokens"] = int(toks.numel())
        meta[f"{split}_docs"] = int(offs.numel())
        # invariants: offsets sorted+unique, first is 0, every doc ends with EOT
        assert offs[0].item() == 0 and (offs[1:] > offs[:-1]).all()
        assert (toks[offs[1:] - 1] == EOT).all() and toks[-1].item() == EOT
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}: " + ", ".join(f"{s}={splits[s][0].numel():,}tok/{splits[s][1].numel():,}docs"
                                            for s in splits), flush=True)


def build_openr1(tok, docs, data_dir, verify_report):
    print("[build openr1] per-doc tokenization + EOT...", flush=True)
    toks, offs, lens = tokenize_docs_with_eot(tok, docs)
    n_total = toks.numel()
    split_doc = split_by_token_mass(offs, n_total, OPENR1_TRAIN_TOKENS / (OPENR1_TRAIN_TOKENS + OPENR1_VAL_TOKENS))
    cut = int(offs[split_doc].item())
    train_toks, val_toks = toks[:cut].clone(), toks[cut:].clone()
    train_offs = offs[:split_doc].clone()
    val_offs = (offs[split_doc:] - cut).clone()
    meta = {
        "vocab_size": 50257, "tokenizer": "gpt2", "eot_separated": True, "eot_token_id": EOT,
        "source": "open-r1/OpenR1-Math-220k", "hf_config": "default", "n_examples": len(docs),
        "recipe": ("doc = problem + '\\n' + solution (dataset order); one EOT (50256) appended per "
                    "doc. The ORIGINAL stream's per-doc trailing '\\n' (a concatenation artifact, "
                    "see original_recipe_verification) is REPLACED by the EOT separator, not kept "
                    "in addition to it."),
        "split": (f"document-aligned 95/5 by token mass at doc {split_doc} -- DISCLOSED DEVIATION "
                   f"from the original flat mid-document token slice at {OPENR1_TRAIN_TOKENS:,} "
                   f"(the original val.pt starts mid-document; a document-aligned split has no "
                   f"partial doc in both splits)"),
        "rebuild": "AUDIT FIX-3, 2026-07-03 (rebuild_lm_corpora_rd.py)",
        "original_recipe_verification": verify_report,
    }
    write_corpus(os.path.join(data_dir, "reasoning_eot"),
                 {"train": (train_toks, train_offs), "val": (val_toks, val_offs)}, meta)


def build_wikitext(tok, data_dir, verify_report):
    name_map = {"train": "train", "validation": "val", "test": "test"}
    splits = {}
    n_docs_by_split = {}
    for hf_split, out_split in name_map.items():
        docs = wikitext_docs(hf_split)
        n_docs_by_split[out_split] = len(docs)
        print(f"[build wikitext/{hf_split}] {len(docs):,} articles, per-doc tokenization + EOT...",
              flush=True)
        toks, offs, _ = tokenize_docs_with_eot(tok, docs)
        splits[out_split] = (toks, offs)
    meta = {
        "vocab_size": 50257, "tokenizer": "gpt2", "eot_separated": True, "eot_token_id": EOT,
        "source": "Salesforce/wikitext", "hf_config": "wikitext-103-raw-v1",
        "recipe": ("doc = one article = ''.join(rows) segmented at top-level heading rows "
                    "(' = Title = ', not ' = = '); one EOT (50256) appended per doc"),
        "split": "official HF train/validation/test splits (already document-aligned)",
        "rebuild": "AUDIT FIX-3, 2026-07-03 (rebuild_lm_corpora_rd.py)",
        "original_recipe_verification": verify_report,
    }
    write_corpus(os.path.join(data_dir, "wikitext103_eot"), splits, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--only", choices=["openr1", "wikitext", "both"], default="both",
                     help="rebuild a single corpus (e.g. after a corpus-specific fix, without "
                          "redoing the other's already-verified build)")
    ap.add_argument("--skip-verify", action="store_true",
                     help="skip the bit-exact recipe verification against the original tensors "
                          "(NOT recommended -- the verification IS the evidence the rebuild "
                          "preserves the original corpora's content)")
    args = ap.parse_args()

    tok = load_tokenizer()

    if args.only in ("openr1", "both"):
        print("=" * 70 + "\n  OPENR1 (reasoning_eot)\n" + "=" * 70, flush=True)
        docs = openr1_docs()
        v_openr1 = {"skipped": True} if args.skip_verify else verify_openr1(tok, docs, args.data_dir)
        build_openr1(tok, docs, args.data_dir, v_openr1)
        del docs

    if args.only in ("wikitext", "both"):
        print("=" * 70 + "\n  WIKITEXT-103 (wikitext103_eot)\n" + "=" * 70, flush=True)
        v_wiki = {"skipped": True} if args.skip_verify else verify_wikitext(tok, args.data_dir)
        build_wikitext(tok, args.data_dir, v_wiki)

    print("\nREBUILD COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
