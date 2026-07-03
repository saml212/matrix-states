"""build_mix_corpora_rd.py -- SCALE_TRANSFER_DESIGN.md sec 5.4 (Track C's
required augmented-data pipeline): builds the two domain-blended "mix"
training corpora -- reasoning_mix_eot (openr1_eot's train split + an
OpenWebMath subset) and wikitext103_mix_eot (wikitext103_eot's train split
+ a FineWeb-Edu subset) -- consumed by BOTH (a) the REQUIRED rung-1
small-model same-mix control cell (sec 5.6 Wave 1, MAJOR-5 -- the Wave-C-
scale 13-14M architecture retrained on these SAME mixes, isolating the
data-mix axis from the scale axis per CLAUDE.md's hold-the-second-axis-
fixed rule) and (b) the actual rung-1 large-model training runs (gated
behind Wave -1's own calibration + a separately-authorized launch -- this
script does NOT train anything).

Reuses rebuild_lm_corpora_rd.py's EOT-tokenization / write_corpus /
load_tokenizer machinery DIRECTLY (same-directory import, matching
lm_pretrain_rd.py's own model_rd.py/rank_utils.py convention -- utility
code is single-sourced; only orchestrator/sweep scripts get copy-cloned
per this codebase's stated pod-safety rule).

License / availability RE-verification (sec 5.6 Wave -1 row, MINOR-5:
"re-verify OpenWebMath/FineWeb-Edu licenses + HF availability at build
time ... the check is re-run, not inherited"): verify_dataset_license()
fetches EACH dataset's OWN README.md YAML front matter license field LIVE
(not the design doc's cached WebSearch pass) and asserts it still reads
odc-by before any tokenization begins. Re-verified this build session
(2026-07-04): both open-web-math/open-web-math and HuggingFaceFW/fineweb-edu
README.md front matter read `license: odc-by`.

Composition rule -- sec 5.4's epoch discipline ("cap any single source's
repetition at <=5 physical epochs ... the remainder is drawn from
OpenWebMath/FineWeb-Edu"), made concrete: lm_pretrain_rd.py's get_batch
samples window START positions UNIFORMLY AT RANDOM over the whole flat
corpus tensor every training step (unmodified by this script). For a mix
of total size M tokens trained for a total budget of B tokens, EVERY
source in the mix (base-corpus material AND augmentation material alike)
is seen in EXPECTATION B/M times -- so "the base corpus is repeated <=5x"
is exactly the constraint M >= B/5. Each built mix's meta.json records the
resulting planning ceiling (B <= 5*M) explicitly; if Wave -1's real rung-1
token budget (post-calibration, sec 5.6) ends up larger, pull more
augmentation before a real Wave 1 launch -- a documented follow-on, not a
silent change to an already-built mix.

Val/test splits are NOT augmented: they stay the base corpus's own
original held-out windows unchanged, so comparisons against the already-
archived Wave C val numbers remain on IDENTICAL windows; only TRAIN is
domain-blended (documented in every mix's own meta.json).

Usage (run ON THE BOX, needs the HF cache + /data + network):
  HF_HOME=/data/hf_cache python build_mix_corpora_rd.py --smoke
  HF_HOME=/data/hf_cache python build_mix_corpora_rd.py \\
      --data-dir /data/deltanet_rd_data --target-augment-tokens 150000000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
from rebuild_lm_corpora_rd import EOT, load_tokenizer, tokenize_docs_with_eot, write_corpus

# ---------------------------------------------------------------------------
# License / availability re-verification (sec 5.6 Wave -1, MINOR-5)
# ---------------------------------------------------------------------------

_AUGMENT_SOURCES = {
    "openr1": {"repo_id": "open-web-math/open-web-math", "config": None,
               "base_corpus_dir": "reasoning_eot", "out_dir": "reasoning_mix_eot"},
    "wikitext": {"repo_id": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
                 "base_corpus_dir": "wikitext103_eot", "out_dir": "wikitext103_mix_eot"},
}


def _parse_license_from_readme(text: str, expected: str) -> tuple[str | None, bool]:
    """Pure regex parse of a HF dataset README.md's YAML front-matter
    `license:` field -- factored out from verify_dataset_license so smoke()
    can unit-test the PARSER without a network call.

    Restricted to the YAML front-matter block (text between the first two
    `---` delimiter lines) so a stray "license:"-looking string in the
    README's BODY prose can never be picked up. Allows LEADING WHITESPACE
    before `license:` -- found live, this build session, that this matters:
    FineWeb-Edu's license key sits at the TOP LEVEL of the front matter,
    but OpenWebMath's sits NESTED one level under `dataset_info:` (both
    verified by direct curl of each README.md, 2026-07-04) -- an
    anchor-at-column-0 regex silently misses the second, indented case."""
    fm = re.search(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL | re.MULTILINE)
    block = fm.group(1) if fm else text
    m = re.search(r"^[ \t]*license:\s*(\S+)", block, re.MULTILINE)
    tag = m.group(1).strip() if m else None
    ok = tag is not None and expected.lower() in tag.lower()
    return tag, ok


def verify_dataset_license(repo_id: str, expected: str = "odc-by", timeout: float = 30.0) -> dict:
    """Live re-fetch of repo_id's README.md (HF raw file endpoint, no auth
    needed for a public dataset) and assert its front-matter license tag
    still matches `expected`. This is the build-time re-verification sec
    5.6's MINOR-5 row requires -- NOT a re-statement of sec 5.4's own
    cached WebSearch pass."""
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/README.md"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    tag, ok = _parse_license_from_readme(text, expected)
    assert ok, (
        f"{repo_id}: README.md front-matter license={tag!r} does not match expected {expected!r} "
        f"-- sec 5.6's 're-verify, not inherited' requirement FAILED. Do not proceed with "
        f"augmentation from this source; investigate (dataset card may have changed) before "
        f"overriding.")
    return {"repo_id": repo_id, "license_tag": tag, "expected": expected,
            "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_url": url}


# ---------------------------------------------------------------------------
# Streaming augmentation tokenization
# ---------------------------------------------------------------------------

def _contains_literal_eot(text: str, eot_string: str = "<|endoftext|>") -> bool:
    """Cheap pre-tokenization filter: True if `text` contains the LITERAL EOT marker as content.
    Factored out for smoke-testability (no network/tokenizer needed) -- see stream_augment_corpus's
    inline comment for why this filter exists (found live, this build session)."""
    return eot_string in text


def stream_augment_corpus(repo_id: str, config_name: str | None, target_tokens: int, tok,
                           batch_docs: int = 1000, shuffle_seed: int = 0,
                           shuffle_buffer: int = 10_000, log_every_docs: int = 20_000) -> tuple:
    """Streams `repo_id` (HF streaming mode -- never downloads the full
    dataset; sec 5.4's own "streamable on the existing box" claim, re-
    verified this session), shuffles with a buffer (avoids drawing a
    contiguous, possibly non-representative run of the source's own
    document order into the mix), and tokenizes with EOT separators via
    rebuild_lm_corpora_rd.tokenize_docs_with_eot (REUSED verbatim -- same
    per-doc-EOT-duplicate assertion the base corpora were built with).
    Stops once >= target_tokens tokens have been accumulated (the batch
    that crosses the threshold is kept whole -- the achieved count is
    documented as `n_tokens` in the returned report, not silently rounded
    to the target).

    Returns (tokens: int64 (N,), doc_offsets: int64 (D,) [0-based, LOCAL to
    this stream], report: dict)."""
    from datasets import load_dataset
    t0 = time.time()
    ds = (load_dataset(repo_id, name=config_name, split="train", streaming=True) if config_name
          else load_dataset(repo_id, split="train", streaming=True))
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)

    id_chunks, offset_chunks, pos, n_docs, n_skipped_empty = [], [], 0, 0, 0
    n_skipped_eot_literal = 0
    batch_texts = []
    exhausted = False
    it = iter(ds)

    def _flush(texts):
        nonlocal pos, n_docs
        ids_t, offs_t, _lens = tokenize_docs_with_eot(tok, texts, batch_size=len(texts))
        id_chunks.append(ids_t)
        offset_chunks.append(offs_t + pos)
        pos += int(ids_t.numel())
        n_docs += int(offs_t.numel())

    while pos < target_tokens:
        try:
            row = next(it)
        except StopIteration:
            exhausted = True
            break
        text = row.get("text")
        if not text or not text.strip():
            n_skipped_empty += 1
            continue
        # tokenize_docs_with_eot HARD-ASSERTS no doc's own tokenization contains the literal EOT id
        # (50256, "<|endoftext|>") -- the EOT<=>boundary invariant every corpus in this harness
        # depends on. OpenR1/WikiText (curated sources) never hit this; real web-scraped text
        # (OpenWebMath/FineWeb-Edu) occasionally DOES contain the literal "<|endoftext|>" string as
        # CONTENT (found live this build session, crashed the first real run at doc ~942) --
        # filtered here at the SOURCE (cheap string check, before tokenizing) rather than patching
        # rebuild_lm_corpora_rd.py's own already-audited assert.
        if _contains_literal_eot(text):
            n_skipped_eot_literal += 1
            continue
        batch_texts.append(text)
        if len(batch_texts) >= batch_docs:
            _flush(batch_texts)
            batch_texts = []
            if n_docs % log_every_docs < batch_docs:
                rate = pos / max(1e-6, time.time() - t0)
                print(f"  [{repo_id}] {pos:,}/{target_tokens:,} tok, {n_docs:,} docs, "
                      f"{rate:,.0f} tok/s, {time.time() - t0:.0f}s elapsed", flush=True)
    if batch_texts:
        _flush(batch_texts)

    toks = torch.cat(id_chunks) if id_chunks else torch.zeros(0, dtype=torch.int64)
    offs = torch.cat(offset_chunks) if offset_chunks else torch.zeros(0, dtype=torch.int64)
    report = {
        "repo_id": repo_id, "config": config_name, "target_tokens": target_tokens,
        "n_tokens": int(toks.numel()), "n_docs": int(offs.numel()),
        "n_skipped_empty_docs": n_skipped_empty,
        "n_skipped_eot_literal_docs": n_skipped_eot_literal,
        "stream_exhausted_before_target": exhausted,
        "wall_s": time.time() - t0, "shuffle_seed": shuffle_seed, "shuffle_buffer": shuffle_buffer,
    }
    print(f"  [{repo_id}] DONE: {report['n_tokens']:,} tokens / {report['n_docs']:,} docs "
          f"(target {target_tokens:,}, exhausted={exhausted}) in {report['wall_s']:.0f}s", flush=True)
    return toks, offs, report


# ---------------------------------------------------------------------------
# Mix construction
# ---------------------------------------------------------------------------

def build_mix(corpus_key: str, data_dir: str, target_augment_tokens: int, tok) -> dict:
    src = _AUGMENT_SOURCES[corpus_key]
    base_dir = os.path.join(data_dir, src["base_corpus_dir"])
    with open(os.path.join(base_dir, "meta.json")) as f:
        base_meta = json.load(f)
    assert base_meta.get("eot_separated") is True, (
        f"{corpus_key}: base corpus {base_dir} is not the EOT-separated rebuild -- run "
        f"rebuild_lm_corpora_rd.py first.")

    lic = verify_dataset_license(src["repo_id"])
    print(f"[{corpus_key}] license re-verified: {src['repo_id']} -> {lic['license_tag']}", flush=True)

    base_train = torch.load(os.path.join(base_dir, "train.pt"), map_location="cpu")
    base_offs = torch.load(os.path.join(base_dir, "train_doc_offsets.pt"), map_location="cpu")
    assert base_train.dtype == torch.int64 and base_offs.dtype == torch.int64

    aug_toks, aug_offs, aug_report = stream_augment_corpus(
        src["repo_id"], src["config"], target_augment_tokens, tok)
    aug_report["license_verification"] = lic

    mix_toks = torch.cat([base_train, aug_toks])
    mix_offs = torch.cat([base_offs, aug_offs + base_train.numel()])
    M = int(mix_toks.numel())

    splits = {"train": (mix_toks, mix_offs)}
    # val (and test, if the base corpus has one) pass through UNCHANGED -- no augmentation in eval.
    for split in ("val", "test"):
        p_toks = os.path.join(base_dir, f"{split}.pt")
        if not os.path.exists(p_toks):
            continue
        splits[split] = (torch.load(p_toks, map_location="cpu"),
                          torch.load(os.path.join(base_dir, f"{split}_doc_offsets.pt"), map_location="cpu"))

    meta = {
        "vocab_size": 50257, "tokenizer": "gpt2", "eot_separated": True, "eot_token_id": EOT,
        "source": (f"mix: base={base_meta.get('source')} (train only, {int(base_train.numel()):,} tok, "
                   f"{int(base_offs.numel()):,} docs) + augment={src['repo_id']}"
                   + (f"[{src['config']}]" if src["config"] else "")
                   + f" (streamed train subset, {aug_report['n_tokens']:,} tok, "
                     f"{aug_report['n_docs']:,} docs)"),
        "mix_composition": {
            "base_corpus_dir": src["base_corpus_dir"], "base_train_tokens": int(base_train.numel()),
            "base_train_docs": int(base_offs.numel()), "augment": aug_report,
            "mix_total_train_tokens": M,
        },
        "epoch_cap_discipline": (
            "get_batch (lm_pretrain_rd.py) samples window start positions UNIFORMLY AT RANDOM over "
            "the whole flat mix tensor every training step, unmodified by this build -- for a "
            "training budget of B tokens over this mix (size M), EVERY source (base + augment) is "
            "seen in EXPECTATION B/M times. sec 5.4's <=5-physical-epoch cap on the base corpus is "
            f"therefore exactly the constraint M>=B/5: this mix (M={M:,} train tokens) supports "
            f"planning budgets up to B<={5 * M:,} tokens before the base corpus would be repeated "
            f"more than 5x in expectation. If a real rung's Wave -1-calibrated budget exceeds this, "
            f"pull more augmentation before that rung's Wave 1 launch (documented follow-on)."
        ),
        "val_test_note": ("val/test are the BASE corpus's own original held-out windows, UNCHANGED "
                           "and un-augmented -- comparisons against archived Wave C val numbers stay "
                           "on identical windows; only TRAIN is domain-blended."),
        "split": "train=mix (base train + augment subset, concatenated, base material first, "
                 "shuffled at read time by get_batch's random-window sampler); val/test=base "
                 "corpus's own splits, unchanged",
        "rebuild": "SCALE_TRANSFER_DESIGN.md sec 5.4 (build_mix_corpora_rd.py)",
    }
    write_corpus(os.path.join(data_dir, src["out_dir"]), splits, meta)
    return meta


# ---------------------------------------------------------------------------
# Smoke gate -- NO network (a hand-built in-memory "stream" stands in for
# load_dataset's streaming iterator), fast, deterministic.
# ---------------------------------------------------------------------------

def smoke():
    print("=" * 60 + "\n  BUILD_MIX_CORPORA_RD SMOKE GATE\n" + "=" * 60)

    print("\n[1] license-parser unit test (regex against hand-built YAML front matter, no network)")
    good_readme = "---\nlicense: odc-by\npretty_name: Foo\n---\nbody text\n"
    tag, ok = _parse_license_from_readme(good_readme, "odc-by")
    assert ok and tag == "odc-by", f"expected odc-by/True, got {tag!r}/{ok}"
    bad_readme = "---\nlicense: cc-by-nc-4.0\n---\n"
    tag2, ok2 = _parse_license_from_readme(bad_readme, "odc-by")
    assert not ok2, "negative control FAILED: cc-by-nc-4.0 was accepted as odc-by"
    missing_readme = "---\npretty_name: Foo\n---\n"
    tag3, ok3 = _parse_license_from_readme(missing_readme, "odc-by")
    assert tag3 is None and not ok3, "missing-license-field case should report tag=None, ok=False"
    # REGRESSION (found live this build session, 2026-07-04): OpenWebMath's README nests
    # `license:` one level under `dataset_info:` rather than at the front-matter top level (unlike
    # FineWeb-Edu's) -- a column-0-anchored regex silently missed it (verify_dataset_license
    # FAILED against the real README before this fix). Exercise the nested case explicitly so this
    # can't regress silently again.
    nested_readme = ("---\ndataset_info:\n  features:\n    - name: text\n      dtype: string\n"
                      "  license: odc-by\n  pretty_name: Bar\n---\nbody\n")
    tag4, ok4 = _parse_license_from_readme(nested_readme, "odc-by")
    assert ok4 and tag4 == "odc-by", f"NESTED license field (OpenWebMath's actual layout) not parsed: got {tag4!r}/{ok4}"
    # a "license:"-looking string in the BODY (outside the front-matter block) must NEVER match
    body_trap_readme = "---\npretty_name: Foo\n---\nSee license: MIT in some unrelated footnote.\n"
    tag5, ok5 = _parse_license_from_readme(body_trap_readme, "odc-by")
    assert tag5 is None and not ok5, "body-text 'license:' string was incorrectly picked up outside the front-matter block"
    print(f"  odc-by README -> tag={tag!r} ok={ok}; cc-by-nc-4.0 README -> ok={ok2} (correctly "
          f"rejected); missing-field README -> tag={tag3!r} ok={ok3} (correctly rejected); "
          f"NESTED odc-by (OpenWebMath's real layout) -> tag={tag4!r} ok={ok4}; body-text trap -> "
          f"ok={ok5} (correctly rejected)")

    print("\n[1b] REGRESSION (found live this build session, 2026-07-04): a document containing the "
          "LITERAL '<|endoftext|>' string as CONTENT (real, found in OpenWebMath -- crashed the "
          "first real build at doc ~942 via tokenize_docs_with_eot's own EOT-duplicate assert) must "
          "be flagged by the pre-tokenization filter; ordinary text must NOT be")
    assert _contains_literal_eot("some text mentioning <|endoftext|> mid-document") is True
    assert _contains_literal_eot("ordinary text with no special tokens") is False
    assert _contains_literal_eot("") is False
    print("  literal-EOT-containing text -> flagged; ordinary text -> not flagged; empty -> not flagged")

    print("\n[2] tokenize_docs_with_eot reuse + mix concatenation offset arithmetic, on tiny "
          "hand-built 'base' + 'augment' token streams (no tokenizer/network needed -- EOT-id "
          "presence is the only invariant tokenize_docs_with_eot itself enforces per-doc, so this "
          "item exercises the SAME reuse path with a trivial identity 'tokenizer')")

    class _IdentityTok:
        """Maps each whitespace-split 'word' to its own int id (offset above EOT) -- avoids a real
        HF tokenizer dependency in the network-free smoke path while still exercising
        tokenize_docs_with_eot's real per-doc EOT-duplicate assertion and batching."""

        def __call__(self, texts, add_special_tokens=False):
            out = []
            for t in texts:
                out.append([EOT + 1 + i for i in range(len(t.split()))])
            return {"input_ids": out}

    tok = _IdentityTok()
    base_docs = ["alpha beta", "gamma delta epsilon"]
    base_toks, base_offs, _ = tokenize_docs_with_eot(tok, base_docs)
    aug_docs = ["zeta", "eta theta iota kappa"]
    aug_toks, aug_offs, _ = tokenize_docs_with_eot(tok, aug_docs)

    mix_toks = torch.cat([base_toks, aug_toks])
    mix_offs = torch.cat([base_offs, aug_offs + base_toks.numel()])
    assert mix_offs[0].item() == 0 and (mix_offs[1:] > mix_offs[:-1]).all(), \
        "mix doc_offsets not strictly increasing from 0"
    assert (mix_toks[mix_offs[1:] - 1] == EOT).all() and mix_toks[-1].item() == EOT, \
        "mix EOT-boundary invariant broken (write_corpus's own assert would also catch this)"
    assert mix_offs.numel() == base_offs.numel() + aug_offs.numel() == 4
    # the augment portion's FIRST doc must start exactly at base_toks.numel() (no overlap, no gap)
    assert mix_offs[base_offs.numel()].item() == base_toks.numel(), \
        "augment doc_offsets were not shifted correctly onto the mix's coordinate system"
    print(f"  mix: {mix_toks.numel()} tok / {mix_offs.numel()} docs "
          f"(base {base_toks.numel()}/{base_offs.numel()} + augment {aug_toks.numel()}/{aug_offs.numel()}), "
          f"offsets strictly increasing, EOT-boundary invariant holds, augment correctly shifted")

    print("\n[3] epoch-cap arithmetic sanity: M>=B/5 <=> B<=5M, on a hand-computed case")
    M_test = 1000
    B_max = 5 * M_test
    assert B_max / M_test == 5.0
    print(f"  M={M_test} -> planning ceiling B<={B_max} (B/M=5.0 exactly at the boundary)")

    print("\n[4] _AUGMENT_SOURCES registry sanity: both corpora map to distinct base dirs / augment "
          "repos / out dirs (a copy-paste bug here would silently mix the wrong sources)")
    keys = list(_AUGMENT_SOURCES)
    assert set(keys) == {"openr1", "wikitext"}
    base_dirs = [_AUGMENT_SOURCES[k]["base_corpus_dir"] for k in keys]
    out_dirs = [_AUGMENT_SOURCES[k]["out_dir"] for k in keys]
    repos = [_AUGMENT_SOURCES[k]["repo_id"] for k in keys]
    assert len(set(base_dirs)) == len(set(out_dirs)) == len(set(repos)) == 2, \
        "registry has a duplicate base_corpus_dir/out_dir/repo_id across corpora"
    print(f"  registry: {json.dumps(_AUGMENT_SOURCES, indent=2)}")

    print("\n" + "=" * 60 + "\n  ALL BUILD_MIX_CORPORA_RD SMOKE CHECKS PASSED\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--only", choices=["openr1", "wikitext", "both"], default="both",
                     help="build a single mix (e.g. after adjusting one source's target), without "
                          "redoing the other's already-built mix")
    ap.add_argument("--target-augment-tokens", type=int, default=150_000_000,
                     help="per-corpus augmentation token budget (streamed from OpenWebMath / "
                          "FineWeb-Edu respectively). Default 150M -- see module docstring's "
                          "epoch-cap-discipline note for the resulting planning ceiling.")
    ap.add_argument("--test-stream", type=int, default=None, metavar="N_TOKENS",
                     help="REAL (network) timing/schema probe: streams N_TOKENS from each "
                          "--only source via stream_augment_corpus directly (skips base-corpus "
                          "loading/mixing entirely -- no --data-dir requirement) and reports "
                          "throughput. Use before committing to a full --target-augment-tokens "
                          "build (calibration-first discipline).")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    tok = load_tokenizer()

    if args.test_stream is not None:
        keys = ["openr1", "wikitext"] if args.only == "both" else [args.only]
        for k in keys:
            src = _AUGMENT_SOURCES[k]
            lic = verify_dataset_license(src["repo_id"])
            print(f"[{k}] license re-verified: {src['repo_id']} -> {lic['license_tag']}", flush=True)
            toks, offs, report = stream_augment_corpus(src["repo_id"], src["config"], args.test_stream, tok)
            rate = report["n_tokens"] / max(1e-6, report["wall_s"])
            print(f"[{k}] PROBE RESULT: {json.dumps(report, indent=2)}\n  -> {rate:,.0f} tok/s "
                  f"measured; a {150_000_000:,}-token build would take ~{150_000_000 / max(1, rate) / 60:.1f} min",
                  flush=True)
        return
    keys = ["openr1", "wikitext"] if args.only == "both" else [args.only]
    reports = {}
    for k in keys:
        print("=" * 70 + f"\n  MIX: {k}\n" + "=" * 70, flush=True)
        reports[k] = build_mix(k, args.data_dir, args.target_augment_tokens, tok)

    print("\nBUILD_MIX_CORPORA_RD COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
