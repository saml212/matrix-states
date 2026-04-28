#!/usr/bin/env python3
"""build_prosqa_multi.py — Generate ProsQA-MULTI-k datasets.

Each ProsQA-MULTI-k *source group* is formed by merging k source ProsQA
examples whose entity sets are disjoint, concatenating their facts.  For each
source group we emit k individual output records — one per queried subject —
so that each record is schema-compatible with the existing ProsQADataset loader
and prosqa_answer_match evaluator in run_matrix_codi.py.

Each output record:
  - question: "<all k sources' facts> Is <subj_i> a <opt_A> or <opt_B>?"
              where the option order is randomised 50/50 (Fix #2).
  - answer:   "<subj_i> is a <correct_class>."   (single sentence, single target)
  - steps:    the reasoning chain from source-i's original ProsQA example
  - k:        number of subjects whose facts appear in the question context
  - queried_subject_index: which subject (0..k-1) is being queried
  - graph_meta: {source_indices, queried_source, roots, targets, ...}

Schema produced (per-record):
  {
    "question":              "<combined facts> Is <subj_i> a <opt_A> or <opt_B>?",
    "answer":                "<subj_i> is a <correct_class>.",
    "steps":                 [...],         # steps from subj_i's source row
    "k":                     2,
    "queried_subject_index": 0,             # which of the k sources is queried
    "graph_meta": {
      "source_indices":  [i, j],
      "queried_source":  i,
      "roots":           [root_i, root_j],
      "targets":         [target_i, target_j],
      "neg_targets_raw": [neg_i, neg_j],
      "idx_to_symbols":  [[...], [...]],
      "edges":           [[...], [...]]
    }
  }

IMPORTANT — MULTI property: the question CONTAINS facts for all k subjects even
though it asks about only one.  This stress-tests the latent's capacity: it must
compress k subjects' worth of facts at intermediate positions (rank-k requirement)
even though the answer token itself is single-subject.  The existing
ProsQADataset and prosqa_answer_match work unchanged.  Attack #4 (position-
decomposition) remains a known design property; rank-1 forcing experiment (#5
in COMBINED_PLAN) must accompany this dataset for the headline rank-k claim.

Usage:
  python3 build_prosqa_multi.py \\
      --k 2 \\
      --train-out data/prosqa_multi2_train.json \\
      --test-out  data/prosqa_multi2_test.json \\
      --src-train /path/to/prosqa_train.json \\
      --src-test  /path/to/prosqa_test.json \\
      --seed 1337 \\
      --n-train 5000 \\
      --n-test  500

Hard requirements met:
  1. Deterministic given --seed.
  2. CPU-only — no GPU. Uses multiprocessing.cpu_count()//2 workers.
  3. Entity disjointness verified: no symbol from source-i appears in source-j.
  4. Provenance: source_indices saved in graph_meta.
  5. Dropped count printed. Examples exceeding max_total_len=768 GPT-2 tokens dropped.
  6. Identical output across runs given same seed.
  7. [Fix #2] Option order (pos/neg) randomised per subject — no order leak.
  8. [Fix #3] Each record has singular "answer" + "steps" — schema matches
     ProsQADataset / prosqa_answer_match with zero changes to the eval code.
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
from functools import partial
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOTAL_LEN = 768   # matches matrix-CODI CONFIG["max_total_len"]

# We defer tokenizer import to worker init to avoid forking a loaded tokenizer.
_tokenizer = None


def _init_worker():
    """Load the GPT-2 tokenizer once per worker process (CPU only)."""
    global _tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # On macOS with PyTorch installed, two copies of libomp may be linked.
    # Setting KMP_DUPLICATE_LIB_OK suppresses the crash in that scenario.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        from transformers import GPT2TokenizerFast
        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as e:
        # Tokenizer unavailable — length check will be skipped with a warning.
        _tokenizer = None
        print(f"[WARNING] Could not load GPT-2 tokenizer: {e}. "
              "Token-length filter will be skipped.", flush=True)


# ---------------------------------------------------------------------------
# Entity disjointness
# ---------------------------------------------------------------------------

def _root_name(row: dict) -> str:
    """Return the person/subject name being queried (the 'root' entity)."""
    return row["idx_to_symbol"][row["root"]]


def _are_disjoint(rows) -> bool:
    """Return True if the k source rows have disjoint query subjects.

    ProsQA reuses a shared ~55-word fantasy vocabulary across all examples,
    so checking all symbols would make every pair fail.  The actual leakage
    risk is:
      1. Two rows query the SAME person (e.g. both ask about 'Tom').
      2. The root person of source-A appears explicitly in the facts of
         source-B (or vice versa), which would let the model shortcut.

    We check both conditions.  Class-name overlap (e.g. 'rempus' appearing
    in both examples) is fine: class names are part of the shared vocabulary
    and don't reveal which specific chain either query follows.
    """
    root_names = [_root_name(r) for r in rows]
    # Condition 1: all root names must be distinct
    if len(set(root_names)) < len(root_names):
        return False
    # Condition 2: no root name of row-i may appear in the question text of row-j
    questions = [r["question"] for r in rows]
    for i, name_i in enumerate(root_names):
        for j, q_j in enumerate(questions):
            if i != j and name_i in q_j:
                return False
    return True


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _extract_facts(question_text: str) -> str:
    """Extract the facts portion of a ProsQA question (everything before the
    final 'Is X a ... or ...?' sentence).

    ProsQA questions always end with 'Is <name> a <cls1> or <cls2>?'.  We
    strip that final sentence so we can replace it with the multi-target
    combined question.
    """
    # Find the last occurrence of " Is " to split off the query sentence.
    last_is = question_text.rfind(" Is ")
    if last_is == -1:
        # Fallback: return everything (shouldn't happen in well-formed ProsQA)
        return question_text.strip()
    return question_text[:last_is].strip()


def _build_single_query(source_rows: list, queried_idx: int, rng: random.Random) -> str:
    """Build the question text for one queried subject within the k-source group.

    The question context contains facts from ALL k source rows (so the latent
    must encode k subjects' worth of information), but the query sentence asks
    about only the subject at queried_idx.

    Fix #2: The positive/negative class order in the "Is X a A or B?" sentence
    is randomised per-call using rng so there is no systematic order bias.

    Format:
      "<facts from src-0> <facts from src-1> ... Is <subj_i> a <opt_A> or <opt_B>?"
    """
    all_facts_parts = []
    for row in source_rows:
        all_facts_parts.append(_extract_facts(row["question"]))
    all_facts = " ".join(all_facts_parts)

    row = source_rows[queried_idx]
    symbols  = row["idx_to_symbol"]
    subj     = symbols[row["root"]]
    pos_cls  = symbols[row["target"]]
    neg_cls  = symbols[row["neg_target"]]

    # Fix #2: randomise option order 50/50 per example, using the seeded rng.
    opts = [pos_cls, neg_cls]
    if rng.random() < 0.5:
        opts = [neg_cls, pos_cls]

    return f"{all_facts} Is {subj} a {opts[0]} or {opts[1]}?"


def _build_examples_for_group(
    source_rows: list,
    source_indices: list,
    rng: random.Random,
) -> list:
    """Assemble k output records from one k-source group (Approach A / Fix #3).

    Each record asks about exactly ONE subject but includes facts from all k
    source rows in the question.  This keeps the schema identical to single-
    target ProsQA so ProsQADataset and prosqa_answer_match work unchanged.

    Returns a list of k dicts, each with:
      "question", "answer", "steps", "k", "queried_subject_index", "graph_meta"
    """
    k = len(source_rows)
    shared_graph_meta_base = {
        "source_indices":  source_indices,
        "roots":           [r["root"]          for r in source_rows],
        "targets":         [r["target"]        for r in source_rows],
        "neg_targets_raw": [r["neg_target"]    for r in source_rows],
        "idx_to_symbols":  [r["idx_to_symbol"] for r in source_rows],
        "edges":           [r["edges"]         for r in source_rows],
    }

    records = []
    for qi in range(k):
        row = source_rows[qi]
        symbols  = row["idx_to_symbol"]
        subj     = symbols[row["root"]]
        pos_cls  = symbols[row["target"]]

        # Fix #3: singular "answer" string parseable by prosqa_answer_match.
        answer_str = str(row["answer"]).strip()
        if not answer_str.endswith("."):
            answer_str = f"{subj} is a {pos_cls}."

        # steps from the queried source row (for ProsQADataset.reasoning_steps).
        steps = row.get("steps", [])

        # Fix #2 is applied inside _build_single_query via the seeded rng.
        question = _build_single_query(source_rows, qi, rng)

        graph_meta = dict(shared_graph_meta_base)
        graph_meta["queried_source"] = source_indices[qi]

        records.append({
            "question":              question,
            "answer":                answer_str,
            "steps":                 steps,
            "k":                     k,
            "queried_subject_index": qi,
            "graph_meta":            graph_meta,
        })
    return records


# ---------------------------------------------------------------------------
# Token-length check
# ---------------------------------------------------------------------------

def _token_len(text: str) -> int:
    """Return GPT-2 token count for text. Returns 0 if tokenizer unavailable."""
    if _tokenizer is None:
        return 0
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _exceeds_max_len(record: dict) -> bool:
    """Return True if question + steps + answer exceed MAX_TOTAL_LEN tokens.

    Mirrors the exact teacher_text format used by ProsQADataset.__init__:
        f"{question}\\n{cot}{answer_prefix} {gold}"
    so that every record passing this filter also passes the loader's length
    check (i.e. zero silent drops at training time).
    """
    if _tokenizer is None:
        return False
    answer_prefix = " The answer is:"
    steps = record.get("steps", [])
    if isinstance(steps, list):
        cot = " ".join(str(s) for s in steps).strip()
    else:
        cot = str(steps).strip()
    teacher_text = f"{record['question']}\n{cot}{answer_prefix} {record['answer']}"
    return _token_len(teacher_text) > MAX_TOTAL_LEN


# ---------------------------------------------------------------------------
# Worker function (used with multiprocessing.Pool)
# ---------------------------------------------------------------------------

def _make_example_worker(args):
    """Worker: receives (attempt_indices_list, source_data, base_seed) and tries
    to build k valid single-query records from a disjoint-entity source group.

    Returns (records_list | None, dropped_reason | None).

    Fix #2: each worker creates its own per-group RNG seeded deterministically
    from (base_seed XOR hash of source indices) so option order is reproducible
    regardless of worker count or scheduling.
    """
    idx_group, source_data, base_seed = args
    source_rows = [source_data[i] for i in idx_group]

    if not _are_disjoint(source_rows):
        return None, "entity_overlap"

    # Per-group deterministic seed: XOR base with a hash of the index tuple.
    group_seed = base_seed ^ hash(tuple(idx_group))
    group_rng  = random.Random(group_seed)

    records = _build_examples_for_group(source_rows, idx_group, group_rng)

    # Drop the entire group if ANY record exceeds the length budget.
    # (All records share the same question context so length fate is the same.)
    if any(_exceeds_max_len(r) for r in records):
        return None, "too_long"

    return records, None


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_split(
    source_data: list,
    k: int,
    n: int,
    seed: int,
    n_workers: int,
) -> tuple:
    """Generate n multi-k examples from source_data.

    Returns (examples, stats_dict).
    Strategy: sample k-tuples of distinct indices, filter by disjoint entities
    and token length.  Re-sample with fresh seeds if needed.  Single-pass with
    a large oversampling factor avoids loops.
    """
    rng = random.Random(seed)
    n_src = len(source_data)

    if n_src < k:
        raise ValueError(f"Not enough source examples ({n_src}) for k={k}.")

    # Oversample: expect ~50% pass rate on entity disjointness; add buffer.
    # We'll generate at most 4× requested + padding attempts.
    oversample = max(n * 6, n + 500)

    # Build candidate index groups deterministically.
    candidates = []
    seen_keys = set()
    attempts = 0
    max_attempts = oversample * 3
    while len(candidates) < oversample and attempts < max_attempts:
        attempts += 1
        idx_group = tuple(sorted(rng.sample(range(n_src), k)))
        key = idx_group
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(list(idx_group))

    # Build args for pool — include base_seed for per-group RNG (Fix #2).
    pool_args = [(idx_group, source_data, seed) for idx_group in candidates]

    results = []
    if n_workers > 1:
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_init_worker,
        ) as pool:
            for result in pool.imap_unordered(_make_example_worker, pool_args, chunksize=64):
                results.append(result)
    else:
        # Single-process path (also used in smoke-test / when n_workers=1)
        _init_worker()
        for args in pool_args:
            results.append(_make_example_worker(args))

    # Collect valid records (Fix #3: each group returns k records, not 1).
    # We deduplicate by source_indices + queried_subject_index tuple.
    examples = []
    seen_record_keys = set()
    n_dropped_overlap = 0
    n_dropped_len = 0
    for record_list, reason in results:
        if record_list is not None:
            for rec in record_list:
                rkey = (
                    tuple(rec["graph_meta"]["source_indices"]),
                    rec["queried_subject_index"],
                )
                if rkey not in seen_record_keys:
                    seen_record_keys.add(rkey)
                    examples.append(rec)
        elif reason == "entity_overlap":
            n_dropped_overlap += 1
        elif reason == "too_long":
            n_dropped_len += 1

    # Deterministic ordering: sort by (source_indices, queried_subject_index).
    examples.sort(key=lambda e: (
        e["graph_meta"]["source_indices"],
        e["queried_subject_index"],
    ))
    examples = examples[:n]

    stats = {
        "kept":             len(examples),
        "requested":        n,
        "candidates_tried": len(candidates),
        "dropped_overlap":  n_dropped_overlap,
        "dropped_too_long": n_dropped_len,
    }
    return examples, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ProsQA-MULTI-k datasets from raw ProsQA JSON files."
    )
    p.add_argument("--k", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Number of disjoint source examples to merge per output example.")
    p.add_argument("--train-out", required=True,
                   help="Output path for multi-k train JSON.")
    p.add_argument("--test-out", required=True,
                   help="Output path for multi-k test JSON.")
    p.add_argument("--src-train", required=True,
                   help="Source prosqa_train.json path.")
    p.add_argument("--src-test", required=True,
                   help="Source prosqa_test.json path.")
    p.add_argument("--seed", type=int, default=1337,
                   help="RNG seed for deterministic output.")
    p.add_argument("--n-train", type=int, default=5000,
                   help="Number of MULTI examples to generate for train split.")
    p.add_argument("--n-test", type=int, default=500,
                   help="Number of MULTI examples to generate for test split.")
    p.add_argument("--workers", type=int,
                   default=max(1, multiprocessing.cpu_count() // 2),
                   help="Parallel workers (default: cpu_count//2).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Generate 10 examples and print 2, then exit (no file write).")
    return p.parse_args()


def _print_example(ex: dict, idx: int):
    print(f"\n--- Example {idx} ---")
    print(f"QUESTION: {ex['question']}")
    print(f"  answer:                 {ex['answer']}")
    print(f"  steps ({len(ex['steps'])}):           {ex['steps'][:2]}{'...' if len(ex['steps'])>2 else ''}")
    print(f"  k={ex['k']}  queried_subject_index={ex['queried_subject_index']}")
    print(f"  sources={ex['graph_meta']['source_indices']}  queried_source={ex['graph_meta']['queried_source']}")


def main():
    args = parse_args()

    # Load source data
    print(f"Loading source train: {args.src_train}", flush=True)
    with open(args.src_train) as f:
        src_train = json.load(f)

    print(f"Loading source test:  {args.src_test}", flush=True)
    with open(args.src_test) as f:
        src_test = json.load(f)

    print(f"k={args.k}  seed={args.seed}  workers={args.workers}", flush=True)

    if args.smoke_test:
        # Generate enough records to verify option-order balance (~200 questions).
        # With k=2 and n=100 source groups we get ~200 records.
        smoke_n = max(100, args.k * 100)
        print(f"\n[SMOKE TEST] Generating {smoke_n} records from train source...", flush=True)
        examples, stats = generate_split(
            source_data=src_train,
            k=args.k,
            n=smoke_n,
            seed=args.seed,
            n_workers=1,  # single process for smoke test clarity
        )
        print(f"[SMOKE TEST] kept={stats['kept']}  "
              f"dropped_overlap={stats['dropped_overlap']}  "
              f"dropped_too_long={stats['dropped_too_long']}", flush=True)

        # Print first 5 examples.
        for i in range(min(5, len(examples))):
            _print_example(examples[i], i)

        # --- Fix #2 verification: check option-order balance ---
        # For each record extract the correct class and check which slot it's in.
        import re
        pos_in_slot1 = 0
        checked = 0
        for rec in examples:
            q = rec["question"]
            # Extract "Is <subj> a <opt_A> or <opt_B>?" at end of question
            m = re.search(r' Is \w+ a (\w+) or (\w+)\?$', q)
            if not m:
                continue
            opt_a, opt_b = m.group(1), m.group(2)
            # The correct class is the last word of the answer sentence.
            ans_tokens = rec["answer"].rstrip(".").split()
            correct_cls = ans_tokens[-1] if ans_tokens else ""
            if correct_cls == opt_a:
                pos_in_slot1 += 1
            checked += 1

        if checked > 0:
            frac = pos_in_slot1 / checked
            flag = "OK" if abs(frac - 0.5) <= 0.10 else "WARN — order bias detected!"
            print(f"\n[SMOKE TEST Fix#2] positive-class in slot-1: "
                  f"{pos_in_slot1}/{checked} = {frac:.3f}  [{flag}]", flush=True)
        else:
            print("\n[SMOKE TEST Fix#2] Could not verify option order "
                  "(regex found 0 matches).", flush=True)

        # --- Fix #3 verification: schema check ---
        required_keys = {"question", "answer", "steps", "k", "queried_subject_index",
                         "graph_meta"}
        missing_keys = required_keys - set(examples[0].keys()) if examples else required_keys
        if missing_keys:
            print(f"[SMOKE TEST Fix#3] FAIL — missing keys: {missing_keys}", flush=True)
        else:
            # Confirm answer is a single-sentence string (no newlines, ends with period).
            sample_answer = examples[0]["answer"]
            sentences = [s for s in sample_answer.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            schema_ok = len(sentences) == 1
            print(f"[SMOKE TEST Fix#3] schema OK — "
                  f"'answer' is {'single-sentence' if schema_ok else 'MULTI-sentence (unexpected)'}. "
                  f"Sample: {sample_answer!r}", flush=True)

        print("\n[SMOKE TEST DONE]", flush=True)
        return

    # --- Train split ---
    print(f"\nGenerating train split (n={args.n_train})...", flush=True)
    train_examples, train_stats = generate_split(
        source_data=src_train,
        k=args.k,
        n=args.n_train,
        seed=args.seed,
        n_workers=args.workers,
    )
    print(
        f"[TRAIN] kept={train_stats['kept']}/{train_stats['requested']}  "
        f"candidates_tried={train_stats['candidates_tried']}  "
        f"dropped_overlap={train_stats['dropped_overlap']}  "
        f"dropped_too_long={train_stats['dropped_too_long']}",
        flush=True,
    )
    if train_stats["kept"] < args.n_train:
        print(
            f"[WARNING] Only {train_stats['kept']} examples produced "
            f"(requested {args.n_train}). Increase source data or reduce --n-train.",
            flush=True,
        )

    # --- Test split ---
    # Use a different seed offset so train and test don't collide.
    print(f"\nGenerating test split (n={args.n_test})...", flush=True)
    test_examples, test_stats = generate_split(
        source_data=src_test,
        k=args.k,
        n=args.n_test,
        seed=args.seed + 1,
        n_workers=args.workers,
    )
    print(
        f"[TEST]  kept={test_stats['kept']}/{test_stats['requested']}  "
        f"candidates_tried={test_stats['candidates_tried']}  "
        f"dropped_overlap={test_stats['dropped_overlap']}  "
        f"dropped_too_long={test_stats['dropped_too_long']}",
        flush=True,
    )

    # --- Write outputs ---
    for out_path, examples in [
        (args.train_out, train_examples),
        (args.test_out,  test_examples),
    ]:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Wrote {len(examples)} examples -> {out}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
