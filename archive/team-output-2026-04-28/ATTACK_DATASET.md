# ATTACK_DATASET.md — adversarial review of `build_prosqa_multi.py`

Target: `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/build_prosqa_multi.py`
Date: 2026-04-24
Reviewer: Attack agent

---

## Attack 1 — Disjointness leakage

**Lines:** 90–114 (`_are_disjoint`); used at 256 in `_make_example_worker`.

**Failure mode investigated:**
ProsQA has only **17 unique person names** (`Alex, Bob, Carol, Davis, Eva, Fae, Gabriel, Jack, Max, Oliver, Polly, Rex, Sally, Sam, Stella, Tom, Wren`), each appearing as the root in ~1000 examples and as a non-root *fact subject* in many more. Two failure surfaces:

1. **Same person queried twice** — blocked by line 106 (`len(set(root_names)) < len(root_names)`). PASSES.
2. **Root of source-i appearing in source-j's facts** — blocked by lines 110–113 (`name_i in q_j`). The check uses `in` on the raw question string, including the trailing `Is <name> a X or Y?` sentence. So if Tom is the queried subject in source-i and Tom appears anywhere in source-j (root OR a side-fact like "Tom is a bompus"), the merge is rejected. PASSES.
3. **Substring false-positive** — names are unique surface tokens (no name is a substring of another in this list — verified: "Alex"/"Bob"/.../"Sally"/"Sam"/"Wren"). No risk.
4. **Shared non-queried persons** — both source-i and source-j may mention "Polly is a vumpus" / "Polly is a fompus" with conflicting class membership. Polly is not queried in the merged example, and ProsQA inheritance chains are person-anchored, so this is harmless. The class-name overlap rationale in the docstring (102–103) is correct.

**Empirical stress test:** Random sampling of 10000 source-pair candidates → 23.6% rejected by the disjointness check. With only 17 root names and ~1000 examples per name, the dropped-overlap rate is high (matches the script's 6× oversample budget).

**Verdict: DOES NOT LAND.** The check is correct given ProsQA's vocabulary structure. The docstring is honest about what it checks and what it doesn't.

**Caveat (NEEDS FIX, minor):** The check uses Python `in` on raw strings. If a future ProsQA variant introduces a name like "Sam" alongside "Samantha", `"Sam" in "Samantha is a bompus"` would falsely reject. **Fix:** use word-boundary regex (`re.search(rf'\b{re.escape(name)}\b', q_j)`). Not currently exploitable, but a latent fragility.

---

## Attack 2 — Order-leak in question construction (FATAL)

**Lines:** 161–165 in `_build_combined_question`:

```
pos_cls = symbols[row["target"]]
neg_cls = symbols[row["neg_target"]]
subject_clauses.append(f"{subj} a {pos_cls} or {neg_cls}")
```

**Failure mode:**
The positive (correct) class is **always** the first option, the negative is **always** the second. Every MULTI question reads:

> "... Are Tom a {CORRECT} or {WRONG}, and Sally a {CORRECT} or {WRONG}?"

A model can ignore facts entirely and learn the rule "answer first option after each name" → 100% accuracy with zero reasoning. This collapses the rank-k argument: a rank-0 Z would still solve MULTI.

**Empirical confirmation:** I checked the *original* ProsQA on 1000 examples — `pos_first=483, neg_first=517`, i.e. the original randomizes ~50/50. The MULTI script **introduces** the leak that doesn't exist upstream.

**Verdict: LANDS — CRITICAL.** This single bug invalidates the entire experiment.

**Fix:** Randomize order per subject, deterministically using the RNG already seeded for split generation. E.g. inside `_build_combined_question`, pass an `rng` and:
```python
opts = [pos_cls, neg_cls]
rng.shuffle(opts)  # OR: if rng.random() < 0.5: opts = opts[::-1]
subject_clauses.append(f"{subj} a {opts[0]} or {opts[1]}")
```
RNG must be seeded per-example (e.g. from `source_indices`) so the worker pool stays deterministic. Without this fix the experiment cannot be run.

---

## Attack 3 — Answer-format incompatibility with `prosqa_answer_match` (FATAL)

**Lines:** 181–183 (build_answers writes a LIST under key `"answers"`) and 215–221 (output schema). Eval site: `run_matrix_codi.py` lines 873–953 (loader expects `row["answer"]` singular and `row["steps"]`); `prosqa_answer_match` at lines 1359–1376.

**Failure mode:**
1. `prosqa_answer_match` truncates the prediction at the FIRST terminal punctuation (`.`, `?`, `!`, `\n`) and returns the last token of that first sentence. For a multi-target answer the model is supposed to emit `"Tom is a zhorpus. Sally is a hilpus."`, the matcher only ever sees `"Tom is a zhorpus"` → can grade at most ONE of k targets.
2. The MULTI schema writes `"answers"` (LIST, plural) and **omits `"steps"`**. The existing `ProsQADataset` loader at line 893 reads `row["answer"]` (singular) and `row.get("steps", [])`. Loaded as-is, it would `KeyError` on `"answer"`. So either the dataset loader needs new code, or the script must emit a flattened format.
3. There is no `"answer_text"` field; eval at line 1416 requires `item["answer_text"]`.

**Verdict: LANDS — CRITICAL.** Even if Attack 2 is fixed, the eval pipeline cannot grade MULTI examples.

**Fix:** Two options.
- **Option A (minimal):** in `_build_example`, also emit `"answer"` as the period-joined concatenation of `answers` (e.g. `"Tom is a zhorpus. Sally is a hilpus."`) and `"steps"` as the concatenation of source step lists. Then update `prosqa_answer_match` to grade EACH gold sentence against the corresponding sentence of `pred_text`, returning `True` only if all k match. This requires a dedicated `prosqa_multi_answer_match` and a branch in `evaluate_gsm8k`.
- **Option B (cleaner):** keep MULTI schema as-is (list of answers) and write a new `ProsQAMultiDataset` + `prosqa_multi_answer_match`. Required either way; Option A is faster, Option B is clearer.

Whichever path, the dataset-builder script and the eval script must be modified together — the current build script is *unusable* by the existing eval as written.

---

## Attack 4 — Trivially decomposable across positions (LANDS — addressed by Validator #5)

**Lines:** 137–178 (`_build_combined_question`).

**Failure mode:**
After Attack 2 is fixed, each subject's classification still depends only on its own facts. There is no cross-subject reasoning required. matrix-CODI has 6 latent positions; a model can encode target-1 at position 1 with rank-1 Z₁ and target-2 at position 2 with rank-1 Z₂. The MULTI task is *position-decomposable*. This is exactly the objection raised by `METH_VERDICT.md` §2 ("Position-decomposition substitutes for rank-decomposition").

The MULTI dataset, by construction, **does not require within-position rank ≥ k**. The headline rank-k claim only follows if combined with the per-position-rank-1 forcing experiment (#5 in COMBINED_PLAN).

**Verdict: LANDS — but acknowledged by the plan.** This is not a bug in the script; it is a design property of the task. The script is faithfully implementing the planned task. The mitigation (#5 per-position rank-1 forcing) is a separate experiment whose existence makes the headline tenable.

**No script-level fix needed.** Document it loudly in the dataset README that MULTI alone does not falsify position-decomposition; the rank-1 forcing experiment is mandatory for the headline.

---

## Attack 5 — Train/test contamination

**Lines:** 425–457 (`main`).

**Failure mode investigated:**
Train uses `--seed` (line 413, 432), test uses `--seed + 1` (line 455). Train pulls from `src_train`, test pulls from `src_test`. Source files `prosqa_train.json` and `prosqa_test.json` are upstream-disjoint from the COCONUT/ProsQA release. No cross-contamination.

There is no shared candidate pool, no leakage of test source rows into train. Provenance is logged via `source_indices` so contamination can be re-checked post hoc.

**Verdict: DOES NOT LAND.** Train/test split discipline is correct.

**Minor note:** seed offsets `seed` and `seed+1` are conventional but not maximally separated; an adversary running with `--seed 0` for one split and `--seed 1` for another would collide. Not exploitable in practice.

---

## Attack 6 — Determinism

**Lines:** 285 (`rng = random.Random(seed)`), 296–306 (candidate generation), 313–318 (`imap_unordered`), 339 (`examples.sort(key=...)`).

**Failure mode investigated:**
- `random.Random(seed).sample` is deterministic.
- Candidate generation uses an ordered loop with a `set` for dedup but the LIST `candidates` preserves insertion order. Determined.
- `imap_unordered` returns results out-of-order across worker counts. **However** the script sorts by `source_indices` after collection (line 339) and dedups by tuple key, so the final ordered output is independent of worker count.
- `examples.sort(key=lambda e: e["graph_meta"]["source_indices"])` — `source_indices` is a list; list comparison is element-wise lexicographic and stable. OK.
- Worker `_init_worker` loads `GPT2TokenizerFast` per process; tokenization is deterministic.

**Verdict: DOES NOT LAND.** Output is deterministic across runs given `--seed`, regardless of `--workers`.

---

## Attack 7 — CPU cap respect

**Lines:** 376–378.

**Failure mode investigated:**
Default workers = `max(1, multiprocessing.cpu_count() // 2)`. On the M4 Mac Mini (10 cores) this picks 5. The user's CPU-cap rule is "use ≤ nproc/2 by default". Line 378 default complies.

User can override with `--workers N` — no enforcement that they remain at-or-below cpu_count/2. The user could set `--workers 16` on a 10-core machine. Not a script bug; user override is intentional.

**Verdict: DOES NOT LAND.** Default complies. User override is allowed by design.

---

## Attack 8 — Length-filter sanity

**Lines:** 235–242 (`_exceeds_max_len`), 261 (filter site).

**Failure mode investigated:**
Source ProsQA questions have median 358 GPT-2 tokens (verified empirically). For k=2 the combined facts alone are ~720 tokens before any answer overhead. **40.8% of random pairs exceed the 768-token cap** before considering the answer prefix.

Bias check: dropped pairs have mean combined steps 7.44 vs kept pairs 7.71 — i.e. dropped pairs are very slightly *easier* (fewer steps), not harder. The dataset is **not** systematically biased toward easy multi-target cases via the length filter. The bias is roughly neutral on reasoning depth; the filter primarily removes pairs with long *fact lists* (more clutter), which correlates only weakly with chain length.

For k=3 and k=4 the filter will become catastrophic — likely > 95% drop rate at k=4, which COMBINED_PLAN already gates as contingent (#10).

**Verdict: DOES NOT LAND for k=2** (mild and roughly unbiased). **NEEDS FIX for k≥3:** raise `MAX_TOTAL_LEN` or fall back to a 1024/2048 budget for k≥3, or warn loudly. At k=2 the 768-cap discards 40% of candidates without much bias and the oversample × 6 factor is barely sufficient.

**Fix recommendation:** make `MAX_TOTAL_LEN` a CLI arg with default 768 for k≤2 and 1280 for k≥3; print a hard warning if `dropped_too_long / candidates_tried > 0.5`.

---

## Additional findings (not in the original attack list)

### A9 — `_extract_facts` brittleness (lines 121–134)

The function locates the question sentence by `question_text.rfind(" Is ")`. If a fact happens to start a sentence with "Is" (e.g., a name beginning with "Is..."), the split point would be wrong. ProsQA names are fixed (17 names, none start with "Is"), so not currently exploitable, but fragile. Recommend regex match on the trailing sentence pattern: `re.search(r' Is \w+ a \w+ or \w+\?\s*$', question_text)`.

### A10 — Worker tokenizer-load failure is silent

Lines 71–78: if `GPT2TokenizerFast.from_pretrained("gpt2")` fails (no internet on first run, or HF cache miss), `_tokenizer = None` and `_exceeds_max_len` returns `False` for everything. The dataset will then contain examples that exceed 768 tokens and will crash or be truncated downstream. The warning prints once per worker but is easily missed in a parallel log.

**Fix:** set a module-level flag from the parent process before forking; if tokenizer fails to load in the parent, abort with a hard error rather than emitting a silently broken dataset.

### A11 — Answer-prefix conservatism (line 241)

`answer_prefix = " The answer is:"` is hard-coded inside `_exceeds_max_len`, but the actual training pipeline in `run_matrix_codi.py` line 884 also uses `" The answer is:"`. Consistent. OK.

### A12 — `seen_keys` sufficiency

Line 304 dedups exact tuple matches but does NOT prevent the same `(root_a, root_b)` pair appearing many times with different secondary subjects. With 17 root names this means many pairs share the same root-pair → the model can memorize root-pair → answer-pair correlations. Not technically a leakage vector (each pair has different facts) but reduces effective task diversity.

---

## Overall verdict

**Two CRITICAL bugs land:** (Attack 2) the option-order leak makes the task trivially solvable without reasoning, and (Attack 3) the schema is incompatible with the existing ProsQA loader and `prosqa_answer_match` matcher. **Either bug alone invalidates any rank-k claim built on this dataset.** Both must be fixed before any GPU run.

The disjointness check (Attack 1), determinism (Attack 6), and train/test split (Attack 5) are all sound. The length filter (Attack 8) is acceptable for k=2 but will need adjustment for k≥3. The position-decomposition concern (Attack 4) is a known property of the task itself, not a script bug, and is gated by experiment #5 in the broader plan.

**Status: HARD NO-GO until Attack 2 and Attack 3 are fixed.** Both are 30-line patches; expected total fix time < 1 hour. After fixes, re-run smoke test and verify (a) option order is ~50/50 and (b) `prosqa_multi_answer_match` grades the published smoke examples correctly.
