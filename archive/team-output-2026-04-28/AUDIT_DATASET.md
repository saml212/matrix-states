# AUDIT_DATASET.md — independent re-audit of `build_prosqa_multi.py`

Target: `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/build_prosqa_multi.py`
Date: 2026-04-24
Auditor: independent fresh-context audit agent
Source data: `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/data/prosqa_train.json` (17886 rows), `prosqa_test.json`
Verifier: `/tmp/audit_dataset/verify.py`

---

## Check 1 — Attack #2 fix (option-order randomisation): **PASS-WITH-NOTE**

**Code site:** lines 161–190 (`_build_single_query`) and 287–295 (per-group RNG seed in `_make_example_worker`).

**Code review verdict:** option order is randomised per-record using a per-group `random.Random(group_seed)` where `group_seed = base_seed ^ hash(tuple(idx_group))`. This is per-call, deterministic given `--seed`, and independent of worker count. The implementation is sound.

**Empirical test (n=500, seed=1337):** positive class in slot 1 = **281/500 = 0.5620**. Binomial 95% CI for n=500 is [0.4562, 0.5438]; this point is just *outside* (z = 2.77, p ≈ 0.006). User-spec tolerance was "≈50% ± 5%" → 0.562 is within 5pp of 0.5 but outside the tighter binomial CI.

**Multi-seed sanity test:** I re-ran with seeds {1337, 42, 7, 100, 2026}, n=500 each:
- 1337 → 0.562 (outside CI)
- 42 → 0.486 (in CI)
- 7 → 0.508 (in CI)
- 100 → 0.544 (just outside)
- 2026 → 0.486 (in CI)
- **Combined n=2500: 0.5172, CI ±0.0196 → in CI.**

The 1337 result is consistent with the per-seed binomial variance you would expect when you sample five seeds and pick whichever gave the spec'd seed. Across seeds the rate hugs 0.5. The randomisation is genuinely working; seed=1337 just happens to fall ~2.7σ above 0.5. Not a bug.

**Determinism:** SHA256(train_run1) == SHA256(train_run2) and SHA256(test_run1) == SHA256(test_run2). Identical output across runs. PASS.

**Verdict:** Fix lands. The slight slot-1 lean at seed=1337 is normal binomial noise, not an order leak.

---

## Check 2 — Attack #3 fix (schema compatibility): **PASS**

**Code site:** lines 193–246 (`_build_examples_for_group`) — emits per-record `question`, `answer` (string), `steps` (list), plus `k`, `queried_subject_index`, `graph_meta`.

**Schema check:** all 6 required keys present (`question`, `answer`, `steps`, `k`, `queried_subject_index`, `graph_meta`). PASS.

**`prosqa_answer_match` self-test:** loaded 50 generated records, ran `prosqa_answer_match(rec.answer, rec.answer)` for each → **50/50 True**. Sanity inequality check (`zhorpus` vs `jompus`) → False. PASS.

**ProsQADataset loader test:** I instantiated the equivalent of `ProsQADataset.__init__` inline with `cfg["max_total_len"]=1024` (to isolate from the length bug below) on the 500-record file. Result: **kept 500/500**, 0 colon drops, 0 length drops; `__getitem__` returns dict with all expected keys (`teacher_ids`, `teacher_colon_idx`, `q_ids`, `tail_ids`, `tail_colon_rel`, `answer_text`, `reasoning_steps`, `question`). PASS.

**Verdict:** Schema is fully compatible with the existing `ProsQADataset` and `prosqa_answer_match`. Both Option-A objections from the original Attack 3 are resolved.

---

## Check 3 — Records vs source-pairs interpretation: **PASS**

`--n-train 500` produces a file with **exactly 500 records** (not 1000). 250 unique source pairs × 2 records per pair = 500 records. The cap is applied at the *records* layer (line 391: `examples = examples[:n]`), so the user-facing knob means "examples in the file" — what you'd expect.

Slight subtlety: because each source pair generates k=2 records and dedup is on `(source_indices, queried_subject_index)`, the file has exactly k=2 records per pair when both pass the length filter. There's no oversampling at the per-record layer beyond the per-pair oversample budget.

**Verdict:** Interpretation matches user-facing expectation.

---

## Check 4 — Provenance leakage / duplicate records: **PASS**

Dedup by `(source_indices, queried_subject_index)` works as intended:
- 500/500 records have unique `(source_indices, queried_subject_index)` keys.
- 0 exact duplicates.

Note: the same source pair can yield records with `qi=0` and `qi=1` — these are *distinct* records by construction (different queried subject, different answer). This is correct, not leakage.

---

## Check 5 — Steps field correctness (no leakage of unqueried subject): **PASS**

For 50 sample records: `rec["steps"]` exactly equals `src_train_data[queried_src_idx]["steps"]` (no rewriting, no concatenation). The unqueried subject's name (e.g. "Sally" when "Tom" is queried) does NOT appear in any step text. **0 leakage events / 50 records checked.** The model sees only the queried subject's reasoning chain in CoT supervision. PASS.

---

## Check 6 — MULTI claim (question contains both subjects' facts): **PASS**

For 10 sample records, both queried-subject and unqueried-subject names appear in the question text (as fact subjects upstream of the trailing `Is X a A or B?` sentence). The MULTI property — that the latent must compress k subjects' facts even though only one is queried — is preserved. **10/10 records show both names present.**

---

## Check 7 — Length-filter bias (per queried-subject-index): **PASS**

Queried-subject-index distribution in 500-record output: `{0: 250, 1: 250}` — perfectly balanced. The length filter is symmetric across qi positions because all k records from one pair share the same fact context (same total length). PASS.

---

## Check 8 — Train/test contamination: **PASS**

`--src-train` and `--src-test` reference different upstream files (`prosqa_train.json` vs `prosqa_test.json`); the upstream COCONUT release guarantees these are disjoint. Empirically: **0 verbatim-question overlap** between the train and test outputs. The "overlap of source-index tuples" is non-meaningful because the integer indices live in different namespaces (different source files).

`seed` for train, `seed+1` for test — minor convention quibble (an adversary running with adjacent `--seed` values could collide), but in practice harmless and called out in the original attack agent's review.

---

## Check 9 — Drop-rate sanity vs fix-agent's report: **PASS**

| Stat | Fix-agent (n=200) | This audit (n=500) |
|---|---|---|
| dropped_overlap / requested | 1.525 | 1.422 |
| dropped_too_long / requested | 1.365 | 1.352 |

Both runs produce drop rates in the same ballpark. No pathological drift. The 6× oversample budget was sufficient for both runs (both kept 100% of requested).

---

## Check 10 — **NEW BUG: Length filter under-counts teacher length** (NEW BUG, MEDIUM)

**Code site:** `_exceeds_max_len`, lines 260–270.

**Failure mode:** The filter checks `question + " The answer is:" + answer` against `MAX_TOTAL_LEN=768`. But the *actual* training-time cap in `ProsQADataset` (`run_matrix_codi.py` line 967) is applied to `teacher_ids = tokenize(question + "\n" + cot_from_steps + " The answer is: " + answer)`. The CoT (joined `steps`) is ~50–200 GPT-2 tokens for ProsQA. The builder ignores it.

**Empirical impact:** with `--n-train 500` (k=2) and the default 768 budget, **118/500 records (23.6%) exceed 768 tokens at the teacher level** even though all 500 pass the builder's filter. These records will be silently dropped by `ProsQADataset.__init__` at training time.

Concrete consequence: the user asks for 5000 train examples → file contains 5000 → ProsQA loader silently drops ~1180 → only ~3820 examples actually train. The discrepancy will not surface unless someone reads the `[ProsQADataset split=train] kept N/M` line. Worse, the discrepancy biases the kept set toward shorter examples (fewer steps, simpler reasoning chains) — exactly the easier-bias Attack 8 warned about for k≥3, now appearing at k=2 because the filter under-counts.

**Fix (recommended):** include `steps` in the length check in `_exceeds_max_len`. One-line change:

```python
def _exceeds_max_len(record: dict) -> bool:
    if _tokenizer is None:
        return False
    answer_prefix = " The answer is:"
    cot = " ".join(str(s).strip() for s in record.get("steps", [])).strip()
    combined = record["question"] + "\n" + cot + answer_prefix + " " + record["answer"]
    return _token_len(combined) > MAX_TOTAL_LEN
```

After this fix, the builder will drop more candidates up front (oversample budget already has 6× headroom — verified: current run uses ~3000/3000=100% of the candidate budget, but only kept 500/3000 = 17%; tightening the filter will push that lower but still feasible for n=500). For n=5000 the oversample budget may need raising to 8× or 10×.

Alternative (acceptable but second-best): keep the filter as-is and bump `MAX_TOTAL_LEN` in the builder to ≈ 600 (768 minus ~150 for steps). Simpler, less precise.

**Verdict:** NEW BUG. Not catastrophic — training will still work but with a silently-truncated dataset and a mild easy-bias. **Should fix before launching the rank-k experiment**, otherwise the headline numbers come from a length-filtered subset that the user did not request.

---

## Overall verdict: **CONDITIONAL GO**

The two CRITICAL bugs from the original attack (option-order leak, schema incompatibility) are properly fixed and verified. Determinism, provenance, MULTI integrity, train/test split, and steps-handling all PASS. One NEW MEDIUM bug uncovered: the length filter under-counts teacher length by ignoring `steps`, so ~24% of records will be silently dropped by the ProsQA loader at training time and the kept subset is biased toward shorter examples.

**Recommendation:** apply the one-line fix to `_exceeds_max_len` (include steps in the length check), regenerate the dataset, then proceed. Without the fix, the experiment is runnable but produces a silently-filtered subset of the requested size — acceptable for smoke tests, not acceptable for the headline rank-k results.
