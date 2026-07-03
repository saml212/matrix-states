# Control A Audit Report

Auditor: audit-agent (Opus 4.7, 1M context)
Date: 2026-04-23
Script under audit: `matrix-thinking/scripts/run_control_a.py`
Reference: `matrix-thinking/scripts/run_matrix_codi.py`
Spec: `matrix-thinking/QUEUE.md` §PRIORITY 0
Research report: `matrix-thinking/CONTROL_A_RESEARCH_REPORT.md`
Implementer notes: `matrix-thinking/CONTROL_A_IMPLEMENTATION_NOTES.md`

## Verdict

**PASS-WITH-FIXES.** The core intervention logic, SVD truncation, and eval
loop are correct. The locked-in decisions (step-0-only intervention, 3 seeds,
k sweep, 16x16 primary variant, `prosqa_answer_match`, `truncate_to_rank`
reuse) are faithfully implemented. Two correctness bugs in the checkpoint
loader, one in the decision rule edge case, plus a handful of logging /
schema / determinism nits. None should block the bug-fixer's next pass.

## Blockers (must fix before the attack agent runs)

### P0-1 — Checkpoint unwrap logic can drop `gpt2.`-prefixed CodiModel saves at the unwrap step
**Location:** `run_control_a.py:152–153`
```python
if isinstance(sd, dict) and "model" in sd and not any(k.startswith("h.") or k.startswith("transformer.") for k in sd.keys()):
    sd = sd["model"]
```
**Impact:** The guard inspects TOP-LEVEL keys (which are `{"model", "cfg", ...}`
for a wrapped save). None of those start with `"h."` or `"transformer."`, so
the unwrap fires correctly for a `{"model": sd, "cfg": cfg}` save. HOWEVER, if
Round 4's `run_vanilla_sft.py` saved a RAW state_dict (no `"model"` wrapper)
that happens to have a key literally named `"model"` — unlikely for GPT-2 but
possible with custom heads — the guard silently does the wrong thing. More
importantly: a CodiModel save (unlikely here but handled by the code) has
`sd["model"]` whose keys are `gpt2.transformer.h.0.*`, `gpt2.lm_head.*`,
`bottleneck.*`, `feedback_proj.*`. After unwrap, the `gpt2.`-prefix strip at
line 158–159 drops every non-`gpt2.`-prefixed key (including `bottleneck.*`,
which IS what we want — but also including any key without a `gpt2.` prefix
that we WOULD want to keep).
**Fix:** The `gpt2.`-prefix strip should only run if ALL or MOST keys are
prefixed; current `any(...)` triggers the strip even if only one key happens
to match, dropping the rest. Change to:
```python
keys = list(sd.keys())
if keys and all(k.startswith("gpt2.") or k.startswith("bottleneck.") or k.startswith("feedback_proj.") for k in keys):
    sd = {k[len("gpt2."):]: v for k, v in sd.items() if k.startswith("gpt2.")}
```
Or simpler: require that MAJORITY of keys start with `"gpt2."` before
stripping, and also verify the stripped dict has at least `transformer.wte.weight`
and `lm_head.weight`. Otherwise abort with a clear error.

### P0-2 — Silent load: `strict=False` will accept a wildly wrong checkpoint
**Location:** `run_control_a.py:161–169`
**Impact:** `model.load_state_dict(sd, strict=False)` prints up to 5 missing /
unexpected keys but does NOT hard-fail. If Round 4's checkpoint shape is
different (e.g., embeddings sized for 50257 because special tokens were NOT
added at save time), the load will succeed with MISSING `wte.weight` /
`lm_head.weight` in the report and the model will silently retain the random
`from_pretrained("gpt2")` initialization for those rows. Control A would
report "accuracy ~0.5%" and the paper-writer could misread that as a valid
null baseline.
**Fix:** Assert that `transformer.wte.weight` and `lm_head.weight` are NOT
in the missing list:
```python
critical_missing = [k for k in missing if k in ("transformer.wte.weight", "lm_head.weight", "transformer.ln_f.weight", "transformer.ln_f.bias")]
assert not critical_missing, f"Critical keys missing from state_dict load: {critical_missing}"
```
Also: print `n_params` BEFORE and AFTER `load_state_dict` and assert the
loaded embedding shape matches `len(tokenizer)`.

### P0-3 — Smoke test does not verify vanilla-SFT baseline accuracy
**Location:** `run_control_a.py:494–573` (`run_smoke_test`)
**Impact:** Research report §6 item 5 requires: "Run on 10 problems at k=1
and k=16. Confirm at least one k=16 prediction is correct (should match
vanilla SFT's ~81% rate on 10 examples → ~8/10). If k=16 accuracy is below
50% on 10 random examples, the intervention is broken." The current smoke
test only checks ONE example and asserts (a) `delta_full < tol` for k=max,
(b) top-1 token identity for k=max, (c) `delta_k1 > 1e-4`. It does NOT
verify that unablated accuracy on the 10-example head of ProsQA matches the
~81% Round 4 number. A checkpoint loaded silently-wrong (see P0-2) would
pass all three current assertions but fail this one.
**Fix:** After the k=1 and k=max_rank sanity checks, run a 10-example
unablated decode (k=None path in `predict_one`) and assert
`correct_count >= 5`. Also print the predicted text for each so reviewers
can eyeball.

## Correctness fixes (should fix)

### P1-1 — `classify` treats NaN Spearman r as "flat"
**Location:** `run_control_a.py:483`
```python
if (abs(r) < 0.3 if r == r else True) or rng_pp < 2.0:
    return "flat"
```
**Impact:** If Spearman computation fails (all accs equal, scipy missing
and fallback returns NaN), the code falls back to `True` and returns "flat".
Better: return "ambiguous" when r is NaN, unless `rng_pp < 2.0` (in which
case the range rule alone is decisive). The current behavior could
accidentally convert a degenerate-data case into a paper-grade "FLAT"
verdict.
**Fix:**
```python
r_nan = not (r == r)
if (not r_nan and abs(r) < 0.3) or rng_pp < 2.0:
    return "flat"
```

### P1-2 — KV cache is seeded from the ORIGINAL h; intervention only affects logits at the first decoded step
**Location:** `run_control_a.py:217–240`
**Impact:** This matches the implementer's flag #3 and the locked-in
"step 0 only" semantics, but the decoding loop (L263–280) does NOT insert
the intervened h into the cache. Specifically: `past = out.past_key_values`
is computed from the un-ablated forward; the intervention re-applies
`lm_head` to a modified h but the NEXT token's self-attention attends to
cached K/V that were computed from un-ablated h at the prompt's last
position. This is the locked-in design, so NOT a bug, but it means the
intervention's influence on the generated text is essentially "which single
first token got picked." If the gold class word is multi-BPE (e.g.,
"bunlion" → 2 pieces), only the first piece is ablation-controlled; the
second piece is generated by the un-ablated model attending to un-ablated
cache plus the first-piece input embedding. This BIASES the experiment
toward flatness: even a rank-1 intervention will likely still let the
second BPE piece get produced by the un-ablated context.
**Fix:** This is an interpretational concern, not a code bug. Recommend the
implementer add a NOTE to the JSON payload flagging this (a new field
`intervention_semantics: "step_0_only; kv_cache_from_unablated_h; subsequent_tokens_use_cached_kv"`)
so the paper-writer and attack agent have it documented. Also: consider
logging per-example the first-token accuracy separately from the
full-sequence `prosqa_answer_match`, since first-token accuracy is a purer
read on the intervention's effect.

### P1-3 — `classify` ambiguous for n=1 k value
**Location:** `run_control_a.py:483–487`
**Impact:** If all ks except one crash (via SVD failure), `len(accs) == 1`,
`max-min == 0`, so `rng_pp == 0 < 2.0`, returns "flat". That's wrong — a
single data point cannot support any decision.
**Fix:** Early-return "ambiguous" if `len(accs) < 3`.

### P1-4 — `first_tok_text` from `tokenizer.decode(generated, ...)` with `generated = [int(first_tok.item())]`
**Location:** `run_control_a.py:242–243`
**Impact:** `first_tok_text` is re-decoded on every iteration branch — fine,
but the variable captures the decode of a single-token list. GPT-2 BPE may
produce leading-space tokens like `" tom"` whose `.decode(skip_special_tokens=True)`
returns `" tom"`. The smoke test prints this correctly, but verify that when
the `first_tok` is a special token (`<eot>`, `<latent>`, etc.), the
`skip_special_tokens=True` flag doesn't silently produce an empty string
that then matches `prosqa_answer_match("", "tom is a bunlion.")` → False.
Probably fine, but worth a line of defense.
**Fix:** Add a log at the first few iterations of the first seed to print
`(generated_ids, predicted_text, gold)` triples so we can confirm the
decoding is producing sensible output before 500 examples × 5 ks elapse.

### P1-5 — `n_params` counts trainable params only (`p.requires_grad`) on an `.eval()` model
**Location:** `run_control_a.py:403`
```python
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
**Impact:** On an eval model post-`model.eval()`, `requires_grad` is still
True by default (eval() only toggles dropout / batchnorm). So this should
return 124,439,808 as expected. But if any future refactor freezes params
for eval, this silently reports 0. Low risk today.
**Fix:** Remove the `if p.requires_grad` filter and rename to `n_params_total`.

### P1-6 — Output JSON lacks per-seed `checkpoint_sha256` at the top field; lives one level deep
**Location:** `run_control_a.py:671–686`
**Impact:** Round 3's schema (`rank_projection_ablation.json`) has
`checkpoint` at top level with a single value. The Control A payload has
`checkpoint_dir` at top level and per-seed `checkpoint_path` +
`checkpoint_sha256` inside `per_seed[seed]`. This is an acceptable
structural adaptation (aggregated multi-seed schema) but the paper-writer
grep'ing `checkpoint_sha256` must walk into `per_seed`. Document this in
the JSON with a `schema_version: "control_a_v1"` field so downstream
consumers know this is NOT the Round 3 schema.
**Fix:** Add `"schema_version": "control_a_v1"` and
`"parent_schema": "rank_projection_ablation.json (adapted for multi-seed)"`
to the payload dict.

### P1-7 — `eval_one` stores full `per_example` (500 entries × 3 seeds × 5 ks = 7500 records) in memory and writes them all to JSON
**Location:** `run_control_a.py:323–330, 432–438`
**Impact:** The resulting JSON could be 50–100 MB. Round 3's
rank_projection_ablation.json is already ~20 MB and caused git-status issues
per `STATE.md` references. Since `per_example` duplicates per-k, total
records = 7500. Not fatal but bulky.
**Fix:** Cap `per_example` to first 50 entries per (seed, k) for JSON
output (matching `cfg["max_rank_samples"]` convention); keep full list in
memory for in-script stats only.

## Style / convention fixes (nice to have)

### P2-1 — Script copy happens BEFORE the smoke-test short-circuit
**Location:** `run_control_a.py:607–612, 625–630`
**Impact:** Correct per CLAUDE.md but means smoke-test runs also produce
`_script_used.py` in the output dir, potentially overwriting a prior full
run's copy if the user uses the same `--output-dir` for smoke then sweep.
Minor; the implementer's convention matches `run_matrix_codi.py:1620`.

### P2-2 — No progress ETA logged; only elapsed-seconds on every 50 examples
**Location:** `run_control_a.py:331–337`
**Fix:** Add `eta = elapsed * (total - (idx+1)) / (idx+1)`.

### P2-3 — `torch.load(..., weights_only=False)` on untrusted checkpoint
**Location:** `run_control_a.py:151`
**Impact:** Matches `run_matrix_codi.py:1931` ("our own file, trusted")
convention. Acceptable because the checkpoint is Sam's own. No fix needed;
noted for the attack agent.

### P2-4 — `_spearman` fallback's p-value is always NaN
**Location:** `run_control_a.py:354–387`
**Impact:** If scipy is unavailable, p-values are NaN — fine for a coarse
decision but means the SUMMARY `p={pooled_p:.4g}` prints "nan". Low risk —
the H100 environment has scipy.

### P2-5 — Smoke test has no variant-B / variant-C coverage
**Location:** `run_control_a.py:494–573`
**Impact:** Implementer explicitly flagged this as a known limitation
(IMPLEMENTATION_NOTES.md L99–102). The smoke test only runs against
`args.variant`; for a full audit of the 16x48 and svd_full_768 variants, the
user must run the smoke test three times. Consider: `if args.smoke_test: for v in _VARIANT_SHAPES: run_smoke_test(ckpt, v, ...)`.

### P2-6 — Emoji-free, no-over-commenting: verified
Script is clean. No emojis. Comments are load-bearing.

### P2-7 — Deterministic dataset ordering across seeds: implicit
`ProsQADataset` reads `prosqa_test.json` in file order, and
`dataset.items[:n_examples]` takes the first N. All three seeds see the
same 500 examples in the same order. Good — matches matrix-CODI protocol.

## Verified correct (items the implementer got right)

- **SVD truncation:** Direct import of `truncate_to_rank` from
  `run_matrix_codi.py:336`. Identical protocol to Round 3 matrix-CODI.
- **`hidden_states[-1]` is post-`ln_f`:** Confirmed from HF transformers
  GPT2Model source — `ln_f` is applied after the block loop, THEN
  `all_hidden_states += (hidden_states,)`. The implementer's flag #2 is
  adjudicated CORRECT.
- **`model.lm_head(h_used)` without re-applying `ln_f`:** Correct.
  `GPT2LMHeadModel.lm_head` is a tied-weight `nn.Linear` (wte.weight.T), no
  embedded LN. Research report §1.4 gotcha handled correctly.
- **Three special tokens added before resize:** L143–149. Matches Round 4's
  124,439,808 param count.
- **`assert_colon_tokenization` guard:** L146. Matches matrix-CODI.
- **All three seeds in default:** L588 `[1337, 42, 7]`. Matches locked decision.
- **k sweep default:** L589 `[1, 2, 4, 8, 16]`. Matches locked decision.
- **n_examples=500 default:** L592. Matches Round 3 / Round 4 ProsQA test split.
- **Variant choices:** L590–591 restricts to the three known variants.
- **Script copy on main:** L610 `shutil.copy(Path(__file__).resolve(), out_dir / "_script_used.py")`.
  Matches CLAUDE.md rule.
- **Per-seed try/except sweep:** L437–441 and L654–656. One crash doesn't
  kill remaining work. Matches CLAUDE.md rule.
- **Empty past_key_values after intervention:** The locked-in design. KV
  cache is seeded by the un-ablated forward at L217–223, then reused; the
  intervention modifies `h` but NOT the K/V cache. This is the "step 0 only"
  semantics.
- **`prosqa_answer_match` reuse:** L52, L318. No custom answer-match logic.
  Matches CLAUDE.md same-dataset-same-metric rule.
- **Smoke-test assertions:** L549–557 asserts `delta_full < tol` AND
  `argmax(logits_orig) == argmax(logits_k_max)`. Both are correct.
- **Decision rule classification:** `classify` (L475–487) implements
  `|r|<0.3 OR range<2pp → flat; range>5pp AND monotone → bending;
  else ambiguous`. Modulo the NaN-handling bug (P1-1), matches the
  QUEUE.md §Success / falsification text exactly.
- **SUMMARY.txt produced:** L732–733. Format matches Round 4's style
  (header bar of `=`, separators of `-`, per-seed lines). Greppable.
- **Effective-rank logging per k:** L236, L543–544. Matches Round 3's
  `mean_effective_rank` field.

## Adjudication of implementer-flagged points (from CONTROL_A_IMPLEMENTATION_NOTES.md)

**Flag #1 (checkpoint loader handles state_dict / wrapped / `module.` /
`gpt2.` prefixes):** PARTIALLY correct. Handles state_dict and `{"model": sd}`
cleanly. The `gpt2.`-strip branch is too aggressive (P0-1 above) and
`strict=False` is too forgiving (P0-2). Fix both before running.

**Flag #2 (post-`ln_f` lm_head usage):** CORRECT. `out.hidden_states[-1]`
is post-`ln_f` in both transformers 4.x and 5.x; `model.lm_head` applies
the tied-weight projection with no embedded normalization. The implementer
nailed this.

**Flag #3 (KV cache does not include intervention):** CORRECT and
INTENTIONAL. Matches the locked-in "step 0 only" protocol. Document the
semantics in the JSON payload (see P1-2 above) so the attack agent and
paper-writer don't misread it as a bug.

**Flag #4 (variant 16x48 / svd_full_768 not smoke-tested):** ACCEPTABLE
for the primary (16x16) run but should be smoke-tested before the
secondary sweeps. See P2-5.

## Uncertainty

Items I cannot conclusively audit without running the code or SSH'ing to
the H100:

1. **Actual Round 4 checkpoint schema.** Neither the local repo nor the
   research report has seen `/workspace/pebble/round4_vanilla_sft/pure_sft_seed{S}.pt`.
   If Round 4 saved a plain HF `save_pretrained` directory instead of a
   `.pt` file, the script's `torch.load` path will fail — the user MUST
   verify the file format before the full sweep runs.
2. **ProsQA multi-BPE class words.** "bunlion", "werpat", etc. may BPE into
   2–3 pieces. The locked-in step-0-only intervention means only the first
   piece is ablation-controlled. I cannot evaluate the impact empirically
   without running on GPU. Flag for the attack agent: a flat curve here
   might be EITHER rank-1-solvable ProsQA OR an artifact of the multi-BPE
   fact that un-ablated cache + first-piece embedding can recover the
   second piece. Recommend the attack agent ask for the first-token
   accuracy-by-k numbers separately from the full-sequence numbers.
3. **scipy availability on the H100 pod.** The fallback `_spearman` works
   but returns NaN for p-values. Assumed-present; implementer should log
   import status at startup.
4. **`weights_only=False` and pickle-unsafe checkpoints.** Consistent with
   matrix-CODI. Not a practical issue for Sam's own checkpoints.
5. **Actual vanilla-SFT 10-example accuracy under the current
   tokenizer + decoding protocol.** The smoke test should verify this
   (see P0-3). I cannot pre-compute it without the checkpoint and GPU.
6. **Determinism of `torch.linalg.svd` on H100 bfloat16 inputs.** `truncate_to_rank`
   casts to float32 for SVD — should be deterministic. Accept as
   correct per matrix-CODI Round 3 protocol.
