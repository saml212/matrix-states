# Control A — Research Report for the Implementer Agent

**Experiment:** forward-only rank-k projection ablation on vanilla GPT-2 SFT (ProsQA) to test whether ProsQA is rank-1-solvable (attack A16).
**Target deadline:** ICML MI Workshop 2026, 2026-05-08.
**Reuses:** Round 4 vanilla SFT checkpoints (seeds 1337, 42, 7). Runs on the H100 the user will provision.

---

## 1. Protocol walkthrough

### 1.1 Checkpoint format and location — **BLOCKER, needs Sam**

The local repo at `experiment-runs/2026-04-13_round4_vanilla_sft/` contains only the three per-seed `_SUMMARY.txt` + `_results.json` pairs. It contains **no `.pt` files and no script**. The Round 4 run used a separate script named `run_vanilla_sft.py` (referenced in `EXPERIMENT_LOG.md` L1255, `experiment-runs/_auto_sync/WAKEUP_RUNBOOK.md` L87, `experiment-runs/_auto_sync/WORKFLOW_FOR_AGENTS.md` L308) that lived only on the pod at `/workspace/pebble/round4_vanilla_sft/scripts/run_vanilla_sft.py` (by analogy with the `/workspace/pebble/round3_gamma0/scripts/run_matrix_codi.py` layout in `WORKFLOW_FOR_AGENTS.md` L110).

Evidence from `experiment-runs/_auto_sync/logs_round4_sft_control/pure_sft_seed1337.log`:
- Line 3: `=== Vanilla SFT (mode=pure_sft, seed=1337) ===`
- Line 4: `Params: 124,439,808` (GPT-2 small with 3 new special tokens added)
- Line 5: `[ProsQAVanillaDataset train pure_sft] kept 17886/17886`
- **No "Saved" / "checkpoint" / `.pt` lines anywhere in the log.**

Two open questions that **Sam must resolve before the implementer runs**:

1. **Does `run_vanilla_sft.py` actually save a best-checkpoint file?** The `results.json` structure (`experiment-runs/2026-04-13_round4_vanilla_sft/pure_sft_seed1337_results.json`) records only `best_accuracy` / `training_curve` / `config`; there is no `ckpt_path` field and the orchestrator log shows no save line. If the script saves nothing, the checkpoints do not exist and Round 4 must be partially re-run (3 seeds × ~37 min = ~2 GPU-h on 1×H100 — still well under the 1-GPU-h spec's spirit, but longer than the spec implies).
2. **If it does save, what is the state-dict schema?** `run_matrix_codi.py` L1809 saves `{"model": model.state_dict(), "cfg": cfg, "use_matrix_bottleneck": use_matrix, "tokenizer_len": len(tokenizer)}`. `run_vanilla_sft.py` is a different script — it may use the HF trainer convention (`model.save_pretrained`) or a plain `torch.save({"model": ...})`. The implementer MUST SSH in and `cat /workspace/pebble/round4_vanilla_sft/scripts/run_vanilla_sft.py` early, before writing any loading code.

**Working assumption for the rest of this report:** the Round 4 script saves `best_pure_sft_seed{S}.pt` as either a raw `state_dict()` of a `GPT2LMHeadModel` with resized embeddings (vocab size ≈ 50260), or a dict `{"model": state_dict, "cfg": cfg}`. If it saves nothing, the implementer must first rerun Round 4 pure_sft with `torch.save({"model": raw.state_dict(), "cfg": cfg, "seed": seed, "tokenizer_len": len(tokenizer)}, out_dir / f"pure_sft_seed{seed}.pt")` added before the experiment proceeds.

### 1.2 ProsQA split and tokenizer

- Dataset path (on pod, matches Round 3/4): `/workspace/pebble/round3_gamma0/data/prosqa_test.json`, 500 examples. Same path is banked in `pure_sft_seed1337_results.json` L152.
- Tokenizer: `GPT2TokenizerFast.from_pretrained("gpt2")` with 3 added special tokens `<bot> <eot> <latent>` and `pad_token = eos_token`, `padding_side = "left"` (matches `run_matrix_codi.py` L1936–1939 and `assert_colon_tokenization` at L1579). **The implementer must add the same three special tokens even though Control A does not use `<latent>` in the forward pass**, because Round 4's checkpoint was saved with a resized embedding table that includes them (Params: 124,439,808 on the log ≈ 124,439,808 - 124,439,808 vanilla GPT-2 = +3 rows in wte/lm_head ≈ consistent with 3 added special tokens).
- Eval problems: 500 (matches Round 4 val split). Cap at 500 for Control A — not `cfg["max_eval_batches"] × eval_batch_size` (which is 32×16=512 in matrix-CODI and would exceed the split). One run per seed, no replication of the eval set.
- Answer-position matching: `prosqa_answer_match(pred_text, gold_text)` at `run_matrix_codi.py` L1230–1247. Strip to first sentence (up to first `. ? ! \n`), lowercase, match last whitespace-separated token. **Reuse verbatim.**

### 1.3 Fake-Z construction — recommended order

The spec lists three variants. Recommended run order:

**Primary (run first):** **Variant 1 — 16×16 from first 256 dims of h_answer.**
At ProsQA's " The answer is:" position, extract the last-layer hidden state `h ∈ ℝ^{768}` (exactly what `probe_z.py` L145–151 already does for vanilla GPT-2). Reshape `h[:256]` to a 16×16 matrix. SVD-truncate to rank k, flatten, splice back into `h[:256]`, leaving `h[256:768]` untouched. Forward the modified `h` through `gpt2.lm_head` (= `wte` weight tie) to get logits. Argmax → next token → greedy decode under KV cache.

Why first: matches matrix-CODI's d=16 bottleneck exactly (`run_matrix_codi.py` L230, `w_up: 768→256`). Most directly comparable to our flat curves.

**Robustness 2 — 16×48 from all 768 dims.** Reshape full `h` to (16, 48). SVD-truncate, flatten back, replace all of `h`. Rank sweep is `k ∈ {1, 2, 4, 8, 16}` (same ks — the matrix is 16×48 so max rank is 16, matching the k-range).

**Robustness 3 — Low-rank SVD on the full 768-dim via 768=24×32 or similar factorization.** Alternative: treat `h` as a (1, 768) row, take outer-product factor... skip this one. Instead, do **Variant 3': rank-k SVD of the entire 768 hidden state treated as (24, 32)**, reasonable square-ish factor. Same k sweep with k_max=24. This is the "all 768 dims reshaped" check the spec mentions.

**If primary (Variant 1) curve is flat AND both robustness curves are flat: strong evidence ProsQA is rank-1-solvable (flag paper §7).**
**If primary is flat but robustness variants bend: the flatness is specific to a low-dim slice of the hidden state; the ablation method HAS discriminating power but the specific 16×16-first-slice construction is what's blind. Paper framing stays close to current.**
**If primary bends: strong win; the ablation discriminates on ProsQA.**

### 1.4 Intervention and forward pass

Pseudocode (not code to write — this is for clarity):

```
Load GPT-2 small + resized embeddings from pure_sft_seed{S}.pt
Build prompt_ids = q_ids + tokenizer.encode(" The answer is:")
out = gpt2.transformer(prompt_ids, output_hidden_states=True)
h_answer = out.last_hidden_state[:, -1, :]        # (1, 768)
h_modified = intervene_fake_Z(h_answer, variant, k)
# forward the modified h through the final layer norm (already applied) + lm_head
logits = gpt2.lm_head(h_modified)
next_tok = argmax(logits)
# then KV-cache greedy decode up to 24 tokens (matches generate_answer L1183 for prosqa)
```

**Gotcha:** `gpt2.transformer` applies `ln_f` inside the transformer before returning `last_hidden_state`, so `h_answer` is post-ln_f. No double-normalization. `gpt2.lm_head` is `wte.weight.T` (tied weights in GPT-2). The implementer should use `model.lm_head(h_modified)` to preserve tied-weight semantics; do NOT manually re-do `ln_f`.

**Gotcha 2:** KV-cache greedy decoding after the intervention. Because the intervention touches only the answer-position hidden state (one token), the KV cache for prior tokens is unaffected, but the intervened `h_answer` is NOT what would go into the attention layers for the NEXT token — rather, the NEXT token's query attends to the cached K/V of prior positions (unchanged) + the prior-position LM head output. The safest single-answer-token protocol: **only predict the first generated token from the intervention**, check its text against the gold answer via `prosqa_answer_match`. This matches Rizvi-Martel-style COCONUT eval and avoids cascading-intervention ambiguity. Do NOT decode multiple tokens with the intervention applied to every step — that would also be defensible but needs a separate design decision.

Recommendation: **single-token first-answer-token protocol.** The gold ProsQA answer is a short sentence like "tom is a bunlion." The gold first post-`:` token is " tom" (the entity name). Matching: take the argmax of the intervened logits → decode → check `first_sentence_final_class(pred) == gold_class`. This simplification is defensible because the experiment tests rank of the answer-position latent, not the full generation chain.

(Open question: the implementer should sanity-check what `first_sentence_final_class` does when the prediction is a single token like "tom". If gold is "tom is a bunlion" and pred is " tom" only, `first_sentence_final_class(" tom") == "tom"` and `first_sentence_final_class("tom is a bunlion") == "bunlion"` — they won't match. This will BIAS Control A toward low accuracy and would falsely suggest bending. **Fix:** use a matching function that compares only the FIRST non-space token of pred against the class-word position in gold, OR greedy-decode 8–16 tokens but only intervene on the first step. The latter is cleaner. Sam/implementer: decide.)

### 1.5 SVD truncation

**Reuse verbatim:** `truncate_to_rank(Z, k)` at `run_matrix_codi.py` L336–347. Works on `(B, d, d)` batch. fp32 SVD inside `torch.autocast("cuda", enabled=False)`. The implementer should import this function directly.

### 1.6 k sweep, seeds, eval set

- k ∈ {1, 2, 4, 8, 16} (primary variant). Variant 3' uses {1, 2, 4, 8, 16, 24}.
- Seeds: all three (1337, 42, 7). Run sequentially; 500 examples × 5 k × 3 seeds × ~0.5s/example = ~60 min. Spec budget of 1 GPU-h holds.
- Optional: full 500 examples per seed. No sub-sampling. Deterministic (`torch.use_deterministic_algorithms(False)` is fine — argmax greedy decode is already deterministic).

---

## 2. Functions to reuse vs write fresh

### Reuse verbatim (import from `run_matrix_codi`)

| Function | File:Line | Why |
|---|---|---|
| `truncate_to_rank(Z, k)` | `run_matrix_codi.py`:336–347 | SVD truncation protocol; fp32-safe, autocast-safe. Match Round 3's truncation for apples-to-apples comparison. |
| `effective_rank(Z)` | `run_matrix_codi.py`:350–357 | Sanity-check post-truncation rank. Use for logging only. |
| `prosqa_answer_match(pred, gold)` | `run_matrix_codi.py`:1230–1247 | Matches Round 3/4 eval. **NOTE 1.4 gotcha** about first-token predictions. |
| `ProsQADataset(split, tokenizer, cfg, special_ids)` | `run_matrix_codi.py`:757–876 | Produces `q_ids`, `tail_ids`, `answer_text`. Reuse for loading eval set. |
| `assert_colon_tokenization(tokenizer)` | `run_matrix_codi.py`:1579 | Guards against ":" tokenization drift between tokenizer versions. |
| `set_seed(seed)` | `run_matrix_codi.py`:1593 | Standard. |
| `build_logger(results_dir, is_main=True)` | `run_matrix_codi.py`:1555–1576 | For the `control_a.log` file. |

### Adapt (copy and modify)

| Source | Change | Why |
|---|---|---|
| `generate_answer(...)` at `run_matrix_codi.py`:1064–1207 | Strip out `student_forward` (no latents), strip KV-cache second-forward loop; keep ONLY: build prompt, run GPT-2 once, intervene at answer position, run lm_head, argmax. | Control A has no latent steps, no `<bot>...<eot>` scaffolding. The prompt is just `q_ids + " The answer is:"`. |
| `evaluate_gsm8k(...)` at `run_matrix_codi.py`:1251–1313 | Replace the `generate_answer` call with a `intervene_and_predict` call. Keep the ProsQA branch of the correctness check. Remove `Z_stack` / `effective_ranks` bookkeeping (no Z matrices to rank — Control A computes effective rank of the fake-Z AFTER truncation for logging only). | The per-problem loop structure is correct; only the intervention is different. |
| Checkpoint loader at `eval_rank_projection` L1930–1954 | Load a vanilla GPT-2 (not CodiModel). Expect `state_dict` key either `"model"` (dict wrapped) or at the top level. Handle BOTH formats (use `sd["model"] if "model" in sd else sd`). Add the 3 special tokens + resize embeddings BEFORE `load_state_dict` to match Round 4's 124,439,808 param count. | Round 4's checkpoint is a plain GPT-2LMHeadModel, not a CodiModel. |

### Write fresh

| Function | Purpose |
|---|---|
| `intervene_fake_Z(h, variant, k)` | Take (1, 768), reshape per variant, SVD-truncate (call `truncate_to_rank`), flatten back, return modified (1, 768). ~30 lines. |
| `predict_one_token_intervened(model, q_ids, q_mask, tokenizer, special_ids, variant, k)` | Single forward, intervention, lm_head, argmax, greedy decode 8–16 tokens (intervention applied at step 0 only). ~50 lines. |
| `run_control_a(cfg, checkpoint_path, variant, seed)` | Top-level: load ckpt, build eval dataset, sweep k, compute accuracy, Spearman. Mirrors `eval_rank_projection` structure. ~100 lines. |
| `main()` argparse with `--checkpoint`, `--seed`, `--variant` (16x16 | 16x48 | 24x32), `--ks`, `--results-dir`. ~30 lines. |

Total implementation: ~250 new lines + ~50 lines of glue. One file, `matrix-thinking/scripts/run_control_a.py`.

---

## 3. External code / literature findings

I did not run external web fetches in this report (cannot verify URLs without tool access this turn) — every citation below is flagged as "candidate reference, verify before citing." The implementer/Sam should spot-check URLs before the paper draws on them.

- **CODI (arXiv 2502.21074)** — Shen et al. 2025, EMNLP 2025. Code at `github.com/zhenyi4/codi` (referenced in `QUEUE.md` L110 and `STATE.md` L139, **candidate reference, verify**). CODI codebase does not contain a vanilla-SFT ProsQA baseline; it trains on GSM8K-Aug. **Not directly reusable for Control A.**
- **COCONUT (Hao et al. 2024, arXiv 2412.06769)** — code at `github.com/facebookresearch/coconut` per `EXPERIMENT_LOG.md` reference context. Has ProsQA training pipeline but NOT a rank-k ablation. Its eval code CAN be cited methodologically for ProsQA accuracy matching. **candidate reference, verify.**
- **Illusion of Superposition (Rizvi-Martel et al., arXiv 2604.06374)** — directly adjacent. From `EXPERIMENT_LOG.md` L1167 and `PAPER_WRITER_BRIEF.md` L132, code at `github.com/michaelrizvi/coconut` (per rebuttal 04_rebuttal_report.md L260, **candidate reference, verify**). They run a latent-ablation experiment (zero-out COCONUT latent tokens) but NOT a rank-k SVD ablation on a vanilla-SFT hidden state. Methodologically closest but protocol is different. Worth citing for the "null-latent control" framing; not worth adapting their code.
- **SIM-CoT (Shen et al., arXiv 2509.20317, ICLR 2026)** — per `PAPER_RESULTS_SUMMARY.md` L125. Diagnoses latent-CoT instability as insufficient step-level supervision; does NOT run rank-based ablation. **Not methodologically informative for Control A but paper-side important.**
- **Nazari & Rusch (arXiv 2602.04852) — "Key to State Reduction..."** — per `PAPER_WRITER_BRIEF.md` L138. Measures effective rank of LINEAR ATTENTION hidden states; proposes post-training rank pruning of K/Q. Different object (fast-weight memory inside attention, not latent thought matrices). **Their truncation protocol is an SVD truncation of the trained attention weight product** — methodologically similar but applied to frozen weights rather than runtime activations. **Recommendation: match their SVD truncation in-spirit** (full-matrix SVD, keep top-k singulars, reconstruct, replace) — which is what `truncate_to_rank` already does. No code adaptation needed; the protocols are compatible.
- **State Rank Dynamics (arXiv 2602.02195)** — per `PAPER_WRITER_BRIEF.md` L140. Descriptive rank measurement across pretraining. No truncation ablation code to borrow.
- **Dynamics Within Latent CoT (arXiv 2602.08783)** — mentioned in `PAPER_WRITER_BRIEF.md` L134. Unknown methodology without fetching; **candidate reference, verify, low priority for Control A.**
- **Kobayashi et al. "Weight decay induces low-rank attention layers" (arXiv 2410.23819, NeurIPS 2024)** — pulled in by FIX-7 (rebuttal L487). Provides the implicit-low-rank-bias theoretical scaffolding that Control A's result feeds into but does NOT directly give code/protocol.

**Verdict on external code:** no published repo exactly does "rank-k SVD truncation on the answer-position hidden state of a vanilla-SFT transformer, sweep k, report accuracy." Control A is genuinely novel as a diagnostic protocol. **No external code should be adapted; reuse in-repo functions only.**

---

## 4. Gotchas and risk calls

The spec's explicit risks (fake-Z artificiality, seed variance, position effect) are in `QUEUE.md` L51–55. Additional risks I flag:

- **LM-head position indexing.** Vanilla GPT-2 SFT predicts the next token given `" The answer is:"`. The answer class word (e.g., "bunlion") is typically ONE generated token (or 2 BPE pieces). The first generated token is the one the intervened `h_answer` most directly affects. **Matching function must handle first-token predictions** — see 1.4 gotcha. `prosqa_answer_match` as written compares the last whitespace-separated token of the predicted FIRST SENTENCE. For a single-token prediction of just " bunlion" that's fine; for " tom" (just the entity name, no class), it's not. **The implementer should decode 8–16 tokens without re-applying the intervention after step 0 and let the model finish the sentence.** Cheaper than it sounds (KV cache).
- **Answer position not well-defined.** ProsQA answers are always `"{name} is a {class}"` per the `answer_text` field in `ProsQADataset` L799–807. The gold class is always the last token before the period. `first_sentence_final_class` is robust to this. Position is well-defined.
- **Is the answer position even the right place to intervene?** This is the DEEP interpretation risk: by the time the model reaches the answer position in vanilla SFT, all reasoning has already happened in the preceding context layers; the answer position is a surface-level decoding step. **If the answer position just does last-entity-copy from context, rank-k ablation at answer position will always be rank-1-solvable — because the "information" is a single class label.** This is PRECISELY the alternative explanation the paper is trying to rule in/out. **If Control A's curve is flat, this is the dominant interpretation** and is exactly what the paper needs to conclude. So flag it loudly but do not pre-judge: this IS the expected behavior if ProsQA is rank-1-solvable, and that's informative.
- **ProsQA tokenization — multi-token answers.** ProsQA class words are invented (e.g., "bunlion", "werpat"). They MAY BPE into 2+ pieces. The single-first-token protocol fails here; decode 8+ tokens (see 1.4) to let the model complete the class word.
- **DDP / device-mismatch from Round 4 checkpoint.** Round 4 trained on 1×H100 (no DDP per `EXPERIMENT_LOG.md` L1147 "Wall time: 37.3 min" for 25 epochs on 17886 examples = consistent with 1 GPU, no DDP). State dict should not have `module.` prefix; if it does, strip via `{k.replace("module.", ""): v for k, v in sd.items()}` (same idiom `probe_z.py` L68 already uses).
- **Token-embedding resize.** Round 4 added `<bot><eot><latent>` (Params: 124,439,808). If Control A loads the checkpoint into a fresh `GPT2LMHeadModel.from_pretrained("gpt2")` WITHOUT adding these tokens, `load_state_dict` will fail with a shape mismatch on `wte.weight` and `lm_head.weight`. **Fix:** always add the three special tokens + `resize_token_embeddings(len(tokenizer))` BEFORE loading. (This mirrors `run_matrix_codi.py` L1950–1952.) Even though Control A's prompt does NOT USE these tokens at inference, the embedding table must match the checkpoint.
- **PyTorch / transformers version drift.** Round 4 ran on "Python 3.11 + PyTorch 2.x + transformers 4.45-ish" per log L1 FutureWarning. If the implementer's pod is on Python 3.12 + PyTorch 2.9.1 (per `H100_SETUP.md` L10, though that doc is stale), `GPT2LMHeadModel.from_pretrained("gpt2")` still works but watch for any tokenizer version skew (the `assert_colon_tokenization` guard at L1579 catches the main case).
- **Round 4 script may not save a checkpoint at all.** This is the #1 blocker. See §1.1. If no checkpoint, do not proceed — SSH in first.
- **`torch.linalg.svd` on a 16×16 matrix of a batch of 1 is ~microseconds.** No numerical-stability concerns expected; `truncate_to_rank` already does fp32 SVD with autocast disabled.

---

## 5. Proposed JSON output schema

One file per seed, following Round 3's schema (`experiment-runs/2026-04-12_round3_gamma0/run1_rank_ablation/rank_projection_ablation.json`) with minimal extensions.

```json
{
  "experiment": "control_a_fake_z_rank_ablation",
  "checkpoint": "/workspace/pebble/round4_vanilla_sft/results/pure_sft_seed1337/best.pt",
  "checkpoint_sha256": "<hex>",
  "dataset": "prosqa",
  "prosqa_split": "val",
  "prosqa_path": "/workspace/pebble/round3_gamma0/data/prosqa_test.json",
  "n_eval": 500,
  "seed": 1337,
  "fake_z_variant": "16x16_first256",
  "ks": [1, 2, 4, 8, 16],
  "config": {
    "base_model": "gpt2",
    "max_q_len": 640,
    "max_ans_len": 32,
    "max_total_len": 768,
    "max_new_tokens": 24,
    "intervention_only_step_zero": true
  },
  "results_by_k": {
    "1":  {"accuracy": 0.000, "n_correct": 0, "n_total": 500, "mean_effective_rank": 1.00},
    "2":  {"accuracy": 0.000, "n_correct": 0, "n_total": 500, "mean_effective_rank": 2.00},
    "4":  {"accuracy": 0.000, "n_correct": 0, "n_total": 500, "mean_effective_rank": 3.8},
    "8":  {"accuracy": 0.000, "n_correct": 0, "n_total": 500, "mean_effective_rank": 6.9},
    "16": {"accuracy": 0.000, "n_correct": 0, "n_total": 500, "mean_effective_rank": 12.4}
  },
  "range_pp": 0.0,
  "spearman": {"spearman_r": 0.0, "p_value": 1.0, "n": 5},
  "pre_registered_decision": {
    "rule": "|r|<0.3 AND range<2pp => FLAT; range>5pp monotone => BENDING; else AMBIGUOUS",
    "outcome": "FLAT | BENDING | AMBIGUOUS"
  },
  "per_problem_records": [
    {"problem_id": 0, "reasoning_steps": 3, "gold": "tom is a bunlion.", "pred_by_k": {"1": "...", "2": "...", "4": "...", "8": "...", "16": "..."}, "correct_by_k": {"1": false, "2": false, "4": true, "8": true, "16": true}}
  ]
}
```

Aggregated across seeds: write a sibling `control_a_aggregated.json` with mean/std accuracy per k across the three seeds and the combined-seed Spearman (correlation of mean accuracy vs k — that's the pre-registered rule).

---

## 6. Smoke test plan

Before the full sweep, the implementer should bake these checks into the script's `--mode smoke` path:

1. **Checkpoint loads cleanly** on a GPT-2 with 3 added tokens. Expected params = 124,439,808. Assert.
2. **One-example forward with k=16 and k=1 differ by at most some epsilon in the intervened h.** Specifically: compute `torch.norm(h_orig - h_intervened_k16)` — should be near zero (truncating to full rank does nothing). Assert < 1e-3.
3. **Compute `torch.norm(h_orig - h_intervened_k1)` on the same example** — should be NONZERO and of similar magnitude to the top singular vector of the 16×16 reshape. This confirms the intervention is actually changing the hidden state.
4. **Forward the intervened h through lm_head; confirm the top-1 predicted token is a plausible BPE id (non-negative, < vocab_size).** Decode it and print. The k=16 prediction should match the NO-intervention baseline's prediction byte-for-byte. Assert.
5. **Run on 10 problems at k=1 and k=16.** Confirm at least one k=16 prediction is correct (should match vanilla SFT's ~81% rate on 10 examples → ~8/10). If k=16 accuracy is below 50% on 10 random examples, the intervention is broken.
6. **Log effective_rank(fake_Z after truncation) at each k.** Mean across 10 examples at k=1 should be ~1.0±0.01. At k=8 should be near 8 (maybe 6–8 in practice). This confirms `truncate_to_rank` behaved correctly.

If any smoke test fails, stop, diagnose, do NOT run the full 500-example sweep.

---

## 7. Pre-registered decision rule (restated)

From `QUEUE.md` L38–42, in my words:

Compute accuracy at each k ∈ {1, 2, 4, 8, 16} per seed. Compute across-k Spearman r between k and accuracy (n=5 per seed, so pool over seeds for a more powerful test: n=15).

- **FLAT (support rank-1-solvable hypothesis for ProsQA):** `|r| < 0.3` OR accuracy range across k is `< 2 percentage points`. Paper must rewrite §3/§5 framing from "CODI is rank-blind" to "on ProsQA, no tested architecture/objective produces rank-dependent behavior — which we interpret as the task being rank-1-solvable." Paper still submits.
- **BENDING (ablation discriminates; objective-level claim holds):** accuracy range `> 5 percentage points` AND monotone decrease as k decreases. Paper's current framing vindicated.
- **AMBIGUOUS (range 2–5pp, or non-monotone):** per the stricter Round-2 lesson-6 pattern, treat as FLAT. Paper goes with the rewritten framing.

The implementer's SUMMARY.txt should print the verdict string verbatim so the paper-writer agent can grep for it.

---

## 8. What I am NOT sure about — questions for Sam

1. **Does Round 4's `run_vanilla_sft.py` save a checkpoint file, and where?** The local repo has no script and the log has no save line. If no checkpoint was saved, Control A cannot proceed without a Round 4 re-run. **This is the #1 blocker.** Need SSH access to inspect `/workspace/pebble/round4_vanilla_sft/` (or wherever it lives on the current pod).
2. **Is the current pod still alive at the IP in `H100_SETUP.md` (154.57.34.103:44178)?** That file mentions pod restarts can invalidate SSH ports (`WORKFLOW_FOR_AGENTS.md` L23, L200). The user said they would provide SSH access after the cascade — confirm.
3. **First-token vs full-sentence decoding under the intervention.** Recommended: greedy-decode 8–16 tokens, intervention applied at step 0 only. Confirm this is the right protocol. The alternative (single-token + classify) bypasses `prosqa_answer_match` which was designed for sentence predictions.
4. **Should I run Variant 3' (24×32 rectangular reshape) or skip it?** The spec lists "16×48" and "low-rank SVD on the full 768" as robustness checks; 24×32 is my own recommendation for a square-ish factor. Sam may prefer just Variant 1 + 16×48.
5. **ProsQA val = 500 problems is small. Should I also run on the train split (17886) as a "high-n-ambiguous-breaker"?** Default: no, val only, matches Round 3 protocol.
6. **The rebuttal report (L258–262, FIX-10) mentions a camera-ready experiment to run rank-k ablation on Rizvi-Martel's released COCONUT checkpoint.** Control A is NOT that experiment — it's on vanilla SFT. Confirm the Control A scope does not absorb that follow-up.
7. **Does the paper need three-seed Control A numbers for the submission, or is seed 1337 enough if unambiguously flat/bending?** Pre-registration in the spec says all three seeds. My protocol follows the spec (all three), but if compute is tight, seed 1337 first, then the other two if the first is ambiguous.
8. **Nazari & Rusch and State Rank Dynamics (arXiv 2602.04852, 2602.02195) — are their truncation protocols worth matching literally?** Recommendation: no — they truncate trained weights (offline post-hoc), Control A truncates a runtime activation. The SVD call is the same, but the framing is different. Confirm.
9. **If Round 4 didn't save checkpoints, re-run using `run_matrix_codi.py --mode train_vanilla` (which IS in the current script, L1602) OR re-run `run_vanilla_sft.py` with a save-patch added?** `train_vanilla` goes through the CodiModel wrapper (with `feedback_proj = LayerNorm` at L410, no matrix bottleneck, but still with 6 latent `<latent>` tokens in the student pass). That's NOT the same as `pure_sft` mode — `pure_sft` uses only the question + answer, NO latent tokens at all. Using `train_vanilla` would make Control A's baseline a DIFFERENT model than the Round 4 numbers in the paper. **The implementer MUST use `pure_sft` semantics** (direct q → answer forward), not `train_vanilla`. If Round 4 checkpoints don't exist, the re-run MUST use the `pure_sft` script.

---

**End of report. Implementer agent: read §1.1 and §8 Q1 first. Everything else is conditional on the checkpoint existing.**
