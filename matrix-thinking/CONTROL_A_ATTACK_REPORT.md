# Control A Attack Report

Attacker: attack-agent (Opus 4.7, 1M context)
Date: 2026-04-23
Prior cascade: research -> implement -> audit (PASS-WITH-FIXES, 3 P0 / 7 P1 / 7 P2)
Script under attack: `matrix-thinking/scripts/run_control_a.py`
Reference intervention: `matrix-thinking/scripts/run_matrix_codi.py` `eval_rank_projection` (L1919) and `student_forward` rank injection (L519, L522).

## Summary

Fatal attacks: **2**. Major attacks: **5**. Minor attacks: **4**.

**Verdict: do not run Control A as currently designed. The two fatal attacks are protocol-asymmetry problems that make a flat-vs-flat outcome uninterpretable. They can be fixed, but they MUST be fixed before H100 time is spent.**

The fatal problems: (F1) Matrix-CODI's rank-k intervention runs at **every latent step** (t=0..5) and propagates through downstream attention; Control A's intervention runs **once at the final answer position** and is not seen by the decoder's KV cache at all. This is not the null of the matrix-CODI experiment. (F2) The answer-position surface-decoding confound that the research report flagged (§4, risk 3) is *not actually mitigated by the 16x48 or svd_full_768 variants* — all three variants still intervene at the SAME answer position, just covering a bigger slice of h. A reviewer reading §5 of the paper will say "you tested whether the rank of h[:256] at one position matters; matrix-CODI tested whether the rank of Z at six latent-reasoning positions matters. These are different objects, and flatness of the former cannot falsify or ground-truth the latter." If we publish this as "the null of matrix-CODI's rank-k ablation was also flat, therefore ProsQA is rank-1-solvable," attack A16 will return stronger than before, because now we've spent an experiment on it and it didn't rule anything in.

Recommendation: fix F1 (intervene at the `<bot>` position and re-feed the intervened h as an embedding; or intervene at ALL 6 analogous "thought positions"); fix F2 (run at least one variant that intervenes at a mid-sequence position, not answer position); land the 3 P0 audit fixes; add the two controls below. Then run.

---

## Fatal attacks

### F1 — The intervention is not the null of the matrix-CODI intervention. Matrix-CODI ablates a latent that is consumed by downstream attention; Control A ablates the final hidden state that goes directly to lm_head.

**What matrix-CODI actually does.** In `run_matrix_codi.py` `student_forward` (L519), the rank-k truncation is applied inside `self.bottleneck(last_hidden, rank_project_k=k)` at EVERY ONE of the `n_latents=6` iterations (L522). Each truncated Z is *projected back to a 768-dim h_next*, *inserted into running_embeds as a new position* (L532–534), and *attended to by all subsequent latent steps, the tail prompt "The answer is:", and the answer decoder* (L542–548). By the time the model emits the answer-class token, the rank-k intervention has influenced 6 latent positions' residual streams, the joint attention patterns over the tail prompt, the cached K/V at every one of those positions, and every downstream layer's activations. The Z ablation is a *mechanism-level* intervention in the middle of the network.

**What Control A does.** In `run_control_a.py:217–240`, Control A runs a single clean forward of the vanilla SFT model over `q_ids + " The answer is:"`, grabs `h = out.hidden_states[-1][:, -1, :]` (last layer, last position, post-ln_f), truncates a reshape of this h to rank k, and passes it directly to `model.lm_head`. It then decodes further tokens *through the UN-ablated KV cache* (L263–280, implementer flag #3, audited as "correct and intentional" at audit P1-2). The intervention touches ONE hidden vector at ONE position right before lm_head.

**Why this is not the null.**
- Matrix-CODI's Z is a latent object that lives INSIDE the computation graph, is fed back through attention, and gets transformed by downstream layers. Flatness of its rank-k curve means "the mechanism is rank-indifferent across six attention-interacting positions."
- Control A's fake-Z is the FINAL hidden state right before the readout. Flatness of its rank-k curve means "lm_head is rank-indifferent to h[:256] reshaped to 16x16." These are different claims.
- A reviewer will say: "Matrix-CODI's flat curve tested the full CODI mechanism. Control A's flat curve tests a trivial linear-readout property. Your null baseline does not bound what you said it would bound."

**What the paper COULD NOT claim afterwards if F1 is left standing.**
- "Vanilla SFT shows the same rank-blindness, so ProsQA is rank-1-solvable" — this conclusion does not follow. Vanilla SFT's flatness here is guaranteed by Lesson 1 (linear lm_head is rank-blind to h[:256]) and Kill List #4 (F1). We already know linear readouts are rank-blind. Running the experiment to rediscover it is worth zero.
- "The ablation method has (or lacks) discriminating power" — the ablation methods are not the same method. Control A's intervention is strictly weaker than matrix-CODI's (one position vs six, no downstream attention exposure).

**Proposed fix.** Make Control A intervene on vanilla SFT's hidden state *at the `<bot>`-analogue position, early in the sequence, and propagate through downstream layers*. Concretely:
1. Tokenize `q_ids + " The answer is:"`.
2. Pick an intervention position `p` well before the last (e.g., the position of the "is" token, or midway through `" The answer is:"`, or after the last question token).
3. Run `model(input_ids, output_hidden_states=True, use_cache=False)`, extract `h_p = hidden_states[-1][:, p, :]`.
4. Apply rank-k SVD to a reshape of `h_p`, splice back.
5. Run a *second* forward where position p's last-layer hidden state is clamped to the intervened value — i.e., use `inputs_embeds` and override the layer-p activations, OR more practically, inject the intervention earlier: extract `wte` embeddings, modify the token embedding at position p to be `h_p_modified` (not perfect but closer), and re-forward. The cleanest protocol: hook the last transformer block's output at position p, overwrite it with the intervened value, let the attention downstream of p (the answer position's query) re-attend through the un-cached last-layer-only intervention. This is a standard activation-patching protocol and is what matrix-CODI is morally doing.

If 5 is too expensive on 500 examples x 3 seeds, the acceptable weaker version is: run the intervention at position p and decode from position p+1 onward with NO KV cache (recompute attention from scratch, so downstream queries see the intervened h_p). This preserves the "intervention propagates through attention" property that matrix-CODI has and Control A currently lacks.

**Citations:** `run_matrix_codi.py:519–548` (matrix-CODI's per-latent intervention), `run_matrix_codi.py:1154–1168` (the second pass where intervention runs again during eval seeding), `run_control_a.py:217–240,263–280` (Control A's single-position, un-propagated intervention), `CONTROL_A_AUDIT_REPORT.md` P1-2, `CONTROL_A_RESEARCH_REPORT.md` §1.4 "Gotcha 2", `KILL_LIST.md` #4 F1.

---

### F2 — The answer-position surface-decoding confound is NOT mitigated by the 16x48 and svd_full_768 variants. All three variants intervene at the SAME answer position.

**The research report explicitly flagged this** (§4, risk "Is the answer position even the right place to intervene?"): by the time vanilla GPT-2 SFT reaches the answer position, all reasoning has already happened in the preceding context layers. The answer position is a *surface decoding step* doing something like "copy the last entity from the preceding context that matches the expected class." A rank-1 ablation at the answer position is intervening on an almost-decoded vocabulary distribution, not a reasoning state. Flatness here is the *default expectation* under TWO competing hypotheses:
- Hypothesis A: ProsQA is rank-1-solvable.
- Hypothesis B: The answer-position hidden state is a decoded surface feature that requires rank-1 for class-label emission, regardless of the underlying task's complexity.

**A flat Control A curve CANNOT distinguish A from B.** Any GPT-2 LM head applied to an almost-emission hidden state will show rank-blindness on the first 256 dims, because the first 256 dims of the residual stream mostly encode output-vocabulary-relevant content for the next token, and the output decision space for ProsQA's first post-`:` token is ~500 class words (KILL_LIST.md #8 F2). A linear map from a 256-dim slice to ~500 output directions is rank-discriminating only if the task NEEDS more than ~50-ish directions to pick the right class — and it doesn't.

**Why the 16x48 and svd_full_768 variants don't save this.** The research report sold these as "robustness checks" that probe different fake-Z constructions. Reading `run_control_a.py:65–69` and `apply_rank_k_intervention`:
```
_VARIANT_SHAPES = {
    "16x16": (16, 16, 256),
    "16x48": (16, 48, 768),
    "svd_full_768": (24, 32, 768),
}
```
All three variants operate on the SAME `h = out.hidden_states[-1][:, -1, :]` (L226). They differ only in the reshape and the slice of h that gets overwritten. **They do not probe earlier positions or earlier reasoning.** So they cannot rule out hypothesis B. They cannot tell us anything about whether ProsQA requires rank > 1 for the *reasoning*; they only tell us what the model's answer-position decoding does.

**Why this is fatal (not major).** The paper wants to use Control A's outcome to inform the A16 rebuttal ("is the task rank-1-solvable?"). If Control A comes back flat (the most likely outcome), the paper will want to say: "Vanilla SFT with no matrix bottleneck also shows flatness — consistent with ProsQA being rank-1-solvable at the decoding position." A reviewer familiar with residual-stream mechanics will immediately say: "Of course the last hidden state decodes flatly — that tells us nothing about whether the reasoning earlier in the sequence required higher rank. The matrix-CODI ablation intervened throughout reasoning; your null baseline intervened only at the decoding step. You have not controlled for the thing you claimed to control for."

**What the paper COULD NOT claim afterwards if F2 is left standing.**
- "Control A supports the rank-1-solvable interpretation of ProsQA" — wrong. It supports "at the answer position, a linear readout is rank-blind," which is KILL_LIST Lesson 1 restated.
- The paper's §7 alternative-explanation discussion (FIX-17 in the rebuttal) would be *weakened*, not strengthened, because a reviewer now has explicit evidence that the authors ran an experiment purportedly to rule out rank-1-solvability and it failed to do so.

**Proposed fix.** Add a fourth variant that intervenes at a position EARLY in the sequence — specifically, at the last token of the question, BEFORE the model has seen `" The answer is:"`. At that position the hidden state encodes "whatever the model has reasoned about this puzzle so far"; it has not yet transitioned to decoding. If a rank-1 truncation there still produces ~81% accuracy on ProsQA, that is genuine evidence of rank-1-solvable reasoning. If it crashes accuracy to near-random, then the task DOES need higher rank but matrix-CODI's objective is still rank-blind — a much better paper finding.

Concretely: add `--variant full_ln_f_at_last_q_token`, which intervenes at position `len(q_ids) - 1` instead of the last position, and (to satisfy F1) propagates the intervention through the rest of the transformer by using a hook on the last transformer block's output. Keep 16x16 as a cheap robustness check but do NOT lead the paper with it.

**Citations:** `run_control_a.py:65–69`, `run_control_a.py:211–226`, `CONTROL_A_RESEARCH_REPORT.md` §4 risk 3 ("Is the answer position even the right place to intervene?"), `KILL_LIST.md` #8 F3 ("Z at the answer position is not 'reasoning state.'"), `submissions/icml-mi-workshop-2026/review/01_attack_report.md` A16 (L336–350).

---

## Major attacks

### M1 — Multi-BPE class words break the single-step intervention; flatness is structurally guaranteed for multi-token classes.

ProsQA class words are invented (e.g., "bunlion", "werpat", "sterpus"). GPT-2 BPE will almost certainly split these into 2–3 tokens. The locked-in "step-0-only intervention" (audit P1-2) means ONLY the first BPE piece is ablation-controlled; the second and third pieces are generated by the un-ablated model attending to un-ablated KV cache plus the first piece's input embedding. If the first piece of "bunlion" is common (e.g., " bun"), and the model has learned that after " bun" the next piece is deterministic (" lion"), then even a rank-1 intervention that corrupts the *first* piece can still recover on the second piece because:
- The un-ablated context (prior attention KV) already encoded the target class.
- The second BPE piece's generation depends on the first piece's token embedding + cached context, NOT on the intervened hidden state.

This structurally BIASES Control A toward flat curves, independent of whether the task is rank-1-solvable or not. `prosqa_answer_match` compares the LAST whitespace-separated token of the first sentence (L1245–1247), which for a 3-BPE class ("bun"+"lion"+".") would still match if the second and third pieces are emitted by the un-ablated model.

**Evidence of the magnitude.** Check the Round 3 rank_projection_ablation.json at k=1 vs k=16: accuracy was 76.8% vs 76.6% for gamma=0. The gold answer for problem 0 is "Sally is a sterpus." The matrix-CODI eval got the predicted answer "Sally is a sterpus." at k=1 (seen in `run1_rank_ablation/rank_projection_ablation.json` L52). If matrix-CODI's much stronger intervention (6 latent positions, propagated through attention) produces only 0.2pp degradation at k=1, then Control A's single-step, un-propagated intervention will produce near-zero degradation for **reasons having nothing to do with the task being rank-1-solvable**. It will produce near-zero degradation because BPE + un-ablated context lets the model recover.

**Fix.** Either (a) apply the intervention at every position of the class word's BPE expansion (requires re-forwarding after each decoded token with the intervention re-applied — expensive), or (b) use a class word that's definitely a single BPE token (impossible for invented ProsQA classes), or (c) acknowledge explicitly in the paper that "Control A measures first-BPE-piece correctness under rank-k ablation of the final hidden state, not ProsQA accuracy under rank-k reasoning ablation," and report first-token-accuracy-by-k separately from full-sequence accuracy. Option (c) is cheapest and best — and it directly implements the audit's P1-2 recommendation.

**Citation:** `run_control_a.py:263–280`, `CONTROL_A_RESEARCH_REPORT.md` §4 "ProsQA tokenization", `CONTROL_A_AUDIT_REPORT.md` P1-2.

---

### M2 — max_new_tokens mismatch with matrix-CODI: 16 (Control A default) vs 24 (matrix-CODI ProsQA).

`run_control_a.py:595` defaults `--max-new-tokens=16`. `run_matrix_codi.py:1183` sets `max_new = 24` for ProsQA. The two eval protocols are therefore NOT apples-to-apples. If any ProsQA gold answer is >16 BPE tokens, Control A truncates mid-answer while matrix-CODI doesn't. For `prosqa_answer_match` (which compares the last word of the first sentence), this could drop accuracy for examples where the period arrives at token 17–24.

**Fix.** Default `--max-new-tokens=24` to match matrix-CODI. Log per-example whether the first period was reached within the budget, so we can flag truncation-induced mismatches.

**Citation:** `run_control_a.py:595` vs `run_matrix_codi.py:1183`.

---

### M3 — Statistical power is likely insufficient to distinguish flat from gently-bending at n=500, and the decision rule does not account for this.

The QUEUE.md decision rule is "flat if |r|<0.3 or range<2pp." Per-seed the ks sweep has only n=5 data points, so Spearman r is trivially noisy (even the binary flat-accuracy case can hit |r|=0.3 by chance). Pooled across 3 seeds is n=15, still small.

Binomial SE on 500 examples at 80% accuracy is `sqrt(0.8*0.2/500) ≈ 1.79 pp`. Two-sided 95% CI on a per-(seed, k) accuracy is ±3.5pp. The "range < 2pp → flat" criterion is *INSIDE* the noise floor. We could observe 2pp range across five k's purely from binomial sampling noise on un-ablated-equivalent decoding.

**Why the Round 3 comparison looked fine anyway.** Matrix-CODI's Round 3 curves had ranges 0.4–0.6pp — comfortably below the noise floor but on a per-seed basis. Pooled across 3 seeds that's effectively n=1500 per k, shrinking the SE to ~1.0pp. But the problem is: if Control A's TRUE underlying curve is gently bending (say 3–4pp range), we have only ~50% power to detect it at 500 examples per seed. The decision rule would return "flat" in a case where the true effect is an informative gentle bend.

**What the paper COULD NOT claim afterwards.** If the decision rule returns "flat" on underpowered data, the paper cannot use this as evidence that "rank-1 is sufficient" — it's evidence that "we couldn't resolve a <3pp effect." The rebuttal would need to be "our null baseline is consistent with flat but not strongly diagnostic."

**Fix.** Add a pre-registered binomial confidence interval check. If the observed range is within 2*SE of zero, the verdict should be AMBIGUOUS, not FLAT. Code this explicitly into `classify` (currently L475–487 doesn't consider sample size at all).

**Citation:** `run_control_a.py:475–487`, `CONTROL_A_IMPLEMENTATION_NOTES.md` L114–116.

---

### M4 — The rank-16 "full-rank" baseline is not an identity operation because the intervention reshape+SVD+reshape introduces float noise; and the vanilla SFT lm_head has never seen this noise distribution.

At k=16 on the 16x16 variant, `truncate_to_rank` does `U @ diag(S) @ Vh` in fp32 and casts back to whatever h's dtype is (bfloat16 during eval per L308). This reconstruction is near-identity but NOT bit-identity. The smoke test at L549 tolerates `delta_full < 5e-2` for non-fp32 inputs. Over 500 examples, bf16 rounding noise injected into the hidden state that goes directly into lm_head could shift argmax for a small fraction of borderline examples — not because of the rank truncation, but because of the SVD round-trip.

**The problem this creates.** The "k=16" row of the table is supposed to be the un-ablated baseline. If it's actually "un-ablated + ~5e-2 bf16 noise," and if that noise drops accuracy by ~1pp, then we're comparing k=1..k=15 against a noisy reference, not against the un-ablated 81.77%. The range-by-k metric is a range relative to the k=16 noisy baseline, not relative to the clean baseline.

**Fix.** Add a k=None ("no intervention, no reshape, no SVD") data point to the sweep. Log its accuracy alongside the k=16 accuracy. Assert `|acc(k=16) - acc(k=None)| < 0.5pp` or investigate. This also doubles as a check that the checkpoint loaded correctly (per audit P0-3). Also: consider running the SVD in fp32 and keeping h in fp32 for the intervention-only path (disable bf16 autocast during the intervention forward), so that the k=16 case is truly identity.

**Citation:** `run_control_a.py:308` (autocast bf16 during eval), `run_control_a.py:549–557` (smoke tol is 5e-2 in bf16), `run_matrix_codi.py:336–347` (truncate_to_rank).

---

### M5 — The pooled Spearman and "mean accuracy by k" aggregation silently drops seeds that crashed on some ks, producing a biased pooled r.

`run_control_a.py:664–669` builds `by_k` only from `pooled_ks`/`pooled_accs`, which are populated only when `"accuracy" in rec` (L651). If one seed happens to crash on k=1 (e.g., due to SVD numerical failure on a near-zero matrix), that seed contributes to k∈{2,4,8,16} but not k=1. The "pooled mean accuracy by k" for k=1 is then averaged over 2 seeds while k∈{2,4,8,16} are averaged over 3. The `pooled_r` (L658–661) is computed over a mixed-n sample (unequal seed counts per k), which biases r toward whichever k's have more/fewer seeds. A reviewer looking at "pooled r = 0.15, conclusion FLAT" would assume balanced data.

**Fix.** Require ALL seeds to have produced accuracies for ALL ks before computing pooled r. If any seed has any error, either (a) drop that seed entirely from the pool, or (b) drop that k for all seeds, or (c) fail loud with AMBIGUOUS. Log which happened.

**Citation:** `run_control_a.py:650–661`.

---

## Minor attacks

### m1 — `set_seed(seed)` is called once per seed but model weights are deterministic from the checkpoint; the only nondeterminism source is argmax tie-breaking. Document this.

`run_control_a.py:395` calls `set_seed(seed)` before loading each checkpoint. For a greedy-argmax decode, the only non-determinism is (a) argmax tie-breaking (deterministic in PyTorch via `torch.argmax` using stable sort), (b) autocast bf16 operations, and (c) `torch.linalg.svd` on the device (can differ between CUDA versions). This should run-to-run reproducibly on the same H100, but it's worth printing `torch.backends.cuda.matmul.allow_tf32`, `torch.use_deterministic_algorithms`, and the GPU driver version at startup for the paper's supplementary reproducibility appendix.

**Fix.** Add a startup log block that dumps environment hashes.

### m2 — `q_ids` tokenization might differ from Round 4 if the training script special-tokens-added BEFORE creating the dataset vs AFTER.

`ProsQADataset.__init__` (called at `run_control_a.py:408`) uses the *passed-in* tokenizer. If Round 4's `run_vanilla_sft.py` built the dataset with the base GPT-2 tokenizer (no special tokens added) and only added special tokens for the model, Control A's tokenization of `" The answer is:"` could mismatch Round 4's by a single BPE boundary. The `assert_colon_tokenization` guard catches the most common case but not all. 

**Fix.** Compare the tokenization of 3 random ProsQA examples in Control A vs the strings that Round 4's log shows (if any are printed). Add to smoke test.

### m3 — `per_example` records encode `predicted[:96]` (L327), but the prosqa_answer_match runs on the full `predicted_text` (L318). If a pathological case has a 200-char prediction, we can't reconstruct the match from the JSON.

Cosmetic but annoying for post-hoc analysis. Either log full text (uncapped) or log the `first_sentence_final_class` output alongside the truncated predicted text.

**Fix.** Add `predicted_first_class: out["predicted_text"]` stripped via `first_sentence_final_class` to the per_example dict.

### m4 — The SUMMARY.txt `decision:` line is uppercase (L704) but the `classify` function returns lowercase (L484–487). Trivial, but a grep script written against one format will silently miss the other.

**Fix.** Normalize to lowercase throughout. Document the grep target in the JSON schema note.

---

## Steelmanning pass

I tried to kill my three strongest attacks. Here's what survived and what didn't.

### Steelman of F1 ("not the null")

**Counter-argument:** Control A isn't claiming to be a *mechanism-level* null; it's claiming to be a *task-level* null. If vanilla SFT produces a flat rank-k curve, the ablation *method* (SVD truncation at some position) doesn't discriminate on ProsQA. That's the actual claim.

**My response:** No. The paper's framing ("either outcome ships a better paper," QUEUE.md L24) implicitly asserts that a flat Control A curve SUPPORTS rank-1-solvability, and a bending Control A curve VALIDATES matrix-CODI's ablation. But a flat Control A curve is *guaranteed by Lesson 1 / Kill List #4 alone*, regardless of ProsQA's rank-solvability — because Control A's final hidden state goes straight into a linear lm_head, which is rank-blind by linear algebra. So a flat curve tells us nothing new. A bending curve would be surprising but still wouldn't validate matrix-CODI's ablation, because the interventions are at different objects.

**F1 survives pushback.** The experiment as designed cannot generate a diagnostic outcome. It has to be fixed.

### Steelman of F2 ("answer-position surface decoding confound")

**Counter-argument:** The 16x48 and svd_full_768 variants cover ALL of h, not just h[:256]. If rank-ablating all 768 dims doesn't bend the curve, that IS evidence the full residual-stream information at the answer position is rank-1-compressible.

**My response:** Partial validity. The 16x48 and svd_full_768 variants DO address the concern that rank-ablating a small slice is trivially non-informative; they test a bigger slice. But they still intervene ONLY at the answer position. The reviewer will still ask: "What about the information in the preceding context positions that never gets compressed? The attention-weighted sum of prior positions builds the answer-position hidden state. Did any of those prior positions require high rank to form their contribution?" Control A doesn't test this. Testing only the final hidden state reduces the claim to "the answer-position readout is rank-blind" — which is a KNOWN result and is in KILL_LIST.md.

**F2 survives pushback but is narrowed.** The 16x48 and svd_full_768 variants make the experiment *less* uninformative than 16x16 alone would be, but they do not make the experiment diagnostic of rank-1-solvable ProsQA. A fourth variant at an earlier position is required.

### Steelman of M1 ("multi-BPE class words")

**Counter-argument:** The `prosqa_answer_match` function strips to the first sentence and takes the last token. Multi-BPE class words like "bunlion" get detokenized and space-split in Python; BPE boundaries are invisible to the matcher. So multi-BPE is fine for matching, and the single-step intervention is the intended design.

**My response:** Matching is fine. The problem is that the intervention corrupts the first BPE piece of the class word only; the model's un-ablated KV cache + first-piece embedding can reconstruct the target class in the second and third BPE pieces. The single-step intervention is therefore structurally weak at attacking the target — the model gets multiple chances to recover.

**M1 survives pushback.** The fix is to report first-token accuracy as a separate metric (this is cheap). If first-token accuracy bends sharply and full-sequence accuracy is flat, we learn: the intervention DID corrupt the first piece, but the model's context recovered — which is a more nuanced and interesting finding than "flat curve, therefore rank-1-solvable."

### Dropped after pushback

- **Initial concern:** "`torch.use_deterministic_algorithms` not set; runs could differ." *Pushback: H100 + bf16 greedy argmax is deterministic in practice on the same hardware; the audit P2-7 verified dataset ordering is fixed. Dropped to minor m1 (document, don't fail).*
- **Initial concern:** "The 50,260-token embedding table size might mismatch Round 4 if special tokens were added in different order." *Pushback: `assert_colon_tokenization` and the `n_params == 124,439,808` print catch this. Dropped.*
- **Initial concern:** "Why first 256 dims of h? Not middle, not last?" *Pushback: matches matrix-CODI's `w_up: 768->256` (L230). Principled. Dropped.*

---

## Recommended action

### Fixes that MUST land before H100 execution

1. **Fix F1 (mid-sequence propagating intervention).** Add a variant `--variant full_at_bot_analogue` (or similar) that intervenes at a position BEFORE the answer position, using an activation hook so the intervention propagates through remaining attention. This is the only variant that is a valid comparison to matrix-CODI's 6-latent-position intervention. Lead the paper with this variant.
2. **Fix F2 (intervene at earlier position).** Related to #1: the intervention position must be before decoding begins. At minimum, intervene at `len(q_ids) - 1` (last question token, before `" The answer is:"` is appended) and show the rank-k curve. A paper that reports only answer-position ablations will be criticized for the surface-decoding confound.
3. **Land the 3 P0 audit fixes** (checkpoint loader prefix logic, strict=False safety assert, smoke test vanilla accuracy check).
4. **Fix M2 (max_new_tokens=24).** Default must match matrix-CODI.
5. **Fix M4 (add k=None baseline).** Sweep must include un-ablated as reference.
6. **Fix M5 (pooled-r uses only seed-complete data).** Explicit AMBIGUOUS fallback.

### Additional controls the paper should include regardless of outcome

- **Randomize h[:256] control.** Before the ablation sweep, run a single `baseline_shuffled` condition where the first 256 dims of h are replaced with random fp32 of matched norm (on each example). Report accuracy. This establishes the *sensitivity floor* of lm_head to the first 256 dims. If randomization drops accuracy to near-chance, then the first 256 dims matter and rank-k ablation is a meaningful intervention. If randomization barely moves accuracy, then Control A's ablation is intervening on something lm_head mostly ignores — and the flat curve is uninformative.
- **First-token accuracy (separate from full-sequence accuracy).** Report both metrics by k per seed. This addresses M1 and provides a cleaner "purely the intervention's effect" number for the paper.
- **Report singular spectrum of h[:256] reshaped.** Pre-intervention, what IS the effective rank of the fake-Z? If it's already ≈2 at k=16 un-ablated, then k=1 vs k=16 is a non-sequitur because the matrix is already near-rank-2. This is a sanity-check for the experiment's discriminability.

### Scope of the claim Control A can actually support (given the design as-is)

Even after all fixes land, Control A is still a *diagnostic* experiment, not a mechanism-validating one. The most it can ever support is:

- **If flat (after F1+F2 fixes):** "On vanilla GPT-2 SFT for ProsQA, a rank-k activation-patching intervention at a mid-sequence position produces flat accuracy curves across k ∈ {1..16}. This is consistent with either (i) the task being rank-1-solvable at this backbone, or (ii) vanilla SFT's residual stream naturally converging to low-rank representations regardless of task, as predicted by Kobayashi et al. 2024 for weight-decayed attention. Matrix-CODI's flat curves are therefore task-plus-backbone-determined, not objective-specific."
- **If bending (after F1+F2 fixes):** "Vanilla SFT shows a rank-k-discriminating curve, whereas matrix-CODI does not. This isolates the matrix-bottleneck objective as the specific source of rank-indifference."

Either outcome is a meaningful paper edit. Neither supports the stronger original claim ("CODI distillation objective is rank-blind through the full chain rule") beyond what's already in the paper.

**As currently designed (without F1+F2 fixes), Control A can support NEITHER claim and should not be run.** A flat result will be correctly attacked as "of course it's flat, linear lm_head is rank-blind, you just rediscovered Lesson 1." A bending result would be suspicious (contrary to Lesson 1) and would probably indicate a bug in Control A, not a meaningful finding.
