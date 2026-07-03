# Control A v2 — Implementation Notes

Target: `matrix-thinking/scripts/run_control_a.py`. Runs a **propagating**
fake-Z rank-k ablation on Round 4 vanilla GPT-2 SFT ProsQA checkpoints as
the null baseline for the matrix-CODI rank-k ablation.

Original spec: `matrix-thinking/QUEUE.md` PRIORITY 0. Redesigned after
`matrix-thinking/CONTROL_A_ATTACK_REPORT.md` found two fatal protocol
asymmetries in the first implementation.

## 1. Final design

**Intervention strategy: hook-based propagating intervention.** A
`torch.nn.Module.register_forward_hook` is installed on
`model.transformer.h[L_INTERVENE]` (default `L_INTERVENE = 6`, middle of
GPT-2 small's 12 blocks). The hook modifies the block's OUTPUT hidden
states at `N_ANALOG_POSITIONS = 6` target positions by SVD-truncating
their fake-Z reshape to rank `k` and writing the reconstruction back.

**Analog-latent positions.** The 6 token positions immediately preceding
the `:` that terminates `" The answer is:"`. In matrix-CODI, 6 latent
positions sit between `<bot>` and `<eot>` and are then followed by the
tail prompt `" The answer is:"` (see `run_matrix_codi.py:519-548` for the
latent loop and `generate_answer` at L1082-1086 for the tail prompt).
Vanilla SFT has no `<bot>`/`<eot>` scaffold, so the closest positional
analog is "the 6 positions whose residual streams are the last thing the
model aggregates before ":" triggers the answer readout". Function
`_compute_analog_positions` picks `[colon_pos - 6, ..., colon_pos - 1]`.

**Propagation.** Blocks `L_INTERVENE+1 .. 11` re-attend over the modified
residual stream, so the `":"` token and every generated answer token see
the intervention through attention — this is the core property the
attack report demanded. Additionally, we generate with **no KV cache**:
every decoded token triggers a full transformer forward over the growing
sequence, which re-fires the hook at every step. This closes the KV-
cache-leakage hole that the original single-position, cached-decoding
design had.

**Default variant: `16x16`.** The attack report (§F2 steelman) notes
that `16x48` and `svd_full_768` cover more of the hidden state but do not
by themselves address the propagation problem. With propagation now
fixed, `16x16` is the principled primary variant because it matches
matrix-CODI's `d=16` semantics exactly (matrix-CODI's bottleneck produces
a `16x16` Z; cf. `run_matrix_codi.py:230`, `w_up: 768 -> 256`). The other
two variants ship as wired-and-smoke-testable robustness checks.

**Controls shipped in-run:**
- `k=None` unablated baseline (hook disabled, same cache-less decode path).
  Confirms that the script reproduces Round 4's ~81.77% seed-1337 number.
- `k="random"` randomized-h sensitivity floor. Replaces the covered slice
  with i.i.d. Gaussian of matched mean/std. If accuracy stays near 81%,
  the transformer is nearly insensitive to that slice at the analog
  positions and Control A is uninformative regardless of outcome.

## 2. Why this addresses F1 and F2

**F1 (single-position, un-propagated intervention was strictly weaker
than matrix-CODI's mechanism-level intervention).** The original script
intervened only on the post-`ln_f` last-position hidden state right
before `lm_head`. Matrix-CODI intervenes at **6 latent positions whose
values are consumed by downstream attention** and by all subsequent
transformer layers at latent steps `t+1..5` and at the tail. The v2
design mirrors this: 6 positions, intervened at a mid-depth layer, and 6
additional blocks of transformer propagation downstream. The generation
loop runs without a KV cache so downstream generated tokens always see
the intervention through fresh attention over the modified prefix.

**F2 (answer-position surface-decoding confound).** The original design
could only ever tell us "lm_head is rank-blind to h[:256] reshaped to
16x16", which is KILL_LIST Lesson 1 restated. The v2 design intervenes
at positions BEFORE `:`, where the residual stream is still
"question/CoT accumulated state" rather than "almost-decoded next-token
distribution". A flat rank-k curve at these earlier positions is
genuinely informative about the task's rank-requirement at the backbone,
not about the trivial readout property. A bending curve is likewise
meaningful. Either outcome now discriminates between the competing
hypotheses that F2 identified.

## 3. Audit P0/P1 fixes — line-by-line

All citations are to `CONTROL_A_AUDIT_REPORT.md`.

- **P0-1 (loader prefix strip)**: `load_vanilla_sft` now uses a majority
  check (`gpt2_count >= 0.5 * len(keys)`) plus an explicit
  `--checkpoint-format` flag (`"state_dict" | "wrapped" | "codi_wrapped"`).
  The wrapper-unwrap branch checks for sibling keys like `cfg`/`seed`
  before unwrapping, avoiding the "a weight happens to be named `model`"
  trap. See `run_control_a.py:265-295`.
- **P0-2 (silent load + param-count assert)**: After resize we assert
  `sum(p.numel()) == 124_439_808`. After `load_state_dict` we assert no
  critical key is missing (`transformer.wte.weight`, `transformer.ln_f.*`,
  `lm_head.weight`, `transformer.h.0.attn.c_attn.weight`) and that the
  post-load count is unchanged. We also assert the `wte` row count
  matches `len(tokenizer)`. See `run_control_a.py:297-330`.
- **P0-3 (10-example unablated smoke)**: `run_smoke_test` now runs a
  `k=None` decode on the first 10 ProsQA problems and asserts
  `n_correct >= 5`. Per-example predictions are logged. See
  `run_control_a.py:685-710`.
- **P1-1 (NaN Spearman → "flat")**: `classify` now treats NaN `r` as
  unknown, not flat. Combined with the attack M3 binomial rule, a NaN r
  returns `ambiguous` unless the range is definitively small relative to
  the noise floor. See `run_control_a.py:635-675`.
- **P1-2 (KV-cache leakage semantics documented)**: No longer a
  "locked-in behavior" — the v2 design explicitly rejects KV-caching
  during intervention-eval. The JSON payload's `intervention_semantics`
  field spells out the new contract.
- **P1-3 (n=1 or 2 data points)**: `classify` returns `ambiguous`
  immediately when `len(accs) < 3`. See `run_control_a.py:655`.
- **P1-4 (log first-token text)**: `per_example` now includes
  `first_token_id`, `first_token`, `first_token_correct`, and
  `predicted_first_class`. Smoke-test prints a few of these per step.
- **P1-5 (`n_params` over `p.requires_grad`)**: Replaced with
  `sum(p.numel() for p in model.parameters())`. See
  `run_control_a.py:590`.
- **P1-6 (schema versioning)**: Top-level JSON now has
  `schema_version: "control_a_v2"` and `parent_schema` fields.
- **P1-7 (JSON size)**: `per_example` capped to
  `--max-per-example-rows=50` per (seed, k) for JSON; full list remains
  in memory for stats. See `run_control_a.py:545-560`.

## 4. Attack F/M/minor fixes — line-by-line

All citations are to `CONTROL_A_ATTACK_REPORT.md`.

- **F1 (propagating intervention at 6 analog positions)**: core redesign.
  `make_intervention_hook` + `predict_one_propagating`. See
  `run_control_a.py:170-215` (hook) and `run_control_a.py:400-475` (forward).
- **F2 (intervene before the answer position)**: positions picked by
  `_compute_analog_positions` are strictly BEFORE the `:` token.
- **M1 (multi-BPE / first-token accuracy as a separate column)**:
  `_gold_class_first_bpe` returns the set of candidate first-BPE ids for
  the gold class word (both with and without leading space to handle
  GPT-2 BPE space-sensitivity). `eval_one` reports
  `first_token_accuracy` alongside full-sequence accuracy, and
  `classify` is run twice (full-seq + first-token) with both verdicts in
  SUMMARY.txt.
- **M2 (max_new_tokens default 24)**: `--max-new-tokens` default is now
  `24` to match `generate_answer` at `run_matrix_codi.py:1183`.
- **M3 (binomial-aware AMBIGUOUS)**: `classify` uses
  `_binomial_se_pp(n_eff, p_mean)` with
  `n_eff = n_examples_per_k * n_seeds` and returns `ambiguous` whenever
  observed range is below `2 * SE`. The decision rule string in
  `_DECISION_RULE` documents this for the paper-writer.
- **M4 (k=None un-ablated baseline)**: `sweep_one_seed` runs `k=None`
  before the k-sweep. SUMMARY.txt prints per-seed unablated accuracy so
  it can be cross-checked against Round 4's 81.77%.
- **M5 (pooled-r uses only complete ks)**: `main` determines
  `complete_ks = [k for k in args.ks if all seeds have accuracy at k]`
  and pools only over those. Seeds with any missing k are excluded from
  the pooled statistic (logged).
- **Minor m1 (env log)**: Torch/CUDA/TF32/GPU-name printed at startup.
- **Minor m2 (tokenization sanity)**: `_compute_analog_positions` uses
  the tokenizer-emitted `:` id, so any BPE-boundary drift in
  `" The answer is:"` surfaces immediately. Smoke test logs the
  `colon_pos` and `analog_positions` for the first example.
- **Minor m3 (full predicted text in JSON)**: `per_example` stores the
  uncapped `predicted` plus `predicted_first_class`.
- **Minor m4 (lowercase decision)**: SUMMARY.txt and JSON both emit
  `decision:` in lowercase.

## 5. Known limitations deliberately accepted

- **Intervention is at block-output, not input-embedding.** Matrix-CODI's
  intervention is on the continuous embedding that gets fed at the next
  latent position — it's effectively an input-embedding-level
  intervention with full 12-layer downstream processing. Our analog is a
  block-6-output-level intervention with 6-layer downstream processing.
  That's the same "propagating" flavor but not exactly equal depth.
  Claim scope: "a propagating rank-k intervention at a mid-sequence
  position in vanilla SFT produces the following curve." We do NOT
  claim it's mechanistically identical to matrix-CODI's protocol — the
  paper's framing should acknowledge the intervention-depth asymmetry.
- **No-KV-cache decoding is ~24x more expensive per example than the
  cached baseline.** At 500 examples x (1 unablated + 5 k + 1 random) x
  3 seeds, that's ~10.5k full-sequence forwards of ~length ~260. On an
  H100 this is still well under 1 GPU-hour, but worth noting.
- **Single `L_INTERVENE`.** We don't sweep the hook layer. The paper
  should note that a sweep over `L_INTERVENE in {3, 6, 9}` would be
  stronger; queued if time allows.
- **`prosqa_answer_match` is inherited as-is.** Multi-BPE class words
  still get a "class word recovered from first BPE piece" opportunity,
  but now first-token accuracy is reported separately (attack M1).
- **`"random"` control uses Gaussian matched in mean/std, not the exact
  residual-stream distribution of natural hidden states.** That's a
  conservative sensitivity floor; a more careful baseline (e.g., swap
  with a hidden state from a random OTHER example) is queued but not
  run.
- **Spearman over n=5 ks per seed remains noisy.** The binomial-aware
  AMBIGUOUS gate in `classify` is our safeguard.

## 6. Import surface from `run_matrix_codi.py`

All top-level, all import-safe:

- `CONFIG` (module constant)
- `GPT2LMHeadModel`, `GPT2TokenizerFast` (re-exported)
- `ProsQADataset`
- `assert_colon_tokenization`
- `build_logger`
- `effective_rank`
- `prosqa_answer_match`
- `set_seed`
- `truncate_to_rank`

No symbols from `run_matrix_codi` are monkey-patched or modified. If any
of these move inside `main()` or get renamed the import block is the
single place to patch.

## 7. Smoke-test commands

Single-variant smoke (seed 1337, variant 16x16):

```
python3 matrix-thinking/scripts/run_control_a.py \
    --checkpoint-dir /workspace/pebble/round4_vanilla_sft \
    --ckpt-name-template pure_sft_seed{seed}.pt \
    --variant 16x16 \
    --intervene-layer 6 \
    --smoke-test \
    --output-dir experiment-runs/2026-04-23_control_a/smoke_16x16/
```

All-variants smoke (exercises 16x16, 16x48, svd_full_768 end-to-end):

```
python3 matrix-thinking/scripts/run_control_a.py \
    --checkpoint-dir /workspace/pebble/round4_vanilla_sft \
    --ckpt-name-template pure_sft_seed{seed}.pt \
    --smoke-test-all-variants \
    --intervene-layer 6 \
    --output-dir experiment-runs/2026-04-23_control_a/smoke_all/
```

The smoke test asserts: checkpoint loads with correct param count and
critical keys, 10-example unablated accuracy >= 5/10, k=k_max
intervention fires at least once, k=1 effective_rank is near 1, and the
k="random" path runs without crashing.

## 8. Full-sweep command

```
python3 matrix-thinking/scripts/run_control_a.py \
    --checkpoint-dir /workspace/pebble/round4_vanilla_sft \
    --ckpt-name-template pure_sft_seed{seed}.pt \
    --seeds 1337 42 7 \
    --ks 1 2 4 8 16 \
    --variant 16x16 \
    --intervene-layer 6 \
    --n-examples 500 \
    --max-new-tokens 24 \
    --output-dir experiment-runs/2026-04-23_control_a/primary_16x16/
```

Robustness variants — re-run with `--variant 16x48` and
`--variant svd_full_768` into separate output dirs so each sweep has its
own JSON/SUMMARY.txt.
