# Stage 2 Spec: Matrix-CODI Round 2 (ProsQA)

Owner: Stage 3 build agent
Date: 2026-04-11
Input script: `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/run_matrix_codi.py` (1956 lines, Round 1 frozen version)
Goal: Train Matrix-CODI on ProsQA (DAG-shaped logical reasoning) and run the rank-k projection ablation. Pair the resulting rank-k curve with the flat Round 1 curve on GSM8K-Aug to establish that the rank-k ablation measures task structure, not model pathology.

This spec is executable. Every path, every number, every code change is specified. Stage 3 should not invent values.

---

## Section A: File system layout

All data, checkpoints, cache, and run artifacts live on the SSD. The only thing that touches the laptop internal volume (`/Users/...`) is the git-tracked script file that will be edited in-place and then copied into the run directory. Do NOT create data, caches, or checkpoints under `/Users/samuellarson/...`.

Root of this round:

```
/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/
├── STAGE2_SPEC.md                          # this file
├── data/
│   ├── prosqa_train.json                   # 17,886 examples, copied from COCONUT
│   ├── prosqa_valid.json                   # 300 examples
│   └── prosqa_test.json                    # 500 examples
├── tmp/
│   └── coconut_repo/                       # transient clone, deleted after data copy
├── hf_cache/                               # HF_HOME for this round (GPT-2 weights, tokenizer)
├── checkpoints/
│   ├── best_run_b_matrix.pt                # Run B best checkpoint (matrix CODI on ProsQA)
│   └── best_run_a_vanilla.pt               # Run A best checkpoint (only if Run A is executed)
├── results/
│   ├── run_b_matrix/                       # Run B outputs (primary)
│   │   ├── script.py                       # exact run_matrix_codi.py copy that ran
│   │   ├── train.log
│   │   ├── results.json
│   │   ├── rank_dynamics.json
│   │   ├── SUMMARY.txt
│   │   └── best_run_b_matrix.pt            # symlink target of checkpoints/best_run_b_matrix.pt
│   ├── run_c_rank_ablation/                # Run C outputs (rank-k projection)
│   │   ├── script.py
│   │   ├── train.log
│   │   ├── rank_projection_ablation.json
│   │   ├── SUMMARY.txt
│   │   └── analysis_plots/
│   │       ├── accuracy_vs_k.svg
│   │       └── rank_vs_correct.svg
│   └── run_a_vanilla/                      # Run A outputs (OPTIONAL, see Section E)
│       ├── script.py
│       ├── train.log
│       ├── results.json
│       └── SUMMARY.txt
├── logs/
│   ├── orchestrate.log                     # top-level driver output
│   ├── smoke_test.log                      # pre-flight smoke test output
│   └── clone.log                           # COCONUT clone + copy output
└── scripts/
    └── run_matrix_codi.py                  # frozen copy of the round 2 script as run
```

Rules:
- `run_b_matrix/`, `run_c_rank_ablation/`, and `run_a_vanilla/` each hold a full `script.py` copy. The training script already does this via `shutil.copy2(__file__, results_dir / "script.py")` — keep that behavior.
- The single "master" frozen copy in `scripts/run_matrix_codi.py` is the version Stage 3 writes. It should be identical to whatever sits inside each `results/*/script.py`.
- Checkpoints are saved by the training script into `results_dir`, so Run B writes `results/run_b_matrix/best_run_b_matrix.pt` directly. The `checkpoints/` directory at the top level is optional convenience — Stage 3 MAY create symlinks there but does not have to.
- Nothing under `/Users/samuellarson/...` should be created, modified, or cached during this experiment except for the in-place edits to `run_matrix_codi.py` in the git repo.
- HF cache: set `HF_HOME=/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/hf_cache` in the environment before any Python invocation.

---

## Section B: Dataset extraction

Clone URL: `https://github.com/facebookresearch/coconut`
Commit pinning: checkout `main` branch; record the HEAD SHA in `logs/clone.log` for reproducibility. If Stage 3 has concerns about stability, pin to whichever SHA is HEAD of main at clone time and note it in `logs/clone.log`.

Commands, in order:

```bash
# 1. Set working dir
cd /Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa
mkdir -p data tmp hf_cache checkpoints results/run_b_matrix results/run_c_rank_ablation results/run_a_vanilla logs scripts

# 2. Clone COCONUT into tmp/
git clone --depth 1 https://github.com/facebookresearch/coconut.git tmp/coconut_repo 2>&1 | tee logs/clone.log
cd tmp/coconut_repo
git rev-parse HEAD | tee -a ../../logs/clone.log
cd ../..

# 3. Verify the three ProsQA files exist at the expected paths
ls -la tmp/coconut_repo/data/prosqa_train.json \
        tmp/coconut_repo/data/prosqa_valid.json \
        tmp/coconut_repo/data/prosqa_test.json 2>&1 | tee -a logs/clone.log

# 4. Copy the three files into data/
cp tmp/coconut_repo/data/prosqa_train.json data/prosqa_train.json
cp tmp/coconut_repo/data/prosqa_valid.json data/prosqa_valid.json
cp tmp/coconut_repo/data/prosqa_test.json  data/prosqa_test.json

# 5. Sanity check record counts and schema
python3 -c "
import json
for split in ['train', 'valid', 'test']:
    path = f'data/prosqa_{split}.json'
    with open(path) as f:
        rows = json.load(f)
    print(split, len(rows), 'example keys:', list(rows[0].keys()))
    assert 'question' in rows[0] and 'answer' in rows[0] and 'steps' in rows[0], \
        f'schema mismatch in {path}: {list(rows[0].keys())}'
" 2>&1 | tee -a logs/clone.log
# Expected output lines (approx):
#   train 17886 example keys: ['question', 'answer', 'steps', 'idx_to_symbol', 'edges', 'root', 'target', 'neg_target']
#   valid 300   example keys: [...]
#   test  500   example keys: [...]

# 6. Delete the clone (keeps only the three JSONs under data/)
rm -rf tmp/coconut_repo
```

If step 3 fails (files not at that path), Stage 3 MUST stop and re-read `tmp/coconut_repo/README.md` and `tmp/coconut_repo/data/` to find the actual locations, then report back. Do not guess. The Stage 1 research report specifies `data/prosqa_{train,valid,test}.json` — trust that first but verify.

File size sanity: each JSON should be on the order of a few MB (train is largest, a few tens of MB). If any file is < 100 KB or > 500 MB, stop and report.

---

## Section C: Adapter design

### Chosen option: Option 2 — add a new dataset class, keep GSM8K adapter untouched

Justification: The GSM8K-Aug adapter is Round 1's reproducibility anchor — CLAUDE.md explicitly says to prefer adding a new code path over modifying the GSM8K adapter. ProsQA has a fundamentally different answer format ("Tom is a zhorpus.") with no colon, so Option 1 (rewriting the data to append " The answer is: ...") would require mutating the reference data AND would also change the token that carries the answer semantics (from a number to a word). Option 3 (finding a consistent delimiter in ProsQA) is fragile because there is no guaranteed delimiter in the ProsQA answer strings. Option 2 adds a parallel class, keeps GSM8K behavior frozen, and localizes all the ProsQA-specific logic.

Critically, for ProsQA we still construct a CODI-style tail of the form `"<eot> The answer is: {answer}"` and anchor the KD alignment on the `:` token in that synthesized tail — this matches the Round 1 KD behavior precisely. The raw ProsQA answer string (e.g. `"Tom is a zhorpus."`) is placed AFTER the colon, so the L1 distillation anchor is identical in semantics to GSM8K. The only thing the adapter does differently is load from a local JSON file and supply the multi-token answer string.

### Code changes to `run_matrix_codi.py`

All line numbers below reference the current version of `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/run_matrix_codi.py` (1956 lines). Stage 3 MUST re-read the file before editing to confirm line ranges have not drifted.

#### Change 1 — CONFIG additions (insert after line 108, before `# MATRIX PRIMITIVES` block)

Add these keys to `CONFIG`:

```python
    # Dataset selection
    "dataset": "gsm8k_aug",   # {"gsm8k_aug", "prosqa"}

    # ProsQA paths (used only when dataset == "prosqa")
    "prosqa_train_path": "/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/data/prosqa_train.json",
    "prosqa_val_path":   "/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/data/prosqa_test.json",
    # Note: we use prosqa_test (500 examples) for eval rather than prosqa_valid (300).
    # Test set is larger and reduces variance on the accuracy signal. Valid set is
    # unused in this round; keep the path in Section A for completeness only.
```

Default stays `"gsm8k_aug"` so any re-run of Round 1 still works unchanged.

#### Change 2 — Add `ProsQADataset` class (insert a new class after `GSM8KDataset.__getitem__`, i.e. after line 684, before `def collate_gsm8k(...)` on line 687)

Spec for the new class:

```
class ProsQADataset(Dataset):
    """Tokenized ProsQA examples for the CODI teacher-student setup.

    Same item schema as GSM8KDataset so the existing collate_gsm8k + compute_codi_loss
    code can consume it unchanged:
      - teacher_ids: "[question] [chain-of-thought steps] The answer is: [answer]"
      - q_ids:       "[question]"
      - tail_ids:    "<eot> The answer is: [answer]"
      - teacher_colon_idx: last ":" index in teacher_ids (matches GSM8K behavior)
      - tail_colon_rel:    first ":" index in tail_ids
      - answer_text: raw ProsQA answer string (multi-word sentence)
      - reasoning_steps: len(row["steps"])
      - question: raw question text
    """

    def __init__(self, split, tokenizer, cfg, special_ids):
        import json
        assert split in ("train", "val"), f"split must be train|val, got {split}"
        if split == "train":
            path = cfg["prosqa_train_path"]
        else:
            path = cfg["prosqa_val_path"]
        with open(path) as f:
            raw_rows = json.load(f)

        self.items = []
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.special_ids = special_ids

        n_total = 0
        n_dropped_colon = 0
        n_dropped_len = 0
        answer_prefix = " The answer is:"
        colon_id = tokenizer.encode(":", add_special_tokens=False)[0]

        for row in raw_rows:
            n_total += 1
            question = row["question"].strip()
            steps = row.get("steps", [])
            # Join ProsQA steps into a single CoT string with periods preserved.
            cot = " ".join(str(s).strip() for s in steps).strip()
            answer_str = str(row["answer"]).strip()

            # Teacher text: question + CoT + " The answer is: <full ProsQA answer>"
            teacher_text = f"{question}\n{cot}{answer_prefix} {answer_str}"
            teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)

            q_ids = tokenizer.encode(question + "\n", add_special_tokens=False)

            tail_text = f"{answer_prefix} {answer_str}"
            tail_token_ids = tokenizer.encode(tail_text, add_special_tokens=False)
            tail_ids = [special_ids["eot"]] + tail_token_ids

            # Locate the FIRST ":" in the tail (right after "answer is").
            tail_colon_rel = None
            for i, tok in enumerate(tail_ids):
                if tok == colon_id:
                    tail_colon_rel = i
                    break
            if tail_colon_rel is None:
                raise AssertionError(
                    f"Could not find colon in ProsQA tail: {tail_text!r}"
                )

            # Locate the LAST ":" in the teacher sequence (matches GSM8KDataset).
            teacher_colon_idx = None
            for i, tok in enumerate(teacher_ids):
                if tok == colon_id:
                    teacher_colon_idx = i
            if teacher_colon_idx is None:
                n_dropped_colon += 1
                continue

            # Length budgets. ProsQA avg is ~242 tokens with a long tail on 5-6 step
            # examples; keep the GSM8K budget of max_total_len=768 which is ample.
            if len(q_ids) > cfg["max_q_len"]:
                q_ids = q_ids[: cfg["max_q_len"]]
            if len(tail_ids) > cfg["max_ans_len"] + 16:
                # ProsQA answers can be up to ~10 BPE tokens — give them a bit more slack.
                keep = cfg["max_ans_len"] + 16
                tail_ids = tail_ids[:keep]
                if tail_colon_rel >= keep:
                    n_dropped_len += 1
                    continue

            if len(teacher_ids) > cfg["max_total_len"]:
                n_dropped_len += 1
                continue

            self.items.append({
                "teacher_ids": teacher_ids,
                "teacher_colon_idx": teacher_colon_idx,
                "q_ids": q_ids,
                "tail_ids": tail_ids,
                "tail_colon_rel": tail_colon_rel,
                "answer_text": answer_str,
                "reasoning_steps": len(steps) if isinstance(steps, list) else 0,
                "question": question,
            })

        n_kept = len(self.items)
        pct = 100.0 * n_kept / max(n_total, 1)
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        if rank == 0:
            print(
                f"[ProsQADataset split={split}] kept {n_kept}/{n_total} ({pct:.1f}%) "
                f"| dropped: colon={n_dropped_colon} len={n_dropped_len}",
                flush=True,
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
```

Note: the class is deliberately isomorphic to `GSM8KDataset`. Same item keys, same collate compatibility, same KD anchor. Do not unify them — the GSM8K class is frozen for reproducibility.

#### Change 3 — Replace GSM8K dataset construction in `train_run` (lines 1435-1437)

Current code:

```python
    train_ds = GSM8KDataset("train", tokenizer, cfg, special_ids)
    val_ds = GSM8KDataset("test", tokenizer, cfg, special_ids)
```

New code:

```python
    if cfg["dataset"] == "prosqa":
        train_ds = ProsQADataset("train", tokenizer, cfg, special_ids)
        val_ds = ProsQADataset("val", tokenizer, cfg, special_ids)
    else:
        train_ds = GSM8KDataset("train", tokenizer, cfg, special_ids)
        val_ds = GSM8KDataset("test", tokenizer, cfg, special_ids)
```

#### Change 4 — Replace GSM8K dataset construction in `eval_rank_projection` (line 1698)

Current code:

```python
    val_ds = GSM8KDataset("test", tokenizer, saved_cfg, model.special_token_ids)
```

New code:

```python
    if saved_cfg.get("dataset", "gsm8k_aug") == "prosqa":
        # Patch the saved config with Round 2 ProsQA paths in case the checkpoint
        # was saved before the paths were added. Prefer the passed-in cfg's paths
        # if present (Run C may be launched from a different working directory).
        saved_cfg.setdefault("prosqa_val_path", cfg["prosqa_val_path"])
        val_ds = ProsQADataset("val", tokenizer, saved_cfg, model.special_token_ids)
    else:
        val_ds = GSM8KDataset("test", tokenizer, saved_cfg, model.special_token_ids)
```

#### Change 5 — Update the `evaluate_gsm8k` function to be dataset-agnostic (the function name stays for backward compat; it's already general)

The function `evaluate_gsm8k` (line 1039) currently calls `parse_predicted_number` and `numbers_match` which only work for numeric answers. ProsQA answers are strings like "Tom is a zhorpus."

Replace the correctness check in `evaluate_gsm8k` at lines 1071-1072:

Current:
```python
        pred_num = parse_predicted_number(pred_text)
        is_correct = numbers_match(pred_num, item["answer_text"])
```

New:
```python
        if cfg.get("dataset", "gsm8k_aug") == "prosqa":
            is_correct = prosqa_answer_match(pred_text, item["answer_text"])
        else:
            pred_num = parse_predicted_number(pred_text)
            is_correct = numbers_match(pred_num, item["answer_text"])
```

And add a new helper function immediately after `numbers_match` (after line 1035):

```python
def prosqa_answer_match(pred_text, gold_text):
    """ProsQA binary-discrimination match.

    ProsQA asks 'Is Tom a X or Y?' and the gold is a full sentence 'Tom is a X.'
    We extract the final content word from both strings (lowercased, stripped of
    punctuation) and compare. This is tolerant of the model producing just the
    class word vs the full sentence.
    """
    import re as _re
    def final_word(s):
        s = s.strip().lower().rstrip(".").rstrip("?").rstrip("!")
        tokens = _re.split(r"\s+", s)
        return tokens[-1] if tokens else ""
    return final_word(pred_text) == final_word(gold_text)
```

This is lenient-correct: if the model generates "zhorpus" OR "Tom is a zhorpus" the match returns True. Tight matching would require the full sentence which the model may not emit verbatim at our tiny scale.

#### Change 6 — `generate_answer` max_new tokens bump (line 991)

Current:
```python
    max_new = 16
```

New:
```python
    max_new = 16 if cfg.get("dataset", "gsm8k_aug") != "prosqa" else 24
```

Rationale: ProsQA answers are up to ~10 BPE tokens ("Tom", " is", " a", " zh", "or", "pus", "."), so 16 is cutting it close. 24 gives safety margin without appreciably slowing eval.

Also: `generate_answer` currently takes `cfg` as a positional argument (line 873) — verify it is passed through. It is: line 1067 passes `cfg`. No wiring needed.

#### Change 7 — CLI flags

In `main()` (line 1875), add two new argparse flags after the existing `--max-eval-batches` flag (after line 1918):

```python
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["gsm8k_aug", "prosqa"],
        help="Override CONFIG['dataset']. Default 'gsm8k_aug' preserves Round 1.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override CONFIG['warmup_steps'].",
    )
```

And wire them into the cfg override block (after line 1929, before `if args.smoke_test`):

```python
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.warmup_steps is not None:
        cfg["warmup_steps"] = args.warmup_steps
```

#### Change 8 — Update the `GSM8K` references in logged messages to be dataset-aware

Cosmetic but important for log clarity. In `evaluate_gsm8k` and in the training loop log message "GSM8K accuracy: ..." (line 1561), change to:

```python
                    logger.log(f"  {cfg['dataset']} accuracy: {acc*100:.2f}%")
```

And in SUMMARY.txt generation (line 1622):

```python
            f"  Best {cfg['dataset']} accuracy: {best_acc*100:.2f}%",
```

Leave the function name `evaluate_gsm8k` alone — renaming it is churn for no gain.

#### Change 9 — Smoke test coverage for ProsQA adapter

Add a sixth smoke test at the bottom of `run_smoke_tests` (before the final "ALL SMOKE TESTS PASSED" print, around line 1328). The test must:

1. Monkeypatch CONFIG with `dataset="prosqa"` and point `prosqa_train_path` / `prosqa_val_path` at a tiny synthetic JSON file written to `/tmp/prosqa_smoke/`.
2. The synthetic JSON must contain 4 rows with the ProsQA schema (`question`, `answer`, `steps`).
3. Construct `ProsQADataset("train", ...)` and assert `len(ds) == 4`, then check that `ds[0]` has all the expected keys.
4. Run `prosqa_answer_match("Tom is a zhorpus.", "Tom is a zhorpus.")` → True
5. Run `prosqa_answer_match("zhorpus", "Tom is a zhorpus.")` → True
6. Run `prosqa_answer_match("Tom is a fompus.", "Tom is a zhorpus.")` → False
7. Delete the synthetic files.

Print a "[6/6] ProsQA adapter smoke test ... PASS" marker.

### What is NOT changing

- `compute_codi_loss` — unchanged. The KD alignment still fires at the `:` token which is now the colon in `" The answer is:"` of the synthesized teacher/tail texts. This works for ProsQA because the tail is synthesized with the same prefix.
- `collate_gsm8k` — unchanged. Both datasets produce items with identical key shapes.
- `MatrixBottleneck`, `MultiplicativeThinkingLayer`, all rank logic — unchanged. This is Round 2's point: same model, different data.

---

## Section D: Training configuration

ProsQA is 17,886 training examples. GSM8K-Aug is ~386k. Ratio ≈ 21.6×.

### Step-count calibration

Round 1 Run B used: `batch_per_gpu=16 × 8 GPUs = 128 global batch` (Wait — spec says Round 1 used 32/GPU × 8 = 256 global. Re-checking: the Round 1 SUMMARY confirms `epochs=5, global batch 256`. This means Round 1 overrode the default 16 via `--batch-size 32`.) Round 1 Run B ran 5 epochs × (386000 / 256) ≈ 7530 total steps. Wall time 30.5 min.

For Round 2 to hit a similar ~7500 total steps we need:

```
steps/epoch = 17886 / global_batch
total_steps = epochs × 17886 / global_batch
```

Targets:
- Global batch = 128 (batch_per_gpu=16 × 8). Steps/epoch = 139.7 → 139. Epochs needed for 7500 steps: ~54.
- Global batch = 64  (batch_per_gpu=8  × 8). Steps/epoch = 279.5 → 279. Epochs needed: ~27.
- Global batch = 256 (batch_per_gpu=32 × 8). Steps/epoch = 69.9  → 69.  Epochs needed: ~109.

54 epochs of GPT-2 124M on 17.9k examples risks catastrophic overfitting (model memorizes the train set in <10 epochs at this size). 109 epochs is even worse.

### Chosen configuration

Use **batch_per_gpu = 16** (global batch = 128) and **epochs = 25**. This gives:
- Total steps: 25 × 139 = **3475 steps**
- Wall time estimate: `(3475 / 7530) × 30.5 min ≈ 14 min` on 8×H100 (roughly — ProsQA sequences are shorter on average than GSM8K-Aug which used max_total_len=768 heavily, so expect closer to 12-14 min).

Fewer than Round 1's 7530 steps, but this is intentional: the 17.9k dataset does not support longer training without overfit. 25 epochs on 17.9k examples is 447,150 example-presentations, comparable to GSM8K-Aug's 5 epochs × 386k = 1.93M presentations (factor ~4× fewer). If Round 2 Run B underperforms baseline, the first retry is to DOUBLE epochs to 50 (see Section G).

### Full hyperparameter table (deltas from Round 1 in bold)

| Param                     | Round 1 (GSM8K-Aug) | Round 2 (ProsQA) | Reason for change                       |
|---------------------------|---------------------|------------------|-----------------------------------------|
| dataset                   | gsm8k_aug           | **prosqa**       | New task                                |
| base_model                | gpt2                | gpt2             | unchanged                               |
| n_latents                 | 6                   | 6                | unchanged                               |
| mat_dim                   | 16                  | 16               | unchanged                               |
| use_thinking_iter         | True                | True             | unchanged                               |
| alpha/beta/gamma          | 1.0 / 1.0 / 1.0     | 1.0 / 1.0 / 1.0  | unchanged                               |
| lr                        | 1e-4                | 1e-4             | unchanged — same budget regime          |
| weight_decay              | 0.01                | 0.01             | unchanged                               |
| betas                     | (0.9, 0.98)         | (0.9, 0.98)      | unchanged                               |
| grad_clip                 | 1.0                 | 1.0              | unchanged                               |
| warmup_steps              | 100                 | **50**           | Fewer total steps — halve warmup so LR reaches peak in the first ~1.5 epochs |
| batch_size_per_gpu        | 32                  | **16**           | Smaller dataset → smaller batch keeps gradient noise usable. Also fits snugly in VRAM. |
| epochs                    | 5                   | **25**           | 5× more epochs to compensate for 21× less data |
| max_q_len                 | 256                 | 256              | unchanged                               |
| max_ans_len               | 32                  | 32               | unchanged                               |
| max_total_len             | 768                 | 768              | unchanged — ProsQA avg 242 tokens       |
| eval_interval_epochs      | 1                   | 1                | unchanged                               |
| max_eval_batches          | 50                  | **32**           | 500-problem ProsQA test set ÷ 16 per-batch eval = ~32 batches; eval the full test set |
| eval_batch_size           | 16                  | 16               | unchanged                               |
| max_rank_samples          | 50                  | 50               | unchanged                               |
| log_interval              | 50                  | 50               | unchanged                               |
| seed                      | 1337                | 1337             | unchanged                               |

Sanity: `(17886 / (16 × 8)) × 25 = 3490.something` steps. Log_interval=50 gives ~70 log lines, plenty for a human to scan.

Global batch = 128 vs Round 1's 256: lr stays at 1e-4 (not scaled down by 2×) because (a) the `lr × steps` budget is already lower and (b) CODI's original recipe uses 1e-4 with similar global batches. Do NOT auto-scale lr.

### Eval cadence

Eval at end of every epoch = 25 evals. Each eval runs 500 problems × ~0.5 sec/problem ≈ 4-5 min. That's 25 × 5 = 125 min of eval on top of 14 min training → total wall time is DOMINATED by eval.

Mitigation: cap `max_eval_batches` at **8** during training (≈128 problems) and do a FINAL full-test eval once, after training ends, with `max_eval_batches=32`. This reduces intermediate eval cost to ~25 × 1 min = 25 min. Add an extra cfg key `final_eval_batches = 32` and a small block at the end of `train_run` that runs one more `evaluate_gsm8k(max_batches=cfg["final_eval_batches"])` and reports that as `best_acc_final`. Add this to Change 3's scope.

Revised plan: Stage 3 should implement this final-eval hook as part of its code changes. Insert after line 1601 (after the `finally: dist.barrier()` block but before "Final summary"):

```python
    # Round 2: one full-test eval at the end on rank 0, with the larger cap.
    try:
        if is_main and cfg.get("dataset") == "prosqa":
            logger.log("--- Final full-test eval ---")
            final_batches = cfg.get("final_eval_batches", cfg["max_eval_batches"])
            acc_final, _, _ = evaluate_gsm8k(
                model_ddp, val_ds, tokenizer, special_ids, cfg, device,
                save_ranks=False, max_batches=final_batches,
            )
            logger.log(f"  Final {cfg['dataset']} accuracy (full test): {acc_final*100:.2f}%")
            training_curve.append({
                "epoch": cfg["epochs"],
                "step": step,
                "accuracy": acc_final,
                "train_loss_avg50": float(sum(_loss_window) / max(len(_loss_window), 1)),
                "final_eval": True,
            })
    finally:
        dist.barrier()
```

Add the corresponding CONFIG key near line 97:

```python
    "final_eval_batches": 32,
```

And set the RUNTIME `max_eval_batches` override to 8 via the CLI flag when launching Run B (see Section H checklist).

---

## Section E: Runs to execute

### Run B — Matrix CODI on ProsQA (REQUIRED, primary)

```bash
HF_HOME=/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/hf_cache \
torchrun --standalone --nproc_per_node=8 \
  /root/run_matrix_codi.py \
  --mode train_matrix \
  --dataset prosqa \
  --results-dir /root/results/run_b_matrix \
  --batch-size 16 \
  --epochs 25 \
  --warmup-steps 50 \
  --max-eval-batches 8
```

Path note: when launched on the H100 pod, `/root/run_matrix_codi.py` is wherever Stage 3 rsyncs the script to. The final results in `/root/results/run_b_matrix/` are then rsynced back to `/Volumes/1TB_SSD/.../results/run_b_matrix/`. Stage 3 decides rsync mechanics; this spec just demands that the final artifacts end up at the SSD path from Section A.

### Run C — Rank-k projection ablation on Run B checkpoint (REQUIRED)

```bash
HF_HOME=/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/hf_cache \
python3 /root/run_matrix_codi.py \
  --mode eval_rank_projection \
  --dataset prosqa \
  --checkpoint /root/results/run_b_matrix/best_run_b_matrix.pt \
  --results-dir /root/results/run_c_rank_ablation \
  --max-eval-batches 32
```

Single GPU (no torchrun). Wall time estimate: ~5 k values × 500 problems × 0.5 sec = ~20-25 min total. Make sure only one GPU is visible (`CUDA_VISIBLE_DEVICES=0`).

### Run A — Vanilla CODI on ProsQA baseline (OPTIONAL, recommended)

**Decision: YES, run it.** Even though we have Run A on GSM8K-Aug from Round 1, adding a vanilla baseline on ProsQA is cheap and directly answers the question "does the matrix bottleneck even help on this task?". Without it, a low Run B accuracy could be blamed on dataset difficulty rather than on the bottleneck being vestigial. The marginal cost is ~14 minutes on 8×H100.

Only skip Run A if the pod budget is extremely tight or if Run B shows a clearly positive rank-k curve in Run C — in which case the paired story is already complete.

```bash
HF_HOME=/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/hf_cache \
torchrun --standalone --nproc_per_node=8 \
  /root/run_matrix_codi.py \
  --mode train_vanilla \
  --dataset prosqa \
  --results-dir /root/results/run_a_vanilla \
  --batch-size 16 \
  --epochs 25 \
  --warmup-steps 50 \
  --max-eval-batches 8
```

### Run order

1. Smoke test first (Section H step 7).
2. Run B (critical path).
3. Run C (depends on Run B checkpoint).
4. Run A (independent of B/C; can run after B while C is also running if GPU is free, but simpler to serialize: run A after C).

---

## Section F: Success criteria

### Reference points

- COCONUT paper reports ~82% accuracy on ProsQA for their best model, and ~54% for explicit CoT GPT-2 fine-tuning (per Stage 1 summary). Our matrix-CODI on GPT-2 124M is closer to the "explicit CoT GPT-2" regime than to the full COCONUT setup.
- Round 1 Run B reached 6.25% on GSM8K (very weak absolute number — the experiment was set up to study rank structure, not compete on accuracy).

### Round 2 accuracy expectations

- Lower bound to consider the training run valid at all: **Run B ≥ 15% on ProsQA test**. Anything below 15% on a binary-discrimination task (majority class ~50%) suggests the model never learned the task, and the rank-k curve is meaningless.
- Target for "clean" paired result: **Run B ≥ 35%**. At this level the model is clearly doing something task-relevant and the rank-k ablation is interpretable.
- Strong result: **Run B ≥ 50%**. Matches or beats the explicit-CoT GPT-2 baseline. Would be a publishable bullet.

If Run A (vanilla) scores within 3 percentage points of Run B, the bottleneck is contributing nothing to the task and the rank-k structure is an artifact — report honestly.

### Rank-k curve shape

The central hypothesis: DAG-shaped reasoning stresses orthogonal latent directions and produces a **steep** rank-k curve. Operationally:

- **Positive result (hypothesis supported):** accuracy at k=1 is ≥ 5 percentage points BELOW accuracy at k=16, and the curve is monotonically increasing in k (or nearly so). Ideally k=1 < k=2 < k=4 < k=8 ≤ k=16. A >10 pp gap between k=1 and k=16 is strongly positive.
- **Negative result (hypothesis refuted):** accuracy at k=1 within 2 pp of k=16. The matrix is vestigial on ProsQA just like on GSM8K — the bottleneck is a reshape trick and nothing more. Document and move on.
- **Ambiguous:** k=1 drops 2-5 pp from k=16. Report both; re-run with more epochs to see if the gap widens.

The PAIRED claim only works if:
(a) Round 1 (GSM8K-Aug) rank-k curve was flat (confirmed).
(b) Round 2 (ProsQA) rank-k curve is steep OR flat with an explanation.

A negative Round 2 result is still publishable as "rank-k ablation is architecture-limited, not task-limited" — but we need to be sure it's not a training/hyperparam artifact before concluding that.

### Training signal floor

If after 25 epochs Run B's train loss has not dropped below 2.5 (nats, on the answer-only tokens), something is wrong with the adapter or KD alignment — stop and debug instead of trusting the eval numbers. Round 1 Run B train loss at the end was in the 1.5-2.0 range on GSM8K-Aug; ProsQA answers are shorter so we should see similar or lower.

---

## Section G: Risks and fallbacks

### Risk 1 — 17,886 examples is too few for GPT-2 124M to converge on this task

**Signs:** Run B train loss plateaus above 3.0; eval accuracy stays near 50% (chance) across all epochs.

**Fallback:** Increase epochs from 25 to 50 and re-run. If still no signal, use the ProsQA test set of 500 as additional training signal (keep valid-set 300 as eval) and go to 50 epochs. If still no signal, generate additional examples from ProntoQA per Stage 1 fallback: `--deduction-rule OrElim --proof-width 3 --min-hops 5 --max-hops 5`, target ~20k extra examples. Document the data mix clearly in `results.json`.

### Risk 2 — Rank curve is also flat on ProsQA (the "matrix is always vestigial" outcome)

**Signs:** k=1 and k=16 accuracies within 1 pp, consistent with Round 1.

**Response:** This is a valid negative result. Document in EXPERIMENT_LOG.md with the paired GSM8K + ProsQA curves side by side. Conclude that a single d×d bottleneck at six fixed latent positions does not learn task-dependent latent rank structure for either arithmetic OR DAG reasoning at GPT-2 124M scale. Possible next directions: (a) larger d (32, 48, 64), (b) more latent positions, (c) per-step adaptive rank. These are for Round 3, not Round 2.

Do NOT quietly re-tune hyperparameters until the curve becomes steep. Honest negative result wins over manufactured positive result.

### Risk 3 — Answer format requires a deeper rewrite than the adapter currently plans

**Signs:** Smoke test passes but training loss is high and flat from step 1, OR the KD term `L_kd` is wildly larger than teacher/student CE terms.

**Diagnosis:** The `:` anchor may be pointing at a wrong token (e.g., if "answer is" tokenizes differently when followed by a ProsQA-style answer). Print the first training example's teacher_ids, tail_ids, teacher_colon_idx, tail_colon_rel and decoded tokens at those positions to verify they land on the colon after "answer is".

**Fallback:** If the colon anchor is wrong, the simplest fix is to synthesize the tail so the colon is the only colon in the string. The Change 2 adapter already does this — the tail is `"<eot> The answer is: {answer}"` and the ProsQA answer "Tom is a zhorpus." contains no colon, so there's only one colon and the alignment is unambiguous. This risk is lower than it looks.

### Risk 4 — Eval wall time blows up and the pod is billed for idle

**Signs:** Intermediate evals take >2 min each, total run time >1 hour.

**Mitigation:** Already baked in — `max_eval_batches=8` during training keeps intermediate eval bounded to ~128 problems. The final full eval runs once. If even this is too slow, drop to `max_eval_batches=4` and keep only the final full eval.

### Risk 5 — Tokenizer handles made-up ProsQA words badly and inflates sequence length past max_total_len

**Signs:** `ProsQADataset` drop log shows `dropped: len=N` with N > 5% of the data.

**Response:** Bump `max_total_len` from 768 to 1024 in a cfg override and re-run. GPT-2's context is 1024 so this is the hard ceiling.

---

## Section H: Checklist for the build agent

Execute these in order. Check off each step; log any deviations in `logs/orchestrate.log`.

1. **Set working dir and env**
   - `cd /Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa`
   - Verify the directory exists (Stage 2 created it). If not, `mkdir -p`.
   - Create subdirs per Section A: `data/`, `tmp/`, `hf_cache/`, `checkpoints/`, `results/run_b_matrix/`, `results/run_c_rank_ablation/`, `results/run_a_vanilla/`, `logs/`, `scripts/`.

2. **Clone COCONUT and extract ProsQA JSONs**
   - Follow Section B commands exactly.
   - Validate schema (question, answer, steps keys present).
   - Validate row counts: 17886, 300, 500.
   - Delete `tmp/coconut_repo/`.
   - Sanity-check: `ls -lh data/prosqa_*.json` shows 3 files, each between 100 KB and 500 MB.

3. **Apply the code changes from Section C to `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/run_matrix_codi.py`**
   - Re-read the file first to confirm line numbers match (Stage 2 reviewed at the version as of 2026-04-11).
   - Apply Change 1 (CONFIG additions), Change 2 (new ProsQADataset class), Change 3 (train_run dataset switch), Change 4 (eval_rank_projection dataset switch), Change 5 (evaluate_gsm8k + prosqa_answer_match helper), Change 6 (max_new bump), Change 7 (CLI flags), Change 8 (log message cosmetics), Change 9 (smoke test).
   - Also apply the final-eval hook from Section D (training loop end).
   - Add `CONFIG["final_eval_batches"] = 32`.

4. **Static review**
   - Run `python3 -m py_compile /Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/run_matrix_codi.py`. Fix any syntax errors.
   - Visually diff against git HEAD to confirm ONLY the changes from Section C are present.

5. **Copy script to the round directory**
   - `cp /Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/run_matrix_codi.py /Volumes/1TB_SSD/learned-representations/experiment-runs/2026-04-11_round2_prosqa/scripts/run_matrix_codi.py`

6. **Launch the smoke test** (on the laptop CPU OR on the H100 pod — CPU is fine for correctness, slower)
   - `HF_HOME=/Volumes/1TB_SSD/.../hf_cache python3 /Volumes/1TB_SSD/.../scripts/run_matrix_codi.py --smoke-test 2>&1 | tee /Volumes/1TB_SSD/.../logs/smoke_test.log`
   - All 6 tests must PASS. If any fail, fix before proceeding.

7. **Spin up the H100 pod** (user action, not agent action — the pod is currently OFF)
   - Agent should STOP here and report ready. User will start the pod.

8. **Rsync the script and data to the pod**
   - `rsync` script to `/root/run_matrix_codi.py`
   - `rsync` `data/prosqa_*.json` to `/root/data/`
   - Adjust `prosqa_train_path` / `prosqa_val_path` in the launch command or via a `CONFIG` patch script. (Simpler: edit the CONFIG in the pod's copy of the script to point at `/root/data/prosqa_train.json` and `/root/data/prosqa_test.json` before launch.)

9. **Run the smoke test on the pod** (single-GPU)
   - `python3 /root/run_matrix_codi.py --smoke-test 2>&1 | tee /root/smoke_test_pod.log`
   - Must PASS. Rsync the log back.

10. **Launch Run B**
    - Execute the Run B command from Section E.
    - Tail the log: `tail -f /root/results/run_b_matrix/train.log` until `=== DONE ===`.
    - Expected wall time: ~14-20 min on 8×H100. If it exceeds 45 min, something is wrong — check eval cost.

11. **Rsync Run B artifacts back to SSD**
    - `/root/results/run_b_matrix/` → `/Volumes/1TB_SSD/.../results/run_b_matrix/`
    - Verify: `SUMMARY.txt`, `results.json`, `rank_dynamics.json`, `best_run_b_matrix.pt`, `script.py`, `train.log` all present.

12. **Launch Run C**
    - Execute the Run C command from Section E (single GPU).
    - Expected wall time: ~20-30 min.
    - Rsync back.

13. **Launch Run A** (optional, see Section E — recommended)
    - Execute the Run A command.
    - Expected wall time: ~14-20 min.
    - Rsync back.

14. **Generate a paired-comparison summary**
    - Write `/Volumes/1TB_SSD/.../results/ROUND2_SUMMARY.md` with:
      - Round 1 rank-k curve (copied from `2026-04-10_matrix_codi_round1/results/run_c_rank_ablation/rank_projection_ablation.json`)
      - Round 2 rank-k curve (from Run C)
      - Side-by-side table of k vs accuracy for both rounds
      - Run B best accuracy, Run A best accuracy (if run), COCONUT reference numbers
      - Honest interpretation per Section F criteria
    - This is the single document Stage 5 (assess) will read.

15. **Shut down the pod** (user action)

16. **Update STATE.md and EXPERIMENT_LOG.md** in the git repo
    - Add Round 2 entry to EXPERIMENT_LOG.md with results and the `ROUND2_SUMMARY.md` path.
    - Update STATE.md with "Round 2 complete, current state: ..."

Done when step 16 is merged to the working copy.

---

## Appendix: One-liner for reviewers

"Round 2 trains the same Matrix-CODI architecture from Round 1 on ProsQA (17.9k DAG-shaped logical reasoning examples) instead of GSM8K-Aug, with adapter changes to load ProsQA JSONs locally and match answers as strings, and runs the same rank-k projection ablation at eval time. A steep rank-k curve on ProsQA paired with the flat curve on GSM8K-Aug would establish that the ablation measures task structure. A flat curve on both is a valid negative result."
