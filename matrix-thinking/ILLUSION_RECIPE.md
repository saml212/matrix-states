# Illusion of Superposition baseline recipe (ProsQA 96.6%)

## Source

arXiv 2604.06374 — Rizvi-Martel, Rabusseau, Mosbach (2026)
"The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models"
Submitted April 7, 2026. 25 pages. CC BY 4.0.

Also cross-referenced against the original COCONUT paper and codebase:
- Hao et al. 2024, arXiv 2412.06769 (the paper being replicated)
- facebookresearch/coconut on GitHub (official COCONUT code)

**IMPORTANT — what the 96.6% number actually is:**
The paper does NOT report a separately trained "fine-tuned COCONUT without latent tokens."
The 96.6% result comes from taking the **fully trained COCONUT model** (which achieved 99.0%
with 6 latent tokens) and running inference on it with **zero latent tokens** — just feeding
the raw question and greedily decoding the answer. Same weights, same checkpoint.
No retraining is involved.

The paper's point: COCONUT apparently learned to extract the answer directly from the
question embedding, because withholding latent tokens only costs 2.4 percentage points
(99.0% → 96.6%). That's their "illusion" finding.

**Table 2(a), Section 5.1 (Fine-Tuned GPT-2, 12 layers):**

| Condition                   | Accuracy |
|-----------------------------|----------|
| CoT (discrete, 5-hop)       | 85.3%    |
| Coconut (6 latent tokens)   | 99.0%    |
| Coconut (no latent tokens)  | 96.6%    |

---

## Model

**GPT-2 base (124M parameters, 12 layers, hidden dimension 768)**
Source: Appendix D.2

- HuggingFace model ID: `openai-community/gpt2`
  Source: `facebookresearch/coconut` → `args/prosqa_coconut.yaml`
- 3 special tokens added to the vocabulary:
  `<|start-latent|>`, `<|latent|>`, `<|end-latent|>`
- All 3 tokens initialized from the `<` token embedding
  Source: Appendix D.2

---

## Training Config

All values below are from Appendix D.2 of arXiv 2604.06374, confirmed against
`args/prosqa_coconut.yaml` in facebookresearch/coconut.

| Hyperparameter             | Value          | Source                              |
|---------------------------|----------------|-------------------------------------|
| Optimizer                 | AdamW          | `run.py` (`optim.AdamW`)            |
| Learning rate             | 1e-4           | Appendix D.2; YAML confirmed        |
| LR schedule               | Constant (none)| No scheduler in `run.py`            |
| Warmup                    | None           | Not mentioned; no warmup in code    |
| AdamW betas               | (0.9, 0.999)   | PyTorch defaults (not overridden)   |
| AdamW epsilon             | 1e-8           | PyTorch defaults (not overridden)   |
| Weight decay              | 0.01           | Appendix D.2; YAML confirmed        |
| Batch size (per device)   | 16             | Appendix D.2                        |
| Gradient accumulation     | 1 step         | YAML: `gradient_accumulation_steps: 1` |
| Global batch size         | 64             | 4 GPUs × 16 per-device × 1 accum   |
| Total epochs              | 50             | Appendix D.2; YAML confirmed        |
| Epochs per stage          | 5              | Appendix D.2                        |
| Number of curriculum stages | 6            | Appendix D.2                        |
| Latent tokens per CoT step (c) | 1        | YAML: `c_thought: 1`               |
| Max latent stage          | 6              | YAML: `max_latent_stage: 6`         |
| Pad latent to max         | True           | YAML: `pad_latent_to_max: True`     |
| Max new tokens (eval)     | 128            | `run.py`: 128 for non-GSM datasets  |
| Random seed               | 0              | YAML: `seed: 0`                     |
| Mixed precision           | None (fp32)    | YAML: `bf16: False`                 |
| Optimizer reset per stage | Yes            | YAML: `reset_optimizer: True`       |
| Loss function             | Cross-entropy  | `coconut.py`: `CrossEntropyLoss`    |
| Label smoothing           | None           | Not mentioned                       |
| Dropout                   | Not specified  | Not mentioned in paper or code      |

**Curriculum schedule** (Section 5.1 / Appendix D.2):
- At stage k (epochs 5k through 5(k+1)−1), the first k CoT reasoning steps are replaced
  by k continuous latent tokens.
- Stage 0 = standard language reasoning (no latent tokens).
- Stage 1 = 1 latent token replacing step 1.
- ...
- Stage 6 = 6 latent tokens replacing all 6 steps.
- Total: 6 stages × 5 epochs each = 30 stage-epochs; model stays in stage 6 until epoch 50.

---

## Data Pipeline

Source: Appendix D.2 of arXiv 2604.06374; dataset.py of facebookresearch/coconut

- **Dataset:** ProsQA — "from the original Coconut codebase" (Appendix D.2 exact quote)
- **Train split:** 17,886 examples
- **Validation split:** 300 examples
- **Task type:** Entity reachability over randomly generated directed graphs of named
  entities. Graph depths range from 3 to 6 hops.
- **Train file:** `data/prosqa_train.json`
- **Val file:** `data/prosqa_valid.json`
- **No filtering mentioned.**

**Prompt format** (from dataset.py structure, since paper does not show exact template):

```
{question text}\n
{step 1 text}\n
{step 2 text}\n
...
### {answer text}{EOS}
```

With COCONUT latent tokens inserted between question and steps:

```
{question text}\n
<|start-latent|> <|latent|> <|latent|> ... <|end-latent|>
{step k+1 text}\n
...
### {answer text}{EOS}
```

- Question tokens: masked with −100 (no loss)
- Latent tokens: masked with −100 (no loss)
- `<|start-latent|>` and `<|end-latent|>`: masked with −100 (no loss)
- Reasoning steps and answer tokens: **loss computed here**

**Answer format example** (from paper Section 5.1 example):
"Every dax is a wug. Every dax is a zug. Every wug is a blicket. Rex is a dax. Is Rex a
blicket or a gorple?" → Answer: `blicket` (single entity name, no period, prefixed by `### ` in training)

**Max sequence length:** Not explicitly stated in paper or YAML. Padding is dynamic
(right-pad to longest in batch). No truncation mentioned.

---

## Eval

- **Decoding:** Greedy decoding (no temperature, no beam search)
  Source: `run.py` — no temperature/top_p in generate call; `synced_gpus` flag for DDP sync
- **Max new tokens:** 128 (for non-GSM8K datasets; `run.py`)
- **Metric:** Exact match accuracy
- **Checkpoint selection:** Best validation accuracy checkpoint
  ("Best Coconut checkpoint is at epoch 50 (99% validation accuracy)" — Section 5.1)
- **"No latent tokens" inference:** Feed only the question, no latent token injection,
  greedy decode. Same trained model checkpoint; no separate fine-tune.

---

## Hardware / Wall Time

- **GPUs:** 4 × A100 80GB (COCONUT codebase default; README note "need 4 gpus")
  The Rizvi-Martel paper uses "4 GPUs" with FSDP wrapping (Appendix D.2), but the paper
  does not specify the GPU model.
- **Parallelism:** FSDP, but "does not shard GPT-2's layers (effectively acting as DDP
  at this model scale)" — Appendix D.2
- **Wall-clock time:** Not reported in the paper.
- **Note:** With GPT-2 at 124M params on 4 GPUs, training 50 epochs on ~18K examples
  should complete in a few hours. On a single A100 or H100, expect similar.

---

## Things I Could Not Find in the Paper

1. **AdamW betas and epsilon** — Not stated. PyTorch defaults are (0.9, 0.999, 1e-8).
2. **Exact prompt template** — Paper gives a natural-language description and one example.
   Exact tokenization template (including `### ` prefix, `\n` separators) comes from
   `dataset.py` in the COCONUT codebase, not from the paper itself.
3. **Sequence length / max_length** — Not stated. Dynamic padding is used.
4. **Dropout rates** — Not mentioned in paper or YAML config.
5. **Early stopping** — Not mentioned. `save_only_improve: False` in YAML.
6. **Wall-clock training time** — Not reported.
7. **GPU model used by Rizvi-Martel** — Paper says "4 GPUs with FSDP" but does not name them.
8. **Multiple random seeds** — Only seed 0 confirmed. The original Hao et al. 2024 paper
   reports ±std intervals (e.g., 97.0±0.3) implying multiple seeds, but Rizvi-Martel 2026
   does not report variance for the fine-tuned condition. Likely single-seed only.
9. **GitHub repository** — No code release from Rizvi-Martel et al. The paper states
   they "follow the methodology of Hao et al. [2024]" and use the official COCONUT codebase
   (facebookresearch/coconut). No paper-specific repo found.
10. **Answer format with period vs. no period** — The example in the paper shows "blicket"
    as the answer. The `### ` prefix appears from `dataset.py` structure but is confirmed
    only by code inspection, not quoted in the paper.

**Reasonable defaults for missing values:**
- betas: (0.9, 0.999)
- epsilon: 1e-8
- dropout: 0.0 (GPT-2 base has built-in 0.1 attn/resid dropout in its architecture,
  but the COCONUT training does not add extra)
- max_seq_len: dynamic (no cap needed for ProsQA; examples are short)

---

## Gap to Our Current Config

Our config: `lr=1e-4, warmup=50, 25 epochs, batch=16, gpt2-small`

| Dimension                  | Paper (COCONUT on ProsQA) | Our Config          | Action Needed                        |
|---------------------------|---------------------------|---------------------|--------------------------------------|
| Model                     | GPT-2 124M                | GPT-2 small (same) | No change                            |
| Learning rate             | 1e-4                      | 1e-4               | No change                            |
| LR warmup                 | None                      | 50 steps           | Remove warmup (or keep — low risk)   |
| LR schedule               | Constant                  | Constant (assumed) | Verify no scheduler is active        |
| Batch size (per device)   | 16                        | 16                 | No change                            |
| Global batch size         | 64 (4 GPUs)               | 16 (1 device?)     | Match with grad accum if single GPU  |
| Gradient accumulation     | 1 step                    | Unknown            | Set to 4 if on 1 GPU to match global=64 |
| Epochs                    | 50 (staged curriculum)    | 25                 | Must increase to 50                  |
| Curriculum                | 6 stages × 5 epochs       | None (assumed)     | This is the core architectural change — implement staged curriculum |
| Optimizer reset           | Yes (per stage)           | Unknown            | Add optimizer reset at each stage    |
| Weight decay              | 0.01                      | Unknown            | Set explicitly to 0.01               |
| Special tokens            | 3 latent tokens added      | Unknown            | Must add + initialize from `<` token |
| Loss masking              | Question + latent = -100  | Unknown            | Only supervise steps + answer        |
| Random seed               | 0                         | Unknown            | Set seed=0 for reproducibility       |
| Mixed precision           | fp32                      | Unknown            | Keep fp32 for comparability          |
| Max new tokens (eval)     | 128                       | Unknown            | Set max_new_tokens=128 at eval       |
| Evaluation decoding       | Greedy                    | Unknown            | Confirm greedy (no temp/beam)        |
| Training data             | 17,886 ProsQA examples    | Unknown            | Use official COCONUT ProsQA data     |

**The critical gap is the curriculum training itself.**
The 96.6% result is NOT from a simple fine-tune — it requires the full 6-stage curriculum
where COCONUT learns to use latent tokens. The "no latent tokens" number is a probe of the
trained COCONUT checkpoint. If you want to reproduce 96.6%, you must first train COCONUT
to 99.0% over 50 epochs with the staged curriculum, then evaluate it with latent tokens
withheld.

To reproduce using the official code:
```bash
git clone https://github.com/facebookresearch/coconut
# Edit args/prosqa_coconut.yaml if needed (seed, paths)
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml
# Then eval without latent tokens: modify inference to skip latent injection
```
