# AGENTS.md

## Workflow: Plan → Research → Build → Audit → Run → Assess → Codify

Every cycle should make the next cycle better.

**Plan:** Talk with the user. Understand the goal before touching code.
**Research:** Send agents to verify claims and check novelty. Never assert facts without evidence.
**Build:** Write code. Keep it clean. Comment the non-obvious.
**Audit:** Send a separate agent to review code before running. Check shapes, gradients, stability. The implementer does not review their own work.
**Run:** Use the hardware. Parallel experiments when possible.
**Assess:** Be honest. Negative results are data. Don't spin.
**Codify:** Update STATE.md and EXPERIMENT_LOG.md. If you learned a lesson, also emit a `[LEARN]` block so it auto-saves to the learnings DB (see "Learnings DB" below).

## Learnings DB

A SQLite DB at `.Codex/memory/workflow.db` persists durable rules, corrections, and gotchas across sessions. Relevant rules auto-inject at prompt time via the `load-relevant-rules.sh` hook.

**When you learn something worth persisting, emit a `[LEARN]` block in your response:**

```
[LEARN] <category>: <one-line rule>
Mistake: <what went wrong>
Correction: <what the right approach is>
```

The `learn-capture.sh` Stop hook parses these automatically, dedupes by (category, rule), and inserts. `Mistake:` and `Correction:` are optional but recommended.

See `AUTOPILOT_HANDOFF.md` for the full harness spec and phase plan.

## Repo Layout

- `STATE.md` — Current project state, what's running, user context, dead ends
- `ARCHITECTURE.md` — Full architecture spec with verified citations
- `EXPERIMENT_LOG.md` — Every experiment and result
- `references.md` — Paper references library
- `matrix-thinking/` — Matrix-valued token representations (active)
  - `ADAPTIVE_THINKING.md` — Certainty-driven mode switching design (next feature)
  - `EXPERIMENTS.md` — Experiment queue and planning
  - `H100_SETUP.md` — H100 environment, access, commands
  - `h100_scripts/` — Training scripts
  - `src/` — Model code (v1/v2, legacy)
  - `research/` — Research agent outputs on matrix operations
- `experiment-runs/` — Archived exact scripts from each experiment
- `byte-agnostic/` — On hold, partially validated
- `archive/` — Dead ends and superseded docs
- `research/` — Literature surveys

## Hardware

- **M4 Mac Mini 32GB** — Dev machine. 10 CPU cores, 12GB MPS GPU. Good for <15M params.
- **H100 80GB HBM3** — Active. SSH in `matrix-thinking/H100_SETUP.md`.
- **M4 Ultra Mac Studio 256GB** — Available for 50-100M param experiments.
- **8×H100 (cloud)** — Available for serious scale. Use after code is proven.

## Data

Code lives in this repo. Data and checkpoints live elsewhere.

- **SSD:** `/Volumes/1TB_SSD/learned-representations/`
  - `data/text.bin` — 100MB WikiText-103 raw UTF-8 bytes
  - `data/images.bin` — 154MB CIFAR-10 raw pixels
  - `checkpoints/` — Model checkpoints (don't commit)
- **H100:** `/root/data/reasoning/` — 43.7M tokens OpenR1-Math (GPT-2 tokenized)
- **Do not store** large data files, checkpoints, or venv in the git repo

## Pre-Experiment Checklist (MANDATORY before every experiment)

1. **State the hypothesis in one sentence.** If you can't, don't run it.
2. **Compute FLOPs, memory, and param count on paper.** 10 minutes. No exceptions.
3. **Try to disprove it in 5 minutes.** "Could a vector do this?" "Is this known to fail?"
4. **Check the literature first.** Send a research agent BEFORE building.
5. **Design the comparison before the experiment.** What's the baseline? Are params matched?
6. **Define success criteria.** What BPB / metric improvement justifies the compute cost?
7. **Verify the claim is novel.** Don't claim uniqueness without checking if vectors can do the same thing.

## Waterfall Process for New Ideas
Before building ANYTHING, run this waterfall with subagents:
1. **Brainstorm agent:** Generate ideas (read all project docs first)
2. **Research agent:** Validate top ideas against literature, find code
3. **Attack agent:** Try to kill every idea. Find fatal flaws.
4. **Validation agent:** Address each attack with evidence. Confirm or deny.
Only build what survives all four stages.

## Hard Rules (Learned From This Project)

- Verify before claiming. Use web search or research agents.
- Audit code with a separate agent before running experiments.
- Smoke test every model (forward pass, backward pass, gradient check) before training.
- Use standard benchmarks for publishable claims. Byte BPC is for internal use.
- Dead directions stay dead. Don't revisit archived ideas unless the user asks.
- When teaching, use real math and verified citations. Send research agents if unsure.
- Save the exact script that was run alongside experiment results for reproducibility.
- Log everything to a file. Produce a human-readable summary at the end.
- PonderNet halting collapses at small scale — use fixed iterations first, adaptive later.
- Thought appending with causal mask is broken — use iterative in-place refinement.
- Making matrix ops cheaper does NOT fix the quality gap. Speed ≠ quality.
- The param-matched flat-vector ablation blocks ALL downstream decisions. Run it first.
- Outer-product embedding init: u,v std must be sqrt(target_std), not target_std. Products have std=σ².
- "Just add layers" beats thought interleaving at 288K params.
- DeltaNet rank-1 updates are for recurrent models, not iterative attention. Don't conflate.
- Combined speedups don't multiply when they target overlapping pipeline stages.
- The reshape equivalence: any d²-dim vector can be reshaped to d×d matrix and vice versa.
  Structure only matters if OPERATIONS preserve it. Flatten = structure gone.
- At 288K params, models barely learn unigram statistics. Can't draw conclusions about
  reasoning, generalization, or abstract thought at this scale. Need 10M+ minimum.
- Sweep experiments (multiple configs, one script, sequential) save GPU downtime.
  Add try/except so one crash doesn't kill remaining configs.
- Never compress matrices to vectors. Use MultiProbeHead (bilinear probes) for output.
- HF cache defaults to container disk (`/root/.cache/`). Symlink to volume immediately.
- DDP eval on rank 0 only will NCCL timeout if eval takes >10 min. Set timeout to 30 min AND cap eval batches (50 max).
- Smoke test batch size includes EVAL batch size — eval can OOM even if training fits.
- batch=96 per GPU is the safe max for mat_dim=32 on H100 80GB (47GB used, room for eval).
- batch=112 fits training but OOMs during eval. Don't go above 96 without testing eval too.
- The 50K vocab logits tensor is the VRAM bottleneck, not the model activations.
- Use the same dataset for ALL experiments in a comparison. Don't swap data between rounds.
- `nn.MultiheadAttention` in PyTorch 2.4 requires explicit `attn_mask` OR `is_causal`, not both.

## Research Direction

**Matrix Thinking (active):** 32×32 matrix tokens, multiplicative composition,
iterative refinement with shared thinking layers. Frobenius attention (flash-compatible).
Novel architecture — verified against literature March 2026.

**Byte-Agnostic (on hold):** Raw byte input for domain-general processing.
Partially validated. Combines with matrix thinking later.

## User Context

The user is learning ML fundamentals. When asked, teach from first principles with
real math. The user wants rigorous experimental process — clean code, verified results,
honest assessment — so the work can scale to H100s with confidence. The user values
continuous iteration over perfection.
