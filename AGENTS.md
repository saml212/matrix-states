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

A SQLite DB at `.codex/memory/workflow.db` persists durable rules, corrections, and gotchas across sessions. Relevant rules auto-inject at prompt time via the `load-relevant-rules.sh` hook.

**When you learn something worth persisting, emit a `[LEARN]` block in your response:**

```
[LEARN] <category>: <one-line rule>
Mistake: <what went wrong>
Correction: <what the right approach is>
```

The `learn-capture.sh` Stop hook parses these automatically, dedupes by (category, rule), and inserts. `Mistake:` and `Correction:` are optional but recommended.

See `AUTOPILOT_HANDOFF.md` for the full harness spec and phase plan.

## Hardware

- **M4 Mac Mini 32GB** — Dev machine. 10 CPU cores, 12GB MPS GPU. Good for <15M params.
- **Brev 8×H100 80GB (active)** — "youthful-indigo-turkey" via the Brev CLI.
  The grant is a 2-month **uptime-metered** hardware window (bills while
  running, cannot be stopped — `brev stop` unsupported, delete only), not a
  fixed GPU-hour utilization budget; operative budget ≈192 GPU-h/day for the
  window (see STATE.md "Hardware" for the full correction, dated 2026-07-03).
  SSH alias set up via `brev login && brev refresh` (writes
  `~/.brev/ssh_config`, Included from `~/.ssh/config`). Details, venv setup, and
  the perpetual-sweep orchestration pattern in `matrix-thinking/H100_SETUP.md`.
- **M4 Ultra Mac Studio 256GB** — Available for 50-100M param experiments.
- Prior single-H100 cloud pods — superseded by the Brev cluster above.

## Data

Code lives in this repo. Data and checkpoints live elsewhere.

- **SSD:** `/Volumes/1TB_SSD/learned-representations/`
  - `data/text.bin` — 100MB WikiText-103 raw UTF-8 bytes
  - `data/images.bin` — 154MB CIFAR-10 raw pixels
  - `checkpoints/` — Model checkpoints (don't commit)
  - `experiment-runs/` — the FULL experiment archive, superset of the
    repo-root `experiment-runs/` directory.
- **experiment-runs/ hybrid archive policy (2026-07-04, corrected from the
  briefly-tried symlink approach — see `experiment-runs/README.md`, the
  source of truth):** the repo-root `experiment-runs/` is a real, committed
  directory, size-capped — files ≤25MB are tracked in git (crash-proof via
  GitHub); larger payloads (Z-dumps, checkpoints, >25MB JSONs) live ONLY on
  the SSD path above. Write new archives to BOTH: small files here
  (commit them), big files to the SSD path. If the SSD is unmounted, stop
  and say so.
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
- A low-rank matrix is NOT low-capacity. A rank-1 `Z = u⊗v₀` (v₀ fixed) stores
  `d` independent items via its free vector side, recoverable by a linear read.
  Rank is only the binding constraint when the readout requires EXACT recovery
  of K independent key→value mappings through a matrix-vector product — not
  when items can be packed as a vector-in-a-matrix-costume. Any "does the model
  use rank K" experiment must close this shortcut by construction (provable
  lower bound), not by assumption.
- Readout must force EXACT CONTINUOUS recovery, never argmax/nearest-neighbor
  over a codebook, when a rank≥K bound depends on it. Under argmax decoding a
  rank-1 matrix can recover ≈d associations (Nichani, Lee & Bietti, ICLR 2025,
  arXiv:2412.06538) — argmax silently breaks provable rank-necessity proofs.
- In any full-attention model, "hold K items" is trivially satisfiable via K
  *positions* at rank-1 each (position-decomposition). A single-state (P=1)
  hard bottleneck — decoder reads ONLY the state, never raw inputs — is
  required to make within-representation rank load-bearing. Verify the
  bottleneck holds with a gradient-based blank-out test (corrupt raw inputs
  post-encoding, confirm decode is unchanged), not a vacuous shape check.
- Permutation-based hop-depth/compositional-generalization tasks need a SINGLE
  full K-cycle, not a general random permutation. A general permutation
  decomposes into short disjoint cycles, and `π^h` is periodic with the cycle
  length — "held-out" hop depths silently collapse via `h mod cycle_length`
  into in-distribution or trivial (identity) queries. Verified: 50-100% of
  nominally held-out queries collapsed across K∈{4,8,12,16} before the fix.
  Stratify results by effective distance (`h mod K`), not raw nominal hop.
- Integer/structural correctness checks (injectivity, rank-equals-K, etc.) need
  EXACT thresholds. A `-1` (or any) numerical-tolerance slack copied from a
  floating-point context into a structural check silently defeats
  single-instance violations. Always run the negative unit test that's
  supposed to prove the check "has teeth" to completion — don't just write it.
- A calibration run (one real training run at the target config) before a big
  sweep is mandatory, not optional. It catches convergence ceilings (a small
  reliable model may plateau below your target metric threshold — re-register
  the metric/threshold rather than assume it) and confirms a "bigger model"
  guess doesn't silently diverge before you commit a sweep's compute to it.
- On a remote box, NEVER run `pkill -f <pattern>` where `<pattern>` could match
  the SSH command string invoking the kill itself — it self-kills the shell
  running it (manifests as SSH exit 255 with no visible error). Use tmux
  session names (`tmux kill-session -t <name>`) or exact PIDs instead.
- For unattended/overnight cluster runs: launch the long-running process inside
  `tmux new-session -d -s <name> "<cmd>"`, not backgrounded shell (`cmd &`)
  over SSH — the latter can die on session/control-master hiccups. Wrap it in a
  self-healing supervisor loop (`while [ ! -f STOP ]; do <cmd>; sleep 15; done`)
  so a crash auto-restarts; make the orchestrator itself resume-safe (skip
  already-completed work by checking output validity, not just existence) so a
  supervisor restart loses nothing. A local Claude Code session restart kills
  local loops/monitoring but NOT an on-box tmux+supervisor process — this is
  the intended failure isolation, verify the box survived before assuming
  anything is lost.
- Multiple independent adversarial audit rounds catch different bugs each
  round — do not stop at one. On one experiment (Task E), round 1 caught 2
  FATALs (a tolerance bug, a periodicity confound); the fix was itself audited
  fresh (round 2) before trusting it clean. A self-audit by the same
  implementer is not a substitute for an independent audit — it caught real
  issues the self-audit missed.
- Hold tokenization (or any second architectural axis) FIXED when testing a
  primary hypothesis (e.g. matrix representation). Bundling two unproven
  changes at once makes any result — positive or negative — uninterpretable.
  Treat the second axis as a separate, explicitly-sequenced follow-on ablation.

## Research Direction

**Matrix Thinking (active):** 32×32 matrix tokens, multiplicative composition,
iterative refinement with shared thinking layers. Frobenius attention (flash-compatible).
Novel architecture — verified against literature March 2026. As of 2026-07-01:
the bolt-on matrix-CODI variant is confirmed dead (rank-blind, workshop paper
accepted on that negative result); the matrix-native-from-scratch variant
(Chapter 2 / `matrix-thinking/chapter2/`) is confirmed alive at d=8,16 — SGD
recruits provably-necessary rank when a task forces it. See STATE.md's
"Chapter 2 — STATUS" section for the full current picture and the running
Task E reasoning-transfer experiment.

**Byte-Agnostic (on hold):** Raw byte input for domain-general processing.
Partially validated. Combines with matrix thinking later.

## User Context

The user is learning ML fundamentals. When asked, teach from first principles with
real math. The user wants rigorous experimental process — clean code, verified results,
honest assessment — so the work can scale to H100s with confidence. The user values
continuous iteration over perfection.
