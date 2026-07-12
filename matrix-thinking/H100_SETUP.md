# H100 Environment

## Access — Brev 8×H100 cluster (active, 2026-07-01 onward)

Instance: "nvidia-pebble / youthful-indigo-turkey", 8×H100 80GB SXM, GCP
asia-southeast1-c, via an NVIDIA Brev accelerator-lab grant (~1.6k GPU-hour
budget — track usage; an experiment class like Task D used only ~76 GPU-h for
a complete, decisive result, so don't over-provision compute up front).

**Setup (one-time, per Mac):**
```bash
brew install brevdev/homebrew-brev/brev
brev login        # browser auth; run interactively (prefix with ! in Claude Code)
brev refresh       # writes ~/.brev/ssh_config, Included from ~/.ssh/config
```
This creates SSH host aliases (e.g. `youthful-indigo-turkey`,
`youthful-indigo-turkey-host`) using `~/.brev/brev.pem` and a `cloudflared`
tunnel — `ssh youthful-indigo-turkey` then works directly, non-interactively,
from any shell (including Claude Code's Bash tool). No manual host/port/key
needed once refreshed. `brev ls` lists workspaces + status (RUNNING/Building).

**Environment on the box:** default user `nvidia`, home `/home/nvidia`. The
base image ships **no torch, no conda** — only system Python 3.12 + Docker +
NVIDIA drivers. `python3 -m venv` needs `python3.12-venv` installed first
(passwordless `sudo apt-get install -y python3.12-venv`). Then:
```bash
python3 -m venv /home/nvidia/tdenv
/home/nvidia/tdenv/bin/pip install -q --upgrade pip
/home/nvidia/tdenv/bin/pip install -q torch numpy
```
Confirmed working: torch 2.12.1+cu130, CUDA available, 8 devices, compute
capability (9,0) = H100/sm_90. 18T `/data` volume + 5.9T `/ephemeral` available
in addition to the 193G root disk.

**Storage layout used by the matrix-thinking chapter2 work:**
`/home/nvidia/chapter2/` — code (scp'd from `matrix-thinking/chapter2/` in the
repo) + `results/<experiment>/` for JSON outputs and logs.

## Perpetual/unattended sweep pattern (learned 2026-07-01, don't skip)

Running many short training jobs unattended across 8 GPUs while keeping them
near 100% utilized, without losing work to a preemption/disconnect/crash:

1. **Launch inside tmux, never a backgrounded SSH shell.** `cmd &` over SSH can
   die on a control-master hiccup. Use
   `tmux new-session -d -s <name> "<cmd>"`. Never `pkill -f <pattern>` where
   the pattern could match your own SSH command string — it self-kills the
   shell issuing the kill (manifests as a mysterious SSH exit 255). Use
   `tmux kill-session -t <name>` or exact PIDs.
2. **Wrap the orchestrator in a self-healing supervisor:**
   ```bash
   while [ ! -f results/<exp>/STOP ]; do
     python run_<exp>_sweep.py --forever ... >> results/<exp>/orchestrator.log 2>&1
     sleep 15   # then loop — auto-restarts on any crash
   done
   ```
   Run the supervisor itself inside tmux. A local Claude Code session restart
   kills local monitoring loops but NOT this on-box tmux+supervisor process —
   verify the box survived (it should) rather than assuming lost work.
3. **The orchestrator itself must smoke-gate first** (run `--smoke` on GPU 0,
   abort the whole queue on failure — never train on unverified code), be
   **exception-isolated per launch** (a transient OS error on one
   `Popen()` must not kill the orchestrator process), use
   **validity-checked resume** (a run counts done only if its output JSON
   parses and has the expected keys — not just "file exists", which can be a
   truncated write from a killed process), apply a **per-run timeout with GPU
   quarantine** (a wedged reap must not let a poisoned GPU get reused), and
   write a **guarded aggregate** (a single malformed record must not prevent
   `SUMMARY.txt` from being written for everything else). **Residual gap
   found on the Task D overnight round-2 re-audit (`chapter2/gauntlet/
   AUDIT_overnight_r2.md`, not fully closed by the fixes above):**
   `write_progress()` calls and the harvest loop's `lf.close()` are
   themselves unguarded — an exception between opening and closing a
   progress/log file can still corrupt or lose that file even with
   exception-isolation and GPU quarantine in place elsewhere. Wrap file
   writes/closes in their own `try/finally`, not just the launch/reap paths.
4. **Pack multiple tiny runs per GPU** (`--per-gpu N`) to reach near-100%
   utilization when models are small (~1GB, <100% of one H100) — but know that
   at full packing this reflects contention, not N× more science/sec; the
   genuinely compute-hungry tranche is one-job-per-GPU at a larger model/batch.
5. **Perpetual refill** (new seed offset each round, same manifest) keeps the
   queue from draining, but oversamples fast — plan the next experiment before
   the current one is 2-3 rounds deep into pure reruns.

See `matrix-thinking/chapter2/run_overnight.py` (Task D) and
`run_task_e_sweep.py` (Task E) for reference implementations of this pattern.
(`chapter2/DEPLOY.md`, the earlier deploy runbook this section was distilled
from, is archived at `archive/chapter2-gauntlet/DEPLOY.md` as of 2026-07-04
— its box-specific details, e.g. the `ubuntu` user, are stale; this section
is the current source of truth.)

## Standing programmatic queue (built 2026-07-11, PI GPU-saturation directive)

A persistent, Claude-independent job queue lives at `~/queue/` on the box,
distinct from (and coexisting with) the per-campaign perpetual-sweep
pattern above. Full spec + operational runbook:
`matrix-thinking/queue/QUEUE_README.md` (repo copy = source of truth; a
synced copy is deployed to `~/queue/QUEUE_README.md`). Generator + runner
live in `matrix-thinking/queue/` (`generate_jobs.py`, `queue_worker.sh`).

**One-line summary:** 8 per-GPU worker daemons (`queue_worker_g0..g7`, each
its own tmux session, each wrapped in the house supervisor loop
`while [ ! -f STOP ]; do bash queue_worker.sh <N>; sleep 15; done`) claim
job specs from `~/queue/pending/` (atomic `mv`, priority-ordered by
filename) and run them as their own child process, ONLY when their assigned
GPU shows zero `nvidia-smi --query-compute-apps` entries AND <2 GiB
`memory.used` — this auto-yields to ANY other job on the box (a filler
session, a priority sweep claiming the whole node) with zero coordination,
polling every 60s while busy. `~/queue/PAUSE` stops new claims box-wide;
`~/queue/STOP` is the global kill switch (every worker finishes its current
job, then exits — never `pkill`, exact tmux session names only, per the
standing hard rule). A job's own `validity_check` command (not its raw
exit code) decides `completed/` vs `failed/`; one bad job never wedges its
worker (the claim loop always continues). Coexists with (does not manage,
does not depend on) any other tmux session on the box — including this
section's own perpetual-sweep pattern above and any priority campaign
sweep — by construction of the free-GPU gate.

**Relationship to the pattern above:** the perpetual-sweep pattern (a
single orchestrator script looping over ITS OWN manifest, e.g.
`run_task_e_sweep.py --forever`) is what a CAMPAIGN uses to keep its own
GPUs busy across a sweep. The standing queue is the box-level backstop
underneath that — it fills whatever GPUs a campaign sweep is NOT currently
using, and hands GPUs back the instant a campaign sweep claims them. A
campaign's own perpetual-sweep orchestrator needs no awareness of the
queue at all; the queue notices the campaign's `nvidia-smi` footprint and
backs off on its own.

## Access (prior single-H100 pod, superseded — kept for reference)
```
ssh root@154.57.34.103 -p 44178 -i ~/.ssh/id_ed25519
```
Volume: `/toy_story_slam/` (200GB persistent)

## Stack
- Python 3.12 + PyTorch 2.9.1+cu128 (CUDA 12.8)
- 1x NVIDIA H100 80GB HBM3
- Flash Attention available via SDPA backend

## File Layout
```
/root/
  run_exp2_iterative.py    ← current experiment script
  exp2_stdout.log          ← stdout/stderr
  src/
    matrix_thinker.py      ← v1 model (3D attention, PonderNet)
    matrix_model.py        ← v1 bilinear
    matrix_model_v2.py     ← v2 multiplicative
    data.py                ← data utils
  data/
    reasoning/             ← OpenR1-Math tokenized
      train.pt, val.pt, meta.json
  results/
    best.pt                ← Run 8 checkpoint
    exp2_iterative/        ← Exp 2 results (script, logs, checkpoint, summary)
```

## Quick Commands
```bash
# Check if training is running
ps aux | grep python

# Check progress
tail -20 /root/exp2_stdout.log

# GPU status
nvidia-smi

# Kill training
kill $(pgrep -f run_exp)
```

## torch.compile (for future experiments)
```python
model = torch.compile(model, dynamic=False, fullgraph=True)
# Requires: fixed shapes, no data-dependent control flow
# Cache at /tmp/torchinductor_root/ — doesn't survive pod restarts
# Always do 1 warmup run first
```

## Multi-GPU (for future 8×H100)
```bash
torchrun --standalone --nproc_per_node=8 train.py
```

## Precision
- bfloat16 autocast for forward/backward (H100 tensor cores: 990 TFLOPS)
- float32 for optimizer states (AdamW)
- SDPA auto-selects flash attention backend with bfloat16

## Pod Setup Checklist (new pod)
```bash
# 1. Install deps
pip install --break-system-packages datasets transformers tokenizers huggingface_hub

# 2. Redirect HF cache to volume (container disk is only 20GB)
rm -rf /root/.cache/huggingface
ln -s /toy_story_slam/.cache/huggingface /root/.cache/huggingface

# 3. Set HF token
export HF_TOKEN=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

# 4. Upload scripts from matrix-thinking/h100_scripts/
# 5. Run data prep (WikiText: 46s, OpenR1-Math full: ~8 min)
# 6. Launch experiment
```

## Lessons Learned (Hard-Won)

### VRAM Budget (mat_dim=32, 8 layers, T=8, H100 80GB)
| Batch/GPU | VRAM | Status |
|-----------|------|--------|
| 32 | 14 GB | Works but wastes 66GB |
| 96 | 47 GB | **Safe max** — room for eval |
| 112 | 50 GB | Training fits, **eval OOMs** |
| 128+ | OOM | Don't try |

The bottleneck is the logits tensor: (B, L, 50257) in bf16.
batch=96 × 512 × 50257 × 2 = 5GB just for logits. With backward ~10GB.

### DDP Eval Timeout
Eval runs on rank 0 only. Other ranks wait at `dist.barrier()`.
NCCL default timeout is 10 minutes. If eval takes longer, ALL ranks abort.
**Fix:** `dist.init_process_group("nccl", timeout=timedelta(minutes=30))`
**AND** cap eval to 50 batches max.

### HuggingFace Cache
HF downloads go to `/root/.cache/huggingface/` by default (container disk, 20GB).
Large datasets fill it instantly. Symlink to volume on first setup.

### PyTorch 2.4 Quirks
- `nn.MultiheadAttention`: pass EITHER `attn_mask` OR `is_causal=True`, not both
- `torch.utils.checkpoint`: FutureWarning about `cpu.amp.autocast` — harmless
- `killall python3` may not match `/usr/bin/python` — use `pkill -f` instead

## Parameter Matching

Matrix models are extremely parameter-efficient. Thinking layers add almost
nothing — embeddings dominate. At mat_dim=32:
- Embeddings: ~3.2M params (63%)
- 8 thinking layers: ~0.2M params (4%)
- Output head: ~1.6M params (31%)

Comparison: LoopFormer (ICLR 2026) at n_embd=96, 2 blocks × 8 loops = 5.3M params.
Script ready at `run_loopformer_baseline.py`.
