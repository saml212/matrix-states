# Continual GPU Monitoring Workflow (for agents)

This document is written for another Claude Code agent (or similar) who needs
to keep a rented GPU pod running experiments **continuously** — including
after spot preemptions, after the agent's conversation is interrupted, and
across context compaction. It describes the system as it actually runs in
this repo, including the gotchas that took multiple sessions to learn.

Target reader: an agent with Bash, SSH, Edit, Write, and Read tools, working
on a personal workstation (macOS) with one or two remote GPU pods.

---

## 1. What the system does

The user rents a pod with 1–2 H100s. The pod is spot-priced, so it gets
preempted every 3–12 hours. The agent's job is to:

1. Keep the GPU(s) at close to 100% utilization at all times.
2. Queue a backlog of training experiments so the GPU has work when one
   finishes.
3. Detect preemption quickly and restart from a safe state when the user
   provides a new SSH port.
4. Never lose a completed training run. Partial training progress *will* be
   lost on preemption — that is an accepted cost.
5. Pull completed experiment results to the local workstation so they
   survive pod destruction.
6. Do all of this without blocking the agent's main Bash conversation on
   long-running SSH commands. Use `run_in_background: true` for anything
   that takes more than ~10 seconds.

There are three persistent components:

| Component | Lives | Purpose |
|---|---|---|
| `dual_queue.sh` (or `master_queue.sh` single-GPU variant) | On the pod | Pops one command from a queue file when the GPU is idle; runs it; repeats forever. |
| `queue_gpu0.txt` / `queue_gpu1.txt` (or `queue.txt` single-GPU) | On the pod, persistent volume | Ordered list of shell commands to run. One command per non-comment line. |
| `pull_loop.sh` | On the local workstation | Every 600 s, scp any new SUMMARY.txt / results.json from the pod to the local repo so completed experiments are archived off the pod. |

On top of those, the agent runs a **wakeup chain**: a `sleep 3540; ssh … probe; tee /tmp/pod_poll_latest.txt` command in the background that fires every ~59 min. When it fires, the agent reads the poll output, reacts (pull new results, refill queues, restart crashed workers), then launches the next wakeup.

---

## 2. The three persistent scripts

### 2.1 `dual_queue.sh` (pod-side, one per GPU)

```bash
#!/usr/bin/env bash
# Usage: dual_queue.sh <gpu_index>
GPU=$1
QUEUE_FILE=/workspace/pebble/queue_gpu${GPU}.txt
LOG=/workspace/pebble/master_queue_gpu${GPU}.log
exec >> "$LOG" 2>&1
log() { echo "[$(date -Iseconds)][gpu${GPU}] $*"; }
log "=== dual_queue gpu${GPU} started ==="

gpu_busy() {
    local util mem
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i ${GPU} 2>/dev/null | tr -d " ")
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU} 2>/dev/null | tr -d " ")
    [ -z "$util" ] && return 0
    if [ "$util" -gt 5 ] 2>/dev/null || [ "$mem" -gt 5000 ] 2>/dev/null; then return 0; fi
    return 1
}

pop_next_cmd() {
    python3 - "$QUEUE_FILE" <<'PYEOF'
import sys
p = sys.argv[1]
with open(p) as f: lines = f.readlines()
out, popped = [], None
for line in lines:
    s = line.strip()
    if popped is None and s and not s.startswith("#"):
        popped = line.rstrip("\n"); continue
    out.append(line)
if popped is not None:
    with open(p, "w") as f: f.writelines(out)
    print(popped)
PYEOF
}

while true; do
    if gpu_busy; then log "gpu${GPU} busy, sleeping 120s"; sleep 120; continue; fi
    NEXT=$(pop_next_cmd)
    if [ -z "$NEXT" ]; then log "queue empty, sleeping 600s"; sleep 600; continue; fi
    log "launching: ${NEXT:0:120}..."
    eval "$NEXT"
    log "completed"
    sleep 5
done
```

Key design choices and why:

- **`eval "$NEXT"`** — the queue contains raw shell commands (with `export`, pipes, etc). `eval` runs them. Alternatives like `bash -c` don't let the command inherit the shell's `exec >>` log redirection cleanly.
- **`gpu_busy` thresholds at `util > 5%` and `mem > 5000 MiB`** — nvidia-smi util briefly drops to 0% during eval steps, data loader refreshes, and checkpoint writes. A strict `util > 0` test would trigger false pops. 5%/5GB rejects idle baseline noise but admits real training gaps of a few seconds.
- **`pop_next_cmd` uses Python, not awk/sed** — shell parsing of multi-line-with-embedded-single-quotes is a nightmare. Python's `with open() as f: f.readlines()` handles it cleanly. The pop also writes the remaining lines back to the file so a kill-9 of dual_queue.sh loses at most one in-flight command, not the whole queue.
- **`sleep 120` when GPU busy, `sleep 600` when queue empty** — both tuned low enough to be responsive but high enough that the polling doesn't show up in nvidia-smi. Do not drop the 120 below 60; torch DataLoader startup can briefly show GPU idle.

### 2.2 Queue files

```
# /workspace/pebble/queue_gpu0.txt
# Comments start with #. Blank lines ignored.
# Each non-comment line is ONE complete shell command.
# The command can include pipes, tee redirects, exports.

export HF_HOME=/workspace/pebble/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0; /workspace/pebble/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 /workspace/pebble/round3_gamma0/scripts/run_matrix_codi.py --mode train_matrix --dataset prosqa --gamma 0 --readout flatten --seed 100 --batch-size 16 --epochs 25 --warmup-steps 50 --max-eval-batches 8 --results-dir /workspace/pebble/round_pc/results/flatten_seed100 2>&1 | tee /workspace/pebble/round_pc/logs/flatten_seed100.log

# Next job goes here
...
```

**Non-obvious rules:**

- Each command is one logical line. Long lines are fine — no backslash continuations.
- `CUDA_VISIBLE_DEVICES=0` or `=1` at the start binds to the right GPU when two dual_queue workers run on two GPUs.
- Always include `2>&1 | tee /path/to/logfile.log` so you can tail the actual training output. The `master_queue_gpu*.log` is for dual_queue meta-events only.
- Use an absolute path to the venv python (`/workspace/pebble/venv/bin/python`), not `python` — your shell PATH may not be set when dual_queue runs as a daemon.
- `--results-dir` should be a distinct path per run. Collisions overwrite.
- `torch.distributed.run --standalone --nproc_per_node=1` is required even for single-GPU runs because the training script hard-asserts torchrun launch (defensive — it prevents someone accidentally running without torchrun).

### 2.3 `pull_loop.sh` (local-side)

```bash
#!/usr/bin/env bash
REPO=/Users/.../learned-representations
POD_HOST=root@<pod-ip>
POD_PORT=<port>
KEY=$HOME/.ssh/id_ed25519
LOG=$REPO/experiment-runs/_auto_sync/pull_loop.log

log() { echo "[$(date -Iseconds)] $*" >> "$LOG"; }

for iter in 1 2 3 4 5 6 7 8 9 10; do
    log "=== iteration $iter ==="
    POD_STATE=$(ssh -o ConnectTimeout=10 -i "$KEY" -p "$POD_PORT" "$POD_HOST" 'cat /workspace/pebble/master_state.json 2>/dev/null; echo "---"; ls /workspace/pebble/round*/results/*/SUMMARY.txt 2>/dev/null' 2>&1)
    # For each SUMMARY.txt found, scp it + its sibling files to a mirror on local disk
    # (full script in experiment-runs/_auto_sync/pull_loop.sh)
    sleep 600
done
```

Run it on a **local** Mac terminal with `nohup bash pull_loop.sh > /dev/null 2>&1 &`. It survives your terminal quitting. If the Mac reboots, you restart it manually.

The pod writes `/workspace/pebble/master_state.json` every iteration of dual_queue with `{queue_remaining, training_running, gpu_busy, gpu_util_pct, gpu_mem_mib}` — this is a cheap status file the pull_loop reads first to decide whether a full directory listing is worth doing.

---

## 3. The wakeup chain (agent-side)

This is the pattern the agent uses to stay in the loop without blocking its own conversation.

```
Bash command pattern:

  echo "=== POLL $(date -Iseconds) ==="
  sleep 3540
  echo "=== WAKE $(date -Iseconds) ==="
  ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p <port> root@<ip> '<probe commands>' 2>&1 | tee /tmp/pod_poll_latest.txt
  echo "=== DONE $(date -Iseconds) ==="

Bash tool args:
  run_in_background: true
  timeout: 3700000   (3,700 seconds = 61 min, slightly larger than sleep + probe)
```

Design points:

- `sleep 3540` is **59 minutes**. The Anthropic prompt cache has a 5-minute TTL, so any wait over 5 min pays a cache miss. Since we're paying the miss anyway, we might as well amortize it over a long sleep. 59 min is a reasonable cadence: long enough to not waste cache, short enough to catch issues before they compound.
- The probe SSH command is a single multi-line `'...'` block. Inside it use `echo "---section---"` separators and `head`/`tail` to keep the output compact. Redirect the result to `/tmp/pod_poll_latest.txt` so the main conversation can read it with a small `cat /tmp/pod_poll_latest.txt | tail -40`.
- The `timeout: 3700000` on the Bash tool call is the agent's tool-level timeout (in **ms**). The clamp in Claude Code's Bash tool caps this around 10 min unless you specify it explicitly; setting 3.7M ms tells the harness to allow the longer sleep.
- `run_in_background: true` is essential. If you run this foreground the conversation blocks for an hour. Backgrounded, you get a `<task-notification>` system-reminder when it completes.
- When you launch the poll, **immediately say "going quiet"** and stop generating text. The user will ping you with the task-notification when the poll finishes. Don't try to fill the hour with other work unless the work is genuinely non-overlapping.

### Reacting to a wakeup

When the task-notification arrives:

1. `cat /tmp/pod_poll_latest.txt | tail -40` — read the latest state.
2. If pod is reachable and GPUs are running: log progress, launch next poll. Optionally append more experiments to the queue if it's running low.
3. If pod is unreachable (SSH refused, ICMP timeout): tell the user. Do **not** retry in a loop. They need to get the pod back; you can't.
4. If a run completed: scp the SUMMARY.txt to local (it'll also get picked up by pull_loop on its next cycle, but eager pulls are cheap insurance).
5. If a run crashed: check the log tail, diagnose, add a fix-up experiment to the queue head if relevant.

---

## 4. Recovery patterns (preemption handling)

Spot preemption means the pod vanishes without warning. SSH stops responding, `ping` fails, running experiments die. When the user restarts the pod they give you a new SSH port. The pod's **persistent volume** (anything in `/workspace/`) survives; anything on overlay disk (`/root/`, installed pip packages) does NOT.

### 4.1 On preemption (pod unreachable)

1. Verify: `ssh … 'echo alive'` returns "connection refused" AND `ping -c 2 -W 2 <ip>` shows 100% loss. Both together = full preemption. SSH refused alone could mean sshd crashed.
2. Stop running polls. The wakeup chain should exit gracefully (the poll will log "connection refused" and complete; don't launch the next one).
3. Tell the user the pod is down and what state was last banked locally. Wait for new port info.

### 4.2 On recovery (new SSH port provided)

```
1. Clear the stale ssh-known-hosts entry:
   ssh-keygen -R "[<ip>]:<new_port>"

2. Probe volume persistence:
   ssh -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -p <new_port> root@<ip> '
   nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
   ls /workspace/pebble/
   grep -v "^#" /workspace/pebble/queue_gpu0.txt | grep -v "^$" | wc -l
   grep -v "^#" /workspace/pebble/queue_gpu1.txt | grep -v "^$" | wc -l
   ps auxf | grep dual_queue | grep -v grep
   '

3. Inventory which banked experiments still exist (they do if /workspace persisted):
   for d in ...; do test -f /workspace/pebble/.../$d/SUMMARY.txt && echo "$d: OK"; done

4. Check which pip-installed packages survived (they often don't — overlay disk):
   /workspace/pebble/venv/bin/python -c "import datasets, transformers, wandb, click; print('deps OK')"
   If any fail, reinstall: /workspace/pebble/venv/bin/pip install <missing>

5. Restart dual_queue workers — they do NOT survive the pod rebooting:
   ssh ... 'bash -c "nohup setsid /workspace/pebble/dual_queue.sh 0 < /dev/null > /dev/null 2>&1 &"'
   ssh ... 'bash -c "nohup setsid /workspace/pebble/dual_queue.sh 1 < /dev/null > /dev/null 2>&1 &"'

6. Update pull_loop.sh's POD_PORT to the new port; restart pull_loop locally:
   Edit tool on experiment-runs/_auto_sync/pull_loop.sh, change POD_PORT=<old> to POD_PORT=<new>
   pkill -f pull_loop; nohup bash pull_loop.sh > /dev/null 2>&1 &

7. Verify GPUs start firing within 2–3 minutes. If they don't, check master_queue_gpu*.log for errors.

8. Launch a fresh wakeup poll with the new port.
```

### 4.3 On recovery, if an experiment was mid-training when preempted

Mid-training state is lost. The experiment that was running needs to be **re-queued**. If the queue popped it but dual_queue died before `eval "$NEXT"` returned, you've lost that queue entry *and* the progress. Prepend a fresh copy of the command to the queue file so it runs next:

```python
# small helper pattern: prepend a command block to queue_gpu0.txt, before the first non-comment line
python3 -c "
with open('/workspace/pebble/queue_gpu0.txt') as f: lines = f.readlines()
block = '# Re-run after preemption\n<your command>\n\n'
out, inserted = [], False
for line in lines:
    if not inserted and line.strip() and not line.strip().startswith('#'):
        out.append(block); inserted = True
    out.append(line)
if not inserted: out.append(block)
with open('/workspace/pebble/queue_gpu0.txt', 'w') as f: f.writelines(out)
"
```

Do this over SSH. Don't try to edit the queue file locally and scp — you'd lose anything the pod wrote between your read and scp.

---

## 5. Known failure modes (things that ate sessions)

These are the bugs I hit that cost real time. List them here so the next agent doesn't repeat them.

### 5.1 `/dev/shm` exhaustion

PyTorch DataLoaders use shared memory (`/dev/shm`) for multi-worker data passing. If the user's pipeline also writes to `/dev/shm` (e.g. synthetic data generation workers writing shards to `/dev/shm/mmd/`), training jobs fail with `RuntimeError: unable to write to file </torch_*>: No space left on device (28)` even though the disk has free space.

**Symptoms:** GPU allocated (nvidia-smi shows VRAM used), util 0%, process in `S` sleeping state, no recent log output. Looks like the GPU stalled but actually shared memory is full.

**Detection:** `df -h /dev/shm` — if it's >95% full, stop queuing new jobs, tell the user. Do NOT try to delete files; the user's pipeline owns them.

**Mitigation:** training jobs should ideally use `num_workers=0` or set `torch.multiprocessing.set_sharing_strategy('file_system')` in the script, but that requires a script edit. Simplest: tell the user about the contention and let them decide.

### 5.2 Path clobbering via local script edits

The canonical `run_matrix_codi.py` lives on the pod. There's a local mirror used as a working copy. The local copy had `/Volumes/1TB_SSD/...` paths hardcoded in its CONFIG dict. When I `scp`'d the local copy over the pod version to add new CLI flags, I clobbered the pod-native data paths. Every subsequent training job crashed with `FileNotFoundError: /Volumes/1TB_SSD/...`.

**Lesson:** before scp'ing a local script to the pod, grep for hardcoded `/Volumes/`, `/Users/`, `/home/` paths and rewrite them to pod-appropriate paths.

### 5.3 Queue popped but command failed to launch

If a queue command has a bug (missing dependency, wrong flag, bad path), `eval "$NEXT"` returns quickly with nonzero exit. The queue entry is popped (gone) but no real work happens. dual_queue treats this as "GPU now idle" and pops the next entry. It can burn through a 10-entry queue in 2 minutes.

**Detection:** in the poll, watch for `grep CMD /workspace/pebble/master_queue_gpu*.log` showing many launches in a short window while nvidia-smi stays at 0%.

**Mitigation:** always include `2>&1 | tee /path/to/logfile.log` in the queue command so failures leave a trace. On preemption-recovery, before restarting dual_queue, verify `/workspace/pebble/venv/bin/python /workspace/pebble/.../run_matrix_codi.py --help` actually runs.

### 5.4 Claude Code Bash tool sleep restrictions

- `sleep N` commands in the foreground are clamped — the harness blocks `sleep 300+` and shorter-sleep-loops. Use `run_in_background: true` for long sleeps.
- `sleep 60` followed by another command in a `&&`-chained line is also blocked ("blocking commands in the foreground"). Use `run_in_background: true`.
- The fix is always `run_in_background: true` + `timeout: <ms>` large enough for the sleep + work. For a 59-min sleep + 1-min probe, use `timeout: 3700000` (ms).

### 5.5 Deferred tool schemas

Some tools (`TaskCreate`, `CronCreate`, etc.) are shown in system-reminders as "deferred" and need `ToolSearch` with `select:<name>` to load their JSONSchema before you can call them. Trying to call a deferred tool directly returns `InputValidationError`.

### 5.6 Multiple dual_queue workers from re-running the start command

If you SSH in and run `nohup setsid /workspace/pebble/dual_queue.sh 0 &` twice, you get two workers fighting over queue_gpu0.txt. They'll race on pop_next_cmd — the file write in Python is atomic per call but the read-modify-write isn't across two processes, so you can end up launching the same job twice.

**Detection:** `ps auxf | grep dual_queue` after start; should show exactly 2 entries (one per GPU), not 3+.

**Mitigation:** always `pkill -f master_queue.sh 2>/dev/null; pkill -f dual_queue.sh 2>/dev/null; sleep 1;` before starting new workers.

### 5.7 torch.distributed.run child process naming

`pgrep -f "python.*run_matrix_codi"` is the intended "is training running" check. But every SSH command that has `run_matrix_codi` in its argv will also match — including the SSH wrapper `bash -c "... cat ... echo ---r7-tail--- ... run_matrix_codi ..."`. This triggers false positives in dual_queue's `any_training_running` check.

**Mitigation:** use `pgrep -f "python.*(run_matrix_codi.py|run_vanilla_sft.py|...)"` — the `python.*` prefix is required. Adjust the regex to match actual python process argv, not SSH-wrapper argv.

### 5.8 Stale Python heredoc processes

If you SSH-execute a Python heredoc that references "run_matrix_codi" in its content, and the heredoc is a long-lived script, the Python process's argv contains "run_matrix_codi" even though it's not training. Confused `pgrep -f "run_matrix_codi"` into reporting training-is-running. Eats 1+ hour of wall clock to diagnose.

**Mitigation:** kill stale heredoc processes explicitly (`kill -9 <pid>`) before restarting dual_queue. Or use the python-prefix regex from 5.7.

---

## 6. Queue hygiene

Rules of thumb:

- **Always keep at least 3 entries ahead** of what's running on each GPU. If one run takes 3h and you have 1 entry left, that's 3h of buffer before the GPU goes idle waiting.
- **Mix short and long jobs.** A 22h run that dies at hour 21 from preemption is a 21h loss. Better: queue mostly 3-5h jobs with one long job as a nice-to-have; even if the long one dies, the short ones bank results.
- **Label with comments.** Each queue entry should have a `# <short description>` comment immediately before it. Makes pop_next_cmd's log output greppable later.
- **Append, don't rewrite.** Use `cat >> queue_gpu0.txt <<'QEOF' ... QEOF` on the pod over SSH, not `scp local_queue.txt pod:queue_gpu0.txt`. Reason: between your local read and your scp, dual_queue may have popped entries from the live file; rewriting loses that state.
- **Version the queue state before big changes.** `cp queue_gpu0.txt queue_gpu0.txt.bak_$(date +%s)` before editing. Cheap insurance.

---

## 7. What gets archived locally

After every completed experiment, the pull_loop.sh and the agent's post-wakeup actions scp these files to the local repo:

- `SUMMARY.txt` — final accuracy + wall time (always small, always safe to pull)
- `rank_projection_ablation.json` — if eval was run (paper-critical for this project)
- `best_run_b_matrix.pt` — checkpoint (~500 MB; pull only the important ones)
- `train.log` — training curve (~1-10 MB; pull for the paper-critical runs)

Do not pull the full `/workspace/pebble/round*/` tree. That includes the HuggingFace cache (~10s of GB) and intermediate checkpoints. Selective pulls.

---

## 8. Anti-patterns

- **Don't** run `kill -9` on dual_queue.sh if a job is actively training. The training process is an eval'd child and loses its stdout pipe when dual_queue dies, but the GPU memory stays allocated until you kill the training process too. Use targeted `pkill -f "run_matrix_codi.*<specific-flag>"` to kill the training without killing dual_queue.
- **Don't** edit `queue_gpu*.txt` through git commits and git pull on the pod. The queue files are mutable state owned by dual_queue. Put them in `.gitignore`.
- **Don't** store experiment checkpoints in the git repo. They're 500MB each. Git-LFS or just leave them on the pod.
- **Don't** use `&` (shell background) to launch dual_queue in an interactive SSH. The moment SSH disconnects, SIGHUP kills the worker. Use `nohup setsid <cmd> < /dev/null > /dev/null 2>&1 &; disown` pattern.
- **Don't** try to poll faster than every ~20 min. You burn prompt-cache + API calls + user money for no additional signal. Run-level events happen at hour+ timescales.

---

## 9. A minimal first-run recipe

If you're an agent that just got handed a fresh pod with 1 GPU and the user says "keep this running":

```bash
# Step 1: SSH in, install deps, set up dirs
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip> 'bash -s' <<'EOF'
mkdir -p /workspace/pebble/{hf_cache,scripts,data,results,logs}
pip install torch transformers datasets wandb click
ln -sf /usr/bin/python3 /workspace/pebble/venv/bin/python  # assumes venv convention
EOF

# Step 2: scp training script + data + dual_queue.sh to pod
scp -i ~/.ssh/id_ed25519 -P <port> /path/to/local/run_script.py root@<ip>:/workspace/pebble/scripts/
scp -i ~/.ssh/id_ed25519 -P <port> /path/to/local/dual_queue.sh root@<ip>:/workspace/pebble/

# Step 3: create the initial queue (ssh heredoc)
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip> "cat > /workspace/pebble/queue_gpu0.txt <<'QEOF'
# First experiment
<your full command line here>
QEOF"

# Step 4: start dual_queue
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip> 'chmod +x /workspace/pebble/dual_queue.sh; bash -c "nohup setsid /workspace/pebble/dual_queue.sh 0 < /dev/null > /dev/null 2>&1 &"'

# Step 5: verify
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip> 'ps auxf | grep dual_queue | grep -v grep; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'

# Step 6: start local pull_loop (on Mac terminal, not via tool)
cd <repo>/experiment-runs/_auto_sync && nohup bash pull_loop.sh > /dev/null 2>&1 &

# Step 7: launch first wakeup poll (agent Bash tool, run_in_background=true, timeout=3700000)
echo "=== POLL $(date -Iseconds) ==="
sleep 3540
echo "=== WAKE $(date -Iseconds) ==="
ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p <port> root@<ip> '
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== queue ==="; wc -l /workspace/pebble/queue_gpu0.txt
echo "=== tail ==="; tail -3 /workspace/pebble/master_queue_gpu0.log
' 2>&1 | tee /tmp/pod_poll_latest.txt
echo "=== DONE $(date -Iseconds) ==="

# Say "going quiet" to the user and wait for the task-notification.
```

---

## 10. What to tell the user proactively

- After the first successful poll: "Both GPUs firing. Queue has N entries. Next poll in 59 min."
- On every preemption: "Pod unreachable. SSH refused + ICMP loss = full preemption. Last banked results: [...]. Waiting for new SSH port."
- When queue drops below 3: "Queue running low (<3 entries). Want me to add [candidate experiments]? Or let it go idle?"
- Never: "I'll check back in 5 minutes." You sleep 59 min. Use 5-min language only for genuinely fast operations.

---

## 11. Files involved in this repo

- `experiment-runs/_auto_sync/pull_loop.sh` — local pull_loop
- `experiment-runs/_auto_sync/WAKEUP_RUNBOOK.md` — per-wakeup checklist (short)
- `experiment-runs/_auto_sync/WORKFLOW_FOR_AGENTS.md` — this document
- `experiment-runs/_auto_sync/pull_loop.log` — pull_loop output
- On pod: `/workspace/pebble/dual_queue.sh`, `queue_gpu{0,1}.txt`, `master_queue_gpu{0,1}.log`, `master_state.json`

---

## 12. When to stop the loop

- User explicitly says "stop".
- Queue has been empty for 2+ polls (1+ hour idle compute — waste of money).
- Pod has been unreachable for 3+ polls (the agent can't bring it back).
- A research goal that was driving the queue has completed (the paper is written, the experiment class is exhausted).
- The user stops responding. This isn't an automatic stop, but if the user is asleep and the queue is healthy, you don't need to do anything proactively — just let it run and the next task-notification will wake you.

Don't stop the loop because you ran out of interesting things to add to the queue. Tell the user you're out of ideas and ask what they want queued next.
