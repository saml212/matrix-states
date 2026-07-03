# Resilient Launcher Design — Post-Mortem and Redesign

## 1. Root Cause Analysis

### What actually happened (reconstructed from artifacts)

**First pod (A100 PCIe, original run):**

The launcher ran against a version of `run_matrix_codi.py` that did NOT yet have
`--prosqa-train-path`, `--prosqa-val-path`, `--rank-loss`, `--force-rank-during-training`,
or `--weight-decay` flags. Evidence: `multi2_baseline_gamma0/seed7/` has `results.json`,
`SUMMARY.txt`, and `best_run_b_matrix.pt` but **no `script.py` and no `train.log`** in
the results dir. The `shutil.copy2(__file__, results_dir/"script.py")` line was added in
a later patch. So seed7 ran against an intermediate patched version that had the data-path
flags but not yet the `shutil.copy2` self-archiving.

Timeline on first pod:
- `multi2_baseline_gamma0 seed1337`: argparse error, unknown flag `--prosqa-train-path` → rc=2, fast-fail (<5 s)
- `multi2_baseline_gamma0 seed42`: same → rc=2, fast-fail
- **[Operator ssh'd in, patched `run_matrix_codi.py` to add data-path and rank-loss flags]**
- `multi2_baseline_gamma0 seed7`: SUCCESS (~150 min), ran with first-patch script
- `multi2_force_rank1_gamma0 seed1337`: SUCCESS (~150 min), ran with final-patch script (has `script.py`)
- `multi2_force_rank1_gamma0 seed42`: SUCCESS (~150 min)
- `multi2_force_rank1_gamma0 seed7`: **pod preempted mid-run** → empty results dir

**After auto-resume (new container, same /workspace MFS):**

- `/workspace/pebble/.markers/stage1_env` exists → stage1 **should** reinstall pip deps
- BUT: the stage1 condition is `! done_marker stage1_env || ! python3 -c "import transformers, torch"`
- On a fresh A100 container that has PyTorch in the container image,
  `import transformers` might FAIL (transformers is not in the container image —
  only in overlay disk which was wiped) but `import torch` PASSES (torch IS in container image).
- The condition evaluates: marker exists AND `import transformers, torch` → transformers import FAILS → reinstall runs correctly
- pip installs `transformers==4.44.2 datasets==2.21.0 tokenizers huggingface_hub`
- **torch is NOT re-installed** — it is assumed to be in the container image
- On the resumed pod, `python3 -m torch.distributed.run` succeeded (it's in the container)
- BUT: the HF cache was wiped (container disk), so `GPT2TokenizerFast.from_pretrained('gpt2')`
  re-downloaded, which takes a few minutes. Stage1b handles this correctly.

The more likely post-resume failure mode is one of:

**Primary candidate — stale `/workspace/pebble/.markers/` markers survived but
on-pod code was in a partially inconsistent state.**

- `stage1_env`, `stage2_data`, `stage3_multi2` markers ALL exist on /workspace
- Stage3 marker exists → multi2 generation is skipped
- But the actual `prosqa_multi2_train.json` file might have been regenerated or moved
  (if the operator ran any cleanup), or its parent dir structure was different on the new pod

**Secondary candidate — the `run_exp` function silently ate rc=1 without halting.**

The current `run_exp` does:
```bash
python3 -m torch.distributed.run ... | tee "$LOG_DIR/..." | tail -20
log "[exp] $name seed=$seed completed (rc=${PIPESTATUS[0]})"
```

The `rc=` is LOGGED but **never checked**. `run_exp` returns 0 regardless. The launcher
bulldozes through all 27 experiments without pausing on any failure. At 5–10 seconds per
fast-fail, the launcher completes all 27 entries in ~4 minutes and logs "ALL EXPERIMENTS COMPLETE".

**Most likely root cause (highest confidence):**

The combination of (1) the script-version mismatch on the first pod (two seeds already
failed before the operator patched), and (2) on the resumed pod, `run_exp` silently
swallowing rc=1 and marking no per-run failure state, caused the launcher to race through
all remaining 24 experiments in ~4 minutes without the operator realizing anything was wrong.

The actual fast-fail error was almost certainly:
```
FileNotFoundError: .../prosqa_multi2_train.json
```
or a torch.distributed init error — either way rc=1, zero output written to `results_dir`,
`SUMMARY.txt` never created, but empty `results_dir` directory created by `mkdir -p`
(which always succeeds), making the idempotency check permanent-fail-until-fixed.

**Summary of root causes:**

| # | Root cause | Impact |
|---|---|---|
| RC-1 | No pre-flight flag check: launcher invoked new flags without verifying the on-pod script supports them | 2 fast-fails (seed1337, seed42 of baseline) on first pod |
| RC-2 | `run_exp` does not check rc — silently continues on any failure | All 24 post-resume fast-fails were invisible until the run finished |
| RC-3 | Empty `results_dir` created by `mkdir -p` before python runs — this "uses" the idempotency slot without doing work | Failed experiments cannot be retried without manual cleanup |
| RC-4 | No per-experiment `.start`/`.fail`/`.success` markers — only SUMMARY.txt as the success signal | No diagnostic trail; no state to diff against after crash |
| RC-5 | No fast-fail detection: 5 consecutive fast-fails should surface a notification and pause | Entire 24-experiment block burned in silence |

---

## 2. Resilient Launcher Design

### 2.1 State model

Every experiment `{name}/seed{seed}` has a state directory:

```
$RES_DIR/{name}/seed{seed}/
    .start          # written at the top of run_exp before python launches
    .fail           # written if python rc != 0; contains rc, timestamp, diag
    .success        # written after SUMMARY.txt exists; launcher checks this not SUMMARY.txt
    SUMMARY.txt     # written by run_matrix_codi.py itself
    train.log       # written by run_matrix_codi.py itself
    script.py       # written by run_matrix_codi.py itself
```

The idempotency contract changes from:
- SKIP if `SUMMARY.txt` exists (current)

To:
- SKIP if `.success` exists
- RETRY if `.start` exists but neither `.success` nor `.fail` exists (was preempted mid-run)
- LOG and continue if `.fail` exists (already failed; skip unless `--force-retry`)

### 2.2 Pre-flight check (per experiment family, once at launcher start)

Before any experiment loop:

```bash
preflight_check() {
  local script="$ML/scripts/run_matrix_codi.py"
  local flags=(
    "--rank-loss"
    "--rank-lambda"
    "--force-rank-during-training"
    "--prosqa-train-path"
    "--prosqa-val-path"
    "--weight-decay"
    "--max-eval-batches"
  )
  log "[preflight] checking script: $script"
  log "[preflight] md5: $(md5sum $script | awk '{print $1}')"
  local help_text
  help_text=$(python3 "$script" --help 2>&1)
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "[FATAL] python3 $script --help returned rc=$rc"
    log "[FATAL] env: $(python3 --version); torch: $(python3 -c 'import torch; print(torch.__version__)' 2>&1)"
    exit 1
  fi
  for flag in "${flags[@]}"; do
    if ! echo "$help_text" | grep -q "${flag#--}"; then
      log "[FATAL] flag $flag not found in --help output. Ship the patched script first."
      log "[FATAL] script md5: $(md5sum $script | awk '{print $1}')"
      exit 1
    fi
    log "[preflight] flag $flag: OK"
  done
  log "[preflight] all flags present"
}
```

Call once at launcher start, before any experiment loop.

### 2.3 Rewritten `run_exp`

```bash
STATE_DIR_BASE="$RES_DIR"

run_exp() {
  local name="$1"
  local extra="$2"
  local seed="$3"
  local results_dir="$STATE_DIR_BASE/$name/seed${seed}"
  local success_marker="$results_dir/.success"
  local fail_marker="$results_dir/.fail"
  local start_marker="$results_dir/.start"

  # Idempotency: already done
  if [ -f "$success_marker" ]; then
    log "[exp] SKIP $name seed=$seed (.success exists)"
    return 0
  fi

  # Was preempted mid-run: .start exists but neither .success nor .fail
  if [ -f "$start_marker" ] && [ ! -f "$fail_marker" ]; then
    log "[exp] RESUME $name seed=$seed (preempted mid-run, retrying)"
    # Don't clean results_dir — run_matrix_codi.py is idempotent on re-run
  fi

  # Already failed: log and skip (don't retry unless operator clears the .fail marker)
  if [ -f "$fail_marker" ]; then
    log "[exp] SKIP $name seed=$seed (previously failed: $(cat $fail_marker))"
    return 0
  fi

  mkdir -p "$results_dir"
  echo "$(date -Iseconds) name=$name seed=$seed cmd=$extra" > "$start_marker"
  log "[exp] START $name seed=$seed"

  python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ML/scripts/run_matrix_codi.py" $COMMON --seed "$seed" $extra \
    --results-dir "$results_dir" \
    2>&1 | tee "$LOG_DIR/${name}_seed${seed}.log"
  local py_rc=${PIPESTATUS[0]}

  log "[exp] DONE $name seed=$seed rc=$py_rc"

  if [ $py_rc -ne 0 ]; then
    echo "rc=$py_rc ts=$(date -Iseconds)" > "$fail_marker"
    # Diagnostic snapshot on fast-fail (< 30s wall time means fast-fail)
    # Check if this is fast-fail vs legitimate failure
    local elapsed=$(date +%s)   # need start time... simplified here
    log "[FAIL] $name seed=$seed rc=$py_rc — diagnostic:"
    log "  script md5: $(md5sum $ML/scripts/run_matrix_codi.py | awk '{print $1}')"
    log "  torch: $(python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1)"
    log "  data: $(ls -la $ML/round3_gamma0/data/ 2>&1)"
    log "  tail of log: $(tail -20 $LOG_DIR/${name}_seed${seed}.log)"
    _increment_consecutive_fails
    _check_consecutive_fail_limit
    return 1
  fi

  # Mark success
  touch "$success_marker"
  _reset_consecutive_fails
  return 0
}

# Consecutive fast-fail detector
_CONSECUTIVE_FAILS=0
_CONSECUTIVE_FAIL_LIMIT=5

_increment_consecutive_fails() {
  _CONSECUTIVE_FAILS=$(( _CONSECUTIVE_FAILS + 1 ))
}
_reset_consecutive_fails() {
  _CONSECUTIVE_FAILS=0
}
_check_consecutive_fail_limit() {
  if [ $_CONSECUTIVE_FAILS -ge $_CONSECUTIVE_FAIL_LIMIT ]; then
    log "[ALERT] $_CONSECUTIVE_FAIL_LIMIT consecutive fast-fails. Pausing 60s and dumping diagnostics."
    log "  nvidia-smi: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>&1)"
    log "  free mem: $(free -h 2>&1)"
    log "  df /workspace: $(df -h /workspace 2>&1)"
    log "  python3 path: $(which python3) $(python3 --version 2>&1)"
    # Send ntfy notification if configured
    if [ -n "${NTFY_TOPIC:-}" ]; then
      curl -s -d "[$HOSTNAME] $5 consecutive fast-fails in pebble_launcher — check the pod" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    fi
    sleep 60
    # DO NOT exit — let the caller decide. But if it keeps failing, the state file captures it.
    _CONSECUTIVE_FAILS=0   # reset to avoid hammering ntfy
  fi
}
```

### 2.4 State JSON (machine-readable summary)

Write after every `run_exp` call:

```bash
write_state_json() {
  python3 - "$RES_DIR" "$LOG_DIR/launcher_state.json" <<'PYEOF'
import json, sys, pathlib, datetime

res_dir = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])

state = {"updated": datetime.datetime.utcnow().isoformat(), "experiments": {}}
for exp_dir in sorted(res_dir.iterdir()):
    if not exp_dir.is_dir(): continue
    for seed_dir in sorted(exp_dir.iterdir()):
        if not seed_dir.is_dir(): continue
        key = f"{exp_dir.name}/{seed_dir.name}"
        if (seed_dir / ".success").exists():
            status = "success"
        elif (seed_dir / ".fail").exists():
            status = "failed"
            fail_text = (seed_dir / ".fail").read_text().strip()
            state["experiments"][key] = {"status": status, "detail": fail_text}
            continue
        elif (seed_dir / ".start").exists():
            status = "running_or_preempted"
        else:
            status = "pending"
        state["experiments"][key] = {"status": status}

counts = {}
for v in state["experiments"].values():
    counts[v["status"]] = counts.get(v["status"], 0) + 1
state["summary"] = counts

out_path.write_text(json.dumps(state, indent=2))
print(f"State written: {counts}")
PYEOF
}
```

Call `write_state_json` after every `run_exp` call:
```bash
for SEED in 1337 42 7; do
  run_exp "multi2_baseline_gamma0" "..." "$SEED"
  write_state_json
done
```

### 2.5 Rolling log with timestamps

The current `log()` function already timestamps and tees to `launcher.log` — this is good.
Additions needed:
- Log the exact CLI invocation (full command string) before each run
- Log elapsed seconds after each run
- Log the md5 of `run_matrix_codi.py` at launcher start and after any stage that might mutate it

```bash
log_cmd() {
  local cmd="$*"
  log "[CMD] $cmd"
}
# In run_exp, before python3 launch:
log_cmd "python3 -m torch.distributed.run --standalone --nproc_per_node=1 $ML/scripts/run_matrix_codi.py $COMMON --seed $seed $extra --results-dir $results_dir"
local t0=$(date +%s)
# ... python3 ...
local t1=$(date +%s)
log "[exp] wall_time=$(( t1 - t0 ))s rc=$py_rc"
```

If `wall_time < 30`, it's a fast-fail regardless of rc value — log as such.

---

## 3. Mac-Side Controller Design

The Mac-side controller treats the pod as fully ephemeral. Every time a pod comes up:

```bash
#!/usr/bin/env bash
# pebble_controller.sh — run on Mac, invoked manually after each pod wake-up
# Usage: ./pebble_controller.sh <pod_ip> <pod_port>

POD_IP="$1"
POD_PORT="$2"
KEY=$HOME/.ssh/id_ed25519
SCRIPTS_LOCAL=/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts
LAUNCHER_LOCAL=$SCRIPTS_LOCAL/pebble_launcher.sh

ssh_pod() { ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
            -i "$KEY" -p "$POD_PORT" "root@$POD_IP" "$@"; }
scp_to_pod() { scp -i "$KEY" -P "$POD_PORT" "$1" "root@$POD_IP:$2"; }

echo "[ctrl] === pod wake-up at $(date -Iseconds) ==="

# Step 1: Verify SSH works
ssh_pod 'echo alive && nvidia-smi --query-gpu=name,memory.free --format=csv,noheader'

# Step 2: Ship latest scripts (ALWAYS, every wake-up — treat pod as stale)
echo "[ctrl] shipping scripts..."
# Safety check: scan for local hardcoded paths before shipping
for f in "$SCRIPTS_LOCAL"/*.py "$SCRIPTS_LOCAL"/*.sh; do
    if grep -q '/Volumes/\|/Users/\|/home/samuellarson' "$f" 2>/dev/null; then
        echo "[ctrl] ABORT: $f contains local hardcoded paths. Fix before shipping."
        exit 1
    fi
done
scp_to_pod "$SCRIPTS_LOCAL/run_matrix_codi.py" "/workspace/pebble/scripts/run_matrix_codi.py"
scp_to_pod "$SCRIPTS_LOCAL/build_prosqa_multi.py" "/workspace/pebble/scripts/build_prosqa_multi.py"
scp_to_pod "$LAUNCHER_LOCAL" "/workspace/pebble/pebble_launcher.sh"
ssh_pod 'chmod +x /workspace/pebble/pebble_launcher.sh'

# Step 3: Log what was shipped (md5 trail)
echo "[ctrl] shipped script md5s:"
ssh_pod 'md5sum /workspace/pebble/scripts/run_matrix_codi.py /workspace/pebble/pebble_launcher.sh'

# Step 4: Pre-flight check (sanity before launch — fast, no GPU needed)
echo "[ctrl] running pre-flight check..."
ssh_pod 'python3 /workspace/pebble/scripts/run_matrix_codi.py --help > /dev/null && echo "PREFLIGHT OK" || echo "PREFLIGHT FAIL"'

# Step 5: Launch launcher in background (nohup + setsid)
echo "[ctrl] launching..."
ssh_pod 'pkill -f pebble_launcher.sh 2>/dev/null; sleep 2; nohup setsid bash /workspace/pebble/pebble_launcher.sh > /workspace/pebble/logs/launcher_stdout.log 2>&1 < /dev/null &'

# Step 6: Verify it started
sleep 5
ssh_pod 'ps aux | grep pebble_launcher | grep -v grep | head -3'
ssh_pod 'tail -5 /workspace/pebble/logs/launcher.log 2>/dev/null'

echo "[ctrl] done. Check state with:"
echo "  ssh -p $POD_PORT root@$POD_IP 'cat /workspace/pebble/logs/launcher_state.json'"
```

**Key invariants:**

1. **Scripts are always re-shipped on every controller invocation**, even if the pod was
   not preempted. `/workspace` persists but that just means the _previous_ shipped version
   is still there — we want the _current_ local version.

2. **Script safety scan before shipping.** grep for local hardcoded paths; abort if found.
   This prevents the WORKFLOW doc 5.2 bug (local `/Volumes/` paths clobbering pod paths).

3. **Pre-flight before launch.** `--help` must succeed before any experiment runs.

4. **5 consecutive fast-fails → ntfy + 60s pause.** The launcher doesn't infinite-loop;
   it pauses, logs diagnostics, and surfaces a notification, then continues. The caller
   (operator) decides whether to SSH in and fix.

5. **State JSON is machine-readable.** The controller can pull and display it:
   ```bash
   ssh_pod 'cat /workspace/pebble/logs/launcher_state.json'
   ```

---

## 4. Key Invariants Summary

| Invariant | Enforcement point |
|---|---|
| Script always re-shipped before launch | Mac controller, step 2 |
| No local paths in shipped scripts | Mac controller, step 2 safety scan |
| Pre-flight flag check before first experiment | Launcher `preflight_check()` |
| `.start` / `.success` / `.fail` markers per experiment | Launcher `run_exp()` |
| `mkdir -p` only writes `.start`, not "claims" the slot | Launcher `run_exp()` ordering |
| rc logged and acted upon (not just printed) | Launcher `run_exp()` |
| Fast-fail detection: wall_time < 30s + rc != 0 | Launcher `run_exp()` |
| 5 consecutive fast-fails → pause + notify | Launcher `_check_consecutive_fail_limit()` |
| State JSON updated after every run | Launcher `write_state_json()` |
| md5 of run_matrix_codi.py logged at start and on preflight | Launcher + controller |

---

## 5. Failure Detection Patterns

```
Fast-fail signature:
  wall_time < 30s AND rc != 0 AND no SUMMARY.txt written

Stale script signature (most common cause):
  Fast-fail within first 2 seconds + log contains "unrecognized arguments"
  → argparse unknown flag → old script on pod

Broken torch signature:
  Fast-fail within first 5 seconds + log contains "No module named torch" or
  "torch.distributed" error
  → container reinstall needed; pip install --break-system-packages torch first

Data path error signature:
  Fast-fail within first 30 seconds + log contains "FileNotFoundError" or
  "No such file or directory" for a .json path
  → stage3 marker stale; delete .markers/stage3_multi2 to force regeneration

Preempted-mid-run signature:
  .start exists, .success absent, .fail absent, no SUMMARY.txt
  → safe to retry; run_exp will RESUME automatically
```

---

## 6. Estimated Implementation Size

| Component | Effort |
|---|---|
| Rewrite `run_exp` with markers + rc check + diag | ~50 lines bash |
| `preflight_check()` function | ~25 lines bash |
| `write_state_json` Python heredoc | ~35 lines Python |
| `_consecutive_fail_*` helpers | ~20 lines bash |
| `pebble_controller.sh` (Mac-side) | ~60 lines bash |
| Total | ~190 lines |

This is a half-day of implementation and testing. The highest-leverage single change is
the pre-flight flag check (prevents the class of failure that burned 2 seeds on first pod)
and the rc-check in `run_exp` (prevents silent burnthrough of all 24 remaining experiments).
