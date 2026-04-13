# Wakeup Runbook

When a 1-hour background poll fires a task notification, this is what to do. Purpose: fast react, no re-derivation.

## Step 1: Read the state snapshot (10 sec)

```
cat /tmp/pod_poll_latest.txt
```

Look for:
- Any new entry under `---r4-teacher-summaries---` (means a seed finished)
- `---r5-sample-eff-results---` now non-empty (means Round 5 started)
- `---r6-vanilla-summary---` has Best/Wall (means Round 6 vanilla finished)
- `---r6-matrix-summary---` has Best/Wall (means Round 6 matrix finished)
- `---r7-illusion-tail---` shows progress (means Illusion repro started)
- `---master-queue-tail---` shows "launching next experiment" (means GPU freed and queue advanced)

## Step 2: Pull any completed results (30 sec)

Any directory on pod that has `SUMMARY.txt` but no corresponding local copy:
```
# Round 4 teacher_ce seeds
for seed in 1337 42 7; do
  scp -i ~/.ssh/id_ed25519 -P 31812 root@38.80.152.148:/workspace/pebble/round4_sft_control/results/teacher_ce_seed${seed}/SUMMARY.txt \
    /Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-04-13_round4_vanilla_sft/teacher_ce_seed${seed}_SUMMARY.txt 2>/dev/null
done

# Round 5 sample-efficiency (when done)
mkdir -p /Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-04-13_round5_sample_eff
scp -ri ~/.ssh/id_ed25519 -P 31812 root@38.80.152.148:/workspace/pebble/round5_sample_eff/results/ \
  /Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-04-13_round5_sample_eff/ 2>/dev/null

# Round 6 GPT-2 medium
mkdir -p /Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-04-13_round6_scale
scp -ri ~/.ssh/id_ed25519 -P 31812 root@38.80.152.148:/workspace/pebble/round6_scale/results/ \
  /Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-04-13_round6_scale/ 2>/dev/null

# Round 7 Illusion
mkdir -p /Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-04-13_round7_illusion
scp -ri ~/.ssh/id_ed25519 -P 31812 root@38.80.152.148:/workspace/pebble/round7_illusion/ \
  /Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-04-13_round7_illusion/ 2>/dev/null
```

(`pull_loop.sh` does most of this automatically every 10 min, but a targeted re-sync at wakeup is cheap insurance.)

## Step 3: Append new results to EXPERIMENT_LOG.md

For each newly-finished experiment: open EXPERIMENT_LOG.md, add a dated section under the bottom, include the best accuracy, final accuracy, wall time, and one-sentence interpretation.

Commit and push — the GitHub Actions workflow will auto-deploy any site changes.

## Step 4: Check queue state

```
ssh -i ~/.ssh/id_ed25519 -p 31812 root@38.80.152.148 'cat /workspace/pebble/queue.txt; echo ---; cat /workspace/pebble/master_state.json'
```

If `queue_remaining: 0` AND GPU is idle, that's a problem — nothing is running. Look at what just finished and what should come next. See "Queue priority" below.

## Step 5: Launch next 1-hour poll

```bash
echo "=== 1-HOUR POLL $(date -Iseconds) ==="
sleep 3540
echo "=== 1-HOUR POLL WAKING AT $(date -Iseconds) ==="
ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p 31812 root@38.80.152.148 '
cat /workspace/pebble/master_state.json 2>/dev/null
echo "---r4-teacher-summaries---"
for f in teacher_ce_seed1337 teacher_ce_seed42 teacher_ce_seed7; do
  if [ -f "/workspace/pebble/round4_sft_control/results/${f}/SUMMARY.txt" ]; then
    echo "${f}:"
    grep -E "Best|Final|Wall" /workspace/pebble/round4_sft_control/results/${f}/SUMMARY.txt
  fi
done
echo "---r5-sample-eff-results---"
ls /workspace/pebble/round5_sample_eff/results/ 2>/dev/null
echo "---r6-vanilla-summary---"
grep -E "Best|Final|Wall" /workspace/pebble/round6_scale/results/vanilla_gpt2medium/SUMMARY.txt 2>/dev/null
echo "---r6-matrix-summary---"
grep -E "Best|Final|Wall" /workspace/pebble/round6_scale/results/matrix_gpt2medium/SUMMARY.txt 2>/dev/null
echo "---r7-illusion-tail---"
tail -3 /workspace/pebble/round7_illusion/logs/run.log 2>/dev/null
echo "---master-queue-tail---"
tail -3 /workspace/pebble/master_queue.log 2>/dev/null
echo "---running-procs---"
pgrep -afl "run_vanilla_sft|run_matrix_codi|illusion_repro" 2>/dev/null | head -10
echo "---queue---"
cat /workspace/pebble/queue.txt 2>/dev/null
' 2>&1 | tee /tmp/pod_poll_latest.txt
echo "=== 1-HOUR POLL COMPLETE $(date -Iseconds) ==="
```

Launch with `run_in_background: true` and `timeout: 3700000`.

## Queue priority (after Round 7 Illusion)

Filled in by the gauntlet agent running right now. Check `/private/tmp/claude-501/.../ac7341df03a5f937e.output` for results and then append the surviving candidates to `/workspace/pebble/queue.txt` via ssh.

## Stopping conditions

- **User sends a message:** respond to them, don't just continue the chain silently
- **Pod unreachable for 3 iterations:** stop chain, report the outage, the autonomous loops on pod can't be monitored anyway
- **All queued experiments done AND queue is empty AND no new candidates survived gauntlet:** stop chain, report completion status, wait for user

## Things to NOT do on wakeup

- Don't rewrite the blog post without user direction
- Don't launch new experiments not already queued without gauntlet
- Don't push to git unless you have new result data to commit
- Don't over-verify — the pull_loop.sh + GitHub Actions + pod master queue are all running autonomously; don't duplicate their work
