#!/usr/bin/env bash
# Local-side loop: every 600s, pull new results from pod and commit to git.
# Runs until killed or 10 iterations (100 min wall-clock per call, so this
# script is relaunched from a bash background every ~100 min).
REPO=/Users/samuellarson/Experiments/learned-representations
POD_HOST=root@38.80.152.148
POD_PORT=31895
KEY=$HOME/.ssh/id_ed25519
LOG=$REPO/experiment-runs/_auto_sync/pull_loop.log
STATE=$REPO/experiment-runs/_auto_sync/pull_state.json

log() { echo "[$(date -Iseconds)] $*" >> "$LOG"; }

for iter in 1 2 3 4 5 6 7 8 9 10; do
    log "=== iteration $iter ==="

    # 1. Probe pod state
    POD_STATE=$(ssh -o ConnectTimeout=10 -i "$KEY" -p "$POD_PORT" "$POD_HOST" 'cat /workspace/pebble/master_state.json 2>/dev/null; echo "---"; ls /workspace/pebble/round4_sft_control/results/ /workspace/pebble/round5_sample_eff/results/ /workspace/pebble/round6_scale/results/ /workspace/pebble/round7_illusion/results/ 2>/dev/null' 2>&1)
    echo "$POD_STATE" > "$STATE"
    log "pod state probed, $(echo "$POD_STATE" | wc -l) lines"

    # 2. Pull any completed result dirs that we do not already have locally.
    for srcdir in \
        /workspace/pebble/round4_sft_control/results/teacher_ce_seed1337 \
        /workspace/pebble/round4_sft_control/results/teacher_ce_seed42 \
        /workspace/pebble/round4_sft_control/results/teacher_ce_seed7 \
        /workspace/pebble/round5_sample_eff/results \
        /workspace/pebble/round6_scale/results/vanilla_gpt2medium \
        /workspace/pebble/round6_scale/results/matrix_gpt2medium \
        /workspace/pebble/round7_illusion/results; do
        base=$(basename "$srcdir")
        parent=$(basename $(dirname "$srcdir"))
        dest="$REPO/experiment-runs/2026-04-13_${parent}/${base}"
        mkdir -p "$(dirname "$dest")"
        if [ ! -f "$dest/.pulled" ]; then
            if ssh -o ConnectTimeout=10 -i "$KEY" -p "$POD_PORT" "$POD_HOST" "test -f ${srcdir}/SUMMARY.txt || test -f ${srcdir}/results.json" 2>/dev/null; then
                log "pulling $srcdir → $dest"
                scp -q -i "$KEY" -P "$POD_PORT" \
                    "$POD_HOST:${srcdir}/SUMMARY.txt" \
                    "$POD_HOST:${srcdir}/results.json" \
                    "$POD_HOST:${srcdir}/final_eval.json" \
                    "$dest/" 2>/dev/null
                # grab rank_projection if it exists
                scp -q -i "$KEY" -P "$POD_PORT" "$POD_HOST:${srcdir}/rank_projection_ablation.json" "$dest/" 2>/dev/null
                touch "$dest/.pulled"
                log "pulled $dest"
            fi
        fi
    done

    # 3. Pull any new log files that aren't too big
    for logname in round4_sft_control round5_sample_eff round6_scale round7_illusion; do
        dest="$REPO/experiment-runs/_auto_sync/logs_${logname}"
        mkdir -p "$dest"
        ssh -o ConnectTimeout=10 -i "$KEY" -p "$POD_PORT" "$POD_HOST" "ls /workspace/pebble/${logname}/logs/*.log 2>/dev/null | xargs -I{} basename {} 2>/dev/null" 2>/dev/null | while read lf; do
            if [ -n "$lf" ]; then
                scp -q -i "$KEY" -P "$POD_PORT" "$POD_HOST:/workspace/pebble/${logname}/logs/${lf}" "$dest/" 2>/dev/null
            fi
        done
    done

    # 4. Commit any changes
    cd "$REPO"
    if ! git diff --quiet experiment-runs/ 2>/dev/null || git ls-files --others --exclude-standard experiment-runs/ | grep -q .; then
        git add experiment-runs/ 2>/dev/null
        git commit -m "auto-sync: pull experiment results from pod ($(date -Iseconds))" 2>/dev/null && log "committed new results"
        git push origin main 2>/dev/null && log "pushed to github"
    fi

    # 5. Sleep 600s before next iteration (maxed out)
    log "sleeping 600s"
    sleep 600
done
log "=== pull_loop finished 10 iterations, re-launch if needed ==="
