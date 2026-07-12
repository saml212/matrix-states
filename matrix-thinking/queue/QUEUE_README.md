# ON-BOX QUEUE SYSTEM — operational reference

Deployed at `/home/nvidia/queue/` on `youthful-indigo-turkey`. Built
2026-07-11 under the PI's standing-queue directive ("make sure there is a
programmatic queue of experiments at all times that will last for at least
two days at full gpu capacity... so the next experiments keep running even
if you [Claude] don't" — the box is uptime-metered, idle GPUs bill anyway).
This file is the box-side copy; the repo copy lives at
`matrix-thinking/queue/QUEUE_README.md` and is the source of truth (deploy
re-syncs this file from there, md5-verified).

## What it is

A directory of plain JSON job-spec files, claimed and run by 8 per-GPU
worker daemons (`queue_worker_g0`..`queue_worker_g7`, one tmux session
each), fully independent of any Claude session, SSH connection, or network
access. Bash + `nvidia-smi` + `python3` only — no coordination service, no
database, no cleverness.

## Directory layout

```
~/queue/
  pending/      job specs waiting to be claimed (filename = priority order)
  claimed/      atomically mv'd here while a worker runs the job
                (filename suffixed .g<N> — which GPU/worker claimed it)
  completed/    validity-checked successes
  failed/       validity-check failures (log preserved, NOT auto-retried)
  parked_*/     job specs pulled OUT of pending/ by a re-gate (never
                claimed by a worker, never deleted — a park is reversible:
                mv back into pending/ to re-activate). Each parked_*/ dir
                carries its own reason at park time; see e.g.
                `parked_k24plus/` (opened 2026-07-12, re-gate note
                `matrix-thinking/queue/regate_2026-07-12.md` — the
                §11.2 K-scaling verdict TRAINABILITY-STILL-LIMITED made
                K≥24 main/deepen cells under the then-current flat-80K
                recipe a known-dead re-measurement, not new information).
  logs/         one <job_id>.log per job (full stdout+stderr)
  worker_g<N>.log   per-worker heartbeat/decision log
  PAUSE         if present: no worker claims a NEW job (running jobs finish)
  STOP          if present: every worker exits cleanly after its current job
```

## The preemption contract

1. **PAUSE sentinel** (`touch ~/queue/PAUSE`) stops all NEW claims
   box-wide. A job already running keeps running to completion. Delete the
   file (`rm ~/queue/PAUSE`) to resume. Use this to hand the whole box to a
   priority sweep without touching any tmux session.

2. **Per-GPU free check, zero coordination.** Before claiming, each worker
   checks its OWN assigned GPU only:
   - `nvidia-smi --query-compute-apps` for that GPU index — ANY listed PID
     means busy. This is checked FIRST and is the primary signal: some of
     this box's own jobs (the NCR K-scaling cells) are tiny (<1 GiB) and
     would slip past a memory-only threshold, so compute-apps (which lists
     every process holding a CUDA context regardless of how little VRAM it
     uses) is what actually catches them.
   - `nvidia-smi --query-gpu=memory.used` < 2048 MiB as a secondary
     fallback (catches a rare leaked/zombie context not listed as a
     compute-app).
   - If either check says busy, the worker sleeps 60s and re-checks — no
     message, no file, no handshake needed on the other side. This is how
     a priority job (e.g. a K-scaling sweep that claims the whole box)
     gets the GPU with zero coordination: it just starts using it, and the
     queue worker notices on its next poll and backs off.

3. **Jobs run as the worker's own child** (`bash -c "$cmd"`, synchronous,
   not backgrounded). Killing a worker's EXACT tmux session name
   (`tmux kill-session -t queue_worker_g<N>`) kills its in-flight job too.
   **Never `pkill`** — a pattern-based kill can match the SSH command
   string invoking the kill itself (CLAUDE.md's standing rule).

4. **Global kill switch:** `touch ~/queue/STOP`. Every worker finishes its
   current job, then exits (the outer supervisor loop's own
   `while [ ! -f STOP ]` also stops re-spawning it). `rm ~/queue/STOP` and
   restart the tmux sessions to resume the whole system.

## Claim atomicity

`mv pending/X claimed/X.g<N>` — `mv` within one filesystem is atomic. If
two workers race the same filename, exactly one `mv` succeeds; the other
gets "No such file or directory" and moves on to try the next candidate in
its own directory listing. No locks, no leader election, no database.

## Resume-safety

On a fresh start (tmux respawns `queue_worker.sh` after a crash or a box
reboot), each worker first sweeps `claimed/*.g<N>.json` for ITS OWN GPU
number and moves any found back to `pending/` for a retry. This is a
**from-scratch retry, not resume-from-checkpoint** — the same disclosed
simplification the project's own fix-at-scale wave already used ("no
compounding-resume logic was built... disclosed, not expected to matter
given zero genuine crashes"). Short/cheap jobs are unaffected; the handful
of long Lane-B jobs (the 1.31B run, ≈38.9 GPU-h) disclose this in their own
job-spec `notes` field. Individual underlying scripts (e.g.
`ncr_earlyln_scale.py`) are ALSO independently resume-safe at the
whole-cell level (skip-if-`COMPLETED`), so a requeue of an already-finished
cell just re-validates instantly rather than re-training.

## Validity, not existence

A job's own `validity_check` command (from its spec, run independently
after `cmd` exits) decides `completed/` vs `failed/` — never the raw exit
code of `cmd` alone. A script can exit 0 with a truncated/malformed output;
the validity check parses the real output JSON and asserts on its content
(status, step count, presence of the expected result keys).

## One bad job never wedges a worker

The worker's outer loop always continues to the next iteration regardless
of a job's outcome (crash, validity failure, malformed spec). A malformed
spec (missing `cmd`/`validity_check`) is routed straight to `failed/`
without ever being executed.

## Harvesting after an outage (for a fresh coordinator)

1. `ssh youthful-indigo-turkey`
2. `tmux ls` — confirm `queue_worker_g0`..`g7` are present (if some are
   missing, `bash ~/queue/launch_workers.sh` is idempotent — it skips any
   session that's already running and starts only the missing ones).
3. `ls ~/queue/completed/ | wc -l` / `ls ~/queue/failed/ | wc -l` — quick
   census. Per-job logs: `~/queue/logs/<job_id>.log`.
4. Raw results land wherever each job's own `output_dir` points (NOT in
   `~/queue/` itself) — see each lane's own results directory:
   - Lane A/C (NCR K-ladder): `~/ncr/results_earlyln_scale/` (main) and
     `~/ncr/results_earlyln_scale_probe/` (rate probes). Harvest via
     `cd ~/ncr && python3 ncr_earlyln_scale.py --harvest --outdir
     results_earlyln_scale`.
   - Lane B (fix-at-scale seed extensions):
     `~/chapter2/deltanet_rd/results/fixscale/train/*.json` — one JSON per
     cell, same schema the CLOSED S13.22 wave used.
   - Lane B (1.31B run + its follow-on probe):
     `~/chapter2/deltanet_rd/results/queue_1p31b/`.
   - Checkpoints (all Lane B; NOT git-tracked, box/SSD only per the repo's
     size-cap archive policy): `/data/fixscale_ckpts/train/<name>/` and
     `/data/queue_1p31b_ckpts/<name>/`.
5. **Recording a verdict is a coordinator/harvester job, never a queue
   job** — nothing in `~/queue/pending/` or the worker script writes to any
   `matrix-thinking/*_DESIGN.md` registry. Raw JSONs + logs only.

## Extending the queue

Regenerate specs from the repo (the generator is the source of truth, not
hand-edited JSON): edit `matrix-thinking/queue/generate_jobs.py`, run
`python3 generate_jobs.py`, then `scp` the new files in `jobs/pending/`
into `~/queue/pending/` on the box (workers pick them up automatically —
no restart needed, they poll `pending/` every 15s when a GPU is free).

**Against a LIVE queue** (some jobs already claimed/completed/parked),
add new job-generating functions to `generate_jobs.py` rather than
editing existing ones (keeps IDs 000-N byte-identical on re-generation —
verify with a diff/md5 check before deploying), then `scp` ONLY the new
files by name — do not run `deploy.sh` or blanket-`scp` all of
`jobs/pending/*.json`, since `deploy.sh`'s own count/md5 guard assumes a
fully-static `pending/` and will correctly refuse once the box's real
`pending/` has diverged from the repo's full generated set (some already
claimed, completed, or parked). See
`matrix-thinking/queue/regate_2026-07-12.md` for a worked example
(re-gate + refill against a live queue, park reversibly instead of
deleting, surgical delta-scp).

## What NOT to do

- Never `pkill` anything on this box (CLAUDE.md standing rule — can
  self-kill the SSH session issuing the command).
- Never edit a `matrix-thinking/*_DESIGN.md` registry from a queue job.
- Never assume a GPU is idle from a Claude-side `nvidia-smi` snapshot —
  the free-GPU gate above is what governs claims, and it re-checks live,
  every 60s, forever.
