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

## DANGER / how to sync safely — READ BEFORE TOUCHING DEPLOYMENT

`matrix-thinking/queue/deploy.sh` was a **live landmine** until it was
hardened on 2026-07-12 (three independent audit rounds, six real bugs
found and closed — see the script's own header comment and git history).
The root cause, and why it matters every time you touch this queue:

**The local `jobs/pending/*.json` tree is a FROZEN snapshot** taken at
generation time. The box's *real* queue moves specs
`pending/` → `claimed/` → `completed/` (or `failed/`, or a `parked_*/`
re-gate dir) continuously as workers run them. The two views diverge
the instant the first worker claims a job. A naive full-tree `scp` of
`jobs/pending/*.json` onto the box's `~/queue/pending/` — which is
exactly what the pre-2026-07-12 `deploy.sh` did — silently **resurrects**
every already-completed/claimed/failed/parked job: workers re-claim and
re-run them from scratch, wasting GPU-hours and potentially overwriting
good `completed/` results (`queue_worker.sh` overwrites-on-name at
completion). This is not hypothetical: on 2026-07-12 the live box had
113 of 181 local "pending" specs already resolved elsewhere on disk,
including the box's mid-flight 1.31B-parameter training job.

**As of 2026-07-12, `deploy.sh` is fixed and safe to run directly against
a live queue** — it enumerates the box's real state (all four
directories, suffix-normalized so a currently-claimed `.g<gpu>.json` job
is correctly recognized) and ships *only* genuinely-new specs, staging
+ md5-verifying + atomically placing them, never touching anything
already claimed/completed/failed/parked. It is idempotent (safe to
re-run) and fails loudly (copies nothing) if it can't reach the box or
enumerate its state. **You no longer need the old surgical
scp-only-the-delta workaround as a safety measure** — `deploy.sh` itself
now does the equivalent check automatically. That workaround
(`matrix-thinking/queue/regate_2026-07-12.md` has several worked
examples) remains a fine pattern for a *targeted* deploy of just a few
new IDs, but is no longer required to avoid resurrecting live jobs.

Before running `deploy.sh` for real, it is still good practice to:
1. `DRY_RUN=1 bash deploy.sh` first — shows the exact ship/skip plan,
   copies nothing.
2. Skim the plan: everything already claimed/completed/failed/parked
   should show up under "already present — SKIPPING", never under
   "specs to ship".
3. If anything looks wrong, STOP and investigate before removing
   `DRY_RUN=1` — do not "just try it and see."

To test changes to `deploy.sh` itself, use `REMOTE_QROOT=<bare-relative-
name>` (a scratch remote dir under the box's own `$HOME`, never a live
path) and `SKIP_STATIC=1` (skips the runner/launcher/docs/`ncr_*.py`
redeploy, which is not parameterized by `REMOTE_QROOT` and would
otherwise still touch live `~/ncr/`) — see the script's own header
comment for the full usage and the tilde-quoting footgun to avoid.
**Never** experiment against the real `~/queue/{pending,claimed,
completed,failed}`.

## Extending the queue

Regenerate specs from the repo (the generator is the source of truth, not
hand-edited JSON): edit `matrix-thinking/queue/generate_jobs.py`, run
`python3 generate_jobs.py`, then either run `bash deploy.sh` (now safe
against a live queue, see DANGER section above) or `scp` the new files in
`jobs/pending/` into `~/queue/pending/` on the box directly (workers pick
them up automatically — no restart needed, they poll `pending/` every
15s when a GPU is free).

**Against a LIVE queue** (some jobs already claimed/completed/parked),
add new job-generating functions to `generate_jobs.py` rather than
editing existing ones (keeps IDs 000-N byte-identical on re-generation —
verify with a diff/md5 check before deploying). Either `deploy.sh` (which
will correctly ship only the new IDs) or a targeted `scp` of just the new
files by name both work now. See
`matrix-thinking/queue/regate_2026-07-12.md` for worked examples (re-gate
+ refill against a live queue, park reversibly instead of deleting,
surgical delta-scp — predates the `deploy.sh` fix, so those rounds used
the manual delta-scp pattern; either pattern is safe going forward).

## What NOT to do

- Never `pkill` anything on this box (CLAUDE.md standing rule — can
  self-kill the SSH session issuing the command).
- Never edit a `matrix-thinking/*_DESIGN.md` registry from a queue job.
- Never assume a GPU is idle from a Claude-side `nvidia-smi` snapshot —
  the free-GPU gate above is what governs claims, and it re-checks live,
  every 60s, forever.
- Never blank-`scp` `jobs/pending/*.json` onto the box by hand, bypassing
  `deploy.sh` — you'd lose the resurrection guard described above and
  reintroduce the exact bug `deploy.sh` was hardened to prevent.
