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
  cancelled/    specs deliberately killed and never to be auto-retried
                (unlike parked_*/, not reversible by convention — e.g.
                `200_laneB_1p31b_arm_per_token_openr1_s0.json.cancelled`,
                the job-200 timeout-drift incident, see Reconciliation
                below). Naming keeps the original ".json" substring so
                anything that greps for it (deploy.sh's dedup, this
                file's own tooling) still recognizes it as a spec.
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

`matrix-thinking/queue/deploy.sh` was a **live landmine** TWICE. It was
first hardened on 2026-07-12 (three independent audit rounds, six real
bugs found and closed — see the script's own header comment and git
history), and hardened a SECOND time on 2026-07-13 (a different failure
mode of the same root cause — see below). The root cause, and why it
matters every time you touch this queue:

**The local `jobs/pending/*.json` tree is a FROZEN snapshot** taken at
generation time. The box's *real* queue moves specs
`pending/` → `claimed/` → `completed/` (or `failed/`, `cancelled/`, or a
`parked_*/` re-gate dir) continuously as workers run them. The two views
diverge the instant the first worker claims a job.

**Incarnation 1 (fixed 2026-07-12): BLIND OVERWRITE.** A naive full-tree
`scp` of `jobs/pending/*.json` onto the box's `~/queue/pending/` — which
is exactly what the pre-2026-07-12 `deploy.sh` did — silently
**resurrects** every already-completed/claimed/failed/parked job: workers
re-claim and re-run them from scratch, wasting GPU-hours and potentially
overwriting good `completed/` results. Not hypothetical: on 2026-07-12
the live box had 113 of 181 local "pending" specs already resolved
elsewhere on disk, including the box's mid-flight 1.31B-parameter
training job.

**Incarnation 2 (fixed 2026-07-13): FILENAME-KEYED DEDUP.** The 2026-07-12
fix enumerated the box's real state and skipped anything already there —
but it matched by **basename** (normalizing only the claim-time `.g<N>`
suffix). A 2026-07-12 re-prioritization round renamed two already-pending
specs on the box for priority ordering (`235_..._s3.json` →
`029_..._s3.json`, `236_..._s3.json` → `030_..._s3.json`) *without*
changing their JSON `"id"` field. By the time this was fixed, both were
claimed, LIVE 392M training jobs. The local repo still has files literally
named `235_....json`/`236_....json` (the generator's `out_path` is always
`f"{id}.json"` — it has no knowledge of an on-box rename). Filename-keyed
dedup saw `029_...`/`030_...` on the box and `235_...`/`236_...` locally
as **disjoint names** — i.e. "not found anywhere" — and would have shipped
a *second*, duplicate spec for two already-running jobs. Confirmed live
against the actual box the day this was found and fixed.

**As of 2026-07-13, `deploy.sh` dedups on each spec's own JSON `"id"`
field, read from file CONTENT on both sides — never on filename.** A
rename on the box no longer matters; the same id anywhere in
`pending/claimed/completed/failed/cancelled/parked_*/` is recognized as
already-resolved regardless of what either copy is named. It is
idempotent (safe to re-run), fails loudly (copies nothing) on any parse
error, missing `"id"` field, or duplicate id (local or remote), and
treats a to-be-placed id that collides with `claimed/` at the final
placement step — which should be structurally impossible given the
plan-time check, but is checked again anyway — as a `*** LIVE-COLLISION
***`-marked, script-**failing** event, never a routine race-skip. See the
script's own header comment for the full incident writeup.

**`DRY_RUN` now DEFAULTS TO 1** (both landmines' root cause was a real run
happening without anyone having looked at the plan first). Pass
`DRY_RUN=0` to actually ship. Before running for real, it is still good
practice to:
1. `bash deploy.sh` (dry run by default) — shows the exact id-keyed
   ship/skip plan, copies nothing. Check the "LIVE JOBS ON THE BOX RIGHT
   NOW" section: every currently-claimed id should appear there, marked
   `*** RENAMED ON BOX ***` if its box filename differs from the local
   one — that line is exactly what would have caught both incarnations
   of this landmine.
2. Skim the rest of the plan: everything already
   claimed/completed/failed/cancelled/parked should show up under
   "already present — SKIPPING", never under "ids safe to ship".
3. If anything looks wrong, STOP and investigate before setting
   `DRY_RUN=0` — do not "just try it and see."

To test changes to `deploy.sh` itself, use `REMOTE_QROOT=<bare-relative-
name>` (a scratch remote dir under the box's own `$HOME`, never a live
path) and `SKIP_STATIC=1` (skips the runner/launcher/docs/`ncr_*.py`
redeploy, which is not parameterized by `REMOTE_QROOT` and would
otherwise still touch live `~/ncr/`) — see the script's own header
comment for the full usage and the tilde-quoting footgun to avoid.
**Never** experiment against the real `~/queue/{pending,claimed,
completed,failed,cancelled}`.

## Reconciliation — catching a fix that never reached a deployed spec

`generate_jobs.py` being fixed does not mean a spec **already deployed**
under that same id picks up the fix — `deploy.sh`'s whole job (see DANGER
above) is to *never* re-ship an id that already has a fate on the box,
which is exactly what stands between a source fix and an already-deployed
snapshot. This bit on 2026-07-13: job 200
(`200_laneB_1p31b_arm_per_token_openr1_s0`) ran with a known-wrong
`--internal-timeout 160000` (44.4h), priced off a stale 0.7135 s/step
rate, when the real measured rate is ~1.3998 s/step (~71h true need).
`generate_jobs.py` had *already* been corrected to price this class of
job at the right rate while job 200's own already-deployed snapshot still
carried the stale one. It self-terminated at ~62% of budget, burning
~44 GPU-hours for nothing (now `cancelled/200_..._s0.json.cancelled`).

`matrix-thinking/queue/reconcile_specs.py` closes this gap. It diffs
every spec currently deployed in the box's `pending/` and `claimed/`
against what `generate_jobs.py` would produce for that same job id if run
right now, and reports drift loudly (`--internal-timeout`/`--steps` at
minimum, plus any other `--flag` that differs). Run it with:

```
DRY_RUN_BYPASS=1 python3 reconcile_specs.py
```

(`DRY_RUN_BYPASS=1` works around this repo's pre-train-gate hook, which
pattern-matches `python3 *.py` as a training launch — a false positive
here, since this script trains nothing.) It is **read-only**: it never
writes to the box and never touches `claimed/`.

**Advisory vs blocking (deliberate, see the script's own docstring for
the full reasoning):** `pending/` drift (or an id the generator no longer
produces at all) is **blocking** (nonzero exit) — the job hasn't started,
so it can and should be fixed (regenerate + `DRY_RUN=0 bash deploy.sh`)
before any worker claims it. `claimed/` drift is **advisory only** — the
job is already running, `queue_worker.sh` already forked its `cmd` before
the check ever runs, rewriting the spec file on disk would not change the
already-running process, and this project's standing rule forbids
killing a live job to "fix" it. Making claimed/ drift block the exit code
would fail this check every time any long-running job is in flight for a
condition nobody can act on — alarm fatigue, not safety. It is still
printed exactly as loudly, in its own section, so a human knows
GPU-hours are being spent on a possibly-stale config.

A narrower, potentially **preventive** version of this check — one that
can actually refuse to launch a job rather than just report on it — is
possible at exactly one moment: right after `queue_worker.sh` claims a
spec (`mv pending/→claimed/`) but *before* it runs `cmd`. At that instant
the file is claimed but nothing has run yet, so blocking costs zero
GPU-hours. This is sketched (`reconcile_specs.py --check-one`) and
documented as a design note in `queue_worker.sh`'s own header, but
**deliberately not wired into the live claim loop** — `generate_jobs.py`
is local-only (not deployed to the box; see `deploy.sh`'s static-file
list), so a real worker-side check needs its own small design (a
deployed drift-manifest, most likely) and audit, not a same-session edit
to a script actively supervising 8 live GPUs.

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
verify with a diff/md5 check before deploying). This is not just style:
an existing function's output silently changing under a live id is
exactly how the Reconciliation section's job-200 finding happened — a
shared pricing constant used by an *old* id's job-builder was updated
alongside a *new* one, so a fresh `python3 generate_jobs.py` now produces
different content for `200_laneB_1p31b_arm_per_token_openr1_s0` than what
was ever deployed under that id (confirmed 2026-07-13: `diff` against a
scratch regeneration shows a different `--internal-timeout` and
`gpu_h_estimate`). Harmless in that specific case (the id is cancelled,
dead, and `deploy.sh`'s id-keyed dedup will never resurrect it under its
old name), but it is real drift a `git diff` on `jobs/pending/` after
regenerating would have caught immediately, and `reconcile_specs.py`
would catch it too if the id were ever redeployed. Run
`DRY_RUN_BYPASS=1 python3 reconcile_specs.py` after any `generate_jobs.py`
edit and before `deploy.sh`, in addition to the diff/md5 check, so drift
in an id you *didn't* mean to touch is caught before it ships. Either
`deploy.sh` (which will correctly ship only the new IDs, id-keyed) or a
targeted `scp` of just the new files by name both work now. See
`matrix-thinking/queue/regate_2026-07-12.md` for worked examples (re-gate
+ refill against a live queue, park reversibly instead of deleting,
surgical delta-scp — predates the `deploy.sh` fix, so those rounds used
the manual delta-scp pattern; either pattern is safe going forward).

## Rung Y (ids `033`/`034`) — the param-axis ATTRIBUTION cell

Queued 2026-07-13 per `matrix-thinking/DSTATE_CONFOUND_PREREG.md` (commit
`1a4add5`) **Option 1**. Generated by `param_axis_rung_y_dstate64_jobs()` in
`generate_jobs.py` — **not hand-authored**, so it is reproducible by
`reconcile_specs.py` and does **not** become a third orphan alongside
`990_t2a3_falconmamba` and `000_..._pricefix`.

**Why.** The param ladder runs `d_state = 64 / 64 / 128` at 14M / 98M / 392M,
so the DeltaNet state (`d_state²`/layer) **quadruples at the single interval a
3-point fit must span**. The fit is *saturated* (3 points, 3 params, zero
residual df) and in closed form `β̂ = (M₉₈ − M₁₄)/(x₉₈ − x₁₄)` **exactly** — the
392M rung contributes **zero** information to the parameter slope, and the
`d_state` step is perfectly aliased with curvature. A RISES verdict therefore
could not be attributed to parameter count, and no re-analysis fixes it.
Rung Y = 392M-scale params at `d_state = 64`, which (a) completes a clean
3-rung `d_state=64` ladder `{14M, 98M, Y}` — second axis held fixed, GATE-A
flips FAIL→PASS — at a **0.46% power cost**, and (b) turns the *existing* 392M
(`d_state=128`) rung into a **matched-params measurement of the state-width
effect** (`γ = M(392M,128) − M(Y,64)`, 1.609% param mismatch), a first-class
result on this program's own fast-weight-capacity thesis.

**Measured param count (NOT arithmetic).** Instantiating the real `DeltaNetLM`
on the box (vocab 50257):

| config | measured params |
|---|---|
| `dm=1536 ds=128 L=16` (existing 392M rung) | **391,869,440** — reproduces §21 V-1 exactly, which *validates the method* |
| `dm=1536 ds=64 L=16` (**rung Y**) | **385,564,672** |

The pre-reg's own §1/§4 hand arithmetic says **385,577,984** and is **wrong by
13,312** (= 16 layers × 832): it omits two `d_state`-scaled terms — the short
causal conv over q/k/v (`3·d_state·conv_size` = 768/layer) and a `d_state`-wide
norm (64/layer). **Immaterial to the design** (delta vs 392M = −1.609% vs the
pre-reg's −1.61%; `S_xx` = 1.0446 clean vs 1.0542 confounded; power loss 0.46%
— all reproduce §4), but recorded here so a later agent does not "correct" the
right number back to the wrong one.

**Timeout pricing — measured, and deliberately assuming NO speedup.** The
`d_state=128` rate was re-measured from the two live 392M jobs (elapsed wall /
steps done, including startup and real 8-way contention): `029` = 39,454 s /
47,000 steps = **0.839 s/step**; `030` = 38,302 s / 45,600 steps = **0.840
s/step**. Rung Y differs *only* by `d_state` 128→64, and every `d_state`-
dependent cost term is monotone non-decreasing in `d_state` (q/k/v/o
projections, the short conv, the kernel's `T·d_state²`) while the dominant FFN
(`8·d_model²`) and the 50,257-way head are **unchanged** — so
`rate(64) ≤ rate(128) = 0.840 s/step` is a **rigorous upper bound, not an
assumed speedup**. Priced at that same rate: **15.76 GPU-h/cell, 31.5 for the
pair**. `--internal-timeout 86400` (24.0 h) = **1.52× that already-conservative
bound** (~1.8× the likely true need) — deliberately *more* headroom than
`029`/`030`'s 72000 s (1.27×). The mispriced-timeout bug has bitten this
project **twice** (job `200`'s 160000 s vs a true ~71 h need, ~44 GPU-h burned;
the `_fixscale_cell` 36000 s default vs 392M's 15.69 h need). A too-small
timeout *guarantees* a wasted run; a too-large one only burns extra on a job
that is already pathological. The timeout is a rail, not a budget.

**Match to `029`/`030` (verified flag-by-flag against their deployed specs).**
11 flags identical — `--corpus`, `--data-dir`, `--d-model 1536`, `--n-layers
16`, `--seq-len 512`, `--batch-size 32`, **`--steps 67547`**, `--ckpt-every
3377`, `--seed 3`, `--frozen-bias-arm per_token`, `--frozen-bias-lambda 0.58`.
Only 4 differ: `--d-state` (128→64, the entire point), `--internal-timeout`
(the repricing above), and the two derived paths (`--ckpt-dir`, `--out`).
Identical `--steps` ⇒ identical token count ⇒ **the common-slice match
(T = 1.10669B) holds by construction and rung Y does not move T.**

**Priority / preemption.** Ids `033`/`034` sort above the entire pending backlog
(`400`–`431`, and `990`), so they claim the first GPUs freed. They **cannot
preempt anything**: `queue_worker.sh` only claims when its *own* GPU is free
and never evicts — so `029`/`030`/`031`/`032`/`000` (and live `231`/`233`/`234`)
are untouchable by this. Note `990_t2a3_falconmamba` already sorted *last*
among pending (`990` > `431`) before rung Y existed; rung Y does not change its
rank relative to the 400-series backlog.

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
- Never rename a spec file already deployed on the box (e.g. for priority
  reordering) as a substitute for actually re-registering it — `mv` it by
  all means (that's how the box's own priority convention works, and
  `deploy.sh`'s dedup is id-keyed specifically so this is safe), but never
  assume the rename is invisible to anything: it changes the filename,
  never the JSON `"id"` field, and any tool that still keys off filename
  (there should be none left in this directory after 2026-07-13, but
  check before trusting a new one) will not see it as the same job.
