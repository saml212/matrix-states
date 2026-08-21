# K-SCALING PRE-LAUNCH RESOURCE / PLACEMENT RED-TEAM — ROUND 1

**Date:** 2026-08-21 · **Commit:** 002a6c5 · **Payload:** 32 specs (2 calibration + 30 sweep), ledger ≈27.6 GPU-h
**Ceremony tier:** 10–50 GPU-h ⇒ audit + pre-launch resource/placement red-team (this document is the second leg)
**Scope:** resources, placement, failure topology, timeline, provenance. Science/statistics are the audit's lane.
**Box access:** read-only inspection only. Nothing queued, nothing killed, nothing modified on the box.

## VERDICT: **BLOCK** — 2 launch-losing findings, both one-decision fixes (no rebuild, no re-audit of the remaining 30 specs)

After L1 and L2 are applied the wave is launch-ready. Everything else is degrading or note-level.

---

## 0. Live box state at time of review (the context that reframes the whole question)

```
GPUs 0-7:      0% util, 0 MiB used, ZERO compute apps
~/queue:       pending=0  claimed=0  completed=454  failed=0  fallback_pool=0
sentinels:     no PAUSE, no STOP, no DISK_CRITICAL; FALLBACK_POOL_DRY raised 2026-08-21 21:45
disk:          / 68% (63G avail, 47.1G of headroom before the 92% auto-PAUSE)
               /ephemeral 5% (294G of 5.9T used)
workers:       all 8 queue_worker_g* tmux sessions alive; 4 cron supervisors ticking
```

**All 8 GPUs are dark right now and have been since ~21:45, with an empty fallback pool.** The remaining
~1-week runway is 8 × 168 h = **1344 GPU-h of box capacity**. The entire K-scaling program is 25.6–27.6 GPU-h
of that — **1.9–2.1%**. After this wave lands, **1318 GPU-h (98.1%) of the window is still unallocated.**

This number governs every placement trade below: **GPU-hours are not the scarce resource, and neither is wall
time. Coordinator/audit latency and risk are.** Any optimization that spends latency or risk to buy GPU-hours
or minutes of wall is trading the abundant resource for the scarce one.

---

## 1. Findings, ranked

### LAUNCH-LOSING

#### **L1 — Specs 0134 and 0137 are byte-for-byte duplicates of the calibration cells 0100 and 0101, sharing `--cell-id`, `--out`, and `--ckpt-dir`.**

Verified by exact string comparison of `cmd`, `validity_check`, and `output_dir`:

| pair | identical `cmd` | identical `validity_check` | shared cell_id / out / ckpt-dir |
|---|---|---|---|
| 0100 ↔ 0134 | **True** | **True** | `kscaling_K32_primary_s0` |
| 0101 ↔ 0137 | **True** | **True** | `kscaling_K32_compB_s0` |

All three collide:
`/ephemeral/kscaling/results/kscaling_K32_primary_s0.json`,
`/ephemeral/kscaling/ckpts/kscaling_K32_primary_s0/` (and the runner's `stop_file` inside it).

Three distinct failure modes, in increasing order of likelihood:

1. **Concurrent same-cell execution.** If both are ever in `pending/` at once — which is the *natural* operator
   action under the "keep GPUs hot" directive ("queue all 32") — two workers on two different GPUs claim them and
   run the identical cell writing the identical checkpoint path and JSON. `atomic_torch_save` (tmp + rename)
   prevents a corrupt file, but the result is last-writer-wins with two 20 000-step runs interleaving on one
   output. Both validity checks may pass. The curve then carries a point of unknown provenance.
2. **Silent destruction of the gate artifact.** Even perfectly serialized, the sweep overwrites the very
   calibration JSON and checkpoint that *licensed* the sweep. The evidence for the LICENSE-SWEEP decision is
   gone the moment the sweep reaches its K=32 block.
3. **Log concatenation.** The `cmd` ends in `tee -a …kscaling_K32_primary_s0.log` (**append**). The second run
   concatenates onto the first, so any harvester that greps for the first occurrence of a metric reads the wrong
   run.

There is no scientific gain to offset this: same K, same recipe, same seed, same runner — 0134/0137 are a
re-run of 0100/0101, not a new sample.

**Fix (chosen):** delete `0134_ncr_kscaling_K32_primary_s0.json` and `0137_ncr_kscaling_K32_compB_s0.json`;
count 0100/0101 as the K=32 seed-0 cells of the curve. Ledger goes **32 → 30 trained cells, 27.62 → 25.63
GPU-h**, and the gated remainder becomes 24 cells = exactly 3 waves of 8 (see §3).
*Alternative if a genuine repeat is wanted:* re-key 0134/0137 to seeds 3/4 with fresh cell-ids — but that is
duplicate compute for no registered question, so it is not recommended.

#### **L2 — Staging the gated sweep specs in `~/queue/fallback_pool/` silently bypasses the calibration gate within 60 seconds.**

`gpu_hot_monitor.sh` runs every minute from cron and its REFILL block fires on:

```
if [ "$n_idle" -gt 0 ] && [ "$n_pending" -eq 0 ] && [ ! -f "$Q/PAUSE" ]; then
    ... promote up to min(n_idle, 8) specs from fallback_pool/ -> pending/
```

During calibration that predicate is **exactly true**: 2 GPUs busy, **6 idle**, `pending=0`, no PAUSE. Within one
cron tick the monitor promotes 6 gated sweep specs into `pending/`, the workers claim them, and the 25.6 GPU-h
sweep launches **without the gate ever being read**. The `idle_fallback_daemon.sh` does the same on a 3-hour
timer.

This is also a doctrine violation, not merely a mechanical one — `idle_fallback_daemon.sh` states the pool
eligibility contract verbatim: the pool holds *"ONLY flat specs … independently runnable in any order, with NO
intra-wave dependencies, stage gates, or staged-escalation semantics (filename-order promotion cannot honor
them)."* Double-gated specs are categorically pool-ineligible.

**Fix:** hold the gated specs **outside `~/queue/` entirely** (e.g. `~/kscaling_staging/`) until the gate returns
LICENSE-SWEEP, then `mv` them directly into `pending/`. Never into `fallback_pool/`.

---

### DEGRADING

#### **D1 — The K=24 anchor re-score cannot run as built. Reproduced live on the box.**

Patch R1 bumps the runner tag; the 58 cells of record carry the old one:

```
patched RUNNER_TAG                       = ncr_kscaling_runner_v1
mob_g3b31_compA_s1.ckpt.pt  runner_tag   = ncr_gate3_wave1_runner_v1
```

`ncr_lm_wave1_runner.py:1162` asserts `ckpt.get("runner_tag") == RUNNER_TAG`, catches the `AssertionError`, and
returns `None`; `kscaling_battery.py:112` then raises `LoudFailure("CHECKPOINT FAIL … failed load_checkpoint
validation")`. Live reproduction against a real cell of record:

```
patched load_checkpoint -> NoneType None
  [checkpoint] .../mob_g3b31_compA_s1.ckpt.pt failed to load/validate (AssertionError()) -- treating as ABSENT
```

The failure is **loud**, which is the correct design — no silent bad curve point. But the consequence is that the
ledger line *"K=24 anchor re-score (eval-only, existing ckpts) — 6 cells, 0.02 GPU-h"* and §13 open item 3 are
**unrunnable**, and the capability-breadth curve loses its anchor at the single K where all 58 cells of record
and every prior verdict live. A 5-point curve with a hole at the one K that connects to the existing program.

**Fix (cheap, recommended):** allow the pinned tag on **read-only scoring** in `kscaling_battery.py`. This is
scientifically sound because (a) patch S3 states and the smoke confirms that at K ≥ 20 `doc_left_pad` is 0, so
*"K ≥ 20 (in particular the K=24 anchor) stays BYTE-IDENTICAL to the pinned construction"*, and (b) the battery's
independent d-guard (`ck_d != KS.D_NCR` → LoudFailure) still makes it impossible to score a K=24 checkpoint as
any other K. The resume path in the runner must keep the strict tag check — only the scorer relaxes.
**Fix (expensive alternative):** retrain 6 K=24 cells with the patched runner, ≈5.0 GPU-h.

#### **D2 — Eval-phase memory was never measured; the house lesson is uncovered by the smoke.**

The smoke's `L_throughput_memory_utilisation` leg times **training only**: `n_steps, bs = 30, 32`, forward +
backward + `opt.step()`. Nothing in `kscaling_smoke.py` calls `eval_both_arms`, and no item runs at
`--eval-batch-size 64`. Measured peaks are therefore train-only:

| K | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|
| peak_mem (GB) | 5.215 | 5.217 | 5.536 | 5.960 | 6.375 | **6.780** |
| SM util median/max | 71/73 | 71/77 | 82/88 | 85/88 | 95/100 | **97/100** |

Assessed **LOW risk on evidence, not modeling**:
* 80 GB per H100 vs a measured 6.78 GB train peak at the largest K ⇒ **73 GB of headroom**; the smoke's own
  `packing_headroom_by_mem_80gb` reads 11–14.
* **55 cells of record ran the identical eval path** (`eval_both_arms` → `eval_arm_at_hops`, default
  `--eval-batch-size 64`) at K=24 to `status=COMPLETED`, `gpu_h=0.8116`. Eval is forward-only under the same
  code path; the K=32 delta is `t_in` 174→230 and `d_ncr` 25→33.
* Even a hypothetical 5× eval spike over train peak (34 GB) fits one-per-GPU **and** two-per-GPU.

**Action:** not a blocker. Record peak eval memory from wave 0 as the first real measurement, and close the
measurement gap in the smoke before the next K wave.

#### **D3 — `GPU_UNDERUTILIZED` is a false alarm under any partial wave.**

The monitor means utilization across **all 8** GPUs, then alarms on `n_claimed > 0 && mean < 50` for 10
consecutive minutes. Under the as-designed 2-cell calibration: 2 GPUs at ~97%, 6 idle ⇒ mean = **24%** ⇒ the
alarm fires after 10 minutes even though *both running cells are at 97%*. The recommended 6-cell wave 0 gives
6 × 97 / 8 = **73%** and avoids it. The same false positive recurs on any ragged sweep tail.
**Action:** do not read that flag as a bug during partial waves; scenario B avoids it by construction.

#### **D4 — Design §11.1's cost table does not match the archived smoke JSONs.**

| K | §11.1 s/step | measured (`smoke_results/`) | §11.1 util | measured util |
|---|---|---|---|---|
| 12 | 0.1268 | 0.1223 | 72/74 | 71/73 |
| 16 | 0.1205 | 0.1238 | 72/74 | 71/77 |
| 20 | 0.1195 | **0.1328** | 86/88 | 82/88 |
| 24 | 0.1289 | 0.1271 | 89/90 | 85/88 |
| 28 | 0.1372 | 0.1387 | 94/96 | 95/100 |
| 32 | 0.1529 | 0.1485 | 93/100 | 97/100 |

The table appears to come from an earlier smoke run than the one archived. Recomputing the ledger from the
**archived** numbers (× the same 1.17 overhead) gives 25.97 GPU-h for the 30-cell sweep vs the design's 25.63 —
**+1.3%**, so no decision changes and the 1.17 multiplier remains validated against the measured K=24 cell of
record (0.8116 GPU-h actual vs 0.838 projected). Re-point §11.1 at `smoke_results/` before publication.

---

### NOTE

* **N1 — Disk.** The guard in `gpu_hot_monitor.sh` watches **`/` only**; `/ephemeral`, which receives every
  checkpoint, is unguarded. No live risk: `/ephemeral` is at 5% (294 G of 5.9 T) and this wave adds
  30 × 2.2 GB ≈ **66 GB** (measured from a cell of record: one atomically-saved `<cell_id>.ckpt.pt`, 2.2 GB,
  overwritten in place — not one file per `--ckpt-every`). Root receives only queue job logs (~180 KB each,
  ~5.8 MB total) against **47.1 GB** of headroom before the 92% auto-PAUSE, so a mid-wave `DISK_CRITICAL` from
  this payload is effectively impossible. If `/ephemeral` ever did fill, nothing would pause the queue and cells
  would die at `torch.save` — the blind spot is worth closing on its own schedule.
* **N2 — Provenance is clean.** Spec ids `0100`–`0139` appear in **none** of the 454 completed / 0 failed / 0
  pooled entries (the existing `100_`–`107_` ids are distinct filenames and do not collide).
  `/ephemeral/kscaling/results` and `/ephemeral/kscaling/ckpts` **do not exist yet** — nothing to overwrite. The
  `kscaling_*` namespace is disjoint from the `mob_g3b31_*` cells of record
  (`/home/nvidia/ncr_g3b31_contrastive/results/`) and from `/ephemeral/reseed_ckpts/` (114 G, `mob_*_ckpts`).
  The runner's per-cell `stop_file` lives at `<ckpt-dir>/STOP` and cannot collide with `~/queue/STOP`.
  **The only collisions in the payload are L1's, and they are internal.**
* **N3 — No pkill/tmux-name hazard.** `grep -rn "pkill|killall|kill -|kill-session|tmux kill"` over every
  script in `matrix-thinking/queue/` returns **nothing**. The documented self-kill footgun does not apply.
  Jobs run as synchronous children of `queue_worker.sh`, so preemption by exact tmux session name works as
  designed, and nothing in the worker path pattern-matches on job names.
* **N4 — Crash-recovery double-run residual (pre-existing, not introduced here).** If `queue_worker.sh` dies
  without its python grandchild, the restart's reclaim loop returns the claim to `pending/` while the orphan
  still holds the GPU; a *different* GPU's worker can then run a second copy of the same cell against the same
  `--ckpt-dir`. `atomic_torch_save` prevents corruption but it is last-writer-wins. Low probability, and the
  same class as L1. **Post-wave check:** assert 30 completed specs with 30 distinct recorded `cell_id`s.
* **N5 — Results JSONs go to `/ephemeral`,** but STATE's BOX DISK POLICY says *"Results JSONs stay on root."*
  Not a launch risk; archive them off `/ephemeral` promptly after the wave.
* **N6 — `--mode calibration` on the sweep specs is CORRECT, not a bug.** It is the runner's training mode and
  every one of the 58 cells of record used it (`mob_g3b31_compA_s0.json`: `mode = calibration`). Flagged so a
  later reviewer does not "fix" it.
* **N7 — Failure topology is sound.** Per-cell isolation is real: one cell per spec, `validity_check` (not exit
  code) decides `completed/` vs `failed/`, a failed cell is not auto-retried, and the worker's outer loop
  continues unconditionally so one bad cell never wedges a GPU. There is no shared driver. A malformed spec
  routes to `failed/` rather than wedging. **With 30 specs in `pending/`, neither the monitor's REFILL nor
  `idle_fallback_daemon.sh` can fire** (both require `n_pending == 0`), so there is no bad interaction during
  the wave — the hazard is only L2's, during the *gate*.

---

## 2. Packing election — **DECLINE 2-per-GPU packing**

The builder measured a real effect and priced it honestly. It does not survive contact with the actual schedule.

**What the measurement shows (one GPU, in isolation):** two K=12 cells packed on one GPU take utilization
72% → 99%, per-cell s/step 0.1268 → 0.150–0.162 (**+23% GPU-h per cell**), and **1.63× wall throughput for the
pair**. All correct.

**What it is worth in the actual 8-GPU schedule.** Simulating list-scheduling in filename order (which is how
`queue_worker.sh` claims: `ls "$PENDING" | sort`, earliest filename wins):

| scenario | cells/jobs | GPU-h | makespan | ideal (GPU-h/8) | occupancy |
|---|---|---|---|---|---|
| A — as-built 30-cell sweep, 1/GPU | 30 | 25.63 | **3.49 h** | 3.20 h | 91.8% |
| B — L1-deduped, 24 gated cells, 1/GPU | 24 | 19.66 | **2.50 h** | 2.46 h | **98.3%** |
| C — B with K=12/K=16 packed 2-per-GPU | 18 jobs | 15.94 occupied | **2.45 h** | — | 81.3% |

**Packing buys 0.05 h — three minutes — of makespan, on a ~168 h runway. That is 0.03% of the window.**

The reason the 1.63× pair-throughput gain does not convert into wall-clock savings: packing pays when cells
outnumber GPU-slots enough that cells must queue *behind each other on the same GPU*. Here the ratio is 24/8 = 3
with near-uniform durations (0.777–0.995 h), so plain list-scheduling already fills the box at **98.3%
occupancy** — there is almost no idle to reclaim. Packing reduces GPU-hours *occupied* (15.94 vs 19.66) but does
not shorten the critical path, and GPU-hours are the resource we have 1318 spare of.

**What declining costs on the utilization directive: nothing.** The operative threshold is *"sustained <50% = a
bug."* Measured medians are **71%** (K=12/16), **82%** (K=20), **95%** (K=28), **97%** (K=32) — every cell clears
it, 18 of 24 gated cells run at 82–97%, and no cell is anywhere near the bug line.

**What electing packing would cost:**
1. **6 new paired specs must be built** — and the audit currently in flight covers the 32 as-built specs, so
   packed specs need a fresh audit round. Under a ~1-week runway, spending hours-to-a-day of audit latency to
   save 3 minutes of wall inverts the scarce and abundant resources exactly.
2. **Coupled fate for 12 of 24 cells.** One `cmd` and one `validity_check` per pair: a single cell's OOM or
   convergence failure routes *both* to `failed/`.
3. **A documented regression of the queue's preemption contract.** `queue_worker.sh` states jobs run
   *"as this worker's own child (`bash -c "$cmd"`, synchronous, no backgrounding) so killing this worker's EXACT
   tmux session name kills its in-flight job too (the intended preemption contract)."* Packing requires
   `cmd1 & cmd2 & wait`; the backgrounded grandchildren survive a session kill as GPU-holding orphans. That is a
   change to the mechanism supervising 8 live GPUs, made to save three minutes.

**Recommendation: decline packing; adopt scenario B.** Revisit packing only if a future wave has cells ≫ GPUs
(ratio ≳ 8) or strongly non-uniform durations, where the pair-throughput gain actually lands on the critical
path — and build it then as its own audited spec form, not as an edit to the live worker.

---

## 3. The keep-GPUs-hot schedule (respecting the calibration gate)

**The gate stays intact.** It is adjudicated on 0100/0101 (K=32, both recipes, seed 0) exactly as pre-registered,
and the 24 cells at K ≤ 28 do not launch until it returns LICENSE-SWEEP.

**The idle problem it creates.** As designed, wave 0 is 2 cells on 2 GPUs for ~1 h ⇒ **6 GPUs dark = 5.97 GPU-h
burned to nothing**, plus 8 GPUs dark for the whole adjudication gap.

### Wave 0 (T+0 → T+0.99 h): all six K=32 cells, six GPUs

Run `0100, 0101, 0135, 0136, 0138, 0139` (0134/0137 deleted per L1). 0.995 h each, 100% occupancy on 6 GPUs.

**Why the four extra K=32 cells may launch pre-gate without weakening the gate.** They are the same K and the
same recipe pair as the gate cells, differing only in seed. Trace both surviving branches:
* **LICENSE-SWEEP** → they were going to run anyway; nothing is wasted.
* **FRONTIER-AT-K\*=32** (leg 3 fails, legs 1+2 pass) → they are *exactly* what powers the frontier claim to
  n=3 instead of n=1. This branch is made **stronger**, not wasted.
* They are wasted **only** if leg (1) Gate-0 convergence or leg (2) in-distribution capability fails — which the
  spec itself classes as *"an instrument/convergence problem, not a science result."*

Exposure in that one branch: 4 × 0.995 = **3.98 GPU-h = 0.30% of the window**. In exchange, wave-0 idle drops
from 5.97 GPU-h to 1.99 GPU-h, and the `GPU_UNDERUTILIZED` false alarm (D3) is avoided.
**Disclosure requirement:** record 0135/0136/0138/0139 as **pre-gate launches** so no later report describes
them as gate-licensed.

Remaining 2 GPUs: the K=24 anchor re-score (after the D1 fix) and the 58-cell battery re-scores are eval-only and
finish in minutes, so ~1.99 GPU-h realistically stays dark unless real filler exists (see below).

### Gate adjudication (T+0.99 → T+0.99+T_adj) — **the single largest idle risk in the plan**

Every hour here is **8 GPU-h dark**. With a live coordinator, T_adj ≈ 0.5 h. With the gate landing overnight,
T_adj can be 8 h+ and **dominates the entire schedule** — one unattended gate costs more GPU-hours (64) than the
whole sweep (25.6).

Two mitigations, both required:
1. **Launch wave 0 only when a coordinator can be present ~1 h later.** This is a scheduling constraint, not a
   nice-to-have.
2. **Put audited filler in `fallback_pool/` before wave 0 finishes.** The monitor auto-promotes on
   (`n_idle > 0` ∧ `n_pending == 0`) within 60 s — which is precisely the adjudication-gap state. This is the
   *legitimate* use of the pool, and it is exactly what L2 forbids for the gated specs. **Filler must not be
   K-scaling sweep specs.**

### Waves 1–3 (T+0.99+T_adj → T+3.49+T_adj): the 24 gated cells

24 cells = **exactly 3 waves of 8**, no ragged tail. Makespan 2.50 h at **98.3% occupancy**. Then
`kscaling_battery` scoring (eval-only, 0.12 GPU-h).

### Realistic wall to curve-in-hand

```
0.99 (wave 0) + T_adj + 2.50 (waves 1-3) + ~0.2 (battery + harvest)  =  3.7 h + T_adj
     coordinator live (T_adj ~ 0.5 h)  ->  ~4.2 h
     gate lands unattended             ->  T_adj dominates; 8 GPU-h dark per hour of gap
```

### Filler: there is none today, and that is a bigger problem than placement

`fallback_pool = 0` and `FALLBACK_POOL_DRY` has been raised since 21:45. The standing directive requires a
**durable ≥2-day on-box queue that survives coordinator death**; the box currently has **zero** hours of runway
queued. In-lane candidates named by the repo — the K=24 anchor re-score (needs the D1 fix), the 58-cell battery
re-scores, the compB entity-adapter-drift analysis (zero-GPU) — are all **minutes-scale** and cannot fill even
the adjudication gap, let alone the 1318 unallocated GPU-hours.

**Real filler at GPU-hour scale needs its own design/audit ceremony** (the pool eligibility contract forbids
un-audited specs, and that gate should not be weakened to solve an idleness problem). This is a separate,
higher-value workstream than any placement decision in this document: **placement optimization on this wave can
move at most ~0.05–6 GPU-h; the empty pool is leaving ~1318 on the table.**

---

## 4. Summary of required actions

| # | severity | action | cost |
|---|---|---|---|
| L1 | launch-losing | Delete specs 0134 and 0137; count 0100/0101 as the K=32 seed-0 cells. Ledger 32→30 cells, 27.62→25.63 GPU-h | 2 file deletions; **saves 1.99 GPU-h** |
| L2 | launch-losing | Stage the 24 gated specs **outside `~/queue/`** until LICENSE-SWEEP. Never in `fallback_pool/` | procedural |
| D1 | degrading | Accept the pinned `runner_tag` for **read-only scoring** in `kscaling_battery.py` (d-guard still enforces K) | 1-line fix, or 5.0 GPU-h to retrain |
| D2 | degrading | Record peak **eval** memory from wave 0; close the smoke's eval-measurement gap next wave | free |
| D4 | degrading | Re-point design §11.1 at `smoke_results/` (ledger impact +1.3%, no decision change) | doc edit |
| — | placement | **Decline packing.** Adopt scenario B: wave 0 = 6 × K=32 on 6 GPUs, then 3 clean waves of 8 | — |
| — | schedule | Launch wave 0 only with a coordinator available ~1 h later; stage non-K-scaling filler in `fallback_pool/` first | — |
| N4 | note | Post-wave: assert 30 completed specs with 30 distinct `cell_id`s | free |
