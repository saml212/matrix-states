# NCR K=32/d=33 budget-rescue dissociation probe (Q4) — archive

**Recorded as:** `matrix-thinking/NOVEL_ARCH_WATERFALL.md` §11.6.
**Pre-registration:** `matrix-thinking/NCR_MAPPING_LAW_DESIGN.md` §4 (`edb1337`).
**Build/audit/deploy:** `matrix-thinking/queue/regate_2026-07-12.md` §9 (queue jobs `192-199`).

## What this is

8 cells: K=32, d=33 (tight-spare, d=K+1), budget in {2x=160,000 steps,
4x=320,000 steps}, seeds {0,1,2,3} each. This is the final probe on the
NCR K-axis mapping-law program -- it tests whether extra training budget
rescues K=32's tight-spare arm (S11.5's "least dead" cell class, closed
`CLOSED-AT-THIS-K` at 1x) into robust convergence AND whether far-depth
composition comes along with any such rescue.

## Files

- `budget2x_earlyln_K32_s{0-3}.json` -- 2x budget cells (160,000 steps),
  pulled from `~/ncr/results_earlyln_budget2x/` on `youthful-indigo-turkey`.
- `budget2x_earlyln_K32_s{0-3}.axis_c_lock.json` -- their Axis-C lock
  siblings.
- `budget4x_earlyln_K32_s{0-3}.json` -- 4x budget cells (320,000 steps),
  pulled from `~/ncr/results_earlyln_budget4x/` on the same box.
- `budget4x_earlyln_K32_s{0-3}.axis_c_lock.json` -- their Axis-C lock
  siblings.
- `md5_manifest.txt` -- md5 of every file above; independently cross-checked
  against `md5sum` run directly on the box for all 16 files, byte-identical.

## Headline result

**S4.5's ANOMALY check fires** (delta non-monotonic across the 1x->2x->4x
trajectory in 3 of 4 seeds -- the same trigger and seed-count as the K16 4x
precedent, `NOVEL_ARCH_WATERFALL.md` S11.4). Per S4.5/S4.6's own pinned
rule, an ANOMALY pre-empts the mechanical 4-way verdict map
((a) BUDGET-RESCUES-BOTH / (b1) SURVIVES-WEAK / (b2) DIES /
(c) BUDGET-DOES-NOTHING) entirely -- no label is assigned.

**This does not leave WAVE-1b/K=48 undecided, however.** Outcome (a) -- the
only one that would unblock K=48's own d(K) grid -- requires BOTH
BUDGET-CONVERGES=TRUE (>=3/4 CONVERGED at some budget) AND that budget's
median failure front >= K=32's own h*=253. Neither half is met: Gate-1
tops out at 2/4 CONVERGED (4x, still short of the 3/4 ROBUST bar) and
the failure front is pinned at the trivial K-3 rung (=29) in literally
all 12 cells now on record for this arm across all three budgets tested
(1x/2x/4x). WAVE-1b (K=48's reserved job IDs `513-524`) and the unparked
2K-reference (`108-111`) stay BLOCKED -- S11.5's `CLOSED-AT-THIS-K`
verdict for K=32 stands, now confirmed under budget escalation as well
as at 1x, and this closes the K-axis book at K=32 with no further probe
recommended or licensed by this record.

## GPU-h

2x (4 cells): 4.2042 GPU-h. 4x (4 cells): 7.8803 GPU-h. Probe total:
**12.0845 GPU-h realized** (88.5% of the 13.651 nominal plan, no breaker
trips).

## Integrity (independently re-verified against the raws, this session)

8/8 `status=COMPLETED`, 8/8 `train.step` matches the requested budget
exactly, 8/8 `blank_out.passed=True`, 8/8 `d==33`, 8/8
`eval.reducer_signature.flagged=False`, 8/8 `train.n_skipped_steps=0`,
0/8 `axis_c_lock_sha256` mismatches (independently recomputed from each
lock file's own content, not merely cited).

Full tables (2x, 4x, the 1x baseline for contrast, the per-seed 3-point
trajectory, and the ANOMALY-check derivation) are in
`NOVEL_ARCH_WATERFALL.md` S11.6 -- this file is a pointer/index for the
archive, not a duplicate of record.
