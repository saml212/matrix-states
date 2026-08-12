# K-wall build-release micro-smokes — box-side, exact commands

Source: `NCR_KWALL_CHARACTERIZATION_DESIGN.md` §4 "KW2.8/KW3.13/KW4.6
close-out — the REFUTED accepted-risk replaced with a real, owned gate
(F3, Rev 3)" (design lines ~2850-2916). Build charter item 3 (R5 §9 /
restated in every subsequent round's §8): "The 3 micro-smokes (K=26/28/30)
pass before queue-eligibility."

**Why these are box-side, not run by this build on the Mac.** Each smoke
is a real 500-step training run (forward+backward+optimizer.step() x 500,
plus, if it reaches COMPLETED status, the full post-train instrument
sequence — z_dump, deep probe, Axis-C lock, trust screen, blank_out_check,
eval_cell). Measured directly during this build (see `BUILD_REPORT.md`):
a *single* forward+backward+optimizer.step() at K=26..30/d=K+1 on this
Mac's CPU (no GPU) takes low seconds, but a full 500-step run at the
harness's real `TRAIN_BATCH=256` took **>120s without finishing even one
step's abort-check boundary** in a timed probe (the CPU cannot keep up
with the real batch size in reasonable build-agent time) — this is
exactly the kind of workload the design's own `device=="cuda"` `ceiling_s`
gate exists for, and is genuinely CUDA-bound in practice, not merely
declared so. The build instead ran a REDUCED-batch (bs=8) 1-step
forward+backward+grad-finite+optimizer.step() proxy check directly against
`NCREarlyLNModel` for K∈{26,28,30} at d=K+1 on the Mac (CPU-runnable,
PASSED — see BUILD_REPORT.md) as the CLAUDE.md smoke-test hard rule's
CPU-runnable substitute; it proves no shape/KeyError crash and a real
gradient step, but is NOT the design's own exact 500-step/real-batch pass
criterion. The three commands below are that exact criterion, to be run
on the box before this orchestrator is promoted to `queue/jobs/pending/`.

## Preconditions

1. The additive `GRID_SHAPES`/`GRIDS` patches (this build's own edits to
   `matrix-thinking/ncr/ncr_earlyln_scale.py` and
   `matrix-thinking/ncr/ncr_task.py` — see `BUILD_REPORT.md` for the
   exact diffs) must be deployed to the box's own copies of those two
   files before these commands can succeed (K∈{26,28,30} are not valid
   `--K` choices, and `nt.GRIDS[K]` raises `KeyError`, without them).
2. Run these on a GPU verified idle by `nvidia-smi` — NOT covered by
   `queue_worker.sh`'s free-GPU gate (design §4: "explicitly NOT covered
   ... a manual pre-launch check, not an automatic one").

## The three commands (verbatim, `NCR_ROOT=/home/nvidia/ncr`)

```bash
cd /home/nvidia/ncr
for K in 26 28 30; do
  D=$((K + 1))
  /home/nvidia/tdenv/bin/python3 ncr_earlyln_scale.py --cell \
    --K "$K" --d-override "$D" --seed 0 --steps 500 --ceiling-gpuh 0.05 \
    --outdir "/home/nvidia/ncr/results_kwall_smoke/K${K}" \
    --stop-file "/home/nvidia/ncr/results_kwall_smoke/K${K}/STOP"
done
```

Budget: 3 cells × 0.05h ceiling = **≤0.15 GPU-h**, outside the
orchestrator's own 15.50h ledger and outside any pool spec (design §4
KW5.10). Total disclosed program spend including these:
`≤15.50 + ≤0.15 = ≤15.65 GPU-h`.

## Exact pass criterion (design §4, "Pass criterion (exact,
build-checkable)")

For each `K∈{26,28,30}`, read
`/home/nvidia/ncr/results_kwall_smoke/K{K}/earlyln_K{K}_s0.json`:

- the subprocess exits without an uncaught exception, **AND**
- `status ∈ {"COMPLETED", "ABORTED-BUDGET"}` (either is fine — this only
  proves the config RUNS, never that it converges), **AND**
- `K == {K}`, `d == {K}+1`, `d_override == {K}+1` (the shape actually
  built is the one asked for, not a silently-defaulted `d=2K`).

`orchestrator.py`'s own `check_smoke()` function (`orchestrator.py`,
top-of-file section) implements this EXACT criterion and is what the
orchestrator itself runs at startup, before any gate check or dispatch
(design §4: "the orchestrator reads
`.../results_kwall_smoke/K{K}/earlyln_K{K}_s0.json` for each K∈{26,28,30}
and applies the Pass criterion above ... If any file is missing,
unparseable, or fails the criterion, the orchestrator REFUSES TO DISPATCH
ANYTHING and exits before touching the ledger"). The three commands above
must therefore be run, and pass, BEFORE the orchestrator job is ever
dispatched — the orchestrator's own startup gate is a defense-in-depth
re-check, not a substitute for running them.

## Verification command (after the three runs above)

```bash
cd /home/nvidia/ncr
/home/nvidia/tdenv/bin/python3 -c "
import sys
sys.path.insert(0, '/home/nvidia/ncr')
sys.path.insert(0, '/home/nvidia/ncr')  # orchestrator.py's own directory once deployed
import orchestrator as orch
smoke = orch.check_smoke('/home/nvidia/ncr/results_kwall_smoke', (26, 28, 30))
print(smoke)
assert all(v == 'PASS' for v in smoke.values()), smoke
print('ALL 3 MICRO-SMOKES PASS')
"
```
