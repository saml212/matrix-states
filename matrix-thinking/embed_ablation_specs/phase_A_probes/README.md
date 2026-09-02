# Phase A -- rate/admission probes (0640-0645)

Audit F2 (FATAL) gating fix. These 6 cells have NO dependency on phase B
and may be queued/run as soon as the PREREQ in each spec's `notes` field
is met (embed_ablation_rd.py scp'd to the box, wikitext-mix-ext corpus
confirmed present).

Each probe trains 500 steps with `--eval-interval 100`
(audit round-2 MJ-1: 500/100 = 5 eval points, so
`--check-admission`'s monotonicity/improvement checks have enough of a
curve to judge, not a 1-2-point stub) and writes its result JSON to
`/home/nvidia/embed_ablation/results/probes/` (a SEPARATE subdirectory
from the phase-B seeded cells' `results/`, per audit F1 -- this keeps
`harvest()`'s plain top-level `os.listdir(results_dir)` from ever seeing
probe files at all, in addition to the explicit `role`/filename filter
`_is_probe()` already applies).

**All 6 probes must exist and be complete -- audit round-2 MJ-2:**
`--check-admission` asserts the SET of `(arm,size)` found among the probe
records' own fields equals exactly `{matrix,flat,flatten}x{S,M}` (6
combos). A probe that crashed and never wrote a file (or wrote one
missing its `arm`/`size` field) makes this set incomplete and FAILS
admission outright -- independent of whether the other 5 probes look
fine. Do not proceed on "5 of 6 looked good."

**After all 6 land, before ANY phase_B_seeded/ spec is queued:**

```
/home/nvidia/tdenv/bin/python3 embed_ablation_rd.py --check-admission \
    --probe-results-dir /home/nvidia/embed_ablation/results/probes \
    --intended-steps 2000 --intended-batch 64
```

This must exit 0. It checks, per probe: (0) the 6-probe admission SET
above; (a) the last three T=1 evals are monotone non-increasing, requiring
>=3 eval points to even judge this (audit round-2 MJ-1 -- a probe with
fewer than 3 points FAILS this check, it is not silently skipped); (a2)
the LAST T1 value is strictly less than the FIRST (net improvement over
the whole probe run, not just local non-increase); and (b) the measured
rate extrapolated to 2000 steps/batch=64 does not exceed 2.0 GPU-h/
cell (audit M4). If it exits nonzero, STOP -- re-derive `--steps` (audit
round-2 MJ-5: `--steps` and this generator's `STEPS_B` constant and every
phase_B_seeded/ spec's exact-match `validity_check` all move together --
they are generated from the SAME constant, never hand-edit one without
regenerating the others) before staging any of phase_B_seeded/. See
EMBEDDING_ABLATION_DESIGN.md S6 for the identical sentence in the
pre-registration itself.
