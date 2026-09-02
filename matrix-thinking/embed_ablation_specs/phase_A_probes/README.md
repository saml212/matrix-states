# Phase A -- rate/admission probes (0640-0645)

Audit F2 (FATAL) gating fix. These 6 cells have NO dependency on phase B
and may be queued/run as soon as the PREREQ in each spec's `notes` field
is met (embed_ablation_rd.py scp'd to the box, wikitext-mix-ext corpus
confirmed present).

Each probe trains 500 steps and writes its result JSON to
`/home/nvidia/embed_ablation/results/probes/` (a SEPARATE subdirectory
from the phase-B seeded cells' `results/`, per audit F1 -- this keeps
`harvest()`'s plain top-level `os.listdir(results_dir)` from ever seeing
probe files at all, in addition to the explicit `role`/filename filter
`_is_probe()` already applies).

**After all 6 land, before ANY phase_B_seeded/ spec is queued:**

```
/home/nvidia/tdenv/bin/python3 embed_ablation_rd.py --check-admission \
    --probe-results-dir /home/nvidia/embed_ablation/results/probes \
    --intended-steps 2000 --intended-batch 64
```

This must exit 0. It checks, per probe: (a) the last three T=1 evals are
monotone non-increasing (a basic "is this actually learning" check), and
(b) the measured rate extrapolated to 2000 steps/batch=64 does not exceed
2.0 GPU-h/cell (audit M4). If it exits nonzero, STOP -- re-derive --steps
for every phase_B_seeded/ spec's `cmd` (or investigate the failing arm)
before staging any of them. See EMBEDDING_ABLATION_DESIGN.md S6 for the
identical sentence in the pre-registration itself.
