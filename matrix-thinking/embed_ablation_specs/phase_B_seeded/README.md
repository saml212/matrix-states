# Phase B -- seeded cells (0646-0669)

Audit F2 (FATAL) gating fix. **DO NOT QUEUE ANY FILE IN THIS DIRECTORY**
until phase_A_probes/'s 6 probes have completed AND
`embed_ablation_rd.py --check-admission --probe-results-dir
/home/nvidia/embed_ablation/results/probes --intended-steps 2000
--intended-batch 64` has exited 0 (see phase_A_probes/README.md, including
its MJ-2 6-probe admission-SET requirement -- a single missing/crashed
probe fails this on its own).

If the admission check fails (missing probe, non-monotone or
non-improving T1 in any arm, or extrapolated GPU-h/cell > 2.0), **--steps
and every validity_check in this directory move together** (audit
round-2 MJ-5): re-run `_gen_specs.py` with an updated `STEPS_B` constant
(and re-derive `gpuh()`'s cost model from the ACTUAL measured probe rate)
rather than hand-editing individual spec files -- see
EMBEDDING_ABLATION_DESIGN.md S6's identical gating sentence.

24 cells: 4 arm-configs (matrix, flatp, flatd, flatten) x 2 sizes (S, M)
x 3 seeds. matrix/flatp/flatten feed the pre-registered decisions
(STRENGTHEN-01: matrix vs flatten; STRENGTHEN-04: matrix vs flatp);
flatd is a disclosed, non-gating params-UNMATCHED control. Every spec's
`validity_check` requires `steps_completed == 2000` EXACTLY (audit
round-2 MJ-6, no "-1" slack) -- `harvest()` itself independently re-checks
this (MJ-5's `steps_completed==steps_target` filter), so a cell that fails
this validity_check would also be silently excluded from every decision;
catching it here, at queue-completion time, is strictly cheaper.
