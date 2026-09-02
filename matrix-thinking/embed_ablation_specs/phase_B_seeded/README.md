# Phase B -- seeded cells (0646-0669)

Audit F2 (FATAL) gating fix. **DO NOT QUEUE ANY FILE IN THIS DIRECTORY**
until phase_A_probes/'s 6 probes have completed AND
`embed_ablation_rd.py --check-admission --probe-results-dir
/home/nvidia/embed_ablation/results/probes --intended-steps 2000
--intended-batch 64` has exited 0 (see phase_A_probes/README.md).

If the admission check fails (non-monotone T1 in any arm, or extrapolated
GPU-h/cell > 2.0), every `cmd` below needs its `--steps` (and this
generator's `gpuh()` cost model) re-derived from the ACTUAL measured
probe rate before anything here is trusted or queued -- see
EMBEDDING_ABLATION_DESIGN.md S6's identical gating sentence.

24 cells: 4 arm-configs (matrix, flatp, flatd, flatten) x 2 sizes (S, M)
x 3 seeds. matrix/flatp/flatten feed the pre-registered decisions
(STRENGTHEN-01: matrix vs flatten; STRENGTHEN-04: matrix vs flatp);
flatd is a disclosed, non-gating params-UNMATCHED control.
