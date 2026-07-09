# 2026-07-09 Z-dump orthogonal-complement piggyback (novel-arch waterfall, zero-GPU)

CPU-only local analysis of already-archived fast-weight state dumps. No
training, no GPU. Full finding: EXPERIMENT_LOG.md entry "Z-DUMP
ORTHOGONAL-COMPLEMENT PIGGYBACK — HARVESTED (2026-07-09)"; idea provenance:
`matrix-thinking/NOVEL_ARCH_WATERFALL.md` §1 "Cheap piggyback".

## Files

- `complement_analysis.py` — the exact analysis script run (numpy + stdlib;
  imports `matrix-thinking/chapter2/analyze_zdump.py`'s subspace machinery).
  Smoke-tested on synthetic ground truth (conformal composite + identity-
  complement + rotation-complement negative case) before the dry-run-gate
  sentinel was registered; smoke transcript in the EXPERIMENT_LOG entry's
  session, assertions embedded in the smoke are described in the script
  docstring.
- `report.txt` — full stdout of the production run (the headline table +
  correlations quoted in the log entry).
- `complement_results.json` — per-run/per-example records (all block
  energies, spectra, Procrustes fits, identity alignments, cycle-echo
  percentiles, cross-example alignments + nulls, Gaussian references).

## Inputs (read-only, SSD archive)

- `2026-07-02_task_e_zdump/task_e_40k_zdump/` — Task E K=8 d=16 (5 frN +
  3 fr7 seeds, 4 eval examples each)
- `2026-07-02_task_e_80k_kwall/` — Task E K=12 d=16 (3 seeds; the K=16=d
  runs are complement-free by construction and auto-skipped)
- `2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/` — keyanchor
  d_state=128, K in {68,76,84,92} x 3 seeds (S_T_raw / s_ideal_effective;
  two-sided decomposition, key-span != value-span)

## Reproduce

```
.venv/bin/python complement_analysis.py \
  --taske-40k <ssd>/2026-07-02_task_e_zdump/task_e_40k_zdump \
  --taske-80k <ssd>/2026-07-02_task_e_80k_kwall \
  --keyanchor <ssd>/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate \
  --out complement_results.json
```
