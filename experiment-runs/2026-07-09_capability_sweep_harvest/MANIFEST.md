# Capability Separation Stage 1 — 58-cell sweep HARVEST

2026-07-09. Harvest dispatch for the sweep launched per
`experiment-runs/2026-07-09_capability_sweep_launch/MANIFEST.md`
(§1.32 authorization). Completion marker
`results/SWEEP_STAGE_DONE` (box, 08:44 UTC) verified before pull; the
`cap_sweep` tmux session had self-exited cleanly (supervisor STOP
convention). Full verdict record: `matrix-thinking/
CAPABILITY_SEPARATION_DESIGN.md` §1.33 (this harvest's registry append).

## Verdict (one line)

**INCONCLUSIVE — diagnosed (D-AMB).** M1 CONFIRM (Spearman ρ=0.9747 =
the tie-capped maximum; all 19 unconstrained cells in-band per seed),
marquee S4-vs-A5 TOST **DECLARE** (diff +0.019 rank-units vs ±0.5
margin), M3 **fails CONFIRM at all 5 groups** (no HARD FALSIFY) because
the as-built target (`groups.py:157-158`, `eye(d_state)` padding) has
rank `d_state`, not `d_min` — force-rank arms trained to their exact
`sqrt(k/d_state)` rank-k ceiling (37/39 cells within 0.07, mean |Δ|
0.028) and paid a 2-rank "ambient identity tax", so no arm delivered
effective rho-rank ≥ d_min. §1.11's diagnosed-INCONCLUSIVE gate arm is
formally met; a cheap (~2.6 GPU-h) zero-padded-target M3 re-run is the
registered fix wave.

## Verify-vs-raws

- Every reported aggregate recomputed from the 58 per-cell JSONs
  (`analyze_sweep_harvest.py`, run via repo venv; output
  `harvest_analysis_output.txt`, machine summary `harvest_summary.json`).
  Decision criteria evaluated by the repo's own pre-registered
  `tost_analysis.py` functions, not re-implemented.
- Pull integrity: 61/61 files md5-match box (`md5_local.txt` vs
  `md5_box.txt`, zero mismatches).
- Inventory: 58/58 cells present (53 new + 5 resume-skipped calibration
  cells, matching the launch manifest), every cell at its exact Rev-7
  pinned step budget, `n_skipped_steps=0` everywhere, no
  aborts/escalations/tracebacks in `cap_sweep.log` (118 KB, full log
  archived here).
- Budget: **2.5907 GPU-h all-58** (2.3867 this launch's 53 new cells +
  0.2039 prior calibration) vs the code gate's 4.62 GPU-h projection
  (44% under) and the design's group-weighted ≈2.51 GPU-h (within 3%).
  Wall-clock cross-check: 06:21:25→08:44:45 UTC = 2.389 h on GPU 0,
  matches the 53-cell sum exactly.

## M2 build-gap disclosure

The sweep persisted no checkpoints and never invoked
`truncation_curve.py` — the pre-registered per-seed M2 knee criterion is
NOT computable from the sweep as-run. Proxy (disclosed, n=1/group):
truncation curves on the md5-verified round-7 pinned-budget diagnosis
checkpoints (`m2_proxy_ckpts/`, `m2_proxy_truncation.py`, outputs
`m2_proxy_output.txt` / `m2_proxy_truncation_curves.json`): knees at
k\*=d_state (S3:4, S4:5, A5:5, A6:7; S5 degenerate) — outside the
[d_min−1, d_min+1] CONFIRM band, corroborating D-AMB, not the pure
d_min knee.

## Files

- `results/` — all 58 per-cell JSONs + `calibration_report.json` (box copies)
- `cap_sweep.log`, `box_cap_sweep_supervisor.sh` — sweep log + supervisor as-run
- `analyze_sweep_harvest.py` → `harvest_analysis_output.txt`, `harvest_summary.json`
- `m2_proxy_ckpts/` (5 × ~190 KB), `m2_proxy_truncation.py` → `m2_proxy_output.txt`, `m2_proxy_truncation_curves.json`
- `md5_local.txt`, `md5_box.txt` — pull-integrity manifests

SSD mirror: `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-09_capability_sweep_harvest/`.
