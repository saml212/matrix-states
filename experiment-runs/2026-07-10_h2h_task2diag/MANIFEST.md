# 2026-07-10 — h2h TASK2 DIAGNOSIS ROUND + transformer_K48 stress cell

Registry: `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` **§1.42** (pre-run
record: DialExhausted tiebreak + ambiguity resolutions A1-A6, committed
`5cfdd80` BEFORE launch) and **§1.43** (the verdict of record).

**VERDICT:** task2 = TRAINABILITY/SEED-VARIANCE CONFIRMED at pooled rate 3/9
(new seeds s5=0.4795, s7=0.3909 clear the frozen 0.09375 bar; 0/9 ablation
seeds ever leave chance) — hard-capability-boundary REJECTED at this
scale/budget; all 3/3 bar-clearing seeds (s2/s5/s7) collapse under horizon
extension AND held-out hops. K48 3-arm locate-only table complete:
contender 0.0189 / ablation 0.0195 / transformer 0.0218 (fresh, this round)
— all ≈chance 0.0208, no arm demonstrates recall at K/d=0.75 quarter-budget.
Realized ≈4.07 GPU-h vs 8.0 ceiling, GPU 7 only.

## Contents

- `h2h_task2diag_rd.py`, `h2h_task2diag_stage.sh` — the EXACT deployed
  scripts (md5 `ff06446c06b3c7bb91cd2746a7973120` /
  `8cef3e75379e02f9de13666515bf601f`, local==box verified at deploy).
- `results/` — 12 training JSONs (`h2h_{contender,ablation}_task2_sweep_s{3..8}.json`),
  `results/remetric/` 12 verdict-grade re-metric JSONs (audited
  `run_cell_round4`, md5 provenance pinned),
  `transformer_task1_stress_K48_round4.json` (fresh K48 cell),
  `TASK2DIAG_VERDICT.json` (pooled analysis: A3 adjudication, n=9 paired CI,
  §16.19.5 batch-effect gates), `contender_task2_s5s7_horizon_refs.json`
  (§1.42 A5 conditional reads) + `ckpt_map_horizon_s5s7.json`,
  `TASK2DIAG_STOP`.
- `logs/` — supervisor + token-probe + blind-check + 12 per-cell + K48 +
  remetric + analyze logs.
- `md5_manifest.txt` — box-side md5s; `md5_manifest_local.txt` — local-side;
  diff verified 47/47 MATCH at harvest.

## Run provenance

- Box: youthful-indigo-turkey, GPU 7 only, tmux `h2h_task2diag` with the
  self-healing supervisor loop; launched 2026-07-10 17:55:54Z; completed
  autonomously (a local session outage mid-round killed only the local
  monitor — the box run was unaffected, TASK2DIAG_STOP written).
- Checkpoints (BOX/SSD-only, >25MB each, per the hybrid archive policy):
  `/data/h2h_rung1_ckpts/h2h_{arch}_task2_sweep_s{3..8}_r4.pt` and
  `h2h_calib_transformer_task1_calib_stress_locate_only_K48_auxrev2_r4.pt`.
- Gates honored: gate-5 tokens, MARGINS_FROZEN blind check (negative test
  exercised at launch), code-level `require_margins_frozen` per cell,
  `H2H_DIAL_ROUND=4` (filename versioning only; every training cell
  role="sweep" → dial structurally never evaluated, AUD2-F4).
- Independent pre-launch audit: CLEAR-TO-RUN (0 blocking/major); box smoke:
  K48 tiny cell 38 s PASS; runner selftests 5/5 local + box.
