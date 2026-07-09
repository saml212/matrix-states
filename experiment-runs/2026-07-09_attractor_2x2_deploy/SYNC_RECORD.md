# ATTRACTOR-ROBUSTNESS 2x2 — deploy closure record (2026-07-09)

Repo commit at sync: `f09254a` (audit corrections: same-corpus noise floor 2.244355/2.216699 +
rec@0.9 NON-DECISIONAL note). Box: youthful-indigo-turkey (`brev-ukptqsu65`),
`/home/nvidia/chapter2/deltanet_rd/`.

## Manifest verification (md5, box vs repo)

Scope: every `*.py` / `*.sh` / `*.json` under `matrix-thinking/deltanet_rd/` excluding
`__pycache__/`, `logs/`, `results/` — 145 repo files.

- **Pre-sync:** 13 md5 mismatches + 11 repo files missing on box (box copy stale from the
  earlier h2h deploy).
- **Synced repo→box (24 files):** `lm_pretrain_rd.py` (2x2 flags + h2h build-fix cc89a4f),
  `h2h_fla_stub_rd.py` (gated stub), `run_attractor_robustness_2x2.py` (new),
  `h2h_cell_train_rd.py`, `h2h_calibration_wrappers_rd.py`, `h2h_box_smoke_checklist.py`,
  `probe_head_rd.py`, `phase2b_chain.sh` (repo-newer, d14fe89 trajectories_by_arm fix),
  5 pinned-threshold JSONs + `sim_cliff_power_results.json` +
  `keyanchor_cliff_niter_check_results.json` (box copies differed ONLY in `generated_at`
  timestamps + ~1e-8 float noise — equivalent regenerations, repo git-canonical), and 11
  repo-only analysis/sim scripts previously never shipped.
- **Post-sync:** full manifest diff EMPTY — all 145 repo files present on box with matching md5
  (`repo_manifest_md5.txt` vs `box_manifest_md5_post_sync.txt`).
- **Box-only artifacts (42, left untouched, recorded):** `box_only_artifacts.txt` — per-GPU
  driver scripts, box-generated result JSONs, and chain scripts from earlier waves.

## Files

- `repo_manifest_md5.txt` — repo-side md5 manifest (BSD `md5 -r` format, sorted by path)
- `box_manifest_md5_post_sync.txt` — box-side md5sum manifest after sync
- `box_only_artifacts.txt` — files present only on box (not in repo scope)
- `box_smoke_results.md` — box smoke outputs (item [18], runner --smoke, h2h smokes)
