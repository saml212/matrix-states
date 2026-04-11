# Experiment Runs

This folder holds artifacts from completed experiments: scripts, summaries, results JSON, and small log excerpts. Large files (full training logs, checkpoints, raw data) live on the SSD at `/Volumes/1TB_SSD/learned-representations/`.

## What's here

- **8xh100-session1/** — 26 experiments run on 8×H100 pods between March and April 2026. Each experiment has a `*_script.py` (the exact script that ran), a `*_SUMMARY.txt` (human-readable summary), a `*_results.json` (structured metrics), and a `*_train.log` (truncated to fit in repo).
- **h100-run1-matrix-thinker-2.46M/** — The first H100 experiment (Run 8 in EXPERIMENT_LOG.md). Has the model code (`matrix_thinker.py`) and training script (`run_train.py`).

## Convention for future runs

For each new experiment, save to this folder:

- `script.py` — exact script that ran (committed to repo)
- `SUMMARY.txt` — human-readable summary (committed to repo)
- `results.json` — structured metrics (committed to repo)
- `rank_dynamics.json` or other small analysis files (committed to repo)
- `analysis_plots/*.png` — small plots (committed to repo)

The following live on the SSD only, not in this folder:

- `train.log` — full training log
- `*.pt` — model checkpoints
- raw activation dumps, large evaluation outputs

The repo's `.gitignore` excludes `*.log` from this folder by default. If you need a log in the repo for grant applications or paper writing, copy a truncated excerpt with a different name (e.g., `train_excerpt.log`).

## Where to find more

- Strategic context for each experiment: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md)
- Project state: [STATE.md](../STATE.md)
- Future experiment queue: [matrix-thinking/QUEUE.md](../matrix-thinking/QUEUE.md)
