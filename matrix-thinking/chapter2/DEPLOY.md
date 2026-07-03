# Task D — deploy + overnight run (Brev 8×H100)

Instance: **nvidia-pebble / youthful-indigo-turkey** — 8×H100 80GB, GCP
asia-southeast1-c, ~1.6k GPU-h budget (this sweep ≈ 40–60 GPU-h, <4%).
Default user on GCP images is usually `ubuntu`, home `/home/ubuntu` (adjust if the
box uses a different user). Wait until status is **Running** (not "Building").

## Granting shell access (one of these)
- **Brev CLI (preferred):** `brew install brevdev/homebrew-brev/brev` →
  `brev login` (browser) → `brev refresh` (writes `~/.ssh/config` Host alias) →
  then `ssh youthful-indigo-turkey` works directly (I can drive it from there).
- **Direct:** once the workspace is Running, share SSH and paste me the exact
  `ssh ...brevlab.com` command + key path.

Run the interactive login yourself with the `!` prefix, e.g. `! brev login`.

## 0. Ship the code (from the Mac)
```bash
cd /Users/samuellarson/Experiments/learned-representations/matrix-thinking
scp -r chapter2 youthful-indigo-turkey:/home/ubuntu/chapter2     # brev ssh-config alias
# (or: scp -P <port> -r chapter2 ubuntu@<host>:/home/ubuntu/chapter2)
```

## 0.5 Preflight — GPUs + torch + CUDA (~10 s)
```bash
ssh youthful-indigo-turkey
cd /home/ubuntu/chapter2 && bash preflight.sh
```
Expects 8 GPUs and a working torch+CUDA. The pipeline needs ONLY torch + stdlib
(no numpy/scipy). If torch is missing the script tries `pip install torch`; if
there's a conda/venv, activate it first and re-run.

## 1. Gate 0 — smoke (must pass; ~1 min)
```bash
python run_task_d.py --smoke        # watch step [2] (eigh degenerate backward) and [6] (resolution)
```
If smoke fails → STOP, do not train. Copy the error back.

## 2. One live calibration run (~2–3 min) — proves GPU path + checks convergence
```bash
python run_task_d.py --K 8 --steps 8000 --out results/calib_K8.json
# inspect results/calib_K8.json: recovered_frac@0.99 should be ~1.0 and the printed
# cosine_loss should have plateaued. If loss is still dropping at 8000, bump --steps
# in run_overnight.py's STEPS dict before launching the sweep.
```

## 3. Launch the overnight queue (survives disconnect)
```bash
cd /home/ubuntu/chapter2 && mkdir -p results/overnight
nohup python run_overnight.py --out-dir results/overnight --gpus 8 \
      > results/overnight/orchestrator.log 2>&1 &
echo "launched PID $!"
```
- Smoke-gates itself first; aborts the whole queue if smoke fails.
- 8-way GPU fan-out, ~421 runs, tier-ordered (decisive d=16 gate first).
- Per-run isolation + `--run-timeout 3600` so one hung run can't block a GPU.

## 4. Peek at progress (any time)
```bash
cat  results/overnight/PROGRESS.txt        # total/done/failed/running
tail -f results/overnight/orchestrator.log
```

## 5. On wake — the results
```bash
cat results/overnight/SUMMARY.txt          # M1 rank-vs-K per d, M3 step per (d,K)
cat results/overnight/AGGREGATE.json
```
Pull back to the repo:
```bash
scp -r youthful-indigo-turkey:/home/ubuntu/chapter2/results/overnight \
      /Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-07-01_task_d_overnight/
```

## Resume after a preemption
Re-run the exact command in step 3 — completed runs (those with a result JSON) are
skipped automatically; only the missing ones re-run.
```
