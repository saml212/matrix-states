#!/usr/bin/env bash
# NCR ORTHO-WRITE v2 re-launch driver -- ONE dedicated (drained) GPU.
# Runs this GPU's ASSIGNED subset of the full 24-cell wave, sequentially
# (one cell at a time -- shared SMs, never stack). Resume-safe: each cell's
# python skips-if-COMPLETED (whole-cell), so a restart loses nothing.
# Emits NO verdict (a separate blind agent assesses). Ceiling raised 3.0->6.0h
# per the § CEILING AMENDMENT (commit 62a6fb6). Science FROZEN: 320K steps,
# h*=40, NS-polar (ns-iter 40, ns-power 12, anneal 0.5), R=4. run commit 3086dfa.
#
# Part A primary (16): arm{free,ortho} x K{24,32} x seed{0..3}
# Part B disc   (8): arm{free,ortho} x K32 x seed{0..3} R=4
# Distributed across GPUs 4,5,7 (balanced ~26h each). Usage: THIS 1 GPU id.
set -u
GPU="${1:?usage: run_ncr_ortho_v2.sh GPU_ID}"
cd /home/nvidia/ncr
OUT=/home/nvidia/ncr/results_ortho_write
LOG="$OUT/run_g${GPU}.log"
PY=/home/nvidia/tdenv/bin/python3
SCRIPT=ncr_ortho_write.py
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DRY_RUN_BYPASS=1
export CUDA_VISIBLE_DEVICES="$GPU"
CEIL=6.0
# External per-cell backstop ABOVE the 6.0h internal ceiling (so the internal
# ceiling fires first with a graceful status write; timeout only catches a hang).
CELL_TO="${NCR_CELL_TIMEOUT:-23400}"   # 6.5h

PRIM () {  # arm K seed
  echo "--- PRIMARY [$1 K$2 s$3] START $(date -u +%FT%TZ) g$GPU ---" >> "$LOG"
  timeout "$CELL_TO" "$PY" "$SCRIPT" --primary-cell --arm "$1" --K "$2" --seed "$3" \
    --steps 320000 --ns-iter 40 --ns-power 12 --anneal-frac 0.5 \
    --ceiling-gpuh "$CEIL" --outdir "$OUT" >> "$LOG" 2>&1
  echo "--- PRIMARY [$1 K$2 s$3] END rc=$? $(date -u +%FT%TZ) g$GPU ---" >> "$LOG"
}
DISC () {  # arm seed
  echo "--- DISC [$1 K32 s$2 R4] START $(date -u +%FT%TZ) g$GPU ---" >> "$LOG"
  timeout "$CELL_TO" "$PY" "$SCRIPT" --disc-cell --arm "$1" --K 32 --seed "$2" \
    --R 4 --steps 320000 --ceiling-gpuh "$CEIL" --outdir "$OUT" >> "$LOG" 2>&1
  echo "--- DISC [$1 K32 s$2 R4] END rc=$? $(date -u +%FT%TZ) g$GPU ---" >> "$LOG"
}

echo "=== v2 RUN START g$GPU $(date -u +%FT%TZ) commit=3086dfa amend=62a6fb6 ceiling=$CEIL ===" >> "$LOG"

case "$GPU" in
  4)  # 6 primary (mixed arms) + 2 disc
    PRIM free 24 0;  PRIM free 24 1;  PRIM ortho 24 0; PRIM ortho 24 1
    PRIM free 32 0;  PRIM ortho 32 0
    DISC free 0;     DISC ortho 0
    ;;
  5)  # 5 primary (mixed arms) + 3 disc
    PRIM free 24 2;  PRIM ortho 24 2; PRIM free 32 1;  PRIM ortho 32 1; PRIM free 32 2
    DISC free 1;     DISC ortho 1;    DISC free 2
    ;;
  7)  # 5 primary (mixed arms) + 3 disc
    PRIM free 24 3;  PRIM ortho 24 3; PRIM ortho 32 2; PRIM ortho 32 3; PRIM free 32 3
    DISC ortho 2;    DISC free 3;     DISC ortho 3
    ;;
  *)
    echo "ERROR: no cell assignment for GPU $GPU -- refusing" >> "$LOG"; exit 2 ;;
esac

echo "=== v2 ALL-DISPATCHED g$GPU $(date -u +%FT%TZ) ===" >> "$LOG"
