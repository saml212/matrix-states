#!/usr/bin/env bash
# Task D preflight — run FIRST on the Brev 8xH100 box, before smoke/overnight.
# Verifies GPUs, python, torch+CUDA, and disk. Exits nonzero if anything critical
# is wrong. The Task D pipeline needs ONLY torch + stdlib (no numpy/scipy/etc).
set -uo pipefail

fail() { echo "PREFLIGHT FAILED: $1"; exit 1; }

echo "=== GPUs (nvidia-smi) ==="
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found (GPUs not visible)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || fail "nvidia-smi errored"
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "GPU count: $N"
[ "${N:-0}" -ge 1 ] || fail "no GPUs detected"
[ "${N:-0}" -ge 8 ] || echo "NOTE: expected 8 GPUs, saw $N — overnight --gpus should match."

echo ""
echo "=== python ==="
PY=$(command -v python3 || command -v python) || fail "no python found"
echo "python: $PY ($($PY --version 2>&1))"

echo ""
echo "=== torch / CUDA ==="
if $PY -c "import torch" 2>/dev/null; then
  $PY - <<'EOF' || exit 1
import torch, sys
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    sys.exit("CUDA not available to torch")
EOF
else
  echo "torch NOT importable in $PY."
  echo "If there is a conda/venv with torch, activate it and re-run this script."
  echo "Otherwise attempting: $PY -m pip install torch  (default CUDA build)"
  $PY -m pip install --quiet torch || fail "torch pip install failed"
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || fail "torch still not working after install"
fi

echo ""
echo "=== disk (working dir) ==="
df -h . | tail -1

echo ""
echo "PREFLIGHT OK — safe to run: python run_task_d.py --smoke"
