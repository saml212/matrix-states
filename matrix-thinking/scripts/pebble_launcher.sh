#!/bin/bash
# Idempotent end-to-end launcher: bootstrap + dataset-gen + experiment-loop.
# Designed to be re-runnable after preemption. Each stage checks for completion
# markers and skips already-done work. Runs experiments back-to-back to keep
# the GPU at 100% utilization.

set -u
export TOKENIZERS_PARALLELISM=false
# Container-local HF cache (NOT on /workspace MFS volume — network reads to MFS
# have intermittent OSError[Errno 6] under load that crashes mid-training).
# Container disk wipes on preemption but launcher re-downloads cheaply.
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/.cache/huggingface

LOG_DIR=/workspace/pebble/logs
RES_DIR=/workspace/pebble/results
mkdir -p "$LOG_DIR" "$RES_DIR" /workspace/pebble/round3_gamma0/data /workspace/pebble/round4_vanilla_sft

ML=/workspace/pebble
log()  { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_DIR/launcher.log"; }
done_marker() { test -f "$ML/.markers/$1"; }
mark()        { mkdir -p "$ML/.markers" && touch "$ML/.markers/$1"; }

log "===== launcher start ====="
log "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Stage 1: env restore
if ! done_marker stage1_env || ! python3 -c "import transformers, torch" 2>/dev/null; then
  log "[stage1] installing deps + ensuring local HF cache (no MFS symlink)"
  # Force /root/.cache/huggingface to be a real local directory, NOT a symlink
  # to /workspace/pebble/hf_cache (the MFS network volume that has intermittent
  # OSError[Errno 6] under load and crashes mid-training).
  if [ -L /root/.cache/huggingface ]; then rm /root/.cache/huggingface; fi
  mkdir -p /root/.cache/huggingface
  pip install --break-system-packages --quiet transformers==4.44.2 datasets==2.21.0 tokenizers huggingface_hub 2>&1 | tail -2
  mark stage1_env
fi
log "[stage1] env ready: $(python3 -c "import torch,transformers; print(f'torch={torch.__version__} cuda={torch.version.cuda} hf={transformers.__version__}')")"

# Stage 1b: pre-download GPT-2 to container-local cache (avoids MFS hot-path on every run)
if ! python3 -c "from transformers import GPT2TokenizerFast, GPT2LMHeadModel; t=GPT2TokenizerFast.from_pretrained('gpt2'); m=GPT2LMHeadModel.from_pretrained('gpt2'); print('ok', len(t))" 2>/dev/null | grep -q ok; then
  log "[stage1b] pre-fetching gpt2 to /root/.cache/huggingface"
  python3 -c "from transformers import GPT2TokenizerFast, GPT2LMHeadModel; GPT2TokenizerFast.from_pretrained('gpt2'); GPT2LMHeadModel.from_pretrained('gpt2')" 2>&1 | tail -3
fi

# Stage 2: ProsQA source data
if ! done_marker stage2_data ; then
  log "[stage2] cloning Coconut for ProsQA"
  cd "$ML" && [ -d coconut ] || git clone --depth 1 https://github.com/facebookresearch/coconut.git 2>&1 | tail -1
  ln -sf "$ML/coconut/data/prosqa_train.json" "$ML/round3_gamma0/data/prosqa_train.json"
  ln -sf "$ML/coconut/data/prosqa_test.json"  "$ML/round3_gamma0/data/prosqa_test.json"
  mark stage2_data
fi
log "[stage2] data symlinks: $(ls $ML/round3_gamma0/data/)"

# Stage 3: ProsQA-MULTI-2 generation
if ! done_marker stage3_multi2 || [ ! -s "$ML/round3_gamma0/data/prosqa_multi2_train.json" ]; then
  log "[stage3] generating ProsQA-MULTI-2"
  python3 "$ML/scripts/build_prosqa_multi.py" \
    --k 2 \
    --src-train "$ML/coconut/data/prosqa_train.json" \
    --src-test  "$ML/coconut/data/prosqa_test.json" \
    --train-out "$ML/round3_gamma0/data/prosqa_multi2_train.json" \
    --test-out  "$ML/round3_gamma0/data/prosqa_multi2_test.json" \
    --n-train 5000 --n-test 500 --seed 1337 \
    2>&1 | tail -10 | tee -a "$LOG_DIR/multi2_gen.log"
  mark stage3_multi2
fi
log "[stage3] MULTI-2 sizes: $(ls -lh $ML/round3_gamma0/data/prosqa_multi2_*.json | awk '{print $5,$9}')"

# Common training args. RTX 4090 24GB → batch=8 conservative
COMMON="--mode train_matrix --dataset prosqa --gamma 0 --readout flatten --batch-size 16 --epochs 25 --warmup-steps 50 --max-eval-batches 8"

run_exp() {
  local name="$1"
  local extra="$2"
  local seed="$3"
  local results_dir="$RES_DIR/$name/seed${seed}"
  if [ -f "$results_dir/SUMMARY.txt" ]; then
    log "[exp] $name seed=$seed already done — skipping"
    return 0
  fi
  log "[exp] $name seed=$seed starting"
  mkdir -p "$results_dir"
  python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ML/scripts/run_matrix_codi.py" $COMMON --seed "$seed" $extra \
    --results-dir "$results_dir" \
    2>&1 | tee "$LOG_DIR/${name}_seed${seed}.log" | tail -20
  log "[exp] $name seed=$seed completed (rc=${PIPESTATUS[0]})"
}

# Experiment plan — runs sequentially, idempotent. Pre-validation gate first.
# Each experiment writes its SUMMARY.txt; re-runs are skipped.

# ===== EXPERIMENT 1: Pre-validation Round 3 γ=0 baseline on MULTI-2 =====
log "===== Experiment #1: matrix-CODI baseline on MULTI-2 ====="
for SEED in 1337 42 7; do
  run_exp "multi2_baseline_gamma0" \
    "--prosqa-train-path $ML/round3_gamma0/data/prosqa_multi2_train.json --prosqa-val-path $ML/round3_gamma0/data/prosqa_multi2_test.json" \
    "$SEED"
done
mark exp1_done

# ===== EXPERIMENT 2: Pre-validation rank-1-forced matrix-CODI on MULTI-2 =====
log "===== Experiment #2: matrix-CODI force-rank=1 on MULTI-2 ====="
for SEED in 1337 42 7; do
  run_exp "multi2_force_rank1_gamma0" \
    "--prosqa-train-path $ML/round3_gamma0/data/prosqa_multi2_train.json --prosqa-val-path $ML/round3_gamma0/data/prosqa_multi2_test.json --force-rank-during-training 1" \
    "$SEED"
done
mark exp2_done

# ===== EXPERIMENT 3: matrix-CODI baseline on ProsQA-1 (Round 3 γ=0 reproduction) =====
log "===== Experiment #3: matrix-CODI baseline on ProsQA-1 (Round 3 reproduction) ====="
for SEED in 1337 42 7; do
  run_exp "prosqa1_baseline_gamma0" \
    "--prosqa-train-path $ML/round3_gamma0/data/prosqa_train.json --prosqa-val-path $ML/round3_gamma0/data/prosqa_test.json" \
    "$SEED"
done
mark exp3_done

# ===== EXPERIMENT 4: HEADLINE — MULTI-2 + entropy reward (λ=0.1) =====
log "===== Experiment #4 HEADLINE: matrix-CODI + entropy reward on MULTI-2 ====="
for SEED in 1337 42 7; do
  run_exp "multi2_entropy_l0.1_gamma0" \
    "--prosqa-train-path $ML/round3_gamma0/data/prosqa_multi2_train.json --prosqa-val-path $ML/round3_gamma0/data/prosqa_multi2_test.json --rank-loss entropy --rank-lambda 0.1" \
    "$SEED"
done
mark exp4_done

# ===== EXPERIMENT 5: NULL CONTROL — ProsQA-1 + entropy reward =====
log "===== Experiment #5 NULL: matrix-CODI + entropy reward on ProsQA-1 ====="
for SEED in 1337 42 7; do
  run_exp "prosqa1_entropy_l0.1_gamma0" \
    "--prosqa-train-path $ML/round3_gamma0/data/prosqa_train.json --prosqa-val-path $ML/round3_gamma0/data/prosqa_test.json --rank-loss entropy --rank-lambda 0.1" \
    "$SEED"
done
mark exp5_done

# ===== EXPERIMENT 6: λ sweep on MULTI-2 — additional λ values =====
log "===== Experiment #6: λ sweep — MULTI-2 + entropy at λ=0.01 and λ=1.0 ====="
for LAMBDA in 0.01 1.0; do
  for SEED in 1337 42 7; do
    run_exp "multi2_entropy_l${LAMBDA}_gamma0" \
      "--prosqa-train-path $ML/round3_gamma0/data/prosqa_multi2_train.json --prosqa-val-path $ML/round3_gamma0/data/prosqa_multi2_test.json --rank-loss entropy --rank-lambda $LAMBDA" \
      "$SEED"
  done
done
mark exp6_done

# ===== EXPERIMENT 7: wd=0 implicit-bias sweep on Round 3 γ=0 =====
log "===== Experiment #7: wd=0 implicit-bias sweep ====="
for SEED in 1337 42 7; do
  run_exp "prosqa1_wd0_gamma0" \
    "--prosqa-train-path $ML/round3_gamma0/data/prosqa_train.json --prosqa-val-path $ML/round3_gamma0/data/prosqa_test.json --weight-decay 0.0" \
    "$SEED"
done
mark exp7_done

# ===== EXPERIMENT 8: nuclear-norm comparison (mechanism baseline) =====
log "===== Experiment #8: matrix-CODI + nuclear-norm reward on MULTI-2 ====="
for SEED in 1337 42 7; do
  run_exp "multi2_nuclear_l0.1_gamma0" \
    "--prosqa-train-path $ML/round3_gamma0/data/prosqa_multi2_train.json --prosqa-val-path $ML/round3_gamma0/data/prosqa_multi2_test.json --rank-loss nuclear --rank-lambda 0.1" \
    "$SEED"
done
mark exp8_done

log "===== ALL EXPERIMENTS COMPLETE ====="
touch "$ML/queue_complete.flag"
