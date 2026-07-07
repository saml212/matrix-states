#!/bin/bash
# REASONING_LINK_DESIGN.md Leg-B rung-3 (1.31B) rows -- registered queue item,
# ~0.02 GPU-h. Fills the ONE cell reasoning_link_chain.sh's own Steps 8-9
# deferred (rung 3 was gated on `results/lm_rd_trackc/wave3/ALL_DONE`, which
# never landed -- trackc wave3 self-terminated at step 155081/183105 =
# 84.69% ~= 84.7% of its token-matched budget, `timed_out: true` in
# /tmp/wave3_training_summaries.json, val-loss trajectory already flattening
# by the final checkpoints -- PLATEAU-NEUTRALIZED: stopping early does not
# compromise this rung's representativeness for the probe, per that
# trajectory). This script bypasses the ALL_DONE gate deliberately, pointing
# `--ckpt` directly at the two step-155000 checkpoints (the run's own final
# saved step) rather than trusting leg_b_ckpt_path_final's glob (which the
# full chain reserves for genuinely-completed rungs only).
#
# Extends the same committed Leg-B grid cell shape rungs 0-2 already ran
# (results/reasoning_link/leg_b_rung{0,1,2}_*_k{32,64}.json, all present on
# this box): rung 3's own committed near-cliff K=64 (LEG_B_RUNG_CFG /
# K_SWEEP, d_state=128), hops 1-4, surgery=native (Leg B has no frozen-bias
# arms to force off), 2 corpora x 1 seed (sec 6.1's PINNED rung-3
# configuration -- seed 0 only, unlike rungs 0-2's 3 seeds).
#
# batch-size=4: NOT the probe's BATCH_SIZE_DEFAULT (16) -- LAUNCH FIX 5/7 in
# reasoning_link_chain.sh (fla 0.5.1's layer_norm_fwd_kernel1 int32 pointer
# overflow at d_model=2560, batch>=8) and Stage 0.5's own real cost
# calibration on this box (results/reasoning_link/stage05_rung3_cost_calibration.json:
# used_batch_size=4, oom_fallback_used=false, ratio_to_baseline=0.042,
# action="OK: within budget, proceed") -- both independently confirm batch=4
# is the safe, already-validated choice for this exact shape.
#
# Checkpoint config: `load_checkpoint` -> `DeltaNetLM(**ckpt["config"])`
# (reasoning_link_probe.py) reads d_model/d_state/n_layers/conv_size etc.
# from the CHECKPOINT'S OWN saved config dict, never from LEG_B_RUNG_CFG or
# any other hardcoded table -- LEG_B_RUNG_CFG is used only by the
# leg_b_ckpt_path_* PATH RESOLVERS, which this script bypasses entirely via
# --ckpt. Verified by direct code read (reasoning_link_probe.py:705-713,
# :1503-1504) before this script was written -- no rung-1/rung-0 dims leak
# into a rung-3 pass.
#
# GPU 0 ONLY -- GPUs 2-7 are the running `keyanchor_scaling_wide` tmux
# session (DO NOT TOUCH). GPU 1 is idle but unassigned; GPU 0 pinned per the
# registered task.
#
# Same dependency-closure list as reasoning_link_chain.sh (unchanged --
# this script imports nothing new); same `set -euo pipefail` discipline,
# same Stage -1 self-test gate as every sibling chain in this dir.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/reasoning_link

RUNG3_GPU=0

TRACKC_CKPT_ROOT=/data/lm_rd_trackc_ckpts
CKPT_OPENR1="$TRACKC_CKPT_ROOT/wave3/lmC_openr1-mix-ext_dm2560_ds128_L22_s0_step155000.pt"
CKPT_WIKITEXT="$TRACKC_CKPT_ROOT/wave3/lmC_wikitext-mix-ext_dm2560_ds128_L22_s0_step155000.pt"

# 84.7%-budget disclosure (mandatory, per task registration) -- computed once
# here, injected into every output JSON's own "harvest_metadata" key below,
# never silently dropped.
BUDGET_STEPS_COMPLETED=155081
BUDGET_STEPS_TARGET=183105
BUDGET_FRAC=0.8469
BUDGET_NOTE="rung-3 (wave3, 1.31B) self-terminated at step 155081/183105 = 84.69% (~84.7%) of its token-matched training budget (timed_out=true, /tmp/wave3_training_summaries.json on box); harvested at its final saved checkpoint, step 155000; val-loss trajectory already flattening by the final checkpoints -- PLATEAU-NEUTRALIZED (early stop does not compromise this rung's representativeness for the reasoning_link probe)."

# ---------------------------------------------------------------------------
# Step 1 -- Stage -1 self-tests (gate). Pure CPU, identical convention to
# every sibling chain in this dir (reasoning_link_chain.sh step 1,
# reasoning_link_phase1b_stage0_chain.sh step 1).
# ---------------------------------------------------------------------------
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY reasoning_link_probe.py --mode selftest \
    2>&1 | tee logs/97_reasoning_link_rung3_stage_minus1.log

# ---------------------------------------------------------------------------
# Step 2 -- the two rung-3 Leg-B cells (openr1-mix-ext, wikitext-mix-ext),
# K=64, hops 1-4, surgery=native, batch=4, --ckpt override (bypasses the
# ALL_DONE-gated leg_b_ckpt_path_final resolver). --family leg_b --rung 3
# --corpus <corpus> --ckpt-seed 0 are passed ALONGSIDE --ckpt so sec 4.6's
# episode-seed formula (condition_idx=rung=3, corpus_idx from --corpus,
# ckpt_seed_idx=0) still fires correctly downstream in main() -- --ckpt only
# overrides the checkpoint PATH resolution, never the seed-formula wiring
# (verified: `ckpt_path = args.ckpt` short-circuits path resolution only;
# condition_idx/corpus_idx computation in main() reads args.family/args.rung/
# args.corpus unconditionally, reasoning_link_probe.py:1851-1861).
# ---------------------------------------------------------------------------
declare -A CORPUS_CKPT=(
  [openr1-mix-ext]="$CKPT_OPENR1"
  [wikitext-mix-ext]="$CKPT_WIKITEXT"
)

for corpus in openr1-mix-ext wikitext-mix-ext; do
  ckpt="${CORPUS_CKPT[$corpus]}"
  out="results/reasoning_link/leg_b_rung3_${corpus}_s0_k64.json"
  log="logs/97_leg_b_rung3_${corpus}.log"
  # Resume-safety (house [LEARN]: skip already-completed work by checking
  # output validity, not just existence).
  if [ -s "$out" ] && $PY -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if d.get('forward_counts') else 1)" 2>/dev/null; then
    echo "SKIP (already complete): $out"
    continue
  fi
  echo "=== rung3 cell: corpus=$corpus ckpt=$ckpt ==="
  CUDA_VISIBLE_DEVICES=$RUNG3_GPU $PY reasoning_link_probe.py \
      --mode cell --ckpt "$ckpt" --family leg_b --rung 3 --corpus "$corpus" --ckpt-seed 0 \
      --k 64 --hops 1,2,3,4 --surgery native --batch-size 4 --device cuda --out "$out" \
      2>&1 | tee "$log"
done

# ---------------------------------------------------------------------------
# Step 3 -- inject the mandatory 84.7%-budget disclosure into each cell's own
# output JSON (additive-only "harvest_metadata" key -- never touches any
# instrument-computed field). Real Python (heredoc), same convention as
# reasoning_link_chain.sh's own Stage-0 abort-check step.
# ---------------------------------------------------------------------------
BUDGET_STEPS_COMPLETED="$BUDGET_STEPS_COMPLETED" BUDGET_STEPS_TARGET="$BUDGET_STEPS_TARGET" \
BUDGET_FRAC="$BUDGET_FRAC" BUDGET_NOTE="$BUDGET_NOTE" $PY - << 'PYEOF'
import json, os

meta = {
    "harvest_task": "reasoning-link Leg-B rung-3 rows -- scale series completion",
    "checkpoint_step": 155000,
    "budget_steps_completed": int(os.environ["BUDGET_STEPS_COMPLETED"]),
    "budget_steps_target": int(os.environ["BUDGET_STEPS_TARGET"]),
    "budget_frac": float(os.environ["BUDGET_FRAC"]),
    "disclosure": os.environ["BUDGET_NOTE"],
}

for corpus in ("openr1-mix-ext", "wikitext-mix-ext"):
    path = f"results/reasoning_link/leg_b_rung3_{corpus}_s0_k64.json"
    with open(path) as f:
        d = json.load(f)
    d["harvest_metadata"] = meta
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    print(f"injected harvest_metadata into {path}")
PYEOF

# ---------------------------------------------------------------------------
# Step 4 -- print an UNBLINDED per-cell gate summary to the log (cell mode
# was never blinded to begin with -- run_cell's own console print already
# excludes per_h from stdout by dict-key exclusion, not a blinding function;
# this step deliberately reads the full JSON since Leg-B rung-3 is a
# scale-series-completion harvest, not a fresh calibration decision point).
# ---------------------------------------------------------------------------
$PY - << 'PYEOF'
import json

print("\n=== rung-3 (1.31B) Leg-B gate summary ===")
for corpus in ("openr1-mix-ext", "wikitext-mix-ext"):
    path = f"results/reasoning_link/leg_b_rung3_{corpus}_s0_k64.json"
    d = json.load(open(path))
    per_h = d["per_h"]
    h1 = per_h.get("1", per_h.get(1))
    print(f"--- corpus={corpus} ---")
    print(f"  recovered_frac(h=1)={h1['recovered_frac']:.4f}  "
          f"premise_iii_pass={h1['premise_iii_pass']}  premise_iv_pass={h1['premise_iv_pass']}  "
          f"h1_confirmatory={h1['h1_confirmatory']}  h_ge2_confirmatory={h1['h_ge2_confirmatory']}")
    for h in ("1", "2", "3", "4"):
        hh = per_h.get(h, per_h.get(int(h)))
        if hh is None:
            continue
        print(f"    h={h}: recovered_frac={hh['recovered_frac']:.4f} cos_mean={hh['cos_mean']:.4f}")
PYEOF

touch results/reasoning_link/REASONING_LINK_RUNG3_DONE
echo "REASONING-LINK Leg-B rung-3 rows: complete."
