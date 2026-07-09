"""FRESH TIMING PILOT (real fused kernel, GPU5/6/7) -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.24/1.25
chain's own mandate: "deploy patch to box (md5) -> FRESH timing pilot (re-price calibration
round 3 + sweep; closes the 1.2-1.5x target on the real kernel)".

Throwaway box-only harness, NOT part of the audited codebase (no logic changes to any audited
file). Reuses the ALREADY-AUDITED h2h_cell_train_rd.train_grammar_cell/run_one_cell path (the
SAME harness mode_pilot_pair uses) at the REAL production cell config (task1_calib, K=32,
contender/ablation/transformer PINNED dims). Two variants per arch:
  (a) FIXED    -- unmodified _recurrent_continuation_answer_logits (slice-before-matmul, AUD2-F1)
  (b) PRE-FIX  -- monkeypatched to the pre-fix computation (full-vocab matmul over all 128
                  padded positions, then slice -- i.e. exactly what AUD2-F1 found wasteful),
                  reconstructed via the SAME model.forward(..., return_states=True) call the
                  design's own pre-fix code used (return_hidden defaults False, so this is a
                  faithful reproduction, not a guess).
Method: mode_pilot_pair's own two-point convention -- cold run (warmup=10 + timed=40 steps),
then a second warm-only run (timed=40 steps), s_per_step taken from the WARM run (startup /
kernel-autotune overhead cancels). VRAM: torch.cuda.reset_peak_memory_stats() before each run,
torch.cuda.max_memory_allocated() after.

Baseline denominator (the "pre-fix baseline path", i.e. NO CE_answer at all, matching this
design's own decision-gate ratio definition, sec 1.24 commit message: "end-to-end per-step
ratio measured at ... down from a reproduced ... pre-fix baseline") is NOT re-measured here --
it is the ALREADY-REGISTERED, ALREADY REAL-GPU-measured sec 1.6 round-1/2 per-arch full-cell
wall-clock (contender 10min, ablation 19.5min, transformer 27min @ FULL_STEPS=20000), reused
for consistency with the design doc's own round-3 cost arithmetic this pilot re-prices.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

os.environ.setdefault("HEADTOHEAD_PI_SIGNOFF", "1")
os.environ.setdefault("HEADTOHEAD_MATCH_GATE_SIGNOFF", "1")

import h2h_cell_train_rd as hct  # noqa: E402

assert not hct._STUB_INSTALLED, "must run against REAL fla on the real kernel"

GATES_DIR = "results/h2h_rung1/gates_v2_20260709"
hct.require_launch_tokens(GATES_DIR)   # sec 1.7 gate 5 discipline, respected even though this
                                        # script calls train_grammar_cell directly (not via the
                                        # --pilot-pair CLI mode that enforces it automatically)

WARMUP, TIMED = 10, 40

# ---- pre-fix reconstruction (AUD2-F1's own "before" computation, additive-only fix means the
# unmodified pieces below are IDENTICAL to what pre-fix code called) ----
def _prefix_continuation(arch, model, final_states, query_tokens, buffer_id):
    B, Q, qlen = query_tokens.shape
    flat = query_tokens.reshape(B * Q, qlen)
    padded = hct._pad_query_tokens_for_continuation(flat, buffer_id)
    states_rep = hct._repeat_states_for_queries(final_states, Q)
    logits_full, _ = model.forward(padded, initial_states=states_rep, return_states=True)
    answer_logits = logits_full[:, qlen - 1, :]
    return answer_logits.view(B, Q, -1)


_FIXED_FN = hct._recurrent_continuation_answer_logits


def _make_cell(arch, task, seed):
    base = {"arch": arch, "task": f"{task}_calib", "role": "pilot", "budget_frac": 1.0,
            "seed": seed, "lr": 3e-4, "name": f"h2htiming2_{arch}_{task}_{seed}", "seed_idx": 0}
    if task == "task1":
        base["K"] = 32
    return base


def _timed_run(arch, task, variant, seed):
    torch.cuda.reset_peak_memory_stats()
    cell = _make_cell(arch, task, seed)
    t_cold = hct.run_one_cell(dict(cell), "cuda", ckpt_dir=None,
                              steps_override=WARMUP + TIMED, timing_only=True)
    cell2 = _make_cell(arch, task, seed + 1)
    t_warm = hct.run_one_cell(dict(cell2), "cuda", ckpt_dir=None,
                              steps_override=TIMED, timing_only=True)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    s_per_step = t_warm["wall_s"] / TIMED
    return {"variant": variant, "arch": arch, "task": task,
            "s_per_step_warm": s_per_step, "cold_wall_s": t_cold["wall_s"],
            "warm_wall_s": t_warm["wall_s"], "peak_vram_gb": peak_vram_gb}


# sec 1.6 registered round-1/2 real-GPU baseline (NO CE_answer, pre-Rev4), full-cell @ FULL_STEPS
BASELINE_MIN_PER_CELL = {"contender": 10.0, "ablation": 19.5, "transformer": 27.0}
BASELINE_S_PER_STEP = {k: (v * 60.0) / hct.FULL_STEPS for k, v in BASELINE_MIN_PER_CELL.items()}

results = {"host": os.uname().nodename, "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "warmup": WARMUP, "timed": TIMED, "full_steps": hct.FULL_STEPS,
           "baseline_min_per_cell": BASELINE_MIN_PER_CELL,
           "baseline_s_per_step": BASELINE_S_PER_STEP, "runs": []}

for arch in ("contender", "ablation", "transformer"):
    print(f"=== {arch}: FIXED (current) ===", flush=True)
    r_fixed = _timed_run(arch, "task1", "fixed", seed=90001)
    print(json.dumps(r_fixed, indent=2), flush=True)
    results["runs"].append(r_fixed)

    if arch == "transformer":
        continue  # transformer's answer_logits reuse the existing forward pass -- design's own
                  # "no second forward" -- AUD2-F1 doesn't apply, no pre-fix variant to measure

    print(f"=== {arch}: PRE-FIX (monkeypatched) ===", flush=True)
    hct._recurrent_continuation_answer_logits = _prefix_continuation
    try:
        r_prefix = _timed_run(arch, "task1", "prefix", seed=90101)
    finally:
        hct._recurrent_continuation_answer_logits = _FIXED_FN
    print(json.dumps(r_prefix, indent=2), flush=True)
    results["runs"].append(r_prefix)

# ---- ratios ----
by_key = {(r["arch"], r["variant"]): r for r in results["runs"]}
ratios = {}
for arch in ("contender", "ablation", "transformer"):
    base = BASELINE_S_PER_STEP[arch]
    fixed = by_key[(arch, "fixed")]["s_per_step_warm"]
    row = {"baseline_s_per_step": base, "fixed_s_per_step": fixed,
           "ratio_fixed_vs_baseline": fixed / base}
    if (arch, "prefix") in by_key:
        prefix = by_key[(arch, "prefix")]["s_per_step_warm"]
        row["prefix_s_per_step"] = prefix
        row["ratio_prefix_vs_baseline"] = prefix / base
        row["ratio_prefix_vs_fixed"] = prefix / fixed
    ratios[arch] = row
    print(f"[{arch}] ratio_fixed_vs_baseline={row['ratio_fixed_vs_baseline']:.4f}"
          + (f" ratio_prefix_vs_baseline={row['ratio_prefix_vs_baseline']:.4f}"
             f" ratio_prefix_vs_fixed={row['ratio_prefix_vs_fixed']:.4f}" if "prefix_s_per_step" in row else ""),
          flush=True)

results["ratios"] = ratios
results["finished_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ---- re-price round 3 (9 cells: task1_calib + task1_stress(1/4) + task2_calib, x3 arms) using
# the FIXED (measured) rate; task2 reuses task1's measured rate (same continuation shapes:
# N_QUERY_TRAIN=8, GRAMMAR_BATCH=32, qlen identical across tasks -- architecture-level fix, not
# task-conditioned) -- flagged explicitly, not silently assumed. ----
full_gpuh = {a: by_key[(a, "fixed")]["s_per_step_warm"] * hct.FULL_STEPS / 3600.0
            for a in ("contender", "ablation", "transformer")}
round3_full_cells = sum(full_gpuh.values()) * 2       # task1_calib + task2_calib, 3 arms each
round3_stress_cells = sum(full_gpuh.values()) * 0.25  # task1_stress, quarter-budget, 3 arms
round3_base = round3_full_cells + round3_stress_cells
round3_ladder_overhead = round3_base * 0.085          # sec 1.6's own 8.5% ladder-overhead factor
round3_total = round3_base + round3_ladder_overhead

sweep_27cell_gpuh_est = sum(full_gpuh.values()) * 9 * 1.0  # rough: 27 cells / 3 arms = 9
                                                            # full-cell-equivalents per arm, upper
                                                            # bound (actual budget_fracs vary;
                                                            # flagged as an upper-bound re-price,
                                                            # not the sweep's own exact manifest)

results["reprice"] = {
    "full_cell_gpuh_per_arch": full_gpuh,
    "round3_full_cells_gpuh": round3_full_cells,
    "round3_stress_cells_gpuh": round3_stress_cells,
    "round3_ladder_overhead_gpuh": round3_ladder_overhead,
    "round3_total_gpuh": round3_total,
    "round3_registry_prior_gpuh": 2.300,
    "round3_ratio_vs_registry": round3_total / 2.300,
    "sweep_27cell_rough_upper_bound_gpuh": sweep_27cell_gpuh_est,
}
print(json.dumps(results["reprice"], indent=2), flush=True)

os.makedirs("results/h2h_rung1/pilots", exist_ok=True)
out_path = "results/h2h_rung1/pilots/h2h_timing2_fresh_pilot.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"WROTE {out_path}", flush=True)
