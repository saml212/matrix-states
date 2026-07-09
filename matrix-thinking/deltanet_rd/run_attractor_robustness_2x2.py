"""run_attractor_robustness_2x2.py -- ATTRACTOR-ROBUSTNESS 2x2 (qk-norm x gating), discharging
iclr-2027's disclosed limitation #2 (09_discussion_limitations.tex: "Single architecture family.
Vanilla, single-head, single-layer DeltaNet only. Gated DeltaNet (the actual Qwen3-Next production
config)... untested"). Directly answers the PI's skepticism: does the write-geometry attractor
(the SAME instrument 06_soft_fixes_fail.tex's F-geo-1/F-geo-2 "one attractor, three approach
angles" table used) survive (a) turning OFF the community-standard qk-l2-norm, and (b) turning ON
Gated-DeltaNet-style decay gating -- this build's approach-angles 4/5, extending that table.

Design (2x2 factorial, 14M dm256/L2 config -- lm_pretrain_rd.DeltaNetLM's own d_model=256/
d_state=64/n_layers=2 probe-tier config, ~14M params, matching iclr-2027's own rung-1 scale-ladder
point and h2h_cell_train_rd.py's CONTENDER_KW architecture pin):

  cell = (qk_l2norm_in_kernel in {True, False}) x (gated_delta_active in {False, True})
  baseline cell = (True, False) -- byte-identical to every prior chapter's own hardcoded behavior.

ONE lm_pretrain_rd.py training run per cell (NOT two): the task's own pre-registered budget,
4 screening cells at the CALIBRATED CALIBRATED_GPU_H_PER_CELL=0.2524 GPU-h/cell rate (cited:
h2h_cell_train_rd.py's own `FULL_STEPS = 20_000  # sec 1.6's 20,000-step cell (0.2524 GPU-h
realized rate)` -- the SAME architecture at the SAME step count), totals to
4 * 0.2524 = 1.0096 GPU-h [?] ~1.0 GPU-h, which is only arithmetically consistent with ONE
20k-step training run per cell. BOTH outcome instruments therefore read off the SAME checkpoint
each cell produces:

  (1) lm_attractor_probe_rd.py's Gram-deviation / effective-rank (UNMODIFIED, imported/subprocessed
      as-is) -- the geometric leg, Track C's own established real-text use of this instrument.
  (2) reasoning_link_probe.py's `run_cell(ckpt_path, K=32, hops=(1,2,3,4), surgery="native", ...)`
      (UNMODIFIED, imported as-is) -- the behavioral rec@0.9 leg, h in {1,2,3,4} (H_TEST tuple).

DISCLOSED, NOT HIDDEN: leg (2) is `reasoning_link_probe.py`'s zero-shot K-cycle-transplant probe
applied to a REAL-TEXT-trained checkpoint -- the EXACT construct iclr-2027's own limitation #11
disclosed as PROBE-INVALID / categorically 0.0 across every one of 108+ (checkpoint, corpus, hop,
surface-form, training-regime) cells tried anywhere in this project to date (never a licensed
REFUTE, per that limitation's own "an instrument that cannot clear its own sanity floor does not
get to report a negative result" rule) -- and HEAD_TO_HEAD_DEMO_DESIGN's own build log explicitly
prefers grammar_rd.py's NATIVE recovered_frac@0.9 (trained-on-the-task) over this construct for
exactly that reason. It is wired here anyway because (a) it is the ONLY reading consistent with
this build's own pre-registered ~1.0 GPU-h / 4-cell budget (a second, from-scratch grammar-training
leg would ~2x that cost, requiring a from-scratch grammar-trained checkpoint per cell -- NOT what
the calibrated rate implies), (b) it is cheap (reuses an already-audited, unmodified instrument,
zero new measurement code), and (c) a categorical-floor-preserved-across-all-4-cells reading is
STILL informative for a ROBUSTNESS check specifically (it would show the null transfers uniformly,
not specifically triggered by qk-norm) -- while an ablation-specific escape from the floor would
itself be a genuinely notable, worth-flagging-to-the-PI finding. The ASSESS phase must read this
leg's numbers with this precedent in hand, not as a fresh/naive readout.

Pre-registered escalation rule (n=1 screening -> n=3, CONFIG data below, ESCALATION_RULE):
fires iff ANY non-baseline cell's Gram-deviation direction (delta vs the baseline cell, SAME-
corpus reading) exceeds 2x the ARCHIVED cross-seed noise floor at this EXACT (dm256/ds64/L2)
config and corpus -- cited source: experiment-runs/2026-07-06_trajectory_probes/
reference_finals_archived.json, "mixcontrol" family, `archived4_gram_deviation_mean` field,
cross-seed {0,1,2} same-corpus std at "openr1-mix-ext" (std=2.244355, n=3, audit-corrected) /
"wikitext-mix-ext" (std=2.216699, n=3) -- see ARCHIVED_SEED_NOISE_GRAM_DEV_STD below,
independently recomputed from that same archive file at build time (not hand-copied from prose).

Hard-abort ceilings (never overridable from the CLI, matching this codebase's own budget_guard
convention): SCREENING_CEILING_GPU_H (~1.0 GPU-h, 4 cells) and ESCALATION_CEILING_GPU_H (3.03
GPU-h, 12 cells) -- both refuse to launch a wave whose PROJECTED cost (n_cells *
CALIBRATED_GPU_H_PER_CELL) exceeds the registered ceiling. The two read-only probe passes per
cell (attractor-probe + run_cell) are NOT separately GPU-h-budgeted -- see project_gpu_hours'
docstring for why folding them into the training-only calibrated rate is the task's own
accounting, not a margin this file invented.

CONCURRENCY: h2h_cell_train_rd.py and probe_head_rd.py are being concurrently edited elsewhere
this session -- this file does NOT import either (their own joint-probe-training machinery is
H2H-specific -- gate tokens, dial mechanism, ProbeRig -- and not needed here; this build's own
leg (2) uses reasoning_link_probe.py's STABLE, independent `run_cell`, never touching those two
files). lm_pretrain_rd.py's own edits (the qk_l2norm_in_kernel / gated_delta_active flags this
runner exercises) are additive and flag-gated -- see that file's own bit-identical-default smoke.

Resume-safe (validity-checked result JSONs, not existence -- mirrors run_lm_rd_trackc_sweep.py's
own is_done_cell convention) and tmux/supervisor-loop compatible (CLAUDE.md's own
`while [ ! -f STOP ]; do <cmd>; sleep 15; done` pattern -- re-invoking `--run` after a crash
resumes from wherever it left off, per-leg, per-cell).

Usage:
  python run_attractor_robustness_2x2.py --smoke
  python run_attractor_robustness_2x2.py --run --out-dir results/attractor_robustness_2x2 \\
      --data-dir /data/deltanet_rd_data --n-seeds 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports, matching every sibling file's own convention

LM_PRETRAIN = os.path.join(HERE, "lm_pretrain_rd.py")
ATTRACTOR_PROBE = os.path.join(HERE, "lm_attractor_probe_rd.py")

# ---------------------------------------------------------------------------
# Config (task's own pre-registered constraints, CONFIG data -- not invented at build time)
# ---------------------------------------------------------------------------

BASE_KW = dict(d_model=256, d_state=64, n_layers=2, conv_size=4, num_heads=1)  # 14M dm256/L2
CORPUS = "openr1-mix-ext"    # SAME corpus the archived seed-noise reference below was measured on
                              # (CLAUDE.md: "use the SAME dataset for ALL experiments in a
                              # comparison") -- Track C's own DEPLOY-PIN-2-style precedent corpus.
SEQ_LEN = 512                 # Track C's own SEQ_LEN
BATCH_SIZE = 32                # Track C's own BATCH_SIZE
LM_STEPS = 20_000              # task's own pre-registered step count
GRAMMAR_K = 32                 # task's own pre-registered K (matches DELTANET_RD_EXACTNESS_
                                # DESIGN.md's own "K=32 <= d_state=64" primary anomaly cell, and
                                # h2h_cell_train_rd.py's DEPLOY-PIN-5 task2 pin)
GRAMMAR_HOPS = (1, 2, 3, 4)     # H_TEST in reasoning_link_probe.py (module-level H_TEST constant)
                                 # -- "h=1-4" per the task brief, matching 06_soft_fixes_fail.tex's
                                 # own h in {1,2,3,4} reporting convention.

CALIBRATED_GPU_H_PER_CELL = 0.2524   # cited: h2h_cell_train_rd.py's own FULL_STEPS=20_000 comment
                                       # ("0.2524 GPU-h realized rate") -- the SAME architecture at
                                       # the SAME step count as this build's own real-text leg.
SCREENING_N_SEEDS = 1
ESCALATION_N_SEEDS = 3
SCREENING_CELLS = 4            # 2x2 x 1 seed
ESCALATION_CELLS = 12           # 2x2 x 3 seeds
# Hard-abort ceilings -- registered here, never overridable from the CLI (budget_guard convention,
# run_lm_rd_trackc_sweep.py). BOTH figures are taken verbatim from the task's own pre-registered
# numbers (screening "4 cells ~1.0 GPU-h", escalation "12 cells, 3.03 GPU-h") -- both equal
# n_cells * CALIBRATED_GPU_H_PER_CELL exactly (no added margin): the two READ-ONLY probe passes
# per cell (attractor-probe + run_cell) are NOT separately GPU-h-budgeted here -- they run on the
# SAME box within the SAME per-cell wall-clock window the training-run rate was calibrated over
# (no backward pass, no optimizer, materially cheaper than the training run they follow), so
# folding them into the training-only calibrated rate is the task's own accounting, not a margin
# this file invented. If on-box realized rates show that assumption wrong, DO NOT silently widen
# these constants -- re-calibrate and re-register first (mirrors the "a calibration run before a
# big sweep is mandatory" hard rule).
SCREENING_CEILING_GPU_H = round(SCREENING_CELLS * CALIBRATED_GPU_H_PER_CELL, 4)      # ~1.0096
ESCALATION_CEILING_GPU_H = 3.03   # task-given hard-abort constant, taken verbatim.

# Archived cross-seed noise floor for the escalation rule's own trigger threshold.
# AUDIT CORRECTION (build audit, 2026-07-09): the build originally cited
# `archived4_gram_deviation_mean`'s per-corpus stds (0.8468 / 2.0344) -- but that field is
# POOLED ACROSS 4 out-of-distribution probe corpora (build_tidy.py:22), NOT a same-corpus
# statistic, and understated openr1-mix-ext's true same-corpus seed noise by 2.65x (biasing
# the n=1 screening toward false-positive escalation on pure seed noise). The values below
# are the audit's independent recomputation of the TRUE same-corpus cross-seed std, computed
# directly from the raw archived probe JSONs
# (experiment-runs/2026-07-06_trajectory_probes/mixcontrol/{corpus}_s{0,1,2}.json, the exact
# same 3 checkpoints at the IDENTICAL dm256/ds64/L2 architecture) using the exact aggregation
# gram_deviation_same_corpus performs. Population std (ddof=0).
ARCHIVED_SEED_NOISE_GRAM_DEV_STD = {
    "openr1-mix-ext": 2.244355,
    "wikitext-mix-ext": 2.216699,
}
ESCALATION_TRIGGER_MULTIPLE = 2.0

BASELINE_CELL_KEY = "qkTrue_gateFalse"   # qk_l2norm_in_kernel=True, gated_delta_active=False --
                                          # every prior chapter's own hardcoded behavior.


def cell_configs(n_seeds: int) -> list[dict]:
    """Builds the 2x2(xN) manifest. n_seeds=1 -> SCREENING_CELLS=4; n_seeds=3 -> ESCALATION_CELLS=12
    (asserted below so a caller passing an unregistered n_seeds is refused, not silently allowed to
    launch an un-costed wave size)."""
    assert n_seeds in (SCREENING_N_SEEDS, ESCALATION_N_SEEDS), (
        f"n_seeds={n_seeds} is not a pre-registered wave size -- only {SCREENING_N_SEEDS} "
        f"(screening) or {ESCALATION_N_SEEDS} (escalation, only after should_escalate() fires) "
        f"are costed/ceiling-checked by this file.")
    cells = []
    for qk_norm in (True, False):
        for gated in (False, True):
            for seed in range(n_seeds):
                key = f"qk{qk_norm}_gate{gated}"
                cells.append({
                    "key": key, "name": f"attrrob_{key}_s{seed}",
                    "qk_l2norm_in_kernel": qk_norm, "gated_delta_active": gated, "seed": seed,
                })
    return cells


def is_baseline(cell: dict) -> bool:
    return cell["qk_l2norm_in_kernel"] is True and cell["gated_delta_active"] is False


# ---------------------------------------------------------------------------
# Real-text training leg (lm_pretrain_rd.py, subprocess -- reuses the fully audited train()
# pipeline unmodified, matching run_lm_rd_trackc_sweep.py's own build_cmd_cell precedent exactly).
# ---------------------------------------------------------------------------

def lm_ckpt_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "checkpoints")


def lm_result_path(out_dir: str, cell: dict) -> str:
    return os.path.join(out_dir, f"{cell['name']}_lm.json")


def is_done_lm_cell(out_dir: str, cell: dict) -> bool:
    """Validity check, NOT existence (CLAUDE.md's own resume-safety rule) -- mirrors
    run_lm_rd_trackc_sweep.is_done_cell's identity-tuple discipline."""
    p = lm_result_path(out_dir, cell)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return False
    if not isinstance(d, dict) or d.get("complete") is not True or d.get("timed_out"):
        return False
    required = ("corpus", "seed", "d_model", "d_state", "n_layers", "seq_len", "steps",
                "steps_completed", "batch_size")
    if not all(k in d for k in required):
        return False
    if d.get("steps_completed", 0) < LM_STEPS:
        return False
    if (d.get("corpus") != CORPUS or d.get("seed") != cell["seed"]
            or d.get("d_model") != BASE_KW["d_model"] or d.get("d_state") != BASE_KW["d_state"]
            or d.get("n_layers") != BASE_KW["n_layers"] or d.get("seq_len") != SEQ_LEN
            or d.get("steps") != LM_STEPS or d.get("batch_size") != BATCH_SIZE):
        return False
    if not (d.get("checkpoints") or []):
        return False
    return True


def final_ckpt_path(out_dir: str, cell: dict) -> str | None:
    """The LAST checkpoint_path recorded in this cell's own (valid, complete) result JSON --
    never re-derived from a filename guess, so a caller always reads the SAME artifact
    lm_pretrain_rd.py itself wrote."""
    p = lm_result_path(out_dir, cell)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    ck = d.get("checkpoints") or []
    return ck[-1].get("checkpoint_path") if ck else None


def build_lm_pretrain_cmd(cell: dict, out_dir: str, data_dir: str, timeout: float) -> list[str]:
    cmd = [sys.executable, LM_PRETRAIN,
           "--corpus", CORPUS, "--data-dir", data_dir,
           "--d-model", str(BASE_KW["d_model"]), "--d-state", str(BASE_KW["d_state"]),
           "--n-layers", str(BASE_KW["n_layers"]), "--conv-size", str(BASE_KW["conv_size"]),
           "--num-heads", str(BASE_KW["num_heads"]), "--seq-len", str(SEQ_LEN),
           "--batch-size", str(BATCH_SIZE), "--steps", str(LM_STEPS),
           "--ckpt-every", str(LM_STEPS),        # ONE checkpoint, at the very end (lean screening)
           "--seed", str(cell["seed"]), "--internal-timeout", str(max(1, timeout - 30)),
           "--ckpt-dir", lm_ckpt_dir(out_dir),
           "--out", lm_result_path(out_dir, cell)]
    if not cell["qk_l2norm_in_kernel"]:
        cmd.append("--qk-l2norm-off")
    if cell["gated_delta_active"]:
        cmd.append("--gated-delta-active")
    return cmd


# ---------------------------------------------------------------------------
# Gram-deviation / effective-rank leg (lm_attractor_probe_rd.py, UNMODIFIED, subprocess).
# ---------------------------------------------------------------------------

def attractor_probe_result_path(out_dir: str, cell: dict) -> str:
    return os.path.join(out_dir, f"{cell['name']}_attractor_probe.json")


def is_done_attractor_probe(out_dir: str, cell: dict) -> bool:
    p = attractor_probe_result_path(out_dir, cell)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return isinstance(d, dict) and "pooled" in d and "per_checkpoint" in d
    except Exception:
        return False


def build_attractor_probe_cmd(ckpt_path: str, out_dir: str, cell: dict, data_dir: str) -> list[str]:
    return [sys.executable, ATTRACTOR_PROBE, "--checkpoints", ckpt_path,
            "--data-dir", data_dir, "--seq-len", str(SEQ_LEN),
            "--out", attractor_probe_result_path(out_dir, cell)]


def gram_deviation_same_corpus(attractor_probe_json: dict, ckpt_path: str) -> float | None:
    """Pooled-across-layers gram_deviation_mean for the SAME corpus the checkpoint trained on
    (matches the archived reference's own same-corpus, pooled-across-both-n_layers=2 convention --
    see ARCHIVED_SEED_NOISE_GRAM_DEV_STD's own docstring; flagged for audit re-check against
    analyze_probe_wave2.py's own exact aggregation if bit-for-bit parity to the archive ever
    matters downstream, not assumed identical by construction here)."""
    per_ckpt = attractor_probe_json.get("per_checkpoint", {}).get(ckpt_path)
    if per_ckpt is None:
        return None
    per_layer = per_ckpt.get("per_corpus", {}).get(CORPUS, {}).get("per_layer", {})
    vals = [v["gram_deviation_mean"] for v in per_layer.values()
            if isinstance(v, dict) and v.get("gram_deviation_mean") is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# rec@0.9 h=1-4 leg (reasoning_link_probe.run_cell, UNMODIFIED, in-process -- lazily imported so
# --smoke never triggers that file's own module-level _ensure_fla_stub() side effect, which would
# collide with THIS file's own extended CPU stub, see smoke()/_install_cpu_fla_stub below).
# ---------------------------------------------------------------------------

def rec_at_09_result_path(out_dir: str, cell: dict) -> str:
    return os.path.join(out_dir, f"{cell['name']}_rec_at_09.json")


def is_done_rec_at_09(out_dir: str, cell: dict) -> bool:
    p = rec_at_09_result_path(out_dir, cell)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return isinstance(d, dict) and "per_h" in d and set(str(h) for h in GRAMMAR_HOPS) <= set(
            str(k) for k in d["per_h"])
    except Exception:
        return False


def run_rec_at_09(ckpt_path: str, out_dir: str, cell: dict, device: str) -> dict:
    """Calls reasoning_link_probe.run_cell UNMODIFIED (outcome instrument, per this build's own
    "outcome instruments UNMODIFIED" mandate) -- 'native' surgery (no frozen-bias/geo3 surgery,
    matching this build's own baseline-attractor scope, per module docstring). seed_override is
    used (not the leg-keyed sec 4.6 formula, which is reserved for reasoning_link_probe.py's OWN
    registered leg_a/leg_b grid) -- a fixed, cell-specific, collision-free seed derived from this
    file's own cell key, disclosed here rather than silently defaulting to seed=0 for every cell."""
    from reasoning_link_probe import run_cell   # lazy: real/stubbed fla resolved at call time only
    seed_override = (hash(("attractor_robustness_2x2", cell["key"])) % 900_000_000) + 100_000_000
    result = run_cell(ckpt_path, K=GRAMMAR_K, hops=GRAMMAR_HOPS, surgery="native", device=device,
                       seed_override=seed_override)
    with open(rec_at_09_result_path(out_dir, cell), "w") as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return result


# ---------------------------------------------------------------------------
# Pre-registered escalation rule (CONFIG data -- pure function, unit-tested by smoke()).
# ---------------------------------------------------------------------------

ESCALATION_RULE = {
    "description": (
        "n=1 screening -> n=3 escalation fires iff ANY non-baseline cell's Gram-deviation "
        "(same-corpus, pooled-across-layers gram_deviation_mean) differs from the baseline cell "
        "(qk_l2norm_in_kernel=True, gated_delta_active=False) by more than "
        f"{ESCALATION_TRIGGER_MULTIPLE}x the archived cross-seed noise floor at this exact "
        "(dm256/ds64/L2) config and corpus."
    ),
    "trigger_multiple": ESCALATION_TRIGGER_MULTIPLE,
    "noise_floor_source": (
        "experiment-runs/2026-07-06_trajectory_probes/reference_finals_archived.json, "
        "'mixcontrol' family, archived4_gram_deviation_mean, cross-seed {0,1,2} population std "
        "at the SAME (dm256/ds64/L2) architecture -- see ARCHIVED_SEED_NOISE_GRAM_DEV_STD."
    ),
    "noise_floor_std": ARCHIVED_SEED_NOISE_GRAM_DEV_STD,
    "baseline_cell_key": BASELINE_CELL_KEY,
}


def should_escalate(gram_dev_by_cell_key: dict, corpus: str = CORPUS) -> tuple[bool, dict]:
    """gram_dev_by_cell_key: {cell_key -> same-corpus gram_deviation_mean float}. Returns
    (fire: bool, detail: dict) -- detail always reports EVERY non-baseline cell's delta and
    whether it individually exceeded the threshold, never just the aggregate bool (so a coordinator
    can see WHICH cell(s) drove the trigger, not merely that one did)."""
    assert corpus in ARCHIVED_SEED_NOISE_GRAM_DEV_STD, (
        f"corpus={corpus!r} has no archived seed-noise reference -- should_escalate cannot "
        f"evaluate the pre-registered rule without one; registered corpora: "
        f"{sorted(ARCHIVED_SEED_NOISE_GRAM_DEV_STD)}")
    assert BASELINE_CELL_KEY in gram_dev_by_cell_key, (
        f"baseline cell {BASELINE_CELL_KEY!r} missing from gram_dev_by_cell_key -- cannot compute "
        f"deltas against it")
    baseline = gram_dev_by_cell_key[BASELINE_CELL_KEY]
    threshold = ESCALATION_TRIGGER_MULTIPLE * ARCHIVED_SEED_NOISE_GRAM_DEV_STD[corpus]
    per_cell = {}
    fire = False
    for key, val in gram_dev_by_cell_key.items():
        if key == BASELINE_CELL_KEY:
            continue
        delta = val - baseline
        exceeds = abs(delta) > threshold
        per_cell[key] = {"gram_deviation_mean": val, "delta_vs_baseline": delta,
                          "threshold": threshold, "exceeds": exceeds}
        fire = fire or exceeds
    return fire, {"baseline_gram_deviation_mean": baseline, "threshold": threshold,
                  "corpus": corpus, "per_cell": per_cell}


# ---------------------------------------------------------------------------
# Orchestration -- sequential, exception-isolated per (cell, leg), resume-safe (validity-checked).
# ---------------------------------------------------------------------------

def project_gpu_hours(n_cells: int) -> float:
    """n_cells * CALIBRATED_GPU_H_PER_CELL -- see SCREENING_CEILING_GPU_H's own definition above
    for why the two read-only probe legs are not separately budgeted."""
    return round(n_cells * CALIBRATED_GPU_H_PER_CELL, 4)


def budget_guard(n_seeds: int) -> None:
    n_cells = SCREENING_CELLS if n_seeds == SCREENING_N_SEEDS else ESCALATION_CELLS
    ceiling = SCREENING_CEILING_GPU_H if n_seeds == SCREENING_N_SEEDS else ESCALATION_CEILING_GPU_H
    projected = project_gpu_hours(n_cells)
    if projected > ceiling:
        print(f"HARD-ABORT: projected {projected} GPU-h for n_seeds={n_seeds} ({n_cells} cells) "
              f"exceeds the registered ceiling {ceiling} GPU-h -- no override.", file=sys.stderr)
        raise SystemExit(3)
    print(f"budget_guard: n_seeds={n_seeds} ({n_cells} cells) projected {projected} GPU-h <= "
          f"ceiling {ceiling} GPU-h -- OK", flush=True)


def run_wave(out_dir: str, data_dir: str, n_seeds: int, device: str, timeout: float,
             dry_run: bool = False) -> dict:
    budget_guard(n_seeds)
    os.makedirs(out_dir, exist_ok=True)
    cells = cell_configs(n_seeds)
    failed = []

    for cell in cells:
        # --- leg 1: real-text training (subprocess, exception-isolated) ---
        if not is_done_lm_cell(out_dir, cell):
            cmd = build_lm_pretrain_cmd(cell, out_dir, data_dir, timeout)
            print(f"[{cell['name']}] leg=lm_pretrain cmd={' '.join(cmd)}", flush=True)
            if not dry_run:
                try:
                    log_path = os.path.join(out_dir, f"{cell['name']}_lm.log")
                    with open(log_path, "w") as lf:
                        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
                    if rc != 0 or not is_done_lm_cell(out_dir, cell):
                        failed.append((cell["name"], "lm_pretrain", f"rc={rc}"))
                        continue
                except Exception as e:
                    failed.append((cell["name"], "lm_pretrain", repr(e)))
                    continue
        else:
            print(f"[{cell['name']}] leg=lm_pretrain already done, resume-skip", flush=True)

        if dry_run:
            continue
        ckpt_path = final_ckpt_path(out_dir, cell)
        if not ckpt_path or not os.path.exists(ckpt_path):
            failed.append((cell["name"], "lm_pretrain", "no valid checkpoint_path in result JSON"))
            continue

        # --- leg 2: Gram-deviation / effective-rank (subprocess, exception-isolated) ---
        if not is_done_attractor_probe(out_dir, cell):
            cmd = build_attractor_probe_cmd(ckpt_path, out_dir, cell, data_dir)
            print(f"[{cell['name']}] leg=attractor_probe cmd={' '.join(cmd)}", flush=True)
            try:
                log_path = os.path.join(out_dir, f"{cell['name']}_attractor_probe.log")
                with open(log_path, "w") as lf:
                    rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
                if rc != 0 or not is_done_attractor_probe(out_dir, cell):
                    failed.append((cell["name"], "attractor_probe", f"rc={rc}"))
            except Exception as e:
                failed.append((cell["name"], "attractor_probe", repr(e)))
        else:
            print(f"[{cell['name']}] leg=attractor_probe already done, resume-skip", flush=True)

        # --- leg 3: rec@0.9 h=1-4 (in-process, exception-isolated) ---
        if not is_done_rec_at_09(out_dir, cell):
            print(f"[{cell['name']}] leg=rec_at_09 (reasoning_link_probe.run_cell, K={GRAMMAR_K}, "
                  f"hops={GRAMMAR_HOPS})", flush=True)
            try:
                run_rec_at_09(ckpt_path, out_dir, cell, device)
            except Exception as e:
                failed.append((cell["name"], "rec_at_09", repr(e)))
        else:
            print(f"[{cell['name']}] leg=rec_at_09 already done, resume-skip", flush=True)

    report = aggregate(out_dir, cells)
    report["failed"] = failed
    with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def aggregate(out_dir: str, cells: list[dict]) -> dict:
    gram_dev_by_key = {}
    rec_at_09_by_key = {}
    for cell in cells:
        if is_done_attractor_probe(out_dir, cell):
            with open(attractor_probe_result_path(out_dir, cell)) as f:
                probe_json = json.load(f)
            ckpt_path = final_ckpt_path(out_dir, cell)
            gd = gram_deviation_same_corpus(probe_json, ckpt_path) if ckpt_path else None
            if gd is not None:
                gram_dev_by_key.setdefault(cell["key"], []).append(gd)
        if is_done_rec_at_09(out_dir, cell):
            with open(rec_at_09_result_path(out_dir, cell)) as f:
                rec_json = json.load(f)
            per_h = rec_json.get("per_h", {})
            rec_at_09_by_key.setdefault(cell["key"], []).append(
                {h: per_h[str(h)].get("recovered_frac") for h in GRAMMAR_HOPS if str(h) in per_h})

    gram_dev_mean_by_key = {k: sum(v) / len(v) for k, v in gram_dev_by_key.items() if v}
    report = {"cells": [c["name"] for c in cells], "gram_dev_mean_by_cell_key": gram_dev_mean_by_key,
              "rec_at_09_by_cell_key": rec_at_09_by_key,
              "rec_at_09_note": ("PROBE-INVALID / categorical-0.0 floor expected in every arm "
                                 "(zero-shot K-cycle transplant, see module docstring) -- "
                                 "NON-DECISIONAL, not read by should_escalate().")}
    if BASELINE_CELL_KEY in gram_dev_mean_by_key and len(gram_dev_mean_by_key) == len(
            {c["key"] for c in cells}):
        fire, detail = should_escalate(gram_dev_mean_by_key)
        report["escalation"] = {"fire": fire, **detail, "rule": ESCALATION_RULE}
    return report


# ---------------------------------------------------------------------------
# Smoke suite (CPU-stub, no CUDA/real-fla/GPU/data required). Covers:
#   [1] extended CPU fla stub install (chunk_delta_rule + chunk_gated_delta_rule)
#   [2] lm_pretrain_rd bit-identical-default assertion under the stub
#   [3] lm_pretrain_rd gated-path forward/backward/grad-finite under the stub
#   [4] escalation-rule unit test (synthetic trigger / don't-trigger)
#   [5] resume-safety / validity-check unit test (is_done_lm_cell on synthetic JSONs)
#   [6] cell_configs() manifest sanity
# The REAL-kernel box-smoke counterpart of [2]/[3] is lm_pretrain_rd.py's own smoke() item [18]
# (registered there, requires CUDA -- run it on-box before any GPU launch of this file).
# ---------------------------------------------------------------------------

def _install_cpu_fla_stub() -> None:
    """Self-contained (does NOT touch/import h2h_fla_stub_rd.py or reasoning_link_probe.py's own
    _ensure_fla_stub -- both are either locked this session or would install an INCOMPATIBLE stub
    missing fla.ops.gated_delta_rule). Extends the SAME "k-ROUTING FIXED" stub pattern this
    codebase already uses (h2h_fla_stub_rd.py / smoke_frozen_bias_lm.py) with a
    chunk_gated_delta_rule stand-in that also depends on g (log-space decay), so a gradient-based
    check on b_alpha_proj is exercised meaningfully, not vacuously."""
    import types
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if getattr(sys.modules.get("fla"), "_ATTRRB_CPU_STUB", False):
        return   # idempotent

    class _StubShortConvolution(nn.Module):
        def __init__(self, hidden_size, kernel_size=4, bias=False, activation="silu"):
            super().__init__()
            self.activation = activation
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                                   padding=kernel_size - 1, bias=bias)

        def forward(self, x, cache=None):
            B, T, D = x.shape
            out = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
            if self.activation == "silu":
                out = F.silu(out)
            return out, None

    class _StubRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    def _stub_chunk_delta_rule(q, k, v, beta, initial_state=None, output_final_state=True,
                                use_qk_l2norm_in_kernel=True):
        B, T, H, Dh = q.shape
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        qk_gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        o = v * torch.sigmoid(beta).unsqueeze(-1) * qk_gate
        final_state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype, device=q.device)
        return o, final_state

    def _stub_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=None, output_final_state=True,
                                      use_qk_l2norm_in_kernel=False, **kwargs):
        B, T, H, Dh = q.shape
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        qk_gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        alpha_gate = torch.exp(g).unsqueeze(-1)   # g is log-space -> alpha=exp(g) in (0,1)
        o = v * torch.sigmoid(beta).unsqueeze(-1) * qk_gate * alpha_gate
        final_state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype, device=q.device)
        return o, final_state

    fla_mod = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_delta_rule = types.ModuleType("fla.ops.delta_rule")
    fla_ops_gated_delta_rule = types.ModuleType("fla.ops.gated_delta_rule")
    fla_modules.ShortConvolution = _StubShortConvolution
    fla_modules.RMSNorm = _StubRMSNorm
    fla_ops_delta_rule.chunk_delta_rule = _stub_chunk_delta_rule
    fla_ops_gated_delta_rule.chunk_gated_delta_rule = _stub_chunk_gated_delta_rule
    fla_mod.modules = fla_modules
    fla_mod.ops = fla_ops
    fla_ops.delta_rule = fla_ops_delta_rule
    fla_ops.gated_delta_rule = fla_ops_gated_delta_rule
    fla_mod._ATTRRB_CPU_STUB = True
    sys.modules["fla"] = fla_mod
    sys.modules["fla.modules"] = fla_modules
    sys.modules["fla.ops"] = fla_ops
    sys.modules["fla.ops.delta_rule"] = fla_ops_delta_rule
    sys.modules["fla.ops.gated_delta_rule"] = fla_ops_gated_delta_rule


def smoke() -> None:
    print("=" * 60 + "\n  ATTRACTOR_ROBUSTNESS_2X2 SMOKE GATE (CPU stub)\n" + "=" * 60)
    FAILURES = []

    def _report(item, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
        if not ok:
            FAILURES.append(item)

    print("\n[1] install extended CPU fla stub (chunk_delta_rule + chunk_gated_delta_rule)")
    _install_cpu_fla_stub()
    import torch
    import torch.nn.functional as F
    from lm_pretrain_rd import DeltaNetLM, DeltaNetLMMixer, _MIN_KERNEL_T
    _report("1", True, "stub installed, lm_pretrain_rd imported under it")

    print("\n[2] bit-identical-default assertion: no-kwargs vs explicit "
          "(qk_l2norm_in_kernel=True, gated_delta_active=False)")
    V, T = 500, _MIN_KERNEL_T
    torch.manual_seed(37)
    m_a = DeltaNetLM(V, d_model=64, d_state=64, n_layers=1, conv_size=4)
    torch.manual_seed(37)
    m_b = DeltaNetLM(V, d_model=64, d_state=64, n_layers=1, conv_size=4,
                      qk_l2norm_in_kernel=True, gated_delta_active=False)
    x = torch.randint(0, V, (4, T))
    with torch.no_grad():
        eq = torch.equal(m_a(x), m_b(x))
    _report("2", eq, "bit-identical" if eq else "DIVERGED -- default-path regression")

    print("\n[3] gated-path forward/backward/grad-finite across the full 2x2 "
          "(qk_l2norm_in_kernel x gated_delta_active), incl. b_alpha_proj construction/gradient")
    all_ok = True
    for qk_norm in (True, False):
        for gated in (False, True):
            torch.manual_seed(7)
            m = DeltaNetLM(V, d_model=64, d_state=64, n_layers=2, conv_size=4,
                            qk_l2norm_in_kernel=qk_norm, gated_delta_active=gated)
            xx = torch.randint(0, V, (4, T))
            logits = m(xx)
            finite_logits = torch.isfinite(logits).all().item()
            loss = F.cross_entropy(logits.reshape(-1, V), xx.reshape(-1))
            loss.backward()
            finite_grad = all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            b_alpha_ok = True
            if gated:
                b_alpha_ok = (m.blocks[0].mixer.b_alpha_proj is not None
                              and m.blocks[0].mixer.b_alpha_proj.weight.grad is not None
                              and torch.isfinite(m.blocks[0].mixer.b_alpha_proj.weight.grad).all().item())
            else:
                b_alpha_ok = m.blocks[0].mixer.b_alpha_proj is None
            cell_ok = finite_logits and finite_grad and b_alpha_ok
            all_ok = all_ok and cell_ok
            print(f"    qk_l2norm_in_kernel={qk_norm!s:>5s} gated_delta_active={gated!s:>5s}: "
                  f"{'PASS' if cell_ok else 'FAIL'} (loss={loss.item():.4f})")
    _report("3", all_ok)

    print("\n[4] escalation-rule unit test: synthetic trigger / don't-trigger")
    baseline_gd = 20.0
    noise = ARCHIVED_SEED_NOISE_GRAM_DEV_STD[CORPUS]
    no_trigger = {BASELINE_CELL_KEY: baseline_gd,
                  "qkTrue_gateTrue": baseline_gd + 0.5 * noise,
                  "qkFalse_gateFalse": baseline_gd - 0.5 * noise,
                  "qkFalse_gateTrue": baseline_gd + 1.9 * noise}
    fire_no, detail_no = should_escalate(no_trigger)
    trigger = dict(no_trigger)
    trigger["qkFalse_gateTrue"] = baseline_gd + (2.5 * noise)
    fire_yes, detail_yes = should_escalate(trigger)
    ok4 = (fire_no is False) and (fire_yes is True) and detail_yes["per_cell"]["qkFalse_gateTrue"]["exceeds"]
    _report("4", ok4, f"no-trigger fire={fire_no}, trigger fire={fire_yes}")

    print("\n[5] resume-safety / validity-check unit test (is_done_lm_cell)")
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="attrrob_smoke_")
    cell = cell_configs(1)[0]
    ok5a = is_done_lm_cell(tmpdir, cell) is False   # no file at all -> not done
    partial = {"complete": False, "corpus": CORPUS, "seed": cell["seed"],
               "d_model": BASE_KW["d_model"], "d_state": BASE_KW["d_state"],
               "n_layers": BASE_KW["n_layers"], "seq_len": SEQ_LEN, "steps": LM_STEPS,
               "steps_completed": 100, "batch_size": BATCH_SIZE, "checkpoints": []}
    with open(lm_result_path(tmpdir, cell), "w") as f:
        json.dump(partial, f)
    ok5b = is_done_lm_cell(tmpdir, cell) is False   # complete=False -> not done
    complete = dict(partial, complete=True, steps_completed=LM_STEPS,
                    checkpoints=[{"checkpoint_path": "/fake/ckpt.pt", "step": LM_STEPS}])
    with open(lm_result_path(tmpdir, cell), "w") as f:
        json.dump(complete, f)
    ok5c = is_done_lm_cell(tmpdir, cell) is True    # complete + identity match -> done
    wrong_seed = dict(complete, seed=cell["seed"] + 1)
    with open(lm_result_path(tmpdir, cell), "w") as f:
        json.dump(wrong_seed, f)
    ok5d = is_done_lm_cell(tmpdir, cell) is False   # identity mismatch -> not done, never silently trusted
    _report("5", ok5a and ok5b and ok5c and ok5d,
            f"no-file={ok5a} partial={ok5b} complete={ok5c} identity-mismatch={ok5d}")

    print("\n[6] cell_configs() manifest sanity")
    screening = cell_configs(SCREENING_N_SEEDS)
    escalation = cell_configs(ESCALATION_N_SEEDS)
    names_screening = {c["name"] for c in screening}
    baseline_present = any(is_baseline(c) for c in screening)
    ok6 = (len(screening) == SCREENING_CELLS and len(escalation) == ESCALATION_CELLS
           and len(names_screening) == SCREENING_CELLS and baseline_present)
    _report("6", ok6, f"screening={len(screening)} escalation={len(escalation)} "
                       f"unique_names={len(names_screening)} baseline_present={baseline_present}")

    print("\n[7] budget_guard hard-abort has teeth: (a) the REAL registered ceilings must pass "
          "cleanly, (b) a genuinely over-ceiling wave must raise SystemExit -- run TO COMPLETION "
          "via a real monkeypatched ceiling, not asserted from a tautological comparison "
          "(CLAUDE.md: negative tests must actually fire, not just be written).")
    ok7a = True
    try:
        budget_guard(SCREENING_N_SEEDS)
        budget_guard(ESCALATION_N_SEEDS)
    except SystemExit:
        ok7a = False
    module = sys.modules[__name__]
    _orig_screening_ceiling = module.SCREENING_CEILING_GPU_H
    module.SCREENING_CEILING_GPU_H = 1e-6   # artificially tiny -- MUST trigger the abort
    ok7b = False
    try:
        budget_guard(SCREENING_N_SEEDS)
    except SystemExit:
        ok7b = True
    finally:
        module.SCREENING_CEILING_GPU_H = _orig_screening_ceiling
    ok7c = True   # restore verified: the (now-restored) real ceiling must pass again
    try:
        budget_guard(SCREENING_N_SEEDS)
    except SystemExit:
        ok7c = False
    _report("7", ok7a and ok7b and ok7c,
            f"real-ceilings-pass={ok7a} tiny-ceiling-aborts={ok7b} restored-ceiling-passes={ok7c}")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  SMOKE FAILED: {FAILURES}")
        print("=" * 60)
        raise SystemExit(1)
    print("  ALL ATTRACTOR_ROBUSTNESS_2X2 SMOKE CHECKS PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the manifest/commands, launch nothing")
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results", "attractor_robustness_2x2"))
    ap.add_argument("--data-dir", type=str, default="/data/deltanet_rd_data")
    ap.add_argument("--n-seeds", type=int, default=SCREENING_N_SEEDS, choices=[SCREENING_N_SEEDS, ESCALATION_N_SEEDS])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--cell-timeout-s", type=float, default=3600.0,
                     help="per-cell subprocess safety-net timeout (NOT the primary budget control "
                          "-- budget_guard's GPU-h ceiling is).")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    if not args.run:
        ap.print_help()
        return

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        print("WARNING: no CUDA available -- lm_pretrain_rd.py / reasoning_link_probe.py's own "
              "kernels have no CPU path for a REAL run; this will fail past the launch commands "
              "unless a stub is force-installed. Use --smoke for CPU-only verification.",
              file=sys.stderr)

    t0 = time.time()
    report = run_wave(args.out_dir, args.data_dir, args.n_seeds, device, args.cell_timeout_s,
                       dry_run=args.dry_run)
    report["wall_s"] = time.time() - t0
    print("\n" + "=" * 70)
    print("ATTRACTOR-ROBUSTNESS 2x2 WAVE REPORT:", json.dumps(report, indent=2, default=str))
    print("=" * 70)
    if report.get("failed"):
        print(f"\n{len(report['failed'])} (cell, leg) failures -- see per-cell logs in {args.out_dir}",
              file=sys.stderr)
        raise SystemExit(1)
    if report.get("escalation", {}).get("fire"):
        print(f"\nESCALATION RULE FIRED -- qualitative split detected, n=3 escalation "
              f"(ESCALATION_CELLS={ESCALATION_CELLS}, ceiling={ESCALATION_CEILING_GPU_H} GPU-h) "
              f"is now pre-registered-authorized. Re-invoke with --n-seeds {ESCALATION_N_SEEDS}.")


if __name__ == "__main__":
    main()
