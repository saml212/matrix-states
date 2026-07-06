"""frozen_bias_lm_sweep.py -- FROZEN_BIAS_LM_DESIGN.md (ROUND-5, RUNG-1-ONLY
descope, sec 6.1/sec 8.5.1) manifest + dispatch + supervisor + budget-guard
launcher for the frozen-bias LM program. Mirrors run_lm_rd_trackc_sweep.py's
own conventions VERBATIM (module docstring's own pattern list: manifest/
is_done/build_cmd, calibration-first discipline, BATCH_SIZE_BY_RUNG,
disk-space gate, budget guard, tmux+supervisor for real launches) -- cloned,
not cross-imported (this codebase's own pod-safety convention, restated in
every sibling sweep script's docstring).

RUNG-1-ONLY (sec 6.2/sec 8.1/sec 8.5.1): rung-2 is PARKED this wave. This
file builds and gates rung-1's manifest ONLY; there is no rung-2 code path
here to accidentally launch (unlike run_lm_rd_trackc_sweep.py's own
--wave {2,3}, which exist but are separately gated -- this design's own
rung-2 requires a FRESH DESIGN REVIEW after rung-1 completes, sec 6.2's PARK
clause, so no --wave 2 flag is wired at all; adding one is future work for
that fresh review, not a silent placeholder here).

Rung-1 manifest (sec 6.1's table, sec 8.5.1's operative budget):
  - Core 3-arm comparison: Arm 1 (off), Arm 2 (per_token, lambda=0.58),
    Arm 2' (global, lambda=0.58) x 2 corpora (openr1-mix-ext, wikitext-mix-ext)
    x 3 seeds (0,1,2) = 18 training runs.
  - lambda mini-sweep (sec 5): Arm 2 (per_token) @ lambda in {0.3, 0.8},
    1 seed (0), 1 corpus (openr1-mix-ext) = 2 training runs (lambda=0.58 is
    already covered by the core row above -- NOT double-counted).
  - Mandatory training total: 20 runs (matches sec 6.1's table exactly).
  - Calibration cell (sec 6.3): the FIRST of the 20 (Arm 2, openr1-mix-ext,
    seed 0, lambda=0.58) -- called out for schedule visibility, not a
    21st run. gate_and_run_rung1's own launch discipline enforces
    "calibration cell alone first, THEN the remaining 19" mechanically
    (--calibration-only / the two-phase launch below), mirroring sec 6.3's
    own "launch Arm 2 alone first; inspect; only then launch the remaining
    19" text.
  - Arm 1'/Arm 1''/pre-blend-k_raw eval-only passes (sec 6.1's second table,
    26 passes) are NOT built here -- lm_attractor_probe_rd.py-style tooling
    (a SEPARATE, near-zero-GPU-h eval-only measurement pass over Arm 1's own
    checkpoints, sec 8.0a) is this program's OWN dedicated retrofit script,
    not a training cell this sweep dispatches. See frozen_bias_retrofit_
    eval_rd.py (built alongside this file).

sec 8.2a's contention gate (mirrored from KEY_ANCHORING_DESIGN.md sec
12.2.3, restated here as this program's own operative check): the
calibration cell (this file's --calibration-only launch) REFUSES to start
unless the K-anchoring wave's own Stage-1 clearance sentinel exists at
STAGE1_RATES_OK_DEFAULT (a parameter, --stage1-sentinel, default
"results/deltanet_rd_exactness/wavekeyanchor-cliff/STAGE1_RATES_OK" as seen
from this box's own deltanet_rd dir) -- a MECHANICAL check (os.path.exists,
never an elapsed-wall-clock guess), loud refusal (sys.exit, the design's own
registered remedy printed), with an explicit --accept-contention-override
escape hatch (store_true, prints a loud WARNING, never a silent default).

Usage (GPU list is an example -- check nvidia-smi FIRST, per house rule):
  python frozen_bias_lm_sweep.py --dry-run
  python frozen_bias_lm_sweep.py --wave rung1 --calibration-only \\
      --out-dir results/frozen_bias_lm --gpus 1 --gpu-offset 2
  python frozen_bias_lm_sweep.py --wave rung1 --out-dir results/frozen_bias_lm \\
      --gpus 6 --gpu-offset 2    # the remaining 19 cells, gated on the calibration cell's
                                  # own inspected result (sec 6.3) -- NOT auto-chained.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAIN = os.path.join(HERE, "lm_pretrain_rd.py")

# ---------------------------------------------------------------------------
# sec 5's architecture (Wave C's own config, reused verbatim): d_model=256,
# d_state=64, n_layers=2, conv_size=4, num_heads=1 -- 14,048,896 trainable
# params (verified SCALE_TRANSFER_DESIGN.md sec 5.9/5.10, this design's own
# sec 6.1).
# ---------------------------------------------------------------------------
RUNG1_CFG = {"d_model": 256, "d_state": 64, "n_layers": 2, "conv_size": 4, "num_heads": 1}
VOCAB_SIZE = 50257
SEQ_LEN = 512
BATCH_SIZE = 32   # sec 5's architecture reuses Wave C's own control-cell batch size unchanged --
                   # this program never registers a different rung-1 batch size anywhere.

CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")     # sec 5/sec 6.1's registered corpora
SEEDS = (0, 1, 2)                                     # sec 6.0's registered seed integers

LAMBDA_PRIMARY = 0.58
LAMBDA_MINI_SWEEP = (0.3, 0.8)                        # sec 5's mini-sweep brackets (0.58 already
                                                       # covered by the core-18 row)
MINI_SWEEP_CORPUS = "openr1-mix-ext"                  # sec 5: "1 seed, 1 corpus (openr1-mix-ext)"
MINI_SWEEP_SEED = 0

# sec 8.5.1's rung-1-only budget: 20 mandatory training runs, 2x contingency,
# ~11.20 GPU-h mandatory / ~14.22 GPU-h incl. optional Arm 3. PROGRAM_SPENT_GPUH
# is a MAINTAINED constant (mirrors run_lm_rd_trackc_sweep.py's own convention) --
# a human/orchestrator updates it as real spend accrues. Starts at 0.0: this
# program has not yet spent any real GPU-h (CPU-only build session, no launch).
PROGRAM_SPENT_GPUH = 0.0
GPU_H_PROGRAM_CEILING = 135.0   # sec 8.1's ceiling (unchanged across rounds -- a headroom cap,
                                 # NOT a target; rung-1-only committed spend is ~11.20-14.22 GPU-h,
                                 # sec 8.5.1, well inside this ceiling).

# House convention (CLAUDE.md / run_lm_rd_trackc_sweep.py's own LAUNCH_TIMEOUT_MARGIN): a real
# launch's timeout is the measured cost times this margin, never the raw measured cost.
LAUNCH_TIMEOUT_MARGIN = 1.6

# sec 8.2a's contention gate default path, AS SEEN FROM THE BOX'S deltanet_rd DIR (a parameter,
# not a hardcoded absolute path, per the task's own instruction) -- the K-anchoring wave's own
# Stage-1 clearance sentinel (keyanchor_cliff_stage_gate's STAGE1_RATES_OK, run_deltanet_rd_
# exactness_sweep.py / keyanchor_cliff_chain.sh).
STAGE1_SENTINEL_DEFAULT = "results/deltanet_rd_exactness/wavekeyanchor-cliff/STAGE1_RATES_OK"

DISK_SAFETY_FACTOR = 1.5   # mirrors run_lm_rd_trackc_sweep.py's own DISK_SAFETY_FACTOR exactly


# ---------------------------------------------------------------------------
# sec 8.2a contention gate (mirrored from KEY_ANCHORING_DESIGN.md sec 12.2.3's own registered
# ordering constraint -- "the frozen-bias LM program's own calibration cell must not launch until
# this wave's Stage-1 rates are observed within bracket"). Mechanical: os.path.exists, never an
# elapsed-wall-clock guess. --accept-contention-override is a loud, explicit human escape hatch
# (store_true, never a silent default), mirroring budget_guard's own override discipline below.
# ---------------------------------------------------------------------------

def contention_gate(stage1_sentinel_path: str, accept_override: bool) -> None:
    exists = os.path.exists(stage1_sentinel_path)
    print(f"CONTENTION GATE (sec 8.2a, mirrors KEY_ANCHORING_DESIGN.md sec 12.2.3): "
          f"K-anchoring wave's Stage-1 clearance sentinel at {stage1_sentinel_path!r} "
          f"{'FOUND' if exists else 'NOT FOUND'}.", flush=True)
    if exists:
        return
    if accept_override:
        print("WARNING: --accept-contention-override passed -- proceeding WITHOUT the K-anchoring "
              "wave's Stage-1 clearance sentinel. This means this program's own calibration cell's "
              "realized rate may be measured under COMPOUNDED CONTENTION (both multi-cell programs "
              "sharing GPUs 2-7 simultaneously) rather than the single-program conditions sec 8.2a's "
              "2x contingency multiplier is calibrated against. A human decision, logged here "
              "explicitly, never a silent default.", flush=True)
        return
    print("=" * 70, file=sys.stderr)
    print(f"ERROR: sec 8.2a's contention gate REFUSES this launch -- "
          f"{stage1_sentinel_path!r} does not exist.", file=sys.stderr)
    print("This program's own calibration cell (sec 6.3) must not start until the K-anchoring "
          "wave's own Stage-1 cells (K=38/K=42) have their realized wall_s/GPU-h rates observed "
          "within the K48-calibrated bracket (keyanchor_cliff_stage_gate's own STAGE1_RATES_OK "
          "sentinel) -- both programs share GPUs 2-7, and this design's own 2x contingency "
          "multiplier is calibrated against a single-sibling precedent, not compounded contention "
          "from two simultaneous multi-cell programs.", file=sys.stderr)
    print("REMEDY: wait for the K-anchoring wave's own Stage-1 completion (or its own re-pricing "
          "if it overran), then re-invoke this launch. Or pass --accept-contention-override to "
          "proceed anyway (a human decision, loud warning printed, never a silent default).",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.exit(4)


# ---------------------------------------------------------------------------
# Manifest builders. ONE spec shape covers every cell (arm/lambda/corpus/seed differ) -- is_done_
# cell/build_cmd_cell are shape-identical to run_lm_rd_trackc_sweep.py's own is_done_cell/
# build_cmd_cell pattern, generalized over frozen_bias_arm/frozen_bias_lambda.
# ---------------------------------------------------------------------------

def cell_name(arm: str, lam: float, corpus: str, seed: int) -> str:
    lam_tag = f"{lam:.2f}".replace(".", "p")
    cfg = RUNG1_CFG
    return (f"frozenbias_lm_{arm}_lam{lam_tag}_{corpus}_dm{cfg['d_model']}_ds{cfg['d_state']}"
            f"_L{cfg['n_layers']}_s{seed}")


def _spec(arm: str, lam: float, corpus: str, seed: int, steps: int, ckpt_every: int) -> dict:
    assert arm in ("off", "per_token", "global"), f"unknown arm {arm!r}"
    return {
        "arm": arm, "lam": lam, "corpus": corpus, "seed": seed,
        "d_model": RUNG1_CFG["d_model"], "d_state": RUNG1_CFG["d_state"], "n_layers": RUNG1_CFG["n_layers"],
        "conv_size": RUNG1_CFG["conv_size"], "num_heads": RUNG1_CFG["num_heads"],
        "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "steps": steps, "ckpt_every": ckpt_every,
        "name": cell_name(arm, lam, corpus, seed),
    }


def core_manifest(steps: int, ckpt_every: int) -> list[dict]:
    """sec 6.1's core 3-arm comparison: Arm 1 (off), Arm 2 (per_token @ 0.58), Arm 2' (global @
    0.58) x 2 corpora x 3 seeds = 18 runs. Arm 1's `lam` field is a placeholder (0.0, unused --
    apply_frozen_bias_blend's own "off" branch never reads it) recorded for manifest-shape
    consistency, not because Arm 1 has a meaningful lambda."""
    runs = []
    for arm, lam in (("off", 0.0), ("per_token", LAMBDA_PRIMARY), ("global", LAMBDA_PRIMARY)):
        for corpus in CORPORA:
            for seed in SEEDS:
                runs.append(_spec(arm, lam, corpus, seed, steps, ckpt_every))
    return runs


def mini_sweep_manifest(steps: int, ckpt_every: int) -> list[dict]:
    """sec 5's mandatory lambda mini-sweep: Arm 2 (per_token) @ lambda in {0.3, 0.8}, seed 0,
    openr1-mix-ext only -- 2 runs. lambda=0.58 is already in core_manifest's own per_token row at
    this same (corpus, seed) -- NOT reconstructed here (would double-count)."""
    return [_spec("per_token", lam, MINI_SWEEP_CORPUS, MINI_SWEEP_SEED, steps, ckpt_every)
            for lam in LAMBDA_MINI_SWEEP]


def rung1_manifest(steps: int, ckpt_every: int = 1000) -> list[dict]:
    """sec 6.1's FULL rung-1 mandatory manifest: 18 (core) + 2 (mini-sweep) = 20 training runs,
    matching sec 6.1's table exactly. Shared by --dry-run's preview and the real launch (this
    codebase's own preview/launch-parity discipline, run_lm_rd_trackc_sweep.py's wave1_manifest
    precedent)."""
    return core_manifest(steps, ckpt_every) + mini_sweep_manifest(steps, ckpt_every)


def calibration_cell(steps: int, ckpt_every: int = 1000) -> dict:
    """sec 6.3's mandatory calibration cell: Arm 2 (per_token), corpus=openr1-mix-ext, seed=0,
    lambda=0.58 -- THE FIRST of the 20 mandatory cells (core_manifest's own per_token/openr1-mix-
    ext/seed-0 row), not a 21st run. Returned here so --calibration-only can launch this ONE cell
    in isolation, matching sec 6.3's "launch Arm 2 alone first" text exactly."""
    return _spec("per_token", LAMBDA_PRIMARY, "openr1-mix-ext", 0, steps, ckpt_every)


def is_done_cell(out_dir: str, spec: dict) -> bool:
    """Identity check keyed on EVERY field that varies the trained artifact (arm/lambda/corpus/
    seed/architecture/steps/batch_size) -- mirrors run_lm_rd_trackc_sweep.py's own is_done_cell
    discipline (re-derives identity from the result JSON's own recorded fields, never trusts the
    filename alone)."""
    p = os.path.join(out_dir, f"{spec['name']}.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict) or d.get("complete") is not True or d.get("timed_out"):
            return False
        fb = d.get("frozen_bias") or {}
        required_top = ("corpus", "seed", "d_model", "d_state", "n_layers", "seq_len",
                         "steps", "steps_completed", "batch_size")
        if not all(k in d for k in required_top):
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if (d.get("corpus") != spec["corpus"] or d.get("seed") != spec["seed"]
                or d.get("d_model") != spec["d_model"] or d.get("d_state") != spec["d_state"]
                or d.get("n_layers") != spec["n_layers"] or d.get("seq_len") != spec["seq_len"]
                or d.get("steps") != spec["steps"] or d.get("batch_size") != spec["batch_size"]
                or fb.get("arm") != spec["arm"]):
            return False
        if spec["arm"] != "off" and abs((fb.get("lambda") or -1.0) - spec["lam"]) > 1e-9:
            return False
        return True
    except Exception:
        return False


def build_cmd_cell(spec: dict, out_dir: str, timeout: float, data_dir: str,
                    ckpt_base_dir: str | None = None) -> list[str]:
    """ckpt_base_dir: the standing "checkpoints to /data, never container disk" rule (CLAUDE.md /
    this codebase's own HF-cache-symlink lesson) -- mirrors run_deltanet_rd_exactness_sweep.py's
    own --ckpt-base-dir convention (a SEPARATE path from --out-dir, which stays repo-relative for
    small result JSONs). Defaults to out_dir/checkpoints ONLY as a --dry-run-safe fallback; a real
    launch (frozen_bias_chain.sh) always passes an explicit /data-rooted ckpt_base_dir."""
    ckpt_dir = os.path.join(ckpt_base_dir, spec["name"]) if ckpt_base_dir else os.path.join(out_dir, "checkpoints")
    cmd = [sys.executable, PRETRAIN,
           "--corpus", spec["corpus"], "--data-dir", data_dir,
           "--d-model", str(spec["d_model"]), "--d-state", str(spec["d_state"]),
           "--n-layers", str(spec["n_layers"]), "--conv-size", str(spec["conv_size"]),
           "--num-heads", str(spec["num_heads"]), "--seq-len", str(spec["seq_len"]),
           "--batch-size", str(spec["batch_size"]), "--steps", str(spec["steps"]),
           "--ckpt-every", str(spec["ckpt_every"]), "--seed", str(spec["seed"]),
           "--internal-timeout", str(max(1, int(timeout - 30))),
           "--ckpt-dir", ckpt_dir,
           "--out", os.path.join(out_dir, f"{spec['name']}.json"),
           "--frozen-bias-arm", spec["arm"]]
    if spec["arm"] != "off":
        cmd += ["--frozen-bias-lambda", str(spec["lam"])]
    return cmd


# ---------------------------------------------------------------------------
# Budget guard + disk-space gate. Mirrors run_lm_rd_trackc_sweep.py's own budget_guard/
# disk_space_check functions (cloned, not cross-imported, per this codebase's own pod-safety
# convention).
# ---------------------------------------------------------------------------

def projected_gpu_hours(manifest: list[dict], per_step_s: float, per_ckpt_s: float) -> float:
    total_s = 0.0
    for spec in manifest:
        n_ckpts = spec["steps"] // spec["ckpt_every"] + 1
        total_s += per_step_s * spec["steps"] + per_ckpt_s * n_ckpts
    return total_s / 3600.0


def budget_guard(projected_gpu_h: float, label: str, accept_override: bool) -> float:
    cumulative = PROGRAM_SPENT_GPUH + projected_gpu_h
    print(f"BUDGET GUARD ({label}): program-spent-so-far={PROGRAM_SPENT_GPUH:.2f} GPU-h + "
          f"this-wave-projected={projected_gpu_h:.2f} GPU-h = cumulative {cumulative:.2f} GPU-h, "
          f"ceiling {GPU_H_PROGRAM_CEILING:.0f} GPU-h (sec 8.1/sec 8.5.1).", flush=True)
    if cumulative > GPU_H_PROGRAM_CEILING and not accept_override:
        print(f"ERROR: projected cumulative spend {cumulative:.2f} GPU-h EXCEEDS the "
              f"{GPU_H_PROGRAM_CEILING:.0f} GPU-h program ceiling -- REFUSING to launch {label}. "
              f"Pass --accept-budget-override to force past this guard (a human decision, never a "
              f"default).", file=sys.stderr)
        sys.exit(5)
    return cumulative


def find_ckpt_size_bytes(ckpt_dir: str, d_model: int, d_state: int, n_layers: int) -> int:
    """Walks `ckpt_dir` RECURSIVELY (not just its top level) -- this program's own per-cell
    checkpoint layout is `ckpt_base_dir/<cell_name>/<run_name>_step<n>.pt` (mirrors
    run_deltanet_rd_exactness_sweep.py's own --ckpt-base-dir/<spec-name> convention), a nested
    layout the flat run_lm_rd_trackc_sweep.py precedent (one shared checkpoints/ dir) does not
    have -- a top-level-only os.listdir would silently never find anything and always raise."""
    needle = f"_dm{d_model}_ds{d_state}_L{n_layers}_"
    if os.path.isdir(ckpt_dir):
        for root, _dirs, files in os.walk(ckpt_dir):
            for fname in sorted(files):
                if needle in fname and fname.endswith(".pt"):
                    return os.path.getsize(os.path.join(root, fname))
    raise FileNotFoundError(
        f"no existing checkpoint matching architecture dm{d_model}/ds{d_state}/L{n_layers} found "
        f"under {ckpt_dir!r} -- the disk-space gate needs a REAL measured checkpoint size; run the "
        f"calibration cell first.")


def disk_space_check(ckpt_dir: str, projected_bytes: int, label: str,
                      safety_factor: float = DISK_SAFETY_FACTOR) -> dict:
    import shutil
    resolved = os.path.realpath(ckpt_dir)
    free_bytes = shutil.disk_usage(resolved).free if os.path.exists(resolved) else 0
    required_bytes = int(projected_bytes * safety_factor)
    return {
        "label": label, "resolved_ckpt_dir": resolved, "projected_ckpt_bytes": projected_bytes,
        "safety_factor": safety_factor, "required_bytes": required_bytes, "free_bytes": free_bytes,
        "ok": os.path.exists(resolved) and free_bytes >= required_bytes,
    }


def default_timeout_pretrain(steps: int, ckpt_every: int, per_step_s: float, per_ckpt_s: float,
                              margin: float = LAUNCH_TIMEOUT_MARGIN,
                              startup_overhead_s: float = 30.0) -> int:
    n_ckpts = steps // ckpt_every + 1
    base = (startup_overhead_s + per_step_s * steps + n_ckpts * per_ckpt_s) * margin
    return int(base)


# ---------------------------------------------------------------------------
# Dispatch/supervisor -- mirrors run_lm_rd_trackc_sweep.py's own _run_wave (exception-isolated
# launch, per-run timeout with GPU quarantine, guarded aggregate, try/except per config so one
# crash doesn't kill the remaining manifest).
# ---------------------------------------------------------------------------

def run_smoke(log_dir: str, gpu: int) -> bool:
    print(f"SMOKE GATE (physical GPU {gpu}) -- lm_pretrain_rd.py --smoke", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with open(os.path.join(log_dir, "smoke_pretrain.log"), "w") as lf:
        rc = subprocess.call([sys.executable, PRETRAIN, "--smoke"], env=env,
                              stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    print(f"  smoke[pretrain]: {'PASS' if rc == 0 else f'FAIL (rc={rc})'}", flush=True)
    return rc == 0


def _run_manifest(label: str, manifest: list[dict], out_dir: str, args) -> bool:
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir, args.gpu_offset):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate failed; no training launched.\n")
        return False

    physical_gpus = list(range(args.gpu_offset, args.gpu_offset + args.gpus))
    slots = [g for _ in range(args.per_gpu) for g in physical_gpus]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done_cell(out_dir, s)]
    already_done_ct = len(manifest) - len(pending)
    print(f"label={label}  manifest={len(manifest)}  pending={len(pending)} "
          f"(already-done={already_done_ct})  slots={n_slots} "
          f"(gpus {physical_gpus} x {args.per_gpu} per-gpu)", flush=True)

    running, free, quarantined = {}, list(slots), []
    done_ct, failed, uid = 0, [], 0

    while pending or running:
        while free and pending:
            gpu = free.pop()
            spec = pending.pop(0)
            timeout = args.timeout if args.timeout is not None else args.timeout_fn(spec)
            try:
                lf = open(os.path.join(log_dir, f"{spec['name']}.log"), "w")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                cmd = build_cmd_cell(spec, out_dir, timeout, args.data_dir,
                                      ckpt_base_dir=getattr(args, "ckpt_base_dir", None))
                proc = subprocess.Popen(cmd, env=env,
                                         stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
                running[uid] = (proc, spec, lf, time.time(), gpu, timeout)
                uid += 1
            except Exception as e:   # noqa: BLE001 -- one launch failure must not kill the manifest
                print(f"  LAUNCH-FAILED {spec['name']}: {e!r}", flush=True)
                failed.append((spec["name"], "LAUNCH_ERR"))
                free.append(gpu)
        for u, (proc, spec, lf, start, gpu, timeout) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                if time.time() - start > timeout:
                    proc.kill()
                    try:
                        proc.wait(timeout=30)
                    except Exception:   # noqa: BLE001
                        pass
                    try:
                        lf.close()
                    except Exception:   # noqa: BLE001
                        pass
                    print(f"  TIMEOUT {spec['name']} (>{timeout:.0f}s) -- GPU {gpu} quarantined", flush=True)
                    failed.append((spec["name"], "TIMEOUT"))
                    quarantined.append(gpu)
                    del running[u]
                continue
            try:
                lf.close()
            except Exception:   # noqa: BLE001
                pass
            if rc == 0 and is_done_cell(out_dir, spec):
                done_ct += 1
                print(f"  DONE {spec['name']}  ({done_ct + already_done_ct}/{len(manifest)})", flush=True)
            else:
                print(f"  FAILED {spec['name']} (rc={rc})", flush=True)
                failed.append((spec["name"], f"rc={rc}"))
            if gpu not in quarantined:
                free.append(gpu)
            del running[u]
        try:
            with open(os.path.join(out_dir, "PROGRESS.txt"), "w") as f:
                f.write(f"label={label} done={done_ct + already_done_ct} failed={len(failed)} "
                        f"running={len(running)}\n")
        except Exception:   # noqa: BLE001
            pass
        if pending or running:
            time.sleep(5)

    all_done = (done_ct + already_done_ct) == len(manifest)
    report = {"label": label, "n_manifest": len(manifest), "n_failed": len(failed),
              "failed": failed, "all_done": all_done}
    try:
        with open(os.path.join(out_dir, f"{label}_summary.json"), "w") as f:
            json.dump(report, f, indent=2)
    except Exception as e:   # noqa: BLE001
        print(f"  (writing {label}_summary.json non-fatal failure: {e!r})", flush=True)
    print(f"label={label} COMPLETE: {report}", flush=True)
    return all_done


# ---------------------------------------------------------------------------
# Gate + launch for --wave rung1
# ---------------------------------------------------------------------------

def gate_and_run_rung1(args) -> None:
    calibration_json_path = os.path.join(args.out_dir, "calibration.json")
    steps = args.rung1_steps

    if steps is None:
        print("ERROR: --rung1-steps is REQUIRED for a real --wave rung1 launch (no silent "
              "fallback to a placeholder) -- run the calibration cell first (--calibration-only) "
              "and derive a real step count from its measured tok/s.", file=sys.stderr)
        sys.exit(2)

    if args.calibration_only:
        # sec 8.2a's contention gate is checked HERE (the calibration cell IS the item sec 8.2a
        # gates) -- never for the remaining-19 launch (that launch's own gate is "the calibration
        # cell already completed and was inspected", sec 6.3, a human judgment this script does
        # not automate away).
        contention_gate(args.stage1_sentinel, args.accept_contention_override)
        manifest = [calibration_cell(steps, args.ckpt_every)]
        label = "rung1_calibration"
    else:
        if not os.path.exists(calibration_json_path) and not args.skip_calibration_check:
            print(f"ERROR: {calibration_json_path} not found and the calibration cell's own result "
                  f"JSON was not found either -- sec 6.3's non-negotiable rule: 'one full Arm-2 "
                  f"training run at rung-1's target step count must complete and be inspected... "
                  f"BEFORE the remaining rung-1 cells launch.' Run --calibration-only first, inspect "
                  f"its span_frac trajectory and val-loss curve, THEN launch the remaining 19 cells. "
                  f"(Pass --skip-calibration-check to override -- a human decision, never a default, "
                  f"e.g. for a --dry-run-style rehearsal.)", file=sys.stderr)
            sys.exit(2)
        calib_spec = calibration_cell(steps, args.ckpt_every)
        if not is_done_cell(args.out_dir, calib_spec) and not args.skip_calibration_check:
            print(f"ERROR: the calibration cell ({calib_spec['name']}) has not completed under "
                  f"{args.out_dir!r} -- sec 6.3's non-negotiable rule (see above). Run "
                  f"--calibration-only first.", file=sys.stderr)
            sys.exit(2)
        full = rung1_manifest(steps, args.ckpt_every)
        manifest = [s for s in full if s["name"] != calib_spec["name"]]
        label = "rung1_remaining19"

    # Timing constants: read from calibration.json if present (real measured numbers), else refuse
    # a real (non-calibration-only) launch without an explicit override.
    per_step_s = per_ckpt_s = None
    if os.path.exists(calibration_json_path):
        with open(calibration_json_path) as f:
            calib = json.load(f)
        tc = (calib.get("timing_constants") or {}).get("rung1")
        if tc:
            per_step_s, per_ckpt_s = tc["per_step_s"], tc["per_ckpt_s"]
    if per_step_s is None:
        if args.calibration_only:
            # Generous, documented planning ceiling for the calibration cell's OWN timeout only
            # (chicken-and-egg: no real numbers exist yet) -- mirrors run_lm_rd_trackc_sweep.py's
            # own _CALIBRATION_PER_STEP_S_GENEROUS discipline (5x headroom over Wave C's own
            # measured ~0.037s/step at this exact 14M architecture).
            per_step_s, per_ckpt_s = 0.25, 60.0
        else:
            print("ERROR: no timing_constants['rung1'] found in calibration.json -- cannot size a "
                  "real launch's timeout/budget projection without the calibration cell's own "
                  "measured numbers. Run --calibration-only first and record its wall_s/steps.",
                  file=sys.stderr)
            sys.exit(2)

    projected = projected_gpu_hours(manifest, per_step_s, per_ckpt_s)
    budget_guard(projected, label, args.accept_budget_override)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = args.ckpt_base_dir or os.path.join(args.out_dir, "checkpoints")
    if os.path.isdir(ckpt_dir) and any(True for _r, _d, files in os.walk(ckpt_dir) for f in files if f.endswith(".pt")):
        try:
            ckpt_size = find_ckpt_size_bytes(ckpt_dir, RUNG1_CFG["d_model"], RUNG1_CFG["d_state"],
                                              RUNG1_CFG["n_layers"])
            n_ckpts_per_run = manifest[0]["steps"] // manifest[0]["ckpt_every"] + 1
            projected_bytes = len(manifest) * n_ckpts_per_run * ckpt_size
            disk_report = disk_space_check(ckpt_dir, projected_bytes, label)
            print(f"Disk-space check: {disk_report}", flush=True)
            if not disk_report["ok"]:
                print(f"ERROR: disk-space check FAILED -- refusing to launch (need "
                      f"{disk_report['required_bytes']:,} bytes free at {DISK_SAFETY_FACTOR}x "
                      f"safety margin, have {disk_report['free_bytes']:,} under "
                      f"{disk_report['resolved_ckpt_dir']}).", file=sys.stderr)
                sys.exit(7)
        except FileNotFoundError as e:
            print(f"NOTE: disk-space check skipped (no existing checkpoint to measure size from "
                  f"yet): {e}", flush=True)
    else:
        print("NOTE: disk-space check skipped -- no existing checkpoint directory yet (first "
              "launch); it will be checked automatically once the calibration cell's own "
              "checkpoint exists on subsequent invocations.", flush=True)

    args.timeout_fn = lambda spec: default_timeout_pretrain(spec["steps"], spec["ckpt_every"],
                                                             per_step_s, per_ckpt_s)
    print(f"RUNG-1 REAL LAUNCH ({label}): {len(manifest)} runs, timing "
          f"per_step_s={per_step_s} per_ckpt_s={per_ckpt_s} (margin {LAUNCH_TIMEOUT_MARGIN}x).",
          flush=True)
    all_done = _run_manifest(label, manifest, args.out_dir, args)
    sys.exit(0 if all_done else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wave", choices=["rung1"], default=None)
    ap.add_argument("--calibration-only", action="store_true",
                     help="sec 6.3: launch ONLY the calibration cell (Arm 2, openr1-mix-ext, "
                          "seed 0, lambda=0.58) -- inspect its span_frac trajectory/val-loss curve "
                          "before launching the remaining 19 cells.")
    ap.add_argument("--skip-calibration-check", action="store_true",
                     help="override sec 6.3's calibration-before-remaining-19 gate (a human "
                          "decision, never a default) -- e.g. for a --dry-run-style rehearsal.")
    ap.add_argument("--rung1-steps", type=int, default=None,
                     help="REQUIRED for a real launch -- no silent fallback to a placeholder.")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/frozen_bias_lm"))
    ap.add_argument("--ckpt-base-dir", default=None,
                     help="standing rule: checkpoints to /data, never container disk. Mirrors "
                          "run_deltanet_rd_exactness_sweep.py's own --ckpt-base-dir convention -- "
                          "a SEPARATE path from --out-dir (which stays repo-relative for small "
                          "result JSONs). Defaults to <out-dir>/checkpoints ONLY as a --dry-run-"
                          "safe fallback; a real launch (frozen_bias_chain.sh) always passes an "
                          "explicit /data-rooted path.")
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--gpu-offset", type=int, default=None)
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--accept-budget-override", action="store_true")
    ap.add_argument("--accept-contention-override", action="store_true",
                     help="sec 8.2a: bypass the K-anchoring-wave contention gate (a human "
                          "decision, loud warning printed, never a silent default).")
    ap.add_argument("--stage1-sentinel", default=STAGE1_SENTINEL_DEFAULT,
                     help="path (as seen from this box's deltanet_rd dir) to the K-anchoring "
                          "wave's Stage-1 clearance sentinel, sec 8.2a.")
    args = ap.parse_args()

    if args.dry_run:
        preview_steps = args.rung1_steps or 20000   # sec 8.3's own illustrative planning number
        full = rung1_manifest(preview_steps, args.ckpt_every)
        calib = calibration_cell(preview_steps, args.ckpt_every)
        print(f"DRY RUN -- rung-1 manifest preview at steps={preview_steps} (illustrative if "
              f"--rung1-steps not given):")
        print(f"  core (Arm1/Arm2/Arm2' x {len(CORPORA)} corpora x {len(SEEDS)} seeds): "
              f"{len(core_manifest(preview_steps, args.ckpt_every))}")
        print(f"  lambda mini-sweep ({LAMBDA_MINI_SWEEP}, 1 seed, 1 corpus): "
              f"{len(mini_sweep_manifest(preview_steps, args.ckpt_every))}")
        print(f"  TOTAL mandatory training cells: {len(full)} (sec 6.1's table: 20)")
        print(f"  calibration cell (already one of the 20): {calib['name']}")
        arms = sorted(set(s["arm"] for s in full))
        seeds_seen = sorted(set(s["seed"] for s in full))
        print(f"  arms present: {arms}")
        print(f"  seeds present: {seeds_seen}")
        for s in full:
            print(f"    {s['name']}")
        return

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for any real (non-dry-run) invocation "
              "(this codebase's own no-silent-default-GPU convention).", file=sys.stderr)
        sys.exit(2)

    if args.wave == "rung1":
        gate_and_run_rung1(args)
    else:
        print("ERROR: --wave rung1 (or --dry-run) required.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
