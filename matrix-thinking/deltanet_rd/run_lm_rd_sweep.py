"""run_lm_rd_sweep.py -- bounded, human-gated wave orchestrator for the
Wave 2 (RD-2) instrumented-LM arm: Wave C (real-corpus LM pretrain,
lm_pretrain_rd.py) and Wave D (inference-time rank-truncation intervention,
lm_intervene_rd.py). See DELTANET_REALDATA_DESIGN.md section 7's manifest
table (Waves C/D) and section 8's operational-harness requirements, SCALED
to the probe tier per this build's task brief: 2 corpora x 3 seeds x
~30 min pretrain (the task brief's own budgeted figure) + cheap eval-only
intervention evals, budget approx. 10-15 GPU-h total -- much smaller than
section 7's own ~12-25 + ~5-10 GPU-h full-scale estimate for this wave
(section 7's own numbers are explicitly upper bounds, not targets, per
this project's repeatedly-relearned estimate-calibration lesson).

**MEASURED, not just estimated (see _PER_STEP_S_PLACEHOLDER's comment):
an on-box calibration probe at the exact probe-tier config (2026-07-03,
GPU 7) found ~438K tokens/s/GPU -- a 100M-token Wave-C run completes in
~5.5 minutes wall-clock, not the ~15-35 min this task brief's own cost
model (derived from section 4.2's conservative 50-200K tok/s/GPU band)
anticipated. At TARGET_TOKENS's current 100M/run default, Wave C's whole
6-run manifest is ~0.5 GPU-h, not ~3+ GPU-h -- the probe tier has
substantial budget HEADROOM left under the 10-15 GPU-h ceiling. This is
recorded here as a measured fact for the audit to weigh (e.g. whether to
raise TARGET_TOKENS closer to section 4.2's own 300-500M Chinchilla-ish
floor before a real launch), not silently acted on by this build phase.**

CLONE of run_deltanet_rd_sweep.py's robustness pattern (smoke gate,
exception-isolated launch, validity-checked resume, per-run timeout with
GPU quarantine, guarded aggregate, REQUIRED --gpus/--gpu-offset with NO
defaults) -- deliberately NOT a cross-directory import (pod-safety
convention).

**This build phase does NOT launch the wave -- it gets an independent
audit first (task brief). This script is complete and smoke-gated, but
`main()` refuses to run without an explicit --wave, exactly like
run_deltanet_rd_sweep.py's own gate, and nothing in this repo invokes it
automatically.**

Wave D depends on Wave C's SAVED checkpoints (lm_pretrain_rd.py's
--ckpt-dir output) -- waveD_manifest() reads Wave C's own result JSONs to
find each (corpus, seed) cell's final checkpoint path, and a cell whose
Wave C run has not completed yet is DROPPED from the pending manifest with
a printed message (not a crash) rather than assumed present.

TODO (documented per audit, deliberately NOT implemented for the probe
tier): PER-RUN mid-training resume. A Wave C run that dies mid-training
restarts from step 0 on the next sweep invocation (is_done() correctly
refuses the partial JSON). At ~5.5 min/run this costs at most one run's
wall time and is not worth the checkpoint/optimizer-state plumbing;
WAVE-level resume (skip already-complete runs, validity-checked) exists
and is the resume that matters at this tier. Revisit if TARGET_TOKENS is
raised to the point where a single run is expensive enough to protect.

Usage (GPU list is an example -- check nvidia-smi first, per house rule):
  python run_lm_rd_sweep.py --dry-run
  python run_lm_rd_sweep.py --wave C --out-dir results/lm_rd --gpus 2 --gpu-offset 6
  python run_lm_rd_sweep.py --wave D --out-dir results/lm_rd --gpus 2 --gpu-offset 6
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
INTERVENE = os.path.join(HERE, "lm_intervene_rd.py")

CORPORA = ("openr1", "wikitext")
SEEDS = (0, 1, 2)

# Probe-tier defaults (task brief: d_model=256, d_state=64, 1-2 layers;
# "~100M tokens ~= 15-35 min/run" -- section 4.2's corrected, head-
# dominated cost model, scaled down from that section's own 400M-token/
# 0.6-2.2h full-scale band). d_model=384 is NOT this run -- that is section
# 4.1's own Wave-2 FULL-SCALE recommendation, not the probe tier this task
# brief asks for.
D_MODEL = 256
D_STATE = 64
N_LAYERS = 2
SEQ_LEN = 512
BATCH_SIZE = 32
TARGET_TOKENS = 100_000_000

# MEASURED per-step wall-clock (s/step), calibrated on-box (youthful-indigo-
# turkey, GPU 7, 2026-07-03): d_model=256/d_state=64/n_layers=2/seq_len=512/
# batch_size=32, fp32 model + bf16-only-at-the-kernel-boundary (this file's
# module docstring / lm_pretrain_rd.py's own build-decision comment). Two
# runs (300 steps w/ 3 checkpoints, 2000 steps w/ 1 checkpoint) solved
# jointly for per-step cost and per-checkpoint overhead: ~0.037 s/step pure
# training, ~15 s/checkpoint (2-corpus val loss + 2-corpus rank-stat
# sampling). This is ~438K tokens/s/GPU -- 2-9x FASTER than
# DELTANET_REALDATA_DESIGN.md section 4.2's own conservative 50-200K
# tokens/s/GPU estimate. ATTRIBUTION CORRECTED (audit, 2026-07-03): an
# earlier revision of this comment credited "PyTorch's default TF32
# fast-path" -- WRONG: torch 2.12.1 on this box defaults
# torch.backends.cuda.matmul.allow_tf32=False (read directly), so the
# measurement ran with TF32 matmuls OFF. The design doc's band was simply
# conservative for this configuration (memory-bound small-d_model ops +
# the bf16 chunk kernel dominate; the head matmul at fp32-non-TF32 is not
# the binding constraint at this scale). lm_pretrain_rd.set_and_log_tf32()
# now SETS the flags explicitly (matmul=False, cudnn=True -- the exact
# state the calibration measured) and logs them into every result JSON, so
# a future torch-default change cannot silently shift numerics or invalidate
# this constant. Kept as a small margin over the raw measurement (0.0374s),
# not the raw number itself, in case a busier box (other concurrent waves
# on GPUs 0-5, as measured at calibration time) slows a real launch down.
_PER_STEP_S_PLACEHOLDER = 0.05
_PER_CHECKPOINT_S_MEASURED = 15.0


def default_steps(batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN, target_tokens: int = TARGET_TOKENS) -> int:
    tokens_per_step = batch_size * seq_len
    return max(1, target_tokens // tokens_per_step)


def default_timeout_pretrain(steps: int, ckpt_every: int, margin: float = 1.6) -> int:
    n_ckpts = steps // ckpt_every + 1
    base = (_PER_STEP_S_PLACEHOLDER * steps + n_ckpts * _PER_CHECKPOINT_S_MEASURED) * margin
    return int(base)


def default_timeout_intervene(n_eval_windows: int, k_grid, margin: float = 1.6) -> int:
    # cheap, eval-only, no backward pass (section 6.2). MEASURED on-box
    # (2026-07-03, GPU 7): a full real intervention run (--n-eval-windows 32,
    # 6-value k-grid, both corpora) completed in ~14s wall (~19s incl. python
    # + tokenizer-cache startup) -- this formula's constants are left
    # deliberately generous (>10x the measurement) rather than tightened to
    # it: a timeout should absorb a busier shared box (see this file's
    # module docstring), not chase the best-case number.
    per_k_s = 3.0
    base = (30.0 + len(k_grid) * per_k_s * (n_eval_windows / 8.0)) * margin * 2   # x2 for two corpora
    return int(base)


# ---------------------------------------------------------------------------
# Wave C -- pretrain manifest
# ---------------------------------------------------------------------------

def _ckpt_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "checkpoints")


def waveC_name(corpus, seed, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS) -> str:
    return f"wC_lm_{corpus}_dm{d_model}_ds{d_state}_L{n_layers}_s{seed}"


def waveC_manifest(steps=None, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS,
                    seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    steps = steps if steps is not None else default_steps(batch_size, seq_len)
    runs = []
    for corpus in CORPORA:
        for seed in SEEDS:
            runs.append({
                "wave": "C", "corpus": corpus, "seed": seed, "d_model": d_model, "d_state": d_state,
                "n_layers": n_layers, "seq_len": seq_len, "batch_size": batch_size, "steps": steps,
                "name": waveC_name(corpus, seed, d_model, d_state, n_layers),
            })
    return runs


def is_done_C(out_dir, spec) -> bool:
    p = os.path.join(out_dir, f"{spec['name']}.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict) or d.get("complete") is not True:
            return False
        if d.get("timed_out"):
            return False
        required = ("corpus", "seed", "d_model", "d_state", "n_layers", "seq_len", "steps", "steps_completed")
        if not all(k in d for k in required):
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if (d.get("corpus") != spec["corpus"] or d.get("seed") != spec["seed"]
                or d.get("d_model") != spec["d_model"] or d.get("d_state") != spec["d_state"]
                or d.get("n_layers") != spec["n_layers"] or d.get("seq_len") != spec["seq_len"]
                or d.get("steps") != spec["steps"]):
            return False
        return True
    except Exception:
        return False


def build_cmd_C(spec, out_dir, timeout, data_dir):
    return [sys.executable, PRETRAIN,
            "--corpus", spec["corpus"], "--data-dir", data_dir,
            "--d-model", str(spec["d_model"]), "--d-state", str(spec["d_state"]),
            "--n-layers", str(spec["n_layers"]), "--seq-len", str(spec["seq_len"]),
            "--batch-size", str(spec["batch_size"]), "--steps", str(spec["steps"]),
            "--seed", str(spec["seed"]), "--internal-timeout", str(max(1, timeout - 30)),
            "--ckpt-dir", _ckpt_dir(out_dir),
            "--out", os.path.join(out_dir, f"{spec['name']}.json")]


# ---------------------------------------------------------------------------
# Wave D -- intervention manifest (depends on Wave C's saved checkpoints)
# ---------------------------------------------------------------------------

def _final_checkpoint_path(waveC_result_json: str) -> str | None:
    if not os.path.exists(waveC_result_json):
        return None
    try:
        with open(waveC_result_json) as f:
            d = json.load(f)
    except Exception:
        return None
    if d.get("complete") is not True:
        return None
    return d.get("final_checkpoint_path")


def waveD_manifest(waveC_out_dir: str, k_grid=(8, 16, 24, 32, 48, 64), n_eval_windows=32,
                    d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS):
    """One intervention run per (corpus, seed) Wave C cell, evaluated
    against BOTH corpora's val slices (section 6.2's two arms, run inside a
    single lm_intervene_rd.py call). A cell whose Wave C run has not
    completed (checkpoint not found) is printed and DROPPED, not crashed
    on -- callers should re-run --wave D after Wave C finishes."""
    runs = []
    for corpus in CORPORA:
        for seed in SEEDS:
            cname = waveC_name(corpus, seed, d_model, d_state, n_layers)
            ckpt = _final_checkpoint_path(os.path.join(waveC_out_dir, f"{cname}.json"))
            runs.append({
                "wave": "D", "corpus": corpus, "seed": seed, "checkpoint": ckpt,
                "k_grid": list(k_grid), "n_eval_windows": n_eval_windows,
                "name": f"wD_lm_{corpus}_s{seed}", "source_run": cname,
            })
    return runs


def is_done_D(out_dir, spec) -> bool:
    p = os.path.join(out_dir, f"{spec['name']}.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict) or d.get("complete") is not True:
            return False
        if d.get("checkpoint") != spec["checkpoint"] or d.get("k_grid") != spec["k_grid"]:
            return False
        return True
    except Exception:
        return False


def build_cmd_D(spec, out_dir, timeout, data_dir):
    # AUDIT FIX-1: --seed is deliberately NOT passed. Pre-fix, the training
    # seed was forwarded here and seeded lm_intervene_rd's window sampler,
    # so each (corpus, seed) cell was measured on DIFFERENT windows;
    # lm_intervene_rd now seeds its windows from corpus_fixed_seed(corpus)
    # (its --seed flag is a documented no-op kept for CLI compatibility).
    cmd = [sys.executable, INTERVENE, "--checkpoint", spec["checkpoint"], "--data-dir", data_dir,
           "--k-grid"] + [str(k) for k in spec["k_grid"]] + \
          ["--n-eval-windows", str(spec["n_eval_windows"]),
           "--out", os.path.join(out_dir, f"{spec['name']}.json")]
    return cmd


# ---------------------------------------------------------------------------
# Shared orchestration (smoke gate, launch/poll loop, aggregate) -- CLONE of
# run_deltanet_rd_sweep.py's pattern, generalized over the two waves' spec
# shapes via is_done_fn/build_cmd_fn.
# ---------------------------------------------------------------------------

def run_smoke(log_dir, gpu):
    print(f"SMOKE GATE (physical GPU {gpu}) -- both lm_pretrain_rd.py and lm_intervene_rd.py ...", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    ok = True
    for name, script in (("pretrain", PRETRAIN), ("intervene", INTERVENE)):
        with open(os.path.join(log_dir, f"smoke_{name}.log"), "w") as lf:
            rc = subprocess.call([sys.executable, script, "--smoke"], env=env,
                                  stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
        print(f"  smoke[{name}]: {'PASS' if rc == 0 else f'FAIL (rc={rc})'}", flush=True)
        ok = ok and (rc == 0)
    return ok


def write_progress(out_dir, done, failed, running, wave):
    try:
        with open(os.path.join(out_dir, "PROGRESS.txt"), "w") as f:
            f.write(f"wave={wave} done={done} failed={failed} running={running}\n")
    except Exception as e:
        print(f"  (write_progress non-fatal: {e!r})", flush=True)


def aggregate(out_dir, manifest, failed, wave):
    """Minimal aggregate -- section 6.3's descriptive+interventional claim
    tier has no premise-valid gating machinery to replicate (unlike
    run_deltanet_rd_sweep.py's aggregate(), Wave 1's premise-classification
    logic does NOT transfer here, section 14.7). Reports raw per-run
    completion + a few headline numbers per cell, straight from each
    result JSON."""
    report = {"wave": wave, "n_manifest": len(manifest), "n_failed": len(failed)}
    try:
        cells = {}
        for spec in manifest:
            p = os.path.join(out_dir, f"{spec['name']}.json")
            if not os.path.exists(p):
                continue
            try:
                with open(p) as f:
                    d = json.load(f)
            except Exception:
                continue
            if d.get("complete") is not True:
                continue
            if wave == "C":
                ck = (d.get("checkpoints") or [])
                final = ck[-1] if ck else {}
                cells[spec["name"]] = {"corpus": d.get("corpus"), "seed": d.get("seed"),
                                        "final_step": d.get("final_step"),
                                        "final_val_loss": final.get("val_loss"),
                                        "wall_s": d.get("wall_s")}
            else:
                cells[spec["name"]] = {"corpus": d.get("checkpoint_train_corpus"),
                                        "checkpoint_step": d.get("checkpoint_step"),
                                        "k_grid": d.get("k_grid"), "wall_s": d.get("wall_s")}
        report["cells"] = cells
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write(f"DeltaNet-RD Wave 2 (instrumented LM) -- wave {wave}\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def _run_wave(wave, manifest, out_dir, args, is_done_fn, build_cmd_fn, timeout_fn):
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir, args.gpu_offset):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate failed; no training/intervention launched.\n")
        sys.exit(1)

    physical_gpus = list(range(args.gpu_offset, args.gpu_offset + args.gpus))
    slots = [g for _ in range(args.per_gpu) for g in physical_gpus]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done_fn(out_dir, s)]
    dropped = [s for s in pending if wave == "D" and s.get("checkpoint") is None]
    pending = [s for s in pending if not (wave == "D" and s.get("checkpoint") is None)]
    for s in dropped:
        print(f"  SKIP {s['name']}: Wave C source run {s['source_run']!r} has no completed "
              f"checkpoint yet -- re-run --wave D after Wave C finishes.", flush=True)
    print(f"wave={wave}  manifest={len(manifest)}  pending={len(pending)}  dropped={len(dropped)}  "
          f"slots={n_slots} (gpus {physical_gpus} x {args.per_gpu} per-gpu)", flush=True)

    running, free, quarantined = {}, list(slots), []
    done_ct, failed, uid = 0, [], 0
    last_agg = time.time()

    try:
        while pending or running:
            while free and pending:
                gpu = free.pop()
                spec = pending.pop(0)
                timeout = args.timeout if args.timeout is not None else timeout_fn(spec)
                try:
                    lf = open(os.path.join(log_dir, f"{spec['name']}.log"), "w")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    proc = subprocess.Popen(build_cmd_fn(spec, out_dir, timeout, args.data_dir), env=env,
                                             stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
                    running[uid] = (proc, spec, lf, time.time(), gpu, timeout)
                    uid += 1
                except Exception as e:
                    print(f"  LAUNCH-FAILED {spec['name']}: {e!r}", flush=True)
                    failed.append((spec["name"], "LAUNCH_ERR"))
                    free.append(gpu)
            for u, (proc, spec, lf, start, gpu, timeout) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    if time.time() - start > timeout:
                        proc.kill()
                        reaped = True
                        try:
                            proc.wait(timeout=30)
                        except Exception:
                            reaped = False
                        try:
                            lf.close()
                        except Exception:
                            pass
                        del running[u]
                        (free if reaped else quarantined).append(gpu)
                        if not reaped:
                            print(f"  QUARANTINE gpu {gpu}: reap timed out", flush=True)
                        failed.append((spec["name"], "TIMEOUT"))
                    continue
                try:
                    lf.close()
                except Exception:
                    pass
                del running[u]
                free.append(gpu)
                if rc == 0 and is_done_fn(out_dir, spec):
                    done_ct += 1
                else:
                    failed.append((spec["name"], rc))
            write_progress(out_dir, done_ct, len(failed), len(running), wave)
            if time.time() - last_agg > 120:
                aggregate(out_dir, manifest, failed, wave)
                last_agg = time.time()
            if pending and not running and not free:
                print(f"  ABORT: no usable GPUs ({len(quarantined)} quarantined)", flush=True)
                break
            time.sleep(args.poll)
    except Exception as e:
        import traceback
        try:
            with open(os.path.join(out_dir, "CRASHED.txt"), "w") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        print(f"ORCHESTRATOR CRASHED: {e!r} -- see CRASHED.txt; rerun --wave {wave} to resume "
              f"(validity-checked, bounded, not perpetual).", flush=True)
    finally:
        aggregate(out_dir, manifest, failed, wave)

    all_done = (done_ct == len(manifest)) and not dropped and not failed
    if all_done:
        with open(os.path.join(out_dir, "ALL_DONE"), "w") as f:
            f.write(f"wave {wave} complete: {done_ct}/{len(manifest)} runs, 0 failed, 0 dropped\n")
    print(f"\nWAVE {wave} DONE. {done_ct} succeeded, {len(failed)} failed, {len(dropped)} dropped "
          f"(pending Wave C). ALL_DONE {'written' if all_done else 'NOT written (wave incomplete)'}.",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/lm_rd"))
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--wave", choices=["C", "D"], default=None,
                     help="which wave to launch; REQUIRED unless --dry-run. Waves launch "
                          "separately with a human gate between them (no perpetual refill).")
    ap.add_argument("--gpus", type=int, default=None,
                     help="GPU COUNT to use. REQUIRED for a real launch, NO DEFAULT ON PURPOSE: "
                          "this box runs concurrent waves from other experiments and the busy-GPU "
                          "set changes day to day. Run nvidia-smi immediately before every launch "
                          "and pass the free set explicitly.")
    ap.add_argument("--gpu-offset", type=int, default=None,
                     help="first physical GPU index to use. REQUIRED for a real launch, NO "
                          "DEFAULT ON PURPOSE -- see --gpus.")
    ap.add_argument("--per-gpu", type=int, default=1, help="runs packed per GPU")
    ap.add_argument("--steps", type=int, default=None,
                     help="Wave C only: override the per-run step count (default: derived from "
                          "TARGET_TOKENS / (batch_size*seq_len), see default_steps())")
    ap.add_argument("--k-grid", type=int, nargs="+", default=None,
                     help="Wave D only: override the truncation grid")
    ap.add_argument("--n-eval-windows", type=int, default=32, help="Wave D only")
    ap.add_argument("--timeout", type=float, default=None, help="override the per-run wall timeout (s)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    if args.dry_run:
        mC = waveC_manifest(args.steps)
        mD = waveD_manifest(os.path.join(args.out_dir, "waveC"),
                             k_grid=args.k_grid or (8, 16, 24, 32, 48, 64), n_eval_windows=args.n_eval_windows)
        steps = args.steps if args.steps is not None else default_steps()
        est_pretrain_h = len(mC) * steps * _PER_STEP_S_PLACEHOLDER / 3600.0
        print(f"wave C: {len(mC)} runs ({CORPORA} x {SEEDS}), steps={steps} "
              f"(~{steps * BATCH_SIZE * SEQ_LEN / 1e6:.1f}M tokens/run), "
              f"est. {est_pretrain_h:.2f} GPU-h @ PLACEHOLDER {_PER_STEP_S_PLACEHOLDER}s/step "
              f"(update after the on-box calibration probe)")
        print(f"wave D: {len(mD)} runs (1 per Wave-C cell, both corpora scored per run), "
              f"k_grid={mD[0]['k_grid'] if mD else []} -- cheap, eval-only, no backward pass")
        if args.gpus is not None and args.gpu_offset is not None:
            print(f"slots = {args.gpus} gpus x {args.per_gpu} per-gpu = {args.gpus * args.per_gpu} "
                  f"concurrent, on physical GPUs {list(range(args.gpu_offset, args.gpu_offset + args.gpus))}")
        else:
            print("slots: pass --gpus/--gpu-offset to preview (REQUIRED for a real launch, no "
                  "defaults -- check nvidia-smi first).")
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)
    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose): the busy-GPU set on this box changes day to day. Run nvidia-smi NOW and "
              "pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "C":
        manifest = waveC_manifest(args.steps)
        out_dir = os.path.join(args.out_dir, "waveC")
        os.makedirs(out_dir, exist_ok=True)
        _run_wave("C", manifest, out_dir, args, is_done_C, build_cmd_C,
                  lambda spec: default_timeout_pretrain(spec["steps"], 1000))
    else:
        waveC_out_dir = os.path.join(args.out_dir, "waveC")
        k_grid = args.k_grid or (8, 16, 24, 32, 48, 64)
        manifest = waveD_manifest(waveC_out_dir, k_grid=k_grid, n_eval_windows=args.n_eval_windows)
        out_dir = os.path.join(args.out_dir, "waveD")
        os.makedirs(out_dir, exist_ok=True)
        _run_wave("D", manifest, out_dir, args, is_done_D, build_cmd_D,
                  lambda spec: default_timeout_intervene(spec["n_eval_windows"], spec["k_grid"]))


if __name__ == "__main__":
    main()
