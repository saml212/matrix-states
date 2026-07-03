"""run_lm_rd_geo3_sweep.py -- bounded, human-gated wave orchestrator for
Track B / geo3-in-LM (SCALE_TRANSFER_DESIGN.md sec 4): Wave 1 (primary,
beta-gated or naive-window per the Wave -1 gate's verdict, sec 4.2),
Wave 2 (naive-fixed-window ablation arm, sec 4.5 -- cut first if squeezed),
Wave 3 (eval-only inference-time rank-truncation grid on Wave 1's
checkpoints, reusing lm_intervene_rd.py UNCHANGED -- geo3 config is
self-describing inside every checkpoint's own saved config dict, sec 4.3).

CLONE of run_lm_rd_sweep.py's robustness pattern (smoke gate,
exception-isolated launch, validity-checked resume, per-run timeout with
GPU quarantine, guarded aggregate, REQUIRED --gpus/--gpu-offset with NO
defaults) -- deliberately NOT a cross-directory (or cross-script) import
(this codebase's own pod-safety convention; run_lm_rd_sweep.py's own module
docstring states the same rationale for its clone of
run_deltanet_rd_sweep.py). Same corpora/seeds/step-budget/eval-protocol as
Wave C/D (sec 4.5: "deliberately identical to the existing baseline so the
comparison is clean").

**THE GATE IS WIRED IN, NOT DOCUMENTATION-ONLY (sec 4.2's Rev-2 MAJOR-2
requirement, this build's own brief item (3)).** Wave 1/2/3 all REQUIRE a
Wave -1 gate JSON (lm_geo3_wave_neg1_gate.py's own output, default path
`<out-dir>/wave_neg1_gate.json`, override via --gate-json) to exist and be
loaded BEFORE any manifest is built. The (b)-failure branch
("no_launch_redesign") is a HARD refusal -- main() exits non-zero (exit
code 3) BEFORE constructing any manifest, launching any subprocess, or even
printing a manifest preview beyond the refusal message itself, for EVERY
wave in this file (1, 2, AND 3 -- Wave 3 evaluates Wave 1's own
checkpoints, so it inherits the same refusal). This is enforced in code,
not left to operator discipline.

**This build phase does NOT launch anything -- it gets an independent audit
first (task brief). This script is complete and smoke-gated, but main()
refuses to run without an explicit --wave, exactly like
run_lm_rd_sweep.py's/run_deltanet_rd_sweep.py's own gate, and nothing in
this repo invokes it automatically.**

Usage (GPU list is an example -- check nvidia-smi first, per house rule):
  python run_lm_rd_geo3_sweep.py --dry-run
  python lm_geo3_wave_neg1_gate.py --checkpoints <waveC ckpts...> \\
      --out results/lm_rd_geo3/wave_neg1_gate.json      # MUST run first
  python run_lm_rd_geo3_sweep.py --wave 1 --out-dir results/lm_rd_geo3 --gpus 2 --gpu-offset 6
  python run_lm_rd_geo3_sweep.py --wave 2 --out-dir results/lm_rd_geo3 --gpus 2 --gpu-offset 6
  python run_lm_rd_geo3_sweep.py --wave 3 --out-dir results/lm_rd_geo3 --gpus 2 --gpu-offset 6
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
GATE_TOOL = os.path.join(HERE, "lm_geo3_wave_neg1_gate.py")

CORPORA = ("openr1", "wikitext")
SEEDS = (0, 1, 2)
K_SELS = (16, 32)

# sec 4.5: "Same corpora, seeds, step budget, and eval protocol as Wave C/D
# ... deliberately identical to the existing baseline." Values match
# run_lm_rd_sweep.py's own D_MODEL/D_STATE/N_LAYERS/SEQ_LEN/BATCH_SIZE/
# TARGET_TOKENS exactly (duplicated here per this codebase's own clone-not-
# import convention, not independently re-derived).
D_MODEL = 256
D_STATE = 64
N_LAYERS = 2
SEQ_LEN = 512
BATCH_SIZE = 32
TARGET_TOKENS = 100_000_000
GEO3_CHUNK_SIZE = 64
GEO3_RESID_TOL = 1e-2

# sec 1.1: K=16 uses n_iter=12 (the original wave, 3/3 admissible). K=32
# is ONLY admissible after the n_iter=20 ESCALATION (0/3 -> 3/3, zero
# fallback steps) -- n_iter=12 at K=32 is the non-admissible, descriptive-
# only config and must NOT be silently reused here. Self-caught during this
# build's own review (an earlier draft of this file applied a single
# GEO3_N_ITER=12 constant to every K_sel, which would have replicated the
# non-admissible K=32 config for every Wave 1/2 K=32 cell).
GEO3_N_ITER_BY_K_SEL = {16: 12, 32: 20}


def geo3_n_iter_for_k_sel(k_sel: int) -> int:
    if k_sel not in GEO3_N_ITER_BY_K_SEL:
        raise ValueError(
            f"k_sel={k_sel} has no registered Newton-Schulz n_iter (sec 1.1 only pins 16->12 and "
            f"32->20) -- add it to GEO3_N_ITER_BY_K_SEL deliberately, not by falling back to a guess")
    return GEO3_N_ITER_BY_K_SEL[k_sel]

# Per-step/per-checkpoint wall-clock: geo3-in-LM adds real compute (per-chunk
# gather/topk/Newton-Schulz/scatter every mixer forward call, sec 4.3) on top
# of run_lm_rd_sweep.py's own measured ~0.05s/step non-geo3 baseline -- NOT
# yet measured on-box for the geo3-active path. Conservatively 3x the
# non-geo3 baseline as a PLANNING placeholder only; Wave -1's own timing (or
# a short calibration run) supersedes this before any real Wave 1 launch,
# per this project's calibration-first discipline (matches sec 9's
# consolidated statement, applied here even though sec 4.5's table prices
# Track B in aggregate GPU-h, not a per-step constant).
_PER_STEP_S_PLACEHOLDER_GEO3 = 0.15
_PER_CHECKPOINT_S_MEASURED = 15.0


def default_steps(batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN, target_tokens: int = TARGET_TOKENS) -> int:
    tokens_per_step = batch_size * seq_len
    return max(1, target_tokens // tokens_per_step)


def default_timeout_pretrain(steps: int, ckpt_every: int, margin: float = 1.6) -> int:
    n_ckpts = steps // ckpt_every + 1
    base = (_PER_STEP_S_PLACEHOLDER_GEO3 * steps + n_ckpts * _PER_CHECKPOINT_S_MEASURED) * margin
    return int(base)


def default_timeout_intervene(n_eval_windows: int, k_grid, margin: float = 1.6) -> int:
    per_k_s = 3.0
    base = (30.0 + len(k_grid) * per_k_s * (n_eval_windows / 8.0)) * margin * 2   # x2 for two corpora
    return int(base)


# ---------------------------------------------------------------------------
# Wave -1 gate loading -- THE hard gate every wave below is refused without
# ---------------------------------------------------------------------------

_VALID_VERDICTS = ("beta_gated_primary", "naive_window_primary", "no_launch_redesign")


def load_gate_verdict(gate_json_path: str) -> tuple[str, dict]:
    if not os.path.exists(gate_json_path):
        print(f"ERROR: Wave -1 gate JSON not found at {gate_json_path!r}. Run "
              f"lm_geo3_wave_neg1_gate.py FIRST (sec 4.2's mandatory pre-check) -- "
              f"Wave 1/2/3 launch is refused without a recorded gate verdict.", file=sys.stderr)
        sys.exit(2)
    with open(gate_json_path) as f:
        gate = json.load(f)
    verdict = gate.get("gate_verdict")
    if verdict not in _VALID_VERDICTS:
        print(f"ERROR: malformed gate JSON at {gate_json_path!r}: gate_verdict={verdict!r} "
              f"not in {_VALID_VERDICTS}.", file=sys.stderr)
        sys.exit(2)
    # AUDIT ROUND-2 MINOR-4: cross-validate the gate JSON's own measurement configuration
    # against THIS file's Wave 1 constants -- a gate run at (say) --chunk-size 32 would otherwise
    # produce a verdict for a DIFFERENT construction than the one Wave 1 actually launches, and be
    # trusted uncritically (same defense-in-depth style as load_corpus's meta.json field checks).
    mismatches = []
    if gate.get("chunk_size") != GEO3_CHUNK_SIZE:
        mismatches.append(f"chunk_size={gate.get('chunk_size')!r} (sweep uses {GEO3_CHUNK_SIZE})")
    measured = set(gate.get("k_sels_measured") or [])
    if not set(K_SELS) <= measured:
        mismatches.append(f"k_sels_measured={sorted(measured)} lacks the sweep's K_SELS={list(K_SELS)}")
    if gate.get("gate_k_sel") != max(K_SELS):
        mismatches.append(f"gate_k_sel={gate.get('gate_k_sel')!r} (sec 4.2's registered gate is "
                           f"top-{max(K_SELS)})")
    if mismatches:
        print(f"ERROR: gate JSON at {gate_json_path!r} was measured under a DIFFERENT "
              f"configuration than this sweep would launch: {'; '.join(mismatches)}. Re-run "
              f"lm_geo3_wave_neg1_gate.py at the sweep's own configuration.", file=sys.stderr)
        sys.exit(2)
    return verdict, gate


def selection_mode_for_verdict(verdict: str) -> str:
    assert verdict in ("beta_gated_primary", "naive_window_primary"), (
        f"selection_mode_for_verdict called with verdict={verdict!r} -- no_launch_redesign must "
        f"be refused by the CALLER before this function is ever reached")
    return "beta_topk" if verdict == "beta_gated_primary" else "naive_window"


def _refuse_if_no_launch(verdict: str, gate_json_path: str) -> None:
    if verdict == "no_launch_redesign":
        print("=" * 70, file=sys.stderr)
        print("HARD NO-LAUNCH (sec 4.2's registered outcome (iii)): the Wave -1 gate's criterion "
              "(b) -- excluded-position write-mass -- FAILED.", file=sys.stderr)
        print(f"Gate JSON: {gate_json_path}", file=sys.stderr)
        print("Track B's Wave 1 (and everything downstream: Wave 2's ablation, Wave 3's "
              "truncation grid) CANNOT launch on the currently-specified construction. Route to "
              "redesign (sec 4.2's own next step: the hard-zero-beta variant, gated behind its own "
              "attack pass) -- this is NOT a bug, it is the gate working as registered.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(3)


# ---------------------------------------------------------------------------
# Wave 1 -- primary (beta-gated OR naive-window, per the gate verdict)
# ---------------------------------------------------------------------------

def _ckpt_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "checkpoints")


def waveB1_name(corpus, seed, k_sel, selection_mode) -> str:
    tag = "beta" if selection_mode == "beta_topk" else "naive"
    return f"wB1_lm_{corpus}_k{k_sel}_{tag}_s{seed}"


def waveB1_manifest(verdict: str, steps=None):
    selection_mode = selection_mode_for_verdict(verdict)
    steps = steps if steps is not None else default_steps()
    runs = []
    for k_sel in K_SELS:
        for corpus in CORPORA:
            for seed in SEEDS:
                runs.append({
                    "wave": "B1", "corpus": corpus, "seed": seed, "k_sel": k_sel,
                    "selection_mode": selection_mode, "n_iter": geo3_n_iter_for_k_sel(k_sel),
                    "steps": steps, "name": waveB1_name(corpus, seed, k_sel, selection_mode),
                })
    return runs


def is_done_B(out_dir, spec) -> bool:
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
        required = ("corpus", "seed", "steps", "steps_completed", "geo3_lm")
        if not all(k in d for k in required):
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        g = d.get("geo3_lm") or {}
        if (d.get("corpus") != spec["corpus"] or d.get("seed") != spec["seed"]
                or d.get("steps") != spec["steps"] or g.get("active") is not True
                or g.get("k_sel") != spec["k_sel"] or g.get("selection_mode") != spec["selection_mode"]
                or g.get("n_iter") != spec["n_iter"]):
            return False
        return True
    except Exception:
        return False


def build_cmd_B(spec, out_dir, timeout, data_dir):
    return [sys.executable, PRETRAIN,
            "--corpus", spec["corpus"], "--data-dir", data_dir,
            "--d-model", str(D_MODEL), "--d-state", str(D_STATE), "--n-layers", str(N_LAYERS),
            "--seq-len", str(SEQ_LEN), "--batch-size", str(BATCH_SIZE), "--steps", str(spec["steps"]),
            "--seed", str(spec["seed"]), "--internal-timeout", str(max(1, timeout - 30)),
            "--use-geo3-lm", "--geo3-k-sel", str(spec["k_sel"]), "--geo3-chunk-size", str(GEO3_CHUNK_SIZE),
            "--geo3-n-iter", str(spec["n_iter"]), "--geo3-resid-tol", str(GEO3_RESID_TOL),
            "--geo3-selection", spec["selection_mode"],
            "--ckpt-dir", _ckpt_dir(out_dir),
            "--out", os.path.join(out_dir, f"{spec['name']}.json")]


# ---------------------------------------------------------------------------
# Wave 2 -- naive-fixed-window ablation (sec 4.5: 1 corpus, cut first if
# squeezed). SKIPPED (empty manifest) when Wave 1's OWN construction is
# already naive_window (gate verdict naive_window_primary) -- a duplicate
# arm would not be a meaningful ablation, sec 4.2 outcome (ii)'s consequence.
# ---------------------------------------------------------------------------

def waveB2_name(corpus, seed, k_sel) -> str:
    return f"wB2_lm_{corpus}_k{k_sel}_naive_s{seed}"


def waveB2_manifest(verdict: str, steps=None):
    if verdict == "naive_window_primary":
        return []
    steps = steps if steps is not None else default_steps()
    runs = []
    for k_sel in K_SELS:
        for seed in SEEDS:
            runs.append({
                "wave": "B2", "corpus": "openr1", "seed": seed, "k_sel": k_sel,
                "selection_mode": "naive_window", "n_iter": geo3_n_iter_for_k_sel(k_sel),
                "steps": steps, "name": waveB2_name("openr1", seed, k_sel),
            })
    return runs


# is_done_B / build_cmd_B are reused for Wave 2 unmodified -- the spec dict
# shape is identical (wave B1 vs B2 differ only in which cells they cover
# and are always explicitly "naive_window" for B2).


# ---------------------------------------------------------------------------
# Wave 3 -- eval-only truncation grid on Wave 1's checkpoints. Reuses
# lm_intervene_rd.py UNCHANGED (sec 4.3: geo3 config lives inside the
# checkpoint's own saved config dict, DeltaNetLM(**ckpt["config"])
# reconstructs it automatically -- no intervention-script code change
# needed at all).
# ---------------------------------------------------------------------------

def _final_checkpoint_path(result_json: str) -> str | None:
    if not os.path.exists(result_json):
        return None
    try:
        with open(result_json) as f:
            d = json.load(f)
    except Exception:
        return None
    if d.get("complete") is not True:
        return None
    return d.get("final_checkpoint_path")


def waveB3_manifest(waveB1_out_dir: str, verdict: str, k_grid=(8, 16, 24, 32, 48, 64), n_eval_windows=32):
    selection_mode = selection_mode_for_verdict(verdict)
    runs = []
    for k_sel in K_SELS:
        for corpus in CORPORA:
            for seed in SEEDS:
                cname = waveB1_name(corpus, seed, k_sel, selection_mode)
                ckpt = _final_checkpoint_path(os.path.join(waveB1_out_dir, f"{cname}.json"))
                runs.append({
                    "wave": "B3", "corpus": corpus, "seed": seed, "k_sel": k_sel, "checkpoint": ckpt,
                    "k_grid": list(k_grid), "n_eval_windows": n_eval_windows,
                    "name": f"wB3_lm_{corpus}_k{k_sel}_s{seed}", "source_run": cname,
                })
    return runs


def is_done_B3(out_dir, spec) -> bool:
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
        # AUDIT ROUND-2 MINOR-2: a re-run at a different --n-eval-windows would otherwise be
        # silently treated as done and reuse the stale, different-fidelity result (lm_intervene_rd
        # records the field; the pre-fix check never compared it).
        if d.get("n_eval_windows") != spec["n_eval_windows"]:
            return False
        return True
    except Exception:
        return False


def build_cmd_B3(spec, out_dir, timeout, data_dir):
    cmd = [sys.executable, INTERVENE, "--checkpoint", spec["checkpoint"], "--data-dir", data_dir,
           "--k-grid"] + [str(k) for k in spec["k_grid"]] + \
          ["--n-eval-windows", str(spec["n_eval_windows"]),
           "--out", os.path.join(out_dir, f"{spec['name']}.json")]
    return cmd


# ---------------------------------------------------------------------------
# Shared orchestration -- CLONE of run_lm_rd_sweep.py's pattern
# ---------------------------------------------------------------------------

def run_smoke(log_dir, gpu):
    print(f"SMOKE GATE (physical GPU {gpu}) -- lm_pretrain_rd.py, lm_intervene_rd.py, AND "
          f"lm_geo3_wave_neg1_gate.py ...", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    ok = True
    for name, script in (("pretrain", PRETRAIN), ("intervene", INTERVENE), ("gate", GATE_TOOL)):
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
            if wave in ("B1", "B2"):
                ck = (d.get("checkpoints") or [])
                final = ck[-1] if ck else {}
                cells[spec["name"]] = {"corpus": d.get("corpus"), "seed": d.get("seed"),
                                        "k_sel": spec.get("k_sel"), "selection_mode": spec.get("selection_mode"),
                                        "final_step": d.get("final_step"),
                                        "final_val_loss": final.get("val_loss"),
                                        "geo3_diagnostics": final.get("geo3_diagnostics"),
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
            f.write(f"DeltaNet-RD Track B (geo3-in-LM) -- wave {wave}\n" + "=" * 50 + "\n")
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
    dropped = [s for s in pending if wave == "B3" and s.get("checkpoint") is None]
    pending = [s for s in pending if not (wave == "B3" and s.get("checkpoint") is None)]
    for s in dropped:
        print(f"  SKIP {s['name']}: Wave 1 source run {s['source_run']!r} has no completed "
              f"checkpoint yet -- re-run --wave 3 after Wave 1 finishes.", flush=True)
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
          f"(pending Wave 1). ALL_DONE {'written' if all_done else 'NOT written (wave incomplete)'}.",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/lm_rd_geo3"))
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--gate-json", default=None,
                     help="path to lm_geo3_wave_neg1_gate.py's own output JSON. Default: "
                          "<out-dir>/wave_neg1_gate.json. REQUIRED to exist for waves 1/2/3 -- "
                          "this is the enforced gate, not documentation (sec 4.2).")
    ap.add_argument("--wave", choices=["1", "2", "3"], default=None,
                     help="which wave to launch; REQUIRED unless --dry-run. Waves launch "
                          "separately with a human gate between them (no perpetual refill).")
    ap.add_argument("--gpus", type=int, default=None,
                     help="GPU COUNT to use. REQUIRED for a real launch, NO DEFAULT ON PURPOSE: "
                          "check nvidia-smi immediately before every launch and pass the free set "
                          "explicitly.")
    ap.add_argument("--gpu-offset", type=int, default=None,
                     help="first physical GPU index to use. REQUIRED for a real launch, NO "
                          "DEFAULT ON PURPOSE -- see --gpus.")
    ap.add_argument("--per-gpu", type=int, default=1, help="runs packed per GPU")
    ap.add_argument("--steps", type=int, default=None,
                     help="Wave 1/2 only: override the per-run step count (default: derived from "
                          "TARGET_TOKENS / (batch_size*seq_len), see default_steps())")
    ap.add_argument("--k-grid", type=int, nargs="+", default=None, help="Wave 3 only: override the truncation grid")
    ap.add_argument("--n-eval-windows", type=int, default=32, help="Wave 3 only")
    ap.add_argument("--timeout", type=float, default=None, help="override the per-run wall timeout (s)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    gate_json_path = args.gate_json or os.path.join(args.out_dir, "wave_neg1_gate.json")

    if args.dry_run:
        steps = args.steps if args.steps is not None else default_steps()
        if os.path.exists(gate_json_path):
            verdict, _ = load_gate_verdict(gate_json_path)
            print(f"Wave -1 gate JSON found at {gate_json_path}: verdict={verdict}")
            if verdict == "no_launch_redesign":
                print("  -> Wave 1/2/3 would be REFUSED (HARD no-launch, sec 4.2 outcome (iii)). "
                      "The manifest previews below are HYPOTHETICAL ONLY (shown for planning, not "
                      "an indication a real launch would proceed).")
            m1 = waveB1_manifest(verdict if verdict != "no_launch_redesign" else "beta_gated_primary", steps)
            m2 = waveB2_manifest(verdict if verdict != "no_launch_redesign" else "beta_gated_primary", steps)
        else:
            print(f"Wave -1 gate JSON NOT YET FOUND at {gate_json_path} -- run "
                  f"lm_geo3_wave_neg1_gate.py first. Manifest preview below assumes the "
                  f"'beta_gated_primary' verdict HYPOTHETICALLY, for planning only.")
            m1 = waveB1_manifest("beta_gated_primary", steps)
            m2 = waveB2_manifest("beta_gated_primary", steps)
        est_h1 = len(m1) * steps * _PER_STEP_S_PLACEHOLDER_GEO3 / 3600.0
        est_h2 = len(m2) * steps * _PER_STEP_S_PLACEHOLDER_GEO3 / 3600.0
        print(f"wave 1: {len(m1)} runs ({K_SELS} x {CORPORA} x {SEEDS}), steps={steps} "
              f"(~{steps * BATCH_SIZE * SEQ_LEN / 1e6:.1f}M tokens/run), "
              f"est. {est_h1:.2f} GPU-h @ PLACEHOLDER {_PER_STEP_S_PLACEHOLDER_GEO3}s/step "
              f"(UNCALIBRATED for the geo3-active path -- update after an on-box timing probe)")
        wave2_status = "SKIPPED (Wave 1 is already naive_window)" if not m2 else "naive-window ablation"
        print(f"wave 2: {len(m2)} runs ({K_SELS} x openr1 x {SEEDS}) -- {wave2_status}, "
              f"est. {est_h2:.2f} GPU-h")
        print(f"wave 3: 1 run per Wave-1 cell (both corpora scored per run), cheap eval-only, "
              f"no backward pass")
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

    # THE GATE (sec 4.2, this build's item (3)): loaded and enforced for EVERY wave, before any
    # manifest is built or any subprocess launched.
    verdict, gate = load_gate_verdict(gate_json_path)
    print(f"Wave -1 gate verdict (from {gate_json_path}): {verdict}", flush=True)
    _refuse_if_no_launch(verdict, gate_json_path)

    if args.wave == "1":
        manifest = waveB1_manifest(verdict, args.steps)
        out_dir = os.path.join(args.out_dir, "waveB1")
        os.makedirs(out_dir, exist_ok=True)
        _run_wave("B1", manifest, out_dir, args, is_done_B, build_cmd_B,
                  lambda spec: default_timeout_pretrain(spec["steps"], 1000))
    elif args.wave == "2":
        manifest = waveB2_manifest(verdict, args.steps)
        out_dir = os.path.join(args.out_dir, "waveB2")
        os.makedirs(out_dir, exist_ok=True)
        if not manifest:
            print("Wave 2 SKIPPED: Wave 1's own primary construction is ALREADY naive_window "
                  "(gate verdict=naive_window_primary) -- a duplicate ablation arm would not be "
                  "meaningful (sec 4.2 outcome (ii)'s consequence). Nothing launched.", flush=True)
            return
        _run_wave("B2", manifest, out_dir, args, is_done_B, build_cmd_B,
                  lambda spec: default_timeout_pretrain(spec["steps"], 1000))
    else:
        waveB1_out_dir = os.path.join(args.out_dir, "waveB1")
        k_grid = args.k_grid or (8, 16, 24, 32, 48, 64)
        manifest = waveB3_manifest(waveB1_out_dir, verdict, k_grid=k_grid, n_eval_windows=args.n_eval_windows)
        out_dir = os.path.join(args.out_dir, "waveB3")
        os.makedirs(out_dir, exist_ok=True)
        _run_wave("B3", manifest, out_dir, args, is_done_B3, build_cmd_B3,
                  lambda spec: default_timeout_intervene(spec["n_eval_windows"], spec["k_grid"]))


if __name__ == "__main__":
    main()
