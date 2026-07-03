"""run_lm_rd_trackc_sweep.py -- bounded, human-gated wave orchestrator for
Track C / the scaling ladder (SCALE_TRANSFER_DESIGN.md sec 5), THIS
SESSION'S SCOPE ONLY: rung 1 (~98M params, sec 5.3/lm_rd_rung_configs.py)
plus the REQUIRED small-model same-mix control cell (sec 5.6 Wave 1,
MAJOR-5). Rungs 2/3 are registered in lm_rd_rung_configs.py for a future
session but this launcher deliberately does not build manifests for them
(BUILD_SCOPE_RUNGS enforces this at the config layer already; this file
does not even try).

CLONE of run_lm_rd_geo3_sweep.py's / run_lm_rd_sweep.py's robustness
pattern (smoke gate, exception-isolated launch, validity-checked resume,
per-run timeout with GPU quarantine, guarded aggregate, REQUIRED
--gpus/--gpu-offset with NO defaults) -- deliberately NOT a cross-script
import (this codebase's own pod-safety convention, restated in both of
those files' own docstrings).

Waves (task brief's build scope -- items (4)/(1)/(2) of the brief map onto
these, in run order):
  calibration (Wave -1, sec 5.6's blocking calibration row): SHORT real
    training runs (default 200 steps) at BOTH the rung-1 config AND the
    control-cell config, on openr1-mix, seed 0 only -- measures REAL
    tok/s + peak memory (lm_pretrain_rd.py's own new peak_memory_*_bytes
    fields) before ANY full-budget run is priced or launched. Cheap
    (<2 GPU-h total per the task brief's own ceiling) -- THIS WAVE MAY BE
    LAUNCHED by this build session; report its measured numbers.
  1 (rung 1 + control cell, FULL manifest -- 6 rung-1 runs + 6 control
    runs per sec 5.6's table): built, smoke-gated, dry-run-previewable,
    and REFUSES to launch for real without an existing calibration.json
    (sec 9's own hard rule: "No track's Wave 1+ manifest is authorized to
    launch ... without first recording its Wave -1 measured numbers").
    **NOT LAUNCHED by this build session (task brief: "Do NOT launch the
    full rung runs") -- this wave exists complete and gated, for a human
    to launch after reviewing calibration.json.**
  4 (fix-effect / geo3-at-scale, sec 5.5 item 3): HARD REFUSED
    unconditionally by this file -- Track B's own Wave -1 gate returned
    `no_launch_redesign` (EXPERIMENT_LOG.md, "SCALE-TRANSFER Track B ...
    HARD NO-LAUNCH"), so sec 5.5 item 3 (which applies Track B's
    construction at scale) has no validated construction to transplant.
    This is checked LIVE against Track B's own gate JSON (not just cited
    from memory) every time --wave 4 is invoked.

Usage (GPU list is an example -- check nvidia-smi first, per house rule):
  python run_lm_rd_trackc_sweep.py --dry-run
  python run_lm_rd_trackc_sweep.py --wave calibration --out-dir results/lm_rd_trackc --gpus 2 --gpu-offset 0
  python run_lm_rd_trackc_sweep.py --wave 1 --out-dir results/lm_rd_trackc --gpus 4 --gpu-offset 0   # REFUSED without calibration.json
  python run_lm_rd_trackc_sweep.py --wave 4                                                          # ALWAYS refused this session
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
ATTRACTOR_PROBE = os.path.join(HERE, "lm_attractor_probe_rd.py")
RUNG_CONFIGS = os.path.join(HERE, "lm_rd_rung_configs.py")
BUILD_MIX = os.path.join(HERE, "build_mix_corpora_rd.py")
TRACKB_GATE_JSON_DEFAULT = os.path.join(HERE, "results", "lm_rd_geo3", "wave_neg1_gate.json")

sys.path.insert(0, HERE)  # pod-safe import of the config-layer module ONLY (not another sweep script)
from lm_rd_rung_configs import RUNGS, VOCAB_SIZE, verify_param_count  # noqa: E402

RUNG = 1                                          # this session's ONLY build/launch scope
RUNG_CFG = RUNGS[RUNG]
CONTROL_CFG = {"d_model": 256, "d_state": 64, "n_layers": 2}   # Wave C's own scale (sec 5.6, MAJOR-5)

MIX_CORPORA = ("openr1-mix", "wikitext-mix")      # sec 5.6 Wave 1: "openr1-mix, wikitext/finewebedu-mix"
SEEDS = (0, 1, 2)
SEQ_LEN = 512
BATCH_SIZE = 32

# sec 5.6: "the Wave-C-scale ... architecture retrained on the SAME augmented mixes ... at Wave C's
# measured ~4.6 min/run" -- the control cell REUSES Wave C's own exact token budget (this codebase's
# established TARGET_TOKENS=100_000_000 constant, run_lm_rd_sweep.py).
CONTROL_TARGET_TOKENS = 100_000_000

# Rung 1's OWN token budget is DELIBERATELY UNCALIBRATED here (sec 5.6: "Wave -1's measured
# throughput is the actual pricing authority, not this table"; sec 9: no Wave 1+ manifest may
# launch on an estimated figure without first recording Wave -1's measured numbers). This constant
# is a documented PLANNING PLACEHOLDER for --dry-run cost preview ONLY -- a real Wave 1 launch
# requires --rung1-steps (an explicit, human-supplied value informed by calibration.json), which
# main() enforces (no silent fallback to this placeholder once a real launch is attempted).
RUNG1_TARGET_TOKENS_PLACEHOLDER = 100_000_000

CALIBRATION_STEPS_DEFAULT = 200
CALIBRATION_CKPT_EVERY = 100                      # >=1 real checkpoint inside a 200-step calibration run

# PLANNING-ONLY per-step/per-checkpoint constants (rung 1 has NEVER been timed on this box -- these
# exist solely to size the calibration run's OWN --internal-timeout, generously, not to price a
# real wave). A 98M-param model at bf16-kernel-boundary is unlikely to be faster per-step than Wave
# C's measured ~0.037s/step (14M params) -- 5x generous headroom assumed here, corrected by the
# calibration run's own real measurement.
_CALIBRATION_PER_STEP_S_GENEROUS = 0.25
_CALIBRATION_PER_CKPT_S_GENEROUS = 60.0


def default_control_steps() -> int:
    return max(1, CONTROL_TARGET_TOKENS // (BATCH_SIZE * SEQ_LEN))


def default_rung1_steps_placeholder() -> int:
    return max(1, RUNG1_TARGET_TOKENS_PLACEHOLDER // (BATCH_SIZE * SEQ_LEN))


def default_timeout_pretrain(steps: int, ckpt_every: int, per_step_s: float, per_ckpt_s: float,
                              margin: float = 1.6) -> int:
    n_ckpts = steps // ckpt_every + 1
    base = (per_step_s * steps + n_ckpts * per_ckpt_s) * margin
    return int(base)


# ---------------------------------------------------------------------------
# Track B cross-check (sec 5.5 item 3 / this build's own scope statement):
# Wave 4 is refused unconditionally, but the refusal reads Track B's REAL
# gate JSON live (not just a cited verdict) so a future re-run of Track B
# that overturns the no-launch verdict is reflected automatically.
# ---------------------------------------------------------------------------

def _trackb_gate_status(gate_json_path: str) -> dict:
    if not os.path.exists(gate_json_path):
        return {"found": False, "verdict": None}
    try:
        with open(gate_json_path) as f:
            gate = json.load(f)
        return {"found": True, "verdict": gate.get("gate_verdict")}
    except Exception as e:
        return {"found": True, "verdict": None, "parse_error": repr(e)}


def refuse_wave4(gate_json_path: str) -> None:
    status = _trackb_gate_status(gate_json_path)
    print("=" * 70, file=sys.stderr)
    print("TRACK C WAVE 4 (fix-effect / geo3-at-scale, sec 5.5 item 3) IS GATED OUT.", file=sys.stderr)
    print(f"Track B gate JSON: {gate_json_path}", file=sys.stderr)
    print(f"Track B status: {status}", file=sys.stderr)
    if status.get("verdict") == "no_launch_redesign":
        print("Confirmed LIVE: Track B's own Wave -1 gate returned 'no_launch_redesign' (HARD "
              "NO-LAUNCH -- EXPERIMENT_LOG.md, 'SCALE-TRANSFER Track B ... HARD NO-LAUNCH'). There "
              "is no validated geo3-in-LM construction for Wave 4 to transplant to rung 1/3 (sec "
              "5.5 item 3's own dependency, sec 11's sequencing rule).", file=sys.stderr)
    elif not status.get("found"):
        print("Track B's gate JSON was not found at the expected path -- Wave 4 stays refused by "
              "DEFAULT (sec 5.5 item 3 requires a POSITIVE, validated Track B construction to "
              "transplant; absence of evidence is not evidence of readiness).", file=sys.stderr)
    else:
        print(f"Track B's verdict is {status.get('verdict')!r}, not the no-launch case this "
              f"session's build assumed -- this refusal message is STALE if that verdict is real; "
              f"a human must re-evaluate sec 5.5 item 3's gate before authoring a Wave 4 manifest "
              f"(this file intentionally contains none).", file=sys.stderr)
    print("This launcher CONTAINS NO Wave 4 manifest-building code -- there is nothing to preview "
          "or force past this refusal short of editing this file after a real re-authorization.",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.exit(3)


# ---------------------------------------------------------------------------
# Cell specs. ONE manifest builder covers rung-1 AND control cells (they
# differ only in d_model/d_state/n_layers/steps/name-prefix) -- is_done/
# build_cmd are shape-identical to run_lm_rd_sweep.py's own Wave C pattern,
# generalized over an explicit `cfg` dict so both cell types share one path
# (task brief item (5): "is_done identity on rung/config/mix").
# ---------------------------------------------------------------------------

def _ckpt_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "checkpoints")


def cell_name(tag: str, corpus: str, seed: int, cfg: dict) -> str:
    return f"{tag}_lm_{corpus}_dm{cfg['d_model']}_ds{cfg['d_state']}_L{cfg['n_layers']}_s{seed}"


def make_manifest(tag: str, cfg: dict, corpora, seeds, steps: int, ckpt_every: int) -> list[dict]:
    runs = []
    for corpus in corpora:
        for seed in seeds:
            runs.append({
                "tag": tag, "corpus": corpus, "seed": seed, "d_model": cfg["d_model"],
                "d_state": cfg["d_state"], "n_layers": cfg["n_layers"], "seq_len": SEQ_LEN,
                "batch_size": BATCH_SIZE, "steps": steps, "ckpt_every": ckpt_every,
                "name": cell_name(tag, corpus, seed, cfg),
            })
    return runs


def is_done_cell(out_dir, spec) -> bool:
    """Identity check keyed on rung/config (d_model/d_state/n_layers -- the
    rung's OWN identity) AND mix (corpus) AND seed AND steps -- task brief
    item (5)'s explicit requirement, cloned from run_lm_rd_sweep.py's
    is_done_C with the corpus set widened to the mix corpora."""
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


def build_cmd_cell(spec, out_dir, timeout, data_dir):
    return [sys.executable, PRETRAIN,
            "--corpus", spec["corpus"], "--data-dir", data_dir,
            "--d-model", str(spec["d_model"]), "--d-state", str(spec["d_state"]),
            "--n-layers", str(spec["n_layers"]), "--seq-len", str(spec["seq_len"]),
            "--batch-size", str(spec["batch_size"]), "--steps", str(spec["steps"]),
            "--ckpt-every", str(spec["ckpt_every"]),
            "--seed", str(spec["seed"]), "--internal-timeout", str(max(1, timeout - 30)),
            "--ckpt-dir", _ckpt_dir(out_dir),
            "--out", os.path.join(out_dir, f"{spec['name']}.json")]


# ---------------------------------------------------------------------------
# Calibration (Wave -1) manifest -- ONE rung-1 cell + ONE control cell,
# seed 0, openr1-mix only, SHORT step budget. May be launched by this build
# session (task brief: "MAY run if cheap (<2 GPU-h total)").
# ---------------------------------------------------------------------------

def calibration_manifest(steps: int, ckpt_every: int) -> list[dict]:
    return (make_manifest("calib_rung1", RUNG_CFG, ("openr1-mix",), (0,), steps, ckpt_every)
            + make_manifest("calib_control", CONTROL_CFG, ("openr1-mix",), (0,), steps, ckpt_every))


def summarize_calibration(out_dir: str, manifest: list[dict]) -> dict:
    """Extracts measured tok/s (from wall_s / steps_completed*batch*seq) and peak memory from each
    calibration cell's own result JSON -- the task brief's explicit deliverable ("report their
    measured numbers"). Written to calibration.json, which --wave 1's real-launch gate (sec 9)
    checks for existence before pricing/launching the full manifest."""
    cells = {}
    for spec in manifest:
        p = os.path.join(out_dir, f"{spec['name']}.json")
        if not os.path.exists(p):
            cells[spec["name"]] = {"status": "missing"}
            continue
        with open(p) as f:
            d = json.load(f)
        if d.get("complete") is not True:
            cells[spec["name"]] = {"status": "incomplete", "steps_completed": d.get("steps_completed")}
            continue
        tokens = d["steps_completed"] * spec["batch_size"] * spec["seq_len"]
        toks_per_s = tokens / d["wall_s"] if d.get("wall_s") else None
        cells[spec["name"]] = {
            "status": "complete", "tag": spec["tag"], "d_model": spec["d_model"],
            "d_state": spec["d_state"], "n_layers": spec["n_layers"], "n_params": d.get("n_params"),
            "steps_completed": d["steps_completed"], "wall_s": d["wall_s"],
            "tokens_per_s_per_gpu": toks_per_s,
            "peak_memory_allocated_gb": (d.get("peak_memory_allocated_bytes") or 0) / 1e9,
            "peak_memory_reserved_gb": (d.get("peak_memory_reserved_bytes") or 0) / 1e9,
        }
    return {"design_ref": "SCALE_TRANSFER_DESIGN.md sec 5.6 Wave -1 (Track C calibration)",
            "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": cells}


# ---------------------------------------------------------------------------
# Shared orchestration -- CLONE of run_lm_rd_geo3_sweep.py's pattern
# (smoke gate, exception-isolated launch, validity-checked resume, per-run
# timeout + GPU quarantine, guarded aggregate, ALL_DONE sentinel).
# ---------------------------------------------------------------------------

def run_smoke(log_dir, gpu):
    print(f"SMOKE GATE (physical GPU {gpu}) -- lm_pretrain_rd.py, lm_rd_rung_configs.py, "
          f"lm_attractor_probe_rd.py, build_mix_corpora_rd.py ...", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    ok = True
    for name, script in (("pretrain", PRETRAIN), ("rung_configs", RUNG_CONFIGS),
                          ("attractor_probe", ATTRACTOR_PROBE), ("build_mix", BUILD_MIX)):
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
            ck = d.get("checkpoints") or []
            final = ck[-1] if ck else {}
            cells[spec["name"]] = {
                "tag": spec.get("tag"), "corpus": d.get("corpus"), "seed": d.get("seed"),
                "d_model": spec.get("d_model"), "d_state": spec.get("d_state"), "n_layers": spec.get("n_layers"),
                "n_params": d.get("n_params"), "final_step": d.get("final_step"),
                "final_val_loss": final.get("val_loss"), "wall_s": d.get("wall_s"),
                "peak_memory_allocated_bytes": d.get("peak_memory_allocated_bytes"),
            }
        report["cells"] = cells
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write(f"DeltaNet-RD Track C (scaling ladder, rung {RUNG}) -- wave {wave}\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def _run_wave(wave, manifest, out_dir, args, is_done_fn, build_cmd_fn, timeout_fn):
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir, args.gpu_offset):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate failed; no training launched.\n")
        sys.exit(1)

    physical_gpus = list(range(args.gpu_offset, args.gpu_offset + args.gpus))
    slots = [g for _ in range(args.per_gpu) for g in physical_gpus]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done_fn(out_dir, s)]
    print(f"wave={wave}  manifest={len(manifest)}  pending={len(pending)}  "
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

    all_done = (done_ct == len(manifest)) and not failed
    if all_done:
        with open(os.path.join(out_dir, "ALL_DONE"), "w") as f:
            f.write(f"wave {wave} complete: {done_ct}/{len(manifest)} runs, 0 failed\n")
    print(f"\nWAVE {wave} DONE. {done_ct} succeeded, {len(failed)} failed. "
          f"ALL_DONE {'written' if all_done else 'NOT written (wave incomplete)'}.", flush=True)
    return all_done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/lm_rd_trackc"))
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--wave", choices=["calibration", "1", "4"], default=None,
                     help="REQUIRED unless --dry-run. 'calibration' (Wave -1) MAY be launched this "
                          "session; '1' is built/gated but NOT launched this session; '4' is ALWAYS "
                          "refused (Track B's own NO-LAUNCH gates it out, sec 5.5 item 3).")
    ap.add_argument("--gpus", type=int, default=None, help="GPU COUNT. REQUIRED for a real launch, "
                                                              "NO DEFAULT -- check nvidia-smi first.")
    ap.add_argument("--gpu-offset", type=int, default=None, help="first physical GPU index. REQUIRED "
                                                                    "for a real launch, NO DEFAULT.")
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--calibration-steps", type=int, default=CALIBRATION_STEPS_DEFAULT)
    ap.add_argument("--calibration-ckpt-every", type=int, default=CALIBRATION_CKPT_EVERY)
    ap.add_argument("--rung1-steps", type=int, default=None,
                     help="Wave 1 only: REQUIRED for a real launch (no silent fallback to the "
                          "uncalibrated placeholder) -- supply a value informed by calibration.json.")
    ap.add_argument("--control-steps", type=int, default=None,
                     help="Wave 1 only: default derived from CONTROL_TARGET_TOKENS (matches Wave C exactly).")
    ap.add_argument("--trackb-gate-json", default=TRACKB_GATE_JSON_DEFAULT)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    if args.dry_run:
        calib_m = calibration_manifest(args.calibration_steps, args.calibration_ckpt_every)
        rung1_steps = args.rung1_steps or default_rung1_steps_placeholder()
        control_steps = args.control_steps or default_control_steps()
        w1_m = (make_manifest("w1_rung1", RUNG_CFG, MIX_CORPORA, SEEDS, rung1_steps, 1000)
                + make_manifest("w1_control", CONTROL_CFG, MIX_CORPORA, SEEDS, control_steps, 1000))
        print(f"rung {RUNG} config: {RUNG_CFG} (verified 0.4% off target this session, see "
              f"lm_rd_rung_configs.py / lm_pretrain_rd.py smoke item [11])")
        print(f"control config: {CONTROL_CFG} (Wave C's own scale)")
        print(f"\ncalibration (Wave -1): {len(calib_m)} runs "
              f"({args.calibration_steps} steps each, ckpt_every={args.calibration_ckpt_every}) -- "
              f"MAY be launched this session (task brief: cheap, <2 GPU-h)")
        print(f"\nwave 1: {len(w1_m)} runs -- rung-1 {rung1_steps} steps"
              f"{' (UNCALIBRATED PLACEHOLDER, pass --rung1-steps after calibration)' if not args.rung1_steps else ''}"
              f" x {len(MIX_CORPORA)} corpora x {len(SEEDS)} seeds, "
              f"control {control_steps} steps (matches Wave C exactly) x {len(MIX_CORPORA)} x {len(SEEDS)} "
              f"-- NOT LAUNCHED this session (task brief); refuses a real launch without an existing "
              f"calibration.json (sec 9)")
        trackb = _trackb_gate_status(args.trackb_gate_json)
        print(f"\nwave 4 (fix-effect / geo3-at-scale): ALWAYS REFUSED. Track B gate status: {trackb}")
        if args.gpus is not None and args.gpu_offset is not None:
            print(f"\nslots = {args.gpus} gpus x {args.per_gpu} per-gpu, physical GPUs "
                  f"{list(range(args.gpu_offset, args.gpu_offset + args.gpus))}")
        else:
            print("\nslots: pass --gpus/--gpu-offset to preview (REQUIRED for a real launch).")
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "4":
        refuse_wave4(args.trackb_gate_json)   # always exits non-zero, never returns

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose) -- run nvidia-smi NOW and pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "calibration":
        manifest = calibration_manifest(args.calibration_steps, args.calibration_ckpt_every)
        out_dir = os.path.join(args.out_dir, "calibration")
        os.makedirs(out_dir, exist_ok=True)
        timeout_fn = lambda spec: default_timeout_pretrain(
            spec["steps"], spec["ckpt_every"], _CALIBRATION_PER_STEP_S_GENEROUS, _CALIBRATION_PER_CKPT_S_GENEROUS)
        all_done = _run_wave("calibration", manifest, out_dir, args, is_done_cell, build_cmd_cell, timeout_fn)
        calib_summary = summarize_calibration(out_dir, manifest)
        with open(os.path.join(out_dir, "calibration.json"), "w") as f:
            json.dump(calib_summary, f, indent=2)
        print("\nCALIBRATION SUMMARY:", json.dumps(calib_summary, indent=2))
        if all_done:
            # also drop a copy at the wave-1 gate's expected lookup location (out-dir root) so
            # --wave 1's real-launch check (sec 9) finds it without extra plumbing
            with open(os.path.join(args.out_dir, "calibration.json"), "w") as f:
                json.dump(calib_summary, f, indent=2)
        return

    # --wave 1
    calibration_json_path = os.path.join(args.out_dir, "calibration.json")
    if not os.path.exists(calibration_json_path):
        print(f"ERROR: {calibration_json_path} not found -- sec 9's own hard rule: 'No track's Wave "
              f"1+ manifest is authorized to launch ... without first recording its Wave -1 measured "
              f"numbers.' Run --wave calibration first.", file=sys.stderr)
        sys.exit(2)
    if args.rung1_steps is None:
        print("ERROR: --rung1-steps is REQUIRED for a real --wave 1 launch (no silent fallback to "
              "the uncalibrated RUNG1_TARGET_TOKENS_PLACEHOLDER) -- derive it from "
              f"{calibration_json_path}'s measured tok/s and pass it explicitly.", file=sys.stderr)
        sys.exit(2)
    print("=" * 70, file=sys.stderr)
    print("TASK-BRIEF GUARD: this build session was instructed NOT to launch the full rung runs. "
          "This launcher is built, smoke-gated, and its manifest is correct, but a real --wave 1 "
          "invocation is refused here pending explicit human re-authorization outside this session.",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.exit(4)


if __name__ == "__main__":
    main()
