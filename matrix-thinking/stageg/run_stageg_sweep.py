"""Stage G orchestrator — see STAGE_G_DESIGN.md section 7 (manifest) and
section 12 (build requirements).

CLONE of matrix-thinking/chapter2/run_stage0_sweep.py's robustness pattern
(smoke gate, exception-isolated launch, validity-checked resume, per-run
timeout with GPU quarantine, guarded aggregate, top-level crash handler),
adapted for Stage G's bounded waves:

  --wave {-1,0,A,B}   -1 = Wave -1's GPU calibration part ONLY (the
                       zero-GPU items live in wave_neg1.py, run
                       separately and first). 0 = Wave 0-R2 (the
                       multi-seed H_d + gap-baseline cells; Wave 0-R1 is
                       NOT built here -- N1's fallback is the MAINLINE
                       planning assumption, so there is no locally-
                       recoverable Run-12 checkpoint to warm-continue from;
                       see wave_neg1.py's checkpoint-recoverability check
                       and the design's section 4 item 2(a) aftermath
                       paragraph). A = the 5-cell OFAT screen. B = the
                       top-1/2 confirmation, REQUIRES --winner (human
                       gate, exactly mirroring Stage 0's --winner
                       requirement).
  NO perpetual refill -- each wave's manifest is fixed and finite; the
  orchestrator drains it once and exits.
  Manifest naming carries family+variant+seed+steps (mirrors Stage 0's
  MAJOR-2 fix) so no two cells collide on the same output path.
  is_done is validity-checked AND cross-checks family/variant/seed/steps
  recorded INSIDE the result JSON (train_stageg.py's `complete` +
  `steps_completed` sentinel, identical discipline to Stage 0's FATAL-1
  fix) against the spec that would produce that path.

Usage:
  python run_stageg_sweep.py --dry-run
  python run_stageg_sweep.py --wave -1 --out-dir results/stageg
  python run_stageg_sweep.py --wave 0  --out-dir results/stageg --standard-steps 3000
  python run_stageg_sweep.py --wave A  --out-dir results/stageg --standard-steps 3000
  python run_stageg_sweep.py --wave B  --out-dir results/stageg --winner h_j_init_matched --standard-steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "train_stageg.py")

sys.path.insert(0, HERE)
from models import VARIANT_AXES, WAVE_A_SCREEN_AXES, WAVE_B_MIRROR

# Provisional GPU-h units (design section 5.4, N5): matrix-side standard
# cell = 1.0 GPU-h flat, vector-side = 0.25 GPU-h flat, AT THE PROVISIONAL
# 3,000-step standard cell -- Wave -1's GPU calibration (this orchestrator's
# --wave -1) is what PINS the real number (N5: "frozen before Wave 0
# launches"); these are only the pre-calibration defaults used to size
# --timeout before real numbers exist.
PROVISIONAL_S_PER_STEP = {"matrix": 1.0 * 3600 / 3000, "vector": 0.25 * 3600 / 3000}


def _spec(wave, family, variant, mat_dim, steps, seed, tag_extra=""):
    tag = f"w{wave}_{family}_{variant}_d{mat_dim}_steps{steps}_s{seed}{tag_extra}"
    return {"wave": str(wave), "family": family, "variant": variant, "mat_dim": mat_dim,
            "steps": steps, "seed": seed, "name": tag}


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def wave_neg1_gpu_manifest(mat_dim=32, calib_steps=500):
    """Live timing calibration (design section 7's Wave -1 GPU row): 2
    cells (matrix + vector baselines, ~500 steps), whose mandatory output
    is pinning the Regime-2 standard-cell step count (N5) before Wave 0
    launches -- read wall_s/steps_completed from these two results and
    set --standard-steps for every later wave accordingly."""
    return [
        _spec(-1, "matrix", "baseline", mat_dim, calib_steps, 0),
        _spec(-1, "vector", "baseline", mat_dim, calib_steps, 0),
    ]


def wave0r2_manifest(mat_dim=32, standard_steps=3000):
    """Wave 0-R2 (design section 7): H_d multi-seed + gap-baseline pair.
    10 cells: extended (3x) matrix x2 seeds + extended (3x) vector x2
    seeds + matched-budget (1x) gap-baseline matrix x3 seeds + vector x3
    seeds. NOTE: Wave 0-R1 (the Regime-1 warm-continuation) is NOT built
    here -- N1's fallback (checkpoint not locally recoverable, re-verified
    by wave_neg1.py's checkpoint_recovery check) is the mainline; that
    datum is out-of-budget per design section 4 item 2(a)."""
    extended = standard_steps * 3
    runs = []
    for s in range(2):
        runs.append(_spec(0, "matrix", "baseline", mat_dim, extended, s, tag_extra="_extended"))
    for s in range(2):
        runs.append(_spec(0, "vector", "baseline", mat_dim, extended, s, tag_extra="_extended"))
    for s in range(3):
        runs.append(_spec(0, "matrix", "baseline", mat_dim, standard_steps, s, tag_extra="_gapbaseline"))
    for s in range(3):
        runs.append(_spec(0, "vector", "baseline", mat_dim, standard_steps, s, tag_extra="_gapbaseline"))
    return runs


def waveA_manifest(mat_dim=32, standard_steps=3000):
    """5-cell OFAT screen (design section 6.1/7): one single-axis swap per
    cell, matrix-all baseline otherwise, 1 seed, budget-matched (standard
    cell) checkpoint scoring. Built from WAVE_A_SCREEN_AXES, NOT from
    VARIANT_AXES -- the registry also carries conditional follow-up
    variants (h_f_tied_matched_init, the pre-registered N3 rerun) that are
    launchable via train_stageg.py --variant but are never screen cells."""
    axes = list(WAVE_A_SCREEN_AXES)
    assert len(axes) == 5, f"expected 5 Wave-A axes, got {len(axes)}: {axes}"
    assert all(v in VARIANT_AXES for v in axes)
    return [_spec("A", "matrix", v, mat_dim, standard_steps, 0) for v in axes]


def waveB_manifest(winners, mat_dim=32, standard_steps=3000):
    """Top-1/2 confirmation x3 seeds (<=6 matrix cells) + bidirectional
    vector-all mirror for ONE winning axis (3 vector cells; design section
    7: "bidirectional vector-all-baseline mirror for the winning axis").
    `winners`: list of 1-2 matrix variant names (the human's read of Wave
    A's results -- REQUIRED, no silent default, mirrors Stage 0's
    --winner gate)."""
    steps2x = standard_steps * 2
    runs = []
    for v in winners[:2]:
        for s in range(3):
            runs.append(_spec("B", "matrix", v, mat_dim, steps2x, s, tag_extra="_confirm"))
    mirror = WAVE_B_MIRROR.get(winners[0])
    if mirror is None:
        print(f"NOTE: winner '{winners[0]}' has no vector-side Wave-B mirror "
              f"(models.py WAVE_B_MIRROR; H_g's depth-match axis has none by construction -- "
              f"design section 6.1). Skipping the vector-mirror cells for this winner.")
    else:
        for s in range(3):
            runs.append(_spec("B", "vector", mirror, mat_dim, steps2x, s, tag_extra="_mirror"))
    return runs


MANIFEST_FNS_NOARG = {}


# ---------------------------------------------------------------------------
# Resume / launch machinery
# ---------------------------------------------------------------------------

def out_path(out_dir, spec):
    return os.path.join(out_dir, f"{spec['name']}.json")


def is_done(out_dir, spec):
    """Validity-checked resume, cross-checked against the spec's own
    identity fields (family/variant/seed/steps/mat_dim) -- closes the same
    is_done weakness run_stage0_sweep.py documents and fixes."""
    p = out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return False
        required = ("family", "variant", "mat_dim", "steps", "seed", "complete", "steps_completed")
        if not all(k in d for k in required):
            return False
        if d.get("timed_out"):
            return False
        if d.get("complete") is not True:
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if (d.get("family") != spec["family"] or d.get("variant") != spec["variant"]
                or d.get("mat_dim") != spec["mat_dim"] or d.get("seed") != spec["seed"]
                or d.get("steps") != spec["steps"]):
            return False
        return True
    except Exception:
        return False


def build_cmd(spec, out_dir, timeout, data_path, max_len, vocab_size, n_iterations, seq_len, batch_size):
    cmd = [sys.executable, RUN, "--family", spec["family"], "--variant", spec["variant"],
           "--mat-dim", str(spec["mat_dim"]), "--max-len", str(max_len),
           "--vocab-size", str(vocab_size), "--n-iterations", str(n_iterations),
           "--steps", str(spec["steps"]), "--seed", str(spec["seed"]),
           "--seq-len", str(seq_len), "--batch-size", str(batch_size),
           "--internal-timeout", str(max(1, timeout - 30)),
           "--out", out_path(out_dir, spec), "--device", "cuda"]
    if data_path:
        cmd += ["--data-path", data_path]
    return cmd


def default_timeout(steps, family, margin=1.6):
    return int(steps * PROVISIONAL_S_PER_STEP[family] * margin) + 120  # +eval overhead margin


def run_smoke(log_dir, smoke_gpu=0):
    # --smoke-gpu (audit MINOR-2): the gate GPU is configurable because
    # GPU 0 is not guaranteed free on a shared box -- check nvidia-smi and
    # point this at an idle one.
    print(f"SMOKE GATE (GPU {smoke_gpu}) ...", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(smoke_gpu)}
    with open(os.path.join(log_dir, "smoke.log"), "w") as lf:
        rc = subprocess.call([sys.executable, os.path.join(HERE, "test_stageg.py")], env=env,
                              stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    print("smoke passed." if rc == 0 else f"SMOKE FAILED (rc={rc}) -- ABORTING.", flush=True)
    return rc == 0


def write_progress(out_dir, done, failed, running, wave):
    try:
        with open(os.path.join(out_dir, "PROGRESS.txt"), "w") as f:
            f.write(f"wave={wave} done={done} failed={failed} running={running}\n")
    except Exception as e:
        print(f"  (write_progress non-fatal: {e!r})", flush=True)


def aggregate(out_dir, failed, wave):
    report = {"wave": wave, "n_failed": len(failed)}
    try:
        runs = []
        for fn in os.listdir(out_dir):
            if fn.endswith(".json") and fn != "AGGREGATE.json":
                try:
                    with open(os.path.join(out_dir, fn)) as f:
                        runs.append(json.load(f))
                except Exception:
                    pass
        report["n_runs"] = len(runs)
        complete_runs = [r for r in runs if r.get("complete") is True]
        report["n_partial_excluded"] = len(runs) - len(complete_runs)
        by_cell = {}
        for r in complete_runs:
            try:
                key = f"{r['family']}_{r['variant']}_steps{r['steps']}"
                cell = by_cell.setdefault(key, {"final_bpb_T8": [], "step0_entry_std": []})
                fe = r.get("final_evals", {}).get("T8", {})
                if "val_bpb" in fe:
                    cell["final_bpb_T8"].append(fe["val_bpb"])
                if r.get("step0_entry_std") is not None:
                    cell["step0_entry_std"].append(r["step0_entry_std"])
            except Exception:
                continue

        def mean(xs):
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 4) if xs else None
        report["by_cell"] = {k: {mk: mean(v) for mk, v in vv.items()} for k, vv in sorted(by_cell.items())}
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write(f"Stage G -- wave {wave}\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "stageg"))
    ap.add_argument("--wave", choices=["-1", "0", "A", "B"], default=None,
                     help="which wave to launch; REQUIRED unless --dry-run.")
    ap.add_argument("--winner", type=str, default=None,
                     help="Wave B only, REQUIRED: comma-separated 1-2 Wave-A winning variant names.")
    ap.add_argument("--mat-dim", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--n-iterations", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--standard-steps", type=int, default=3000,
                     help="the Regime-2 standard-cell step count -- PROVISIONAL until Wave -1's "
                          "GPU calibration pins it (N5); pass the calibrated value explicitly "
                          "for Waves 0/A/B.")
    ap.add_argument("--calib-steps", type=int, default=500, help="Wave -1 GPU calibration step count")
    ap.add_argument("--data-path", type=str, default=None)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0,
                    help="first physical GPU index to use (mirrors the audited "
                         "deltanet sweep fix; lets waves share a box with other "
                         "running work, e.g. --gpu-offset 3 --gpus 2)")
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-gpu", type=int, default=0,
                     help="GPU index for the pre-launch smoke gate (audit MINOR-2: GPU 0 may be "
                          "busy on a shared box -- check nvidia-smi first).")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    def build_manifest(wave):
        if wave == "-1":
            return wave_neg1_gpu_manifest(args.mat_dim, args.calib_steps)
        if wave == "0":
            return wave0r2_manifest(args.mat_dim, args.standard_steps)
        if wave == "A":
            return waveA_manifest(args.mat_dim, args.standard_steps)
        if wave == "B":
            winners = [w.strip() for w in (args.winner or "").split(",") if w.strip()] or ["h_j_init_matched"]
            return waveB_manifest(winners, args.mat_dim, args.standard_steps)
        raise ValueError(wave)

    if args.dry_run:
        from collections import Counter
        total = 0
        for w in ("-1", "0", "A", "B"):
            runs = build_manifest(w)
            total += len(runs)
            print(f"wave {w}: {len(runs)} runs | by (family,variant) "
                  f"{dict(sorted(Counter((s['family'], s['variant']) for s in runs).items()))}")
        print(f"CORE TOTAL: {total} runs  (design section 7 budget table: 2 + 10 + 5 + <=9 = <=26)")
        print(f"slots = {args.gpus} gpus x {args.per_gpu} per-gpu = {args.gpus * args.per_gpu} concurrent")
        if args.winner is None:
            print("NOTE: --winner not given -- Wave B used a PLACEHOLDER variant for "
                  "manifest-SHAPE purposes only; a real launch requires it.")
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)
    if args.wave == "B" and not args.winner:
        print("ERROR: --wave B requires --winner (the human gate: 1-2 confirmed Wave-A "
              "winning variant names, comma-separated).", file=sys.stderr)
        sys.exit(1)

    manifest = build_manifest(args.wave)
    out_dir = os.path.join(args.out_dir, f"wave{args.wave}")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir, smoke_gpu=args.smoke_gpu):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate (test_stageg.py) failed; no training launched.\n")
        sys.exit(1)

    slots = [g + args.gpu_offset
             for _ in range(args.per_gpu) for g in range(args.gpus)]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done(out_dir, s)]
    print(f"wave={args.wave}  manifest={len(manifest)}  pending={len(pending)}  "
          f"slots={n_slots} ({args.gpus}x{args.per_gpu})", flush=True)

    running = {}
    free = list(slots)
    quarantined = []
    done_ct = 0
    failed = []
    uid = 0
    last_agg = time.time()

    try:
        while pending or running:
            while free and pending:
                gpu = free.pop()
                spec = pending.pop(0)
                timeout = args.timeout if args.timeout is not None else default_timeout(spec["steps"], spec["family"])
                try:
                    lf = open(os.path.join(log_dir, f"{spec['name']}.log"), "w")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    cmd = build_cmd(spec, out_dir, timeout, args.data_path, args.max_len,
                                     args.vocab_size, args.n_iterations, args.seq_len, args.batch_size)
                    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
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
                        if reaped:
                            free.append(gpu)
                        else:
                            quarantined.append(gpu)
                            print(f"  QUARANTINE gpu {gpu}: reap timed out", flush=True)
                        failed.append((spec["name"], "TIMEOUT"))
                    continue
                try:
                    lf.close()
                except Exception:
                    pass
                del running[u]
                free.append(gpu)
                if rc == 0 and is_done(out_dir, spec):
                    done_ct += 1
                else:
                    failed.append((spec["name"], rc))
            write_progress(out_dir, done_ct, len(failed), len(running), args.wave)
            if time.time() - last_agg > 120:
                aggregate(out_dir, failed, args.wave)
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
        print(f"ORCHESTRATOR CRASHED: {e!r} -- see CRASHED.txt; rerun --wave {args.wave} to "
              f"resume (validity-checked, bounded, not perpetual).", flush=True)
    finally:
        aggregate(out_dir, failed, args.wave)
    print(f"\nWAVE {args.wave} DONE. {done_ct} succeeded, {len(failed)} failed.", flush=True)
    print("HUMAN GATE: review results before launching the next wave "
          "(Wave A -> read the OFAT survivors -> supply --winner for Wave B).", flush=True)


if __name__ == "__main__":
    main()
