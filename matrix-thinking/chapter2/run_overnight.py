"""Perpetual Task D queue — packs multiple runs per GPU, auto-refills forever.

Keeps the 8xH100 cluster saturated (~100%): each tiny run only uses ~1GB / ~65%
of one H100, so we pack --per-gpu runs onto each GPU, and when the queue runs low
we refill with a fresh-seed round. Runs until a STOP sentinel appears
(results/overnight/STOP) or --forever is off.

Priority: Tier 1 (d=16 decisive gate, M3/M1 interleaved) -> Tier 2 (generality
d=8/32/64 + K>d lossy) -> Tier 3 (d=128). Each refill round re-runs the grid with
new seeds (tighter statistics + seed-variance, which this project cares about).

Robustness (rank_aware_v1 lessons): smoke-gate abort; per-run subprocess isolation;
exception-isolated launch; validity-checked resume; per-d timeout with GPU quarantine
on wedged reap; guarded progress/aggregate; top-level crash handler (CRASHED.txt).

Uses ONLY audited run_task_d.py code paths (small default model; NaN-robust train).

Usage:
  python run_overnight.py --dry-run
  python run_overnight.py --out-dir results/overnight --gpus 8 --per-gpu 4 --forever
Stop gracefully:  touch results/overnight/STOP
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_task_d.py")
TAU = "recovered_frac@0.9"

SEEDS_M1 = (0, 1, 2, 3, 4, 5, 6, 7)
SEEDS_M3 = (0, 1, 2, 3, 4)
SEEDS_T2 = (0, 1, 2, 3)
SEEDS_T3 = (0, 1, 2)
STEPS = {8: 8000, 16: 8000, 32: 8000, 64: 10000, 128: 10000}
TIMEOUT = {8: 1800, 16: 1800, 32: 3000, 64: 5400, 128: 7200}


def straddle(K, d):
    xs = {1, 2, max(1, K - 2), max(1, K - 1), K, min(d, K + 1), min(d, K + 2), d}
    return sorted(x for x in xs if 1 <= x <= d)


def _spec(tier, d, K, fr, seed, orthogonal):
    tag = f"t{tier}_d{d}_K{K}_fr{'N' if fr is None else fr}_s{seed}"
    return {"tier": tier, "d": d, "K": K, "force_rank_k": fr, "seed": seed,
            "orthogonal": orthogonal, "steps": STEPS[d], "name": tag}


def _interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def build_manifest(seed_offset=0):
    """One full grid. seed_offset shifts all seeds so refill rounds use new seeds
    (and thus new run names -> not skipped by resume)."""
    def so(seeds):
        return tuple(s + seed_offset for s in seeds)
    # Tier 1: d=16 gate, M3 (primary causal) interleaved with M1
    d = 16
    t1_m1 = [_spec(1, d, K, None, s, True)
             for K in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16) for s in so(SEEDS_M1)]
    t1_m3 = [_spec(1, d, K, fr, s, True)
             for K in (8, 4, 12) for fr in straddle(K, d) for s in so(SEEDS_M3)]
    tier1 = _interleave(t1_m3, t1_m1)
    # Tier 2: generality
    tier2 = []
    for d in (8, 32, 64):
        for K in [k for k in (1, 2, 4, 8, 16, 24, 32, 48, 64) if k <= d]:
            for s in so(SEEDS_T2):
                tier2.append(_spec(2, d, K, None, s, True))
        Km = d // 2
        for fr in straddle(Km, d):
            for s in so(SEEDS_T2):
                tier2.append(_spec(2, d, Km, fr, s, True))
    for K in (20, 24, 32):
        for s in so(SEEDS_T2):
            tier2.append(_spec(2, 16, K, None, s, False))
    # Tier 3: d=128
    d = 128
    tier3 = [_spec(3, d, K, None, s, True)
             for K in (8, 16, 32, 64, 96, 128) for s in so(SEEDS_T3)]
    tier3 += [_spec(3, d, 64, fr, s, True)
              for fr in straddle(64, d) for s in so(SEEDS_T3)]
    return tier1 + tier2 + tier3


def out_path(out_dir, spec):
    return os.path.join(out_dir, f"{spec['name']}.json")


def is_done(out_dir, spec):
    p = out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return isinstance(d, dict) and "mean_cos" in d and "effective_rank_mean" in d
    except Exception:
        return False


def build_cmd(spec, out_dir):
    cmd = [sys.executable, RUN, "--model", "matrix",
           "--d", str(spec["d"]), "--K", str(spec["K"]),
           "--steps", str(spec["steps"]), "--seed", str(spec["seed"]),
           "--out", out_path(out_dir, spec)]
    cmd.append("--orthogonal" if spec["orthogonal"] else "--gaussian")
    if spec["force_rank_k"] is not None:
        cmd += ["--force-rank-k", str(spec["force_rank_k"])]
    return cmd


def run_smoke(log_dir):
    print("SMOKE GATE (GPU 0) ...", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    with open(os.path.join(log_dir, "smoke.log"), "w") as lf:
        rc = subprocess.call([sys.executable, RUN, "--smoke"], env=env,
                             stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    print("smoke passed." if rc == 0 else f"SMOKE FAILED (rc={rc}) -- ABORTING.", flush=True)
    return rc == 0


def write_progress(out_dir, done, failed, running, round_idx):
    try:
        with open(os.path.join(out_dir, "PROGRESS.txt"), "w") as f:
            f.write(f"round={round_idx} done={done} failed={failed} running={running}\n")
    except Exception as e:
        print(f"  (write_progress non-fatal: {e!r})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/overnight"))
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--per-gpu", type=int, default=4, help="runs packed per GPU")
    ap.add_argument("--forever", action="store_true", help="refill with new seeds until STOP")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    if args.dry_run:
        from collections import Counter
        m = build_manifest(0)
        print(f"one round: {len(m)} runs | by tier {dict(sorted(Counter(s['tier'] for s in m).items()))} "
              f"| by d {dict(sorted(Counter(s['d'] for s in m).items()))}")
        print(f"slots = {args.gpus} gpus x {args.per_gpu} per-gpu = {args.gpus*args.per_gpu} concurrent")
        print("tier-1 head:", [s['name'] for s in m[:6]])
        return

    out_dir = args.out_dir
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stop_file = os.path.join(out_dir, "STOP")

    if not args.skip_smoke and not run_smoke(log_dir):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate failed; no training launched.\n")
        sys.exit(1)

    slots = [g for _ in range(args.per_gpu) for g in range(args.gpus)]  # gpu per slot
    n_slots = len(slots)
    print(f"perpetual={args.forever}  slots={n_slots} ({args.gpus}x{args.per_gpu})", flush=True)

    round_idx = 0
    pending = [s for s in build_manifest(0) if not is_done(out_dir, s)]
    running = {}          # uid -> (proc, spec, lf, start, gpu)
    free = list(slots)    # available gpu-slots
    quarantined = []
    done_ct = 0
    failed = []
    uid = 0
    last_agg = time.time()

    try:
        while pending or running:
            # refill to keep the pipeline full (perpetual)
            if (args.forever and not os.path.exists(stop_file)
                    and len(pending) < 2 * n_slots):
                round_idx += 1
                more = [s for s in build_manifest(round_idx * 1000) if not is_done(out_dir, s)]
                pending.extend(more)
                print(f"  REFILL round {round_idx}: +{len(more)} runs "
                      f"(pending now {len(pending)})", flush=True)
            # launch on every free slot (exception-isolated)
            while free and pending:
                gpu = free.pop()
                spec = pending.pop(0)
                try:
                    lf = open(os.path.join(log_dir, f"{spec['name']}.log"), "w")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    proc = subprocess.Popen(build_cmd(spec, out_dir), env=env,
                                            stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
                    running[uid] = (proc, spec, lf, time.time(), gpu)
                    uid += 1
                except Exception as e:
                    print(f"  LAUNCH-FAILED {spec['name']}: {e!r}", flush=True)
                    failed.append((spec["name"], "LAUNCH_ERR"))
                    free.append(gpu)
            # harvest
            for u, (proc, spec, lf, start, gpu) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    if time.time() - start > TIMEOUT.get(spec["d"], 3600):
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
            write_progress(out_dir, done_ct, len(failed), len(running), round_idx)
            # periodic aggregate checkpoint (so SUMMARY updates during long runs)
            if time.time() - last_agg > 120:
                aggregate(out_dir, failed)
                last_agg = time.time()
            # deadlock guard
            if pending and not running and not free:
                print(f"  ABORT: no usable GPUs ({len(quarantined)} quarantined)", flush=True)
                break
            # graceful stop: STOP appeared and nothing left to drain-launch
            if os.path.exists(stop_file) and not args.forever:
                pass
            time.sleep(args.poll)
    except Exception as e:
        import traceback
        try:
            with open(os.path.join(out_dir, "CRASHED.txt"), "w") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        print(f"ORCHESTRATOR CRASHED: {e!r} -- see CRASHED.txt; rerun to resume.", flush=True)
    finally:
        aggregate(out_dir, failed)
    print(f"\nSTOPPED. {done_ct} succeeded, {len(failed)} failed, {round_idx} rounds.", flush=True)


def aggregate(out_dir, failed):
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    report = {"n_failed": len(failed)}
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
        m1 = {}
        for r in runs:
            if r.get("force_rank_k") is None and "effective_rank_mean" in r:
                m1.setdefault(r["d"], {}).setdefault(r["K"], []).append(r["effective_rank_mean"])
        report["M1_effrank_vs_K_by_d"] = {
            d: {K: mean(v) for K, v in sorted(ks.items())} for d, ks in sorted(m1.items())}
        m3 = {}
        for r in runs:
            if r.get("force_rank_k") is not None and TAU in r:
                m3.setdefault((r["d"], r["K"]), {}).setdefault(r["force_rank_k"], []).append(r[TAU])
        report["M3_" + TAU + "_vs_forcerank"] = {
            f"d{d}_K{K}": {fr: mean(v) for fr, v in sorted(frs.items())}
            for (d, K), frs in sorted(m3.items())}
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write("Task D — perpetual sweep\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
