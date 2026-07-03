"""Perpetual Task E queue — packs multiple runs per GPU, auto-refills forever.

CLONE of run_overnight.py (Task D's audited orchestrator), adapted for
run_task_e.py's compositional multi-hop sweep (NEXT_EXPERIMENT_DESIGN.md
section 6 (measurements) + section 8 (compute plan)). All d=16 (Stage 2's
guaranteed-trainable operating point), so runs are tiny (~1GB, well under one
H100) -- same packing story as Task D, --per-gpu 4 default.

Priority: Tier 1 (PRIMARY, d=16 gate -- M1_E/M2_E/M3_E interleaved with M4_E
force-rank-k causal test) -> Tier 2 (controls: C_MLP floor + chain-variant
robustness). Each refill round re-runs the grid with new seeds (tighter
statistics + seed-variance, same rationale as run_overnight.py).

Robustness (identical to run_overnight.py -- cloned, not reinvented):
smoke-gate abort; per-run subprocess isolation; exception-isolated launch;
validity-checked resume; per-run timeout with GPU quarantine on wedged reap;
guarded progress/aggregate; top-level crash handler (CRASHED.txt); perpetual
seed-offset refill.

Uses ONLY audited run_task_e.py code paths (--h-train/test/extra defaults are
NOT overridden -- NEXT_EXPERIMENT_DESIGN.md section 6's fixed operating point
and the periodicity guard in TaskEConfig.__post_init__ both depend on them).

Usage:
  python run_task_e_sweep.py --dry-run
  python run_task_e_sweep.py --out-dir results/task_e_sweep --gpus 8 --per-gpu 4 --forever
Stop gracefully:  touch results/task_e_sweep/STOP
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_task_e.py")
TAU = "recovered_frac@0.9"
IDEAL_TAU = "ideal_" + TAU

# Stage-2 d=16 gate (NEXT_EXPERIMENT_DESIGN.md section 6/8): fixed d, long
# multi-hop curriculum, orthonormal gate default always on. h-train/test/extra
# are NEVER overridden here -- they stay at run_task_e.py's audited defaults
# ({1,2,3}/{4,5,6}/{7,21}).
#
# 2026-07-01 calibration finding (20K-step round 0): training exhibits a late,
# seed-stochastic phase transition (loss flat ~1.0 until ~step 12.5K, then
# sharp collapse); 20K steps truncates it mid-flight (1/5 seeds converged at
# K=8, 0/5 at K=12, fr grid all-zero). STEPS/TIMEOUT are therefore CLI-settable
# (--steps/--timeout); the 40K calibration relaunch must use a FRESH --out-dir
# (run names don't encode steps, so is_done would silently skip against old
# 20K results).
D = 16
STEPS = 20000                 # overridden by --steps in main()
TIMEOUT = {16: 3600}          # overridden by --timeout in main()

SEEDS_M = (0, 1, 2, 3, 4)     # Tier 1: M1_E/M2_E/M3_E + M4_E (>=5 seeds, per design doc)
SEEDS_T2 = (0, 1, 2)          # Tier 2: controls/robustness

K_GRID = (8, 12, 16)          # K=4 dropped 2026-07-01 (fully confounded, see task_e.py)
K_M4 = 8                      # M4_E's primary operating point
FR_GRID_M4 = (2, 4, 6, 7, 8, 9, 10)   # straddle grid around K=8


def _spec(tier, model, variant, K, fr, seed):
    tag = (f"t{tier}_{model}_{variant}_K{K}_fr{'N' if fr is None else fr}_s{seed}")
    return {"tier": tier, "model": model, "variant": variant, "d": D, "K": K,
            "force_rank_k": fr, "seed": seed, "steps": STEPS, "orthogonal": True,
            "name": tag}


def _interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def build_manifest(seed_offset=0, calibrate=False):
    """One full grid. seed_offset shifts all seeds so refill rounds use new
    seeds (and thus new run names -> not skipped by resume), same trick as
    run_overnight.py's build_manifest.

    calibrate=True builds the reduced 40K-step calibration round (2026-07-01,
    per the late-phase-transition finding): all unconstrained (frN) Tier-1
    cells at full seeds, but only a 3-seed probe of fr in {7,8,9} (the cells
    the 20K round showed beginning to transition) instead of the full M4 grid.
    The full fr grid is gated on this calibration converging (>=3/5 seeds at
    K=8 frN) -- the mandatory calibration-run discipline from CLAUDE.md."""
    def so(seeds):
        return tuple(s + seed_offset for s in seeds)

    # Tier 1 (PRIMARY): M1_E+M2_E+M3_E (unconstrained rank, K sweep) interleaved
    # with M4_E (force-rank-k causal test at the K=8 operating point).
    t1_m1 = [_spec(1, "matrix", "permutation", K, None, s)
             for K in K_GRID for s in so(SEEDS_M)]
    fr_grid = (7, 8, 9) if calibrate else FR_GRID_M4
    fr_seeds = SEEDS_M[:3] if calibrate else SEEDS_M
    t1_m4 = [_spec(1, "matrix", "permutation", K_M4, fr, s)
             for fr in fr_grid for s in so(fr_seeds)]
    tier1 = _interleave(t1_m1, t1_m4)

    # Tier 2: C_MLP floor (rank-blind shortcut baseline) + chain-variant
    # robustness (matrix model, "chain" edge function instead of "permutation").
    tier2_mlp = [_spec(2, "mlp", "permutation", K, None, s)
                 for K in K_GRID for s in so(SEEDS_T2)]
    # chain-variant robustness DEFERRED (audit 2026-07-01): at d=16 the acyclic
    # chain requires K >= H_max=21 (default H_extra=(7,21)) and N=K+4 <= d=16, so
    # every K in {8,12,16} is rejected by run_task_e.py's own preflight. Re-add as
    # a follow-on with a K-scaled --h-extra/--N override. Headline (Tier 1 matrix
    # + C_MLP floor) is unaffected.
    tier2 = tier2_mlp

    return tier1 + tier2


def out_path(out_dir, spec):
    return os.path.join(out_dir, f"{spec['name']}.json")


def is_done(out_dir, spec):
    """Validity-checked resume (run_overnight.py's is_done, adapted to Task
    E's nested per-hop schema: run_task_e.py's result dict has no top-level
    'mean_cos'/'effective_rank_mean' -- those live inside each hop entry of
    M2_in_distribution / M3_held_out). Requires: valid JSON, the run-identity
    fields Task E always records, both hop dicts non-empty, and the PRIMARY
    decision metric (+ its C7 ideal counterpart) present on every hop entry."""
    p = out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return False
        if not all(k in d for k in ("model_type", "variant", "d", "K", "seed",
                                     "steps", "force_rank_k")):
            return False
        m2, m3 = d.get("M2_in_distribution"), d.get("M3_held_out")
        if not isinstance(m2, dict) or not m2 or not isinstance(m3, dict) or not m3:
            return False
        for hop_dict in (m2, m3):
            for entry in hop_dict.values():
                if not isinstance(entry, dict) or TAU not in entry or IDEAL_TAU not in entry:
                    return False
        return True
    except Exception:
        return False


def build_cmd(spec, out_dir):
    """--model, --variant, --d, --K, [--force-rank-k], --steps, --seed,
    --orthogonal, --out. h-train/test/extra deliberately NOT passed -- the
    audited defaults are the gate's fixed operating point."""
    cmd = [sys.executable, RUN,
           "--model", spec["model"], "--variant", spec["variant"],
           "--d", str(spec["d"]), "--K", str(spec["K"]),
           "--steps", str(spec["steps"]), "--seed", str(spec["seed"]),
           "--orthogonal", "--out", out_path(out_dir, spec)]
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
    global STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/task_e_sweep"))
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--per-gpu", type=int, default=4, help="runs packed per GPU")
    ap.add_argument("--forever", action="store_true", help="refill with new seeds until STOP")
    ap.add_argument("--steps", type=int, default=STEPS,
                    help="train steps per run (40000 for the calibration relaunch; "
                         "MUST pair a steps change with a fresh --out-dir)")
    ap.add_argument("--timeout", type=int, default=TIMEOUT[D],
                    help="per-run wall timeout seconds (scale with --steps: "
                         "20K steps took 35-58 min under 4x packing)")
    ap.add_argument("--calibrate", action="store_true",
                    help="reduced round: all frN cells + fr {7,8,9} x3 seeds only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    STEPS = args.steps
    TIMEOUT[D] = args.timeout

    if args.dry_run:
        from collections import Counter
        m = build_manifest(0, args.calibrate)
        print(f"one round: {len(m)} runs | by tier {dict(sorted(Counter(s['tier'] for s in m).items()))} "
              f"| by model {dict(sorted(Counter(s['model'] for s in m).items()))} "
              f"| by variant {dict(sorted(Counter(s['variant'] for s in m).items()))} "
              f"| by K {dict(sorted(Counter(s['K'] for s in m).items()))}")
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
    print(f"perpetual={args.forever}  calibrate={args.calibrate}  steps={STEPS}  "
          f"timeout={TIMEOUT[D]}s  slots={n_slots} ({args.gpus}x{args.per_gpu})", flush=True)

    round_idx = 0
    pending = [s for s in build_manifest(0, args.calibrate) if not is_done(out_dir, s)]
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
                more = [s for s in build_manifest(round_idx * 1000, args.calibrate)
                        if not is_done(out_dir, s)]
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


def _hop_entries(run, key):
    """M2_in_distribution / M3_held_out dict keys are hop depths, but json
    round-trips dict keys through str() -- yield (hop:int, entry) pairs,
    silently skipping anything malformed (guarded aggregate, per-record)."""
    d = run.get(key)
    if not isinstance(d, dict):
        return
    for hk, entry in d.items():
        if not isinstance(entry, dict):
            continue
        try:
            yield int(hk), entry
        except Exception:
            continue


def aggregate(out_dir, failed):
    """Guarded, per-record: a single malformed run.json can drop itself from
    a breakdown but must never stop SUMMARY.txt/AGGREGATE.json from being
    written (mirrors run_overnight.py's aggregate() contract)."""
    def mean(xs):
        xs = [x for x in xs if x is not None]
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

        # M1_E: learned effective rank vs K (matrix/permutation, unconstrained
        # rank). Z's effective rank doesn't depend on the query hop, so pool
        # effective_rank_mean across every M2+M3 hop entry of a run, then
        # average across seeds, keyed by K.
        m1 = {}
        for r in runs:
            try:
                if r.get("model_type") != "matrix" or r.get("variant") != "permutation" \
                        or r.get("force_rank_k") is not None:
                    continue
                ers = [entry.get("effective_rank_mean")
                       for _, entry in (*_hop_entries(r, "M2_in_distribution"),
                                        *_hop_entries(r, "M3_held_out"))]
                ers = [e for e in ers if e is not None]
                if ers:
                    m1.setdefault(r["K"], []).append(sum(ers) / len(ers))
            except Exception:
                continue
        report["M1_effrank_vs_K"] = {K: mean(v) for K, v in sorted(m1.items())}

        # M3_E: held-out recovered_frac@0.9, keyed by (K, effective_hop) --
        # matrix (unconstrained rank) vs its own C7 ideal_ ceiling vs C_MLP
        # floor, all read off the SAME per-hop entries so gap-to-ideal is a
        # number (design doc section 6's stratification requirement).
        m3_matrix, m3_ideal, m3_mlp = {}, {}, {}
        for r in runs:
            try:
                if r.get("force_rank_k") is not None or r.get("variant") != "permutation":
                    continue
                K = r["K"]
                for h, entry in _hop_entries(r, "M3_held_out"):
                    eh = entry.get("effective_hop", h)
                    if r.get("model_type") == "matrix":
                        if TAU in entry:
                            m3_matrix.setdefault(K, {}).setdefault(eh, []).append(entry[TAU])
                        if IDEAL_TAU in entry:
                            m3_ideal.setdefault(K, {}).setdefault(eh, []).append(entry[IDEAL_TAU])
                    elif r.get("model_type") == "mlp" and TAU in entry:
                        m3_mlp.setdefault(K, {}).setdefault(eh, []).append(entry[TAU])
            except Exception:
                continue
        Ks = sorted(set(m3_matrix) | set(m3_ideal) | set(m3_mlp))
        report["M3_" + TAU + "_matrix_vs_ideal_vs_mlp"] = {
            f"K{K}": {
                f"eh{eh}": {"matrix": mean(m3_matrix.get(K, {}).get(eh, [])),
                            "ideal": mean(m3_ideal.get(K, {}).get(eh, [])),
                            "mlp_floor": mean(m3_mlp.get(K, {}).get(eh, []))}
                for eh in sorted(set(m3_matrix.get(K, {})) | set(m3_ideal.get(K, {}))
                                 | set(m3_mlp.get(K, {})))
            } for K in Ks
        }

        # M4_E: recovered_frac@0.9 vs force_rank_k at K=8, split by
        # in-distribution (M2) vs held-out (M3) hops (design doc section 6).
        m4_m2, m4_m3 = {}, {}
        for r in runs:
            try:
                if r.get("model_type") != "matrix" or r.get("variant") != "permutation" \
                        or r.get("force_rank_k") is None or r.get("K") != K_M4:
                    continue
                fr = r["force_rank_k"]
                for _, entry in _hop_entries(r, "M2_in_distribution"):
                    if TAU in entry:
                        m4_m2.setdefault(fr, []).append(entry[TAU])
                for _, entry in _hop_entries(r, "M3_held_out"):
                    if TAU in entry:
                        m4_m3.setdefault(fr, []).append(entry[TAU])
            except Exception:
                continue
        frs = sorted(set(m4_m2) | set(m4_m3))
        report[f"M4_{TAU}_vs_forcerank_K{K_M4}"] = {
            fr: {"M2_in_distribution": mean(m4_m2.get(fr, [])),
                 "M3_held_out": mean(m4_m3.get(fr, []))}
            for fr in frs
        }
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write("Task E — perpetual sweep\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
