"""Stage 0 orchestrator — see STAGE0_DESIGN.md section 6 (manifest) and
section 6.3 (build requirements).

CLONE of run_overnight.py's robustness pattern (smoke gate, exception-
isolated launch, validity-checked resume, per-run timeout with GPU
quarantine, guarded aggregate, top-level crash handler), adapted for Stage
0's four bounded, human-gated waves instead of one perpetually-refilling
queue:

  - NO perpetual refill. Each wave's manifest is fixed and finite; the
    orchestrator drains it once and exits (design doc section 6.3 / task
    brief: "bounded rounds only", "NO perpetual refill").
  - `--wave {-1,0,A,B}` launches exactly one wave; the human reviews results
    (and, for Wave A, decides which candidates "show life" per section 6.1;
    for Wave B, which is the confirmed top-1 survivor) before launching the
    next wave. This IS the "human gate between waves" the task brief
    requires — Wave A needs `--alive` and Wave B needs `--winner` supplied
    explicitly for a real (non-dry-run) launch; there is no default that
    silently proceeds.
  - Manifest naming carries the `variant` field (MAJOR-2, section 6.3) so
    two different interventions at the same (d, K, seed) never collide on
    the same output path — same pattern as run_task_e_sweep.py::_spec.
  - `is_done` is validity-checked AND cross-checks steps/K/seed/variant
    (and d/force_rank_k) recorded INSIDE the result JSON against the spec
    that would produce that path — run_overnight.py's is_done only checks
    key PRESENCE, which would silently accept a stale JSON left over from a
    different run that happens to share this run's output path (e.g. after
    a manifest edit). Closed here per the task brief's explicit callout of
    this as "a known is_done weakness".

Usage:
  python run_stage0_sweep.py --dry-run
  python run_stage0_sweep.py --wave -1 --out-dir results/stage0
  python run_stage0_sweep.py --wave 0  --out-dir results/stage0
  python run_stage0_sweep.py --wave A  --out-dir results/stage0 --alive c1_warmup,c2_orthogonal
  python run_stage0_sweep.py --wave B  --out-dir results/stage0 --winner c2_orthogonal
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_stage0.py")
TAU = "recovered_frac@0.9"

# Tier-default ("standard") step counts — mirrors run_overnight.py's STEPS
# dict for d in {16,32,64} (verified identical: 8000/8000/10000). Not
# imported from run_overnight.py to keep this orchestrator's dependency
# surface self-contained (same choice run_task_e_sweep.py made — it is a
# "CLONE", not an importer, of run_overnight.py).
TIER_STEPS = {16: 8000, 32: 8000, 64: 10000}

CANDIDATES = ["c1_warmup", "c2_orthogonal", "c3_mup", "c4_selfattn"]  # Wave-A screen (design section 4)
FALLBACK = ["c7_rowout"]  # not screened in Wave A; deployed only via pre-registered additions (section 6)

# Per-step wall-clock estimate (s/step), derived from run_overnight.py's
# TIMEOUT/STEPS ratio at each d (0.75h/8000 steps @ d<=32, ~5400s/10000 @
# d=64). h=128 cells are UNMEASURED until Wave -1 times one (MAJOR-3) — the
# >64 branch below applies a conservative extra margin until that lands.
_PER_STEP_S = {16: 0.225, 32: 0.375, 64: 0.54}

# run_stage0.py's --ckpt-every default; this orchestrator never overrides it,
# so the timeout formula below assumes it. Keep in sync.
CKPT_EVERY = 2000


def default_timeout(d: int, steps: int, h_dim: int = 64, margin: float = 1.6,
                    ckpt_every: int = CKPT_EVERY) -> int:
    """Wall timeout = (train + checkpoint-eval) * margin, h>64-scaled.

    The checkpoint-eval term (audit MINOR-6): unlike run_task_d.py (ONE eval
    at the end), run_stage0.py runs a full eval at EVERY <=2K-step checkpoint,
    and each unconstrained eval computes the M2 curve — up to d
    truncate_to_rank (eigh) passes over 4 eval batches. Estimated at
    (5 + 0.5*d) s/checkpoint — deliberately conservative (a full d=64 M2
    checkpoint ~37s); Wave -1's calibration cells measure the real number
    before Waves 0/A/B commit to it (attack #9)."""
    n_ckpts = steps // ckpt_every + 1                  # +1: the final step==steps eval
    base = (_PER_STEP_S.get(d, 0.6) * steps + n_ckpts * (5.0 + 0.5 * d)) * margin
    if h_dim > 64:
        base *= (h_dim / 64.0)   # conservative headroom for the 3.9x-param h=128 cells (section 5/9, attack #9)
    return int(base)


def _spec(wave, variant, d, K, h_dim, steps, fr, seed, tag_extra=""):
    tag = (f"w{wave}_{variant}_d{d}_K{K}_h{h_dim}_fr{'N' if fr is None else fr}_s{seed}{tag_extra}")
    return {"wave": str(wave), "variant": variant, "d": d, "K": K, "h": h_dim,
            "steps": steps, "force_rank_k": fr, "seed": seed, "name": tag}


# ---------------------------------------------------------------------------
# Manifests — one function per wave. Run counts are pinned to the design's
# budget table (section 6): -1:3, 0:12, A:11, B:12 (38 core total).
# ---------------------------------------------------------------------------

def wave_neg1_manifest():
    """Live calibration/timing (mandatory per CLAUDE.md), standard steps, 1
    seed. K is not pinned by the design doc for these cells; K=8 (d=32) and
    K=32 (d=64) are chosen as the minimal-risk reading — K=8 is the design's
    own "friendlier probe" (MAJOR-5) and K=32=d/2 matches run_overnight.py's
    existing Tier-2 convention, so both calibration cells reproduce an
    ALREADY-CHARACTERIZED point in the 991-run baseline sweep."""
    return [
        _spec(-1, "baseline", 32, 8, 64, TIER_STEPS[32], None, 0),
        _spec(-1, "baseline", 64, 32, 64, TIER_STEPS[64], None, 0),
        _spec(-1, "baseline", 32, 8, 128, TIER_STEPS[32], None, 0),  # MAJOR-3: h=128 wall-clock calibration
    ]


def wave0_manifest():
    """Unmodified baseline at 2.5x steps (section 6): 3 primary cells x3
    seeds (FATAL-1's >=3-seed requirement) + the d=16,K=4 opportunistic probe
    (section 7, 2 seeds, explicitly non-gating) + the d=16,K=8 healthy-regime
    reference at STANDARD steps (1 seed). These 9 primary-cell runs ALSO
    serve as Wave A's Arm 0 (section 6)."""
    runs = []
    for K, n_seeds in ((8, 3), (16, 3)):
        steps = int(TIER_STEPS[32] * 2.5)
        runs += [_spec(0, "baseline", 32, K, 64, steps, None, s) for s in range(n_seeds)]
    steps64 = int(TIER_STEPS[64] * 2.5)
    runs += [_spec(0, "baseline", 64, 32, 64, steps64, None, s) for s in range(3)]
    steps16 = int(TIER_STEPS[16] * 2.5)
    runs += [_spec(0, "baseline", 16, 4, 64, steps16, None, s) for s in range(2)]   # opportunistic, section 7
    runs += [_spec(0, "baseline", 16, 8, 64, TIER_STEPS[16], None, 0)]              # healthy-regime reference
    return runs


def waveA_manifest(alive_candidates=None):
    """Cheap probe @ 2x steps, single seed (section 6): candidates 1-4 x
    {d=32 K=8, d=32 K=16} (8 runs) + 1 standard-steps d=16 sanity probe of
    candidate 3's h=128 compensation recipe against the known h=256/d=16
    divergence (attack #8) + up to 2 sequential d=64 K=32 probes for
    candidates "alive" at d=32 (section 6.1's life criterion; budget 2).

    `alive_candidates` is populated by a HUMAN reading Wave 0 + this wave's
    own K=8/K=16 results (section 6.1) — there is no way to compute it ahead
    of the K=8/K=16 runs finishing, so a real launch REQUIRES `--alive`
    (enforced in main()). When None (dry-run manifest-shape purposes only),
    the first 2 candidates in the canonical list stand in as a placeholder
    so the run COUNT matches the design's budget table even before any
    result exists."""
    runs = []
    steps32 = int(TIER_STEPS[32] * 2)
    for c in CANDIDATES:
        h_dim = 128 if c == "c3_mup" else 64   # candidate 3's representative test point (section 5)
        for K in (8, 16):
            runs.append(_spec("A", c, 32, K, h_dim, steps32, None, 0))
    runs.append(_spec("A", "c3_mup", 16, 8, 128, TIER_STEPS[16], None, 0, tag_extra="_sanity"))
    alive = alive_candidates if alive_candidates is not None else CANDIDATES[:2]  # PLACEHOLDER (dry-run only)
    steps64 = int(TIER_STEPS[64] * 2)
    for c in alive[:2]:
        h_dim = 128 if c == "c3_mup" else 64
        runs.append(_spec("A", c, 64, 32, h_dim, steps64, None, 0, tag_extra="_d64probe"))
    return runs


def waveB_manifest(winner_variant=None):
    """Full-seed confirmation @ 2x steps of the top-1 Wave-A survivor
    (section 6): M1 unconstrained K in {4,8,16} x2 seeds (6 runs, also
    yielding the free M2 truncation curve) + M3 train-time force-rank
    k in {1,7,8} at K=8 x2 seeds (6 runs).

    `winner_variant` is the HUMAN's read of Wave A's life-criterion ranking
    (section 6.1) — a real launch REQUIRES `--winner` (enforced in main()).
    When None (dry-run only), "c2_orthogonal" stands in as a placeholder."""
    winner = winner_variant if winner_variant is not None else "c2_orthogonal"  # PLACEHOLDER (dry-run only)
    steps32 = int(TIER_STEPS[32] * 2)
    runs = []
    for K in (4, 8, 16):
        for s in (0, 1):
            runs.append(_spec("B", winner, 32, K, 64, steps32, None, s))
    for fr in (1, 7, 8):
        for s in (0, 1):
            runs.append(_spec("B", winner, 32, 8, 64, steps32, fr, s))
    return runs


MANIFEST_FNS = {"-1": wave_neg1_manifest, "0": wave0_manifest}


def _parse_alive(s):
    """None (flag absent) -> None. Otherwise the parse must yield >=1 name —
    a malformed value like "," or " " would otherwise pass main()'s
    truthiness gate yet parse to [], silently dropping the d=64 probes
    (audit MINOR-4). Hard-error instead."""
    if s is None:
        return None
    alive = [x.strip() for x in s.split(",") if x.strip()]
    if not alive:
        print(f"ERROR: --alive {s!r} parsed to an empty candidate list; give a "
              f"comma-separated list of variant names (e.g. c1_warmup,c2_orthogonal).",
              file=sys.stderr)
        sys.exit(1)
    return alive


def out_path(out_dir, spec):
    return os.path.join(out_dir, f"{spec['name']}.json")


def is_done(out_dir, spec):
    """Validity-checked resume, cross-checked against the spec's own identity
    fields (see module docstring — closes run_overnight.py's is_done
    weakness).

    Crash-safety (audit FATAL-1): run_stage0.py's INCREMENTAL checkpoint
    dumps carry "steps" = the REQUESTED total, so a run SIGKILL'd at 2%
    budget leaves a JSON whose identity fields all match the spec —
    presence + identity checks alone would accept it as done forever.
    Require the completion sentinel (complete == True, written ONLY by
    run_stage0.py main()'s post-training write; every incremental dump has
    complete=False) AND steps_completed >= the spec's step count."""
    p = out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return False
        required = ("mean_cos", "effective_rank_mean", "variant", "K", "d",
                    "seed", "steps", "force_rank_k", "complete", "steps_completed")
        if not all(k in d for k in required):
            return False
        if d.get("timed_out"):
            return False   # a timed-out run is a PARTIAL result, not "done"
        if d.get("complete") is not True:
            return False   # FATAL-1 sentinel: incremental/killed dumps are never done
        if d.get("steps_completed", 0) < spec["steps"]:
            return False   # belt-and-braces: sentinel must agree with the step count
        if (d.get("variant") != spec["variant"] or d.get("K") != spec["K"]
                or d.get("d") != spec["d"] or d.get("seed") != spec["seed"]
                or d.get("steps") != spec["steps"]
                or d.get("force_rank_k") != spec["force_rank_k"]):
            return False
        return True
    except Exception:
        return False


def build_cmd(spec, out_dir, timeout):
    cmd = [sys.executable, RUN, "--variant", spec["variant"],
           "--d", str(spec["d"]), "--K", str(spec["K"]), "--h", str(spec["h"]),
           "--steps", str(spec["steps"]), "--seed", str(spec["seed"]),
           "--orthogonal", "--internal-timeout", str(max(1, timeout - 30)),
           "--out", out_path(out_dir, spec)]
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


def write_progress(out_dir, done, failed, running, wave):
    try:
        with open(os.path.join(out_dir, "PROGRESS.txt"), "w") as f:
            f.write(f"wave={wave} done={done} failed={failed} running={running}\n")
    except Exception as e:
        print(f"  (write_progress non-fatal: {e!r})", flush=True)


def aggregate(out_dir, failed, wave):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

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
        # Audit round-2 finding B (2026-07-02): incremental checkpoint writes
        # mean partial (complete!=True) files exist on disk mid-wave; pooling
        # them into by_cell means dilutes the human gate's reading material
        # (demonstrated: one complete 0.9 + one 10%-budget partial 0.1 -> cell
        # mean 0.5). Aggregate ONLY complete runs; count partials separately.
        complete_runs = [r for r in runs if r.get("complete") is True]
        report["n_partial_excluded"] = len(runs) - len(complete_runs)
        by_cell = {}
        for r in complete_runs:
            try:
                fr = "N" if r.get("force_rank_k") is None else r["force_rank_k"]
                key = f"{r['variant']}_d{r['d']}_K{r['K']}_h{r.get('h')}_fr{fr}"
                cell = by_cell.setdefault(key, {"mean_cos": [], "effective_rank_mean": [], TAU: []})
                cell["mean_cos"].append(r.get("mean_cos"))
                cell["effective_rank_mean"].append(r.get("effective_rank_mean"))
                cell[TAU].append(r.get(TAU))
            except Exception:
                continue
        report["by_cell"] = {k: {mk: mean(v) for mk, v in vv.items()} for k, vv in sorted(by_cell.items())}
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write(f"Stage 0 -- wave {wave}\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/stage0"))
    ap.add_argument("--wave", choices=["-1", "0", "A", "B"], default=None,
                    help="which wave to launch; REQUIRED unless --dry-run. Waves launch "
                         "separately with a human gate between them (no perpetual refill).")
    ap.add_argument("--alive", type=str, default=None,
                    help="Wave A only, REQUIRED for a real launch: comma-separated list of "
                         "the Wave-A K=8/K=16 'life'-criterion survivors (section 6.1) to "
                         "sequentially probe at d=64 K=32 (budget: first 2).")
    ap.add_argument("--winner", type=str, default=None,
                    help="Wave B only, REQUIRED for a real launch: the single confirmed "
                         "top-1 Wave-A survivor variant name.")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--per-gpu", type=int, default=4, help="runs packed per GPU")
    ap.add_argument("--timeout", type=float, default=None,
                    help="override the per-run wall timeout (s) for every run in this wave; "
                         "default is computed per-spec from step count + a 1.6x margin "
                         "(section 6.3: 'timeouts scale with the step multiplier')")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    if args.dry_run:
        from collections import Counter
        waves = {
            "-1": wave_neg1_manifest(),
            "0": wave0_manifest(),
            "A": waveA_manifest(_parse_alive(args.alive)),
            "B": waveB_manifest(args.winner),
        }
        total = 0
        for w, runs in waves.items():
            total += len(runs)
            print(f"wave {w}: {len(runs)} runs | by variant "
                  f"{dict(sorted(Counter(s['variant'] for s in runs).items()))} "
                  f"| by (d,K) {dict(sorted(Counter((s['d'], s['K']) for s in runs).items()))}")
        print(f"CORE TOTAL: {total} runs  (design budget table, section 6: 38 = 3/12/11/12)")
        print(f"slots = {args.gpus} gpus x {args.per_gpu} per-gpu = {args.gpus * args.per_gpu} concurrent")
        if args.alive is None:
            print("NOTE: --alive not given -- Wave A's d=64-probe count uses a PLACEHOLDER "
                  "candidate pair for manifest-SHAPE purposes only; a real launch requires it.")
        if args.winner is None:
            print("NOTE: --winner not given -- Wave B uses a PLACEHOLDER variant for "
                  "manifest-SHAPE purposes only; a real launch requires it.")
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)
    if args.wave == "A" and not args.alive:
        print("ERROR: --wave A requires --alive (the human gate: which Wave-0-reviewed "
              "candidates to sequentially probe at d=64).", file=sys.stderr)
        sys.exit(1)
    if args.wave == "B" and not args.winner:
        print("ERROR: --wave B requires --winner (the human gate: the confirmed top-1 "
              "Wave-A survivor).", file=sys.stderr)
        sys.exit(1)

    if args.wave in MANIFEST_FNS:
        manifest = MANIFEST_FNS[args.wave]()
    elif args.wave == "A":
        manifest = waveA_manifest(_parse_alive(args.alive))
    else:
        manifest = waveB_manifest(args.winner)

    out_dir = os.path.join(args.out_dir, f"wave{args.wave}")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate failed; no training launched.\n")
        sys.exit(1)

    slots = [g for _ in range(args.per_gpu) for g in range(args.gpus)]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done(out_dir, s)]
    print(f"wave={args.wave}  manifest={len(manifest)}  pending={len(pending)}  "
          f"slots={n_slots} ({args.gpus}x{args.per_gpu})", flush=True)

    running = {}          # uid -> (proc, spec, lf, start, gpu, timeout)
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
                timeout = args.timeout if args.timeout is not None \
                    else default_timeout(spec["d"], spec["steps"], spec["h"])
                try:
                    lf = open(os.path.join(log_dir, f"{spec['name']}.log"), "w")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    proc = subprocess.Popen(build_cmd(spec, out_dir, timeout), env=env,
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
    print("HUMAN GATE: review results (and, for Wave A, section 6.1's life criterion) "
          "before launching the next wave.", flush=True)


if __name__ == "__main__":
    main()
