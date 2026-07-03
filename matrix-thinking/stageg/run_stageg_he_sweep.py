"""run_stageg_he_sweep.py -- bounded, human-gated launcher for the Stage G
Wave C (H_e task-swap) probe. See STAGE_G_DESIGN.md section 4 item 9 and
section 6.4 item 3 (Wave C, gated), and the task brief's exact manifest:
matrix baseline + h_b_factored_r4 (the H_b Wave-B winner, models.py's
VARIANT_AXES) + vector reference, x2 seeds each = 6 cells.

WAVE-C GATE NOTE (audit item 2f). STAGE_G_DESIGN.md gates H_e's task-swap
control on "a distributed-tax verdict or a confirmed-but-narrow dominant
site (i.e., only invoked if the 'why' question remains open after the
cheaper waves)" (section 4 item 9; restated at section 7 stretch item 3:
"only on a distributed-tax verdict or a narrow confirm that leaves the
generalization question open"). Section 14.6's verdict satisfies the
CONFIRMED-BUT-NARROW branch: a dominant site was formally named per
section 8's bars -- the Kronecker-separable RowThenColProjection
restriction, recovering ~64.1% of the gap at matched params
(h_b_factored_r3_nl3, 3 seeds, every seed >= 0.5) and ~70.1% at the
exact-param zero-depth-confound point (h_b_factored_r1, n=1), with every
other screened axis at <= +0.006 -- while the per-FLOP tax against a
capacity-matched vector reference SURVIVES at ~16.5x (sections 14.4 and
14.6 item 3). The named site is therefore narrow (a projection-family fix
inside a still-per-FLOP-dominated architecture), leaving exactly the
generalization question -- "does the matrix-vs-vector ranking invert on a
composition-heavy task?" -- open. That is the gate's stated invocation
condition: Wave C is legitimately invoked.

STALE-PRICING FLAG (audit). Section 7 stretch item 3 prices Wave C at
"~10 GPU-h if invoked". At this build's calibrated contended-box rates
(0.723 s/step matrix, 0.0563 s/step vector -- constants below), the real
6-cell manifest costs ~2.5-3.4 GPU-h at the 3,000-step default but
~16.7 GPU-h at the 20K steps the calibration finding (below) suggests may
be needed -- 1.7-3.3x the design's price depending on the step decision.
The step-budget decision is therefore also a budget decision; neither is
made unilaterally here.

CLONE of run_stageg_sweep.py's / run_deltanet_sweep.py's robustness pattern
(smoke gate, exception-isolated launch, validity-checked resume, per-run
timeout with GPU quarantine, guarded aggregate, top-level crash handler) --
deliberately NOT a cross-directory import, matching this repo's established
pod-safety convention.

  NO perpetual refill -- the manifest is fixed (6 cells) and finite; the
  orchestrator drains it once and exits.
  Manifest naming carries family+variant+seed+steps+K (mirrors Stage 0's
  MAJOR-2 fix / run_deltanet_sweep.py's F17 discipline).
  is_done is validity-checked AND cross-checks the "H_e_task" identity
  sentinel PLUS family/variant/seed/steps/K recorded INSIDE the result
  JSON -- an H_e-task result must never satisfy a plain Stage-G Wave-A/B
  spec or vice versa (they share no output directory, but the sentinel
  makes the distinction checkable, not just directory-separated).
  `--gpus`/`--gpu-offset` are REQUIRED for a real launch, NO DEFAULTS (the
  deltanet sweep's audit round-1 FINDING 2 discipline, adopted here too:
  this box runs concurrent waves from other builders and the busy-GPU set
  changes day to day -- check nvidia-smi and pass the free set explicitly).

Usage:
  python run_stageg_he_sweep.py --dry-run
  python run_stageg_he_sweep.py --gpus 2 --gpu-offset 6 --out-dir results/stageg_he
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_stageg_he.py")

# The task brief's exact 3-arm manifest: matrix-all baseline, the H_b
# Wave-B winner (h_b_factored_r4, models.py's VARIANT_AXES), and the
# vector-all reference -- x2 seeds each = 6 cells.
CELLS = (("matrix", "baseline"), ("matrix", "h_b_factored_r4"), ("vector", "baseline"))
SEEDS = (0, 1)

# s/step, matrix vs vector -- MEASURED 2026-07-03 on this box (GPUs 6/7,
# box concurrently running >=2 other builders' waves on GPUs 0-5 -- see the
# build report; not an idle-box number) at the REAL manifest config (K=12,
# Q_per_doc=8, doc_len=96, mat_dim=32, n_iterations=8, batch_size=64):
#   vector: 84.43s  / 1500 steps (3 checkpoint evals @ ckpt_every=500)
#           = 0.0563 s/step
#   matrix: 1019.5s / 1500 steps (complete=true, same 3-eval schedule)
#           = 0.680 s/step -- ~12x vector's per-step cost, matching
#           STAGE_G_DESIGN.md's own 11.8-14.9x analytic FLOPs ratio
#           (section 5.1). The 0.723 constant below is the slightly more
#           conservative 500-step partial-run figure (early steps include
#           CUDA warmup), kept ON PURPOSE: timeouts should over- not
#           under-estimate. REPLACES the earlier PROVISIONAL guess
#           (0.35/0.10) -- both were under-estimates, matrix especially so.
# CALIBRATION FINDING (flag for the audit round, CLAUDE.md: "a calibration
# run before a big sweep is mandatory, not optional"): over the FULL 1500
# calibration steps (half the --steps 3000 default), BOTH families show
# composition_accuracy pinned at/near chance (~0.08-0.10 at K=12)
# in-distribution at EVERY checkpoint (500/1000/1500), despite BPB dropping
# sharply (matrix 8.24->1.64, vector 8.05->1.44 -- the R/Q/digit STRUCTURAL
# format is learned quickly; no sign of the COMPOSITIONAL circuit emerging).
# Held-out hops sit at/below chance (matrix h=4/h=7 EXACTLY 0.0 at all 3
# checkpoints, h=5 ~0.05 -- the model confidently emits format-consistent
# but wrong bytes for unseen hop digits; the eval's negative-control oracle
# proves exact-0 is a real, discriminating reading, not a metric bug). This
# mirrors Task E's own precedent (chapter2/TASK_E_FINDINGS.md: convergence
# took a 40K-step re-registration, not the original budget) and
# STAGE_G_DESIGN.md section 4 item 9's own characterization of this task as
# "the single largest engineering-risk item in the whole manifest".
# --steps 3000 (this file's default) is UNCHANGED pending the audit round's
# decision on whether to raise it (e.g. 10-30x) before any real launch.
PROVISIONAL_S_PER_STEP_HE = {"matrix": 0.723, "vector": 0.0563}


def _spec(family, variant, seed, steps, K, Q_per_doc, mat_dim, answer_loss_weight):
    # FIX-C: answer_loss_weight is part of the run's NAME and is_done
    # identity -- a W=1.0 result must never satisfy a W=5.0 spec on a
    # resume path or vice versa ({:g} strips trailing zeros: alw5, alw1).
    tag = (f"wC_he_{family}_{variant}_K{K}_Q{Q_per_doc}_steps{steps}"
           f"_alw{answer_loss_weight:g}_s{seed}")
    return {"family": family, "variant": variant, "seed": seed, "steps": steps,
            "K": K, "Q_per_doc": Q_per_doc, "mat_dim": mat_dim,
            "answer_loss_weight": answer_loss_weight, "name": tag}


def build_manifest(steps, K, Q_per_doc, mat_dim, answer_loss_weight):
    return [_spec(family, variant, seed, steps, K, Q_per_doc, mat_dim, answer_loss_weight)
            for family, variant in CELLS for seed in SEEDS]


def out_path(out_dir, spec):
    return os.path.join(out_dir, f"{spec['name']}.json")


def is_done(out_dir, spec):
    """Validity-checked resume (F17 discipline), cross-checked against the
    "H_e_task" identity sentinel PLUS the spec's own identity fields."""
    p = out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return False
        if d.get("H_e_task") is not True:
            return False
        required = ("family", "variant", "seed", "steps", "K", "complete", "steps_completed")
        if not all(k in d for k in required):
            return False
        if d.get("timed_out"):
            return False
        if d.get("complete") is not True:
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if (d.get("family") != spec["family"] or d.get("variant") != spec["variant"]
                or d.get("seed") != spec["seed"] or d.get("steps") != spec["steps"]
                or d.get("K") != spec["K"]):
            return False
        # FIX-C identity: no legacy-default fallback here on purpose -- no
        # pre-FIX-C H_e result JSONs exist (nothing was ever launched), so
        # a missing answer_loss_weight field means a malformed/foreign file.
        if d.get("answer_loss_weight") != spec["answer_loss_weight"]:
            return False
        return True
    except Exception:
        return False


def build_cmd(spec, out_dir, timeout, max_len, vocab_size, n_iterations, batch_size,
              h_train, h_test, h_extra):
    cmd = [sys.executable, RUN, "--family", spec["family"], "--variant", spec["variant"],
           "--mat-dim", str(spec["mat_dim"]), "--max-len", str(max_len),
           "--vocab-size", str(vocab_size), "--n-iterations", str(n_iterations),
           "--K", str(spec["K"]), "--Q-per-doc", str(spec["Q_per_doc"]),
           "--h-train", *[str(h) for h in h_train], "--h-test", *[str(h) for h in h_test],
           "--h-extra", *[str(h) for h in h_extra],
           "--steps", str(spec["steps"]), "--seed", str(spec["seed"]),
           "--batch-size", str(batch_size),
           "--answer-loss-weight", str(spec["answer_loss_weight"]),
           "--internal-timeout", str(max(1, timeout - 30)),
           "--out", out_path(out_dir, spec), "--device", "cuda"]
    return cmd


def default_timeout(steps, family, margin=1.6):
    return int(steps * PROVISIONAL_S_PER_STEP_HE[family] * margin) + 120


def run_smoke(log_dir, smoke_gpu=0):
    print(f"SMOKE GATE (GPU {smoke_gpu}) ...", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(smoke_gpu)}
    with open(os.path.join(log_dir, "smoke.log"), "w") as lf:
        rc = subprocess.call([sys.executable, RUN, "--smoke"], env=env,
                              stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    print("smoke passed." if rc == 0 else f"SMOKE FAILED (rc={rc}) -- ABORTING.", flush=True)
    return rc == 0


def write_progress(out_dir, done, failed, running):
    try:
        with open(os.path.join(out_dir, "PROGRESS.txt"), "w") as f:
            f.write(f"wave=C_he done={done} failed={failed} running={running}\n")
    except Exception as e:
        print(f"  (write_progress non-fatal: {e!r})", flush=True)


def aggregate(out_dir, failed):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    report = {"wave": "C_he", "n_failed": len(failed)}
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
                key = f"{r['family']}_{r['variant']}"
                # FIX-A: the in-distribution HEADLINE excludes h=1 -- the
                # R<start><succ(start)> line makes every h=1 query answerable
                # by pure copy (find the R-line with a matching 2nd byte,
                # emit its 3rd byte), zero composition (task_he.py module
                # docstring). h=1 is reported separately as a single-hop
                # LOOKUP metric; held_out already excludes h=1 by
                # construction (H_test/H_extra start at 4).
                cell = by_cell.setdefault(key, {
                    "n_params": [], "capacity_ratio_vs_matrix_baseline": [],
                    "held_out_comp_acc": [], "held_out_chance_adjusted_acc": [],
                    "in_dist_comp_acc_excl_h1": [], "in_dist_chance_adjusted_excl_h1": [],
                    "single_hop_lookup_acc_h1": []})
                cell["n_params"].append(r.get("n_params"))
                cell["capacity_ratio_vs_matrix_baseline"].append(
                    (r.get("capacity_confound") or {}).get("ratio_vs_matrix_baseline"))  # FIX-B
                fc = r.get("final_composition") or {}
                held = fc.get("held_out") or {}
                indist = fc.get("in_distribution") or {}
                if held:
                    cell["held_out_comp_acc"].append(
                        sum(e["composition_accuracy"] for e in held.values()) / len(held))
                    cell["held_out_chance_adjusted_acc"].append(
                        sum(e["chance_adjusted_acc"] for e in held.values()) / len(held))
                # JSON round-trip stringifies hop keys -- compare as ints.
                indist_comp = {h: e for h, e in indist.items() if int(h) >= 2}
                indist_h1 = {h: e for h, e in indist.items() if int(h) == 1}
                if indist_comp:
                    cell["in_dist_comp_acc_excl_h1"].append(
                        sum(e["composition_accuracy"] for e in indist_comp.values()) / len(indist_comp))
                    cell["in_dist_chance_adjusted_excl_h1"].append(
                        sum(e["chance_adjusted_acc"] for e in indist_comp.values()) / len(indist_comp))
                if indist_h1:
                    cell["single_hop_lookup_acc_h1"].append(
                        next(iter(indist_h1.values()))["composition_accuracy"])
            except Exception:
                continue
        report["by_cell"] = {k: {mk: mean(v) for mk, v in vv.items()} for k, vv in sorted(by_cell.items())}
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write("Stage G Wave C (H_e task-swap)\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "stageg_he"))
    ap.add_argument("--mat-dim", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--n-iterations", type=int, default=8)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--Q-per-doc", type=int, default=8)
    ap.add_argument("--h-train", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--h-test", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--h-extra", type=int, nargs="+", default=[7])
    ap.add_argument("--steps", type=int, default=3000,
                     help="PROVISIONAL until this build's timing probe pins a calibrated "
                          "value (mirrors run_stageg_sweep.py's --standard-steps convention).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--answer-loss-weight", type=float, default=5.0,
                     help="FIX-C (audit): forwarded to every run; carried in the run NAME "
                          "and the is_done identity. 1.0 recovers the plain LM objective "
                          "bitwise (verified in run_stageg_he.py's smoke gate).")
    ap.add_argument("--gpus", type=int, default=None,
                     help="GPU COUNT to use. REQUIRED for a real launch, NO DEFAULT ON "
                          "PURPOSE (matches run_deltanet_sweep.py's audit round-1 FINDING 2 "
                          "discipline): this box runs concurrent waves from other builders "
                          "and the busy-GPU set changes day to day. Run nvidia-smi "
                          "immediately before every launch and pass the free set explicitly.")
    ap.add_argument("--gpu-offset", type=int, default=None,
                     help="first physical GPU index to use. REQUIRED for a real launch, "
                          "NO DEFAULT ON PURPOSE -- see --gpus.")
    ap.add_argument("--per-gpu", type=int, default=2, help="runs packed per GPU")
    ap.add_argument("--timeout", type=float, default=None,
                     help="override the per-run wall timeout (s); default is computed "
                          "per-spec from PROVISIONAL_S_PER_STEP_HE")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-gpu", type=int, default=0,
                     help="GPU index for the pre-launch smoke gate -- check nvidia-smi first, "
                          "a shared box may have GPU 0 busy.")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    manifest = build_manifest(args.steps, args.K, args.Q_per_doc, args.mat_dim,
                               args.answer_loss_weight)

    # FIX-B launch-time warning (printed for dry runs AND real launches):
    # h_b_factored_r4 carries ~2.69x the matrix baseline's params (781,848
    # vs 290,328 at the manifest config) -- STAGE_G_DESIGN.md section 14.4
    # showed this exact cell's Wave-B win partially collapsed against a
    # capacity-matched control. Per-run JSONs carry the exact measured
    # ratio in their capacity_confound block (run_stageg_he.py).
    confounded = sorted({s["variant"] for s in manifest if s["variant"].startswith("h_b_factored")})
    if confounded:
        print(f"CAPACITY-CONFOUND WARNING (FIX-B, STAGE_G_DESIGN.md section 14.4): manifest "
              f"arm(s) {confounded} run at ~2.69x the matrix baseline's params -- any "
              f"win/loss vs the baselines confounds the projection swap with raw capacity. "
              f"A capacity-matched arm is DELIBERATELY not included yet (budget decision "
              f"deferred to the calibration outcome); read these cells with the per-run "
              f"capacity_confound JSON block in view.", flush=True)

    if args.dry_run:
        print(f"Wave C (H_e task-swap): {len(manifest)} runs")
        for s in manifest:
            print(f"  {s['name']}")
        print(f"CORE TOTAL: {len(manifest)} runs (3 arms x {len(SEEDS)} seeds = "
              f"{len(CELLS)}x{len(SEEDS)}={len(CELLS) * len(SEEDS)}, matching the task "
              f"brief's ~6-cell manifest)")
        if args.gpus is not None and args.gpu_offset is not None:
            print(f"slots = {args.gpus} gpus x {args.per_gpu} per-gpu = {args.gpus * args.per_gpu} "
                  f"concurrent, on physical GPUs {list(range(args.gpu_offset, args.gpu_offset + args.gpus))}")
        else:
            print("slots: pass --gpus/--gpu-offset to preview (REQUIRED for a real launch, "
                  "no defaults -- check nvidia-smi first).")
        return

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults "
              "on purpose): the busy-GPU set on this box changes day to day as other "
              "experiments' waves come and go. Run nvidia-smi NOW and pass the free set "
              "explicitly.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir, args.smoke_gpu):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate (run_stageg_he.py --smoke) failed; no training launched.\n")
        sys.exit(1)

    physical_gpus = list(range(args.gpu_offset, args.gpu_offset + args.gpus))
    slots = [g for _ in range(args.per_gpu) for g in physical_gpus]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done(out_dir, s)]
    print(f"wave=C_he  manifest={len(manifest)}  pending={len(pending)}  "
          f"slots={n_slots} (gpus {physical_gpus} x {args.per_gpu} per-gpu)", flush=True)

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
                timeout = args.timeout if args.timeout is not None \
                    else default_timeout(spec["steps"], spec["family"])
                try:
                    lf = open(os.path.join(log_dir, f"{spec['name']}.log"), "w")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    cmd = build_cmd(spec, out_dir, timeout, args.max_len, args.vocab_size,
                                     args.n_iterations, args.batch_size, args.h_train,
                                     args.h_test, args.h_extra)
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
            write_progress(out_dir, done_ct, len(failed), len(running))
            if time.time() - last_agg > 120:
                aggregate(out_dir, failed)
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
        print(f"ORCHESTRATOR CRASHED: {e!r} -- see CRASHED.txt; rerun to resume "
              f"(validity-checked, bounded, not perpetual).", flush=True)
    finally:
        aggregate(out_dir, failed)
    print(f"\nWAVE C (H_e) DONE. {done_ct} succeeded, {len(failed)} failed.", flush=True)


if __name__ == "__main__":
    main()
