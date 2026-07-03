"""run_deltanet_rd_sweep.py -- bounded, human-gated wave orchestrator for
DeltaNet-RD (DELTANET_REALDATA_DESIGN.md, frozen revision 2.1), Wave 1 path
(RD-1: section 5, section 7's manifest table, section 8's operational
requirements).

CLONE of matrix-thinking/deltanet/run_deltanet_sweep.py's robustness pattern
(smoke gate, exception-isolated launch, validity-checked resume, per-run
timeout with GPU quarantine, guarded aggregate) -- deliberately NOT a
cross-directory import (pod-safety convention).

Differences from the synthetic design's sweep, both flagged here rather than
silently inherited:
  - **No ``--primary-d`` gate.** Section 4.1's continuity decision FIXES
    d_state=64 / d_model=256 (the design's own recommendation, "keep the
    single-head state at d=64... while letting d_model=256 carry the param
    budget") -- there is no d-grid to screen in Wave 0 for this design, only
    a K-grid. ``--primary-k`` (Wave 0's "which K is the healthier primary
    cell" human read) plays the equivalent gating role for Bprobe/B.
  - **No ``--trunc-impl`` choice.** Only "eigh" is implemented (see
    run_deltanet_rd.py's own --trunc-impl help text for why the synthetic
    design's svd_lowrank fallback is not blindly re-added here).
  - Manifest naming carries (wave, K, force_rank_k, seed) -- no d/variant
    axis, no tokenizer_mode axis (section 8's call for one is not needed
    yet: C18's frozen-pretrained-embedding arm is an explicitly-labeled
    Reserve-wave variant, out of scope for this build phase, matching
    run_deltanet_sweep.py's own Reserve-wave non-implementation).
  - Wave -1's/Wave 0's/Wave A's exact K-cell choices are BUILD-TIME
    REASONABLE DEFAULTS (flagged below, same "PLACEHOLDER discipline" as
    run_deltanet_sweep.py's own waveA_manifest) -- section 7's budget table
    does not pin an exact K-grid for this design either.

Usage (GPU list is an example -- check nvidia-smi first, per house rule):
  python run_deltanet_rd_sweep.py --dry-run
  python run_deltanet_rd_sweep.py --wave -1      --out-dir results/deltanet_rd --gpus 1 --gpu-offset 7
  python run_deltanet_rd_sweep.py --wave 0       --out-dir results/deltanet_rd --gpus 1 --gpu-offset 7
  python run_deltanet_rd_sweep.py --wave A       --out-dir results/deltanet_rd --gpus 1 --gpu-offset 7 --primary-k 32
  python run_deltanet_rd_sweep.py --wave Bprobe  --out-dir results/deltanet_rd --gpus 1 --gpu-offset 7 --primary-k 32
  python run_deltanet_rd_sweep.py --wave B       --out-dir results/deltanet_rd --gpus 1 --gpu-offset 7 --primary-k 32 --force-launch-b
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_deltanet_rd.py")
TAU = "recovered_frac@0.9"

# Tier-default step count -- CONTINUITY with the synthetic design's own
# TIER_STEPS[64] (matched d_state=64, section 4.1's continuity decision).
# Flat across K (K affects PER-STEP wall-clock via T_bind = K*7, not step
# COUNT) -- a build-time default, Wave -1's job to calibrate/revise.
TIER_STEPS = 10000
CKPT_EVERY = 2000

# MEASURED per-step wall-clock (s/step) -- filled in from a short on-box
# profiling probe (see the module-level comment this gets updated with once
# Wave -1 / a pre-Wave--1 profiling pass actually measures it; until then
# these are PROVISIONAL, deliberately conservative placeholders carried the
# same way run_deltanet_sweep.py's own _PER_STEP_S table was before ITS
# probe_trunc.py measurement existed). Keyed by (K, forced: bool).
_PER_STEP_S_PLACEHOLDER = 0.15   # s/step, unconstrained, K<=32 -- conservative guess pending measurement
_PER_STEP_S_PLACEHOLDER_FORCED = 0.40   # s/step, force-rank (eigh truncation adds an SVD/eigh per step)


def default_timeout(K: int, steps: int, forced: bool, margin: float = 1.6,
                     ckpt_every: int = CKPT_EVERY) -> int:
    """Wall timeout = (train + checkpoint-eval) * margin. PROVISIONAL
    constants (see module docstring) -- Wave -1's explicit job is to
    replace _PER_STEP_S_PLACEHOLDER{,_FORCED} with measured values before
    Wave 0 is priced against them (mirrors the synthetic design's own
    calibration-first discipline, STAGE0_DESIGN.md section 12)."""
    n_ckpts = steps // ckpt_every + 1
    per_step = (_PER_STEP_S_PLACEHOLDER_FORCED if forced else _PER_STEP_S_PLACEHOLDER) * max(1.0, K / 32)
    base = (per_step * steps + n_ckpts * 20.0) * margin
    return int(base)


def _spec(wave, K, k, steps, seed, tag_extra=""):
    tag = f"w{wave}_rd_K{K}_fr{'N' if k is None else k}_s{seed}{tag_extra}"
    return {"wave": str(wave), "K": K, "force_rank_k": k, "steps": steps, "seed": seed, "name": tag}


# ---------------------------------------------------------------------------
# Manifests -- one function per wave.
# ---------------------------------------------------------------------------

def wave_neg1_manifest():
    """Timing + instability calibration (mandatory, CLAUDE.md). K in
    {16,32,64} (a build-time-reasonable spread across Wave A's planned grid,
    not pinned by the design text) x {unconstrained, force-rank at k=K} = 6
    runs, matching section 7's "~6-8 runs" budget."""
    runs = []
    for K in (16, 32, 64):
        runs.append(_spec(-1, K, None, TIER_STEPS, 0))
        runs.append(_spec(-1, K, K, TIER_STEPS, 0, tag_extra="_fr"))
    return runs


def wave0_manifest():
    """Extended-budget transition calibration (mandatory, STAGE0_DESIGN.md
    section 12 lesson): unconstrained arm only, 2.5x tier-default steps,
    5 seeds per cell -- section 7's own Wave-0 gate is phrased ">=3/5 seeds
    converge at the primary K by the calibrated budget", which pins the
    denominator at 5, and 2 cells x 5 seeds = 10 runs sits inside the
    design's ~10-15 run budget row. K in {16,32} is the build-time default
    cell pair (mirrors the synthetic design's own 2-cells-per-d Wave-0
    convention); Wave 0's human-read output is ``--primary-k`` for Wave A
    onward, PLUS the R2-4 tau/alignment-threshold re-registration decision
    (recorded in Wave 0's summary BEFORE Wave A's manifest is generated)."""
    steps = int(TIER_STEPS * 2.5)
    runs = []
    for K in (16, 32):
        runs += [_spec(0, K, None, steps, s) for s in range(5)]
    return runs


def waveA_manifest(primary_k=None):
    """Main screening grid at the Wave-0-selected primary K (human gate,
    section 7 item 9 / section 6.1 of the synthetic design's own
    equivalent: NOT assumed a priori). Two parts: (i) unconstrained
    screening across a K-grid, 2 seeds each (BUILD-TIME REASONABLE DEFAULT,
    log-ish spaced around primary_k, capped by pool size -- see
    grammar_rd.py's build_entity_pools docstring: both name pools cover K up
    to 64 at heldout_frac=0.5); (ii) compositional multi-hop at primary_k, 3
    seeds. When primary_k is None (dry-run manifest-SHAPE purposes only),
    K=32 stands in as a placeholder."""
    K = primary_k if primary_k is not None else 32          # PLACEHOLDER (dry-run only)
    steps = int(TIER_STEPS * 2.0)
    k_grid = sorted({max(1, K // 4), max(1, K // 2), K, min(64, int(K * 1.5)), min(64, K * 2)})
    runs = []
    for Kg in k_grid:
        runs += [_spec("A", Kg, None, steps, s) for s in range(2)]
    runs += [_spec("A", K, None, steps, s, tag_extra="_composition") for s in range(3)]
    return runs


def waveBprobe_manifest(primary_k=None):
    """Force-rank calibration probe (do not launch the full grid blind,
    section 7's B-probe row): force-rank at k in {K-1,K,K+1}, primary K, 3
    seeds. 9 runs, matching the design's budget table."""
    K = primary_k if primary_k is not None else 32            # PLACEHOLDER (dry-run only)
    steps = int(TIER_STEPS * 2.0)
    runs = []
    for k in (K - 1, K, K + 1):
        runs += [_spec("Bprobe", K, k, steps, s) for s in range(3)]
    return runs


def waveB_manifest(primary_k=None):
    """Full force-rank straddle grid (PRIMARY causal test, the decisive RD-1
    evidence). LAUNCHED ONLY IF B-probe shows life (enforced in main() via
    --force-launch-b). k in {K-1,K,K+1} x 5 seeds = 15 runs, matching the
    design's stated ~15-20-run range."""
    K = primary_k if primary_k is not None else 32            # PLACEHOLDER (dry-run only)
    steps = int(TIER_STEPS * 2.0)
    runs = []
    for k in (K - 1, K, K + 1):
        runs += [_spec("B", K, k, steps, s) for s in range(5)]
    return runs


MANIFEST_FNS = {"-1": wave_neg1_manifest, "0": wave0_manifest}


def out_path(out_dir, spec):
    return os.path.join(out_dir, f"{spec['name']}.json")


def is_done(out_dir, spec):
    """Validity-checked resume, cross-checked against the spec's own
    identity fields (run_stage0_sweep.py's fix for run_overnight.py's known
    is_done weakness, reused verbatim here). Requires the completion
    sentinel (complete == True)."""
    p = out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return False
        required = ("K", "force_rank_k", "seed", "steps", "complete", "steps_completed")
        if not all(k in d for k in required):
            return False
        if d.get("timed_out"):
            return False
        if d.get("complete") is not True:
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if (d.get("K") != spec["K"] or d.get("seed") != spec["seed"]
                or d.get("steps") != spec["steps"] or d.get("force_rank_k") != spec["force_rank_k"]):
            return False
        # trunc_impl identity (ported from the synthetic sweep's FINDING-1
        # pattern, activated by deviation #6's Wave -1 firing): an eigh
        # result must never satisfy an svd_lowrank spec or vice versa. The
        # cross-check applies to FORCED specs only -- unconstrained runs
        # never truncate, so the impl axis does not exist for them (see
        # main()'s impl-application note).
        if spec["force_rank_k"] is not None and \
                d.get("trunc_impl", "eigh") != spec.get("trunc_impl", "eigh"):
            return False
        return True
    except Exception:
        return False


def build_cmd(spec, out_dir, timeout):
    cmd = [sys.executable, RUN, "--K", str(spec["K"]), "--steps", str(spec["steps"]),
           "--seed", str(spec["seed"]), "--internal-timeout", str(max(1, timeout - 30)),
           "--out", out_path(out_dir, spec)]
    if spec["force_rank_k"] is not None:
        cmd += ["--force-rank-k", str(spec["force_rank_k"])]
        if spec.get("trunc_impl", "eigh") != "eigh":
            cmd += ["--trunc-impl", spec["trunc_impl"]]
    return cmd


def run_smoke(log_dir, gpu):
    print(f"SMOKE GATE (physical GPU {gpu}) ...", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
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


def _run_arm(r):
    """Which R2-1 arm rule applies to this run (section 5.2 + audit MAJOR-2)."""
    fr = r.get("force_rank_k")
    if fr is None:
        return "unconstrained"
    return "causal_k_ge_K" if fr >= r.get("K", 10**9) else "causal_k_lt_K"


def _premise_valid_entry(entry, arm, th):
    """Arm-specific premise classification of ONE per-hop eval entry
    (section 5.2's R2-1 rules + the 2026-07-02 MAJOR-1 value-side addendum,
    re-registered for real-data arms by the §14.7 gate decision,
    2026-07-03), recomputed from the RAW logged diagnostics against the
    run's own recorded thresholds -- never from the convenience booleans,
    so the rule applied here is exactly the pre-registered one even if a
    run's JSON predates a flag. Returns True/False, or None for the k<K
    collapse leg (the bound is never invoked there -- R2-1(iii); its
    numbers are diagnostics, not premise-gated headlines)."""
    if arm == "causal_k_lt_K":
        return None
    # R2-2's frozen rule is PER-ITEM ("cos(k_eff_bind_j, q_eff_query_j) >=
    # 0.9 for ALL K items") -- the gate is alignment_cos_min (the strict
    # min over every item in the eval batches), never the mean (round-2
    # audit pre-registration-fidelity fix; the mean is a logged diagnostic
    # only). A missing field reads as -1.0 -> conservatively invalid.
    align_ok = entry.get("alignment_cos_min", -1.0) >= th.get("alignment_cos", 0.9)
    # §14.7 item (2) (dated 2026-07-03, executed on measured Wave 0 data):
    # tau/tau_v (near-orthonormal-key Gram deviation, anchored on the
    # synthetic harness's orthonormal-key construction) are RETIRED as a
    # validity criterion for real-data arms -- measured real-data key/value
    # Gram deviation fails tau=0.03 unconditionally even on demonstrably
    # binding checkpoints, because real learned representations are
    # linearly independent but never orthonormal (never the bound's actual
    # premise, R2-1's premise (i): linear independence, not
    # orthonormality). The replacement validity stack is the salvage tier
    # on BOTH sides (sigma_K/sigma_1 >= 0.1, R2-1(ii)) + the per-item
    # alignment gate above -- this UNIFIES the "unconstrained" arm's rule
    # with the causal k>=K leg's rule below (both are real-data arms;
    # tau/tau_v stay logged as clean-regime descriptors only, never
    # gating, on both arms alike).
    key_ok = entry.get("salvage_ratio_mean", -1.0) >= th.get("salvage_ratio", 0.1)
    val_ok = entry.get("value_salvage_ratio_mean", -1.0) >= th.get("value_salvage_ratio", 0.1)
    return key_ok and val_ok and align_ok


def aggregate(out_dir, failed, wave):
    """MAJOR-2 (audit round 1, 2026-07-02): headline means are computed ONLY
    over premise-valid checkpoints under the applicable arm rule -- a run's
    number comes from its LAST premise-valid checkpoint (the spec's own
    companion rule: "the last premise-valid checkpoint carries that run's
    number"), and runs with NO premise-valid checkpoint are surfaced as
    n_premise_failed, a separate outcome category alongside (never inside)
    the headline (the "premise-failed" category of section 5.2). The k<K
    collapse leg is never premise-gated (R2-1 iii) -- its cell is labeled
    diagnostic-only and uses final checkpoints. Pre-fix behavior (flat mean
    over all complete runs' final checkpoints) mixed premise-valid and
    premise-invalid runs into one misleading number."""
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
        complete_runs = [r for r in runs if r.get("complete") is True]
        report["n_partial_excluded"] = len(runs) - len(complete_runs)
        by_cell = {}
        for r in complete_runs:
            try:
                fr = "N" if r.get("force_rank_k") is None else r["force_rank_k"]
                key = f"K{r['K']}_fr{fr}"
                arm = _run_arm(r)
                th = r.get("c16_thresholds") or {}
                cell = by_cell.setdefault(key, {
                    "arm": arm, "skip_rate": [],
                    "headline_recovered_frac_h1": [],
                    "headline_entity_subspace_eff_rank_h1": [],
                    "headline_key_gram_deviation_h1": [],
                    "headline_value_gram_deviation_h1": [],
                    "headline_C17_recovered_frac_h1": [],
                    "headline_C19_recovered_frac_h1": [],
                    "headline_steps": [],   # raw list per run (which checkpoint carried each
                                            # run's number) -- NOT averaged; a mean of step
                                            # indices is meaningless
                    "n_premise_valid": 0, "n_premise_failed": 0,
                })
                cell["skip_rate"].append(r.get("skip_rate"))
                ckpts = r.get("checkpoints") or []
                if not ckpts:
                    continue
                h1 = str(min(r.get("H_train", [1])))

                def h1_entry(ck):
                    m2 = ck.get("M2_in_distribution") or {}
                    return m2.get(h1) or (list(m2.values())[0] if m2 else {})

                if arm == "causal_k_lt_K":
                    # never premise-gated (R2-1 iii): final checkpoint,
                    # diagnostic-only cell
                    chosen = ckpts[-1]
                    cell["premise_gating"] = "not_applicable_R2-1(iii)_diagnostics_only"
                else:
                    chosen = None
                    for ck in reversed(ckpts):        # LAST premise-valid carries the number
                        if _premise_valid_entry(h1_entry(ck), arm, th):
                            chosen = ck
                            break
                    if chosen is None:
                        cell["n_premise_failed"] += 1
                        continue
                    cell["n_premise_valid"] += 1
                entry = h1_entry(chosen)
                cell["headline_recovered_frac_h1"].append(entry.get(TAU))
                cell["headline_entity_subspace_eff_rank_h1"].append(
                    entry.get("entity_subspace_effective_rank_mean"))
                cell["headline_key_gram_deviation_h1"].append(entry.get("key_gram_deviation_mean"))
                cell["headline_value_gram_deviation_h1"].append(entry.get("value_gram_deviation_mean"))
                c17 = (chosen.get("C17_heldout_entities") or {}).get(h1) or {}
                cell["headline_C17_recovered_frac_h1"].append(c17.get(TAU))
                c19 = (chosen.get("C19_heldout_template") or {}).get(h1) or {}
                cell["headline_C19_recovered_frac_h1"].append(c19.get(TAU))
                cell["headline_steps"].append(chosen.get("step"))
            except Exception:
                continue
        report["by_cell"] = {
            k: {mk: (v if mk == "headline_steps"
                     else (mean(v) if isinstance(v, list) else v))
                for mk, v in vv.items()}
            for k, vv in sorted(by_cell.items())
        }
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write(f"DeltaNet-RD causal-rank -- wave {wave}\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/deltanet_rd"))
    ap.add_argument("--wave", choices=["-1", "0", "A", "Bprobe", "B", "Reserve"], default=None,
                     help="which wave to launch; REQUIRED unless --dry-run. Waves launch "
                          "separately with a human gate between them (no perpetual refill). "
                          "'Reserve' is accepted for a clear error message only -- NOT "
                          "implemented in this build phase (C18's frozen-pretrained-embedding "
                          "arm, multi-token-name / free-form-template stress arms).")
    ap.add_argument("--primary-k", type=int, default=None,
                     help="Wave A/Bprobe/B: REQUIRED for a real launch. The human's read of "
                          "Wave 0's 'which K is the healthier primary cell' decision.")
    ap.add_argument("--force-launch-b", action="store_true",
                     help="Wave B: REQUIRED for a real launch. The human's read of B-probe's "
                          "'life' verdict (do not launch the full grid blind).")
    ap.add_argument("--gpus", type=int, default=None,
                     help="GPU COUNT to use. REQUIRED for a real launch, NO DEFAULT ON "
                          "PURPOSE: this box runs concurrent waves from other experiments and "
                          "the busy-GPU set changes day to day. Run nvidia-smi immediately "
                          "before every launch and pass the free set explicitly.")
    ap.add_argument("--gpu-offset", type=int, default=None,
                     help="first physical GPU index to use. REQUIRED for a real launch, NO "
                          "DEFAULT ON PURPOSE -- see --gpus.")
    ap.add_argument("--per-gpu", type=int, default=2, help="runs packed per GPU")
    ap.add_argument("--trunc-impl", choices=["eigh", "svd_lowrank"], default="eigh",
                     help="truncation implementation, applied to the wave's FORCE-RANK specs "
                          "and forwarded to every such run (deviation #6's fallback -- Wave -1 "
                          "2026-07-02: all 3 fr cells crashed in eigh's FORWARD on "
                          "real-embedding states). Non-default impls are suffixed into forced "
                          "run names (no resume collisions) and cross-checked by is_done. "
                          "FLAGGED DEVIATION from the synthetic sweep's wave-WIDE application: "
                          "here the impl attaches to FORCED specs only, because unconstrained "
                          "runs never truncate (the impl axis does not exist for them) AND this "
                          "is what lets a re-launch with a different impl resume past already-"
                          "complete unconstrained cells instead of re-running them.")
    ap.add_argument("--timeout", type=float, default=None,
                     help="override the per-run wall timeout (s); default is computed per-spec "
                          "(PROVISIONAL constants -- see module docstring)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    if args.dry_run:
        from collections import Counter
        waves = {
            "-1": wave_neg1_manifest(), "0": wave0_manifest(),
            "A": waveA_manifest(args.primary_k),
            "Bprobe": waveBprobe_manifest(args.primary_k),
            "B": waveB_manifest(args.primary_k),
        }
        total = 0
        for w, runs in waves.items():
            total += len(runs)
            counts = Counter((s["K"], s["force_rank_k"]) for s in runs)
            counts_sorted = dict(sorted(counts.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])))
            print(f"wave {w}: {len(runs)} runs | by (K,force_rank_k) {counts_sorted}")
        sizes = " + ".join(str(len(waves[w])) for w in ("-1", "0", "A", "Bprobe", "B"))
        print(f"CORE TOTAL: {total} runs ({sizes}; vs section 7's own budget-table ESTIMATES "
              f"~6-8 + ~10-15 + ~20-25 + ~9 + ~15-20 -- those are upper-bound guesses, not "
              f"targets; Wave A's K-grid width is a build-time interpretive choice for the "
              f"human to widen/narrow, and Wave B is contingent on the B-probe gate either way)")
        if args.gpus is not None and args.gpu_offset is not None:
            print(f"slots = {args.gpus} gpus x {args.per_gpu} per-gpu = {args.gpus * args.per_gpu} "
                  f"concurrent, on physical GPUs {list(range(args.gpu_offset, args.gpu_offset + args.gpus))}")
        else:
            print("slots: pass --gpus/--gpu-offset to preview (REQUIRED for a real launch, "
                  "no defaults -- check nvidia-smi first).")
        if args.primary_k is None:
            print("NOTE: --primary-k not given -- Wave A/Bprobe/B use a PLACEHOLDER K=32 for "
                  "manifest-SHAPE purposes only; a real launch requires it.")
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)
    if args.wave == "Reserve":
        print("ERROR: the Reserve wave (C18's frozen-pretrained-embedding robustness check, "
              "section 5.2's multi-token-name / free-form-template stress arms) is NOT "
              "implemented in this build phase. Build and independently audit those first "
              "(section 11 sequencing); do not improvise Reserve runs through this harness.",
              file=sys.stderr)
        sys.exit(1)
    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose): the busy-GPU set on this box changes day to day. Run nvidia-smi NOW "
              "and pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)
    if args.wave in ("A", "Bprobe", "B") and args.primary_k is None:
        print(f"ERROR: --wave {args.wave} requires --primary-k (the human gate: Wave 0's "
              f"'which K is healthier' decision).", file=sys.stderr)
        sys.exit(1)
    if args.wave == "B" and not args.force_launch_b:
        print("ERROR: --wave B requires --force-launch-b (the human gate: B-probe's 'life' "
              "verdict -- 'do not launch the full grid blind').", file=sys.stderr)
        sys.exit(1)

    if args.wave in MANIFEST_FNS:
        manifest = MANIFEST_FNS[args.wave]()
    elif args.wave == "A":
        manifest = waveA_manifest(args.primary_k)
    elif args.wave == "Bprobe":
        manifest = waveBprobe_manifest(args.primary_k)
    else:
        manifest = waveB_manifest(args.primary_k)

    # Apply the truncation impl to FORCED specs only (see --trunc-impl's
    # help for why this deviates from the synthetic sweep's wave-wide
    # application). Non-default impls get a name suffix so eigh and
    # svd_lowrank runs of the same (K,k,seed) cell never collide on a
    # resume path -- and is_done cross-checks the recorded trunc_impl.
    for s in manifest:
        if s["force_rank_k"] is not None:
            s["trunc_impl"] = args.trunc_impl
            if args.trunc_impl != "eigh":
                s["name"] += f"_t{args.trunc_impl}"

    out_dir = os.path.join(args.out_dir, f"wave{args.wave}")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir, args.gpu_offset):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate failed; no training launched.\n")
        sys.exit(1)

    physical_gpus = list(range(args.gpu_offset, args.gpu_offset + args.gpus))
    slots = [g for _ in range(args.per_gpu) for g in physical_gpus]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done(out_dir, s)]
    print(f"wave={args.wave}  manifest={len(manifest)}  pending={len(pending)}  "
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
                    else default_timeout(spec["K"], spec["steps"], forced=spec["force_rank_k"] is not None)
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
    print("HUMAN GATE: review results (Wave 0 -> --primary-k; B-probe -> --force-launch-b) "
          "before launching the next wave.", flush=True)


if __name__ == "__main__":
    main()
