"""run_deltanet_sweep.py -- bounded, human-gated wave orchestrator for the
DeltaNet causal-rank experiment. See DELTANET_CAUSAL_RANK_DESIGN.md section
6.4 (manifest table) and section 6.5 (F17, operational-harness build
requirements, blocking).

CLONE of chapter2/run_stage0_sweep.py's robustness pattern (smoke gate,
exception-isolated launch, validity-checked resume, per-run timeout with GPU
quarantine, guarded aggregate) -- deliberately NOT a cross-directory import,
matching this repo's established pod-safety convention (see rank_utils.py's
docstring).

  - NO perpetual refill. Each wave's manifest is fixed and finite; the
    orchestrator drains it once and exits (design section 6.4/6.5).
  - `--wave {-1,0,A,Bprobe,B}` launches exactly one wave (the design's own
    manifest table row set, task brief's explicit list -- the Reserve wave
    needs H>1 multi-head support and the production-block secondary-check
    harness, NEITHER of which this build phase implements; selecting it
    prints a clear NotImplementedError-style message rather than silently
    doing something wrong).
  - Wave A requires `--primary-d` (Wave 0's human-read "healthier cell"
    decision, design section 6.1/6.4/7 item 6 -- explicitly NOT assumed a
    priori). Wave B requires `--primary-k` AND `--force-launch-b` (Wave
    B-probe's human-read "life" verdict, design section 6.4's B-probe
    gate -- "do not launch the full grid blind"). There is no default that
    silently proceeds.
  - Manifest naming carries (wave, variant, d, K, k, seed) -- F17's first
    bullet. "variant" is "bespoke_h1" for every run this build can launch
    (the ONLY architecture this phase implements is the primary H=1 bespoke
    harness); the field is carried anyway so Reserve-wave variants
    (H=2/4 probes, a production-block variant) added later never collide
    with these run names.
  - `is_done` is validity-checked AND cross-checks steps/K/d/force_rank_k/
    trunc_impl/seed recorded INSIDE the result JSON against the spec that
    would produce that path (run_stage0_sweep.py's own fix for
    run_overnight.py's known is_done weakness, reused verbatim here).
  - `--gpus` and `--gpu-offset` are REQUIRED for a real launch, with NO
    defaults (audit round-1 FINDING 2): this box runs concurrent waves from
    OTHER experiments and the busy-GPU set changes day to day (a default
    baked in at build time collided with a bridge wave within 24 hours).
    Check `nvidia-smi` immediately before every launch and pass the free
    set explicitly.
  - `--trunc-impl {eigh,svd_lowrank}` (default eigh) is applied wave-wide
    and forwarded to every run (audit round-1 FINDING 1: the eigh force-
    rank path's measured skip rates fire the F13 stop branch -- see
    model_dn.truncate_to_rank_svd_lowrank and the measured table in the
    _PER_STEP_S comment below). Non-default impls are carried in the run
    NAME (no resume collisions between impls) and cross-checked by is_done.

Usage (GPU list is an example -- check nvidia-smi first, see above):
  python run_deltanet_sweep.py --dry-run
  python run_deltanet_sweep.py --wave -1      --out-dir results/deltanet --gpus 1 --gpu-offset 7
  python run_deltanet_sweep.py --wave 0       --out-dir results/deltanet --gpus 1 --gpu-offset 7
  python run_deltanet_sweep.py --wave A       --out-dir results/deltanet --gpus 1 --gpu-offset 7 --primary-d 64
  python run_deltanet_sweep.py --wave Bprobe  --out-dir results/deltanet --gpus 1 --gpu-offset 7 --primary-d 64 --primary-k 32
  python run_deltanet_sweep.py --wave B       --out-dir results/deltanet --gpus 1 --gpu-offset 7 --primary-d 64 --primary-k 32 --force-launch-b
  python run_deltanet_sweep.py --wave ReserveMH --out-dir results/deltanet --gpus 1 --gpu-offset 7 --primary-d 64 --primary-k 32
      # F11 pre-registered H in {2,4} multi-head probe ONLY (unconstrained, primary cell,
      # --reserve-mh-seeds seeds each, default 3 -> 6 runs). NOT the rest of the Reserve
      # wave (production-block check, Gaussian-key order probe) -- see 'Reserve''s own
      # error text for what remains unimplemented.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_deltanet.py")
TAU = "recovered_frac@0.9"
VARIANT = "bespoke_h1"    # the only architecture this build phase implements (design section 4.2/4.3, C11)

# Provisional tier-default step counts for d in {64, 128} -- NOT yet
# calibrated against real wall-clock (that is exactly Wave -1's job, F13).
# d=64's value matches chapter2/run_stage0_sweep.py's own TIER_STEPS[64]
# for continuity; d=128 extrapolates modestly. Wave 0 runs at 2.5x these
# (design section 6.4), matching Stage 0's own convention for budgeting the
# "late, seed-stochastic phase transition" lesson (STAGE0_DESIGN.md section
# 12) from the start rather than discovering it after a wasted first round.
TIER_STEPS = {64: 10000, 128: 12000}

# MEASURED per-step wall-clock (s/step), keyed by (d, force_rank_arm) --
# audit round-1 FINDING 3, constants baked from the probe below (NOT
# placeholders). Methodology: probe_trunc.py (this directory), one cell per
# run on this box's GPU 7 (H100 80GB), batch=128, lr=3e-4, seed=0, K=d/2
# (each d's LARGEST manifest K, so these are conservative for smaller-K
# cells), warmup=10 steps excluded from timing, measured 2026-07-02 on the
# deployed harness (fp32, pure-PyTorch recurrence, C15 assert included in
# the force-rank path -- i.e. the real per-step cost, not a stripped loop).
#
#   cell                     impl          skip@150   skip@400   per-step
#   d=64  K=32 unconstrained --            0.00%      --          46 ms
#   d=128 K=64 unconstrained --            0.00%      --          75 ms
#   d=64  K=32 fr=32         eigh          30.00%     71.25%     295 ms
#   d=64  K=32 fr=32         svd_lowrank    0.00%      0.00%     468 ms
#   d=128 K=64 fr=64         eigh          82.00%     93.00%     764 ms
#   d=128 K=64 fr=64         svd_lowrank    0.00%      0.00%    1132 ms
#
# Readings (FINDING 1 + FINDING 3):
#  - The eigh force-rank arm fires the design's F13 STOP branch (skip rate
#    > 0.1%, section 6.4) by ORDERS OF MAGNITUDE at both d, and the rate
#    GROWS with training progress (30%->71% at d=64) -- consistent with,
#    and worse than, the independent audit's own measurements (2% @ d=64,
#    37% @ 150 / 75% @ 400 steps @ d=128; different snapshot, same
#    conclusion). eigh force-rank runs are effectively dead at both d.
#  - svd_lowrank measured ZERO skips over 400 steps at both d -- the
#    pre-registered mitigation works at probe scale. It costs ~1.5x eigh's
#    per-step time; the (d, True) constants below use the svd_lowrank
#    numbers (the slower impl) so timeouts cover either choice.
#  - PROMINENT d-SELECTION FLAG (for Wave 0's --primary-d human gate):
#    at these constants a d=128 force-rank run at 2x tier steps (24000)
#    costs ~7.5 h wall (~3.8 h at tier default) vs d=64's ~2.6 h -- ~3x --
#    and d=128 is also the LESS numerically stable cell under eigh (93% vs
#    71% skip @ 400). Rough wave totals (training only, ex-margin):
#    d=64-primary path ~= 5.5 (W-1) + 5.7 (W0) + 3.3 (WA) + 23.5 (Bprobe)
#    + 39 (WB) ~= 77 GPU-h -- fits the design's <=120 ceiling; a
#    d=128-primary path ~= 199 GPU-h -- would BLOW the ceiling on Wave B
#    alone. Pick d=128 only with an explicit re-priced budget.
_PER_STEP_S = {(64, False): 0.05, (64, True): 0.50, (128, False): 0.08, (128, True): 1.15}
CKPT_EVERY = 2000

# F11 / Reserve-wave multi-head probe timing -- MEASURED 2026-07-03 on this
# box (GPU 6, box concurrently running >=2 other builders' waves on GPUs
# 0-5/7 -- see the build report; not an idle-box number) via a clean
# 2-point linear timing probe: run_deltanet.py --H {2,4} --d 64 --K 32 at
# steps in {200,600} with --ckpt-every set above steps (so exactly ONE
# checkpoint-eval fires in both runs), then per_step = (wall_600 -
# wall_200)/400 and eval_cost = wall_200 - 200*per_step. Methodology note
# (CLAUDE.md: "a calibration run before a big sweep is mandatory, not
# optional" -- this replaces the earlier PROVISIONAL guess entirely, not a
# supplement to it):
#   H   per_step (train-only)   per-checkpoint eval_cost (n_batches=4 default)
#   1   42.35 ms                25.16 s   (H=1 sanity check -- matches the
#                                           existing _PER_STEP_S[(64,False)]
#                                           = 46ms/step within measurement
#                                           noise, on a box now also running
#                                           concurrent unrelated waves)
#   2   42.24 ms                49.17 s   (~= H=1's per-step -- heads are
#                                           folded into ONE batched
#                                           delta_rule_state call, not H
#                                           separate Python-loop calls; the
#                                           GPU is underutilized enough at
#                                           this d/K that doubling the
#                                           folded batch dim costs ~nothing)
#   4   53.46 ms                93.06 s   (~1.26x H=1's per-step, NOT 4x --
#                                           same batch-fold reasoning;
#                                           eval_cost scales closer to
#                                           linear in H, as expected: H
#                                           separate per-head SVDs per hop)
# d=128 cells are UNMEASURED (this build's launch manifest is d=64 only,
# per the task brief's "primary d=64 K=32") -- the fallback below is a
# documented, untested extrapolation, not a measurement.
_PER_STEP_S_MH = {(64, 2): 0.04224, (64, 4): 0.05346}
_EVAL_COST_S_MH = {(64, 2): 49.17, (64, 4): 93.06}


def default_timeout_mh(d: int, H: int, steps: int, margin: float = 1.6,
                        ckpt_every: int = CKPT_EVERY) -> int:
    """F11 multi-head sibling of default_timeout (unconstrained-only manifest
    -- no `forced` axis). MEASURED per-step + per-checkpoint-eval constants
    at d=64 (see the table above); (d,H) cells outside that table (i.e.
    d=128, unmeasured) fall back to a documented, conservative formula."""
    n_ckpts = steps // ckpt_every + 1
    per_step = _PER_STEP_S_MH.get((d, H), 0.05 * H)
    eval_cost = _EVAL_COST_S_MH.get((d, H), (10.0 + 0.25 * d) * H)
    base = (per_step * steps + n_ckpts * eval_cost) * margin
    return int(base)


def default_timeout(d: int, steps: int, K: int, forced: bool, margin: float = 1.6,
                     ckpt_every: int = CKPT_EVERY) -> int:
    """Wall timeout = (train + checkpoint-eval) * margin, from the MEASURED
    per-step constants above. Constants were measured at the LARGEST-K cell
    of each d's manifest (K=d/2), so they are a conservative upper bound for
    every smaller-K cell of the same d (per-step cost grows with T_bind =
    K*(1+buf_len)); K is accepted for signature clarity but deliberately
    not used to scale DOWN. The checkpoint-eval term is an estimate (8 hops
    x 4 batches x forward + 2 SVD-metric passes per hop), deliberately
    generous."""
    n_ckpts = steps // ckpt_every + 1
    per_step = _PER_STEP_S.get((d, forced), 1.2)
    base = (per_step * steps + n_ckpts * (10.0 + 0.25 * d)) * margin
    return int(base)


def _spec(wave, d, K, k, steps, seed, tag_extra="", H=1, variant=None):
    """H=1 (default): byte-identical spec shape to every already-completed
    wave (VARIANT stays "bespoke_h1", no "_H.." suffix -- resume paths for
    completed Wave -1/0/A/Bprobe runs are UNCHANGED). H>1 (F11 / Reserve-
    wave multi-head probe): the run name AND the returned spec dict both
    carry H explicitly (F17's "run naming carries (wave, variant, d, K, k,
    seed)" extended with H, per the task brief's "add H to run names/
    is_done identity" instruction) -- an H=2 and an H=4 run of the same
    (wave,d,K,k,seed) can never collide on a resume path."""
    v = variant or VARIANT
    tag = f"w{wave}_{v}_d{d}_K{K}_fr{'N' if k is None else k}_s{seed}{tag_extra}"
    if H != 1:
        tag += f"_H{H}"
    return {"wave": str(wave), "variant": v, "d": d, "K": K, "force_rank_k": k,
            "steps": steps, "seed": seed, "name": tag, "H": H}


# ---------------------------------------------------------------------------
# Manifests -- one function per wave. See module docstring for the human-gate
# arguments each wave beyond -1/0 requires for a REAL (non-dry-run) launch.
# ---------------------------------------------------------------------------

def wave_neg1_manifest():
    """Timing + instability calibration (mandatory, CLAUDE.md). (a) 2
    unconstrained runs (d=64,K=32 and d=128,K=64 -- the LARGER K of each
    d's Wave-0 pair below, a deliberate worst-case-cost/most-rank-headroom
    pick, mirroring run_stage0_sweep.py's own "friendlier probe" reasoning
    for an otherwise-unpinned calibration cell); (b) the SAME 2 cells with
    the two-kernel-split force-rank arm at k=K (a neutral, non-adversarial
    rank -- B-probe is where k straddles K on purpose). 4 runs total,
    matching the design's budget table exactly."""
    runs = []
    for d, K in ((64, 32), (128, 64)):
        runs.append(_spec(-1, d, K, None, TIER_STEPS[d], 0))
        runs.append(_spec(-1, d, K, K, TIER_STEPS[d], 0, tag_extra="_fr"))
    return runs


def wave0_manifest():
    """Extended-budget transition calibration (mandatory, STAGE0_DESIGN.md
    section 12 lesson): unconstrained arm only, 2.5x tier-default steps,
    >=3 seeds per primary cell: d=64,K in {16,32} and d=128,K in {32,64}
    (design section 6.4's exact cell list). 4 cells x 3 seeds = 12 runs.

    KNOWN LIMITATION for the d=64, K=16 cell (audit round-1 MINOR-4): with
    the default hop sets, H_extra's h=21 has 21 % 16 == 5, colliding with
    H_test hop h=5's residue -- at K=16, h=21 is NOT a new held-out residue,
    it re-measures effective distance 5 at 4x the raw iteration count (read
    it via the per-hop `effective_hop` field as depth-amplification data
    only). The K=32 and K=64 cells are unaffected (21 % 32 == 21,
    21 % 64 == 21).

    d-SELECTION COST NOTE (audit round-1 FINDING 3): the measured per-step
    costs in _PER_STEP_S make d=128 force-rank cells ~3x the wall-clock of
    d=64 AND ~19x less numerically stable under eigh force-rank (see the
    measured table in the _PER_STEP_S comment). Wave 0's --primary-d human
    decision should weigh BOTH facts, not accuracy alone."""
    runs = []
    for d, Ks in ((64, (16, 32)), (128, (32, 64))):
        steps = int(TIER_STEPS[d] * 2.5)
        for K in Ks:
            runs += [_spec(0, d, K, None, steps, s) for s in range(3)]
    return runs


def waveA_manifest(primary_d=None):
    """Main screening grid at the Wave-0-selected primary d (design section
    6.1/6.4/7 item 6: NOT assumed a priori -- a human reads Wave 0's
    per-cell results and supplies --primary-d). Two parts:
      (i) unconstrained M1/M2-equivalent across a K-grid, 2 seeds each --
          the design text does not pin an exact K-grid ("across K grid");
          this is a BUILD-TIME REASONABLE DEFAULT (log-ish spaced around
          the Wave-0 K pair for this d, clipped to [1,d]), flagged here for
          human confirmation/adjustment before a real launch, exactly the
          same "PLACEHOLDER" discipline run_stage0_sweep.py's own Wave A/B
          manifests use for underspecified pieces.
      (ii) Task-E-equivalent compositional multi-hop at the primary K
           (the larger Wave-0 K for this d), 3 seeds -- pinned by the
           design text exactly.
    When primary_d is None (dry-run manifest-SHAPE purposes only), d=64
    stands in as a placeholder so the run COUNT is inspectable before Wave
    0 finishes."""
    d = primary_d if primary_d is not None else 64          # PLACEHOLDER (dry-run only)
    k_pair = (16, 32) if d == 64 else (32, 64)
    k_primary = k_pair[1]
    steps = int(TIER_STEPS.get(d, 10000) * 2.0)
    # NOTE (audit round-1 MINOR-4): at d=64 this grid includes K=16, where
    # H_extra's h=21 collides with H_test h=5's residue (21 % 16 == 5) --
    # see wave0_manifest's KNOWN LIMITATION note; applies identically here.
    k_grid = sorted({max(1, k_primary // 4), max(1, k_primary // 2), k_primary,
                      min(d, int(k_primary * 1.5)), min(d, k_primary * 2)})
    runs = []
    for K in k_grid:
        runs += [_spec("A", d, K, None, steps, s) for s in range(2)]
    runs += [_spec("A", d, k_primary, None, steps, s, tag_extra="_composition") for s in range(3)]
    return runs


def waveBprobe_manifest(primary_d=None, primary_k=None):
    """Force-rank calibration probe (Task E M4_E discipline -- do not
    launch the full grid blind, design section 6.4). Two-kernel-split
    force-rank at k in {K-1, K, K+1}, primary d/K, 3 seeds. 9 runs, exactly
    matching the design's budget table."""
    d = primary_d if primary_d is not None else 64            # PLACEHOLDER (dry-run only)
    K = primary_k if primary_k is not None else (32 if d == 64 else 64)   # PLACEHOLDER
    steps = int(TIER_STEPS.get(d, 10000) * 2.0)
    runs = []
    for k in (K - 1, K, K + 1):
        runs += [_spec("Bprobe", d, K, k, steps, s) for s in range(3)]
    return runs


def waveB_manifest(primary_d=None, primary_k=None):
    """Full force-rank straddle grid (PRIMARY causal test, M3-equivalent).
    LAUNCHED ONLY IF B-probe shows life (design section 6.4's explicit
    gate) -- enforced in main() via --force-launch-b, not by this function.
    k straddle grid {K-1, K, K+1} (matching B-probe's own straddle -- the
    design text pins "5 seeds" but leaves the exact grid width unspecified;
    3 points x 5 seeds = 15 lands inside the design's stated ~15-20-run
    range) x 5 seeds."""
    d = primary_d if primary_d is not None else 64            # PLACEHOLDER (dry-run only)
    K = primary_k if primary_k is not None else (32 if d == 64 else 64)   # PLACEHOLDER
    steps = int(TIER_STEPS.get(d, 10000) * 2.0)
    runs = []
    for k in (K - 1, K, K + 1):
        runs += [_spec("B", d, K, k, steps, s) for s in range(5)]
    return runs


def wave_reserve_multihead_manifest(primary_d=None, primary_k=None, n_seeds=3):
    """F11 / Reserve-wave multi-head probe (design section 6.4/8/10, task
    brief's explicit ask): H in {2,4}, UNCONSTRAINED (force_rank_k=None)
    ONLY, at the PRIMARY CELL only (no K-grid, no d-grid -- F11
    pre-registers "2-4 runs at the primary cell", not a full multi-head
    study, explicitly deferred per section 9), n_seeds seeds each (2-3 per
    the task brief) x 2.5x tier-default steps (Wave 0's own extended-budget
    convention, applied here too so a late-transition result isn't missed
    by an under-budgeted probe). n_seeds=3 -> 6 runs; at the MEASURED
    per-step/per-checkpoint-eval constants above (steps=25000,
    ckpt_every=2000 -> 13 checkpoints), expected wall-clock (no margin) is
    ~0.47 GPU-h/run at H=2 and ~0.71 GPU-h/run at H=4 -> ~1.41 + ~2.12 =
    ~3.5 GPU-h total for 3+3 runs, inside the task brief's ~2-4 GPU-h
    budget (measured, not the earlier provisional guess).

    NOT the full Reserve wave (design section 6.4's Reserve row also lists
    a production-block secondary check and a Gaussian-key order-dependence
    probe -- NEITHER is built here; this manifest is the F11 multi-head
    sub-probe only). Kept under its own wave name ("ReserveMH") rather than
    folded into "Reserve" so the sweep script never silently claims full
    Reserve-wave coverage it does not have (run_deltanet_sweep.py's
    existing "Reserve" wave selector still errors with its original,
    accurate message for the other two items)."""
    d = primary_d if primary_d is not None else 64            # PLACEHOLDER (dry-run only)
    K = primary_k if primary_k is not None else (32 if d == 64 else 64)   # PLACEHOLDER
    steps = int(TIER_STEPS.get(d, 10000) * 2.5)
    runs = []
    for H in (2, 4):
        runs += [_spec("ReserveMH", d, K, None, steps, s, H=H, variant=f"multihead_H{H}")
                 for s in range(n_seeds)]
    return runs


MANIFEST_FNS = {"-1": wave_neg1_manifest, "0": wave0_manifest}


def out_path(out_dir, spec):
    return os.path.join(out_dir, f"{spec['name']}.json")


def is_done(out_dir, spec):
    """Validity-checked resume, cross-checked against the spec's own
    identity fields (run_stage0_sweep.py's fix for run_overnight.py's known
    is_done weakness, reused verbatim). Requires the completion sentinel
    (complete == True, written ONLY by run_deltanet.py main()'s post-
    training write -- every incremental checkpoint dump has complete=False,
    F17/FATAL-1 discipline)."""
    p = out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return False
        required = ("K", "d", "force_rank_k", "seed", "steps", "complete", "steps_completed")
        if not all(k in d for k in required):
            return False
        if d.get("timed_out"):
            return False
        if d.get("complete") is not True:
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if (d.get("K") != spec["K"] or d.get("d") != spec["d"]
                or d.get("seed") != spec["seed"] or d.get("steps") != spec["steps"]
                or d.get("force_rank_k") != spec["force_rank_k"]):
            return False
        # trunc_impl identity (audit FINDING 1): an eigh result must never
        # satisfy an svd_lowrank spec or vice versa. .get defaults keep old
        # eigh-era JSONs (no trunc_impl field) valid for eigh specs only.
        if d.get("trunc_impl", "eigh") != spec.get("trunc_impl", "eigh"):
            return False
        # H identity (F11 / Reserve-wave multi-head probe, task brief's
        # "add H to run names/is_done identity"): an H=2 result must never
        # satisfy an H=4 spec or vice versa. .get defaults keep every
        # pre-F11 result JSON (no "H" field -- every completed Wave
        # -1/0/A/Bprobe run) valid for H=1 specs only, unchanged.
        if d.get("H", 1) != spec.get("H", 1):
            return False
        return True
    except Exception:
        return False


def build_cmd(spec, out_dir, timeout):
    cmd = [sys.executable, RUN, "--d", str(spec["d"]), "--K", str(spec["K"]),
           "--steps", str(spec["steps"]), "--seed", str(spec["seed"]),
           "--internal-timeout", str(max(1, timeout - 30)),
           "--out", out_path(out_dir, spec)]
    if spec["force_rank_k"] is not None:
        cmd += ["--force-rank-k", str(spec["force_rank_k"])]
    if spec.get("trunc_impl", "eigh") != "eigh":
        cmd += ["--trunc-impl", spec["trunc_impl"]]
    if spec.get("H", 1) != 1:
        cmd += ["--H", str(spec["H"])]
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
        complete_runs = [r for r in runs if r.get("complete") is True]
        report["n_partial_excluded"] = len(runs) - len(complete_runs)
        by_cell = {}
        for r in complete_runs:
            try:
                fr = "N" if r.get("force_rank_k") is None else r["force_rank_k"]
                H = r.get("H", 1)
                key = f"d{r['d']}_K{r['K']}_fr{fr}" + (f"_H{H}" if H != 1 else "")
                if H == 1:
                    cell = by_cell.setdefault(key, {"skip_rate": [], "final_recovered_frac_h1": [],
                                                     "final_entity_subspace_eff_rank_h1": []})
                    cell["skip_rate"].append(r.get("skip_rate"))
                    ckpts = r.get("checkpoints") or []
                    if ckpts:
                        h1 = str(min(r.get("H_train", [1])))
                        m2 = ckpts[-1].get("M2_in_distribution", {})
                        entry = m2.get(h1) or (list(m2.values())[0] if m2 else {})
                        cell["final_recovered_frac_h1"].append(entry.get(TAU))
                        cell["final_entity_subspace_eff_rank_h1"].append(
                            entry.get("entity_subspace_effective_rank_mean"))
                else:
                    # F11 multi-head cell (task brief: "add H to run names/is_done identity"):
                    # the PRIMARY metric is the joint bound Sigma_head rank(S_T^(head)) vs K,
                    # not a single-head entity-subspace number. (Per-head values are LIST-
                    # valued in the per-run JSON -- deliberately NOT summarized here, since
                    # mean() below is scalar-only; read the per-run checkpoints for the
                    # per-head breakdown.)
                    cell = by_cell.setdefault(key, {"skip_rate": [], "final_recovered_frac_h1": [],
                                                     "final_joint_entity_subspace_rank_sum_h1": []})
                    cell["skip_rate"].append(r.get("skip_rate"))
                    ckpts = r.get("checkpoints") or []
                    if ckpts:
                        h1 = str(min(r.get("H_train", [1])))
                        m2 = ckpts[-1].get("M2_in_distribution", {})
                        entry = m2.get(h1) or (list(m2.values())[0] if m2 else {})
                        cell["final_recovered_frac_h1"].append(entry.get(TAU))
                        cell["final_joint_entity_subspace_rank_sum_h1"].append(
                            entry.get("joint_entity_subspace_effective_rank_sum_mean"))
            except Exception:
                continue
        report["by_cell"] = {k: {mk: mean(v) for mk, v in vv.items()} for k, vv in sorted(by_cell.items())}
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write(f"DeltaNet causal-rank -- wave {wave}\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/deltanet"))
    ap.add_argument("--wave", choices=["-1", "0", "A", "Bprobe", "B", "Reserve", "ReserveMH"], default=None,
                     help="which wave to launch; REQUIRED unless --dry-run. Waves launch "
                          "separately with a human gate between them (no perpetual refill). "
                          "'Reserve' is accepted for a clear error message only -- it is "
                          "NOT implemented in this build phase (see the error text). "
                          "'ReserveMH' IS implemented: the F11 pre-registered H in {2,4} "
                          "multi-head probe ONLY (unconstrained, primary cell) -- NOT the "
                          "rest of the Reserve wave (production-block check, Gaussian-key "
                          "order probe), which stays unimplemented under plain 'Reserve'.")
    ap.add_argument("--reserve-mh-seeds", type=int, default=3,
                     help="ReserveMH only: seeds per H value (2 or 3 per the task brief's "
                          "4-6 total cells -- default 3 gives 6 cells, H in {2,4}).")
    ap.add_argument("--primary-d", type=int, default=None,
                     help="Wave A/Bprobe/B: REQUIRED for a real launch. The human's read of "
                          "Wave 0's 'which d is the healthier primary cell' decision "
                          "(design section 6.1/6.4/7 item 6).")
    ap.add_argument("--primary-k", type=int, default=None,
                     help="Wave Bprobe/B: REQUIRED for a real launch. The primary K "
                          "(design section 6.4) around which the force-rank straddle is built.")
    ap.add_argument("--force-launch-b", action="store_true",
                     help="Wave B: REQUIRED for a real launch. The human's read of B-probe's "
                          "'life' verdict (design section 6.4: 'do not launch the full grid "
                          "blind' -- >=1 seed reaching non-trivial recovery at k>=K and a "
                          "step relative to k<K).")
    ap.add_argument("--gpus", type=int, default=None,
                     help="GPU COUNT to use. REQUIRED for a real launch, NO DEFAULT ON "
                          "PURPOSE (audit round-1 FINDING 2): this box runs concurrent "
                          "waves from other experiments and the busy-GPU set changes day "
                          "to day -- a hardcoded default built one day collided with a "
                          "running bridge wave the next. Run nvidia-smi immediately "
                          "before every launch and pass the free set explicitly.")
    ap.add_argument("--gpu-offset", type=int, default=None,
                     help="first physical GPU index to use. REQUIRED for a real launch, "
                          "NO DEFAULT ON PURPOSE -- see --gpus.")
    ap.add_argument("--trunc-impl", choices=["eigh", "svd_lowrank"], default="eigh",
                     help="truncation implementation applied WAVE-WIDE and forwarded to "
                          "every run (audit round-1 FINDING 1; see module docstring). "
                          "Non-default impls are suffixed into every run name so eigh and "
                          "svd_lowrank runs of the same cell never collide on resume.")
    ap.add_argument("--per-gpu", type=int, default=2, help="runs packed per GPU")
    ap.add_argument("--timeout", type=float, default=None,
                     help="override the per-run wall timeout (s); default is computed "
                          "per-spec (PROVISIONAL constants -- see module docstring)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    if args.dry_run:
        from collections import Counter
        waves = {
            "-1": wave_neg1_manifest(), "0": wave0_manifest(),
            "A": waveA_manifest(args.primary_d),
            "Bprobe": waveBprobe_manifest(args.primary_d, args.primary_k),
            "B": waveB_manifest(args.primary_d, args.primary_k),
            "ReserveMH": wave_reserve_multihead_manifest(args.primary_d, args.primary_k,
                                                          args.reserve_mh_seeds),
        }
        total = 0
        for w, runs in waves.items():
            total += len(runs)
            counts = Counter((s['d'], s['K'], s['force_rank_k']) for s in runs)
            counts_sorted = dict(sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                                          -1 if kv[0][2] is None else kv[0][2])))
            print(f"wave {w}: {len(runs)} runs | by (d,K,force_rank_k) {counts_sorted}")
        print(f"CORE TOTAL: {total} runs  (this build's manifest: 4 + 12 + 13(A, build-time "
              f"default K-grid) + 9 + 15 = 53, vs. the design's own budget-table ESTIMATE "
              f"4+12+22+9+15-20=~62-67 -- section 6.4 explicitly flags these as upper-bound "
              f"guesses, not targets; Wave A's K-grid width is a build-time interpretive "
              f"choice for the human to widen/narrow, and Wave B is contingent on the "
              f"B-probe gate either way)")
        if args.gpus is not None and args.gpu_offset is not None:
            print(f"slots = {args.gpus} gpus x {args.per_gpu} per-gpu = {args.gpus * args.per_gpu} "
                  f"concurrent, on physical GPUs {list(range(args.gpu_offset, args.gpu_offset + args.gpus))}")
        else:
            print("slots: pass --gpus/--gpu-offset to preview (REQUIRED for a real launch, "
                  "no defaults -- check nvidia-smi first; see --gpus help / FINDING 2).")
        if args.primary_d is None:
            print("NOTE: --primary-d not given -- Wave A/Bprobe/B use a PLACEHOLDER d=64 for "
                  "manifest-SHAPE purposes only; a real launch requires it.")
        if args.primary_k is None:
            print("NOTE: --primary-k not given -- Wave Bprobe/B use a PLACEHOLDER K for "
                  "manifest-SHAPE purposes only; a real launch requires it.")
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)
    if args.wave == "Reserve":
        print("ERROR: the Reserve wave (design section 6.4: production-block secondary "
              "check with C13/C14 on the real fla-org block, Gaussian-key order-dependence "
              "probe, H in {2,4} multi-head per-head-bound probe) is NOT FULLY implemented "
              "in this build phase. The H in {2,4} multi-head probe (F11) IS implemented --  "
              "use --wave ReserveMH. The production-block secondary check and the "
              "Gaussian-key order-dependence probe still require (a) an fla production-"
              "block harness with the extended two-sided blank-out and (b) neither exists "
              "yet. Build and independently audit those first (design section 9 "
              "sequencing); do not improvise them through the bespoke harness.", file=sys.stderr)
        sys.exit(1)
    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults "
              "on purpose, audit round-1 FINDING 2): the busy-GPU set on this box changes "
              "day to day as other experiments' waves come and go. Run nvidia-smi NOW and "
              "pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)
    if args.wave in ("A", "Bprobe", "B", "ReserveMH") and args.primary_d is None:
        print(f"ERROR: --wave {args.wave} requires --primary-d (the human gate: Wave 0's "
              f"'which d is healthier' decision -- ReserveMH runs at the SAME primary cell "
              f"as Wave A/Bprobe/B, section 6.4/8/10).", file=sys.stderr)
        sys.exit(1)
    if args.wave in ("Bprobe", "B", "ReserveMH") and args.primary_k is None:
        print(f"ERROR: --wave {args.wave} requires --primary-k (the human gate: the primary "
              f"K around which the force-rank straddle -- or, for ReserveMH, the primary-"
              f"cell multi-head probe -- is built).", file=sys.stderr)
        sys.exit(1)
    if args.wave == "B" and not args.force_launch_b:
        print("ERROR: --wave B requires --force-launch-b (the human gate: B-probe's 'life' "
              "verdict -- design section 6.4 'do not launch the full grid blind').",
              file=sys.stderr)
        sys.exit(1)

    if args.wave in MANIFEST_FNS:
        manifest = MANIFEST_FNS[args.wave]()
    elif args.wave == "A":
        manifest = waveA_manifest(args.primary_d)
    elif args.wave == "Bprobe":
        manifest = waveBprobe_manifest(args.primary_d, args.primary_k)
    elif args.wave == "ReserveMH":
        manifest = wave_reserve_multihead_manifest(args.primary_d, args.primary_k, args.reserve_mh_seeds)
    else:
        manifest = waveB_manifest(args.primary_d, args.primary_k)

    # Apply the wave-wide truncation impl (FINDING 1). Non-default impls get
    # a name suffix so eigh and svd_lowrank runs of the same (d,K,k,seed)
    # cell never collide on a resume path (F17 naming rule) -- and is_done
    # cross-checks the recorded trunc_impl either way.
    for s in manifest:
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
                if args.timeout is not None:
                    timeout = args.timeout
                elif spec.get("H", 1) != 1:
                    timeout = default_timeout_mh(spec["d"], spec["H"], spec["steps"])
                else:
                    timeout = default_timeout(spec["d"], spec["steps"], spec["K"],
                                               forced=spec["force_rank_k"] is not None)
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
    print("HUMAN GATE: review results (Wave 0 -> --primary-d; B-probe -> --force-launch-b) "
          "before launching the next wave.", flush=True)


if __name__ == "__main__":
    main()
