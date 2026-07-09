"""fixscale_wave.py -- FIX-AT-SCALE wave (FROZEN_BIAS_LM_DESIGN.md sec 13, Rev 1,
c6436fb; micro-attack-round-2-CLEARED sec 13.15) FULL SWEEP orchestrator: the
28-cell n=3/n=3 primary manifest (arm_off, arm_per_token) + arm_global_probe
(n=1), the sec 13.10 gate-5 blind-pin (write BEFORE arm_per_token/arm_global_probe
launch, assert_blind_not_broken WIRED at the launch call site, not merely
documented), and the sec 13.8 1.5x per-cell circuit breaker. This is the BUILD
stage sec 13.12's build list describes; NOT the gate-tier script
(fixscale_gate_tier.py/fixscale_bank.sh, already deployed+running on the box per
commit ea5c3d7 -- timing pilots + arm_off seed-0 calibration cells on GPUs 1-4).
This module BUILDS ON that gate tier rather than duplicating it: an arm_off
seed=0 cell is satisfied by the gate tier's own
`results/fixscale/calib/fixscale_calib_off_<scale>_<corpus>_s0.json` output when
present and complete (see `gate_tier_calib_out_path` / `cell_state` below) --
the resume-safe convention recognizes it directly, never re-runs it.

Design inputs (all read-only, none of these files are edited by this build):
  - lm_pretrain_rd.py's EXISTING CLI already runs every arm this wave needs
    (`--frozen-bias-arm {off,per_token,global}`) at both new d_state shapes
    (64/128) -- confirmed by the gate tier's own box smoke (MANIFEST.md item
    [17]). `global` IS `arm_global_probe`'s construction already
    (`frozen_bias_global_vector`: `b_global = normalize(mean_i B[i])`,
    lm_pretrain_rd.py:217-221) -- no new model-side code, this wave is pure
    orchestration.
  - lm_pretrain_rd.py's `run_name` carries no frozen-bias-arm tag (confirmed,
    MANIFEST.md finding) -- every cell here gets its OWN `--ckpt-dir` (never
    relying on run_name for collision-avoidance), same fix the gate tier
    already applies.
  - bands_pinned_frozenbias.py's `write_bands_pinned_frozenbias` /
    `validate_bands_pinned_frozenbias` / `assert_blind_not_broken_frozenbias`
    ARE "rung-1's own BANDS_PINNED-FrozenBias.json pattern" sec 13.10 gate 5
    names -- reused UNMODIFIED here, only `path` is new (one file per scale).
    This is the semantically-correct reuse target for this wave's val-loss +
    span_frac schema (as opposed to key_anchoring.py's own
    assert_blind_not_broken, which is the K-anchoring-exactness-sweep's
    DIFFERENT drift-based schema -- read for precedent per the build brief,
    not literally called; documented here so a build audit can verify the
    substitution is deliberate, not a missed instruction).
  - frozen_bias_retrofit_eval_rd.py's `apply_frozen_bias_blend`/
    `capture_raw_keys`/`build_frozen_bias_table`/`frozen_bias_global_vector`
    are reused for the comparator (see `run_shared_comparator_measurement`
    below) -- sec 13.15 finding 1's shared-forward-pass fix.

Cost-note header (sec 13.15 finding 1, BINDING): the arm_global_probe
comparator is priced INSIDE the primary arm_off' retrofit pass, not as a
separate eval-only line -- `run_shared_comparator_measurement` captures
k_raw ONCE per (checkpoint, corpus) via `capture_raw_keys` (the one real
forward pass) and derives span_frac for THREE modes (per_token/global/kraw)
from that single capture via `apply_frozen_bias_blend`, a cheap O(batch*seq*d)
tensor op, not a second model pass. Calling
`frozen_bias_retrofit_eval_rd.run_retrofit_measurement` once per `--mode`
(the naive approach) would re-run `capture_raw_keys` per mode -- 3x the
forward passes for identical captured keys. This build implements the
shared-pass branch (the PRIMARY-satisfying option sec 13.15 finding 1
offers); the disclosed fallback (booking +-2.79 GPU-h, 2x) is NOT invoked.

Subcommands:
  manifest [--scale S] [--phase {arm_off,post_pin}]   print/dump the cell list
  cell --scale S --corpus C --arm A --seed N --gpu G   run ONE cell (resume-safe,
      1.5x circuit-breaker wired)
  sweep --scale S --phase {arm_off,post_pin} --gpus N --gpu-offset K --slot I
      run this slot's cells sequentially (Track C's own --gpus/--gpu-offset/
      static-partition convention, sec 13.9's own cited precedent -- no new
      orchestration pattern). phase=post_pin REFUSES per-cell
      (assert_blind_not_broken_frozenbias) unless this scale's bands pin
      already validates.
  pin --scale S    writes BANDS_PINNED-FrozenBias-<SCALE>.json from this
      scale's 6 COMPLETE arm_off cells (3 seeds x 2 corpora) -- val loss +
      the shared-pass comparator's arm1prime/arm1double/kraw span_frac.
      REFUSES (loud, no partial pin) unless all 6 are complete.
  await-armoff-and-pin --scale S    barrier: polls until all 6 arm_off cells
      are terminal, then calls pin() once (idempotent: skips if the pin file
      already exists and validates).
  comparator --checkpoint PATH --corpus C --out PATH   the shared-pass
      retrofit measurement, standalone (used by `pin`, also directly
      invocable).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PRETRAIN = os.path.join(HERE, "lm_pretrain_rd.py")
DATA_DIR = "/data/deltanet_rd_data"

# results/fixscale/ is SHARED with the gate tier (fixscale_gate_tier.py's own OUT_ROOT) --
# this wave writes into NEW subdirectories (train/, pins/) so nothing collides with the
# gate tier's own pilots/ and calib/ subdirectories.
OUT_ROOT = os.path.join(HERE, "results", "fixscale")
TRAIN_ROOT = os.path.join(OUT_ROOT, "train")
PIN_ROOT = os.path.join(OUT_ROOT, "pins")
CKPT_ROOT = "/data/fixscale_ckpts/train"          # gate tier uses .../calib -- disjoint by design
GATE_TIER_CALIB_ROOT = os.path.join(OUT_ROOT, "calib")

LAMBDA = 0.58                       # FROZEN_BIAS_LAMBDA_PRIMARY, sec 5/sec 13.3 -- passed explicitly
CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")
SEEDS_PRIMARY = (0, 1, 2)           # sec 6.0's registered seed integers (frozen_bias_lm_sweep.py SEEDS)
SEED_PROBE = 0                      # arm_global_probe: n=1, seed 0 (matches the gate tier's own
                                     # arm_off seed-0 convention, so the probe's comparator can reuse
                                     # the SAME arm_off seed-0 checkpoint the primary bar also uses)

# sec 13.1's param-count labels; configs = lm_rd_rung_configs.py RUNGS[1]/RUNGS[2] verbatim.
# ref_per_step_s / per_cell_gpuh = sec 13.7's corrected REALIZED Track-C rates (the 6x error is
# NOT inherited) -- IDENTICAL constants to fixscale_gate_tier.py's own SCALES dict, so a pilot
# verdict computed by that script and this wave's own circuit breaker agree on what "reference
# rate" means.
SCALES = {
    "98m":  dict(d_model=768,  d_state=64,  n_layers=12, steps=67547,
                 ref_per_step_s=0.236, per_cell_gpuh=4.478, batch_size=32, internal_timeout_s=36000),
    "392m": dict(d_model=1536, d_state=128, n_layers=16, steps=20000,
                 ref_per_step_s=0.836, per_cell_gpuh=4.671, batch_size=32, internal_timeout_s=36000),
}

# arm (this wave's own name) -> lm_pretrain_rd.py's --frozen-bias-arm value. arm_global_probe uses
# the ALREADY-BUILT "global" mode (frozen_bias_global_vector: b_global = normalize(mean_i B[i])) --
# sec 13.3's Rev 1 construction, verbatim, no new model-side code.
ARM_TO_FROZEN_BIAS_ARM = {"arm_off": "off", "arm_per_token": "per_token", "arm_global_probe": "global"}

# sec 13.8: "1.5x measured/calibrated per-step rate, hard-abort per cell." Two independent signals
# feed the same abort action (defense in depth -- a hung process with no new step-lines can't be
# caught by the rate signal alone):
#   (a) per-step rate vs 1.5x SCALES[scale]['ref_per_step_s'], checked at ~checkpoint cadence
#       (parses the log tail for the last logged step, mirrors fixscale_gate_tier.py's own
#       _rate_watch regex);
#   (b) an absolute wall-clock ceiling of 1.5x SCALES[scale]['per_cell_gpuh'] hours (the task's own
#       "1.5x the sec 13 priced per-cell rate" framing) as a backstop.
BUDGET_ABORT_FACTOR = 1.5
WATCH_INTERVAL_S = 120
STEP_LINE = re.compile(r"^\s*step\s+(\d+)\s+loss")

TIER_PROBE = "exploratory-probe — NOT a confirmatory bar, n=1"   # sec 13.15 finding 3 (BINDING):
# stamped DIRECTLY, literal string -- mech_schema.wrap_exploratory() hardcodes a DIFFERENT tier
# string (mech_schema.py:36, "exploratory-mechanism-wave -- NOT a confirmatory bar") and takes no
# custom parameter, so it must NOT be (mis)used here.


# ---------------------------------------------------------------------------
# Cell identity / paths.
# ---------------------------------------------------------------------------

def cell_name(arm: str, scale: str, corpus: str, seed: int) -> str:
    return f"fixscale_train_{arm}_{scale}_{corpus}_s{seed}"


def gate_tier_calib_out_path(scale: str, corpus: str) -> str:
    """The gate tier's OWN arm_off seed-0 calibration output path
    (fixscale_gate_tier.py's `do_calib`) -- resume-safe skip checks this FIRST for arm_off/seed=0
    cells before considering this wave's own train/ output path."""
    name = f"fixscale_calib_off_{scale}_{corpus}_s0"
    return os.path.join(GATE_TIER_CALIB_ROOT, f"{name}.json")


def _assert_gate_tier_config_identity(gate_path: str, scale: str) -> None:
    """sec 13.17 l10: runtime defense-in-depth -- before silently reusing the gate tier's own
    arm_off/seed=0 result JSON as this wave's own cell output (out_path's resume-safe convention),
    assert the reused JSON's OWN recorded config actually matches what this wave expects
    (frozen_bias.arm=='off', steps_completed>=steps, and this scale's own d_model/d_state/n_layers).
    Constants are verified identical TODAY (fixscale_gate_tier.py's own SCALES dict, sec 13.16's
    build record) -- but a future independent edit to either script's SCALES dict could silently
    drift; this catches that drift at RUNTIME instead of trusting the constants forever. Raises
    AssertionError (loud REFUSAL, this codebase's own established convention -- see
    _collect_pin_inputs' PIN REFUSED / check_blind's BLIND CHECK REFUSED) rather than silently
    falling back to re-training, since a mismatch here means the two scripts' configs have
    genuinely diverged and a human must look, not that this cell is merely not-ready-yet."""
    cfg = SCALES[scale]
    with open(gate_path) as f:
        d = json.load(f)
    arm = d.get("frozen_bias", {}).get("arm")
    assert arm == "off", (
        f"gate-tier reuse REFUSED: {gate_path} has frozen_bias.arm={arm!r}, expected 'off' "
        f"(sec 13.17 l10 config-identity check)")
    steps_completed = d.get("steps_completed", 0)
    assert steps_completed >= cfg["steps"], (
        f"gate-tier reuse REFUSED: {gate_path} has steps_completed={steps_completed}, expected "
        f">= {cfg['steps']} (sec 13.17 l10 config-identity check)")
    for field in ("d_model", "d_state", "n_layers"):
        assert d.get(field) == cfg[field], (
            f"gate-tier reuse REFUSED: {gate_path} has {field}={d.get(field)!r}, expected "
            f"{cfg[field]!r} (sec 13.17 l10 config-identity check)")


def out_path(arm: str, scale: str, corpus: str, seed: int) -> str:
    if arm == "arm_off" and seed == 0:
        gate_path = gate_tier_calib_out_path(scale, corpus)
        if _out_json_state(gate_path) == "complete":
            _assert_gate_tier_config_identity(gate_path, scale)
            return gate_path
    name = cell_name(arm, scale, corpus, seed)
    return os.path.join(TRAIN_ROOT, f"{name}.json")


def ckpt_dir_for(arm: str, scale: str, corpus: str, seed: int) -> str:
    if arm == "arm_off" and seed == 0 and _out_json_state(gate_tier_calib_out_path(scale, corpus)) == "complete":
        # gate-tier-owned checkpoint dir (fixscale_gate_tier.py's own convention); this wave never
        # writes into it, only reads the result JSON's own fields.
        return os.path.join("/data/fixscale_ckpts/calib", f"fixscale_calib_off_{scale}_{corpus}_s0")
    return os.path.join(CKPT_ROOT, cell_name(arm, scale, corpus, seed))


def bands_pin_path(scale: str) -> str:
    return os.path.join(PIN_ROOT, f"BANDS_PINNED-FrozenBias-{scale.upper()}.json")


def aborted_budget_marker_path(op: str) -> str:
    return op[:-5] + ".ABORTED_BUDGET.json" if op.endswith(".json") else op + ".ABORTED_BUDGET.json"


def refused_marker_path(op: str) -> str:
    return op[:-5] + ".REFUSED" if op.endswith(".json") else op + ".REFUSED"


# ---------------------------------------------------------------------------
# Manifest.
# ---------------------------------------------------------------------------

def build_manifest() -> list[dict]:
    """sec 13.7's Rev1 total: 28 training cells = (3 seeds x 2 corpora x 2 scales) arm_off
    + (3 seeds x 2 corpora x 2 scales) arm_per_token + (1 seed x 2 corpora x 2 scales)
    arm_global_probe = 12 + 12 + 4 = 28."""
    cells = []
    for scale in SCALES:
        for corpus in CORPORA:
            for seed in SEEDS_PRIMARY:
                cells.append(_cell("arm_off", scale, corpus, seed))
            for seed in SEEDS_PRIMARY:
                cells.append(_cell("arm_per_token", scale, corpus, seed))
            cells.append(_cell("arm_global_probe", scale, corpus, SEED_PROBE))
    return cells


def _cell(arm: str, scale: str, corpus: str, seed: int) -> dict:
    cfg = SCALES[scale]
    return {
        "cell_id": cell_name(arm, scale, corpus, seed),
        "arm": arm, "frozen_bias_arm": ARM_TO_FROZEN_BIAS_ARM[arm],
        "scale": scale, "corpus": corpus, "seed": seed, "lambda": LAMBDA,
        "steps": cfg["steps"], "out_path": out_path(arm, scale, corpus, seed),
        "ckpt_dir": ckpt_dir_for(arm, scale, corpus, seed),
        "phase": "arm_off" if arm == "arm_off" else "post_pin",
    }


def cells_for(scale: str | None = None, phase: str | None = None, arm: str | None = None) -> list[dict]:
    cells = build_manifest()
    if scale is not None:
        cells = [c for c in cells if c["scale"] == scale]
    if phase is not None:
        cells = [c for c in cells if c["phase"] == phase]
    if arm is not None:
        cells = [c for c in cells if c["arm"] == arm]
    return cells


# ---------------------------------------------------------------------------
# Resume-safe state.
# ---------------------------------------------------------------------------

def _out_json_state(path: str) -> str:
    if not os.path.exists(path):
        return "absent"
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return "absent"          # corrupt/partial write = re-run
    if d.get("complete") is True:
        return "complete"
    if d.get("timed_out") is True:
        return "timed_out"
    return "absent"


def cell_state(cell: dict) -> str:
    """'complete' | 'timed_out' | 'aborted_budget' | 'refused' | 'absent' -- the first three (plus
    'refused') are TERMINAL (never silently re-run); only 'absent' is re-run. Order matters:
    ABORTED_BUDGET/REFUSED markers are checked even if out_path itself is absent (a budget-aborted
    or blind-refused cell never produced a valid out_path)."""
    if os.path.exists(aborted_budget_marker_path(cell["out_path"])):
        return "aborted_budget"
    if os.path.exists(refused_marker_path(cell["out_path"])):
        return "refused"
    return _out_json_state(cell["out_path"])


TERMINAL_STATES = {"complete", "timed_out", "aborted_budget", "refused"}


# ---------------------------------------------------------------------------
# Cell command + run-with-circuit-breaker.
# ---------------------------------------------------------------------------

def base_cmd(cell: dict) -> list[str]:
    """Mirrors run_lm_rd_trackc_sweep.py's build_cmd_cell / fixscale_gate_tier.py's base_cmd flag
    set exactly (batch/seq/ckpt-every/lr/warmup defaults untouched) plus the frozen-bias arm flags
    (lm_pretrain_rd.py's EXISTING CLI, nothing new)."""
    cfg = SCALES[cell["scale"]]
    return [
        PY, PRETRAIN,
        "--corpus", cell["corpus"], "--data-dir", DATA_DIR,
        "--d-model", str(cfg["d_model"]), "--d-state", str(cfg["d_state"]),
        "--n-layers", str(cfg["n_layers"]), "--seq-len", "512",
        "--batch-size", str(cfg["batch_size"]), "--steps", str(cell["steps"]),
        "--ckpt-every", "1000", "--seed", str(cell["seed"]),
        "--internal-timeout", str(cfg["internal_timeout_s"]),
        "--frozen-bias-arm", cell["frozen_bias_arm"],
        "--frozen-bias-lambda", str(cell["lambda"]),
        "--ckpt-dir", cell["ckpt_dir"],
        "--out", cell["out_path"],
    ]


def _last_logged_step(log_path: str) -> int | None:
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 65536))
            tail = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        m = STEP_LINE.match(line)
        if m:
            return int(m.group(1))
    return None


class BudgetBreach(Exception):
    pass


def _budget_watchdog(proc: subprocess.Popen, log_path: str, t_launch: float, cell: dict,
                      stop_evt: threading.Event, watch_csv: str | None,
                      breach_evt: threading.Event) -> None:
    """sec 13.8's 1.5x circuit breaker, WIRED (not a manual-log-only watcher like the gate tier's
    own pilot-phase _rate_watch -- this one actually kills the process on breach). Two signals:
    (a) per-step rate (elapsed_wall / steps_so_far) vs 1.5x ref_per_step_s, re-checked every
        WATCH_INTERVAL_S against the CUMULATIVE average rate since launch (matches sec 13.8's own
        text: "wall_s_so_far / steps_so_far... every checkpoint");
    (b) absolute wall-clock ceiling 1.5x per_cell_gpuh hours, independent of whether any step-line
        has been logged at all (catches a hang before the first checkpoint)."""
    cfg = SCALES[cell["scale"]]
    ref_rate = cfg["ref_per_step_s"]
    ceiling_s = BUDGET_ABORT_FACTOR * cfg["per_cell_gpuh"] * 3600.0
    write_header = watch_csv is not None and not os.path.exists(watch_csv)
    csv_f = open(watch_csv, "a") if watch_csv else None
    try:
        if csv_f and write_header:
            csv_f.write("unix_time,step,cum_rate_s_per_step,flag\n")
            csv_f.flush()
        while not stop_evt.wait(WATCH_INTERVAL_S):
            now = time.time()
            elapsed = now - t_launch
            step = _last_logged_step(log_path)
            flag = ""
            if elapsed > ceiling_s:
                flag = f"BREACH:wallclock>{BUDGET_ABORT_FACTOR}x_priced_cell({ceiling_s:.0f}s)"
            # sec 13.17 F2 fix: the CUMULATIVE rate signal (elapsed_wall / steps_so_far) is gated on
            # step >= 1000 -- matching sec 13.8's OWN checkpoint-cadence spec (ckpt_every=1000) --
            # rather than firing at step > 0. lm_pretrain_rd.py logs an unconditional 'step 1' line
            # ("or step == 1") before any real throughput is established, so a healthy cell whose
            # startup (imports/corpus-load/model-init/Triton-compile, tens of seconds) delays the
            # NEXT log line past the first watch tick has step=1 as its last logged line ->
            # cum_rate = elapsed/1 is dominated by one-time startup cost, not steady-state
            # per-step rate, and falsely exceeds 1.5x ref (repro: repro_watchdog_false_abort.py).
            # Chose step>=1000 over an interval-rate/first-N-ticks-grace alternative because it ties
            # the grace period to a REAL, already-registered constant (ckpt_every=1000) instead of a
            # NEW magic tick-count this wave would have to separately calibrate and justify. The
            # wall-clock ceiling above (signal (a)) is UNCHANGED -- it still catches a true hang at
            # any elapsed time, independent of step count.
            elif step is not None and step >= 1000:
                cum_rate = elapsed / step
                if cum_rate > BUDGET_ABORT_FACTOR * ref_rate:
                    flag = f"BREACH:rate>{BUDGET_ABORT_FACTOR}x_ref({ref_rate}s/step,cum={cum_rate:.4f})"
            if csv_f:
                csv_f.write(f"{now:.0f},{step if step is not None else ''},"
                             f"{(elapsed / step) if step else ''},{flag}\n")
                csv_f.flush()
            if flag:
                print(f"BUDGET BREACH {cell['cell_id']}: {flag} -- terminating cell", flush=True)
                breach_evt.set()
                try:
                    proc.terminate()
                    proc.wait(timeout=30)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                return
    finally:
        if csv_f:
            csv_f.close()


GPU_BUSY_THRESHOLD_MB = 2048   # sec 13.17 M4: "above ~2GB used" -- refuse to co-schedule


def gpu_occupancy_ok(gpu: int, max_used_mb: int = GPU_BUSY_THRESHOLD_MB, force: bool = False,
                      _run_fn=subprocess.run) -> tuple[bool, str]:
    """sec 13.17 M4: refuse to launch a 23-39GB training cell onto a GPU that's already busy
    (>~2GB used, e.g. co-scheduled with live work) unless `force=True` (--force-gpu-busy). Handles
    `nvidia-smi` absence/failure gracefully -- FAIL-OPEN (permits launch), never fail-closed, so a
    CPU-only dev box (this machine) or a transient query hiccup never blocks a smoke run or a real
    launch; the guard is a co-scheduling safety net, not a hard CUDA-presence requirement.
    `_run_fn` is injectable (defaults to `subprocess.run`) so this is unit-testable without a real
    `nvidia-smi` binary or a real GPU."""
    if force:
        return True, "forced (--force-gpu-busy)"
    try:
        out = _run_fn(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
            capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return True, "nvidia-smi not found -- skipping GPU-occupancy guard (CPU/dev box)"
    except Exception as e:
        return True, f"nvidia-smi check failed ({e}) -- skipping GPU-occupancy guard (fail-open)"
    if out.returncode != 0:
        return True, f"nvidia-smi rc={out.returncode} -- skipping GPU-occupancy guard (fail-open)"
    try:
        used_mb = int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return True, "nvidia-smi output unparseable -- skipping GPU-occupancy guard (fail-open)"
    if used_mb > max_used_mb:
        return False, (f"GPU {gpu} already has {used_mb} MiB used (> {max_used_mb} MiB threshold) -- "
                        f"refusing to co-schedule (sec 13.17 M4); pass --force-gpu-busy to override")
    return True, f"GPU {gpu}: {used_mb} MiB used (<= {max_used_mb} MiB threshold) -- clear to launch"


def run_cell(cell: dict, gpu: int, dry_run: bool = False, force_gpu_busy: bool = False) -> str:
    """Resume-safe: returns the terminal state without launching anything if `cell_state` is
    already terminal. Otherwise launches lm_pretrain_rd.py with the 1.5x circuit breaker wired;
    on breach, writes an ABORTED_BUDGET marker (this wave's own per-cell "result", since
    lm_pretrain_rd.py itself has no knowledge of an externally-imposed budget) and returns
    'aborted_budget' -- the caller (sweep/supervisor) continues to the next cell, never retries
    an aborted cell automatically (mirrors the gate tier's own timed_out-is-terminal discipline)."""
    state = cell_state(cell)
    if state in TERMINAL_STATES:
        print(f"cell {cell['cell_id']}: {state} -- resume-safe skip", flush=True)
        return state
    if dry_run:
        print("DRY-RUN:", " ".join(base_cmd(cell)), flush=True)
        return "absent"

    ok, msg = gpu_occupancy_ok(gpu, force=force_gpu_busy)
    print(f"GPU occupancy check gpu={gpu}: {msg}", flush=True)
    if not ok:
        print(f"cell {cell['cell_id']}: REFUSED (GPU busy, retryable) -- {msg}", flush=True)
        return "absent"   # transient, NOT terminal -- the supervisor retries once the GPU frees up

    os.makedirs(TRAIN_ROOT, exist_ok=True)
    os.makedirs(cell["ckpt_dir"], exist_ok=True)
    log_path = os.path.join(TRAIN_ROOT, f"{cell['cell_id']}.log")
    watch_csv = os.path.join(TRAIN_ROOT, f"{cell['cell_id']}_rate_watch.csv")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = base_cmd(cell)
    print(f"LAUNCH gpu={gpu}: {' '.join(cmd)}", flush=True)
    t_launch = time.time()
    with open(log_path, "a") as logf:
        logf.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} UTC gpu={gpu} =====\n"
                    f"{' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        stop_evt = threading.Event()
        breach_evt = threading.Event()
        watcher = threading.Thread(target=_budget_watchdog,
                                    args=(proc, log_path, t_launch, cell, stop_evt, watch_csv, breach_evt),
                                    daemon=True)
        watcher.start()
        rc = proc.wait()
        stop_evt.set()
        watcher.join(timeout=5)

    if breach_evt.is_set():
        marker = aborted_budget_marker_path(cell["out_path"])
        with open(marker, "w") as f:
            json.dump({"cell_id": cell["cell_id"], "reason": "1.5x circuit breaker (sec 13.8)",
                       "wall_s_at_abort": time.time() - t_launch,
                       "aborted_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
        print(f"cell {cell['cell_id']}: ABORTED_BUDGET (rc={rc}) -> {marker}", flush=True)
        return "aborted_budget"

    final = _out_json_state(cell["out_path"])
    if final == "absent":
        print(f"cell {cell['cell_id']}: no valid output (rc={rc}) -- transient, supervisor retries", flush=True)
        return "absent"
    print(f"cell {cell['cell_id']}: {final} (rc={rc})", flush=True)
    return final


# ---------------------------------------------------------------------------
# Wave -1 pre/post-blend bit-identity gate (sec 13.10 gate 1 / sec 13.17 M3). MISSING entirely at
# d_state=128 in the audited build: lm_pretrain_rd.py's OWN smoke() item [17] performs this exact
# check (frozen_bias_arm='off', blend machinery present, vs no frozen_bias_* kwargs at all, blend
# machinery absent -- must be torch.equal) but hardcodes d_state=64 and takes no parameter; this
# wave introduces d_state=128 (the 392m scale) and does not own lm_pretrain_rd.py, so this wave
# carries its OWN parameterized copy of the SAME pattern rather than editing that file. Deferred
# torch/lm_pretrain_rd import (module-import-time stays torch-free, this file's own convention).
# ---------------------------------------------------------------------------

def check_off_path_bit_identity(d_state: int, device: str, d_model: int = 64, n_layers: int = 1,
                                 vocab: int = 500, seq_len: int = 128, batch: int = 4,
                                 seed: int = 31) -> None:
    """Raises AssertionError on divergence; returns None (no exception) on a bit-identical pass.
    Runs equally under the CPU stub (proves the PYTHON construction path is identical regardless of
    device/kernel backend -- smoke_fixscale.py's own CPU-stub leg calls this with device='cpu') and
    on real CUDA/fla kernels (the on-box pre-launch gate this module's own `wave-minus1-check`
    subcommand runs with device='cuda' -- real-kernel coverage a CPU-stub run cannot provide, per
    this codebase's own CPU-stub-tests-logic-only hard rule)."""
    import torch

    import h2h_fla_stub_rd   # matches run_shared_comparator_measurement's own convention: a no-op
    h2h_fla_stub_rd.ensure_fla_stub()   # when real fla is importable, so `wave-minus1-check` works
    from lm_pretrain_rd import DeltaNetLM   # standalone (no pre-loaded stub needed) on EITHER box

    torch.manual_seed(seed)
    m_a = DeltaNetLM(vocab, d_model=d_model, d_state=d_state, n_layers=n_layers, conv_size=4).to(device)
    torch.manual_seed(seed)
    m_b = DeltaNetLM(vocab, d_model=d_model, d_state=d_state, n_layers=n_layers, conv_size=4,
                      frozen_bias_arm="off").to(device)
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    with torch.no_grad():
        out_a = m_a(x)
        out_b = m_b(x)
    assert torch.equal(out_a, out_b), (
        f"frozen_bias_arm='off' (blend machinery present) diverges from the no-frozen_bias-kwargs "
        f"construction (blend machinery absent) at d_state={d_state} device={device!r} -- Wave -1 "
        f"pre/post-blend bit-identity is BROKEN (sec 13.10 gate 1 / sec 13.17 M3).")


# ---------------------------------------------------------------------------
# Disk-headroom pre-launch gate (sec 13.17 m7). Reuses run_lm_rd_trackc_sweep.py's OWN
# disk_space_check / projected_ckpt_bytes / find_ckpt_size_bytes / DISK_SAFETY_FACTOR helpers
# DIRECTLY (import, never reimplemented -- that file's own "one shared helper... not three copies"
# convention, restated here rather than re-derived by hand). CHOICE (m7's own "wire OR document"
# option): WIRED as a real subcommand (`disk-check`) because reuse-by-import made the wiring a few
# lines, not a reimplementation -- disproportionate would have meant re-deriving disk_space_check's
# live shutil.disk_usage logic from scratch, which this does not do. Its INVOCATION point is
# documented (fixscale_supervisor.sh's launch-scale usage header), not auto-run inside
# launch_scale/do_sweep, matching this same header's existing "GPU occupancy re-verified live before
# calling this... NOT hardcoded here" convention: headroom should be re-checked fresh, by a human,
# immediately before committing to a launch, not baked into an unattended retry loop.
# ---------------------------------------------------------------------------

CKPT_SIZE_BYTES_ESTIMATE = {
    # sec 13's own literal per-checkpoint estimate (this wave's d_model/d_state/n_layers configs) --
    # used ONLY as a fallback when no real checkpoint of this exact architecture exists yet on this
    # box (find_ckpt_size_bytes raises FileNotFoundError in that case -- e.g. before the gate tier's
    # first calibration checkpoint lands); once a real checkpoint exists, the LIVE measured size is
    # used instead, never this literal.
    "98m": int(0.39 * (1024 ** 3)),
    "392m": int(1.57 * (1024 ** 3)),
}


def projected_disk_bytes(scale: str) -> int:
    """Mirrors run_lm_rd_trackc_sweep.py's own projected_ckpt_bytes convention exactly (n_runs *
    n_ckpts * ckpt_size_bytes), scoped to cells NOT YET terminal for `scale` (a resumed/partial
    launch does not need headroom for cells that will resume-skip)."""
    from run_lm_rd_trackc_sweep import find_ckpt_size_bytes, projected_ckpt_bytes
    cfg = SCALES[scale]
    remaining = [c for c in cells_for(scale=scale) if cell_state(c) not in TERMINAL_STATES]
    try:
        ckpt_size = find_ckpt_size_bytes(CKPT_ROOT, cfg["d_model"], cfg["d_state"], cfg["n_layers"])
    except FileNotFoundError:
        ckpt_size = CKPT_SIZE_BYTES_ESTIMATE[scale]
    return projected_ckpt_bytes(len(remaining), cfg["steps"], 1000, ckpt_size)


def disk_headroom_check(scale: str) -> dict:
    """The pre-launch gate itself: live free-space check against ALL remaining cells' projected
    checkpoint volume for `scale`, at DISK_SAFETY_FACTOR=1.5 (run_lm_rd_trackc_sweep.py's own
    constant, imported -- never re-declared with a possibly-drifting value)."""
    from run_lm_rd_trackc_sweep import disk_space_check
    projected = projected_disk_bytes(scale)
    return disk_space_check(CKPT_ROOT, projected, f"fixscale {scale}")


# ---------------------------------------------------------------------------
# Blind-pin gate (sec 13.10 gate 5 / sec 13.15 finding-3-adjacent binding item 3): the launcher
# call site. Wired at TWO points: (1) `sweep`/`cell` REFUSE to launch a post_pin-phase cell unless
# this scale's bands pin already exists and is content-valid; (2) immediately before that cell's
# OWN subprocess launch, assert_blind_not_broken_frozenbias(doc, [this cell's own start timestamp])
# is called -- not "the pin file exists" alone, the ORDERING assertion itself, every single time,
# matching sec 13.12's "call site... immediately after arm_off pin-write, strictly before
# arm_per_token training START" requirement literally (checked per-cell, not once per sweep).
# ---------------------------------------------------------------------------

from bands_pinned_frozenbias import (  # noqa: E402  (after constants, matches this file's own layout)
    assert_blind_not_broken_frozenbias, validate_bands_pinned_frozenbias, write_bands_pinned_frozenbias)


def load_pin_doc(scale: str) -> dict | None:
    path = bands_pin_path(scale)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def check_blind(scale: str, started_at: float | None = None) -> None:
    """Raises AssertionError (via assert_blind_not_broken_frozenbias) if the pin is missing OR its
    pinned_at does not strictly precede `started_at` (default: time.time() at call time). This is
    the exact call site sec 13.12/sec 13.15 binding item 3 requires -- called once per post_pin
    cell, immediately before that cell's own subprocess launch."""
    doc = load_pin_doc(scale)
    assert doc is not None, (
        f"BLIND CHECK REFUSED: no bands pin at {bands_pin_path(scale)} for scale={scale!r} -- "
        f"arm_per_token/arm_global_probe cells must not launch before arm_off's pin is written "
        f"(sec 13.10 gate 5).")
    ts = started_at if started_at is not None else time.time()
    assert_blind_not_broken_frozenbias(doc, [ts])


# ---------------------------------------------------------------------------
# Shared-forward-pass comparator (sec 13.15 finding 1, BINDING). Deferred torch/fla imports so
# `import fixscale_wave` for manifest/resume/breaker logic never requires CUDA or a fla stub --
# only this function (and its CLI subcommand) does.
# ---------------------------------------------------------------------------

def run_shared_comparator_measurement(checkpoint_path: str, corpus: str, lam: float,
                                       frozen_bias_seed: int, data_dir: str, chunk_size: int,
                                       n_windows: int, batch_size: int, seq_len: int,
                                       device: str) -> dict:
    """ONE capture_raw_keys forward pass per (checkpoint, corpus); span_frac derived for THREE
    modes (per_token = arm_off' primary comparator, global = arm_global_probe's comparator, kraw =
    the sec 4.a-i co-primary) from that single capture. This is the sec 13.15 finding-1 fix:
    frozen_bias_retrofit_eval_rd.run_retrofit_measurement, called once per --mode, would redo
    capture_raw_keys per mode for byte-identical keys -- 3x the forward passes for nothing. Every
    tensor helper below (capture_raw_keys, build_frozen_bias_table, frozen_bias_global_vector,
    apply_frozen_bias_blend, measure_population) is the EXACT function object
    frozen_bias_retrofit_eval_rd.py's own arm1prime/arm1double/kraw modes call -- this function
    reimplements the LOOP (single capture, three derivations) but never reimplements the
    arithmetic (sec 8.0b's code-path-equality discipline preserved)."""
    import torch

    import h2h_fla_stub_rd
    h2h_fla_stub_rd.ensure_fla_stub()
    from frozen_bias_retrofit_eval_rd import capture_raw_keys, load_checkpoint
    from lm_pretrain_rd import (EOT_TOKEN_ID, build_frozen_bias_table, corpus_fixed_seed,
                                 frozen_bias_global_vector, get_batch, load_corpus)

    model, ckpt = load_checkpoint(checkpoint_path, device)
    num_heads = model.blocks[0].mixer.num_heads
    d_state = model.blocks[0].mixer.d_state
    vocab_size = model.vocab_size
    table = build_frozen_bias_table(vocab_size, d_state, seed=frozen_bias_seed).to(device)
    global_vec = frozen_bias_global_vector(table)

    _, val_tokens, meta, _, val_offs = load_corpus(data_dir, corpus, device)
    gen = torch.Generator(device=device).manual_seed(corpus_fixed_seed(corpus) + 95_000)
    n_batches = max(1, -(-n_windows // batch_size))
    batches = [get_batch(val_tokens, batch_size, seq_len, gen)[0] for _ in range(n_batches)]

    keys_by_layer, token_ids_cat = capture_raw_keys(model, batches, device)   # THE one forward pass
    content_mask = (token_ids_cat != EOT_TOKEN_ID)

    per_mode = derive_three_modes_from_capture(
        keys_by_layer, token_ids_cat, content_mask, num_heads, chunk_size, table, global_vec, lam)

    del model, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 13.15 finding 1 (shared-forward-pass comparator)",
        "checkpoint": checkpoint_path, "corpus": corpus, "lam": lam, "frozen_bias_seed": frozen_bias_seed,
        "forward_passes": 1, "modes_derived_from_that_pass": ["per_token", "global", "kraw"],
        "per_mode_per_layer": per_mode,
    }


def derive_three_modes_from_capture(keys_by_layer: dict, token_ids_cat, content_mask, num_heads: int,
                                     chunk_size: int, table, global_vec, lam: float) -> dict:
    """The pure-tensor half of the shared-forward-pass fix (sec 13.15 finding 1), split out of
    `run_shared_comparator_measurement` so it is unit-testable on CPU with a SYNTHETIC capture
    (no data_dir / no real corpus files needed -- `smoke_fixscale.py` exercises this directly with
    a toy CPU-stub model's own real `capture_raw_keys` output, proving the "one capture, three
    modes" property against the ACTUAL reused functions, not a re-implementation). Takes the
    ALREADY-CAPTURED `keys_by_layer`/`token_ids_cat` (one real forward pass, done by the caller)
    and derives per_token/global/kraw span_frac summaries from it -- no model, no forward pass,
    no data loading in this function at all."""
    from frozen_bias_retrofit_eval_rd import measure_population
    from lm_pretrain_rd import apply_frozen_bias_blend

    per_mode = {"per_token": {}, "global": {}, "kraw": {}}
    for layer_idx, k_raw in keys_by_layer.items():
        per_mode["kraw"][layer_idx] = measure_population(k_raw, content_mask, num_heads, chunk_size)
        for mode, arm_mode in (("per_token", "per_token"), ("global", "global")):
            k_measured = apply_frozen_bias_blend(
                k_raw, token_ids_cat, arm_mode,
                table if arm_mode == "per_token" else None,
                global_vec if arm_mode == "global" else None, lam)
            per_mode[mode][layer_idx] = measure_population(k_measured, content_mask, num_heads, chunk_size)
    return per_mode


def _span_frac_for_corpus(measurement: dict, mode: str) -> float:
    """Pools span_frac across layers (mean of per-layer span_frac, None-safe) -- mirrors
    frozen_bias_retrofit_eval_rd.py's own per-layer-then-pooled convention."""
    vals = [v["span_frac"] for v in measurement["per_mode_per_layer"][mode].values()
            if v.get("span_frac") is not None]
    assert vals, f"no scored layers for mode={mode!r} -- cannot derive a span_frac"
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Pin writer.
# ---------------------------------------------------------------------------

def write_wave_pin(scale: str, lam: float = LAMBDA, frozen_bias_seed: int | None = None,
                    data_dir: str = DATA_DIR, chunk_size: int = 64, n_windows: int = 32,
                    batch_size: int = 16, seq_len: int = 512, device: str | None = None) -> dict:
    """sec 13.10 gate 5: computed and written from arm_off's OWN fresh per-seed data, BEFORE
    arm_per_token/arm_global_probe are inspected. REFUSES (raises) unless all 6 arm_off cells for
    this scale (3 seeds x 2 corpora) are 'complete' -- no partial pin, matching key_anchoring's own
    'ONLY after every reference arm's result JSON validates complete' writer requirement."""
    inputs = _collect_pin_inputs(scale, lam, frozen_bias_seed, data_dir, chunk_size, n_windows,
                                  batch_size, seq_len, device)
    os.makedirs(PIN_ROOT, exist_ok=True)
    doc = write_bands_pinned_frozenbias(
        bands_pin_path(scale), inputs["val_loss"], inputs["per_token_sf"], inputs["global_sf"],
        inputs["kraw_sf"], inputs["result_paths"], inputs["result_paths"], inputs["result_paths"],
        inputs["result_paths"])
    print(f"pin written: {bands_pin_path(scale)} (pinned_at={doc['pinned_at']})", flush=True)
    return doc


def _collect_pin_inputs(scale: str, lam: float, frozen_bias_seed: int | None, data_dir: str,
                         chunk_size: int, n_windows: int, batch_size: int, seq_len: int,
                         device: str | None) -> dict:
    """Shared by `write_wave_pin` (first write) and `verify_pin` (tamper-evidence re-check,
    sec 7.3's 'mechanical, not a paper exercise' discipline) -- re-extracting via the SAME
    function guarantees the writer and the verifier can never silently drift apart (the e633862-
    audit F2 fix's own load-bearing property, applied here rather than re-derived by hand)."""
    import torch

    from key_anchoring import ANCHOR_INIT_SEED
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    frozen_bias_seed = frozen_bias_seed if frozen_bias_seed is not None else ANCHOR_INIT_SEED

    val_loss_per_corpus: dict[str, list[float]] = {c: [] for c in CORPORA}
    result_paths_per_corpus: dict[str, list[str]] = {c: [] for c in CORPORA}
    per_token_sf: dict[str, list[float]] = {c: [] for c in CORPORA}
    global_sf: dict[str, list[float]] = {c: [] for c in CORPORA}
    kraw_sf: dict[str, list[float]] = {c: [] for c in CORPORA}

    for corpus in CORPORA:
        for seed in SEEDS_PRIMARY:
            cell = _cell("arm_off", scale, corpus, seed)
            state = cell_state(cell)
            assert state == "complete", (
                f"PIN REFUSED: arm_off cell {cell['cell_id']} is {state!r}, not 'complete' -- "
                f"sec 13.10 gate 5 requires all 6 arm_off cells complete before pinning.")
            with open(cell["out_path"]) as f:
                d = json.load(f)
            val_loss_per_corpus[corpus].append(d["checkpoints"][-1]["val_loss"][corpus])
            result_paths_per_corpus[corpus].append(cell["out_path"])
            meas = run_shared_comparator_measurement(
                _checkpoint_path_for(cell, d), corpus, lam, frozen_bias_seed, data_dir,
                chunk_size, n_windows, batch_size, seq_len, device)
            per_token_sf[corpus].append(_span_frac_for_corpus(meas, "per_token"))
            global_sf[corpus].append(_span_frac_for_corpus(meas, "global"))
            kraw_sf[corpus].append(_span_frac_for_corpus(meas, "kraw"))

    return {"val_loss": val_loss_per_corpus, "result_paths": result_paths_per_corpus,
            "per_token_sf": per_token_sf, "global_sf": global_sf, "kraw_sf": kraw_sf}


def verify_pin(scale: str, lam: float = LAMBDA, frozen_bias_seed: int | None = None,
               data_dir: str = DATA_DIR, chunk_size: int = 64, n_windows: int = 32,
               batch_size: int = 16, seq_len: int = 512, device: str | None = None) -> dict | None:
    """Tamper-evidence re-check (sec 13.10 gate 7's 'Harvest with verify-vs-raws' convention,
    box-only -- re-runs the comparator over all 6 arm_off checkpoints and re-derives every band,
    requiring value-identity with what's stored via `validate_bands_pinned_frozenbias`). Returns
    the validated doc on success, None on ANY tamper/mismatch (missing file, hash mismatch, or a
    re-derived value that disagrees with what was pinned)."""
    inputs = _collect_pin_inputs(scale, lam, frozen_bias_seed, data_dir, chunk_size, n_windows,
                                  batch_size, seq_len, device)
    return validate_bands_pinned_frozenbias(
        bands_pin_path(scale), inputs["val_loss"], inputs["per_token_sf"], inputs["global_sf"],
        inputs["kraw_sf"])


def _checkpoint_path_for(cell: dict, result_doc: dict) -> str:
    """The comparator needs a MODEL checkpoint (.pt), not the result JSON. lm_pretrain_rd.py's
    train() writes `{ckpt_dir}/{run_name}_step{step}.pt` per checkpoint (line ~2113) -- the FINAL
    one (last entry in `checkpoints`) is what Arm 1's own eval-only retrofit reads, matching
    frozen_bias_retrofit_eval_rd.py's own convention of taking a fully-trained checkpoint."""
    final_step = result_doc["checkpoints"][-1]["step"]
    return os.path.join(cell["ckpt_dir"], f"{result_doc['run_name']}_step{final_step}.pt")


def stamp_probe_tier(payload: dict) -> dict:
    """sec 13.15 finding 3 (BINDING): stamps TIER_PROBE directly, literal string, `"tier"` inserted
    first (mirrors mech_schema.wrap_exploratory's own insert-first convention for readability, but
    NEVER calls that function -- it hardcodes a different tier string and takes no parameter,
    per finding 3's own text). Returns a NEW dict; does not mutate `payload`."""
    out = {"tier": TIER_PROBE}
    out.update(payload)
    return out


def build_probe_report(scale: str, corpus: str) -> dict:
    """Assembles the arm_global_probe cell's own descriptive-tier artifact: its training result
    (val loss, peak VRAM, completion state) alongside the comparator baseline it is read against
    (arm_off' retrofitted with the SAME b_global blend at eval time -- this pin's own `global_sf`
    entry, sec 13.3's 'mirrors arm_off1's own eval-only-retrofit convention... rung-1's own Arm 1``
    control'). Stamped with TIER_PROBE -- explicitly NOT a WIN/PARTIAL/NULL input (sec 13.6 Rev 1
    note: the probe never gates the primary bar)."""
    probe_cell = _cell("arm_global_probe", scale, corpus, SEED_PROBE)
    probe_state = cell_state(probe_cell)
    report = {
        "wave": "fix-at-scale sec 13, arm_global_probe (sec 13.3 Rev 1 addition)",
        "non_gating": "does NOT gate the WIN/PARTIAL/NULL bar (sec 13.6 Rev 1 note) -- "
                       "descriptive/exploratory only, n=1, no CI.",
        "scale": scale, "corpus": corpus, "seed": SEED_PROBE, "lambda": LAMBDA,
        "probe_cell_id": probe_cell["cell_id"], "probe_state": probe_state,
        "probe_result_path": probe_cell["out_path"] if probe_state == "complete" else None,
    }
    if probe_state == "complete":
        with open(probe_cell["out_path"]) as f:
            d = json.load(f)
        report["probe_val_loss"] = d["checkpoints"][-1]["val_loss"].get(corpus)
        report["probe_peak_memory_allocated_bytes"] = d.get("peak_memory_allocated_bytes")
        report["probe_peak_memory_reserved_bytes"] = d.get("peak_memory_reserved_bytes")
        report["probe_run_name"] = d.get("run_name")
    pin_doc = load_pin_doc(scale)
    if pin_doc is not None:
        report["comparator_arm_off_prime_global_blend_span_frac_band"] = \
            pin_doc.get("arm1double_span_frac_bands", {}).get(corpus)
        report["comparator_source"] = "this scale's BANDS_PINNED-FrozenBias pin, arm1double " \
                                       "(global-blend eval-only retrofit of arm_off's own checkpoints)"
    else:
        report["comparator_arm_off_prime_global_blend_span_frac_band"] = None
        report["comparator_source"] = "pin not yet written"
    return stamp_probe_tier(report)


def await_armoff_and_pin(scale: str, poll_s: int = 60, stop_path: str | None = None) -> str:
    """Barrier: polls until all 6 arm_off cells for `scale` are terminal, THEN writes the pin (or
    skips if a valid pin already exists -- idempotent). Returns 'pinned' | 'pin_exists' |
    'stopped'."""
    existing = load_pin_doc(scale)
    if existing is not None:
        print(f"pin already exists at {bands_pin_path(scale)} -- skip (idempotent)", flush=True)
        return "pin_exists"
    while True:
        if stop_path and os.path.exists(stop_path):
            return "stopped"
        states = {cell_state(c) for c in cells_for(scale=scale, arm="arm_off")}
        if states == {"complete"}:
            write_wave_pin(scale)
            return "pinned"
        non_complete = [c["cell_id"] for c in cells_for(scale=scale, arm="arm_off") if cell_state(c) != "complete"]
        if any(s in ("timed_out", "aborted_budget", "refused") for s in states):
            print(f"WARNING: arm_off has a terminal-but-not-complete cell among {non_complete} -- "
                  f"pin cannot be written honestly (needs 3/3 seeds per corpus); waiting for a human "
                  f"decision (relaunch that seed or re-scope), not auto-pinning a partial set.", flush=True)
        else:
            print(f"awaiting arm_off completion for {scale}: pending {non_complete}", flush=True)
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# Static-partition sweep (Track C's own --gpus/--gpu-offset convention, sec 13.9's cited precedent).
# ---------------------------------------------------------------------------

STOP_FILE_DEFAULT = os.path.join(HERE, "STOP_fixscale_wave")   # matches fixscale_supervisor.sh's own
                                                                 # STOP_FILE constant (same directory)


def _post_pin_blind_gate(cell: dict, scale: str, dry_run: bool = False) -> bool:
    """sec 13.17 F1/m5: the ONE shared post_pin blind-check gate, called from BOTH `sweep`
    (do_sweep) and the single-cell `cell` subcommand (m5 -- a human calling `cell` directly on a
    post_pin cell must not bypass the same gate a sweep slot enforces). Returns True if the cell may
    proceed to run_cell, False if the caller should treat this as a transient/retryable failure (rc
    1) -- NOT terminal.

    F1 FIX: on a check_blind failure this function writes NO marker at all (the pre-fix bug: do_sweep
    wrote a terminal `.REFUSED` marker on the ORDINARY pre-pin race -- the supervisor launches
    post_pin slots before the pin exists BY DESIGN -- permanently skipping 7/16 cells; see
    repro_refused_terminal.py). `.REFUSED` is reserved for a genuine ordering violation (a cell whose
    OWN training subprocess actually started before the pin, which this call site -- checked
    immediately before launch, every single cell, every attempt -- prevents from ever happening by
    construction); no code path in this build writes `.REFUSED` automatically now, but a stale one
    left behind by a PRE-FIX run (or a human's manual intervention) is cleared here, the moment the
    blind check next passes, so a cell is never permanently stuck on history once the pin genuinely
    lands (sec 13.17 F1's second acceptable fix option)."""
    try:
        check_blind(scale, started_at=time.time())
    except AssertionError as e:
        print(f"cell {cell['cell_id']}: blind check not yet satisfied ({e}) -- transient, "
              f"retryable, NO terminal marker written (sec 13.17 F1 fix)", flush=True)
        return False
    if not dry_run:
        stale = refused_marker_path(cell["out_path"])
        if os.path.exists(stale):
            os.remove(stale)
            print(f"cell {cell['cell_id']}: cleared stale .REFUSED marker (blind check now passes)",
                  flush=True)
    return True


def do_sweep(scale: str, phase: str, gpus: int, gpu_offset: int, slot: int, dry_run: bool,
             stop_path: str | None = STOP_FILE_DEFAULT, force_gpu_busy: bool = False) -> int:
    assert 0 <= slot < gpus, f"--slot {slot} must be in [0,{gpus})"
    phase_cells = cells_for(scale=scale, phase=phase)
    my_cells = [c for i, c in enumerate(phase_cells) if i % gpus == slot]
    gpu = gpu_offset + slot
    print(f"sweep scale={scale} phase={phase} slot={slot}/{gpus} gpu={gpu}: "
          f"{len(my_cells)}/{len(phase_cells)} cells assigned", flush=True)
    for cell in my_cells:
        # sec 13.17 m6: check the STOP file BETWEEN cells, not just at the shell retry boundary --
        # fixscale_supervisor.sh's run_until_terminal only re-checks STOP_FILE when this process
        # EXITS non-zero; without this in-loop check, a multi-cell slot mid-way through a LONG
        # training cell would only honor STOP after that cell (hours) finishes, not promptly. A
        # non-zero, non-"transient-retry" return here (rc 2) still routes through
        # run_until_terminal's own already-wired `[ -f "$STOP_FILE" ]` check, so no shell-side
        # change was needed to make this effective end to end.
        if stop_path and os.path.exists(stop_path):
            print(f"sweep {scale}/{phase} slot={slot}: STOP file {stop_path} present -- halting "
                  f"before cell {cell['cell_id']} (sec 13.17 m6)", flush=True)
            return 2
        if phase == "post_pin":
            if not _post_pin_blind_gate(cell, scale, dry_run=dry_run):
                return 1   # retryable: the pin may not be written yet
        state = run_cell(cell, gpu, dry_run=dry_run, force_gpu_busy=force_gpu_busy)
        if state == "absent" and not dry_run:
            return 1   # transient failure -- supervisor retries the whole slot loop
    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest")
    p.add_argument("--scale", choices=sorted(SCALES), default=None)
    p.add_argument("--phase", choices=("arm_off", "post_pin"), default=None)
    p.add_argument("--with-state", action="store_true")

    c = sub.add_parser("cell")
    c.add_argument("--scale", choices=sorted(SCALES), required=True)
    c.add_argument("--corpus", choices=CORPORA, required=True)
    c.add_argument("--arm", choices=sorted(ARM_TO_FROZEN_BIAS_ARM), required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--gpu", type=int, required=True)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--force-gpu-busy", action="store_true",
                    help="sec 13.17 M4: bypass the GPU-occupancy guard (intentional co-scheduling)")

    s = sub.add_parser("sweep")
    s.add_argument("--scale", choices=sorted(SCALES), required=True)
    s.add_argument("--phase", choices=("arm_off", "post_pin"), required=True)
    s.add_argument("--gpus", type=int, required=True)
    s.add_argument("--gpu-offset", type=int, default=0)
    s.add_argument("--slot", type=int, required=True)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--stop-path", type=str, default=STOP_FILE_DEFAULT,
                    help="sec 13.17 m6: checked BETWEEN cells, not just at the shell retry "
                         "boundary; defaults to fixscale_supervisor.sh's own STOP_FILE")
    s.add_argument("--force-gpu-busy", action="store_true",
                    help="sec 13.17 M4: bypass the GPU-occupancy guard (intentional co-scheduling)")

    w1 = sub.add_parser("wave-minus1-check",
                         help="sec 13.17 M3: Wave -1 pre/post-blend bit-identity gate, "
                              "parameterized to an arbitrary d_state (392m's new d_state=128 shape "
                              "is not covered by lm_pretrain_rd.py's own hardcoded-d_state=64 smoke "
                              "item [17]) -- run on-box with a real device before launching 392m.")
    w1.add_argument("--d-state", type=int, default=128)
    w1.add_argument("--device", default=None)

    dk = sub.add_parser("disk-check",
                         help="sec 13.17 m7: live disk-headroom pre-launch gate for this scale's "
                              "remaining (not-yet-terminal) cells -- run before launch-scale.")
    dk.add_argument("--scale", choices=sorted(SCALES), required=True)

    pn = sub.add_parser("pin")
    pn.add_argument("--scale", choices=sorted(SCALES), required=True)

    vp = sub.add_parser("verify-pin")
    vp.add_argument("--scale", choices=sorted(SCALES), required=True)

    aw = sub.add_parser("await-armoff-and-pin")
    aw.add_argument("--scale", choices=sorted(SCALES), required=True)
    aw.add_argument("--poll-s", type=int, default=60)
    aw.add_argument("--stop-path", type=str, default=None)

    cb = sub.add_parser("check-blind")
    cb.add_argument("--scale", choices=sorted(SCALES), required=True)

    pr = sub.add_parser("probe-report")
    pr.add_argument("--scale", choices=sorted(SCALES), required=True)
    pr.add_argument("--corpus", choices=CORPORA, required=True)
    pr.add_argument("--out", required=True)

    cp = sub.add_parser("comparator")
    cp.add_argument("--checkpoint", required=True)
    cp.add_argument("--corpus", choices=CORPORA, required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--lam", type=float, default=LAMBDA)
    cp.add_argument("--frozen-bias-seed", type=int, default=None)
    cp.add_argument("--data-dir", default=DATA_DIR)

    args = ap.parse_args()

    if args.cmd == "manifest":
        cells = cells_for(scale=args.scale, phase=args.phase)
        if args.with_state:
            # sec 13.17 l11 fix: the pre-fix loop rebound the LOCAL `c` to a copy and mutated that
            # copy, never writing it back into `cells` -- --with-state was a silent no-op. Rebuild
            # the list instead of mutating in a loop.
            cells = [dict(c, state=cell_state(c)) for c in cells]
        print(json.dumps(cells, indent=2))
        return 0
    if args.cmd == "cell":
        cell = _cell(args.arm, args.scale, args.corpus, args.seed)
        if cell["phase"] == "post_pin":   # sec 13.17 m5: same gate the `sweep` slot loop enforces
            if not _post_pin_blind_gate(cell, args.scale, dry_run=args.dry_run):
                return 1
        state = run_cell(cell, args.gpu, dry_run=args.dry_run, force_gpu_busy=args.force_gpu_busy)
        return 0 if state in TERMINAL_STATES or args.dry_run else 1
    if args.cmd == "sweep":
        return do_sweep(args.scale, args.phase, args.gpus, args.gpu_offset, args.slot, args.dry_run,
                         stop_path=args.stop_path, force_gpu_busy=args.force_gpu_busy)
    if args.cmd == "pin":
        write_wave_pin(args.scale)
        return 0
    if args.cmd == "verify-pin":
        doc = verify_pin(args.scale)
        print(f"verify-pin {args.scale}: {'VALID (tamper-evidence check passed)' if doc else 'FAILED (missing/tampered/mismatched)'}", flush=True)
        return 0 if doc is not None else 1
    if args.cmd == "await-armoff-and-pin":
        result = await_armoff_and_pin(args.scale, args.poll_s, args.stop_path)
        return 0 if result in ("pinned", "pin_exists") else 3
    if args.cmd == "check-blind":
        check_blind(args.scale)
        print(f"blind OK for {args.scale}", flush=True)
        return 0
    if args.cmd == "wave-minus1-check":
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        check_off_path_bit_identity(args.d_state, device)
        print(f"wave-minus1-check d_state={args.d_state} device={device}: BIT-IDENTICAL "
              f"(sec 13.10 gate 1 / sec 13.17 M3 satisfied)", flush=True)
        return 0
    if args.cmd == "disk-check":
        report = disk_headroom_check(args.scale)
        print(json.dumps(report, indent=2))
        print(f"disk-check {args.scale}: {'OK' if report['ok'] else 'INSUFFICIENT HEADROOM'}", flush=True)
        return 0 if report["ok"] else 1
    if args.cmd == "probe-report":
        report = build_probe_report(args.scale, args.corpus)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out} (tier={report['tier']!r})", flush=True)
        return 0
    if args.cmd == "comparator":
        from key_anchoring import ANCHOR_INIT_SEED
        fb_seed = args.frozen_bias_seed if args.frozen_bias_seed is not None else ANCHOR_INIT_SEED
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = run_shared_comparator_measurement(
            args.checkpoint, args.corpus, args.lam, fb_seed, args.data_dir,
            chunk_size=64, n_windows=32, batch_size=16, seq_len=512, device=device)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
