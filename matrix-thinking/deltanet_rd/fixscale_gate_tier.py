"""fixscale_gate_tier.py -- FIX-AT-SCALE wave (FROZEN_BIAS_LM_DESIGN.md sec 13),
GATE TIER ONLY: sec 13.10 item 3 (timing pilot per scale, + the sec 13.13 Rev-1
item-2 VRAM-logging mandate) and sec 13.10 item 4 (calibration-first per scale,
task-directed arm_off control cells anchored to Track C's archived val-loss
bands). Both gates are REQUIRED before the full sweep and are INDEPENDENT of
Rev 1's additive delta (the global-probe arm) -- launching them needs NO new
training code: every cell below is a plain lm_pretrain_rd.py CLI invocation
(committed copy, box md5 a16fff66bb121b60ecd906ae6f9ef018, byte-verified vs
git HEAD before launch).

Explicitly NOT this script (the BUILD stage, sec 13.12's build list): the full
sweep launcher (manifest/resume across 24+ cells), blind-pin enforcement via
assert_blind_not_broken, the 1.5x per-cell circuit breaker (sec 13.8), the
arm_global_probe cells, and the frozen_bias_lm_sweep.py scale-parametrization
(that file is 14M-hardcoded: RUNG1_CFG line 77).

Subcommands:
  pilot --scale {98m,392m} --gpu N
     Two-point timing pilots (the house two-point convention,
     run_lm_rd_trackc_sweep.py --wave calibration's own method: rate =
     (wall_long - wall_short) / (steps_long - steps_short), cancelling
     corpus-load/compile startup overhead a single short run would fold in).
     For each arm in (off, per_token@lambda=0.58): one 200-step + one
     1000-step cell, openr1-mix-ext, seed 0, NO checkpoints (pilots are
     timing-measurement throwaways -- their outputs are EXCLUDED from every
     span_frac/val-loss analysis by construction; the per_token pilots
     precede the blind pin but are never read as measurement cells).
     Writes results/fixscale/pilots/PILOT_<SCALE>_VERDICT.json: per-arm
     steady-state s/step, peak VRAM GB (lm_pretrain_rd.py's own
     peak_memory_*_bytes fields, from the 1000-step runs), blend-overhead
     ratio, and PASS/FAIL vs 1.5x the archived Track-C realized rate
     (sec 13.7's corrected table: 98M 0.236 s/step, 392M 0.836 s/step).
  calib --scale S --corpus {openr1-mix-ext,wikitext-mix-ext} --gpu N
     One full-length arm_off cell (98M: 67,547 steps, Track-C-matched;
     392M: 20,000 steps, the sec 13.7 reduced budget). REFUSES (terminal
     .REFUSED marker, exit 0 so a supervisor loop does not spin) unless the
     scale's pilot verdict exists and reads PASS. Per-cell ckpt dir --
     lm_pretrain_rd.py's run_name (line 3312) carries NO frozen-bias arm
     tag, so cells of different arms at one config/seed would collide in a
     shared dir. Watchdog thread appends (unix_time, last_logged_step) to
     <cell>_rate_watch.csv every 120 s (~500 steps at 98M's reference rate)
     -- the MANUAL stand-in for the not-yet-built 1.5x breaker; it logs a
     loud warning line on a >1.5x interval rate but NEVER aborts (the
     harvest re-checks these rows; abort authority is the build stage's).

Resume-safety: a cell whose --out JSON parses and reports complete=true OR
timed_out=true is terminal (skipped; timed_out is flagged loudly, never
silently re-run). Anything else (missing/corrupt/partial) is re-run.
Archived val-loss anchors for the harvest (EXPERIMENT_LOG.md:3980-3984,
4046-4047; sec 13.10 item 4): 98M openr1 1.290 / wikitext 3.189; 392M openr1
1.135 / wikitext 2.847 -- 392M cells here run 20k of 91,552 steps, so their
val loss should read HIGHER than the archived full-length band, not at it.
"""
from __future__ import annotations

import argparse
import csv
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
OUT_ROOT = os.path.join(HERE, "results", "fixscale")
CKPT_ROOT = "/data/fixscale_ckpts/calib"
LAMBDA = 0.58                       # FROZEN_BIAS_LAMBDA_PRIMARY, sec 5 / sec 13.3 -- passed explicitly
PILOT_CORPUS = "openr1-mix-ext"
PILOT_STEPS_SHORT, PILOT_STEPS_LONG = 200, 1000   # task-directed ~500-1000; <= sec 13.10's 2,000 cap
RATE_GATE_FACTOR = 1.5              # sec 13.10 item 3: within the 1.5x circuit-breaker band
WATCH_INTERVAL_S = 120

# sec 13.1's param-count labels; configs = lm_rd_rung_configs.py RUNGS[1]/RUNGS[2] verbatim;
# ref_per_step_s = sec 13.7's corrected REALIZED Track-C rates (the 6x error is NOT inherited).
SCALES = {
    "98m":  dict(d_model=768,  d_state=64,  n_layers=12, steps_full=67547,
                 ref_per_step_s=0.236, batch_size=32, calib_timeout_s=36000),
    "392m": dict(d_model=1536, d_state=128, n_layers=16, steps_full=20000,
                 ref_per_step_s=0.836, batch_size=32, calib_timeout_s=36000),
}
CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")
ARCHIVED_VAL_BANDS = {  # harvest anchors only -- never enforced at launch time
    "98m": {"openr1-mix-ext": 1.290, "wikitext-mix-ext": 3.189},
    "392m": {"openr1-mix-ext": 1.135, "wikitext-mix-ext": 2.847},
}

STEP_LINE = re.compile(r"^\s*step\s+(\d+)\s+loss")


def out_json_state(path: str) -> str:
    """'complete' | 'timed_out' | 'absent' -- terminal states vs re-run."""
    if not os.path.exists(path):
        return "absent"
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return "absent"          # corrupt = re-run
    if d.get("complete") is True:
        return "complete"
    if d.get("timed_out") is True:
        return "timed_out"
    return "absent"              # partial/invalid = re-run


def run_cell(cmd: list[str], gpu: int, log_path: str, watch_csv: str | None = None,
             ref_rate: float | None = None) -> int:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"LAUNCH gpu={gpu}: {' '.join(cmd)}", flush=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} UTC{time.strftime('%z')} "
                   f"gpu={gpu} =====\n{' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        stop_evt = threading.Event()
        watcher = None
        if watch_csv is not None:
            watcher = threading.Thread(
                target=_rate_watch, args=(log_path, watch_csv, stop_evt, ref_rate), daemon=True)
            watcher.start()
        rc = proc.wait()
        stop_evt.set()
        if watcher is not None:
            watcher.join(timeout=5)
    print(f"EXIT rc={rc}: {os.path.basename(log_path)}", flush=True)
    return rc


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


def _rate_watch(log_path: str, watch_csv: str, stop_evt: threading.Event,
                ref_rate: float | None) -> None:
    """Manual watchdog (sec 13.8's 1.5x breaker is NOT built -- this only LOGS).
    Appends (unix_time, step); on a >1.5x interval rate vs ref, writes a loud
    WARN row. The harvest re-checks every row; nothing here aborts anything."""
    prev_t, prev_step = None, None
    write_header = not os.path.exists(watch_csv)
    with open(watch_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["unix_time", "step", "interval_s_per_step", "flag"])
            f.flush()
        while not stop_evt.wait(WATCH_INTERVAL_S):
            step = _last_logged_step(log_path)
            now = time.time()
            if step is None:
                continue
            rate, flag = "", ""
            if prev_step is not None and step > prev_step:
                r = (now - prev_t) / (step - prev_step)
                rate = f"{r:.4f}"
                if ref_rate is not None and r > RATE_GATE_FACTOR * ref_rate:
                    flag = f"WARN>1.5x-ref({ref_rate})"
            w.writerow([f"{now:.0f}", step, rate, flag])
            f.flush()
            prev_t, prev_step = now, step


def base_cmd(scale: str, corpus: str, arm: str, steps: int, out_path: str,
             timeout_s: int, ckpt_dir: str | None) -> list[str]:
    """Mirrors run_lm_rd_trackc_sweep.py build_cmd_cell (line 575) exactly --
    same flag set, same defaults left untouched (lr/warmup/eval-*) so the
    calibration cells are comparable to Track C's archived bands -- plus the
    frozen-bias arm flags (lm_pretrain_rd.py's existing CLI, nothing new)."""
    cfg = SCALES[scale]
    cmd = [PY, PRETRAIN,
           "--corpus", corpus, "--data-dir", DATA_DIR,
           "--d-model", str(cfg["d_model"]), "--d-state", str(cfg["d_state"]),
           "--n-layers", str(cfg["n_layers"]), "--seq-len", "512",
           "--batch-size", str(cfg["batch_size"]), "--steps", str(steps),
           "--ckpt-every", "1000", "--seed", "0",
           "--internal-timeout", str(timeout_s),
           "--frozen-bias-arm", arm,
           "--frozen-bias-lambda", str(LAMBDA),
           "--out", out_path]
    if ckpt_dir is not None:
        cmd += ["--ckpt-dir", ckpt_dir]
    return cmd


def do_pilot(scale: str, gpu: int, dry_run: bool) -> int:
    cfg = SCALES[scale]
    pdir = os.path.join(OUT_ROOT, "pilots")
    os.makedirs(pdir, exist_ok=True)
    verdict_path = os.path.join(pdir, f"PILOT_{scale.upper()}_VERDICT.json")
    if os.path.exists(verdict_path):
        with open(verdict_path) as f:
            v = json.load(f)
        print(f"pilot {scale}: verdict already exists ({v['verdict']}) -- resume-safe skip", flush=True)
        return 0

    runs = {}
    for arm in ("off", "per_token"):
        for steps in (PILOT_STEPS_SHORT, PILOT_STEPS_LONG):
            name = f"fixscale_pilot_{scale}_{arm}_{steps}"
            out_path = os.path.join(pdir, f"{name}.json")
            state = out_json_state(out_path)
            if state == "complete":
                print(f"  {name}: complete -- skip", flush=True)
            elif state == "timed_out":
                print(f"  {name}: TIMED_OUT (terminal, flagged) -- pilot cannot conclude; "
                      f"inspect before relaunch", flush=True)
                return 2
            else:
                cmd = base_cmd(scale, PILOT_CORPUS, arm, steps, out_path,
                               timeout_s=3600, ckpt_dir=None)
                if dry_run:
                    print("  DRY-RUN:", " ".join(cmd), flush=True)
                    continue
                rc = run_cell(cmd, gpu, os.path.join(pdir, f"{name}.log"))
                if rc != 0 or out_json_state(out_path) != "complete":
                    print(f"  {name}: FAILED (rc={rc}) -- supervisor will retry", flush=True)
                    return 1
            runs[(arm, steps)] = out_path
    if dry_run:
        return 0

    arms = {}
    for arm in ("off", "per_token"):
        with open(runs[(arm, PILOT_STEPS_SHORT)]) as f:
            short = json.load(f)
        with open(runs[(arm, PILOT_STEPS_LONG)]) as f:
            long_ = json.load(f)
        per_step = (long_["wall_s"] - short["wall_s"]) / (PILOT_STEPS_LONG - PILOT_STEPS_SHORT)
        arms[arm] = {
            "per_step_s_two_point": per_step,
            "wall_s_short": short["wall_s"], "wall_s_long": long_["wall_s"],
            "peak_vram_alloc_gb": long_["peak_memory_allocated_bytes"] / 1024**3,
            "peak_vram_reserved_gb": long_["peak_memory_reserved_bytes"] / 1024**3,
            "within_1p5x_ref": per_step <= RATE_GATE_FACTOR * cfg["ref_per_step_s"],
        }
    verdict = "PASS" if all(a["within_1p5x_ref"] for a in arms.values()) else "FAIL"
    payload = {
        "wave": "fix-at-scale sec13 GATE TIER", "gate": "sec 13.10 item 3 (timing pilot) "
        "+ sec 13.13 Rev-1 item 2 (VRAM logging)",
        "tier": "timing-measurement only -- outputs EXCLUDED from every span_frac/val-loss analysis",
        "scale": scale, "config": {k: cfg[k] for k in ("d_model", "d_state", "n_layers", "batch_size")},
        "corpus": PILOT_CORPUS, "seed": 0, "lambda": LAMBDA,
        "method": "two-point (wall_1000 - wall_200)/800, run_lm_rd_trackc_sweep.py --wave "
                  "calibration's own convention -- startup overhead cancels",
        "ref_per_step_s_trackc": cfg["ref_per_step_s"],
        "gate_factor": RATE_GATE_FACTOR,
        "arms": arms,
        "blend_overhead_ratio_pt_over_off":
            arms["per_token"]["per_step_s_two_point"] / arms["off"]["per_step_s_two_point"],
        "verdict": verdict,
        "written_unix": time.time(), "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    with open(verdict_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"pilot {scale}: VERDICT {verdict} -- "
          f"off {arms['off']['per_step_s_two_point']:.4f} s/step, "
          f"per_token {arms['per_token']['per_step_s_two_point']:.4f} s/step "
          f"(ref {cfg['ref_per_step_s']}, gate {RATE_GATE_FACTOR}x), "
          f"peak VRAM alloc off/pt "
          f"{arms['off']['peak_vram_alloc_gb']:.1f}/{arms['per_token']['peak_vram_alloc_gb']:.1f} GB "
          f"-> {verdict_path}", flush=True)
    return 0


def do_calib(scale: str, corpus: str, gpu: int, dry_run: bool) -> int:
    cfg = SCALES[scale]
    assert corpus in CORPORA, corpus
    cdir = os.path.join(OUT_ROOT, "calib")
    os.makedirs(cdir, exist_ok=True)
    name = f"fixscale_calib_off_{scale}_{corpus}_s0"
    out_path = os.path.join(cdir, f"{name}.json")
    refused_path = os.path.join(cdir, f"{name}.REFUSED")

    state = out_json_state(out_path)
    if state == "complete":
        print(f"calib {name}: complete -- resume-safe skip", flush=True)
        return 0
    if state == "timed_out":
        print(f"calib {name}: TIMED_OUT (terminal, flagged for harvest) -- not silently re-run", flush=True)
        return 0

    verdict_path = os.path.join(OUT_ROOT, "pilots", f"PILOT_{scale.upper()}_VERDICT.json")
    if not os.path.exists(verdict_path):
        print(f"calib {name}: REFUSED -- no pilot verdict at {verdict_path} "
              f"(sec 13.10 item 3 precedes item 4)", flush=True)
        return 1   # retryable: the pilot may still be running
    with open(verdict_path) as f:
        v = json.load(f)
    if v["verdict"] != "PASS":
        with open(refused_path, "w") as f:
            f.write(f"pilot verdict {v['verdict']} at {verdict_path} -- calibration refused "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC\n")
        print(f"calib {name}: TERMINAL REFUSAL (pilot verdict {v['verdict']}) -- "
              f"wrote {refused_path}; exit 0 so the supervisor does not spin", flush=True)
        return 0

    ckpt_dir = os.path.join(CKPT_ROOT, name)   # per-cell dir: run_name carries no arm tag
    os.makedirs(ckpt_dir, exist_ok=True)
    cmd = base_cmd(scale, corpus, "off", cfg["steps_full"], out_path,
                   timeout_s=cfg["calib_timeout_s"], ckpt_dir=ckpt_dir)
    if dry_run:
        print("DRY-RUN:", " ".join(cmd), flush=True)
        return 0
    log_path = os.path.join(cdir, f"{name}.log")
    watch_csv = os.path.join(cdir, f"{name}_rate_watch.csv")
    rc = run_cell(cmd, gpu, log_path, watch_csv=watch_csv, ref_rate=cfg["ref_per_step_s"])
    final = out_json_state(out_path)
    if final == "absent":
        print(f"calib {name}: no valid output (rc={rc}) -- supervisor will retry", flush=True)
        return 1
    with open(out_path) as f:
        d = json.load(f)
    last_val = (d.get("checkpoints") or [{}])[-1].get("val_loss", {})
    band = ARCHIVED_VAL_BANDS[scale][corpus]
    print(f"calib {name}: {final} (rc={rc}); final val_loss={last_val} vs archived full-length "
          f"anchor {band} ({'20k/91,552 steps -- expect HIGHER than anchor' if scale == '392m' else 'step-matched'})"
          f"; harvest re-checks {os.path.basename(watch_csv)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--scale", choices=sorted(SCALES), required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("calib")
    c.add_argument("--scale", choices=sorted(SCALES), required=True)
    c.add_argument("--corpus", choices=CORPORA, required=True)
    c.add_argument("--gpu", type=int, required=True)
    c.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "pilot":
        return do_pilot(args.scale, args.gpu, args.dry_run)
    return do_calib(args.scale, args.corpus, args.gpu, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
