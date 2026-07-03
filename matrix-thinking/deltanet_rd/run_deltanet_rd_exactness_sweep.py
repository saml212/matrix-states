"""run_deltanet_rd_exactness_sweep.py -- bounded, human-gated wave
orchestrator for DELTANET_RD_EXACTNESS_DESIGN.md (Rev 3)'s Waves -1/0/1.

A DELIBERATE PARALLEL SCRIPT, not an extension of run_deltanet_rd_sweep.py
(the audited RD-1 causal-necessity pipeline) -- CLAUDE.md's standing rule
("do not modify audited Wave-1 pipeline for existing configs") plus a
genuine naming collision: the ORIGINAL sweep's wave labels ("-1","0","A",
"Bprobe","B") mean specific, already-run things (RD-1's own timing
calibration, unconstrained screening grid, force-rank probe/grid); THIS
design has its OWN, DIFFERENT "-1"/"0"/"1" wave taxonomy (sec 6). Reusing
the same labels in the same script would make every results/ directory
ambiguous about which design's "wave -1" it holds. `run_smoke` is
IMPORTED (read-only reuse, not a modification) from the original sweep --
it already calls `run_deltanet_rd.py --smoke`, which now includes this
design's own new smoke sections, so importing it costs nothing and keeps
one smoke gate for the whole harness.

Waves built here (sec 6's manifest table, EXCLUDING Waves 2/3/4/NCE-
ablation/alignment-contingency/Wave-F -- out of THIS build's scope, see
the build report):
  -1  9 short GPU timing/instability probes across the new arms/
      instruments (~500-1,000 steps each) + arm (iv)'s raw-table sanity
      smoke (CPU-only, folded into the wave's own smoke gate via
      embed_arms._self_test(), already exercised by run_deltanet_rd.py
      --smoke's own [6]/[7]/[8] sections -- not re-launched here as a
      separate GPU spec). Arm (iv)'s CLOSED-LOOP (alpha,rho) calibration
      probes are NOT included in this wave's manifest -- they are
      inherently SEQUENTIAL (each iteration depends on the last), so
      calibrate_arm_iv.py is a separate driver, pointed to from --wave -1's
      own printed next-step message.
   0  Free, no GPU: analyze_exactness_w0.py against the archived
      experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA}
      dumps. `--wave 0` here just SHELLS OUT to that script (no manifest,
      no polling loop) and writes its --out-json into the wave's out_dir.
   1  32 runs: arms (i)/(ii) x K in {8,16,32} x 3 seeds (18); arm (iv) x
      K in {16,32} x 3 seeds (6, needs --gram-rho from a completed
      calibrate_arm_iv.py run -- REQUIRED, no silent default); arm
      (i-strong) K=32 x 3 seeds (3); arm (iii-beta) K=16 x2 + K=32 x3,
      unchanged learned baseline + the new beta/succ/tgt_slot dump (5).

Usage (GPU list is an example -- check nvidia-smi first, per house rule):
  python run_deltanet_rd_exactness_sweep.py --dry-run
  python run_deltanet_rd_exactness_sweep.py --wave -1 --out-dir results/deltanet_rd_exactness --gpus 1 --gpu-offset 6
  python run_deltanet_rd_exactness_sweep.py --wave 0  --out-dir results/deltanet_rd_exactness \\
      --w0-dirs results/deltanet_rd_w0b/wave0 results/deltanet_rd_w0b/waveA
      # (canonical BOX paths -- FIX-3; on the local Mac use experiment-runs/
      # 2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA}. NEVER
      # results/deltanet_rd/wave0: stale pre-rerun data, same filenames.)
  python run_deltanet_rd_exactness_sweep.py --wave 1 --out-dir results/deltanet_rd_exactness \\
      --gpus 1 --gpu-offset 6 --gram-rho-16 0.xx --gram-rho-32 0.yy
      # (rho values gated against calibrate_arm_iv.py's CALIBRATION_SUMMARY.json,
      # FIX-2; omit both flags to run without arm (iv) -- 26 runs, loud warning.)
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
ANALYZE_W0 = os.path.join(HERE, "analyze_exactness_w0.py")

sys.path.insert(0, HERE)
from run_deltanet_rd_sweep import run_smoke, write_progress   # read-only reuse, audited file untouched

# Continuity anchor: waveA's own measured 2,091s/run at 20,000 steps
# (DELTANET_REALDATA_DESIGN.md sec 16.9) -- the same per-step placeholder
# run_deltanet_rd_sweep.py uses, WIDENED by 1.3x for the new/unmeasured
# code paths this design adds (frozen-row lookups, ZCA, F-nce subsampling
# -- each cheap in principle, but genuinely unmeasured until Wave -1's own
# probes report back; sec 6's own "wider, explicitly-uncertain band until
# Wave -1 measures them directly" discipline).
_PER_STEP_S_ANCHOR = 0.15 * 1.3
TIER_STEPS_WAVE1 = 20000          # continuity with waveA's own 2x-tier-steps convention
NEG1_PROBE_STEPS = 750             # "~500-1,000 steps" (sec 6's Wave -1 row)


def default_timeout(K: int, steps: int, margin: float = 1.7) -> int:
    per_step = _PER_STEP_S_ANCHOR * max(1.0, K / 32)
    n_ckpts = max(1, steps // 2000 + 1)
    return int((per_step * steps + n_ckpts * 20.0) * margin)


def _spec(wave, K, seed, steps, arm, embed_source="learned", gram_alpha=None, gram_rho=None,
          strong_pin=False, lambda_orth=0.0, use_zca=False, fnce_m=None, force_rank_k=None,
          geo3_active=False, geo3_n_iter=12, geo3_resid_tol=1e-2):
    """Variant-carrying run spec (sec 5's naming requirement) -- the `arm`
    tag alone would NOT be resume-safe if two arms shared identical other
    fields, so the name also encodes the fields that actually vary the
    trained artifact (embed_source, gram_rho, strong_pin, lambda_orth,
    use_zca, fnce_m, geo3_active/geo3_n_iter); is_done() below re-derives
    the SAME identity from the result JSON's own exactness_config, never
    trusting the filename alone."""
    bits = [f"w{wave}_rdx", f"K{K}", f"arm{arm}", f"s{seed}"]
    if embed_source == "frozen_gram_matched" and gram_rho is not None:
        bits.append(f"rho{gram_rho:.3f}".replace(".", "p"))
    if strong_pin:
        bits.append("sp")
    if lambda_orth:
        bits.append(f"lorth{lambda_orth}".replace(".", "p"))
    if use_zca:
        bits.append("zca")
    if fnce_m is not None:
        bits.append(f"fnce{fnce_m}")
    if geo3_active:
        bits.append(f"geo3n{geo3_n_iter}")
    name = "_".join(bits)
    return {"wave": str(wave), "K": K, "seed": seed, "steps": steps, "force_rank_k": force_rank_k,
            "arm": arm, "embed_source": embed_source, "gram_alpha": gram_alpha, "gram_rho": gram_rho,
            "strong_pin": strong_pin, "lambda_orth": lambda_orth, "use_zca": use_zca, "fnce_m": fnce_m,
            "geo3_active": geo3_active, "geo3_n_iter": geo3_n_iter if geo3_active else None,
            "geo3_resid_tol": geo3_resid_tol if geo3_active else None,
            "name": name}


# ---------------------------------------------------------------------------
# Wave -1: 9 short GPU timing/instability probes across every new arm/
# instrument this build adds (sec 6's Wave -1 row, scoped to what THIS
# build touches -- Wave 3/4's own probes are out of scope, see the module
# docstring).
# ---------------------------------------------------------------------------

def wave_neg1_manifest():
    runs = [
        _spec(-1, 16, 0, NEG1_PROBE_STEPS, "i", embed_source="frozen_orthonormal"),
        _spec(-1, 32, 0, NEG1_PROBE_STEPS, "i", embed_source="frozen_orthonormal"),
        _spec(-1, 16, 0, NEG1_PROBE_STEPS, "ii", embed_source="frozen_gpt2_span"),
        _spec(-1, 32, 0, NEG1_PROBE_STEPS, "ii", embed_source="frozen_gpt2_span"),
        _spec(-1, 16, 0, NEG1_PROBE_STEPS, "iv", embed_source="frozen_gram_matched",
              gram_alpha=1.0, gram_rho=0.3),   # production timing only -- NOT the calibration probes
        _spec(-1, 32, 0, NEG1_PROBE_STEPS, "i-strong", strong_pin=True),   # i-strong's only valid K
        _spec(-1, 16, 0, NEG1_PROBE_STEPS, "fnce", fnce_m=7),
        _spec(-1, 16, 0, NEG1_PROBE_STEPS, "geo1", lambda_orth=1.0),
        _spec(-1, 16, 0, NEG1_PROBE_STEPS, "geo2", use_zca=True),
    ]
    assert len(runs) == 9, f"Wave -1 manifest drifted from its documented 9 short probes: {len(runs)}"
    return runs


# ---------------------------------------------------------------------------
# Wave 1: the 32-run primary interpolation (sec 6's Wave 1 row).
# ---------------------------------------------------------------------------

def wave1_manifest(gram_rho_16: float | None, gram_rho_32: float | None):
    steps = TIER_STEPS_WAVE1
    runs = []
    for arm, embed_source in (("i", "frozen_orthonormal"), ("ii", "frozen_gpt2_span")):
        for K in (8, 16, 32):
            for s in range(3):
                runs.append(_spec(1, K, s, steps, arm, embed_source=embed_source))
    if gram_rho_16 is not None:
        for s in range(3):
            runs.append(_spec(1, 16, s, steps, "iv", embed_source="frozen_gram_matched",
                               gram_alpha=1.0, gram_rho=gram_rho_16))
    if gram_rho_32 is not None:
        for s in range(3):
            runs.append(_spec(1, 32, s, steps, "iv", embed_source="frozen_gram_matched",
                               gram_alpha=1.0, gram_rho=gram_rho_32))
    for s in range(3):
        runs.append(_spec(1, 32, s, steps, "i-strong", strong_pin=True))
    for s in range(2):
        runs.append(_spec(1, 16, s, steps, "iii-beta", embed_source="learned"))
    for s in range(3):
        runs.append(_spec(1, 32, s, steps, "iii-beta", embed_source="learned"))
    return runs


# ---------------------------------------------------------------------------
# F-geo-3 Wave -1/1 (DELTANET_RD_EXACTNESS_DESIGN.md sec 14.6/14.7/14.12).
# A DELIBERATE parallel manifest, not folded into wave_neg1_manifest/
# wave1_manifest above -- those two are this design's PRE-registered
# F-geo-1/F-geo-2 track (sec 6/5.5), already run to completion (section 15,
# "Wave F results"). F-geo-3 is a THIRD candidate proposed AFTER that track
# closed (sec 14.0), reusing BLOCK-2's reduced-scope-fallback template (sec
# 14.7: "one candidate, minimal cells, standards never relaxed") rather than
# widening the original manifest functions.
#
# sec 14.6's own gating diagnostic (the per-entity cross-episode
# orthogonalized-key drift measurement + geo3_simulator.launch_read) is a
# SEPARATE, sequential driver -- geo3_drift_diagnostic.py -- matching this
# file's own precedent for arm (iv)'s closed-loop calibration
# (calibrate_arm_iv.py, "inherently SEQUENTIAL... a separate driver", see
# this module's docstring). It is NOT launched by this script; its OUTPUT
# JSON is what gate_geo3_drift() below reads.
# ---------------------------------------------------------------------------

def geo3_wave1_manifest(include_k48: bool = False):
    """sec 14.7's cell table: K in {16,32} x 3 seeds x 20,000 steps,
    mandatory (6 runs); K=48 x 3 seeds, OPTIONAL Reserve-eligible stretch
    (+3 runs) -- "launches only after both mandatory cells complete without
    tripping sec 14.8 outcome D" (sec 14.12), a HUMAN assessment this
    script does not automate; --include-k48 is an explicit opt-in, not a
    default, for exactly that reason."""
    steps = TIER_STEPS_WAVE1
    runs = []
    for K in (16, 32):
        for s in range(3):
            runs.append(_spec("geo3", K, s, steps, "geo3", geo3_active=True,
                               geo3_n_iter=12, geo3_resid_tol=1e-2))
    if include_k48:
        for s in range(3):
            # K=48: n_iter=20 per audit — NS at n_iter=12 lands at resid 0.104 > tol
            # on realistic near-collinear probes; 20 converges to ~1e-6 (verified).
            runs.append(_spec("geo3", 48, s, steps, "geo3", geo3_active=True,
                               geo3_n_iter=20, geo3_resid_tol=1e-2))
    return runs


def gate_geo3_drift(drift_json_path: str, accept_override: bool) -> dict:
    """sec 14.6/14.12's Wave-1 launch gate: geo3's training cells launch
    ONLY IF geo3_drift_diagnostic.py's output JSON has launch_read.launch
    == True. --accept-gate-override is an explicit, LOUDLY-logged human
    decision (mirrors gate_gram_rho's FIX-2 discipline above) -- this
    script never silently overrides a failed or missing gate.

    Sanity-checks the JSON is a REAL measurement, not a stub/placeholder:
    requires per_k[16].after_probe.mean/p10 and launch_read.gate_cell to be
    present (the fields geo3_drift_diagnostic.py always writes)."""
    if accept_override:
        print("=" * 70 + "\nWARNING: --accept-gate-override -- the sec 14.6 drift launch-read "
              "gate is being BYPASSED by an explicit human decision. This is recorded here and "
              "in the launch log; it does NOT mean the gate passed.\n" + "=" * 70, flush=True)
        return {"gate_bypassed": True, "drift_json_path": drift_json_path}
    if not os.path.exists(drift_json_path):
        print(f"ERROR: --geo3-drift-json {drift_json_path!r} does not exist. Run "
              f"geo3_drift_diagnostic.py first (sec 14.6's gating diagnostic), or pass "
              f"--accept-gate-override for an explicit, documented human override.", file=sys.stderr)
        sys.exit(1)
    with open(drift_json_path) as f:
        d = json.load(f)
    try:
        gate_cell = d["launch_read"]["gate_cell"]
        gate_value = d["launch_read"]["predicted_gate_value"]
        launch = d["launch_read"]["launch"]
        k16_mean = d["per_k"]["16"]["after_probe"]["mean"]
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"ERROR: --geo3-drift-json {drift_json_path!r} is missing an expected field "
              f"({e!r}) -- this does not look like a real geo3_drift_diagnostic.py output. "
              f"Refusing to launch; pass --accept-gate-override for an explicit override.",
              file=sys.stderr)
        sys.exit(1)
    if not launch:
        print(f"ERROR: sec 14.6 GATE FAILED -- {gate_cell} predicted rec@0.9={gate_value:.4f} < 0.8 "
              f"(K=16 trained-checkpoint drift mean={k16_mean:.4f}). Per sec 14.6: 'Wave 1 does not "
              f"launch as-is: the stability question routes to a follow-on design (sec 14.8 outcome "
              f"F), never to on-the-fly fix iteration.' Pass --accept-gate-override for an explicit, "
              f"documented human override.", file=sys.stderr)
        sys.exit(1)
    print(f"sec 14.6 GATE PASSED: {gate_cell} predicted rec@0.9={gate_value:.4f} >= 0.8 "
          f"(K=16 trained-checkpoint drift mean={k16_mean:.4f})", flush=True)
    return {"gate_bypassed": False, "drift_json_path": drift_json_path,
            "gate_cell": gate_cell, "predicted_gate_value": gate_value}


MANIFEST_FNS = {"-1": wave_neg1_manifest}


# ---------------------------------------------------------------------------
# Resume / launch machinery (pattern cloned from run_deltanet_rd_sweep.py,
# NOT imported -- the wave-selection/manifest logic differs enough that a
# shared driver would need to branch on "which design" throughout; the
# smoke gate and progress-writer ARE imported above, since those are
# genuinely generic).
# ---------------------------------------------------------------------------

def out_path(out_dir, spec):
    return os.path.join(out_dir, f"{spec['name']}.json")


def is_done(out_dir, spec):
    p = out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return False
        required = ("K", "seed", "steps", "complete", "steps_completed", "exactness_config")
        if not all(k in d for k in required):
            return False
        if d.get("timed_out") or d.get("complete") is not True:
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if d.get("K") != spec["K"] or d.get("seed") != spec["seed"] or d.get("steps") != spec["steps"]:
            return False
        ec = d.get("exactness_config") or {}
        # identity check on every arm-relevant field this spec sets --
        # cross-checked against the RESULT's own recorded config (never
        # the filename alone), same discipline as the original sweep's
        # trunc_impl check.
        for field in ("embed_source", "gram_alpha", "gram_rho", "strong_pin", "lambda_orth",
                       "use_zca", "fnce_m"):
            if ec.get(field) != spec.get(field):
                return False
        # geo3 fields (sec 14) were added to exactness_config POST-HOC -- already-archived
        # pre-geo3 result JSONs (Wave F's 18/18 completed cells, sec 15) lack these keys
        # entirely, so ec.get(...) returns None there while a freshly-derived non-geo3 spec
        # carries False/None explicitly. Default the MISSING-key case to "off" on both sides
        # so old archives stay resume-safe matches for their own (geo3-less) specs, rather
        # than spuriously failing is_done() on a None-vs-False key-presence mismatch.
        if bool(ec.get("geo3_active", False)) != bool(spec.get("geo3_active", False)):
            return False
        if spec.get("geo3_active"):
            if ec.get("geo3_n_iter") != spec.get("geo3_n_iter") or \
               ec.get("geo3_resid_tol") != spec.get("geo3_resid_tol"):
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
    if spec["embed_source"] != "learned":
        cmd += ["--embed-source", spec["embed_source"]]
    if spec["embed_source"] == "frozen_gram_matched":
        cmd += ["--gram-alpha", str(spec["gram_alpha"]), "--gram-rho", str(spec["gram_rho"])]
    if spec["strong_pin"]:
        cmd += ["--strong-pin"]
    if spec["lambda_orth"]:
        cmd += ["--lambda-orth", str(spec["lambda_orth"])]
    if spec["use_zca"]:
        cmd += ["--use-zca"]
    if spec["fnce_m"] is not None:
        cmd += ["--fnce-m", str(spec["fnce_m"])]
    if spec.get("geo3_active"):
        cmd += ["--use-geo3", "--geo3-n-iter", str(spec["geo3_n_iter"]),
                "--geo3-resid-tol", str(spec["geo3_resid_tol"])]
    return cmd


def run_wave0(out_dir, w0_dirs):
    """Free, no GPU: shells out to analyze_exactness_w0.py (pure numpy) --
    no manifest, no polling loop, no smoke gate (nothing here touches
    CUDA/fla)."""
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "w0_report.json")
    cmd = [sys.executable, ANALYZE_W0, "--dirs", *w0_dirs, "--out-json", out_json]
    print(f"WAVE 0 (free, no GPU): {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=HERE)
    if rc == 0 and os.path.exists(out_json):
        with open(os.path.join(out_dir, "ALL_DONE"), "w") as f:
            f.write(f"wave 0 complete: {out_json}\n")
        print(f"WAVE 0 DONE -- report at {out_json}", flush=True)
    else:
        print(f"WAVE 0 FAILED (rc={rc})", file=sys.stderr)
        sys.exit(1)


def gate_gram_rho(gram_rho_16: float | None, gram_rho_32: float | None,
                    calib_summary_path: str, accept_unconverged: bool) -> tuple[float | None, float | None]:
    """FIX-2 (2026-07-03 audit, MAJOR): the Wave-1 arm-(iv) launch gate.

    - NEITHER --gram-rho-16/32 given: loud warning, arm (iv)'s 6 cells are
      EXCLUDED from the manifest (26 runs launch instead of 32) -- the
      behavior the pre-fix error message CLAIMED but did not implement
      (it sys.exit(1)'d instead). The exclusion is resume-safe: a later
      re-launch WITH the flags adds exactly the missing arm-(iv) cells
      (is_done() passes the already-complete 26).
    - A --gram-rho-K value is REFUSED unless calibrate_arm_iv.py's
      CALIBRATION_SUMMARY.json exists with converged:true for that K
      (and is not a simulation record) -- override with
      --accept-unconverged-rho for an explicit, recorded human decision.
      A value that differs from the recorded final_rho by >1e-3 gets a
      WARNING (typo guard), not a refusal.
    """
    if gram_rho_16 is None and gram_rho_32 is None:
        print("=" * 70 + "\nWARNING (FIX-2): neither --gram-rho-16 nor --gram-rho-32 given --\n"
              "arm (iv)'s 6 cells (frozen_gram_matched, K in {16,32} x 3 seeds) are\n"
              "EXCLUDED from this Wave 1 manifest; the other 26 runs proceed. Run\n"
              "calibrate_arm_iv.py, then RE-LAUNCH this wave with the calibrated\n"
              "value(s) -- the resume path adds exactly the missing arm-(iv) cells.\n" + "=" * 70,
              flush=True)
        return None, None
    summary = None
    for K, val in ((16, gram_rho_16), (32, gram_rho_32)):
        if val is None:
            print(f"WARNING (FIX-2): --gram-rho-{K} not given -- arm (iv)'s K={K} cells "
                  f"EXCLUDED from this manifest (resume-safe, re-launch with the flag to add them).",
                  flush=True)
            continue
        if accept_unconverged:
            print(f"WARNING (FIX-2): --accept-unconverged-rho -- K={K} rho={val} accepted "
                  f"WITHOUT calibration verification (explicit human override, recorded here).",
                  flush=True)
            continue
        if summary is None:
            if not os.path.exists(calib_summary_path):
                print(f"ERROR (FIX-2): --gram-rho-{K}={val} given but no calibration summary at\n"
                      f"  {calib_summary_path}\n"
                      f"Run calibrate_arm_iv.py first (its CALIBRATION_SUMMARY.json is the "
                      f"required provenance), pass --calib-summary if it lives elsewhere, or "
                      f"pass --accept-unconverged-rho for an explicit override.", file=sys.stderr)
                sys.exit(1)
            with open(calib_summary_path) as f:
                summary = json.load(f)
        entry = summary.get(str(K)) or summary.get(K)
        if not entry or entry.get("converged") is not True:
            print(f"ERROR (FIX-2): calibration summary {calib_summary_path} has no "
                  f"converged:true entry for K={K} "
                  f"(found: {entry and {k: entry.get(k) for k in ('converged', 'final_rho', 'simulated')}}). "
                  f"Re-run calibrate_arm_iv.py to convergence, or pass --accept-unconverged-rho "
                  f"for an explicit override.", file=sys.stderr)
            sys.exit(1)
        if entry.get("simulated"):
            print(f"ERROR (FIX-2): K={K}'s summary entry is a SIMULATION record (raw-row oracle, "
                  f"no GPU probe) -- a simulated calibration must never gate a real Wave 1 "
                  f"launch. Run the real calibration, or pass --accept-unconverged-rho.",
                  file=sys.stderr)
            sys.exit(1)
        rec = entry.get("final_rho")
        if rec is not None and abs(rec - val) > 1e-3:
            print(f"WARNING (FIX-2): --gram-rho-{K}={val} differs from the calibrated "
                  f"final_rho={rec} by {abs(rec - val):.4f} -- typo? Proceeding with the "
                  f"PASSED value; the discrepancy is recorded in this log.", flush=True)
    return gram_rho_16, gram_rho_32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/deltanet_rd_exactness"))
    ap.add_argument("--wave", choices=["-1", "0", "1", "geo3"], default=None,
                     help="'geo3' launches F-geo-3's Wave-1-style cells (sec 14.7) -- GATED on "
                          "--geo3-drift-json (sec 14.6's launch-read result) unless "
                          "--accept-gate-override is passed. The sec 14.6 drift diagnostic ITSELF "
                          "is a separate driver (geo3_drift_diagnostic.py), not launched by this "
                          "script -- same precedent as arm (iv)'s calibrate_arm_iv.py.")
    ap.add_argument("--gpus", type=int, default=None,
                     help="GPU COUNT. REQUIRED for a real (wave -1/1) launch, NO DEFAULT ON "
                          "PURPOSE -- check nvidia-smi before every launch (GPUs 0-5,7 run other "
                          "waves on this box; GPU 6 frees when ReserveMH drains).")
    ap.add_argument("--gpu-offset", type=int, default=None,
                     help="first physical GPU index. REQUIRED for a real launch, no default.")
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--w0-dirs", nargs="+", default=None,
                     help="--wave 0: directories to scan. CANONICAL (FIX-3): on the box "
                          "results/deltanet_rd_w0b/{wave0,waveA}; on the local Mac "
                          "experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA}. "
                          "NEVER results/deltanet_rd/wave0 (stale pre-rerun tree, same "
                          "filenames, different data -- analyze_exactness_w0.py's content-hash "
                          "guard aborts on any such mix).")
    ap.add_argument("--gram-rho-16", type=float, default=None,
                     help="--wave 1: arm (iv)'s CALIBRATED rho at K=16 (from calibrate_arm_iv.py's "
                          "CALIBRATION_SUMMARY.json -- FIX-2 verifies converged:true there before "
                          "accepting the value; --accept-unconverged-rho overrides). Omit BOTH "
                          "gram-rho flags to run Wave 1 WITHOUT arm (iv) (loud warning, 26 runs; "
                          "resume-safe -- re-launch with the flags later to add those cells).")
    ap.add_argument("--gram-rho-32", type=float, default=None,
                     help="same as --gram-rho-16, at K=32.")
    ap.add_argument("--calib-summary",
                     default=os.path.join(HERE, "results/deltanet_rd_exactness/arm_iv_calib/CALIBRATION_SUMMARY.json"),
                     help="FIX-2: path to calibrate_arm_iv.py's REAL calibration summary (the "
                          "provenance check for --gram-rho-16/32; simulation summaries are "
                          "written to a different filename and are rejected).")
    ap.add_argument("--accept-unconverged-rho", action="store_true",
                     help="FIX-2's explicit human override: accept --gram-rho-16/32 without a "
                          "converged calibration record (loud warning, recorded in the launch log).")
    ap.add_argument("--geo3-drift-json", type=str, default=None,
                     help="--wave geo3: path to geo3_drift_diagnostic.py's output JSON (sec 14.6's "
                          "gating diagnostic). REQUIRED unless --accept-gate-override is passed.")
    ap.add_argument("--accept-gate-override", action="store_true",
                     help="--wave geo3: bypass the sec 14.6 drift launch-read gate with an "
                          "explicit, loudly-logged human decision. Use only after a documented "
                          "review of WHY the gate failed (or was not run) -- this is the same "
                          "class of override as --accept-unconverged-rho above, never a default.")
    ap.add_argument("--include-k48", action="store_true",
                     help="--wave geo3: include the K=48 OPTIONAL Reserve-eligible stretch cells "
                          "(sec 14.7). Per sec 14.12 these should only launch after both mandatory "
                          "K=16/32 cells complete without tripping sec 14.8 outcome D -- a human "
                          "assessment, not automated here; this flag is an explicit opt-in.")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    if args.dry_run:
        from collections import Counter
        neg1 = wave_neg1_manifest()
        w1 = wave1_manifest(args.gram_rho_16 or 0.3, args.gram_rho_32 or 0.3)
        wg3 = geo3_wave1_manifest(args.include_k48)
        print(f"wave -1: {len(neg1)} runs | by arm {Counter(s['arm'] for s in neg1)}")
        print(f"wave  0: free, no GPU -- shells out to analyze_exactness_w0.py")
        print(f"wave  1: {len(w1)} runs | by arm {Counter(s['arm'] for s in w1)} "
              f"(gram_rho placeholder 0.3 used above if --gram-rho-16/32 not given)")
        print(f"wave geo3: {len(wg3)} runs (K=16,32 x3 seeds mandatory"
              f"{', +K=48 x3 seeds (--include-k48)' if args.include_k48 else ' -- pass --include-k48 for the +3 optional stretch cells'}) "
              f"-- GATED on --geo3-drift-json / --accept-gate-override, NOT shown as launchable here")
        print(f"TOTAL new GPU runs (waves -1 + 1): {len(neg1) + len(w1)} "
              f"(design's own count: 9 + 32 = 41, excluding arm (iv)'s <=6 sequential "
              f"calibration probes -- calibrate_arm_iv.py, run separately)")
        if args.gpus is not None and args.gpu_offset is not None:
            print(f"slots = {args.gpus} gpus x {args.per_gpu} per-gpu = {args.gpus * args.per_gpu} "
                  f"concurrent, on physical GPUs {list(range(args.gpu_offset, args.gpu_offset + args.gpus))}")
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "0":
        if not args.w0_dirs:
            print("ERROR: --wave 0 requires --w0-dirs.", file=sys.stderr)
            sys.exit(1)
        run_wave0(os.path.join(args.out_dir, "wave0"), args.w0_dirs)
        return

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose): the busy-GPU set on this box changes day to day. Run nvidia-smi NOW "
              "and pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "-1":
        manifest = wave_neg1_manifest()
    elif args.wave == "geo3":
        if not args.geo3_drift_json and not args.accept_gate_override:
            print("ERROR: --wave geo3 requires --geo3-drift-json <path> (sec 14.6's gating "
                  "diagnostic output -- run geo3_drift_diagnostic.py first) or an explicit "
                  "--accept-gate-override.", file=sys.stderr)
            sys.exit(1)
        gate_result = gate_geo3_drift(args.geo3_drift_json, args.accept_gate_override)
        manifest = geo3_wave1_manifest(args.include_k48)
        print(f"wave geo3 manifest: {len(manifest)} runs (K=16,32 x3 seeds mandatory"
              f"{', +K=48 x3 seeds' if args.include_k48 else ''}); gate={gate_result}", flush=True)
    else:
        g16, g32 = gate_gram_rho(args.gram_rho_16, args.gram_rho_32,
                                   args.calib_summary, args.accept_unconverged_rho)
        manifest = wave1_manifest(g16, g32)
        print(f"wave 1 manifest: {len(manifest)} runs "
              f"(arm (iv) K=16 {'INCLUDED' if g16 is not None else 'EXCLUDED'}, "
              f"K=32 {'INCLUDED' if g32 is not None else 'EXCLUDED'})", flush=True)

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

    try:
        while pending or running:
            while free and pending:
                gpu = free.pop()
                spec = pending.pop(0)
                timeout = args.timeout if args.timeout is not None else default_timeout(spec["K"], spec["steps"])
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
                if rc == 0 and is_done(out_dir, spec):
                    done_ct += 1
                else:
                    failed.append((spec["name"], rc))
            write_progress(out_dir, done_ct, len(failed), len(running), args.wave)
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

    write_progress(out_dir, done_ct, len(failed), 0, args.wave)
    # ALL_DONE sentinel: written ONLY if every manifest item validity-checks
    # as done (never on a partial/failed wave -- a resume must remain safe).
    still_pending = [s for s in manifest if not is_done(out_dir, s)]
    if not still_pending:
        with open(os.path.join(out_dir, "ALL_DONE"), "w") as f:
            f.write(f"wave {args.wave}: {len(manifest)}/{len(manifest)} runs done, "
                    f"{len(failed)} failed-then-recovered or transient.\n")
        print(f"ALL_DONE written -- every manifest item for wave {args.wave} is validity-checked done.",
              flush=True)
    print(f"\nWAVE {args.wave} DONE. {done_ct} succeeded this session, {len(failed)} failed, "
          f"{len(still_pending)} still pending.", flush=True)
    if args.wave == "-1":
        print("NEXT STEP (not run by this script -- sequential, not embarrassingly parallel): "
              "python calibrate_arm_iv.py --k 16 32 --w0-report "
              "<wave0-out-dir>/w0_report.json --gpu <free GPU>", flush=True)


if __name__ == "__main__":
    main()
