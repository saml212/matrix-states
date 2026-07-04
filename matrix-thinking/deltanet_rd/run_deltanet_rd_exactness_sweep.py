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
import key_anchoring as ka   # KEY_ANCHORING_DESIGN.md Rev 5 -- fla-free, CPU-safe shared module

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
          geo3_active=False, geo3_n_iter=12, geo3_resid_tol=1e-2,
          anchor_active=False, anchor_lambda_mode="learned", anchor_lambda_fixed=None,
          lambda_anchor=0.0, drift_probe=False):
    """Variant-carrying run spec (sec 5's naming requirement) -- the `arm`
    tag alone would NOT be resume-safe if two arms shared identical other
    fields, so the name also encodes the fields that actually vary the
    trained artifact (embed_source, gram_rho, strong_pin, lambda_orth,
    use_zca, fnce_m, geo3_active/geo3_n_iter, and now KEY_ANCHORING_DESIGN.md's
    anchor_active/anchor_lambda_mode/anchor_lambda_fixed/lambda_anchor/
    drift_probe); is_done() below re-derives the SAME identity from the
    result JSON's own exactness_config, never trusting the filename alone."""
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
    if anchor_active:
        bits.append("anchor")
        bits.append(anchor_lambda_mode)
        if anchor_lambda_mode == "fixed":
            bits.append(f"lam{anchor_lambda_fixed}".replace(".", "p"))
    if lambda_anchor:
        bits.append(f"lanc{lambda_anchor}".replace(".", "p"))
    if drift_probe:
        bits.append("dprobe")
    name = "_".join(bits)
    return {"wave": str(wave), "K": K, "seed": seed, "steps": steps, "force_rank_k": force_rank_k,
            "arm": arm, "embed_source": embed_source, "gram_alpha": gram_alpha, "gram_rho": gram_rho,
            "strong_pin": strong_pin, "lambda_orth": lambda_orth, "use_zca": use_zca, "fnce_m": fnce_m,
            "geo3_active": geo3_active, "geo3_n_iter": geo3_n_iter if geo3_active else None,
            "geo3_resid_tol": geo3_resid_tol if geo3_active else None,
            "anchor_active": anchor_active,
            "anchor_lambda_mode": anchor_lambda_mode if anchor_active else None,
            "anchor_lambda_fixed": anchor_lambda_fixed if anchor_active else None,
            "lambda_anchor": lambda_anchor, "drift_probe": drift_probe,
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


# ---------------------------------------------------------------------------
# KEY_ANCHORING_DESIGN.md (Rev 5) -- the follow-on wave named at the end of
# this design's own §16.6 outcome-F attribution. A DELIBERATE parallel
# manifest set (same precedent as geo3_wave1_manifest above): reuses this
# harness's `_spec`/`is_done`/`build_cmd` machinery but adds its OWN wave
# labels ("ref", "keyanchor") and its OWN sec-3.6 mechanical blinding gate
# (BANDS_PINNED.json) -- reference arms MUST complete and validate before
# any keyanchor cell is allowed to launch (enforced in main(), not by
# analyst discipline; sec 3.6 Rev 4).
#
# 28-run mandatory baseline (sec 5's budget table): 10 Wave -1 (8 short GPU
# probes via keyanchor_wave_neg1_manifest below + 2 SEQUENTIAL
# keyanchor_drift_diagnostic.py driver invocations, K=16/K=32, NOT part of
# this parallel manifest -- same "separate sequential driver" precedent as
# geo3_drift_diagnostic.py/calibrate_arm_iv.py, see this module's own
# docstring) + 6 reference arms + 6 candidate-(d) + 6 candidate-(c) = 28.
# ---------------------------------------------------------------------------

KEYANCHOR_TIER_STEPS = 20000       # continuity with TIER_STEPS_WAVE1
KEYANCHOR_NEG1_PROBE_STEPS = 750   # continuity with NEG1_PROBE_STEPS
KEYANCHOR_DRIFT_PROBE_STEPS = 5000  # the 2 SEQUENTIAL keyanchor_drift_diagnostic.py runs (not manifest cells)
# sec 2.4's single pre-registered lambda_anchor value (candidate (c),
# "run at a single, pre-registered lambda_anchor value" -- no tuning loop).
# Chosen by analogy to F-geo-1's own lambda_orth=1.0 pre-registered value
# (DELTANET_RD_EXACTNESS_DESIGN.md sec 5.5) -- the same "1.0, not searched"
# convention, since L_anchor's own per-hop cos-distance term is already
# O(1)-scaled like L_orth's Frobenius-squared term is for that design.
KEYANCHOR_LAMBDA_ANCHOR = 1.0
BANDS_PINNED_PATH_DEFAULT = "BANDS_PINNED.json"   # relative to the keyanchor out_dir


def reference_arms_manifest():
    """sec 3.6: 3 fresh bare-geo3 seeds x K in {16,32} = 6 runs, 20,000
    steps, --drift-probe active (post-NS AND pre-NS pooled drift logged at
    EVERY checkpoint, key_anchoring.measure_drift). MUST run and validate-
    complete BEFORE `BANDS_PINNED.json` is written (sec 3.6 writer
    requirement (a)) and before any keyanchor cell launches (requirement
    (b)) -- enforced in main(), see gate_bands_pinned below. Seeds {1,2,3}
    (NOT {0,1,2}) -- seed 0 is the existing archived 5,000-step PROBE
    measurement, a different training stage, never pooled with these."""
    steps = KEYANCHOR_TIER_STEPS
    runs = []
    for K in (16, 32):
        n_iter = 12 if K == 16 else 20      # sec 16.3's own per-K NS tier, unchanged from bare geo3
        for s in (1, 2, 3):
            runs.append(_spec("ref", K, s, steps, "georef", geo3_active=True,
                               geo3_n_iter=n_iter, geo3_resid_tol=1e-2, drift_probe=True))
    assert len(runs) == 6, f"reference-arm manifest drifted from its registered 6 runs: {len(runs)}"
    return runs


def keyanchor_wave_neg1_manifest():
    """8 short GPU timing/instability probes (NEG1_PROBE_STEPS-class,
    mirroring wave_neg1_manifest's own precedent) across candidate (d)'s
    learned-lambda mode at both K, its fixed-lambda grid at both K (a light
    probe of the conditional fallback diagnostic's own regime -- cheap
    insurance, not a commitment to run the full grid unconditionally), and
    candidate (c) at both K. Composition is a documented judgment call
    (the design text itemizes exactly what the 9-item CPU smoke list
    covers, sec 5, but does not itemize which 8 GPU-probe cells belong
    here) -- flagged for auditor scrutiny."""
    steps = KEYANCHOR_NEG1_PROBE_STEPS
    runs = []
    for K in (16, 32):
        n_iter = 12 if K == 16 else 20
        runs.append(_spec(-1, K, 0, steps, "keyanchor-d-learned", geo3_active=True, geo3_n_iter=n_iter,
                           geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned"))
    for K in (16, 32):
        n_iter = 12 if K == 16 else 20
        runs.append(_spec(-1, K, 0, steps, "keyanchor-c", geo3_active=True, geo3_n_iter=n_iter,
                           geo3_resid_tol=1e-2, lambda_anchor=KEYANCHOR_LAMBDA_ANCHOR))
    for lam in (0.3, 0.9):
        runs.append(_spec(-1, 32, 0, steps, "keyanchor-d-fixed", geo3_active=True, geo3_n_iter=20,
                           geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="fixed",
                           anchor_lambda_fixed=lam))
    runs.append(_spec(-1, 16, 0, steps, "keyanchor-d-fixed", geo3_active=True, geo3_n_iter=12,
                       geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="fixed",
                       anchor_lambda_fixed=0.6))
    runs.append(_spec(-1, 32, 0, steps, "keyanchor-d-fixed", geo3_active=True, geo3_n_iter=20,
                       geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="fixed",
                       anchor_lambda_fixed=0.6))
    assert len(runs) == 8, f"keyanchor Wave -1 GPU-probe manifest drifted from its registered 8 runs: {len(runs)}"
    return runs


def keyanchor_wave1_manifest():
    """sec 5's mandatory Wave 1 cells: candidate (d) K in {16,32} x 3 seeds
    (learned lambda, the headline cells) + candidate (c) K in {16,32} x 3
    seeds (the always-run ablation comparison, sec 2.4) = 12 runs, 20,000
    steps each. §3.4's early-stop is a run_deltanet_rd.py-internal concern
    (not yet wired into THIS harness's polling loop -- see the build
    report's scrutiny list) -- both candidates' per-entity/lambda logging
    is active by construction (anchor_active/lambda_anchor threaded via
    build_cmd)."""
    steps = KEYANCHOR_TIER_STEPS
    runs = []
    for K in (16, 32):
        n_iter = 12 if K == 16 else 20
        for s in range(3):
            runs.append(_spec("keyanchor", K, s, steps, "d", geo3_active=True, geo3_n_iter=n_iter,
                               geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned"))
    for K in (16, 32):
        n_iter = 12 if K == 16 else 20
        for s in range(3):
            runs.append(_spec("keyanchor", K, s, steps, "c", geo3_active=True, geo3_n_iter=n_iter,
                               geo3_resid_tol=1e-2, lambda_anchor=KEYANCHOR_LAMBDA_ANCHOR))
    assert len(runs) == 12, f"keyanchor Wave 1 manifest drifted from its registered 12 runs: {len(runs)}"
    return runs


def keyanchor_ceiling_by_k() -> dict:
    """sec 2.2's computed lambda=1 post-NS drift ceilings (frame-potential
    init) -- the band-derivation guard's `ceiling_by_k` input."""
    return {16: 0.9745, 32: 0.9423}


def gate_bands_pinned(out_dir: str, accept_override: bool, override_at: float | None) -> dict:
    """sec 3.6's launcher-gate requirement (b): keyanchor cells REFUSE to
    launch unless `BANDS_PINNED.json` exists AND validates (hash-matches
    every referenced reference-arm result JSON). `--unblind-override`
    (mirrored here) bypasses the refusal but AUTO-DEMOTES every anchor-arm
    readout to descriptive tier -- the demotion is threaded through
    build_cmd as `--unblind-override-at <override_at>` so EVERY spawned
    keyanchor run's OWN result JSON carries the stamp at assembly time
    (Rev 5 R4-1 fix), never only this launch-console print."""
    bp_path = os.path.join(out_dir, "BANDS_PINNED.json")
    if accept_override:
        print("=" * 70 + "\nWARNING: --unblind-override -- the sec 3.6 mechanical BANDS_PINNED "
              "blinding gate is being BYPASSED by an explicit human decision. EVERY keyanchor run "
              "launched this session will have its OWN result JSON stamped claim_tier='descriptive' "
              "+ unblind_override=True (Rev 5 R4-1) -- this is recorded per-run, not just here.\n"
              + "=" * 70, flush=True)
        return {"gate_bypassed": True, "bands_pinned_path": bp_path, "override_at": override_at}
    doc = ka.validate_bands_pinned(bp_path)
    if doc is None:
        print(f"ERROR: {bp_path!r} does not exist or fails hash validation. Run the 6 reference "
              f"arms to completion first (--wave ref), then --wave keyanchor-bands to pin, or pass "
              f"--unblind-override for an explicit, documented, tier-demoting override.",
              file=sys.stderr)
        sys.exit(1)
    for K, entry in doc["bands"].items():
        if entry["unresolvable"]:
            print(f"  NOTE: K={K} post-NS engagement bands are UNRESOLVABLE (engaged_K "
                  f"{entry['engaged_k']:.4f} >= ceiling {entry['ceiling']:.4f} - 0.005) -- see the "
                  f"leave-one-out sensitivity report in {bp_path}.", flush=True)
    print(f"sec 3.6 GATE PASSED: {bp_path} validates (hashes match every referenced reference-arm "
          f"result JSON).", flush=True)
    return {"gate_bypassed": False, "bands_pinned_path": bp_path, "bands": doc["bands"]}


def write_bands_pinned_if_ready(out_dir: str) -> bool:
    """sec 3.6 writer requirement (a): called by `--wave keyanchor-bands`.
    Reads the 6 reference-arm result JSONs from `out_dir`, validates EVERY
    ONE as complete (key_anchoring.reference_arm_result_valid), and ONLY
    THEN writes BANDS_PINNED.json. Returns True iff the file was written
    this call (False -> at least one reference arm is not yet valid;
    prints exactly which)."""
    manifest = reference_arms_manifest()
    per_k_final_drift = {16: [], 32: []}
    ref_paths = {16: [], 32: []}
    all_valid = True
    for spec in manifest:
        p = out_path(out_dir, spec)
        if not os.path.exists(p):
            print(f"  NOT READY: {p} does not exist yet.", flush=True)
            all_valid = False
            continue
        with open(p) as f:
            result = json.load(f)
        if not ka.reference_arm_result_valid(result, spec["steps"]):
            print(f"  NOT READY: {p} exists but does not validate as complete "
                  f"(complete/steps_completed/drift_probe fields).", flush=True)
            all_valid = False
            continue
        final_drift = result["checkpoints"][-1]["drift_probe"]["post_ns"]["mean"]
        per_k_final_drift[spec["K"]].append(final_drift)
        ref_paths[spec["K"]].append(p)
    if not all_valid:
        print("BANDS_PINNED.json NOT written -- not every reference arm validates complete yet.",
              file=sys.stderr)
        return False
    bp_path = os.path.join(out_dir, "BANDS_PINNED.json")
    doc = ka.write_bands_pinned(bp_path, per_k_final_drift, keyanchor_ceiling_by_k(), ref_paths)
    print(f"BANDS_PINNED.json written at {bp_path}:\n{json.dumps(doc['bands'], indent=2)}", flush=True)
    return True


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
        # KEY_ANCHORING_DESIGN.md sec 2.2/2.4 fields -- SAME missing-key-
        # defaults-to-off discipline as the geo3 fields above (archived
        # pre-KEY_ANCHORING result JSONs lack these keys entirely).
        if bool(ec.get("anchor_active", False)) != bool(spec.get("anchor_active", False)):
            return False
        if spec.get("anchor_active"):
            if ec.get("anchor_lambda_mode") != spec.get("anchor_lambda_mode") or \
               ec.get("anchor_lambda_fixed") != spec.get("anchor_lambda_fixed"):
                return False
        if ec.get("lambda_anchor", 0.0) != spec.get("lambda_anchor", 0.0):
            return False
        return True
    except Exception:
        return False


def build_cmd(spec, out_dir, timeout, unblind_override_at: float | None = None):
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
    if spec.get("anchor_active"):
        cmd += ["--anchor-active", "--anchor-lambda-mode", str(spec["anchor_lambda_mode"])]
        if spec["anchor_lambda_mode"] == "fixed":
            cmd += ["--anchor-lambda-fixed", str(spec["anchor_lambda_fixed"])]
    if spec.get("lambda_anchor"):
        cmd += ["--lambda-anchor", str(spec["lambda_anchor"])]
    if spec.get("drift_probe"):
        cmd += ["--drift-probe"]
    if unblind_override_at is not None and (spec.get("anchor_active") or spec.get("lambda_anchor")):
        # sec 3.6 Rev 5 (R4-1 fix): the override stamp is threaded down to
        # EVERY spawned anchor-arm run so its OWN result JSON carries the
        # demotion at assembly time, never only this launch console.
        cmd += ["--unblind-override-at", str(unblind_override_at)]
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
    ap.add_argument("--wave", choices=["-1", "0", "1", "geo3", "ref", "keyanchor-neg1",
                                        "keyanchor-bands", "keyanchor"], default=None,
                     help="'geo3' launches F-geo-3's Wave-1-style cells (sec 14.7) -- GATED on "
                          "--geo3-drift-json (sec 14.6's launch-read result) unless "
                          "--accept-gate-override is passed. The sec 14.6 drift diagnostic ITSELF "
                          "is a separate driver (geo3_drift_diagnostic.py), not launched by this "
                          "script -- same precedent as arm (iv)'s calibrate_arm_iv.py. "
                          "KEY_ANCHORING_DESIGN.md (Rev 5) waves: 'ref' launches the 6 mandatory "
                          "reference arms (sec 3.6, MUST run and validate-complete FIRST); "
                          "'keyanchor-bands' is a NO-GPU, no-manifest action that reads the "
                          "reference arms' own result JSONs and writes BANDS_PINNED.json if (and "
                          "only if) all 6 validate complete (sec 3.6 writer requirement (a)); "
                          "'keyanchor-neg1' launches the 8 short GPU probes; 'keyanchor' launches "
                          "the 12 mandatory Wave-1 cells (candidates (d)+(c)) -- REFUSES to launch "
                          "without a valid BANDS_PINNED.json (sec 3.6 requirement (b)) unless "
                          "--unblind-override is passed.")
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
    ap.add_argument("--unblind-override", action="store_true",
                     help="--wave keyanchor: KEY_ANCHORING_DESIGN.md sec 3.6's explicit, "
                          "loudly-logged human override of the BANDS_PINNED mechanical blinding "
                          "gate. EVERY keyanchor run launched under this flag has its OWN result "
                          "JSON stamped claim_tier='descriptive' + unblind_override=True + a "
                          "timestamp at assembly time (Rev 5 R4-1 fix) -- never only this launch "
                          "console or a wave-summary artifact.")
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

        print("\n" + "=" * 70)
        print("KEY_ANCHORING_DESIGN.md (Rev 5) -- 28-run mandatory-baseline preview")
        print("=" * 70)
        ref = reference_arms_manifest()
        kneg1 = keyanchor_wave_neg1_manifest()
        kw1 = keyanchor_wave1_manifest()
        n_manifest = len(ref) + len(kneg1) + len(kw1)
        n_sequential = 2   # keyanchor_drift_diagnostic.py --k 16, --k 32 (NOT parallel-manifest cells)
        n_total = n_manifest + n_sequential
        # sec 5's own per-run GPU-h anchors, realized from geo3's own
        # measured spend (sec 5's table; ~0.24-0.28 GPU-h/run at the
        # instrumented rate, ~0.83 GPU-h/run for the drift-instrumented
        # reference arms).
        # per-run GPU-h anchors (sec 5): a full 20,000-step candidate (d)/(c)
        # cell costs ~0.24-0.28 GPU-h; a short NEG1_PROBE_STEPS-class probe
        # scales down proportionally to its step count.
        _PER_RUN_GPUH_FULL = 0.28
        gpuh_neg1 = len(kneg1) * _PER_RUN_GPUH_FULL * (KEYANCHOR_NEG1_PROBE_STEPS / KEYANCHOR_TIER_STEPS)
        gpuh_seq = 0.85   # ~0.7-1.0 total, sec 5's own "8 probes + 2 drift-diagnostic probes" row
        gpuh_ref = len(ref) * 0.83
        gpuh_kw1 = len(kw1) * 0.28
        gpuh_baseline = gpuh_neg1 + gpuh_seq + gpuh_ref + gpuh_kw1
        print(f"  reference arms (--wave ref):        {len(ref)} runs | by K "
              f"{Counter(s['K'] for s in ref)} | seeds {{1,2,3}} | ~{gpuh_ref:.2f} GPU-h")
        print(f"  keyanchor Wave -1 GPU probes:        {len(kneg1)} runs | by arm "
              f"{Counter(s['arm'] for s in kneg1)} | ~{gpuh_neg1:.2f} GPU-h")
        print(f"  keyanchor Wave -1 sequential probes: {n_sequential} runs "
              f"(keyanchor_drift_diagnostic.py --k 16 / --k 32, NOT in this parallel manifest) "
              f"| ~{gpuh_seq:.2f} GPU-h")
        print(f"  keyanchor Wave 1 (candidates d+c):   {len(kw1)} runs | by arm "
              f"{Counter(s['arm'] for s in kw1)} | ~{gpuh_kw1:.2f} GPU-h")
        print(f"  ---")
        print(f"  TOTAL mandatory baseline: {n_total} runs (10 Wave-1 [8+2] + 6 ref + 6 d + 6 c = 28 "
              f"expected) | projected ~{gpuh_baseline:.2f} GPU-h")
        print(f"  BUDGET: wave ceiling <=10 GPU-h nominal / <=15 with reserve (sec 5); program cap "
              f"80 GPU-h, ~34.9 GPU-h already spent (R3-verified) -> "
              f"~{34.9 + gpuh_baseline:.2f} GPU-h program total under this preview "
              f"({'WITHIN' if 34.9 + gpuh_baseline <= 80 else 'EXCEEDS'} the 80 GPU-h cap)")
        assert n_manifest + n_sequential == 28, \
            f"KEY_ANCHORING mandatory-baseline run count drifted from the registered 28: {n_manifest + n_sequential}"

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

    if args.wave == "keyanchor-bands":
        # sec 3.6 writer requirement (a): NO GPU, no manifest -- reads the 6
        # reference-arm result JSONs and writes BANDS_PINNED.json iff ALL
        # SIX validate complete.
        ref_out_dir = os.path.join(args.out_dir, "waveref")
        ok = write_bands_pinned_if_ready(ref_out_dir)
        sys.exit(0 if ok else 1)

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose): the busy-GPU set on this box changes day to day. Run nvidia-smi NOW "
              "and pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)

    unblind_override_at = None
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
    elif args.wave == "ref":
        manifest = reference_arms_manifest()
        print(f"wave ref manifest: {len(manifest)} runs (bare-geo3, seeds {{1,2,3}} x K in "
              f"{{16,32}}, --drift-probe active) -- sec 3.6: MUST complete + validate BEFORE "
              f"'--wave keyanchor-bands' can pin BANDS_PINNED.json", flush=True)
    elif args.wave == "keyanchor-neg1":
        manifest = keyanchor_wave_neg1_manifest()
        print(f"wave keyanchor-neg1 manifest: {len(manifest)} short GPU probes "
              f"({KEYANCHOR_NEG1_PROBE_STEPS} steps each) -- NOT gated on BANDS_PINNED (these are "
              f"timing/instability probes, not admission-tier cells)", flush=True)
    elif args.wave == "keyanchor":
        # sec 3.6 launcher-gate requirement (b): REFUSES to launch without a
        # valid BANDS_PINNED.json (written from the "ref" wave's own
        # results), unless --unblind-override is passed.
        ref_out_dir = os.path.join(args.out_dir, "waveref")
        if args.unblind_override:
            unblind_override_at = time.time()
        gate_result = gate_bands_pinned(ref_out_dir, args.unblind_override, unblind_override_at)
        manifest = keyanchor_wave1_manifest()
        print(f"wave keyanchor manifest: {len(manifest)} runs (candidates (d)+(c), K in {{16,32}} "
              f"x3 seeds each); gate={gate_result}", flush=True)
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
                    proc = subprocess.Popen(build_cmd(spec, out_dir, timeout,
                                                       unblind_override_at=unblind_override_at),
                                             env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
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
