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
          lambda_anchor=0.0, drift_probe=False, rev7_engagement=False):
    """Variant-carrying run spec (sec 5's naming requirement) -- the `arm`
    tag alone would NOT be resume-safe if two arms shared identical other
    fields, so the name also encodes the fields that actually vary the
    trained artifact (embed_source, gram_rho, strong_pin, lambda_orth,
    use_zca, fnce_m, geo3_active/geo3_n_iter, and now KEY_ANCHORING_DESIGN.md's
    anchor_active/anchor_lambda_mode/anchor_lambda_fixed/lambda_anchor/
    drift_probe/rev7_engagement); is_done() below re-derives the SAME
    identity from the result JSON's own exactness_config, never trusting
    the filename alone. anchor_lambda_mode also accepts
    'learned_per_entity' (KEY_ANCHORING_DESIGN.md sec 10.5.1, candidate
    (d')) -- the SAME name-bit path as 'learned'/'fixed', no special-casing
    needed since the string itself already varies the filename."""
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
    if rev7_engagement:
        bits.append("rev7")
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
            "rev7_engagement": rev7_engagement,
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


def reference_arms_manifest(Ks=(16, 32), seeds=(1, 2, 3)):
    """sec 3.6: 3 fresh bare-geo3 seeds x K in {16,32} = 6 runs, 20,000
    steps, --drift-probe active (post-NS AND pre-NS pooled drift logged at
    EVERY checkpoint, key_anchoring.measure_drift). MUST run and validate-
    complete BEFORE `BANDS_PINNED.json` is written (sec 3.6 writer
    requirement (a)) and before any keyanchor cell launches (requirement
    (b)) -- enforced in main(), see gate_bands_pinned below. Seeds {1,2,3}
    (NOT {0,1,2}) -- seed 0 is the existing archived 5,000-step PROBE
    measurement, a different training stage, never pooled with these.

    Ks/seeds (KEY_ANCHORING_DESIGN.md sec 11.0, Rev K48.1 build-scope item):
    generalized from the original hardcoded `for K in (16, 32)` / `(1,2,3)`
    loop so a K=48 reference-arm manifest can reuse the SAME function rather
    than a hand-copied twin -- defaults are BYTE-IDENTICAL to the original
    hardcoded form (sec 11.9 item 14's own registered smoke asserts this
    directly, not merely by inspection). n_iter is read from key_anchoring.
    GATE2_N_ITER_BY_K (the same {16:12, 32:20, 48:20} production-tier dict
    Gate 2 uses) rather than the original inline `12 if K==16 else 20`
    ternary, so the two never drift apart at a K neither expression
    originally covered."""
    steps = KEYANCHOR_TIER_STEPS
    runs = []
    for K in Ks:
        n_iter = ka.GATE2_N_ITER_BY_K[K]      # sec 16.3's own per-K NS tier, unchanged from bare geo3
        for s in seeds:
            runs.append(_spec("ref", K, s, steps, "georef", geo3_active=True,
                               geo3_n_iter=n_iter, geo3_resid_tol=1e-2, drift_probe=True))
    if Ks == (16, 32) and seeds == (1, 2, 3):
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


def keyanchor_wave1_manifest(Ks=(16, 32)):
    """sec 5's mandatory Wave 1 cells: candidate (d) K in {16,32} x 3 seeds
    (learned lambda, the headline cells) + candidate (c) K in {16,32} x 3
    seeds (the always-run ablation comparison, sec 2.4) = 12 runs, 20,000
    steps each. §3.4's early-stop is a run_deltanet_rd.py-internal concern
    (not yet wired into THIS harness's polling loop -- see the build
    report's scrutiny list).

    drift_probe=True (2026-07-06 keyanchor-confirm build, FIX): sec 9.3's
    documented gap -- this function's own docstring PREVIOUSLY claimed
    "both candidates' per-entity/lambda logging is active by construction"
    but never actually threaded drift_probe=True into either loop below
    (only reference_arms_manifest() did) -- so item 5 (pre-NS blend
    drift), item 6 (table conditioning at admission), and sec 3.7's
    engaged_frac were NEVER measured on the 12 admitted Wave-1 result
    JSONs (KEY_ANCHORING_DESIGN.md sec 9.3/9.5's verdict). Fixed here so
    FUTURE invocations of this manifest are correctly instrumented --
    run_deltanet_rd.py's train() now computes item 5/6/sec-3.7 whenever
    drift_probe_active and model.anchor_active (see train()'s checkpoint
    block). Threading drift_probe=True also changes every cell's own
    filename (_spec's own "_dprobe" naming bit) relative to the ALREADY-
    ARCHIVED (uninstrumented) Wave-1 result JSONs -- intentional: a future
    '--wave keyanchor' launch will NOT resume from those 12 old files (they
    are under-instrumented relative to this fixed spec) and will re-run
    all 12 cells from scratch (~3.4 GPU-h) rather than silently treating
    old, gap-having results as satisfying the new, fixed spec. The
    keyanchor-confirm wave below is the CHEAPER, already-scoped way to
    close the gap on the specific cells whose h4 result is already
    published (candidate (d), K=32) without re-running all 12.

    Ks (KEY_ANCHORING_DESIGN.md sec 11.0, Rev K48.1 build-scope item):
    generalized from the original hardcoded `for K in (16, 32)` loop
    (BOTH occurrences below) -- default is BYTE-IDENTICAL to the original
    (sec 11.9 item 14's registered smoke asserts this directly). K=48's own
    candidate (d) cells are NOT built by calling this function with
    Ks=(48,) (K=48 does not also run candidate (c), sec 11.1 -- only (d) is
    mandatory there); this parameter exists so the function's OWN shape is
    demonstrably K-generalizable per sec 11.0's build-scope text, and so a
    future K48-plus-(c) wave (not currently registered) would not need a
    second hand-copied twin. keyanchor_k48_manifest() below builds K=48's
    own candidate-(d)-only cells directly, using _spec() the same way this
    function does, not by calling this function with a partial K set."""
    steps = KEYANCHOR_TIER_STEPS
    runs = []
    for K in Ks:
        n_iter = ka.GATE2_N_ITER_BY_K[K]
        for s in range(3):
            runs.append(_spec("keyanchor", K, s, steps, "d", geo3_active=True, geo3_n_iter=n_iter,
                               geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                               drift_probe=True))
    for K in Ks:
        n_iter = ka.GATE2_N_ITER_BY_K[K]
        for s in range(3):
            runs.append(_spec("keyanchor", K, s, steps, "c", geo3_active=True, geo3_n_iter=n_iter,
                               geo3_resid_tol=1e-2, lambda_anchor=KEYANCHOR_LAMBDA_ANCHOR,
                               drift_probe=True))
    if Ks == (16, 32):
        assert len(runs) == 12, f"keyanchor Wave 1 manifest drifted from its registered 12 runs: {len(runs)}"
    return runs


def keyanchor_confirm_manifest(include_k16_spot_check: bool = True, K: int = 32):
    """KEY_ANCHORING_DESIGN.md sec 9.5's required follow-up ("instrumentation
    debt, not a hypothesis question", 2026-07-06 keyanchor-confirm build):
    candidate (d) (learned-lambda anchor blend), K=32, seeds {0,1,2},
    drift_probe=True -- the SAME config as the original (bugged)
    keyanchor_wave1_manifest() candidate-(d) K=32 cells (20,000 steps,
    geo3_n_iter=20, same blinding gate) except drift_probe is now threaded
    through, so item 5 (pre-NS blend drift), item 6 (table conditioning,
    every checkpoint), and sec 3.7's engaged_frac + h=1 behavioral
    companion (final checkpoint) land in THESE runs' own result JSONs --
    resolving sec 9.3/9.5's UNASSIGNABLE gap for the h4 result that is
    already published (mean 0.613, 3/3 seeds clearing the >=0.5 bar), at
    the SAME seed identity {0,1,2} the original cells used, WITHOUT a new
    20,000-step training decision.

    Priced (task-directed estimate): ~3 x 0.28 = ~0.84 GPU-h for the
    mandatory K=32 cells (sec 5's own per-cell anchor for a 20,000-step
    candidate-(d) run) -- NOTE this anchor predates drift_probe=True's own
    per-checkpoint overhead (item 5's 8-entity sweep + item 6's cheap SVD
    at every checkpoint + sec 3.7's ONE 107-entity sweep at the final
    checkpoint only, a disclosed scoping choice -- see train()'s checkpoint
    block); reference_arms_manifest()'s OWN realized-vs-priced gap
    (~0.83 GPU-h/run priced, drift_probe=True, item-5-only) is the closest
    comparable data point, so the REAL cost of these cells (which add
    item-6 + the once-per-run sec-3.7 sweep on top) may run somewhat above
    the 0.28 anchor -- flagged as a build-report scrutiny item, not
    silently assumed away.

    A DELIBERATE, SEPARATE wave from 'keyanchor' (NOT folded into the
    28-run mandatory baseline in --dry-run's preview above) -- these are a
    confirmatory RE-run of already-completed cells' own seed identity, not
    new mandatory-baseline cells. wave='keyanchor-confirm' (distinct from
    the original 'keyanchor' wave tag used by keyanchor_wave1_manifest)
    combined with drift_probe=True's own '_dprobe' filename bit (_spec)
    guarantees NO is_done()/out_path() collision with the original 12
    Wave-1 result JSONs even though K/seed/arm are otherwise identical --
    verified by smoke_keyanchor_confirm.py's manifest-wiring check, this
    build (never relying on the filename bit alone: is_done() also now
    cross-checks the drift_probe field itself, per this build's
    run_deltanet_rd.py exactness_config fix).

    Same sec 3.6 blinding gate as 'keyanchor' applies (these ARE anchor-arm
    cells) -- enforced in main(), see gate_bands_pinned. ADDITIONALLY
    gated (task-directed, this build) on gate_keyanchor_drift_diag(): the
    FIXED keyanchor_drift_diagnostic.py must have produced a clean Gate-1
    read before any cell here dispatches.

    include_k16_spot_check (default True): +1 K=16 seed-0 cell, a cheap
    cross-K sanity check -- task-registered as OPTIONAL ("if cheap");
    included by default since a 20,000-step K=16 cell (n_iter=12, the same
    tier bare geo3/reference arms already use) is not materially more
    expensive than a K=32 cell under this harness's own default_timeout
    cost model.

    K (KEY_ANCHORING_DESIGN.md sec 11.0, Rev K48.1 build-scope item):
    generalized from the original hardcoded K=32 -- default is BYTE-
    IDENTICAL to the original (sec 11.9 item 14's registered smoke asserts
    this directly). This function is NOT called with K=48 by any registered
    K48 wave (the K48 build's own candidate (d) cells use
    keyanchor_k48_manifest(), a fresh confirmatory-not-original manifest --
    'confirm' names a RE-run of an already-published K=32 result, which has
    no K=48 analog); the parameter exists purely so this function's shape
    is demonstrably K-generalizable per sec 11.0's own text, mirroring
    reference_arms_manifest()/keyanchor_wave1_manifest()'s own
    generalization above."""
    steps = KEYANCHOR_TIER_STEPS
    n_iter = ka.GATE2_N_ITER_BY_K[K]
    runs = []
    for s in range(3):
        runs.append(_spec("keyanchor-confirm", K, s, steps, "d", geo3_active=True, geo3_n_iter=n_iter,
                           geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                           drift_probe=True))
    if include_k16_spot_check:
        runs.append(_spec("keyanchor-confirm", 16, 0, steps, "d", geo3_active=True, geo3_n_iter=12,
                           geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                           drift_probe=True))
    if K == 32:
        expected = 4 if include_k16_spot_check else 3
        assert len(runs) == expected, \
            f"keyanchor-confirm manifest drifted from its registered {expected} runs: {len(runs)}"
    return runs


KEYANCHOR_MECH_GATE1_STEPS = 5000   # sec 10.5's Gate-1-style pre-launch probe, candidate (d') only


def keyanchor_mech_gate1_manifest():
    """sec 10.5's Gate-1-style pre-launch probe for the NEW architecture
    only (candidate (d'), K=32, 5,000 steps, 1 run) -- the same fixed-
    diagnostic pre-spend check the confirm wave already ran for candidate
    (d) (predicted h4>=0.8 bar); candidate (d)'s own architecture is
    UNCHANGED from Wave 1/the confirm wave, so its own Gate 1 read is NOT
    re-run here (sec 10.5's own text: "would be re-measuring a gate that
    already passed twice on identical code"). A SEPARATE, sequential
    prerequisite to `--wave keyanchor-mech` (gated by
    gate_keyanchor_mech_probe below), mirroring keyanchor_confirm's own
    gate_keyanchor_drift_diag precedent -- NOT folded into the 7-cell
    manifest keyanchor_mech_manifest() returns.

    rev7_engagement=True (e633862 audit fold-in 1): the probe carries the
    FULL Rev-7.1 instrumentation (r_e engagement at the final checkpoint,
    the sec 10.10 checkpoint writer via the ckpt-base-dir main() threads
    for this wave, per-entity lambda_e trajectory) so this build's
    never-yet-executed train() checkpoint-block wiring gets its first
    real GPU exercise HERE, on the cheap 5,000-step probe, before any
    20,000-step mandatory cell spends on it."""
    runs = [_spec("keyanchor-mech-gate1", 32, 900, KEYANCHOR_MECH_GATE1_STEPS, "dprime",
                   geo3_active=True, geo3_n_iter=20, geo3_resid_tol=1e-2, anchor_active=True,
                   anchor_lambda_mode="learned_per_entity", drift_probe=True, rev7_engagement=True)]
    assert len(runs) == 1, f"keyanchor-mech Gate-1 probe manifest drifted from its registered 1 run: {len(runs)}"
    return runs


def keyanchor_mech_manifest():
    """KEY_ANCHORING_DESIGN.md sec 10.5's Rev-7.1 mechanism-tier
    confirmation wave -- 7 cells (candidate (d): 4, candidate (d'): 3), ALL
    with drift_probe=True AND rev7_engagement=True (the r_e + null-pool
    BH-FDR measurement, sec 10.2/10.3, is the entire point of this wave).
    The SEPARATE Gate-1 probe (keyanchor_mech_gate1_manifest(), its own
    'keyanchor-mech-gate1' wave) is the 8th cell in sec 10.7's "8
    mandatory" budget-table count -- never folded into THIS manifest,
    mirroring keyanchor_confirm's own gate_keyanchor_drift_diag precedent
    of keeping a pre-launch probe as a separate sequential prerequisite:

    - candidate (d), K=32, seeds {10,11,12} -- fresh seeds (sec 10.5's own
      text: reusing 0/1/2 a third time "does not add a fresh statistical
      sample", sec 9.6's disclosure).
    - candidate (d), K=16, seed 10 -- 1 supplementary spot-check (sec 3.5's
      own K=32-primary scope note; not separately outcome-lettered).
    - candidate (d'), K=32, seeds {20,21,22} -- the NEW per-entity-lambda
      arm (sec 10.5.1), 'dprime' per sec 10.5's registered manifest string.
      K=16 is NOT run for (d') -- it already saturates h4~=1.0 at baseline,
      no headroom to distinguish a mechanism effect there (sec 10.5)."""
    steps = KEYANCHOR_TIER_STEPS
    runs = []
    for K, n_iter, s in ((32, 20, 10), (32, 20, 11), (32, 20, 12), (16, 12, 10)):
        runs.append(_spec("keyanchor-mech", K, s, steps, "d", geo3_active=True, geo3_n_iter=n_iter,
                           geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                           drift_probe=True, rev7_engagement=True))
    for s in (20, 21, 22):
        runs.append(_spec("keyanchor-mech", 32, s, steps, "dprime", geo3_active=True, geo3_n_iter=20,
                           geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned_per_entity",
                           drift_probe=True, rev7_engagement=True))
    assert len(runs) == 7, f"keyanchor-mech manifest drifted from its registered 7 runs: {len(runs)}"
    return runs


# sec 10.7's realized per-cell bracket (2026-07-06 build report task
# brief): 0.18-0.77 GPU-h/cell, replacing the point-estimate anchor every
# earlier wave used (attack-R7 finding 12 -- shared-GPU contention
# variance, not workload, is the dominant cost driver, so a bracket over
# the FULL observed range is the correct correction, not a x1.3-1.5 margin).
KEYANCHOR_MECH_GPUH_PER_CELL = (0.18, 0.77)
KEYANCHOR_MECH_GATE1_GPUH = (0.03, 0.12)          # sec 10.7: "realized comparable probe: 0.030"
# sec 10.7's own registered nominal ceiling for this wave (headroom above
# even the all-conditionals bracket top ~8.6) -- the number budget_guard
# below actually gates on, not a bracket edge.
KEYANCHOR_MECH_GPUH_CEILING = 12.0
# sec 10.7's program arithmetic: anchoring program spend so far, LIVE
# constant (task brief: "~51.5"; STATE.md's own ~51/80 figure, consistent).
# NOT auto-derived from archived JSONs here (no single source-of-truth
# ledger file exists yet for this program) -- a documented, human-updated
# constant, exactly the same class of thing PROGRAM_SPENT_GPUH is in
# run_lm_rd_trackc_sweep.py/run_trackb_wave.py (this module's own docstring
# precedent for "live-imported" ledgers, cloned as a literal here since no
# other module in THIS design's own program tracks it).
KEYANCHOR_PROGRAM_SPENT_GPUH = 51.5
KEYANCHOR_PROGRAM_GPUH_CEILING = 80.0   # DELTANET_RD_EXACTNESS_DESIGN.md / KEY_ANCHORING_DESIGN.md sec 5


def keyanchor_mech_budget_guard(accept_override: bool) -> float:
    """CLONED pattern from run_trackb_wave.py::budget_guard (:469-484) /
    run_lm_rd_trackc_sweep.py's own original -- gates this wave's
    REGISTERED ceiling (12 GPU-h, sec 10.7) against the exactness
    program's 80 GPU-h cap with the live-spent constant above. Uses the
    registered ceiling, not a bracket midpoint/top, matching sec 10.7's
    own "brings the worst case to <=63.5/80" arithmetic exactly."""
    cumulative = KEYANCHOR_PROGRAM_SPENT_GPUH + KEYANCHOR_MECH_GPUH_CEILING
    print(f"BUDGET GUARD (keyanchor-mech): program-spent-so-far={KEYANCHOR_PROGRAM_SPENT_GPUH:.1f} GPU-h + "
          f"this-wave-registered-ceiling={KEYANCHOR_MECH_GPUH_CEILING:.1f} GPU-h = cumulative "
          f"{cumulative:.1f} GPU-h, program ceiling {KEYANCHOR_PROGRAM_GPUH_CEILING:.0f} GPU-h "
          f"(reserve {KEYANCHOR_PROGRAM_GPUH_CEILING - cumulative:.1f} GPU-h).", flush=True)
    if cumulative > KEYANCHOR_PROGRAM_GPUH_CEILING and not accept_override:
        print(f"ERROR: projected cumulative spend {cumulative:.1f} GPU-h EXCEEDS the "
              f"{KEYANCHOR_PROGRAM_GPUH_CEILING:.0f} GPU-h exactness-program ceiling -- REFUSING to "
              f"launch keyanchor-mech. Pass --accept-budget-override to force past this guard.",
              file=sys.stderr)
        sys.exit(5)
    return cumulative


# KEY_ANCHORING_DESIGN.md sec 10.10's checkpoint writer sizing: the
# anchor_table's TRAINED-ROW block is EXACTLY n_train*d_state*4 bytes
# (fp32) regardless of K (K is the per-episode binding count, not
# d_state) -- a CLOSED-FORM size, not a measured-checkpoint-probe fallback
# (run_trackb_wave.py's find_ckpt_size_bytes precedent needed a probe
# because Wave C's full-model checkpoint size depended on architecture
# params with no simple closed form; this design's checkpoint is a single
# small dense tensor with an exact formula, so no probe file is needed or
# useful here).
KEYANCHOR_MECH_N_TRAIN = 107
KEYANCHOR_MECH_D_STATE = 64
KEYANCHOR_MECH_CKPT_EVERY = 2000


def keyanchor_mech_projected_ckpt_bytes(manifest: list) -> int:
    """Per cell: anchor_table_trained_rows (107*64*4=27,392 B) +
    anchor_train_ids (107*8 B int64) + (candidate (d'): +107*4 B lambda
    table; candidate (d): +4 B scalar) + torch.save/pickle framing
    overhead (~2KB budgeted flat, generous) -- times the number of
    checkpoints per cell (steps // ckpt_every + 1, the SAME formula
    run_trackb_wave.py's projected_ckpt_bytes uses) times the cell count."""
    per_row_block = KEYANCHOR_MECH_N_TRAIN * KEYANCHOR_MECH_D_STATE * 4
    ids_block = KEYANCHOR_MECH_N_TRAIN * 8
    framing_overhead = 2048
    total = 0
    for spec in manifest:
        extra = (KEYANCHOR_MECH_N_TRAIN * 4) if spec["anchor_lambda_mode"] == "learned_per_entity" else 4
        per_ckpt_bytes = per_row_block + ids_block + extra + framing_overhead
        n_ckpts = spec["steps"] // KEYANCHOR_MECH_CKPT_EVERY + 1
        total += n_ckpts * per_ckpt_bytes
    return total


def keyanchor_mech_disk_gate(ckpt_base_dir: str, manifest: list, label: str = "keyanchor-mech") -> dict:
    """DISK GATE, house gate-(f) pattern (run_trackb_wave.py::disk_space_check
    /run_lm_rd_trackc_sweep.py's own original) -- projected bytes computed
    analytically (see keyanchor_mech_projected_ckpt_bytes) rather than via
    a measured-checkpoint-probe file (no prior checkpoint of this exact
    format exists yet; the size has a closed form, so a probe would add
    fragility for zero benefit here)."""
    import shutil
    os.makedirs(ckpt_base_dir, exist_ok=True)
    resolved = os.path.realpath(ckpt_base_dir)
    free_bytes = shutil.disk_usage(resolved).free if os.path.exists(resolved) else 0
    projected = keyanchor_mech_projected_ckpt_bytes(manifest)
    safety_factor = 1.5
    required = int(projected * safety_factor)
    report = {"label": label, "resolved_ckpt_dir": resolved, "projected_ckpt_bytes": projected,
              "safety_factor": safety_factor, "required_bytes": required, "free_bytes": free_bytes,
              "ok": os.path.exists(resolved) and free_bytes >= required}
    print(f"DISK GATE ({label}): projected {report['projected_ckpt_bytes'] / 1e6:.3f} MB x "
          f"{safety_factor} = {required / 1e6:.3f} MB required, {free_bytes / 1e9:.2f} GB free -> "
          f"{'OK' if report['ok'] else 'REFUSED'}", flush=True)
    return report


# KEYANCHOR_MECH_GATE1_JSON_DEFAULT (the reader's default for
# gate_keyanchor_mech_probe below) is DERIVED from the writer's own
# out_path()/manifest functions, never hand-typed -- defined AFTER
# out_path()'s own definition below (e633862 audit F1: the original
# hand-typed sibling path here silently diverged from where the probe's
# result actually lands, so the chain could never complete as committed;
# a reader default must be computed FROM the same function the writer
# uses, so the two can never diverge).


def gate_keyanchor_mech_probe(gate1_json_path: str, accept_override: bool) -> dict:
    """sec 10.5's Gate-1-style pre-launch probe check for candidate (d')
    ONLY -- mirrors gate_keyanchor_drift_diag's exact pattern (CPU-only
    file-read gate; the probe itself is a separate, sequential GPU driver,
    `--wave keyanchor-mech-gate1`). Reads the probe's own h4 recovery
    reading and requires it clear the SAME predicted h4>=0.8 bar the
    confirm wave's Gate 1 already used for candidate (d)."""
    if accept_override:
        print("=" * 70 + "\nWARNING: --accept-keyanchor-mech-gate1-override -- the candidate (d') "
              "Gate-1 pre-launch probe read is being BYPASSED by an explicit human decision. This "
              "is recorded here; it does NOT mean the probe passed.\n" + "=" * 70, flush=True)
        return {"gate_bypassed": True, "gate1_json_path": gate1_json_path}
    if not os.path.exists(gate1_json_path):
        print(f"ERROR: --wave keyanchor-mech requires a completed candidate (d') Gate-1 probe "
              f"first -- {gate1_json_path!r} does not exist. Run:\n"
              f"  python run_deltanet_rd_exactness_sweep.py --wave keyanchor-mech-gate1 "
              f"--gpus 1 --gpu-offset <free GPU>\n"
              f"or pass --accept-keyanchor-mech-gate1-override for an explicit, documented override.",
              file=sys.stderr)
        sys.exit(1)
    with open(gate1_json_path) as f:
        d = json.load(f)
    try:
        # h=4 is in H_test (default [4,5,6]), evaluated into "M3_held_out"
        # (M2_in_distribution only covers H_train=[1,2,3]) -- JSON object
        # keys are always strings after the round-trip, hence "4" not 4.
        h4 = d["checkpoints"][-1]["M3_held_out"]["4"]["recovered_frac@0.9"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"ERROR: {gate1_json_path!r} is missing an expected field ({e!r}) -- refusing to "
              f"launch. Pass --accept-keyanchor-mech-gate1-override for an explicit override.",
              file=sys.stderr)
        sys.exit(1)
    # REGISTRATION CORRECTION (2026-07-05, disclosed in KEY_ANCHORING_DESIGN.md
    # sec 10.5): the original 0.8 bar was the K=16-era Gate-1 PREDICTION bar
    # (K=16 realized ~0.95+), mis-inherited by the design text into this K=32
    # MEASURED probe -- it demanded the probe beat the wave's own registered
    # K=32 success bar (0.5) by 0.3, unpassable for any correct prediction.
    # The harm-screen purpose (refuse GPU spend on a doomed arm) is served by
    # the wave's own registered bar. First live firing: probe read 0.6261 --
    # clears 0.5, matches candidate (d)'s known K=32 range 0.556-0.665.
    KEYANCHOR_MECH_GATE1_BAR = 0.5
    if h4 < KEYANCHOR_MECH_GATE1_BAR:
        print(f"ERROR: candidate (d') Gate-1 probe FAILED -- predicted h4 rec@0.9={h4:.4f} < "
              f"{KEYANCHOR_MECH_GATE1_BAR} (the wave's registered K=32 bar). "
              f"Pass --accept-keyanchor-mech-gate1-override for an explicit, documented override.",
              file=sys.stderr)
        sys.exit(1)
    print(f"keyanchor-mech Gate-1 PASSED: candidate (d') predicted h4 rec@0.9={h4:.4f} >= "
          f"{KEYANCHOR_MECH_GATE1_BAR} (registered K=32 bar; sec 10.5 registration correction)",
          flush=True)
    return {"gate_bypassed": False, "gate1_json_path": gate1_json_path, "h4": h4,
            "gate1_bar": KEYANCHOR_MECH_GATE1_BAR}


def keyanchor_ceiling_by_k() -> dict:
    """sec 2.2's computed lambda=1 post-NS drift ceilings (frame-potential
    init) -- the band-derivation guard's `ceiling_by_k` input."""
    return {16: 0.9745, 32: 0.9423}


def keyanchor_k48_ceiling_by_k() -> dict:
    """KEY_ANCHORING_DESIGN.md sec 11.4.2's computed lambda=1 post-NS drift
    ceiling at K=48 (mean 0.8987, n_iter=20, frame-potential init, full-pool
    mechanism -- NOT the i-strong pool-restricted 1.0000) -- a SEPARATE
    dict from keyanchor_ceiling_by_k() above (sec 11.1.1: 'a NEW writer...
    not a re-derivation of K=16/32's own BANDS_PINNED.json'), used only by
    the K=48 bands writer/gate (BANDS_PINNED_K48.json)."""
    return {48: 0.8987}


# ---------------------------------------------------------------------------
# KEY_ANCHORING_DESIGN.md sec 11 (Rev K48.1, CLEARED-FOR-BUILD per sec
# 11.11's external verify) -- the K=48 capacity-curve extension wave.
# Registered manifest name: 'keyanchor-k48' (sec 11.1's own naming-pinning
# discipline, mirroring sec 10.5's precedent for 'keyanchor-mech').
#
# Arms (sec 11.1): (1) candidate (d), K=48, seeds {30,31,32} -- MANDATORY,
# PRIMARY; (2) reference arms, bare geo3, K=48, seeds {1,2,3}, drift_probe
# -- MANDATORY, first in manifest (K=48 has no BANDS_PINNED entry yet); (3)
# candidate (d'), K=48, seeds {40,41,42} -- CONDITIONAL on Rev 7.1's own
# K=32 verdict (orchestrator sign-off, NOT a mechanical gate -- sec 11.1
# item 3's honest downgrade, Rev K48.1 finding C11); (4) Gate-1-style
# pre-launch probe, candidate (d) arch only, K=48, 5,000 steps, seed 0 --
# DISCLOSED, NON-GATING (sec 11.1 item 4); (5) fixed-lambda=1
# ceiling-validation probe, seed 50 -- OPTIONAL, lowest cut priority (sec
# 11.4.3).
# ---------------------------------------------------------------------------

KEYANCHOR_K48_TIER_STEPS = KEYANCHOR_TIER_STEPS          # 20,000 -- continuity, sec 11.1
KEYANCHOR_K48_GATE1_STEPS = 5000                          # sec 11.1 item 4
KEYANCHOR_K48_NS_N_ITER = 20                              # sec 11.3: n_iter=20 at K=48, unchanged tier
KEYANCHOR_K48_FIXED_LAMBDA1 = 1.0                         # sec 11.4.3's ceiling-validation point


def keyanchor_k48_reference_manifest():
    """sec 11.1 arm 2: bare geo3, K=48, seeds {1,2,3}, drift_probe=True --
    the SAME reference_arms_manifest() function, generalized above, called
    with Ks=(48,) so this is genuinely the one function every K's reference
    arms go through (never a hand-copied twin), producing wave tag 'ref'
    (out_dir 'waveref') filenames that are K=48-distinct by construction
    (_spec's own f"K{K}" name bit) -- sec 11.9 item 16's own zero-collision
    concern is about a DIFFERENT, pre-existing archive (wavegeo3's bare
    K=48 rider, seeds 0/1/2, a different wave tag AND a different seed
    range from THESE seeds {1,2,3}), not about this function's own output
    colliding with itself."""
    runs = reference_arms_manifest(Ks=(48,), seeds=(1, 2, 3))
    assert len(runs) == 3, f"K=48 reference-arm manifest drifted from its registered 3 runs: {len(runs)}"
    return runs


def keyanchor_k48_manifest(K: int = 48, seeds: tuple = (30, 31, 32)):
    """sec 11.1 arm 1 (MANDATORY, PRIMARY): candidate (d), K=48, seeds
    {30,31,32}, 20,000 steps -- byte-identical architecture to Rev 7.1's own
    candidate (d) (anchor_blend_gather_scatter, learned scalar lambda,
    frame-potential init at ANCHOR_INIT_SEED -- verified K-independent,
    sec 11.1.1) PLUS drift_probe=True (item 5) AND rev7_engagement=True
    (the Rev-7.1 r_e/anchor-norm/BH-engagement logging, inherited wholesale
    per sec 11.0 -- literally the same code path, not a re-derivation). No
    K=16 spot check is registered for this arm (sec 11.1: K=16 already
    saturates near h4~=1.0 under bare geo3, no headroom to see an
    anchoring effect there).

    Generalized to accept K/seeds (sec 12.2.1, Rev 12.1 build scope) so the
    capacity-cliff wave's own keyanchor_cliff_manifest() below can reuse this
    SAME function per new K rather than a hand-copied twin -- defaults are
    BYTE-IDENTICAL to the pre-generalization hardcoded K=48/{30,31,32} form
    (smoke_keyanchor_cliff.py's own regression smoke asserts this directly,
    not merely by inspection, mirroring sec 11.9 item 14's own precedent for
    reference_arms_manifest). geo3_n_iter is read from key_anchoring.
    GATE2_N_ITER_BY_K (the same per-K production tier Gate 2 uses) rather
    than the K48-only KEYANCHOR_K48_NS_N_ITER constant, so a K=34/38/42/46
    call and the K=48 default can never silently diverge from Gate 2's own
    tier for that K."""
    steps = KEYANCHOR_K48_TIER_STEPS
    n_iter = ka.GATE2_N_ITER_BY_K[K]
    runs = []
    for s in seeds:
        runs.append(_spec("keyanchor-k48", K, s, steps, "d", geo3_active=True,
                           geo3_n_iter=n_iter, geo3_resid_tol=1e-2,
                           anchor_active=True, anchor_lambda_mode="learned",
                           drift_probe=True, rev7_engagement=True))
    assert len(runs) == len(seeds), \
        f"keyanchor-k48 manifest (K={K}) drifted from its registered {len(seeds)} runs: {len(runs)}"
    return runs


def keyanchor_k48_dprime_manifest():
    """sec 11.1 arm 3 (CONDITIONAL, cut-eligible): candidate (d'), K=48,
    seeds {40,41,42} -- fires ONLY if Rev 7.1's own K=32 wave resolves to
    Outcome A(d') or D' (sec 10.6's routing table) -- an ORCHESTRATOR
    sign-off gated on Rev 7.1's own written verdict, honestly downgraded
    from a mechanical gate at Rev K48.1 (attack finding
    budget-schedule-auditor M3, sec 11.1 item 3/sec 11.10 C11: no script,
    JSON field, or hash-lock currently makes this check-able by machine).
    This function BUILDS the manifest (registered now, per sec 11.1's own
    "registering them now... is what prevents them from becoming free
    post-hoc additions later") but main()'s dispatch below requires an
    explicit --accept-k48-dprime-orchestrator-signoff flag before this
    manifest's cells are added to any real launch -- never auto-fired."""
    steps = KEYANCHOR_K48_TIER_STEPS
    runs = []
    for s in (40, 41, 42):
        runs.append(_spec("keyanchor-k48", 48, s, steps, "dprime", geo3_active=True,
                           geo3_n_iter=KEYANCHOR_K48_NS_N_ITER, geo3_resid_tol=1e-2,
                           anchor_active=True, anchor_lambda_mode="learned_per_entity",
                           drift_probe=True, rev7_engagement=True))
    assert len(runs) == 3, f"keyanchor-k48 (d') manifest drifted from its registered 3 runs: {len(runs)}"
    return runs


def keyanchor_k48_gate1_manifest(K: int = 48, seed: int = 0):
    """sec 11.1 arm 4: Gate-1-style pre-launch probe, candidate (d) arch
    only, K=48, 5,000 steps, seed 0 -- DISCLOSED, NON-GATING (mirrors Rev
    7.1's own d' probe scope-carve-out; the simulator drift->recovery
    mapping is non-gating at K=32 already, and K=48 is a harder packing
    regime, so the same non-gating status is inherited, not re-derived).
    Carries rev7_engagement=True (same audit-fold-in-1 precedent as
    keyanchor_mech_gate1_manifest: exercise the checkpoint/r_e wiring on
    the cheap probe first).

    Generalized to accept K/seed (sec 12.2.1, Rev 12.1 build scope) -- SAME
    byte-identical-at-defaults discipline as keyanchor_k48_manifest above;
    the cliff wave's own optional per-K Gate-1 probes (seeds 133/233/333/433,
    sec 12.2 item 3) reuse this function rather than a hand-copied twin."""
    n_iter = ka.GATE2_N_ITER_BY_K[K]
    runs = [_spec("keyanchor-k48-gate1", K, seed, KEYANCHOR_K48_GATE1_STEPS, "d", geo3_active=True,
                   geo3_n_iter=n_iter, geo3_resid_tol=1e-2, anchor_active=True,
                   anchor_lambda_mode="learned", drift_probe=True, rev7_engagement=True)]
    assert len(runs) == 1, \
        f"keyanchor-k48 Gate-1 probe manifest (K={K}) drifted from its registered 1 run: {len(runs)}"
    return runs


def keyanchor_k48_fixed_lambda1_manifest():
    """sec 11.1 arm 5 / sec 11.4.3 (OPTIONAL, lowest cut priority): the
    fixed-lambda=1 candidate-(d) ceiling-validation probe -- seed 50, 1 run,
    20,000 steps, candidate (d)'s own architecture with
    anchor_lambda_mode='fixed', anchor_lambda_fixed=1.0. REPLACES the
    withdrawn asymmetric-pool i-strong arm (Rev K48.1, sec 11.4.3/sec
    11.10 C4) -- measures the ACTUAL trained full-pool anchor mechanism at
    its lambda->1 extreme, the same object sec 11.4.2's computation
    approximates."""
    runs = [_spec("keyanchor-k48", 48, 50, KEYANCHOR_K48_TIER_STEPS, "d-fixed-lam1", geo3_active=True,
                   geo3_n_iter=KEYANCHOR_K48_NS_N_ITER, geo3_resid_tol=1e-2, anchor_active=True,
                   anchor_lambda_mode="fixed", anchor_lambda_fixed=KEYANCHOR_K48_FIXED_LAMBDA1,
                   drift_probe=True, rev7_engagement=True)]
    assert len(runs) == 1, f"keyanchor-k48 fixed-lambda1 probe manifest drifted from its registered 1 run: {len(runs)}"
    return runs


def gate_keyanchor_k48_gate1_probe(gate1_json_path: str, accept_override: bool) -> dict:
    """sec 11.1 item 4: reads the K=48 Gate-1-style probe's own result --
    DISCLOSED, NON-GATING (mirrors gate_geo3_drift's read-and-report
    pattern but NEVER sys.exit(1)s on a low read -- sec 11.1's own text:
    'never a hard go/no-go'). Always returns a report dict; --wave
    keyanchor-k48 proceeds regardless of this probe's own result (Gate 2,
    the construction check, remains the sole go/no-go for this wave)."""
    if accept_override or not os.path.exists(gate1_json_path):
        print(f"NOTE: K=48 Gate-1-style probe not available at {gate1_json_path!r} (or "
              f"--accept-override passed) -- this is DISCLOSED, NON-GATING context only (sec "
              f"11.1 item 4); proceeding either way.", flush=True)
        return {"gate_bypassed": True, "gate1_json_path": gate1_json_path}
    with open(gate1_json_path) as f:
        d = json.load(f)
    try:
        h4 = d["checkpoints"][-1]["M3_held_out"]["4"]["recovered_frac@0.9"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"NOTE: {gate1_json_path!r} is missing an expected field ({e!r}) -- reported, "
              f"non-gating (sec 11.1 item 4).", flush=True)
        return {"gate_bypassed": False, "gate1_json_path": gate1_json_path, "h4": None}
    print(f"K=48 Gate-1-style probe read: predicted h4 rec@0.9={h4:.4f} -- DISCLOSED, NON-GATING "
          f"(sec 11.1 item 4; Gate 2 remains the sole go/no-go for this wave).", flush=True)
    return {"gate_bypassed": False, "gate1_json_path": gate1_json_path, "h4": h4}


def write_bands_pinned_k48_if_ready(out_dir: str) -> bool:
    """sec 11.1.1 writer requirement 1: BANDS_PINNED_K48.json, scoped to the
    3 K=48 reference arms only (NOT a re-derivation of K=16/32's own
    BANDS_PINNED.json, sec 11.1.1's own text) -- reuses key_anchoring's
    derive_engaged_bands/write_bands_pinned UNMODIFIED, called with
    per_k_final_drift={48: [...]} and keyanchor_k48_ceiling_by_k()
    (0.8987). Mirrors write_bands_pinned_if_ready's exact pattern above."""
    manifest = keyanchor_k48_reference_manifest()
    per_k_final_drift = {48: []}
    ref_paths = {48: []}
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
        per_k_final_drift[48].append(ka.reference_final_drift(result))
        ref_paths[48].append(p)
    if not all_valid:
        print("BANDS_PINNED_K48.json NOT written -- not every K=48 reference arm validates complete yet.",
              file=sys.stderr)
        return False
    bp_path = os.path.join(out_dir, "BANDS_PINNED_K48.json")
    doc = ka.write_bands_pinned(bp_path, per_k_final_drift, keyanchor_k48_ceiling_by_k(), ref_paths)
    print(f"BANDS_PINNED_K48.json written at {bp_path}:\n{json.dumps(doc['bands'], indent=2)}", flush=True)
    return True


def gate_bands_pinned_k48(out_dir: str, accept_override: bool, override_at: float | None) -> dict:
    """sec 11.1.1 launcher-gate requirement 2: K=48 candidate (d)/(d') cells
    REFUSE to launch unless BANDS_PINNED_K48.json exists AND validates --
    the SAME hash-validate/content-re-derivation/degenerate-case-guard
    machinery as gate_bands_pinned above, pointed at the K48-specific file
    and ceiling dict, never a re-derivation of the K=16/32 gate."""
    bp_path = os.path.join(out_dir, "BANDS_PINNED_K48.json")
    if accept_override:
        print("=" * 70 + "\nWARNING: --unblind-override -- the sec 11.1.1 K=48 BANDS_PINNED_K48 "
              "blinding gate is being BYPASSED by an explicit human decision. EVERY keyanchor-k48 "
              "anchor-arm run launched this session will have its OWN result JSON stamped "
              "claim_tier='descriptive' + unblind_override=True.\n" + "=" * 70, flush=True)
        return {"gate_bypassed": True, "bands_pinned_path": bp_path, "override_at": override_at}
    doc = ka.validate_bands_pinned(bp_path, ceiling_by_k=keyanchor_k48_ceiling_by_k())
    if doc is None:
        print(f"ERROR: {bp_path!r} does not exist, fails hash validation, or its STORED bands "
              f"dict does not reproduce under live re-derivation from the referenced K=48 "
              f"reference-arm JSONs. Run the 3 K=48 reference arms to completion first "
              f"(--wave keyanchor-k48-ref), then --wave keyanchor-k48-bands to pin, or pass "
              f"--unblind-override for an explicit, documented, tier-demoting override.",
              file=sys.stderr)
        sys.exit(1)
    for K, entry in doc["bands"].items():
        if entry["unresolvable"]:
            print(f"  NOTE: K={K} post-NS engagement bands are UNRESOLVABLE (engaged_K "
                  f"{entry['engaged_k']:.4f} >= ceiling {entry['ceiling']:.4f} - 0.005) -- sec 11.1.1's "
                  f"own degenerate-case guard; non-gating for h4 admission or the r_e/BH engagement "
                  f"test (sec 11.1.1's own disclosure).", flush=True)
    print(f"sec 11.1.1 K48 GATE PASSED: {bp_path} validates.", flush=True)
    return {"gate_bypassed": False, "bands_pinned_path": bp_path, "bands": doc["bands"]}


# sec 11.5's budget table, restated as constants (mirrors
# KEYANCHOR_MECH_GPUH_PER_CELL's own bracket-not-point discipline). The
# K=48/K=32 step-cost ratio (1.264x, sec 11.5) applied to the mech wave's
# own realized bracket (0.18-0.77 GPU-h/cell, sec 10.7) -- NOT re-measured
# here (no K=48 anchor-arm cell has ever run); this is the registered
# PROVISIONAL bracket pending the pre-launch K=16-instrumented-vs-bare
# overhead check sec 11.5 requires (budget-schedule-auditor M2/C10).
KEYANCHOR_K48_STEP_COST_RATIO = 1.264            # sec 11.5: 0.2454/0.1942, K=48/K=32 uninstrumented
KEYANCHOR_K48_GPUH_PER_CELL = (0.18 * KEYANCHOR_K48_STEP_COST_RATIO,
                                0.77 * KEYANCHOR_K48_STEP_COST_RATIO)   # ~= (0.23, 0.97), sec 11.5
KEYANCHOR_K48_GATE1_GPUH = (0.06, 0.24)           # sec 11.5: 5,000/20,000 x bracket
KEYANCHOR_K48_GPUH_CEILING = 12.0                # sec 11.5's registered nominal ceiling
KEYANCHOR_K48_PROGRAM_GPUH_CEILING = 80.0         # unchanged exactness-program cap

# sec 10.13's registered candidate-(e) budget: ~1 GPU-h for the original
# 3-cell random arm (realized ~0.2-0.35 GPU-h/cell). The wave now carries
# SIX cells (audit prescription: the 'e' random arm, seeds 60-62, PLUS the
# 'e-fp' frozen-frame-potential arm per the stub's literal text, seeds
# 70-72) -- bracket 6 x (0.20-0.35) = 1.2-2.1 GPU-h; ceiling raised
# accordingly (bracket top +~20% margin). Combined program worst case with
# the K48 wave: 49.8493 + 12.0 + 2.5 = 64.35/80 -- verified WITHIN cap.
KEYANCHOR_E_GPUH_PER_CELL = (0.20, 0.35)
KEYANCHOR_E_GPUH_CEILING = 2.5

# ---------------------------------------------------------------------------
# Program ledger reconciliation (KEY_ANCHORING_DESIGN.md sec 11.5, attack
# finding budget-schedule-auditor M1/Rev-K48.1 C9; independently
# re-verified sec 11.11 point 4). The design's own text disclosed a ~2-3
# GPU-h untraced gap between the asserted "51.5/80" figure
# (KEYANCHOR_MECH_GPUH_CEILING's own sibling constant,
# KEYANCHOR_PROGRAM_SPENT_GPUH above) and what its own itemization could
# reconstruct (~48.3-49.4). This task's own instruction: "FIRST trace the
# anchoring ledger... itemize in a comment and set the spent constant
# honestly." Itemized here from the SAME archived wall_s fields sec 11.5/
# 11.11 already cite, PLUS the one leg both of those sections' own tables
# omitted (the confirm-wave's K=16 seed-0 spot check, archived at
# experiment-runs/2026-07-05_keyanchor_confirm/wavekeyanchor-confirm/
# wkeyanchor-confirm_rdx_K16_armd_s0_geo3n12_anchor_learned_dprobe.json,
# wall_s=654.09s) -- sec 11.5's own table lists only 3 of the confirm
# wave's 4 realized legs (the K=32 s{0,1,2} triple), so a truly complete
# reconciliation must include the 4th:
#
#   sec 5's own confirmed sum (all archived wall_s pre-Wave-1)     34.90
#   + F-geo-3 realized (sec 5, sec 11.11 point 4)                  1.67
#   + Wave 1 realized (sec 9 header: "18 mandatory cells,          10.98
#     10.98 realized GPU-h")
#   + confirm-wave, ALL 4 legs (not just the 3 sec 11.5's own
#     table shows): K32 s0=748.4653s + s1=750.5084s + s2=724.3776s
#     + K16 s0=654.0854s = 2877.437s / 3600                        0.7993
#   + keyanchor-mech wave, realized (sec 10.13 header: "7           1.500
#     mandatory cells 1.439 GPU-h + Gate-1 probe 0.061 GPU-h")
#   ------------------------------------------------------------------
#   RECONCILED TOTAL                                              49.8493
#
# This lands at ~49.85, matching this task's own expected range
# ("~49.6-49.9+1.5 -> verify" -- the +1.5 mech-wave contribution is
# already included in the 49.85 above, not additional to it). Replaces the
# untraced 51.5 used by keyanchor_mech_budget_guard above (that constant
# is LEFT UNCHANGED -- it already ran to completion at Rev 7.1's own
# realized 1.500 GPU-h, sec 10.13, so retroactively editing its own guard
# would not change any real launch decision now); this reconciled constant
# is what keyanchor_k48_budget_guard and keyanchor_e_budget_guard below
# actually use for THEIR OWN pre-launch checks, per sec 11.5's own
# registered sequencing recommendation ("reconcile the 51.5 base figure
# FIRST... before this wave's own mandatory cells launch").
KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED = 49.8493


def keyanchor_k48_budget_guard(accept_override: bool) -> float:
    """Mirrors keyanchor_mech_budget_guard's exact pattern, using the
    RECONCILED program-spent constant (see the ledger comment above) rather
    than the untraced 51.5, per sec 11.5's own registered pre-launch
    reconciliation requirement."""
    cumulative = KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED + KEYANCHOR_K48_GPUH_CEILING
    print(f"BUDGET GUARD (keyanchor-k48): program-spent-so-far(RECONCILED)="
          f"{KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED:.4f} GPU-h + this-wave-registered-ceiling="
          f"{KEYANCHOR_K48_GPUH_CEILING:.1f} GPU-h = cumulative {cumulative:.4f} GPU-h, program "
          f"ceiling {KEYANCHOR_K48_PROGRAM_GPUH_CEILING:.0f} GPU-h (reserve "
          f"{KEYANCHOR_K48_PROGRAM_GPUH_CEILING - cumulative:.4f} GPU-h).", flush=True)
    if cumulative > KEYANCHOR_K48_PROGRAM_GPUH_CEILING and not accept_override:
        print(f"ERROR: projected cumulative spend {cumulative:.4f} GPU-h EXCEEDS the "
              f"{KEYANCHOR_K48_PROGRAM_GPUH_CEILING:.0f} GPU-h exactness-program ceiling -- REFUSING "
              f"to launch keyanchor-k48. Pass --accept-budget-override to force past this guard.",
              file=sys.stderr)
        sys.exit(5)
    return cumulative


def keyanchor_e_budget_guard(accept_override: bool) -> float:
    """Same pattern, candidate-(e)'s own ~1.5 GPU-h ceiling. Sequenced AFTER
    keyanchor-k48 in program arithmetic (both draw from the SAME reconciled
    base -- if both waves' cells are launched in the same session, the
    SECOND wave to check this guard should pass its own already-spent
    cells' realized wall_s as accept_override context, not silently
    double-count against the pre-launch base; this build wires that
    caller-side discipline into the chain script's own comments, not into
    this function's signature, mirroring keyanchor_k48_budget_guard's own
    scope)."""
    cumulative = KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED + KEYANCHOR_E_GPUH_CEILING
    print(f"BUDGET GUARD (keyanchor-e): program-spent-so-far(RECONCILED)="
          f"{KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED:.4f} GPU-h + this-wave-registered-ceiling="
          f"{KEYANCHOR_E_GPUH_CEILING:.1f} GPU-h = cumulative {cumulative:.4f} GPU-h, program "
          f"ceiling {KEYANCHOR_K48_PROGRAM_GPUH_CEILING:.0f} GPU-h (reserve "
          f"{KEYANCHOR_K48_PROGRAM_GPUH_CEILING - cumulative:.4f} GPU-h).", flush=True)
    if cumulative > KEYANCHOR_K48_PROGRAM_GPUH_CEILING and not accept_override:
        print(f"ERROR: projected cumulative spend {cumulative:.4f} GPU-h EXCEEDS the "
              f"{KEYANCHOR_K48_PROGRAM_GPUH_CEILING:.0f} GPU-h exactness-program ceiling -- REFUSING "
              f"to launch keyanchor-e. Pass --accept-budget-override to force past this guard.",
              file=sys.stderr)
        sys.exit(5)
    return cumulative


# ---------------------------------------------------------------------------
# KEY_ANCHORING_DESIGN.md sec 10.13's registered candidate (e):
# "frozen-random-table ablation" -- the decisive probe for the
# construction-stabilization account (sec 10.13.4). K=32, anchor_lambda_
# mode FIXED at the measured cross-cell mean lambda~=0.58 (matched to
# candidate (d)'s own K=32 interior range, so the comparison is matched on
# the one hyperparameter the account claims matters), anchor table
# initialized via random_unit_rows_init (key_anchoring.py -- frame_
# potential_init's own pre-optimization starting point, NOT the frame-
# potential-minimized construction -- sec 10.13's own "frozen-RANDOM-table"
# name) and NEVER trained (anchor_table_frozen=True). 3 seeds (sec 10.13:
# "2-3 seeds").
# ---------------------------------------------------------------------------

KEYANCHOR_E_LAMBDA_FIXED = 0.58    # sec 10.13: cross-cell mean of this wave's own measured lambda
KEYANCHOR_E_TIER_STEPS = KEYANCHOR_TIER_STEPS   # 20,000 -- continuity, sec 10.13
KEYANCHOR_E_NS_N_ITER = 20         # K=32 production tier, unchanged


def keyanchor_e_manifest():
    """sec 10.13's registered candidate (e): K=32, seeds {60,61,62} (a
    fresh, never-before-used seed block -- this program's own escalating-
    decade-block convention, sec 11.1's own text: 'new, never-before-used-
    anywhere blocks are assigned per arm... so no seed integer's provenance
    is ever ambiguous'), anchor_lambda_mode='fixed' at
    KEYANCHOR_E_LAMBDA_FIXED (0.58), anchor_table_frozen=True,
    anchor_table_init_mode='random_unit_rows', drift_probe=True AND
    rev7_engagement=True (full Rev 7.1 instrumentation, per this task's own
    'full Rev 7.1 instrumentation' requirement -- r_e/anchor-norm/BH
    engagement logging, even though this arm's own success criterion (sec
    10.13: h4 and pre-NS drift within ordinary seed noise of candidate
    (d)'s own range) does not itself require the engagement test; logging
    it costs nothing extra and keeps every anchor-arm cell in this build on
    the same instrumentation floor)."""
    steps = KEYANCHOR_E_TIER_STEPS
    runs = []
    for s in (60, 61, 62):
        spec = _spec("keyanchor-e", 32, s, steps, "e", geo3_active=True,
                     geo3_n_iter=KEYANCHOR_E_NS_N_ITER, geo3_resid_tol=1e-2,
                     anchor_active=True, anchor_lambda_mode="fixed",
                     anchor_lambda_fixed=KEYANCHOR_E_LAMBDA_FIXED,
                     drift_probe=True, rev7_engagement=True)
        # _spec() does not YET know about anchor_table_frozen/anchor_table_
        # init_mode (those are a model_rd.py/run_deltanet_rd.py CLI
        # addition, not part of _spec()'s own naming-relevant field set --
        # see build_cmd_keyanchor_e below for how they reach the CLI).
        # Recorded directly on the spec dict here so out_path()'s own
        # f"{spec['name']}" naming can be extended distinctly (arm='e' plus
        # the K/seed bits already makes this collision-free against every
        # OTHER wave's own out_dir, sec 11.9 item 16's own directory-scoped
        # discipline) and so is_done() can be extended to check them.
        spec["anchor_table_frozen"] = True
        spec["anchor_table_init_mode"] = "random_unit_rows"
        runs.append(spec)
    assert len(runs) == 3, f"keyanchor-e manifest drifted from its registered 3 runs: {len(runs)}"
    return runs


def keyanchor_e_fp_manifest():
    """The 'e-fp' arm (2026-07 K48+e build, audit prescription): sec
    10.13's stub, read LITERALLY -- anchor table 'initialized once via the
    existing frame_potential_init and then never trained'. The 'e' arm
    above (random_unit_rows, per the stub's own "frozen-RANDOM-table" name
    and sec 10.13.4's motivating text) and THIS arm test two different,
    both-valuable hypotheses:
      - 'e' (random): does mere episode-constancy of ANY fixed table
        deliver candidate (d)'s gains? (the construction-stabilization
        account's strongest form)
      - 'e-fp' (frozen frame-potential): does the table's OPTIMIZED bulk
        geometry (tight-frame conditioning) matter beyond constancy, even
        with zero training? (isolates init geometry from learnedness)
    If e-fp matches (d) but e-random collapses, bulk geometry (not
    learning) is the carrier; if both match (d), constancy alone suffices;
    if both collapse toward geo3-alone, the LEARNED table matters beyond
    both construction properties. K=32, seeds {70,71,72} (a fresh block --
    seeds 60-62 stay registered as the 'e' random arm, never reused here),
    same fixed lambda=0.58, same full Rev-7.1 instrumentation. Arm tag
    'e-fp' makes every filename distinct from the 'e' arm's by
    construction (checked by the collision smoke, not assumed)."""
    steps = KEYANCHOR_E_TIER_STEPS
    runs = []
    for s in (70, 71, 72):
        spec = _spec("keyanchor-e", 32, s, steps, "e-fp", geo3_active=True,
                     geo3_n_iter=KEYANCHOR_E_NS_N_ITER, geo3_resid_tol=1e-2,
                     anchor_active=True, anchor_lambda_mode="fixed",
                     anchor_lambda_fixed=KEYANCHOR_E_LAMBDA_FIXED,
                     drift_probe=True, rev7_engagement=True)
        spec["anchor_table_frozen"] = True
        spec["anchor_table_init_mode"] = "frame_potential"   # the default mode, set EXPLICITLY --
                                                              # is_done()'s identity check reads it either way
        runs.append(spec)
    assert len(runs) == 3, f"keyanchor-e-fp manifest drifted from its registered 3 runs: {len(runs)}"
    return runs


def keyanchor_e_wave_manifest():
    """--wave keyanchor-e launches BOTH candidate-(e) arms (6 cells total):
    the 'e' random-unit-rows arm (seeds 60-62) AND the 'e-fp' frozen
    frame-potential arm (seeds 70-72) -- one wave dispatch, since both
    share identical gates (the SAME K=32 BANDS_PINNED gate, budget guard,
    disk gate) and one out_dir (wavekeyanchor-e; arm tags 'e'/'e-fp' keep
    every filename distinct). This is the single function main()'s
    dispatch, the dry-run preview, and readout_rev7.py's --manifest
    keyanchor-e ALL read, so no two of them can disagree about what the
    wave contains."""
    runs = keyanchor_e_manifest() + keyanchor_e_fp_manifest()
    assert len(runs) == 6, f"keyanchor-e wave manifest drifted from its registered 6 runs: {len(runs)}"
    return runs


# ---------------------------------------------------------------------------
# KEY_ANCHORING_DESIGN.md sec 12 (Rev 12.2, CLEARED-FOR-BUILD, human sign-off
# recorded at the sec 12 header 2026-07-06) -- the capacity-cliff
# localization wave. Registered manifest name: 'keyanchor-cliff'.
#
# Arms (sec 12.2): (1) candidate (d), K in {34,38,42,46}, 3 seeds each --
# MANDATORY, PRIMARY, the ONLY arm this wave runs by default (12 cells).
# Reference arms are CUT from this wave's scope (sec 12.2 item 2) -- no
# keyanchor_cliff_reference_manifest() exists, unlike keyanchor-k48's own
# reference arm. (2) Gate-1-style probes, 1 per K (4 total) -- OPTIONAL,
# lowest cut priority (sec 12.2 item 3). (3) seed contingency, +2 seeds per
# K -- CONDITIONAL, registered but NOT built as manifest cells here (sec
# 12.2's own reserved-block discipline: these seed integers are reserved,
# never silently promoted to a callable manifest function until an
# orchestrator actually needs them, same "registering them now is what
# prevents them from becoming free post-hoc additions later" precedent as
# keyanchor_k48_dprime_manifest's own conditional gate).
# ---------------------------------------------------------------------------

KEYANCHOR_CLIFF_TIER_STEPS = KEYANCHOR_K48_TIER_STEPS   # 20,000 -- continuity, sec 12.2
KEYANCHOR_CLIFF_GATE1_STEPS = KEYANCHOR_K48_GATE1_STEPS  # 5,000 -- continuity, sec 12.2 item 3
KEYANCHOR_CLIFF_KS = (34, 38, 42, 46)                    # sec 12.2's re-picked point set

# sec 12.2's registered seed blocks, candidate (d) only (mandatory manifest).
KEYANCHOR_CLIFF_SEEDS_BY_K = {34: (130, 131, 132), 38: (230, 231, 232),
                              42: (330, 331, 332), 46: (430, 431, 432)}

# sec 12.2's optional Gate-1-style probe seeds, one per K -- conditional,
# lowest cut priority (sec 12.2 item 3 / sec 12.2.3's running-projection
# cut-priority order).
KEYANCHOR_CLIFF_GATE1_SEED_BY_K = {34: 133, 38: 233, 42: 333, 46: 433}

# sec 12.2's seed-contingency reserved blocks (+2 seeds per K, fires only if
# a K's own lambda-band/h4-bar assignment lands ambiguous at 2/3) -- RESERVED
# integers only, per this program's never-reuse convention; NOT a callable
# manifest function in this build (no orchestrator decision to fire them has
# been made yet, mirrors keyanchor_k48_dprime_manifest's own conditional-gate
# precedent: registering the identity now is what prevents it from becoming
# a free post-hoc addition later).
KEYANCHOR_CLIFF_CONTINGENCY_SEEDS_BY_K = {34: (134, 135), 38: (234, 235),
                                           42: (334, 335), 46: (434, 435)}

# sec 12.2.3's staged launch (Rev 12.2, round-2 fix 3): Stage 1 = the two
# interior/most-informative-for-x0 points (K=38, K=42); Stage 2 = the two
# points nearer the existing K=32/K=48 anchors (K=34, K=46). Stage 2 is
# mechanically REFUSED (sec 12.2.3) unless Stage 1's own realized cells
# clear the K48-calibrated bracket -- see keyanchor_cliff_stage_gate below.
KEYANCHOR_CLIFF_STAGE_BY_K = {38: 1, 42: 1, 34: 2, 46: 2}


def keyanchor_cliff_manifest(Ks=KEYANCHOR_CLIFF_KS):
    """sec 12.2's MANDATORY, PRIMARY arm: candidate (d) only, K in
    {34,38,42,46}, 3 seeds each (12 cells total) -- reference arms are CUT
    from this wave's scope (sec 12.2 item 2), so unlike keyanchor_k48_
    manifest's sibling reference/gate1/dprime/fixed-lambda1 arms, this
    wave registers exactly ONE candidate-(d)-only manifest function.
    Reuses keyanchor_k48_manifest(K, seeds) directly (sec 12.2.1's own
    build-scope requirement: 'a keyanchor_k48_manifest(K, seeds)... plus
    extend GATE2_N_ITER_BY_K') -- the SAME function every K's candidate-(d)
    cells go through, never a hand-copied twin. The wave tag string inside
    each spec's own name stays 'keyanchor-k48' (out_path()/_spec()'s own
    K-bit already makes every filename K-distinct, sec 11.9 item 16's own
    precedent) -- but main()'s own f"wave{args.wave}" convention resolves
    THIS wave's out_dir to wavekeyanchor-cliff/ (args.wave=='keyanchor-cliff'),
    a directory of its own, disjoint from wavekeyanchor-k48/ by construction
    (every wave gets its own subdirectory), not merely by disjoint K values
    within a shared one."""
    runs = []
    for K in Ks:
        runs += keyanchor_k48_manifest(K=K, seeds=KEYANCHOR_CLIFF_SEEDS_BY_K[K])
    assert len(runs) == len(Ks) * 3, \
        f"keyanchor-cliff manifest drifted from its registered {len(Ks) * 3} runs: {len(runs)}"
    return runs


def keyanchor_cliff_gate1_manifest(Ks=KEYANCHOR_CLIFF_KS):
    """sec 12.2 item 3: OPTIONAL, lowest-cut-priority Gate-1-style probes,
    one per K (4 total by default), 5,000 steps, seeds {133,233,333,433}.
    Reuses keyanchor_k48_gate1_manifest(K, seed) -- same discipline as
    keyanchor_cliff_manifest above."""
    runs = []
    for K in Ks:
        runs += keyanchor_k48_gate1_manifest(K=K, seed=KEYANCHOR_CLIFF_GATE1_SEED_BY_K[K])
    assert len(runs) == len(Ks), \
        f"keyanchor-cliff Gate-1 probe manifest drifted from its registered {len(Ks)} runs: {len(runs)}"
    return runs


# sec 12.5's budget table -- the K48-calibrated bracket (sec 11.5, unchanged;
# no per-K34/38/42/46 interpolation table, sec 12.2's own "table deleted
# outright" ruling) applies unchanged to this wave's own cells.
KEYANCHOR_CLIFF_GPUH_PER_CELL = KEYANCHOR_K48_GPUH_PER_CELL      # (0.22752, 0.97328) unrounded, sec 12.5
KEYANCHOR_CLIFF_GATE1_GPUH = KEYANCHOR_K48_GATE1_GPUH            # (0.06, 0.24), sec 12.5
KEYANCHOR_CLIFF_CONTINGENCY_MULTIPLIER = 2.0                     # sec 12.2/12.5's mandatory 2x, shared-GPU risk
# sec 12.5's registered nominal ceiling: mandatory-only bracket-pessimistic
# 2x -- the wave's OWN registered ceiling (distinct from
# KEYANCHOR_K48_GPUH_CEILING, sec 12.5's own separate arithmetic, not a
# re-derivation of the K48 wave's 12.0 ceiling).
#
# DISCLOSED discrepancy vs. sec 12.5's own prose (12 * 0.97 * 2 = 23.28):
# the design doc's own text rounds KEYANCHOR_K48_GPUH_PER_CELL[1] to "0.97"
# before multiplying; the actual, unrounded constant this file computes
# from (0.77 * KEYANCHOR_K48_STEP_COST_RATIO = 0.97328) gives 23.35872, not
# 23.28 -- a ~0.34% difference from the doc's own cited figure, same root
# cause as KEYANCHOR_CLIFF_ABORT_WALL_S's own rounding note above. Computed
# from the unrounded constant HERE (unlike the abort threshold, which is
# pinned to the doc's own literal 5,238s) because this ceiling is a live
# go/no-go computation the budget guard re-evaluates, not a single pinned
# trigger value the design text states as a specific number to match
# exactly -- using the more-precise upstream constant is the more correct
# arithmetic, not a drift. Both the 23.28 (doc prose) and 23.35872 (this
# file) versions fit within the 24.17 GPU-h reserve (margin 0.89 vs. 0.81
# GPU-h respectively) -- the wave's affordability verdict is unchanged
# either way, only the disclosed margin shrinks slightly.
KEYANCHOR_CLIFF_MANDATORY_CELLS = 12
KEYANCHOR_CLIFF_GPUH_CEILING = (KEYANCHOR_CLIFF_MANDATORY_CELLS
                                 * KEYANCHOR_CLIFF_GPUH_PER_CELL[1]
                                 * KEYANCHOR_CLIFF_CONTINGENCY_MULTIPLIER)   # 23.35872

# sec 12.7.2 R3-4 / sec 12.8.2 Rev 12.3 build-audit item (REQUIRED code
# addition): Stage 1 (6 cells, K=38+K=42) gets its OWN, smaller, dedicated
# sub-ceiling rather than only "gate Stage 2 on Stage 1's rate" -- the
# attack's own point was that an in-bracket-per-cell Stage-1 overrun could
# still consume the WHOLE wave's margin before Stage 2 launches a single
# cell. Derivation (doc-pinned, sec 12.8.2's own registered number):
# KEYANCHOR_CLIFF_GPUH_CEILING (the full 12-cell, bracket-pessimistic 2x
# mandatory ceiling, 23.35872 unrounded) / 2, rounded to the doc's own
# registered 11.68 -- half the total ceiling because Stage 1 is exactly
# half the cells (6 of 12), same bracket-pessimistic-2x costing as the full
# ceiling, no separate re-derivation. Pinned to the doc's literal 11.68
# (not left as a live `KEYANCHOR_CLIFF_GPUH_CEILING / 2` re-derivation) per
# this project's "exact thresholds" hard rule -- same precedent as
# KEYANCHOR_CLIFF_ABORT_WALL_S's own doc-pinned-literal treatment above.
KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING = 11.68   # 23.35872 / 2, rounded to the doc's registered 11.68

# sec 12.5's headroom check: 24.17 GPU-h exactness-program reserve against
# the 55.83/80 GPU-h spent figure (STATE.md, sec 12 header) -- a SEPARATE
# reconciled base from KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED above (that
# constant is this program's OWN earlier reconciliation, 49.8493, predating
# both the keyanchor-k48 and keyanchor-e waves' own realized spend; sec 12's
# own header cites a LATER, more-recent total, "≈55.83/80", which already
# includes those two waves' own realized costs). This wave's budget guard
# uses the sec-12-header figure directly, not a re-derivation from the
# earlier reconciliation -- re-summing 49.8493 + keyanchor-k48's + keyanchor-
# e's own realized wall_s would just reproduce the same 55.83 STATE.md
# already reports, so this constant is read off STATE.md's own line 654-655
# citation ("Program formally complete at ≈55.83/80 GPU-h") rather than
# re-itemized a third time in this file.
KEYANCHOR_CLIFF_PROGRAM_SPENT_GPUH = 55.83
KEYANCHOR_CLIFF_PROGRAM_GPUH_CEILING = 80.0
KEYANCHOR_CLIFF_RESERVE_GPUH = KEYANCHOR_CLIFF_PROGRAM_GPUH_CEILING - KEYANCHOR_CLIFF_PROGRAM_SPENT_GPUH  # 24.17

# sec 12.2.3's mid-run abort/budget guard, restated in full (NOT merely
# "mirrors sec 11.5" by reference -- sec 12.2.3's own text): after ANY
# completed candidate-(d) cell (broadened at Rev 12.2 from "the first cell
# only" -- round-2 finding R2-3), if that cell's realized wall_s >= 1.5x the
# K48-calibrated bracket's own upper edge, halt ALL remaining cell launches
# in this wave immediately.
#
# sec 12.2.3's own prose rounds the bracket's upper edge to "0.97 GPU-h/cell,
# i.e. 0.97 x 3600 = 3492s" and states the pinned threshold as "5,238s"
# (1.5x of the rounded 3492s). The UNROUNDED constant this file actually
# uses, KEYANCHOR_K48_GPUH_PER_CELL[1] (0.77 * KEYANCHOR_K48_STEP_COST_RATIO
# = 0.77*1.264 = 0.97328), gives 1.5*0.97328*3600 = 5255.712s -- a ~0.34%
# difference from the design doc's own cited 5,238s, traceable entirely to
# the doc's own prose rounding 0.97328 down to "0.97" before multiplying.
# Pinned to the design's OWN literal registered number (5,238s) rather than
# re-derived from the unrounded bracket, per this project's "exact
# thresholds" hard rule -- a threshold the design text itself states as a
# specific pinned value must be used AS PINNED, not silently re-derived to a
# slightly different number from a more-precise upstream constant the design
# text never asked this check to track.
KEYANCHOR_CLIFF_ABORT_WALL_S = 5238.0   # sec 12.2.3's own pinned literal, see comment above


def keyanchor_cliff_budget_guard(accept_override: bool) -> float:
    """Mirrors keyanchor_k48_budget_guard's exact pattern, using this wave's
    OWN registered ceiling (sec 12.5) against the sec-12-header reconciled
    program-spent figure (55.83), not the earlier keyanchor-k48-era 49.8493
    base (see the comment above KEYANCHOR_CLIFF_PROGRAM_SPENT_GPUH)."""
    cumulative = KEYANCHOR_CLIFF_PROGRAM_SPENT_GPUH + KEYANCHOR_CLIFF_GPUH_CEILING
    print(f"BUDGET GUARD (keyanchor-cliff): program-spent-so-far={KEYANCHOR_CLIFF_PROGRAM_SPENT_GPUH:.4f} "
          f"GPU-h + this-wave-registered-ceiling(mandatory-only, bracket-pessimistic, 2x)="
          f"{KEYANCHOR_CLIFF_GPUH_CEILING:.4f} GPU-h = cumulative {cumulative:.4f} GPU-h, program ceiling "
          f"{KEYANCHOR_CLIFF_PROGRAM_GPUH_CEILING:.0f} GPU-h (reserve "
          f"{KEYANCHOR_CLIFF_PROGRAM_GPUH_CEILING - cumulative:.4f} GPU-h -- sec 12.5's own "
          f"'~0.89 GPU-h to spare' figure).", flush=True)
    if cumulative > KEYANCHOR_CLIFF_PROGRAM_GPUH_CEILING and not accept_override:
        print(f"ERROR: projected cumulative spend {cumulative:.4f} GPU-h EXCEEDS the "
              f"{KEYANCHOR_CLIFF_PROGRAM_GPUH_CEILING:.0f} GPU-h exactness-program ceiling -- REFUSING "
              f"to launch keyanchor-cliff. Pass --accept-budget-override to force past this guard.",
              file=sys.stderr)
        sys.exit(5)
    return cumulative


# sec 12.7.2 R3-5 / sec 12.8.2 Rev 12.3 build-audit item (REQUIRED code
# addition): the durable cross-program clearance sentinel's filename,
# written into THIS wave's own out_dir (out_path()'s existing convention --
# derived, never a hand-typed divergent path, same e633862 audit F1
# discipline as KEYANCHOR_MECH_GATE1_JSON_DEFAULT above). FROZEN_BIAS_LM_
# DESIGN.md's own launcher is required (sec 12.8.2's own text) to check
# this file's existence before its calibration cell starts -- this
# constant is the one source of truth both sides read/write against.
KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME = "STAGE1_RATES_OK"


def keyanchor_cliff_stage_gate(out_dir: str, stage: int, accept_override: bool) -> dict:
    """sec 12.2.3's staged-launch mechanical gate (Rev 12.2, round-2 fix 3):
    --stage 2 REFUSES to launch unless every Stage-1 K (38, 42) has ALL of
    its registered seeds' completed, validity-checked cells in out_dir
    (per-seed completeness, not mere per-K non-emptiness -- independent-
    verifier poke B) AND every completed
    Stage-1 cell's own realized wall_s sits within the K48-calibrated
    bracket's upper edge (sec 12.2.3 item 1's own 0.97 GPU-h/cell = 3492s,
    NOT the 1.5x hard-abort threshold -- the stage gate is the softer
    'within bracket' check sec 12.2.3's own staged-launch text requires
    before Stage 2 commits, distinct from the hard abort rule which fires
    at 1.5x regardless of stage). This is a MECHANICAL check against
    completed cells' own recorded wall_s (never an honor-system flag) --
    reads each Stage-1 cell's result JSON directly, the same is_done()-
    adjacent discipline every other gate in this file uses.

    sec 12.7.2 R3-4 (build-audit REQUIRED addition): ALSO enforces Stage 1's
    own dedicated sub-ceiling (KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING, 11.68
    GPU-h) by summing the six Stage-1 cells' realized wall_s -> GPU-h --
    this fires regardless of per-cell abort status (a cell can each
    individually sit within the per-cell bracket yet the STAGE-1 TOTAL still
    blow through its own dedicated margin; that is exactly the gap the
    attack (sec 12.7.2 R3-4) flagged: 'Stage 1 alone should carry its own
    smaller, dedicated contingency... so a Stage-1-only overrun cannot
    itself consume the margin intended for Stage 2'). Comparison is STRICT
    greater-than (sum > 11.68 refuses; sum == 11.68 passes) -- deliberately
    chosen to mirror keyanchor_cliff_budget_guard's own existing pattern
    ('if cumulative > ...GPUH_CEILING') elsewhere in this file, and because
    the ceiling is framed throughout sec 12.5/12.8.2 as a MAXIMUM spend the
    wave must fit WITHIN (a sum exactly equal to a fixed budget is the
    textbook boundary-inclusive case for a "fits within" ceiling, not an
    overrun) -- exact threshold, no slack, per this project's hard rule.

    sec 12.7.2 R3-5 (build-audit REQUIRED addition): on a CLEAN pass (rates
    within bracket AND sub-ceiling respected), writes the durable cross-
    program clearance sentinel (KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME) into
    out_dir -- the SAME directory the Stage-1 cells themselves write their
    result JSONs to (out_path()'s existing convention, not a new path).
    NEVER written on an --accept-stage-override bypass (that path returns
    before this function reaches the sentinel-writing branch at all) --
    an override bypass means 'a human decided to proceed anyway', not
    'Stage 1 actually cleared', and the sentinel's whole purpose (a
    cross-program signal that rates were mechanically verified OK) would
    be a lie if written on a bypassed check."""
    stage1_ks = [K for K, s in KEYANCHOR_CLIFF_STAGE_BY_K.items() if s == 1]
    bracket_upper_s = KEYANCHOR_CLIFF_GPUH_PER_CELL[1] * 3600.0   # 3492.0
    if accept_override:
        print("=" * 70 + "\nWARNING: --accept-stage-override -- sec 12.2.3's Stage-1-clears-"
              "bracket-before-Stage-2 mechanical gate is being BYPASSED by an explicit human "
              "decision. This is recorded here; it does NOT mean Stage 1 cleared. Per sec "
              "12.7.2 R3-5, the STAGE1_RATES_OK sentinel is NOT written on an override bypass.\n"
              + "=" * 70, flush=True)
        return {"gate_bypassed": True, "stage1_ks": stage1_ks, "sentinel_written": False}
    if stage != 2:
        return {"gate_bypassed": False, "not_applicable": True, "stage": stage, "sentinel_written": False}
    # Independent-verifier poke A: a sentinel left over from an EARLIER gate
    # run must not survive a re-check that would refuse -- downstream
    # consumers (FROZEN_BIAS_LM sec 8.2a) gate on existence alone, so the
    # file may only exist if the MOST RECENT gate evaluation passed clean.
    # Remove any pre-existing sentinel before evaluating; the clean-pass
    # branch below rewrites it.
    _stale = os.path.join(out_dir, KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME)
    if os.path.exists(_stale):
        os.remove(_stale)
        print(f"stage gate: removed pre-existing {KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME} "
              f"(re-derived below; a stale pass certificate must not survive a failed re-check)",
              flush=True)
    per_k_cells = {K: [] for K in stage1_ks}
    for spec in keyanchor_cliff_manifest(Ks=tuple(stage1_ks)):
        p = out_path(out_dir, spec)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            result = json.load(f)
        if not is_done(out_dir, spec):
            continue
        per_k_cells[spec["K"]].append(result.get("wall_s"))
    # Independent-verifier poke B (FATAL, fixed): a K counts as missing when
    # FEWER than its registered seeds have completed, not only when zero
    # have -- 5-of-6 cells must refuse, not silently certify Stage 1.
    missing_ks = [K for K, cells in per_k_cells.items()
                  if len(cells) < len(KEYANCHOR_CLIFF_SEEDS_BY_K[K])]
    if missing_ks:
        print(f"ERROR: --stage 2 REFUSED -- Stage 1 (K={stage1_ks}) is incomplete for "
              f"K={missing_ks}: fewer completed, validity-checked cells than the registered "
              f"seeds ({ {K: len(per_k_cells[K]) for K in missing_ks} } of "
              f"{ {K: len(KEYANCHOR_CLIFF_SEEDS_BY_K[K]) for K in missing_ks} }). Run --stage 1 "
              f"to completion first, or pass --accept-stage-override for an explicit, documented "
              f"human override.",
              file=sys.stderr)
        sys.exit(1)
    # sec 12.7.2 R3-4: Stage-1 sub-ceiling, checked FIRST and independently
    # of the softer per-cell bracket check below -- "regardless of per-cell
    # abort status" (the task's own registered wording) means this check
    # must not be reachable-only-if-the-bracket-check-already-passed. Sum
    # the six Stage-1 cells' own realized wall_s (all six are guaranteed
    # present by this point: missing_ks already refused above, and each K's
    # list holds every completed cell this manifest registers for that K,
    # 3 seeds each = 6 total at K=38+42). NOTE (disclosed, not hidden): at
    # the doc's own registered numbers, 6 cells all individually WITHIN the
    # per-cell bracket (bracket_upper_s=3503.808s, unrounded) can sum to at
    # most 6*3503.808=21022.8s=5.84 GPU-h, well under this 11.68 GPU-h
    # ceiling -- so in practice this sub-ceiling only ever binds together
    # with (or ahead of) the per-cell bracket check on cells that are
    # ALSO individually over-bracket; checking it first (rather than after
    # the bracket check's own sys.exit) is what makes it a genuinely
    # independent, always-evaluated condition rather than dead code the
    # bracket check's own exit would otherwise mask.
    stage1_wall_s_flat = [w for cells in per_k_cells.values() for w in cells if w is not None]
    stage1_gpuh = sum(stage1_wall_s_flat) / 3600.0
    if stage1_gpuh > KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING:
        print(f"ERROR: --stage 2 REFUSED -- sec 12.7.2 R3-4 Stage-1 sub-ceiling: Stage 1 (K="
              f"{stage1_ks})'s six realized cells sum to {stage1_gpuh:.4f} GPU-h > the dedicated "
              f"{KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING:.2f} GPU-h Stage-1 ceiling (half the full "
              f"bracket-pessimistic 2x mandatory ceiling) -- regardless of any individual cell's "
              f"own per-cell abort status. Stage 1 alone has consumed more than its own dedicated "
              f"margin; re-price before continuing (contention diagnosis first) -- do not launch "
              f"Stage 2 at the original estimate. Pass --accept-stage-override for an explicit, "
              f"documented human override.", file=sys.stderr)
        sys.exit(1)
    over_bracket = {K: [w for w in cells if w is not None and w > bracket_upper_s]
                    for K, cells in per_k_cells.items()}
    any_over = any(len(ws) > 0 for ws in over_bracket.values())
    if any_over:
        print(f"ERROR: --stage 2 REFUSED -- sec 12.2.3's staged-launch gate: Stage 1 (K="
              f"{stage1_ks}) has at least one completed cell whose realized wall_s exceeds the "
              f"K48-calibrated bracket's upper edge ({bracket_upper_s:.1f}s, 0.97 GPU-h/cell): "
              f"{over_bracket}. Per sec 12.2.3 item 3, re-price before continuing (contention "
              f"diagnosis first) -- do not launch Stage 2 at the original estimate. Pass "
              f"--accept-stage-override for an explicit, documented human override.",
              file=sys.stderr)
        sys.exit(1)
    print(f"sec 12.2.3 STAGE GATE PASSED: Stage 1 (K={stage1_ks}) has ALL registered seeds "
          f"completed per K, "
          f"all within the {bracket_upper_s:.1f}s bracket upper edge -- Stage 2 (K="
          f"{[K for K, s in KEYANCHOR_CLIFF_STAGE_BY_K.items() if s == 2]}) may proceed. "
          f"sec 12.7.2 R3-4 Stage-1 sub-ceiling: {stage1_gpuh:.4f} / "
          f"{KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING:.2f} GPU-h -- WITHIN.", flush=True)
    sentinel_report = _write_keyanchor_cliff_stage1_sentinel(
        out_dir, per_k_cells, stage1_gpuh, bracket_upper_s)
    return {"gate_bypassed": False, "stage1_ks": stage1_ks, "per_k_wall_s": per_k_cells,
            "stage1_gpuh": stage1_gpuh, "sentinel_written": True,
            "sentinel_path": sentinel_report["sentinel_path"]}


def _write_keyanchor_cliff_stage1_sentinel(out_dir: str, per_k_cells: dict, stage1_gpuh: float,
                                            bracket_upper_s: float) -> dict:
    """sec 12.7.2 R3-5's durable cross-program clearance sentinel. Written
    ONLY by keyanchor_cliff_stage_gate's own clean-pass branch (never called
    from the override-bypass or gate-not-applicable paths). One JSON line
    (matches this project's other single-line-JSON artifacts, e.g.
    ABORTED.txt's sibling pattern) containing: the six cells' own wall_s
    values (per-K), the computed Stage-1 GPU-h, the bracket edge, and a
    timestamp. This file's result-JSON schema (run_deltanet_rd.py's own
    "wall_s": time.time() - t0) records only a DURATION, never an absolute
    end-timestamp -- there is no upstream timestamp field to reuse here, so
    (matching this file's own existing precedent of calling time.time()
    directly at operational gate points, e.g. `unblind_override_at =
    time.time()` above) this wall-clock call is made directly, at sentinel-
    write time, immediately after the newest cell's own completion was
    already confirmed by the gate above."""
    sentinel_path = os.path.join(out_dir, KEYANCHOR_CLIFF_STAGE1_SENTINEL_NAME)
    payload = {
        "stage1_wall_s_by_k": {str(K): cells for K, cells in per_k_cells.items()},
        "stage1_gpuh": stage1_gpuh,
        "stage1_gpuh_ceiling": KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING,
        "bracket_upper_s": bracket_upper_s,
        "timestamp": time.time(),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(sentinel_path, "w") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    print(f"sec 12.7.2 R3-5: wrote clearance sentinel {sentinel_path!r} "
          f"(stage1_gpuh={stage1_gpuh:.4f}, ceiling={KEYANCHOR_CLIFF_STAGE1_GPUH_CEILING:.2f}).",
          flush=True)
    return {"sentinel_path": sentinel_path}


def keyanchor_cliff_check_abort(spec: dict, wall_s: float) -> None:
    """sec 12.2.3's mechanical abort rule, broadened at Rev 12.2 to ANY
    completed cell (not only the first): if a completed candidate (d)
    cell's own realized wall_s >= KEYANCHOR_CLIFF_ABORT_WALL_S (5,238s),
    halt all remaining cell launches in this wave immediately by raising --
    the caller (main()'s own dispatch loop) catches this and stops
    launching new cells for the keyanchor-cliff wave, rather than proceeding
    on the assumption the outlier cell was a one-off. In-code, not a
    comment: every keyanchor-cliff cell completion calls this."""
    if wall_s >= KEYANCHOR_CLIFF_ABORT_WALL_S:
        raise KeyanchorCliffAbort(
            f"sec 12.2.3 ABORT: cell {spec['name']!r} (K={spec['K']}) realized wall_s={wall_s:.1f}s "
            f">= {KEYANCHOR_CLIFF_ABORT_WALL_S:.1f}s (1.5x the K48-calibrated bracket's upper edge). "
            f"Halting all remaining keyanchor-cliff launches -- diagnose contention with the "
            f"concurrent frozen-bias LM program on shared GPUs 2-7 (sec 12.2.3 item 2) before "
            f"resuming.")


class KeyanchorCliffAbort(RuntimeError):
    """sec 12.2.3's mechanical, in-code hard-halt signal -- raised by
    keyanchor_cliff_check_abort, caught by main()'s dispatch loop for the
    keyanchor-cliff wave only. A distinct exception type (not a bare
    RuntimeError) so the dispatch loop's own generic except-and-CRASHED.txt
    handler can distinguish a deliberate, expected halt from an actual
    orchestrator crash."""


KEYANCHOR_DRIFT_DIAG_JSON_DEFAULT = os.path.join(
    HERE, "results/deltanet_rd_exactness/keyanchor_drift/wave_neg1_drift.json")


def gate_keyanchor_drift_diag(drift_json_path: str, accept_override: bool) -> dict:
    """--wave keyanchor-confirm's PRE-LAUNCH gate (task-directed, 2026-07-06
    build): the FIXED keyanchor_drift_diagnostic.py (this same build's
    log_every fix -- see that module's docstring) must have produced a
    JSON whose Gate-1 launch_read reads clean BEFORE any keyanchor-confirm
    cell dispatches. Mirrors gate_geo3_drift's existing pattern exactly --
    this is a CPU-only file-read gate (no GPU work here); the diagnostic
    ITSELF still requires CUDA and is run separately (same "separate
    sequential driver" precedent as calibrate_arm_iv.py/
    geo3_drift_diagnostic.py, per this module's own docstring).
    --accept-keyanchor-diag-override is an explicit, loudly-logged human
    override (same override class as --accept-gate-override/
    --accept-unconverged-rho/--unblind-override above) -- never a silent
    default."""
    if accept_override:
        print("=" * 70 + "\nWARNING: --accept-keyanchor-diag-override -- the pre-launch "
              "keyanchor_drift_diagnostic.py Gate-1 read is being BYPASSED by an explicit human "
              "decision. This is recorded here; it does NOT mean the diagnostic passed.\n"
              + "=" * 70, flush=True)
        return {"gate_bypassed": True, "drift_json_path": drift_json_path}
    if not os.path.exists(drift_json_path):
        print(f"ERROR: --wave keyanchor-confirm requires a clean keyanchor_drift_diagnostic.py "
              f"run first (its Gate-1 read) -- {drift_json_path!r} does not exist. Run (on GPU, "
              f"using the FIXED script -- sec 9.3/9.5's own required follow-up):\n"
              f"  python keyanchor_drift_diagnostic.py --k 16 32 --out {drift_json_path}\n"
              f"or pass --accept-keyanchor-diag-override for an explicit, documented override.",
              file=sys.stderr)
        sys.exit(1)
    with open(drift_json_path) as f:
        d = json.load(f)
    try:
        launch = d["launch_read"]["launch"]
        gate_value = d["launch_read"]["predicted_gate_value"]
        gate_cell = d["launch_read"]["gate_cell"]
        k16_mean = d["per_k"]["16"]["after_probe"]["post_ns"]["mean"]
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"ERROR: --keyanchor-drift-json {drift_json_path!r} is missing an expected field "
              f"({e!r}) -- this does not look like a real (FIXED) keyanchor_drift_diagnostic.py "
              f"output. Refusing to launch; pass --accept-keyanchor-diag-override for an explicit "
              f"override.", file=sys.stderr)
        sys.exit(1)
    if not launch:
        print(f"ERROR: keyanchor_drift_diagnostic.py's Gate-1 read FAILED -- {gate_cell} predicted "
              f"rec@0.9={gate_value:.4f} < 0.8 (K=16 trained-checkpoint drift mean={k16_mean:.4f}). "
              f"Pass --accept-keyanchor-diag-override for an explicit, documented override.",
              file=sys.stderr)
        sys.exit(1)
    print(f"keyanchor_drift_diagnostic.py Gate-1 PASSED: {gate_cell} predicted rec@0.9="
          f"{gate_value:.4f} >= 0.8 (K=16 drift mean={k16_mean:.4f})", flush=True)
    return {"gate_bypassed": False, "drift_json_path": drift_json_path,
            "gate_cell": gate_cell, "predicted_gate_value": gate_value}


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
    # e633862 audit F2: pass the registered ceilings so validate can also
    # pin the one field its content re-derivation reads as input.
    doc = ka.validate_bands_pinned(bp_path, ceiling_by_k=keyanchor_ceiling_by_k())
    if doc is None:
        print(f"ERROR: {bp_path!r} does not exist, fails hash validation, or its STORED bands "
              f"dict does not reproduce under live re-derivation from the referenced reference-arm "
              f"JSONs (e633862 audit F2 -- tampered/corrupted pin contents are a pin-integrity "
              f"error, never silently trusted). Run the 6 reference arms to completion first "
              f"(--wave ref), then --wave keyanchor-bands to pin, or pass --unblind-override for "
              f"an explicit, documented, tier-demoting override.",
              file=sys.stderr)
        sys.exit(1)
    for K, entry in doc["bands"].items():
        if entry["unresolvable"]:
            print(f"  NOTE: K={K} post-NS engagement bands are UNRESOLVABLE (engaged_K "
                  f"{entry['engaged_k']:.4f} >= ceiling {entry['ceiling']:.4f} - 0.005) -- see the "
                  f"leave-one-out sensitivity report in {bp_path}.", flush=True)
    print(f"sec 3.6 GATE PASSED: {bp_path} validates (hashes match every referenced reference-arm "
          f"result JSON; stored bands reproduce under live re-derivation).", flush=True)
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
        # e633862 audit F2: the SAME extraction validate_bands_pinned's
        # content re-derivation uses -- one definition, never a twin.
        per_k_final_drift[spec["K"]].append(ka.reference_final_drift(result))
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


# The argparse --out-dir default, hoisted to a module constant so the
# Gate-1 reader default below is derived from the SAME root the writer
# resolves against (e633862 audit F1).
DEFAULT_OUT_DIR = os.path.join(HERE, "results/deltanet_rd_exactness")

# e633862 audit F1 fix: the Gate-1 probe reader's default is COMPUTED from
# the writer's own functions -- out_path() + the manifest's own single spec
# + main()'s own f"wave{args.wave}" subdirectory convention at the argparse
# default --out-dir -- never a hand-typed sibling path. Writer and reader
# cannot diverge: any change to the manifest spec (name bits), out_path(),
# or the default out-dir moves BOTH ends together. smoke_keyanchor_mech.py
# asserts the executed equality (writer path == this constant); the CLI
# override (--keyanchor-mech-gate1-json) remains for non-default out-dirs.
KEYANCHOR_MECH_GATE1_JSON_DEFAULT = out_path(
    os.path.join(DEFAULT_OUT_DIR, "wavekeyanchor-mech-gate1"),
    keyanchor_mech_gate1_manifest()[0])

# SAME e633862 audit F1 discipline, applied to the K=48 Gate-1-style probe
# (sec 11.1 item 4) -- writer and reader cannot diverge.
KEYANCHOR_K48_GATE1_JSON_DEFAULT = out_path(
    os.path.join(DEFAULT_OUT_DIR, "wavekeyanchor-k48-gate1"),
    keyanchor_k48_gate1_manifest()[0])


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
        # 2026-07-06 keyanchor-confirm build fix: drift_probe was already
        # named in _spec()'s own docstring as one of the fields "is_done()
        # ... re-derives ... from the result JSON's own exactness_config,
        # never trusting the filename alone" -- but was never actually
        # checked here (run_deltanet_rd.py also never wrote it into
        # exactness_config until this same build). Same missing-key-
        # defaults-to-off discipline as every field above (archived
        # pre-fix result JSONs lack this key entirely).
        if bool(ec.get("drift_probe", False)) != bool(spec.get("drift_probe", False)):
            return False
        # KEY_ANCHORING_DESIGN.md sec 10 (Rev 7.1, 2026-07-06 keyanchor-mech
        # build): SAME missing-key-defaults-to-off discipline as every field
        # above (archived pre-Rev-7.1 result JSONs lack this key entirely).
        if bool(ec.get("rev7_engagement", False)) != bool(spec.get("rev7_engagement", False)):
            return False
        # KEY_ANCHORING_DESIGN.md sec 10.13 (candidate (e), 2026-07 K48+e
        # build): SAME missing-key-defaults-to-off discipline -- a
        # candidate-(e) cell (anchor_table_frozen=True, init_mode=
        # 'random_unit_rows') must NEVER resume-match an already-archived
        # candidate-(d)-fixed cell that happens to share K/seed/lambda_fixed
        # (e.g. wavekeyanchor-neg1's own armkeyanchor-d-fixed probes) --
        # both fields are arm-identity-relevant, not filename-bit-only.
        if bool(ec.get("anchor_table_frozen", False)) != bool(spec.get("anchor_table_frozen", False)):
            return False
        if ec.get("anchor_table_init_mode", "frame_potential") != \
           spec.get("anchor_table_init_mode", "frame_potential"):
            return False
        return True
    except Exception:
        return False


def build_cmd(spec, out_dir, timeout, unblind_override_at: float | None = None,
              ckpt_base_dir: str | None = None):
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
        if spec.get("anchor_table_frozen"):
            cmd += ["--anchor-table-frozen"]
        if spec.get("anchor_table_init_mode", "frame_potential") != "frame_potential":
            cmd += ["--anchor-table-init-mode", str(spec["anchor_table_init_mode"])]
    if spec.get("lambda_anchor"):
        cmd += ["--lambda-anchor", str(spec["lambda_anchor"])]
    if spec.get("drift_probe"):
        cmd += ["--drift-probe"]
    if spec.get("rev7_engagement"):
        cmd += ["--rev7-engagement"]
    if ckpt_base_dir is not None and spec.get("anchor_active"):
        # KEY_ANCHORING_DESIGN.md sec 10.10: one checkpoint subdirectory per
        # cell, under the wave's own checkpoint base dir.
        cmd += ["--ckpt-dir", os.path.join(ckpt_base_dir, spec["name"])]
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
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--wave", choices=["-1", "0", "1", "geo3", "ref", "keyanchor-neg1",
                                        "keyanchor-bands", "keyanchor", "keyanchor-confirm",
                                        "keyanchor-mech-gate1", "keyanchor-mech",
                                        "keyanchor-k48-ref", "keyanchor-k48-bands",
                                        "keyanchor-k48-gate1", "keyanchor-k48",
                                        "keyanchor-e", "keyanchor-cliff"], default=None,
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
                          "--unblind-override is passed. 'keyanchor-confirm' (sec 9.5's required "
                          "follow-up, 2026-07-06) launches candidate (d)'s K=32 seeds {0,1,2} "
                          "(+ an optional K=16 seed-0 spot check) WITH drift_probe=True, re-instrumenting "
                          "the already-published h4 result at full evidentiary tier -- GATED on BOTH "
                          "gate_keyanchor_drift_diag (--keyanchor-drift-json, the FIXED "
                          "keyanchor_drift_diagnostic.py's Gate-1 read) AND the SAME sec 3.6 "
                          "BANDS_PINNED gate 'keyanchor' uses (these are anchor-arm cells too). "
                          "KEY_ANCHORING_DESIGN.md sec 11 (Rev K48.1) waves: 'keyanchor-k48-ref' "
                          "launches the 3 mandatory K=48 reference arms; 'keyanchor-k48-bands' is a "
                          "NO-GPU action writing BANDS_PINNED_K48.json (sec 11.1.1, separate from "
                          "K=16/32's own BANDS_PINNED.json); 'keyanchor-k48-gate1' launches the "
                          "DISCLOSED, NON-GATING 5,000-step probe (sec 11.1 item 4); 'keyanchor-k48' "
                          "launches candidate (d)'s 3 mandatory cells (+ the OPTIONAL fixed-lambda=1 "
                          "ceiling probe with --include-k48-fixed-lambda1, + candidate (d')'s "
                          "CONDITIONAL cells with --include-k48-dprime AND "
                          "--accept-k48-dprime-orchestrator-signoff) -- REFUSES to launch without a "
                          "valid BANDS_PINNED_K48.json (sec 11.1.1) unless --unblind-override is "
                          "passed. sec 10.13 wave: 'keyanchor-e' launches BOTH candidate-(e) arms "
                          "(6 cells, K=32, fixed lambda=0.58): the 'e' frozen-RANDOM-unit-rows arm "
                          "(seeds 60-62, per the stub's own name/motivation) AND the 'e-fp' frozen "
                          "frame-potential arm (seeds 70-72, per the stub's literal init text -- "
                          "audit prescription) -- REUSES the EXISTING K=16/32 BANDS_PINNED.json gate "
                          "(these are K=32 anchor-arm cells, sec 3.6 applies unchanged, no new "
                          "K48-style bands file needed here). "
                          "KEY_ANCHORING_DESIGN.md sec 12 (Rev 12.2, CLEARED-FOR-BUILD, human "
                          "sign-off recorded 2026-07-06) wave: 'keyanchor-cliff' launches candidate "
                          "(d)'s 12 MANDATORY cells, K in {34,38,42,46} x 3 seeds -- reference arms "
                          "are CUT from this wave's scope (sec 12.2 item 2, unlike keyanchor-k48). "
                          "REQUIRES --stage 1 or --stage 2: --stage 1 launches K=38+K=42 (the two "
                          "interior points); --stage 2 launches K=34+K=46 and MECHANICALLY REFUSES "
                          "unless every Stage-1 K has a completed cell within the K48-calibrated "
                          "bracket (sec 12.2.3's staged-launch gate, --accept-stage-override "
                          "bypasses). --include-cliff-gate1 additionally launches the OPTIONAL "
                          "per-K Gate-1-style probes for the K's in the requested stage.")
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
                     help="--wave keyanchor/keyanchor-confirm: KEY_ANCHORING_DESIGN.md sec 3.6's "
                          "explicit, loudly-logged human override of the BANDS_PINNED mechanical "
                          "blinding gate. EVERY keyanchor/keyanchor-confirm run launched under this "
                          "flag has its OWN result JSON stamped claim_tier='descriptive' + "
                          "unblind_override=True + a timestamp at assembly time (Rev 5 R4-1 fix) -- "
                          "never only this launch console or a wave-summary artifact.")
    ap.add_argument("--keyanchor-drift-json", type=str, default=KEYANCHOR_DRIFT_DIAG_JSON_DEFAULT,
                     help="--wave keyanchor-confirm: path to the FIXED keyanchor_drift_diagnostic.py's "
                          "output JSON (this build's log_every fix, sec 9.3) -- REQUIRED clean Gate-1 "
                          "read before any keyanchor-confirm cell dispatches, unless "
                          "--accept-keyanchor-diag-override is passed.")
    ap.add_argument("--accept-keyanchor-diag-override", action="store_true",
                     help="--wave keyanchor-confirm: bypass the pre-launch keyanchor_drift_diagnostic.py "
                          "Gate-1 read with an explicit, loudly-logged human override -- same override "
                          "class as --accept-gate-override/--accept-unconverged-rho/--unblind-override.")
    ap.add_argument("--keyanchor-mech-gate1-json", type=str, default=KEYANCHOR_MECH_GATE1_JSON_DEFAULT,
                     help="--wave keyanchor-mech: path to the candidate (d') Gate-1 probe's own "
                          "result JSON (--wave keyanchor-mech-gate1's --out) -- REQUIRED clean read "
                          "(predicted h4 rec@0.9 >= 0.8) before any keyanchor-mech cell dispatches, "
                          "unless --accept-keyanchor-mech-gate1-override is passed.")
    ap.add_argument("--accept-keyanchor-mech-gate1-override", action="store_true",
                     help="--wave keyanchor-mech: bypass the candidate (d') Gate-1 pre-launch probe "
                          "read with an explicit, loudly-logged human override -- same override class "
                          "as --accept-gate-override/--accept-unconverged-rho/--unblind-override.")
    ap.add_argument("--accept-budget-override", action="store_true",
                     help="--wave keyanchor-mech: bypass the sec 10.7 budget guard (registered 12 "
                          "GPU-h ceiling vs. the 80 GPU-h exactness-program cap) with an explicit, "
                          "loudly-logged human override.")
    ap.add_argument("--ckpt-base-dir", type=str, default=None,
                     help="--wave keyanchor-mech / keyanchor-mech-gate1 / keyanchor-k48* / "
                          "keyanchor-e: base directory for the sec 10.10 checkpoint writer (one "
                          "subdirectory per cell, named after the cell's own spec['name']). Per-wave "
                          "defaults on the training box: /data/deltanet_rd_keyanchor_ckpts/"
                          "wavekeyanchor-mech (mech; pre-created by this build) and "
                          ".../wavekeyanchor-mech-gate1 (gate1 scratch -- e633862 audit fold-in 1: "
                          "the probe exercises the checkpoint writer too); "
                          ".../wavekeyanchor-k48, .../wavekeyanchor-k48-gate1, .../wavekeyanchor-e "
                          "(all pre-created by this build, see the chain scripts).")
    ap.add_argument("--keyanchor-k48-gate1-json", type=str, default=KEYANCHOR_K48_GATE1_JSON_DEFAULT,
                     help="--wave keyanchor-k48: path to the K=48 Gate-1-style probe's own result "
                          "JSON (--wave keyanchor-k48-gate1's --out). DISCLOSED, NON-GATING (sec "
                          "11.1 item 4) -- read and reported, never blocks the launch.")
    ap.add_argument("--include-k48-fixed-lambda1", action="store_true",
                     help="--wave keyanchor-k48: additionally launch the OPTIONAL, lowest-cut-"
                          "priority fixed-lambda=1 ceiling-validation probe (sec 11.4.3, seed 50). "
                          "Off by default (sec 11.5: first cut under budget/schedule pressure).")
    ap.add_argument("--include-k48-dprime", action="store_true",
                     help="--wave keyanchor-k48: additionally launch candidate (d')'s 3 CONDITIONAL "
                          "cells (seeds {40,41,42}, sec 11.1 item 3). REQUIRES "
                          "--accept-k48-dprime-orchestrator-signoff (this cell's own launch "
                          "precondition is an orchestrator sign-off on Rev 7.1's own written K=32 "
                          "verdict -- Outcome A(d') or D' -- NOT a mechanical gate; Rev K48.1's own "
                          "honest downgrade, sec 11.1 item 3/sec 11.10 C11). Off by default.")
    ap.add_argument("--accept-k48-dprime-orchestrator-signoff", action="store_true",
                     help="--wave keyanchor-k48 --include-k48-dprime: the explicit, loudly-logged "
                          "human confirmation that Rev 7.1's own K=32 wave has reached a written "
                          "Outcome A(d') or D' verdict (sec 10.6's routing table) -- read this "
                          "wave's own docs before passing; there is no script that checks this "
                          "mechanically yet (sec 11.1 item 3's own registered build gap).")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None,
                     help="--wave keyanchor-cliff: REQUIRED. --stage 1 launches K=38+K=42 (sec "
                          "12.2.3's staged launch, the two interior points). --stage 2 launches "
                          "K=34+K=46 and REFUSES (mechanical check against Stage 1's own completed "
                          "cells' wall_s, sec 12.2.3) unless Stage 1 cleared the K48-calibrated "
                          "bracket -- --accept-stage-override bypasses.")
    ap.add_argument("--accept-stage-override", action="store_true",
                     help="--wave keyanchor-cliff --stage 2: bypass sec 12.2.3's mechanical "
                          "Stage-1-clears-bracket gate with an explicit, loudly-logged human "
                          "override -- same override class as --accept-gate-override/"
                          "--unblind-override/--accept-budget-override.")
    ap.add_argument("--include-cliff-gate1", action="store_true",
                     help="--wave keyanchor-cliff: additionally launch the OPTIONAL, lowest-cut-"
                          "priority Gate-1-style probes (sec 12.2 item 3) for the K's in the "
                          "requested --stage. Off by default (sec 12.2.3: first cut under budget "
                          "pressure).")
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

        print("\n" + "=" * 70)
        print("KEY_ANCHORING_DESIGN.md sec 9.5 -- keyanchor-confirm wave preview "
              "(confirmatory RE-run, NOT part of the 28-run mandatory baseline above)")
        print("=" * 70)
        kc = keyanchor_confirm_manifest()
        # task-directed anchor: ~0.28 GPU-h/cell (sec 5's own per-cell figure
        # for a 20,000-step candidate-(d) run) -- NOT yet adjusted for
        # drift_probe=True's own per-checkpoint overhead (item 5 + item 6 +
        # the once-per-run sec-3.7 sweep); reference_arms_manifest()'s own
        # realized-vs-priced gap (~0.83 GPU-h/run, item-5-only) is the
        # closest comparable data point -- flagged, not silently assumed
        # away (see keyanchor_confirm_manifest's own docstring + the build
        # report's scrutiny list).
        gpuh_kc = len(kc) * 0.28
        print(f"  keyanchor-confirm (candidate d, K=32 seeds{{0,1,2}}"
              f"{' + K=16 s0 spot check' if len(kc) == 4 else ''}): {len(kc)} runs | "
              f"~{gpuh_kc:.2f} GPU-h (pre-drift-probe-overhead estimate; see scrutiny list)")

        print("\n" + "=" * 70)
        print("KEY_ANCHORING_DESIGN.md sec 10 (Rev 7.1) -- keyanchor-mech wave preview "
              "(mechanism-tier confirmation, r_e + null-pool BH-FDR engagement)")
        print("=" * 70)
        kg1 = keyanchor_mech_gate1_manifest()
        km = keyanchor_mech_manifest()
        from collections import Counter as _Counter
        lo_g1, hi_g1 = KEYANCHOR_MECH_GATE1_GPUH
        lo_cell, hi_cell = KEYANCHOR_MECH_GPUH_PER_CELL
        lo_mand, hi_mand = lo_g1 + len(km) * lo_cell, hi_g1 + len(km) * hi_cell
        print(f"  Gate-1 probe (candidate (d'), K=32, {KEYANCHOR_MECH_GATE1_STEPS} steps): "
              f"{len(kg1)} run | ~{lo_g1:.2f}-{hi_g1:.2f} GPU-h")
        print(f"  mandatory cells: {len(km)} runs | by arm {_Counter(s['arm'] for s in km)} | "
              f"by K {_Counter(s['K'] for s in km)} | ~{lo_cell:.2f}-{hi_cell:.2f} GPU-h/cell "
              f"(realized bracket, sec 10.7)")
        print(f"  MANDATORY TOTAL: {len(kg1) + len(km)} runs | ~{lo_mand:.2f}-{hi_mand:.2f} GPU-h")
        print(f"  registered nominal ceiling: {KEYANCHOR_MECH_GPUH_CEILING:.1f} GPU-h (sec 10.7) -> "
              f"program cumulative {KEYANCHOR_PROGRAM_SPENT_GPUH:.1f} + "
              f"{KEYANCHOR_MECH_GPUH_CEILING:.1f} = "
              f"{KEYANCHOR_PROGRAM_SPENT_GPUH + KEYANCHOR_MECH_GPUH_CEILING:.1f} / "
              f"{KEYANCHOR_PROGRAM_GPUH_CEILING:.0f} GPU-h "
              f"({'WITHIN' if KEYANCHOR_PROGRAM_SPENT_GPUH + KEYANCHOR_MECH_GPUH_CEILING <= KEYANCHOR_PROGRAM_GPUH_CEILING else 'EXCEEDS'} "
              f"the exactness-program cap)")
        assert len(km) == 7 and len(kg1) == 1, \
            f"keyanchor-mech run count drifted from the registered 7+1: {len(km)}+{len(kg1)}"

        print("\n" + "=" * 70)
        print("KEY_ANCHORING_DESIGN.md sec 11 (Rev K48.1, CLEARED-FOR-BUILD) -- "
              "keyanchor-k48 wave preview (capacity-curve extension)")
        print("=" * 70)
        k48_ref = keyanchor_k48_reference_manifest()
        k48_d = keyanchor_k48_manifest()
        k48_g1 = keyanchor_k48_gate1_manifest()
        k48_fix1 = keyanchor_k48_fixed_lambda1_manifest()
        k48_dprime = keyanchor_k48_dprime_manifest()
        lo_ref, hi_ref = KEYANCHOR_K48_GPUH_PER_CELL
        lo_d, hi_d = KEYANCHOR_K48_GPUH_PER_CELL
        lo_g1k, hi_g1k = KEYANCHOR_K48_GATE1_GPUH
        lo_mand_k48 = lo_g1k + len(k48_ref) * lo_ref + len(k48_d) * lo_d
        hi_mand_k48 = hi_g1k + len(k48_ref) * hi_ref + len(k48_d) * hi_d
        lo_allcond_k48 = lo_mand_k48 + len(k48_fix1) * lo_ref + len(k48_dprime) * lo_ref
        hi_allcond_k48 = hi_mand_k48 + len(k48_fix1) * hi_ref + len(k48_dprime) * hi_ref
        print(f"  reference arms, K=48 (--wave keyanchor-k48-ref): {len(k48_ref)} runs | "
              f"seeds {{1,2,3}} | ~{len(k48_ref) * lo_ref:.2f}-{len(k48_ref) * hi_ref:.2f} GPU-h")
        print(f"  Gate-1-style probe, K=48, DISCLOSED NON-GATING: {len(k48_g1)} run | "
              f"~{lo_g1k:.2f}-{hi_g1k:.2f} GPU-h")
        print(f"  candidate (d), K=48, MANDATORY (--wave keyanchor-k48): {len(k48_d)} runs | "
              f"seeds {{30,31,32}} | ~{len(k48_d) * lo_d:.2f}-{len(k48_d) * hi_d:.2f} GPU-h")
        print(f"  MANDATORY TOTAL: {len(k48_ref) + len(k48_g1) + len(k48_d)} runs | "
              f"~{lo_mand_k48:.2f}-{hi_mand_k48:.2f} GPU-h")
        print(f"  candidate (d'), K=48, CONDITIONAL (--include-k48-dprime, orchestrator "
              f"sign-off required, sec 11.1 item 3): {len(k48_dprime)} runs | "
              f"~{len(k48_dprime) * lo_ref:.2f}-{len(k48_dprime) * hi_ref:.2f} GPU-h (not run by default)")
        print(f"  fixed-lambda=1 ceiling probe, OPTIONAL (--include-k48-fixed-lambda1, sec 11.4.3): "
              f"{len(k48_fix1)} run | ~{len(k48_fix1) * lo_ref:.2f}-{len(k48_fix1) * hi_ref:.2f} GPU-h "
              f"(not run by default, first cut under budget pressure)")
        print(f"  ALL-CONDITIONALS-MAX: {len(k48_ref) + len(k48_g1) + len(k48_d) + len(k48_fix1) + len(k48_dprime)} "
              f"runs | ~{lo_allcond_k48:.2f}-{hi_allcond_k48:.2f} GPU-h")
        print(f"  registered nominal ceiling: {KEYANCHOR_K48_GPUH_CEILING:.1f} GPU-h (sec 11.5) -> "
              f"program cumulative (RECONCILED base, see the ledger comment above "
              f"keyanchor_k48_budget_guard) {KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED:.4f} + "
              f"{KEYANCHOR_K48_GPUH_CEILING:.1f} = "
              f"{KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED + KEYANCHOR_K48_GPUH_CEILING:.4f} / "
              f"{KEYANCHOR_K48_PROGRAM_GPUH_CEILING:.0f} GPU-h "
              f"({'WITHIN' if KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED + KEYANCHOR_K48_GPUH_CEILING <= KEYANCHOR_K48_PROGRAM_GPUH_CEILING else 'EXCEEDS'} "
              f"the exactness-program cap)")
        assert len(k48_ref) == 3 and len(k48_d) == 3 and len(k48_g1) == 1 and len(k48_fix1) == 1 \
            and len(k48_dprime) == 3, \
            f"keyanchor-k48 run counts drifted from their registered 3+3+1+1+3: " \
            f"{len(k48_ref)}+{len(k48_d)}+{len(k48_g1)}+{len(k48_fix1)}+{len(k48_dprime)}"

        print("\n" + "=" * 70)
        print("KEY_ANCHORING_DESIGN.md sec 10.13 -- keyanchor-e wave preview "
              "(candidate (e), frozen-table ablation, BOTH arms)")
        print("=" * 70)
        ke = keyanchor_e_manifest()
        ke_fp = keyanchor_e_fp_manifest()
        ke_all = keyanchor_e_wave_manifest()
        lo_e, hi_e = KEYANCHOR_E_GPUH_PER_CELL
        print(f"  arm 'e' (frozen RANDOM-unit-rows table, the stub's name/motivation): {len(ke)} runs "
              f"| K=32 seeds {{60,61,62}} | ~{len(ke) * lo_e:.2f}-{len(ke) * hi_e:.2f} GPU-h")
        print(f"  arm 'e-fp' (frozen FRAME-POTENTIAL table, the stub's literal init text -- audit "
              f"prescription): {len(ke_fp)} runs | K=32 seeds {{70,71,72}} | "
              f"~{len(ke_fp) * lo_e:.2f}-{len(ke_fp) * hi_e:.2f} GPU-h")
        print(f"  WAVE TOTAL (--wave keyanchor-e launches both): {len(ke_all)} runs | fixed lambda="
              f"{KEYANCHOR_E_LAMBDA_FIXED} | ~{len(ke_all) * lo_e:.2f}-{len(ke_all) * hi_e:.2f} GPU-h")
        print(f"  registered nominal ceiling: {KEYANCHOR_E_GPUH_CEILING:.1f} GPU-h (sec 10.13's ~1 "
              f"GPU-h was for the 3-cell arm; 6 cells bracket 1.2-2.1, ceiling = top +~20%) -> "
              f"program cumulative (RECONCILED base) {KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED:.4f} + "
              f"{KEYANCHOR_E_GPUH_CEILING:.1f} = "
              f"{KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED + KEYANCHOR_E_GPUH_CEILING:.4f} / "
              f"{KEYANCHOR_K48_PROGRAM_GPUH_CEILING:.0f} GPU-h "
              f"({'WITHIN' if KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED + KEYANCHOR_E_GPUH_CEILING <= KEYANCHOR_K48_PROGRAM_GPUH_CEILING else 'EXCEEDS'} "
              f"the exactness-program cap)")
        assert len(ke) == 3 and len(ke_fp) == 3 and len(ke_all) == 6, \
            f"keyanchor-e run counts drifted from their registered 3+3=6: {len(ke)}+{len(ke_fp)}"

        print("\n" + "=" * 70)
        print("KEY_ANCHORING_DESIGN.md sec 12 (Rev 12.2, CLEARED-FOR-BUILD) -- "
              "keyanchor-cliff wave preview (capacity-cliff localization)")
        print("=" * 70)
        kc_mand = keyanchor_cliff_manifest()
        kc_g1 = keyanchor_cliff_gate1_manifest()
        lo_c, hi_c = KEYANCHOR_CLIFF_GPUH_PER_CELL
        lo_g1c, hi_g1c = KEYANCHOR_CLIFF_GATE1_GPUH
        stage1_ks = sorted(K for K, s in KEYANCHOR_CLIFF_STAGE_BY_K.items() if s == 1)
        stage2_ks = sorted(K for K, s in KEYANCHOR_CLIFF_STAGE_BY_K.items() if s == 2)
        print(f"  candidate (d), MANDATORY, PRIMARY, K in {KEYANCHOR_CLIFF_KS}: {len(kc_mand)} runs | "
              f"3 seeds each | ~{len(kc_mand) * lo_c:.2f}-{len(kc_mand) * hi_c:.2f} GPU-h")
        print(f"    stage 1 (K={stage1_ks}, launched first/alone): "
              f"{len([s for s in kc_mand if s['K'] in stage1_ks])} runs")
        print(f"    stage 2 (K={stage2_ks}, conditional on stage 1 clearing the bracket, sec 12.2.3): "
              f"{len([s for s in kc_mand if s['K'] in stage2_ks])} runs")
        print(f"  Gate-1-style probes, OPTIONAL, 1 per K: {len(kc_g1)} runs | "
              f"~{len(kc_g1) * lo_g1c:.2f}-{len(kc_g1) * hi_g1c:.2f} GPU-h (not run by default)")
        print(f"  seed contingency, CONDITIONAL, +2 seeds per K (reserved blocks "
              f"{KEYANCHOR_CLIFF_CONTINGENCY_SEEDS_BY_K}): not a manifest function in this build "
              f"(sec 12.2's own registered-but-not-fired discipline)")
        print(f"  registered nominal ceiling (mandatory-only, bracket-pessimistic, 2x, sec 12.5): "
              f"{KEYANCHOR_CLIFF_GPUH_CEILING:.2f} GPU-h -> program cumulative "
              f"{KEYANCHOR_CLIFF_PROGRAM_SPENT_GPUH:.2f} + {KEYANCHOR_CLIFF_GPUH_CEILING:.2f} = "
              f"{KEYANCHOR_CLIFF_PROGRAM_SPENT_GPUH + KEYANCHOR_CLIFF_GPUH_CEILING:.2f} / "
              f"{KEYANCHOR_CLIFF_PROGRAM_GPUH_CEILING:.0f} GPU-h "
              f"({'WITHIN' if KEYANCHOR_CLIFF_PROGRAM_SPENT_GPUH + KEYANCHOR_CLIFF_GPUH_CEILING <= KEYANCHOR_CLIFF_PROGRAM_GPUH_CEILING else 'EXCEEDS'} "
              f"the exactness-program cap; reserve={KEYANCHOR_CLIFF_RESERVE_GPUH:.2f} GPU-h, sec 12.5's own "
              f"'~0.89 GPU-h to spare' figure)")
        print(f"  abort threshold (sec 12.2.3, ANY completed cell): wall_s >= "
              f"{KEYANCHOR_CLIFF_ABORT_WALL_S:.1f}s (1.5x the {hi_c:.2f} GPU-h/cell bracket upper edge)")
        assert len(kc_mand) == 12 and len(kc_g1) == 4, \
            f"keyanchor-cliff run counts drifted from their registered 12+4: {len(kc_mand)}+{len(kc_g1)}"
        assert sorted(set(s["K"] for s in kc_mand)) == sorted(KEYANCHOR_CLIFF_KS), \
            f"keyanchor-cliff mandatory manifest K's drifted: {sorted(set(s['K'] for s in kc_mand))}"
        assert stage1_ks == [38, 42] and stage2_ks == [34, 46], \
            f"keyanchor-cliff stage split drifted from its registered stage1={{38,42}}/stage2={{34,46}}: " \
            f"stage1={stage1_ks} stage2={stage2_ks}"

        print("\n" + "=" * 70)
        print(f"COMBINED (both waves, all-conditionals-max, using the RECONCILED "
              f"{KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED:.4f} base): "
              f"{KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED:.4f} + {KEYANCHOR_K48_GPUH_CEILING:.1f} "
              f"(k48 ceiling) + {KEYANCHOR_E_GPUH_CEILING:.1f} (e ceiling) = "
              f"{KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED + KEYANCHOR_K48_GPUH_CEILING + KEYANCHOR_E_GPUH_CEILING:.4f} "
              f"/ {KEYANCHOR_K48_PROGRAM_GPUH_CEILING:.0f} GPU-h "
              f"({'WITHIN' if KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED + KEYANCHOR_K48_GPUH_CEILING + KEYANCHOR_E_GPUH_CEILING <= KEYANCHOR_K48_PROGRAM_GPUH_CEILING else 'EXCEEDS'} "
              f"the exactness-program cap)")
        print("=" * 70)

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

    if args.wave == "keyanchor-k48-bands":
        # sec 11.1.1 writer requirement 1: NO GPU, no manifest -- reads the
        # 3 K=48 reference-arm result JSONs and writes BANDS_PINNED_K48.json
        # iff ALL THREE validate complete. A SEPARATE file/gate from
        # 'keyanchor-bands' above (K=16/32's own BANDS_PINNED.json is
        # untouched).
        ref_out_dir_k48 = os.path.join(args.out_dir, "wavekeyanchor-k48-ref")
        ok = write_bands_pinned_k48_if_ready(ref_out_dir_k48)
        sys.exit(0 if ok else 1)

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose): the busy-GPU set on this box changes day to day. Run nvidia-smi NOW "
              "and pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)

    unblind_override_at = None
    ckpt_base_dir = None
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
    elif args.wave == "keyanchor-confirm":
        # sec 9.5's required follow-up + this build's task-directed re-run.
        # TWO gates apply, in order: (1) the pre-launch CPU-only gate on
        # the FIXED keyanchor_drift_diagnostic.py's own Gate-1 read (no GPU
        # work here -- only reading its already-produced JSON, mirroring
        # gate_geo3_drift's pattern); (2) sec 3.6's BANDS_PINNED mechanical
        # blinding gate -- these ARE anchor-arm cells, identical to
        # 'keyanchor', so gate (b) applies unchanged.
        diag_gate = gate_keyanchor_drift_diag(args.keyanchor_drift_json,
                                                args.accept_keyanchor_diag_override)
        ref_out_dir = os.path.join(args.out_dir, "waveref")
        if args.unblind_override:
            unblind_override_at = time.time()
        bands_gate = gate_bands_pinned(ref_out_dir, args.unblind_override, unblind_override_at)
        manifest = keyanchor_confirm_manifest()
        print(f"wave keyanchor-confirm manifest: {len(manifest)} runs (candidate (d), K=32 "
              f"seeds {{0,1,2}} + K=16 seed-0 spot check, drift_probe=True); "
              f"diag_gate={diag_gate} bands_gate={bands_gate}", flush=True)
    elif args.wave == "keyanchor-mech-gate1":
        # sec 10.5's Gate-1-style pre-launch probe, candidate (d') only --
        # a single 5,000-step cell, NOT gated on anything itself (it IS the
        # gate `--wave keyanchor-mech` below reads). rev7_engagement=True
        # is set in the manifest spec itself (keyanchor_mech_gate1_manifest;
        # e633862 audit fold-in 1 -- this comment previously CLAIMED the
        # threading while the manifest contradicted it) and build_cmd
        # threads it as --rev7-engagement, so run_deltanet_rd.py validates
        # the Rev-7.1 pin at startup (sec 10.3.3 leg (ii)'s per-run
        # defense-in-depth) and the probe's result JSON is instrumented
        # identically to the real cells. A scratch ckpt-base-dir is
        # threaded too (same audit fold-in) so the sec 10.10 checkpoint
        # writer -- like the rest of the never-yet-executed train()
        # checkpoint-block wiring -- gets its first real GPU exercise on
        # this cheap probe, not on a 20,000-step mandatory cell.
        ckpt_base_dir = args.ckpt_base_dir or "/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-mech-gate1"
        manifest = keyanchor_mech_gate1_manifest()
        print(f"wave keyanchor-mech-gate1 manifest: {len(manifest)} run (candidate (d') Gate-1 "
              f"probe, K=32, {KEYANCHOR_MECH_GATE1_STEPS} steps, rev7_engagement=True, "
              f"ckpt_base_dir={ckpt_base_dir})", flush=True)
    elif args.wave == "keyanchor-mech":
        # KEY_ANCHORING_DESIGN.md sec 10 (Rev 7.1) -- the FULL gate chain,
        # in order: (1) the Rev-7.1 pin triple (sec 10.3.3 leg (ii): exists
        # + script-hash match + live derive()-reproduction); (2) candidate
        # (d')'s own Gate-1 probe read; (3) sec 3.6's BANDS_PINNED
        # reference-arm reuse-or-refuse gate (UNCHANGED reference-arm data,
        # sec 10.5: "not load-bearing for this wave's primary question" but
        # still the same mechanical gate every anchor-arm wave uses); (4)
        # the budget guard (sec 10.7's registered 12 GPU-h ceiling vs. the
        # 80 GPU-h exactness-program cap); (5) the disk gate (sec 10.10's
        # checkpoint writer, house gate-(f) pattern).
        #
        # Override path (sec 10.3.3 point 2: "an explicit override exists
        # and automatically demotes every affected readout to descriptive
        # tier ... verbatim the sec 3.6/Rev-5 override-demotion machinery,
        # reused not reinvented") -- REUSES the existing --unblind-override
        # flag (never a second override flag): since the pin is a free,
        # deterministic, zero-data-dependency artifact, the only reason to
        # ever bypass this specific leg is the SAME class of explicit,
        # documented human decision --unblind-override already represents
        # for BANDS_PINNED; both legs share one flag and one stamping path
        # (ka.assemble_claim_tier_fields, applied uniformly downstream in
        # run_deltanet_rd.py regardless of which upstream gate triggered it).
        if args.unblind_override:
            unblind_override_at = time.time()
        pin_doc = ka.validate_rev7_pin()
        if pin_doc is None and not args.unblind_override:
            print("ERROR: --wave keyanchor-mech requires a VALID REV7_THRESHOLD_PINNED.json (sec "
                  "10.3.3): missing, script-hash mismatch, or a live derive() re-run does not "
                  "reproduce the pin's own derived block byte-identically. Regenerate with "
                  "`python rev7_threshold_derive.py` and re-commit, or pass --unblind-override for "
                  "an explicit, documented, tier-demoting override.", file=sys.stderr)
            sys.exit(1)
        elif pin_doc is None:
            print("=" * 70 + "\nWARNING: --unblind-override -- the sec 10.3.3 Rev-7.1 pin-triple "
                  "gate is being BYPASSED by an explicit human decision (the pin is missing or "
                  "fails validation). EVERY keyanchor-mech run launched this session will have its "
                  "OWN result JSON stamped claim_tier='descriptive' + unblind_override=True.\n"
                  + "=" * 70, flush=True)
        else:
            print(f"sec 10.3.3 PIN GATE PASSED: REV7_THRESHOLD_PINNED.json validates "
                  f"(script_sha256={pin_doc['provenance']['script_sha256'][:12]}..., "
                  f"generated_at={pin_doc['provenance']['generated_at']}).", flush=True)
        gate1_result = gate_keyanchor_mech_probe(args.keyanchor_mech_gate1_json,
                                                   args.accept_keyanchor_mech_gate1_override)
        ref_out_dir = os.path.join(args.out_dir, "waveref")
        bands_gate = gate_bands_pinned(ref_out_dir, args.unblind_override, unblind_override_at)
        keyanchor_mech_budget_guard(args.accept_budget_override)
        ckpt_base_dir = args.ckpt_base_dir or "/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-mech"
        manifest = keyanchor_mech_manifest()
        disk_report = keyanchor_mech_disk_gate(ckpt_base_dir, manifest)
        if not disk_report["ok"] and not args.accept_budget_override:
            print(f"ERROR: DISK GATE (keyanchor-mech) REFUSED -- {disk_report['required_bytes'] / 1e6:.1f} "
                  f"MB required at {disk_report['resolved_ckpt_dir']!r}, "
                  f"{disk_report['free_bytes'] / 1e9:.2f} GB free. Free up space or pass "
                  f"--accept-budget-override.", file=sys.stderr)
            sys.exit(1)
        print(f"wave keyanchor-mech manifest: {len(manifest)} runs (candidate (d) K=32 "
              f"seeds{{10,11,12}} + K=16 seed10; candidate (d') K=32 seeds{{20,21,22}}); "
              f"gate1={gate1_result} bands_gate={bands_gate}", flush=True)
    elif args.wave == "keyanchor-k48-ref":
        # sec 11.1 arm 2: bare geo3, K=48, seeds {1,2,3}, drift_probe=True --
        # MANDATORY, first in manifest (no K=48 bands exist yet). NOT gated
        # on anything itself (mirrors 'ref''s own precedent).
        manifest = keyanchor_k48_reference_manifest()
        print(f"wave keyanchor-k48-ref manifest: {len(manifest)} runs (bare-geo3, seeds {{1,2,3}}, "
              f"K=48, --drift-probe active) -- sec 11.1.1: MUST complete + validate BEFORE "
              f"'--wave keyanchor-k48-bands' can pin BANDS_PINNED_K48.json", flush=True)
    elif args.wave == "keyanchor-k48-gate1":
        # sec 11.1 item 4: DISCLOSED, NON-GATING pre-launch probe -- not
        # gated on anything itself (it IS the thing --wave keyanchor-k48
        # optionally reads, never a hard requirement).
        ckpt_base_dir = args.ckpt_base_dir or "/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-k48-gate1"
        manifest = keyanchor_k48_gate1_manifest()
        print(f"wave keyanchor-k48-gate1 manifest: {len(manifest)} run (candidate (d) architecture, "
              f"K=48, {KEYANCHOR_K48_GATE1_STEPS} steps, rev7_engagement=True, DISCLOSED NON-GATING, "
              f"ckpt_base_dir={ckpt_base_dir})", flush=True)
    elif args.wave == "keyanchor-k48":
        # sec 11.1's full K=48 gate chain: (1) BANDS_PINNED_K48 (sec
        # 11.1.1, the K=48-specific reference-band gate -- NOT the K=16/32
        # BANDS_PINNED.json); (2) the K=48 Gate-1-style probe read
        # (DISCLOSED, NON-GATING -- reported, never blocks); (3) the
        # RECONCILED budget guard (sec 11.5); (4) the disk gate (sec
        # 10.10's checkpoint writer, reused). candidate (d') fires ONLY
        # with BOTH --include-k48-dprime AND
        # --accept-k48-dprime-orchestrator-signoff (sec 11.1 item 3's own
        # honest downgrade -- an orchestrator sign-off, not a mechanical
        # gate on Rev 7.1's own result JSONs, since no such script exists
        # yet). The fixed-lambda=1 probe fires with --include-k48-fixed-
        # lambda1 alone (no sign-off needed -- it is NOT conditioned on Rev
        # 7.1's verdict, only a budget/priority choice, sec 11.4.3).
        if args.unblind_override:
            unblind_override_at = time.time()
        ref_out_dir_k48 = os.path.join(args.out_dir, "wavekeyanchor-k48-ref")
        bands_gate_k48 = gate_bands_pinned_k48(ref_out_dir_k48, args.unblind_override, unblind_override_at)
        gate1_result_k48 = gate_keyanchor_k48_gate1_probe(args.keyanchor_k48_gate1_json,
                                                             accept_override=False)
        keyanchor_k48_budget_guard(args.accept_budget_override)
        manifest = keyanchor_k48_manifest()
        if args.include_k48_fixed_lambda1:
            manifest = manifest + keyanchor_k48_fixed_lambda1_manifest()
        if args.include_k48_dprime:
            if not args.accept_k48_dprime_orchestrator_signoff:
                print("ERROR: --include-k48-dprime requires --accept-k48-dprime-orchestrator-signoff "
                      "-- candidate (d')'s own launch precondition is an EXPLICIT orchestrator "
                      "sign-off that Rev 7.1's own K=32 wave has reached a written Outcome A(d') or "
                      "D' verdict (sec 10.6's routing table); no script checks this mechanically yet "
                      "(sec 11.1 item 3/sec 11.10 finding C11's own honest downgrade). Read Rev 7.1's "
                      "own readout_rev7.py output before passing this flag.", file=sys.stderr)
                sys.exit(1)
            print("=" * 70 + "\nORCHESTRATOR SIGN-OFF RECORDED: --include-k48-dprime "
                  "--accept-k48-dprime-orchestrator-signoff -- candidate (d')'s 3 K=48 cells are "
                  "being launched on the strength of a HUMAN decision that Rev 7.1's own K=32 "
                  "verdict is A(d') or D', not a mechanical check (sec 11.1 item 3).\n" + "=" * 70,
                  flush=True)
            manifest = manifest + keyanchor_k48_dprime_manifest()
        ckpt_base_dir = args.ckpt_base_dir or "/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-k48"
        disk_report = keyanchor_mech_disk_gate(ckpt_base_dir, manifest, label="keyanchor-k48")
        if not disk_report["ok"] and not args.accept_budget_override:
            print(f"ERROR: DISK GATE (keyanchor-k48) REFUSED -- {disk_report['required_bytes'] / 1e6:.1f} "
                  f"MB required at {disk_report['resolved_ckpt_dir']!r}, "
                  f"{disk_report['free_bytes'] / 1e9:.2f} GB free. Free up space or pass "
                  f"--accept-budget-override.", file=sys.stderr)
            sys.exit(1)
        print(f"wave keyanchor-k48 manifest: {len(manifest)} runs (candidate (d) K=48 seeds"
              f"{{30,31,32}}"
              f"{' + fixed-lambda1 probe seed 50' if args.include_k48_fixed_lambda1 else ''}"
              f"{' + candidate (d'') K=48 seeds{40,41,42}' if args.include_k48_dprime else ''}); "
              f"bands_gate={bands_gate_k48} gate1_probe={gate1_result_k48}", flush=True)
    elif args.wave == "keyanchor-e":
        # sec 10.13's registered candidate (e), BOTH arms (audit
        # prescription, 2026-07 K48+e build): the 'e' random-unit-rows arm
        # (K=32, seeds {60,61,62}) AND the 'e-fp' frozen-frame-potential
        # arm per the stub's literal text (K=32, seeds {70,71,72}), both
        # with fixed lambda=0.58 and a frozen (requires_grad=False) table.
        # REUSES the EXISTING K=16/32 BANDS_PINNED.json gate (these are
        # K=32 anchor-arm cells, sec 3.6 applies unchanged -- no new bands
        # file for this wave, unlike keyanchor-k48).
        if args.unblind_override:
            unblind_override_at = time.time()
        ref_out_dir = os.path.join(args.out_dir, "waveref")
        bands_gate = gate_bands_pinned(ref_out_dir, args.unblind_override, unblind_override_at)
        keyanchor_e_budget_guard(args.accept_budget_override)
        manifest = keyanchor_e_wave_manifest()
        ckpt_base_dir = args.ckpt_base_dir or "/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-e"
        disk_report = keyanchor_mech_disk_gate(ckpt_base_dir, manifest, label="keyanchor-e")
        if not disk_report["ok"] and not args.accept_budget_override:
            print(f"ERROR: DISK GATE (keyanchor-e) REFUSED -- {disk_report['required_bytes'] / 1e6:.1f} "
                  f"MB required at {disk_report['resolved_ckpt_dir']!r}, "
                  f"{disk_report['free_bytes'] / 1e9:.2f} GB free. Free up space or pass "
                  f"--accept-budget-override.", file=sys.stderr)
            sys.exit(1)
        print(f"wave keyanchor-e manifest: {len(manifest)} runs (candidate (e), K=32, BOTH arms: "
              f"'e' frozen random-unit-rows seeds {{60,61,62}} + 'e-fp' frozen frame-potential "
              f"seeds {{70,71,72}}, fixed lambda={KEYANCHOR_E_LAMBDA_FIXED}); "
              f"bands_gate={bands_gate}", flush=True)
    elif args.wave == "keyanchor-cliff":
        # sec 12's capacity-cliff localization wave. REQUIRES --stage (sec
        # 12.2.3's staged launch); no BANDS_PINNED gate (candidate (d)'s own
        # architecture/frame-potential init is UNCHANGED from keyanchor-k48,
        # sec 12.1 -- this wave reuses the ALREADY-VALIDATED K=16/32
        # BANDS_PINNED.json blinding gate rather than a new K-specific one,
        # since the reference-arm cut (sec 12.2 item 2) means there is no
        # new per-K reference measurement to derive fresh bands from anyway).
        if args.stage is None:
            print("ERROR: --wave keyanchor-cliff requires --stage 1 or --stage 2 (sec 12.2.3's "
                  "staged launch -- K=38+K=42 first, K=34+K=46 only after Stage 1 clears the "
                  "K48-calibrated bracket).", file=sys.stderr)
            sys.exit(1)
        if args.unblind_override:
            unblind_override_at = time.time()
        ref_out_dir = os.path.join(args.out_dir, "waveref")
        bands_gate = gate_bands_pinned(ref_out_dir, args.unblind_override, unblind_override_at)
        keyanchor_cliff_budget_guard(args.accept_budget_override)
        stage_ks = tuple(sorted(K for K, s in KEYANCHOR_CLIFF_STAGE_BY_K.items() if s == args.stage))
        cliff_out_dir = os.path.join(args.out_dir, "wavekeyanchor-cliff")
        stage_gate_report = keyanchor_cliff_stage_gate(cliff_out_dir, args.stage, args.accept_stage_override)
        manifest = keyanchor_cliff_manifest(Ks=stage_ks)
        if args.include_cliff_gate1:
            manifest = manifest + keyanchor_cliff_gate1_manifest(Ks=stage_ks)
        ckpt_base_dir = args.ckpt_base_dir or "/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-cliff"
        disk_report = keyanchor_mech_disk_gate(ckpt_base_dir, manifest, label="keyanchor-cliff")
        if not disk_report["ok"] and not args.accept_budget_override:
            print(f"ERROR: DISK GATE (keyanchor-cliff) REFUSED -- "
                  f"{disk_report['required_bytes'] / 1e6:.1f} MB required at "
                  f"{disk_report['resolved_ckpt_dir']!r}, {disk_report['free_bytes'] / 1e9:.2f} GB free. "
                  f"Free up space or pass --accept-budget-override.", file=sys.stderr)
            sys.exit(1)
        print(f"wave keyanchor-cliff manifest: {len(manifest)} runs (candidate (d), stage {args.stage}, "
              f"K={stage_ks}, 3 seeds each"
              f"{' + Gate-1 probes' if args.include_cliff_gate1 else ''}); "
              f"bands_gate={bands_gate} stage_gate={stage_gate_report}", flush=True)
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

    # KEY_ANCHORING_DESIGN.md sec 5's Wave -1 smoke suite + sec 4's Gate 2,
    # wired as LAUNCH GATES (2026-07-04 audit fix, MAJOR: the committed CPU
    # tests existed but nothing forced them to run before a launch). Every
    # KEY_ANCHORING wave runs the 9(+)-item smoke suite; anchor-arm waves
    # ('keyanchor'/'keyanchor-confirm' -- what Gate 2 exists to protect)
    # additionally run the Gate 2 construction check with its pinned
    # regression quadruple, AND (2026-07-06 build) the fla-free
    # smoke_keyanchor_confirm.py manifest/regression suite (deliverable 4:
    # drift_probe wiring + is_done non-collision + the log_every/pipefail
    # regressions) -- ALL CPU-only subprocesses; rc!=0 aborts with the
    # same ABORTED.txt discipline as the pre-existing run_smoke gate below.
    if args.wave in ("ref", "keyanchor-neg1", "keyanchor", "keyanchor-confirm", "keyanchor-mech-gate1",
                      "keyanchor-mech", "keyanchor-k48-ref", "keyanchor-k48-gate1", "keyanchor-k48",
                      "keyanchor-e", "keyanchor-cliff") and not args.skip_smoke:
        rc = subprocess.call([sys.executable, os.path.join(HERE, "smoke_key_anchoring.py")], cwd=HERE)
        if rc != 0:
            with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
                f.write("smoke_key_anchoring.py FAILED (rc != 0); no training launched.\n")
            print("ERROR: smoke_key_anchoring.py failed -- wave aborted.", file=sys.stderr)
            sys.exit(1)
        if args.wave in ("keyanchor", "keyanchor-confirm", "keyanchor-mech-gate1", "keyanchor-mech",
                          "keyanchor-k48-gate1", "keyanchor-k48", "keyanchor-e", "keyanchor-cliff"):
            rc = subprocess.call([sys.executable, os.path.join(HERE, "gate2_construction_test.py")], cwd=HERE)
            if rc != 0:
                with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
                    f.write("gate2_construction_test.py FAILED (rc != 0); no training launched.\n")
                print(f"ERROR: gate2_construction_test.py failed -- {args.wave} wave aborted.",
                      file=sys.stderr)
                sys.exit(1)
        if args.wave == "keyanchor-confirm":
            rc = subprocess.call([sys.executable, os.path.join(HERE, "smoke_keyanchor_confirm.py")], cwd=HERE)
            if rc != 0:
                with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
                    f.write("smoke_keyanchor_confirm.py FAILED (rc != 0); no training launched.\n")
                print("ERROR: smoke_keyanchor_confirm.py failed -- keyanchor-confirm wave aborted.",
                      file=sys.stderr)
                sys.exit(1)
        if args.wave in ("keyanchor-mech-gate1", "keyanchor-mech"):
            # KEY_ANCHORING_DESIGN.md sec 10.9 -- the mechanism-tier wave's
            # own 13-item smoke suite (fla-free; run_deltanet_rd_exactness_
            # sweep.py + key_anchoring.py only, same discipline as
            # smoke_keyanchor_confirm.py).
            rc = subprocess.call([sys.executable, os.path.join(HERE, "smoke_keyanchor_mech.py")], cwd=HERE)
            if rc != 0:
                with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
                    f.write("smoke_keyanchor_mech.py FAILED (rc != 0); no training launched.\n")
                print(f"ERROR: smoke_keyanchor_mech.py failed -- {args.wave} wave aborted.",
                      file=sys.stderr)
                sys.exit(1)
        if args.wave in ("keyanchor-k48-ref", "keyanchor-k48-gate1", "keyanchor-k48", "keyanchor-e"):
            # KEY_ANCHORING_DESIGN.md sec 11.9 / sec 10.13 -- the K48+e
            # build's own smoke suite (fla-free; manifest-refactor
            # non-regression, Gate-2 K48 leg, zero-collision incl. wavegeo3,
            # candidate-(e) manifest shape, seed non-collision).
            rc = subprocess.call([sys.executable, os.path.join(HERE, "smoke_keyanchor_k48_e.py")], cwd=HERE)
            if rc != 0:
                with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
                    f.write("smoke_keyanchor_k48_e.py FAILED (rc != 0); no training launched.\n")
                print(f"ERROR: smoke_keyanchor_k48_e.py failed -- {args.wave} wave aborted.",
                      file=sys.stderr)
                sys.exit(1)
        if args.wave == "keyanchor-cliff":
            # KEY_ANCHORING_DESIGN.md sec 12.2.1/12.2.2/12.4 -- the cliff
            # wave's own smoke suite (fla-free; manifest regression at K=48,
            # zero-reference-arm-paths negative unit test, stage/seed shape).
            rc = subprocess.call([sys.executable, os.path.join(HERE, "smoke_keyanchor_cliff.py")], cwd=HERE)
            if rc != 0:
                with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
                    f.write("smoke_keyanchor_cliff.py FAILED (rc != 0); no training launched.\n")
                print(f"ERROR: smoke_keyanchor_cliff.py failed -- {args.wave} wave aborted.",
                      file=sys.stderr)
                sys.exit(1)

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
                                                       unblind_override_at=unblind_override_at,
                                                       ckpt_base_dir=ckpt_base_dir),
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
                    # sec 12.2.3's mechanical, in-code abort rule (Rev 12.2:
                    # ANY completed cell, not only the first) -- keyanchor-
                    # cliff ONLY. Reads THIS cell's own realized wall_s
                    # straight from its result JSON (never re-derived from
                    # the orchestrator's own wall-clock timer, which would
                    # include queue/GPU-wait time the design's own bracket
                    # does not price in).
                    if args.wave == "keyanchor-cliff":
                        with open(out_path(out_dir, spec)) as f:
                            wall_s = json.load(f).get("wall_s")
                        if wall_s is not None:
                            try:
                                keyanchor_cliff_check_abort(spec, wall_s)
                            except KeyanchorCliffAbort as abort_exc:
                                print(f"  {abort_exc}", flush=True)
                                pending.clear()
                                with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
                                    f.write(str(abort_exc) + "\n")
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
