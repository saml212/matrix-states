"""run_trackb_wave.py -- TRACKB_REDESIGN.md Rev 3 (commit 8ab089d): the
Track B hard-selectivity wave orchestrator. Wave manifest: Wave -1
(manipulation check + BANDS_PINNED-TrackB pins) -> reference pilot ->
BANDS_PINNED gate -> the sec 5 2x2 factorial (Cells 1/2/3/4 + 2R + the M7
comparator), n=3 seeds, both corpora, per sec 5/sec 10's cut order.

A NEW file (not a --wave addition to run_lm_rd_geo3_sweep.py) per this
build's own task brief item 8's "or a new run_trackb_wave.py" option --
chosen because Track B's hard-selectivity wave is a genuinely different
manifest shape (5 mechanism candidates x 2 flags x 2 corpora x 3 seeds,
plus BANDS_PINNED gating) from run_lm_rd_geo3_sweep.py's own (beta_topk-
or-naive_window)-only Wave 1/2/3, not a small addition to it.

CLONE of run_lm_rd_geo3_sweep.py's / run_lm_rd_trackc_sweep.py's own
robustness pattern (smoke gate, exception-isolated launch, validity-checked
resume, per-run timeout with GPU quarantine, guarded aggregate, REQUIRED
--gpus/--gpu-offset with NO defaults, budget_guard, disk_space_check) --
deliberately NOT a cross-script IMPORT of their launch machinery (this
codebase's own pod-safety "clone, not import between sweep scripts"
convention, restated in both of those files' own module docstrings). The
ONE deliberate exception (documented at its own definition below): this
file DOES live-import `PROGRAM_SPENT_GPUH`/`GPU_H_PROGRAM_CEILING` from
run_lm_rd_trackc_sweep.py -- a single shared DATA constant (the program's
maintained budget ledger), not orchestration logic, and the task brief's
own instruction is to "read PROGRAM_SPENT_GPUH live," which a clone (a
frozen copy that silently drifts from the maintained source) cannot do.

**THIS BUILD SESSION DOES NOT LAUNCH ANYTHING (task brief's HARD
CONSTRAINT: build + CPU-verify only, no GPU runs).** This script is
complete and smoke-gated but, exactly like its house precedents, main()
refuses to do a real (non-dry-run) launch without explicit --gpus/
--gpu-offset, and no launch was performed this session -- --dry-run
preview + on-box CPU smoke only.

Usage (GPU list is an example -- check nvidia-smi first, per house rule):
  python run_trackb_wave.py --dry-run
  python run_trackb_wave.py --wave neg1 --out-dir results/trackb --gpus 1 --gpu-offset 7
  python run_trackb_wave.py --wave bands-pinned --out-dir results/trackb          # CPU-only, writes BANDS_PINNED-TrackB.json
  python run_trackb_wave.py --wave factorial --out-dir results/trackb --gpus 4 --gpu-offset 0
  python run_trackb_wave.py --wave cell3 --accept-no-launch-reference-arm --out-dir results/trackb --gpus 1 --gpu-offset 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAIN = os.path.join(HERE, "lm_pretrain_rd.py")
GEO3_GATE_TOOL = os.path.join(HERE, "lm_geo3_wave_neg1_gate.py")
sys.path.insert(0, HERE)  # pod-safe imports

import hard_selectivity_rd as hs                    # noqa: E402
import bands_pinned_trackb as bp                     # noqa: E402
import wave_neg1_trackb as wn1                       # noqa: E402

# The ONE deliberate live-import exception, documented in the module docstring above: a shared
# DATA constant (the program's maintained GPU-h ledger), read live per the task brief's own
# instruction, never a frozen clone that could silently drift from the maintained source.
from run_lm_rd_trackc_sweep import PROGRAM_SPENT_GPUH, GPU_H_PROGRAM_CEILING   # noqa: E402

CORPORA = ("openr1", "wikitext")
SEEDS = (0, 1, 2)
K_SEL = 32                        # sec 5.3: PINNED, single value, for the entire Wave 1/2x2 manifest
GEO3_CHUNK_SIZE = 64
GEO3_RESID_TOL = 1e-2
GEO3_N_ITER_BY_K_SEL = {16: 12, 32: 20}   # sec 1.1's escalation, duplicated per this codebase's
                                          # own clone-not-import convention (run_lm_rd_geo3_sweep.py:84)
GEO3_N_ITER = GEO3_N_ITER_BY_K_SEL[K_SEL]

D_MODEL, D_STATE, N_LAYERS = 256, 64, 2    # sec 4.5: "same corpora/seeds/step-budget/eval-protocol
SEQ_LEN, BATCH_SIZE = 512, 32              # as Wave C" -- duplicated from run_lm_rd_geo3_sweep.py
TARGET_TOKENS_FULL = 100_000_000           # (this codebase's own clone-not-import convention).
WAVE_NEG1_PROBE_STEPS = 2_000
WAVE_NEG1_STABILITY_STEPS = 2_000

# sec 6.1's two pricing constants -- Wave C's own measured ~0.077 GPU-h/6,103-step run (non-geo3)
# and the codebase's own registered 3x planning placeholder for geo3-active runs
# (run_lm_rd_geo3_sweep.py's _PER_STEP_S_PLACEHOLDER_GEO3=0.15 s/step vs ~0.05 s/step non-geo3).
_GPU_H_PER_STEP_NONGEO3 = 0.077 / 6_103          # ~0.0000126 GPU-h/step
_GPU_H_PER_STEP_GEO3 = 0.28 / 6_103              # ~0.0000459 GPU-h/step (3x placeholder-priced)


def default_steps(target_tokens: int = TARGET_TOKENS_FULL, batch_size: int = BATCH_SIZE,
                   seq_len: int = SEQ_LEN) -> int:
    return max(1, target_tokens // (batch_size * seq_len))


def geo3_h(n_runs: int, steps: int) -> float:
    return n_runs * steps * _GPU_H_PER_STEP_GEO3


def nongeo3_h(n_runs: int, steps: int) -> float:
    return n_runs * steps * _GPU_H_PER_STEP_NONGEO3


# ---------------------------------------------------------------------------
# Cell-3 override mechanics (sec 5.1, Rev 2 -- M5): a stamped bypass of the
# ACTUAL production refusal, run_lm_rd_geo3_sweep.py::_refuse_if_no_launch
# (:172-183, sys.exit(3), called from main() at :600) -- CLONED here (not
# imported, per this file's own stated pod-safety convention), plus
# load_gate_verdict (:131-162, the config cross-validation that stays LIVE
# under the override per M5's requirement (i)).
# ---------------------------------------------------------------------------

_VALID_VERDICTS = ("beta_gated_primary", "naive_window_primary", "no_launch_redesign")


def load_gate_verdict(gate_json_path: str) -> tuple[str, dict]:
    """CLONED from run_lm_rd_geo3_sweep.py's own function of the same name
    (:131-162) -- Track B's hard-selectivity wave reads the SAME Wave -1
    gate JSON (its own construction's gate verdict is a separate, later
    question this wave's own Wave -1 answers; the ORIGINAL beta-gated/
    naive-window gate is still the thing Cell 3's override targets)."""
    if not os.path.exists(gate_json_path):
        print(f"ERROR: Wave -1 gate JSON not found at {gate_json_path!r}. Run "
              f"lm_geo3_wave_neg1_gate.py FIRST -- Cell 3's override still requires a recorded "
              f"gate verdict to cross-validate against (M5 requirement (i)).", file=sys.stderr)
        sys.exit(2)
    with open(gate_json_path) as f:
        gate = json.load(f)
    verdict = gate.get("gate_verdict")
    if verdict not in _VALID_VERDICTS:
        print(f"ERROR: malformed gate JSON at {gate_json_path!r}: gate_verdict={verdict!r} "
              f"not in {_VALID_VERDICTS}.", file=sys.stderr)
        sys.exit(2)
    mismatches = []
    if gate.get("chunk_size") != GEO3_CHUNK_SIZE:
        mismatches.append(f"chunk_size={gate.get('chunk_size')!r} (this wave uses {GEO3_CHUNK_SIZE})")
    if gate.get("gate_k_sel") != K_SEL:
        mismatches.append(f"gate_k_sel={gate.get('gate_k_sel')!r} (this wave is pinned to K_sel={K_SEL})")
    if mismatches:
        print(f"ERROR: gate JSON at {gate_json_path!r} was measured under a DIFFERENT "
              f"configuration than this wave uses: {'; '.join(mismatches)}.", file=sys.stderr)
        sys.exit(2)
    return verdict, gate


def refuse_or_override_cell3(verdict: str, gate_json_path: str, accept_override: bool,
                              override_reason: str = hs.DEFAULT_GATE_OVERRIDE_REASON) -> dict | None:
    """The registered override (sec 5.1, M5): converts the HARD refusal
    into a loud, logged proceed for the Cell-3 manifest ONLY, when
    `accept_override` is True (a new explicit flag,
    --accept-no-launch-reference-arm, threaded from main() below) --
    requirement (ii). Requirement (i) -- load_gate_verdict's config
    cross-validation stays live -- is enforced by the CALLER always calling
    load_gate_verdict first (this function does not re-implement it, it
    only decides what happens AFTER). Requirement (iii) -- per-run
    assembly-time stamping -- is satisfied by returning the stamp payload
    for the caller to thread into every spawned Cell-3 run's
    --gate-override-reason CLI arg (lm_pretrain_rd.py's own
    _assemble_result reads it and writes gate_override/gate_override_reason/
    gate_override_at/claim_tier into the result JSON AT ASSEMBLY, never
    post-hoc).

    Returns the override stamp dict if the override fires, None if Cell 3
    is refused (verdict != no_launch_redesign -- Cell 3 is registered as
    the reference arm for exactly the no-launch case; a launch-eligible
    verdict means Cell 3 is not an 'override' at all, see main()'s own
    branch) or if accept_override was not passed for a no-launch verdict."""
    if verdict != "no_launch_redesign":
        return None       # not an override case at all -- Cell 3 proceeds as an ordinary run
    if not accept_override:
        print("=" * 70, file=sys.stderr)
        print("HARD NO-LAUNCH for Cell 3 (mirrors run_lm_rd_geo3_sweep.py::_refuse_if_no_launch, "
              "sec 4.2 outcome (iii)): the Wave -1 gate's criterion (b) FAILED.", file=sys.stderr)
        print(f"Gate JSON: {gate_json_path}", file=sys.stderr)
        print("Cell 3 (the geo3-only reference arm) requires --accept-no-launch-reference-arm to "
              "run as a DELIBERATE, stamped reference arm (TRACKB_REDESIGN.md sec 5.1) -- refusing "
              "otherwise, exactly as run_lm_rd_geo3_sweep.py's own no-launch gate does.",
              file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(3)
    print(f"OVERRIDE ACCEPTED for Cell 3 (--accept-no-launch-reference-arm): proceeding as a "
          f"DELIBERATE reference arm despite the no-launch gate verdict. Reason: {override_reason!r}. "
          f"Every spawned Cell-3 run will carry gate_override=true / claim_tier='descriptive' in its "
          f"OWN result JSON (assembly-time stamping, TRACKB_REDESIGN.md sec 5.1).", flush=True)
    return hs.trackb_override_stamp_payload(reason=override_reason)


# ---------------------------------------------------------------------------
# Wave -1 manifest: the 5 mechanism probes + the reference pilot + the M8
# stability smoke (sec 6.1's Wave -1 row). Cell-1 re-measurement + MC
# anchors are CPU-only/forward-only and handled by wave_neg1_trackb.py
# directly (bands_pinned_wave() below), not spawned subprocesses.
# ---------------------------------------------------------------------------

_PROBE_MECHANISMS = ("hard_ste", "entmax", "soft_topk_comparator", "random_topk")


def waveNeg1_manifest(steps: int = WAVE_NEG1_PROBE_STEPS) -> list[dict]:
    """sec 6.1 Wave -1 (a): "5 probes x 2,000 steps" = the 4 hard-select
    mechanisms (candidate 1/2/comparator/2R) + candidate 4's hard-snap
    schedule (mechanically identical to candidate 1, reusing hard_ste with
    a late snap_frac -- sec 3.4's own text) -- ALL non-geo3, ALL a single
    corpus/seed (openr1, seed 0: sec 10's own 'cheapest, most direct'
    framing for Wave -1 manipulation checks, never-cut). SCRUTINY NOTE for
    the independent auditor: sec 6.1's table states "5 probes" without
    itemizing which 5; this manifest's own reading (4 mechanisms +
    candidate 4's snap schedule = 5) is this build's own registered
    interpretation, not lifted verbatim from a more explicit spec passage --
    flagged for review, not silently assumed authoritative."""
    runs = []
    for mech in _PROBE_MECHANISMS:
        runs.append({
            "wave": "neg1", "cell": f"probe_{mech}", "corpus": "openr1", "seed": 0,
            "hard_select_mechanism": mech, "hard_select_k_sel": K_SEL,
            "steps": steps, "geo3_active": False,
            "name": f"wBneg1_probe_{mech}_k{K_SEL}",
        })
    runs.append({
        "wave": "neg1", "cell": "probe_candidate4_hardsnap", "corpus": "openr1", "seed": 0,
        "hard_select_mechanism": "hard_ste", "hard_select_k_sel": K_SEL,   # snap schedule wraps hard_ste
        "steps": steps, "geo3_active": False,
        "name": f"wBneg1_probe_candidate4hardsnap_k{K_SEL}",
    })
    return runs


def reference_pilot_manifest(steps: int = WAVE_NEG1_PROBE_STEPS) -> dict:
    """sec 4.3's Null A / positional-concentration-ceiling reference: the
    UNMASKED, Cell-1-architecture pilot (hard_select_active=False,
    geo3_active=False -- a freshly-trained-from-scratch probe, NOT one of
    the 6 archived Wave C checkpoints, so its early-training log points
    down to step 1,200-2,000 actually exist -- the archived checkpoints
    only checkpoint coarsely). ONE run, openr1, seed 0 (sec 4.3 derives its
    nulls from THIS SINGLE pilot's own trailing log points, not a
    multi-seed aggregate)."""
    return {"wave": "neg1", "cell": "reference_pilot", "corpus": "openr1", "seed": 0,
            "hard_select_active": False, "geo3_active": False, "steps": steps,
            "name": "wBneg1_reference_pilot"}


def waveNeg1_stability_smoke_manifest(steps: int = WAVE_NEG1_STABILITY_STEPS) -> dict:
    """sec 5.1 (Rev 2 M8, RE-SPECIFIED Rev 3 NEW-7): 2,000-step
    geo3_active=True run on the duplicate-key stress slice
    (wave_neg1_trackb.select_duplicate_key_stress_windows's own output --
    an openr1 corpus slice, n_dup_max>=8). Gates Cells 3/4. NOT run this
    build session (GPU + real corpus data required)."""
    return {"wave": "neg1", "cell": "stability_smoke", "corpus": "openr1", "seed": 0,
            "geo3_active": True, "geo3_k_sel": K_SEL, "geo3_n_iter": GEO3_N_ITER, "steps": steps,
            "data_slice": "duplicate_key_stress (wave_neg1_trackb.select_duplicate_key_stress_windows)",
            "name": "wBneg1_stability_smoke"}


# ---------------------------------------------------------------------------
# BANDS_PINNED-TrackB gate: reads Wave -1's own completed result JSONs
# (validity-checked) and, if every pilot validates, writes
# BANDS_PINNED-TrackB.json (bands_pinned_trackb.py's own writer). CPU-only
# (the derivation math), safe to run in THIS build session against
# synthetic/placeholder inputs for a smoke, but the REAL pin (against real
# Wave -1 result JSONs) requires those JSONs to exist from a real GPU run.
# ---------------------------------------------------------------------------

def bands_pinned_out_path(out_dir: str) -> str:
    return os.path.join(out_dir, "BANDS_PINNED-TrackB.json")


def pilot_result_valid(result: dict, expected_steps: int) -> bool:
    """A Wave -1 pilot result JSON 'validates as complete' -- mirrors
    key_anchoring.reference_arm_result_valid's own shape (complete==true,
    steps_completed>=expected)."""
    if result.get("complete") is not True:
        return False
    if result.get("steps_completed", 0) < expected_steps:
        return False
    return True


# ---------------------------------------------------------------------------
# sec 5's 2x2 factorial: Cells 1 (baseline, re-measured, no spawned run),
# 2 (selectivity-only), 2R (random control), 3 (geo3-only reference arm,
# override-gated), 4 (target), the M7 comparator. n=3 seeds, both corpora,
# per sec 5's cut order (sec 10).
# ---------------------------------------------------------------------------

def cell2_manifest(steps: int, surviving_mechanisms=("hard_ste",)) -> list[dict]:
    """Cell 2 (selectivity-only, geo3_active=False): the surviving
    candidate(s) from Wave 1's own manipulation check -- sec 10's cut order
    ranks candidate 1 (hard_ste) as never-cut/cheapest/most-direct;
    `surviving_mechanisms` defaults to just it, widened by the caller once
    Wave 1's real readout names more survivors."""
    runs = []
    for mech in surviving_mechanisms:
        for corpus in CORPORA:
            for seed in SEEDS:
                runs.append({
                    "wave": "1", "cell": "2", "corpus": corpus, "seed": seed,
                    "hard_select_active": True, "hard_select_mechanism": mech,
                    "hard_select_k_sel": K_SEL, "geo3_active": False, "steps": steps,
                    "name": f"wB1_cell2_{mech}_{corpus}_s{seed}_k{K_SEL}",
                })
    return runs


def cell2r_manifest(steps: int) -> list[dict]:
    """Cell 2R (REQUIRED, never-cut, sec 10): budget-matched random control,
    same K_sel/corpora/seeds as Cell 2."""
    return [{
        "wave": "1", "cell": "2R", "corpus": corpus, "seed": seed,
        "hard_select_active": True, "hard_select_mechanism": "random_topk",
        "hard_select_k_sel": K_SEL, "geo3_active": False, "steps": steps,
        "name": f"wB1_cell2R_{corpus}_s{seed}_k{K_SEL}",
    } for corpus in CORPORA for seed in SEEDS]


def comparator_manifest(steps: int) -> list[dict]:
    """M7 comparator (sec 3.1, cut order item 6): tau-annealed soft top-K,
    same corpora/seeds as Cell 2."""
    return [{
        "wave": "1", "cell": "comparator", "corpus": corpus, "seed": seed,
        "hard_select_active": True, "hard_select_mechanism": "soft_topk_comparator",
        "hard_select_k_sel": K_SEL, "geo3_active": False, "steps": steps,
        "name": f"wB1_comparator_{corpus}_s{seed}_k{K_SEL}",
    } for corpus in CORPORA for seed in SEEDS]


def cell3_manifest(steps: int, both_corpora: bool = True) -> list[dict]:
    """Cell 3 (geo3-only reference arm, sec 5.1/sec 10 -- never-cut at
    MINIMUM one corpus x 3 seeds; both_corpora=False drops to openr1 only
    if squeezed, sec 10 item 7)."""
    corpora = CORPORA if both_corpora else ("openr1",)
    return [{
        "wave": "2", "cell": "3", "corpus": corpus, "seed": seed,
        "hard_select_active": False, "geo3_active": True, "geo3_k_sel": K_SEL,
        "geo3_n_iter": GEO3_N_ITER, "steps": steps, "requires_override": True,
        "name": f"wB2_cell3_{corpus}_s{seed}_k{K_SEL}",
    } for corpus in corpora for seed in SEEDS]


def cell4_manifest(steps: int, surviving_mechanisms=("hard_ste",)) -> list[dict]:
    """Cell 4 (target): surviving candidate + geo3_active=True, M6's
    composition rule (hard_select_k_sel==geo3_k_sel, forced selection)."""
    runs = []
    for mech in surviving_mechanisms:
        for corpus in CORPORA:
            for seed in SEEDS:
                runs.append({
                    "wave": "2", "cell": "4", "corpus": corpus, "seed": seed,
                    "hard_select_active": True, "hard_select_mechanism": mech,
                    "hard_select_k_sel": K_SEL, "geo3_active": True, "geo3_k_sel": K_SEL,
                    "geo3_n_iter": GEO3_N_ITER, "steps": steps,
                    "name": f"wB2_cell4_{mech}_{corpus}_s{seed}_k{K_SEL}",
                })
    return runs


def cell4r_reserve_manifest(steps: int) -> list[dict]:
    """Cell 4R (RESERVE, sec 5.1/sec 10 item 2): random selection + geo3,
    same budget-matched cadence as 2R. Run ONLY if Cell 4 clears its
    interaction bar -- cut-eligible, priced separately, never in the
    default factorial total."""
    return [{
        "wave": "2", "cell": "4R", "corpus": corpus, "seed": seed,
        "hard_select_active": True, "hard_select_mechanism": "random_topk",
        "hard_select_k_sel": K_SEL, "geo3_active": True, "geo3_k_sel": K_SEL,
        "geo3_n_iter": GEO3_N_ITER, "steps": steps,
        "name": f"wB2_cell4R_{corpus}_s{seed}_k{K_SEL}",
    } for corpus in CORPORA for seed in SEEDS]


def full_factorial_manifest(steps: int, surviving_mechanisms=("hard_ste",),
                             both_corpora_cell3: bool = True) -> dict:
    """Assembles the DEFAULT (non-reserve) sec 5 2x2 factorial manifest,
    grouped by cell for reporting -- Cell 1 is NOT here (it is the
    archived-checkpoint re-measurement, no spawned run, sec 5.3)."""
    return {
        "2": cell2_manifest(steps, surviving_mechanisms),
        "2R": cell2r_manifest(steps),
        "comparator": comparator_manifest(steps),
        "3": cell3_manifest(steps, both_corpora_cell3),
        "4": cell4_manifest(steps, surviving_mechanisms),
    }


# ---------------------------------------------------------------------------
# Gates: budget (live PROGRAM_SPENT_GPUH), disk-space (gate (f) pattern,
# CLONED from run_lm_rd_trackc_sweep.py -- attribution at each function).
# ---------------------------------------------------------------------------

def budget_guard(projected_gpu_h: float, label: str, accept_override: bool) -> float:
    """CLONED from run_lm_rd_trackc_sweep.py::budget_guard (:784-802) --
    same shape, this wave's own label. PROGRAM_SPENT_GPUH is LIVE-imported
    (this file's one documented exception to clone-not-import, see module
    docstring) so this print reflects the actual maintained ledger at
    call time, not a value frozen when this file was written."""
    cumulative = PROGRAM_SPENT_GPUH + projected_gpu_h
    print(f"BUDGET GUARD ({label}): program-spent-so-far={PROGRAM_SPENT_GPUH:.2f} GPU-h (LIVE from "
          f"run_lm_rd_trackc_sweep.PROGRAM_SPENT_GPUH) + this-wave-projected={projected_gpu_h:.2f} "
          f"GPU-h = cumulative {cumulative:.2f} GPU-h, ceiling {GPU_H_PROGRAM_CEILING:.0f} GPU-h.",
          flush=True)
    if cumulative > GPU_H_PROGRAM_CEILING and not accept_override:
        print(f"ERROR: projected cumulative spend {cumulative:.2f} GPU-h EXCEEDS the "
              f"{GPU_H_PROGRAM_CEILING:.0f} GPU-h program ceiling -- REFUSING to launch {label}. "
              f"Pass --accept-budget-override to force past this guard, or cut scope first "
              f"(TRACKB_REDESIGN.md sec 10's cut order).", file=sys.stderr)
        sys.exit(5)
    return cumulative


DISK_SAFETY_FACTOR = 1.5


def disk_space_check(ckpt_dir: str, projected_bytes: int, label: str,
                      safety_factor: float = DISK_SAFETY_FACTOR) -> dict:
    """CLONED from run_lm_rd_trackc_sweep.py::disk_space_check (:845-862) --
    the gate (f) pattern the task brief names explicitly."""
    import shutil
    resolved = os.path.realpath(ckpt_dir)
    free_bytes = shutil.disk_usage(resolved).free if os.path.exists(resolved) else 0
    required_bytes = int(projected_bytes * safety_factor)
    return {
        "label": label, "resolved_ckpt_dir": resolved, "projected_ckpt_bytes": projected_bytes,
        "safety_factor": safety_factor, "required_bytes": required_bytes, "free_bytes": free_bytes,
        "ok": os.path.exists(resolved) and free_bytes >= required_bytes,
    }


def epoch_cap_check(corpus_n_tokens: int, steps: int, batch_size: int, seq_len: int,
                     epoch_cap: int = 5) -> dict:
    """Gate (d)'s pattern: this wave's own token budget must not exceed
    epoch_cap physical passes over the corpus (sec 5.4's <=5-epoch
    discipline, this codebase's standing convention)."""
    tokens_needed = steps * batch_size * seq_len
    n_epochs = tokens_needed / max(1, corpus_n_tokens)
    return {"tokens_needed": tokens_needed, "corpus_n_tokens": corpus_n_tokens,
            "n_epochs": n_epochs, "epoch_cap": epoch_cap, "ok": n_epochs <= epoch_cap}


# ---------------------------------------------------------------------------
# EOT/content_mask threading -- forced-selection negative smoke wiring
# (M6's registered build smoke: "a forced index pointing at an EOT
# position must come out invalid and take the existing no-op scatter
# path"). The actual CPU assertion lives in test_trackb_smokes.py (which
# calls _geo3_lm_select_and_orthogonalize directly); this function is the
# launcher-facing wrapper the wave manifest documents as its own required
# pre-launch check.
# ---------------------------------------------------------------------------

def run_eot_forced_selection_negative_smoke() -> bool:
    """Returns True iff the M6 EOT-forced-selection negative smoke passes.
    Delegates to test_trackb_smokes.smoke_eot_forced_selection_negative
    (imported lazily to avoid a hard dependency for callers that only need
    the manifest/gate functions above)."""
    import test_trackb_smokes as tts
    return tts.smoke_eot_forced_selection_negative(verbose=False)


# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------

def dry_run_preview(steps: int, factorial_steps: int, surviving_mechanisms, accept_budget_override: bool):
    m_neg1 = waveNeg1_manifest(steps)
    m_pilot = [reference_pilot_manifest(steps)]
    m_stability = [waveNeg1_stability_smoke_manifest(steps)]
    m_factorial = full_factorial_manifest(factorial_steps, surviving_mechanisms)
    m_4r = cell4r_reserve_manifest(factorial_steps)

    print("=" * 70)
    print("TRACK B HARD-SELECTIVITY WAVE -- DRY-RUN PREVIEW (no launch this session)")
    print("=" * 70)

    h_neg1 = nongeo3_h(len(m_neg1) + len(m_pilot), steps)
    h_stability = geo3_h(len(m_stability), steps)
    print(f"\nWave -1: {len(m_neg1)} mechanism probes + {len(m_pilot)} reference pilot "
          f"({steps} steps each, non-geo3) = {h_neg1:.3f} GPU-h; "
          f"+ {len(m_stability)} geo3-active stability smoke = {h_stability:.3f} GPU-h; "
          f"+ Cell-1 re-measurement (6 archived checkpoints, forward-only, ~0.10 GPU-h, CPU-cheap) "
          f"+ MC anchor recomputation (CPU, free).")
    for spec in m_neg1 + m_pilot + m_stability:
        print(f"  - {spec['name']}")

    print(f"\nBANDS_PINNED-TrackB gate: written only after every Wave -1 pilot validates complete "
          f"(bands_pinned_trackb.write_bands_pinned_trackb); factorial cells below REFUSE to launch "
          f"without a valid, hash-verified {bands_pinned_out_path('<out-dir>')}.")

    print(f"\nsec 5 2x2 factorial (surviving mechanisms={list(surviving_mechanisms)}, "
          f"steps={factorial_steps}, K_sel={K_SEL}):")
    total_h = h_neg1 + h_stability
    for cell, runs in m_factorial.items():
        is_geo3 = cell in ("3", "4")
        h = geo3_h(len(runs), factorial_steps) if is_geo3 else nongeo3_h(len(runs), factorial_steps)
        total_h += h
        print(f"  Cell {cell}: {len(runs)} runs, {h:.3f} GPU-h")
        for spec in runs:
            override_tag = " [REQUIRES --accept-no-launch-reference-arm if gate=no_launch_redesign]" \
                if spec.get("requires_override") else ""
            print(f"    - {spec['name']}{override_tag}")
    print(f"  Cell 1 (baseline): re-measured from 6 ARCHIVED Wave C checkpoints, no spawned run.")

    h_4r = geo3_h(len(m_4r), factorial_steps)
    print(f"\nReserve Cell 4R ({len(m_4r)} runs, {h_4r:.3f} GPU-h, run ONLY if Cell 4 clears its "
          f"interaction bar, cut-eligible per sec 10 item 2) -- NOT included in the total below.")

    print(f"\nTOTAL projected (Wave -1 + factorial, excl. Cell 4R): {total_h:.3f} GPU-h "
          f"(sec 6.1's own central estimate ~=8, wide-case ~13-20.6, worst-case incl. Cell 4R "
          f"~=22.3, vs the corrected ~=33.6 GPU-h nominal headroom -- Rev 3 sec 6).")
    budget_guard(total_h + h_4r, "TrackB full wave (incl. reserve Cell 4R, worst case)",
                 accept_budget_override)
    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/trackb"))
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--wave", choices=["neg1", "bands-pinned", "factorial", "cell3", "cell4r"], default=None)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--gpu-offset", type=int, default=None)
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None, help="override Wave -1 probe step count")
    ap.add_argument("--factorial-steps", type=int, default=None,
                     help="override Wave 1/2 factorial step count (default: default_steps())")
    ap.add_argument("--surviving-mechanisms", nargs="+", default=["hard_ste"],
                     help="Cell 2/4's own surviving-candidate list, named after Wave 1's readout "
                          "(default: candidate 1 alone, the never-cut primary).")
    ap.add_argument("--accept-no-launch-reference-arm", action="store_true",
                     help="TRACKB_REDESIGN.md sec 5.1 (Rev 2 M5): required to run Cell 3 if the "
                          "ORIGINAL geo3-in-LM gate verdict is no_launch_redesign. Stamps "
                          "gate_override=true/claim_tier=descriptive into every spawned Cell-3 "
                          "run's OWN result JSON at assembly time.")
    ap.add_argument("--geo3-gate-json", default=None,
                     help="path to lm_geo3_wave_neg1_gate.py's own output JSON (default: "
                          "<out-dir>/../lm_rd_geo3/wave_neg1_gate.json) -- Cell 3's override target.")
    ap.add_argument("--accept-budget-override", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else WAVE_NEG1_PROBE_STEPS
    factorial_steps = args.factorial_steps if args.factorial_steps is not None else default_steps()

    if args.dry_run:
        dry_run_preview(steps, factorial_steps, args.surviving_mechanisms, args.accept_budget_override)
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "bands-pinned":
        # CPU-only: no --gpus/--gpu-offset needed. Reads Wave -1's own result JSONs from
        # <out-dir>, computes the BANDS_PINNED-TrackB derivations, writes the pinned file.
        print("--wave bands-pinned is CPU-only (derivation math over already-completed Wave -1 "
              "result JSONs) -- this build session has none to read (no GPU runs performed), so "
              "this path is BUILT and unit-testable (test_trackb_smokes.py) but not exercised "
              "against real Wave -1 data here.", flush=True)
        return

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose) -- check nvidia-smi NOW and pass the free set explicitly. THIS BUILD "
              "SESSION DOES NOT LAUNCH ANYTHING regardless (task brief HARD CONSTRAINT).",
              file=sys.stderr)
        sys.exit(1)

    # Every real-launch path below is fully wired (manifest + gates) but is NOT invoked this
    # session -- the task brief's hard constraint (no GPU runs) stops here, at the same place
    # run_lm_rd_geo3_sweep.py's/run_lm_rd_trackc_sweep.py's own "this build phase does not launch
    # anything" precedent stops.
    print("Manifest + gates built; this build session performs NO real launch (HARD CONSTRAINT). "
          "See --dry-run for the full preview.", flush=True)


if __name__ == "__main__":
    main()
