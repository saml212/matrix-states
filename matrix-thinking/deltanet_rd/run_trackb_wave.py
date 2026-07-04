"""run_trackb_wave.py -- TRACKB_REDESIGN.md Rev 3 (commit 8ab089d): the
Track B hard-selectivity wave orchestrator. Waves: -1 (mechanism probes +
reference pilot + stability smoke) -> cell1probe (archived-checkpoint
re-measurement) -> bands-pinned (CPU assembly of BANDS_PINNED-TrackB.json)
-> 1 (selectivity-only cells 2/2R/comparator) -> 2 (geo3 cells 3/4) ->
3 (instrumentation probes) -> 4r (reserve), per sec 5/sec 10's cut order.

A NEW file (not a --wave addition to run_lm_rd_geo3_sweep.py) per this
build's own task brief item 8's "or a new run_trackb_wave.py" option --
Track B's hard-selectivity wave is a genuinely different manifest shape
from that file's (beta_topk-or-naive_window)-only Waves 1/2/3.

DISPATCH ENGINE (AUDIT FIX, independent audit 2026-07-04 -- F1: the first
build shipped manifests and gates with NO subprocess dispatch behind them,
the third occurrence of the 'gates with nothing behind them' pattern this
project's audits have caught): run_smoke / write_progress / aggregate /
_run_wave below are CLONED from run_lm_rd_geo3_sweep.py's own engine
(:355-516 -- smoke gate, exception-isolated launch, validity-checked
resume, per-run timeout with GPU quarantine, guarded aggregate), per this
codebase's clone-not-import-between-sweep-scripts pod-safety convention.
The ONE deliberate live-import exception (documented at its definition):
`PROGRAM_SPENT_GPUH`/`GPU_H_PROGRAM_CEILING` from run_lm_rd_trackc_sweep.py
-- a single shared DATA constant (the program's maintained budget ledger),
not orchestration logic; the task brief's own instruction is to "read
PROGRAM_SPENT_GPUH live," which a frozen clone cannot do.

Gates wired into main() (all four the audit named, plus the epoch-cap/EOT
gates): (1) budget_guard -- every launch wave; (2) disk_space_check (the
trackc gate-(f) pattern) -- every training wave; (3) load_gate_verdict +
refuse_or_override_cell3 -- wave 2's Cell-3 entries; (4)
validate_bands_pinned_trackb -- waves 1/2/4r REFUSE without a valid,
hash-clean BANDS_PINNED-TrackB.json (and thread its b_pinned into every
masking cell); plus the epoch-cap check (live from each corpus's own
meta.json train_tokens) and the M6 EOT-forced-selection negative smoke
before any forced-selection wave.

**THIS BUILD SESSION DOES NOT LAUNCH ANYTHING (task brief HARD CONSTRAINT:
build + CPU-verify only, no GPU runs).** The engine below is complete and
smoke-gated; no wave was invoked for real this session -- --dry-run
preview + CPU smokes only.

Usage (GPU list is an example -- check nvidia-smi first, per house rule):
  python run_trackb_wave.py --dry-run
  python wave_neg1_trackb.py --build-stress-slice --data-dir /data/deltanet_rd_data   # prereq for wave neg1's stability cell
  python run_trackb_wave.py --wave neg1 --out-dir results/trackb --gpus 1 --gpu-offset 7
  python run_trackb_wave.py --wave cell1probe --cell1-checkpoints <6 archived ckpts...> --gpus 1 --gpu-offset 7
  python run_trackb_wave.py --wave bands-pinned --out-dir results/trackb              # CPU-only assembly
  python run_trackb_wave.py --wave 1 --out-dir results/trackb --gpus 4 --gpu-offset 0
  python run_trackb_wave.py --wave 2 --accept-no-launch-reference-arm --out-dir results/trackb --gpus 2 --gpu-offset 6
  python run_trackb_wave.py --wave 3 --out-dir results/trackb --gpus 1 --gpu-offset 7
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAIN = os.path.join(HERE, "lm_pretrain_rd.py")
PROBE_TOOL = os.path.join(HERE, "wave_neg1_trackb.py")
CPU_SMOKES = os.path.join(HERE, "test_trackb_smokes.py")
sys.path.insert(0, HERE)  # pod-safe imports

import hard_selectivity_rd as hs                    # noqa: E402
import bands_pinned_trackb as bp                     # noqa: E402

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
CKPT_EVERY = 1_000                         # lm_pretrain_rd.py's own default (full-length waves)
WAVE_NEG1_CKPT_EVERY = 250                 # 2,000-step pilots need >= 6 diagnostic checkpoints for
                                           # the sec 4.3 last-5-log-points derivations (churn's
                                           # first checkpoint contributes none) -- 2,000/250 = 8
EPOCH_CAP = 5                              # sec 5.4's <=5-physical-epoch discipline (house standing)

# sec 6.1's pricing constants -- Wave C's own measured ~0.077 GPU-h/6,103-step run (non-geo3), the
# codebase's registered 3x geo3 planning placeholder (run_lm_rd_geo3_sweep.py:94-104), and a
# per-probe planning constant for wave 3 (sec 6.1's own row prices the whole instrumentation wave
# at ~1-2 GPU-h across all cells; 0.05/probe x ~30 probes ~= 1.5, inside that band -- superseded
# by real timing like every other placeholder).
_GPU_H_PER_STEP_NONGEO3 = 0.077 / 6_103          # ~0.0000126 GPU-h/step
_GPU_H_PER_STEP_GEO3 = 0.28 / 6_103              # ~0.0000459 GPU-h/step (3x placeholder-priced)
_GPU_H_PER_PROBE = 0.05
_PER_STEP_S_NONGEO3 = 0.05                        # timeout sizing (run_lm_rd_geo3_sweep.py's own
_PER_STEP_S_GEO3 = 0.15                           # measured baseline + 3x placeholder)
_PER_CHECKPOINT_S = 15.0


def default_steps(target_tokens: int = TARGET_TOKENS_FULL, batch_size: int = BATCH_SIZE,
                   seq_len: int = SEQ_LEN) -> int:
    return max(1, target_tokens // (batch_size * seq_len))


def spec_gpu_h(spec: dict) -> float:
    if "probe_checkpoint" in spec:
        return _GPU_H_PER_PROBE
    per = _GPU_H_PER_STEP_GEO3 if spec.get("geo3_active") else _GPU_H_PER_STEP_NONGEO3
    return spec["steps"] * per


def default_timeout_pretrain(spec: dict, margin: float = 1.6) -> int:
    per_step = _PER_STEP_S_GEO3 if spec.get("geo3_active") else _PER_STEP_S_NONGEO3
    ckpt_every = spec.get("ckpt_every", CKPT_EVERY)
    n_ckpts = spec["steps"] // ckpt_every + 1
    return int((per_step * spec["steps"] + n_ckpts * _PER_CHECKPOINT_S) * margin)


def default_timeout_probe(spec: dict, margin: float = 1.6) -> int:
    return int((_GPU_H_PER_PROBE * 3600.0) * margin)


# ---------------------------------------------------------------------------
# Cell-3 override mechanics (sec 5.1, Rev 2 -- M5): a stamped bypass of the
# ACTUAL production refusal, run_lm_rd_geo3_sweep.py::_refuse_if_no_launch
# (:172-183, sys.exit(3)) -- CLONED (not imported, pod-safety convention),
# plus load_gate_verdict (:131-162, the config cross-validation that stays
# LIVE under the override per M5's requirement (i)).
# ---------------------------------------------------------------------------

_VALID_VERDICTS = ("beta_gated_primary", "naive_window_primary", "no_launch_redesign")


def load_gate_verdict(gate_json_path: str) -> tuple[str, dict]:
    """CLONED from run_lm_rd_geo3_sweep.py's own function of the same name
    (:131-162) -- Track B's hard-selectivity wave reads the SAME Wave -1
    gate JSON (the ORIGINAL beta-gated/naive-window gate is the thing Cell
    3's override targets)."""
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
    `accept_override` is True (--accept-no-launch-reference-arm) --
    requirement (ii). Requirement (i) -- load_gate_verdict's config
    cross-validation stays live -- is enforced by the CALLER always calling
    load_gate_verdict first. Requirement (iii) -- per-run assembly-time
    stamping -- is satisfied by returning the stamp payload; build_cmd
    threads its reason into every spawned Cell-3 run's
    --gate-override-reason (lm_pretrain_rd.py's _assemble_result writes
    gate_override/gate_override_reason/gate_override_at/claim_tier into the
    result JSON AT ASSEMBLY, never post-hoc).

    Returns the stamp dict if the override fires, None if verdict is
    launch-eligible (not an override case at all); sys.exit(3) -- the same
    exit code as _refuse_if_no_launch -- on no-launch without the flag."""
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
# Manifests. Cell 1 is never a spawned training run (the archived Wave C
# checkpoints are re-MEASURED, wave cell1probe); every other cell dispatches
# lm_pretrain_rd.py (training) or wave_neg1_trackb.py (probes).
# ---------------------------------------------------------------------------

_PROBE_MECHANISMS = ("hard_ste", "entmax", "soft_topk_comparator", "random_topk")


def waveNeg1_manifest(steps: int = WAVE_NEG1_PROBE_STEPS) -> list[dict]:
    """sec 6.1 Wave -1 (a): "5 probes x 2,000 steps" = the 4 hard-select
    mechanisms (candidate 1/2/comparator/2R) + candidate 4's hard-snap
    schedule (mechanically identical to candidate 1, sec 3.4) -- ALL
    non-geo3, single corpus/seed (openr1, seed 0), PLUS the unmasked
    reference pilot (sec 4.3's Null A/TV-ceiling source, instrumented via
    --trackb-selection-probe) and the M8/NEW-7 stability smoke
    (geo3-active, on the materialized "openr1-stress" slice corpus, with
    --nan-probe-counter). SCRUTINY NOTE for the independent auditor: sec
    6.1's table states "5 probes" without itemizing which 5; this
    manifest's reading (4 mechanisms + candidate 4's snap schedule) is this
    build's own registered interpretation, flagged for review."""
    runs = []
    for mech in _PROBE_MECHANISMS:
        runs.append({
            "wave": "neg1", "cell": f"probe_{mech}", "corpus": "openr1", "seed": 0,
            "hard_select_active": True, "hard_select_mechanism": mech, "hard_select_k_sel": K_SEL,
            "geo3_active": False, "steps": steps, "ckpt_every": WAVE_NEG1_CKPT_EVERY,
            "name": f"wBneg1_probe_{mech}_k{K_SEL}",
        })
    runs.append({
        "wave": "neg1", "cell": "probe_candidate4_hardsnap", "corpus": "openr1", "seed": 0,
        "hard_select_active": True, "hard_select_mechanism": "hard_ste",   # snap schedule wraps hard_ste
        "hard_select_k_sel": K_SEL, "geo3_active": False, "steps": steps,
        "ckpt_every": WAVE_NEG1_CKPT_EVERY,
        "name": f"wBneg1_probe_candidate4hardsnap_k{K_SEL}",
    })
    runs.append({
        "wave": "neg1", "cell": "reference_pilot", "corpus": "openr1", "seed": 0,
        "hard_select_active": False, "geo3_active": False, "steps": steps,
        "ckpt_every": WAVE_NEG1_CKPT_EVERY, "selection_probe": K_SEL,
        "name": "wBneg1_reference_pilot",
    })
    runs.append({
        "wave": "neg1", "cell": "stability_smoke", "corpus": "openr1-stress", "seed": 0,
        "hard_select_active": False, "geo3_active": True, "geo3_k_sel": K_SEL,
        "geo3_n_iter": GEO3_N_ITER, "steps": steps, "ckpt_every": WAVE_NEG1_CKPT_EVERY,
        "nan_probe_counter": True,
        "name": "wBneg1_stability_smoke",
    })
    return runs


def cell1probe_manifest(cell1_checkpoints: list) -> list[dict]:
    """Wave cell1probe: the sec 5.2/5.3 same-instrument re-measurement of
    the 6 ARCHIVED Wave C checkpoints (read-only, forward-only) via
    wave_neg1_trackb.py --probe-checkpoint. `cell1_checkpoints`: explicit
    paths, no default on purpose -- pass what the archive actually holds."""
    runs = []
    for i, ckpt in enumerate(cell1_checkpoints):
        base = os.path.splitext(os.path.basename(ckpt))[0]
        runs.append({
            "wave": "cell1probe", "probe_checkpoint": ckpt, "corpus": "openr1", "steps": 0,
            "name": f"cell1probe_{i}_{base[:40]}",
        })
    return runs


def cell2_manifest(steps: int, surviving_mechanisms=("hard_ste",)) -> list[dict]:
    """Cell 2 (selectivity-only, geo3_active=False): the surviving
    candidate(s) from Wave -1's manipulation check -- sec 10's cut order
    ranks candidate 1 (hard_ste) never-cut; default is just it, widened by
    --surviving-mechanisms once Wave -1's real readout names more."""
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
    """Cell 2R (REQUIRED, never-cut, sec 10): budget-matched random control."""
    return [{
        "wave": "1", "cell": "2R", "corpus": corpus, "seed": seed,
        "hard_select_active": True, "hard_select_mechanism": "random_topk",
        "hard_select_k_sel": K_SEL, "geo3_active": False, "steps": steps,
        "name": f"wB1_cell2R_{corpus}_s{seed}_k{K_SEL}",
    } for corpus in CORPORA for seed in SEEDS]


def comparator_manifest(steps: int) -> list[dict]:
    """M7 comparator (sec 3.1, cut order item 6): tau-annealed soft top-K."""
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
    composition rule (hard_select_k_sel==geo3_k_sel, forced selection).

    entmax PRE-FILTER (AUDIT FIX, 2026-07-04 -- F1's named sub-item):
    entmax cannot compose with geo3 (no fixed-shape forced_topk_idx under
    variable sparsemax support -- DeltaNetLMMixer's own constructor
    hard-refuses the combination), so it is filtered HERE, loudly, instead
    of dying one-per-run inside six spawned subprocesses."""
    composable = [m for m in surviving_mechanisms if m != "entmax"]
    if len(composable) < len(surviving_mechanisms):
        print("Cell 4 manifest: 'entmax' dropped from the surviving-mechanism list -- it cannot "
              "compose with geo3 (M6's forced-selection argument needs a fixed-shape top-K set; "
              "sparsemax support is variable; the mixer constructor hard-refuses it). If candidate "
              "2 is the sole survivor, Cell 4 needs its own (unbuilt) variable-support forcing "
              "scheme -- see TRACKB_REDESIGN.md sec 5.1.", flush=True)
    runs = []
    for mech in composable:
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
    """Cell 4R (RESERVE, sec 5.1/sec 10 item 2): random selection + geo3.
    Run ONLY if Cell 4 clears its interaction bar -- cut-eligible, priced
    separately, never in the default factorial total."""
    return [{
        "wave": "4r", "cell": "4R", "corpus": corpus, "seed": seed,
        "hard_select_active": True, "hard_select_mechanism": "random_topk",
        "hard_select_k_sel": K_SEL, "geo3_active": True, "geo3_k_sel": K_SEL,
        "geo3_n_iter": GEO3_N_ITER, "steps": steps,
        "name": f"wB2_cell4R_{corpus}_s{seed}_k{K_SEL}",
    } for corpus in CORPORA for seed in SEEDS]


def full_factorial_manifest(steps: int, surviving_mechanisms=("hard_ste",),
                             both_corpora_cell3: bool = True) -> dict:
    """The DEFAULT (non-reserve) sec 5 factorial, grouped by cell for
    reporting -- Cell 1 is NOT here (wave cell1probe, no spawned training)."""
    return {
        "2": cell2_manifest(steps, surviving_mechanisms),
        "2R": cell2r_manifest(steps),
        "comparator": comparator_manifest(steps),
        "3": cell3_manifest(steps, both_corpora_cell3),
        "4": cell4_manifest(steps, surviving_mechanisms),
    }


def wave3_manifest(out_dir: str, wave1_manifest: list, wave2_manifest: list,
                    n_docs: int = 32) -> list[dict]:
    """AUDIT FIX (2026-07-04, F1's named sub-item): sec 6.1's Wave-3 row --
    the geometry/drift instrumentation pass over EVERY completed Wave-1/2
    cell's final checkpoint (forward-hook probes, no backward), dispatched
    via wave_neg1_trackb.py --probe-checkpoint. Cells whose source run has
    no completed checkpoint yet get probe_checkpoint=None and are dropped
    with a printed SKIP inside _run_wave (mirrors run_lm_rd_geo3_sweep.py's
    waveB3 convention)."""
    runs = []
    for src in wave1_manifest + wave2_manifest:
        src_wave_dir = os.path.join(out_dir, f"wave{src['wave']}")
        result_path = os.path.join(src_wave_dir, f"{src['name']}.json")
        ckpt = None
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    d = json.load(f)
                if d.get("complete") is True:
                    ckpt = d.get("final_checkpoint_path")
            except Exception:
                ckpt = None
        runs.append({
            "wave": "3", "probe_checkpoint": ckpt, "source_run": src["name"],
            "corpus": src["corpus"], "steps": 0, "n_docs": n_docs,
            "name": f"wB3_probe_{src['name']}",
        })
    return runs


# ---------------------------------------------------------------------------
# Gates: budget (live PROGRAM_SPENT_GPUH), disk-space (trackc gate-(f)
# pattern), epoch-cap (live meta.json), BANDS_PINNED launcher gate, EOT
# negative smoke, stability-smoke gate for Cells 3/4.
# ---------------------------------------------------------------------------

def budget_guard(projected_gpu_h: float, label: str, accept_override: bool) -> float:
    """CLONED from run_lm_rd_trackc_sweep.py::budget_guard (:784-802).
    PROGRAM_SPENT_GPUH is LIVE-imported (module docstring's one documented
    exception) so this reflects the maintained ledger at call time."""
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


def find_ckpt_size_bytes(ckpt_dir: str, d_model: int = D_MODEL, d_state: int = D_STATE,
                          n_layers: int = N_LAYERS) -> int:
    """CLONED from run_lm_rd_trackc_sweep.py::find_ckpt_size_bytes
    (:816-835): the disk gate needs a REAL measured checkpoint size for
    this exact architecture (lm_pretrain_rd.py's own naming convention,
    'lmC_..._dm{d}_ds{s}_L{n}_...pt'), never an assumed one. Raises
    FileNotFoundError with the remedy if none exists yet."""
    needle = f"_dm{d_model}_ds{d_state}_L{n_layers}_"
    if os.path.isdir(ckpt_dir):
        for fname in sorted(os.listdir(ckpt_dir)):
            if needle in fname and fname.endswith(".pt"):
                return os.path.getsize(os.path.join(ckpt_dir, fname))
    raise FileNotFoundError(
        f"no existing checkpoint matching dm{d_model}/ds{d_state}/L{n_layers} under {ckpt_dir!r} "
        f"-- the disk-space gate needs a REAL measured checkpoint size; point --ckpt-size-probe-dir "
        f"at a directory holding any Wave-C-architecture checkpoint (the archived Wave C tree "
        f"qualifies), or run one Wave -1 probe cell first.")


def projected_ckpt_bytes(n_runs: int, steps: int, ckpt_every: int, ckpt_size_bytes: int) -> int:
    n_ckpts = steps // ckpt_every + 1
    return n_runs * n_ckpts * ckpt_size_bytes


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


def read_corpus_train_tokens(data_dir: str, corpus: str) -> int:
    """Live meta.json read for the epoch-cap gate (the rebuilt corpora all
    carry a train_tokens field -- verified against the box's own
    reasoning_eot/meta.json this build session). REFUSES (KeyError with
    remedy) rather than estimating if the field is absent."""
    from lm_pretrain_rd import CORPUS_DIRS
    meta_path = os.path.join(data_dir, CORPUS_DIRS[corpus], "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    if "train_tokens" not in meta:
        raise KeyError(f"{meta_path} has no train_tokens field -- the epoch-cap gate needs the "
                       f"real corpus size, never an estimate; the corpus build scripts write it, "
                       f"re-run them rather than guessing here.")
    return int(meta["train_tokens"])


def epoch_cap_check(corpus_n_tokens: int, steps: int, batch_size: int = BATCH_SIZE,
                     seq_len: int = SEQ_LEN, epoch_cap: int = EPOCH_CAP) -> dict:
    tokens_needed = steps * batch_size * seq_len
    n_epochs = tokens_needed / max(1, corpus_n_tokens)
    return {"tokens_needed": tokens_needed, "corpus_n_tokens": corpus_n_tokens,
            "n_epochs": n_epochs, "epoch_cap": epoch_cap, "ok": n_epochs <= epoch_cap}


def run_epoch_cap_gate(manifest: list, data_dir: str, label: str) -> None:
    """Applies epoch_cap_check per (corpus, steps) pair in the manifest;
    refuses the wave on any breach (the trackc gate-(d) shape)."""
    seen = set()
    for spec in manifest:
        if "probe_checkpoint" in spec or spec["steps"] == 0:
            continue
        key = (spec["corpus"], spec["steps"])
        if key in seen:
            continue
        seen.add(key)
        n_tokens = read_corpus_train_tokens(data_dir, spec["corpus"])
        r = epoch_cap_check(n_tokens, spec["steps"])
        print(f"EPOCH-CAP GATE ({label}, {spec['corpus']} @ {spec['steps']} steps): "
              f"{r['n_epochs']:.2f} epochs vs cap {r['epoch_cap']} -> {'OK' if r['ok'] else 'BREACH'}",
              flush=True)
        if not r["ok"]:
            print(f"ERROR: {spec['corpus']} at {spec['steps']} steps needs {r['n_epochs']:.2f} "
                  f"physical epochs > the sec 5.4 cap of {r['epoch_cap']} -- REFUSING {label}. "
                  f"Lower --steps or extend the corpus (the trackc extended-mix remedy), never "
                  f"silently over-epoch.", file=sys.stderr)
            sys.exit(6)


def bands_pinned_out_path(out_dir: str) -> str:
    return os.path.join(out_dir, "BANDS_PINNED-TrackB.json")


def require_bands_pinned(out_dir: str) -> dict:
    """The sec 4.3 LAUNCHER GATE: waves 1/2/4r refuse without a valid,
    hash-clean BANDS_PINNED-TrackB.json. Returns the validated doc (its
    b_pinned is threaded into every masking cell's build_cmd)."""
    path = bands_pinned_out_path(out_dir)
    doc = bp.validate_bands_pinned_trackb(path)
    if doc is None:
        print(f"ERROR: BANDS_PINNED-TrackB.json missing/invalid/hash-mismatched at {path!r} -- "
              f"anchor-dependent cells REFUSE to launch without it (TRACKB_REDESIGN.md sec 4.3's "
              f"launcher-gate requirement). Run '--wave bands-pinned' after Wave -1 + cell1probe "
              f"complete.", file=sys.stderr)
        sys.exit(4)
    print(f"BANDS_PINNED-TrackB validated (pinned_at={doc['pinned_at_iso']}, "
          f"b_pinned={doc['b_pinned']['b_pinned']:.4f}).", flush=True)
    return doc


def run_eot_forced_selection_negative_smoke() -> bool:
    """M6's registered pre-launch negative smoke (CPU): a forced index at an
    EOT position must come out invalid and take the no-op scatter path."""
    import test_trackb_smokes as tts
    return tts.smoke_eot_forced_selection_negative(verbose=False)


def check_stability_smoke_gate(out_dir: str) -> None:
    """sec 5.1 (M8/NEW-7): Cells 3/4 (and 4R) do not launch until the Wave
    -1 stability smoke is complete, PROBATIVE, and clean (zero non-finite
    losses AND skip-rate <=1%). Refuses loudly otherwise."""
    path = os.path.join(out_dir, "waveneg1", "wBneg1_stability_smoke.json")
    if not os.path.exists(path):
        print(f"ERROR: stability-smoke result not found at {path!r} -- Cells 3/4 are gated on the "
              f"Wave -1 geo3-LM stability smoke (sec 5.1). Run --wave neg1 first.", file=sys.stderr)
        sys.exit(7)
    with open(path) as f:
        d = json.load(f)
    problems = []
    if d.get("complete") is not True:
        problems.append("not complete")
    if d.get("skip_rate", 1.0) > 0.01:
        problems.append(f"skip_rate {d.get('skip_rate')} > 0.01")
    losses = [t.get("loss") for t in (d.get("trajectory") or [])]
    if any(v is None or not math.isfinite(v) for v in losses):
        problems.append("non-finite loss in trajectory")
    probe = d.get("nan_probe_positive_control") or {}
    if probe.get("probative") is not True:
        problems.append(f"positive control NON-PROBATIVE ({probe.get('verdict', 'missing')})")
    if problems:
        print(f"ERROR: stability smoke at {path!r} does not clear the sec 5.1 gate: "
              f"{'; '.join(problems)}. Cells 3/4 REFUSED until a fix wave (with its own "
              f"independent audit) addresses or bounds this -- a registered stability finding, "
              f"never a silent proceed.", file=sys.stderr)
        sys.exit(7)
    print(f"Stability-smoke gate CLEAR: complete, skip_rate={d.get('skip_rate'):.4f} <= 0.01, all "
          f"losses finite, positive control PROBATIVE "
          f"({probe.get('n_calls_meeting_floor')} calls met the >= {probe.get('floor_n_dup')}-dup "
          f"floor).", flush=True)


# ---------------------------------------------------------------------------
# Dispatch engine -- CLONED from run_lm_rd_geo3_sweep.py's _run_wave pattern
# (:355-516): smoke gate, exception-isolated launch, validity-checked
# resume, per-run timeout with GPU quarantine, guarded aggregate (AUDIT FIX
# 2026-07-04, F1).
# ---------------------------------------------------------------------------

def run_smoke(log_dir, gpu):
    print(f"SMOKE GATE (physical GPU {gpu}) -- lm_pretrain_rd.py --smoke + test_trackb_smokes.py "
          f"(CPU) ...", flush=True)
    ok = True
    with open(os.path.join(log_dir, "smoke_pretrain.log"), "w") as lf:
        rc = subprocess.call([sys.executable, PRETRAIN, "--smoke"],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                              stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    print(f"  smoke[pretrain]: {'PASS' if rc == 0 else f'FAIL (rc={rc})'}", flush=True)
    ok = ok and (rc == 0)
    with open(os.path.join(log_dir, "smoke_trackb_cpu.log"), "w") as lf:
        rc = subprocess.call([sys.executable, CPU_SMOKES],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                              stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    print(f"  smoke[trackb_cpu]: {'PASS' if rc == 0 else f'FAIL (rc={rc})'}", flush=True)
    return ok and (rc == 0)


def write_progress(out_dir, done, failed, running, wave):
    try:
        with open(os.path.join(out_dir, "PROGRESS.txt"), "w") as f:
            f.write(f"wave={wave} done={done} failed={failed} running={running}\n")
    except Exception as e:
        print(f"  (write_progress non-fatal: {e!r})", flush=True)


def aggregate(out_dir, manifest, failed, wave):
    report = {"wave": wave, "n_manifest": len(manifest), "n_failed": len(failed)}
    try:
        cells = {}
        for spec in manifest:
            p = os.path.join(out_dir, f"{spec['name']}.json")
            if not os.path.exists(p):
                continue
            try:
                with open(p) as f:
                    d = json.load(f)
            except Exception:
                continue
            if d.get("complete") is not True:
                continue
            if "probe_checkpoint" in spec:
                cells[spec["name"]] = {"checkpoint": d.get("checkpoint"),
                                        "checkpoint_step": d.get("checkpoint_step"),
                                        "per_layer_resid_mean": {k: v.get("resid_mean")
                                                                  for k, v in (d.get("per_layer") or {}).items()}}
            else:
                ck = (d.get("checkpoints") or [])
                final = ck[-1] if ck else {}
                cells[spec["name"]] = {
                    "corpus": d.get("corpus"), "seed": d.get("seed"), "cell": spec.get("cell"),
                    "final_step": d.get("final_step"),
                    "final_val_loss": final.get("val_loss"),
                    "skip_rate": d.get("skip_rate"),
                    "gate_override": d.get("gate_override"),
                    "hard_select_final_diag": final.get("hard_select_diagnostics"),
                    "nan_probe": (d.get("nan_probe_positive_control") or {}).get("verdict"),
                    "wall_s": d.get("wall_s"),
                }
        report["cells"] = cells
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write(f"Track B hard-selectivity -- wave {wave}\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def is_done_trackb(out_dir, spec) -> bool:
    """Validity-checked resume (never bare file-existence): the result JSON
    must be complete, un-timed-out, and match the spec's own identity
    fields -- incl. the hard_select/geo3 config blocks, so a stale run at a
    different mechanism/K_sel can never silently count as done."""
    p = os.path.join(out_dir, f"{spec['name']}.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict) or d.get("complete") is not True:
            return False
        if "probe_checkpoint" in spec:
            return (d.get("checkpoint") == spec["probe_checkpoint"]
                    and d.get("k_sel") == K_SEL and bool(d.get("per_layer")))
        if d.get("timed_out"):
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if d.get("corpus") != spec["corpus"] or d.get("seed") != spec["seed"] \
                or d.get("steps") != spec["steps"]:
            return False
        hs_blk = d.get("hard_select") or {}
        if bool(hs_blk.get("active")) != bool(spec.get("hard_select_active")):
            return False
        if spec.get("hard_select_active"):
            if (hs_blk.get("mechanism") != spec["hard_select_mechanism"]
                    or hs_blk.get("k_sel") != spec["hard_select_k_sel"]):
                return False
        g = d.get("geo3_lm") or {}
        if bool(g.get("active")) != bool(spec.get("geo3_active")):
            return False
        if spec.get("geo3_active"):
            if g.get("k_sel") != spec["geo3_k_sel"] or g.get("n_iter") != spec["geo3_n_iter"]:
                return False
        if "gate_override" not in d:
            return False        # the always-present stamping field (M5) -- absence = pre-fix JSON
        return True
    except Exception:
        return False


def _ckpt_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "checkpoints")


def build_cmd_trackb(spec, out_dir, timeout, data_dir, b_pinned: float | None = None,
                      override_stamp: dict | None = None):
    """Assembles the real subprocess command for a manifest spec. Training
    specs -> lm_pretrain_rd.py with the geo3/hard-select/probe flags;
    probe specs -> wave_neg1_trackb.py --probe-checkpoint."""
    if "probe_checkpoint" in spec:
        return [sys.executable, PROBE_TOOL, "--probe-checkpoint", spec["probe_checkpoint"],
                "--data-dir", data_dir, "--corpus", spec["corpus"],
                "--n-docs", str(spec.get("n_docs", 8)), "--seq-len", str(SEQ_LEN),
                "--chunk-size", str(GEO3_CHUNK_SIZE), "--k-sel", str(K_SEL),
                "--n-iter", str(GEO3_N_ITER), "--resid-tol", str(GEO3_RESID_TOL),
                "--out", os.path.join(out_dir, f"{spec['name']}.json")]
    cmd = [sys.executable, PRETRAIN,
           "--corpus", spec["corpus"], "--data-dir", data_dir,
           "--d-model", str(D_MODEL), "--d-state", str(D_STATE), "--n-layers", str(N_LAYERS),
           "--seq-len", str(SEQ_LEN), "--batch-size", str(BATCH_SIZE), "--steps", str(spec["steps"]),
           "--seed", str(spec["seed"]), "--internal-timeout", str(max(1, timeout - 30)),
           "--ckpt-every", str(spec.get("ckpt_every", CKPT_EVERY)),
           "--ckpt-dir", _ckpt_dir(out_dir),
           "--out", os.path.join(out_dir, f"{spec['name']}.json")]
    if spec.get("geo3_active"):
        cmd += ["--use-geo3-lm", "--geo3-k-sel", str(spec["geo3_k_sel"]),
                "--geo3-chunk-size", str(GEO3_CHUNK_SIZE),
                "--geo3-n-iter", str(spec["geo3_n_iter"]), "--geo3-resid-tol", str(GEO3_RESID_TOL),
                "--geo3-selection", "beta_topk"]
    if spec.get("hard_select_active"):
        cmd += ["--hard-select-active", "--hard-select-mechanism", spec["hard_select_mechanism"],
                "--hard-select-k-sel", str(spec["hard_select_k_sel"]),
                "--hard-select-chunk-size", str(GEO3_CHUNK_SIZE),
                "--hard-select-seed", str(spec["seed"])]
        if b_pinned is not None:
            cmd += ["--hard-select-b-pinned", str(b_pinned)]
    if spec.get("selection_probe"):
        cmd += ["--trackb-selection-probe", str(spec["selection_probe"])]
    if spec.get("nan_probe_counter"):
        cmd += ["--nan-probe-counter"]
    if spec.get("requires_override") and override_stamp is not None:
        cmd += ["--gate-override-reason", override_stamp["gate_override_reason"]]
    return cmd


def _run_wave(wave, manifest, out_dir, args, build_cmd_fn, timeout_fn):
    """CLONED from run_lm_rd_geo3_sweep.py::_run_wave (:418-516):
    exception-isolated launch, validity-checked resume, per-run timeout
    with GPU quarantine, guarded aggregate, ABORT on all-GPUs-quarantined."""
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir, args.gpu_offset):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate failed; nothing launched.\n")
        sys.exit(1)

    physical_gpus = list(range(args.gpu_offset, args.gpu_offset + args.gpus))
    slots = [g for _ in range(args.per_gpu) for g in physical_gpus]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done_trackb(out_dir, s)]
    dropped = [s for s in pending if "probe_checkpoint" in s and s["probe_checkpoint"] is None]
    pending = [s for s in pending if not ("probe_checkpoint" in s and s["probe_checkpoint"] is None)]
    for s in dropped:
        print(f"  SKIP {s['name']}: source run {s.get('source_run')!r} has no completed checkpoint "
              f"yet -- re-run this wave after it finishes.", flush=True)
    print(f"wave={wave}  manifest={len(manifest)}  pending={len(pending)}  dropped={len(dropped)}  "
          f"slots={n_slots} (gpus {physical_gpus} x {args.per_gpu} per-gpu)", flush=True)

    running, free, quarantined = {}, list(slots), []
    done_ct, failed, uid = 0, [], 0
    last_agg = time.time()

    try:
        while pending or running:
            while free and pending:
                gpu = free.pop()
                spec = pending.pop(0)
                timeout = args.timeout if args.timeout is not None else timeout_fn(spec)
                try:
                    lf = open(os.path.join(log_dir, f"{spec['name']}.log"), "w")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    proc = subprocess.Popen(build_cmd_fn(spec, out_dir, timeout, args.data_dir),
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
                if rc == 0 and is_done_trackb(out_dir, spec):
                    done_ct += 1
                else:
                    failed.append((spec["name"], rc))
            write_progress(out_dir, done_ct, len(failed), len(running), wave)
            if time.time() - last_agg > 120:
                aggregate(out_dir, manifest, failed, wave)
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
        print(f"ORCHESTRATOR CRASHED: {e!r} -- see CRASHED.txt; rerun --wave {wave} to resume "
              f"(validity-checked, bounded, not perpetual).", flush=True)
    finally:
        aggregate(out_dir, manifest, failed, wave)

    all_done = (done_ct == len(manifest)) and not dropped and not failed
    if all_done:
        with open(os.path.join(out_dir, "ALL_DONE"), "w") as f:
            f.write(f"wave {wave} complete: {done_ct}/{len(manifest)} runs, 0 failed, 0 dropped\n")
    print(f"\nWAVE {wave} DONE. {done_ct} succeeded, {len(failed)} failed, {len(dropped)} dropped. "
          f"ALL_DONE {'written' if all_done else 'NOT written (wave incomplete)'}.", flush=True)


# ---------------------------------------------------------------------------
# bands-pinned wave (CPU): assembles BANDS_PINNED-TrackB.json from Wave -1's
# own completed result JSONs + the cell1probe outputs + a live MC anchor
# recomputation (sec 4.3's writer, all inputs validity-checked first).
# ---------------------------------------------------------------------------

def assemble_bands_pinned(out_dir: str, mc_samples: int = 500_000) -> dict:
    import torch
    neg1_dir = os.path.join(out_dir, "waveneg1")
    pilot_path = os.path.join(neg1_dir, "wBneg1_reference_pilot.json")
    entmax_path = os.path.join(neg1_dir, f"wBneg1_probe_entmax_k{K_SEL}.json")
    cell1_dir = os.path.join(out_dir, "wavecell1probe")

    for p, what in ((pilot_path, "reference pilot"), (entmax_path, "entmax probe")):
        if not os.path.exists(p):
            print(f"ERROR: {what} result missing at {p!r} -- run --wave neg1 first.", file=sys.stderr)
            sys.exit(4)
    with open(pilot_path) as f:
        pilot = json.load(f)
    with open(entmax_path) as f:
        entmax = json.load(f)
    for r, what in ((pilot, "reference pilot"), (entmax, "entmax probe")):
        if r.get("complete") is not True:
            print(f"ERROR: {what} result is not complete -- the writer refuses partial pilots "
                  f"(sec 4.3 writer requirement).", file=sys.stderr)
            sys.exit(4)

    pilot_lists = bp.extract_selection_logpoint_lists(pilot, last_n=5)
    entmax_lists = bp.extract_selection_logpoint_lists(entmax, last_n=5)
    churn_null_a = bp.derive_churn_null_a(pilot_lists["churn"])
    pos_ceiling = bp.derive_pos_ceiling(pilot_lists["tv"])
    support_floor = bp.derive_support_floor(entmax_lists["support_p10_final"])

    cell1_paths = sorted(os.path.join(cell1_dir, f) for f in os.listdir(cell1_dir)
                          if f.endswith(".json") and f.startswith("cell1probe_")) \
        if os.path.isdir(cell1_dir) else []
    if not cell1_paths:
        print(f"ERROR: no cell1probe results under {cell1_dir!r} -- run --wave cell1probe over the "
              f"6 archived Wave C checkpoints first (B_pinned/cell1_ref_32 have no other source).",
              file=sys.stderr)
        sys.exit(4)
    masses, resids = [], []
    for p in cell1_paths:
        with open(p) as f:
            d = json.load(f)
        if d.get("complete") is not True:
            print(f"ERROR: cell1probe result {p!r} incomplete.", file=sys.stderr)
            sys.exit(4)
        for layer in (d.get("per_layer") or {}).values():
            masses.append(layer["per_chunk_total_mass"]["mean"])
            resids.append(layer["resid_mean"])
    b_pinned = bp.derive_b_pinned(torch.tensor(masses))
    rt = torch.tensor(resids, dtype=torch.float64)
    cell1_ref_32 = {"mean": rt.mean().item(),
                    "std": rt.std(unbiased=True).item() if rt.numel() > 1 else 0.0,
                    "n_layer_readings": int(rt.numel()), "n_checkpoints": len(cell1_paths)}

    mc = hs.mc_recompute_anchors(K_SEL, 64, n_samples=mc_samples, seed=0)
    doc = bp.write_bands_pinned_trackb(
        bands_pinned_out_path(out_dir), churn_null_a, pos_ceiling, support_floor, b_pinned, mc,
        cell1_ref_32, [pilot_path, entmax_path] + cell1_paths, k_sel=K_SEL)
    print(f"BANDS_PINNED-TrackB.json written at {bands_pinned_out_path(out_dir)}:\n"
          + json.dumps({k: doc[k] for k in ("churn_null_a", "positional_concentration_ceiling",
                                             "support_band", "b_pinned")}, indent=2),
          flush=True)
    return doc


# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------

def dry_run_preview(steps: int, factorial_steps: int, surviving_mechanisms, accept_budget_override: bool):
    m_neg1 = waveNeg1_manifest(steps)
    m_factorial = full_factorial_manifest(factorial_steps, surviving_mechanisms)
    m_4r = cell4r_reserve_manifest(factorial_steps)
    n_wave3_probes = sum(len(v) for v in m_factorial.values())

    print("=" * 70)
    print("TRACK B HARD-SELECTIVITY WAVE -- DRY-RUN PREVIEW (no launch this session)")
    print("=" * 70)

    h_neg1 = sum(spec_gpu_h(s) for s in m_neg1)
    print(f"\nWave -1 ({len(m_neg1)} cells: 5 mechanism probes + reference pilot + geo3-active "
          f"stability smoke, {steps} steps each) = {h_neg1:.3f} GPU-h; + cell1probe "
          f"(6 archived checkpoints x {_GPU_H_PER_PROBE} = {6 * _GPU_H_PER_PROBE:.2f} GPU-h, "
          f"forward-only) + MC anchor recomputation (CPU, free).")
    for spec in m_neg1:
        print(f"  - {spec['name']}")

    print(f"\nBANDS_PINNED-TrackB gate: written by --wave bands-pinned only after the pilot + "
          f"entmax probe + cell1probe results all validate complete; waves 1/2/4r REFUSE to "
          f"launch without a valid, hash-clean {bands_pinned_out_path('<out-dir>')} and thread "
          f"its b_pinned into every masking cell.")

    print(f"\nsec 5 2x2 factorial (surviving mechanisms={list(surviving_mechanisms)}, "
          f"steps={factorial_steps}, K_sel={K_SEL}):")
    total_h = h_neg1 + 6 * _GPU_H_PER_PROBE
    for cell, runs in m_factorial.items():
        h = sum(spec_gpu_h(s) for s in runs)
        total_h += h
        print(f"  Cell {cell}: {len(runs)} runs, {h:.3f} GPU-h")
        for spec in runs:
            override_tag = " [REQUIRES --accept-no-launch-reference-arm if gate=no_launch_redesign]" \
                if spec.get("requires_override") else ""
            print(f"    - {spec['name']}{override_tag}")
    print(f"  Cell 1 (baseline): re-measured via --wave cell1probe, no spawned training run.")

    h_wave3 = n_wave3_probes * _GPU_H_PER_PROBE
    total_h += h_wave3
    print(f"\nWave 3 (instrumentation probes over every completed factorial cell): "
          f"{n_wave3_probes} probes x {_GPU_H_PER_PROBE} = {h_wave3:.2f} GPU-h (sec 6.1's own "
          f"~1-2 GPU-h row).")

    h_4r = sum(spec_gpu_h(s) for s in m_4r)
    print(f"\nReserve Cell 4R ({len(m_4r)} runs, {h_4r:.3f} GPU-h, run ONLY if Cell 4 clears its "
          f"interaction bar, cut-eligible per sec 10 item 2) -- NOT included in the total below.")

    # AUDIT FIX (2026-07-04, minor): headroom computed LIVE from the maintained ledger, never a
    # hardcoded design-time figure (the sec 6 "~=33.6" was a post-rung-3 PROJECTION; the live
    # ledger is the authority, per sec 11 item 5's own standing re-derive rule).
    live_remaining = GPU_H_PROGRAM_CEILING - PROGRAM_SPENT_GPUH
    print(f"\nTOTAL projected (Wave -1 + cell1probe + factorial + Wave 3, excl. Cell 4R): "
          f"{total_h:.3f} GPU-h. Live remaining headroom = {live_remaining:.2f} GPU-h "
          f"(= ceiling {GPU_H_PROGRAM_CEILING:.0f} - live PROGRAM_SPENT_GPUH "
          f"{PROGRAM_SPENT_GPUH:.2f}; TRACKB_REDESIGN.md sec 6's ~=33.6 figure was a post-rung-3 "
          f"projection, superseded by this live ledger read).")
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
    ap.add_argument("--wave", choices=["neg1", "cell1probe", "bands-pinned", "1", "2", "3", "4r"],
                     default=None)
    ap.add_argument("--gpus", type=int, default=None,
                     help="GPU COUNT. REQUIRED for a real launch, NO DEFAULT ON PURPOSE: check "
                          "nvidia-smi immediately before every launch and pass the free set "
                          "explicitly.")
    ap.add_argument("--gpu-offset", type=int, default=None,
                     help="first physical GPU index. REQUIRED for a real launch, NO DEFAULT ON "
                          "PURPOSE -- see --gpus.")
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None, help="override Wave -1 probe step count")
    ap.add_argument("--factorial-steps", type=int, default=None,
                     help="override Wave 1/2/4r step count (default: default_steps())")
    ap.add_argument("--surviving-mechanisms", nargs="+", default=["hard_ste"],
                     help="Cell 2/4's surviving-candidate list, named after Wave -1's readout "
                          "(default: candidate 1 alone, the never-cut primary).")
    ap.add_argument("--cell1-checkpoints", nargs="+", default=None,
                     help="wave cell1probe: the 6 archived Wave C checkpoint paths (explicit, no "
                          "default -- pass what the archive actually holds).")
    ap.add_argument("--accept-no-launch-reference-arm", action="store_true",
                     help="TRACKB_REDESIGN.md sec 5.1 (Rev 2 M5): required to run Cell 3 if the "
                          "ORIGINAL geo3-in-LM gate verdict is no_launch_redesign. Stamps "
                          "gate_override=true/claim_tier=descriptive into every spawned Cell-3 "
                          "run's OWN result JSON at assembly time.")
    ap.add_argument("--accept-cell4r-reserve", action="store_true",
                     help="wave 4r only: an explicit human assertion that Cell 4 CLEARED its "
                          "sec 5.3 interaction bar (the reserve's own registered precondition).")
    ap.add_argument("--geo3-gate-json", default=None,
                     help="path to lm_geo3_wave_neg1_gate.py's output JSON (default: "
                          "<HERE>/results/lm_rd_geo3/wave_neg1_gate.json) -- Cell 3's override "
                          "target.")
    ap.add_argument("--ckpt-size-probe-dir", default=None,
                     help="directory holding an existing Wave-C-architecture checkpoint for the "
                          "disk gate's measured size (fallback: <HERE>/results/lm_rd/checkpoints).")
    ap.add_argument("--accept-budget-override", action="store_true")
    ap.add_argument("--timeout", type=float, default=None, help="override the per-run wall timeout (s)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else WAVE_NEG1_PROBE_STEPS
    factorial_steps = args.factorial_steps if args.factorial_steps is not None else default_steps()
    gate_json_path = args.geo3_gate_json or os.path.join(HERE, "results", "lm_rd_geo3",
                                                          "wave_neg1_gate.json")

    if args.dry_run:
        dry_run_preview(steps, factorial_steps, args.surviving_mechanisms, args.accept_budget_override)
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "bands-pinned":
        # CPU-only assembly -- no --gpus needed, no subprocess dispatch.
        assemble_bands_pinned(args.out_dir)
        return

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose): check nvidia-smi NOW and pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)

    def _disk_gate(manifest, wave_out_dir, label):
        train_specs = [s for s in manifest if "probe_checkpoint" not in s and s["steps"] > 0]
        if not train_specs:
            return
        probe_dir = args.ckpt_size_probe_dir or _ckpt_dir(wave_out_dir)
        try:
            size = find_ckpt_size_bytes(probe_dir)
        except FileNotFoundError:
            fallback = os.path.join(HERE, "results", "lm_rd", "checkpoints")
            size = find_ckpt_size_bytes(fallback)   # raises with remedy if this too is empty
        worst = max(train_specs, key=lambda s: s["steps"] // s.get("ckpt_every", CKPT_EVERY))
        projected = projected_ckpt_bytes(len(train_specs), worst["steps"],
                                          worst.get("ckpt_every", CKPT_EVERY), size)
        os.makedirs(_ckpt_dir(wave_out_dir), exist_ok=True)
        report = disk_space_check(_ckpt_dir(wave_out_dir), projected, label)
        print(f"DISK GATE ({label}): projected {report['projected_ckpt_bytes']/1e9:.2f} GB x "
              f"{report['safety_factor']} = {report['required_bytes']/1e9:.2f} GB required, "
              f"{report['free_bytes']/1e9:.2f} GB free -> {'OK' if report['ok'] else 'REFUSED'}",
              flush=True)
        if not report["ok"]:
            print(f"ERROR: disk-space gate REFUSED {label} -- free up {report['resolved_ckpt_dir']} "
                  f"or lower the checkpoint cadence.", file=sys.stderr)
            sys.exit(8)

    if args.wave == "neg1":
        manifest = waveNeg1_manifest(steps)
        out_dir = os.path.join(args.out_dir, "waveneg1")
        os.makedirs(out_dir, exist_ok=True)
        assert run_eot_forced_selection_negative_smoke(), \
            "M6 EOT-forced-selection negative smoke FAILED -- refusing Wave -1 (forced selection is broken)"
        print("EOT-forced-selection negative smoke: PASS", flush=True)
        stress_dir = os.path.join(args.data_dir, "reasoning_stress_eot")
        if not os.path.isdir(stress_dir):
            print(f"ERROR: stress-slice corpus not found at {stress_dir!r} -- the stability smoke "
                  f"trains on it. Run 'python wave_neg1_trackb.py --build-stress-slice --data-dir "
                  f"{args.data_dir}' first (CPU, reads the real openr1 corpus).", file=sys.stderr)
            sys.exit(9)
        run_epoch_cap_gate(manifest, args.data_dir, "wave neg1")
        budget_guard(sum(spec_gpu_h(s) for s in manifest), "wave neg1", args.accept_budget_override)
        _disk_gate(manifest, out_dir, "wave neg1")
        _run_wave("neg1", manifest, out_dir, args, build_cmd_trackb, default_timeout_pretrain)

    elif args.wave == "cell1probe":
        assert args.cell1_checkpoints, "--cell1-checkpoints is required for wave cell1probe"
        manifest = cell1probe_manifest(args.cell1_checkpoints)
        out_dir = os.path.join(args.out_dir, "wavecell1probe")
        os.makedirs(out_dir, exist_ok=True)
        budget_guard(sum(spec_gpu_h(s) for s in manifest), "wave cell1probe", args.accept_budget_override)
        _run_wave("cell1probe", manifest, out_dir, args, build_cmd_trackb, default_timeout_probe)

    elif args.wave == "1":
        bands = require_bands_pinned(args.out_dir)
        b_pinned = bands["b_pinned"]["b_pinned"]
        manifest = (cell2_manifest(factorial_steps, tuple(args.surviving_mechanisms))
                    + cell2r_manifest(factorial_steps) + comparator_manifest(factorial_steps))
        out_dir = os.path.join(args.out_dir, "wave1")
        os.makedirs(out_dir, exist_ok=True)
        run_epoch_cap_gate(manifest, args.data_dir, "wave 1")
        budget_guard(sum(spec_gpu_h(s) for s in manifest), "wave 1", args.accept_budget_override)
        _disk_gate(manifest, out_dir, "wave 1")
        _run_wave("1", manifest, out_dir, args,
                  lambda spec, od, t, dd: build_cmd_trackb(spec, od, t, dd, b_pinned=b_pinned),
                  default_timeout_pretrain)

    elif args.wave == "2":
        bands = require_bands_pinned(args.out_dir)
        b_pinned = bands["b_pinned"]["b_pinned"]
        verdict, _gate = load_gate_verdict(gate_json_path)
        print(f"Original geo3-in-LM gate verdict (from {gate_json_path}): {verdict}", flush=True)
        override_stamp = refuse_or_override_cell3(verdict, gate_json_path,
                                                   args.accept_no_launch_reference_arm)
        check_stability_smoke_gate(args.out_dir)
        assert run_eot_forced_selection_negative_smoke(), \
            "M6 EOT-forced-selection negative smoke FAILED -- refusing wave 2 (Cell 4 forces selection)"
        print("EOT-forced-selection negative smoke: PASS", flush=True)
        manifest = (cell3_manifest(factorial_steps)
                    + cell4_manifest(factorial_steps, tuple(args.surviving_mechanisms)))
        out_dir = os.path.join(args.out_dir, "wave2")
        os.makedirs(out_dir, exist_ok=True)
        run_epoch_cap_gate(manifest, args.data_dir, "wave 2")
        budget_guard(sum(spec_gpu_h(s) for s in manifest), "wave 2", args.accept_budget_override)
        _disk_gate(manifest, out_dir, "wave 2")
        _run_wave("2", manifest, out_dir, args,
                  lambda spec, od, t, dd: build_cmd_trackb(spec, od, t, dd, b_pinned=b_pinned,
                                                            override_stamp=override_stamp),
                  default_timeout_pretrain)

    elif args.wave == "3":
        w1 = (cell2_manifest(factorial_steps, tuple(args.surviving_mechanisms))
              + cell2r_manifest(factorial_steps) + comparator_manifest(factorial_steps))
        w2 = (cell3_manifest(factorial_steps)
              + cell4_manifest(factorial_steps, tuple(args.surviving_mechanisms)))
        manifest = wave3_manifest(args.out_dir, w1, w2)
        out_dir = os.path.join(args.out_dir, "wave3")
        os.makedirs(out_dir, exist_ok=True)
        budget_guard(sum(spec_gpu_h(s) for s in manifest), "wave 3", args.accept_budget_override)
        _run_wave("3", manifest, out_dir, args, build_cmd_trackb, default_timeout_probe)

    else:   # 4r
        assert args.accept_cell4r_reserve, (
            "wave 4r is the RESERVE cell (sec 5.1): it runs ONLY if Cell 4 cleared its "
            "interaction bar -- pass --accept-cell4r-reserve to assert that explicitly (a human "
            "decision, never a default).")
        bands = require_bands_pinned(args.out_dir)
        b_pinned = bands["b_pinned"]["b_pinned"]
        check_stability_smoke_gate(args.out_dir)
        manifest = cell4r_reserve_manifest(factorial_steps)
        out_dir = os.path.join(args.out_dir, "wave4r")
        os.makedirs(out_dir, exist_ok=True)
        run_epoch_cap_gate(manifest, args.data_dir, "wave 4r")
        budget_guard(sum(spec_gpu_h(s) for s in manifest), "wave 4r", args.accept_budget_override)
        _disk_gate(manifest, out_dir, "wave 4r")
        _run_wave("4r", manifest, out_dir, args,
                  lambda spec, od, t, dd: build_cmd_trackb(spec, od, t, dd, b_pinned=b_pinned),
                  default_timeout_pretrain)


if __name__ == "__main__":
    main()
