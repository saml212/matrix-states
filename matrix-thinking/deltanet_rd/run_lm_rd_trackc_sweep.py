"""run_lm_rd_trackc_sweep.py -- bounded, human-gated wave orchestrator for
Track C / the scaling ladder (SCALE_TRANSFER_DESIGN.md sec 5). Original
build covered rung 1 (~98M params) + the REQUIRED small-model same-mix
control cell (sec 5.6 Wave 1, MAJOR-5) ONLY; Wave 1 has since completed
and closed (sec 5.9). trackC-rung23-build extended the launcher with REAL
--wave 2 / --wave 3 paths for rung 2 (~392M) and rung 3 (~1.31B), per
lm_rd_rung_configs.py's RUNGS table -- rungs 2/3 are no longer
registered-but-unbuilt; BUILD_SCOPE_RUNGS now covers all three
(lm_rd_rung_configs.py), so every smoke gate call now verifies all three
rungs' parameter counts, not just rung 1's.

THIS BUILD (sec 5.6 AMENDMENT, Rev 2.1, 2026-07-04, registered before any
Wave-2/3 data existed) makes three changes: (1) rung 2's seed count goes
1->3 per corpus (2->6 runs, WAVE23_SEEDS_BY_RUNG) to match rung 1's
evidentiary tier; rung 3 stays 1 seed as registered. (2) waves >=2 now
train on the EXTENDED mixes (WAVE23_CORPORA), not the original MIX_CORPORA
-- the epoch-cap remedy for rung 2's 1.5B-token/run budget exceeding the
originals' <=5-epoch ceiling (sec 5.4); wave 1 / calibration / control
paths are UNCHANGED, still MIX_CORPORA. (3) a new `--wave mixcontrol`
re-isolates the mix axis at Wave-C scale for the extended mixes (CONTROL_CFG
x WAVE23_CORPORA x 3 seeds, control-length steps) -- mirrors MAJOR-5's
original control logic, applied to the new corpora.

CLONE of run_lm_rd_geo3_sweep.py's / run_lm_rd_sweep.py's robustness
pattern (smoke gate, exception-isolated launch, validity-checked resume,
per-run timeout with GPU quarantine, guarded aggregate, REQUIRED
--gpus/--gpu-offset with NO defaults) -- deliberately NOT a cross-script
import (this codebase's own pod-safety convention, restated in both of
those files' own docstrings).

LEARNED FROM WAVE 1'S OWN HISTORY (trackC-audit finding #1, restated here
because it is exactly the mistake this build must not repeat): an earlier
revision of this file gated --wave 1 with real-looking preconditions but
contained NO manifest-build/launch code behind the gate at all -- an
unconditional `sys.exit(4)` fired regardless of whether the gates were
satisfied. Every wave below (2 and 3 included) is verified, by direct
re-read of this file after editing, to have a REAL `_run_wave(...)` call
reachable when its gates pass -- a gate with nothing behind it is a no-op
disguised as a guardrail.

Waves (in run order):
  calibration (Wave -1, sec 5.6's blocking calibration row): TWO-POINT real
    training runs (trackC-audit finding #2 -- CALIBRATION_TWO_POINT_STEPS,
    two single-checkpoint runs per config at different step counts), on
    openr1-mix, seed 0 only -- measures REAL per_step_s/per_ckpt_s (solved
    from the two points, not blended) + tok/s + peak memory
    (lm_pretrain_rd.py's own peak_memory_*_bytes fields) before ANY
    full-budget run is priced or launched. `--calib-rungs` selects WHICH
    configs' two-point cells are included (default "1,control" --
    UNCHANGED from the original build, so a bare `--wave calibration`
    invocation with no new flags is a byte-for-byte behavioral no-op vs.
    the pre-extension file -- rung 2/3 calibration requires the explicit
    opt-in `--calib-rungs 2` / `--calib-rungs 3`, matching this codebase's
    "no accidental launch, no silent defaults on new/larger cells"
    convention). Already-complete cells resume-skip regardless of
    inclusion (cheap to re-list). Cheap (<2 GPU-h total per rung) -- MAY be
    launched any time; report its measured numbers.
  1 (rung 1 + control cell, FULL manifest): CLOSED -- already ran to
    completion (sec 5.9). Behavior UNCHANGED by this build.
  2 (rung 2, FULL manifest -- 6 runs, 2 EXTENDED-mix corpora x 3 seeds per
    the Rev 2.1 amendment item 1, no control cell at this rung per
    MAJOR-5/sec 8) and
  3 (rung 3, FULL manifest -- 2 runs, 2 EXTENDED-mix corpora x 1 seed,
    unchanged from sec 5.6's table): built, smoke-gated, dry-run-previewable,
    and gated on: (a) an existing calibration.json with populated
    `timing_constants` for the rung (sec 9's hard rule), (b) --rung{2,3}-steps
    supplied explicitly (no silent fallback to the placeholder), (c) a
    PASSING memory-headroom readout at the real batch size, recomputed live
    from calibration.json's own recorded peak-memory cells (not assumed),
    (d) a PASSING epoch-cap check on BOTH extended-mix corpora (WAVE23_CORPORA,
    amendment item 2) at the resulting per-run token budget (sec 5.4's
    <=5-physical-epoch discipline, read live from each mix's own meta.json),
    and (e) a budget guard: PROGRAM_SPENT_GPUH (this file's own maintained
    tracker) + this wave's projected GPU-h (computed from the calibration's
    measured timing constants and the ACTUAL manifest length -- never a
    hardcoded run count -- PRINTED before any launch decision) must not
    exceed the program's 300 GPU-h ceiling (sec 7) without an explicit
    --accept-budget-override. ANY of (a)-(e) failing refuses the launch with
    the design's own registered remedy, never a silent proceed. THIS SESSION
    DOES NOT LAUNCH --wave 2 or --wave 3 for real -- only rung 2's
    calibration cells are launched (see STATE.md / the audit report for the
    measured numbers); the gates above are built and exercised via
    --dry-run + the calibration run only.
  mixcontrol (Rev 2.1 amendment item 3, NEW): CONTROL_CFG (Wave C's own
    ~14M scale) retrained on WAVE23_CORPORA (the extended mixes) x 3 seeds =
    6 cells, at the SAME control-length step budget as Wave 1's own
    same-mix control cell (default_control_steps()) -- re-isolates the mix
    axis at Wave-C scale for the extended mixes, mirroring MAJOR-5's
    original control logic. Gated on the SAME (a)/(c)/(d)/(e) chain as
    waves 2/3 (keyed to the 'control' calibration/timing cell, already
    banked from Wave 1 -- no new calibration needed); the epoch-cap check
    trivially passes at control-length steps but is NOT skipped. THIS
    SESSION DOES NOT LAUNCH --wave mixcontrol for real -- --dry-run preview
    only.
  4 (fix-effect / geo3-at-scale, sec 5.5 item 3): HARD REFUSED
    unconditionally by this file -- Track B's own Wave -1 gate returned
    `no_launch_redesign` (EXPERIMENT_LOG.md, "SCALE-TRANSFER Track B ...
    HARD NO-LAUNCH"), so sec 5.5 item 3 (which applies Track B's
    construction at scale) has no validated construction to transplant.
    This is checked LIVE against Track B's own gate JSON (not just cited
    from memory) every time --wave 4 is invoked.

Batch sizing (sec 5 is SILENT on rung-2/3 batch size -- an attack-yourself
item for this build, not an oversight): the design registers SEQ_LEN=512
throughout but never specifies a per-GPU batch for the larger rungs. The
MINIMAL REGISTERED CHOICE, per this build's own brief, is to reuse rung-1/
control's existing BATCH_SIZE=32 unchanged (BATCH_SIZE_BY_RUNG below) --
this is verified, not assumed, by the memory-headroom check derived from
each rung's own two-point calibration cells (run at that same real batch
size) before any wave-2/3 launch is gated open. If a rung's headroom check
ever fails, BATCH_SIZE_BY_RUNG's entry for that rung must be lowered and
the change documented here and in that rung's calibration.json -- never
silently.

Usage (GPU list is an example -- check nvidia-smi first, per house rule):
  python run_lm_rd_trackc_sweep.py --dry-run
  python run_lm_rd_trackc_sweep.py --wave calibration --out-dir results/lm_rd_trackc --gpus 2 --gpu-offset 0                       # rung 1 + control (unchanged default)
  python run_lm_rd_trackc_sweep.py --wave calibration --calib-rungs 2 --out-dir results/lm_rd_trackc --gpus 1 --gpu-offset 0        # rung 2 two-point cells only
  python run_lm_rd_trackc_sweep.py --wave 1 --out-dir results/lm_rd_trackc --gpus 6 --gpu-offset 0 --rung1-steps N   # CLOSED, sec 5.9 -- shown for reference only
  python run_lm_rd_trackc_sweep.py --wave 2 --out-dir results/lm_rd_trackc --gpus 6 --gpu-offset 0 --rung2-steps N   # gated per the (a)-(e) chain above; NOT launched this session
  python run_lm_rd_trackc_sweep.py --wave 3 --out-dir results/lm_rd_trackc --gpus 6 --gpu-offset 0 --rung3-steps N   # gated per the (a)-(e) chain above; NOT launched this session
  python run_lm_rd_trackc_sweep.py --wave mixcontrol --out-dir results/lm_rd_trackc --gpus 6 --gpu-offset 0          # Rev 2.1 amendment item 3; gated per the (a)/(c)/(d)/(e) chain; NOT launched this session
  python run_lm_rd_trackc_sweep.py --wave 4                                                          # ALWAYS refused
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
ATTRACTOR_PROBE = os.path.join(HERE, "lm_attractor_probe_rd.py")
RUNG_CONFIGS = os.path.join(HERE, "lm_rd_rung_configs.py")
BUILD_MIX = os.path.join(HERE, "build_mix_corpora_rd.py")
TRACKB_GATE_JSON_DEFAULT = os.path.join(HERE, "results", "lm_rd_geo3", "wave_neg1_gate.json")

sys.path.insert(0, HERE)  # pod-safe import of the config-layer module ONLY (not another sweep script)
from lm_rd_rung_configs import RUNGS, VOCAB_SIZE, verify_param_count  # noqa: E402

RUNG = 1                                          # Wave 1's already-closed rung (kept -- existing
                                                   # call sites reference RUNG_CFG by this name);
                                                   # rungs 2/3 are RUNG2_CFG/RUNG3_CFG below, both
                                                   # now in build scope (trackC-rung23-build).
RUNG_CFG = RUNGS[RUNG]
RUNG2_CFG = RUNGS[2]
RUNG3_CFG = RUNGS[3]
CONTROL_CFG = {"d_model": 256, "d_state": 64, "n_layers": 2}   # Wave C's own scale (sec 5.6, MAJOR-5)

MIX_CORPORA = ("openr1-mix", "wikitext-mix")      # sec 5.6 Wave 1: "openr1-mix, wikitext/finewebedu-mix"
                                                   # -- wave 1 / calibration / control paths ONLY;
                                                   # waves >=2 use WAVE23_CORPORA below (amendment item 2).
SEEDS = (0, 1, 2)                                 # rung 1 / control (sec 5.6: "2 corpora x 3 seeds")

# sec 5.6 AMENDMENT (Rev 2.1, 2026-07-04) item 2: waves >=2 train on the EXTENDED mixes, NOT
# MIX_CORPORA -- rung 2's 1.5B-token/run target exceeds the original mixes' <=5-epoch ceilings
# (sec 5.4), and pulling more augmentation (not more epochs) is the registered remedy. A NEW
# constant, not a redefinition of MIX_CORPORA, so wave 1 / calibration / control stay byte-identical.
WAVE23_CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")

# sec 5.6 AMENDMENT (Rev 2.1) item 1: rung 2's seed count goes 1->3 per corpus (matches rung 1's
# evidentiary tier for the primary monotonic-trend readout); rung 3 stays 1 seed as registered
# (its pricing is the open budget question, item 4). Keyed by rung so every call site derives the
# seed count (and therefore the manifest length / run count / projected GPU-h) from THIS dict,
# never a hardcoded literal.
WAVE23_SEEDS_BY_RUNG = {2: (0, 1, 2), 3: (0,)}
SEQ_LEN = 512
BATCH_SIZE = 32

# sec 5 is SILENT on rung-2/3 batch size (module docstring's "Batch sizing" paragraph). Minimal
# registered choice: reuse BATCH_SIZE=32 for every rung unless a rung's own memory-headroom check
# (derived from its two-point calibration cells, run at this same batch) fails -- then lower ONLY
# that rung's entry here and document why. No entry has been lowered as of this build; see
# calibration.json's rung-2 memory-headroom readout for the empirical basis.
BATCH_SIZE_BY_RUNG = {1: BATCH_SIZE, 2: BATCH_SIZE, 3: BATCH_SIZE}

# sec 5.6: "the Wave-C-scale ... architecture retrained on the SAME augmented mixes ... at Wave C's
# measured ~4.6 min/run" -- the control cell REUSES Wave C's own exact token budget (this codebase's
# established TARGET_TOKENS=100_000_000 constant, run_lm_rd_sweep.py).
CONTROL_TARGET_TOKENS = 100_000_000

# Rung 1's OWN token budget is DELIBERATELY UNCALIBRATED here (sec 5.6: "Wave -1's measured
# throughput is the actual pricing authority, not this table"; sec 9: no Wave 1+ manifest may
# launch on an estimated figure without first recording Wave -1's measured numbers). This constant
# is a documented PLANNING PLACEHOLDER for --dry-run cost preview ONLY -- a real Wave 1 launch
# requires --rung1-steps (an explicit, human-supplied value informed by calibration.json), which
# main() enforces (no silent fallback to this placeholder once a real launch is attempted).
RUNG1_TARGET_TOKENS_PLACEHOLDER = 100_000_000

# Rung 2/3's own planning placeholders, sec 5.6's table ("~1.5B tokens/run" / "~3B tokens/run") --
# same UNCALIBRATED-placeholder discipline as rung 1 above: --dry-run preview ONLY, never a silent
# fallback for a real --wave 2/--wave 3 launch (--rung{2,3}-steps is REQUIRED there, enforced below).
RUNG2_TARGET_TOKENS_PLACEHOLDER = 1_500_000_000
RUNG3_TARGET_TOKENS_PLACEHOLDER = 3_000_000_000

# ---------------------------------------------------------------------------
# Epoch-cap discipline (sec 5.4: "cap any single source's repetition at <=5 physical epochs ...
# the remainder is drawn from OpenWebMath/FineWeb-Edu") -- read LIVE from each mix corpus's own
# meta.json (build_mix_corpora_rd.py already computes and records "mix_total_train_tokens" there;
# recomputed here from that field rather than trusted-by-citation, since a rebuilt/re-augmented mix
# would change it). DUPLICATED from lm_pretrain_rd.py's own CORPUS_DIRS mapping (pod-safe, no heavy
# cross-import of a torch/fla-importing module just to read a directory name -- this file's own
# stated convention).
# ---------------------------------------------------------------------------
MIX_CORPUS_DIRS = {
    "openr1-mix": "reasoning_mix_eot", "wikitext-mix": "wikitext103_mix_eot",
    # sec 5.6 amendment (Rev 2.1) item 2: waves >=2 (and mixcontrol) read their epoch-cap ceiling
    # through these entries -- gate (d) resolves WAVE23_CORPORA's meta.json via this same mapping.
    "openr1-mix-ext": "reasoning_mix_eot_extended", "wikitext-mix-ext": "wikitext103_mix_eot_extended",
}
EPOCH_CAP = 5

# ---------------------------------------------------------------------------
# Program-wide GPU-h budget guard (sec 7's 300 GPU-h program ceiling). PROGRAM_SPENT_GPUH is a
# MAINTAINED constant, NOT auto-computed from any log -- a human/orchestrator updates it as real
# spend accrues across ALL FOUR tracks. As of this build: Track A ~0 (zero-GPU by design) + Track B
# ~1 (Wave -1 calibration only; Track B's own gate returned HARD NO-LAUNCH past that, sec 11) +
# Track C ~31 (rung-1 Wave 1's full manifest + control + the sec 5.9 attractor probe, already
# banked/closed) + Track D ~1 (Phase 1 only, sec 6.8) + this build's rung-2 calibration (cold pair
# 104s, discarded as invalid, + warm pair 66s + process overheads ~= 0.1 GPU-h measured, well under
# the ~0.5 pre-estimate) ~= 33.1 GPU-h. UPDATE THIS after any further real spend.
PROGRAM_SPENT_GPUH = 33.1
GPU_H_PROGRAM_CEILING = 300.0

CALIBRATION_STEPS_DEFAULT = 200
CALIBRATION_CKPT_EVERY = 100                      # legacy single-point knobs -- retained for CLI
                                                   # back-compat (--calibration-steps/-ckpt-every) but
                                                   # NO LONGER drive the actual calibration manifest,
                                                   # see CALIBRATION_TWO_POINT_STEPS below (trackC-audit
                                                   # finding #2: a single blended run cannot separate
                                                   # per-step compute time from one-time per-checkpoint
                                                   # overhead -- eval + rank-stat sampling + torch.save).

# PLANNING-ONLY per-step/per-checkpoint constants (rung 1 had NEVER been timed on this box before
# this build session -- these exist solely to size the CALIBRATION run's OWN --internal-timeout,
# generously, not to price a real wave -- chicken-and-egg: the calibration run needs *a* timeout
# before its own real numbers exist). A 98M-param model at bf16-kernel-boundary is unlikely to be
# faster per-step than Wave C's measured ~0.037s/step (14M params) -- 5x generous headroom assumed
# here. Validated post-hoc (trackC-audit, this session): measured rung1 per_step_s~=0.237,
# control per_step_s~=0.044 -- both comfortably under this 0.25 generous ceiling.
_CALIBRATION_PER_STEP_S_GENEROUS = 0.25
_CALIBRATION_PER_CKPT_S_GENEROUS = 60.0

# trackC-rung23-build: rung1/control's 0.25s/60s generous ceiling above was fit (and validated
# post-hoc) to a 98M-param model -- rung 2 (392M, ~4x rung1's approx_params) and rung 3 (1.31B,
# ~13.4x) would very plausibly BREACH a flat 0.25s/step ceiling if compute-bound scaling holds
# (this file's own FLOPs estimate puts rung 2 at ~5.3x, rung 3 at ~14.6x rung 1's per-token FLOPs,
# CALIBRATION_TWO_POINT_STEPS' own comment) -- reusing the flat constant would risk the
# CALIBRATION run itself being killed by its own --internal-timeout before finishing. Scaled here
# by each rung's approx-param ratio (a param-count-proportional FLOPs proxy for this dense-matmul
# architecture) x an EXTRA 2x safety margin on top of rung1's already-generous constant (this
# scaling is itself unvalidated until a rung's calibration actually completes -- unlike rung1/
# control's own post-hoc-checked numbers).
_RUNG1_APPROX_PARAMS = RUNGS[1]["approx_params"]


def _calibration_generous_timing(tag_base: str) -> tuple:
    """Returns (per_step_s_generous, per_ckpt_s_generous) for sizing a calibration cell's OWN
    --internal-timeout (never a real wave's timeout, which always uses MEASURED constants)."""
    if tag_base in ("calib_rung1", "calib_control"):
        return _CALIBRATION_PER_STEP_S_GENEROUS, _CALIBRATION_PER_CKPT_S_GENEROUS
    cfg = CALIBRATION_RUNG_CFGS[tag_base]
    ratio = cfg["approx_params"] / _RUNG1_APPROX_PARAMS
    return _CALIBRATION_PER_STEP_S_GENEROUS * ratio * 2.0, _CALIBRATION_PER_CKPT_S_GENEROUS * ratio * 2.0

# ---------------------------------------------------------------------------
# Two-point timing calibration (trackC-audit finding #2). A SINGLE run of N steps with
# ckpt_every=N (exactly one checkpoint, at the end) mixes per-step compute time with the one-time
# per-checkpoint overhead (eval on both corpora + rank-stat sampling + torch.save) into a single
# wall_s number -- Wave 1's real per-run timeout can't be safely derived from that blend alone.
# Running TWO such single-checkpoint points per config at DIFFERENT step counts and solving the
# resulting 2x2 linear system isolates the two constants cleanly:
#   wall_s = per_step_s * steps + per_ckpt_s * 1            (exactly one checkpoint per point)
#   per_step_s = (wall_B - wall_A) / (steps_B - steps_A)
#   per_ckpt_s = wall_A - per_step_s * steps_A
# Step counts chosen small (rung 1: seconds-to-tens-of-seconds; control: single-digit seconds) so
# the whole 4-point manifest stays far under the calibration wave's own <2 GPU-h ceiling, while far
# enough apart that per-step noise doesn't dominate the subtraction. Validated (trackC-audit, this
# session) by extrapolating both points to the ORIGINAL single-point calibration's own 200-step/
# 2-checkpoint config: predicted vs. actually-measured wall_s agreed to 0.2% (rung 1) and ~7%
# (control) -- both comfortably inside the 1.6x launch margin applied below.
CALIBRATION_TWO_POINT_STEPS = {
    "calib_rung1": (40, 120),
    "calib_control": (100, 300),
    # trackC-rung23-build: rung 2/3 points chosen from the per-layer FLOPs ratio vs. rung 1
    # (8*d_model^2*n_layers + 4*d_model*d_state*n_layers, sec 5.3's own formula) -- rung 2 is
    # ~5.3x rung 1's per-token FLOPs, rung 3 ~14.6x -- so fewer steps are used per point to keep
    # each point's wall time in the same seconds-to-tens-of-seconds band as rung 1's own points,
    # not because the two-point METHOD changes. Unvalidated until run for real (unlike rung
    # 1/control's cross-check against an original single-point run) -- report actual measured
    # wall_s alongside these when the calibration is run, don't just cite this comment.
    "calib_rung2": (10, 40),
    "calib_rung3": (5, 20),
}

CALIBRATION_RUNG_CFGS = {
    "calib_rung1": RUNG_CFG, "calib_control": CONTROL_CFG,
    "calib_rung2": RUNG2_CFG, "calib_rung3": RUNG3_CFG,
}

# Per-tag calibration batch size -- mirrors BATCH_SIZE_BY_RUNG so each rung's calibration cells are
# run at the SAME real batch size its wave would use (the whole point of the memory-headroom
# check); the control cell has no separate BATCH_SIZE_BY_RUNG entry (it isn't a "rung"), so it uses
# the global BATCH_SIZE directly, matching its original (pre-extension) behavior exactly.
CALIBRATION_BATCH_SIZE = {
    "calib_rung1": BATCH_SIZE_BY_RUNG[1], "calib_control": BATCH_SIZE,
    "calib_rung2": BATCH_SIZE_BY_RUNG[2], "calib_rung3": BATCH_SIZE_BY_RUNG[3],
}

# UNCHANGED default (trackC-rung23-build): a bare `--wave calibration` with no `--calib-rungs`
# behaves EXACTLY as the pre-extension file did -- rung 1 + control only. Rung 2/3 calibration is
# opt-in ONLY (`--calib-rungs 2` / `--calib-rungs 3` / `--calib-rungs 1,control,2,3`), never launched
# by a default invocation -- this codebase's own "no accidental launch of a new/larger cell"
# convention (REQUIRED --gpus/--gpu-offset with no defaults, restated here for the same reason).
CALIBRATION_TAGS_DEFAULT = ("calib_rung1", "calib_control")
_CALIB_RUNG_ALIAS = {"1": "calib_rung1", "2": "calib_rung2", "3": "calib_rung3", "control": "calib_control"}


def parse_calib_rungs(spec: str) -> tuple[str, ...]:
    """Parses --calib-rungs' comma-separated token list (from {"1","2","3","control"}) into the
    CALIBRATION_TWO_POINT_STEPS tag names calibration_manifest() expects. Raises SystemExit with a
    clear message on an unknown token rather than silently ignoring it (a typo'd rung number should
    never silently calibrate the WRONG thing)."""
    tags = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok not in _CALIB_RUNG_ALIAS:
            print(f"ERROR: --calib-rungs token {tok!r} not recognized -- expected a comma list from "
                  f"{sorted(_CALIB_RUNG_ALIAS)}.", file=sys.stderr)
            sys.exit(1)
        tags.append(_CALIB_RUNG_ALIAS[tok])
    return tuple(tags)


# House convention (CLAUDE.md-adjacent, restated in the task brief): a real launch's timeout is the
# measured cost times this margin, not the raw measured cost.
LAUNCH_TIMEOUT_MARGIN = 1.6


def default_control_steps() -> int:
    return max(1, CONTROL_TARGET_TOKENS // (BATCH_SIZE * SEQ_LEN))


def default_rung1_steps_placeholder() -> int:
    return max(1, RUNG1_TARGET_TOKENS_PLACEHOLDER // (BATCH_SIZE * SEQ_LEN))


def default_rung23_steps_placeholder(rung: int) -> int:
    target = {2: RUNG2_TARGET_TOKENS_PLACEHOLDER, 3: RUNG3_TARGET_TOKENS_PLACEHOLDER}[rung]
    return max(1, target // (BATCH_SIZE_BY_RUNG[rung] * SEQ_LEN))


# trackC-audit (this session, live-measured via `/usr/bin/time -v` on a real rung-1 10-step run):
# total process wall clock 14.33s vs. train()'s OWN internal wall_s of 5.94s -- ~8.4s of FIXED
# per-process overhead (python/torch/fla imports, corpus tensor load off disk, model construction +
# .to(device), first-kernel CUDA warmup) that happens BEFORE train()'s t0 and so is INVISIBLE to the
# two-point method above (which only differences two in-train() wall_s samples). Negligible at
# production step counts (6103 steps ~= 24-39 min, <1% of runtime, already well inside the 1.6x
# margin) but NOT negligible at small step counts -- confirmed live: a 5-step wave-1 smoke launch
# with this term OMITTED derived a ~8s timeout and failed all 12 cells (killed before finishing
# startup); adding this term (generous headroom over the measured ~8.4s) fixed it without an
# explicit --timeout override.
PROCESS_STARTUP_OVERHEAD_S_GENEROUS = 30.0


def default_timeout_pretrain(steps: int, ckpt_every: int, per_step_s: float, per_ckpt_s: float,
                              margin: float = 1.6,
                              startup_overhead_s: float = PROCESS_STARTUP_OVERHEAD_S_GENEROUS) -> int:
    n_ckpts = steps // ckpt_every + 1
    base = (startup_overhead_s + per_step_s * steps + n_ckpts * per_ckpt_s) * margin
    return int(base)


# ---------------------------------------------------------------------------
# Track B cross-check (sec 5.5 item 3 / this build's own scope statement):
# Wave 4 is refused unconditionally, but the refusal reads Track B's REAL
# gate JSON live (not just a cited verdict) so a future re-run of Track B
# that overturns the no-launch verdict is reflected automatically.
# ---------------------------------------------------------------------------

def _trackb_gate_status(gate_json_path: str) -> dict:
    if not os.path.exists(gate_json_path):
        return {"found": False, "verdict": None}
    try:
        with open(gate_json_path) as f:
            gate = json.load(f)
        return {"found": True, "verdict": gate.get("gate_verdict")}
    except Exception as e:
        return {"found": True, "verdict": None, "parse_error": repr(e)}


def refuse_wave4(gate_json_path: str) -> None:
    status = _trackb_gate_status(gate_json_path)
    print("=" * 70, file=sys.stderr)
    print("TRACK C WAVE 4 (fix-effect / geo3-at-scale, sec 5.5 item 3) IS GATED OUT.", file=sys.stderr)
    print(f"Track B gate JSON: {gate_json_path}", file=sys.stderr)
    print(f"Track B status: {status}", file=sys.stderr)
    if status.get("verdict") == "no_launch_redesign":
        print("Confirmed LIVE: Track B's own Wave -1 gate returned 'no_launch_redesign' (HARD "
              "NO-LAUNCH -- EXPERIMENT_LOG.md, 'SCALE-TRANSFER Track B ... HARD NO-LAUNCH'). There "
              "is no validated geo3-in-LM construction for Wave 4 to transplant to rung 1/3 (sec "
              "5.5 item 3's own dependency, sec 11's sequencing rule).", file=sys.stderr)
    elif not status.get("found"):
        print("Track B's gate JSON was not found at the expected path -- Wave 4 stays refused by "
              "DEFAULT (sec 5.5 item 3 requires a POSITIVE, validated Track B construction to "
              "transplant; absence of evidence is not evidence of readiness).", file=sys.stderr)
    else:
        print(f"Track B's verdict is {status.get('verdict')!r}, not the no-launch case this "
              f"session's build assumed -- this refusal message is STALE if that verdict is real; "
              f"a human must re-evaluate sec 5.5 item 3's gate before authoring a Wave 4 manifest "
              f"(this file intentionally contains none).", file=sys.stderr)
    print("This launcher CONTAINS NO Wave 4 manifest-building code -- there is nothing to preview "
          "or force past this refusal short of editing this file after a real re-authorization.",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.exit(3)


# ---------------------------------------------------------------------------
# Cell specs. ONE manifest builder covers rung-1 AND control cells (they
# differ only in d_model/d_state/n_layers/steps/name-prefix) -- is_done/
# build_cmd are shape-identical to run_lm_rd_sweep.py's own Wave C pattern,
# generalized over an explicit `cfg` dict so both cell types share one path
# (task brief item (5): "is_done identity on rung/config/mix").
# ---------------------------------------------------------------------------

def _ckpt_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "checkpoints")


def cell_name(tag: str, corpus: str, seed: int, cfg: dict) -> str:
    return f"{tag}_lm_{corpus}_dm{cfg['d_model']}_ds{cfg['d_state']}_L{cfg['n_layers']}_s{seed}"


def make_manifest(tag: str, cfg: dict, corpora, seeds, steps: int, ckpt_every: int,
                   batch_size: int = None) -> list[dict]:
    """`batch_size` defaults to the global BATCH_SIZE (rung-1/control's original, unchanged
    behavior) -- trackC-rung23-build's callers for rung 2/3 pass BATCH_SIZE_BY_RUNG[rung] explicitly
    (see module docstring's "Batch sizing" paragraph)."""
    if batch_size is None:
        batch_size = BATCH_SIZE
    runs = []
    for corpus in corpora:
        for seed in seeds:
            runs.append({
                "tag": tag, "corpus": corpus, "seed": seed, "d_model": cfg["d_model"],
                "d_state": cfg["d_state"], "n_layers": cfg["n_layers"], "seq_len": SEQ_LEN,
                "batch_size": batch_size, "steps": steps, "ckpt_every": ckpt_every,
                "name": cell_name(tag, corpus, seed, cfg),
            })
    return runs


def is_done_cell(out_dir, spec) -> bool:
    """Identity check keyed on rung/config (d_model/d_state/n_layers -- the
    rung's OWN identity) AND mix (corpus) AND seed AND steps -- task brief
    item (5)'s explicit requirement, cloned from run_lm_rd_sweep.py's
    is_done_C with the corpus set widened to the mix corpora."""
    p = os.path.join(out_dir, f"{spec['name']}.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        if not isinstance(d, dict) or d.get("complete") is not True:
            return False
        if d.get("timed_out"):
            return False
        # "batch_size" ADDED (trackC-rung23-build): cell_name()/is_done_cell's identity tuple never
        # included batch_size, which was harmless while every rung shared one global BATCH_SIZE --
        # now that BATCH_SIZE_BY_RUNG can differ per rung, a stale JSON from a PRIOR batch size would
        # otherwise silently pass this check (same name, same d_model/d_state/n_layers/steps) and be
        # trusted as "done" for a re-calibration at a NEW batch size. Zero behavior change for
        # rung 1/control (their JSONs already carry batch_size=32=BATCH_SIZE, always matching).
        required = ("corpus", "seed", "d_model", "d_state", "n_layers", "seq_len", "steps",
                    "steps_completed", "batch_size")
        if not all(k in d for k in required):
            return False
        if d.get("steps_completed", 0) < spec["steps"]:
            return False
        if (d.get("corpus") != spec["corpus"] or d.get("seed") != spec["seed"]
                or d.get("d_model") != spec["d_model"] or d.get("d_state") != spec["d_state"]
                or d.get("n_layers") != spec["n_layers"] or d.get("seq_len") != spec["seq_len"]
                or d.get("steps") != spec["steps"] or d.get("batch_size") != spec["batch_size"]):
            return False
        return True
    except Exception:
        return False


def build_cmd_cell(spec, out_dir, timeout, data_dir):
    return [sys.executable, PRETRAIN,
            "--corpus", spec["corpus"], "--data-dir", data_dir,
            "--d-model", str(spec["d_model"]), "--d-state", str(spec["d_state"]),
            "--n-layers", str(spec["n_layers"]), "--seq-len", str(spec["seq_len"]),
            "--batch-size", str(spec["batch_size"]), "--steps", str(spec["steps"]),
            "--ckpt-every", str(spec["ckpt_every"]),
            "--seed", str(spec["seed"]), "--internal-timeout", str(max(1, timeout - 30)),
            "--ckpt-dir", _ckpt_dir(out_dir),
            "--out", os.path.join(out_dir, f"{spec['name']}.json")]


# ---------------------------------------------------------------------------
# Calibration (Wave -1) manifest -- ONE rung-1 cell + ONE control cell,
# seed 0, openr1-mix only, SHORT step budget. May be launched by this build
# session (task brief: "MAY run if cheap (<2 GPU-h total)").
# ---------------------------------------------------------------------------

def calibration_manifest(steps: int = None, ckpt_every: int = None,
                          tags: tuple = CALIBRATION_TAGS_DEFAULT) -> list[dict]:
    """Two-point manifest (trackC-audit finding #2): for each config named in `tags` (default:
    rung1 + control ONLY, byte-identical to the pre-extension behavior -- trackC-rung23-build),
    ONE run at the SHORT step count and ONE at the LONG step count from
    CALIBRATION_TWO_POINT_STEPS, each with ckpt_every == its own steps (exactly one checkpoint, at
    the very end) -- so each point's wall_s is a clean single-checkpoint sample, not a
    multi-checkpoint blend. Each cell runs at CALIBRATION_BATCH_SIZE[tag] (matches the real batch
    its wave would use -- the memory-headroom check's whole point). `steps`/`ckpt_every`
    parameters are accepted for CLI back-compat (--calibration-steps/--calibration-ckpt-every) but
    IGNORED here -- the two-point cells are fixed by CALIBRATION_TWO_POINT_STEPS, not by those
    flags (a mismatch would defeat the clean-solve property); main() prints a warning if a
    non-default value is passed."""
    runs = []
    for tag_base in tags:
        cfg = CALIBRATION_RUNG_CFGS[tag_base]
        bs = CALIBRATION_BATCH_SIZE[tag_base]
        short_steps, long_steps = CALIBRATION_TWO_POINT_STEPS[tag_base]
        runs += make_manifest(f"{tag_base}_ptA", cfg, ("openr1-mix",), (0,), short_steps, short_steps,
                               batch_size=bs)
        runs += make_manifest(f"{tag_base}_ptB", cfg, ("openr1-mix",), (0,), long_steps, long_steps,
                               batch_size=bs)
    return runs


def derive_timing_constants(out_dir: str) -> dict:
    """Two-point solve (trackC-audit finding #2): reads the ptA/ptB result JSONs for EVERY
    registered config in CALIBRATION_RUNG_CFGS (rung1/control/rung2/rung3 -- not just whichever
    `tags` a given calibration invocation actually launched) and solves the 2x2 linear system
    (module docstring above) for clean per_step_s/per_ckpt_s constants, per config. A config
    missing either point (or either point incomplete) is simply omitted from the result -- the
    caller (a wave's real-launch gate) must check for its own key before trusting the timeout it
    derives. Safe to widen unconditionally to all four configs: a config whose cells were never
    launched just has no files on disk yet and is skipped, exactly as before this build for
    rung2/rung3 (this is a strict superset of the pre-extension rung1/control-only loop)."""
    constants = {}
    for tag_base, cfg in CALIBRATION_RUNG_CFGS.items():
        short_steps, long_steps = CALIBRATION_TWO_POINT_STEPS[tag_base]
        pA = os.path.join(out_dir, f"{cell_name(f'{tag_base}_ptA', 'openr1-mix', 0, cfg)}.json")
        pB = os.path.join(out_dir, f"{cell_name(f'{tag_base}_ptB', 'openr1-mix', 0, cfg)}.json")
        if not (os.path.exists(pA) and os.path.exists(pB)):
            continue
        with open(pA) as f:
            dA = json.load(f)
        with open(pB) as f:
            dB = json.load(f)
        if dA.get("complete") is not True or dB.get("complete") is not True:
            continue
        steps_a, wall_a = dA["steps_completed"], dA["wall_s"]
        steps_b, wall_b = dB["steps_completed"], dB["wall_s"]
        if steps_b == steps_a:
            continue
        per_step_s = (wall_b - wall_a) / (steps_b - steps_a)
        per_ckpt_s = wall_a - per_step_s * steps_a
        # VALIDITY GUARD (trackC-rung23-build, caught LIVE on rung 2's first calibration run):
        # a non-positive per_step_s (or per_ckpt_s) is PHYSICALLY IMPOSSIBLE and means the
        # two-point method's equal-fixed-overhead assumption broke -- observed cause: the SHORT
        # point was the first-ever run at a new kernel shape (d_state=128) and paid one-time
        # Triton compile (disk-cache miss, ~26s) that the LONG point (cache hit) didn't, making
        # wall_A > wall_B (measured: ptA 10 steps 59.6s vs ptB 40 steps 44.0s -> solved
        # per_step_s=-0.52). Recording that would silently poison every timeout/budget projection
        # derived from it. Remedy: delete BOTH points' JSONs and re-run this config's calibration
        # now that the kernel cache is warm (both fresh points then share the same fixed overhead).
        if per_step_s <= 0 or per_ckpt_s <= 0:
            print(f"WARNING: derive_timing_constants[{tag_base}]: solved per_step_s={per_step_s:.4f}"
                  f"/per_ckpt_s={per_ckpt_s:.4f} is non-positive -- REJECTED, key omitted "
                  f"(cold-kernel-cache asymmetry between the two points; see the comment at this "
                  f"check). Delete both point JSONs and re-run --wave calibration for this config "
                  f"with a warm cache.", file=sys.stderr)
            continue
        constants[tag_base.replace("calib_", "")] = {
            "per_step_s": per_step_s, "per_ckpt_s": per_ckpt_s,
            "point_a": {"steps": steps_a, "wall_s": wall_a},
            "point_b": {"steps": steps_b, "wall_s": wall_b},
            "method": ("two-point (trackC-audit finding #2): wall_s = per_step_s*steps + "
                       "per_ckpt_s*1 (one checkpoint per point), solved from 2 single-checkpoint runs"),
        }
    return constants


def wave1_manifest(rung1_steps: int, control_steps: int) -> list[dict]:
    """Shared by --dry-run's preview and the real --wave 1 launch (trackC-audit: previously
    duplicated inline in main()'s dry-run block only, with no real-launch counterpart at all --
    factored out so preview and real launch can never silently drift apart)."""
    return (make_manifest("w1_rung1", RUNG_CFG, MIX_CORPORA, SEEDS, rung1_steps, 1000)
            + make_manifest("w1_control", CONTROL_CFG, MIX_CORPORA, SEEDS, control_steps, 1000))


def summarize_calibration(out_dir: str, manifest: list[dict]) -> dict:
    """Extracts measured tok/s (from wall_s / steps_completed*batch*seq) and peak memory from each
    calibration cell's own result JSON -- the task brief's explicit deliverable ("report their
    measured numbers"). Written to calibration.json, which a wave's real-launch gate (sec 9) checks
    for existence before pricing/launching the full manifest.

    IMPORTANT (trackC-rung23-build): callers should pass the FULL cross-rung manifest here
    (`calibration_manifest(tags=tuple(CALIBRATION_RUNG_CFGS))`), NOT just whatever subset this
    particular `--wave calibration --calib-rungs ...` invocation actually launched -- summarizing
    only the just-launched subset would silently DROP previously-recorded cells (e.g. a
    `--calib-rungs 2`-only run would otherwise overwrite calibration.json's rung1/control entries
    with nothing), clobbering a resume-safe, monotonically-growing record. `_run_wave` itself still
    launches/tracks only the requested subset -- this function's `manifest` argument is purely
    about what to SUMMARIZE from disk afterward."""
    cells = {}
    for spec in manifest:
        p = os.path.join(out_dir, f"{spec['name']}.json")
        if not os.path.exists(p):
            cells[spec["name"]] = {"status": "missing"}
            continue
        with open(p) as f:
            d = json.load(f)
        if d.get("complete") is not True:
            cells[spec["name"]] = {"status": "incomplete", "steps_completed": d.get("steps_completed")}
            continue
        tokens = d["steps_completed"] * spec["batch_size"] * spec["seq_len"]
        toks_per_s = tokens / d["wall_s"] if d.get("wall_s") else None
        cells[spec["name"]] = {
            "status": "complete", "tag": spec["tag"], "d_model": spec["d_model"],
            "d_state": spec["d_state"], "n_layers": spec["n_layers"], "n_params": d.get("n_params"),
            "steps_completed": d["steps_completed"], "wall_s": d["wall_s"],
            "tokens_per_s_per_gpu": toks_per_s,
            "peak_memory_allocated_gb": (d.get("peak_memory_allocated_bytes") or 0) / 1e9,
            "peak_memory_reserved_gb": (d.get("peak_memory_reserved_bytes") or 0) / 1e9,
        }
    return {"design_ref": "SCALE_TRANSFER_DESIGN.md sec 5.6 Wave -1 (Track C calibration)",
            "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": cells,
            # trackC-audit finding #2: clean per_step_s/per_ckpt_s per config, NOT the blended
            # single-point wall_s above -- this is what Wave 1's real-launch timeout (finding #1)
            # is derived from, with LAUNCH_TIMEOUT_MARGIN applied on top.
            "timing_constants": derive_timing_constants(out_dir)}


# ---------------------------------------------------------------------------
# Wave 2 / Wave 3 (trackC-rung23-build): rung-2/3 FULL manifests, plus the three extra gates
# sec 5's own text requires before any real launch at these rungs (memory headroom, epoch cap,
# program budget) -- none of which existed for rung 1 (rung 1's own calibration didn't need a
# memory-headroom gate because BATCH_SIZE_BY_RUNG was uniform and rung 1's own headroom was never
# in question at 98M params/80GB; rung 1's epoch-cap arithmetic also happened to clear its mix's
# ceiling, see this build's own audit note). Both are now load-bearing at rung 2/3's scale.
# ---------------------------------------------------------------------------

WAVE_RUNG_CFGS = {2: RUNG2_CFG, 3: RUNG3_CFG}
WAVE_TIMING_KEY = {2: "rung2", 3: "rung3"}          # derive_timing_constants()' own key naming
WAVE_CALIB_TAG = {2: "calib_rung2", 3: "calib_rung3"}

# 80GB H100 nominal capacity, in bytes -- deliberately the SMALLER of the two plausible denominators
# (nvidia-smi on this box reports 81,559 MiB =~ 85.5GB; using the vendor-nominal 80,000,000,000
# bytes is the conservative choice, per this codebase's own "leave more headroom than the raw
# number suggests" convention, CLAUDE.md's batch=96-not-112 lesson).
H100_VRAM_BYTES = 80_000_000_000

# House convention (CLAUDE.md: "batch=96 fits training but leaves room for eval; batch=112 fits
# training but OOMs during eval"): reserve >=15% of the card for eval batches (smaller than train
# batch but run on the SAME device, sequentially, right after training in this harness) and
# allocator fragmentation -- a calibration cell that trains but has NO headroom for its own eval
# step is not actually a passing memory check.
MEMORY_HEADROOM_SAFE_FRACTION = 0.85


def memory_headroom_report(peak_allocated_bytes: float, peak_reserved_bytes: float, label: str) -> dict:
    frac_reserved = peak_reserved_bytes / H100_VRAM_BYTES
    return {
        "label": label,
        "peak_allocated_gb": peak_allocated_bytes / 1e9,
        "peak_reserved_gb": peak_reserved_bytes / 1e9,
        "frac_of_80gb_reserved": frac_reserved,
        "safe_fraction_threshold": MEMORY_HEADROOM_SAFE_FRACTION,
        "within_safe_headroom": frac_reserved <= MEMORY_HEADROOM_SAFE_FRACTION,
    }


def epoch_cap_check(data_dir: str, corpus: str, planned_tokens: int, epoch_cap: int = EPOCH_CAP) -> dict:
    """sec 5.4's discipline, made concrete by build_mix_corpora_rd.py's own meta.json
    ("mix_total_train_tokens" / equivalently "train_tokens" at the mix's top level): a training
    budget B sampled uniformly over a mix of size M repeats the WHOLE mix (base corpus included)
    B/M times in expectation -- so the <=5-physical-epoch cap on the base corpus is exactly
    B <= 5*M. Reads M live from the corpus's own meta.json (not cached/assumed) so a rebuilt,
    bigger mix is picked up automatically without editing this launcher."""
    meta_path = os.path.join(data_dir, MIX_CORPUS_DIRS[corpus], "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    m_tokens = meta["train_tokens"]
    ceiling = epoch_cap * m_tokens
    return {
        "corpus": corpus, "meta_path": meta_path, "mix_train_tokens": m_tokens,
        "epoch_cap": epoch_cap, "epoch_cap_ceiling_tokens": ceiling,
        "planned_tokens": planned_tokens, "ok": planned_tokens <= ceiling,
    }


def projected_gpu_hours(manifest: list[dict], timing: dict) -> float:
    """Sums (per_step_s*steps + per_ckpt_s*n_ckpts) over every cell in `manifest` using ONE
    config's measured timing_constants (all cells in a rung-2/3 manifest share the same
    d_model/d_state/n_layers/batch, so one timing dict covers the whole wave) -- the number the
    budget guard prints and gates on BEFORE any launch decision, not a pre-calibration guess."""
    total_s = 0.0
    for spec in manifest:
        n_ckpts = spec["steps"] // spec["ckpt_every"] + 1
        total_s += timing["per_step_s"] * spec["steps"] + timing["per_ckpt_s"] * n_ckpts
    return total_s / 3600.0


def budget_guard(projected_gpu_h: float, label: str, accept_override: bool) -> float:
    """sec 7's 300 GPU-h program ceiling, enforced live: PRINTS the projection (task brief's
    explicit deliverable) before any launch, and REFUSES (exits non-zero) if PROGRAM_SPENT_GPUH +
    projected would exceed the ceiling, unless --accept-budget-override is passed. The override is
    a human decision (explicit CLI flag), never a default."""
    cumulative = PROGRAM_SPENT_GPUH + projected_gpu_h
    print(f"BUDGET GUARD ({label}): program-spent-so-far={PROGRAM_SPENT_GPUH:.1f} GPU-h "
          f"(maintained constant, sec 7) + this-wave-projected={projected_gpu_h:.2f} GPU-h "
          f"= cumulative {cumulative:.2f} GPU-h, ceiling {GPU_H_PROGRAM_CEILING:.0f} GPU-h.",
          flush=True)
    if cumulative > GPU_H_PROGRAM_CEILING and not accept_override:
        print(f"ERROR: projected cumulative spend {cumulative:.2f} GPU-h EXCEEDS the "
              f"{GPU_H_PROGRAM_CEILING:.0f} GPU-h program ceiling (sec 7) -- REFUSING to launch "
              f"{label}. Pass --accept-budget-override to force past this guard (a human decision, "
              f"never a default), or cut scope first (sec 8's cut order: Track D Phase 2, rung-3's "
              f"fix-effect sub-wave, rung-2's frontier-probe reminder, Track B's naive-window arm, "
              f"rung-3's second corpus, rung 2 wholesale -- in that order).", file=sys.stderr)
        sys.exit(5)
    return cumulative


def wave23_manifest(rung: int, steps: int) -> list[dict]:
    """sec 5.6 AMENDMENT (Rev 2.1) item 1: rung 2 is now "2 corpora x 3 seeds" (6 runs), rung 3
    stays "2 corpora x 1 seed" as registered -- WAVE23_SEEDS_BY_RUNG[rung] selects the per-rung seed
    set, so the manifest length (and every downstream run-count / GPU-h projection) derives from
    this dict, never a hardcoded literal. No control cell at these rungs (MAJOR-5's control is
    rung-1-specific; sec 8's cut order never asks for one at 2/3 either). Trains on WAVE23_CORPORA
    (amendment item 2's EXTENDED mixes), NOT MIX_CORPORA -- wave 1 / calibration / control are
    unaffected. Shared by --dry-run's preview and the real --wave {2,3} launch (wave1_manifest's own
    preview/launch-parity discipline, restated here)."""
    cfg = WAVE_RUNG_CFGS[rung]
    bs = BATCH_SIZE_BY_RUNG[rung]
    return make_manifest(f"w{rung}_rung{rung}", cfg, WAVE23_CORPORA, WAVE23_SEEDS_BY_RUNG[rung], steps, 1000,
                          batch_size=bs)


def gate_and_run_wave23(rung: int, args) -> None:
    """The full (a)-(e) gate chain for a real --wave {2,3} launch (module docstring). Every check
    below EXITS non-zero on failure with the design's own registered remedy -- there is no silent
    proceed path. Mirrors --wave 1's calibration.json/--rungN-steps gate exactly for (a)/(b), then
    ADDS (c) memory headroom, (d) epoch cap, (e) budget guard -- all three genuinely new
    requirements at rung 2/3's scale (module docstring's top-of-file rationale)."""
    calibration_json_path = os.path.join(args.out_dir, "calibration.json")
    timing_key = WAVE_TIMING_KEY[rung]
    steps = getattr(args, f"rung{rung}_steps")

    # (a) calibration.json must exist and carry this rung's timing_constants.
    if not os.path.exists(calibration_json_path):
        print(f"ERROR: {calibration_json_path} not found -- sec 9's own hard rule: 'No track's Wave "
              f"1+ manifest is authorized to launch ... without first recording its Wave -1 measured "
              f"numbers.' Run --wave calibration --calib-rungs {rung} first.", file=sys.stderr)
        sys.exit(2)
    with open(calibration_json_path) as f:
        calib = json.load(f)
    timing = calib.get("timing_constants") or {}
    if timing_key not in timing:
        print(f"ERROR: {calibration_json_path} has no timing_constants[{timing_key!r}] -- rerun "
              f"--wave calibration --calib-rungs {rung} (two-point method) so per_step_s/per_ckpt_s "
              f"are populated before a real --wave {rung} launch.", file=sys.stderr)
        sys.exit(2)

    # (b) --rungN-steps must be supplied explicitly (no silent fallback to the placeholder).
    if steps is None:
        print(f"ERROR: --rung{rung}-steps is REQUIRED for a real --wave {rung} launch (no silent "
              f"fallback to RUNG{rung}_TARGET_TOKENS_PLACEHOLDER) -- derive it from "
              f"{calibration_json_path}'s measured tok/s and pass it explicitly.", file=sys.stderr)
        sys.exit(2)

    # (c) memory headroom, recomputed LIVE from calibration.json's own recorded peak-memory cells
    # (never assumed from a prior session's cited number) -- both the short and long calibration
    # points are checked (peak memory is a function of batch/shape, not step count, but checking
    # both catches a fluke single-point misread).
    cfg = WAVE_RUNG_CFGS[rung]
    tag_base = WAVE_CALIB_TAG[rung]
    mem_reports = []
    for pt in ("ptA", "ptB"):
        cell_key = cell_name(f"{tag_base}_{pt}", "openr1-mix", 0, cfg)
        cell = (calib.get("cells") or {}).get(cell_key)
        if cell and cell.get("status") == "complete":
            mem_reports.append(memory_headroom_report(
                cell["peak_memory_allocated_gb"] * 1e9, cell["peak_memory_reserved_gb"] * 1e9, cell_key))
    if not mem_reports:
        print(f"ERROR: no COMPLETE calibration memory readouts found for rung {rung} in "
              f"{calibration_json_path} -- the memory-headroom check is a blocking Wave -1 item "
              f"(module docstring's 'Batch sizing' paragraph), not optional. Run --wave calibration "
              f"--calib-rungs {rung} first.", file=sys.stderr)
        sys.exit(2)
    bad_mem = [r for r in mem_reports if not r["within_safe_headroom"]]
    if bad_mem:
        print(f"ERROR: memory-headroom check FAILED for rung {rung}: {bad_mem} -- "
              f"BATCH_SIZE_BY_RUNG[{rung}] (currently {BATCH_SIZE_BY_RUNG[rung]}) must be lowered "
              f"and this rung re-calibrated before a real launch (module docstring's 'Batch sizing' "
              f"paragraph's own remedy).", file=sys.stderr)
        sys.exit(2)
    print(f"Memory headroom OK for rung {rung} (batch={BATCH_SIZE_BY_RUNG[rung]}): {mem_reports}",
          flush=True)

    manifest = wave23_manifest(rung, steps)

    # (d) epoch cap, BOTH EXTENDED mix corpora (amendment item 2 -- waves >=2 train on WAVE23_CORPORA,
    # not MIX_CORPORA), at this rung's resulting per-run token budget.
    planned_tokens = steps * BATCH_SIZE_BY_RUNG[rung] * SEQ_LEN
    epoch_reports = [epoch_cap_check(args.data_dir, c, planned_tokens) for c in WAVE23_CORPORA]
    failing = [r for r in epoch_reports if not r["ok"]]
    if failing:
        print("=" * 70, file=sys.stderr)
        print(f"EPOCH-CAP CHECK FAILED for rung {rung} (sec 5.4: '<=5-physical-epoch' discipline "
              f"on each mix's base corpus):", file=sys.stderr)
        for r in failing:
            print(f"  corpus={r['corpus']!r}: planned {r['planned_tokens']:,} tokens > ceiling "
                  f"{r['epoch_cap_ceiling_tokens']:,} (= {r['epoch_cap']} x mix train_tokens "
                  f"{r['mix_train_tokens']:,}, read live from {r['meta_path']})", file=sys.stderr)
        print("REMEDY (sec 5.4's own registered rule, restated in build_mix_corpora_rd.py's meta.json "
              "'epoch_cap_discipline' field): pull more augmentation BEFORE launching -- re-run "
              "build_mix_corpora_rd.py with a larger --target-augment-tokens for the failing "
              "corpus/corpora so mix_train_tokens * 5 >= this rung's planned per-run token budget, "
              "then re-run --wave calibration --calib-rungs "
              f"{rung} for this rung (a bigger mix may shift timing slightly) before retrying this "
              "launch.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(6)
    print(f"Epoch-cap check OK for rung {rung}: {epoch_reports}", flush=True)

    # (e) program-wide GPU-h budget guard -- printed and gated BEFORE launch, per the task brief.
    projected = projected_gpu_hours(manifest, timing[timing_key])
    budget_guard(projected, f"wave {rung}", args.accept_budget_override)

    out_dir = os.path.join(args.out_dir, f"wave{rung}")
    os.makedirs(out_dir, exist_ok=True)

    def timeout_fn(spec):
        c = timing[timing_key]
        return default_timeout_pretrain(spec["steps"], spec["ckpt_every"],
                                         c["per_step_s"], c["per_ckpt_s"], margin=LAUNCH_TIMEOUT_MARGIN)

    # run count derived from the ACTUAL manifest length (len(manifest)) -- the "x corpora x seeds"
    # breakdown below is descriptive only, itself read from WAVE23_CORPORA/WAVE23_SEEDS_BY_RUNG
    # (amendment item 1's per-rung seed count), never a hardcoded literal.
    print(f"WAVE {rung} REAL LAUNCH: {len(manifest)} runs ({steps} steps x {len(WAVE23_CORPORA)} "
          f"corpora x {len(WAVE23_SEEDS_BY_RUNG[rung])} seed(s)). Timing constants from "
          f"{calibration_json_path} "
          f"(margin {LAUNCH_TIMEOUT_MARGIN}x): {json.dumps(timing[timing_key], indent=2)}", flush=True)
    all_done = _run_wave(str(rung), manifest, out_dir, args, is_done_cell, build_cmd_cell, timeout_fn)
    sys.exit(0 if all_done else 1)


# ---------------------------------------------------------------------------
# mixcontrol (sec 5.6 AMENDMENT, Rev 2.1, item 3, NEW): re-isolates the mix axis at Wave-C scale for
# the EXTENDED mixes -- CONTROL_CFG retrained on WAVE23_CORPORA x 3 seeds, mirroring MAJOR-5's
# original same-mix control logic (Wave 1's own w1_control cell, which used the ORIGINAL MIX_CORPORA
# and stays unchanged). Reuses Wave 1's own control-length step budget (default_control_steps() /
# --control-steps) -- "control-length steps" per the amendment, never a separate step count.
# ---------------------------------------------------------------------------

def mixcontrol_manifest(steps: int) -> list[dict]:
    """sec 5.6 amendment (Rev 2.1) item 3: CONTROL_CFG x WAVE23_CORPORA x SEEDS (0,1,2) = 6 cells,
    at the SAME control-length step budget as Wave 1's own same-mix control cell (this file's
    default_control_steps() / --control-steps, CONTROL_TARGET_TOKENS convention -- unchanged).
    Shared by --dry-run's preview and the real --wave mixcontrol launch (wave1_manifest's own
    preview/launch-parity discipline, restated here)."""
    return make_manifest("mixcontrol", CONTROL_CFG, WAVE23_CORPORA, SEEDS, steps, 1000)


def gate_and_run_mixcontrol(args) -> None:
    """The gate chain for a real --wave mixcontrol launch, mirroring gate_and_run_wave23's (a)/(c)/
    (d)/(e) chain (module docstring's "LEARNED FROM WAVE 1'S OWN HISTORY" paragraph applies here
    too -- a gate with nothing behind it is a no-op). (b)'s --rungN-steps analog does not apply:
    this wave reuses Wave 1's own --control-steps flag/default (default_control_steps()), matching
    the amendment's "control-length steps" wording exactly -- no new step-count flag is introduced.
    Keyed to the 'control' calibration/timing cell (already banked in Wave 1's calibration.json --
    no new calibration is required before this wave can gate open)."""
    calibration_json_path = os.path.join(args.out_dir, "calibration.json")

    # (a) calibration.json must exist and carry the control config's timing constants.
    if not os.path.exists(calibration_json_path):
        print(f"ERROR: {calibration_json_path} not found -- sec 9's own hard rule: 'No track's Wave "
              f"1+ manifest is authorized to launch ... without first recording its Wave -1 measured "
              f"numbers.' Run --wave calibration first.", file=sys.stderr)
        sys.exit(2)
    with open(calibration_json_path) as f:
        calib = json.load(f)
    timing = calib.get("timing_constants") or {}
    if "control" not in timing:
        print(f"ERROR: {calibration_json_path} has no timing_constants['control'] -- rerun "
              f"--wave calibration (two-point method) so the control config's per_step_s/per_ckpt_s "
              f"are populated before a real --wave mixcontrol launch.", file=sys.stderr)
        sys.exit(2)

    steps = args.control_steps or default_control_steps()

    # (c) memory headroom, recomputed LIVE from calibration.json's own recorded peak-memory cells --
    # reuses the SAME calib_control_ptA/ptB cells Wave 1's control gate used (memory is a function of
    # batch/shape/architecture, not corpus content, gate_and_run_wave23's own assumption for rung 2/3
    # restated here for the control architecture).
    mem_reports = []
    for pt in ("ptA", "ptB"):
        cell_key = cell_name(f"calib_control_{pt}", "openr1-mix", 0, CONTROL_CFG)
        cell = (calib.get("cells") or {}).get(cell_key)
        if cell and cell.get("status") == "complete":
            mem_reports.append(memory_headroom_report(
                cell["peak_memory_allocated_gb"] * 1e9, cell["peak_memory_reserved_gb"] * 1e9, cell_key))
    if not mem_reports:
        print(f"ERROR: no COMPLETE calibration memory readouts found for the control config in "
              f"{calibration_json_path} -- the memory-headroom check is a blocking Wave -1 item "
              f"(module docstring's 'Batch sizing' paragraph), not optional. Run --wave calibration "
              f"first.", file=sys.stderr)
        sys.exit(2)
    bad_mem = [r for r in mem_reports if not r["within_safe_headroom"]]
    if bad_mem:
        print(f"ERROR: memory-headroom check FAILED for mixcontrol: {bad_mem} -- BATCH_SIZE "
              f"(currently {BATCH_SIZE}) must be lowered and the control config re-calibrated before "
              f"a real launch.", file=sys.stderr)
        sys.exit(2)
    print(f"Memory headroom OK for mixcontrol (batch={BATCH_SIZE}): {mem_reports}", flush=True)

    manifest = mixcontrol_manifest(steps)

    # (d) epoch cap, BOTH extended mix corpora, at control's per-run token budget -- trivially
    # passes at control-length steps (the whole point of reusing Wave 1's control budget: it's ~100M
    # tokens against a >=1.7B-token extended-mix ceiling), but the gate STILL RUNS (amendment item 3:
    # "still run the gate, don't skip it").
    planned_tokens = steps * BATCH_SIZE * SEQ_LEN
    epoch_reports = [epoch_cap_check(args.data_dir, c, planned_tokens) for c in WAVE23_CORPORA]
    failing = [r for r in epoch_reports if not r["ok"]]
    if failing:
        print("=" * 70, file=sys.stderr)
        print("EPOCH-CAP CHECK FAILED for mixcontrol (sec 5.4: '<=5-physical-epoch' discipline on "
              "each mix's base corpus):", file=sys.stderr)
        for r in failing:
            print(f"  corpus={r['corpus']!r}: planned {r['planned_tokens']:,} tokens > ceiling "
                  f"{r['epoch_cap_ceiling_tokens']:,} (= {r['epoch_cap']} x mix train_tokens "
                  f"{r['mix_train_tokens']:,}, read live from {r['meta_path']})", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(6)
    print(f"Epoch-cap check OK for mixcontrol: {epoch_reports}", flush=True)

    # (e) program-wide GPU-h budget guard -- printed and gated BEFORE launch, per the task brief.
    projected = projected_gpu_hours(manifest, timing["control"])
    budget_guard(projected, "wave mixcontrol", args.accept_budget_override)

    out_dir = os.path.join(args.out_dir, "mixcontrol")
    os.makedirs(out_dir, exist_ok=True)

    def timeout_fn(spec):
        c = timing["control"]
        return default_timeout_pretrain(spec["steps"], spec["ckpt_every"],
                                         c["per_step_s"], c["per_ckpt_s"], margin=LAUNCH_TIMEOUT_MARGIN)

    print(f"WAVE mixcontrol REAL LAUNCH: {len(manifest)} runs ({steps} steps x {len(WAVE23_CORPORA)} "
          f"corpora x {len(SEEDS)} seeds). Timing constants from {calibration_json_path} "
          f"(margin {LAUNCH_TIMEOUT_MARGIN}x): {json.dumps(timing['control'], indent=2)}", flush=True)
    all_done = _run_wave("mixcontrol", manifest, out_dir, args, is_done_cell, build_cmd_cell, timeout_fn)
    sys.exit(0 if all_done else 1)


# ---------------------------------------------------------------------------
# Shared orchestration -- CLONE of run_lm_rd_geo3_sweep.py's pattern
# (smoke gate, exception-isolated launch, validity-checked resume, per-run
# timeout + GPU quarantine, guarded aggregate, ALL_DONE sentinel).
# ---------------------------------------------------------------------------

def run_smoke(log_dir, gpu):
    print(f"SMOKE GATE (physical GPU {gpu}) -- lm_pretrain_rd.py, lm_rd_rung_configs.py, "
          f"lm_attractor_probe_rd.py, build_mix_corpora_rd.py ...", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    ok = True
    for name, script in (("pretrain", PRETRAIN), ("rung_configs", RUNG_CONFIGS),
                          ("attractor_probe", ATTRACTOR_PROBE), ("build_mix", BUILD_MIX)):
        with open(os.path.join(log_dir, f"smoke_{name}.log"), "w") as lf:
            rc = subprocess.call([sys.executable, script, "--smoke"], env=env,
                                  stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
        print(f"  smoke[{name}]: {'PASS' if rc == 0 else f'FAIL (rc={rc})'}", flush=True)
        ok = ok and (rc == 0)
    return ok


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
            ck = d.get("checkpoints") or []
            final = ck[-1] if ck else {}
            cells[spec["name"]] = {
                "tag": spec.get("tag"), "corpus": d.get("corpus"), "seed": d.get("seed"),
                "d_model": spec.get("d_model"), "d_state": spec.get("d_state"), "n_layers": spec.get("n_layers"),
                "n_params": d.get("n_params"), "final_step": d.get("final_step"),
                "final_val_loss": final.get("val_loss"), "wall_s": d.get("wall_s"),
                "peak_memory_allocated_bytes": d.get("peak_memory_allocated_bytes"),
            }
        report["cells"] = cells
    except Exception as e:
        report["aggregate_error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "AGGREGATE.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
            f.write(f"DeltaNet-RD Track C (scaling ladder) -- wave {wave}\n" + "=" * 50 + "\n")
            f.write(json.dumps(report, indent=2) + "\n")
    except Exception:
        pass


def _run_wave(wave, manifest, out_dir, args, is_done_fn, build_cmd_fn, timeout_fn):
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_smoke and not run_smoke(log_dir, args.gpu_offset):
        with open(os.path.join(out_dir, "ABORTED.txt"), "w") as f:
            f.write("smoke gate failed; no training launched.\n")
        sys.exit(1)

    physical_gpus = list(range(args.gpu_offset, args.gpu_offset + args.gpus))
    slots = [g for _ in range(args.per_gpu) for g in physical_gpus]
    n_slots = len(slots)
    pending = [s for s in manifest if not is_done_fn(out_dir, s)]
    # trackC-rung23-build BUGFIX: cells that were ALREADY done before this invocation started (and
    # so never entered `pending`) must still count toward `all_done` below -- pre-fix, re-invoking
    # a wave whose entire manifest was already complete (pending=0, loop body never runs, done_ct
    # stays 0) reported all_done=False, which this build's calibration-summary logic depends on
    # (a false all_done=False would skip refreshing the root-level calibration.json even though
    # every cell is genuinely complete). Caught live: re-running `--wave calibration` with the
    # DEFAULT --calib-rungs (rung1+control, both already complete from Wave 1) reproduced exactly
    # this false-negative on-box before this fix.
    already_done_ct = len(manifest) - len(pending)
    print(f"wave={wave}  manifest={len(manifest)}  pending={len(pending)} "
          f"(already-done={already_done_ct})  "
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
                    proc = subprocess.Popen(build_cmd_fn(spec, out_dir, timeout, args.data_dir), env=env,
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
                if rc == 0 and is_done_fn(out_dir, spec):
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

    all_done = (done_ct + already_done_ct == len(manifest)) and not failed
    if all_done:
        with open(os.path.join(out_dir, "ALL_DONE"), "w") as f:
            f.write(f"wave {wave} complete: {done_ct}/{len(manifest)} runs, 0 failed\n")
    print(f"\nWAVE {wave} DONE. {done_ct} succeeded, {len(failed)} failed. "
          f"ALL_DONE {'written' if all_done else 'NOT written (wave incomplete)'}.", flush=True)
    return all_done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/lm_rd_trackc"))
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--wave", choices=["calibration", "1", "2", "3", "4", "mixcontrol"], default=None,
                     help="REQUIRED unless --dry-run. 'calibration' (Wave -1) MAY be launched any "
                          "time (cheap, <2 GPU-h per rung; --calib-rungs selects which); '1' is "
                          "CLOSED (sec 5.9 -- already ran to completion; behavior unchanged by this "
                          "build); '2'/'3' are gated on an existing calibration.json + "
                          "--rung{2,3}-steps + a passing memory-headroom/epoch-cap/budget-guard "
                          "chain (trackC-rung23-build; rung 2 is now 3 seeds/corpus and rungs 2/3 "
                          "train on the EXTENDED mixes, sec 5.6 amendment Rev 2.1; see module "
                          "docstring); 'mixcontrol' (Rev 2.1 amendment item 3, NEW) retrains "
                          "CONTROL_CFG on the extended mixes x 3 seeds, gated on the same (a)/(c)/"
                          "(d)/(e) chain keyed to the already-banked 'control' calibration cell; "
                          "'4' is ALWAYS refused (Track B's own NO-LAUNCH gates it out, sec 5.5 "
                          "item 3).")
    ap.add_argument("--gpus", type=int, default=None, help="GPU COUNT. REQUIRED for a real launch, "
                                                              "NO DEFAULT -- check nvidia-smi first.")
    ap.add_argument("--gpu-offset", type=int, default=None, help="first physical GPU index. REQUIRED "
                                                                    "for a real launch, NO DEFAULT.")
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--calibration-steps", type=int, default=CALIBRATION_STEPS_DEFAULT,
                     help="LEGACY, ignored by the manifest itself (trackC-audit finding #2: the "
                          "calibration manifest is now the fixed two-point CALIBRATION_TWO_POINT_STEPS "
                          "grid, not this flag) -- kept only so old invocations don't hard-fail on an "
                          "unrecognized argument; a non-default value prints a warning.")
    ap.add_argument("--calibration-ckpt-every", type=int, default=CALIBRATION_CKPT_EVERY,
                     help="LEGACY, ignored -- see --calibration-steps.")
    ap.add_argument("--calib-rungs", default="1,control",
                     help="comma list from {1,2,3,control}, selects which configs' two-point "
                          "calibration cells --wave calibration includes. DEFAULT UNCHANGED from "
                          "the pre-extension file (rung 1 + control only) -- rung 2/3 calibration "
                          "is opt-in ONLY (e.g. --calib-rungs 2), never launched by a bare "
                          "'--wave calibration' invocation (trackC-rung23-build).")
    ap.add_argument("--rung1-steps", type=int, default=None,
                     help="Wave 1 only: CLOSED (sec 5.9); retained for reference/reruns only.")
    ap.add_argument("--control-steps", type=int, default=None,
                     help="Wave 1 AND mixcontrol: default derived from CONTROL_TARGET_TOKENS (matches "
                          "Wave C exactly) -- mixcontrol reuses this same control-length budget "
                          "(sec 5.6 amendment Rev 2.1 item 3), never a separate step count.")
    ap.add_argument("--rung2-steps", type=int, default=None,
                     help="Wave 2 only: REQUIRED for a real launch (no silent fallback to the "
                          "uncalibrated placeholder) -- derive from calibration.json's measured tok/s.")
    ap.add_argument("--rung3-steps", type=int, default=None,
                     help="Wave 3 only: REQUIRED for a real launch (no silent fallback to the "
                          "uncalibrated placeholder) -- derive from calibration.json's measured tok/s.")
    ap.add_argument("--accept-budget-override", action="store_true",
                     help="Force past the sec-7 300 GPU-h program budget guard for --wave 2/3 (a "
                          "human decision -- there is no default override).")
    ap.add_argument("--trackb-gate-json", default=TRACKB_GATE_JSON_DEFAULT)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()

    if args.calibration_steps != CALIBRATION_STEPS_DEFAULT or args.calibration_ckpt_every != CALIBRATION_CKPT_EVERY:
        print("WARNING: --calibration-steps/--calibration-ckpt-every are legacy and IGNORED -- the "
              "calibration manifest is the fixed two-point CALIBRATION_TWO_POINT_STEPS grid "
              f"({CALIBRATION_TWO_POINT_STEPS}), trackC-audit finding #2.", file=sys.stderr)

    if args.dry_run:
        calib_tags = parse_calib_rungs(args.calib_rungs)
        calib_m = calibration_manifest(tags=calib_tags)
        rung1_steps = args.rung1_steps or default_rung1_steps_placeholder()
        control_steps = args.control_steps or default_control_steps()
        w1_m = wave1_manifest(rung1_steps, control_steps)
        calibration_json_path = os.path.join(args.out_dir, "calibration.json")
        timing = {}
        timing_status = "no calibration.json found yet"
        if os.path.exists(calibration_json_path):
            with open(calibration_json_path) as f:
                timing = (json.load(f) or {}).get("timing_constants") or {}
            missing = [k for k in ("rung1", "control") if k not in timing]
            timing_status = (f"WIRED, both configs present: {timing}" if not missing
                              else f"calibration.json exists but missing timing_constants for {missing}")
        print(f"rung {RUNG} config: {RUNG_CFG} (verified 0.4% off target this session, see "
              f"lm_rd_rung_configs.py / lm_pretrain_rd.py smoke item [11])")
        print(f"control config: {CONTROL_CFG} (Wave C's own scale)")
        print(f"rung 2 config: {RUNG2_CFG}, rung 3 config: {RUNG3_CFG} (lm_rd_rung_configs.py / "
              f"lm_pretrain_rd.py smoke items [12]/[13], BATCH_SIZE_BY_RUNG={BATCH_SIZE_BY_RUNG})")
        calib_tag_steps = {t: CALIBRATION_TWO_POINT_STEPS[t] for t in calib_tags}
        print(f"\ncalibration (Wave -1), --calib-rungs={args.calib_rungs!r}: {len(calib_m)} runs "
              f"(two-point per selected config: {calib_tag_steps}) -- MAY be launched any "
              f"time (cheap, <2 GPU-h per rung)")
        print(f"\nwave 1: {len(w1_m)} runs -- CLOSED (sec 5.9) -- rung-1 {rung1_steps} steps x "
              f"{len(MIX_CORPORA)} corpora x {len(SEEDS)} seeds, control {control_steps} steps x "
              f"{len(MIX_CORPORA)} x {len(SEEDS)}. Wave-1 timeout wiring status: {timing_status}")

        for rung in (2, 3):
            placeholder_steps = default_rung23_steps_placeholder(rung)
            steps_arg = getattr(args, f"rung{rung}_steps")
            steps = steps_arg or placeholder_steps
            w_m = wave23_manifest(rung, steps)
            key = WAVE_TIMING_KEY[rung]
            n_seeds = len(WAVE23_SEEDS_BY_RUNG[rung])
            wave_timing_status = (f"WIRED: {timing[key]}" if key in timing
                                   else f"NOT calibrated yet (run --wave calibration --calib-rungs {rung})")
            # {len(w_m)} is the run count that matters (derived from the actual manifest, which is
            # itself built from WAVE23_SEEDS_BY_RUNG[rung] -- amendment item 1 -- never a hardcoded
            # literal); the "x corpora x seeds" text below is a descriptive breakdown of that count.
            print(f"\nwave {rung}: {len(w_m)} runs -- rung-{rung} {steps} steps"
                  f"{' (UNCALIBRATED PLACEHOLDER, pass --rung' + str(rung) + '-steps after calibration)' if not steps_arg else ''}"
                  f" x {len(WAVE23_CORPORA)} extended-mix corpora x {n_seeds} seed(s), batch="
                  f"{BATCH_SIZE_BY_RUNG[rung]} -- real launch requires (a) calibration.json's "
                  f"timing_constants[{key!r}], (b) --rung{rung}-steps, (c) a passing memory-headroom "
                  f"readout, (d) a passing epoch-cap check on both extended-mix corpora, (e) the sec-7 "
                  f"budget guard (PROGRAM_SPENT_GPUH={PROGRAM_SPENT_GPUH} + projected <= "
                  f"{GPU_H_PROGRAM_CEILING} GPU-h, else --accept-budget-override). "
                  f"Timing wiring status: {wave_timing_status}")
            if key in timing:
                projected = projected_gpu_hours(w_m, timing[key])
                print(f"  projected GPU-h at these steps: {projected:.2f} (cumulative with "
                      f"PROGRAM_SPENT_GPUH: {PROGRAM_SPENT_GPUH + projected:.2f} / "
                      f"{GPU_H_PROGRAM_CEILING:.0f})")
            for corpus in WAVE23_CORPORA:
                try:
                    rep = epoch_cap_check(args.data_dir, corpus, steps * BATCH_SIZE_BY_RUNG[rung] * SEQ_LEN)
                    print(f"  epoch-cap[{corpus}]: planned {rep['planned_tokens']:,} tokens vs. "
                          f"ceiling {rep['epoch_cap_ceiling_tokens']:,} -- "
                          f"{'OK' if rep['ok'] else 'WOULD REFUSE (pull more augmentation first)'}")
                except FileNotFoundError:
                    print(f"  epoch-cap[{corpus}]: meta.json not found under --data-dir {args.data_dir!r} "
                          f"(preview only reachable on-box)")

        # sec 5.6 amendment (Rev 2.1) item 3, NEW: --wave mixcontrol preview, same launch-parity
        # discipline as waves 2/3 above -- reuses Wave 1's own control_steps (computed above) so the
        # preview and a real launch (gate_and_run_mixcontrol) can never silently drift apart.
        mixc_m = mixcontrol_manifest(control_steps)
        mixc_timing_status = (f"WIRED: {timing['control']}" if "control" in timing
                               else "NOT calibrated yet (run --wave calibration)")
        print(f"\nwave mixcontrol: {len(mixc_m)} runs -- CONTROL_CFG {control_steps} steps (same "
              f"control-length budget as Wave 1's own control cell) x {len(WAVE23_CORPORA)} "
              f"extended-mix corpora x {len(SEEDS)} seeds -- real launch requires (a) "
              f"calibration.json's timing_constants['control'] (already banked from Wave 1), (c) a "
              f"passing memory-headroom readout, (d) a passing epoch-cap check on both extended-mix "
              f"corpora (trivially passes at control-length steps, but not skipped), (e) the sec-7 "
              f"budget guard. Timing wiring status: {mixc_timing_status}")
        if "control" in timing:
            mixc_projected = projected_gpu_hours(mixc_m, timing["control"])
            print(f"  projected GPU-h at these steps: {mixc_projected:.2f} (cumulative with "
                  f"PROGRAM_SPENT_GPUH: {PROGRAM_SPENT_GPUH + mixc_projected:.2f} / "
                  f"{GPU_H_PROGRAM_CEILING:.0f})")
        for corpus in WAVE23_CORPORA:
            try:
                rep = epoch_cap_check(args.data_dir, corpus, control_steps * BATCH_SIZE * SEQ_LEN)
                print(f"  epoch-cap[{corpus}]: planned {rep['planned_tokens']:,} tokens vs. "
                      f"ceiling {rep['epoch_cap_ceiling_tokens']:,} -- "
                      f"{'OK' if rep['ok'] else 'WOULD REFUSE (pull more augmentation first)'}")
            except FileNotFoundError:
                print(f"  epoch-cap[{corpus}]: meta.json not found under --data-dir {args.data_dir!r} "
                      f"(preview only reachable on-box)")

        trackb = _trackb_gate_status(args.trackb_gate_json)
        print(f"\nwave 4 (fix-effect / geo3-at-scale): ALWAYS REFUSED. Track B gate status: {trackb}")
        if args.gpus is not None and args.gpu_offset is not None:
            print(f"\nslots = {args.gpus} gpus x {args.per_gpu} per-gpu, physical GPUs "
                  f"{list(range(args.gpu_offset, args.gpu_offset + args.gpus))}")
        else:
            print("\nslots: pass --gpus/--gpu-offset to preview (REQUIRED for a real launch).")
        return

    if args.wave is None:
        print("ERROR: --wave is required for a real (non-dry-run) launch.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "4":
        refuse_wave4(args.trackb_gate_json)   # always exits non-zero, never returns

    if args.gpus is None or args.gpu_offset is None:
        print("ERROR: --gpus and --gpu-offset are REQUIRED for a real launch (no defaults on "
              "purpose) -- run nvidia-smi NOW and pass the free set explicitly.", file=sys.stderr)
        sys.exit(1)

    if args.wave == "calibration":
        calib_tags = parse_calib_rungs(args.calib_rungs)
        manifest = calibration_manifest(tags=calib_tags)
        out_dir = os.path.join(args.out_dir, "calibration")
        os.makedirs(out_dir, exist_ok=True)

        def timeout_fn(spec):
            tag_base = spec["tag"].rsplit("_pt", 1)[0]
            per_step_gen, per_ckpt_gen = _calibration_generous_timing(tag_base)
            return default_timeout_pretrain(spec["steps"], spec["ckpt_every"], per_step_gen, per_ckpt_gen)

        all_done = _run_wave("calibration", manifest, out_dir, args, is_done_cell, build_cmd_cell, timeout_fn)
        # Summarize the FULL cross-rung universe of calibration cells found on disk (NOT just this
        # invocation's `manifest` subset) -- a --calib-rungs-restricted run must never clobber
        # previously-recorded cells for OTHER configs (summarize_calibration's own docstring).
        full_manifest = calibration_manifest(tags=tuple(CALIBRATION_RUNG_CFGS))
        calib_summary = summarize_calibration(out_dir, full_manifest)
        with open(os.path.join(out_dir, "calibration.json"), "w") as f:
            json.dump(calib_summary, f, indent=2)
        print("\nCALIBRATION SUMMARY:", json.dumps(calib_summary, indent=2))
        if all_done:
            # also drop a copy at the wave-1/2/3 gates' expected lookup location (out-dir root) so
            # a real launch check (sec 9) finds it without extra plumbing. Written even when this
            # invocation only targeted a SUBSET of configs (--calib-rungs) -- calib_summary itself
            # is always the full cross-rung union (see full_manifest above), so this is safe.
            with open(os.path.join(args.out_dir, "calibration.json"), "w") as f:
                json.dump(calib_summary, f, indent=2)
        else:
            print(f"NOTE: --wave calibration did not fully complete this invocation's requested "
                  f"subset ({calib_tags}) -- the out-dir-root calibration.json was NOT overwritten "
                  f"this run (whatever was there before, if anything, is unchanged). Re-run "
                  f"--wave calibration --calib-rungs {args.calib_rungs} to retry the pending/failed "
                  f"cells; already-complete cells resume-skip.", flush=True)
        return

    if args.wave in ("2", "3"):
        # trackC-rung23-build: REAL manifest-build + _run_wave call behind these gates -- verified
        # by direct re-read of gate_and_run_wave23 after writing it (module docstring's "LEARNED
        # FROM WAVE 1'S OWN HISTORY" paragraph: a gate with nothing behind it is a no-op).
        gate_and_run_wave23(int(args.wave), args)
        return  # gate_and_run_wave23 always sys.exit()s; return is unreachable but keeps main() honest

    if args.wave == "mixcontrol":
        # sec 5.6 amendment (Rev 2.1) item 3, NEW: same "verified by direct re-read" discipline as
        # waves 2/3 above -- gate_and_run_mixcontrol always sys.exit()s.
        gate_and_run_mixcontrol(args)
        return  # unreachable; keeps main() honest, matching the wave-2/3 convention above

    # --wave 1 (trackC-audit finding #1: this used to be an unconditional `sys.exit(4)` after the
    # two checks below -- no manifest was ever built and no run was ever launchable, regardless of
    # calibration/--rung1-steps state. Replaced with the actual launch path, gated the SAME way
    # every other wave in this codebase is gated: calibration.json must exist AND carry real
    # timing_constants for BOTH configs (not just exist), AND --rung1-steps must be supplied. There
    # is no other special-case refusal here -- a human/orchestrator satisfying those two normal
    # gates can launch for real. CLOSED as of sec 5.9 -- kept working, unmodified, for reruns.)
    calibration_json_path = os.path.join(args.out_dir, "calibration.json")
    if not os.path.exists(calibration_json_path):
        print(f"ERROR: {calibration_json_path} not found -- sec 9's own hard rule: 'No track's Wave "
              f"1+ manifest is authorized to launch ... without first recording its Wave -1 measured "
              f"numbers.' Run --wave calibration first.", file=sys.stderr)
        sys.exit(2)
    if args.rung1_steps is None:
        print("ERROR: --rung1-steps is REQUIRED for a real --wave 1 launch (no silent fallback to "
              "the uncalibrated RUNG1_TARGET_TOKENS_PLACEHOLDER) -- derive it from "
              f"{calibration_json_path}'s measured tok/s and pass it explicitly.", file=sys.stderr)
        sys.exit(2)
    with open(calibration_json_path) as f:
        calib = json.load(f)
    timing = calib.get("timing_constants") or {}
    missing = [k for k in ("rung1", "control") if k not in timing]
    if missing:
        print(f"ERROR: {calibration_json_path} has no timing_constants for {missing} -- rerun "
              f"--wave calibration (two-point method, trackC-audit finding #2) so BOTH configs' "
              f"per_step_s/per_ckpt_s are populated before a real --wave 1 launch (sec 9's hard "
              f"rule: no launch on an unmeasured/estimated timeout).", file=sys.stderr)
        sys.exit(2)

    control_steps = args.control_steps or default_control_steps()
    manifest = wave1_manifest(args.rung1_steps, control_steps)
    out_dir = os.path.join(args.out_dir, "wave1")
    os.makedirs(out_dir, exist_ok=True)

    def timeout_fn(spec):
        key = "rung1" if spec["tag"] == "w1_rung1" else "control"
        c = timing[key]
        return default_timeout_pretrain(spec["steps"], spec["ckpt_every"],
                                         c["per_step_s"], c["per_ckpt_s"], margin=LAUNCH_TIMEOUT_MARGIN)

    print(f"WAVE 1 REAL LAUNCH: {len(manifest)} runs (rung-1 {args.rung1_steps} steps x "
          f"{len(MIX_CORPORA)} corpora x {len(SEEDS)} seeds + control {control_steps} steps x "
          f"{len(MIX_CORPORA)} x {len(SEEDS)}). Timing constants from {calibration_json_path} "
          f"(margin {LAUNCH_TIMEOUT_MARGIN}x): {json.dumps(timing, indent=2)}", flush=True)
    all_done = _run_wave("1", manifest, out_dir, args, is_done_cell, build_cmd_cell, timeout_fn)
    sys.exit(0 if all_done else 1)


if __name__ == "__main__":
    main()
