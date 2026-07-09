"""CAPABILITY_SEPARATION_DESIGN.md S1.4.2/S1.6/S1.7 -- the calibration-first
wave runner + the 58-cell sweep manifest, escalation triggers wired to
budget_guard.BudgetGuard, resume-safe (checks output VALIDITY, not mere
existence, per CLAUDE.md's hard rule: "make the orchestrator itself
resume-safe (skip already-completed work by checking output validity, not
just existence)"), tmux/supervisor-compatible (a single invocation processes
the pending manifest and exits cleanly -- pair with `while [ ! -f STOP ];
do <cmd>; sleep 15; done` on the box, CLAUDE.md's supervisor-loop pattern).

BUILD-STAGE SCOPE (per this build agent's own mandate: "you BUILD; you do
not launch GPU sweeps and do not self-certify"): this script is fully wired
and unit-exercised in `--smoke` mode (a handful of cells, ~20 steps each, on
CPU) below -- it does NOT launch the real 8,000-step/58-cell production
sweep here. `--smoke` proves the manifest/resume/budget-guard/TOST wiring
end-to-end; the real launch is a separate, PI-signed-off, GPU-hardware step
(S1.7 gate 5: `CAPABILITY_SEP_PI_SIGNOFF=1` required before any GPU cell).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groups import GROUP_NAMES, D_MIN, D_STATE, group_seed_salt, batched_targets
from group_task import generating_set, sample_train_batch, sample_eval_batch
from group_word_encoder import GroupWordModel, cosine_loss, recovery_cosine
from force_rank_arms import force_rank_grid, cell_types_for_group, n_cells_per_group
import readout
import budget_guard as bg
import tost_analysis as tost

PI_SIGNOFF_VAR = "CAPABILITY_SEP_PI_SIGNOFF"
RESULTS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_STEPS = 8000       # S1.6's original planned-default step budget (pre-Rev-7 uniform value;
                           # kept as the smoke/CLI-override fallback, NOT the production per-group rate)

# S1.6/S1.7 gate 1(a) Rev 7 -- per-group step-budget pins (adjudication round
# 7, S1.30, EXECUTED against the real box escalation cells, not asserted):
# S3/S5 clear on their FIRST measurement at 8,000 steps; S4/A5 clear after
# their standard 2-2.5x retest to 20,000; A6 alone needed the rule-(c)
# diagnostic-routed SECOND escalation to 40,000 (20,000 cleared >=0.9 by
# only +0.0023, below the new >=0.02 margin rule). See S1.7 gate 1(a)'s
# "Box re-verification (Rev 7...)" table for the exact per-group min-L2-5
# values these pins were derived from.
STEP_BUDGET = {"S3": 8000, "S4": 20000, "A5": 20000, "S5": 8000, "A6": 40000}

# S1.7 gate 1(a) Rev 7 (H-ENC fix, S1.30 item 1) -- the HARD convergence bar
# now applies to L in {2,3,4,5} only (L=1 demoted/disclosed, a proven
# query-independent single-key-attention degeneracy, not a convergence
# defect); a pinned budget must clear tau=0.9 by >= GATE1A_MARGIN, not a
# bare >0 clearance (S1.30 item 1's own noise-margin justification).
GATE1A_L_RANGE = (2, 3, 4, 5)
GATE1A_TAU = 0.9
GATE1A_MARGIN = 0.02

# S1.7 gate 1(a) rule (c), RECALIBRATED Rev 7 (S1.30 item 4): at most 2
# escalations/group; a second consecutive miss requires a mandatory
# <=0.1 GPU-h mechanism diagnostic BEFORE any further action, routed by
# the diagnostic's own finding.
MAX_ESCALATIONS_PER_GROUP = 2
MECHANISM_DIAGNOSTIC_BUDGET_H = 0.1
GATE1A_ROUTES = ("instrument_fix", "one_capped_escalation", "demote_disclose", "hard_stop")


# ---------------------------------------------------------------------------
# Manifest construction (S1.4.2's exact cell table).
# ---------------------------------------------------------------------------

def build_sweep_manifest() -> list:
    """The full 58-cell manifest (S1.4.2): for each group, for each arm
    (unconstrained/k_dmin_minus_1/k_dmin/k_dmin_plus_1), one cell per seed
    at that cell type's BASE seed count (S4/A5 already unconditional n=5 at
    unconstrained+k_dmin). Escalation-triggered cells are NOT pre-included
    here -- they are appended dynamically by `run_escalations` after the
    base sweep's results are in, gated through budget_guard.

    S1.6/S1.7 gate 1(a) Rev 7: each cell carries its own group's PINNED
    step budget (`STEP_BUDGET`) rather than a single sweep-wide scalar --
    `train_and_eval_cell`/`run_cell_resume_safe` read `cell["steps"]` by
    default (a `steps_override` still exists for smoke/testing)."""
    manifest = []
    for name in GROUP_NAMES:
        unconditional_n5 = name in ("S4", "A5")
        cell_types = cell_types_for_group(name, unconditional_n5)
        for arm_label, n_seeds in cell_types.items():
            for seed in range(n_seeds):
                cell_id = f"{name}__{arm_label}__seed{seed}"
                manifest.append(dict(cell_id=cell_id, group=name, arm=arm_label, seed=seed,
                                     steps=STEP_BUDGET[name],
                                     is_calibration=(arm_label == "unconstrained" and seed == 0)))
    assert len(manifest) == 58, f"expected 58 base cells, got {len(manifest)}"
    return manifest


def calibration_wave() -> list:
    """S1.7 gate 1: the 5 calibration cells (1/group, unconstrained arm,
    seed=0) -- a SUBSET of build_sweep_manifest(), reused (not
    double-charged, S1.6)."""
    full = build_sweep_manifest()
    calib = [c for c in full if c["is_calibration"]]
    assert len(calib) == 5, f"expected 5 calibration cells, got {len(calib)}"
    return calib


# ---------------------------------------------------------------------------
# S1.7 gate 1(a) Rev 7 -- per-L convergence profile, the L in {2..5}/margin
# bar, and rule (c)'s diagnostic-before-action escalation routing.
# ---------------------------------------------------------------------------

def _eval_batch_at_L(name: str, L: int, batch_size: int, gen, device="cpu",
                     dtype=torch.float32, target_padding: str = "eye") -> dict:
    """Fresh, length-HOMOGENEOUS eval batch at a SPECIFIC L (S1.7 gate
    1(a)'s per-L convergence metric) -- reuses the SAME
    generating_set/batched_targets primitives sample_train_batch/
    sample_eval_batch already wrap (group_task.py), just pinned to one L
    rather than randomly drawn from a range.

    `target_padding` (S1.33 M3 FIX WAVE flag, default UNCHANGED "eye"):
    threaded through to groups.batched_targets so this diagnostic scores
    against the SAME target family the cell was actually trained on."""
    n_gens = len(generating_set(name))
    d_state = D_STATE[name]
    token_idx = torch.randint(0, n_gens, (batch_size, L), generator=gen).to(device)
    target = batched_targets(name, token_idx, d_state, dtype=dtype, target_padding=target_padding)
    return {"token_idx": token_idx, "target": target, "L": L}


def per_L_convergence_profile(model, name: str, device: str, seed: int,
                              batch_size: int = 64, l_lo: int = 1, l_hi: int = 8,
                              target_padding: str = "eye") -> dict:
    """S1.7 gate 1(a) (Rev 5/7): for each L in {l_lo..l_hi} (default the
    TRAIN-support range, S1.4), draw a FRESH, length-homogeneous eval batch
    (never used for M1/M3 scoring or training) and score the SAME
    `recovery_cosine` primitive already used for training's `cosine_loss`,
    in `model.eval()`/`no_grad` mode, against the RAW (not degauged --
    this is a diagnostic convergence check, not the M1/M3 decision metric)
    `rho_G_embedded` target. Returns {L: mean_cosine} for L=l_lo..l_hi --
    an 8-point profile at the default range, replacing reliance on the
    training loop's last logged batch loss (S1.25's own diagnosed
    per-batch-fixed-L confound).

    `target_padding` (S1.33 M3 FIX WAVE flag, default UNCHANGED "eye"):
    threaded through to `_eval_batch_at_L` -- see its docstring."""
    model.eval()
    gen = torch.Generator().manual_seed(seed + 555_000 + group_seed_salt(name))
    profile = {}
    with torch.no_grad():
        for L in range(l_lo, l_hi + 1):
            batch = _eval_batch_at_L(name, L, batch_size, gen, device=device,
                                     target_padding=target_padding)
            Z = model.encode(batch["token_idx"])
            profile[L] = float(recovery_cosine(Z, batch["target"]).mean().item())
    return profile


def gate1a_check(profile: dict, l_range=GATE1A_L_RANGE, tau: float = GATE1A_TAU,
                 margin: float = GATE1A_MARGIN) -> dict:
    """S1.7 gate 1(a) Rev 7's HARD bar: min mean-cosine over L in `l_range`
    (default {2,3,4,5}) must clear `tau` (0.9) by AT LEAST `margin` (0.02)
    -- a bare `>tau` clearance is NOT sufficient (S1.30 item 1's own
    noise-margin justification: A6@20K cleared 0.9 by only +0.0023 yet sat
    materially below its own higher-budget ceiling). `L=1` is intentionally
    excluded (demoted/disclosed, the H-ENC mechanism, S1.30) but reported
    alongside for the harvest's own per-L disclosure. Pure function, no
    GPU/model dependency -- unit-testable directly (this file's `smoke()`,
    section 13's new gate1a sub-tests)."""
    vals = {L: profile[L] for L in l_range}
    min_L = min(vals, key=vals.get)
    min_val = vals[min_L]
    gap = min_val - tau
    clears = gap >= margin
    return dict(l_range=list(l_range), min_val=min_val, at_L=min_L, margin_over_tau=gap,
                required_margin=margin, tau=tau, clears=clears,
                l1_disclosed=profile.get(1), full_profile=dict(profile))


def route_second_miss(mechanism_diagnostic: dict, escalations_used: int,
                      max_escalations: int = MAX_ESCALATIONS_PER_GROUP) -> str:
    """S1.7 gate 1(a) rule (c), RECALIBRATED Rev 7 (S1.30 item 4): on a
    group's SECOND consecutive gate1a_check miss at its own pinned budget,
    a mechanism diagnostic is MANDATORY before any further action (the
    caller must supply `mechanism_diagnostic` -- there is no default/skip
    path, enforcing "diagnostic-before-action" mechanically, not just as a
    written rule). Routes on `mechanism_diagnostic["class"]`, one of
    {"instrument", "moving_below_ceiling", "ceiling", "inconclusive"}
    (this round's own `l1_micro_diag.py` suite is the diagnostic
    TEMPLATE, S1.30/S1.7 gate 1(a)): instrument defect -> fix, no
    escalation consumed; genuinely moving but below ceiling -> one further
    capped escalation IF the `<=2`-escalations/group cap has room;
    architectural ceiling -> demote the affected L + disclose + flag
    Stage 2 (mirrors the L=1 demotion this round produced); anything else
    (inconclusive, or the escalation cap is already exhausted with no
    ceiling/plateau evidence) -> HARD-STOP, reserved for genuine
    pathology, not the default outcome."""
    if mechanism_diagnostic is None:
        raise ValueError(
            "route_second_miss: a mechanism diagnostic is MANDATORY on a second consecutive "
            "gate1a_check miss (S1.7 gate 1(a) rule (c), Rev 7) -- no diagnostic-free routing "
            "path exists. Run the <=0.1 GPU-h diagnostic (l1_micro_diag.py's suite as template) "
            "before calling route_second_miss()."
        )
    cls = mechanism_diagnostic.get("class")
    if cls == "instrument":
        return "instrument_fix"
    if cls == "moving_below_ceiling":
        if escalations_used < max_escalations:
            return "one_capped_escalation"
        return "hard_stop"   # cap exhausted -- moving but out of budget is a genuine stop, not silent
    if cls == "ceiling":
        return "demote_disclose"
    return "hard_stop"   # "inconclusive" or any unrecognized class -- reserved for genuine pathology


# ---------------------------------------------------------------------------
# S1.22 BA-F1 fix -- the two-step CLI's gate: `--calibration-only` writes a
# calibration report (the REAL measured per-cell rate, S1.7 gate 1 duty d);
# `--sweep` REFUSES to start unless (i) that report exists and is valid for
# the requested --steps, (ii) the projection from the MEASURED rate clears
# the 30 GPU-h cap via budget_guard.check_base_sweep_projection (S1.7 gate
# 2's mechanical enforced abort), and (iii) the PI-signoff token is present.
# Before this fix, check_base_sweep_projection/BaseSweepAbort were never
# called anywhere in this file -- the calibration-first gate did NOT
# actually block the sweep (S1.22 BA-F1).
# ---------------------------------------------------------------------------

def calibration_report_path(results_dir: str) -> str:
    return os.path.join(results_dir, "calibration_report.json")


def write_calibration_report(results_dir: str, calib_results: list) -> dict:
    """S1.7 gate 1 duty (d): aggregate the 5 calibration cells' measured
    wall-clock into a real per-cell GPU-h rate, written to disk so `--sweep`
    can gate on it without re-running calibration. One GPU trains one cell
    serially in this harness's execution model (run_manifest processes the
    manifest sequentially, one cell at a time), so wall_clock_s/3600 IS the
    GPU-h/cell figure -- the real rate that supersedes S1.6's 0.3 GPU-h/cell
    planning estimate.

    S1.6/S1.7 gate 1(a) Rev 7 build item: `steps` becomes `steps_per_group`
    (a per-GROUP dict, derived from each calibration cell's own
    `steps_completed`) rather than a single scalar -- `--sweep`'s gate
    (`load_and_validate_calibration_report` below) validates the report
    against the ACTUAL steps EACH GROUP will run, not one value that can no
    longer describe a manifest with per-group-pinned budgets."""
    rates = [r["wall_clock_s"] / 3600.0 for r in calib_results]
    real_rate_per_cell = sum(rates) / len(rates)
    steps_per_group = {r["group"]: r["steps_completed"] for r in calib_results}
    report = dict(
        real_rate_per_cell=real_rate_per_cell,
        per_cell_rates=rates,
        n_cells=len(calib_results),
        steps_per_group=steps_per_group,
        cell_ids=sorted(r["cell_id"] for r in calib_results),
        groups=sorted({r["group"] for r in calib_results}),
        written_at=time.time(),
    )
    path = calibration_report_path(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, path)   # atomic write, same convention as run_cell_resume_safe
    return report


def load_and_validate_calibration_report(results_dir: str, expected_steps_per_group: dict | None = None) -> dict:
    """`--sweep` precondition (i): a valid, on-disk calibration report whose
    PER-GROUP step budgets (Rev 7) match `expected_steps_per_group`
    (default `STEP_BUDGET`, the production pins). Raises RuntimeError (not
    BaseSweepAbort -- that is reserved for a budget-projection failure, not
    a missing/invalid report) with a specific, actionable reason."""
    expected = STEP_BUDGET if expected_steps_per_group is None else expected_steps_per_group
    path = calibration_report_path(results_dir)
    if not os.path.exists(path):
        raise RuntimeError(
            f"no calibration report at {path} -- run `--calibration-only` first and let it "
            f"complete (S1.7 gate 1: calibration must run to completion before the sweep)."
        )
    try:
        with open(path) as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"calibration report at {path} is corrupt/unreadable: {e}")
    required = {"real_rate_per_cell", "n_cells", "steps_per_group", "cell_ids", "groups"}
    missing = required - report.keys()
    if missing:
        raise RuntimeError(f"calibration report at {path} missing required keys {missing}")
    if report["n_cells"] != 5 or len(report["cell_ids"]) != 5:
        raise RuntimeError(
            f"calibration report at {path} has {report['n_cells']} cells "
            f"({len(report['cell_ids'])} cell_ids) -- expected the full 5-group calibration wave"
        )
    if set(report["groups"]) != set(GROUP_NAMES):
        raise RuntimeError(f"calibration report groups {report['groups']} != expected {list(GROUP_NAMES)}")
    if report["steps_per_group"] != expected:
        raise RuntimeError(
            f"calibration report at {path} was measured at steps_per_group="
            f"{report['steps_per_group']}, but this launch expects steps_per_group={expected} -- "
            f"re-run `--calibration-only` first (a rate measured at different per-group step "
            f"budgets does not project this sweep's per-cell cost)."
        )
    rate = report["real_rate_per_cell"]
    if not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
        raise RuntimeError(f"calibration report at {path} has an invalid real_rate_per_cell={rate!r}")
    return report


def gate_sweep_launch(results_dir: str, expected_steps_per_group: dict | None = None) -> dict:
    """The three `--sweep` preconditions (S1.22 BA-F1), in the design's own
    order: (i) a valid calibration report on disk, (ii) its measured rate
    projects the base sweep under the 30 GPU-h cap (raises BaseSweepAbort
    via budget_guard.check_base_sweep_projection if not -- S1.7 gate 2's
    mechanical enforced abort, now actually wired), (iii) the PI-signoff
    token. Factored out of run_sweep() so smoke can exercise the gate in
    isolation without needing a full 58-cell GPU sweep to run."""
    report = load_and_validate_calibration_report(results_dir, expected_steps_per_group)  # (i)
    projection = bg.check_base_sweep_projection(report["real_rate_per_cell"])          # (ii)
    if os.environ.get(PI_SIGNOFF_VAR) != "1":                                          # (iii)
        raise RuntimeError(f"{PI_SIGNOFF_VAR}=1 required before any GPU cell (S1.7 gate 5).")
    return dict(calibration_report=report, projection=projection)


def _synthetic_calibration_results(steps, rate_per_cell: float) -> list:
    """Build cell-shaped result dicts (NOT real training) with `wall_clock_s`
    set so `write_calibration_report`'s averaged rate equals `rate_per_cell`
    exactly -- for smoke ONLY, to exercise the report/gate wiring without
    paying for 5 real training runs on every smoke invocation. `steps` is
    either a single int (uniform across all 5 groups, the common smoke case)
    or a per-group dict (to exercise the Rev 7 per-group pins directly)."""
    calib = calibration_wave()
    wall_s = rate_per_cell * 3600.0
    def steps_for(name):
        return steps[name] if isinstance(steps, dict) else steps
    return [dict(cell_id=c["cell_id"], group=c["group"], arm=c["arm"], seed=c["seed"],
                force_rank_k=None, steps_completed=steps_for(c["group"]), n_skipped_steps=0,
                wall_clock_s=wall_s, mean_cos=0.95, recovered_frac_90=0.95,
                crosscheck_mean_cos=0.95, crosscheck_recovered_frac_90=0.95,
                restricted_effective_rank=3.0, restricted_stable_rank=3.0,
                whole_matrix_effective_rank=4.0, c_hat=1.0)
           for c in calib]


# ---------------------------------------------------------------------------
# Resume-safety: output VALIDITY, not mere existence.
# ---------------------------------------------------------------------------

def cell_output_path(results_dir: str, cell_id: str) -> str:
    return os.path.join(results_dir, f"{cell_id}.json")


def is_valid_output(path: str) -> bool:
    """A cell's output is VALID iff the file exists, parses as JSON, has the
    expected keys, and contains no NaN/Inf in the decision-relevant fields
    -- CLAUDE.md's hard rule: "skip already-completed work by checking
    output validity, not just existence" (a truncated/corrupted write from
    a crashed prior attempt must NOT be treated as done)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    required = {"cell_id", "group", "arm", "seed", "mean_cos", "recovered_frac_90",
               "restricted_effective_rank", "steps_completed"}
    if not required.issubset(d.keys()):
        return False
    for key in ("mean_cos", "recovered_frac_90", "restricted_effective_rank"):
        v = d[key]
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return False
    return True


# ---------------------------------------------------------------------------
# One cell: train + S1.4.1 eval pipeline.
# ---------------------------------------------------------------------------

def train_and_eval_cell(cell: dict, device: str, log_every: int = 500,
                        steps_override: int | None = None) -> dict:
    """S1.6/S1.7 gate 1(a) Rev 7: trains at the cell's OWN per-group pinned
    step budget (`cell["steps"]`, set by `build_sweep_manifest`/
    `STEP_BUDGET`) unless `steps_override` is given (smoke/testing only --
    a uniform tiny step count for every cell regardless of group).

    S1.33 M3 FIX WAVE: a cell dict MAY carry its own explicit `force_rank_k`
    (int or None) and `target_padding` ("eye"/"zero") keys -- when present,
    these OVERRIDE the base sweep's `force_rank_grid(name)[arm]` lookup and
    the default "eye" padding, respectively (build_m3fix_manifest's cells
    always set both explicitly). The base 58-cell manifest's cells NEVER
    set these keys, so this is byte-identical to the pre-S1.33 behavior for
    every existing sweep/smoke call site -- checked via `"force_rank_k" in
    cell` (not `.get(..., default)`) so an EXPLICIT `force_rank_k=None`
    (the m3fix zero_pad variant's unconstrained anchor) is honored
    correctly, distinct from "key absent, derive via arm/force_rank_grid"."""
    name = cell["group"]
    arm = cell["arm"]
    seed = cell["seed"]
    steps = steps_override if steps_override is not None else cell["steps"]
    k = cell["force_rank_k"] if "force_rank_k" in cell else force_rank_grid(name)[arm]
    target_padding = cell.get("target_padding", "eye")

    # NOTE: the GLOBAL torch seed (model init) is deliberately left UNSALTED --
    # S4/A5 sharing bit-identical init weights at the same cell seed is a KNOWN,
    # ADJUDICATED channel (S1.4.2.1: exactly why the marquee check uses Welch's
    # UNPAIRED TOST rather than a paired test that would assume it washes out).
    torch.manual_seed(seed)
    d_state = D_STATE[name]
    n_gens = len(generating_set(name))
    model = GroupWordModel(d_state, n_gens, L_max=16, h=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    # S1.22 BA-F3 fix: the TRAINING-BATCH generator IS salted by group name --
    # unlike model init above, this is an undisclosed, unadjudicated correlation
    # channel (S4/A5 share |gens|=4 too, so an unsalted seed here would draw
    # byte-identical training batches, not just byte-identical init weights).
    gen = torch.Generator().manual_seed(seed + group_seed_salt(name))

    n_skipped = 0
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = sample_train_batch(name, 256, gen, device=device, target_padding=target_padding)
        Z = model.encode(batch["token_idx"], force_rank_k=k)
        loss = cosine_loss(Z, batch["target"])
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1
        if step % log_every == 0 or step == 1:
            print(f"  [{cell['cell_id']}] step {step}/{steps}  loss={loss.item():.4f}"
                 f"{'  [skipped '+str(n_skipped)+']' if n_skipped else ''}", flush=True)
    wall_s = time.time() - t0

    eval_seed = seed + 10_000
    scores = readout.run_subspace_restriction_pipeline(model, name, base_seed=eval_seed, device=device,
                                                        force_rank_k=k)

    # S1.7 gate 1(a) Rev 5/7: per-L convergence profile (diagnostic, NOT the
    # M1/M3 decision metric) + the L in {2..5}/margin HARD-bar check.
    profile = per_L_convergence_profile(model, name, device=device, seed=eval_seed,
                                        target_padding=target_padding)
    gate1a = gate1a_check(profile)

    # S1.5 M1 harvest-reporting disclosure (Rev 7, S1.30 item 5): the L>=2
    # robustness split, computed from the SAME degauged eval-set cosines
    # `run_subspace_restriction_pipeline` already scored (readout.py's new
    # `eval_L` field) -- never re-drawn, so it costs nothing extra and stays
    # index-consistent with the decisional (full-sample) M1 number.
    eval_L = scores.get("eval_L", [])
    coses = scores.get("coses", [])
    l_ge2_coses = [c for c, L in zip(coses, eval_L) if L >= 2]
    l_ge2_mean_cos = float(np.mean(l_ge2_coses)) if l_ge2_coses else None
    l_ge2_recovered_frac_90 = float(np.mean([c > 0.9 for c in l_ge2_coses])) if l_ge2_coses else None

    result = dict(cell_id=cell["cell_id"], group=name, arm=arm, seed=seed, force_rank_k=k,
                 steps_completed=steps, n_skipped_steps=n_skipped, wall_clock_s=wall_s,
                 mean_cos=scores["mean_cos"], recovered_frac_90=scores["recovered_frac_90"],
                 crosscheck_mean_cos=scores["crosscheck_mean_cos"],
                 crosscheck_recovered_frac_90=scores["crosscheck_recovered_frac_90"],
                 restricted_effective_rank=scores["restricted_effective_rank"],
                 restricted_stable_rank=scores["restricted_stable_rank"],
                 whole_matrix_effective_rank=scores["whole_matrix_effective_rank"],
                 c_hat=scores["c_hat"],
                 convergence_profile=profile, gate1a=gate1a,
                 l_ge2_mean_cos=l_ge2_mean_cos, l_ge2_recovered_frac_90=l_ge2_recovered_frac_90,
                 l_ge2_n_eval=len(l_ge2_coses), l_ge2_n_total=len(eval_L),
                 # S1.33 M3 FIX WAVE provenance (harmless additions for the base 58-cell sweep,
                 # whose cells never set "target_padding"/"variant" -- target_padding defaults to
                 # "eye" and m3fix_variant to None, both cosmetic/disclosure-only, never gating).
                 target_padding=target_padding, m3fix_variant=cell.get("variant"))
    return result


def run_cell_resume_safe(cell: dict, results_dir: str, device: str,
                         steps_override: int | None = None) -> dict:
    path = cell_output_path(results_dir, cell["cell_id"])
    if is_valid_output(path):
        with open(path) as f:
            print(f"  [{cell['cell_id']}] SKIP (valid output already on disk)")
            return json.load(f)
    result = train_and_eval_cell(cell, device, steps_override=steps_override)
    os.makedirs(results_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, path)   # atomic write -- a crash mid-write never leaves a corrupt "final" file
    return result


# ---------------------------------------------------------------------------
# Sweep orchestration.
# ---------------------------------------------------------------------------

def run_manifest(manifest: list, results_dir: str, device: str,
                 require_pi_signoff: bool = True, steps_override: int | None = None) -> list:
    """S1.6/S1.7 gate 1(a) Rev 7: each cell trains at its OWN group's pinned
    step budget (`cell["steps"]`) unless `steps_override` is given (smoke/
    testing convenience -- a uniform tiny step count for every cell)."""
    if require_pi_signoff and os.environ.get(PI_SIGNOFF_VAR) != "1":
        raise RuntimeError(
            f"{PI_SIGNOFF_VAR}=1 required before any GPU cell (S1.7 gate 5). "
            f"Set it explicitly to launch; this is NOT set automatically by this script."
        )
    results = []
    for cell in manifest:
        results.append(run_cell_resume_safe(cell, results_dir, device, steps_override=steps_override))
    return results


def run_calibration_only(args) -> dict:
    """S1.22 BA-F1's `--calibration-only` step: run the 5 calibration cells
    to completion (each at its OWN group's Rev-7-pinned step budget unless
    `--steps` overrides uniformly) and write the measured real per-cell rate
    to disk -- the precondition `--sweep` gates on below."""
    manifest = calibration_wave()
    steps_override = args.steps   # None (per-group STEP_BUDGET pins) unless the CLI overrides
    print(f"[calibration-only] launching {len(manifest)} calibration cells "
         f"(steps={'per-group STEP_BUDGET pins' if steps_override is None else steps_override}, "
         f"device={args.device}, results_dir={args.results_dir})")
    results = run_manifest(manifest, args.results_dir, args.device, steps_override=steps_override)
    report = write_calibration_report(args.results_dir, results)
    print(f"[calibration-only] wrote calibration report: real_rate_per_cell="
         f"{report['real_rate_per_cell']:.4f} GPU-h/cell, steps_per_group={report['steps_per_group']} "
         f"-> {calibration_report_path(args.results_dir)}")
    return report


def run_sweep(args) -> tuple:
    """S1.22 BA-F1's `--sweep` step: gated launch. Raises BaseSweepAbort (via
    gate_sweep_launch -> budget_guard.check_base_sweep_projection) if the
    calibration-measured rate would blow the 30 GPU-h cap, and RuntimeError
    if the calibration report is missing/invalid/stale or the PI-signoff
    token isn't set -- BEFORE building or launching the 58-cell manifest."""
    expected = {name: args.steps for name in GROUP_NAMES} if args.steps is not None else STEP_BUDGET
    gate = gate_sweep_launch(args.results_dir, expected)
    print(f"[sweep] gate PASSED: measured rate {gate['calibration_report']['real_rate_per_cell']:.4f} "
         f"GPU-h/cell -> base-sweep projection {gate['projection']['projected_total']:.2f} GPU-h "
         f"(cap {bg.CAP:.1f})")
    manifest = build_sweep_manifest()
    print(f"[sweep] launching {len(manifest)} cells (steps={'per-group STEP_BUDGET pins' if args.steps is None else args.steps}, "
         f"device={args.device}, results_dir={args.results_dir})")
    results = run_manifest(manifest, args.results_dir, args.device, steps_override=args.steps)
    guard = bg.BudgetGuard(cap=bg.CAP, spend_to_date=gate["projection"]["projected_total"])
    esc_log = run_escalations(results, args.results_dir, args.device, guard, steps_override=args.steps)
    return results, esc_log


def group_results_by(results: list, group: str, arm: str) -> np.ndarray:
    return np.array([r["restricted_effective_rank"] for r in results
                     if r["group"] == group and r["arm"] == arm])


def run_escalations(base_results: list, results_dir: str, device: str,
                    guard: bg.BudgetGuard, steps_override: int | None = None) -> dict:
    """S1.4.2/S1.6/S1.20's escalation pass: evaluate ambiguity triggers off
    the base-sweep results, run the marquee TOST check (tost_analysis.py,
    wired to `guard`), and any GENERAL escalation-to-5 triggers, granting
    through the guard in the pinned order. Escalation-triggered cells that
    ARE granted get extra seeds trained via run_cell_resume_safe (resume-safe,
    same as the base sweep, each at its OWN group's pinned step budget
    unless `steps_override`); denied ones are recorded with status
    'budget-denied' and NOT trained."""
    s4_unconstrained = group_results_by(base_results, "S4", "unconstrained")
    a5_unconstrained = group_results_by(base_results, "A5", "unconstrained")
    marquee = tost.marquee_check(s4_unconstrained, a5_unconstrained, budget_guard=guard)

    escalation_log = dict(marquee=marquee, general=[])
    if marquee["escalation_status"] == "granted":
        for name in ("S4", "A5"):
            for arm in ("unconstrained", "k_dmin"):
                for seed in range(bg.MARQUEE_N5, bg.MARQUEE_N7):
                    cell = dict(cell_id=f"{name}__{arm}__seed{seed}", group=name, arm=arm, seed=seed,
                               steps=STEP_BUDGET[name], is_calibration=False)
                    run_cell_resume_safe(cell, results_dir, device, steps_override=steps_override)

    # general escalation-to-5: this build-stage wiring exposes the mechanism;
    # ambiguity detection itself (per-cell-type CI-straddles-bar, S1.4.2) is
    # computed by the caller from base_results and passed in as a set of
    # (group, cell_type) tuples that are eligible AND ambiguous -- left to
    # the harvest/analysis stage (not re-derived here) since it depends on
    # each cell type's own M1/M3 CI, which is downstream analysis, not part
    # of the budget-arbitration mechanism itself.
    return escalation_log


# ---------------------------------------------------------------------------
# S1.33 M3 FIX WAVE -- CAPABILITY_SEPARATION_DESIGN.md S1.33's D-AMB
# (ambient-identity capacity-tax) diagnosis voided every force-rank arm in
# the base 58-cell sweep: groups.py:157-158's eye()-padded target has rank
# d_state (ALL singular values 1), so a rank-k-capped arm's best achievable
# direct cosine is EXACTLY sqrt(k/d_state) -- 37/39 real force-rank cells
# sat within 0.07 of that ceiling, and the causal CONFIRM half of M3
# (recovery returning once the capped rank reaches d_min) was never
# actually purchased. TWO pre-registered fix variants (S1.33's own text),
# BOTH built here (the DEPLOY-stage choice of which to launch, or both, is
# the coordinator's decision, not made by this build):
#
#   "zero_pad"     (groups.py target_padding="zero") -- the as-built target
#                   has rank EXACTLY d_min (no tax); the ORIGINAL untaxed
#                   grid {d_min-1, d_min, d_min+1} is now the causally-
#                   correct straddle. Because the target FAMILY changes, a
#                   fresh 5-cell unconstrained anchor (1/group,
#                   target_padding="zero") re-verifies M1's clean
#                   rank<->d_min tracking still holds before trusting the
#                   capped arms (S1.33's own text: "requires re-checking
#                   gate-1(b)'s injection figures since the target family
#                   changes" -- this anchor is the cheap empirical version
#                   of that re-check, at BUILD-stage scope).
#   "tax_adjusted" (groups.py target_padding="eye", UNCHANGED) -- no target-
#                   family change, no gate re-validation needed (S1.33's
#                   "minimal-delta option"); the grid shifts UP by the
#                   2-dim ambient tax (D_STATE = D_MIN + 2 always,
#                   groups.py's uniform-margin rule) to raw k in
#                   {d_min+1, d_min+2}, delivering EFFECTIVE rho-rank
#                   {d_min-1, d_min} once the tax is paid first (D-AMB P3:
#                   capped arms buy the constant I_2 block before the
#                   genuine rho signal). REACHABILITY CONSTRAINT (found by
#                   this build's own smoke, not asserted away): the THIRD
#                   causal point (effective d_min+1) would need raw k =
#                   d_min+3 = d_state+1 -- IMPOSSIBLE, since force_rank_k
#                   can never exceed the matrix's own dimension d_state.
#                   Variant B is therefore a NECESSARY 2-point grid, not 3
#                   -- a genuine, disclosed structural asymmetry against
#                   variant A (which has 1 spare dimension of margin and
#                   fits its full 3-point straddle). No anchor needed for
#                   variant B -- the ORIGINAL sweep's unconstrained-arm M1
#                   numbers already cover this variant's (unchanged)
#                   target family.
# ---------------------------------------------------------------------------

M3FIX_VARIANT_A = "zero_pad"
M3FIX_VARIANT_B = "tax_adjusted"
M3FIX_VARIANTS = (M3FIX_VARIANT_A, M3FIX_VARIANT_B)

# D_STATE = D_MIN + 2 always (groups.py's uniform-margin rule) -- duplicated
# here as an independent literal (not imported/derived) so the variant-B
# tax-arithmetic teeth below catch a drift in groups.py's OWN margin rule
# too, not just a bug local to this module's grid formula.
M3FIX_AMBIENT_TAX = 2

# S1.33's own "per-cell budget from the measured Rev-7 rates" mandate: the
# realized Rev-7 sweep rate (EXPERIMENT_LOG.md / CAPABILITY_SEPARATION_DESIGN.md
# S1.33 harvest record) -- 2.5907 GPU-h realized across the 58-cell sweep's
# 1,120,000 step-cells (sum over the 5 groups of STEP_BUDGET[group] *
# n_cells_per_group(group): 10*8000 + 14*20000 + 14*20000 + 10*8000 +
# 10*40000 = 1,120,000). NOT re-measured here -- no new calibration run,
# per this build's own BUILD-not-LAUNCH scope; used only for the printed
# budget projection below.
M3FIX_RATE_PER_STEP_GPU_H = 2.5907 / 1_120_000   # ~= 2.3131e-6 GPU-h/step


def m3fix_force_rank_grid(name: str, variant: str) -> dict:
    """S1.33's two pre-registered fix grids, keyed by the SAME causal
    labels the original (voided) M3 grid used -- each label names the
    EFFECTIVE rho-rank position (d_min-1/d_min/[d_min+1]) the cell tests,
    not the raw `force_rank_k` passed to the model (which differs between
    variants: variant A's raw k IS the effective rho-rank; variant B's raw
    k is the effective rho-rank PLUS the 2-dim ambient tax).

    REACHABILITY CONSTRAINT (found by this build's OWN smoke -- a real
    AssertionError on first run, not asserted away): D_STATE = D_MIN + 2
    always (groups.py's uniform-margin rule), so `force_rank_k` can never
    exceed `d_min+2` (a matrix cannot be truncated to a rank above its own
    dimension). Variant A (zero_pad, no tax) fits its full 3-point straddle
    {d_min-1, d_min, d_min+1} comfortably inside [1, d_min+2] (1 spare
    dimension of margin at the top). Variant B (tax_adjusted) pays that
    SAME 2-dim margin as its tax, so its raw-k straddle would need to reach
    d_min+3 = d_state+1 to test the "+1" causal point -- IMPOSSIBLE.
    Variant B's grid is therefore NECESSARILY 2 points, not 3:
    "k_dmin_minus_1" (raw k=d_min+1, effective d_min-1, the FALSIFY
    boundary) and "k_dmin" (raw k=d_min+2=d_state, effective d_min, the
    CONFIRM point -- note this raw k is numerically IDENTICAL to
    "unconstrained": truncating a d_state x d_state matrix to rank d_state
    is a no-op). There is NO raw k under eye-padding that ever delivers
    effective rank d_min+1 -- a genuine, disclosed STRUCTURAL asymmetry
    between the two fix variants (variant A can test 1 unit of margin
    above d_min; variant B categorically cannot), not an implementation
    gap, flagged prominently for the audit and for §1.33's own record."""
    assert variant in M3FIX_VARIANTS, f"unknown m3fix variant {variant!r}"
    d_min = D_MIN[name]
    d_state = D_STATE[name]
    if variant == M3FIX_VARIANT_A:
        grid = {"k_dmin_minus_1": d_min - 1, "k_dmin": d_min, "k_dmin_plus_1": d_min + 1}
    else:
        grid = {"k_dmin_minus_1": (d_min - 1) + M3FIX_AMBIENT_TAX,
                "k_dmin": d_min + M3FIX_AMBIENT_TAX}
    for label, k in grid.items():
        assert 1 <= k <= d_state, f"{name}/{variant}/{label}: k={k} outside [1,{d_state}]"
    return grid


def m3fix_target_padding(variant: str) -> str:
    assert variant in M3FIX_VARIANTS, f"unknown m3fix variant {variant!r}"
    return "zero" if variant == M3FIX_VARIANT_A else "eye"


def build_m3fix_manifest(seed: int = 0) -> list:
    """S1.33's fix-wave manifest, BOTH variants, n=1 seed/cell PER CALL -- a
    single-seed force-rank step is the established convention for this
    class of causal diagnostic (EXPERIMENT_LOG.md's own Task D M3
    precedent: "force-rank {1,2,3}->0.0 recovery, force-rank 4->0.97.
    Razor-sharp.").

    Per group: variant A (zero_pad) contributes 4 cells (its full 3-point
    grid + the 1-cell unconstrained anchor); variant B (tax_adjusted)
    contributes only 2 cells (its REACHABILITY-CONSTRAINED 2-point grid --
    see m3fix_force_rank_grid's docstring: the "+1" causal point is
    mathematically unreachable under eye-padding for every group, since
    D_STATE = D_MIN + 2 exactly consumes the tax as its own margin). 5
    groups * (4+2) = 30 cells total -- closely matching S1.33's own rough
    "~28 cells" estimate. Priced (`price_m3fix_manifest` below) at
    ~1.33 GPU-h, inside S1.33's own registered 1.3-2.6 GPU-h budget
    window.

    `seed` (S1.36 S3 SEED-PARAMETERIZATION EXTENSION build item #1,
    2026-07-09): threaded into EVERY cell's `seed=` field AND its `cell_id`
    f-string (`...__seed{seed}`), so a non-default call produces a manifest
    of cells whose seed differs from the original single-seed fix wave's --
    used by the pre-registered 3-seed S3 extension (§1.36) to train
    genuinely different-seeded replicates rather than resume-skip-colliding
    with the seed=0 cells already on disk (`is_valid_output` keys off
    `cell_id`, so a seed-less cell_id would silently alias). Default `seed=0`
    reproduces the ORIGINAL manifest byte-identically (same cell_ids, same
    seed field) -- verified against the independent-literal pin in
    `_test_m3fix_manifest_literal_pin` and the seed-threading teeth in
    `_test_m3fix_seed_parameterization`."""
    manifest = []
    for name in GROUP_NAMES:
        # variant A: 1-cell unconstrained anchor (re-verifies M1 tracking
        # under the new zero-padded target family) + the full 3-point grid.
        manifest.append(dict(
            cell_id=f"{M3FIX_VARIANT_A}__{name}__unconstrained__seed{seed}",
            group=name, arm="unconstrained", variant=M3FIX_VARIANT_A, seed=seed,
            steps=STEP_BUDGET[name], force_rank_k=None,
            target_padding=m3fix_target_padding(M3FIX_VARIANT_A), is_anchor=True,
        ))
        grid_a = m3fix_force_rank_grid(name, M3FIX_VARIANT_A)
        for arm_label, k in grid_a.items():
            manifest.append(dict(
                cell_id=f"{M3FIX_VARIANT_A}__{name}__{arm_label}__seed{seed}",
                group=name, arm=arm_label, variant=M3FIX_VARIANT_A, seed=seed,
                steps=STEP_BUDGET[name], force_rank_k=k,
                target_padding=m3fix_target_padding(M3FIX_VARIANT_A), is_anchor=False,
            ))
        # variant B: the reachability-constrained 2-point grid only, no
        # anchor (target family unchanged -- the original sweep's own
        # unconstrained-arm M1 numbers already cover this variant).
        grid_b = m3fix_force_rank_grid(name, M3FIX_VARIANT_B)
        for arm_label, k in grid_b.items():
            manifest.append(dict(
                cell_id=f"{M3FIX_VARIANT_B}__{name}__{arm_label}__seed{seed}",
                group=name, arm=arm_label, variant=M3FIX_VARIANT_B, seed=seed,
                steps=STEP_BUDGET[name], force_rank_k=k,
                target_padding=m3fix_target_padding(M3FIX_VARIANT_B), is_anchor=False,
            ))
    expected_n = len(GROUP_NAMES) * (4 + 2)   # variant A: 1 anchor + 3 grid; variant B: 2 grid
    assert len(manifest) == expected_n == 30, (
        f"S1.33 m3fix manifest: expected 30 cells (5 groups x [4 variant-A + 2 variant-B]), got "
        f"{len(manifest)}"
    )
    assert len({c["cell_id"] for c in manifest}) == len(manifest), "m3fix manifest has duplicate cell_ids"
    assert all(c["seed"] == seed for c in manifest), "seed field did not thread onto every cell"
    return manifest


def filter_m3fix_manifest(manifest: list, groups: list | None = None) -> list:
    """S1.36 S3 SEED-PARAMETERIZATION EXTENSION build item #2: restrict an
    m3fix manifest to the requested `groups` only (subset of GROUP_NAMES),
    preserving manifest order. `groups=None` or `[]` is a no-op (returns
    `manifest` unchanged, i.e. "all groups" -- the pre-extension default
    behavior). Used to launch a seed extension for JUST the group whose
    marginality trigger fired (S3: 4 variant-A + 2 variant-B = 6 cells)
    instead of re-running the full 30-cell both-variant manifest at each
    new seed. Rejects unknown group names loudly (a typo'd --m3fix-groups
    value must not silently filter to an empty/wrong manifest)."""
    if not groups:
        return manifest
    unknown = set(groups) - set(GROUP_NAMES)
    assert not unknown, (
        f"filter_m3fix_manifest: unknown group(s) {sorted(unknown)} -- expected a subset of "
        f"{list(GROUP_NAMES)}"
    )
    return [c for c in manifest if c["group"] in groups]


def _parse_m3fix_groups(groups_arg) -> list | None:
    """Parses `--m3fix-groups`' comma-separated CLI string (e.g. "S3" or
    "S3,A6") into a list, or `None` for "all groups" (the default/unset
    case). Also accepts an already-parsed list/tuple directly (so smoke's
    args stand-ins can pass one without round-tripping through a string)."""
    if groups_arg is None:
        return None
    if isinstance(groups_arg, (list, tuple)):
        parsed = list(groups_arg)
    else:
        parsed = [g.strip() for g in str(groups_arg).split(",") if g.strip()]
    return parsed or None


def price_m3fix_manifest(manifest: list, rate_per_step: float = M3FIX_RATE_PER_STEP_GPU_H) -> dict:
    """Prices the m3fix manifest from the cell's OWN group step count
    (`cell["steps"]`, the Rev-7 STEP_BUDGET pin) times the measured Rev-7
    per-step rate -- "per-cell budget from the measured Rev-7 rates" per
    this build's own mandate, not a fresh calibration run."""
    total_steps = sum(c["steps"] for c in manifest)
    total_gpu_h = total_steps * rate_per_step
    by_variant: dict = {}
    for c in manifest:
        v = by_variant.setdefault(c["variant"], {"n_cells": 0, "steps": 0})
        v["n_cells"] += 1
        v["steps"] += c["steps"]
    for v in by_variant.values():
        v["gpu_h"] = v["steps"] * rate_per_step
    return dict(n_cells=len(manifest), total_steps=total_steps, total_gpu_h=total_gpu_h,
                by_variant=by_variant, rate_per_step=rate_per_step)


def run_m3fix(args) -> list:
    """S1.33 --m3fix: build the BOTH-variant fix-wave manifest, print its
    priced budget (measured Rev-7 rate, no new calibration run), and launch
    resume-safe -- gated on the SAME PI-signoff token as --sweep (S1.7 gate
    5), since it launches real GPU cells.

    S1.36 S3 SEED-PARAMETERIZATION EXTENSION: `args.m3fix_seed` (default 0,
    byte-identical legacy manifest) threads through to
    `build_m3fix_manifest`; `args.m3fix_groups` (default None = all groups)
    threads through `_parse_m3fix_groups` to `filter_m3fix_manifest`, so a
    seed-extension launch can target e.g. just S3's 6 cells at seed=1/2/3
    instead of the full 30-cell both-variant manifest.

    BUILD-STAGE SCOPE: this function is fully wired and unit-exercised by
    `smoke()` below on a tiny CPU slice; it is NOT invoked by any GPU launch
    in this build pass (build, not launch; independent audit follows, per
    this build agent's own mandate)."""
    seed = args.m3fix_seed
    groups = _parse_m3fix_groups(args.m3fix_groups)
    manifest = build_m3fix_manifest(seed=seed)
    manifest = filter_m3fix_manifest(manifest, groups=groups)
    price = price_m3fix_manifest(manifest)
    print(f"[m3fix] seed={seed} groups={groups or 'ALL'} -- {price['n_cells']} cells, "
         f"{price['total_steps']} step-cells, projected {price['total_gpu_h']:.4f} GPU-h "
         f"(rate={price['rate_per_step']:.4e} GPU-h/step, sourced from the Rev-7 harvest "
         f"realized rate, S1.33)")
    for v, stats in price["by_variant"].items():
        print(f"  [{v}] {stats['n_cells']} cells, {stats['steps']} step-cells, "
             f"{stats['gpu_h']:.4f} GPU-h")
    if os.environ.get(PI_SIGNOFF_VAR) != "1":
        raise RuntimeError(f"{PI_SIGNOFF_VAR}=1 required before any GPU cell (S1.7 gate 5).")
    results = run_manifest(manifest, args.results_dir, args.device, steps_override=args.steps)
    return results


def _test_m3fix_variant_b_grid_arithmetic():
    """S1.33 TEETH #3 -- variant B (tax_adjusted, eye-padding UNCHANGED)
    per-group arithmetic, both directions: (1) the RAW `force_rank_k` grid
    must be EXACTLY {d_min+1, d_min+2} (TWO points, not three -- the
    REACHABILITY CONSTRAINT this build's own smoke caught: raw k can never
    exceed d_state=d_min+2, so the "+1" causal point, which would need raw
    k=d_min+3=d_state+1, is structurally unreachable and must NOT appear)
    -- checked against an INDEPENDENT literal `D_MIN` dict (not the
    imported `D_MIN` from groups.py, so a drift in groups.py's OWN table
    would be caught, not silently propagated); (2) because eye-padding
    pays the 2-dim ambient identity block FIRST (D-AMB P3, S1.33), the
    resulting EFFECTIVE rho-rank (raw k minus the independently-literal-
    pinned 2-dim tax) must land EXACTLY on {d_min-1, d_min} -- the SAME
    two lower causal points variant A's zero-padded grid also tests,
    proving both variants exercise the IDENTICAL causal claim over the
    range where a comparison is even reachable; (3) a REGRESSION check
    that a 3rd ("k_dmin_plus_1") entry does NOT reappear in variant B's
    grid."""
    print("=" * 88)
    print("S1.33 TEETH #3 -- m3fix variant B (tax_adjusted): raw-k grid + k-2 effective-rank "
          "arithmetic, per group (2-point reachability-constrained grid)")
    print("=" * 88)
    D_MIN_LITERAL = {"S3": 2, "S4": 3, "A5": 3, "S5": 4, "A6": 5}   # independent of groups.D_MIN
    AMBIENT_TAX_LITERAL = 2                                        # independent of M3FIX_AMBIENT_TAX
    assert D_MIN_LITERAL == D_MIN, (
        f"D_MIN_LITERAL pin mismatch: {D_MIN_LITERAL} != groups.D_MIN {dict(D_MIN)} -- "
        f"S1.4's per-group d_min table has drifted"
    )
    for name in GROUP_NAMES:
        d_min = D_MIN_LITERAL[name]
        grid = m3fix_force_rank_grid(name, M3FIX_VARIANT_B)
        assert "k_dmin_plus_1" not in grid, (
            f"{name}: variant-B grid contains 'k_dmin_plus_1' -- this causal point is "
            f"mathematically unreachable under eye-padding (raw k would need to exceed d_state); "
            f"its reappearance is a regression, not progress (S1.33 TEETH #3)"
        )
        raw_ks = sorted(grid.values())
        expected_raw = [d_min + 1, d_min + 2]
        assert raw_ks == expected_raw, (
            f"{name}: variant-B raw k grid {raw_ks} != expected {expected_raw} (S1.33 TEETH #3)"
        )
        d_state = D_STATE[name]
        assert max(raw_ks) == d_state, (
            f"{name}: variant-B's highest raw k ({max(raw_ks)}) should equal d_state ({d_state}) "
            f"-- it is the CEILING, one raw-k unit shy of what an unreachable '+1' point would need"
        )
        effective_rho_ranks = sorted(k - AMBIENT_TAX_LITERAL for k in raw_ks)
        expected_effective = [d_min - 1, d_min]
        assert effective_rho_ranks == expected_effective, (
            f"{name}: variant-B effective rho-rank {effective_rho_ranks} != expected "
            f"{expected_effective} (k-2 tax arithmetic, S1.33 TEETH #3)"
        )
        # cross-check: variant A's own (untaxed) grid's lower two points ARE the SAME
        # effective-rank straddle, directly (variant A additionally reaches a 3rd point
        # variant B categorically cannot).
        grid_a_lower = sorted(v for k_label, v in m3fix_force_rank_grid(name, M3FIX_VARIANT_A).items()
                              if k_label != "k_dmin_plus_1")
        assert grid_a_lower == expected_effective, (
            f"{name}: variant-A's lower grid points {grid_a_lower} != the shared causal straddle "
            f"{expected_effective} variant-B's effective rank is supposed to match"
        )
        print(f"  [{name}] variant B: raw k grid={raw_ks} (ceiling=d_state={d_state})  "
              f"effective rho-rank (k-2)={effective_rho_ranks}  == variant-A's lower grid "
              f"{grid_a_lower}  no 'k_dmin_plus_1' present  PASS")
    print("\nRESULT: variant-B's 2-point raw-k grid and k-2 effective-rank arithmetic verified "
          "per group; the unreachable 3rd point is confirmed absent, not silently dropped.\n")
    return True


def _test_m3fix_manifest_literal_pin():
    """S1.33 TEETH #4 -- manifest teeth, independent-literal assert of the
    cell list (the STEP_BUDGET-style pin per the audit precedent 27c97a1:
    self-referential assertions kill zero mutants -- pin decision constants
    with independent duplicated literals, assert observable properties of
    drawn data, not just downstream pass/fail). The expected cell_id SET
    below is constructed from LITERAL group/arm-label lists typed directly
    in this test (not imported from GROUP_NAMES / m3fix_force_rank_grid's
    own arm-label dict), so a corruption of either the group roster or the
    per-variant arm-shape in the production code is caught, not silently
    mirrored on both sides of the comparison."""
    print("=" * 88)
    print("S1.33 TEETH #4 -- m3fix manifest independent-literal pin (cell_id set, per-variant "
          "counts, per-group steps)")
    print("=" * 88)
    GROUPS_LITERAL = ["S3", "S4", "A5", "S5", "A6"]                       # independent of GROUP_NAMES
    ZERO_PAD_ARMS_LITERAL = ["k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"]  # variant A: full 3-point grid
    TAX_ADJUSTED_ARMS_LITERAL = ["k_dmin_minus_1", "k_dmin"]               # variant B: reachability-
                                                                            # constrained 2-point grid
    STEP_BUDGET_LITERAL = {"S3": 8000, "S4": 20000, "A5": 20000, "S5": 8000, "A6": 40000}

    expected_ids = set()
    for g in GROUPS_LITERAL:
        expected_ids.add(f"zero_pad__{g}__unconstrained__seed0")
        for a in ZERO_PAD_ARMS_LITERAL:
            expected_ids.add(f"zero_pad__{g}__{a}__seed0")
        for a in TAX_ADJUSTED_ARMS_LITERAL:
            expected_ids.add(f"tax_adjusted__{g}__{a}__seed0")
    assert len(expected_ids) == 30, f"literal pin construction error: {len(expected_ids)} != 30"

    manifest = build_m3fix_manifest()
    actual_ids = {c["cell_id"] for c in manifest}
    assert actual_ids == expected_ids, (
        f"m3fix manifest cell_id set mismatch (S1.33 TEETH #4).\n"
        f"  missing from manifest: {sorted(expected_ids - actual_ids)}\n"
        f"  unexpected in manifest: {sorted(actual_ids - expected_ids)}"
    )
    print(f"  cell_id SET: {len(actual_ids)} cells == independent-literal-pinned {len(expected_ids)} "
          f"(5 groups x [1 anchor + 3 zero_pad grid + 2 tax_adjusted grid])  PASS")

    n_zero_pad = sum(1 for c in manifest if c["variant"] == "zero_pad")
    n_tax_adjusted = sum(1 for c in manifest if c["variant"] == "tax_adjusted")
    assert (n_zero_pad, n_tax_adjusted) == (20, 10), (
        f"per-variant cell count {(n_zero_pad, n_tax_adjusted)} != expected (20, 10)"
    )
    print(f"  per-variant counts: zero_pad={n_zero_pad} (expect 20)  "
          f"tax_adjusted={n_tax_adjusted} (expect 10)  PASS")

    assert all(c["steps"] == STEP_BUDGET_LITERAL[c["group"]] for c in manifest), (
        "m3fix manifest did not thread the Rev-7 STEP_BUDGET pins onto every cell "
        "(independent-literal check)"
    )
    print(f"  per-cell steps match the independent-literal STEP_BUDGET pin "
          f"{STEP_BUDGET_LITERAL}  PASS")

    n_anchors = sum(1 for c in manifest if c.get("is_anchor"))
    assert n_anchors == 5, f"expected exactly 5 anchor cells, got {n_anchors}"
    assert all(c["variant"] == "zero_pad" for c in manifest if c.get("is_anchor")), \
        "an anchor cell was found outside variant zero_pad"
    print(f"  anchor cells: {n_anchors} (expect 5, all variant=zero_pad)  PASS")

    print("\nRESULT: m3fix manifest cell list, per-variant counts, per-group steps, and anchor "
          "placement all verified against INDEPENDENT literals (not self-referential).\n")
    return True


def _test_m3fix_seed_parameterization():
    """S1.36 S3 SEED-PARAMETERIZATION EXTENSION TEETH #1 -- build_m3fix_manifest(seed=N)
    threading, both directions: (1) seed=0 (the default, and an explicit
    seed=0 call) reproduces the ORIGINAL manifest byte-identically (same
    cell_id LIST, in order); (2) for seed in {1,2,3}, EVERY cell's cell_id
    ends with the literal substring `__seed{seed}` AND `cell["seed"] ==
    seed`, checked INDEPENDENTLY of each other -- the prior extension
    attempt's exact bug (agent report, S1.36) was `build_m3fix_manifest()`
    hardcoding `seed=0` baked into the cell_id f-strings with no seed
    threading at all; if the cell_id interpolation regresses back to a
    hardcoded "seed0" while the `seed=` dict field is still threaded
    correctly (or vice versa), THIS test must fail -- a mutation that only
    reverts one of the two checks below (e.g. dropping the f-string
    interpolation but leaving `seed=seed` in the dict) must NOT pass
    silently; (3) the resulting cell_id SET at seed=N is disjoint from
    seed=0's set (no accidental string-collision aliasing that would make
    `is_valid_output`'s resume-skip treat a new-seed cell as already done)."""
    print("=" * 88)
    print("S1.36 S3 SEED-PARAMETERIZATION TEETH #1 -- build_m3fix_manifest(seed=N) threading")
    print("=" * 88)

    # (1) seed=0 legacy-identical -- bare-default call vs an explicit seed=0 call, AND vs the
    # §1.34 independent-literal pin (re-derived here, not reused, so a drift in either this
    # test's own construction or _test_m3fix_manifest_literal_pin's is caught independently).
    m_default = build_m3fix_manifest()
    m_seed0 = build_m3fix_manifest(seed=0)
    ids_default = [c["cell_id"] for c in m_default]
    ids_seed0 = [c["cell_id"] for c in m_seed0]
    assert ids_default == ids_seed0, (
        "build_m3fix_manifest(seed=0) diverges from the bare-default call -- the default must "
        "be seed=0 (legacy byte-identical behavior)"
    )
    assert all("__seed0" in cid for cid in ids_seed0), "seed=0 cell_ids must all contain '__seed0'"
    assert all(c["seed"] == 0 for c in m_seed0), "seed=0 cells must all have seed field == 0"
    print(f"  seed=0 (default == explicit): {len(m_seed0)} cells, all cell_ids contain '__seed0' "
          f"and cell['seed']==0  PASS")

    # (2) mutation-catching: seed threads into BOTH cell_id AND cell["seed"], checked
    # independently -- neither check is a proxy for the other.
    ids_seed0_set = set(ids_seed0)
    for test_seed in (1, 2, 3):
        m = build_m3fix_manifest(seed=test_seed)
        assert len(m) == 30, f"seed={test_seed}: manifest length {len(m)} != 30"
        expected_suffix = f"__seed{test_seed}"
        cellid_ok = all(c["cell_id"].endswith(expected_suffix) for c in m)
        seedfield_ok = all(c["seed"] == test_seed for c in m)
        assert cellid_ok, (
            f"MUTATION CAUGHT (cell_id interpolation): at least one cell_id at seed={test_seed} "
            f"does not end with {expected_suffix!r} -- e.g. {[c['cell_id'] for c in m if not c['cell_id'].endswith(expected_suffix)][:3]} "
            f"-- the cell_id f-string interpolation has been dropped/hardcoded (this IS the "
            f"mislabel bug the prior extension attempt found)"
        )
        assert seedfield_ok, (
            f"MUTATION CAUGHT (seed field): at least one cell at seed={test_seed} has "
            f"seed != {test_seed} even though its cell_id may look correct (checked "
            f"INDEPENDENTLY of the cell_id assert above) -- "
            f"e.g. {[(c['cell_id'], c['seed']) for c in m if c['seed'] != test_seed][:3]}"
        )
        ids_n = {c["cell_id"] for c in m}
        assert ids_n.isdisjoint(ids_seed0_set), (
            f"seed={test_seed} cell_id set OVERLAPS seed=0's set -- resume-skip aliasing risk: "
            f"{sorted(ids_n & ids_seed0_set)[:3]}"
        )
        print(f"  seed={test_seed}: {len(m)} cells, every cell_id ends {expected_suffix!r} AND "
              f"cell['seed']=={test_seed} (checked independently), disjoint from seed=0's set  PASS")

    print("\nRESULT: build_m3fix_manifest(seed=N) threads N into both cell_id and cell['seed'] "
          "correctly for N in {0,1,2,3}; default N=0 stays byte-identical to the pre-"
          "parameterization manifest; no cross-seed cell_id aliasing.\n")
    return True


def _test_m3fix_groups_filter():
    """S1.36 S3 SEED-PARAMETERIZATION EXTENSION TEETH #2 -- filter_m3fix_manifest
    restricts the manifest to the requested groups only. S3 alone must
    yield EXACTLY 6 cells (4 variant-A: 1 anchor + 3-point grid; 2
    variant-B: 2-point grid) -- the exact per-seed cell count the S3
    extension launch uses (3 seeds x 6 cells = 18 cells total, per §1.36's
    routed trigger). `groups=None`/`[]` must be a no-op (all 30 cells,
    unchanged pre-extension behavior); an unknown group name must be
    rejected loudly, not silently filtered to empty/wrong."""
    print("=" * 88)
    print("S1.36 S3 SEED-PARAMETERIZATION TEETH #2 -- filter_m3fix_manifest group filter")
    print("=" * 88)
    full = build_m3fix_manifest(seed=0)
    assert len(full) == 30

    no_filter = filter_m3fix_manifest(full, groups=None)
    empty_filter = filter_m3fix_manifest(full, groups=[])
    assert no_filter == full, "filter_m3fix_manifest(groups=None) must be a no-op"
    assert empty_filter == full, "filter_m3fix_manifest(groups=[]) must be a no-op"
    print(f"  groups=None/[] no-op: {len(no_filter)}/{len(empty_filter)} cells (expect 30/30)  PASS")

    s3_only = filter_m3fix_manifest(full, groups=["S3"])
    assert len(s3_only) == 6, f"S3-only filter: expected 6 cells, got {len(s3_only)}"
    assert all(c["group"] == "S3" for c in s3_only)
    n_a = sum(1 for c in s3_only if c["variant"] == M3FIX_VARIANT_A)
    n_b = sum(1 for c in s3_only if c["variant"] == M3FIX_VARIANT_B)
    assert (n_a, n_b) == (4, 2), f"S3-only variant split {(n_a, n_b)} != expected (4, 2)"
    print(f"  groups=['S3']: {len(s3_only)} cells (4 variant-A + 2 variant-B, matches §1.36's "
          f"per-seed S3 cell count)  PASS")

    two_groups = filter_m3fix_manifest(full, groups=["S3", "A6"])
    assert len(two_groups) == 12, f"S3+A6 filter: expected 12 cells, got {len(two_groups)}"
    assert {c["group"] for c in two_groups} == {"S3", "A6"}
    print(f"  groups=['S3','A6']: {len(two_groups)} cells (6+6)  PASS")

    # _parse_m3fix_groups: comma-string parsing + list passthrough + None/empty -> None.
    assert _parse_m3fix_groups("S3") == ["S3"]
    assert _parse_m3fix_groups("S3,A6") == ["S3", "A6"]
    assert _parse_m3fix_groups(" S3 , A6 ") == ["S3", "A6"]
    assert _parse_m3fix_groups(None) is None
    assert _parse_m3fix_groups("") is None
    assert _parse_m3fix_groups(["S3", "A6"]) == ["S3", "A6"]
    print("  _parse_m3fix_groups: comma-string/list/None parsing all verified  PASS")

    raised = False
    try:
        filter_m3fix_manifest(full, groups=["NOPE"])
    except AssertionError as e:
        raised = True
        print(f"  unknown-group rejection: {e}")
    assert raised, "filter_m3fix_manifest did not reject an unknown group name"

    print("\nRESULT: filter_m3fix_manifest correctly restricts to requested groups (S3-only = 6 "
          "cells) and rejects unknown group names; _parse_m3fix_groups handles comma-strings, "
          "lists, and None/empty correctly.\n")
    return True


# ---------------------------------------------------------------------------
# --smoke mode: exercises the FULL wiring (manifest, resume-safety,
# budget-guard, TOST) on a tiny handful of cells / steps, on CPU.
# ---------------------------------------------------------------------------

def _test_gate1a_check_bar_and_margin():
    """NEGATIVE-then-POSITIVE TEST (S1.7 gate 1(a) Rev 7, S1.30 item 1) --
    a synthetic gate check at the new L in {2..5} bar + >=0.02 margin rule,
    run to completion (no GPU/model needed -- gate1a_check is a pure
    function): (1) a bare-clearance profile (min L2-5 = 0.9023, matching
    this round's own A6@20K box value) must FAIL the margin rule even
    though it numerically exceeds tau=0.9; (2) a profile with real margin
    (>=0.02) must PASS; (3) L=1 is excluded from the bar (a profile with a
    terrible L=1 but healthy L2-5 must still PASS -- the exact case S1.30's
    H-ENC fix is FOR)."""
    print("=" * 88)
    print("NEGATIVE-then-POSITIVE TEST -- gate1a_check: L in {2..5} bar + >=0.02 margin rule")
    print("=" * 88)
    # F2 (build-audit fix, 2026-07-09 overnight): pin GATE1A_L_RANGE's exact
    # membership with an INDEPENDENT duplicated literal -- every check below
    # calls gate1a_check with the (possibly-mutated) module constant as its
    # OWN default, so a shrink of GATE1A_L_RANGE itself would silently
    # relabel what "the bar" means without tripping anything upstream.
    assert tuple(GATE1A_L_RANGE) == (2, 3, 4, 5), (
        f"GATE1A_L_RANGE literal-pin mismatch (F2): got {tuple(GATE1A_L_RANGE)}, expected "
        f"(2, 3, 4, 5) -- S1.7 gate 1(a) Rev 7's L-range has shrunk or otherwise drifted"
    )
    print(f"  GATE1A_L_RANGE independent-literal pin (F2): {tuple(GATE1A_L_RANGE)} == (2, 3, 4, 5)  OK")

    # (1) bare clearance (A6@20K's real box value) -- FAILS the margin rule.
    bare_profile = {1: 0.8410, 2: 0.9947, 3: 0.9914, 4: 0.9668, 5: 0.9023, 6: 0.90, 7: 0.85, 8: 0.80}
    bare = gate1a_check(bare_profile)
    print(f"  bare-clearance profile (A6@20K real box value): min_val={bare['min_val']:.4f} "
          f"at L={bare['at_L']}, margin_over_tau={bare['margin_over_tau']:.4f}, clears={bare['clears']}")
    assert bare["min_val"] > GATE1A_TAU, "test setup error: bare profile should exceed tau numerically"
    assert not bare["clears"], "gate1a_check FAILED to reject a bare (<0.02-margin) clearance"

    # (2) real margin (A6@40K's real box value) -- PASSES.
    healthy_profile = {1: 0.8509, 2: 0.9964, 3: 0.9959, 4: 0.9881, 5: 0.9633, 6: 0.91, 7: 0.87, 8: 0.81}
    healthy = gate1a_check(healthy_profile)
    print(f"  healthy-margin profile (A6@40K real box value): min_val={healthy['min_val']:.4f} "
          f"at L={healthy['at_L']}, margin_over_tau={healthy['margin_over_tau']:.4f}, clears={healthy['clears']}")
    assert healthy["clears"], "gate1a_check FAILED to accept a genuinely-clear (>=0.02-margin) profile"

    # (3) L=1 excluded from the bar -- a terrible L=1 with healthy L2-5 still PASSES.
    l1_excluded_profile = {1: 0.05, 2: 0.99, 3: 0.98, 4: 0.97, 5: 0.96, 6: 0.9, 7: 0.85, 8: 0.8}
    l1_excl = gate1a_check(l1_excluded_profile)
    print(f"  L=1-excluded profile (L=1 catastrophic, L2-5 healthy): clears={l1_excl['clears']} "
          f"(L=1 disclosed separately: {l1_excl['l1_disclosed']})")
    assert l1_excl["clears"], "gate1a_check incorrectly let a catastrophic L=1 value block the bar"
    assert l1_excl["l1_disclosed"] == 0.05, "L=1 should be disclosed, not silently dropped"

    # (4) F2 (build-audit fix): a profile whose MINIMUM sits AT L=2
    # specifically -- the build audit found mutation (c2) (L-range shrunk
    # to (3,4,5) or even (5,)) INVISIBLE, because every profile above has
    # its min at L>=5. This profile fails the margin rule (0.905 - 0.9 =
    # 0.005 < 0.02) ONLY if L=2 is actually a member of the active
    # l_range -- if L=2 were dropped, the remaining L in {3,4,5} (or just
    # {5}) all clear with margin 0.09, flipping `clears` to True and
    # tripping the assertion below.
    l2_min_profile = {2: 0.905, 3: 0.99, 4: 0.99, 5: 0.99}
    l2_check = gate1a_check(l2_min_profile)
    print(f"  L=2-min profile (F2 shrinkage tripwire): min_val={l2_check['min_val']:.4f} "
          f"at L={l2_check['at_L']}, margin_over_tau={l2_check['margin_over_tau']:.4f}, "
          f"clears={l2_check['clears']}")
    assert not l2_check["clears"], (
        "gate1a_check incorrectly CLEARED the L=2-min tripwire profile -- GATE1A_L_RANGE no "
        "longer includes L=2 (shrunk to (3,4,5) or narrower), the exact (c2) mutation"
    )

    print("\nRESULT: gate1a_check correctly enforces the L in {2..5}/>=0.02-margin bar and "
          "correctly excludes (but still discloses) L=1; GATE1A_L_RANGE membership independently "
          "pinned (F2).\n")
    return True


def _test_route_second_miss_diagnostic_before_action():
    """NEGATIVE-then-POSITIVE TEST (S1.7 gate 1(a) rule (c), Rev 7, S1.30
    item 4): (1) calling route_second_miss with NO diagnostic must raise --
    the diagnostic-before-action requirement has real teeth, not just a
    docstring; (2) each diagnostic class routes to its pre-registered
    outcome, including the cap-exhausted -> hard_stop edge case."""
    print("=" * 88)
    print("NEGATIVE-then-POSITIVE TEST -- route_second_miss: diagnostic-before-action + routing table")
    print("=" * 88)
    raised = False
    try:
        route_second_miss(None, escalations_used=1)
    except ValueError as e:
        raised = True
        print(f"  no-diagnostic call correctly REJECTED: {str(e)[:120]}...")
    assert raised, "route_second_miss did NOT enforce the mandatory-diagnostic precondition"

    checks = [
        ({"class": "instrument"}, 0, "instrument_fix"),
        ({"class": "moving_below_ceiling"}, 1, "one_capped_escalation"),   # this round's A6 case
        ({"class": "moving_below_ceiling"}, 2, "hard_stop"),               # cap exhausted, still moving
        ({"class": "ceiling"}, 1, "demote_disclose"),
        ({"class": "inconclusive"}, 1, "hard_stop"),
    ]
    for diag, used, expected in checks:
        route = route_second_miss(diag, escalations_used=used)
        print(f"  diagnostic={diag} escalations_used={used} -> route={route!r} (expect {expected!r})")
        assert route == expected, f"route_second_miss({diag}, {used}) = {route!r}, expected {expected!r}"
        assert route in GATE1A_ROUTES

    print("\nRESULT: route_second_miss correctly enforces diagnostic-before-action and routes "
          "every diagnostic class to its pre-registered outcome (incl. cap-exhausted -> hard_stop).\n")
    return True


def smoke():
    print("=" * 88)
    print("  run_capability_sep.py SMOKE -- manifest + resume-safety + escalation wiring")
    print("=" * 88)
    manifest = build_sweep_manifest()
    print(f"  build_sweep_manifest(): {len(manifest)} cells (expect 58)")
    assert len(manifest) == 58

    # F3 (build-audit fix, 2026-07-09 overnight): pin STEP_BUDGET's exact
    # per-group values with an INDEPENDENT duplicated literal. The check
    # right below this one (per-cell steps match STEP_BUDGET) is
    # self-referential -- build_sweep_manifest() reads STEP_BUDGET[name] to
    # populate cell["steps"] in the first place, so an S4<->S5 swap (or any
    # other corruption of STEP_BUDGET itself) is invisible to it: both sides
    # of that comparison would be mutated together.
    STEP_BUDGET_LITERAL = {"S3": 8000, "S4": 20000, "A5": 20000, "S5": 8000, "A6": 40000}
    assert STEP_BUDGET == STEP_BUDGET_LITERAL, (
        f"STEP_BUDGET literal-pin mismatch (F3): got {STEP_BUDGET}, expected "
        f"{STEP_BUDGET_LITERAL} -- S1.6/S1.7 gate 1(a) Rev 7's per-group step-budget pins have "
        f"drifted (e.g. an S4<->S5 swap)"
    )
    print(f"  STEP_BUDGET independent-literal pin (F3): {STEP_BUDGET} == {STEP_BUDGET_LITERAL}  OK")

    print(f"  per-cell steps match STEP_BUDGET pins: "
          f"{all(c['steps'] == STEP_BUDGET[c['group']] for c in manifest)}")
    assert all(c["steps"] == STEP_BUDGET[c["group"]] for c in manifest), \
        "build_sweep_manifest did not thread the Rev 7 STEP_BUDGET pins onto every cell"
    calib = calibration_wave()
    print(f"  calibration_wave(): {len(calib)} cells (expect 5, 1/group unconstrained seed=0)")
    assert len(calib) == 5
    assert {c["group"] for c in calib} == set(GROUP_NAMES)

    # Rev 7 pure-function negative/positive tests (S1.30 items 1/4) -- no
    # GPU/model needed, run first (cheap, and dependency-ordered before the
    # heavier CPU-model smoke below per this file's own convention).
    r_gate1a = _test_gate1a_check_bar_and_margin()
    r_route = _test_route_second_miss_diagnostic_before_action()
    assert r_gate1a and r_route

    # S1.33 M3 FIX WAVE pure-function teeth (#3/#4) -- no GPU/model needed,
    # same dependency-ordering convention (cheap, run before the heavier
    # CPU-model smoke below).
    r_m3fix_grid = _test_m3fix_variant_b_grid_arithmetic()
    r_m3fix_pin = _test_m3fix_manifest_literal_pin()
    assert r_m3fix_grid and r_m3fix_pin

    # S1.36 S3 seed-parameterization extension teeth (#1/#2) -- same
    # dependency-ordering convention (pure functions, no GPU/model needed).
    r_m3fix_seed = _test_m3fix_seed_parameterization()
    r_m3fix_groups = _test_m3fix_groups_filter()
    assert r_m3fix_seed and r_m3fix_groups

    import tempfile
    with tempfile.TemporaryDirectory() as results_dir:
        os.environ[PI_SIGNOFF_VAR] = "1"    # local smoke, not a real launch -- BUILD-scope only
        try:
            # a tiny slice (S4's unconstrained arm, all 5 seeds + one k_dmin seed, PLUS A5's
            # unconstrained arm, all 5 seeds -- so run_escalations() below has real data for
            # BOTH marquee groups) at a uniform steps_override=20 (CPU, smoke only -- overrides
            # every group's real STEP_BUDGET pin so the smoke stays fast regardless of group).
            tiny_manifest = [c for c in manifest if c["group"] == "S4" and c["arm"] == "unconstrained"]
            tiny_manifest += [c for c in manifest if c["group"] == "S4" and c["arm"] == "k_dmin"][:1]
            tiny_manifest += [c for c in manifest if c["group"] == "A5" and c["arm"] == "unconstrained"]
            print(f"\n  running a {len(tiny_manifest)}-cell tiny slice at steps_override=20 (CPU, smoke only):")
            t0 = time.time()
            results1 = run_manifest(tiny_manifest, results_dir, device="cpu", steps_override=20)
            print(f"  first pass: {len(results1)} results written in {time.time()-t0:.1f}s")
            for r in results1:
                assert os.path.exists(cell_output_path(results_dir, r["cell_id"]))
                assert is_valid_output(cell_output_path(results_dir, r["cell_id"]))
                assert "convergence_profile" in r and len(r["convergence_profile"]) == 8, \
                    "per-L convergence_profile missing/incomplete in cell output"
                assert "gate1a" in r and set(r["gate1a"]["l_range"]) == set(GATE1A_L_RANGE), \
                    "gate1a check missing/malformed in cell output"
                assert "l_ge2_mean_cos" in r, "L>=2 robustness split missing from cell output"

            # RESUME-SAFETY test: re-run the SAME manifest -- must SKIP every cell
            # (valid output already on disk), not re-train.
            t1 = time.time()
            results2 = run_manifest(tiny_manifest, results_dir, device="cpu", steps_override=20)
            dt2 = time.time() - t1
            print(f"  second pass (resume): {len(results2)} results in {dt2:.2f}s "
                  f"(should be near-instant -- all cells SKIPPED)")
            assert dt2 < (time.time() - t0), "resume pass took as long as the first -- resume-safety broken"
            for r1, r2 in zip(results1, results2):
                assert r1["cell_id"] == r2["cell_id"]
                assert r1["mean_cos"] == r2["mean_cos"], "resumed result differs from the original -- not truly skipped"

            # CORRUPTION test: a truncated/invalid JSON must NOT be treated as valid.
            corrupt_cell = tiny_manifest[0]
            corrupt_path = cell_output_path(results_dir, corrupt_cell["cell_id"])
            with open(corrupt_path, "w") as f:
                f.write("{not valid json")
            assert not is_valid_output(corrupt_path), "corrupted output was incorrectly treated as valid"
            print(f"  corruption test: a truncated JSON at {os.path.basename(corrupt_path)} correctly "
                  f"detected as INVALID (will be re-run, not skipped)")
            result3 = run_cell_resume_safe(corrupt_cell, results_dir, device="cpu", steps_override=20)
            assert is_valid_output(corrupt_path), "re-run after corruption did not produce valid output"
            print(f"  re-run after corruption: cell '{corrupt_cell['cell_id']}' correctly RE-TRAINED and "
                  f"now valid again")

            # escalation wiring: run_escalations with a real BudgetGuard.
            guard = bg.BudgetGuard(cap=bg.CAP, spend_to_date=0.0)
            esc_log = run_escalations(results1, results_dir, device="cpu", guard=guard, steps_override=20)
            print(f"\n  run_escalations() marquee verdict: {esc_log['marquee']['verdict']}  "
                  f"final_verdict: {esc_log['marquee']['final_verdict']}")
        finally:
            del os.environ[PI_SIGNOFF_VAR]

    # PI signoff gate must actually block without the token.
    try:
        run_manifest(manifest[:1], "/tmp/should_not_write", device="cpu", steps_override=1)
        raise AssertionError("run_manifest did NOT enforce the PI signoff gate")
    except RuntimeError as e:
        assert PI_SIGNOFF_VAR in str(e)
        print(f"\n  PI-signoff gate: run_manifest() correctly refuses without {PI_SIGNOFF_VAR}=1")

    # -------------------------------------------------------------------
    # S1.33 M3 FIX WAVE -- manifest/pricing sanity + a REAL tiny end-to-end
    # CPU run exercising the per-cell force_rank_k/target_padding OVERRIDE
    # path in train_and_eval_cell (not just the manifest's own shape).
    # -------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("  S1.33 m3fix smoke -- manifest/pricing sanity + tiny end-to-end CPU run (both variants)")
    print("=" * 88)
    m3fix_manifest = build_m3fix_manifest()
    print(f"  build_m3fix_manifest(): {len(m3fix_manifest)} cells (expect 30)")
    assert len(m3fix_manifest) == 30
    price = price_m3fix_manifest(m3fix_manifest)
    in_window = 1.3 <= price["total_gpu_h"] <= 2.6
    print(f"  price_m3fix_manifest(): {price['n_cells']} cells, {price['total_steps']} step-cells, "
         f"{price['total_gpu_h']:.4f} GPU-h projected (S1.33's registered 1.3-2.6 GPU-h window: "
         f"{'INSIDE' if in_window else 'OUTSIDE'})")
    assert in_window, (
        f"m3fix manifest priced at {price['total_gpu_h']:.4f} GPU-h, outside S1.33's registered "
        f"1.3-2.6 GPU-h budget window"
    )

    with tempfile.TemporaryDirectory() as m3fix_dir:
        os.environ[PI_SIGNOFF_VAR] = "1"    # local smoke, not a real launch -- BUILD-scope only
        try:
            # a tiny cross-variant slice, ONE group (S3): the zero_pad anchor + zero_pad k_dmin +
            # tax_adjusted k_dmin -- the SAME causal label ("k_dmin") under BOTH variants, which
            # must train at DIFFERENT raw force_rank_k values (the exact override case that would
            # silently collapse to the SAME wrong value if train_and_eval_cell's `"force_rank_k"
            # in cell` sentinel logic regressed to the base sweep's shared
            # force_rank_grid(name)[arm] lookup for m3fix cells too).
            tiny_m3fix = [c for c in m3fix_manifest if c["cell_id"] in (
                "zero_pad__S3__unconstrained__seed0",
                "zero_pad__S3__k_dmin__seed0",
                "tax_adjusted__S3__k_dmin__seed0",
            )]
            assert len(tiny_m3fix) == 3
            print(f"\n  running a {len(tiny_m3fix)}-cell m3fix cross-variant slice (group S3) at "
                  f"steps_override=15 (CPU, smoke only):")
            m3fix_results = run_manifest(tiny_m3fix, m3fix_dir, device="cpu", steps_override=15)
            by_id = {r["cell_id"]: r for r in m3fix_results}

            anchor = by_id["zero_pad__S3__unconstrained__seed0"]
            zp_k = by_id["zero_pad__S3__k_dmin__seed0"]
            ta_k = by_id["tax_adjusted__S3__k_dmin__seed0"]

            assert anchor["force_rank_k"] is None and anchor["target_padding"] == "zero", \
                "m3fix anchor cell did not train unconstrained/zero-padded as configured"
            assert zp_k["force_rank_k"] == D_MIN["S3"] and zp_k["target_padding"] == "zero"
            assert ta_k["force_rank_k"] == D_MIN["S3"] + M3FIX_AMBIENT_TAX and ta_k["target_padding"] == "eye"
            assert zp_k["force_rank_k"] != ta_k["force_rank_k"], (
                "zero_pad and tax_adjusted k_dmin cells trained at the SAME force_rank_k -- "
                "train_and_eval_cell's per-cell force_rank_k override is NOT being honored"
            )
            for r in m3fix_results:
                assert is_valid_output(cell_output_path(m3fix_dir, r["cell_id"]))
                assert r["m3fix_variant"] in M3FIX_VARIANTS
            print(f"  cross-variant override verified: zero_pad k_dmin force_rank_k="
                 f"{zp_k['force_rank_k']}  tax_adjusted k_dmin force_rank_k={ta_k['force_rank_k']} "
                 f"(differ, as required)  anchor force_rank_k={anchor['force_rank_k']} "
                 f"target_padding={anchor['target_padding']}")

            # RESUME-SAFETY for m3fix cells too (same machinery, new cell shape).
            t_r = time.time()
            m3fix_results2 = run_manifest(tiny_m3fix, m3fix_dir, device="cpu", steps_override=15)
            print(f"  m3fix resume pass: {len(m3fix_results2)} results in {time.time()-t_r:.2f}s "
                 f"(all cells SKIPPED)")
            for r1, r2 in zip(m3fix_results, m3fix_results2):
                assert r1["mean_cos"] == r2["mean_cos"], "resumed m3fix result differs -- not truly skipped"
        finally:
            del os.environ[PI_SIGNOFF_VAR]

    # PI-signoff gate must block run_m3fix too.
    class _M3FixArgs:
        results_dir = "/tmp/should_not_write_m3fix"
        device = "cpu"
        steps = 1
        m3fix_seed = 0
        m3fix_groups = None
    try:
        run_m3fix(_M3FixArgs())
        raise AssertionError("run_m3fix did NOT enforce the PI signoff gate")
    except RuntimeError as e:
        assert PI_SIGNOFF_VAR in str(e)
        print(f"  PI-signoff gate (m3fix path): run_m3fix() correctly refuses without "
             f"{PI_SIGNOFF_VAR}=1")

    # -------------------------------------------------------------------
    # S1.36 S3 seed-parameterization extension -- end-to-end run_m3fix() smoke: --m3fix-seed
    # + --m3fix-groups threaded together through the REAL CLI-facing entry point (not just
    # build_m3fix_manifest()/filter_m3fix_manifest() called directly above), the exact call
    # shape the S3 seed-extension launch uses (seed=N, groups="S3").
    # -------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("  S1.36 S3 seed-extension smoke -- run_m3fix() with --m3fix-seed + --m3fix-groups")
    print("=" * 88)
    with tempfile.TemporaryDirectory() as seedext_dir:
        os.environ[PI_SIGNOFF_VAR] = "1"
        try:
            class _SeedExtArgs:
                results_dir = seedext_dir
                device = "cpu"
                steps = 12
                m3fix_seed = 7
                m3fix_groups = "S3"
            seedext_results = run_m3fix(_SeedExtArgs())
            assert len(seedext_results) == 6, (
                f"run_m3fix(seed=7, groups='S3'): expected 6 results, got {len(seedext_results)}"
            )
            assert all(r["group"] == "S3" for r in seedext_results), \
                "run_m3fix groups filter leaked a non-S3 cell through"
            assert all(r["seed"] == 7 for r in seedext_results), \
                "run_m3fix seed did not thread through to the trained cell results"
            assert all(r["cell_id"].endswith("__seed7") for r in seedext_results), \
                "run_m3fix seed did not thread into the cell_id of the trained cell results"
            print(f"  run_m3fix(seed=7, groups='S3'): {len(seedext_results)} results, all "
                  f"group==S3, seed==7, cell_id ends '__seed7'  PASS")

            # resume-safety carries through the seed+groups path too: re-invoking must SKIP
            # every cell (same results_dir, same seed/groups) rather than re-train.
            t_r = time.time()
            seedext_results2 = run_m3fix(_SeedExtArgs())
            dt_r = time.time() - t_r
            for r1, r2 in zip(seedext_results, seedext_results2):
                assert r1["cell_id"] == r2["cell_id"] and r1["mean_cos"] == r2["mean_cos"], \
                    "resumed seed-extension result differs -- not truly skipped"
            print(f"  resume pass: {len(seedext_results2)} results in {dt_r:.2f}s (all SKIPPED)  PASS")
        finally:
            del os.environ[PI_SIGNOFF_VAR]

    # -------------------------------------------------------------------
    # S1.22 BA-F1 -- calibration report + --sweep gate wiring. Before this
    # fix, check_base_sweep_projection/BaseSweepAbort were dead code (never
    # called from anywhere in this file); the calibration-first gate did NOT
    # actually block the 53 remaining sweep cells. A synthetic OVER-RATE
    # calibration report must abort --sweep via BaseSweepAbort (negative
    # test, run to completion); a HEALTHY rate must pass (positive test).
    # Rev 7 extends this: `steps` is now per-group (STEP_BUDGET), so the
    # mismatch test below uses a DELIBERATELY WRONG per-group dict, not a
    # +1 scalar.
    # -------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("  BA-F1 smoke -- calibration report + --sweep gate (calibration-only -> gate -> sweep)")
    print("=" * 88)
    gate_steps = 20   # uniform smoke steps -- exercises the gate/report wiring, not the real pins
    with tempfile.TemporaryDirectory() as gate_dir:
        os.environ[PI_SIGNOFF_VAR] = "1"
        try:
            # Negative: an over-rate calibration report must abort --sweep via BaseSweepAbort.
            bad_results = _synthetic_calibration_results(gate_steps, rate_per_cell=1.0)
            bad_report = write_calibration_report(gate_dir, bad_results)
            print(f"  wrote OVER-RATE calibration report: real_rate_per_cell="
                 f"{bad_report['real_rate_per_cell']:.4f} GPU-h/cell, steps_per_group="
                 f"{bad_report['steps_per_group']}")
            expected_uniform = {name: gate_steps for name in GROUP_NAMES}
            try:
                gate_sweep_launch(gate_dir, expected_uniform)
                raise AssertionError("gate_sweep_launch did NOT abort on an over-rate calibration report")
            except bg.BaseSweepAbort as e:
                print(f"  NEGATIVE TEST PASSED -- gate_sweep_launch correctly raised BaseSweepAbort:")
                print(f"    {e}")

            # Positive: a healthy-rate calibration report must clear the gate cleanly.
            good_results = _synthetic_calibration_results(gate_steps, rate_per_cell=0.3)
            good_report = write_calibration_report(gate_dir, good_results)
            print(f"\n  wrote HEALTHY calibration report: real_rate_per_cell="
                 f"{good_report['real_rate_per_cell']:.4f} GPU-h/cell")
            gate = gate_sweep_launch(gate_dir, expected_uniform)
            assert gate["projection"]["ok"]
            print(f"  POSITIVE TEST PASSED -- gate_sweep_launch cleared: projected_total="
                 f"{gate['projection']['projected_total']:.2f} GPU-h (cap {bg.CAP})")

            # (i) missing-report refusal.
            with tempfile.TemporaryDirectory() as empty_dir:
                try:
                    gate_sweep_launch(empty_dir, expected_uniform)
                    raise AssertionError("gate_sweep_launch did not refuse with no calibration report on disk")
                except RuntimeError as e:
                    assert "no calibration report" in str(e)
                    print(f"\n  missing-report refusal: {e}")
            # (i) per-group step-budget-mismatch refusal (Rev 7: a report whose
            # steps_per_group differs from what THIS launch expects is stale --
            # tested with a genuinely different per-group dict, not a +1 scalar).
            mismatched = dict(expected_uniform); mismatched["A6"] = gate_steps + 1
            try:
                gate_sweep_launch(gate_dir, mismatched)
                raise AssertionError("gate_sweep_launch did not refuse on a per-group step-budget mismatch")
            except RuntimeError as e:
                assert "steps_per_group" in str(e)
                print(f"  per-group step-mismatch refusal: {e}")
        finally:
            del os.environ[PI_SIGNOFF_VAR]

    # (iii) PI-signoff must gate --sweep too, even with an otherwise-healthy report.
    with tempfile.TemporaryDirectory() as gate_dir2:
        good_results = _synthetic_calibration_results(gate_steps, rate_per_cell=0.3)
        write_calibration_report(gate_dir2, good_results)
        try:
            gate_sweep_launch(gate_dir2, expected_uniform)
            raise AssertionError("gate_sweep_launch did not enforce the PI-signoff token")
        except RuntimeError as e:
            assert PI_SIGNOFF_VAR in str(e)
            print(f"  PI-signoff gate (sweep path): {e}")

    print("\n" + "=" * 88)
    print("  run_capability_sep.py SMOKE PASSED (manifest=58 cells w/ Rev-7 STEP_BUDGET pins,")
    print("  calibration=5 cells, resume-safety verified [skip valid / re-run corrupted],")
    print("  PI-signoff gate enforced, per-L convergence_profile + gate1a + L>=2 split present")
    print("  on every cell output, gate1a_check/route_second_miss NEGATIVE+POSITIVE tests passed,")
    print("  escalation wiring exercises tost_analysis.py + budget_guard.py end-to-end,")
    print("  BA-F1 calibration-report/--sweep gate verified [over-rate abort / healthy pass /")
    print("  missing report / per-group stale steps / PI-signoff, all refused or passed correctly],")
    print("  S1.33 m3fix wave verified [30-cell manifest, both-variant grid arithmetic + literal")
    print("  cell-list pin, priced inside the 1.3-2.6 GPU-h window, cross-variant force_rank_k")
    print("  override + resume-safety exercised end-to-end on CPU, PI-signoff gate enforced],")
    print("  S1.36 S3 seed-parameterization extension verified [seed=0 byte-identical to the")
    print("  original manifest, seed threads into cell_id AND cell['seed'] independently for")
    print("  N in {1,2,3}, --m3fix-groups filters to just S3's 6 cells, run_m3fix(seed,groups)")
    print("  end-to-end + resume-safety on CPU, PI-signoff gate enforced])")
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibration-only", action="store_true",
                    help="S1.7 gate 1: run ONLY the 5 calibration cells (each at its OWN group's "
                         "Rev-7-pinned STEP_BUDGET, unless --steps overrides uniformly) and write "
                         "a calibration report (the real measured per-cell rate + per-group steps) "
                         "to --results-dir. Must be run to completion, and re-run whenever --steps "
                         "or the STEP_BUDGET pins change, before --sweep.")
    ap.add_argument("--sweep", action="store_true",
                    help="S1.7 gates 1/2/5: run the full 58-cell sweep. REFUSES to start unless "
                         "(i) a valid calibration report from --calibration-only exists whose "
                         "steps_per_group matches the expected pins, (ii) its measured rate "
                         f"projects the base sweep under the 30 GPU-h cap (BaseSweepAbort "
                         f"otherwise), and (iii) {PI_SIGNOFF_VAR}=1 is set.")
    ap.add_argument("--m3fix", action="store_true",
                    help="S1.33 M3 FIX WAVE: run the 30-cell, both-variant (zero_pad + "
                         "tax_adjusted) fix-wave manifest that closes the D-AMB ambient-identity "
                         "capacity-tax instrument defect (CAPABILITY_SEPARATION_DESIGN.md S1.33) "
                         "that voided the base 58-cell sweep's M3 force-rank arms. Prices from the "
                         "measured Rev-7 sweep rate (no new --calibration-only run needed) and "
                         f"REFUSES to start unless {PI_SIGNOFF_VAR}=1 is set (S1.7 gate 5). "
                         "Mutually exclusive with --calibration-only/--sweep (a separate launch).")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override: train EVERY cell at this single step count, regardless of "
                         "group (smoke/testing/debugging only). Default (unset) uses the Rev 7 "
                         f"per-group STEP_BUDGET pins {STEP_BUDGET} (S1.6/S1.7 gate 1(a)), the "
                         "production behavior.")
    ap.add_argument("--m3fix-seed", type=int, default=0, dest="m3fix_seed",
                    help="S1.36 S3 seed-parameterization extension (--m3fix only): the seed used "
                         "for EVERY cell in the m3fix manifest. Default 0 reproduces the original "
                         "single-seed fix wave's manifest byte-identically (same cell_ids, same "
                         "seed field). Set to 1/2/3 etc. for a seed-extension launch (pair with "
                         "--m3fix-groups to scope it to just the marginal group).")
    ap.add_argument("--m3fix-groups", type=str, default=None, dest="m3fix_groups",
                    help="S1.36 S3 seed-parameterization extension (--m3fix only): comma-separated "
                         "list of group names (subset of S3/S4/A5/S5/A6) to restrict the m3fix "
                         "manifest to, e.g. 'S3' for just S3's 6 cells (4 variant-A + 2 variant-B) "
                         "instead of the full 30-cell both-variant manifest. Default (unset) = all "
                         "groups, unchanged behavior.")
    ap.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    modes_set = sum([args.calibration_only, args.sweep, args.m3fix])
    if modes_set > 1:
        ap.error("--calibration-only, --sweep, and --m3fix are mutually exclusive -- each is a "
                "SEPARATE invocation (S1.22 BA-F1's two-step CLI convention, extended by S1.33's "
                "--m3fix, which reuses the measured Rev-7 rate rather than gating on its own "
                "calibration report).")

    if args.calibration_only:
        run_calibration_only(args)
        return

    if args.sweep:
        run_sweep(args)
        return

    if args.m3fix:
        run_m3fix(args)
        return

    ap.error("specify one of --smoke, --calibration-only, --sweep, or --m3fix -- S1.22 BA-F1 "
            "removed the old implicit/ungated full-sweep default.")


if __name__ == "__main__":
    main()
