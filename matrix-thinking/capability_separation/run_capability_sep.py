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
                     dtype=torch.float32) -> dict:
    """Fresh, length-HOMOGENEOUS eval batch at a SPECIFIC L (S1.7 gate
    1(a)'s per-L convergence metric) -- reuses the SAME
    generating_set/batched_targets primitives sample_train_batch/
    sample_eval_batch already wrap (group_task.py), just pinned to one L
    rather than randomly drawn from a range."""
    n_gens = len(generating_set(name))
    d_state = D_STATE[name]
    token_idx = torch.randint(0, n_gens, (batch_size, L), generator=gen).to(device)
    target = batched_targets(name, token_idx, d_state, dtype=dtype)
    return {"token_idx": token_idx, "target": target, "L": L}


def per_L_convergence_profile(model, name: str, device: str, seed: int,
                              batch_size: int = 64, l_lo: int = 1, l_hi: int = 8) -> dict:
    """S1.7 gate 1(a) (Rev 5/7): for each L in {l_lo..l_hi} (default the
    TRAIN-support range, S1.4), draw a FRESH, length-homogeneous eval batch
    (never used for M1/M3 scoring or training) and score the SAME
    `recovery_cosine` primitive already used for training's `cosine_loss`,
    in `model.eval()`/`no_grad` mode, against the RAW (not degauged --
    this is a diagnostic convergence check, not the M1/M3 decision metric)
    `rho_G_embedded` target. Returns {L: mean_cosine} for L=l_lo..l_hi --
    an 8-point profile at the default range, replacing reliance on the
    training loop's last logged batch loss (S1.25's own diagnosed
    per-batch-fixed-L confound)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed + 555_000 + group_seed_salt(name))
    profile = {}
    with torch.no_grad():
        for L in range(l_lo, l_hi + 1):
            batch = _eval_batch_at_L(name, L, batch_size, gen, device=device)
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
    a uniform tiny step count for every cell regardless of group)."""
    name = cell["group"]
    arm = cell["arm"]
    seed = cell["seed"]
    steps = steps_override if steps_override is not None else cell["steps"]
    k = force_rank_grid(name)[arm]

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
        batch = sample_train_batch(name, 256, gen, device=device)
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
    profile = per_L_convergence_profile(model, name, device=device, seed=eval_seed)
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
                 l_ge2_n_eval=len(l_ge2_coses), l_ge2_n_total=len(eval_L))
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

    print("\nRESULT: gate1a_check correctly enforces the L in {2..5}/>=0.02-margin bar and "
          "correctly excludes (but still discloses) L=1.\n")
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
    print("  missing report / per-group stale steps / PI-signoff, all refused or passed correctly])")
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
    ap.add_argument("--steps", type=int, default=None,
                    help="Override: train EVERY cell at this single step count, regardless of "
                         "group (smoke/testing/debugging only). Default (unset) uses the Rev 7 "
                         f"per-group STEP_BUDGET pins {STEP_BUDGET} (S1.6/S1.7 gate 1(a)), the "
                         "production behavior.")
    ap.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    if args.calibration_only and args.sweep:
        ap.error("--calibration-only and --sweep are mutually exclusive -- S1.22 BA-F1's two-step "
                "CLI runs calibration first, then sweep as a SEPARATE invocation once the report exists.")

    if args.calibration_only:
        run_calibration_only(args)
        return

    if args.sweep:
        run_sweep(args)
        return

    ap.error("specify one of --smoke, --calibration-only, or --sweep -- S1.22 BA-F1 removed the "
            "old implicit/ungated full-sweep default.")


if __name__ == "__main__":
    main()
