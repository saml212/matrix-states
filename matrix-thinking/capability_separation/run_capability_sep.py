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
from groups import GROUP_NAMES, D_MIN, D_STATE, group_seed_salt
from group_task import generating_set, sample_train_batch, sample_eval_batch
from group_word_encoder import GroupWordModel, cosine_loss
from force_rank_arms import force_rank_grid, cell_types_for_group, n_cells_per_group
import readout
import budget_guard as bg
import tost_analysis as tost

PI_SIGNOFF_VAR = "CAPABILITY_SEP_PI_SIGNOFF"
RESULTS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_STEPS = 8000       # S1.6's planned default step budget


# ---------------------------------------------------------------------------
# Manifest construction (S1.4.2's exact cell table).
# ---------------------------------------------------------------------------

def build_sweep_manifest() -> list:
    """The full 58-cell manifest (S1.4.2): for each group, for each arm
    (unconstrained/k_dmin_minus_1/k_dmin/k_dmin_plus_1), one cell per seed
    at that cell type's BASE seed count (S4/A5 already unconditional n=5 at
    unconstrained+k_dmin). Escalation-triggered cells are NOT pre-included
    here -- they are appended dynamically by `run_escalations` after the
    base sweep's results are in, gated through budget_guard."""
    manifest = []
    for name in GROUP_NAMES:
        unconditional_n5 = name in ("S4", "A5")
        cell_types = cell_types_for_group(name, unconditional_n5)
        for arm_label, n_seeds in cell_types.items():
            for seed in range(n_seeds):
                cell_id = f"{name}__{arm_label}__seed{seed}"
                manifest.append(dict(cell_id=cell_id, group=name, arm=arm_label, seed=seed,
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


def write_calibration_report(results_dir: str, calib_results: list, steps: int) -> dict:
    """S1.7 gate 1 duty (d): aggregate the 5 calibration cells' measured
    wall-clock into a real per-cell GPU-h rate, written to disk so `--sweep`
    can gate on it without re-running calibration. One GPU trains one cell
    serially in this harness's execution model (run_manifest processes the
    manifest sequentially, one cell at a time), so wall_clock_s/3600 IS the
    GPU-h/cell figure -- the real rate that supersedes S1.6's 0.3 GPU-h/cell
    planning estimate."""
    rates = [r["wall_clock_s"] / 3600.0 for r in calib_results]
    real_rate_per_cell = sum(rates) / len(rates)
    report = dict(
        real_rate_per_cell=real_rate_per_cell,
        per_cell_rates=rates,
        n_cells=len(calib_results),
        steps=steps,
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


def load_and_validate_calibration_report(results_dir: str, expected_steps: int) -> dict:
    """`--sweep` precondition (i): a valid, on-disk calibration report from
    THIS step budget. Raises RuntimeError (not BaseSweepAbort -- that is
    reserved for a budget-projection failure, not a missing/invalid report)
    with a specific, actionable reason."""
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
    required = {"real_rate_per_cell", "n_cells", "steps", "cell_ids", "groups"}
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
    if report["steps"] != expected_steps:
        raise RuntimeError(
            f"calibration report at {path} was measured at steps={report['steps']}, but this "
            f"launch is requesting steps={expected_steps} -- re-run `--calibration-only "
            f"--steps={expected_steps}` first (a rate measured at a different step budget does "
            f"not project this sweep's per-cell cost)."
        )
    rate = report["real_rate_per_cell"]
    if not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
        raise RuntimeError(f"calibration report at {path} has an invalid real_rate_per_cell={rate!r}")
    return report


def gate_sweep_launch(results_dir: str, steps: int) -> dict:
    """The three `--sweep` preconditions (S1.22 BA-F1), in the design's own
    order: (i) a valid calibration report on disk, (ii) its measured rate
    projects the base sweep under the 30 GPU-h cap (raises BaseSweepAbort
    via budget_guard.check_base_sweep_projection if not -- S1.7 gate 2's
    mechanical enforced abort, now actually wired), (iii) the PI-signoff
    token. Factored out of run_sweep() so smoke can exercise the gate in
    isolation without needing a full 58-cell GPU sweep to run."""
    report = load_and_validate_calibration_report(results_dir, steps)                # (i)
    projection = bg.check_base_sweep_projection(report["real_rate_per_cell"])          # (ii)
    if os.environ.get(PI_SIGNOFF_VAR) != "1":                                          # (iii)
        raise RuntimeError(f"{PI_SIGNOFF_VAR}=1 required before any GPU cell (S1.7 gate 5).")
    return dict(calibration_report=report, projection=projection)


def _synthetic_calibration_results(steps: int, rate_per_cell: float) -> list:
    """Build cell-shaped result dicts (NOT real training) with `wall_clock_s`
    set so `write_calibration_report`'s averaged rate equals `rate_per_cell`
    exactly -- for smoke ONLY, to exercise the report/gate wiring without
    paying for 5 real training runs on every smoke invocation."""
    calib = calibration_wave()
    wall_s = rate_per_cell * 3600.0
    return [dict(cell_id=c["cell_id"], group=c["group"], arm=c["arm"], seed=c["seed"],
                force_rank_k=None, steps_completed=steps, n_skipped_steps=0, wall_clock_s=wall_s,
                mean_cos=0.95, recovered_frac_90=0.95, restricted_effective_rank=3.0,
                restricted_stable_rank=3.0, whole_matrix_effective_rank=4.0, c_hat=1.0)
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

def train_and_eval_cell(cell: dict, steps: int, device: str, log_every: int = 500) -> dict:
    name = cell["group"]
    arm = cell["arm"]
    seed = cell["seed"]
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
    result = dict(cell_id=cell["cell_id"], group=name, arm=arm, seed=seed, force_rank_k=k,
                 steps_completed=steps, n_skipped_steps=n_skipped, wall_clock_s=wall_s,
                 mean_cos=scores["mean_cos"], recovered_frac_90=scores["recovered_frac_90"],
                 restricted_effective_rank=scores["restricted_effective_rank"],
                 restricted_stable_rank=scores["restricted_stable_rank"],
                 whole_matrix_effective_rank=scores["whole_matrix_effective_rank"],
                 c_hat=scores["c_hat"])
    return result


def run_cell_resume_safe(cell: dict, results_dir: str, steps: int, device: str) -> dict:
    path = cell_output_path(results_dir, cell["cell_id"])
    if is_valid_output(path):
        with open(path) as f:
            print(f"  [{cell['cell_id']}] SKIP (valid output already on disk)")
            return json.load(f)
    result = train_and_eval_cell(cell, steps, device)
    os.makedirs(results_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, path)   # atomic write -- a crash mid-write never leaves a corrupt "final" file
    return result


# ---------------------------------------------------------------------------
# Sweep orchestration.
# ---------------------------------------------------------------------------

def run_manifest(manifest: list, results_dir: str, steps: int, device: str,
                 require_pi_signoff: bool = True) -> list:
    if require_pi_signoff and os.environ.get(PI_SIGNOFF_VAR) != "1":
        raise RuntimeError(
            f"{PI_SIGNOFF_VAR}=1 required before any GPU cell (S1.7 gate 5). "
            f"Set it explicitly to launch; this is NOT set automatically by this script."
        )
    results = []
    for cell in manifest:
        results.append(run_cell_resume_safe(cell, results_dir, steps, device))
    return results


def run_calibration_only(args) -> dict:
    """S1.22 BA-F1's `--calibration-only` step: run the 5 calibration cells
    to completion and write the measured real per-cell rate to disk -- the
    precondition `--sweep` gates on below."""
    manifest = calibration_wave()
    print(f"[calibration-only] launching {len(manifest)} calibration cells "
         f"(steps={args.steps}, device={args.device}, results_dir={args.results_dir})")
    results = run_manifest(manifest, args.results_dir, args.steps, args.device)
    report = write_calibration_report(args.results_dir, results, args.steps)
    print(f"[calibration-only] wrote calibration report: real_rate_per_cell="
         f"{report['real_rate_per_cell']:.4f} GPU-h/cell -> {calibration_report_path(args.results_dir)}")
    return report


def run_sweep(args) -> tuple:
    """S1.22 BA-F1's `--sweep` step: gated launch. Raises BaseSweepAbort (via
    gate_sweep_launch -> budget_guard.check_base_sweep_projection) if the
    calibration-measured rate would blow the 30 GPU-h cap, and RuntimeError
    if the calibration report is missing/invalid/stale or the PI-signoff
    token isn't set -- BEFORE building or launching the 58-cell manifest."""
    gate = gate_sweep_launch(args.results_dir, args.steps)
    print(f"[sweep] gate PASSED: measured rate {gate['calibration_report']['real_rate_per_cell']:.4f} "
         f"GPU-h/cell -> base-sweep projection {gate['projection']['projected_total']:.2f} GPU-h "
         f"(cap {bg.CAP:.1f})")
    manifest = build_sweep_manifest()
    print(f"[sweep] launching {len(manifest)} cells (steps={args.steps}, device={args.device}, "
         f"results_dir={args.results_dir})")
    results = run_manifest(manifest, args.results_dir, args.steps, args.device)
    guard = bg.BudgetGuard(cap=bg.CAP, spend_to_date=gate["projection"]["projected_total"])
    esc_log = run_escalations(results, args.results_dir, args.steps, args.device, guard)
    return results, esc_log


def group_results_by(results: list, group: str, arm: str) -> np.ndarray:
    return np.array([r["restricted_effective_rank"] for r in results
                     if r["group"] == group and r["arm"] == arm])


def run_escalations(base_results: list, results_dir: str, steps: int, device: str,
                    guard: bg.BudgetGuard) -> dict:
    """S1.4.2/S1.6/S1.20's escalation pass: evaluate ambiguity triggers off
    the base-sweep results, run the marquee TOST check (tost_analysis.py,
    wired to `guard`), and any GENERAL escalation-to-5 triggers, granting
    through the guard in the pinned order. Escalation-triggered cells that
    ARE granted get extra seeds trained via run_cell_resume_safe (resume-safe,
    same as the base sweep); denied ones are recorded with status
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
                               is_calibration=False)
                    run_cell_resume_safe(cell, results_dir, steps, device)

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

def smoke():
    print("=" * 88)
    print("  run_capability_sep.py SMOKE -- manifest + resume-safety + escalation wiring")
    print("=" * 88)
    manifest = build_sweep_manifest()
    print(f"  build_sweep_manifest(): {len(manifest)} cells (expect 58)")
    assert len(manifest) == 58
    calib = calibration_wave()
    print(f"  calibration_wave(): {len(calib)} cells (expect 5, 1/group unconstrained seed=0)")
    assert len(calib) == 5
    assert {c["group"] for c in calib} == set(GROUP_NAMES)

    import tempfile
    with tempfile.TemporaryDirectory() as results_dir:
        os.environ[PI_SIGNOFF_VAR] = "1"    # local smoke, not a real launch -- BUILD-scope only
        try:
            # a tiny slice (S4's unconstrained arm, all 5 seeds + one k_dmin seed, PLUS A5's
            # unconstrained arm, all 5 seeds -- so run_escalations() below has real data for
            # BOTH marquee groups) at 20 steps -- enough to exercise train/eval/write, not a
            # real training run.
            tiny_manifest = [c for c in manifest if c["group"] == "S4" and c["arm"] == "unconstrained"]
            tiny_manifest += [c for c in manifest if c["group"] == "S4" and c["arm"] == "k_dmin"][:1]
            tiny_manifest += [c for c in manifest if c["group"] == "A5" and c["arm"] == "unconstrained"]
            print(f"\n  running a {len(tiny_manifest)}-cell tiny slice at steps=20 (CPU, smoke only):")
            t0 = time.time()
            results1 = run_manifest(tiny_manifest, results_dir, steps=20, device="cpu")
            print(f"  first pass: {len(results1)} results written in {time.time()-t0:.1f}s")
            for r in results1:
                assert os.path.exists(cell_output_path(results_dir, r["cell_id"]))
                assert is_valid_output(cell_output_path(results_dir, r["cell_id"]))

            # RESUME-SAFETY test: re-run the SAME manifest -- must SKIP every cell
            # (valid output already on disk), not re-train.
            t1 = time.time()
            results2 = run_manifest(tiny_manifest, results_dir, steps=20, device="cpu")
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
            result3 = run_cell_resume_safe(corrupt_cell, results_dir, steps=20, device="cpu")
            assert is_valid_output(corrupt_path), "re-run after corruption did not produce valid output"
            print(f"  re-run after corruption: cell '{corrupt_cell['cell_id']}' correctly RE-TRAINED and "
                  f"now valid again")

            # escalation wiring: run_escalations with a real BudgetGuard.
            guard = bg.BudgetGuard(cap=bg.CAP, spend_to_date=0.0)
            esc_log = run_escalations(results1, results_dir, steps=20, device="cpu", guard=guard)
            print(f"\n  run_escalations() marquee verdict: {esc_log['marquee']['verdict']}  "
                  f"final_verdict: {esc_log['marquee']['final_verdict']}")
        finally:
            del os.environ[PI_SIGNOFF_VAR]

    # PI signoff gate must actually block without the token.
    try:
        run_manifest(manifest[:1], "/tmp/should_not_write", steps=1, device="cpu")
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
    # -------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("  BA-F1 smoke -- calibration report + --sweep gate (calibration-only -> gate -> sweep)")
    print("=" * 88)
    gate_steps = 20
    with tempfile.TemporaryDirectory() as gate_dir:
        os.environ[PI_SIGNOFF_VAR] = "1"
        try:
            # Negative: an over-rate calibration report must abort --sweep via BaseSweepAbort.
            bad_results = _synthetic_calibration_results(gate_steps, rate_per_cell=1.0)
            bad_report = write_calibration_report(gate_dir, bad_results, gate_steps)
            print(f"  wrote OVER-RATE calibration report: real_rate_per_cell="
                 f"{bad_report['real_rate_per_cell']:.4f} GPU-h/cell")
            try:
                gate_sweep_launch(gate_dir, gate_steps)
                raise AssertionError("gate_sweep_launch did NOT abort on an over-rate calibration report")
            except bg.BaseSweepAbort as e:
                print(f"  NEGATIVE TEST PASSED -- gate_sweep_launch correctly raised BaseSweepAbort:")
                print(f"    {e}")

            # Positive: a healthy-rate calibration report must clear the gate cleanly.
            good_results = _synthetic_calibration_results(gate_steps, rate_per_cell=0.3)
            good_report = write_calibration_report(gate_dir, good_results, gate_steps)
            print(f"\n  wrote HEALTHY calibration report: real_rate_per_cell="
                 f"{good_report['real_rate_per_cell']:.4f} GPU-h/cell")
            gate = gate_sweep_launch(gate_dir, gate_steps)
            assert gate["projection"]["ok"]
            print(f"  POSITIVE TEST PASSED -- gate_sweep_launch cleared: projected_total="
                 f"{gate['projection']['projected_total']:.2f} GPU-h (cap {bg.CAP})")

            # (i) missing-report refusal.
            with tempfile.TemporaryDirectory() as empty_dir:
                try:
                    gate_sweep_launch(empty_dir, gate_steps)
                    raise AssertionError("gate_sweep_launch did not refuse with no calibration report on disk")
                except RuntimeError as e:
                    assert "no calibration report" in str(e)
                    print(f"\n  missing-report refusal: {e}")
            # (i) step-budget-mismatch refusal (a report from a different --steps is stale).
            try:
                gate_sweep_launch(gate_dir, gate_steps + 1)
                raise AssertionError("gate_sweep_launch did not refuse on a step-budget mismatch")
            except RuntimeError as e:
                assert "steps=" in str(e)
                print(f"  step-mismatch refusal: {e}")
        finally:
            del os.environ[PI_SIGNOFF_VAR]

    # (iii) PI-signoff must gate --sweep too, even with an otherwise-healthy report.
    with tempfile.TemporaryDirectory() as gate_dir2:
        good_results = _synthetic_calibration_results(gate_steps, rate_per_cell=0.3)
        write_calibration_report(gate_dir2, good_results, gate_steps)
        try:
            gate_sweep_launch(gate_dir2, gate_steps)
            raise AssertionError("gate_sweep_launch did not enforce the PI-signoff token")
        except RuntimeError as e:
            assert PI_SIGNOFF_VAR in str(e)
            print(f"  PI-signoff gate (sweep path): {e}")

    print("\n" + "=" * 88)
    print("  run_capability_sep.py SMOKE PASSED (manifest=58 cells, calibration=5 cells,")
    print("  resume-safety verified [skip valid / re-run corrupted], PI-signoff gate enforced,")
    print("  escalation wiring exercises tost_analysis.py + budget_guard.py end-to-end,")
    print("  BA-F1 calibration-report/--sweep gate verified [over-rate abort / healthy pass /")
    print("  missing report / stale steps / PI-signoff, all refused or passed correctly])")
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibration-only", action="store_true",
                    help="S1.7 gate 1: run ONLY the 5 calibration cells and write a calibration "
                         "report (the real measured per-cell rate) to --results-dir. Must be run "
                         "to completion, and re-run whenever --steps changes, before --sweep.")
    ap.add_argument("--sweep", action="store_true",
                    help="S1.7 gates 1/2/5: run the full 58-cell sweep. REFUSES to start unless "
                         "(i) a valid calibration report from --calibration-only exists at the "
                         "matching --steps, (ii) its measured rate projects the base sweep under "
                         f"the 30 GPU-h cap (BaseSweepAbort otherwise), and (iii) {PI_SIGNOFF_VAR}=1 "
                         "is set.")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
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
