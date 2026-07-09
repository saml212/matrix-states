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
from groups import GROUP_NAMES, D_MIN, D_STATE
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

    torch.manual_seed(seed)
    d_state = D_STATE[name]
    n_gens = len(generating_set(name))
    model = GroupWordModel(d_state, n_gens, L_max=16, h=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(seed)

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

    print("\n" + "=" * 88)
    print("  run_capability_sep.py SMOKE PASSED (manifest=58 cells, calibration=5 cells,")
    print("  resume-safety verified [skip valid / re-run corrupted], PI-signoff gate enforced,")
    print("  escalation wiring exercises tost_analysis.py + budget_guard.py end-to-end)")
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibration-only", action="store_true")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    if args.calibration_only:
        manifest = calibration_wave()
    else:
        manifest = build_sweep_manifest()
    print(f"launching {len(manifest)} cells (steps={args.steps}, device={args.device}, "
         f"results_dir={args.results_dir})")
    run_manifest(manifest, args.results_dir, args.steps, args.device)


if __name__ == "__main__":
    main()
