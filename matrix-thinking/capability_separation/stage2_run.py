"""CAPABILITY_SEPARATION_DESIGN.md S2.5/S2.7/S2.8/S2.11 (Rev 3,
DESIGN-CLEARED-FOR-BUILD per S2.18) -- the Stage-2 calibration-first CLI:
cell-grid construction (68 new training cells: 50 primary + 18 n_h-grid),
the 11-cell calibration-first set, per-cell and ledger-level budget guards,
resume-safe cell running, and the orchestration glue tying
`stage2_composer.py` + `stage2_task.py` + `stage2_instrument.py` together.

This module builds and self-tests the ORCHESTRATION LOGIC on CPU with tiny
synthetic cells; it does NOT launch any real GPU training (out of this
BUILD agent's scope -- build, not launch; an independent audit follows,
S2.11's own sequencing). `main()`'s `--calibration-only`/real-launch paths
mirror `run_capability_sep.py`'s own PI-signoff-gated convention but are
NOT exercised end-to-end here.

Grid (S2.5): primary = 5 groups x {Arm 2 (beta in [0,1]), Arm 3 (beta in
[0,2])} x n=5 seeds, n_h=2 default = 50 cells. n_h force-arm grid = {S5,A6}
x n_h in {1,2,4} x n=3 seeds, Arm 3 only = 18 cells. Total 68. Calibration
set (S2.8 item 2, 11 cells): the 10 base (group,arm) cells at n_h=2, seed
index 0, PLUS the promoted (S5, Arm 3, n_h=4, seed index 0) cell -- drawn
from, not added to, the 68 (S2.5 Rev 2 pin: the base (S5,Arm3,n_h=2) cell
and the promoted (S5,Arm3,n_h=4) cell are DISTINCT, not a near-duplicate).

Budget (S2.7 Rev 2): per-cell abort ceiling = 1.5x the pricier end of the
0.018-0.054 GPU-h/cell planning band = 0.081 GPU-h/cell (this dispatch's
own "per-cell budget guard at 1.5x the S2.7 band" instruction). Ledger
circuit breaker mirrors `budget_guard.py`'s `check_base_sweep_projection`
MECHANISM, re-keyed to Stage 2's own 25 GPU-h cap and 68-cell grid (Stage
1's own module constants -- BASE_SEEDS, GENERAL_ESCALATION_ELIGIBLE, the
58-cell table -- are specific to Stage 1's own arm/cell-type structure and
do not apply here; re-implementing the MECHANISM against Stage 2's own
numbers, rather than importing Stage-1-specific constants, is what "reuses
the exact mechanism, re-keyed" means, S2.8 item 3).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import readout
import run_capability_sep as rcs   # Stage 1's own module -- IMPORTED and called, never edited
                                    # (FILE OWNERSHIP): STEP_BUDGET, calibration_wave, GroupWordModel
                                    # re-export, force_rank_grid re-export -- see
                                    # retrain_and_save_arm1_checkpoints's docstring.
import stage2_composer as sc
import stage2_instrument as si
import stage2_task as st
from groups import GROUP_NAMES, D_STATE, generating_set, group_seed_salt
from group_word_encoder import GroupWordModel, cosine_loss

PI_SIGNOFF_VAR = "CAPABILITY_SEP_STAGE2_PI_SIGNOFF"

# ---------------------------------------------------------------------------
# Grid construction (S2.5).
# ---------------------------------------------------------------------------

PRIMARY_SEEDS = 5
NH_GRID_SEEDS = 3
NH_GRID_GROUPS = ("S5", "A6")
NH_GRID_VALUES = (1, 2, 4)
ARM_BETA_MAX = {"arm2_beta01": 1.0, "arm3_beta02": 2.0}
ARMS = tuple(ARM_BETA_MAX)


def build_primary_grid() -> list[dict]:
    """5 groups x 2 beta-arms x n=5 seeds, n_h=2 default -- 50 cells."""
    cells = []
    for name in GROUP_NAMES:
        for arm in ARMS:
            for seed in range(PRIMARY_SEEDS):
                cells.append(dict(cell_id=f"{name}__{arm}__nh2__seed{seed}", group=name, arm=arm,
                                  n_h=2, seed=seed))
    assert len(cells) == 50, f"primary grid: expected 50 cells, got {len(cells)}"
    return cells


def build_nh_grid() -> list[dict]:
    """{S5,A6} x n_h in {1,2,4} x n=3 seeds, Arm 3 ONLY (S2.5 Rev 2 pin:
    the n_h axis tests DeltaProduct's published sufficiency claims, measured
    at allow_neg_eigval=True -- running it on Arm 2 would confound n_h with
    the beta-range exclusion Theorems 1/3 already predict binds Arm 2) --
    18 cells."""
    cells = []
    for name in NH_GRID_GROUPS:
        for n_h in NH_GRID_VALUES:
            for seed in range(NH_GRID_SEEDS):
                cells.append(dict(cell_id=f"{name}__arm3_beta02__nh{n_h}__seed{seed}", group=name,
                                  arm="arm3_beta02", n_h=n_h, seed=seed))
    assert len(cells) == 18, f"n_h grid: expected 18 cells, got {len(cells)}"
    return cells


def build_calibration_set(primary: list[dict], nh_grid: list[dict]) -> list[dict]:
    """11 cells: the 10 base (group,arm) cells at n_h=2/seed0, PLUS the
    promoted (S5, Arm3, n_h=4, seed0) cell -- drawn from the two grids
    above, zero incremental cells (S2.5/S2.8 item 2)."""
    base = [c for c in primary if c["seed"] == 0]
    assert len(base) == 10, f"base calibration set: expected 10, got {len(base)}"
    promoted = [c for c in nh_grid
               if c["group"] == "S5" and c["arm"] == "arm3_beta02" and c["n_h"] == 4 and c["seed"] == 0]
    assert len(promoted) == 1, f"promoted (S5,Arm3,n_h=4) calibration cell: expected 1, got {len(promoted)}"
    calib = base + promoted
    assert len(calib) == 11
    # S2.5 Rev 2 pin: the base (S5,Arm3,n_h=2) cell and the promoted
    # (S5,Arm3,n_h=4) cell must be DISTINCT (not a near-duplicate).
    s5_arm3_calib = [c for c in calib if c["group"] == "S5" and c["arm"] == "arm3_beta02"]
    assert len(s5_arm3_calib) == 2 and {c["n_h"] for c in s5_arm3_calib} == {2, 4}, (
        f"S5/Arm3 calibration cells must be the DISTINCT (n_h=2 base) and (n_h=4 promoted) pair, "
        f"got {s5_arm3_calib}"
    )
    return calib


# ---------------------------------------------------------------------------
# Budget guards (S2.7/S2.8 item 3).
# ---------------------------------------------------------------------------

PLANNING_RATE_LO, PLANNING_RATE_HI = 0.018, 0.054   # GPU-h/cell, S2.7 Rev 1 band
PER_CELL_ABORT_CEILING = round(PLANNING_RATE_HI * 1.5, 4)   # 0.081 GPU-h/cell
STAGE2_CAP = 25.0
N_CALIBRATION_CELLS = 11
N_TOTAL_CELLS = 68
N_REMAINING_AFTER_CALIBRATION = N_TOTAL_CELLS - N_CALIBRATION_CELLS   # 57


class PerCellBudgetAbort(RuntimeError):
    pass


class Stage2BudgetAbort(RuntimeError):
    pass


def check_per_cell_projection(elapsed_h: float, steps_done: int, steps_total: int) -> dict:
    """Per-cell circuit breaker: if the IN-PROGRESS cell's wall-clock rate
    projects past 1.5x the pricier end of the S2.7 planning band, hard-
    abort THIS cell before it burns further budget."""
    if steps_done <= 0:
        return dict(ok=True, projected=0.0)
    projected = round(elapsed_h * (steps_total / steps_done), 6)
    ok = projected <= PER_CELL_ABORT_CEILING
    result = dict(elapsed_h=elapsed_h, steps_done=steps_done, steps_total=steps_total,
                 projected=projected, ceiling=PER_CELL_ABORT_CEILING, ok=ok)
    if not ok:
        raise PerCellBudgetAbort(
            f"cell projected {projected:.4f} GPU-h exceeds the per-cell abort ceiling "
            f"{PER_CELL_ABORT_CEILING} GPU-h (1.5x the {PLANNING_RATE_HI} GPU-h/cell planning-"
            f"band ceiling, S2.7) at {steps_done}/{steps_total} steps -- HARD ABORT this cell. {result}"
        )
    return result


def check_stage2_sweep_projection(real_rate_per_cell: float,
                                  n_remaining: int = N_REMAINING_AFTER_CALIBRATION,
                                  spend_so_far: float = 0.0, cap: float = STAGE2_CAP) -> dict:
    """S2.8 item 3's timing pilot / circuit breaker: mirrors
    `budget_guard.py::check_base_sweep_projection`'s MECHANISM (project the
    remaining grid from the calibration cells' REAL measured rate,
    hard-abort before launching it if the projection exceeds the cap),
    re-keyed to Stage 2's own 25 GPU-h cap and 57-cell remainder (Stage 1's
    own module constants are specific to its 58-cell/arm-type structure and
    do not transfer, S2.0's file-ownership constraint)."""
    projected = round(spend_so_far + n_remaining * real_rate_per_cell, 4)
    ok = projected <= cap
    result = dict(real_rate_per_cell=real_rate_per_cell, n_remaining=n_remaining,
                 spend_so_far=spend_so_far, projected_total=projected, cap=cap, ok=ok)
    if not ok:
        raise Stage2BudgetAbort(
            f"Stage-2 sweep projection {projected:.2f} GPU-h exceeds the {cap:.1f} GPU-h cap at "
            f"real rate {real_rate_per_cell:.4f} GPU-h/cell -- HARD ABORT before spending the "
            f"remaining {n_remaining}-cell grid. Re-scope before relaunch. {result}"
        )
    return result


# ---------------------------------------------------------------------------
# Resume-safe cell running (mirrors run_capability_sep.py's exact
# atomic-write / output-validity convention, CLAUDE.md's "skip already-
# completed work by checking output validity, not just existence" rule).
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_KEYS = {"cell_id", "group", "arm", "n_h", "seed", "status"}


def cell_output_path(results_dir: str, cell_id: str) -> str:
    return os.path.join(results_dir, f"{cell_id}.json")


def is_valid_output(path: str) -> bool:
    """S2.20 F4: the `D_test_results`-not-None check below was VACUOUS
    before this fix -- nothing ever wrote that key, so `key in d` was
    always False and the check was a silent no-op. `run_real_cell` (S2.20
    F4, below) now always writes a real (non-None) `D_test_results` list,
    making this check load-bearing for real-runner outputs (a corrupted/
    truncated write with the key explicitly set to `None` is now actually
    caught); it stays a harmless no-op for `train_cell_tiny`/calibration-
    wave outputs, which never set this key at all (by design -- those are
    NOT the D_test grid evaluation, S2.6)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not REQUIRED_OUTPUT_KEYS.issubset(d.keys()):
        return False
    if d["status"] == "completed":
        for key in ("D_test_results",):
            if key in d and d[key] is None:
                return False
    return True


# ---------------------------------------------------------------------------
# Parameter-fingerprint protection (S2.20 F3): the §2.19 self-caught bug --
# the calibration gate initially probed a FRESHLY-REINITIALIZED composer,
# not the trained one -- was fixed but left UNPROTECTED (regressing it
# passes every existing smoke, since nothing ever compares the probed
# object's actual parameters against the checkpoint on disk). This closes
# that gap structurally: a per-tensor SHA-256 checksum of a composer's
# state_dict, asserted equal between "the composer about to be probed" and
# "what is actually persisted on disk" before any gate result is trusted.
# ---------------------------------------------------------------------------

def state_dict_fingerprint(state_dict: dict) -> str:
    """A deterministic (sorted-key) per-tensor SHA-256 checksum of a
    composer's state_dict (S2.20 F3)."""
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        t = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class ParamFingerprintMismatch(AssertionError):
    pass


def assert_fingerprint_matches(probed_composer: "sc.GroupWordDeltaComposer", ckpt_path: str,
                               cell_id: str, device="cpu") -> str:
    """S2.20 F3: hard-fail if the composer about to be probed by the
    calibration gate does not match what is actually persisted at
    `ckpt_path` -- protects the EXACT class of the §2.19 self-caught bug
    (a freshly-reinitialized composer probed instead of the trained one),
    which passed every prior smoke because nothing ever compared the two.
    Returns the (matching) fingerprint on success."""
    on_disk = torch.load(ckpt_path, map_location=device)
    disk_fp = state_dict_fingerprint(on_disk)
    probed_fp = state_dict_fingerprint(probed_composer.state_dict())
    if probed_fp != disk_fp:
        raise ParamFingerprintMismatch(
            f"S2.20 F3 PARAMETER-FINGERPRINT MISMATCH for {cell_id!r}: probed={probed_fp[:12]}... "
            f"disk={disk_fp[:12]}... -- refusing to trust a query-dependence gate result computed "
            f"against a composer that does not match its own just-persisted checkpoint (the §2.19 "
            f"fresh-reinit regression class)."
        )
    return disk_fp


def run_cell_resume_safe(cell: dict, results_dir: str, run_fn) -> dict:
    """`run_fn(cell) -> dict` performs the actual (build-time: synthetic;
    launch-time: real GPU) work. Atomic write via a .tmp + os.replace, the
    SAME crash-safety convention `run_capability_sep.py::run_cell_resume_safe`
    uses."""
    path = cell_output_path(results_dir, cell["cell_id"])
    if is_valid_output(path):
        with open(path) as f:
            print(f"  [{cell['cell_id']}] SKIP (valid output already on disk)")
            return json.load(f)
    result = run_fn(cell)
    os.makedirs(results_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp_path, path)
    return result


# ---------------------------------------------------------------------------
# One-cell orchestration: build the composer, train (build-time: a tiny
# synthetic loop; real launches pass a real step budget), run the
# query-dependence gate at calibration time, evaluate the D_test grid.
# NOT a real training loop -- this wires the pieces together and is
# exercised end-to-end only with `steps_override` tiny values in smoke.
# ---------------------------------------------------------------------------

def build_cell_composer(cell: dict, device="cpu") -> sc.GroupWordDeltaComposer:
    name = cell["group"]
    d_state, n_gens = D_STATE[name], len(generating_set(name))
    torch.manual_seed(cell["seed"])
    return sc.GroupWordDeltaComposer(d_state, n_gens, h=32, n_h=cell["n_h"],
                                     beta_max=ARM_BETA_MAX[cell["arm"]]).to(device)


def checkpoint_path(results_dir: str, cell_id: str) -> str:
    return os.path.join(results_dir, f"{cell_id}.pt")


def train_cell_tiny(cell: dict, results_dir: str, device="cpu", steps: int = 20,
                    budget_guard: bool = True) -> tuple[dict, sc.GroupWordDeltaComposer]:
    """*** SMOKE-ONLY -- DO NOT USE FOR REAL CELLS. *** (S2.20 F4 marks this
    explicitly; `run_real_cell` below is the real cell runner: cosine_loss
    not MSE, real per-group step budgets not a uniform tiny `steps`, the
    M-D0 convergence profile, the full 7-depth 2(e) gate, and the D_test
    grid -- none of which this function does.) A tiny-step-count/tiny-
    batch/MSE-loss training loop exercising ONLY the wire-up (composer ->
    loss -> backward -> per-cell budget guard -> checkpoint persistence);
    NOT the real 8K+-step Stage-2 training regime, NOT `cosine_loss` (the
    pinned objective), and NOT evaluated at all (no M-D0/2(e)/D_test).
    Kept for `smoke_stage2.py`'s own fast wiring checks only.

    UNLIKE Stage 1's own convention (`run_capability_sep.py::
    train_and_eval_cell` trains then discards the model, S1.33 M2's own
    disclosed build gap), Arms 2-3 DO need their trained state persisted:
    the last-K-window shortcut control (S2.9 item 4) is pinned as
    EVAL-TIME truncation of the EXISTING trained Arm-3 checkpoint, which
    requires the checkpoint to actually exist on disk after training
    completes. Returns (json-safe result dict, the trained composer) --
    the composer is ALSO what the calibration gate (S2.8 item 2(e)) must
    run against, never a freshly-reinitialized stand-in."""
    name = cell["group"]
    composer = build_cell_composer(cell, device=device)
    opt = torch.optim.Adam(composer.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(cell["seed"] + 1)

    t0 = time.time()
    final_loss = None
    for step in range(1, steps + 1):
        batch = st.sample_train_batch_stage2(name, 32, gen, device=device)
        Z = composer(batch["token_idx"])
        loss = torch.nn.functional.mse_loss(Z, batch["target"])  # any finite differentiable loss suffices for the wiring smoke
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
        if budget_guard and step % 5 == 0:
            elapsed_h = (time.time() - t0) / 3600.0
            check_per_cell_projection(elapsed_h, step, steps)
    wall_s = time.time() - t0

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = checkpoint_path(results_dir, cell["cell_id"])
    torch.save(composer.state_dict(), ckpt_path)

    result = dict(cell_id=cell["cell_id"], group=cell["group"], arm=cell["arm"], n_h=cell["n_h"],
                 seed=cell["seed"], status="completed", steps_completed=steps,
                 final_loss=final_loss, wall_clock_s=wall_s, checkpoint_path=ckpt_path)
    return result, composer


def load_cell_composer(cell: dict, results_dir: str, device="cpu") -> sc.GroupWordDeltaComposer:
    """Round-trip loader for the resume-skip path: rebuild the composer's
    ARCHITECTURE (deterministic from `cell`) and load its trained weights
    from the checkpoint `train_cell_tiny` persisted -- the SAME on-disk
    round-trip discipline `gate1_synthetic_injection.py` uses for its own
    dump/reload (write, then read back, before trusting downstream code)."""
    composer = build_cell_composer(cell, device=device)
    ckpt_path = checkpoint_path(results_dir, cell["cell_id"])
    composer.load_state_dict(torch.load(ckpt_path, map_location=device))
    return composer.to(device)


def run_calibration_gate_for_cell(cell: dict, composer: sc.GroupWordDeltaComposer,
                                  depths=si.PROBE_DEPTHS, device="cpu") -> dict:
    """S2.8 item 2(e): the query-dependence diagnostic on a trained (or, in
    smoke, an untrained-but-forward-capable) composer's own reader."""
    n_gens = composer.n_gens

    def real_state_fn(D):
        tok = si.build_probe_tokens(n_gens, D, device=device)
        composer.eval()
        with torch.no_grad():
            tok_embed = composer.embed_tokens(tok)
            states = composer.states_from_embedding(tok_embed)
        return states[-1]

    report = si.run_query_dependence_gate(
        composer.readout.row_queries, composer.readout.reader, cell["n_h"], real_state_fn,
        prepare_mem=composer.readout.prepare_mem, depths=depths, seed=si.PROBE_SEED, h=composer.h,
    )
    route = si.route_gate_result(report, bos_already_applied=composer.use_bos_row)
    return dict(cell_id=cell["cell_id"], report=report, route=route)


# ---------------------------------------------------------------------------
# THE REAL CELL RUNNER (S2.20 F4). `train_cell_tiny` above is smoke-only;
# this is what a real 57-cell-remainder launch actually calls.
# ---------------------------------------------------------------------------

# S2.7's own text: "if Stage 2's cells need the SAME per-group budgets" as
# Stage 1's Rev-7 pins (S1.30/S1.31) -- reused verbatim from
# run_capability_sep.STEP_BUDGET (imported, not re-derived) rather than
# hand-copied, so a future Stage-1 re-pin cannot silently drift out of sync
# with Stage 2's own runner.
STEP_BUDGET_BY_GROUP = rcs.STEP_BUDGET

# M-D0 (S2.6): hard/disclosed depth split at ceil(0.6 * D_train_max) -- the
# "~5-of-8 ratio S1.27's own resolution empirically landed on, applied here
# as a provisional starting split, explicitly flagged for RECALIBRATION at
# gate time against real data." This build wires the PROFILE machinery
# (every D=1..D_train_max reported, hard/disclosed tagged); it does NOT
# invent a numeric convergence BAR for the hard band -- the design text is
# explicit that bar VALUE is a post-launch recalibration decision, not a
# build-time constant to hardcode.
M_D0_HARD_DEPTH_MAX = math.ceil(0.6 * st.D_TRAIN_MAX)   # = 5


def m_d0_convergence_profile(composer: sc.GroupWordDeltaComposer, name: str, seed: int,
                             device="cpu") -> list[dict]:
    """S2.6 M-D0: per-depth TRAIN-support convergence profile, D=1..
    D_train_max, hard/disclosed split. D=1 is structurally unevaluable via
    the degauging pipeline (S2.20 m4, `Stage2DepthOneCoverageUnsupported`)
    -- reported as an explicit excluded point (mirrors
    `evaluate_arm1_at_depth`'s own D>ARM1_L_MAX convention), not silently
    dropped or crashed on; D=1's own health is covered separately by the
    2(e) query-dependence gate (which DOES probe D=1, via `build_probe_tokens`,
    never through this degauging path)."""
    profile = []
    for D in range(1, st.D_TRAIN_MAX + 1):
        gating = "hard" if D <= M_D0_HARD_DEPTH_MAX else "disclosed"
        if D == 1:
            profile.append(dict(D=D, gating=gating, excluded=True,
                                recovered_frac_90=None, mean_cos=None,
                                note="D=1 structurally unevaluable via the degauging pipeline "
                                     "(S2.20 m4) -- covered by the 2(e) query-dependence gate instead"))
            continue
        s = st.evaluate_composer_at_depth(composer, name, D, seed=seed * 1000 + D, device=device)
        profile.append(dict(D=D, gating=gating, excluded=False,
                            recovered_frac_90=s["recovered_frac_90"], mean_cos=s["mean_cos"]))
    return profile


def d_test_grid_eval(composer: sc.GroupWordDeltaComposer, name: str, seed: int,
                     device="cpu") -> list[dict]:
    """S2.6 M-D1/M-D2: the actual held-out depth-generalization evaluation
    over `stage2_task.D_TEST_GRID` (S2.20 F4) -- what `D_test_results`
    (below) is FOR, making `is_valid_output`'s not-None check on that key
    real instead of vacuous."""
    results = []
    for D in st.D_TEST_GRID:
        s = st.evaluate_composer_at_depth(composer, name, D, seed=seed * 1000 + D, device=device)
        results.append(dict(
            D=D, recovered_frac_90=s["recovered_frac_90"], mean_cos=s["mean_cos"],
            crosscheck_recovered_frac_90=s["crosscheck_recovered_frac_90"],
            crosscheck_mean_cos=s["crosscheck_mean_cos"],
            restricted_effective_rank=s["restricted_effective_rank"],
            near_plateau=D in st.NEAR_PLATEAU_BAND, far_depth=D in st.FAR_DEPTH_BAND,
        ))
    return results


def run_real_cell(cell: dict, results_dir: str, device="cpu", steps: int | None = None,
                  budget_guard: bool = True) -> dict:
    """S2.20 F4 -- THE REAL cell runner (train_cell_tiny is smoke-only, see
    its docstring): `cosine_loss` (the pinned objective, S1.4 -- imported
    from `group_word_encoder` at this module's top, never redefined;
    `stage2_composer.py::composer_scoring_fn` wraps the identical function),
    FINAL-STATE-ONLY supervision (`composer(token_idx)` already reads only
    the terminal state, S2.3), per-batch-fixed-D sampling
    (`stage2_task.sample_train_batch_stage2`, byte-identical in form to
    Stage 1's own scheme), REAL per-group step budgets
    (`STEP_BUDGET_BY_GROUP`, S2.7's Rev-7-pin reuse instruction), the M-D0
    per-depth convergence profile, parameter-fingerprint-protected
    persistence (S2.20 F3) before the gate trusts the composer, the full
    7-depth S2.8 item 2(e) query-dependence gate
    (`run_calibration_gate_for_cell`'s own `si.PROBE_DEPTHS` default, S2.20
    F5 -- no override here either), the +-15% param-match check/report
    (S2.20 F6), and the D_test grid evaluation (writing `D_test_results`,
    S2.20 F4)."""
    name = cell["group"]
    composer = build_cell_composer(cell, device=device)
    opt = torch.optim.Adam(composer.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(cell["seed"] + 1)
    steps_total = steps if steps is not None else STEP_BUDGET_BY_GROUP[name]

    t0 = time.time()
    final_loss = None
    log_every = max(1, steps_total // 20)
    for step in range(1, steps_total + 1):
        batch = st.sample_train_batch_stage2(name, 32, gen, device=device)
        Z = composer(batch["token_idx"])
        loss = sc.composer_scoring_fn(Z, batch["target"])   # cosine_loss, pinned objective (S2.20 F4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
        if budget_guard and step % log_every == 0:
            elapsed_h = (time.time() - t0) / 3600.0
            check_per_cell_projection(elapsed_h, step, steps_total)
    wall_s = time.time() - t0

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = checkpoint_path(results_dir, cell["cell_id"])
    torch.save(composer.state_dict(), ckpt_path)
    # S2.20 F3: fingerprint-protect BEFORE the gate/eval trust this composer.
    disk_fp = assert_fingerprint_matches(composer, ckpt_path, cell["cell_id"], device=device)

    # S2.6 M-D0.
    m_d0_profile = m_d0_convergence_profile(composer, name, seed=cell["seed"], device=device)

    # S2.8 item 2(e), full 7-depth (S2.20 F5 -- no depths= override).
    gate_result = run_calibration_gate_for_cell(cell, composer, device=device)

    # S2.2.2/S2.9 item 7 param-match (S2.20 F6): exact counts reported
    # regardless; assert where the design pins it (every real training cell).
    arm1_ref = GroupWordModel(D_STATE[name], len(generating_set(name)), L_max=st.ARM1_L_MAX, h=32)
    param_match = sc.check_param_match(sc.count_params(composer), sc.count_params(arm1_ref))
    assert param_match["within_tolerance"], (
        f"S2.2.2/S2.20 F6 param-match FAILED for {cell['cell_id']}: composer="
        f"{param_match['composer_params']} Arm1={param_match['arm1_params']} "
        f"delta={param_match['delta_frac'] * 100:+.1f}% (tolerance +-15%)"
    )

    # S2.6 M-D1/M-D2, S2.20 F4: the D_test grid.
    D_test_results = d_test_grid_eval(composer, name, seed=cell["seed"], device=device)

    result = dict(cell_id=cell["cell_id"], group=cell["group"], arm=cell["arm"], n_h=cell["n_h"],
                 seed=cell["seed"], status="completed", steps_completed=steps_total,
                 final_loss=final_loss, wall_clock_s=wall_s, checkpoint_path=ckpt_path,
                 param_fingerprint=disk_fp, m_d0_profile=m_d0_profile,
                 gate_route=gate_result["route"]["route"], gate_report=gate_result["report"],
                 param_match=param_match, D_test_results=D_test_results)
    return result


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def run_calibration_wave(results_dir: str, device="cpu", steps: int = 20,
                         cells: list[dict] | None = None) -> list[dict]:
    """`cells=None` (default, production path): the full 11-cell
    calibration-first set. `cells=<subset>`: S2.20 F3's own smoke needs a
    fast end-to-end exercise of THIS function (not a hand-rolled
    imitation of it) -- a non-breaking optional override lets smoke pass
    1-2 cells instead of paying for all 11 on every CPU smoke run.

    S2.20 F3: fingerprint-protected -- the gate never runs against a
    composer that doesn't match its own just-persisted checkpoint (the
    §2.19 self-caught bug class). S2.20 F5: gates at ALL SEVEN
    `si.PROBE_DEPTHS` (the Rev-1 `(1,8,64)` override removed -- it silently
    skipped 4 of 7 pinned depths, incl. the §2.14-MAJOR-2 norm-accumulation
    band)."""
    if cells is None:
        primary, nh_grid = build_primary_grid(), build_nh_grid()
        cells = build_calibration_set(primary, nh_grid)
    results = []
    for cell in cells:
        def _run(c, _results_dir=results_dir):
            train_result, composer = train_cell_tiny(c, _results_dir, device=device, steps=steps)
            ckpt = train_result["checkpoint_path"]
            # S2.20 F3: hard-fail if the composer about to be gated doesn't
            # match its own just-persisted checkpoint.
            disk_fp = assert_fingerprint_matches(composer, ckpt, c["cell_id"], device=device)
            train_result["param_fingerprint"] = disk_fp
            # gate runs on the ACTUALLY TRAINED composer (never a fresh
            # reinit -- that was a build-time bug caught and fixed before
            # this file was committed, see the git history / final report),
            # and now at ALL SEVEN pinned depths (S2.20 F5).
            gate_result = run_calibration_gate_for_cell(c, composer, device=device)
            train_result["gate_route"] = gate_result["route"]["route"]
            # S2.20 F5 proof-by-run: record how many depths were ACTUALLY
            # gated, so a regression to the old (1,8,64) override is
            # directly observable in the result JSON, not just inferable.
            train_result["n_depths_gated"] = len(gate_result["report"]["per_depth"])
            return train_result
        result = run_cell_resume_safe(cell, results_dir, _run)
        if "gate_route" not in result:
            # resumed from disk: the checkpoint exists, but the gate wasn't
            # re-run this process -- re-run it against the LOADED (not
            # fresh) composer so a resumed run still reports a gate route.
            composer = load_cell_composer(cell, results_dir, device=device)
            ckpt = checkpoint_path(results_dir, cell["cell_id"])
            disk_fp = assert_fingerprint_matches(composer, ckpt, cell["cell_id"], device=device)
            gate_result = run_calibration_gate_for_cell(cell, composer, device=device)
            result["gate_route"] = gate_result["route"]["route"]
            result["param_fingerprint"] = disk_fp
            result["n_depths_gated"] = len(gate_result["report"]["per_depth"])
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# ARM-1 RETRAIN-AND-SAVE UTILITY (S2.20 audit box item 4 / this dispatch's
# own instruction). Stage 1's 58-cell sweep (S1.33 M2 disclosed gap)
# trained-then-discarded every model, so `stage2_task.py::load_arm1_checkpoint`
# has nothing to load. This retrains the 5 unconstrained-arm CALIBRATION
# cells (`run_capability_sep.calibration_wave()`, one per group, seed=0 --
# "the 5 Stage-1 unconstrained cells" this dispatch names) at their EXACT
# Rev-7 pinned per-group step budgets (`run_capability_sep.STEP_BUDGET`),
# mirroring `train_and_eval_cell`'s own seeding/model-construction/training-
# loop procedure EXACTLY -- IMPORTED calls (`rcs.GroupWordModel` via
# `group_word_encoder`, `group_task.sample_train_batch`, `cosine_loss`,
# `groups.group_seed_salt`, `force_rank_arms.force_rank_grid` via `rcs`),
# NOT a reimplementation of Stage 1's pinned recipe -- the ONE addition is
# `torch.save`, the one thing `train_and_eval_cell` never does (it doesn't
# even return the model object). FILE OWNERSHIP: run_capability_sep.py is
# NOT edited, only imported from and called.
#
# PRICE (reported here, NOT executed by this build/fix pass): 5 cells at
# Stage 1's own §1.6 measured rate (0.0179 GPU-h/cell at the 8,000-step
# anchor) scaled by each group's own Rev-7 step-budget multiplier (S3/S5 at
# 1x = 8,000 steps, S4/A5 at 2.5x = 20,000, A6 at 5x = 40,000):
#   0.0179 * (1 + 1 + 2.5 + 2.5 + 5) = 0.0179 * 12 = 0.2148 GPU-h (~0.21
#   GPU-h), comfortably inside this dispatch's own "~0.5 GPU-h class"
#   ceiling estimate -- cheap enough to fold into any GPU-hot window
#   without a dedicated ledger line.
# ---------------------------------------------------------------------------

ARM1_RETRAIN_PRICE_GPU_H = round(0.0179 * sum(rcs.STEP_BUDGET[g] / rcs.STEP_BUDGET["S3"]
                                              for g in ("S3", "S4", "A5", "S5", "A6")), 4)


def arm1_checkpoint_path(results_dir: str, group: str, seed: int) -> str:
    return os.path.join(results_dir, f"arm1__{group}__seed{seed}.pt")


def retrain_and_save_arm1_checkpoints(results_dir: str, device: str = "cpu",
                                      steps_override: int | None = None) -> list[dict]:
    """Retrains + persists the 5 Stage-1 unconstrained-arm calibration
    cells (one per group, seed=0) so `stage2_task.py::load_arm1_checkpoint`
    has something real to load for the D_test grid. See module-level
    comment above for the price and the "imported, not reimplemented"
    provenance of every call below."""
    cells = rcs.calibration_wave()   # 5 cells: unconstrained arm, seed=0, one per group
    assert len(cells) == 5, f"expected 5 Stage-1 unconstrained calibration cells, got {len(cells)}"
    results = []
    for cell in cells:
        name = cell["group"]
        steps = steps_override if steps_override is not None else cell["steps"]
        k = cell["force_rank_k"] if "force_rank_k" in cell else rcs.force_rank_grid(name)[cell["arm"]]
        target_padding = cell.get("target_padding", "eye")

        # EXACT mirror of run_capability_sep.py::train_and_eval_cell's own
        # seeding/model-construction (imported constants/functions, not
        # re-derived) -- the only addition is CAPTURING `model` so this
        # pass can persist its state_dict, which train_and_eval_cell itself
        # never returns or saves.
        torch.manual_seed(cell["seed"])
        d_state = D_STATE[name]
        n_gens = len(generating_set(name))
        model = GroupWordModel(d_state, n_gens, L_max=st.ARM1_L_MAX, h=32).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        gen = torch.Generator().manual_seed(cell["seed"] + group_seed_salt(name))

        t0 = time.time()
        for step in range(1, steps + 1):
            batch = rcs.sample_train_batch(name, 256, gen, device=device, target_padding=target_padding)
            Z = model.encode(batch["token_idx"], force_rank_k=k)
            loss = cosine_loss(Z, batch["target"])
            opt.zero_grad()
            loss.backward()
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        wall_s = time.time() - t0

        os.makedirs(results_dir, exist_ok=True)
        ckpt_path = arm1_checkpoint_path(results_dir, name, cell["seed"])
        torch.save(model.state_dict(), ckpt_path)

        eval_seed = cell["seed"] + 10_000
        scores = readout.run_subspace_restriction_pipeline(model, name, base_seed=eval_seed,
                                                            device=device, force_rank_k=k)
        results.append(dict(cell_id=cell["cell_id"], group=name, seed=cell["seed"],
                            steps_completed=steps, wall_clock_s=wall_s, checkpoint_path=ckpt_path,
                            mean_cos=scores["mean_cos"], recovered_frac_90=scores["recovered_frac_90"]))
        print(f"  [arm1-retrain] {name} seed={cell['seed']} steps={steps} -> {ckpt_path}  "
              f"mean_cos={scores['mean_cos']:.4f}")
    return results


def smoke():
    print("=" * 88)
    print("  stage2_run.py SMOKE -- grid construction, budget guards, resume-safety, calibration wave")
    print("=" * 88)

    primary, nh_grid = build_primary_grid(), build_nh_grid()
    calib = build_calibration_set(primary, nh_grid)
    total = len(primary) + len(nh_grid)
    print(f"  primary grid: {len(primary)} cells (expect 50)")
    print(f"  n_h grid: {len(nh_grid)} cells (expect 18)")
    print(f"  total new training cells: {total} (expect 68)")
    print(f"  calibration set: {len(calib)} cells (expect 11)")
    assert len(primary) == 50 and len(nh_grid) == 18 and total == 68 and len(calib) == 11

    print(f"\n  budget: PER_CELL_ABORT_CEILING={PER_CELL_ABORT_CEILING} GPU-h/cell "
          f"(1.5x {PLANNING_RATE_HI})")
    assert abs(PER_CELL_ABORT_CEILING - 0.081) < 1e-9

    print("\n  Stage-2 ledger projection (NEGATIVE-then-POSITIVE):")
    ok_proj = check_stage2_sweep_projection(real_rate_per_cell=0.03)
    print(f"    real_rate=0.03 GPU-h/cell -> projected={ok_proj['projected_total']} <= 25.0: PASS")
    raised = False
    try:
        check_stage2_sweep_projection(real_rate_per_cell=1.0)
    except Stage2BudgetAbort as e:
        raised = True
        print(f"    real_rate=1.0 GPU-h/cell -> Stage2BudgetAbort raised as expected: {str(e)[:90]}...")
    assert raised, "Stage2BudgetAbort has no teeth -- an absurd rate did not trip the ledger breaker"

    print("\n  per-cell budget guard (NEGATIVE-then-POSITIVE):")
    ok = check_per_cell_projection(elapsed_h=0.001, steps_done=10, steps_total=20)
    print(f"    small elapsed_h -> ok={ok['ok']} projected={ok['projected']}")
    assert ok["ok"]
    raised = False
    try:
        check_per_cell_projection(elapsed_h=1.0, steps_done=1, steps_total=20)
    except PerCellBudgetAbort as e:
        raised = True
        print(f"    huge elapsed_h at 1/20 steps -> PerCellBudgetAbort raised as expected: {str(e)[:90]}...")
    assert raised, "PerCellBudgetAbort has no teeth"

    print("\n  resume-safety (tiny synthetic cell, CPU, steps=6):")
    import tempfile
    with tempfile.TemporaryDirectory() as results_dir:
        cell = calib[0]

        def _run_and_ckpt(c, _rd=results_dir):
            result, _composer = train_cell_tiny(c, _rd, steps=6)
            return result

        t0 = time.time()
        r1 = run_cell_resume_safe(cell, results_dir, _run_and_ckpt)
        dt1 = time.time() - t0
        t0 = time.time()
        r2 = run_cell_resume_safe(cell, results_dir, _run_and_ckpt)
        dt2 = time.time() - t0
        print(f"    first run: {dt1:.3f}s  second run (resume/skip): {dt2:.3f}s")
        assert dt2 < dt1, "resume pass took as long as the first -- resume-safety broken"
        assert r1["cell_id"] == r2["cell_id"] == cell["cell_id"]
        assert os.path.exists(checkpoint_path(results_dir, cell["cell_id"])), \
            "training completed but no checkpoint was persisted -- the last-K-window control " \
            "(S2.9 item 4) needs this on disk"

        # corrupted-output re-run: truncated JSON must NOT be treated as done.
        path = cell_output_path(results_dir, cell["cell_id"])
        with open(path, "w") as f:
            f.write('{"cell_id": "' + cell["cell_id"] + '", "status": "completed"')  # truncated, invalid JSON
        assert not is_valid_output(path), "corrupted output was accepted as valid -- resume-safety has no teeth"
        r3 = run_cell_resume_safe(cell, results_dir, _run_and_ckpt)
        assert r3["status"] == "completed", "did not properly re-run after a corrupted output"
        print("    corrupted (truncated) output correctly triggers a re-run, not a false-skip  OK")

        print("\n  last-K-window eval-time reload from the JUST-PERSISTED checkpoint (S2.9 item 4):")
        reloaded = load_cell_composer(cell, results_dir)
        d_state = D_STATE[cell["group"]]
        tok = torch.randint(0, reloaded.n_gens, (4, 8))
        Z_full = reloaded(tok, reset_every=None)
        Z_trunc = reloaded(tok, reset_every=4)
        assert Z_full.shape == Z_trunc.shape == (4, d_state, d_state)
        assert not torch.allclose(Z_full, Z_trunc), \
            "reloaded checkpoint's truncated-eval read is identical to the full read -- suspicious"
        print("    reloaded composer forward()s correctly with and without reset_every  OK")

    print("\n  one full calibration-cell pipeline (composer train + query-dependence gate), CPU, tiny:")
    torch.manual_seed(0)
    cell = dict(cell_id="smoke__S3__arm3_beta02__nh2__seed0", group="S3", arm="arm3_beta02", n_h=2, seed=0)
    with tempfile.TemporaryDirectory() as results_dir2:
        train_result, trained_composer = train_cell_tiny(cell, results_dir2, steps=6)
        gate_result = run_calibration_gate_for_cell(cell, trained_composer, depths=(1, 8))
    print(f"    train status={train_result['status']}  gate route={gate_result['route']['route']}")
    assert train_result["status"] == "completed"
    assert gate_result["route"]["route"] in ("pass", "apply_bos_fix_rerun_all_11",
                                             "instrument_defect", "mechanism_diagnostic_required")

    print("\n  target-rank unit test (S1.33 [LEARN], invoked as a stage2_run smoke section):")
    st._test_target_rank_matches_necessity()

    print("\n  S2.20 F3 -- parameter-fingerprint protection (positive + NEGATIVE mutation):")
    with tempfile.TemporaryDirectory() as fp_dir:
        torch.manual_seed(0)
        fp_cell = dict(cell_id="fp_test__S3__arm3_beta02__nh2__seed0", group="S3",
                       arm="arm3_beta02", n_h=2, seed=0)
        train_result, trained_composer = train_cell_tiny(fp_cell, fp_dir, steps=4)
        ckpt = train_result["checkpoint_path"]
        fp = assert_fingerprint_matches(trained_composer, ckpt, fp_cell["cell_id"])
        print(f"    positive: trained composer fingerprint matches checkpoint ({fp[:12]}...)  OK")

        # NEGATIVE (S2.20 F3 mutation): reproduce the EXACT §2.19 self-caught bug --
        # probe a FRESH, differently-seeded composer instead of the trained one --
        # and confirm the fingerprint check now catches it.
        fresh = build_cell_composer({**fp_cell, "seed": fp_cell["seed"] + 99})
        raised_fp = False
        try:
            assert_fingerprint_matches(fresh, ckpt, fp_cell["cell_id"])
        except ParamFingerprintMismatch as e:
            raised_fp = True
            print(f"    negative (untrained/fresh-composer mutation) CAUGHT: {str(e)[:90]}...")
        assert raised_fp, (
            "S2.20 F3 regression: probing a FRESH composer instead of the trained one was NOT "
            "caught by the parameter-fingerprint check -- no teeth"
        )

    print("\n  S2.20 F3/F5 -- run_calibration_wave END-TO-END (small subset via the new `cells=` "
          "override, fingerprint-protected, gates at all 7 si.PROBE_DEPTHS):")
    with tempfile.TemporaryDirectory() as wave_dir:
        mini_cell = dict(cell_id="wave_smoke__S3__arm3_beta02__nh2__seed0", group="S3",
                         arm="arm3_beta02", n_h=2, seed=0)
        wave_results = run_calibration_wave(wave_dir, steps=4, cells=[mini_cell])
        assert len(wave_results) == 1
        assert wave_results[0]["status"] == "completed"
        assert "param_fingerprint" in wave_results[0], "run_calibration_wave did not report a fingerprint"
        assert wave_results[0]["gate_route"] in ("pass", "apply_bos_fix_rerun_all_11",
                                                 "instrument_defect", "mechanism_diagnostic_required")
        assert wave_results[0]["n_depths_gated"] == len(si.PROBE_DEPTHS) == 7, (
            f"S2.20 F5 REGRESSION: run_calibration_wave gated {wave_results[0]['n_depths_gated']} "
            f"depths, not all {len(si.PROBE_DEPTHS)} pinned si.PROBE_DEPTHS -- the (1,8,64) "
            f"override is back"
        )
        print(f"    cell={wave_results[0]['cell_id']}  fingerprint={wave_results[0]['param_fingerprint'][:12]}...  "
              f"gate_route={wave_results[0]['gate_route']}  n_depths_gated={wave_results[0]['n_depths_gated']}")
        # resume path must ALSO carry a fingerprint (re-run branch, not just the first-run branch).
        wave_results_resumed = run_calibration_wave(wave_dir, steps=4, cells=[mini_cell])
        assert "param_fingerprint" in wave_results_resumed[0], \
            "resume branch of run_calibration_wave did not report a fingerprint"
        print(f"    resume pass also reports fingerprint={wave_results_resumed[0]['param_fingerprint'][:12]}...  OK")

    print("\n  S2.20 F4 -- run_real_cell END-TO-END (tiny steps_override, real cosine_loss/M-D0/"
          "7-depth-gate/param-match/D_test-grid wiring, NOT a real launch):")
    with tempfile.TemporaryDirectory() as real_dir:
        real_cell = dict(cell_id="real_smoke__S3__arm3_beta02__nh2__seed0", group="S3",
                         arm="arm3_beta02", n_h=2, seed=0)
        real_result = run_real_cell(real_cell, real_dir, steps=6, budget_guard=False)
        assert real_result["status"] == "completed"
        assert "param_fingerprint" in real_result
        assert "m_d0_profile" in real_result and len(real_result["m_d0_profile"]) == st.D_TRAIN_MAX
        assert real_result["m_d0_profile"][0]["D"] == 1 and real_result["m_d0_profile"][0]["excluded"] is True, \
            "M-D0 profile's D=1 point must be reported as excluded (S2.20 m4), not crash or silently skip"
        assert real_result["m_d0_profile"][1]["D"] == 2 and real_result["m_d0_profile"][1]["excluded"] is False
        assert "param_match" in real_result and real_result["param_match"]["within_tolerance"]
        assert "D_test_results" in real_result and len(real_result["D_test_results"]) == len(st.D_TEST_GRID)
        assert real_result["gate_route"] in ("pass", "apply_bos_fix_rerun_all_11",
                                             "instrument_defect", "mechanism_diagnostic_required")
        # is_valid_output's D_test_results-not-None check is now REAL (S2.20 F4) -- prove it by
        # round-tripping this cell's actual JSON-serialized output through the on-disk check.
        out_path = cell_output_path(real_dir, real_cell["cell_id"])
        os.makedirs(real_dir, exist_ok=True)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(real_result, f, indent=2, default=str)
        os.replace(tmp_path, out_path)
        assert is_valid_output(out_path), "a genuine run_real_cell output was rejected as invalid"
        # NEGATIVE: corrupt D_test_results to None and confirm is_valid_output now correctly rejects it.
        corrupted = dict(real_result)
        corrupted["D_test_results"] = None
        with open(tmp_path, "w") as f:
            json.dump(corrupted, f, indent=2, default=str)
        os.replace(tmp_path, out_path)
        assert not is_valid_output(out_path), (
            "S2.20 F4 REGRESSION: is_valid_output did not reject a D_test_results=None output -- "
            "the check is vacuous again"
        )
        print(f"    status={real_result['status']}  param_match delta={real_result['param_match']['delta_frac'] * 100:+.1f}%  "
              f"D_test points={len(real_result['D_test_results'])}  gate_route={real_result['gate_route']}  "
              f"is_valid_output: genuine=True, D_test_results=None-corrupted=False  OK")

    print("\n  S2.20 ARM-1 RETRAIN utility -- WIRING smoke only (tiny steps_override; the REAL "
          f"~{ARM1_RETRAIN_PRICE_GPU_H} GPU-h launch is NOT run here, per this dispatch):")
    with tempfile.TemporaryDirectory() as arm1_dir:
        arm1_results = retrain_and_save_arm1_checkpoints(arm1_dir, steps_override=3)
        assert len(arm1_results) == 5, f"expected 5 Arm-1 checkpoints, got {len(arm1_results)}"
        for r in arm1_results:
            assert os.path.exists(r["checkpoint_path"]), f"{r['cell_id']}: no checkpoint on disk"
        # the checkpoint stage2_task.py::load_arm1_checkpoint expects must actually load.
        loaded = st.load_arm1_checkpoint(arm1_results[0]["checkpoint_path"],
                                         D_STATE[arm1_results[0]["group"]],
                                         len(generating_set(arm1_results[0]["group"])))
        print(f"    5/5 checkpoints saved; load_arm1_checkpoint round-trips the first "
              f"({arm1_results[0]['group']}) OK  |  priced GPU-h={ARM1_RETRAIN_PRICE_GPU_H}")

    print("\n" + "=" * 88 + "\n  stage2_run.py SMOKE PASSED\n" + "=" * 88)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibration-only", action="store_true")
    ap.add_argument("--results-dir", default="stage2_results")
    ap.add_argument("--retrain-arm1", action="store_true",
                    help="ARM-1 RETRAIN-AND-SAVE utility (S2.20 box item 4 / this dispatch's own "
                         f"instruction): retrain the 5 Stage-1 unconstrained calibration cells at "
                         f"their Rev-7 pinned per-group step budgets and SAVE checkpoints (Stage 1 "
                         f"never persisted them). Priced at ~{ARM1_RETRAIN_PRICE_GPU_H} GPU-h "
                         f"(0.0179 GPU-h/cell x 12 group-budget-multiplier-units, S1.6 measured "
                         f"rate). NOT launched by this build/fix pass.")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    if args.retrain_arm1:
        if os.environ.get(PI_SIGNOFF_VAR) != "1":
            raise RuntimeError(
                f"{PI_SIGNOFF_VAR}=1 required before the ARM-1 retrain-and-save pass (same "
                f"PI-signoff convention as the real cell launch below). Priced at "
                f"~{ARM1_RETRAIN_PRICE_GPU_H} GPU-h (S2.20 box item 4) -- not launched by this "
                f"build/fix pass."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = retrain_and_save_arm1_checkpoints(args.results_dir, device=device)
        print(f"ARM-1 retrain-and-save: {len(results)} checkpoints -> {args.results_dir}")
        return

    if os.environ.get(PI_SIGNOFF_VAR) != "1":
        raise RuntimeError(
            f"{PI_SIGNOFF_VAR}=1 required before any GPU cell (mirrors run_capability_sep.py's "
            f"own gate). Not set automatically by this script. This build agent does not launch "
            f"real cells -- an independent audit follows before any real dispatch."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_calibration_wave(args.results_dir, device=device, steps=20)
    print(f"calibration wave: {len(results)} cells run/skipped -> {args.results_dir}")
    if not args.calibration_only:
        print("real 57-cell remainder launch is NOT wired in this build -- gated on the "
              "calibration wave's own query-dependence PASS + real-rate re-derivation, "
              "S2.8 items 2-3.")


if __name__ == "__main__":
    main()
