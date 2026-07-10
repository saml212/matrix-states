"""CAPABILITY_SEPARATION_DESIGN.md S2.6/S2.9 item 4/S2.20 box item 7 -- the
Stage-2 harvest analysis script, mirroring `tost_analysis.py`'s conventions
(pinned constants at module top, small pure statistical/verdict functions,
disclosed operationalizations where the design text leaves a numeric edge
unpinned, a `if __name__ == "__main__":` unit-test block exercising every
verdict branch on synthetic data) -- REQUIRED pre-sweep (S2.20 box item 7 /
S2.23's own NEXT line: "WRITE the harvest analysis script (still
outstanding, required pre-sweep)").

Consumes the per-cell JSON outputs `run_real_cell`/`run_calibration_wave_real`
write (`stage2_run.py::cell_output_path`, `{cell_id}.json`) and computes:

  - M-D1 (S2.6): accuracy-vs-depth curve, recovered_frac_90 vs D, per
    (group, arm, n_h), aggregated across seeds.
  - M-D2 (S2.6): rank-vs-depth curve, restricted_effective_rank vs D, same
    grouping -- corroborating-only, never independently decisive (mirrors
    Stage 1's own M1 precedent).
  - M-D3 (S2.1/S2.6): the pre-registered CONFIRM/FALSIFY/INCONCLUSIVE
    verdict at the mid (~4x D_train_max, D=32, corroborating/secondary) and
    far (~8x D_train_max, D=64, PRIMARY decisive) checkpoints, including the
    S2.9 item 4 last-K shortcut-control downgrade trigger
    (`contender_mean - control_mean <= max(2*sigma_seed, 0.05)`,
    `sigma_seed` computed ddof=1 over the contender's own n=5 seeds).
  - Per-cell config-match verification against an INDEPENDENT-LITERAL
    manifest (the A3 precedent, `2026-07-09_m3fix_harvest`'s
    `analyze_m3fix_*_harvest.py` convention: re-derive the expected grid
    from THIS design's own S2.5/S2.7 spec, not by importing
    `stage2_run.build_primary_grid`/`build_nh_grid`/`build_calibration_set`,
    so a drift between the run code's actual grid and the design text is
    independently catchable rather than silently inherited from the same
    source the check is meant to audit).

This module does NOT train anything and does NOT require a GPU or `torch`
for the harvest/verdict path -- it is a pure JSON + numpy post-processor,
runnable standalone on any machine holding a pulled results directory.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Pinned constants -- INDEPENDENT-LITERAL re-derivations from
# CAPABILITY_SEPARATION_DESIGN.md S2.4/S2.5/S2.6/S2.7, not imported from
# `stage2_run.py`/`stage2_task.py` (the A3 precedent: this file is the
# thing checking those modules' actual output against the design spec, so
# it must not share their possibly-drifted source of truth for the
# structural constants below). `groups.GROUP_NAMES` IS imported -- the
# group family itself is Stage-1 file-ownership, unchanged and re-verified
# upstream (S1.3), not a Stage-2 grid-construction detail this script exists
# to audit.
# ---------------------------------------------------------------------------

from groups import GROUP_NAMES  # ["S3", "S4", "A5", "S5", "A6"], S1.2's pinned order

D_TRAIN_MAX = 8
D_TEST_GRID = (9, 10, 12, 14, 16, 20, 24, 32, 48, 64)
PROBE_DEPTHS = (1, 2, 4, 8, 16, 32, 64)          # S2.8 item 2(e)
MID_DEPTH = 32                                    # ~4x D_train_max, S2.6 secondary-decisive
FAR_DEPTH = 64                                    # ~8x D_train_max, S2.6 PRIMARY decisive
assert MID_DEPTH in D_TEST_GRID and FAR_DEPTH in D_TEST_GRID

ARMS = ("arm2_beta01", "arm3_beta02")
CONTENDER_ARM = "arm3_beta02"   # beta in [0,2] -- S2.2.4 Arm 3
BASELINE_ARM = "arm2_beta01"    # beta in [0,1] -- S2.2.4 Arm 2

PRIMARY_SEEDS = 5
NH_GRID_GROUPS = ("S5", "A6")
NH_GRID_VALUES = (1, 2, 4)
NH_GRID_SEEDS = 3

# S2.2.4: S5's DECISIVE Arm-3 configuration is n_h=4 (DeltaProduct's own
# published requirement), not the n_h=2 primary-grid default every other
# group uses -- the promoted (S5, Arm3, n_h=4) calibration cell (S2.5), not
# the base (S5, Arm3, n_h=2) cell, is what M-D3 reads for S5's contender leg.
DECISIVE_CONTENDER_NH = {g: (4 if g == "S5" else 2) for g in GROUP_NAMES}

# S1.30/S1.31 Rev-7 pinned per-group step budgets (S2.7's "reused verbatim"
# instruction) -- re-derived here independently of `stage2_run.STEP_BUDGET_BY_GROUP`
# (which itself just imports `run_capability_sep.STEP_BUDGET`) so a drift in
# EITHER module is independently catchable, not silently inherited.
STEP_BUDGET_BY_GROUP_INDEPENDENT = {"S3": 8_000, "S5": 8_000, "S4": 20_000, "A5": 20_000, "A6": 40_000}

CONFIRM_FRAC_OF_CEILING = 0.9        # S2.1/S2.6: contender >= 90% of its own D_train ceiling
ARM2_COLLAPSE_FRAC_OF_CEILING = 0.5  # S2.1/S2.6: Arm 2 drops BELOW 50%-of-ceiling on S5/A6
ARM2_GATING_GROUPS = ("S5", "A6")    # minimum gating pair (S2.1/S2.6); A5 reported, NOT gating
LAST_K_TRIGGER_FLOOR = 0.05          # S2.9 item 4: max(2*sigma_seed, 0.05)


# ---------------------------------------------------------------------------
# Independent-literal manifest re-derivation (the A3 precedent).
# ---------------------------------------------------------------------------

def expected_primary_cell_ids() -> set[str]:
    """5 groups x 2 beta-arms x n=5 seeds, n_h=2 default -- 50 cells (S2.5)."""
    return {f"{g}__{a}__nh2__seed{s}"
            for g in GROUP_NAMES for a in ARMS for s in range(PRIMARY_SEEDS)}


def expected_nh_grid_cell_ids() -> set[str]:
    """{S5,A6} x n_h in {1,2,4} x n=3 seeds, Arm 3 ONLY -- 18 cells (S2.5)."""
    return {f"{g}__arm3_beta02__nh{n}__seed{s}"
            for g in NH_GRID_GROUPS for n in NH_GRID_VALUES for s in range(NH_GRID_SEEDS)}


def expected_calibration_cell_ids() -> set[str]:
    """The 11-cell calibration-first set (S2.5/S2.8 item 2): the 10 base
    (group, arm) cells at n_h=2/seed0, PLUS the promoted (S5, Arm3, n_h=4,
    seed0) cell -- drawn from the two grids above, zero incremental cells."""
    base = {c for c in expected_primary_cell_ids() if c.endswith("__seed0")}
    assert len(base) == 10, f"independent-literal base calibration set: expected 10, got {len(base)}"
    promoted = {"S5__arm3_beta02__nh4__seed0"}
    calib = base | promoted
    assert len(calib) == 11, f"independent-literal calibration manifest: expected 11, got {len(calib)}"
    return calib


def expected_full_grid_cell_ids() -> set[str]:
    """Total 68 new training cells (S2.5): 50 primary + 18 n_h-grid."""
    full = expected_primary_cell_ids() | expected_nh_grid_cell_ids()
    assert len(full) == 68, f"independent-literal full grid: expected 68, got {len(full)}"
    return full


def cell_id_to_config(cell_id: str) -> dict:
    """Parse `{group}__{arm}__nh{n_h}__seed{seed}` back into fields, so a
    per-cell config-match check has an independent ground truth (the
    cell_id's own encoding) rather than trusting only the JSON body's
    (possibly-corrupted-in-transit) fields against each other."""
    parts = cell_id.split("__")
    assert len(parts) == 4, f"malformed cell_id {cell_id!r} (expected group__arm__nh{{N}}__seed{{S}})"
    group, arm, nh_tag, seed_tag = parts
    assert nh_tag.startswith("nh"), f"malformed cell_id {cell_id!r}: n_h tag {nh_tag!r}"
    assert seed_tag.startswith("seed"), f"malformed cell_id {cell_id!r}: seed tag {seed_tag!r}"
    return dict(group=group, arm=arm, n_h=int(nh_tag[2:]), seed=int(seed_tag[4:]))


def verify_config_match(cell: dict) -> list[str]:
    """A3 precedent, generalized from the m3fix harvest's own per-cell
    config-match verification: cross-checks a loaded cell JSON against (a)
    its own cell_id encoding, (b) the independently-pinned per-group step
    budget, (c) the expected D_test/probe-depth/m_d0 shapes. Returns a list
    of mismatch strings (empty == clean)."""
    problems = []
    cid = cell.get("cell_id", "<missing cell_id>")
    try:
        expected = cell_id_to_config(cid)
    except AssertionError as e:
        return [str(e)]

    for key in ("group", "arm", "n_h", "seed"):
        if cell.get(key) != expected[key]:
            problems.append(f"{cid}: field {key}={cell.get(key)!r} != cell_id-encoded {expected[key]!r}")

    if cell.get("runner") != "real":
        problems.append(f"{cid}: runner={cell.get('runner')!r} != 'real' (strict_real config-match)")
    if cell.get("status") != "completed":
        problems.append(f"{cid}: status={cell.get('status')!r} != 'completed'")

    group = expected["group"]
    expected_steps = STEP_BUDGET_BY_GROUP_INDEPENDENT.get(group)
    if expected_steps is not None and cell.get("steps_completed") != expected_steps:
        problems.append(f"{cid}: steps_completed={cell.get('steps_completed')!r} != "
                        f"Rev-7 pinned {expected_steps} for group {group!r}")
    if cell.get("n_skipped_steps", 0) not in (0, None) and cell.get("status") == "completed":
        # non-zero skips are disclosed, not fatal -- flagged, not a hard mismatch.
        problems.append(f"{cid}: DISCLOSED n_skipped_steps={cell['n_skipped_steps']} (non-fatal, report)")

    dtr = cell.get("D_test_results")
    if dtr is None:
        problems.append(f"{cid}: D_test_results is None")
    else:
        got_d = {row["D"] for row in dtr}
        if got_d != set(D_TEST_GRID):
            problems.append(f"{cid}: D_test_results D-set {sorted(got_d)} != D_TEST_GRID {sorted(D_TEST_GRID)}")

    m_d0 = cell.get("m_d0_profile")
    if not m_d0:
        problems.append(f"{cid}: m_d0_profile missing/empty")
    else:
        got_d = {row["D"] for row in m_d0}
        if got_d != set(range(1, D_TRAIN_MAX + 1)):
            problems.append(f"{cid}: m_d0_profile D-set {sorted(got_d)} != 1..{D_TRAIN_MAX}")
        d1_rows = [row for row in m_d0 if row["D"] == 1]
        if d1_rows and not d1_rows[0].get("excluded"):
            problems.append(f"{cid}: m_d0_profile D=1 not marked excluded (S2.20 m4)")

    gate_report = cell.get("gate_report")
    if not gate_report or "per_depth" not in gate_report:
        problems.append(f"{cid}: gate_report/per_depth missing")
    else:
        got_d = {row["D"] for row in gate_report["per_depth"]}
        if got_d != set(PROBE_DEPTHS):
            problems.append(f"{cid}: gate_report per_depth D-set {sorted(got_d)} != PROBE_DEPTHS "
                            f"{sorted(PROBE_DEPTHS)} (S2.20 F5 -- all 7 depths must be gated)")

    return problems


def verify_manifest(loaded: dict[str, dict], expected_ids: set[str]) -> dict:
    """Cell-id-SET-exact verification (the A3 precedent) plus per-cell
    config-match, aggregated. Never silently proceeds on a partial pull."""
    got_ids = set(loaded.keys())
    missing = expected_ids - got_ids
    unexpected = got_ids - expected_ids
    per_cell_problems = {cid: verify_config_match(loaded[cid]) for cid in sorted(got_ids & expected_ids)}
    dirty = {cid: probs for cid, probs in per_cell_problems.items() if probs}
    clean = bool(not missing and not unexpected and not dirty)
    return dict(clean=clean, n_expected=len(expected_ids), n_loaded=len(got_ids),
               missing=sorted(missing), unexpected=sorted(unexpected), dirty_cells=dirty)


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------

def load_cell_results(results_dir: str, cell_ids: set[str] | None = None) -> dict[str, dict]:
    """Loads every `{cell_id}.json` under `results_dir` (S2.20/S2.22:
    `stage2_run.cell_output_path`'s own naming, `{cell_id}.json`), restricted
    to `cell_ids` if given. Does NOT raise on missing cells itself -- callers
    that need "raise if incomplete" should follow up with `verify_manifest`
    and check `clean`/`missing`, mirroring the m3fix harvest's own
    verify-then-report (not verify-or-crash) convention, so a partial pull
    is REPORTED, not silently hidden by an early exception."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                continue
        cid = d.get("cell_id") or os.path.splitext(os.path.basename(path))[0]
        if cell_ids is not None and cid not in cell_ids:
            continue
        out[cid] = d
    return out


# ---------------------------------------------------------------------------
# M-D0 pass-through: the contender's own in-support ceiling (the denominator
# of every M-D3 "X%-of-own-ceiling" bar). No numeric convergence BAR is
# invented here for the hard band (D<=5) -- the design text is explicit that
# bar VALUE is a post-launch recalibration decision (S2.6), not a build-time
# constant; this script only reads the PROFILE, never asserts a pass/fail on
# it beyond what M-D3 itself needs (the D=D_train_max ceiling value).
# ---------------------------------------------------------------------------

def own_d_train_ceiling(cell: dict) -> float | None:
    for row in cell.get("m_d0_profile", []):
        if row["D"] == D_TRAIN_MAX and not row.get("excluded"):
            return row["recovered_frac_90"]
    return None


# ---------------------------------------------------------------------------
# M-D1 -- accuracy-vs-depth curve; M-D2 -- rank-vs-depth curve
# (corroborating-only, never independently decisive, S2.6).
# ---------------------------------------------------------------------------

def _curve(cells: list[dict], field: str) -> dict[int, dict]:
    by_d: dict[int, list[float]] = {}
    for cell in cells:
        for row in cell["D_test_results"]:
            by_d.setdefault(row["D"], []).append(row[field])
    curve = {}
    for D, vals in sorted(by_d.items()):
        arr = np.asarray(vals, dtype=float)
        curve[D] = dict(mean=float(arr.mean()), std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                        n=len(arr), values=vals)
    return curve


def m_d1_curve(cells: list[dict]) -> dict[int, dict]:
    """Accuracy-vs-depth: recovered_frac_90 by D, aggregated over whatever
    seeds `cells` holds (typically all cells sharing one (group, arm, n_h))."""
    return _curve(cells, "recovered_frac_90")


def m_d2_curve(cells: list[dict]) -> dict[int, dict]:
    """Rank-vs-depth: restricted_effective_rank by D. Corroborating-only."""
    return _curve(cells, "restricted_effective_rank")


# ---------------------------------------------------------------------------
# M-D3 -- pre-registered CONFIRM/FALSIFY/INCONCLUSIVE margins (S2.1/S2.6).
# ---------------------------------------------------------------------------

def group_mean_at_depth(cells: list[dict], D: int) -> float | None:
    vals = [row["recovered_frac_90"] for c in cells for row in c["D_test_results"] if row["D"] == D]
    return float(np.mean(vals)) if vals else None


def group_ceiling(cells: list[dict]) -> float | None:
    vals = [v for v in (own_d_train_ceiling(c) for c in cells) if v is not None]
    return float(np.mean(vals)) if vals else None


def sigma_seed(cells: list[dict], D: int) -> float:
    """S2.9 item 4: the CONTENDER's own seed-to-seed SAMPLE std (ddof=1),
    computed at the decisive depth. Pinned ddof=1 (S2.14 minor -- the
    population/ddof=0 form is ~11% smaller at n=5 and would silently
    NARROW the trigger band, an anti-conservative drift for a shortcut
    control)."""
    vals = [row["recovered_frac_90"] for c in cells for row in c["D_test_results"] if row["D"] == D]
    arr = np.asarray(vals, dtype=float)
    return float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def last_k_downgrade_fires(contender_mean: float, control_mean: float, seed_values: list[float]) -> dict:
    """S2.9 item 4's exact pinned trigger: fires iff
    `contender_mean - control_mean <= max(2*sigma_seed, 0.05)`, where
    `sigma_seed` is the ddof=1 sample std of the CONTENDER's own per-seed
    values at the decisive point (n=5 seeds nominal). Pure function --
    `control_mean`/`seed_values` are supplied by the caller from whichever
    source is live: the FIRST-level control is EVAL-TIME truncation of the
    already-trained Arm-3 checkpoint (S2.9 item 4, zero new training cells,
    computed by re-loading the checkpoint and re-running the forward pass
    with `reset_every=D_train_max` -- out of THIS pure-JSON harvest's own
    scope, since it requires `torch`+the checkpoint file, not just the
    result JSON; wire it from a caller that has loaded the checkpoint). If
    that first-level control lands within the trigger band, S2.9 item 4
    escalates to a MANDATORY trained last-K instance (15 new cells,
    `{group}__arm3_beta02_lastk__nh2__seed{s}`-style cell_ids once launched)
    -- this same function re-used, unchanged, against that cell's own
    D_test_results once it exists."""
    arr = np.asarray(seed_values, dtype=float)
    sigma = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    threshold = max(2 * sigma, LAST_K_TRIGGER_FLOOR)
    gap = contender_mean - control_mean
    fires = bool(gap <= threshold)
    return dict(contender_mean=contender_mean, control_mean=control_mean, gap=gap,
                sigma_seed=sigma, threshold=threshold, fires=fires)


def m_d3_verdict(cells_by_group_arm_nh: dict[tuple, list[dict]],
                 last_k_controls: dict[str, dict] | None = None) -> dict:
    """The exhaustive S2.1/S2.6 CONFIRM/FALSIFY/INCONCLUSIVE logic.

    `cells_by_group_arm_nh`: {(group, arm, n_h): [cell dict, ...]}.

    `last_k_controls`: optional {group: {"control_mean": float}} at the FAR
    depth -- if a group is absent, its last-K check is explicitly reported
    NOT_CHECKED (never silently treated as "does not fire").

    DISCLOSED OPERATIONALIZATION (the design text pins the CONFIRM and the
    canonical dissociation criteria exactly, S2.1/S2.6, but -- like
    `tost_analysis.py`'s own disclosed REJECT-test choice for TOST's
    unpinned "reject" branch -- leaves the exact mechanical boundary of
    "does NOT measurably separate" under-specified for every possible mixed
    pattern; that ambiguity is intentional, since S2.6 itself routes
    INCONCLUSIVE to human diagnosis via M-D0/M-D2, not a fully mechanical
    verdict). This function pins:
      - CONFIRM iff the contender clears its >=90%-of-own-ceiling far-depth
        bar on ALL FIVE groups AND Arm 2 collapses (<50%-of-ITS-OWN-ceiling)
        on BOTH S5 and A6 (the exact S2.6 sentence).
      - FALSIFY iff, at EACH of S5 and A6 independently, the contender-
        clears/Arm-2-collapses PATTERN, is simultaneously ABSENT (i.e.
        neither gating group shows any separation at all) -- the direct
        reading of "Arm 3 does NOT measurably separate from Arm 2 on S5/A6"
        (S2.6): no separation exists at the decisive pair, regardless of
        what the rest of the family does.
      - Otherwise INCONCLUSIVE (a mixed pattern -- e.g. separation at one of
        S5/A6 but not the other, or the family doesn't clear everywhere
        despite S5/A6 dissociating correctly) -- "diagnose before scaling"
        per S2.6, M-D0's convergence profile and M-D2's rank curve are the
        first two things to check.
      - A last-K trigger firing for ANY group downgrades an otherwise-
        CONFIRM verdict to INCONCLUSIVE (S2.9 item 4), reported separately
        from the "mixed pattern" INCONCLUSIVE reason.
    """
    per_group = {}
    for g in GROUP_NAMES:
        nh = DECISIVE_CONTENDER_NH[g]
        contender_cells = cells_by_group_arm_nh.get((g, CONTENDER_ARM, nh), [])
        baseline_cells = cells_by_group_arm_nh.get((g, BASELINE_ARM, 2), [])
        if not contender_cells or not baseline_cells:
            per_group[g] = dict(status="MISSING_CELLS", decisive_n_h=nh,
                                contender_n=len(contender_cells), baseline_n=len(baseline_cells))
            continue

        ceiling = group_ceiling(contender_cells)
        baseline_ceiling = group_ceiling(baseline_cells)
        mid = group_mean_at_depth(contender_cells, MID_DEPTH)
        far = group_mean_at_depth(contender_cells, FAR_DEPTH)
        far_baseline = group_mean_at_depth(baseline_cells, FAR_DEPTH)

        mid_bar = ceiling * CONFIRM_FRAC_OF_CEILING if ceiling is not None else None
        far_bar = ceiling * CONFIRM_FRAC_OF_CEILING if ceiling is not None else None
        arm2_bar = baseline_ceiling * ARM2_COLLAPSE_FRAC_OF_CEILING if baseline_ceiling is not None else None

        mid_clears = bool(mid is not None and mid_bar is not None and mid >= mid_bar)
        far_clears = bool(far is not None and far_bar is not None and far >= far_bar)
        arm2_collapses = bool(far_baseline is not None and arm2_bar is not None and far_baseline < arm2_bar)
        separates = bool(far_clears and arm2_collapses)

        last_k = None
        if last_k_controls and g in last_k_controls and far is not None:
            seed_vals = [row["recovered_frac_90"] for c in contender_cells for row in c["D_test_results"]
                        if row["D"] == FAR_DEPTH]
            last_k = last_k_downgrade_fires(far, last_k_controls[g]["control_mean"], seed_vals)

        per_group[g] = dict(
            status="OK", decisive_n_h=nh, ceiling=ceiling, baseline_ceiling=baseline_ceiling,
            mid_depth=MID_DEPTH, mid_recovered_frac_90=mid, mid_bar=mid_bar, mid_clears=mid_clears,
            far_depth=FAR_DEPTH, far_recovered_frac_90=far, far_bar=far_bar, far_clears=far_clears,
            arm2_far_recovered_frac_90=far_baseline, arm2_bar=arm2_bar, arm2_collapses=arm2_collapses,
            separates_from_arm2=separates, last_k=last_k,
        )

    missing = [g for g, r in per_group.items() if r["status"] == "MISSING_CELLS"]
    if missing:
        return dict(verdict="INCOMPLETE",
                    reason=f"cells missing for {missing}; cannot read an M-D3 verdict",
                    per_group=per_group)

    family_clears = all(per_group[g]["far_clears"] for g in GROUP_NAMES)
    gating_dissociation = all(per_group[g]["separates_from_arm2"] for g in ARM2_GATING_GROUPS)
    gating_no_separation_anywhere = all(not per_group[g]["separates_from_arm2"] for g in ARM2_GATING_GROUPS)
    a5_disclosed = per_group["A5"]["separates_from_arm2"]   # reported, non-gating (S2.2.3/S2.6)

    last_k_checked = any(per_group[g]["last_k"] is not None for g in GROUP_NAMES)
    any_last_k_fires = any(per_group[g]["last_k"]["fires"] for g in GROUP_NAMES if per_group[g]["last_k"])

    common = dict(per_group=per_group, a5_separates_from_arm2_disclosed=a5_disclosed,
                 last_k_checked=last_k_checked)

    if family_clears and gating_dissociation:
        if last_k_checked and any_last_k_fires:
            return dict(verdict="INCONCLUSIVE",
                        reason="the pre-registered CONFIRM pattern held (far-depth contender clears "
                               ">=90% of its own D_train ceiling on all 5 groups, Arm-2 collapses "
                               "below 50%-of-ceiling on S5 and A6) BUT the S2.9 item 4 last-K "
                               "shortcut-control trigger fired for at least one group -- downgraded "
                               "pending task redesign (adversarially-chosen slower-mixing words), "
                               "not a false CONFIRM",
                        **common)
        return dict(verdict="CONFIRM",
                    reason="contender (Arm 3) clears >=90% of its own D_train recovered_frac@0.9 "
                           "ceiling at the far (~8x) checkpoint on ALL FIVE groups AND Arm 2 (beta "
                           "in [0,1]) drops below 50%-of-ceiling on S5 and A6 at minimum (S2.1/S2.6's "
                           "canonical M-D3 criterion); A5's own dissociation is disclosed, not gating",
                    **common)

    if gating_no_separation_anywhere:
        return dict(verdict="FALSIFY",
                    reason="the contender does NOT measurably separate from Arm 2 at EITHER S5 or A6 "
                           "at the far (~8x) checkpoint (S2.1's FALSIFY trigger: beta range makes no "
                           "detectable difference on this task/architecture/scale)",
                    **common)

    return dict(verdict="INCONCLUSIVE",
                reason="mixed pattern: the contender does not clear on all 5 groups, or the S5/A6 "
                       "dissociation is present at one gating group but not the other, or some other "
                       "non-clean combination -- neither a clean CONFIRM nor a clean FALSIFY. "
                       "Diagnose via M-D0's per-depth convergence profile and M-D2's rank-vs-depth "
                       "curve BEFORE scaling (S2.6, the S1.25/S1.27/S1.29 precedent pattern).",
                **common)


# ---------------------------------------------------------------------------
# Report / CLI.
# ---------------------------------------------------------------------------

def build_cells_by_group_arm_nh(loaded: dict[str, dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for cell in loaded.values():
        key = (cell["group"], cell["arm"], cell["n_h"])
        out.setdefault(key, []).append(cell)
    return out


def print_report(manifest: dict, verdict: dict) -> None:
    print("=" * 100)
    print("STAGE-2 HARVEST -- MANIFEST VERIFICATION")
    print("=" * 100)
    print(f"  expected={manifest['n_expected']}  loaded={manifest['n_loaded']}  clean={manifest['clean']}")
    if manifest["missing"]:
        print(f"  MISSING: {manifest['missing']}")
    if manifest["unexpected"]:
        print(f"  UNEXPECTED (not in the expected set): {manifest['unexpected']}")
    if manifest["dirty_cells"]:
        for cid, probs in manifest["dirty_cells"].items():
            for p in probs:
                print(f"  CONFIG-MISMATCH: {p}")

    print("\n" + "=" * 100)
    print(f"STAGE-2 M-D3 VERDICT: {verdict['verdict']}")
    print("=" * 100)
    print(f"  {verdict['reason']}")
    if "per_group" in verdict:
        print(f"\n  {'group':<6}{'decisive_nh':<13}{'ceiling':<10}{'far@'+str(FAR_DEPTH):<10}"
              f"{'far_bar':<10}{'clears':<8}{'arm2_far':<10}{'arm2_bar':<10}{'collapses':<11}{'separates'}")
        for g, r in verdict["per_group"].items():
            if r["status"] != "OK":
                print(f"  {g:<6}MISSING CELLS (contender_n={r['contender_n']}, baseline_n={r['baseline_n']})")
                continue
            print(f"  {g:<6}{r['decisive_n_h']:<13}{_fmt(r['ceiling']):<10}{_fmt(r['far_recovered_frac_90']):<10}"
                  f"{_fmt(r['far_bar']):<10}{str(r['far_clears']):<8}{_fmt(r['arm2_far_recovered_frac_90']):<10}"
                  f"{_fmt(r['arm2_bar']):<10}{str(r['arm2_collapses']):<11}{r['separates_from_arm2']}")
            if r["last_k"] is not None:
                lk = r["last_k"]
                print(f"         last-K: contender={_fmt(lk['contender_mean'])} control={_fmt(lk['control_mean'])} "
                      f"gap={_fmt(lk['gap'])} threshold={_fmt(lk['threshold'])} fires={lk['fires']}")
    print("=" * 100)


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="stage2_results")
    ap.add_argument("--calibration-only", action="store_true",
                    help="verify against the 11-cell calibration manifest instead of the full 68-cell grid")
    args = ap.parse_args()

    expected = expected_calibration_cell_ids() if args.calibration_only else expected_full_grid_cell_ids()
    loaded = load_cell_results(args.results_dir, cell_ids=expected)
    manifest = verify_manifest(loaded, expected)
    grouped = build_cells_by_group_arm_nh(loaded)
    verdict = m_d3_verdict(grouped)
    print_report(manifest, verdict)

    out_path = os.path.join(args.results_dir, "stage2_harvest_report.json")
    with open(out_path, "w") as f:
        json.dump(dict(manifest=manifest, verdict=verdict), f, indent=2, default=str)
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# CPU self-test -- synthetic cell JSONs, all three verdict branches
# (CONFIRM / FALSIFY / INCONCLUSIVE) plus the last-K downgrade and the
# config-match / manifest-set negative tests.
# ---------------------------------------------------------------------------

def _synthetic_cell(group, arm, n_h, seed, ceiling=0.95, far_frac_of_ceiling=1.0,
                    mid_frac_of_ceiling=1.0, steps=None) -> dict:
    """Builds a schema-correct synthetic cell (S2.20 F4's real-runner shape)
    with recovered_frac_90 = far_frac_of_ceiling * ceiling at D=64,
    mid_frac_of_ceiling * ceiling at D=32, and `ceiling` itself at
    D=D_train_max via m_d0_profile -- every other D_test point interpolated
    linearly for a schema-complete (not decisively-used) curve."""
    steps = steps if steps is not None else STEP_BUDGET_BY_GROUP_INDEPENDENT[group]
    m_d0_profile = [dict(D=1, gating="hard", excluded=True, recovered_frac_90=None, mean_cos=None,
                         note="D=1 structurally unevaluable")]
    for D in range(2, D_TRAIN_MAX + 1):
        gating = "hard" if D <= 5 else "disclosed"
        frac = ceiling * (0.5 + 0.5 * D / D_TRAIN_MAX)   # monotone ramp into the ceiling, schema filler
        m_d0_profile.append(dict(D=D, gating=gating, excluded=False, recovered_frac_90=frac, mean_cos=0.9))

    D_test_results = []
    for D in D_TEST_GRID:
        if D == MID_DEPTH:
            frac = mid_frac_of_ceiling * ceiling
        elif D == FAR_DEPTH:
            frac = far_frac_of_ceiling * ceiling
        else:
            # smooth interpolation between the D_train ceiling and the far
            # value, schema-filler only -- M-D3 only reads MID_DEPTH/FAR_DEPTH.
            t = (D - D_TRAIN_MAX) / (FAR_DEPTH - D_TRAIN_MAX)
            frac = ceiling * (1 - t) + far_frac_of_ceiling * ceiling * t
        D_test_results.append(dict(D=D, recovered_frac_90=max(0.0, min(1.0, frac)), mean_cos=0.9,
                                   crosscheck_recovered_frac_90=frac, crosscheck_mean_cos=0.95,
                                   restricted_effective_rank=3.0, near_plateau=D <= 16, far_depth=D >= 32))

    gate_report = dict(per_depth=[dict(D=D, T=0.3, T_anchor=0.4, R=0.5, R_anchor=0.6, T_median=0.28,
                                       anchor_floor_ok=True, bar_T_ok=True, bar_R_ok=True, depth_pass=True)
                                  for D in PROBE_DEPTHS],
                       overall_pass=True, floor_violated=False, floor_violation_depths=[])

    cell_id = f"{group}__{arm}__nh{n_h}__seed{seed}"
    return dict(cell_id=cell_id, group=group, arm=arm, n_h=n_h, seed=seed, status="completed",
               steps_completed=steps, n_skipped_steps=0, final_loss=0.05, wall_clock_s=120.0,
               checkpoint_path=f"/tmp/{cell_id}.pt", param_fingerprint="deadbeef" * 8,
               m_d0_profile=m_d0_profile, gate_route="pass", gate_report=gate_report,
               param_match=dict(within_tolerance=True, delta_frac=0.02), D_test_results=D_test_results,
               runner="real")


def _synth_calibration_cells(contender_far_frac: dict, baseline_far_frac: dict,
                             ceiling=0.95) -> dict[str, dict]:
    """Builds the REAL 11-cell calibration set synthetically (S2.5/S2.8 item
    2): the 10 base (group, arm) cells at n_h=2, PLUS the promoted (S5,
    Arm3, n_h=4) cell -- so S5 gets BOTH a base n_h=2 contender cell (not
    decisive, present in the real set, unread by `m_d3_verdict`) and the
    n_h=4 promoted cell (decisive, what `m_d3_verdict` actually reads).
    `contender_far_frac`/`baseline_far_frac` map group -> far_frac_of_ceiling
    for the DECISIVE contender leg / the baseline arm respectively (defaults
    to 1.0 if a group is absent); the S5 base n_h=2 cell always uses the
    group's own far_frac_of_ceiling too (its own value is never read by
    M-D3, only its presence/shape matters for the manifest check)."""
    cells = {}
    for g in GROUP_NAMES:
        c_frac = contender_far_frac.get(g, 1.0)
        b_frac = baseline_far_frac.get(g, 1.0)
        decisive_nh = DECISIVE_CONTENDER_NH[g]
        c = _synthetic_cell(g, CONTENDER_ARM, decisive_nh, seed=0, ceiling=ceiling, far_frac_of_ceiling=c_frac)
        b = _synthetic_cell(g, BASELINE_ARM, 2, seed=0, ceiling=ceiling, far_frac_of_ceiling=b_frac)
        cells[c["cell_id"]] = c
        cells[b["cell_id"]] = b
        if decisive_nh != 2:
            # the base n_h=2 cell also exists in the real 11-cell set (S2.5
            # Rev 2: base and promoted are DISTINCT, not a near-duplicate).
            base = _synthetic_cell(g, CONTENDER_ARM, 2, seed=0, ceiling=ceiling, far_frac_of_ceiling=c_frac)
            cells[base["cell_id"]] = base
    return cells


def _test_confirm_path():
    print("=" * 88)
    print("UNIT TEST 1 -- CONFIRM path (contender clears all 5 groups, Arm-2 collapses on S5/A6)")
    print("=" * 88)
    cells = _synth_calibration_cells(
        contender_far_frac={g: 0.95 for g in GROUP_NAMES},        # clears the 0.9x bar everywhere
        baseline_far_frac={"S3": 0.9, "S4": 0.85, "A5": 0.7, "S5": 0.2, "A6": 0.15},  # collapses at S5/A6
    )
    grouped = build_cells_by_group_arm_nh(cells)
    verdict = m_d3_verdict(grouped)
    print(f"  verdict={verdict['verdict']}  reason={verdict['reason'][:80]}...")
    assert verdict["verdict"] == "CONFIRM", f"expected CONFIRM, got {verdict['verdict']}: {verdict['reason']}"
    print("  PASS\n")


def _test_falsify_path():
    print("=" * 88)
    print("UNIT TEST 2 -- FALSIFY path (no separation from Arm-2 at EITHER S5 or A6)")
    print("=" * 88)
    cells = _synth_calibration_cells(
        contender_far_frac={g: 0.3 for g in GROUP_NAMES},          # contender itself collapses everywhere
        baseline_far_frac={g: 0.3 for g in GROUP_NAMES},           # Arm-2 tracks it -- no separation
    )
    grouped = build_cells_by_group_arm_nh(cells)
    verdict = m_d3_verdict(grouped)
    print(f"  verdict={verdict['verdict']}  reason={verdict['reason'][:80]}...")
    assert verdict["verdict"] == "FALSIFY", f"expected FALSIFY, got {verdict['verdict']}: {verdict['reason']}"
    print("  PASS\n")


def _test_inconclusive_mixed_path():
    print("=" * 88)
    print("UNIT TEST 3 -- INCONCLUSIVE path (mixed: S5 dissociates, A6 does not; S3 also misses)")
    print("=" * 88)
    cells = _synth_calibration_cells(
        contender_far_frac={"S3": 0.5, "S4": 0.95, "A5": 0.95, "S5": 0.95, "A6": 0.95},  # S3 misses its own bar
        baseline_far_frac={"S3": 0.9, "S4": 0.85, "A5": 0.7, "S5": 0.2, "A6": 0.9},       # A6 does NOT collapse
    )
    grouped = build_cells_by_group_arm_nh(cells)
    verdict = m_d3_verdict(grouped)
    print(f"  verdict={verdict['verdict']}  reason={verdict['reason'][:80]}...")
    assert verdict["verdict"] == "INCONCLUSIVE", f"expected INCONCLUSIVE, got {verdict['verdict']}"
    print("  PASS\n")


def _test_last_k_downgrade():
    print("=" * 88)
    print("UNIT TEST 4 -- CONFIRM pattern present but last-K trigger fires -> downgraded to INCONCLUSIVE")
    print("=" * 88)
    cells = _synth_calibration_cells(
        contender_far_frac={g: 0.95 for g in GROUP_NAMES},
        baseline_far_frac={"S3": 0.9, "S4": 0.85, "A5": 0.7, "S5": 0.2, "A6": 0.15},
    )
    grouped = build_cells_by_group_arm_nh(cells)
    # S5's contender reads ceiling*0.95 at D=64 (single seed here, n=1 -> sigma_seed=0,
    # so the trigger floor is the 0.05 absolute term) -- plant a last-K control that
    # sits within 0.05 of the contender's own far-depth mean.
    s5_far = group_mean_at_depth([cells["S5__arm3_beta02__nh4__seed0"]], FAR_DEPTH)
    last_k_controls = {"S5": dict(control_mean=s5_far - 0.03)}   # gap 0.03 <= max(0, 0.05) -- fires
    verdict = m_d3_verdict(grouped, last_k_controls=last_k_controls)
    print(f"  verdict={verdict['verdict']}  S5 last_k={verdict['per_group']['S5']['last_k']}")
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert verdict["per_group"]["S5"]["last_k"]["fires"] is True
    assert "last-K" in verdict["reason"]
    # a control comfortably far below the contender must NOT fire.
    last_k_controls_safe = {"S5": dict(control_mean=s5_far - 0.5)}
    verdict_safe = m_d3_verdict(grouped, last_k_controls=last_k_controls_safe)
    assert verdict_safe["verdict"] == "CONFIRM", "a control far below the contender wrongly fired the trigger"
    print("  PASS (fires when within threshold, does not fire when clearly separated)\n")


def _test_sigma_seed_ddof1():
    print("=" * 88)
    print("UNIT TEST 5 -- sigma_seed uses ddof=1 (S2.14 minor pin), not ddof=0")
    print("=" * 88)
    cells = [_synthetic_cell("S4", CONTENDER_ARM, 2, seed=s, ceiling=0.9,
                             far_frac_of_ceiling=v)
            for s, v in enumerate([1.00, 0.98, 1.02, 0.95, 1.05])]
    vals = [row["recovered_frac_90"] for c in cells for row in c["D_test_results"] if row["D"] == FAR_DEPTH]
    manual_ddof1 = float(np.std(vals, ddof=1))
    manual_ddof0 = float(np.std(vals, ddof=0))
    got = sigma_seed(cells, FAR_DEPTH)
    print(f"  sigma_seed(ddof=1)={got:.6f}  manual ddof=1={manual_ddof1:.6f}  manual ddof=0={manual_ddof0:.6f}")
    assert abs(got - manual_ddof1) < 1e-9, "sigma_seed does not match the ddof=1 sample std"
    assert abs(got - manual_ddof0) > 1e-6, "ddof=1 and ddof=0 unexpectedly coincide -- test is not discriminating"
    print("  PASS\n")


def _test_config_match_negative():
    print("=" * 88)
    print("NEGATIVE TEST -- config-match catches a cell_id/field mismatch and a truncated D_test_results")
    print("=" * 88)
    good = _synthetic_cell("S4", CONTENDER_ARM, 2, seed=0)
    assert verify_config_match(good) == [], f"a clean synthetic cell must report zero problems: {verify_config_match(good)}"

    corrupt_field = dict(good)
    corrupt_field["group"] = "A5"   # field says A5, cell_id still says S4 -- must be CAUGHT
    problems = verify_config_match(corrupt_field)
    assert any("field group=" in p for p in problems), f"field/cell_id mismatch not caught: {problems}"

    corrupt_steps = dict(good)
    corrupt_steps["steps_completed"] = 999
    problems2 = verify_config_match(corrupt_steps)
    assert any("steps_completed" in p for p in problems2), f"steps_completed mismatch not caught: {problems2}"

    corrupt_grid = dict(good)
    corrupt_grid["D_test_results"] = [row for row in good["D_test_results"] if row["D"] != FAR_DEPTH]
    problems3 = verify_config_match(corrupt_grid)
    assert any("D_test_results D-set" in p for p in problems3), f"truncated D_test_results not caught: {problems3}"
    print(f"  clean cell: 0 problems. corrupted group field: {len(problems)} caught. "
          f"corrupted steps_completed: {len(problems2)} caught. truncated D_test_results: {len(problems3)} caught.")
    print("  PASS\n")


def _test_manifest_set_mismatch():
    print("=" * 88)
    print("NEGATIVE TEST -- manifest verification catches a missing cell and an unexpected extra cell")
    print("=" * 88)
    expected = expected_calibration_cell_ids()
    cells = _synth_calibration_cells(contender_far_frac={}, baseline_far_frac={})
    del cells["S5__arm3_beta02__nh4__seed0"]   # drop the promoted calibration cell
    extra = _synthetic_cell("S3", CONTENDER_ARM, 2, seed=4)   # not in the calibration set (seed=4)
    cells[extra["cell_id"]] = extra
    manifest = verify_manifest(cells, expected)
    print(f"  clean={manifest['clean']}  missing={manifest['missing']}  unexpected={manifest['unexpected']}")
    assert manifest["clean"] is False
    assert "S5__arm3_beta02__nh4__seed0" in manifest["missing"]
    assert extra["cell_id"] in manifest["unexpected"]
    print("  PASS\n")


def smoke():
    _test_confirm_path()
    _test_falsify_path()
    _test_inconclusive_mixed_path()
    _test_last_k_downgrade()
    _test_sigma_seed_ddof1()
    _test_config_match_negative()
    _test_manifest_set_mismatch()
    print("=" * 88)
    print("stage2_harvest.py: ALL SELF-TESTS PASSED (CONFIRM / FALSIFY / INCONCLUSIVE / "
          "last-K downgrade / sigma_seed ddof=1 / config-match negative / manifest-set negative)")
    print("=" * 88)


if __name__ == "__main__":
    if "--smoke" in sys.argv or "--self-test" in sys.argv:
        smoke()
    else:
        main()
