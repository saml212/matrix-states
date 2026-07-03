#!/usr/bin/env python3
"""
TRACK A — steps-to-criterion sample-efficiency analysis (zero GPU).

Executes matrix-thinking/SCALE_TRANSFER_DESIGN.md Rev 2, §3. Reads already-archived
per-checkpoint trajectories in experiment-runs/2026-07-03_deltanet_rd_waves/exactness/
and computes, per (arm, K, seed, hop, threshold), the training step at which
`recovered_frac@0.9` first crosses each of three thresholds (0.5, 0.8, 0.9).

No training, no model code, no GPU. Pure JSON parsing + linear interpolation.

Resolution ceiling (state plainly, per §3.2/§3.4/§3.7 item 2): checkpoints are spaced
every 2,000 steps (10 checkpoints per 20,000-step run). A "crossing step" is either:
  - the raw checkpoint step at which the threshold is first met (native resolution,
    always a multiple of 2,000), or
  - a LINEAR INTERPOLATION between the two bracketing checkpoints, explicitly labeled
    `interpolated=True`. Interpolation assumes local linearity between adjacent
    checkpoints, which the data do not always satisfy (recovered_frac@0.9 can dip
    between checkpoints, e.g. K=16 baseline h=2 at steps 6000->8000). It is reported
    as a best-effort estimate, never as a measurement at finer-than-2000-step
    precision.
  - LEFT-CENSORED: the very first checkpoint (step 2000) already meets the threshold.
    There is no data between step 0 and step 2000 (the `trajectory` array has no
    recovery metric, only loss/eff_rank proxies), so we cannot localize the crossing
    further than "<=2000". Reported as such, never rounded down.
  - RIGHT-CENSORED: the threshold is never met within the full 20,000-step run.
    Reported as ">20000", never as a missing value (per §3.4).

Two-tier K=32 admissibility (§3.7 item 5) and the K=48 admissibility read (found live
in the archive during this analysis session, committed at HEAD as
"geo3: K=48 stretch results", commit fc3ded1) are read directly from each geo3 file's
own `geo3_admission.admissible` boolean — not inferred from the `geo3n12`/`geo3n20`
filename suffix, which is a necessary-but-not-sufficient signal (K=48 uses n_iter=20
throughout but is non-admissible for a DIFFERENT reason than K=32's original n_iter=12
runs: value_salvage_tier_pass=False, not an unconverged Newton-Schulz fallback).

Usage:
    python3 analyze_sample_efficiency.py [--out-dir DIR]

Writes:
    <out-dir>/track_a_sample_efficiency.json   -- full per-seed + aggregated tables
    <out-dir>/track_a_summary.md               -- human-readable headline tables
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO_ROOT / "experiment-runs" / "2026-07-03_deltanet_rd_waves" / "exactness"
DEFAULT_OUT_DIR = REPO_ROOT / "experiment-runs" / "2026-07-04_track_a"

EXPECTED_N_PARAMS = 12_899_841

# ---------------------------------------------------------------------------
# Pre-registered analysis parameters (§3.4)
# ---------------------------------------------------------------------------

# §3.4's own two pre-registered thresholds (0.5, 0.8, bracketing WaveF's own 0.8
# "minimum publishable" bar). 0.9 is added at this run's explicit request (it is
# also, not coincidentally, the metric's own name -- "when does frac@0.9 itself
# reach 0.9" is a valid third, stricter criterion) and is labeled as
# ADDED-NOT-PREREGISTERED everywhere it appears in the output.
THRESHOLDS_PREREGISTERED = [0.5, 0.8]
THRESHOLDS_ADDED = [0.9]
THRESHOLDS = THRESHOLDS_PREREGISTERED + THRESHOLDS_ADDED

# §3.4: primary hops h=1 (in-distribution, no-sacrifice guard) and h=4 (the headline
# held-out hop every prior verdict in this program used). h=2 is added because this
# run's own instructions ask for it explicitly as a second headline hop.
HOPS_HEADLINE = [1, 2, 4]
HOPS_SECONDARY = [3, 5, 6, 7, 21]
GENERALIZATION_SECTIONS = ["C17_heldout_entities", "C19_heldout_template"]
GENERALIZATION_HOPS = [1, 2, 3]

METRIC = "recovered_frac@0.9"


def section_for_hop(h: int) -> str:
    if h in (1, 2, 3):
        return "M2_in_distribution"
    if h in (4, 5, 6, 7, 21):
        return "M3_held_out"
    raise ValueError(f"no known section for hop {h}")


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------
# Every cell below is drawn from SCALE_TRANSFER_DESIGN.md §3.5's table, PLUS two
# out-of-manifest bonus arms this analysis includes because the dumps already exist
# and were explicitly requested for this pass (i-strong, K=48). Every out-of-manifest
# cell is tagged `in_manifest=False` in the output and reported separately from the
# §3.5 headline comparison so it cannot be silently mistaken for a pre-registered cell.

def _baseline_files(K: int, seeds: list[int]) -> list[dict]:
    if K in (16, 32):
        return [
            {
                "arm": "baseline",
                "arm_label": "learned (arm iii-beta)",
                "K": K,
                "seed": s,
                "path": ARCHIVE_ROOT / f"w1_iiibeta_K{K}_s{s}.json",
                "tier": "baseline",
                "in_manifest": True,
            }
            for s in seeds
        ]
    if K == 48:
        return [
            {
                "arm": "baseline",
                "arm_label": "learned (arm iii-beta equivalent, K=48 rider)",
                "K": K,
                "seed": s,
                "path": ARCHIVE_ROOT / f"k48_learned_s{s}.json",
                "tier": "baseline",
                "in_manifest": False,
                "manifest_note": (
                    "K=48 is outside SCALE_TRANSFER_DESIGN.md §3.5's manifest "
                    "(§3.7 item 3 explicitly warns no K=48 geo3 run existed at "
                    "design time). A matched K=48 geo3 run landed live in the archive "
                    "during this analysis session (commit fc3ded1, 'geo3: K=48 stretch "
                    "results') -- included here as a bonus third K-point, not as a "
                    "silent extension of the pre-registered manifest."
                ),
            }
            for s in seeds
        ]
    raise ValueError(K)


def _geo3_files(K: int, tier_suffix: str, seeds: list[int], tier: str, primary: bool) -> list[dict]:
    return [
        {
            "arm": "geo3",
            "arm_label": f"geo3 ({tier_suffix})",
            "K": K,
            "seed": s,
            "path": ARCHIVE_ROOT / "wavegeo3" / f"wgeo3_rdx_K{K}_armgeo3_s{s}_{tier_suffix}.json",
            "tier": tier,
            "primary_for_K": primary,
            "in_manifest": K in (16, 32),
            **(
                {}
                if K in (16, 32)
                else {
                    "manifest_note": (
                        "K=48 geo3 is outside §3.5's manifest (see baseline "
                        "manifest_note above) -- bonus cell, non-admissible per its "
                        "own geo3_admission field (value_salvage_tier_pass=False, a "
                        "DIFFERENT failure mode than K=32 n12's NS-fallback failure)."
                    )
                }
            ),
        }
        for s in seeds
    ]


def _i_strong_files(K: int, seeds: list[int]) -> list[dict]:
    return [
        {
            "arm": "i-strong",
            "arm_label": "arm i-strong (strong_pin=True)",
            "K": K,
            "seed": s,
            "path": ARCHIVE_ROOT / "wave1" / f"w1_rdx_K{K}_armi-strong_s{s}_sp.json",
            "tier": "baseline-variant",
            "in_manifest": False,
            "manifest_note": (
                "arm i-strong is not part of §3.5's manifest (which specifies "
                "only arm iii-beta as the baseline comparator). Included because "
                "dumps exist (K=32 only) and were explicitly requested for this "
                "pass. i-strong is a strong-pinning baseline VARIANT, not a geo3 "
                "arm -- it has no `geo3_admission` field and is reported for "
                "context, not as part of the geo3-vs-baseline ratio headline."
            ),
        }
        for s in seeds
    ]


def build_manifest() -> list[dict]:
    m = []
    m += _baseline_files(16, [0, 1])
    m += _baseline_files(32, [0, 1, 2])
    m += _baseline_files(48, [0, 1, 2])
    m += _geo3_files(16, "geo3n12", [0, 1, 2], tier="admissible-primary", primary=True)
    m += _geo3_files(32, "geo3n20", [0, 1, 2], tier="admissible-primary", primary=True)
    m += _geo3_files(32, "geo3n12", [0, 1, 2], tier="descriptive-only", primary=False)
    m += _geo3_files(48, "geo3n20", [0, 1, 2], tier="non-admissible", primary=True)
    m += _i_strong_files(32, [0, 1, 2])
    return m


# ---------------------------------------------------------------------------
# Crossing-step computation
# ---------------------------------------------------------------------------

def crossing_info(steps: list[int], values: list[float], threshold: float) -> dict:
    """First step at which `values` crosses >= threshold, walking the checkpoint
    grid in step order. Returns a dict with explicit censoring status -- never a
    silently-missing value (§3.4)."""
    assert steps == sorted(steps)
    if values[0] >= threshold:
        return {
            "status": "left_censored",
            "checkpoint_step": steps[0],
            "interpolated_step": None,
            "note": f"already >= {threshold} at the first checkpoint (step {steps[0]}); "
                    f"cannot localize further, no data before step {steps[0]}",
        }
    for i in range(1, len(steps)):
        if values[i] >= threshold:
            s_prev, s_cur = steps[i - 1], steps[i]
            v_prev, v_cur = values[i - 1], values[i]
            if v_cur == v_prev:
                interp = float(s_cur)
            else:
                frac = (threshold - v_prev) / (v_cur - v_prev)
                interp = s_prev + frac * (s_cur - s_prev)
            return {
                "status": "crossed",
                "checkpoint_step": s_cur,
                "interpolated_step": round(interp, 1),
                "note": f"native-resolution checkpoint at step {s_cur}; linear "
                        f"interpolation between ({s_prev},{v_prev:.4f}) and "
                        f"({s_cur},{v_cur:.4f}) estimates step {interp:.1f}",
            }
    return {
        "status": "right_censored",
        "checkpoint_step": None,
        "interpolated_step": None,
        "note": f"never reached {threshold} within {steps[-1]} steps "
                f"(max observed {max(values):.4f})",
    }


def load_series(d: dict, section: str, hop: int) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    for ck in d["checkpoints"]:
        entry = ck.get(section, {}).get(str(hop))
        if entry is None:
            continue
        steps.append(ck["step"])
        vals.append(entry[METRIC])
    return steps, vals


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_file(meta: dict) -> dict:
    path = meta["path"]
    with open(path) as f:
        d = json.load(f)

    n_params = d.get("n_params")
    if n_params != EXPECTED_N_PARAMS:
        raise AssertionError(
            f"{path}: n_params={n_params}, expected {EXPECTED_N_PARAMS} "
            "(param-count mismatch would silently mix incomparable model sizes)"
        )

    row = dict(meta)
    row["path"] = str(path.relative_to(REPO_ROOT))
    row["n_params"] = n_params
    row["steps_completed"] = d.get("steps_completed")
    row["timed_out"] = d.get("timed_out")
    if "geo3_admission" in d and d["geo3_admission"] is not None:
        row["geo3_admission"] = {
            "admissible": d["geo3_admission"].get("admissible"),
            "ns_converged_no_fallback": d["geo3_admission"].get("ns_converged_no_fallback"),
            "n_geo3_fallback_train_steps": d["geo3_admission"].get("n_geo3_fallback_train_steps"),
            "value_salvage_tier_pass": d["geo3_admission"].get("value_salvage_tier_pass"),
        }

    crossings = {}
    all_hops = [(section_for_hop(h), h) for h in HOPS_HEADLINE + HOPS_SECONDARY]
    all_hops += [(sec, h) for sec in GENERALIZATION_SECTIONS for h in GENERALIZATION_HOPS]

    for section, hop in all_hops:
        steps, vals = load_series(d, section, hop)
        if not steps:
            continue
        key = f"{section}:h{hop}"
        crossings[key] = {
            "section": section,
            "hop": hop,
            "series": list(zip(steps, [round(v, 4) for v in vals])),
            "thresholds": {
                str(t): crossing_info(steps, vals, t) for t in THRESHOLDS
            },
        }
    row["crossings"] = crossings
    return row


def aggregate(rows: list[dict]) -> dict:
    """Group per-seed rows into (arm, tier, K, hop-key, threshold) cells and compute
    seed-resolved + pooled summaries. Per attack item 4: never report only a pooled
    mean, always keep the per-seed list alongside it."""
    groups: dict[tuple, list] = {}
    for row in rows:
        gkey = (row["arm"], row.get("arm_label"), row["K"], row.get("tier"))
        for hop_key, hop_data in row["crossings"].items():
            for thr, info in hop_data["thresholds"].items():
                cell_key = gkey + (hop_key, thr)
                groups.setdefault(cell_key, []).append(
                    {"seed": row["seed"], **info}
                )

    agg = []
    for (arm, arm_label, K, tier, hop_key, thr), seed_results in sorted(
        groups.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][3] or "", kv[0][4], float(kv[0][5]))
    ):
        n = len(seed_results)
        crossed = [r for r in seed_results if r["status"] == "crossed"]
        left = [r for r in seed_results if r["status"] == "left_censored"]
        right = [r for r in seed_results if r["status"] == "right_censored"]
        interp_vals = [r["interpolated_step"] for r in crossed]
        agg.append({
            "arm": arm,
            "arm_label": arm_label,
            "K": K,
            "tier": tier,
            "hop_key": hop_key,
            "threshold": float(thr),
            "threshold_preregistered": float(thr) in THRESHOLDS_PREREGISTERED,
            "n_seeds": n,
            "n_crossed": len(crossed),
            "n_left_censored": len(left),
            "n_right_censored": len(right),
            "interpolated_step_mean_among_crossed": (
                round(sum(interp_vals) / len(interp_vals), 1) if interp_vals else None
            ),
            "interpolated_step_min_among_crossed": min(interp_vals) if interp_vals else None,
            "interpolated_step_max_among_crossed": max(interp_vals) if interp_vals else None,
            "per_seed": [
                {"seed": r["seed"], "status": r["status"],
                 "checkpoint_step": r["checkpoint_step"],
                 "interpolated_step": r["interpolated_step"]}
                for r in sorted(seed_results, key=lambda r: r["seed"])
            ],
        })
    return agg


LEFT_BOUND_STEP = 2000    # first checkpoint; a left-censored cell crossed at or before this
RIGHT_BOUND_STEP = 20000  # last checkpoint; a right-censored cell never crossed by this


def _effective_step(cell: dict) -> tuple[str, float | None]:
    """Reduce a cell's per-seed censoring pattern to one comparable (kind, step) pair.

    kind is one of:
      'exact'       -- every seed crossed; step is the mean interpolated crossing step.
      'left_bound'  -- every non-crossed... actually every seed is left-censored
                       (already >= threshold at step 2000); true step is <=2000, unknown
                       how much earlier. Uses 2000 as a conservative (late) bound.
      'right_bound' -- every seed is right-censored (never crosses within 20000); true
                       step is >20000, uses 20000 as a conservative (early) bound.
      'mixed'       -- seeds disagree in status; no single honest number, caller must
                       fall back to a descriptive note, not a computed ratio.
    """
    n = cell["n_seeds"]
    if cell["n_crossed"] == n and cell["interpolated_step_mean_among_crossed"] is not None:
        return "exact", cell["interpolated_step_mean_among_crossed"]
    if cell["n_left_censored"] == n:
        return "left_bound", float(LEFT_BOUND_STEP)
    if cell["n_right_censored"] == n:
        return "right_bound", float(RIGHT_BOUND_STEP)
    return "mixed", None


def ratio_note(baseline_cell: dict | None, geo3_cell: dict | None) -> dict:
    if baseline_cell is None or geo3_cell is None:
        return {"comparable": False, "note": "missing cell"}

    b_kind, b_step = _effective_step(baseline_cell)
    g_kind, g_step = _effective_step(geo3_cell)

    if b_kind == "mixed" or g_kind == "mixed":
        return {
            "comparable": False,
            "note": "seeds disagree in censoring status within at least one arm "
                    "(some crossed, some left/right-censored) -- no single honest "
                    "step number exists at this n; see per_seed for the full "
                    "breakdown rather than trusting a pooled ratio",
        }
    if b_kind == "left_bound" and g_kind == "left_bound":
        return {
            "comparable": False,
            "note": "both arms already at/above threshold at the first checkpoint "
                    "(step<=2000) in every seed -- no speed advantage measurable at "
                    "this resolution, ceiling-only comparison possible",
        }
    if g_kind == "right_bound":
        return {
            "comparable": False,
            "note": "geo3 never crosses this threshold within 20,000 steps in any "
                    "seed -- this is a ceiling failure for geo3 at this threshold, "
                    "not a sample-efficiency question",
        }
    if b_kind == "left_bound" and g_kind != "left_bound":
        return {
            "comparable": False,
            "note": "baseline already at/above threshold by the first checkpoint "
                    "(step<=2000); geo3 crosses later ({}) -- geo3 shows NO speed "
                    "advantage at this hop/threshold (already an easy cell for "
                    "baseline)".format(g_step),
        }
    if b_kind == "right_bound" and g_kind == "left_bound":
        return {
            "comparable": "lower_bound_only",
            "note": f"baseline never crosses within 20,000 steps in any seed; geo3 "
                    f"is already >= threshold by its first checkpoint (step<=2000, "
                    f"true crossing could be earlier). Speedup is AT LEAST "
                    f"{round(RIGHT_BOUND_STEP / LEFT_BOUND_STEP, 1)}x (both bounds "
                    f"conservative -- true ratio is likely larger)",
            "lower_bound_x": round(RIGHT_BOUND_STEP / LEFT_BOUND_STEP, 1),
        }
    if b_kind == "right_bound" and g_kind == "exact":
        return {
            "comparable": "lower_bound_only",
            "note": f"baseline never crosses within 20,000 steps in any seed; geo3 "
                    f"crosses at interpolated step {g_step}. Speedup is AT LEAST "
                    f"{round(RIGHT_BOUND_STEP / g_step, 1)}x (true baseline step is "
                    f"right-censored, ratio is a lower bound, not exact)",
            "lower_bound_x": round(RIGHT_BOUND_STEP / g_step, 1),
        }
    if b_kind == "exact" and g_kind == "left_bound":
        return {
            "comparable": "lower_bound_only",
            "note": f"baseline crosses at interpolated step {b_step}; geo3 is already "
                    f">= threshold by its first checkpoint (step<=2000, true crossing "
                    f"could be earlier). Speedup is AT LEAST "
                    f"{round(b_step / LEFT_BOUND_STEP, 1)}x (geo3's true step is "
                    f"left-censored, ratio is a lower bound, not exact)",
            "lower_bound_x": round(b_step / LEFT_BOUND_STEP, 1) if b_step else None,
        }
    # Note: (b_kind=="exact", g_kind=="right_bound") is already handled by the
    # `g_kind == "right_bound"` branch above (checked first, independent of b_kind)
    # -- no separate case needed here.
    if b_kind == "exact" and g_kind == "exact":
        ratio = round(b_step / g_step, 2) if g_step else None
        return {
            "comparable": True,
            "baseline_steps": b_step,
            "geo3_steps": g_step,
            "ratio_x": ratio,
            "note": f"geo3 reaches threshold in {g_step} steps (interpolated) vs "
                    f"baseline's {b_step} steps -- {ratio}x faster" if ratio else "n/a",
        }
    return {
        "comparable": False,
        "note": f"unhandled censoring combination (baseline={b_kind}, geo3={g_kind}) "
                "-- see per_seed for raw data",
    }


def build_headline_ratios(agg: list[dict]) -> list[dict]:
    """The report's headline: geo3 (primary tier per K) vs baseline steps-to-threshold
    at h=1 and h=2, threshold=0.9, per K -- exactly what this run's instructions ask
    for. Also computed at 0.5/0.8 and h=4 for completeness since the analysis is free."""
    def find(arm, K, tier, hop_key, thr):
        for c in agg:
            if (c["arm"] == arm and c["K"] == K and c["hop_key"] == hop_key
                    and c["threshold"] == thr and c["tier"] == tier):
                return c
        return None

    headline = []
    geo3_primary_tier = {16: "admissible-primary", 32: "admissible-primary", 48: "non-admissible"}
    for K in (16, 32, 48):
        for hop in HOPS_HEADLINE:
            hop_key = f"{section_for_hop(hop)}:h{hop}"
            for thr in THRESHOLDS:
                b = find("baseline", K, "baseline", hop_key, thr)
                g = find("geo3", K, geo3_primary_tier[K], hop_key, thr)
                headline.append({
                    "K": K,
                    "hop": hop,
                    "threshold": thr,
                    "threshold_preregistered": thr in THRESHOLDS_PREREGISTERED,
                    "baseline_cell": b,
                    "geo3_cell": g,
                    "ratio": ratio_note(b, g),
                })
    return headline


def build_n12_vs_n20_bonus(agg: list[dict]) -> list[dict]:
    """Free bonus readout named in §3.5: do the n_iter=12 and n_iter=20 K=32 geo3
    trajectories match, not just their endpoints (already shown near-identical in
    DELTANET_RD_EXACTNESS_DESIGN.md §16.5)?"""
    def find(tier, hop_key, thr):
        for c in agg:
            if c["arm"] == "geo3" and c["K"] == 32 and c["tier"] == tier \
                    and c["hop_key"] == hop_key and c["threshold"] == thr:
                return c
        return None

    out = []
    for hop in HOPS_HEADLINE:
        hop_key = f"{section_for_hop(hop)}:h{hop}"
        for thr in THRESHOLDS:
            n12 = find("descriptive-only", hop_key, thr)
            n20 = find("admissible-primary", hop_key, thr)
            if n12 is None or n20 is None:
                continue
            n12m, n20m = n12["interpolated_step_mean_among_crossed"], n20["interpolated_step_mean_among_crossed"]
            if n12m is not None and n20m is not None:
                delta = round(abs(n12m - n20m), 1)
                within_resolution = "yes" if delta <= 2000 else "no"
            elif n12m is None and n20m is None:
                delta = None
                within_resolution = "n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed)"
            else:
                delta = None
                within_resolution = "n/a (one arm crossed, the other is censored at this cell -- see per_seed)"
            out.append({
                "hop": hop, "threshold": thr,
                "n12_interpolated_step_mean": n12m,
                "n20_interpolated_step_mean": n20m,
                "abs_delta_steps": delta,
                "within_one_checkpoint_resolution": within_resolution,
            })
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _display_step(cell: dict | None) -> str:
    """Human-readable step display for a cell, honestly distinguishing left- vs
    right-censoring rather than collapsing both to one placeholder string."""
    if cell is None:
        return "n/a"
    n = cell["n_seeds"]
    if cell["n_crossed"] == n and cell["interpolated_step_mean_among_crossed"] is not None:
        return str(cell["interpolated_step_mean_among_crossed"])
    if cell["n_left_censored"] == n:
        return "<=2000 (left-censored, all seeds)"
    if cell["n_right_censored"] == n:
        return ">20000 (right-censored, all seeds)"
    return (
        f"mixed ({cell['n_crossed']} crossed / {cell['n_left_censored']} left-cens "
        f"/ {cell['n_right_censored']} right-cens of {n})"
    )


def render_markdown(summary: dict) -> str:
    lines = []
    lines.append("# Track A — steps-to-criterion sample efficiency (zero GPU)")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}  ")
    lines.append(f"Repo commit at analysis time: `{summary['git_commit']}`  ")
    lines.append(f"Archive: `{summary['archive_root']}`  ")
    lines.append("")
    lines.append(
        "**Resolution ceiling: checkpoints are spaced every 2,000 steps.** "
        "`interpolated_step` linearly interpolates between the two bracketing "
        "checkpoints and is explicitly an estimate, never a finer-grained "
        "measurement. `checkpoint_step` is the raw, non-interpolated first "
        "checkpoint at which the threshold is met -- the true resolution-honest "
        "number. Left-censored = already above threshold at step 2000 (no earlier "
        "data exists). Right-censored = never crosses within 20,000 steps."
    )
    lines.append("")
    lines.append(
        f"Thresholds {THRESHOLDS_PREREGISTERED} are pre-registered "
        f"(SCALE_TRANSFER_DESIGN.md §3.4); threshold(s) {THRESHOLDS_ADDED} "
        "are ADDED for this run per its explicit instructions, not part of the "
        "original pre-registration."
    )
    lines.append("")
    lines.append(
        "**The '100-steps-vs-0.23' teaser remains untraceable (§3.3).** "
        "Everything below is NEW analysis, not verification of a prior number."
    )
    lines.append("")

    lines.append("## Headline: geo3 (primary tier) vs baseline, h=1/h=2/h=4, all thresholds")
    lines.append("")
    lines.append("| K | hop | threshold | prereg? | baseline steps (interp, mean) | geo3 steps (interp, mean) | ratio | note |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in summary["headline_ratios"]:
        b = row["baseline_cell"]
        g = row["geo3_cell"]
        r = row["ratio"]
        ratio_str = r.get("ratio_x") or r.get("lower_bound_x") or "n/a"
        prereg = "yes" if row["threshold_preregistered"] else "ADDED"
        lines.append(
            f"| {row['K']} | h={row['hop']} | {row['threshold']} | {prereg} | "
            f"{_display_step(b)} | "
            f"{_display_step(g)} | {ratio_str} | {r['note']} |"
        )
    lines.append("")

    lines.append("## Per-seed detail (headline hops, threshold=0.9)")
    lines.append("")
    for c in summary["aggregated"]:
        if c["threshold"] != 0.9 or c["hop_key"] not in (
            "M2_in_distribution:h1", "M2_in_distribution:h2", "M3_held_out:h4"
        ):
            continue
        seeds_str = ", ".join(
            f"s{p['seed']}={p['interpolated_step'] if p['interpolated_step'] is not None else p['status']}"
            for p in c["per_seed"]
        )
        lines.append(
            f"- **{c['arm_label']}, K={c['K']}, tier={c['tier']}, {c['hop_key']}**: "
            f"{seeds_str}"
        )
    lines.append("")

    lines.append("## n_iter=12 vs n_iter=20 trajectory-match bonus readout (K=32 geo3)")
    lines.append("")
    lines.append("| hop | threshold | n12 mean step | n20 mean step | |delta| | within one checkpoint (2000 steps)? |")
    lines.append("|---|---|---|---|---|---|")
    for row in summary["n12_vs_n20_bonus"]:
        lines.append(
            f"| h={row['hop']} | {row['threshold']} | {row['n12_interpolated_step_mean']} | "
            f"{row['n20_interpolated_step_mean']} | {row['abs_delta_steps']} | "
            f"{row['within_one_checkpoint_resolution']} |"
        )
    lines.append("")

    lines.append("## Admissibility tags (two-tier K=32, bonus K=48)")
    lines.append("")
    for row in summary["rows"]:
        if row["arm"] != "geo3":
            continue
        adm = row.get("geo3_admission", {})
        lines.append(
            f"- K={row['K']} seed={row['seed']} tier={row['tier']} "
            f"admissible={adm.get('admissible')} "
            f"(ns_converged_no_fallback={adm.get('ns_converged_no_fallback')}, "
            f"value_salvage_tier_pass={adm.get('value_salvage_tier_pass')}) "
            f"in_manifest={row['in_manifest']}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    rows = [analyze_file(m) for m in manifest]
    agg = aggregate(rows)
    headline = build_headline_ratios(agg)
    n12_vs_n20 = build_n12_vs_n20_bonus(agg)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "archive_root": str(ARCHIVE_ROOT.relative_to(REPO_ROOT)),
        "design_doc": "matrix-thinking/SCALE_TRANSFER_DESIGN.md §3 (Rev 2)",
        "n_files_analyzed": len(rows),
        "expected_n_params": EXPECTED_N_PARAMS,
        "thresholds_preregistered": THRESHOLDS_PREREGISTERED,
        "thresholds_added": THRESHOLDS_ADDED,
        "checkpoint_resolution_steps": 2000,
        "caveats": [
            "Checkpoint grid resolution is 2000 steps; interpolated_step is a linear "
            "estimate between bracketing checkpoints, not a finer measurement.",
            "The '100-steps-vs-0.23' teaser is untraceable in this repo (SCALE_TRANSFER_DESIGN.md §3.3); "
            "this file is NEW analysis, not verification of that number.",
            "K=48 (baseline + geo3) and arm i-strong (K=32) are OUTSIDE SCALE_TRANSFER_DESIGN.md "
            "§3.5's pre-registered manifest; included as bonus cells, flagged in_manifest=False.",
            "K=32 geo3 has two tiers: n_iter=20 (admissible 3/3, PRIMARY) and n_iter=12 "
            "(non-admissible 0/3, descriptive-only). K=48 geo3 is non-admissible 0/3 for a "
            "DIFFERENT reason (value_salvage_tier_pass=False, not an NS fallback).",
            "n=2-3 seeds per cell; per-seed results are always reported alongside pooled means.",
        ],
        "rows": rows,
        "aggregated": agg,
        "headline_ratios": headline,
        "n12_vs_n20_bonus": n12_vs_n20,
    }

    json_path = args.out_dir / "track_a_sample_efficiency.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = args.out_dir / "track_a_summary.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(summary))

    print(f"Analyzed {len(rows)} archived run files, zero GPU.")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
