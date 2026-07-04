"""bands_pinned_trackb.py -- TRACKB_REDESIGN.md Rev 3 sec 4.3's
BANDS_PINNED-TrackB machinery: the writer / launcher-gate validator /
readout-timestamp assertion, reusing KEY_ANCHORING_DESIGN.md Rev 5 sec 3.6's
house `BANDS_PINNED` pattern BY NAME (sec 4.3: "Mechanics, reused from sec
3.6 by name, all three parts"). key_anchoring.py's own
write_bands_pinned/validate_bands_pinned/assert_blind_not_broken is the
precedent this module mirrors structurally -- NOT imported wholesale
(the CONTENT differs completely: churn ceiling/support band/positional-TV
ceiling/B_pinned distribution here, vs. per-K engaged-drift bands there),
but `sha256_of_file` (a pure, content-agnostic utility) IS imported from
key_anchoring.py rather than re-duplicated.

Covers, per sec 4.3/sec 2 principle 4 (all four "parts" the task brief's
deliverable 5 names):
  1. Churn ceiling: max(Null A [unmasked-pilot mean+2s], Null B [this run's
     own first-10%-of-steps mean+2s]) -- Null A is PINNED here at Wave -1;
     Null B is computed PER RUN at readout time (churn_ceiling_for_run).
  2. Support-size band: [max(4, p10_pilot), k_sel] (candidate 2's floor/
     ceiling).
  3. TV-distance positional-artifact ceiling: mean+2s over the unmasked
     pilot's own TV-from-uniform.
  4. Per-chunk B_pinned distribution (sec 2 principle 4's baseline
     per-chunk total write mass + its own dispersion, feeding every masking
     candidate's shortfall accounting).

Mechanics (sec 4.3, all three parts): (1) the WRITER writes
BANDS_PINNED-TrackB.json only after every pilot validates complete; (2) the
LAUNCHER GATE (Wave 1/2 cells) refuses to launch unless the file exists,
validates, and re-hashes clean; (3) the READOUT ASSERTION checks the pin
timestamp strictly precedes the earliest Wave-1 start time, else every
affected readout demotes to descriptive tier.
"""
from __future__ import annotations

import json
import os
import time

import torch

from key_anchoring import sha256_of_file  # content-agnostic utility, reused not re-duplicated


# ---------------------------------------------------------------------------
# Derivation rules (sec 4.3) -- registered NOW, applied mechanically at Wave
# -1 readout. All use the "mean + 2*sample_std" house formula
# (KEY_ANCHORING_DESIGN.md Rev 5 sec 3.6's own `derive_engaged_bands`
# convention, restated here for Track B's different quantities).
# ---------------------------------------------------------------------------

def _mean_plus_2s(values: list) -> dict:
    t = torch.tensor([float(v) for v in values], dtype=torch.float64)
    mean = t.mean().item()
    s = t.std(unbiased=True).item() if t.numel() > 1 else 0.0
    return {"mean": mean, "s": s, "n": t.numel(), "ceiling": mean + 2.0 * s}


def derive_churn_null_a(pilot_churn_last5_logpoints: list) -> dict:
    """sec 4.3 Null A: mean_ref + 2*s_ref over the UNMASKED, Cell-1-
    architecture reference pilot's implicit top-K_sel-by-beta churn at its
    last 5 log points (steps past init transients). `pilot_churn_last5_
    logpoints` is that list of scalar churn values (one per log point,
    already pooled over chunk/head episodes by the caller).

    len==5 EXACTLY (AUDIT FIX, independent audit 2026-07-04): the spec
    registers "its last 5 log points" -- a shorter list means the pilot did
    not run/checkpoint long enough (or the extractor mis-sliced) and the
    pin must REFUSE, not degrade to a noisier estimate; a longer list means
    an un-sliced trajectory, silently widening the null past its
    registration."""
    assert len(pilot_churn_last5_logpoints) == 5, (
        f"Null A requires EXACTLY the last 5 log points (sec 4.3's registered convention), got "
        f"{len(pilot_churn_last5_logpoints)} -- use extract_selection_logpoint_lists on a pilot "
        f"with enough checkpoints, never a hand-sliced or shorter list")
    return _mean_plus_2s(pilot_churn_last5_logpoints)


def derive_pos_ceiling(pilot_tv_last5_logpoints: list) -> dict:
    """sec 4.3 (Rev 3 NEW-5): mean_ref + 2*s_ref over the unmasked
    reference pilot's own TV-from-uniform at its last 5 log points -- same
    reference-null shape as churn Null A, same EXACTLY-5 requirement (see
    derive_churn_null_a's audit-fix note)."""
    assert len(pilot_tv_last5_logpoints) == 5, (
        f"positional-concentration ceiling requires EXACTLY the last 5 log points, got "
        f"{len(pilot_tv_last5_logpoints)} -- see derive_churn_null_a's len==5 note")
    return _mean_plus_2s(pilot_tv_last5_logpoints)


def extract_selection_logpoint_lists(result: dict, last_n: int = 5, need_churn: bool = True) -> dict:
    """AUDIT FIX (independent audit 2026-07-04 -- F2's reader half): turns a
    COMPLETED lm_pretrain_rd.py result JSON (parsed dict) into exactly the
    inputs derive_churn_null_a / derive_pos_ceiling / derive_support_floor
    expect. Reads the per-checkpoint `hard_select_diagnostics` blocks
    train() writes (lm_pretrain_rd.py's sample_hard_select_diagnostics --
    present when the run was hard_select_active OR ran with
    --trackb-selection-probe, the unmasked reference pilot's own
    implicit-selection instrumentation).

    need_churn (AUDIT FIX round 2, 2026-07-04 -- FATAL-1): the entmax
    probe's diagnostics NEVER carry churn (sample_hard_select_diagnostics'
    entmax branch sets churn_vs_prev=None by design -- variable support has
    no fixed-K set difference; sec 4.3 assigns the churn bar to candidate
    1/hard-snap only), so the round-1 unconditional churn>=last_n assertion
    fired for EVERY possible real entmax probe and made the bands assembly
    (which only needs the entmax probe's SUPPORT numbers) structurally
    unable to run. Callers that only need support/TV pass need_churn=False;
    the churn list is then returned as-is (possibly empty), unasserted.

    Per-checkpoint scalars are pooled across layers by MEAN (sec 4.3
    defines pooling over (chunk, head) episodes; layers are not addressed
    there -- mean-over-layers is this build's own registered reading,
    stated here rather than buried).

    Returns {"churn": last_n values (the first checkpoint has no previous
    selection -> contributes None, excluded before slicing), "tv": last_n
    values, "support_median_final", "support_p10_final" (the sec 4.3
    support-floor input), "n_checkpoints"}. Raises AssertionError (refuse,
    never silently pad) if fewer than last_n usable values exist."""
    checkpoints = result.get("checkpoints") or []
    diags = [c["hard_select_diagnostics"] for c in checkpoints
             if isinstance(c.get("hard_select_diagnostics"), dict)]
    assert diags, (
        "result JSON carries no hard_select_diagnostics blocks -- the run was neither "
        "hard_select_active nor --trackb-selection-probe'd; not a valid pilot/gated run for the "
        "sec 4.3 derivations")

    def _layer_mean(d: dict, key: str):
        vals = [layer[key] for layer in d["per_layer"].values() if layer.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    churn_series = [v for v in (_layer_mean(d, "churn_vs_prev") for d in diags) if v is not None]
    tv_series = [v for v in (_layer_mean(d, "tv_from_uniform") for d in diags) if v is not None]
    if need_churn:
        assert len(churn_series) >= last_n, (
            f"only {len(churn_series)} churn log points available, need >= {last_n} (churn needs "
            f"a previous selection, so the first checkpoint contributes none -- the pilot must "
            f"checkpoint at least {last_n + 1} times; for a churn-less entmax probe pass "
            f"need_churn=False)")
    assert len(tv_series) >= last_n, f"only {len(tv_series)} TV log points, need >= {last_n}"
    final = diags[-1]
    return {
        "churn": churn_series[-last_n:] if need_churn else churn_series,
        "tv": tv_series[-last_n:],
        "support_median_final": _layer_mean(final, "support_median"),
        "support_p10_final": _layer_mean(final, "support_p10"),
        "n_checkpoints": len(diags),
    }


def derive_support_floor(p10_support_final: float) -> dict:
    """sec 4.3: support_floor = max(4, p10_pilot) -- the pilot's own 10th-
    percentile support at its FINAL log point, floored at 4 (below 4
    selected writes per chunk, geo3 orthogonalization is near-trivial)."""
    floor = max(4.0, float(p10_support_final))
    return {"p10_pilot": float(p10_support_final), "floor": floor}


def derive_b_pinned(per_chunk_total_mass: torch.Tensor) -> dict:
    """sec 2 principle 4: B_pinned = the unconstrained baseline's OWN MEAN
    per-chunk total write mass (Cell 1's same-instrument re-measurement,
    sec 5.3), plus its own per-chunk dispersion -- feeds the predicted-
    shortfall distribution every masking candidate's realized shortfall is
    later read against (sec 2 principle 4's own registered requirement).
    per_chunk_total_mass: any-shape tensor of per-(chunk,head[,batch,...])
    total mass values (hard_selectivity_rd.per_chunk_total_mass's output,
    flattened by the caller)."""
    flat = per_chunk_total_mass.reshape(-1).double()
    return {
        "b_pinned": flat.mean().item(),
        "std": flat.std(unbiased=True).item() if flat.numel() > 1 else 0.0,
        "p10": torch.quantile(flat, 0.10).item(),
        "p90": torch.quantile(flat, 0.90).item(),
        "n_chunks_pooled": int(flat.numel()),
    }


# ---------------------------------------------------------------------------
# Writer / launcher-gate validator / readout-timestamp assertion (sec 4.3's
# three-part mechanics, KEY_ANCHORING Rev 5 sec 3.6 pattern reused by name).
# ---------------------------------------------------------------------------

def write_bands_pinned_trackb(path: str, churn_null_a: dict, pos_ceiling: dict, support_floor: dict,
                               b_pinned: dict, mc_anchors: dict, cell1_ref_32: dict,
                               pilot_result_paths: list, k_sel: int = 32,
                               formula_version: str = "TRACKB_REDESIGN.md sec 4.3/5.3 Rev 3") -> dict:
    """Writer requirement (1): the CALLER is responsible for having already
    verified every pilot result JSON validates complete (mirrors
    key_anchoring.write_bands_pinned's own division of labor) -- this
    function's job is to assemble the pinned numbers + sha256 hashes of
    every referenced pilot JSON + timestamp, matching the design's own
    content spec (derived bands + per-pilot inputs + formula version +
    sha256 hashes + timestamp)."""
    hashes = {p: sha256_of_file(p) for p in pilot_result_paths}
    doc = {
        "formula_version": formula_version,
        "k_sel": k_sel,
        "churn_null_a": churn_null_a,
        "positional_concentration_ceiling": pos_ceiling,
        "support_band": {**support_floor, "ceiling_a_priori": k_sel},
        "b_pinned": b_pinned,
        "mc_anchors": mc_anchors,
        "cell1_ref_32": cell1_ref_32,
        "pilot_result_paths": list(pilot_result_paths),
        "pilot_result_sha256": hashes,
        "pinned_at": time.time(),
        "pinned_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def validate_bands_pinned_trackb(path: str) -> dict | None:
    """Launcher-gate requirement (2): Wave 1/2 cells REFUSE to launch
    unless this file exists, validates, and re-hashes clean. Returns the
    parsed doc on success, None on ANY validation failure (missing file,
    missing field, hash mismatch) -- callers own the loud refusal message."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception:
        return None
    for p, expected in doc.get("pilot_result_sha256", {}).items():
        if not os.path.exists(p):
            return None
        if sha256_of_file(p) != expected:
            return None
    required = ("churn_null_a", "positional_concentration_ceiling", "support_band", "b_pinned",
                "mc_anchors", "cell1_ref_32", "pinned_at")
    if not all(k in doc for k in required):
        return None
    return doc


def assert_blind_not_broken_trackb(bands_doc: dict, wave1_start_ats: list) -> None:
    """Readout-assertion requirement (3): pin timestamp must STRICTLY
    PRECEDE the earliest Wave-1 cell start time. Raises AssertionError (the
    readout demotes to descriptive tier) if violated."""
    assert wave1_start_ats, "no Wave-1 start times given -- nothing to check the pin against"
    earliest = min(wave1_start_ats)
    pinned_at = bands_doc["pinned_at"]
    assert pinned_at < earliest, (
        f"BANDS_PINNED-TrackB BLIND BROKEN: pinned_at={pinned_at} does NOT strictly precede the "
        f"earliest Wave-1 start_at={earliest} (sec 4.3's mechanical readout assertion). Every "
        f"affected readout must report at descriptive tier only.")


def churn_ceiling_for_run(bands_doc: dict, run_own_early_churn_values: list) -> dict:
    """sec 4.3 (Rev 3 NEW-4): the registered ceiling for a GIVEN run = max(
    pinned Null A, that run's OWN Null B [mean+2s over its first-10%-of-
    steps churn log points]). Null B is necessarily computed PER RUN (it
    depends on that run's own early-training trajectory), never pinned at
    Wave -1 -- this function is called at readout time, once per Wave-1
    candidate-1 cell."""
    null_a_ceiling = bands_doc["churn_null_a"]["ceiling"]
    null_b = _mean_plus_2s(run_own_early_churn_values) if len(run_own_early_churn_values) >= 2 else \
        {"mean": run_own_early_churn_values[0] if run_own_early_churn_values else float("nan"),
         "s": 0.0, "n": len(run_own_early_churn_values),
         "ceiling": run_own_early_churn_values[0] if run_own_early_churn_values else float("nan")}
    ceiling = max(null_a_ceiling, null_b["ceiling"])
    return {"null_a": bands_doc["churn_null_a"], "null_b": null_b, "ceiling": ceiling,
            "binding_null": "A" if null_a_ceiling >= null_b["ceiling"] else "B",
            "calibration_limitation": (
                "both nulls are approximations (Null A cannot see mask-specific dynamics; Null B "
                "overweights init transients) -- a coarse degeneracy screen, not a calibrated "
                "bound; a breach routes to diagnosis, never to silent acceptance or rejection "
                "(TRACKB_REDESIGN.md sec 4.3, Rev 3 NEW-4)."),
            }


def classify_selection_degenerate(steady_state_churn: float, ceiling_dict: dict) -> bool:
    """sec 4.3: a Wave-1 candidate-1 cell whose STEADY-STATE churn
    (checkpoint-resolution, past the first 10% of steps) exceeds the
    ceiling is flagged selection-degenerate -- excluded from Cell-4
    inheritance, reported descriptively."""
    return steady_state_churn > ceiling_dict["ceiling"]


def classify_support_degenerate(median_support: float, support_band: dict) -> bool:
    """sec 4.3: median support outside [floor, k_sel_ceiling] (checked past
    20% of training) = degenerate-collapse flag, cell excluded from the
    factorial, reported descriptively."""
    return not (support_band["floor"] <= median_support <= support_band["ceiling_a_priori"])


def classify_positionally_degenerate(final_tv: float, pos_ceiling: dict) -> bool:
    """sec 4.3 (Rev 3 NEW-5): a gated run whose final-checkpoint TV exceeds
    pos_ceiling is flagged positionally degenerate -- excluded from Cell-4
    inheritance, reported descriptively."""
    return final_tv > pos_ceiling["ceiling"]
