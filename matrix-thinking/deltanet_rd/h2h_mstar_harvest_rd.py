"""h2h_mstar_harvest_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.41's axis-2 M* HARVEST:
reads the RAW 90-pass fan-out JSONs (capped transformer acc_A) + the contender horizon-ref
JSONs, builds the per-(task, M) paired per-seed gap arrays at the PRIMARY horizon (H4, 902
tokens), runs the frozen sec 1.4.2 descending Maurer-Hothorn-Lehmacher walk (via
h2h_mstar_analysis_rd, at the sec 1.31.1 re-registered acc_A metric, 0.20/point margin), and
applies the two frozen-registration clauses the walk alone cannot see:

  - DEGENERATE-BASELINE (sec 1.31.1): if the UNCAPPED transformer fails the 3x-chance
    demonstration bar at the primary cell (its sweep-harvest acc_A reads, sec 1.40:
    [0.02710, 0.02930, 0.02856] -- at chance), the M* walk is definitionally degenerate;
    the verdict of record is "baseline non-competitive at matched params/tokens", NEVER
    certified M*=inf/strongest-win. The walk is still run and reported as informative data
    (capping could in principle HELP via forced locality -- report what the data says).
  - JOINT-NO-RECALL (Rev 5.1, sec 1.32 M3): a grid point where BOTH arms sit at or below
    the bar is TIE-equivalent -- walked past, never sets M* (h2h_mstar_analysis_rd's own
    tie_equivalent_ms machinery, smoke-8-protected).

Pure statistics over the archived JSONs -- no torch, no GPU. Run the smoke gate:
  python h2h_mstar_harvest_rd.py --smoke
Real harvest (after the box fan-out completes):
  python h2h_mstar_harvest_rd.py --fanout-dir <dir> --contender-refs <json> \
      --uncapped-transformer-primary '<json list of 3 per-seed sweep acc_A>' --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_mstar_analysis_rd import (compute_ci_from_raw_seed_values, run_mstar_procedure,
                                   ELIGIBLE_M_DESCENDING, MARGIN_DEFAULT)

TASKS = ("task1_sweep", "task2_sweep")   # task1 PRIMARY, task2 secondary non-gating (sec 1.4.2)
HORIZONS = ("H2", "H4", "H8")
PRIMARY_HORIZON = "H4"
N_SEEDS = 3


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _require_acc_a(doc: dict, what: str) -> float:
    """Hard-refuses an rf-era result doc (sec 1.38 pre-flight item 1's re-registration,
    enforced at HARVEST too, not only at fan-out validity time)."""
    assert "acc_A" in doc, (
        f"{what}: result doc has no acc_A -- an rf-era artifact cannot enter the M* walk "
        f"(sec 1.31.1 re-registration); re-run the pass with the fixed capped_eval_pass")
    return float(doc["acc_A"])


def load_fanout_acc(fanout_dir: str, task: str) -> tuple[dict, float]:
    """{M: {horizon: [per-seed acc_A]}} + the demonstration bar (asserted identical
    across every doc for the task -- one K, one bar)."""
    by_m: dict = {}
    bar = None
    for m in ELIGIBLE_M_DESCENDING:
        by_m[m] = {}
        for h in HORIZONS:
            vals = []
            for s in range(N_SEEDS):
                path = os.path.join(fanout_dir, f"h2h_msweep_{task}_s{s}_M{m}_{h}.json")
                doc = _load_json(path)
                vals.append(_require_acc_a(doc, path))
                b = doc.get("demonstration_bar")
                assert b is not None, f"{path}: no demonstration_bar recorded"
                assert bar is None or abs(bar - b) < 1e-12, (
                    f"{path}: demonstration_bar {b} != {bar} seen earlier for {task}")
                bar = b
            by_m[m][h] = vals
    return by_m, bar


def load_refs_acc(refs_path: str, task: str) -> dict:
    """{horizon: [per-seed acc_A]} from a mode_horizon_ref output JSON."""
    doc = _load_json(refs_path)
    out = {}
    for h in HORIZONS:
        out[h] = [_require_acc_a(doc[f"{task}|s{s}|{h}"], f"{refs_path}[{task}|s{s}|{h}]")
                  for s in range(N_SEEDS)]
    return out


def mean(vals: list) -> float:
    return sum(vals) / len(vals)


def harvest_task(task: str, fanout_dir: str, contender_refs_path: str,
                 uncapped_transformer_primary: list[float],
                 transformer_uncapped_refs_path: str | None = None,
                 m1_descriptive_path: str | None = None,
                 margin: float = MARGIN_DEFAULT) -> dict:
    capped, bar = load_fanout_acc(fanout_dir, task)
    contender = load_refs_acc(contender_refs_path, task)
    cont_h4 = contender[PRIMARY_HORIZON]

    # sec 1.31.1 joint-NO-RECALL classification per eligible M, at the decision horizon,
    # mean-of-seeds with per-seed disclosure (the sec 1.40 harvest convention).
    joint_no_recall = set()
    for m in ELIGIBLE_M_DESCENDING:
        if mean(cont_h4) <= bar and mean(capped[m][PRIMARY_HORIZON]) <= bar:
            joint_no_recall.add(m)

    cis_by_m = {m: compute_ci_from_raw_seed_values(cont_h4, capped[m][PRIMARY_HORIZON])
                for m in ELIGIBLE_M_DESCENDING if m not in joint_no_recall}
    walk = run_mstar_procedure(cis_by_m, margin, tie_equivalent_ms=joint_no_recall)

    # DEGENERATE-BASELINE clause (sec 1.31.1): keyed on the UNCAPPED transformer's own
    # primary-cell demonstration-bar failure (the sweep-harvest T_bind reads, sec 1.40).
    clause_fires = mean(uncapped_transformer_primary) <= bar
    if clause_fires:
        verdict_of_record = "BASELINE_NON_COMPETITIVE_AT_MATCHED_BUDGET"
        # never certified as M*=inf / strongest-win under the clause
        walk = dict(walk, confirmed_no_crossover=False,
                    degenerate_baseline_note=(
                        "uncapped transformer fails the demonstration bar at the primary "
                        "cell -- every capped M is trivially cleared; walk reported as "
                        "informative data only, never a certified M* tier"))
    else:
        verdict_of_record = walk["tier"]

    result = {
        "task": task,
        "demonstration_bar": bar,
        "margin": margin,
        "contender_refs_acc_A": contender,
        "capped_transformer_acc_A": {str(m): capped[m] for m in ELIGIBLE_M_DESCENDING},
        "cis_by_m_H4": {str(m): cis_by_m[m] for m in cis_by_m},
        "joint_no_recall_ms": sorted(joint_no_recall),
        "zero_seed_variance_ms": sorted(m for m, ci in cis_by_m.items()
                                        if ci.get("zero_seed_variance")),
        "walk": walk,
        "uncapped_transformer_primary_cell_acc_A": uncapped_transformer_primary,
        "degenerate_baseline_clause_fires": clause_fires,
        "verdict_of_record": verdict_of_record,
    }
    if transformer_uncapped_refs_path:
        result["transformer_uncapped_refs_acc_A"] = load_refs_acc(
            transformer_uncapped_refs_path, task)
    if m1_descriptive_path:
        result["m1_descriptive_acc_A"] = load_refs_acc(m1_descriptive_path, task)
    return result


def print_m_by_m_table(r: dict) -> None:
    task = r["task"]
    print(f"\n=== {task} (bar {r['demonstration_bar']:.5f}, margin {r['margin']}) ===")
    print("arm/M          " + "".join(f"{h:>26}" for h in HORIZONS))
    cont = r["contender_refs_acc_A"]
    print("contender      " + "".join(
        f"{'/'.join(f'{v:.3f}' for v in cont[h]):>26}" for h in HORIZONS))
    if "transformer_uncapped_refs_acc_A" in r:
        tu = r["transformer_uncapped_refs_acc_A"]
        print("tfm uncapped   " + "".join(
            f"{'/'.join(f'{v:.3f}' for v in tu[h]):>26}" for h in HORIZONS))
    if "m1_descriptive_acc_A" in r:
        m1 = r["m1_descriptive_acc_A"]
        print("tfm M=1 (desc) " + "".join(
            f"{'/'.join(f'{v:.3f}' for v in m1[h]):>26}" for h in HORIZONS))
    for m in ELIGIBLE_M_DESCENDING:
        row = r["capped_transformer_acc_A"][str(m)]
        print(f"tfm M={m:<9}" + "".join(
            f"{'/'.join(f'{v:.3f}' for v in row[h]):>26}" for h in HORIZONS))
    w = r["walk"]
    print(f"walk: chain={w['chain']} stop_m={w.get('stop_m')} "
          f"skipped_tie_equivalent={w.get('skipped_tie_equivalent')} tier={w['tier']} "
          f"m_star={w.get('m_star')}")
    print(f"degenerate_baseline_clause_fires={r['degenerate_baseline_clause_fires']} "
          f"-> VERDICT OF RECORD: {r['verdict_of_record']}")


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    print(f"[{item}] {'PASS' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def _write_synthetic(dirpath: str, task: str, capped_fn, cont_fn, bar: float = 0.09375):
    """Plants a full synthetic fan-out + refs for one task; returns refs path."""
    for m in ELIGIBLE_M_DESCENDING:
        for h in HORIZONS:
            for s in range(N_SEEDS):
                doc = {"M": m, "horizon": h, "task": task, "seed_idx": s,
                       "acc_A": capped_fn(m, h, s), "demonstration_bar": bar,
                       "recovered_frac": 0.0}
                with open(os.path.join(
                        dirpath, f"h2h_msweep_{task}_s{s}_M{m}_{h}.json"), "w") as f:
                    json.dump(doc, f)
    refs = {}
    for h in HORIZONS:
        for s in range(N_SEEDS):
            refs[f"{task}|s{s}|{h}"] = {"acc_A": cont_fn(h, s), "demonstration_bar": bar}
    refs_path = os.path.join(dirpath, f"refs_{task}.json")
    with open(refs_path, "w") as f:
        json.dump(refs, f)
    return refs_path


def smoke_1_normal_walk_no_clause():
    """Healthy baseline (clause does NOT fire): capped transformer catches up at
    M>=8 (gap CI below margin there), clean stop at 4 -> M*=8 WIN, verdict of
    record = the tier itself."""
    with tempfile.TemporaryDirectory() as td:
        def capped(m, h, s):
            return (0.92 + 0.001 * s) if m >= 8 else (0.30 + 0.001 * s)

        def cont(h, s):
            return 0.99 + 0.001 * s

        refs = _write_synthetic(td, "task1_sweep", capped, cont)
        r = harvest_task("task1_sweep", td, refs, uncapped_transformer_primary=[0.95, 0.96, 0.94])
        ok = (r["degenerate_baseline_clause_fires"] is False and r["walk"]["m_star"] == 8
              and r["verdict_of_record"] == "WIN" and r["joint_no_recall_ms"] == [])
    _report("smoke 1: healthy baseline -> walk verdict IS the verdict of record (M*=8 WIN)",
            ok, f"verdict={r['verdict_of_record']} m_star={r['walk']['m_star']}")


def smoke_2_degenerate_baseline_clause():
    """The sec 1.40 live case: uncapped transformer at chance -> the clause fires;
    the verdict of record is BASELINE_NON_COMPETITIVE..., the walk is reported as
    informative only, and confirmed_no_crossover can never certify under it (teeth:
    the same walk WITHOUT the clause would read CONFIRMED no-crossover WIN)."""
    with tempfile.TemporaryDirectory() as td:
        def capped(m, h, s):
            return 0.028 + 0.001 * s     # capped transformer at chance everywhere

        def cont(h, s):
            return 0.99 + 0.001 * s      # contender at ceiling

        refs = _write_synthetic(td, "task1_sweep", capped, cont)
        r_fires = harvest_task("task1_sweep", td, refs,
                               uncapped_transformer_primary=[0.0271, 0.0293, 0.0286])
        r_healthy = harvest_task("task1_sweep", td, refs,
                                 uncapped_transformer_primary=[0.95, 0.96, 0.94])
        ok = (r_fires["degenerate_baseline_clause_fires"] is True
              and r_fires["verdict_of_record"] == "BASELINE_NON_COMPETITIVE_AT_MATCHED_BUDGET"
              and r_fires["walk"]["confirmed_no_crossover"] is False
              and "degenerate_baseline_note" in r_fires["walk"]
              and r_healthy["walk"]["confirmed_no_crossover"] is True)   # teeth
    _report("smoke 2 (sec 1.31.1 degenerate-baseline): clause fires -> verdict of record is "
            "baseline-non-competitive, never certified M*=inf (teeth: same data without the "
            "clause WOULD certify)", ok,
            f"fires_verdict={r_fires['verdict_of_record']} "
            f"healthy_confirmed={r_healthy['walk']['confirmed_no_crossover']}")


def smoke_3_joint_no_recall_feeds_walk():
    """A grid point where BOTH arms are below the bar is classified joint-NO-RECALL
    here and skipped by the walk (contender below bar requires a horizon-collapsed
    contender -- built synthetically)."""
    with tempfile.TemporaryDirectory() as td:
        def capped(m, h, s):
            return 0.02      # capped transformer below bar everywhere

        def cont(h, s):
            return 0.03      # contender ALSO below bar (synthetic collapse)

        refs = _write_synthetic(td, "task1_sweep", capped, cont)
        r = harvest_task("task1_sweep", td, refs, uncapped_transformer_primary=[0.95, 0.96, 0.94])
        ok = (r["joint_no_recall_ms"] == [2, 4, 8, 16, 32]
              and r["walk"]["tier"] == "TIE" and r["walk"]["m_star"] is None)
    _report("smoke 3 (sec 1.32 M3 joint-NO-RECALL): both arms below bar at every M -> all "
            "cells TIE-equivalent -> joint-failure TIE, never a LOSE/WIN", ok,
            f"joint_ms={r['joint_no_recall_ms']} tier={r['walk']['tier']}")


def smoke_4_rf_era_doc_hard_refused():
    """NEGATIVE TEETH, run to completion: an rf-era fan-out doc (no acc_A) must
    hard-fail the harvest, never silently enter the walk."""
    with tempfile.TemporaryDirectory() as td:
        refs = _write_synthetic(td, "task1_sweep", lambda m, h, s: 0.5, lambda h, s: 0.99)
        rf_path = os.path.join(td, "h2h_msweep_task1_sweep_s0_M32_H4.json")
        with open(rf_path, "w") as f:
            json.dump({"M": 32, "horizon": "H4", "task": "task1_sweep", "seed_idx": 0,
                       "recovered_frac": 0.5, "demonstration_bar": 0.09375}, f)   # NO acc_A
        raised = False
        try:
            harvest_task("task1_sweep", td, refs, uncapped_transformer_primary=[0.95, 0.96, 0.94])
            print("NEGATIVE FAILED TO FAIL: rf-era doc entered the harvest", file=sys.stderr)
        except AssertionError as e:
            raised = "acc_A" in str(e)
    _report("smoke 4 (NEGATIVE TEETH): an rf-era doc (no acc_A) hard-fails the harvest", raised)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fanout-dir", type=str)
    ap.add_argument("--contender-refs", type=str)
    ap.add_argument("--transformer-uncapped-refs", type=str, default=None)
    ap.add_argument("--m1-descriptive", type=str, default=None)
    ap.add_argument("--uncapped-transformer-primary", type=str,
                    help="JSON list: the sweep-harvest per-seed acc_A of the UNCAPPED "
                         "transformer at the task's primary cell (degenerate-baseline basis); "
                         "keyed by task as a JSON object, or a bare list applied to task1 only")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        print("=" * 70)
        print("h2h_mstar_harvest_rd.py -- sec 1.41 M* harvest smoke suite")
        print("=" * 70)
        smoke_1_normal_walk_no_clause()
        smoke_2_degenerate_baseline_clause()
        smoke_3_joint_no_recall_feeds_walk()
        smoke_4_rf_era_doc_hard_refused()
        print("=" * 70)
        if FAILURES:
            print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
            return 1
        print("SMOKE SUITE: ALL ITEMS PASSED")
        return 0

    assert args.fanout_dir and args.contender_refs and args.uncapped_transformer_primary, (
        "real harvest needs --fanout-dir, --contender-refs, --uncapped-transformer-primary")
    primary = json.loads(args.uncapped_transformer_primary)
    if isinstance(primary, list):
        primary = {"task1_sweep": primary}
    out = {"margin": MARGIN_DEFAULT, "primary_horizon": PRIMARY_HORIZON, "per_task": {}}
    for task in TASKS:
        if task not in primary:
            print(f"NOTE: no uncapped-transformer primary reads for {task}; skipping "
                  f"(task2 is secondary/non-gating)")
            continue
        r = harvest_task(task, args.fanout_dir, args.contender_refs, primary[task],
                         transformer_uncapped_refs_path=args.transformer_uncapped_refs,
                         m1_descriptive_path=args.m1_descriptive)
        out["per_task"][task] = r
        print_m_by_m_table(r)
    if args.out:
        with open(args.out + ".tmp", "w") as f:
            json.dump(out, f, indent=2)
        os.replace(args.out + ".tmp", args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
