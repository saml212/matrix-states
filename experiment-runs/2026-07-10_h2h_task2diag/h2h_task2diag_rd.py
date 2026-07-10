#!/usr/bin/env python
"""h2h_task2diag_rd.py -- sec 1.42 runner: the TASK2 DIAGNOSIS ROUND (the sec 1.8
seed-extension template, 6 fresh seeds x contender+ablation = 12 cells, trained
IDENTICALLY to the sec 1.38 sweep cohort) + the transformer_task1_stress_K48 fresh
cell via sec 1.37's own "runnable in the sweep stage" path (sec 1.42 A6 tiebreak:
role="sweep" in the TRAINING config only -- the AUD2-F4 structural dial guard --
because the dial-armed calibration path deterministically re-fires DIAL_EXHAUSTED
at step 500; the re-metric spec keeps role="stress_locate_only" verbatim).

Every metric read goes through the sec 1.31.4-item-6 audited `run_cell_round4`
(loader-side md5 provenance pinning, pinned EVAL_SEED episodes, frozen sec 1.31.1
bar/tiers). NO new metrics, NO new thresholds. Analysis: pooled n=9 paired CI
(pinned t=2.306) behind the sec 16.19.5 `batch_effect_gate` (sec 1.42 A2), the
A3 adjudication map applied mechanically, K48 3-arm locate-only table.

Box (from /home/nvidia/chapter2/deltanet_rd, tdenv python, GPU 7 via the stage
script h2h_task2diag_stage.sh -- tmux h2h_task2diag).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rd_episode_seed import rd_episode_seed                            # noqa: E402

TASK2DIAG_SEED_IDXS = tuple(range(3, 9))     # extends the sweep's s0-2 to n=9 (sec 1.8)
DIAG_ARMS = ("contender", "ablation")        # sec 1.42 A1: the sec 1.8 "BOTH compared arms"
BAR_K32, CHANCE_K32 = 0.09375, 0.03125       # frozen sec 1.31.1 (MARGINS_FROZEN verbatim)
MARGIN = 0.30                                # frozen sec 1.31.1 WIN/LOSE margin
BAR_K48, CHANCE_K48 = 0.0625, 1.0 / 48.0     # locate-only disclosure only, never a tier
K48_CELL_ID = "transformer_task1_stress_K48"


def diag_cells() -> list[dict]:
    """12 seed-extension cells, shaped exactly like sweep_cells() entries (same
    TASK_BASE key, same name series continuing s3..s8, same role='sweep')."""
    cells = []
    for arch in DIAG_ARMS:
        for i in TASK2DIAG_SEED_IDXS:
            cells.append({"arch": arch, "task": "task2_sweep", "seed_idx": i,
                          "seed": rd_episode_seed("task2_sweep", seed_idx=i, ckpt_idx=0),
                          "name": f"h2h_{arch}_task2_sweep_s{i}",
                          "budget_frac": 1.0, "role": "sweep", "lr": 3e-4})
    assert len(cells) == 12, f"expected 12 diag cells, got {len(cells)}"
    names = [c["name"] for c in cells]
    assert len(names) == len(set(names)), "diag cell name collision"
    return cells


def build_k48_spec(fresh_cfgs: dict) -> dict:
    """sec 1.42 A6: the ROUND4_CELL_SPEC K48 entry + the FRESH_CELL_CONFIGS.json
    cell_config VERBATIM except role='sweep' in the TRAINING config only."""
    from h2h_round4_driver_rd import ROUND4_CELL_SPEC
    spec = dict(next(c for c in ROUND4_CELL_SPEC if c["cell_id"] == K48_CELL_ID))
    cfg = dict(fresh_cfgs[K48_CELL_ID])
    # pin the pre-registered knobs before touching anything (refuse a drifted config)
    assert cfg["role"] == "stress_locate_only" and cfg["K"] == 48, cfg
    assert cfg["budget_frac"] == 0.25 and cfg["lr"] == 3e-4 and cfg["seed"] == 500000, cfg
    assert cfg["name"].endswith("_auxrev2"), cfg
    assert spec["fresh"] is True and spec["role"] == "stress_locate_only", spec
    cfg["role"] = "sweep"                    # AUD2-F4 structural dial guard (sec 1.42 A6)
    spec["cell_config"] = cfg
    return spec


def _valid_remetric(path: str) -> bool:
    """Same validity rule as the sec 1.40 harvest remetric (leg_a + leg_b present)."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return "leg_a" in doc and "leg_b_ridge" in doc


def mode_run_cell(args) -> int:
    """Mirrors h2h_cell_train_rd.mode_run_cell for the 12 diag cells (same gate
    discipline: launch tokens, code-level margin-freeze release, resume-safety)."""
    import h2h_cell_train_rd as ct
    from h2h_sweep_runner_rd import is_valid_result, REQUIRED_RESULT_KEYS
    ct.require_launch_tokens(args.gates_dir)
    ct.require_margins_frozen(args.margins_token)
    cells = {c["name"]: c for c in diag_cells()}
    assert args.run_cell in cells, f"unknown diag cell {args.run_cell!r}"
    cell = ct.apply_margin_freeze_overrides([cells[args.run_cell]], args.margins_token)[0]
    if is_valid_result(args.out):
        print(f"SKIP (already valid): {args.out}")
        return 0
    result = ct.run_one_cell(cell, args.device, args.ckpt_dir)
    result = {**cell, **result}
    assert all(k in result for k in REQUIRED_RESULT_KEYS)
    ct._atomic_dump(args.out, result)
    print(f"CELL COMPLETE: {args.run_cell} final_metric={result['final_metric']} "
          f"wall_s={result['wall_s']:.1f}")
    return 0


def mode_run_k48(args) -> int:
    """Fresh-train + full re-metric of the K48 stress cell in one audited
    run_cell_round4 call (train saves the ckpt BEFORE the rung-2 fit, sec 1.31.4
    item 4, so a metric-pass crash can never lose the weights again)."""
    import h2h_cell_train_rd as ct
    from h2h_round4_driver_rd import run_cell_round4
    ct.require_launch_tokens(args.gates_dir)
    ct.require_margins_frozen(args.margins_token)
    with open(args.fresh_cell_configs) as f:
        spec = build_k48_spec(json.load(f))
    out_path = os.path.join(args.out_dir, f"{K48_CELL_ID}_round4.json")
    if _valid_remetric(out_path):
        print(f"SKIP (already valid): {out_path}")
        return 0
    result = run_cell_round4(spec, {}, args.device, args.out_dir, ckpt_dir=args.ckpt_dir,
                             fresh_steps_override=args.steps_override)
    acc = result["leg_a"]["acc_A"]
    print(f"K48 CELL COMPLETE: acc_A={acc:.4f} (chance {CHANCE_K48:.5f}, locate-only bar "
          f"{BAR_K48} -- gate-EXEMPT, sec 1.31.1) wall_s={result['wall_s']:.1f}")
    return 0


def mode_remetric(args) -> int:
    """Verdict-grade acc_A reads for the 12 diag cells: the sec 1.40 harvest
    protocol verbatim (md5 provenance manifest at build, checked at load)."""
    from h2h_round4_driver_rd import run_cell_round4, _md5_of_file
    specs, manifest = [], {}
    for c in diag_cells():
        path = os.path.join(args.ckpt_dir, f"{c['name']}_r4.pt")
        assert os.path.isfile(path), f"missing diag checkpoint: {path}"
        manifest[c["name"]] = {"path": path, "md5": _md5_of_file(path),
                               "mtime": os.path.getmtime(path)}
        specs.append({"cell_id": c["name"], "arch": c["arch"], "task": "task2_sweep",
                      "K": 32, "role": "sweep_remetric", "fresh": False, "seed": c["seed"]})
    os.makedirs(args.remetric_dir, exist_ok=True)
    for spec in specs:
        out_path = os.path.join(args.remetric_dir, f"{spec['cell_id']}_round4.json")
        if _valid_remetric(out_path):
            print(f"SKIP (already valid): {out_path}")
            continue
        r = run_cell_round4(spec, manifest, args.device, args.remetric_dir)
        print(f"REMETRIC {spec['cell_id']}: acc_A={r['leg_a']['acc_A']:.4f}")
    return 0


def strict_tier(delta_ci: dict, contender_demonstrates: bool, ablation_demonstrates: bool) -> str:
    """The frozen sec 1.31.1 tier arithmetic, applied mechanically (sec 1.42 A4:
    disclosed reading only -- task2 stays non-verdict-bearing for axis 1)."""
    lo, hi = delta_ci["ci_low"], delta_ci["ci_high"]
    if not contender_demonstrates and not ablation_demonstrates:
        return "TIE (joint task-learning failure)"
    if contender_demonstrates and lo >= MARGIN:
        return "WIN"
    if ablation_demonstrates and hi <= -MARGIN:
        return "LOSE"
    if -MARGIN < lo and hi < MARGIN:
        return "TIE"
    return "INDETERMINATE (TIE-adjacent, never a WIN)"


def adjudicate(m: int, b: int) -> str:
    """sec 1.42 A3 map, pinned before data."""
    if b >= 1:
        return ("NOT-CONTENDER-SPECIFIC: >=1 new ablation seed clears the bar -- any "
                "contender-advantage story on task2 is DEAD (disclosed first in any task2 claim)")
    if m >= 1:
        return (f"TRAINABILITY/SEED-VARIANCE CONFIRMED at pooled rate {(1 + m)}/9 -- the "
                "hard-capability-boundary hypothesis is REJECTED for task2 at this scale/budget; "
                "task-difficulty-vs-objective-tuning stays open (curve reads descriptive only)")
    return ("NO REPLICATION: the trainability-variance reframing rests on a single seed "
            "(1/9); the capability-boundary hypothesis is NOT rejected "
            "(least-favorable reading; diagnosis UNRESOLVED-toward-unfavorable)")


def analyze(old_dir: str, new_dir: str, k48_new_path: str, round4_dir: str) -> dict:
    """Pure function over the re-metric JSONs (unit-tested via synthetic dirs)."""
    from reasoning_link_probe import delta_ci_n
    from phase2_trajectory_analysis import batch_effect_gate

    def acc(d, arch, i):
        with open(os.path.join(d, f"h2h_{arch}_task2_sweep_s{i}_round4.json")) as f:
            return json.load(f)["leg_a"]["acc_A"]

    old_idx, new_idx = (0, 1, 2), TASK2DIAG_SEED_IDXS
    cont = {i: acc(old_dir, "contender", i) for i in old_idx}
    abl = {i: acc(old_dir, "ablation", i) for i in old_idx}
    cont.update({i: acc(new_dir, "contender", i) for i in new_idx})
    abl.update({i: acc(new_dir, "ablation", i) for i in new_idx})
    idxs = sorted(cont)
    cont_v, abl_v = [cont[i] for i in idxs], [abl[i] for i in idxs]
    deltas = [c - a for c, a in zip(cont_v, abl_v)]

    m = sum(1 for i in new_idx if cont[i] > BAR_K32)
    b = sum(1 for i in new_idx if abl[i] > BAR_K32)
    gates = {
        "contender_acc": batch_effect_gate([cont[i] for i in old_idx], [cont[i] for i in new_idx]),
        "ablation_acc": batch_effect_gate([abl[i] for i in old_idx], [abl[i] for i in new_idx]),
        "paired_delta": batch_effect_gate([deltas[i] for i in range(3)], deltas[3:]),
    }
    any_flag = any(g["flagged"] for g in gates.values())
    ci = delta_ci_n(cont_v, abl_v)
    cont_mean = sum(cont_v) / len(cont_v)
    abl_mean = sum(abl_v) / len(abl_v)
    tier = strict_tier(ci, cont_mean > BAR_K32, abl_mean > BAR_K32)

    k48 = {}
    for arm, path in (("contender", os.path.join(round4_dir, "contender_task1_stress_K48_round4.json")),
                      ("ablation", os.path.join(round4_dir, "ablation_task1_stress_K48_round4.json")),
                      ("transformer", k48_new_path)):
        with open(path) as f:
            d = json.load(f)
        k48[arm] = {"acc_A": d["leg_a"]["acc_A"], "path": path,
                    "clears_locate_only_bar": d["leg_a"]["acc_A"] > BAR_K48}

    return {
        "task2": {
            "acc_A_contender_by_seed": cont, "acc_A_ablation_by_seed": abl,
            "bar": BAR_K32, "chance": CHANCE_K32,
            "new_contender_seeds_clearing_bar_m": m, "new_ablation_seeds_clearing_bar_b": b,
            "pooled_n9_paired_delta_ci": ci,
            "batch_effect_gates_sec_16_19_5": gates,
            "pooled_reading_decision_grade": not any_flag,
            "strict_tier_reading_disclosed_only": tier
            + ("" if not any_flag else " [NON-DECISION-GRADE: batch-effect gate flagged]"),
            "adjudication_sec_1_42_A3": adjudicate(m, b),
            "horizon_read_trigger_seeds": [i for i in new_idx if cont[i] > BAR_K32],
            "note": "task2 remains NON-verdict-bearing for axis 1 (sec 1.42 A4); "
                    "Nichani caveat travels with every acc_A number",
        },
        "k48_stress_locate_only_3arm_table": {
            **k48, "bar": BAR_K48, "chance": CHANCE_K48,
            "note": "locate-only, gate-EXEMPT (sec 1.31.1/sec 1.38 ADJ-3); no tier arithmetic; "
                    "the round-4 DIAL_EXHAUSTED datum belongs to THIS lane (sec 1.42 tiebreak)",
        },
    }


def mode_analyze(args) -> int:
    verdict = analyze(args.sweep_remetric_dir, args.remetric_dir,
                      os.path.join(args.out_dir, f"{K48_CELL_ID}_round4.json"),
                      args.round4_dir)
    from h2h_cell_train_rd import _atomic_dump
    out = os.path.join(args.out_dir, "TASK2DIAG_VERDICT.json")
    _atomic_dump(out, verdict)
    t2 = verdict["task2"]
    print(f"TASK2DIAG VERDICT written: {out}\n"
          f"  m={t2['new_contender_seeds_clearing_bar_m']} b={t2['new_ablation_seeds_clearing_bar_b']}\n"
          f"  pooled n=9 delta CI: ({t2['pooled_n9_paired_delta_ci']['ci_low']:.5f}, "
          f"{t2['pooled_n9_paired_delta_ci']['ci_high']:.5f}) "
          f"decision_grade={t2['pooled_reading_decision_grade']}\n"
          f"  tier(disclosed): {t2['strict_tier_reading_disclosed_only']}\n"
          f"  adjudication: {t2['adjudication_sec_1_42_A3']}\n"
          f"  horizon trigger seeds: {t2['horizon_read_trigger_seeds']}")
    for arm, row in verdict["k48_stress_locate_only_3arm_table"].items():
        if isinstance(row, dict) and "acc_A" in row:
            print(f"  K48 {arm}: acc_A={row['acc_A']:.4f}")
    return 0


def mode_selftest() -> int:
    import tempfile
    ok_all = True

    def rep(item, ok, detail=""):
        nonlocal ok_all
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {item}" + (f" -- {detail}" if detail else ""))

    # 1) manifest invariants + no collision with the sweep's own 27 names
    cells = diag_cells()
    from h2h_sweep_runner_rd import build_27_cell_manifest
    sweep_names = {c["name"] for c in build_27_cell_manifest()}
    rep("selftest 1: 12 cells, s3-8 x contender/ablation, zero overlap with sweep names",
        len(cells) == 12 and not (sweep_names & {c["name"] for c in cells})
        and all(c["seed"] == 2_000_000 + 10_000 * c["seed_idx"] for c in cells)
        and all(c["role"] == "sweep" and c["budget_frac"] == 1.0 and c["lr"] == 3e-4 for c in cells))

    # 2) K48 spec builder: role flip lands in the TRAINING config ONLY; drifted config refused
    fixture = {K48_CELL_ID: {"arch": "transformer", "task": "task1_calib", "kd": 0.75, "K": 48,
                             "role": "stress_locate_only", "budget_frac": 0.25, "seed": 500000,
                             "lr": 3e-4, "warmup_steps": 100, "seed_idx": 0,
                             "name": "h2h_calib_transformer_task1_calib_stress_locate_only_K48_auxrev2"}}
    spec = build_k48_spec(fixture)
    ok2 = (spec["cell_config"]["role"] == "sweep" and spec["role"] == "stress_locate_only"
           and spec["fresh"] is True and spec["K"] == 48
           and fixture[K48_CELL_ID]["role"] == "stress_locate_only")   # input never mutated
    drifted = {K48_CELL_ID: {**fixture[K48_CELL_ID], "budget_frac": 1.0}}
    try:
        build_k48_spec(drifted)
        ok2 = False
    except AssertionError:
        pass
    rep("selftest 2: K48 role flip training-config-only + drifted-config refusal (teeth)", ok2)

    # 3) analysis math on synthetic remetric dirs -- all A3 branches + tier + gate teeth
    def fake_dirs(cont_new, abl_new, td):
        old, new, r4 = (os.path.join(td, d) for d in ("old", "new", "r4"))
        for d in (old, new, r4):
            os.makedirs(d, exist_ok=True)
        sweep_c, sweep_a = [0.03955, 0.02905, 0.33447], [0.028, 0.031, 0.029]   # sec 1.40 shape
        for i in range(3):
            for arch, v in (("contender", sweep_c[i]), ("ablation", sweep_a[i])):
                with open(os.path.join(old, f"h2h_{arch}_task2_sweep_s{i}_round4.json"), "w") as f:
                    json.dump({"leg_a": {"acc_A": v}, "leg_b_ridge": {}}, f)
        for j, i in enumerate(TASK2DIAG_SEED_IDXS):
            for arch, v in (("contender", cont_new[j]), ("ablation", abl_new[j])):
                with open(os.path.join(new, f"h2h_{arch}_task2_sweep_s{i}_round4.json"), "w") as f:
                    json.dump({"leg_a": {"acc_A": v}, "leg_b_ridge": {}}, f)
        for arm, v in (("contender", 0.0189), ("ablation", 0.0195)):
            with open(os.path.join(r4, f"{arm}_task1_stress_K48_round4.json"), "w") as f:
                json.dump({"leg_a": {"acc_A": v}}, f)
        k48p = os.path.join(td, "k48.json")
        with open(k48p, "w") as f:
            json.dump({"leg_a": {"acc_A": 0.0210}}, f)
        return old, new, k48p, r4

    with tempfile.TemporaryDirectory() as td:
        v = analyze(*fake_dirs([0.31, 0.02, 0.03, 0.03, 0.02, 0.03],
                               [0.03, 0.03, 0.03, 0.03, 0.03, 0.03], td))
        t = v["task2"]
        ok3a = (t["new_contender_seeds_clearing_bar_m"] == 1
                and t["new_ablation_seeds_clearing_bar_b"] == 0
                and "TRAINABILITY/SEED-VARIANCE CONFIRMED at pooled rate 2/9"
                in t["adjudication_sec_1_42_A3"]
                and t["horizon_read_trigger_seeds"] == [3]
                and v["k48_stress_locate_only_3arm_table"]["transformer"]["acc_A"] == 0.0210
                and not v["k48_stress_locate_only_3arm_table"]["transformer"]["clears_locate_only_bar"])
    with tempfile.TemporaryDirectory() as td:
        v = analyze(*fake_dirs([0.02] * 6, [0.03] * 6, td))
        ok3b = ("NO REPLICATION" in v["task2"]["adjudication_sec_1_42_A3"]
                and v["task2"]["new_contender_seeds_clearing_bar_m"] == 0)
    with tempfile.TemporaryDirectory() as td:
        v = analyze(*fake_dirs([0.02] * 6, [0.03, 0.12, 0.03, 0.03, 0.03, 0.03], td))
        ok3c = "NOT-CONTENDER-SPECIFIC" in v["task2"]["adjudication_sec_1_42_A3"]
    rep("selftest 3: A3 adjudication branches (m>=1&b=0 incl. rate + trigger; m=0; b>=1)",
        ok3a and ok3b and ok3c)

    # 4) strict tier arithmetic (frozen sec 1.31.1) -- all five outcomes, run to completion
    mk = lambda lo, hi: {"ci_low": lo, "ci_high": hi, "mean": (lo + hi) / 2}
    ok4 = (strict_tier(mk(0.35, 0.9), True, False) == "WIN"
           and strict_tier(mk(-0.9, -0.35), False, True) == "LOSE"
           and strict_tier(mk(-0.1, 0.1), True, False) == "TIE"
           and strict_tier(mk(-0.1, 0.1), False, False) == "TIE (joint task-learning failure)"
           and strict_tier(mk(-0.32, 0.52), True, False).startswith("INDETERMINATE")
           and strict_tier(mk(0.35, 0.9), False, False) == "TIE (joint task-learning failure)")
    rep("selftest 4: frozen tier arithmetic (WIN/LOSE/TIE/joint-TIE/INDETERMINATE + no-demo guard)", ok4)

    # 5) batch-effect gate teeth surface in the verdict (variance-explosion cohort must flag)
    with tempfile.TemporaryDirectory() as td:
        v = analyze(*fake_dirs([0.9, 0.01, 0.9, 0.01, 0.9, 0.01], [0.03] * 6, td))
        ok5 = (not v["task2"]["pooled_reading_decision_grade"]
               and "NON-DECISION-GRADE" in v["task2"]["strict_tier_reading_disclosed_only"])
    rep("selftest 5: sec 16.19.5 gate flag propagates to NON-DECISION-GRADE marking (teeth)", ok5)

    print("SELFTEST:", "ALL PASS" if ok_all else "FAILURES PRESENT")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--run-cell", type=str)
    ap.add_argument("--run-k48", action="store_true")
    ap.add_argument("--remetric", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--steps-override", type=int, default=None,
                    help="K48 smoke ONLY (tiny fixture train); never set on the real run")
    ap.add_argument("--out", type=str)
    ap.add_argument("--out-dir", type=str, default="results/h2h_rung1/task2diag")
    ap.add_argument("--remetric-dir", type=str, default="results/h2h_rung1/task2diag/remetric")
    ap.add_argument("--sweep-remetric-dir", type=str, default="results/h2h_rung1/sweep_remetric")
    ap.add_argument("--round4-dir", type=str, default="results/h2h_rung1/round4")
    ap.add_argument("--ckpt-dir", type=str, default="/data/h2h_rung1_ckpts")
    ap.add_argument("--fresh-cell-configs", type=str,
                    default="results/h2h_rung1/round4/FRESH_CELL_CONFIGS.json")
    ap.add_argument("--gates-dir", type=str, default="results/h2h_rung1/gates")
    ap.add_argument("--margins-token", type=str, default="results/h2h_rung1/MARGINS_FROZEN.token")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.selftest:
        return mode_selftest()
    if args.list_cells:
        for c in diag_cells():
            print(c["name"])
        return 0
    if args.run_cell:
        return mode_run_cell(args)
    if args.run_k48:
        return mode_run_k48(args)
    if args.remetric:
        return mode_remetric(args)
    if args.analyze:
        return mode_analyze(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
