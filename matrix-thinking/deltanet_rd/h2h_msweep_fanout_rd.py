"""h2h_msweep_fanout_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.6 item F /
sec 1.7 gate 2's 90-pass axis-2 memory-multiplier sweep fan-out: 5 eligible
M-values ({2,4,8,16,32}, M=1 floor-excluded per R3-F3) x 3 horizons (H2/H4/
H8) x 2 tasks (Task 1 primary, Task 2 secondary) x 3 seeds = 90 SHORT,
forward-only inference passes over the ALREADY-TRAINED (b) Transformer
checkpoints (R3-F6-corrected count -- the design's own fixed cross-
reference).

Config-driven, resume-safe, tmux+supervisor-compatible (`H100_SETUP.md`'s
own perpetual-sweep conventions, reused here exactly as
`h2h_sweep_runner_rd.py` reuses them for the 27-cell training sweep) --
resume-safe by checking output VALIDITY, not existence; checkpoints held
RESIDENT across all passes for a given `(task, seed)` via
`h2h_calibration_wrappers_rd.ResidentCheckpointCache` (R3-F4's own build
requirement, re-used here, not reimplemented, for the REAL 90-pass fan-out
itself, not just its timing pilot).

**BUILD-STAGE SCOPE:** `eval_pass_fn`/`checkpoint_loader_fn` are
dependency-injected -- no concrete GPU inference call is imported or
invoked here; this module's own manifest/residency/resume/aggregate LOGIC
is fully unit-testable with synthetic fakes (this file's own smoke suite).

Run the smoke gate: python h2h_msweep_fanout_rd.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rd_episode_seed import rd_episode_seed
from h2h_calibration_wrappers_rd import ResidentCheckpointCache

ELIGIBLE_M_VALUES = (2, 4, 8, 16, 32)         # R3-F3: M=1 floor-excluded, never in this fan-out
HORIZONS = ("H2", "H4", "H8")
TASKS = ("task1_sweep", "task2_sweep")         # Task 1 primary, Task 2 secondary (sec 1.4.2)
N_SEEDS = 3
REQUIRED_RESULT_KEYS = ("M", "horizon", "task", "seed_idx", "recovered_frac")
EXPECTED_TOTAL_PASSES = 90                     # R3-F6-corrected: 5 x 3 x 2 x 3


def build_90_pass_manifest() -> list[dict]:
    """5 M-values x 3 horizons x 2 tasks x 3 seeds = 90 passes (R3-F6's
    own corrected count -- M=1's descriptive-only pass is priced
    separately in the eval-overhead line, never double-counted here)."""
    cells = []
    for task in TASKS:
        for seed_idx in range(N_SEEDS):
            seed = rd_episode_seed(task, seed_idx=seed_idx, ckpt_idx=0)
            for m in ELIGIBLE_M_VALUES:
                for horizon in HORIZONS:
                    cells.append({
                        "M": m, "horizon": horizon, "task": task, "seed_idx": seed_idx,
                        "seed": seed, "ckpt_key": (task, seed_idx),
                        "name": f"h2h_msweep_{task}_s{seed_idx}_M{m}_{horizon}",
                    })
    assert len(cells) == EXPECTED_TOTAL_PASSES, (
        f"expected {EXPECTED_TOTAL_PASSES} passes, got {len(cells)}")
    names = [c["name"] for c in cells]
    assert len(names) == len(set(names)), "cell name collision in the 90-pass manifest"
    return cells


def is_valid_result(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return all(k in doc for k in REQUIRED_RESULT_KEYS)


def run_pass_exception_isolated(cell: dict, eval_pass_fn, checkpoint_loader_fn,
                                 cache: ResidentCheckpointCache, out_dir: str) -> dict:
    """ONE inference pass, exception-isolated, checkpoint fetched via the
    RESIDENT cache (keyed by `(task, seed_idx)` -- loaded once, reused for
    every M/horizon combination at that key, per R3-F4's own requirement)."""
    out_path = os.path.join(out_dir, cell["name"] + ".json")
    if is_valid_result(out_path):
        return {"cell": cell["name"], "status": "skipped_already_valid"}
    try:
        ckpt = cache.get_or_load(cell["ckpt_key"], lambda: checkpoint_loader_fn(cell))
        result = eval_pass_fn(ckpt, cell["M"], cell["horizon"])
        result_doc = {**cell, **result}
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(result_doc, f)
        os.replace(tmp_path, out_path)
        return {"cell": cell["name"], "status": "completed"}
    except Exception as e:  # noqa: BLE001 -- deliberate: isolate ANY pass-level failure
        return {"cell": cell["name"], "status": "failed", "error": f"{type(e).__name__}: {e}"}


def run_msweep_fanout(manifest: list[dict], eval_pass_fn, checkpoint_loader_fn, out_dir: str) -> dict:
    """Groups passes by `ckpt_key` so ONE `ResidentCheckpointCache` per
    `(task, seed_idx)` genuinely holds that checkpoint resident across
    every M/horizon combination for it (never reloaded per pass)."""
    os.makedirs(out_dir, exist_ok=True)
    by_ckpt_key: dict = {}
    for cell in manifest:
        by_ckpt_key.setdefault(cell["ckpt_key"], []).append(cell)

    outcomes, load_counts = [], {}
    for ckpt_key, cells in by_ckpt_key.items():
        cache = ResidentCheckpointCache()
        for cell in cells:
            outcomes.append(run_pass_exception_isolated(cell, eval_pass_fn, checkpoint_loader_fn,
                                                          cache, out_dir))
        load_counts[ckpt_key] = cache.load_count.get(ckpt_key, 0)

    summary = {"completed": 0, "skipped_already_valid": 0, "failed": 0, "failures": []}
    for o in outcomes:
        summary[o["status"]] = summary.get(o["status"], 0) + 1
        if o["status"] == "failed":
            summary["failures"].append({"cell": o["cell"], "error": o.get("error", "?")})
    summary["n_total"] = len(manifest)
    summary["checkpoint_load_counts"] = load_counts
    return summary


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def smoke_1_manifest_shape():
    m = build_90_pass_manifest()
    m_values = sorted(set(c["M"] for c in m))
    horizons = sorted(set(c["horizon"] for c in m))
    ok = len(m) == 90 and m_values == sorted(ELIGIBLE_M_VALUES) and horizons == sorted(HORIZONS)
    _report("smoke 1: 90-pass manifest (5 M-values x 3 horizons x 2 tasks x 3 seeds)", ok,
            f"n={len(m)} m_values={m_values} horizons={horizons}")


def smoke_2_checkpoint_residency_across_full_fanout():
    """Across the FULL 90-pass fan-out, each of the 6 distinct (task,
    seed_idx) checkpoint keys (2 tasks x 3 seeds) must be loaded EXACTLY
    ONCE, even though it is reused across 5 M-values x 3 horizons = 15
    passes each (R3-F4's own residency requirement, exercised at REAL
    fan-out scale, not just the 2-pass timing pilot)."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_90_pass_manifest()
        load_calls = {"n": 0}

        def _fake_loader(cell):
            load_calls["n"] += 1
            return {"ckpt_for": cell["ckpt_key"]}

        def _fake_eval(ckpt, m, horizon):
            return {"recovered_frac": 0.5}

        summary = run_msweep_fanout(manifest, _fake_eval, _fake_loader, tmp)
        n_distinct_keys = len(set(c["ckpt_key"] for c in manifest))
        all_loaded_once = all(v == 1 for v in summary["checkpoint_load_counts"].values())
        ok = (summary["completed"] == 90 and load_calls["n"] == n_distinct_keys == 6
              and all_loaded_once)
        _report("smoke 2: EVERY checkpoint loaded exactly once across the full 90-pass fan-out "
                "(6 distinct (task,seed) keys, 15 passes each)", ok,
                f"n_distinct_keys={n_distinct_keys} load_calls={load_calls['n']} "
                f"load_counts={summary['checkpoint_load_counts']}")


def smoke_3_resume_skips_completed_passes():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_90_pass_manifest()[:15]   # one full (task,seed) group
        calls = {"n": 0}

        def _fake_loader(cell):
            calls["n"] += 1
            return {}

        def _fake_eval(ckpt, m, horizon):
            return {"recovered_frac": 0.7}

        run_msweep_fanout(manifest, _fake_eval, _fake_loader, tmp)
        first_pass_loads = calls["n"]
        run_msweep_fanout(manifest, _fake_eval, _fake_loader, tmp)   # re-run: all already valid
        second_pass_loads = calls["n"] - first_pass_loads

        ok = first_pass_loads == 1 and second_pass_loads == 0
        _report("smoke 3: re-running the fan-out with already-valid outputs re-loads NO "
                "checkpoints and re-runs NO passes (resume-safe)", ok,
                f"first_pass_loads={first_pass_loads} second_pass_loads={second_pass_loads}")


def smoke_4_exception_isolation_one_bad_pass():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_90_pass_manifest()[:15]   # one full (task,seed) group -- covers every M

        def _fake_loader(cell):
            return {}

        def _flaky_eval(ckpt, m, horizon):
            if m == 8:
                raise RuntimeError("simulated poisoned pass")
            return {"recovered_frac": 0.6}

        summary = run_msweep_fanout(manifest, _flaky_eval, _fake_loader, tmp)
        ok = summary["failed"] >= 1 and summary["completed"] == len(manifest) - summary["failed"]
        _report("smoke 4: one poisoned pass does not kill the rest of the fan-out", ok,
                f"summary={ {k: v for k, v in summary.items() if k != 'checkpoint_load_counts'} }")


def smoke_5_r3f6_pass_count_excludes_m1():
    """R3-F6's own fix: item F's pass count EXCLUDES M=1 (already priced
    separately in the eval-overhead line) -- confirmed directly, not
    merely asserted in a docstring."""
    m = build_90_pass_manifest()
    ok = 1 not in set(c["M"] for c in m) and len(m) == 90
    _report("smoke 5: M=1 is EXCLUDED from the 90-pass manifest (R3-F6's own double-count fix)", ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    print("=" * 70)
    print("h2h_msweep_fanout_rd.py -- sec 1.6 item F 90-pass M-sweep fan-out smoke suite")
    print("=" * 70)
    smoke_1_manifest_shape()
    smoke_2_checkpoint_residency_across_full_fanout()
    smoke_3_resume_skips_completed_passes()
    smoke_4_exception_isolation_one_bad_pass()
    smoke_5_r3f6_pass_count_excludes_m1()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
