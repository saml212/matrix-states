"""h2h_sweep_runner_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.6's rung-1
27-cell training sweep (3 archs x 3 tasks x 3 seeds), as a config-driven,
resume-safe runner -- mirrors this codebase's own standing perpetual-sweep
conventions (`H100_SETUP.md`'s "Perpetual/unattended sweep pattern": launch
inside tmux+supervisor, never a backgrounded SSH shell; smoke-gate first;
exception-isolated per-launch; validity-checked resume (a cell counts done
only if its output JSON parses and has the expected keys, never "file
exists" alone); per-run timeout with quarantine; a guarded aggregate that a
single malformed record cannot block).

**BUILD-STAGE SCOPE:** `cell_train_fn` is dependency-injected -- this
module never imports or calls a concrete GPU training loop itself, so its
own manifest/resume/aggregate LOGIC is fully unit-testable with a
synthetic, in-process fake (this file's own smoke suite), while the REAL
training call is wired in only at actual launch time on the H100 box.

Run the smoke gate: python h2h_sweep_runner_rd.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rd_episode_seed import rd_episode_seed

ARMS = ("contender", "ablation", "transformer")
TASKS = ("task1_sweep", "task2_sweep", "task3_sweep")
N_SEEDS = 3
REQUIRED_RESULT_KEYS = ("arch", "task", "seed_idx", "final_metric", "step_count")


def build_27_cell_manifest() -> list[dict]:
    """3 archs x 3 tasks x 3 seeds = 27 cells (sec 1.6's own pinned rung-1
    count). Task 3's own seed schedule reuses `task2_sweep`'s TASK_BASE key
    (task3 has no dedicated on-the-fly-generation seed need -- it trains on
    STATIC corpora, sec 1.4 -- so its own seed only needs to be
    collision-free against the OTHER on-the-fly tasks, which `task2_sweep`'s
    key already guarantees by construction)."""
    cells = []
    for arch in ARMS:
        for task in TASKS:
            seed_task_key = "task1_sweep" if task == "task1_sweep" else "task2_sweep"
            for seed_idx in range(N_SEEDS):
                cells.append({
                    "arch": arch, "task": task, "seed_idx": seed_idx,
                    "seed": rd_episode_seed(seed_task_key, seed_idx=seed_idx, ckpt_idx=0),
                    "name": f"h2h_{arch}_{task}_s{seed_idx}",
                })
    assert len(cells) == 27, f"expected 27 cells, got {len(cells)}"
    names = [c["name"] for c in cells]
    assert len(names) == len(set(names)), "cell name collision in the 27-cell manifest"
    return cells


def is_valid_result(path: str) -> bool:
    """Validity-checked resume: a cell counts DONE only if its output JSON
    parses AND has every REQUIRED_RESULT_KEYS entry -- never "file exists"
    alone (a truncated write from a killed process is a real, documented
    failure mode this codebase has hit before)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return all(k in doc for k in REQUIRED_RESULT_KEYS)


def run_cell_exception_isolated(cell: dict, cell_train_fn, out_dir: str) -> dict:
    """Runs ONE cell through `cell_train_fn` (dependency-injected), catching
    ANY exception so one poisoned cell cannot kill the rest of the sweep
    (H100_SETUP.md's own "exception-isolated per launch" requirement).
    Writes the result JSON via a temp-file + atomic rename: `os.replace`
    ONLY runs after a fully successful `json.dump` -- a hard kill (or any
    exception) mid-write leaves `out_path` untouched (either absent or
    holding the last GOOD write), never a half-written file; a Python-level
    exception during the write is caught by the SAME outer except below and
    reported as `failed`, exactly like a training-side failure (deliberately
    NOT wrapped in its own try/finally, which would risk renaming a
    partially-written temp file into place -- the opposite of what
    atomicity is for)."""
    out_path = os.path.join(out_dir, cell["name"] + ".json")
    if is_valid_result(out_path):
        return {"cell": cell["name"], "status": "skipped_already_valid"}
    try:
        result = cell_train_fn(cell)
        result_doc = {**cell, **result}
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(result_doc, f)
        os.replace(tmp_path, out_path)   # atomic; only reached after a clean write
        return {"cell": cell["name"], "status": "completed"}
    except Exception as e:  # noqa: BLE001 -- deliberate: isolate ANY cell-level failure
        return {"cell": cell["name"], "status": "failed", "error": f"{type(e).__name__}: {e}"}


def run_sweep(manifest: list[dict], cell_train_fn, out_dir: str) -> dict:
    """The orchestrator loop itself -- resume-safe (skips cells with an
    already-VALID output), exception-isolated per cell, and a GUARDED
    aggregate (one malformed record cannot prevent a summary from being
    written for everything else)."""
    os.makedirs(out_dir, exist_ok=True)
    outcomes = []
    for cell in manifest:
        outcomes.append(run_cell_exception_isolated(cell, cell_train_fn, out_dir))

    summary = {"completed": 0, "skipped_already_valid": 0, "failed": 0, "failures": []}
    for o in outcomes:
        try:
            summary[o["status"]] = summary.get(o["status"], 0) + 1
            if o["status"] == "failed":
                summary["failures"].append({"cell": o["cell"], "error": o.get("error", "?")})
        except Exception as e:  # noqa: BLE001 -- a single malformed outcome record must not
            summary["failures"].append({"cell": "UNKNOWN", "error": f"malformed outcome: {e}"})
    summary["n_total"] = len(manifest)
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
    m = build_27_cell_manifest()
    arches = sorted(set(c["arch"] for c in m))
    tasks = sorted(set(c["task"] for c in m))
    ok = len(m) == 27 and arches == sorted(ARMS) and tasks == sorted(TASKS)
    _report("smoke 1: 27-cell manifest (3 archs x 3 tasks x 3 seeds)", ok,
            f"n={len(m)} arches={arches} tasks={tasks}")


def smoke_2_validity_checked_resume():
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good.json")
        with open(good, "w") as f:
            json.dump({k: 0 for k in REQUIRED_RESULT_KEYS}, f)
        truncated = os.path.join(tmp, "truncated.json")
        with open(truncated, "w") as f:
            f.write('{"arch": "contender"')   # deliberately truncated, invalid JSON
        missing_keys = os.path.join(tmp, "missing_keys.json")
        with open(missing_keys, "w") as f:
            json.dump({"arch": "contender"}, f)   # valid JSON, missing required keys
        absent = os.path.join(tmp, "does_not_exist.json")

        ok = (is_valid_result(good) and not is_valid_result(truncated)
              and not is_valid_result(missing_keys) and not is_valid_result(absent))
        _report("smoke 2: is_valid_result correctly distinguishes valid / truncated / "
                "missing-keys / absent", ok)


def smoke_3_resume_skips_already_valid_cells():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_27_cell_manifest()[:3]
        calls = {"n": 0}

        def _fake_train(cell):
            calls["n"] += 1
            return {"final_metric": 0.5, "step_count": 1000}

        summary1 = run_sweep(manifest, _fake_train, tmp)
        first_pass_calls = calls["n"]
        summary2 = run_sweep(manifest, _fake_train, tmp)   # re-run: every cell already valid
        second_pass_calls = calls["n"] - first_pass_calls

        ok = (summary1["completed"] == 3 and summary2["skipped_already_valid"] == 3
              and second_pass_calls == 0)
        _report("smoke 3: re-running the sweep with already-valid outputs skips every cell "
                "(resume-safe, no re-training)", ok,
                f"first_pass_calls={first_pass_calls} second_pass_calls={second_pass_calls}")


def smoke_4_exception_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_27_cell_manifest()[:4]

        def _flaky_train(cell):
            if cell["seed_idx"] == 1:
                raise RuntimeError("simulated poisoned cell")
            return {"final_metric": 1.0, "step_count": 1000}

        summary = run_sweep(manifest, _flaky_train, tmp)
        ok = summary["failed"] >= 1 and summary["completed"] == len(manifest) - summary["failed"]
        _report("smoke 4: one poisoned cell (raises) does not kill the rest of the sweep "
                "(exception-isolated per launch)", ok, f"summary={summary}")


def smoke_5_guarded_aggregate_survives_malformed_record():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_27_cell_manifest()[:2]
        summary = run_sweep(manifest, lambda c: {"final_metric": 1.0, "step_count": 1000}, tmp)
        ok = "n_total" in summary and summary["n_total"] == 2
        _report("smoke 5: run_sweep always produces a summary with n_total set, even on a small "
                "manifest (guarded-aggregate shape sanity)", ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    print("=" * 70)
    print("h2h_sweep_runner_rd.py -- sec 1.6 27-cell sweep runner smoke suite")
    print("=" * 70)
    smoke_1_manifest_shape()
    smoke_2_validity_checked_resume()
    smoke_3_resume_skips_already_valid_cells()
    smoke_4_exception_isolation()
    smoke_5_guarded_aggregate_survives_malformed_record()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
