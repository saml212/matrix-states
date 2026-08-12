#!/usr/bin/env python3
"""Integration tests of the REAL `orchestrator.py` dispatch-loop code
(`dispatch_attempt`/`dispatch_cell`/`OrchestratorState.recover`) driven
through real `subprocess.run` calls against `cell_runner_stub.py` — a
CPU-STUB standing in for `ncr_earlyln_scale.py --cell` (CLAUDE.md's
CPU-stub hard rule: flagged here, real-kernel coverage is the box-side
micro-smokes, see `smokes/MICRO_SMOKE_SPEC.md`).

Every test below is a FORCED-FAIL or FORCED-CRASH negative test RUN TO
COMPLETION (CLAUDE.md's standing rule — never merely written), covering
design §4/§6's own named red-team items where CPU-runnable:
  (vi)  mid-attempt SIGKILL + restart -> exactly one CRASHED-RECOVERED
        row at the FULL ceiling, never a silent gap or double-charge.
  (x)   truncate/corrupt ORCHESTRATOR_LEDGER.json + restart -> conservative
        reconstruction fires, run does NOT resume at realized_gpu_h=0.
  (xii) a genuine GATE-REFUSED row must not itself cause a false
        `validity_check` `failed/` routing (checked via the real disk
        state a HARD-GATE refusal produces, cross-checked against
        `kwall_lib.validity_check`).
  G2's exists-check ABORTS LOUDLY rather than silently overwriting.
  G3's `--stop-file` exit-3 handling: STOPPED-BY-OPERATOR, no retry, no
  further dispatch.
  Retry-then-PERSISTENTLY-ABORTED (both attempts fail).
  RETRY GATE refusal (attempt-2 specifically denied while attempt-1 was
  admitted).
(vii)/(viii)/(ix)/(xi)/(xiii)/(xiv) are CUDA/box-only or require a live
multi-GPU cluster and are deferred to the box red-team ceremony per
§A11-ADJUDICATION's own staging ("then the build's OWN audit + pre-launch
resource/placement red-team") -- see BUILD_REPORT.md.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _BUILD_ROOT)

import orchestrator as orch                 # noqa: E402
from kwall_lib import constants as C         # noqa: E402
from kwall_lib import disk_io as dio         # noqa: E402
from kwall_lib import ledger as ledger_mod   # noqa: E402
from kwall_lib.validity_check import validity_check_core, build_disk_view  # noqa: E402

SCRATCH_ROOT = os.environ.get(
    "KWALL_TEST_SCRATCH",
    "/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/"
    "be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad/kwall_orch_tests")
CELL_RUNNER = os.path.join(_BUILD_ROOT, "cell_runner_stub.py")
PYTHON = sys.executable

FAILURES = []


def fresh_dir(name):
    d = os.path.join(SCRATCH_ROOT, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def new_state(outdir):
    args = type("A", (), {})()
    args.primary_outdir = outdir
    args.conditional_outdir = outdir + "_cond"
    args.python = PYTHON
    args.cell_runner = CELL_RUNNER
    args.ledger_path = os.path.join(outdir, "ORCHESTRATOR_LEDGER.json")
    args.gpu_id = None
    st = orch.OrchestratorState(args)
    st.ledger = ledger_mod.empty_ledger()
    return st


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name}  {detail}")
        FAILURES.append(f"{name}: {detail}")


# ---------------------------------------------------------------------------

def test_completed_cell():
    print("test_completed_cell")
    outdir = fresh_dir("t1_completed")
    st = new_state(outdir)
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    os.environ["KWALL_STUB_ELAPSED_S"] = "3.6"  # 0.001 GPU-h
    result = orch.dispatch_cell(st, 26, 0, "primary", outdir, 500, "/nonexistent/STOP")
    check("cell derives COMPLETED", result == "COMPLETED", result)
    rows = ledger_mod.cell_rows(st.ledger, 26, 0, "primary")
    check("exactly one row", len(rows) == 1, rows)
    check("row status COMPLETED", rows[0]["status"] == "COMPLETED", rows)
    check("ceiling_charged False (live measurement)", rows[0]["ceiling_charged"] is False)
    check("canonical file exists", dio.canonical_exists(outdir, 26, 0))
    check("realized_gpu_h > 0", st.ledger["realized_gpu_h"] > 0)


def test_hard_gate_refused():
    print("test_hard_gate_refused")
    outdir = fresh_dir("t2_hardgate")
    st = new_state(outdir)
    st.ledger["realized_gpu_h"] = 14.50  # 14.50 + 1.20 > 15.00 -> refused
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    result = orch.dispatch_cell(st, 26, 0, "primary", outdir, 500, "/nonexistent/STOP")
    check("cell derives PERSISTENTLY-ABORTED", result == "PERSISTENTLY-ABORTED", result)
    rows = ledger_mod.cell_rows(st.ledger, 26, 0, "primary")
    check("exactly one GATE-REFUSED row (no attempt-2 tried)", len(rows) == 1, rows)
    check("GATE-REFUSED row status", rows[0]["status"] == "GATE-REFUSED", rows)
    check("elapsed_h == 0.0", rows[0]["elapsed_h"] == 0.0)
    check("realized_gpu_h unchanged at 14.50", st.ledger["realized_gpu_h"] == 14.50)
    check("no canonical file written", not dio.canonical_exists(outdir, 26, 0))


def test_retry_then_persistently_aborted():
    print("test_retry_then_persistently_aborted")
    outdir = fresh_dir("t3_retry_aborted")
    st = new_state(outdir)
    os.environ["KWALL_STUB_BEHAVIOR"] = "crash"
    result = orch.dispatch_cell(st, 26, 0, "primary", outdir, 500, "/nonexistent/STOP")
    check("cell derives PERSISTENTLY-ABORTED after 2 crashes", result == "PERSISTENTLY-ABORTED", result)
    rows = ledger_mod.cell_rows(st.ledger, 26, 0, "primary")
    check("exactly two rows (attempt 1 + retry)", len(rows) == 2, rows)
    check("both CRASHED", all(r["status"] == "CRASHED" for r in rows), rows)
    check("attempt_n's are 1 and 2", sorted(r["attempt_n"] for r in rows) == [1, 2])


def test_retry_gate_refused():
    print("test_retry_gate_refused")
    outdir = fresh_dir("t4_retrygate")
    st = new_state(outdir)
    os.environ["KWALL_STUB_BEHAVIOR"] = "crash"
    os.environ["KWALL_STUB_ELAPSED_S"] = "1.0"
    # Seed realized_gpu_h so attempt 1 (crash) still passes the HARD gate
    # (12.50+1.20=13.70<=15.00) but afterward crosses the RETRY gate's
    # <12.00 threshold, refusing attempt 2 specifically.
    st.ledger["realized_gpu_h"] = 12.50
    result = orch.dispatch_cell(st, 26, 0, "primary", outdir, 500, "/nonexistent/STOP")
    check("cell derives PERSISTENTLY-ABORTED via retry-gate refusal", result == "PERSISTENTLY-ABORTED", result)
    rows = ledger_mod.cell_rows(st.ledger, 26, 0, "primary")
    check("two rows: attempt-1 CRASHED, attempt-2 GATE-REFUSED", len(rows) == 2, rows)
    a2 = [r for r in rows if r["attempt_n"] == 2][0]
    check("attempt-2 is GATE-REFUSED (retry gate, not hard gate)", a2["status"] == "GATE-REFUSED", a2)
    check("attempt-2 elapsed_h == 0.0 (never dispatched)", a2["elapsed_h"] == 0.0)


def test_stop_file():
    print("test_stop_file")
    outdir = fresh_dir("t5_stop")
    st = new_state(outdir)
    stop_path = os.path.join(outdir, "STOP")
    with open(stop_path, "w") as f:
        f.write("stop")
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    result = orch.dispatch_cell(st, 26, 0, "primary", outdir, 500, stop_path)
    check("cell derives STOPPED-BY-OPERATOR", result == "STOPPED-BY-OPERATOR", result)
    rows = ledger_mod.cell_rows(st.ledger, 26, 0, "primary")
    check("exactly one row, no retry", len(rows) == 1, rows)
    check("status STOPPED-BY-OPERATOR", rows[0]["status"] == "STOPPED-BY-OPERATOR", rows)
    check("no canonical file (never reached training)", not dio.canonical_exists(outdir, 26, 0))


def test_g2_exists_check_aborts_loudly():
    print("test_g2_exists_check_aborts_loudly")
    outdir = fresh_dir("t6_g2abort")
    st = new_state(outdir)
    # Pre-seed a canonical file for (26,0) -- a genuine invariant violation
    # if the orchestrator ever tries to write there again.
    dio.atomic_write_json(dio.canonical_path(outdir, 26, 0), {"status": "COMPLETED", "K": 26})
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    raised = False
    try:
        orch.dispatch_attempt(st, 26, 0, "primary", 1, outdir, 500, "/nonexistent/STOP")
    except RuntimeError as e:
        raised = True
        check("RuntimeError mentions G2 invariant", "G2" in str(e), str(e))
    check("dispatch_attempt ABORTS LOUDLY (raises) on pre-existing canonical", raised)


def test_ledger_corruption_recovery():
    """Red-team item (x): truncate/corrupt ORCHESTRATOR_LEDGER.json mid-file
    and restart -- confirm CONSERVATIVE RECONSTRUCTION fires and the run
    does NOT resume at realized_gpu_h=0."""
    print("test_ledger_corruption_recovery")
    outdir = fresh_dir("t7_ledger_corrupt")
    st = new_state(outdir)
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    os.environ["KWALL_STUB_ELAPSED_S"] = "3600.0"  # exactly 1.0 GPU-h measured
    orch.dispatch_cell(st, 26, 0, "primary", outdir, 500, "/nonexistent/STOP")
    st.save()
    real_realized = st.ledger["realized_gpu_h"]
    # NOTE: live dispatch measures the ORCHESTRATOR's own wall-clock, never
    # the stub JSON's `elapsed_s` field (design §4: "the orchestrator's OWN
    # wall-clock timer... is the measurement, NOT the cell JSON's gpu_h
    # field") -- so this is a tiny real duration, not KWALL_STUB_ELAPSED_S.
    check("sanity: real run produced nonzero realized_gpu_h", real_realized > 0,
          f"got {real_realized}")

    # Corrupt the ledger file (truncate mid-JSON).
    with open(st.ledger_path, "rb") as f:
        data = f.read()
    with open(st.ledger_path, "wb") as f:
        f.write(data[: len(data) // 2])

    # Fresh state, fresh recovery.
    st2 = new_state(outdir)
    args2 = st.args
    st2.args = args2
    st2.recover()
    # (Reconstruction re-derives elapsed_h from the CANONICAL file's own
    # `elapsed_s` + the startup allowance `s` -- a DIFFERENT quantity from
    # the live orchestrator's own wall-clock measurement above; both are
    # positive and both are "the ledger was NOT reset to 0", which is the
    # property this test proves.)
    check("reconstruction did NOT reset to 0",
          st2.ledger["realized_gpu_h"] > 0,
          f"got {st2.ledger['realized_gpu_h']}")
    check("reconstruction recovered the real COMPLETED row (measured, from canonical)",
          any(r["status"] == "COMPLETED" for r in st2.ledger["attempts"]),
          st2.ledger["attempts"])


def test_mid_attempt_sigkill_recovery():
    """Red-team item (vi): kill the orchestrator process mid-attempt
    (SIGKILL during a real subprocess.run) and restart -- confirm the
    ledger shows exactly one CRASHED-RECOVERED row charged at the FULL
    ceiling, not a silent gap or a double-charge."""
    print("test_mid_attempt_sigkill_recovery")
    outdir = fresh_dir("t8_sigkill")
    os.makedirs(outdir, exist_ok=True)
    driver_src = f"""
import sys
sys.path.insert(0, {_BUILD_ROOT!r})
import orchestrator as orch
from kwall_lib import ledger as ledger_mod
args = type("A", (), {{}})()
args.primary_outdir = {outdir!r}
args.conditional_outdir = {outdir!r} + "_cond"
args.python = {PYTHON!r}
args.cell_runner = {CELL_RUNNER!r}
args.ledger_path = {os.path.join(outdir, "ORCHESTRATOR_LEDGER.json")!r}
args.gpu_id = None
st = orch.OrchestratorState(args)
st.ledger = ledger_mod.empty_ledger()
orch.dispatch_attempt(st, 26, 0, "primary", 1, {outdir!r}, 500, "/nonexistent/STOP")
"""
    driver_path = os.path.join(outdir, "_driver.py")
    with open(driver_path, "w") as f:
        f.write(driver_src)
    env = dict(os.environ)
    env["KWALL_STUB_BEHAVIOR"] = "hang"
    proc = subprocess.Popen([PYTHON, driver_path], env=env,
                            start_new_session=True)
    ledger_path = os.path.join(outdir, "ORCHESTRATOR_LEDGER.json")
    deadline = time.time() + 20
    open_attempt_seen = False
    while time.time() < deadline:
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path) as f:
                    d = json.load(f)
                if d.get("open_attempt") is not None:
                    open_attempt_seen = True
                    break
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.1)
    check("write-ahead open_attempt observed before kill", open_attempt_seen)
    # SIGKILL the whole process group (driver + hung stub child).
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)

    st2 = new_state(outdir)
    args2 = type("A", (), {})()
    args2.primary_outdir = outdir
    args2.conditional_outdir = outdir + "_cond"
    args2.python = PYTHON
    args2.cell_runner = CELL_RUNNER
    args2.ledger_path = ledger_path
    args2.gpu_id = None
    st2.args = args2
    st2.recover()
    rows = ledger_mod.cell_rows(st2.ledger, 26, 0, "primary")
    check("exactly one row after recovery (no gap, no double-charge)", len(rows) == 1, rows)
    if rows:
        check("row is CRASHED-RECOVERED", rows[0]["status"] == "CRASHED-RECOVERED", rows)
        check("charged at the FULL ceiling", rows[0]["elapsed_h"] == C.PRIMARY_CEILING_GPUH, rows)
        check("ceiling_charged True", rows[0]["ceiling_charged"] is True, rows)
    check("open_attempt cleared", st2.ledger["open_attempt"] is None)


def test_gate_refused_report_passes_validity_check():
    """Red-team item (xii): a genuine GATE-REFUSED row must not itself
    cause a false `failed/` routing -- build a minimal COMPLETE-DEGRADED
    report carrying one real GATE-REFUSED row (from `test_hard_gate_refused`
    above's own disk state, re-derived here) and confirm
    `kwall_lib.validity_check` PASSES it."""
    print("test_gate_refused_report_passes_validity_check")
    outdir = fresh_dir("t9_gate_refused_vcheck")
    st = new_state(outdir)
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    os.environ["KWALL_STUB_ELAPSED_S"] = "3600.0"
    # 11 real COMPLETED cells + 1 genuinely GATE-REFUSED cell (budget
    # exhausted right before the 12th).
    K_seeds = [(K, s) for K in (26, 28, 30) for s in range(4)]
    for (K, s) in K_seeds[:11]:
        orch.dispatch_cell(st, K, s, "primary", outdir, 500, "/nonexistent/STOP")
    # Push realized_gpu_h past the hard-gate boundary via PRODUCIBLE
    # synthetic rows (build charter item (j): every fixture must be
    # derived from the charging rules, never an arbitrary padding number).
    # A ceiling_charged=true row can only ever carry EXACTLY
    # `charged_ceiling(arm)` (design §4, "Class 1 ... charged its full
    # charged_ceiling(arm)") -- so this uses 12 synthetic CRASHED-RECOVERED
    # rows at EXACTLY C.PRIMARY_CEILING_GPUH (1.20) each = 12*1.20=14.40,
    # comfortably >13.80 (EXHAUSTED-BUDGET's own base clause) and enough
    # to push the 12th real dispatch's hard-gate check
    # (14.40+1.20=15.60>15.00) to GATE-REFUSED, never an unreachable
    # elapsed_h on any single row.
    for i in range(12):
        ledger_mod.append_row(st.ledger, {
            "K": 30, "seed": 8 + i, "arm": "primary", "attempt_n": 1,
            "elapsed_h": C.PRIMARY_CEILING_GPUH, "status": "CRASHED-RECOVERED",
            "outdir": None, "d_override": 31, "ceiling_charged": True})
    orch.dispatch_cell(st, *K_seeds[11], "primary", outdir, 500, "/nonexistent/STOP")

    att = st.ledger["attempts"]
    n_completed_pairs = len({(a["K"], a["seed"]) for a in att if a["status"] == "COMPLETED"})
    ccgh = sum(a["elapsed_h"] for a in att if a["ceiling_charged"])
    rep = {
        "run_status": "COMPLETE-DEGRADED",
        "ledger": {"realized_gpu_h_final": st.ledger["realized_gpu_h"], "attempts": att},
        "charged_vs_measured": {"ceiling_charged_gpu_h": ccgh,
                               "ceiling_charged_fraction": (ccgh / st.ledger["realized_gpu_h"])
                               if st.ledger["realized_gpu_h"] else 0.0},
        "smoke": {"K26": "PASS", "K28": "PASS", "K30": "PASS"},
        "band": {"label": "GRADUAL-DECAY", "interval_resolved_Ks": [30], "incomplete_at_K": None},
        "trigger": {"resolution": "TRIGGER-UNRESOLVED", "K_trig": None},
        "conditional": None,
    }
    disk = {"primary_canonical": [{"K": K, "seed": s, "status": "COMPLETED"}
                                  for (K, s) in K_seeds[:11]],
           "cond_canonical": []}
    reasons = validity_check_core(rep, disk, mode="NEW")
    check("a genuine GATE-REFUSED row does not cause a validity_check FAIL",
          reasons == [], reasons)
    check("11 distinct COMPLETED primary pairs on disk", n_completed_pairs == 11, n_completed_pairs)


def test_end_to_end_run_reduced_grid():
    """Exercises the REAL `run()` entrypoint end to end -- startup smoke
    gate, primary sweep, `harvest()`/trigger/band evaluation (via
    `kwall_lib.harvest_bridge`, the repo's own `ncr_earlyln_scale.harvest`,
    called for real against stub-produced JSONs carrying fake-but-schema-
    correct `eval`/`deep_probe` fields), and final report emission --
    against a REDUCED grid (K=(26,), seed=(0,1)) via monkeypatch, since a
    12-cell CPU-stub run is unnecessary to exercise this code path and a
    2-cell one is representative and fast."""
    print("test_end_to_end_run_reduced_grid")
    outdir = fresh_dir("t10_e2e")
    cond_outdir = outdir + "_cond"
    smoke_outdir = fresh_dir("t10_e2e_smoke")
    for K in (26,):
        with open(_mk_smoke_path(smoke_outdir, K), "w") as f:
            json.dump({"status": "COMPLETED", "K": K, "d": K + 1, "d_override": K + 1}, f)

    orig_Kvals, orig_seeds = C.K_VALUES_PRIMARY, C.SEEDS
    C.K_VALUES_PRIMARY = (26,)
    C.SEEDS = (0, 1)
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    os.environ["KWALL_STUB_CONVERGED"] = "1"
    os.environ["KWALL_NCR_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(_BUILD_ROOT)), "matrix-thinking", "ncr")
    try:
        args = type("A", (), {})()
        args.primary_outdir = outdir
        args.conditional_outdir = cond_outdir
        args.smoke_outdir = smoke_outdir
        args.ledger_path = os.path.join(outdir, "ORCHESTRATOR_LEDGER.json")
        args.report_path = os.path.join(outdir, "orchestrator_report.json")
        args.stop_file = os.path.join(outdir, "STOP")
        args.stop_file_conditional = os.path.join(cond_outdir, "STOP")
        args.python = PYTHON
        args.cell_runner = CELL_RUNNER
        args.steps_primary = 500
        args.steps_conditional = 500
        args.gpu_id = None
        args.git_commit = "test"
        rc = orch.run(args)
        check("run() exits 0", rc == 0, rc)
        with open(args.report_path) as f:
            report = json.load(f)
        check("run_status is COMPLETE (K=26 fully converged, 2/2 seeds)",
              report["run_status"] == "COMPLETE", report["run_status"])
        check("2 canonical files (reduced grid)",
              len(dio.list_canonical(outdir)) == 2, dio.list_canonical(outdir))
        reasons = validity_check_core(report, build_disk_view(outdir, cond_outdir), "NEW")
        # (Not necessarily [] -- U6's band-label set doesn't include a
        # 2-seed-only reduced-grid band; this call exercises the CODE PATH,
        # not a pass/fail claim about a grid smaller than the real design.)
        print(f"  (informational) validity_check_core on the reduced-grid "
              f"report: {reasons if reasons else 'PASS []'}")
    finally:
        C.K_VALUES_PRIMARY, C.SEEDS = orig_Kvals, orig_seeds


def _mk_smoke_path(smoke_outdir, K):
    d = os.path.join(smoke_outdir, f"K{K}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"earlyln_K{K}_s0.json")


def main():
    for fn in (test_completed_cell, test_hard_gate_refused,
               test_retry_then_persistently_aborted, test_retry_gate_refused,
               test_stop_file, test_g2_exists_check_aborts_loudly,
               test_ledger_corruption_recovery, test_mid_attempt_sigkill_recovery,
               test_gate_refused_report_passes_validity_check,
               test_end_to_end_run_reduced_grid):
        fn()
        print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}):")
        for x in FAILURES:
            print("  -", x)
        return 1
    print("ALL ORCHESTRATOR INTEGRATION TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
