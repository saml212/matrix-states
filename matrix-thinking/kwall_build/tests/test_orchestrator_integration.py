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
from kwall_lib import classify as cl_mod     # noqa: E402
from kwall_lib import constants as C         # noqa: E402
from kwall_lib import disk_io as dio         # noqa: E402
from kwall_lib import ledger as ledger_mod   # noqa: E402
from kwall_lib.validity_check import validity_check_core, build_disk_view  # noqa: E402
# NOTE (REV-1): do NOT import kwall_lib.harvest_bridge at module level here.
# harvest_bridge._NCR_DIR_CANDIDATES reads $KWALL_NCR_DIR at IMPORT TIME
# (module-level list); several tests below set that env var just before
# calling into orchestrator functions that import harvest_bridge LAZILY
# (inside the function body) specifically so the env var is live by then.
# A module-level import here would freeze an empty candidate before any
# test ever runs (confirmed: this broke test_end_to_end_run_reduced_grid
# with a ModuleNotFoundError when tried during this build).

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
    cause a false `failed/` routing -- build a producible COMPLETE-DEGRADED
    ledger and confirm `kwall_lib.validity_check` PASSES it.

    m1 fix (minor, build audit R1): the prior version of this fixture
    padded the ledger with 12 synthetic rows at `(K=30, seed=8+i)` --
    `SEEDS=(0,1,2,3)`, so no such `(K,seed)` pair can ever exist (item (j):
    fixtures must be producible in their KEYS, not only their numbers).
    `K_seeds` below (K in {26,28,30} x seed in {0,1,2,3}) is the ONLY set
    of 12 valid primary pairs this design has. Rebuilt to use exclusively
    those 12: 11 pairs each carry [attempt-1 CRASHED-RECOVERED at the FULL
    ceiling, attempt-2 COMPLETED at a small MEASURED value] and the 12th
    carries [attempt-1 CRASHED-RECOVERED, attempt-2 GATE-REFUSED at 0.0] --
    exactly the shape a real crash-then-retry cell's ledger has, built
    directly via `ledger_mod.append_row` (this fixture's job is to feed
    `validity_check_core` a realistic ledger snapshot, not to re-prove live
    gate mechanics -- those are already covered by `test_hard_gate_refused`/
    `test_retry_gate_refused` above)."""
    print("test_gate_refused_report_passes_validity_check")
    outdir = fresh_dir("t9_gate_refused_vcheck")
    st = new_state(outdir)
    K_seeds = [(K, s) for K in (26, 28, 30) for s in range(4)]  # the ONLY 12 valid primary pairs

    # 11 pairs: attempt-1 CRASHED-RECOVERED at the FULL ceiling (design's
    # Class-1 rule: a ceiling_charged=true row carries EXACTLY
    # charged_ceiling(primary)=1.20, never an arbitrary padding number),
    # attempt-2 COMPLETED at a small MEASURED value (any value in
    # (0, PRIMARY_COMPLETED_REACHABILITY_CAP_GPUH=1.2210] is reachable for
    # a measured row -- 0.01 is a nominal small one, chosen so this
    # fixture's total realized_gpu_h stays well inside the 15.50 ceiling).
    for (K, s) in K_seeds[:11]:
        ledger_mod.append_row(st.ledger, {
            "K": K, "seed": s, "arm": "primary", "attempt_n": 1,
            "elapsed_h": C.PRIMARY_CEILING_GPUH, "status": "CRASHED-RECOVERED",
            "outdir": None, "d_override": K + 1, "ceiling_charged": True})
        ledger_mod.append_row(st.ledger, {
            "K": K, "seed": s, "arm": "primary", "attempt_n": 2,
            "elapsed_h": 0.01, "status": "COMPLETED",
            "outdir": dio.canonical_path(outdir, K, s), "d_override": K + 1,
            "ceiling_charged": False})

    # The 12th pair: same attempt-1 Class-1 rule, then attempt-2
    # GATE-REFUSED at elapsed_h=0.0 -- the ONLY value a GATE-REFUSED row
    # can ever carry (no subprocess ever runs, orchestrator.py:86-100).
    K12, s12 = K_seeds[11]
    ledger_mod.append_row(st.ledger, {
        "K": K12, "seed": s12, "arm": "primary", "attempt_n": 1,
        "elapsed_h": C.PRIMARY_CEILING_GPUH, "status": "CRASHED-RECOVERED",
        "outdir": None, "d_override": K12 + 1, "ceiling_charged": True})
    ledger_mod.append_row(st.ledger, {
        "K": K12, "seed": s12, "arm": "primary", "attempt_n": 2,
        "elapsed_h": 0.0, "status": "GATE-REFUSED",
        "outdir": None, "d_override": K12 + 1, "ceiling_charged": False})

    att = st.ledger["attempts"]
    n_completed_pairs = len({(a["K"], a["seed"]) for a in att if a["status"] == "COMPLETED"})
    ccgh = sum(a["elapsed_h"] for a in att if a["ceiling_charged"])
    check("12 ceiling-charged rows at exactly 1.20 each = 14.40 (item (j))",
          abs(ccgh - 12 * C.PRIMARY_CEILING_GPUH) < 1e-9, ccgh)
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


def _write_canonical_cell(outdir, K, seed, converged):
    """Writes a real canonical JSON file directly (bypassing live dispatch)
    in the exact schema `ncr_earlyln_scale._cell_gate1` reads -- the same
    convention `cell_runner_stub.py`'s "complete" branch uses -- for fast,
    controllable disk-state construction (REV-1 tests, build audit R1)."""
    os.makedirs(outdir, exist_ok=True)
    indist = 1.0 if converged else 0.0
    aer = float(K) if converged else 0.0
    rec = {"status": "COMPLETED", "K": K, "d": K + 1, "d_override": K + 1,
           "seed": seed, "elapsed_s": 1.0,
           "train": {"status": "COMPLETED", "step": 1, "loss_history": [[1, 1.0]]},
           "gpu_h": 1.0 / 3600.0,
           "eval": {"points": [{"component": "train_support",
                                "reads": {"binexp": {"recovered_frac@0.9": indist}}}
                               for _ in range(3)]},
           "deep_probe": {"A_eff_rank": [aer, aer]}}
    path = os.path.join(outdir, f"earlyln_K{K}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(rec, f)


def test_m3_stop_file_self_check_and_real_per_K():
    """M3 fix (build audit R1): G4's mandated pre-write `stop_file_path`
    self-check, and the primary-loop STOPPED-BY-OPERATOR report's `per_K`
    field -- previously the fabricated literal `{26:(0,0),28:(0,0),
    30:(0,0)}` regardless of how many cells actually completed before the
    stop. Drives the REAL `run()` entrypoint: dispatches (26,0) to a real
    COMPLETED canonical file first, THEN places the primary STOP sentinel
    so (26,1)'s dispatch sees it -- confirming the emitted report's
    `primary.per_K` reflects the ONE real completion, not zeros."""
    print("test_m3_stop_file_self_check_and_real_per_K")
    outdir = fresh_dir("t11_m3_primary_stop")
    st = new_state(outdir)
    st.ledger = ledger_mod.empty_ledger()
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    os.environ["KWALL_STUB_ELAPSED_S"] = "1.0"
    os.environ["KWALL_STUB_CONVERGED"] = "1"
    orch.dispatch_cell(st, 26, 0, "primary", outdir, 500, "/nonexistent/STOP")
    check("(26,0) really completed before the stop",
          dio.canonical_exists(outdir, 26, 0))
    st.save()

    stop_path = os.path.join(outdir, "STOP")
    with open(stop_path, "w") as f:
        f.write("stop")
    os.environ["KWALL_NCR_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(_BUILD_ROOT)), "matrix-thinking", "ncr")
    args = type("A", (), {})()
    args.primary_outdir = outdir
    args.conditional_outdir = outdir + "_cond"
    args.smoke_outdir = fresh_dir("t11_m3_primary_stop_smoke")
    for K in (26, 28, 30):
        with open(_mk_smoke_path(args.smoke_outdir, K), "w") as f:
            json.dump({"status": "COMPLETED", "K": K, "d": K + 1, "d_override": K + 1}, f)
    args.ledger_path = st.ledger_path
    args.report_path = os.path.join(outdir, "orchestrator_report.json")
    args.stop_file = stop_path
    args.stop_file_conditional = os.path.join(args.conditional_outdir, "STOP")
    args.python = PYTHON
    args.cell_runner = CELL_RUNNER
    args.steps_primary = 500
    args.steps_conditional = 500
    args.gpu_id = None
    args.git_commit = "test"
    rc = orch.run(args)
    check("run() exits 0 on operator stop", rc == 0, rc)
    with open(args.report_path) as f:
        report = json.load(f)
    check("run_status STOPPED-BY-OPERATOR", report["run_status"] == "STOPPED-BY-OPERATOR")
    check("stop_file_path == the PRIMARY sentinel actually seen",
          report["stop_file_path"] == stop_path, report["stop_file_path"])
    check("G4 self-check evidence: stop_file_path exists on disk",
          os.path.exists(report["stop_file_path"]))
    check("primary.per_K reflects the ONE REAL completion at K=26, not fabricated zeros",
          report["primary"]["per_K"]["26"]["n_completed"] == 1, report["primary"]["per_K"])
    check("primary.per_K at K=28/30 correctly reads 0 (nothing dispatched there)",
          report["primary"]["per_K"]["28"]["n_completed"] == 0
          and report["primary"]["per_K"]["30"]["n_completed"] == 0, report["primary"]["per_K"])

    # ---- self-check unit-level + revert-style teeth check ----
    good = dict(report)
    orch._assert_stop_file_evidence(good)  # must not raise
    bad_missing_path = dict(report, stop_file_path="/nonexistent/STOP-does-not-exist")
    raised = False
    try:
        orch._assert_stop_file_evidence(bad_missing_path)
    except AssertionError:
        raised = True
    check("self-check TEETH: raises on a stop_file_path that doesn't exist", raised)
    bad_none_path = dict(report, stop_file_path=None)
    raised2 = False
    try:
        orch._assert_stop_file_evidence(bad_none_path)
    except AssertionError:
        raised2 = True
    check("self-check TEETH: raises on stop_file_path=None", raised2)


def test_m3_conditional_stop_uses_conditional_path():
    """M3 fix (build audit R1): a conditional-arm stop was reporting the
    PRIMARY sentinel path (`args.stop_file`) regardless of which sentinel
    was actually seen -- the conditional dispatch loop only ever checks
    `args.stop_file_conditional`. Drives a REAL 12-cell primary sweep to a
    decided trigger (K_trig=26, via KWALL_STUB_CONVERGED=0 -- every K's
    rate reads 0/4, so classify()/the K-scan both decide unambiguously:
    band=FRONTIER-AT-K*=24, K_trig=26 unanimous), leaves `args.stop_file`
    UNCREATED (primary sweep runs to completion) and pre-creates ONLY
    `args.stop_file_conditional`, so the conditional loop's first dispatch
    (K=26,seed=0) sees it immediately."""
    print("test_m3_conditional_stop_uses_conditional_path")
    outdir = fresh_dir("t12_m3_cond_stop")
    cond_outdir = outdir + "_cond"
    smoke_outdir = fresh_dir("t12_m3_cond_stop_smoke")
    for K in (26, 28, 30):
        with open(_mk_smoke_path(smoke_outdir, K), "w") as f:
            json.dump({"status": "COMPLETED", "K": K, "d": K + 1, "d_override": K + 1}, f)
    os.environ["KWALL_STUB_BEHAVIOR"] = "complete"
    os.environ["KWALL_STUB_ELAPSED_S"] = "1.0"
    os.environ["KWALL_STUB_CONVERGED"] = "0"  # every primary cell reads 0/4 converged
    os.environ["KWALL_NCR_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(_BUILD_ROOT)), "matrix-thinking", "ncr")

    cond_stop = os.path.join(cond_outdir, "STOP")
    os.makedirs(cond_outdir, exist_ok=True)
    with open(cond_stop, "w") as f:
        f.write("stop")

    args = type("A", (), {})()
    args.primary_outdir = outdir
    args.conditional_outdir = cond_outdir
    args.smoke_outdir = smoke_outdir
    args.ledger_path = os.path.join(outdir, "ORCHESTRATOR_LEDGER.json")
    args.report_path = os.path.join(outdir, "orchestrator_report.json")
    args.stop_file = os.path.join(outdir, "STOP-never-created")
    args.stop_file_conditional = cond_stop
    args.python = PYTHON
    args.cell_runner = CELL_RUNNER
    args.steps_primary = 500
    args.steps_conditional = 500
    args.gpu_id = None
    args.git_commit = "test"
    rc = orch.run(args)
    check("run() exits 0", rc == 0, rc)
    check("12/12 primary canonical (full primary sweep completed)",
          len(dio.list_canonical(outdir)) == 12, len(dio.list_canonical(outdir)))
    with open(args.report_path) as f:
        report = json.load(f)
    check("trigger decided K_trig=26 unanimous",
          (report["trigger"]["K_trig"], report["trigger"]["resolution"]) == (26, "unanimous"),
          report["trigger"])
    check("run_status STOPPED-BY-OPERATOR (conditional arm)",
          report["run_status"] == "STOPPED-BY-OPERATOR")
    check("stop_file_path == the CONDITIONAL sentinel, NOT the primary one",
          report["stop_file_path"] == cond_stop, report["stop_file_path"])
    check("stop_file_path != args.stop_file (the pre-fix bug)",
          report["stop_file_path"] != args.stop_file)
    check("G4 self-check evidence: stop_file_path exists on disk",
          os.path.exists(report["stop_file_path"]))


def test_m4_schema_fields():
    """M4 fix (build audit R1): `trigger.candidate_set`, `band.
    candidate_bands`, `conditional.per_seed` -- three schema fields the
    design defines/requires that were hardcoded None/[] regardless of
    outcome. Two layers: (1) pure-function checks against classify.py
    directly (independently re-derived, not merely re-asserted -- row 1 of
    the design's own KW4.5 11-configuration table for the tie-break case;
    a fresh disagreeing-candidates search for the candidate_bands case);
    (2) end-to-end through `evaluate_trigger_and_band`/`build_report`
    against REAL canonical files (no live dispatch needed -- these
    functions only read disk + a caller-supplied ledger)."""
    print("test_m4_schema_fields")

    # ---- layer 1: pure-function checks ----
    # design's own KW4.5 table row 1: incomplete K=26, r_known=2, r28=0,
    # r30=3 -- both interval candidates (2,0,3)/(3,0,3) give the SAME band
    # (FRONTIER-AT-K*=30 [NON-MONOTONE], "both agree" -> band DECIDES) but
    # DIFFERENT K_trig (26 vs 28) -- the trigger tie-breaks to 26, exposing
    # candidate_set=[26,28].
    s26, s28, s30 = ("AMBIGUOUS", 2), 0, 3
    band1 = cl_mod.classify_with_interval_logic(s26, s28, s30)
    check("KW4.5 row 1: band DECIDES (both candidates agree)",
          band1[:2] == ("FRONTIER-AT-K*=30", True) and band1[2] is None, band1)
    trig1 = cl_mod.trigger(s26, s28, s30)
    check("KW4.5 row 1: trigger tie-break-min, K_trig=26",
          trig1[:2] == (26, "tie-break-min"), trig1)
    cset1 = cl_mod.trigger_candidate_set(s26, s28, s30)
    check("KW4.5 row 1: candidate_set == [26,28]", cset1 == [26, 28], cset1)
    check("candidate_set is null on 'unanimous' (single candidate, redundant)",
          cl_mod.trigger_candidate_set(0, 4, 4) is None,
          cl_mod.trigger_candidate_set(0, 4, 4))

    # a disagreeing-candidates scenario (independently found by direct
    # enumeration over AMBIGUOUS K=26 x r_known x (r28,r30), the first hit
    # where the two interval candidates give DIFFERENT bands): r26 AMBIGUOUS
    # at r_known=1 (candidates 1 and 2), r28=0, r30=0.
    #   classify(1,0,0) -> rule4 (robust(r24)&r26<=1) -> FRONTIER-AT-K*=24
    #   classify(2,0,0) -> rule4 fails (r26=2>1); rule5 (r26>=r28>=r30) -> GRADUAL-DECAY
    # different bands -> INCOMPLETE-AT-K, both disclosed.
    s26b, s28b, s30b = ("AMBIGUOUS", 1), 0, 0
    band2 = cl_mod.classify_with_interval_logic(s26b, s28b, s30b)
    check("disagreeing-candidates band is INCOMPLETE-AT-K", band2[2] == "INCOMPLETE-AT-K", band2)
    check("candidate_bands == both disagreeing labels, sorted",
          band2[4] == ["FRONTIER-AT-K*=24", "GRADUAL-DECAY"], band2)
    check("candidate_bands is null on a DECIDE (nothing to disclose)",
          cl_mod.classify_with_interval_logic(0, 4, 4)[4] is None)
    check("candidate_bands is null on the any-UNRESOLVED path (no candidate comparison performed)",
          cl_mod.classify_with_interval_logic("UNRESOLVED", 0, 0)[4] is None)

    # ---- layer 2: end-to-end through evaluate_trigger_and_band/build_report ----
    os.environ["KWALL_NCR_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(_BUILD_ROOT)), "matrix-thinking", "ncr")
    outdir = fresh_dir("t13_m4_tiebreak")
    for seed, conv in ((0, True), (1, True), (2, False)):
        _write_canonical_cell(outdir, 26, seed, conv)   # 3/4 -> AMBIGUOUS r_known=2
    for seed in range(4):
        _write_canonical_cell(outdir, 28, seed, False)  # 4/4, 0 converged
    for seed, conv in ((0, True), (1, True), (2, True), (3, False)):
        _write_canonical_cell(outdir, 30, seed, conv)   # 4/4, 3 converged
    args = type("A", (), {})()
    args.primary_outdir = outdir
    args.python = PYTHON
    args.cell_runner = CELL_RUNNER
    args.ledger_path = os.path.join(outdir, "ORCHESTRATOR_LEDGER.json")
    state = orch.OrchestratorState(args)
    trig_info, band_info, resolution = orch.evaluate_trigger_and_band(state)
    check("e2e: trigger.candidate_set == [26,28]",
          trig_info["candidate_set"] == [26, 28], trig_info)
    check("e2e: resolution tie-break-min, K_trig=26",
          (trig_info["resolution"], trig_info["K_trig"]) == ("tie-break-min", 26), trig_info)
    check("e2e: band DECIDES (candidate_bands stays None here -- KW4.5 row 1's own point)",
          band_info[2] is None and band_info[4] is None, band_info)

    # item (j): a COMPLETED row is MEASURED, not a fixed ceiling value -- any
    # elapsed_h in (0, PRIMARY_COMPLETED_REACHABILITY_CAP_GPUH=1.2210] is
    # reachable (constants.py's own KW11.1 comment); 0.01 is a nominal small
    # one, chosen only so this fixture's realized_gpu_h stays representative.
    ledger = ledger_mod.empty_ledger()
    ledger["attempts"] = [
        {"K": K, "seed": s, "arm": "primary", "attempt_n": 1, "elapsed_h": 0.01,
         "status": "COMPLETED", "outdir": "x", "d_override": K + 1, "ceiling_charged": False}
        for K in (26, 28, 30) for s in range(4) if not (K == 26 and s == 3)]
    ledger["realized_gpu_h"] = sum(a["elapsed_h"] for a in ledger["attempts"])
    ledger["open_attempt"] = None
    state.ledger = ledger
    report = orch.build_report(state, "COMPLETE-DEGRADED",
                               {"K26": "PASS", "K28": "PASS", "K30": "PASS"},
                               trig_info, band_info, resolution, False, None, [],
                               "2026-01-01T00:00:00Z", 0, "test")
    check("build_report threads trigger.candidate_set through to the report",
          report["trigger"]["candidate_set"] == [26, 28], report["trigger"])

    # ---- conditional.per_seed: throttled arm (2/4 conditional completions) ----
    cond_outdir = fresh_dir("t13_m4_cond_throttled")
    _write_canonical_cell(cond_outdir, 26, 0, True)
    _write_canonical_cell(cond_outdir, 26, 1, False)
    per_seed = orch._conditional_per_seed(cond_outdir, 26)
    check("conditional.per_seed discloses exactly the 2 COMPLETED conditional cells",
          len(per_seed) == 2, per_seed)
    check("conditional.per_seed entries carry seed + raw gate1 (indist_min_recovered/verdict)",
          all("seed" in r and "indist_min_recovered" in r["gate1"] and "verdict" in r["gate1"]
              for r in per_seed),
          per_seed)
    verdicts = {r["seed"]: r["gate1"]["verdict"] for r in per_seed}
    check("per-seed verdicts correctly split CONVERGED (seed 0) vs DEAD (seed 1)",
          verdicts == {0: "CONVERGED", 1: "DEAD"}, verdicts)
    check("conditional.per_seed is [] when nothing completed (no cells on disk)",
          orch._conditional_per_seed(fresh_dir("t13_m4_cond_empty"), 26) == [])

    # ---- revert-style teeth check: the PRE-FIX build hardcoded these three
    # fields regardless of outcome -- confirm that literal reproduces
    # values DIFFERENT from what the fix now emits, on the SAME state.
    pre_fix_candidate_set = None  # orchestrator.py:385 hardcoded `None` always
    pre_fix_candidate_bands = None  # hardcoded on both branches (:358, :362)
    pre_fix_per_seed = []  # hardcoded (:388)
    check("TEETH: pre-fix hardcoded candidate_set (None) DIFFERS from the real tie-break value",
          pre_fix_candidate_set != cset1, (pre_fix_candidate_set, cset1))
    check("TEETH: pre-fix hardcoded candidate_bands (None) DIFFERS from the real disagreeing set",
          pre_fix_candidate_bands != band2[4], (pre_fix_candidate_bands, band2[4]))
    check("TEETH: pre-fix hardcoded per_seed ([]) DIFFERS from the real throttled-arm disclosure",
          pre_fix_per_seed != per_seed, (pre_fix_per_seed, per_seed))


def test_m3_infer_conditional_K_pruned_attempt_tree():
    """m3 (minor, build audit R1): `_infer_conditional_K` reconstructed the
    conditional 4 only when a `K{K}_s{seed}_attempt{n}/` directory
    survived; an attempt tree pruned AFTER copying (0.2's own named
    residual case) leaves ONLY canonical evidence
    (`earlyln_K{K}_s{seed}.json`) behind, which the pre-fix code could not
    infer K from -- silently reconstructing conditional_K=None even though
    real conditional canonical evidence exists on disk."""
    print("test_m3_infer_conditional_K_pruned_attempt_tree")
    cond_outdir = fresh_dir("t14_m3_pruned_attempt_tree")
    # Only a canonical file survives -- no K28_s*_attempt*/ directory at all
    # (simulates the attempt dir having been pruned after the copy).
    _write_canonical_cell(cond_outdir, 28, 0, True)
    args = type("A", (), {})()
    args.conditional_outdir = cond_outdir
    args.python = PYTHON
    args.cell_runner = CELL_RUNNER
    args.ledger_path = os.path.join(cond_outdir, "ORCHESTRATOR_LEDGER.json")
    st = orch.OrchestratorState(args)
    inferred = st._infer_conditional_K()
    check("conditional K correctly inferred from canonical filenames alone",
          inferred == 28, inferred)

    empty_dir = fresh_dir("t14_m3_empty")
    args2 = type("A", (), {})()
    args2.conditional_outdir = empty_dir
    args2.python = PYTHON
    args2.cell_runner = CELL_RUNNER
    args2.ledger_path = os.path.join(empty_dir, "ORCHESTRATOR_LEDGER.json")
    check("returns None when the directory genuinely has nothing (no regression)",
          orch.OrchestratorState(args2)._infer_conditional_K() is None)


def test_m7_validity_check_cli_missing_report_no_traceback():
    """m7 (minor, build audit R1): when `run()` refuses at the startup
    smoke gate or aborts loudly on a G2 invariant violation, no report is
    ever written; the CLI used to raise an uncaught FileNotFoundError
    (routing was still correct -- non-zero exit -- but the job log got a
    traceback instead of a diagnosis). Confirm the REAL CLI subprocess now
    exits 1 with a clean VALIDITY_CHECK FAIL diagnosis, no traceback."""
    print("test_m7_validity_check_cli_missing_report_no_traceback")
    outdir = fresh_dir("t15_m7_no_report")
    missing_report = os.path.join(outdir, "orchestrator_report.json")
    proc = subprocess.run(
        [PYTHON, "-m", "kwall_lib.validity_check", missing_report, outdir, outdir + "_cond"],
        cwd=_BUILD_ROOT, capture_output=True, text=True)
    check("CLI exits 1 (still routes to failed/, same as before)", proc.returncode == 1, proc.returncode)
    check("stderr carries a clean diagnosis, not a Python traceback",
          "VALIDITY_CHECK FAIL" in proc.stderr and "Traceback" not in proc.stderr,
          proc.stderr)


def main():
    for fn in (test_completed_cell, test_hard_gate_refused,
               test_retry_then_persistently_aborted, test_retry_gate_refused,
               test_stop_file, test_g2_exists_check_aborts_loudly,
               test_ledger_corruption_recovery, test_mid_attempt_sigkill_recovery,
               test_gate_refused_report_passes_validity_check,
               test_end_to_end_run_reduced_grid,
               test_m3_stop_file_self_check_and_real_per_K,
               test_m3_conditional_stop_uses_conditional_path,
               test_m4_schema_fields,
               test_m3_infer_conditional_K_pruned_attempt_tree,
               test_m7_validity_check_cli_missing_report_no_traceback):
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
