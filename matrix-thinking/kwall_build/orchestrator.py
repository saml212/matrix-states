#!/usr/bin/env python3
"""K-wall wall-characterization ORCHESTRATOR — implements
`NCR_KWALL_CHARACTERIZATION_DESIGN.md` §4's ORCHESTRATOR CONTRACT
(STATUS: RELEASED, commit 1c99cc5, §A11-ADJUDICATION) end to end: cell
order, the per-cell-attempt dispatch/retry state machine, the HARD/RETRY
gates, write-ahead ledger persistence + crash recovery, G2's
canonical-path copy-on-accept contract, the trigger + band classification,
the conditional 160K arm, and `orchestrator_report.json` emission.

This is the ONE pool artifact this design produces (§6 POOL-ELIGIBILITY
STATEMENT) — the job-spec `cmd` invokes THIS script; it dispatches all up
to 16 cell-attempts itself, sequentially, on the ONE GPU it occupies.
`queue_worker.sh` never sees the 16 cells, only this one job.

Every section below cites its design-doc anchor in a comment so the build
audit can trace code back to spec line-for-line.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kwall_lib import constants as C          # noqa: E402
from kwall_lib import disk_io as dio          # noqa: E402
from kwall_lib import ledger as ledger_mod    # noqa: E402
from kwall_lib import classify as cl          # noqa: E402
from kwall_lib.reconstruction import derive_cell_state  # noqa: E402


# ---------------------------------------------------------------------------
# Startup smoke gate (design §4/"KW2.8/KW3.13/KW4.6 close-out" — F3):
# "At startup, BEFORE any gate check or dispatch, the orchestrator reads
# .../results_kwall_smoke/K{K}/earlyln_K{K}_s0.json for each K in {26,28,30}
# ... If any file is missing, unparseable, or fails the criterion, the
# orchestrator REFUSES TO DISPATCH ANYTHING and exits before touching the
# ledger."

def check_smoke(smoke_outdir: str, K_values) -> dict:
    smoke = {}
    for K in K_values:
        key = f"K{K}"
        path = os.path.join(smoke_outdir, f"K{K}", f"earlyln_K{K}_s0.json")
        ok = False
        try:
            with open(path) as f:
                rec = json.load(f)
            ok = (rec.get("status") in ("COMPLETED", "ABORTED-BUDGET")
                  and rec.get("K") == K and rec.get("d") == K + 1
                  and rec.get("d_override") == K + 1)
        except (OSError, json.JSONDecodeError, ValueError):
            ok = False
        smoke[key] = "PASS" if ok else "FAIL"
    return smoke


# ---------------------------------------------------------------------------
# Gate checks (design §4 "Gate check points, exact").

def hard_gate_ok(ledger: dict, ceiling: float) -> bool:
    return ledger["realized_gpu_h"] + ceiling <= C.HARD_GATE_CAP_GPUH


def retry_gate_ok(ledger: dict) -> bool:
    return ledger["realized_gpu_h"] < C.RETRY_GATE_THRESHOLD_GPUH


# ---------------------------------------------------------------------------
# Per-attempt dispatch (design §4, dispatch loop steps 1-2: write-ahead,
# classify (exit-code-exact, G3), copy-then-fold (G2, §R5 H2)).

def dispatch_attempt(state: "OrchestratorState", K: int, seed: int, arm: str,
                      attempt_n: int, outdir: str, steps: int, stop_file: str) -> str:
    ceiling = C.PRIMARY_CEILING_GPUH if arm == "primary" else C.CONDITIONAL_CEILING_GPUH
    ledger = state.ledger

    # ---- Gate check points, BEFORE dispatch (design §4).
    if not hard_gate_ok(ledger, ceiling):
        row = {"K": K, "seed": seed, "arm": arm, "attempt_n": attempt_n,
               "elapsed_h": 0.0, "status": "GATE-REFUSED", "outdir": None,
               "d_override": K + 1, "ceiling_charged": False}
        ledger_mod.append_row(ledger, row)
        ledger["open_attempt"] = None
        state.save()
        return "GATE-REFUSED"
    if attempt_n == 2 and not retry_gate_ok(ledger):
        row = {"K": K, "seed": seed, "arm": arm, "attempt_n": attempt_n,
               "elapsed_h": 0.0, "status": "GATE-REFUSED", "outdir": None,
               "d_override": K + 1, "ceiling_charged": False}
        ledger_mod.append_row(ledger, row)
        ledger["open_attempt"] = None
        state.save()
        return "GATE-REFUSED"

    # ---- Write-ahead (G1): persisted BEFORE subprocess.run, the ONLY
    # ledger write that ever happens before a subprocess runs.
    ledger["open_attempt"] = {"K": K, "seed": seed, "arm": arm,
                              "attempt_n": attempt_n, "charged_ceiling": ceiling,
                              "dispatch_ts": time.monotonic()}
    state.save()

    adir = dio.attempt_dir(outdir, K, seed, attempt_n)
    cmd = [state.python, state.cell_runner, "--cell",
           "--K", str(K), "--d-override", str(K + 1), "--seed", str(seed),
           "--steps", str(steps), "--ceiling-gpuh", str(ceiling),
           "--outdir", adir, "--stop-file", stop_file]
    t0 = time.time()
    proc = subprocess.run(cmd)
    attempt_elapsed_h = (time.time() - t0) / 3600.0
    exit_code = proc.returncode

    # ---- Classify (exit-code-exact, G3 — the design's own bullet list is
    # the single source of truth for the LIVE build).
    jpath = dio.attempt_json_path(outdir, K, seed, attempt_n)
    kind, rec = dio._read_json_or(jpath)
    on_disk_status = rec.get("status") if (kind == "parseable" and isinstance(rec, dict)) else None

    if exit_code == 3:
        classification = "STOPPED-BY-OPERATOR"
    elif on_disk_status == "COMPLETED":
        classification = "COMPLETED"
    elif on_disk_status == "ABORTED-BUDGET":
        classification = "ABORTED-BUDGET"
    else:
        classification = "CRASHED"   # deterministic crash OR exit-0-no-JSON default arm

    if classification == "STOPPED-BY-OPERATOR":
        row = {"K": K, "seed": seed, "arm": arm, "attempt_n": attempt_n,
               "elapsed_h": attempt_elapsed_h, "status": "STOPPED-BY-OPERATOR",
               "outdir": None, "d_override": K + 1, "ceiling_charged": False}
        ledger_mod.append_row(ledger, row)
        ledger["open_attempt"] = None
        state.save()
        return "STOPPED-BY-OPERATOR"

    if classification == "COMPLETED":
        # G2: copy-then-fold. Pre-copy exists-check ABORTS LOUDLY on a
        # genuine invariant violation (never overwrites).
        if dio.canonical_exists(outdir, K, seed):
            raise RuntimeError(
                f"G2 INVARIANT VIOLATION: canonical already exists for "
                f"K={K} seed={seed} in {outdir} — refusing to overwrite "
                f"(this must never happen in normal live dispatch)")
        dio.promote_to_canonical(outdir, K, seed, attempt_n)
        row = {"K": K, "seed": seed, "arm": arm, "attempt_n": attempt_n,
               "elapsed_h": attempt_elapsed_h, "status": "COMPLETED",
               "outdir": dio.canonical_path(outdir, K, seed),
               "d_override": K + 1, "ceiling_charged": False}
    else:
        row = {"K": K, "seed": seed, "arm": arm, "attempt_n": attempt_n,
               "elapsed_h": attempt_elapsed_h, "status": classification,
               "outdir": None, "d_override": K + 1, "ceiling_charged": False}
    ledger_mod.append_row(ledger, row)
    ledger["open_attempt"] = None
    state.save()
    return classification


def dispatch_cell(state: "OrchestratorState", K: int, seed: int, arm: str,
                   outdir: str, steps: int, stop_file: str) -> str:
    """Dispatch attempts for one cell until it derives TERMINAL or an
    operator stop is seen (design §4, "Per-cell-attempt dispatch loop")."""
    while True:
        existing = ledger_mod.cell_rows(state.ledger, K, seed, arm)
        cell_state = derive_cell_state(existing)
        if cell_state in ("COMPLETED", "PERSISTENTLY-ABORTED"):
            return cell_state
        next_n = max((r["attempt_n"] for r in existing), default=0) + 1
        if next_n > 2:
            return "PERSISTENTLY-ABORTED"  # defensive; should be unreachable
        result = dispatch_attempt(state, K, seed, arm, next_n, outdir, steps, stop_file)
        if result == "STOPPED-BY-OPERATOR":
            return "STOPPED-BY-OPERATOR"
        # loop: derive_cell_state re-evaluated against the freshly appended row


# ---------------------------------------------------------------------------
# Recovery procedure (design §4, steps 0/1/2/3/4).

class OrchestratorState:
    def __init__(self, args):
        self.args = args
        self.python = args.python
        self.cell_runner = args.cell_runner
        self.ledger_path = args.ledger_path
        self.ledger = None

    def save(self):
        ledger_mod.save_ledger(self.ledger_path, self.ledger)

    def recover(self):
        """Design §4 recovery steps 0-2."""
        loaded = ledger_mod.load_ledger(self.ledger_path)
        if loaded is None:
            conditional_K = self._infer_conditional_K()
            self.ledger = ledger_mod.reconstruct_ledger_from_disk(
                self.args.primary_outdir, self.args.conditional_outdir, conditional_K)
            self.save()
        else:
            self.ledger = loaded
        self._close_dangling_open_attempt()

    def _infer_conditional_K(self):
        """No ledger exists to read `trigger_result` from (this IS the
        from-scratch-reconstruction path) -- infer K_trig, if any
        conditional dispatch already happened, from the conditional
        archival tree's own directory names (`K{K}_s{seed}_attempt{n}/`)."""
        outdir = self.args.conditional_outdir
        if not os.path.isdir(outdir):
            return None
        for name in os.listdir(outdir):
            if name.startswith("K") and "_s" in name and "_attempt" in name:
                try:
                    return int(name[1:name.index("_s")])
                except ValueError:
                    continue
        return None

    def _close_dangling_open_attempt(self):
        """Design §4 recovery step 2: a non-null `open_attempt` means the
        orchestrator died mid-attempt."""
        oa = self.ledger.get("open_attempt")
        if oa is None:
            return
        K, seed, arm, attempt_n = oa["K"], oa["seed"], oa["arm"], oa["attempt_n"]
        outdir = (self.args.primary_outdir if arm == "primary"
                  else self.args.conditional_outdir)
        d_override = K + 1

        # 2.1 (§R6 I2): read the attempt's OWN JSON first.
        jpath = dio.attempt_json_path(outdir, K, seed, attempt_n)
        kind, rec = dio._read_json_or(jpath)
        if kind == "parseable" and isinstance(rec, dict) and rec.get("status") == "COMPLETED":
            if not (dio.canonical_exists(outdir, K, seed)
                    and dio.canonical_status(outdir, K, seed) == "COMPLETED"):
                dio.promote_to_canonical(outdir, K, seed, attempt_n)
            elapsed_h = rec.get("elapsed_s", 0.0) / 3600.0 + C.STARTUP_ALLOWANCE_S_GPUH
            row = {"K": K, "seed": seed, "arm": arm, "attempt_n": attempt_n,
                   "elapsed_h": elapsed_h, "status": "COMPLETED",
                   "outdir": dio.canonical_path(outdir, K, seed),
                   "d_override": d_override, "ceiling_charged": False}
        else:
            # 2.2: canonical file + dangling record -> proof of completion;
            # otherwise genuine crash.
            if dio.canonical_exists(outdir, K, seed) and dio.canonical_status(outdir, K, seed) == "COMPLETED":
                row = {"K": K, "seed": seed, "arm": arm, "attempt_n": attempt_n,
                       "elapsed_h": oa["charged_ceiling"], "status": "COMPLETED",
                       "outdir": dio.canonical_path(outdir, K, seed),
                       "d_override": d_override, "ceiling_charged": True}
            else:
                row = {"K": K, "seed": seed, "arm": arm, "attempt_n": attempt_n,
                       "elapsed_h": oa["charged_ceiling"], "status": "CRASHED-RECOVERED",
                       "outdir": None, "d_override": d_override, "ceiling_charged": True}

        # 2.3 (§R5 KW6.17): verify no live process still holds the GPU
        # before closing. Real check is nvidia-smi/CUDA-only; on a
        # dev/CI box without nvidia-smi this degrades to a disclosed
        # no-op (see `gpu_reap_check` below).
        gpu_reap_check(self.args.gpu_id)

        ledger_mod.append_row(self.ledger, row)
        self.ledger["open_attempt"] = None
        self.save()


def gpu_reap_check(gpu_id) -> None:
    """§R5 KW6.17: "verify recovery reaps/verifies the assigned GPU is free
    before closing a dangling open_attempt... simulate an orphaned CUDA
    process surviving a parent-only kill and confirm recovery aborts
    loudly rather than re-dispatching onto it." Real implementation is
    CUDA-box-only (`nvidia-smi --query-compute-apps`); this build ships
    the check and its ABORT-LOUDLY behavior, but degrades to a disclosed,
    printed no-op when `nvidia-smi` is unavailable (Mac/CI dev
    environment) rather than silently pretending the GPU is free."""
    if gpu_id is None:
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
             "-i", str(gpu_id)],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[gpu_reap_check] nvidia-smi unavailable in this environment -- "
              "GPU-liveness check is a DISCLOSED NO-OP here (CUDA-box-only "
              "check; see red-team item (xiii), deferred to the box).",
              file=sys.stderr)
        return
    pids = [line for line in out.stdout.splitlines() if line.strip()]
    if pids:
        raise RuntimeError(
            f"GPU {gpu_id} still holds a live compute process ({pids}) while "
            f"closing a dangling open_attempt -- ABORTING LOUDLY rather than "
            f"re-dispatching onto a GPU attempt 1 may still be running on "
            f"(§R5 KW6.17).")


# ---------------------------------------------------------------------------
# Trigger + band evaluation (design §4 F2/G5, §5).

def resolution_state(n_completed: int, n_converged: int):
    """Design §4's "Per-K resolution state" table."""
    if n_completed == 4:
        return n_converged  # EXACT
    if n_completed == 3:
        return ("AMBIGUOUS", n_converged)  # r_known among the 3 resolved
    return "UNRESOLVED"  # n_completed <= 2


def evaluate_trigger_and_band(state: "OrchestratorState"):
    """Returns (trig_info, band_info, resolution) where `trig_info` is an
    unambiguous dict (not the bare 4-tuple `classify.trigger` returns) --
    `diag`'s MEANING (`blocking_K` vs `band_blocked_K_trig`) depends on
    which of the two `TRIGGER-UNRESOLVED` return sites fired inside
    `trigger()` (design's own note: "a consumer must key off the RETURN
    SITE, never tuple position alone"); disambiguated here, once, using
    whether any per-K state is the raw `"UNRESOLVED"` sentinel (the ONLY
    way the raw K-scan itself can return early with `blocking_K`, as
    opposed to G5's post-hoc band-precondition override)."""
    from kwall_lib import harvest_bridge
    resolution = harvest_bridge.per_K_resolution(state.args.primary_outdir, (26, 28, 30))
    states = {K: resolution_state(*resolution[K]) for K in (26, 28, 30)}
    any_unresolved = any(s == "UNRESOLVED" for s in states.values())
    K_trig, res, res_detail, diag = cl.trigger(states[26], states[28], states[30])
    trig_info = {
        "K_trig": K_trig, "resolution": res, "resolution_detail": res_detail,
        "blocking_K": diag if (res == "TRIGGER-UNRESOLVED" and any_unresolved) else None,
        "band_blocked_K_trig": diag if (res == "TRIGGER-UNRESOLVED" and not any_unresolved) else None,
    }
    band_label, band_tag, band_incomplete, band_resolved = cl.classify_with_interval_logic(
        states[26], states[28], states[30])
    band_info = (band_label, band_tag, band_incomplete, band_resolved)
    return trig_info, band_info, resolution


# ---------------------------------------------------------------------------
# Report emission (design §4, "Output JSON" schema; G4 run_status).

def build_report(state: "OrchestratorState", run_status: str, smoke: dict,
                  trig_info, band, resolution, conditional_launched: bool,
                  qualifier_band, wall_clock_start: str, gpu_id, git_commit: str) -> dict:
    ledger = state.ledger
    att = ledger["attempts"]
    ccgh = sum(a["elapsed_h"] for a in att if a["ceiling_charged"])
    realized = ledger["realized_gpu_h"]
    frac = (ccgh / realized) if realized > 0 else 0.0
    label, tag, incomplete_marker, resolved_or_incomplete_Ks = band
    if incomplete_marker == "INCOMPLETE-AT-K":
        band_dict = {"label": "INCOMPLETE-AT-K", "non_monotone_tag": False,
                     "interval_resolved_Ks": [],
                     "incomplete_at_K": resolved_or_incomplete_Ks,
                     "candidate_bands": None}
    else:
        band_dict = {"label": label, "non_monotone_tag": bool(tag),
                     "interval_resolved_Ks": resolved_or_incomplete_Ks or [],
                     "incomplete_at_K": None, "candidate_bands": None}
    return {
        "run_status": run_status,
        "ledger": {
            "realized_gpu_h_final": realized,
            "hard_gate_cap": C.HARD_GATE_CAP_GPUH,
            "retry_gate_threshold": C.RETRY_GATE_THRESHOLD_GPUH,
            "declared_pool_ceiling": C.DECLARED_POOL_CEILING_GPUH,
            "open_attempt": ledger["open_attempt"],
            "attempts": att,
        },
        "charged_vs_measured": {
            "measured_gpu_h": realized - ccgh,
            "ceiling_charged_gpu_h": ccgh,
            "ceiling_charged_fraction": frac,
        },
        "stop_file_path": state.args.stop_file if run_status == "STOPPED-BY-OPERATOR" else None,
        "smoke": smoke,
        "primary": {"per_K": {str(K): {"n_completed": resolution[K][0],
                                        "n_converged": resolution[K][1]}
                              for K in (26, 28, 30)}},
        "trigger": {"resolution": trig_info["resolution"],
                   "resolution_detail": trig_info["resolution_detail"],
                   "K_trig": trig_info["K_trig"], "candidate_set": None,
                   "blocking_K": trig_info["blocking_K"],
                   "band_blocked_K_trig": trig_info["band_blocked_K_trig"]},
        "conditional": ({"launched": conditional_launched, "per_seed": [],
                         "qualifier_band": qualifier_band}
                        if (conditional_launched or qualifier_band is not None) else None),
        "band": band_dict,
        "wall_clock": {"start": wall_clock_start,
                      "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "gpu_id": gpu_id, "git_commit": git_commit,
        "n_cells_attempted": len({(a["K"], a["seed"], a["arm"]) for a in att}),
        "n_attempts_total": len(att),
    }


def determine_run_status(state: "OrchestratorState") -> str:
    """G4's exhaustive enum (design §4), encoded directly from the prose
    definitions. `dispatch_cell`'s own loop invariant (it never returns to
    its caller without appending >=1 terminal row for the cell, except on
    an operator stop, which short-circuits the whole run before this
    function is ever reached) guarantees every primary `(K,seed)` pair
    already has >=1 `ledger.attempts` row and the canonical/`COMPLETED`-row
    counts already agree (both are read from the SAME real disk/ledger
    state) by the time this runs — so, UNLIKE `validity_check` (which
    verifies an EXTERNALLY-claimed report against disk evidence), this
    producer does not need to re-derive those invariants defensively."""
    ledger = state.ledger
    att = ledger["attempts"]
    primary_first_gate_refused = any(
        a["arm"] == "primary" and a["attempt_n"] == 1 and a["status"] == "GATE-REFUSED"
        for a in att)
    n_primary_canonical = len([f for f in dio.list_canonical(state.args.primary_outdir)
                               if f["status"] == "COMPLETED"])

    # EXHAUSTED-BUDGET*: the hard gate refused a PRIMARY cell's OWN FIRST
    # attempt -- the 12-cell baseline itself could not complete.
    if (ledger["realized_gpu_h"] > 13.80 and n_primary_canonical < 12
            and primary_first_gate_refused):
        ccgh = sum(a["elapsed_h"] for a in att if a["ceiling_charged"])
        frac = ccgh / ledger["realized_gpu_h"] if ledger["realized_gpu_h"] > 0 else 0.0
        return ("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE" if frac > 0.50
                else "EXHAUSTED-BUDGET")

    # COMPLETE: "no budget-caused refusal of any dispatch -- first attempt
    # OR retry -- anywhere in the run" (primary OR conditional).
    no_budget_refusal_anywhere = not any(a["status"] == "GATE-REFUSED" for a in att)
    return "COMPLETE" if no_budget_refusal_anywhere else "COMPLETE-DEGRADED"


# ---------------------------------------------------------------------------
# Main.

def run(args) -> int:
    smoke = check_smoke(args.smoke_outdir, C.K_VALUES_PRIMARY)
    if not all(v == "PASS" for v in smoke.values()):
        print(f"STARTUP SMOKE GATE FAILED: {smoke} -- refusing to dispatch "
              f"anything, ledger untouched.", file=sys.stderr)
        return 3

    state = OrchestratorState(args)
    state.recover()
    wall_clock_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Primary sweep, K-major cell order (design §4, "Cell order").
    for K in C.K_VALUES_PRIMARY:
        for seed in C.SEEDS:
            result = dispatch_cell(state, K, seed, "primary",
                                   args.primary_outdir, args.steps_primary,
                                   args.stop_file)
            if result == "STOPPED-BY-OPERATOR":
                no_trig = {"K_trig": None, "resolution": "TRIGGER-UNRESOLVED",
                          "resolution_detail": None, "blocking_K": None,
                          "band_blocked_K_trig": None}
                report = build_report(state, "STOPPED-BY-OPERATOR", smoke,
                                      no_trig, (None, False, "INCOMPLETE-AT-K", []),
                                      {26: (0, 0), 28: (0, 0), 30: (0, 0)},
                                      False, None, wall_clock_start, args.gpu_id, args.git_commit)
                dio.atomic_write_json(args.report_path, report)
                print("STOPPED-BY-OPERATOR -- run terminated, no trigger evaluated.")
                return 0

    # All 12 primary cells terminal -- the ONLY point the trigger is
    # evaluated (design §4, "Trigger evaluation point + harvest invocations").
    trig_info, band, resolution = evaluate_trigger_and_band(state)
    K_trig = trig_info["K_trig"]
    conditional_launched = False
    qualifier_band = None

    if K_trig is not None and K_trig in C.K_VALUES_PRIMARY:
        conditional_launched = True
        for seed in C.SEEDS:
            result = dispatch_cell(state, K_trig, seed, "conditional",
                                   args.conditional_outdir, args.steps_conditional,
                                   args.stop_file_conditional)
            if result == "STOPPED-BY-OPERATOR":
                report = build_report(state, "STOPPED-BY-OPERATOR", smoke, trig_info, band,
                                      resolution, conditional_launched, None,
                                      wall_clock_start, args.gpu_id, args.git_commit)
                dio.atomic_write_json(args.report_path, report)
                print("STOPPED-BY-OPERATOR (conditional arm) -- run terminated.")
                return 0
        n_cond_completed = len([f for f in dio.list_canonical(args.conditional_outdir)
                                if f["status"] == "COMPLETED"])
        # §R8 K1: qualifier band reported ONLY on 4/4 conditional completion;
        # a throttled arm reports `qualifier_band=None` (data-only, disclosed
        # via `ledger.attempts`), never rounded into one of the 3 named bands.
        qualifier_band = (qualifier_band_for(args.conditional_outdir, K_trig)
                          if n_cond_completed == 4 else None)
    elif K_trig == 32:
        # The $0 archive branch (design §5): no new cells dispatched: the
        # K=32 disambiguation is already on record (§3's budget table).
        conditional_launched = False
        qualifier_band = "CONFIRMED-WALL-AT-160K"

    run_status = determine_run_status(state)
    report = build_report(state, run_status, smoke, trig_info, band, resolution,
                          conditional_launched, qualifier_band, wall_clock_start,
                          args.gpu_id, args.git_commit)
    dio.atomic_write_json(args.report_path, report)
    print(f"orchestrator finished: run_status={run_status} "
          f"realized_gpu_h={state.ledger['realized_gpu_h']:.4f}")
    return 0


def qualifier_band_for(conditional_outdir: str, K_trig: int) -> str:
    """§5's 3-band qualifier rule read from the conditional canonical
    directory's own harvest, at the matched 160K budget."""
    from kwall_lib import harvest_bridge
    resolution = harvest_bridge.per_K_resolution(conditional_outdir, (K_trig,))
    n_completed, n_converged = resolution[K_trig]
    if n_converged <= 1:
        return "CONFIRMED-WALL-AT-160K"
    if n_converged >= 3:
        return "SLOW-CONVERGENCE-AT-160K"
    return "PARTIAL-IMPROVEMENT-AT-160K"  # ==2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary-outdir", default=C.PRIMARY_OUTDIR)
    ap.add_argument("--conditional-outdir", default=C.CONDITIONAL_OUTDIR)
    ap.add_argument("--smoke-outdir", default=C.SMOKE_OUTDIR)
    ap.add_argument("--ledger-path", default=None,
                    help="default: <primary-outdir>/ORCHESTRATOR_LEDGER.json")
    ap.add_argument("--report-path", default=None,
                    help="default: <primary-outdir>/orchestrator_report.json")
    ap.add_argument("--stop-file", default=None,
                    help="default: <primary-outdir>/STOP")
    ap.add_argument("--stop-file-conditional", default=None,
                    help="default: <conditional-outdir>/STOP")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--cell-runner", default="/home/nvidia/ncr/ncr_earlyln_scale.py")
    ap.add_argument("--steps-primary", type=int, default=C.STEPS_PRIMARY)
    ap.add_argument("--steps-conditional", type=int, default=C.STEPS_CONDITIONAL)
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--git-commit", default="unknown")
    args = ap.parse_args(argv)
    if args.ledger_path is None:
        args.ledger_path = os.path.join(args.primary_outdir, "ORCHESTRATOR_LEDGER.json")
    if args.report_path is None:
        args.report_path = os.path.join(args.primary_outdir, "orchestrator_report.json")
    if args.stop_file is None:
        args.stop_file = os.path.join(args.primary_outdir, "STOP")
    if args.stop_file_conditional is None:
        args.stop_file_conditional = os.path.join(args.conditional_outdir, "STOP")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.primary_outdir, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
