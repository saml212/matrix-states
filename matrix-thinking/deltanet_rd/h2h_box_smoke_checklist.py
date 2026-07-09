"""h2h_box_smoke_checklist.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.20's
REGISTERED BOX-SMOKE CHECKLIST, as a machine-readable manifest the deploy
stage can consume (sec 1.20's own "binding" checklist language). Deliberately
dumb and declarative -- a flat list of dicts, no logic beyond structural
self-validation; this file does NOT run any of items (1)-(10) itself (every
one of them needs real fla/Triton on the H100 box, per each item's own
`source_finding`), it only records WHAT the deploy stage owes before
HEADTOHEAD_MATCH_GATE_SIGNOFF / launch can be considered box-verified.

Items (9)-(10) (NEW, sec 1.23 scoped build-fix, item 5) are the Rev 4
CE_answer continuation's own blank-out negative tests -- see
h2h_cell_train_rd.py's `mode_selftest` for the CPU-provable halves (both
items ARE exercised meaningfully on CPU, unlike items 2/3's stub-vacuity
limitation: neither test depends on S_T being nonzero, only on the
CONTINUATION CALL's own signature/determinism, which the CPU stub does not
disconnect) and the box-only real-kernel residual each one still owes.

Item (11) (NEW, sec 1.34 F3) BREAKS this file's own "never runs anything"
convention, deliberately: sec 1.34's build-fix audit demonstrated that
h2h_cell_train_rd.py's own M4 "forced-fail" selftest 21 is "pure arithmetic
decoupled from the code under test" -- restoring the EXACT sec 1.27 OOM
shape at the real call site (`transformer_native_tap`, the K=48 rung-2-fit
crash) left every suite green, because selftest 21 only recomputes a
byte-count formula, never actually calls the code path it is meant to guard.
Item 11 closes that hole by being genuinely executable: `item_11_bound_check`
is the CPU-provable comparison logic (proven with SYNTHETIC peak-allocation
numbers, smoke_8 below -- no CUDA needed, mirrors this file's own items
9/10 CPU-provable-half discipline), and `item_11_real_k48_rung2_fit` is the
box-only real-kernel half that actually runs
`fit_rung2_identity_classifier` at the real K=48 stress shape on CUDA and
feeds its MEASURED peak allocation into that same comparison -- never a
second, independently-drifting bound. Device-guarded (smoke_9): refuses to
ever claim PASS off a real CUDA device.

Each CHECKLIST entry:
  id            -- short, stable slug (never renumbered/reused across edits).
  description   -- one-line statement of the box-only check itself.
  blocking      -- True: must PASS before deploy-stage signoff; False: gated
                    on a separate, later authorization (currently only item 8,
                    the d=128 escalation diagnostic, sec 1.5's own "NOT
                    pre-authorized" rule).
  source_finding -- where this item's requirement comes from (a design-doc
                    section, an audit finding ID, or a CPU-side smoke file's
                    own disclosed box-only limitation), so a reader can trace
                    every entry back to its origin rather than trusting this
                    file's own prose.

Run: python h2h_box_smoke_checklist.py --list   (prints the checklist table
and runs the structural self-validation smoke; exit 0 = manifest well-formed)
Run item 11 for real (box-only, needs CUDA):
  python h2h_box_smoke_checklist.py --run-item-11 --gates-dir <gates_dir> --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# sec 1.20's own "REGISTERED BOX-SMOKE CHECKLIST (deploy stage, binding)" list, items (1)-(8),
# transcribed verbatim in content, restructured into machine-readable rows here. Item 3 is the
# fix stage's OWN addition (AUD-F1), inserted at its documented position, never renumbered past it.
CHECKLIST: tuple[dict, ...] = (
    {
        "id": "real_kernel_fwd_bwd_grad",
        "description": "Real fla/Triton forward+backward+grad smoke for DeltaNetLM (the "
                        "production kernel path the CPU stub in h2h_fla_stub_rd.py cannot "
                        "exercise at all).",
        "blocking": True,
        "source_finding": "sec 1.20 REGISTERED BOX-SMOKE CHECKLIST item 1",
    },
    {
        "id": "tap_changes_with_q_real_state",
        "description": "End-to-end tap-changes-with-q on a REAL nonzero S_T_last -- closes "
                        "probe_head_rd.py smoke_3's own disclosed box-only limitation (the CPU "
                        "stub's all-zero final_state makes state_summary_raw identically zero "
                        "regardless of q_last there).",
        "blocking": True,
        "source_finding": "sec 1.20 REGISTERED BOX-SMOKE CHECKLIST item 2; "
                           "probe_head_rd.py smoke_3 docstring",
    },
    {
        "id": "aux_only_grad_isolation_contender",
        "description": "[NEW, AUD-F1] Aux-loss-ONLY (CE-excluded) gradient isolation for the "
                        "contender arm on real kernels -- k_proj/v_proj/b_proj must show nonzero "
                        "grad_abs_sum (q_proj excluded, see probe_head_rd.py's own "
                        "_ISOLATION_CHECK_ATTR_PATH docstring for why). The CPU-side box-only "
                        "branch already lives in probe_head_rd.py's smoke_8 (gated on "
                        "`not _STUB_INSTALLED`) -- this box-smoke item requires zero new wiring, "
                        "only a real-fla environment.",
        "blocking": True,
        "source_finding": "AUD-F1 (build audit sec 1.20 substantive finding); "
                           "probe_head_rd.py smoke_8c box-only branch",
    },
    {
        "id": "state_bytes_dtype_roundtrip",
        "description": "State-bytes/dtype round-trip == 32,768 fp32 on the real kernel (M2's own "
                        "pin), not merely the CPU stub's zero-matrix shape check.",
        "blocking": True,
        "source_finding": "sec 1.20 REGISTERED BOX-SMOKE CHECKLIST item 4; "
                           "verify_match_gate.py pass1_measured_state_bytes (M2)",
    },
    {
        "id": "msweep_timing_pilot",
        "description": "R3-F4's own M-sweep timing pilot (2 M-values x 1 checkpoint x 1 horizon, "
                        "checkpoints-resident-across-passes) run for REAL before the 90-pass "
                        "fan-out is authorized -- replaces the design-time ~5s/pass assumption "
                        "with a measured figure.",
        "blocking": True,
        "source_finding": "R3-F4 (sec 1.18/1.19); "
                           "h2h_calibration_wrappers_rd.py run_msweep_timing_pilot",
    },
    {
        "id": "gate2_timing_pilots",
        "description": "sec 1.7 gate 2's per-arch x task timing pilots, run for real, projected "
                        "and gated against the remaining GPU-h headroom before the 27-cell sweep "
                        "launches.",
        "blocking": True,
        "source_finding": "sec 1.7 gate 2; "
                           "h2h_calibration_wrappers_rd.py project_and_gate_arch_task_pilot",
    },
    {
        "id": "gate1_calibration_run",
        "description": "sec 1.7 gate 1's full 14-cell, 3-arm calibration run, to completion, with "
                        "every measured band checked against its predicted range -- a failed band "
                        "is a hard abort of the 27-cell sweep, never a silent launch.",
        "blocking": True,
        "source_finding": "sec 1.7 gate 1; "
                           "h2h_calibration_wrappers_rd.py build_full_calibration_manifest "
                           "(14 cells) + check_calibration_band",
    },
    {
        "id": "continuation_blankout_inplace",
        "description": "[NEW, sec 1.23 build-fix item 5a] CE_answer continuation in-place "
                        "blank-out: corrupt bind_tokens IN PLACE (fresh torch.randint_like draw) "
                        "AFTER caching S_T, re-run model.forward(query_tokens, initial_states=S_T) "
                        "with the SAME cached S_T -- logits_query must be numerically IDENTICAL. "
                        "CPU-PROVABLE IN FULL (unlike items 2/3): the claim is purely structural "
                        "(the continuation call's own signature has no argument, and therefore no "
                        "path, back to bind_tokens), true identically whether S_T is the CPU "
                        "stub's zero constant or a real nonzero Triton-kernel state -- this box "
                        "item only re-confirms the SAME claim holds on the real kernel path, no "
                        "new plumbing.",
        "blocking": True,
        "source_finding": "sec 1.3.1.3 Rev 4 (blank-out, extended); sec 1.23 item 5a; "
                           "h2h_cell_train_rd.py mode_selftest continuation blank-out item",
    },
    {
        "id": "continuation_blankout_fresh_instance",
        "description": "[NEW, sec 1.23 build-fix item 5b, R5-F4] Fresh-model-instance blank-out "
                        "companion: load the SAME trained weights (state_dict) into a FRESH model "
                        "instance and re-run the continuation from the SAME cached S_T and query "
                        "tokens -- logits must be BIT-IDENTICAL to the original instance's own "
                        "run, closing the hidden-module-cache channel the in-place test above "
                        "cannot (any per-instance cache/buffer not captured by state_dict() would "
                        "show up here as a divergence). CPU-PROVABLE for the ablation and "
                        "transformer arms (real, differentiable recurrence/attention, no stub "
                        "disconnect) and for the contender's OWN stub-constant case (S_T=0 "
                        "regardless of instance, so CPU proves determinism-given-equal-inputs); "
                        "BOX-ONLY residual is specifically whether the REAL Triton kernel harbors "
                        "any instance-level nondeterminism (e.g. a compiled-kernel/memory-pool "
                        "cache) invisible to state_dict() -- CPU cannot exercise a real kernel at "
                        "all, mirrors probe_head_rd.py smoke_3/smoke_8's own box-only-registration "
                        "discipline for the contender arm specifically.",
        "blocking": True,
        "source_finding": "R5-F4 (attack round 5, sec 1.23); sec 1.3.1.3 Rev 4 (blank-out); "
                           "h2h_cell_train_rd.py mode_selftest fresh-instance blank-out item",
    },
    {
        "id": "d128_diagnostic_deferred",
        "description": "d=128 diagnostic -- ONLY if the escalation decision (sec 1.5, mechanical, "
                        "NOT pre-authorized) actually authorizes the escalation rung; this item is "
                        "deliberately NOT part of the base deploy-stage gate.",
        "blocking": False,
        "source_finding": "sec 1.20 REGISTERED BOX-SMOKE CHECKLIST item 8; "
                           "sec 1.5 escalation decision ('NOT pre-authorized')",
    },
    {
        "id": "real_kernel_k48_rung2_fit_peak_alloc",
        "description": "[NEW, sec 1.34 F3] Real-kernel fit_rung2_identity_classifier at the REAL "
                        "K=48 stress shape (arch=transformer, task=task1_calib, K=48 -- the exact "
                        "transformer_task1_calib_stress_K48 cell sec 1.27's OOM occurred on) run "
                        "on CUDA -- asserts the MEASURED peak allocation < the 2 GiB bound "
                        "h2h_cell_train_rd.py selftest 21 only checks analytically. Closes the "
                        "sec 1.34-demonstrated coverage hole (selftest 21 is 'pure arithmetic "
                        "decoupled from the code under test' -- restoring the exact OOM shape at "
                        "the real call site left every suite green). Run via "
                        "`item_11_real_k48_rung2_fit` / `--run-item-11`, THIS file's one "
                        "deliberate exception to its own 'never runs anything' convention.",
        "blocking": True,
        "source_finding": "sec 1.34 F3; sec 1.27 (the OOM site, transformer_native_tap); "
                           "h2h_cell_train_rd.py selftest 21's own disclosed "
                           "'pure arithmetic decoupled from the code under test' limitation",
    },
)

REQUIRED_KEYS = ("id", "description", "blocking", "source_finding")

# ---------------------------------------------------------------------------
# Item 11 (sec 1.34 F3) -- the ONE executable entry in this otherwise-declarative file.
# ---------------------------------------------------------------------------

GIB = 1024 ** 3
ITEM_11_BOUND_BYTES = 2 * GIB   # mirrors h2h_cell_train_rd.py selftest 21's own PINNED_BOUND_BYTES
ITEM_11_TOKEN_NAME = "BOX_SMOKE_ITEM_11_K48_REAL_KERNEL_PASSED.token"


def item_11_bound_check(peak_bytes: int, bound_bytes: int = ITEM_11_BOUND_BYTES) -> dict:
    """The pure peak-allocation-vs-bound comparison `item_11_real_k48_rung2_fit` (below) applies
    to its MEASURED number -- factored out so the comparison's own teeth are provable on CPU
    with a SYNTHETIC peak-allocation number (smoke_8, below), closing the exact coverage hole
    sec 1.34 F3 demonstrated in h2h_cell_train_rd.py's own selftest 21 ('pure arithmetic
    decoupled from the code under test'): that selftest recomputes a byte-count formula and
    never calls the real code path; THIS function is the one true bound both the CPU-provable
    synthetic smoke and the box-only real-kernel run share -- never two independently-drifting
    copies of '< 2 GiB'."""
    return {"peak_bytes": peak_bytes, "bound_bytes": bound_bytes, "passed": peak_bytes < bound_bytes}


def item_11_real_k48_rung2_fit(gates_dir: str | None, device: str) -> dict:
    """sec 1.34 F3: actually RUNS `fit_rung2_identity_classifier` at the REAL K=48 stress shape
    (arch='transformer', task='task1_calib', K=48 -- the exact `transformer_task1_calib_stress_
    K48` cell sec 1.27's OOM occurred on: `probe_head_rd.py:173 transformer_native_tap` ->
    `transformer_baseline_rd.py:218`, an unsliced full-vocab LM head over the K=48 episode set)
    on a REAL CUDA device, and checks the REAL measured peak allocation against
    `item_11_bound_check`'s bound -- never the analytic-only estimate selftest 21 uses.

    Device-guarded: refuses to claim PASS on CPU or when CUDA is unavailable -- returns
    `passed=None` ('box_only_skipped'), NEVER True, so a CPU 'run' of this function can never be
    mistaken for verification (mirrors this file's own items 2/3/9/10 registrations' real-kernel-
    vs-CPU-stub discipline).

    On PASS, writes `gates_dir/BOX_SMOKE_ITEM_11_K48_REAL_KERNEL_PASSED.token` -- mirrors the
    existing GATE6/GATE7/BOX_SMOKE_ITEMS_1_4 token discipline (h2h_rung1_chain.sh Stage 0's own
    `for tok in ...` pre-flight); h2h_rung1_chain.sh's Stage-B3 pre-flight refuses to launch
    round 4 without this exact token present (sec 1.34 F3(b))."""
    import torch   # deferred: this file's structural smoke (smoke_1-7) must stay torch-free

    if device != "cuda" or not torch.cuda.is_available():
        return {"passed": None, "status": "box_only_skipped",
                "detail": f"item 11 needs a real CUDA kernel (device={device!r}, "
                          f"cuda_available={torch.cuda.is_available()}) -- cannot verify here, "
                          f"never claims PASS off-box"}

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from h2h_cell_train_rd import build_arm_model, fit_rung2_identity_classifier, get_pools, task_cfg

    device_obj = "cuda"
    pools = get_pools(device_obj)
    K48 = 48
    cfg_eval = task_cfg("task1_calib", K48, n_query=None)
    model = build_arm_model("transformer", pools.vocab_size_total, seed=0, device=device_obj)
    torch.cuda.reset_peak_memory_stats(device_obj)
    fit_rung2_identity_classifier("transformer", model, cfg_eval, tuple(cfg_eval.H_train), pools,
                                  device_obj, K48, seed=0)
    peak_bytes = torch.cuda.max_memory_allocated(device_obj)
    result = {"status": "ran_on_cuda", **item_11_bound_check(peak_bytes)}
    if result["passed"] and gates_dir:
        os.makedirs(gates_dir, exist_ok=True)
        token_path = os.path.join(gates_dir, ITEM_11_TOKEN_NAME)
        with open(token_path, "w") as f:
            json.dump(result, f)
        result["token_path"] = token_path
    return result


# ---------------------------------------------------------------------------
# Smoke gate -- structural self-validation only (this file runs NO box-only
# check itself; see module docstring).
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def smoke_1_checklist_shape():
    ok = len(CHECKLIST) == 11
    _report("smoke 1: checklist has exactly 11 items (sec 1.20's own registered 8 + sec 1.23 "
            "build-fix item 5's new items 9-10 + sec 1.34 F3's new item 11)", ok,
            f"n_items={len(CHECKLIST)}")


def smoke_2_every_entry_has_required_keys():
    missing = [row["id"] if "id" in row else f"<unnamed row {i}>"
               for i, row in enumerate(CHECKLIST)
               for key in REQUIRED_KEYS if key not in row]
    ok = not missing
    _report("smoke 2: every entry has all 4 required keys (id/description/blocking/source_finding)",
            ok, f"rows_missing_a_key={missing}")


def smoke_3_ids_unique_and_slug_shaped():
    ids = [row["id"] for row in CHECKLIST]
    unique = len(set(ids)) == len(ids)
    slug_shaped = all(isinstance(i, str) and i and " " not in i for i in ids)
    ok = unique and slug_shaped
    _report("smoke 3: every id is unique and slug-shaped (no spaces)", ok,
            f"unique={unique} slug_shaped={slug_shaped} ids={ids}")


def smoke_4_blocking_is_bool_and_matches_sec_1_20():
    all_bool = all(isinstance(row["blocking"], bool) for row in CHECKLIST)
    # sec 1.20's own text: items 1-7 are the base deploy-stage gate; item 8 (d=128) is explicitly
    # "deferred ... only if escalation authorized" -- the ONE non-blocking entry.
    non_blocking_ids = [row["id"] for row in CHECKLIST if not row["blocking"]]
    ok = all_bool and non_blocking_ids == ["d128_diagnostic_deferred"]
    _report("smoke 4: blocking is bool everywhere; exactly one non-blocking item "
            "(d128_diagnostic_deferred, sec 1.5's own 'NOT pre-authorized' rule)", ok,
            f"all_bool={all_bool} non_blocking_ids={non_blocking_ids}")


def smoke_5_aud_f1_item_present_and_positioned():
    """The fix stage's own addition (AUD-F1, checklist item 3) must be present, tagged, and
    sit at the documented position (sec 1.20's own numbered list: item 3 of 10, unmoved by
    sec 1.23's items 9-10, which are appended at the END, never renumbered past it)."""
    ids = [row["id"] for row in CHECKLIST]
    idx = ids.index("aux_only_grad_isolation_contender") if "aux_only_grad_isolation_contender" in ids else -1
    tagged = idx >= 0 and "[NEW, AUD-F1]" in CHECKLIST[idx]["description"]
    positioned = idx == 2   # 0-indexed -> 3rd of 10
    ok = tagged and positioned
    _report("smoke 5: AUD-F1's new item is present, tagged '[NEW, AUD-F1]', at position 3 of 10",
            ok, f"idx0based={idx} tagged={tagged}")


def smoke_6_descriptions_nonempty():
    ok = all(isinstance(row["description"], str) and len(row["description"]) > 10
             for row in CHECKLIST)
    _report("smoke 6: every description is a real, non-trivial string", ok)


def smoke_7_sec_1_23_items_present_and_positioned():
    """sec 1.23 build-fix item 5's new items (9)-(10) must be present, tagged '[NEW, sec 1.23',
    blocking, and sit immediately before the pre-existing d128 deferred item (never renumbered
    past it -- this file's own house convention, mirrored from smoke_5's AUD-F1 check)."""
    ids = [row["id"] for row in CHECKLIST]
    idx9 = ids.index("continuation_blankout_inplace") if "continuation_blankout_inplace" in ids else -1
    idx10 = ids.index("continuation_blankout_fresh_instance") if "continuation_blankout_fresh_instance" in ids else -1
    tagged = (idx9 >= 0 and "[NEW, sec 1.23" in CHECKLIST[idx9]["description"]
              and idx10 >= 0 and "[NEW, sec 1.23" in CHECKLIST[idx10]["description"])
    positioned = idx9 == 7 and idx10 == 8 and ids[9] == "d128_diagnostic_deferred"
    both_blocking = CHECKLIST[idx9]["blocking"] is True and CHECKLIST[idx10]["blocking"] is True
    ok = tagged and positioned and both_blocking
    _report("smoke 7: sec 1.23 items 9-10 (continuation blank-out, in-place + fresh-instance) "
            "present, tagged, positioned immediately before the d128 deferred item, both blocking",
            ok, f"idx9={idx9} idx10={idx10} tagged={tagged} both_blocking={both_blocking}")


def smoke_8_item11_present_and_positioned():
    """[NEW, sec 1.34 F3] item 11 must be present, tagged, positioned as the 11th (last) entry,
    and blocking -- mirrors smoke_5/smoke_7's own positional-check convention for prior
    additions."""
    ids = [row["id"] for row in CHECKLIST]
    idx = ids.index("real_kernel_k48_rung2_fit_peak_alloc") if "real_kernel_k48_rung2_fit_peak_alloc" in ids else -1
    tagged = idx >= 0 and "[NEW, sec 1.34 F3]" in CHECKLIST[idx]["description"]
    positioned = idx == 10   # 0-indexed -> 11th of 11, appended after d128_diagnostic_deferred
    blocking = idx >= 0 and CHECKLIST[idx]["blocking"] is True
    ok = tagged and positioned and blocking
    _report("smoke 8 (sec 1.34 F3): item 11 (real_kernel_k48_rung2_fit_peak_alloc) present, "
            "tagged '[NEW, sec 1.34 F3]', positioned last (11th of 11), blocking", ok,
            f"idx0based={idx} tagged={tagged} positioned={positioned} blocking={blocking}")


def smoke_9_item11_bound_logic_has_teeth():
    """[NEW, sec 1.34 F3] `item_11_bound_check`'s own comparison logic, proven on SYNTHETIC
    peak-allocation numbers (no CUDA needed) -- the exact coverage hole sec 1.34 demonstrated in
    h2h_cell_train_rd.py's selftest 21 (an analytic-only bound with no path back to a real
    measured allocation) cannot recur here: an in-bound synthetic number passes, an over-bound
    one (mirroring sec 1.27's own ~98 GiB OOM figure) fails, and the exact boundary fails closed
    (strict '<', never '<='). All three run to completion."""
    under = item_11_bound_check(int(0.3 * GIB))
    over = item_11_bound_check(int(98 * GIB))
    exact_bound = item_11_bound_check(2 * GIB)
    ok = under["passed"] is True and over["passed"] is False and exact_bound["passed"] is False
    _report("smoke 9 (sec 1.34 F3): item_11_bound_check's comparison logic has teeth on "
            "synthetic peak-allocation numbers -- in-bound (0.3 GiB) passes, over-bound "
            "(~98 GiB, mirroring sec 1.27's OOM figure) fails, exact-boundary (2 GiB) fails "
            "closed (strict <)", ok,
            f"under_passed={under['passed']} over_passed={over['passed']} "
            f"exact_bound_passed={exact_bound['passed']}")


def smoke_10_item11_device_guard_refuses_off_cuda():
    """[NEW, sec 1.34 F3] `item_11_real_k48_rung2_fit` must NEVER claim PASS off a real CUDA
    device -- device='cpu' (this dev box's reality) must return passed=None and write NO gate
    token, never silently 'pass' by skipping the check."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = item_11_real_k48_rung2_fit(td, device="cpu")
        token_path = os.path.join(td, ITEM_11_TOKEN_NAME)
        ok = (r["passed"] is None and r["status"] == "box_only_skipped"
              and not os.path.isfile(token_path))
    _report("smoke 10 (sec 1.34 F3): item_11_real_k48_rung2_fit device-guards -- device='cpu' "
            "returns passed=None (never True) and writes NO gate token", ok, f"result={r}")


def print_checklist_table() -> None:
    print("=" * 90)
    print(f"{'id':<36} {'blocking':<9} {'source_finding'}")
    print("-" * 90)
    for row in CHECKLIST:
        print(f"{row['id']:<36} {str(row['blocking']):<9} {row['source_finding']}")
    print("=" * 90)
    n_blocking = sum(1 for row in CHECKLIST if row["blocking"])
    print(f"{len(CHECKLIST)} items total, {n_blocking} blocking, {len(CHECKLIST) - n_blocking} deferred")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                     help="print the checklist table, then run the structural self-validation smoke")
    ap.add_argument("--smoke", action="store_true", help="alias for --list (house convention)")
    ap.add_argument("--run-item-11", action="store_true",
                     help="[NEW, sec 1.34 F3] BOX-ONLY: actually runs "
                          "fit_rung2_identity_classifier at the real K=48 shape on CUDA, asserts "
                          "peak allocation < 2 GiB, writes the gate token on PASS. "
                          "Device-guarded: refuses to claim PASS on CPU.")
    ap.add_argument("--gates-dir", type=str, default=None,
                     help="gates dir to write BOX_SMOKE_ITEM_11_K48_REAL_KERNEL_PASSED.token "
                          "into (required with --run-item-11).")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.run_item_11:
        if not args.gates_dir:
            print("REFUSE: --run-item-11 requires --gates-dir.", file=sys.stderr)
            return 2
        result = item_11_real_k48_rung2_fit(args.gates_dir, args.device)
        print(f"item 11: {result}")
        if result["passed"] is None:
            print("BOX-ONLY: item 11 could not be verified off a real CUDA device -- no token "
                  "written, h2h_rung1_chain.sh Stage B3's pre-flight will refuse.", file=sys.stderr)
            return 3
        if not result["passed"]:
            print("ITEM 11 FAILED: measured peak allocation >= the 2 GiB bound.", file=sys.stderr)
            return 1
        print(f"ITEM 11 PASSED: token written to {result.get('token_path')}")
        return 0

    if args.list or args.smoke or not sys.argv[1:]:
        print_checklist_table()
    print("=" * 70)
    print("h2h_box_smoke_checklist.py -- sec 1.20 REGISTERED BOX-SMOKE CHECKLIST self-validation")
    print("=" * 70)
    smoke_1_checklist_shape()
    smoke_2_every_entry_has_required_keys()
    smoke_3_ids_unique_and_slug_shaped()
    smoke_4_blocking_is_bool_and_matches_sec_1_20()
    smoke_5_aud_f1_item_present_and_positioned()
    smoke_6_descriptions_nonempty()
    smoke_7_sec_1_23_items_present_and_positioned()
    smoke_8_item11_present_and_positioned()
    smoke_9_item11_bound_logic_has_teeth()
    smoke_10_item11_device_guard_refuses_off_cuda()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
