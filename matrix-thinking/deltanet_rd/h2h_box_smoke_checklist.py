"""h2h_box_smoke_checklist.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.20's
REGISTERED BOX-SMOKE CHECKLIST, as a machine-readable manifest the deploy
stage can consume (sec 1.20's own "binding" checklist language). Deliberately
dumb and declarative -- a flat list of dicts, no logic beyond structural
self-validation; this file does NOT run any of the 10 items itself (every one
of them needs real fla/Triton on the H100 box, per each item's own
`source_finding`), it only records WHAT the deploy stage owes before
HEADTOHEAD_MATCH_GATE_SIGNOFF / launch can be considered box-verified.

Items (9)-(10) (NEW, sec 1.23 scoped build-fix, item 5) are the Rev 4
CE_answer continuation's own blank-out negative tests -- see
h2h_cell_train_rd.py's `mode_selftest` for the CPU-provable halves (both
items ARE exercised meaningfully on CPU, unlike items 2/3's stub-vacuity
limitation: neither test depends on S_T being nonzero, only on the
CONTINUATION CALL's own signature/determinism, which the CPU stub does not
disconnect) and the box-only real-kernel residual each one still owes.

Each entry:
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
"""
from __future__ import annotations

import argparse
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
)

REQUIRED_KEYS = ("id", "description", "blocking", "source_finding")


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
    ok = len(CHECKLIST) == 10
    _report("smoke 1: checklist has exactly 10 items (sec 1.20's own registered 8 + sec 1.23 "
            "build-fix item 5's new items 9-10)", ok, f"n_items={len(CHECKLIST)}")


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
    args = ap.parse_args()
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
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
