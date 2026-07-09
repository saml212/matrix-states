"""CAPABILITY_SEPARATION_DESIGN.md S2 (Rev 3, DESIGN-CLEARED-FOR-BUILD per
S2.18) -- the Stage-2 master pre-launch smoke gate, mirroring
`smoke_capability_sep.py`'s own dependency-ordered convention (a
downstream component's smoke should not be trusted until its dependencies'
smokes have already passed) and `chapter2/run_task_d.py --smoke`'s
original pattern.

Run FIRST, before any GPU cell launches:

    DRY_RUN_BYPASS=1 .venv/bin/python smoke_stage2.py

(S2.20 m3: this file's own filename starts with "smoke_", but the
invocation above passes no `--smoke`/`--dry-run`/`--test`/`--quick` FLAG
literally in the command line -- it IS the smoke gate, not a script that
takes a smoke flag -- so the repo's `pre-train-gate.sh` hook, which
pattern-matches ANY `python *.py` invocation and only exempts commands
containing one of those flag substrings, would otherwise require a
dry-run sentinel for this file. `DRY_RUN_BYPASS=1` is therefore the
correct, sanctioned bypass here (same established convention as
`deltanet_rd/mech_stage05_selftests.py`/`poolmargin_stage_minus1.py`'s own
docstrings) -- every section below is pure-CPU (box-only items
self-skip loudly, listed at the bottom), no GPU touched.)

Sections, in dependency order:
  1. stage2_composer.py  -- GroupWordDeltaComposer forward/backward, the
     PROVEN rank bound, n_h in {1,2,4}, beta in {1,2}, use_bos_row,
     last-K-window truncation, forward_all trajectory, the S2.20 F6
     +-15%-of-Arm-1 param-match check (all 5 groups x n_h in {1,2,4}).
  2. stage2_composer.py  -- blank-out / P=1 bottleneck (+ planted-leak
     NEGATIVE TEST) -- a genuinely different computation graph from
     GroupWordEncoder's own blank-out result, re-verified here, not
     inherited (S2.2.2/S2.9 item 6).
  3. stage2_composer.py  -- one-cell fla numerical cross-check (BOX-ONLY,
     self-skips here -- CUDA+real-fla required, disclosed).
  4. stage2_task.py       -- TARGET RANK UNIT TEST (S1.33 [LEARN], with a
     planted eye-padding-style mutant that must be DETECTED, not silently
     passed), per-fixed-depth coverage bars, the D_test grid evaluation
     pipeline, the S2.20 F1 Arm-1 depth-eval PROOF-BY-RUN (D=12 must score
     eval_L==12, not silently fall back to the L~Uniform{1,8} train-support
     default), the S2.20 m4 D=1 guard (every entry point raises cleanly,
     incl. with an explicit bar override), Arm-1 L_max-ceiling / missing-
     checkpoint handling, prefix fidelity.
  5. stage2_instrument.py -- THE QUERY-DEPENDENCE DIAGNOSTIC (S2.8 item
     2(e)): anchor rank EXACTLY min(32,n_h*D) (proven, not just tested),
     QR-basis determinism, POSITIVE CONTROLS (a planted-healthy state must
     PASS) and NEGATIVE CONTROLS (a planted-degenerate reader must FAIL
     both bars; a zeroed anchor must trip the health floor and route to
     instrument-defect, not the BOS fix; a ceiling-demoted depth must
     discharge the launch condition; the S2.20 F2 partial-collapse control
     -- a genuine 31/32-dead-channel plant, exact axis-aligned linear
     algebra, not a hacked reader -- FAILS under the pinned MEAN
     aggregation and is proven to FALSELY PASS under a mean->max mutation
     applied live and reverted) -- both directions have teeth, per S1.28's
     positive-control rule this dispatch cites.
  6. stage2_run.py        -- grid construction (68 cells: 50 primary + 18
     n_h-grid; 11-cell calibration-first set), per-cell + ledger budget
     guards (NEGATIVE-then-POSITIVE), resume-safety (incl. a corrupted-
     output re-run and a checkpoint round-trip reload for the last-K
     control), one full calibration-cell pipeline end-to-end, the S2.20 F3
     parameter-fingerprint protection (positive + the exact §2.19-class
     fresh-composer mutation caught), `run_calibration_wave` end-to-end at
     ALL SEVEN pinned depths (S2.20 F5), `run_real_cell` end-to-end (S2.20
     F4 -- cosine_loss, M-D0 profile, the 7-depth gate, param-match,
     D_test_results making `is_valid_output`'s not-None check real, incl.
     a NEGATIVE corrupted-D_test_results-to-None proof), and a WIRING-only
     smoke of the ARM-1 retrain-and-save utility (S2.20 box item 4 -- the
     real ~0.21 GPU-h launch is NOT run here).

BOX-ONLY items, disclosed (not run here, not silently assumed clean):
  - stage2_composer.py::fla_cross_check -- CUDA + real `fla` required.
  - Any real GPU training / the real 25 GPU-h calibration+sweep launch --
    this BUILD agent does not launch (an independent audit follows,
    S2.11's own sequencing; PI-signoff env var gate wired in
    stage2_run.py::main, unset here).
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _section(title):
    print("\n" + "#" * 100)
    print(f"# {title}")
    print("#" * 100)


def main():
    failures, box_only = [], []

    def run(label, fn):
        _section(label)
        try:
            fn()
            print(f"\n[OK] {label}")
        except Exception:
            print(f"\n[FAIL] {label}")
            traceback.print_exc()
            failures.append(label)

    def _composer_core():
        import stage2_composer as sc
        sc.smoke("cpu")
    run("1. stage2_composer.py -- forward/backward, PROVEN rank bound, n_h/beta grid, "
        "use_bos_row, last-K truncation, forward_all", _composer_core)

    def _composer_blankout():
        import torch
        import stage2_composer as sc
        torch.manual_seed(0)
        m = sc.GroupWordDeltaComposer(d_state=5, n_gens=4, h=32, n_h=2, beta_max=2.0)
        sc.run_composer_blank_out_test(m, device="cpu")
        sc.run_composer_blank_out_planted_leak_test(m, device="cpu")
    run("2. stage2_composer.py -- P=1 bottleneck blank-out (new forward pass, re-verified, "
        "not inherited from GroupWordEncoder) + planted-leak NEGATIVE TEST", _composer_blankout)

    def _fla_cross_check():
        import stage2_composer as sc
        result = sc.fla_cross_check(device="cpu")
        assert result["status"] in ("skipped_cpu_or_stub", "ran_real_fla")
        if result["box_only"]:
            box_only.append("stage2_composer.py::fla_cross_check (CUDA + real fla required)")
        else:
            assert result["all_ok"], f"fla cross-check FAILED on real hardware: {result['results']}"
    run("3. stage2_composer.py -- one-cell fla numerical cross-check (BOX-ONLY, self-skips here)",
        _fla_cross_check)

    def _task():
        import stage2_task as st
        st.smoke()
    run("4. stage2_task.py -- TARGET RANK UNIT TEST (S1.33 [LEARN], planted-mutant detection), "
        "per-depth coverage bars, D_test grid eval, Arm-1 L_max handling, prefix fidelity", _task)

    def _instrument():
        import stage2_instrument as si
        si.smoke()
    run("5. stage2_instrument.py -- query-dependence diagnostic: anchor rank EXACT proof, QR "
        "determinism, planted-degenerate NEGATIVE control, planted-healthy POSITIVE control, "
        "anchor-health-floor routing, ceiling-demotion discharge", _instrument)

    def _run_cli():
        import stage2_run as sr
        sr.smoke()
    run("6. stage2_run.py -- grid construction (68/11), per-cell + ledger budget guards, "
        "resume-safety (+ corrupted-output re-run, checkpoint round-trip reload), "
        "one full calibration-cell pipeline", _run_cli)

    print("\n" + "=" * 100)
    if box_only:
        print("BOX-ONLY ITEMS (disclosed, not exercised on this CPU build; verify on the box "
              "before the calibration gate trusts them):")
        for item in box_only:
            print(f"  - {item}")
        print("=" * 100)
    if failures:
        print(f"STAGE-2 SMOKE GATE: {len(failures)} FAILURE(S): {failures}")
        print("=" * 100)
        sys.exit(1)
    print("STAGE-2 SMOKE GATE: ALL 6 SECTIONS PASSED.")
    print("=" * 100)


if __name__ == "__main__":
    main()
