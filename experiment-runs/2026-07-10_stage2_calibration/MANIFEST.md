# 2026-07-10 Stage 2 deploy + calibration chain — HALTED at the fla cross-check gate

Registry: `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` §2.25 (full record).
EXPERIMENT_LOG.md: "CAPABILITY SEPARATION STAGE 2 — DEPLOY+CALIBRATION CHAIN HALTED..." entry.

## What happened

1. **Deploy (md5-verified).** The five never-before-deployed Stage-2 files
   (`stage2_composer.py`, `stage2_instrument.py`, `stage2_run.py`,
   `stage2_task.py`, `smoke_stage2.py`) were scp'd to
   `/home/nvidia/chapter2/capability_separation/` and md5-verified
   byte-exact local==box. All 17 shared capability-separation files were
   already current (zero redeploy needed).
2. **Box smoke** (`box_smoke_stage2.log`, 6 sections): 4/6 PASS, 2/6 FAIL
   (sections 1 and 3, both touching `stage2_composer.py::fla_cross_check`).
3. **The box-only fla cross-check item** (run explicitly with
   `device="cuda"` at the pinned configs {(1,1),(1,8),(2,8)}): crashed
   (`fla_cross_check_box_crash.log`) — `chunk_delta_rule` never receives
   `output_final_state=True`, so `final_state=None`. A due-diligence
   diagnostic (NOT a permanent edit — the deployed file was verified
   byte-unchanged afterward) patched exactly that one kwarg on a `/tmp`
   scratch copy and re-ran the real cross-check using the composer's own
   already-audited conventions: it FAILS hard and deterministically at
   all three configs (`fla_cross_check_diagnostic_patched_result.log`,
   diff in `fla_cross_check_diagnostic_patch.diff`):

   | config (n_h, D) | rel-Frobenius | tolerance | verdict |
   |---|---|---|---|
   | (1, 1) | 1.4008 | ≤1e-2 | FAIL (140x over) |
   | (1, 8) | 1.3589 | ≤5e-2 | FAIL (27x over) |
   | (2, 8) | 1.3825 | ≤5e-2 | FAIL (28x over) |

4. **Per this dispatch's pre-registered charter, the chain HALTED here.**
   Arm-1 retrain and the 11-cell calibration gate were NOT launched.
   Zero GPU-h spent training.
5. **The harvest analysis script was written and self-tested regardless**
   (`matrix-thinking/capability_separation/stage2_harvest.py`, repo-committed,
   not yet box-deployed): 7/7 CPU self-tests pass.

## Files in this archive

- `box_smoke_stage2.log` — full 6-section box smoke output (md5 `fcf641ac412fca0ca79d9a0f271c3b7d`)
- `fla_cross_check_box_crash.log` — the real (unpatched) `fla_cross_check(device='cuda')` crash on box GPU 0 (md5 `cce8c6da05aa5e38798fc2e6db1fc020`)
- `fla_cross_check_diagnostic_patched_result.log` — the diagnostic-patched-copy re-run's real numbers (deterministic across two runs)
- `fla_cross_check_diagnostic_patch.diff` — the exact one-line diff applied to the `/tmp` scratch copy (never applied to the deployed/repo file)

## Deploy md5 table (local == box, all verified)

```
ac8b046f9c85398891731fd07a956345  beta_fla_smoke.py        (already current)
30579188b21634b844fdc0223eac4aa5  blank_out.py              (already current)
08eebfa8aa699d09bff88ae4940540dd  budget_guard.py            (already current)
656e6030447cc3a143c68d0103bed65e  coverage_calibration.py    (already current)
3ec5f06206fa01bb983e2dfd9750dca8  force_rank_arms.py         (already current)
90428f43a9367c97dea506a12e1802ab  gate1_synthetic_injection.py (already current)
d1c9f379b15d74d1b3aaca5ba06843f2  group_task.py               (already current)
d51922a29a7df08b7ffd8cf897799d72  group_word_encoder.py       (already current)
9bd8db6829eb9c9bfd9411070720e569  groups.py                   (already current)
9d4e8f1bdffad0a4a44c11400bc6cc77  marquee_power_simulation.py (already current)
1707f3f9e86259fb4977fe82d3f438c8  readout.py                  (already current)
41e0e65e8ad9c8d1b5b538edac6e62bf  run_capability_sep.py       (already current)
e3e0c0eb44bc18026e8b47b255666386  smoke_capability_sep.py     (already current)
10ff599627d3be54f0e39ed4aa85d1f3  spearman_null_calibration.py (already current)
2fcb8e1fbd2210919ee0e0ff6feb97c2  tost_analysis.py            (already current)
07402b61a4b73926903580bfd9745865  truncation_curve.py         (already current)
4590bb77260ba2c251c200f08d89e85c  verify_option_a_readout.py  (already current)
6ce5689a8d35bf46054f072d5fc848f6  smoke_stage2.py             (NEWLY DEPLOYED)
39c7f5f476e67e75a622feaa9d33cdfd  stage2_composer.py          (NEWLY DEPLOYED)
6b26ee7004602ae2fc2d5a29e84e12fe  stage2_instrument.py        (NEWLY DEPLOYED)
18db11ac2caf6551bc02f19b84bd2132  stage2_run.py               (NEWLY DEPLOYED)
9c0a7de45c6006faf354bdec52a4e725  stage2_task.py              (NEWLY DEPLOYED)
```

## Next

A build/fix + independent-audit round on `stage2_composer.py::fla_cross_check`
(two defects: the self-skip guard's device-blindness, and the
`output_final_state`/`allow_neg_eigval` kwarg mismatch that unmasks a
≥28x real numerical disagreement), then re-attempt
deploy→box-smoke→cross-check→retrain→calibration-gate from
`CAPABILITY_SEPARATION_DESIGN.md` §2.25's registry pointer.
