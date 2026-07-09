"""h2h_box_smoke_driver.py -- DEPLOY-STAGE box-side driver for
h2h_box_smoke_checklist.py items 1-4 and sec 1.7 gates 6+7, on REAL
fla/Triton kernels. Zero logic changes to any audited file: this driver
only (a) sets torch's default device to CUDA before importing the audited
suites (their smoke functions build models/tensors device-agnostically --
on the dev box they run under the CPU stub; here every factory call lands
on a real GPU), (b) runs them, (c) runs the two checks the audited files
themselves REGISTERED as deploy-stage obligations rather than implementing
(checklist item 2's end-to-end tap-changes-with-q on a real NONZERO
S_T_last -- probe_head_rd smoke_3's disclosed box-only follow-up -- and a
direct-numbers re-read of item 3's isolation sums for the token record),
and (d) writes the gate token files ONLY if every blocking item passed.

Token files written (consumed by h2h_rung1_chain.sh's stage-0 gate; sec
1.7 gate 5's HEADTOHEAD_MATCH_GATE_SIGNOFF is only exported by the chain
after these exist):
  GATE6_MATCH_GATE_PASSED.token            (verify_match_gate 2-pass, real kernel)
  GATE7_PROBE_CAPACITY_NULL_PASSED.token   (per-arm nulls at rung-1 shapes)
  BOX_SMOKE_ITEMS_1_4_PASSED.token         (checklist items 1-4 details)

Run (on the box): CUDA_VISIBLE_DEVICES=0 /home/nvidia/tdenv/bin/python \
    h2h_box_smoke_driver.py --token-dir results/h2h_rung1/gates
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VOCAB_TOTAL_GRAMMAR = 50259    # GPT-2 base + BUFFER + <Q> (h2h_cell_train_rd DEPLOY-PIN-3)
VALUE_DIM = 64
RUNG1_TAP_DIMS = {"contender": 64, "ablation": 64, "transformer": 256}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token-dir", type=str, default="results/h2h_rung1/gates")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "box driver requires CUDA (this IS the real-kernel gate)"
    torch.set_default_device("cuda")

    import probe_head_rd as ph
    import verify_match_gate as vg
    from lm_pretrain_rd import DeltaNetLM

    assert not ph._STUB_INSTALLED, (
        "CPU stub is installed -- this driver must run against REAL fla (do not set "
        "REASONING_LINK_FORCE_CPU_STUB here)")

    report: dict = {"host": os.uname().nodename, "started_iso":
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "torch": torch.__version__, "stub_installed": False}
    hard_fail = False

    # ---- checklist item 1: production-path fwd/bwd/grad smoke, real kernel (own process,
    # no default-device override -- lm_pretrain_rd.smoke() manages its own device) ----
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "lm_pretrain_rd.py"), "--smoke"],
                       capture_output=True, text=True)
    item1_ok = r.returncode == 0
    report["item1_lm_pretrain_smoke"] = {"exit": r.returncode,
                                         "tail": r.stdout.strip().splitlines()[-3:] if r.stdout else [],
                                         "stderr_tail": r.stderr.strip().splitlines()[-3:] if r.stderr else []}
    print(f"[item 1] lm_pretrain_rd.py --smoke (real kernel): {'PASS' if item1_ok else 'FAIL'}")
    hard_fail = hard_fail or not item1_ok

    # ---- full audited probe-head suite on real kernels (covers item 3's smoke_8c BOX branch,
    # smoke_2/5/7, and smoke_3's two CPU-provable halves now on the real kernel). smoke_1 and
    # smoke_6 are fla-FREE and use an explicit CPU torch.Generator inside
    # random_unit_rows_init (which the cuda default-device would break); they are exactly the
    # items the design calls CPU-feasible/no-backbone (sec 1.3.1.4), so they run under a CPU
    # device context -- same code, correct device semantics. ----
    ph.FAILURES.clear()
    with torch.device("cpu"):
        ph.smoke_1_target_table_frozen_unit_rows()
        ph.smoke_6_probe_capacity_null_all_three_arm_shapes()
    ph.smoke_2_contender_tap_forward_backward()
    ph.smoke_3_contender_tap_p1_bottleneck_blankout()
    ph.smoke_4_ablation_tap_forward_and_bottleneck()
    ph.smoke_5_transformer_tap_uncapped_vs_capped()
    ph.smoke_7_joint_loss_gradients_flow_to_backbone()
    ph.smoke_8_aux_loss_only_gradient_isolation()
    probe_suite_ok = not ph.FAILURES
    report["probe_head_suite_real_kernel"] = {"failures": list(ph.FAILURES)}
    print(f"[probe suite] real-kernel run of the audited suite: "
          f"{'PASS' if probe_suite_ok else 'FAIL ' + str(ph.FAILURES)}")
    hard_fail = hard_fail or not probe_suite_ok

    # ---- checklist item 2: END-TO-END tap-changes-with-q on a real NONZERO S_T_last
    # (smoke_3's registered box-only follow-up -- implemented here, the deploy stage) ----
    torch.manual_seed(20260708)
    m = DeltaNetLM(300, d_model=32, d_state=64, n_layers=1, conv_size=4)
    ctx = torch.randint(0, 300, (2, 128))
    q_a = torch.randint(0, 300, (2, 3, 5))
    q_b = torch.randint(0, 300, (2, 3, 5))
    with torch.no_grad():
        _, fs = m(ctx, return_states=True)
        s_abs = fs[-1].abs().sum().item()
        tap_a = ph.contender_native_tap(m, ctx, q_a)
        tap_b = ph.contender_native_tap(m, ctx, q_b)
        _, fs2 = m(ctx, return_states=True)
    state_nonzero = s_abs > 0.0
    tap_changes = not torch.allclose(tap_a, tap_b)
    state_ctx_only = torch.allclose(fs[-1], fs2[-1])   # P=1 bottleneck half, real kernel
    item2_ok = state_nonzero and tap_changes and state_ctx_only
    report["item2_tap_changes_with_q_real_state"] = {
        "S_T_abs_sum": s_abs, "tap_changes_with_q": tap_changes,
        "state_is_context_only": state_ctx_only,
        "tap_a_minus_b_max_abs": (tap_a - tap_b).abs().max().item()}
    print(f"[item 2] end-to-end tap-changes-with-q, REAL nonzero S_T "
          f"(|S|={s_abs:.4f}): {'PASS' if item2_ok else 'FAIL'}")
    hard_fail = hard_fail or not item2_ok

    # ---- checklist item 3 (AUD-F1): direct-numbers re-read of the contender's aux-ONLY
    # isolation on the real kernel (smoke_8c's box branch above already pass/failed it;
    # this records the actual grad sums in the token file) ----
    torch.manual_seed(42)
    m3 = DeltaNetLM(300, d_model=32, d_state=64, n_layers=1, conv_size=4)
    adapter = ph.build_adapter_arm(64, 12)
    probe = ph.build_shared_probe(12)
    tap = ph.contender_native_tap(m3, ctx, q_a)
    target = torch.randn(2, 3, 12)
    (ph.AUX_WEIGHT_DEFAULT * ph.probe_aux_loss(probe(adapter(tap)), target)).backward()
    sums = ph._grad_abs_sums(m3.blocks[-1], ph._ISOLATION_CHECK_ATTR_PATH["contender"])
    item3_ok = ph._all_positive(sums)
    m3.zero_grad()
    (ph.AUX_WEIGHT_DEFAULT * ph.probe_aux_loss(probe(adapter(tap.detach())), target)).backward()
    sums_neg = ph._grad_abs_sums(m3.blocks[-1], ph._ISOLATION_CHECK_ATTR_PATH["contender"])
    item3_neg_ok = ph._all_zero_or_none(sums_neg)
    report["item3_aux_only_isolation_contender"] = {"grad_abs_sums": sums,
                                                    "detach_negative_sums": sums_neg}
    print(f"[item 3] contender aux-ONLY k/v/b grads nonzero on real kernel: "
          f"{'PASS' if item3_ok and item3_neg_ok else 'FAIL'} -- {sums}")
    hard_fail = hard_fail or not (item3_ok and item3_neg_ok)

    # ---- gate 6 (+ checklist item 4): the 2-pass MATCH-GATE with a REAL measured forward ----
    vg.FAILURES.clear()
    teeth_ok = vg.negative_test_gate_has_teeth()
    gate6_ok = vg.run_match_gate() and teeth_ok
    p1 = vg.run_pass1()
    item4_ok = p1["measured_state_bytes"] == 32_768
    report["gate6_match_gate"] = {"teeth_ok": teeth_ok, "failures": list(vg.FAILURES),
                                  "pass1": {k: v for k, v in p1.items() if k != "cap_length_table"},
                                  "cap_length_table": {str(k): v for k, v in p1["cap_length_table"].items()}}
    report["item4_state_bytes_roundtrip"] = {"measured_state_bytes": p1["measured_state_bytes"],
                                             "expected": 32_768,
                                             "dtype_assert": "float32 (asserted inside "
                                             "pass1_measured_state_bytes, M2 pin)"}
    print(f"[gate 6] MATCH-GATE 2-pass, real kernel: {'PASS' if gate6_ok else 'FAIL'}")
    print(f"[item 4] measured_state_bytes == 32768 fp32: {'PASS' if item4_ok else 'FAIL'} "
          f"({p1['measured_state_bytes']})")
    hard_fail = hard_fail or not (gate6_ok and item4_ok)

    # ---- gate 7: probe-capacity null at every arm's REAL rung-1 adapter shape,
    # vocab_size_total-sized T_val (sec 1.3.1.4 / sec 1.7 gate 7). The null is the design's own
    # CPU-feasible no-backbone harness (no fla anywhere) -- run under the CPU device context for
    # the same generator-semantics reason as smoke_1/6 above. ----
    nulls = {}
    gate7_ok = True
    # Literal per-arm seeds (deploy audit MAJOR-3): NEVER Python's salted hash() -- the same
    # rule probe_head_rd smoke_6's own docstring pins; literal pins are the reproducible form.
    gate7_seeds = {"contender": 20260801, "ablation": 20260802, "transformer": 20260803}
    with torch.device("cpu"):
        T_val = ph.build_probe_target_table(VOCAB_TOTAL_GRAMMAR, VALUE_DIM)
        for arch, tap_dim in RUNG1_TAP_DIMS.items():
            n = ph.run_probe_capacity_null(tap_dim, VALUE_DIM, T_val, seed=gate7_seeds[arch])
            nulls[arch] = n
            gate7_ok = gate7_ok and n["passed"]
            print(f"[gate 7] {arch} (tap_dim={tap_dim}): recovered_frac={n['recovered_frac']:.6f} "
                  f"{'PASS' if n['passed'] else 'FAIL'} (bar < 0.05)")
    report["gate7_probe_capacity_null"] = nulls
    hard_fail = hard_fail or not gate7_ok

    report["finished_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["all_blocking_passed"] = not hard_fail

    if hard_fail:
        print("BOX SMOKE: BLOCKING FAILURE(S) -- NO token files written, deploy stage must STOP "
              "and report (h2h_box_smoke_checklist.py's own blocking rule).", file=sys.stderr)
        print(json.dumps(report, indent=2, default=str), file=sys.stderr)
        return 1

    os.makedirs(args.token_dir, exist_ok=True)
    with open(os.path.join(args.token_dir, "GATE6_MATCH_GATE_PASSED.token"), "w") as f:
        json.dump({"gate": 6, **report["gate6_match_gate"], "written_iso": report["finished_iso"],
                   "host": report["host"]}, f, indent=2, default=str)
    with open(os.path.join(args.token_dir, "GATE7_PROBE_CAPACITY_NULL_PASSED.token"), "w") as f:
        json.dump({"gate": 7, "nulls": nulls, "written_iso": report["finished_iso"],
                   "host": report["host"]}, f, indent=2, default=str)
    with open(os.path.join(args.token_dir, "BOX_SMOKE_ITEMS_1_4_PASSED.token"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"BOX SMOKE: ALL BLOCKING ITEMS PASSED -- token files written under {args.token_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
