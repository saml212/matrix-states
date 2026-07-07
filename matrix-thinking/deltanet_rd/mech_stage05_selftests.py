"""mech_stage05_selftests.py -- FROZEN_BIAS_LM_DESIGN.md sec 12.3.4, Stage 0.5:
synthetic self-tests for sec 12's THREE new statistics (`repeat_excess` (H1/H5),
the parameter-diff norm/cosine (H4), and gradient-flow norm telemetry (H3)),
each against a hand-derived/hand-picked EXACT expected value, per this project's
own standing "run the negative unit test that's supposed to prove the check has
teeth to completion" [LEARN], mirroring fit_frozenbias_estimation.py's own
_self_test() rigor.

GATE (sec 12.3.4 / sec 12.4 header): Stage 1 may not launch until every test
below has been RUN and PASSED. This script IS that gate -- run it directly:

    DRY_RUN_BYPASS=1 python3 mech_stage05_selftests.py

(the repo's pre-train-gate hook pattern-matches any `python *.py` invocation;
DRY_RUN_BYPASS=1 is the correct, sanctioned bypass here since this script does
no training and touches no GPU at all -- pure-Python/CPU arithmetic plus one
CPU-only torch autograd hook exercise for self-test 3.)

All three tests are formula-level correctness checks against hand-picked
inputs with hand-derived expected outputs -- they do NOT exercise the real
`capture_raw_keys`/`torch.load`/hook-registration code paths the real
Stage-1/Stage-2 scripts will run (sec 12.9 item 4, an openly disclosed gap,
carried forward, not closed by this script).

Persisted via mech_schema.wrap_exploratory, per sec 12.3.1's schema requirement
(binding on every sec 12 JSON artifact, Stage 0 through 2 alike).
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mech_schema import wrap_exploratory

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mech_wave")
TOL_ARITH = 1e-4   # self-test 1's own registered tolerance
TOL_TIGHT = 1e-6   # self-test 2's tighter tolerance (norm 1e-9, cosine 1e-6 per spec)
TOL_NORM = 1e-9
TOL_GRAD = 1e-5    # self-test 3's registered tolerance


# ===========================================================================
# Self-test 1 -- repeat_excess (H1/H5), sec 12.3.4 "Self-test 1"
# ===========================================================================

def _l2_normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def repeat_excess_stats(vectors, tok_ids):
    """Sec 12.4's own defining formula, at the flat (already-pooled-episode)
    level self-test 1 exercises directly (the real Stage-1 script pools this
    same computation across many (b, chunk, head) episodes, n-weighted, the
    way summarize_gram_records already pools gram_deviation -- out of scope
    for this formula-level self-test).

    vectors: list of raw (unnormalized) coordinate tuples.
    tok_ids: list of token identifiers, same length, position-aligned.

    Returns dict with same_tok_sim, diff_tok_sim, repeat_excess, n_same, n_diff
    -- ordered pairs (i, j), i != j (both (i,j) and (j,i) counted, matching the
    n_same=24/n_diff=108 and n_same=30/n_diff=102 counts sec 12.3.4 pins for its
    two constructions on 12 vectors)."""
    assert len(vectors) == len(tok_ids)
    normed = [_l2_normalize(v) for v in vectors]
    n = len(normed)
    same_sims, diff_sims = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = sum(a * b for a, b in zip(normed[i], normed[j]))
            if tok_ids[i] == tok_ids[j]:
                same_sims.append(c)
            else:
                diff_sims.append(c)
    assert same_sims, "no same-token pairs found -- construction has no repeats"
    assert diff_sims, "no diff-token pairs found -- construction is degenerate"
    same_tok_sim = sum(same_sims) / len(same_sims)
    diff_tok_sim = sum(diff_sims) / len(diff_sims)
    return {
        "same_tok_sim": same_tok_sim,
        "diff_tok_sim": diff_tok_sim,
        "repeat_excess": same_tok_sim - diff_tok_sim,
        "n_same": len(same_sims),
        "n_diff": len(diff_sims),
    }


# The 12 hand-picked (not RNG-drawn) 3-D vectors, sec 12.3.4 self-test 1.
_ST1_RAW = [
    (1, 2, -1), (0.5, -1.5, 2), (-2, 0.5, 1), (1.5, 1.5, 1.5),
    (-1, -2, 0.5), (2, -0.5, -1.5), (0, 1, -2), (-1.5, 0.5, -1),
    (1, -1, 1), (-0.5, 2, 0.5), (2, 2, -0.5), (-1, 0, 2),
]


def selftest_1_repeat_excess():
    print("\n" + "=" * 70)
    print("SELF-TEST 1 -- repeat_excess (H1/H5)")
    print("=" * 70)
    ok = True

    # (a) Uncorrelated-assignment construction: round-robin ['A','B','C','D']*3.
    tok_a = ["A", "B", "C", "D"] * 3
    r_a = repeat_excess_stats(_ST1_RAW, tok_a)
    print(f"(a) uncorrelated: repeat_excess={r_a['repeat_excess']:.6f} "
          f"same_tok_sim={r_a['same_tok_sim']:.6f} diff_tok_sim={r_a['diff_tok_sim']:.6f} "
          f"n_same={r_a['n_same']} n_diff={r_a['n_diff']}")
    exp_a = {"repeat_excess": -0.238404, "same_tok_sim": -0.252186, "diff_tok_sim": -0.013781,
              "n_same": 24, "n_diff": 108}
    a_ok = (
        abs(r_a["repeat_excess"] - exp_a["repeat_excess"]) < TOL_ARITH
        and abs(r_a["same_tok_sim"] - exp_a["same_tok_sim"]) < TOL_ARITH
        and abs(r_a["diff_tok_sim"] - exp_a["diff_tok_sim"]) < TOL_ARITH
        and r_a["n_same"] == exp_a["n_same"]
        and r_a["n_diff"] == exp_a["n_diff"]
    )
    print(f"    expected repeat_excess=-0.238404 (+/-1e-4), n_same=24, n_diff=108  PASS={a_ok}")
    ok = ok and a_ok

    # (b) Planted-clustering construction: positions (0,10),(1,8),(2,11) (cosines
    # 0.924/0.906/0.781) merged into ONE shared token id; every other position a
    # unique singleton (no other repeats).
    tok_b = [str(i) for i in range(12)]
    shared = "SHARED"
    for p in (0, 10, 1, 8, 2, 11):
        tok_b[p] = shared
    r_b = repeat_excess_stats(_ST1_RAW, tok_b)
    print(f"(b) planted:      repeat_excess={r_b['repeat_excess']:.6f} "
          f"same_tok_sim={r_b['same_tok_sim']:.6f} diff_tok_sim={r_b['diff_tok_sim']:.6f} "
          f"n_same={r_b['n_same']} n_diff={r_b['n_diff']}")
    exp_b = {"repeat_excess": 0.034437, "same_tok_sim": -0.030517, "diff_tok_sim": -0.064954,
              "n_same": 30, "n_diff": 102}
    b_ok = (
        abs(r_b["repeat_excess"] - exp_b["repeat_excess"]) < TOL_ARITH
        and abs(r_b["same_tok_sim"] - exp_b["same_tok_sim"]) < TOL_ARITH
        and abs(r_b["diff_tok_sim"] - exp_b["diff_tok_sim"]) < TOL_ARITH
        and r_b["n_same"] == exp_b["n_same"]
        and r_b["n_diff"] == exp_b["n_diff"]
    )
    print(f"    expected repeat_excess=+0.034437 (+/-1e-4), n_same=30, n_diff=102  PASS={b_ok}")
    ok = ok and b_ok

    # Planted delta must exceed construction (a) by >= 0.2 (registered delta = 0.272841).
    delta = r_b["repeat_excess"] - r_a["repeat_excess"]
    delta_ok = delta >= 0.2
    print(f"    planted delta = {delta:.6f} (registered 0.272841), >= 0.2 required  PASS={delta_ok}")
    ok = ok and delta_ok

    return ok, {
        "construction_a_uncorrelated": r_a,
        "construction_b_planted": r_b,
        "planted_delta": delta,
        "expected": {"construction_a": exp_a, "construction_b": exp_b, "min_planted_delta": 0.2},
    }


# ===========================================================================
# Self-test 2 -- parameter-diff norm/cosine (H4), sec 12.3.4 "Self-test 2"
# ===========================================================================

def frobenius_norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine(a, b):
    na, nb = frobenius_norm(a), frobenius_norm(b)
    assert na > 0 and nb > 0, "cosine undefined for a zero vector"
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


_ST2_B_GLOBAL = (0.3, -0.1, 0.2, 0.4, -0.2, 0.1)


def selftest_2_paramdiff():
    print("\n" + "=" * 70)
    print("SELF-TEST 2 -- parameter-diff norm/cosine (H4)")
    print("=" * 70)
    ok = True

    # (a) Planted-coherent construction: ΔW = 2.5 * b_global exactly.
    dW_a = tuple(2.5 * x for x in _ST2_B_GLOBAL)
    norm_a = frobenius_norm(dW_a)
    cos_a = cosine(dW_a, _ST2_B_GLOBAL)
    print(f"(a) planted-coherent: ||dW||_F={norm_a!r} cos={cos_a!r}")
    a_ok = abs(norm_a - 1.479019945774904) < TOL_NORM and abs(cos_a - 1.0) < 1e-6
    print(f"    expected ||dW||_F=1.479019945774904 (+/-1e-9), cos=1.0 (+/-1e-6)  PASS={a_ok}")
    ok = ok and a_ok

    # (b) Incoherent construction (hand-picked, not a multiple of b_global).
    dW_b = (0.4, 0.4, -0.3, -0.1, 0.35, 0.5)
    norm_b = frobenius_norm(dW_b)
    cos_b = cosine(dW_b, _ST2_B_GLOBAL)
    print(f"(b) incoherent:       ||dW||_F={norm_b!r} cos={cos_b!r}")
    b_ok = (
        abs(norm_b - 0.8902246907382427) < TOL_NORM
        and abs(cos_b - (-0.0759497473859234)) < 1e-6
        and abs(cos_b) < 0.2
    )
    print(f"    expected ||dW||_F=0.8902246907382427 (+/-1e-9), "
          f"cos=-0.0759497473859234 (+/-1e-6), |cos|<0.2  PASS={b_ok}")
    ok = ok and b_ok

    # Comparative reading: construction (a)'s cosine must exceed (b)'s by >= 0.5.
    gap = cos_a - cos_b
    gap_ok = gap >= 0.5
    print(f"    coherence gap = {gap:.6f}, >= 0.5 required  PASS={gap_ok}")
    ok = ok and gap_ok

    return ok, {
        "b_global": list(_ST2_B_GLOBAL),
        "construction_a_planted_coherent": {"dW": list(dW_a), "frobenius_norm": norm_a, "cosine": cos_a},
        "construction_b_incoherent": {"dW": list(dW_b), "frobenius_norm": norm_b, "cosine": cos_b},
        "coherence_gap": gap,
        "expected": {
            "a_norm": 1.479019945774904, "a_cos": 1.0,
            "b_norm": 0.8902246907382427, "b_cos": -0.0759497473859234,
            "min_coherence_gap": 0.5,
        },
    }


# ===========================================================================
# Self-test 3 -- gradient-flow norm telemetry (H3), sec 12.3.4 "Self-test 3"
# ===========================================================================

class GradFlowRecorder:
    """Mirrors sec 12.5's planned instrumentation: a backward hook capturing
    the gradient-norm reaching a target tensor (the real Stage-2 script hooks
    `k_raw`, the post-conv/pre-blend key tensor `apply_frozen_bias_blend`
    consumes; here the hook target is a mock 1x3 "k_raw" stand-in). Tracks how
    many times the hook has fired (call_count) and the captured L2 norms,
    without modifying the gradient (returns None from the hook, matching the
    real probe's read-only telemetry role)."""

    def __init__(self):
        self.captured_norms = []
        self.call_count = 0

    def hook(self, grad):
        self.call_count += 1
        self.captured_norms.append(grad.norm().item())
        return None

    def register(self, tensor):
        return tensor.register_hook(self.hook)


def _run_gradflow_pass(recorder):
    """Closed-form linear construction, sec 12.3.4 self-test 3: mock "k_raw"
    stand-in x=(1.0,-2.0,0.5) (1x3), downstream weight W (3x2, fixed, not a
    Parameter -- matches the real probe's read of an already-existing
    downstream consumer, not something this hook trains), target t=(0.1,-0.2),
    y = x @ W, L = 0.5*sum((y-t)^2). x is derived from a leaf param via an
    identity op (x = x_raw * 1.0) so the hook is registered on a NON-LEAF
    tensor, mirroring the real target (k_raw sits downstream of k_conv1d, not
    a leaf) -- this is the case where a hook could observe a stale/mismatched
    tensor if wired wrong, exactly what item (i) below is checking."""
    import torch
    x_raw = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64, requires_grad=True)
    x = x_raw * 1.0
    handle = recorder.register(x)
    W = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.5, 0.6]], dtype=torch.float64)
    t = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    y = x @ W
    loss = 0.5 * ((y - t) ** 2).sum()
    loss.backward()
    handle.remove()
    return loss.item()


def selftest_3_gradflow():
    print("\n" + "=" * 70)
    print("SELF-TEST 3 -- gradient-flow norm telemetry (H3)")
    print("=" * 70)
    ok = True

    try:
        import torch  # noqa: F401
    except ImportError:
        print("    torch not importable -- self-test 3 CANNOT run (hard FAIL, not skip)")
        return False, {"error": "torch not importable"}

    recorder = GradFlowRecorder()

    # Hand-derived closed form (no autodiff needed for this reference value):
    # y=(-0.85,-0.2), resid=(-0.95,0.0), L=0.45125, dL/dx=resid@W^T=(-0.19,-0.38,0.475),
    # ||dL/dx||_2 = 0.6372793735874401.
    expected_norm = 0.6372793735874401
    expected_loss = 0.45125

    # Run TWO independent forward/backward passes (fresh graph each time) --
    # tests "fires exactly once per forward/backward pair" across multiple
    # pairs, not just that it fires at all on a single one.
    loss1 = _run_gradflow_pass(recorder)
    count_after_1 = recorder.call_count
    norm_after_1 = recorder.captured_norms[-1] if recorder.captured_norms else None

    loss2 = _run_gradflow_pass(recorder)
    count_after_2 = recorder.call_count
    norm_after_2 = recorder.captured_norms[-1] if recorder.captured_norms else None

    print(f"    pass 1: loss={loss1:.6f} call_count={count_after_1} captured_norm={norm_after_1!r}")
    print(f"    pass 2: loss={loss2:.6f} call_count={count_after_2} captured_norm={norm_after_2!r}")

    check_i = (
        norm_after_1 is not None and abs(norm_after_1 - expected_norm) < TOL_GRAD
        and norm_after_2 is not None and abs(norm_after_2 - expected_norm) < TOL_GRAD
    )
    print(f"    (i) captured norm == {expected_norm} (+/-1e-5), both passes  PASS={check_i}")
    ok = ok and check_i

    check_ii = (count_after_1 == 1) and (count_after_2 == 2)
    print(f"    (ii) hook fires exactly once per forward/backward pair "
          f"(count: 1 after pass 1, 2 after pass 2)  PASS={check_ii}")
    ok = ok and check_ii

    check_iii = all(math.isfinite(n) and n != 0.0 for n in recorder.captured_norms)
    print(f"    (iii) captured values finite and nonzero  PASS={check_iii}")
    ok = ok and check_iii

    loss_check = abs(loss1 - expected_loss) < 1e-9 and abs(loss2 - expected_loss) < 1e-9
    print(f"    (bonus) loss == {expected_loss} (+/-1e-9), both passes  PASS={loss_check}")
    ok = ok and loss_check

    return ok, {
        "expected_norm": expected_norm, "expected_loss": expected_loss,
        "pass_1": {"loss": loss1, "call_count": count_after_1, "captured_norm": norm_after_1},
        "pass_2": {"loss": loss2, "call_count": count_after_2, "captured_norm": norm_after_2},
        "all_captured_norms": recorder.captured_norms,
    }


# ===========================================================================
# Driver
# ===========================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ok1, res1 = selftest_1_repeat_excess()
    ok2, res2 = selftest_2_paramdiff()
    ok3, res3 = selftest_3_gradflow()

    all_ok = ok1 and ok2 and ok3

    print("\n" + "=" * 70)
    print(f"STAGE 0.5 GATE -- self-test 1 (repeat_excess): {'PASS' if ok1 else 'FAIL'}")
    print(f"STAGE 0.5 GATE -- self-test 2 (param-diff cosine): {'PASS' if ok2 else 'FAIL'}")
    print(f"STAGE 0.5 GATE -- self-test 3 (gradient-flow norm): {'PASS' if ok3 else 'FAIL'}")
    print(f"STAGE 0.5 GATE -- OVERALL: {'ALL PASSED -- Stage 1 may launch' if all_ok else 'FAILURES PRESENT -- Stage 1 BLOCKED'}")
    print("=" * 70)

    payload = wrap_exploratory({
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 12.3.4 (Stage 0.5 synthetic self-tests)",
        "gate_passed": all_ok,
        "selftest_1_repeat_excess": {"passed": ok1, **res1},
        "selftest_2_paramdiff": {"passed": ok2, **res2},
        "selftest_3_gradflow": {"passed": ok3, **res3},
    })
    out_path = os.path.join(RESULTS_DIR, "mech_stage05_selftests_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
