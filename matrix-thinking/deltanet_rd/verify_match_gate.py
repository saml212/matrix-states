"""verify_match_gate.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.7 gate 6, the
make-or-break MATCH-GATE. Structured as TWO INDEPENDENT passes that must
agree (CLAUDE.md: "the implementer does not review their own work" +
"multiple independent adversarial audit rounds catch different bugs"):

  Pass 1 (real instantiation) -- builds the ACTUAL `DeltaNetLM`/`AblationLM`/
  `TransformerLM` nn.Modules at the rung-1 config, sums real `numel()`
  param counts (mirrors `lm_rd_rung_configs.verify_param_count`'s own
  precedent), runs a REAL forward pass with `return_states=True` to MEASURE
  the contender's total-across-layers state bytes (not assumed from a
  formula), and derives `cap_length(M)` from that measured figure.

  Pass 2 (independent closed-form re-derivation) -- computes every one of
  the SAME quantities from CLOSED-FORM formulas over the raw rung-1 config
  numbers alone (vocab/d_model/d_state/n_layers/conv_size), NEVER
  instantiating a real nn.Module and NEVER calling any Pass-1 function --
  a genuinely different code path, not a relabeled call to the same helper.
  Also independently reimplements the sink+FIFO mask via a plain nested
  Python loop (never the tensor-broadcast construction
  `transformer_baseline_rd.sink_fifo_mask` uses), and cross-checks it.

Checks asserted (hard-fail, full stop, on ANY mismatch or tolerance
breach -- sec 1.7 gate 6's own "disagreement between passes is a hard
launch-block"):
  1. Pass 1 and Pass 2 param counts agree EXACTLY (both are exact integer
     architecture counts of the SAME config -- no tolerance needed between
     the passes themselves, only between ARCHITECTURES).
  2. Ablation param count within <=1% of the contender's (sec 1.3's table).
  3. Transformer training-FLOPs/token within <=5% of the contender's,
     AT THE PINNED n_layers_transformer=2 (R3-F3).
  4. The contender's REAL measured total-across-all-layers state bytes
     equals the pinned 32,768 (rung-1, fp32 -- M2).
  5. cap_length(M) for M in {1,2,4,8,16,32} matches EXACTLY between passes,
     and matches the design doc's own worked table
     ({1:8,2:16,4:32,8:64,16:128,32:256} at n_layers=2/d_model=256/fp32).
  6. The floor-eligibility rule: M=1 is EXCLUDED (cap_length=8 < the LIVE
     `query_len+clause_len` floor from `grammar_rd.DeltaNetRDTaskConfig`,
     never hardcoded to 13 -- round-4's own "re-verify against the LIVE
     assert, not the doc's claim" discipline); M in {2,4,8,16,32} are all
     ELIGIBLE.
  7. The sink+FIFO mask's independent nested-loop reimplementation agrees
     element-wise with `transformer_baseline_rd.sink_fifo_mask`'s own
     tensor-broadcast construction.

Run: python verify_match_gate.py --smoke   (identical to the default run --
this script's own real invocation IS its smoke gate, mirroring
lm_rd_rung_configs.py's own convention: no separate toy-shape smoke exists
because the REAL rung-1 config is already CPU-cheap to construct/count).
Exit code 0 = MATCH-GATE PASSED (both signoff-relevant checks clean).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_fla_stub_rd import ensure_fla_stub

_STUB_INSTALLED = ensure_fla_stub()
from lm_pretrain_rd import DeltaNetLM                      # noqa: E402
from ablation_mixer_rd import AblationLM                    # noqa: E402
from transformer_baseline_rd import (                       # noqa: E402
    TransformerLM, cap_length_tokens, sink_fifo_mask,
)
from grammar_rd import DeltaNetRDTaskConfig                 # noqa: E402

VOCAB_SIZE = 50257
RUNG1_CFG = {"d_model": 256, "d_state": 64, "n_layers": 2, "conv_size": 4, "ffn_mult": 4}
TRANSFORMER_RUNG1_CFG = {"d_model": 256, "n_layers": 2, "n_heads": 4, "ffn_mult": 4}
PARAM_TOLERANCE_ABLATION = 0.01
FLOP_TOLERANCE_TRANSFORMER = 0.05
M_GRID = (1, 2, 4, 8, 16, 32)
K_SINK = 4
BYTES_PER_ELT_FP32 = 4


def floor_min_tokens() -> int:
    """query_len + one bind clause (clause_len), LIVE from
    `grammar_rd.DeltaNetRDTaskConfig` at the registered conv_size -- NEVER
    hardcoded to 13 (round-4's own re-verification discipline: "floor 13
    re-verified against grammar_rd.py:552's LIVE smoke assert, not the
    doc's claim")."""
    cfg = DeltaNetRDTaskConfig(K=32, conv_size=RUNG1_CFG["conv_size"])
    return cfg.query_len + cfg.clause_len


# ---------------------------------------------------------------------------
# PASS 1 -- real instantiation + a real measured forward pass
# ---------------------------------------------------------------------------

def pass1_build_models():
    contender = DeltaNetLM(VOCAB_SIZE, **RUNG1_CFG)
    ablation = AblationLM(VOCAB_SIZE, d_model=RUNG1_CFG["d_model"], d_state=RUNG1_CFG["d_state"],
                           n_layers=RUNG1_CFG["n_layers"], conv_size=RUNG1_CFG["conv_size"],
                           ffn_mult=RUNG1_CFG["ffn_mult"])
    transformer = TransformerLM(VOCAB_SIZE, **TRANSFORMER_RUNG1_CFG)
    return contender, ablation, transformer


def pass1_flops_per_token(n_total_params: int, d_model: int) -> int:
    """sec 4.2's head-dominated method: FLOPs/token = 6*N_body + 6*d_model*V."""
    n_body = n_total_params - d_model * VOCAB_SIZE
    return 6 * n_body + 6 * d_model * VOCAB_SIZE


def pass1_measured_state_bytes(contender_model: DeltaNetLM) -> int:
    """REAL measured TOTAL-across-all-layers state bytes from an ACTUAL
    forward pass (not assumed from a formula), at the pinned fp32
    accounting dtype (lm_pretrain_rd.py:986's own persisted dtype, M2)."""
    x = torch.randint(0, VOCAB_SIZE, (1, 128))
    with torch.no_grad():
        _, final_states = contender_model(x, return_states=True)
    total = 0
    for s in final_states:
        assert s.dtype == torch.float32, f"contender state dtype {s.dtype} != float32 (M2 pin)"
        total += (s.numel() // s.shape[0]) * s.element_size()   # per-batch-item bytes, B factored out
    return total


def run_pass1() -> dict:
    contender, ablation, transformer = pass1_build_models()
    n_contender = sum(p.numel() for p in contender.parameters())
    n_ablation = sum(p.numel() for p in ablation.parameters())
    n_transformer = sum(p.numel() for p in transformer.parameters())

    flops_contender = pass1_flops_per_token(n_contender, RUNG1_CFG["d_model"])
    flops_transformer = pass1_flops_per_token(n_transformer, TRANSFORMER_RUNG1_CFG["d_model"])
    state_bytes = pass1_measured_state_bytes(contender)
    cap_table = {m: cap_length_tokens(m, TRANSFORMER_RUNG1_CFG["n_layers"],
                                       TRANSFORMER_RUNG1_CFG["d_model"],
                                       contender_state_bytes=state_bytes) for m in M_GRID}
    floor = floor_min_tokens()
    return {
        "n_contender": n_contender, "n_ablation": n_ablation, "n_transformer": n_transformer,
        "flops_contender": flops_contender, "flops_transformer": flops_transformer,
        "measured_state_bytes": state_bytes, "cap_length_table": cap_table, "floor": floor,
        "param_rel_err": abs(n_ablation - n_contender) / n_contender,
        "flop_rel_err": abs(flops_transformer - flops_contender) / flops_contender,
    }


# ---------------------------------------------------------------------------
# PASS 2 -- independent closed-form re-derivation (no nn.Module instantiation)
# ---------------------------------------------------------------------------

def pass2_contender_params_formula(d_model: int, d_state: int, n_layers: int, conv_size: int,
                                    vocab_size: int) -> int:
    """norm1+norm2(2*d_model) + q/k/v/o_proj(4*d_model*d_state) + b_proj(d_model) +
    3 short-convs(3*d_state*conv_size) + o_norm(d_state) + ffn(8*d_model^2, mult=4), per layer,
    plus the tied embed(vocab*d_model) and norm_f(d_model)."""
    per_layer = (2 * d_model + d_model + 4 * d_model * d_state + 3 * d_state * conv_size
                 + d_state + 8 * d_model ** 2)
    return vocab_size * d_model + n_layers * per_layer + d_model


def pass2_ablation_params_formula(d_model: int, d_state: int, n_layers: int, conv_size: int,
                                   vocab_size: int) -> int:
    """norm1+norm2(2*d_model) + q/v/g/o_proj(4*d_model*d_state) + 2 short-convs(2*d_state*conv_size)
    + o_norm(d_state) + ffn(8*d_model^2), per layer, plus embed + norm_f."""
    per_layer = 2 * d_model + 4 * d_model * d_state + 2 * d_state * conv_size + d_state + 8 * d_model ** 2
    return vocab_size * d_model + n_layers * per_layer + d_model


def pass2_transformer_params_formula(d_model: int, n_layers: int, vocab_size: int) -> int:
    """norm1+norm2(2*d_model) + q/k/v/o_proj(4*d_model^2) + ffn(8*d_model^2), per layer, plus
    embed + norm_f."""
    per_layer = 2 * d_model + 4 * d_model ** 2 + 8 * d_model ** 2
    return vocab_size * d_model + n_layers * per_layer + d_model


def pass2_flops_per_token(n_total_params: int, d_model: int) -> int:
    n_body = n_total_params - d_model * VOCAB_SIZE
    return 6 * n_body + 6 * d_model * VOCAB_SIZE


def pass2_state_bytes_formula(n_layers: int, d_state: int, bytes_per_elt: int = BYTES_PER_ELT_FP32) -> int:
    return n_layers * d_state * d_state * bytes_per_elt


def pass2_cap_length_formula(m: int, n_layers_t: int, d_model_t: int, state_bytes: int,
                              bytes_per_elt: int = BYTES_PER_ELT_FP32) -> int:
    num, den = m * state_bytes, 2 * n_layers_t * d_model_t * bytes_per_elt
    assert num % den == 0, f"cap_length(M={m}) not an exact integer -- {num}/{den}"
    return num // den


def pass2_sink_fifo_mask_independent(seq_len: int, cap_length: int, k_sink: int) -> list:
    """Plain nested Python loop -- deliberately NOT the tensor-broadcast
    construction `transformer_baseline_rd.sink_fifo_mask` uses (a
    genuinely different code path, per gate 6's own "not something
    quietly more or less sophisticated" cross-check requirement)."""
    fifo_window = cap_length - k_sink
    return [[(j <= i) and (j < k_sink or j > i - fifo_window) for j in range(seq_len)]
            for i in range(seq_len)]


def run_pass2() -> dict:
    n_contender = pass2_contender_params_formula(RUNG1_CFG["d_model"], RUNG1_CFG["d_state"],
                                                  RUNG1_CFG["n_layers"], RUNG1_CFG["conv_size"], VOCAB_SIZE)
    n_ablation = pass2_ablation_params_formula(RUNG1_CFG["d_model"], RUNG1_CFG["d_state"],
                                                RUNG1_CFG["n_layers"], RUNG1_CFG["conv_size"], VOCAB_SIZE)
    n_transformer = pass2_transformer_params_formula(TRANSFORMER_RUNG1_CFG["d_model"],
                                                      TRANSFORMER_RUNG1_CFG["n_layers"], VOCAB_SIZE)
    flops_contender = pass2_flops_per_token(n_contender, RUNG1_CFG["d_model"])
    flops_transformer = pass2_flops_per_token(n_transformer, TRANSFORMER_RUNG1_CFG["d_model"])
    state_bytes = pass2_state_bytes_formula(RUNG1_CFG["n_layers"], RUNG1_CFG["d_state"])
    cap_table = {m: pass2_cap_length_formula(m, TRANSFORMER_RUNG1_CFG["n_layers"],
                                              TRANSFORMER_RUNG1_CFG["d_model"], state_bytes) for m in M_GRID}
    floor = floor_min_tokens()
    return {
        "n_contender": n_contender, "n_ablation": n_ablation, "n_transformer": n_transformer,
        "flops_contender": flops_contender, "flops_transformer": flops_transformer,
        "measured_state_bytes": state_bytes, "cap_length_table": cap_table, "floor": floor,
        "param_rel_err": abs(n_ablation - n_contender) / n_contender,
        "flop_rel_err": abs(flops_transformer - flops_contender) / flops_contender,
    }


# ---------------------------------------------------------------------------
# Gate: cross-check the two passes + assert every tolerance/table
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "", sink: list[str] | None = None) -> None:
    """sink=None (default) reports to the module-global FAILURES (the real gate's own record);
    a caller-supplied list (the negative test below) isolates a deliberately-corrupted check from
    that global record so the negative test cannot itself trip the real gate's exit code."""
    target = FAILURES if sink is None else sink
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        target.append(item)


def _check_passes_agree(p1: dict, p2: dict, sink: list[str] | None = None) -> None:
    for key in ("n_contender", "n_ablation", "n_transformer", "flops_contender",
                "flops_transformer", "measured_state_bytes", "floor"):
        ok = p1[key] == p2[key]
        _report(f"agreement: {key} (Pass1={p1[key]:,} vs Pass2={p2[key]:,})", ok, sink=sink)
    ok = p1["cap_length_table"] == p2["cap_length_table"]
    _report(f"agreement: cap_length_table (Pass1={p1['cap_length_table']} vs "
            f"Pass2={p2['cap_length_table']})", ok, sink=sink)


def _check_tolerances(p1: dict, sink: list[str] | None = None) -> None:
    _report(f"param match: ablation vs contender <= {PARAM_TOLERANCE_ABLATION:.0%}",
            p1["param_rel_err"] <= PARAM_TOLERANCE_ABLATION, f"rel_err={p1['param_rel_err']:.6%}",
            sink=sink)
    _report(f"FLOP match: transformer (n_layers=2, R3-F3 pin) vs contender <= "
            f"{FLOP_TOLERANCE_TRANSFORMER:.0%}", p1["flop_rel_err"] <= FLOP_TOLERANCE_TRANSFORMER,
            f"rel_err={p1['flop_rel_err']:.6%}", sink=sink)
    _report("contender total state bytes == 32,768 (rung-1, fp32, M2)",
            p1["measured_state_bytes"] == 32_768, f"measured={p1['measured_state_bytes']:,}",
            sink=sink)


def _check_cap_length_table(p1: dict) -> None:
    expected = {1: 8, 2: 16, 4: 32, 8: 64, 16: 128, 32: 256}
    _report("cap_length(M) table matches sec 1.4.2's own worked table exactly",
            p1["cap_length_table"] == expected, f"got={p1['cap_length_table']}")
    eligible = sorted(m for m, cl in p1["cap_length_table"].items() if cl >= p1["floor"])
    excluded = sorted(set(M_GRID) - set(eligible))
    _report("floor-eligibility: eligible M = {2,4,8,16,32}, excluded M = {1} (R3-F3)",
            eligible == [2, 4, 8, 16, 32] and excluded == [1],
            f"eligible={eligible} excluded={excluded} floor={p1['floor']}")


def _check_sink_fifo_independent_agreement() -> None:
    T, cap_length, k_sink = 20, 10, K_SINK
    ref = sink_fifo_mask(T, cap_length=cap_length, k_sink=k_sink).tolist()
    independent = pass2_sink_fifo_mask_independent(T, cap_length, k_sink)
    ok = ref == independent
    _report("sink+FIFO mask: independent nested-loop reimplementation agrees element-wise with "
            "transformer_baseline_rd.sink_fifo_mask", ok)


def negative_test_gate_has_teeth() -> bool:
    """CLAUDE.md's own structural-check discipline: "run the negative unit
    test that's supposed to prove the check has teeth to completion" --
    proves this gate's own comparison/tolerance logic ACTUALLY detects a
    real mismatch, on a corrupted copy, isolated from the module-global
    FAILURES record (via the `sink=` param) so this test cannot itself
    trip the real gate's own exit code."""
    sink: list[str] = []
    p1, p2 = run_pass1(), run_pass2()

    # (a) a corrupted Pass-2 param count must be DETECTED by the agreement check.
    p2_corrupt = dict(p2)
    p2_corrupt["n_ablation"] = p2["n_ablation"] + 1
    _check_passes_agree(p1, p2_corrupt, sink=sink)
    agreement_check_has_teeth = len(sink) > 0

    # (b) an artificially inflated FLOP mismatch must be DETECTED by the tolerance check.
    sink2: list[str] = []
    p1_corrupt = dict(p1)
    p1_corrupt["flop_rel_err"] = 0.50   # 50% >> the 5% tolerance
    _check_tolerances(p1_corrupt, sink=sink2)
    tolerance_check_has_teeth = len(sink2) > 0

    ok = agreement_check_has_teeth and tolerance_check_has_teeth
    print(f"[negative test: gate has teeth] {'PASS' if ok else 'FAIL'} -- "
          f"agreement_check_detected_corruption={agreement_check_has_teeth} "
          f"tolerance_check_detected_corruption={tolerance_check_has_teeth}", flush=True)
    return ok


def run_match_gate() -> bool:
    p1, p2 = run_pass1(), run_pass2()
    print("=" * 78)
    print("Pass 1 (real instantiation):", {k: v for k, v in p1.items() if k != "cap_length_table"})
    print("Pass 2 (closed-form re-derivation):",
          {k: v for k, v in p2.items() if k != "cap_length_table"})
    print("=" * 78)
    _check_passes_agree(p1, p2)
    _check_tolerances(p1)
    _check_cap_length_table(p1)
    _check_sink_fifo_independent_agreement()
    return not FAILURES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="no-op alias -- this script's real "
                     "invocation IS the gate 6 check, mirroring lm_rd_rung_configs.py's convention.")
    ap.parse_args()
    print(f"fla stub installed (real fla absent or forced): {_STUB_INSTALLED}")
    teeth_ok = negative_test_gate_has_teeth()
    print("=" * 78)
    ok = run_match_gate()
    print("=" * 78)
    if not teeth_ok:
        print("MATCH-GATE: negative test FAILED -- the gate's own comparison/tolerance logic did "
              "NOT detect a deliberately corrupted input; the gate cannot be trusted until this is "
              "fixed (CLAUDE.md's own structural-check discipline).", file=sys.stderr)
        return 1
    if not ok:
        print(f"MATCH-GATE: {len(FAILURES)} FAILURE(S), HARD LAUNCH-BLOCK: {FAILURES}", file=sys.stderr)
        return 1
    print("MATCH-GATE: PASSED (params/FLOPs/state-bytes/cap_length table/floor-eligibility/"
          "sink+FIFO mask all agree across both independent passes, within tolerance; negative "
          "test confirms the gate has teeth)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
