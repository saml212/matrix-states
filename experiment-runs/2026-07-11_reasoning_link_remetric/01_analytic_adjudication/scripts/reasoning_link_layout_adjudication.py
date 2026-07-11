"""reasoning_link_layout_adjudication.py -- REASONING-LINK INSTRUMENT
VERIFIER, chain step 1: ANALYTIC ADJUDICATION at a zero-accumulation
(single-write, one-step) config, run against the REAL fla chunk_delta_rule
kernel on the box, to determine EXACTLY which state layout
reasoning_link_probe.py's `squeeze_state_head` assumes vs receives.

WHY THIS EXISTS: reasoning_link_poscontrol.py's real-checkpoint run showed
production `squeeze_state_head` recovering 0/256 while a manually-transposed
S_T recovers 1.0000 -- the exact "§2.26 class" signature (a [K,V]-vs-[V,K]
state-layout transpose bug already found and fixed once before in this
codebase, model_rd.py's `kernel_state_design_layout`, audit round-1
FATAL-0). This script closes the loop with a genuine, hand-computed closed
form (no accumulation, no numerics, a pure convention adjudicator -- mirrors
CAPABILITY_SEPARATION_DESIGN.md sec 2.26's own method) rather than inferring
the defect only from the poscontrol script's 128-token multi-step run.

METHOD: ONE real write at position 0 (k=e0, v=e1, beta=1 -- basis vectors,
so the outer product's four index combinations are maximally distinguishable
by inspection, no cancellation possible), all T-1=127 remaining positions
k=0 (state-neutral by construction under the delta rule -- CLAUDE.md's own
established fact, reused verbatim by reasoning_link_poscontrol.py and
model_rd.py's own docstring), satisfying chunk_delta_rule's _MIN_KERNEL_T=128
floor without perturbing the analytic answer. The PINNED design convention
(model_rd.py's own docstring, deltanet_core.py sec 3.1, apply_state_power's
einsum): S[value_dim, key_dim], so S_1 = beta * v (outer) k ->
S_1[value=1, key=0] = 1.0, every other entry exactly 0. This is THE closed
form; everything below is comparison against a REAL kernel call, never a
re-derivation.

Three things are computed from the SAME real chunk_delta_rule call:
  1. `model_rd.kernel_state_design_layout(k,v,beta)` -- the ALREADY-audited,
     KNOWN-CORRECT reference (does the [K,V]->[V,K] transpose explicitly).
  2. `reasoning_link_probe.squeeze_state_head(final_state)` -- the PRODUCTION
     function under test (imported verbatim, never reimplemented), applied
     to a chunk_delta_rule call built identically to (1)'s own internal call.
  3. The hand-computed closed form in both the design [V,K] and fla-native
     [K,V] layouts (the second is exactly the first's transpose).

Run (box only, GPU 0 or 1 -- coordinator-assigned, never 2-7):
  CUDA_VISIBLE_DEVICES=<idle gpu> /home/nvidia/tdenv/bin/python \
      reasoning_link_layout_adjudication.py 2>&1 | tee layout_adjudication_run.log
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.pop("REASONING_LINK_FORCE_CPU_STUB", None)  # real fla required -- this adjudication is worthless under the CPU stub

import reasoning_link_probe as rlp  # noqa: E402
from model_rd import kernel_state_design_layout  # noqa: E402
from fla.ops.delta_rule import chunk_delta_rule  # noqa: E402

D_STATE = 64          # matches the poscontrol checkpoint's own d_state (representative, not load-bearing)
T = 128                # == model_rd._MIN_KERNEL_T, the kernel's own floor
B = 1


def main() -> dict:
    report: dict = {"d_state": D_STATE, "T": T, "B": B}

    assert rlp.FLA_STUB_INSTALLED is False, "REAL fla required -- FLA_STUB_INSTALLED=True, STOP."
    assert torch.cuda.is_available(), "chunk_delta_rule has no CPU path -- CUDA required."
    device = torch.device("cuda")

    import fla
    report["fla_version"] = getattr(fla, "__version__", "unknown")
    report["torch_version"] = torch.__version__
    report["cuda_device_name"] = torch.cuda.get_device_name(0)

    # --- ONE real write at position 0: k=e0, v=e1, beta=1. All other positions k=0 (state-neutral). ---
    k_full = torch.zeros(B, T, D_STATE, device=device, dtype=torch.float32)
    v_full = torch.zeros(B, T, D_STATE, device=device, dtype=torch.float32)
    beta_full = torch.zeros(B, T, device=device, dtype=torch.float32)
    k_full[0, 0, 0] = 1.0   # k = e0 (basis vector, index 0)
    v_full[0, 0, 1] = 1.0   # v = e1 (basis vector, index 1) -- deliberately != k's index, so
                            # [K,V] vs [V,K] confusion cannot hide behind k==v symmetry
    beta_full[0, 0] = 1.0

    # --- (1) the ALREADY-audited, KNOWN-CORRECT reference, unmodified ---
    S_design = kernel_state_design_layout(k_full, v_full, beta_full)  # (B,d,d), DESIGN [V,K] layout by construction

    # --- (2) the PRODUCTION function under test, on an IDENTICAL real kernel call ---
    k_bf = k_full.unsqueeze(2).to(torch.bfloat16)
    v_bf = v_full.unsqueeze(2).to(torch.bfloat16)
    beta_bf = beta_full.unsqueeze(-1).to(torch.bfloat16)
    _, S_kv = chunk_delta_rule(q=k_bf, k=k_bf, v=v_bf, beta=beta_bf,
                                output_final_state=True, use_qk_l2norm_in_kernel=False)
    S_squeeze = rlp.squeeze_state_head(S_kv)  # (B,d,d) -- PRODUCTION reasoning_link_probe.py function, verbatim

    # --- (3) the hand-computed closed forms (design [V,K] and its transpose, fla-native [K,V]) ---
    closed_form_design = torch.zeros(B, D_STATE, D_STATE, device=device)
    closed_form_design[0, 1, 0] = 1.0  # S[value=1, key=0] = beta * v[1] * k[0] = 1*1*1
    closed_form_fla = closed_form_design.transpose(-1, -2).contiguous()  # S[key=0, value=1] = 1.0

    def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()

    report["S_design_vs_closed_form_design_rel_fro"] = rel_fro(S_design, closed_form_design)
    report["S_design_vs_closed_form_fla_rel_fro"] = rel_fro(S_design, closed_form_fla)
    report["S_squeeze_vs_closed_form_design_rel_fro"] = rel_fro(S_squeeze, closed_form_design)
    report["S_squeeze_vs_closed_form_fla_rel_fro"] = rel_fro(S_squeeze, closed_form_fla)
    report["S_design_vs_S_squeeze_transposed_rel_fro"] = rel_fro(S_design, S_squeeze.transpose(-1, -2))

    report["S_squeeze_raw_entries"] = {
        "[0,0,1] (key=0,value=1 slot)": S_squeeze[0, 0, 1].item(),
        "[0,1,0] (value=1,key=0 slot)": S_squeeze[0, 1, 0].item(),
    }
    report["S_design_raw_entries"] = {
        "[0,0,1] (key=0,value=1 slot)": S_design[0, 0, 1].item(),
        "[0,1,0] (value=1,key=0 slot)": S_design[0, 1, 0].item(),
    }

    # --- verdict: does squeeze_state_head deliver the RAW fla [K,V] layout (missing the
    # [K,V]->[V,K] transpose kernel_state_design_layout applies), or the DESIGN [V,K] layout? ---
    squeeze_matches_fla_native = (report["S_squeeze_vs_closed_form_fla_rel_fro"] < 1e-2
                                   and report["S_squeeze_vs_closed_form_design_rel_fro"] > 0.5)
    squeeze_matches_design = (report["S_squeeze_vs_closed_form_design_rel_fro"] < 1e-2
                               and report["S_squeeze_vs_closed_form_fla_rel_fro"] > 0.5)
    design_matches_design = report["S_design_vs_closed_form_design_rel_fro"] < 1e-2

    report["verdict"] = {
        "kernel_state_design_layout_matches_closed_form_design (sanity: reference is correct)":
            bool(design_matches_design),
        "squeeze_state_head_returns_RAW_fla_[K,V]_layout (the defect)": bool(squeeze_matches_fla_native),
        "squeeze_state_head_returns_DESIGN_[V,K]_layout (no defect)": bool(squeeze_matches_design),
    }

    return report


if __name__ == "__main__":
    print("=" * 78)
    print("  reasoning_link_layout_adjudication -- zero-accumulation closed-form")
    print("  single-write real-kernel adjudication (instrument verifier chain step 1)")
    print("=" * 78)
    rep = main()
    out_path = os.path.join(HERE, "reasoning_link_layout_adjudication_result.json")
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print(f"  wrote {out_path}")
    print("=" * 78)
