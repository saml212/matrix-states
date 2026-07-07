"""KEY_ANCHORING_SCALING_DRAFT.md sec 15.2 item 1 blocking gate -- kernel-safety
smoke for chunk_delta_rule at the UNTESTED d_state values 80/96, run to the
wave's OWN registered protocol: the full T in {128, 224, 448} forward+backward
sweep (matching f15_lm_checkpoint.py's original crash-matrix methodology), not
a single-T spot check. T=128 is load-bearing: model_rd.py pads every real BIND
sequence to _MIN_KERNEL_T=128, making it the most common real value -- and it
is where D=16 was flaky and D=32 crashed historically. (A first version of
this smoke tested only T=256 and "passed" D=32 -- attack-round-1 FATAL-2
correctly rejected that as evidence, since T=256 was never in the original
crash matrix. This version supersedes it.)

Writes a committed-artifact JSON (results/smoke_dstate_kernel_result.json on
the box; archived to the repo) so the chain script can mechanically refuse to
launch without a PASSING artifact on file (the draft's own Q1 TODO).

64/128 run as positive controls; 32 is the known-crash reference (allowed to
fail, informative either way). Calls the fla kernel DIRECTLY -- model_rd.py's
_SAFE_D_STATE=(64,128) assert would refuse 80/96 before the kernel is reached,
and stays authoritative for the full model path regardless of this smoke.

Exit 0 iff BOTH 80 and 96 pass forward+backward with finite grads at ALL
three T values. Exit 3 if either fails any T (sec 15 wave dead as designed).
Exit 2 if a positive control (64/128) fails anywhere -- environment problem,
no d_state verdict.
"""
import json
import os
import sys
import traceback

import torch

T_SWEEP = (128, 224, 448)  # sec 15.2 item 1's registered protocol, verbatim
CANDIDATES = (80, 96)
CONTROLS = (64, 128)
CRASH_REFERENCE = 32


def try_cell(d_state: int, T: int, device: str) -> tuple[bool, str]:
    from fla.ops.delta_rule import chunk_delta_rule
    B, H = 2, 1
    try:
        q = torch.randn(B, T, H, d_state, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, d_state, device=device, dtype=torch.bfloat16), p=2, dim=-1
        ).requires_grad_(True)
        v = torch.randn(B, T, H, d_state, device=device, dtype=torch.bfloat16, requires_grad=True)
        beta = torch.rand(B, T, H, device=device, dtype=torch.bfloat16).requires_grad_(True)
        o, S = chunk_delta_rule(q, k, v, beta, output_final_state=True)
        loss = o.float().square().mean() + S.float().square().mean()
        loss.backward()
        for name, t in (("q", q), ("k", k), ("v", v), ("beta", beta)):
            g = t.grad
            if g is None:
                return False, f"no grad for {name}"
            if not torch.isfinite(g.float()).all():
                return False, f"non-finite grad for {name}"
        # silent-corruption guard, not just crash detection: outputs finite too
        if not torch.isfinite(o.float()).all() or not torch.isfinite(S.float()).all():
            return False, "non-finite forward output"
        return True, f"ok (loss={loss.item():.4f})"
    except Exception:
        return False, traceback.format_exc(limit=3).strip().replace("\n", " | ")


def main() -> int:
    device = "cuda"
    grid: dict[str, dict[str, bool]] = {}
    msgs = {}
    for d in CONTROLS + CANDIDATES + (CRASH_REFERENCE,):
        grid[str(d)] = {}
        for T in T_SWEEP:
            ok, msg = try_cell(d, T, device)
            grid[str(d)][str(T)] = ok
            msgs[f"d{d}_T{T}"] = msg
            print(f"d_state={d} T={T}: {'PASS' if ok else 'FAIL'} -- {msg}")

    controls_ok = all(grid[str(d)][str(T)] for d in CONTROLS for T in T_SWEEP)
    candidates_ok = all(grid[str(d)][str(T)] for d in CANDIDATES for T in T_SWEEP)

    if not controls_ok:
        verdict, code = "CONTROL-FAILURE (environment problem, no d_state verdict)", 2
    elif candidates_ok:
        verdict, code = "d_state 80+96 KERNEL-SAFE at full T sweep -- sec 15 gate CLEARED", 0
    else:
        verdict, code = "d_state 80/96 FAIL at >=1 T -- sec 15 wave dead as designed", 3

    artifact = {
        "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.2 item 1 (attack-round-1 FATAL-2 protocol)",
        "t_sweep": list(T_SWEEP),
        "grid_pass": grid,
        "messages": msgs,
        "verdict": verdict,
        "exit_code": code,
        "torch_version": torch.__version__,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "smoke_dstate_kernel_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"VERDICT: {verdict}")
    print(f"wrote {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
