"""smoke_dstate_kernel_wide.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.20.1
(Rev 1, 2026-07-07)'s NEW supplementary kernel-safety probe: chunk_delta_rule
at d_state=96, T in {504, 546, 588, 630} -- the wide grid's own new T_bind
values (T_bind(K) = 7K, sec 15.20.1's own registered formula, verified
against grammar_rd.py's T_bind property + its K=8/clause_len=7 regression
check). The ORIGINAL sec 15.2 protocol tested T in {128,224,448} only --
T=448 is that protocol's own registered ceiling, and K=90's own T_bind=630 is
41% beyond it (K=72 already reaches T_bind=504, 12% beyond). This is a NEW,
untested extrapolation direction that sec 15.20.1 explicitly refuses to
license "by proximity to a working one" (this project's own repeated
"verify, don't assume by analogy" discipline, restated there).

Clones smoke_dstate_kernel.py's own protocol exactly (same try_cell shape,
same finite-grad/finite-output silent-corruption guard, same fresh-
subprocess-free-in-process harness) at a NEW T-sweep, d_state=96 ONLY (sec
15.20.1's own disclosure: d=80's escalation cells reuse already-tested T_bind
values, 336/371, both well inside the ORIGINAL 448 ceiling -- no new T check
needed there). T=448 is INCLUDED as an internal, in-file POSITIVE CONTROL
(sec 15.20.1's own "controls d=96 at T=448 known-good" requirement) -- if
d=96/T=448 were to fail HERE, that would mean the environment itself
regressed since the ORIGINAL smoke_dstate_kernel.py run, not a new finding
about the wide grid's own T's.

Writes a committed-artifact JSON (results/smoke_dstate_kernel_wide_result.json
on the box; archived to the repo) so keyanchor_scaling_wide_stage_gate /
keyanchor_scaling_wide_chain.sh can mechanically refuse to launch any d=96-
wide cell without a PASSING artifact on file (mirrors the ORIGINAL gate's own
enforcement shape exactly, sec 15.2/15.18 Q1).

Exit 0 iff d_state=96 passes forward+backward with finite grads+outputs at
ALL of {504,546,588,630} (the CANDIDATE T's) AND at T=448 (the CONTROL).
Exit 3 if any CANDIDATE T fails (sec 15.20 wide grid dead as designed at
d=96). Exit 2 if the T=448 CONTROL fails -- environment problem /
regression, no verdict about the new T's.

Usage: python smoke_dstate_kernel_wide.py
"""
import json
import os
import sys
import traceback

import torch

CANDIDATE_T_SWEEP = (504, 546, 588, 630)   # sec 15.20.1's new T_bind values (K=72,78,84,90)
CONTROL_T = 448                             # sec 15.20.1's own "controls d=96 at T=448 known-good"
D_STATE = 96                                # sec 15.20.1: only d=96 needs this NEW long-T probe


def try_cell(d_state: int, T: int, device: str) -> tuple[bool, str]:
    """Byte-identical protocol to smoke_dstate_kernel.py's own try_cell --
    forward+backward through chunk_delta_rule DIRECTLY (never model_rd.py's
    _SAFE_D_STATE-gated path), finite-grad AND finite-forward-output checks
    (the "silent-corruption guard, not just crash detection" the ORIGINAL
    smoke's own docstring names)."""
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
        if not torch.isfinite(o.float()).all() or not torch.isfinite(S.float()).all():
            return False, "non-finite forward output"
        return True, f"ok (loss={loss.item():.4f})"
    except Exception:
        return False, traceback.format_exc(limit=3).strip().replace("\n", " | ")


def main() -> int:
    device = "cuda"
    grid: dict[str, dict[str, bool]] = {str(D_STATE): {}}
    msgs = {}
    full_sweep = (CONTROL_T,) + CANDIDATE_T_SWEEP   # control FIRST, so a regressed environment
                                                      # is caught before any candidate-T verdict
    for T in full_sweep:
        ok, msg = try_cell(D_STATE, T, device)
        grid[str(D_STATE)][str(T)] = ok
        msgs[f"d{D_STATE}_T{T}"] = msg
        print(f"d_state={D_STATE} T={T}{' (CONTROL)' if T == CONTROL_T else ''}: "
              f"{'PASS' if ok else 'FAIL'} -- {msg}")

    control_ok = grid[str(D_STATE)][str(CONTROL_T)]
    candidates_ok = all(grid[str(D_STATE)][str(T)] for T in CANDIDATE_T_SWEEP)

    if not control_ok:
        verdict, code = ("CONTROL-FAILURE at d_state=96/T=448 (environment problem or "
                          "regression since the ORIGINAL smoke_dstate_kernel.py run) -- no "
                          "verdict about the new wide-grid T's"), 2
    elif candidates_ok:
        verdict, code = ("d_state=96 KERNEL-SAFE at the full wide-grid T sweep "
                          "{504,546,588,630} -- sec 15.20.1 supplementary gate CLEARED"), 0
    else:
        verdict, code = ("d_state=96 FAILS at >=1 wide-grid T -- sec 15.20 d=96-wide cells "
                          "dead as designed"), 3

    artifact = {
        "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.20.1 (Rev 1, NEW supplementary "
                       "long-T kernel-safety probe)",
        "t_sweep": list(full_sweep),
        "candidate_t_sweep": list(CANDIDATE_T_SWEEP),
        "control_t": CONTROL_T,
        "grid_pass": grid,
        "messages": msgs,
        "verdict": verdict,
        "exit_code": code,
        "torch_version": torch.__version__,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                        "smoke_dstate_kernel_wide_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"VERDICT: {verdict}")
    print(f"wrote {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
