"""smoke_dstate_kernel_t483_probe.py -- KEY_ANCHORING_SCALING_DRAFT.md sec
15.20.4 MAJOR-2 disclosure + sec 15.22 next-step 2's registered K=69/d=96
contingency seed (1733): closes a real kernel-safety-gate gap this RUN
session found while pre-flighting that cell.

T_bind(K) = 7K (grammar_rd.py's own T_bind property, verified sec 15.20.1).
K=69 -> T_bind=483. The ORIGINAL kernel-safety protocol (sec 15.2 item 1)
tested T in {128,224,448} (ceiling 448); the NEW wide-grid supplementary
probe (smoke_dstate_kernel_wide.py, sec 15.20.1) tested T in
{448,504,546,588,630} (floor 504). 483 sits strictly BETWEEN both swept
ranges (448 < 483 < 504) -- NEITHER committed artifact's own t_sweep covers
it. sec 15.20.1's own text discloses this precisely for K=69 ("T_bind=483,
8% beyond 448") as "disclosed, non-dispositive precedent" (seeds
1730/1731/1732 already ran 3/3 clean, zero CUDA-crash reports) -- but per
this program's own repeated "verify, don't assume by analogy" discipline
(the same discipline sec 15.20.1 invoked to justify building
smoke_dstate_kernel_wide.py in the first place, rather than resting on
K=69's own precedent to license K=72/78/84/90), an untested T is not
licensed by proximity to two working ones either. This is a genuinely tiny,
sub-minute, zero-measurable-GPU-h probe -- run once, before launching the
contingency seed 1733 cell itself.

Clones smoke_dstate_kernel_wide.py's own try_cell body VERBATIM (same
shapes, same fresh-subprocess-free in-process harness, same finite-grad AND
finite-forward-output silent-corruption guard) -- only the T-sweep differs
(candidate={483}, control=448, d_state=96 only, matching this program's own
"d=80 needs no new T check" disclosure: d=80 is not implicated by this
contingency cell).

Writes its OWN artifact (results/smoke_dstate_kernel_t483_probe_result.json)
-- does NOT touch or overwrite either existing committed gate artifact
(smoke_dstate_kernel_result.json / smoke_dstate_kernel_wide_result.json),
since this is a narrow, purpose-built supplementary check for one specific
untested T, not a replacement for either registered protocol.

Exit 0 iff d_state=96 passes forward+backward with finite grads+outputs at
T=483 (CANDIDATE) AND T=448 (CONTROL). Exit 3 if the candidate fails
(K=69's own T_bind is kernel-unsafe -- the contingency cell must NOT
launch). Exit 2 if the T=448 CONTROL fails (environment regression, no
verdict about T=483).

Usage: python smoke_dstate_kernel_t483_probe.py
"""
import json
import os
import sys
import traceback

import torch

CANDIDATE_T = 483   # K=69's own T_bind = 7*69 (this contingency cell's actual shape)
CONTROL_T = 448     # ORIGINAL protocol's own known-good ceiling, same control precedent as
                     # smoke_dstate_kernel_wide.py
D_STATE = 96


def try_cell(d_state: int, T: int, device: str) -> tuple[bool, str]:
    """Byte-identical protocol to smoke_dstate_kernel.py / smoke_dstate_kernel_wide.py's own
    try_cell -- forward+backward through chunk_delta_rule DIRECTLY (never model_rd.py's
    _SAFE_D_STATE-gated path), finite-grad AND finite-forward-output checks."""
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
    full_sweep = (CONTROL_T, CANDIDATE_T)   # control FIRST, same precedent as smoke_dstate_kernel_wide.py
    for T in full_sweep:
        ok, msg = try_cell(D_STATE, T, device)
        grid[str(D_STATE)][str(T)] = ok
        msgs[f"d{D_STATE}_T{T}"] = msg
        print(f"d_state={D_STATE} T={T}{' (CONTROL)' if T == CONTROL_T else ' (CANDIDATE, K=69 T_bind)'}: "
              f"{'PASS' if ok else 'FAIL'} -- {msg}")

    control_ok = grid[str(D_STATE)][str(CONTROL_T)]
    candidate_ok = grid[str(D_STATE)][str(CANDIDATE_T)]

    if not control_ok:
        verdict, code = ("CONTROL-FAILURE at d_state=96/T=448 (environment problem or "
                          "regression since the ORIGINAL smoke_dstate_kernel.py run) -- no "
                          "verdict about T=483"), 2
    elif candidate_ok:
        verdict, code = ("d_state=96 KERNEL-SAFE at K=69's own T_bind=483 (the "
                          "smoke_dstate_kernel_wide.py <-> smoke_dstate_kernel.py gap this RUN "
                          "session found and closed) -- contingency seed 1733 cleared to launch"), 0
    else:
        verdict, code = ("d_state=96 FAILS at T=483 (K=69's own T_bind) -- contingency seed 1733 "
                          "MUST NOT launch, K=69/d=96 is kernel-unsafe at this exact shape despite "
                          "seeds 1730/1731/1732's own precedent"), 3

    artifact = {
        "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.20.4 MAJOR-2 / sec 15.22 next-step 2 "
                       "(RUN-session pre-flight for the registered K=69/d=96 contingency seed 1733 "
                       "-- closes the T_bind=483 gap between the ORIGINAL {128,224,448} kernel gate "
                       "and the wide {448,504,546,588,630} supplementary gate, neither of which "
                       "covers 483)",
        "t_sweep": list(full_sweep),
        "candidate_t": CANDIDATE_T,
        "control_t": CONTROL_T,
        "grid_pass": grid,
        "messages": msgs,
        "verdict": verdict,
        "exit_code": code,
        "torch_version": torch.__version__,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                        "smoke_dstate_kernel_t483_probe_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"VERDICT: {verdict}")
    print(f"wrote {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
