"""KEY_ANCHORING_SCALING_DRAFT.md sec 15 blocking gate -- direct kernel-safety
smoke for chunk_delta_rule at the UNTESTED d_state values 80/96 (64/128 run
as positive controls; 32 is the known-crash reference, tested last and
allowed to fail). model_rd.py's _SAFE_D_STATE=(64,128) hard-assert exists
precisely because d=32 crashed BETWEEN two working values -- interpolation
is not evidence, only this smoke is. Calls the fla kernel DIRECTLY (not
through model_rd.py, whose assert would refuse 80/96 before the kernel is
ever reached).

Exit 0 iff BOTH 80 and 96 pass forward+backward with finite grads.
Exit 3 if either fails (the sec 15 wave is then dead as designed).
Exit 2 if a positive control (64/128) fails -- environment problem, result
meaningless, do not read as a d_state verdict.
"""
import sys
import traceback

import torch


def try_dstate(d_state: int, device: str) -> tuple[bool, str]:
    from fla.ops.delta_rule import chunk_delta_rule
    B, H, T = 2, 1, 256
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
        return True, f"ok (loss={loss.item():.4f})"
    except Exception:
        return False, traceback.format_exc(limit=3).strip().replace("\n", " | ")


def main() -> int:
    device = "cuda"
    results = {}
    # positive controls first, then the candidates, then the known-crash reference
    for d in (64, 128, 80, 96, 32):
        ok, msg = try_dstate(d, device)
        results[d] = ok
        print(f"d_state={d}: {'PASS' if ok else 'FAIL'} -- {msg}")
    if not (results[64] and results[128]):
        print("VERDICT: positive control FAILED -- environment problem, no d_state verdict")
        return 2
    if results[80] and results[96]:
        print("VERDICT: d_state 80+96 KERNEL-SAFE -- sec 15 wave may proceed to attack rounds")
        return 0
    print("VERDICT: d_state 80/96 NOT kernel-safe -- sec 15 wave dead as designed")
    return 3


if __name__ == "__main__":
    sys.exit(main())
