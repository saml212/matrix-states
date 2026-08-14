"""Tests for write_supervision_loss.py -- run to completion (CPU, fp32,
torch 2.8.0, no GPU, no box contact). Every negative test below is actually
EXECUTED (CLAUDE.md: "always run the negative unit test that's supposed to
prove the check 'has teeth' to completion -- don't just write it").

Reproduces, at real K=24/d=25 dims, the SAME algebraic properties attack R4
independently re-derived and executed (V1-V5) -- this file re-verifies them
against the ACTUAL shipped module, not the attack's own throwaway scripts.
"""
from __future__ import annotations

import torch

from write_supervision_loss import band2_check, null_directions, write_supervision_loss

torch.manual_seed(0)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_episode(B=64, K=24, d=25, dtype=torch.float32):
    keys_v = torch.randn(B, K, d, dtype=dtype)
    values_v = torch.randn(B, K, d, dtype=dtype)
    # Z_ideal via the SAME construction teacher_force_operator uses
    # (ncr_lm_wave1_runner.py:348-362, pinv min-norm fit), reproduced here
    # verbatim -- this is the pinned runner's own formula, not a
    # reimplementation:
    z_t = torch.linalg.pinv(keys_v) @ values_v          # (B,d,K)@(B,K,d) -> (B,d,d) == Z_ideal^T
    Z_ideal = z_t.transpose(-1, -2)
    return keys_v, values_v, Z_ideal


# ---------------------------------------------------------------------------
# Test group 1: D1.3's zero-set proof, at the ACTUAL K=24/d=25 dims.
# ---------------------------------------------------------------------------
def test_zero_set_exact():
    keys_v, values_v, Z_ideal = make_episode()
    out = write_supervision_loss(Z_ideal, keys_v, values_v, lambda_t=1.0)
    check("V1: L_key(Z_ideal) flat to float noise", out["L_key"].max().item() < 1e-8,
          f"max L_key={out['L_key'].max().item():.3e}")
    check("V2: L_transverse(Z_ideal) flat to float noise", out["L_transverse"].max().item() < 1e-8,
          f"max L_trans={out['L_transverse'].max().item():.3e}")
    check("D1.3 combined: L_write(Z_ideal) ~ 0", out["L_write"].item() < 1e-8,
          f"L_write={out['L_write'].item():.3e}")


def test_zero_set_hole_at_u_nonzero():
    """Step 2 of D1.3: perturbing Z_ideal ALONG the null directions (u!=0)
    keeps L_key at zero but makes L_transverse strictly positive --
    confirms the "25-dim family" is NOT in the zero set once lambda_t>0."""
    keys_v, values_v, Z_ideal = make_episode(B=16)
    W = null_directions(keys_v, K=24)                      # (B,1,25) at K=d-1
    u = torch.randn(16, 25, 1) * 5.0
    Delta = torch.einsum('bpk,bkj->bpj', u, W)              # (B,25,25), u w^T
    Z = Z_ideal + Delta
    out = write_supervision_loss(Z, keys_v, values_v, lambda_t=1.0)
    check("L_key stays ~0 under a pure null-space perturbation", out["L_key"].max().item() < 1e-6,
          f"max L_key={out['L_key'].max().item():.3e}")
    check("L_transverse becomes strictly positive", out["L_transverse"].min().item() > 1e-3,
          f"min L_trans={out['L_transverse'].min().item():.3e}")
    # analytic check: L_transverse should equal ||u||^2 / v_bar2 (D1.3 Step 2, Zw = u since w^Tw=1)
    v_bar2 = values_v.pow(2).sum(-1).mean(-1) + 1e-6
    expected = u.squeeze(-1).pow(2).sum(-1) / v_bar2
    check("L_transverse matches the closed-form ||u||^2/v_bar2 (D1.3 Step 2)",
          torch.allclose(out["L_transverse"], expected, rtol=1e-3, atol=1e-4),
          f"max abs diff={ (out['L_transverse']-expected).abs().max().item():.3e}")


def test_zero_set_hole_at_span_perturbation():
    """The complementary negative test: perturbing Z_ideal WITHIN
    span(keys) (i.e. a direction the per-key residual CAN see) makes
    L_key strictly positive while L_transverse stays exactly zero (V5:
    the two gradients occupy orthogonal subspaces of matrix-space)."""
    keys_v, values_v, Z_ideal = make_episode(B=16)
    # Delta = sum_i a_i (e_i outer k_i) -- lives entirely in span(keys) by
    # construction (every column of Delta is a combination of the k_i's).
    a = torch.randn(16, 25, 24)
    Delta = torch.einsum('bpi,bik->bpk', a, keys_v)          # (B,25,25) = sum_i a[:,:,i] outer keys_v[:,i,:]
    Z = Z_ideal + Delta
    out = write_supervision_loss(Z, keys_v, values_v, lambda_t=1.0)
    check("L_key becomes strictly positive under a span(keys) perturbation",
          out["L_key"].min().item() > 1e-3, f"min L_key={out['L_key'].min().item():.3e}")
    check("L_transverse stays ~0 under a span(keys)-only perturbation (V5 gradient orthogonality)",
          out["L_transverse"].max().item() < 1e-6, f"max L_trans={out['L_transverse'].max().item():.3e}")


# ---------------------------------------------------------------------------
# Test group 2: M11's d-K generalization (K < d-1, multi-dim null space).
# ---------------------------------------------------------------------------
def test_dK_generalization_zero_set():
    B, K, d = 32, 3, 6          # a genuinely multi-dim null space (d-K=3)
    keys_v = torch.randn(B, K, d)
    values_v = torch.randn(B, K, d)
    z_t = torch.linalg.pinv(keys_v) @ values_v
    Z_ideal = z_t.transpose(-1, -2)
    out0 = write_supervision_loss(Z_ideal, keys_v, values_v, lambda_t=1.0, K=K)
    check("M11 (K<d-1): W has shape (B,d-K,d)", tuple(out0["W"].shape) == (B, d - K, d),
          f"got {tuple(out0['W'].shape)}")
    check("M11 (K<d-1): Z_ideal is still the zero of the generalized loss",
          out0["L_write"].item() < 1e-6, f"L_write={out0['L_write'].item():.3e}")

    W = out0["W"]
    check("M11: W's rows are orthonormal (W W^T = I_{d-K})",
          torch.allclose(torch.einsum('bpj,bqj->bpq', W, W), torch.eye(d - K).expand(B, d - K, d - K), atol=1e-4))

    # Perturb along the null space: Delta = U @ W, U: (B,d,d-K)
    U = torch.randn(B, d, d - K) * 3.0
    Delta_null = torch.einsum('bpk,bkj->bpj', U, W)
    Z_null = Z_ideal + Delta_null
    out_null = write_supervision_loss(Z_null, keys_v, values_v, lambda_t=1.0, K=K)
    check("M11: null-space perturbation keeps L_key ~0", out_null["L_key"].max().item() < 1e-5,
          f"max={out_null['L_key'].max().item():.3e}")
    check("M11: null-space perturbation makes L_transverse strictly positive",
          out_null["L_transverse"].min().item() > 1e-3, f"min={out_null['L_transverse'].min().item():.3e}")

    # Perturb within span(keys): Delta = sum_i a_i (e_i outer k_i)
    a = torch.randn(B, d, K)
    Delta_span = torch.einsum('bpi,bik->bpk', a, keys_v)
    Z_span = Z_ideal + Delta_span
    out_span = write_supervision_loss(Z_span, keys_v, values_v, lambda_t=1.0, K=K)
    check("M11: span(keys) perturbation makes L_key strictly positive",
          out_span["L_key"].min().item() > 1e-4, f"min={out_span['L_key'].min().item():.3e}")
    check("M11: span(keys) perturbation keeps L_transverse ~0",
          out_span["L_transverse"].max().item() < 1e-5, f"max={out_span['L_transverse'].max().item():.3e}")


# ---------------------------------------------------------------------------
# Test group 3: D3's detach coverage.
# ---------------------------------------------------------------------------
def test_detach_covers_both_subterms():
    keys_v, values_v, Z_ideal = make_episode(B=8)
    keys_v.requires_grad_(True)
    values_v.requires_grad_(True)
    Z = Z_ideal.clone().requires_grad_(True)
    out = write_supervision_loss(Z, keys_v, values_v, lambda_t=1.0)
    out["L_write"].backward()
    check("D3: Z receives a gradient (the encoder path must stay trainable)", Z.grad is not None)
    check("D3: keys_v receives NO gradient (L_key's route closed)", keys_v.grad is None)
    check("D3: values_v receives NO gradient (L_key's route closed)", values_v.grad is None)
    check("D3: W's SVD route is also detached (no grad_fn survives on ZW's inputs)",
          not out["W"].requires_grad, f"W.requires_grad={out['W'].requires_grad}")


# ---------------------------------------------------------------------------
# Test group 4: F1's repair -- Band 2 scale invariance + the counter-example.
# ---------------------------------------------------------------------------
def test_band2_scale_invariance_case1():
    """F1's CASE 1 (the report's own executed table): Z = c . Z_ideal for
    c in {0.01, 0.1, 1, 1.5, 10, 100} must ALL pass the repaired Band 2's
    L_key(c*.Z) gate -- the exact counter-example the un-repaired band
    failed on (c=1.5 -> FAIL, per the attack's own table)."""
    keys_v, values_v, Z_ideal = make_episode(B=32)
    for c in (0.01, 0.1, 1.0, 1.5, 10.0, 100.0):
        Z = c * Z_ideal
        b2 = band2_check(Z, keys_v, values_v)
        check(f"F1 repaired Band2: c={c} passes L_key(c*.Z) gate (scale-invariant)",
              (b2["L_key_cstar"] <= 3e-4).all().item(),
              f"max L_key_cstar={b2['L_key_cstar'].max().item():.3e}")
        # The UN-repaired (raw, bare) gate is exactly what F1 showed fails at c=1.5/0.01/etc:
        raw_would_fail = not (b2["L_key_raw"] <= 3e-4).all().item()
        if c != 1.0:
            check(f"F1 regression check: the RAW (un-repaired) gate DOES fail at c={c} (confirms the bug existed)",
                  raw_would_fail, f"L_key_raw max={b2['L_key_raw'].max().item():.3e}")


def test_band2_ratio_catches_collapsed_operator():
    """F1's diagnosed permissive-direction failure: an operator with a
    large transverse component but a TINY global scale used to pass the
    old bare ||Zw||<=3 gate "at chance" (the report's own case: ||Z||_F~1,
    ||Zw||=0.20 -- passes bare 3.0 comfortably, reads at chance). The
    repaired RATIO gate must fail this operator regardless of scale c."""
    B, K, d = 32, 24, 25
    keys_v, values_v, Z_ideal = make_episode(B=B, K=K, d=d)
    W = null_directions(keys_v, K=K)
    # An operator whose transverse component DOMINATES its own total norm
    # (u's norm fixed, independent of c) -- this is the shape of the
    # report's own case ("||Z||_F~1, ||Zw||=0.20 -- passes the bare 3.0
    # comfortably, reads at chance"): at small c the BARE ||Zw||<=3.0 gate
    # is fooled (c shrinks the absolute transverse norm below 3.0), while
    # the RATIO ||ZW||/||Z||_F does not change with c at all.
    u = torch.randn(B, d, 1) * 50.0
    Delta = torch.einsum('bpk,bkj->bpj', u, W)
    Z_bad_direction = Z_ideal + Delta
    for c in (1.0, 0.1, 0.01):    # small c mimics a "collapsed"-norm operator
        Z = c * Z_bad_direction
        b2 = band2_check(Z, keys_v, values_v)
        old_bare_would_pass = (b2["Zw_fro"] <= 3.0).float().mean().item()
        check(f"F1 repair: ratio gate correctly FAILS the bad-direction operator at c={c} "
              f"(closes the collapsed-operator-passes-at-chance loophole)",
              (b2["Zw_ratio"] > 0.12).all().item(),
              f"Zw_ratio median={b2['Zw_ratio'].median().item():.4f}, "
              f"old bare ||Zw||<=3 would have PASSED a {old_bare_would_pass*100:.0f}% fraction at this c")


def test_band2_thresholds_never_invented():
    """Sanity: the module's public thresholds are EXACTLY the report's own
    numbers (3e-4, 0.12) -- no silently-different default."""
    import inspect
    sig = inspect.signature(band2_check)
    check("l_key_bound default is exactly 3e-4", sig.parameters["l_key_bound"].default == 3e-4)
    check("zw_ratio_bound default is exactly 0.12", sig.parameters["zw_ratio_bound"].default == 0.12)


if __name__ == "__main__":
    test_zero_set_exact()
    test_zero_set_hole_at_u_nonzero()
    test_zero_set_hole_at_span_perturbation()
    test_dK_generalization_zero_set()
    test_detach_covers_both_subterms()
    test_band2_scale_invariance_case1()
    test_band2_ratio_catches_collapsed_operator()
    test_band2_thresholds_never_invented()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")
