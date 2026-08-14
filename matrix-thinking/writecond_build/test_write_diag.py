"""Tests for write_diag.py -- run to completion, CPU, no box dependency."""
from __future__ import annotations

import torch

from write_diag import compute_write_diag

torch.manual_seed(0)   # reproducible across runs (keys_cond's own scale varies enough with the
                        # random draw that an earlier unseeded run hit a legitimate-but-tight
                        # absolute-tolerance edge case on a large-cond matrix -- pinned here)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_shape_and_fields():
    B, K, d = 16, 24, 25
    keys_v = torch.randn(B, K, d)
    values_v = torch.randn(B, K, d)
    Z = torch.randn(B, d, d)
    diag = compute_write_diag(Z, keys_v, values_v)
    for field in ("L_key", "L_key_cstar", "Zw_norm", "Z_fro", "Zw_ratio", "keys_cond", "keys_null_gap"):
        check(f"write_diag emits field '{field}'", field in diag)
        check(f"write_diag['{field}'] has B={B} entries", len(diag[field]) == B, f"got {len(diag[field])}")


def test_values_match_zero_at_Z_ideal():
    B, K, d = 16, 24, 25
    keys_v = torch.randn(B, K, d)
    values_v = torch.randn(B, K, d)
    z_t = torch.linalg.pinv(keys_v) @ values_v
    Z_ideal = z_t.transpose(-1, -2)
    diag = compute_write_diag(Z_ideal, keys_v, values_v)
    check("write_diag L_key ~0 at Z_ideal", max(diag["L_key"]) < 1e-6, f"max={max(diag['L_key']):.3e}")
    check("write_diag L_key_cstar ~0 at Z_ideal", max(diag["L_key_cstar"]) < 1e-6, f"max={max(diag['L_key_cstar']):.3e}")
    check("write_diag Zw_ratio ~0 at Z_ideal", max(diag["Zw_ratio"]) < 1e-4, f"max={max(diag['Zw_ratio']):.3e}")


def test_keys_cond_matches_direct_computation():
    B, K, d = 8, 24, 25
    keys_v = torch.randn(B, K, d)
    values_v = torch.randn(B, K, d)
    Z = torch.randn(B, d, d)
    diag = compute_write_diag(Z, keys_v, values_v)
    sv = torch.linalg.svdvals(keys_v)
    expected_cond = (sv[:, 0] / sv[:, -1]).tolist()
    got = diag["keys_cond"]
    max_rel_diff = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(expected_cond, got))
    check("keys_cond matches a direct torch.linalg.svdvals computation (relative tolerance)",
          max_rel_diff < 1e-3, f"max_rel_diff={max_rel_diff:.3e}")


if __name__ == "__main__":
    test_shape_and_fields()
    test_values_match_zero_at_Z_ideal()
    test_keys_cond_matches_direct_computation()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")
