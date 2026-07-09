"""CAPABILITY_SEPARATION_DESIGN.md S1.4 -- the canonical REAL-MATRIX group
family for the production task grammar (GroupWordEncoder training targets),
built from the ALREADY-VERIFIED reference representations in
verify_option_a_readout.py (S1.3) and the A6 generator-construction
primitives coverage_calibration.py already uses for its own centralizer
check (S1.4.1 step 4's `_get_real_generator_pairs`) -- imported directly,
per S1.11's "functions imported directly into the eval harness, not
re-implemented" discipline. No group theory (Rodrigues' formula, icosahedron
axis geometry, hyperplane bases, A5 generator-class verification) is
re-derived here; this module only assembles S1.4's EXACT symmetric
generating sets (matching the design doc's table sizes: S3=3, S4=4, A5=4,
S5=3, A6=4) and provides thin numpy/torch word-product helpers.

Distinct from coverage_calibration.py's GROUPS dict: that module's
generators are PERMUTATION stand-ins used only for Monte Carlo bar
calibration (abstract Cayley-graph mixing statistics, isomorphism-invariant
to realization choice per S1.3.4's A5 isomorphism proof). This module's
generators are the REAL rho_G matrices the production model is trained
against -- the two must never be confused, and are kept in separate files
on purpose.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_option_a_readout as voa
import coverage_calibration as cc  # only for GROUPS["A6"]["gens"] (permutation pair labels)

GROUP_NAMES = ["S3", "S4", "A5", "S5", "A6"]

D_MIN = {"S3": 2, "S4": 3, "A5": 3, "S5": 4, "A6": 5}
ORDER = {"S3": 6, "S4": 24, "A5": 60, "S5": 120, "A6": 360}
SOLVABLE = {"S3": True, "S4": True, "A5": False, "S5": False, "A6": False}
D_STATE = {name: D_MIN[name] + 2 for name in GROUP_NAMES}   # S1.4's uniform-margin rule

_RESULTS_CACHE = None


def group_seed_salt(name: str) -> int:
    """S1.22 BA-F3 fix: a deterministic, per-group integer salt for the
    eval/train word seed derivation. Two groups sharing BOTH |generating
    set| and d_state (S4, A5 -- both d_min=3, so both |gens|=4, d_state=5)
    otherwise draw BYTE-IDENTICAL generator-index sequences from the SAME
    nominal (base_seed, offset) -- an undisclosed second correlation channel
    beyond the adjudicated shared-init-weight one (S1.4.2.1's Welch-unpaired
    justification covers ONLY the init-weight channel, not this one).
    hashlib (not Python's process-randomized str `hash()`) so the salt is
    stable across processes/runs; callers add this to an existing seed, so
    determinism per (group, seed, N) is preserved exactly."""
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) % 10_000_000


def _get_results() -> dict:
    """Run verify_option_a_readout.py's Part 1 ONCE per process (deterministic
    given its own pinned RNG_SEED=20260709) and cache. stdout is captured
    (already shown verbatim in the design doc S1.3.1) to keep this module's
    own output clean."""
    global _RESULTS_CACHE
    if _RESULTS_CACHE is None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _RESULTS_CACHE = voa.part1_reference_representations()
    return _RESULTS_CACHE


def generating_set(name: str) -> list[np.ndarray]:
    """S1.4's exact symmetric generating set (list of d_min x d_min real
    orthogonal matrices), built from verify_option_a_readout.py's pinned
    generators -- sizes match the design table exactly (S3=3, S4=4, A5=4,
    S5=3, A6=4). Every generator is orthogonal (S1.3.1, verified), so
    g^-1 == g.T exactly (no separate inverse computation needed)."""
    results = _get_results()
    if name == "S3":
        r = results["S3"]["generators"]["r"]
        s = results["S3"]["generators"]["s"]
        return [r, r.T, s]                       # {r, r^-1, s}, size 3
    if name == "S4":
        r = results["S4"]["generators"]["r"]
        s = results["S4"]["generators"]["s"]
        return [r, r.T, s, s.T]                  # {r, r^-1, s, s^-1}, size 4
    if name == "A5":
        g5 = results["A5"]["generators"]["g5"]
        g3 = results["A5"]["generators"]["g3"]
        return [g5, g5.T, g3, g3.T]               # {g5, g5^-1, g3, g3^-1}, size 4
    if name == "S5":
        t = results["S5"]["generators"]["transposition_01"]
        c = results["S5"]["generators"]["5cycle_01234"]
        return [t, c, c.T]                        # {t, c, c^-1}, size 3
    if name == "A6":
        B6 = voa.hyperplane_basis(6)
        a6_gens = cc.GROUPS["A6"]["gens"]          # [a, a^-1, b, b^-1] permutation tuples
        c3 = B6.T @ voa.perm_matrix(6, a6_gens[0]) @ B6   # 3-cycle (123)
        c5 = B6.T @ voa.perm_matrix(6, a6_gens[2]) @ B6   # 5-cycle (23456)
        return [c3, c3.T, c5, c5.T]                # {(123), (123)^-1, (23456), (23456)^-1}, size 4
    raise ValueError(f"unknown group {name!r}")


TARGET_PADDING_MODES = ("eye", "zero")


def rho_G_embedded(rho: np.ndarray, d_state: int, target_padding: str = "eye") -> np.ndarray:
    """Option A (S1.4): the block-embedded target rho_G(g) (+) pad --
    block-diagonal, `pad` on the ambient complement. Fixed, non-learned.

    `target_padding` (S1.33 M3 FIX WAVE flag -- CAPABILITY_SEPARATION_DESIGN.md
    S1.33, groups.py:157-158's D-AMB diagnosis): default UNCHANGED "eye" so
    NOTHING retroactively shifts for the already-harvested 58-cell sweep.
    "eye" (original) pads the ambient d_state-d_min complement with I --
    the as-built target then has rank d_state with EVERY singular value
    equal to 1, so a rank-k-capped arm's best achievable direct cosine is
    exactly sqrt(k/d_state) (the "ambient-identity capacity tax" that
    voided every base-sweep M3 force-rank arm: 37/39 cells landed within
    0.07 of that ceiling, not evidence of failed recruitment). "zero"
    (S1.33 fix variant A) pads with 0 instead -- rank(target) = d_min
    EXACTLY (rho is always a real orthogonal representation, hence always
    full rank d_min), removing the tax entirely."""
    assert target_padding in TARGET_PADDING_MODES, \
        f"target_padding={target_padding!r} not in {TARGET_PADDING_MODES}"
    d_min = rho.shape[0]
    assert d_state >= d_min, f"d_state={d_state} < d_min={d_min}"
    out = np.eye(d_state) if target_padding == "eye" else np.zeros((d_state, d_state))
    out[:d_min, :d_min] = rho
    return out


def word_product(gens: list[np.ndarray], idx_seq) -> np.ndarray:
    """product(w) = gens[idx_seq[0]] @ gens[idx_seq[1]] @ ... (left-to-right
    matrix multiplication, a well-defined group multiplication for any word
    -- S1.4's exact `rho_G(product(w))` computed by multiplying the pinned
    reference generator matrices)."""
    d = gens[0].shape[0]
    M = np.eye(d)
    for i in idx_seq:
        M = M @ gens[int(i)]
    return M


def gen_tensor(name: str, device=None, dtype=torch.float32) -> torch.Tensor:
    """(|S_G|, d_min, d_min) torch tensor of the generating set, for
    batched training-time target computation."""
    gens = generating_set(name)
    return torch.tensor(np.stack(gens), dtype=dtype, device=device)


def batched_word_product(gens_t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """gens_t: (|S_G|, d, d). idx: (B, L) long tensor of generator indices.
    Returns (B, d, d): product gens_t[idx[b,0]] @ ... @ gens_t[idx[b,L-1]]
    per batch row, left-to-right -- matching word_product's numpy convention
    exactly (bit-identical up to float precision when the same idx sequence
    is fed to both)."""
    B, L = idx.shape
    d = gens_t.shape[-1]
    M = torch.eye(d, device=gens_t.device, dtype=gens_t.dtype).expand(B, d, d).clone()
    for t in range(L):
        M = torch.bmm(M, gens_t[idx[:, t]])
    return M


def batched_targets(name: str, idx: torch.Tensor, d_state: int, dtype=torch.float32,
                    target_padding: str = "eye") -> torch.Tensor:
    """idx: (B, L) generator indices -> (B, d_state, d_state) block-embedded
    targets rho_G_embedded(product(w)) (Option A, S1.4), computed on the
    same device as idx.

    `target_padding` (S1.33 M3 FIX WAVE flag, default UNCHANGED "eye"): see
    rho_G_embedded's docstring -- torch's exact analog, bit-for-bit agreement
    verified in this module's own self-test."""
    assert target_padding in TARGET_PADDING_MODES, \
        f"target_padding={target_padding!r} not in {TARGET_PADDING_MODES}"
    gens_t = gen_tensor(name, device=idx.device, dtype=dtype)
    prod = batched_word_product(gens_t, idx)          # (B, d_min, d_min)
    B, d_min = idx.shape[0], prod.shape[-1]
    if target_padding == "eye":
        target = torch.eye(d_state, device=idx.device, dtype=dtype).expand(B, d_state, d_state).clone()
    else:
        target = torch.zeros(B, d_state, d_state, device=idx.device, dtype=dtype)
    target[:, :d_min, :d_min] = prod
    return target


# ---------------------------------------------------------------------------
# S1.33 M3 FIX WAVE teeth (CAPABILITY_SEPARATION_DESIGN.md S1.33) -- both
# run to completion, both directions, all 5 groups, per the fix wave's own
# mandate ("never trust a check without running it to completion").
# ---------------------------------------------------------------------------

def _test_target_rank_under_padding() -> bool:
    """S1.33 TEETH #1 (the D-AMB [LEARN] rule: verify numerically, not
    assert by construction) -- rank(as-built target) must equal d_min under
    target_padding='zero' (S1.33 fix variant A) and d_state under 'eye'
    (the UNCHANGED default, the D-AMB-tainted original), for EVERY group,
    run to completion BOTH directions, on several distinct words each (not
    just one). Also cross-checks numpy rho_G_embedded against torch
    batched_targets in EACH mode, same convention as _self_test's existing
    numpy/torch product-agreement check."""
    print("=" * 88)
    print("S1.33 TEETH #1 -- target-rank unit test: rank(target)==d_min under 'zero', "
          "==d_state under 'eye' (both directions, all 5 groups)")
    print("=" * 88)
    for name in GROUP_NAMES:
        gens = generating_set(name)
        d_min, d_state = D_MIN[name], D_STATE[name]
        rng = np.random.default_rng(2026)
        for trial in range(3):    # several distinct words per group, not just one
            L = 1 + trial * 3
            idx_np = rng.integers(0, len(gens), size=L)
            prod = word_product(gens, idx_np)
            for padding, expected_rank in (("eye", d_state), ("zero", d_min)):
                target_np = rho_G_embedded(prod, d_state, target_padding=padding)
                rank = int(np.linalg.matrix_rank(target_np))
                assert rank == expected_rank, (
                    f"{name}/{padding}/trial{trial}: rank(target)={rank}, expected "
                    f"{expected_rank} (S1.33 TEETH #1)"
                )
                idx_t = torch.tensor(idx_np, dtype=torch.long).unsqueeze(0)
                target_t = batched_targets(name, idx_t, d_state, dtype=torch.float64,
                                           target_padding=padding)[0].numpy()
                max_diff = float(np.abs(target_np - target_t).max())
                assert max_diff < 1e-8, (
                    f"{name}/{padding}: numpy rho_G_embedded vs torch batched_targets mismatch "
                    f"({max_diff:.2e})"
                )
        print(f"  [{name}] rank(target): eye->{d_state} (=d_state)  zero->{d_min} (=d_min)  "
              f"numpy/torch agree  PASS  (3 trials)")
    print("\nRESULT: target rank verified numerically in BOTH directions, all 5 groups.\n")
    return True


def _test_eye_padding_ceiling_signature() -> bool:
    """S1.33 TEETH #2 -- NEGATIVE CONTROL, via a REAL numpy SVD-truncation
    fit (not symbolically re-derived): under the UNCHANGED 'eye' padding,
    the as-built target rho (+) I is orthogonal (every singular value
    exactly 1), so its Eckart-Young-optimal rank-k Frobenius approximation
    has cosine similarity EXACTLY sqrt(k/d_state) against the full target
    -- the S1.33 D-AMB diagnosis's P2 ceiling-match signature (37/39 real
    force-rank cells sat within 0.07 of this). Proves this smoke suite CAN
    detect the exact defect class the M3 fix wave exists to fix: a
    regression that silently reverted 'zero' padding back to 'eye', or
    broke the orthogonality of the eye-padded target, would trip this
    test's own assertion."""
    print("=" * 88)
    print("S1.33 TEETH #2 -- eye-padding sqrt(k/d_state) ceiling signature (NEGATIVE CONTROL, "
          "real SVD-truncation fit)")
    print("=" * 88)
    for name in GROUP_NAMES:
        gens = generating_set(name)
        d_state = D_STATE[name]
        rng = np.random.default_rng(7)
        idx_np = rng.integers(0, len(gens), size=5)
        prod = word_product(gens, idx_np)
        target = rho_G_embedded(prod, d_state, target_padding="eye")
        U, S, Vt = np.linalg.svd(target)
        assert np.allclose(S, 1.0, atol=1e-8), \
            f"{name}: eye-padded target is not orthogonal (singular values {S}, expected all 1)"
        for k in range(1, d_state + 1):
            approx = (U[:, :k] * S[:k]) @ Vt[:k, :]     # Eckart-Young rank-k optimum
            cos = float(np.sum(approx * target) / (np.linalg.norm(approx) * np.linalg.norm(target)))
            predicted = float(np.sqrt(k / d_state))
            assert abs(cos - predicted) < 1e-6, (
                f"{name}: rank-{k} SVD-optimal-fit cosine {cos:.6f} != predicted sqrt(k/d_state)="
                f"{predicted:.6f} -- the smoke's ability to DETECT the D-AMB ceiling signature "
                f"is broken (S1.33 TEETH #2)"
            )
        print(f"  [{name}] eye-padding ceiling: rank-k SVD-optimal cosine == sqrt(k/{d_state}) "
              f"for all k in 1..{d_state}  PASS")
    print("\nRESULT: the sqrt(k/d_state) ceiling signature reproduces exactly via a real "
          "SVD-truncation fit -- this smoke CAN detect the D-AMB defect class.\n")
    return True


# ---------------------------------------------------------------------------
# Self-test: closure + numpy/torch product agreement (CPU-only, torch+numpy)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    print("=" * 88)
    print("groups.py self-test -- generating-set closure (via voa.bfs_closure) + "
          "numpy/torch word-product agreement")
    print("=" * 88)
    for name in GROUP_NAMES:
        gens = generating_set(name)
        assert len(gens) == {"S3": 3, "S4": 4, "A5": 4, "S5": 3, "A6": 4}[name], \
            f"{name}: generating-set size mismatch vs S1.4's table"
        d_min = D_MIN[name]
        # closure check: BFS-close these EXACT generators, must hit ORDER[name].
        closure = voa.bfs_closure(gens, d_min, max_size=2 * ORDER[name])
        ok = len(closure) == ORDER[name]
        print(f"  [{name}] |gens|={len(gens)}  closure={len(closure)} (expect {ORDER[name]}): "
              f"{'PASS' if ok else 'FAIL'}")
        assert ok, f"{name}: generating set does not close to |G|={ORDER[name]}"

        # numpy vs torch product agreement on a random word.
        rng = np.random.default_rng(0)
        L = 7
        idx_np = rng.integers(0, len(gens), size=L)
        prod_np = word_product(gens, idx_np)
        idx_t = torch.tensor(idx_np, dtype=torch.long).unsqueeze(0)
        gens_t = gen_tensor(name, dtype=torch.float64)
        prod_t = batched_word_product(gens_t, idx_t)[0].numpy()
        max_diff = float(np.abs(prod_np - prod_t).max())
        assert max_diff < 1e-8, f"{name}: numpy/torch word-product mismatch ({max_diff:.2e})"
        print(f"         numpy/torch word-product agreement: max_diff={max_diff:.2e} PASS")

        # rho_G_embedded sanity: block-diagonal, identity complement, orthogonal.
        d_state = D_STATE[name]
        emb = rho_G_embedded(prod_np, d_state)
        assert emb.shape == (d_state, d_state)
        assert np.allclose(emb[d_min:, d_min:], np.eye(d_state - d_min))
        assert np.allclose(emb @ emb.T, np.eye(d_state), atol=1e-6), "embedded rep not orthogonal"

    # S1.22 BA-F3: group_seed_salt must be deterministic (repeat calls agree)
    # and must actually DIFFER between S4/A5 (the exact |gens|=4/d_state=5
    # collision pair the audit flagged) -- proven here, not merely claimed.
    salts = {name: group_seed_salt(name) for name in GROUP_NAMES}
    assert group_seed_salt("S4") == salts["S4"], "group_seed_salt is not deterministic"
    assert len(set(salts.values())) == len(GROUP_NAMES), \
        f"group_seed_salt collision across groups: {salts}"
    print(f"  group_seed_salt (S1.22 BA-F3): {salts}  all distinct, deterministic  PASS")

    # S1.33 M3 FIX WAVE teeth -- both run to completion, both directions.
    r1 = _test_target_rank_under_padding()
    r2 = _test_eye_padding_ceiling_signature()
    assert r1 and r2

    print("\ngroups.py self-test PASSED (all 5 groups).")


if __name__ == "__main__":
    _self_test()
