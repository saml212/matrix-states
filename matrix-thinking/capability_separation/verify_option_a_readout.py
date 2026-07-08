"""CAPABILITY_SEPARATION_DESIGN.md Stage 1 -- Option A readout verification.

Two independent checks, both executed (not asserted) and cited verbatim in
the design doc:

  PART 1 -- construct + numerically verify the five pinned reference
  representations rho_G for the marquee group family (S3, S4, A5, S5, A6):
  orthogonality, correct group order, and faithfulness. For A5 specifically,
  faithfulness also follows from group-theoretic simplicity (any nontrivial
  homomorphic image of a simple group is faithful), stated and checked here,
  not just asserted in prose.

  PART 2 -- toy numerical verification of the Procrustes/scale degauging
  pipeline (the gauge freedom g -> c*Q@rho(g)@Q.T the model may implement
  rho_G under): recovers the true (Q, c) from noisy conjugated data on a
  FIT set, scores on a disjoint EVAL set, and confirms a rank-deficient
  corruption is NOT rescued (rank is an invariant of orthogonal conjugation,
  so no valid degauging can restore it -- verified, not just claimed).

numpy + stdlib only, mirrors this project's pod-safe-dependency convention
(rank_utils.py, eigen_utils.py, analyze_zdump.py in ../chapter2/).

Run: python3 verify_option_a_readout.py
(no GPU, no torch; ~1s wall-clock; deterministic given RNG_SEED)
"""
from __future__ import annotations

import itertools

import numpy as np

RNG_SEED = 20260709  # distinct from every seed already pinned elsewhere in this repo


# =============================================================================
# PART 1 -- reference representations rho_G
# =============================================================================

def rot_axis_angle(axis, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula (3x3 proper rotation about `axis` by `theta`)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def bfs_closure(gens: list, dim: int, max_size: int = 2000) -> list:
    """BFS closure of <gens> under matrix multiplication. Returns all
    distinct group elements as dim x dim matrices (rounded-key deduplication).
    """
    def key(M):
        return tuple(np.round(M, 6).flatten())
    I = np.eye(dim)
    elements = {key(I): I}
    frontier = [I]
    while frontier:
        new_frontier = []
        for M in frontier:
            for g in gens:
                for cand in (g @ M, M @ g):
                    k = key(cand)
                    if k not in elements:
                        elements[k] = cand
                        new_frontier.append(cand)
                        if len(elements) > max_size:
                            raise RuntimeError("closure exceeded max_size")
        frontier = new_frontier
    return list(elements.values())


def hyperplane_basis(n: int) -> np.ndarray:
    """Orthonormal basis (n x (n-1)) for {x in R^n : sum(x)=0} -- the
    ambient space of S_n / A_n's real "standard" / "deleted-permutation"
    representation. Built via one Householder reflection carrying the
    all-ones direction onto a coordinate axis, then dropping that column.
    """
    ones = np.ones(n) / np.sqrt(n)
    e = np.zeros(n); e[-1] = 1.0
    u = ones - e if np.linalg.norm(ones - e) > 1e-8 else ones + e
    u = u / np.linalg.norm(u)
    H = np.eye(n) - 2 * np.outer(u, u)
    dots = np.abs(H.T @ ones)
    idx_ones = int(np.argmax(dots))
    cols = [i for i in range(n) if i != idx_ones]
    B = H[:, cols]
    assert np.allclose(B.T @ B, np.eye(n - 1), atol=1e-8)
    assert np.allclose(ones @ B, 0, atol=1e-8)
    return B


def perm_matrix(n: int, p: tuple) -> np.ndarray:
    P = np.zeros((n, n))
    for i, pi in enumerate(p):
        P[pi, i] = 1.0
    return P


def standard_rep_group(n: int, alternating: bool = False):
    """Full S_n (or A_n, if alternating=True) realized on the (n-1)-dim
    zero-sum hyperplane -- the "standard" (S_n) / "deleted-permutation"
    (A_n) real representation. Exact enumeration (itertools.permutations),
    not BFS closure -- exercises every group element directly.
    """
    B = hyperplane_basis(n)
    mats = {}
    for p in itertools.permutations(range(n)):
        if alternating:
            inv = sum(1 for i in range(n) for j in range(i + 1, n) if p[i] > p[j])
            if inv % 2 != 0:
                continue
        P = perm_matrix(n, p)
        M = B.T @ P @ B
        mats[tuple(np.round(M, 6).flatten())] = M
    return list(mats.values()), n - 1


def verify_group(name: str, elements: list, expected_order: int, dim: int) -> bool:
    ok_order = len(elements) == expected_order
    ok_ortho = all(np.allclose(M @ M.T, np.eye(dim), atol=1e-6) for M in elements)
    ok_det = all(abs(abs(np.linalg.det(M)) - 1.0) < 1e-5 for M in elements)
    passed = ok_order and ok_ortho and ok_det
    print(f"[{name}] dim={dim}  |group|={len(elements)} (expect {expected_order}): "
          f"{'OK' if ok_order else 'FAIL'}  |  all orthogonal: {ok_ortho}  |  "
          f"all det=+-1: {ok_det}  |  PASS={passed}")
    return passed


def part1_reference_representations():
    print("=" * 88)
    print("PART 1 -- reference representations rho_G (Option A pin), constructed + verified")
    print("=" * 88)
    results = {}

    # S3 -- 2-dim standard rep (D3 realization: symmetries of the equilateral
    # triangle), solvable, |S3|=6. Generators: 120deg rotation r (order 3),
    # reflection s (order 2).
    r_s3 = np.array([[np.cos(2 * np.pi / 3), -np.sin(2 * np.pi / 3)],
                      [np.sin(2 * np.pi / 3), np.cos(2 * np.pi / 3)]])
    s_s3 = np.array([[1.0, 0.0], [0.0, -1.0]])
    elts_s3 = bfs_closure([r_s3, s_s3], 2)
    results["S3"] = dict(elements=elts_s3, dim=2, order=6, solvable=True,
                          generators={"r": r_s3, "s": s_s3})
    verify_group("S3", elts_s3, 6, 2)

    # S4 -- 3-dim cube-rotation rep, solvable, |S4|=24. Generators: 90deg
    # about z (order 4), 120deg about the (1,1,1) body diagonal (order 3).
    r_s4 = rot_axis_angle([0, 0, 1], np.pi / 2)
    s_s4 = rot_axis_angle([1, 1, 1], 2 * np.pi / 3)
    elts_s4 = bfs_closure([r_s4, s_s4], 3)
    results["S4"] = dict(elements=elts_s4, dim=3, order=24, solvable=True,
                          generators={"r": r_s4, "s": s_s4})
    verify_group("S4", elts_s4, 24, 3)

    # A5 -- 3-dim icosahedral rep, NON-solvable, |A5|=60. 12 icosahedron
    # vertices = all even permutations of (0, +-1, +-phi). 5-fold axis
    # through a vertex; 3-fold axis through an adjacent face's centroid
    # (found from the vertex's 5 nearest neighbors, taking a pair of them
    # that are themselves adjacent -- i.e. a genuine triangular face).
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for a, b in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        verts.append((0, a * 1, b * phi))
        verts.append((a * 1, b * phi, 0))
        verts.append((b * phi, 0, a * 1))
    verts = np.unique(np.round(np.array(verts, dtype=float), 8), axis=0)
    assert verts.shape == (12, 3)
    v0 = verts[0]
    dists = np.linalg.norm(verts - v0, axis=1)
    order_idx = np.argsort(dists)
    nearest5 = verts[order_idx[1:6]]
    edge_len = dists[order_idx[1]]
    face = None
    for i in range(5):
        for j in range(i + 1, 5):
            if abs(np.linalg.norm(nearest5[i] - nearest5[j]) - edge_len) < 1e-4:
                face = (nearest5[i], nearest5[j])
                break
        if face:
            break
    assert face is not None, "no adjacent pair found among v0's 5 nearest neighbors"
    axis3 = v0 + face[0] + face[1]
    g5_a5 = rot_axis_angle(np.array([0.0, 1.0, phi]), 2 * np.pi / 5)
    g3_a5 = rot_axis_angle(axis3, 2 * np.pi / 3)

    def vertex_set_preserved(R, verts, tol=1e-4):
        mapped = verts @ R.T
        return all(np.any(np.all(np.abs(verts - m) < tol, axis=1)) for m in mapped)

    assert vertex_set_preserved(g5_a5, verts), "g5 is not a genuine icosahedron symmetry"
    assert vertex_set_preserved(g3_a5, verts), "g3 is not a genuine icosahedron symmetry"
    elts_a5 = bfs_closure([g5_a5, g3_a5], 3, max_size=200)
    ok_a5 = verify_group("A5", elts_a5, 60, 3)
    print("  A5 simplicity argument: A5 is simple (only normal subgroups {e}, A5); "
          "image order == |A5| == 60 forces kernel = {e} => the representation is "
          "FAITHFUL as a consequence of group simplicity, not merely spot-checked.")
    results["A5"] = dict(elements=elts_a5, dim=3, order=60, solvable=False,
                          generators={"g5": g5_a5, "g3": g3_a5})

    # S5 -- 4-dim standard rep (zero-sum hyperplane in R^5), NON-solvable,
    # |S5|=120. Full enumeration (all 120 permutations), not BFS closure --
    # exercises every element directly.
    mats_s5, dim_s5 = standard_rep_group(5, alternating=False)
    ok_s5 = verify_group("S5", mats_s5, 120, dim_s5)
    p_transposition = (1, 0, 2, 3, 4)
    p_5cycle = (1, 2, 3, 4, 0)
    B5 = hyperplane_basis(5)
    results["S5"] = dict(elements=mats_s5, dim=4, order=120, solvable=False,
                          generators={"transposition_01": B5.T @ perm_matrix(5, p_transposition) @ B5,
                                      "5cycle_01234": B5.T @ perm_matrix(5, p_5cycle) @ B5})

    # A6 -- 5-dim standard(deleted-permutation) rep, NON-solvable, |A6|=360.
    # Full enumeration of all even permutations of {0..5}.
    mats_a6, dim_a6 = standard_rep_group(6, alternating=True)
    ok_a6 = verify_group("A6", mats_a6, 360, dim_a6)
    results["A6"] = dict(elements=mats_a6, dim=5, order=360, solvable=False)

    print("\nSummary: d_min(G) by group (all verified faithful + orthogonal this run):")
    for g, info in results.items():
        print(f"  {g:<4} d_min={info['dim']}  |G|={info['order']}  "
              f"solvable={info['solvable']}")
    all_ok = ok_a5 and ok_s5 and ok_a6  # S3/S4 checked via verify_group's own bool, re-asserted below
    assert verify_group("S3 (re-check)", results["S3"]["elements"], 6, 2)
    assert verify_group("S4 (re-check)", results["S4"]["elements"], 24, 3)
    assert all_ok, "one or more reference representations failed verification"
    print("\nPART 1: ALL FIVE REFERENCE REPRESENTATIONS VERIFIED (orthogonal, correct "
          "order, faithful).")
    return results


# =============================================================================
# PART 2 -- Procrustes/scale degauging pipeline, toy numerical verification
# =============================================================================

def fit_scale(Z_list, rho_list):
    ratios = [np.linalg.norm(Z) / np.linalg.norm(r) for Z, r in zip(Z_list, rho_list)]
    return float(np.median(ratios)), ratios


def fit_orthogonal_intertwiner(Z_list, rho_list, c_hat, d):
    """Recover the single orthogonal Q with Z(g) ~= c_hat * Q @ rho(g) @ Q.T
    for every g in the fit set. Z(g)@Q - c_hat*Q@rho(g) = 0 is LINEAR in Q;
    vectorized via vec(AXB) = (B^T kron A) vec(X), stacked over the fit set,
    solved as the smallest-singular-value right-singular vector of the
    stack, then projected to the nearest orthogonal matrix (polar
    decomposition). This is the Schur's-lemma intertwiner recovery, not a
    plain single-sided Procrustes fit.
    """
    I_d = np.eye(d)
    rows = [np.kron(I_d, Z) - c_hat * np.kron(rho.T, I_d) for Z, rho in zip(Z_list, rho_list)]
    L_stack = np.concatenate(rows, axis=0)
    U, S, Vt = np.linalg.svd(L_stack)
    Q_raw = Vt[-1, :].reshape(d, d, order="F")
    Uo, _, Vo = np.linalg.svd(Q_raw)
    Q_hat = Uo @ Vo
    return Q_hat, S[-1], S[-2]


def score_eval(Z_list, rho_list, Q_hat, c_hat):
    coses, rel_errs = [], []
    for Z, rho in zip(Z_list, rho_list):
        Z_pred = c_hat * (Q_hat @ rho @ Q_hat.T)
        v_pred, v_true = Z_pred.flatten(), Z.flatten()
        cos = float(np.dot(v_pred, v_true) / (np.linalg.norm(v_pred) * np.linalg.norm(v_true) + 1e-12))
        rel_err = float(np.linalg.norm(Z_pred - Z) / (np.linalg.norm(Z) + 1e-12))
        coses.append(cos)
        rel_errs.append(rel_err)
    return coses, rel_errs


def run_condition(name, Z_fit, Z_eval, rho_fit, rho_eval, d):
    c_hat, ratios = fit_scale(Z_fit, rho_fit)
    Q_hat, sv_min, sv_2nd = fit_orthogonal_intertwiner(Z_fit, rho_fit, c_hat, d)
    ortho_err = float(np.linalg.norm(Q_hat @ Q_hat.T - np.eye(d)))
    coses, rel_errs = score_eval(Z_eval, rho_eval, Q_hat, c_hat)
    rec90 = float(np.mean([c > 0.9 for c in coses]))
    print(f"\n=== {name} ===")
    print(f"  fit-set c_hat = {c_hat:.4f}  (ratio spread: min={min(ratios):.4f}, max={max(ratios):.4f})")
    print(f"  intertwiner smallest sv = {sv_min:.6f}, 2nd-smallest sv = {sv_2nd:.6f}  "
          f"(gap = {sv_2nd - sv_min:.4f})")
    print(f"  ||Q_hat Q_hat^T - I||_F = {ortho_err:.2e}")
    print(f"  EVAL SET (n={len(Z_eval)}, never used to fit c_hat/Q_hat): "
          f"mean_cos={np.mean(coses):.6f}  mean_rel_err={np.mean(rel_errs):.6f}  "
          f"recovered_frac@0.9={rec90:.4f}")
    return dict(name=name, c_hat=c_hat, sv_min=sv_min, sv_2nd=sv_2nd,
                mean_cos=float(np.mean(coses)), mean_rel_err=float(np.mean(rel_errs)),
                recovered_frac_90=rec90)


def part2_degauging_verification(group_data: dict):
    print("\n" + "=" * 88)
    print("PART 2 -- Procrustes/scale degauging pipeline: toy numerical verification (S4)")
    print("=" * 88)
    rng = np.random.default_rng(RNG_SEED)
    group = group_data["S4"]["elements"]
    d = group_data["S4"]["dim"]

    idx = list(range(len(group)))
    rng.shuffle(idx)
    fit_idx, eval_idx = idx[:14], idx[14:]
    assert set(fit_idx).isdisjoint(eval_idx)
    rho_fit = [group[i] for i in fit_idx]
    rho_eval = [group[i] for i in eval_idx]
    print(f"[setup] |S4| = {len(group)} elements; fit set = {len(rho_fit)}, "
          f"eval set = {len(rho_eval)}, disjoint = {set(fit_idx).isdisjoint(eval_idx)}")

    X = rng.standard_normal((d, d))
    Qraw, Rraw = np.linalg.qr(X)
    Q_true = Qraw * np.sign(np.diag(Rraw))  # Mezzadri (2007) sign correction, Haar-uniform
    c_true = 1.7
    print(f"[ground truth] c_true={c_true}, Q_true orthogonal "
          f"(||QQ^T-I||={np.linalg.norm(Q_true @ Q_true.T - np.eye(d)):.2e}), "
          f"det(Q_true)={np.linalg.det(Q_true):.4f}")

    results = []
    Z_fit_A = [c_true * (Q_true @ r @ Q_true.T) for r in rho_fit]
    Z_eval_A = [c_true * (Q_true @ r @ Q_true.T) for r in rho_eval]
    results.append(run_condition("A: exact conjugation, zero noise (sanity check)",
                                  Z_fit_A, Z_eval_A, rho_fit, rho_eval, d))

    NOISE_STD = 0.03
    Z_fit_B = [c_true * (Q_true @ r @ Q_true.T) + rng.normal(0, NOISE_STD, (d, d)) for r in rho_fit]
    Z_eval_B = [c_true * (Q_true @ r @ Q_true.T) + rng.normal(0, NOISE_STD, (d, d)) for r in rho_eval]
    results.append(run_condition(f"B: exact conjugation + noise(std={NOISE_STD})",
                                  Z_fit_B, Z_eval_B, rho_fit, rho_eval, d))

    Qdef = Q_true.copy(); Qdef[-1, :] = 0.0
    rank_Qdef = np.linalg.matrix_rank(Qdef)
    print(f"\n[negative control] rank(Qdef)={rank_Qdef} (of {d}) -- Q_true's last row zeroed")
    Z_fit_C = [c_true * (Qdef @ r @ Qdef.T) + rng.normal(0, NOISE_STD, (d, d)) for r in rho_fit]
    Z_eval_C = [c_true * (Qdef @ r @ Qdef.T) + rng.normal(0, NOISE_STD, (d, d)) for r in rho_eval]
    corrupt_ranks = sorted({np.linalg.matrix_rank(c_true * (Qdef @ r @ Qdef.T), tol=1e-6) for r in rho_eval})
    print(f"[negative control] rank(Z_corrupt(g)) on eval set (pre-noise): {corrupt_ranks} "
          f"(expect <= {d - 1}, never {d})")
    results.append(run_condition("C: rank-deficient corruption (NEGATIVE CONTROL -- "
                                  "must NOT be rescued)", Z_fit_C, Z_eval_C, rho_fit, rho_eval, d))

    print("\n" + "=" * 88)
    print("SUMMARY TABLE (eval-set metrics only; fit-set never scored)")
    print("=" * 88)
    header = f"{'condition':<58}{'mean_cos':>10}{'rel_err':>10}{'rec@0.9':>10}{'sv_gap':>10}"
    print(header); print("-" * len(header))
    for r in results:
        print(f"{r['name']:<58}{r['mean_cos']:>10.4f}{r['mean_rel_err']:>10.4f}"
              f"{r['recovered_frac_90']:>10.4f}{(r['sv_2nd'] - r['sv_min']):>10.4f}")

    assert results[0]["mean_cos"] > 0.9999
    assert results[1]["mean_cos"] > 0.95 and results[1]["recovered_frac_90"] >= 0.9
    assert results[2]["mean_cos"] < results[1]["mean_cos"] - 0.3
    assert results[2]["recovered_frac_90"] < 0.5
    print("\nPART 2: ALL ASSERTIONS PASSED -- degauging recovers the true gauge under "
          "noise (B), and provably cannot rescue a rank-deficient corruption on the "
          "held-out eval set (C). The fit/eval split has teeth.")
    return results


if __name__ == "__main__":
    group_data = part1_reference_representations()
    part2_degauging_verification(group_data)
