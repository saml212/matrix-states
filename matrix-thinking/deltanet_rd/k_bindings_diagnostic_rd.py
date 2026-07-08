"""k_bindings_diagnostic_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1.5
(R3-F1, replacing the Rev-2 single-pair diagonal/off-diagonal diagnostic
that attack round 3 numerically proved mathematically broken -- cosine
similarity is scale-invariant, so a single-`(q,t)`-pair reachability test
can never separate "diagonal read" from "full matvec read": a Hadamard
(diagonal) tap can ALWAYS hit exactly one target exactly, `d` unknowns
against `d` constraints. The real matvec-vs-Hadamard expressivity gap only
bites under MULTIPLE SIMULTANEOUS bindings held in one state -- this
script is the K-simultaneous-bindings replacement, itself numerically
executed pre-commit (design doc, the table at sec 1.3.1.5) and reproduced
here as a repo-committed, re-runnable artifact (gate 7's own "no second
broken diagnostic" mandate: build it, run it, and check the numbers agree
with the expected analytic behavior BEFORE trusting it).

Construction, pinned exactly (sec 1.3.1.5):
  For K simultaneous (q_i, t_i) bindings, draw q_i, t_i as independent
  random unit vectors in R^d (fresh draws every trial from ONE seeded
  generator -- reproducible, never literally re-seeded per trial), fit
  each tap family's BEST POSSIBLE state to ALL K pairs SIMULTANEOUSLY:

    matvec tap:   S = T @ pinv(Q)              (least-squares/minimum-norm)
                  matvec_pred_i = S @ q_i
    Hadamard tap: s_j = sum_i q_i[j] t_i[j] / sum_i q_i[j]^2   (separable
                  per-coordinate closed form -- no numerical solver needed)
                  had_pred_i = s (.) q_i

Score both families' predictions against the SAME frozen-formula cosine-
recovery metric this design uses everywhere (F.cosine_similarity @ 0.9),
report recovered_frac@0.9 = fraction of (trial, i) pairs clearing the
threshold, over >=100 trials per K (200 here, matching the design's own
pre-commit run).

Expected analytic behavior (sec 1.3.1.5, stated before the numbers): matvec
has d^2 free params against K*d constraints -- for K<=d and generic
(random) q_i, recovers every target NEAR-EXACTLY (recovered_frac@0.9 -> 1
for all K<=d). Hadamard has only d free params -- exactly solvable at K=1
(trivially, one scalar per coordinate against one constraint), collapsing
toward the noise floor of an uninformative 1-parameter regression for
K>=2 (recovered_frac@0.9 -> 0, mean cosine decaying ~1/sqrt(K)).

Re-run at d_state=128 for the escalation rung via --d 128 (same code, new
d, per the design's own "same code, new d" instruction) -- only the d=64
K-grid is pinned/validated by the design doc's own pre-commit numbers;
`default_k_grid` generalizes to other d as a documented, undecided
extrapolation (flagged in this script's own report, not silently assumed
equivalent).

Run: python k_bindings_diagnostic_rd.py
Exit code 0 = every runtime sanity anchor PASSED.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from probe_constants_rd import PROBE_TARGET_SEED

RECOVERY_THRESHOLD = 0.9
N_TRIALS_DEFAULT = 200
# sec 1.3.1.5's own pinned rung-1 grid (d_state=64) -- the ONLY grid this design doc's own
# pre-commit numerical execution validated.
K_GRID_D64 = (1, 2, 4, 8, 16, 32, 48, 64)


def default_k_grid(d: int) -> tuple[int, ...]:
    """The pinned K_GRID_D64 when d==64 (byte-identical to the design doc's own grid); for any
    other d (e.g. the d_state=128 escalation rung), an UNDECIDED, documented extrapolation --
    powers of two from 1 up to d, plus 3*d/4, deduplicated/sorted/capped at d. The design doc
    only says "same code, new d" without re-pinning the grid itself; this generalization is this
    build's own choice, flagged here (and in the printed report) rather than silently presented
    as equally validated."""
    if d == 64:
        return K_GRID_D64
    grid = {1}
    k = 1
    while k < d:
        k *= 2
        grid.add(min(k, d))
    grid.add(max(1, (3 * d) // 4))
    return tuple(sorted(grid))


def _unit_vectors(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    """k independent random unit vectors in R^d, returned as a (d,k) column-stacked array."""
    x = rng.standard_normal((d, k))
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    return x / norms


def _cosine_per_column(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-column cosine similarity between two (d,k) arrays -- the same
    frozen F.cosine_similarity(..., dim=-1) primitive this design uses
    everywhere, applied along the vector (row) axis of each column."""
    num = np.sum(pred * target, axis=0)
    denom = np.linalg.norm(pred, axis=0) * np.linalg.norm(target, axis=0)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return num / denom


def fit_matvec(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """S = T @ pinv(Q) (least-squares/minimum-norm solution of S@Q=T);
    returns S @ Q, the matvec tap's own best-possible simultaneous fit to
    all K bindings."""
    s = t @ np.linalg.pinv(q)
    return s @ q


def fit_hadamard(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-coordinate closed-form least squares (separable: coordinate j
    appears only in the j'th term of every ||...||^2). q,t: (d,K)."""
    numer = np.sum(q * t, axis=1)
    denom = np.sum(q * q, axis=1)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    s = numer / denom
    return s[:, None] * q


def run_k_bindings_diagnostic(d: int, k_grid: tuple[int, ...], n_trials: int, seed: int,
                               threshold: float = RECOVERY_THRESHOLD) -> list[dict]:
    """Runs the diagnostic for every K in k_grid, n_trials trials each,
    fresh independent (q_i,t_i) draws every trial from ONE seeded
    generator (reproducible from `seed` alone, never literally re-seeded
    per trial). Returns a list of per-K result dicts."""
    rng = np.random.default_rng(seed)
    results = []
    # K==d makes Q square and OCCASIONALLY near-singular by chance (a random draw can land close
    # to a rank-deficient Q) -- the design doc's own pre-commit run disclosed this exact edge case
    # at K=64/d=64: "produced large-but-non-corrupting intermediate values in the least-squares
    # solve, verified clean" (checked directly below via the min-cosine anchor, not just the mean).
    # Suppressing the resulting divide/overflow/invalid RuntimeWarnings here is a presentation
    # choice ONLY -- np.linalg.pinv's SVD-based solve is numerically robust to a near-singular Q
    # (it does not silently corrupt the returned array), which is exactly what the anchor check
    # below verifies rather than assumes.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for k in k_grid:
            matvec_cos, had_cos = [], []
            for _ in range(n_trials):
                q = _unit_vectors(rng, d, k)
                t = _unit_vectors(rng, d, k)
                matvec_cos.append(_cosine_per_column(fit_matvec(q, t), t))
                had_cos.append(_cosine_per_column(fit_hadamard(q, t), t))
            matvec_cos_k = np.concatenate(matvec_cos)
            had_cos_k = np.concatenate(had_cos)
            results.append({
                "K": k, "n_pairs": matvec_cos_k.size,
                "matvec_recovered_frac": float(np.mean(matvec_cos_k >= threshold)),
                "matvec_min_cosine": float(np.min(matvec_cos_k)),
                "hadamard_recovered_frac": float(np.mean(had_cos_k >= threshold)),
                "hadamard_mean_cosine": float(np.mean(had_cos_k)),
            })
    return results


def print_table(results: list[dict], d: int, n_trials: int, threshold: float) -> None:
    print(f"d_state={d}, n_trials={n_trials}, threshold={threshold}, seed={PROBE_TARGET_SEED}")
    print(f"{'K':>4} | {'matvec recovered_frac@0.9':>26} | {'Hadamard recovered_frac@0.9':>28} "
          f"| {'Hadamard mean cosine':>21}")
    print("-" * 92)
    for r in results:
        print(f"{r['K']:>4} | {r['matvec_recovered_frac']:>26.6f} | "
              f"{r['hadamard_recovered_frac']:>28.6f} | {r['hadamard_mean_cosine']:>21.4f}")


# ---------------------------------------------------------------------------
# Runtime sanity anchors -- gate 7's own instruction: assert the K=1 Hadamard=1.0 and K=2
# Hadamard=0.0 anchors mechanically, not merely eyeball the printed table.
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def check_sanity_anchors(results: list[dict], d: int) -> None:
    by_k = {r["K"]: r for r in results}

    if 1 in by_k:
        ok = by_k[1]["hadamard_recovered_frac"] == 1.0
        _report("anchor: K=1 Hadamard recovered_frac@0.9 == 1.0 (exactly solvable, d unknowns "
                "against d constraints)", ok, f"got {by_k[1]['hadamard_recovered_frac']}")
    else:
        _report("anchor: K=1 present in k_grid", False, "K=1 not in the swept grid -- cannot check")

    if 2 in by_k:
        ok = by_k[2]["hadamard_recovered_frac"] == 0.0
        _report("anchor: K=2 Hadamard recovered_frac@0.9 == 0.0 (collapses once K>=2 generically "
                "inconsistent equations hit the same 1 free parameter per coordinate)", ok,
                f"got {by_k[2]['hadamard_recovered_frac']}")
    else:
        _report("anchor: K=2 present in k_grid", False, "K=2 not in the swept grid -- cannot check")

    # matvec must recover EVERY K <= d near-exactly (this is the load-bearing structural claim the
    # whole diagnostic rests on -- checked directly, not merely eyeballed from the printed table).
    all_le_d_exact = all(r["matvec_recovered_frac"] == 1.0 for r in results if r["K"] <= d)
    _report(f"anchor: matvec recovered_frac@0.9 == 1.0 for every swept K <= d={d}", all_le_d_exact)

    # Hadamard must collapse (0.0) for every K>=2 in the swept grid (the qualitative pattern the
    # design doc's own pre-commit table reports across its whole K range).
    all_ge2_zero = all(r["hadamard_recovered_frac"] == 0.0 for r in results if r["K"] >= 2)
    _report("anchor: Hadamard recovered_frac@0.9 == 0.0 for every swept K >= 2", all_ge2_zero)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64,
                     help="d_state (64 = pinned rung-1 grid; 128 = escalation-rung re-run)")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--threshold", type=float, default=RECOVERY_THRESHOLD)
    ap.add_argument("--smoke", action="store_true", help="alias for the default behavior (this "
                     "script IS its own smoke gate) -- accepted so callers using this codebase's "
                     "standard --smoke convention work unmodified.")
    args = ap.parse_args()

    k_grid = default_k_grid(args.d)
    if args.d != 64:
        print(f"NOTE: d={args.d} != 64 -- using the UNDECIDED generalized grid {k_grid} "
              f"(only d=64's grid {K_GRID_D64} is validated by the design doc's own pre-commit "
              f"numbers).")

    print("=" * 92)
    print("k_bindings_diagnostic_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1.5 "
          "(K-simultaneous-bindings tap-expressivity diagnostic)")
    print("=" * 92)
    results = run_k_bindings_diagnostic(args.d, k_grid, args.n_trials, PROBE_TARGET_SEED,
                                         args.threshold)
    print_table(results, args.d, args.n_trials, args.threshold)
    print("=" * 92)
    check_sanity_anchors(results, args.d)
    print("=" * 92)
    if FAILURES:
        print(f"DIAGNOSTIC: {len(FAILURES)} SANITY-ANCHOR FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("DIAGNOSTIC: ALL SANITY ANCHORS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
