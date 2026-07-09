"""CAPABILITY_SEPARATION_DESIGN.md Rev 2 -- CA1-F1 fix (query-coverage bar
calibration) + CA2-M1 fix (A5 generator CLASS verification), both
numerically executed (not asserted).

Attack round 1 (design doc S1.13, CA1-F1) found the Rev 0 bar
"|>=200 distinct elements for A5/S5/A6" mathematically impossible for A5
(|A5|=60) and S5 (|S5|=120): a group cannot yield more distinct elements
than it has. This script computes, by direct Monte Carlo simulation of the
ACTUAL held-out random-walk sampler (L ~ Uniform{9,16}, i.i.d. generator
draws from each group's pinned symmetric generating set, matching
CAPABILITY_SEPARATION_DESIGN.md S1.4's table exactly), what fraction of
|G| a HEALTHY sampler reaches in expectation at the pinned N=50-word
floor -- and what an UNDERSAMPLED (pathological) sampler reaches under the
same N -- so the replacement bar is picked from real numbers, not guessed.

Group elements are represented as permutations (tuples), NOT the 3D/4D/5D
matrix realizations in verify_option_a_readout.py -- distinct-element
COUNTING under a random walk depends only on the abstract Cayley graph
(group + generating set), not on which faithful representation later reads
it out. Each generator's cycle type is chosen to match the ORDER structure
S1.4's table specifies (order-3/order-4/order-5 rotations, a
transposition, ...), and closure size is verified against |G| before any
simulation runs (see the printed PASS lines below) -- this is the same
discipline S1.3's bfs_closure check applies to the matrix realizations,
applied here to confirm the permutation stand-ins are faithful too.

numpy + stdlib only. Run: python3 coverage_calibration.py
Deterministic given RNG_SEED (distinct from every seed already pinned
elsewhere in this repo).
"""
from __future__ import annotations

import contextlib
import io
import itertools
import os
import sys

import numpy as np

# Ensure the sibling script (verify_option_a_readout.py, same directory) is
# importable regardless of invocation cwd -- CA2-M1 fix imports its A5
# generator construction directly rather than re-implementing the
# icosahedron geometry a second time (avoids drift between the two scripts,
# same "functions imported, not re-implemented" discipline S1.11 already
# applies to the production eval harness).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RNG_SEED = 20260710
N_EVAL_WORDS = 50          # the pinned floor, "N>=50 words per group" (S1.4.1)
N_TRIALS = 20000           # Monte Carlo repetitions of the N=50-word draw
L_LO, L_HI = 9, 16         # held-out word-length range, inclusive (S1.4)


def compose(p: tuple, q: tuple) -> tuple:
    """(p o q)(i) = p[q[i]] -- apply q then p."""
    return tuple(p[q[i]] for i in range(len(p)))


def bfs_closure(gens: list, n: int, max_size: int = 1000) -> set:
    ident = tuple(range(n))
    elements = {ident}
    frontier = [ident]
    while frontier:
        new_frontier = []
        for p in frontier:
            for g in gens:
                for cand in (compose(g, p), compose(p, g)):
                    if cand not in elements:
                        elements.add(cand)
                        new_frontier.append(cand)
                        if len(elements) > max_size:
                            raise RuntimeError("closure exceeded max_size")
        frontier = new_frontier
    return elements


def inverse(p: tuple) -> tuple:
    n = len(p)
    inv = [0] * n
    for i, pi in enumerate(p):
        inv[pi] = i
    return tuple(inv)


# =============================================================================
# Generating sets -- permutation stand-ins for S1.4's table, same generator
# SIZES and order structure (verified against |G| via bfs_closure below).
# =============================================================================

GROUPS = {}

# S3 (n=3): r = 3-cycle (order 3), s = transposition (order 2, self-inverse).
# S1.4 generating set: {r, r^-1, s}, size 3.
r = (1, 2, 0)
s = (1, 0, 2)
GROUPS["S3"] = dict(n=3, dmin=2, order=6, gens=[r, inverse(r), s])

# S4 (n=4): r = 4-cycle (order 4), s = 3-cycle (order 3).
# S1.4 generating set: {r, r^-1, s, s^-1}, size 4.
r = (1, 2, 3, 0)
s = (1, 2, 0, 3)
GROUPS["S4"] = dict(n=4, dmin=3, order=24, gens=[r, inverse(r), s, inverse(s)])

# A5 (n=5): g5 = 5-cycle (order 5), g3 = 3-cycle (order 3).
# S1.4 generating set: {g5, g5^-1, g3, g3^-1}, size 4.
g5 = (1, 2, 3, 4, 0)
g3 = (1, 2, 0, 3, 4)
GROUPS["A5"] = dict(n=5, dmin=3, order=60, gens=[g5, inverse(g5), g3, inverse(g3)])

# S5 (n=5): t = transposition (self-inverse), c = 5-cycle (order 5).
# S1.4 generating set: {t, c, c^-1}, size 3.
t = (1, 0, 2, 3, 4)
c = (1, 2, 3, 4, 0)
GROUPS["S5"] = dict(n=5, dmin=4, order=120, gens=[t, c, inverse(c)])

# A6 (n=6): a = 3-cycle (0 1 2), b = 5-cycle (1 2 3 4 5) -- shares one point,
# the standard even-permutation generating pair S1.4 cites.
# S1.4 generating set: {a, a^-1, b, b^-1}, size 4.
a = (1, 2, 0, 3, 4, 5)
b = (0, 2, 3, 4, 5, 1)
GROUPS["A6"] = dict(n=6, dmin=5, order=360, gens=[a, inverse(a), b, inverse(b)])


# =============================================================================
# CA2-M1 -- A5 order-5 generator CLASS verification (attack round 2, S1.15).
#
# A5's order-5 elements split into TWO non-conjugate classes in A5 itself
# (though they fuse into one class of 5-cycles in S5): cycle type "(5)" has a
# single odd-length part, which is exactly the pattern that splits under the
# index-2 restriction from S_n to A_n (the same fact behind A5 having two
# inequivalent 3-dimensional real irreps, rather than one). Attack round 2
# found this permutation stand-in's g5 was never checked against the SAME
# class as verify_option_a_readout.py's real icosahedral 2*pi/5 rotation
# generator g5_a5, and that the choice is NOT cosmetic: swapping to the
# other class empirically changes the Monte Carlo coverage numbers (S1.15:
# mean 33.79->31.76, p1 28->26, dropping BELOW the pinned bar of 27).
#
# Verified here via an EXPLICIT, generator-matched group isomorphism, not a
# trusted/recalled character-table value (the attack's own parenthetical
# guess, "traces phi-1 vs -phi", is checked directly against the REAL
# rotation matrices below and found NOT to match this representation's
# actual values -- see the printed trace check).
# =============================================================================

def _mat_key(M: np.ndarray) -> tuple:
    return tuple(np.round(M, 6).flatten())


def _simultaneous_closure(perm_gens: list, matrix_gens: list):
    """BFS-close BOTH the permutation group and the matrix group AT ONCE,
    multiplying by GENERATOR-MATCHED pairs (perm_gens[i] <-> matrix_gens[i])
    on both sides at every step -- both left- and right-multiplication,
    mirroring exactly how bfs_closure() above and verify_option_a_readout.py's
    own bfs_closure() each build their respective groups. Returns
    (is_consistent, n_perm_elements, n_matrix_elements). `is_consistent`
    goes False the instant the SAME permutation is reached via two different
    generator-words that assign it TWO DIFFERENT matrices -- i.e. the
    generator correspondence is not a well-defined function, hence not a
    homomorphism. `is_consistent and n_perm==n_matrix==|G|` is a
    constructive, exhaustive proof of a genuine group isomorphism (every one
    of the |G| elements checked, not a sample) -- not an assumption from
    matching orders/traces alone.
    """
    ident_p = tuple(range(len(next(iter(perm_gens)))))
    dim = matrix_gens[0].shape[0]
    ident_m = np.eye(dim)
    perm_to_matkey = {ident_p: _mat_key(ident_m)}
    mat_elements = {_mat_key(ident_m): ident_m}
    frontier = [(ident_p, ident_m)]
    consistent = True
    while frontier:
        new_frontier = []
        for p, m in frontier:
            for gi in range(len(perm_gens)):
                for cand_p, cand_m in (
                    (compose(perm_gens[gi], p), matrix_gens[gi] @ m),
                    (compose(p, perm_gens[gi]), m @ matrix_gens[gi]),
                ):
                    mk = _mat_key(cand_m)
                    if cand_p in perm_to_matkey:
                        if perm_to_matkey[cand_p] != mk:
                            consistent = False
                    else:
                        perm_to_matkey[cand_p] = mk
                        mat_elements[mk] = cand_m
                        new_frontier.append((cand_p, cand_m))
        frontier = new_frontier
    return consistent, len(perm_to_matkey), len(mat_elements)


def _get_real_a5_generators():
    """Import the REAL icosahedral A5 generators directly from
    verify_option_a_readout.py (sibling script, same directory) rather than
    re-implementing the icosahedron-axis geometry a second time -- avoids
    drift between the two scripts. Its own Part-1/Part-2 print output
    (already shown verbatim in S1.3.1/S1.3.2) is captured and discarded
    here; only the two A5 generator matrices are needed.
    """
    import verify_option_a_readout as voa
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = voa.part1_reference_representations()
    gens = results["A5"]["generators"]
    return gens["g5"], gens["g3"]


def verify_a5_generator_class():
    print("=" * 88)
    print("CA2-M1 -- A5 order-5 generator CLASS verification")
    print("=" * 88)
    g5_a5, g3_a5 = _get_real_a5_generators()
    g5_a5_sq = g5_a5 @ g5_a5

    def order_of(M):
        P = np.eye(M.shape[0])
        for n in range(1, 8):
            P = P @ M
            if np.allclose(P, np.eye(M.shape[0]), atol=1e-5):
                return n
        raise RuntimeError("order not found")

    phi = (1 + np.sqrt(5)) / 2
    tr_g5 = float(np.trace(g5_a5))
    tr_g5sq = float(np.trace(g5_a5_sq))
    print(f"  order(g5_a5) = {order_of(g5_a5)}   order(g5_a5^2) = {order_of(g5_a5_sq)}"
          f"   order(g3_a5) = {order_of(g3_a5)}")
    print(f"  trace(g5_a5)   = {tr_g5:.6f}   (== phi = {phi:.6f}? {abs(tr_g5 - phi) < 1e-6})")
    print(f"  trace(g5_a5^2) = {tr_g5sq:.6f}   (== 1-phi = {1 - phi:.6f}? {abs(tr_g5sq - (1 - phi)) < 1e-6})")
    print(f"  [note] the attack's own parenthetical guess ('traces phi-1={phi - 1:.4f} vs "
          f"-phi={-phi:.4f}') does NOT match either computed trace above -- computed "
          f"directly from the rep matrices, not trusted from the parenthetical, per "
          f"the CA2-M1 instruction.")

    info = GROUPS["A5"]
    g5_perm, g5inv_perm, g3_perm, g3inv_perm = info["gens"]

    print()
    print("  Testing candidate generator-matched isomorphisms (full 60-element")
    print("  simultaneous BFS closure, well-definedness checked at every step):")
    candidates = [
        ("g5 <-> g5_a5,    g3 <-> g3_a5   (naive as-labeled pairing)",
         [g5_perm, g5inv_perm, g3_perm, g3inv_perm],
         [g5_a5, g5_a5.T, g3_a5, g3_a5.T]),
        ("g5 <-> g5_a5,    g3 <-> g3_a5^-1 (inverse-labeled g3; SAME symmetric set)",
         [g5_perm, g5inv_perm, g3_perm, g3inv_perm],
         [g5_a5, g5_a5.T, g3_a5.T, g3_a5]),
        ("g5 <-> g5_a5^2,  g3 <-> g3_a5   (wrong-class control)",
         [g5_perm, g5inv_perm, g3_perm, g3inv_perm],
         [g5_a5_sq, g5_a5_sq.T, g3_a5, g3_a5.T]),
        ("g5 <-> g5_a5^2,  g3 <-> g3_a5^-1 (wrong-class control, inverse-labeled)",
         [g5_perm, g5inv_perm, g3_perm, g3inv_perm],
         [g5_a5_sq, g5_a5_sq.T, g3_a5.T, g3_a5]),
    ]
    results = {}
    for label, pg, mg in candidates:
        ok, n_p, n_m = _simultaneous_closure(pg, mg)
        results[label] = ok
        print(f"    [{'PASS' if ok and n_p == 60 and n_m == 60 else 'FAIL'}] {label}"
              f"  (perm elems={n_p}, matrix elems={n_m})")

    same_class_ok = results["g5 <-> g5_a5,    g3 <-> g3_a5^-1 (inverse-labeled g3; SAME symmetric set)"]
    wrong_class_ok = any(v for k, v in results.items() if "g5_a5^2" in k)
    print()
    if same_class_ok and not wrong_class_ok:
        print("  VERDICT: coverage_calibration.py's CURRENT stand-in (g5, unmodified) IS")
        print("  the CORRECT class -- it forms a genuine, exhaustively-checked group")
        print("  isomorphism (all 60 elements, zero inconsistencies) with the REAL")
        print("  production generator g5_a5, matched via g3 <-> g3_a5^-1 (a pure")
        print("  inverse-labeling artifact, immaterial to the symmetric generating set")
        print("  {g5,g5^-1,g3,g3^-1} actually used for the random walk, since that set")
        print("  already contains both g3_a5 and g3_a5^-1 regardless of which one is")
        print("  called 'primary'). The g5_a5^2 (wrong-class) controls above both FAIL")
        print("  to close to the full group under the naive as-labeled pairing, cleanly")
        print("  separating right-class from wrong-class. NO RE-CALIBRATION NEEDED --")
        print("  S1.3.3's existing A5 numbers (generator = g5, unmodified) already used")
        print("  the class-correct generator.")
    else:
        print("  VERDICT: UNRESOLVED BY THIS CHECK -- see printed PASS/FAIL rows above;")
        print("  this is a real finding requiring hand inspection, not silently passed.")
    print()
    return same_class_ok and not wrong_class_ok


def verify_closure():
    print("=" * 88)
    print("STEP 0 -- verify each generating set's closure matches |G| (faithfulness")
    print("of the permutation stand-in, same discipline as S1.3's bfs_closure check)")
    print("=" * 88)
    for name, info in GROUPS.items():
        closure = bfs_closure(info["gens"], info["n"], max_size=2 * info["order"])
        ok = len(closure) == info["order"]
        print(f"  [{name}] closure size = {len(closure)}  (expect {info['order']}): "
              f"{'PASS' if ok else 'FAIL'}  |  gen set size = {len(info['gens'])}")
        assert ok, f"{name} generating set does not close to the correct group order"
    print()


# =============================================================================
# Monte Carlo: distinct-element coverage of N=50 held-out words, L~U{9,16}
# =============================================================================

def sample_word_result(rng, gens, L):
    """Compose L i.i.d.-drawn generators (order = draw order) via repeated
    left-multiplication, matching S1.4's word w = g_i1 g_i2 ... g_iL
    (product taken in the order drawn)."""
    idxs = rng.integers(0, len(gens), size=L)
    n = len(gens[0])
    acc = tuple(range(n))
    for i in idxs:
        acc = compose(gens[i], acc)
    return acc


def healthy_trial(rng, info):
    """One N=50-word draw under the DESIGN'S OWN pinned sampler: L~U{9,16},
    i.i.d. generator draws from the full pinned generating set."""
    results = set()
    for _ in range(N_EVAL_WORDS):
        L = rng.integers(L_LO, L_HI + 1)
        results.add(sample_word_result(rng, info["gens"], L))
    return len(results)


def undersampled_trial(rng, info):
    """Pathological/undersampled negative control for gate 3's
    _test_coverage_guard_detects_undersampling: L fixed at 1 (a plausible
    real bug -- e.g. the held-out-length sampler silently defaults to the
    trivial case, or an off-by-range error clamps L to {1}). At L=1 a word
    IS a single generator draw, so at most |gens| distinct elements can
    EVER be reached regardless of N -- this is a hard ceiling failure mode,
    not a statistical one, by construction."""
    results = set()
    for _ in range(N_EVAL_WORDS):
        L = 1
        results.add(sample_word_result(rng, info["gens"], L))
    return len(results)


def run_calibration():
    rng_master = np.random.default_rng(RNG_SEED)
    print("=" * 88)
    print(f"STEP 1 -- Monte Carlo: N={N_EVAL_WORDS} words/trial, "
          f"{N_TRIALS} trials/group/sampler, L~Uniform{{{L_LO}..{L_HI}}}")
    print("=" * 88)
    header = (f"{'Group':<6}{'|G|':>6}{'d_min':>6}  "
              f"{'healthy: mean':>14}{'p1':>7}{'p5':>7}{'min':>6}  |  "
              f"{'undersamp: mean':>16}{'max':>6}")
    print(header)
    print("-" * len(header))
    table = {}
    for name, info in GROUPS.items():
        healthy = np.array([healthy_trial(rng_master, info) for _ in range(N_TRIALS)])
        undersamp = np.array([undersampled_trial(rng_master, info) for _ in range(N_TRIALS)])
        g = info["order"]
        h_mean, h_p1, h_p5, h_min = healthy.mean(), np.percentile(healthy, 1), np.percentile(healthy, 5), healthy.min()
        u_mean, u_max = undersamp.mean(), undersamp.max()
        table[name] = dict(order=g, dmin=info["dmin"], h_mean=h_mean, h_p1=h_p1, h_p5=h_p5,
                            h_min=h_min, u_mean=u_mean, u_max=u_max, n_gens=len(info["gens"]))
        print(f"{name:<6}{g:>6}{info['dmin']:>6}  "
              f"{h_mean:>10.2f} ({100*h_mean/g:4.1f}%){h_p1:>7.0f}{h_p5:>7.0f}{h_min:>6.0f}  |  "
              f"{u_mean:>12.2f} ({100*u_mean/g:4.1f}%){u_max:>6.0f}")
    return table


def pick_bars(table):
    print()
    print("=" * 88)
    print("STEP 2 -- calibrated bar selection: largest 5%-of-|G| bar such that")
    print("(a) it clears the healthy-sampler 1st-percentile floor (>=99% of")
    print("healthy N=50 draws pass) AND (b) it strictly exceeds the")
    print("undersampled (L=1 pathological) ceiling -- note undersamp max is a")
    print("DETERMINISTIC hard cap at L=1 (== |gens|, never exceedable by")
    print("construction, not just a statistical tail), so bar>u_max alone")
    print("already guarantees a 100%, not just high-probability, fail rate")
    print("for that corruption mode -- both conditions numerically checked.")
    print("=" * 88)
    header = (f"{'Group':<6}{'|G|':>6}{'healthy p1':>12}{'undersamp max':>16}"
              f"{'bar (frac)':>12}{'bar (count)':>13}{'p1 margin':>11}{'undersamp mult':>16}")
    print(header)
    print("-" * len(header))
    bars = {}
    candidate_fracs = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
                       0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    for name, row in table.items():
        g = row["order"]
        chosen = None
        for frac in candidate_fracs:
            bar_count = frac * g
            # require >=1-element buffer below the healthy p1 floor (not just
            # a tie at the boundary) AND a strict exceedance of the
            # deterministic undersampled ceiling.
            if row["h_p1"] - bar_count >= 1.0 and bar_count > row["u_max"]:
                chosen = frac
                break
        assert chosen is not None, f"{name}: no candidate fraction satisfies both conditions"
        bars[name] = dict(frac=chosen, count=chosen * g)
        bar_count = chosen * g
        print(f"{name:<6}{g:>6}{row['h_p1']:>12.0f}{row['u_max']:>16.0f}"
              f"{chosen:>12.2f}{bar_count:>13.1f}"
              f"{(row['h_p1'] - bar_count):>11.1f}{(bar_count / max(row['u_max'], 1)):>16.2f}")
    return bars


# =============================================================================
# CA1-M3 -- degauging fit-set diversity floor: at n_fit=30 (60% of the N=50
# floor, S1.14's pinned 60/40 fit/eval split), does the fit subset realize
# >=3*d_min(G) distinct target elements with margin? (The orthogonal
# intertwiner Q is d_min(G) x d_min(G), i.e. d_min(G)^2 unknowns; 3*d_min(G)
# is a diversity floor well below the toy verification's own ~4.7*d ratio
# (14 fit elements / d=3 in S1.3.2), chosen conservatively and checked here.)
# =============================================================================

N_FIT = 30  # 60% of N_EVAL_WORDS=50, S1.14's pinned split


def fit_set_diversity_check(table_keys):
    print()
    print("=" * 88)
    print(f"STEP 3 -- CA1-M3 fit-set diversity floor: n_fit={N_FIT} words/trial, "
          f"{N_TRIALS} trials/group, bar = 3*d_min(G) distinct elements")
    print("=" * 88)
    rng_master = np.random.default_rng(RNG_SEED + 1)
    header = f"{'Group':<6}{'|G|':>6}{'d_min':>6}{'bar=min(3d,.8|G|)':>19}{'fit p1':>9}{'fit mean':>10}{'margin':>9}"
    print(header)
    print("-" * len(header))
    for name, info in GROUPS.items():
        results = np.array([
            len({sample_word_result(rng_master, info["gens"], int(rng_master.integers(L_LO, L_HI + 1)))
                 for _ in range(N_FIT)})
            for _ in range(N_TRIALS)
        ])
        # bar = 3*d_min(G), capped at floor(0.8*|G|) for small groups where
        # 3*d_min would exceed what's practically achievable with margin
        # (S3: 3*2=6=|G| exactly, no room for a healthy-trial safety margin).
        bar = min(3 * info["dmin"], int(0.8 * info["order"]))
        p1 = np.percentile(results, 1)
        ok = p1 >= bar
        print(f"{name:<6}{info['order']:>6}{info['dmin']:>6}{bar:>19}{p1:>9.0f}"
              f"{results.mean():>10.2f}{(p1 - bar):>9.0f}  {'PASS' if ok else 'FAIL'}")
        assert ok, f"{name}: fit-set diversity floor 3*d_min not cleared at n_fit={N_FIT}"


# =============================================================================
# CA2-m2 -- eval-set (n_eval=20, the DISJOINT scoring subset, S1.4.1 step 4)
# diversity floor. Attack round 2 found the fit-set floor above (S1.3.3 STEP
# 3) has no analog on the SCORING subset: an eval draw that happens to
# collapse to very few distinct targets would still "score" (mean_cos /
# recovered_frac@0.9), just as a poorly-powered estimate of the checkpoint's
# real accuracy, not a correctness bug -- unguarded, though empirically fine
# per the round-2 finding. A MODEST floor (not the fit set's equation-
# solving-strength bar; the eval set does no fitting) anchored to the SAME
# MC machinery: bar = min(2*d_min(G), floor(0.6*|G|)) -- half the fit
# floor's d_min coefficient (3->2) and cap fraction (0.8->0.6), proportioned
# roughly to n_eval/n_fit = 20/30 = 2/3 of the fit set's own size.
# =============================================================================

N_EVAL = 20  # 40% of N_EVAL_WORDS=50, S1.14's pinned 60/40 fit/eval split


def eval_set_diversity_check(table_keys):
    print()
    print("=" * 88)
    print(f"STEP 4 -- CA2-m2 eval-set diversity floor: n_eval={N_EVAL} words/trial, "
          f"{N_TRIALS} trials/group, bar = min(2*d_min(G), floor(0.6*|G|))")
    print("=" * 88)
    rng_master = np.random.default_rng(RNG_SEED + 2)
    header = f"{'Group':<6}{'|G|':>6}{'d_min':>6}{'bar=min(2d,.6|G|)':>19}{'eval p1':>9}{'eval mean':>11}{'margin':>9}"
    print(header)
    print("-" * len(header))
    for name, info in GROUPS.items():
        results = np.array([
            len({sample_word_result(rng_master, info["gens"], int(rng_master.integers(L_LO, L_HI + 1)))
                 for _ in range(N_EVAL)})
            for _ in range(N_TRIALS)
        ])
        bar = min(2 * info["dmin"], int(0.6 * info["order"]))
        p1 = np.percentile(results, 1)
        ok = p1 >= bar
        print(f"{name:<6}{info['order']:>6}{info['dmin']:>6}{bar:>19}{p1:>9.0f}"
              f"{results.mean():>11.2f}{(p1 - bar):>9.0f}  {'PASS' if ok else 'FAIL'}")
        assert ok, f"{name}: eval-set diversity floor 2*d_min not cleared at n_eval={N_EVAL}"


if __name__ == "__main__":
    a5_class_ok = verify_a5_generator_class()
    verify_closure()
    table = run_calibration()
    bars = pick_bars(table)
    print()
    print("Bars (fraction of |G|):", {k: v["frac"] for k, v in bars.items()})
    fit_set_diversity_check(table)
    eval_set_diversity_check(table)
    print()
    print("=" * 88)
    print(f"CA2-M1 A5 generator class: {'VERIFIED CORRECT, no bar changes' if a5_class_ok else 'MISMATCH -- see verdict above'}")
    print("=" * 88)
