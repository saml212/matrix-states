"""K-SCALING SINGLE SOURCE OF TRUTH -- NCR_KSCALING_DESIGN.md DRAFT-R0.

Every K-dependent quantity in the K-scaling sweep is DERIVED here, in ONE
place, from ONE integer: K_NCR (read from the NCR_K environment variable).
Nothing downstream is allowed to hand-set a second K-dependent constant.
This closes the gate memo's PATCH-required finding
(research/kscaling-novelty-2026-08-21.md leg 1): the pinned harness carries
`K_NCR = 24` and `D_NCR = 25` as two INDEPENDENT hand-set constants, so a
K change silently desynchronises the operator dimension from the number of
bindings.

Pure python -- no torch, no fla, no CUDA. Importable on any machine, which
is what lets the ladder derivation and all of its negative tests run as
plain unit tests off the box.

WHAT IS DERIVED HERE
--------------------
  K_NCR          the ONE free integer (env NCR_K, default 24)
  D_NCR          == K_NCR + 1                       (the ONLY definition)
  NCR_PARAM_EXACT  via ncr_param_exact(h_enc)        (re-derived per K)
  DEEP_LADDER    per-K, residue-verified (see below)
  FIXED_DIST_PROBE  the cross-K fixed-effective-distance control
  DOC_LEFT_PAD   the chunk_delta_rule T-floor pad    (MEASURED, see below)
  CHANCE         == 1.0 / K_NCR

THE LADDER (gate memo carried requirement 3: "fresh residue-verified deep
ladder per K -- no carried ladders")
--------------------------------------------------------------------------
Task-1's ground truth is a single Hamiltonian K-cycle (grammar_rd.py
`_permutation_graph`), so pi^h depends ONLY on h mod K. A deep-ladder depth
is therefore informative only if its residue is (a) non-zero (else pi^h is
the identity), (b) not a training residue ({1,2,3}), and (c) not shared
with another ladder depth (else two rungs silently measure the SAME ground
truth while the run reports them as two independent probes).

The pinned harness enforces (a) and (b) but NOT (c). Evaluated against every
K in this sweep, the pinned ladder (5,12,20,29,40,61) fails as follows --
reproduced exactly by derive_ladder()'s own negative tests:

    K=12  h=12 -> residue 0 (IDENTITY)          guard CRASHES
          h=61 -> residue 1 (train)             guard CRASHES
    K=16  29 and 61 BOTH -> residue 13          SILENT collision
    K=20  h=20,40 -> residue 0 (IDENTITY)       guard CRASHES
          h=61 -> residue 1 (train)             guard CRASHES
    K=24  5 and 29 BOTH -> residue 5            SILENT collision  (!)
    K=28  h=29 -> residue 1 (train)             guard CRASHES
          5/61 -> residue 5; 12/40 -> residue 12  SILENT collisions
    K=32  29 and 61 BOTH -> residue 29          SILENT collision

The K=24 line is NEW and is a disclosure about the harness of record, not
just about this sweep: the ladder the 55 existing K=24 cells were evaluated
on measures 5 distinct residues at 6 rungs. See NCR_KSCALING_DESIGN.md
sec 4.3 for how the K=24 anchor point of the curve is handled.

DERIVATION RULE (deterministic, hand-checkable, one rule for every K):

  Band profile.  Six rungs whose binexp_read SQUARING counts are
  (2,3,4,4,5,5) -- exactly the pinned K=24 ladder's own profile. Since
  n_squarings = floor(log2 h), that pins the six physical-depth bands to
  [4,7], [8,15], [16,31], [16,31], [32,63], [32,63]. Holding this profile
  FIXED across K is load-bearing: EXPERIMENT_LOG 2026-08-21 #3 result B
  measured a real monotone fp-depth DRIFT in the in-LM binexp read
  (1.0000 -> 0.9219 over 3->11 squarings), so an unmatched squaring count
  would confound the K axis with numerical depth.

  Top rung.  h_top(K) = the smallest h in [32,63] with h == K/2 (mod K).
  Residue K/2 is the antipodal point of the K-cycle -- the maximum possible
  effective distance from the identity, i.e. the hardest reachable query --
  and it is admissible for every K >= 12 (K/2 >= 6 > 3). Every K in this
  sweep is even, so K/2 is an integer.

  Rungs 1-5.  In band order, the smallest h in the band whose residue is
  admissible and not already used, strictly increasing in h.

CROSS-K CONFOUND AND ITS CONTROL
--------------------------------
At h_top the effective composition distance is K/2, so it GROWS with K:
6 hops at K=12 but 16 at K=32. The curve therefore measures capability
against the (K, d=K+1, effective-distance=K/2) triple, and a decline with K
could be breadth (more bindings) OR depth (longer composition).
FIXED_DIST_PROBE separates them: h_fix(K) = the smallest h in [32,63] with
h == 4 (mod K) -- effective distance 4 for EVERY K, at the SAME squaring
count (5) as h_top. It DELIBERATELY shares residue 4 with ladder rung 1
(which sits at squaring count 2), so it is kept OUT of DEEP_LADDER and
carried as its own labelled probe rather than being smuggled past the
distinctness assert. Same discipline as EXPERIMENT_LOG 2026-08-21 #2's
experiment B ("SAME ground truth by construction, labeled as such").

THE T-FLOOR PAD (MEASURED, not assumed)
---------------------------------------
The Task-1 document is doc_len = K*clause_len + query_len + 1 = 7K+7 tokens
(conv_size=4 -> buf_len=3 -> clause_len=7, query_len=6), and the backbone is
fed doc[:, :-1], i.e. T_in = 7K+6. lm_pretrain_rd.DeltaNetLMMixer.forward
hard-asserts T >= 128 (chunk_delta_rule's backward crashes below it;
F15-LM, measured 2026-07-02). MEASURED on this box 2026-08-21, fresh
process per T, batch 8, rung-1 backbone:

    T_in= 90 (K=12)  AssertionError  "sequence length 90 < _MIN_KERNEL_T=128"
    T_in=118 (K=16)  AssertionError  "sequence length 118 < _MIN_KERNEL_T=128"
    T_in=128         PASS  loss 0.999997  grad_norm 0.0726  peak 0.96 GB
    T_in=146 (K=20)  PASS  loss 0.999997  grad_norm 0.0726  peak 1.04 GB
    T_in=174 (K=24)  PASS  loss 0.999997  grad_norm 0.0725  peak 1.14 GB
    T_in=230 (K=32)  PASS  loss 0.999997  grad_norm 0.0725  peak 1.37 GB

So K=12 and K=16 are UNRUNNABLE unpadded -- a launch-losing crash on step 1,
not a quality question. The fix is a LEFT pad of DOC_LEFT_PAD inert BUFFER
tokens (the same reserved token grammar_rd already uses as intra-clause
filler, so no new token type enters the vocabulary), with every position
index shifted by the same amount. Minimum pad only:

    DOC_LEFT_PAD(K) = max(0, 128 - (7K+6))
    K=12 -> 38    K=16 -> 10    K=20,24,28,32 -> 0

K >= 20 is therefore BYTE-IDENTICAL to the pinned construction; in
particular K=24 stays bit-comparable with the 55 existing K=24 cells, which
is what lets them anchor the curve. DISCLOSED CONSEQUENCE: the resulting
T_in(K) = max(128, 7K+6) is flat at 128 for K in {12,16} and then linear,
i.e. sequence length co-varies with K with a kink at K=20. Recorded in
every output JSON as t_in / doc_left_pad; see NCR_KSCALING_DESIGN.md sec 7
for the pre-registered read if the curve shows a step at K=20.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# The ONE free parameter.
# --------------------------------------------------------------------------
SWEEP_K_GRID = (12, 16, 20, 24, 28, 32)
TRAIN_HOPS = (1, 2, 3)
MIN_KERNEL_T = 128                 # lm_pretrain_rd._MIN_KERNEL_T, measured above
CONV_SIZE = 4                      # RUNG1_BACKBONE["conv_size"] -- drives buf_len
N_LADDER_RUNGS = 6

# Squaring-count profile of the pinned K=24 ladder (5,12,20,29,40,61); see
# module docstring. floor(log2 h) == the band index below.
LADDER_BANDS = ((4, 7), (8, 15), (16, 31), (16, 31), (32, 63), (32, 63))
REFERENCE_K24_LADDER = (5, 12, 20, 29, 40, 61)   # the PINNED ladder, for negative tests only
FIXED_DIST_RESIDUE = 4             # the cross-K fixed-effective-distance control


def _k_from_env() -> int:
    raw = os.environ.get("NCR_K", "24")
    try:
        k = int(raw)
    except ValueError as e:
        raise ValueError(f"NCR_K={raw!r} is not an integer") from e
    assert k in SWEEP_K_GRID, (
        f"NCR_K={k} is not in the pre-registered K grid {SWEEP_K_GRID}. This sweep's "
        f"ladders, bands and param formula are derived and audited for those K only; "
        f"add K to SWEEP_K_GRID and re-run the ladder derivation + smoke before using it.")
    return k


# ==========================================================================
# THE derivation. D_NCR is defined HERE and NOWHERE ELSE.
# ==========================================================================
K_NCR: int = _k_from_env()
D_NCR: int = K_NCR + 1               # <-- the ONLY definition of d_ncr in the program
CHANCE: float = 1.0 / K_NCR


def ncr_param_exact(h_enc: int, k: int | None = None) -> int:
    """NCR_REAL_LM_DESIGN.md sec N2.2(c) parameter formula, RE-DERIVED per K.

    n = 40*h^2 + 4*d*h + 46*h + d, with d = k+1. The pinned harness hard-codes
    the K=24 value (173_209) in an assert; that assert is correct ONLY at
    K=24, so it is re-derived here and re-asserted against the MEASURED count
    at every K by the smoke (item 2)."""
    d = (k if k is not None else K_NCR) + 1
    return 40 * h_enc ** 2 + 4 * d * h_enc + 46 * h_enc + d


def integ_param_exact(d_model: int, k: int | None = None) -> int:
    """NCRIntegration = entity_adapter (d_model x d) + read_injector (d x d_model),
    both bias-free -- the sec G3-B12 single-shared-adapter wiring."""
    d = (k if k is not None else K_NCR) + 1
    return 2 * d_model * d


# --------------------------------------------------------------------------
# Document geometry + the measured T-floor pad.
# --------------------------------------------------------------------------
def buf_len(conv_size: int = CONV_SIZE) -> int:
    return max(1, conv_size - 1)


def doc_len(k: int | None = None, conv_size: int = CONV_SIZE) -> int:
    """grammar_rd geometry: K clauses of (buf..., KEY, REL, VALUE, '.') plus one
    query window (buf..., KEY, REL, <Q>) plus the answer token."""
    k = K_NCR if k is None else k
    b = buf_len(conv_size)
    clause_len, query_len = b + 4, b + 3
    return k * clause_len + query_len + 1


def t_in(k: int | None = None, conv_size: int = CONV_SIZE) -> int:
    """Length actually fed to the backbone (doc[:, :-1]), AFTER left-padding."""
    return doc_len(k, conv_size) - 1 + doc_left_pad(k, conv_size)


def doc_left_pad(k: int | None = None, conv_size: int = CONV_SIZE) -> int:
    """Inert BUFFER tokens prepended so T_in clears MIN_KERNEL_T. Minimum pad
    only -- 0 for every K >= 20, so those K stay byte-identical to the pinned
    construction (see module docstring)."""
    return max(0, MIN_KERNEL_T - (doc_len(k, conv_size) - 1))


# --------------------------------------------------------------------------
# Ladder derivation + the assert the pinned harness is missing.
# --------------------------------------------------------------------------
def n_squarings(h: int) -> int:
    """binexp_read's own count: the number of `base = base @ base` steps, i.e.
    floor(log2 h). Verified against ncr_models.binexp_read's loop structure."""
    n, hh = 0, h
    while hh > 0:
        hh >>= 1
        if hh > 0:
            n += 1
    return n


def n_applies(h: int) -> int:
    """binexp_read's own count of `v = base @ v` steps, i.e. popcount(h)."""
    return bin(h).count("1")


def admissible_residues(k: int) -> set[int]:
    """Residues a held-out depth may take: non-zero (not the identity) and not
    a training residue."""
    train = {h % k for h in TRAIN_HOPS}
    return {r for r in range(k) if r != 0 and r not in train}


def derive_ladder(k: int) -> tuple[int, ...]:
    """The module docstring's rule, executed. Deterministic; no search state."""
    adm = admissible_residues(k)
    assert len(adm) >= N_LADDER_RUNGS, (
        f"K={k}: only {len(adm)} admissible residues, cannot fill {N_LADDER_RUNGS} "
        f"residue-distinct rungs")
    assert k % 2 == 0, f"K={k} is odd -- the h_top rule needs an integer K/2"

    lo, hi = LADDER_BANDS[-1]
    top_cands = [h for h in range(lo, hi + 1) if h % k == k // 2]
    assert top_cands, f"K={k}: no h in [{lo},{hi}] with residue K/2={k//2}"
    h_top = min(top_cands)

    ladder: list[int | None] = [None] * N_LADDER_RUNGS
    ladder[-1] = h_top
    used = {h_top % k}
    prev = 0
    for i in range(N_LADDER_RUNGS - 1):
        lo, hi = LADDER_BANDS[i]
        pick = next((h for h in range(max(lo, prev + 1), hi + 1)
                     if h % k in adm and (h % k) not in used and h < h_top), None)
        assert pick is not None, f"K={k}: rung {i} (band {LADDER_BANDS[i]}) unfillable"
        ladder[i], prev = pick, pick
        used.add(pick % k)
    return tuple(ladder)                            # type: ignore[return-value]


def derive_fixed_dist_probe(k: int) -> int:
    """h_fix(K): smallest h in the TOP band with residue FIXED_DIST_RESIDUE.
    Same squaring count as h_top, effective distance 4 at every K."""
    lo, hi = LADDER_BANDS[-1]
    cands = [h for h in range(lo, hi + 1) if h % k == FIXED_DIST_RESIDUE]
    assert cands, f"K={k}: no h in [{lo},{hi}] with residue {FIXED_DIST_RESIDUE}"
    return min(cands)


# HAND-DERIVED TABLE (NCR_KSCALING_DESIGN.md sec 4.2 shows the derivation for
# every K). The literal table is what an auditor reads; assert_ladder_table()
# below proves derive_ladder() reproduces it, so a typo in either one fails
# loudly at import rather than silently changing what was evaluated.
LADDER_TABLE: dict[int, tuple[int, ...]] = {
    12: (4, 8, 17, 19, 33, 42),
    16: (4, 9, 21, 22, 39, 40),
    20: (4, 8, 16, 17, 32, 50),
    24: (4, 8, 16, 17, 33, 36),
    28: (4, 8, 16, 17, 33, 42),
    32: (4, 8, 17, 18, 37, 48),
}
FIXED_DIST_TABLE: dict[int, int] = {12: 40, 16: 36, 20: 44, 24: 52, 28: 32, 32: 36}


def assert_ladder_sound(ladder, k: int, train_hops=TRAIN_HOPS) -> None:
    """The pinned `_assert_ladder_sound` (identity + train-residue) PLUS the
    check it is missing: PAIRWISE residue distinctness. Without the third
    check the pinned ladder silently loses a rung at K in {16,24,32} -- two
    depths report as two probes while measuring one ground truth (gate memo
    leg 1). Also enforces strictly-increasing physical depth and a matched
    squaring profile, both of which the pinned guard leaves unchecked."""
    train_residues = {h % k for h in train_hops}
    residues = []
    for h in ladder:
        assert isinstance(h, int) and h >= 1, f"ladder depth {h!r} is not a positive int"
        r = h % k
        assert r != 0, (
            f"deep-ladder h={h} is IDENTITY mod K={k} (h%K=0) -- confounded, not held-out")
        assert r not in train_residues, (
            f"deep-ladder h={h} has h%K={r} colliding with a train-residue "
            f"{sorted(train_residues)} -- secretly in-distribution, not held-out")
        residues.append(r)
    dupes = sorted({r for r in residues if residues.count(r) > 1})
    assert not dupes, (
        f"deep-ladder {tuple(ladder)} at K={k} has PAIRWISE residue collisions at "
        f"{dupes} (residues {residues}) -- two rungs measure the SAME ground truth "
        f"while reporting as independent probes. This is the check the pinned "
        f"_assert_ladder_sound is missing (gate memo leg 1).")
    assert list(ladder) == sorted(ladder) and len(set(ladder)) == len(tuple(ladder)), (
        f"deep-ladder {tuple(ladder)} is not strictly increasing in physical depth")
    profile = tuple(n_squarings(h) for h in ladder)
    expected = tuple(n_squarings(lo) for lo, _ in LADDER_BANDS)
    assert profile == expected, (
        f"deep-ladder {tuple(ladder)} squaring profile {profile} != the pinned K=24 "
        f"reference profile {expected} -- the fp-depth axis (EXPERIMENT_LOG 2026-08-21 #3 "
        f"result B) would not be matched across K")


def assert_ladder_table() -> None:
    """Import-time proof that the literal table == the rule's output, for every
    K in the grid, and that each entry passes the strengthened soundness check."""
    for k in SWEEP_K_GRID:
        derived = derive_ladder(k)
        assert LADDER_TABLE[k] == derived, (
            f"K={k}: LADDER_TABLE says {LADDER_TABLE[k]} but derive_ladder() says "
            f"{derived} -- table and rule disagree")
        assert_ladder_sound(derived, k)
        fx = derive_fixed_dist_probe(k)
        assert FIXED_DIST_TABLE[k] == fx, (
            f"K={k}: FIXED_DIST_TABLE says {FIXED_DIST_TABLE[k]} but rule says {fx}")
        assert fx % k == FIXED_DIST_RESIDUE and n_squarings(fx) == n_squarings(derived[-1]), (
            f"K={k}: fixed-distance probe h={fx} is not (residue {FIXED_DIST_RESIDUE}, "
            f"squarings {n_squarings(derived[-1])})")
        assert derived[-1] % k == k // 2, f"K={k}: top rung is not at residue K/2"


assert_ladder_table()

DEEP_LADDER: tuple[int, ...] = LADDER_TABLE[K_NCR]
FIXED_DIST_PROBE: int = FIXED_DIST_TABLE[K_NCR]
H_TOP: int = DEEP_LADDER[-1]                 # the PRIMARY readout depth
EVAL_HOPS: tuple[int, ...] = DEEP_LADDER + (FIXED_DIST_PROBE,)


def provenance(h_enc: int | None = None, d_model: int = 768) -> dict:
    """The block every output JSON embeds, so a downstream reader never has to
    re-derive (or guess) what K, d, ladder, chance or pad a number came from."""
    rec = {
        "K": K_NCR, "d_ncr": D_NCR, "d_equals_k_plus_1": D_NCR == K_NCR + 1,
        "chance": CHANCE, "train_hops": list(TRAIN_HOPS),
        "deep_ladder": list(DEEP_LADDER),
        "deep_ladder_residues": [h % K_NCR for h in DEEP_LADDER],
        "deep_ladder_n_squarings": [n_squarings(h) for h in DEEP_LADDER],
        "deep_ladder_n_applies": [n_applies(h) for h in DEEP_LADDER],
        "h_top": H_TOP, "h_top_residue": H_TOP % K_NCR, "h_top_is_antipodal": H_TOP % K_NCR == K_NCR // 2,
        "fixed_dist_probe": FIXED_DIST_PROBE,
        "fixed_dist_probe_residue": FIXED_DIST_PROBE % K_NCR,
        "fixed_dist_probe_shares_residue_with_rung1": FIXED_DIST_PROBE % K_NCR == DEEP_LADDER[0] % K_NCR,
        "eval_hops": list(EVAL_HOPS),
        "doc_len": doc_len(), "doc_left_pad": doc_left_pad(), "t_in": t_in(),
        "min_kernel_T": MIN_KERNEL_T,
        "ladder_rule": "6 rungs, squaring profile (2,3,4,4,5,5); h_top = min h in [32,63] with h==K/2 mod K",
        "config_module": "kscaling_config.py",
    }
    if h_enc is not None:
        rec["ncr_param_exact"] = ncr_param_exact(h_enc)
        rec["integ_param_exact"] = integ_param_exact(d_model)
    return rec


if __name__ == "__main__":
    hdr = (f"{'K':>3} {'d':>3} {'1/K':>7} {'T_in':>5} {'pad':>4} | {'deep ladder':<26} | "
           f"{'residues mod K':<24} | {'n_sq':<16} | {'n_ap':<16} | {'h_fix':>5}")
    print(hdr)
    print("-" * len(hdr))
    for _k in SWEEP_K_GRID:
        _l = LADDER_TABLE[_k]
        print(f"{_k:>3} {_k+1:>3} {1.0/_k:>7.4f} {t_in(_k):>5} {doc_left_pad(_k):>4} | "
              f"{str(_l):<26} | {str([h % _k for h in _l]):<24} | "
              f"{str([n_squarings(h) for h in _l]):<16} | {str([n_applies(h) for h in _l]):<16} | "
              f"{FIXED_DIST_TABLE[_k]:>5}")
    print(f"\nncr_param_exact(h_enc=64) per K: "
          f"{ {k: ncr_param_exact(64, k) for k in SWEEP_K_GRID} }")
    print(f"integ_param_exact(d_model=768) per K: "
          f"{ {k: integ_param_exact(768, k) for k in SWEEP_K_GRID} }")
    print("\nassert_ladder_table(): PASS (ran at import)")
