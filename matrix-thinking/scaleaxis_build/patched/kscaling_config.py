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
    K=36  residues 5,12,20,29,4,25 -- SOUND     nothing to reject  (!)
    K=40  h=40 -> residue 0 (IDENTITY)          guard CRASHES

The K=36 row is a DISCLOSURE, not a convenience: K=36 is the one K in the
admitted grid where the pinned ladder happens to be sound (all six residues
distinct, none 0, none in {1,2,3}, profile (2,3,4,4,5,5) intact), because at
K > 32 only h=61 wraps and 61 mod 36 = 25 collides with nothing. The smoke's
item B (NEG: pinned ladder rejected) therefore has NOTHING to fire at K=36 and
is recorded N/A there with a POSITIVE soundness proof instead of a vacuous
"did not fire" FAIL. The pairwise-distinctness assert is still exercised at
K=36 by smoke item C, which builds a residue-colliding fixture on purpose.
This changes nothing about which ladder is USED -- derive_ladder() is still the
only source of DEEP_LADDER at every K.

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
    K=12 -> 38    K=16 -> 10    K >= 20 -> 0  (incl. the K=36/40 extension)

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
# K=36/40 FRONTIER EXTENSION (EXPERIMENT_LOG 2026-08-22 #3, design sec 14).
# Kept as a SEPARATE tuple rather than appended to SWEEP_K_GRID so that every
# existing reference to "the six K of the curve of record" keeps its meaning
# and no already-scored cell can be re-interpreted by this edit. Only the env
# guard and assert_ladder_table() read the union.
FRONTIER_K_GRID = (36, 40)
ADMITTED_K_GRID = SWEEP_K_GRID + FRONTIER_K_GRID
TRAIN_HOPS = (1, 2, 3)

# ---------------------------------------------------------------------------
# SCALE-AXIS PORT PATCH C1 (NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3.1-sec 3.4).
# THE ONE DICT, and everything that must re-derive from it.
#
# MIN_KERNEL_T PROVENANCE, sec 3.2 item 19 / m4 -- READ THIS BEFORE TRUSTING IT.
# 128 is a MEASURED constant and this module's own docstring records the
# measurement as taken at the RUNG-1 backbone, i.e. d_state = 64. It has NEVER
# been validated at d_state = 128, and the ported K=16 cell sits EXACTLY on the
# boundary (t_in = 128, zero margin). Stage A0.2 is a HARD pre-sweep gate that
# re-measures the floor at the resolved backbone before any K=16 cell is
# queued; scaleaxis_gates.py item B_MINKT is that gate, and it writes
# MIN_KERNEL_T_VALIDATED_AT into its record. Nothing here assumes it.
# ---------------------------------------------------------------------------
VOCAB_SIZE = 50257                 # GPT-2 BPE, lm_rd_rung_configs.py VOCAB_SIZE
VOCAB_RESERVED_EXTRA = 2           # grammar_rd's BUFFER + <Q> ids -> vocab_size_total 50259
PARAM_COUNT_TOLERANCE = 0.15       # lm_rd_rung_configs.py, unchanged by the port

# lm_rd_rung_configs.py RUNGS, verbatim. Rung 2 is the port target.
RUNGS: dict[int, dict] = {
    1: dict(d_model=768,  d_state=64,  n_layers=12, conv_size=4, num_heads=1, ffn_mult=4),
    2: dict(d_model=1536, d_state=128, n_layers=16, conv_size=4, num_heads=1, ffn_mult=4),
}
RUNG_OF_SCALE = {"98m": 1, "392m": 2}
BACKBONE_PARAM_TARGET_OF_SCALE = {"98m": 98_000_000, "392m": 392_000_000}
# sec 4.5: only these four K are ported. K=12/20/28/36 are deliberately NOT.
PORTED_K_GRID_392M = (16, 24, 32, 40)


def _scale_from_env() -> str:
    raw = os.environ.get("NCR_SCALE", "98m").strip().lower()
    assert raw in RUNG_OF_SCALE, (
        f"NCR_SCALE={raw!r} is not one of {sorted(RUNG_OF_SCALE)}. The backbone rung, the "
        f"param target and the ported-K restriction all resolve from it; refusing to guess.")
    return raw


SCALE: str = _scale_from_env()
RUNG: int = RUNG_OF_SCALE[SCALE]
BACKBONE: dict = dict(RUNGS[RUNG])                 # the resolved rung dict -- ONE definition
BACKBONE_PARAM_TARGET: int = BACKBONE_PARAM_TARGET_OF_SCALE[SCALE]
BACKBONE_PARAM_TOLERANCE: float = PARAM_COUNT_TOLERANCE

MIN_KERNEL_T = 128                 # lm_pretrain_rd._MIN_KERNEL_T, MEASURED AT d_state=64 (above)
MIN_KERNEL_T_MEASURED_AT_DSTATE = 64
CONV_SIZE = BACKBONE["conv_size"]  # sec 3.1's shadow constant, now READ FROM THE DICT rather
                                    # than hand-copied. It is invariant under this port (4 -> 4),
                                    # but it was a second source of truth and is now not one.
N_LADDER_RUNGS = 6


def backbone_param_exact(vocab: int, bb: dict | None = None) -> int:
    """NCR_SCALE_AXIS_DESIGN.md sec 3.4, executed.

      vocab*d_model                                  tied embedding / head
    + n_layers * ( 2*ffn_mult*d_model^2              FFN
                 + 4*d_model*d_state                 q,k,v,o
                 + 3*d_state*conv_size               short conv
                 + d_model                           beta projection
                 + 2*d_model + d_state )             2 norms at d_model, 1 head-norm at d_state
    + d_model                                        final norm

    VALIDATED AGAINST FOUR INDEPENDENTLY-MEASURED ENDPOINTS (sec 3.4):
      rung1/50257 -> 97,618,176  (fixscale_pilot_98m_off_1000.json "n_params")
      rung1/50259 -> 97,619,712  (every archived NCR cell's params.backbone)
      rung2/50257 -> 391,869,440 (fixscale_pilot_392m_off_1000.json "n_params")
      rung2/50259 -> 391,872,512 (what the NCR graft builds at rung 2)
    assert_param_table() below proves all four at import."""
    b = BACKBONE if bb is None else bb
    dm, ds, nl = b["d_model"], b["d_state"], b["n_layers"]
    per_layer = (2 * b["ffn_mult"] * dm * dm + 4 * dm * ds + 3 * ds * b["conv_size"]
                 + dm + (2 * dm + ds))
    return vocab * dm + nl * per_layer + dm


def total_param_exact(k: int | None = None, bb: dict | None = None,
                      h_enc: int = 64) -> int:
    """Total parameters PER ARM = backbone(vocab_size_total) + NCR head + INTEG.
    This is what sec 3.4's table states and what every spec's validity_check
    asserts (sec 3.2 item 15 -- gen_job_specs.PARAMS_PER_ARM's hard-coded 98M
    table is WRONG FOR EVERY K at 392M and is replaced by this function)."""
    b = BACKBONE if bb is None else bb
    kk = K_NCR if k is None else k
    return (backbone_param_exact(VOCAB_SIZE + VOCAB_RESERVED_EXTRA, b)
            + ncr_param_exact(h_enc, kk)
            + integ_param_exact(b["d_model"], kk))


def assert_param_table() -> None:
    """Import-time proof against the four measured endpoints of record."""
    checks = [(1, 50257, 97_618_176), (1, 50259, 97_619_712),
              (2, 50257, 391_869_440), (2, 50259, 391_872_512)]
    for rung, vocab, want in checks:
        got = backbone_param_exact(vocab, RUNGS[rung])
        assert got == want, (
            f"sec 3.4 formula gives {got:,} at rung {rung}/vocab {vocab}, but the MEASURED "
            f"count of record is {want:,} -- the port arithmetic is wrong, HALT.")


TOTAL_PARAM_TABLE_392M = {16: 392_095_889, 24: 392_122_521,
                          32: 392_149_153, 40: 392_175_785}   # sec 3.4's table, verbatim
TOTAL_PARAM_TABLE_98M = {12: 97_809_805, 16: 97_816_977, 20: 97_824_149,
                         24: 97_831_321, 28: 97_838_493, 32: 97_845_665,
                         36: 97_852_837, 40: 97_860_009}      # gen_job_specs MEASURED table

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
    assert k in ADMITTED_K_GRID, (
        f"NCR_K={k} is not in the pre-registered K grid {ADMITTED_K_GRID}. This sweep's "
        f"ladders, bands and param formula are derived and audited for those K only; "
        f"add K to FRONTIER_K_GRID and re-run the ladder derivation + smoke before using it.")
    return k


# ==========================================================================
# THE derivation. D_NCR is defined HERE and NOWHERE ELSE.
# ==========================================================================
K_NCR: int = _k_from_env()
D_NCR: int = K_NCR + 1               # <-- the ONLY definition of d_ncr in the program
CHANCE: float = 1.0 / K_NCR

# SCALE-AXIS PORT PATCH C2. At 392M only sec 4.5's four ported K exist. Without
# this, `NCR_K=12 NCR_SCALE=392m` would build a perfectly valid cell that no
# band, no reference table and no cross-scale stratum in the design covers.
if SCALE == "392m":
    assert K_NCR in PORTED_K_GRID_392M, (
        f"NCR_K={K_NCR} is not one of the FOUR ported K {PORTED_K_GRID_392M} "
        f"(NCR_SCALE_AXIS_DESIGN.md sec 4.5). K in {{12,20,28,36}} are deliberately NOT "
        f"ported: no 98M reference stratum, no band and no cross-scale test covers them.")


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
    # ---- K=36/40 FRONTIER EXTENSION (design sec 14) ------------------------
    # HAND-DERIVATION, K=36. Admissible residues = {4..35} (drop 0 and {1,2,3}).
    #   Top rung: smallest h in [32,63] with h == 36/2 = 18 (mod 36).
    #             32..53 give residues 32..35,0,1,...,17 -- none is 18;
    #             h=54 = 1*36 + 18  ->  residue 18 = K/2  ANTIPODAL. h_top=54.
    #   Rung 1  band [4,7]:   h=4  -> residue 4  admissible, unused.
    #   Rung 2  band [8,15]:  h=8  -> residue 8  admissible, unused.
    #   Rung 3  band [16,31]: h=16 -> residue 16 admissible, unused.
    #   Rung 4  band [16,31]: h=17 (must exceed 16) -> residue 17, unused.
    #   Rung 5  band [32,63]: h=32 -> residue 32 admissible, unused, < h_top.
    #   => (4, 8, 16, 17, 32, 54); residues (4, 8, 16, 17, 32, 18) -- 6 distinct,
    #      none 0, none in {1,2,3}, strictly increasing in h;
    #      floor(log2 h) = (2, 3, 4, 4, 5, 5) = the pinned profile.
    #   h_fix(36): smallest h in [32,63] with h == 4 (mod 36). 32..39 give
    #      32,33,...,39; h=40 = 1*36 + 4 -> residue 4, floor(log2 40)=5. h_fix=40.
    #
    # HAND-DERIVATION, K=40. Admissible residues = {4..39}.
    #   Top rung: smallest h in [32,63] with h == 40/2 = 20 (mod 40).
    #             32..59 give residues 32..39,0,1,...,19 -- none is 20;
    #             h=60 = 1*40 + 20  ->  residue 20 = K/2  ANTIPODAL. h_top=60.
    #   Rungs 1-5, same walk: 4 (r=4), 8 (r=8), 16 (r=16), 17 (r=17), 32 (r=32).
    #   => (4, 8, 16, 17, 32, 60); residues (4, 8, 16, 17, 32, 20) -- 6 distinct,
    #      none 0, none in {1,2,3}, strictly increasing;
    #      floor(log2 h) = (2, 3, 4, 4, 5, 5) = the pinned profile.
    #   h_fix(40): smallest h in [32,63] with h == 4 (mod 40). 32..39 -> 32..39;
    #      40..43 -> 0,1,2,3; h=44 = 1*40 + 4 -> residue 4, floor(log2 44)=5.
    #      h_fix=44.
    #
    # DISCLOSED RESIDUAL (extends design sec 4.2's existing n_applies note): both
    # top rungs have popcount 4 (54 = 0b110110, 60 = 0b111100), above the 2-3
    # spanned by K=12..32. n_squarings -- the axis the 2026-08-21 #3 result-B
    # fp-DRIFT finding actually implicates -- is still 5 at every K.
    36: (4, 8, 16, 17, 32, 54),
    40: (4, 8, 16, 17, 32, 60),
}
FIXED_DIST_TABLE: dict[int, int] = {12: 40, 16: 36, 20: 44, 24: 52, 28: 32, 32: 36,
                                    36: 40, 40: 44}


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
    for k in ADMITTED_K_GRID:
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
assert_param_table()          # SCALE-AXIS PORT PATCH C5: sec 3.4's four measured endpoints

DEEP_LADDER: tuple[int, ...] = LADDER_TABLE[K_NCR]
FIXED_DIST_PROBE: int = FIXED_DIST_TABLE[K_NCR]
H_TOP: int = DEEP_LADDER[-1]                 # the PRIMARY readout depth
EVAL_HOPS: tuple[int, ...] = DEEP_LADDER + (FIXED_DIST_PROBE,)


def provenance(h_enc: int | None = None, d_model: int | None = None) -> dict:
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
        # ---- SCALE-AXIS PORT PATCH C4 -------------------------------------
        # sec 3.2 item 22 (found by B1, NOT in the design's 21-item list):
        # `d_model` used to DEFAULT to the literal 768, and kscaling_battery.py
        # called provenance(64, 768) positionally. At 392M that silently
        # recorded HALF the true integ param count in every score record. The
        # default is now None and the integ count is refused rather than
        # guessed.
        "config_module_tree": os.path.dirname(os.path.abspath(__file__)),
        "scale": SCALE, "rung": RUNG, "backbone": dict(BACKBONE),
        "backbone_param_target": BACKBONE_PARAM_TARGET,
        "vocab_size": VOCAB_SIZE, "vocab_size_total": VOCAB_SIZE + VOCAB_RESERVED_EXTRA,
        "backbone_param_exact": backbone_param_exact(VOCAB_SIZE + VOCAB_RESERVED_EXTRA),
        "total_param_exact_per_arm": total_param_exact(),
        "min_kernel_T_measured_at_dstate": MIN_KERNEL_T_MEASURED_AT_DSTATE,
        "min_kernel_T_validated_at_this_dstate": (
            MIN_KERNEL_T_MEASURED_AT_DSTATE == BACKBONE["d_state"]),
    }
    if h_enc is not None:
        assert d_model is not None, (
            "provenance(h_enc, d_model): d_model is REQUIRED when h_enc is given. It used to "
            "default to the literal 768, which is silently WRONG at rung 2 (sec 3.2 item 22).")
        assert d_model == BACKBONE["d_model"], (
            f"provenance(d_model={d_model}) disagrees with the resolved backbone "
            f"d_model={BACKBONE['d_model']} -- one of the two is stale.")
        rec["ncr_param_exact"] = ncr_param_exact(h_enc)
        rec["integ_param_exact"] = integ_param_exact(d_model)
    return rec


if __name__ == "__main__":
    hdr = (f"{'K':>3} {'d':>3} {'1/K':>7} {'T_in':>5} {'pad':>4} | {'deep ladder':<26} | "
           f"{'residues mod K':<24} | {'n_sq':<16} | {'n_ap':<16} | {'h_fix':>5}")
    print(hdr)
    print("-" * len(hdr))
    for _k in ADMITTED_K_GRID:
        _l = LADDER_TABLE[_k]
        print(f"{_k:>3} {_k+1:>3} {1.0/_k:>7.4f} {t_in(_k):>5} {doc_left_pad(_k):>4} | "
              f"{str(_l):<26} | {str([h % _k for h in _l]):<24} | "
              f"{str([n_squarings(h) for h in _l]):<16} | {str([n_applies(h) for h in _l]):<16} | "
              f"{FIXED_DIST_TABLE[_k]:>5}")
    print(f"\nncr_param_exact(h_enc=64) per K: "
          f"{ {k: ncr_param_exact(64, k) for k in ADMITTED_K_GRID} }")
    print(f"integ_param_exact(d_model=768) per K: "
          f"{ {k: integ_param_exact(768, k) for k in ADMITTED_K_GRID} }")
    print("\nassert_ladder_table(): PASS (ran at import)")
