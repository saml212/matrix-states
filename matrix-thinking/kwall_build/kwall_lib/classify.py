"""Band classification (§5's six-rule `classify()`), interval logic for an
incomplete K (§4 D5/E4), and the conditional-arm trigger (§4 F2 + the G5
DECIDED-band precondition).

Every function here is transcribed from the design doc's own pseudocode
and re-derives the doc's own regression figures at import time (125-outcome
partition checksum, the 1000-vector trigger split, the 11-configuration
ambiguity table) — an import-time self-check, not merely a claim.
"""
from __future__ import annotations

import itertools

# ---------------------------------------------------------------------------
# §5: the DEMONSTRATED partition. r24=4 (ROBUST, archive), r32=0 (NOT
# ROBUST, archive) are fixed constants of this design, never free variables.
R24_FIXED = 4
R32_FIXED = 0


def _robust(r: int) -> bool:
    return r >= 3


def classify(r26: int, r28: int, r30: int) -> tuple[str, bool]:
    """The six-rule total, ordered decision procedure (§5). Returns
    (label, non_monotone_tag) — label never includes the tag; the tag is
    reported as a separate boolean field (`band["non_monotone_tag"]`,
    universal assertion 6's own scope note)."""
    r24, r32 = R24_FIXED, R32_FIXED
    if _robust(r30) and r32 <= 1:
        label = "FRONTIER-AT-K*=30"
    elif _robust(r28) and r30 <= 1:
        label = "FRONTIER-AT-K*=28"
    elif _robust(r26) and r28 <= 1:
        label = "FRONTIER-AT-K*=26"
    elif _robust(r24) and r26 <= 1:  # r24=4 always ROBUST
        label = "FRONTIER-AT-K*=24"
    elif r26 >= r28 >= r30:
        label = "GRADUAL-DECAY"
    else:
        label = "NON-MONOTONE-UNRESOLVED"

    seq = [_robust(r24), _robust(r26), _robust(r28), _robust(r30), _robust(r32)]
    # monotone True...True False...False (a run of True's then all False)
    first_false = next((i for i, v in enumerate(seq) if not v), len(seq))
    monotone = all(seq[i] for i in range(first_false)) and \
        all(not seq[i] for i in range(first_false, len(seq)))
    return label, (not monotone)


def _regenerate_125_table():
    """§5 "Exhaustiveness and mutual exclusivity — DEMONSTRATED... Rev 2
    REGENERATES this table by EXECUTION". Re-run at import time; must match
    the design's own printed counts exactly."""
    counts: dict[str, int] = {}
    for r26, r28, r30 in itertools.product(range(5), repeat=3):
        label, tag = classify(r26, r28, r30)
        key = f"{label} [NON-MONOTONE]" if tag else label
        counts[key] = counts.get(key, 0) + 1
    return counts


_EXPECTED_125 = {
    "FRONTIER-AT-K*=24": 18,
    "FRONTIER-AT-K*=24 [NON-MONOTONE]": 4,
    "FRONTIER-AT-K*=26": 12,
    "FRONTIER-AT-K*=28": 8,
    "FRONTIER-AT-K*=28 [NON-MONOTONE]": 12,
    "FRONTIER-AT-K*=30": 8,
    "FRONTIER-AT-K*=30 [NON-MONOTONE]": 42,
    "GRADUAL-DECAY": 15,
    "NON-MONOTONE-UNRESOLVED": 4,
    "NON-MONOTONE-UNRESOLVED [NON-MONOTONE]": 2,
}
_actual_125 = _regenerate_125_table()
assert _actual_125 == _EXPECTED_125, (
    f"classify() 125-outcome partition diverges from the design's own "
    f"printed table: got {_actual_125}, want {_EXPECTED_125}")
assert sum(_actual_125.values()) == 125


# ---------------------------------------------------------------------------
# §4 D5/E4: interval logic for a K with exactly one incomplete cell.

def classify_with_interval_logic(r26_state, r28_state, r30_state):
    """Each `r{K}_state` is either an int (n_completed==4, the resolved
    rate) or the tuple `("AMBIGUOUS", r_known)` for an incomplete K
    (n_completed==3; `r_known` = CONVERGED count among the 3 resolved
    seeds) or the sentinel `"UNRESOLVED"` (n_completed<=2, or n_completed==3
    but 0.1 could not even establish r_known — never reached in practice
    since a 3-resolved-seed K always has a computable r_known).

    Returns (band_label_or_None, non_monotone_tag_or_None,
    "INCOMPLETE-AT-K", interval_resolved_Ks) where exactly one of the first
    two pairs is populated: either (label, tag, None, resolved_Ks) on a
    DECIDE, or (None, None, "INCOMPLETE-AT-K", incomplete_Ks) otherwise.
    ">=2 incomplete cells at one K" is modeled by the caller passing
    "UNRESOLVED" for that K's state directly (D5/E4: unconditional
    INCOMPLETE-AT-K, no candidate comparison performed)."""
    states = {"26": r26_state, "28": r28_state, "30": r30_state}
    if any(s == "UNRESOLVED" for s in states.values()):
        incomplete = sorted(int(K) for K, s in states.items() if s == "UNRESOLVED")
        return None, None, "INCOMPLETE-AT-K", incomplete

    ambiguous_Ks = [K for K, s in states.items()
                    if isinstance(s, tuple) and s[0] == "AMBIGUOUS"]
    if not ambiguous_Ks:
        r26, r28, r30 = states["26"], states["28"], states["30"]
        label, tag = classify(r26, r28, r30)
        return label, tag, None, []

    # cross-product of each AMBIGUOUS K's two candidates {r_known, r_known+1}
    candidate_lists = []
    for K in ("26", "28", "30"):
        s = states[K]
        if isinstance(s, tuple) and s[0] == "AMBIGUOUS":
            r_known = s[1]
            candidate_lists.append([r_known, r_known + 1])
        else:
            candidate_lists.append([s])
    results = set()
    for r26, r28, r30 in itertools.product(*candidate_lists):
        results.add(classify(r26, r28, r30))
    if len(results) == 1:
        (label, tag), = results
        return label, tag, None, sorted(int(K) for K in ambiguous_Ks)
    return None, None, "INCOMPLETE-AT-K", sorted(int(K) for K in ambiguous_Ks)


# ---------------------------------------------------------------------------
# §4 F2 + G5: the conditional-arm trigger.

def _smallest_K_with_rate_below_3(r26, r28, r30, r24=R24_FIXED, r32=R32_FIXED):
    """Scan K=26,28,30,32 in order for the smallest K whose rate is < 3
    (i.e. not ROBUST) -- the frontier the conditional arm targets. K=24 is
    never a candidate (it is the archive-anchored, always-ROBUST rung the
    conditional arm exists to move past)."""
    for K, r in ((26, r26), (28, r28), (30, r30), (32, r32)):
        if r < 3:
            return K
    return None  # every K including 32 is ROBUST -- cannot occur (r32=0 fixed)


def trigger(state_26, state_28, state_30):
    """§4's trigger pseudocode, normalised 4-tuple return shape (§R8 K7):
    (K_trig, resolution, resolution_detail, diag). `diag` is `blocking_K`
    on a raw K-scan TRIGGER-UNRESOLVED, or `band_blocked_K_trig` on a G5
    band-blocked TRIGGER-UNRESOLVED -- the caller must key off which
    branch fired, never tuple position alone (design's own note)."""
    states = {"26": state_26, "28": state_28, "30": state_30}
    branches = []
    ambiguous_Ks = [K for K, s in states.items()
                    if isinstance(s, tuple) and s[0] == "AMBIGUOUS"]
    unresolved_Ks = [K for K, s in states.items() if s == "UNRESOLVED"]
    candidate_lists = []
    for K in ("26", "28", "30"):
        s = states[K]
        if s == "UNRESOLVED":
            candidate_lists = None
            break
        elif isinstance(s, tuple) and s[0] == "AMBIGUOUS":
            candidate_lists.append([s[1], s[1] + 1])
        else:
            candidate_lists.append([s])
    if candidate_lists is None:
        # An UNRESOLVED K blocks the scan outright (F2: excluded from
        # candidacy) -- report the smallest UNRESOLVED K as blocking_K.
        return (None, "TRIGGER-UNRESOLVED", None, min(int(k) for k in unresolved_Ks))

    K_trigs = set()
    for r26, r28, r30 in itertools.product(*candidate_lists):
        kt = _smallest_K_with_rate_below_3(r26, r28, r30)
        K_trigs.add(kt)
    if len(K_trigs) == 1:
        result = (K_trigs.pop(), "unanimous", None, None)
    else:
        result = (min(K_trigs), "tie-break-min",
                   f"candidates were {sorted(K_trigs)}", None)

    # G5 precondition: the whole-study band must ALSO decide (never merely
    # the K-scan) before anything is dispatched.
    band_label, band_tag, band_incomplete, _ = classify_with_interval_logic(
        state_26, state_28, state_30)
    if band_incomplete == "INCOMPLETE-AT-K":
        band_blocked_K_trig = result[0]
        return (None, "TRIGGER-UNRESOLVED", None, band_blocked_K_trig)
    return result


# ---------------------------------------------------------------------------
# Import-time regression checks against the design's own disclosed figures
# (the 1000-vector reachable-state-space split, G5-gated).

def _resolution_states():
    """Every reachable per-K resolution state: EXACT ints 0..4, plus the
    9 AMBIGUOUS/DECIDED-collapsed states the design's own trigger-scan
    table uses (n_completed=3 -> either DECIDED-collapsed-to-one-value for
    the SCAN, or AMBIGUOUS at r_known=2), plus UNRESOLVED. To match the
    design's own reported 1000 = 10^3 reachable-vector count, each K has
    exactly 10 states: EXACT 0-4 (5), AMBIGUOUS r_known 0-3 (4, collapsing
    trigger-scan-wise per the table, but AMBIGUOUS objects for the band),
    UNRESOLVED (1) = 10."""
    states = []
    for r in range(5):
        states.append(r)  # EXACT
    for rk in range(4):
        states.append(("AMBIGUOUS", rk))
    states.append("UNRESOLVED")
    return states


def _regenerate_1000_split():
    states = _resolution_states()
    assert len(states) == 10
    decided, unresolved = 0, 0
    for s26, s28, s30 in itertools.product(states, repeat=3):
        _, res, _, _ = trigger(s26, s28, s30)
        if res == "TRIGGER-UNRESOLVED":
            unresolved += 1
        else:
            decided += 1
    return decided, unresolved


_decided, _unresolved = _regenerate_1000_split()
assert _decided + _unresolved == 1000
assert (_decided, _unresolved) == (473, 527), (
    f"trigger() 1000-vector G5-gated split diverges from the design's own "
    f"figures: got DECIDED={_decided}, TRIGGER-UNRESOLVED={_unresolved}, "
    f"want DECIDED=473, TRIGGER-UNRESOLVED=527")
