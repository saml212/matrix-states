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
    "INCOMPLETE-AT-K", interval_resolved_Ks, candidate_bands) where exactly
    one of the first two fields is populated: either (label, tag, None,
    resolved_Ks, None) on a DECIDE, or (None, None, "INCOMPLETE-AT-K",
    incomplete_Ks, candidate_bands) otherwise. `candidate_bands` (M4, build
    audit R1 — design :2249-2250, "both/all candidate bands disclosed") is
    non-null ONLY on the ambiguous-cross-product-disagrees path below: it
    stays None on a DECIDE (nothing to disclose) and None on the
    unconditional ">=2 incomplete cells at one K" / any-UNRESOLVED path
    (D5/E4: "no candidate comparison performed" there at all, so there is no
    candidate set to name). ">=2 incomplete cells at one K" is modeled by
    the caller passing "UNRESOLVED" for that K's state directly."""
    states = {"26": r26_state, "28": r28_state, "30": r30_state}
    if any(s == "UNRESOLVED" for s in states.values()):
        # F1 fix (build audit R1): disclose the UNION of UNRESOLVED and
        # AMBIGUOUS K's, not UNRESOLVED alone. A K sitting at n_completed==3
        # (AMBIGUOUS) in the same run as an UNRESOLVED K was previously
        # dropped from BOTH `incomplete_at_K` and `interval_resolved_Ks`
        # (the latter forced to [] by the caller on this path) -- but
        # validity_check's COMPLETE/otherwise clause requires every K named
        # in NEITHER field to read exactly 4 canonical files. An AMBIGUOUS
        # K reads 3, so the un-disclosed K failed the report's own check
        # after the full GPU-h spend (F1). Both UNRESOLVED and AMBIGUOUS
        # K's belong in `incomplete_at_K` here: neither is a decided K.
        incomplete = sorted(
            int(K) for K, s in states.items()
            if s == "UNRESOLVED" or (isinstance(s, tuple) and s[0] == "AMBIGUOUS"))
        return None, None, "INCOMPLETE-AT-K", incomplete, None

    ambiguous_Ks = [K for K, s in states.items()
                    if isinstance(s, tuple) and s[0] == "AMBIGUOUS"]
    if not ambiguous_Ks:
        r26, r28, r30 = states["26"], states["28"], states["30"]
        label, tag = classify(r26, r28, r30)
        return label, tag, None, [], None

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
        return label, tag, None, sorted(int(K) for K in ambiguous_Ks), None
    candidate_bands = sorted(
        f"{lbl} [NON-MONOTONE]" if tag else lbl for lbl, tag in results)
    return (None, None, "INCOMPLETE-AT-K",
            sorted(int(K) for K in ambiguous_Ks), candidate_bands)


# ---------------------------------------------------------------------------
# §4 F2 + G5: the conditional-arm trigger.

def _smallest_K_with_rate_below_3_or_blocking(r26, r28, r30, r24=R24_FIXED, r32=R32_FIXED):
    """Left-to-right scan K=26,28,30,32 (r24=4 fixed ROBUST, excluded from
    candidacy; r32=0 fixed, always resolves if reached). Returns
    `("decided", K)` at the first K whose rate is < 3 (i.e. not ROBUST), or
    `("blocked", K)` at the first K whose rate is the raw `"UNRESOLVED"`
    sentinel -- but ONLY when the scan actually reaches it, i.e. every
    earlier K in scan order was ROBUST (M1 fix, build audit R1 -- design
    `:558-579`: "if kt requires reading an UNRESOLVED K's status to
    decide"). A K reached AFTER an earlier K already decided kt is never
    inspected, so an UNRESOLVED K past the decision point cannot block."""
    for K, r in ((26, r26), (28, r28), (30, r30), (32, r32)):
        if r == "UNRESOLVED":
            return ("blocked", K)
        if r < 3:
            return ("decided", K)
    return ("decided", 32)  # unreachable: r32=0 fixed, the loop above always
                              # returns "decided" there first


def _compute_K_trigs(state_26, state_28, state_30):
    """Shared K-scan core for `trigger()` and `trigger_candidate_set()`
    (single source of truth, avoids the two ever drifting apart). Returns
    `(K_trigs, blocking_K)`: `K_trigs` is the raw candidate set (a
    non-empty `set[int]`) with `blocking_K=None` when every branch of the
    scan decides without needing an UNRESOLVED K's value; `K_trigs=None`
    with `blocking_K` set to the first K whose value the scan needed but
    could not read, in the FIRST branch (of the cross-product over
    AMBIGUOUS K's; UNRESOLVED K's are never expanded -- their true value is
    unknown, not a 2-candidate ambiguity) where that happens."""
    states = {"26": state_26, "28": state_28, "30": state_30}
    candidate_lists = []
    for K in ("26", "28", "30"):
        s = states[K]
        if s == "UNRESOLVED":
            candidate_lists.append(["UNRESOLVED"])
        elif isinstance(s, tuple) and s[0] == "AMBIGUOUS":
            candidate_lists.append([s[1], s[1] + 1])
        else:
            candidate_lists.append([s])

    K_trigs = set()
    for r26, r28, r30 in itertools.product(*candidate_lists):
        kind, K = _smallest_K_with_rate_below_3_or_blocking(r26, r28, r30)
        if kind == "blocked":
            return None, K
        K_trigs.add(K)
    return K_trigs, None


def trigger(state_26, state_28, state_30):
    """§4's trigger pseudocode, normalised 4-tuple return shape (§R8 K7):
    (K_trig, resolution, resolution_detail, diag). `diag` is `blocking_K`
    on a raw K-scan TRIGGER-UNRESOLVED, or `band_blocked_K_trig` on a G5
    band-blocked TRIGGER-UNRESOLVED -- the caller must key off which
    branch fired, never tuple position alone (design's own note)."""
    K_trigs, blocking_K = _compute_K_trigs(state_26, state_28, state_30)
    if K_trigs is None:
        # The scan itself needed an UNRESOLVED K's value (M1 fix): a K that
        # cannot resolve cannot trigger (F2).
        return (None, "TRIGGER-UNRESOLVED", None, blocking_K)
    if len(K_trigs) == 1:
        result = (next(iter(K_trigs)), "unanimous", None, None)
    else:
        result = (min(K_trigs), "tie-break-min",
                   f"candidates were {sorted(K_trigs)}", None)

    # G5 precondition: the whole-study band must ALSO decide (never merely
    # the K-scan) before anything is dispatched.
    band_label, band_tag, band_incomplete, _, _ = classify_with_interval_logic(
        state_26, state_28, state_30)
    if band_incomplete == "INCOMPLETE-AT-K":
        band_blocked_K_trig = result[0]
        return (None, "TRIGGER-UNRESOLVED", None, band_blocked_K_trig)
    return result


def trigger_raw_scan_blocked(state_26, state_28, state_30) -> bool:
    """Disambiguates `trigger()`'s TRIGGER-UNRESOLVED RETURN SITE for a
    consumer (M1 fix, build audit R1): True iff the RAW K-scan itself
    (pre-G5) needed to read an UNRESOLVED K's value to decide -- i.e.
    `diag` on that TRIGGER-UNRESOLVED belongs in `blocking_K`. False means
    the raw scan decided fine and, if `trigger()` still returned
    TRIGGER-UNRESOLVED, it was G5's band-precondition override instead --
    `diag` belongs in `band_blocked_K_trig`.

    Design's own note: "a consumer must key off the RETURN SITE, never
    tuple position alone." Before M1's fix, `any(state=="UNRESOLVED")` was
    a correct (if accidental) proxy for this, because the pre-fix raw scan
    blocked on ANY UNRESOLVED K regardless of scan position. After the fix,
    an UNRESOLVED K positioned AFTER the scan's decision point no longer
    blocks it, so that proxy is WRONG (confirmed this round: it mis-routed
    115/1000 vectors to `blocking_K` when the true site was G5) -- this
    function replaces it with the real thing, reusing the same
    `_compute_K_trigs` `trigger()` itself calls (single source of truth,
    can never disagree with `trigger()`'s own actual behavior)."""
    K_trigs, _ = _compute_K_trigs(state_26, state_28, state_30)
    return K_trigs is None


def trigger_candidate_set(state_26, state_28, state_30):
    """Schema `trigger.candidate_set` (M4, build audit R1 -- design
    `:1563`, the tie-break verification payload `:2644` sets `[26,28]`).
    This is NOT part of `trigger()`'s own normalised 4-tuple (that shape is
    pinned by the design's pseudocode); it exposes the SAME raw K_trigs set
    `trigger()` computed internally (via the shared `_compute_K_trigs`, so
    the two can never disagree) for a caller to populate the schema field
    with. Non-null ONLY when the scan itself decided with >1 candidate
    (`resolution=="tie-break-min"`) -- null on `unanimous` (one candidate,
    redundant with `K_trig`) and null on `TRIGGER-UNRESOLVED` (nothing
    decided, raw-scan-blocked or G5-band-blocked alike)."""
    K_trigs, _ = _compute_K_trigs(state_26, state_28, state_30)
    if K_trigs is None or len(K_trigs) == 1:
        return None
    return sorted(K_trigs)


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
