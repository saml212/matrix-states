"""phase2_hexachotomy.py -- REASONING_LINK_DESIGN.md sec 16.2.1's
pre-registered outcome classifier for a single (K, corpus, seed) Phase-2
familiarization trajectory. Pure Python, zero torch/fla dependency (this
module operates on already-computed booleans/floats -- CI bounds, deltas,
gate pass/fail flags -- never on tensors or checkpoints directly), so it is
fully unit-testable without a GPU, mirroring bands_pinned_frozenbias.py's
own "pure Python, no torch" discipline.

Implements the FIXED-PRECEDENCE, MECE-by-construction six-way hexachotomy
(sec 16.2.1, Rev 3's MAJOR-R3-2 fix; c1 anchor corrected Rev 4's MAJOR-R4-1;
totality count corrected Rev 4's MAJOR-R4-2) PLUS a seventh, build-time-
registered outcome this BUILD's own obligation (4) resolves an inherited
open item into: **UNRESOLVED-GATE** -- the disclosed pin for "Stage-0.5
never passes at any non-terminal checkpoint inside an otherwise-monotone
holds-true run" (sec 16.13's own "one pre-existing observation, disclosed"
paragraph, explicitly handed to this build round as a named open item
rather than silently rediscovered). See REASONING_LINK_DESIGN.md sec
16.2.1's own "Six pre-registered outcomes..." block for the full prose
this module implements mechanically -- every design citation below quotes
that block, not a re-derivation.

Mechanical primitives (`det`/`holds`/`det_arm`/`agree`) reuse the SAME CI
convention this program's `delta_ci_n3`/killer-prediction logic already
uses (a CI "excludes zero" iff both bounds share sign) -- this module does
not compute CIs itself (that is `reasoning_link_probe.delta_ci_n3`'s job,
torch-dependent); it consumes already-computed `(ci_low, ci_high)` pairs.
"""
from __future__ import annotations

import itertools

CHECKPOINTS = (250, 500, 1000, 2500, 5000)
NON_TERMINAL_CHECKPOINTS = (250, 500, 1000, 2500)
TERMINAL_CHECKPOINT = 5000

OUTCOMES = (
    "PERSISTENT", "TRANSIENT", "LATE-EMERGENT", "CONVERGED-EQUIVALENT",
    "UNRESOLVED", "NON-MONOTONE", "UNRESOLVED-GATE",
)


# ---------------------------------------------------------------------------
# Mechanical primitives (sec 16.2.1's own "Mechanical primitives" paragraph).
# ---------------------------------------------------------------------------

def det(ci_low: float, ci_high: float) -> bool:
    """`det(K,c)` / `det_arm(arm,c)` -- TRUE iff the pinned 3-seed CI EXCLUDES
    zero (both bounds share sign); FALSE ("indeterminate") iff the CI
    straddles zero. `det_arm` reuses this exact same primitive (sec
    16.2.1: "det_arm(arm,c) -- TRUE iff arm's own training-effect Delta(K=32,c)
    CI ... excludes zero") -- one function, two names at the call site for
    readability, never two implementations that could drift apart."""
    return (ci_low > 0.0) or (ci_high < 0.0)


det_arm = det  # sec 16.2.1: same primitive, different call-site name (arm's own effect vs. off)


def holds(det32: bool, det20: bool, abs_delta32: float, abs_delta20: float) -> bool:
    """The full killer-prediction pass condition at one checkpoint (sec 16.2.1's
    K-sweep paragraph, reapplied per-checkpoint): `det(32,c)=TRUE AND
    det(20,c)=FALSE AND |Delta(32,c)| > |Delta(20,c)|`."""
    return bool(det32 and (not det20) and (abs_delta32 > abs_delta20))


def agree(ci_a_low: float, ci_a_high: float, ci_b_low: float, ci_b_high: float) -> bool:
    """`agree(c)` -- TRUE iff the global-arm's and per-token-arm's own
    Delta(K=32,c) CIs OVERLAP each other (a direct interval-overlap test)."""
    return ci_a_low <= ci_b_high and ci_b_low <= ci_a_high


# ---------------------------------------------------------------------------
# The classifier itself.
# ---------------------------------------------------------------------------

def classify_trajectory(holds_by_c: dict, stage05_pass_by_c: dict,
                         det_arm_global_5000: bool, det_arm_per_token_5000: bool,
                         agree_5000: bool) -> dict:
    """Classifies ONE (K, corpus, seed) trajectory into exactly one of the
    seven outcomes (OUTCOMES above), fixed precedence order, per sec
    16.2.1's own registered algorithm.

    `holds_by_c`: {250: bool, 500: bool, 1000: bool, 2500: bool, 5000: bool}
      -- the killer-prediction pass condition, already computed via `holds()`
      above, at every trajectory checkpoint.
    `stage05_pass_by_c`: {250: bool, ...} -- the per-checkpoint Stage-0.5
      gate's own go/no-go (premises (iii)/(iv) AND h1 floor, both required;
      sec 16.2.1's MAJOR-4/MAJOR-R3-3 per-checkpoint gate), on the SAME
      familiarized OFF-arm checkpoint this trajectory's `holds` values were
      computed relative to.
    `det_arm_global_5000` / `det_arm_per_token_5000` / `agree_5000`: the
      TERMINAL-checkpoint-only quantities outcomes #4/#5 need (sec 16.2.1's
      own "read at the TERMINAL checkpoint (5,000, the best-powered point)"
      requirement).

    Returns {"outcome": <str>, "c1": <int|None>, "detail": <str>} -- `c1` is
    populated only for PERSISTENT (the checkpoint that serves as its early
    corroboration) and None for every other outcome.
    """
    assert set(holds_by_c) == set(CHECKPOINTS), (
        f"holds_by_c must have exactly the 5 registered checkpoints, got {sorted(holds_by_c)}")
    assert set(stage05_pass_by_c) == set(CHECKPOINTS), (
        f"stage05_pass_by_c must have exactly the 5 registered checkpoints, got {sorted(stage05_pass_by_c)}")

    h = holds_by_c
    g = stage05_pass_by_c

    # --- Outcome 1: PERSISTENT (c1 anchor per Rev-4's MAJOR-R4-1 fix; Stage-0.5-at-c1
    # re-check + skip-past-to-later-c1 per Rev-3's MAJOR-R3-3; UNRESOLVED-GATE pin, this
    # BUILD's obligation (4), for the case no valid c1 exists at all). ---
    # Step 1: find the FIRST non-terminal checkpoint where holds() is TRUE, in trajectory order.
    c1_raw = None
    for c in NON_TERMINAL_CHECKPOINTS:
        if h[c]:
            c1_raw = c
            break

    if c1_raw is not None:
        # Step 2: is this the start of a monotone holds-TRUE run all the way through 5,000?
        idx = CHECKPOINTS.index(c1_raw)
        monotone_run = all(h[c] for c in CHECKPOINTS[idx:])
        if monotone_run:
            # Step 3: within THIS monotone run, find the first NON-TERMINAL checkpoint (starting at
            # c1_raw, scanning forward) where Stage-0.5 ALSO passed -- sec 16.2.1's own "skipping
            # past any earlier holds-true-but-Stage-0.5-failed checkpoint" instruction. c1 must
            # remain non-terminal (the overarching tie-break rule); the terminal checkpoint (5,000)
            # is never eligible to serve as c1 even if it is the only one with Stage-0.5 passing.
            c1_candidates = [c for c in NON_TERMINAL_CHECKPOINTS if c >= c1_raw]
            c1_final = next((c for c in c1_candidates if g[c]), None)
            if c1_final is not None:
                return {"outcome": "PERSISTENT", "c1": c1_final,
                        "detail": (f"monotone holds-TRUE run from {c1_raw} through {TERMINAL_CHECKPOINT}; "
                                   f"Stage-0.5 passed at c1={c1_final}"
                                   + ("" if c1_final == c1_raw else
                                      f" (re-identified past c1_raw={c1_raw}, whose own Stage-0.5 failed)"))}
            # --- UNRESOLVED-GATE pin (this BUILD's obligation (4), resolving the inherited open
            # item sec 16.13 disclosed: "the Stage-0.5 skip-past provision does not state what
            # happens if Stage-0.5 never passes at ANY non-terminal checkpoint within an otherwise-
            # confirmed monotone holds-true run"). PINNED HERE: the trajectory is reported
            # UNRESOLVED-GATE -- a named, disclosed outcome DISTINCT from the six-way hexachotomy
            # (never silently folded into PERSISTENT, NON-MONOTONE, or UNRESOLVED), per-seed
            # disclosure ONLY, NEVER averaged into a headline number alongside cleanly-classified
            # seeds -- mirrors NON-MONOTONE's own disclosure discipline exactly (sec 16.2.1's
            # "every NON-MONOTONE trajectory triggers a mandatory seed-level disclosure ... rather
            # than being folded into a percentage or a mean effect size").
            return {"outcome": "UNRESOLVED-GATE", "c1": None,
                    "detail": (f"monotone holds-TRUE run from {c1_raw} through {TERMINAL_CHECKPOINT} "
                               f"confirmed, but Stage-0.5 FAILED at every non-terminal checkpoint in "
                               f"that run ({c1_candidates}) -- no checkpoint can serve as a valid, "
                               f"gate-passing c1. The arm-contrast itself may be real, but this "
                               f"instrument cannot certify it as interpretable at any point in the "
                               f"run; report UNRESOLVED-GATE, never PERSISTENT.")}

    # --- Outcome 2: TRANSIENT -- holds(c)=TRUE for >=1 c in the first four, holds(5000)=FALSE. ---
    if any(h[c] for c in NON_TERMINAL_CHECKPOINTS) and not h[TERMINAL_CHECKPOINT]:
        return {"outcome": "TRANSIENT", "c1": None,
                "detail": "holds() fired at >=1 non-terminal checkpoint but not at 5,000"}

    # --- Outcome 3: LATE-EMERGENT -- holds(5000)=TRUE, holds(c)=FALSE at every earlier checkpoint. ---
    if h[TERMINAL_CHECKPOINT] and not any(h[c] for c in NON_TERMINAL_CHECKPOINTS):
        return {"outcome": "LATE-EMERGENT", "c1": None,
                "detail": "holds() TRUE only at the terminal (5,000) checkpoint"}

    # --- Outcomes 4/5: holds(c)=FALSE at EVERY checkpoint -- terminal-checkpoint-only split. ---
    if not any(h[c] for c in CHECKPOINTS):
        if det_arm_global_5000 and det_arm_per_token_5000 and agree_5000:
            return {"outcome": "CONVERGED-EQUIVALENT", "c1": None,
                     "detail": "holds() never fires; both arms show a determinate, mutually-"
                               "indistinguishable training effect at the terminal checkpoint"}
        sub_case = ("power problem (>=1 arm's own effect vs. off never clears noise)"
                    if not (det_arm_global_5000 and det_arm_per_token_5000)
                    else "real-but-differently-shaped effect (both arms determinate, but disagree "
                         "with each other -- agree(5000)=FALSE)")
        return {"outcome": "UNRESOLVED", "c1": None,
                "detail": f"holds() never fires and the terminal CONVERGED-EQUIVALENT condition does "
                          f"not hold in full -- sub-case: {sub_case}"}

    # --- Outcome 6: NON-MONOTONE -- the catch-all (everything else, by construction). ---
    return {"outcome": "NON-MONOTONE", "c1": None,
            "detail": "holds() fires at >=1 checkpoint but in a shape none of PERSISTENT/TRANSIENT/"
                      "LATE-EMERGENT/CONVERGED-EQUIVALENT/UNRESOLVED can honestly claim -- "
                      "inconclusive-without-rerun; report this seed's own holds(c) pattern verbatim, "
                      "never averaged into a headline number"}


# ---------------------------------------------------------------------------
# Totality self-test: every one of the 2^5=32 holds(c) truth-assignments maps to EXACTLY one
# outcome (sec 16.2.1's own "checked exhaustively, not merely asserted" totality paragraph,
# 1+15+1+4+11=32). Stage-0.5 is held ALWAYS-PASSING here (this enumeration is scoped to the base
# hexachotomy's own MECE property over `holds` patterns alone, matching the design's own worked
# table, which is holds-pattern-only) -- the UNRESOLVED-GATE branch is a SEPARATE, additional
# dimension (Stage-0.5 pass/fail at the candidate c1's), exercised by its own dedicated test
# (see phase2_stage_minus1.py) rather than folded into this 32-pattern count, which would change
# the registered 1+15+1+4+11=32 arithmetic this design's own Rev-4/Rev-5 audits verified byte-
# accurate.
# ---------------------------------------------------------------------------

def totality_check() -> dict:
    """Enumerates all 32 `holds(c)` boolean patterns across the 5 checkpoints (Stage-0.5 held
    always-passing), classifies each, and returns counts per outcome + the full pattern->outcome
    map -- for the Stage -1 self-test to assert against the registered 1+15+1+4+11=32 split."""
    counts = {o: 0 for o in OUTCOMES}
    pattern_to_outcome = {}
    always_pass_gate = {c: True for c in CHECKPOINTS}
    for pattern in itertools.product([False, True], repeat=5):
        holds_by_c = dict(zip(CHECKPOINTS, pattern))
        # terminal-only quantities matter only for the all-FALSE pattern (outcomes 4/5); use a
        # fixed "both arms determinate + agree" reading for the totality count (matches the design's
        # own single all-FALSE-pattern row, which routes to CONVERGED-EQUIVALENT under that reading).
        result = classify_trajectory(holds_by_c, always_pass_gate,
                                      det_arm_global_5000=True, det_arm_per_token_5000=True,
                                      agree_5000=True)
        counts[result["outcome"]] += 1
        pattern_to_outcome[pattern] = result["outcome"]
    return {"counts": counts, "pattern_to_outcome": pattern_to_outcome, "total": sum(counts.values())}


if __name__ == "__main__":
    r = totality_check()
    print("phase2_hexachotomy totality check:", r["counts"], "total =", r["total"])
    expected = {"PERSISTENT": 4, "TRANSIENT": 15, "LATE-EMERGENT": 1, "CONVERGED-EQUIVALENT": 1,
                "UNRESOLVED": 0, "NON-MONOTONE": 11, "UNRESOLVED-GATE": 0}
    assert r["counts"] == expected, f"MISMATCH: {r['counts']} != {expected}"
    print("MATCHES sec 16.2.1's registered 1+15+1+4+11=32 split "
          "(PERSISTENT=4, TRANSIENT=15, LATE-EMERGENT=1, CONVERGED-EQUIVALENT=1, "
          "NON-MONOTONE=11) -- ALL PASSED")
