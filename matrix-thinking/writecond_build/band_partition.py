"""D5.2's NULL/WIN/PARTIAL partition, M3-repaired: the CONTROL_C-vs-PRIMARY
margin predicate added to the compB WIN clause (adopted verbatim by the
coordinator adjudication).

Source: NCR_WRITE_CONDITIONING_DESIGN.md D5.1 (per-recipe references) +
D5.2 (the partition, "PARTIAL as the set complement" -- closes M5(a)'s hole
STRUCTURALLY) + NCR_WRITECOND_ATTACK_R4.md M3 (CONTROL C is funded but
enters no adjudicating predicate; worked counter-example: a PRIMARY reading
of 0.7100 scores WIN identically whether CONTROL C reads 0.0664 -- ortho
removal did nothing -- or 0.7000 -- ortho removal alone did everything).

No box dependency: pure Python floats. CPU-testable, no torch needed.

Every threshold below is EXACTLY the report's own number -- TAU (0.09162),
CHANCE_WIN_FLOOR (0.19167 = chance 1/24 + 0.15), FRACTION_CLOSED_WIN (0.70),
GAP_WIN (0.15), C_MARGIN_WIN (0.15, M3's own repair number). No new
threshold is introduced.
"""
from __future__ import annotations

TAU = 0.09162                  # D5.1, global sampling-noise threshold at n=256
CHANCE_WIN_FLOOR = 0.19167     # chance(1/24=0.04167) + 0.15, D5.2
FRACTION_CLOSED_WIN = 0.70     # D5.2
GAP_WIN = 0.15                 # D5.2 (full_graft - backbone_only gap)
C_MARGIN_WIN = 0.15            # M3's repair: x - CONTROL_C(recipe) > 0.15, compB only

# D5.1's own raw anchors (verified against the raw JSONs by attack R4's V7,
# reproduced here verbatim -- NOT re-derived):
RECIPE_REFS = {
    "compB":   dict(P0_ref=0.0664, P1b_ref=0.9766),
    "compA":   dict(P0_ref=0.0350, P1b_ref=0.9961),
    "primary": dict(P0_ref=0.0390, P1b_ref=1.0000),
}


def fraction_closed(x: float, p0_ref: float, p1b_ref: float) -> float:
    return (x - p0_ref) / (p1b_ref - p0_ref)


def classify(x: float, recipe: str, gap_full_minus_backbone: float,
             control_c_reading: float | None = None) -> dict:
    """D5.2's partition, M3-repaired.

    x: retrieval24@61 (or the recipe's own PRIMARY signal reading).
    recipe: "compB" | "compA" | "primary".
    gap_full_minus_backbone: Band 3's own GAP(full_graft - backbone_only, h*).
    control_c_reading: CONTROL C's own retrieval24@61 reading, REQUIRED for
        compB (M3's repair: `x - CONTROL_C(recipe) > 0.15` is added to the
        compB WIN predicate -- CONTROL C is the only recipe with a funded,
        matched ortho-off/no-supervision cell, D2). For compA/primary, no
        matched CONTROL C cell is funded this wave -- a WIN there is
        retained under the ORIGINAL (pre-M3) predicate but the caller MUST
        read `ortho_confounded_disclosed=True` off the return value and
        label the verdict accordingly (the report's own second option,
        "pre-register explicitly that their WINs are ortho-confounded-
        and-disclosed", adopted here since no new Stage-1 cell is
        authorized this round).

    NULL and WIN are POSITIVE, mutually-exclusive-by-construction
    predicates (TAU=0.09162 < CHANCE_WIN_FLOOR=0.19167, so x<=TAU and
    x>CHANCE_WIN_FLOOR can never both hold, REGARDLESS of any additional
    AND-clause added to WIN -- additional clauses can only SHRINK WIN,
    never break this disjointness). PARTIAL = NOT NULL AND NOT WIN, the
    set complement -- true by construction for ANY well-defined,
    disjoint (NULL, WIN) pair, closing M5(a)'s hole structurally rather
    than patching an OR-clause (D5.2's own house discipline, unchanged
    here).
    """
    p0_ref = RECIPE_REFS[recipe]["P0_ref"]
    p1b_ref = RECIPE_REFS[recipe]["P1b_ref"]
    fc = fraction_closed(x, p0_ref, p1b_ref)

    null = x <= TAU
    win_base = (x > CHANCE_WIN_FLOOR) and (fc >= FRACTION_CLOSED_WIN) and (gap_full_minus_backbone > GAP_WIN)

    if recipe == "compB":
        assert control_c_reading is not None, "classify(): compB's WIN predicate requires control_c_reading (M3's repair)"
        c_margin_ok = (x - control_c_reading) > C_MARGIN_WIN
        win = win_base and c_margin_ok
        ortho_confounded_disclosed = False   # compB's confound is CLOSED by the margin predicate, not disclosed-around
    else:
        c_margin_ok = None
        win = win_base
        ortho_confounded_disclosed = win     # compA/primary: no matched Control C funded -- WIN carries the disclosure

    win = win and (not null)   # defense-in-depth; algebraically redundant given TAU < CHANCE_WIN_FLOOR (see docstring)
    partial = (not null) and (not win)

    label = "NULL" if null else ("WIN" if win else "PARTIAL")
    return dict(label=label, null=null, win=win, partial=partial, fraction_closed=fc,
                c_margin_ok=c_margin_ok, ortho_confounded_disclosed=ortho_confounded_disclosed,
                win_base_before_c_margin=win_base)
