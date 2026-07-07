"""mech_schema.py -- FROZEN_BIAS_LM_DESIGN.md sec 12.3.1's shared schema helper,
binding on EVERY sec 12 (MECHANISM-WAVE) JSON artifact -- Stage 0, 0.5, 1, and 2
alike, not just the H2 reharvest this subsection was written for (MAJOR-2 fix).

Every sec 12 script must persist its output through `wrap_exploratory()` before
writing to disk. This enforces three things, all load-bearing for sec 12.0's own
framing ("explicitly EXPLORATORY/MECHANISTIC... nothing in sec 12 can claim the
blind-pin discipline sec 7.3/sec 10 built for the primary bars"):

1. Every persisted JSON carries a top-level
   `"tier": "exploratory-mechanism-wave -- NOT a confirmatory bar"` field,
   unconditionally, so no downstream reader (human or script) can mistake a
   sec 12 artifact for a sec 7-9 confirmatory one from its shape alone.
2. `derive_estimation()`'s own `confirm_direction_consistent` field (the word
   "confirm" is exactly the kind of shape-level leak sec 12.0 disclaims) gets a
   `direction_consistent_with_hypothesis` field added alongside it, wherever it
   appears, at any nesting depth. Per sec 12.3.1 item 2's own text ("renamed (OR
   wrapped under an added key, leaving the original in place for traceability)"
   -- both are licensed fixes), this module takes the wrap-and-keep option: the
   raw `derive_estimation()` output stays inspectable for traceability, and the
   non-confirmatory-sounding name is what every sec 12 script/report should read
   from.
3. `headline_verdict()` (fit_frozenbias_estimation.py) must NEVER be called
   anywhere in sec 12 -- not in Stage 0, 1, or 2, not even "just to see." This
   module enforces that two ways: (a) it does NOT import
   `fit_frozenbias_estimation` at all (see the guard comment below -- do not add
   that import to this file), and (b) `wrap_exploratory()` refuses to persist
   any payload that structurally matches `headline_verdict()`'s own return shape
   (a dict carrying both "verdict" and "per_corpus" keys), in case that shape
   ever reaches this module via some other caller's import.
"""
from __future__ import annotations

import copy

TIER = "exploratory-mechanism-wave -- NOT a confirmatory bar"

# --------------------------------------------------------------------------
# DO NOT import `fit_frozenbias_estimation.headline_verdict` (or the module
# itself) anywhere in this file. Sec 12.3.1 item 3 is explicit: that function
# must never be called anywhere in sec 12. `derive_estimation` is fine to
# import in CALLING scripts (build_fit_inputs_rankstats.py does) -- it is
# `headline_verdict` specifically that is forbidden, because its entire job is
# to emit a CONFIRM/INCONCLUSIVE/ESTIMATION verdict under blind-pin discipline
# that sec 12.0 already establishes does not apply here.
# --------------------------------------------------------------------------


def _looks_like_headline_verdict(d) -> bool:
    """Structural fingerprint of `headline_verdict()`'s own return dict
    (`{"per_corpus": {...}, "verdict": "CONFIRM"|"INCONCLUSIVE (...)"|"ESTIMATION (...)"}`).
    If this shape shows up anywhere in a sec 12 payload, something imported and
    called the forbidden function -- persistence must refuse, not silently
    write it."""
    return isinstance(d, dict) and "verdict" in d and "per_corpus" in d


def _rename_confirm_field(obj) -> None:
    """Recursively walk obj (dicts/lists) IN PLACE, and wherever a dict carries
    `confirm_direction_consistent` (derive_estimation()'s own field name), add
    `direction_consistent_with_hypothesis` alongside it with the same value.
    Also asserts, at every nesting level visited, that nothing matches
    `headline_verdict()`'s own return shape."""
    if isinstance(obj, dict):
        assert not _looks_like_headline_verdict(obj), (
            "mech_schema: payload contains a dict shaped like headline_verdict()'s "
            "own return value ('verdict' + 'per_corpus' keys) -- headline_verdict() "
            "must NEVER be called anywhere in sec 12 (FROZEN_BIAS_LM_DESIGN.md "
            "sec 12.3.1 item 3). Refusing to persist.")
        if "confirm_direction_consistent" in obj and "direction_consistent_with_hypothesis" not in obj:
            obj["direction_consistent_with_hypothesis"] = obj["confirm_direction_consistent"]
        for v in obj.values():
            _rename_confirm_field(v)
    elif isinstance(obj, list):
        for v in obj:
            _rename_confirm_field(v)


def wrap_exploratory(payload: dict) -> dict:
    """FROZEN_BIAS_LM_DESIGN.md sec 12.3.1's schema requirement (MAJOR-2 fix).
    Returns a NEW dict (does not mutate the caller's payload in place) with
    `"tier"` inserted first (so it reads first in any printed/serialized JSON),
    every `confirm_direction_consistent` field wrapped per item 2 above, and a
    hard refusal if the payload (at any depth) already looks like
    `headline_verdict()`'s own output.

    Every sec 12 script's JSON-writing call site must route its payload through
    this function before `json.dump(...)`.
    """
    assert isinstance(payload, dict), "wrap_exploratory expects a dict payload"
    out = copy.deepcopy(payload)
    assert not _looks_like_headline_verdict(out), (
        "mech_schema.wrap_exploratory: top-level payload itself is shaped like "
        "headline_verdict()'s own return value -- refusing to persist.")
    _rename_confirm_field(out)
    wrapped = {"tier": TIER}
    wrapped.update(out)
    return wrapped


# ---------------------------------------------------------------------------
# Self-test (run directly: `python mech_schema.py`) -- exercises all three
# schema requirements plus the negative case (headline_verdict-shaped payload
# must be refused).
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    ok = True

    print("[test 1] tier field added unconditionally")
    payload = {"foo": "bar"}
    wrapped = wrap_exploratory(payload)
    t1 = wrapped.get("tier") == TIER and wrapped["foo"] == "bar" and "tier" not in payload
    print(f"    tier={wrapped.get('tier')!r} original_untouched={'tier' not in payload}  PASS={t1}")
    ok = ok and t1

    print("[test 2] confirm_direction_consistent wrapped (added alongside, original kept), nested")
    payload2 = {
        "post_blend_primary": {
            "openr1-mix-ext": {"mean_delta": -2.09, "confirm_direction_consistent": True},
        },
        "list_of_results": [
            {"confirm_direction_consistent": False, "corpus": "wikitext-mix-ext"},
        ],
    }
    wrapped2 = wrap_exploratory(payload2)
    inner = wrapped2["post_blend_primary"]["openr1-mix-ext"]
    inner_list = wrapped2["list_of_results"][0]
    t2 = (
        inner["direction_consistent_with_hypothesis"] is True
        and inner["confirm_direction_consistent"] is True
        and inner_list["direction_consistent_with_hypothesis"] is False
        and "confirm_direction_consistent" in payload2["post_blend_primary"]["openr1-mix-ext"]
    )
    print(f"    nested + list rename applied, original field retained  PASS={t2}")
    ok = ok and t2

    print("[test 3] headline_verdict-shaped payload (top-level) is REFUSED")
    fake_headline = {"per_corpus": {"openr1-mix-ext": {"both_confirm": True}}, "verdict": "CONFIRM"}
    raised = False
    try:
        wrap_exploratory(fake_headline)
    except AssertionError:
        raised = True
    print(f"    raised AssertionError: {raised}  PASS={raised}")
    ok = ok and raised

    print("[test 4] headline_verdict-shaped payload NESTED inside a larger dict is also REFUSED")
    fake_nested = {"outer": {"inner": {"per_corpus": {}, "verdict": "ESTIMATION (...)"}}}
    raised2 = False
    try:
        wrap_exploratory(fake_nested)
    except AssertionError:
        raised2 = True
    print(f"    raised AssertionError: {raised2}  PASS={raised2}")
    ok = ok and raised2

    print("[test 5] wrap_exploratory does not mutate the caller's original payload")
    payload5 = {"confirm_direction_consistent": True}
    wrap_exploratory(payload5)
    t5 = "direction_consistent_with_hypothesis" not in payload5
    print(f"    caller's dict unchanged: {t5}  PASS={t5}")
    ok = ok and t5

    return ok


if __name__ == "__main__":
    import sys
    passed = _self_test()
    print("=" * 60)
    print("mech_schema SELF-TEST: " + ("ALL PASSED" if passed else "FAILURES PRESENT"))
    sys.exit(0 if passed else 1)
