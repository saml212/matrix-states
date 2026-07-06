"""bands_pinned_frozenbias.py -- FROZEN_BIAS_LM_DESIGN.md sec 7.3's
`BANDS_PINNED-FrozenBias.json` writer/validator, mirroring the house
`key_anchoring.py` BANDS_PINNED pattern (`write_bands_pinned` /
`validate_bands_pinned` / `assert_blind_not_broken`) exactly -- same
sha256-hash-of-referenced-result-JSONs discipline, same "re-derive from the
hashed inputs and require value-identity with the stored pin" tamper-
evidence discipline (the e633862-audit F2 fix `key_anchoring.validate_
bands_pinned` already applies), same strict-precedes timestamp assertion.
Cloned, not cross-imported for the top-level write/validate/assert
functions (this codebase's own pod-safety convention -- this file DOES
import `key_anchoring.sha256_of_file` directly, since that one helper has
no fla/torch dependency at all and hashing raw file bytes is not a
mechanism worth re-deriving).

This program's own pinned quantities (sec 7.1/sec 7.1a/sec 7.2/sec 7.3,
ALL computed from Arm 1's own fresh rung-1 data, BEFORE any Arm 2/Arm 2'
(TRAINED) result is inspected):
  - Arm 1's own per-seed val loss mean/std, per corpus (sec 7.2's derived
    val-loss tolerance gate).
  - Arm 1'/Arm 1'''s own freshly-measured post-blend span_frac mean/std,
    per corpus (sec 7.1/sec 7.1a's primary bar + control reference).
  - Arm 1's own pre-blend k_raw span_frac mean/std, per corpus (sec 4.a-i's
    co-primary reference).

PINNED CI FORMULA (sec 7.1-real.1, ROUND-5 fix 1 -- the SINGLE operative
formula for every two-sided effect-existence claim in this document):
  mean_delta +/- t(n-1, 0.975) * s / sqrt(n),  n=3 seeds -> t(2,0.975)=4.303.
This module's `derive_frozenbias_bands` computes exactly this quantity (NOT
the flat k=2*s_ref the pre-round-5 text used) for every span_frac reference
band; sec 7.2's val-loss gate is the ONE exception (its own k=2*s_ref,
one-sided tolerance, disclosed explicitly as NOT adopting the pinned
t-formula -- see sec 7.2's own ROUND-5 disclosure) and is computed by a
SEPARATE function (`derive_val_loss_tolerance`) so the two statistical
objects are never silently conflated in one derivation.
"""
from __future__ import annotations

import json
import os
import time

from key_anchoring import sha256_of_file   # the ONE dependency-free helper worth reusing directly

# t(df, 0.975) for df=2 (n=3 seeds) -- sec 7.1-real.1's pinned two-sided 95% small-sample multiplier.
# A single literal constant, not computed from scipy (this codebase's CPU dev-box precedent already
# established: no scipy dependency for this exact quantity, sim_frozen_bias_training_mediated.py's
# own README/comments cite the same numeric value from a standard t-table).
T_DIST_975_DF2 = 4.303
N_SEEDS = 3


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Sample mean and sample std (n-1 denominator, i.e. `unbiased=True`/df=n-1) -- the SAME
    convention key_anchoring.derive_engaged_bands and every archived span_frac std in this
    program's own sec 7.1-real table use (`torch.Tensor.std(unbiased=True)`, reimplemented here in
    pure Python so this module never needs torch at all -- it operates on already-extracted floats,
    not tensors)."""
    n = len(values)
    assert n >= 2, f"sample std requires n>=2, got n={n}"
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def pinned_ci_half_width(s: float, n: int = N_SEEDS, t_mult: float = T_DIST_975_DF2) -> float:
    """sec 7.1-real.1's pinned formula: `t(n-1,0.975) * s / sqrt(n)`. This IS the half-width of the
    two-sided CI around a measured mean_delta; it is ALSO, independently, the "pinned threshold" a
    fixed-baseline reading (mean_ref +/- this half-width) uses -- both readings share this one
    function so the two never silently drift apart."""
    return t_mult * s / (n ** 0.5)


def derive_frozenbias_span_frac_bands(per_corpus_values: dict) -> dict:
    """sec 7.1/sec 7.1a/sec 4.a-i: for EACH corpus, computes {mean, s (sample std), n,
    pinned_ci_half_width, ci_lower, ci_upper} from that corpus's n=3 per-seed span_frac values.
    `per_corpus_values`: {corpus_name: [v0, v1, v2]} -- caller's job to have already validated each
    list has exactly N_SEEDS entries from genuinely COMPLETE, hash-matched result JSONs (this
    function does not re-open any file; write_bands_pinned_frozenbias does that and passes the
    already-extracted floats here, mirroring key_anchoring.derive_engaged_bands' own
    separation-of-concerns)."""
    out = {}
    for corpus, vals in per_corpus_values.items():
        assert len(vals) == N_SEEDS, (
            f"corpus={corpus!r}: band derivation requires exactly {N_SEEDS} seeds, got {len(vals)}")
        mean, s = _mean_std(vals)
        half_width = pinned_ci_half_width(s)
        out[corpus] = {
            "mean": mean, "s": s, "n": N_SEEDS,
            "t_mult": T_DIST_975_DF2, "pinned_ci_half_width": half_width,
            "ci_lower": mean - half_width, "ci_upper": mean + half_width,
            "per_seed": list(vals),
        }
    return out


def derive_val_loss_tolerance(per_corpus_values: dict, k: float = 2.0) -> dict:
    """sec 7.2's derived val-loss tolerance -- DELIBERATELY the flat `k=2*s_ref` (raw per-seed std,
    NOT the pinned t-corrected/SE-scaled formula above): a ONE-SIDED tolerance on how much WORSE
    Arm 2's loss is allowed to get, a different statistical object than the two-sided CI-excludes-
    zero claims the pinned formula governs (sec 7.2's own ROUND-5 disclosure, restated here so this
    module's own two derivation functions can never be confused for one another)."""
    out = {}
    for corpus, vals in per_corpus_values.items():
        assert len(vals) == N_SEEDS, (
            f"corpus={corpus!r}: val-loss tolerance requires exactly {N_SEEDS} seeds, got {len(vals)}")
        mean, s = _mean_std(vals)
        tolerance_abs = k * s
        out[corpus] = {
            "mean": mean, "s": s, "n": N_SEEDS, "k": k, "tolerance_abs": tolerance_abs,
            "pass_ceiling": mean + tolerance_abs, "per_seed": list(vals),
        }
    return out


# ---------------------------------------------------------------------------
# Writer -- mirrors key_anchoring.write_bands_pinned exactly: hash every referenced result JSON,
# stamp pinned_at BEFORE any Arm 2/Arm 2' result is inspected.
# ---------------------------------------------------------------------------

def write_bands_pinned_frozenbias(
        path: str,
        arm1_val_loss_per_corpus: dict,
        arm1prime_span_frac_per_corpus: dict,
        arm1double_span_frac_per_corpus: dict,
        arm1_kraw_span_frac_per_corpus: dict,
        arm1_result_paths: dict,
        arm1prime_result_paths: dict,
        arm1double_result_paths: dict,
        arm1_kraw_result_paths: dict) -> dict:
    """Writes BANDS_PINNED-FrozenBias.json. Every `*_per_corpus` argument is {corpus: [v0,v1,v2]}
    (already-extracted floats, caller's responsibility -- mirrors key_anchoring's own writer
    contract). Every `*_result_paths` argument is {corpus: [path0,path1,path2]} -- the REAL result
    JSON paths this pin's numbers were extracted from, hashed here (never re-derived silently
    later, sec 7.3's own "mechanical, not a paper exercise" requirement)."""
    val_loss_tol = derive_val_loss_tolerance(arm1_val_loss_per_corpus)
    arm1prime_bands = derive_frozenbias_span_frac_bands(arm1prime_span_frac_per_corpus)
    arm1double_bands = derive_frozenbias_span_frac_bands(arm1double_span_frac_per_corpus)
    kraw_bands = derive_frozenbias_span_frac_bands(arm1_kraw_span_frac_per_corpus)

    def _hash_group(paths_by_corpus: dict) -> dict:
        return {corpus: [sha256_of_file(p) for p in paths] for corpus, paths in paths_by_corpus.items()}

    doc = {
        "formula_version": (
            "FROZEN_BIAS_LM_DESIGN.md sec 7.1-real.1 (pinned, ROUND-5 fix 1): two-sided "
            "mean_delta +/- t(n-1,0.975)*s/sqrt(n), n=3, t(2,0.975)=4.303, for the span_frac bands "
            "below; sec 7.2 (UNCHANGED, one-sided): mean_ref + 2*s_ref for the val-loss tolerance "
            "-- a DELIBERATELY DIFFERENT statistical object, not unified with the pinned formula "
            "(sec 7.2's own ROUND-5 disclosure)."
        ),
        "arm1_val_loss_tolerance": val_loss_tol,
        "arm1prime_span_frac_bands": arm1prime_bands,
        "arm1double_span_frac_bands": arm1double_bands,
        "arm1_kraw_span_frac_bands": kraw_bands,
        "result_paths": {
            "arm1": arm1_result_paths, "arm1prime": arm1prime_result_paths,
            "arm1double": arm1double_result_paths, "arm1_kraw": arm1_kraw_result_paths,
        },
        "result_sha256": {
            "arm1": _hash_group(arm1_result_paths), "arm1prime": _hash_group(arm1prime_result_paths),
            "arm1double": _hash_group(arm1double_result_paths),
            "arm1_kraw": _hash_group(arm1_kraw_result_paths),
        },
        "pinned_at": time.time(),
        "pinned_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


# ---------------------------------------------------------------------------
# Validator -- mirrors key_anchoring.validate_bands_pinned's F2-fixed content-re-derivation
# discipline: re-hash every referenced file AND re-derive every band from the (hash-validated)
# extracted values, requiring value-identity with what's stored -- a tampered NUMBER inside the pin
# (not just a tampered referenced FILE) is caught, exactly like the house pattern's own audit fix.
# ---------------------------------------------------------------------------

def validate_bands_pinned_frozenbias(
        path: str,
        arm1_val_loss_per_corpus: dict,
        arm1prime_span_frac_per_corpus: dict,
        arm1double_span_frac_per_corpus: dict,
        arm1_kraw_span_frac_per_corpus: dict) -> dict | None:
    """Re-hashes every referenced result JSON path recorded in the pin AND re-derives every band
    from the SAME already-extracted-float arguments the writer would have used, requiring
    value-identity with the stored doc. Returns the parsed doc on success, None on ANY validation
    failure (missing file, missing field, hash mismatch, or re-derived-value mismatch) -- mirrors
    key_anchoring.validate_bands_pinned's own "never raise for an ordinary not-ready-yet state, only
    for a genuinely malformed/tampered file" contract.

    Callers pass the SAME per-corpus float dicts a fresh writer call would use (re-extracted from
    the SAME result JSON paths the pin itself records) -- this function does not re-read those
    result JSONs' own span_frac/val_loss fields from disk (the caller already has them, e.g. from
    its own analysis pipeline); it DOES re-hash the raw JSON BYTES to confirm those files have not
    been touched since pinning."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    try:
        for group_name, paths_by_corpus in doc.get("result_paths", {}).items():
            recorded_hashes = doc["result_sha256"][group_name]
            for corpus, paths in paths_by_corpus.items():
                expected_list = recorded_hashes[corpus]
                for p, expected in zip(paths, expected_list):
                    if not os.path.exists(p):
                        return None
                    if sha256_of_file(p) != expected:
                        return None

        rederived_val_loss = derive_val_loss_tolerance(arm1_val_loss_per_corpus)
        rederived_arm1prime = derive_frozenbias_span_frac_bands(arm1prime_span_frac_per_corpus)
        rederived_arm1double = derive_frozenbias_span_frac_bands(arm1double_span_frac_per_corpus)
        rederived_kraw = derive_frozenbias_span_frac_bands(arm1_kraw_span_frac_per_corpus)

        # json round-trip normalizes container types (tuple->list etc.); floats serialize
        # losslessly and the recomputation is the same deterministic arithmetic on the same
        # inputs, so equality is exact, not approximate (mirrors key_anchoring's own F2 fix).
        def _norm(d):
            return json.loads(json.dumps(d))

        if _norm(rederived_val_loss) != _norm(doc["arm1_val_loss_tolerance"]):
            return None
        if _norm(rederived_arm1prime) != _norm(doc["arm1prime_span_frac_bands"]):
            return None
        if _norm(rederived_arm1double) != _norm(doc["arm1double_span_frac_bands"]):
            return None
        if _norm(rederived_kraw) != _norm(doc["arm1_kraw_span_frac_bands"]):
            return None
    except (KeyError, IndexError, TypeError, AssertionError):
        return None
    return doc


def assert_blind_not_broken_frozenbias(bands_doc: dict, arm2_and_arm2prime_started_ats: list) -> None:
    """sec 7.3's readout-assertion requirement: the pin's `pinned_at` must STRICTLY PRECEDE the
    EARLIEST start time across BOTH Arm 2 AND Arm 2' training runs (sec 7.3's own text: "extended
    to check both arms' start times, not just Arm 2's"). Raises AssertionError if violated --
    mirrors key_anchoring.assert_blind_not_broken exactly, generalized to the two-arm check this
    program's own design registers."""
    assert arm2_and_arm2prime_started_ats, (
        "no Arm-2/Arm-2' start times given -- nothing to check the blind against")
    earliest_start = min(arm2_and_arm2prime_started_ats)
    pinned_at = bands_doc["pinned_at"]
    assert pinned_at < earliest_start, (
        f"BLIND BROKEN: BANDS_PINNED-FrozenBias.json pinned_at={pinned_at} does NOT strictly "
        f"precede the earliest Arm-2/Arm-2' start_at={earliest_start} -- sec 7.3's mechanical "
        f"readout assertion. Every affected Arm-2/Arm-2' readout must report at descriptive tier "
        f"only.")
