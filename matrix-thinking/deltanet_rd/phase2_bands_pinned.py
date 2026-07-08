"""phase2_bands_pinned.py -- REASONING_LINK_DESIGN.md sec 16.2.1's
"Familiarization-mix val-loss tolerance band, blind-pinned before launch"
(Rev 1 fix MAJOR-6) and its per-checkpoint granularity extension (Rev 2 fix
MINOR-NEW-1): FIVE separate `mean_ref +/- 2*s_ref` tolerance bands, one per
trajectory checkpoint step, computed from the OFF arm's own 6 cells
(2 corpora x 3 seeds) val-loss ON THE NEW FAMILIARIZATION MIX, committed to
a `BANDS_PINNED`-style JSON BEFORE the per_token/global familiarization
runs launch.

Mirrors `bands_pinned_frozenbias.py`'s own writer/validator/blind-check
pattern EXACTLY -- same sha256-hash-of-referenced-result-JSONs discipline,
same "re-derive from the hashed inputs and require value-identity with the
stored pin" tamper-evidence discipline, same strict-precedes timestamp
assertion (`assert_blind_not_broken`). Cloned, not cross-imported, for the
top-level write/validate/assert functions (this codebase's own pod-safety
convention -- this file DOES import `key_anchoring.sha256_of_file`
directly, the one dependency-free helper worth reusing).

**Registered sequencing this module exists to enforce (sec 16.2.1's own
MAJOR-6/MINOR-NEW-1 paragraphs, and the disclosed process-failure precedent
`FROZEN_BIAS_LM_DESIGN.md` sec 7.3 names -- "pin before launch" as a hard
rule, not a lesson to remember later):** the OFF arm's 6 cells run to the
FULL 5,000-step familiarization budget ALONE first; `write_bands_pinned_
phase2` is called on that standalone OFF-arm run's own results BEFORE any
per_token/global familiarization run starts; `assert_blind_not_broken_
phase2` is the mechanical proof that ordering held.

**Disclosure (sec 16.2.1's own MINOR-R3-4 paragraph, reproduced here so a
future reader of THIS module cannot mistake it for a real check):** each of
the 5 pinned bands is computed FROM the OFF arm's own per-seed val-loss at
that checkpoint -- the OFF arm's own 6 cells are therefore, barring a
same-corpus cross-seed outlier beyond the band's own 2-standard-deviation
width, near-tautologically inside their own band. "OFF passed its own
val-loss gate" is NEVER evidence of anything and must never be cited as if
it corroborated the gate's real target, which is whether per_token's or
global's own val-loss -- an INDEPENDENT quantity, never used to construct
the band -- falls inside a band built without reference to either of them.
"""
from __future__ import annotations

import json
import os
import time

from key_anchoring import sha256_of_file   # the ONE dependency-free helper worth reusing directly

CHECKPOINTS = (250, 500, 1000, 2500, 5000)
N_SEEDS = 3
K_TOLERANCE = 2.0   # house k=2*s_ref one-sided convention (sec 16.2.1's own explicit citation:
                     # FROZEN_BIAS_LM_DESIGN.md sec 7.3's derive_val_loss_tolerance formula, reused
                     # verbatim, NOT the pinned t(n-1,0.975)*s/sqrt(n) two-sided formula -- a
                     # DELIBERATELY different statistical object, sec 7.2's own disclosure, carried
                     # forward unchanged here since sec 16.2.1 cites it by name).


def _mean_std(values: list) -> tuple:
    """Sample mean and sample std (n-1 denominator) -- pure Python, mirrors
    bands_pinned_frozenbias._mean_std exactly, so this module never needs
    torch either."""
    n = len(values)
    assert n >= 2, f"sample std requires n>=2, got n={n}"
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def derive_val_loss_tolerance_at_checkpoint(per_corpus_values: dict, k: float = K_TOLERANCE) -> dict:
    """ONE checkpoint's worth of the tolerance band -- `per_corpus_values`:
    {corpus: [v0,v1,v2]} (already-extracted floats from the OFF arm's own 3
    seeds AT THIS CHECKPOINT). Same one-sided `mean_ref + k*s_ref` ceiling
    FROZEN_BIAS_LM_DESIGN.md sec 7.2 / bands_pinned_frozenbias.derive_val_
    loss_tolerance already establishes, applied here per-checkpoint."""
    out = {}
    for corpus, vals in per_corpus_values.items():
        assert len(vals) == N_SEEDS, (
            f"corpus={corpus!r}: val-loss tolerance requires exactly {N_SEEDS} seeds, got {len(vals)}")
        mean, s = _mean_std(vals)
        tolerance_abs = k * s
        out[corpus] = {"mean": mean, "s": s, "n": N_SEEDS, "k": k, "tolerance_abs": tolerance_abs,
                        "pass_ceiling": mean + tolerance_abs, "per_seed": list(vals)}
    return out


def derive_all_checkpoint_bands(per_checkpoint_per_corpus_values: dict) -> dict:
    """`per_checkpoint_per_corpus_values`: {checkpoint_step: {corpus: [v0,v1,v2]}} for all 5
    registered checkpoints -- returns {checkpoint_step: <band dict>} (sec 16.2.1's own "pin FIVE
    separate tolerance bands, one per trajectory checkpoint step" requirement)."""
    assert set(per_checkpoint_per_corpus_values) == set(CHECKPOINTS), (
        f"must supply values for exactly the 5 registered checkpoints {CHECKPOINTS}, got "
        f"{sorted(per_checkpoint_per_corpus_values)}")
    return {c: derive_val_loss_tolerance_at_checkpoint(v) for c, v in per_checkpoint_per_corpus_values.items()}


# ---------------------------------------------------------------------------
# Writer -- mirrors bands_pinned_frozenbias.write_bands_pinned_frozenbias exactly.
# ---------------------------------------------------------------------------

def write_bands_pinned_phase2(path: str, per_checkpoint_per_corpus_values: dict,
                               off_result_paths: dict) -> dict:
    """Writes BANDS_PINNED-Phase2Familiarization.json.
    `per_checkpoint_per_corpus_values`: {step: {corpus: [v0,v1,v2]}}.
    `off_result_paths`: {step: {corpus: [path0,path1,path2]}} -- the REAL trajectory-checkpoint
    result JSON paths these numbers were extracted from, one set per checkpoint step, hashed here
    (never re-derived silently later)."""
    bands = derive_all_checkpoint_bands(per_checkpoint_per_corpus_values)

    def _hash_group(paths_by_corpus: dict) -> dict:
        return {corpus: [sha256_of_file(p) for p in paths] for corpus, paths in paths_by_corpus.items()}

    result_sha256 = {step: _hash_group(paths_by_corpus) for step, paths_by_corpus in off_result_paths.items()}

    doc = {
        "formula_version": (
            "REASONING_LINK_DESIGN.md sec 16.2.1 MAJOR-6/MINOR-NEW-1: FIVE per-checkpoint "
            "mean_ref + 2*s_ref one-sided val-loss tolerance bands (house k=2 convention, "
            "FROZEN_BIAS_LM_DESIGN.md sec 7.2's derive_val_loss_tolerance formula reused verbatim), "
            "computed from the OFF arm's own 6 cells on the NEW familiarization mix, one band per "
            "trajectory checkpoint step."
        ),
        "checkpoints": list(CHECKPOINTS),
        "bands_by_checkpoint": {str(c): b for c, b in bands.items()},
        "result_paths": {str(c): p for c, p in off_result_paths.items()},
        "result_sha256": {str(c): h for c, h in result_sha256.items()},
        "pinned_at": time.time(),
        "pinned_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


# ---------------------------------------------------------------------------
# Validator -- re-hashes + re-derives, requires value-identity with the stored doc.
# ---------------------------------------------------------------------------

def validate_bands_pinned_phase2(path: str, per_checkpoint_per_corpus_values: dict) -> dict | None:
    """Returns the parsed doc on success, None on ANY validation failure (missing file, missing
    field, hash mismatch, or re-derived-value mismatch) -- mirrors bands_pinned_frozenbias.validate_
    bands_pinned_frozenbias's own "never raise for an ordinary not-ready-yet state" contract."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    try:
        for step_str, paths_by_corpus in doc.get("result_paths", {}).items():
            recorded_hashes = doc["result_sha256"][step_str]
            for corpus, paths in paths_by_corpus.items():
                expected_list = recorded_hashes[corpus]
                for p, expected in zip(paths, expected_list):
                    if not os.path.exists(p):
                        return None
                    if sha256_of_file(p) != expected:
                        return None

        rederived = derive_all_checkpoint_bands(per_checkpoint_per_corpus_values)

        def _norm(d):
            return json.loads(json.dumps(d))

        rederived_keyed = {str(c): b for c, b in rederived.items()}
        if _norm(rederived_keyed) != _norm(doc["bands_by_checkpoint"]):
            return None
    except (KeyError, IndexError, TypeError, AssertionError):
        return None
    return doc


def assert_blind_not_broken_phase2(bands_doc: dict, per_token_and_global_started_ats: list) -> None:
    """sec 16.2.1's own registered sequencing requirement, restated as a mechanical assertion
    (mirrors bands_pinned_frozenbias.assert_blind_not_broken_frozenbias exactly): the pin's
    `pinned_at` must STRICTLY PRECEDE the EARLIEST start time across BOTH per_token AND global
    familiarization runs. Raises AssertionError if violated -- the exact process-failure shape this
    program already committed once (FROZEN_BIAS_LM_DESIGN.md's own rung-1 wave wrote its
    BANDS_PINNED-FrozenBias.json AFTER all 20 training cells had already completed) and registered
    as a hard build requirement here rather than a lesson to remember later."""
    assert per_token_and_global_started_ats, (
        "no per_token/global start times given -- nothing to check the blind against")
    earliest_start = min(per_token_and_global_started_ats)
    pinned_at = bands_doc["pinned_at"]
    assert pinned_at < earliest_start, (
        f"BLIND BROKEN: BANDS_PINNED-Phase2Familiarization.json pinned_at={pinned_at} does NOT "
        f"strictly precede the earliest per_token/global start_at={earliest_start} -- sec 16.2.1's "
        f"own MAJOR-6/MINOR-NEW-1 mechanical readout assertion. Every affected per_token/global "
        f"trajectory checkpoint reading must report at descriptive tier only.")


def gate_val_loss_at_checkpoint(bands_doc: dict, checkpoint_step: int, corpus: str, val_loss: float) -> tuple:
    """Reads the ALREADY-PINNED band for one (checkpoint, corpus) and checks a single val_loss
    reading against it -- pure dict lookup + comparison, never recomputes the band itself. Returns
    (in_band: bool, reason: str). Used by phase2_chain.sh (via a thin CLI, see __main__ below) to
    gate an individual per_token/global cell's own trajectory-checkpoint reading, per sec 16.2.1's
    own "every checkpoint reading is then gated against its OWN band" requirement."""
    band = bands_doc["bands_by_checkpoint"].get(str(checkpoint_step), {}).get(corpus)
    if band is None:
        return False, f"no pinned band for checkpoint={checkpoint_step} corpus={corpus!r}"
    in_band = val_loss <= band["pass_ceiling"]
    reason = (f"val_loss={val_loss:.4f} vs pass_ceiling={band['pass_ceiling']:.4f} "
              f"(mean={band['mean']:.4f} + {band['k']}*s={band['s']:.4f})")
    return in_band, reason


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["gate-check"], required=True,
                     help="CLI is a thin --mode=gate-check reader for phase2_chain.sh; the write/"
                          "validate/assert_blind functions are called directly from Python by the "
                          "chain-adjacent orchestration code (mirrors bands_pinned_frozenbias.py's "
                          "own library-not-CLI convention -- this __main__ block exists only for "
                          "the one operation a bash chain script needs to shell out for).")
    ap.add_argument("--bands-json", required=True)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--val-loss", type=float, required=True)
    args = ap.parse_args()

    with open(args.bands_json) as f:
        doc = json.load(f)
    in_band, reason = gate_val_loss_at_checkpoint(doc, args.checkpoint_step, args.corpus, args.val_loss)
    print(f"[phase2-bands-gate] checkpoint={args.checkpoint_step} corpus={args.corpus}: "
          f"{'IN-BAND' if in_band else 'OUT-OF-BAND'} -- {reason}")
    import sys
    sys.exit(0 if in_band else 1)
