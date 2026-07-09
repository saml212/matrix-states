"""CAPABILITY_SEPARATION_DESIGN.md S1.4/S1.4.1/S1.3.3 -- the per-group word-
sampling grammar, wired onto the REAL matrix generating sets (groups.py),
NOT the permutation stand-ins coverage_calibration.py used for bar
CALIBRATION only. The bars below are the EXACT numbers
coverage_calibration.py's STEP 2 computed and S1.4's table pins (imported as
constants here, not re-derived by Monte Carlo at runtime -- re-deriving on
every training run would be needlessly expensive and would silently drift
from the pinned, five-round-attacked table if the calibration script's
RNG/trial count ever changed later; the calibration script itself remains
the reproducible source of truth, S1.12).

Provides:
  - per-batch-fixed-L training sampler (S1.4 delta 3).
  - per-SAMPLE-L held-out word sampler for the N=50 degauging/coverage
    sample (S1.4.1 -- NOT batched training data, drawn once per cell).
  - the query-coverage guard with retry-once-then-hard-fail (S1.4.1 step 4),
    generalized (`check_coverage_with_retry`) so the SAME machinery drives
    both the top-level N=50 coverage bar (S1.4) and readout.py's fit/eval
    diversity floors (S1.4.1 step 4's "eval-diversity floor variant").
  - BOTH negative tests (S1.7 gate 3), run to completion at the bottom of
    this file.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groups import GROUP_NAMES, D_MIN, ORDER, D_STATE, generating_set, gen_tensor, \
    batched_word_product, word_product, batched_targets

# S1.4's exact calibrated query-coverage bars (coverage_calibration.py's
# STEP 2, reproduced verbatim in S1.3.3/S1.4's table). Pinned as constants.
COVERAGE_BAR = {"S3": 5, "S4": 17, "A5": 27, "S5": 30, "A6": 36}
COVERAGE_FRAC = {"S3": 0.80, "S4": 0.70, "A5": 0.45, "S5": 0.25, "A6": 0.10}

# S1.4.1 step 4's fit/eval diversity floors (coverage_calibration.py STEP
# 3/4's exact formulas: fit = min(3*d_min, floor(0.8*|G|)); eval =
# min(2*d_min, floor(0.6*|G|))).
FIT_FLOOR = {name: min(3 * D_MIN[name], int(0.8 * ORDER[name])) for name in GROUP_NAMES}
EVAL_FLOOR = {name: min(2 * D_MIN[name], int(0.6 * ORDER[name])) for name in GROUP_NAMES}

L_TRAIN_LO, L_TRAIN_HI = 1, 8      # S1.4: train L ~ Uniform{1..8}
L_EVAL_LO, L_EVAL_HI = 9, 16       # S1.4: held-out eval L ~ Uniform{9..16}
N_EVAL_WORDS = 50                  # S1.4.1's pinned floor, "N>=50 words per group"
N_FIT, N_EVAL_SPLIT = 30, 20       # S1.4.1 step 4's 60/40 split of the N=50 sample


# ---------------------------------------------------------------------------
# Training sampler: per-batch-fixed-L (S1.4 delta 3).
# ---------------------------------------------------------------------------

def sample_train_batch(name: str, batch_size: int, gen: torch.Generator,
                       device="cpu", dtype=torch.float32) -> dict:
    """ONE batch: ONE L drawn (per-batch-fixed-L) from Uniform{1..8}, shared
    by every episode in the batch; L varies ACROSS batches, not within one
    (S1.4's pinned scheme -- zero new padding/mask code, BindingEncoder's
    fixed-shape (B,L,h) forward pass is reused completely unmodified)."""
    n_gens = len(generating_set(name))
    d_state = D_STATE[name]
    L = int(torch.randint(L_TRAIN_LO, L_TRAIN_HI + 1, (1,), generator=gen).item())
    token_idx = torch.randint(0, n_gens, (batch_size, L), generator=gen).to(device)
    target = batched_targets(name, token_idx, d_state, dtype=dtype)
    return {"token_idx": token_idx, "target": target, "L": L}


def sample_eval_batch(name: str, batch_size: int, gen: torch.Generator,
                      device="cpu", dtype=torch.float32) -> dict:
    """Held-out generalization batch: per-batch-fixed-L drawn from the
    HELD-OUT range Uniform{9..16} (S1.4/C5), never seen at train time."""
    n_gens = len(generating_set(name))
    d_state = D_STATE[name]
    L = int(torch.randint(L_EVAL_LO, L_EVAL_HI + 1, (1,), generator=gen).item())
    token_idx = torch.randint(0, n_gens, (batch_size, L), generator=gen).to(device)
    target = batched_targets(name, token_idx, d_state, dtype=dtype)
    return {"token_idx": token_idx, "target": target, "L": L}


# ---------------------------------------------------------------------------
# Held-out N=50 word sample, per-SAMPLE-L (S1.4.1 -- the degauging/coverage
# sample, distinct from batched TRAINING data above; drawn once per cell).
# ---------------------------------------------------------------------------

def sample_eval_words(name: str, seed: int, n_words: int = N_EVAL_WORDS):
    """Draw n_words i.i.d. words, L~Uniform{L_EVAL_LO..L_EVAL_HI} PER WORD
    (matches coverage_calibration.py's own healthy_trial convention exactly,
    now on the REAL matrix generating set rather than the permutation
    stand-in). Returns (idx_list: list of (L,) int arrays, prod_list: list
    of (d_min,d_min) real matrices, product(w) via groups.word_product)."""
    rng = np.random.default_rng(seed)
    gens = generating_set(name)
    n_gens = len(gens)
    idx_list, prod_list = [], []
    for _ in range(n_words):
        L = int(rng.integers(L_EVAL_LO, L_EVAL_HI + 1))
        idx = rng.integers(0, n_gens, size=L)
        idx_list.append(idx)
        prod_list.append(word_product(gens, idx))
    return idx_list, prod_list


def _undersampled_words(name: str, seed: int, n_words: int = N_EVAL_WORDS):
    """Pathological L=1 sampler (coverage_calibration.py's undersampled_trial
    analog, on the REAL matrix generating set): a DETERMINISTIC ceiling of
    |generating set| distinct elements, never exceedable regardless of
    n_words -- a plausible real bug (e.g. the held-out-length sampler
    silently defaults to the trivial case)."""
    rng = np.random.default_rng(seed)
    gens = generating_set(name)
    n_gens = len(gens)
    prod_list = []
    for _ in range(n_words):
        idx = rng.integers(0, n_gens, size=1)
        prod_list.append(word_product(gens, idx))
    return prod_list


def _distinct_count(matrices: list) -> int:
    """Distinct-element count via the SAME rounded-key dedup convention
    verify_option_a_readout.py's bfs_closure() uses (round to 6 decimals,
    tuple key)."""
    return len({tuple(np.round(M, 6).flatten()) for M in matrices})


# ---------------------------------------------------------------------------
# Query-coverage guard, retry-once-then-hard-fail (S1.4.1 step 4), and its
# "eval-diversity floor variant" -- the SAME machinery, different
# bar/label/offset, reused by readout.py's fit/eval diversity floors.
# ---------------------------------------------------------------------------

class CoverageGuardError(RuntimeError):
    """Raised on a second consecutive coverage-check miss (S1.4.1 step 4's
    hard-fail rule: two independent draws both failing is evidence of a
    real problem, not sampling noise)."""


def check_coverage_with_retry(name: str, base_seed: int, bar: int, n_words: int = N_EVAL_WORDS,
                              label: str = "query-coverage", offset: int = 10_000) -> dict:
    """S1.4.1 step 4's retry-once rule: if the first N=50 draw (seed =
    base_seed + offset) fails the bar, resample ONCE with a shifted seed
    (base_seed + 1 + offset -- "the same cell seed + 1", S1.4.1's exact
    wording, applied to the BASE seed before the purpose-offset, Task D's
    own seed+10_000 precedent). Logs both attempts; hard-fails (raises
    CoverageGuardError) on a second consecutive miss. Returns a log dict on
    success (with `final_seed`, the seed whose draw the caller should
    reuse for the actual eval-word sample)."""
    log = {"label": label, "group": name, "bar": bar, "attempts": []}
    for attempt, seed_shift in enumerate((0, 1)):
        seed = base_seed + seed_shift + offset
        _, prod_list = sample_eval_words(name, seed, n_words)
        count = _distinct_count(prod_list)
        passed = count >= bar
        log["attempts"].append({"attempt": attempt, "seed": seed, "distinct_count": count,
                                "passed": passed})
        if passed:
            log["result"] = "pass"
            log["final_seed"] = seed
            return log
    log["result"] = "hard_fail"
    raise CoverageGuardError(
        f"{label} guard HARD-FAILED for {name}: two independent N={n_words} draws "
        f"(seeds {log['attempts'][0]['seed']}, {log['attempts'][1]['seed']}) both scored "
        f"below bar={bar} (counts {log['attempts'][0]['distinct_count']}, "
        f"{log['attempts'][1]['distinct_count']}). Full log: {log}"
    )


# ---------------------------------------------------------------------------
# NEGATIVE TESTS (S1.7 gate 3) -- run to completion, output shown.
# ---------------------------------------------------------------------------

def _test_coverage_guard_detects_undersampling():
    """NEGATIVE TEST 1: the L=1 pathological sampler must FAIL the
    calibrated bar for EVERY one of the 5 groups -- proven by actually
    running the check against REAL matrix words (not merely re-quoting
    coverage_calibration.py's permutation-stand-in numbers)."""
    print("=" * 88)
    print("NEGATIVE TEST 1 -- coverage guard detects L=1 undersampling (all 5 groups)")
    print("=" * 88)
    all_detected = True
    for name in GROUP_NAMES:
        bar = COVERAGE_BAR[name]
        prod_list = _undersampled_words(name, seed=999_000)
        count = _distinct_count(prod_list)
        n_gens = len(generating_set(name))
        detected = count < bar
        all_detected = all_detected and detected
        print(f"  [{name}] L=1 undersampled distinct_count={count} (<= |gens|={n_gens}) "
              f"vs bar={bar}: {'DETECTED (count < bar)' if detected else 'MISSED'}")
        assert detected, f"{name}: undersampled L=1 sampler was NOT caught by the coverage bar"
    print(f"\nRESULT: undersampling DETECTED for all {len(GROUP_NAMES)} groups.\n")
    return all_detected


def _test_coverage_guard_second_miss_hard_fails():
    """NEGATIVE TEST 2: feeding the guard TWO consecutive undersampled draws
    (via a monkeypatched sample_eval_words) must raise CoverageGuardError,
    not silently pass or retry a third time."""
    print("=" * 88)
    print("NEGATIVE TEST 2 -- second consecutive coverage-check miss HARD-FAILS")
    print("=" * 88)
    # Patch THIS module's own globals (sys.modules[__name__], not a fresh
    # `import group_task` -- when this file runs as __main__, a second
    # `import group_task` creates a SEPARATE module object with its own
    # globals copy, so patching it would silently no-op against the
    # __main__-resident check_coverage_with_retry actually under test).
    _self = sys.modules[__name__]
    name = "A5"
    bar = COVERAGE_BAR[name]

    def _always_undersampled(name_, seed, n_words=N_EVAL_WORDS):
        return None, _undersampled_words(name_, seed, n_words)

    orig = _self.sample_eval_words
    _self.sample_eval_words = _always_undersampled
    raised = False
    try:
        check_coverage_with_retry(name, base_seed=123, bar=bar)
    except CoverageGuardError as e:
        raised = True
        msg = str(e)
        print(f"  CoverageGuardError raised as expected:\n    {msg[:220]}...")
    finally:
        _self.sample_eval_words = orig
    assert raised, "second consecutive miss did NOT hard-fail (retry-once rule has no teeth)"
    # sanity: guard is restored and now passes normally again.
    ok_log = check_coverage_with_retry(name, base_seed=123, bar=bar)
    assert ok_log["result"] == "pass", "guard did not recover after monkeypatch teardown"
    print(f"  post-teardown sanity: guard passes normally again (seed={ok_log['final_seed']})  OK")
    print("\nRESULT: second consecutive miss correctly raises CoverageGuardError (hard fail).\n")
    return raised


if __name__ == "__main__":
    r1 = _test_coverage_guard_detects_undersampling()
    r2 = _test_coverage_guard_second_miss_hard_fails()
    print("=" * 88)
    print(f"group_task.py negative tests: undersampling_detected={r1}  second_miss_hard_fails={r2}")
    print("=" * 88)
    assert r1 and r2
