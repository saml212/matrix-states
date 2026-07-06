"""sim_frozen_bias_direction.py -- FROZEN_BIAS_LM_DESIGN.md sec 7.1 revision,
attack-round finding 1 (FATAL: "bar direction underived"). CPU-only,
NO GPU work, no training.

Purpose: before launching FROZEN_BIAS_LM_DESIGN.md's Wave 1, determine
(with evidence, not assumption) which DIRECTION Arm 2's mean pooled
`span_frac` is expected to move relative to Arm 1 (baseline, no frozen
bias) -- lower (the design's original, undefended assumption) or higher
(the collinearity risk the attack round raised: blending every key toward
one of 50,257 mutually-non-orthogonal random rows in a d=64 space could
RAISE Gram deviation, not lower it, especially under Zipfian token-
frequency skew where a small set of high-frequency tokens dominate a
chunk and get pulled toward NEARLY THE SAME handful of frozen anchor rows
whenever those rows happen to be close in the ambient (n=50257, d=64)
geometry -- recall a (50257,64) random table cannot be even close to an
orthonormal/tight frame; Gram off-diagonal magnitude is set by the
Johnson-Lindenstrauss floor at that (n,d), not by 0).

INSTRUMENT REUSE (not reimplementation) -- the design's own explicit
requirement:
  - `gram_deviation` -- imported UNCHANGED from model_rd.py (this file's
    own sec 210-220: ||Eff^T Eff - I||_F over an L2-normalized (B,K,d)
    key batch).
  - `chunk_key_gram_stats` / `summarize_gram_records` -- imported
    UNCHANGED from lm_attractor_probe_rd.py, the ACTUAL LM probe's own
    per-chunk statistic computation (calls gram_deviation + rank_utils'
    effective_rank/stable_rank internally) -- this sim builds synthetic
    (B,T,d_state) key tensors and content masks in EXACTLY the shape this
    function expects, then calls it exactly as run_measurement() does,
    never touching its internals.
  - `anchors()` / `span_frac()` -- imported UNCHANGED, by direct file
    path, from the already-archived, cross-session byte-identical (md5
    7b2b8d7ca617e7b3855165fb57b86ff5 in all 3 archived copies, verified
    this session) `analyze_probe_wave2.py` -- the SAME closed-form
    normalization (`(gd - anchor_random) / (anchor_collapse -
    anchor_random)`, anchor_random = sqrt(K(K-1)/d), anchor_collapse =
    sqrt(K(K-1))) already used for every span_frac number quoted in
    SCALE_TRANSFER_DESIGN.md sec 5.9/5.10 (verified: this sim's own
    `anchors(64,64)` reproduces 7.937253933193772 / 63.49803146555018,
    matching the design doc's quoted 7.94/63.50 to the decimal places
    printed there).
  - `random_unit_rows_init` -- imported UNCHANGED from key_anchoring.py,
    the SAME frozen-table construction FROZEN_BIAS_LM_DESIGN.md sec 5
    Arm 2 specifies for B.

IMPORT-PATH NOTE (documented per the task's own instruction: "if import
is genuinely impossible, extract with a documented byte-identical copy
and say so"): model_rd.py / lm_attractor_probe_rd.py / lm_pretrain_rd.py
import `fla.modules.ShortConvolution` / `fla.ops.delta_rule.chunk_delta_rule`
at MODULE SCOPE for their (GPU-only, Triton-backed) training-time kernel
classes -- classes this sim never instantiates or calls. `fla` is not
installed on this CPU dev machine (verified: `import fla` ->
ModuleNotFoundError). Rather than hand-copy gram_deviation/
chunk_key_gram_stats's bodies (the exact instrument-mismatch risk this
task's own brief warns against), this sim installs a minimal, documented,
NEVER-CALLED stub `fla` package (2 dummy classes + 1 dummy function, each
raising NotImplementedError if actually invoked) purely to satisfy the
import statement, then imports the REAL functions from the REAL files
unmodified. Verified this session: with the stub present,
`lm_attractor_probe_rd.py --smoke` items [1]-[5] (the ENTIRE
chunk_key_gram_stats/summarize_gram_records positive/negative/EOT-
exclusion/min-valid/multi-head battery -- i.e. exactly the statistic this
sim depends on) PASS UNCHANGED; only item [6] (an end-to-end DeltaNetLM
forward pass, needing the real Triton kernel) fails, and this sim never
reaches that code path. See ensure_fla_stub() below; the stub's own
source is printed into this sim's output JSON for audit transparency.

Regime matched to the LM probe (FROZEN_BIAS_LM_DESIGN.md sec 5/8, verified
against lm_pretrain_rd.py / run_lm_rd_trackc_sweep.py this session, not
assumed): d_state=64, num_heads=1 (head_dim=64), chunk_size=64 (K in the
anchors formula), seq_len=512 (8 chunks/window), vocab_size=50257. Token
identities per chunk are drawn from a Zipfian distribution over the full
GPT-2 vocab (s=1.0, the standard natural-language-frequency exponent) --
this is the LM-realistic regime the design's own risk (2) names
explicitly ("realistic token-frequency skew"), not a uniform-token
strawman which would understate any collinearity effect.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------------------
# Minimal, documented, NEVER-CALLED fla stub -- see module docstring.
# Installed into sys.modules directly (no on-disk package needed), so this
# script is fully self-contained and leaves no stray files outside its own
# output. Guarded so it never overwrites a REAL fla if one is ever present
# (e.g. if this sim is later run on the H100 box's own tdenv).
# ---------------------------------------------------------------------------
_FLA_STUB_SOURCE = '''\
class ShortConvolution:
    def __init__(self, *a, **kw):
        raise NotImplementedError("fla stub (sim_frozen_bias_direction.py): real "
                                   "ShortConvolution not available/needed for this CPU sim")
class RMSNorm:
    def __init__(self, *a, **kw):
        raise NotImplementedError("fla stub (sim_frozen_bias_direction.py): real "
                                   "RMSNorm not available/needed for this CPU sim")
def chunk_delta_rule(*a, **kw):
    raise NotImplementedError("fla stub (sim_frozen_bias_direction.py): real "
                               "chunk_delta_rule not available/needed for this CPU sim")
'''


def ensure_fla_stub() -> bool:
    """Returns True if a stub was installed (real fla absent), False if a
    real fla was already importable (stub not needed/not installed)."""
    try:
        import fla  # noqa: F401
        return False
    except ImportError:
        pass
    import types
    fla_mod = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_delta_rule = types.ModuleType("fla.ops.delta_rule")
    ns: dict = {}
    exec(_FLA_STUB_SOURCE, ns)
    fla_modules.ShortConvolution = ns["ShortConvolution"]
    fla_modules.RMSNorm = ns["RMSNorm"]
    fla_ops_delta_rule.chunk_delta_rule = ns["chunk_delta_rule"]
    fla_mod.modules = fla_modules
    fla_mod.ops = fla_ops
    fla_ops.delta_rule = fla_ops_delta_rule
    sys.modules["fla"] = fla_mod
    sys.modules["fla.modules"] = fla_modules
    sys.modules["fla.ops"] = fla_ops
    sys.modules["fla.ops.delta_rule"] = fla_ops_delta_rule
    return True


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_STUB_INSTALLED = ensure_fla_stub()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model_rd import gram_deviation  # noqa: E402  -- REAL, unmodified import
from lm_attractor_probe_rd import chunk_key_gram_stats, summarize_gram_records  # noqa: E402
from key_anchoring import random_unit_rows_init  # noqa: E402  -- REAL, unmodified import

# analyze_probe_wave2.py is not a package member (lives under experiment-runs/),
# loaded by explicit path so `anchors`/`span_frac` are the EXACT functions
# already used for every span_frac number in SCALE_TRANSFER_DESIGN.md.
_ANALYZE_PATH = os.path.normpath(os.path.join(
    _HERE, "..", "..", "experiment-runs", "2026-07-05_trackc_rung2", "analyze_probe_wave2.py"))


def _load_span_frac_fns():
    import importlib.util
    assert os.path.exists(_ANALYZE_PATH), (
        f"analyze_probe_wave2.py not found at {_ANALYZE_PATH} -- this sim requires the SAME "
        f"span_frac implementation already used for every archived span_frac number, not a "
        f"reimplementation.")
    spec = importlib.util.spec_from_file_location("analyze_probe_wave2_frozenbias", _ANALYZE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.anchors, mod.span_frac, _ANALYZE_PATH


ANCHORS_FN, SPAN_FRAC_FN, ANALYZE_PATH_USED = _load_span_frac_fns()

# ---------------------------------------------------------------------------
# Regime constants -- matched to the LM probe (verified against
# lm_pretrain_rd.py / run_lm_rd_trackc_sweep.py this session, see module
# docstring).
# ---------------------------------------------------------------------------
D_STATE = 64
NUM_HEADS = 1
HEAD_DIM = D_STATE // NUM_HEADS          # 64 -- "d" in anchors(K, d)
CHUNK_SIZE = 64                          # "K" in anchors(K, d)
SEQ_LEN = 512                            # 8 chunks/window, matches the Track C probe default
N_CHUNKS_PER_WINDOW = SEQ_LEN // CHUNK_SIZE
VOCAB_SIZE = 50257
ZIPF_S = 1.0                             # standard natural-language Zipf exponent
LAMBDA_PRIMARY = 0.58                    # FROZEN_BIAS_LM_DESIGN.md sec 5, Arm 2's fixed lambda


def sample_zipf_token_ids(n: int, vocab_size: int, s: float, generator: torch.Generator) -> torch.Tensor:
    """Token ids for n positions, drawn i.i.d. from a Zipf(s) law over
    ranks {0,...,vocab_size-1} (rank 0 = most frequent), the standard
    natural-language frequency-skew model this sim's own docstring
    registers ("realistic token-frequency skew, e.g. Zipfian over
    vocab"). Implemented via inverse-CDF sampling over the full,
    explicitly-normalized discrete pmf (exact, not approximate --
    vocab_size=50257 makes an explicit pmf array cheap: <1MB)."""
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float64)
    pmf = ranks.pow(-s)
    pmf = pmf / pmf.sum()
    cdf = torch.cumsum(pmf, dim=0)
    u = torch.rand(n, generator=generator, dtype=torch.float64)
    idx = torch.searchsorted(cdf, u)
    idx = idx.clamp(max=vocab_size - 1)
    return idx.to(torch.int64)


def build_unblended_population(n_windows: int, seed: int, shared_frac: float = 0.0,
                                 n_shared_dirs: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Arm 1 analog: raw keys with NO frozen bias applied at all.

    `shared_frac=0.0` (the naive/literal reading): pure random unit vectors
    per position -- i.i.d., no structure at all. This is the DEGENERATE
    case: K=64 i.i.d. random unit vectors in d=64 dims are already close to
    a random tight frame BY CONSTRUCTION, i.e. this population's own
    span_frac sits at ~0 (it effectively IS the `anchors()["random"]`
    reference point). Measuring a blend's effect relative to a baseline
    that is already at the random anchor is close to measuring the blend
    in isolation, not "the effect of blending onto an ALREADY non-
    orthonormal (attractor-affected) trained population" -- which is what
    Arm 1 in the real design actually is (its own archived reference,
    SCALE_TRANSFER_DESIGN.md, sits at span_frac 0.248-0.389, NOT ~0).

    `shared_frac>0` (the CALIBRATED, primary regime this sim actually
    reports as its headline): each position's raw key is a convex blend of
    (a) an independent random unit vector and (b) one of `n_shared_dirs`
    fixed shared random directions (a stand-in for "whatever collapsed/
    attractor structure SGD's own training converges to" -- this sim has
    no trained checkpoint, so a shared-direction construction is the
    simplest structural proxy for 'some tokens' keys cluster onto a few
    directions', which is exactly what a positive raw Gram deviation
    MEANS geometrically), re-normalized. `shared_frac` is calibrated (see
    CALIBRATED_SHARED_FRAC below) so this population's OWN measured
    span_frac lands close to the real archived Arm-1 reference (0.248-0.27
    at the 14M scale) BEFORE any bias is applied -- i.e. this baseline is
    built to resemble what a real checkpoint's raw key population would
    give this same instrument, not an arbitrary or convenient choice.
    """
    g = torch.Generator().manual_seed(seed)
    if shared_frac <= 0.0:
        k_raw = F.normalize(torch.randn(n_windows, SEQ_LEN, D_STATE, generator=g, dtype=torch.float32), dim=-1)
    else:
        shared_dirs = F.normalize(torch.randn(n_shared_dirs, D_STATE, generator=g, dtype=torch.float32), dim=-1)
        raw = torch.randn(n_windows, SEQ_LEN, D_STATE, generator=g, dtype=torch.float32)
        idx = torch.randint(0, n_shared_dirs, (n_windows, SEQ_LEN), generator=g)
        shared_component = shared_dirs[idx]
        mixed = (1.0 - shared_frac) * F.normalize(raw, dim=-1) + shared_frac * shared_component
        k_raw = F.normalize(mixed, dim=-1)
    token_ids = torch.stack([sample_zipf_token_ids(SEQ_LEN, VOCAB_SIZE, ZIPF_S,
                                                     torch.Generator().manual_seed(seed * 7919 + w))
                               for w in range(n_windows)], dim=0)
    return k_raw, token_ids


# Calibrated so build_unblended_population's OWN measured span_frac lands at
# ~0.27, matching SCALE_TRANSFER_DESIGN.md's real archived 14M Arm-1
# reference (0.248, sec 5.9/5.10) to within the per-run seed band already
# documented there (~20.1-25.8 raw gd, i.e. roughly +/-0.03 span_frac) --
# found by a manual scan (shared_frac in {0.0,...,0.7}, see
# run_crossover_scan() below for the full scan this constant is chosen
# from), not an arbitrarily convenient number.
CALIBRATED_SHARED_FRAC = 0.6


def build_dense_blended_population(k_raw: torch.Tensor, token_ids: torch.Tensor, B: torch.Tensor,
                                     lam: float) -> torch.Tensor:
    """Arm 2 (PRIMARY intervention) analog -- FROZEN_BIAS_LM_DESIGN.md sec 2's
    exact construction: k_biased = normalize((1-lam)*k_raw + lam*B[token_id]),
    applied to EVERY position (dense, no selection), per-token identity via
    B[token_id]."""
    Bk = B[token_ids]                                    # (n_windows, T, d)
    blended = (1.0 - lam) * k_raw + lam * Bk
    return F.normalize(blended, dim=-1)


def build_global_blended_population(k_raw: torch.Tensor, B_global_row: torch.Tensor,
                                      lam: float) -> torch.Tensor:
    """Arm 2' (mandatory control, attack-round finding 2) analog -- the SAME
    frozen table's single fixed row (this sim uses B's own row-mean,
    re-normalized to a unit vector, matching 'the SAME frozen table's
    global mean row' framing the attack round's own Q1 proposed) blended
    at the SAME lambda, with NO per-token lookup -- isolates whether any
    effect is about being a per-token-KEYED bias at all, or merely about
    adding any constant vector."""
    blended = (1.0 - lam) * k_raw + lam * B_global_row.view(1, 1, -1)
    return F.normalize(blended, dim=-1)


def measure_population(k_pop: torch.Tensor, K: int = CHUNK_SIZE, d: int = HEAD_DIM) -> dict:
    """Runs the REAL chunk_key_gram_stats/summarize_gram_records (imported
    unmodified above) on a synthetic (n_windows, T, d_state) key tensor,
    exactly as run_measurement() does for a real checkpoint's captured
    keys, then applies the REAL anchors()/span_frac() transform (imported
    unmodified from analyze_probe_wave2.py)."""
    n_windows = k_pop.shape[0]
    content_mask = torch.ones(n_windows, SEQ_LEN, dtype=torch.bool)   # no EOT/padding in this synthetic regime
    records = chunk_key_gram_stats(k_pop, content_mask, num_heads=NUM_HEADS, chunk_size=CHUNK_SIZE)
    summary = summarize_gram_records(records)
    a = ANCHORS_FN(K, d)
    sf = SPAN_FRAC_FN(summary["gram_deviation_mean"], K, d) if summary.get("n_scored", 0) > 0 else None
    summary["anchors"] = a
    summary["span_frac"] = sf
    return summary


def run_cell(seed: int, n_windows: int, lam: float, shared_frac: float = 0.0) -> dict:
    """One sim 'cell': builds all three populations from the SAME unblended
    draw (seed-matched, so the ONLY difference across arms is the blend
    itself -- mirrors the design's own architecture/token/seed-matching
    discipline), measures each with the real instrument.

    `shared_frac` selects the baseline regime (see build_unblended_population):
    0.0 = naive i.i.d.-random baseline (span_frac~0, degenerate);
    CALIBRATED_SHARED_FRAC = baseline calibrated to the real archived
    Arm-1 reference (span_frac~0.25-0.27)."""
    k_raw, token_ids = build_unblended_population(n_windows, seed, shared_frac=shared_frac)
    B = random_unit_rows_init(VOCAB_SIZE, D_STATE, seed=20260705)   # ANCHOR_INIT_SEED convention, same seed as the design's own B
    B_global_row = F.normalize(B.mean(dim=0, keepdim=True), dim=-1).squeeze(0)

    k_dense = build_dense_blended_population(k_raw, token_ids, B, lam)
    k_global = build_global_blended_population(k_raw, B_global_row, lam)

    return {
        "seed": seed, "n_windows": n_windows, "lambda": lam, "shared_frac": shared_frac,
        "unblended": measure_population(k_raw),
        "dense_blended_arm2": measure_population(k_dense),
        "global_blended_arm2prime": measure_population(k_global),
    }


# ---------------------------------------------------------------------------
# Sensitivity grid -- seeds x lambda x zipf-exponent x BASELINE REGIME, so a
# directional conclusion is not a single-draw artifact (the design's own
# attack-round discipline: "if the simulated direction is confident and
# stable across the sim's own seed/param sensitivity grid"). The baseline-
# regime axis (naive vs. calibrated) turned out, empirically, to be THE
# decision-relevant axis -- see the crossover scan below and the results
# writeup in main().
# ---------------------------------------------------------------------------
SENSITIVITY_SEEDS = [0, 1, 2, 3, 4]     # 5 independent draws, distinct from every archived training seed >3
SENSITIVITY_LAMBDAS = [0.3, 0.58, 0.8]  # brackets the design's own mini-sweep grid (finding 3)
SENSITIVITY_ZIPF_S = [0.5, 1.0, 1.3]    # mild / standard / strong frequency skew
N_WINDOWS_PER_CELL = 200                 # 200 windows x 8 chunks/window = 1,600 chunk-episodes/cell


def run_sensitivity_grid(shared_frac: float) -> dict:
    global ZIPF_S
    primary_cells = []
    for seed in SENSITIVITY_SEEDS:
        primary_cells.append(run_cell(seed, N_WINDOWS_PER_CELL, LAMBDA_PRIMARY, shared_frac=shared_frac))

    lambda_cells = []
    for lam in SENSITIVITY_LAMBDAS:
        lambda_cells.append(run_cell(seed=0, n_windows=N_WINDOWS_PER_CELL, lam=lam, shared_frac=shared_frac))

    zipf_cells = []
    orig_zipf = ZIPF_S
    for s in SENSITIVITY_ZIPF_S:
        ZIPF_S = s
        zipf_cells.append({**run_cell(seed=0, n_windows=N_WINDOWS_PER_CELL, lam=LAMBDA_PRIMARY,
                                        shared_frac=shared_frac), "zipf_s": s})
    ZIPF_S = orig_zipf

    return {"primary_seed_grid": primary_cells, "lambda_grid": lambda_cells, "zipf_grid": zipf_cells}


def summarize_direction(grid: dict) -> dict:
    """Reduces the grid to the one decision-relevant question: does Arm 2
    (dense blended) move span_frac ABOVE or BELOW the unblended baseline,
    and how stable is the sign/magnitude across the sensitivity axes?"""
    def delta(cell):
        u = cell["unblended"]["span_frac"]
        d2 = cell["dense_blended_arm2"]["span_frac"]
        g2 = cell["global_blended_arm2prime"]["span_frac"]
        return {"unblended": u, "dense_arm2": d2, "global_arm2prime": g2,
                "delta_arm2_minus_unblended": d2 - u, "delta_global_minus_unblended": g2 - u,
                "delta_arm2_minus_global": d2 - g2}

    seed_deltas = [delta(c) for c in grid["primary_seed_grid"]]
    lambda_deltas = [{"lambda": c["lambda"], **delta(c)} for c in grid["lambda_grid"]]
    zipf_deltas = [{"zipf_s": c["zipf_s"], **delta(c)} for c in grid["zipf_grid"]]

    seed_arm2_deltas = torch.tensor([d["delta_arm2_minus_unblended"] for d in seed_deltas])
    all_positive = bool((seed_arm2_deltas > 0).all())
    all_negative = bool((seed_arm2_deltas < 0).all())
    sign_stable = all_positive or all_negative
    direction = ("ABOVE (span_frac RISES under dense per-token blending)" if all_positive else
                 "BELOW (span_frac FALLS under dense per-token blending)" if all_negative else
                 "MIXED/UNSTABLE (sign flips across seeds)")

    return {
        "seed_grid_deltas": seed_deltas,
        "lambda_grid_deltas": lambda_deltas,
        "zipf_grid_deltas": zipf_deltas,
        "seed_grid_mean_delta_arm2_minus_unblended": seed_arm2_deltas.mean().item(),
        "seed_grid_std_delta_arm2_minus_unblended": seed_arm2_deltas.std(unbiased=True).item(),
        "sign_stable_across_seeds": sign_stable,
        "direction_verdict": direction,
    }


# ---------------------------------------------------------------------------
# Crossover scan -- THE decisive evidence. Sweeps shared_frac (i.e. how
# non-orthonormal the UNBLENDED baseline already is) and asks: at what
# baseline span_frac does the sign of (Arm2 - baseline) flip? This
# resolves the apparent contradiction between the naive regime (span_frac
# RISES) and the calibrated regime (span_frac FALLS) -- both are real
# results of the SAME code, at different points on one continuous axis;
# the crossover scan shows where the real archived Arm-1 reference sits
# relative to the flip point.
# ---------------------------------------------------------------------------
CROSSOVER_SHARED_FRACS = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def run_crossover_scan(lam: float = LAMBDA_PRIMARY, seed: int = 0) -> list[dict]:
    rows = []
    for sf in CROSSOVER_SHARED_FRACS:
        cell = run_cell(seed=seed, n_windows=N_WINDOWS_PER_CELL, lam=lam, shared_frac=sf)
        u, d2 = cell["unblended"]["span_frac"], cell["dense_blended_arm2"]["span_frac"]
        rows.append({"shared_frac": sf, "baseline_span_frac": u, "dense_arm2_span_frac": d2,
                      "delta": d2 - u})
    return rows


def main():
    t0 = time.time()

    grid_naive = run_sensitivity_grid(shared_frac=0.0)
    direction_naive = summarize_direction(grid_naive)

    grid_calibrated = run_sensitivity_grid(shared_frac=CALIBRATED_SHARED_FRAC)
    direction_calibrated = summarize_direction(grid_calibrated)

    crossover = run_crossover_scan()

    # Sanity check: does the calibrated baseline itself land near the real
    # archived Arm-1 reference (span_frac 0.248-0.27, SCALE_TRANSFER_DESIGN.md
    # sec 5.9/5.10)? Computed from the calibrated grid's own seed-0 cell.
    calibrated_baseline_check = grid_calibrated["primary_seed_grid"][0]["unblended"]["span_frac"]

    out = {
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 7.1 revision, attack finding 1",
        "instrument_provenance": {
            "gram_deviation_source": "model_rd.py:gram_deviation (imported unmodified)",
            "chunk_key_gram_stats_source": "lm_attractor_probe_rd.py:chunk_key_gram_stats (imported unmodified)",
            "summarize_gram_records_source": "lm_attractor_probe_rd.py:summarize_gram_records (imported unmodified)",
            "span_frac_source": ANALYZE_PATH_USED + ":anchors/span_frac (imported unmodified, by file path)",
            "random_unit_rows_init_source": "key_anchoring.py:random_unit_rows_init (imported unmodified)",
            "fla_stub_installed": _STUB_INSTALLED,
            "fla_stub_source_for_audit": _FLA_STUB_SOURCE if _STUB_INSTALLED else None,
            "smoke_verification_note": (
                "lm_attractor_probe_rd.py --smoke items [1]-[5] (the entire chunk_key_gram_stats/"
                "summarize_gram_records positive/negative/EOT-exclusion/min-valid/multi-head battery)"
                " verified PASSING under this exact stub, this session, before this sim was written."
                " Item [6] (a real DeltaNetLM forward pass) was not exercised and is not needed here."),
        },
        "regime": {"d_state": D_STATE, "num_heads": NUM_HEADS, "head_dim": HEAD_DIM,
                    "chunk_size": CHUNK_SIZE, "seq_len": SEQ_LEN, "vocab_size": VOCAB_SIZE,
                    "zipf_s_primary": 1.0, "lambda_primary": LAMBDA_PRIMARY,
                    "n_windows_per_cell": N_WINDOWS_PER_CELL,
                    "n_chunk_episodes_per_cell": N_WINDOWS_PER_CELL * N_CHUNKS_PER_WINDOW,
                    "calibrated_shared_frac": CALIBRATED_SHARED_FRAC},
        "anchors_check_against_archived_numbers": {
            "computed": ANCHORS_FN(CHUNK_SIZE, HEAD_DIM),
            "archived_quoted": {"random": 7.94, "collapse": 63.50},
            "note": "SCALE_TRANSFER_DESIGN.md sec 5.9/5.10 quotes 7.94/63.50 at K=64,d=64 -- matches "
                    "to the printed decimal places.",
        },
        "calibrated_baseline_vs_archived_reference": {
            "sim_calibrated_baseline_span_frac": calibrated_baseline_check,
            "archived_14M_reference_span_frac_range": [0.248, 0.389],
            "note": "The calibrated regime's own baseline (no bias applied) is built to resemble the "
                    "REAL archived Arm-1 span_frac (0.248 at 14M up to 0.389 at 392M), not the "
                    "degenerate near-zero value the naive i.i.d.-random regime gives by construction.",
        },
        "regime_naive_iid_random_baseline": {
            "description": "shared_frac=0.0 -- baseline span_frac ~0 (degenerate, close to the "
                            "anchors()['random'] reference point itself, NOT representative of a real "
                            "trained checkpoint's Arm-1 baseline).",
            "raw_grid": grid_naive,
            "direction_summary": direction_naive,
        },
        "regime_calibrated_to_archived_baseline": {
            "description": f"shared_frac={CALIBRATED_SHARED_FRAC} -- baseline span_frac calibrated to "
                            f"~{calibrated_baseline_check:.3f}, matching the REAL archived Arm-1 "
                            f"reference range (0.248-0.389). THIS IS THE PRIMARY, DECISION-RELEVANT "
                            f"REGIME.",
            "raw_grid": grid_calibrated,
            "direction_summary": direction_calibrated,
        },
        "crossover_scan": {
            "description": "Sweeps the baseline's own non-orthonormality (shared_frac, x-axis) and "
                            "reports where the sign of (Arm2 dense-blended span_frac - baseline "
                            "span_frac) flips. Resolves the apparent naive-vs-calibrated contradiction: "
                            "both are the SAME code, different points on one continuous axis.",
            "rows": crossover,
        },
        "wall_s": time.time() - t0,
    }

    out_dir = os.path.join(_HERE, "results", "frozen_bias_sim")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sim_frozen_bias_direction_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "direction_summary_NAIVE_regime": direction_naive,
        "direction_summary_CALIBRATED_regime (PRIMARY)": direction_calibrated,
        "calibrated_baseline_vs_archived_reference": out["calibrated_baseline_vs_archived_reference"],
        "crossover_scan_rows": crossover,
        "anchors_check": out["anchors_check_against_archived_numbers"],
        "wrote": out_path,
    }, indent=2))


if __name__ == "__main__":
    main()
