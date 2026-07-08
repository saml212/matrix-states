"""diag_c17_repro_analysis.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.24
(Rev 4, DESIGN-CLEARED-FOR-BUILD): the standalone offline analysis for the
C17 eval-admission repro instrument. FILENAME PIN (sec 15.24 build task,
this build session): the design's own sec 15.24.4 left this script's
filename unpinned ("build a NEW standalone analysis script"); this session
pins it as `diag_c17_repro_analysis.py`, recorded here and in the build
commit message per the build task's own instruction.

Loads dump sink(s) from <=3 launches (--launch CKPT_DIR SEED, repeatable --
primary seed 1940 alone, or primary + up to 2 reserved contingency seeds
1943/1944 on a prior NO-REPRO -- sec 15.24.4/15.24.7's own registered,
human-loop-gated follow-up; this script NEVER fires a contingency seed
itself) and runs sec 15.24.4's Total precedence, EXACTLY:

    reconstruction gate (prerequisite, HARD-ABORT-only, no verdict)
        > 0b (structural)
        > Step -1 (reproduction)
        > Step 1 (agreement)
        > Step 2 (tolerance)

producing a verdict JSON (RECONSTRUCTION-FAILURE / INSTRUMENT-BUG /
NO-REPRO / AMBIGUOUS-NONDETERMINISM / REAL-CAPACITY-BOUNDARY /
TOLERANCE-MISCALIBRATION / AMBIGUOUS-RESIDUAL) with the full per-event/
per-episode table each step computed.

DESIGN DECISION (this build session, disclosed per the build task's own
instruction -- resolves the round-5 verify's own unpinned item 1): UNIT-
TEST ISOLATION. Every step in the precedence above is its OWN testable,
importable, top-level (or leading-underscore, see below) function -- the
reconstruction gate (`reconstruct_pools_offline`/`check_reconstruction_
gate`), Step 0b (`step_0b_pool_membership`), Step -1 (`step_neg1_event_
guard`), Step 0a (`step_0a_exact_rank`), Step 1 (`step_1_live_offline_
agreement`, itself split into a BOX-ONLY recompute half and a pure,
CPU-only floor-counting half, see below), and Step 2 (`step_2_niter_sweep`,
split the same way). `main()` ALWAYS runs the FULL precedence via
`run_full_precedence()` -- there is no partial/bypass CLI path. Stage -1's
own fixtures (diag_c17_repro_stage_minus1.py) call these step functions
DIRECTLY, never through main()/the CLI, so a fixture can feed a hand-built
event list to exactly one step in isolation.

A SECOND, related disclosed decision this isolation makes possible: Step 1/
Step 2's REAL offline recompute needs `model_rd.newton_schulz_orthogonalize`
-- the model_rd.py module pulls in `fla.modules.ShortConvolution` (a CUDA-
oriented dependency) at module scope, so it CANNOT be imported off-box
(verified this session; this is an ALREADY-ESTABLISHED constraint in this
codebase -- see smoke_keyanchor_scaling.py's own "model_rd.py cannot be
imported in this fla-less CPU sandbox" note). This script therefore imports
`newton_schulz_orthogonalize` LAZILY, inside the two specific functions that
need it (`_step_1_recompute_disagreements`, `_step_2_recompute_resolve_
niter`) -- mirroring grammar_rd.py's own `load_gpt2_tokenizer()` lazy-import
convention ("transformers is a real dependency only on the box; keeping it
out of the module top level lets the rest of this file be imported/read
anywhere"). Every OTHER function here (the reconstruction gate, Step 0b,
Step -1, Step 0a, the two-level floor, Step 2's verdict logic, and
`run_full_precedence`/`main` themselves) is importable and unit-testable
with NO CUDA/fla dependency at all -- only the two real-recompute
sub-functions are BOX-ONLY.

Exit code convention: `main()` returns 0 whenever `run_full_precedence`
completes without raising -- EVERY documented state (RECONSTRUCTION-FAILURE,
NO-REPRO, AMBIGUOUS-NONDETERMINISM included) is a well-defined, successful
analysis outcome, not a script bug; a chain/caller inspects the `verdict`
field of the written JSON to decide what happens next (sec 15.24.4's own
follow-ups are human-loop gated, never auto-fired by this script).
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import grammar_rd as grd                              # noqa: E402 (fla-free)
import key_anchoring as ka                             # noqa: E402 (fla-free)
import run_deltanet_rd_exactness_sweep as rdx          # noqa: E402 (fla-free)

# ---------------------------------------------------------------------------
# Pinned constants (sec 15.24.4: "All thresholds below are EXACT, no
# numerical-tolerance slack"; resid_tol is the SAME production constant this
# program uses everywhere -- never re-derived).
# ---------------------------------------------------------------------------

RECONSTRUCTION_FAILURE = "RECONSTRUCTION-FAILURE"
INSTRUMENT_BUG = "INSTRUMENT-BUG"
NO_REPRO = "NO-REPRO"
AMBIGUOUS_NONDETERMINISM = "AMBIGUOUS-NONDETERMINISM"
REAL_CAPACITY_BOUNDARY = "REAL-CAPACITY-BOUNDARY"
TOLERANCE_MISCALIBRATION = "TOLERANCE-MISCALIBRATION"
AMBIGUOUS_RESIDUAL = "AMBIGUOUS-RESIDUAL"

K_REPRO_CELL = 84                        # sec 15.24.2's own pinned cell (K=84, seed=1940, d_state=96)
D_STATE_REPRO_CELL = 96
STEP_NEG1_MIN_EVENTS = 3                 # sec 15.24.4's pinned Step -1 minimum
STEP2_NITER_SWEEP = (24, 28, 32, 40)     # sec 15.24.4, the SAME grid sec 15.23's own sweep used


def _load_ns_orthogonalize():
    """Lazy import of model_rd.newton_schulz_orthogonalize -- see this
    module's own docstring for why this cannot be a top-level import."""
    from model_rd import newton_schulz_orthogonalize
    return newton_schulz_orthogonalize


@contextlib.contextmanager
def _tf32_mode(matmul: bool, cudnn: bool):
    """Sec 15.24.4's TF32 precision pinning (Step 1 only): temporarily force
    torch.backends.cuda.matmul.allow_tf32 / torch.backends.cudnn.allow_tf32
    to the given values, restoring whatever was there before on exit --
    never leaves the process-global backend state mutated after this
    context manager returns (safe to nest / call repeatedly per event)."""
    orig_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = orig_matmul
        torch.backends.cudnn.allow_tf32 = orig_cudnn


def event_key(event: dict) -> tuple:
    """Sec 15.24.4's event identity: (seed, step, hop, batch_idx) --
    launch-unique across a combined, up-to-3-launch sink (Rev 3 MAJOR-A
    fix: `seed` is what makes two independent launches' events at the
    identical (step,hop,batch_idx) coordinates distinguishable)."""
    return (event["seed"], event["step"], event["hop"], event["batch_idx"])


def episode_key(event: dict, row_idx: int) -> tuple:
    """Sec 15.24.4's episode identity: (seed, step, hop, batch_idx,
    row_idx) -- ONE within-batch (K,d) problem."""
    return event_key(event) + (row_idx,)


# ---------------------------------------------------------------------------
# Prerequisite: offline pool reconstruction + the two-seeds-trap gate
# (Rev 4, MAJOR-1 fix). Runs ONCE PER LAUNCH, ahead of EVERY verdict-
# producing step, including Step 0b.
# ---------------------------------------------------------------------------

def reconstruct_pools_offline(tokenizer):
    """Sec 15.24.4's pinned, hardcoded reconstruction call -- the two-seeds
    trap fix (Rev 4 MAJOR-1): offline pool reconstruction is ALWAYS
    `grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)`, NEVER
    the launch seed (1940/1943/1944), under ANY circumstance -- a builder
    threading the launch seed here would silently reconstruct the WRONG
    train/held-out partition (a different `random.Random(seed).shuffle`
    permutation), producing a confidently wrong, total INSTRUMENT-BUG
    verdict on healthy code. heldout_frac=0.5 matches this cell's own CLI
    default -- run_deltanet_rd_exactness_sweep.py's build_cmd() never emits
    --heldout-frac for a keyanchor-scaling cell, so the launch always takes
    run_deltanet_rd.py's own CLI default (0.5)."""
    pools, _report = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    return pools


def check_reconstruction_gate(pools, archived_anchor_train_ids: torch.Tensor) -> dict:
    """Sec 15.24.4's new prerequisite gate (Rev 4 MAJOR-1 fix): assert the
    offline-reconstructed `pools.train_name_ids` is SET-EQUAL (order-
    independent) to that SAME launch's own archived `anchor_train_ids`
    tensor (run_deltanet_rd.py:934-936, `model.anchor_train_ids_buf`,
    registered directly from that launch's own `pools.train_name_ids` at
    construction time) -- the live run's own ground truth, zero new
    telemetry cost. Does NOT itself abort; the caller (run_full_precedence)
    decides what a failed check means."""
    reconstructed = set(int(x) for x in pools.train_name_ids.tolist())
    archived = set(int(x) for x in archived_anchor_train_ids.tolist())
    passed = reconstructed == archived
    return {
        "passed": passed,
        "n_reconstructed": len(reconstructed), "n_archived": len(archived),
        "symmetric_diff_size": len(reconstructed ^ archived),
    }


# ---------------------------------------------------------------------------
# Step 0b: pool-membership precheck. Structural, dispositive on ANY SINGLE
# violation (Rev 3 FATAL fix) -- no event-count or episode-count minimum.
# ---------------------------------------------------------------------------

def step_0b_pool_membership(events: list, pools, k: int = K_REPRO_CELL) -> dict:
    """Sec 15.24.4 Step 0b: a dumped `key_ids` row's membership in
    `pools.heldout_name_ids` is a STRUCTURAL fact (zero floating-point
    arithmetic) -- a lone violation, in a sink of ANY size (including a
    single-event sink, before Step -1's own <3-event gate has even run),
    is already deterministic proof the probe path is broken. No "below the
    floor, excluded" case exists for this row (Rev 3, superseding Rev 2's
    floor-gated version)."""
    heldout_set = set(int(x) for x in pools.heldout_name_ids.tolist())
    violations = []
    for ev in events:
        key_ids = ev["key_ids"]
        B, dumped_k = key_ids.shape[0], key_ids.shape[1]
        if dumped_k != k:
            violations.append({"event": event_key(ev), "row_idx": None,
                                "reason": f"dumped K={dumped_k} != expected {k} (cfg.K mismatch)"})
            continue
        for row_idx, row in enumerate(key_ids.tolist()):
            bad_ids = [eid for eid in row if eid not in heldout_set]
            if bad_ids:
                violations.append({"event": event_key(ev), "row_idx": row_idx, "bad_ids": bad_ids})
    return {"dispositive": len(violations) > 0, "n_violations": len(violations), "violations": violations}


# ---------------------------------------------------------------------------
# Step -1: zero/low-event guard.
# ---------------------------------------------------------------------------

def step_neg1_event_guard(events: list, min_events: int = STEP_NEG1_MIN_EVENTS) -> dict:
    """Sec 15.24.4 Step -1: counted over seed-qualified DISTINCT
    (seed,step,hop,batch_idx) events (Rev 3 MAJOR-A fix) -- checked AFTER
    Step 0b, BEFORE Step 0a/1/2. `<min_events` -> refuse Step 0a/1/2, emit
    NO-REPRO (single launch) or AMBIGUOUS-NONDETERMINISM (combined sink
    still short) -- see run_full_precedence's own branch, which decides
    which of the two applies (this function only counts)."""
    distinct = {event_key(ev) for ev in events}
    return {"n_distinct_events": len(distinct), "min_events": min_events,
            "cleared": len(distinct) >= min_events}


# ---------------------------------------------------------------------------
# Step 0a: exact-rank precheck. CORROBORATING ONLY, never dispositive on its
# own (Rev 1 MINOR m2 demotion, unaffected by Rev 2/Rev 3).
# ---------------------------------------------------------------------------

def _apply_two_level_floor(anomalous_episodes: list) -> dict:
    """Pure, CPU-only floor-counting logic (no NS recompute) -- sec
    15.24.4's two-level absolute floor: >=2 anomalous episodes (rows),
    occurring in >=2 DISTINCT events, BOTH required jointly (Rev 2 FATAL
    fix; Rev 3 scopes this specific floor to Step 1's own marker alone,
    since Step 0b no longer has a floor of its own -- 0a's own recurrence
    disclosure reuses this SAME floor, sec 15.24.4). `anomalous_episodes`:
    a list of `{"event": event_key_tuple, "row_idx": int, ...}` dicts.

    NOT floor_met -> every episode in `anomalous_episodes` is flagged,
    EXCLUDED, and disclosed (E2's own worked example, sec 15.24.4: 3
    anomalous episodes concentrated in ONE event still means ALL 3 are
    excluded, not just the first) -- the verdict is computed on whatever
    remains outside `excluded_episodes`.

    Split out from the real recompute (`_step_1_recompute_disagreements`)
    so diag_c17_repro_stage_minus1.py's own boundary fixtures (item 7) can
    feed hand-built anomaly lists directly, with NO CUDA/fla dependency at
    all -- this is the unit-test-isolation decision this module's own
    docstring registers."""
    n_events = len({e["event"] for e in anomalous_episodes})
    floor_met = len(anomalous_episodes) >= 2 and n_events >= 2
    excluded_episodes = []
    if anomalous_episodes and not floor_met:
        excluded_episodes = [tuple(e["event"]) + (e["row_idx"],) for e in anomalous_episodes]
    return {"n_anomalous_episodes": len(anomalous_episodes), "n_distinct_events": n_events,
            "floor_met": floor_met, "excluded_episodes": excluded_episodes}


def step_0a_exact_rank(events: list, k: int = K_REPRO_CELL, tol: float = 1e-4) -> dict:
    """Sec 15.24.4 Step 0a: `torch.linalg.matrix_rank(k_eff_raw_episode,
    tol=1e-4) < K`, per episode, for every dumped episode at or past Step
    -1's gate. CORROBORATING evidence for INSTRUMENT-BUG only -- reported
    and disclosed alongside whatever Step 1 finds, never gating on its
    own. Pure CPU-only tensor op (`torch.linalg.matrix_rank`) -- no
    model_rd/fla dependency."""
    flagged = []
    for ev in events:
        k_eff_raw = ev["k_eff_raw"]
        for row_idx in range(k_eff_raw.shape[0]):
            rank = int(torch.linalg.matrix_rank(k_eff_raw[row_idx], tol=tol).item())
            if rank < k:
                flagged.append({"event": event_key(ev), "row_idx": row_idx, "rank": rank})
    floor = _apply_two_level_floor(flagged)
    return {"flagged": flagged, **floor,
            "note": "CORROBORATING ONLY -- never dispositive on its own (Rev 1 MINOR m2 demotion)."}


# ---------------------------------------------------------------------------
# Step 1: live/offline NS agreement. Dispositive once its OWN two-level
# floor is met (Rev 2, unchanged by Rev 3's own Step-0b split).
# ---------------------------------------------------------------------------

def _step_1_recompute_disagreements(events: list, tf32_by_seed: dict, n_iter: int = 20,
                                     resid_tol: float | None = None, device: str | None = None) -> dict:
    """BOX-ONLY (imports model_rd.newton_schulz_orthogonalize -- see this
    module's own docstring). Sec 15.24.4 Step 1 / Rev 1 MAJOR M1 (dual TF32
    mode) / Rev 3 MAJOR-B (batched, NEVER a per-row singleton slice --
    cuBLAS/cuDNN GEMM kernel selection is batch-size-dependent, so a
    singleton recompute can silently route through a different kernel and
    flip a near-boundary residual from that alone). For every event: ONE
    batched call on the event's FULL dumped (B,K,d) tensor, in BOTH
    matched-mode (this event's OWN source-run TF32 flags, PRIMARY) and
    strict-fp32 (corroborating). Returns raw per-episode disagreements +
    the TF32-SENSITIVE sub-finding (matched vs strict flip, independent of
    whether either agrees with the live flag -- routed to Step 2, never
    counted toward the INSTRUMENT-BUG floor)."""
    newton_schulz_orthogonalize = _load_ns_orthogonalize()
    resid_tol = ka.GATE2_RESID_TOL if resid_tol is None else resid_tol
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    disagreements, tf32_sensitive = [], []
    for ev in events:
        k_eff_raw = ev["k_eff_raw"].to(device)
        live_resid = ev["resid"]
        tf32 = tf32_by_seed.get(ev["seed"], {"tf32_matmul": False, "tf32_cudnn": False})
        with torch.no_grad():
            with _tf32_mode(tf32["tf32_matmul"], tf32["tf32_cudnn"]):
                _, resid_matched = newton_schulz_orthogonalize(k_eff_raw, n_iter=n_iter)
            with _tf32_mode(False, False):
                _, resid_strict = newton_schulz_orthogonalize(k_eff_raw, n_iter=n_iter)
        resid_matched = resid_matched.detach().cpu()
        resid_strict = resid_strict.detach().cpu()
        for row_idx in range(k_eff_raw.shape[0]):
            live_flag = bool(live_resid[row_idx].item() > resid_tol)
            matched_flag = bool(resid_matched[row_idx].item() > resid_tol)
            strict_flag = bool(resid_strict[row_idx].item() > resid_tol)
            if live_flag != matched_flag:
                disagreements.append({"event": event_key(ev), "row_idx": row_idx,
                                       "live_resid": live_resid[row_idx].item(),
                                       "offline_resid_matched": resid_matched[row_idx].item()})
            if matched_flag != strict_flag:
                tf32_sensitive.append({"event": event_key(ev), "row_idx": row_idx,
                                        "offline_resid_matched": resid_matched[row_idx].item(),
                                        "offline_resid_strict": resid_strict[row_idx].item()})
    return {"disagreements": disagreements, "tf32_sensitive": tf32_sensitive}


def step_1_live_offline_agreement(events: list, tf32_by_seed: dict, n_iter: int = 20,
                                   resid_tol: float | None = None, device: str | None = None) -> dict:
    """Sec 15.24.4 Step 1, full (recompute + floor). BOX-ONLY -- see
    `_step_1_recompute_disagreements`'s own docstring. `floor_met` ->
    dispositive INSTRUMENT-BUG (checked by the caller, run_full_
    precedence); NOT floor_met -> `excluded_episodes` names whatever must
    be dropped before Step 2 runs."""
    recompute = _step_1_recompute_disagreements(events, tf32_by_seed, n_iter=n_iter,
                                                 resid_tol=resid_tol, device=device)
    floor = _apply_two_level_floor(recompute["disagreements"])
    return {**recompute, **floor}


# ---------------------------------------------------------------------------
# Step 2: n_iter sweep. Only reached if Step 1 agrees everywhere (after
# excluding whatever Step 1's own floor logic excluded).
# ---------------------------------------------------------------------------

def _step_2_recompute_resolve_niter(events: list, excluded_episodes=(), sweep=STEP2_NITER_SWEEP,
                                     resid_tol: float | None = None, device: str | None = None) -> dict:
    """BOX-ONLY (same reason as _step_1_recompute_disagreements). Sweeps
    `n_iter` on the SAME dumped `k_eff_raw`, per episode (excluding
    whatever Step 1 already excluded). Returns {episode_5tuple:
    n_iter_first_resolved (int) or None (never resolves in the grid)}.

    C17 build audit MEDIUM-2 fix (mirrors Step 1's own Rev 3 MAJOR-B fix,
    `_step_1_recompute_disagreements` above): for EACH `n` in `sweep`, this
    runs ONE BATCHED `newton_schulz_orthogonalize` call on the event's FULL
    dumped `(B,K,d)` tensor -- NEVER a `k_eff_raw[row_idx:row_idx+1]`
    singleton slice inside the per-row loop, which reintroduces the exact
    batch-size GEMM-kernel-selection confound MAJOR-B killed for Step 1
    (cuBLAS/cuDNN kernel selection is batch-size-dependent; a singleton
    recompute can silently route through a different kernel and flip a
    near-boundary residual from batching alone, independent of any real
    numerical disagreement). `resid_by_n[n][row_idx]` is indexed from the
    batched result, never recomputed singleton -- same discipline as Step
    1's `resid_offline[row_idx]`."""
    newton_schulz_orthogonalize = _load_ns_orthogonalize()
    resid_tol = ka.GATE2_RESID_TOL if resid_tol is None else resid_tol
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    excluded = set(excluded_episodes)
    per_episode: dict = {}
    for ev in events:
        k_eff_raw = ev["k_eff_raw"].to(device)           # (B,K,d) -- FULL event tensor
        B = k_eff_raw.shape[0]
        resid_by_n = {}
        for n in sweep:
            with torch.no_grad():
                _, resid = newton_schulz_orthogonalize(k_eff_raw, n_iter=n)   # ONE batched call
            resid_by_n[n] = resid
        for row_idx in range(B):
            episode = episode_key(ev, row_idx)
            if episode in excluded:
                continue
            resolved_at = None
            for n in sweep:
                if resid_by_n[n][row_idx].item() <= resid_tol:
                    resolved_at = n
                    break
            per_episode[episode] = resolved_at
    return per_episode


def _step_2_verdict(per_episode_resolve_niter: dict) -> str | None:
    """Pure verdict-decision logic from a precomputed {episode:
    n_iter_or_None} map -- NO NS recompute, fully CPU-testable.

    DISCLOSED INTERPRETATION (this build session): sec 15.24.4's own Step-2
    table states three clauses that are NOT, read literally and
    independently, a mutually-exclusive partition -- "ANY episode still
    >0.01 at n_iter=40" and "Mixed... some resolve, some don't" overlap
    verbatim as written, and this ambiguity is UNCHANGED across all 4
    attack rounds on this exact wording (never flagged as a finding).
    Resolved here as a mutually-exclusive, exhaustive partition matching
    candidates (a)/(c)'s own PRECISE sec 15.24.1 definitions:
      - TOLERANCE-MISCALIBRATION iff EVERY episode resolves, each one at
        n_iter<=32 (candidate (c)'s own text: "the sweep DOES drive every
        flagged episode below 0.01, at or before n_iter=32").
      - REAL-CAPACITY-BOUNDARY iff >=1 episode NEVER resolves within the
        tested grid -- still failing at n_iter=40, the widest tested value
        (candidate (a)'s own text: "sweeping... does NOT drive every
        flagged episode's residual below 0.01").
      - AMBIGUOUS-RESIDUAL is the residual case: EVERY episode eventually
        resolves, but >=1 needed the FULL grid (n_iter=40 specifically) to
        do so -- neither a clean fast fix (TOLERANCE-MISCALIBRATION) nor a
        structural wall (REAL-CAPACITY-BOUNDARY)."""
    if not per_episode_resolve_niter:
        return None
    resolves = list(per_episode_resolve_niter.values())
    if all(n is not None and n <= 32 for n in resolves):
        return TOLERANCE_MISCALIBRATION
    if any(n is None for n in resolves):
        return REAL_CAPACITY_BOUNDARY
    return AMBIGUOUS_RESIDUAL


def step_2_niter_sweep(events: list, excluded_episodes=(), sweep=STEP2_NITER_SWEEP,
                        resid_tol: float | None = None, device: str | None = None) -> dict:
    """Sec 15.24.4 Step 2, full (recompute + verdict). BOX-ONLY -- see
    `_step_2_recompute_resolve_niter`'s own docstring."""
    per_episode = _step_2_recompute_resolve_niter(events, excluded_episodes=excluded_episodes,
                                                   sweep=sweep, resid_tol=resid_tol, device=device)
    verdict = _step_2_verdict(per_episode)
    return {"per_episode_resolve_niter": {str(k): v for k, v in per_episode.items()},
            "n_episodes": len(per_episode), "verdict": verdict}


# ---------------------------------------------------------------------------
# Launch loading.
# ---------------------------------------------------------------------------

def load_launch_dump(ckpt_dir: str, seed: int) -> dict:
    """Loads ONE launch's own `c17fallback_step*.pt` files (concatenating
    their `events` lists into one -- the FULL sink for THAT launch) + its
    own archived `anchor_train_ids` (from any anchor-table `step*.pt`
    checkpoint in the same ckpt_dir -- run_deltanet_rd.py:934-936, already
    logged, zero new cost). Asserts every loaded event's own `seed` field
    matches the `seed` passed here (a wrong --launch SEED argument, or a
    mixed-launch ckpt_dir, is a build-time-catchable mistake, not a silent
    corruption) and that the two TF32 flags are constant across every file
    in this ckpt_dir (a process-level setting; genuinely differing values
    would mean two different processes wrote into the same directory)."""
    fallback_paths = sorted(glob.glob(os.path.join(ckpt_dir, "c17fallback_step*.pt")))
    assert fallback_paths, f"no c17fallback_step*.pt files found in {ckpt_dir!r} for seed={seed}"
    events: list = []
    tf32_matmul = tf32_cudnn = None
    for p in fallback_paths:
        doc = torch.load(p, map_location="cpu", weights_only=False)
        if tf32_matmul is None:
            tf32_matmul, tf32_cudnn = doc["tf32_matmul"], doc["tf32_cudnn"]
        else:
            assert doc["tf32_matmul"] == tf32_matmul and doc["tf32_cudnn"] == tf32_cudnn, (
                f"TF32 flags differ across {ckpt_dir!r}'s own c17fallback files ({p!r}) -- a "
                f"process-level setting should be CONSTANT within one launch")
        for ev in doc["events"]:
            assert ev["seed"] == seed, (
                f"event seed={ev['seed']!r} != the launch seed {seed!r} passed via --launch for "
                f"{ckpt_dir!r} -- wrong --launch SEED argument, or a mixed-launch ckpt_dir")
            # C17 build audit MEDIUM-3 fix: k_blend_raw is dumped (sec 15.24.2 item (ii)) but
            # every downstream step (0b/0a/1/2) reads k_eff_raw ONLY -- k_blend_raw would be dead
            # cargo unless something actually reads it back. Defense-in-depth, near-zero cost
            # (one torch.equal on already-loaded tensors): re-assert the SAME bitwise-equality
            # invariant run_deltanet_rd.py's `_dump_c17_fallback_event` HARD-ABORTS on at dump
            # time (sec 15.24.2's own "100% of C17 keys are trained_here=False" claim) -- a
            # second, independent check at LOAD time, on the PERSISTED artifact, catching a
            # corrupted/hand-edited dump file the dump-time assert never saw. Same failure mode,
            # same pinned hard-abort path: an uncaught AssertionError, never a soft warning.
            assert torch.equal(ev["k_eff_raw"], ev["k_blend_raw"]), (
                "C17 REPRO ANALYSIS HARD-ABORT (KEY_ANCHORING_SCALING_DRAFT.md sec 15.24.5 "
                "MEDIUM-3 defense-in-depth): k_eff_raw and k_blend_raw are NOT bitwise-identical "
                f"for an event loaded from {p!r} (seed={seed} step={ev.get('step')} "
                f"hop={ev.get('hop')} batch_idx={ev.get('batch_idx')}) -- either the dump-time "
                "HARD-ABORT (run_deltanet_rd.py's _dump_c17_fallback_event) was bypassed, or this "
                "file was corrupted/hand-edited after being written. Every downstream Step "
                "0b/0a/1/2 verdict computed from k_eff_raw assumes this equality; refusing to "
                "proceed on an unverified assumption.")
            events.append(ev)

    anchor_paths = sorted(
        p for p in glob.glob(os.path.join(ckpt_dir, "step*.pt"))
        if not os.path.basename(p).startswith("full_step"))
    assert anchor_paths, f"no anchor-table step*.pt checkpoint found in {ckpt_dir!r}"
    anchor_doc = torch.load(anchor_paths[0], map_location="cpu", weights_only=False)

    return {"seed": seed, "ckpt_dir": ckpt_dir, "events": events,
            "tf32_matmul": tf32_matmul, "tf32_cudnn": tf32_cudnn,
            "anchor_train_ids": anchor_doc["anchor_train_ids"]}


# ---------------------------------------------------------------------------
# Total precedence orchestrator.
# ---------------------------------------------------------------------------

def run_full_precedence(launches: list, tokenizer, k: int = K_REPRO_CELL) -> dict:
    """Sec 15.24.4's Total precedence, EXACTLY: reconstruction gate
    (prerequisite, HARD-ABORT-only, no verdict) > 0b (structural) > Step -1
    (reproduction) > Step 1 (agreement) > Step 2 (tolerance). `launches`:
    1-3 dicts, each `load_launch_dump()`'s own return shape -- the primary
    seed alone, or primary + up to 2 contingency seeds on a PRIOR NO-REPRO
    (sec 15.24.4/15.24.7's own registered, human-loop-gated follow-up --
    this function NEVER fires a contingency seed itself; it only combines
    whatever `launches` the caller already assembled).

    Step -1's own "<3, single launch -> NO-REPRO" vs "<3, combined (>1
    launches) -> AMBIGUOUS-NONDETERMINISM" branch (sec 15.24.4) is decided
    HERE, from `len(launches)`, since `step_neg1_event_guard` itself only
    counts -- it has no way to know whether its input is a primary-only or
    an already-combined sink."""
    assert 1 <= len(launches) <= 3, \
        f"sec 15.24.2: at most 3 launches (primary + 2 contingency), got {len(launches)}"
    result = {"launches": [{"seed": l["seed"], "ckpt_dir": l.get("ckpt_dir"), "n_events": len(l["events"])}
                            for l in launches]}

    # Prerequisite: reconstruction gate, run ONCE PER LAUNCH, before that
    # launch's events are folded into the combined sink (sec 15.24.4's own
    # "Total precedence" paragraph).
    pools = reconstruct_pools_offline(tokenizer)
    recon_checks = []
    for l in launches:
        chk = check_reconstruction_gate(pools, l["anchor_train_ids"])
        chk["seed"] = l["seed"]
        recon_checks.append(chk)
    result["reconstruction_checks"] = recon_checks
    failed = [c for c in recon_checks if not c["passed"]]
    if failed:
        result["verdict"] = RECONSTRUCTION_FAILURE
        result["note"] = (
            "offline-reconstructed pools.train_name_ids (grd.build_entity_pools(tokenizer, "
            "heldout_frac=0.5, seed=0) -- the pinned, hardcoded call, NEVER the launch seed) does "
            f"NOT set-equal the archived anchor_train_ids for launch seed(s) "
            f"{[c['seed'] for c in failed]} -- HARD-ABORT before ANY verdict-producing step runs "
            "(sec 15.24.4's two-seeds-trap prerequisite gate). No verdict emitted.")
        return result

    combined = []
    for l in launches:
        combined.extend(l["events"])
    tf32_by_seed = {l["seed"]: {"tf32_matmul": l["tf32_matmul"], "tf32_cudnn": l["tf32_cudnn"]}
                     for l in launches}

    # Step 0b -- structural, dispositive on ANY single violation, no floor,
    # runs on WHATEVER events exist so far, ahead of Step -1's own gate.
    b0 = step_0b_pool_membership(combined, pools, k=k)
    result["step_0b"] = b0
    if b0["dispositive"]:
        result["verdict"] = INSTRUMENT_BUG
        result["verdict_source"] = "step_0b"
        return result

    # Step -1 -- zero/low-event guard, checked AFTER 0b, BEFORE 0a/1/2.
    neg1 = step_neg1_event_guard(combined)
    result["step_neg1"] = neg1
    if not neg1["cleared"]:
        if len(launches) == 1:
            result["verdict"] = NO_REPRO
            # KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K only has entries for the REAL, registered
            # K-grid (72/78/84/90 at d_state=96, etc) -- a synthetic/non-repro-cell k (e.g. a Stage
            # -1 fixture's small test K) has no reserved contingency pair, so the seed numbers are
            # only named when this IS the real, pinned repro cell.
            contingency = (rdx.KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K.get(D_STATE_REPRO_CELL, {})
                           .get(k))
            seed_note = (f"fire contingency seeds {contingency}" if contingency is not None
                         else f"fire the reserved contingency seed pair for K={k} (not registered "
                              f"for this non-repro-cell k -- look up the real cell's own K)")
            result["followup"] = (
                "registered follow-up (sec 15.24.4/15.24.7, human-loop gated, NEVER auto-fired by "
                f"this script): {seed_note}, re-run the identical cell + telemetry, combine the "
                "resulting dump sink with this launch's own (each event carrying its own seed), and "
                "re-run this script with --launch given for EVERY launch present so far -- "
                "re-running Step 0b on the newly-combined sink FIRST (0b's precedence is "
                "unconditional, applies to every combined-sink state).")
        else:
            result["verdict"] = AMBIGUOUS_NONDETERMINISM
            result["followup"] = (
                "combined total STILL below the <3-event minimum (and 0b has not fired) -- promotes "
                "candidate (1), sec 15.24.3's fixed-batch checkpoint-payload extension, to the "
                "primary next instrument (sec 15.24.4).")
        return result

    # Step 0a -- corroborating only, never dispositive.
    a0 = step_0a_exact_rank(combined, k=k)
    result["step_0a"] = a0

    # Step 1 -- live/offline NS agreement.
    s1 = step_1_live_offline_agreement(combined, tf32_by_seed)
    result["step_1"] = s1
    if s1["floor_met"]:
        result["verdict"] = INSTRUMENT_BUG
        result["verdict_source"] = "step_1"
        return result

    # Step 2 -- n_iter sweep, only reached if Step 1 agrees everywhere
    # (after excluding whatever Step 1's own floor logic excluded).
    s2 = step_2_niter_sweep(combined, excluded_episodes=s1["excluded_episodes"])
    result["step_2"] = s2
    result["verdict"] = s2["verdict"]
    result["verdict_source"] = "step_2"
    return result


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launch", nargs=2, action="append", default=[], metavar=("CKPT_DIR", "SEED"),
                     help="one launch's ckpt_dir (containing c17fallback_step*.pt + an anchor-table "
                          "step*.pt checkpoint) + its own --seed value. Repeat up to 3x (sec 15.24.2: "
                          "primary seed 1940, plus contingency 1943/1944 on a PRIOR NO-REPRO). This "
                          "script NEVER launches contingency seeds itself (sec 15.24.7) -- combine "
                          "launches only after a human has fired the registered follow-up.")
    ap.add_argument("--k", type=int, default=K_REPRO_CELL)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()
    assert args.launch, "--launch CKPT_DIR SEED is required at least once (sec 15.24.2)"
    assert len(args.launch) <= 3, f"sec 15.24.2: at most 3 launches, got {len(args.launch)}"

    launches = [load_launch_dump(ckpt_dir, int(seed_str)) for ckpt_dir, seed_str in args.launch]
    tokenizer = grd.load_gpt2_tokenizer()
    result = run_full_precedence(launches, tokenizer, k=args.k)

    print(json.dumps(result, indent=2, default=str))
    src = f" (via {result['verdict_source']})" if "verdict_source" in result else ""
    print(f"\nVERDICT: {result.get('verdict')}{src}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
