#!/usr/bin/env python3
"""lm_recall_gap_probe_v2_rd.py -- PARAM_AXIS_SCALING_DESIGN.md sec 9 (REV 2)
instrument REBUILD, replacing the VOID `lm_recall_gap_probe_rd.py`
(matrix-thinking/queue/regate_2026-07-12.md sec 10: FATAL-1 shared-tensor
ablation, F-4 differential candidate cap, F-3 toothless Wave -1, S-6/S-7
provenance bugs, M-11 T2 weakened post-failure). That file is NOT deleted
(it is the artifact of record for the build+audit chain) but must NEVER be
imported by anything that wants a verdict-grade number. THIS file supersedes
it per sec 9's pinned re-registration.

WHAT CHANGED, one line each (full rationale in sec 9 of the design doc):
  1. ONE corrupted token per forward-pass ROW, via row-replication --
     batching over CANDIDATES, not over rows (sec 9.2). FATAL-1's fix.
  2. A PLACEBO-ABLATION arm (one non-antecedent token, Delta-distribution-
     matched, same replacement RNG stream) and DiD = gap_true - gap_placebo
     as the ONLY quantity that may carry a verdict (sec 9.2). No function in
     this module will hand back a raw AR-hit slice or an un-placebo'd gap as
     if it were a capacity measurement -- see `finalize_cell` / VerdictGradeError.
  3. T2 split into T2a (instrument-teeth, absolute bar, reference models --
     wired but NOT executed in this build session, sec 9.4) and T2b
     (rung-admissibility, mechanism-present test T2b-1 + the NEW ceiling
     check T2b-2: DiD(r) <= acc_copy(r) + 2*SE -- the single line that would
     have caught the VOID build's wikitext self-contradiction, regate 10.2).
  4. The eval micro-batch is fully decoupled from the token-arithmetic
     batch_size and from candidate selection (kills F-4: the old per-BATCH
     candidate cap silently made the batch-16 1.31B rung the only uncapped
     cell).
  5. Checkpoints must be QUIESCED (md5-stable across a wait window) before
     they may be loaded for a verdict-grade read (kills S-7: a prior run
     raced a live --ckpt-every writer).

WHAT DID NOT CHANGE (sec 9.0: "read off the (VOID) implementation's
candidate construction, which is correct and is retained"):
  - The candidate definition (second occurrence of a bigram whose first
    occurrence is >min_sep tokens back, and whose continuation is NOT the
    corpus TRAIN-split modal continuation).
  - The FIX-A token-matched-checkpoint-slice resolver (`resolve_token_matched_
    checkpoint` / `list_checkpoint_steps`) -- reused VERBATIM, unmodified,
    per this build's explicit instruction. It was never named as defective.
  - The T2 marker-token picker (`pick_t2_marker_tokens`, high-entropy /
    OOD-pairing selection) -- reused verbatim; only the DISTANCE and BAR it
    fed into were the M-11 violation, not the token-selection logic itself.

THE NORMALIZATION SLOT (sec 9.1) -- PINNED 2026-07-12 (commit 7381d2a, blind
agent, post-quarantine): M(r) = DiD(r), denominator 1 -- FORCED by N1-N3's
algebraic identity (sec 9.1.1: the intact arm cancels exactly, so DiD is
already competence-invariant and any denominator that varies with general
competence would re-inject the exact confound the numerator purged).
Registered here as `"raw_did"`, in the exact form the pin's own text
specifies: `register_normalization("raw_did", lambda cell: cell["did"])`.
This was NOT this build's decision to make -- this build's earlier session
read regate sec 10.2 and was therefore contaminated for *which* form leans
which way, so the slot originally shipped EMPTY on purpose. It is filled now
only because a SEPARATE, genuinely blind agent derived the forced answer
independently (sec 9.1.1) and the coordinator directed it be wired in.
`NORMALIZATION_REGISTRY` is still a live, general mechanism -- it is not
special-cased or hardcoded away: `register_normalization` and
`compute_capacity_metric`'s registry lookup are UNCHANGED by the pin
landing, so a future metric (or a PI override) can still be added the same
way. `normalization` remains a REQUIRED keyword with no default; passing
`None` or an unregistered name still raises loudly, and nothing here ever
silently falls back to a particular form.

sec 9.1.5 MANDATORY SENSITIVITIES (verdict-WITHHOLDING only -- can only
downgrade a verdict to INDETERMINATE, never create or strengthen one):
  S1 `compute_s1_utilization(cell)` -- the utilization ratio DiD/acc_copy
     (rejected normalization candidate C5, promoted to a companion metric).
     Never fed into compute_verdict/compute_capacity_metric.
  S2 `DiD_logp` -- log-prob readout guarding the argmax-floor bias. BUILD
     REQUIREMENT: every candidate record carries `logp_intact`,
     `logp_true_ablated`, `logp_placebo_ablated` (log_softmax+gather on the
     SAME forward passes already run -- zero extra compute). A cell missing
     any of these on a resolved candidate is VOID (`cell_void_missing_s2_
     fields`, enforced in `compute_capacity_metric`) -- sec 9.6's stop rule:
     S2 "cannot be added after a read." `compute_verdict_with_s2` enforces
     the pre-committed disagreement rule: if S2's Factor-1 trend classification
     differs from the primary's, the verdict is forced to INDETERMINATE.

Zero training. Reuses DeltaNetLM / load_corpus / get_batch / CORPUS_DIRS /
EOT_TOKEN_ID / corpus_fixed_seed / DEFAULT_DATA_DIR from lm_pretrain_rd.py
(VERBATIM import, not reimplemented) via h2h_fla_stub_rd.ensure_fla_stub()
(this repo's own CPU-stub-or-real-fla convention -- lets `--smoke` run
logic-only on a CPU dev box while `--run` on the H100 box picks up real fla
automatically).

Usage:
  python lm_recall_gap_probe_v2_rd.py --smoke
  python lm_recall_gap_probe_v2_rd.py --resolve-slice --ckpt-dir <dir> \
      --batch-size 32 --seq-len 512 --target-tokens 327680000 --prefix <p>
  python lm_recall_gap_probe_v2_rd.py --verify-quiesced --checkpoint <path.pt>
  python lm_recall_gap_probe_v2_rd.py --run --checkpoint <path.pt> \
      --corpus openr1-mix-ext --data-dir /data/deltanet_rd_data \
      --attest-job-terminated --out r.json
      (GPU + real fla required. Computes DiD/T1a ONLY -- NOT a verdict, and NOT
      T2b/sec-9.6 admissibility, which need the driver that sec 9.1's pin unblocks.
      `--attest-job-terminated` is REQUIRED: file-quiescence cannot establish sec 9.6
      item 3's second clause, because a --ckpt-every-10000 writer leaves a byte-stable
      file for HOURS between writes -- that is the S-7 read. `--compute-verdict` is
      REFUSED by this entry point.)

AUDIT (2026-07-12, independent adversarial audit of this rebuild). Fixed in place:
  * F-4 REBORN (was FATAL, latent): run_did_eval drew its windows in `batch_size`-sized
    get_batch chunks. On CUDA (Philox advances per KERNEL LAUNCH) that makes the WINDOW
    POPULATION batch-size-dependent -- measured on the H100: batch_size 16 vs 32 at the
    same seed shared only 50% of their 512 windows. The rungs use different
    token-arithmetic batch sizes (three at 32, the 1.31B at 16), so the top rung would
    again have been measured against a different candidate population. Now: one
    get_batch call; `batch_size` is provenance-only. Guarded by smoke [10c] (teeth on
    CUDA only) + a forced-fail negative test.
  * The FATAL-1 invariant is now ASSERTED ON THE PRODUCTION PATH (run_ablation_arm), not
    only on `build_replicated_ablation_batch`, which the real path never calls.
  * A placebo-match failure (>5% flagged) now maps to VOID, not FLOOR (sec 9.2 is literal).
  * compute_capacity_metric now REQUIRES `t2b2_pass=` and refuses placebo-unmatched cells;
    the `_verdict_grade` stamp only ever meant "a placebo arm was run."
  * tost_flat returned an 80% CI while claiming (and needing) the pinned 90% CI -- FLAT was
    declared more easily than the pre-registration permits.
  * `wilson_se` was Wald, not Wilson (behaviour right, name wrong). Renamed `binomial_se`.
  * The realized placebo Delta profile is now REPORTED (`delta_match`); it is ~35% shorter
    than the true profile by construction (draw-then-reject truncates at k) and nothing
    measured it before.

AUDIT 2 (2026-07-12, independent follow-up audit, scoped to the sec 9.1/9.1.5 S1+S2+
normalization code that landed AFTER the audit above). The S2 log-prob ARITHMETIC and the
candidate-detection REORDER both verified CLEAN against hand-built ground truth (owner-map
skips, cross-chunk indexing, k-vs-k+1 target, batch-size and eval-micro-batch invariance).
Fixed in place:
  * S2's log_softmax was run on the FULL (n_chunk, T, V) logits tensor and then read ONE
    element per row -- DOUBLING the peak VRAM of the pipeline's dominant allocation (measured
    on the H100 at the pinned eval_micro_batch=64/T=512/V=50257/fp32: 6.14 GiB -> 12.27 GiB)
    to keep 1/512 of it. Now gathers the query positions FIRST and log_softmaxes the
    (n_cand, V) slice: 6.16 GiB, identical arithmetic (max |diff| 9.5e-7, fp32 reduction-order
    noise). "Zero extra compute" was true; "zero extra memory" was not, and the top rung is
    exactly where that bites.
  * sec 9.1.5's S2 disagreement rule was enforced ONLY in the `compute_verdict_with_s2`
    wrapper, while `compute_verdict` -- the name sec 9.5's own prose uses -- stayed callable
    with NO S2 disposition and would hand back a full COUPLED/DECOUPLED/FLAT-COUPLED verdict
    with S2 silently skipped. The pin demands the rule be "structurally true, not just a
    convention someone remembers to apply." `factor1_trend_s2` is now a REQUIRED keyword-only
    argument OF `compute_verdict` itself; the rule is enforced there; a stale S2-less call
    raises TypeError. Guarded by smoke [9b] (negative test + the full 4x4 grid, both entry
    points).
  * `register_normalization` would SILENTLY REBIND the pinned "raw_did" to any other function.
    It now refuses (override=True required); re-registering the same object stays a no-op.
  * `--normalization` was parsed and then never read -- silently ignored on every --run. Now
    refused out loud (mode_run cannot apply a normalization at all, by construction).
  * DOC: sec 9.1.5's "S1 is bounded to [0,1]" is not exactly true (T2b-2 admits
    DiD <= acc_copy + 2*SE, so S1 can reach 1 + 2*SE/acc_copy > 1 on a PASSING rung; and DiD<0
    makes it negative). Corrected in `compute_s1_utilization`'s docstring. Zero verdict
    consequence -- S1 is structurally verdict-withholding -- so it is a note, not a defect.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # pod-safe imports
sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

from h2h_fla_stub_rd import ensure_fla_stub  # noqa: E402
ensure_fla_stub()   # real fla on the H100 box; CPU stub everywhere else / when forced

from lm_pretrain_rd import (   # noqa: E402
    DeltaNetLM, load_corpus, get_batch, CORPUS_DIRS, EOT_TOKEN_ID,
    DEFAULT_DATA_DIR, corpus_fixed_seed,
)

VOCAB_SIZE = 50257  # GPT-2, asserted by load_corpus's own meta.json check

# sec 9.6 item 1: the common token slice is FORCED at 0.328B (14M's own final
# checkpoint step*batch*seq_len) -- the design's original 1.0B-slice dual
# cross-validation is RETRACTED as unrunnable (regate S-6: those checkpoints
# do not exist on disk). Do not "improve" this back to 1.0B.
TARGET_TOKENS_FORCED_SLICE = 327_680_000

# sec 9.2 "Cost" pinned sampling constants.
N_ROWS_DEFAULT = 512
C_MAX_DEFAULT = 8            # per-ROW, fixed, rung-invariant (kills F-4)
EVAL_MICRO_BATCH_DEFAULT = 64  # decoupled from token-arithmetic batch_size (kills F-4 pt. 4)
PLACEBO_FLAGGED_FRAC_VOID_THRESHOLD = 0.05   # sec 9.2
TOK_PER_PARAM_PRIMARY_FIT_FLOOR = 1.0        # sec 9.6 item 2


# =============================================================================
# 0. Deterministic seed combination (NEVER Python's hash() -- salted per
#    process; this file follows lm_pretrain_rd.py's own corpus_fixed_seed
#    rationale for exactly the same reason: reproducibility across runs).
# =============================================================================
def _combine_seed(*parts) -> int:
    s = ":".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) & 0x7FFFFFFF


# =============================================================================
# 1. FIX-A -- common-token-count checkpoint slice resolver.
#    REUSED VERBATIM from lm_recall_gap_probe_rd.py (regate: never named
#    defective). Do not rewrite; sec 9.6 item 1 depends on this exact logic.
# =============================================================================
_STEP_RE = re.compile(r"_step(\d+)\.pt$")


def list_checkpoint_steps(ckpt_dir: str) -> list:
    steps = []
    for f in glob.glob(os.path.join(ckpt_dir, "*.pt")):
        m = _STEP_RE.search(f)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def resolve_token_matched_checkpoint(ckpt_dir: str, batch_size: int, seq_len: int,
                                      target_tokens: int, prefix: str) -> dict:
    """Picks the on-disk step whose token count (step*batch_size*seq_len) is
    CLOSEST to target_tokens (ties broken toward the lower step, i.e. never
    overshoot on a tie). Returns a dict recording the chosen step, its exact
    token count, the signed miss, and the full path -- never silently picks
    "final checkpoint"."""
    steps = list_checkpoint_steps(ckpt_dir)
    if not steps:
        return {"ckpt_dir": ckpt_dir, "error": "NO_CHECKPOINTS_FOUND"}
    best = None
    for s in steps:
        tok = s * batch_size * seq_len
        miss = tok - target_tokens
        key = (abs(miss), miss > 0)  # prefer closer, then prefer under-shoot on ties
        if best is None or key < best[0]:
            best = (key, s, tok, miss)
    _, step, tok, miss = best
    path = os.path.join(ckpt_dir, f"{prefix}_step{step}.pt")
    assert os.path.isfile(path), f"resolved step {step} but file missing: {path}"
    return {
        "ckpt_dir": ckpt_dir, "batch_size": batch_size, "seq_len": seq_len,
        "target_tokens": target_tokens, "chosen_step": step,
        "chosen_tokens": tok, "miss_tokens": miss,
        "miss_frac": miss / target_tokens if target_tokens else None,
        "n_steps_available": len(steps), "min_step": steps[0], "max_step": steps[-1],
        "path": path,
    }


# =============================================================================
# 2. Checkpoint quiescence + md5 pinning (sec 9.6 item 3, kills S-7).
# =============================================================================
def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_quiesced_checkpoint(path: str, wait_s: float = 5.0) -> dict:
    """sec 9.6 item 3, FIRST HALF ONLY: detects a checkpoint FILE that is
    actively being written. Compares (size, mtime, md5) before/after a wait
    window; a writer mid --ckpt-every-N write changes at least one. `md5` is
    populated ONLY when quiesced=True -- it is the value sec 9.6 requires to
    be "md5-pinned in the result JSON," and pinning an md5 taken WHILE a file
    is changing would be meaningless.

    *** THIS CHECK IS NECESSARY BUT NOT SUFFICIENT, AND ON ITS OWN IT WOULD NOT HAVE
    CAUGHT S-7. *** sec 9.6 item 3 has two clauses: the file must be quiesced AND "its job
    must be terminated." File-stability proves only the former. The S-7 precedent was a
    read of a LIVE 1.31B job's step-40000 checkpoint under a `--ckpt-every 10000` writer
    -- at ~1.4 s/step those writes are ~3.9 HOURS apart, so that file was perfectly
    byte-stable across any 5-second window while the job was still training. A live job
    passes this check for hours at a time, by construction. The "job terminated" clause is
    therefore enforced separately, as an explicit operator attestation recorded in the
    result JSON (`--attest-job-terminated`, stamped as `job_terminated_attested`), because
    nothing this process can measure about the FILE can establish it.
    (AUDIT NOTE 2026-07-12: an atomic-rename writer -- torch.save to a tmp path then
    os.replace -- is actually SAFE here: the visible file is always a complete older or
    newer checkpoint, and a mid-window swap changes size/mtime/md5 and is caught. The
    residual false-PASS is a writer that CRASHED mid-write, leaving a stable truncated
    file; that is not silently accepted -- torch.load raises on it.)"""
    st0 = os.stat(path)
    md5_0 = md5_file(path)
    time.sleep(wait_s)
    st1 = os.stat(path)
    md5_1 = md5_file(path)
    quiesced = (st0.st_size == st1.st_size) and (st0.st_mtime == st1.st_mtime) and (md5_0 == md5_1)
    return {
        "path": path, "quiesced": quiesced, "md5": md5_1 if quiesced else None,
        "size0": st0.st_size, "size1": st1.st_size,
        "mtime0": st0.st_mtime, "mtime1": st1.st_mtime,
        "md5_0": md5_0, "md5_1": md5_1, "wait_s": wait_s,
    }


class CheckpointNotQuiescedError(RuntimeError):
    pass


def load_checkpoint(path: str, device: str, require_quiesced: bool = True,
                     wait_s: float = 5.0) -> tuple:
    """sec 9.6 item 3, non-negotiable for any verdict-grade read: refuses to
    load (raises CheckpointNotQuiescedError) a checkpoint that fails the
    quiescence check.

    AUDIT CORRECTION (2026-07-12): the previous docstring claimed
    `require_quiesced=False` "exists ONLY for the smoke gate's own synthetic fixtures ...
    every real-run code path in this module leaves it at the default True." **That was
    false.** The smoke gate never calls load_checkpoint at all, and `--no-quiesce-check`
    wires straight into the `--run` path -- i.e. the ONLY caller of the escape hatch is
    the real, verdict-grade run. It is retained (a legitimately-terminated job whose
    filesystem reports jittery mtimes is a real scenario) but `mode_run` now STAMPS
    `quiesce_check_skipped: true` into the result JSON, so a skipped guard can never be
    invisible to a downstream reader, and the md5 pin sec 9.6 requires is `None` in that
    case -- which by itself makes the cell non-conformant."""
    md5_pin = None
    if require_quiesced:
        q = verify_quiesced_checkpoint(path, wait_s=wait_s)
        if not q["quiesced"]:
            raise CheckpointNotQuiescedError(
                f"CHECKPOINT NOT QUIESCED: {path} changed during a {wait_s}s stability "
                f"window (size {q['size0']}->{q['size1']}, md5 {q['md5_0']}->{q['md5_1']}). "
                f"A rung may NOT be read from a live, still-training job "
                f"(PARAM_AXIS_SCALING_DESIGN.md sec 9.6 item 3; regate_2026-07-12.md sec 10.3 "
                f"S-7 precedent: a prior run raced a live --ckpt-every writer). Retry once the "
                f"writer has terminated."
            )
        md5_pin = q["md5"]
    ckpt = torch.load(path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, md5_pin


# =============================================================================
# 3. Modal-continuation table -- REUSED VERBATIM (sound per sec 9.0).
# =============================================================================
def build_bigram_mode_table(train_tokens: torch.Tensor, vocab_size: int, device: str) -> torch.Tensor:
    """mode_next[a] = the most frequent b immediately following a in the
    TRAIN split (ties broken toward smaller b, deterministic); -1 if a
    never occurred followed by anything."""
    t = train_tokens.to(device)
    a = t[:-1].to(torch.int64)
    b = t[1:].to(torch.int64)
    pair_ids = a * vocab_size + b
    uniq, counts = torch.unique(pair_ids, return_counts=True)
    a_idx = uniq // vocab_size
    b_idx = uniq % vocab_size
    max_count_per_a = torch.full((vocab_size,), -1.0, device=device)
    max_count_per_a.scatter_reduce_(0, a_idx, counts.to(torch.float64).to(torch.float32),
                                     reduce="amax", include_self=True)
    is_max = counts.to(torch.float32) >= max_count_per_a[a_idx]
    b_where_max = torch.where(is_max, b_idx, torch.full_like(b_idx, vocab_size))
    mode_next = torch.full((vocab_size,), vocab_size, dtype=torch.int64, device=device)
    mode_next.scatter_reduce_(0, a_idx, b_where_max, reduce="amin", include_self=True)
    mode_next[mode_next == vocab_size] = -1
    return mode_next.cpu()


def _make_window(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, y[:, -1:]], dim=1)


# =============================================================================
# 4. Candidate + baseline detection (sec 9.0's construction, retained;
#    UNCAPPED here -- capping is a separate, rung-independent step, sec 9.2
#    "kills F-4").
# =============================================================================
def detect_candidates_and_baseline(x_cpu: list, y_cpu: list, mode_next_cpu: list,
                                    min_sep: int) -> tuple:
    """Per-row scan implementing sec 9.0's candidate construction; returns candidates
    UNCAPPED, one list per row -- capping happens in `select_candidates_per_row`,
    decoupled from batch size entirely.

    AUDIT CORRECTION (2026-07-12): this is NOT, as the previous docstring claimed, an
    exact mirror of the VOID build's detection loop. There is one real behavioural
    difference, and it is deliberate: the old loop did `seen[key] = k` when a repeat fell
    WITHIN min_sep, advancing the antecedent pointer to the later occurrence; this one
    never advances it, so `j` is always the FIRST occurrence. sec 9.0 is explicit -- "the
    antecedent is the single token at position j+1 -- the continuation token of the FIRST
    occurrence" -- so the new behaviour is the spec-faithful one and the old was subtly
    off. It affects only bigrams that repeat within min_sep tokens, applies identically at
    every rung (so it cannot bias a cross-rung comparison), and yields slightly MORE
    candidates. Recorded rather than silently carried.

    Returns (row_candidates, row_baseline) where
    row_candidates[b] = [(k, j), ...] and row_baseline[b] = [k, ...]
    (ordinary first-occurrence positions, subsampled every 7th, for the
    non-AR baseline slice)."""
    B = len(x_cpu)
    row_candidates = []
    row_baseline = []
    for b in range(B):
        xb, yb = x_cpu[b], y_cpu[b]
        T = len(xb)
        seen = {}
        cands = []
        baseline = []
        for k in range(T):
            a, bb = xb[k], yb[k]
            if a == EOT_TOKEN_ID or bb == EOT_TOKEN_ID:
                continue
            key = (a, bb)
            if key in seen:
                j = seen[key]
                if k - j > min_sep:
                    modal = mode_next_cpu[a]
                    if modal != bb:
                        cands.append((k, j))
                continue
            seen[key] = k
            if k % 7 == 0:
                baseline.append(k)
        row_candidates.append(cands)
        row_baseline.append(baseline)
    return row_candidates, row_baseline


def select_candidates_per_row(row_candidates: list, c_max: int, seed: int) -> list:
    """sec 9.2 Cost: C_max candidates per row, uniform random WITHIN the row,
    seeded ONLY by (seed, row index) -- never by rung/params/batch_size. Since
    every rung samples the identical window sequence (corpus_fixed_seed-based
    generator, same corpus), row_candidates[i] is IDENTICAL across rungs
    before this call; this function then subsamples the SAME candidates at
    every rung. This is the direct fix for F-4 (the old per-batch cap that
    silently made one rung's population differ from the other three's)."""
    out = []
    for i, cands in enumerate(row_candidates):
        if len(cands) <= c_max:
            out.append(list(cands))
            continue
        rng = random.Random(_combine_seed(seed, "select", i))
        idxs = sorted(rng.sample(range(len(cands)), c_max))
        out.append([cands[j] for j in idxs])
    return out


# =============================================================================
# 5. The TRUE-arm candidate spec + PLACEBO position assignment (sec 9.2).
# =============================================================================
def true_arm_specs(sel_row_candidates: list) -> list:
    """Flattens per-row selected candidates into a flat, stable-ordered list
    of dicts: {row_idx, k, j, p(=j+1), delta(=k-p)} -- the TRUE arm's spec."""
    specs = []
    for row_idx, cands in enumerate(sel_row_candidates):
        for (k, j) in cands:
            p = j + 1
            specs.append({"row_idx": row_idx, "k": k, "j": j, "p": p, "delta": k - p})
    return specs


def assign_placebo_positions(specs: list, sel_row_candidates: list, x_cpu: list,
                              seed: int, max_redraws: int = 100) -> list:
    """sec 9.2: for each candidate, draw a placebo position p' at a distance
    Delta' resampled (with replacement, fixed seed) from the POOLED empirical
    distribution of Delta over `specs` -- per-candidate EXACT distance
    matching is impossible by construction (same distance => same position),
    so the match is distributional, exactly as pinned. Admissibility: p' is
    rejected/redrawn if it lands on p, on j (the antecedent's key token), on
    k or later, on an EOT, or on the antecedent position of ANY OTHER
    candidate in the same row. Cap 100 redraws; fall back to a uniform draw
    over admissible positions and set `placebo_flagged=True`. Mutates and
    returns `specs` with `p_placebo`/`delta_placebo`/`placebo_flagged` added
    (p_placebo=None iff even the fallback found no admissible position)."""
    deltas_pool = [s["delta"] for s in specs]
    if not deltas_pool:
        return specs
    resample_rng = random.Random(_combine_seed(seed, "placebo_delta_pool"))
    # per-row forbidden sets: every OTHER candidate's antecedent position (j+1) in that row
    forbidden_by_row = {}
    for row_idx, cands in enumerate(sel_row_candidates):
        forbidden_by_row[row_idx] = {j + 1 for (_, j) in cands}

    for idx, s in enumerate(specs):
        row_idx, k, p, j = s["row_idx"], s["k"], s["p"], s["j"]
        xb = x_cpu[row_idx]
        forbidden = set(forbidden_by_row[row_idx])
        forbidden.add(p)
        forbidden.add(j)
        p_placebo = None
        delta_placebo = None
        flagged = False
        for _attempt in range(max_redraws):
            delta_p = resample_rng.choice(deltas_pool)
            cand_p = k - delta_p
            if cand_p < 0 or cand_p >= k:
                continue
            if cand_p in forbidden:
                continue
            if xb[cand_p] == EOT_TOKEN_ID:
                continue
            p_placebo, delta_placebo = cand_p, delta_p
            break
        if p_placebo is None:
            admissible = [pos for pos in range(0, k)
                          if pos not in forbidden and xb[pos] != EOT_TOKEN_ID]
            flagged = True
            if admissible:
                fallback_rng = random.Random(_combine_seed(seed, "placebo_fallback", idx))
                p_placebo = fallback_rng.choice(admissible)
                delta_placebo = k - p_placebo
        specs[idx] = {**s, "p_placebo": p_placebo, "delta_placebo": delta_placebo,
                      "placebo_flagged": flagged}
    return specs


# =============================================================================
# 6. Row-replication ablation -- ONE corrupted token per forward-pass ROW.
#    This is the FATAL-1 fix. `build_replicated_ablation_batch` is the unit
#    the smoke gate's "one-token-diff" test targets directly.
# =============================================================================
def draw_exclusive_replacement(window_row: list, pos: int, target: int, vocab_size: int,
                                rng: random.Random) -> int:
    """Uniform draw excluding {the token currently at `pos`, `target`, EOT} --
    same rule for TRUE and PLACEBO arms, and both draw from the SAME `rng`
    stream (continuing it, not reseeding) per sec 9.2 ('same replacement
    RNG')."""
    true_next = window_row[pos]
    while True:
        repl = rng.randrange(vocab_size)
        if repl != true_next and repl != target and repl != EOT_TOKEN_ID:
            return repl


def build_replicated_ablation_batch(window_row: torch.Tensor, corrupt_positions: list,
                                     replacement_tokens: list) -> torch.Tensor:
    """THE FATAL-1 FIX, as its own unit: given ONE window row (length T+1,
    the `_make_window` convention) and M (position, token) corruptions,
    returns an (M, T+1) tensor where row m is an EXACT COPY of window_row
    except position corrupt_positions[m] is replaced by replacement_tokens[m].
    Every OUTPUT ROW THEREFORE CARRIES EXACTLY ONE CORRUPTED TOKEN relative to
    window_row -- never a shared tensor with every candidate's antecedent
    corrupted simultaneously (regate FATAL-1). `smoke()`'s unit test asserts
    this directly on the returned tensor's Hamming distance to the base row."""
    M = len(corrupt_positions)
    assert M == len(replacement_tokens)
    rep = window_row.unsqueeze(0).repeat(M, 1).clone()
    if M:
        row_idx_t = torch.arange(M, device=window_row.device)
        pos_t = torch.tensor(corrupt_positions, device=window_row.device, dtype=torch.long)
        tok_t = torch.tensor(replacement_tokens, device=window_row.device, dtype=torch.long)
        rep[row_idx_t, pos_t] = tok_t
    return rep


def run_ablation_arm(model, all_windows: list, specs: list, pos_key: str, repl_key: str,
                      pred_key: str, eval_micro_batch: int, device: str,
                      logp_key: str = None) -> None:
    """Builds ONE replicated row PER CANDIDATE (batching over candidates, not
    rows, per sec 9.2's exact prescription -- a candidate's replicated row is
    a full clone of its source window with exactly its own single corruption
    applied; different candidates from the SAME source row each get their own
    independent clone, so no two candidates' corruptions ever land in the
    same forward-pass row). Runs the whole candidate population's replicated
    rows through the model in `eval_micro_batch`-sized chunks (fully
    decoupled from the token-arithmetic batch_size -- kills F-4 pt 4) and
    writes each spec's predicted token (argmax at its own query position k)
    into `spec[pred_key]`. Candidates whose `pos_key` is None (an unresolved
    placebo fallback) are skipped entirely -- never silently defaulted.

    sec 9.1.5 S2 BUILD REQUIREMENT: if `logp_key` is given, also writes
    `spec[logp_key] = log p(spec['target'])` under this arm -- a
    `log_softmax` + `gather` on the SAME logits this function already
    computes for the argmax read, zero extra forward passes. Requires
    `spec['target']` to already be set (true for every caller in this
    module)."""
    big_rows = []
    owner = []
    for i, s in enumerate(specs):
        pos = s.get(pos_key)
        if pos is None:
            continue
        base = all_windows[s["row_idx"]]
        rep = base.clone()
        rep[pos] = s[repl_key]
        big_rows.append(rep)
        owner.append(i)
    if not big_rows:
        return
    big = torch.stack(big_rows, dim=0)

    # AUDIT FIX (2026-07-12): ENFORCE THE FATAL-1 INVARIANT ON THE PRODUCTION PATH.
    # The smoke gate asserted "exactly one corrupted token per row" against
    # `build_replicated_ablation_batch` -- a function this, the real code path, never
    # calls. That is an F-3-class toothless test (a gate that passes without exercising
    # the thing it certifies). This assertion checks the invariant on the ACTUAL tensor
    # about to be fed to the model. It is O(M*T) elementwise on an (<=4096, 513) tensor
    # -- negligible next to the forward passes -- and it is the single line whose failure
    # means the instrument has regressed to the VOID build's defect.
    base_stack = torch.stack([all_windows[specs[i]["row_idx"]] for i in owner], dim=0)
    n_corrupted = (big != base_stack).sum(dim=1)
    if not bool((n_corrupted == 1).all()):
        bad = (n_corrupted != 1).nonzero().flatten().tolist()[:8]
        raise AssertionError(
            f"FATAL-1 INVARIANT VIOLATED in run_ablation_arm({pos_key!r}): "
            f"{int((n_corrupted != 1).sum())} of {big.shape[0]} forward-pass rows carry "
            f"!= 1 corrupted token (offending row idxs {bad}, counts "
            f"{n_corrupted[bad].tolist() if bad else []}). Every row MUST differ from its "
            f"source window at EXACTLY the one position belonging to its own candidate "
            f"(PARAM_AXIS_SCALING_DESIGN.md sec 9.2: 'Every forward pass carries exactly "
            f"one corrupted token. This is non-negotiable.'; regate_2026-07-12.md sec 10.2 "
            f"FATAL-1 is the realized failure of exactly this invariant)."
        )

    preds_chunks = []
    for start in range(0, big.shape[0], eval_micro_batch):
        chunk = big[start:start + eval_micro_batch, :-1]
        with torch.no_grad():
            logits = model(chunk)
        preds_chunks.append(logits.argmax(dim=-1))
        if logp_key is not None:
            # sec 9.1.5 S2: log p(target) at each row's OWN query position k, off the SAME logits
            # already computed for the argmax read -- zero extra forward passes.
            #
            # AUDIT FIX (2026-07-12, S2 follow-up audit): GATHER THE QUERY POSITIONS FIRST, then
            # log_softmax the (n_chunk, V) slice. The first S2 build ran `logits.log_softmax(dim=-1)`
            # on the FULL (n_chunk, T, V) tensor and then read ONE element per row -- discarding
            # 511/512 of it unread, and DOUBLING the peak VRAM of the largest allocation in the
            # whole pipeline. MEASURED ON THE H100 at the pinned eval_micro_batch=64, T=512,
            # V=50257, fp32: logits alone 6.14 GiB -> 12.27 GiB peak with the full log_softmax,
            # vs 6.16 GiB with this gather-first form (+0.02 GiB). Softmax is independent per
            # position, so the arithmetic is IDENTICAL (measured max |diff| 9.5e-7 -- fp32
            # reduction-order noise on a quantity of order 1-10 nats). This repo's own rule:
            # "The 50K vocab logits tensor is the VRAM bottleneck, not the model activations."
            n_chunk = chunk.shape[0]
            rows_t = torch.arange(n_chunk, device=logits.device)
            ks_t = torch.tensor([specs[owner[start + i]]["k"] for i in range(n_chunk)],
                                 device=logits.device, dtype=torch.long)
            tgt_t = torch.tensor([specs[owner[start + i]]["target"] for i in range(n_chunk)],
                                  device=logits.device, dtype=torch.long)
            lp = logits[rows_t, ks_t, :].float().log_softmax(dim=-1)[rows_t, tgt_t]   # (n_chunk,)
            lp_list = lp.tolist()
            for i in range(n_chunk):
                specs[owner[start + i]][logp_key] = float(lp_list[i])
            del lp
    preds = torch.cat(preds_chunks, dim=0)
    for local_i, spec_i in enumerate(owner):
        k = specs[spec_i]["k"]
        specs[spec_i][pred_key] = int(preds[local_i, k].item())


# =============================================================================
# 7. Top-level per-cell DiD evaluation.
# =============================================================================
def run_did_eval(model, val_tokens: torch.Tensor, batch_size: int, seq_len: int,
                  n_windows: int, device: str, mode_next: torch.Tensor, seed: int,
                  c_max: int = C_MAX_DEFAULT, eval_micro_batch: int = EVAL_MICRO_BATCH_DEFAULT,
                  min_sep: int = 2, vocab_size: int = VOCAB_SIZE) -> dict:
    """Samples n_windows rows, runs ONE shared intact pass (permitted -- it
    is unmodified, sec 9.2), detects candidates+baseline (uncapped), caps per
    row (rung-independent), assigns placebo positions (Delta-matched), builds
    the TRUE and PLACEBO row-replicated batches, and returns raw per-candidate
    records (never an aggregate -- aggregation with the placebo-refusal rule
    lives in `finalize_cell`)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    mode_next_cpu = mode_next.tolist()

    # AUDIT FIX (2026-07-12, independent adversarial audit): WINDOW SAMPLING MUST BE
    # BATCH-SIZE-INDEPENDENT. The first rebuild drew the windows in `batch_size`-sized
    # get_batch() chunks. On CPU (MT19937) that happens to be batch-size-invariant, but
    # the real run is on CUDA, where get_batch's torch.randint uses the PHILOX generator
    # whose offset advances PER KERNEL LAUNCH -- so 32 draws of 16 does NOT reproduce
    # 16 draws of 32. MEASURED ON THE H100 AT THE SAME SEED: batch_size=16 and
    # batch_size=32 share only 50% of their 512 windows. The design's rungs use DIFFERENT
    # token-arithmetic batch sizes (three rungs at 32, the 1.31B at 16), and `--batch-size`
    # feeds straight in here -- so chunked sampling would hand the top rung a DIFFERENT
    # candidate population than the others. That is F-4 (regate sec 10.3) reborn in a new
    # guise, and it is exactly what sec 9.2's "the eval batch size is DECOUPLED from the
    # token-arithmetic batch size and from candidate selection entirely" forbids.
    # Drawing all n_windows offsets in ONE call makes the window population a pure
    # function of (seed, corpus, seq_len, n_windows). `batch_size` is retained as a
    # RECORDED PROVENANCE FIELD ONLY and no longer influences any measurement.
    x0, y0 = get_batch(val_tokens, n_windows, seq_len, gen)
    window = _make_window(x0, y0)
    all_windows = [window[b] for b in range(window.shape[0])]
    N_rows = len(all_windows)

    intact_batch = torch.stack(all_windows, dim=0)
    x_intact = intact_batch[:, :-1]
    y_intact = intact_batch[:, 1:]
    x_cpu = x_intact.cpu().tolist()
    y_cpu = y_intact.cpu().tolist()
    window_cpu = intact_batch.cpu().tolist()

    # Candidate detection happens BEFORE the intact forward pass (it is pure token-content --
    # y_cpu already carries the true next-token target, so no model output is needed) --
    # this is what lets the intact pass extract sec 9.1.5's S2 log-probs inline, per chunk,
    # without ever materializing the full (N_rows,T,V) log-prob tensor (>50GB at N_rows=512,
    # T=512, V=50257 -- infeasible; a per-chunk tensor of the SAME size as the logits already
    # computed is not).
    row_candidates, row_baseline = detect_candidates_and_baseline(x_cpu, y_cpu, mode_next_cpu, min_sep)
    sel_candidates = select_candidates_per_row(row_candidates, c_max, seed)
    specs = true_arm_specs(sel_candidates)
    specs = assign_placebo_positions(specs, sel_candidates, x_cpu, seed)

    ablate_rng = random.Random(_combine_seed(seed, "ablate_replacement"))   # ONE stream, TRUE then PLACEBO
    for s in specs:
        target = window_cpu[s["row_idx"]][s["k"] + 1]
        s["target"] = target
        s["repl_true"] = draw_exclusive_replacement(window_cpu[s["row_idx"]], s["p"], target,
                                                      vocab_size, ablate_rng)
    for s in specs:
        if s["p_placebo"] is not None:
            target = s["target"]
            s["repl_placebo"] = draw_exclusive_replacement(window_cpu[s["row_idx"]], s["p_placebo"],
                                                             target, vocab_size, ablate_rng)
        else:
            s["repl_placebo"] = None

    cand_by_row = {}
    for i, s in enumerate(specs):
        cand_by_row.setdefault(s["row_idx"], []).append(i)

    pred_chunks = []
    for start in range(0, N_rows, eval_micro_batch):
        chunk = x_intact[start:start + eval_micro_batch]
        with torch.no_grad():
            logits = model(chunk)
        pred_chunks.append(logits.argmax(dim=-1))
        # sec 9.1.5 S2 BUILD REQUIREMENT: log p(target) under the INTACT context, from the
        # SAME logits just computed for the argmax read -- zero extra forward passes.
        # Gathered at the candidate query positions ONLY, never as a full-tensor log_softmax
        # (see run_ablation_arm's note: the full form doubles peak VRAM -- measured +6.13 GiB
        # per chunk on the H100 at the pinned eval_micro_batch=64 -- to read at most c_max=8
        # of each row's 512 positions). AUDIT FIX (2026-07-12, S2 follow-up audit).
        # `row_idx = start + local_i` is the chunk-local -> global row map; cand_by_row was
        # built above from the FULL, final `specs` population (post-cap, post-placebo).
        pairs = [(local_i, spec_i)
                 for local_i in range(chunk.shape[0])
                 for spec_i in cand_by_row.get(start + local_i, [])]
        if pairs:
            li_t = torch.tensor([p[0] for p in pairs], device=logits.device, dtype=torch.long)
            ks_t = torch.tensor([specs[p[1]]["k"] for p in pairs], device=logits.device, dtype=torch.long)
            tgt_t = torch.tensor([specs[p[1]]["target"] for p in pairs], device=logits.device,
                                  dtype=torch.long)
            idx_t = torch.arange(len(pairs), device=logits.device)
            lp = logits[li_t, ks_t, :].float().log_softmax(dim=-1)[idx_t, tgt_t]   # (n_pairs,)
            lp_list = lp.tolist()
            for n_i, (_, spec_i) in enumerate(pairs):
                specs[spec_i]["logp_intact"] = float(lp_list[n_i])
            del lp
    pred_intact = torch.cat(pred_chunks, dim=0)  # (N_rows, T)

    run_ablation_arm(model, all_windows, specs, "p", "repl_true", "pred_true", eval_micro_batch, device,
                      logp_key="logp_true")
    run_ablation_arm(model, all_windows, specs, "p_placebo", "repl_placebo", "pred_placebo",
                      eval_micro_batch, device, logp_key="logp_placebo")

    records = []
    for s in specs:
        row_idx, k = s["row_idx"], s["k"]
        target = s["target"]
        hit_intact = int(pred_intact[row_idx, k].item() == target)
        hit_true = int(s.get("pred_true") == target) if "pred_true" in s else None
        hit_placebo = int(s.get("pred_placebo") == target) if s.get("pred_placebo") is not None else None
        records.append({
            "row_idx": row_idx, "k": k, "delta": s["delta"],
            # AUDIT FIX (2026-07-12): carry delta_placebo so the pinned "distribution-
            # matched" property can be VERIFIED downstream instead of merely asserted.
            "delta_placebo": s.get("delta_placebo"),
            "hit_intact": hit_intact, "hit_true_ablated": hit_true,
            "hit_placebo_ablated": hit_placebo, "placebo_flagged": s["placebo_flagged"],
            # sec 9.1.5 S2 (log-prob sensitivity -- guards the argmax-floor bias toward
            # RISES/DECOUPLED). None iff the corresponding arm never ran (unresolved placebo).
            "logp_intact": s.get("logp_intact"),
            "logp_true_ablated": s.get("logp_true"),
            "logp_placebo_ablated": s.get("logp_placebo"),
        })

    baseline_hits, baseline_n = 0, 0
    for row_idx, ks in enumerate(row_baseline):
        for k in ks:
            baseline_hits += int(pred_intact[row_idx, k].item() == y_cpu[row_idx][k])
            baseline_n += 1
    acc_baseline_nonAR = baseline_hits / baseline_n if baseline_n else None

    return {
        "records": records, "n_windows": N_rows, "batch_size": batch_size, "seq_len": seq_len,
        "c_max": c_max, "acc_baseline_nonAR": acc_baseline_nonAR, "n_baseline": baseline_n,
    }


# =============================================================================
# 8. Clustered (row) bootstrap -- T1a/T1b's CI machinery.
# =============================================================================
def clustered_bootstrap_ci(records: list, stat_fn, n_boot: int = 2000, seed: int = 0,
                            alpha: float = 0.05) -> tuple:
    """Resamples ROWS (clusters) with replacement -- candidates within a row
    share a context and are not independent (sec 9.3 T1a). `stat_fn(records)
    -> float`. Returns (point, ci_lo, ci_hi)."""
    if not records:
        return None, None, None
    rows = sorted(set(r["row_idx"] for r in records))
    by_row = {ri: [] for ri in rows}
    for r in records:
        by_row[r["row_idx"]].append(r)
    point = stat_fn(records)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        subset = []
        for _ in range(len(rows)):
            subset.extend(by_row[rng.choice(rows)])
        if subset:
            boots.append(stat_fn(subset))
    if not boots:
        return point, None, None
    boots.sort()
    lo = boots[max(0, int(alpha / 2 * len(boots)))]
    hi = boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)) - 1)]
    return point, lo, hi


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stat_gap_true(records: list) -> float:
    return _mean([r["hit_intact"] for r in records]) - _mean([r["hit_true_ablated"] for r in records])


def stat_gap_placebo(records: list) -> float:
    return _mean([r["hit_intact"] for r in records]) - _mean([r["hit_placebo_ablated"] for r in records])


def stat_did(records: list) -> float:
    # DiD(r) = gap_true - gap_placebo = acc_placebo_ablated - acc_true_ablated
    # (acc_intact cancels: both arms are compared to the SAME candidate
    # population's SAME intact reads -- a paired design, sec 9.2's formula).
    return _mean([r["hit_placebo_ablated"] for r in records]) - _mean([r["hit_true_ablated"] for r in records])


def stat_did_logp(records: list) -> float:
    """sec 9.1.5 S2: DiD_logp = E[logp_placebo_ablated - logp_true_ablated] -- the identical
    estimand as `stat_did` (intact cancels the same way) under a continuous, floor-free
    readout instead of a hard argmax threshold. Same clustered-bootstrap machinery applies
    unchanged (it is generic over any per-record scalar)."""
    return (_mean([r["logp_placebo_ablated"] for r in records])
            - _mean([r["logp_true_ablated"] for r in records]))


def has_complete_s2_fields(records: list) -> bool:
    """sec 9.1.5 BUILD REQUIREMENT / sec 9.6 stop rule: 'a cell missing these fields is
    VOID.' True iff every record carries a non-None logp in all three arms."""
    return all(r.get("logp_intact") is not None and r.get("logp_true_ablated") is not None
               and r.get("logp_placebo_ablated") is not None for r in records)


# =============================================================================
# 9. finalize_cell -- THE VERDICT-GRADE REFUSAL POINT.
#    "No un-placebo'd gap, and no raw AR-hit slice, may carry a verdict."
# =============================================================================
class VerdictGradeError(RuntimeError):
    pass


def finalize_cell(rung: str, corpus: str, did_eval_result: dict,
                   flagged_frac_threshold: float = PLACEBO_FLAGGED_FRAC_VOID_THRESHOLD,
                   n_boot: int = 2000, boot_seed: int = 0, t2b_result: dict = None) -> dict:
    """Aggregates run_did_eval's raw per-candidate records into a
    verdict-eligible cell summary. THIS is the enforcement point: any record
    lacking a resolved placebo read is EXCLUDED from every statistic below,
    and if NO record has a resolved placebo, this raises VerdictGradeError
    rather than silently falling back to the un-placebo'd gap. The returned
    dict is stamped `_verdict_grade=True` -- `compute_capacity_metric` below
    refuses any input dict lacking that stamp, closing the loophole of a
    caller hand-building a look-alike dict.

    `t2b_result`: optional `run_t2_planted_copy(...)` output for this SAME
    (rung, corpus) -- if given, its `acc_copy`/`acc_copy_se` are merged into
    the returned cell so `acc_copy` is in the schema alongside `did` (sec
    9.1.5 S1's own inputs, and the coordinator's schema requirement)."""
    all_records = did_eval_result["records"]
    resolved = [r for r in all_records if r["hit_placebo_ablated"] is not None]
    if not resolved:
        raise VerdictGradeError(
            f"{rung}/{corpus}: zero candidates have a resolved placebo read out of "
            f"{len(all_records)} total -- cannot compute a verdict-grade DiD. Refusing to "
            f"fall back to the un-placebo'd gap (PARAM_AXIS_SCALING_DESIGN.md sec 9.2: 'no "
            f"un-placebo'd gap, and no raw AR-hit slice, may carry a verdict')."
        )
    flagged_frac = placebo_flagged_fraction_from_records(all_records)
    cell_void_placebo_match = flagged_frac > flagged_frac_threshold

    did_pt, did_lo, did_hi = clustered_bootstrap_ci(resolved, stat_did, n_boot=n_boot, seed=boot_seed)
    gt_pt, gt_lo, gt_hi = clustered_bootstrap_ci(resolved, stat_gap_true, n_boot=n_boot, seed=boot_seed + 1)
    gp_pt, gp_lo, gp_hi = clustered_bootstrap_ci(resolved, stat_gap_placebo, n_boot=n_boot, seed=boot_seed + 2)

    t1a_pass = (did_lo is not None) and (did_lo > 0)  # DiD CI excludes 0, and positive

    # sec 9.1.5 S2 (BUILD REQUIREMENT, non-optional): a cell missing the log-prob fields on
    # any resolved candidate is VOID (sec 9.6's stop rule -- S2 "cannot be added after a
    # read"). When complete, DiD_logp gets the SAME clustered-row bootstrap as the primary.
    s2_complete = has_complete_s2_fields(resolved)
    if s2_complete:
        didlogp_pt, didlogp_lo, didlogp_hi = clustered_bootstrap_ci(
            resolved, stat_did_logp, n_boot=n_boot, seed=boot_seed + 3)
        per_row_did_logp = _per_row_stat(resolved, stat_did_logp)
    else:
        didlogp_pt = didlogp_lo = didlogp_hi = None
        per_row_did_logp = {}

    cell = {
        "_verdict_grade": True,
        "rung": rung, "corpus": corpus,
        "n_candidates_total": len(all_records), "n_candidates_resolved": len(resolved),
        "placebo_flagged_frac": flagged_frac,
        "cell_void_placebo_match": cell_void_placebo_match,
        "did": did_pt, "did_ci": [did_lo, did_hi],
        "gap_true": gt_pt, "gap_true_ci": [gt_lo, gt_hi],
        "gap_placebo": gp_pt, "gap_placebo_ci": [gp_lo, gp_hi],
        "t1a_pass_did_ci_excludes_zero": t1a_pass,
        "acc_baseline_nonAR": did_eval_result.get("acc_baseline_nonAR"),
        "n_baseline": did_eval_result.get("n_baseline"),
        "rows_used": did_eval_result.get("n_windows"),
        "per_row_did": _per_row_stat(resolved, stat_did),
        "delta_match": summarize_delta_match(resolved),
        # --- sec 9.1.5 S2 ---
        "cell_void_missing_s2_fields": not s2_complete,
        "did_logp": didlogp_pt, "did_logp_ci": [didlogp_lo, didlogp_hi],
        "per_row_did_logp": per_row_did_logp,
    }
    if t2b_result is not None:
        cell["acc_copy"] = t2b_result.get("acc_copy")
        cell["acc_copy_se"] = t2b_result.get("acc_copy_se")
    else:
        cell["acc_copy"] = None
        cell["acc_copy_se"] = None
    return cell


def summarize_delta_match(resolved: list) -> dict:
    """AUDIT FIX (2026-07-12). sec 9.2 pins the placebo as DISTRIBUTION-matched in Delta,
    but nothing in the instrument ever MEASURED the match it claims -- it was asserted by
    construction and never reported. It is not exact, and it cannot be: Delta' is drawn
    from the pooled empirical Delta and then REJECTED if it lands before position 0, so
    for a candidate with a small k every large Delta' is rejected and the realized placebo
    Delta is pulled toward SHORTER distances. (Measured on a synthetic 1,600-candidate
    population with a realistic k-spread: placebo mean Delta ran ~35% BELOW true mean
    Delta, with zero fallback flags -- i.e. the 5%-flag gate does not see this at all.)
    That residual is inherent to the pinned procedure, not a deviation from it, but it is
    load-bearing: gap_placebo is the model's generic damage-sensitivity AT A DISTANCE
    PROFILE, and if the placebo's profile is systematically shorter than the true arm's,
    DiD is biased by whatever the model's damage-sensitivity-vs-distance slope is. This
    reports the realized profiles so the residual can be sized, disclosed, and (if it
    matters) corrected in analysis. It is a REPORT, not a gate -- adding a gate here would
    be inventing a pre-registration the design does not have."""
    d_true = [r["delta"] for r in resolved if r.get("delta") is not None]
    d_plac = [r["delta_placebo"] for r in resolved if r.get("delta_placebo") is not None]
    if not d_true or not d_plac:
        return {"n_true": len(d_true), "n_placebo": len(d_plac)}
    st, sp = sorted(d_true), sorted(d_plac)
    return {
        "n_true": len(st), "n_placebo": len(sp),
        "mean_delta_true": _mean(st), "mean_delta_placebo": _mean(sp),
        "median_delta_true": st[len(st) // 2], "median_delta_placebo": sp[len(sp) // 2],
        "mean_shift_placebo_minus_true": _mean(sp) - _mean(st),
    }


def placebo_flagged_fraction_from_records(records: list) -> float:
    if not records:
        return 1.0
    return sum(1 for r in records if r.get("placebo_flagged")) / len(records)


def _per_row_stat(records: list, stat_fn) -> dict:
    """Per-row (cluster) values of `stat_fn`, needed by the trend-fit layer's
    own row-level bootstrap (sec 9.5)."""
    by_row = {}
    for r in records:
        by_row.setdefault(r["row_idx"], []).append(r)
    return {ri: stat_fn(recs) for ri, recs in by_row.items() if recs}


# =============================================================================
# 10. sec 9.1 -- THE NORMALIZATION SLOT. Pluggable; kept general even though
#     sec 9.1 is now PINNED (commit 7381d2a, blind agent, forced by N1-N3's
#     algebraic argument -- PARAM_AXIS_SCALING_DESIGN.md sec 9.1.1). The
#     mechanism (register/lookup/fail-loud-if-unset-or-unknown) is UNCHANGED
#     by the pin landing -- only ONE entry is now registered by name below,
#     exactly as the pin's own text prescribes registering it, so a future
#     metric (or a PI override) can still add/replace entries without a code
#     change here.
# =============================================================================
NORMALIZATION_REGISTRY: dict = {}


def register_normalization(name: str, fn, override: bool = False) -> None:
    """Registers a normalization form `fn(cell: dict) -> float`, i.e. the
    WHOLE verdict-grade cell (not just `did`/`acc_baseline_nonAR` positionally)
    -- this is the calling convention sec 9.1's pin itself specifies
    verbatim: `register_normalization("raw_did", lambda cell: cell["did"])`.
    Passing the whole cell is what lets a future normalization reach fields
    other than `did` (e.g. `acc_copy`) without a signature change here.

    AUDIT FIX (2026-07-12, S2 follow-up audit): refuses to SILENTLY REPLACE an
    already-registered name. sec 9.1's pin is registered at import time as `"raw_did"`;
    before this fix any later `register_normalization("raw_did", <anything>)` -- a stray
    line in the driver, a copy-paste of the pin's own literal text with a different body --
    would have overwritten the PINNED form with no warning at all, and
    `compute_capacity_metric(cell, "raw_did", ...)` would have kept returning a number under
    the pinned name while computing something else. That is a silent pre-registration
    violation, which is the one failure mode this registry exists to prevent. Re-registering
    the IDENTICAL function object is a no-op (so `importlib.reload` / double-import stay
    safe); replacing a name with a different function requires `override=True`, out loud."""
    if name in NORMALIZATION_REGISTRY and NORMALIZATION_REGISTRY[name] is not fn and not override:
        raise RuntimeError(
            f"REFUSING TO SILENTLY OVERWRITE the registered normalization {name!r} with a "
            f"different function. sec 9.1's pin (PARAM_AXIS_SCALING_DESIGN.md sec 9.1, commit "
            f"7381d2a) registers 'raw_did' -> M(r) = DiD(r) at import time, and a silent "
            f"rebinding of a pinned name is a pre-registration violation, not a code detail. "
            f"If you genuinely mean to replace it, pass override=True and record why."
        )
    NORMALIZATION_REGISTRY[name] = fn


def _raw_did(cell: dict) -> float:
    """sec 9.1 THE PIN (PARAM_AXIS_SCALING_DESIGN.md sec 9.1, commit 7381d2a):
    M(r) = DiD(r), denominator 1. FORCED by N1-N3 (sec 9.1.1's algebraic
    identity: the intact arm cancels exactly, DiD = E[C-B], so it is already
    competence-invariant and a denominator that varies with general
    competence would re-inject exactly the confound the numerator purged --
    "a double correction, applied in the wrong functional form"). Registered
    under this exact call, per the pin's own text: `register_normalization
    ("raw_did", lambda cell: cell["did"])`. NOT hardcoded into
    `compute_capacity_metric` -- it goes through the SAME registry lookup as
    any future normalization would, and normalization=None or an unknown
    name still fails loudly (see compute_capacity_metric)."""
    return cell["did"]


register_normalization("raw_did", _raw_did)


def compute_capacity_metric(cell: dict, normalization, *, t2b2_pass: bool) -> float:
    """The ONLY function that turns a verdict-grade cell (sec 9's DiD, N1)
    into the reported capacity metric M(r). Requires `cell['_verdict_grade']
    is True` (produced only by `finalize_cell`), `normalization` to be a
    NON-None, REGISTERED name, the caller to supply this cell's T2b-2
    disposition, and (sec 9.1.5 BUILD REQUIREMENT) the cell to carry complete
    S2 log-prob fields. Never defaults.

    AUDIT FIX (2026-07-12): `t2b2_pass` is a REQUIRED keyword with no default, and a
    placebo-unmatched cell is refused outright. Before this fix the guard chain checked
    only (stamp + registered normalization), so a cell that FAILED T2b-2 -- i.e. a rung
    reporting a recall gap LARGER than its own demonstrated copy ability, which sec 9.4
    declares VOID and calls "the single line that would have caught the VOID build" --
    still returned a number, as did a cell whose placebo never distribution-matched. The
    `_verdict_grade` stamp only ever certified "a placebo arm was run"; it never certified
    "this rung may carry a verdict." Those are now separate, and both are enforced here.
    Note this deliberately does NOT gate on T1a: a T1a-failing rung is FLOOR, and sec 9.5
    requires FLOOR rungs to be REPORTED (just excluded from the fit), not refused."""
    if not cell.get("_verdict_grade"):
        raise VerdictGradeError(
            "compute_capacity_metric received a cell dict that was not produced by "
            "finalize_cell() (missing _verdict_grade=True stamp) -- refusing to compute a "
            "verdict-grade metric from an unvalidated/un-placebo'd input."
        )
    if cell.get("cell_void_placebo_match", True):
        raise VerdictGradeError(
            f"CELL VOID (placebo not distribution-matched): "
            f"{cell.get('rung')}/{cell.get('corpus')} has placebo_flagged_frac="
            f"{cell.get('placebo_flagged_frac')} > {PLACEBO_FLAGGED_FRAC_VOID_THRESHOLD} "
            f"(PARAM_AXIS_SCALING_DESIGN.md sec 9.2). A cell whose placebo arm is not "
            f"distribution-matched cannot carry a DiD, and therefore cannot carry M(r)."
        )
    if cell.get("cell_void_missing_s2_fields", True):
        raise VerdictGradeError(
            f"CELL VOID (missing sec 9.1.5 S2 log-prob fields): "
            f"{cell.get('rung')}/{cell.get('corpus')} has at least one resolved candidate "
            f"without logp_intact/logp_true_ablated/logp_placebo_ablated. sec 9.1.5: the S2 "
            f"log-prob sensitivity 'cannot be added after a read (that would be a re-read, "
            f"banned by sec 9.6)' -- refusing to emit M(r) for a cell that cannot support it."
        )
    if not t2b2_pass:
        raise VerdictGradeError(
            f"RUNG VOID (T2b-2 ceiling violated): {cell.get('rung')}/{cell.get('corpus')} "
            f"reports DiD={cell.get('did')} exceeding acc_copy + 2*SE. sec 9.4: 'A rung "
            f"reporting an in-context recall gap larger than its own demonstrated in-context "
            f"copy ability is internally contradictory and its gap is measuring something "
            f"else. Fail => the rung is VOID (not FLOOR -- the instrument is returning an "
            f"impossible number at that rung, which is a defect, not a measurement).' "
            f"Refusing to emit M(r) for it."
        )
    if normalization is None:
        raise RuntimeError(
            "NORMALIZATION_UNSET: PARAM_AXIS_SCALING_DESIGN.md sec 9.1 has not been pinned yet. "
            "This is not a bug -- sec 9.1 is deliberately left open pending a blind pin (sec 9.7's "
            "handoff protocol) or a direct PI decision. Do not pass a guessed default; wait for "
            "the pin, then `register_normalization(name, fn)` and pass that `name` here."
        )
    if normalization not in NORMALIZATION_REGISTRY:
        raise KeyError(
            f"Unknown normalization {normalization!r}. Registered: "
            f"{sorted(NORMALIZATION_REGISTRY)!r}. The sec 9.1 pin must call "
            f"register_normalization({normalization!r}, <fn>) before this can run for real."
        )
    return NORMALIZATION_REGISTRY[normalization](cell)


# =============================================================================
# 11. T2 -- marker-token picker REUSED VERBATIM (sound per this build's
#     instruction; only the distance/bar fed into it were the M-11 sin).
# =============================================================================
def pick_t2_marker_tokens(train_tokens: torch.Tensor, vocab_size: int, device: str,
                           top_k: int = 300, min_freq: int = 200,
                           entropy_pool: int = 400) -> tuple:
    """Picks two WELL-TRAINED (individually frequent) token ids whose PAIRING
    never co-occurs adjacently in train -- an OOD transition for a model with
    otherwise well-trained embeddings, chosen by next-token ENTROPY (not raw
    frequency) so a low-entropy token's own crushing prior can't be mistaken
    for a copy-mechanism failure."""
    t = train_tokens.to(device)
    counts = torch.bincount(t, minlength=vocab_size)
    counts[EOT_TOKEN_ID] = 0
    freq_ids = torch.topk(counts, entropy_pool).indices.tolist()
    freq_ids = [i for i in freq_ids if counts[i].item() >= min_freq]

    a_next = t[:-1]; b_next = t[1:]
    pair_ids = a_next.to(torch.int64) * vocab_size + b_next.to(torch.int64)
    uniq, pair_counts = torch.unique(pair_ids, return_counts=True)
    a_of_pair = uniq // vocab_size

    entropy_per_a = {}
    for a in freq_ids:
        mask = a_of_pair == a
        c = pair_counts[mask].to(torch.float64)
        if c.numel() == 0:
            continue
        p = c / c.sum()
        entropy_per_a[a] = float(-(p * p.log()).sum().item())
    ranked_a = sorted(entropy_per_a, key=lambda a: -entropy_per_a[a])[:top_k]

    observed = set(uniq.tolist())
    b_candidates = list(freq_ids)
    for a in ranked_a:
        for b in b_candidates:
            if b == a:
                continue
            if (a * vocab_size + b) not in observed and (b * vocab_size + a) not in observed:
                return a, b
    raise RuntimeError("no high-entropy OOD-pairing found -- widen entropy_pool/top_k")


# =============================================================================
# 12. T2a (instrument-teeth, reference models) -- WIRED, NOT EXECUTED this
#     session (needs RWKV7-Goose-1.5B / falcon-mamba-7b on the box's
#     /data/hf_cache; out of this build's scope, disclosed in the report).
#     T2b (rung-admissibility) -- IMPLEMENTED AND VALIDATED below.
# =============================================================================
def run_t2_planted_copy(model, val_tokens: torch.Tensor, batch_size: int, seq_len: int,
                         device: str, seed: int, delta_pool: list, tok_a: int, tok_b: int,
                         n_plants: int, vocab_size: int = VOCAB_SIZE,
                         eval_micro_batch: int = EVAL_MICRO_BATCH_DEFAULT) -> dict:
    """One planted (tok_a,tok_b) repeat per window-row, at a distance drawn
    from `delta_pool` -- the main metric's OWN empirical Delta distribution
    (sec 9.4: 'at distances drawn from the main metric's own empirical Delta
    distribution', replacing both of the old file's arbitrary 350/20
    constants). T2 IS the main instrument applied to a synthetic ground-
    truth-known candidate (sec 9.4's own framing: 'a positive control at the
    measured task's true difficulty') -- this function reuses
    `assign_placebo_positions` / `run_ablation_arm` verbatim, the SAME
    machinery as `run_did_eval`, not a parallel reimplementation. Returns
    per-plant records shaped exactly like run_did_eval's (ready for
    check_t2b1_mechanism_exists / check_t2b2_ceiling) plus acc_copy/SE for
    the ceiling check's own inputs. This is the function T2b (rung-
    admissibility, on OUR OWN checkpoints) uses; T2a (the reference-model
    instrument-teeth gate) calls the SAME function pointed at a reference
    model instead -- neither is executed against a real model in this build
    session (disclosed scope boundary, see the build report)."""
    gen = torch.Generator(device=device).manual_seed(seed + 555)
    plant_rng = random.Random(_combine_seed(seed, "t2_plant"))
    # AUDIT FIX (2026-07-12): one get_batch call, NOT `batch_size`-sized chunks -- the
    # host windows must be batch-size-independent for the same reason run_did_eval's are
    # (CUDA Philox advances per kernel launch; measured 50% window divergence between
    # batch_size 16 and 32 at the same seed). `batch_size` is provenance-only here too.
    x0, y0 = get_batch(val_tokens, n_plants, seq_len, gen)
    window = _make_window(x0, y0).clone()
    all_windows, specs = [], []
    for b in range(window.shape[0]):
        delta = max(2, min(plant_rng.choice(delta_pool), seq_len - 4))
        k0 = plant_rng.randrange(delta + 2, seq_len - 1)
        j0 = k0 - delta
        window[b, j0] = tok_a
        window[b, j0 + 1] = tok_b
        window[b, k0] = tok_a
        all_windows.append(window[b].clone())
        specs.append({"row_idx": len(specs), "k": k0, "j": j0, "p": j0 + 1, "delta": delta})

    if not specs:
        return {"records": [], "acc_copy": None, "acc_copy_se": None, "n_plants": 0}

    sel_row_candidates = [[(s["k"], s["j"])] for s in specs]   # one plant per row -- itself is the
                                                                 # only "other candidate" in its row
    window_cpu = [w.cpu().tolist() for w in all_windows]
    specs = assign_placebo_positions(specs, sel_row_candidates, window_cpu, seed)

    ablate_rng = random.Random(_combine_seed(seed, "t2_ablate"))
    for s in specs:
        s["target"] = tok_b
        s["repl_true"] = draw_exclusive_replacement(window_cpu[s["row_idx"]], s["p"], tok_b,
                                                      vocab_size, ablate_rng)
    for s in specs:
        if s["p_placebo"] is not None:
            s["repl_placebo"] = draw_exclusive_replacement(window_cpu[s["row_idx"]], s["p_placebo"],
                                                             tok_b, vocab_size, ablate_rng)
        else:
            s["repl_placebo"] = None

    x_all = torch.stack([w[:-1] for w in all_windows])
    pred_chunks = []
    for start in range(0, x_all.shape[0], eval_micro_batch):
        with torch.no_grad():
            logits = model(x_all[start:start + eval_micro_batch])
        pred_chunks.append(logits.argmax(dim=-1))
    pred_intact = torch.cat(pred_chunks, dim=0)

    run_ablation_arm(model, all_windows, specs, "p", "repl_true", "pred_true", eval_micro_batch, device)
    run_ablation_arm(model, all_windows, specs, "p_placebo", "repl_placebo", "pred_placebo",
                      eval_micro_batch, device)

    records = []
    for s in specs:
        hit_intact = int(pred_intact[s["row_idx"], s["k"]].item() == tok_b)
        hit_true = int(s.get("pred_true") == tok_b) if "pred_true" in s else None
        hit_placebo = int(s.get("pred_placebo") == tok_b) if s.get("pred_placebo") is not None else None
        records.append({"row_idx": s["row_idx"], "k": s["k"], "delta": s["delta"],
                         "hit_intact": hit_intact, "hit_true_ablated": hit_true,
                         "hit_placebo_ablated": hit_placebo, "placebo_flagged": s["placebo_flagged"]})
    hits = sum(r["hit_intact"] for r in records)
    acc_copy = hits / len(records)
    acc_copy_se = binomial_se(hits, len(records))
    return {"records": records, "acc_copy": acc_copy, "acc_copy_se": acc_copy_se, "n_plants": len(records)}


def check_t2b1_mechanism_exists(paired_records: list) -> dict:
    """T2b-1 (sec 9.4): exact paired sign test on discordant pairs -- n_plus
    = #{placebo-ablation still correct AND true-ablation wrong} (true
    corruption did the damage, placebo didn't), n_minus = the reverse.
    H0: p=0.5 (no asymmetry -> no mechanism). Reject at p<0.001 AND
    n_plus>n_minus to claim 'mechanism exists'."""
    n_plus = sum(1 for r in paired_records
                 if r["hit_placebo_ablated"] == 1 and r["hit_true_ablated"] == 0)
    n_minus = sum(1 for r in paired_records
                  if r["hit_placebo_ablated"] == 0 and r["hit_true_ablated"] == 1)
    n = n_plus + n_minus
    if n == 0:
        return {"n_plus": 0, "n_minus": 0, "n_informative": 0, "p_value": 1.0, "passes": False,
                "note": "no discordant pairs -- true and placebo ablation always agree"}
    p_value = _exact_binomial_two_sided_p(n_plus, n)
    return {"n_plus": n_plus, "n_minus": n_minus, "n_informative": n, "p_value": p_value,
            "passes": bool(p_value < 0.001 and n_plus > n_minus)}


def _exact_binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0

    def pmf(x):
        return math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

    p_obs = pmf(k)
    total = 0.0
    for x in range(n + 1):
        px = pmf(x)
        if px <= p_obs * (1 + 1e-9):
            total += px
    return min(1.0, total)


def check_t2b2_ceiling(did_value: float, acc_copy: float, acc_copy_se: float) -> dict:
    """T2b-2 (sec 9.4), THE NEW CHECK: one-shot planted copy is the maximally
    favourable case of the mechanism the main metric probes, so acc_copy is
    an UPPER BOUND on the fraction of candidates whose answer can be
    antecedent-attributable. A rung with DiD(r) > acc_copy(r) + 2*SE is
    internally contradictory -- its gap measures something else.
    `passes=False` => the rung is VOID (a defect, not merely FLOOR). This is
    the single line regate 10.2 says would have caught the VOID build's
    wikitext self-contradiction -- a non-trivial recall gap reported at a
    rung with zero demonstrated copy ability. Exact figures QUARANTINED,
    see matrix-thinking/QUARANTINE_r0_did_values.md (this file is on the
    permitted-read list for T2-repair work; the value is not)."""
    bound = acc_copy + 2.0 * acc_copy_se
    return {"did": did_value, "acc_copy": acc_copy, "acc_copy_se": acc_copy_se,
            "bound": bound, "margin": did_value - bound, "passes": did_value <= bound}


def binomial_se(hits: int, n: int) -> float:
    """The plain (Wald / normal-approximation) binomial SE, sqrt(p(1-p)/n) -- this is
    the 'SE' of sec 9.4's `DiD(r) <= acc_copy(r) + 2*SE`.

    AUDIT FIX (2026-07-12): this was named `wilson_se`, which it is NOT. The name mattered:
    at acc_copy = 0 the Wald SE is exactly 0, so the T2b-2 bound collapses to 0.0 and ANY
    DiD > 0 voids the rung -- which is the strict, teeth-ful behaviour the spec's literal
    'acc_copy + 2*SE' asks for, and is what fires on the VOID build's zero-copy wikitext
    cells. A genuine Wilson-based bound would be ~0.019 at p=0, n=200 and would NOT collapse.
    Behaviour is unchanged (it always was Wald); only the misleading name is."""
    if n == 0:
        return float("nan")
    p = hits / n
    return math.sqrt(p * (1 - p) / n)


# =============================================================================
# 13. sec 9.6 admissibility gates.
# =============================================================================
def check_rung_admissible(cell_by_corpus: dict, tok_per_param: float, checkpoint_quiesced: bool,
                           t2b2_by_corpus: dict, t2b1_by_corpus: dict,
                           min_candidates: int = 4096) -> dict:
    """sec 9.6: a rung enters the admissible set A iff ALL items hold, on
    BOTH corpora (item 6 -- 'A rung is admissible only if admissible on
    both'; never silently narrowed to the corpus where the instrument
    passes, regate S-5's precedent). Returns a per-item breakdown plus the
    overall boolean so a caller can see exactly which gate failed."""
    reasons = {}
    corpora = sorted(cell_by_corpus)
    reasons["both_corpora_present"] = set(corpora) >= {"wikitext-mix-ext", "openr1-mix-ext"} \
        if len(corpora) else False
    reasons["checkpoint_quiesced"] = checkpoint_quiesced
    reasons["tok_per_param_floor"] = tok_per_param >= TOK_PER_PARAM_PRIMARY_FIT_FLOOR

    per_corpus_ok = {}
    for c in corpora:
        cell = cell_by_corpus[c]
        ok = (
            not cell.get("cell_void_placebo_match", True)
            and cell.get("n_candidates_resolved", 0) >= min_candidates
            and cell.get("t1a_pass_did_ci_excludes_zero", False)
            and t2b1_by_corpus.get(c, {}).get("passes", False)
            and t2b2_by_corpus.get(c, {}).get("passes", False)
        )
        per_corpus_ok[c] = ok
    reasons["per_corpus"] = per_corpus_ok
    any_t2b2_fail = any(not t2b2_by_corpus.get(c, {}).get("passes", True) for c in corpora)
    # AUDIT FIX (2026-07-12): a placebo-match failure is a VOID, not a FLOOR.
    # sec 9.2 is literal: "If the flagged fraction exceeds 5% in any (rung, corpus) cell,
    # the placebo is not distribution-matched and the cell is VOID." Before this fix the
    # only VOID trigger was T2b-2, so a cell whose placebo never distribution-matched was
    # reported as FLOOR_OR_EXCLUDED -- and FLOOR/EXCLUDED means "quietly drop this rung and
    # fit the trend through the others," whereas VOID means "HALT, no verdict, diagnose"
    # (sec 9.5's precedence VOID -> FLOOR -> table). Opposite operational consequences from
    # the same failure. Defaults to True (fail-closed) when the key is absent.
    any_placebo_void = any(cell_by_corpus[c].get("cell_void_placebo_match", True) for c in corpora)
    reasons["placebo_distribution_matched"] = not any_placebo_void

    admissible = (
        reasons["both_corpora_present"]
        and reasons["checkpoint_quiesced"]
        and reasons["tok_per_param_floor"]
        and all(per_corpus_ok.get(c, False) for c in corpora)
    )
    if any_t2b2_fail or any_placebo_void:
        verdict = "VOID"
    else:
        verdict = "ADMISSIBLE" if admissible else "FLOOR_OR_EXCLUDED"
    return {"admissible": admissible, "verdict_if_not_admissible": verdict, "reasons": reasons}


# =============================================================================
# 14. sec 9.5 verdict-map machinery (trend fit + TOST-FLAT + the exhaustive
#     table). Generic/pluggable: the exact statistical treatment of the
#     normalized M(r) gets a final pass once sec 9.1's form is known; the
#     DiD-level machinery above is what this build's validation targets.
# =============================================================================
def ols_slope(xs: list, ys: list) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / den


def bootstrap_trend_ci(per_rung_row_values: dict, log10_params: dict, n_boot: int = 2000,
                        seed: int = 0) -> dict:
    """per_rung_row_values: {rung_name: {row_idx: value}} (already-normalized
    M(r) per row, i.e. the caller has ALREADY run compute_capacity_metric per
    row via a per-row DiD/baseline pair -- left to the caller since the exact
    per-row normalization depends on sec 9.1's still-open form). Resamples
    rows WITHIN EACH RUNG independently per bootstrap draw, refits beta =
    OLS slope of per-rung mean M against log10(params)."""
    rungs = sorted(per_rung_row_values)
    xs = [log10_params[r] for r in rungs]
    point_ys = [_mean(list(per_rung_row_values[r].values())) for r in rungs]
    point_beta = ols_slope(xs, point_ys)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ys = []
        for r in rungs:
            vals = list(per_rung_row_values[r].values())
            resampled = [rng.choice(vals) for _ in range(len(vals))]
            ys.append(_mean(resampled))
        boots.append(ols_slope(xs, ys))
    boots.sort()
    return {"beta_point": point_beta, "boots": boots,
            "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]]}


def tost_flat(boots: list, delta: float, alpha: float = 0.10) -> dict:
    """sec 9.5: 'FLAT' requires the 95% CI to include 0 AND 'TOST at 90% CI' to
    establish |beta| < delta.

    AUDIT FIX (2026-07-12): the previous body took `b[int(alpha*n)] .. b[int((1-alpha)*n)-1]`
    with alpha=0.10, i.e. the 10th..90th percentiles -- an **80% interval**, while the
    returned key was (and the spec requires) `ci90`. An 80% interval is NARROWER, so TOST
    passed more easily and FLAT would have been declared MORE often than the
    pre-registration permits -- a verdict-boundary error in the "manufacture a result"
    direction, on the one branch (FLAT-COUPLED) sec 9.5 calls "the third outcome, and it is
    the one sec 5.2 could not express." `alpha` is the TOTAL tail mass (the standard TOST
    convention: alpha=0.10 <=> two one-sided tests at 5% <=> the 90% CI)."""
    b = sorted(boots)
    n = len(b)
    lo = b[int(alpha / 2 * n)]
    hi = b[min(n - 1, int((1 - alpha / 2) * n) - 1)]
    return {"tost_pass": bool(lo > -delta and hi < delta), "ci90": [lo, hi],
            "delta": delta, "alpha_total_tail_mass": alpha}


def classify_recall_trend(ci95: list, tost: dict) -> str:
    lo, hi = ci95
    if lo > 0:
        return "RISES"
    if hi < 0:
        return "DECLINES"
    return "FLAT" if tost["tost_pass"] else "INDETERMINATE"


def compute_verdict(factor1_trend: str, span_frac_monotone_over_A: bool,
                     n_admissible_rungs: int, t1a_positive_count: int,
                     *, factor1_trend_s2: str) -> str:
    """sec 9.5's exhaustive table, precedence VOID -> FLOOR -> the table, with sec 9.1.5's
    S2 disagreement rule enforced FIRST (it can only withhold a verdict, never create one).
    Callers are responsible for having already produced BOTH classifications via
    `classify_recall_trend` (one on the primary DiD trend, one on the DiD_logp trend) and
    for having enforced sec 9.6 admissibility upstream (this function does not re-derive
    VOID from T1c/T2a/sec9.6 -- those live in check_rung_admissible / T2a).

    AUDIT FIX (2026-07-12, S2 follow-up audit): `factor1_trend_s2` is a REQUIRED,
    KEYWORD-ONLY argument with NO default, and the disagreement rule is enforced HERE.
    sec 9.1.5 pins that rule as something that "must be structurally true, not just a
    convention someone remembers to apply" -- but as first built, the rule lived ONLY inside
    the `compute_verdict_with_s2` wrapper while `compute_verdict` (the obvious name, and the
    one sec 9.5's own prose describes) remained callable with no S2 disposition at all. A
    driver author calling it directly would have received a full COUPLED / DECOUPLED /
    FLAT-COUPLED verdict with S2 silently skipped -- precisely the forgotten-convention
    failure the pin forbids. There is now NO path through this module that yields a verdict
    without both classifications: a stale 4-positional-argument caller raises TypeError
    (loud), it does not quietly return a verdict."""
    if factor1_trend != factor1_trend_s2:
        # sec 9.1.5's PRE-COMMITTED DISAGREEMENT RULE. Checked before FLOOR so that the
        # behaviour is identical to the wrapper's original ordering; either way the result is
        # a NON-verdict, so the ordering cannot create or strengthen one.
        return "INDETERMINATE"
    if n_admissible_rungs < 3 or t1a_positive_count < 3:
        return "FLOOR"
    if factor1_trend == "INDETERMINATE":
        return "INDETERMINATE"
    coupling_licensed = span_frac_monotone_over_A
    if not coupling_licensed:
        return "RECALL-TREND-ONLY"
    return {"DECLINES": "COUPLED", "RISES": "DECOUPLED", "FLAT": "FLAT-COUPLED"}[factor1_trend]


def compute_verdict_with_s2(factor1_trend_primary: str, factor1_trend_s2: str,
                             span_frac_monotone_over_A: bool, n_admissible_rungs: int,
                             t1a_positive_count: int) -> str:
    """sec 9.1.5's PRE-COMMITTED DISAGREEMENT RULE, made structural (not a
    convention a caller has to remember): 'If S2's Factor-1 classification
    differs from the primary's, the verdict is INDETERMINATE and we say so.'
    S2 is a readout-robustness check on the SAME estimand (sec 9.1.5: 'so a
    disagreement is an instrument defect, not a finding') -- this is
    verdict-WITHHOLDING only, exactly like S1 (below): it can only downgrade
    to INDETERMINATE, never create or strengthen a verdict. Both
    `factor1_trend_primary` and `factor1_trend_s2` must already have been
    produced by `classify_recall_trend` (one on the primary DiD trend, one
    on the DiD_logp trend) -- this function does not recompute them.

    AUDIT FIX (2026-07-12, S2 follow-up audit): the disagreement rule itself now lives in
    `compute_verdict` (which REQUIRES `factor1_trend_s2=`), so it can no longer be bypassed
    by calling the inner function directly. This wrapper is retained as sec 9.1.5's named
    entry point and is a pure delegation -- the behaviour of every one of the 16
    primary x S2 combinations is unchanged."""
    return compute_verdict(factor1_trend_primary, span_frac_monotone_over_A,
                            n_admissible_rungs, t1a_positive_count,
                            factor1_trend_s2=factor1_trend_s2)


def compute_s1_utilization(cell: dict) -> dict:
    """sec 9.1.5 S1: the utilization ratio DiD(r)/acc_copy(r) (the rejected
    normalization candidate C5, promoted to a MANDATORY, NON-VERDICT-CARRYING
    sensitivity). Requires `cell['acc_copy']` (wire via `finalize_cell`'s
    `t2b_result=` param).

    AUDIT NOTE (2026-07-12, S2 follow-up audit): sec 9.1.5's prose says S1 is "bounded to
    [0,1] by the already-pinned T2b-2 ceiling." IT IS NOT, quite -- and this docstring said so
    too before the correction. T2b-2 admits `DiD <= acc_copy + 2*SE`, so the ratio's true
    ceiling is `1 + 2*SE/acc_copy`, which exceeds 1 whenever the copy ability is small
    (worked, T2b-2-PASSING example: acc_copy=0.05, SE=0.015 -> bound 0.08 -> S1 = 1.60), and
    DiD < 0 makes S1 negative. Report S1 as the raw ratio it is; do NOT clip it to [0,1] and
    do not read a >1 value as an instrument failure. This has ZERO verdict consequence (S1 is
    structurally verdict-withholding -- it is never fed to compute_verdict /
    compute_capacity_metric), which is the only reason it is a note and not a defect.

    At acc_copy == 0 (a rung with zero demonstrated copy ability) the ratio is undefined and is
    reported as None with a reason rather than raising ZeroDivisionError or
    silently reading +inf -- sec 9.4's T2b-2 already voids any such rung
    before this could be called on it (acc_copy=0 forces DiD<=0 to pass the
    ceiling), so this is defensive, not a code path this instrument's own
    gates are expected to reach in practice. THIS FUNCTION MUST NEVER BE FED
    INTO compute_verdict / compute_capacity_metric -- sec 9.1.5: 'S1 cannot
    change the verdict,' it is reported alongside the primary per the
    pre-registered reading table in sec 9.1.5, never substituted for it."""
    did = cell.get("did")
    acc_copy = cell.get("acc_copy")
    if did is None or acc_copy is None:
        return {"value": None, "did": did, "acc_copy": acc_copy,
                "reason": "cell carries no acc_copy -- finalize_cell was not given a "
                          "t2b_result for this (rung, corpus)"}
    if acc_copy == 0:
        return {"value": None, "did": did, "acc_copy": acc_copy,
                "reason": "acc_copy == 0 -- utilization ratio undefined (T2b-2 already "
                          "voids any rung where this combines with DiD > 0)"}
    return {"value": did / acc_copy, "did": did, "acc_copy": acc_copy, "reason": None}


# =============================================================================
# SMOKE GATE -- CPU-stub-friendly logic validation, PLUS two hand-constructed
# demonstrations the design's build brief explicitly asks for: the synthetic
# no-recall-but-context-fragile model (placebo teeth) and the T2b-2 ceiling
# check firing on a violating case (VOID teeth). Also a real-checkpoint-shaped
# (but toy-vocab, untrained) end-to-end pass through DeltaNetLM, matching this
# repo's own smoke-gate convention.
# =============================================================================
class _SyntheticContextFragileModel(torch.nn.Module):
    """A hand-built next-token 'model' with ZERO in-context recall of any
    SPECIFIC antecedent, but WITH generic sensitivity to ANY corrupted
    upstream token -- exactly the confound the placebo arm exists to remove
    (design sec 9.2). Given a fixed REFERENCE window, it predicts the
    reference's own next token at position t IFF the input's ENTIRE prefix
    through t matches the reference exactly; otherwise it predicts a fixed
    `wrong_token`. It never inspects WHICH position changed -- antecedent-
    blind by construction. A correctly-identifying instrument must therefore
    read DiD ~= 0 (in fact EXACTLY 0, since this model is deterministic and
    both the TRUE and PLACEBO corruption positions are always < k, i.e.
    always in the queried prefix) even though the RAW gap (intact accuracy
    vs true-ablated accuracy) is LARGE."""

    def __init__(self, reference_window: torch.Tensor, vocab_size: int, wrong_token: int):
        """reference_window: the FULL (T+1,)-length window (x followed by the
        final y token) -- NOT just the x-half. Storing the full window is
        what lets forward() read the TRUE next-token at every one of the T
        input positions, including the last; slicing to just the x-half and
        reconstructing "next token" via a wraparound roll() was an earlier
        draft's off-by-one bug (wrapped to position 0 at the final step)."""
        super().__init__()
        self.register_buffer("reference", reference_window.clone())
        self.vocab_size = vocab_size
        self.wrong_token = wrong_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        ref_x = self.reference[:T].to(x.device).unsqueeze(0).expand(B, -1)
        ref_next = self.reference[1:T + 1].to(x.device)   # TRUE next-token per position, no wraparound
        matches = (x == ref_x)
        clean_prefix = torch.cumprod(matches.long(), dim=1).bool()  # True at t iff x[0..t]==ref[0..t]
        logits = torch.full((B, T, self.vocab_size), -10.0, device=x.device)
        logits[:, :, self.wrong_token] = 0.0
        b_idx, t_idx = clean_prefix.nonzero(as_tuple=True)
        if b_idx.numel():
            correct_tok = ref_next[t_idx]
            logits[b_idx, t_idx, :] = -10.0
            logits[b_idx, t_idx, correct_tok] = 0.0
        return logits


def _build_synthetic_ref_window(V: int, T: int, wrong_token: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ref = torch.randint(0, V - 1, (T + 1,), generator=g)  # never touches wrong_token = V-1
    # plant TWO genuine repeats at well-separated distances so the placebo has
    # somewhere admissible to land and the pooled Delta distribution has >1 value
    ref[10], ref[11] = 5, 6
    ref[150], ref[151] = 5, 6      # candidate k=150, j=10, delta=140
    ref[30], ref[31] = 7, 8
    ref[300], ref[301] = 7, 8      # candidate k=300, j=30, delta=270
    return ref


def smoke(device: str) -> int:
    print("=" * 70 + "\n  LM_RECALL_GAP_PROBE_V2_RD SMOKE GATE (sec 9 rebuild)\n" + "=" * 70)
    ok_all = True

    def report(name, ok, detail=""):
        nonlocal ok_all
        ok_all = ok_all and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")

    # --- [0] FIX-A resolver, reused verbatim -- sanity re-check ---
    with tempfile.TemporaryDirectory() as td:
        for s in (1000, 2000, 3000, 5000):
            open(os.path.join(td, f"x_step{s}.pt"), "w").close()
        r = resolve_token_matched_checkpoint(td, batch_size=32, seq_len=512,
                                              target_tokens=2000 * 32 * 512 + 100, prefix="x")
        report("FIX-A resolves nearest step (2000)", r["chosen_step"] == 2000, str(r))

    # --- [1] THE FATAL-1 UNIT TEST: row-replication carries EXACTLY ONE
    #     corrupted token per output row, assert on the tensor diff. ---
    base = torch.arange(20, dtype=torch.int64)
    corrupt_positions = [3, 3, 7, 15]
    repl_tokens = [99, 98, 97, 96]
    rep = build_replicated_ablation_batch(base, corrupt_positions, repl_tokens)
    diffs = (rep != base.unsqueeze(0)).sum(dim=1)
    report("row-replication: EVERY row differs from base at EXACTLY one position",
           bool((diffs == 1).all()), f"per-row diff counts={diffs.tolist()} (expected all 1)")
    correct_positions = all(rep[m].ne(base).nonzero().item() == corrupt_positions[m] for m in range(4))
    report("row-replication: the ONE diff is at the INTENDED position for every row",
           correct_positions)
    report("row-replication: two candidates sharing a corrupt POSITION (3) get "
           "INDEPENDENT rows (not merged/overwritten)",
           rep[0, 3].item() == 99 and rep[1, 3].item() == 98
           and torch.equal(rep[0][torch.arange(20) != 3], base[torch.arange(20) != 3])
           and torch.equal(rep[1][torch.arange(20) != 3], base[torch.arange(20) != 3]))

    # --- [2] Placebo admissibility: forbidden positions are genuinely excluded ---
    x_cpu_toy = [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2] + [9] * 8]
    row_candidates_toy = [[(11, 1), (6, 0)]]  # two candidates in the SAME row, antecedents at j+1=2 and j+1=1
    specs_toy = true_arm_specs(row_candidates_toy)
    specs_toy = assign_placebo_positions(specs_toy, row_candidates_toy, x_cpu_toy, seed=1)
    forbidden_check = all(
        s["p_placebo"] not in {s["p"], s["j"], 2, 1} and s["p_placebo"] < s["k"]
        for s in specs_toy if s["p_placebo"] is not None
    )
    report("placebo assignment: never lands on p, j, k-or-later, or ANOTHER candidate's antecedent",
           forbidden_check, str(specs_toy))

    # --- [3] THE SYNTHETIC NO-RECALL DEMONSTRATION (placebo teeth). ---
    print("\n  [3] SYNTHETIC NO-RECALL DEMO: a context-FRAGILE-but-recall-BLIND model")
    print("      must yield DiD ~= 0 despite a LARGE raw gap_true.")
    V, T, wrong_token = 64, 400, 63
    ref = _build_synthetic_ref_window(V, T, wrong_token, seed=0)
    model_syn = _SyntheticContextFragileModel(ref, V, wrong_token).to(device)
    model_syn.eval()
    mode_next_syn = [-1] * V  # every continuation counts as non-modal -- pure plumbing test
    all_windows_syn = [ref.to(device)] * 3   # replicate the SAME reference across "rows" (independent
                                              # per-row corruption draws still apply -- exercises the
                                              # multi-row path, not just n=1)
    x_syn = torch.stack([w[:-1] for w in all_windows_syn])
    y_syn = torch.stack([w[1:] for w in all_windows_syn])
    with torch.no_grad():
        pred_intact_syn = model_syn(x_syn).argmax(dim=-1)
    acc_intact_syn = (pred_intact_syn == y_syn).float().mean().item()
    report("  synthetic model reads its OWN reference perfectly (acc_intact==1.0)",
           abs(acc_intact_syn - 1.0) < 1e-9, f"acc_intact={acc_intact_syn}")

    x_cpu_syn = x_syn.cpu().tolist()
    y_cpu_syn = y_syn.cpu().tolist()
    row_cands_syn, _ = detect_candidates_and_baseline(x_cpu_syn, y_cpu_syn, mode_next_syn, min_sep=2)
    sel_syn = select_candidates_per_row(row_cands_syn, c_max=C_MAX_DEFAULT, seed=42)
    n_found = sum(len(c) for c in sel_syn)
    # >=6: the 2 hand-planted repeats x 3 rows are guaranteed present; a V=64/T=400 random
    # sequence also contains INCIDENTAL bigram repeats by chance (birthday-paradox density) --
    # those are legitimate candidates too and do not undermine the demo (the synthetic model's
    # antecedent-blind rule applies uniformly to every candidate, planted or incidental).
    report("  hand-planted candidates detected (>=6: 2/row x 3 rows, plus any incidental repeats)",
           n_found >= 6, f"found={n_found}, per-row={[len(c) for c in sel_syn]}")

    specs_syn = true_arm_specs(sel_syn)
    specs_syn = assign_placebo_positions(specs_syn, sel_syn, x_cpu_syn, seed=42)
    ablate_rng_syn = random.Random(_combine_seed(42, "syn_ablate"))
    window_cpu_syn = [w.cpu().tolist() for w in all_windows_syn]
    for s in specs_syn:
        target = window_cpu_syn[s["row_idx"]][s["k"] + 1]
        s["target"] = target
        s["repl_true"] = draw_exclusive_replacement(window_cpu_syn[s["row_idx"]], s["p"], target, V, ablate_rng_syn)
    for s in specs_syn:
        if s["p_placebo"] is not None:
            s["repl_placebo"] = draw_exclusive_replacement(window_cpu_syn[s["row_idx"]], s["p_placebo"],
                                                             s["target"], V, ablate_rng_syn)
        else:
            s["repl_placebo"] = None
    run_ablation_arm(model_syn, all_windows_syn, specs_syn, "p", "repl_true", "pred_true", 64, device)
    run_ablation_arm(model_syn, all_windows_syn, specs_syn, "p_placebo", "repl_placebo", "pred_placebo", 64, device)

    records_syn = []
    for s in specs_syn:
        target = s["target"]
        hit_intact = int(pred_intact_syn[s["row_idx"], s["k"]].item() == target)
        hit_true = int(s["pred_true"] == target)
        hit_placebo = int(s["pred_placebo"] == target) if s.get("pred_placebo") is not None else None
        records_syn.append({"row_idx": s["row_idx"], "k": s["k"], "delta": s["delta"],
                             "hit_intact": hit_intact, "hit_true_ablated": hit_true,
                             "hit_placebo_ablated": hit_placebo, "placebo_flagged": s["placebo_flagged"]})
    resolved_syn = [r for r in records_syn if r["hit_placebo_ablated"] is not None]
    gap_true_syn = stat_gap_true(resolved_syn)
    gap_placebo_syn = stat_gap_placebo(resolved_syn)
    did_syn = stat_did(resolved_syn)
    report("  raw gap_true is LARGE (any single corrupted token breaks the WHOLE suffix "
           "in this model, by construction)", gap_true_syn > 0.9,
           f"gap_true={gap_true_syn:.4f}")
    report("  gap_placebo is ALSO large (generic context damage, not antecedent-specific)",
           gap_placebo_syn > 0.9, f"gap_placebo={gap_placebo_syn:.4f}")
    report("  DiD ~= 0 EXACTLY (deterministic model: both arms corrupt a position < k, so "
           "both always break the prefix identically) -- THE PLACEBO HAS TEETH",
           abs(did_syn) < 1e-9, f"DiD={did_syn:.6f} (gap_true={gap_true_syn:.4f}, "
           f"gap_placebo={gap_placebo_syn:.4f} -- an un-placebo'd instrument would have "
           f"reported the raw gap_true as 'recall capacity' and been WRONG)")

    # --- [4] finalize_cell REFUSES to emit a verdict-grade number w/o placebo ---
    print("\n  [4] VERDICT-GRADE REFUSAL: finalize_cell must raise if NO candidate has a "
          "resolved placebo read.")
    fake_no_placebo = {"records": [{"row_idx": 0, "k": 5, "delta": 3, "hit_intact": 1,
                                     "hit_true_ablated": 0, "hit_placebo_ablated": None,
                                     "placebo_flagged": True}],
                        "acc_baseline_nonAR": 0.5, "n_baseline": 10, "n_windows": 1}
    raised = False
    try:
        finalize_cell("fake_rung", "fake_corpus", fake_no_placebo)
    except VerdictGradeError:
        raised = True
    report("  finalize_cell raises VerdictGradeError with zero resolved placebos", raised)

    fake_ok = {"records": [{"row_idx": 0, "k": 5, "delta": 3, "hit_intact": 1,
                             "hit_true_ablated": 0, "hit_placebo_ablated": 1,
                             "placebo_flagged": False,
                             # sec 9.1.5 S2 fields -- present, so this cell is S2-complete.
                             "logp_intact": -0.1, "logp_true_ablated": -5.0,
                             "logp_placebo_ablated": -0.3}] * 3,
               "acc_baseline_nonAR": 0.5, "n_baseline": 10, "n_windows": 1}
    cell_ok = finalize_cell("fake_rung", "fake_corpus", fake_ok, n_boot=50,
                             t2b_result={"acc_copy": 0.4, "acc_copy_se": 0.05})
    report("  finalize_cell stamps _verdict_grade=True on a valid cell", cell_ok.get("_verdict_grade") is True)
    report("  finalize_cell's cell is S2-complete when every record carries logp fields",
           cell_ok.get("cell_void_missing_s2_fields") is False)
    report("  finalize_cell merges acc_copy/acc_copy_se from an optional t2b_result",
           cell_ok.get("acc_copy") == 0.4 and cell_ok.get("acc_copy_se") == 0.05)

    # --- sec 9.1.5 S2 teeth: a cell MISSING the log-prob fields on even one resolved
    #     candidate must be refused by compute_capacity_metric (sec 9.6's stop rule: "a
    #     cell missing these fields is VOID"). ---
    fake_missing_s2 = {"records": [{"row_idx": 0, "k": 5, "delta": 3, "hit_intact": 1,
                                     "hit_true_ablated": 0, "hit_placebo_ablated": 1,
                                     "placebo_flagged": False,
                                     "logp_intact": None, "logp_true_ablated": None,
                                     "logp_placebo_ablated": None}] * 3,
                        "acc_baseline_nonAR": 0.5, "n_baseline": 10, "n_windows": 1}
    cell_missing_s2 = finalize_cell("fake_rung", "fake_corpus", fake_missing_s2, n_boot=50)
    report("  finalize_cell correctly flags a cell missing S2 fields (cell_void_missing_s2_fields=True)",
           cell_missing_s2.get("cell_void_missing_s2_fields") is True
           and cell_missing_s2.get("did_logp") is None)
    raised_s2 = False
    try:
        compute_capacity_metric(cell_missing_s2, "raw_did", t2b2_pass=True)
    except VerdictGradeError as e:
        raised_s2 = "S2" in str(e) or "log-prob" in str(e)
    report("  compute_capacity_metric REFUSES a cell missing S2 fields (sec 9.6 stop rule)",
           raised_s2)

    unstamped = dict(cell_ok)
    del unstamped["_verdict_grade"]
    raised2 = False
    try:
        compute_capacity_metric(unstamped, "anything", t2b2_pass=True)
    except VerdictGradeError:
        raised2 = True
    report("  compute_capacity_metric refuses an UNSTAMPED dict (bypass attempt caught)", raised2)

    # AUDIT FIX (2026-07-12) teeth: a cell that FAILED T2b-2 is a VOID rung and must not
    # yield an M(r); before the fix the guard chain checked only (stamp + normalization),
    # so a rung reporting a gap larger than its own copy ability still returned a number.
    raised_t2b2 = False
    try:
        compute_capacity_metric(cell_ok, "anything", t2b2_pass=False)
    except VerdictGradeError as e:
        raised_t2b2 = "T2b-2" in str(e)
    report("  compute_capacity_metric REFUSES a T2b-2-failing (=VOID) cell", raised_t2b2)

    # ... and a cell whose placebo never distribution-matched (sec 9.2: "the cell is VOID")
    void_placebo_cell = dict(cell_ok, cell_void_placebo_match=True)
    raised_pv = False
    try:
        compute_capacity_metric(void_placebo_cell, "anything", t2b2_pass=True)
    except VerdictGradeError as e:
        raised_pv = "placebo not distribution-matched" in str(e)
    report("  compute_capacity_metric REFUSES a placebo-unmatched (=VOID) cell", raised_pv)

    # --- [5] NORMALIZATION: sec 9.1 is now PINNED (raw_did) -- fails loudly when unset /
    #     unregistered for anything else; the mechanism stays generic and pluggable. ---
    raised3 = False
    try:
        compute_capacity_metric(cell_ok, None, t2b2_pass=True)
    except RuntimeError as e:
        raised3 = "NORMALIZATION_UNSET" in str(e)
    report("  compute_capacity_metric raises RuntimeError('NORMALIZATION_UNSET') when normalization=None",
           raised3)
    raised4 = False
    try:
        compute_capacity_metric(cell_ok, "not_registered_yet", t2b2_pass=True)
    except KeyError:
        raised4 = True
    report("  compute_capacity_metric raises KeyError on an unregistered normalization name", raised4)
    report("  NORMALIZATION_REGISTRY carries EXACTLY the sec 9.1 pin ('raw_did'), no other "
           "convenience default", sorted(NORMALIZATION_REGISTRY) == ["raw_did"],
           f"registered={sorted(NORMALIZATION_REGISTRY)}")
    m_raw = compute_capacity_metric(cell_ok, "raw_did", t2b2_pass=True)
    report("  'raw_did' (sec 9.1's pin) computes M(r) == cell['did'] exactly, denominator 1",
           m_raw == cell_ok["did"], f"M(r)={m_raw}, did={cell_ok['did']}")
    register_normalization("_smoke_test_only_identity", lambda cell: cell["did"])
    report("  register_normalization + a registered name WORKS for a hypothetical future metric "
           "(the mechanism stayed generic after the pin landed)",
           compute_capacity_metric(cell_ok, "_smoke_test_only_identity",
                                    t2b2_pass=True) == cell_ok["did"])
    del NORMALIZATION_REGISTRY["_smoke_test_only_identity"]   # leave the registry at just the pin

    # AUDIT FIX teeth (S2 follow-up audit): the PINNED name may not be SILENTLY rebound.
    raised_ovw = False
    try:
        register_normalization("raw_did", lambda cell: 999.0)
    except RuntimeError:
        raised_ovw = True
    report("  register_normalization REFUSES to silently rebind the PINNED 'raw_did' to a "
           "different function, and M(r) still reads the pin afterwards",
           raised_ovw and compute_capacity_metric(cell_ok, "raw_did", t2b2_pass=True) == cell_ok["did"])
    register_normalization("raw_did", _raw_did)   # the SAME function object: idempotent, allowed
    report("  ... but re-registering the IDENTICAL function object is a no-op "
           "(importlib.reload / double-import stay safe)",
           NORMALIZATION_REGISTRY["raw_did"] is _raw_did)

    # --- [5d] sec 9.1.5 S1 -- the utilization ratio DiD/acc_copy, verdict-WITHHOLDING only ---
    print("\n  [5d] sec 9.1.5 S1 (utilization ratio) + S2 disagreement-rule teeth")
    s1 = compute_s1_utilization(cell_ok)
    report("  S1 = did/acc_copy on a normal cell", abs(s1["value"] - cell_ok["did"] / 0.4) < 1e-9, str(s1))
    s1_zero = compute_s1_utilization({"did": 0.1, "acc_copy": 0.0})
    report("  S1 reports None (not ZeroDivisionError/inf) at acc_copy==0",
           s1_zero["value"] is None and s1_zero["reason"] is not None, str(s1_zero))
    s1_missing = compute_s1_utilization({"did": 0.1, "acc_copy": None})
    report("  S1 reports None when acc_copy was never merged into the cell",
           s1_missing["value"] is None, str(s1_missing))

    # --- S2 pre-committed disagreement rule: primary and S2 factor-1 classifications
    #     disagreeing MUST force INDETERMINATE, even when both individually look decisive. ---
    v_agree = compute_verdict_with_s2("DECLINES", "DECLINES", True, 4, 4)
    report("  S2 agrees with primary (both DECLINES) -> normal table result (COUPLED)",
           v_agree == "COUPLED", v_agree)
    v_disagree = compute_verdict_with_s2("RISES", "DECLINES", True, 4, 4)
    report("  S2 DISAGREES with primary (RISES vs DECLINES) -> forced INDETERMINATE "
           "(never creates/strengthens a verdict, only withholds)",
           v_disagree == "INDETERMINATE", v_disagree)
    v_disagree2 = compute_verdict_with_s2("FLAT", "RISES", True, 4, 4)
    report("  S2 disagreement fires even when BOTH individually look like a clean, "
           "differently-decisive verdict", v_disagree2 == "INDETERMINATE", v_disagree2)

    # --- stat_did_logp / has_complete_s2_fields unit checks ---
    logp_records = [{"logp_placebo_ablated": -0.2, "logp_true_ablated": -3.0},
                     {"logp_placebo_ablated": -0.5, "logp_true_ablated": -4.0}]
    expected_didlogp = ((-0.2 - -3.0) + (-0.5 - -4.0)) / 2
    report("  stat_did_logp computes E[logp_placebo_ablated - logp_true_ablated] correctly",
           abs(stat_did_logp(logp_records) - expected_didlogp) < 1e-9,
           f"got={stat_did_logp(logp_records)}, expected={expected_didlogp}")
    report("  has_complete_s2_fields(True case)",
           has_complete_s2_fields([{"logp_intact": -0.1, "logp_true_ablated": -1.0,
                                     "logp_placebo_ablated": -0.2}]))
    report("  has_complete_s2_fields(False case: one None)",
           not has_complete_s2_fields([{"logp_intact": -0.1, "logp_true_ablated": None,
                                         "logp_placebo_ablated": -0.2}]))

    # --- [5b] AUDIT FIX teeth: the placebo-match VOID must reach the VERDICT as VOID,
    #     not as FLOOR (VOID = HALT the wave; FLOOR = quietly drop the rung and fit
    #     through the rest -- opposite consequences from the same failure). ---
    pv_cells = {c: dict(cell_ok, cell_void_placebo_match=True, n_candidates_resolved=9999)
                for c in ("wikitext-mix-ext", "openr1-mix-ext")}
    adm_pv = check_rung_admissible(pv_cells, tok_per_param=23.0, checkpoint_quiesced=True,
                                    t2b2_by_corpus={c: {"passes": True} for c in pv_cells},
                                    t2b1_by_corpus={c: {"passes": True} for c in pv_cells},
                                    min_candidates=1)
    report("  check_rung_admissible maps a placebo-match failure to VOID (not FLOOR)",
           adm_pv["verdict_if_not_admissible"] == "VOID" and not adm_pv["admissible"],
           str(adm_pv["verdict_if_not_admissible"]))

    # --- [5c] AUDIT FIX teeth: tost_flat must return the PINNED 90% CI (5th..95th
    #     percentile), not the 80% CI the previous body returned. ---
    unit_boots = [i / 1000.0 for i in range(1000)]
    tf = tost_flat(unit_boots, delta=1.0, alpha=0.10)
    report("  tost_flat returns the pinned 90% CI (5th..95th pct), not an 80% CI",
           abs(tf["ci90"][0] - 0.05) < 2e-3 and abs(tf["ci90"][1] - 0.949) < 2e-3,
           f"ci90={tf['ci90']} (an 80% CI would read [0.100, 0.899] and would declare "
           f"FLAT more often than the pre-registration permits)")

    # --- [6] T2b-2 CEILING CHECK: fires on a violating case shaped like the VOID
    #     build's own wikitext defect (nonzero DiD reported at acc_copy=0), does
    #     NOT fire on a compliant case. Synthetic values -- the VOID build's real
    #     figures are QUARANTINED, see QUARANTINE_r0_did_values.md. ---
    print("\n  [6] T2b-2 CEILING CHECK teeth: DiD <= acc_copy + 2*SE")
    violating = check_t2b2_ceiling(did_value=0.15, acc_copy=0.0000, acc_copy_se=0.01)
    report("  FIRES (passes=False) on a VOID-build-shaped case "
           "(nonzero DiD at acc_copy=0, regate 10.2 -- see QUARANTINE_r0_did_values.md "
           "for the retired build's real figures)", violating["passes"] is False, str(violating))
    compliant = check_t2b2_ceiling(did_value=0.02, acc_copy=0.30, acc_copy_se=0.02)
    report("  does NOT fire on a compliant case (DiD well under the copy-ability ceiling)",
           compliant["passes"] is True, str(compliant))
    boundary = check_t2b2_ceiling(did_value=0.05, acc_copy=0.05, acc_copy_se=0.0)
    report("  boundary case (DiD == bound exactly) reads passes=True (<=, not <)",
           boundary["passes"] is True, str(boundary))

    # --- [7] T2b-1 exact paired sign test: fires on an asymmetric case, does not
    #     on a symmetric/no-signal case. ---
    print("\n  [7] T2b-1 mechanism-exists paired test")
    asymmetric = [{"hit_placebo_ablated": 1, "hit_true_ablated": 0} for _ in range(20)] + \
                 [{"hit_placebo_ablated": 0, "hit_true_ablated": 1} for _ in range(1)]
    r_asym = check_t2b1_mechanism_exists(asymmetric)
    report("  fires (passes=True, p<0.001) on a strongly asymmetric 20-vs-1 split", r_asym["passes"],
           str(r_asym))
    symmetric = [{"hit_placebo_ablated": 1, "hit_true_ablated": 0} for _ in range(5)] + \
                [{"hit_placebo_ablated": 0, "hit_true_ablated": 1} for _ in range(5)]
    r_sym = check_t2b1_mechanism_exists(symmetric)
    report("  does NOT fire (passes=False) on a perfectly symmetric 5-vs-5 split", not r_sym["passes"],
           str(r_sym))

    # --- [8] QUIESCENCE / md5-pin: catches a checkpoint being actively written,
    #     clears once the writer stops. ---
    print("\n  [8] CHECKPOINT QUIESCENCE (partially-written checkpoint detection)")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fake_ckpt.pt")
        with open(path, "wb") as f:
            f.write(b"x" * 1000)
        stop_event = threading.Event()

        def _writer():
            while not stop_event.is_set():
                with open(path, "ab") as f:
                    f.write(b"y")
                time.sleep(0.03)

        th = threading.Thread(target=_writer, daemon=True)
        th.start()
        time.sleep(0.15)
        q_live = verify_quiesced_checkpoint(path, wait_s=0.25)
        stop_event.set()
        th.join(timeout=2)
        time.sleep(0.05)
        q_done = verify_quiesced_checkpoint(path, wait_s=0.25)
    report("  a LIVE writer is caught (quiesced=False) while it is still appending",
           q_live["quiesced"] is False, f"size {q_live['size0']}->{q_live['size1']}")
    report("  the SAME file reads quiesced=True once the writer has stopped",
           q_done["quiesced"] is True and q_done["md5"] is not None)

    raised5 = False
    with tempfile.TemporaryDirectory() as td:
        path2 = os.path.join(td, "growing.pt")
        with open(path2, "wb") as f:
            f.write(b"a" * 100)
        stop2 = threading.Event()

        def _writer2():
            while not stop2.is_set():
                with open(path2, "ab") as f:
                    f.write(b"b")
                time.sleep(0.02)

        th2 = threading.Thread(target=_writer2, daemon=True)
        th2.start()
        time.sleep(0.1)
        try:
            # load_checkpoint will fail at torch.load AFTER the quiescence check passes/fails --
            # here we only need the quiescence gate itself to raise before ever touching torch.load
            q = verify_quiesced_checkpoint(path2, wait_s=0.2)
            if not q["quiesced"]:
                raise CheckpointNotQuiescedError("simulated load_checkpoint refusal")
        except CheckpointNotQuiescedError:
            raised5 = True
        stop2.set()
        th2.join(timeout=2)
    report("  load_checkpoint's refusal path (CheckpointNotQuiescedError) fires on a live writer",
           raised5)

    # --- [9] Verdict-map classification sanity (pure arithmetic, no model needed) ---
    print("\n  [9] Verdict-map (sec 9.5) classification sanity")
    report("  beta CI entirely > 0 -> RISES",
           classify_recall_trend([0.05, 0.20], {"tost_pass": False}) == "RISES")
    report("  beta CI entirely < 0 -> DECLINES",
           classify_recall_trend([-0.20, -0.05], {"tost_pass": False}) == "DECLINES")
    report("  beta CI includes 0 + TOST passes -> FLAT",
           classify_recall_trend([-0.01, 0.01], {"tost_pass": True}) == "FLAT")
    report("  beta CI includes 0 + TOST fails -> INDETERMINATE",
           classify_recall_trend([-0.5, 0.5], {"tost_pass": False}) == "INDETERMINATE")
    report("  compute_verdict: DECLINES + coupling licensed -> COUPLED",
           compute_verdict("DECLINES", True, 4, 4, factor1_trend_s2="DECLINES") == "COUPLED")
    report("  compute_verdict: RISES + coupling licensed -> DECOUPLED",
           compute_verdict("RISES", True, 4, 4, factor1_trend_s2="RISES") == "DECOUPLED")
    report("  compute_verdict: FLAT + coupling licensed -> FLAT-COUPLED",
           compute_verdict("FLAT", True, 4, 4, factor1_trend_s2="FLAT") == "FLAT-COUPLED")
    report("  compute_verdict: trend present but coupling NOT licensed -> RECALL-TREND-ONLY",
           compute_verdict("RISES", False, 4, 4, factor1_trend_s2="RISES") == "RECALL-TREND-ONLY")
    report("  compute_verdict: fewer than 3 admissible rungs -> FLOOR",
           compute_verdict("RISES", True, 2, 2, factor1_trend_s2="RISES") == "FLOOR")

    # --- [9b] AUDIT FIX teeth (S2 follow-up audit): the sec 9.1.5 disagreement rule must be
    #     STRUCTURAL, not a convention. `compute_verdict` -- the function sec 9.5's prose
    #     names, and the one a driver author would reach for first -- must be INCAPABLE of
    #     returning a verdict without an S2 disposition. THE NEGATIVE TEST: the old
    #     4-positional-argument call must now FAIL LOUDLY, not quietly hand back "DECOUPLED".
    stale_caller_raised = False
    try:
        compute_verdict("RISES", True, 4, 4)     # the pre-fix signature: S2 silently skipped
    except TypeError:
        stale_caller_raised = True
    report("  compute_verdict REFUSES a stale S2-less call (TypeError) -- the sec 9.1.5 rule "
           "cannot be bypassed by calling the inner function directly", stale_caller_raised)
    # ... and the full 4x4 grid: NO disagreement may reach a substantive verdict, via EITHER
    #     entry point (the wrapper and the inner function must agree on all 16).
    _trends = ["RISES", "DECLINES", "FLAT", "INDETERMINATE"]
    _substantive = {"COUPLED", "DECOUPLED", "FLAT-COUPLED", "RECALL-TREND-ONLY"}
    _leaks, _mismatch = [], []
    for _p in _trends:
        for _s in _trends:
            for _lic in (True, False):
                _w = compute_verdict_with_s2(_p, _s, _lic, 4, 4)
                _d = compute_verdict(_p, _lic, 4, 4, factor1_trend_s2=_s)
                if _w != _d:
                    _mismatch.append((_p, _s, _lic, _w, _d))
                if _p != _s and _w in _substantive:
                    _leaks.append((_p, _s, _lic, _w))
    report("  FULL 4x4 grid (x2 licensing = 32 calls): no primary!=S2 combination reaches ANY "
           "substantive verdict, and the wrapper and the inner function agree on all 32",
           not _leaks and not _mismatch, f"leaks={_leaks} mismatches={_mismatch}")

    # --- [10] END-TO-END through the REAL DeltaNetLM class (stub or real fla,
    #     tiny toy vocab, untrained) -- exercises the full run_did_eval plumbing
    #     against the actual model API/shapes, matching this repo's own
    #     smoke-gate convention (CPU-stub tests LOGIC only; see [11] for the
    #     narrow real-kernel note). ---
    print("\n  [10] END-TO-END plumbing through DeltaNetLM (toy vocab, untrained)")
    torch.manual_seed(0)
    V10 = 200
    seq_len10 = 160  # >= _MIN_KERNEL_T floor per this repo's own convention
    tiny = DeltaNetLM(V10, d_model=64, d_state=64, n_layers=1, conv_size=4).to(device)
    tiny.eval()
    base10 = torch.randint(12, V10, (seq_len10 + 1,))
    base10[5], base10[6] = 10, 11
    base10[140], base10[141] = 10, 11   # genuine repeat, delta=134
    val_tokens10 = base10.repeat(20).to(device)   # a tiny "corpus" get_batch can sample windows from
    mode_next10 = [-1] * V10
    mode_next_t = torch.tensor(mode_next10)
    try:
        result10 = run_did_eval(tiny, val_tokens10, batch_size=4, seq_len=seq_len10, n_windows=4,
                                 device=device, mode_next=mode_next_t, seed=123, c_max=4,
                                 eval_micro_batch=8, min_sep=2, vocab_size=V10)
        report("  run_did_eval completes end-to-end on the real DeltaNetLM class",
               True, f"n_candidates={len(result10['records'])}")
        # sec 9.1.5 S2: every record from a REAL forward pass must carry real (non-None,
        # finite, <=0) log-probs in all three arms -- not just the hand-built smoke fixtures.
        resolved10 = [r for r in result10["records"] if r["hit_placebo_ablated"] is not None]
        s2_ok_real = resolved10 and all(
            r.get("logp_intact") is not None and r["logp_intact"] <= 1e-6
            and r.get("logp_true_ablated") is not None and r["logp_true_ablated"] <= 1e-6
            and r.get("logp_placebo_ablated") is not None and r["logp_placebo_ablated"] <= 1e-6
            for r in resolved10
        )
        report("  S2 log-prob fields populate from the REAL forward pass (log p(target) <= 0 "
               "in all three arms, every resolved candidate)", bool(s2_ok_real) if resolved10 else True,
               f"n_resolved={len(resolved10)}"
               + (f", sample logp_intact={resolved10[0].get('logp_intact')}" if resolved10 else ""))
        if result10["records"]:
            cell10 = finalize_cell("toy_rung", "toy_corpus", result10, n_boot=50)
            report("  finalize_cell produces a verdict-grade cell from a real forward pass",
                   cell10.get("_verdict_grade") is True,
                   f"DiD={cell10['did']:.4f} n_resolved={cell10['n_candidates_resolved']}")
            report("  the real-pass cell is S2-complete and did_logp is a real number",
                   cell10.get("cell_void_missing_s2_fields") is False and cell10.get("did_logp") is not None,
                   f"did_logp={cell10.get('did_logp')}")
        else:
            report("  (no candidates found in this tiny random corpus sample -- plumbing still "
                   "ran without error; not a failure of the instrument)", True)
    except Exception as e:  # noqa: BLE001 -- smoke gate: report, don't hide
        report("  run_did_eval end-to-end pass", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [10b] T2b end-to-end: run_t2_planted_copy + check_t2b1/T2b2 against the
    #     SAME real DeltaNetLM class (untrained -- plumbing, not a competence claim). ---
    print("\n  [10b] T2b END-TO-END: planted-copy DiD pipeline through DeltaNetLM")
    try:
        delta_pool10 = [10, 20, 40, 80]
        t2_result = run_t2_planted_copy(tiny, val_tokens10, batch_size=4, seq_len=seq_len10,
                                         device=device, seed=321, delta_pool=delta_pool10,
                                         tok_a=13, tok_b=14, n_plants=12, vocab_size=V10,
                                         eval_micro_batch=8)
        report("  run_t2_planted_copy completes end-to-end and returns one record per plant",
               t2_result["n_plants"] == 12, f"n_plants={t2_result['n_plants']}")
        t2b1 = check_t2b1_mechanism_exists(t2_result["records"])
        t2b2 = check_t2b2_ceiling(did_value=0.0, acc_copy=t2_result["acc_copy"] or 0.0,
                                   acc_copy_se=t2_result["acc_copy_se"] or 0.0)
        report("  check_t2b1_mechanism_exists / check_t2b2_ceiling run on REAL-pipeline output "
               "without error", True,
               f"acc_copy={t2_result['acc_copy']}, t2b1_p={t2b1['p_value']:.4g}, t2b2={t2b2['passes']}")
    except Exception as e:  # noqa: BLE001
        report("  T2b end-to-end pipeline", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [10c] AUDIT FIX teeth (F-4, second guise): the CANDIDATE POPULATION must not
    #     depend on --batch-size. The design's rungs use DIFFERENT token-arithmetic batch
    #     sizes (three at 32, the 1.31B at 16); if batch_size reaches the window sampler,
    #     the top rung is measured against a different candidate population than the rest
    #     -- which IS F-4. NOTE: this test is TOOTHLESS ON CPU (MT19937 consumes its stream
    #     sequentially, so chunked draws happen to agree) and only bites on CUDA, where
    #     Philox advances its offset PER KERNEL LAUNCH. Measured on the H100 before the fix:
    #     batch_size 16 vs 32 at the same seed shared only 50% of their 512 windows. Run the
    #     smoke with --device cuda to give this test its teeth.
    print("\n  [10c] BATCH-SIZE INDEPENDENCE of the candidate population (F-4, second guise)")
    try:
        rA = run_did_eval(tiny, val_tokens10, batch_size=4, seq_len=seq_len10, n_windows=8,
                          device=device, mode_next=mode_next_t, seed=777, c_max=4,
                          eval_micro_batch=8, min_sep=2, vocab_size=V10)
        rB = run_did_eval(tiny, val_tokens10, batch_size=7, seq_len=seq_len10, n_windows=8,
                          device=device, mode_next=mode_next_t, seed=777, c_max=4,
                          eval_micro_batch=8, min_sep=2, vocab_size=V10)
        keyA = [(r["row_idx"], r["k"], r["delta"]) for r in rA["records"]]
        keyB = [(r["row_idx"], r["k"], r["delta"]) for r in rB["records"]]
        report("  same seed, batch_size 4 vs 7 -> IDENTICAL candidate population",
               keyA == keyB and rA["n_windows"] == rB["n_windows"],
               f"n_cand {len(keyA)} vs {len(keyB)}"
               + ("" if device.startswith("cuda")
                  else "  [NOTE: on CPU this test is weak -- it only bites on CUDA]"))
    except Exception as e:  # noqa: BLE001
        report("  batch-size-independence check", False, f"EXCEPTION: {type(e).__name__}: {e}")

    print("\n  [11] NOTE (not a gate): T2a (the SAME T2b machinery pointed at reference models "
          "RWKV7-Goose-1.5B / falcon-mamba-7b) and a full real-checkpoint R0 read are wired but "
          "were NOT executed in this build session -- disclosed scope boundary, see the build "
          "report.")

    print("\n" + "=" * 70)
    print(f"  SMOKE {'PASSED' if ok_all else 'FAILED'}")
    print("=" * 70)
    return 0 if ok_all else 1


# =============================================================================
# CLI
# =============================================================================
def mode_resolve_slice(args) -> int:
    r = resolve_token_matched_checkpoint(args.ckpt_dir, args.batch_size, args.seq_len,
                                          args.target_tokens, args.prefix)
    print(json.dumps(r, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
    return 0


def mode_verify_quiesced(args) -> int:
    r = verify_quiesced_checkpoint(args.checkpoint, wait_s=args.wait_s)
    print(json.dumps(r, indent=2))
    return 0 if r["quiesced"] else 1


def mode_run(args) -> int:
    """Real single-cell TEETH-GATE run (DiD/T1/baseline; NOT a verdict --
    sec 9.1 is not pinned). Requires GPU + real fla. Intentionally NOT
    invoked against any real checkpoint by this build session (per this
    task's explicit instruction) -- wired and ready."""
    device = args.device
    # sec 9.6 item 3 has TWO clauses; file-quiescence is only the first (see
    # verify_quiesced_checkpoint's docstring: a live --ckpt-every-10000 writer leaves a
    # byte-stable file for HOURS between writes, which is precisely the S-7 read). The
    # second clause -- "its job must be terminated" -- is not measurable from the file, so
    # it is an explicit, recorded operator attestation. AUDIT FIX (2026-07-12).
    if not args.attest_job_terminated:
        raise SystemExit(
            "REFUSING TO RUN: --attest-job-terminated is required.\n"
            "PARAM_AXIS_SCALING_DESIGN.md sec 9.6 item 3: 'No rung may be read from a live, "
            "still-training job ... and its job must be terminated.' The file-quiescence check "
            "CANNOT establish this: the S-7 precedent (regate sec 10.3) read a live 1.31B job's "
            "step-40000 checkpoint under a --ckpt-every-10000 writer, and at ~1.4 s/step those "
            "writes are ~3.9 hours apart -- the file was perfectly byte-stable the whole time.\n"
            "Confirm the writing job is dead (e.g. `tmux ls`, `nvidia-smi`, the queue's own "
            "completed/ marker), then re-run with --attest-job-terminated. The attestation is "
            "recorded in the result JSON as `job_terminated_attested`."
        )
    # AUDIT FIX (2026-07-12, S2 follow-up audit): `--normalization` was accepted by the parser
    # and then never read by anything -- an operator who passed `--normalization raw_did` (or,
    # worse, some other name) got it SILENTLY IGNORED and a JSON that looks like it honoured it.
    # mode_run cannot compute M(r) at all (see the --compute-verdict refusal below), so the flag
    # has no meaning here; it is now refused out loud rather than swallowed.
    if args.normalization is not None:
        raise SystemExit(
            f"--normalization={args.normalization!r} is REFUSED by this entry point. `mode_run` "
            f"computes DiD / T1a / DiD_logp for ONE (checkpoint, corpus) cell and NEVER applies a "
            f"normalization (M(r) requires T2b-1/T2b-2 + the sec 9.6 admissibility gates, which "
            f"this entry point does not run -- see the --compute-verdict refusal). sec 9.1's pin "
            f"('raw_did') is applied by the multi-rung driver, via compute_capacity_metric. "
            f"Passing it here would have been silently ignored; that is worse than an error."
        )
    model, ckpt, md5_pin = load_checkpoint(args.checkpoint, device, require_quiesced=not args.no_quiesce_check,
                                            wait_s=args.wait_s)
    n_params = sum(p.numel() for p in model.parameters())
    train_tokens, val_tokens, meta, _, _ = load_corpus(args.data_dir, args.corpus, device)
    mode_next = build_bigram_mode_table(train_tokens, VOCAB_SIZE, device)
    seed = corpus_fixed_seed(args.corpus) + 424242
    result = run_did_eval(model, val_tokens, args.batch_size, args.seq_len, args.n_windows, device,
                           mode_next, seed, c_max=args.c_max, eval_micro_batch=args.eval_micro_batch,
                           min_sep=args.min_sep)
    cell = finalize_cell(args.rung or "unlabeled", args.corpus, result)
    cell["checkpoint"] = args.checkpoint
    cell["checkpoint_md5"] = md5_pin
    cell["n_params"] = n_params
    cell["ckpt_step"] = ckpt.get("step")
    cell["job_terminated_attested"] = True
    cell["quiesce_check_skipped"] = bool(args.no_quiesce_check)
    # sec 9.1 IS pinned (raw_did, PARAM_AXIS_SCALING_DESIGN.md sec 9.1, commit 7381d2a) --
    # this field is a fixed provenance fact now, not a runtime toggle. `mode_run` STILL
    # cannot emit a verdict (see the --compute-verdict refusal below): the missing piece
    # is no longer the normalization, it is that this entry point runs neither T2b nor
    # sec 9.6 admissibility.
    cell["sec91_normalization_pinned"] = True
    cell["sec91_normalization_name"] = "raw_did"
    print(json.dumps({k: v for k, v in cell.items() if k != "per_row_did"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(cell, f, indent=2)

    # sec 9.6 item 7's sample-size floor, checked LOUDLY at the point of production.
    n_res = cell["n_candidates_resolved"]
    if n_res < 4096:
        print(f"\n  *** WARNING: n_candidates_resolved={n_res} < 4096, the sec 9.6 item 7 floor. "
              f"This rung is INADMISSIBLE as it stands.\n"
              f"      Note the pinned sampling (N_rows={N_ROWS_DEFAULT} x C_max={C_MAX_DEFAULT}) has a "
              f"MAXIMUM of {N_ROWS_DEFAULT * C_MAX_DEFAULT} candidates, so the floor is only reachable if "
              f"EVERY row yields >= {C_MAX_DEFAULT} candidates AND every placebo resolves.\n"
              f"      Raise --n-windows (and use the SAME value at every rung -- an n_windows that "
              f"varies by rung reintroduces F-4).", flush=True)

    if args.compute_verdict:
        raise SystemExit(
            "--compute-verdict is REFUSED by this entry point (AUDIT FIX 2026-07-12; STILL "
            "REFUSED after sec 9.1's pin landed -- the reason changed, the refusal did not).\n"
            "sec 9.1's normalization IS pinned (raw_did) -- that is no longer the blocker. "
            "M(r) may still not be computed from a cell that has not been through T2b-1/T2b-2 "
            "and the sec 9.6 admissibility gates, and `mode_run` runs NEITHER (it computes "
            "DiD/T1a/DiD_logp for ONE (checkpoint, corpus) cell only). "
            "compute_capacity_metric() REQUIRES the caller to pass this cell's T2b-2 disposition "
            "(`t2b2_pass=`), which mode_run has no way to supply on its own.\n"
            "A verdict needs a driver (not yet built) that runs run_t2_planted_copy -> "
            "check_t2b1_mechanism_exists / check_t2b2_ceiling -> check_rung_admissible across "
            "BOTH corpora and ALL rungs, and only then calls compute_capacity_metric with "
            "normalization='raw_did'. That driver is the gap sec 9.1's pin unblocks, not this "
            "single-cell entry point."
        )
    print("\n  NOTE: DiD / T1a / DiD_logp for ONE cell only. NOT a verdict -- T2b-1/T2b-2 and "
          "sec 9.6 admissibility are not run by this entry point (see the --compute-verdict "
          "refusal above). sec 9.1's normalization ('raw_did') is pinned, but per sec 9.6's stop "
          "rule this is still a single eval-only pass with no re-read: do not compute a verdict "
          "from this JSON piecemeal outside the proper multi-rung/multi-corpus driver.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resolve-slice", action="store_true")
    ap.add_argument("--verify-quiesced", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--checkpoint", type=str)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--corpus", type=str, choices=sorted(CORPUS_DIRS))
    ap.add_argument("--rung", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-windows", type=int, default=N_ROWS_DEFAULT)
    ap.add_argument("--c-max", type=int, default=C_MAX_DEFAULT)
    ap.add_argument("--eval-micro-batch", type=int, default=EVAL_MICRO_BATCH_DEFAULT)
    ap.add_argument("--min-sep", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--ckpt-dir", type=str)
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS_FORCED_SLICE)
    ap.add_argument("--prefix", type=str)
    ap.add_argument("--wait-s", type=float, default=5.0)
    ap.add_argument("--no-quiesce-check", action="store_true",
                     help="DANGEROUS, testing only -- skips sec 9.6 item 3's live-writer guard. "
                          "Stamps quiesce_check_skipped=true into the result JSON and leaves the "
                          "sec-9.6-mandated md5 pin NULL, which alone makes the cell non-conformant.")
    ap.add_argument("--attest-job-terminated", action="store_true",
                     help="REQUIRED for --run. Operator attestation of sec 9.6 item 3's second "
                          "clause ('its job must be terminated'), which file-quiescence cannot "
                          "establish -- a --ckpt-every-10000 writer leaves a byte-stable file for "
                          "hours between writes (the S-7 read). Recorded in the result JSON.")
    ap.add_argument("--compute-verdict", action="store_true")
    ap.add_argument("--normalization", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        return smoke(args.device if torch.cuda.is_available() else "cpu")
    if args.resolve_slice:
        return mode_resolve_slice(args)
    if args.verify_quiesced:
        assert args.checkpoint, "--checkpoint required for --verify-quiesced"
        return mode_verify_quiesced(args)
    if args.run:
        assert args.checkpoint and args.corpus, "--checkpoint and --corpus required for --run"
        return mode_run(args)
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
