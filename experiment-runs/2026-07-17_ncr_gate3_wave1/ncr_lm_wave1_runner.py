"""NCR REAL-LM WAVE-1 CALIBRATION RUNNER -- sec G3-B6 (build agent, 2026-07-17).
matrix-thinking/NCR_REAL_LM_DESIGN.md sec G3-B5 (COORDINATOR ADJUDICATION of
the sec G3-B4 audit): implements the TWO-ARM design + frozen attribution rule
that closes DEFECT G3-B4-1 [FATAL] (the ratified graft's read is additive/
non-bottlenecked, so a FAIL cannot be localized to the NCR head without a
backbone-only control). This is a REAL TRAINING DRIVER (step loop, LR
schedule, checkpoint/resume, ceiling-gated, blind), not a smoke -- it WRAPS
the already-audited graft (sec G3-B4 dimension 1: PASS, no defects in the
wiring itself) without reopening any of its ratified logic.

============================================================================
WHAT THIS FILE DOES NOT DO (scope discipline, per the build brief: "build
EXACTLY [sec 6.2's small Phase-1 calibration cell] spec, do not invent a
full sweep"):
============================================================================
  - No real+synthetic batch-level mixing (sec 5.2 Option 1) -- sec G3-B3's
    own build already deferred this ("out of the BUILD+SMOKE... scope"); this
    wave inherits that deferral rather than silently expanding scope. Every
    document trained/evaluated here is a pure grammar_rd synthetic Task-1
    document (disclosed, not fabricated as literal design text).
  - Task 1 / abelian construction ONLY (K=24, d_ncr=25, the audited Wave-1
    pin) -- Task 2 (non-solvable-group word problem, sec 3.2) is NOT built
    here; it is gated on the SEPARATE bridge cell (sec 6.2 Phase 0b), per
    the coordinator's own task scope.
  - The "flat-vector ablation" arm from sec 6.2's ORIGINAL Phase-1 spec is
    NOT one of this cell's two arms. Sec G3-B5 REPLACES that comparison for
    THIS calibration with (full-graft, backbone-only/read-ablated) --
    the FATAL's own required control, not sec 6.2's original arm list.
  - The teacher-force-operator / mlp-adapter / mlp-logits ablation arms
    (sec G3-B2's OTHER pre-wired flags, already smoke-verified in
    ncr_lm_wave1_smoke.py items 10-11) are FAIL-diagnosis tools for AFTER a
    result is read, not part of this calibration's own two arms.
  - This runner does NOT compute or print a WIN/PARTIAL/NULL/FAIL verdict
    anywhere (blind discipline, matching NCR_ORTHO_WRITE.md sec 6's own
    "the runner emits NO verdict" convention) -- see the BLIND DISCIPLINE
    note below.

============================================================================
TWO-ARM WIRING (sec G3-B5's own required construction)
============================================================================
Both arms use the IDENTICAL architecture (backbone + ncr_head + integ, same
param counts, BIT-IDENTICAL initial weights -- see build_two_arms() below)
and train on the EXACT SAME per-step batch (one shared data generator,
consumed once per step, its output batch object reused for both arms'
forward/backward/opt-step calls) -- "same data, same seed, same everything
else" per sec G3-B5. The ONLY difference:
  - full_graft:      o_injected = o_raw   (the real NCR read, unmodified)
  - backbone_only:    o_injected = torch.zeros_like(o_raw)   (read-ablated)
`torch.zeros_like` creates a FRESH tensor with no autograd edge back to
o_raw's own computation graph -- so in the backbone_only arm, NO gradient
reaches the write adapters or the NCR encoder at all (they stay frozen at
their shared random init for the entire run); the read literally
contributes ZERO to backbone_only's logits, verified by an EXACT (not
tolerance-based) equality assertion, both before AND after training
(assert_read_ablation_is_exact_zero, called from run_two_arm_cell). This is
the "wiring already supports it trivially" property sec G3-B5 names -- no
edit to ncr_lm_wave1_smoke.py's own NCRIntegration/ncr_lm_forward was
needed or made; this file composes the SAME audited public building blocks
(integ.extract_kv / integ.query_key / ncr_head.encode / nm.binexp_read /
integ.inject_and_logits_last) into its own read-ablatable wrapper,
ncr_lm_forward_ablatable(), leaving the audited ncr_lm_forward() untouched.

============================================================================
ATTRIBUTION FIELDS (registered in the results JSON, sec G3-B5's own
required registration "so the blind assessor applies it mechanically")
============================================================================
Two DISTINCT metrics are recorded per arm, per depth (a genuine build-time
interpretation of "recovered_frac@0.9 ... for BOTH arms", disclosed here,
not silently invented):
  (a) recovered_frac@0.9 -- the AUDITED ncr_lm_wave1_smoke.recovered_frac_at_09
      metric, computed identically for both arms on o_raw (the NCR
      pathway's own internal read, BEFORE the ablation override). For
      full_graft, o_raw IS the injected read (real gradient reaches it).
      For backbone_only, o_raw's write/encode pathway receives NO gradient
      (see above) -- so this number tracks an UNTRAINED, frozen-at-init
      NCR head's read quality throughout the run, a genuine null baseline
      for the recovery metric itself. THE PRIMARY SIGNAL (sec G3-B5:
      "recovery GAP (full-graft - backbone-only), at DEEP composition
      depth") is full_graft.recovered_frac@0.9(o_raw) minus
      backbone_only.recovered_frac@0.9(o_raw), reported at every deep-ladder
      depth AND as a single deepest-rung number (h=61).
  (b) answer_accuracy -- argmax(logits) == answer_token, computed on each
      arm's ACTUAL logits (i.e. backbone_only's logits use o_injected=0, so
      this measures whether h_q ALONE solves the task). This is what sec
      G3-B4's own attribution-rule PROSE literally names ("its answer
      accuracy is materially below the full graft's -- the NCR read is
      demonstrably load-bearing") -- the blind assessor uses THIS field,
      not (a), to evaluate the attribution rule's own precondition.
Both fields are written for in-distribution h in {1,2,3} AND the deep
ladder h in {5,12,20,29,40,61} (sec 3.1's own eval grid, re-verified sound
for K=24 by _assert_ladder_sound below -- sec 3.1's own residue check was
run at K in {15,32}, not 24, so this is NOT assumed transferred).
The frozen sec G3-B5 rule TEXT is written verbatim into the JSON
(attribution.frozen_rule_text) so a blind assessor applies it against these
raw numbers without needing this script or the design doc open side by side.

============================================================================
BLIND DISCIPLINE (sec G3-B5 build brief: "runner writes results, does not
print metric values to the launcher")
============================================================================
Precedent: NCR_ORTHO_WRITE.md sec 6 / ncr_ortho_fallback_stage0_v3.py's own
run_disc_cell/run_primary_cell -- both print per-step TRAIN LOSS to stdout
(operational liveness/divergence telemetry) but NEVER print an eval metric
(recovered_frac, cosine, accuracy) to stdout; those go ONLY into the
results JSON, unaccompanied by any verdict. This file follows the identical
split: step/loss/elapsed/status ARE printed (needed to confirm the run is
alive and not diverging, and to read Phase-0's own throughput numbers,
which are NOT part of the interpretive signal); recovered_frac@0.9,
answer_accuracy, and the attribution GAP are NEVER printed -- only written
to --out. No PASS/FAIL/WIN/PARTIAL/NULL string appears anywhere in this
file.

Run (box only -- chunk_delta_rule has no CPU path):
  python3 ncr_lm_wave1_runner.py --mode phase0-timing --device cuda \
      --out results/phase0_timing.json
  python3 ncr_lm_wave1_runner.py --mode calibration --device cuda \
      --steps 20000 --ceiling-gpuh <FROM PHASE-0, contended-priced> \
      --ckpt-dir results/ckpts --out results/wave1_calib_K24_s0.json
  python3 ncr_lm_wave1_runner.py --mode smoke --device cuda \
      --out results/runner_smoke.json   # this build's OWN required smoke
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The AUDITED graft module (sec G3-B4 dimension 1: PASS, no wiring defects),
# reused as a LIBRARY -- every constant/builder/function below is imported,
# never reimplemented, except the smoke-5 fix (disclosed in that file's own
# diff) and this file's own NEW read-ablation wrapper (composed from the
# audited public building blocks, see module docstring above).
import ncr_lm_wave1_smoke as graft
import ncr_models as nm                      # noqa: E402 (verbatim; binexp_read)
import grammar_rd as gr                       # noqa: E402 (verbatim; sample_batch_rd via graft helpers)
from lm_pretrain_rd import get_lr             # noqa: E402 (verbatim; the ONE reusable trainer utility --
                                                # not lm_pretrain_rd.py's own train() loop, which is
                                                # coupled to real-corpus doc_offsets/geo3 diagnostics
                                                # this synthetic-only two-arm cell does not use)

VOCAB_SIZE = graft.VOCAB_SIZE
RUNG1_BACKBONE = graft.RUNG1_BACKBONE
K_NCR, D_NCR, H_NCR = graft.K_NCR, graft.D_NCR, graft.H_NCR
NCRIntegration = graft.NCRIntegration
build_backbone = graft.build_backbone
build_ncr_head = graft.build_ncr_head
build_grammar_pools_and_cfg = graft.build_grammar_pools_and_cfg
build_task1_document = graft.build_task1_document
recovered_frac_at_09 = graft.recovered_frac_at_09

RUNNER_TAG = "ncr_gate3_wave1_runner_v1"          # sec G3-B6
TRAIN_HOPS = (1, 2, 3)                             # sec 3.1 Task-1 train range, verbatim
DEEP_LADDER = (5, 12, 20, 29, 40, 61)              # sec 3.1's eval ladder (NCR_ORTHO_WRITE.md sec 3), reused verbatim
CONTENDED_MULTIPLIER = 3.3                         # sec G3-B1 item 2 / sec G3-B3 launch-command note, established precedent
LOG_EVERY = 25

ATTRIBUTION_RULE_TEXT = (
    "sec G3-B5 (COORDINATOR ADJUDICATION of the sec G3-B4 audit, 2026-07-17): "
    "a Phase-1 Gate-0 in-distribution-recovery FAIL may be attributed to \"the "
    "NCR head can't train through a real LM\" ONLY IF the backbone-only arm "
    "does NOT itself solve the task in-distribution (i.e., its answer "
    "accuracy is materially below the full graft's -- the NCR read is "
    "demonstrably load-bearing). If backbone-only already solves it, the "
    "calibration is UNINTERPRETABLE and the graft must be re-bottlenecked "
    "(e.g. read-only decode / harder P=1 bottleneck) BEFORE main-wave GPU. "
    "PASS (NCR head trains AND is load-bearing) = full-graft recovers deep "
    "composition (recovered_frac@0.9 >= the sec 6.2 Gate-0 bar at deep h) "
    "AND materially exceeds backbone-only there. PRIMARY interpretable "
    "signal = the recovery GAP (full-graft - backbone-only), at DEEP "
    "composition depth."
)


# ---------------------------------------------------------------------------
# CLAUDE.md mod-K guard, ENFORCED (not merely inherited): sec 3.1's own
# residue-soundness citation was verified at K in {15,32} for this exact
# ladder -- K_NCR here is 24 (sec N2.1's Wave-1 pin), a THIRD value, so this
# is re-checked fresh, not assumed transferred.
# ---------------------------------------------------------------------------
def _assert_ladder_sound(ladder: tuple, k: int, train_hops: tuple) -> None:
    train_residues = {h % k for h in train_hops}
    for h in ladder:
        r = h % k
        assert r != 0, f"deep-ladder h={h} is IDENTITY mod K={k} (h%K=0) -- confounded, not held-out"
        assert r not in train_residues, (
            f"deep-ladder h={h} has h%K={r} colliding with a train-residue "
            f"{sorted(train_residues)} -- secretly in-distribution, not held-out")


_assert_ladder_sound(DEEP_LADDER, K_NCR, TRAIN_HOPS)


# ---------------------------------------------------------------------------
# IO helpers (pattern reused verbatim from matrix-thinking/ncr/run_ncr.py's
# own git_commit/atomic_write_json/stop_requested -- not imported directly
# since that module drags in unrelated heavy deps; the three functions
# themselves are tiny and self-contained).
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:
        return "UNKNOWN"


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def atomic_torch_save(path: str, obj) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def stop_requested(stop_file: str) -> bool:
    return bool(stop_file) and os.path.exists(stop_file)


# ---------------------------------------------------------------------------
# Read-ablation forward -- composed from AUDITED building blocks ONLY
# (integ.extract_kv / integ.query_key / ncr_head.encode / nm.binexp_read /
# integ.inject_and_logits_last are the SAME functions ncr_lm_forward() in
# ncr_lm_wave1_smoke.py calls; this wrapper does NOT edit that audited
# function, so its wiring stays byte-identical to what sec G3-B4 reviewed).
# ---------------------------------------------------------------------------
def ncr_lm_forward_ablatable(backbone, ncr_head, integ: NCRIntegration, batch: dict, read_ablate: bool,
                              teacher_force: bool = False):
    """Returns (logits, o_raw, o_injected, hidden, Z, keys_v, values_v).
    read_ablate=False (full_graft): o_injected is o_raw (same tensor, real
    gradient flows through it to the write pathway).
    read_ablate=True (backbone_only): o_injected = torch.zeros_like(o_raw) --
    a FRESH tensor with NO autograd edge to o_raw's own graph, so the write
    adapters/encoder receive ZERO gradient and the read contributes EXACTLY
    zero to the logits (verified by assert_read_ablation_is_exact_zero).

    teacher_force=True (sec G3-B9 diagnostic, ported from ncr_lm_wave1_smoke.
    ncr_lm_forward's own audited teacher_force branch, smoke item 10): Z is
    built by integ.teacher_force_operator(keys_v, values_v) -- a closed-form
    least-squares fit on DETACHED key/value-adapter outputs -- instead of
    ncr_head.encode(keys_v, values_v). ncr_head's parameters therefore NEVER
    enter the autograd graph in this branch (asserted explicitly in the
    training loop, not merely assumed here). read_ablate and teacher_force
    compose orthogonally: read_ablate still controls o_injected independently
    of where Z came from (backbone_only zeros o_injected regardless).

    sec G3-B12 (ported from ncr_lm_wave1_smoke.ncr_lm_forward's own fix):
    keys_v/values_v/q_key are now extracted from RAW `input_ids` through
    `backbone.embed` + the single shared `integ.entity_adapter` (was
    contextualized `hidden` through separate key_adapter/value_adapter) --
    see graft.NCRIntegration.extract_kv/query_key docstrings for the exact
    mechanism. `hidden` is still computed (unavoidable backbone forward) and
    still used for read-injection's own tap point (RULING 2, unaffected)."""
    input_ids = batch["doc"][:, :-1]
    hidden = backbone(input_ids, return_hidden=True)
    keys_v, values_v = integ.extract_kv(input_ids, batch["key_pos"], batch["val_pos"], backbone.embed)
    if teacher_force:
        Z = integ.teacher_force_operator(keys_v, values_v)
    else:
        Z = ncr_head.encode(keys_v, values_v)
    q_key = integ.query_key(input_ids, batch["query_key_col"], backbone.embed)
    o_raw = nm.binexp_read(Z, q_key.unsqueeze(1), h=batch["hop"])["o"].squeeze(1)
    o_injected = torch.zeros_like(o_raw) if read_ablate else o_raw
    logits = integ.inject_and_logits_last(hidden, o_injected, batch["query_mark_col"], backbone.embed.weight)
    return logits, o_raw, o_injected, hidden, Z, keys_v, values_v


@torch.no_grad()
def assert_read_ablation_is_exact_zero(backbone, ncr_head, integ: NCRIntegration, batch: dict) -> float:
    """Hard EXACT-equality check (CLAUDE.md: structural correctness checks
    need exact thresholds, not tolerance): the ablated arm's logits must be
    BYTE-FOR-BYTE identical to the tied head applied to h_q alone -- proves
    the read contributes literally zero, the property sec G3-B5's whole
    control rests on."""
    logits_ablated, o_raw, o_inj, hidden, _, _, _ = ncr_lm_forward_ablatable(
        backbone, ncr_head, integ, batch, read_ablate=True)
    h_q = hidden[:, batch["query_mark_col"], :]
    logits_pure = F.linear(h_q, backbone.embed.weight)
    max_diff = (logits_ablated - logits_pure).abs().max().item()
    assert torch.equal(o_inj, torch.zeros_like(o_inj)), "o_injected is not the exact zero tensor"
    assert torch.equal(logits_ablated, logits_pure), (
        f"READ ABLATION IS NOT EXACT-ZERO: max|logits_ablated-logits_pure|={max_diff:.3e} "
        f"(expected EXACT bitwise equality) -- the backbone-only control is NOT clean, "
        f"sec G3-B5's attribution rule does not hold")
    del o_raw
    return max_diff


# ---------------------------------------------------------------------------
# mean_cos diagnostic (sec G3-B9, in response to sec G3-B8's own flagged
# instrument-ambiguity: "recovered=0 in-dist consistent with didn't-learn OR
# o_raw-recovery-instrument-mis-wired"). recovered_frac@0.9 is a THRESHOLDED
# view of read quality (cos>=0.9 or nothing); this exposes the RAW mean
# cosine alongside it so the two failure modes are visually distinguishable:
# high mean_cos with rec@0.9=0 => the threshold discarded real (sub-0.9, but
# non-trivial) signal; near-zero mean_cos => the read genuinely carries
# nothing. Re-derives the IDENTICAL target/cosine computation
# graft.recovered_frac_at_09 (ncr_lm_wave1_smoke.py, AUDITED, sec G3-B12
# RE-BASED: target = entity_adapter(RAW embed(answer_token)), o's OWN
# space -- was key_adapter(hidden at the ANSWER entity's OWN bind-clause KEY
# position), sec G3-B11 defect 3d) uses internally, byte-for-byte --
# duplicated here (not edited into that audited file, keeping its md5/audit
# status untouched, per the build brief's "do NOT disturb the audited
# two-arm path") solely to obtain the raw per-row cosine tensor that
# function computes but does not expose (it only returns the thresholded
# fraction). recovered_frac@0.9 and mean_cos below are therefore ALWAYS
# derived from the SAME cosine tensor -- guaranteed consistent by
# construction, not two independently-invented metrics that could silently
# disagree.
# ---------------------------------------------------------------------------
def cosine_and_recovered_frac(integ: NCRIntegration, embed: torch.nn.Embedding, o: torch.Tensor,
                               answer_token: torch.Tensor) -> tuple[float, float]:
    """Returns (recovered_frac@0.9, mean_cos) from ONE shared cosine tensor.
    See the module comment immediately above for why this duplicates (not
    reimplements-differently) graft.recovered_frac_at_09's own (sec G3-B12
    re-based) target. answer_token: (B,) int64, the true answer entity's OWN
    token id (batch["answer_token"]) -- embed(answer_token) is context-free,
    no position-gather through `hidden` needed any more."""
    target = integ.entity_adapter(embed(answer_token).float())
    cos = F.cosine_similarity(o, target, dim=-1)
    return (cos >= 0.9).float().mean().item(), cos.mean().item()


# ---------------------------------------------------------------------------
# Two-arm construction -- BIT-IDENTICAL initial weights (same seed reset
# immediately before each arm's construction, same class/order of
# submodule creation) so the ONLY difference between arms across the whole
# run is the read-ablation itself ("same everything else", sec G3-B5).
# ---------------------------------------------------------------------------
def build_arm(vocab_size_total: int, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    backbone = build_backbone(vocab_size=vocab_size_total).to(device)
    ncr_head = build_ncr_head().to(device)
    integ = NCRIntegration(RUNG1_BACKBONE["d_model"], D_NCR, vocab_size_total,
                            adapter="linear", read_inject="add").to(device)
    return dict(backbone=backbone, ncr=ncr_head, integ=integ)


def build_two_arms(vocab_size_total: int, seed: int, device: str) -> dict:
    full_graft = build_arm(vocab_size_total, seed, device)
    backbone_only = build_arm(vocab_size_total, seed, device)     # SAME seed -> bit-identical init
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(
                sum((list(m.named_parameters()) for m in full_graft.values()), []),
                sum((list(m.named_parameters()) for m in backbone_only.values()), [])):
            assert n1 == n2 and torch.equal(p1, p2), (
                f"two-arm init mismatch at {n1}/{n2} -- 'same everything else' violated at construction")
    return dict(full_graft=full_graft, backbone_only=backbone_only)


def arm_params(arm: dict) -> int:
    return sum(p.numel() for m in arm.values() for p in m.parameters())


def build_optimizer(arm: dict, lr: float) -> torch.optim.Optimizer:
    params = list(arm["backbone"].parameters()) + list(arm["ncr"].parameters()) + list(arm["integ"].parameters())
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)     # weight_decay=0.0: lm_pretrain_rd.py
                                                                    # smoke()'s own convention, build-time
                                                                    # choice (not pinned by the design text)


# ---------------------------------------------------------------------------
# Eval -- FIXED per-h eval seeds (independent of training step and of which
# arm is being evaluated), so both arms are scored on LITERALLY the same
# held eval documents at every checkpoint.
# ---------------------------------------------------------------------------
EVAL_SEED_OFFSET = 999_983


@torch.no_grad()
def eval_arm_at_hops(arm: dict, pools, cfg, hops: tuple, batch_size: int, device: str,
                      base_seed: int, read_ablate: bool, teacher_force: bool = False) -> dict:
    backbone, ncr_head, integ = arm["backbone"], arm["ncr"], arm["integ"]
    backbone.eval(); ncr_head.eval(); integ.eval()
    out = {}
    for h in hops:
        gen = torch.Generator(device=device).manual_seed(base_seed + EVAL_SEED_OFFSET + h)
        batch = build_task1_document(cfg, pools, gen, batch_size, h, device)
        logits, o_raw, o_inj, hidden, Z, keys_v, values_v = ncr_lm_forward_ablatable(
            backbone, ncr_head, integ, batch, read_ablate=read_ablate, teacher_force=teacher_force)
        rf, mean_cos = cosine_and_recovered_frac(integ, backbone.embed, o_raw, batch["answer_token"])
        acc = (logits.argmax(dim=-1) == batch["answer_token"]).float().mean().item()
        out[f"h={h}"] = {"recovered_frac@0.9": float(rf), "mean_cos": float(mean_cos),
                          "answer_accuracy": float(acc), "n": batch_size}
    rf_vals = [v["recovered_frac@0.9"] for v in out.values()]
    cos_vals = [v["mean_cos"] for v in out.values()]
    acc_vals = [v["answer_accuracy"] for v in out.values()]
    out["mean_recovered_frac@0.9"] = float(sum(rf_vals) / len(rf_vals))
    out["mean_mean_cos"] = float(sum(cos_vals) / len(cos_vals))
    out["mean_answer_accuracy"] = float(sum(acc_vals) / len(acc_vals))
    backbone.train(); ncr_head.train(); integ.train()
    return out


def eval_both_arms(arms: dict, pools, cfg, batch_size: int, device: str, seed: int,
                    teacher_force: bool = False) -> dict:
    """teacher_force (sec G3-B9) is applied to the full_graft arm ONLY --
    backbone_only's o_raw always uses the normal ncr_head.encode() path
    regardless of this flag, preserving its role as the untrained,
    frozen-at-init encoder null baseline (sec G3-B5's own definition of
    what backbone_only's o_raw metric means); swapping in a teacher-forced
    fit for backbone_only too would change that baseline's meaning without
    being asked for, and its o_raw never touches the loss either way."""
    result = {}
    for arm_name, read_ablate in (("full_graft", False), ("backbone_only", True)):
        arm = arms[arm_name]
        tf_this_arm = teacher_force and arm_name == "full_graft"
        result[arm_name] = {
            "in_dist": eval_arm_at_hops(arm, pools, cfg, TRAIN_HOPS, batch_size, device, seed, read_ablate, tf_this_arm),
            "deep": eval_arm_at_hops(arm, pools, cfg, DEEP_LADDER, batch_size, device, seed, read_ablate, tf_this_arm),
        }
    return result


def build_attribution(eval_result: dict) -> dict:
    """Registers the frozen sec G3-B5 rule TEXT plus the raw numbers a blind
    assessor needs to apply it -- computes NO verdict (blind discipline)."""
    fg, bo = eval_result["full_graft"], eval_result["backbone_only"]
    in_dist_gap = {f"h={h}": fg["in_dist"][f"h={h}"]["recovered_frac@0.9"] - bo["in_dist"][f"h={h}"]["recovered_frac@0.9"]
                   for h in TRAIN_HOPS}
    deep_gap = {f"h={h}": fg["deep"][f"h={h}"]["recovered_frac@0.9"] - bo["deep"][f"h={h}"]["recovered_frac@0.9"]
                for h in DEEP_LADDER}
    # sec G3-B9 diagnostic addition (additive only, does not alter any
    # existing key above): the SAME gap construction but on mean_cos instead
    # of the thresholded recovered_frac@0.9, so a blind assessor can see
    # whether the read carries graded signal even when rec@0.9 floors at 0.
    in_dist_cos_gap = {f"h={h}": fg["in_dist"][f"h={h}"]["mean_cos"] - bo["in_dist"][f"h={h}"]["mean_cos"]
                        for h in TRAIN_HOPS}
    deep_cos_gap = {f"h={h}": fg["deep"][f"h={h}"]["mean_cos"] - bo["deep"][f"h={h}"]["mean_cos"]
                    for h in DEEP_LADDER}
    return {
        "frozen_rule_text": ATTRIBUTION_RULE_TEXT,
        "primary_signal_definition": (
            "recovered_frac@0.9 GAP (full_graft.o_raw - backbone_only.o_raw), at DEEP composition "
            "depth (h in {5,12,20,29,40,61}). backbone_only.o_raw's write/encode pathway receives NO "
            "gradient (read-ablated -- see read_ablation_check), so it is a frozen-at-init NULL "
            "baseline for the recovery metric, not a trained comparison."
        ),
        "recovered_frac_gap_in_dist": in_dist_gap,
        "recovered_frac_gap_deep": deep_gap,
        "primary_signal_deepest_gap_h61": deep_gap["h=61"],
        "mean_cos_gap_in_dist": in_dist_cos_gap,     # sec G3-B9 diagnostic addition
        "mean_cos_gap_deep": deep_cos_gap,           # sec G3-B9 diagnostic addition
        "attribution_precondition_metric": "answer_accuracy (argmax(logits)==answer_token), NOT recovered_frac@0.9 "
                                            "-- this is what the frozen rule text's own prose names",
        "answer_accuracy_in_dist": {"full_graft": {h: fg["in_dist"][h]["answer_accuracy"] for h in fg["in_dist"] if h.startswith("h=")},
                                     "backbone_only": {h: bo["in_dist"][h]["answer_accuracy"] for h in bo["in_dist"] if h.startswith("h=")}},
        "answer_accuracy_deep": {"full_graft": {h: fg["deep"][h]["answer_accuracy"] for h in fg["deep"] if h.startswith("h=")},
                                  "backbone_only": {h: bo["deep"][h]["answer_accuracy"] for h in bo["deep"] if h.startswith("h=")}},
    }


# ---------------------------------------------------------------------------
# Checkpoint save/load -- TRUE step-level resume (a deliberate, disclosed
# strengthening of this repo's own toy-cell "whole-cell skip-if-COMPLETED,
# else restart from scratch" precedent, ncr_ortho_fallback_stage0_v3.py's
# run_disc_cell/run_primary_cell -- justified because this cell is priced at
# multiple contended GPU-hours, not the toy cells' ~1-3 GPU-h, and the build
# brief explicitly requires "checkpoint/resume works" as its own smoke item,
# distinct from "ceiling fires correctly"). Generator states are saved so a
# resumed run reproduces the EXACT same data stream, not merely "a" stream.
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, step: int, arms: dict, opts: dict, data_gen: torch.Generator,
                     cumulative_elapsed_s: float, cell_id: str) -> None:
    ckpt = {
        "runner_tag": RUNNER_TAG, "cell_id": cell_id, "step": step,
        "cumulative_elapsed_s": cumulative_elapsed_s,
        "data_gen_state": data_gen.get_state(),
    }
    for arm_name, arm in arms.items():
        ckpt[arm_name] = {
            "backbone_state": arm["backbone"].state_dict(), "backbone_config": arm["backbone"].config(),
            "ncr_state": arm["ncr"].state_dict(), "ncr_config": {"d": D_NCR, "h": H_NCR},
            "integ_state": arm["integ"].state_dict(), "integ_config": arm["integ"].config(),
            "opt_state": opts[arm_name].state_dict(),
        }
    atomic_torch_save(path, ckpt)


def load_checkpoint(path: str, device: str) -> dict | None:
    """Validity-checked resume (CLAUDE.md: 'counts as done only if the
    output parses and has the expected keys, not just file exists')."""
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location=device)
        assert ckpt.get("runner_tag") == RUNNER_TAG and "step" in ckpt and "full_graft" in ckpt and "backbone_only" in ckpt
        return ckpt
    except Exception as e:
        print(f"  [checkpoint] {path} failed to load/validate ({e!r}) -- treating as ABSENT, starting fresh", flush=True)
        return None


def restore_arms_and_opts(ckpt: dict, vocab_size_total: int, lr: float, device: str) -> tuple[dict, dict, torch.Generator]:
    arms, opts = {}, {}
    for arm_name in ("full_graft", "backbone_only"):
        c = ckpt[arm_name]
        backbone = graft.DeltaNetLM(**c["backbone_config"]).to(device)
        backbone.load_state_dict(c["backbone_state"])
        ncr_head = graft.els.NCREarlyLNModel(**c["ncr_config"]).to(device)
        ncr_head.load_state_dict(c["ncr_state"])
        integ_cfg = c["integ_config"]
        integ = NCRIntegration(integ_cfg["d_model"], integ_cfg["d_ncr"], integ_cfg["vocab_size"],
                                adapter=integ_cfg["adapter"], read_inject=integ_cfg["read_inject"]).to(device)
        integ.load_state_dict(c["integ_state"])
        arm = dict(backbone=backbone, ncr=ncr_head, integ=integ)
        arms[arm_name] = arm
        opt = build_optimizer(arm, lr)
        opt.load_state_dict(c["opt_state"])
        opts[arm_name] = opt
    data_gen = torch.Generator(device=device)
    # torch.load(map_location="cuda") in load_checkpoint moves EVERY saved
    # tensor -- including the generator's own state ByteTensor -- onto the
    # GPU; Generator.set_state() requires a CPU ByteTensor (a torch RNG state
    # is always CPU-resident regardless of the generator's device), so a raw
    # set_state(cuda_bytetensor) raises "RNG state must be a torch.ByteTensor".
    # Force it back to a CPU uint8 ByteTensor before restoring. (Caught by the
    # runner-smoke's own sub-test B, sec G3-B6 -- the resume path is exactly
    # what that smoke exists to exercise.)
    gen_state = ckpt["data_gen_state"]
    if gen_state.device.type != "cpu":
        gen_state = gen_state.cpu()
    data_gen.set_state(gen_state.to(torch.uint8))
    return arms, opts, data_gen


# ---------------------------------------------------------------------------
# Main two-arm training+eval cell.
# ---------------------------------------------------------------------------
def run_two_arm_cell(cell_id: str, steps: int, batch_size: int, eval_batch_size: int,
                      lr: float, warmup_steps: int, ceiling_gpuh: float, seed: int,
                      device: str, out_path: str, ckpt_path: str, stop_file: str,
                      ckpt_every: int, eval_every: int, teacher_force_operator: bool = False) -> dict:
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "COMPLETED":
            print(f"[{cell_id}] already COMPLETED -- skipping (resume-safe)", flush=True)
            return prev

    pools, cfg, pool_report = build_grammar_pools_and_cfg(seed=seed)
    vocab_size_total = pool_report["vocab_size_total"]
    pools = pools.to(device)

    ckpt = load_checkpoint(ckpt_path, device)
    if ckpt is not None:
        print(f"[{cell_id}] RESUMING from checkpoint at step {ckpt['step']} "
              f"(cumulative_elapsed_s={ckpt['cumulative_elapsed_s']:.0f})", flush=True)
        arms, opts, data_gen = restore_arms_and_opts(ckpt, vocab_size_total, lr, device)
        start_step = ckpt["step"]
        cumulative_elapsed_s = ckpt["cumulative_elapsed_s"]
    else:
        arms = build_two_arms(vocab_size_total, seed, device)
        opts = {name: build_optimizer(arm, lr) for name, arm in arms.items()}
        data_gen = torch.Generator(device=device).manual_seed(seed + 777)
        start_step = 0
        cumulative_elapsed_s = 0.0

    n_params = {name: arm_params(arm) for name, arm in arms.items()}
    assert n_params["full_graft"] == n_params["backbone_only"], "two arms must be param-count-identical"

    rec = dict(
        cell_id=cell_id, runner_tag=RUNNER_TAG, mode="calibration",
        status="RUNNING", step=start_step, steps_target=steps,
        config=dict(K=K_NCR, d_ncr=D_NCR, h_ncr=H_NCR, backbone=RUNG1_BACKBONE,
                    vocab_size_total=vocab_size_total, seed=seed, batch_size=batch_size,
                    eval_batch_size=eval_batch_size, lr=lr, warmup_steps=warmup_steps,
                    train_hops=list(TRAIN_HOPS), deep_ladder=list(DEEP_LADDER),
                    ceiling_gpuh=ceiling_gpuh, teacher_force_operator=teacher_force_operator),
        params=dict(per_arm=n_params["full_graft"],
                    backbone=sum(p.numel() for p in arms["full_graft"]["backbone"].parameters()),
                    ncr_head=sum(p.numel() for p in arms["full_graft"]["ncr"].parameters()),
                    integ=sum(p.numel() for p in arms["full_graft"]["integ"].parameters())),
        git_commit=git_commit(), host=socket.gethostname(), device=device,
        torch_version=torch.__version__,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Read-ablation exact-zero check (BEFORE training) -- the property
    # sec G3-B5's whole control rests on. Loud on failure: this is not
    # something a calibration should silently proceed past.
    probe_gen = torch.Generator(device=device).manual_seed(seed + 424242)
    probe_batch = build_task1_document(cfg, pools, probe_gen, 4, TRAIN_HOPS[0], device)
    diff_pre = assert_read_ablation_is_exact_zero(
        arms["backbone_only"]["backbone"], arms["backbone_only"]["ncr"], arms["backbone_only"]["integ"], probe_batch)
    rec["read_ablation_check"] = {"pre_train_max_abs_diff": diff_pre, "pre_train_verified_exact_zero": True}
    print(f"[{cell_id}] read-ablation exact-zero check PASSED (pre-train, max_abs_diff={diff_pre:.2e})", flush=True)
    assert steps >= 1, "run_two_arm_cell requires steps >= 1 (use --mode phase0-timing for a zero-training-step rate probe)"

    ceiling_s = ceiling_gpuh * 3600.0 if device == "cuda" else float("inf")
    t0 = time.time() - cumulative_elapsed_s
    loss_hist = {"full_graft": [], "backbone_only": []}
    n_skipped = {"full_graft": 0, "backbone_only": 0}
    final_status = "COMPLETED"
    # sec G3-B9: counts PASSED per-step encoder-zero-grad assertions (full_graft
    # arm only, see the training loop below) -- NOT restored across a resume
    # (fresh per-process, same convention as n_skipped above), so it reports
    # checks passed IN THIS PROCESS's run only.
    teacher_force_ncr_zero_grad_checks = 0
    rec["teacher_force_check"] = {"active": teacher_force_operator, "ncr_zero_grad_checks_passed": 0}

    for step in range(start_step + 1, steps + 1):
        cur_lr = get_lr(step, max_lr=lr, warmup_steps=warmup_steps, total_steps=steps)
        idx = torch.randint(0, len(TRAIN_HOPS), (1,), generator=data_gen, device=device).item()
        hop_value = TRAIN_HOPS[idx]
        batch = build_task1_document(cfg, pools, data_gen, batch_size, hop_value, device)

        step_losses = {}
        for arm_name, read_ablate in (("full_graft", False), ("backbone_only", True)):
            arm, opt = arms[arm_name], opts[arm_name]
            for g in opt.param_groups:
                g["lr"] = cur_lr
            # sec G3-B9: teacher_force applies to full_graft ONLY (see
            # eval_both_arms's own docstring for the identical rationale --
            # backbone_only's o_raw stays the untrained-encoder null baseline
            # regardless of this flag, and never touches its own loss anyway).
            tf_this_arm = teacher_force_operator and arm_name == "full_graft"
            logits, o_raw, o_inj, hidden, Z, keys_v, values_v = ncr_lm_forward_ablatable(
                arm["backbone"], arm["ncr"], arm["integ"], batch, read_ablate=read_ablate,
                teacher_force=tf_this_arm)
            loss = F.cross_entropy(logits, batch["answer_token"])
            opt.zero_grad()
            loss.backward()
            if tf_this_arm:
                # sec G3-B9 isolation proof, ported from ncr_lm_wave1_smoke.py smoke
                # item 10's construction-time check into this training loop, run
                # EVERY step this mode is active (not merely once): teacher-forcing
                # bypasses ncr_head.encode() entirely (see ncr_lm_forward_ablatable),
                # so ncr_head's parameters must receive EXACTLY zero gradient. Loud
                # AssertionError on violation -- CLAUDE.md: structural correctness
                # checks need exact thresholds, never silently trusted.
                ncr_untouched = all(p.grad is None for p in arm["ncr"].parameters())
                assert ncr_untouched, (
                    f"--teacher-force-operator step {step}: ncr_head (BindingEncoder) received a "
                    f"non-None gradient -- teacher-force isolation broken, the encoder pathway is "
                    f"not actually bypassed")
                teacher_force_ncr_zero_grad_checks += 1
            all_params = list(arm["backbone"].parameters()) + list(arm["ncr"].parameters()) + list(arm["integ"].parameters())
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in all_params)
            if finite:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
            else:
                n_skipped[arm_name] += 1
            step_losses[arm_name] = loss.item()

        if step % LOG_EVERY == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            loss_hist["full_graft"].append([step, step_losses["full_graft"]])
            loss_hist["backbone_only"].append([step, step_losses["backbone_only"]])
            # BLIND: loss is operational telemetry (liveness/divergence), never an eval metric --
            # matches ncr_ortho_fallback_stage0_v3.py's own precedent.
            print(f"[{cell_id}] step {step}/{steps}  full_graft_loss={step_losses['full_graft']:.4f} "
                  f"backbone_only_loss={step_losses['backbone_only']:.4f}  lr={cur_lr:.2e}  "
                  f"{elapsed:.0f}s", flush=True)

        if step % ckpt_every == 0 or step == steps:
            save_checkpoint(ckpt_path, step, arms, opts, data_gen, time.time() - t0, cell_id)

        if step % eval_every == 0 or step == steps:
            eval_result = eval_both_arms(arms, pools, cfg, eval_batch_size, device, seed,
                                          teacher_force=teacher_force_operator)
            rec["arms"] = eval_result
            rec["attribution"] = build_attribution(eval_result)
            rec["step"] = step
            rec["elapsed_s"] = time.time() - t0
            rec["loss_history"] = loss_hist
            rec["n_skipped_steps"] = n_skipped
            rec["teacher_force_check"]["ncr_zero_grad_checks_passed"] = teacher_force_ncr_zero_grad_checks
            atomic_write_json(out_path, rec)
            print(f"[{cell_id}] eval computed at step {step} -> {out_path} updated "
                  f"(values withheld from stdout, blind discipline sec G3-B6)", flush=True)

        if stop_requested(stop_file):
            save_checkpoint(ckpt_path, step, arms, opts, data_gen, time.time() - t0, cell_id)
            print(f"[{cell_id}] STOP file detected at step {step} -- checkpoint saved, exiting", flush=True)
            sys.exit(3)

        elapsed = time.time() - t0
        if elapsed > ceiling_s:
            final_status = "ABORTED-BUDGET"
            save_checkpoint(ckpt_path, step, arms, opts, data_gen, elapsed, cell_id)
            eval_result = eval_both_arms(arms, pools, cfg, eval_batch_size, device, seed,
                                          teacher_force=teacher_force_operator)
            rec["arms"] = eval_result
            rec["attribution"] = build_attribution(eval_result)
            rec["step"] = step
            rec["loss_history"] = loss_hist
            rec["n_skipped_steps"] = n_skipped
            rec["teacher_force_check"]["ncr_zero_grad_checks_passed"] = teacher_force_ncr_zero_grad_checks
            break

    # Post-train read-ablation re-check (paranoia: prove the invariant held
    # THROUGHOUT training, not just at init).
    diff_post = assert_read_ablation_is_exact_zero(
        arms["backbone_only"]["backbone"], arms["backbone_only"]["ncr"], arms["backbone_only"]["integ"], probe_batch)
    rec["read_ablation_check"]["post_train_max_abs_diff"] = diff_post
    rec["read_ablation_check"]["post_train_verified_exact_zero"] = True
    print(f"[{cell_id}] read-ablation exact-zero check PASSED (post-train, max_abs_diff={diff_post:.2e})", flush=True)

    rec["elapsed_s"] = time.time() - t0
    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
    rec["status"] = final_status
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(out_path, rec)
    print(f"[{cell_id}] {final_status} at step {rec['step']}/{steps} in {rec['elapsed_s']:.0f}s -> {out_path}", flush=True)
    return rec


# ---------------------------------------------------------------------------
# Phase-0 timing probe -- measures the REAL per-step wall-clock rate of the
# exact two-arm loop above (not a synthetic/toy timing proxy), at the
# operating point the calibration cell will actually use. Reports (never
# hides -- this is operational telemetry, not the interpretive signal) the
# uncontended rate, the CONTENDED_MULTIPLIER-scaled projection, and a
# suggested --ceiling-gpuh for a given --target-steps.
# ---------------------------------------------------------------------------
def run_phase0_timing(batch_size: int, warmup_steps: int, probe_steps: int, target_steps: int,
                       lr: float, seed: int, device: str, out_path: str) -> dict:
    pools, cfg, pool_report = build_grammar_pools_and_cfg(seed=seed)
    vocab_size_total = pool_report["vocab_size_total"]
    pools = pools.to(device)
    arms = build_two_arms(vocab_size_total, seed, device)
    opts = {name: build_optimizer(arm, lr) for name, arm in arms.items()}
    data_gen = torch.Generator(device=device).manual_seed(seed + 777)

    def one_step():
        idx = torch.randint(0, len(TRAIN_HOPS), (1,), generator=data_gen, device=device).item()
        hop_value = TRAIN_HOPS[idx]
        batch = build_task1_document(cfg, pools, data_gen, batch_size, hop_value, device)
        per_arm_s = {}
        for arm_name, read_ablate in (("full_graft", False), ("backbone_only", True)):
            if device == "cuda":
                torch.cuda.synchronize()
            t_arm0 = time.time()
            arm, opt = arms[arm_name], opts[arm_name]
            logits, o_raw, o_inj, hidden, Z, keys_v, values_v = ncr_lm_forward_ablatable(
                arm["backbone"], arm["ncr"], arm["integ"], batch, read_ablate=read_ablate)
            loss = F.cross_entropy(logits, batch["answer_token"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            per_arm_s[arm_name] = time.time() - t_arm0
        return per_arm_s

    print(f"[phase0-timing] warmup: {warmup_steps} steps (untimed, CUDA kernel compile/cache)...", flush=True)
    for _ in range(warmup_steps):
        one_step()

    print(f"[phase0-timing] probe: {probe_steps} steps (timed)...", flush=True)
    per_step_full, per_step_backbone, per_step_total = [], [], []
    t_probe0 = time.time()
    for i in range(probe_steps):
        s = one_step()
        per_step_full.append(s["full_graft"])
        per_step_backbone.append(s["backbone_only"])
        per_step_total.append(s["full_graft"] + s["backbone_only"])
        if (i + 1) % 10 == 0:
            print(f"[phase0-timing] {i+1}/{probe_steps} probe steps done ({time.time()-t_probe0:.1f}s so far)", flush=True)
    wall = time.time() - t_probe0

    mean_full = sum(per_step_full) / len(per_step_full)
    mean_backbone = sum(per_step_backbone) / len(per_step_backbone)
    mean_total = sum(per_step_total) / len(per_step_total)
    doc_len = cfg.T_bind + cfg.query_len          # the fixed document length (independent of hop value)
    tokens_per_step_per_arm = batch_size * doc_len

    contended_total_s_per_step = mean_total * CONTENDED_MULTIPLIER
    projected_uncontended_s = target_steps * mean_total
    projected_contended_s = target_steps * contended_total_s_per_step
    suggested_ceiling_gpuh = round(projected_contended_s / 3600.0 * 1.15, 3)   # +15% pad, matches
                                                                                  # stage0_v3's own internal-
                                                                                  # ceiling-vs-external-backstop ratio

    rec = dict(
        runner_tag=RUNNER_TAG, mode="phase0-timing",
        config=dict(K=K_NCR, d_ncr=D_NCR, backbone=RUNG1_BACKBONE, vocab_size_total=vocab_size_total,
                    seed=seed, batch_size=batch_size, warmup_steps=warmup_steps, probe_steps=probe_steps,
                    target_steps=target_steps, lr=lr, doc_len=doc_len),
        measured=dict(
            mean_s_per_step_full_graft=mean_full, mean_s_per_step_backbone_only=mean_backbone,
            mean_s_per_step_both_arms_combined=mean_total,
            tokens_per_step_per_arm=tokens_per_step_per_arm,
            tokens_per_sec_per_arm_full_graft=tokens_per_step_per_arm / mean_full,
            tokens_per_sec_per_arm_backbone_only=tokens_per_step_per_arm / mean_backbone,
            probe_wall_clock_s=wall,
        ),
        projected=dict(
            contended_multiplier=CONTENDED_MULTIPLIER,
            uncontended_s_for_target_steps=projected_uncontended_s,
            uncontended_gpuh_for_target_steps=projected_uncontended_s / 3600.0,
            contended_s_for_target_steps=projected_contended_s,
            contended_gpuh_for_target_steps=projected_contended_s / 3600.0,
            suggested_ceiling_gpuh=suggested_ceiling_gpuh,
        ),
        host=socket.gethostname(), device=device, torch_version=torch.__version__,
        git_commit=git_commit(),
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    atomic_write_json(out_path, rec)
    print("=" * 70, flush=True)
    print(f"[phase0-timing] MEASURED (uncontended): {mean_total:.4f} s/step (both arms combined), "
          f"full_graft={mean_full:.4f}s backbone_only={mean_backbone:.4f}s", flush=True)
    print(f"[phase0-timing] PROJECTED contended ({CONTENDED_MULTIPLIER}x) rate for "
          f"target_steps={target_steps}: {projected_contended_s/3600.0:.3f} GPU-h", flush=True)
    print(f"[phase0-timing] SUGGESTED --ceiling-gpuh {suggested_ceiling_gpuh} "
          f"(contended projection + 15% pad)", flush=True)
    print(f"[phase0-timing] results written to {out_path}", flush=True)
    print("=" * 70, flush=True)
    return rec


# ---------------------------------------------------------------------------
# Runner smoke -- THIS build's OWN required real-CUDA smoke of the RUNNER
# (not the graft, which sec G3-B3/B4 already smoked/audited). Drives the
# script via REAL subprocess invocations of --mode calibration (the
# faithful way the tmux supervisor will actually invoke it), not in-process
# function calls, so the resume test exercises a genuine fresh-process
# checkpoint load.
# ---------------------------------------------------------------------------
def run_runner_smoke(device: str, outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    py = sys.executable
    script = os.path.abspath(__file__)
    failures = []

    def invoke(mode, cell_id, steps, ceiling_gpuh, extra=()):
        out_path = os.path.join(outdir, f"{cell_id}.json")
        ckpt_path = os.path.join(outdir, f"{cell_id}.ckpt.pt")
        cmd = [py, script, "--mode", mode, "--device", device, "--cell-id", cell_id,
               "--steps", str(steps), "--ceiling-gpuh", str(ceiling_gpuh),
               "--batch-size", "8", "--eval-batch-size", "16", "--warmup-steps", "10",
               "--ckpt-every", "50", "--eval-every", "50",
               "--out", out_path, "--ckpt-dir", outdir, "--seed", "0", *extra]
        print(f"\n[runner-smoke] SUBPROCESS: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout[-4000:], flush=True)
        if proc.returncode not in (0,):
            print(f"[runner-smoke] subprocess stderr:\n{proc.stderr[-4000:]}", file=sys.stderr, flush=True)
        wall = time.time() - t0
        rec = None
        if os.path.exists(out_path):
            with open(out_path) as f:
                rec = json.load(f)
        return proc.returncode, rec, wall

    # --- sub-test A: a full short run (300 steps), start-to-finish ---------
    cid_a = "runner_smoke_main"
    rc_a, rec_a, wall_a = invoke("calibration", cid_a, 300, 5.0)
    ok_a = (rc_a == 0 and rec_a is not None and rec_a.get("status") == "COMPLETED"
            and rec_a.get("step") == 300 and "attribution" in rec_a and "arms" in rec_a
            and rec_a["read_ablation_check"].get("pre_train_verified_exact_zero") is True
            and rec_a["read_ablation_check"].get("post_train_verified_exact_zero") is True
            and rec_a["read_ablation_check"]["pre_train_max_abs_diff"] == 0.0
            and rec_a["read_ablation_check"]["post_train_max_abs_diff"] == 0.0
            and "recovered_frac_gap_deep" in rec_a["attribution"]
            and "answer_accuracy_in_dist" in rec_a["attribution"])
    print(f"[runner-smoke A: full 300-step run, both arms train, attribution fields present, "
          f"read-ablation exact-zero pre+post] {'PASS' if ok_a else 'FAIL'} (wall={wall_a:.1f}s)", flush=True)
    if not ok_a:
        failures.append("A:full-run")

    # --- sub-test B: checkpoint/resume -- run to step 100, kill, resume to 300 ---
    cid_b = "runner_smoke_resume"
    rc_b1, rec_b1, wall_b1 = invoke("calibration", cid_b, 100, 5.0)
    ok_b1 = rc_b1 == 0 and rec_b1 is not None and rec_b1.get("status") == "COMPLETED" and rec_b1.get("step") == 100
    # A FRESH subprocess, pointed at the same out/ckpt paths but a HIGHER
    # step target -- since out_path already reads status=COMPLETED at
    # step=100, the whole-cell skip would normally fire; delete the results
    # JSON (simulating "the supervisor is re-launched with a raised step
    # target after review") while leaving the CHECKPOINT in place, so the
    # resume path (not the whole-cell-skip path) is what gets exercised.
    out_path_b = os.path.join(outdir, f"{cid_b}.json")
    if os.path.exists(out_path_b):
        os.remove(out_path_b)
    rc_b2, rec_b2, wall_b2 = invoke("calibration", cid_b, 300, 5.0)
    ok_b2 = (rc_b2 == 0 and rec_b2 is not None and rec_b2.get("status") == "COMPLETED"
             and rec_b2.get("step") == 300)
    # resume correctness: the second invocation's own stdout must show it
    # RESUMED from step 100 (not restarted from step 0) -- re-run with
    # captured stdout to check for the "RESUMING from checkpoint at step
    # 100" line (invoke() already printed it above; re-derive from the
    # elapsed step count instead, which is a stronger, code-level check):
    # loss_history's first logged step in the second invocation's OWN run
    # should start at >100, not at 1, IF resume truly continued rather than
    # restarted (a restart would re-log step=1..300 from scratch).
    resumed_correctly = False
    if rec_b2 is not None:
        fg_hist = rec_b2.get("loss_history", {}).get("full_graft", [])
        first_logged_step = fg_hist[0][0] if fg_hist else None
        # LOG_EVERY=25 and ckpt/eval at 50 -- if truly resumed from 100, the
        # NEXT loop iteration is step 101, and the first log/ckpt/eval line
        # this SECOND process itself would hit is step 125 (next LOG_EVERY
        # multiple >100) or the final step=300 catch-all. A cold restart
        # would instead log step=1 first. So first_logged_step>100 proves resume.
        resumed_correctly = first_logged_step is not None and first_logged_step > 100
    ok_b = ok_b1 and ok_b2 and resumed_correctly
    print(f"[runner-smoke B: checkpoint/resume -- run to 100 (fresh proc), delete results JSON only, "
          f"re-invoke to 300 (fresh proc) -- must RESUME from ckpt not restart] {'PASS' if ok_b else 'FAIL'} "
          f"(resumed_correctly={resumed_correctly}, wall={wall_b1+wall_b2:.1f}s)", flush=True)
    if not ok_b:
        failures.append("B:checkpoint-resume")

    # --- sub-test C: ceiling fires correctly (tiny ceiling, expects ABORTED-BUDGET well before target) ---
    cid_c = "runner_smoke_ceiling"
    rc_c, rec_c, wall_c = invoke("calibration", cid_c, 5000, 0.0003)   # 0.0003 GPU-h ~= 1.1s
    ok_c = (rc_c == 0 and rec_c is not None and rec_c.get("status") == "ABORTED-BUDGET"
            and rec_c.get("step", 10**9) < 5000)
    print(f"[runner-smoke C: tiny --ceiling-gpuh 0.0003 must ABORT-BUDGET well before steps=5000] "
          f"{'PASS' if ok_c else 'FAIL'} (final_step={rec_c.get('step') if rec_c else None}, wall={wall_c:.1f}s)", flush=True)
    if not ok_c:
        failures.append("C:ceiling")

    # --- sub-test D: whole-cell skip-if-COMPLETED (re-invoke A's cell, must skip instantly) ---
    t0 = time.time()
    rc_d, rec_d, wall_d = invoke("calibration", cid_a, 300, 5.0)
    ok_d = rc_d == 0 and rec_d is not None and rec_d.get("status") == "COMPLETED" and wall_d < wall_a
    print(f"[runner-smoke D: re-invoking a COMPLETED cell must skip instantly, not re-train] "
          f"{'PASS' if ok_d else 'FAIL'} (wall={wall_d:.1f}s vs original {wall_a:.1f}s)", flush=True)
    if not ok_d:
        failures.append("D:whole-cell-skip")

    # --- sub-test E: --teacher-force-operator diagnostic mode (sec G3-B9) ---
    cid_e = "runner_smoke_teacher_force"
    rc_e1, rec_e1, wall_e1 = invoke("calibration", cid_e, 100, 5.0, extra=("--teacher-force-operator",))

    def _mean_cos_present(rec) -> bool:
        if rec is None or "arms" not in rec:
            return False
        try:
            return all("mean_cos" in rec["arms"][arm][band][f"h={h}"]
                       for arm in ("full_graft", "backbone_only")
                       for band, hops in (("in_dist", TRAIN_HOPS), ("deep", DEEP_LADDER))
                       for h in hops)
        except (KeyError, TypeError):
            return False

    ok_e1 = (rc_e1 == 0 and rec_e1 is not None and rec_e1.get("status") == "COMPLETED"
             and rec_e1.get("step") == 100
             and rec_e1.get("config", {}).get("teacher_force_operator") is True
             and rec_e1.get("teacher_force_check", {}).get("active") is True
             and rec_e1.get("teacher_force_check", {}).get("ncr_zero_grad_checks_passed", 0) > 0
             and _mean_cos_present(rec_e1))
    print(f"[runner-smoke E1: --teacher-force-operator, 100-step run -- COMPLETED without the "
          f"encoder-zero-grad AssertionError firing (checks_passed>0), mean_cos present at every "
          f"arm/band/h] {'PASS' if ok_e1 else 'FAIL'} "
          f"(checks_passed={rec_e1.get('teacher_force_check', {}).get('ncr_zero_grad_checks_passed') if rec_e1 else None}, "
          f"wall={wall_e1:.1f}s)", flush=True)
    if not ok_e1:
        failures.append("E1:teacher-force-run")

    # checkpoint/resume under teacher-force: delete results JSON (keep ckpt), re-invoke to a
    # higher step target with the SAME flag -- must RESUME (not restart), and the per-process
    # encoder-zero-grad check count must be > 0 again post-resume (proving the assertion re-ran
    # in the fresh process, not silently skipped).
    out_path_e = os.path.join(outdir, f"{cid_e}.json")
    if os.path.exists(out_path_e):
        os.remove(out_path_e)
    rc_e2, rec_e2, wall_e2 = invoke("calibration", cid_e, 200, 5.0, extra=("--teacher-force-operator",))
    resumed_e = False
    if rec_e2 is not None:
        fg_hist_e = rec_e2.get("loss_history", {}).get("full_graft", [])
        first_logged_step_e = fg_hist_e[0][0] if fg_hist_e else None
        resumed_e = first_logged_step_e is not None and first_logged_step_e > 100
    ok_e2 = (rc_e2 == 0 and rec_e2 is not None and rec_e2.get("status") == "COMPLETED"
             and rec_e2.get("step") == 200 and resumed_e
             and rec_e2.get("teacher_force_check", {}).get("ncr_zero_grad_checks_passed", 0) > 0
             and _mean_cos_present(rec_e2))
    print(f"[runner-smoke E2: --teacher-force-operator checkpoint/resume -- must RESUME from 100 to "
          f"200 (not restart), encoder-zero-grad checks keep passing post-resume, mean_cos still "
          f"present] {'PASS' if ok_e2 else 'FAIL'} (resumed={resumed_e}, wall={wall_e2:.1f}s)", flush=True)
    if not ok_e2:
        failures.append("E2:teacher-force-resume")

    summary = dict(runner_tag=RUNNER_TAG, mode="smoke", failures=failures,
                    sub_results=dict(A=rec_a, B_first=rec_b1, B_second=rec_b2, C=rec_c, D=rec_d,
                                      E1=rec_e1, E2=rec_e2),
                    finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_write_json(os.path.join(outdir, "runner_smoke_summary.json"), summary)
    print("=" * 70, flush=True)
    if failures:
        print(f"RUNNER SMOKE: {len(failures)} FAILURE(S): {failures}", file=sys.stderr, flush=True)
    else:
        print("RUNNER SMOKE: ALL 6 SUB-TESTS PASSED", flush=True)
    print("=" * 70, flush=True)
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("phase0-timing", "calibration", "smoke"))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20_000,
                     help="sec 6.2's own stated Phase-1 reduced-calibration step budget, reused directly "
                          "(disclosed transfer -- this cell's docs are shorter than sec 6.1's seq_len=512 "
                          "real-corpus operating point the number was originally token-budgeted against)")
    ap.add_argument("--ceiling-gpuh", type=float, default=None,
                     help="REQUIRED for --mode calibration (no silent default -- must be set from a "
                          "--mode phase0-timing measurement, contended-priced)")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--probe-steps", type=int, default=50, help="--mode phase0-timing only")
    ap.add_argument("--target-steps", type=int, default=20_000, help="--mode phase0-timing only: "
                     "the step count to project the contended GPU-h ceiling for")
    ap.add_argument("--cell-id", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--teacher-force-operator", action="store_true",
                     help="sec G3-B9 diagnostic mode (--mode calibration only): replace the "
                          "ncr_head-encoded operator Z with the closed-form least-squares "
                          "teacher-forced fit (ncr_lm_wave1_smoke.py's own audited "
                          "teacher_force_operator, smoke item 10) in BOTH the training loss and "
                          "eval, for the full_graft arm ONLY -- backbone_only is unaffected (its "
                          "read stays exact-zero and its o_raw stays the untrained-encoder null "
                          "baseline regardless of this flag). Isolates write-learning (the "
                          "encoder) from read/inject/backbone-learning: LEARNS in-distribution "
                          "under this flag => the encoder (write side) is the blocker; STAYS at "
                          "chance => the read-injection or task/loss setup is broken. The encoder "
                          "is asserted to receive EXACTLY zero gradient every step this is active "
                          "(loud AssertionError on violation, never silently trusted).")
    args = ap.parse_args()

    print("=" * 70)
    print(f"NCR REAL-LM WAVE-1 CALIBRATION RUNNER (sec G3-B6) mode={args.mode} device={args.device} "
          f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "cuda requested but not available"
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 70)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.mode == "phase0-timing":
        rec = run_phase0_timing(args.batch_size, args.warmup_steps, args.probe_steps, args.target_steps,
                                 args.lr, args.seed, args.device, args.out)
        return 0

    if args.mode == "smoke":
        outdir = os.path.dirname(os.path.abspath(args.out)) or "."
        return run_runner_smoke(args.device, outdir)

    # --- calibration ---
    assert args.ceiling_gpuh is not None, (
        "--ceiling-gpuh is REQUIRED for --mode calibration -- run --mode phase0-timing first and pass "
        "its suggested_ceiling_gpuh explicitly (no silent default; sec G3-B6)")
    ckpt_dir = args.ckpt_dir or os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    cell_id = args.cell_id or f"wave1_calib_K{K_NCR}_s{args.seed}"
    ckpt_path = os.path.join(ckpt_dir, f"{cell_id}.ckpt.pt")
    stop_file = args.stop_file or os.path.join(ckpt_dir, "STOP")

    rec = run_two_arm_cell(
        cell_id=cell_id, steps=args.steps, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, lr=args.lr, warmup_steps=args.warmup_steps,
        ceiling_gpuh=args.ceiling_gpuh, seed=args.seed, device=args.device,
        out_path=args.out, ckpt_path=ckpt_path, stop_file=stop_file,
        ckpt_every=args.ckpt_every, eval_every=args.eval_every,
        teacher_force_operator=args.teacher_force_operator)
    return 0


if __name__ == "__main__":
    sys.exit(main())
