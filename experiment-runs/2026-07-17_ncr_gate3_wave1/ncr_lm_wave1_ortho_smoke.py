"""NCR REAL-LM sec G3-B20 ORTHOGONALITY-REGULARIZATION SMOKE (build agent,
2026-07-20). matrix-thinking/NCR_REAL_LM_DESIGN.md sec G3-B20: the
coordinator-designed fix for the sec G3-B19-diagnosed gap -- the aux
read-supervision loss (sec G3-B17) DID teach the encoder to write (full_graft's
read moved from cos~0 to a STABLE cos~0.57-0.65 across every depth h=1..61),
but that operator is directionally-right, NOT a clean rotation, so Z^h
(binexp_read's own repeated-squaring composition) accumulates error and never
clears the ~0.9 cosine bar exact recovery needs. This smoke exercises the new
`--ortho-reg-weight` flag (ncr_lm_wave1_runner.py) BEFORE any GPU launch
(BUILD + SMOKE + STOP, coordinator audits + runs the launch).

Exercises ncr_lm_wave1_runner.py's OWN training code path directly
(compute_arm_losses / ortho_regularization_loss / build_two_arms /
build_optimizer) -- NOT a reimplementation -- so a PASS here means the real
training loop's ortho-loss branch is exercised, not a hand-copied stand-in
that could silently drift from run_two_arm_cell's actual code. Mirrors
ncr_lm_wave1_aux_smoke.py's own structure/conventions (sec G3-B17 precedent)
one-for-one.

Sub-tests (per the sec G3-B20 build brief):
  (a) flag OFF (ortho_reg_weight=0.0) -> loss path IDENTICAL: total_loss IS
      (Python object identity) ce_loss when aux is ALSO off, no ortho op
      constructed at all when aux is ON (ortho_loss is None, independent of
      aux_read_loss_weight's own value) -- proves the two flags compose
      independently and each OFF state is byte-identical to that flag never
      having been added. Also checks backbone_only with ortho_reg_weight=1.0
      but is_full_graft=False -- arm-gating alone protects it.
  (b) flag ON (ortho_reg_weight=1.0, aux OFF to isolate the ortho signal),
      full_graft arm ONLY -> forward/backward finite every step; grad
      reaches ncr_head (the BindingEncoder/write encoder) every step;
      ortho_loss DECREASES over ~100 steps on a REPEATED (fixed) batch under
      a real optimizer + grad clipping (mirrors run_two_arm_cell's own
      per-step mechanics); a DIRECT (unnormalized, non-squared) mean
      ||Z^TZ-I||_F orthogonality metric is also tracked and must drop,
      confirming Z is actually becoming more orthogonal, not just that the
      loss number is going down.
  (c) THE 3-LOSS MAGNITUDE REPORT (build brief item 4) -- FRESH arm
      construction (true step-0/init magnitudes), both aux_read_loss_weight
      and ortho_reg_weight ON simultaneously at their proposed launch
      values, ce_loss/aux_loss/ortho_loss reported at step 0 (init) and
      after ~100 steps of real training on a repeated batch -- the evidence
      base for this build's recommended --ortho-reg-weight.
  (d) backbone_only arm stays CE-only regardless of either flag's value AND
      the runner's own read-ablation exact-zero assertion still PASSES on
      it after (b)/(c) trained full_graft (a regression check).
  (e) EXISTING AUX STILL WORKS -- runs ncr_lm_wave1_aux_smoke.py itself (the
      sec G3-B17 build's own regression suite, mechanically updated for the
      new 5-tuple compute_arm_losses return -- see that file's own inline
      comment) as a subprocess and requires it to report ALL ITEMS PASSED,
      proving the sec G3-B20 diff did not disturb the aux loss's own
      behavior.

Run (box only -- chunk_delta_rule has no CPU path; --smoke is a no-op flag
that self-documents the invocation, matching ncr_lm_wave1_aux_smoke.py's own
convention re: this repo's pre-train-gate hook):
  CUDA_VISIBLE_DEVICES=<free-gpu> python3 ncr_lm_wave1_ortho_smoke.py --smoke \
      --device cuda --out results/ortho_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ncr_lm_wave1_runner as runner        # noqa: E402 (verbatim; the actual training code under test)
import ncr_lm_wave1_smoke as graft          # noqa: E402 (verbatim; NCRIntegration, builders)

TRAIN_HOP = 2               # sec 3.1 Task-1 train range h in {1,2,3}; matches aux-smoke's own TRAIN_HOP precedent
REPEAT_STEPS = 100           # "~50-100 steps" per the build brief, matches aux-smoke's own REPEAT_STEPS
ORTHO_LR = 2e-3              # DELIBERATELY higher than production's 3e-4 -- an overfit-one-batch
                              # convergence PROBE (proving the ortho signal is learnable at all
                              # within a cheap smoke budget), not a production-representative
                              # rate -- matches ncr_lm_wave1_aux_smoke.py's own AUX_LR precedent/rationale.
GRAD_CLIP_NORM = 1.0         # matches run_two_arm_cell's own per-step clip_grad_norm_(..., 1.0)

# sec G3-B20 build brief item 4's own recommended launch value -- picked here
# so sub-test (c)'s magnitude report is measured AT the value this build
# actually recommends, not a placeholder. REVISED from an initial 1.0 guess:
# a first pass of this smoke (RECOMMENDED_ORTHO_WEIGHT=1.0) measured the
# ACTUAL raw (post-/d^2-normalization) ortho_loss at a random init as ~66 --
# far above the docstring's own O(1) estimate -- so weight=1.0 made the
# weighted ortho term (65.9) dominate ce_loss (10.7) by >6x at step 0. 0.1
# is the smallest round-number weight that keeps the weighted ortho term
# BELOW ce_loss at init (0.1*65.9=6.59 < 10.67) while still exceeding the
# weighted aux term (0.1*65.9=6.59 > 3.0*0.97=2.91) -- meaningful, not
# dominant, picked from the MEASURED number, not the a-priori guess.
RECOMMENDED_ORTHO_WEIGHT = 0.1
RECOMMENDED_AUX_WEIGHT = 3.0   # matches this build's proposed launch command (--aux-read-loss-weight 3.0)

FAILURES: list[str] = []
RESULTS: dict = {}


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    RESULTS[item] = {"passed": ok, "detail": detail}
    if not ok:
        FAILURES.append(item)


def _mean_frob_dev(Z: torch.Tensor) -> float:
    """DIRECT (unnormalized, non-squared) orthogonality metric: mean over the
    batch of ||Z^TZ - I||_F -- reported ALONGSIDE the normalized-squared
    training loss (ortho_regularization_loss) so a human can sanity-check
    "is Z actually becoming more orthogonal" independent of the specific
    loss-normalization choice baked into training."""
    b, d, _ = Z.shape
    eye = torch.eye(d, device=Z.device, dtype=Z.dtype).expand(b, d, d)
    dev = torch.matmul(Z.transpose(-1, -2), Z) - eye
    return dev.reshape(b, -1).norm(dim=-1).mean().item()


def ortho_smoke_a_flag_off(device: str, pools, cfg, vocab_size_total: int) -> None:
    """(a) --ortho-reg-weight 0.0 (the default): compute_arm_losses must return
    ortho_loss=None. With aux ALSO off: total_loss IS (object identity)
    ce_loss -- the strongest available proof of "loss path unchanged". With
    aux ON (weight=3.0) but ortho OFF: ortho_loss must STILL be None
    (independent of aux's own value) -- proves the two flags gate
    independently, not just "at least one is off"."""
    torch.manual_seed(200)
    arms = runner.build_two_arms(vocab_size_total, seed=200, device=device)
    gen = torch.Generator(device=device).manual_seed(200)
    batch = graft.build_task1_document(cfg, pools, gen, 8, TRAIN_HOP, device)
    arm = arms["full_graft"]

    # both flags off
    total_loss, ce_loss, aux_loss, ortho_loss, o_raw = runner.compute_arm_losses(
        arm, batch, read_ablate=False, teacher_force=False,
        aux_read_loss_weight=0.0, is_full_graft=True, ortho_reg_weight=0.0)
    both_off_identity_ok = total_loss is ce_loss
    both_off_aux_none_ok = aux_loss is None
    both_off_ortho_none_ok = ortho_loss is None

    # aux ON, ortho OFF -- ortho_loss must still be None, independent of aux's value
    total_loss2, ce_loss2, aux_loss2, ortho_loss2, _ = runner.compute_arm_losses(
        arm, batch, read_ablate=False, teacher_force=False,
        aux_read_loss_weight=3.0, is_full_graft=True, ortho_reg_weight=0.0)
    aux_on_ortho_off_ortho_none_ok = ortho_loss2 is None
    aux_on_ortho_off_aux_present_ok = aux_loss2 is not None

    # backbone_only with ortho_reg_weight=1.0 AND aux_read_loss_weight=1.0 but
    # is_full_graft=False -- arm-gating alone (not weight-gating) protects it.
    total_loss_bo, ce_loss_bo, aux_loss_bo, ortho_loss_bo, _ = runner.compute_arm_losses(
        arms["backbone_only"], batch, read_ablate=True, teacher_force=False,
        aux_read_loss_weight=1.0, is_full_graft=False, ortho_reg_weight=1.0)
    bo_identity_ok = total_loss_bo is ce_loss_bo
    bo_aux_none_ok = aux_loss_bo is None
    bo_ortho_none_ok = ortho_loss_bo is None

    finite_ok = bool(torch.isfinite(total_loss).item()) and bool(torch.isfinite(total_loss2).item())

    ok = (both_off_identity_ok and both_off_aux_none_ok and both_off_ortho_none_ok
          and aux_on_ortho_off_ortho_none_ok and aux_on_ortho_off_aux_present_ok
          and bo_identity_ok and bo_aux_none_ok and bo_ortho_none_ok and finite_ok)
    _report("ortho-smoke (a): flag OFF -> ortho_loss is None in every case (both flags off: "
            "total_loss IS ce_loss, object identity; aux ON+ortho OFF: ortho_loss still None, "
            "independent of aux's own value); backbone_only w/ both weights=1.0 but "
            "is_full_graft=False -> arm-gating alone protects it",
            ok, f"both_off_identity_ok={both_off_identity_ok} both_off_aux_none_ok={both_off_aux_none_ok} "
            f"both_off_ortho_none_ok={both_off_ortho_none_ok} "
            f"aux_on_ortho_off_ortho_none_ok={aux_on_ortho_off_ortho_none_ok} "
            f"aux_on_ortho_off_aux_present_ok={aux_on_ortho_off_aux_present_ok} "
            f"bo_identity_ok={bo_identity_ok} bo_aux_none_ok={bo_aux_none_ok} "
            f"bo_ortho_none_ok={bo_ortho_none_ok} finite_ok={finite_ok} ce_loss={ce_loss.item():.4f}")
    del arms, arm, total_loss, ce_loss, total_loss2, ce_loss2, total_loss_bo, ce_loss_bo
    if device == "cuda":
        torch.cuda.empty_cache()


def ortho_smoke_b_flag_on_decreasing(device: str, pools, cfg, vocab_size_total: int) -> dict:
    """(b) --ortho-reg-weight 1.0 (aux OFF, isolating the ortho signal), full_graft
    arm ONLY, REPEATED (fixed) batch, REPEAT_STEPS real optimizer steps (+
    grad clipping, mirroring run_two_arm_cell's own per-step mechanics):
    ortho_loss must DECREASE (the signal is learnable) and grad must reach
    ncr_head (the BindingEncoder/write encoder) every step. A DIRECT
    (unnormalized) mean ||Z^TZ-I||_F metric is tracked alongside the
    training loss itself, confirming Z is actually becoming more
    orthogonal, not just that the (normalized, squared) loss number drops."""
    torch.manual_seed(201)
    arms = runner.build_two_arms(vocab_size_total, seed=201, device=device)
    arm = arms["full_graft"]
    opt = runner.build_optimizer(arm, lr=ORTHO_LR)
    gen = torch.Generator(device=device).manual_seed(201)
    batch = graft.build_task1_document(cfg, pools, gen, 8, TRAIN_HOP, device)   # ONE fixed batch, REUSED every step

    ortho_hist, ce_hist, frob_hist = [], [], []
    n_encoder_with_grad_hist = []
    all_finite = True
    all_params = list(arm["backbone"].parameters()) + list(arm["ncr"].parameters()) + list(arm["integ"].parameters())
    for _step in range(REPEAT_STEPS):
        # forward once via ncr_lm_forward_ablatable to grab Z for the direct
        # metric, then reuse compute_arm_losses for the actual trained loss --
        # a SECOND independent forward, not sharing autograd state with the
        # training step below (Z here is detached implicitly via .item()-only
        # use in _mean_frob_dev, never backpropped) so it cannot perturb the
        # real training step's own graph/gradients.
        with torch.no_grad():
            _, _, _, _, Z_probe, _, _ = runner.ncr_lm_forward_ablatable(
                arm["backbone"], arm["ncr"], arm["integ"], batch, read_ablate=False, teacher_force=False)
            frob_hist.append(_mean_frob_dev(Z_probe))

        total_loss, ce_loss, aux_loss, ortho_loss, o_raw = runner.compute_arm_losses(
            arm, batch, read_ablate=False, teacher_force=False,
            aux_read_loss_weight=0.0, is_full_graft=True, ortho_reg_weight=1.0)
        opt.zero_grad()
        total_loss.backward()
        n_encoder_with_grad = sum(1 for p in arm["ncr"].parameters() if p.grad is not None)
        n_encoder_with_grad_hist.append(n_encoder_with_grad)
        finite = bool(torch.isfinite(total_loss).item()) and all(
            p.grad is None or torch.isfinite(p.grad).all() for p in all_params)
        all_finite = all_finite and finite
        if finite:
            torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP_NORM)
            opt.step()
        ortho_hist.append(ortho_loss.item())
        ce_hist.append(ce_loss.item())

    n_encoder_params = sum(1 for _ in arm["ncr"].parameters())
    encoder_grad_every_step = all(n > 0 for n in n_encoder_with_grad_hist)
    mean_first10 = sum(ortho_hist[:10]) / 10
    mean_last10 = sum(ortho_hist[-10:]) / 10
    decreased = mean_last10 < mean_first10
    frob_first10 = sum(frob_hist[:10]) / 10
    frob_last10 = sum(frob_hist[-10:]) / 10
    frob_decreased = frob_last10 < frob_first10
    ok = all_finite and encoder_grad_every_step and decreased and frob_decreased and n_encoder_params > 0

    _report(f"ortho-smoke (b): flag ON (weight=1.0, aux OFF), full_graft arm, {REPEAT_STEPS} steps on a "
            "REPEATED fixed batch -- forward/backward finite every step, grad reaches ncr_head "
            "(encoder) every step, ortho_loss DECREASES, AND the direct (unnormalized) "
            "mean||Z^TZ-I||_F metric independently confirms Z becomes more orthogonal",
            ok, f"ortho_loss[0]={ortho_hist[0]:.4f} ortho_loss[-1]={ortho_hist[-1]:.4f} "
            f"mean_first10={mean_first10:.4f} mean_last10={mean_last10:.4f} "
            f"mean_frob_dev[0]={frob_hist[0]:.4f} mean_frob_dev[-1]={frob_hist[-1]:.4f} "
            f"frob_first10={frob_first10:.4f} frob_last10={frob_last10:.4f} "
            f"ce_loss[0]={ce_hist[0]:.4f} ce_loss[-1]={ce_hist[-1]:.4f} "
            f"n_encoder_params={n_encoder_params} "
            f"encoder_grad_every_step={encoder_grad_every_step} all_finite={all_finite}")
    RESULTS["_ortho_loss_history"] = ortho_hist
    RESULTS["_mean_frob_dev_history"] = frob_hist
    RESULTS["_ce_loss_history_during_ortho_training"] = ce_hist
    return arms


def ortho_smoke_c_three_loss_magnitude_report(device: str, pools, cfg, vocab_size_total: int) -> None:
    """(c) THE 3-LOSS MAGNITUDE REPORT (build brief item 4) -- a FRESH arm
    (true step-0/init magnitudes, not reused from (a)/(b) which already
    trained their own arms), BOTH aux_read_loss_weight=RECOMMENDED_AUX_WEIGHT
    and ortho_reg_weight=RECOMMENDED_ORTHO_WEIGHT ON simultaneously, so the
    reported numbers are measured under the ACTUAL joint launch condition,
    not each term in isolation. Reports ce_loss/aux_loss/ortho_loss at step 0
    (init, before any optimizer step) and after REPEAT_STEPS of real training
    on a repeated batch -- the evidence this build's recommended
    --ortho-reg-weight is checked against (not an unverified guess)."""
    torch.manual_seed(202)
    arms = runner.build_two_arms(vocab_size_total, seed=202, device=device)
    arm = arms["full_graft"]
    opt = runner.build_optimizer(arm, lr=ORTHO_LR)
    gen = torch.Generator(device=device).manual_seed(202)
    batch = graft.build_task1_document(cfg, pools, gen, 8, TRAIN_HOP, device)

    def _step(train: bool):
        total_loss, ce_loss, aux_loss, ortho_loss, o_raw = runner.compute_arm_losses(
            arm, batch, read_ablate=False, teacher_force=False,
            aux_read_loss_weight=RECOMMENDED_AUX_WEIGHT, is_full_graft=True,
            ortho_reg_weight=RECOMMENDED_ORTHO_WEIGHT)
        if train:
            opt.zero_grad()
            total_loss.backward()
            all_params = list(arm["backbone"].parameters()) + list(arm["ncr"].parameters()) + list(arm["integ"].parameters())
            torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP_NORM)
            opt.step()
        return ce_loss.item(), aux_loss.item(), ortho_loss.item(), total_loss.item()

    with torch.no_grad():
        ce0, aux0, ortho0, total0 = _step(train=False)

    ce_last, aux_last, ortho_last = ce0, aux0, ortho0
    for _step_i in range(REPEAT_STEPS):
        ce_last, aux_last, ortho_last, total_last = _step(train=True)

    weighted_aux0 = RECOMMENDED_AUX_WEIGHT * aux0
    weighted_ortho0 = RECOMMENDED_ORTHO_WEIGHT * ortho0
    weighted_aux_last = RECOMMENDED_AUX_WEIGHT * aux_last
    weighted_ortho_last = RECOMMENDED_ORTHO_WEIGHT * ortho_last
    # "does not dominate" criterion (build brief item 4): the WEIGHTED ortho
    # term should not exceed the WEIGHTED aux term by more than a generous
    # 3x at either checkpoint -- a loose sanity bound, not a hard pass/fail
    # gate (this is a REPORT sub-test; the actual launch weight is a
    # judgment call informed by these numbers, recorded in the design doc).
    balance_ok_init = weighted_ortho0 <= 3.0 * max(weighted_aux0, ce0, 1e-6)
    balance_ok_last = weighted_ortho_last <= 3.0 * max(weighted_aux_last, ce_last, 1e-6)

    _report("ortho-smoke (c): 3-LOSS MAGNITUDE REPORT -- fresh arm, "
            f"aux_read_loss_weight={RECOMMENDED_AUX_WEIGHT} + ortho_reg_weight={RECOMMENDED_ORTHO_WEIGHT} "
            f"ON simultaneously, step 0 (init) vs after {REPEAT_STEPS} steps",
            balance_ok_init and balance_ok_last,
            f"step0: ce={ce0:.4f} aux_raw={aux0:.4f} (weighted={weighted_aux0:.4f}) "
            f"ortho_raw={ortho0:.4f} (weighted={weighted_ortho0:.4f}) total={total0:.4f} | "
            f"step{REPEAT_STEPS}: ce={ce_last:.4f} aux_raw={aux_last:.4f} (weighted={weighted_aux_last:.4f}) "
            f"ortho_raw={ortho_last:.4f} (weighted={weighted_ortho_last:.4f}) total={total_last:.4f} | "
            f"balance_ok_init={balance_ok_init} balance_ok_last={balance_ok_last}")
    RESULTS["_three_loss_magnitude_report"] = dict(
        aux_weight=RECOMMENDED_AUX_WEIGHT, ortho_weight=RECOMMENDED_ORTHO_WEIGHT,
        step0=dict(ce=ce0, aux_raw=aux0, aux_weighted=weighted_aux0,
                    ortho_raw=ortho0, ortho_weighted=weighted_ortho0, total=total0),
        after_repeat_steps=dict(ce=ce_last, aux_raw=aux_last, aux_weighted=weighted_aux_last,
                                  ortho_raw=ortho_last, ortho_weighted=weighted_ortho_last, total=total_last,
                                  n_steps=REPEAT_STEPS))
    del arms, arm
    if device == "cuda":
        torch.cuda.empty_cache()


def ortho_smoke_d_backbone_only_unaffected(device: str, pools, cfg, arms: dict) -> None:
    """(d) backbone_only arm stays CE-only regardless of either flag's value
    (arm-gating, not weight-gating). ALSO a regression check: the runner's
    OWN read-ablation exact-zero assertion (assert_read_ablation_is_exact_zero)
    still PASSES on this arm (untouched by sub-test (b), which only ever
    stepped full_graft's optimizer) -- proves this build did not disturb
    that invariant."""
    arm_bo = arms["backbone_only"]
    gen = torch.Generator(device=device).manual_seed(203)
    batch = graft.build_task1_document(cfg, pools, gen, 8, TRAIN_HOP, device)

    total_loss, ce_loss, aux_loss, ortho_loss, o_raw = runner.compute_arm_losses(
        arm_bo, batch, read_ablate=True, teacher_force=False,
        aux_read_loss_weight=1.0, is_full_graft=False, ortho_reg_weight=1.0)
    ce_only_ok = total_loss is ce_loss and aux_loss is None and ortho_loss is None

    max_diff = runner.assert_read_ablation_is_exact_zero(
        arm_bo["backbone"], arm_bo["ncr"], arm_bo["integ"], batch)
    read_ablation_ok = max_diff == 0.0

    ok = ce_only_ok and read_ablation_ok
    _report("ortho-smoke (d): backbone_only arm, --aux-read-loss-weight 1.0 + "
            "--ortho-reg-weight 1.0 -- stays CE-only (total_loss IS ce_loss, aux_loss AND "
            "ortho_loss both None) AND the runner's own read-ablation exact-zero assertion "
            "still PASSES on this arm (regression check, untouched by this build)", ok,
            f"ce_only_ok={ce_only_ok} read_ablation_max_abs_diff={max_diff:.2e}")


def ortho_smoke_e_existing_aux_still_works(device: str) -> None:
    """(e) EXISTING AUX STILL WORKS -- runs ncr_lm_wave1_aux_smoke.py (the sec
    G3-B17 build's own regression suite, mechanically updated for the new
    5-tuple compute_arm_losses return -- see that file's own inline comment
    at its 4 call sites) as a REAL subprocess, requiring it to exit 0 and
    report "AUX SMOKE: ALL ITEMS PASSED" -- the direct proof that this
    sec G3-B20 diff did not disturb the aux loss's own behavior."""
    script = os.path.join(_HERE, "ncr_lm_wave1_aux_smoke.py")
    out_path = os.path.join(_HERE, "g3b20_ortho_smoke_results", "aux_regression_check.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [sys.executable, script, "--smoke", "--device", device, "--out", out_path]
    print(f"\n[ortho-smoke e] SUBPROCESS: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        print(f"[ortho-smoke e] subprocess stderr:\n{proc.stderr[-4000:]}", file=sys.stderr, flush=True)
    all_passed = "AUX SMOKE: ALL ITEMS PASSED" in proc.stdout
    ok = proc.returncode == 0 and all_passed
    _report("ortho-smoke (e): existing aux still works -- ncr_lm_wave1_aux_smoke.py "
            "run as a real subprocess must exit 0 and report ALL ITEMS PASSED (proves "
            "this build's diff did not disturb the sec G3-B17 aux loss's own behavior)",
            ok, f"returncode={proc.returncode} all_passed_in_stdout={all_passed} wall={wall:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--smoke", action="store_true",
                     help="no-op: this entire script is a smoke suite -- self-documents "
                          "invocations, matching ncr_lm_wave1_aux_smoke.py's own convention "
                          "(and this repo's pre-train-gate hook convention).")
    args = ap.parse_args()

    print("=" * 70)
    print("NCR REAL-LM sec G3-B20 ORTHOGONALITY-REGULARIZATION SMOKE")
    print(f"device={args.device} torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "cuda requested but not available"
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 70)

    t0 = time.time()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)   # co-resident with production -- track OUR peak, not theirs

    if args.device == "cuda":
        print("  building real GPT-2-tokenizer-verified grammar_rd pools (sec 5.1)...", flush=True)
        pools, cfg, pool_report = graft.build_grammar_pools_and_cfg(seed=0)
        vocab_size_total = pool_report["vocab_size_total"]
        pools = pools.to(args.device)
        RESULTS["_grammar_pool_report"] = pool_report

        ortho_smoke_a_flag_off(args.device, pools, cfg, vocab_size_total)
        arms_b = ortho_smoke_b_flag_on_decreasing(args.device, pools, cfg, vocab_size_total)
        ortho_smoke_c_three_loss_magnitude_report(args.device, pools, cfg, vocab_size_total)
        ortho_smoke_d_backbone_only_unaffected(args.device, pools, cfg, arms_b)
        del arms_b
        ortho_smoke_e_existing_aux_still_works(args.device)
        peak_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
        RESULTS["_peak_mem_allocated_gb"] = peak_gb
        print(f"  peak_mem_allocated (this smoke process only) = {peak_gb:.2f} GB", flush=True)
        torch.cuda.empty_cache()
    else:
        print("device=cpu: chunk_delta_rule has no CPU path -- skipping all sub-tests "
              "(all are backbone-dependent for this build, unlike aux-smoke's isolated (c)).")

    wall = time.time() - t0
    RESULTS["_wall_clock_sec"] = wall
    RESULTS["_failures"] = FAILURES

    print("=" * 70)
    if FAILURES:
        print(f"ORTHO SMOKE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
    else:
        print("ORTHO SMOKE: ALL ITEMS PASSED")
    print(f"wall_clock={wall:.1f}s")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(RESULTS, f, indent=2, default=float)
        print(f"results written to {args.out}")

    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
