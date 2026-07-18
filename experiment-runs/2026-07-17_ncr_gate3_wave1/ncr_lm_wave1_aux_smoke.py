"""NCR REAL-LM sec G3-B17 DIRECT-READ-SUPERVISION AUX-LOSS SMOKE (build agent,
2026-07-19). matrix-thinking/NCR_REAL_LM_DESIGN.md sec G3-B17: the
coordinator-designed fix for the sec G3-B16 WRITE-LEARNING gap (the
make-or-break came back UNINTERPRETABLE -- both arms floored at the
answer-marginal after 20K steps of CE-only INDIRECT signal, ~32x under the
standalone free-write toy's own convergence budget, AND missing the toy's
own DIRECT cosine read-loss). Real-CUDA smoke of the new
`--aux-read-loss-weight` flag (ncr_lm_wave1_runner.py) BEFORE any GPU
launch (BUILD + SMOKE + STOP, coordinator audits + runs the launch).

Exercises ncr_lm_wave1_runner.py's OWN training code path directly
(compute_arm_losses / aux_read_supervision_loss / build_two_arms /
build_optimizer) -- NOT a reimplementation -- so a PASS here means the
real training loop's aux-loss branch is exercised, not a hand-copied
stand-in that could silently drift from run_two_arm_cell's actual code.

Sub-tests (per the build brief):
  (a) flag OFF (weight=0.0) -> loss path IDENTICAL: total_loss IS (Python
      object identity) ce_loss, no aux op constructed at all -- the
      strongest available proof of "unchanged" (not merely numerically
      equal). Also checks backbone_only with weight=1.0 but
      is_full_graft=False -- arm-gating alone (not weight-gating) protects
      it, the same guarantee the real training loop relies on.
  (b) flag ON (weight=1.0), full_graft arm ONLY -> forward/backward finite
      every step; grad reaches ncr_head (the BindingEncoder / write
      encoder) every step; aux_loss DECREASES over ~100 steps on a
      REPEATED (fixed) batch under a real optimizer + grad clipping
      (mirrors run_two_arm_cell's own per-step training mechanics) --
      proving the signal is learnable, matching the regime that let the
      standalone free-write toy converge.
  (c) the aux target is DETACHED -- an ISOLATED fresh-tensor proof: an
      independent leaf `o` (no graph connection to entity_adapter/embed)
      backward through aux_read_supervision_loss must populate o.grad but
      leave embed.weight.grad and entity_adapter's own params' .grad
      EXACTLY None -- an undetached target would populate both (its own
      entity_adapter(embed(answer_token)) call would be part of the same
      backward graph).
  (d) backbone_only arm stays CE-only regardless of the flag's value AND
      the runner's own read-ablation exact-zero assertion still PASSES on
      it post sub-test (b) (a regression check -- (b) only ever steps
      full_graft's optimizer, backbone_only is untouched throughout).

Run (box only -- chunk_delta_rule has no CPU path; --smoke is a no-op flag
that self-documents the invocation, matching ncr_lm_wave1_smoke.py's own
convention re: this repo's pre-train-gate hook):
  CUDA_VISIBLE_DEVICES=<free-gpu> python3 ncr_lm_wave1_aux_smoke.py --smoke \
      --device cuda --out results/aux_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ncr_lm_wave1_runner as runner        # noqa: E402 (verbatim; the actual training code under test)
import ncr_lm_wave1_smoke as graft          # noqa: E402 (verbatim; NCRIntegration, builders)

TRAIN_HOP = 2           # sec 3.1 Task-1 train range h in {1,2,3}; matches graft.TRAIN_HOP precedent
REPEAT_STEPS = 100      # "~50-100 steps" per the build brief
AUX_LR = 2e-3           # DELIBERATELY higher than production's 3e-4 -- this sub-test is an
                          # overfit-one-batch convergence PROBE (proving the aux signal is
                          # learnable at all within a cheap smoke budget), not a
                          # production-representative training rate; disclosed, not a guess
                          # smuggled in as a real config.
GRAD_CLIP_NORM = 1.0    # matches run_two_arm_cell's own per-step clip_grad_norm_(..., 1.0)

FAILURES: list[str] = []
RESULTS: dict = {}


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    RESULTS[item] = {"passed": ok, "detail": detail}
    if not ok:
        FAILURES.append(item)


def aux_smoke_a_flag_off(device: str, pools, cfg, vocab_size_total: int) -> None:
    """(a) --aux-read-loss-weight 0.0 (the default): compute_arm_losses must return
    aux_loss=None and total_loss IS (object identity) ce_loss -- proves no aux
    computation is constructed at all: the strongest available proof of "loss
    path unchanged" (LITERALLY the same tensor object / computation graph /
    backward pass as pre-G3-B17), not merely numerically equal."""
    torch.manual_seed(100)
    arms = runner.build_two_arms(vocab_size_total, seed=100, device=device)
    gen = torch.Generator(device=device).manual_seed(100)
    batch = graft.build_task1_document(cfg, pools, gen, 8, TRAIN_HOP, device)
    arm = arms["full_graft"]

    total_loss, ce_loss, aux_loss, o_raw = runner.compute_arm_losses(
        arm, batch, read_ablate=False, teacher_force=False, aux_read_loss_weight=0.0, is_full_graft=True)
    identity_ok = total_loss is ce_loss
    aux_none_ok = aux_loss is None
    finite_ok = bool(torch.isfinite(total_loss).item())

    # arm-gating (is_full_graft=False), NOT weight-gating, is what actually protects
    # backbone_only in the real training loop -- prove that holds even with a
    # non-zero weight passed in (the real loop never does this, but the guarantee
    # should not depend on the caller getting the weight right).
    total_loss_bo, ce_loss_bo, aux_loss_bo, _ = runner.compute_arm_losses(
        arms["backbone_only"], batch, read_ablate=True, teacher_force=False,
        aux_read_loss_weight=1.0, is_full_graft=False)
    bo_identity_ok = total_loss_bo is ce_loss_bo
    bo_aux_none_ok = aux_loss_bo is None

    ok = identity_ok and aux_none_ok and finite_ok and bo_identity_ok and bo_aux_none_ok
    _report("aux-smoke (a): flag OFF (weight=0.0) -> total_loss IS ce_loss (object identity, "
            "no aux op constructed), aux_loss is None; backbone_only w/ weight=1.0 but "
            "is_full_graft=False -> SAME (arm-gating alone protects it, independent of weight)",
            ok, f"identity_ok={identity_ok} aux_none_ok={aux_none_ok} finite_ok={finite_ok} "
            f"bo_identity_ok={bo_identity_ok} bo_aux_none_ok={bo_aux_none_ok} "
            f"ce_loss={ce_loss.item():.4f}")
    # co-resident with production (GPU2, ~37GB free, all 8 GPUs 100% util) -- free
    # this sub-test's two-arm build (backbone x2 + ncr x2 + integ x2) immediately,
    # not held past its own scope, matching ncr_lm_wave1_smoke.py's own del+empty_cache
    # convention at the end of every smoke_N function.
    del arms, arm, total_loss, ce_loss, total_loss_bo, ce_loss_bo
    if device == "cuda":
        torch.cuda.empty_cache()


def aux_smoke_b_flag_on_decreasing(device: str, pools, cfg, vocab_size_total: int) -> dict:
    """(b) --aux-read-loss-weight 1.0, full_graft arm ONLY, REPEATED (fixed) batch,
    REPEAT_STEPS real optimizer steps (+ grad clipping, mirroring
    run_two_arm_cell's own per-step mechanics): aux_loss must DECREASE (the
    signal is learnable) and grad must reach ncr_head (the BindingEncoder /
    write encoder) every step."""
    torch.manual_seed(101)
    arms = runner.build_two_arms(vocab_size_total, seed=101, device=device)
    arm = arms["full_graft"]
    opt = runner.build_optimizer(arm, lr=AUX_LR)
    gen = torch.Generator(device=device).manual_seed(101)
    batch = graft.build_task1_document(cfg, pools, gen, 8, TRAIN_HOP, device)   # ONE fixed batch, REUSED every step

    aux_hist, ce_hist = [], []
    n_encoder_with_grad_hist = []
    all_finite = True
    all_params = list(arm["backbone"].parameters()) + list(arm["ncr"].parameters()) + list(arm["integ"].parameters())
    for _step in range(REPEAT_STEPS):
        total_loss, ce_loss, aux_loss, o_raw = runner.compute_arm_losses(
            arm, batch, read_ablate=False, teacher_force=False, aux_read_loss_weight=1.0, is_full_graft=True)
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
        aux_hist.append(aux_loss.item())
        ce_hist.append(ce_loss.item())

    n_encoder_params = sum(1 for _ in arm["ncr"].parameters())
    encoder_grad_every_step = all(n > 0 for n in n_encoder_with_grad_hist)
    mean_first10 = sum(aux_hist[:10]) / 10
    mean_last10 = sum(aux_hist[-10:]) / 10
    decreased = mean_last10 < mean_first10
    ok = all_finite and encoder_grad_every_step and decreased and n_encoder_params > 0

    _report(f"aux-smoke (b): flag ON (weight=1.0), full_graft arm, {REPEAT_STEPS} steps on a "
            "REPEATED fixed batch -- forward/backward finite every step, grad reaches ncr_head "
            "(encoder) every step, aux_loss DECREASES (mean first-10 vs mean last-10)",
            ok, f"aux_loss[0]={aux_hist[0]:.4f} aux_loss[-1]={aux_hist[-1]:.4f} "
            f"mean_first10={mean_first10:.4f} mean_last10={mean_last10:.4f} "
            f"ce_loss[0]={ce_hist[0]:.4f} ce_loss[-1]={ce_hist[-1]:.4f} "
            f"n_encoder_params={n_encoder_params} "
            f"encoder_grad_every_step={encoder_grad_every_step} all_finite={all_finite}")
    RESULTS["_aux_loss_history"] = aux_hist
    RESULTS["_ce_loss_history_during_aux_training"] = ce_hist
    return arms


def aux_smoke_c_target_detached(device: str) -> None:
    """(c) ISOLATED fresh-tensor proof that the target is detached: `o` is an
    INDEPENDENT leaf tensor (requires_grad=True, NO graph connection to
    entity_adapter/embed at all). Backward through aux_read_supervision_loss
    must populate o.grad (the loss DOES depend on o) but leave
    embed.weight.grad and entity_adapter's own parameters' .grad EXACTLY
    None -- if the target were NOT detached, the target's own
    entity_adapter(embed(answer_token)) call would be part of this same
    backward graph and those .grad fields would be populated too."""
    torch.manual_seed(102)
    d_model, d_ncr, vocab = 32, 9, 50
    integ = graft.NCRIntegration(d_model, d_ncr, vocab, adapter="linear", read_inject="add").to(device)
    embed = nn.Embedding(vocab, d_model).to(device)
    B = 5
    o = torch.randn(B, d_ncr, device=device, requires_grad=True)   # INDEPENDENT leaf -- no graph tie to integ/embed
    answer_token = torch.randint(0, vocab, (B,), device=device)

    loss = runner.aux_read_supervision_loss(integ, embed, o, answer_token)
    loss.backward()

    o_grad_ok = o.grad is not None and bool(torch.isfinite(o.grad).all().item())
    embed_grad_none = embed.weight.grad is None
    entity_adapter_grad_none = all(p.grad is None for p in integ.entity_adapter.parameters())

    ok = o_grad_ok and embed_grad_none and entity_adapter_grad_none
    _report("aux-smoke (c): ISOLATED fresh-tensor proof -- o is an independent leaf (no graph "
            "tie to entity_adapter/embed); backward through aux_read_supervision_loss populates "
            "o.grad but leaves embed.weight.grad AND entity_adapter's own params' .grad EXACTLY "
            "None (proves the target is detached -- an undetached target would populate both)",
            ok, f"o_grad_ok={o_grad_ok} embed_grad_none={embed_grad_none} "
            f"entity_adapter_grad_none={entity_adapter_grad_none}")


def aux_smoke_d_backbone_only_unaffected(device: str, pools, cfg, arms: dict) -> None:
    """(d) backbone_only arm stays CE-only regardless of --aux-read-loss-weight's
    value (arm-gating, not weight-gating -- see also sub-test (a)'s single-call
    version of this check). ALSO a regression check: the runner's OWN
    read-ablation exact-zero assertion (assert_read_ablation_is_exact_zero)
    still PASSES on this arm (untouched by sub-test (b), which only ever
    stepped full_graft's optimizer) -- proves this build did not disturb
    that invariant."""
    arm_bo = arms["backbone_only"]
    gen = torch.Generator(device=device).manual_seed(103)
    batch = graft.build_task1_document(cfg, pools, gen, 8, TRAIN_HOP, device)

    total_loss, ce_loss, aux_loss, o_raw = runner.compute_arm_losses(
        arm_bo, batch, read_ablate=True, teacher_force=False, aux_read_loss_weight=1.0, is_full_graft=False)
    ce_only_ok = total_loss is ce_loss and aux_loss is None

    max_diff = runner.assert_read_ablation_is_exact_zero(
        arm_bo["backbone"], arm_bo["ncr"], arm_bo["integ"], batch)
    read_ablation_ok = max_diff == 0.0

    ok = ce_only_ok and read_ablation_ok
    _report("aux-smoke (d): backbone_only arm, --aux-read-loss-weight 1.0 -- stays CE-only "
            "(total_loss IS ce_loss, aux_loss is None) AND the runner's own read-ablation "
            "exact-zero assertion still PASSES on this arm (regression check, untouched by "
            "this build)", ok, f"ce_only_ok={ce_only_ok} read_ablation_max_abs_diff={max_diff:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--smoke", action="store_true",
                     help="no-op: this entire script is a smoke suite -- self-documents "
                          "invocations, matching ncr_lm_wave1_smoke.py's own convention "
                          "(and this repo's pre-train-gate hook convention).")
    args = ap.parse_args()

    print("=" * 70)
    print("NCR REAL-LM sec G3-B17 DIRECT-READ-SUPERVISION AUX-LOSS SMOKE")
    print(f"device={args.device} torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "cuda requested but not available"
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 70)

    t0 = time.time()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)   # co-resident with production -- track OUR peak, not theirs
    aux_smoke_c_target_detached(args.device)     # isolated, no pools/backbone needed -- run first (fast)

    if args.device == "cuda":
        print("  building real GPT-2-tokenizer-verified grammar_rd pools (sec 5.1)...", flush=True)
        pools, cfg, pool_report = graft.build_grammar_pools_and_cfg(seed=0)
        vocab_size_total = pool_report["vocab_size_total"]
        pools = pools.to(args.device)
        RESULTS["_grammar_pool_report"] = pool_report

        aux_smoke_a_flag_off(args.device, pools, cfg, vocab_size_total)
        arms_b = aux_smoke_b_flag_on_decreasing(args.device, pools, cfg, vocab_size_total)
        aux_smoke_d_backbone_only_unaffected(args.device, pools, cfg, arms_b)
        del arms_b
        peak_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
        RESULTS["_peak_mem_allocated_gb"] = peak_gb
        print(f"  peak_mem_allocated (this smoke process only) = {peak_gb:.2f} GB", flush=True)
        torch.cuda.empty_cache()
    else:
        print("device=cpu: chunk_delta_rule has no CPU path -- skipping (a)/(b)/(d) "
              "(backbone-dependent); (c) already ran (isolated, no backbone needed).")

    wall = time.time() - t0
    RESULTS["_wall_clock_sec"] = wall
    RESULTS["_failures"] = FAILURES

    print("=" * 70)
    if FAILURES:
        print(f"AUX SMOKE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
    else:
        print("AUX SMOKE: ALL ITEMS PASSED")
    print(f"wall_clock={wall:.1f}s")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(RESULTS, f, indent=2, default=float)
        print(f"results written to {args.out}")

    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
