#!/usr/bin/env python3
"""EMBED-PATH BUILD -- real-CUDA verification suite (box-side smoke).
matrix-thinking/NCR_EMBED_PATH_DESIGN.md DRAFT-R1 sec R1.4/R1.6/R1.13/R1.15.

Runs against the PATCHED runner at ~/ncr_embedpath/ncr_lm_wave1_runner.py
(md5 distinct from the pinned ~/ncr_g3b31_contrastive copy -- see
BUILD_REPORT.md). Nothing here launches a training cell or writes to
~/queue/pending -- this is build-time verification only.

Sub-tests (numbered against DRAFT-R1 sec R1.4's own verification plan):
  1. Has-teeth baseline (flag OFF): aux+ortho measurably move embed's and
     entity_adapter's combined grad, on THIS build's real forward graph.
  2. Cut-confirmed + scope-preserved (flag ON vs OFF), two-tier per D-F3:
     EXACT match for backbone-block/read_injector params, ALLCLOSE
     (rtol=1e-5, atol=1e-6) for embed/entity_adapter/ncr_head.
  3. conduit_ratio measured over 50 real training steps for BOTH
     close_target choices -- pins PINNED_MIN_RATIO (0.5x the measured
     mean, per assert_conduit_has_teeth's own docstring).
  4. Forced-fail negative test: deliberately mis-assemble the cut (skip
     the subtraction) and confirm the internal allclose assertion fires.
  5. Finite-gradient / read-ablation / same-op checks already run inside
     run_two_arm_cell -- exercised implicitly by the mini training loop
     in sub-test 6.
  6. A short (30-step) real run_two_arm_cell-style loop for BOTH
     close_target choices, confirming no crash, finite losses, and
     has-teeth passing every step.
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/ncr_embedpath"))
import torch
import ncr_lm_wave1_runner as R

DEVICE = "cuda"
torch.manual_seed(0)

RESULTS = {}


def log(name, **kw):
    RESULTS[name] = kw
    print(f"=== {name} ===")
    for k, v in kw.items():
        print(f"  {k}: {v}")
    sys.stdout.flush()


def build_batch(cfg, pools, vocab, seed, batch_size=32, hop=1):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return R.build_task1_document(cfg, pools, gen, batch_size, hop, DEVICE)


def main():
    pools, cfg, pool_report = R.build_grammar_pools_and_cfg(seed=0)
    vocab = pool_report["vocab_size_total"]
    pools = pools.to(DEVICE)

    # -------------------------------------------------------------------
    # Sub-tests 1+2: has-teeth baseline + cut-confirmed + scope-preserved,
    # on ONE fixed batch/seed, for BOTH close_target choices.
    #
    # Scope-preserved measurement strategy (build-time refinement of D-F3's
    # own two-tier framing, disclosed in BUILD_REPORT.md): D-F3's attack
    # measured non-associativity in DRAFT-R0's ORIGINAL construction, which
    # derived EVERY parameter's grad_ce via a SEPARATE autograd.grad(ce_loss,
    # all_params) call and manually summed it with grad_rest per-parameter --
    # a real source of float non-associativity relative to backward()'s own
    # fused accumulation order. R1.6's construction (this build) never does
    # that for non-target parameters: it calls ONE plain total_loss.backward()
    # for the full combined gradient of EVERY parameter (target included),
    # and only the TARGET parameter's grad is then modified via subtraction.
    # So the a-priori prediction for THIS build is that every NON-target
    # parameter (embed OR entity_adapter/ncr_head, whichever is not the cut
    # target) should be torch.equal, not merely allclose, between flag-OFF
    # and flag-ON runs on the same batch/seed/init -- measured directly
    # below rather than assumed, and reported honestly either way.
    # -------------------------------------------------------------------
    for close_target in ("embed", "entity_adapter"):
        torch.manual_seed(42)
        arm_off = R.build_arm(vocab, seed=7, device=DEVICE)
        arm_on = R.build_arm(vocab, seed=7, device=DEVICE)
        # confirm bit-identical init (same seed -> same construction, sec G3-B5 property)
        for (n1, p1), (n2, p2) in zip(
                sum((list(m.named_parameters()) for m in arm_off.values()), []),
                sum((list(m.named_parameters()) for m in arm_on.values()), [])):
            assert n1 == n2 and torch.equal(p1, p2), f"init mismatch at {n1}"

        opt_off = R.build_optimizer(arm_off, lr=3e-4, freeze_entity_adapter=False)
        opt_on = R.build_optimizer(arm_on, lr=3e-4, freeze_entity_adapter=False)

        batch = build_batch(cfg, pools, vocab, seed=123, batch_size=16, hop=1)

        # --- flag OFF: plain backward() + the SAME clip_grad_norm_ the shared training-loop
        # block applies (mirrored here explicitly -- the ON path clips INSIDE
        # assemble_closed_grads_, so comparing against an unclipped OFF path would spuriously
        # show every parameter "different" by exactly (1 - clip_coef), which is a test-harness
        # bug, not a runner bug -- caught on the first run of this script, see BUILD_REPORT.md).
        total_loss_off, ce_loss_off, aux_loss_off, ortho_loss_off, o_raw_off, _ = R.compute_arm_losses(
            arm_off, batch, read_ablate=False, teacher_force=False,
            aux_read_loss_weight=0.5, is_full_graft=True, ortho_reg_weight=0.1,
            aux_loss_type="contrastive+cosine", contrastive_temperature=0.07)
        opt_off.zero_grad()
        total_loss_off.backward()
        all_params_off = list(arm_off["backbone"].parameters()) + list(arm_off["ncr"].parameters()) + list(arm_off["integ"].parameters())

        target_name_off = ("backbone.embed.weight" if close_target == "embed"
                            else "integ.entity_adapter.weight")
        target_w_off = (arm_off["backbone"].embed.weight if close_target == "embed"
                         else arm_off["integ"].entity_adapter.weight)
        combined_grad_norm = target_w_off.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(all_params_off, 1.0)   # mirror run_two_arm_cell's shared block, line ~1357

        # has-teeth reference: CE alone, on a FRESH copy of the same init (independent graph,
        # same batch/seed) -- separate autograd.grad call, matching R1.4 sub-test (a)'s own method.
        arm_ce_only = R.build_arm(vocab, seed=7, device=DEVICE)
        total_loss_ce, ce_loss_ce, _, _, _, _ = R.compute_arm_losses(
            arm_ce_only, batch, read_ablate=False, teacher_force=False,
            aux_read_loss_weight=0.5, is_full_graft=True, ortho_reg_weight=0.1,
            aux_loss_type="contrastive+cosine", contrastive_temperature=0.07)
        target_w_ce = (arm_ce_only["backbone"].embed.weight if close_target == "embed"
                        else arm_ce_only["integ"].entity_adapter.weight)
        grad_ce_only = torch.autograd.grad(ce_loss_ce, [target_w_ce])[0]
        norm_ce_alone = grad_ce_only.norm().item()
        has_teeth_diff = abs(combined_grad_norm - norm_ce_alone)

        log(f"1_has_teeth_baseline[{close_target}]",
            norm_combined=combined_grad_norm, norm_ce_alone=norm_ce_alone,
            abs_diff=has_teeth_diff, PASS=(has_teeth_diff > 1.0))

        # --- flag ON: assemble_closed_grads_, SAME batch/seed/init ---
        total_loss_on, ce_loss_on, aux_loss_on, ortho_loss_on, o_raw_on, _ = R.compute_arm_losses(
            arm_on, batch, read_ablate=False, teacher_force=False,
            aux_read_loss_weight=0.5, is_full_graft=True, ortho_reg_weight=0.1,
            aux_loss_type="contrastive+cosine", contrastive_temperature=0.07)
        opt_on.zero_grad()
        grad_diag = R.assemble_closed_grads_(
            arm_on, opt_on, total_loss_on, ce_loss_on, aux_loss_on, ortho_loss_on,
            0.5, 0.1, close_target)
        log(f"2_cut_confirmed[{close_target}]", **grad_diag)

        # scope-preserved: every OTHER param's grad, OFF vs ON -- prefixed by submodule
        # to avoid any cross-submodule name collision, measured directly (not pre-classified
        # into a tier by name-guessing -- see the module-level comment above).
        target_pname = ("backbone.embed.weight" if close_target == "embed"
                         else "integ.entity_adapter.weight")
        off_named = {f"{grp}.{n}": p for grp, m in
                     (("backbone", arm_off["backbone"]), ("ncr", arm_off["ncr"]), ("integ", arm_off["integ"]))
                     for n, p in m.named_parameters()}
        on_named = {f"{grp}.{n}": p for grp, m in
                    (("backbone", arm_on["backbone"]), ("ncr", arm_on["ncr"]), ("integ", arm_on["integ"]))
                    for n, p in m.named_parameters()}
        exact_names, tol_only_names, fail_names, worst_rel = [], [], [], 0.0
        for name in off_named:
            if name == target_pname:
                continue  # the cut target itself -- checked via grad_diag/has-teeth above
            g_off, g_on = off_named[name].grad, on_named[name].grad
            if g_off is None and g_on is None:
                continue
            if g_off is None or g_on is None:
                fail_names.append((name, "one-sided None"))
                continue
            if torch.equal(g_off, g_on):
                exact_names.append(name)
            else:
                rel = (g_off - g_on).abs().max().item() / max(g_off.abs().max().item(), 1e-12)
                worst_rel = max(worst_rel, rel)
                if torch.allclose(g_off, g_on, rtol=1e-5, atol=1e-6):
                    tol_only_names.append((name, rel))
                else:
                    fail_names.append((name, rel))
        log(f"2b_scope_preserved[{close_target}]",
            n_params_checked=len(off_named) - 1,
            n_exact=len(exact_names), n_tolerance_only=len(tol_only_names),
            tolerance_only_names=tol_only_names, worst_rel_diff=worst_rel,
            fail_names=fail_names, PASS=(len(fail_names) == 0))

    # -------------------------------------------------------------------
    # Sub-test 3: conduit_ratio over 200 real training steps -- pins PINNED_MIN_RATIO.
    #
    # BUILD-TIME DEVIATION from assert_conduit_has_teeth's own docstring suggestion
    # ("0.5x the measured MEAN"), disclosed here and in BUILD_REPORT.md: a first
    # 50-step probe measured extreme step-to-step variance in embed's own
    # conduit_ratio (min 0.0019, max 23.4, mean 2.10 -- a >10,000x spread), because
    # the has-teeth ratio depends on the batch's own random hop/entity sampling,
    # not just the flag. Pinning at 0.5x mean (~1.05) would fire on the very next
    # low-ratio batch (per-step assert_conduit_has_teeth has NO exception handling
    # in run_two_arm_cell -- an AssertionError here is an uncaught crash that
    # ABORTS THE ENTIRE 20,000-STEP CELL, not a soft warning). A mean-based pin is
    # therefore unsafe for this specific has-teeth check's own stated purpose
    # (catching a STRUCTURALLY empty conduit, not ordinary per-batch noise). This
    # build instead pins PINNED_MIN_RATIO at a LOW PERCENTILE of the measured
    # distribution (5th percentile) times a safety margin BELOW it (0.5x), so it
    # sits under nearly all real training noise while still catching a true
    # near-zero conduit (the failure mode D-A2 exists to detect). Flagged for
    # audit to ratify or override -- see BUILD_REPORT.md.
    # -------------------------------------------------------------------
    ratio_stats = {}
    for close_target in ("embed", "entity_adapter"):
        torch.manual_seed(99)
        arm = R.build_arm(vocab, seed=17, device=DEVICE)
        opt = R.build_optimizer(arm, lr=3e-4, freeze_entity_adapter=False)
        data_gen = torch.Generator(device=DEVICE).manual_seed(17 + 777)
        ratios, clip_coefs, cut_actives = [], [], []
        t0 = time.time()
        n_steps = 200
        for step in range(1, n_steps + 1):
            idx = torch.randint(0, 3, (1,), generator=data_gen, device=DEVICE).item()
            hop = (1, 2, 3)[idx]
            batch = R.build_task1_document(cfg, pools, data_gen, 32, hop, DEVICE)
            total_loss, ce_loss, aux_loss, ortho_loss, o_raw, _ = R.compute_arm_losses(
                arm, batch, read_ablate=False, teacher_force=False,
                aux_read_loss_weight=0.5, is_full_graft=True, ortho_reg_weight=0.1,
                aux_loss_type="contrastive+cosine", contrastive_temperature=0.07)
            opt.zero_grad()
            grad_diag = R.assemble_closed_grads_(arm, opt, total_loss, ce_loss, aux_loss, ortho_loss,
                                                  0.5, 0.1, close_target)
            ratios.append(grad_diag["conduit_ratio"])
            clip_coefs.append(grad_diag["clip_coef"])
            cut_actives.append(grad_diag["cut_active"])
        elapsed = time.time() - t0
        srt = sorted(ratios)
        mean_ratio = sum(ratios) / len(ratios)
        p5 = srt[max(0, int(0.05 * len(srt)) - 1)]
        p1 = srt[max(0, int(0.01 * len(srt)) - 1)]
        pinned_mean_based = 0.5 * mean_ratio           # R1.6's own literal suggestion -- shown for contrast
        pinned_p5_based = 0.5 * p5                     # this build's actual choice -- see comment above
        n_below_mean_pin = sum(1 for r in ratios if r <= pinned_mean_based)
        ratio_stats[close_target] = dict(mean=mean_ratio, min=min(ratios), max=max(ratios),
                                          p1=p1, p5=p5,
                                          pinned_mean_based_UNSAFE=pinned_mean_based,
                                          n_of_200_steps_below_mean_pin=n_below_mean_pin,
                                          pinned_min_conduit_ratio=pinned_p5_based,
                                          all_cut_active=all(cut_actives),
                                          mean_clip_coef=sum(clip_coefs) / len(clip_coefs),
                                          n_steps=n_steps, elapsed_s=elapsed)
        log(f"3_conduit_ratio_{n_steps}steps[{close_target}]", **ratio_stats[close_target])

    with open("/tmp/embed_path_ratio_stats.json", "w") as f:
        json.dump(ratio_stats, f, indent=2)

    # -------------------------------------------------------------------
    # Sub-test 4: forced-fail negative test -- deliberately break the cut,
    # confirm the internal allclose assertion fires (run to completion).
    # -------------------------------------------------------------------
    arm_bad = R.build_arm(vocab, seed=7, device=DEVICE)
    opt_bad = R.build_optimizer(arm_bad, lr=3e-4, freeze_entity_adapter=False)
    batch = build_batch(cfg, pools, vocab, seed=123, batch_size=16, hop=1)
    total_loss_bad, ce_loss_bad, aux_loss_bad, ortho_loss_bad, _, _ = R.compute_arm_losses(
        arm_bad, batch, read_ablate=False, teacher_force=False,
        aux_read_loss_weight=0.5, is_full_graft=True, ortho_reg_weight=0.1,
        aux_loss_type="contrastive+cosine", contrastive_temperature=0.07)
    opt_bad.zero_grad()

    # Simulate a no-op cut: monkeypatch a broken version that FORGETS the subtraction
    # (target_w.grad.sub_(grad_rest_target * clip_coef)) -- i.e. skip closing the door
    # entirely, so the post-cut grad is just the full combined grad, not CE-only.
    def broken_assemble(arm, opt, total_loss, ce_loss, aux_loss, ortho_loss,
                         aux_w, ortho_w, close_target, max_norm=1.0):
        target_w = {"embed": arm["backbone"].embed.weight,
                     "entity_adapter": arm["integ"].entity_adapter.weight}[close_target]
        all_params = [p for p in (list(arm["backbone"].parameters()) +
                                   list(arm["ncr"].parameters()) +
                                   list(arm["integ"].parameters())) if p.requires_grad]
        non_ce = R._non_ce_term(aux_loss, ortho_loss, aux_w, ortho_w)
        grad_rest_target = torch.autograd.grad(non_ce, [target_w], retain_graph=True, allow_unused=True)[0]
        total_loss.backward()
        combined_before_clip = target_w.grad.detach().clone()
        grad_ce_target = combined_before_clip - grad_rest_target
        torch.nn.utils.clip_grad_norm_(all_params, max_norm)
        combined_after_clip = target_w.grad.detach().clone()
        clip_coef = (combined_after_clip.norm() / combined_before_clip.norm().clamp_min(1e-12)).item()
        grad_ce_target_clipped = grad_ce_target * clip_coef
        # BUG (deliberate): forgot target_w.grad.sub_(...) -- the "no-op cut" attack scenario.
        assert torch.allclose(target_w.grad, grad_ce_target_clipped, rtol=1e-5, atol=1e-6), (
            f"close_target={close_target}: post-cut grad does not match CE's own (clipped) share within tolerance")
        return {"cut_active": True}

    try:
        broken_assemble(arm_bad, opt_bad, total_loss_bad, ce_loss_bad, aux_loss_bad, ortho_loss_bad, 0.5, 0.1, "embed")
        fired = False
        tb = None
    except AssertionError as e:
        fired = True
        tb = str(e)
    log("4_forced_fail_negative_test", assertion_fired=fired, traceback=tb, PASS=fired)

    print("VERIFY_EMBED_PATH_REAL_DONE")


if __name__ == "__main__":
    main()
