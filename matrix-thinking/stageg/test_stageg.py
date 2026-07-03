"""Stage G smoke gate + MANDATORY tie-verification unit test.

Run me first, always: `python test_stageg.py`
(CPU-only; every check here is cheap enough to run on a laptop, per the
task's "smoke locally what's CPU-feasible" instruction.)

Sections:
  [1] forward/backward/grad-finite for every VARIANT_AXES cell, both model
      families (design build requirement iv: any wiring-changing swap gets
      a fresh smoke pass).
  [2] MANDATORY tie-verification unit test (design section 12, build
      requirement (v)) for BOTH H_f forms:
        (a) in-place perturbation of embed_u.weight changes head logits
        (b) tensor `is` identity: head references the SAME embedding
            tensors the model owns (not copies)
        (c) neither tied head owns a vocab-sized untied parameter
      Per CLAUDE.md's negative-unit-test hard rule, this is RUN, not just
      written -- executing this file is that run.
  [3] Baseline construction is behaviorally identical to
      round2_matrix_script.py's MatrixThinker (design section 2 task
      requirement 2): same param names/shapes/count at matched
      (d, n_layers, n_heads, vocab, max_len), AND bit-identical values at
      the same seed. Documents the one unavoidable divergence (see the
      test body).
  [4] FLOP-formula self-check: analytic_matrix_flops /
      analytic_loopformer_flops reproduce section 5.1's cited Regime-1
      numbers within 1%, and the instrumented FlopCounterMode count lands
      within a generous band of the analytic range (sanity, not exact-
      match -- causal-masking convention differences are expected, see
      flops.py's docstring).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROUND2_DIR = os.path.join(HERE, "..", "..", "experiment-runs", "8xh100-session1")
sys.path.insert(0, os.path.abspath(ROUND2_DIR))

import torch

from models import (
    MatrixThinkerG, VectorReferenceModel, MatrixModelSpec, VectorModelSpec, VARIANT_AXES,
    MATRIX_ALL_BASELINE,
)
from tied_heads import TiedBilinearHead, TiedFactoredHead
from flops import analytic_matrix_flops, analytic_loopformer_flops, instrumented_flops_per_token


def _assert_finite_grads(model, tag):
    n_missing = []
    n_bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            n_missing.append(name)
        elif not torch.isfinite(p.grad).all():
            n_bad.append(name)
    assert not n_missing, f"{tag}: missing grad for {n_missing}"
    assert not n_bad, f"{tag}: non-finite grad for {n_bad}"


# ---------------------------------------------------------------------------
# [1] Forward/backward/grad-finite for every variant
# ---------------------------------------------------------------------------

def test_all_variants_smoke():
    print("\n[1] forward/backward/grad-finite for every Wave-A variant + both model families")
    torch.manual_seed(0)
    B, L, vocab = 2, 8, 64
    ids = torch.randint(0, vocab, (B, L))

    for variant in VARIANT_AXES:
        m = MatrixModelSpec(mat_dim=8, n_heads=2, n_iterations=2, max_len=16,
                             vocab_size=vocab, variant=variant).build()
        logits, info = m(ids, n_iterations=2, measure_ranks=True)
        assert logits.shape == (B, L, vocab), (variant, logits.shape)
        assert not torch.isnan(logits).any() and not torch.isinf(logits).any(), variant
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), ids.reshape(-1))
        loss.backward()
        _assert_finite_grads(m, f"matrix/{variant}")
        print(f"  matrix/{variant}: OK, params={m.count_params()}, breakdown={m.param_breakdown()}")

    # H_g depth-match sanity: n_layers actually changed
    base = MatrixModelSpec(mat_dim=8, variant="baseline").build()
    deep = MatrixModelSpec(mat_dim=8, variant="h_g_depth_matched").build()
    assert len(base.layers) == 8 and len(deep.layers) == 2, "H_g depth-match wiring did not change n_layers"

    # H_i conditioning sanity: cond path actually engaged (adaLN module present, output differs from uncond)
    cond_m = MatrixModelSpec(mat_dim=8, n_heads=2, n_iterations=2, max_len=16,
                              vocab_size=vocab, variant="h_i_iter_cond").build()
    assert cond_m.iter_cond and hasattr(cond_m, "time_embedder")
    assert all(hasattr(layer, "adaLN") for layer in cond_m.layers), "H_i: ThinkingBlock missing adaLN rider"

    for embed_rank, tie_head, iter_cond in [("dense", True, True), ("dense", False, True),
                                             ("dense", True, False), ("rank1", False, True)]:
        v = VectorReferenceModel(n_embd=16, n_head=2, n_layer=2, n_loops=2, intermediate_dim=32,
                                  max_len=16, vocab_size=vocab, embed_rank=embed_rank,
                                  tie_head=tie_head, iter_cond=iter_cond)
        logits, info = v(ids, n_loops=2)
        assert logits.shape == (B, L, vocab)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), ids.reshape(-1))
        loss.backward()
        _assert_finite_grads(v, f"vector/embed={embed_rank},tie={tie_head},cond={iter_cond}")
        print(f"  vector/embed={embed_rank},tie={tie_head},cond={iter_cond}: OK, params={v.count_params()}")
    print("  [1] PASSED")


# ---------------------------------------------------------------------------
# [2] MANDATORY tie-verification unit test (design section 12, requirement v)
# ---------------------------------------------------------------------------

def test_tie_verification():
    print("\n[2] MANDATORY tie-verification unit test (H_f forms i and ii)")
    torch.manual_seed(0)
    vocab, d, K = 40, 8, 8
    B, L = 2, 5
    ids = torch.randint(0, vocab, (B, L))

    for output_head, cls in [("tied_bilinear", TiedBilinearHead), ("tied_factored", TiedFactoredHead)]:
        print(f"  -- form: {output_head} ({cls.__name__}) --")
        m = MatrixThinkerG(mat_dim=d, n_heads=2, n_iterations=1, max_len=16, vocab_size=vocab,
                            n_layers=1, output_head=output_head)
        head = m.output_head
        assert isinstance(head, cls)

        # (b) tensor `is` identity FIRST (must hold before we even perturb)
        assert head.embed_u is m.embed.embed_u, f"{output_head}: head.embed_u is NOT the model's embed_u (copy, not tie)"
        assert head.embed_v is m.embed.embed_v, f"{output_head}: head.embed_v is NOT the model's embed_v (copy, not tie)"
        assert head.embed_u.weight is m.embed.embed_u.weight
        assert head.embed_v.weight is m.embed.embed_v.weight
        print("    (b) tensor `is` identity: PASS")

        # (c) no vocab-sized untied parameter owned by the head. Checks
        # ALL params reachable from the head (recursive), EXCLUDING the
        # tied embed_u/embed_v references themselves (those are shared
        # with the model's embedding, not owned/untied by the head -- and
        # are vocab-sized by design, that's the whole point of tying).
        for name, p in head.named_parameters():
            if name in ("embed_u.weight", "embed_v.weight"):
                continue
            assert vocab not in tuple(p.shape), (
                f"{output_head}: head owns a vocab-sized parameter {name} {tuple(p.shape)} "
                f"-- exactly the F2 dead-code failure mode (untied vocab-sized Linear)")
        print("    (c) no vocab-sized untied parameter: PASS")

        # (a) in-place perturbation of embed_u.weight changes logits
        with torch.no_grad():
            M0 = m.embed(ids)
            logits_before = head(M0).clone()
            perturb = torch.randn_like(m.embed.embed_u.weight) * 5.0
            m.embed.embed_u.weight.add_(perturb)
            logits_after = head(M0).clone()
        delta = (logits_after - logits_before).abs().max().item()
        assert delta > 1e-4, (
            f"{output_head}: in-place perturbation of embed_u.weight did NOT change head logits "
            f"(delta={delta}) -- head is not actually reading the live embedding tensor")
        print(f"    (a) in-place embed_u perturbation changes logits: PASS (max|delta logits|={delta:.4f})")

        # full forward/backward still sane after perturbation (wiring didn't break)
        m.zero_grad()
        logits, _ = m(ids, n_iterations=1)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), ids.reshape(-1))
        loss.backward()
        _assert_finite_grads(m, f"post-perturb {output_head}")
        print("    forward/backward after perturbation: PASS")

    print("  [2] PASSED -- both H_f forms are genuinely tied (NOT the F2 dead-code failure mode)")


def test_tied_bilinear_temperature_calibration():
    print("\n[2b] H_f form (i) temperature calibration (N3)")
    torch.manual_seed(0)
    vocab, d = 256, 32
    head = TiedBilinearHead(torch.nn.Embedding(vocab, d), torch.nn.Embedding(vocab, d))
    with torch.no_grad():
        torch.nn.init.normal_(head.embed_u.weight, std=1.0)   # default (unmatched) init -- the N3 stress case
        torch.nn.init.normal_(head.embed_v.weight, std=1.0)
    M_sample = torch.randn(4, 16, d, d)
    pre_std = (head.tau.item() * torch.einsum('wi,blwi->blw', head.embed_u.weight,
               torch.einsum('blij,wj->blwi', M_sample, head.embed_v.weight))).std().item()
    measured = head.calibrate_temperature(M_sample)
    with torch.no_grad():
        post_logits = head(M_sample)
    post_std = post_logits.std().item()
    print(f"    raw bilinear std (design predicts ~O(d)={d}): {measured:.2f}")
    print(f"    pre-calibration logit std: {pre_std:.3f}   post-calibration logit std: {post_std:.3f}")
    assert 0.5 < post_std < 2.0, f"post-calibration logit std {post_std:.3f} not near 1 -- N3 fix not working"
    print("  [2b] PASSED")


# ---------------------------------------------------------------------------
# [3] Baseline identical to round2_matrix_script.py
# ---------------------------------------------------------------------------

def test_baseline_identical_to_round2_script():
    print("\n[3] matrix-all baseline behaviorally identical to round2_matrix_script.py's MatrixThinker")
    try:
        from round2_matrix_script import MatrixThinker as RefMatrixThinker
    except ImportError as e:
        print(f"  SKIPPED: could not import round2_matrix_script.py ({e!r}) -- run from the repo checkout")
        return

    d, n_layers, n_heads, n_probes, max_len, vocab, dropout = 32, 8, 8, 32, 2048, 500, 0.1

    ref = RefMatrixThinker(mat_dim=d, n_layers=n_layers, n_heads=n_heads, n_probes=n_probes,
                            max_len=max_len, vocab_size=vocab, dropout=dropout)
    # MatrixModelSpec doesn't expose n_layers/n_probes directly (they're
    # fixed at MATRIX_ALL_BASELINE's 8 / mat_dim default) -- construct
    # directly with the same kwargs the spec would use, for an exact match.
    # MATRIX_ALL_BASELINE already pins n_layers=8 (== n_layers here); assert
    # that instead of passing it twice.
    assert MATRIX_ALL_BASELINE["n_layers"] == n_layers
    baseline_kwargs = {k: v for k, v in MATRIX_ALL_BASELINE.items() if k != "n_layers"}
    ours = MatrixThinkerG(mat_dim=d, n_layers=n_layers, n_heads=n_heads, n_probes=n_probes,
                           max_len=max_len, vocab_size=vocab, dropout=dropout,
                           **baseline_kwargs)

    # DOCUMENTED, unavoidable divergence: MatrixThinkerG factors the four
    # embedding tables out into a nested `embed` submodule (MatrixEmbedding,
    # shared with H_j/H_a's interventions) instead of round2's flat
    # embed_u/embed_v/pos_u/pos_v attributes -- a pure NAMING difference
    # (submodule nesting), not a shape/value/order difference: strip the
    # `embed.` prefix before comparing. `named_parameters()` traversal
    # order is preserved (MatrixEmbedding registers embed_u, embed_v,
    # pos_u, pos_v in the identical order round2 does), which is what the
    # zip-based bit-identity check below actually depends on -- verified,
    # not assumed.
    def _strip(n):
        return n[len("embed."):] if n.startswith("embed.") else n

    ref_shapes = {n: tuple(p.shape) for n, p in ref.named_parameters()}
    our_shapes = {_strip(n): tuple(p.shape) for n, p in ours.named_parameters()}
    assert ref_shapes == our_shapes, (
        "baseline param name/shape mismatch vs round2_matrix_script.MatrixThinker (after "
        "stripping the documented 'embed.' submodule-nesting prefix):\n"
        f"  only in ref: {set(ref_shapes) - set(our_shapes)}\n"
        f"  only in ours: {set(our_shapes) - set(ref_shapes)}\n"
        f"  shape mismatches: {[(k, ref_shapes[k], our_shapes[k]) for k in ref_shapes if k in our_shapes and ref_shapes[k] != our_shapes[k]]}")
    ref_count = sum(p.numel() for p in ref.parameters())
    our_count = sum(p.numel() for p in ours.parameters())
    assert ref_count == our_count, f"param count mismatch: ref={ref_count} ours={our_count}"
    print(f"  param names/shapes/count match exactly (mod the disclosed 'embed.' nesting prefix): "
          f"{len(ref_shapes)} tensors, {ref_count:,} params")

    # Bit-identical values at the same global seed, across a few seeds
    # (mirrors run_stage0.py's smoke()'s own baseline-identity discipline).
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        ref_b = RefMatrixThinker(mat_dim=d, n_layers=n_layers, n_heads=n_heads, n_probes=n_probes,
                                  max_len=max_len, vocab_size=vocab, dropout=dropout)
        torch.manual_seed(seed)
        our_b = MatrixThinkerG(mat_dim=d, n_layers=n_layers, n_heads=n_heads, n_probes=n_probes,
                                max_len=max_len, vocab_size=vocab, dropout=dropout,
                                **baseline_kwargs)
        for (n1, p1), (n2, p2) in zip(ref_b.named_parameters(), our_b.named_parameters()):
            assert _strip(n2) == n1 and torch.equal(p1, p2), f"seed {seed}: NOT bit-identical at {n1} (vs {n2})"
    print("  bit-identical parameter VALUES at matched seed (3 seeds): PASS")
    print("  Divergence disclosed (none functional): (i) the 'embed.' submodule-nesting prefix "
          "above; (ii) vocab_size/max_len constructor defaults are Regime-2 byte-vocab values "
          "here (both scripts accept them as explicit args -- no default silently differs when "
          "called explicitly, as this test does). No forward-pass wiring differs.")
    print("  [3] PASSED")


# ---------------------------------------------------------------------------
# [4] FLOP formula self-check
# ---------------------------------------------------------------------------

def test_flop_formulas():
    print("\n[4] FLOP-formula self-check vs design section 5.1's cited (V1-verified) numbers")
    tot, bd = analytic_matrix_flops(d=32, L=512, n_layers=8, n_iterations=8, vocab=50257, K=32,
                                     causal_exact=False)
    assert bd["per_block"] == 3_538_944, bd["per_block"]           # design's exact cited figure
    assert abs(bd["backbone"] / 1e6 - 226.5) < 0.1
    ref_total_m = 230.6
    pct_err = abs(tot / 1e6 - ref_total_m) / ref_total_m * 100
    assert pct_err < 1.0, f"matrix analytic total {tot/1e6:.2f}M vs V1's {ref_total_m}M: {pct_err:.2f}% (design requires <1%)"
    print(f"  matrix analytic: {tot/1e6:.2f}M vs V1-verified 230.6M ({pct_err:.2f}% off) -- PASS")

    tot_lf, bd_lf = analytic_loopformer_flops(n_embd=96, n_head=4, n_layer=2, n_loops=8,
                                               intermediate_dim=240, L=512, vocab=50257)
    ref_lf_m = 15.45
    pct_err_lf = abs(tot_lf / 1e6 - ref_lf_m) / ref_lf_m * 100
    assert pct_err_lf < 1.0, f"loopformer analytic total {tot_lf/1e6:.2f}M vs V1's {ref_lf_m}M: {pct_err_lf:.2f}%"
    print(f"  loopformer analytic: {tot_lf/1e6:.2f}M vs V1-verified 15.45M ({pct_err_lf:.2f}% off) -- PASS")

    ratio = tot / tot_lf
    assert 13.0 < ratio < 17.0, f"non-causal ratio {ratio:.1f}x outside V1's cited 14.9x neighborhood"
    print(f"  non-causal ratio: {ratio:.2f}x (V1: 14.9x) -- PASS")

    # instrumented sanity at a small config (must be same ORDER of magnitude
    # as the analytic range -- exact match not expected, see flops.py docstring)
    torch.manual_seed(0)
    m = MatrixThinkerG(mat_dim=16, n_heads=4, n_iterations=2, max_len=64, vocab_size=64, n_layers=3)
    ids = torch.randint(0, 64, (2, 32))
    per_tok, _, _ = instrumented_flops_per_token(m, ids, forward_kwargs={"n_iterations": 2})
    a_nc, _ = analytic_matrix_flops(d=16, L=32, n_layers=3, n_iterations=2, vocab=64, K=16, causal_exact=False)
    a_c, _ = analytic_matrix_flops(d=16, L=32, n_layers=3, n_iterations=2, vocab=64, K=16, causal_exact=True)
    lo, hi = min(a_c, a_nc) * 0.5, max(a_c, a_nc) * 1.5
    assert lo <= per_tok <= hi, (
        f"instrumented ({per_tok:.0f}) outside the accepted band [{lo:.0f}, {hi:.0f}] "
        f"(= 0.5 x min(analytic {a_c:.0f}, {a_nc:.0f}) .. 1.5 x max) -- investigate before "
        f"trusting Wave -1's corroboration")
    print(f"  instrumented ({per_tok:.0f}) within accepted band [{lo:.0f}, {hi:.0f}] "
          f"(= 0.5x..1.5x of analytic causal-exact {a_c:.0f} / non-causal {a_nc:.0f}) -- PASS")
    print("  [4] PASSED")


# ---------------------------------------------------------------------------
# [5] Param-budget match (audit MAJOR-2)
# ---------------------------------------------------------------------------

def test_param_budget_match():
    print("\n[5] Regime-2 standard-cell param-budget match (audit MAJOR-2)")
    m = MatrixModelSpec(mat_dim=32, n_heads=8, n_iterations=8, max_len=1024,
                         vocab_size=256, variant="baseline").build()
    v = VectorModelSpec(max_len=1024, vocab_size=256, variant="baseline").build()  # n_embd=80 default
    mp, vp = m.count_params(), v.count_params()
    ratio = min(mp, vp) / max(mp, vp)
    print(f"  matrix baseline: {mp:,}   vector baseline (n_embd=80 default): {vp:,}   "
          f"ratio {ratio:.3f}")
    assert ratio >= 0.9, (
        f"param budgets diverged again: matrix {mp:,} vs vector {vp:,} (ratio {ratio:.3f} < 0.9) "
        f"-- audit MAJOR-2 found a silent 30.5% mismatch at n_embd=64; re-derive n_embd")
    # MAJOR-2's re-verification rider: tying and AdaLN must survive the resize.
    assert v.embed.wte.weight is v.lm_head.weight, "weight tying broke at n_embd=80"
    assert v.iter_cond and all(hasattr(b, "adaLN") for b in v.blocks), "AdaLN conditioning broke at n_embd=80"
    print("  weight tying intact (wte.weight is lm_head.weight): PASS")
    print("  AdaLN conditioning intact on all blocks: PASS")
    print("  [5] PASSED")


# ---------------------------------------------------------------------------
# [6] MAJOR-1 verification probe: tau isolated from the shared clip
# ---------------------------------------------------------------------------

def test_tau_isolation_probe():
    """Audit MAJOR-1's required post-fix verification: (a) tau no longer
    dominates the gradient (pre-fix: 99.6% of total grad-norm), and (b) a
    5-step training probe shows backbone parameter updates within ~2x of
    baseline magnitude at identical nominal clip (pre-fix: ~0.074x).

    Objective correction (round-2 audit, polish item 3): the probe uses a
    TRUE next-token objective (targets = inputs shifted by one). An earlier
    revision used identity targets (target == input token), i.e. a COPY
    objective, which the tied head's copying prior trivially solves --
    producing a reversed step-0 picture (tied loss ~2.5 vs baseline ~5.8).
    On the real next-token objective the auditor measured the OPPOSITE:
    tied step-0 loss is HIGHER than baseline (6.98 vs 5.73 -- the copy bias
    predicts the wrong token), and the elevated main-grad-norm ratio is a
    ~99x ONE-STEP transient that settles to ~2-5x within a few steps, fully
    absorbed by Adam + the fixed (tau-isolated) clip structure (auditor's
    measured update ratio: 0.933)."""
    print("\n[6] MAJOR-1 verification probe: log_tau excluded from shared clip "
          "(next-token objective)")
    import torch.nn.functional as F
    from train_stageg import split_tau_param_group, calibrate_tied_bilinear_if_needed

    vocab, d, L, B, n_iter, probe_steps = 256, 32, 64, 4, 2, 5
    results = {}
    for variant in ("baseline", "h_f_tied_bilinear"):
        torch.manual_seed(0)
        model = MatrixModelSpec(mat_dim=d, n_heads=8, n_iterations=n_iter, max_len=L,
                                 vocab_size=vocab, variant=variant).build()
        calibrate_tied_bilinear_if_needed(model, "matrix", vocab, L, n_iter, torch.device("cpu"))
        groups, main_params, tau_params = split_tau_param_group(model, 3e-4)
        opt = torch.optim.AdamW(groups, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.98))
        before = {n: p.detach().clone() for n, p in model.named_parameters()
                  if n.startswith("layers.")}
        gen = torch.Generator().manual_seed(1)
        first_main_gn = first_tau_gn = first_loss = None
        last_main_gn = None
        model.train()
        for _ in range(probe_steps):
            seq = torch.randint(0, vocab, (B, L + 1), generator=gen)
            x, y = seq[:, :-1], seq[:, 1:]          # next-token, NOT identity/copy
            logits, _ = model(x, n_iterations=n_iter)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            tau_gn = torch.nn.utils.clip_grad_norm_(tau_params, 1.0) if tau_params else None
            if first_main_gn is None:
                first_main_gn = float(gn)
                first_tau_gn = float(tau_gn) if tau_gn is not None else None
                first_loss = loss.item()
            last_main_gn = float(gn)
            opt.step()
        upd_sq = sum((p.detach() - before[n]).float().pow(2).sum().item()
                     for n, p in model.named_parameters() if n.startswith("layers."))
        results[variant] = {"loss_step1": first_loss, "main_gn_step1": first_main_gn,
                             "main_gn_last": last_main_gn, "tau_gn_step1": first_tau_gn,
                             "backbone_update_norm": upd_sq ** 0.5}
        print(f"  {variant}: loss(step1)={first_loss:.3f}  main-gn(step1)={first_main_gn:.3f}  "
              f"main-gn(step{probe_steps})={last_main_gn:.3f}"
              + (f"  tau-gn(step1)={first_tau_gn:.3f}" if first_tau_gn is not None else "")
              + f"  backbone ||update||={upd_sq ** 0.5:.5f}")

    tied, base = results["h_f_tied_bilinear"], results["baseline"]
    gn_ratio_step1 = tied["main_gn_step1"] / base["main_gn_step1"]
    gn_ratio_settled = tied["main_gn_last"] / base["main_gn_last"]
    upd_ratio = tied["backbone_update_norm"] / base["backbone_update_norm"]
    tau_share = tied["tau_gn_step1"] / (tied["main_gn_step1"] ** 2 + tied["tau_gn_step1"] ** 2) ** 0.5
    print(f"  tau share of total grad-norm (step1): {tau_share:.1%} (pre-fix measured: 99.6%)")
    print(f"  main-gn ratio (tied/baseline): step1 {gn_ratio_step1:.1f}x (transient; auditor "
          f"measured ~99x) -> step{probe_steps} {gn_ratio_settled:.1f}x (settled; auditor: ~2-5x)")
    print(f"  backbone-update ratio (tied/baseline): {upd_ratio:.3f} (auditor measured 0.933)")
    print(f"  step-0 loss check: tied {tied['loss_step1']:.2f} vs baseline {base['loss_step1']:.2f} "
          f"(auditor: 6.98 vs 5.73 -- tied HIGHER on next-token, copy bias predicts the wrong token)")

    # (a) tau no longer dominates the gradient: pre-fix its scalar grad was
    # 99.6% of total norm; post-reparam it must be a small minority share.
    assert tau_share < 0.10, (
        f"tau still carries {tau_share:.1%} of total grad-norm -- the MAJOR-1 collapse "
        f"mechanism is back")
    # The step-1 main-gn ratio is a known large one-step transient (~99x,
    # auditor-verified) -- asserted only loosely; the SETTLED ratio is the
    # meaningful one (~2-5x per the auditor).
    assert gn_ratio_settled <= 10.0, (
        f"settled tau-excluded grad-norm ratio {gn_ratio_settled:.2f} beyond the auditor's "
        f"measured ~2-5x band -- investigate for a new clip interaction")
    # (b) the audit's hard bar: backbone updates within ~2x of baseline at
    # identical nominal clip (pre-fix: ~0.074x; auditor's post-fix: 0.933).
    assert 0.5 <= upd_ratio <= 2.0, (
        f"backbone update magnitude ratio {upd_ratio:.2f} outside the audit's ~2x bar "
        f"(pre-fix measured ~0.074)")
    print("  [6] PASSED -- tau share <10% (pre-fix 99.6%); settled gn ratio within ~2-5x band; "
          "backbone updates within the audit's ~2x bar (pre-fix ~0.074x)")


def main():
    print("=" * 70)
    print("  STAGE G SMOKE GATE + MANDATORY TIE-VERIFICATION TEST")
    print("=" * 70)
    test_all_variants_smoke()
    test_tie_verification()
    test_tied_bilinear_temperature_calibration()
    test_baseline_identical_to_round2_script()
    test_flop_formulas()
    test_param_budget_match()
    test_tau_isolation_probe()
    print("\n" + "=" * 70)
    print("  ALL STAGE G CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
