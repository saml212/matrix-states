"""Tests for stage0prime_helpers.py -- run to completion, CPU, fp32, torch
2.8.0, no GPU, no box contact, no `fla` dependency (the whole point of this
module's factoring). Uses the REAL BindingEncoder/NCREarlyLNModel classes
(matrix-thinking/ncr/, matrix-thinking/chapter2/model_v4.py) -- not a
hand-reimplemented stand-in.

item 6's grid here is DELIBERATELY smaller than the production Stage-0'
probe (2 lambda_t values x 1 lr x 400 steps, vs the design's 4x2x>=8000) --
disclosed explicitly: this proves the FUNCTION's logic/wiring is correct
and produces the qualitatively expected behavior (lambda_t=0 leaves the
transverse direction uncontrolled; lambda_t>0 suppresses it) at CPU-
testable scale. The production grid runs on the box, using this SAME
tested code path (stage0prime_eval.py imports this module unmodified).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from stage0prime_helpers import (
    D_NCR, H_NCR, K_NCR,
    build_fresh_encoder,
    discriminability_metrics,
    item_1_2_keygeom,
    item_3_reachability,
    item_4_transverse,
    item_5_ortho_conflict,
    item_6_achievability_probe,
    ortho_regularization_loss,
)

torch.manual_seed(0)
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def teacher_force_operator_stub(keys_v, values_v):
    """Verbatim pinv min-norm reproduction of
    NCRIntegration.teacher_force_operator (runner:348-362) -- the SAME
    formula write_supervision_loss's own test suite uses."""
    k, v = keys_v.detach(), values_v.detach()
    z_t = torch.linalg.pinv(k) @ v
    return z_t.transpose(-1, -2)


def make_episode(B, K=K_NCR, d=D_NCR):
    return torch.randn(B, K, d), torch.randn(B, K, d)


# ---------------------------------------------------------------------------
# discriminability_metrics / ortho_regularization_loss -- verbatim-duplicate
# sanity (shapes + a hand-checkable degenerate case).
# ---------------------------------------------------------------------------
def test_discriminability_metrics_shapes_and_chance():
    B, K, d = 16, 24, 25
    entity_adapter = nn.Identity()
    vocab = 40
    embed = nn.Embedding(vocab, d)
    entity_ids = torch.randint(0, vocab, (B, K))
    tgt_slot = torch.randint(0, K, (B,))
    o = torch.randn(B, d)   # UNCORRELATED with any target -- should read near chance (1/24)
    disc = discriminability_metrics(entity_adapter, embed, o, entity_ids, tgt_slot)
    for f in ("offtarget_margin", "retrieval24_acc", "o_pairwise_cos", "target_pairwise_cos"):
        check(f"discriminability_metrics emits '{f}'", f in disc)
    check("random o reads retrieval24_acc near chance (1/24~0.042)", disc["retrieval24_acc"] < 0.20,
          f"got {disc['retrieval24_acc']:.4f}")


def test_discriminability_metrics_perfect_read():
    """o EXACTLY equal to the true target -> retrieval24_acc must be 1.0
    (as long as targets are distinguishable -- entity_ids sampled WITHOUT
    replacement per row so no row has a duplicate/tied target, which would
    otherwise make argmax's tie-break ambiguous even at a perfect read)."""
    B, K, d = 16, 24, 25
    entity_adapter = nn.Identity()
    vocab = 40
    embed = nn.Embedding(vocab, d)
    entity_ids = torch.stack([torch.randperm(vocab)[:K] for _ in range(B)])
    tgt_slot = torch.randint(0, K, (B,))
    T = entity_adapter(embed(entity_ids))
    o = T.gather(1, tgt_slot.view(B, 1, 1).expand(-1, 1, d)).squeeze(1)   # o = the TRUE target, exactly
    disc = discriminability_metrics(entity_adapter, embed, o, entity_ids, tgt_slot)
    check("o == true target -> retrieval24_acc == 1.0", disc["retrieval24_acc"] == 1.0, disc["retrieval24_acc"])


def test_ortho_loss_zero_at_orthogonal_Z():
    B, d = 8, 25
    Q = torch.linalg.qr(torch.randn(B, d, d))[0]     # orthogonal per-batch-item
    loss = ortho_regularization_loss(Q)
    check("ortho_regularization_loss ~0 at an orthogonal Z", loss.item() < 1e-8, f"loss={loss.item():.3e}")


# ---------------------------------------------------------------------------
# Item 1/2.
# ---------------------------------------------------------------------------
def test_item_1_2_keygeom():
    keys_v, values_v = make_episode(B=32)
    geom = item_1_2_keygeom(teacher_force_operator_stub, keys_v, values_v)
    for f in ("cond_med", "null_gap_med", "within_episode_ratio_p99", "frac_episodes_target_violates_win",
              "L_key_at_Z_ideal_med"):
        check(f"item_1_2_keygeom emits '{f}'", f in geom)
    check("well-conditioned random keys: fraction violating WIN band is 0",
          geom["frac_episodes_target_violates_win"] == 0.0, geom["frac_episodes_target_violates_win"])
    check("well-conditioned random keys: L_key_at_Z_ideal ~0 (Z_ideal IS the zero-residual fit)",
          geom["L_key_at_Z_ideal_med"] < 1e-6, geom["L_key_at_Z_ideal_med"])


def test_item_1_2_pinv_truncation_cliff():
    """M5's own finding, reproduced: make one key progressively collinear
    with another -- cond(keys) blows up and the fraction of episodes whose
    OWN target violates the WIN band should climb off zero."""
    B, K, d = 8, 24, 25
    keys_v, values_v = make_episode(B=B, K=K, d=d)
    keys_v = keys_v.clone()
    keys_v[:, 1, :] = keys_v[:, 0, :] + 1e-6 * torch.randn(B, d)   # near-collinear pair -> huge cond
    geom = item_1_2_keygeom(teacher_force_operator_stub, keys_v, values_v)
    check("near-collinear keys: cond_max is large (M5's cliff)", geom["cond_max"] > 1e4, geom["cond_max"])
    check("near-collinear keys: SOME episode's target now violates the WIN band",
          geom["frac_episodes_target_violates_win"] > 0.0, geom["frac_episodes_target_violates_win"])


# ---------------------------------------------------------------------------
# Item 3 -- the .encoder attribute route (A2) + LayerNorm-affine correction (M4b).
# ---------------------------------------------------------------------------
def test_item_3_uses_encoder_row_out_not_head_row_out():
    head = build_fresh_encoder()
    check("A2: ncr_head has NO row_out attribute directly", not hasattr(head, "row_out"))
    check("A2: ncr_head.encoder DOES have row_out", hasattr(head.encoder, "row_out"))
    check("M4(b): ncr_head.encoder.row_norm has elementwise_affine=True (learnable, NOT fixed-norm)",
          head.encoder.row_norm.elementwise_affine is True)

    keys_v, values_v = make_episode(B=16)
    geom = item_1_2_keygeom(teacher_force_operator_stub, keys_v, values_v)
    reach = item_3_reachability(head.encoder, geom)
    for f in ("cond_row_out", "required_dynamic_range", "absolute_ceiling", "gate_pass"):
        check(f"item_3_reachability emits '{f}'", f in reach)
    check("item_3's required_dynamic_range uses the WITHIN-EPISODE p99 (A3), not the global stat",
          reach["required_dynamic_range"] == geom["within_episode_ratio_p99"])


# ---------------------------------------------------------------------------
# Item 4 -- M10 polarity + M11 generalization.
# ---------------------------------------------------------------------------
def test_item_4_polarity_and_scale():
    B, K, d = 16, 24, 25
    keys_v, values_v = make_episode(B=B, K=K, d=d)
    Z_ideal = teacher_force_operator_stub(keys_v, values_v)
    t4_clean = item_4_transverse(Z_ideal, keys_v)
    check("item_4: Z_ideal itself has transverse_gain ~0", t4_clean["transverse_gain_med"] < 1e-3,
          t4_clean["transverse_gain_med"])
    check("item_4: transverse_gain_exceeds_3 is False for Z_ideal (correct polarity, M10)",
          t4_clean["transverse_gain_exceeds_3"] is False)

    from write_supervision_loss import null_directions
    W = null_directions(keys_v, K=K)
    u = torch.randn(B, d, 1) * 10.0
    Z_bad = Z_ideal + torch.einsum('bpk,bkj->bpj', u, W)
    t4_bad = item_4_transverse(Z_bad, keys_v)
    check("item_4: a large transverse perturbation flips transverse_gain_exceeds_3 to True (M10 has teeth)",
          t4_bad["transverse_gain_exceeds_3"] is True, t4_bad["transverse_gain_med"])


# ---------------------------------------------------------------------------
# Item 5 -- A5 predicate + M8's 1/B fix.
# ---------------------------------------------------------------------------
def test_item_5_1B_correction_and_predicate():
    keys_v, values_v = make_episode(B=64)
    r5 = item_5_ortho_conflict(teacher_force_operator_stub, keys_v, values_v)
    check("item_5 emits grad_norm_med_per_example >= grad_norm_med_as_carded * (B-ish factor)",
          r5["grad_norm_med_per_example"] > r5["grad_norm_med_as_carded"],
          f"per_example={r5['grad_norm_med_per_example']:.4f} as_carded={r5['grad_norm_med_as_carded']:.6f}")
    ratio = r5["grad_norm_med_per_example"] / max(r5["grad_norm_med_as_carded"], 1e-12)
    check("item_5: the correction factor is exactly B=64 (M8's 1/B fix)", abs(ratio - 64.0) < 1e-2, ratio)
    check("item_5 emits a boolean conflict_reproduces per the A5 numeric predicate", isinstance(r5["conflict_reproduces"], bool))
    check("item_5 discloses joint_min_curve_reproduced=False (M8's second substitution, disclosed not silent)",
          r5["joint_min_curve_reproduced"] is False)


# ---------------------------------------------------------------------------
# Item 6 -- A6, the substantive replacement. Reduced grid for CPU speed
# (disclosed above); confirms the qualitative F2 behavior: lambda_t=0
# leaves the transverse direction uncontrolled, lambda_t>0 suppresses it.
# ---------------------------------------------------------------------------
def make_scoring_rig(B, K, d, vocab, seed_offset=0):
    g = torch.Generator().manual_seed(1000 + seed_offset)
    table = torch.randn(vocab, d, generator=g)
    entity_adapter = nn.Identity()
    embed = nn.Embedding.from_pretrained(table, freeze=True)

    def score_fn(o, entity_ids, tgt_slot):
        return discriminability_metrics(entity_adapter, embed, o, entity_ids, tgt_slot)

    keys_v = torch.randn(B, K, d, generator=g)
    value_entity_ids = torch.stack([torch.randperm(vocab, generator=g)[:K] for _ in range(B)])  # (B,K), distinct per row
    values_v = table[value_entity_ids]                                                            # (B,K,d)
    tgt_slot = torch.randint(0, K, (B,), generator=g)
    query_key = keys_v.gather(1, tgt_slot.view(B, 1, 1).expand(-1, 1, d)).squeeze(1)                # bit-identical property
    return keys_v, values_v, value_entity_ids, tgt_slot, query_key, score_fn


def test_item_6_wiring_and_qualitative_lambda_t_effect():
    B_train, B_held, K, d, vocab = 24, 12, K_NCR, D_NCR, 40
    keys_train, values_train, _, _, _, _ = make_scoring_rig(B_train, K, d, vocab, seed_offset=1)
    keys_held1, values_held1, ids_held1, tgt_held1, qkey_held1, score_fn = make_scoring_rig(B_held, K, d, vocab, seed_offset=2)
    keys_held61, values_held61, ids_held61, tgt_held61, qkey_held61, _ = make_scoring_rig(B_held, K, d, vocab, seed_offset=4)
    held_out_by_hop = {
        1: (keys_held1, values_held1, qkey_held1, ids_held1, tgt_held1),
        61: (keys_held61, values_held61, qkey_held61, ids_held61, tgt_held61),
    }

    torch.manual_seed(42)   # pin the fresh-encoder init draw explicitly -- this test's own conclusion
                            # (lambda_t=3.0 < lambda_t=0.0 on held-out Zw_ratio) must not depend on
                            # whatever global RNG state earlier tests happened to leave behind.
    t0 = time.time()
    out = item_6_achievability_probe(
        keys_train, values_train, held_out_by_hop, score_fn,
        lambda_t_grid=(0.0, 3.0), lr_grid=(1e-3,), n_steps=800, log_every=100)
    elapsed = time.time() - t0
    print(f"    (item_6 reduced-grid probe: {elapsed:.1f}s wall, 2 cells x 800 steps, trained ONCE, scored at 2 hops)")

    check("item_6 returns one result per grid cell (2x1=2)", len(out["cells"]) == 2, len(out["cells"]))
    check("item_6 reports the GO/NO-GO gate string", out["stage1_gate"] in ("GO", "NO-GO-ON-CURRENT-BAND"))

    cell_lam0 = next(c for c in out["cells"] if c["lambda_t"] == 0.0)
    cell_lam3 = next(c for c in out["cells"] if c["lambda_t"] == 3.0)
    for c in (cell_lam0, cell_lam3):
        check(f"item_6 cell(lambda_t={c['lambda_t']}) has a non-empty learning curve", len(c["curve"]) > 0)
        check(f"item_6 cell(lambda_t={c['lambda_t']}) scored both hops (1x train, per-hop-matched held-out score)",
              set(c["retrieval_by_hop"].keys()) == {"h=1", "h=61"})

    for h_key in ("h=1", "h=61"):
        check(f"item_6 cell(lambda_t={cell_lam0['lambda_t']}) reports held_out_band2_by_hop['{h_key}']",
              h_key in cell_lam0["held_out_band2_by_hop"])

    check("F2's qualitative effect reproduced (at h=61): lambda_t=3.0 achieves a LOWER held-out Zw_ratio than lambda_t=0.0",
          cell_lam3["held_out_band2_by_hop"]["h=61"]["Zw_ratio_med"] < cell_lam0["held_out_band2_by_hop"]["h=61"]["Zw_ratio_med"],
          f"lam0={cell_lam0['held_out_band2_by_hop']['h=61']['Zw_ratio_med']:.4f} "
          f"lam3={cell_lam3['held_out_band2_by_hop']['h=61']['Zw_ratio_med']:.4f}")
    check("F2's qualitative effect reproduced (at h=1 too, the non-cherry-picking check): "
          "lambda_t=3.0 <= lambda_t=0.0 on held-out Zw_ratio",
          cell_lam3["held_out_band2_by_hop"]["h=1"]["Zw_ratio_med"] <= cell_lam0["held_out_band2_by_hop"]["h=1"]["Zw_ratio_med"],
          f"lam0={cell_lam0['held_out_band2_by_hop']['h=1']['Zw_ratio_med']:.4f} "
          f"lam3={cell_lam3['held_out_band2_by_hop']['h=1']['Zw_ratio_med']:.4f}")


def test_item_6_per_hop_scoring_uses_the_matching_held_out_set():
    """Confirms the h=1 score and the h=61 score genuinely come from
    DIFFERENT held-out tensors (F3.5's fix has teeth). R5 M3 repair
    changed the fixture: both hops now carry VALID (nonzero) held-out
    sets from different seeds, since an all-zero fixture is VOIDed by the
    M3 guard (see the new soundness test below)."""
    B_train, K, d, vocab = 16, K_NCR, D_NCR, 40
    keys_train, values_train, _, _, _, score_fn = make_scoring_rig(B_train, K, d, vocab, seed_offset=5)
    keys_held1, values_held1, ids_held1, tgt_held1, qkey_held1, _ = make_scoring_rig(6, K, d, vocab, seed_offset=7)
    keys_held61, values_held61, ids_held61, tgt_held61, qkey_held61, _ = make_scoring_rig(6, K, d, vocab, seed_offset=6)

    calls = []

    def spy_score_fn(o, entity_ids, tgt_slot):
        calls.append(o.clone())
        return dict(retrieval24_acc=0.0)

    held_out_by_hop = {
        1: (keys_held1, values_held1, qkey_held1, ids_held1, tgt_held1),
        61: (keys_held61, values_held61, qkey_held61, ids_held61, tgt_held61),
    }
    item_6_achievability_probe(keys_train, values_train, held_out_by_hop, spy_score_fn,
                                lambda_t_grid=(1.0,), lr_grid=(1e-3,), n_steps=50, log_every=50)
    check("item_6 called score_fn exactly twice (once per hop)", len(calls) == 2, len(calls))
    check("item_6's h=1 and h=61 scoring calls used DIFFERENT `o` tensors (per-hop held-out sets are actually distinct)",
          not torch.equal(calls[0], calls[1]))


def test_item_6_zero_values_hop_voids_and_blocks_gate():
    """R5 M3 soundness (the assertion the old fixture never made): an
    all-zero held-out hop must VOID -- score_fn never sees it, the hop
    carries VOID=True, and the cell can NOT reach band2 (no free PASS),
    so the stage1_gate cannot read GO off a hop that measured nothing."""
    B_train, K, d, vocab = 16, K_NCR, D_NCR, 40
    keys_train, values_train, _, _, _, _ = make_scoring_rig(B_train, K, d, vocab, seed_offset=5)
    keys_held61, values_held61, ids_held61, tgt_held61, qkey_held61, _ = make_scoring_rig(6, K, d, vocab, seed_offset=6)

    calls = []

    def spy_score_fn(o, entity_ids, tgt_slot):
        calls.append(o.clone())
        return dict(retrieval24_acc=0.0)

    held_out_by_hop = {
        1: (torch.zeros(6, K, d), torch.zeros(6, K, d), torch.zeros(6, d), ids_held61, tgt_held61),
        61: (keys_held61, values_held61, qkey_held61, ids_held61, tgt_held61),
    }
    out = item_6_achievability_probe(keys_train, values_train, held_out_by_hop, spy_score_fn,
                                      lambda_t_grid=(1.0,), lr_grid=(1e-3,), n_steps=50, log_every=50)
    check("zero-values hop VOIDs: score_fn called exactly once (h=61 only)", len(calls) == 1, len(calls))
    cell = out["cells"][0]
    check("VOIDed hop carries VOID=True in band2_by_hop",
          cell["held_out_band2_by_hop"]["h=1"].get("VOID", False) is True)
    check("a cell with a VOIDed hop cannot reach band2 at median",
          cell["reaches_band2_median"] is False)
    check("a cell with a VOIDed hop cannot reach band2 at p90",
          cell["reaches_band2_p90"] is False)
    check("stage1_gate is not GO off a VOIDed hop", out["stage1_gate"] != "GO", out["stage1_gate"])


if __name__ == "__main__":
    test_discriminability_metrics_shapes_and_chance()
    test_discriminability_metrics_perfect_read()
    test_ortho_loss_zero_at_orthogonal_Z()
    test_item_1_2_keygeom()
    test_item_1_2_pinv_truncation_cliff()
    test_item_3_uses_encoder_row_out_not_head_row_out()
    test_item_4_polarity_and_scale()
    test_item_5_1B_correction_and_predicate()
    test_item_6_wiring_and_qualitative_lambda_t_effect()
    test_item_6_per_hop_scoring_uses_the_matching_held_out_set()
    test_item_6_zero_values_hop_voids_and_blocks_gate()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")
