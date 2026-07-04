"""test_trackb_smokes.py -- TRACKB_REDESIGN.md Rev 3's deliverable 9: CPU
smokes for every mechanism that IS CPU-testable (no chunk_delta_rule call
anywhere in this file -- the Triton kernel has no CPU path). Run with:

  CUDA_VISIBLE_DEVICES= python3 test_trackb_smokes.py

Every item below imports and exercises the ACTUAL production code
(hard_selectivity_rd.py, lm_pretrain_rd.py's _geo3_lm_select_and_orthogonalize
and DeltaNetLMMixer's constructor asserts, bands_pinned_trackb.py,
wave_neg1_trackb.py, trackb_candidate3.py) -- never a reimplementation
standing in for it. Mirrors lm_pretrain_rd.py's own smoke() idiom: numbered
items, loud prints, hard asserts, a final PASSED banner.
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hard_selectivity_rd as hs
import bands_pinned_trackb as bp
import wave_neg1_trackb as wn1
import trackb_candidate3 as c3
from lm_pretrain_rd import _geo3_lm_select_and_orthogonalize, DeltaNetLMMixer, DeltaNetLM, \
    HARD_SELECT_MECHANISMS


def _mk_content_mask(B, T, exclude_last_n=0):
    m = torch.ones(B, T, dtype=torch.bool)
    if exclude_last_n:
        m[:, -exclude_last_n:] = False
    return m


# ---------------------------------------------------------------------------
# [1] Candidate 1: hard top-K mask correctness (exact cardinality, exact
# zero off-selection) + STE gradient pass-through (dense, even at
# non-selected positions).
# ---------------------------------------------------------------------------

def smoke_hard_topk_mask(verbose=True):
    torch.manual_seed(0)
    B, T, H, chunk_size, k_sel = 2, 128, 1, 64, 16
    beta = torch.sigmoid(torch.randn(B, T, H))
    content_mask = _mk_content_mask(B, T, exclude_last_n=3)   # last 3 positions of the FINAL chunk excluded

    mask, topk_idx, valid_sel = hs.hard_topk_beta_mask(beta, content_mask, chunk_size, k_sel)
    assert mask.shape == (B, T, H)
    n_chunks = T // chunk_size
    mask_c = mask.view(B, n_chunks, chunk_size, H)
    per_chunk_count = mask_c.sum(dim=2)
    assert (per_chunk_count == k_sel).all(), f"expected exactly {k_sel} ones per chunk, got {per_chunk_count}"
    # excluded positions must NEVER be selected
    excl = ~content_mask
    assert (mask[excl] == 0).all(), "an EOT/padding-excluded position was selected"

    beta_soft = beta.clone().requires_grad_(True)
    mask_det, _, _ = hs.hard_topk_beta_mask(beta_soft.detach(), content_mask, chunk_size, k_sel)
    beta_hard = hs.apply_hard_select_ste(beta_soft, mask_det)
    assert torch.equal(beta_hard.detach(), beta_soft.detach() * mask_det), "forward value mismatch"
    beta_hard.sum().backward()
    assert (beta_soft.grad != 0).all(), (
        "STE backward must be DENSE (identity pass-through) -- gradient must be nonzero "
        "at every position, including ones the forward pass masked to zero")
    if verbose:
        print(f"  [1] hard top-K mask: exact k_sel={k_sel} per chunk, EOT-excluded never selected, "
              f"STE gradient dense (all {beta_soft.grad.numel()} positions nonzero)")
    return True


# ---------------------------------------------------------------------------
# [2] Candidate 2: sparsemax exact-zeros + chunk-sum<=1 + gradient
# finiteness + excluded positions never in the support.
# ---------------------------------------------------------------------------

def smoke_sparsemax(verbose=True):
    torch.manual_seed(1)
    B, T, H, chunk_size = 2, 128, 1, 64
    content_mask = _mk_content_mask(B, T, exclude_last_n=5)
    scores = torch.randn(B, T, H, requires_grad=True)
    beta = hs.chunk_sparsemax_beta(scores, content_mask, chunk_size)
    n_chunks = T // chunk_size
    chunk_sums = beta.view(B, n_chunks, chunk_size, H).sum(dim=2)
    assert (chunk_sums <= 1.0 + 1e-5).all(), f"chunk sums must be <=1, got max={chunk_sums.max().item()}"
    assert (beta[~content_mask] == 0).all(), "excluded position has nonzero sparsemax mass"
    assert (beta >= 0).all()
    beta.sum().backward()
    assert torch.isfinite(scores.grad).all(), "non-finite sparsemax gradient"
    # a support-degenerate case: near-tied scores should not NaN
    torch.manual_seed(2)
    scores2 = torch.zeros(1, chunk_size, 1, requires_grad=True)
    beta2 = hs.chunk_sparsemax_beta(scores2, torch.ones(1, chunk_size, dtype=torch.bool), chunk_size)
    beta2.sum().backward()
    assert torch.isfinite(scores2.grad).all(), "tied-score sparsemax produced non-finite gradient"
    if verbose:
        print(f"  [2] sparsemax: chunk sums <=1 (max {chunk_sums.max().item():.4f}), exact zeros at "
              f"excluded positions, gradient finite (incl. the fully-tied-scores edge case)")
    return True


# ---------------------------------------------------------------------------
# [3] M7 comparator: tau schedule shape (1->0 over first 10%, hard 0
# after) + registered numeric equivalence at the tau=0 endpoint (<=1e-6
# max-abs diff vs candidate 1's own forward value).
# ---------------------------------------------------------------------------

def smoke_comparator(verbose=True):
    total_steps = 1000
    assert hs.tau_schedule(1, total_steps) == 1.0 - 1.0 / 100 or abs(hs.tau_schedule(1, total_steps) - 0.99) < 1e-9
    assert hs.tau_schedule(100, total_steps) == 0.0, "tau must be EXACTLY 0 at/after the 10% endpoint"
    assert hs.tau_schedule(999, total_steps) == 0.0
    assert hs.tau_schedule(50, total_steps) == 0.5

    torch.manual_seed(3)
    B, T, H, chunk_size, k_sel = 2, 128, 1, 64, 16
    content_mask = _mk_content_mask(B, T)
    beta_soft = torch.sigmoid(torch.randn(B, T, H))
    max_diff = hs.comparator_endpoint_matches_hard_mask(beta_soft, content_mask, chunk_size, k_sel)
    assert max_diff <= 1e-6, f"comparator tau=0 endpoint diverges from candidate 1 by {max_diff}"
    if verbose:
        print(f"  [3] comparator: tau(1000-step run)=1@step1(~0.99), =0 at step>=100 (10% pin), "
              f"tau=0 endpoint matches candidate 1's forward value (max-abs diff {max_diff:.2e} <= 1e-6)")
    return True


# ---------------------------------------------------------------------------
# [4] Cell 2R / 4R: per-(chunk,step) RNG stream varies across steps,
# reproducible at a fixed (seed,step), AND the continuation contract
# (build-phase note (1)): a "training-time" call and a later "eval-probe"
# call at the SAME step draw identically.
# ---------------------------------------------------------------------------

def smoke_random_topk_and_continuation(verbose=True):
    B, T, H, chunk_size, k_sel = 2, 128, 1, 64, 16
    content_mask = _mk_content_mask(B, T)
    m1, _, _ = hs.random_topk_mask((B, T, H), content_mask, chunk_size, k_sel, seed=0, step=1)
    m2, _, _ = hs.random_topk_mask((B, T, H), content_mask, chunk_size, k_sel, seed=0, step=2)
    m1_again, _, _ = hs.random_topk_mask((B, T, H), content_mask, chunk_size, k_sel, seed=0, step=1)
    assert not torch.equal(m1, m2), "random_topk_mask must vary across training steps (Rev 3 NEW-2)"
    assert torch.equal(m1, m1_again), "random_topk_mask must be reproducible at a fixed (seed,step)"

    # continuation contract: simulate "training at step=500" then "an eval-probe reading the
    # checkpoint's own recorded step=500" -- both MUST draw identically (build-phase note (1)).
    training_time_mask, _, _ = hs.random_topk_mask((B, T, H), content_mask, chunk_size, k_sel,
                                                    seed=7, step=500)
    # the "probe" call site is structurally identical: same function, same (seed,step) -- the
    # continuation contract is enforced BY CONSTRUCTION (derive_step_rng is a pure function of
    # (seed,step) with no other hidden state), verified here as a regression guard.
    eval_probe_mask, _, _ = hs.random_topk_mask((B, T, H), content_mask, chunk_size, k_sel,
                                                 seed=7, step=500)
    assert torch.equal(training_time_mask, eval_probe_mask), (
        "CONTINUATION CONTRACT VIOLATED: an eval-time probe at the same (seed,step) as a "
        "training-time call must draw IDENTICALLY (build-phase note (1))")

    # also verify the underlying generator itself continues correctly across two DIFFERENT
    # Generator objects constructed at the same (seed,step) (derive_step_rng's own contract)
    g1 = hs.derive_step_rng(7, 500)
    g2 = hs.derive_step_rng(7, 500)
    t1 = torch.rand(10, generator=g1)
    t2 = torch.rand(10, generator=g2)
    assert torch.equal(t1, t2), "derive_step_rng(seed,step) must produce identical generator state"
    if verbose:
        print("  [4] Cell 2R random_topk_mask: varies across steps, reproducible per (seed,step), "
              "continuation contract holds (training-time call == eval-probe call at same step)")
    return True


# ---------------------------------------------------------------------------
# [5] sec 2 principle 4: B_pinned renorm + clamp + shortfall + the
# SYMMETRIC BUDGET-PARTIAL classification rule (Rev 3 NEW-1).
# ---------------------------------------------------------------------------

def smoke_b_pinned_renorm(verbose=True):
    torch.manual_seed(4)
    B, T, H, chunk_size, k_sel = 3, 128, 1, 64, 16
    content_mask = _mk_content_mask(B, T)
    beta = torch.sigmoid(torch.randn(B, T, H))
    mask, _, _ = hs.hard_topk_beta_mask(beta, content_mask, chunk_size, k_sel)
    beta_masked = beta * mask

    b_pinned = 0.5   # deliberately tiny -- forces the clamp to bind for most chunks (k_sel=16, clamp=1.0
                      # each -> max representable mass 16, so b_pinned=0.5 is trivially reachable; use a
                      # LARGE b_pinned instead to force clamp-induced shortfall)
    b_pinned_large = float(k_sel) * 2.0   # unreachable under the <=1.0-per-position clamp -- EVERY
                                          # chunk must show shortfall > 0
    renorm, shortfall = hs.renormalize_to_b_pinned(beta_masked, chunk_size, b_pinned_large)
    n_chunks = T // chunk_size
    assert renorm.view(B, n_chunks, chunk_size, H).max() <= 1.0 + 1e-5, "clamp_max=1.0 violated"
    assert (shortfall > 0).all(), "b_pinned_large (2x k_sel) must be unreachable under the clamp everywhere"
    cls = hs.classify_budget_partial(shortfall)
    assert cls["budget_partial"] is True, "an always-shortfall scenario must classify BUDGET-PARTIAL"

    # a trivially-reachable b_pinned (small) must show ~zero shortfall and NOT classify PARTIAL
    renorm2, shortfall2 = hs.renormalize_to_b_pinned(beta_masked, chunk_size, b_pinned)
    assert (shortfall2 < 1e-6).all(), f"small b_pinned should be exactly reachable, got {shortfall2}"
    cls2 = hs.classify_budget_partial(shortfall2)
    assert cls2["budget_partial"] is False

    # a fully-zero chunk (candidate-2-collapse edge case): no NaN/inf, shortfall == 1.0
    beta_zero_chunk = beta_masked.clone()
    beta_zero_chunk[:, :chunk_size, :] = 0.0
    renorm3, shortfall3 = hs.renormalize_to_b_pinned(beta_zero_chunk, chunk_size, b_pinned)
    assert torch.isfinite(renorm3).all() and torch.isfinite(shortfall3).all()
    assert torch.allclose(shortfall3[:, 0, :], torch.ones(B, H)), "zero-mass chunk must show shortfall=1.0"
    if verbose:
        print(f"  [5] B_pinned renorm: clamp respected, shortfall symmetric rule fires correctly "
              f"(PARTIAL={cls['budget_partial']} at unreachable b_pinned, "
              f"{cls2['budget_partial']} at reachable b_pinned), zero-mass-chunk edge case finite")
    return True


# ---------------------------------------------------------------------------
# [6] sec 4.3 metrics: churn rate, TV-distance-from-uniform, support size.
# ---------------------------------------------------------------------------

def smoke_sec43_metrics(verbose=True):
    sel_prev = torch.tensor([[0, 1, 2, 3]])
    sel_curr_same = torch.tensor([[0, 1, 2, 3]])
    sel_curr_disjoint = torch.tensor([[4, 5, 6, 7]])
    sel_curr_half = torch.tensor([[0, 1, 8, 9]])
    assert hs.churn_rate(sel_prev, sel_curr_same, 4).item() == 0.0
    assert hs.churn_rate(sel_prev, sel_curr_disjoint, 4).item() == 1.0
    assert hs.churn_rate(sel_prev, sel_curr_half, 4).item() == 0.5

    uniform_counts = torch.ones(64)
    assert abs(hs.tv_distance_from_uniform(uniform_counts) - 0.0) < 1e-9
    concentrated = torch.zeros(64)
    concentrated[0] = 100
    tv_concentrated = hs.tv_distance_from_uniform(concentrated)
    assert 0.9 < tv_concentrated <= 1.0, f"fully-concentrated TV should be near 1.0, got {tv_concentrated}"

    beta = torch.zeros(1, 64, 1)
    beta[0, :10, 0] = 0.5
    ss = hs.support_size(beta, 64)
    assert ss.item() == 10

    mass = hs.per_chunk_total_mass(beta, 64)
    assert abs(mass.item() - 5.0) < 1e-6
    if verbose:
        print(f"  [6] churn_rate correct at 0/0.5/1.0 overlap fractions; TV-distance 0 at uniform, "
              f"{tv_concentrated:.3f} at full concentration; support_size/per_chunk_total_mass exact")
    return True


# ---------------------------------------------------------------------------
# [7] MC anchor recomputation at (K=32,d=64), cross-checked against the
# closed forms.
# ---------------------------------------------------------------------------

def smoke_mc_anchors(verbose=True):
    result = hs.mc_recompute_anchors(32, 64, n_samples=20_000, seed=0)
    assert abs(result["anchor_random_closed_form"] - 3.937) < 0.01, result["anchor_random_closed_form"]
    assert abs(result["anchor_collapse_closed_form"] - 31.50) < 0.01, result["anchor_collapse_closed_form"]
    # MC estimate should be within a few percent of the closed form at 20k samples
    rel_err = abs(result["anchor_random_mc_minus_closed"]) / result["anchor_random_closed_form"]
    assert rel_err < 0.05, f"MC random anchor {rel_err*100:.2f}% off the closed form -- too loose"
    assert abs(result["anchor_collapse_mc_minus_closed"]) < 1e-9, "collapse anchor is deterministic, must match exactly"
    if verbose:
        print(f"  [7] MC anchors (K=32,d=64): random={result['anchor_random']:.4f} "
              f"(closed {result['anchor_random_closed_form']:.4f}, {rel_err*100:.2f}% off), "
              f"collapse={result['anchor_collapse']:.4f} (closed {result['anchor_collapse_closed_form']:.4f}, exact)")
    return True


# ---------------------------------------------------------------------------
# [8] M6 (Cell 4 composition): forced_topk_idx threading through
# _geo3_lm_select_and_orthogonalize, incl. the EOT-exclusion negative
# smoke (a forced index pointing at an EOT position comes out invalid and
# takes the no-op scatter path).
# ---------------------------------------------------------------------------

def smoke_eot_forced_selection_negative(verbose=True):
    torch.manual_seed(5)
    B, n_chunks, chunk_size, H, d, k_sel = 1, 1, 8, 1, 8, 4
    T = n_chunks * chunk_size
    k_raw = F.normalize(torch.randn(B, T, H, d), dim=-1)
    beta = torch.sigmoid(torch.randn(B, T, H))
    content_mask = torch.ones(B, T, dtype=torch.bool)
    content_mask[0, 6:] = False    # positions 6,7 are EOT/padding

    # force selection to include an EOT position (position 6) among the k_sel=4 selected slots
    forced_idx = torch.tensor([[[[0], [1], [6], [2]]]]).view(B, n_chunks, k_sel, H)
    k_out, diag = _geo3_lm_select_and_orthogonalize(
        k_raw, beta, content_mask, chunk_size, k_sel, n_iter=8, resid_tol=1e-2,
        selection_mode="beta_topk", forced_topk_idx=forced_idx)
    valid_sel = diag["_valid_sel"]     # (B,n_chunks,k_sel,H)
    assert valid_sel[0, 0, 2, 0].item() is False, "a forced index at an EOT position must be INVALID"
    assert valid_sel[0, 0, 0, 0].item() is True and valid_sel[0, 0, 1, 0].item() is True
    # no-op scatter path: k_out at the EOT position must be UNCHANGED from k_raw (never zeroed,
    # never orthogonalized -- lm_pretrain_rd.py's own :407-409 comment)
    assert torch.equal(k_out[0, 6, 0], k_raw[0, 6, 0]), (
        "forced-selection EOT negative smoke FAILED: an EOT position's key was modified, expected "
        "the exact no-op scatter path")
    assert diag["selection_mode"] == "forced"

    # shape mismatch is caught with a clear assert, not a silent broadcast/crash
    bad_idx = torch.zeros(B, n_chunks, k_sel + 1, H, dtype=torch.int64)
    raised = False
    try:
        _geo3_lm_select_and_orthogonalize(k_raw, beta, content_mask, chunk_size, k_sel, 8, 1e-2,
                                           "beta_topk", forced_topk_idx=bad_idx)
    except AssertionError:
        raised = True
    assert raised, "a shape-mismatched forced_topk_idx must raise, not silently misbehave"
    if verbose:
        print("  [8] M6 forced selection: EOT-forced index correctly INVALID + no-op scatter "
              "preserved (k_out unchanged at that position); shape-mismatch guard has teeth")
    return True


def smoke_m6_construction_asserts(verbose=True):
    """DeltaNetLMMixer construction-level checks (CPU-safe -- constructor
    never calls the kernel): M6's hard_select_k_sel==geo3_k_sel
    requirement, and entmax+geo3_active's hard refusal."""
    # positive: matched k_sel, hard_ste + geo3 -- constructs fine
    DeltaNetLMMixer(d_model=32, d_state=64, geo3_active=True, geo3_k_sel=16,
                     hard_select_active=True, hard_select_mechanism="hard_ste", hard_select_k_sel=16)

    # negative: mismatched k_sel
    raised_mismatch = False
    try:
        DeltaNetLMMixer(d_model=32, d_state=64, geo3_active=True, geo3_k_sel=32,
                         hard_select_active=True, hard_select_mechanism="hard_ste", hard_select_k_sel=16)
    except AssertionError:
        raised_mismatch = True
    assert raised_mismatch, "M6 mismatched k_sel guard FAILED to reject"

    # negative: entmax + geo3
    raised_entmax = False
    try:
        DeltaNetLMMixer(d_model=32, d_state=64, geo3_active=True, geo3_k_sel=16,
                         hard_select_active=True, hard_select_mechanism="entmax", hard_select_k_sel=16)
    except AssertionError:
        raised_entmax = True
    assert raised_entmax, "entmax+geo3_active guard FAILED to reject"

    if verbose:
        print("  [8b] DeltaNetLMMixer construction: M6 k_sel-mismatch guard AND entmax+geo3 guard "
              "both have teeth; a valid hard_ste+geo3 composition constructs cleanly")
    return True


# ---------------------------------------------------------------------------
# [9] Cell-3 override stamping -- both paths (override True/False), and
# DEFAULT_GATE_OVERRIDE_REASON's own content.
# ---------------------------------------------------------------------------

def smoke_override_stamping(verbose=True):
    stamp = hs.trackb_override_stamp_payload(timestamp=123.0)
    fields = hs.trackb_assemble_gate_override_fields(stamp)
    assert fields == {"gate_override": True, "gate_override_reason": hs.DEFAULT_GATE_OVERRIDE_REASON,
                       "gate_override_at": 123.0, "claim_tier": "descriptive"}

    fields_none = hs.trackb_assemble_gate_override_fields(None)
    assert fields_none == {"gate_override": False}, (
        "non-override path must write gate_override=False explicitly, never omit the field")

    # the run_trackb_wave.py wrapper: no-launch verdict + no --accept flag -> sys.exit(3)
    import run_trackb_wave as rtw
    exited_3 = False
    try:
        rtw.refuse_or_override_cell3("no_launch_redesign", "/nonexistent.json", accept_override=False)
    except SystemExit as e:
        exited_3 = (e.code == 3)
    assert exited_3, "refuse_or_override_cell3 must sys.exit(3) on no-launch without --accept"

    stamp2 = rtw.refuse_or_override_cell3("no_launch_redesign", "/nonexistent.json", accept_override=True)
    assert stamp2 is not None and stamp2["gate_override"] is True

    stamp3 = rtw.refuse_or_override_cell3("beta_gated_primary", "/nonexistent.json", accept_override=False)
    assert stamp3 is None, "a launch-eligible verdict is not an override case at all"
    if verbose:
        print("  [9] Cell-3 override stamping: both paths (True/False) correct, "
              "run_trackb_wave.refuse_or_override_cell3 exits(3)/proceeds/no-ops as registered")
    return True


# ---------------------------------------------------------------------------
# [10] BANDS_PINNED-TrackB: writer -> validate (hash-check) -> readout
# timestamp assertion, round-tripped against synthetic pilot files, incl.
# the churn_ceiling_for_run max-of-two-nulls rule.
# ---------------------------------------------------------------------------

def smoke_bands_pinned_trackb(verbose=True):
    with tempfile.TemporaryDirectory() as d:
        pilot_path = os.path.join(d, "pilot.json")
        with open(pilot_path, "w") as f:
            f.write('{"complete": true, "steps_completed": 2000}')

        churn_null_a = bp.derive_churn_null_a([0.05, 0.06, 0.04, 0.05, 0.055])
        pos_ceiling = bp.derive_pos_ceiling([0.01, 0.02, 0.015, 0.018, 0.012])
        support_floor = bp.derive_support_floor(6.0)
        mass = torch.tensor([10.0, 12.0, 11.0, 9.0, 13.0])
        b_pinned = bp.derive_b_pinned(mass)
        mc_anchors = hs.mc_recompute_anchors(32, 64, n_samples=5_000, seed=0)
        cell1_ref_32 = {"mean": 9.5, "std": 1.2}

        out_path = os.path.join(d, "BANDS_PINNED-TrackB.json")
        doc = bp.write_bands_pinned_trackb(out_path, churn_null_a, pos_ceiling, support_floor,
                                            b_pinned, mc_anchors, cell1_ref_32, [pilot_path])
        assert os.path.exists(out_path)

        validated = bp.validate_bands_pinned_trackb(out_path)
        assert validated is not None, "a freshly-written, unmodified BANDS_PINNED-TrackB.json must validate"
        assert validated["b_pinned"]["b_pinned"] == b_pinned["b_pinned"]

        # readout assertion: pin precedes a later Wave-1 start -> OK
        bp.assert_blind_not_broken_trackb(validated, [validated["pinned_at"] + 10.0])
        # readout assertion: pin does NOT precede an earlier "Wave-1 start" -> raises
        raised = False
        try:
            bp.assert_blind_not_broken_trackb(validated, [validated["pinned_at"] - 10.0])
        except AssertionError:
            raised = True
        assert raised, "assert_blind_not_broken_trackb failed to catch a broken blind"

        # tamper with the pilot file post-pin -> validation must fail (hash mismatch)
        with open(pilot_path, "w") as f:
            f.write('{"complete": true, "steps_completed": 2000, "TAMPERED": true}')
        assert bp.validate_bands_pinned_trackb(out_path) is None, (
            "a post-pin-modified pilot result JSON must FAIL hash validation, not silently pass")

        # churn_ceiling_for_run: max-of-two-nulls (Rev 3 NEW-4)
        low_null_b = bp.churn_ceiling_for_run(doc, [0.001, 0.002])   # Null B far below Null A
        assert low_null_b["binding_null"] == "A"
        assert abs(low_null_b["ceiling"] - churn_null_a["ceiling"]) < 1e-9
        high_null_b = bp.churn_ceiling_for_run(doc, [0.5, 0.6, 0.55])   # Null B far above Null A
        assert high_null_b["binding_null"] == "B"
        assert high_null_b["ceiling"] > churn_null_a["ceiling"]

        assert bp.classify_selection_degenerate(0.5, low_null_b) is True
        assert bp.classify_selection_degenerate(0.0, low_null_b) is False
        assert bp.classify_support_degenerate(2.0, doc["support_band"]) is True     # below floor
        assert bp.classify_support_degenerate(10.0, doc["support_band"]) is False   # within [floor,32]
        assert bp.classify_positionally_degenerate(0.9, doc["positional_concentration_ceiling"]) is True

    if verbose:
        print("  [10] BANDS_PINNED-TrackB: writer/validate/assert round-trip correct, tamper "
              "detection fires, churn_ceiling_for_run's max-of-two-nulls picks the binding null "
              "correctly in both directions")
    return True


# ---------------------------------------------------------------------------
# [11] Duplicate-key stress-slice builder (Rev 3 NEW-7) + positive-control
# counting logic.
# ---------------------------------------------------------------------------

def smoke_duplicate_key_stress_slice(verbose=True):
    torch.manual_seed(6)
    chunk_size, gram_n = 16, 4
    # construct ONE chunk with 8 positions sharing an IDENTICAL ending 4-gram: an 11-token-long
    # constant run (value 100) at positions 0..10 makes every ending position t in [3,10] (8
    # positions: gram_n-1=3 through 10 inclusive) read tokens[t-3:t+1] == (100,100,100,100)
    # identically -- the rest of the chunk stays random (with overwhelming probability at
    # vocab=5000, non-duplicating).
    chunk = torch.randint(0, 5000, (chunk_size,))
    chunk[0:11] = 100
    tokens = chunk.unsqueeze(0)   # (1, chunk_size)
    n_dup = wn1.chunk_n_dup_max(tokens, chunk_size, gram_n=gram_n)
    assert n_dup.shape == (1, 1)
    assert n_dup.item() == 8, f"expected n_dup_max=8, got {n_dup.item()}"

    # a chunk with no repeats (small collision probability at vocab=50000) reports a low n_dup_max
    torch.manual_seed(7)
    chunk_no_dup = torch.randint(0, 50_000, (chunk_size,))
    n_dup_none = wn1.chunk_n_dup_max(chunk_no_dup.unsqueeze(0), chunk_size, gram_n=gram_n)
    assert n_dup_none.item() <= 2, f"expected near-zero duplication at vocab=50000, got {n_dup_none.item()}"

    # window selection: build a small corpus with ONE stress document, verify it gets selected.
    # select_duplicate_key_stress_windows requires >=seq_len+1 tokens after a doc start
    # (mirrors lm_pretrain_rd.py's own eligible-window convention, sample_geo3_diagnostics etc.),
    # so pad one extra token past the chunk_size-length window under test.
    corpus = torch.cat([chunk, torch.randint(0, 5000, (1,))])
    doc_offsets = torch.tensor([0])
    n_dup_windows = wn1.select_duplicate_key_stress_windows(
        corpus, doc_offsets, seq_len=chunk_size, chunk_size=chunk_size, n_dup_min=8, gram_n=gram_n,
        corpus_name="openr1")
    assert n_dup_windows["n_selected_windows"] == 1
    assert n_dup_windows["selected_n_dup_max"] == [8]

    # positive-control counting: gathered selected rows with >=6 exact duplicates
    n_ep, k_sel, d = 2, 32, 8
    k_selected = torch.randn(n_ep, k_sel, d)
    k_selected[0, :7, :] = k_selected[0, 0:1, :]   # 7 exact duplicates in episode 0
    max_dup = wn1.count_max_duplicate_group_in_selected(k_selected)
    assert max_dup == 7, f"expected max_dup=7, got {max_dup}"

    counter = wn1.NanStabilityProbeCounter(floor_n_calls=3, floor_n_dup=6)
    for _ in range(2):
        counter.observe(k_selected)          # 2 calls meeting the floor (max_dup=7>=6)
    k_no_dup = torch.randn(n_ep, k_sel, d)
    counter.observe(k_no_dup)                # 1 call NOT meeting the floor
    assert counter.n_calls_meeting_floor == 2
    assert counter.is_probative() is False, "2 meeting-floor calls < floor_n_calls=3 -> NOT probative"
    counter.observe(k_selected)
    assert counter.is_probative() is True, "3rd meeting-floor call should cross the floor"
    if verbose:
        print(f"  [11] duplicate-key stress slice: n_dup_max=8 detected correctly on a constructed "
              f"chunk (vs <=2 on a random one), window selection picks exactly the stress doc; "
              f"positive-control counter crosses PROBATIVE only once its floor is met")
    return True


# ---------------------------------------------------------------------------
# [12] Candidate 3 (periodic schedule): preprocessing insertion correctness
# + the periodic hard-beta mask's exact architectural-zero property.
# ---------------------------------------------------------------------------

def smoke_candidate3_periodic(verbose=True):
    torch.manual_seed(8)
    # two tiny "documents"
    tokens = torch.tensor([10, 11, 12, 13, 14, 15, 20, 21, 22, 23])
    doc_offsets = torch.tensor([0, 6])
    write_id = 99999
    new_tokens, write_mask, new_offsets = c3.insert_periodic_write_tokens(tokens, doc_offsets,
                                                                          period_w=2, write_token_id=write_id)
    # doc 0 (len 6): pieces of 2 -> 3 inserted tokens; doc 1 (len 4): pieces of 2 -> 2 inserted tokens
    assert int(write_mask.sum().item()) == 5
    assert new_tokens.numel() == tokens.numel() + 5
    assert (new_tokens[write_mask] == write_id).all()
    # no insertion should straddle the original doc boundary -- verify doc 1 starts correctly
    # (new_offsets[1] should point at the first token of doc 1's content, which is 20's own
    # eventual position, not merely SOME position within doc 0's inserted tail)
    assert new_tokens[new_offsets[1]].item() == 20, (
        f"doc boundary violated: token at new doc-1 start is {new_tokens[new_offsets[1]].item()}, expected 20")

    beta_logit = torch.randn(1, new_tokens.numel(), 1)
    beta = c3.periodic_hard_beta(beta_logit, write_mask)
    assert (beta[:, ~write_mask, :] == 0).all(), "non-write positions must be EXACTLY zero"
    assert torch.equal(beta[:, write_mask, :], beta_logit[:, write_mask, :]), (
        "write positions must carry the UNMODIFIED learned logit")

    grid = c3.sensitivity_period_grid(chunk_size=64, k_sel=32)
    assert grid == sorted(set(grid)) and all(w >= 1 for w in grid) and 2 in grid
    if verbose:
        print(f"  [12] candidate 3: periodic insertion respects doc boundaries "
              f"({int(write_mask.sum().item())} reserved tokens inserted), hard-mask is exactly "
              f"zero off-schedule, exactly the raw logit on-schedule; sensitivity grid={grid}")
    return True


# ---------------------------------------------------------------------------
# [13] run_trackb_wave.py: manifest shapes, budget_guard/disk_space_check
# gate behavior, dry-run preview runs without error.
# ---------------------------------------------------------------------------

def smoke_run_trackb_wave_manifests(verbose=True):
    import run_trackb_wave as rtw

    m_neg1 = rtw.waveNeg1_manifest(steps=2000)
    assert len(m_neg1) == 5, f"expected 5 Wave -1 mechanism probes, got {len(m_neg1)}"
    names = {r["name"] for r in m_neg1}
    assert len(names) == 5, "Wave -1 probe names must be unique"

    factorial = rtw.full_factorial_manifest(steps=100, surviving_mechanisms=("hard_ste",))
    assert len(factorial["2"]) == len(rtw.CORPORA) * len(rtw.SEEDS)
    assert len(factorial["2R"]) == len(rtw.CORPORA) * len(rtw.SEEDS)
    assert len(factorial["3"]) == len(rtw.CORPORA) * len(rtw.SEEDS)
    assert len(factorial["4"]) == len(rtw.CORPORA) * len(rtw.SEEDS)
    for spec in factorial["3"]:
        assert spec.get("requires_override") is True
    for spec in factorial["4"]:
        assert spec["hard_select_k_sel"] == spec["geo3_k_sel"], "M6 composition rule violated in the manifest"

    # budget guard: a tiny projection under headroom passes; an enormous one refuses without override
    cum = rtw.budget_guard(0.001, "smoke-tiny", accept_override=False)
    assert cum == rtw.PROGRAM_SPENT_GPUH + 0.001
    exited = False
    try:
        rtw.budget_guard(1_000_000.0, "smoke-huge", accept_override=False)
    except SystemExit as e:
        exited = (e.code == 5)
    assert exited, "budget_guard must refuse (exit 5) an over-ceiling projection without override"
    cum2 = rtw.budget_guard(1_000_000.0, "smoke-huge-override", accept_override=True)
    assert cum2 > rtw.GPU_H_PROGRAM_CEILING

    # disk_space_check: a nonexistent dir reports ok=False, never raises
    disk = rtw.disk_space_check("/this/path/does/not/exist", projected_bytes=10**12, label="smoke")
    assert disk["ok"] is False and disk["free_bytes"] == 0

    epoch = rtw.epoch_cap_check(corpus_n_tokens=10_000_000, steps=1000, batch_size=32, seq_len=512,
                                 epoch_cap=5)
    assert epoch["ok"] is True

    # dry-run preview must execute end-to-end with no exception (prints only, no launch)
    rtw.dry_run_preview(steps=2000, factorial_steps=100, surviving_mechanisms=("hard_ste",),
                        accept_budget_override=True)

    if verbose:
        print(f"  [13] run_trackb_wave.py: manifests correctly shaped (5 Wave -1 probes; factorial "
              f"cells {len(rtw.CORPORA)}x{len(rtw.SEEDS)} each; Cell 3 override-tagged; Cell 4's "
              f"M6 k_sel match holds); budget_guard/disk_space_check/epoch_cap_check gate correctly; "
              f"dry-run preview runs end-to-end")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_ALL_SMOKES = [
    smoke_hard_topk_mask,
    smoke_sparsemax,
    smoke_comparator,
    smoke_random_topk_and_continuation,
    smoke_b_pinned_renorm,
    smoke_sec43_metrics,
    smoke_mc_anchors,
    smoke_eot_forced_selection_negative,
    smoke_m6_construction_asserts,
    smoke_override_stamping,
    smoke_bands_pinned_trackb,
    smoke_duplicate_key_stress_slice,
    smoke_candidate3_periodic,
    smoke_run_trackb_wave_manifests,
]


def main():
    print("=" * 70)
    print("  TRACK B HARD-SELECTIVITY WAVE -- CPU SMOKE SUITE")
    print("  (TRACKB_REDESIGN.md Rev 3; run under CUDA_VISIBLE_DEVICES= -- no kernel calls here)")
    print("=" * 70)
    for fn in _ALL_SMOKES:
        fn(verbose=True)
    print("\n" + "=" * 70)
    print(f"  ALL {len(_ALL_SMOKES)} TRACK B CPU SMOKES PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
