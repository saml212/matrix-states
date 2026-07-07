"""reasoning_link_stage_minus1.py -- REASONING_LINK_DESIGN.md sec 9's Stage -1
self-tests, ALL 14 items, each an EXECUTABLE test with its own registered
pass criterion (never a printed value trusted by eye, per this project's
"a specification that has not been executed is not a passed gate"
convention). Every item imports and exercises `reasoning_link_probe.py`'s
OWN real functions -- never a reimplemented copy -- so a Stage -1 pass
actually certifies the probe's own code, not a parallel stand-in.

PLUS items 15-18 (sec 16.1's PHASE-1B natural-language-surface-form
addition, registered by this feature's own BUILD brief rather than by sec 9
itself, since sec 9 predates sec 16.1): Candidate B's succession-verb
tokenizer verification, the natural-template T_bind/K_min re-derivation,
and the natural-query causality/q_conv1d-hook re-checks -- plus a
gate-enforcement negative test (test_extra_gate_enforcement_negative_test,
alongside the sec-12 outcome-routing extra check) closing sec 15.2's own
DISCREPANCY finding for the Phase-1b chain.

PLUS item 19 (mini-audit FIX-1, this build round, closing MAJOR-1 + MAJOR-2
of the Phase-1b gate build's mini-audit): natural-query SEMANTIC-CONTENT
verification. Items 15-18 verify the natural query's tokenizer validity,
shape, causal-locality, and hook-vs-direct-call agreement -- but never (a)
drive it through `measure_cell_all_h`'s own `query_mode` DISPATCH (they all
call `build_natural_query_tokens` directly), and never (b) check the
window's DECODED CONTENT at all (only its shape/numerics). Item 19 closes
both gaps with one assertion set: it exercises the real dispatch (catching
a typo'd string comparison that would silently fall through to the marker
branch) and decodes the actual window content (catching a KEY/REL order
swap inside the builder, which no shape/causality/hook check can see).

CPU-runnable throughout. Items needing "a real checkpoint" (5, 6, 11, 13,
14, 17, 18, 19) use a small, freshly-initialized `DeltaNetLM` instance (via this
repo's own CPU fla-stub, see `reasoning_link_probe.py`'s module docstring)
as a stand-in "checkpoint" -- mirrors `lm_attractor_probe_rd.py`'s own
smoke() item [6] convention exactly ("a TINY (but REAL-vocab-sized)
SYNTHETIC (untrained) model ... NOT a claim about real key geometry"). What
this DOES verify for real: the hook-registration code, the direct-submodule-
call code, the surgery-toggle conditional, and the causal-locality guarantee
of `ShortConvolution` itself (a structural property of the operator,
independent of trained-vs-random weights or stub-vs-real kernel -- see
item 5's own docstring for the argument). What it does NOT verify: the
real CUDA Triton `chunk_delta_rule` kernel's own behavior, or any property
of a REAL trained checkpoint's weights -- both disclosed as box-only per
item, never silently treated as verified here.

Run: python reasoning_link_stage_minus1.py
"""
from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

import reasoning_link_probe as rlp
import grammar_rd
from lm_pretrain_rd import DeltaNetLM

RESULTS = []


def _report(item: int, name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append({"item": item, "name": name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[Stage-1 item {item}] {name}: {status}" + (f" -- {detail}" if detail else ""))
    assert passed, f"Stage -1 item {item} ({name}) FAILED: {detail}"


# ---------------------------------------------------------------------------
# Item 1 -- single-token/non-overlap verification for the adapted template.
# ---------------------------------------------------------------------------

def test_item_1_tokenizer_verification():
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    pools, report = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
    marker_report = rlp.verify_query_marker_candidates(tokenizer, pools)
    assert marker_report["n_ok"] >= 2, "need >=2 independently-qualifying markers (sec 7.6)"
    # conv_size=4 is the ONLY value every registered checkpoint (Leg A rung-1, Leg B rungs 1-3,
    # per lm_rd_rung_configs.py/frozen_bias_lm_sweep.py, verified this build session) uses --
    # verify K_min/clause_len for conv_size=4 explicitly (both checkpoint families).
    for conv_size in (4,):
        cl = rlp.clause_len_for_conv_size(conv_size)
        kmin = rlp.k_min_for_conv_size(conv_size)
        assert cl == max(1, conv_size - 1) + 4
        assert kmin == math.ceil(128 / cl)
    detail = (f"marker={pools.query_id!r} n_ok_markers={marker_report['n_ok']} "
              f"n_train={pools.train_name_ids.numel()} n_heldout={pools.heldout_name_ids.numel()} "
              f"buffer_id==period_id: {pools.buffer_id == pools.period_id}")
    _report(1, "tokenizer + query-marker + conv_size=4 verification", True, detail)


# ---------------------------------------------------------------------------
# Item 2 -- hand-built S/q matrices, known composition answer.
# ---------------------------------------------------------------------------

def test_item_2_hand_built_composition():
    S = torch.tensor([[2.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]).unsqueeze(0)              # (1,4,4)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).unsqueeze(1)               # (1,1,4) -- one query
    for h, expected in [(0, [1.0, 0, 0, 0]), (1, [2.0, 0, 0, 0]), (2, [4.0, 0, 0, 0]),
                        (3, [8.0, 0, 0, 0]), (4, [16.0, 0, 0, 0])]:
        pred = rlp.apply_state_power(S, q, h)
        exp = torch.tensor(expected).view(1, 1, 4)
        assert torch.allclose(pred, exp, atol=1e-6), f"h={h}: pred={pred} expected={exp}"
    # cosine-recovery scorer: pred exactly aligned with target -> recovered True, cos~1.0
    target = torch.tensor([16.0, 0.0, 0.0, 0.0]).view(1, 1, 4)
    pred4 = rlp.apply_state_power(S, q, 4)
    cos, recovered = rlp.cosine_and_recovered(pred4, target)
    assert abs(cos.item() - 1.0) < 1e-6 and bool(recovered.item()) is True
    # target orthogonal -> recovered False, cos~0
    target_orth = torch.tensor([0.0, 1.0, 0.0, 0.0]).view(1, 1, 4)
    cos2, recovered2 = rlp.cosine_and_recovered(pred4, target_orth)
    assert abs(cos2.item()) < 1e-6 and bool(recovered2.item()) is False
    _report(2, "hand-built S^h@q + cosine scorer reproduce pinned values to 1e-6", True)


# ---------------------------------------------------------------------------
# Item 3 -- bigram-shortcut adversarial construction.
# ---------------------------------------------------------------------------

def test_item_3_bigram_shortcut_scorer():
    true_value = F.normalize(torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]), dim=-1)         # true in-context value
    bigram_prior = F.normalize(torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]), dim=-1)        # orthogonal "most common" value
    assert abs(F.cosine_similarity(true_value, bigram_prior, dim=-1).item()) < 1e-6, "fixture not orthogonal"
    # scorer, given pred == true in-context value, must recover (never fooled by the bigram prior
    # simply because it wasn't even compared against here -- this checks the SCORER, not the model)
    cos_true, rec_true = rlp.cosine_and_recovered(true_value, true_value)
    assert bool(rec_true.item()) is True and abs(cos_true.item() - 1.0) < 1e-6
    # scorer, given pred == the bigram-prior completion, must NOT recover against the true target
    # (demonstrates the scorer discriminates the two rather than passing either way)
    cos_bigram, rec_bigram = rlp.cosine_and_recovered(bigram_prior, true_value)
    assert bool(rec_bigram.item()) is False
    _report(3, "scorer distinguishes true in-context value from an orthogonal bigram-prior completion", True)


# ---------------------------------------------------------------------------
# Item 4 -- label-shuffle null generator destroys the true pairing.
# ---------------------------------------------------------------------------

def test_item_4_label_shuffle_null():
    d = 6                                                               # K=6 (<=d) entities, mutually orthogonal
    Q_ortho, _ = torch.linalg.qr(torch.randn(d, d))                    # (d,d) orthonormal columns
    v_eff_items = Q_ortho.T.unsqueeze(0)                                # (1,K=6,d), mutually orthogonal rows

    succ_true = torch.tensor([[1, 2, 3, 4, 5, 0]])                     # a hand-built 6-cycle: i -> i+1 mod 6
    a_slot = torch.tensor([[0]])
    # h=2 (not h=1): h=1's own prev_slot = _iterate_permutation(succ, a_slot, hops-1=0), whose
    # hops==0 BASE CASE returns a_slot directly WITHOUT ever reading `succ` -- an h=1 fixture
    # would make true and null prev_slot identical by construction regardless of the shuffle,
    # a vacuous test. h=2's prev_slot = succ[a_slot] (hops-1=1) genuinely depends on succ.
    hops = torch.tensor([[2]])

    # FATAL-2 fix (this audit round, item-4 tautology): the earlier draft built `pred` FROM
    # `prev_slot_true` -- the just-computed OUTPUT of the very function under test -- so
    # cos_true==1.0 was guaranteed by pure algebra (pred and target_true were the same tensor by
    # construction), never actually checking whether `compute_prev_slot_and_target` computed the
    # RIGHT slot. Fix: derive the expected slot by an INDEPENDENT hand computation that never calls
    # `compute_prev_slot_and_target` -- succ_true is the literal cycle 0->1->2->3->4->5->0, so its
    # inverse is the reverse cycle (succ_inv[j] = (j-1) mod 6, read off directly from the hand-built
    # array, not from any grammar_rd/rlp function); `prev_slot = succ^-1(succ^h(a))` is a
    # mathematically distinct route to the same quantity from the one `_iterate_permutation`
    # actually takes (forward iteration by hops-1), per REASONING_LINK_DESIGN.md sec 4.4's own
    # "equivalently, prev_slot = inv_succ[tgt_slot]" remark.
    succ_inv = [(j - 1) % 6 for j in range(6)]
    a, h = 0, 2
    target_slot_hand = a
    for _ in range(h):
        target_slot_hand = succ_true[0, target_slot_hand].item()          # pi^h(a) by direct iteration
    expected_prev_slot = succ_inv[target_slot_hand]                       # pi^-1(pi^h(a)) = pi^(h-1)(a)
    assert expected_prev_slot == 1, f"hand computation itself is wrong: expected 1, got {expected_prev_slot}"

    prev_slot_true, target_true = rlp.compute_prev_slot_and_target(succ_true, a_slot, hops, v_eff_items)
    assert prev_slot_true.item() == expected_prev_slot, (
        f"compute_prev_slot_and_target's own prev_slot ({prev_slot_true.item()}) disagrees with the "
        f"independently hand-derived slot ({expected_prev_slot}) -- a real correctness check, not a "
        f"tautology built from the function's own output")

    pred = v_eff_items[:, expected_prev_slot, :].unsqueeze(1)   # PLANTED from the INDEPENDENT hand
                                                                  # index (never from prev_slot_true)
    cos_true, rec_true = rlp.cosine_and_recovered(pred, target_true)
    assert bool(rec_true.item()) is True and abs(cos_true.item() - 1.0) < 1e-6, "planted-recoverable setup broken"

    gen = torch.Generator().manual_seed(0)
    succ_null = rlp.build_null_labeling(succ_true, gen)
    assert not torch.equal(succ_null, succ_true), "null cycle draw coincided with the true cycle -- reseed"
    prev_slot_null, target_null = rlp.compute_prev_slot_and_target(succ_null, a_slot, hops, v_eff_items)
    cos_null, rec_null = rlp.cosine_and_recovered(pred, target_null)
    assert bool(rec_null.item()) is False, (
        f"label-shuffle null FAILED to destroy the true pairing: cos={cos_null.item()}")
    _report(4, "label-shuffle null destroys a planted-recoverable true pairing (planted pred derived "
               "from an INDEPENDENT hand computation of the target slot, not from the function's own "
               "output -- not a tautology)", True,
            f"cos_true={cos_true.item():.4f} cos_null={cos_null.item():.4f}")


# ---------------------------------------------------------------------------
# Item 5 -- buffer/query blank-out test (structural conv-locality guarantee).
# ---------------------------------------------------------------------------

def test_item_5_blank_out_test():
    """Verifies a STRUCTURAL guarantee of ShortConvolution (causal, window
    conv_size): corrupting clause j's tokens cannot change clause j+1's
    k_eff, since j+1's own conv window (size conv_size, causal) never
    reaches back into clause j's positions once buf_len>=1 separates them.
    This holds identically for a synthetic (untrained) stand-in model --
    the property is about the OPERATOR's definition, not learned weights
    (module docstring's own argument, mirroring sec 9 item 5's Rev 3
    disclosure)."""
    torch.manual_seed(0)
    V = 50257
    model = DeltaNetLM(V, d_model=32, d_state=64, n_layers=2, conv_size=4, num_heads=1)
    model.eval()
    pools, _ = rlp.build_reasoning_link_pools(seed=0)
    cfg = rlp.episode_config_for_checkpoint(conv_size=4, K=20)
    gen = torch.Generator().manual_seed(0)
    batch = grammar_rd.sample_batch_rd(cfg, 4, gen, hop_set=(1,), pools=pools)
    token_ids = batch["token_ids"].clone()
    item_pos = batch["item_pos"]

    j = 5  # an interior clause, margin on both sides
    key_pos_j, rel_pos_j, val_pos_j, per_pos_j = (item_pos[:, j] - 2, item_pos[:, j] - 1,
                                                    item_pos[:, j], item_pos[:, j] + 1)
    buf_start_j = j * cfg.clause_len
    buf_end_j = key_pos_j[0].item()  # exclusive

    with torch.no_grad():
        x_before = model.embed(token_ids)
        blk0 = model.blocks[0]
        k_raw_before, _ = blk0.mixer.k_conv1d(blk0.mixer.k_proj(blk0.norm1(x_before)))

    corrupted = token_ids.clone()
    torch.manual_seed(123)
    corrupted[:, buf_start_j:buf_end_j] = torch.randint(0, V, (corrupted.shape[0], buf_end_j - buf_start_j))
    for pos in (key_pos_j, rel_pos_j, val_pos_j, per_pos_j):
        corrupted.scatter_(1, pos.unsqueeze(1), torch.randint(0, V, (corrupted.shape[0], 1)))
    assert not torch.equal(corrupted[:, buf_start_j:per_pos_j[0].item() + 1],
                            token_ids[:, buf_start_j:per_pos_j[0].item() + 1]), "corruption was a no-op"

    with torch.no_grad():
        x_after = model.embed(corrupted)
        k_raw_after, _ = blk0.mixer.k_conv1d(blk0.mixer.k_proj(blk0.norm1(x_after)))

    item_pos_jp1 = item_pos[:, j + 1]
    k_before_jp1 = torch.gather(k_raw_before, 1, item_pos_jp1.view(-1, 1, 1).expand(-1, 1, 64)).squeeze(1)
    k_after_jp1 = torch.gather(k_raw_after, 1, item_pos_jp1.view(-1, 1, 1).expand(-1, 1, 64)).squeeze(1)
    max_diff = (k_before_jp1 - k_after_jp1).abs().max().item()
    assert max_diff < 1e-6, f"clause j+1's k_eff CHANGED after corrupting clause j: max_diff={max_diff}"
    _report(5, "corrupting clause j leaves clause j+1's k_eff unchanged (1e-6)", True, f"max_diff={max_diff:.2e}")


# ---------------------------------------------------------------------------
# Item 6 -- surgery-toggle equivalence smoke with 2 genuine negative controls.
# ---------------------------------------------------------------------------

def test_item_6_surgery_toggle_smoke():
    """FATAL-2 fix (this audit round): drives the PRODUCTION `rlp.frozen_bias_surgery` context
    manager + `rlp.run_forward_b` directly -- the exact functions `measure_cell_all_h`/
    `causality_check` call in real use -- rather than a hand-reproduced parallel copy of
    `DeltaNetLMMixer.forward`'s pre-kernel sequence (the earlier draft's `reproduce_forward_
    pre_kernel`, which could pass even if `frozen_bias_surgery` itself were broken or a no-op,
    since it never called that function at all). Two controls, both driven through the real path:
    (a) live-gate-fires -- forcing the blend off via surgery must change the model's own forward
    OUTPUT (`hidden`, post-norm_f) vs. the native (arm='global') pass; (b) surgery-mechanics -- the
    context manager itself must flip `frozen_bias_arm` to 'off' while inside the `with` block and
    restore the ORIGINAL arm on exit, including when an exception is raised inside the block. A
    no-op `frozen_bias_surgery` mutation (the exact case audited) fails control (b)'s in-context
    assertion immediately, and control (a)'s output-difference assertion as a second, independent
    line of defense.

    Why `hidden` (via run_forward_b), not `S_T`/`final_state` (via run_forward_a): under this
    file's own CPU fla-stub, `chunk_delta_rule`'s `final_state` is a DELIBERATE hardcoded zero
    regardless of q/k/v/beta (see reasoning_link_probe.py's stub docstring: "box-only-verifiable
    items depending on a NONZERO, trained S_T ... cannot be given meaningful numbers by this
    stub") -- comparing S_T here would compare zero to zero regardless of whether surgery did
    anything, a vacuous check under the stub (caught empirically this audit-fix session: an
    S_T-based draft of this control passed with mean_abs_diff=0.0 even with a genuinely-working
    surgery). The stub's own output `o = v * sigmoid(beta) * sigmoid((q*k).sum(-1))` DOES
    genuinely depend on `k` (the stub's own disclosed k-routing fix, module docstring) and flows
    into `hidden` via `model.norm_f`, so `hidden` is the correct CPU-observable signal here."""
    torch.manual_seed(41)
    V, D_STATE = 300, 64
    model = DeltaNetLM(V, d_model=64, d_state=D_STATE, n_layers=1, conv_size=4, num_heads=1,
                        frozen_bias_arm="global", frozen_bias_lambda=0.58,
                        frozen_bias_vocab_size=V, frozen_bias_seed=999)
    model.eval()
    x_ids = torch.randint(0, V, (4, 128))
    mixer = model.blocks[0].mixer
    original_arm = mixer.frozen_bias_arm
    assert original_arm == "global"

    # native pass -- surgery_off=False must be a documented no-op (frozen_bias_surgery's own
    # `if not force_off: yield; return` branch).
    hidden_native, _, _ = rlp.run_forward_b(model, x_ids, need_option2_hidden=True, surgery_off=False)
    assert mixer.frozen_bias_arm == "global", "surgery_off=False must never touch frozen_bias_arm"

    # THE surgery, through the production context manager + run_forward_b -- never a hand-copied
    # pre-kernel sequence.
    hidden_off, _, _ = rlp.run_forward_b(model, x_ids, need_option2_hidden=True, surgery_off=True)
    assert mixer.frozen_bias_arm == "global", (
        "frozen_bias_surgery FAILED to restore the original arm after run_forward_b's own context exited")

    # control (a): live-gate-fires -- forcing the blend off through the REAL forward path must
    # change the model's own output. A no-op frozen_bias_surgery mutation makes native and off
    # compute with the IDENTICAL (still-'global') arm both times, collapsing this diff to ~0 and
    # failing here.
    mean_abs_diff = (hidden_native - hidden_off).abs().mean().item()
    assert mean_abs_diff > 1e-4, f"live-gate-fires control FAILED (surgery had no effect on the model's output): mean_abs_diff={mean_abs_diff}"

    # control (b): surgery-mechanics -- entering/exiting the context manager DIRECTLY (not via
    # run_forward_b this time) must flip frozen_bias_arm to 'off' strictly inside the `with` block
    # and restore the ORIGINAL value immediately on exit -- including exit via an exception, the
    # `finally` clause's own job.
    with rlp.frozen_bias_surgery(model, force_off=True):
        assert mixer.frozen_bias_arm == "off", "frozen_bias_surgery did not flip frozen_bias_arm to 'off' inside the context"
    assert mixer.frozen_bias_arm == "global", "frozen_bias_surgery did not restore the original arm on normal exit"
    try:
        with rlp.frozen_bias_surgery(model, force_off=True):
            assert mixer.frozen_bias_arm == "off"
            raise RuntimeError("deliberate -- exercising the finally-restores-on-exception path")
    except RuntimeError:
        pass
    assert mixer.frozen_bias_arm == "global", "frozen_bias_surgery did not restore the arm after an exception inside the context"

    _report(6, "surgery toggle driven through the PRODUCTION frozen_bias_surgery + run_forward_b "
               "path: live-gate-fires (model output changes, >1e-4) AND surgery-mechanics "
               "(flip+restore, incl. under an exception) controls all pass",
            True, f"mean_abs_diff(hidden)={mean_abs_diff:.4f}")


# ---------------------------------------------------------------------------
# Item 7 -- conv_size-match assertion.
# ---------------------------------------------------------------------------

def test_item_7_conv_size_match_assertion():
    fake_ckpt_config = {"conv_size": 4}
    cfg = rlp.episode_config_for_checkpoint(fake_ckpt_config["conv_size"], K=32)
    assert fake_ckpt_config["conv_size"] == cfg.conv_size

    raised = False
    try:
        assert 5 == rlp.episode_config_for_checkpoint(4, K=32).conv_size, "deliberate mismatch"
    except AssertionError:
        raised = True
    assert raised, "conv_size-mismatch assertion did not fire on a deliberately wrong comparison"
    _report(7, "conv_size-match assertion holds on real config, fires on deliberate mismatch", True)


# ---------------------------------------------------------------------------
# Item 8 -- render-adjacency vs. cycle-adjacency correlation measurement.
# ---------------------------------------------------------------------------

def test_item_8_render_adjacency_measurement():
    pools, _ = rlp.build_reasoning_link_pools(seed=0)
    n_episodes = 10_000
    detail_lines = []
    for K in (20, 32, 40, 48, 64, 96):
        cfg = rlp.episode_config_for_checkpoint(conv_size=4, K=K)
        gen = torch.Generator().manual_seed(K)
        batch = grammar_rd.sample_batch_rd(cfg, n_episodes, gen, hop_set=(1,), pools=pools)
        succ = batch["succ"]                                             # (B,K)
        is_adjacent = torch.zeros(n_episodes, K, dtype=torch.bool)
        for i in range(K):
            neighbors = [n for n in (i - 1, i + 1) if 0 <= n < K]
            row_match = torch.zeros(n_episodes, dtype=torch.bool)
            for n in neighbors:
                row_match |= (succ[:, i] == n)
            is_adjacent[:, i] = row_match
        empirical = is_adjacent.float()
        theoretical_rate = 2.0 / K   # sec 4.2's own closed-form: (K-2)*2/(K-1) + 2*1/(K-1), averaged over K slots
        lo, hi = rlp.bootstrap_ci_95(empirical, seed=K)
        ok = lo <= theoretical_rate <= hi
        detail_lines.append(f"K={K}: theoretical={theoretical_rate:.5f} CI=[{lo:.5f},{hi:.5f}] pass={ok}")
        assert ok, f"K={K}: theoretical rate {theoretical_rate} outside empirical 95% CI [{lo},{hi}]"
    _report(8, "render-adjacency rate statistically indistinguishable from chance at every swept K",
            True, "; ".join(detail_lines))


# ---------------------------------------------------------------------------
# Item 9 -- T_bind >= _MIN_KERNEL_T floor assertion.
# ---------------------------------------------------------------------------

def test_item_9_kernel_t_floor():
    from lm_pretrain_rd import _MIN_KERNEL_T
    # MINOR-1 fix (this audit round): a direct, hard-coded-literal check on k_min_for_conv_size(4)
    # ITSELF, independent of item 1's own check (item 1 verifies the FORMULA structurally --
    # `kmin == math.ceil(128 / cl)` -- which would still pass even if `k_min_for_conv_size`'s
    # registered constant floor changed; this assertion pins the actual number sec 4.1 registers,
    # 19, so a mutation that silently changes what k_min_for_conv_size(4) returns is caught HERE,
    # not only transitively through item 1's relative formula check).
    assert rlp.k_min_for_conv_size(4) == 19, (
        f"k_min_for_conv_size(4) expected the sec-4.1-registered value 19, got {rlp.k_min_for_conv_size(4)}")
    for K in (20, 32, 40, 48, 64, 96):
        cfg = rlp.episode_config_for_checkpoint(conv_size=4, K=K)
        assert cfg.T_bind >= _MIN_KERNEL_T, f"K={K}: T_bind={cfg.T_bind} < _MIN_KERNEL_T={_MIN_KERNEL_T}"
    _report(9, "k_min_for_conv_size(4)==19 (direct literal check) and every registered K clears "
               "T_bind >= _MIN_KERNEL_T=128 at conv_size=4", True)


# ---------------------------------------------------------------------------
# Item 10 -- episode-seed non-collision assertion.
# ---------------------------------------------------------------------------

def test_item_10_seed_non_collision():
    combos = rlp.enumerate_registered_seed_combinations()
    seeds = [rlp.episode_seed(*c) for c in combos]
    assert len(set(seeds)) == len(seeds), (
        f"seed collision detected: {len(combos)} combinations produced {len(set(seeds))} unique seeds")
    _report(10, "episode_seed() is collision-free over every registered combination", True,
            f"n_combinations={len(combos)}")


# ---------------------------------------------------------------------------
# Item 11 -- two-forward causality assertion.
# ---------------------------------------------------------------------------

def test_item_11_causality_assertion():
    """Verifies the PROBE'S OWN forward-A/forward-B slicing never leaks
    forward-B's trailing query content backward into the shared BIND prefix
    -- a property guaranteed by ShortConvolution's causal locality
    (identical whether the kernel underneath is the stub or the real
    Triton kernel, see item 5's argument) plus this file's own correct
    tensor bookkeeping. BOX-ONLY CAVEAT: `S_T` itself (the recurrent
    state) is a hardcoded ZERO matrix under the CPU stub (see
    reasoning_link_probe.py's stub docstring) -- this test cannot exercise
    the REAL kernel's own state-recurrence causality (that is an
    established property of the production Triton kernel, not re-verified
    here); it verifies the PREFIX-IDENTITY bookkeeping this design's own
    two-forward protocol depends on."""
    torch.manual_seed(7)
    V = 50257
    model = DeltaNetLM(V, d_model=32, d_state=64, n_layers=2, conv_size=4, num_heads=1)
    model.eval()
    pools, _ = rlp.build_reasoning_link_pools(seed=0)
    cfg = rlp.episode_config_for_checkpoint(conv_size=4, K=20)
    gen = torch.Generator().manual_seed(0)
    batch = grammar_rd.sample_batch_rd(cfg, 4, gen, hop_set=(1,), pools=pools)
    query_one = batch["query_tokens"][:, 0, :]
    result = rlp.causality_check(model, batch["token_ids"], query_one, tol=1e-6)
    assert result["pass"], f"causality check FAILED: max_abs_diff={result['max_abs_diff']}"
    _report(11, "forward-A/forward-B produce identical k/v_conv1d outputs over the shared BIND prefix "
                "(box-only: real Triton kernel's own recurrence not exercised, see docstring)",
            True, f"max_abs_diff={result['max_abs_diff']:.2e}")


# ---------------------------------------------------------------------------
# Item 12 -- 3-entity worked example (off-by-one-hop fix), with negative control.
# ---------------------------------------------------------------------------

def test_item_12_three_entity_worked_example():
    """FATAL-2 fix (this audit round): calls `rlp.compute_prev_slot_and_target` -- the PRODUCTION
    function `measure_cell_all_h` actually uses for every real cell -- rather than hand-reproducing
    its two-line body (`_iterate_permutation(succ, a_slot, hops-1)` + gather) directly against
    `grammar_rd._iterate_permutation`. The earlier draft called `_iterate_permutation` itself twice
    (once for `tgt_slot`, once with `hops-1` for `prev_slot`) -- a TEST COPY of the production
    function's own logic, which would keep passing even if `compute_prev_slot_and_target` itself
    regressed (e.g. the exact `hops-1` -> `hops` mutation an independent audit applies), since the
    test never actually calls that function at all."""
    succ = torch.tensor([[1, 2, 0]])                # A->B->C->A
    entity_ids = torch.tensor([[100, 200, 300]])     # A=100, B=200, C=300 (slots 0,1,2)
    value_ids = torch.gather(entity_ids, 1, succ)    # [B, C, A] = [200, 300, 100]
    assert torch.equal(value_ids, torch.tensor([[200, 300, 100]]))

    a_slot = torch.tensor([[0]])   # querying about A
    # `compute_prev_slot_and_target` takes `v_eff_items` shaped (B,K,d) -- here d=1, with the
    # "effective value" simply BEING the entity id (no learned representation involved at this
    # scale), letting gather_at_positions' generic (B,K,d) machinery stand in for value_ids
    # directly and route this test through the REAL production function, not a reimplementation.
    v_eff_items = value_ids.unsqueeze(-1).float()    # (1,3,1)
    for hops_val, expected_correct, expected_naive_wrong in [(1, 200, 300), (2, 300, 100)]:
        hops = torch.tensor([[hops_val]])
        prev_slot, v_eff_target = rlp.compute_prev_slot_and_target(succ, a_slot, hops, v_eff_items)
        scored_correct = int(v_eff_target.item())
        # negative control: the KNOWN one-hop-past bug this fix corrects -- gathering at tgt_slot
        # (pi^h(a)'s own slot) instead of prev_slot (pi^(h-1)(a)'s slot). Uses grammar_rd's own
        # `_iterate_permutation` directly (the production primitive `compute_prev_slot_and_target`
        # itself calls), fed the WRONG hop-count on purpose to reproduce the bug -- not a parallel
        # reimplementation of the FIXED path, only of the deliberately-wrong one.
        tgt_slot = grammar_rd._iterate_permutation(succ, a_slot, hops)
        scored_naive = torch.gather(value_ids, 1, tgt_slot).item()
        assert scored_correct == expected_correct, (
            f"h={hops_val}: compute_prev_slot_and_target gave entity {scored_correct}, expected {expected_correct}")
        assert scored_naive == expected_naive_wrong, (
            f"h={hops_val}: naive tgt_slot gather gave {scored_naive}, "
            f"expected it to reproduce the KNOWN one-hop-past bug ({expected_naive_wrong})")
        assert scored_correct != scored_naive, "negative control vacuous: fixed and naive indices agree"
    _report(12, "3-entity worked example: rlp.compute_prev_slot_and_target (the PRODUCTION function "
                "measure_cell_all_h uses) correct at h=1,2; naive tgt_slot gather reproduces the "
                "known one-hop-past bug (negative control non-vacuous)", True)


# ---------------------------------------------------------------------------
# Item 13 -- v_conv1d hook equivalence smoke.
# ---------------------------------------------------------------------------

def test_item_13_v_conv1d_hook_equivalence():
    torch.manual_seed(3)
    V = 50257
    model = DeltaNetLM(V, d_model=32, d_state=64, n_layers=2, conv_size=4, num_heads=1)
    model.eval()
    x_ids = torch.randint(0, V, (2, 128))
    handles, captured = rlp.register_kqv_hooks(model)
    try:
        with torch.no_grad():
            _ = rlp.forward_body(model, x_ids, need_hidden=False)
    finally:
        rlp.remove_hooks(handles)
    # NOTE: DeltaNetLMBlock.forward calls `self.mixer(self.norm1(x), ...)` -- the mixer's REAL
    # input is the block's own pre-norm, not the raw embedding/residual stream directly. The
    # direct-submodule comparison below must reproduce that pre-norm step, or it compares against
    # the wrong input entirely (caught empirically this build session: an earlier draft omitted
    # `blk.norm1(...)` and got a spurious ~1.35 "divergence" that was really just "normed vs
    # unnormed input", not a real hook-vs-module mismatch).
    with torch.no_grad():
        xemb = model.embed(x_ids)
    blk0 = model.blocks[0]
    direct_v, _ = blk0.mixer.v_conv1d(blk0.mixer.v_proj(blk0.norm1(xemb)))
    hook_v = captured["v"][0]
    max_diff = (direct_v - hook_v).abs().max().item()
    assert max_diff < 1e-6, f"layer 0: v_conv1d hook diverges from direct call by {max_diff}"
    _report(13, "v_conv1d forward-hook output matches a direct submodule call (1e-6)", True)


# ---------------------------------------------------------------------------
# Item 14 -- q_conv1d hook equivalence smoke, on forward-B's own input construction.
# ---------------------------------------------------------------------------

def test_item_14_q_conv1d_hook_equivalence():
    torch.manual_seed(4)
    V = 50257
    model = DeltaNetLM(V, d_model=32, d_state=64, n_layers=2, conv_size=4, num_heads=1)
    model.eval()
    pools, _ = rlp.build_reasoning_link_pools(seed=0)
    cfg = rlp.episode_config_for_checkpoint(conv_size=4, K=20)
    gen = torch.Generator().manual_seed(0)
    batch = grammar_rd.sample_batch_rd(cfg, 2, gen, hop_set=(1,), pools=pools)
    concat = torch.cat([batch["token_ids"], batch["query_tokens"][:, 0, :]], dim=1)  # forward-B's own construction

    handles, captured = rlp.register_kqv_hooks(model)
    try:
        with torch.no_grad():
            _ = rlp.forward_body(model, concat, need_hidden=False)
    finally:
        rlp.remove_hooks(handles)

    with torch.no_grad():
        xemb = model.embed(concat)
    blk0 = model.blocks[0]
    direct_q, _ = blk0.mixer.q_conv1d(blk0.mixer.q_proj(blk0.norm1(xemb)))  # see item 13's own norm1 note
    hook_q = captured["q"][0]
    max_diff = (direct_q - hook_q).abs().max().item()
    assert max_diff < 1e-6, f"layer 0: q_conv1d hook diverges from direct call by {max_diff}"
    _report(14, "q_conv1d forward-hook output (on forward-B's bind+query input) matches a direct "
                "submodule call (1e-6)", True, f"max_diff={max_diff:.2e}")


# ---------------------------------------------------------------------------
# Items 15-18 -- sec 16.1 PHASE-1B (natural-language surface form) additions.
# Not part of the design's original 14 (sec 9 was written before sec 16.1
# existed) -- registered by THIS BUILD's own brief ("Stage -1 additions per
# sec 16.1's registered items"), numbered onward from 14 following this
# file's own convention of one executable item per registered build
# artifact.
# ---------------------------------------------------------------------------

def test_item_15_candidate_b_verb_verification():
    """sec 16.1.1 Candidate B's own registered build item: the succession-
    family verb pool ("succeeded"/"replaced") is NEW, Phase-1b-only
    vocabulary, never verified anywhere in this codebase before -- single-
    token + round-trip-decode + non-collision, against the REAL GPT-2
    tokenizer (never assumed)."""
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    pools, _ = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
    report = rlp.verify_natural_template_b_verbs(tokenizer, pools)
    assert report["ok"], f"Candidate B verb verification FAILED: {report['rejected']!r}"
    assert report["n_verified"] == len(rlp._CANDIDATE_B_VERBS) == 2
    # end-to-end: build_natural_pools(template='B') must actually succeed and substitute rel_a_ids
    # with exactly these verified ids (not silently fall back to the gift-verb pool on failure).
    nat_pools_b, nat_report_b = rlp.build_natural_pools(tokenizer=tokenizer, seed=0, template="B")
    verified_ids = {tid for _, tid in report["verified"]}
    assert set(nat_pools_b.rel_a_ids.tolist()) == verified_ids, (
        "build_natural_pools(template='B') did not substitute rel_a_ids with the verified succession ids")
    _report(15, "Candidate B succession-verb pool ('succeeded'/'replaced') single-token + "
                "non-collision verified against the real GPT-2 tokenizer; build_natural_pools "
                "substitutes rel_a_ids with exactly these ids", True,
            f"verified={report['verified']}")


def test_item_16_natural_template_floor_rederivation():
    """This BUILD's own brief: "clause length CHANGES -- re-derive T_bind
    and the K_min floor for the natural template; assert them." Re-derived
    (not assumed) and checked directly: the BIND-side clause shape is
    UNCHANGED under sec 16.1 ("Both candidates keep the period-repeat
    buffer convention unchanged" -- same buf+KEY+REL+VALUE+PERIOD template,
    only WHICH verb pool renders REL differs) -- so T_bind/K_min are
    IDENTICAL to the marker template's, a checked conclusion, not an
    inherited assumption. The quantity that GENUINELY changes is the QUERY
    window (marker dropped, not replaced) -- natural_query_len is exactly
    ONE TOKEN SHORTER than the marker template's query_len at every
    conv_size, checked directly against a real episode_cfg rather than by
    formula alone."""
    for conv_size in (4,):
        assert rlp.natural_clause_len(conv_size) == rlp.clause_len_for_conv_size(conv_size) == 7, (
            "BIND clause_len must be unchanged under the natural template (sec 16.1)")
        assert rlp.natural_k_min_for_conv_size(conv_size) == rlp.k_min_for_conv_size(conv_size) == 19, (
            "K_min must be unchanged under the natural template (T_bind depends only on the "
            "unchanged BIND clause_len)")
        cfg = rlp.episode_config_for_checkpoint(conv_size, K=20)
        assert rlp.natural_query_len(conv_size) == cfg.query_len - 1 == 5, (
            "natural_query_len must be exactly one token shorter than the marker template's "
            "query_len (the <Q> marker dropped, not replaced)")
    for K in (20, 32, 40, 48, 64, 96):
        T_bind_natural = K * rlp.natural_clause_len(4)
        assert T_bind_natural >= 128, f"K={K}: natural T_bind={T_bind_natural} < _MIN_KERNEL_T=128"
        # forward-B's total T (T_bind + query_len) only grows relative to T_bind alone -- a SHORTER
        # query_len cannot newly violate the kernel floor once T_bind alone already clears it.
        assert T_bind_natural + rlp.natural_query_len(4) >= 128
    _report(16, "natural-template T_bind/K_min re-derived and IDENTICAL to the marker template's "
                "(BIND clause shape unchanged, K_min=19); natural_query_len re-derived and CHANGED "
                "(one token shorter than the marker template's query_len, marker dropped not "
                "replaced) -- every registered K clears the re-derived floor", True)


def test_item_17_natural_query_causality_assertion():
    """Mirrors item 11 (two-forward causality assertion) but drives it
    through sec 16.1's own natural-completion query construction, for BOTH
    Candidate A and Candidate B -- the query window's SHAPE changed (marker
    dropped, one token shorter), so the shared-BIND-prefix causality
    guarantee is re-verified here rather than assumed to still hold
    unchanged from item 11's marker-template check."""
    torch.manual_seed(70)
    V = 50257
    model = DeltaNetLM(V, d_model=32, d_state=64, n_layers=2, conv_size=4, num_heads=1)
    model.eval()
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    cfg = rlp.episode_config_for_checkpoint(conv_size=4, K=20)
    for template in rlp.NATURAL_TEMPLATES:
        pools, _ = rlp.build_natural_pools(tokenizer=tokenizer, seed=0, template=template)
        gen = torch.Generator().manual_seed(0)
        batch = grammar_rd.sample_batch_rd(cfg, 4, gen, hop_set=(1,), pools=pools)
        qk_one = torch.gather(batch["entity_ids"], 1, batch["a_slot"][:, :1])
        natural_query_one = rlp.build_natural_query_tokens(cfg, pools, qk_one, batch["rel_id"])[:, 0, :]
        assert natural_query_one.shape[-1] == rlp.natural_query_len(4)
        result = rlp.causality_check(model, batch["token_ids"], natural_query_one, tol=1e-6)
        assert result["pass"], (
            f"template={template!r}: natural-query causality check FAILED: "
            f"max_abs_diff={result['max_abs_diff']}")
    _report(17, "forward-A/forward-B produce identical k/v_conv1d outputs over the shared BIND "
                "prefix under sec 16.1's natural-completion query (marker dropped), Candidates A "
                "and B both verified", True)


def test_item_18_natural_q_conv1d_hook_equivalence():
    """Mirrors item 14 (q_conv1d hook equivalence) but on sec 16.1's natural
    query window -- confirms q_eff is correctly captured at the query's own
    NEW final position (the relation-verb token REL, since the marker that
    used to sit there is dropped) and matches a direct submodule call,
    for both candidates."""
    torch.manual_seed(71)
    V = 50257
    model = DeltaNetLM(V, d_model=32, d_state=64, n_layers=2, conv_size=4, num_heads=1)
    model.eval()
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    cfg = rlp.episode_config_for_checkpoint(conv_size=4, K=20)
    for template in rlp.NATURAL_TEMPLATES:
        pools, _ = rlp.build_natural_pools(tokenizer=tokenizer, seed=0, template=template)
        gen = torch.Generator().manual_seed(0)
        batch = grammar_rd.sample_batch_rd(cfg, 2, gen, hop_set=(1,), pools=pools)
        qk_one = torch.gather(batch["entity_ids"], 1, batch["a_slot"][:, :1])
        natural_query_one = rlp.build_natural_query_tokens(cfg, pools, qk_one, batch["rel_id"])[:, 0, :]
        concat = torch.cat([batch["token_ids"], natural_query_one], dim=1)  # forward-B's own construction

        handles, captured = rlp.register_kqv_hooks(model)
        try:
            with torch.no_grad():
                _ = rlp.forward_body(model, concat, need_hidden=False)
        finally:
            rlp.remove_hooks(handles)

        with torch.no_grad():
            xemb = model.embed(concat)
        blk0 = model.blocks[0]
        direct_q, _ = blk0.mixer.q_conv1d(blk0.mixer.q_proj(blk0.norm1(xemb)))  # item 13's own norm1 note
        hook_q = captured["q"][0]
        max_diff = (direct_q - hook_q).abs().max().item()
        assert max_diff < 1e-6, f"template={template!r}: q_conv1d hook diverges from direct call by {max_diff}"
        # the query's own final position is now REL (marker dropped) -- confirm the last-position
        # read this module's forward-B scoring convention relies on lands on a REAL token, not an
        # out-of-range/padding position: query_len == buf_len(3) + KEY + REL == 5 for conv_size=4,
        # and concat's last natural_query_len(4)=5 columns ARE exactly natural_query_one, so index
        # -1 is unambiguously REL's own hook output.
        assert concat.shape[1] - rlp.natural_query_len(4) == batch["token_ids"].shape[1]
    _report(18, "q_conv1d forward-hook output (on forward-B's natural-query bind+query input) "
                "matches a direct submodule call (1e-6); the query's own final position (REL, "
                "marker dropped) is confirmed unambiguous, Candidates A and B both verified", True)


# ---------------------------------------------------------------------------
# Item 19 -- mini-audit FIX-1 (this build round): natural-query
# semantic-content verification, closing MAJOR-1 + MAJOR-2 with one
# assertion set.
# ---------------------------------------------------------------------------

def test_item_19_natural_query_semantic_content_verification():
    """Mini-audit MAJOR-1 + MAJOR-2 fix (this build round).

    MAJOR-1: `measure_cell_all_h`'s own `query_mode` DISPATCH (the
    `if query_mode == "natural":` branch, sec 16.1) is a bare string
    comparison that items 1-18 never independently exercise -- items 17/18
    call `build_natural_query_tokens` DIRECTLY, bypassing the dispatch
    entirely. A typo'd comparison (e.g. 'natura1' vs 'natural') would
    silently fall through to the marker branch -- `measure_cell_all_h`
    would keep running (forward_a/forward_b counts unchanged) and feed
    forward-B a MARKER-template window while every caller (including
    `run_stage0_natural`) believes it asked for, and the returned JSON is
    later interpreted as, the natural-completion query. No prior item can
    catch this: it is purely a property of the DISPATCH, not the builder.

    MAJOR-2: items 17/18 assert shape, causal-locality, and hook-vs-
    direct-call agreement on the natural window -- but never its CONTENT.
    A KEY/REL order swap inside `build_natural_query_tokens` (sec 16.1's
    own registered [buf...,KEY,REL] order) would still pass every one of
    those checks unchanged (identical shape, identical causal-locality
    property, identical hook-vs-direct agreement -- none of them depend on
    WHICH entity sits in which of the last two slots) while silently
    handing forward-B's own "read q_eff at position -1" convention (this
    module's docstring, `measure_cell_all_h`'s own docstring) the WRONG
    (entity, not verb) token to score against.

    Fix: spy on the PRODUCTION `rlp.build_natural_query_tokens` (a
    module-level rebind, restored in `finally` even on exception) and
    drive it through `measure_cell_all_h`'s REAL `query_mode='natural'`
    dispatch -- never a hand call, never a parallel reimplementation. The
    spy's call count is itself the MAJOR-1 assertion (0 calls means the
    dispatch silently fell through to the marker branch). Its captured
    (args, return) is then decoded against the REAL GPT-2 tokenizer and
    checked for MAJOR-2's content properties: (a) the natural window
    differs from the marker path's window FOR THE SAME BATCH (reproduced
    directly via `grammar_rd.sample_batch_rd` with the identical seed/
    pools/episode_cfg `measure_cell_all_h` itself uses -- torch.Generator's
    own determinism makes this a bit-identical redraw, not an
    approximation); (b) no query_id/marker token appears anywhere in the
    natural window (checked against the GENUINELY DISTINCT reserved marker
    `build_reasoning_link_pools` selects -- not `build_natural_pools`'s own
    `query_id`, which sec 16.1 defensively collapses to `buffer_id`/
    `period_id` and therefore DOES legitimately appear at the natural
    window's buf positions; checking that inert placeholder for "absence"
    would be vacuously false on every correct run); (c) the LAST token
    (the position `measure_cell_all_h` actually reads q_eff from) is drawn
    from the ACTIVE `rel_a_ids` verb pool, never the entity pool -- pinned
    against the live-tokenizer-verified literal ids for Candidate B
    (' succeeded'->14131, ' replaced'->6928, cross-checked against item
    15's own verification); (d) the entity token appears at its expected
    KEY slot (index -2, immediately before REL). A KEY/REL swap makes (a)
    fail outright (the reconstructed prefix-equality breaks) and, even if
    (a) were somehow weakened, makes (c) and (d) fail independently and
    precisely (last token becomes the entity id, KEY-slot token becomes
    the verb id) -- MAJOR-1 and MAJOR-2 are both caught by this one
    assertion set, per the mini-audit's own prescription.
    """
    torch.manual_seed(190)
    V = 50257
    model = DeltaNetLM(V, d_model=32, d_state=64, n_layers=2, conv_size=4, num_heads=1)
    model.eval()
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    cfg = rlp.episode_config_for_checkpoint(conv_size=4, K=20)
    readout_layer = rlp.readout_layer_index(model)

    # Literal-token pins (mirrors item 9's own "direct hard-coded-literal, not only a
    # relative/transitive check" convention), computed fresh against the REAL GPT-2
    # tokenizer -- never assumed -- so a tokenizer-version drift is caught loudly here
    # rather than silently comparing against a stale hardcoded id.
    succeeded_id_live = tokenizer.encode(" succeeded")[0]
    replaced_id_live = tokenizer.encode(" replaced")[0]
    assert (succeeded_id_live, replaced_id_live) == (14131, 6928), (
        f"Candidate B verb ids drifted from the registered literals (item 15's own "
        f"verification): got ({succeeded_id_live}, {replaced_id_live}), expected (14131, 6928)")

    # The GENUINELY DISTINCT reserved query-marker token -- from `build_reasoning_link_pools`
    # (the ORIGINAL marker-template path), never `build_natural_pools`'s own `query_id` (which
    # sec 16.1 defensively collapses to `buffer_id`, see this function's own docstring point (b)).
    marker_pools, _ = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
    real_marker_id = int(marker_pools.query_id)
    assert real_marker_id != int(marker_pools.buffer_id), (
        "degenerate marker selection: query_id collided with buffer_id -- (b)'s absence check "
        "would be vacuous")

    for template in rlp.NATURAL_TEMPLATES:
        pools, _ = rlp.build_natural_pools(tokenizer=tokenizer, seed=0, template=template)

        # Spy on the PRODUCTION function -- module-attribute rebind, so calls made FROM WITHIN
        # measure_cell_all_h (which resolves `build_natural_query_tokens` via this module's own
        # globals at call time, standard Python late-binding) are intercepted too, not only calls
        # made directly from this test.
        calls = []
        original_fn = rlp.build_natural_query_tokens

        def _spy(cfg_arg, pools_arg, query_key_ids_arg, rel_id_arg, _orig=original_fn):
            out = _orig(cfg_arg, pools_arg, query_key_ids_arg, rel_id_arg)
            calls.append({"query_key_ids": query_key_ids_arg.clone(), "tokens": out.clone()})
            return out

        rlp.build_natural_query_tokens = _spy
        try:
            rlp.measure_cell_all_h(
                model, cfg, pools, readout_layer, 20, (1,), 4, 0, "native", "cpu",
                compute_option2=False, compute_premises=False, query_mode="natural")
        finally:
            rlp.build_natural_query_tokens = original_fn

        # MAJOR-1: the dispatch must have ACTUALLY entered the natural branch and called the
        # production builder exactly once. A 'natura1'-typo'd (or any other silently-false)
        # comparison never calls it at all -- caught here directly, not via a downstream numeric
        # side effect that might coincidentally still look plausible.
        assert len(calls) == 1, (
            f"template={template!r}: measure_cell_all_h(query_mode='natural') called "
            f"build_natural_query_tokens {len(calls)} time(s), expected exactly 1 -- the dispatch "
            f"silently fell through to the marker branch (MAJOR-1: e.g. a 'natural' vs 'natura1' "
            f"typo in the dispatch's own string comparison)")

        natural_window = calls[0]["tokens"][0, 0, :]
        query_key_id = int(calls[0]["query_key_ids"][0, 0])

        # Reproduce the SAME episode's marker-style window directly via grammar_rd.sample_batch_rd
        # -- the EXACT call measure_cell_all_h itself makes before query_mode='natural' overwrites
        # batch["query_tokens"] (same episode_cfg/batch_size/hop_set/pools/seed reproduces the
        # IDENTICAL draw, torch.Generator's own determinism) -- a REAL "marker path window for the
        # same batch" to diff against, not a synthetic stand-in.
        gen_repro = torch.Generator(device="cpu").manual_seed(0)
        batch_repro = grammar_rd.sample_batch_rd(cfg, 4, gen_repro, hop_set=(1,), pools=pools, device="cpu")
        marker_window = batch_repro["query_tokens"][0, 0, :]

        # (a) the natural window differs from the marker path's window for the same batch --
        # checked PRECISELY: natural_window must equal marker_window's own [buf...,KEY,REL] prefix
        # with ONLY the trailing marker slot dropped (sec 16.1: "dropped entirely, not replaced").
        # A KEY/REL order swap breaks this equality directly (the last two elements would no
        # longer match marker_window's own KEY,REL order).
        assert natural_window.shape[-1] == marker_window.shape[-1] - 1, (
            f"template={template!r}: natural_query_len={natural_window.shape[-1]} is not exactly "
            f"one token shorter than the marker window's query_len={marker_window.shape[-1]}")
        assert torch.equal(natural_window, marker_window[:-1]), (
            f"template={template!r}: natural window {natural_window.tolist()} is not the marker "
            f"window's own [buf...,KEY,REL] prefix {marker_window[:-1].tolist()} -- content diverged "
            f"(MAJOR-2: e.g. a KEY/REL order swap inside build_natural_query_tokens)")
        natural_decoded = tokenizer.decode(natural_window.tolist())
        marker_decoded = tokenizer.decode(marker_window.tolist())
        assert natural_decoded != marker_decoded, (
            f"template={template!r}: natural/marker windows decode to the SAME string: "
            f"{natural_decoded!r}")

        # (b) NO query_id/marker token appears anywhere in the natural window -- checked against
        # the GENUINELY DISTINCT reserved marker (build_reasoning_link_pools's own query_id), not
        # build_natural_pools's collapsed one (see this function's own docstring).
        assert real_marker_id not in natural_window.tolist(), (
            f"template={template!r}: the reserved marker token {real_marker_id} leaked into the "
            f"natural window {natural_window.tolist()} -- sec 16.1 requires it DROPPED, not merely "
            f"unused")

        # (c) the LAST token (the scored q_eff position, forward-B's own "-1" convention) is drawn
        # from the ACTIVE rel_a_ids verb pool, never the entity pool.
        last_token = int(natural_window[-1])
        active_verb_ids = set(pools.rel_a_ids.tolist())
        entity_pool_ids = set(pools.train_name_ids.tolist()) | set(pools.heldout_name_ids.tolist())
        assert last_token in active_verb_ids, (
            f"template={template!r}: natural window's last (scored) token {last_token} is not in "
            f"the active rel_a_ids verb pool {sorted(active_verb_ids)} -- q_eff would be read at "
            f"the WRONG (non-verb) position")
        assert last_token not in entity_pool_ids, (
            f"template={template!r}: natural window's last token {last_token} collides with the "
            f"entity pool -- cannot distinguish a KEY/REL order swap from a correct build")
        if template == "B":
            assert last_token in (succeeded_id_live, replaced_id_live), (
                f"template=B: last token {last_token} is not one of the verified succession-verb "
                f"literals ({succeeded_id_live}, {replaced_id_live})")

        # (d) the entity token appears at its expected slot (index -2 -- KEY, immediately before
        # REL, per build_natural_query_tokens's own registered [buf...,KEY,REL] order).
        entity_slot_token = int(natural_window[-2])
        assert entity_slot_token == query_key_id, (
            f"template={template!r}: expected the KEY slot (index -2) to hold the query entity id "
            f"{query_key_id}, found {entity_slot_token} instead -- KEY/REL order swap (MAJOR-2)")
        assert entity_slot_token in entity_pool_ids, (
            f"template={template!r}: KEY slot token {entity_slot_token} is not itself in the "
            f"entity pool -- unexpected content at the expected entity slot")

    _report(19, "natural query semantic-content verification: measure_cell_all_h's REAL "
                "query_mode='natural' DISPATCH is exercised (not just the builder, closing "
                "MAJOR-1) and its window's DECODED content is checked -- differs from the marker "
                "window (a), no reserved-marker token present (b), last (scored) token is a verb "
                "from the active rel_a_ids pool (c), entity token at its expected KEY slot (d, "
                "closing MAJOR-2) -- Candidates A and B both verified", True)


# ---------------------------------------------------------------------------
# Extra (non-numbered) checks: the sec-12 outcome-routing gates have teeth.
# Not part of the official 14 items, but cheap and worth exercising since
# they are pure functions this build also delivers (sec 12's own
# instruction: "implement the mechanical gates").
# ---------------------------------------------------------------------------

def test_extra_outcome_routing_gates():
    ci_pos = {"mean": 0.2, "ci_low": 0.05, "ci_high": 0.35}
    ci_straddle = {"mean": 0.02, "ci_low": -0.05, "ci_high": 0.09}
    assert rlp.killer_prediction_verdict(ci_pos, ci_straddle, "agree") == "CONFIRM"
    assert rlp.killer_prediction_verdict(ci_straddle, ci_straddle, "agree") == "REFUTE"
    assert rlp.killer_prediction_verdict(ci_pos, ci_straddle, "disagree") == "READOUT-DIVERGENCE"

    assert rlp.leg_b_scale_gate(["agree", "agree", "agree", "disagree"]) == "ELIGIBLE_FOR_TREND_READ"
    assert rlp.leg_b_scale_gate(["agree", "agree", "disagree", "disagree"]) == "AMBIGUOUS"

    assert rlp.readout_form_invalid(True, True) is True
    assert rlp.readout_form_invalid(False, True) is False
    assert rlp.outcome_precedence(True, True) == "READOUT-FORM-INVALID"
    assert rlp.outcome_precedence(False, True) == "AMBIGUOUS"
    assert rlp.outcome_precedence(False, False) is None

    d = rlp.delta_ci_n3([0.5, 0.6, 0.55], [0.4, 0.42, 0.41])
    assert d["ci_low"] < d["mean"] < d["ci_high"]
    print("[Stage-1 extra] outcome-routing gates (sec 12) mechanically fire as pre-registered: PASS")


def test_extra_gate_enforcement_negative_test():
    """This BUILD's own brief, item (c): the sec-16.1 chain's gate
    enforcement (reasoning_link_gate_enforce.py, closing sec 15.2's
    DISCREPANCY finding) "must READ the gate verdict from the output JSON
    and refuse/announce mechanically -- verified by a negative test." Wired
    into the standard Stage -1 suite so `--mode selftest` exercises it too,
    in addition to its own standalone `--selftest` entrypoint (the chain
    calls the SAME module, so a regression here would be caught either
    way)."""
    import reasoning_link_gate_enforce as glinke
    ok = glinke._run_selftest()
    assert ok, ("reasoning_link_gate_enforce's own negative-test fixture suite FAILED -- the "
                "Phase-1b chain's gate enforcement does not have teeth (sec 15.2's own gap, unfixed)")
    print("[Stage-1 extra] gate-enforcement negative test (sec 15.2 fix, this BUILD item (c)): PASS")


ALL_ITEMS = [
    test_item_1_tokenizer_verification, test_item_2_hand_built_composition,
    test_item_3_bigram_shortcut_scorer, test_item_4_label_shuffle_null,
    test_item_5_blank_out_test, test_item_6_surgery_toggle_smoke,
    test_item_7_conv_size_match_assertion, test_item_8_render_adjacency_measurement,
    test_item_9_kernel_t_floor, test_item_10_seed_non_collision,
    test_item_11_causality_assertion, test_item_12_three_entity_worked_example,
    test_item_13_v_conv1d_hook_equivalence, test_item_14_q_conv1d_hook_equivalence,
    test_item_15_candidate_b_verb_verification, test_item_16_natural_template_floor_rederivation,
    test_item_17_natural_query_causality_assertion, test_item_18_natural_q_conv1d_hook_equivalence,
    test_item_19_natural_query_semantic_content_verification,
]


def run_all_selftests() -> bool:
    print("=" * 70)
    print("REASONING-LINK Stage -1 SELF-TESTS (14 items, sec 9, + items 15-18, sec 16.1 Phase-1b, "
          "+ item 19, mini-audit FIX-1)")
    print(f"fla_stub_installed={rlp.FLA_STUB_INSTALLED}")
    print("=" * 70)
    t0 = time.time()
    failures = []
    for fn in ALL_ITEMS:
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"  ** FAILURE in {fn.__name__}: {e}")
    try:
        test_extra_outcome_routing_gates()
    except AssertionError as e:
        failures.append(("test_extra_outcome_routing_gates", str(e)))
        print(f"  ** FAILURE in test_extra_outcome_routing_gates: {e}")
    try:
        test_extra_gate_enforcement_negative_test()
    except AssertionError as e:
        failures.append(("test_extra_gate_enforcement_negative_test", str(e)))
        print(f"  ** FAILURE in test_extra_gate_enforcement_negative_test: {e}")

    wall = time.time() - t0
    print("=" * 70)
    if failures:
        print(f"REASONING-LINK Stage -1: {len(failures)} FAILURE(S) in {wall:.1f}s")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        print("=" * 70)
        return False
    print(f"REASONING-LINK Stage -1: ALL {len(ALL_ITEMS)} ITEMS + extra gate checks PASSED in {wall:.1f}s")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = run_all_selftests()
    sys.exit(0 if ok else 1)
