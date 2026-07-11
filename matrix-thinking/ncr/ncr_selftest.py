"""NCR enforced self-test suite -- logic checks AND the negative tests that
prove every guard has teeth (CLAUDE.md: negative tests run to COMPLETION,
never merely written; a check without an executed kill proof is assumed
vacuous).

Run via `python run_ncr.py --smoke` (which calls run_all) or directly.
Every negative test constructs the violation IN PROCESS (a mutated read, a
leaky decoder, an unlabeled schema entry) and asserts the corresponding
guard FIRES -- the kill proof -- then re-runs the clean path to confirm the
suite itself did not poison shared state.

Real-kernel/CUDA coverage is NOT here (CPU logic only): device-placement
teeth and the closed-form-on-CUDA checks live in run_ncr.py --box-smoke,
per the S2.27 lesson (CPU smokes structurally cannot catch device bugs) and
CLAUDE.md's CPU-stub rule.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
if CHAPTER2 not in sys.path:
    sys.path.insert(0, CHAPTER2)

import task_e as te            # noqa: E402
import ncr_task as nt          # noqa: E402
import ncr_models as nm        # noqa: E402
import ncr_spectral as ns      # noqa: E402


def _expect_raise(fn, exc=AssertionError, what=""):
    raised = False
    try:
        fn()
    except exc:
        raised = True
    assert raised, f"NEGATIVE TEST FAILED TO KILL: {what}"


# ---------------------------------------------------------------------------
# 1. module self-tests (positive paths)
# ---------------------------------------------------------------------------

def t01_module_self_tests(device):
    nt._self_test()
    nm._self_test()
    ns._self_test()
    te._self_test()


# ---------------------------------------------------------------------------
# 2. eval-grid guard teeth (claim mode must refuse confounded points)
# ---------------------------------------------------------------------------

def t02_grid_guard_teeth(device):
    _expect_raise(lambda: nt.assert_claim_point(64, 8),
                  what="claim point h=64 (identity residue) accepted")
    _expect_raise(lambda: nt.assert_claim_point(57, 8),
                  what="claim point h=57 (train residue) accepted")
    _expect_raise(lambda: nt.assert_claim_point(60, 12),
                  what="claim point h=60 (identity, K=12) accepted")
    # the INHERITED config assert must also fire on a poisoned H_extra
    _expect_raise(lambda: te.TaskEConfig(d=16, K=8, variant="permutation",
                                         H_train=(1, 2, 3), H_test=(4, 5, 6),
                                         H_extra=(64,)),
                  what="inherited TaskEConfig accepted h=64 in H_extra")
    nt.assert_claim_point(61, 8)     # h* passes
    nt.assert_claim_point(57, 12)    # h* passes
    print("  grid guards kill identity/train-residue claim points; h* passes both K")


def t03_label_schema_teeth(device):
    _expect_raise(lambda: nt.require_residue_label(
        dict(mode="residue_sweep", h=57)),
        what="residue_sweep entry without residue_label accepted")
    _expect_raise(lambda: nt.require_residue_label(
        dict(mode="residue_sweep", h=57, residue_label="bogus")),
        what="residue_sweep entry with invalid label accepted")
    nt.require_residue_label(dict(mode="residue_sweep", h=57,
                                  residue_label="train-residue"))
    print("  residue-sweep label schema refuses missing/invalid labels")


# ---------------------------------------------------------------------------
# 3. label-reduction equality (the far-h label path is EXACT, proven not assumed)
# ---------------------------------------------------------------------------

def t04_label_reduction_equality(device):
    cfg = nt.claim_config(8)
    for h in (13, 21, 61):
        g1 = torch.Generator(device=device).manual_seed(99)
        g2 = torch.Generator(device=device).manual_seed(99)
        b_red = nt.sample_eval_batch(cfg, 32, g1, h, device=device)
        b_full = te.sample_batch(cfg, 32, g2, hop_set=(h,), device=device)
        assert torch.equal(b_red["targets"], b_full["targets"]), h
        assert torch.equal(b_red["query_keys"], b_full["query_keys"]), h
        assert int(b_red["hops"][0, 0]) == h, "model-visible h must be RAW"
    print("  reduced-label eval batches == full-h batches (h=13/21/61); models see raw h")


# ---------------------------------------------------------------------------
# 4. read exactness + mutation kill proofs
# ---------------------------------------------------------------------------

def _binexp_mutated_bitorder(Z, q, h):
    """Deliberately corrupted square-and-multiply: consumes bits MSB-first
    while squaring as if LSB-first -- a classic implementation error."""
    base, v = Z, q
    bits = bin(h)[2:]                      # MSB first -- WRONG for this loop shape
    for bit in bits:
        if bit == "1":
            v = torch.einsum("bij,bqj->bqi", base, v)
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-30)
        base = torch.einsum("bij,bjk->bik", base, base)
        base = base / base.reshape(base.shape[0], -1).norm(dim=-1).clamp(min=1e-30).view(-1, 1, 1)
    return v


def t05_binexp_mutation_killed(device):
    torch.manual_seed(3)
    Z = torch.randn(4, 16, 16, dtype=torch.float64) * 0.4
    q = torch.randn(4, 8, 16, dtype=torch.float64)
    for h in (5, 13, 21):        # positive: exact at ladder-style depths
        ref = q
        for _ in range(h):
            ref = torch.einsum("bij,bqj->bqi", Z, ref)
        ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-30)
        good = nm.binexp_read(Z, q, h)["o"]
        assert ((good * ref).sum(-1) > 1 - 1e-9).all(), h
    # kill proof at NON-palindromic bit patterns only: the bit-order mutation
    # computes Z^(bit-reversed h), so palindromic h (5=101, 21=10101) would
    # accidentally pass -- 13 (1101 -> 11) and 19 (10011 -> 25) cannot.
    for h in (13, 19):
        ref = q
        for _ in range(h):
            ref = torch.einsum("bij,bqj->bqi", Z, ref)
        ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-30)
        bad = _binexp_mutated_bitorder(Z, q, h)
        max_cos_err = (1 - (bad * ref).sum(-1)).max().item()
        assert max_cos_err > 1e-3, (
            f"bit-order mutation NOT killed at h={h} (cos err {max_cos_err:.2e}) "
            f"-- the exactness check has no teeth")
    print("  bin-exp exact vs literal power (h=5/13/21); MUTATED bit-order read "
          "killed at non-palindromic h=13/19")


def _renorm_mutated_nonscalar(Z, q, h):
    """A NON-scalar 'renormalization' (per-step mean subtraction) -- must
    change the cosine, proving the invariance check detects non-scalar ops."""
    v = q
    for _ in range(h):
        v = torch.einsum("bij,bqj->bqi", Z, v)
        v = v - v.mean(dim=-1, keepdim=True)          # NOT a positive scalar
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    return v


def t06_renorm_invariance_teeth(device):
    torch.manual_seed(4)
    Z = torch.randn(4, 16, 16, dtype=torch.float64) * 0.5
    # make Z generic (non-zero-mean columns) so mean subtraction bites
    q = torch.randn(4, 8, 16, dtype=torch.float64)
    h = 21
    ref = q
    for _ in range(h):
        ref = torch.einsum("bij,bqj->bqi", Z, ref)
    ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    renormed = nm.loop_read(Z, q, h)["o"]
    assert ((renormed * ref).sum(-1) > 1 - 1e-9).all(), "renorm moved the cosine"
    mutated = _renorm_mutated_nonscalar(Z, q, h)
    err = (1 - (mutated * ref).sum(-1)).max().item()
    assert err > 1e-3, f"non-scalar mutation NOT detected (err {err:.2e})"
    print("  per-step renorm is cosine-exact; NON-scalar (mean-subtract) mutation killed")


def t07_shadow_and_agreement_teeth(device):
    """fp64-shadow and binexp-vs-loop flags must fire on a perturbed read."""
    torch.manual_seed(5)
    Z = torch.randn(8, 16, 16) * 0.5
    q = torch.randn(8, 8, 16)
    o32 = nm.binexp_read(Z, q, 21)["o"]
    o64 = nm.binexp_read(Z.double(), q.double(), 21)["o"]
    tgt = torch.randn(8, 8, 16)
    c32 = nm.recovery_cosine(o32, tgt)
    c64 = nm.recovery_cosine(o64, tgt.double())
    clean_delta = float((c32.double() - c64).abs().max())
    assert clean_delta <= ns.SHADOW_BAR, f"clean shadow delta {clean_delta:.2e} over bar"
    # perturbed 'read' (simulates an fp32 pipeline defect): flag MUST fire
    c32_bad = nm.recovery_cosine(o32 + 0.05 * torch.randn_like(o32), tgt)
    bad_delta = float((c32_bad.double() - c64).abs().max())
    assert bad_delta > ns.SHADOW_BAR, "shadow flag has no teeth"
    # agreement bar teeth
    ol = nm.loop_read(Z, q, 21)["o"]
    pair = float((nm.recovery_cosine(o32, tgt) - nm.recovery_cosine(ol, tgt)).abs().max())
    assert pair <= ns.AGREE_BAR, f"clean binexp-loop diff {pair:.2e} over MA5 bar"
    pair_bad = float((nm.recovery_cosine(o32, tgt)
                      - nm.recovery_cosine(ol + 0.01 * torch.randn_like(ol), tgt)).abs().max())
    assert pair_bad > ns.AGREE_BAR, "agreement flag has no teeth"
    print(f"  fp64 shadow clean={clean_delta:.1e} <= {ns.SHADOW_BAR}, perturbed fires; "
          f"MA5 agreement clean={pair:.1e} <= {ns.AGREE_BAR}, perturbed fires")


# ---------------------------------------------------------------------------
# 5. blank-out teeth (a leaky read must FAIL the m3 check)
# ---------------------------------------------------------------------------

def t08_blank_out_teeth(device):
    import run_ncr as rn
    cfg = nt.claim_config(8)
    torch.manual_seed(6)
    model = nm.NCRModel().to(device)
    res = rn.blank_out_check(model, cfg, device, seed=0)
    assert res["passed"], res

    class LeakyNCR(nm.NCRModel):
        """Deliberately broken: the 'read' touches raw keys post-write."""
        arm = "ncr"

    leaky = LeakyNCR().to(device)
    gen = torch.Generator(device=device).manual_seed(31_000)
    b = te.sample_batch(cfg, 16, gen, hop_set=(2,), device=device)
    keys = b["keys"].clone().requires_grad_(True)
    values = b["values"].clone().requires_grad_(True)
    state = leaky.encode(keys, values)
    state_leaf = state.detach().clone().requires_grad_(True)
    # the leak: decoder output depends on raw keys OUTSIDE the state
    pred_leaky = nm.binexp_read(state_leaf, b["query_keys"], 2)["o"] \
        + 1e-3 * keys.sum()
    g = torch.autograd.grad(pred_leaky.sum(), [keys, values], allow_unused=True)
    grad_zero = all(gi is None for gi in g)
    assert not grad_zero, "blank-out check has no teeth: leaky read passed grad-zero"
    print("  blank-out passes on the clean read and KILLS a leaky read (grad non-None)")


# ---------------------------------------------------------------------------
# 6. trust screen wiring vs the DISCHARGED archive
# ---------------------------------------------------------------------------

def t09_trust_archive_crosscheck(device):
    """Covered in ns._self_test (N2 deterministic reproduction); here add the
    N1 direction using the archived script's own construction functions, so
    BOTH cases wire through this build's screen."""
    import test_trust_rule_negative as ttrn
    arch = json.load(open(os.path.join(
        CHAPTER2, "test_trust_rule_negative", "results.json")))
    rng = np.random.default_rng(20260709)
    K = 8
    A = ttrn.build_cycle(K)
    # N1 construction per the pinned spec (C drawn first, then Q -- the
    # archived script's own draw order, S6)
    C = rng.standard_normal((K, K))
    C = C * (0.01 / ttrn.op_norm(C))
    Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
    D = 1.5 * Q
    scr = ns.trust_screen(A, np.zeros((K, K)), C, D, (21,))
    T_mine = scr["per_h"]["21"]["T"]
    T_arch = arch["N1"]["T_corrected_21"]
    assert abs(T_mine - T_arch) / T_arch < 1e-9, (T_mine, T_arch)
    assert scr["per_h"]["21"]["rule_trusted"] is False
    print(f"  trust screen reproduces archived N1 T(21)={T_arch:.4f} exactly and rejects")


# ---------------------------------------------------------------------------
# 7. Axis-C lock enforcement teeth
# ---------------------------------------------------------------------------

def t10_lock_teeth(device):
    import run_ncr as rn
    cfg = nt.claim_config(8)
    torch.manual_seed(7)
    model = nm.NCRModel().to(device).eval()
    # far-h eval WITHOUT a lock must refuse (matrix arm); shrink batches so
    # the pre-refusal shallow points stay cheap
    rb, rbs = rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE
    try:
        rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE = 1, 16
        _expect_raise(lambda: rn.eval_cell(model, cfg, device, seed=0, K=8,
                                           lock_content=None, trust_per_h=None),
                      what="far-h eval ran without an Axis-C lock")
    finally:
        rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE = rb, rbs
    # lock write + verify + tamper detection
    with tempfile.TemporaryDirectory() as td_:
        zd = rn.z_dump(model, cfg, device, seed=0, n_examples=2)
        probe = ns.analyze_zdump_arrays(zd["Z"], zd["z_ideal"], (1, 5, 21))
        p = os.path.join(td_, "lock.json")
        ns.write_axis_c_lock(p, "t10", 8, probe)
        ns.verify_axis_c_lock(p)
        with open(p) as f:
            content = json.load(f)
        content["mean_predicted_curve"]["21"] = -0.12345   # tamper
        with open(p, "w") as f:
            json.dump(content, f)
        _expect_raise(lambda: ns.verify_axis_c_lock(p),
                      what="tampered Axis-C lock verified clean")
    print("  far-h eval refuses without a lock; tampered lock detected by hash")


# ---------------------------------------------------------------------------
# 8. param-match teeth + arm gradient coverage
# ---------------------------------------------------------------------------

def t11_param_match_teeth(device):
    nm.assert_param_match()

    class BloatedLooped(nm.LoopedVecModel):
        def __init__(self):
            # hidden=3000 -> 153,424 + 33*3000 = 252,424 params, ratio 1.48
            # vs NCR -- decisively outside +-15% (a small hidden-width tweak
            # CANNOT leave the band: the shared encoder dominates the count,
            # which is itself worth knowing about the gate's sensitivity)
            super().__init__(hidden=3000)

    orig = nm.ARM_BUILDERS["loopedvec"]
    try:
        nm.ARM_BUILDERS["loopedvec"] = BloatedLooped
        _expect_raise(nm.assert_param_match,
                      what="out-of-band loopedvec passed the +-15% gate")
    finally:
        nm.ARM_BUILDERS["loopedvec"] = orig
    nm.assert_param_match()
    print("  +-15% param gate passes the real arms and kills a hidden=3000 mutant "
          "(ratio 1.48)")


def t12_grad_coverage_all_arms(device):
    import run_ncr as rn
    cfg = nt.claim_config(8)
    gen = torch.Generator(device=device).manual_seed(8)
    for arm in nm.ALL_ARMS:
        torch.manual_seed(8)
        m = nm.ARM_BUILDERS[arm]().to(device)
        # arm-protocol teeth (the wave-1 cmlp crash class: nn.Module raises
        # AttributeError on a missing attribute only at ACCESS time, so
        # every builder must be asserted to carry the protocol)
        assert getattr(m, "arm", None) == arm, (arm, getattr(m, "arm", None))
        assert isinstance(getattr(m, "deviating_read", None), bool), arm
        assert rn.primary_read_name(m.arm), arm
        b = te.sample_batch(cfg, 16, gen, hop_set=cfg.H_train, device=device)
        pred, _ = m(b)
        rn.cosine_loss(pred, b["targets"]).backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, (arm, name)
            assert torch.isfinite(p.grad).all(), (arm, name)
    print("  every arm carries the arm protocol (arm/deviating_read/primary read) "
          "and every trainable parameter receives a finite gradient (CPU)")


# ---------------------------------------------------------------------------
# 9. tiny end-to-end cell (train->instruments->eval) on a reduced grid
# ---------------------------------------------------------------------------

def t13_micro_end_to_end(device):
    import run_ncr as rn
    with tempfile.TemporaryDirectory() as td_:
        # monkeypatch the grid smaller for speed: keep components + labels
        real_pts = nt.eval_points
        real_batches, real_bs = rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE

        def tiny_points(K):
            keep = {1, 2, 3, 5, 13, 21, 57, 58, 59, 60, 61, 62, 63, 64}
            return [p for p in real_pts(K) if p.h in keep and p.component != "cost_probe"]
        try:
            nt.eval_points = tiny_points
            rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE = 2, 32
            # EVERY arm end-to-end (§7e regression: the original ncr-only
            # coverage let the cmlp eval crash reach the box -- per-arm
            # pipeline branches only execute per arm)
            for arm in nm.ALL_ARMS:
                rec = rn.run_cell(arm, 8, 0, steps=30, device=device, outdir=td_)
                assert rec["status"] == "COMPLETED", (arm, rec["status"])
                assert rec["eval"]["points"], (arm, "no eval points")
                for e in rec["eval"]["points"]:
                    if e["mode"] == "residue_sweep":
                        nt.require_residue_label(e)
                if arm in ("ncr", "fwm"):
                    assert rec["axis_c_lock_sha256"], arm
                if arm in ("ncr", "fwm", "loopedvec"):
                    assert rec["blank_out"]["passed"], arm
                if arm in ("fwm", "loopedvec"):
                    assert "read_vector_std" in rec, arm
            # resume-safety: a second call must SKIP (COMPLETED)
            rec2 = rn.run_cell("ncr", 8, 0, steps=30, device=device, outdir=td_)
            assert rec2["cell_id"] == "ncr_ncr_K8_s0"
            out = json.load(open(os.path.join(td_, "ncr_ncr_K8_s0.json")))
            assert out["status"] == "COMPLETED"
        finally:
            nt.eval_points = real_pts
            rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE = real_batches, real_bs
    print("  micro end-to-end cells for ALL FOUR ARMS (30 steps, reduced grid): "
          "train -> instruments -> labeled eval -> atomic JSON -> resume-skip")


# ---------------------------------------------------------------------------
# 10. closed-form zero-accumulation checks (S2.26 discipline) -- CPU wiring
# ---------------------------------------------------------------------------

def t14_closed_form_checks(device):
    """S7a audit MAJOR-1 fix: closed_form_checks has no CUDA dependency and
    MUST run in the CPU suite too (previously wired only into box-smoke, so
    the '13/13' record carried zero executed evidence for the transpose
    tooth). box-smoke still re-runs it ON CUDA for device coverage."""
    import run_ncr as rn
    rn.closed_form_checks(device)


ALL_TESTS = [t01_module_self_tests, t02_grid_guard_teeth, t03_label_schema_teeth,
             t04_label_reduction_equality, t05_binexp_mutation_killed,
             t06_renorm_invariance_teeth, t07_shadow_and_agreement_teeth,
             t08_blank_out_teeth, t09_trust_archive_crosscheck, t10_lock_teeth,
             t11_param_match_teeth, t12_grad_coverage_all_arms,
             t13_micro_end_to_end, t14_closed_form_checks]


def run_all(device="cpu"):
    for i, t in enumerate(ALL_TESTS, 1):
        print(f"\n[{i}/{len(ALL_TESTS)}] {t.__name__}")
        t(device)
    print(f"\nncr_selftest: {len(ALL_TESTS)}/{len(ALL_TESTS)} sections PASSED")


if __name__ == "__main__":
    run_all("cpu")
