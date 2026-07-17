"""D1-fix verification (design doc §B4 -> §B5), run on real CUDA.

Re-runs §B4's exact numeric verification (float64, non-normal Z,
sigma_min in {1e-4, 1e-9, exact 0}, eps_rel=1e-3) -- but through the ACTUAL
fixed encode() code path (module-attr capture of newton_schulz_polar's
input), not a scratch re-implementation, so the numbers certify the code
as deployed. Then the CUDA grad-finiteness smoke through the fixed path,
including the §B4-noted coverage gap: a synthetic rank-deficient
(sigma_min ~ 0) input in production fp32.
"""
import sys
sys.path.insert(0, "/home/nvidia/ncr")
import torch
import ncr_ortho_fallback_stage0 as m

assert torch.cuda.is_available(), "CUDA required"
dev = "cuda"

# Confirm the fixed line is the one loaded (guard decoupled from NS_EPS).
import inspect
src = inspect.getsource(m.NCROrthoWriteModel.encode)
assert "torch.finfo(S.dtype).tiny" in src, "fixed guard not present in loaded module"
assert "S.clamp_min(NS_EPS)" not in src, "old NS_EPS guard still present in encode()"
print("g0 PASS: loaded encode() carries the D1-fixed guard (finfo.tiny), old NS_EPS guard absent")

# ---------------------------------------------------------------------------
# Part A: §B4 numeric floor table, float64, non-normal Z, through real encode()
# ---------------------------------------------------------------------------
def make_nonnormal(d, sigma_min_val, gen):
    """Z = U0 diag(sigma) V0^T with independent random U0 != V0 (non-normal),
    sigma_max = 5.0, sigma_min set exactly to sigma_min_val (incl. literal 0.0)."""
    U0, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64, generator=gen))
    V0, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64, generator=gen))
    sig = torch.linspace(5.0, 1.0, d, dtype=torch.float64)
    sig[-1] = sigma_min_val
    return (U0 @ torch.diag(sig) @ V0.T).to(dev)

captured = {}
_real_ns = m.newton_schulz_polar
def capture_ns(Z, n_iter=m.NS_ITER_DEFAULT, n_power=m.NS_POWER_DEFAULT, eps=m.NS_EPS):
    captured["Z_damped"] = Z.detach().clone()
    return _real_ns(Z, n_iter=n_iter, n_power=n_power)

gen = torch.Generator().manual_seed(0)
eps_rel = 1e-3
print("\nPart A: floor property through the REAL fixed encode() "
      "(float64, non-normal Z, eps_rel=1e-3)")
print(f"{'d':>3} {'raw sigma_min (set)':>20} {'computed raw s_min':>20} "
      f"{'target floor':>14} {'achieved s_min':>16} {'holds?':>7}")
noise_ratios = []
try:
    m.newton_schulz_polar = capture_ns
    for d in (5, 25):          # 5 = §B4's exact replication shape; 25 = production d (K=24+1)
        model = m.NCROrthoWriteModel(d=d, h=64, orthogonal=True, damped=True,
                                     eps_rel=eps_rel).double().to(dev)
        del model.encoder            # detach the nn.Module child so a plain
        for smin in (1e-4, 1e-9, 0.0):   # callable can stand in (test-only)
            Z_case = make_nonnormal(d, smin, gen).unsqueeze(0)   # (1,d,d)
            model.encoder = (lambda Zc: (lambda k, v: Zc))(Z_case)
            _ = model.encode(None, None)
            Zd = captured["Z_damped"]
            sv_raw = torch.linalg.svdvals(Z_case)
            target = eps_rel * float(sv_raw[0, 0])
            achieved = float(torch.linalg.svdvals(Zd)[0, -1])
            exact = achieved >= target * (1 - 1e-10)
            label = "EXACT" if exact else f"~{achieved / target:.2f}x"
            print(f"{d:>3} {smin:>20.3e} {float(sv_raw[0,-1]):>20.3e} "
                  f"{target:>14.6e} {achieved:>16.6e} {label:>8}")
            assert torch.isfinite(Zd).all(), "Z_damped not finite"
            if smin > 0.0:
                # RESOLVABLE regime (computed sigma has full relative accuracy):
                # the D1 operating-regime defect -- floor must be EXACT here.
                assert exact, ("floor not exact in resolvable regime", d, smin,
                               target, achieved)
            else:
                # NOISE-FLOOR regime (raw sigma_min = dtype noise ~eps*sigma_max):
                # exactness is NOT numerically achievable by ANY M@Z construction
                # (SVD backward error ~eps*sigma_max is amplified by scale to
                # O(floor); left-mult cannot raise the rank of an effectively
                # singular matrix). Measure + report; sanity-bound only.
                noise_ratios.append((d, achieved / target))
                assert achieved >= 0.05 * target, (
                    "noise-floor achieved sigma_min catastrophically short",
                    d, target, achieved)
        # characterize the noise-floor spread at production d over extra frames
        if d == 25:
            for _ in range(4):
                Z_case = make_nonnormal(d, 0.0, gen).unsqueeze(0)
                model.encoder = (lambda Zc: (lambda k, v: Zc))(Z_case)
                _ = model.encode(None, None)
                sv_raw = torch.linalg.svdvals(Z_case)
                target = eps_rel * float(sv_raw[0, 0])
                achieved = float(torch.linalg.svdvals(captured["Z_damped"])[0, -1])
                noise_ratios.append((d, achieved / target))
finally:
    m.newton_schulz_polar = _real_ns
print("Part A: RESOLVABLE-regime rows (1e-4, 1e-9 -- incl. far below old NS_EPS) "
      "all EXACT: the D1 defect is fixed on its own axis.")
print(f"Part A finding (noise-floor rows, constructed literal sigma_min=0): "
      f"achieved/target ratios {[f'{r:.2f} (d={dd})' for dd, r in noise_ratios]} "
      f"-- O(1)-approximate, NOT exact; see report/§B5.")

# ---------------------------------------------------------------------------
# Part B: CUDA grad-finiteness through the fixed path (production fp32)
# ---------------------------------------------------------------------------
# B1: healthy regime, full real model (re-run of the prior s2 check on v2 code)
K = 24
torch.manual_seed(0)
model = m.NCROrthoWriteModel(d=K + 1, h=64, orthogonal=True, damped=True,
                             eps_rel=1e-3).to(dev)
keys = torch.randn(256, 16, K + 1, device=dev, requires_grad=True)
values = torch.randn(256, 16, K + 1, device=dev, requires_grad=True)
Z = model.encode(keys, values)
Z.sum().backward()
pn = [float(p.grad.norm()) for p in model.parameters() if p.grad is not None]
assert all(torch.isfinite(torch.tensor(x)) and x > 0 for x in pn), pn
assert torch.isfinite(keys.grad).all() and torch.isfinite(values.grad).all()
print(f"\nB1 PASS: healthy-regime forward+backward on v2 code -- "
      f"{len(pn)} param grad norms finite&nonzero (range [{min(pn):.3e}, {max(pn):.3e}])")

# B2: the §B4 coverage gap -- synthetic sigma_min ~ 0 (rank-deficient) input,
# fp32, through the REAL encode() (encoder monkeypatched to a leaf-derived
# rank-deficient Z so gradient must flow back through M @ Z).
torch.manual_seed(1)
B, d = 8, K + 1
W_leaf = torch.randn(B, d, d, device=dev, requires_grad=True)
D_kill = torch.eye(d, device=dev)
D_kill[-1, -1] = 0.0
D_kill[-2, -2] = 0.0            # rank <= d-2: at least two ~zero singular values
model2 = m.NCROrthoWriteModel(d=d, h=64, orthogonal=True, damped=True,
                              eps_rel=1e-3).to(dev)
del model2.encoder               # same test-only stand-in as Part A
model2.encoder = lambda k, v: W_leaf @ D_kill
Z_raw_probe = (W_leaf @ D_kill).detach()
sv_probe = torch.linalg.svdvals(Z_raw_probe)
print(f"B2 input: computed raw sigma_min range [{float(sv_probe[:, -1].min()):.3e}, "
      f"{float(sv_probe[:, -1].max()):.3e}] (rank-deficient by construction), "
      f"sigma_max ~ {float(sv_probe[:, 0].mean()):.2f}")
# MEASURE, do not crash: this case (computed sigma bit-exact 0.0 in fp32)
# probes the prescribed finfo.tiny guard at its own extreme.
try:
    m.newton_schulz_polar = capture_ns
    Q2 = model2.encode(None, None)
    Zd2 = captured["Z_damped"]
finally:
    m.newton_schulz_polar = _real_ns
zd2_finite = bool(torch.isfinite(Zd2).all())
q2_finite = bool(torch.isfinite(Q2).all())
scale_implied = (1e-3 * sv_probe[:, 0].max() / torch.finfo(torch.float32).tiny)
grad2_finite = None
if q2_finite:
    Q2.sum().backward()
    grad2_finite = bool(W_leaf.grad is not None and torch.isfinite(W_leaf.grad).all())
print(f"B2 RESULT (bit-exact-zero computed sigma, fp32): Z_damped finite={zd2_finite} "
      f"(|Z_damped| max {float(Zd2.abs().max()):.3e}), encode() output finite={q2_finite}, "
      f"grads finite={grad2_finite}; implied scale ~ {float(scale_implied):.2e} "
      f"(S_floor/finfo.tiny) -- if not finite, this is the NEW fp32 finding "
      f"for the delta re-audit (see report/§B5), NOT a pass.")

# B2b: the REALISTIC production trap regime in fp32 -- sigma_min constructed
# tiny (1e-9) so the STORED fp32 matrix's sigma_min lands at fp32 noise level
# (~1e-6-1e-7 relative), NOT bit-exact zero -- i.e. the actual regime SS10
# diagnosed sigma_min random-walking into during training.
gen32b = torch.Generator().manual_seed(7)
U0b, _ = torch.linalg.qr(torch.randn(d, d, generator=gen32b))
V0b, _ = torch.linalg.qr(torch.randn(d, d, generator=gen32b))
sigb = torch.linspace(5.0, 1.0, d)
sigb[-1] = 1e-9                     # below fp32 resolution -> stored matrix noise-level
W_leaf_b = (U0b @ torch.diag(sigb) @ V0b.T).unsqueeze(0).to(dev).requires_grad_(True)
model2b = m.NCROrthoWriteModel(d=d, h=64, orthogonal=True, damped=True,
                               eps_rel=1e-3).to(dev)
del model2b.encoder
model2b.encoder = lambda k, v: W_leaf_b * 1.0   # graph-connected passthrough
sv_b = torch.linalg.svdvals(W_leaf_b.detach())
try:
    m.newton_schulz_polar = capture_ns
    Q2b = model2b.encode(None, None)
    Zd2b = captured["Z_damped"]
finally:
    m.newton_schulz_polar = _real_ns
assert torch.isfinite(Q2b).all(), "encode() not finite in the REALISTIC fp32 trap regime"
Q2b.sum().backward()
assert W_leaf_b.grad is not None and torch.isfinite(W_leaf_b.grad).all()
assert float(W_leaf_b.grad.abs().sum()) > 0
ach_b = float(torch.linalg.svdvals(Zd2b)[0, -1])
tgt_b = 1e-3 * float(sv_b[0, 0])
print(f"B2b PASS (realistic fp32 trap regime): computed raw sigma_min "
      f"{float(sv_b[0, -1]):.3e} (fp32 noise level, sub-NS_EPS, not bit-exact 0) -- "
      f"encode() finite, grads finite & nonzero (norm {float(W_leaf_b.grad.norm()):.3e}); "
      f"achieved/target floor ratio {ach_b / tgt_b:.3f}")

# B3: fp32 RESOLVABLE-regime exactness (sigma_min well above fp32 noise floor):
# the floor must be exact to fp32 precision in the regime fp32 can resolve.
torch.manual_seed(2)
gen32 = torch.Generator().manual_seed(3)
U0, _ = torch.linalg.qr(torch.randn(d, d, generator=gen32))
V0, _ = torch.linalg.qr(torch.randn(d, d, generator=gen32))
sig = torch.linspace(5.0, 1.0, d)
sig[-1] = 1e-4                     # resolvable in fp32 (>> 1.2e-7*sigma_max)
Z32 = (U0 @ torch.diag(sig) @ V0.T).unsqueeze(0).to(dev)
model3 = m.NCROrthoWriteModel(d=d, h=64, orthogonal=True, damped=True,
                              eps_rel=1e-3).to(dev)
del model3.encoder
model3.encoder = lambda k, v: Z32
try:
    m.newton_schulz_polar = capture_ns
    _ = model3.encode(None, None)
    Zd3 = captured["Z_damped"]
finally:
    m.newton_schulz_polar = _real_ns
tgt3 = 1e-3 * float(torch.linalg.svdvals(Z32)[0, 0])
ach3 = float(torch.linalg.svdvals(Zd3)[0, -1])
# fp32 exactness bound: LAPACK sigma_i carries ABSOLUTE error O(eps*sigma_max),
# so relative accuracy at sigma_min=1e-4 (sigma_max=5) is ~1.2e-7*5/1e-4 ~ 6e-3.
# Tolerance 2e-2 = ~3x that intrinsic limit; "exact to fp32 precision" means
# exactly this bound, not fp64-style 1e-10.
assert ach3 >= tgt3 * (1 - 2e-2), (tgt3, ach3)
print(f"B3 PASS: fp32 resolvable-regime floor exact to fp32's intrinsic SVD "
      f"accuracy (achieved/target = {ach3 / tgt3:.5f}; intrinsic limit ~6e-3 rel)")

print("\nD1-FIX VERIFICATION COMPLETE (g0, A, B1, B2, B3) -- resolvable-regime "
      "exactness restored at every magnitude incl. below old NS_EPS; noise-floor "
      "regime measured + reported as O(1)-approximate (see report/§B5 finding).")
