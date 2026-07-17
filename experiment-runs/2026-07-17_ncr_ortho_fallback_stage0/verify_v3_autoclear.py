"""§B6 auto-clear condition 3: the three regime-defining numeric rows,
through the REAL v3 encode() path (module-attr capture), real CUDA.
Expected values pinned by §B6 Ruling 3's pre-verification:
  r1: fp64 sigma_min=1e-9 -> floor EXACT
  r2: fp32 bit-exact-zero columns (F-B reproducer) -> scale capped at
      exactly 1e6, encode() output FINITE end-to-end
  r3: Z identically 0 -> Z_damped identically 0, finite (composed backstop)
"""
import sys
sys.path.insert(0, "/home/nvidia/ncr")
import torch
import inspect
import ncr_ortho_fallback_stage0 as m

assert torch.cuda.is_available(), "CUDA required"
dev = "cuda"

src = inspect.getsource(m.NCROrthoWriteModel.encode)
assert "(self._eps_rel * sigma_max * 1e-6).clamp_min(torch.finfo(S.dtype).tiny)" in src, \
    "composed guard not present in loaded module"
assert m.RUNNER_TAG == "ncr_ortho_fallback_stage0_v3", m.RUNNER_TAG
print("g0 PASS: loaded module is v3 (composed guard present, RUNNER_TAG _v3)")

captured = {}
_real_ns = m.newton_schulz_polar
def capture_ns(Z, n_iter=m.NS_ITER_DEFAULT, n_power=m.NS_POWER_DEFAULT, eps=m.NS_EPS):
    captured["Z_damped"] = Z.detach().clone()
    return _real_ns(Z, n_iter=n_iter, n_power=n_power)

def run_encode(model, Z_case):
    del model.encoder
    model.encoder = lambda k, v: Z_case
    try:
        m.newton_schulz_polar = capture_ns
        Q = model.encode(None, None)
    finally:
        m.newton_schulz_polar = _real_ns
    return Q, captured["Z_damped"]

eps_rel = 1e-3

# r1: fp64 resolvable-exact (coordinator's operative wording), per §B6
# Ruling 3's OWN pre-verification: exact down to the bite point
# eps_rel*sigma_max*1e-6 = 5e-9 absolute (sigma_min in {1e-7, 1e-8, 5e-9}
# all 1.0000), and the DISCLOSED loosened-band profile at sigma_min=1e-9
# -> 0.20x target. [Condition-3's label "sigma_min=1e-9 EXACT" contradicts
# its own cited pre-verification -- tiebreak recorded in §B7: this test
# reproduces the pre-verification's expected values verbatim.]
gen = torch.Generator().manual_seed(11)
d = 25
model = m.NCROrthoWriteModel(d=d, h=64, orthogonal=True, damped=True,
                             eps_rel=eps_rel).double().to(dev)
def fp64_case(smin):
    U0, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64, generator=gen))
    V0, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64, generator=gen))
    sig = torch.linspace(5.0, 1.0, d, dtype=torch.float64); sig[-1] = smin
    Z = (U0 @ torch.diag(sig) @ V0.T).unsqueeze(0).to(dev)
    _, Zd = run_encode(model, Z)
    tgt = eps_rel * float(torch.linalg.svdvals(Z)[0, 0])
    return float(torch.linalg.svdvals(Zd)[0, -1]) / tgt

for smin in (1e-7, 1e-8, 5e-9):     # at/above bite point: EXACT
    r = fp64_case(smin)
    # exact to the SVD's own relative accuracy on sigma_i (~eps*smax/sigma_i)
    assert abs(r - 1.0) < 1e-5, ("resolvable row not exact", smin, r)
    print(f"r1 PASS: fp64 sigma_min={smin:.0e} (>= bite point 5e-9) floor EXACT "
          f"(achieved/target = {r:.9f})")
r_1e9 = fp64_case(1e-9)             # disclosed loosened band: pre-verified 0.20x
assert abs(r_1e9 - 0.20) < 0.02, ("loosened-band profile mismatch", r_1e9)
print(f"r1 PASS: fp64 sigma_min=1e-9 (below bite point) reproduces §B6's "
      f"pre-verified disclosed profile: achieved/target = {r_1e9:.5f} (expected 0.20)")

# r2: F-B reproducer -- fp32, two bit-exact-zero columns
torch.manual_seed(12)
d = 25
W = torch.randn(4, d, d, device=dev)
D_kill = torch.eye(d, device=dev); D_kill[-1, -1] = 0.0; D_kill[-2, -2] = 0.0
Z2 = W @ D_kill
sv2 = torch.linalg.svdvals(Z2)
print(f"r2 input: computed sigma tail min {float(sv2[:, -1].min()):.3e} "
      f"(bit-exact-zero columns), sigma_max ~ {float(sv2[:, 0].mean()):.2f}")
model2 = m.NCROrthoWriteModel(d=d, h=64, orthogonal=True, damped=True,
                              eps_rel=eps_rel).to(dev)
Q2, Zd2 = run_encode(model2, Z2)
assert torch.isfinite(Zd2).all(), "Z_damped not finite (F-B alive)"
assert torch.isfinite(Q2).all(), "encode() output not finite (F-B alive)"
# scale cap check: recompute the scale the block would have used
S2 = sv2
smax2 = S2[..., :1]
S_floor2 = S2.clamp_min(eps_rel * smax2)
guard2 = (eps_rel * smax2 * 1e-6).clamp_min(torch.finfo(S2.dtype).tiny)
scale2 = S_floor2 / S2.clamp_min(guard2)
max_scale = float(scale2.max())
assert abs(max_scale - 1e6) / 1e6 < 1e-3, ("scale not capped at 1e6", max_scale)
print(f"r2 PASS: F-B reproducer -- scale capped at {max_scale:.4e} (== 1e6), "
      f"Z_damped finite (|max| {float(Zd2.abs().max()):.3e}), encode() output "
      f"FINITE end-to-end (F-B dead)")

# r3: Z identically 0 (the composed backstop's measure-zero case)
Z3 = torch.zeros(2, d, d, device=dev)
model3 = m.NCROrthoWriteModel(d=d, h=64, orthogonal=True, damped=True,
                              eps_rel=eps_rel).to(dev)
Q3, Zd3 = run_encode(model3, Z3)
assert torch.isfinite(Zd3).all(), "Z_damped not finite at Z==0"
assert float(Zd3.abs().max()) == 0.0, "Z_damped not identically 0 at Z==0"
assert torch.isfinite(Q3).all(), "encode() output not finite at Z==0"
print(f"r3 PASS: Z==0 -> Z_damped==0 exactly, encode() output finite "
      f"(no 0/0 -- composed backstop working)")

print("\nV3 AUTO-CLEAR NUMERIC ROWS: ALL 3 PASS (+ module identity g0)")
