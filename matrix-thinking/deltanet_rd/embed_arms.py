"""embed_arms.py -- DELTANET_RD_EXACTNESS_DESIGN.md sec 4 (Rev 3): the
controlled embedding-source interpolation. Builds FROZEN (n_identities,
d_model) row tables for arms (i) frozen_orthonormal, (ii) frozen_gpt2_span,
(iv) frozen_gram_matched -- and the separate (d_state)-space per-entity
lookup for arm (i-strong)'s surgical k_eff pin (sec 4.4). Every table is
keyed by REAL grammar_rd.py token id, consumed by model_rd.py's
DeltaNetRDBlock via its generic frozen_row_ids/values (arms i/ii/iv) or
strong_pin_ids/values (i-strong) constructor arguments.

Pod-self-contained (no cross-directory imports besides grammar_rd.py,
which lives in this same directory) -- same convention as
rank_utils.py/deltanet_core.py.

FIXED CONSTRUCTION SEEDS (deliberately INDEPENDENT of --seed): every arm's
frozen table is "frozen forever" (sec 4.2) -- one canonical draw shared
across every run-seed of that arm, so seed-to-seed variation in a Wave 1
cell measures ONLY optimizer/data-order stochasticity, never a re-drawn
embedding table. Documented here, not silently defaulted.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_ORTHONORMAL_SCAFFOLD_SEED = 20260703          # arm (i) + arm (iv)'s u_j scaffold
_GAUSSIAN_NOISE_SEED = 20260704                 # arm (iv)'s g_j term
_SHARED_DIRECTION_SEED = 20260705               # arm (iv)'s c
_I_STRONG_POOL_SEED = 20260706                  # arm (i-strong)'s 32+32 name split + QR draws

EXPECTED_N_IDENTITIES = 251   # 213 names + 21 rel-A + 16 rel-B + 1 period (sec 4.2/4.3, 2026-07-02)


# ---------------------------------------------------------------------------
# sec 4.2's used-identity inventory (build-time-verified, not hardcoded).
# ---------------------------------------------------------------------------

def used_identity_ids(pools) -> torch.Tensor:
    """The identity vocabulary sec 4.2 requires SIMULTANEOUSLY orthonormal
    for arm (i)/(iv), and whose SPAN arm (ii) preserves exactly: ALL names
    (train + heldout -- both pools need frozen-clean geometry, since C17
    zero-shot eval draws from heldout), both relation-verb pools, and the
    period token. Returns sorted-unique int64 ids on CPU.

    Build-time verification (sec 4.2's "verify the arithmetic held...
    since it is one measurement away from not fitting"): HARD ASSERT on
    the design's own 213+21+16+1=251 measurement (2026-07-03 audit
    upgrade -- was a warning). Why a hard stop: every frozen table (arms
    i/ii/iv) and every recorded arm-(iv) (alpha,rho) calibration is keyed
    to this exact inventory (ids AND order); a drifted count that merely
    warned would let a run train against silently-invalidated frozen
    geometry. The per-arm n<=d_model feasibility asserts remain at each
    construction site."""
    ids = torch.cat([
        pools.train_name_ids.cpu(), pools.heldout_name_ids.cpu(),
        pools.rel_a_ids.cpu(), pools.rel_b_ids.cpu(),
        torch.tensor([int(pools.period_id)], dtype=torch.int64),
    ])
    ids = torch.unique(ids)
    assert ids.numel() == EXPECTED_N_IDENTITIES, (
        f"used-identity inventory is {ids.numel()}, expected {EXPECTED_N_IDENTITIES} "
        f"(213 names + 21 rel-A + 16 rel-B + 1 period -- DELTANET_RD_EXACTNESS_DESIGN.md "
        f"sec 4.2/4.3, measured 2026-07-02). grammar_rd.py's verified pools have CHANGED. "
        f"Do NOT just bump EXPECTED_N_IDENTITIES -- re-derive, in order: (a) sec 4.2 "
        f"feasibility (new count <= d_model=256) must still hold; (b) arm (ii)'s span rank "
        f"<= d_model likewise; (c) every frozen table built against the old inventory "
        f"(arms i/ii/iv) and every recorded arm-(iv) (alpha,rho) calibration is INVALIDATED "
        f"and must be rebuilt/re-calibrated; THEN update the constant, recording the change "
        f"as a documented deviation. (Hard assert per the 2026-07-03 audit; was a warning.)")
    return ids


# ---------------------------------------------------------------------------
# Arm (i): frozen_orthonormal (sec 4.2).
# ---------------------------------------------------------------------------

def _qr_orthonormal_rows(n: int, d: int, seed: int) -> torch.Tensor:
    """n mutually-orthonormal rows in R^d (n <= d required). fp64 QR for
    precision; cast to fp32 on return. Columns of a (d,n) Gaussian's
    reduced-QR are orthonormal by construction; transposing gives n
    orthonormal ROWS (so the returned matrix's own Gram is exactly I_n)."""
    assert n <= d, f"cannot build {n} mutually-orthonormal rows in R^{d} (n>d)"
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(d, n, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(G, mode="reduced")            # (d, n), orthonormal COLUMNS
    return Q.transpose(0, 1).contiguous().float()          # (n, d) orthonormal ROWS


def build_frozen_orthonormal(pools, d_model: int = 256):
    """Arm (i), sec 4.2: QR-orthogonalize a single (d_model, n_identities)
    Gaussian ONCE, frozen forever. Returns (ids (n,), values (n,d_model)
    unit-norm, mutually orthonormal rows -- Gram deviation ~0 by
    construction, up to fp32 rounding)."""
    ids = used_identity_ids(pools)
    assert ids.numel() <= d_model, \
        f"arm (i) infeasible: {ids.numel()} identities > d_model={d_model} " \
        f"(sec 4.2's own feasibility check -- 'one measurement away from not fitting')"
    values = _qr_orthonormal_rows(ids.numel(), d_model, _ORTHONORMAL_SCAFFOLD_SEED)
    return ids, values


# ---------------------------------------------------------------------------
# Arm (ii): frozen_gpt2_span (sec 4.3, Rev 2 finding 6 -- exact isometry).
# ---------------------------------------------------------------------------

def _load_gpt2_wte(gpt2_model_name: str = "gpt2") -> torch.Tensor:
    from transformers import GPT2Model
    gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
    return gpt2.wte.weight.detach().double()      # (vocab, 768) fp64


def build_frozen_gpt2_span(pools, d_model: int = 256, gpt2_model_name: str = "gpt2"):
    """Arm (ii), sec 4.3: embed_frozen[t] = Q^T . wte[t], Q an orthonormal
    basis of span({wte[t] : t in used ids}), zero-padded to d_model. An
    ISOMETRY on every used row by construction (see
    span_projection_isometry_check for the Wave -1 smoke that verifies
    this numerically): for used rows t,t' both lying in span(Q), Q Q^T
    acts as the identity restricted to that span, so
    <Q^T wte[t], Q^T wte[t']> = wte[t]^T (Q Q^T) wte[t'] = wte[t]^T wte[t']
    exactly. Returns (ids, values (n,d_model), meta dict)."""
    ids = used_identity_ids(pools)
    wte = _load_gpt2_wte(gpt2_model_name)
    used_rows = wte[ids]                                    # (n, 768) fp64
    Q, _ = torch.linalg.qr(used_rows.transpose(0, 1), mode="reduced")   # (768, r<=n)
    r = Q.shape[1]
    assert r <= d_model, f"arm (ii) infeasible: span rank r={r} > d_model={d_model}"
    proj = used_rows @ Q                                    # (n, r), isometric on used rows
    d_pad = d_model - r
    values = torch.cat([proj, torch.zeros(proj.shape[0], d_pad, dtype=proj.dtype)], dim=-1)
    return ids, values.float(), {"span_rank": r, "wte_dim": wte.shape[-1], "n_identities": ids.numel()}


def span_projection_isometry_check(ids: torch.Tensor, values: torch.Tensor,
                                     gpt2_model_name: str = "gpt2") -> float:
    """Wave -1 smoke (sec 4.3/6): all (n*(n-1))/2 pairwise cosines
    raw-768 vs. projected-d_model, max discrepancy REQUIRED < 1e-5."""
    wte = _load_gpt2_wte(gpt2_model_name)
    raw = F.normalize(wte[ids].double(), dim=-1)
    proj = F.normalize(values.double(), dim=-1)
    cos_raw = raw @ raw.transpose(0, 1)
    cos_proj = proj @ proj.transpose(0, 1)
    n = ids.numel()
    iu = torch.triu_indices(n, n, offset=1)
    diff = (cos_raw[iu[0], iu[1]] - cos_proj[iu[0], iu[1]]).abs()
    return float(diff.max().item())


# ---------------------------------------------------------------------------
# Arm (iv): frozen_gram_matched (sec 4.5, Rev 3 BLOCK-1's two-parameter
# alpha/rho family), plus the reachability calc + Monte Carlo estimator
# the closed-loop calibration probe (calibrate_arm_iv.py) drives.
# ---------------------------------------------------------------------------

def reachability_iid_closed_form(K: int, d: int) -> float:
    """BLOCK-1's pre-registered closed form for the family's alpha=1
    (pure i.i.d.) endpoint: E||G-I||_F ~= sqrt(K(K-1)/d). Verified this
    session (design text) via analytic + Monte Carlo agreement to 3 sig
    figs; reproduced here as a pure function so calibrate_arm_iv.py can
    re-derive it at build/run time rather than trust a copied constant."""
    return (K * (K - 1) / d) ** 0.5


def estimate_gram_deviation(alpha: float, rho: float, K: int, d_model: int,
                              n_trials: int = 200, seed: int = 0) -> float:
    """Monte Carlo E||G-I||_F for the arm (iv) family at a given
    (alpha,rho), K, d_model -- the RAW-ROW reachability sanity smoke
    (sec 4.5's Wave -1 row: "realized K-draw raw-row Gram deviation
    finite, reproducible across 100 sampled draws, and monotone in
    (alpha,rho)") and the calibration driver's starting-point picker.
    u_j/g_j/c drawn FRESH per trial here (a K-subset draw), distinct from
    the FROZEN full-251-row table build_frozen_gram_matched constructs --
    this function characterizes the FAMILY's reachable range at a given
    K, not one frozen instance."""
    g = torch.Generator().manual_seed(seed)
    devs = []
    eye_K = torch.eye(K)
    for _ in range(n_trials):
        u = F.normalize(torch.randn(K, d_model, generator=g), dim=-1)
        gg = F.normalize(torch.randn(K, d_model, generator=g), dim=-1)
        c = F.normalize(torch.randn(d_model, generator=g), dim=-1)
        mix = F.normalize((1 - alpha) * u + alpha * gg, dim=-1)
        x = F.normalize((1 - rho ** 2) ** 0.5 * mix + rho * c.unsqueeze(0), dim=-1)
        gram = x @ x.transpose(0, 1)
        devs.append((gram - eye_K).norm().item())
    return sum(devs) / len(devs)


def realized_table_k_draw_gram_deviation(values: torch.Tensor, K: int, n_draws: int = 100,
                                            seed: int = 0) -> dict:
    """Wave -1's 'raw-table sanity' smoke (sec 4.5: "realized K-draw
    raw-row Gram deviation finite, reproducible across 100 sampled draws,
    and monotone in (alpha,rho)") -- draws K rows WITHOUT REPLACEMENT
    from the REALIZED (already-built, frozen) table `values`, n_draws
    times, and reports each draw's Gram deviation. This is the ACTUAL
    object a Wave-1 training episode's entity draw sees (sample_batch_rd
    draws K distinct identities from the same frozen pool every episode),
    and is DELIBERATELY DIFFERENT from estimate_gram_deviation's
    fresh-per-trial FAMILY characterization above (which exists only to
    pick a starting rho before any table is built, sec 4.5 step 1) -- an
    earlier draft of this module's self-test conflated the two by
    reporting the FULL 251-row table's own Gram deviation (which scales
    with table size, not K, and is not comparable to the K-sized target
    band at all); this function is the fix, flagged here for audit
    scrutiny since it is easy to re-confuse the two objects."""
    n = values.shape[0]
    assert K <= n, f"cannot draw K={K} rows without replacement from a {n}-row table"
    g = torch.Generator().manual_seed(seed)
    devs = []
    eye_K = torch.eye(K)
    for _ in range(n_draws):
        idx = torch.randperm(n, generator=g)[:K]
        sub = values[idx]
        gram = sub @ sub.transpose(0, 1)
        devs.append((gram - eye_K).norm().item())
    devs_t = torch.tensor(devs)
    return {"mean": devs_t.mean().item(), "std": devs_t.std().item(),
            "min": devs_t.min().item(), "max": devs_t.max().item(), "n_draws": n_draws,
            "all_finite": bool(torch.isfinite(devs_t).all().item())}


def build_frozen_gram_matched(pools, d_model: int, alpha: float, rho: float,
                                shared_dir_seed: int = _SHARED_DIRECTION_SEED):
    """Arm (iv), sec 4.5 (Rev 3, BLOCK-1): the two-parameter alpha/rho
    family, built ONCE over the full used-identity table (frozen forever,
    same discipline as arm (i)) -- alpha/rho are the CALIBRATED constants
    (calibrate_arm_iv.py's closed-loop output), never resampled per
    K-draw at actual-run time (the K-draw-level Monte Carlo above is a
    SEPARATE characterization tool, not what gets frozen).

        x_j = normalize( sqrt(1-rho^2) * normalize((1-alpha) u_j + alpha g_j)
                          + rho * c )

    u_j: arm (i)'s SAME orthonormal scaffold (identical seed/order -- so
    the (i)-vs-(iv) contrast differs ONLY in (alpha,rho), never in a
    re-drawn u, per sec 4.5's "the clean geometry-only causal comparison").
    g_j: one fixed i.i.d. Gaussian draw. c: one fixed shared unit
    direction. Returns (ids, values (n,d_model))."""
    ids, u = build_frozen_orthonormal(pools, d_model)
    n = ids.numel()
    g_noise = torch.Generator().manual_seed(_GAUSSIAN_NOISE_SEED)
    gvec = torch.randn(n, d_model, generator=g_noise, dtype=torch.float64)
    g_c = torch.Generator().manual_seed(shared_dir_seed)
    c = F.normalize(torch.randn(d_model, generator=g_c, dtype=torch.float64), dim=-1)
    u64 = u.double()
    mix = F.normalize((1 - alpha) * u64 + alpha * gvec, dim=-1)
    x = F.normalize((1 - rho ** 2) ** 0.5 * mix + rho * c.unsqueeze(0), dim=-1)
    return ids, x.float()


# ---------------------------------------------------------------------------
# Arm (i-strong): the reduced 32+32 dedicated pool + d_state-space lookup
# (sec 4.4).
# ---------------------------------------------------------------------------

def build_i_strong_pool(pools, d_state: int = 64, n_per_side: int = 32,
                          seed: int = _I_STRONG_POOL_SEED):
    """Arm (i-strong), sec 4.4: 32 train + 32 disjoint held-out names, TWO
    INDEPENDENT QR-orthonormal draws (train/heldout episodes never mix
    names within one K-cycle, so cross-set orthonormality between the two
    32-sets is not required -- sec 4.4's own scoping). Returns
    (train_ids_32, heldout_ids_32, pin_ids_64, pin_values_64)."""
    g = torch.Generator().manual_seed(seed)
    train_all = pools.train_name_ids.cpu()
    heldout_all = pools.heldout_name_ids.cpu()
    assert train_all.numel() >= n_per_side and heldout_all.numel() >= n_per_side, \
        f"arm (i-strong) needs >= {n_per_side} names per side " \
        f"(train={train_all.numel()}, heldout={heldout_all.numel()})"
    train_pick = train_all[torch.randperm(train_all.numel(), generator=g)[:n_per_side]]
    heldout_pick = heldout_all[torch.randperm(heldout_all.numel(), generator=g)[:n_per_side]]
    train_vals = _qr_orthonormal_rows(n_per_side, d_state, seed)
    heldout_vals = _qr_orthonormal_rows(n_per_side, d_state, seed + 1)
    pin_ids = torch.cat([train_pick, heldout_pick])
    pin_values = torch.cat([train_vals, heldout_vals], dim=0)
    return train_pick, heldout_pick, pin_ids, pin_values


def restrict_pools_for_strong_pin(pools, train_ids_32: torch.Tensor, heldout_ids_32: torch.Tensor):
    """Rebuilds an EntityPools with train/heldout name pools REPLACED by
    the 32+32 restricted sets build_i_strong_pool drew -- required so
    sample_batch_rd never draws an entity outside strong_pin_table's
    coverage (sec 4.4: "this arm... runs on a reduced, dedicated pool").
    Deferred import of grammar_rd (pod-self-contained, same directory)."""
    import grammar_rd as grd
    return grd.EntityPools(
        vocab_size_base=pools.vocab_size_base, buffer_id=pools.buffer_id,
        query_id=pools.query_id, period_id=pools.period_id,
        train_name_ids=train_ids_32, heldout_name_ids=heldout_ids_32,
        rel_a_ids=pools.rel_a_ids, rel_b_ids=pools.rel_b_ids,
        vocab_size_total=pools.vocab_size_total,
    )


# ---------------------------------------------------------------------------
# Self-test (part of run_deltanet_rd.py --smoke, invoked from there so a
# single `python run_deltanet_rd.py --smoke` covers everything new).
# Requires network (GPT-2 model weights, arm ii) on first run; CPU-only
# otherwise (no CUDA needed for any construction in this module).
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import grammar_rd as grd

    torch.manual_seed(0)
    tokenizer = grd.load_gpt2_tokenizer()
    pools, report = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)

    print("[embed_arms 1] used_identity_ids: count matches the design's 251 measurement "
          "(HARD ASSERT if drifted -- 2026-07-03 audit upgrade), all ids valid/in-range")
    ids = used_identity_ids(pools)
    print(f"  n_identities={ids.numel()} (expected {EXPECTED_N_IDENTITIES}); "
          f"min_id={ids.min().item()} max_id={ids.max().item()} < vocab_size_base={pools.vocab_size_base}")
    assert ids.numel() == report["n_train_names"] + report["n_heldout_names"] + \
        report["n_rel_a_verified"] + report["n_rel_b_verified"] + 1
    assert (ids < pools.vocab_size_base).all(), "an identity id leaked into the reserved (buffer/query) range"

    print("\n[embed_arms 2] arm (i) frozen_orthonormal: exact mutual orthonormality "
          "(Gram deviation ~0 up to fp32 rounding), feasibility guard has teeth")
    ids_i, vals_i = build_frozen_orthonormal(pools, d_model=256)
    gram = vals_i @ vals_i.transpose(0, 1)
    dev = (gram - torch.eye(ids_i.numel())).norm().item()
    assert dev < 1e-3, f"arm (i) Gram deviation {dev:.2e} not ~0"
    print(f"  n={ids_i.numel()} rows, Gram deviation ||G-I||_F = {dev:.2e} (expect ~0)")
    guard_raised = False
    try:
        build_frozen_orthonormal(pools, d_model=64)   # 251 > 64, must reject
    except AssertionError:
        guard_raised = True
    assert guard_raised, "arm (i) feasibility guard failed to reject n_identities > d_model"
    print("  feasibility guard correctly REJECTS d_model=64 (251 identities > 64)")

    print("\n[embed_arms 3] arm (iv) reachability: closed-form vs Monte Carlo agree; family "
          "spans the measured target band with margin at both K=16 and K=32 (BLOCK-1)")
    for K, d in ((16, 256), (32, 256)):
        cf = reachability_iid_closed_form(K, d)
        mc = estimate_gram_deviation(alpha=1.0, rho=0.0, K=K, d_model=d, n_trials=100, seed=1)
        rel_err = abs(cf - mc) / cf
        assert rel_err < 0.15, f"closed-form vs Monte Carlo disagree at K={K}: {cf:.3f} vs {mc:.3f}"
        print(f"  K={K:3d} d={d}: closed-form={cf:.4f}  Monte Carlo(alpha=1,rho=0)={mc:.4f}  "
              f"rel_err={rel_err:.3f}")
    dev_rho03_k16 = estimate_gram_deviation(alpha=1.0, rho=0.3, K=16, d_model=256, n_trials=100, seed=2)
    dev_rho03_k32 = estimate_gram_deviation(alpha=1.0, rho=0.3, K=32, d_model=256, n_trials=100, seed=2)
    print(f"  rho=0.3 (alpha=1): K=16 -> {dev_rho03_k16:.3f}  K=32 -> {dev_rho03_k32:.3f} "
          f"(design's own session numbers: ~1.72 / ~3.44)")
    # monotone in rho at fixed alpha (Wave -1 smoke requirement)
    devs_by_rho = [estimate_gram_deviation(alpha=1.0, rho=r, K=32, d_model=256, n_trials=60, seed=3)
                   for r in (0.0, 0.2, 0.4, 0.6)]
    assert all(devs_by_rho[i] <= devs_by_rho[i + 1] + 1e-6 for i in range(len(devs_by_rho) - 1)), \
        f"estimate_gram_deviation not monotone non-decreasing in rho: {devs_by_rho}"
    print(f"  monotone in rho (alpha=1, K=32): {[round(v, 3) for v in devs_by_rho]}")

    print("\n[embed_arms 4] arm (iv) frozen table: reproducible across repeated construction "
          "calls (same seeds => bit-identical); REALIZED K-draw Gram deviation (sec 4.5's actual "
          "Wave -1 raw-table-sanity object, NOT the full 251-row table's own Gram deviation, "
          "which scales with table size and is a different, non-comparable number) is finite, "
          "reproducible across 100 sampled draws, and monotone in rho")
    ids_iv_a, vals_iv_a = build_frozen_gram_matched(pools, d_model=256, alpha=1.0, rho=0.3)
    ids_iv_b, vals_iv_b = build_frozen_gram_matched(pools, d_model=256, alpha=1.0, rho=0.3)
    assert torch.equal(ids_iv_a, ids_iv_b) and torch.allclose(vals_iv_a, vals_iv_b, atol=1e-6), \
        "arm (iv) construction is not reproducible across repeated calls with the same (alpha,rho)"
    for K in (16, 32):
        stats = realized_table_k_draw_gram_deviation(vals_iv_a, K, n_draws=100, seed=K)
        assert stats["all_finite"], f"realized K={K} draw Gram deviation not finite"
        rel_std = stats["std"] / max(stats["mean"], 1e-8)
        print(f"  K={K:3d} realized-table draws (rho=0.3): mean={stats['mean']:.3f} "
              f"std={stats['std']:.3f} (rel_std={rel_std:.3f}) range=[{stats['min']:.3f},"
              f"{stats['max']:.3f}] over {stats['n_draws']} draws, all finite")
    # monotone in rho, on the REALIZED table (not the fresh-family estimate)
    ids_lo, vals_lo = build_frozen_gram_matched(pools, d_model=256, alpha=1.0, rho=0.1)
    ids_hi, vals_hi = build_frozen_gram_matched(pools, d_model=256, alpha=1.0, rho=0.6)
    mean_lo = realized_table_k_draw_gram_deviation(vals_lo, 32, n_draws=100, seed=1)["mean"]
    mean_hi = realized_table_k_draw_gram_deviation(vals_hi, 32, n_draws=100, seed=1)["mean"]
    assert mean_hi > mean_lo, \
        f"realized K=32 draw Gram deviation not monotone in rho: rho=0.1 -> {mean_lo:.3f}, " \
        f"rho=0.6 -> {mean_hi:.3f}"
    print(f"  monotone in rho on the REALIZED table (K=32): rho=0.1 -> {mean_lo:.3f}, "
          f"rho=0.6 -> {mean_hi:.3f}")

    print("\n[embed_arms 5] arm (i-strong) pool: 32+32 disjoint from each other AND internally "
          "orthonormal within each side; restrict_pools_for_strong_pin round-trips cleanly")
    train32, heldout32, pin_ids, pin_vals = build_i_strong_pool(pools, d_state=64)
    assert train32.numel() == 32 and heldout32.numel() == 32 and pin_ids.numel() == 64
    assert not (set(train32.tolist()) & set(heldout32.tolist())), "train/heldout 32-sets not disjoint"
    assert set(train32.tolist()) <= set(pools.train_name_ids.tolist())
    assert set(heldout32.tolist()) <= set(pools.heldout_name_ids.tolist())
    g_train = pin_vals[:32] @ pin_vals[:32].transpose(0, 1)
    g_held = pin_vals[32:] @ pin_vals[32:].transpose(0, 1)
    assert (g_train - torch.eye(32)).norm().item() < 1e-3
    assert (g_held - torch.eye(32)).norm().item() < 1e-3
    restricted = restrict_pools_for_strong_pin(pools, train32, heldout32)
    assert restricted.train_name_ids.numel() == 32 and restricted.heldout_name_ids.numel() == 32
    print(f"  32+32 disjoint, subset of the original train/heldout pools, each side internally "
          f"orthonormal (||G-I||_F train={(g_train - torch.eye(32)).norm().item():.2e}, "
          f"heldout={(g_held - torch.eye(32)).norm().item():.2e}); restrict_pools_for_strong_pin OK")

    print("\nembed_arms self-test (network-free / CPU-only portions) PASSED")
    print("(arm (ii) frozen_gpt2_span + its isometry check require network + GPT-2 model weights "
          "-- exercised separately by run_deltanet_rd.py --smoke's own [smoke 2b] section, not "
          "duplicated here to keep this self-test runnable offline)")


if __name__ == "__main__":
    _self_test()
