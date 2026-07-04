"""geo3_simulator.py -- the F2/F1 simulator behind sec 14.6's pre-registered
F-geo-3 launch read (DELTANET_RD_EXACTNESS_DESIGN.md sec 14, Rev B).

PROVENANCE: refactored minimally from the sec-14 attack round's own GPU
probe script (geo3_ns_check.py -- preserved on the box at
/home/nvidia/geo3_ns_check.py and in the 2026-07-03 session scratchpad).
That script produced the F2 bar-tolerance measurements (bars hold at
key-Gram residual ~1.0, crater by ~2.5) and the F1 cross-episode drift
measurements (pairwise cos 0.94-0.95 / 0.87-0.89 / 0.80-0.82 at
K=16/32/48). Round-2 verification CHECK 1(b) required this to live in the
repo as importable, documented code rather than as a session artifact.
The original probe sections (A/A2/B/C) are preserved under __main__ for
regression/provenance; the importable surface is the functions below.

THE REGISTERED drift->simulator MAPPING (sec 14.6 Rev B -- written down
BEFORE Wave -1 runs, per the round-2 verifier's requirement):

  Cross-episode drift enters as the value->key cross-alignment factor
  (sec 14.5's channel). Each entity's value-side representation is a
  FIXED unit vector tilted to cosine c from that entity's own episode
  key (deviation direction i.i.d. random orthogonal), where c is the
  MEASURED mean pairwise cross-episode drift cosine from the F1
  diagnostic (a second, worst-case run uses the p10 pairwise cosine).
  Rationale: v_eff_X is per-identity-only and cannot condition on which
  K-1 entities co-occur, so its best achievable alignment to the
  episode-specific orthogonalized key is limited by that key's own
  cross-episode consistency; the mean pairwise cosine is a conservative
  stand-in for cos-to-population-mean (which is >= mean pairwise cos for
  concentrated direction distributions).

  Compounding: hop 1 scores directly against the value representation
  (no c factor -- harness eval targets ARE v_eff); every subsequent hop
  re-enters the retrieved value as a query, paying ~c per re-entry ->
  ~c^(h-1) on top of the key-Gram-residual term, PLUS interference from
  the orthogonal deviation component re-reading S, which the simulation
  captures and no closed form does. (The verification round's "c^h"
  shorthand is the conservative analytic bound; this registered
  simulation refines it. Registered as the better mapping per the
  round-2 instruction's own allowance.)

  Launch read (sec 14.6): keys at the post-orthogonalization guaranteed
  Gram residual (resid_tol = 1e-2), values tilted to the measured mean
  drift c; predicted K=16 h=4 rec@0.9 must be >= 0.8 for Wave 1 to
  launch. The p10 run is reported alongside, non-gating. K=32's
  prediction is recorded but not gating (sec 14.6's rationale).

  F1-diagnostic statistic + sampling spec (pinned): >= 8 entities per K,
  drawn randomly WITHOUT replacement from the train name pool; >= 32
  episode-context resamples per entity; aggregation = pairwise cosines
  pooled WITHIN entity across resamples, then pooled across entities;
  report mean and p10 of that pooled distribution.

No fla/chunk_delta_rule dependency -- pure torch, runs anywhere.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core pieces (verbatim math from the attack probe / sec 14.1-14.2)
# ---------------------------------------------------------------------------

def newton_schulz(A: torch.Tensor, n_iter: int) -> tuple[torch.Tensor, list[float]]:
    """The exact sec-14.1 iteration: X_{t+1} = 1.5 X_t - 0.5 (X_t X_t^T) X_t,
    pre-scaled 1/sqrt(K). A: (B,K,d), rows unit-norm. Returns (Q, resid_hist)."""
    B, K, d = A.shape
    X = A / (K ** 0.5)
    resid_hist = []
    I_K = torch.eye(K, device=A.device, dtype=A.dtype)
    for _ in range(n_iter):
        G = X @ X.transpose(-1, -2)
        X = 1.5 * X - 0.5 * (G @ X)
        with torch.no_grad():
            r = (X @ X.transpose(-1, -2) - I_K).norm(dim=(-2, -1))
            resid_hist.append(r.mean().item())
    return X, resid_hist


def delta_rule_exact(keys: torch.Tensor, vals: torch.Tensor,
                      beta: float = 1.0) -> torch.Tensor:
    """Exact delta-rule recursion, S@k-retrieves DESIGN convention
    (model_rd.py::kernel_state_design_layout). keys/vals: (B,K,d)."""
    B, K, d = keys.shape
    S = torch.zeros(B, d, d, device=keys.device, dtype=keys.dtype)
    I = torch.eye(d, device=keys.device, dtype=keys.dtype).unsqueeze(0)
    for j in range(K):
        k = keys[:, j].unsqueeze(-1)
        v = vals[:, j].unsqueeze(-1)
        S = S @ (I - beta * (k @ k.transpose(-1, -2))) + beta * (v @ k.transpose(-1, -2))
    return S


def perturbed_orthonormal(B: int, K: int, d: int, target_resid: float,
                           gen: torch.Generator, device) -> torch.Tensor:
    """QR-orthonormal rows perturbed by bisection to a target mean
    Gram-deviation ||KK^T - I||_F (the attack probe's construction)."""
    Q0, _ = torch.linalg.qr(torch.randn(B, d, K, generator=gen, device=device))
    Q0 = Q0.transpose(-1, -2).contiguous()
    if target_resid == 0.0:
        return Q0
    lo, hi, cand = 0.0, 2.0, Q0
    I_K = torch.eye(K, device=device).expand(B, K, K)
    for _ in range(30):
        mid = (lo + hi) / 2
        cand = F.normalize(Q0 + mid * torch.randn(B, K, d, generator=gen, device=device), dim=-1)
        resid = ((cand @ cand.transpose(-1, -2)) - I_K).norm(dim=(-2, -1)).mean().item()
        if resid < target_resid:
            lo = mid
        else:
            hi = mid
    return cand


def tilt_to_cos(keys: torch.Tensor, c: float, gen: torch.Generator) -> torch.Tensor:
    """Unit vectors at cosine EXACTLY c from each key row (the registered
    mapping's value-representation construction). Deviation direction is
    i.i.d. random, projected orthogonal to the key. keys: (B,K,d)."""
    if c >= 1.0:
        return keys.clone()
    n = torch.randn(keys.shape, generator=gen, device=keys.device, dtype=keys.dtype)
    n = n - (n * keys).sum(-1, keepdim=True) * keys        # project out key component
    n = F.normalize(n, dim=-1)
    return c * keys + (1.0 - c * c) ** 0.5 * n


def simulate_recovery(K: int, gram_resid: float, align_cos: float,
                       B: int = 512, d: int = 64, hops=(1, 2, 3, 4),
                       seed: int = 0, device: str | None = None) -> dict:
    """The registered launch-read simulation for one (K, drift) cell.

    Keys at Gram residual `gram_resid`; each entity's value representation
    tilted to `align_cos` from its own key (the drift mapping); vals in
    clause order follow the harness grammar (clause j binds key j ->
    value-rep of entity perm[j], a single K-cycle); scoring is
    harness-faithful: pred vs. the TARGET ENTITY'S VALUE representation
    (eval targets are v_eff), rec@0.9 absolute-cosine bar. Returns
    {"rec": {h: frac}, "mean_cos": {h: cos}, "actual_gram_resid": float}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    keys = perturbed_orthonormal(B, K, d, gram_resid, gen, device)
    val_rep = tilt_to_cos(keys, align_cos, gen)            # (B,K,d): entity x's value-rep
    # single K-cycle permutation per row (grammar_rd's own convention)
    perm = torch.stack([torch.randperm(K, generator=gen, device=device) for _ in range(B)])
    vals = torch.gather(val_rep, 1, perm.unsqueeze(-1).expand(-1, -1, d))  # clause j -> val_rep[perm[j]]
    with torch.no_grad():
        I_K = torch.eye(K, device=device).expand(B, K, K)
        actual = ((keys @ keys.transpose(-1, -2)) - I_K).norm(dim=(-2, -1)).mean().item()
        S = delta_rule_exact(keys, vals, beta=1.0)
        a0 = torch.randint(0, K, (B,), generator=gen, device=device)
        cur = torch.gather(keys, 1, a0.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        tgt_idx = a0
        rec, mean_cos = {}, {}
        for h in range(1, max(hops) + 1):
            cur = torch.einsum("bij,bj->bi", S, cur)
            tgt_idx = torch.gather(perm, 1, tgt_idx.unsqueeze(-1)).squeeze(-1)
            tgt = torch.gather(val_rep, 1, tgt_idx.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
            cos = F.cosine_similarity(cur, tgt, dim=-1)
            if h in hops:
                rec[h] = (cos > 0.9).float().mean().item()
                mean_cos[h] = cos.mean().item()
    return {"rec": rec, "mean_cos": mean_cos, "actual_gram_resid": actual}


def pairwise_drift_stats(out_rows: torch.Tensor) -> tuple[float, float]:
    """(mean, p10) of all pairwise cosines among one entity's orthogonalized
    keys across context resamples. out_rows: (n_draws, d). The harness-side
    F1 diagnostic pools these within entity, then across entities."""
    n = out_rows.shape[0]
    pw = F.cosine_similarity(out_rows.unsqueeze(0), out_rows.unsqueeze(1), dim=-1)
    off = pw[~torch.eye(n, dtype=torch.bool, device=out_rows.device)]
    return off.mean().item(), torch.quantile(off, 0.10).item()


def launch_read(drift_by_k: dict, gram_resid: float = 1e-2, seed: int = 0,
                 device: str | None = None) -> dict:
    """The pre-registered sec-14.6 launch read. GATE: predicted K=16 h=4
    rec@0.9 >= 0.8 under the MEAN drift mapping. The p10 (worst-case) run
    and both K=32 predictions are reported alongside, non-gating.

    KEY_ANCHORING_DESIGN.md sec 3.4 caveat box / sec 4 (the shared-c bug,
    corrected in DELTANET_RD_EXACTNESS_DESIGN.md sec 16.7 from the
    coordinator's GPU re-measurement, archive:
    experiment-runs/2026-07-04_geo3_simulator_recheck/): the PRE-FIX
    signature took a single scalar `(drift_mean, drift_p10)` applied to
    EVERY K in the loop below -- so a caller that measured DIFFERENT drift
    at K=16 vs K=32 (the realistic case) silently had K=32's own
    measurement discarded in favor of K=16's. THIS signature takes a
    per-K dict instead (`drift_by_k = {16: {"mean":.., "p10":..},
    32: {"mean":.., "p10":..}, ...}`) so each K is ALWAYS simulated at its
    OWN measured drift -- the bug is closed by construction, not by
    caller discipline. See `_self_test_per_k_c_differs` below for the
    registered regression test (K=16 and K=32 calls must receive
    different `c` whenever the measured per-K drifts differ)."""
    out = {"gate_cell": "K=16 h=4", "gate_bar": 0.8, "inputs": {"drift_by_k": drift_by_k, "gram_resid": gram_resid}}
    for label in ("mean", "p10"):
        out[label] = {K: simulate_recovery(K, gram_resid, drift_by_k[K][label], seed=seed, device=device)
                      for K in drift_by_k}
    out["predicted_gate_value"] = out["mean"][16]["rec"][4]
    out["launch"] = bool(out["predicted_gate_value"] >= 0.8)
    return out


# ---------------------------------------------------------------------------
# Synthetic input generators (Wave -1 smoke reuse; from the attack probe)
# ---------------------------------------------------------------------------

def make_near_collinear(B, K, d, n_clusters, spread, gen, device):
    """Early-training regime: K unit keys clustered around few directions."""
    centers = F.normalize(torch.randn(B, n_clusters, d, generator=gen, device=device), dim=-1)
    assign = torch.randint(0, n_clusters, (B, K), generator=gen, device=device)
    base = torch.gather(centers, 1, assign.unsqueeze(-1).expand(-1, -1, d))
    return F.normalize(base + torch.randn(B, K, d, generator=gen, device=device) * spread, dim=-1)


def make_adversarial_duplicates(B, K, d, gen, device, n_dup_pairs=3, eps=1e-5):
    """Near-duplicate row pairs -- the fallback-trigger case."""
    A = F.normalize(torch.randn(B, K, d, generator=gen, device=device), dim=-1)
    for p in range(min(n_dup_pairs, K // 2)):
        i, j = 2 * p, 2 * p + 1
        A[:, j] = F.normalize(A[:, i] + eps * torch.randn(B, d, generator=gen, device=device), dim=-1)
    return A


# ---------------------------------------------------------------------------
# KEY_ANCHORING_DESIGN.md sec 4 / sec 3.4 caveat box's registered regression
# test: the per-K drift-threading unit test. Asserts K=16 and K=32 calls
# receive DIFFERENT `c` values from a per_k dict when the measured drifts
# differ -- the exact bug geo3_drift_diagnostic.py::main() + the pre-fix
# launch_read had (K=16's drift silently applied to both K).
# ---------------------------------------------------------------------------

def _self_test_per_k_c_differs() -> None:
    """CPU-only, no GPU needed (this module has no fla/chunk_delta_rule
    dependency). Monkeypatches simulate_recovery to RECORD the `c`
    (align_cos) it was called with per K, then asserts launch_read threads
    DIFFERENT per-K drift into DIFFERENT per-K calls."""
    import unittest.mock as mock

    calls = []
    real_simulate_recovery = simulate_recovery

    def _spy(K, gram_resid, align_cos, **kwargs):
        calls.append((K, align_cos))
        return real_simulate_recovery(K, gram_resid, align_cos, **kwargs)

    drift_by_k = {16: {"mean": 0.9416, "p10": 0.9243}, 32: {"mean": 0.9037, "p10": 0.8576}}
    with mock.patch(__name__ + ".simulate_recovery", side_effect=_spy):
        out = launch_read(drift_by_k, seed=0, device="cpu")

    c_by_k_mean = {K: c for K, c in calls if K in (16, 32)}
    mean_calls = [(K, c) for K, c in calls]
    # every (K, "mean") call must have used K's OWN drift_by_k[K]["mean"], never the other K's
    for K in (16, 32):
        assert any(k == K and abs(c - drift_by_k[K]["mean"]) < 1e-9 for k, c in mean_calls), \
            f"K={K}'s 'mean' simulate_recovery call never used its OWN drift_by_k[{K}]['mean']"
    k16_cs = {c for k, c in mean_calls if k == 16}
    k32_cs = {c for k, c in mean_calls if k == 32}
    assert k16_cs != k32_cs, (
        f"REGRESSION: K=16 and K=32 calls used the SAME c values ({k16_cs} vs {k32_cs}) despite "
        f"differing measured per-K drift ({drift_by_k}) -- this is exactly the shared-c bug "
        f"(sec 3.4 caveat box) reappearing.")
    assert out["predicted_gate_value"] == out["mean"][16]["rec"][4]
    print(f"  [geo3_simulator per-K threading] K=16 mean-mapping c values used: {sorted(k16_cs)}")
    print(f"  [geo3_simulator per-K threading] K=32 mean-mapping c values used: {sorted(k32_cs)}")
    print("  per-K drift threading OK -- K=16/K=32 calls received DIFFERENT c "
          "for differing measured drift (the shared-c bug does NOT reproduce)")


# ---------------------------------------------------------------------------
# Provenance / regression main: reproduces the attack probe's own checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    gen = torch.Generator(device=DEV).manual_seed(1)

    print("(B) bar-tolerance sweep (attack F2's measurement, idealized alignment c=1):")
    for K in (16, 32):
        for tr in (0.0, 1e-2, 1e-1, 3e-1, 1.0, 2.55):
            r = simulate_recovery(K, tr, 1.0, device=DEV)
            print(f"  K={K} resid~{tr:<5} actual={r['actual_gram_resid']:.3f} "
                  f"rec@0.9 h1-4={[round(r['rec'][h], 3) for h in (1, 2, 3, 4)]}")

    print("\n(C->launch_read) example: attack-measured drift, K=16/K=32 DIFFERENT (0.9416/0.9037):")
    lr = launch_read(drift_by_k={16: {"mean": 0.9416, "p10": 0.9243}, 32: {"mean": 0.9037, "p10": 0.8576}},
                      device=DEV)
    print(f"  gate {lr['gate_cell']} >= {lr['gate_bar']}: predicted={lr['predicted_gate_value']:.3f} "
          f"launch={lr['launch']}")
    for label in ("mean", "p10"):
        for K in (16, 32):
            print(f"  [{label}] K={K}: rec h1-4="
                  f"{[round(lr[label][K]['rec'][h], 3) for h in (1, 2, 3, 4)]}")

    print("\n(D) per-K drift-threading regression test (sec 4 / sec 3.4 caveat box):")
    _self_test_per_k_c_differs()

    print("\n(A) Newton-Schulz convergence on near-collinear + adversarial inputs:")
    for K in (16, 32, 48):
        A_real = make_near_collinear(64, K, 64, max(2, K // 6), 0.15, gen, DEV).requires_grad_(True)
        Q, hist = newton_schulz(A_real, 12)
        Q.sum().backward()
        A_adv = make_adversarial_duplicates(64, K, 64, gen, DEV)
        _, hist_adv = newton_schulz(A_adv, 12)
        print(f"  K={K}: resid@12={hist[-1]:.6f} grad_finite={torch.isfinite(A_real.grad).all().item()} "
              f"| adversarial resid@12={hist_adv[-1]:.6f} (fallback: {hist_adv[-1] > 1e-2})")
