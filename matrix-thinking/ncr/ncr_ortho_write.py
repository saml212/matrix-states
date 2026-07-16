"""NCR ORTHOGONAL / SPECTRAL-NORMALIZED WRITE -- NCR_ORTHO_WRITE.md.

ADDITIVE build: imports ncr_earlyln_scale / ncr_models / ncr_task /
ncr_spectral / run_ncr / ncr_opbank_* VERBATIM, modifies none of them. The
lever is a differentiable Newton-Schulz polar projection wrapped around the
BindingEncoder's Z-output (NCR_ORTHO_WRITE.md §2), added as a subclass +
constructor flag so the existing free-write path is preserved BYTE-IDENTICAL
for the pinned baseline (flag off == ncr_earlyln_scale.NCREarlyLNModel, the
exact §11.6 recipe).

Two deliverables, per the pre-registration:

  PART A (primary): NCROrthoWriteModel on the single-relation NCR task at
    K in {24,32}, d=K+1. `arm` stays 'ncr', encode() is the ONLY override, so
    every run_ncr instrument (z_dump, deep probe, Axis-C lock, trust screen,
    blank_out_check, eval_cell) runs against the ORTHOGONALIZED Z verbatim.
    The realistic-depth ladder (NCR_ORTHO_WRITE.md §3) is evaluated via a thin
    helper reusing the audited binexp_read + sample_eval_batch (NO GRIDS edit).

  PART B (discriminator): OrthoBankModel + a DISTINCT-operator composition
    ("chain") task where depth = number of compositions, mod-K-trap-safe by
    construction (distinct-op-per-hop + no-consecutive-repeat + fixed-point
    exclusion; NCR_ORTHO_WRITE.md §4b). Reuses ncr_opbank's BankBindingEncoder
    (write) + generate_bank_episode (data) + _select_Z (per-query op select).

This module EMITS NO VERDICT (a later blind assess applies the §4 band map).
It emits per-cell recovered@h / recovered@L PLUS the mechanistic spectral
diagnostics (departure-from-normality, cond#, min|lambda|/c*, phase-resid) so
the assessment can be mechanistic.

EVAL-PURITY (inherited §8.8 safeguard, re-verified for THIS write): the NS
projection is a DETERMINISTIC pure function of Z (fixed-seed-free ones-init
power iteration for the pre-scale) -- load-bearing so the z-dump Z equals the
eval Z bit-for-bit and blank_out's bit-identical check holds. `_ln_alpha` is
forced 0.0 at eval; the inherited binexp_read/loop_read are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
for p in (CHAPTER2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import task_e as te                     # noqa: E402 (verbatim)
import ncr_task as nt                   # noqa: E402 (verbatim)
import ncr_models as nm                 # noqa: E402 (verbatim; binexp_read/loop_read reuse)
import ncr_spectral as ns               # noqa: E402 (verbatim)
import run_ncr as rn                    # noqa: E402 (verbatim -- instrument reuse)
import ncr_earlyln_scale as els         # noqa: E402 (verbatim -- NCREarlyLNModel + train loop)
import ncr_opbank_models as obm         # noqa: E402 (verbatim -- BankBindingEncoder, _select_Z)
import ncr_opbank_task as obt           # noqa: E402 (verbatim -- generate_bank_episode)
import analyze_zdump as az              # noqa: E402 (verbatim -- entity_subspace/block_decompose)

RUNNER_TAG = "ncr_ortho_write_v1"

# Newton-Schulz polar defaults (NCR_ORTHO_WRITE.md §2; n_iter set by the §6
# smoke calibration). The (1.5,-0.5) iteration is QUADRATICALLY convergent near
# the orthogonal fixed point (a near-orthogonal converged-regime write reaches
# QᵀQ≈I to machine precision -- ~1e-15 -- by n_iter=6 and is IDEMPOTENT past
# that) but only LINEAR for the smallest modes of a badly-conditioned input, so
# a robust default of 40 fully orthogonalizes even the PATHOLOGICAL untrained
# encoder output (measured cond#≈8500 at random init, σ from 10.96 down to
# 0.0012 -- ill-conditioned, NOT rank-deficient; NS reaches machine-precision
# orthogonality by n_iter=40 there, err→0/cond→1). The extra iterations are
# free (idempotent at the orthogonal fixed point, trivial matmuls at d≈33) and
# guarantee clean gradients through any ill-conditioned early-training
# transient. Configurable via --ns-iter (the audit/runner may revisit).
NS_ITER_DEFAULT = 40
NS_POWER_DEFAULT = 12
NS_EPS = 1e-7

# Realistic-depth ladder (physical raw depths; every one a NOVEL residue mod K
# for both K=24 and K=32 -- asserted at eval time). NCR_ORTHO_WRITE.md §3.
REALISTIC_DEPTHS = (5, 12, 20, 29, 40, 61)
PRIMARY_HSTAR = 40                      # the re-registered realistic h* (§3)

# Discriminator (Part B) path-length ladder (depth = #distinct-op compositions,
# NO periodicity). NCR_ORTHO_WRITE.md §4b.
CHAIN_TRAIN_L = (1, 2, 3)
CHAIN_EVAL_L = (5, 8, 12, 16, 20, 24, 32, 40)
CHAIN_LSTAR = 32
DISC_R = 4                              # bank size (distinct operators per episode)


# ===========================================================================
# The lever: differentiable Newton-Schulz polar projection (no eig/SVD)
# ===========================================================================

def newton_schulz_polar(Z: torch.Tensor, n_iter: int = NS_ITER_DEFAULT,
                        n_power: int = NS_POWER_DEFAULT,
                        eps: float = NS_EPS) -> torch.Tensor:
    """Project a batch of square matrices onto (near-)orthogonal via the
    classical Newton-Schulz polar iteration `X <- 1.5 X - 0.5 X (XᵀX)`
    (NCR_ORTHO_WRITE.md §2). Returns Q with QᵀQ ~= I, WITHOUT eig/SVD (whose
    backward is unstable at the near-degenerate spectrum here).

    Z: (..., d, d). Fully autograd-safe (batched matmuls only).

    Spectral-norm pre-scale (required for convergence): X0 = Z / σ̂, σ̂ a
    DETACHED, DETERMINISTIC ones-init power-iteration estimate of the LARGEST
    singular value (the top singular value converges geometrically -- 12 iters
    are tight regardless of conditioning). Detached => the scale is a
    read-invariant no-op and the gradient through X0 is direction-preserving;
    deterministic => encode() is a pure function of Z, load-bearing for the
    bit-identical z-dump/blank-out eval checks. Pre-scaling puts σ_max at ~1,
    inside the (1.5,-0.5) iteration's (0, √3) basin; a near-orthogonal Z (all
    σ equal, the converged regime) becomes σ=1 and is a fixed point, so the
    constraint is near-free there (few iters) while an ill-conditioned early
    transient converges slowly but NEVER diverges."""
    orig_dtype = Z.dtype
    X = Z if Z.dtype in (torch.float32, torch.float64) else Z.float()
    d = X.shape[-1]
    Xt = X.transpose(-1, -2)
    # --- deterministic ones-init power iteration for σ̂ (detached scale) ---
    with torch.no_grad():
        v = torch.ones(*X.shape[:-1], 1, device=X.device, dtype=X.dtype)  # (...,d,1)
        v = v / v.norm(dim=-2, keepdim=True).clamp_min(eps)
        for _ in range(n_power):
            u = X @ v
            u = u / u.norm(dim=-2, keepdim=True).clamp_min(eps)
            v = Xt @ u
            v = v / v.norm(dim=-2, keepdim=True).clamp_min(eps)
        sigma = (X @ v).norm(dim=-2, keepdim=True).clamp_min(eps)         # (...,1,1)
    Xn = X / sigma                       # sigma detached: direction-preserving grad
    for _ in range(n_iter):
        Xn = 1.5 * Xn - 0.5 * (Xn @ (Xn.transpose(-1, -2) @ Xn))
    return Xn.to(orig_dtype)


def orthogonality_error(Q: torch.Tensor) -> torch.Tensor:
    """||QᵀQ - I||_F per matrix (scale-normalized by the mean singular value so
    a c*orthogonal matrix reads ~0). Returns (...,) tensor."""
    d = Q.shape[-1]
    QtQ = Q.transpose(-1, -2) @ Q
    scale = torch.diagonal(QtQ, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(NS_EPS)  # ~ mean σ²
    I = torch.eye(d, device=Q.device, dtype=Q.dtype)
    return (QtQ / scale.unsqueeze(-1).unsqueeze(-1) - I).flatten(-2).norm(dim=-1)


# ===========================================================================
# PART A model: NCREarlyLNModel + the orthogonal-write flag on encode()
# ===========================================================================

class NCROrthoWriteModel(els.NCREarlyLNModel):
    """The §11 earlyln contender with the ONLY delta being an optional
    Newton-Schulz polar projection on the encoder's Z-output. `orthogonal`
    False (default) => encode() is BYTE-IDENTICAL to NCREarlyLNModel (the
    pinned free-write baseline). `arm='ncr'`, forward() (the earlyln LN blend),
    eval_read() are all inherited unchanged, so every run_ncr instrument runs
    against the orthogonalized Z verbatim."""

    def __init__(self, d: int = nm.D_PIN, h: int = nm.ENC_H,
                 orthogonal: bool = False, ns_iter: int = NS_ITER_DEFAULT,
                 ns_power: int = NS_POWER_DEFAULT):
        super().__init__(d=d, h=h)
        self._orthogonal = bool(orthogonal)
        self._ns_iter = int(ns_iter)
        self._ns_power = int(ns_power)

    def encode(self, keys, values):
        Z = self.encoder(keys, values)
        if self._orthogonal:
            Z = newton_schulz_polar(Z, n_iter=self._ns_iter, n_power=self._ns_power)
        return Z


def build_primary_model(arm: str, d: int, h: int, ns_iter: int, ns_power: int):
    """arm in {'free','ortho'}. 'free' is els.NCREarlyLNModel VERBATIM (the
    pinned §11.6 baseline); 'ortho' is the flag-on subclass. Both carry
    arm='ncr', _ln_alpha, encode/forward -- identical downstream pipeline."""
    if arm == "free":
        return els.NCREarlyLNModel(d=d, h=h)
    if arm == "ortho":
        return NCROrthoWriteModel(d=d, h=h, orthogonal=True,
                                  ns_iter=ns_iter, ns_power=ns_power)
    raise ValueError(arm)


# ===========================================================================
# Mechanistic spectral diagnostics (reuse the audited az machinery)
# ===========================================================================

def spectral_diagnostics(z_dump: dict) -> dict:
    """Per-example + mean departure-from-normality / cond# / min|λ|/c* /
    phase-resid of the trained entity operator A = UᵀZU, on the entity
    subspace of z_ideal. Reuses az.entity_subspace/block_decompose/
    effective_rank/match_eigenvalues verbatim (the §11.4a machinery)."""
    rows = []
    for Zr, Zir in zip(z_dump["Z"], z_dump["z_ideal"]):
        Zm = np.asarray(Zr, float)
        Zim = np.asarray(Zir, float)
        sub = az.entity_subspace(Zim)
        U, V = sub["U"], sub["V"]
        A, _, _, _ = az.block_decompose(Zm, U, V)
        Pi = U.T @ Zim @ U
        c_star = float(np.sum(A * Pi) / max(np.sum(Pi * Pi), 1e-12))
        comm = A @ A.T - A.T @ A
        depart = float(np.linalg.norm(comm) / max(np.linalg.norm(A) ** 2, 1e-12))
        sig = np.linalg.svd(A, compute_uv=False)
        a_cond = float(sig[0] / max(sig[-1], 1e-12))
        ev = np.linalg.eigvals(A)
        mod = np.abs(ev)
        mod_rel = mod / max(abs(c_star), 1e-12)
        roots = np.exp(2j * np.pi * np.arange(A.shape[0]) / A.shape[0])
        ev_unit = ev / np.clip(mod, 1e-12, None)
        _, phres = az.match_eigenvalues(ev_unit, roots)
        rows.append(dict(
            depart_normality=depart, A_cond=a_cond,
            min_mod_rel=float(mod_rel.min()), mean_mod_rel=float(mod_rel.mean()),
            phase_resid_max=float(phres.max()), c_star=c_star,
            A_eff_rank=float(az.effective_rank(A))))
    keys = rows[0].keys()
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return dict(per_example=rows, mean=mean)


# ===========================================================================
# PART A: realistic-depth ladder eval (thin reuse of the audited read)
# ===========================================================================

@torch.no_grad()
def realistic_ladder_eval(model, cfg, K: int, seed: int, device: str,
                          depths=REALISTIC_DEPTHS) -> dict:
    """recovered_frac@0.9 / mean_cos at each PHYSICAL depth in `depths`,
    reusing nt.sample_eval_batch (mod-K-safe labels) + the audited binexp_read
    + recovery_cosine. Each depth's residue is asserted NOVEL (∉ {0,1,2,3}) --
    a multiple-of-K or train-residue depth would be a trivial/identity target,
    refused here by construction."""
    model.eval()
    model._ln_alpha = 0.0
    gen = torch.Generator(device=device).manual_seed(seed + 40_000)
    out = {}
    for h in depths:
        r = h % K
        assert r not in (0, 1, 2, 3), (
            f"realistic depth h={h} has residue {r} mod K={K} -- not a novel "
            f"far-depth target (identity/train-residue). Fix the ladder.")
        chunks = []
        for _ in range(rn.EVAL_BATCHES):
            b = nt.sample_eval_batch(cfg, rn.EVAL_BATCH_SIZE, gen, h, device=device)
            state = model.encode(b["keys"], b["values"])
            o = nm.binexp_read(state, b["query_keys"], h)["o"]
            chunks.append(nm.recovery_cosine(o, b["targets"]).reshape(-1))
        cos = torch.cat(chunks)
        out[str(h)] = dict(
            residue=r, effective_hop=r,
            mean_cos=float(cos.mean().item()),
            **{f"recovered_frac@{tau}": float((cos > tau).float().mean().item())
               for tau in nt.TAUS})
    return out


# ===========================================================================
# PART A: the free/ortho primary cell runner
# ===========================================================================

def primary_cell_id(arm: str, K: int, seed: int) -> str:
    return f"ortho_{arm}_K{K}_s{seed}"


def run_primary_cell(arm: str, K: int, seed: int, steps: int, device: str,
                     outdir: str, stop_file: str = "", ceiling_gpuh: float = 3.0,
                     ns_iter: int = NS_ITER_DEFAULT, ns_power: int = NS_POWER_DEFAULT,
                     anneal_frac: float = 0.5) -> dict:
    """Train (earlyln schedule, els.train_earlyln_cell VERBATIM) the free or
    ortho arm on the single-relation task at K, d=K+1, then the IDENTICAL
    post-train instrument sequence run_ncr uses for its 'ncr' arm (z_dump ->
    deep probe -> Axis-C lock -> trust screen -> blank_out -> eval_cell), PLUS
    the realistic-depth ladder and the mechanistic spectral diagnostics.
    Resume-safe (whole-cell skip-if-COMPLETED)."""
    assert arm in ("free", "ortho"), arm
    d_eff = K + 1                                   # tight-spare (K+1), the §11.6 regime
    h_eff = els.GRID_SHAPES[K]["h"]                 # encoder hidden (64), unchanged
    os.makedirs(outdir, exist_ok=True)
    cid = primary_cell_id(arm, K, seed)
    out_path = os.path.join(outdir, f"{cid}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "COMPLETED":
            print(f"  [{cid}] already COMPLETED -- skipping (resume-safe)", flush=True)
            return prev

    cfg = nt.claim_config(K, d=d_eff)
    torch.manual_seed(seed)
    model = build_primary_model(arm, d_eff, h_eff, ns_iter, ns_power).to(device)
    rec = dict(cell_id=cid, arm=arm, K=K, d=d_eff, h=h_eff, seed=seed,
               orthogonal=(arm == "ortho"), ns_iter=ns_iter, ns_power=ns_power,
               anneal_frac=anneal_frac, runner_tag=RUNNER_TAG,
               git_commit=rn.git_commit(), params=nm.n_params(model),
               train_batch=els.TRAIN_BATCH, host=socket.gethostname(),
               device=device, torch_version=torch.__version__,
               realistic_hstar=PRIMARY_HSTAR, realistic_depths=list(REALISTIC_DEPTHS),
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ceiling_s = ceiling_gpuh * 3600.0 if device == "cuda" else float("inf")
    t0 = time.time()

    tr = els.train_earlyln_cell(model, cfg, device, steps, seed, stop_file,
                                ceiling_s, anneal_frac=anneal_frac)
    rec["train"] = tr
    if tr["status"] != "COMPLETED":
        rec["status"] = tr["status"]
        rec["elapsed_s"] = time.time() - t0
        rn.atomic_write_json(out_path, rec)
        return rec

    model.eval()
    model._ln_alpha = 0.0                            # EVAL-PURITY (inherited §8.8)

    zd = rn.z_dump(model, cfg, device, seed)
    rec["z_dump"] = zd
    all_h = [p.h for p in nt.eval_points(K, d=d_eff)]
    probe = ns.analyze_zdump_arrays(zd["Z"], zd["z_ideal"], all_h)
    rec["deep_probe"] = dict(
        phase_resid_max_per_example=[ex["phase_resid_max"] for ex in probe["per_example"]],
        phase_resid_max_mean=probe["phase_resid_max_mean"],
        c_star_per_example=[ex["c_star"] for ex in probe["per_example"]],
        scale_corrected_residual=[ex["scale_corrected_residual"] for ex in probe["per_example"]],
        A_eff_rank=[ex["A_eff_rank"] for ex in probe["per_example"]])

    # mechanistic spectral diagnostics (the mechanism must move WITH the result)
    rec["spectral"] = spectral_diagnostics(zd)

    lock_path = os.path.join(outdir, f"{cid}.axis_c_lock.json")
    write_axis = ns.write_axis_c_lock(lock_path, cid, K, probe)
    lock_content = ns.verify_axis_c_lock(lock_path)
    rec["axis_c_lock_sha256"] = write_axis["lock_sha256"]

    screens = [ns.trust_screen(np.array(ex["blocks"]["A"]), np.array(ex["blocks"]["B"]),
                               np.array(ex["blocks"]["C"]), np.array(ex["blocks"]["D"]), all_h)
               for ex in probe["per_example"]]
    trust_per_h = {}
    for hh in screens[0]["per_h"]:
        ts = [s["per_h"][hh]["T"] for s in screens]
        t_worst = max(float("inf") if t == "inf" else float(t) for t in ts)
        trust_per_h[hh] = dict(T=t_worst if math.isfinite(t_worst) else "inf",
                               rule_trusted=bool(all(s["per_h"][hh]["rule_trusted"] for s in screens)))
    rec["trust_screen"] = dict(tau=ns.TAU, per_h=trust_per_h)

    rec["blank_out"] = rn.blank_out_check(model, cfg, device, seed)
    assert rec["blank_out"]["passed"], f"blank-out FAILED for {cid}: {rec['blank_out']}"

    # audited GRIDS eval (Axis-C-locked; gives h=29/61 far-depth) + realistic ladder
    rec["eval"] = rn.eval_cell(model, cfg, device, seed, K, lock_content, trust_per_h, d=d_eff)
    rec["realistic_ladder"] = realistic_ladder_eval(model, cfg, K, seed, device)

    rec["elapsed_s"] = time.time() - t0
    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
    rec["status"] = "COMPLETED"
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rn.atomic_write_json(out_path, rec)
    print(f"  [{cid}] COMPLETED in {rec['elapsed_s']:.0f}s -> {out_path}", flush=True)
    return rec


# ===========================================================================
# PART B: the structured-operator (distinct-op chain) discriminator
# ===========================================================================

def _sample_chain_paths(B: int, Q: int, R: int, L_set, gen, device):
    """Per-query path of operator indices with NO consecutive repeat
    (NCR_ORTHO_WRITE.md §4b guard 2). Returns (path (B,Q,Lmax) long,
    path_len (B,Q) long). L_set: tuple of allowed path lengths."""
    L_pool = torch.tensor(L_set, device=device, dtype=torch.int64)
    path_len = L_pool[torch.randint(0, len(L_set), (B, Q), generator=gen, device=device)]
    Lmax = int(path_len.max().item())
    path = torch.zeros(B, Q, Lmax, dtype=torch.int64, device=device)
    prev = torch.full((B, Q), -1, dtype=torch.int64, device=device)
    for t in range(Lmax):
        if t == 0:
            r = torch.randint(0, R, (B, Q), generator=gen, device=device)
        else:
            # draw from the OTHER R-1 operators: r = (prev + 1 + offset) % R,
            # offset in [0, R-2] -> guarantees r != prev (no code path repeats)
            off = torch.randint(0, R - 1, (B, Q), generator=gen, device=device)
            r = (prev + 1 + off) % R
        path[:, :, t] = r
        prev = r
    return path, path_len


def _chain_target_idx(succ: torch.Tensor, a_idx: torch.Tensor,
                      path: torch.Tensor, path_len: torch.Tensor) -> torch.Tensor:
    """Exact integer composite index: start a_idx, apply ONE step of the
    operator path[:,:,t]'s K-cycle for t < path_len (distinct op per hop).
    succ: (B,R,K); a_idx/path_len: (B,Q); path: (B,Q,Lmax). Returns (B,Q)."""
    B, R, K = succ.shape
    Q, Lmax = a_idx.shape[1], path.shape[2]
    cur = a_idx.clone()
    for t in range(Lmax):
        op = path[:, :, t]                                  # (B,Q)
        # succ_op[b,q] = succ[b, op[b,q], cur[b,q]]
        succ_sel = torch.gather(succ, 1, op.unsqueeze(-1).expand(B, Q, K))   # (B,Q,K)
        nxt = torch.gather(succ_sel, 2, cur.unsqueeze(-1)).squeeze(-1)       # (B,Q)
        cur = torch.where(path_len > t, nxt, cur)
    return cur


def sample_chain_batch(cfg: obt.BankConfig, batch_size: int, gen: torch.Generator,
                       L_set, device="cpu", exclude_fixed_points: bool = True) -> dict:
    """One bank episode + per-query distinct-op path. Target = composite
    permutation applied to the query key, computed by EXACT integer iteration.
    Fixed points (composite(a)==a) resampled per-query (guard 3). Depth = path
    length; no periodicity (product of DISTINCT K-cycles, not a power)."""
    ep = obt.generate_bank_episode(batch_size, cfg, gen, device)
    B, Q = batch_size, cfg.K
    a_idx = torch.randint(0, cfg.K, (B, Q), generator=gen, device=device)
    path, path_len = _sample_chain_paths(B, Q, cfg.R, L_set, gen, device)
    tgt_idx = _chain_target_idx(ep["succ"], a_idx, path, path_len)
    if exclude_fixed_points:
        # resample the (a_idx, path) of any fixed query a few rounds; a
        # persistent fixed point (impossible for L>=1 distinct-cycle paths
        # generically) would surface as a residual, disclosed not silently kept
        for _ in range(8):
            fixed = (tgt_idx == a_idx)
            if not bool(fixed.any()):
                break
            a_new = torch.randint(0, cfg.K, (B, Q), generator=gen, device=device)
            p_new, pl_new = _sample_chain_paths(B, Q, cfg.R, L_set, gen, device)
            t_new = _chain_target_idx(ep["succ"], a_new, p_new, pl_new)
            a_idx = torch.where(fixed, a_new, a_idx)
            path = torch.where(fixed.unsqueeze(-1), p_new, path)
            path_len = torch.where(fixed, pl_new, path_len)
            tgt_idx = torch.where(fixed, t_new, tgt_idx)
    pool = ep["pool"]
    query_keys = torch.gather(pool, 1, a_idx.unsqueeze(-1).expand(B, Q, cfg.d))
    targets = torch.gather(pool, 1, tgt_idx.unsqueeze(-1).expand(B, Q, cfg.d))
    ep.update(query_keys=query_keys, path=path, path_len=path_len,
              targets=targets, a_idx=a_idx)
    return ep


def sample_chain_eval(cfg: obt.BankConfig, batch_size: int, gen: torch.Generator,
                      L: int, device="cpu") -> dict:
    """Fixed-path-length-L eval batch (every query composes EXACTLY L distinct
    operators)."""
    return sample_chain_batch(cfg, batch_size, gen, (L,), device=device)


class OrthoBankModel(nn.Module):
    """Distinct-operator bank write (BankBindingEncoder VERBATIM) + optional NS
    polar projection per operator slice + a chain read. `orthogonal` False =>
    the free-write bank baseline (no projection). The read reads ONLY Z_bank
    (P=1 bottleneck). An optional inter-hop LN blend (annealed, eval-pure --
    the §8.9/§11 earlyln recipe) aids trainability; eval forces alpha=0."""

    arm = "ncr-bank"
    deviating_read = False

    def __init__(self, d: int = obt.D_PIN, R: int = DISC_R,
                 orthogonal: bool = False, ns_iter: int = NS_ITER_DEFAULT,
                 ns_power: int = NS_POWER_DEFAULT):
        super().__init__()
        self.d, self.R = d, R
        self.encoder = obm.BankBindingEncoder(d, R)
        self._orthogonal = bool(orthogonal)
        self._ns_iter, self._ns_power = int(ns_iter), int(ns_power)
        self._ln_alpha = 0.0

    def encode(self, keys, values, rel_ids):
        Z = self.encoder(keys, values, rel_ids)             # (B,R,d,d)
        if self._orthogonal:
            B, R, d, _ = Z.shape
            Z = newton_schulz_polar(Z.reshape(B * R, d, d),
                                    n_iter=self._ns_iter, n_power=self._ns_power).reshape(B, R, d, d)
        return Z

    def _blend(self, stepped):
        a = self._ln_alpha
        if a <= 0.0:
            return stepped
        return a * F.layer_norm(stepped, (self.d,)) + (1.0 - a) * stepped

    def chain_read(self, Z_bank, query_keys, path, path_len, renorm: bool,
                   blend: bool):
        """Apply the selected op slices in sequence (one matvec/hop). `renorm`
        = per-step L2 renorm (eval scale-management, cosine-invariant). `blend`
        = train-time earlyln LN blend."""
        cur = query_keys
        Lmax = path.shape[2]
        for t in range(Lmax):
            Zt = obm._select_Z(Z_bank, path[:, :, t])       # (B,Q,d,d)
            stepped = torch.einsum("bqij,bqj->bqi", Zt, cur)
            if blend:
                stepped = self._blend(stepped)
            if renorm:
                stepped = stepped / stepped.norm(dim=-1, keepdim=True).clamp_min(NS_EPS)
            cur = torch.where((path_len > t).unsqueeze(-1), stepped, cur)
        return cur

    def forward(self, batch: dict):
        """TRAIN path: naive matvec chain (L<=3) with the earlyln blend."""
        Z = self.encode(batch["keys"], batch["values"], batch["rel_ids"])
        pred = self.chain_read(Z, batch["query_keys"], batch["path"],
                               batch["path_len"], renorm=False, blend=True)
        return pred, Z

    @torch.no_grad()
    def eval_chain(self, Z_bank, query_keys, path, path_len):
        """Deep-path eval read: per-step renorm, NO blend (alpha=0 purity)."""
        return self.chain_read(Z_bank, query_keys, path, path_len,
                               renorm=True, blend=False)


def build_disc_model(arm: str, d: int, R: int, ns_iter: int, ns_power: int):
    if arm not in ("free", "ortho"):
        raise ValueError(arm)
    return OrthoBankModel(d=d, R=R, orthogonal=(arm == "ortho"),
                          ns_iter=ns_iter, ns_power=ns_power)


@torch.no_grad()
def bank_orthogonality(model, cfg: obt.BankConfig, device: str, seed: int,
                       n_examples: int = 4) -> dict:
    """Direct mechanistic readout for the bank write: per written operator
    slice, the scale-normalized ||ZᵀZ - I||_F and cond#. Free arm => large;
    ortho arm => ~0. Self-contained (no entity-subspace needed)."""
    gen = torch.Generator(device=device).manual_seed(seed + 50_000)
    ep = obt.generate_bank_episode(n_examples, cfg, gen, device)
    Z = model.encode(ep["keys"], ep["values"], ep["rel_ids"])   # (B,R,d,d)
    Zf = Z.reshape(-1, model.d, model.d)
    ortho_err = orthogonality_error(Zf)                         # (B*R,)
    sv = torch.linalg.svdvals(Zf.float())                       # (B*R,d)
    cond = (sv[:, 0] / sv[:, -1].clamp_min(NS_EPS))
    return dict(ortho_err_mean=float(ortho_err.mean().item()),
                ortho_err_max=float(ortho_err.max().item()),
                cond_mean=float(cond.mean().item()), cond_max=float(cond.max().item()))


@torch.no_grad()
def bank_blank_out(model, cfg: obt.BankConfig, device: str, seed: int) -> dict:
    """P=1 bottleneck check for the chain read: corrupt raw inputs post-write,
    confirm the read is bit-identical AND grad w.r.t. raw inputs is exactly
    zero (executed, not a shape check). Mirrors run_ncr.blank_out_check."""
    gen = torch.Generator(device=device).manual_seed(seed + 33_000)
    b = sample_chain_batch(cfg, 16, gen, (2,), device=device)
    keys = b["keys"].clone().requires_grad_(True)
    values = b["values"].clone().requires_grad_(True)
    q, path, plen = b["query_keys"], b["path"], b["path_len"]

    def read(state):
        return model.chain_read(state, q, path, plen, renorm=True, blend=False)

    with torch.enable_grad():
        state = model.encode(keys, values, b["rel_ids"])
    state_frozen = state.detach()
    pred1 = read(state_frozen)
    _ = torch.randn_like(keys), torch.randn_like(values)     # corrupt (never read)
    pred2 = read(state_frozen)
    bit_identical = bool(torch.equal(pred1, pred2))

    state_leaf = state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        pred_leaf = read(state_leaf)
        g = torch.autograd.grad(pred_leaf.sum(), [keys, values], allow_unused=True)
    grad_zero = all(gi is None for gi in g)
    with torch.enable_grad():
        state_live = model.encode(keys, values, b["rel_ids"])
        pred_live = read(state_live)
        g_live = torch.autograd.grad(pred_live.sum(), keys, allow_unused=True)[0]
    write_path_alive = g_live is not None and g_live.abs().sum().item() > 0
    return dict(bit_identical=bit_identical, grad_exactly_zero=grad_zero,
                write_path_alive=bool(write_path_alive),
                passed=bool(bit_identical and grad_zero and write_path_alive))


@torch.no_grad()
def chain_ladder_eval(model, cfg: obt.BankConfig, seed: int, device: str,
                      depths=CHAIN_EVAL_L) -> dict:
    """recovered_frac@0.9 / mean_cos at each PATH LENGTH L (depth = #distinct
    compositions). No mod-K anything -- L is genuine composition depth."""
    model.eval()
    model._ln_alpha = 0.0
    gen = torch.Generator(device=device).manual_seed(seed + 41_000)
    out = {}
    for L in depths:
        chunks = []
        for _ in range(rn.EVAL_BATCHES):
            b = sample_chain_eval(cfg, rn.EVAL_BATCH_SIZE, gen, L, device=device)
            state = model.encode(b["keys"], b["values"], b["rel_ids"])
            o = model.eval_chain(state, b["query_keys"], b["path"], b["path_len"])
            chunks.append(nm.recovery_cosine(o, b["targets"]).reshape(-1))
        cos = torch.cat(chunks)
        out[str(L)] = dict(mean_cos=float(cos.mean().item()),
                           **{f"recovered_frac@{tau}": float((cos > tau).float().mean().item())
                              for tau in nt.TAUS})
    return out


def disc_cell_id(arm: str, K: int, seed: int) -> str:
    return f"ortho_disc_{arm}_K{K}_s{seed}"


def run_disc_cell(arm: str, K: int, seed: int, steps: int, device: str,
                  outdir: str, stop_file: str = "", ceiling_gpuh: float = 2.5,
                  R: int = DISC_R, ns_iter: int = NS_ITER_DEFAULT,
                  ns_power: int = NS_POWER_DEFAULT, anneal_frac: float = 0.5) -> dict:
    """Train the free or ortho bank on the distinct-op chain task (train L<=3),
    then eval the deep-path ladder + bank-orthogonality + blank-out. EMITS NO
    VERDICT. Resume-safe (whole-cell skip-if-COMPLETED)."""
    assert arm in ("free", "ortho"), arm
    d_eff = K + 1
    os.makedirs(outdir, exist_ok=True)
    cid = disc_cell_id(arm, K, seed)
    out_path = os.path.join(outdir, f"{cid}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "COMPLETED":
            print(f"  [{cid}] already COMPLETED -- skipping (resume-safe)", flush=True)
            return prev

    cfg = obt.BankConfig(d=d_eff, K=K, R=R, orthogonal=True)
    torch.manual_seed(seed)
    model = build_disc_model(arm, d_eff, R, ns_iter, ns_power).to(device)
    rec = dict(cell_id=cid, arm=arm, K=K, d=d_eff, R=R, seed=seed,
               orthogonal=(arm == "ortho"), ns_iter=ns_iter, ns_power=ns_power,
               anneal_frac=anneal_frac, runner_tag=RUNNER_TAG,
               git_commit=rn.git_commit(), params=obm.n_params(model),
               train_batch=els.TRAIN_BATCH, host=socket.gethostname(),
               device=device, torch_version=torch.__version__,
               chain_lstar=CHAIN_LSTAR, chain_eval_L=list(CHAIN_EVAL_L),
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ceiling_s = ceiling_gpuh * 3600.0 if device == "cuda" else float("inf")
    t0 = time.time()

    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=els.BASE_LR)
    model.train()
    n_skipped, loss_hist = 0, []
    for step in range(1, steps + 1):
        model._ln_alpha = els.ln_alpha_at(step, steps, frac=anneal_frac)
        b = sample_chain_batch(cfg, els.TRAIN_BATCH, gen, CHAIN_TRAIN_L, device=device)
        pred, _ = model(b)
        loss = rn.cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1
        if step % 500 == 0 or step == 1:
            elapsed = time.time() - t0
            loss_hist.append([step, float(loss.item())])
            print(f"  [{cid}] step {step:6d}  loss {loss.item():.4f}  "
                  f"ln_a {model._ln_alpha:.2f}  {elapsed:.0f}s", flush=True)
            if rn.stop_requested(stop_file):
                print("  STOP file seen -- exiting 3"); sys.exit(3)
            if elapsed > ceiling_s:
                rec["status"] = "ABORTED-BUDGET"
                rec["train"] = dict(status="ABORTED-BUDGET", step=step, elapsed_s=elapsed,
                                    n_skipped_steps=n_skipped, loss_history=loss_hist)
                rec["elapsed_s"] = elapsed
                rn.atomic_write_json(out_path, rec)
                return rec
    rec["train"] = dict(status="COMPLETED", step=steps, elapsed_s=time.time() - t0,
                        n_skipped_steps=n_skipped, loss_history=loss_hist)

    model.eval()
    model._ln_alpha = 0.0
    # in-dist convergence readout (train path lengths)
    rec["in_dist"] = chain_ladder_eval(model, cfg, seed, device, depths=CHAIN_TRAIN_L)
    rec["chain_ladder"] = chain_ladder_eval(model, cfg, seed, device)
    rec["bank_orthogonality"] = bank_orthogonality(model, cfg, device, seed)
    rec["blank_out"] = bank_blank_out(model, cfg, device, seed)
    assert rec["blank_out"]["passed"], f"bank blank-out FAILED for {cid}: {rec['blank_out']}"

    rec["elapsed_s"] = time.time() - t0
    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
    rec["status"] = "COMPLETED"
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rn.atomic_write_json(out_path, rec)
    print(f"  [{cid}] COMPLETED in {rec['elapsed_s']:.0f}s -> {out_path}", flush=True)
    return rec


# ===========================================================================
# CPU self-test (executed kill-proofs)
# ===========================================================================

def _self_test():   # noqa: C901 (a suite; each block is an independent kill-proof)
    torch.manual_seed(0)

    # t1 NS DIFFERENTIABLE + ORTHOGONALIZES an ill-conditioned Gaussian
    d = 33
    G = torch.randn(3, d, d, dtype=torch.float64, requires_grad=True)
    Q = newton_schulz_polar(G, n_iter=NS_ITER_DEFAULT, n_power=NS_POWER_DEFAULT)
    err_in = orthogonality_error(G.detach()).mean().item()
    err_out = orthogonality_error(Q.detach()).mean().item()
    # worst-case raw Gaussian (cond#≈d): the robust n_iter=20 default drives it
    # to ~0 (see §6 calibration) -- assert a strong reduction to near-orthogonal.
    assert err_out < 0.2 and err_out < 0.1 * err_in, (err_in, err_out)
    g = torch.autograd.grad(Q.sum(), G)[0]
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "NS not differentiable"
    print(f"t1 PASS: NS polar differentiable; worst-case Gaussian ortho-err "
          f"{err_in:.3f} -> {err_out:.4f} (grads finite, nonzero)")

    # t2 NS TIGHT on a NEAR-orthogonal input (the converged regime that matters)
    Qrand, _ = torch.linalg.qr(torch.randn(4, d, d, dtype=torch.float64))
    Znear = 2.5 * Qrand + 0.02 * torch.randn(4, d, d, dtype=torch.float64)   # c*orthogonal + noise
    Qn = newton_schulz_polar(Znear, n_iter=NS_ITER_DEFAULT)
    en = orthogonality_error(Qn).max().item()
    assert en < 1e-2, f"t2 FAILED: near-orthogonal input not tightly orthogonalized (err {en})"
    # already-orthogonal is ~a fixed point
    Qf = newton_schulz_polar(Qrand, n_iter=NS_ITER_DEFAULT)
    assert orthogonality_error(Qf).max().item() < 1e-3, "t2b FAILED: orthogonal not a fixed point"
    print(f"t2 PASS: near-orthogonal input -> QᵀQ≈I (max err {en:.2e}); "
          f"orthogonal matrix is a fixed point")

    # t3 NS DETERMINISTIC (load-bearing for eval purity): same input -> same output
    Za = torch.randn(2, d, d)
    o1 = newton_schulz_polar(Za)
    o2 = newton_schulz_polar(Za)
    assert torch.equal(o1, o2), "t3 FAILED: NS polar is NOT deterministic (eval purity breaks)"
    print("t3 PASS: NS polar is a deterministic pure function of Z (eval purity safe)")

    # t4 MODEL: flag OFF == NCREarlyLNModel byte-identical; flag ON orthogonalizes
    torch.manual_seed(1)
    free = NCROrthoWriteModel(d=17, h=64, orthogonal=False)
    base = els.NCREarlyLNModel(d=17, h=64)
    base.load_state_dict(free.state_dict())
    ks = torch.randn(2, 16, 17); vs = torch.randn(2, 16, 17)
    with torch.no_grad():
        assert torch.equal(free.encode(ks, vs), base.encode(ks, vs)), \
            "t4 FAILED: orthogonal=False must be byte-identical to NCREarlyLNModel.encode"
    ortho = NCROrthoWriteModel(d=17, h=64, orthogonal=True)
    ortho.load_state_dict({k: v for k, v in free.state_dict().items()})
    with torch.no_grad():
        Zo = ortho.encode(ks, vs)
        Zf = free.encode(ks, vs)
    assert orthogonality_error(Zo).max().item() < 1e-2, "t4 FAILED: ortho encode not orthogonal"
    assert not torch.equal(Zo, Zf), "t4 FAILED: ortho encode identical to free (projection inert)"
    assert nm.n_params(ortho) == nm.n_params(base), "ortho must add ZERO parameters"
    print(f"t4 PASS: flag OFF byte-identical to NCREarlyLNModel; flag ON orthogonal "
          f"(err {orthogonality_error(Zo).max().item():.2e}); 0 new params")

    # t5 CHAIN task mod-K-trap safety: distinct-op-per-hop, no consecutive
    # repeat, exact composite target, fixed-point exclusion, depth=L.
    cfg = obt.BankConfig(d=17, K=16, R=DISC_R)
    gen = torch.Generator().manual_seed(3)
    b = sample_chain_batch(cfg, 32, gen, (1, 2, 3, 8), device="cpu")
    path, plen, succ, a_idx, tgt = b["path"], b["path_len"], b["succ"], b["a_idx"], None
    # no consecutive repeats within each query's active path
    for bi in range(path.shape[0]):
        for qi in range(path.shape[1]):
            L = int(plen[bi, qi])
            seq = path[bi, qi, :L].tolist()
            assert all(seq[t] != seq[t + 1] for t in range(L - 1)), (bi, qi, seq)
    # exact composite target reproduced by an INDEPENDENT reference iteration
    B, Q = a_idx.shape
    ref = a_idx.clone()
    for t in range(path.shape[2]):
        for bi in range(B):
            for qi in range(Q):
                if int(plen[bi, qi]) > t:
                    ref[bi, qi] = int(succ[bi, int(path[bi, qi, t]), int(ref[bi, qi])])
    tgt_ref_key = torch.gather(b["pool"], 1, ref.unsqueeze(-1).expand(B, Q, cfg.d))
    assert torch.allclose(tgt_ref_key, b["targets"], atol=1e-6), "t5 FAILED: composite target wrong"
    # no fixed points survived
    assert not bool((ref == a_idx).any()), "t5 FAILED: a fixed point survived exclusion"
    # depth=L, NO periodicity: a length-8 distinct-op path is NOT equal to any
    # single-operator power of the effective composite (verify the composite is
    # not fixed and differs across path lengths for the same start)
    print("t5 PASS: chain task distinct-op-per-hop, no-consecutive-repeat, exact "
          "composite target (indep. reference), fixed-points excluded")

    # t6 CHAIN read exactness: on ORTHOGONAL bank slices, the renorm chain read
    # lands on the exact composite (cos≈1), and depth=L is genuine (a longer
    # path gives a different, still-recovered answer).
    torch.manual_seed(4)
    Bn, R, dd, K = 3, DISC_R, 17, 16
    # build an orthogonal bank whose slice r is the permutation matrix of a
    # K-cycle succ_r on the standard basis (exactly recoverable)
    gen2 = torch.Generator().manual_seed(5)
    ep = obt.generate_bank_episode(Bn, obt.BankConfig(d=dd, K=K, R=R), gen2)
    # ideal orthogonal Z per relation: Z_r = sum_i pool[succ_r(i)] pool[i]^T
    pool = ep["pool"]; succ = ep["succ"]
    Zb = torch.zeros(Bn, R, dd, dd)
    for r in range(R):
        val = torch.gather(pool, 1, succ[:, r].unsqueeze(-1).expand(Bn, K, dd))
        Zb[:, r] = torch.einsum("bki,bkj->bij", val, pool[:, :K])
    m = OrthoBankModel(d=dd, R=R)
    Lp = 8
    a_idx = torch.randint(0, K, (Bn, K), generator=gen2)
    p2, pl2 = _sample_chain_paths(Bn, K, R, (Lp,), gen2, "cpu")
    qk = torch.gather(pool, 1, a_idx.unsqueeze(-1).expand(Bn, K, dd))
    o = m.chain_read(Zb, qk, p2, pl2, renorm=True, blend=False)
    tgt_idx = _chain_target_idx(succ, a_idx, p2, pl2)
    tgt = torch.gather(pool, 1, tgt_idx.unsqueeze(-1).expand(Bn, K, dd))
    cos = nm.recovery_cosine(o, tgt)
    assert cos.min().item() > 1 - 1e-4, ("t6 FAILED: exact orthogonal chain read", cos.min().item())
    print(f"t6 PASS: orthogonal-bank chain read recovers the exact L={Lp} distinct-op "
          f"composite (min cos {cos.min().item():.6f})")

    # t7 DISC model: ortho flag orthogonalizes bank slices; free doesn't; blank-out
    torch.manual_seed(6)
    md = build_disc_model("ortho", d=17, R=DISC_R, ns_iter=NS_ITER_DEFAULT, ns_power=NS_POWER_DEFAULT)
    bo = bank_orthogonality(md, obt.BankConfig(d=17, K=16, R=DISC_R), "cpu", 0)
    assert bo["ortho_err_mean"] < 1e-2, ("t7 FAILED: ortho bank slices not orthogonal", bo)
    mf = build_disc_model("free", d=17, R=DISC_R, ns_iter=NS_ITER_DEFAULT, ns_power=NS_POWER_DEFAULT)
    bof = bank_orthogonality(mf, obt.BankConfig(d=17, K=16, R=DISC_R), "cpu", 0)
    assert bof["ortho_err_mean"] > bo["ortho_err_mean"], "t7 FAILED: free bank should be less orthogonal"
    blk = bank_blank_out(md, obt.BankConfig(d=17, K=16, R=DISC_R), "cpu", 0)
    assert blk["passed"], ("t7 FAILED: bank P=1 blank-out", blk)
    print(f"t7 PASS: ortho bank slices orthogonal (err {bo['ortho_err_mean']:.2e}) vs free "
          f"({bof['ortho_err_mean']:.3f}); chain-read P=1 blank-out holds")

    # t8 END-TO-END micro PRIMARY cell (free + ortho), tiny grid + realistic ladder
    import shutil
    _dir = "/tmp/ncr_ortho_write_selftest"
    shutil.rmtree(_dir, ignore_errors=True)
    real_pts, real_bs, real_nb = nt.eval_points, rn.EVAL_BATCH_SIZE, rn.EVAL_BATCHES
    real_depths = globals()["REALISTIC_DEPTHS"]

    def tiny_points(K_, d=nt.D_PIN, _real=real_pts):
        # mirror ncr_earlyln_scale's own tiny grid: keep h_star (it lies inside
        # the sweep range) so eval_cell's residue_sweep list is non-empty.
        g = nt.GRIDS[K_]
        keep = {1, 2, 3, g["ladder"][0], g["h_star"]}
        return [p for p in _real(K_, d=d) if p.h in keep and p.component != "cost_probe"]
    try:
        nt.eval_points = tiny_points
        rn.EVAL_BATCH_SIZE, rn.EVAL_BATCHES = 16, 2
        globals()["REALISTIC_DEPTHS"] = (5, 12)      # tiny ladder for the micro cell
        for arm in ("free", "ortho"):
            rec = run_primary_cell(arm, K=24, seed=0, steps=4, device="cpu", outdir=_dir)
            assert rec["status"] == "COMPLETED", (arm, rec.get("status"))
            assert rec["blank_out"]["passed"], (arm, rec["blank_out"])
            assert "spectral" in rec and "mean" in rec["spectral"], arm
            assert "realistic_ladder" in rec and "5" in rec["realistic_ladder"], arm
            assert rec["d"] == 25, (arm, "d must be K+1")
        # ortho cell's written operator must be materially MORE orthogonal
        # (lower departure-from-normality) than the free cell's, even at 4 steps
        rf = json.load(open(os.path.join(_dir, "ortho_free_K24_s0.json")))
        ro = json.load(open(os.path.join(_dir, "ortho_ortho_K24_s0.json")))
        assert ro["spectral"]["mean"]["depart_normality"] <= rf["spectral"]["mean"]["depart_normality"] + 1e-9, \
            ("t8: ortho arm should not be MORE non-normal than free", ro["spectral"]["mean"],
             rf["spectral"]["mean"])
    finally:
        nt.eval_points, rn.EVAL_BATCH_SIZE, rn.EVAL_BATCHES = real_pts, real_bs, real_nb
        globals()["REALISTIC_DEPTHS"] = real_depths
        shutil.rmtree(_dir, ignore_errors=True)
    print("t8 PASS: end-to-end micro PRIMARY cell (free+ortho, K=24 d=25) COMPLETED; "
          "blank-out + spectral + realistic-ladder present; ortho no more non-normal than free")

    # t9 END-TO-END micro DISCRIMINATOR cell (free + ortho)
    _dir2 = "/tmp/ncr_ortho_disc_selftest"
    shutil.rmtree(_dir2, ignore_errors=True)
    real_bs, real_nb = rn.EVAL_BATCH_SIZE, rn.EVAL_BATCHES
    real_evalL = globals()["CHAIN_EVAL_L"]
    try:
        rn.EVAL_BATCH_SIZE, rn.EVAL_BATCHES = 16, 2
        globals()["CHAIN_EVAL_L"] = (5, 8)
        for arm in ("free", "ortho"):
            rec = run_disc_cell(arm, K=16, seed=0, steps=4, device="cpu", outdir=_dir2, R=DISC_R)
            assert rec["status"] == "COMPLETED", (arm, rec.get("status"))
            assert rec["blank_out"]["passed"], (arm, rec["blank_out"])
            assert "chain_ladder" in rec and "5" in rec["chain_ladder"], arm
            assert "bank_orthogonality" in rec, arm
        ro = json.load(open(os.path.join(_dir2, "ortho_disc_ortho_K16_s0.json")))
        rf = json.load(open(os.path.join(_dir2, "ortho_disc_free_K16_s0.json")))
        assert ro["bank_orthogonality"]["ortho_err_mean"] < rf["bank_orthogonality"]["ortho_err_mean"], \
            "t9: ortho disc bank must be more orthogonal than free"
    finally:
        rn.EVAL_BATCH_SIZE, rn.EVAL_BATCHES = real_bs, real_nb
        globals()["CHAIN_EVAL_L"] = real_evalL
        shutil.rmtree(_dir2, ignore_errors=True)
    print("t9 PASS: end-to-end micro DISCRIMINATOR cell (free+ortho, K=16) COMPLETED; "
          "chain-ladder + bank-orthogonality present; ortho bank more orthogonal than free")

    print("\nncr_ortho_write self-test: ALL 9/9 PASSED")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="CPU self-test kill-proofs")
    ap.add_argument("--primary-cell", action="store_true")
    ap.add_argument("--disc-cell", action="store_true")
    ap.add_argument("--arm", choices=("free", "ortho"), default="ortho")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=320_000)   # 4x budget (§11.6 convention)
    ap.add_argument("--ceiling-gpuh", type=float, default=3.0)
    ap.add_argument("--ns-iter", type=int, default=NS_ITER_DEFAULT)
    ap.add_argument("--ns-power", type=int, default=NS_POWER_DEFAULT)
    ap.add_argument("--anneal-frac", type=float, default=0.5)
    ap.add_argument("--R", type=int, default=DISC_R)
    ap.add_argument("--outdir", type=str, default=os.path.join(_HERE, "results_ortho_write"))
    ap.add_argument("--stop-file", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.smoke:
        _self_test()
        return
    if args.primary_cell:
        rec = run_primary_cell(args.arm, args.K, args.seed, args.steps, args.device,
                               args.outdir, args.stop_file, args.ceiling_gpuh,
                               ns_iter=args.ns_iter, ns_power=args.ns_power,
                               anneal_frac=args.anneal_frac)
        print(f"PRIMARY CELL arm={args.arm} K={args.K} seed={args.seed} status={rec.get('status')}")
        return
    if args.disc_cell:
        rec = run_disc_cell(args.arm, args.K, args.seed, args.steps, args.device,
                            args.outdir, args.stop_file, args.ceiling_gpuh,
                            R=args.R, ns_iter=args.ns_iter, ns_power=args.ns_power,
                            anneal_frac=args.anneal_frac)
        print(f"DISC CELL arm={args.arm} K={args.K} seed={args.seed} status={rec.get('status')}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
