#!/usr/bin/env python3
"""Orthogonal-complement characterization of archived fast-weight Z-dumps.

Novel-arch waterfall zero-GPU piggyback (NOVEL_ARCH_WATERFALL.md section 1):
decompose each trained state Z into its task-subspace block and the
ORTHOGONAL COMPLEMENT, then characterize what lives in the complement --
near-zero / structured noise / SYSTEMATIC structure. Extends the C/D-block
machinery of TASK_E_FINDINGS.md sections 9-10 (analyze_zdump.py) with:

  1. Energy fractions: ||A||^2, ||B||^2, ||C||^2, ||D||^2 as fractions of
     ||Z||_F^2 (exact decomposition; the four blocks tile Z in the [E, E-perp]
     basis, so the fractions sum to 1 up to fp noise).
  2. Complement spectrum: full singular values of D; condition number,
     coefficient of variation (flatness), top-direction energy share.
  3. Conformal-coupling test (the systematicity candidate visible in the
     published section 9/10 tables: rho(D) tracks c* to ~1%): orthogonal
     Procrustes fit D ~= c_hat * Q_hat (Q_hat = W X^T from D = W Sigma X^T,
     c_hat = mean singular value); report relative residual ||D - c_hat
     Q_hat||_F / ||D||_F, plus the ratios rho(D)/c* and c_hat/c* where c* is
     analyze_zdump's task-block scale fit. Reference: the same Procrustes
     residual for iid Gaussian matrices of the same shape (sampled null).
  4. Identity alignment tau(D) = tr(D) / (||D||_F sqrt(d-K)) in [-1,1]
     (near-1 would mean D is a scaled identity, e.g. a decay/bias artifact;
     near-0 is what a generic rotation gives). One-sided cases only.
  5. Cycle-echo test: Hungarian phase-residual of D's unit-normalized
     eigenvalues against the (d-K)-th roots of unity, ranked against a
     random-orthogonal null -- does the complement echo the task's cyclic
     phase structure? One-sided cases only.
  6. Cross-example content stability (the novelty-detector question): the
     ambient complement component Zc = P_perp Z P_perp (one-sided) or
     Pv_perp S Pk_perp (two-sided) is a basis-free object; pairwise Frobenius
     cosine across the 4 eval examples within a run, against a matched null
     (same subspaces, complement content replaced by c_hat * random
     orthogonal) isolates content alignment from subspace-geometry overlap.
     Full-Z and task-block (U A U^T) pairwise cosines reported for context.
  7. Cross-run correlations: complement energy fraction / Procrustes residual
     vs the runs' own recovery metrics (Task E: M3 h=21 mean_cos and
     recovered_frac@0.9; keyanchor: final-checkpoint M3 max-hop mean_cos and
     final training loss), Spearman, small-n caveats apply.

Datasets (all final-state eval-time dumps; NO per-checkpoint states exist in
any archive, so the across-checkpoint axis is out of scope by data
availability):
  - Task E K=8 d=16 (40k round, 5 frN + 3 fr7 seeds), complement dim 8
  - Task E K=12 d=16 (80k round, 3 seeds), complement dim 4
  - Task E K=16=d excluded: complement is 0-dimensional by construction
  - keyanchor dstate wave d_state=128, K in {68,76,84,92} x 3 seeds,
    complement dims {60,52,44,36}. Keys are exactly orthonormal and
    s_ideal_effective has a clean rank-K cut (gap ~1e-7), but key-span !=
    value-span (principal angles up to ~67 deg), so the decomposition is
    TWO-SIDED there: A = Uv^T S Uk, B = Uv^T S Vk, C = Vv^T S Uk,
    D = Vv^T S Vk. Identity/eigenphase tests are basis-pairing-dependent and
    therefore skipped for the two-sided case; energy fractions, spectrum
    flatness, and Procrustes conformality are basis-choice-invariant.

Reuses analyze_zdump.py's machinery (entity_subspace, block_decompose,
effective_rank, Hungarian matcher) by import. numpy + stdlib only, CPU,
runs in ~1 minute.

Usage:
  .venv/bin/python complement_analysis.py \
      --taske-40k <dir> --taske-80k <dir> --keyanchor <dir> \
      --out complement_results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

# Same Accelerate-BLAS spurious-RuntimeWarning silencing as analyze_zdump.py
# (documented in TASK_E_FINDINGS.md section 10 methodology note); every block
# is still asserted finite by block_decompose.
warnings.filterwarnings("ignore", category=RuntimeWarning)

_CH2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "matrix-thinking", "chapter2")
sys.path.insert(0, os.path.abspath(_CH2))
from analyze_zdump import (  # noqa: E402
    block_decompose, effective_rank, entity_subspace, match_eigenvalues,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rand_orthogonal(n: int) -> np.ndarray:
    """Haar-ish random orthogonal via QR of a Gaussian draw (sign-fixed)."""
    M = RNG.standard_normal((n, n))
    Q, R = np.linalg.qr(M)
    return Q * np.sign(np.diag(R))


def procrustes_conformal(D: np.ndarray):
    """Best conformal (scaled-orthogonal) fit: min_{c,Q orth} ||D - cQ||_F.
    Closed form: D = W Sigma X^T, Q_hat = W X^T, c_hat = mean(Sigma).
    Returns (c_hat, rel_resid = ||D - c_hat Q_hat||_F / ||D||_F)."""
    W, sv, Xt = np.linalg.svd(D)
    c_hat = float(sv.mean())
    resid = np.linalg.norm(D - c_hat * (W @ Xt), "fro")
    return c_hat, float(resid / max(np.linalg.norm(D, "fro"), 1e-15)), sv


def gaussian_procrustes_null(n: int, n_samples: int = 200):
    """Procrustes relative residual for iid Gaussian n x n matrices (norm-
    invariant statistic, so no scaling needed). Returns (mean, std)."""
    vals = []
    for _ in range(n_samples):
        _, r, _ = procrustes_conformal(RNG.standard_normal((n, n)))
        vals.append(r)
    return float(np.mean(vals)), float(np.std(vals))


def eigenphase_cycle_echo(D: np.ndarray, n_null: int = 200):
    """Hungarian phase residual (mean over modes) of D's unit-normalized
    eigenvalues vs the m-th roots of unity, plus the percentile of that
    residual under a random-orthogonal null (small percentile => D's phases
    align with the roots more than a random rotation's would => cycle echo)."""
    m = D.shape[0]
    roots = np.exp(2j * np.pi * np.arange(m) / m)

    def resid(M):
        ev = np.linalg.eigvals(M)
        ev = ev / np.clip(np.abs(ev), 1e-12, None)
        _, r = match_eigenvalues(ev, roots)
        return float(np.mean(r))

    obs = resid(D)
    null = np.array([resid(rand_orthogonal(m)) for _ in range(n_null)])
    pct = float(np.mean(null <= obs))
    return obs, float(null.mean()), float(null.std()), pct


def pairwise_abs_cos(mats):
    """Mean |Frobenius cosine| over all unordered pairs of a list of same-
    shape matrices (nan-guarded for zero matrices)."""
    cs = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            a, b = mats[i], mats[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                continue
            cs.append(abs(float(np.sum(a * b) / (na * nb))))
    return float(np.mean(cs)) if cs else float("nan"), cs


def spearman(x, y):
    """Spearman rho, stdlib-style (average ranks, then Pearson on ranks)."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def ranks(v):
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        # average ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# per-example analysis
# ---------------------------------------------------------------------------
def analyze_example(Z, U_out, V_out, U_in, V_in, Zi, one_sided: bool,
                    n_null_pairs: int = 0):
    """Block-decompose Z with output-side bases (U_out, V_out) and input-side
    bases (U_in, V_in) (identical for the one-sided Task E case). Returns the
    per-example record. Ambient complement component returned for the cross-
    example step."""
    A = U_out.T @ Z @ U_in
    B = U_out.T @ Z @ V_in
    C = V_out.T @ Z @ U_in
    D = V_out.T @ Z @ V_in
    for name, M in (("A", A), ("B", B), ("C", C), ("D", D)):
        if not np.isfinite(M).all():
            raise FloatingPointError(f"non-finite block {name}")

    nZ2 = float(np.linalg.norm(Z, "fro") ** 2)
    fA = float(np.linalg.norm(A, "fro") ** 2 / nZ2)
    fB = float(np.linalg.norm(B, "fro") ** 2 / nZ2)
    fC = float(np.linalg.norm(C, "fro") ** 2 / nZ2)
    fD = float(np.linalg.norm(D, "fro") ** 2 / nZ2)

    Pi = U_out.T @ Zi @ U_in
    c_star = float(np.sum(A * Pi) / max(np.sum(Pi * Pi), 1e-15))

    rec = dict(fA=fA, fB=fB, fC=fC, fD=fD, block_sum=fA + fB + fC + fD,
               normZ=float(np.sqrt(nZ2)), c_star=c_star,
               m=int(D.shape[0]))
    if D.size:
        c_hat, proc_resid, sv = procrustes_conformal(D)
        ev_abs = np.abs(np.linalg.eigvals(D)) if one_sided else None
        rec.update(
            D_svals=sv.tolist(),
            D_cond=float(sv[0] / max(sv[-1], 1e-15)),
            D_sval_cv=float(sv.std() / max(sv.mean(), 1e-15)),
            D_top_dir_energy=float(sv[0] ** 2 / max((sv ** 2).sum(), 1e-15)),
            D_eff_rank=effective_rank(D),
            D_rho=float(ev_abs.max()) if one_sided else None,
            c_hat=c_hat,
            proc_resid=proc_resid,
            rho_over_cstar=(float(ev_abs.max() / c_star)
                            if one_sided and abs(c_star) > 1e-9 else None),
            chat_over_cstar=(float(c_hat / c_star)
                             if abs(c_star) > 1e-9 else None),
            tau_identity=(float(np.trace(D) /
                                (np.linalg.norm(D, "fro") * np.sqrt(D.shape[0])))
                          if one_sided else None),
        )
    if one_sided:
        # Direct test of the "scaled-identity + low-rank task correction"
        # reading implied by D ~= c* I with B,C ~= 0: if that holds exactly,
        # Z = c* I_d + U (A - c* I_K) U^T, a rank-<=K ambient correction
        # (rank K-1 for an exact cycle, whose Pi has one eigenvalue at +1).
        d_amb = Z.shape[0]
        U = U_out
        recon = rec["c_star"] * np.eye(d_amb) + \
            U @ (A - rec["c_star"] * np.eye(A.shape[0])) @ U.T
        rec["id_lowrank_resid"] = float(
            np.linalg.norm(Z - recon, "fro") /
            max(np.linalg.norm(Z, "fro"), 1e-15))
        rec["effrank_Z_minus_cI"] = effective_rank(
            Z - rec["c_star"] * np.eye(d_amb))
    # ambient complement component (basis-free)
    Zc = (V_out @ V_out.T) @ Z @ (V_in @ V_in.T)
    ZE = (U_out @ U_out.T) @ Z @ (U_in @ U_in.T)
    return rec, D, Zc, ZE


def analyze_run_taske(path: str):
    with open(path) as f:
        d = json.load(f)
    if "Z_dump" not in d:
        return None
    K, dim = d["K"], d["d"]
    if dim - K <= 0:
        return None
    Zs = np.array(d["Z_dump"]["Z"], dtype=np.float64)
    Zis = np.array(d["Z_dump"]["z_ideal"], dtype=np.float64)
    recs, Ds, Zcs, ZEs, Vs = [], [], [], [], []
    for ex in range(Zs.shape[0]):
        sub = entity_subspace(Zis[ex])
        assert sub["k_eff"] == K, f"k_eff {sub['k_eff']} != K {K} in {path}"
        U, V = sub["U"], sub["V"]
        rec, D, Zc, ZE = analyze_example(Zs[ex], U, V, U, V, Zis[ex],
                                         one_sided=True)
        echo_obs, echo_null_mean, echo_null_std, echo_pct = \
            eigenphase_cycle_echo(D)
        rec.update(echo_resid=echo_obs, echo_null_mean=echo_null_mean,
                   echo_null_std=echo_null_std, echo_pct=echo_pct)
        recs.append(rec)
        Ds.append(D)
        Zcs.append(Zc)
        ZEs.append(ZE)
        Vs.append(V)
    xex = cross_example_stats(Zs, Zcs, ZEs, Vs, recs, one_sided=True)
    name = os.path.basename(path).replace("t1_matrix_permutation_", "") \
                                 .replace(".json", "")
    m3 = d.get("M3_held_out", {}).get("21", {})
    return dict(name=name, K=K, d=dim, family="taskE",
                force_rank_k=d.get("force_rank_k"), seed=d.get("seed"),
                h21_mean_cos=m3.get("mean_cos"),
                h21_recov=m3.get("recovered_frac@0.9"),
                per_example=recs, cross_example=xex)


def analyze_run_keyanchor(path: str):
    with open(path) as f:
        d = json.load(f)
    if "Z_dump" not in d:
        return None
    K, dim = d["K"], d["d_state"]
    zd = d["Z_dump"]
    Ss = np.array(zd["S_T_raw"], dtype=np.float64)
    Sis = np.array(zd["s_ideal_effective"], dtype=np.float64)
    recs, Zcs, ZEs = [], [], []
    key_val_angle_max = []
    for ex in range(Ss.shape[0]):
        u, s, vt = np.linalg.svd(Sis[ex])
        thresh = s[0] * 1e-3
        k_eff = int(np.sum(s > thresh))
        assert k_eff == K, f"k_eff {k_eff} != K {K} in {path}"
        Uv, Vv = u[:, :K], u[:, K:]          # value/output side
        Uk, Vk = vt[:K].T, vt[K:].T          # key/input side
        pa = np.degrees(np.arccos(np.clip(
            np.linalg.svd(Uv.T @ Uk, compute_uv=False), -1, 1)))
        key_val_angle_max.append(float(pa.max()))
        rec, D, Zc, ZE = analyze_example(Ss[ex], Uv, Vv, Uk, Vk, Sis[ex],
                                         one_sided=False)
        recs.append(rec)
        Zcs.append(Zc)
        ZEs.append(ZE)
    xex = cross_example_stats(Ss, Zcs, ZEs, None, recs, one_sided=False)
    ck = d["checkpoints"][-1]
    m3 = ck.get("M3_held_out", {})
    max_hop = max((int(h) for h in m3), default=None)
    m3e = m3.get(str(max_hop), {}) if max_hop else {}
    name = os.path.basename(path).split("_geo3")[0].replace(
        "wkeyanchor-k48_rdx_", "")
    return dict(name=name, K=K, d=dim, family="keyanchor",
                seed=d.get("seed"),
                key_val_angle_max_deg=float(np.mean(key_val_angle_max)),
                final_loss=d["trajectory"][-1]["loss"],
                m3_max_hop=max_hop,
                m3_mean_cos=m3e.get("mean_cos"),
                m3_recov=m3e.get("recovered_frac@0.9"),
                per_example=recs, cross_example=xex)


def cross_example_stats(Zs, Zcs, ZEs, Vs, recs, one_sided: bool,
                        n_null: int = 100):
    """Within-run, across-example content alignment of the ambient complement
    component, with a matched random-content null (same subspaces, D replaced
    by c_hat * random orthogonal / random Gaussian for the two-sided case --
    both give the rotation-invariant null needed here)."""
    comp_cos, _ = pairwise_abs_cos(Zcs)
    full_cos, _ = pairwise_abs_cos(list(Zs))
    task_cos, _ = pairwise_abs_cos(ZEs)
    # subspace-geometry-only null
    null_means = []
    if one_sided and Vs is not None:
        for _ in range(n_null):
            fake = []
            for V, rec in zip(Vs, recs):
                m = V.shape[1]
                fake.append(V @ (rec.get("c_hat", 1.0) * rand_orthogonal(m)) @ V.T)
            nc, _ = pairwise_abs_cos(fake)
            null_means.append(nc)
    else:
        # two-sided: replace each Zc by an equal-norm Gaussian supported on
        # the same (row, col) complement subspaces. Rebuild the projectors
        # from the stored ambient components' own row/col spaces.
        for _ in range(n_null):
            fake = []
            for Zc in Zcs:
                G = RNG.standard_normal(Zc.shape)
                u, s, vt = np.linalg.svd(Zc)
                m = int(np.sum(s > s[0] * 1e-9)) if s[0] > 0 else 0
                if m == 0:
                    fake.append(np.zeros_like(Zc))
                    continue
                P_out = u[:, :m] @ u[:, :m].T
                P_in = vt[:m].T @ vt[:m]
                F = P_out @ G @ P_in
                F *= np.linalg.norm(Zc) / max(np.linalg.norm(F), 1e-15)
                fake.append(F)
            nc, _ = pairwise_abs_cos(fake)
            null_means.append(nc)
    null_means = np.array(null_means)
    return dict(complement_pair_cos=comp_cos,
                complement_pair_cos_null_mean=float(null_means.mean()),
                complement_pair_cos_null_std=float(null_means.std()),
                complement_pair_cos_null_p=(
                    float(np.mean(null_means >= comp_cos))),
                fullZ_pair_cos=full_cos,
                taskblock_pair_cos=task_cos)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def summarize_run(run):
    pe = run["per_example"]

    def m(k):
        vals = [r[k] for r in pe if r.get(k) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return dict(run=run["name"], family=run["family"], K=run["K"], d=run["d"],
                m=pe[0]["m"], fA=m("fA"), fBC=m("fB") + m("fC"), fD=m("fD"),
                c_star=m("c_star"), rho_over_cstar=m("rho_over_cstar"),
                chat_over_cstar=m("chat_over_cstar"),
                proc_resid=m("proc_resid"), D_cond=m("D_cond"),
                D_sval_cv=m("D_sval_cv"), tau_identity=m("tau_identity"),
                echo_pct=m("echo_pct"), id_lowrank_resid=m("id_lowrank_resid"),
                effrank_Z_minus_cI=m("effrank_Z_minus_cI"),
                comp_pair_cos=run["cross_example"]["complement_pair_cos"],
                comp_pair_null=run["cross_example"][
                    "complement_pair_cos_null_mean"],
                comp_pair_null_p=run["cross_example"][
                    "complement_pair_cos_null_p"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taske-40k", required=True)
    ap.add_argument("--taske-80k", required=True)
    ap.add_argument("--keyanchor", required=True)
    ap.add_argument("--out", default="complement_results.json")
    args = ap.parse_args()

    runs = []
    for p in sorted(glob.glob(os.path.join(args.taske_40k, "*.json"))):
        r = analyze_run_taske(p)
        if r:
            runs.append(r)
    for p in sorted(glob.glob(os.path.join(args.taske_80k, "*.json"))):
        r = analyze_run_taske(p)
        if r:
            runs.append(r)
    for p in sorted(glob.glob(os.path.join(args.keyanchor, "*.json"))):
        r = analyze_run_keyanchor(p)
        if r:
            runs.append(r)

    # Gaussian Procrustes references per complement size
    sizes = sorted({r["per_example"][0]["m"] for r in runs})
    gauss_ref = {m: gaussian_procrustes_null(m) for m in sizes}

    print("=" * 110)
    print("Z-DUMP ORTHOGONAL-COMPLEMENT CHARACTERIZATION "
          "(complement block D; means over 4 eval examples per run)")
    print("=" * 110)
    hdr = (f"{'run':<24}{'K':>4}{'m':>4} {'fA':>7} {'fB+fC':>7} {'fD':>7} "
           f"{'c*':>7} {'rho/c*':>7} {'chat/c*':>8} {'procR':>7} "
           f"{'gaussR':>7} {'cond':>7} {'svCV':>6} {'tauI':>6} {'echo%':>6} "
           f"{'idLRres':>8} {'rk(Z-cI)':>9} {'xcos':>6} {'null':>6} {'p':>5}")
    print(hdr)
    print("-" * 110)
    summaries = []
    for run in runs:
        s = summarize_run(run)
        summaries.append(s)
        gm, gs = gauss_ref[s["m"]]

        def f(x, nd=3):
            return "  n/a" if x is None or (isinstance(x, float) and
                                            np.isnan(x)) else f"{x:.{nd}f}"

        print(f"{s['run']:<24}{s['K']:>4}{s['m']:>4} {f(s['fA']):>7} "
              f"{f(s['fBC']):>7} {f(s['fD']):>7} {f(s['c_star']):>7} "
              f"{f(s['rho_over_cstar']):>7} {f(s['chat_over_cstar']):>8} "
              f"{f(s['proc_resid']):>7} {gm:>7.3f} {f(s['D_cond']):>7} "
              f"{f(s['D_sval_cv']):>6} {f(s['tau_identity']):>6} "
              f"{f(s['echo_pct'], 2):>6} {f(s['id_lowrank_resid']):>8} "
              f"{f(s['effrank_Z_minus_cI'], 2):>9} {f(s['comp_pair_cos']):>6} "
              f"{f(s['comp_pair_null']):>6} {f(s['comp_pair_null_p'], 2):>5}")
    print("-" * 110)
    print("procR = ||D - c_hat*Q_hat||_F/||D||_F (conformal Procrustes "
          "residual); gaussR = same statistic, iid Gaussian null (mean)")
    print("tauI = tr(D)/(||D||_F sqrt(m)) identity alignment; echo% = "
          "percentile of D's eigenphase-vs-roots residual under a random-"
          "orthogonal null (small => cycle echo)")
    print("xcos = mean |Frobenius cos| of ambient complement components "
          "across the 4 eval examples; null = matched random-content null; "
          "p = P(null >= observed)")
    print("idLRres = ||Z - (c* I_d + U(A - c* I_K)U^T)||_F/||Z||_F -- direct "
          "residual of the 'scaled-identity + rank-K task correction' model; "
          "rk(Z-cI) = effective_rank(Z - c* I_d), ~K-1 if that model holds")

    # ---------------- correlations ----------------
    print()
    print("=" * 110)
    print("CROSS-RUN CORRELATIONS (Spearman; small n -- read qualitatively)")
    print("=" * 110)
    te = [(s, r) for s, r in zip(summaries, runs) if r["family"] == "taskE"
          and r.get("h21_mean_cos") is not None]
    # converged Task E = exclude the two dead fr7 seeds (rank-collapsed A)
    te_conv = [(s, r) for s, r in te
               if s["proc_resid"] < 0.5 and s["fA"] > 0.05]
    for label, group in (("Task E all (n=%d)" % len(te), te),
                         ("Task E non-dead (n=%d)" % len(te_conv), te_conv)):
        if len(group) < 3:
            continue
        fD = [s["fD"] for s, _ in group]
        pr = [s["proc_resid"] for s, _ in group]
        cos21 = [r["h21_mean_cos"] for _, r in group]
        rec21 = [r["h21_recov"] for _, r in group]
        print(f"{label}: rho(fD, h21_mean_cos)={spearman(fD, cos21):+.3f}  "
              f"rho(fD, h21_recov)={spearman(fD, rec21):+.3f}  "
              f"rho(procR, h21_mean_cos)={spearman(pr, cos21):+.3f}  "
              f"rho(procR, h21_recov)={spearman(pr, rec21):+.3f}")
    ka = [(s, r) for s, r in zip(summaries, runs)
          if r["family"] == "keyanchor"]
    if len(ka) >= 3:
        fD = [s["fD"] for s, _ in ka]
        pr = [s["proc_resid"] for s, _ in ka]
        Ks = [s["K"] for s, _ in ka]
        loss = [r["final_loss"] for _, r in ka]
        mcos = [r["m3_mean_cos"] for _, r in ka]
        print(f"keyanchor (n={len(ka)}): rho(fD, final_loss)="
              f"{spearman(fD, loss):+.3f}  rho(fD, m3_mean_cos)="
              f"{spearman(fD, mcos):+.3f}  rho(fD, K)="
              f"{spearman(fD, Ks):+.3f}  rho(procR, final_loss)="
              f"{spearman(pr, loss):+.3f}")
        print(f"keyanchor fD by K: " + "  ".join(
            f"K{k}={np.mean([s['fD'] for s, _ in ka if s['K'] == k]):.4f}"
            for k in sorted(set(Ks))))

    # ---------------- persist ----------------
    out = dict(
        gaussian_procrustes_ref={str(k): v for k, v in gauss_ref.items()},
        summaries=summaries,
        runs=[{k: v for k, v in r.items()} for r in runs],
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
