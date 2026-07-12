#!/usr/bin/env python3
"""Q3 mechanism analysis (NCR_MAPPING_LAW_DESIGN.md, commit d90abff, section
"Q3 -- mechanism hypotheses"): why does the conventional d=2K mapping kill
NCR trainability/far-depth while d=K+1 works?

CPU-only, zero new GPU. Recomputes from the persisted z_dump.Z / z_dump.z_ideal
arrays via ncr_spectral.analyze_zdump_arrays -- NOT from any deep_probe
per_example/normB/normC/normD path (an earlier design draft asserted those
fields were already recorded; the independent attack round verified this is
false by grepping actual cell JSONs, 0 matches for "normB" in either a 2K or
K+1 cell -- see the design's own Q3 "Data-provenance correction"). This
script re-verifies that correction itself, on every cell, not just the one
spot-check the design round already did.

Data: the exact 16-cell set the design names (no more, no less -- see the
design's own H1/H2 "Discriminating data" paragraphs):
  K=16, d=17 (K+1)  x4 seeds  experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/
  K=16, d=32 (2K)   x4 seeds  experiment-runs/2026-07-11_ncr_earlyln_scale/
  K=24, d=25 (K+1)  x4 seeds  experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/
  K=24, d=48 (2K)   x4 seeds  experiment-runs/2026-07-11_ncr_earlyln_scale/

Hypotheses tested (design's exact wording, condensed):
  H1 -- over-parameterized write space -> diffuse operators. Normalized
       leakage (normB+normC+normD)/normA, normA := |c*| * ||Pi||_F (the
       same denominator convention as the already-recorded
       scale_corrected_residual field), materially LARGER at d=2K than
       d=K+1 for matched K.
  H2 -- refines H1: is the leakage "eye-padding" (D near-zero / near-
       identity-scaled, near-flat condition number, cost nothing
       functionally) or "diffuse corruption" (D large & peaked, B/C
       comparable to D, competes with the entity subspace)? Same blocks,
       one more column.
  H3 -- LN-anneal/dimension interaction. loss_history (logged every 500
       steps) during the anneal window (step < anneal_frac * total_steps)
       inspected for a d-correlated divergence. Flagged LOW-CONFIDENCE by
       the design itself (a real in-house precedent -- the K16 4x NO-LAW
       anomaly -- already showed a write-geometry regression can be
       loss-invisible; a negative H3 read here is not informative, only a
       clean positive would be).

n discipline: 4 independent trained seeds per (K,d) cell; each seed's Z-dump
carries 4 eval EXAMPLES (different synthetic key/value draws through the
SAME trained weights -- confirmed non-identical, not re-samples of one
example). Examples within a seed are correlated (shared weights); seeds are
the independent unit. Every statistic below is reported BOTH ways: per-seed
means (n=4, the honest independent-unit count) and the full per-example
pool (n=16, descriptive only, explicitly labeled correlated-within-seed).
No significance test is computed or implied -- n=4 does not support one;
this script reports numbers, not verdicts.

Determinism: fixed cell order, fixed hops, no RNG anywhere in this script
(all recomputation is a deterministic function of the archived arrays).
Every input file's md5 is logged in the output JSON's "input_md5" block.

Usage:
    /Users/samuellarson/Experiments/learned-representations/.venv/bin/python3 \
        analyze_dratio_blocks.py [--out OUTPATH]

Self-contained beyond numpy + ncr_spectral/analyze_zdump (both already
audited elsewhere in this program, imported verbatim, never re-derived).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
if _CHAPTER2 not in sys.path:
    sys.path.insert(0, _CHAPTER2)

import ncr_spectral as ns             # noqa: E402 (audited, verbatim import)
import analyze_zdump as az            # noqa: E402 (audited, verbatim import -- ns's own lineage)

# ---------------------------------------------------------------------------
# The exact 16-cell set the design names. Paths relative to repo root.
# ---------------------------------------------------------------------------
CELLS = [
    dict(K=16, d=17, ratio="K+1",
         dirpath="experiment-runs/2026-07-12_ncr_nextlever_wave/dratio",
         fname_tmpl="earlyln_K16_s{seed}.json"),
    dict(K=16, d=32, ratio="2K",
         dirpath="experiment-runs/2026-07-11_ncr_earlyln_scale",
         fname_tmpl="earlyln_K16_s{seed}.json"),
    dict(K=24, d=25, ratio="K+1",
         dirpath="experiment-runs/2026-07-12_ncr_nextlever_wave/dratio",
         fname_tmpl="earlyln_K24_s{seed}.json"),
    dict(K=24, d=48, ratio="2K",
         dirpath="experiment-runs/2026-07-11_ncr_earlyln_scale",
         fname_tmpl="earlyln_K24_s{seed}.json"),
]
SEEDS = (0, 1, 2, 3)
HOPS = (1,)   # only c*/A_eff_rank/phase_resid/norms are used below; these do
              # not depend on `hops` (analyze_zdump_arrays's mean_curve does,
              # but mean_curve is not consumed by H1/H2/H3) -- kept minimal
              # for speed, not because a richer hops set would change H1/H2/H3.
SCALAR_TOL = 1e-6   # relative tolerance for the recorded-scalar cross-check


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cell(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cross_verify_scalars(cell_json: dict, probe: dict, cell_tag: str) -> list:
    """Compare analyze_zdump_arrays's recomputed per-example scalars against
    the cell's own recorded deep_probe fields. Returns a list of mismatch
    dicts (empty if everything agrees within SCALAR_TOL). This is the exact
    check the design's attack round ran on ONE cell (earlyln_K24_s0,
    Probe A) -- here it runs on all 16."""
    dp = cell_json.get("deep_probe")
    mismatches = []
    if not dp:
        mismatches.append(dict(cell=cell_tag, field="deep_probe", issue="MISSING"))
        return mismatches
    recorded = dict(
        c_star=dp.get("c_star_per_example"),
        phase_resid_max=dp.get("phase_resid_max_per_example"),
        scale_corrected_residual=dp.get("scale_corrected_residual"),
        A_eff_rank=dp.get("A_eff_rank"),
    )
    for field, rec_list in recorded.items():
        if rec_list is None:
            mismatches.append(dict(cell=cell_tag, field=field, issue="MISSING in deep_probe"))
            continue
        recomputed = [ex[field] for ex in probe["per_example"]]
        if len(recomputed) != len(rec_list):
            mismatches.append(dict(cell=cell_tag, field=field,
                                    issue=f"length mismatch rec={len(rec_list)} recomp={len(recomputed)}"))
            continue
        for i, (r, c) in enumerate(zip(rec_list, recomputed)):
            denom = max(abs(r), 1e-12)
            rel = abs(r - c) / denom
            if rel > SCALAR_TOL:
                mismatches.append(dict(cell=cell_tag, field=field, example=i,
                                        recorded=r, recomputed=c, rel_err=rel))
    return mismatches


def analyze_cell(spec: dict, seed: int) -> dict:
    fname = spec["fname_tmpl"].format(seed=seed)
    path = os.path.join(_REPO_ROOT, spec["dirpath"], fname)
    cell_tag = f"K{spec['K']}_d{spec['d']}_{spec['ratio']}_s{seed}"
    cell_json = load_cell(path)

    # Guard against silent shape coercion / wrong-file loads: assert the
    # loaded cell's own K/d match what this script asked for BEFORE any
    # numpy reshaping happens. d=17 vs d=32 arrays differ in shape --
    # mishandling here is exactly the failure mode the audit charter warns
    # about.
    assert cell_json["K"] == spec["K"], (path, "K mismatch", cell_json["K"], spec["K"])
    assert cell_json["d"] == spec["d"], (path, "d mismatch", cell_json["d"], spec["d"])
    assert cell_json["status"] == "COMPLETED", (path, cell_json["status"])
    assert cell_json["blank_out"]["passed"] is True, (path, "blank_out did not pass")

    zdump = cell_json["z_dump"]
    Z_list_raw, Zi_list_raw = zdump["Z"], zdump["z_ideal"]
    assert len(Z_list_raw) == len(Zi_list_raw) == 4, (path, "expected 4 eval examples")
    for i, (Zr, Zir) in enumerate(zip(Z_list_raw, Zi_list_raw)):
        Za = np.asarray(Zr, dtype=np.float64)
        Zia = np.asarray(Zir, dtype=np.float64)
        assert Za.shape == (spec["d"], spec["d"]), (path, i, "Z shape", Za.shape, spec["d"])
        assert Zia.shape == (spec["d"], spec["d"]), (path, i, "z_ideal shape", Zia.shape, spec["d"])

    probe = ns.analyze_zdump_arrays(Z_list_raw, Zi_list_raw, HOPS)
    mismatches = cross_verify_scalars(cell_json, probe, cell_tag)

    # Per-example derived quantities not returned by analyze_zdump_arrays:
    # normA_ref = |c*| * ||Pi||_F (H1's own denominator convention, matching
    # the already-recorded scale_corrected_residual's denominator), plus D's
    # spectral shape (H2's eye-padding-vs-diffuse-corruption discriminator),
    # recomputed with the SAME az.entity_subspace/block_decompose calls
    # analyze_zdump_arrays uses internally (no drift -- verbatim reuse).
    examples = []
    for i, (Zr, Zir, ex) in enumerate(zip(Z_list_raw, Zi_list_raw, probe["per_example"])):
        Z = np.asarray(Zr, dtype=np.float64)
        Zi = np.asarray(Zir, dtype=np.float64)
        sub = az.entity_subspace(Zi)
        U, V = sub["U"], sub["V"]
        A, B, C, D = az.block_decompose(Z, U, V)
        Pi = U.T @ Zi @ U
        c_star = ex["c_star"]
        normPi = float(np.linalg.norm(Pi, "fro"))
        normA_ref = abs(c_star) * normPi
        normB, normC, normD = ex["normB"], ex["normC"], ex["normD"]
        leak_sum = normB + normC + normD
        leak_ratio = leak_sum / max(normA_ref, 1e-12)
        D_share = normD / max(leak_sum, 1e-12)
        D_eff_rank = az.effective_rank(D) if D.size else 0.0
        D_stable_rank = az.stable_rank(D) if D.size else 0.0
        D_spectral_radius = float(np.max(np.abs(np.linalg.eigvals(D)))) if D.size else 0.0
        D_svals = np.linalg.svd(D, compute_uv=False) if D.size else np.array([0.0])
        D_condition_number = float(D_svals[0] / max(D_svals[-1], 1e-12)) if D.size else 0.0

        examples.append(dict(
            example=i,
            k_eff=ex["k_eff"],
            c_star=c_star,
            normPi=normPi,
            normA_ref=normA_ref,
            normB=normB, normC=normC, normD=normD,
            leak_ratio=leak_ratio,
            D_share_of_leak=D_share,
            D_eff_rank=D_eff_rank,
            D_ambient_dim=int(D.shape[0]),
            D_stable_rank=D_stable_rank,
            D_spectral_radius=D_spectral_radius,
            D_condition_number=D_condition_number,
            phase_resid_max=ex["phase_resid_max"],
            scale_corrected_residual=ex["scale_corrected_residual"],
            A_eff_rank=ex["A_eff_rank"],
        ))

    # H3: loss_history during the anneal window.
    train = cell_json["train"]
    total_steps = train["step"]
    anneal_frac = cell_json.get("anneal_frac")
    anneal_frac_source = "recorded"
    if anneal_frac is None:
        # Older archive vintage (2026-07-11 dir, pre-Probe-B): anneal_frac
        # was not yet a persisted field; that script version hardcoded a
        # first-half anneal (module docstring: "annealed 1.0->0.0 over the
        # first half of training"), i.e. the same effective 0.5. Assumed,
        # not verified from this JSON alone -- disclosed, not silently
        # imputed as if measured.
        anneal_frac = 0.5
        anneal_frac_source = "assumed_0.5_unrecorded_pre_probeB_vintage"
    anneal_end_step = anneal_frac * total_steps
    loss_history = train["loss_history"]  # list of [step, loss]
    anneal_pts = [(s, l) for s, l in loss_history if s <= anneal_end_step]
    losses = [l for _, l in anneal_pts]
    increases = [losses[i + 1] - losses[i] for i in range(len(losses) - 1) if losses[i + 1] > losses[i]]
    h3 = dict(
        anneal_frac=anneal_frac,
        anneal_frac_source=anneal_frac_source,
        total_steps=total_steps,
        anneal_end_step=anneal_end_step,
        n_points_in_anneal_window=len(anneal_pts),
        loss_mean_anneal=float(np.mean(losses)) if losses else None,
        loss_std_anneal=float(np.std(losses)) if losses else None,
        loss_max_anneal=float(np.max(losses)) if losses else None,
        loss_final_pre_anneal_end=losses[-1] if losses else None,
        loss_final_overall=loss_history[-1][1] if loss_history else None,
        n_step_to_step_increases=len(increases),
        max_increase_delta=float(max(increases)) if increases else 0.0,
    )

    return dict(
        cell_tag=cell_tag, K=spec["K"], d=spec["d"], ratio=spec["ratio"], seed=seed,
        path=os.path.relpath(path, _REPO_ROOT),
        md5=md5_of(path),
        mismatches=mismatches,
        examples=examples,
        h3=h3,
        gpu_h=cell_json.get("gpu_h"),
        params=cell_json.get("params"),
    )


def pool_stat(values: list) -> dict:
    if not values:
        return dict(n=0)
    a = np.asarray(values, dtype=np.float64)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), max=float(a.max()))


def summarize(cells_out: list) -> dict:
    """Per-(K,d) aggregation, two ways: per-seed means (n=4, independent
    units) and pooled per-example (n=16, correlated within seed, descriptive
    only). Both reported -- no verdict is forced here."""
    by_kd = {}
    for c in cells_out:
        key = (c["K"], c["d"], c["ratio"])
        by_kd.setdefault(key, []).append(c)

    out = {}
    for (K, d, ratio), cells in sorted(by_kd.items()):
        tag = f"K{K}_d{d}_{ratio}"
        seed_means = dict(leak_ratio=[], D_share_of_leak=[], normB=[], normC=[], normD=[],
                           D_condition_number=[], D_eff_rank=[], phase_resid_max=[])
        pooled = dict(leak_ratio=[], D_share_of_leak=[], normB=[], normC=[], normD=[],
                      D_condition_number=[], D_eff_rank=[], phase_resid_max=[])
        loss_std_anneal = []
        loss_max_anneal = []
        n_increases = []
        for c in cells:
            for k in seed_means:
                per_ex = [ex[k] for ex in c["examples"]]
                seed_means[k].append(float(np.mean(per_ex)))
                pooled[k].extend(per_ex)
            loss_std_anneal.append(c["h3"]["loss_std_anneal"])
            loss_max_anneal.append(c["h3"]["loss_max_anneal"])
            n_increases.append(c["h3"]["n_step_to_step_increases"])
        out[tag] = dict(
            K=K, d=d, ratio=ratio, n_seeds=len(cells),
            spare_fraction_s=(d - K) / d,
            per_seed_mean=[
                {k: v for k, v in pool_stat(seed_means[k]).items()} | {"metric": k}
                for k in seed_means
            ],
            pooled_per_example={k: pool_stat(pooled[k]) for k in pooled},
            h3_per_seed=dict(
                loss_std_anneal=pool_stat(loss_std_anneal),
                loss_max_anneal=pool_stat(loss_max_anneal),
                n_step_to_step_increases=pool_stat(n_increases),
            ),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_HERE, "q3_mechanism_results.json"))
    args = ap.parse_args()

    all_mismatches = []
    cells_out = []
    input_md5 = {}
    for spec in CELLS:
        for seed in SEEDS:
            res = analyze_cell(spec, seed)
            cells_out.append(res)
            all_mismatches.extend(res["mismatches"])
            input_md5[res["path"]] = res["md5"]

    summary = summarize(cells_out)

    result = dict(
        script="matrix-thinking/ncr/analyze_dratio_blocks.py",
        design_doc="matrix-thinking/NCR_MAPPING_LAW_DESIGN.md",
        design_commit="d90abff",
        n_cells=len(cells_out),
        scalar_cross_verify=dict(
            n_mismatches=len(all_mismatches),
            tolerance=SCALAR_TOL,
            mismatches=all_mismatches,
        ),
        summary_by_Kd=summary,
        cells=cells_out,
        input_md5=input_md5,
    )

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    # ---- stdout report ----
    print("=" * 100)
    print("Q3 MECHANISM ANALYSIS -- d=K+1 vs d=2K, recomputed from z_dump.Z/z_ideal")
    print("=" * 100)
    print(f"cells analyzed: {len(cells_out)}  (16 expected: K16 d17 x4, K16 d32 x4, "
          f"K24 d25 x4, K24 d48 x4)")
    n_comparisons = len(cells_out) * 4 * 4  # 16 cells x 4 fields x 4 examples
    print(f"scalar cross-verify vs recorded deep_probe fields: "
          f"{len(all_mismatches)} mismatches over {n_comparisons} element-wise comparisons "
          f"(16 cells x 4 fields x 4 examples, tol={SCALAR_TOL})")
    if all_mismatches:
        for m in all_mismatches[:20]:
            print("  MISMATCH:", m)
    else:
        print("  ALL CLEAR -- analyze_zdump_arrays reproduces every recorded "
              "c_star/phase_resid_max/scale_corrected_residual/A_eff_rank scalar "
              "within tolerance, across all 16 cells (not just the design round's "
              "one spot-checked cell).")
    print()
    print("-" * 100)
    print("H1/H2 -- leakage magnitude and shape, per (K,d), two views")
    print("-" * 100)
    for tag, s in summary.items():
        print(f"\n{tag}  (K={s['K']} d={s['d']} {s['ratio']}  s=(d-K)/d={s['spare_fraction_s']:.4f}  "
              f"n_seeds={s['n_seeds']})")
        psm = {row["metric"]: row for row in s["per_seed_mean"]}
        for metric in ("leak_ratio", "D_share_of_leak", "normB", "normC", "normD",
                       "D_condition_number", "D_eff_rank", "phase_resid_max"):
            row = psm[metric]
            pooled = s["pooled_per_example"][metric]
            print(f"  {metric:<20} per-seed-mean(n={row['n']}): "
                  f"mean={row['mean']:.5f} std={row['std']:.5f} "
                  f"[{row['min']:.5f},{row['max']:.5f}]   "
                  f"| pooled-per-example(n={pooled['n']}): mean={pooled['mean']:.5f}")
    print()
    print("-" * 100)
    print("H3 -- loss_history during the anneal window (LOW-CONFIDENCE per design's own "
          "loss-invisibility precedent)")
    print("-" * 100)
    for tag, s in summary.items():
        h3 = s["h3_per_seed"]
        print(f"{tag}: loss_std_anneal per-seed mean={h3['loss_std_anneal']['mean']:.5f} "
              f"(n={h3['loss_std_anneal']['n']})  loss_max_anneal mean={h3['loss_max_anneal']['mean']:.5f}  "
              f"n_step_increases mean={h3['n_step_to_step_increases']['mean']:.2f}")
    print()
    print(f"Full results written to: {args.out}")
    print(f"Input file count: {len(input_md5)}  (md5s logged in output JSON's 'input_md5' block)")


if __name__ == "__main__":
    main()
