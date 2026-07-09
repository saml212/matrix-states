#!/usr/bin/env python3
"""S1.36a M3FIX S3 SEED-EXTENSION harvest analysis.

Verifies the 18 new seed{1,2,3} x S3-only cells (4 variant-A + 2 variant-B
per seed) against an independent-literal manifest, then answers §1.36's
routed S3 question: does k=d_min recover to >= the 0.495 pre-registered
bar (0.9 * seed=0's own anchor xrec90) at the SEED-MEAN across all 4
seeds {0 (original),1,2,3 (extension)}, with the per-seed values disclosed?

Usage: python3 analyze_m3fix_s3ext_harvest.py <seed0_dir> <ext_dir>
  seed0_dir: experiment-runs/2026-07-09_m3fix_harvest/ (the original S3 seed=0 cells)
  ext_dir:   the pulled results_m3fix_s3ext/ (18 new JSONs)
"""
import json, glob, os, sys

GROUP = "S3"
D_MIN = 2
STEP_PIN = 8000
AMBIENT_TAX = 2
RATE_GPU_H_PER_STEP = 2.5907 / 1_120_000  # measured Rev-7 rate (S1.34 pricing basis, unchanged)
BAR_LITERAL = 0.495   # pre-registered in §1.36: 0.9 * seed=0's own zero_pad-anchor xrec90 (0.550)

def expected_manifest(seed):
    exp = {}
    exp[f"zero_pad__{GROUP}__unconstrained__seed{seed}"] = ("zero_pad", "unconstrained", None, "zero", STEP_PIN, seed)
    for lab, k in (("k_dmin_minus_1", D_MIN - 1), ("k_dmin", D_MIN), ("k_dmin_plus_1", D_MIN + 1)):
        exp[f"zero_pad__{GROUP}__{lab}__seed{seed}"] = ("zero_pad", lab, k, "zero", STEP_PIN, seed)
    for lab, k in (("k_dmin_minus_1", D_MIN - 1 + AMBIENT_TAX), ("k_dmin", D_MIN + AMBIENT_TAX)):
        exp[f"tax_adjusted__{GROUP}__{lab}__seed{seed}"] = ("tax_adjusted", lab, k, "eye", STEP_PIN, seed)
    return exp

def main():
    seed0_dir, ext_dir = sys.argv[1], sys.argv[2]

    # ---- load all 4 seeds' worth of S3 cells ----
    cells = {}
    for f in glob.glob(os.path.join(seed0_dir, "*S3*seed0.json")):
        d = json.load(open(f))
        cells[d["cell_id"]] = d
    for f in glob.glob(os.path.join(ext_dir, "*.json")):
        d = json.load(open(f))
        cells[d["cell_id"]] = d

    # ---- A3 config-match: independent-literal manifest across all 4 seeds ----
    expected = {}
    for seed in (0, 1, 2, 3):
        expected.update(expected_manifest(seed))
    assert len(expected) == 24, f"expected 24 cells (4 seeds x 6), literal construction error: {len(expected)}"

    missing = set(expected) - set(cells)
    extra = set(cells) - set(expected)
    problems = []
    if missing: problems.append(f"MISSING cells: {sorted(missing)}")
    if extra:   problems.append(f"EXTRA cells: {sorted(extra)}")
    for cid, (variant, arm, k, pad, steps, seed) in sorted(expected.items()):
        d = cells.get(cid)
        if d is None: continue
        checks = [
            ("variant", d.get("m3fix_variant"), variant),
            ("group", d.get("group"), GROUP),
            ("arm", d.get("arm"), arm),
            ("force_rank_k", d.get("force_rank_k"), k),
            ("target_padding", d.get("target_padding"), pad),
            ("steps_completed", d.get("steps_completed"), steps),
            ("n_skipped_steps", d.get("n_skipped_steps"), 0),
            ("seed", d.get("seed"), seed),
        ]
        for name, got, want in checks:
            if got != want:
                problems.append(f"{cid}: {name} got {got!r} want {want!r}")

    print("=" * 84)
    print("A3 CONFIG-MATCH + VERIFY-VS-RAWS (S3, seeds 0/1/2/3, 24 cells total)")
    print("=" * 84)
    print(f"cells found: {len(cells)}/24; independent-literal manifest match: "
          f"{'EXACT' if not (missing or extra) else 'MISMATCH'}")
    if problems:
        print("PROBLEMS:")
        for p in problems: print("  -", p)
    else:
        print("ALL 24 cells: steps==8000, force_rank_k + target_padding per manifest, "
              "n_skipped_steps==0, seed matches its own file. CONFIG-MATCH CLEAN.")

    # ---- realized cost (extension only: seeds 1,2,3) ----
    ext_cells = {cid: d for cid, d in cells.items() if d["seed"] in (1, 2, 3)}
    total_steps = sum(d["steps_completed"] for d in ext_cells.values())
    wall_s = sum(d["wall_clock_s"] for d in ext_cells.values())
    print(f"\nEXTENSION realized (seeds 1,2,3 only): {total_steps:,} step-cells, "
          f"wall-clock sum {wall_s/3600:.4f} GPU-h vs priced "
          f"{total_steps*RATE_GPU_H_PER_STEP:.4f} GPU-h")

    # ---- per-seed C1 table (zero_pad variant) ----
    def get(seed, arm, key):
        cid = f"zero_pad__{GROUP}__{arm}__seed{seed}"
        return cells[cid][key] if cid in cells else float("nan")

    ARMS = ["unconstrained", "k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"]
    print("\n" + "=" * 84)
    print("PER-SEED C1 DECISIONAL (S3, variant A zero_pad): xrec90 [xcos]")
    print("=" * 84)
    print(f"{'seed':<6}" + "".join(f"{a:<24}" for a in ["unconstr.", "k=d_min-1", "k=d_min", "k=d_min+1"]))
    k_dmin_vals = []
    for seed in (0, 1, 2, 3):
        row = f"{seed:<6}"
        for a in ARMS:
            xr = get(seed, a, "crosscheck_recovered_frac_90")
            xc = get(seed, a, "crosscheck_mean_cos")
            row += f"{xr:.3f} [{xc:.3f}]           "
        print(row)
        k_dmin_vals.append(get(seed, "k_dmin", "crosscheck_recovered_frac_90"))

    seed_mean_all4 = sum(k_dmin_vals) / len(k_dmin_vals)
    seed_mean_ext3 = sum(k_dmin_vals[1:]) / 3
    print(f"\nk=d_min xrec90 per seed: {[round(v,3) for v in k_dmin_vals]}")
    print(f"seed-mean (all 4: 0,1,2,3): {seed_mean_all4:.4f}")
    print(f"seed-mean (extension only: 1,2,3): {seed_mean_ext3:.4f}")
    print(f"pre-registered bar (§1.36, 0.9 x seed=0 anchor): {BAR_LITERAL}")

    verdict = "S3 CONFIRMED" if seed_mean_all4 >= BAR_LITERAL else "S3 DEMOTED-to-disclosed"
    print(f"\n{'='*84}\n§1.36a VERDICT: {verdict} "
          f"(seed-mean {seed_mean_all4:.4f} {'>=' if seed_mean_all4 >= BAR_LITERAL else '<'} bar {BAR_LITERAL})\n{'='*84}")

    # ---- variant B disclosure (tax_adjusted, corroboration only, A2) ----
    print("\nVARIANT B (tax_adjusted, corroboration only) per seed:")
    for seed in (0, 1, 2, 3):
        for lab in ("k_dmin_minus_1", "k_dmin"):
            cid = f"tax_adjusted__{GROUP}__{lab}__seed{seed}"
            if cid in cells:
                d = cells[cid]
                print(f"  seed={seed} {lab}: xrec90={d['crosscheck_recovered_frac_90']:.3f} "
                      f"[xcos={d['crosscheck_mean_cos']:.3f}]")

    # ---- health disclosure (gate1a) ----
    print("\nGATE1A (min L in {2..5} cos >= 0.9+0.02margin) per seed, zero_pad anchor + k_dmin:")
    for seed in (0, 1, 2, 3):
        for a in ("unconstrained", "k_dmin"):
            cid = f"zero_pad__{GROUP}__{a}__seed{seed}"
            if cid in cells:
                g1a = cells[cid].get("gate1a", {})
                print(f"  seed={seed} {a}: min_val={g1a.get('min_val'):.4f} clears={g1a.get('clears')}")

if __name__ == "__main__":
    main()
