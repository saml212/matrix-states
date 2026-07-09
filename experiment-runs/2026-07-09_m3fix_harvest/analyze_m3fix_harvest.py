#!/usr/bin/env python3
"""S1.35/S1.36 M3 FIX-WAVE harvest analysis.

A3 config-match (steps == Rev-7 pins, target_padding + force_rank_k per the
S1.34 manifest, independent-literal cell list) + the C1 DECISIONAL table
(crosscheck_recovered_frac_90 / crosscheck_mean_cos, full-Q Procrustes) +
the disclosed-diagnostic scale-only primary + variant-B tax corroboration
(A2) + realized GPU-h vs the 1.3324 price + the +/-0.05 marginality screen.

Run from the harvest dir: python3 analyze_m3fix_harvest.py
"""
import json, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- independent literals (S1.34 manifest; NOT imported from the run code) ----
GROUPS = ["S3", "S4", "A5", "S5", "A6"]
D_MIN = {"S3": 2, "S4": 3, "A5": 3, "S5": 4, "A6": 5}
D_STATE = {g: D_MIN[g] + 2 for g in GROUPS}
STEP_PINS = {"S3": 8000, "S4": 20000, "A5": 20000, "S5": 8000, "A6": 40000}
AMBIENT_TAX = 2
RATE_GPU_H_PER_STEP = 2.5907 / 1_120_000  # measured Rev-7 rate (S1.34 pricing basis)
PRICE_REGISTERED = 1.3324

# expected manifest: cell_id -> (variant, group, arm, force_rank_k, target_padding, steps)
EXPECTED = {}
for g in GROUPS:
    # variant A (zero_pad): unconstrained anchor + full 3-point straddle
    EXPECTED[f"zero_pad__{g}__unconstrained__seed0"] = ("zero_pad", g, "unconstrained", None, "zero", STEP_PINS[g])
    for lab, k in (("k_dmin_minus_1", D_MIN[g]-1), ("k_dmin", D_MIN[g]), ("k_dmin_plus_1", D_MIN[g]+1)):
        EXPECTED[f"zero_pad__{g}__{lab}__seed0"] = ("zero_pad", g, lab, k, "zero", STEP_PINS[g])
    # variant B (tax_adjusted): 2-point grid, raw k = effective + tax
    for lab, k in (("k_dmin_minus_1", D_MIN[g]-1+AMBIENT_TAX), ("k_dmin", D_MIN[g]+AMBIENT_TAX)):
        EXPECTED[f"tax_adjusted__{g}__{lab}__seed0"] = ("tax_adjusted", g, lab, k, "eye", STEP_PINS[g])
assert len(EXPECTED) == 30

# ---- load + A3 verify ----
cells, problems = {}, []
files = sorted(glob.glob(os.path.join(HERE, "*.json")))
files = [f for f in files if not f.endswith("manifest.json")]
for f in files:
    d = json.load(open(f))
    cells[d["cell_id"]] = d

missing = set(EXPECTED) - set(cells)
extra = set(cells) - set(EXPECTED)
if missing: problems.append(f"MISSING cells: {sorted(missing)}")
if extra:   problems.append(f"EXTRA cells: {sorted(extra)}")

for cid, (variant, g, arm, k, pad, steps) in sorted(EXPECTED.items()):
    d = cells.get(cid)
    if d is None: continue
    checks = [
        ("variant", d.get("m3fix_variant"), variant),
        ("group", d.get("group"), g),
        ("arm", d.get("arm"), arm),
        ("force_rank_k", d.get("force_rank_k"), k),
        ("target_padding", d.get("target_padding"), pad),
        ("steps_completed", d.get("steps_completed"), steps),
        ("n_skipped_steps", d.get("n_skipped_steps"), 0),
        ("seed", d.get("seed"), 0),
    ]
    for name, got, want in checks:
        if got != want:
            problems.append(f"{cid}: {name} got {got!r} want {want!r}")

print("=" * 78)
print("A3 CONFIG-MATCH + VERIFY-VS-RAWS")
print("=" * 78)
print(f"cells found: {len(cells)}/30; independent-literal manifest match: "
      f"{'EXACT' if not (missing or extra) else 'MISMATCH'}")
if problems:
    print("PROBLEMS:")
    for p in problems: print("  -", p)
else:
    print("ALL 30 cells: steps == Rev-7 pins, force_rank_k + target_padding per "
          "manifest, n_skipped_steps == 0, seed == 0. CONFIG-MATCH CLEAN.")

# ---- realized cost ----
total_steps = sum(d["steps_completed"] for d in cells.values())
wall_s = sum(d["wall_clock_s"] for d in cells.values())
print(f"\nrealized: {total_steps:,} step-cells; wall-clock sum {wall_s/3600:.4f} h "
      f"(single GPU => {wall_s/3600:.4f} GPU-h) vs registered price {PRICE_REGISTERED}; "
      f"pricing-basis check {total_steps*RATE_GPU_H_PER_STEP:.4f}")

# ---- C1 decisional table (variant A) ----
def get(v, g, arm, key):
    cid = f"{v}__{g}__{arm}__seed0"
    return cells[cid][key] if cid in cells else float("nan")

ARMS_A = ["unconstrained", "k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"]
print("\n" + "=" * 78)
print("C1 DECISIONAL (variant A zero_pad): crosscheck_recovered_frac_90 [xrec90] "
      "/ crosscheck_mean_cos [xcos]")
print("=" * 78)
hdr = f"{'G':<4}{'d_min':<6}" + "".join(f"{a:<22}" for a in ["unconstr.", "k=d_min-1", "k=d_min", "k=d_min+1"])
print(hdr)
verdicts = {}
for g in GROUPS:
    row = f"{g:<4}{D_MIN[g]:<6}"
    vals = {}
    for a in ARMS_A:
        xr = get("zero_pad", g, a, "crosscheck_recovered_frac_90")
        xc = get("zero_pad", g, a, "crosscheck_mean_cos")
        vals[a] = (xr, xc)
        row += f"{xr:.3f} / {xc:.3f}      "
    print(row)
    verdicts[g] = vals

# ---- disclosed-diagnostic primary (variant A) ----
print("\nDISCLOSED-DIAGNOSTIC ONLY (scale-only primary rec90/mean_cos; expected "
      "to diverge per S1.35 oracle-injection):")
for g in GROUPS:
    row = f"{g:<4}{D_MIN[g]:<6}"
    for a in ARMS_A:
        r = get("zero_pad", g, a, "recovered_frac_90")
        c = get("zero_pad", g, a, "mean_cos")
        row += f"{r:.3f} / {c:.3f}      "
    print(row)

# ---- old-defect-signature check: sqrt(k/d_state) climb on DIRECT cosine ----
import math
print("\nOLD-DEFECT SIGNATURE CHECK (variant-A direct mean_cos vs sqrt(k/d_state) "
      "tax ceiling -- should NOT track it now that the tax is removed):")
for g in GROUPS:
    for a, k in (("k_dmin_minus_1", D_MIN[g]-1), ("k_dmin", D_MIN[g]), ("k_dmin_plus_1", D_MIN[g]+1)):
        pred = math.sqrt(k / D_STATE[g])
        xc = get("zero_pad", g, a, "crosscheck_mean_cos")
        print(f"  {g} k={k}: xcos={xc:.3f} vs old-tax-ceiling sqrt(k/d_state)={pred:.3f} "
              f"(delta {xc-pred:+.3f})")

# ---- restricted effective rank (context) ----
print("\nrestricted_effective_rank (variant A):")
for g in GROUPS:
    row = f"  {g}: "
    for a in ARMS_A:
        row += f"{a}={get('zero_pad', g, a, 'restricted_effective_rank'):.2f}  "
    print(row)

# ---- variant B (A2: tax-narrative corroboration only) ----
print("\n" + "=" * 78)
print("VARIANT B tax_adjusted (A2: corroboration only; k_dmin cell == unconstrained "
      "re-run by construction): xrec90 / xcos [direct mean_cos vs sqrt(k/d_state)]")
print("=" * 78)
for g in GROUPS:
    for lab, keff in (("k_dmin_minus_1", D_MIN[g]-1), ("k_dmin", D_MIN[g])):
        kraw = keff + AMBIENT_TAX
        xr = get("tax_adjusted", g, lab, "crosscheck_recovered_frac_90")
        xc = get("tax_adjusted", g, lab, "crosscheck_mean_cos")
        dc = get("tax_adjusted", g, lab, "mean_cos")
        pred = math.sqrt(min(kraw, D_STATE[g]) / D_STATE[g])
        print(f"  {g} {lab} (raw k={kraw}, eff rho-rank={keff}): {xr:.3f} / {xc:.3f} "
          f"[direct {dc:.3f} vs tax-ceiling {pred:.3f}]")

# ---- per-group razor-step verdict + marginality ----
print("\n" + "=" * 78)
print("PER-GROUP RAZOR-STEP READING (pre-registered S1.33/S1.35): k=d_min-1 FAILS "
      "(oracle-bounded <=0.894), k>=d_min RECOVERS (well above bar-class)")
print("=" * 78)
# bar-class reference from S1.33: failed cells sat at xrec90 = 0.000; recovered
# unconstrained cells sat at xrec90 0.48-0.73. The razor step = below-arm near
# fail-class AND k=d_min arm at/above the recovered class (>= 0.9x own anchor is
# the S1.33 sufficiency framing; we report both the step size and anchor ratio).
MARGIN = 0.05
marginal = []
n_confirm = 0
for g in GROUPS:
    v = verdicts[g]
    anchor_r, anchor_c = v["unconstrained"]
    below_r, below_c = v["k_dmin_minus_1"]
    at_r, at_c = v["k_dmin"]
    above_r, above_c = v["k_dmin_plus_1"]
    recovers_at = at_r >= 0.9 * anchor_r
    recovers_above = above_r >= 0.9 * anchor_r
    fails_below = below_r < 0.9 * anchor_r and below_c <= 0.894
    step = at_r - below_r
    ok = fails_below and recovers_at
    n_confirm += ok
    # marginality: any decisive cell within +/-0.05 of its own boundary
    marg_flags = []
    if abs(at_r - 0.9 * anchor_r) <= MARGIN: marg_flags.append(f"k_dmin xrec90 {at_r:.3f} vs 0.9*anchor {0.9*anchor_r:.3f}")
    if abs(below_c - 0.894) <= MARGIN: marg_flags.append(f"k_dmin-1 xcos {below_c:.3f} vs oracle bound 0.894")
    if abs(below_r - 0.9 * anchor_r) <= MARGIN: marg_flags.append(f"k_dmin-1 xrec90 {below_r:.3f} vs 0.9*anchor {0.9*anchor_r:.3f}")
    if marg_flags: marginal.append((g, marg_flags))
    print(f"{g}: below FAILS={fails_below} (xrec90 {below_r:.3f}, xcos {below_c:.3f}) | "
          f"k=d_min RECOVERS={recovers_at} (xrec90 {at_r:.3f} vs 0.9x-anchor {0.9*anchor_r:.3f}) | "
          f"k=d_min+1 RECOVERS={recovers_above} ({above_r:.3f}) | razor step {step:+.3f} "
          f"=> {'RAZOR-STEP CONFIRM' if ok else 'no-confirm'}")

print(f"\nrazor step present in {n_confirm}/5 groups")
print("marquee-pair coverage (S4 or A5 must be among confirms):",
      any(g in ("S4", "A5") and (verdicts[g]['k_dmin'][0] >= 0.9*verdicts[g]['unconstrained'][0]
          and verdicts[g]['k_dmin_minus_1'][0] < 0.9*verdicts[g]['unconstrained'][0]) for g in ("S4","A5")))
if marginal:
    print("\nMARGINALITY TRIGGER (+/-0.05 of a decision boundary) FIRED:")
    for g, flags in marginal:
        for fl in flags: print(f"  {g}: {fl}")
else:
    print("\nNo decisive cell within +/-0.05 of a decision boundary.")
