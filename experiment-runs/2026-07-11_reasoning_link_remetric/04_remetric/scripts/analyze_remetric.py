"""Aggregate results/reasoning_link_remetric/*.json (the FIXED-instrument
re-run) into the same summary shape REASONING_LINK_DESIGN.md sec 15.3/15.4
and sec 16.11 report, for direct before/after comparison. Run on the box
(no GPU needed, pure JSON aggregation) or copy the JSONs locally first.
"""
import glob
import json
import os
import sys
import statistics

DIR = sys.argv[1] if len(sys.argv) > 1 else "results/reasoning_link_remetric"

files = sorted(glob.glob(os.path.join(DIR, "*.json")))
files = [f for f in files if not f.endswith("REASONING_LINK_REMETRIC_DONE")]

leg_a = []
leg_b = []
for f in files:
    d = json.load(open(f))
    name = os.path.basename(f)
    if name.startswith("leg_a_"):
        leg_a.append((name, d))
    elif name.startswith("leg_b_"):
        leg_b.append((name, d))

print(f"Loaded {len(leg_a)} Leg-A cells, {len(leg_b)} Leg-B cells ({len(files)} total files)")
print()


def per_h_recovered(d):
    # per_h dict keyed by h (string or int) -> result dict with recovered_frac
    out = {}
    per_h = d.get("per_h", {})
    for h, r in per_h.items():
        out[str(h)] = r.get("recovered_frac")
    return out


def per_h_cos_mean(d):
    out = {}
    for h, r in d.get("per_h", {}).items():
        out[str(h)] = r.get("cos_mean")
    return out


print("=" * 100)
print("LEG A -- per-cell recovered_frac@0.9 by h (fixed instrument)")
print("=" * 100)
n_nonzero_cells_a = 0
n_total_readings_a = 0
n_nonzero_readings_a = 0
all_h1_a = []
for name, d in leg_a:
    ph = per_h_recovered(d)
    vals = [ph.get(str(h)) for h in (1, 2, 3, 4)]
    if any(v not in (None,) and v > 0 for v in vals):
        n_nonzero_cells_a += 1
    for v in vals:
        if v is not None:
            n_total_readings_a += 1
            if v > 0:
                n_nonzero_readings_a += 1
    if ph.get("1") is not None:
        all_h1_a.append(ph["1"])
    print(f"  {name}: h1={ph.get('1')} h2={ph.get('2')} h3={ph.get('3')} h4={ph.get('4')}")

print()
print(f"LEG A: {n_nonzero_cells_a}/{len(leg_a)} cells have >=1 nonzero h reading; "
      f"{n_nonzero_readings_a}/{n_total_readings_a} (cell,h) readings nonzero")
if all_h1_a:
    print(f"LEG A h=1 recovered_frac: mean={statistics.mean(all_h1_a):.4f} "
          f"min={min(all_h1_a):.4f} max={max(all_h1_a):.4f}")

print()
print("=" * 100)
print("LEG B -- per-cell recovered_frac@0.9 by h (fixed instrument)")
print("=" * 100)
n_nonzero_cells_b = 0
n_total_readings_b = 0
n_nonzero_readings_b = 0
all_h1_b = []
for name, d in leg_b:
    ph = per_h_recovered(d)
    vals = [ph.get(str(h)) for h in (1, 2, 3, 4)]
    if any(v not in (None,) and v > 0 for v in vals):
        n_nonzero_cells_b += 1
    for v in vals:
        if v is not None:
            n_total_readings_b += 1
            if v > 0:
                n_nonzero_readings_b += 1
    if ph.get("1") is not None:
        all_h1_b.append(ph["1"])
    print(f"  {name}: h1={ph.get('1')} h2={ph.get('2')} h3={ph.get('3')} h4={ph.get('4')}")

print()
print(f"LEG B: {n_nonzero_cells_b}/{len(leg_b)} cells have >=1 nonzero h reading; "
      f"{n_nonzero_readings_b}/{n_total_readings_b} (cell,h) readings nonzero")
if all_h1_b:
    print(f"LEG B h=1 recovered_frac: mean={statistics.mean(all_h1_b):.4f} "
          f"min={min(all_h1_b):.4f} max={max(all_h1_b):.4f}")

print()
print("=" * 100)
total_cells = len(leg_a) + len(leg_b)
total_readings = n_total_readings_a + n_total_readings_b
total_nonzero = n_nonzero_readings_a + n_nonzero_readings_b
print(f"GRAND TOTAL: {total_nonzero}/{total_readings} (cell,h) readings nonzero across "
      f"{total_cells} cells (vs the pre-fix 0/320 baseline, sec 15/16.11)")

# premise gates, if present
print()
print("=" * 100)
print("Premise (iii)/(iv) pass rates (h1_confirmatory-relevant gates), fixed instrument")
print("=" * 100)
n_iii_pass = n_iv_pass = n_prem_total = 0
n_h1_confirmatory = n_hge2_confirmatory = 0
for name, d in leg_a + leg_b:
    ph = d.get("per_h", {})
    for h, r in ph.items():
        if "premise_iii_pass" in r or "premise_iv_pass" in r:
            n_prem_total += 1
            if r.get("premise_iii_pass"):
                n_iii_pass += 1
            if r.get("premise_iv_pass"):
                n_iv_pass += 1
            if r.get("h1_confirmatory"):
                n_h1_confirmatory += 1
            if r.get("h_ge2_confirmatory"):
                n_hge2_confirmatory += 1
print(f"premise iii pass: {n_iii_pass}/{n_prem_total}; premise iv pass: {n_iv_pass}/{n_prem_total}")
print(f"h1_confirmatory=True: {n_h1_confirmatory}/{n_prem_total}; "
      f"h_ge2_confirmatory=True: {n_hge2_confirmatory}/{n_prem_total}")

print()
print("=" * 100)
print("cos_mean distribution by h (the fixed instrument's real-valued signal, vs the")
print("pre-fix ~0.0 uniform baseline) -- diagnostic, not a decision-rule input")
print("=" * 100)
cos_by_h = {"1": [], "2": [], "3": [], "4": []}
for name, d in leg_a + leg_b:
    for h, v in per_h_cos_mean(d).items():
        if v is not None and h in cos_by_h:
            cos_by_h[h].append(v)
for h in ("1", "2", "3", "4"):
    vals = cos_by_h[h]
    if vals:
        print(f"h={h}: n={len(vals)} mean={statistics.mean(vals):.4f} "
              f"min={min(vals):.4f} max={max(vals):.4f} "
              f"stdev={statistics.pstdev(vals):.4f}")

# state condition number, for completeness (diagnostic, pre-registered in original harvest too)
print()
cond_vals = []
for name, d in leg_a + leg_b:
    for h, r in d.get("per_h", {}).items():
        v = r.get("state_condition_number_mean")
        if v is not None:
            cond_vals.append(v)
if cond_vals:
    print(f"state_condition_number_mean across all readings: mean={statistics.mean(cond_vals):.2f} "
          f"median={statistics.median(cond_vals):.2f} min={min(cond_vals):.2f} max={max(cond_vals):.2f}")
