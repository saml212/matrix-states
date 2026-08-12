#!/usr/bin/env python3
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad")
from r9rev_vcheck import validity_check_NEW, validity_check_OLD
from r9rev_payloads import PAYLOADS

CORE_24_NAMES = [  # the payloads relevant to M1/m1's delta claim + regression baseline
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L7'",
    "A1", "A6", "A6'",
    "B1", "B1'", "B2", "B2'", "B3-OLD-STYLE", "B3-AMENDED", "B3-NEG", "B4",
]

print("=" * 78)
print("SUITE RUN 1 -- core payload set (delta-relevant + regression baseline)")
print("=" * 78)
mismatches = 0
old_verdicts = {}
new_verdicts = {}
for name in CORE_24_NAMES:
    p = PAYLOADS[name]
    old_pass, old_fail = validity_check_OLD(p)
    new_pass, new_fail = validity_check_NEW(p)
    old_v = "PASS" if old_pass else "FAIL"
    new_v = "PASS" if new_pass else "FAIL"
    old_verdicts[name] = old_v
    new_verdicts[name] = new_v
    exp_old_ok = (old_v == p["expect_OLD"])
    exp_new_ok = (new_v == p["expect_NEW"])
    flag = "" if (exp_old_ok and exp_new_ok) else "  <<< MISMATCH vs expectation"
    if flag:
        mismatches += 1
    old_fail_s = ",".join(old_fail)
    new_fail_s = ",".join(new_fail)
    print(f"{name:18s} OLD={old_v:4s} {old_fail_s:45s} "
          f"NEW={new_v:4s} {new_fail_s}{flag}")

print()
print(f"Expectation mismatches: {mismatches}/{len(CORE_24_NAMES)}")

print()
print("=" * 78)
print("DELTA TABLE (OLD verdict != NEW verdict)")
print("=" * 78)
flips = []
for name in CORE_24_NAMES:
    if old_verdicts[name] != new_verdicts[name]:
        flips.append(name)
        print(f"  {name:18s} {old_verdicts[name]} -> {new_verdicts[name]}")
print(f"\nTotal flips: {len(flips)}  (expected exactly 6: B1,B1',B2,B2',B3-NEG,B4)")
expected_flips = {"B1", "B1'", "B2", "B2'", "B3-NEG", "B4"}
print(f"Flip set matches expected exactly: {set(flips) == expected_flips}")
print(f"B3-AMENDED verdict OLD={old_verdicts['B3-AMENDED']} NEW={new_verdicts['B3-AMENDED']} "
      f"(non-flip, both PASS expected)")

print()
print("=" * 78)
print("L6 REBUILD CHECK (M1)")
print("=" * 78)
p = PAYLOADS["L6"]
new_pass, new_fail = validity_check_NEW(p)
print(f"L6 NEW verdict: {'PASS' if new_pass else 'FAIL'}  failures={new_fail}")
print(f"L6 declared fraction 12.00/14.00 = {12.00/14.00:.6f}")

print()
print("=" * 78)
print("A6/A6' ROUNDING CHECK (m4) -- exact quotient vs literal 0.9296")
print("=" * 78)
for name in ["A6", "A6-literal-0.9296", "A6'", "A6'-literal-0.9296"]:
    p = PAYLOADS[name]
    new_pass, new_fail = validity_check_NEW(p)
    print(f"{name:22s} NEW={'PASS' if new_pass else 'FAIL':4s} failures={new_fail}")

print()
print("=" * 78)
print("m3 VERIFICATION -- D2/D2' (K1-mirror bypass via conditional=null/launched:false)")
print("=" * 78)
for name in ["D2", "D2'"]:
    p = PAYLOADS[name]
    new_pass, new_fail = validity_check_NEW(p)
    old_pass, old_fail = validity_check_OLD(p)
    print(f"{name:6s} OLD={'PASS' if old_pass else 'FAIL':4s} {old_fail}   "
          f"NEW={'PASS' if new_pass else 'FAIL':4s} {new_fail}")

print()
print("=" * 78)
print("m7 VERIFICATION -- D1/D1' (COMPLETE strict branch ledger clause)")
print("=" * 78)
for name in ["D1", "D1'"]:
    p = PAYLOADS[name]
    new_pass, new_fail = validity_check_NEW(p)
    old_pass, old_fail = validity_check_OLD(p)
    print(f"{name:6s} OLD={'PASS' if old_pass else 'FAIL':4s} {old_fail}   "
          f"NEW={'PASS' if new_pass else 'FAIL':4s} {new_fail}")
