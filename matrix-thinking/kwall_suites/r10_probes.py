from r10_vcheck import validity_check
from r10_payloads import row, disk, rep, PRIM
from fractions import Fraction as F

P = {}
# D1 / D1' — m7 probes (COMPLETE strict + vacuous ledger)
gr12 = [row(K, s, "primary", 1, "0.00", "GATE-REFUSED", False) for (K, s) in PRIM]
P["D1"]  = (rep("COMPLETE", gr12, "0.00", "0.00", "0.00"), disk(PRIM))
P["D1'"] = (rep("COMPLETE", [],   "0.00", "0.00", "0.00"), disk(PRIM))
# D2 / D2' — m3 probes (4 conditional canonical files, report claims non-dispatch)
att = [row(K, s, "primary", 1, "1.00", "COMPLETED", False) for (K, s) in PRIM]
P["D2"]  = (rep("COMPLETE", att, "12.00", "0.00", "0.00", conditional=None),
            disk(PRIM, n_cond=4))
P["D2'"] = (rep("COMPLETE", att, "12.00", "0.00", "0.00",
                conditional={"launched": False, "per_seed": [], "qualifier_band": None}),
            disk(PRIM, n_cond=4))
for n, (r, d) in P.items():
    fo, fn = validity_check(r, d, "OLD"), validity_check(r, d, "NEW")
    print("%-5s OLD=%s NEW=%s  %s" % (n, "PASS" if not fo else "FAIL",
                                      "PASS" if not fn else "FAIL", fn))
print()
print("2.84/14.20 ==", F("2.84")/F("14.20"), " exactly 0.20:", F("2.84")/F("14.20") == F("0.20"))
print("12.00/14.00 ==", float(F("12")/F("14")), "-> 4dp", round(float(F("12")/F("14")), 4))
print("13.20/14.20 ==", float(F("13.2")/F("14.2")), "-> 4dp", round(float(F("13.2")/F("14.2")), 4))
print("|0.8571 - 12/14| =", float(abs(F("0.8571") - F("12")/F("14"))), "> 1e-6:",
      abs(F("0.8571") - F("12")/F("14")) > F(1,1000000))
print("primary per-attempt reachability: ceiling 1.20 + tau 0.0157 =", 1.20+0.0157,
      "; +startup s 0.0053 =", 1.20+0.0157+0.0053, "; L6 states 2.00")
