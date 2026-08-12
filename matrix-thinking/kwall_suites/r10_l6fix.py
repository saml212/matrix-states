from r10_vcheck import validity_check
from r10_payloads import row, disk, rep, PRIM
from fractions import Fraction as F
# Candidate A: all-ceiling variant (10 single-attempt CR pairs + 1 pair crashed
# twice + 1 GATE-REFUSED pair) -> realized 14.40, ccgh 14.40, frac 1.0000, 0 canonical
a = [row(K,s,"primary",1,"1.20","CRASHED-RECOVERED",True) for (K,s) in PRIM[:10]]
a += [row(30,2,"primary",1,"1.20","CRASHED-RECOVERED",True),
      row(30,2,"primary",2,"1.20","CRASHED-RECOVERED",True),
      row(30,3,"primary",1,"0.00","GATE-REFUSED",False)]
rA = rep("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", a, "14.40", "14.40", "1.00")
print("candidate A (14.40/14.40, frac 1.0000):", validity_check(rA, disk([]), "NEW") or "PASS []")
# Candidate B: A6's own composition relabelled -SUSPECT-OVERCHARGE (== A6')
b = [row(K,s,"primary",1,"1.20","CRASHED-RECOVERED",True) for (K,s) in PRIM[:9]]
b += [row(30,1,"primary",1,"1.20","CRASHED-RECOVERED",True),
      row(30,1,"primary",2,"1.20","CRASHED-RECOVERED",True),
      row(30,2,"primary",1,"1.00","COMPLETED",False),
      row(30,3,"primary",1,"0.00","GATE-REFUSED",False)]
rB = rep("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", b, "14.20", "13.20", str(F("13.2")/F("14.2")))
print("candidate B (14.20/13.20, frac 13.20/14.20):", validity_check(rB, disk([(30,2)]), "NEW") or "PASS []")
# Candidate C: keep a COMPLETED pair at the reachability cap 1.2210
c = [row(K,s,"primary",1,"1.20","CRASHED-RECOVERED",True) for (K,s) in PRIM[:10]]
c += [row(30,1,"primary",2,"1.20","CRASHED-RECOVERED",True),
      row(30,2,"primary",1,"1.2210","COMPLETED",False),
      row(30,3,"primary",1,"0.00","GATE-REFUSED",False)]
tot = F("1.20")*11 + F("1.2210")
rC = rep("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", c, str(tot), str(F("1.20")*11),
         str(F("1.20")*11/tot))
print("candidate C (realized %s, ccgh %s, frac %s):" % (float(tot), 13.2, float(F('1.20')*11/tot)),
      validity_check(rC, disk([(30,2)]), "NEW") or "PASS []")
