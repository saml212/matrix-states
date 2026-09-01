#!/usr/bin/env python3
"""1.31B HARVEST AGGREGATOR -- every pinned test in NCR_SCALE_AXIS_DESIGN.md R2 §5/§6.

kappa = (acc - 1/K)/(1 - 1/K)   (design line 117, M2 election)
Aggregator = MEDIAN everywhere (§5.1); drift = median of per-seed differences
(§5.1 m1), with difference-of-medians reported alongside in every drift table.
Verdicts are printed with their pinned thresholds; nothing is elected here.

Inputs (all raw JSONs, no transcription):
  392M depth  : depthext6_392m_*_depthext.json          (6 rungs, this wave)
  392M battery: {sweep,calib}_scaleaxis392m_*_kscaling.json
  98M  depth  : depthext6_*_depthext.json               (6-rung re-score of record)
  98M  battery: sweep_kscaling_K16_*, anchor_mob_g3b31_*, k32_wave0_*,
                frontier_kscaling_K40_*                  (§2.1's named sources)
Seed-31337 re-measures are EXCLUDED from every reference/verdict (§2.1).
"""
import glob
import itertools
import json
import os
import statistics
import sys
from collections import defaultdict

KS = (16, 24, 32, 40)
SQ6 = (5, 7, 9, 11, 13, 15)
DELTA = 0.05            # §5.2 breadth margin (Curves 1, 4)
DELTA_DEPTH = 0.095     # §5.2 Rule R-δ OUTPUT, elected on the 98M six-rung re-score
S_STAR = 13             # §5.2 Rule R-δ elected readout depth
CAP_BAR = 0.90          # §6.1 Curve 1 capability bar
DRIFT_TOL = 0.05        # §6.1 Curve 5a


def kappa(acc, k):
    p = 1.0 / k
    return (acc - p) / (1.0 - p)


def med(xs):
    return statistics.median(xs)


# ---------------------------------------------------------------- exact null
def stratum_U(a, b):
    """# of the 9 cross-condition pairs where a-side exceeds b-side, ties 1/2."""
    return sum(1.0 if x > y else (0.5 if x == y else 0.0) for x in a for y in b)


def exact_null(strata_vals):
    dist = {0.0: 1.0}
    for vals in strata_vals:
        sub = defaultdict(float)
        idx = range(len(vals))
        combos = list(itertools.combinations(idx, len(vals) // 2))
        for c in combos:
            f = [vals[i] for i in c]
            t = [vals[i] for i in idx if i not in c]
            sub[stratum_U(f, t)] += 1.0 / len(combos)
        new = defaultdict(float)
        for x, px in dist.items():
            for y, py in sub.items():
                new[x + y] += px * py
        dist = dict(new)
    return dist


def two_sided_p(dist, T):
    hi = sum(pr for t, pr in dist.items() if t >= T - 1e-9)
    lo = sum(pr for t, pr in dist.items() if t <= T + 1e-9)
    return min(1.0, 2 * min(hi, lo)), hi


def exact_threshold(S):
    """Smallest T with one-sided P(T'>=T) < 0.005 -- the §5.3 construction."""
    dist = exact_null([[1, 1, 1, 0, 0, 0]] * S)   # placeholder shape only
    # build the true null from the canonical 3v3 U-distribution
    base = {u: c / 20.0 for u, c in zip(range(10), (1, 1, 2, 3, 3, 3, 3, 2, 1, 1))}
    dist = {0.0: 1.0}
    for _ in range(S):
        new = defaultdict(float)
        for x, px in dist.items():
            for u, pu in base.items():
                new[x + u] += px * pu
        dist = dict(new)
    ts = sorted(dist)
    for t in ts:
        if sum(pr for tt, pr in dist.items() if tt >= t - 1e-9) < 0.005:
            return t, dist
    return None, dist


# ---------------------------------------------------------------- loading
def load_depth(pattern, scale_label):
    out = {}
    for p in glob.glob(pattern):
        r = json.load(open(p))
        if r["K"] not in KS:
            continue
        if int(r.get("base_seed", 90210)) != 90210:
            continue
        key = (r["K"], bool(r["freeze_entity_adapter"]), int(r["ckpt_seed"]))
        kk = {e["n_squarings"]: kappa(e["acc"], r["K"])
              for e in r["matched"]["P1b"]["per_hop"].values()}
        raw = {e["n_squarings"]: e["acc"] for e in r["matched"]["P1b"]["per_hop"].values()}
        p0 = {e["n_squarings"]: e["acc"] for e in r["matched"]["P0"]["per_hop"].values()}
        out[key] = dict(kappa=kk, acc=raw, p0=p0, rec=r, scale=scale_label)
    return out


def load_battery(patterns, scale_label):
    out = {}
    for pat in patterns:
        for p in glob.glob(pat):
            base = os.path.basename(p)
            if base.startswith("remeasure_"):
                continue
            r = json.load(open(p))
            # battery records carry K under `kscaling` provenance, not top level
            rk = r.get("K") if r.get("K") is not None else r["kscaling"]["K"]
            if rk not in KS or int(r.get("base_seed", 90210)) != 90210:
                continue
            r["K"] = rk
            key = (rk, bool(r["freeze_entity_adapter"]), int(r["ckpt_seed"]))
            ph = r["matched"]["P1b"]["per_hop"]
            htop = next(e for e in ph.values() if e["role"] == "ladder_top")
            hfix = next(e for e in ph.values() if e["role"] == "fixed_dist_control")
            out[key] = dict(
                htop_acc=htop["acc"], htop_kappa=kappa(htop["acc"], rk), htop_h=htop["h"],
                hfix_acc=hfix["acc"], hfix_kappa=kappa(hfix["acc"], rk), hfix_h=hfix["h"],
                p0=[e["acc"] for e in r["matched"]["P0"]["per_hop"].values()],
                p0_hops={e["h"]: e["acc"] for e in r["matched"]["P0"]["per_hop"].values()},
                p0_roles={e["h"]: e["role"] for e in r["matched"]["P0"]["per_hop"].values()},
                band=r["wall_band"], chance=r["chance"], rec=r, scale=scale_label)
    return out


def cell(d, k, fz, field):
    return [d[(k, fz, s)][field] for s in (0, 1, 2) if (k, fz, s) in d]


def hdr(t):
    print("\n" + "=" * 112)
    print(t)
    print("=" * 112)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    # NAMING: in this file the "1310M" slots hold the 1.31B (NEW) side and the
    # "98M" slots hold the 392M (REFERENCE) side, so every pairwise test below
    # is the (392M, 1.31B) pair that Ruling 2 pins -- same instruments, same
    # thresholds, one scale-pair further along. The 98M-vs-392M results are NOT
    # recomputed here; they stand from #21 and are cited.
    D392 = load_depth(os.path.join(d, "depthext6_1310m_*_depthext.json"), "1310M")
    D98 = load_depth(os.path.join(d, "ref392m_depth", "depthext6_392m_*_depthext.json"), "1310M")
    B392 = load_battery([os.path.join(d, "sweep1b_*_kscaling.json"),
                         os.path.join(d, "calib1b_*_kscaling.json")], "1310M")
    B98 = load_battery([os.path.join(d, "ref392m_battery", "*scaleaxis392m_*_kscaling.json")], "1310M")

    print(f"loaded: 1310M depth {len(D392)}, 392M depth {len(D98)}, "
          f"1310M battery {len(B392)}, 392M battery {len(B98)}  (expect 24 each)")
    bad = [k for k, v in D392.items() if v["rec"]["self_check"] != "PASS"
           or v["rec"]["ckpt_step"] != 20000]
    print("1310M depth invalid:", bad if bad else "none")
    print("1310M ladders:", {k: v["rec"]["depth_ladder"] for k, v in
                            sorted(D392.items()) if k[1] and k[2] == 0})
    for label, dd in (("1310M", D392), ("1310M", D98)):
        miss = [(k, f, s) for k in KS for f in (True, False) for s in (0, 1, 2)
                if (k, f, s) not in dd]
        if miss:
            print(f"!!! MISSING {label} depth cells: {miss}")

    # ============================ §6.1 Curve 1 ============================
    hdr("§6.1 CURVE 1 -- CAPABILITY at h_top (5 squarings), P1b, 1.31B. Bar: kappa >= 0.90 on >=2/3 seeds")
    print(f"{'K':>3s} {'h_top':>6s} {'frozen kappa (s0,s1,s2)':>34s} {'med':>8s} {'>=2/3':>6s}   "
          f"{'trainable kappa':>30s} {'med':>8s}")
    cap = {}
    for k in KS:
        fk = sorted(cell(B392, k, True, "htop_kappa"))
        tk = sorted(cell(B392, k, False, "htop_kappa"))
        n_ok = sum(1 for x in fk if x >= CAP_BAR)
        cap[k] = n_ok >= 2
        h = B392[(k, True, 0)]["htop_h"]
        print(f"{k:>3d} {h:>6d} {str([round(x,4) for x in fk]):>34s} {med(fk):8.4f} "
              f"{str(n_ok)+'/3':>6s}   {str([round(x,4) for x in tk]):>30s} {med(tk):8.4f}")
    print(f"\nCURVE 1 VERDICT: {'CAPABILITY-HOLDS (curve) -- holds at all four K' if all(cap.values()) else 'FRONTIER-AT-K=' + str(min(k for k in KS if not cap[k]))}")

    # ============================ §6.1 Curve 4 ============================
    hdr("§6.1 CURVE 4 -- BREADTH-vs-DEPTH (h_fix control, effective distance 4 at 5 squarings)")
    print(f"{'K':>3s} {'h_fix':>6s} {'frozen kappa med':>17s} {'trainable kappa med':>20s}")
    ft, tt, fx, tx = [], [], [], []
    for k in KS:
        a = med(cell(B392, k, True, "hfix_kappa")); b = med(cell(B392, k, False, "hfix_kappa"))
        fx.append(a); tx.append(b)
        ft.append(med(cell(B392, k, True, "htop_kappa")))
        tt.append(med(cell(B392, k, False, "htop_kappa")))
        print(f"{k:>3d} {B392[(k,True,0)]['hfix_h']:>6d} {a:17.4f} {b:20.4f}")
    def label4(r_fix, r_top):
        if r_fix <= DELTA and r_top <= DELTA:
            return "BOTH-FLAT (neither declines)"
        if r_fix <= DELTA and r_top > DELTA:
            return "DEPTH-DRIVEN (h_fix flat, h_top declines)"
        return "BREADTH-DRIVEN (h_fix declines comparably to h_top)"
    r_fix_f, r_top_f = max(fx) - min(fx), max(ft) - min(ft)
    r_fix_t, r_top_t = max(tx) - min(tx), max(tt) - min(tt)
    print(f"\nrange over K   frozen:    h_fix {r_fix_f:.4f}  h_top {r_top_f:.4f}  -> {label4(r_fix_f, r_top_f)}")
    print(f"range over K   trainable: h_fix {r_fix_t:.4f}  h_top {r_top_t:.4f}  -> {label4(r_fix_t, r_top_t)}")
    if label4(r_fix_f, r_top_f) == label4(r_fix_t, r_top_t):
        print(f"CURVE 4 VERDICT: {label4(r_fix_f, r_top_f)} (both arms agree)")
    else:
        print(f"CURVE 4 VERDICT: ARM-DEPENDENT -- frozen {label4(r_fix_f, r_top_f)}; "
              f"trainable {label4(r_fix_t, r_top_t)}")
        print("!!! PRE-REGISTRATION AMBIGUITY: KSCALING §7.4 (carried verbatim by §6.1 Curve 4)")
        print("    defines the three labels on 'kappa' without naming an arm, and the two arms")
        print("    return DIFFERENT labels here. Reported per-arm rather than elected. FLAGGED.")

    # ============================ §6.1 Curve 2 ============================
    hdr("§6.1 CURVE 2 -- WALL (P0, all 10 hops x 6 cells per K), per-K band 1/K +/- 3sd")
    print(f"{'K':>3s} {'band':>20s} {'n readings':>11s} {'max':>8s} {'# above band':>13s}  offending cells (hop)")
    wall = {}
    for k in KS:
        allv, over = [], []
        for fz in (True, False):
            for s in (0, 1, 2):
                c = B392.get((k, fz, s))
                if not c:
                    continue
                for h, a in c["p0_hops"].items():
                    allv.append(a)
                    if a > c["band"][1] + 1e-12:
                        over.append((("frozen" if fz else "trainable"), s, h,
                                     c["p0_roles"][h], round(a, 4)))
        wall[k] = over
        b = B392[(k, True, 0)]["band"]
        print(f"{k:>3d} [{b[0]:.4f},{b[1]:.4f}]{'':>4s} {len(allv):>11d} {max(allv):8.4f} "
              f"{len(over):>13d}  {over if over else ''}")
    print("\nNote: a SINGLE-seed excursion is re-measured at base seed 31337 before it is called a")
    print("breach (§6.1 Curve 2 / KSCALING §7.2). Replication across >=2 seeds is required.")

    # ============================ §6.1 Curve 5a ===========================
    hdr("§6.1 CURVE 5a -- DEPTH DRIFT at 392M (median of per-seed kappa@11sq - kappa@5sq); "
        "band: within +/-0.05 of the 392M K=24 value")
    print(f"{'recipe':>10s} {'K':>3s} " + " ".join(f"{'@'+str(s)+'sq':>9s}" for s in SQ6)
          + f" {'drift(11-5)':>12s} {'diff-of-med':>12s} {'vs K=24':>9s}")
    drift = {}
    for fz, lab in ((True, "frozen"), (False, "trainable")):
        for k in KS:
            per = {s: [D392[(k, fz, sd)]["kappa"][s] for sd in (0, 1, 2)] for s in SQ6}
            drift[(k, fz)] = med([D392[(k, fz, sd)]["kappa"][11] - D392[(k, fz, sd)]["kappa"][5]
                                  for sd in (0, 1, 2)])
        for k in KS:
            per = {s: med([D392[(k, fz, sd)]["kappa"][s] for sd in (0, 1, 2)]) for s in SQ6}
            dom = per[11] - per[5]
            delta = "  (ref)" if k == 24 else f"{drift[(k,fz)]-drift[(24,fz)]:+9.4f}"
            print(f"{lab:>10s} {k:>3d} " + " ".join(f"{per[s]:9.4f}" for s in SQ6)
                  + f" {drift[(k,fz)]:+12.4f} {dom:+12.4f} {delta:>9s}")
    viol = [(("frozen" if fz else "trainable"), k, round(drift[(k, fz)] - drift[(24, fz)], 4))
            for fz in (True, False) for k in KS
            if k != 24 and abs(drift[(k, fz)] - drift[(24, fz)]) > DRIFT_TOL]
    for fz, lab in ((True, "frozen"), (False, "trainable")):
        v = [k for k in KS if k != 24 and abs(drift[(k, fz)] - drift[(24, fz)]) > DRIFT_TOL]
        print(f"   {lab:>10s}: max |drift(K) - drift(K=24)| = "
              f"{max(abs(drift[(k,fz)]-drift[(24,fz)]) for k in KS if k != 24):.4f} -> "
              f"{'DRIFT-K-INDEPENDENT' if not v else 'DRIFT-K-DEPENDENT at K=' + str(v)}")
    print(f"\nCURVE 5a VERDICT: {'DRIFT-K-INDEPENDENT' if not viol else 'DRIFT-K-DEPENDENT -- ' + str(viol)}")
    print("   (§6.1 Curve 5a: 'report per-arm, since the 98M record already shows frozen flat")
    print("    and trainable worsening' -- the per-arm split above is that report.)")

    # ============================ TEST-W ==================================
    thr4, dist4 = exact_threshold(4)
    thr3, dist3 = exact_threshold(3)
    for readout, getter, ref_T, ref_U in (
            ("11 squarings (depth ladder, §5.3.1's definition)",
             lambda dd, k, fz: sorted(dd[(k, fz, s)]["kappa"][11] for s in (0, 1, 2)),
             30.5, (6.5, 9.0, 6.0, 9.0)),
            ("h_top (5 squarings, breadth battery)",
             None, None, None)):
        hdr(f"TEST-W -- within-scale freeze ordering, 4 strata, at {readout}")
        U392, U98, strata = [], [], []
        for k in KS:
            if getter:
                a = getter(D392, k, True); b = getter(D392, k, False)
                a9 = getter(D98, k, True); b9 = getter(D98, k, False)
            else:
                a = sorted(cell(B392, k, True, "htop_kappa"))
                b = sorted(cell(B392, k, False, "htop_kappa"))
                a9 = sorted(cell(B98, k, True, "htop_kappa"))
                b9 = sorted(cell(B98, k, False, "htop_kappa"))
            U392.append(stratum_U(a, b)); U98.append(stratum_U(a9, b9))
            strata.append(a + b)
        T392, T98 = sum(U392), sum(U98)
        print(f"{'K':>4s} " + " ".join(f"{k:>8d}" for k in KS) + f" {'T/36':>9s}")
        print(f"{'1310M':>4s} " + " ".join(f"{u:8.1f}" for u in U392) + f" {T392:9.1f}")
        print(f"{'98M':>4s} " + " ".join(f"{u:8.1f}" for u in U98) + f" {T98:9.1f}")
        if ref_T is not None:
            ok = abs(T98 - ref_T) < 1e-9 and tuple(U98) == ref_U
            print(f"     [392M reference cross-check vs Stage-C record ({ref_U}, T={ref_T}): "
                  f"{'REPRODUCED' if ok else 'MISMATCH -- ' + str(tuple(U98))}]")
        pd_, _ = two_sided_p(exact_null(strata), T392)
        print(f"\n392M T_W = {T392}/36  (4-strata exact bar T>={thr4:.0f}, two-sided p<0.01); "
              f"exact two-sided p = {pd_:.4e}")
        print(f"descriptive cross-scale delta: T_W(1310M) - T_W(392M) = {T392-T98:+.1f}")
        print(f"\nLOSO (3 strata, exact bar T >= {thr3:.0f}/27):")
        nfail = 0
        for i, k in enumerate(KS):
            t = sum(U392[j] for j in range(4) if j != i)
            good = t >= thr3
            nfail += (not good)
            print(f"   drop K={k:2d}: T = {t:5.1f}/27  {'clears' if good else 'FAILS'}")
        # §5.3.1 partition
        if T392 > 31.5 and nfail <= 1:
            v = "ORDERING-CONFIRMED (within-1.31B) / ORDERING-SCALE-STABLE"
            if T392 == 36:
                v += "  [ORDERING-SCALE-STRENGTHENS: T=36/36, perfect separation]"
        elif T392 <= 6:
            v = "ORDERING-INVERTED"
        elif 29.5 <= T392 <= 31.5 or (nfail >= 2 and T392 > 29.5):
            v = "ORDERING-INDETERMINATE-AT-4-STRATA (INDETERMINATE dominates, §5.3.1)"
        elif 6 < T392 < 29.5:
            v = "ORDERING-NEGLIGIBLE / ORDERING-SCALE-LOST"
        else:
            v = "UNCLASSIFIED -- report raw"
        print(f"\nTEST-W VERDICT (§5.3.1 partition): {v}   [LOSO failures: {nfail}/4]")

    # ============================ TEST-X ==================================
    for name, get392, get98 in (
            ("Curve 1: kappa @ h_top (5 squarings)",
             lambda k, fz: sorted(cell(B392, k, fz, "htop_kappa")),
             lambda k, fz: sorted(cell(B98, k, fz, "htop_kappa"))),
            ("Curve 5b: kappa @ 11 squarings (continuity with #4/#8)",
             lambda k, fz: sorted(D392[(k, fz, s)]["kappa"][11] for s in (0, 1, 2)),
             lambda k, fz: sorted(D98[(k, fz, s)]["kappa"][11] for s in (0, 1, 2))),
            (f"Curve 5b: kappa @ s*={S_STAR} squarings (Rule R-delta's elected depth)",
             lambda k, fz: sorted(D392[(k, fz, s)]["kappa"][S_STAR] for s in (0, 1, 2)),
             lambda k, fz: sorted(D98[(k, fz, s)]["kappa"][S_STAR] for s in (0, 1, 2)))):
        hdr(f"TEST-X -- cross-scale (392M vs 1.31B), 8 strata (4K x 2 recipes). {name}")
        rows, strata, T = [], [], 0.0
        ties_tot = 0
        for k in KS:
            for fz in (True, False):
                a, b = get392(k, fz), get98(k, fz)
                u = stratum_U(a, b)
                ties = sum(1 for x in a for y in b if x == y)
                ties_tot += ties
                T += u
                strata.append(a + b)
                rows.append((k, "frozen" if fz else "trainable", a, b, u, ties))
        print(f"{'K':>3s} {'recipe':>10s} {'1310M kappa':>26s} {'392M kappa':>26s} {'U/9':>6s} {'ties':>5s}")
        for k, r, a, b, u, ti in rows:
            print(f"{k:>3d} {r:>10s} {str([round(x,4) for x in a]):>26s} "
                  f"{str([round(x,4) for x in b]):>26s} {u:6.1f} {ti:5d}")
        p2, _ = two_sided_p(exact_null(strata), T)
        v = ("SCALE-IMPROVES" if T >= 53 else "SCALE-DEGRADES" if T <= 19
             else "no detectable directional shift")
        print(f"\nT_X = {T:.1f}/72  -> {v}   (bars: >=53 improves, <=19 degrades); "
              f"exact two-sided p = {p2:.4e}; realized ties = {ties_tot}/72")
        # 6-strata sensitivity: drop K=24 pair (§5.4). Governs on disagreement.
        s6 = [strata[i] for i, (k, *_r) in enumerate(rows) if k != 24]
        T6 = sum(u for (k, _r, _a, _b, u, _t) in rows if k != 24)
        v6 = ("SCALE-IMPROVES" if T6 >= 42 else "SCALE-DEGRADES" if T6 <= 12
              else "no detectable directional shift")
        p6, _ = two_sided_p(exact_null(s6), T6)
        print(f"6-strata sensitivity (K=24 pair dropped, §5.4): T = {T6:.1f}/54 -> {v6}; "
              f"exact two-sided p = {p6:.4e}")
        print(f"   {'AGREES with the 8-strata verdict' if v6 == v else '!!! DISAGREES -- per §5.4 the 6-STRATA VERDICT GOVERNS and this is the headline instrument note'}")
        thr7, _ = exact_threshold(7)
        lo7 = 63 - thr7   # exact mirror bar for the DEGRADES (lower) tail
        tail = ("upper (IMPROVES)" if T >= 53 else "lower (DEGRADES)" if T <= 19
                else "neither -- no directional verdict to stress-test")
        print(f"LOSO over all 8 strata. 7-strata exact bars: T >= {thr7:.0f}/63 (improves) / "
              f"T <= {lo7:.0f}/63 (degrades). NOT enumerated in §5.3's table (which stops at 8) "
              f"-- computed here by the same construction. Stress-testing the {tail} tail:")
        held = 0
        for i, (k, r, *_rest) in enumerate(rows):
            t = T - rows[i][4]
            if T <= 19:
                ok = t <= lo7
            elif T >= 53:
                ok = t >= thr7
            else:
                ok = None
            mark = "" if ok is None else ("holds" if ok else "FAILS")
            held += (ok is True)
            print(f"   drop (K={k}, {r:>9s}): T = {t:5.1f}/63  {mark}")
        if T <= 19 or T >= 53:
            print(f"   -> the verdict survives {held}/8 leave-one-stratum-out subsets")

    # ======================= §6.2 per-K cross-scale =======================
    for cname, delta, get392, get98 in (
            ("Curve 1 (kappa @ h_top)", DELTA,
             lambda k, fz: med(cell(B392, k, fz, "htop_kappa")),
             lambda k, fz: med(cell(B98, k, fz, "htop_kappa"))),
            ("Curve 4 (kappa @ h_fix)", DELTA,
             lambda k, fz: med(cell(B392, k, fz, "hfix_kappa")),
             lambda k, fz: med(cell(B98, k, fz, "hfix_kappa"))),
            (f"Curve 5b (kappa @ s*={S_STAR})", DELTA_DEPTH,
             lambda k, fz: med([D392[(k, fz, s)]["kappa"][S_STAR] for s in (0, 1, 2)]),
             lambda k, fz: med([D98[(k, fz, s)]["kappa"][S_STAR] for s in (0, 1, 2)]))):
        hdr(f"§6.2 CROSS-SCALE (392M -> 1.31B) per K per recipe -- {cname}, delta = {delta}")
        print(f"{'K':>3s} {'recipe':>10s} {'1310M med':>10s} {'392M med':>10s} {'D_scale':>10s} "
              f"{'verdict':>16s} {'cap gate':>9s}")
        labels = []
        for k in KS:
            for fz in (True, False):
                a, b = get392(k, fz), get98(k, fz)
                dsc = a - b
                gate = med(cell(B392, k, fz, "htop_kappa")) >= CAP_BAR
                if dsc > delta:
                    lab = "SCALE-IMPROVES"
                elif dsc < -delta:
                    lab = "SCALE-DEGRADES"
                else:
                    lab = "SCALE-STABLE" if gate else "STABLE*(gate fails)"
                labels.append(lab)
                print(f"{k:>3d} {'frozen' if fz else 'trainable':>10s} {a:10.4f} {b:10.4f} "
                      f"{dsc:+10.4f} {lab:>16s} {'ok' if gate else 'FAIL':>9s}")
        if all(l == "SCALE-STABLE" for l in labels):
            print(f"\nCURVE VERDICT: SCALE-STABLE (curve) -- holds at all four K in both recipes")
        else:
            bad_ = [(k, "frozen" if fz else "trainable", labels[i])
                    for i, (k, fz) in enumerate([(k, f) for k in KS for f in (True, False)])
                    if labels[i] != "SCALE-STABLE"]
            print(f"\nCURVE VERDICT: not stable everywhere -- {bad_}")

    # ==================== Rule R-delta mechanics on 392M ==================
    hdr("Rule R-delta MECHANICS APPLIED TO 1.31B DATA -- DESCRIPTIVE ONLY, NOT A RE-ELECTION")
    print("§4.6.1 step 3 / §5.2 pin R-delta to the 98M six-rung re-score, evaluated BEFORE any")
    print("392M cell was queued; it elected s*=13, delta_depth=0.095 (EXPERIMENT_LOG #16).")
    print("'No 392M number can influence any threshold in this design.' The table below is")
    print("reported because it was asked for, and is NOT used by any verdict above.\n")
    print(f"{'s':>3s} " + " ".join(f"{'K'+str(k)+('F' if f else 'T'):>8s}" for k in KS for f in (True, False))
          + f" {'3rd-smallest':>13s} {'delta*(s)':>10s} {'>=0.05':>7s}")
    for s in (11, 13, 15):
        H = []
        for k in KS:
            for fz in (True, False):
                H.append(1.0 - med([D392[(k, fz, sd)]["kappa"][s] for sd in (0, 1, 2)]))
        third = sorted(H)[2]
        dstar = (int(third / 0.005)) * 0.005
        print(f"{s:>3d} " + " ".join(f"{h:8.4f}" for h in H)
              + f" {third:13.4f} {dstar:10.3f} {str(dstar>=0.05):>7s}")
    print("\n(The SAME table on the 392M six-rung data -- the reference side of this pair --")
    print(" s*=13 / delta_depth=0.095; recomputed here as a receipt:)")
    print(f"{'s':>3s} " + " ".join(f"{'K'+str(k)+('F' if f else 'T'):>8s}" for k in KS for f in (True, False))
          + f" {'3rd-smallest':>13s} {'delta*(s)':>10s} {'>=0.05':>7s}")
    for s in (11, 13, 15):
        H = []
        for k in KS:
            for fz in (True, False):
                H.append(1.0 - med([D98[(k, fz, sd)]["kappa"][s] for sd in (0, 1, 2)]))
        third = sorted(H)[2]
        dstar = (int(third / 0.005)) * 0.005
        print(f"{s:>3d} " + " ".join(f"{h:8.4f}" for h in H)
              + f" {third:13.4f} {dstar:10.3f} {str(dstar>=0.05):>7s}")

    # ==================== cross-scale drift comparison ====================
    hdr("CROSS-SCALE DRIFT COMPARISON 392M vs 1.31B (median of per-seed kappa@11sq - kappa@5sq; "
        "difference-of-medians in brackets, §5.1 m1)")
    print(f"{'recipe':>10s} {'K':>3s} {'392M drift':>22s} {'98M drift':>22s} {'delta':>9s}")
    for fz, lab in ((True, "frozen"), (False, "trainable")):
        for k in KS:
            d392 = med([D392[(k, fz, s)]["kappa"][11] - D392[(k, fz, s)]["kappa"][5] for s in (0, 1, 2)])
            d98 = med([D98[(k, fz, s)]["kappa"][11] - D98[(k, fz, s)]["kappa"][5] for s in (0, 1, 2)])
            m392 = med([D392[(k, fz, s)]["kappa"][11] for s in (0, 1, 2)]) - med([D392[(k, fz, s)]["kappa"][5] for s in (0, 1, 2)])
            m98 = med([D98[(k, fz, s)]["kappa"][11] for s in (0, 1, 2)]) - med([D98[(k, fz, s)]["kappa"][5] for s in (0, 1, 2)])
            print(f"{lab:>10s} {k:>3d} {d392:+10.4f} [{m392:+8.4f}] {d98:+10.4f} [{m98:+8.4f}] "
                  f"{d392-d98:+9.4f}")


if __name__ == "__main__":
    main()
