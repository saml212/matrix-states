#!/usr/bin/env python3
"""ATTRIBUTION ARM harvest -- the three pre-registered verdicts.

Bars (ATTRIBUTION_ARM.md §3, ratified by EXPERIMENT_LOG 2026-08-22 #22):
  V1 (K=32 trainable) / V2 (K=40 trainable), Curve 1 @ h_top:
      TOKEN-BUDGET-LIMITED iff kappa >= 0.90 on 2/2 extended seeds.
      Otherwise SCALE-DEGRADES stands and is STRENGTHENED by the control.
  V3 (Curve 5b @ s*=13, BOTH arms, all six (K,recipe) cells):
      TOKEN-BUDGET-LIMITED iff Delta_scale vs the SAME 98M twin moves back
      inside +/- delta_depth = 0.095.  (kappa>=0.90 at 13sq is barred as a
      capability bar by NCR_SCALE_AXIS_DESIGN §5.5(ii).)

kappa = (acc - 1/K)/(1 - 1/K).  Aggregator = median (§5.1).
"""
import glob
import json
import os
import statistics
import sys

CELLS = [(16, False), (24, False), (32, False), (40, False), (32, True), (40, True)]
S_STAR = 13
DELTA_DEPTH = 0.095
CAP_BAR = 0.90
STAGEC = "../2026-08-22_scaleaxis_stagec"


def kappa(acc, k):
    p = 1.0 / k
    return (acc - p) / (1.0 - p)


def med(xs):
    return statistics.median(xs)


def load_batt(paths):
    out = {}
    for pat in paths:
        for p in glob.glob(pat):
            if os.path.basename(p).startswith("remeasure_"):
                continue
            r = json.load(open(p))
            k = r.get("K") or r["kscaling"]["K"]
            key = (k, bool(r["freeze_entity_adapter"]), int(r["ckpt_seed"]))
            ph = r["matched"]["P1b"]["per_hop"]
            top = next(e for e in ph.values() if e["role"] == "ladder_top")
            fix = next(e for e in ph.values() if e["role"] == "fixed_dist_control")
            out[key] = dict(htop=kappa(top["acc"], k), hfix=kappa(fix["acc"], k),
                            step=r["ckpt_step"], rec=r)
    return out


def load_depth(paths):
    out = {}
    for pat in paths:
        for p in glob.glob(pat):
            r = json.load(open(p))
            k = r["K"]
            key = (k, bool(r["freeze_entity_adapter"]), int(r["ckpt_seed"]))
            out[key] = {e["n_squarings"]: kappa(e["acc"], k)
                        for e in r["matched"]["P1b"]["per_hop"].values()}
            out[key]["_step"] = r["ckpt_step"]
    return out


def nm(k, fz):
    return f"K={k:2d} {'primary/frozen' if fz else 'compB/trainable'}"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    sc = os.path.join(d, STAGEC)
    B40 = load_batt([os.path.join(d, "attrib_*_kscaling.json")])
    D40 = load_depth([os.path.join(d, "depthext6_attrib_*_depthext.json")])
    B20 = load_batt([os.path.join(sc, "*scaleaxis392m_*_kscaling.json")])
    D20 = load_depth([os.path.join(sc, "depthext6_392m_*_depthext.json")])
    D98 = load_depth([os.path.join(sc, "ref98m_depth", "depthext6_*_depthext.json")])
    B98 = load_batt([os.path.join(sc, "ref98m_battery", "*_kscaling.json")])
    print(f"loaded  40k: battery {len(B40)} depth {len(D40)} | 20k: battery {len(B20)} "
          f"depth {len(D20)} | 98M: battery {len(B98)} depth {len(D98)}")
    have = [(k, fz) for k, fz in CELLS if all((k, fz, s) in B40 and (k, fz, s) in D40
                                             for s in (0, 1))]
    missing = [(k, fz) for k, fz in CELLS if (k, fz) not in have]
    if missing:
        print(f"!!! CELLS NOT YET SCORED AT 40k: {[nm(k,f) for k,f in missing]}")
    for key, lab in ((B40, "40k battery"), (D40, "40k depth")):
        bad = [kk for kk, v in key.items() if (v.get("step") or v.get("_step")) != 40000]
        if bad:
            print(f"!!! {lab}: cells not at ckpt_step 40000: {bad}")

    # =================== V1 / V2 -- Curve 1 @ h_top ======================
    print("\n" + "=" * 104)
    print("V1 / V2 -- Curve 1 @ h_top.  BAR: kappa >= 0.90 on 2/2 extended seeds "
          "=> TOKEN-BUDGET-LIMITED")
    print("=" * 104)
    print(f"{'cell':26s} {'20k s0':>8s} {'20k s1':>8s} {'20k s2':>8s} {'20k med':>8s} | "
          f"{'40k s0':>8s} {'40k s1':>8s} {'40k med':>8s} {'>=0.90':>7s}  verdict")
    v12 = {}
    for k, fz in [(32, False), (40, False)]:
        if (k, fz) not in have:
            print(f"{nm(k,fz):26s}  -- not yet scored at 40k --")
            continue
        a = [B20[(k, fz, s)]["htop"] for s in (0, 1, 2)]
        b = [B40[(k, fz, s)]["htop"] for s in (0, 1)]
        n_ok = sum(1 for x in b if x >= CAP_BAR)
        verdict = ("TOKEN-BUDGET-LIMITED" if n_ok == 2
                   else "SCALE-DEGRADES STANDS (strengthened by the control)")
        v12[(k, fz)] = verdict
        print(f"{nm(k,fz):26s} {a[0]:8.4f} {a[1]:8.4f} {a[2]:8.4f} {med(a):8.4f} | "
              f"{b[0]:8.4f} {b[1]:8.4f} {med(b):8.4f} {str(n_ok)+'/2':>7s}  {verdict}")
    for lbl, k in (("V1", 32), ("V2", 40)):
        if (k, False) in v12:
            print(f"\n{lbl} (K={k} trainable): {v12[(k, False)]}")

    # ======================== V3 -- Curve 5b @ s* ========================
    print("\n" + "=" * 104)
    print(f"V3 -- Curve 5b @ s*={S_STAR}.  BAR: |Delta_scale vs the SAME 98M twin| "
          f"<= {DELTA_DEPTH} => TOKEN-BUDGET-LIMITED")
    print("=" * 104)
    print(f"{'cell':26s} {'98M med':>8s} {'20k med':>8s} {'D_20k':>8s} | "
          f"{'40k s0':>8s} {'40k s1':>8s} {'40k med':>8s} {'D_40k':>9s} {'|D|<=.095':>10s}  verdict")
    v3 = {}
    for k, fz in CELLS:
        r98 = med([D98[(k, fz, s)][S_STAR] for s in (0, 1, 2)])
        r20 = med([D20[(k, fz, s)][S_STAR] for s in (0, 1, 2)])
        d20 = r20 - r98
        if (k, fz) not in have:
            print(f"{nm(k,fz):26s} {r98:8.4f} {r20:8.4f} {d20:+8.4f} |   -- not yet scored at 40k --")
            continue
        b = [D40[(k, fz, s)][S_STAR] for s in (0, 1)]
        r40 = med(b)
        d40 = r40 - r98
        ok = abs(d40) <= DELTA_DEPTH
        verdict = "TOKEN-BUDGET-LIMITED" if ok else "SCALE-DEGRADES STANDS"
        v3[(k, fz)] = verdict
        print(f"{nm(k,fz):26s} {r98:8.4f} {r20:8.4f} {d20:+8.4f} | {b[0]:8.4f} {b[1]:8.4f} "
              f"{r40:8.4f} {d40:+9.4f} {str(ok):>10s}  {verdict}")
    if len(v3) == len(CELLS):
        n_lim = sum(1 for v in v3.values() if v == "TOKEN-BUDGET-LIMITED")
        print(f"\nV3 (depth tail, both arms): {n_lim}/{len(CELLS)} cells TOKEN-BUDGET-LIMITED; "
              f"{len(CELLS)-n_lim}/{len(CELLS)} SCALE-DEGRADES STANDS")

    # ---- SEED-MATCHED SENSITIVITY (disclosed ambiguity) -----------------
    # The 20k record aggregates 3 seeds; the extended arm has 2 (s0,s1 -- the
    # design's own "seeds 0-1" shape). Comparing a 2-seed median to a 3-seed
    # median is not seed-matched. The pre-registration does not say which to
    # use, so BOTH are reported and the verdict is checked for sensitivity.
    print("\n" + "-" * 104)
    print("SEED-MATCHED SENSITIVITY -- 20k restricted to seeds {0,1}, the same seeds the arm ran")
    print("-" * 104)
    print(f"{'cell':26s} {'98M med(3)':>11s} {'20k med(3)':>11s} {'20k med(0,1)':>13s} "
          f"{'40k med(0,1)':>13s} {'D_40k vs 98M(3)':>16s} {'verdict same?':>14s}")
    for k, fz in CELLS:
        if (k, fz) not in have:
            continue
        r98 = med([D98[(k, fz, s)][S_STAR] for s in (0, 1, 2)])
        r98b = med([D98[(k, fz, s)][S_STAR] for s in (0, 1)])
        r20 = med([D20[(k, fz, s)][S_STAR] for s in (0, 1, 2)])
        r20b = med([D20[(k, fz, s)][S_STAR] for s in (0, 1)])
        r40 = med([D40[(k, fz, s)][S_STAR] for s in (0, 1)])
        same = (abs(r40 - r98) <= DELTA_DEPTH) == (abs(r40 - r98b) <= DELTA_DEPTH)
        print(f"{nm(k,fz):26s} {r98:11.4f} {r20:11.4f} {r20b:13.4f} {r40:13.4f} "
              f"{r40-r98:+16.4f} {str(same):>14s}")
    print("(last column: does the V3 verdict change if the 98M twin is also restricted to "
          "seeds {0,1}?)")

    # ---- DIRECTION-OF-EFFECT CHECK (the arm's own assumption) -----------
    print("\n" + "-" * 104)
    print("DIRECTION OF THE EXTENSION EFFECT -- does the extra budget HELP or HURT?")
    print("ATTRIBUTION_ARM.md §1: 'Under-training can only manufacture DEGRADES. It cannot")
    print("manufacture STABLE and it cannot manufacture IMPROVES.' That argument assumes the")
    print("extension is non-harmful. Cells where 40k < 20k violate that assumption.")
    print("-" * 104)
    print(f"{'cell':26s} {'readout':>9s} {'20k med':>9s} {'40k med':>9s} {'change':>9s}  direction")
    for k, fz in CELLS:
        if (k, fz) not in have:
            continue
        for lab, a, b in (("h_top", med([B20[(k, fz, s)]["htop"] for s in (0, 1, 2)]),
                           med([B40[(k, fz, s)]["htop"] for s in (0, 1)])),
                          (f"s*={S_STAR}", med([D20[(k, fz, s)][S_STAR] for s in (0, 1, 2)]),
                           med([D40[(k, fz, s)][S_STAR] for s in (0, 1)]))):
            ch = b - a
            print(f"{nm(k,fz):26s} {lab:>9s} {a:9.4f} {b:9.4f} {ch:+9.4f}  "
                  f"{'HELPED' if ch > 0.01 else 'HURT' if ch < -0.01 else 'flat'}")

    # ============== full six-rung before/after (context) =================
    print("\n" + "=" * 104)
    print("FULL DEPTH PROFILE, median kappa -- 20k (3 seeds) vs 40k (2 seeds) vs 98M (3 seeds)")
    print("=" * 104)
    print(f"{'cell':26s} {'budget':>7s} " + " ".join(f"{'@'+str(s)+'sq':>9s}" for s in (5, 7, 9, 11, 13, 15)))
    for k, fz in CELLS:
        for lab, src, seeds in (("98M", D98, (0, 1, 2)), ("20k", D20, (0, 1, 2)),
                                ("40k", D40, (0, 1))):
            if lab == "40k" and (k, fz) not in have:
                continue
            row = [med([src[(k, fz, s)][sq] for s in seeds]) for sq in (5, 7, 9, 11, 13, 15)]
            print(f"{nm(k,fz):26s} {lab:>7s} " + " ".join(f"{x:9.4f}" for x in row))

    # ==================== marginal-CE table, all 12 ======================
    print("\n" + "=" * 104)
    print("MARGINAL-CE TABLE (all 12 cells) -- full_graft CE over the RESUMED segment")
    print("is the #2 plateau finding universal, or specific to 0230/0233?")
    print("=" * 104)
    print(f"{'cell':26s} {'seed':>4s} {'status':>10s} {'step':>6s} {'CE@start':>9s} "
          f"{'CE@end':>9s} {'delta':>8s} {'GATE-0 clause':>14s}")
    res = "/ephemeral/scaleaxis/attribution/results"
    local = os.path.join(d, "artifacts")
    src_dir = local if os.path.isdir(local) else res
    deltas = []
    for k, fz in CELLS:
        for s in (0, 1):
            name = f"attrib40k_K{k}_{'primary' if fz else 'compB'}_s{s}"
            p = os.path.join(src_dir, f"{name}.json")
            if not os.path.exists(p):
                print(f"{nm(k,fz):26s} {s:>4d}  -- artifact not found --")
                continue
            r = json.load(open(p))
            h = (r.get("loss_history") or {}).get("full_graft") or []
            if not h:
                print(f"{nm(k,fz):26s} {s:>4d} {r.get('status'):>10s} -- no loss_history --")
                continue
            c0, c1 = h[0][1], h[-1][1]
            dlt = c1 - c0
            deltas.append((k, fz, s, dlt))
            print(f"{nm(k,fz):26s} {s:>4d} {str(r.get('status')):>10s} {str(r.get('step')):>6s} "
                  f"{c0:9.4f} {c1:9.4f} {dlt:+8.4f} {'FIRES' if dlt >= 0 else 'passes':>14s}")
    if deltas:
        ds = [x[3] for x in deltas]
        up = [x for x in deltas if x[3] >= 0]
        print(f"\nsummary: n={len(ds)}  min={min(ds):+.4f}  max={max(ds):+.4f}  "
              f"median={med(ds):+.4f}  |delta| max={max(abs(x) for x in ds):.4f}")
        print(f"cells with CE NOT decreasing over the marginal segment (GATE-0 clause fires): "
              f"{len(up)}/{len(ds)} -> "
              + ", ".join(f"K{k}{'F' if f else 'T'}s{s}({dl:+.4f})" for k, f, s, dl in up))
        print("#2 measured +0.026/+0.030 at 0230/0233 against +/-0.03 plateau noise.")


if __name__ == "__main__":
    main()
