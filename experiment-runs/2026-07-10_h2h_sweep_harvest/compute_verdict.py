#!/usr/bin/env python3
"""Verdict-of-record computation for the h2h 27-cell sweep (sec 1.31.1 frozen tiers).

Inputs: the 27 sweep training JSONs + the 18 sweep_remetric per-cell JSONs
(both in this directory). Recomputes every aggregate from raws; no numbers
are trusted from any prose.

Frozen pins (MARGINS_FROZEN.token / sec 1.31.1):
  chance = 1/32 = 0.03125; demonstration bar = 3x chance = 0.09375
  WIN  : contender clears bar AND delta >= 0.30 with paired CI excluding 0.30
  LOSE : ablation clears bar AND -delta >= 0.30 with CI excluding 0.30
  TIE  : CI entirely inside (-0.30, 0.30), or joint failure (neither arm clears)
  INDET: CI straddles +-0.30
  paired CI: half_width = t(n-1, .975) * s / sqrt(n); t = 4.303 at df=2 (n=3)
  seed aggregation: mean of seeds, per-seed table disclosed.
"""
import json
import math

T975_DF2 = 4.303
CHANCE = 1.0 / 32
BAR = 3 * CHANCE  # 0.09375
MARGIN = 0.30

ARCHS = ["contender", "ablation", "transformer"]
SEEDS = [0, 1, 2]


def rm(arch, task, s):
    with open(f"h2h_{arch}_{task}_sweep_s{s}_round4.json") as f:
        return json.load(f)


def tr(arch, task, s):
    with open(f"h2h_{arch}_{task}_sweep_s{s}.json") as f:
        return json.load(f)


def paired_ci(deltas):
    n = len(deltas)
    assert n == 3
    m = sum(deltas) / n
    s = math.sqrt(sum((d - m) ** 2 for d in deltas) / (n - 1))
    hw = T975_DF2 * s / math.sqrt(n)
    return m, s, hw, (m - hw, m + hw)


print("=" * 100)
print("PER-CELL TABLE (grammar tasks; acc_A = Leg-A primary, H_train)")
print("=" * 100)
hdr = (f"{'cell':44s} {'acc_A':>8s} {'bar?':>5s} {'rung2':>8s} {'rf@.9':>7s} "
       f"{'S0fire':>6s} {'instr':>5s} {'steps':>6s} {'trainGPUh':>9s}")
print(hdr)
acc = {}
for task in ["task1", "task2"]:
    for arch in ARCHS:
        for s in SEEDS:
            d = rm(arch, task, s)
            t = tr(arch, task, s)
            a = d["leg_a"]["acc_A"]
            acc[(arch, task, s)] = a
            s0 = d.get("s0_necessity_check")
            s0f = "-" if s0 is None else ("FIRE" if s0["hard_stop_fires"] else "ok")
            s0pass = "-" if s0 is None else ("P" if s0["passed"] else "F")
            print(f"h2h_{arch}_{task}_sweep_s{s:<38d} {a:8.4f} "
                  f"{'Y' if a > BAR else 'n':>5s} "
                  f"{d['rung2_identity_classifier']['accuracy']:8.4f} "
                  f"{d['leg_b_ridge']['rf@0.9']:7.4f} {s0f:>6s}/{s0pass} "
                  f"{'P' if d['instrument_health']['passed'] else 'F':>4s} "
                  f"{t['step_count']:6d} {t['wall_s']/3600:9.3f}")

print()
print("=" * 100)
print("AXIS-1 VERDICT — task1 primary (contender vs ablation; frozen sec 1.31.1 tiers)")
print("=" * 100)
for base in ["ablation", "transformer"]:
    deltas = [acc[("contender", "task1", s)] - acc[(base, "task1", s)] for s in SEEDS]
    m, sd, hw, (lo, hi) = paired_ci(deltas)
    print(f"contender - {base}: per-seed deltas = "
          f"{[round(d, 6) for d in deltas]}")
    print(f"  mean={m:.6f} sd={sd:.6f} half_width={hw:.6f} CI=({lo:.6f}, {hi:.6f})")
    print(f"  CI excludes +{MARGIN}? {'YES' if lo > MARGIN else 'NO'}")

cont = [acc[("contender", "task1", s)] for s in SEEDS]
abl = [acc[("ablation", "task1", s)] for s in SEEDS]
tfm = [acc[("transformer", "task1", s)] for s in SEEDS]
print(f"\nper-seed acc_A: contender={cont} mean={sum(cont)/3:.6f}")
print(f"                ablation ={abl} mean={sum(abl)/3:.6f}")
print(f"                transformer={tfm} mean={sum(tfm)/3:.6f}")
print(f"contender clears bar every seed? {all(a > BAR for a in cont)}")
print(f"ablation clears bar any seed?    {any(a > BAR for a in abl)}")
print(f"transformer clears bar any seed? {any(a > BAR for a in tfm)}")

d_ab = [acc[("contender", "task1", s)] - acc[("ablation", "task1", s)] for s in SEEDS]
m, sd, hw, (lo, hi) = paired_ci(d_ab)
if all(a > BAR for a in cont) and m >= MARGIN and lo > MARGIN:
    verdict = "WIN"
elif lo > -MARGIN and hi < MARGIN:
    verdict = "TIE"
elif lo <= MARGIN <= hi or lo <= -MARGIN <= hi:
    verdict = "INDETERMINATE"
else:
    verdict = "CHECK-BY-HAND"
print(f"\nAXIS-1 TASK1 VERDICT (contender vs ablation): {verdict}")

print()
print("=" * 100)
print("TASK2 — pre-registered joint-failure TIE check (H_train primary)")
print("=" * 100)
for arch in ARCHS:
    vals = [acc[(arch, "task2", s)] for s in SEEDS]
    print(f"{arch:12s}: {[round(v, 4) for v in vals]} mean={sum(vals)/3:.4f} "
          f"any>bar? {any(v > BAR for v in vals)}")
sec = {}
for arch in ARCHS:
    sec[arch] = [rm(arch, "task2", s)["leg_a_secondary_h_test"]["acc_A"] for s in SEEDS]
print(f"secondary H_test=(3,4), disclosed, non-comparable: "
      f"{ {a: [round(v,4) for v in vs] for a, vs in sec.items()} }")

print()
print("=" * 100)
print("LEG-B (diagnostic) + S0 + task3 bands")
print("=" * 100)
for arch in ["contender", "ablation"]:
    vals = [rm(arch, "task1", s)["leg_b_ridge"]["rf@0.9"] for s in SEEDS]
    cosv = [rm(arch, "task1", s)["leg_b_ridge"]["cos_mean"] for s in SEEDS]
    print(f"task1 {arch:10s} rf@0.9={[round(v,4) for v in vals]} "
          f"cos_mean={[round(v,4) for v in cosv]}")
for arch in ARCHS:
    for task in ["task1", "task2"]:
        ok = all(rm(arch, task, s)["leg_b_ridge_harness_sanity_control"]["passed"]
                 for s in SEEDS)
        ih = all(rm(arch, task, s)["instrument_health"]["passed"] for s in SEEDS)
        assert ok and ih, f"instrument failure {arch}/{task}"
print("all 18 cells: ridge harness sanity PASS, instrument health PASS")
for arch in ["contender", "ablation"]:
    for task in ["task1", "task2"]:
        for s in SEEDS:
            c = rm(arch, task, s)["s0_necessity_check"]
            assert not c["hard_stop_fires"], f"S0 HARD-STOP fires: {arch}/{task}/s{s}"
print("no S0 hard-stop fires anywhere (12 recurrent-arm cells checked)")

print("\ntask3 (LM control) final_val_loss_own; anchored band [1.90, 2.60] "
      "(ablation-primary anchor); transformer calib-grid band [1.90, 5.50]:")
for arch in ARCHS:
    vals = [tr(arch, "task3", s)["final_metric"] for s in SEEDS]
    print(f"  {arch:12s}: {[round(v, 4) for v in vals]}")

wall_grammar = sum(tr(a, t, s)["wall_s"] for a in ARCHS for t in ["task1", "task2"]
                   for s in SEEDS)
wall_task3 = sum(tr(a, "task3", s)["wall_s"] for a in ARCHS for s in SEEDS)
wall_rm = sum(rm(a, t, s)["wall_s"] for a in ARCHS for t in ["task1", "task2"]
              for s in SEEDS)
print(f"\nGPU-h: training cells {(wall_grammar+wall_task3)/3600:.3f} "
      f"(grammar {wall_grammar/3600:.3f} + task3 {wall_task3/3600:.3f}); "
      f"re-metric pass {wall_rm/3600:.4f}; "
      f"supervisor wall-clock projection at completion = 9.598 (2 GPUs); "
      f"ceiling 13.25")
