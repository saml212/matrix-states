"""Harvest analysis for the attractor-robustness 2x2 n=3 escalation
(STATE.md campaign 3; runner matrix-thinking/deltanet_rd/
run_attractor_robustness_2x2.py; chained behind the capability sweep per
experiment-runs/2026-07-09_capability_sweep_launch/MANIFEST.md).

Recomputes every per-cell, per-seed same-corpus (openr1-mix-ext)
layer-pooled gram-deviation directly from the raw *_attractor_probe.json
files in ./box_results/ (verify-vs-raws against the runner's own
AGGREGATE.json), applies the pre-registered escalation/confirm rule
(|delta vs baseline| > 2 x 2.244355 corrected same-corpus seed-noise
floor), and reports n=3 mean +- sd per combo plus paired per-seed deltas.

Run with the repo venv: .venv/bin/python analyze_2x2_escalation.py
"""
import json, os, itertools
import numpy as np

CORPUS = "openr1-mix-ext"
FLOOR = 2.244355           # audit-corrected same-corpus cross-seed std (f09254a)
THRESH = 2.0 * FLOOR       # 4.48871, pre-registered
here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(here, "box_results")

per_seed = {}
for qk, gate in itertools.product(["True", "False"], repeat=2):
    key = f"qk{qk}_gate{gate}"
    for s in [0, 1, 2]:
        d = json.load(open(os.path.join(res, f"attrrob_qk{qk}_gate{gate}_s{s}_attractor_probe.json")))
        pc = list(d["per_checkpoint"].values())
        assert len(pc) == 1
        per_layer = pc[0]["per_corpus"][CORPUS]["per_layer"]
        vals = [v["gram_deviation_mean"] for v in per_layer.values()
                if v.get("gram_deviation_mean") is not None]
        assert len(vals) == 2
        per_seed.setdefault(key, {})[s] = sum(vals) / len(vals)

agg = json.load(open(os.path.join(res, "AGGREGATE.json")))
base = per_seed["qkTrue_gateFalse"]
base_mean = float(np.mean(list(base.values())))
print(f"pre-registered rule: |delta vs baseline| > 2 x {FLOOR} = {THRESH:.5f} (same-corpus floor)")
print(f"baseline (qkTrue_gateFalse) n=3: {[round(base[s],4) for s in [0,1,2]]} "
      f"mean={base_mean:.4f} sd={np.std(list(base.values()), ddof=1):.4f}\n")

verdicts = {}
for key in ["qkTrue_gateTrue", "qkFalse_gateFalse", "qkFalse_gateTrue"]:
    v = [per_seed[key][s] for s in [0, 1, 2]]
    m = float(np.mean(v))
    runner = agg["gram_dev_mean_by_cell_key"][key]
    assert abs(m - runner) < 1e-6, (key, m, runner)   # verify-vs-raws
    delta = m - base_mean
    paired = [per_seed[key][s] - base[s] for s in [0, 1, 2]]
    same_dir = all(d > 0 for d in paired) or all(d < 0 for d in paired)
    exceeds = abs(delta) > THRESH
    verdicts[key] = dict(per_seed=v, mean=m, sd=float(np.std(v, ddof=1)), delta=delta,
                         sigma_floor=delta / FLOOR, paired_deltas=paired,
                         same_direction=bool(same_dir), exceeds_threshold=bool(exceeds))
    print(f"{key}: n=3 {[round(x,4) for x in v]} mean={m:.4f} sd={np.std(v, ddof=1):.4f}")
    print(f"    delta vs baseline = {delta:+.4f} ({delta/FLOOR:+.2f} sigma_floor)  "
          f"paired per-seed deltas={[round(d,4) for d in paired]}  same-direction={same_dir}  "
          f"exceeds 2-sigma bar={exceeds}")

fire_recomputed = any(v["exceeds_threshold"] for v in verdicts.values())
print(f"\nrunner escalation.fire={agg['escalation']['fire']}  recomputed={fire_recomputed}  "
      f"match={agg['escalation']['fire'] == fire_recomputed}")
rec_zero = all(all(h[k] == 0.0 for k in h)
               for key in agg["rec_at_09_by_cell_key"] for h in agg["rec_at_09_by_cell_key"][key])
print(f"rec@0.9 all-zero (PROBE-INVALID floor, NON-DECISIONAL as pre-registered): {rec_zero}")
print(f"failed cells: {agg['failed'] or 'none'}")

json.dump({"per_seed": per_seed, "verdicts": verdicts, "fire_recomputed": fire_recomputed},
          open(os.path.join(here, "n3_recompute_summary.json"), "w"), indent=2)
print("\nwritten n3_recompute_summary.json")
