"""Validation: every archived seed-0 trajectory point that overlaps the new
dense grid must reproduce from the new probes to <=1e-6 (same instrument,
same fixed per-corpus sampling seed, same GPU class => deterministic).

Overlap: wave1ext archived steps {1000,11000,...,61000,67547} are ALL on the
new stride-2 grid (odd thousands); wave2 archived steps overlap at
{1000,41000,81000,91552} on the new stride-8 grid.

Run: python3 check_trajectories_vs_archived.py   (from the repo root)
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tidy import pooled_subset_one_checkpoint, ARCHIVED_4

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.dirname(HERE)


def main():
    new_rows = {(r["family"], r["corpus_trained_on"], r["seed"], r["step"]): r
                for r in json.load(open(os.path.join(HERE, "trajectories_tidy.json")))}

    total_exact = total_diff = 0
    for family, arch_dir, prefix, steps in [
        ("wave1ext", "2026-07-05_wave1ext", "rung1ext_traj_step",
         [1000, 11000, 21000, 31000, 41000, 51000, 61000, 67547]),
        ("wave2", "2026-07-05_trackc_rung2", "rung2_traj_step",
         [1000, 41000, 81000, 91552]),
    ]:
        n_exact = n_diff = 0
        for step in steps:
            arch = json.load(open(os.path.join(RUNS, arch_dir, f"{prefix}{step}.json")))
            for ckpath, cinfo in arch["per_checkpoint"].items():
                corpus, seed = cinfo["corpus_trained_on"], cinfo["seed"]
                ref = pooled_subset_one_checkpoint(cinfo, ARCHIVED_4)
                new = new_rows[(family, corpus, seed, step)]
                dv = abs(ref["gram_deviation_mean"] - new["archived4_gram_deviation_mean"])
                if dv <= 1e-6:
                    n_exact += 1
                else:
                    n_diff += 1
                    print(f"  DIFF {family} step={step} {corpus} s{seed}: "
                          f"ref={ref['gram_deviation_mean']:.9f} "
                          f"new={new['archived4_gram_deviation_mean']:.9f} |d|={dv:.3e}")
        print(f"{family}: {n_exact} exact, {n_diff} diffs "
              f"({len(steps)} steps x 2 seed-0 runs)")
        total_exact += n_exact
        total_diff += n_diff

    print(f"\nTOTAL trajectory overlap: {total_exact} exact, {total_diff} diffs")
    sys.exit(1 if total_diff else 0)


if __name__ == "__main__":
    main()
