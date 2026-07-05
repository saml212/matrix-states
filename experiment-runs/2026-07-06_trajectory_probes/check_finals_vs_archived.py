"""Red-flag check: the dense-trajectory probes' FINAL-step rows must match the
archived harvests' own per-run final-checkpoint numbers.

The archived finals and the new trajectory finals score the SAME checkpoint
file with the SAME instrument (md5-matched) on the SAME eval corpora with the
SAME per-corpus fixed sampling seed (corpus_fixed_seed(corpus)+95_000,
independent of which/how many checkpoints share the probe call) -- so the
per-(run, corpus, layer) summaries should agree to float-reproducibility
levels, and any real mismatch is a red flag to REPORT, not paper over.

Inputs: tidy.json (built by build_tidy.py from the new dense probes) and
reference tidy built from the archived final-pooled JSONs.
"""
import json
import sys

TOL_HARD = 1e-6      # bitwise-reproducible expectation (same GPU class, TF32 state, seeds)
TOL_SOFT = 0.05      # nondeterminism allowance before we call it a true red flag

# archived family label -> (new family label, final step)
FAMILY_MAP = {
    "wave1": ("wave1", 67547),
    "wave1_ctrl": ("mixcontrol_ORIG_NOT_RERUN", 6103),   # orig-mix 14M control: not re-run (no dense ckpts beyond mixcontrol's)
    "wave1ext": ("wave1ext", 67547),
    "wave2": ("wave2", 91552),
    "mixcontrol": ("mixcontrol", 6103),
}


def key(r):
    return (r["family"], r["corpus_trained_on"], r["seed"], r["step"])


def main():
    new_tidy_path, ref_tidy_path = sys.argv[1], sys.argv[2]
    new_rows = {key(r): r for r in json.load(open(new_tidy_path))}
    ref_rows = json.load(open(ref_tidy_path))

    n_checked = n_exact = n_soft = n_flag = 0
    for ref in ref_rows:
        fam_new, final_step = FAMILY_MAP.get(ref["family"], (None, None))
        if fam_new is None or "NOT_RERUN" in fam_new:
            continue
        if ref["step"] != final_step:
            continue
        k = (fam_new, ref["corpus_trained_on"], ref["seed"], ref["step"])
        if k not in new_rows:
            print(f"MISSING in new tidy: {k}")
            n_flag += 1
            continue
        new = new_rows[k]
        for col in ("archived4_gram_deviation_mean", "archived4_span_frac",
                    "all7_gram_deviation_mean"):
            if col not in ref or col not in new:
                continue
            # LIKE-FOR-LIKE GUARD (first run of this check found exactly this,
            # 2026-07-05): wave1's archived harvest (2026-07-04) ran when the
            # box had only the 4 original eval corpora, so its "all7" pool IS
            # its archived-4 pool; the new dense probe pools 7. Comparing those
            # is a corpus-set mismatch, not an instrument mismatch -- the
            # archived-4 columns (the registered cross-scale convention) match
            # to <=1e-6 on all 24 runs. Skip all7 where the archived reference
            # never scored 7 corpora.
            if col.startswith("all7") and ref["family"] == "wave1":
                continue
            dv = abs(ref[col] - new[col])
            n_checked += 1
            if dv <= TOL_HARD:
                n_exact += 1
            elif dv <= TOL_SOFT:
                n_soft += 1
                print(f"soft-diff {k} {col}: ref={ref[col]:.9f} new={new[col]:.9f} |d|={dv:.3e}")
            else:
                n_flag += 1
                print(f"RED FLAG {k} {col}: ref={ref[col]:.9f} new={new[col]:.9f} |d|={dv:.3e}")

    print(f"\nchecked={n_checked} exact(<=1e-6)={n_exact} soft(<=0.05)={n_soft} RED_FLAGS={n_flag}")
    sys.exit(1 if n_flag else 0)


if __name__ == "__main__":
    main()
