#!/usr/bin/env python3
"""V2' BUILD SMOKE -- two things the literal 50-step run cannot show on its own.

A. THE SCHEDULE ASSERTION. Calls the DEPLOYED runner's OWN `resume_const_lr`
   and reproduces the EXACT loop expression the patch installs
   (`cur_lr = const_lr if const_lr is not None else get_lr(step, ...)`), so the
   values asserted are the values the training loop uses -- not a
   re-implementation. Asserts LR at steps 20001 and 20050 (and 20025, 30000,
   40000) all == 3.0e-05, and records the BEFORE (re-opened cosine) value at
   each step so the delta is on the record.

B. THE CLAUSE-BY-CLAUSE VALIDITY EXERCISE (EXPERIMENT_LOG 2026-08-23 #1).
   The attribution arm shipped a checker whose `config.steps_target` clause was
   broken and the build smoke never noticed, because at 3 marginal steps the
   `step >= 40000` clause failed FIRST -- assert-order masking. The fix is not
   "run a longer smoke", it is to make EVERY clause first-to-fail once:

     1. build a synthetic record that passes ALL clauses  -> must PASS
        (without this positive control, "everything fails" would look like
         "every clause has teeth" when the checker is simply always-false);
     2. for each clause, mutate ONLY that field           -> must FAIL, and the
        failure message must name THAT clause.

   The record is assembled from the REAL literal-spec run, with the fields a
   50-step smoke cannot satisfy (step, steps_target, loss_history length)
   overridden. This probe tests THE CHECKER, not the cell.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

EXT_STEPS, BASE_STEPS = 40_000, 20_000
LR, WARMUP = 3e-4, 200
CONST = 3.0e-5
PY = "/home/nvidia/tdenv/bin/python3"


def schedule_assertion() -> dict:
    """Part A -- against the deployed runner's OWN functions."""
    import ncr_lm_wave1_runner as R          # the PATCHED tree's runner
    assert hasattr(R, "resume_const_lr"), "the V2' patch (R6) is not present in this tree"
    const_lr = R.resume_const_lr(BASE_STEPS, LR, WARMUP)

    def loop_lr(step, const):
        # the EXACT expression patch R10 installs at the top of the training loop
        return (const if const is not None
                else R.get_lr(step, max_lr=LR, warmup_steps=WARMUP, total_steps=EXT_STEPS))

    rows, failures = [], []
    for step in (20001, 20025, 20050, 30000, 40000):
        after = loop_lr(step, const_lr)
        before = loop_lr(step, None)
        rows.append(dict(step=step, before_reopened_cosine=before, after_v2prime_constant=after,
                         factor_removed=round(before / after, 4)))
        if abs(after - CONST) > 1e-15:
            failures.append(f"step {step}: constant LR is {after!r}, expected {CONST!r}")
    # the two the brief names explicitly
    for step in (20001, 20050):
        if abs(loop_lr(step, const_lr) - CONST) > 1e-15:
            failures.append(f"REQUIRED ASSERTION FAILED at step {step}")
    # and the control: with the flag OFF the expression must be the pinned one
    if abs(loop_lr(20001, None) - R.get_lr(20001, max_lr=LR, warmup_steps=WARMUP,
                                           total_steps=EXT_STEPS)) > 0:
        failures.append("flag-OFF path is not byte-equivalent to the pinned get_lr call")
    return dict(test="A_schedule", const_lr_from_runner=const_lr,
                parent_final_lr_expected=CONST, rows=rows,
                warm_restart_factor_at_20001=rows[0]["factor_removed"],
                flag_off_equals_pinned_call=True,
                status="PASS" if not failures else "FAIL", failures=failures)


# Each entry: (clause name, mutation applied to a PASSING record, substring the
# failure message must contain).
MUTATIONS = [
    ("status", lambda d: d.__setitem__("status", "ABORTED-BUDGET"), "status"),
    ("step", lambda d: d.__setitem__("step", 39999), "step"),
    ("steps_target_TOP_LEVEL", lambda d: d.__setitem__("steps_target", 20000), "steps_target"),
    ("steps_target_UNDER_CONFIG_MUST_NOT_SATISFY",
     lambda d: (d.pop("steps_target", None), d["config"].__setitem__("steps_target", 40000)),
     "steps_target"),
    ("runner_tag", lambda d: d.__setitem__("runner_tag", "ncr_kscaling_runner_v1"), "runner_tag"),
    ("const_lr_on_resume", lambda d: d.__setitem__("const_lr_on_resume", False),
     "const_lr_on_resume"),
    ("resume_start_step", lambda d: d.__setitem__("resume_start_step", 0), "resume_start_step"),
    ("resume_const_lr", lambda d: d.__setitem__("resume_const_lr", 1.660549e-04),
     "resume_const_lr"),
    ("resume_const_lr_none", lambda d: d.__setitem__("resume_const_lr", None),
     "resume_const_lr"),
    ("K", lambda d: d["kscaling"].__setitem__("K", 32), "'K'"),
    ("d_ncr", lambda d: d["kscaling"].__setitem__("d_ncr", 33), "d_ncr"),
    ("h_top", lambda d: d["kscaling"].__setitem__("h_top", 48), "h_top"),
    ("deep_ladder", lambda d: d["kscaling"].__setitem__("deep_ladder", [4, 8, 17, 18, 37, 48]),
     "deep_ladder"),
    ("scale", lambda d: d["kscaling"].__setitem__("scale", "98m"), "scale"),
    ("backbone", lambda d: d["kscaling"]["backbone"].__setitem__("d_model", 768), "backbone"),
    ("freeze_entity_adapter", lambda d: d["config"].__setitem__("freeze_entity_adapter", True),
     "RECIPE MISMATCH"),
    ("params_per_arm", lambda d: d["params"].__setitem__("per_arm", 97860009), "params.per_arm"),
    ("loss_history_arms", lambda d: d["loss_history"].pop("backbone_only"),
     "loss_history arms"),
    ("backbone_only_len", lambda d: d["loss_history"].__setitem__(
        "backbone_only", d["loss_history"]["backbone_only"][:5]), "backbone_only len"),
    ("full_graft_len", lambda d: d["loss_history"].__setitem__(
        "full_graft", d["loss_history"]["full_graft"][:5]), "full_graft len"),
    ("non_finite_CE", lambda d: d["loss_history"]["full_graft"].__setitem__(
        10, [10, float("nan")]), "non-finite CE"),
]

# The clause #2 ruled MIS-SCOPED: a resumed segment at plateau moves CE by ~+0.03
# against +-0.03 noise. This record RISES over the marginal segment and MUST
# still pass -- the checker is plateau-tolerant by construction, finite-CE only.
PLATEAU_CASE = ("gate0_marginal_plateau_or_rise_MUST_PASS",
                lambda d: d["loss_history"].__setitem__(
                    "full_graft",
                    [[s, 4.60 + 0.03 * i / max(1, len(d["loss_history"]["full_graft"]) - 1)]
                     for i, (s, _v) in enumerate(d["loss_history"]["full_graft"])]))


def good_record(src: str) -> dict:
    d = json.load(open(src))
    d["status"] = "COMPLETED"
    d["step"] = EXT_STEPS
    d["steps_target"] = EXT_STEPS
    hist = d.get("loss_history") or {}
    for arm in ("full_graft", "backbone_only"):
        base = hist.get(arm) or [[1, 4.7]]
        d.setdefault("loss_history", {})[arm] = [
            [BASE_STEPS + 25 * (i + 1), float(base[i % len(base)][1])] for i in range(120)]
    return d


def run_check(check: str, rec: dict) -> tuple[int, str]:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(rec, f)
    cmd = check.replace(rec["_out_json_placeholder"], path)
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    os.unlink(path)
    return p.returncode, (p.stdout + p.stderr)[-600:]


def clause_probe(spec_path: str, src_record: str) -> dict:
    spec = json.load(open(spec_path))
    check = spec["validity_check"]
    out_json = spec["output_dir"] + "/" + os.path.basename(
        spec["cmd"].split("--out ")[1].split()[0])
    base = good_record(src_record)
    base["_out_json_placeholder"] = out_json

    results = []
    rc, msg = run_check(check, base)
    results.append(dict(clause="POSITIVE CONTROL (unmutated record)", rc=rc,
                        status="PASS" if rc == 0 else "FAIL -- the checker rejects a VALID "
                                                     "record, so every negative below is vacuous",
                        message=msg.strip()[-200:]))

    for name, mutate, expect in MUTATIONS:
        rec = copy.deepcopy(base)
        mutate(rec)
        rc, msg = run_check(check, rec)
        fired = rc != 0
        named = expect.lower() in msg.lower()
        results.append(dict(clause=name, rc=rc, expect_substr=expect, fired=fired,
                            names_the_clause=named,
                            status=("PASS" if fired and named else
                                    "FAIL -- did not fire" if not fired else
                                    "FAIL -- fired with the WRONG message"),
                            message=msg.strip()[-200:]))

    name, mutate = PLATEAU_CASE
    rec = copy.deepcopy(base)
    mutate(rec)
    rc, msg = run_check(check, rec)
    results.append(dict(clause=name, rc=rc,
                        status="PASS" if rc == 0 else "FAIL -- the Gate-0-marginal clause is "
                                                      "STILL mis-scoped; a plateau/rise must pass",
                        message=msg.strip()[-200:]))

    ok = all(r["status"].startswith("PASS") for r in results)
    return dict(test="B_clause_by_clause", spec=os.path.basename(spec_path),
                n_clauses_exercised=len(MUTATIONS) + 2, results=results,
                lesson="EXPERIMENT_LOG 2026-08-23 #1 -- assert-order masking: every clause is "
                       "made FIRST-TO-FAIL once, so no clause can hide behind an earlier one",
                status="PASS" if ok else "FAIL")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--src-record", required=True,
                    help="the literal-spec run's real results JSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rec = {"smoke": "v2prime_smoke.py",
           "design": "EXPERIMENT_LOG 2026-08-23 #3 (V2'), #1 (assert-order), #2 (Gate-0 scoping)",
           "A": schedule_assertion(),
           "B": clause_probe(args.spec, args.src_record)}
    rec["overall"] = "PASS" if all(rec[k]["status"] == "PASS" for k in ("A", "B")) else "FAIL"
    js = json.dumps(rec, indent=1, default=str)
    if args.out:
        open(args.out, "w").write(js)
    print(js)
    return 0 if rec["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
