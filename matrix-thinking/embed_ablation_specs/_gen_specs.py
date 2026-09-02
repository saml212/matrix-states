#!/usr/bin/env python3
"""One-off generator for the embed-ablation queue spec JSONs (0640-0669).
Not part of the runner CLI -- run once to (re)produce the specs, then this
file can be deleted or kept for reproducibility. Schema copied from
experiment-runs/2026-08-29_box_final_archive/queue/completed/
005_laneA_probe_K128_s0.json (id, lane, hypothesis, cmd, gpu_h_estimate,
output_dir, validity_check, notes).

Audit F2 (FATAL) gating fix: the worker script has NO internal dependency
gating between phases, so this generator enforces it structurally --
phase A (probes, 0640-0645) and phase B (seeded cells, 0646-0669) are
written to SEPARATE directories, and phase_B is not to be staged/queued
until the 6 phase-A probe JSONs have been read and --check-admission has
passed (see the README.md this script also writes, and
EMBEDDING_ABLATION_DESIGN.md S6's identical gating sentence)."""
import json
import os

PY = "/home/nvidia/tdenv/bin/python3"
CODE_DIR = "/home/nvidia/embed_ablation"
CKPT_ROOT = "/data/embed_ablation_ckpts"
OUT_DIR = "/home/nvidia/embed_ablation/results"
PROBE_OUT_DIR = "/home/nvidia/embed_ablation/results/probes"   # audit F1: probes get their OWN subdir
DATA_DIR = "/data/deltanet_rd_data"
CORPUS = "wikitext-mix-ext"
HERE = os.path.dirname(os.path.abspath(__file__))
PHASE_A_DIR = os.path.join(HERE, "phase_A_probes")
PHASE_B_DIR = os.path.join(HERE, "phase_B_seeded")

PREREQ_NOTE = (
    "PREREQ (see EMBEDDING_ABLATION_DESIGN.md S6 'Deployment'): before this cmd can "
    "run, scp matrix-thinking/src/embed_ablation_rd.py to "
    f"{CODE_DIR}/embed_ablation_rd.py on the box (neither this script nor any "
    "matrix-thinking/src/ file is deployed there by default). Corpus wikitext-mix-ext "
    f"must already exist under {DATA_DIR}/wikitext103_mix_eot_extended/ (meta.json + "
    "train.pt + val.pt + *_doc_offsets.pt) -- the SAME corpus dir the unrelated "
    "FROZEN_BIAS_LM_DESIGN.md fixscale-seedext cells already train on (spec id "
    "645_laneB_392m_seedext_off_wikitext-mix-ext_s29, "
    "experiment-runs/2026-08-29_box_final_archive/queue/completed/), confirming the "
    "corpus is already materialized on the box -- id 645 there is a coincidental "
    "number from that campaign's own sequence, unrelated to this design's 0640+ ids."
)

GATING_NOTE = (
    "GATING (audit F2, FATAL): this is a PHASE-B (seeded) cell. Do NOT queue/run it "
    "until ALL 6 phase-A probe cells (phase_A_probes/0640-0645) have completed, their "
    "result JSONs have been read, and "
    "`embed_ablation_rd.py --check-admission --probe-results-dir "
    f"{PROBE_OUT_DIR} --intended-steps 2000 --intended-batch 64` "
    "exits 0. If it exits nonzero (non-monotone T1 curve in any arm, or extrapolated "
    "GPU-h/cell > 2.0), STOP -- re-derive --steps for every phase-B cmd below (or "
    "investigate the failing arm) before staging ANY of phase_B_seeded/. See "
    "phase_B_seeded/README.md and phase_A_probes/README.md."
)

SPECS_A = []
SPECS_B = []

n_a = 0
n_b = 0


def add_probe(name, arm, size, gpu_h_estimate):
    global n_a
    cid = f"{640 + n_a:04d}"
    n_a += 1
    out_path = f"{PROBE_OUT_DIR}/{name}.json"
    ckpt_dir = f"{CKPT_ROOT}/{name}"
    cmd = (
        f"mkdir -p {ckpt_dir} {PROBE_OUT_DIR} && cd {CODE_DIR} && {PY} embed_ablation_rd.py "
        f"--run-cell --arm {arm} --size {size} --match P --seed 0 "
        f"--steps 500 --batch-size 64 --seq-len 512 "
        f"--data-dir {DATA_DIR} --corpus {CORPUS} --role probe "
        f"--ckpt-dir {ckpt_dir} --out {out_path} --ceiling-gpuh 0.5"
    )
    validity = (
        f"{PY} -c \"import json; d=json.load(open('{out_path}')); "
        f"assert d.get('role') == 'probe'; "
        f"assert d.get('steps_completed', 0) >= 500 - 1; "
        f"assert 'T1' in d.get('final_evals', {{}})\""
    )
    hyp = (
        f"Rate/admission probe (phase A, audit F2), arm={arm} size={size}. Measures real "
        f"per-step wall time and the last-three-T1-evals monotonicity signal BEFORE any "
        f"phase-B seeded cell is staged, mirroring 005_laneA_probe_K128_s0's own Phase-0a "
        f"discipline. Not itself a trainability readout for the pre-registered decision "
        f"rule (harvest() filters role='probe' records out unconditionally, F1). Feeds "
        f"--check-admission (audit M4), which sizes phase B's --steps and can BLOCK phase "
        f"B outright if the measured rate implies >2 GPU-h/cell or if T1 is not improving."
    )
    spec = {
        "id": f"{cid}_embed_ablation_probe_{name}",
        "lane": "embed-ablation",
        "hypothesis": hyp,
        "cmd": cmd,
        "gpu_h_estimate": round(gpu_h_estimate, 3),
        "output_dir": PROBE_OUT_DIR,
        "validity_check": validity,
        "notes": PREREQ_NOTE + " Formula-extrapolated estimate; --ceiling-gpuh 0.5 hard-bounds "
                 "worst case. This is a PHASE-A cell -- no gating dependency (phase A has none "
                 "by construction), safe to queue/run immediately once the PREREQ above is met.",
    }
    SPECS_A.append((f"{cid}_embed_ablation_probe_{name}.json", spec))


def add_seeded(name, arm, size, match, seed, gpu_h_estimate, hyp):
    global n_b
    cid = f"{646 + n_b:04d}"
    n_b += 1
    out_path = f"{OUT_DIR}/{name}.json"
    ckpt_dir = f"{CKPT_ROOT}/{name}"
    cmd = (
        f"mkdir -p {ckpt_dir} {OUT_DIR} && cd {CODE_DIR} && {PY} embed_ablation_rd.py "
        f"--run-cell --arm {arm} --size {size} --match {match} --seed {seed} "
        f"--steps 2000 --batch-size 64 --seq-len 512 --eval-batches 50 "
        f"--data-dir {DATA_DIR} --corpus {CORPUS} --role cell "
        f"--ckpt-dir {ckpt_dir} --out {out_path} --ceiling-gpuh 2.0"
    )
    validity = (
        f"{PY} -c \"import json; d=json.load(open('{out_path}')); "
        f"assert d.get('complete') is True; "
        f"assert d.get('steps_completed', 0) >= 2000 - 1; "
        f"assert 'T1' in d.get('final_evals', {{}}) and 'T8' in d.get('final_evals', {{}})\""
    )
    spec = {
        "id": f"{cid}_embed_ablation_{name}",
        "lane": "embed-ablation",
        "hypothesis": hyp,
        "cmd": cmd,
        "gpu_h_estimate": round(gpu_h_estimate, 3),
        "output_dir": OUT_DIR,
        "validity_check": validity,
        "notes": PREREQ_NOTE + " " + GATING_NOTE + " Formula-extrapolated (anchor: "
                 "exp_d16_v2_SUMMARY.txt's CONFIG dict -- mat_dim=16, n_thinking_layers=12 "
                 "(NOT the SUMMARY.txt prose's stale '8 layers'), max_len=2048, T=8, "
                 "batch=96/GPU, 3000 steps, 2,552,788 params, 82.0 min measured -- scaled "
                 "linearly by batch/steps/params to this cell's own config). --ceiling-gpuh "
                 "2.0 hard-bounds worst case per EMBEDDING_ABLATION_DESIGN.md's <=2 GPU-h/"
                 "cell requirement; --eval-batches 50 per audit M1.",
    }
    SPECS_B.append((f"{cid}_embed_ablation_{name}.json", spec))


# ── Params (from embed_ablation_rd.py --selftest's own verified output,
#    re-derived here for the GPU-h formula, not re-typed by hand) ───────
PARAMS = {
    ("matrix", "S"): 2466562, ("flat", "P", "S"): 2468016, ("flat", "D", "S"): 3309120,
    ("flatten", "S"): 2452544,
    ("matrix", "M"): 3755808, ("flat", "P", "M"): 3765168, ("flat", "D", "M"): 5075520,
    ("flatten", "M"): 3770412,
}
ANCHOR_GPUH = 1.367       # exp_d16_v2, 82.0 min, batch=96, steps=3000, params=2552788
ANCHOR_PARAMS = 2552788
ANCHOR_BATCH = 96
ANCHOR_STEPS = 3000


def gpuh(params, batch, steps):
    return ANCHOR_GPUH * (batch / ANCHOR_BATCH) * (steps / ANCHOR_STEPS) * (params / ANCHOR_PARAMS)


# ── Phase A: 6 rate/admission probes (500 steps) ────────────────────────
add_probe("matrix_S", "matrix", "S", gpuh(PARAMS[("matrix", "S")], 64, 500))
add_probe("flat_S", "flat", "S", gpuh(PARAMS[("flat", "P", "S")], 64, 500))
add_probe("flatten_S", "flatten", "S", gpuh(PARAMS[("flatten", "S")], 64, 500))
add_probe("matrix_M", "matrix", "M", gpuh(PARAMS[("matrix", "M")], 64, 500))
add_probe("flat_M", "flat", "M", gpuh(PARAMS[("flat", "P", "M")], 64, 500))
add_probe("flatten_M", "flatten", "M", gpuh(PARAMS[("flatten", "M")], 64, 500))

# ── Phase B: 4 arm-configs x 2 sizes x 3 seeds = 24 seeded cells ────────
HYP = {
    "matrix_S": (
        "Verdict-carrier cell (matrix arm, size S). Outer-product embedding + RowThenCol "
        "matrix ops, mat_dim=16, n_layers=6, T=8 shared-weight iterations, GPT-2-tokenized "
        "wikitext-mix-ext, explicit outer-product init (std=sqrt(0.02), audit M2). Feeds "
        "BOTH decisions: STRENGTHEN-01 (vs flatten_S) and STRENGTHEN-04 (vs flatp_S)."
    ),
    "flatp_S": (
        "Verdict-carrier cell (flat-P arm, size S, STRENGTHEN-04). Direct (V, d_model=24) "
        "embedding table (std=0.02, audit M2) + standard nn.MultiheadAttention/FFN, params "
        "matched to matrix_S to within 0.059%. Unlike Run 22's flat arm (2.2x MORE params, "
        "d_model fixed at mat_dim^2=256), this run's d_model is solved for total-param "
        "equality."
    ),
    "flatd_S": (
        "Disclosed control cell (flat-D arm, size S), PRE-REGISTERED as params-UNMATCHED "
        "(d_model=2*mat_dim=32). Ratio vs matrix_S is ~1.34x (flat has MORE params), "
        "matching the historical Round-1/Run-22 comparison shape. NOT a verdict carrier -- "
        "never gates either decision (harvest() only reads match='P' groups)."
    ),
    "flatten_S": (
        "Verdict-carrier cell (flatten arm, size S, STRENGTHEN-01, audit M3 NEW). SAME "
        "outer-product embedding as matrix_S, flattened to a 256-dim vector, resized "
        "(Linear(256,16)) into a standard dense VectorThinkingBlock backbone at d_model=16, "
        "n_heads=8 (solved to 0.568%). Run 18's own historical recipe, now params-matched: "
        "isolates whether MATRIX OPERATIONS (not just the embedding) drive the T=1 "
        "advantage."
    ),
}
HYP["matrix_M"] = HYP["matrix_S"].replace("size S", "size M").replace(
    "mat_dim=16, n_layers=6", "mat_dim=24, n_layers=8").replace("flatten_S", "flatten_M").replace("flatp_S", "flatp_M")
HYP["flatp_M"] = HYP["flatp_S"].replace("size S", "size M").replace(
    "d_model=24", "d_model=36").replace("matrix_S", "matrix_M").replace("0.059%", "0.249%")
HYP["flatd_M"] = HYP["flatd_S"].replace("size S", "size M").replace(
    "d_model=2*mat_dim=32", "d_model=2*mat_dim=48").replace("matrix_S", "matrix_M")
HYP["flatten_M"] = HYP["flatten_S"].replace("size S", "size M").replace("matrix_S", "matrix_M").replace(
    "256-dim vector, resized (Linear(256,16)) into a standard dense VectorThinkingBlock "
    "backbone at d_model=16, n_heads=8 (solved to 0.568%)",
    "576-dim vector, resized (Linear(576,25)) into a standard dense VectorThinkingBlock "
    "backbone at d_model=25, n_heads=1 (solved to 0.389% -- n_heads=4 could not hit +/-1% "
    "at this size, see solve_matched_width's docstring)")

for seed in (0, 1, 2):
    add_seeded(f"matrix_S_s{seed}", "matrix", "S", "P", seed, gpuh(PARAMS[("matrix", "S")], 64, 2000), HYP["matrix_S"])
for seed in (0, 1, 2):
    add_seeded(f"flatp_S_s{seed}", "flat", "S", "P", seed, gpuh(PARAMS[("flat", "P", "S")], 64, 2000), HYP["flatp_S"])
for seed in (0, 1, 2):
    add_seeded(f"flatd_S_s{seed}", "flat", "S", "D", seed, gpuh(PARAMS[("flat", "D", "S")], 64, 2000), HYP["flatd_S"])
for seed in (0, 1, 2):
    add_seeded(f"flatten_S_s{seed}", "flatten", "S", "P", seed, gpuh(PARAMS[("flatten", "S")], 64, 2000), HYP["flatten_S"])
for seed in (0, 1, 2):
    add_seeded(f"matrix_M_s{seed}", "matrix", "M", "P", seed, gpuh(PARAMS[("matrix", "M")], 64, 2000), HYP["matrix_M"])
for seed in (0, 1, 2):
    add_seeded(f"flatp_M_s{seed}", "flat", "M", "P", seed, gpuh(PARAMS[("flat", "P", "M")], 64, 2000), HYP["flatp_M"])
for seed in (0, 1, 2):
    add_seeded(f"flatd_M_s{seed}", "flat", "M", "D", seed, gpuh(PARAMS[("flat", "D", "M")], 64, 2000), HYP["flatd_M"])
for seed in (0, 1, 2):
    add_seeded(f"flatten_M_s{seed}", "flatten", "M", "P", seed, gpuh(PARAMS[("flatten", "M")], 64, 2000), HYP["flatten_M"])


total_gpu_h = sum(s["gpu_h_estimate"] for _, s in SPECS_A + SPECS_B)
print(f"Generated {len(SPECS_A)} phase-A + {len(SPECS_B)} phase-B = {len(SPECS_A)+len(SPECS_B)} specs, "
      f"sum(gpu_h_estimate) = {total_gpu_h:.3f} GPU-h")
assert total_gpu_h <= 30.0, f"ledger over budget: {total_gpu_h} > 30 GPU-h"
assert all(s["gpu_h_estimate"] <= 2.0 for _, s in SPECS_A + SPECS_B), "a cell exceeds the 2 GPU-h/cell cap"

for fname, spec in SPECS_A:
    with open(os.path.join(PHASE_A_DIR, fname), "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    print("wrote phase_A_probes/" + fname)

for fname, spec in SPECS_B:
    with open(os.path.join(PHASE_B_DIR, fname), "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    print("wrote phase_B_seeded/" + fname)

README_A = """# Phase A -- rate/admission probes (0640-0645)

Audit F2 (FATAL) gating fix. These 6 cells have NO dependency on phase B
and may be queued/run as soon as the PREREQ in each spec's `notes` field
is met (embed_ablation_rd.py scp'd to the box, wikitext-mix-ext corpus
confirmed present).

Each probe trains 500 steps and writes its result JSON to
`/home/nvidia/embed_ablation/results/probes/` (a SEPARATE subdirectory
from the phase-B seeded cells' `results/`, per audit F1 -- this keeps
`harvest()`'s plain top-level `os.listdir(results_dir)` from ever seeing
probe files at all, in addition to the explicit `role`/filename filter
`_is_probe()` already applies).

**After all 6 land, before ANY phase_B_seeded/ spec is queued:**

```
/home/nvidia/tdenv/bin/python3 embed_ablation_rd.py --check-admission \\
    --probe-results-dir /home/nvidia/embed_ablation/results/probes \\
    --intended-steps 2000 --intended-batch 64
```

This must exit 0. It checks, per probe: (a) the last three T=1 evals are
monotone non-increasing (a basic "is this actually learning" check), and
(b) the measured rate extrapolated to 2000 steps/batch=64 does not exceed
2.0 GPU-h/cell (audit M4). If it exits nonzero, STOP -- re-derive --steps
for every phase_B_seeded/ spec's `cmd` (or investigate the failing arm)
before staging any of them. See EMBEDDING_ABLATION_DESIGN.md S6 for the
identical sentence in the pre-registration itself.
"""

README_B = """# Phase B -- seeded cells (0646-0669)

Audit F2 (FATAL) gating fix. **DO NOT QUEUE ANY FILE IN THIS DIRECTORY**
until phase_A_probes/'s 6 probes have completed AND
`embed_ablation_rd.py --check-admission --probe-results-dir
/home/nvidia/embed_ablation/results/probes --intended-steps 2000
--intended-batch 64` has exited 0 (see phase_A_probes/README.md).

If the admission check fails (non-monotone T1 in any arm, or extrapolated
GPU-h/cell > 2.0), every `cmd` below needs its `--steps` (and this
generator's `gpuh()` cost model) re-derived from the ACTUAL measured
probe rate before anything here is trusted or queued -- see
EMBEDDING_ABLATION_DESIGN.md S6's identical gating sentence.

24 cells: 4 arm-configs (matrix, flatp, flatd, flatten) x 2 sizes (S, M)
x 3 seeds. matrix/flatp/flatten feed the pre-registered decisions
(STRENGTHEN-01: matrix vs flatten; STRENGTHEN-04: matrix vs flatp);
flatd is a disclosed, non-gating params-UNMATCHED control.
"""

with open(os.path.join(PHASE_A_DIR, "README.md"), "w") as f:
    f.write(README_A)
with open(os.path.join(PHASE_B_DIR, "README.md"), "w") as f:
    f.write(README_B)
print("wrote phase_A_probes/README.md and phase_B_seeded/README.md")
