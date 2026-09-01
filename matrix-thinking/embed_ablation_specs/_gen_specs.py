#!/usr/bin/env python3
"""One-off generator for the embed-ablation queue spec JSONs (0640-0661).
Not part of the runner CLI -- run once to (re)produce the specs, then this
file can be deleted or kept for reproducibility. Schema copied from
experiment-runs/2026-08-29_box_final_archive/queue/completed/
005_laneA_probe_K128_s0.json (id, lane, hypothesis, cmd, gpu_h_estimate,
output_dir, validity_check, notes)."""
import json
import os

PY = "/home/nvidia/tdenv/bin/python3"
CODE_DIR = "/home/nvidia/embed_ablation"
CKPT_ROOT = "/data/embed_ablation_ckpts"
OUT_DIR = "/home/nvidia/embed_ablation/results"
DATA_DIR = "/data/deltanet_rd_data"
CORPUS = "wikitext-mix-ext"
HERE = os.path.dirname(os.path.abspath(__file__))

# gpu_h_estimate values: 1.367 GPU-h anchor (exp_d16_v2_SUMMARY.txt: mat_dim=16,
# n_layers=8, T=8, batch=96/GPU, seq=512, 3000 steps, 82.0 min on 1 GPU-equivalent
# since DDP replicates per-GPU work) scaled linearly by (batch/96)*(steps/3000)*
# (params/2,552,788) -- see EMBEDDING_ABLATION_DESIGN.md S4 for the derivation.
SPECS = []

def cell_id(n):
    return f"{640 + n:04d}"

n = 0

def add(name, arm, size, match, seed, steps, batch, ceiling_gpuh, gpu_h_estimate, hyp, note):
    global n
    cid = cell_id(n); n += 1
    out_path = f"{OUT_DIR}/{name}.json"
    ckpt_dir = f"{CKPT_ROOT}/{name}"
    cmd = (
        f"mkdir -p {ckpt_dir} {OUT_DIR} && cd {CODE_DIR} && {PY} embed_ablation_rd.py "
        f"--run-cell --arm {arm} --size {size} --match {match} --seed {seed} "
        f"--steps {steps} --batch-size {batch} --seq-len 512 "
        f"--data-dir {DATA_DIR} --corpus {CORPUS} "
        f"--ckpt-dir {ckpt_dir} --out {out_path} --ceiling-gpuh {ceiling_gpuh}"
    )
    validity = (
        f"{PY} -c \"import json; d=json.load(open('{out_path}')); "
        f"assert d.get('complete') is True; "
        f"assert d.get('steps_completed', 0) >= {steps} - 1; "
        f"assert 'T1' in d.get('final_evals', {{}}) and 'T{8 if steps > 500 else 8}' in d.get('final_evals', {{}})\""
    )
    spec = {
        "id": f"{cid}_embed_ablation_{name}",
        "lane": "embed-ablation",
        "hypothesis": hyp,
        "cmd": cmd,
        "gpu_h_estimate": round(gpu_h_estimate, 3),
        "output_dir": OUT_DIR,
        "validity_check": validity,
        "notes": note,
    }
    SPECS.append((f"{cid}_embed_ablation_{name}.json", spec))


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

# ── Wave 0: rate probes (500 steps) -- confirm real per-step wall time
#    BEFORE committing the seeded-cell budget, mirroring 005's own
#    Phase-0a discipline (see that file's own hypothesis field). ─────────
add("probe_matrix_S", "matrix", "S", "P", 0, 500, 64, 0.5, 0.1468,
    "Rate probe only: measures real per-step wall time for the matrix arm at size S "
    "(mat_dim=16, n_layers=6) on the box BEFORE the seeded-cell budget commits, mirroring "
    "005_laneA_probe_K128_s0's own Phase-0a discipline. Not itself a trainability readout; "
    "its measured GPU-h/step re-derives the seeded cells' step counts if the formula "
    "estimate (1.367 GPU-h anchor from exp_d16_v2, scaled by batch/steps/params) is off.",
    PREREQ_NOTE + " Formula-extrapolated estimate; --ceiling-gpuh 0.5 hard-bounds worst case.")

add("probe_flat_S", "flat", "S", "P", 0, 500, 64, 0.5, 0.1469,
    "Rate probe for the flat-P arm at size S (d_model solved to match matrix-S's "
    "2,466,562 params to within 0.06%, d_model=24). Same purpose as probe_matrix_S: "
    "confirm wall time before committing the seeded budget -- the flat arm's per-step "
    "cost is the untested unknown here (standard nn.MultiheadAttention/FFN at a solved, "
    "non-power-of-two d_model has not been timed on this box before).",
    PREREQ_NOTE + " Formula-extrapolated estimate; --ceiling-gpuh 0.5 hard-bounds worst case.")

add("probe_matrix_M", "matrix", "M", "P", 0, 500, 64, 0.5, 0.2235,
    "Rate probe for the matrix arm at size M (mat_dim=24, n_layers=8, 3,755,808 params). "
    "Same Phase-0a purpose as probe_matrix_S, at the larger registered size.",
    PREREQ_NOTE + " Formula-extrapolated estimate; --ceiling-gpuh 0.5 hard-bounds worst case.")

add("probe_flat_M", "flat", "M", "P", 0, 500, 64, 0.5, 0.2241,
    "Rate probe for the flat-P arm at size M (d_model=36, matched to matrix-M's "
    "3,755,808 params to within 0.25%). Same Phase-0a purpose as the other three probes.",
    PREREQ_NOTE + " Formula-extrapolated estimate; --ceiling-gpuh 0.5 hard-bounds worst case.")

# ── Wave 1: seeded cells, size S (mat_dim=16, n_layers=6), 3 seeds each ──
S_MATRIX_GPUH = 0.5871
S_FLATP_GPUH = 0.5874
S_FLATD_GPUH = 0.7877
M_MATRIX_GPUH = 0.8940
M_FLATP_GPUH = 0.8963
M_FLATD_GPUH = 1.2081

VERDICT_HYP_MATRIX_S = (
    "Verdict-carrier cell (matrix arm, size S). Outer-product embedding + RowThenCol "
    "matrix ops, mat_dim=16, n_layers=6, T=8 shared-weight iterations, GPT-2-tokenized "
    "wikitext-mix-ext. Reports T=1 and T=8 token-BPB; harvested against the size-S "
    "flat-P cells (0647-0649) under EMBEDDING_ABLATION_DESIGN.md's pre-registered "
    "decision rule. Fixes Run 22's incompleteness (its flat arm died at step ~2800, "
    "never reaching a comparable finished T=1/T=8 pair) by running both arms to full "
    "completion or an explicit CEILING_STOP, never silently partial."
)
VERDICT_HYP_FLATP_S = (
    "Verdict-carrier cell (flat-P arm, size S). Direct (V, d_model=24) embedding table "
    "+ standard nn.MultiheadAttention/FFN, params matched to matrix-S (0644-0646) to "
    "within 0.06% (gate gets negative-tested in --selftest; a >1% mismatch here would "
    "raise before training starts, not silently under-report). Unlike Run 22's flat arm "
    "(2.2x MORE params than its matrix twin, d_model fixed at mat_dim^2=256), this run's "
    "d_model is solved for total-param equality, closing the historical asymmetry."
)
VERDICT_HYP_MATRIX_M = VERDICT_HYP_MATRIX_S.replace("size S", "size M").replace(
    "mat_dim=16, n_layers=6", "mat_dim=24, n_layers=8").replace("0647-0649", "0656-0658")
VERDICT_HYP_FLATP_M = VERDICT_HYP_FLATP_S.replace("size S", "size M").replace(
    "d_model=24", "d_model=36").replace("0644-0646", "0653-0655")

DISCLOSED_HYP_FLATD_S = (
    "Disclosed control cell (flat-D arm, size S), PRE-REGISTERED as params-UNMATCHED "
    "(d_model=2*mat_dim=32 -- the 'natural' width giving the flat embedding the same "
    "per-token free-parameter count as the matrix embedding's (u,v) pair, NOT total-"
    "param parity). Ratio vs matrix-S is ~1.34x (flat has MORE params), matching the "
    "historical Round-1/Run-22 shape of comparison for continuity. NOT the verdict "
    "carrier -- EMBEDDING_ABLATION_DESIGN.md S5 names flat-P (0647-0649) as that. "
    "Reported alongside the primary result so a reader can see how much the +1%-gated "
    "matching (vs the old un-gated reshape-parity choice) actually moves the outcome."
)
DISCLOSED_HYP_FLATD_M = DISCLOSED_HYP_FLATD_S.replace("size S", "size M").replace(
    "mat_dim=24, size S", "mat_dim=24, size M").replace("d_model=2*mat_dim=32", "d_model=2*mat_dim=48").replace(
    "0647-0649", "0656-0658")

for seed in (0, 1, 2):
    add(f"matrix_S_s{seed}", "matrix", "S", "P", seed, 2000, 64, 2.0, S_MATRIX_GPUH,
        VERDICT_HYP_MATRIX_S, PREREQ_NOTE + " Formula-extrapolated (anchor: exp_d16_v2_"
        "SUMMARY.txt, 1.367 GPU-h @ mat_dim=16/L=8/T=8/batch=96/steps=3000/params=2.55M, "
        "scaled by batch=64/steps=2000/params=2.47M). --ceiling-gpuh 2.0 hard-bounds worst "
        "case per EMBEDDING_ABLATION_DESIGN.md's <=2 GPU-h/cell requirement.")

for seed in (0, 1, 2):
    add(f"flatp_S_s{seed}", "flat", "S", "P", seed, 2000, 64, 2.0, S_FLATP_GPUH,
        VERDICT_HYP_FLATP_S, PREREQ_NOTE + " Formula-extrapolated, same basis as "
        "matrix_S cells (flat-P's backbone FLOPs land close to matrix's own, since "
        "d_model=24 is solved rather than fixed at mat_dim^2 -- see design doc S4, "
        "this is the key upshot that makes flat-P's wall time NOT the historical "
        "8-128x blowup). --ceiling-gpuh 2.0 hard-bounds worst case.")

for seed in (0, 1, 2):
    add(f"flatd_S_s{seed}", "flat", "S", "D", seed, 2000, 64, 2.0, S_FLATD_GPUH,
        DISCLOSED_HYP_FLATD_S, PREREQ_NOTE + " Formula-extrapolated; flat-D's larger "
        "d_model=32 (vs flat-P's 24) makes this the single most expensive size-S cell "
        "(~0.79 GPU-h formula estimate) -- --ceiling-gpuh 2.0 still hard-bounds it with "
        ">2.5x headroom.")

for seed in (0, 1, 2):
    add(f"matrix_M_s{seed}", "matrix", "M", "P", seed, 2000, 64, 2.0, M_MATRIX_GPUH,
        VERDICT_HYP_MATRIX_M, PREREQ_NOTE + " Formula-extrapolated, same anchor as the "
        "size-S matrix cells, scaled to size M's 3,755,808 params. --ceiling-gpuh 2.0 "
        "hard-bounds worst case.")

for seed in (0, 1, 2):
    add(f"flatp_M_s{seed}", "flat", "M", "P", seed, 2000, 64, 2.0, M_FLATP_GPUH,
        VERDICT_HYP_FLATP_M, PREREQ_NOTE + " Formula-extrapolated; d_model=36 solved to "
        "match matrix-M's params to within 0.25%. --ceiling-gpuh 2.0 hard-bounds worst "
        "case.")

for seed in (0, 1, 2):
    add(f"flatd_M_s{seed}", "flat", "M", "D", seed, 2000, 64, 2.0, M_FLATD_GPUH,
        DISCLOSED_HYP_FLATD_M, PREREQ_NOTE + " Formula-extrapolated; the single most "
        "expensive cell in the whole ledger (~1.21 GPU-h formula estimate, d_model=48). "
        "--ceiling-gpuh 2.0 hard-bounds worst case with ~1.65x headroom.")


total_gpu_h = sum(s["gpu_h_estimate"] for _, s in SPECS)
print(f"Generated {len(SPECS)} specs, sum(gpu_h_estimate) = {total_gpu_h:.3f} GPU-h")
assert total_gpu_h <= 30.0, f"ledger over budget: {total_gpu_h} > 30 GPU-h"
assert all(s["gpu_h_estimate"] <= 2.0 for _, s in SPECS), "a cell exceeds the 2 GPU-h/cell cap"

for fname, spec in SPECS:
    path = os.path.join(HERE, fname)
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    print("wrote", fname)
