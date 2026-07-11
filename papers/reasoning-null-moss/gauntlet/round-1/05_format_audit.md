# Stage 05 — Format & Acceptance Audit

**Paper:** *Three Bounds on a Null: Testing the Link Between Fast-Weight Write
Geometry and In-Context Composition* — MOSS @ COLM 2026 (double-blind, 4pp main).
**Audited:** 2026-07-10, fresh context, verify-not-trust.

## Summary

**0 critical / 1 serious / 3 minor. Nothing blocks submission.** The draft is
**critical-clean**: no broken cross-reference, no unmatched/fabricated number, no
anonymization leak. I independently recomputed every load-bearing headline number
from the named raw artifacts and **all of them match exactly** — the five
CI-excluding-zero cells (and that they are the only 5 of 40), the corpus-level
UNRESOLVED×2 provenance, the transient (det32/det20, fig3 CIs, held-out-hop
corroboration), the full §5 replication battery (variance ratio 4.47, cohort SDs,
threshold 1.382, all three CIs, 7/10 + 7/10 pooled CIs contain zero), the §3 wave
counts (312/312, 0/78, 8/8, 80/80, 16/16, 30/30, 21.8–46.4%/35.9%), the analytic
F(2,8)≈4.46 and t(2,.975)=4.303, and the C0 ladder configs. Both newest bib
entries verify against the arXiv API. The abstract is 217 words (in band). Main
content ends exactly on page 4.

The single serious item is a **traceability-comment hygiene gap**: the body
sections carry `% evidence:` comments diligently, but `main.tex` (Appendix A + all
three figure captions — the most number-dense location in the paper) omits them.
Every number there was independently recomputed and matches, so this is a missing
token, not a fabrication; it does not block on integrity grounds but should be
closed.

---

## Critical

None.

---

## Serious

### S1 — `main.tex` Appendix A and all three figure captions carry numerical claims with no `% evidence:` comment

**File / location:** `main.tex` — Appendix A (lines ~76, 97–111, 119) and the three
`\caption{...}` blocks (lines ~132–138, 145–152, 159–172). The **only** evidence
comment in `main.tex` is `C9-compute` (line 187); everything else in the appendix
and captions is uncommented.

**The problem:** The paper's core integrity rule ("every numerical claim maps to an
evidence row") is enforced in every `sections/*.tex` file but not in `main.tex`,
whose appendix is where the most precise numbers live. Uncommented numerical claims
include:

- **App A, the 40-interval enumeration** (lines 97–111): the five excluding-zero
  cells with full CIs — reasoning-dense×global@1000 `−0.203 [−0.402,−0.004]` /
  `−0.199 [−0.325,−0.074]`; encyclopedic×per-token@1000 `−0.168 [−0.301,−0.036]` /
  `−0.109 [−0.211,−0.006]`; encyclopedic×per-token@2500 `−0.500 [−0.624,−0.376]`;
  plus `t(2,.975)=4.303` (line 76), the `4.0` variance cutoff and `F(2,8)≈4.46`
  (line 119).
- **Fig 1 caption:** "All 90 cells".
- **Fig 2 caption:** "21.8 to 46.4 percent"; "0.0 at all 30 readings"; "0.10 …
  floor".
- **Fig 3 caption:** `−0.500 [−0.624,−0.376]`; `−0.252 [−0.920,+0.416]`;
  `−0.795 [−2.513,+0.923]`; `[−0.506,+0.357]`; `4.47` vs `4.0`; `≈4.46`.

**Why serious and not critical:** I recomputed **every one** of these against the
raw artifacts the corresponding brief rows name, and they all match exactly (see
Recomputation ledger below). The claim-ids all exist and point to raw JSON. So the
numbers trace and match — the `% evidence:` token is simply not repeated at the
appendix/caption loci. The rendered PDF never shows comments, so a reviewer is not
misled; this is an internal-convention/process gap, not an untraceable or
fabricated number (the two conditions the CRITICAL gate exists to catch).

**Fix:** Add the comments at these loci — `C6-transient` for the 40-interval
enumeration and the fig-3 CIs; `C5-power` for `t(2,.975)`; `C7-replication` for
`F(2,8)`, the `4.0`/`4.47` variance figures, and `[−0.506,+0.357]`; `C4-dissoc`
for the 30-readings and 21.8–46.4% caption numbers; `C1/C2/C3/C4` for the 90-cell
total.

---

## Minor

### M1 — "14M to 1.31B parameters" repeated in abstract + intro without an inline evidence comment
`sections/00_abstract.tex:7` and `sections/01_introduction.tex:19` state the model
scale with no adjacent `% evidence:` comment. The canonical claim locus,
`sections/02_setting.tex:11–12`, carries `% evidence: C0-ladder`, and I verified the
ladder configs against the raw `ckpt_config` fields (state dim 64/64/128/128, depth
2/12/16/22, d_model up to 2560 — all match). Fully traceable; consider echoing
`C0-ladder` at the two summary loci for consistency. Not blocking.

### M2 — Six section labels defined but never referenced
`sec:geometric`, `sec:behavioral`, `sec:replication`, `sec:related`, `sec:scope`,
`app:repro` are `\label`'d but never `\ref`'d. Harmless (standard practice to label
all sections); no dangling references exist. No action required.

### M3 — Bundle-hygiene note (for the later bundle stage, not a source-tree finding)
`sections/` contains 8 committed markdown drafts (`00_abstract.md` …
`07_discussion.md`) that are drafting artifacts, not part of the LaTeX build; the
`bundle/` directory is currently empty. Exclude the `.md` drafts from the
submission archive when the bundle is assembled.

---

## Recomputation ledger (all MATCH the draft)

**Five CI-excluding-zero cells (§4 / App A), from `2026-07-08_phase2b/results/trajectory_{openr1,wikitext}-mix-ext_phase2b.json`:** exactly 5 of the 40 intervals
exclude zero, and they are the five named — openr1×global@1000 K32 `−0.2030
[−0.4025,−0.0036]`, K20 `−0.1992 [−0.3247,−0.0737]`; wikitext×per_token@1000 K32
`−0.1683 [−0.3006,−0.0361]`, K20 `−0.1089 [−0.2114,−0.0064]`;
wikitext×per_token@2500 K32 `−0.5000 [−0.6241,−0.3758]`. No others. ✓

**Corpus-level provenance:** `PHASE2B_SUMMARY.json.trajectories` → both
`openr1-mix-ext` and `wikitext-mix-ext` = `UNRESOLVED`. ✓ (matches "returns both
corpora unresolved").

**Transient (C6):** wikitext×per_token raw — c=2500 `det32=True, det20=False`,
`holds_by_c[2500]=True`; c=1000 both det True (→ fails differential condition, as
the draft states); fig-3 values c=2500 K20 `−0.2518 [−0.9200,+0.4164]` and c=5000
K32 `−0.7949 [−2.5128,+0.9230]` match. Held-out-hop `secondary_ood.per_token[2500]`
= `det32=True, mean −0.5261 [−0.6483,−0.4039]` (harm direction, same cell/ckpt). ✓

**Replication (§5), from `2026-07-08_phase2b_seedext/results/trajectory_seedext_wikitext_n12.json` (primary/2500/delta_k32):** `sd_old=1.1543` (→1.154),
`sd_new=0.5459` (→0.546), ratio 1.154/0.546=2.11 (→"2.1 times"),
`var_ratio=4.4715` (→4.47), `mean_shift=0.3637` (→0.364), `2×pooled_se`=2×0.69084
=1.382, old CI `[−0.6241,−0.3758]`, new(n=9) CI `[−0.5060,+0.3569]`, `pooled=null`,
`outcome=BATCH-EFFECT-FLAGGED`. Naive 12-seed pool recomputed from the 12 deltas
(t₁₁=2.201): `[−0.5090,+0.1472]` (→[−0.509,+0.147]). Computable pooled pairs:
primary 7/10, held-out 7/10, all 14 contain zero. ✓

**§3 wave counts:** Phase 1 = 78 leg files (60 leg_a intervention + 18 leg_b
ladder), 312/312 `recovered_frac==0.0`, premise-iii pass 0/78, premise-iv pass
0/78, premise-iii median span `[−0.327,+0.763]` (→[−0.33,0.76]). Rung-3 = 2 cells,
8/8 zeros, both premise gates false both corpora; ladder series 72(leg_b)+8=80/80.
Phase-1b = 4 `stage0_natural_*` files, 16/16 zeros, `gate_result_h1_probe_valid`
False 4/4, `STAGE0_GATE_REFUSED` sentinel present. Phase-2 familiarization = 30
`stage05_gate_*` files, 30/30 `recovered_frac==0.0` (`n_scored=512` each); L_query
fall from step-250 to step-5000 over the 6 `off_*` cells = [41.2, 21.8, 43.1, 32.4,
46.4, 30.4]% → range [21.8%, 46.4%], mean 35.9%. ✓

**Distributional constants:** F₀.₉₅(2,8) analytic = 4.459 (→≈4.46, via closed form
`(1+x/4)⁴=20`); t₀.₉₇₅(2) = 4.3027 (→4.303). ✓

**C9 compute:** `PHASE2B_SUMMARY` `elapsed_s=2833, n_gpus=2`; `PHASE2B_SEEDEXT_SUMMARY`
`elapsed_s=4119, n_gpus=2`; brief per-wave sum = 9.511 → paper's "approximately 9.5
GPU-hours" is consistent and traces to C9. ✓

**arXiv (two newest bib entries):** `2411.12537` → "Unlocking State-Tracking in
Linear RNNs Through Negative Eigenvalues", authors Grazzi/Siems/Zela/Franke/Hutter/
Pontil — matches bib exactly. `2404.08819` → "The Illusion of State in State-Space
Models", authors Merrill/Petty/Sabharwal — matches bib exactly. ✓

---

## Counts

- **Body word count:** ≈2,001 words (sections 1–7 + abstract, LaTeX-stripped
  estimate). Rendered main content (§§1–7) ends on **page 4**; References start
  page 5; Appendices A–C span pages 5–6 (8 pages total incl. appendix). Within the
  venue limit (4pp main content + unlimited supplementary). Render-inspector stage
  is the authoritative page count.
- **Abstract length:** 217 words (band 200–230). ✓
- **Figures:** 3 referenced (`fig:gates`, `fig:dissoc`, `fig:transient`) vs 3
  present (`fig1_validity_gates.pdf`, `fig2_dissociation.pdf`,
  `fig3_transient_replication.pdf`), all `\includegraphics`'d. `figure-gen.py` is a
  build script, not an orphan figure. No orphans.
- **Citations:** 11 unique in-text vs 11 bib entries; **0 orphans either
  direction**. Entry types sensible (9 `@article` arXiv preprints; `bertinetto2021`
  `@inproceedings` with booktitle; `forde2020` `@proceedings` with editor/volume).
- **Cross-refs:** all `\ref` resolve; **0 broken**; **0 `??`** in `main.pdf`. 6
  section labels unused (M2, harmless).
- **Anonymization matches:** **0** across `main.tex`, `sections/*.tex`, `refs.bib`,
  and figure-PDF `strings`/`pdftotext` (figure metadata shows only "Matplotlib
  v3.9.4" as Creator/Producer — not an identity leak).
- **Banned-word hits:** **0** (full 17-word list, whole-word, case-insensitive over
  rendered prose).
- **Literal placeholders:** **0** (TODO/pending/forthcoming/[CITE]/Table X/Figure
  Y/`\textcolor{red}` — none).

**Termination contribution:** format audit is **critical-clean**. No CRITICAL
format finding. S1 is a comment-hygiene gap with every number independently
verified correct; it is recommended-fix, not a blocker.
