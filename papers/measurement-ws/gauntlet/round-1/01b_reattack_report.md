# 01b — Scoped re-attack (closure check + hostile read of the FIX-1..FIX-16 wave)

Paper: `papers/measurement-ws/` — "The Instrument Is the First Suspect."
Fix wave under review: commit `adba049` (FIX-1..FIX-16).
Re-attacker: fresh hostile context. Every number below was recomputed from the
raw artifact, not read from prose. All 7 checklist raw artifacts md5-matched
their brief pointers exactly (see per-item notes).

---

## 1. Verdict summary

**Round-1 CRITICAL A1 (the "30 baseline cells" count) — CLOSED.**
The raw `crosscheck_lens_verdict_output.json` (md5 `f26a769d…`, matched)
`flat_per_cell_table` holds **exactly 62 rows = 37 `arm3_beta02` + 25
`arm2_beta01`**. The 25 arm-2 cells are precisely 5 groups {A5,A6,S3,S4,S5} ×
5 seeds {0–4}, all n_h=2. Lens agreement on arm-2 is **exact**: 0 disagreements
on `primary_rf90_far64` vs `crosscheck_rf90_far64`, 0 on `mid32`, 0 on
`ceiling_d8`. §5 now reads "all 25 Arm-2 baseline cells (5 groups × 5 seeds, 50
checkpoint values) the lenses agree exactly." The false 30 is gone everywhere
(grep clean). **A1 is dead.**

**Re-rated CRITICAL A7 (zero probing-methodology citations) — CLOSED.**
`refs.bib` now carries all five MUST-CITEs with correct titles/authors/venues
(`hewitt2019control`, `elazar2021amnesic`, `adebayo2018sanity`,
`belinkov2022probing`, `schaeffer2023emergent`; all render in the PDF, all
resolve). §8 engages them with a real distinguishing claim ("each diagnoses one
instrument class, where this catalogue is a serial, pre-registered,
cross-instrument adjudication record with falsifier teeth"). §4 carries the two
required inline cites: `\citep{hewitt2019control}` at the shuffled-tap control
and `\citep{elazar2021amnesic}` at the state-zeroing experiment. **A7 is dead.**

**No CRITICAL remains open. All 10 checklist items PASS. Zero new CRITICAL or
SERIOUS defects.** Three MINOR nits (B1–B3 below), none touching a headline
claim.

### Per-checklist PASS/FAIL

| # | Fix | Verdict | Evidence (recomputed) |
|---|-----|---------|-----------------------|
| 1 | FIX-1 A1 closure + all bare counts | **PASS** | 62=37+25; arm2=25 (5×5); lenses agree exact (0/0/0). 4608 = 4313@n_iter24 + 295@n_iter28 (Counter on `per_episode_resolve_niter`); 107/107 (`n_reconstructed=n_archived=107`, sym-diff 0); 36 events (`step_neg1.n_distinct_events=36`, min 3 → 12×); 0 violations, 0 disagreements; verdict TOLERANCE-MISCALIBRATION. §3 "one of 12 new admissible" verified from the 16 wide JSONs (only K72 s1741 admissible; K78/84/90 all-False; 11 quarantined). §6 "37 of 39"/"0.17/0.08 fifth (S3)" verified (mean\|dev\|=0.02757, only the S5 k−1 pair beyond 0.07; S3 force-rank arms 0.167/0.075, all others 0.000). |
| 2 | FIX-2 A7 closure | **PASS** | 5 bib entries correct; §8 distinguishing block; §4 hewitt@shuffled-tap + elazar@state-zeroing inline. |
| 3 | FIX-3 "zeros at every rank" removed | **PASS** | grep: no "zero(s) at every rank" anywhere; §6 body + appendix B2 both read "zero in four of five groups; 0.17 and 0.08 in the fifth"; catalogue row reads "near-chance at every rank" (true, not the false universal). Intro's old "reading zero at every rank" bullet deleted in the compression. |
| 4 | FIX-4 catalogue-in-appendix fallback | **PASS (disposition sound)** | Body = **exactly 4 pages**: §related label resolves to page 4, references on page 5 (visually confirmed), appendix labels on pages 6–8. Body-only read is coherent — §3/§4/§5 = Cases I–III, §6 = the three brief lenses; all six incidents carried by §-structure. Source comment in `01_intro.tex` records the ≥5pp reversal condition. Forward-ref to appendix Table 1 from the intro is acceptable. |
| 5 | FIX-5 S5 causal-outlier claim | **PASS** | grep: no "because one seed"/"dragged below". §7 + `tab:remetric` caption both say the group misses its bar "with or without the outlier." Arithmetic verified: S5 nh4 triad far64 = 0.80/0.00/0.65 (mean 0.4833 → 0.483), ceilings 1.0/0.45/1.0, bar 0.735; siblings 80%/65% both < 90%; without the 0.00 seed mean = 0.725 < 0.735. |
| 6 | FIX-7 abstract + §2 assembly | **PASS** | grep: "one and the same" gone; abstract now "assembled across the catalogue and in full force before the final, decisive one." §2 names both mid-catalogue clauses (Rule 3 positive-control → §wronglayer/Case II; Rule 2 graded crosscheck → Appendix/covariance) and states "all five rules were in force before the decisive adjudication of §gauge." |
| 7 | FIX-9 cliff anchor + B2 softening | **PASS** | §3 anchors the cliff anomaly to "the same grid's own post-fix re-fit below" (in-paper) with `\citep{companion2026capacity}` as corroboration. Appendix B2 punchline softened: "a corrected-target, zero-padded re-run subsequently delivered the test (companion work, anonymized)." |
| 8 | FIX-11 same verdict/reason + S3 bar | **PASS** | `stage2_harvest_report.json` (primary) and `crosscheck_lens_verdict_output.json` (crosscheck) carry **byte-identical** verdict `FALSIFY` and identical reason string. Primary S3: far 0.54, bar 0.459, clears=true → §5 "S3 cleared its bar, 0.54 against 0.459" exact. grep: "no signal anywhere" gone. |
| 9 | FIX-12 threshold-robust clause | **PASS** | grep: no "identical for any cut"/"cut-invariant"/"13-cell". §5 reads "every cell with disagreement ≥0.5 is converged (13); threshold-robust: disagreement >0.3 → converged, 16 of 16." Recomputed: at cuts 0.3/0.4/0.5/0.6/0.7 the counts are **16/14/13/12/11** (≥ semantics), all converged (final_loss<0.02) at every cut; the attacker's false invariant-count claim is not present. |
| 10 | FIX-10/13/14/15 economics + scope | **PASS** | 0.50 = `chain_span_s` 1782.3 → `chain_span_gpu_h` 0.4951 (raw). 1.10 = `wrapper_wall_s` 2608.7 (K84_s1943) + 1343.5 (K90_s2043) = 3952.2 → 1.098. grep: no unsourced "0.82" (only "0.828" bar values). §8 admits the missing false-alarm denominator. §4 ridge: no "global optimum" (grep clean), no lambda-sweep claim (raw shows λ=100 fixed; over-regularization killed by the tap-iv 0.674 recovery, not an optimality claim). `tab:remetric` caption states the separation rule ("clear bar AND baseline collapses below 50% of its ceiling; S3: 0.15>0.125 → no"). Abstract carries the synthetic-oracle qualifier. |

Independent cross-checks beyond the checklist (all clean): Case II tap table
`tab:taps` matches `tap_localization_SUMMARY.json` to the digit (gaps
+0.060/+0.006/+0.063/+0.800 contender, +0.005/+0.002/+0.005/+0.006 ablation;
iv rf@0.9=0.6743→0.674); state-zeroing 0.0286/0.9990/0.9990 exact; converged-cell
count 17 matches Fig. 1 legend (n=17 conv / n=45 non-conv = 62); teeth block
`all_pass=true`, rows 1.000/0.000, 0.800/0.000, 1.000/0.050 match `tab:teeth`;
B1 covariance de-confound verified against `gate1b_recheck.txt` (uncentered→
centered under matched full-Q: 0.705→0.9996; scale-only centered 0.046 vs
uncentered 0.084); all 62 primary ceilings reproduce; PDF has **0 undefined
refs/citations** (all 17 labels + 15 citations resolve in `main.aux`), abstract
= **210 words** (in the 200–230 band).

---

## 2. New attacks introduced by the rewrite

None reach SERIOUS. Three MINOR items:

### B1: Broken noun phrase in the abstract (compression left it standing)
**Severity:** MINOR | **Type:** style/readability
**Attack.** Abstract, line: "a probe reading a layer causal state-zeroing later
proved inert." The relative pronoun is missing — the intended sense is "a probe
reading a layer *that* causal state-zeroing later proved inert." As written it
is a garden-path fragment that a copy-desk reviewer will circle.
**Evidence.** Pre-existing (present before `adba049`; the abstract diff did not
touch this clause), so the compression pass had the chance to fix it and did
not. Same construction survives in the intro-era phrasing.
**Defuse.** Insert "that": "…a probe reading a layer that causal state-zeroing
later proved inert." One-word edit.

### B2: Intro's "six apparent headline model failures" lost the synthetic-oracle qualifier the abstract gained
**Severity:** MINOR | **Type:** claim-scope
**Attack.** FIX (abstract) now honestly says "a model, **in one case a synthetic
oracle**, appeared to fail." The intro was rewritten in the same commit but
still reads "six apparent **headline model failures**" with no qualifier — yet
incident 4 (the uncentered-covariance case) is a *synthetic perfect model*
failing an acceptance test, and incident 6 (the transposed convention) is an
internal cross-check disagreement, neither a "headline model failure." The
abstract and intro now disagree on scope for the same six.
**Evidence.** `00_abstract.tex` (qualifier present) vs `01_intro.tex` line
"six apparent headline model failures traced…" (qualifier absent); catalogue
row 4 "Perfect synthetic model fails acceptance."
**Defuse.** Mirror the abstract's qualifier in the intro, or drop "headline"
(e.g. "six apparent model failures"). The body/appendix already disclose the
synthetic nature, so this is honesty-of-framing, not a false claim.

### B3: Brief evidence-map pointer for the §5 ground-3 numbers is imprecise (brief-only, not a draft defect)
**Severity:** MINOR | **Type:** number-provenance hygiene
**Attack.** §5's basis-brittleness ground ("per-seed mean cosine 0.03 to 0.69
where the crosscheck read 0.86 to 0.95") is **correct** — I recomputed it:
`results/S4__unconstrained__seed{0–4}.json` give primary `mean_cos`
0.028/0.694/0.330/0.479/0.112 (range 0.03–0.69) and `crosscheck_mean_cos`
0.864–0.952 (0.86–0.95). But the brief row X3 points ground 3 only at the
aggregate `harvest_summary.json` (md5 `7dce…`), which does **not** contain these
per-seed values (it holds per-arm cos/xcos and effective-rank means). A reviewer
auditing the brief map would fail to locate the number in the cited file.
**Evidence.** `harvest_summary.json` m1/m3 blocks (no per-seed S4 cos);
per-seed data lives in the sibling `results/` JSONs of the same harvest dir.
**Defuse.** Repoint the X3 ground-3 evidence row to
`.../2026-07-09_capability_sweep_harvest/results/S4__unconstrained__seed*.json`.
No change to the paper text — the printed numbers are right.

---

## 3. Items checked and found clean

- **All 7 checklist raw artifacts md5-matched** their brief pointers (no
  tampered/substituted files).
- **Every headline number recomputed from raw** (Cases I/II/III, the three brief
  lenses, economics) — no mismatch, no prose-only number surfaced in the fix
  wave.
- **FALSIFY verdict + reason string byte-identical** across primary and
  crosscheck harvests — the "same machine-emitted reason" claim holds.
- **Teeth control** `all_pass=true`; the three pre-registered checkpoints and
  their real/shuffled reads match `tab:teeth` exactly; falsifier does not fire.
- **`tab:remetric`** all five rows (ceilings/far/bar/clears/baseline/separates)
  reproduce the crosscheck per-group block exactly; separation-rule caption
  arithmetic (S3 0.15 > 0.125) correct.
- **LaTeX integrity:** 17 labels + 15 citations all resolve, no `??` in the
  rendered PDF, body exactly 4 pp, references start p.5, appendix pp.6–8,
  abstract 210 w. The committed `main.pdf` (117,981 B) matches the post-fix
  source (built inside `adba049`).
- **New citations** are real papers with correct metadata (Hewitt & Liang 2019
  EMNLP; Elazar et al. 2021 TACL; Adebayo et al. 2018 NeurIPS; Belinkov 2022
  Comp. Ling.; Schaeffer et al. 2023 NeurIPS).

**Recommendation:** the CRITICAL closures (A1, A7) hold against the raws; the
paper clears the gauntlet's termination gate. B1/B2 are one-line copy edits that
should land before submission; B3 is a brief-map pointer fix with no paper-text
impact.
