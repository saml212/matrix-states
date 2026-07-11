# 05 — Format Audit (stage 5, the acceptance gate)

**Auditor:** fresh-context format auditor. Cold read of the 11 section files,
the brief's claims-to-evidence map, `references.bib`, `figures/`, and
`venue-requirements.md`. Raw artifacts read read-only from
`experiment-runs/`. Numbers recomputed with `python3` (numpy absent; JSON read
directly). Raw file is the tiebreak, never the prose.

## Verdict

**1 critical / 2 serious / 6 minor.**

**One CRITICAL blocks the gauntlet:** the head-to-head contender's parameter
count in the draft (14,048,896) does not match the raw artifact the R0 row
designates (the R4 contender training JSONs, all three seeds read
`n_params = 14,049,408`). 14,048,896 is the *Section-5 14M ladder rung's*
count (a different model), evidently imported into the contender slot. The
science is unaffected (both arms remain matched within ~0.01%), but the gate is
a number-to-raw-artifact fidelity gate and this headline setup number fails it,
along with its derived "0.004 percent" figure. Fix is a one-line correction
(see C1). Everything else — cross-references, evidence-tag coverage, banned
words, contractions, anonymization of render-visible prose, the abstract band,
the figure/PDF/manifest triangle — is clean. No embedded/injected instructions
were found in any file (security note at the end).

Seven evidence rows (R1, R3, R4, R5, R6, R8, R11) were recomputed one level
deep from the named raw artifacts; six matched exactly, the seventh (R0) is the
CRITICAL. All three deep-checked md5s (R1 harvest, R3 harvest, R3 s3-ext)
matched the brief.

---

## CRITICAL

### C1 — Contender parameter count mismatches its raw artifact (and its derived percentage is wrong)

- **Files/locations:**
  - `04_capability_separation.md:11` — "a two-block DeltaNet-family LM
    ($d_{model}=256$, $d_{state}=64$, **14,048,896 parameters**
    `<!-- evidence: R0 -->`)"
  - `04_capability_separation.md:16` — "(14,048,384 parameters, **a 0.004
    percent difference** `<!-- evidence: R0 -->`)"
  - `10_appendix_b_reproducibility.md:8` — "contender ... **14,048,896 params**;
    ablation 14,048,384 params ... `<!-- evidence: R0 -->`"
- **What fails:** R0's named raw artifact is "`n_params`/config fields in R4's
  27 training JSONs" (brief R0). The contender's task-1 training JSONs read
  `n_params = 14,049,408` in **all three seeds**
  (`experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_contender_task1_sweep_s{0,1,2}.json`).
  The draft says **14,048,896**, a 512-param mismatch. Tracing 14,048,896
  through the archive: it is the count of the **14M ladder rung** (mixcontrol,
  `trajectories_tidy.json`, R6) and the ungated frozen-bias LM runs — a
  *different* model. The contender that carries the R4 WIN is the
  14,049,408-param variant (the same count appears in the gated `attrrob`
  runs). The ablation number (14,048,384) is correct against raw.
- **Downstream number:** "a 0.004 percent difference" is computed from the
  wrong contender count (512 / 14,048,896 = 0.0036%). Against the raw contender
  the ablation gap is 1,024 params, i.e. ~**0.007 percent**.
- **The fix:** set the contender to **14,049,408** in
  `04_capability_separation.md:11`, `:16` and
  `10_appendix_b_reproducibility.md:8`, and update the percentage to **~0.007
  percent**. Leave the Section-5 14M ladder rung at 14,048,896 (that value is
  correct there — do not global-replace). If 14,048,896 is intended (e.g. a
  design-doc pin), the R0 row must be re-pointed to the artifact that actually
  contains it, because the R4 training JSONs do not.
- **Severity rationale:** per the gate's own rule ("a headline number that does
  not match its raw artifact is CRITICAL; the raw file is the tiebreak"), this
  is CRITICAL. It is low-magnitude and fix-the-number (not fix-the-science);
  the coordinator may adjudicate quickly, but it must not ship as written.

---

## SERIOUS

### S1 — Six in-text citations have no `references.bib` entry (documented-deferred, but the bib cannot compile them yet)

- **Locations:** Nichani et al. 2025 (`02_background_setup.md:52`,
  `04_capability_separation.md:64,161`, `06_related_work.md:3`);
  `arXiv:2602.04852` and `arXiv:2602.02195` (`06_related_work.md:30`);
  Kimi Linear `arXiv:2510.26692` (`06_related_work.md:37`); DeltaNet /
  Gated-DeltaNet "Yang et al." (`06_related_work.md:44`); the ICML-2026
  MI-workshop companion "The Gradient Does Not See Rank"
  (`06_related_work.md:56`).
- **What fails:** none of these resolve to a `references.bib` entry. The bib
  header **documents** them as "added at LaTeX-assembly time from the same
  programmatic route," so they are flagged rather than silently missing (this
  satisfies the task's "flagged as needing one at assembly" clause and is why
  this is not CRITICAL). But the central citation of the paper — Nichani-Lee-
  Bietti, on which the argmax caveat and the decode-proof razor both rest — is
  absent, so today's bib cannot compile the draft's core references.
- **The fix:** fetch and add all six (Nichani 2412.06538; 2602.04852;
  2602.02195; Kimi Linear 2510.26692; DeltaNet/Gated-DeltaNet — Yang et al.
  2412.06464 / 2406.06484 as applicable; the ICML-ws companion) via the
  programmatic route before LaTeX assembly, as a hard pre-assembly gate.

### S2 — Qwen3-Next is cited but is neither in the bib nor in the documented deferral list

- **Location:** `06_related_work.md:37` — "(Kimi Linear, Section 4 of
  arXiv:2510.26692; **Qwen3-Next**)".
- **What fails:** Qwen3-Next is named as part of the qk-normalization stability
  line but has no bib entry, no arXiv ID in prose, and — unlike the S1 set — is
  **not** named in the bib header's assembly-time deferral list. It is the one
  citation gap the documented plan does not cover.
- **The fix:** add Qwen3-Next to the assembly-time citation list (technical
  report / model card) or cite it explicitly; or, if it is only an illustrative
  system name, keep Kimi Linear as the cited anchor and drop Qwen3-Next to
  avoid an uncitable reference.

---

## MINOR

- **M1 — Abstract is exactly 230 words (band ceiling, zero slack).**
  `00_abstract.md`, stripping the two HTML comments and the heading, counts 230
  rendered words = the top of the 200–230 band. It passes, but a render that
  splits hyphenated math tokens (`$S_4$-versus-$A_5$`, `32,768-byte`) could tip
  it over. Trim ~5 words for margin.
- **M2 — Figure captions restate numbers without inline evidence tags.**
  Figures 1–7 and A1 captions carry numbers (e.g. `0.99902` to `1.00000`,
  `0.248` to `0.455`, `-0.103`/`0.05σ`, `+4.312`/`1.92σ`, `0.674`, the 84.7%
  window) with no `<!-- evidence: -->` comment. Every caption number duplicates
  a body number that *is* tagged and was verified, so no untraceable number is
  introduced, and LaTeX captions will not carry HTML comments anyway.
  Informational.
- **M3 — Three orphan bib entries (in bib, never cited).**
  `arora2025simplelinearattention` (2402.18668), `gu2024mambalineartime`
  (2312.00752), `ramsauer2021hopfieldnetworks` (2008.02217) are present but
  uncited, labeled "SHOULD-CITE (optional strengtheners)". Either cite or drop
  before camera-ready.
- **M4 — A few derived/config numbers sit just after (not before) their
  paragraph's evidence tag.** `05_pathology_at_scale.md:31-32` "0.003" and
  "0.066" (arithmetic on the R6-tagged 0.4584/0.4554 and 0.455/0.389);
  `:41` noise floor "2.244355" and `:56` "λ = 0.58" (R7/R8/R0 config
  constants). All trace to their rows; a tag-placement tidy-up only.
- **M5 — All bib entries are `@misc` arXiv exports.** Several cited works have
  archival venues (Jelassi, ICML 2024; Xiao, ICLR 2024; Arora Zoology, ICLR
  2024). `@misc` is acceptable under the stated programmatic-fetch policy;
  adding venue fields at camera-ready is optional, not blocking.
- **M6 — Named-build author email present in an HTML comment (not a
  render-visible leak).** `00_abstract.md:7` contains
  `samlarson@pebbleml.com` inside the `<!-- Authors: ... (named build only) -->`
  comment. It is comment-scoped (invisible in render) and build-scoped, so it
  satisfies the "inside a build-scoped HTML comment" clause and is acceptable
  for the arXiv build. Confirm the ICLR assembly drops HTML comments and emits
  anonymous author metadata. Not blocking.

---

## Sample of verified numbers (claim → row → match)

Deep = recomputed from the named raw artifact. 7 rows taken to raw (≥5
required); the rest verified draft↔brief.

| # | Draft number (location) | Row | Brief record | Raw check | Match |
|---|---|---|---|---|---|
| 1 | Spearman ρ=0.9747 (§3.1) | R1 | 0.9747 | — | ✓ |
| 2 | per-group means 1.877/2.852/2.832/3.591/4.736 (§3.1) | R1 | same | **deep**: means 1.877/2.852/2.832/3.591/4.736 from 19 cell JSONs; md5 `7dce77…` matches | ✓ |
| 3 | P(ρ≥0.8)=8/120≈6.67% (§3.1) | R1 | 6.67% | 8/120=6.667% | ✓ |
| 4 | 19/19 cells in band (§3.1) | R1 | 19/19 | — | ✓ |
| 5 | TOST diff 0.0194, se 0.0368, df 7.83, t₁=13.06/t₂=14.12, t_crit 1.865 (§3.2) | R2 | identical | — | ✓ |
| 6 | force-rank k−1=0.000 all 5; S4 .800/.585, A5 .700/.630, S5 .600/.450, A6 .650/.585; S3 .450/.495; S3 4-seed mean .5625 (§3.4) | R3 | same | **deep**: k−1=0.000×5; anchors/bars reproduce exactly; S3 ext mean 0.5625, k−1=0 all 4; md5 `77be9c…`/`d44133…` match | ✓ |
| 7 | 36/39 within 0.07, one at 0.072, two outliers 0.15/0.17 (§3.3) | R1b | 36/39; 0.0716; 0.15/0.17 | — | ✓ (0.0716→0.072) |
| 8 | scale-only cos 0.01–0.22 (§3.3) | R1b | 0.01–0.22 | — | ✓ |
| 9 | contender 0.99951/1.00000/0.99902 mean 0.99951 (§4.2) | R4 | same | **deep**: `leg_a.acc_A` = 0.99951/1.0/0.99902 | ✓ |
| 10 | ablation 0.03223/0.03271/0.03687 mean 0.03394 (§4.2) | R4 | same | **deep**: exact | ✓ |
| 11 | transformer 0.02710/0.02930/0.02856 mean 0.02832 (§4.2) | R4 | same | **deep**: exact | ✓ |
| 12 | Δ(cont−abl) 0.96558 CI (0.95822,0.97293); Δ(cont−tfm) 0.97119 CI (0.96855,0.97383) (§4.2) | R4 | same | — | ✓ |
| 13 | transformer loss 7.81–7.84 @500, 7.45–7.51 @20k (§4.1) | R4a | s0/s1/s2 500→20k | — | ✓ |
| 14 | S₀-zero 0.9990→0.0286; S₁-zero 0.9990; pre-LM-head rf@0.9=0.674, cos 0.894; ablation 0.0/0.119 (§4.3) | R5 | same | **deep**: 0.99902/0.02856/0.99902; rf@0.9 0.67432; cos_mean 0.89396 | ✓ |
| 15 | horizons 454/902/1798, acc≥0.998, 32,768-B state; capped 0.020–0.033, M∈{1,2,4,8,16,32}; CI floor ≥0.958 vs 0.20 (§4.4) | R10/R0 | same | — | ✓ |
| 16 | span 0.248/0.344/0.389/0.455; 84.7%; 0.4584→0.4554 (§5.1) | R6 | same | **deep**: means 0.248/0.344/0.389; rung3 pooled 0.45544 | ✓ |
| 17 | qk-off Δ=−0.103=0.05σ, floor 2.244355; gating +4.312=1.92σ, bar 4.489, p=0.062 (§5.2) | R7 | identical | — | ✓ |
| 18 | 98M +0.1133 (+0.0543,+0.1723) openr1 / +0.1011 (+0.0541,+0.1482) wiki; val-loss 3.2038 vs 3.2020, arm mean 3.1961 (§5.3) | R8 | deltas match; val-loss not spelled in row | **deep**: PRIMARY_delta 0.11332/0.10115, CIs match; `per_token_per_seed[0]`=3.20379, `pass_ceiling`=3.20203, `per_token_mean`=3.19608 | ✓ |
| 19 | task2 pooled interval (−0.020,+0.268); seeds 0.334/0.391/0.479; K48 chance reads (§4.5,§7) | R11 | seeds 0.33447/0.47949/0.39087 | **deep**: `ci_low`=−0.01999, `ci_high`=0.26782; 0.33447 & 0.47949 present | ✓ |
| 20 | c*I residual 0.5–2.9%, τ≥0.9997; K=8 dev 0.03–0.26; K=12 10.22–10.41 vs 11; ρ=−0.973; fD≤3.2e-12 (App A) | R9 | identical | — | ✓ |
| 21 | contender 14,048,896 params; 0.004% gap (§4.1, App B) | R0 | brief says 14,048,896 | **deep**: raw `n_params`=**14,049,408** (3/3 seeds) | ✗ **C1** |

---

## Anonymization grep results

Case-insensitive search across all section files for the brief's identity
surface plus URL/funding patterns:

| Token | Result |
|---|---|
| `Sam Larson` | 0 |
| `samlarson` | 1 — `00_abstract.md:7`, inside the `(named build only)` authors HTML comment |
| `samuellarson` | 0 |
| `pebbleml` | 1 — `00_abstract.md:7`, same named-build comment (`samlarson@pebbleml.com`) |
| `learned-representations` | 0 |
| `youthful-indigo-turkey` | 0 |
| `github.com/` | 0 |
| `huggingface.co/` | 0 |
| `acknowledg` | 0 |
| `funded` | 0 |
| `Berkeley` / `Stanford` | 0 |

**Both matches are inside one build-scoped HTML comment** (invisible in render,
labeled "named build only"). No identity token appears in render-visible prose.
This satisfies the double-blind requirement provided the ICLR assembly strips
HTML comments and uses anonymous author metadata (M6). **No blocking leak.**

The two sanctioned placeholders — `[WORKING TITLE — PI]`
(`00_abstract.md:3-5`) and `[AUTHORS — PI decision pending]`
(`00_abstract.md:6-7`) — are both fully comment-scoped and invisible in render.
The only "placeholder" grep hit ("pending") is inside the authors comment. No
other placeholder (TODO/TBD/forthcoming/will-be-added/[CITE]/Table X/Figure Y)
appears anywhere.

---

## Counts

- **Body word count (§1–8, excl. references & appendix):** ~5,879 words
  (math-collapsed, table rows dropped, captions included). Plausible for the
  ICLR 9-page single-column basis but figure-heavy; the render inspector is the
  authoritative page check.
- **Abstract:** 230 words — within the 200–230 band, at the ceiling (M1).
- **Figures referenced vs present:** 8 captioned (Fig 1–7 + A1) ↔ 8 PDFs in
  `figures/`, names correspond; **0 orphans** either direction. `figure-gen.py`
  `SOURCE_MD5` covers every figure's raw inputs (fig1: 19 rank cells; fig2:
  m3fix + s3ext cells; fig3: 9 remetric + MSTAR; fig4: 2 tap JSONs; fig5:
  trajectories_tidy + rung3; fig6: n3_recompute; fig7: fixscale verdict; figA1:
  complement) — every `_load` path is in the manifest and each `_assert_md5`
  has a recorded checksum.
- **Evidence tags:** every referenced row (R0, R1, R1b, R2, R3, R4, R4a, R5,
  R6, R7, R8, R9, R10, R11) exists in the brief. **0 tags naming a nonexistent
  row.** No render-visible numerical claim in body prose was found without a
  governing evidence tag (caption restatements per M2).
- **Citations in-text vs in-bib:** 11 distinct in-text targets; 7 bib entries
  (4 cited: arora2023, jelassi2024, olsson2022, xiao2024 — years all match
  prose; 3 orphan per M3). Cited-but-absent = 6 documented-deferred (S1) + 1
  undocumented (Qwen3-Next, S2).
- **Cross-references:** ~70 Section/Appendix/Figure/Table mentions; all resolve
  (Sections 1–8, 2.1–2.5, 3.1–3.4, 4.1–4.5, 5.1–5.4, B.1–B.4, Appendices A/B,
  Figures 1–7/A1, Table 1). **0 broken.** (The grep's "Figure R" hit is a false
  match on the heading "Figure Regeneration".)
- **Banned words:** 0 hits across the 17-word styleguide list (whole-word,
  case-insensitive, comments included).
- **Contractions:** 0. **Em-dash-as-pause / rhetorical-question headings:** none
  observed. **First-person "I":** none.
- **Anonymization matches:** 2, both inside one named-build HTML comment (not
  render-visible).

---

## Security note

I read all 11 section files, the brief, `references.bib`, `figure-gen.py`,
`venue-requirements.md`, the styleguide, the role prompt, and the raw JSON
artifacts under `experiment-runs/`. **No embedded or injected instructions were
found** — no fake system-reminder blocks, no date-change claims, no
"file was modified — do not tell the user" concealment strings, no directives
to alter permissions/config. All tool outputs were plain data (numeric JSON
fields and md5 digests). Nothing in any file directed the audit; the only
instructions I followed are the role prompt and the audit task. Had any such
block appeared, I would have disregarded it, verified against git/md5, and
reported it here rather than complying.
