# Bundle — reasoning-null-moss (MOSS @ COLM 2026, Small-Scale Frontier Track, late window)

Anonymized double-blind submission build. `reasoning-null-moss-submission.tex`
is the flattened single-file source (sections inlined at bundle time from
`../sections/*.tex`, the single source of truth);
`reasoning-null-moss-submission.pdf` is the compiled artifact (10 pages: 4 main
content + references + appendices A--C, the appendices expanded by the
2026-07-11 instrument-re-verification revision).

## Build

From this directory (the bundle is self-contained):

```
tectonic --keep-intermediates reasoning-null-moss-submission.tex
```

Figures load from `figures/` (PDFs produced by `../figures/figure-gen.py`,
which asserts the md5 of every archived raw file it loads before plotting;
see the MANIFEST dict at the top of that script and the claims-to-evidence
map in `../brief.md`).

## Template provenance

Official COLM 2026 kit, GitHub `COLM-org/Template`, release tag `2026`,
fetched live 2026-07-10; see `../venue-requirements.md`. Files:
`colm2026_conference.sty`, `colm2026_conference.bst`, `fancyhdr.sty`,
`natbib.sty`, `math_commands.tex`. The `[submission]` option anonymizes the
author block and enables line numbers.

## Citations

Every `refs.bib` entry was verified programmatically on 2026-07-10: arXiv
entries against the arXiv export API
(`export.arxiv.org/api/query?id_list=<id>`), PMLR entries against the
proceedings.mlr.press volume pages. No entry was written from memory. The two
entries added at the gauntlet round (Grazzi et al. 2024, arXiv:2411.12537;
Merrill et al. 2024, arXiv:2404.08819) were re-verified independently by the
round's re-attack and format-audit stages; an author-order error in the
rebuttal's proposed Grazzi entry was caught against the API before it landed.

## Review provenance

The draft passed a five-stage adversarial gauntlet
(`../gauntlet/round-1/`): attack (2 CRITICAL / 4 SERIOUS / 2 MINOR), defense,
rebuttal (8-fix ordered list), a scoped re-attack that closed both CRITICALs
against the raw archives, style (PASS, zero violations), format audit
(critical-clean; every load-bearing number independently recomputed from the
raw artifacts), and three render-inspection passes ending in PASS with zero
findings.

After the round-1 sign-off, the pre-submission real-kernel positive control
(the round-1 FIX-6 requirement) FAILED, exposing a state-layout transpose
defect in the geometric readout; the full instrument re-verification
(`REASONING_LINK_DESIGN.md` §17.1--§17.7) fixed it, re-metricked the
wave-1/wave-2 grid under the corrected instrument, and killed the resulting
apparent signal with two pre-registered correspondence nulls
(TRIVIAL-ARTIFACT, lane re-closes doubly instrument-validated). Bound 1 and
all dependent passages were rewritten to the corrected claim shape
("recovery is null-indistinguishable," not "recovery reads zero") and passed
a targeted round-2 re-gauntlet on the changed material
(`../gauntlet/round-2/`): attack (0 CRITICAL / 4 SERIOUS / 3 MINOR), a
verify-vs-raws defense that closed all seven findings with every cited md5
re-confirmed, style (all violations fixed), format audit (0 critical / 0
serious), and a render-inspection pass ending in PASS (body confirmed within
the 4pp limit).

## Reproducibility

Figure source md5s are asserted in `../figures/figure-gen.py` (MANIFEST
dict). Key artifacts (repo-relative under `experiment-runs/`):

- `2026-07-08_phase2b/results/trajectory_wikitext-mix-ext_phase2b.json`
  md5:5c727ac669aea02601790b4fb1dac8b4
- `2026-07-08_phase2b/results/trajectory_openr1-mix-ext_phase2b.json`
  md5:bdbc2352172b79b206d9d7bf7be303fe
- `2026-07-08_phase2b/results/PHASE2B_SUMMARY.json`
  md5:9f5b14f372187ab181c1981158a23f14
- `2026-07-08_phase2b_seedext/results/trajectory_seedext_wikitext_n12.json`
  md5:989f9997c50973cc299f25d89efff64f
- `2026-07-07_reasoning_link_phase1/results/leg_*.json`
  (78 files, manifest-md5 3c885fbe49ada8cb62afb33f3f6faf60)
- `2026-07-08_phase2_familiarization/` gate + trajectory sets
  (manifest-md5s 85d381404dd89769eda382beacbee673 /
  091c33c4786048cd3f94bb75496465ec)

Bound-1 instrument re-verification (2026-07-11, text-only, no figure):

- `2026-07-11_reasoning_null_poscontrol/results/reasoning_link_poscontrol_result.json`
  md5:71e53be7953812a60004a5ee08d77e10 (pre-fix positive control, 0/256)
- `2026-07-11_reasoning_link_remetric/04_remetric/results/AGGREGATE_SUMMARY.txt`
  md5:34186cbc6771bd4c8c631816ba1d90a5 (78/320 fixed-lens re-metric)
- `2026-07-11_reasoning_link_validation/01_item12_shuffle_resample/results/ANALYSIS_SUMMARY.txt`
  md5:922bfc8b34d622ccdfe88236ac69731d (label-shuffle null, 0/320)
- `2026-07-11_reasoning_link_validation/02_derange_control/results/DERANGE_SUMMARY.txt`
  md5:2edc16daa9c604ae978e6e2d3e853da1 (derangement null, 0/320)

The full map from every claim in the paper to its verdict record and raw
artifact is `../brief.md` § "Claims-to-evidence-to-figure map".

## Submission status

The venue deadline (Jul 3, 2026 AoE) has passed; entry is via the
capacity-gated late window. The required first step — the late-add email to
the organizers (colm2026-moss-workshop [at] googlegroups [dot] com) — is a PI
decision and was NOT sent by this run. This bundle is prepared so the
submission can go out the moment that email receives a yes.
