# Bundle — reasoning-null-moss (MOSS @ COLM 2026, Small-Scale Frontier Track, late window)

Anonymized double-blind submission build. `reasoning-null-moss-submission.tex`
is the flattened single-file source (sections inlined at bundle time from
`../sections/*.tex`, the single source of truth);
`reasoning-null-moss-submission.pdf` is the compiled artifact (8 pages: 4 main
content + references + appendices A--C).

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

The full map from every claim in the paper to its verdict record and raw
artifact is `../brief.md` § "Claims-to-evidence-to-figure map".

## Submission status

The venue deadline (Jul 3, 2026 AoE) has passed; entry is via the
capacity-gated late window. The required first step — the late-add email to
the organizers (colm2026-moss-workshop [at] googlegroups [dot] com) — is a PI
decision and was NOT sent by this run. This bundle is prepared so the
submission can go out the moment that email receives a yes.
