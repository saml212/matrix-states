# Bundle — capacity-colm-er (COLM 2026 Efficient Reasoning workshop)

Anonymized double-blind submission build.

## Build

From this directory:

```
tectonic --keep-intermediates main.tex
```

Sections are included from `../sections/*.tex` (single source of truth);
figures from `../figures/` (PDFs produced by `../figures/figure-gen.py`,
which asserts the md5 of all 48 archived raw files it loads before
plotting; see the manifest at the top of that script and the
claims-to-evidence map in `../brief.md`).

## Template provenance

Official COLM 2026 kit, GitHub `COLM-org/Template`, release tag `2026`
(published 2025-12-08), fetched live 2026-07-10; see
`../venue-requirements.md`. Files: `colm2026_conference.sty`,
`colm2026_conference.bst`, `fancyhdr.sty`, `natbib.sty`. The
`[submission]` option anonymizes the author block and enables line
numbers.

## Citations

Every `refs.bib` entry was verified programmatically against the arXiv
API (or CrossRef for Welch 1974) on 2026-07-10; the verification record
is `citation-verification.json` in this directory. Two author-list errors
inherited from the base draft's bib were caught and corrected
(`nazari2026rank`, `sun2026staterank`).

## Reproducibility

Figure source md5s: asserted in `../figures/figure-gen.py` (MANIFEST
dict, 48 files). Key artifacts:

- `experiment-runs/2026-07-06_keyanchor_cliff/fit_cliff_curve_results.json`
  md5:c4e233fe3c7c6ec246c6e92e35134258
- `experiment-runs/2026-07-06_keyanchor_dstate/fit_cliff_curve_d128_results.json`
  md5:0801d804c89e739da9712e9beb24b50f
- `experiment-runs/2026-07-07_keyanchor_scaling_wide/fits/fit_cliff_curve_d80_refit_results.json`
  md5:05dd2f9e79747f871756201165fdd548
- `experiment-runs/2026-07-08_c17_repro/fits/fit_d96_unlocked_results.json`
  md5:61eaffe1744a56086af2f4115f9a9cf4
- `experiment-runs/2026-07-08_d96_scatter_resolution_design/fits/sim_d96_scatter_resolution_power_results.json`
  md5:fbd8342d266989ab9112b14997d69196
- `experiment-runs/2026-07-07_keyanchor_scaling/fit_cliff_curve_d96_results.json`
  md5:8f48aaddf77df460cf0f241b9232a559
