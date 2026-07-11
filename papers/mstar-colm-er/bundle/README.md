# Submission bundle — Constant-Memory Recall (COLM 2026 Efficient Reasoning workshop)

Anonymized submission package. Contents:

- `mstar-colm-er-submission.pdf` — the compiled paper (identical to the
  reviewed render: 10 pages total; main text ends on page 8, inside the
  workshop's 4-10 main-text-page limit, references excluded).
- `mstar-colm-er-submission.tex` — single flattened LaTeX source (all
  section files inlined).
- `colm2026_conference.sty` / `.bst`, `math_commands.tex`, `fancyhdr.sty`,
  `natbib.sty` — the official COLM 2026 kit (release tag `2026`), unmodified.
- `refs.bib` — bibliography; every entry carries an arXiv id or DOI and was
  checked against the arXiv API or CrossRef before submission.
- `figures/fig1_horizon.pdf`, `figures/fig2_szero.pdf`,
  `figures/fig3_traincurve.pdf` — the three figures.
- `figures/figure-gen.py` — the single versioned script that generates all
  three figures AND every table cell (`figures/tables_generated.tex`) from
  archived evaluation artifacts. It asserts the MD5 checksum of each source
  file before loading it; a changed artifact fails the build rather than
  silently changing a number.
- `figures/tables_generated.tex` — the auto-generated table blocks
  (`\tableone`, `\tablecapped`, `\tabletaps`, `\tabletasktwo`,
  `\tablekstress`); never hand-edited.

## Reproducing the numbers

Every inline numerical claim in the LaTeX source carries a machine-checkable
comment of the form `% <!-- evidence: Cn -->` naming its row in a
claims-to-evidence map maintained alongside the draft. Each row names the raw
evaluation artifact (a results JSON under an `experiment-runs/` archive, by
relative path) and its MD5. To regenerate every figure and table:

```
python figures/figure-gen.py --repo <archive-root> --out figures/
```

where `<archive-root>` contains the `experiment-runs/` directory. The script
fails loudly on any checksum mismatch. The archived artifacts, the full
claims-to-evidence map, and the experiment scripts will be released with the
de-anonymized version of this paper (the workshop is non-archival and
double-blind; no identifying link is included here).

## Compiling from source

```
tectonic mstar-colm-er-submission.tex   # run three times, or use latexmk
```

(pdflatex + bibtex also works: pdflatex, bibtex, pdflatex, pdflatex.)
