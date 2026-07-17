# ICLR-2027 scaling paper — ship punch-list (ship-readiness review 2026-07-17)

Science RESOLVED (numbers final, 13 figures generated + provenance-traced,
abstract real, honest-disclosure clean at every layer — the 0.44-vs-0.5 miss
foregrounded, never spun). Blocker is WRITING, not research. Zero GPU-h.
Honest total: **4–6 agent-days** (prose drafting dominates). Venue: ICLR full
track (late-Sept) — content volume can't fit a 4–8pp workshop without gutting
the evidence chain; arXiv preprint once items 1–9 done is the interim marker.
Template blocker (iclr2027_conference.sty doesn't exist yet) gates only
camera-ready, NOT the writing.

## BLOCKERS
1. **Draft body prose** — convert 11 itemize-bullet sections (03–10) to
   connected paragraphs; compress ~14.3k words to 9–10pp main; push
   wave-by-wave detail to appendix WITHOUT losing the causal chain the
   disclosure story needs. **2–3 agent-days — the one judgment-call item,
   the real risk to any timeline.**
2. **Wire 11 orphaned figures** — only fig_cliff/fig_dose inserted; fig01–fig11
   (incl. headline fig11_capacity_curve) generated on disk but never
   `\includegraphics`'d, yet prose already cites Fig.~2–11 (9 dangling
   callouts). Mechanical: match assets to callouts, captions, subfigure
   packing. 0.5–1 day.
3. **refs.bib + `\cite{}` conversion** — ~15 inline-prose citations, no BibTeX
   machinery at all. Most IDs already VERIFIED in-repo. 0.25–0.5 day.
4. **Appendix A2 tables** — currently a 5-item TODO stub; populate from named
   archived JSONs (no hand-transcription). 0.5 day.
5. **Anonymization leak** `sections/02_related_work.tex:105-106` — literal
   "matrix-thinking/chapter2/" in rendered body → anonymous self-cite. <1h.
6. **Scrub REPRODUCIBILITY.md** — names youthful-indigo-turkey (line 65, a
   documented-but-unexecuted TODO). <1h.
7. **LaTeX source comments** name internal design docs — don't render, but
   arXiv keeps source; strip pre-upload or upload PDF-only. <1h.
8. **Verify Barnfield et al. arXiv:2605.05189** — the ONE citation with no
   in-repo verification trail (cited in 3 docs); live fetch. 0.25–0.5h.
9. **Close M²RNN 2603.14360 "VERIFY SPECIFICS" flag** (§02:51) — paper real,
   but the theorem/table claims attributed to it unchecked vs full text. 0.5–1h.

## POLISH
10. Delete stale `main.tex:20-21` "Track C rung-3 open readout" comment
    (rung-3 1.31B result is resolved in §10).
11. Rename 00_abstract_placeholder.tex (content is real; filename misleads).
12. Drop unused todonotes import.

## Notes
- Compiles clean (24pp) with a stub .sty in scratchpad — LaTeX plumbing sound,
  all 16 ref/label pairs resolve, no missing-figure/syntax errors. Only the
  unavailable venue template + missing refs.bib block a real build.
- kwall (papers/kwall, 13pp) is the SEPARATE, verdict-complete workshop artifact
  — ships independently, faster; add the §B8 Stage-0-FAIL pointer to its §6.

## STATUS UPDATE 2026-07-17 (post-full-draft)
DONE: mechanical blockers (13 figs, refs.bib, anonymization, A2 tables incl.
the added Program F scaling-ladder compute); FULL BODY PROSE §01-§10 (voice
approved, 12,079 body + 4,759 appendix words); static checks clean (0 dangling
refs, bijective cites, balanced envs, no unescaped specials); stale-skeleton
fix-at-scale PARTIAL verdict folded.

REMAINING pre-arXiv (all NON-GPU, deferred to a TeX-capable machine — local
tectonic dylib-broken/sandboxed, no pdflatex local or on-box; do NOT install
multi-GB texlive on the production box for this):
1. **Compile + real page count.** 12,079 body words very likely EXCEEDS
   ICLR's ~9-10pp main-text limit → expect a COMPRESSION PASS is needed
   (push more to appendix). Confirm actual pages on any machine with TeX.
2. **Length trim to venue limit** (informed by #1).
3. **Item-7:** strip ~49 internal design-doc filenames from LaTeX source
   comments before arXiv source upload (or upload PDF-only).
4. **Final honest-disclosure read** end-to-end (the 0.44-vs-0.5 framing).
The CONTENT is draft-complete; these are submission-polish, not research.
kwall (13pp) remains the separate, faster, verdict-complete workshop ship.
