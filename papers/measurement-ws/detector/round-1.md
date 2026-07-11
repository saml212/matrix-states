# Detector gate — round 1 — measurement-ws

Protocol per `../gauntlet/round-1/07_final_review.md` §5 (the
rank-recruitment-ws precedent, all four of its justifications judged to
transfer): two independent fresh-context Opus judges, dispatched with
the `detector-judge.md` prompt verbatim (from platform-skills' `paper`
skill) plus the 10 concatenated section files (`sections/00_abstract.tex`
through `sections/09_appendix.tex`), told to ignore `% <!-- evidence:
Rx -->` build-metadata comments and raw LaTeX markup and to judge prose
only. Bounded to a maximum of 2 rounds; discharge requires both judges
at $\geq$90%-human AND zero cited mechanical tells; iteration licensed
only on cited mechanical tells. Draft state: post-final-review, pre-change-1
prose is unaffected by change 1 (the §3 numeric fix touches only digits,
not sentence structure) — this round ran on the section tree as it stood
after change 1 landed and before any detector-driven edits.

## Verdicts

| Judge | Classification | Percentage-human read | Cited tells |
|---|---|---|---|
| A | human-written | 99% human | none counted (six-item abstract enumeration considered and explicitly cleared as content-driven, not templated) |
| B | human-written | 90% human | 2 — (1) over-balanced sentences: the "X, not Y" / colon-verdict antithesis pattern recurring across cases, cited verbatim at 5 sites ("leaves it bit-unchanged at 0.9990: the probed layer is causally inert"; "undelivered by the instrument, not failed by the model"; "the dissociation, not the zero, is the finding"; "Fixing the instrument changed what the verdict means, not the verdict"; "the full rescue is centering plus a full-intertwiner degauge, not centering alone"), with the fix suggestion "let two or three cases state the finding plainly without the antithetical snap"; (2) featureless rhythm (inverted — airless uniform density, no low-information breathing sentences), with the fix suggestion "add an occasional plain orienting sentence at a case opening" |

Judge A's read cleared cleanly (99%, no tell counted against the verdict).
Judge B's read cleared the $\geq$90% floor exactly but cited two concrete,
actionable tells with quoted sites and named fixes — not "considered and
cleared" residue, since both were given as the explicit reason the score
sat at 90 rather than 100.

## Disposition

**Round result: NOT DISCHARGED.** Judge B's citations are genuine
mechanical tells under the protocol (concrete quotes, named category,
actionable fix), not structural-symmetry residue waved off by the judge
— so "zero mechanical tells" is not met even though both judges clear
$\geq$90%-human. Per the pre-declared rule ("iterate only on mechanical
tells"), a second round is licensed, scoped strictly to the cited sites.

**Fixes applied (round 1 → round 2), all prose-only, no evidence/numeric
content touched:**
1. `sections/04_case_wronglayer.tex`: "leaves it bit-unchanged at
   $0.9990$: the probed layer is causally inert." → split into two
   sentences, colon-verdict removed: "...leaves it bit-unchanged at
   $0.9990$. That layer plays no causal role in the recall."
2. `sections/09_appendix.tex`: "...the full rescue is centering plus a
   full-intertwiner degauge, not centering alone." → recast without the
   "X, not Y" template: "...the full rescue needed a second step on top
   of it: a full-intertwiner degauge."
3. `sections/05_case_gauge.tex`: "Fixing the instrument changed what the
   verdict means, not the verdict." → recast: "The verdict itself never
   moved; fixing the instrument changed how to read it."
4. `sections/03_case_tolerance.tex`: opening run-on (one sentence, three
   semicolon clauses) split into three plain sentences to add rhythm
   variation ("An eval-time admission gate decides which trained cells
   enter a capacity fit. One leg of that gate checks that...").

Deliberately left unchanged (load-bearing, not a clean tell to fix): the
§4 punchline "the dissociation, not the zero, is the finding (Rule 5)"
— the final review (`../gauntlet/round-1/07_final_review.md` §1)
explicitly names this sentence as the paper's stated arc; and
`sections/09_appendix.tex`'s "undelivered by the instrument, not failed
by the model," which recurs in the Table 1 catalogue schema and was left
as the compact table's fixed column language.

Rebuilt `main.pdf` + `bundle/` after the fixes; page count held at 8
(body p1-4, refs p5, appendix p6-8, no reflow); anonymization grep and
bundle/main.pdf byte-identity re-verified clean.

**Terminal state: round 1 of MAX_ROUNDS=2, not discharged, round 2
dispatched with a fresh pair of judges per protocol.**
