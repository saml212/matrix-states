# 08 — Companion decision record + change-list disposition

Authority: `07_final_review.md` (Fable, READY-AFTER-CHANGES). Coordinator
ratified the review's PRIMARY recommendation under the PI's impact-first
delegation: **both EAs stay at NeurReps**; `papers/neurreps-ea/` (the
restating sibling) receives the de-dup edits. Executed by a fresh
change-applier agent, 2026-07-11. This record discharges §4's closing
instruction ("record the decision in both trees' gauntlet logs before
either paper is submitted") for this tree; the mirror entry is
`papers/neurreps-ea/gauntlet/round-2/07_decision_record.md`.

## §6 numbered change list — disposition

1. **PI DECISION (companion resolution):** RATIFIED — primary path.
   Sibling edits S1-S3 applied to `papers/neurreps-ea/`
   (`sections/03_binding.tex`, `sections/07_limitations.tex`, and — as a
   consistency extension beyond the review's literal S1-S3 list, since
   the review's own §4(a) named the abstract as a third detectability
   signal — `main.tex`'s abstract). Recompiled (3x tectonic, 8pp,
   unchanged), render-inspected, bundle rebuilt, arXiv named-build kit
   rebuilt. No edit made to THIS tree's text (as the review specified —
   this tree is the primary carrier, "no de-dup edit available in this
   tree that removes overlap without deleting this paper's own
   contribution").
2. **PROCESS GATE (detector):** bounded 2-round discharge run this
   session; see `../detector/` for the round record and disposition.
3. **PROCESS GATE (CFP re-verify + anonymous.4open.science):** left
   open, flagged in `bundle/README.md`'s checklist — both wait on the
   2026-07-11 accepted-workshop-list drop, out of this session's scope
   per the review's own framing ("waits on the CFP drop").
4. **TEX/BUNDLE HYGIENE (orphan bib entries):** VERIFIED ALREADY DONE.
   Direct check of `refs.bib` and `bundle/refs.bib` (git-tracked,
   current working tree) found zero occurrences of `plate1995hrr` or
   `smolensky1990tensor` in either file (both 18 entries, byte-identical
   to each other). `git log`/`git show` traced this to commit `9b55740`
   ("gauntlet round-1 stage 5 — format audit 0/0/4, all four minors
   fixed", Jul 10 18:56:18) which deleted both entries as part of
   disposing format-audit M1 — this landed roughly 19 hours *before*
   `07_final_review.md` was written (Jul 11 13:32:51), so the review's
   claim that "format-audit M1 deferred this as a writer call" and that
   deletion was still outstanding is stale against the repo's actual
   history (the 05_format_audit.md document itself frames M1 as a
   writer call, but the writer made the call — delete — within the same
   commit that landed the audit). Per the CLAUDE.md tiebreak rule
   (contradictory claims about the same artifact get resolved by
   reading the raw artifact, not by trusting the more recent prose),
   this record settles it against the raw `refs.bib`/git history:
   **no action needed.** `make bundle` re-run as a no-op confirmation
   (zero files changed).
5. **OPTIONAL (camera-ready transparency, eigh-backward instability
   note):** left open — optional per the review, not exercised this
   session; no claim depends on it.
6. **BRIEF HYGIENE (false companion-disclosure claim):** FIXED. The
   "Companion papers and overlap management" section's
   `papers/neurreps-ea/` bullet asserted disclosure "in both briefs and
   in each paper's related-work/limitations pointer" while the
   sibling's paper text carried zero disclosure (verified false at the
   time the review wrote it). Now true after S3 landed in the sibling;
   brief.md's bullet rewritten to describe the S1-S3 edits and cite
   this record + the sibling's decision record as the discharge trail.

## Verification trail

- `git log --oneline -- papers/rank-recruitment-ws/refs.bib` and
  `git show 9b55740 --stat` — orphan-bib timeline.
- `diff refs.bib bundle/refs.bib` — zero diff (18/18 entries, no
  orphans in either).
- Sibling's four rendered pages (1, 2, 5, 7) inspected as PNGs — see
  the sibling's own decision record for the render-inspection log.
