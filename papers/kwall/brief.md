# Paper brief -- kwall ("the K-axis closes at 32 -- and why")

Stage 0/1 of the `paper` skill (repo mode), revised 2026-07-16 to reflect
the diagnosis wave that landed after this brief's first draft (2026-07-12/
13, preserved in git history). The earlier draft's headline was the
"four levers" attack on the K-wall (trainability fix, budget, anneal
shape, state-dimension convention, leakage mechanism). That material is
now background (Section 4 of the draft, "The K=32 Trainability Wall") for
a sharper headline: the NCR K=8 capability separation, the K=32 wall as a
closed, replicated fact, a MEASURED mechanism (spectrum-blind write
objective -> non-normal operators -> far-depth annihilation), and a
pre-registered, in-flight fix reported as pending. This revision follows
the write-up task's explicit five-item content spec, not the earlier
brief's own four-lever framing.

**Subject.** Native Composition Reads (NCR) writes relation operators
in-context into a $d\times d$ fast-weight state and reads
`o = read(Z^h, q)` via exact $O(\log h)$ repeated squaring. At $K{=}8$
this wins a pre-registered capability separation against $O(h)$
baselines. Scaling the identical mechanism to $K{=}32$ closes under a
pre-registered stopping rule (24 independently re-verified cells, zero
far-depth recovery in every one). A follow-on diagnosis
(`matrix-thinking/NCR_ORTHO_WRITE.md`) finds the wall's proximate cause --
a spectrum-blind shallow-depth cosine loss -- and shows, via a no-retraining
counterfactual, that the wall is synthetic (an objective property), not a
capacity ceiling. A live GPU wave training the fix directly is in flight
as of this draft and is reported pending, not read.

## Venue

Carried forward from the 2026-07-12/13 live verification (not re-fetched
this session): **MOSS @ COLM 2026, Small-Scale Frontier Track** --
4pp main-content cap, non-archival, double-blind, dual-submission-safe,
COLM 2026 official template (`colm2026_conference.sty`, tag `2026`,
identical vendored kit to `papers/reasoning-null-moss/` and
`papers/measurement-ws/`). Every model in this program is under 200K
parameters, squarely inside MOSS's own scope statement. **Re-verify the
CFP/late-window status before submission** -- three days have passed
since the last live check and this revision did not re-fetch it.
Backup: 2nd Workshop on Efficient Reasoning @ COLM 2026 (deadline was
Jul 19, 2026 AoE as of the last check).

## Thesis (one falsifiable sentence)

> Native Composition Reads holds exact, in-context-written relational
> composition to a pre-registered separation depth at $K{=}8$
> (median recovery 1.0 vs. the best $O(h)$ baseline's 0.158 at
> $h^\ast{=}61$) via an $O(\log h)$ query-time read, but the identical
> training recipe fails to converge at $K{=}32$ under every dimension
> convention and training budget tested (24 cells, far-depth recovery
> exactly 0.0000 in all); a measured mechanism (a shallow-depth cosine
> loss exerts no gradient pressure on write-operator normality, so the
> trained operator becomes ill-conditioned -- condition number
> 320--2952 vs. ~1 healthy -- and a far-depth matrix-power read
> annihilates its weakest eigenmode) shows this wall is a property of the
> objective, not the architecture, via a no-retraining polar-projection
> counterfactual that extends surviving depth from h~6 to h~27-51 on the
> same trained weights.

**Falsifier (would void the mechanism half):** the pending fix
(Section 6 of the draft) lands NULL with a mechanistic signature already
near its healthy target (departure-from-normality and condition number
already close to the pinned WIN bar) -- that combination would mean the
diagnosis is incomplete or wrong, not merely that the fix underperformed.
Not yet observed; the run has not been read.

## Content map (five items, per the write-up task; each cites its source)

1. **NCR K=8 win.** `matrix-thinking/NOVEL_ARCH_WATERFALL.md` S7e (Axis A
   WIN, median rec@0.9=1.0 at h*=61 vs. fwm 0.158; Axis B WIN, 20.9x;
   Axis C TIE) and S3.2/S3.2a (the capability-regime re-scope, the
   mod-K collapse trap F2, the WIN/TIE/LOSE band definitions). Draft
   section: `sections/03_win.tex`.
2. **The K=32 wall.** `NOVEL_ARCH_WATERFALL.md` S11 (Gate 1/Gate 2,
   pre-registration), S9.10 (origin cliff K14 vs K15/16), S11.2
   (earlyln K-ladder, K15 SCALES / K16,K24 TRAINABILITY-STILL-LIMITED),
   S11.4 (d=K+1 convention CONFIRM at K16/K24) + S11.4a (leakage
   mechanism, the CORRECTED scope -- K<=24 only), S11.5 (K=32 full d(K)
   grid CLOSES, WAVE-1b BLOCKED -- the CORRECTED "bounded to K<=24"
   finding), S11.6 (budget-rescue dissociation probe, ANOMALY fires,
   still CLOSES). Draft section: `sections/04_wall.tex`.
3. **The diagnosis.** `matrix-thinking/NCR_ORTHO_WRITE.md` S1 (hypothesis
   + mechanism), S2 (the fix, exact parametrization), S5 (the in-silico
   polar-projection counterfactual table, reproduced verbatim). Diagnosis
   scripts cited by path:
   `experiment-runs/2026-07-16_ncr_ortho_write/diagnosis/{k32_spectral_diag.py,k32_counterfactual.py,k32_nonnormal_frontier.py}`.
   Draft section: `sections/05_diagnosis.tex`.
4. **Pending verdict.** `NCR_ORTHO_WRITE.md` S3 (re-registered h*=40),
   S4 (WIN/PARTIAL/NULL/FAIL bands, Part A + Part B), S7 (pre-registered
   odds). Draft section: `sections/06_pending.tex`. **This paper does not
   read `experiment-runs/2026-07-16_ncr_ortho_write/results_ortho_write*`
   -- the run is live and blind as of this draft.**
5. **Related work.** `research/ncr_separation_grounding.md` and
   `research/ortho_write_grounding.md` (both dated 2026-07-16, live
   web-search verified). Draft section: `sections/07_related.tex`,
   citations in `refs.bib`.

## What changed from the 2026-07-12/13 draft of this brief

- Headline moved from "four levers, mechanism, bound" to "win, wall,
  diagnosis, pending fix" -- the ortho-write diagnosis (opened
  2026-07-16, after this brief's first draft) supersedes leakage/D_share
  as the paper's mechanism headline. The leakage mechanism (S11.4a) is
  retained as background/context in Section 4 (it explains relative
  d-ratio differences at K<=24; the ortho-write diagnosis explains the
  K=32 failure itself and is scoped differently -- the draft states this
  relationship explicitly, does not conflate the two).
- The "T-BOUND" title candidate from the earlier draft is now the actual
  title (adapted): *"The K-Axis Closes at 32 -- And Why: Diagnosing a
  Spectrum-Blind Write Objective in a Fast-Weight Composition Reader."*
- Related work is no longer TO-VERIFY placeholders: both grounding memos
  landed during this session (staged in git,
  `research/ncr_separation_grounding.md` +
  `research/ortho_write_grounding.md`, 2026-07-16) and were read directly
  by this write-up (not taken on any intermediary's word) before citing.
  One entry (Nichani, Lee & Bietti's rank-1/argmax remark) carries the
  grounding memo's own NEEDS-FINAL-SPOT-CHECK flag through into the draft
  text and `refs.bib`, unresolved. Several bibliography entries are
  SURNAME-ONLY because the grounding memos name authors by surname alone
  in places -- given/first names are never invented; two entries
  (Log-Linear Attention, HOLA) have no bibtex entry at all and are cited
  via inline arXiv URL for the same reason.
- Novelty framing corrected: the draft does NOT lead with "matrix state
  can state-track" (already published, Grazzi et al. negative
  eigenvalues + RWKV-7 both cited) -- it leads with the O(log h)
  query-time depth-access complexity claim against every published
  alternative's O(h) sequential rollout, per
  `research/ncr_separation_grounding.md` Part 3's own adversarial
  novelty-boundary analysis.
- The orthogonal-write fix is explicitly NOT presented as unprecedented:
  MuonSSM (arXiv:2606.30461, ICML 2026 Oral) is cited as the closest
  prior art, with the three-axis differentiation (full-rank vs. rank-1,
  40-iteration cubic vs. single-iteration quintic, compositional-depth
  motivation vs. general SSM stability) stated in `sections/05_diagnosis.tex`
  and `sections/07_related.tex`.

## Build status

`main.tex` + all 9 section files + `refs.bib` compile cleanly with
`tectonic` (three-pass, house convention) to a 10-page PDF (main text +
two appendices + references) -- zero errors, only benign font-substitution
and underfull/overfull-box warnings, zero undefined citations. **Not yet
page-fit to MOSS's 4pp main-content cap** -- this draft was not trimmed
for length; a page-fit pass (the same kind of structural-compression pass
`papers/rank-recruitment-ws/outline.md` records, e.g. folding tables into
prose, moving detail to the appendix) is a follow-on step before
submission, not performed by this write-up.

## Dual output

- [ ] Venue submission (anonymized page-fit to MOSS's 4pp; the current
  draft is un-anonymized-by-default via `[submission]` mode but has not
  been grepped against the program's standing anonymization token list)
- [ ] Public write-up (pebbleml.com per this program's per-finding-
  publisher convention) -- **blocked on the pending ortho-write verdict**
  landing; the honest version of this paper cannot be finalized while
  Section 6 is a placeholder.

Neither has been produced. This is a draft, not a submission-ready
artifact.
