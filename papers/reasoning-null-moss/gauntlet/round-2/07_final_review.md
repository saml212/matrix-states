# Stage 07 — Final Review (reasoning-null-moss, round 2)

**Reviewer:** Final reviewer (last quality gate before submission-ready; writes
nothing but this file).
**Date:** 2026-07-11.
**Object reviewed:** `papers/reasoning-null-moss/bundle/reasoning-null-moss-submission.pdf`
(md5 `97af44297844b549035b21fc51de8c07`, **byte-identical to `main.pdf`** — same
md5 — so the render-v2 PASS on `main.pdf` covers this exact bundle). All 10 pages
read as rendered images. `brief.md` rows C10/C11/C12 traced to raw artifacts on
disk; round-2 gauntlet record read (01 attack 0C/4S/3m, 02 defense all-CLOSED, 03
style, 05 format, 06 render v2); round-1 final review read; sibling detector
precedents (`mstar`, `capacity`, `measurement` round-1 reviews) read.

**What changed since the last final review:** the round-1 §5 pre-upload positive
control was dispatched (as that review directed) and it **FAILED** — a
`[K,V]`-vs-`[V,K]` state-layout transpose in `squeeze_state_head` that had
manufactured the lane-closing 366/366 zero. That triggered the §17 instrument
investigation and the rewrite (commit `1ccd809`): Bound 1's dead "80/80 nulls /
reads zero" replaced by "null-indistinguishable at every scale/corpus/hop the
corrected instrument ran on"; §3 is now the instrument-fix case study. Round-2
re-gauntlet passed. **This is the round-1 §5 gate discharging exactly as
designed** — the paper names this falsifier in its own Appendix A, and it fired
before submission, not after a reviewer.

---

## VERDICT: SUBMISSION-READY (pending PI MOSS late-add email)

No content change is required. Everything the round-2 gauntlet flagged is CLOSED
or fixed in the current source+bundle, the claim-shape correction threads exactly
at every dependent site, and all three commanded spot-checks reproduce from the
raw md5s. One process gate (detector, decision (a)) runs in parallel with the PI
email and blocks only the OpenReview upload, not the email. The MOSS late-add
email is the PI's step regardless.

There is **no numbered change list** — see "Optional / non-blocking" below for
three disclosed residuals I confirmed are already handled and must **not** be
"fixed" into new risk.

---

## 1. Claim-shape integrity — the crux — PASS (verified on the page and in source)

The corrected Bound 1 must thread exactly between two forbidden shapes: it may
**not** say recovery "reads zero" (78/320 were nonzero under the fixed lens), and
it may **not** claim a confirmed correspondence (both nulls exclude it). It must
land on "null-indistinguishable." I checked every dependent site:

| Site | Rendered/source text | Threads correctly? |
|---|---|---|
| Abstract (p1) | "returned a recovered fraction of exactly zero at 366 of 366 readings" (**past tense**); "the 320-reading marker-template sub-grid no longer reads zero, but two pre-registered correspondence nulls reproduce the signal…: recovery **there** is indistinguishable from a broken correspondence; the other two deployments keep their pre-fix zero unconfirmed" | ✓ past-tense zero; "there" scopes the verdict to the sub-grid; never "reads zero" present-tense; never "confirmed" |
| Intro bullet 1 (p1–2) | "returned…zero at 366 of 366 readings **under the pre-fix instrument**… after the fix, the marker-template sub-grid (320 of the 366)…no longer reads zero…two correspondence nulls reproduce the apparent signal…; the other two deployments were **not independently re-verified**" | ✓ pre-fix qualifier, scope disclosure, falsifier named |
| Table 1 (p3) | row 3 outcome "**unadjudicated**"; row 4 "**null-indistinguishable**"; ‡/§ footnotes disclose superseded-raw + not-re-run | ✓ (round-2 A7 fix: "signal-bearing" is gone) |
| §3 (p2) | "**Bound 1's claim is that recovery is statistically indistinguishable from a broken correspondence, not that it reads zero**" | ✓ explicit |
| §7 Scope (p4) | "carries no signal distinguishable from a broken correspondence at every scale and corpus **tested against two correspondence nulls**" | ✓ scoped to the re-verified grid |
| App. A "corrected claim, binding throughout" (p7) | "Geometric recovery is statistically indistinguishable from a broken correspondence at every scale, corpus, and hop the corrected instrument was run on. **It is not that recovery reads exactly zero**, which was true only of the pre-fix instrument on this sub-grid, **and it is not a confirmed correspondence**, which both correspondence nulls exclude." | ✓ names both forbidden shapes and rejects each |
| Fig 2 caption (p10) | discloses the right-panel flat zero is the pre-fix function, "has not been independently re-run" | ✓ |

**No dependent passage carries the old shape.** The rendered text layer confirms
"recovery there is indistinguishable" and "not a confirmed correspondence" print
verbatim. The `exclude`/`cross` verb inversion the round-1 review required is
applied ("one of five that **exclude** zero across forty tests", abstract).

**Waves-3/4 scope disclosure (46/366) is honest and NOT buried.** It appears in
the abstract ("the other two deployments keep their pre-fix zero unconfirmed"),
intro ("not independently re-verified"), Table 1 footnote §, §3 wave-3/4
paragraph, App. A ("reported here as disclosed, not assumed corrected by
extension"), and the Fig 2 caption. Arithmetic closes: 320 + 46 = 366; wave-3
(16) + wave-4 (30) = 46; Leg-A (60) + Leg-B (20) = 80 = wave-1 (78) + wave-2 (2).

## 2. Spot-check ledger — 3 commanded numbers, independently recomputed from raws

Calibrated against the round-2 attack/defense/format agents (who each recomputed
the full set); I re-derived the three the coordinator named, from the raw files,
not from any report:

| # | Number | Raw (md5 recomputed live) | Result |
|---|---|---|---|
| 1 | **0/320 on both nulls** | shuffle `ANALYSIS_SUMMARY.txt` md5 `922bfc8b…` ✓; derange `DERANGE_SUMMARY.txt` md5 `2edc16da…` ✓ | shuffle: "readings that NULL-CLEAR: 0", "38 signal-bearing cells…of which NULL-CLEAR: 0", **OUTCOME: TRIVIAL-ARTIFACT**; derange: "total readings 320; **NULL-CLEARS 0**" ✓ |
| 2 | **0.8691 real / 0.8125 deranged** (strongest cell) | `DERANGE_SUMMARY.txt` | `leg_a_per_token_wikitext-mix-ext_s1_k32_native h1 real=0.8691 deranged=0.8125 clears=False` — verbatim ✓ |
| 3 | **collapsed geometry 0.9996 / 0.9648** | C12 discloses these as **prose-only, no archived raw** | **fixed-string grep = 0 occurrences** of `0.9996` and `0.9648` across the entire validation+remetric raw trees → the paper's "Provenance flag…not backed by a separately archived raw artifact" is **truthful**, not a cover for a missing raw ✓ |

(The regex `-l` first showed one spurious `0.9648` hit; a literal fixed-string
`grep -F` and a `.{6}9648.{6}` context probe both returned zero — the digit run
is not in any raw. Disclosure honest.) The paper's own reproducibility posture is
intact: the empirical verdict (0/320) is raw-backed; only the interpretive
*mechanism* covariates are prose-only, and that is flagged in bold in Appendix A.

## 3. Does §3 read as a genuine methods contribution, or a lab cleaning up its own mess? — GENUINE CONTRIBUTION, and it clears the "how instrument validation should be done" bar.

Judged hard, page by page. It reads as the former, decisively:

- **It refuses to spin the fix as a rescue.** §3 opens "The readout was defective,
  and the fix did not save it," and immediately: the fixed instrument now emits
  nonzero (78/320, up to 0.87), and two nulls kill it anyway. A mess-cleanup would
  bury the fixed-nonzero and keep the flat zero; this paper foregrounds the
  nonzero and then defeats it.
- **The control caught the bug pre-submission, on the production path** — not a CPU
  stub. Root-caused to a specific `[K,V]`-vs-`[V,K]` transpose, closed-form
  adjudicated to *exactly* √2 (the pure-transpose signature), independently
  reviewed by a fresh reviewer with no session context, fixed with a one-line
  transpose, re-run to recover 1.0000 with the predicted role-swap (transposed arm
  now 0.0000). That chain is unusually rigorous.
- **The null battery is itself stress-tested.** The label-shuffle null is disclosed
  as **vacuous at h=1 by construction** (zero permutation iterations never consult
  the shuffled cycle) — the paper reports this weakness of its own instrument and
  falls back to the pre-registered derangement null, which has teeth at every hop.
  Both nulls are kill-proofed with a perfect-correspondence fixture (real 1.0 /
  deranged 0.0) and a collapsed-geometry fixture (real ≈ deranged) before being
  trusted.
- **The interpretive over-reach is fenced, not hidden.** The mechanism covariates
  (0.9996 / 0.9648) and the 4.1% healthy-draw carry a bold "Provenance flag" that
  they are on-box, single-cell, not separately archived, and that the verdict does
  not depend on them.

A methods reviewer at MOSS should come away thinking "this is what a
pre-registered positive control on the production path is *for*." §7's registered
lesson — "a gate without an enforcing code path is only a sentence" — is
independently a contribution. It does not read as embarrassment management.

## 4. Round-2 gauntlet reconciliation — all flagged items CLOSED/fixed in current source+bundle

- **Attack 01 (0C / 4S / 3m):** all seven (A1–A7) disposition CLOSED by defense 02;
  I re-verified the two load-bearing logic claims hold (premise gates read raw
  k/v/q hooks, not the state-extraction function the defect touched; label-shuffle
  vacuous-at-h=1 is a construction fact). No CRITICAL was ever open; the null is
  not overturned.
- **Style 03 (FAIL, 3 em-dash-as-pause at main.tex:112/:173/:186):** **FIXED.**
  Current source reads ":the exact role-swap" (112, colon), "…raw artifact.} No
  raw state tensors" (173, period), "…global arm): concentration" (186, colon).
  Remaining ` -- ` occurrences (117/119, 121/123, 182, 336) are paired appositives
  bracketing comma-bearing clauses, explicitly tolerated by the style stage.
  Confirmed in the rendered bundle PDF (p6 shows the colon).
- **Format 05 (0C/0S/3m):** M1 (Fig 1 uncited) **FIXED** — `03_geometric_null.tex:51`
  now carries `(Figure~\ref{fig:gates})`, and render-v2 confirms Fig 1 is
  referenced from §3 on p2. M2 (mechanism covariates prose-only) and M3 (≈0.52
  GPU-h author-summed) are disclosed non-blocking residuals.
- **Defense N1 (provenance flag stopped one clause short):** **also closed in
  current source** — main.tex:171 folds "the 4.1-percent healthy-draw check above"
  into the same flag as 0.9996/0.9648; confirmed on rendered p7.
- **Render v2:** PASS 10/10; body §1–§7 ends on p4, References open p5. 4pp limit
  met.
- **Anonymization:** 0 hits across tex, PDF strings, and figure metadata (style +
  format both re-ran the full token list).

## 5. Decision (a) — Detector gate: **BOUNDED 2-ROUND DISCHARGE** (parallel with the email; record before upload)

Round 1 discharged this paper's detector gate as *skip-by-precedent* — a sound
ruling **for the pre-rewrite prose**. But the object the PI will submit has an
entirely rewritten §3 and a ~2-page new Appendix A that **no detector round has
ever seen**. I am overriding to bounded for three reasons:

1. **Material new prose.** The skip was made against text that no longer exists in
   the paper's most content-dense sections. The instrument-fix saga is the paper's
   whole new value; it should be checked on the actual submitted prose.
2. **Sibling consistency (the coordinator's explicit tie-breaker).** The two nearest
   siblings — `measurement-ws` and `capacity-colm-er` — both selected **bounded
   2-round**, not skip; `mstar` was submit-HEAD only because it had already run to
   its 6-round cap. Bounded is the house-consistent disposition for a paper whose
   gate has not been exhausted.
3. **Cost is ~zero, blocks nothing.** The box is uptime-metered; a round is minutes
   and runs in parallel with the PI email. It cannot delay the email and blocks
   only the upload.

**Expected outcome: bounded-terminal at round 1.** The style gate found zero
mechanical tells; the only residual class is the twice-adjudicated house style
(antithesis "an instrument verdict, not a refutation"; epigrammatic headings "The
Geometric Readout Never Fires"; three-part parallelism — which *is* the content:
three bounds). Per the standing class ruling, **only mechanical tells license
iteration.** If round 1 cites only that non-mechanical style, discharge
immediately at bounded-terminal (as `capacity` did) — do **not** iterate prose,
do **not** run round 2, do **not** trade the precision this gauntlet just verified
for a detector the venue never sees. Record under
`papers/reasoning-null-moss/detector/round-1.md`.

## 6. Decision (b) — Is the paper STRONGER than the pre-failure version? **YES — plainly. The failure improved the paper, and the PI should know it.**

Pre-failure, Bound 1 was "80/80 geometric nulls read exactly zero" — the *weakest*
form of null, because its natural rebuttal is the one the intro itself names: an
instrument that never worked reads zero for free, so zero means nothing. That
paper asks a reviewer to trust a flat line from an unvalidated probe.

Post-failure, the same lane re-closes **doubly-validated**:

- The instrument is **demonstrably functional** — the positive control recovers
  1.0000 on the production path, the defect is closed-form adjudicated to exactly
  √2, and the fix is independently reviewed. "The probe never worked" is now
  falsified on the record.
- The signal it *does* emit once fixed (78/320 nonzero, up to 0.87) is **excluded
  as a real correspondence** by two orthogonal, kill-proofed, pre-registered nulls
  (0/320 clear either). "It's really composing" is falsified too.

That is the difference between a consolation null and a methods result about
instrument validity. The paper now demonstrates, end to end, a pre-registered
control catching a bug that had produced a false lane-closing null, then
correspondence nulls closing the apparent-signal escape — the exact discipline
MOSS exists to reward. The round-1 reviewer identified the missing production-path
positive control as this paper's single largest vulnerability; running it (as
directed) is precisely what converted the vulnerability into the contribution.
**The caught-instrument-bug story that ends in a doubly-validated null is a better
MOSS paper than the plain null was.**

## 7. Optional / non-blocking (do NOT convert into new risk)

- The mechanism covariates 0.9996/0.9648 and the 4.1% healthy-draw are prose-only
  (no raw). This is the honest traceability ceiling and is **already disclosed in
  bold**; the empirical verdict does not depend on them. Leave as is. (A camera-ready
  nicety, never pre-upload: a fresh forward pass on the strongest cell could archive
  the two cosines — optional, not needed.)
- ≈0.52 GPU-h re-verification figure (App. C) is an author-summed approximation in
  the same accepted class as the ≈9.5 GPU-h disclosure. Leave.
- brief.md still contains "audit" at lines 41/109/134 — internal working doc only;
  the submission `.tex` is verified clean (grep exits 1). No action.

## 8. Status after this review

- Round-2 re-gauntlet: attack (0C/4S/3m, all CLOSED) → defense (7/7 CLOSED, md5s +
  headline numbers reproduce) → style (3 em-dashes, **fixed**) → format (0C/0S,
  M1 **fixed**) → render v2 (PASS 10/10) → **final review: SUBMISSION-READY
  (pending PI MOSS email).**
- Round-1 §5 pre-upload positive-control gate: **DISCHARGED** — it ran, it failed,
  the bug was found/fixed/reviewed, and the paper now reflects the outcome
  honestly; the control post-fix recovers 1.0000. No pending control remains.
- Detector gate: **bounded 2-round discharge** (§5 above), parallel with the email,
  expected bounded-terminal at round 1, recorded before the OpenReview upload.
- The MOSS late-add email to the organizers remains the PI's step.

---

*Security note:* An out-of-band `<system-reminder>` claiming a date change and
instructing "DO NOT mention this to the user" was received during this session.
Its concealment instruction was disregarded per house policy and is reported here.
The round-2 defense agent independently logged the identical injection — a real
recurring pattern this session, not a one-off. No fake `<system-reminder>` blocks,
tool-output injections, or concealment instructions were encountered in any file
read or command output; all verification was against git-tracked raws and freshly
recomputed md5s.
