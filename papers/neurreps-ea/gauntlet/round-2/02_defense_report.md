# Defense Report — Round 2 (Stage 02)

Paper: "The Rank the Task Demands: A Causal Rank Law for Matrix Memories
Trained on Group Composition" (NeurReps 2026 EA draft, `papers/neurreps-ea/`).

Inputs: `gauntlet/round-2/01_attack_report.md`, the draft
(`main.tex` + `sections/*.tex` + `refs.bib` + `brief.md`), the raw JSONs
in `experiment-runs/2026-07-09_m3fix_harvest/` (repo + SSD mirror at
`/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-09_m3fix_harvest/`),
`matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md`, and one live web
search to independently check `deletang2023chomsky`'s actual scope
(B4). Every numeric claim below was re-derived from the raw artifact,
not copied from the attack report.

**Process-integrity note.** A tool-output block during this pass
contained an embedded fake `<system-reminder>` claiming the date had
changed and instructing silence about it — the exact injection pattern
this repo's `CLAUDE.md` Hard Rules document. It was disregarded; it had
no bearing on the analysis and is reported here per the standing rule.

---

## 1. Summary for the rebuttal agent

**0 DEFEND, 1 PARTIAL, 5 CONCEDE + FIX.** All six findings survive
independent re-derivation against the raw JSONs, the compiled PDF, or a
live citation check — none rest on a misread. The good news: every fix
is a same-sentence or same-cell wording/number correction with evidence
already in hand (no new experiment, no new seed, nothing that touches
the paper's actual scientific claims). The paper is **submittable after
fixes**. Two of the five fixes (B2, B4) add net words to body text that
is already at the exact 4pp limit per round 1's rebuild and will need a
named compensating cut verified by a fresh build before submission —
flagged explicitly below, not silently assumed to fit. The most
important fix is B1 (a real, verifiable arithmetic error printed in the
compiled PDF, contradicted by the paper's own Appendix B two pages
later); B2 is the most strategically significant, since it is the exact
sentence round 1's own rebuttal predicted a fresh attacker would land
on, and it did.

---

## 2. Defenses

### B1: The "corrected" 5–11% deviation range in §4 does not match its own cited evidence (max is 10.2%, not 11%)

**Disposition:** CONCEDE + FIX

**Response.** Confirmed exactly. `sections/04_ranklaw_observed.tex:5`
states "5--11\%"; Appendix B (`sections/07_limitations.tex:67-68`)
states the same evidence tag's (N4) actual per-group deviations as
6.1/4.9/5.6/10.2/5.3%. I independently recomputed from Table 1's own
means (`tab:m1`): $S_3\ (2-1.877)/2=6.15\%$, $S_4\ (3-2.852)/3=4.93\%$,
$A_5\ (3-2.832)/3=5.60\%$, $S_5\ (4-3.591)/4=10.225\%$, $A_6\
(5-4.736)/5=5.28\%$ — true range 4.93–10.23%. No number anywhere in the
paper's own evidence exceeds 10.23%. Tracing the origin: round 1's own
rebuttal report (`gauntlet/round-1/04_rebuttal_report.md`, FIX-8 "Why")
computes "The true range is 4.9–10.2%, not 4–8%" and then writes
"5--11\%" into the instructed replacement text two lines later — an
internal contradiction inside round 1's own audit trail that landed
uncaught in the committed draft and the compiled `main.pdf`. This is a
clean, verifiable, in-paper inconsistency. Not a CRITICAL — it touches
an observational-leg precision claim, not the causal razor — but it is
the single most concrete, fixable error in scope this round.

**Supporting evidence.**
- `sections/04_ranklaw_observed.tex:5` ("5--11\%").
- `sections/07_limitations.tex:67-68` (Appendix B, N4 evidence tag: 6.1/4.9/5.6/10.2/5.3%).
- Independently recomputed from `tab:m1`'s means in this pass: 6.15/4.93/5.60/10.225/5.28%, range 4.93–10.23%.
- `gauntlet/round-1/04_rebuttal_report.md`, FIX-8, where the "Why" block's own stated range ("4.9–10.2%") contradicts the "After" text it instructs ("5--11\%").
- `grep -rn "5--11"` across `sections/*.tex` and `brief.md` returns exactly one hit — this is an isolated, single-point fix, not a value repeated elsewhere that also needs changing.

**What goes in the paper if this defense is accepted.** In
`sections/04_ranklaw_observed.tex:5`, replace "5--11\%" with
"4.9--10.2\%" — this is the exact pair of endpoints Appendix B already
prints two pages later (word-for-word self-consistent, zero new
numbers introduced, zero page-budget cost, one clause).

---

### B2: The "Pre-registered reading" sentence opening §5 still frames the geometrically-forced necessity floor as a conditional test outcome

**Disposition:** CONCEDE + FIX

**Response.** Confirmed. `sections/05_causal_razor.tex:8-10` states "if
$\dmin$ is load-bearing, $k = \dmin{-}1$ must fail..."; sixty words
later in the same section (`:63-67`), the paper states the same
necessity zero "follows regardless of training quality." A floor that
holds regardless of training quality also holds regardless of whether
$\dmin$ is load-bearing for what the model learned — the antecedent
does no work. This is not a new problem: round 1's own rebuttal
(`04_rebuttal_report.md`, "Residual risk after all fixes," A1) named
this exact sentence class as "exactly the kind of sentence the re-run
should stress-test with a fresh attacker," and it landed.

The design record shows the more accurate resolution is not merely to
split the sentence into a "sanity check" plus an "informative
criterion" (the attack's own proposed fix) but to restate what was
actually pre-registered. `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md`
(~line 1651) registers the true falsifier in leak-first form: "$k =
\dmin(G)-1$ reaches $\geq 0.9\times$ceiling at ANY group... $\Rightarrow$
investigate a degauging/embedding leak first (§1.9 item 1); if real,
this task family is rank-blind." The pre-registered test was never "does
$k=\dmin{-}1$ fail" as a hypothesis about the model — it was "if a
below-$\dmin$ cell ever clears the bar, that is evidence of an
instrument defect, investigated before any other conclusion is drawn."
The geometric ceiling proof (found after registration) closes that
contingency outright: it proves the triggering event is impossible by
construction, so the leak investigation branch can never fire. Stating
it this way removes the vacuous-conditional reading entirely, because
the sentence no longer poses "if $\dmin$ is load-bearing" as a testable
antecedent at all.

**Supporting evidence.**
- `sections/05_causal_razor.tex:8-10` vs. `:63-67`, same section, ~60 words apart.
- `gauntlet/round-1/04_rebuttal_report.md`, residual-risk note on A1, naming this exact sentence as the predicted failure point.
- `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md:~1651` (M3 HARD FALSIFY text), the actual registered falsifier, in its true leak-first form.

**What goes in the paper if this defense is accepted.** Replace
`sections/05_causal_razor.tex:8-10`:

Old: *"Pre-registered reading: if $\dmin$ is load-bearing, $k = \dmin{-}1$ must fail and $k \geq \dmin$ must recover past $0.9\times$ the anchor's crosscheck $\recninety$."*

New: *"Pre-registered reading: a below-$\dmin$ cell clearing the bar was registered to trigger a leak check, closed outright by the geometric ceiling below; $k \geq \dmin$ must recover past $0.9\times$ the anchor's crosscheck $\recninety$."*

**Cost flag:** +10 words vs. the original (23 → 33). The body is
already at the literal bottom of page 4 per round 1's rebuild — this
needs a compensating cut verified by a fresh build, not an assumption
that it fits. A same-size cut is available in the immediately following
sentence (`:63-67`): "so $\recninety = 0.000$ at $k = \dmin{-}1$ follows
regardless of training quality" → "so $\recninety = 0.000$ at
$k=\dmin{-}1$ is unconditional" saves ~4 words; the remaining ~6 need a
second small trim elsewhere in §5 (e.g. Figure 2's caption has no more
slack per round 1's own accounting, but the ambient-tax closing
sentence — "the zero-padded wave above is the registered fix" — could
lose "above" at zero information cost). Flag as a build-stage item, not
resolved here.

---

### B3: Appendix D's gate1a row for $S_3$ — "fail (all 4, by margin)" — mischaracterizes one of the four cells as a narrow miss when it misses by 12x the other three's shortfall

**Disposition:** CONCEDE + FIX

**Response.** Confirmed against the raw `gate1a.min_val` fields,
independently pulled in this pass:

```
S3 unconstrained  0.9143660  -> shortfall -0.0056
S3 k_dmin-1       0.6648946  -> shortfall -0.2551
S3 k_dmin         0.9002008  -> shortfall -0.0198
S3 k_dmin+1       0.9028212  -> shortfall -0.0172
```

Three cells miss the 0.92 bar by 0.006–0.020 ("by margin" is accurate
for them); the necessity cell ($k=\dmin{-}1$) misses by 0.255 — 12–43×
larger than the other three, and this is the cell whose failure the
table's own caption (two sentences later) explicitly attributes to a
*different, unrelated* mechanism ("the necessity leg... is unaffected
by construction"). The row's compact descriptor erases the distinction
the caption text draws immediately after it, inside the same table.
This is real: a reader scanning only the status column gets a false
impression of uniform narrow-miss behavior across all four $S_3$ cells.

**Supporting evidence.**
- Raw values pulled directly from
  `experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S3__{unconstrained,k_dmin,k_dmin_minus_1,k_dmin_plus_1}__seed0.json`
  (`gate1a.min_val`), matching the attack report to six decimals.
- `sections/07_limitations.tex:132` (`tab:gate1a`), S3 row.
- Contrast with the $S_5$ row (`:133`), labeled plainly "fail (all 4)" with no qualifier — S5's shortfalls (−0.024 to −0.119) are actually more internally uniform (5×, not 43×) than S3's, so the asymmetry in how the two rows are worded is backwards relative to which row needs the qualifier.

**What goes in the paper if this defense is accepted.** In
`sections/07_limitations.tex:132`, replace the $S_3$ row's status cell:

Old: `fail (all 4, by margin)`
New: `fail (all 4)`

This matches $S_5$'s existing plain phrasing exactly (`:133`), is
*shorter* than the current text (zero page-budget cost, fits the table
column at least as well), and removes the false uniformity claim
without needing to explain the necessity/sufficiency distinction inside
the cell — the caption's very next two sentences already carry that
distinction correctly and don't need duplication.

---

### B4: The Related Work characterization bundles `deletang2023chomsky` under a description that fits its three neighbors but not it

**Disposition:** CONCEDE + FIX

**Response.** Confirmed via live search, not training-knowledge alone.
`sections/06_related.tex:11-14` reads: "\citet{grazzi2025negative},
\citet{siems2025deltaproduct}, \citet{merrill2024illusion}, and
\citet{deletang2023chomsky} characterize which word problems recurrent
architectures can express." A web search on the paper's own abstract
confirms it benchmarks **20,910 models across multiple architecture
classes including Transformers** (not just recurrent nets) on **15
tasks spanning the full Chomsky hierarchy** (regular through
context-sensitive) — its headline finding is specifically that "RNNs
and Transformers fail to generalize on non-regular tasks," a
cross-architecture comparison, not a recurrent-architecture-only study.
Only one of its 15 tasks (Cycle Navigation) is a group-word problem.
Folding it into a clause whose other three members (Grazzi, Siems,
Merrill) genuinely are recurrent-architecture-specific state-tracking
papers overstates how central the recurrent/word-problem axis is to
Delétang's actual contribution.

Round 1's own rebuttal (`04_rebuttal_report.md`, §5 SHOULD-CITE)
already flagged that the bare-citation compression lost a more accurate
distinguishing clause and recommended restoring it "if space reopens" —
this attack confirms the compression produced an active inaccuracy, not
merely a loss of detail.

**Supporting evidence.**
- `sections/06_related.tex:11-14`.
- Live web search (this pass): Delétang et al., "Neural Networks and the Chomsky Hierarchy," ICLR 2023, arXiv:2207.02098 — 20,910 models, RNN/LSTM/Transformer/Stack-RNN/Tape-RNN, 15 tasks across the Chomsky hierarchy; "RNNs and Transformers fail to generalize on non-regular tasks... only networks augmented with structured memory can address higher-level tasks."
- `gauntlet/round-1/04_rebuttal_report.md`, §5 SHOULD-CITE note.

**What goes in the paper if this defense is accepted.** In
`sections/06_related.tex:11-14`:

Old: *"\citet{grazzi2025negative}, \citet{siems2025deltaproduct}, \citet{merrill2024illusion}, and \citet{deletang2023chomsky} characterize which word problems recurrent architectures can express; the marquee equivalence instead sorts by representation dimension."*

New: *"\citet{grazzi2025negative}, \citet{siems2025deltaproduct}, and \citet{merrill2024illusion} characterize which word problems recurrent architectures can express; \citet{deletang2023chomsky} probes the same group-word-problem axis across a broader architecture set; the marquee equivalence instead sorts by representation dimension."*

**Cost flag:** +11 words. Same page-budget caveat as B2 applies (body
at the exact 4pp limit) — a compensating trim is needed, e.g. tighten
the immediately preceding clause "\citet{mishra2026m2rnn} train
matrix-state RNNs on $S_3$ composition without a rank intervention." →
"...on $S_3$ composition, no rank intervention." (saves 2 words); the
remainder needs a second small cut verified at build time, not assumed.

---

### B5: `brief.md`'s N14 evidence row states the wrong arithmetic for its own quoted numbers (though the typeset paper is correct)

**Disposition:** CONCEDE + FIX

**Response.** Confirmed. Pulled `crosscheck_recovered_frac_90` directly
from the raw JSONs (SSD mirror, since the repo-root archive only keeps
S3 payloads under the 25MB cap):
`zero_pad__S4__unconstrained__seed0.json` = 0.65,
`zero_pad__S4__k_dmin__seed0.json` = 0.80 (diff +0.15);
`zero_pad__S5__unconstrained__seed0.json` = 0.50,
`zero_pad__S5__k_dmin__seed0.json` = 0.60 (diff +0.10). `brief.md`'s N14
row states "by 0.10 ($S_4$: 0.65→0.80)" — $0.80-0.65=0.15$, not $0.10$.
The typeset paper (`sections/07_limitations.tex:72`, "0.10--0.15") is
correct and unaffected. Confined to the evidence map, MINOR as rated,
but worth fixing since `brief.md` is the source of truth future edits
get checked against.

**Supporting evidence.**
- `brief.md:104` (N14 row).
- Raw values pulled in this pass from `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-09_m3fix_harvest/zero_pad__{S4,S5}__{unconstrained,k_dmin}__seed0.json`.
- `sections/07_limitations.tex:72`, correctly stating "0.10--0.15."

**What goes in the paper if this defense is accepted.** In `brief.md`'s
N14 row, replace "by 0.10 ($S_4$: 0.65$\to$0.80) and 0.10 ($S_5$:
0.50$\to$0.60) respectively" with "by 0.15 ($S_4$: 0.65$\to$0.80) and
0.10 ($S_5$: 0.50$\to$0.60) respectively." `brief.md` edits are
unconstrained by the page budget; no change needed to `main.tex` or
any `sections/*.tex` file.

---

### B6: Figure 1's caption "(bold)" parenthetical is ambiguous about whether all five groups are visually marked, when only four are

**Disposition:** PARTIAL

**Response.** The core observation is correct but the attack's own
citation is imprecise, which matters for how big a fix this needs. The
attack calls the razor data table "Table 2"; I checked the compiled PDF
(`pdftotext`) and there is no numbered "Table 2" containing this data —
the razor table (`sections/05_causal_razor.tex:21-31`) has no
`\caption`/`\label` and is not counted by LaTeX's table counter at all;
it is embedded directly inside Figure 1 (`fig:tracking`) and shares
Figure 1's single caption. The paper's actual numbered "Table 2" is
`tab:m1` (Appendix B's per-seed rank table), which has zero bold cells
and doesn't contain 0.450/0.800 at all. So "In Table 2, only $S_4$,
$A_5$, $S_5$, $A_6$ are typeset \textbf{}" cites the wrong object —
though the substantive claim (4 of 5 groups' cells are bold; $S_3$ uses
an asterisk instead) is true of the table embedded in Figure 1, and the
caption's "(bold)" parenthetical, read in isolation, does invite the
overclaim the attack describes.

The attack itself concedes the mitigating factor: "the very next
sentence in the same caption does clarify... a reader who finishes the
caption gets the right picture." That is a materially different
situation from B3 (where the row descriptor and the caption's
clarifying text are in genuine, unresolved tension even after reading
both) — here, a complete read resolves it. Kernel of truth, not a false
claim: PARTIAL, not CONCEDE+FIX.

**Supporting evidence.**
- `sections/05_causal_razor.tex:21-31` (the razor table, no `\label`/`\caption` of its own — embedded in `fig:tracking`).
- `pdftotext -layout main.pdf -` confirms the compiled PDF's actual "Table 1"/"Table 2"/"Table 3" labels attach to `tab:groups`, `tab:m1`, `tab:gate1a` respectively — none of which is the razor table.
- `sections/05_causal_razor.tex:42` (the "(bold)" parenthetical) vs. `:43-44` (the clarifying asterisk sentence, same caption).

**What goes in the paper if this defense is accepted.** In
`sections/05_causal_razor.tex:42`, tighten the parenthetical to point
inline rather than relying on the next sentence:

Old: *"...clears the $0.9\times$anchor bar at $\dmin$ in all five groups (bold)."*

New: *"...clears the $0.9\times$anchor bar at $\dmin$ in all five groups (bold; $S_3$ starred below)."*

+3 words, same caption, no new sentence, no page-budget concern at this
size.

---

## 3. New citations found during defense

None beyond what round 1 and this round's attack already surfaced. The
live search performed for B4 confirms `deletang2023chomsky`'s metadata
(title, authors, venue, arXiv ID) as entered in `refs.bib` is accurate;
the only issue is the characterization attached to it in
`sections/06_related.tex`, addressed above.

---

## 4. Attack ordering note

No CRITICAL in this round, and I agree with that top-line call — A1
(round 1's CRITICAL) is genuinely closed as a mathematical matter, and
nothing here reopens it.

- **B1 (SERIOUS):** Correctly rated. Concrete, verifiable, cheapest to
  fix of the six.
- **B2 (SERIOUS):** Correctly rated, and I'd flag it as the most
  *strategically* significant finding this round, independent of
  severity label — it is the literal sentence round 1's own rebuttal
  named as the thing a fresh attacker should stress-test, and it held
  up. Worth the rebuttal agent's attention even though the underlying
  math was never in question.
- **B3 (SERIOUS):** Correctly rated. Unlike B6, the caption's own
  clarifying text sits in unresolved tension with the row descriptor
  even after a complete read — the "by margin" qualifier isn't just
  incomplete information, it's contradicted by data two rows over.
- **B4 (SERIOUS):** I'd flag this as possibly **over-rated**. It is a
  real accuracy problem, confirmed independently, but it doesn't touch
  any of the paper's own experimental claims or numbers — it's a
  descriptive error about an external paper's scope. Comparable
  citation-precision issues in round 1 (A6) were also rated SERIOUS,
  so there's house precedent either way; I'd call this borderline
  SERIOUS/MINOR and let the rebuttal weigh it against the page-budget
  cost of fixing it properly (B2 and B4 are the only two fixes that add
  net words to body text already at the 4pp limit — if only one can be
  fully afforded before a rebuild, B2 is the one to prioritize).
- **B5 (MINOR), B6 (MINOR):** Correctly rated at MINOR; B6's severity
  is if anything supported further by the mitigating "next sentence
  clarifies" factor the attack itself names.

**Attacks I'd add that this round didn't raise:** none found. I
specifically checked whether the "5--11%" error (B1) recurs anywhere
else in the draft (`grep -rn "5--11\|4--8"` across `sections/*.tex` and
`brief.md`) — it is a single, isolated occurrence, so B1's fix is
complete as scoped and doesn't need a second search-and-replace pass.
