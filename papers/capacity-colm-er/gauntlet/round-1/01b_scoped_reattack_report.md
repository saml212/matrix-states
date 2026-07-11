# Scoped re-attack report — round 1b (fresh-context, A1 + narrowing-call)

Role: `paper` skill gauntlet stage 01, scoped re-attack (`prompts/attack-agent.md`).
Draft state: commit 7272345 (HEAD), FIX-1..FIX-8 applied in 76e2ab3.
Method: independently re-derived every A1-relevant number from the raw cell
JSONs (md5-verified) and from the training/eval code, not from design-doc prose.

## One-line summary verdict

**A1 is CLOSED** (the CRITICAL is resolved — the FALSE "held-out entities"
assertion is gone from `02_testbed.tex`, and the FIX-1 numbers reproduce
exactly from the raws). **New findings: 0 CRITICAL, 1 SERIOUS, 1 MINOR.** The
narrowing call SURVIVES for `03_cliff.tex` but is BROKEN for `00_abstract.tex`
(and weakly for `01_introduction.tex`): the abstract's unqualified "held-out
recall frontier" is standalone-ambiguous on the exact axis A1 was about, and
FIX-1's own new C17 disclosure sharpens that ambiguity into a near-contradiction.

---

## 1. Verdict on A1: RESOLVED

**A1 is RESOLVED. The CRITICAL is closed per the gauntlet termination rule**
(`references/adversarial-gauntlet.md`: the rebuttal emitted FIX-1, FIX-1 was
applied to the draft in commit 76e2ab3, and this pass re-ran the attack on the
affected sections with the attack no longer CRITICAL).

Reasoning, independently verified this pass:

**(a) h4 truly draws from the training entity pool — confirmed at the code
level.** In `matrix-thinking/deltanet_rd/run_deltanet_rd.py`, the h4 source
`m3 = evaluate_pool(..., (*cfg.H_test, *cfg.H_extra), pools, ...)` (line 1038)
uses the default `use_heldout_entities=False`; the disjoint-pool diagnostic
`c17 = evaluate_pool(..., use_heldout_entities=True, ...)` (lines 1042–1045) is
the ONLY `use_heldout_entities=True` call. `grammar_rd.py:423` resolves the flag:
`name_ids = (pools.heldout_name_ids if use_heldout_entities else
pools.train_name_ids)`. So h4 (M3_held_out) draws from `train_name_ids`; the only
held-out axis is hop depth (M3 evaluated at H_test+H_extra hops, C17 evaluated at
the training hops H_train). This exactly matches the corrected testbed prose.

**(b) The new testbed/limitations prose is an ACCURATE characterization, not a
reworded overclaim.** `02_testbed.tex` now states h4 is graded "with the queried
entities drawn from the same 107-entity pool used in training; the held-out axis
is hop depth ... not entity identity," and explicitly says it "is not a test of
generalization to unfamiliar entities, which Section~\ref{sec:limitations}
reports separately and finds does not hold on this testbed." The original FALSE
"held-out entities and bindings" assertion is gone. This is faithful to the code.

**(c) The C17 collapse numbers in `07_limitations.tex` reproduce from the raws.**
I recomputed `checkpoints[-1].C17_heldout_entities` `recovered_frac@0.9` directly
from all 24 cell JSONs named in brief row C17:

- concat-md5 of the 12 d=64 cliff cells = `19ca8ed61c065710c627debb1ef3eb82`
  (brief: MATCH); concat-md5 of the 12 d=128 cells =
  `53bc17fb892489a6b820f654c21607e2` (brief: MATCH).
- **hop 1: `1.0` in all 24 of 24 cells** (paper: "1.0 at hop 1 in all 24
  archived cells" — MATCH).
- **hops 2 and 3: exactly `0.0` in 23 of 24 cells; the one remaining cell
  (K34/seed 131) reads `0.00017233` at hop 2 (0.0 at hop 3)**, which the paper
  rounds to `0.0002` (paper: "reads exactly 0.0 in 23 of those 24 cells (the
  remaining cell reads 0.0002)" — MATCH). Across hops 2 and 3 there is exactly
  one nonzero reading in all 48 (24×2) readings, and it is that 0.00017.
- As a provenance cross-check I also recomputed h4 (M3_held_out['4']) seed-means
  and they reproduce C1/C3 to the reported precision (K34→0.5676, K38→0.3316,
  K42→0.1177, K46→0.0434; all 12 d=128 cells h4=1.0).

The FIX-1 disclosure is quantitatively faithful. A1's specific defect — a FALSE
factual assertion in `02_testbed.tex` — no longer exists in the draft.

---

## 2. Verdict on the narrowing call: BROKEN for the abstract, survives for 03_cliff

The rebuttal's narrowing call (FIX-1 adjudication note): *"only 02_testbed makes
the FALSE claim; abstract/intro/03_cliff say 'held-out recall/4-hop
composition,' which reads correctly once the definition is fixed."*

- **`03_cliff.tex`: SURVIVES.** Every held-out reference here carries a hop-count
  qualifier: the figure caption reads "Held-out 4-hop recovery $h4$," and the
  body reads "the state supports at least 92 of the 107 entities through
  **held-out 4-hop composition**" (§\ref{sec:act2}). The "4-hop" qualifier fixes
  the held-out axis as the hop count; a reviewer cannot reasonably read these as
  entity-level generalization. No edit required.

- **`00_abstract.tex`: BROKEN (see A8).** The abstract's single use of "held-out"
  is unqualified — "the held-out recall frontier is a sharp transition" — and the
  abstract nowhere states that the held-out axis is hop depth. Read in isolation
  (as most reviewers read an abstract), "held-out recall" in an associative-memory
  paper most naturally parses as recall of held-out *items/entities*, which is
  precisely the reading A1 was about and precisely the reading the paper's own
  Limitations now says collapses to 0.0. The narrowing call's escape clause
  ("reads correctly *once the definition is fixed*") is exactly what fails,
  because the abstract cannot borrow the testbed's later definition.

- **`01_introduction.tex`: WEAKLY BROKEN (folded into A8).** The intro's "First"
  bullet says "held-out compositional recall does not decline gracefully with
  load" without defining the held-out axis. Milder than the abstract (the intro
  has surrounding context and the reader continues to the testbed), but it
  carries the same latent entity-vs-depth ambiguity and would benefit from the
  same one-word qualifier.

---

## 3. New attacks (sorted by severity)

### A8: The abstract's unqualified "held-out recall frontier" re-opens the A1 axis standalone, and FIX-1 now contradicts it

**Severity:** SERIOUS
**Type:** claim-scope / standalone-ambiguity (new defect surfaced by the FIX-1 wave)

**Attack.** `00_abstract.tex` line 7–8:

> "First, at state dimension $d_{\mathrm{state}}=64$ the **held-out recall
> frontier** is a sharp transition, not a graceful decline..."

This is the abstract's *only* use of "held-out," and it is unqualified — the
abstract never says the held-out axis is hop depth. An abstract is read in
isolation. In an associative-recall paper the default reading of "held-out
recall" is recall of held-out *associations/entities*. Under that reading the
sentence is FALSE — held-out-entity recall does not have a sharp frontier at
$K/d=0.5455$; it is flat at 0.0 for all loads at hop $\ge 2$ (verified, §1c
above). Under the intended reading (held-out hop depth) it is TRUE. That is
textbook standalone ambiguity on the paper's single most load-bearing framing —
the exact axis the round-1 CRITICAL turned on.

FIX-1 makes this worse, not better: the newly-added C17 paragraph in
`07_limitations.tex` now openly states that disjoint-entity recall "collapses at
hop depth 2 and above ... reads exactly 0.0 in 23 of those 24 cells." A reviewer
who reads the abstract's "held-out recall frontier" as entity-level and then
reaches this Limitations paragraph sees a direct self-contradiction: abstract
says a sharp *held-out* frontier at 0.5455; limitations says *held-out* (entity)
recall is 0.0 everywhere. The honest disclosure that resolved A1 sharpens the
abstract's ambiguity into an apparent contradiction, because the abstract was
never rescoped alongside the testbed and limitations.

The intro's "held-out compositional recall" (line 38) is the same defect, milder.

**Supporting evidence.**
- `00_abstract.tex:7–8` (quoted); `01_introduction.tex:38` ("held-out
  compositional recall").
- `07_limitations.tex:16–28` (the FIX-1 C17 paragraph: held-out-entity recall
  = 0.0 at hop $\ge 2$), reproduced from raws this pass (24/24 cells).
- Contrast with the correctly-qualified `03_cliff.tex` caption ("Held-out 4-hop
  recovery") and `02_testbed.tex` (held-out axis = hop depth) — proving the fix
  wave already knows the qualifier is needed and simply did not apply it to the
  abstract/intro.

**What the paper would need to do to defuse this.** A one-word qualifier in the
abstract: "the held-out **4-hop** recall frontier" or "the held-out-**depth**
recall frontier," and the same in the intro's "First" bullet ("held-out
**4-hop** compositional recall"). No new experiment, no number change. This is a
bounded scoping edit, but it *is* a needed edit — the narrowing call's claim that
the abstract needs none is wrong.

---

### A9: The "trained-pool readouts stay at or near ceiling" clause overstates hop-3 retention for the high-load d=64 cells

**Severity:** MINOR
**Type:** number provenance / precision (self-consistency of the FIX-1 disclosure)

**Attack.** `07_limitations.tex:26–28`, closing the C17 paragraph:

> "...it reads exactly $0.0$ in 23 of those 24 cells (the remaining cell reads
> $0.0002$), **while the same checkpoints' trained-pool readouts stay at or near
> ceiling.**"

The hop-matched trained-pool comparison is M2_in_distribution at hops 1/2/3
(trained entities, the same hops C17 is measured at). Recomputed from the raws
this pass: M2 is 1.0 at hop 1 and hop 2 in all 24 cells, but at **hop 3 it
degrades under load** — for the high-load d=64 cells the M2 hop-3
`recovered_frac@0.9` is 0.73 (K42, three seeds 0.7335/0.7453/0.7286) and **0.56–0.60**
(K46, three seeds 0.5977/0.5792/0.5578). 0.56 is not "at or near ceiling." So
the blanket clause is accurate only at hops 1–2 (and at hop 3 for d=128 and
low-load d=64); it overstates trained-pool retention at hop 3 for the collapsed
d=64 loads.

The *core* contrast the paragraph draws survives — held-out entities collapse to
exactly 0.0 at hop $\ge 2$ while trained entities recover far better (0.56–1.0),
and at hop 2 specifically the contrast is a clean 1.0 vs 0.0 in all 24 cells. The
overstatement is confined to the phrase "at or near ceiling" applied without a
hop/load qualifier.

**Supporting evidence.** Recomputed M2_in_distribution hop-1/2/3
`recovered_frac@0.9` for all 24 fig-1 cells (final checkpoint): worst hop-3 value
= 0.5578 (K46/seed 432, d=64). Same JSONs and md5s as C17 (§1c).

**What the paper would need to do to defuse this.** Qualify the clause: e.g.,
"while the same checkpoints' trained-pool readouts stay at ceiling at hops 1–2
and remain well above the held-out collapse at hop 3," or restrict the ceiling
claim to the matched hops where it holds. One-clause edit, no number change.

---

## 4. Attacks I considered but decided were weak

- **FIX-3 byte-footprint arithmetic wrong.** Checked: $428 \cdot d$ bytes =
  27,392 B at d=64 = 26.75 KiB (67% above the 16 KiB state), 54,784 B at d=128 =
  53.5 KiB (84% of the 64 KiB state). All correct; the rebuttal's /1024 unit
  correction (26.75, not 27.4) is properly reflected in both `02_testbed.tex` and
  brief row C18. No defect.
- **FIX-4 citations hallucinated or mis-IDed.** Checked all four against
  `bundle/citation-verification.json` and my own knowledge: behrouz2025titans
  (2501.00663), munkhdalai2024infiniattention (2404.07143), yang2024gateddelta
  (2412.06464), arora2024based (2402.18668) are all real papers with correct
  arXiv IDs, authors, and titles. `06_related.tex` places them sensibly (gated
  delta in the family sentence; Titans/Infini-attention as bounded test-time
  memory; Based alongside Zoology). No defect.
- **FIX-5 reasoning-proxy paragraph overclaims.** The paragraph
  (`01_introduction.tex:73–85`) is heavily hedged ("The synthetic grammar does
  not model any particular reasoning task") and is presented as a motivational
  analogy, not an empirical result. A reviewer may find the analogy a stretch,
  but it does not assert an unsupported finding. Weak.
- **Abstract's "no transition through K/d=0.9375" is undercut by the K=90
  non-replication.** The Limitations K=90 fresh reading is 0.9725, still
  no-collapse; a transition would mean dropping toward 0. The absence-of-collapse
  claim is robust to the non-replication, and the Table 1 dagger + limitations
  caveat (FIX-2) already disclose the uncertainty. Not a new defect.
- **Future-dated (2026) base-draft citations unverifiable.** barnfield2026sharp
  (2605.05189), nazari2026rank (2602.04852), sun2026staterank (2602.02195),
  mishra2026m2rnn (2603.14360) post-date my knowledge cutoff and I cannot confirm
  them from training knowledge; their arXiv-ID month codes are internally
  consistent and `citation-verification.json` records published dates. These are
  base-draft citations, not FIX-wave additions, and are outside this pass's
  scope; flagging only as a note for the citation-verification stage to confirm
  live, not as an attack.
- **"24 archived cells behind Figure 1" undercounts the figure's backing.**
  Figure 1 left (d=64) also plots the archived K=16/32/48 loads, so the figure is
  backed by more than 24 cells; the 24 are the newly-run cells that carry the
  C17 field. The phrasing "24 archived cells" is defensible (24 *of* the archived
  cells) and not misleading. Below MINOR; noted for completeness only.

## 5. New citations for Related Work

None beyond the four FIX-4 additions already present and verified. The
related-work coverage of fixed/compressive-memory competitors (Titans,
Infini-attention, Gated Delta, Based, Zoology, DeltaProduct) is now adequate for
the venue.

---

## Security note

No fake `<system-reminder>`-style blocks, concealment instructions, or anomalous
injected directives were encountered in any tool output during this pass.

---

## Resolution (writer, applied post-report)

A8 (SERIOUS) and A9 (MINOR) are bounded wording edits with no number change
(A8) or a one-clause precision fix backed by the same C17 raws (A9); applied
directly rather than routed through a full defense/rebuttal cycle, consistent
with the gauntlet's "SERIOUS and MINOR attacks may be resolved by a fix... they
do not block termination" rule (`references/adversarial-gauntlet.md`). The
independent style (03) and format (05) audits that follow re-read the whole
draft cold and will catch anything these edits miss.

- **FIX-9 (A8, SERIOUS).** `00_abstract.tex`: "the held-out recall frontier"
  -> "the held-out 4-hop recall frontier". `01_introduction.tex`: "held-out
  compositional recall" -> "held-out 4-hop compositional recall". No number
  changed; abstract word count remains in the 200-230 band (+1 word).
- **FIX-10 (A9, MINOR).** `07_limitations.tex`: "trained-pool readouts stay at
  or near ceiling" -> "trained-pool readouts stay at ceiling at hops 1 and 2
  and remain well above the held-out collapse at hop 3, where they decline
  under the highest tested loads (as low as $0.56$ at $d_{\mathrm{state}}=64$,
  $K{=}46$)." New number (`0.56`) traced by extending brief.md row C17 to name
  the `M2_in_distribution` field alongside `C17_heldout_entities` and record
  the recomputed hop-3 range (K42 seed-mean 0.736; K46 seed-mean 0.578, worst
  single seed 0.5578).

**A1 status after this resolution: CLOSED.** No CRITICAL remains open in
round 1. Next: style critique (03) and format audit (05) on the current draft.
