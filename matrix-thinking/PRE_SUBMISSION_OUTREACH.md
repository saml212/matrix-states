# Pre-Submission Outreach Package — Matrix-CODI Rank-Blindness Paper

**Status:** Draft outreach plan for "The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA." Paper is in `submissions/icml-mi-workshop-2026/`.

**Goal:** Send the de-anonymized arXiv-style draft to 4–5 specific researchers for ~14-day candid review BEFORE the ICML MI Workshop deadline (May 8, 2026 AOE) and BEFORE the arXiv drop.

**Why pre-submission outreach matters more than after:**
- Catches mechanism / framing errors while you can still fix them
- Builds the citation network early — these are the people most likely to cite the paper later
- Converts the most adversarial reviewer attack vectors to neutral *before* a hostile reviewer encounters them at workshop review

---

## Two paper versions

| Version | Length | Audience | Identity | Where it lives |
|---|---|---|---|---|
| **ICML MI Workshop** | 8 pages two-column (~5,912 body words) | Workshop reviewers | Anonymous (double-blind on OpenReview) | `submissions/icml-mi-workshop-2026/main.pdf` after compile |
| **arXiv preprint** | Same body + extended appendix | Public, pre-submission readers, future citations | De-anonymized (Sam Larson, pebbleml.com) | TBD — submit to arXiv after workshop deadline lands or shortly before |

**For pre-submission outreach, send the arXiv version.** The reviewer needs to know who wrote it (you're asking for their goodwill) and the appendix gives them the data tables they'll want to scrutinise. Their feedback then informs both the ICML camera-ready and the public arXiv release.

**Critical timing detail:** ICML MI Workshop review is double-blind. If any of these pre-submission reviewers ends up as your assigned workshop reviewer, that creates a conflict-of-interest problem. To mitigate, in the outreach email, **explicitly note that this is "pre-submission feedback, not a formal review"** — most senior researchers know to recuse themselves, but stating it is cleaner.

---

## Reviewer list — 4 primary + 2 secondary

Drawn from the R5 adversarial team's outreach analysis, filtered to **current-paper relevance** (not the future matrix-mvp work). Reviewers ranked by ROI on the rank-blindness paper specifically.

### Tier 1 — must reach out

#### 1. **Guillaume Rabusseau** (Mila / Université de Montréal)
- **His paper:** *"The Illusion of Superposition?"* — arXiv 2604.06374 (April 2026), with Adel Rizvi-Martel and Marius Mosbach
- **Why his opinion matters specifically:** Their paper observed the phenomenon (fine-tuned COCONUT reaches 96.6% on ProsQA without latent tokens — the "illusion"). **Your paper gives the mechanism** (the gradient can't see rank through any of the readouts they would have considered). This is a direct scientific dialogue. Contact:
- **What he'll catch:** (a) whether your "all four positive controls flat" framing genuinely supersedes their result or sits alongside it; (b) any over-reach in the rank-blindness mechanism claim relative to what the experiments actually show; (c) the linear-probe-AUC comparison methodology (matrix-CODI 0.673 vs vanilla GPT-2 0.846 — does this match their entity-probe results?)
- **Email address:** `guillaume.rabusseau@umontreal.ca` (publicly listed at Mila)
- **Tone:** technical, peer-to-peer; cite his paper in the first paragraph
- **Predicted response:** he WILL respond — your paper extends his finding, and academic norms strongly favour responding to direct intellectual descendants

#### 2. **Tom Goldstein** (University of Maryland, tomg-group-umd)
- **His paper:** *"Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer"* — arXiv 2507.02199 (Huginn-3.5B, COLM 2025 Workshop)
- **Why his opinion matters specifically:** The closest methodological precedent for your work. They tracked rank trajectories in depth-recurrent latent reasoning and reported them inconsistent. Your paper's finding (matrix-CODI rank trajectories under flat rank-k ablation are essentially measurement-uninformative) is convergent evidence from a different architecture. Together you tell a story.
- **What he'll catch:** (a) whether your CODI-style training is mechanistically distinct enough from depth-recurrent that "rank is gradient-blind in CODI" actually generalizes or is a one-off; (b) whether the negative-control vanilla-GPT-2-SFT result you report (also flat) tightens or weakens the mechanism claim
- **Email address:** `tomg@umd.edu`
- **Tone:** brief, specific; reference 2507.02199 as the closest prior result
- **Predicted response:** medium-likely (40–60%) — Goldstein is busy but cites direct rebuttals/extensions of his group's work

#### 3. **Neel Nanda** (Google DeepMind, mechanistic interpretability)
- **Why his opinion matters specifically:** The most-cited researcher on mech-interp / superposition / feature geometry in transformers. He has *written extensively about rank and superposition* and will give an honest, uncompromising verdict on whether your mechanism claim is real or a measurement artifact. Plus: **if he tweets it, you reach the entire mechinterp community in one move.** That's worth more than any single workshop slot.
- **What he'll catch:** (a) whether the "rank-blind gradient" framing maps onto how he thinks about feature/rank superposition; (b) whether the linear-probe gap is explained by rank-blindness specifically or by some other readout-side issue; (c) whether your story should engage SAEs / transcoder methodology (probably not for THIS paper, but he'll flag if it should)
- **Contact:** he's responsive on X (@NeelNanda05) more than email. **Recommend a different protocol for him:** post a 2-paragraph X thread tagging him with the rank-trajectory animation + a link to the arXiv draft. Cold X DM is OK but lower hit rate than tagged thread.
- **Tone:** technical, ungushy
- **Predicted response:** ~30% likely on a cold ask (he gets a lot); ~50%+ if you send him the arXiv link AND an X thread tagging him

#### 4. **Imanol Schlag** (ETH AI Center, Apertus project)
- **His paper:** *"Linear Transformers Are Secretly Fast Weight Programmers"* — Schlag, Irie, Schmidhuber, ICML 2021 (and many follow-ups)
- **Why his opinion matters specifically:** The intellectual ancestor of matrix-valued state architectures. Your matrix-CODI bottleneck is a CODI-era descendant of the FWP line. While the current paper is more about the *negative result* than the architectural lineage, Schlag's sign-off (or specific objection) on the framing of "matrix-valued latent → rank as observable" matters because the FWP literature is the obvious citation target for any review.
- **What he'll catch:** (a) whether the matrix-CODI architecture is a fair instantiation of the matrix-state lineage or a methodological one-off; (b) whether the failure mode generalizes to other matrix-state architectures (M²RNN, RWKV-7, DeltaNet) — this affects how you frame §6 Discussion
- **Email address:** Through ETH AI Center directory or via ICML 2025 contact info (publicly available)
- **Tone:** academic, peer-to-peer; reference 2021 paper directly
- **Predicted response:** medium-likely (35–50%) — he's active and known for responding to direct technical asks

### Tier 2 — optional, send if Tier 1 takes <half a day

#### 5. **Jianyu Zhang or Yuandong Tian** (CoT2 paper, arXiv 2505.23648, ICLR 2026)
- **Why:** Theorized the parallelism-vs-dimension framework. Your finding (rank is gradient-blind in CODI training) is adjacent to their setting. They predicted parallelism scales with embedding dimension; your paper suggests the *measurement methodology* used to test that prediction is broken in CODI-style training. Useful for them; useful for you (citation, possible co-citation).
- **Lower priority because** their paper is theoretical and you've already cited it; their feedback won't change the experimental story.

#### 6. **Adel Rizvi-Martel** (PhD student at Mila, lead author of Illusion paper)
- **Why:** Same group as Rabusseau (#1). Contacting one of the two is sufficient. Pick Rabusseau (senior) for hierarchical reasons — but if you want a deeper technical dialogue specifically on entity-probe methodology, Rizvi-Martel is closer to the experimental details.
- **Recommend:** contact Rabusseau as primary; CC Rizvi-Martel.

---

## Email template — single template, per-reviewer customization

Subject lines (pick one per recipient):

- **Rabusseau:** "Pre-submission read on a Mila-adjacent ICML MI submission?"
- **Goldstein:** "Pre-submission read: rank-blindness in CODI, sibling result to your Huginn 2507.02199"
- **Nanda:** "Pre-submission feedback on a rank/superposition mech claim — have ~2 paragraphs?" (X DM) or "ICML MI submission — quick read?" (email)
- **Schlag:** "Pre-submission read: matrix-valued CODI rank-indifference, FWP-adjacent"

### Body template (customize the bracketed sections per recipient)

```
Hi [First name],

I'm Sam Larson at Pebble (pebbleml.com), submitting a paper to the ICML 2026
Mechanistic Interpretability Workshop in three weeks.

The paper extends [PAPER-SPECIFIC HOOK — see below] by giving a mechanism for
rank-indifference in CODI-style matrix-valued latent reasoning models. Across
four training regimes plus four nonlinear-in-Z positive-control readouts
(bilinear, bilinear+GELU, SVD-augmented, quadratic), the rank-k truncation
ablation curve is flat. A three-seed replication shows accuracy at 81.5±1.2pp
while the final effective rank of Z spans {4, 12, 13}. A linear probe on Z
underperforms a raw pretrained GPT-2 hidden state at target prediction
(AUC 0.673 vs 0.846).

The mechanism we propose: the matrix-CODI training objective produces
rank-indifferent gradients through the full chain rule, not just through a
flatten-then-project linearity. The four positive-control readouts test that
mechanism and falsify the simpler "Jacobian linearity" version of the claim —
they're nonlinear in Z but the curves stay flat.

[PAPER-SPECIFIC ASK — see below]

This is pre-submission feedback, not a formal review — the workshop submission
will be double-blind on OpenReview and I'm happy to recuse you from any review
slot that lands on this paper later.

Draft is at: [arXiv link if posted, otherwise: a 14-day private link to the
PDF — Dropbox/Google Drive/anonymous.4open.science]

If you're up for it, ~30 minutes of your time would be enormously helpful and
I'll credit you in the acknowledgements (or omit you if you prefer).

Thanks,
Sam Larson
Pebble — pebbleml.com
```

### Per-reviewer customisation

#### → Rabusseau (Mila)

**[PAPER-SPECIFIC HOOK]:**
> "extends your *Illusion of Superposition* paper (arXiv 2604.06374) with Adel Rizvi-Martel and Marius Mosbach"

**[PAPER-SPECIFIC ASK]:**
> "Two specific things I'd value your read on:
> 1. The matrix-CODI mechanism we propose for rank-indifference is a structural extension of what your paper observed empirically. Does the framing accurately position our work relative to yours, or am I overclaiming the mechanism scope?
> 2. The linear-probe AUC gap (Z at 0.673 vs raw GPT-2 hidden at 0.846) is the corroborating evidence. Does this match what you'd expect from your entity-probe methodology applied here?"

#### → Goldstein (UMD)

**[PAPER-SPECIFIC HOOK]:**
> "is methodologically adjacent to your group's Huginn rank-trajectory probing (arXiv 2507.02199) — your finding of inconsistent rank trajectories in depth-recurrent latent reasoning is convergent with our finding of rank-indifferent gradients in CODI"

**[PAPER-SPECIFIC ASK]:**
> "Your paper showed rank trajectories noisy and inconsistent on Huginn-3.5B. Our paper shows rank-k ablation is flat to within 0.6pp on a CODI-style matrix bottleneck — different architecture, same overall message. Two questions:
> 1. Is the mechanism we describe (rank-indifferent gradients through CODI's training objective) compatible with what you observed in Huginn, or distinct?
> 2. Does our negative-control on vanilla GPT-2 SFT (also flat) strengthen or weaken our mechanism claim relative to your setting?"

#### → Nanda (DeepMind)

**[PAPER-SPECIFIC HOOK]:**
> "addresses a question I think you've written about — whether rank in latent-CoT models is doing real work"

**[PAPER-SPECIFIC ASK]:**
> "Specific feedback I'd value: the mechanism claim — that the matrix-CODI training objective produces rank-indifferent gradients through the full chain rule, not just through readout linearity — is the part most likely to be wrong or overreaching. Three seeds yield Z effective rank in {4, 12, 13} at matched accuracy; four nonlinear-in-Z readouts also flat. Is this the right mechanistic story, or am I missing a simpler explanation?"

(For X DM, compress to 2 paragraphs and link the arXiv draft. Don't include the recusal note — X DMs aren't formal-review territory.)

#### → Schlag (ETH)

**[PAPER-SPECIFIC HOOK]:**
> "is FWP-lineage adjacent — the matrix-valued latent in our setup descends from the line your 2021 ICML paper started"

**[PAPER-SPECIFIC ASK]:**
> "Two questions I'd value:
> 1. Our matrix-CODI bottleneck places matrix structure on the explicit chain-of-thought feedback path rather than as a recurrent fast-weight state. Does that placement break the FWP-lineage analogy, or is it a legitimate variant?
> 2. The negative result (rank-indifferent gradient through the full chain rule) — does this generalize to other matrix-state architectures in the FWP family, or is it CODI-specific?"

---

## Logistics and timing

### When to send

**Send all four emails today or tomorrow.** ICML deadline is May 8 AOE, ~3 weeks out. Friendly reviewers need 14 days to read. Sending after April 24 risks not getting feedback before submission.

### What to attach / link

You have two options:

**Option A — link to the arXiv version (preferred):**
1. Pre-post the de-anonymized arXiv version *now* (or by April 25 latest) under your name.
2. Email the arXiv link in the outreach.
3. The de-anonymized version is on arXiv; the workshop version remains anonymous.
4. **Risk:** if the workshop has a strict no-prior-distribution policy, the arXiv post could conflict. Check ICML MI's CFP — most workshops are fine with arXiv preprints.

**Option B — private 14-day link:**
1. Don't pre-post.
2. Email a Dropbox / Google Drive / anonymous.4open.science link valid for ~14 days.
3. Set the link to expire on ~May 6 so it doesn't leak past submission.

For ICML MI Workshop specifically, Option A (arXiv pre-post) is fine and is in fact normal practice. The workshop is non-archival. Multiple submissions to arXiv + workshop are explicitly allowed.

**Recommendation: Option A.** Post arXiv this week (de-anonymized version with extended appendix), use that link in all 4 emails.

### Follow-up cadence

- Send all 4 emails on Monday/Tuesday this week.
- If no response after 7 days, send a single follow-up: "Quick bump — does timing work? Happy to wait until after ICML deadline if it's a bad week."
- Do NOT chase a third time. If the second outreach is silent at day 14, that researcher is unavailable for this round; shift to the next-tier list.

### Track responses in `outreach_log.md`

In `matrix-thinking/outreach_log.md`, log per-reviewer:
- Email sent date
- Response date
- Verdict (positive / specific objection / negative / silent)
- Specific feedback lines for citation in acknowledgements
- Any agreement on follow-up exchanges

This is also useful evidence if a reviewer becomes hostile post-submission ("you didn't engage prior literature" — well, here's the email thread).

### What to do with negative feedback

**If a theorem / mechanism error is flagged:** halt arXiv revision and ICML submission until fixed. Three-week timeline can absorb 3–5 days of mechanism-fix work; more than that and you defer to next venue.

**If framing / scope issue:** edit immediately. The reviewer caught what would have killed the workshop submission too; cheap to fix.

**If methodological challenge (different mech expected):** add the reviewer's expected mechanism as an alternative hypothesis in §6 Discussion, with a footnote like "we are grateful to [reviewer] for suggesting this alternative." This defuses it both for them and for any post-submission reviewer who would have raised the same point.

**If a citation gap:** add the citation. Read the paper, check if it invalidates anything. If yes, engage explicitly in the body.

**Never** "defend" against pre-submission feedback. These reviewers are simulating the actual reviewer pool. Defending = pretending the simulation is wrong.

---

## After the workshop submission

The same 4 reviewers can be re-engaged in two ways:

1. **Acknowledgement-only credit** in the camera-ready / arXiv version. Default if they responded.
2. **Active citation thread** if they responded substantively. Cite their paper in §6 Discussion AND link their feedback in the email exchange ("we are grateful to X for [feedback]").

For the *next* paper in the line (the matrix-mvp future work — the one we built v4 for), you'll want to escalate Schlag, Irie, and Grazzi to Tier 1 and reach out to them with the matrix-mvp draft. That outreach reuses this template structure.

---

## Two-paragraph version of the paper for cold outreach

If you want a shorter cold-pitch (e.g., for X thread), here's a draft:

> Continuous chain-of-thought models compress reasoning into latent tokens. Matrix-valued variants introduce a single-sample structural observable — the rank of the latent matrix Z — and the natural prediction is that truncating Z to low rank should hurt accuracy when the task needs more than k reasoning paths.
>
> We trained a matrix-CODI model on ProsQA across four regimes plus four nonlinear-in-Z positive-control readouts (bilinear, bilinear+GELU, SVD-augmented, quadratic). All eight rank-k truncation curves are flat. Three-seed replication: 81.5±1.2pp accuracy with effective rank of Z spanning {4, 12, 13}. The training objective is rank-indifferent through the full chain rule — not just through readout linearity. Rank-k ablation is not a valid probe for superposition in CODI-trained matrix latents. Linear probe on Z under-fires raw GPT-2 by 0.17 AUC. We argue the readout's effective Jacobian rank, not its in-principle expressiveness, gates what training can shape.
>
> [Link to arXiv]

That's the elevator pitch. Edit per recipient as needed.

---

## Action checklist for today

1. **Today**: post de-anonymized arXiv version (extended appendix, named authors). If not ready, set up 14-day private links.
2. **Today**: send 4 outreach emails (Rabusseau, Goldstein, Nanda via X DM, Schlag).
3. **Day 3**: log responses, follow up on silences.
4. **Day 7**: if any 2 of 4 responded substantively, you have your pre-submission review feedback.
5. **Day 10**: integrate feedback into camera-ready and arXiv revisions.
6. **Day 14**: ICML submission ready; final polish week.

This is the first paper in what will be a multi-paper programme — the outreach pattern repeats for each subsequent submission. The template is reusable.
