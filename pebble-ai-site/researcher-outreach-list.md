# Researcher Outreach List — Pebble AI
**Updated:** April 16, 2026 (full refresh, Bay Area coffee focus)
**Strategy:** Target junior researchers (PhD students, postdocs, early RS) actively publishing in 2025-2026. Lead with results + honest negatives. Frame Pebble's work as extending theirs, not competing.

**Confidence tags:**
- ✓ Verified — affiliation and recent work confirmed via lab page / recent paper / personal site
- ⚠ Needs check — one or more details (advisor, current role, Bay Area status) not fully verified; worth a 30-sec sanity check before emailing

---

## TOP 5 COFFEE PICKS (start here)

These five are Bay Area, junior, actively publishing in 2025-2026, and have a specific connection to Pebble's matrix-rank-as-thought-complexity angle. They are the best bang-for-buck cold emails.

1. **Julie Kallini** — Stanford NLP, 3rd-year PhD. MrT5 (ICLR 2025) on dynamic byte merging. Directly overlaps with Pebble's byte-level angle.
2. **Houjun Liu** — Stanford NLP. ThoughtBubbles (2025) on unsupervised parallel latent thinking. Directly overlaps with Pebble's matrix-token iterative refinement.
3. **Niklas Muennighoff** — Stanford PhD. s1: Simple Test-Time Scaling (2026). The current "budget forcing" SOTA for test-time compute. Pebble's passage-level uncertainty gate is a natural next step.
4. **Druv Pai** — Berkeley EECS PhD, Yi Ma lab. Works on rate reduction and structured representations. Pebble's rank enrichment is in his language.
5. **Jordan Juravsky** — Stanford PhD (Mirhoseini lab). Large Language Monkeys / inference-time scaling laws. Junior, responsive, directly in the test-time-compute conversation.

If Sam can only send five emails this month, send these five.

---

## TIER 1 — BAY AREA JUNIOR RESEARCHERS (highest ROI for coffee)

### Stanford NLP / CoCoLab / Scaling Intelligence Lab

**Julie Kallini** ✓
- **Role:** 3rd-year PhD, Stanford CS (NLP Group). Advisors: Christopher Potts, Dan Jurafsky.
- **Key paper:** *MrT5: Dynamic Token Merging for Efficient Byte-level Language Models* (ICLR 2025) — cuts byte sequence length ~75% vs ByT5 via learned dynamic merging.
- **Website:** https://juliekallini.com
- **Why Pebble:** MrT5 merges bytes compositionally; Pebble does structured (matrix) composition on bytes. Same efficiency premise, different algebraic structure.
- **Angle:** "Your MrT5 work shows dynamic byte composition beats static ByT5. We've been running matrix-structured composition on bytes — curious whether the two ideas compose."

**Houjun Liu** ✓
- **Role:** Senior undergrad/researcher, Stanford NLP. Working with Manning's group.
- **Key paper:** *ThoughtBubbles: Unsupervised Method for Parallel Thinking in Latent Space* (2025).
- **Website:** https://nlp.stanford.edu/~houjun/ | https://www.jemoka.com/
- **X:** @houjun_liu
- **Why Pebble:** ThoughtBubbles shows parallel latent thinking emerges unsupervised from LM loss. Pebble measures the same phenomenon numerically via matrix rank.
- **Angle:** "ThoughtBubbles shows parallel thinking emerges unsupervised from LM loss. We've been measuring that same emergent parallelism via matrix rank — would love your take."

**Yijia Shao** ⚠ (verify current advisor before sending)
- **Role:** PhD, Stanford CS. Advisor not cleanly confirmed — Quiet-STaR co-authorship is verified; agent claimed Manning as advisor but this should be double-checked (STORM work is Diyi Yang's lab).
- **Key paper:** *Quiet-STaR* (2024) as co-author; also STORM (2024).
- **Website:** https://cs.stanford.edu/~shaoyj/
- **X:** @EchoShao8899
- **Why Pebble:** Quiet-STaR is the direct predecessor of Pebble's matrix-token thinking — she implemented internal thought generation.
- **Angle:** "Quiet-STaR's internal thought tokens set a template; we extended it to structured matrix tokens where rank gives a measurable complexity axis."

**Niklas Muennighoff** ✓
- **Role:** PhD, Stanford. Co-author on the s1 line of work.
- **Key paper:** *s1: Simple Test-Time Scaling* (EMNLP 2025 / Jan 2026) — 1K curated traces + budget forcing beats o1-preview by 27% on math benchmarks.
- **Why Pebble:** s1's "budget forcing" is a crude fixed-budget scheduler. Pebble's passage-level uncertainty gate is the learned version of the same idea.
- **Angle:** "s1's budget forcing is elegant. We've been working on a learned uncertainty-driven analogue — passage-level rather than per-token. Would like to compare notes."

**Jordan Juravsky** ✓
- **Role:** PhD, Stanford (Azalia Mirhoseini's Scaling Intelligence Lab).
- **Key paper:** *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* (2024-2025).
- **Why Pebble:** Monkeys shows pass@k scaling follows power laws — coverage scaling. Pebble's matrix-rank-gated thinking is an orthogonal lever on the same question.
- **Angle:** "Monkeys shows coverage scaling follows a power law. We're exploring a depth-scaling analogue grounded in matrix rank and passage-level uncertainty — curious how they compose."

### UC Berkeley — Yi Ma Lab (structured representations hub)

**Druv Pai** ✓
- **Role:** PhD, Berkeley EECS, Yi Ma lab.
- **Key work:** Rate reduction / CRATE-style structured representations, 2025 follow-ups.
- **Why Pebble:** Yi Ma's group explicitly treats representation rank as the thing being optimized (rate reduction). Pebble's rank-enrichment phenomenon is in their native vocabulary.
- **Angle:** "Your lab's rate-reduction framing treats rank directly as the training target. We've been measuring a rank enrichment phenomenon at inference time — output head determines whether rank grows or collapses. Would love to sanity-check the framing."

**Ziyang Wu** ⚠ (confirm still at Berkeley, not graduated)
- **Role:** PhD, Berkeley, Yi Ma lab.
- **Key work:** Token compression / structured attention.
- **Why Pebble:** Overlaps with Pebble's matrix-token compression story.
- **Angle:** Similar to Druv — rank/structure as first-class citizens in the representation.

**Brent Yi** ⚠ (confirm recent publications)
- **Role:** PhD, Berkeley.
- **Key work:** Geometric / equivariant attention.
- **Angle:** Position matrix tokens as a lightweight structured prior analogous to equivariance.

### Stanford — Structured Reps / Interpretability

**Albert Gu** ⚠ (NOTE: Albert Gu is CMU faculty, not Stanford. He co-founded Cartesia in SF. Bay Area presence is real but not his primary affiliation.)
- **Role:** Assistant Professor, CMU; Co-founder, Cartesia AI (SF).
- **Key work:** Mamba, SSM line of research. Cartesia productionizes SSMs.
- **Why Pebble:** SSMs are structured-representation wins at production scale. Pebble's matrix tokens are adjacent.
- **Angle:** "Cartesia is the clearest proof structured representations can beat transformer defaults at scale. We're testing a different structure axis (matrix rank). Could I buy you coffee next time you're in SF?"

**Karan Goel** ⚠ (confirm current role)
- **Role:** Co-founder, Cartesia AI (SF).
- **Why Pebble:** Same as Albert — production validation of structured reps.

### Anthropic (San Francisco) — Interpretability

**Jack Lindsey** ✓
- **Role:** Research scientist, Anthropic (SF). Leads "model psychiatry" / circuit-tracing team.
- **Key work:** *On the Biology of a Large Language Model* (2025); *Emergent Introspective Awareness* (2025); Scaling Monosemanticity (2024).
- **Website:** https://jlindsey15.github.io/
- **X:** @Jack_W_Lindsey
- **Why Pebble:** Circuit tracing reveals qualitative reasoning pathways. Pebble's rank-as-complexity is a quantitative dual — how much independent reasoning capacity a representation holds.
- **Angle:** "Circuit tracing maps reasoning pathways qualitatively. We've been measuring a quantitative dual via matrix rank — output-head-dependent rank enrichment at inference time. Does the framing track?"

**Michael Hanna** ⚠ (confirm still at Anthropic post-fellowship)
- **Role:** Anthropic Fellow (based Berkeley through late 2025, then extended).
- **Key work:** circuit-tracer open-source library; MIB mechanistic interpretability benchmark (ICML 2025).
- **Website:** https://hannamw.github.io/
- **Angle:** "circuit-tracer is the tool I'd use to probe rank-enrichment in a frontier model. Want to chat about what a 'complexity' probe on top of your library would look like?"

### xAI / humans& (San Francisco)

**Eric Zelikman** ✓
- **Role:** CEO/co-founder, humans& (SF, 2025). Stanford PhD 2025 (advisors: Nick Haber, Noah Goodman). Early xAI employee.
- **Key work:** Quiet-STaR (2024); Fast Quiet-STaR (EMNLP 2025 Findings); STaR (original).
- **Website:** https://zelikman.me/
- **X:** @ericzelikman
- **Why Pebble:** Pebble's matrix thoughts are Quiet-STaR with a measurable complexity axis.
- **Angle:** "Quiet-STaR and Fast Quiet-STaR set the template for internal thought generation. We've given those thoughts structure — rank as a measurable complexity axis. Would like your opinion."

### Google DeepMind (Mountain View)

**Shikhar Murty** ✓
- **Role:** Research Scientist, Google DeepMind (MTV). Recent Stanford PhD (Manning).
- **Key work:** ThoughtBubbles (2025) co-author; MrT5 (ICLR 2025) co-author; stack-augmented reasoning.
- **Website:** https://murtyshikhar.github.io/
- **X:** @ShikharMurty
- **Why Pebble:** Sits at the intersection of byte-level + latent reasoning + structure — Pebble's three pillars.
- **Angle:** "You're one of the few people who's published on both ThoughtBubbles and MrT5. Pebble sits exactly at that intersection (matrix tokens on bytes, iterative refinement). Would love 20 minutes."

---

## TIER 2 — BAY AREA SENIOR RESEARCHERS (harder cold reach, useful as warm intros)

**Christopher Potts** — Stanford. Advisor on MrT5. Senior but approachable. Julie Kallini is a faster path.
**Christopher Manning** — Stanford. Advises much of the Stanford NLP latent-reasoning work. Extremely hard to cold-reach directly.
**Noah Goodman** — Stanford CoCoLab. Advised Zelikman (Quiet-STaR). Theoretical grounding. Best reached via grad student warm intro.
**Azalia Mirhoseini** — Stanford, Scaling Intelligence Lab. Directly leads test-time-compute scaling research. Reach via Juravsky/Muennighoff.
**Percy Liang** — Stanford, s1 co-author. CRFM director. Extremely hard to cold-reach; go through s1 students.
**Tatsu Hashimoto** — Stanford, s1 co-author. Teaches CS 336 (reasoning LLMs). Marginally more approachable than Percy.
**Chelsea Finn** — Stanford. SpeedTuning (when to stop). Complementary angle: Pebble controls depth, she controls termination.
**Sanmi Koyejo** — Stanford. Pass@k scaling laws.
**Yi Ma** — Berkeley. The structured-reps godfather. Reach via Druv Pai.
**Sergey Levine** — Berkeley. RL + adaptive compute framing (optimal control view).
**Pieter Abbeel** — Berkeley / Amazon AGI. Information-gain RL.
**Dan Klein** — Berkeley NLP. Byte-level and structure-aware NLP — his students are worth scanning for fresh names.
**Nick Haber** — Stanford GSE (courtesy CS). Zelikman's advisor on Quiet-STaR. Active in latent reasoning.

---

## TIER 3 — OUT OF BAY AREA BUT DIRECTLY RELEVANT (email only, not coffee)

Include these for written outreach; skip for in-person. Several work on problems Pebble directly touches.

**Artidoro Pagnoni** (UW Seattle / Meta FAIR) — BLT first author. X: @ArtidoroPagnoni.
**Chunting Zhou** (Meta FAIR Seattle) — BLT co-author.
**Shibo Hao** (UCSD PhD) — COCONUT first author. ber666.github.io.
**Zhiting Hu** (UCSD) — COCONUT senior author. Theoretical foundations of latent reasoning.
**Sean Welleck** (CMU) — Inference Scaling Laws (ICLR 2025), L1 length-controlled reasoning (COLM 2025). Very Pebble-adjacent.
**Aviral Kumar** (CMU) — Meta-RL framing of test-time compute (ICML 2025).
**Ahmadreza Jeddi** (U Toronto) — LoopFormer first author. Pebble's primary baseline.
**Lingpeng Kong** (HKU) — EvaByte. Not Bay Area but actively publishes; conferences are the meet point.
**Benjamin Minixhofer** (Cambridge / AI2 Seattle) — Bolmo (Feb 2026): subword → byte conversion at 1B/7B.
**Md Rifat Arefin / Irina Rish / Ravid Shwartz-Ziv** (Mila / Montreal) — Seq-VCR representation collapse prevention.

---

## VERIFICATION NOTES (read before emailing)

Several entries below are flagged ⚠ because one or more of (advisor, current role, Bay Area status) came from the research agents without full cross-verification. Before sending a personalized email, take 30 seconds to confirm:

- **Yijia Shao's advisor.** Agent said Manning. STORM is from Diyi Yang's lab. Worth checking her site before writing "working with Manning" in an email.
- **Albert Gu's location.** He's CMU faculty, SF via Cartesia. Framing the email as "when you're in SF" is safer than assuming Bay Area primary.
- **Michael Hanna's current role.** Fellowship end dates drift. Check his site.
- **Ziyang Wu, Brent Yi, Chun-Hsiao Yeh, Tianzhe Chu, Yifu Lu, Amir Zur, Alexa Tartaglini, Ryan Ehrlich, Lianmin Zheng.** All were surfaced by the structured-reps and inference-time agents but with thinner verification than the Tier-1 picks above. Treat them as leads to investigate, not ready-to-email contacts.
- **Emmanuel Amiesen.** Flagged by the continuous-reasoning agent as Anthropic; spelling and exact role unverified. Skip until confirmed.

---

## OUTREACH STRATEGY

1. **Lead with a specific result from their paper, not a generic compliment.** "Your MrT5 75% sequence-length reduction" beats "I loved your paper."
2. **State Pebble's specific finding in one sentence.** Not the whole pitch — just one concrete thing that connects.
3. **Ask for 20 minutes, not "a chat."** Specificity raises response rate.
4. **Propose a Bay Area coffee location or a 20-min video call, not an open-ended "let's connect."**
5. **Include the honest negative results.** "Matrix ops lose to vectors at matched FLOPs" disarms skepticism and signals rigor.
6. **Never ask for a job or compute in the first email.** You're asking for their opinion, that's it.
7. **Follow up once after 7-10 days, then drop it.** No third email.

### Email template (60-second version)

> Subject: quick question on [their paper] + matrix-rank thinking
>
> Hi [Name],
>
> Your [specific paper + specific result] stuck with me. I've been running a related line — [one-sentence Pebble finding, e.g., "matrix-token representations where rank is a measurable complexity axis; we see rank enrichment during iterative refinement whose direction is determined by the output head"].
>
> I'd love 20 minutes to show you a few plots and get your opinion on where I'm wrong. Honest negative results included — the matched-FLOPs ablation still favors vectors, so I'm trying to figure out whether the structure story holds at scale. Happy to grab coffee in [SF/Palo Alto/Berkeley] or jump on a call.
>
> — Sam Larson (Pebble AI, pebbleml.com)

---

## TIMELINE

- **This month:** Send the Top 5. One email each. Personalized. Wait 7 days before batch #2.
- **Batch 2 (after any Top-5 replies):** Tier-1 Anthropic + Berkeley picks.
- **Batch 3 (only after scale-up experiments run):** Senior faculty (Tier 2) — but only with a concrete result.
- **Conferences:** NeurIPS 2026, ICLR 2027, ICML 2026 — in-person is 10x the response rate of cold email.

---

## APPENDIX — ORIGINAL LIST (April 10, 2026)

The prior version of this list is preserved in git history for reference. It had useful names (Zelikman, Sukhbaatar, Tian, Pagnoni, Hu, Hao, Jaegle, Arefin, Rish, Shwartz-Ziv, Jeddi, Ciccone, Kong, Olah, Manning, Song) — most of which are folded above. Nothing on this page contradicts that list; it just reprioritizes around Bay Area junior researchers with 2025-2026 publications.
