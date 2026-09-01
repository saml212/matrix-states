# Collaborator Map — researchers whose work this program cites or touches

> **NO OUTREACH BEFORE BATCH 1 IS LIVE ON ARXIV (strategy-session rule).**
> This document is planning material only. Nothing here has been sent, and
> nothing should be sent, until `mstar-colm-er`, `neurreps-ea`, and the ICML
> rank-blindness arXiv post (TRIAGE batch 1) are actually live on arXiv.
> Every draft email below is a DRAFT for that future moment, not a queued
> message.

Compiled 2026-09-01. Sources read: `CLAUDE.md`, `papers/TRIAGE_2026-09-01.md`,
`papers/mstar-colm-er/brief.md`, `papers/neurreps-ea/brief.md`,
`papers/flagship/brief.md` + `papers/flagship/sections/06_related_work.md`,
`papers/*/refs.bib` (mstar-colm-er, neurreps-ea, rank-recruitment-ws, kwall),
`papers/flagship/references.bib`, `references.md`,
`research/novelty-gate-2026-07-27.md`, `EXPERIMENT_LOG.md` (2026-08-29
entries #1–#3, 2026-09-01 #2). All affiliations and arXiv IDs below were
checked live (web search / direct arXiv abstract-page fetch) on 2026-09-01
by research subagents; anything not independently confirmed is marked
**[UNVERIFIED]**. No paper or arXiv ID is invented — every ID here already
appears in this repo's own `refs.bib`/`references.bib` files, re-verified
live rather than trusted from memory.

**The asset, stated once:** our public Hugging Face dataset
`Slamin/ncr-scaling-artifacts` (confirmed PUBLIC as of 2026-09-01, ~3.28 TB,
~2,000 files) holds full checkpoint tiers for the DeltaNet-family models at
98M / 392M / 1.31B parameters across the K-way exact-composition sweep
(frozen hand-built-operator arm + SGD-trainable-write arm), the capability-
separation recall models (14M-class contender/ablation/transformer triple),
and the rank-law group-composition encoders. Every "what's in it for them"
note below points at a specific slice of this dataset and a concrete
experiment runnable against it without retraining from scratch.

**Co-authorship bar, stated once (applies to every entry below):** running a
new experiment against our public checkpoints (or contributing analysis code
we then run) that changes or sharpens a recorded verdict — closes a gap,
finds a failure mode we didn't have, or supplies a fix that gets tested and
reported — earns authorship on the paper it feeds. Citing the dataset,
confirming our numbers, or a substantive email exchange without new
experimental content earns an acknowledgment, not authorship. This bar is
Sam's to apply per-paper; it is stated once here rather than repeated in
every entry.

---

## 1. DeltaNet / Gated Linear Attention lineage

DeltaNet and Gated DeltaNet are the architecture substrate for the program's
delta-rule work: the capability-separation recall result (mstar C1–C11), the
write-geometry attractor (flagship R6–R8), and the three-scale native-
composition-reads result (`EXPERIMENT_LOG.md` 2026-08-29 #1–#3) are all run
on DeltaNet-family models. None of this group's own papers characterize
what the state represents or where SGD-learned writes stop working — that
gap is exactly what our results fill.

### Songlin Yang
- **Current affiliation:** **Thinking Machines Lab**, Member of Technical
  Staff (confirmed via her own site, sustcsonglin.github.io — she completed
  her MIT CSAIL PhD, advised by Yoon Kim). Both arXiv:2406.06484 and
  arXiv:2412.06464 list her at MIT; that affiliation is now stale.
- **Claim touched:** arXiv:2406.06484 ("Parallelizing Linear Transformers
  with the Delta Rule over Sequence Length") and arXiv:2412.06464 ("Gated
  Delta Networks") establish the delta-rule recurrence as an efficient,
  high-quality architecture; neither paper measures what the resulting
  fast-weight state represents or characterizes a capability/limitation
  boundary. Our results are a direct empirical answer to "what is a
  DeltaNet state actually holding, and where does it stop working":
  (a) EXTEND — the state causally carries an episodic-recall capability
  matched transformer/vector baselines lack, localized to layer 0 by
  causal zeroing (mstar C1, C2, C5, C6); (b) EXTEND — a hand-built,
  frozen exact-composition read (repeated squaring) executes flawlessly on
  this substrate at ceiling from 98M to 1.31B parameters (13.4×, both
  pairwise no-detectable-shift tests, `EXPERIMENT_LOG.md` 2026-08-29 #3),
  while (c) BOUND — the SGD-*learned* write hits a hard depth/breadth wall
  that scale does not fix (K=40 trainable degrades at both scale steps;
  K=32's 392M token-budget rescue does NOT repeat at 1.31B — the
  budget-rescue is itself scale-dependent, `EXPERIMENT_LOG.md` 2026-08-29
  #3 "THE MOAT"); and (d) BOUND — the same write mechanism drives a
  population-geometry pathology (span-fraction 0.248→0.455, 14M→1.31B)
  that qk-norm does not remove (flagship R6, R7).
- **What's in it for them:** the public checkpoint ladder
  (`Slamin/ncr-scaling-artifacts`) has DeltaNet-family models at all three
  scales with BOTH the frozen exact-operator arm and the SGD-trainable-write
  arm on the same K-way composition task — a ready-made testbed for any new
  DeltaNet variant (e.g. a future gating or normalization change) to check
  against the composition wall without re-running our sweep. Concrete
  experiment: graft a candidate write-side fix (e.g. a variant of Gated
  DeltaNet-2's gating) onto our trainable-arm checkpoints at K=32/K=40,
  1.31B, and see whether it moves the needle above 0.90 — the exact
  threshold our attribution arm (40k steps) failed to clear.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: A DeltaNet composition wall that scale doesn't fix
  >
  > Hi Songlin — I've been running DeltaNet-family models (98M–1.31B) on a
  > K-way exact-composition task with two write arms: a frozen hand-built
  > operator (executes perfect O(log h) composition reads at every scale)
  > and an SGD-trained write (hits a hard wall at K≥32 that a 13.4×
  > parameter increase doesn't move, and whose 392M token-budget rescue
  > stops working at 1.31B). Full checkpoints for both arms are public on
  > HF (`Slamin/ncr-scaling-artifacts`). Question: in your experience
  > building Gated DeltaNet / Gated DeltaNet-2, did you ever see write-side
  > learnability (not read-side expressivity) become the bottleneck at
  > scale, and is there a gating change you'd expect to help a
  > deep-composition write specifically rather than recall generally?

### Yoon Kim
- **Current affiliation:** MIT EECS, Associate Professor; PI at CSAIL and
  the MIT-IBM Watson AI Lab (eecs.mit.edu faculty page). Senior/last author
  on arXiv:2406.06484.
- **Claim touched:** same as Songlin Yang above (co-architect of the delta
  rule's efficient parallelization); the recall-capability and
  composition-wall results are read as "what does the mechanism he
  co-designed actually store, and where does its trainability stop."
  Verdict: EXTEND + BOUND (same evidence rows).
- **What's in it for them:** same dataset/experiment pointer as Yang. As a
  PI with a research group, a more likely path than a personal
  experimental contribution is a graduate student running the
  write-side-fix experiment above, or advising on whether the
  execute-vs-learn framing (frozen operator executes; SGD write doesn't
  learn) matches known FLA-library failure modes.
- **Draft email:** same technical question as above, addressed to Yoon
  as PI rather than re-drafted — sending both to the same thread once
  outreach opens, not two separate cold emails, is the intended use.

### Others in this lineage (brief, no separate outreach draft)
- **Bailin Wang** — now a researcher at Meta (multimodal/long-context),
  per his own site (berlino.github.io); was at MIT when arXiv:2406.06484
  was written.
- **Yu Zhang** — Soochow University, School of Computer Science and
  Technology (per the NeurIPS 2024 GSA paper author list and the FLA
  library association); also did related work during a Tencent AI Lab
  internship.
- **Yikang Shen** — MIT-IBM Watson AI Lab / IBM Research (Google Scholar,
  ResearchGate). Notably also adjacent to the M²RNN authorship cluster
  below (IBM/MIT-IBM matrix-state lineage) — worth noting as a bridge
  between the DeltaNet and state-space groups.
- **Jan Kautz** — NVIDIA, VP of Learning and Perception Research
  (research.nvidia.com/labs/lpr). Confirmed NVIDIA.
- **Ali Hatamizadeh** — NVIDIA, Staff Research Scientist / LLM Tech Lead
  (research.nvidia.com/person/ali-hatamizadeh). Confirmed NVIDIA; recent
  2026 work includes "Gated DeltaNet-2" (May 2026) and NVIDIA Nemotron 3
  Super (Mar 2026) — Gated DeltaNet-2 is the most current variant to test
  our composition-wall probe against.

---

## 2. Fast-weight programmers (classical lineage)

### Imanol Schlag
- **Current affiliation:** ETH Zurich, AI Research Scientist at the ETH AI
  Center; co-leads Apertus (the Swiss AI Initiative's LLM effort); also
  lecturer (isg.inf.ethz.ch, ETH CS department pages, both live as of
  2026). `references.md` already flags him as a "Tier-1 outreach target"
  for this project independent of this task.
- **Claim touched:** Schlag, Irie & Schmidhuber (arXiv:2102.11174, "Linear
  Transformers Are Secretly Fast Weight Programmers") originates the
  framing that a linear-attention/fast-weight state is an associative
  memory — a conceptual claim, not a measured or trained one. Our rank law
  (neurreps N1–N9: recruited effective rank tracks the task's minimal
  faithful dimension d_min, causally necessary AND sufficient) and the
  recall capability separation (mstar C1–C6) give that 2021 framing
  empirical and causal teeth: SGD really does recruit exactly the
  structure the FWP framing predicts a fast-weight memory should need, and
  the recruited structure is causally load-bearing (force-rank to d_min−1
  reads 0.000 recovery in all 5 groups). Verdict: **CONFIRM + EXTEND** —
  their conceptual claim is confirmed quantitatively and given a causal
  necessity/sufficiency proof they didn't attempt.
- **What's in it for them:** the group-composition encoder checkpoints
  (5 permutation groups, d_min 2→5) and the recall triple (contender/
  ablation/transformer at 14M) are both on the public dataset. Concrete
  experiment: Schlag's own 2021 FWP formalism gives an explicit update
  rule distinct from the delta rule — running his original FWP update (not
  DeltaNet's parallelized delta rule) through our rank-law instrument
  (restricted effective rank vs. d_min, force-rank causal ablation) would
  test whether the rank law is delta-rule-specific or a property of
  fast-weight writes generally.
- **Co-author bar:** running that FWP-update rank-law comparison (or a
  variant) and reporting a result — even a null one — earns authorship on
  whichever paper it lands in (`neurreps-ea` if pre-submission, an
  arXiv v2 if after).
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: Testing the causal rank law against the original FWP update
  >
  > Hi Imanol — your 2021 paper with Kazuki and Jürgen framed linear
  > attention as a fast-weight associative memory; I've been running that
  > framing as a literal, testable claim: on group-composition tasks, a
  > trained matrix state's effective rank tracks the group's minimal
  > faithful representation dimension d_min, and force-ranking to d_min−1
  > causally collapses exact recovery to zero in all five permutation
  > groups we tested (checkpoints public on HF). My question: does this
  > rank-vs-d_min law depend on the DeltaNet parallelization specifically,
  > or would you expect it to hold under your original FWP outer-product
  > update rule too? I don't have a principled argument either way.

### Kazuki Irie
- **Current affiliation:** Yale University, tenure-track Assistant
  Professor of Computer Science, and Investigator at the Wu Tsai Institute
  — as of July 2026 (Yale CS department page). Previously a postdoc at
  Harvard (Department of Psychology). Both arXiv:2102.11174 and the FWP
  memory paper (arXiv:2011.07831, cited in `papers/kwall/refs.bib`) list
  older affiliations (IDSIA/Harvard-era).
- **Claim touched:** same conceptual lineage as Schlag above; Irie is also
  a co-author on the Yau et al. "Sequential-Parallel Duality in Prefix
  Scannable Models" (arXiv:2506.10918) that the program's own novelty gate
  flagged as a near-scoop on the K-wall's collapse-vs-hold construction
  (`papers/kwall/refs.bib` yau2026sequential entry: "T-PSM holds to test
  length 180 where transformer and Mamba both collapse; empirical/learned,
  no exactness argument"). Verdict: **BOUND** — our frozen-operator arm's
  exact composition at every tested scale is the case Yau et al.'s
  empirical/learned T-PSM result does not cover; our SGD-trainable arm's
  wall is closer in spirit to what they observe.
- **What's in it for them:** the K-wall dataset (frozen vs. trainable arms
  at K=12..48, 98M–1.31B) is a direct extension target for the T-PSM
  collapse-vs-hold framing — a concrete experiment is running T-PSM's own
  probe methodology against our public checkpoints to see whether its
  "holds to length 180" finding extends to our much larger scale range and
  to the specific point where our trainable arm walls.
- **Co-author bar / email:** same standing offer as Schlag's entry above;
  a separate email is optional rather than required — Sam's call whether
  to reach Irie directly (re: Yau et al.) or fold into the Schlag thread.

### Jürgen Schmidhuber
- **Current affiliation:** KAUST (King Abdullah University of Science and
  Technology), Professor of Computer Science and Director/co-chair of the
  Center of Excellence for Generative AI (kaust.edu.sa, cemse.kaust.edu.sa
  faculty pages); retains a historical IDSIA affiliation.
- **Claim touched:** originator of the fast-weight-memory idea itself
  (Schmidhuber 1992, "Learning to Control Fast-Weight Memories," Neural
  Computation 4(1):131-139, DOI 10.1162/neco.1992.4.1.131 — confirmed via
  MIT Press Direct / ACM Digital Library) and co-author of the 2021 FWP
  paper above. Verdict: **CONFIRM + EXTEND** (same evidence as Schlag).
- **What's in it for them:** same dataset pointer; given his role as
  originator rather than an active experimentalist on this specific
  question, the most realistic ask is a short technical exchange rather
  than a bench experiment — the co-author bar above still applies if that
  changes.
- **Draft email:** not separately drafted — Schlag's email above already
  addresses "the FWP lineage"; cc'ing Schmidhuber (and Irie) on that same
  thread once outreach opens is the intended path rather than three
  near-identical cold emails.

---

## 3. State-space model lineage

### Albert Gu & Tri Dao (Mamba, background)
- **Current affiliations:** Albert Gu — Carnegie Mellon University,
  Assistant Professor, Machine Learning Department (csd.cmu.edu). Tri Dao
  — Princeton University, Assistant Professor, AND Chief Scientist at
  Together AI (both confirmed, Princeton CS directory + Together AI's own
  materials). arXiv:2312.00752 ("Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces") confirmed as their paper.
- **Claim touched:** Mamba is cited across this program (`references.md`,
  mstar refs.bib) as the SSM-lineage background/baseline citation, not as
  an architecture our results directly test. Verdict: **adjacent only** —
  no evidence row in this program runs a Mamba baseline; the touch is
  citation-level (fixed-size-state sequence modeling as the category our
  fast-weight results sit inside), not a claim we confirm/extend/bound.
- **What's in it for them / email:** no dedicated ask for the Mamba paper
  itself; see Tri Dao's dedicated entry below for the sharper touch point
  (M²RNN), which is where a real technical question exists.

### Tri Dao — M²RNN (arXiv:2603.14360)
- **Current affiliation:** as above (Princeton + Together AI). Confirmed
  co-author, with Mayank Mishra, Shawn Tan, Ion Stoica, Joseph Gonzalez, on
  arXiv:2603.14360 ("M²RNN: Non-Linear RNNs with Matrix-Valued States for
  Scalable Language Modeling," submitted March 2026) — fetched and
  confirmed directly against the arXiv abstract page.
- **Claim touched:** M²RNN engineers a matrix-valued state via a fixed
  nonlinear recurrence (their own vector-GRU baseline matches it on their
  reported task, per `papers/neurreps-ea/brief.md`'s related-work note) —
  it is a concurrent, competing answer to "should the state be a matrix,"
  with no rank measurement and no causal rank intervention. Our rank law
  (neurreps N1–N9) makes rank itself the measured and manipulated quantity
  across five groups spanning the solvable/non-solvable divide, and our
  capability separation (mstar) plus the composition-wall result
  (`EXPERIMENT_LOG.md` 2026-08-29 #3) supply exactly the causal mechanism
  and the failure boundary M²RNN's engineering-first paper doesn't
  establish. Verdict: **EXTEND + BOUND** — we don't contradict M²RNN's
  architecture claim, but we supply the "why/when a matrix state helps"
  and "where it stops working" that their paper is silent on (their own
  reported vector-GRU parity result is itself a data point consistent
  with our finding that rank must be causally forced to matter, not
  merely made available).
- **What's in it for them:** the group-composition rank-law checkpoints
  are a direct, ready-to-run test of whether M²RNN's matrix-valued
  mechanism recruits rank the same way our matrix-state encoder does, or
  recruits it differently (their fixed nonlinear recurrence vs. our
  in-context-written operators is exactly the axis to check). Concrete
  experiment: run their restricted-effective-rank instrument (or ours,
  code is in the public checkpoints' companion scripts) on M²RNN states
  trained on our S3/S4/A5/S5/A6 permutation-group tasks, and see if
  rank tracks d_min the same way.
- **Co-author bar:** that specific cross-architecture rank-law replication
  (M²RNN vs. our matrix-state encoder, same tasks) is exactly the kind of
  result that would earn authorship on a follow-up paper — it's a real
  open question neither paper currently answers.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: Does M²RNN's matrix state recruit rank the way ours does?
  >
  > Hi Tri — I saw M²RNN's matrix-valued states and noticed your own
  > vector-GRU baseline matches it on your reported task; we found
  > something that might explain that. On group-composition tasks, a
  > trained matrix state's effective rank tracks the group's minimal
  > faithful dimension d_min ONLY when we force it causally (force-rank
  > below d_min collapses recovery to zero in every group we tested;
  > merely having a matrix-shaped state available doesn't guarantee the
  > rank gets used). Do you have a read on whether M²RNN's fixed nonlinear
  > recurrence recruits rank the same way, or whether that's exactly why
  > your vector-GRU control catches up? Checkpoints + rank instrument are
  > public on HF if useful for a direct comparison.

### Other M²RNN co-authors (brief, no separate outreach draft)
- **Mayank Mishra** — affiliation unsettled across sources: previously led
  pretraining/architecture research at MIT-IBM Watson AI Lab for IBM
  Granite; one source suggests a move to a UC Berkeley PhD, another a
  "stealth AI startup" — **[UNVERIFIED, conflicting]**, needs a direct
  check before use.
- **Shawn Tan** — sources conflict between MIT-IBM Watson AI Lab / Google
  DeepMind (current) and a January 2027 tenure-track start at NUS
  (National University of Singapore) CS — **[UNVERIFIED which is
  "current"]** as of Sept 2026.
- **Ion Stoica** — UC Berkeley, Professor (Xu Bao Chancellor Chair), EECS;
  Director, Sky Computing Lab; also Executive Chairman, Databricks and
  Anyscale (confirmed via his own post announcing the M²RNN paper).
- **Joseph Gonzalez** — UC Berkeley, Professor, EECS; co-director, Sky
  Computing Lab / BAIR (confirmed).

---

## 4. Test-time training (TTT)

### Yu Sun & Tatsunori Hashimoto
- **Current affiliations (live-verified):** Yu Sun — postdoc, Stanford
  STAIR lab, and also a researcher at NVIDIA (yueatsprograms.github.io;
  stairlab.stanford.edu/members/yu_sun.html); his PhD was at UC Berkeley,
  not Stanford. Tatsunori Hashimoto — still Assistant Professor, Stanford
  CS (thashim.github.io; cs.stanford.edu/people/tatsunori-hashimoto), no
  sign of a move. arXiv:2407.04620 ("Learning to (Learn at Test Time):
  RNNs with Expressive Hidden States," full author list Yu Sun, Xinhao
  Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois,
  Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos
  Guestrin) is confirmed and is now v4, accepted as an **ICML 2025
  spotlight poster** — not just an arXiv preprint.
- **Claim touched:** TTT frames a weight matrix as the hidden state,
  updated online by a self-supervised test-time gradient step — the same
  "matrix as memory, updated by a learning rule" object DeltaNet's delta
  rule instantiates (the delta rule is a known one-step online-gradient
  special case of the TTT idea). This program does not currently cite
  arXiv:2407.04620 in any paper's `refs.bib` — it appears only in
  `references.md`'s literature library — so this is a conceptual touch
  the paper program has not yet formalized as a citation. Our capability
  separation (mstar) and rank law (neurreps) results, run on the
  DeltaNet substrate, bear directly on TTT's core claim that a
  weight-matrix state updated by a learning rule can serve as memory:
  we give a causal necessity/sufficiency proof of what such a state must
  hold (rank ≥ K, exactly d_min for group tasks) and a demonstrated
  capability gap over parameter-matched fixed-state baselines. The
  composition-wall result (`EXPERIMENT_LOG.md` 2026-08-29 #3) is a BOUND
  on TTT-style state updates specifically: even where capacity is
  sufficient (the frozen/hand-set operator proves the state CAN hold
  exact deep composition), the SGD-learned analogue of a TTT-style
  online update does not learn to reach that capability at K≥32,
  and scale does not fix it. Verdict: **EXTEND** (rank law, capability
  separation) **+ BOUND** (the composition wall as a limit on
  what self-supervised test-time-style updates learn, independent of
  whether the state has room for the answer).
- **What's in it for them:** the checkpoint ladder gives a direct
  testbed for whether TTT's own online-update rule (rather than the
  parallelized delta rule) hits the same composition wall, or whether
  their specific self-supervised objective avoids it. Concrete
  experiment: swap the trainable arm's delta-rule write for a TTT-style
  online-gradient write on the same K-way composition task and re-run
  our exact instrument suite (P0/P1b thresholds, pinned before any
  number) at 98M/392M/1.31B.
- **Co-author bar:** that swap-and-rerun result — does TTT's own update
  rule hit the same wall — is squarely a co-authorship-grade contribution
  if it changes the recorded verdict either way.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: Does a TTT-style write hit the same composition wall as ours?
  >
  > Hi Yu (and Tatsunori) — I've been studying a specific failure mode of
  > weight-matrix-as-state models: on a K-way exact-composition task, a
  > hand-built frozen operator executes perfect O(log h) composition reads
  > at every scale we tested (98M–1.31B params), but an SGD-trained write
  > on the identical architecture hits a hard wall at K≥32 that scale
  > doesn't fix — even a 40k-step attribution run tops out at 0.80–0.89.
  > I don't yet know if this is a delta-rule-specific pathology or a
  > general property of gradient-updated matrix memories, which is
  > exactly what TTT's framing would predict either way. Would your
  > online test-time update rule be easy to drop into the same probe?
  > Checkpoints (both arms, all scales) are public on HF if that helps.

### Other TTT co-authors (brief, from paper footnotes only, not independently re-verified)
Stanford / UCSD / Meta AI mix: Xinhao Li, Karan Dalal, Arjun Vikram, Genghan
Zhang, Yann Dubois (Stanford); Jiarui Xu, Xiaolong Wang (UCSD); Xinlei Chen
(Meta FAIR); Sanmi Koyejo, Carlos Guestrin (Stanford faculty).

---

## 5. Continuous chain-of-thought / CODI / Coconut

This is the one group where our program has ALREADY published directly on
the same substrate: the ICML 2026 MI-workshop paper ("The Gradient Does Not
See Rank," `larson2026gradient`/`larson2026the`, cs.LG, no arXiv record yet
— TRIAGE cluster C5, currently blocked on arXiv endorsement) trains
matrix-CODI on ProsQA, the exact benchmark Coconut introduced.

### Shibo Hao & Yuandong Tian (Coconut)
- **Current affiliations (live-verified):** Shibo Hao — PhD candidate,
  UC San Diego Halıcıoğlu Data Science Institute (2024 Bloomberg PhD
  Fellow); still enrolled. Yuandong Tian — **no longer at Meta FAIR**;
  co-founder of Recursive Superintelligence, an AI startup reported at a
  roughly $4.65B valuation (per SCMP), confirming the broader 2025-26
  wave of Meta FAIR departures. arXiv:2412.06769 (Coconut) is confirmed
  correct and was accepted at **COLM 2025**.
- **Claim touched:** Coconut (arXiv:2412.06769, "Training Large Language
  Models to Reason in a Continuous Latent Space") proposes continuous
  latent thoughts as a compressed chain-of-thought medium, introducing
  ProsQA as the benchmark. Our ICML paper bolts a matrix-valued state onto
  a pretrained GPT-2 backbone via a flatten-then-project readout and
  shows the training gradient is rank-indifferent (four positive-control
  nonlinear-in-Z readouts fail to bend the rank-k curve) — a NEGATIVE,
  bounding result on this specific bolt-on construction, numerically
  consistent with Rizvi-Martel et al.'s independent finding that latent
  feedback moves COCONUT's ProsQA accuracy less than seed noise (96.6%
  without latent feedback vs. 99.0% with it; our own SFT baseline reaches
  81.77%, matrix-CODI 82.03%, both well below and differing by less than
  three-seed noise — `matrix-thinking/submissions/icml-mi-workshop-2026/
  sections/06_related_work.tex`). The flagship's from-scratch, native-state
  result (rank law + capability separation) is the positive counterpart:
  when the state is native (not bolted on) and the readout preserves
  structure, SGD DOES see rank. Verdict: **BOUND** on bolt-on
  matrix-valued extensions of continuous-CoT specifically (a mechanism
  for why such extensions may fail to improve on the vector-latent
  baseline) **+ CONFIRM** of the independent Rizvi-Martel finding that
  latent feedback's ProsQA gain is within seed noise at GPT-2 scale.
- **What's in it for them:** the from-scratch matrix-state checkpoints
  (rank law + capability separation) are a concrete existence proof that
  matrix-valued latent state CAN support a real capability under the
  right training regime (native, structure-preserving readout) — directly
  relevant to whether a future Coconut variant should go matrix-valued,
  and how to avoid the rank-blind gradient failure mode our ICML paper
  diagnoses.
- **Co-author bar:** designing and testing a Coconut variant with a
  native (not bolted-on) matrix-valued latent, using our rank-law
  instrument to check whether the gradient sees rank in that setting,
  would be squarely co-authorship-grade on a follow-up.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: A rank-blindness failure mode on ProsQA, downstream of Coconut
  >
  > Hi Shibo (and Yuandong, if this reaches you) — I've been running a
  > matrix-valued extension of continuous-CoT reasoning on ProsQA, the
  > benchmark from your Coconut paper. Bolting a matrix-shaped latent onto
  > a pretrained GPT-2 via a flatten-then-project readout, I find the
  > training gradient is rank-indifferent — four different nonlinear
  > readouts fail to make accuracy depend on the latent's truncation rank,
  > and the resulting accuracy (82.03%) sits within three-seed noise of
  > plain SFT (81.77%), consistent with an independent replication
  > (Rizvi-Martel et al.) finding latent feedback's own ProsQA gain is
  > within seed noise at this scale. Separately, in from-scratch
  > matrix-state models (not bolted onto a pretrained backbone) we DO see
  > SGD recruit rank causally. Curious whether that native-vs-bolt-on
  > distinction matches your own intuition for why latent CoT signal is
  > hard to move past noise at small scale.

### Zhenyi Shen, Hanqi Yan, Yulan He (CODI)
- **Current affiliations (live-verified):** Zhenyi Shen — PhD student,
  King's College London (Department of Informatics), advised by Yulan He
  and Yali Du. Hanqi Yan — Lecturer / Assistant Professor, King's College
  London (confirmed via her own Scholar profile listing both this paper
  and the token-uniformity paper below — same person). Yulan He —
  Professor of Natural Language Processing, King's College London; UKRI
  Turing AI Fellow (confirmed as senior/last author on both papers,
  unambiguously KCL). arXiv:2502.21074 (CODI) is confirmed correct and
  was accepted at **EMNLP 2025** (verified via ACL Anthology); a reported
  RAI-UK best-paper award for CODI is **[UNVERIFIED against a primary
  source]**.
- **Claim touched:** CODI (Shen, Yan, Zhang, Hu, Du, He; EMNLP 2025,
  arXiv:2502.21074, "CODI: Compressing Chain-of-Thought into Continuous
  Space via Self-Distillation") is the literal substrate of our ICML
  paper — matrix-CODI is CODI with a matrix-valued latent bolted onto its
  self-distillation pipeline. Our result is a direct diagnosis of a
  training-objective failure mode (rank-indifferent gradients under
  flatten-then-project) that would affect any matrix-valued extension of
  CODI's own distillation approach. Verdict: **BOUND** — a specific,
  falsifiable limitation on extending CODI to matrix-valued latents via
  the naive bolt-on route, discovered by literally trying it on their
  method.
- **What's in it for them:** same as above — our public matrix-state
  checkpoints show the fix (native, structure-preserving) that a
  matrix-valued CODI variant would need to avoid the failure we found.
  Concrete experiment: apply CODI's self-distillation objective natively
  to our matrix-state encoder (rather than bolting matrix state onto a
  frozen backbone) and check with our rank-truncation instrument whether
  the distillation gradient sees rank.
- **Co-author bar:** that native-CODI-with-matrix-state experiment,
  reported either way, is co-authorship-grade.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: A rank-blindness failure mode when CODI's latent goes matrix-valued
  >
  > Hi Hanqi (and Zhenyi, Yulan) — I ran a matrix-valued extension of CODI
  > on ProsQA (bolting a matrix latent onto a pretrained GPT-2 backbone via
  > a flatten-then-project readout) and found the training gradient is
  > rank-indifferent: four different nonlinear-in-Z readouts all fail to
  > make accuracy depend on the truncation rank, and the resulting
  > accuracy (82.03%) is within three-seed noise of plain SFT (81.77%) —
  > consistent with Rizvi-Martel et al.'s finding that latent feedback's
  > own gain is within noise at this scale. Separately, in from-scratch
  > matrix-state models (not bolted onto a pretrained backbone) we do see
  > SGD recruit rank causally. Does that match your intuition for why
  > CODI's distillation signal might not "see" a matrix-shaped latent
  > the way it sees a vector one, or do you think the flatten-then-project
  > readout is the whole story?

### Other CODI / token-uniformity co-authors (brief, live-verified where noted)
- **Linhai Zhang, Zhanghao Hu, Yali Du** — all King's College London,
  Department of Informatics (per CODI's own footnotes).
- **Lin Gui** — also King's College London-affiliated (a later
  co-authorship with the same group confirms this).
- **Wenjie Li** — 2022-era affiliation was Hong Kong Polytechnic
  University; **[UNVERIFIED as current]**.

---

## 6. Argmax-vs-exact associative memory (Nichani, Lee, Bietti)

### Eshaan Nichani, Jason D. Lee, Alberto Bietti
- **Current affiliations (live-verified):** Eshaan Nichani — PhD candidate,
  Princeton ECE (advised by Jason D. Lee and Yuxin Chen), graduated
  roughly May 2026 per his own site. Jason D. Lee — **moved from
  Princeton to UC Berkeley**: now Associate Professor, EECS & Statistics,
  UC Berkeley, and Senior Scientist at the Simons Institute (confirmed
  from his own homepage, generated Jul 2026 — some stale search results
  still say Princeton; disregard those). Alberto Bietti — Research
  Scientist, Center for Computational Mathematics, Flatiron Institute
  (Simons Foundation) / Polymathic AI, joined Aug 2023, no change.
- **Claim touched:** Nichani, Lee & Bietti (ICLR 2025 Spotlight,
  arXiv:2412.06538, "Understanding Factual Recall in Transformers via
  Associative Memories") prove that under argmax decoding a rank-1 matrix
  can recover ≈d associations — a hand-built existence construction, not
  a measured or trained rank, and not a necessity bound for exact
  continuous recovery. This is the single most load-bearing piece of
  outside work in the whole program: every rank/recall number in
  `neurreps-ea` and `mstar-colm-er` carries their caveat explicitly
  (mstar C1 "Nichani caveat on every acc_A number"; neurreps' entire
  causal-razor design exists specifically to close their argmax loophole
  by construction — force-rank at k=d_min−1 reads exactly 0.000 recovery
  under EXACT continuous recovery, in all 5 permutation groups and all 4
  independent S3 seeds, neurreps N1/N2). Verdict: **EXTEND** — we don't
  contradict their argmax-regime capacity result, we supply the
  complementary causal necessity-and-sufficiency result for the
  exact-recovery regime their own analysis flags as the harder case, in
  a TRAINED (not hand-constructed) system.
- **What's in it for them:** the group-composition rank-law checkpoints
  are a direct test of their argmax-capacity theory's boundary: our
  causal razor shows that under exact recovery, a rank-(d_min−1) state
  provably cannot recover, while their argmax analysis says a much
  lower-rank state CAN recover under argmax decoding on the SAME kind of
  associative task. Concrete experiment: run their argmax-decoding
  readout against our own trained states (same checkpoints, same
  permutation-group tasks) to get an empirical crossover curve between
  their capacity regime and ours as decoding moves from exact to argmax —
  something neither paper currently has.
- **Co-author bar:** that crossover-curve experiment (empirical argmax-
  vs-exact capacity on the same trained states) is a natural and
  co-authorship-grade joint result.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: An empirical exact-recovery floor to go with your argmax ceiling
  >
  > Hi Eshaan (and Jason, Alberto) — your rank-1-recovers-≈d-associations
  > result under argmax decoding is the reason our own rank-law work pins
  > its readout to exact continuous recovery everywhere a rank claim
  > depends on it. On group-composition tasks, we find a trained matrix
  > state's effective rank tracks the group's minimal faithful dimension
  > d_min exactly, and force-ranking to d_min−1 drives exact recovery to
  > 0.000 in all five permutation groups we tested (checkpoints public on
  > HF). I don't have your argmax-decoding readout implemented against our
  > states — would an empirical crossover curve (same trained states,
  > sweeping from exact to argmax decoding) be a natural quick check
  > against your theory, or is there a reason the two regimes wouldn't
  > meet cleanly on this kind of task?

### Nicholas Barnfield, Juno Kim, Yue M. Lu (the follow-up capacity theorem)
- **Current affiliations (live-verified):** Nicholas Barnfield — PhD
  student, Statistics, Harvard University. Juno Kim — UC Berkeley EECS
  PhD student, advised by Jason D. Lee and Song Mei (prior MS, University
  of Tokyo). Yue M. Lu — Gordon McKay Professor of Electrical Engineering
  and Applied Mathematics, Harvard SEAS, and Harvard College Professor
  since 2024.
- **ID/title flag — fix before citing:** arXiv:2605.05189's author list
  (Barnfield, Kim, Nichani, Lee, Lu) is confirmed exact, but its **real
  title is "Sharp Capacity Thresholds in Linear Associative Memory: From
  Top-1 Retrieval to Tail-Average Learning,"** not "From Winner-Take-All
  to Listwise Retrieval" as this program's own `refs.bib` files record it
  (`papers/rank-recruitment-ws/refs.bib`, `barnfield2026sharp`). Submitted
  May 6 2026, v2 revised Aug 19 2026. This is the same paper (exact
  author match, same core clause, same topic/date) — the subtitle in our
  bib is wrong and should be corrected before this ID is cited anywhere
  else; flag for Will's citation audit (`papers/TRIAGE_2026-09-01.md`
  §4 item 5, "two fabricated entries were caught before; assume more
  exist" — this is a real instance, a wrong subtitle rather than a
  fabrication, but the same audit should have caught it).
- **Claim touched:** the paper formalizes the argmax-vs-exact distinction
  as a scaling theorem (d²≍n log n under argmax vs. d²≍n under exact
  retrieval) — this is the paper the program's own novelty gate calls
  "the strongest dilution found" for the rank-law claim
  (`research/novelty-gate-2026-07-27.md` M6: "PARTIAL-OVERLAP (med-high)").
  Our causal razor is a task-CONSTRUCTION rule for rank lower bounds in a
  TRAINED system; theirs is a capacity theorem for a hand-analyzed model
  class. Verdict: **CONFIRM** — our empirical d_min necessity-and-
  sufficiency result is an independent, trained-system confirmation of
  the "exact retrieval needs the naive d²≍n scaling" side of their own
  theorem, arrived at from a completely different (causal-intervention,
  not capacity-theoretic) route.
- **What's in it for them:** a genuine independent empirical confirmation
  of one arm of their theorem, from a different methodology, is itself
  valuable to cite in their own follow-up work; the checkpoint dataset
  additionally lets them check whether their sharp-threshold predictions
  hold quantitatively (not just directionally) against our five trained
  permutation-group encoders.
- **Co-author bar / email:** Lee overlaps both author lists — treat as one
  outreach thread with the Nichani/Bietti email above (mention the
  Barnfield et al. theorem in the same message) rather than a second,
  near-duplicate email to Barnfield/Kim/Lu.

---

## 7. Rank / representation geometry of linear-attention states

### Philipp Nazari & T. Konstantin Rusch
- **Current affiliations (live-verified):** Philipp Nazari — PhD student,
  Department of Computer Science, ETH Zürich, also affiliated with the
  Max Planck Institute for Intelligent Systems (Tübingen) / ELLIS
  Institute Tübingen (confirmed directly from the paper's own HTML at
  arxiv.org/html/2602.04852). T. Konstantin Rusch — the paper itself
  (Feb 2026) lists MPI-IS Tübingen / ELLIS Institute Tübingen / Tübingen
  AI Center / Liquid AI; separately, multiple 2025-era sources report he
  is joining KAIST AI (Seoul) as an Associate Professor from Feb 2026,
  running a "CAMAIL" lab — these two data points are in tension for a
  Sept-2026 "current" affiliation and could not be reconciled against a
  primary KAIST source. **[UNVERIFIED — conflicting]:** treat "MPI-IS/
  ELLIS Tübingen (+ Liquid AI)" as the paper-of-record affiliation and
  KAIST as an unconfirmed pending/parallel appointment.
- **Claim touched:** Nazari & Rusch ("The Key to State Reduction in
  Linear Attention: A Rank-based Perspective," arXiv:2602.04852) measure
  effective rank of the linear-attention state matrix `S_t = Σ v_t k_t^T`
  descriptively on pretrained LLMs over real text, proving the upper
  bound rank(S_t) ≤ t with uncontrolled t, and propose post-training rank
  pruning. Our rank law (neurreps N1–N9) is the mirror image: controlled
  K/d_min, a LOWER bound (rank(Z) ≥ K for exact recovery, and exactly
  d_min for group tasks), and a causal force-rank training-time
  intervention — neither paper has any of the three (per
  `references.md`'s own "Added 2026-07-01" note and the neurreps brief's
  related-work section). Verdict: **EXTEND** — complementary axis
  (descriptive upper bound vs. our causal lower-bound-and-sufficiency
  result); no contradiction, since neither paper's regime (uncontrolled
  real-text t vs. our controlled synthetic K) overlaps enough to conflict.
- **What's in it for them:** our public checkpoint ladder gives them a
  controlled setting to check whether their descriptive rank-pruning
  recipe, applied at each scale (98M/392M/1.31B), respects the causal
  floor our razor establishes (pruning below d_min should break exact
  recovery, by our result) — a way to validate their pruning method
  against a ground-truth necessity bound they don't currently have access
  to in real-LLM settings.
- **Co-author bar:** running their pruning method against our
  ground-truth-necessity checkpoints and reporting where it breaks (or
  doesn't) is co-authorship-grade.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: A ground-truth necessity floor for your rank-pruning recipe
  >
  > Hi Philipp (and Konstantin) — your rank-based pruning recipe measures
  > effective rank descriptively on pretrained checkpoints; we have a
  > complementary result that might be useful for validating it. On
  > group-composition tasks, we causally force a trained matrix state's
  > rank via a training-time projection and find recovery is exactly
  > 0.000 below the group's minimal faithful dimension d_min and returns
  > sharply at d_min, across five permutation groups (checkpoints public
  > on HF). Would your pruning method, applied to these states, respect
  > that floor — i.e., does it ever prune below what we can show is
  > causally necessary? I don't have your method implemented to check
  > this myself.

### Ao Sun et al. ("State Rank Dynamics in Linear Attention LLMs")
- **Current affiliation (lead author, live-verified):** Ao Sun — primary
  affiliation The Chinese University of Hong Kong, Shenzhen (as a
  student); the paper states the work was "done during internship at
  Meituan" (industry), per the full text at arxiv.org/html/2602.02195,
  with correspondence emails at @meituan.com. The 12-author paper
  originates from a Meituan-centric industry/academia collaboration
  spanning CAS, USTC, Harbin Institute of Technology, Oxford, and
  Tsinghua.
- **Claim touched:** Sun et al. report "state-rank stratification" during
  pretraining — linear-attention heads bifurcate into persistently
  low-rank and high-rank groups — again a descriptive, uncontrolled-t
  observation on pretrained LLMs. Same relationship to our rank law as
  Nazari & Rusch above: complementary axis, no overlap sharp enough to
  confirm or contradict directly. Verdict: **EXTEND**.
- **What's in it for them:** our composition-wall result
  (`EXPERIMENT_LOG.md` 2026-08-29 #1: "WALL-BREACHED-AT-K=16 AT 1.31B...
  the model begins to LEARN a faint one-hop write at the smallest
  breadth... while the deep-composition wall and the exact-write
  capability stand untouched") is itself a scale-dependent rank/capability
  stratification finding, in a controlled setting — a natural point of
  contact with their own stratification-during-pretraining observation.
  Concrete experiment: apply their head-bifurcation instrument to our
  1.31B checkpoints at the K=16 wall-breach point and see whether the
  breach corresponds to a stratification event in their sense.
- **Co-author bar:** applying their stratification instrument to the
  K=16 wall-breach checkpoints and reporting whether it detects the
  breach is co-authorship-grade.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: Does state-rank stratification show up at our K=16 wall-breach?
  >
  > Hi Ao — your paper's state-rank stratification finding (heads
  > bifurcating into low-rank/high-rank groups during pretraining) is
  > close in spirit to something we found in a controlled setting: at
  > 1.31B parameters, on a K-way exact-composition task, the model begins
  > to learn a faint one-hop write at the smallest breadth tested (K=16)
  > that didn't exist at smaller scale, while a deeper composition wall
  > stays untouched at every scale up to 1.31B. Checkpoints at that exact
  > point are public on HF. Would your head-bifurcation instrument be
  > easy to run against them — I'm curious whether our wall-breach
  > corresponds to a stratification event in your sense, or is a
  > different phenomenon entirely.

---

## 8. Newton–Schulz / orthogonalized fast-weight memory

### arXiv:2607.19390 — live-verified, and it is NOT what the internal memos call it
- **Fetched directly from arxiv.org/abs/2607.19390.** The internal program
  description (`research/novelty-gate-2026-07-27.md` M4: "READ-time
  Newton-Schulz orthogonalization of mLSTM memory, argued to be a
  removable training scaffold") matches the paper's actual abstract, but
  the **title and author-list on record in this repo do not exist
  anywhere in `refs.bib`** — this ID has never actually been cited, only
  described in prose (`papers/TRIAGE_2026-09-01.md` confirms: "novelty-gate
  obligation to cite arXiv:2607.19390 unmet"). The real record:
  - **Real title:** "The Orthogonalized Read Is a Removable Training
    Scaffold for Recurrent Memory."
  - **Real author:** Keston Aquino-Michaels (single author).
  - **Stated affiliation (from the paper's own HTML):** "No Way Labs."
- **Affiliation/identity flags — read before doing anything with this
  citation:**
  - "No Way Labs" has no findable website, no other papers under that
    affiliation, and does not appear in standard AI-lab trackers —
    **[UNVERIFIED as an established institution]**, not necessarily
    illegitimate (independent/solo researchers do publish), but not a
    checkable affiliation either.
  - A "Keston Aquino-Michaels" with a documented 2016 PhD (advisor Nancy
    Cox) is a University of Chicago cancer-genomics/genetics researcher
    (PubMed/ResearchGate/AACR record), last publishing in genetics around
    2018 — a 17-year gap before an ML-arXiv persona under the same name
    starts (Feb 2026, "Routing Absorption in Sparse Attention," then this
    July 2026 paper — only 2 ML papers total). **It could not be
    confirmed whether this is the same individual, a relative, or an
    unrelated name match** — reported here as a finding, not an
    accusation, and not something to repeat as fact in outreach.
  - Consequently this author also falls short of the ≥3-recent-cs.LG-
    papers endorser bar (2 papers in a ~7-month window).
- **Claim touched:** their abstract argues read-time Newton-Schulz
  orthogonalization of mLSTM memory is a removable training scaffold, not
  a real memory improvement. Our composition-wall result is a different
  axis: a write-side SGD-trainability wall, not a read-time
  renormalization question — a hand-built EXACT operator (perfectly
  structured by construction) coexists with an SGD-learned write that
  cannot reach the same composition depth at any tested scale up to
  1.31B (K=40 trainable degrades at every scale step; K=32's 392M
  token-budget rescue does not repeat at 1.31B, `EXPERIMENT_LOG.md`
  2026-08-29 #3). That suggests orthogonality per se is not the
  bottleneck their "removable scaffold" framing might imply it should be.
  Verdict: **BOUND** — a candidate scope limit on their claim (their
  result concerns read-time renormalization of an already-written
  memory; ours concerns whether a deep-composition write can be learned
  at all), not a contradiction of it.
- **What's in it for them:** our public 1.31B checkpoints (both the
  frozen exact-write arm and the SGD-trainable arm, at every K) are a
  direct testbed for their scaffold-removal claim on a different failure
  mode: does removing (or adding) Newton-Schulz orthogonalization on our
  trainable arm's write path move the K=32/K=40 composition wall at
  1.31B, where our own 40k-step attribution run could not clear it?
- **Co-author bar:** running that specific ablation (their orthogonalization
  scaffold applied to our trainable write path) and reporting whether it
  moves the wall is co-authorship-grade, same bar as every other entry.
- **Recommendation before outreach:** given the affiliation is
  unverifiable and the author-identity question above is genuinely
  unresolved, Sam should independently confirm there is a real,
  reachable person behind this arXiv account (e.g., a homepage, GitHub,
  or LinkedIn under this exact name in an ML context) before sending
  anything — this is the one entry in this map where "who am I even
  emailing" is not yet answered, separate from the general no-outreach-
  before-batch-1 rule. No draft email is written for this entry.

---

## 9. Huginn rank-trajectory probing (Lu et al. 2025)

### Wenquan Lu
- **Current affiliation (live-verified, with caveat):** incoming PhD
  student, University of Michigan–Ann Arbor CSE, per his own site
  (wenquanlu.github.io) — he was at Brown University at the time the
  Huginn paper was written. **[UNVERIFIED current enrollment status]**
  ("incoming" per his own site, not confirmed as started). Has at least
  ~4 papers in the last ~18 months (this paper, a NeurIPS 2025 work, and
  a Feb 2026 GRPO paper), accepted at a COLM 2025 workshop.
- **Claim touched:** Lu, Yang, Lee, Li & Liu ("Latent Chain-of-Thought?
  Decoding the Depth-Recurrent Transformer," arXiv:2507.02199) probe rank
  trajectories across recurrent blocks in Huginn-3.5B (a depth-recurrent
  transformer reusing layers at inference) and find limited evidence of
  interpretable latent CoT via rank-trajectory analysis — already cited
  by name in our own ICML paper's related work
  (`matrix-thinking/submissions/icml-mi-workshop-2026/sections/
  06_related_work.tex`, `lu2025huginn`): "the two papers reach a kindred
  negative reading: rank-based probing of latent CoT does not, in either
  setting, straightforwardly reveal a multi-path superposition picture."
  Verdict: **CONFIRM** — an independent convergent negative finding
  across two different model classes (depth-recurrent shared-weight
  blocks vs. decoder-only GPT-2 with explicit per-position matrix
  latents) and two different observables (rank trajectory across
  recurrent depth vs. rank-k truncation of a trained latent).
- **What's in it for them:** our public matrix-state checkpoints
  (capability separation, mstar) give them a substrate where the
  underlying capability IS causally real and localized (S₀-zeroing
  collapses recall; the rank law causally holds) — unlike Huginn, where
  the interpretability question is open. Concrete experiment: apply
  their rank-trajectory probing methodology to our checkpoints across
  the composition-wall sweep (where we know exactly which cells have real
  capability and which don't) to test whether their negative
  rank-trajectory finding is architecture-general or specific to
  depth-recurrence — our ground-truth capability labels are something
  Huginn's setting doesn't offer them.
- **Co-author bar:** that cross-architecture replication, using our
  known-capability checkpoints as ground truth, is co-authorship-grade.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: A model where rank-based latent-CoT probing has ground truth
  >
  > Hi Wenquan — your Huginn rank-trajectory work and my own ICML paper
  > (matrix-CODI, rank-truncation on ProsQA) landed on the same kindred
  > negative reading: rank-based probing doesn't straightforwardly reveal
  > multi-path latent CoT in either setting. Separately, in from-scratch
  > matrix-state models we've built, there IS a real, causally-verified
  > capability living in the state (state-zeroing collapses it cleanly;
  > a rank law holds with a proven necessity/sufficiency floor) — a
  > setting where we know the ground truth, unlike Huginn's open
  > interpretability question. Checkpoints are public on HF. Would your
  > rank-trajectory probing method be easy to point at a system with
  > known ground truth, to see whether the negative finding is
  > architecture-general or specific to depth-recurrence?

### Other Lu et al. co-authors (brief, live-verified where noted)
- **Yuechuan Yang, Kyle Lee, Yanshu Li** — listed as Brown University at
  the time of the paper; no update found — **[UNVERIFIED current]**.
- **Enqi Liu** — Harvard University at the time of the paper; still
  collaborating with Lu as of a Feb 2026 paper.

---

## 10. Token uniformity (Hanqi Yan)

### Hanqi Yan
- **Current affiliation (live-verified):** Lecturer / Assistant Professor,
  King's College London — confirmed the same person as CODI's second
  author (Group 5 above) via her own Scholar profile, which lists both
  this paper and CODI. Has roughly 15 papers in the last ~18 months —
  comfortably clears the endorser paper-count bar. arXiv:2208.11790 is
  confirmed correct and was a **UAI 2022 spotlight**.
- **Claim touched:** Yan, Gui, Li & He ("Addressing Token Uniformity in
  Transformers via Singular Value Transformation," arXiv:2208.11790)
  characterize token-uniformity via the singular-value distribution of
  BERT-family encoder layer outputs and propose a spectrum-flattening
  transformation — rank decay in the ACTIVATIONS of a stack of attention
  layers. Our program's rank-law and rank-truncation work measures rank
  of an explicit matrix latent on a feedback path (ICML paper) or of a
  fast-weight/recurrent state (neurreps, mstar) — a different object, as
  our own related-work text already states explicitly ("the two objects
  of study are distinct," `matrix-thinking/submissions/icml-mi-workshop-
  2026/sections/06_related_work.tex`). Verdict: **BOUND** in the weak
  sense of scope-clarification only — we neither confirm nor contradict
  the token-uniformity phenomenon; we note our causal force-rank
  methodology is a transferable instrument, not a result about her
  specific object of study.
- **What's in it for them:** our causal force-rank / force-uniformity
  training-time intervention methodology (the same instrument that
  produced the neurreps causal razor) is directly transferable to test
  whether token uniformity is causally NECESSARY for some downstream
  capability, rather than merely a correlated phenomenon — something
  Yan et al.'s original paper does not test (they propose a fix, not a
  causal necessity claim). Concrete experiment: a training-time
  projection that forces the residual-stream singular-value spectrum
  toward or away from uniformity (analogous to our force-rank arms) on a
  downstream task with a known capability boundary, using our
  instrumentation as a template.
- **Co-author bar:** designing and running that causal
  uniformity-intervention experiment (even if it shows uniformity is
  NOT causally necessary for whatever task is tested) is co-authorship-
  grade — it would be a genuinely new causal result in her line of work.
- **Draft email (from Samuel Larson, Pebble ML):**
  > Subject: A causal-necessity test for token uniformity, borrowed from rank-law work
  >
  > Hi Hanqi — separately from CODI, I wanted to ask about your token-
  > uniformity paper. We've built a training-time causal intervention
  > (force a matrix state's rank to a target value via a training-time
  > projection, then measure downstream recovery) that cleanly separates
  > "the state has this property" from "this property is necessary for
  > the capability" on group-composition tasks. Your uniformity-fix
  > paper measures and corrects the phenomenon but, as far as I can
  > tell, doesn't test whether uniformity (or its absence) is causally
  > necessary for any specific downstream capability. Does a causal
  > version of that question already have an answer I'm missing, or is
  > it open? I'd be glad to share the intervention code if it's useful
  > as a starting point.

### Other co-authors (brief, live-verified where noted)
- **Lin Gui, Wenjie Li** — co-authors on arXiv:2208.11790. Wenjie Li's
  2022-era affiliation was Hong Kong Polytechnic University —
  **[UNVERIFIED as current]**. Lin Gui's later co-authorship with the CODI
  group above suggests a King's College London connection but this was
  not independently re-checked for this entry.
- **Yulan He** — senior/last author on both arXiv:2208.11790 and CODI
  (arXiv:2502.21074) — Professor of NLP, King's College London; UKRI
  Turing AI Fellow (confirmed, see Group 5 above; the same bridging role
  as Yan). Has roughly 25 papers in the last ~18 months.

---

## Candidate arXiv cs.LG / cs.AI endorsers (fact only — not an outreach list)

**Endorsement asks are explicitly NOT part of any email drafted above.**
This section only records, as a fact for later reference, which of the
people named in this map appear (on qualitative web-search evidence, not a
precise per-person arXiv API count) to have ≥3 recent cs.LG/cs.AI arXiv
papers and would therefore plausibly be eligible to endorse a cs.LG/cs.AI
submission on arXiv:

- **Songlin Yang** — well over 3 in the last ~18 months (Gated DeltaNet,
  GSA, Log-Linear Attention (ICLR 2026), Sparse State Expansion, and more,
  per her own publications page).
- **Yoon Kim** — senior/prolific MIT faculty; qualitatively certain, exact
  count [UNVERIFIED].
- **Jan Kautz** — NVIDIA VP, senior/prolific; qualitatively certain, exact
  count [UNVERIFIED].
- **Ali Hatamizadeh** — active 2026 record (Gated DeltaNet-2 May 2026,
  NVIDIA Nemotron 3 Super Mar 2026); qualitatively likely, exact count
  [UNVERIFIED].
- **Imanol Schlag** — multiple arXiv entries across years (2404.06508,
  2309.11197, 2102.11174, 2011.07831, 1811.12143); whether ≥3 fall in the
  strict last-12-18-months window is [UNVERIFIED] without a fresh listing
  pull.
- **Kazuki Irie** — active 2025-2026 record (Nature MI Oct 2025, ACL
  Findings Jul 2025, a Feb 2026 paper) but mixes journal/conference and
  arXiv venues; strict arXiv-only ≥3 count [UNVERIFIED].
- **Jürgen Schmidhuber** — senior/extremely prolific; qualitatively
  certain, exact count [UNVERIFIED].
- **Albert Gu, Tri Dao** — senior/prolific CMU and Princeton faculty;
  qualitatively certain, exact count [UNVERIFIED] for both.
- **Ion Stoica, Joseph Gonzalez** — senior/prolific Berkeley faculty;
  qualitatively certain, exact count [UNVERIFIED] for both.

Additional candidates confirmed in the second verification round:

- **Yu Sun** — 2-4 papers found in the window (incl. a Dec 2025 "End-to-End
  TTT for Long Context" paper); [UNVERIFIED] exhaustive count, common name
  makes a clean listing hard to pull.
- **Tatsunori Hashimoto** — ~15 papers per DBLP (5 in 2025, ~10 more in
  2026); comfortably clears the bar.
- **Shibo Hao** — ~13 papers in 2025-26; clears the bar.
- **Yuandong Tian** — ~144 total, dozens in 2025-26; clears the bar
  comfortably (now at Recursive Superintelligence, not Meta FAIR — see
  Group 5).
- **Zhenyi Shen** — ~5 papers in 2025-26; clears the bar.
- **Hanqi Yan** — ~15 papers in 2025-26; clears the bar.
- **Yulan He** — ~25 papers in 2025-26; clears the bar comfortably.
- **Wenquan Lu** — ~4 papers in the window; borderline-clears the bar,
  enrollment status itself [UNVERIFIED].
- **Eshaan Nichani** — ≥7 papers since Mar 2025; clears the bar.
- **Jason D. Lee** — prolific senior researcher (dozens/year); clears
  the bar comfortably.
- **Alberto Bietti** — ≥10 papers in the window; clears the bar.
- **Nicholas Barnfield** — ≥6 papers since Jan 2025; clears the bar.
- **Yue M. Lu** — senior Harvard faculty, prolific; qualitatively certain,
  exact count [UNVERIFIED].
- **T. Konstantin Rusch** — ≥15 papers in the window, extremely prolific;
  clears the bar comfortably (affiliation itself is the open question, not
  the paper count — see Group 7).
- **Ao Sun** — ≥10 cs.LG papers in the window; clears the bar.

**Explicitly does NOT clear the bar, or is flagged as unusable:**
- **Philipp Nazari** — only 2 confirmed cs.LG papers found via targeted
  search (a same-named astrophysicist pollutes broad searches, so this
  may be an undercount rather than a hard fail) — **[UNVERIFIED: count
  may be incomplete]**, do not treat as endorser-eligible without a
  cleaner search.
- **Keston Aquino-Michaels (arXiv:2607.19390, Group 8)** — only 2 ML
  papers found total, in a ~7-month-old persona, with an unresolved
  identity/affiliation question (see Group 8 above) — not usable as an
  endorser candidate, and not a safe outreach target at all without
  independent identity confirmation first.

Everyone else in this map (most Group 1 DeltaNet co-authors beyond Yang/
Kim/Kautz/Hatamizadeh, most Group 4/9 co-author footnotes, Group 5's
Zhang/Hu/Du/Gui/Li, Group 6's Juno Kim) was not individually checked for
arXiv paper-count purposes and should not be assumed either way.

**Note on precision:** none of the counts above come from a per-author
`arxiv.org/a/<author-id>` listing pull — they are qualitative
web-search/publications-page inferences. Before actually requesting an
endorsement from anyone on this list, pull that listing directly and count
cs.LG/cs.AI submissions in the literal last-12-months window arXiv's
endorsement system uses.

---

## Summary of what changed on live verification (read this before acting on anything above)

All three research subagents dispatched to verify affiliations have now
returned; every group above (1–10) carries live-verified affiliations and,
where a genuine outreach target exists, a draft email. Three findings are
significant enough to flag here explicitly rather than leave buried in
their sections:

1. **A citation error in this program's own `refs.bib`.** Group 6:
   `papers/rank-recruitment-ws/refs.bib`'s `barnfield2026sharp` entry
   records the wrong subtitle for arXiv:2605.05189 ("From Winner-Take-All
   to Listwise Retrieval" instead of the real "From Top-1 Retrieval to
   Tail-Average Learning" — author list is correct, only the subtitle is
   wrong). This is exactly the kind of error Will's citation audit
   (`papers/TRIAGE_2026-09-01.md` §4 item 5) is meant to catch — worth
   fixing before this ID is cited again anywhere.
2. **An unresolved author-identity question.** Group 8: arXiv:2607.19390
   is real and matches its internal description, but its actual author
   (Keston Aquino-Michaels, affiliation "No Way Labs") could not be
   confirmed as a reachable, verifiable person — there is a same-named
   individual with an unrelated 2016-2018 publication history in a
   different field, and no independent evidence of "No Way Labs" as an
   institution. No draft email was written for this entry; independently
   confirm identity before any contact.
3. **Several affiliations moved since the cited papers were written** and
   should be used as given here, not as the papers list them: Songlin Yang
   (MIT → Thinking Machines Lab), Bailin Wang (MIT → Meta), Kazuki Irie
   (Harvard postdoc → Yale faculty), Jason D. Lee (Princeton → UC
   Berkeley), Yuandong Tian (Meta FAIR → co-founder, Recursive
   Superintelligence). T. Konstantin Rusch and the two junior M²RNN
   co-authors (Mishra, Tan) carry unresolved/conflicting affiliation
   reports — treat those specific three as [UNVERIFIED] rather than
   picking one source.

Everything else — every remaining [UNVERIFIED] tag inline above — reflects
a genuine limit of a single web-search pass (a common name, a stale cached
page, a count that wasn't worth a full per-author arXiv-listing pull for
this draft), not a missing verification round. Re-check anything tagged
[UNVERIFIED] immediately before it is actually used, since affiliations
and paper counts are exactly the kind of fact that drifts between now and
whenever batch 1 actually goes live on arXiv — which, per the banner at
the top, is the earliest any of this should be acted on regardless.
