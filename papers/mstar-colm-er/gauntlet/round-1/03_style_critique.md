# Style critique — round 1

Scope: `sections/00_abstract.tex` through `sections/10_appendix.tex` (11 files) plus
`main.tex` (title/author only). Checked mechanically against
`platform-skills/skills/paper/references/styleguide.md`, with the project-specific
DO-NOT substitution given in the task: no memory-multiplier performance ratios; no
rank/exact-recovery/capacity claims for the recall accuracy; multi-hop/task2 never
phrased as a capability result; no GPU-hours, cost, or cluster mentions; double-blind
(no names, orgs, github/huggingface links, acknowledgments). `%` comments (the
`<!-- evidence: Cn -->` tags) are not prose and were excluded from every check below.

### Banned words

Checked the verbatim list (honest, actually, really, just, clearly, obviously,
interestingly, nicely, remarkable, surprising, unfortunately, essentially, wildly,
literally, parsimonious, cleanest, sharpest), case-insensitive, whole-word, across
all 11 section files. Zero hits.

**No violations.**

### First-person / narrative-process

Zero instances of first-person "I"/"my"/"me" in prose. The one raw regex hit (`I` in
`sections/02_setup.tex:37`) is the identity matrix in the delta-rule equation
$S_t = S_{t-1}(I - \beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}$, not the pronoun.

Checked for narrative-process phrasing ("our original hypothesis", "we report a
negative result with a mechanism", "the paper's sharpest claim", self-referential
storytelling about the research process). The draft's several "this paper does not
claim X" / "this paper demonstrates Y" constructions (`03_recall.tex:40`,
`08_limitations.tex:4,24`, `06_scope.tex:61`, `02_setup.tex:24,87`,
`07_related.tex:13,19`, `10_appendix.tex:64`) are claim-scoping statements ("this
paper does not claim the additive update is architecturally incapable"), not
narration of the discovery process, and match the styleguide's own permitted pattern
of active "we"/"this paper" for findings. No instance of narrating what was
originally hypothesized, expected, or discovered was found (checked
"we found/discovered/noticed/observed/expected", "our finding", "hypothesized",
"we set out", "we decided" — zero hits).

**No violations.**

### Contractions

Zero contractions. Every `[A-Za-z]+'[A-Za-z]+` match across all 11 files (43 hits) is
a possessive ('s on a noun: "transformer's", "block's", "contender's", "model's",
"baseline's", "episode's", "project's", "variant's", "state's", "value's", "arm's",
"seed's", "control's", "workshop's", "paper's"), not a contraction. No "don't",
"can't", "it's", "we're", or similar found.

**No violations.**

### Em-dash-as-pause

No literal em-dash character and no `---` (LaTeX em-dash) anywhere in the 11 files.
Every `--` match is one of two non-violating cases: (a) inside a `% <!-- evidence:
Cn -->` comment tag (exempt, not prose), or (b) a standard en-dash compound/range in
prose — "key--value" (`00_abstract.tex:3`), "hop depths 1--2"
(`06_scope.tex:23`), "hop depths 3--4" (`06_scope.tex:34`) — which is correct LaTeX
typography for a range, not a dramatic pause.

**No violations.**

### Headings

All 16 `\section`/`\subsection` headings are noun phrases; none is phrased as a
question (no interrogative syntax, no "?"). Includes borderline-informal but still
non-interrogative titles such as "Where the recall lives" (`05_mechanism.tex:1`) and
"Scope: what does not separate" (`06_scope.tex:1`) — both are noun-phrase/relative-
clause constructions, not questions.

**No violations.**

### Captions

All 8 captions (`03_recall.tex`, `04_horizon.tex`, `05_mechanism.tex`,
`06_scope.tex` ×2, `10_appendix.tex` ×3) are self-contained: each names the
subtask/cell, the compared arms, and the takeaway inline, with no deferral to body
text. Zero hits for "TODO", "pending", "will be added", "forthcoming", "to be
added", "TBD" anywhere in the 11 files.

**No violations.**

### Abstract length

`sections/00_abstract.tex` word count (prose only, `%` evidence tags stripped):
**219 words**. Within the required 200–230 band.

**No violation.**

### DO-NOT / apparatus

- **Memory-multiplier ratios:** every "$M\times$"/"$1\times$ to $32\times$"/"$64\times$"
  figure in the draft either (a) describes the swept KV-cache budget range as an
  experimental-design fact (`01_intro.tex:38,57`), or (b) is an explicit *disclaimer*
  that no memory-multiplier headline is licensed — "no memory multiplier is claimed"
  (`00_abstract.tex:20`), "licenses no ``$M\times$ less memory'' headline"
  (`01_intro.tex:41`), "cannot certify a memory-multiplier tier"
  (`04_horizon.tex:53`), "we quote no ``... $M\times$ less memory'' figure"
  (`04_horizon.tex:56`), "no memory-multiplier tier" (`02_setup.tex:91`), "yields no
  memory multiplier" (`09_conclusion.tex:13`). None of these states a claimed
  performance-at-memory-ratio result. The one raw state-size ratio,
  "$64\times$ less raw state" (`03_recall.tex:44`), reports the ablation control's
  byte size as a disclosed confound, immediately followed by "capacity and
  write-conditioning are not disentangled by this ablation" — it narrows a claim
  rather than asserting a memory-for-performance ratio, so it does not read as the
  banned headline pattern.
- **Rank / exact-recovery / capacity claims on recall accuracy:** every "rank" and
  "capacity" mention (`02_setup.tex:25-28,52,60`, `03_recall.tex:16,48,57`,
  `07_related.tex:41-46`, `08_limitations.tex:11-13`) is a disclaimer scoping what the
  accuracy metric does *not* establish ("accuracy here never establishes rank,
  continuous recovery, or storage capacity"; "supports no statement about rank, exact
  recovery, or how much the 32,768-byte state holds"), consistent with the
  project rule.
- **Multi-hop/task2 as capability result:** every "capability" mention tied to the
  multi-hop variant (`01_intro.tex:65`, `06_scope.tex:17,29,32,39`) explicitly negates
  a capability claim ("trainability finding, not a capability separation";
  "trainability disclosure, not a capability claim"; "no capability separation is
  claimed in either direction"; "not a demonstrated capability"). The one other
  "capability" use (`06_scope.tex:60`, "where between the two load points the
  capability degrades") refers to the single-hop $K{=}48$ stress cell, not the
  multi-hop variant, so it is outside this rule's scope; it is also hedged as an open
  question, not a claim.
- **GPU-hours / cost / cluster:** zero mentions of GPU count, GPU-hours, dollar cost,
  or cluster/hardware naming anywhere in the draft. The two "cost" hits
  (`01_intro.tex:7`, `08_limitations.tex:26`) refer to the general deployment-cost
  motivation of KV-cache growth (the paper's problem statement), not the paper's own
  experiment compute budget.
- **Apparatus/experiment-count bragging:** zero hits for "extensive",
  "large-scale", "numerous", "comprehensive", or similar self-congratulatory
  apparatus language.

**No violations.**

### Anonymization

`main.tex:18` correctly uses `\author{Anonymous authors}`. Grep across all 11
section files plus `main.tex` for author/org identifiers, de-anonymizing URL
patterns (`github.com/`, `huggingface.co/`), and acknowledgment/funding language
(`acknowledg`, `self-funded`, `funded by`), plus project-specific identity tokens
(names, "pebble", "anthropic", "brev", hardware names, institution suffixes):
**zero matches.** The appendix's "will be released with the de-anonymized version of
this paper" (`10_appendix.tex:71`) is a standard double-blind release disclosure, not
an identity leak.

**No violations.**

---

## Verdict

**PASS (0 violations)**
