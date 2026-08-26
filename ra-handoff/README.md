# Start here

**For:** Will Larson (Williamlarson2023@gmail.com)
**From:** Sam Larson (sam@pebbleml.com), PI
**Repo:** https://github.com/saml212/matrix-states (private) — this folder is `ra-handoff/`
**Written:** 2026-08-24 · **Updated:** 2026-08-25

Welcome. You're taking over the entire paper portfolio — fifteen paper trees,
one published, most of them finished-but-unsubmitted. This file is for you, the
human. It's short on purpose. The other files go deep.

---

## The job

Own the papers, end to end:

1. **Understand the research.** Read into it until you could defend it in a
   hallway. Sam would rather talk you through it than have you reverse-engineer
   his reasoning from documents — call him once you've read this.
2. **Rewrite them in your voice.** This is the real work. The prose in there was
   written with heavy machine assistance and it reads like it. One draft went
   through six rounds of AI-detection review and never cleared the bar, which is
   the honest signal that it needs an author rather than another polish pass.
3. **Make the figures and the presentation genuinely good.** The figures are the
   weakest part of everything in the portfolio, and they're the highest-leverage
   thing you can fix fastest. Add figures where a figure argues better than a
   table.
4. **Decide where each paper goes**, post to arXiv, and run each submission
   through its venue end to end.
5. **Sign as co-author** on every paper you take through that process. That's
   the PI's instruction, not a courtesy — the editorial and writing work is real
   contribution.

---

## What the work is, in plain language

A standard transformer remembers its context by keeping a cache that grows with
every token it reads. There's a family of models that instead keeps a
**fixed-size square grid of numbers** — a matrix — and writes new memories into
it by overwriting. The whole field evaluates those models on one axis: how much
quality you get per unit of compute. Nobody asks what that matrix actually
*stores*.

That's the program. Treat the matrix state as the thing being studied.

Four results came out of it:

- **Gradient descent recruits exactly the rank a task needs.** Force it one rank
  lower and the task dies outright; restore it and it comes back.
- **These models can do something matched transformers can't.** A two-layer
  model does single-pass episodic recall at 99.95% accuracy while a
  parameter-matched transformer sits at chance — and stays at chance after you
  tune its learning rate four different ways.
- **The same mechanism has a pathology.** At language-model scale the write
  operation drives its own memory keys into a collapsed geometry, and it gets
  monotonically worse from 14M to 1.31B parameters. The standard fix doesn't fix
  it.
- **The spearhead: the wall.** Graft a composition head onto a real 98M-parameter
  language model and it can *execute* exact multi-step composition through its own
  read path in logarithmic time — but it cannot *learn to write* the operator that
  makes that work. It reads chance, everywhere, at every depth tested. The gap
  between what the model can execute and what it can learn is the result.

Two chapters landed proving that gap is scale-stable and that its moat widens: it
holds across binding loads from 12 to 40, and from 98M to 392M parameters. A
1.31B run is finishing on the GPUs right now.

---

## Where everything lives

| What | Where | Why you'd open it |
|---|---|---|
| **This file** | `ra-handoff/README.md` | Orientation. You're here |
| **The inventory** | `ra-handoff/PAPERS.md` | Every paper tree: what it claims, how it builds, how finished it is, its venue situation. **Read this second** |
| **How to ship** | `ra-handoff/SUBMISSION_PLAYBOOK.md` | arXiv mechanics, OpenReview flows, the venue decisions nobody has made, the pre-submission checklist |
| **Where the numbers are** | `ra-handoff/DATA_MAP.md` | Every artifact location, and what must come off the GPU box before it dies |
| **For your Claude** | `ra-handoff/AGENT_INSTRUCTIONS.md` | The machine-facing brief. Point your agent at this, not at this README |
| **The email** | `ra-handoff/EMAIL_DRAFT.md` | The handoff email Sam sent you, plus a security note |
| **The papers** | `papers/` and `matrix-thinking/submissions/` | The actual trees. `papers/flagship/` is the big one |
| **Ground truth** | `EXPERIMENT_LOG.md` (repo root) | ~900 KB, append-only. Every experiment and every correction, dated and numbered. Nothing is ever edited in place |
| **The raw data** | `experiment-runs/<date>_<campaign>/` | Per-cell result JSONs with md5 manifests. Every number in every paper traces back to one of these |

---

## Day one: three things

1. **Read `PAPERS.md`.** It's the map. Fifteen trees, and it tells you which
   ones are alive.
2. **Open the flagship and build it.** `papers/flagship/` is the ICLR 2027 full
   paper and the biggest thing in the portfolio.
   ```bash
   cd papers/flagship/latex
   python3 md2tex.py              # sections/*.md is the source; .tex is generated
   tectonic -X compile main.tex   # -> main.pdf
   ```
   It compiles clean, exit 0, 26 pages. Six open editorial questions are listed
   in `PAPERS.md` §A1 — including one that's just arithmetic: 23 pages of main
   text against a 9-page limit.
3. **Check the deadline table** in `SUBMISSION_PLAYBOOK.md` §4–§5, then read the
   next section of this file. The calendar is the one thing that can't wait.

---

## The calendar problem — raise this with Sam in week one

The portfolio was frozen on 2026-07-17 while the GPUs ran, and most venue
deadlines have since passed. A live CFP check on 2026-08-24 turned up three
things worth knowing before you plan anything:

- **NeurReps 2026 EA closed Aug 24 AoE** — not Aug 29, which was the
  NeurIPS-wide *suggested* default. Two finished papers (`neurreps-ea`,
  `rank-recruitment-ws`) were built for exactly that track. **Ask Sam whether he
  submitted before the deadline.** If not, both need re-homing.
- **UniReps 2026 does not exist as a venue this year** — the site says TBA and
  it's absent from all 102 accepted NeurIPS workshops. `unireps-ea` needs a new
  home regardless.
- **ICBINB pivoted to biology**, which kills the on-record backup venue for the
  null paper.

So: no live clock is currently known, four papers need re-homing, and the
NeurIPS accepted-workshop list (102 workshops, now published) has never been
scanned end-to-end for candidates. That scan is probably a high-value first
week's afternoon. Sam's stated priority is the flagship, and over two months
that's right — the flagship's ICLR deadline is late September. Tell him what you
think the order should be.

---

## How to work with your Claude on this repo

There's a second file, **`AGENT_INSTRUCTIONS.md`**, written for the agent rather
than for you. Point your Claude at it at the start of a session:

> Read `ra-handoff/AGENT_INSTRUCTIONS.md` before touching anything in this repo.

It covers the rules that keep this work defensible — how the evidence chain
works, which files are read-only, how each paper tree builds, what it may change
freely and what it has to ask you or Sam about first. It also ends with a
glossary of the program's vocabulary, which you'll want too; we invented a lot
of terms and defined almost none of them in the papers.

The short version, if you only remember one thing: **this repo's whole strength
is that every number traces to a checksummed raw file.** An agent that "improves"
a number without opening the archive behind it destroys the thing that makes the
work believable. `AGENT_INSTRUCTIONS.md` exists to prevent exactly that.

---

## Latitude, and the two rules

**You have real latitude.** Restructure papers. Cut sections. Add figures. Change
the framing if you think the framing is wrong. Re-home a paper to a better venue.
Ask Sam for more data — the GPU grant runs to about **Aug 31**, and after that
it's a conversation about renting hardware. And mine `EXPERIMENT_LOG.md`: there's
considerably more in there than made it into papers, including at least one clean
mechanistic result nobody has written up.

**Two rules, and they're the whole ethic:**

1. **Don't claim a number that doesn't trace back to a raw archive.**
2. **Don't quietly drop a caveat that a verdict record attached to a claim.** The
   caveats are what make the strong claims believable.

**Authorship.** You sign as co-author on what you take through submission. Author
order and the full list are Sam's call — the trees currently carry
`[AUTHORS — PI decision pending]` placeholders. Settle it with him **before the
first arXiv posting**, because arXiv author lists are painful to change after the
fact.

Glad you're doing this. Call Sam when you've read `PAPERS.md`.
