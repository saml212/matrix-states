# Email draft — Sam → Will

---

## 🔴 BEFORE YOU SEND: rotate the Hugging Face token

**Do not add Will as a collaborator on `saml212/matrix-states` until this is
done.** Adding a collaborator grants full read access to the git *history*, and
there is a live credential in it.

**What's there.** A real hardcoded Hugging Face token was committed at
`matrix-thinking/H100_SETUP.md` in commit `ecb9f74`. It was redacted from HEAD in
commit `f19f5d3`, but **it is still in the history** — two history commits touch
an `hf_`-shaped literal in that file. Anyone with repo access can `git log -p`
and read it.

**Do this, in order:**

1. **Rotate the token at huggingface.co** (Settings → Access Tokens → revoke the
   old one, issue a new one). This is the step that actually matters — it kills
   the leaked value everywhere at once. It is **safe for any in-flight box
   work**: models are already cached on the box, so rotation won't trigger
   re-downloads.
2. *Optional but recommended before any wider sharing:* scrub the history with
   `git-filter-repo` or BFG and force-push. This is **destructive** and rewrites
   every SHA, so do it at a quiet moment when nothing is pushing to `origin`.
   Every commit SHA cited in `EXPERIMENT_LOG.md`, `STATE.md`, and the paper
   trees would change — which is a real cost for a repo whose audit trail *is*
   its git history. **Rotating alone is sufficient to make the leaked value
   worthless**; the scrub is hygiene, and given the SHA-citation cost it may be
   right to skip it.
3. Then add Will as a collaborator.

Related: the repo is still private for this reason, and this is also the blocker
on ever making it public.

**Also confirm before sending:** the five attachments and `ra_package.zip` are at
`ra-handoff/` in the repo, and a copy of the zip and this draft are on your
Desktop.

---

## The email

**To:** Williamlarson2023@gmail.com
**From:** sam@pebbleml.com
**Subject:** Taking over the paper portfolio — everything you need

---

Will,

Here it is. I want you to take over the whole paper portfolio — read into the
research, rewrite the papers in your own voice, make the figures and the
presentation genuinely good, decide where each one belongs, get them onto arXiv,
and run each submission end to end. You sign as co-author on every paper you
take through that process. That's not a courtesy line; the editorial and writing
work is real contribution and it should be credited as such.

Some honest context on why I'm handing this over rather than finishing it
myself. Over the last few months I've built something I think is genuinely
interesting, and I've built it with a lot of machine assistance and a very
heavy verification process — every number in every paper traces to a
checksummed raw artifact, and independent recomputation has caught real errors
more than once. That part I trust completely. What I don't have is prose that a
human researcher owns. There's a draft in there that went through six rounds of
AI-detection review and never cleared the bar, which is the honest signal that
it needs an author, not another polish pass. That's the job.

**What the work is.** Standard transformers carry a key-value cache that grows
with the sequence. There's a family of models that instead carries a fixed-size
matrix state, written by an outer-product rule. Everyone evaluates those models
on efficiency — quality per unit of compute. Nobody asks what the matrix state
actually *stores*. That's the whole program: treat the state as the object of
study.

Four results came out of it. Gradient descent recruits exactly the rank a task
needs — force it one rank lower and the task dies, restore it and it comes back.
A two-layer model does episodic recall at 0.9995 while a parameter-matched
transformer sits at chance and stays there even after you tune its learning rate
four ways. The same write mechanism drives a geometric pathology that gets
monotonically worse from 14M to 1.31B parameters and that the standard fix
doesn't fix. And the newest one, which is the spearhead: graft a composition
head onto a real 98M-parameter language model and it can execute exact
variable-depth composition in O(log h) steps through its own read path — but it
cannot *learn to write* the operator that makes that work. It's at chance,
everywhere, at every depth we've tested. We call that the wall, and the gap
between what the model can execute and what it can learn to write is the result.

Two chapters just landed proving that separation is scale-stable: across binding
loads from 12 to 40, and from 98M to 392M parameters, the capability stays flat
at ceiling and the wall holds. A 1.31B run is on the GPUs right now and should
land around the 26th.

**What I'm sending.** A folder called `ra-handoff/` in the repo, with five
documents:

- **README.md** — the master brief. The whole program in two pages, how the
  evidence chain works, how to work in the repo, and a glossary because we
  invented a lot of vocabulary and defined almost none of it.
- **PAPERS.md** — every paper tree, what it claims, how it builds, how finished
  it is, and what its venue situation actually is. Fifteen trees. One published,
  one live, seven finished-but-unsubmitted, six superseded.
- **SUBMISSION_PLAYBOOK.md** — arXiv mechanics, the OpenReview flows, and the
  venue decisions nobody has made.
- **DATA_MAP.md** — where every artifact lives.
- **EMAIL_DRAFT.md** — this.

Plus a zip with all of the above and the current PDF of every paper.

**Where to start.** Read the README, then PAPERS.md, then skim two or three of
the PDFs to get the flavor. Then the flagship — `papers/flagship/` — it's the
ICLR 2027 full paper and the biggest thing in the portfolio. It has six open
editorial questions listed out for you, including one that's just arithmetic: it
currently runs 23 pages of main text against a 9-page limit. After that, the
workshop stack.

One thing worth raising with me in your first week: I froze all of this in
mid-July while the GPUs ran, and most of the venue deadlines have since passed.
There's one that might still be live — a NeurIPS workshop deadline around Aug 29
— and three papers are sitting completely finished for exactly those venues. It's
in the playbook. Tell me what you think we should do about it; I may have the
priority order wrong for this particular week.

**A few things I want to say plainly.**

You have real latitude. Restructure the papers. Cut sections. Add figures — the
figures are the weakest part of everything in there and I think that's where you
can make the biggest difference fastest. Change the framing if you think the
framing is wrong. Ask me for more data; I have GPUs until about September 1st
and can talk about renting after that. And mine the experiment log — there's
more in there than made it into papers, including at least one clean mechanistic
result nobody has written up.

There are only two rules. Don't claim a number that doesn't trace back to a raw
archive. And don't quietly drop a caveat that a verdict record attached to a
claim — the caveats are what make the strong claims believable, and this whole
body of work lives or dies on being exactly as honest about its limits as it is
about its results.

On disclosure: several venues have policies about AI assistance, they differ,
and they change. Follow whatever the venue you're submitting to asks for at the
time you submit. We're in a good position on this — the science is fully
auditable from raw data by anyone who wants to check, which is a much stronger
place to stand than most people have.

**One housekeeping item.** The repo is private and I need to rotate a credential
before I add you. I'll do that and send the invite separately, probably same
day. The zip is self-contained in the meantime, so you can start reading
immediately.

Call me once you've read the README. I'd rather talk through the program than
have you reverse-engineer my reasoning from the docs.

Really glad you're doing this.

Sam

---

## Attachments

**`ra_package.zip`** (2.0 MB, 21 files) — send this one file. It contains:

| Contents | Notes |
|---|---|
| `MANIFEST.txt` | One-line description of every file, with page counts and status |
| `README.md` | Master brief |
| `PAPERS.md` | Portfolio inventory |
| `SUBMISSION_PLAYBOOK.md` | Submission mechanics + open decisions |
| `DATA_MAP.md` | Artifact locations |
| `EMAIL_DRAFT.md` | This file |
| `papers/` | 10 PDFs — the flagship, the eight workshop papers, the published one |
| `superseded/` | 3 PDFs — Gen-1 drafts, reference only, do not submit |

The zip contains papers only — no raw archives. Those are in the repo (and the
big payloads on the SSD); `DATA_MAP.md` says where.

One tree has **no PDF**: `matrix-thinking/submissions/iclr-2027/`, the attractor
draft. Its `main.tex` calls a style file that doesn't exist, so it has never
compiled. It is ~85% done and it is the portfolio's biggest open question
(absorb it into the flagship, or ship it separately?). `PAPERS.md` §C2 has the
detail. Note that `tectonic` works fine on this machine now — the "no TeX
toolchain" note in that tree is stale, so compiling it is a short task.
