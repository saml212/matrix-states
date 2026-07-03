# Ship Runbook — Full Backlog Distribution

Distribution plan for everything currently on pebbleml.com: five findings + the literature synthesis. **One finding per week** to respect platform frequency caps and avoid attention dilution.

---

## What's already in the repo

**Site infrastructure (visible site changes — push to publish):**

- `pebble-ai-site/llms.txt` — GEO-spec file pointing LLMs at all five canonical findings + the literature synthesis.
- `pebble-ai-site/index.html` — JSON-LD `Organization` schema, canonical link, AI-authorship disclosure footer, footer date refreshed.
- `pebble-ai-site/findings/matrix-codi-rank-blindness.html` — JSON-LD `ScholarlyArticle` + canonical + disclosure footer.
- `pebble-ai-site/findings/outer-product-embedding.html` — same.
- `pebble-ai-site/findings/rank-enrichment.html` — same.
- `pebble-ai-site/findings/output-head-dynamics.html` — same.
- `pebble-ai-site/findings/parameter-efficiency.html` — same.
- `pebble-ai-site/findings/literature-synthesis.html` — same.

**Distribution drafts (`pebble-ai-site/_drafts/<slug>/` — Jekyll-excluded):**

- `rank-blindness/` — full set: substack, reddit-mlr, x-thread, linkedin, discord, github-readme, hf-dataset-card.
- `outer-product-embedding/` — substack, reddit-mlr, x-thread, linkedin, discord.
- `rank-enrichment/` — substack, reddit-mlr, x-thread, linkedin, discord.
- `output-head-dynamics/` — substack, x-thread, linkedin, discord. **No Reddit draft** (intentionally; the Reddit angle overlaps with rank-enrichment — see week 4 below).
- `parameter-efficiency/` — substack, reddit-mlr, x-thread, linkedin, discord.

**No drafts yet:**

- `literature-synthesis/` — deferred to week 6 capstone Substack, generated when we get there. Long piece; not Reddit-able as-is.

---

## Two discrepancies to fix in the canonical site (your call)

I noticed these while reading. Not blockers, but worth catching now:

1. **Parameter ratio mismatch.** `index.html` says "130× more parameter-efficient per layer." The canonical paper at `findings/parameter-efficiency.html` says **128×** (and the math gives exactly 128: 65,536 / 512). All my drafts use 128×. Update `index.html` for consistency.
2. **Finding numbering.** `index.html` lists rank-blindness as "no. 05." The rank-blindness paper itself is labeled "finding 06" in its title block. Either bump the index to 06 (matrix-CODI is post-rank-enrichment-and-output-head, so 06 is correct) or change the paper's "finding 06" to "finding 05."

---

## Release calendar

Default cadence: ship one finding per week, Tuesday morning Pacific (best news.smol.ai scrape window). Each finding gets ~5 days of attention before the next one ships.

| week | finding | substack | reddit | x | linkedin | discord |
|---|---|---|---|---|---|---|
| 1 | **rank-blindness** (paper) | ✅ | r/ML | ✅ | ✅ | optional |
| 2 | **outer-product embedding** | ✅ | r/ML | ✅ | ✅ | optional |
| 3 | **rank enrichment** | ✅ | r/ML | ✅ | ✅ | optional |
| 4 | **output head dynamics** | ✅ | (skip — see note) | ✅ | ✅ | optional |
| 5 | **parameter efficiency** | ✅ | r/ML [D] | ✅ | ✅ | optional |
| 6 | **literature synthesis** (capstone) | ✅ | (skip) | thread-of-threads | ✅ | (skip) |

**Why this order:**

1. Lead with the freshest, strongest paper — rank-blindness is an ICML workshop submission, peak relevance now.
2. Outer-product is the oldest finding but has the cleanest single-stat hook (-26% T=1 BPB at 2.2× param disadvantage). Strong second.
3. Rank-enrichment is the most novel-feeling result ("this runs counter to the literature"). Good X-thread bait.
4. Output-head-dynamics extends rank-enrichment — natural follow-up that timelines will read as a series.
5. Parameter-efficiency is structural, [D] flair on Reddit (discussion, not [R] research) — rounds out the picture without redundancy.
6. Literature synthesis is the framing capstone. Substack-only as a "what we've shipped, read against the field" piece.

**Why no Reddit on week 4 (output-head-dynamics):** The Reddit angle is largely redundant with the rank-enrichment Reddit post; cross-posting two highly-related findings within 14 days will get flagged by r/ML mods and dilute both. Drop a follow-up *comment* on the rank-enrichment Reddit post linking to output-head-dynamics instead. Drafts already note this.

**Why no Reddit on week 6 (synthesis):** Literature synthesis is a long-form context piece. It plays well on Substack and LinkedIn (long-form audiences), but Reddit punishes long posts that are not new experimental work.

---

## Weekly platform-by-platform sequence (template)

For each week's finding, run this sequence on Tuesday. Replace `<slug>` with the relevant draft folder name (e.g., `rank-blindness`, `outer-product-embedding`).

### Tuesday morning (9am PT)

1. **Substack first.** Open `_drafts/<slug>/substack.md`. Paste body. Title and subtitle from frontmatter. Set canonical URL in post settings → `https://pebbleml.com/findings/<slug>.html`. Publish.
2. **Reddit r/MachineLearning.** Open `_drafts/<slug>/reddit-mlr.md`. Title + body from file. Flair per frontmatter ([R] for empirical results, [D] for parameter-efficiency).
3. **30 min after Reddit:** drop a self-reply comment with the GitHub repo link.

### Tuesday afternoon

4. **X thread.** Open `_drafts/<slug>/x-thread.md`. Before posting, edit the link tweet to add 2–3 `@`-mentions of researchers from `pebble-ai-site/researcher-outreach-list.md` whose work is directly relevant to that finding. Pin tweet 1 to your profile for ~48 hours.
5. **LinkedIn.** Open `_drafts/<slug>/linkedin.md`. Attach the figure noted in frontmatter (screenshot from the canonical page). Publish.
6. **Discord (optional, manual).** Open `_drafts/<slug>/discord.md`. Long form for `#papers`-style channels, short form for `#share` / general. One server at a time. Wait until Wednesday or Thursday to give the Reddit/X traffic a chance to land first.

### Wednesday — surfacing check

72 hours after Tuesday's post, check news.smol.ai for pickup:

```bash
for d in $(date -v-3d -j +%y-%m-%d) $(date -v-2d -j +%y-%m-%d) $(date -v-1d -j +%y-%m-%d); do
  echo "=== $d ==="
  curl -s "https://news.smol.ai/issues/$d" | grep -iE "pebble|rank-blind|matrix-codi|outer-product|rank-enrich|output-head|parameter-effic|matrix-thinking|matrix-valued" || echo "(no hit)"
done
```

Log the result. If 4 weeks in a row produce zero hits, the upstream tactic isn't working — rethink.

### Friday — engagement check

By Friday, log per finding:
- Reddit upvotes + top-level comment quality.
- X thread impressions on tweet 1, retweets from researcher accounts.
- Substack subscribers gained that week.
- LinkedIn reactions, especially from any lab/recruiter accounts.

Numbers below thresholds (Reddit < 20 upvotes / X tweet 1 < 5k impressions) suggest the hook missed for that finding — note the pattern and adjust subsequent drafts.

---

## What you need to do BEFORE week 1 ships

These are the unblockers. Without them the ship sequence has nothing to publish into.

1. **Confirm the deploy mechanism for pebbleml.com.** The site footer links to `github.com/saml212/learned-representations`, but the prior handoff said the live site repo was `saml212/pebble-ai-site`. Tell me which one Pages actually serves and I'll adjust the push instructions. The HTML edits I made are local-only until you push to whichever repo serves the domain.
2. **Substack publication.** Create at https://substack.com. Suggested name "Pebble ML Research Log", URL slug `pebble-ml`. Generate an API key and save it in `~/.config/pebble-ml/substack.token`. (Round 1 is paste-and-publish even with the key; the API isn't generally available for posting on free tier as of April 2026.)
3. **GitHub org `pebble-ml`.** Create at https://github.com/account/organizations/new. Fine-grained PAT in `~/.config/pebble-ml/github.token`. Only required for week 1's HF dataset / GitHub repo creation; weeks 2–5 are paste-driven and don't need new repos.
4. **HuggingFace org `pebble-ml`.** Optional for week 1 (rank-blindness has eval JSONs that benefit from a HF dataset repo). Other weeks don't need HF.
5. **ORCID** (5 min, optional). Once you have one, paste it here and I'll bake it into all 6 finding HTMLs as `author.identifier`:
   ```
   ORCID: __________
   ```

---

## Per-finding notes / known issues

### Week 1 — rank-blindness

- **Repro scripts in the GitHub README do not exist.** Per CLAUDE.md, do not ship scripts you have not run. Recommendation: edit `_drafts/rank-blindness/github-readme.md` to remove the `repro/` section before pushing the GitHub repo. Ship the eval JSONs alone for round 1; add verified repro in a follow-up.
- HF dataset card draft exists at `_drafts/rank-blindness/hf-dataset-card.md`. It assumes you upload the JSON files from `experiment-runs/2026-04-17_round_pc/rank_evals/` to the HF dataset repo. Worth doing for this paper because it's a workshop submission with reviewers checking eval data; not strictly required to ship the Substack/Reddit/X.

### Week 2 — outer-product embedding

- Older finding (Feb 28). Frame on social as "an older finding I'm surfacing alongside the rank-blindness paper" — the drafts already do this.
- The init gotcha (σ = √target_std for outer products of N(0, σ²) tables) is a useful "anyone replicating will hit this" detail in the X thread and Reddit post. Keep it.
- Mechanism (rank-1 structure vs factored parameterization) is unresolved. The drafts say so. Don't let the X thread accidentally claim "matrices beat vectors" — frame it as embedding-layer-only.

### Week 3 — rank enrichment

- Single-seed result. Lead with the caveat in the Reddit post — drafts do.
- The cross-architecture pairing with rank-blindness is the most interesting part: rank-enrichment shows the bilinear-head case where rank rises and matters; rank-blindness shows the flatten-readout case where rank is rank-indifferent by construction. The X thread, Substack, and LinkedIn drafts all surface this pairing.

### Week 4 — output head dynamics

- Reddit post intentionally skipped (overlap with week 3). Drafts include only Substack, X, LinkedIn, Discord.
- LinkedIn / Substack should explicitly position this as a follow-on to rank-enrichment. Drafts do this.

### Week 5 — parameter efficiency

- Use [D] (discussion) flair on Reddit, not [R]. This is a structural observation, not an experimental result.
- The honest FLOP framing (8× per-step, not 128×) MUST be in tweet 1 or 2 of the X thread and the first paragraph of every other draft. Without it the post reads dishonestly. Drafts already lead with this.
- Realized speedup on H100 has not been measured. Drafts say so.

### Week 6 — literature synthesis (deferred draft generation)

- I haven't generated the social drafts for this one. We'll do it the week of, after seeing what got picked up from weeks 1-5. The synthesis is long-form (65KB) and naturally only fits Substack + LinkedIn + a "thread-of-threads" X format that points back at the previous five threads.

---

## After the 6-week run — the automation conversation

After all six findings ship, we'll have data on:
- Which findings drew engagement (which platforms, which framings)
- Whether news.smol.ai picked anything up
- Where drafts needed manual rewriting before posting (signal that the template is wrong)

That data tells us what to automate and what to leave manual:

- **Automate (probably):** the JSON-LD scaffolding for new findings, the canonical-mirror Substack body generation, the Discord paste-text generation. These are mechanical.
- **Keep manual (probably):** the Reddit post, the X thread (especially the @-mention selection), the LinkedIn tone calibration, the Discord channel selection. Per the brief's "do not auto-post to Reddit/X/LinkedIn" rule, and because tone/audience targeting is the part that matters and humans do better.
- **Build (likely):** the `/promote` skill that takes a `findings/<slug>.html` and emits a `_drafts/<slug>/` tree. We have a template now from this round; the skill turns it into one command.

We talk about that after week 6. Don't pre-automate based on what I think might work — we'll have actual signal by then.

---

## Open §9 questions still owed (from the original brief)

These don't block the calendar but I need them before round 2 of automation:

- ORCID ID.
- `upload-post` API location, if it exists.
- Substack publication slug + display name.
- Frequency cap confirmation: default is Substack ≤1/week, LinkedIn ≤2/week, X daily, Reddit ≤1 per subreddit per 2 weeks. The 6-week calendar respects all of these.
