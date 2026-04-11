# Handoff Prompt for Fresh Agent — Pebble AI Setup Busy Work

Copy everything below this line into the new agent.

---

You are picking up a multi-step setup task for Sam Larson, CEO of Pebble AI (samlarson16@gmail.com). The previous agent ran out of context. Your job is the **busy work**: fixing the live GitHub Pages site and pulling up the right browser tabs so Sam can create accounts on each grant program. **Sam will create the accounts himself** — you are NOT allowed to create accounts on his behalf. Once Sam has logged in to each platform, the original agent will return to write the actual application content. You are the setup crew, not the writer.

## Project context

- **Company:** Pebble AI. Research-stage ML company building novel matrix-valued token architectures.
- **Domain:** `pebbleml.com` (bought on Namecheap, fully configured).
- **Landing page repo:** `https://github.com/saml212/pebble-ai-site` (owner: `saml212`).
- **Landing page file:** `/sessions/funny-nifty-lamport/mnt/learned-representations/pebble-ai-site/index.html` (already committed to the repo).
- **Pitch materials:** `/sessions/funny-nifty-lamport/mnt/learned-representations/pebble-ai-site/pitch-materials.md` — use this for one-liners and company description.
- **Master action list:** `/sessions/funny-nifty-lamport/mnt/learned-representations/pebble-ai-site/SAM-ACTION-LIST.md` — read this first to understand the full scope.
- **Researcher outreach:** `/sessions/funny-nifty-lamport/mnt/learned-representations/pebble-ai-site/researcher-outreach-list.md`.

## State of DNS and GitHub Pages (verified)

- Namecheap Advanced DNS on `pebbleml.com` has these records (verified in the Namecheap tab and via `dns.google/resolve`):
  - `A @ 185.199.108.153`
  - `A @ 185.199.109.153`
  - `A @ 185.199.110.153`
  - `A @ 185.199.111.153`
  - `CNAME www saml212.github.io.`
  - `TXT @ v=spf1 include:zohomail.com ~all` (Zoho)
  - `TXT zmail._domainkey v=DKIM1; k=rsa; p=MIGfMA0GCSq...` (Zoho DKIM, propagating)
  - MX records for Zoho Mail are set.
- Google public DNS confirms both the four A records and the CNAME are live globally.
- GitHub Pages settings (`https://github.com/saml212/pebble-ai-site/settings/pages`):
  - Source: "Deploy from a branch" → `main` → `/ (root)`
  - Custom domain: `pebbleml.com` (set, CNAME file exists in repo)
  - Status as of handoff: **"DNS Check in Progress"** — GitHub's own verifier had not finished yet. It should be done by now. Refresh.
  - `https://pebbleml.com` was still returning "Site not found · GitHub Pages" at handoff time because GitHub's DNS check hadn't completed.
- **Sam said he sees problems on the GitHub Pages settings page.** You need to open that tab, screenshot it, and figure out what's broken. The previous agent only saw "DNS Check in Progress" but Sam noticed something more. Investigate and fix whatever is wrong.

## Open Chrome tabs at handoff

Use `mcp__Claude_in_Chrome__tabs_context_mcp` to get fresh tab IDs. These existed at handoff:
- `https://ap.www.namecheap.com/domains/domaincontrolpanel/pebbleml.com/advancedns` (Namecheap DNS — logged in)
- `https://github.com/saml212/pebble-ai-site/settings/pages` (GitHub Pages settings — logged in as saml212)
- `https://mailadmin.zoho.com/hosting?domain=pebbleml.com` (Zoho Mail admin)
- A Gmail tab

## Your task list (in order)

### 1. Fix the GitHub Pages site

- Navigate to `https://github.com/saml212/pebble-ai-site/settings/pages`, screenshot it, and get the full page text.
- Identify what's wrong. Possible issues to check:
  - DNS check still failing / domain not verified.
  - Build source somehow got reset to "GitHub Actions" again (previous agent had to switch it to "Deploy from a branch" twice).
  - Branch or folder setting wrong.
  - Custom domain field empty / mismatched.
  - A GitHub Actions workflow run failing (check `https://github.com/saml212/pebble-ai-site/actions`).
- Fix whatever is broken. If you need to toggle anything in the UI, use Chrome tools (the React form state is touchy — real click events via `computer.left_click` tend to work; JS value setters work for text inputs but only if you use `Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set` + dispatch `input` and `change` events).
- Confirm `https://pebbleml.com` eventually loads the landing page (the site in `index.html` in the repo). You may need to wait up to 10 min after DNS passes for the build.
- Once DNS check passes and the site loads on HTTP, wait for TLS and tick "Enforce HTTPS". TLS provisioning can take up to 24 hours — if it's still unchecked and greyed out, that's fine, leave a note and move on.

### 2. Pull up grant program tabs for Sam to make accounts

Sam explicitly said: **"I will make all the accounts that you mentioned. Just pull them up for me on chrome please."** So your job is to open one tab per program, navigated to the exact sign-up / apply page. Do NOT try to fill anything in. Do NOT create accounts. Just open the tabs and tell Sam they're ready.

Open these, one new tab each:

1. **NVIDIA Inception** — `https://www.nvidia.com/en-us/startups/` (click through to "Apply Now" / "Join Inception" so Sam lands on the sign-up form). Biggest prize: up to $100K.
2. **Microsoft for Startups** — `https://www.microsoft.com/en-us/startups` — Sam will sign in with a Microsoft account. $5K Azure.
3. **Modal Labs** — `https://modal.com/signup` (or the signup page reachable from modal.com). $30 automatic + potential $25K–$50K research credits.
4. **Google Cloud for Startups** — `https://cloud.google.com/startup` — click through to the application. $2K+ base tier.
5. **AWS Activate (Founders tier)** — `https://aws.amazon.com/activate/` — $1K Founders tier for self-funded startups.

After each tab is open, verify with a screenshot that it actually landed on the right page (these sites sometimes redirect based on geo or session state). Report the tab IDs back to Sam in a compact list so he knows which tab is which.

### 3. DO NOT do these (hard rules)

- **Do not create accounts** on any of the above services. Prohibited.
- **Do not enter passwords, credit card numbers, SSNs, or any sensitive financial info** in any form. Prohibited.
- **Do not send the 5 Gmail drafts** (Vast.ai, Together AI, Lambda Labs, OVHcloud, CoreWeave). Those need explicit per-draft approval from Sam and the original agent will handle review.
- **Do not write application content.** Once Sam has accounts made, he will come back to the *original* agent (not you) to write the actual applications using the Pebble AI pitch materials. Your job ends at "account signup page is loaded".
- **Do not touch the Namecheap DNS records** unless the GitHub Pages fix specifically requires it. DNS is known-good.
- **Do not modify** `SAM-ACTION-LIST.md`, `pitch-materials.md`, `researcher-outreach-list.md`, or `index.html` unless the GitHub Pages fix specifically requires an edit to `index.html`.

## Known quirks the previous agent learned the hard way

- **Namecheap React form state:** `computer.type` events sometimes land in the wrong input because the screenshot coordinate system is scaled from viewport `1823x1186` to screenshot `1372x893` (ratio ≈ 0.753). If you need to type into a Namecheap input, use the JS native-setter approach: get `Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set`, call it on the target input, then dispatch bubbling `input` and `change` events. That properly triggers React's dirty state. Clicking save buttons works fine via `element.querySelector('a.save').click()`.
- **Save button looks like a green checkmark ✓** on each row. On the row-level form it's `a.save`; the red X is `a.cancel`.
- **GitHub Pages source setting has flipped to "GitHub Actions" before when the previous agent tried to save.** If you see that, switch the dropdown to "Deploy from a branch" and save again.
- **The `CNAME` file** in the repo root contains exactly `pebbleml.com` with a newline. Don't let GitHub Pages clear it.
- **The landing page is a single-file `index.html`** — no CSS/JS files alongside it. If you need to edit it, the full file lives at `/sessions/funny-nifty-lamport/mnt/learned-representations/pebble-ai-site/index.html`.

## Report format when you're done

Give Sam a concise status report in this format:

```
GITHUB PAGES:
- Issue found: ...
- Fix applied: ...
- Site loading: yes / not yet (reason)
- HTTPS enforced: yes / no (reason)

ACCOUNT TABS OPENED (ready for Sam to sign up):
- Tab <id>: NVIDIA Inception  <url>
- Tab <id>: Microsoft for Startups  <url>
- Tab <id>: Modal Labs  <url>
- Tab <id>: Google Cloud for Startups  <url>
- Tab <id>: AWS Activate  <url>

BLOCKERS / QUESTIONS: ...
```

Then stop and wait. Sam will create the accounts, then switch back to the original agent for application writing.

## Tools you have

You're running in Cowork mode with Chrome browser tools (`mcp__Claude_in_Chrome__*`), file tools (Read/Write/Edit), Bash in a sandbox, TodoWrite, and the usual skills. Sandbox has no network, so for DNS lookups use `https://dns.google/resolve?name=X&type=Y` via a Chrome tab, not `dig` or `curl`.

Start by calling `mcp__Claude_in_Chrome__tabs_context_mcp` to see existing tabs, then go to the GitHub Pages settings tab and screenshot it to find the problem Sam noticed.
