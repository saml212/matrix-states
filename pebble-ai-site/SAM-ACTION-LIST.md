# Sam's Action List — GPU Credits & Pebble AI Setup
**Created:** April 1, 2026

Everything below is stuff only you can do. I've done everything I can on my end already.

---

## ALREADY DONE (by me)

- [x] **CloudRift AI Grant** — Web form submitted. Confirmation received.
- [x] **Together AI Research Credits** — Web form submitted. Confirmation received.
- [x] **Pebble AI landing page** created → `pebble-ai-site/index.html`
- [x] **Pitch materials doc** created → `pebble-ai-site/pitch-materials.md`
- [x] **Researcher outreach list** created → `pebble-ai-site/researcher-outreach-list.md`
- [x] **5 Gmail drafts** ready for you to review and send

---

## STEP 1: Send the 5 Gmail Drafts (5 minutes)

Go to [Gmail Drafts](https://mail.google.com/mail/u/0/#drafts). Review each one, then hit send.

1. **Vast.ai Startup Program** → startups@vast.ai
2. **Together AI Research Credits** → research-credits@together.ai (backup email to the web form already submitted)
3. **Lambda Labs Research Grant** → research@lambdalabs.com
4. **OVHcloud Startup Program** → startup@us.ovhcloud.com
5. **CoreWeave Startup Accelerator** → startups@coreweave.com

---

## STEP 2: Buy a Domain (~$7/year, 10 minutes)

**Why:** NVIDIA Inception (up to $100K in credits) requires a business email — Gmail is rejected. A domain also makes every other application stronger.

**Best cheap options** (check availability on Google Domains or Namecheap):
- `pebbleai.com` — first choice
- `pebble-ai.com` — backup
- `pebbleml.com` — if both taken
- `pebbleresearch.com` — fallback

**Where to buy:**
- **Google Domains** (domains.google.com) — ~$12/year .com, clean interface
- **Namecheap** — ~$7/year .com first year
- **Cloudflare Registrar** — at-cost pricing (~$9/year), no markup ever

**DO NOT** buy a `.ai` domain — they're $75+/year and not worth it at this stage.

---

## STEP 3: Set Up Free Business Email (15 minutes)

Once you have the domain, set up a free email so you have `sam@pebbleai.com` (or whatever domain you bought).

### Option A: Zoho Mail (Recommended — totally free)
1. Go to [zoho.com/mail](https://www.zoho.com/mail/) → click "Free Plan"
2. Sign up, add your domain
3. Follow their DNS verification steps (they walk you through it)
4. Add MX records to your domain registrar
5. Create `sam@yourdomain.com`
6. Free plan: 5 users, 5GB each, webmail access

### Option B: Cloudflare Email Routing (free, forwards to Gmail)
1. Transfer or add your domain to Cloudflare (free plan)
2. Go to Email → Email Routing
3. Set up `sam@yourdomain.com` → forwards to `samlarson16@gmail.com`
4. You receive at Gmail, but need to configure "Send as" in Gmail settings to reply from the business address
5. No extra inbox to check — everything lands in Gmail

---

## STEP 4: Host the Landing Page on GitHub Pages (15 minutes)

1. Go to [github.com](https://github.com) and sign in (or create account)
2. Create a new repository called `pebble-ai-site` (or `pebbleai.com`)
3. Upload `pebble-ai-site/index.html` from this folder as `index.html` in the repo root
4. Go to repo **Settings → Pages**
5. Source: "Deploy from a branch" → select `main` → `/ (root)` → Save
6. Your site will be live at `https://yourusername.github.io/pebble-ai-site/`
7. **Optional:** Add a custom domain (your new domain) in the Pages settings
   - Add a CNAME record pointing your domain to `yourusername.github.io`
   - GitHub provides free HTTPS

---

## STEP 5: Apply to NVIDIA Inception (THE BIG ONE — up to $100K)

**This is blocked until you have a business email.** Once Step 3 is done:

1. Go to [nvidia.com/en-us/startups/](https://www.nvidia.com/en-us/startups/)
2. Click "Apply Now" or "Join Inception"
3. Fill in with your **business email** (not Gmail — they reject it)
4. Company: Pebble AI
5. Industry: AI/ML Research
6. Stage: Pre-seed / R&D
7. Description: Use the "For startup programs" one-liner from `pitch-materials.md`:
   > Pebble AI is building next-generation AI architectures that enable models to reason through structured matrix representations — unlocking measurable abstract thought and cross-domain generalization from raw bytes.
8. Website: your GitHub Pages URL (or custom domain)
9. Benefits: Up to $100K in cloud credits, DGX access, technical support, networking

---

## STEP 6: Sign Up for Microsoft for Startups ($5K Azure credits)

1. Go to [microsoft.com/startups](https://www.microsoft.com/startups)
2. Click "Apply now" — sign in with a Microsoft account (create one if needed)
3. Company: Pebble AI
4. Use the startup one-liner from pitch-materials.md
5. This gives $5K in Azure credits including GPU VMs
6. **I can't create accounts for you** — that's why this is on your list

---

## STEP 7: Sign Up for Modal Labs ($25K–$50K potential)

1. Go to [modal.com](https://modal.com)
2. Create an account and a workspace
3. New accounts get $30 free credits automatically
4. Once you have a workspace name, go to their startup/research program page
5. Apply referencing the research — they're generous with ML researchers
6. **Tell me your workspace name** and I can help fill out the research credits form

---

## STEP 8: Google Cloud for Startups ($2K+)

1. Go to [cloud.google.com/startup](https://cloud.google.com/startup)
2. Sign in with your Google account
3. Apply to the Startups program
4. Use Pebble AI company info from pitch-materials.md
5. Base tier: $2,000 in Cloud credits
6. Higher tiers available if accepted to an accelerator

---

## STEP 9: AWS Activate ($1K–$100K)

1. Go to [aws.amazon.com/activate/](https://aws.amazon.com/activate/)
2. Create an AWS account if you don't have one
3. Apply to the **Founders** tier (self-funded startups, no accelerator needed):
   - $1,000 in AWS credits
   - Business support
4. If you get into any accelerator later, apply for **Portfolio** tier ($25K–$100K)

---

## STEP 10: Additional Programs to Apply To Later

These are worth doing once the domain/email/website are set up:

- **Scaleway Startup Program** (EU cloud, GPU instances) — [startups.scaleway.com](https://startups.scaleway.com)
- **AMD Instinct Developer Program** — free access to MI250/MI300 GPUs for research
- **RunPod** — Community credits for ML researchers, apply via their Discord
- **Paperspace (DigitalOcean)** — Gradient free tier + startup credits
- **Hugging Face** — Free community GPU grants for open-source research

---

## PRIORITY ORDER

If you're short on time, do these in order of impact:

1. **Send the 5 Gmail drafts** (5 min, no dependencies)
2. **Buy a domain** (10 min, unlocks everything below)
3. **Set up free email** (15 min, unlocks NVIDIA)
4. **Apply to NVIDIA Inception** (biggest single prize: up to $100K)
5. **Host landing page** (makes all applications stronger)
6. **Microsoft for Startups** (quick $5K)
7. **Modal Labs** (quick $30 free, potential $25K+)
8. Everything else as time allows

---

## WHAT I CAN STILL HELP WITH

Once you've done steps 1-4, come back and I can:
- Update the landing page with your real domain/URL
- Help fill out any application forms via browser
- Draft more tailored emails for specific programs
- Prepare a slide deck if any program wants one
- Set up the GitHub repo structure
- Write application essays/technical descriptions
