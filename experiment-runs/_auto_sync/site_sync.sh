#!/usr/bin/env bash
# Site sync: copy pebble-ai-site changes from the workspace mirror in
# learned-representations/pebble-ai-site/ into the real deploy repo at
# /tmp/pebble-ai-site, commit, and push. Safe to run repeatedly.
WORKSPACE=/Users/samuellarson/Experiments/learned-representations/pebble-ai-site
DEPLOY=/tmp/pebble-ai-site
LOG=/Users/samuellarson/Experiments/learned-representations/experiment-runs/_auto_sync/site_sync.log

log() { echo "[$(date -Iseconds)] $*" >> "$LOG"; }

if [ ! -d "$DEPLOY" ]; then
    log "deploy clone missing at $DEPLOY, reconstructing"
    cd /tmp && gh repo clone saml212/pebble-ai-site >> "$LOG" 2>&1 || exit 1
fi

cd "$DEPLOY"
git pull --quiet origin main 2>> "$LOG"

# Copy over everything from the workspace mirror except CNAME (already correct)
# and .git etc.
rsync -a --exclude='.git' --exclude='CNAME' --exclude='HANDOFF-PROMPT.md' --exclude='SAM-ACTION-LIST.md' --exclude='pitch-materials.md' --exclude='researcher-outreach-list.md' "$WORKSPACE/" "$DEPLOY/"

if [ -z "$(git status --porcelain)" ]; then
    log "no changes to deploy"
    exit 0
fi

git add -A
git commit -m "site-sync: update from workspace mirror ($(date -Iseconds))

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" >> "$LOG" 2>&1
git push origin main >> "$LOG" 2>&1 && log "pushed to deploy repo"
