# Autopilot Build — Phase 1a Handoff

Phase 1a (notification transport + learnings DB + hooks) is complete. This
doc is for the next agent session picking up at Phase 1b or running Phase 1a
for the first time.

## What Phase 1a delivers

| Piece | File | What it does |
|---|---|---|
| Learnings SQLite DB | `.claude/memory/workflow.db` | Persistent store for `[LEARN]` blocks + seeded Hard Rules |
| Schema | `.claude/memory/schema.sql` | `learnings` + `learnings_fts` (FTS5) + `sessions` + `notifications` |
| Init script | `.claude/scripts/init_db.sh` | Idempotent — creates/updates DB from schema |
| Seed script | `.claude/scripts/seed_hard_rules.py` | One-shot — seeds 34 rules from CLAUDE.md + MEMORY.md |
| Stop hook | `.claude/hooks/learn-capture.sh` | Parses `[LEARN]` blocks from assistant responses → DB |
| UserPromptSubmit hook | `.claude/hooks/correction-detect.sh` | Nudges Claude when user uses correction language |
| UserPromptSubmit hook | `.claude/hooks/load-relevant-rules.sh` | Queries FTS5 for relevant rules → injects via stderr |
| Notify script | `.claude/scripts/notify.sh` | Fire-and-forget ntfy.sh publisher with tier support |
| Response poll | `.claude/scripts/ntfy_poll_responses.sh` | Cursor-based poll for user replies (filters autopilot posts by `robot_face` tag) |

## User setup (one-time)

1. **Install ntfy Android app.** F-Droid build is preferred (no Firebase
   dependency, more reliable background listener):
   - F-Droid: https://f-droid.org/en/packages/io.heckel.ntfy/
   - Or Play Store: `io.heckel.ntfy` if F-Droid sideload is a hassle

2. **Subscribe to the topic.** In the app: + → Subscribe to topic → enter
   `sam-autopilot-c77a80627ff7e5bf` → server `ntfy.sh` (default) → save.

3. **Whitelist from Do Not Disturb.** Settings → Notifications → Do Not
   Disturb → Apps → add ntfy. Without this, Tier 1 critical alerts will
   be silenced at night.

4. **Whitelist from battery optimization.** Settings → Battery → Battery
   optimization → ntfy → Don't optimize. Otherwise Android kills the
   background socket after ~30 min.

5. **Test the loop:**
   ```bash
   .claude/scripts/notify.sh 2 "Phase 1a verification" "If you see this on your phone, transport works."
   ```

6. **Send a reply back.** In the ntfy app, tap the topic, tap the
   publish/send icon, type anything, send. Then on Mac:
   ```bash
   .claude/scripts/ntfy_poll_responses.sh
   ```
   Should print your message as JSON.

7. **Restart Claude Code** so hooks in `.claude/settings.local.json` load.
   Hooks only register on session start.

## Conventions

### `[LEARN]` block format (what Claude emits)

When you learn something worth persisting, emit a block like:

```
[LEARN] <category>: <one-line rule>
Mistake: <what went wrong>
Correction: <what the right approach is>
```

Categories currently in use: `process`, `scale`, `matrix-arch`, `training`,
`distributed`, `pytorch`, `data`, `feedback`, `harness`, `sqlite-environment`,
`hook-design`, `bash-sqlite`.

`Mistake:` and `Correction:` are optional but recommended. The Stop hook
dedupes by `(project, category, rule)` so re-emitting the same rule is safe.

### Notification tiers

| Tier | Priority | Use |
|---|---|---|
| 1 | max (wakes through DND if whitelisted) | Unrecoverable halt, novel result |
| 2 | high (normal push) | Ready-for-review (blog draft, proposed experiments) |
| 3 | low (silent, tray only) | Informational (run completed, queue refilled) |

### Message filtering

Autopilot's own notifications include `robot_face` in their tags. The
response poll script filters these out so autopilot never re-reads its own
posts. If you publish from another tool, don't use `robot_face` as a tag
or it will be filtered.

## Known issues & caveats

- `.claude/` is in `.gitignore`. The entire harness lives in `.claude/`,
  so it's not committed. Options: (a) leave as-is, harness exists on
  this Mac only; (b) selectively untrack `.claude/hooks/`, `.claude/scripts/`,
  `.claude/memory/schema.sql`, `.claude/memory/seed_hard_rules.py`; or
  (c) symlink these to a committed dir like `harness/`. **Recommend (b)
  before Phase 2** so the harness survives a fresh clone.

- SQLite must be `/usr/bin/sqlite3` — the Homebrew/Android sqlite3 on
  PATH may lack FTS5 or have different PRAGMA defaults. This is pinned
  in all scripts.

- FTS5 triggers require `PRAGMA trusted_schema=1` for every INSERT.
  All hook scripts prepend this. If you write custom SQL, remember it.

- ntfy free public topic has 12-hour message cache. If phone is offline
  >12h, missed messages drop. Response poll tolerates gaps (just advances
  cursor past them).

## Phase 1b — Stone Claude Hub (next)

Target: modify Stone's Claude tile to open an intermediate "hub" screen
(modeled on the `connect` app pattern) that:
- Lists recent ntfy topic messages with Markwon markdown rendering
- Has a reply field that publishes back to the topic
- Keeps the "Open Claude app" link as one of the options on the hub

Files to touch in `/Users/samuellarson/Pebble/github/stone-app/`:
- `android/app/src/main/java/com/stonelauncher/MainActivity.kt` — rebind
  Claude tile to the new hub activity
- New: `android/app/src/main/java/com/stonelauncher/ui/ClaudeHubActivity.kt`
- `android/app/build.gradle` — add `io.noties.markwon:core:4.6.2` dep

Stone's existing `connect` app (multi-choice hub) is the UX template.
Copy its Activity + layout structure, then swap the choice grid for a
message list + reply field.

Stone is mostly bare-bones — it's an app launcher, not voice-native yet
despite what its README implies. So this is pure additive work, no fighting
existing architecture.

## Phase 2 — `/deploy-team` skill

Generic configurable Claude-agent team runtime (local, Max-subscription).
Templates: `gauntlet`, `pre-launch-audit`, `post-run-analysis`,
`blog-coherence-audit`. See the v2 spec the user provided earlier; trim
the devcontainer/repo-profile parts.

## Phase 3 — `/autopilot` skill

Wraps `experiment-runs/_auto_sync/WORKFLOW_FOR_AGENTS.md` substrate and
adds:
- Queue refill when <2h compute remains (invokes gauntlet team)
- Pre-launch code audit of queue candidates
- Post-run analysis team invocation on headline results
- Blog draft staging (team-audited, `pebble-ai-site/drafts/`)
- Notification dispatch on state changes

Never-idle invariant: `notify()` is fire-and-forget; state machine has no
"wait for user" state; work tiers (refill, audit, analysis, research,
blog) guarantee autopilot always has something to do while GPU trains
and user reads/responds async.
