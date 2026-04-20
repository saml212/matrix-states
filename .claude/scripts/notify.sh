#!/usr/bin/env bash
# notify.sh — fire-and-forget notification to the user's phone via ntfy.sh.
# Records each send in workflow.db.notifications so we can match user replies
# back to the original and avoid re-notifying about the same thing.
#
# Usage:
#   notify.sh <tier> <title> <body> [correlation_id]
#
# Tiers:
#   1 = critical (Priority: max, wakes phone; use sparingly)
#   2 = review   (Priority: high, normal push)
#   3 = info     (Priority: low, silent / tray only)
#
# Env overrides:
#   NTFY_TOPIC    — defaults to value below
#   NTFY_SERVER   — defaults to https://ntfy.sh
#
# Always exits 0 so a caller relying on notifications never fails because
# of notification problems (the "never idle" invariant).
set +e

NTFY_TOPIC="${NTFY_TOPIC:-sam-autopilot-c77a80627ff7e5bf}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/memory/workflow.db"
SQLITE=/usr/bin/sqlite3

TIER="${1:-3}"
TITLE="${2:-autopilot}"
BODY="${3:-}"
CORR="${4:-}"

# Tier → ntfy priority
case "$TIER" in
  1) PRIORITY="max";    TAGS="rotating_light,robot_face" ;;
  2) PRIORITY="high";   TAGS="eyes,robot_face"            ;;
  3) PRIORITY="low";    TAGS="information_source,robot_face" ;;
  *) PRIORITY="default"; TAGS="robot_face" ;;
esac

[ -z "$BODY" ] && { echo "notify.sh: empty body, skipping" >&2; exit 0; }

# Fire-and-forget via curl. 5s connect timeout, 10s total timeout.
# The ntfy response contains the message id we can correlate later.
RESPONSE=$(curl -sS --max-time 10 \
  -H "Title: $TITLE" \
  -H "Priority: $PRIORITY" \
  -H "Tags: $TAGS" \
  ${CORR:+-H "X-Correlation: $CORR"} \
  -d "$BODY" \
  "$NTFY_SERVER/$NTFY_TOPIC" 2>/dev/null)

NTFY_ID=$(echo "$RESPONSE" | python3 -c 'import sys,json
try: print(json.loads(sys.stdin.read()).get("id",""))
except: print("")' 2>/dev/null)

# Record in DB (best-effort)
if [ -f "$DB" ]; then
  ESC_TITLE=$(echo "$TITLE" | sed "s/'/''/g")
  ESC_BODY=$(echo "$BODY" | sed "s/'/''/g")
  if [ -n "$CORR" ]; then
    ESC_CORR=$(echo "$CORR" | sed "s/'/''/g")
    CORR_SQL="'$ESC_CORR'"
  else
    CORR_SQL="NULL"
  fi
  "$SQLITE" "$DB" "PRAGMA trusted_schema=1; INSERT INTO notifications (tier, title, body, topic, ntfy_id, correlation) VALUES ($TIER, '$ESC_TITLE', '$ESC_BODY', '$NTFY_TOPIC', '$NTFY_ID', $CORR_SQL);" 2>/dev/null
fi

# Secondary channel: macOS notification (only works if Mac is awake & unlocked)
if command -v osascript >/dev/null 2>&1; then
  SAFE_TITLE=$(echo "$TITLE" | sed 's/"/\\"/g')
  SAFE_BODY=$(echo "$BODY" | head -c 240 | sed 's/"/\\"/g')
  osascript -e "display notification \"$SAFE_BODY\" with title \"$SAFE_TITLE\" subtitle \"autopilot · tier $TIER\"" 2>/dev/null &
fi

echo "notify: tier=$TIER title=\"$TITLE\" ntfy_id=$NTFY_ID" >&2
exit 0
