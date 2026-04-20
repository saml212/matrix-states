#!/usr/bin/env bash
# UserPromptSubmit hook: advisory detection of correction language.
# Emits a stderr nudge so Claude knows the user is correcting it and should
# consider emitting a [LEARN] block. Also increments sessions.corrections_count.
# Does NOT block the prompt — fail-open.
set +e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/memory/workflow.db"
SQLITE=/usr/bin/sqlite3
echo "[$(date -Iseconds)] correction-detect: fired" >> "$ROOT/memory/hook.log"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("prompt",""))' 2>/dev/null)
SESSION_ID=$(echo "$INPUT" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("session_id",""))' 2>/dev/null)

[ -z "$PROMPT" ] && exit 0

# High-confidence correction patterns — targeted to avoid false positives
# on normal developer speech ("I should probably..." is not a correction).
if echo "$PROMPT" | grep -qE -i \
  "that'?s (wrong|not right|incorrect|not what (i|we))|you (forgot|shouldn'?t have|should have|need to)|wrong (file|function|variable|approach|directory|repo)|(undo|revert|rollback|roll back) that|i (said|told you|already told)|no,? (don'?t|not|it should|you should)|stop doing that"; then

  # Nudge to Claude
  echo "[correction-detect] User appears to be correcting you. If this is a durable rule, emit a [LEARN] block in your response so it's saved to the learnings DB." >&2

  # Increment counter (best-effort)
  if [ -n "$SESSION_ID" ] && [ -f "$DB" ]; then
    "$SQLITE" "$DB" "INSERT OR IGNORE INTO sessions (id, project) VALUES ('$SESSION_ID','learned-representations'); UPDATE sessions SET corrections_count = corrections_count + 1 WHERE id = '$SESSION_ID';" 2>/dev/null
  fi
fi

exit 0
