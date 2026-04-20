#!/usr/bin/env bash
# Idempotent: creates workflow.db from schema.sql if missing, applies schema otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/memory/workflow.db"
SCHEMA="$ROOT/memory/schema.sql"
# Pin to /usr/bin/sqlite3 — the Android commandlinetools sqlite3 on PATH
# lacks FTS5. /usr/bin/sqlite3 ships with macOS and has FTS5 compiled in.
SQLITE=/usr/bin/sqlite3
# Delete and recreate if journal was corrupted by a failed FTS5 run
[ -f "$DB" ] && [ ! -s "$DB" ] && rm -f "$DB" "$DB-wal" "$DB-shm"
"$SQLITE" "$DB" < "$SCHEMA"
echo "db initialized: $DB"
"$SQLITE" "$DB" "SELECT count(*) || ' learnings, ' || (SELECT count(*) FROM sessions) || ' sessions, ' || (SELECT count(*) FROM notifications) || ' notifications' FROM learnings;"
