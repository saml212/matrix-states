#!/usr/bin/env bash
# Stop hook: scan the assistant's last response in the transcript for [LEARN]
# blocks and insert them into workflow.db's learnings table.
#
# [LEARN] format (Claude is instructed to emit these in CLAUDE.md):
#   [LEARN] <category>: <rule>
#   Mistake: <what went wrong>
#   Correction: <what the right approach is>
#
# Fail-open: any error exits 0 so a broken hook never blocks the session.
set +e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/memory/workflow.db"
LOG="$ROOT/memory/hook.log"
SQLITE=/usr/bin/sqlite3

# Always log fire even if nothing captured, so we can confirm hook is loaded
echo "[$(date -Iseconds)] learn-capture: fired" >> "$LOG"

# Read hook stdin JSON
INPUT=$(cat)
TRANSCRIPT=$(echo "$INPUT" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("transcript_path",""))' 2>/dev/null)
[ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ] && exit 0

PROJECT=$(basename "$ROOT" | sed 's/^\.//')  # 'learned-representations' when invoked

# Extract last assistant text + insert any [LEARN] blocks
python3 - "$TRANSCRIPT" "$DB" "$PROJECT" "$SQLITE" "$LOG" <<'PYEOF'
import json, re, sys, subprocess, pathlib, datetime

transcript, db, project, sqlite_bin, log = sys.argv[1:6]

def log_msg(msg):
    with open(log, 'a') as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] learn-capture: {msg}\n")

try:
    # Read transcript.jsonl, collect last assistant turn's text blocks
    lines = pathlib.Path(transcript).read_text().strip().split('\n')
    last_assistant_text = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get('type') == 'assistant' and entry.get('message', {}).get('content'):
            for block in entry['message']['content']:
                if block.get('type') == 'text':
                    last_assistant_text.insert(0, block.get('text', ''))
            # only walk back until we hit the last assistant turn
            break
    text = '\n'.join(last_assistant_text)

    if not text:
        sys.exit(0)

    # Robust regex — matches [LEARN] category: rule, with optional Mistake/Correction
    pattern = re.compile(
        r'\[LEARN\]\s*([\w][\w\s\-/]*?)\s*:\s*(.+?)'
        r'(?:\n\s*Mistake:\s*(.+?))?'
        r'(?:\n\s*Correction:\s*(.+?))?'
        r'(?=\n\s*\[LEARN\]|\n\s*\n|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    matches = pattern.findall(text)
    if not matches:
        sys.exit(0)

    inserted = 0
    for category, rule, mistake, correction in matches:
        category = category.strip()
        rule = rule.strip().split('\n')[0].strip()  # first line only for rule
        mistake = (mistake or '').strip() or None
        correction = (correction or '').strip() or None

        # Dedupe: skip if an identical (category, rule) already exists for this project
        check = subprocess.run(
            [sqlite_bin, db, '-cmd', '.timeout 2000',
             f"SELECT id FROM learnings WHERE project = ? AND category = ? AND rule = ? LIMIT 1;"],
            input=None, capture_output=True, text=True
        )
        # Easier: use parameterized query via python subprocess + .read
        # Actually let's use sqlite3 -separator and -cmd
        q = subprocess.run(
            [sqlite_bin, db],
            input=f"SELECT id FROM learnings WHERE project='{project.replace(chr(39),chr(39)+chr(39))}' AND category='{category.replace(chr(39),chr(39)+chr(39))}' AND rule='{rule.replace(chr(39),chr(39)+chr(39))}' LIMIT 1;",
            capture_output=True, text=True
        )
        if q.stdout.strip():
            log_msg(f"dup skip: [{category}] {rule[:60]}")
            continue

        # Insert via parameterized approach — write to a temp script
        esc = lambda s: s.replace("'", "''") if s else None
        fields = f"'{esc(project)}','{esc(category)}','{esc(rule)}'"
        fields += f",'{esc(mistake)}'" if mistake else ",NULL"
        fields += f",'{esc(correction)}'" if correction else ",NULL"
        ins = subprocess.run(
            [sqlite_bin, db],
            input=f"PRAGMA trusted_schema=1;\nINSERT INTO learnings (project, category, rule, mistake, correction, source) VALUES ({fields},'claude');",
            capture_output=True, text=True
        )
        if ins.returncode == 0:
            inserted += 1
            log_msg(f"captured: [{category}] {rule[:80]}")
        else:
            log_msg(f"insert failed: {ins.stderr.strip()}")

    if inserted:
        log_msg(f"inserted {inserted} learning(s)")
        print(f"[learn-capture] {inserted} new learning(s) captured", file=sys.stderr)

except Exception as e:
    log_msg(f"error: {e}")

sys.exit(0)
PYEOF
exit 0
