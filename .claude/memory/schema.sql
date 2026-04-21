-- Learnings DB — adapted from rohitg00/pro-workflow
-- Two tables + FTS5. Queryable via recency and full-text search.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS learnings (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  project       TEXT,
  category      TEXT NOT NULL,
  rule          TEXT NOT NULL,
  mistake       TEXT,
  correction    TEXT,
  source        TEXT,                -- 'claude', 'seed', 'manual'
  times_applied INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_learnings_category  ON learnings(category);
CREATE INDEX IF NOT EXISTS idx_learnings_project   ON learnings(project);
CREATE INDEX IF NOT EXISTS idx_learnings_created   ON learnings(created_at);

-- UNIQUE on (project, category, rule) — atomic dedupe via INSERT OR IGNORE.
-- COALESCE(project, '') so rows with NULL project still participate.
CREATE UNIQUE INDEX IF NOT EXISTS idx_learnings_unique
  ON learnings(COALESCE(project,''), category, rule);

CREATE VIRTUAL TABLE IF NOT EXISTS learnings_fts USING fts5(
  category, rule, mistake, correction,
  content=learnings, content_rowid=id,
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS learnings_ai AFTER INSERT ON learnings BEGIN
  INSERT INTO learnings_fts(rowid, category, rule, mistake, correction)
  VALUES (new.id, new.category, new.rule, new.mistake, new.correction);
END;

CREATE TRIGGER IF NOT EXISTS learnings_ad AFTER DELETE ON learnings BEGIN
  INSERT INTO learnings_fts(learnings_fts, rowid, category, rule, mistake, correction)
  VALUES ('delete', old.id, old.category, old.rule, old.mistake, old.correction);
END;

CREATE TRIGGER IF NOT EXISTS learnings_au AFTER UPDATE ON learnings BEGIN
  INSERT INTO learnings_fts(learnings_fts, rowid, category, rule, mistake, correction)
  VALUES ('delete', old.id, old.category, old.rule, old.mistake, old.correction);
  INSERT INTO learnings_fts(rowid, category, rule, mistake, correction)
  VALUES (new.id, new.category, new.rule, new.mistake, new.correction);
END;

CREATE TABLE IF NOT EXISTS sessions (
  id                 TEXT PRIMARY KEY,
  project            TEXT,
  started_at         TEXT NOT NULL DEFAULT (datetime('now')),
  ended_at           TEXT,
  edit_count         INTEGER NOT NULL DEFAULT 0,
  corrections_count  INTEGER NOT NULL DEFAULT 0,
  prompts_count      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project, started_at);

-- Notifications sent via notify.sh — tracked so autopilot can reference by id
-- when matching responses and avoid re-notifying about the same thing.
CREATE TABLE IF NOT EXISTS notifications (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  sent_at     TEXT NOT NULL DEFAULT (datetime('now')),
  tier        INTEGER NOT NULL,      -- 1=critical, 2=review, 3=info
  title       TEXT,
  body        TEXT NOT NULL,
  topic       TEXT,
  ntfy_id     TEXT,                  -- message id returned by ntfy
  correlation TEXT,                  -- e.g. 'blog-draft-round9', 'queue-refill'
  acked_at    TEXT,
  response    TEXT                   -- user's reply text if any
);

CREATE INDEX IF NOT EXISTS idx_notif_sent ON notifications(sent_at);
CREATE INDEX IF NOT EXISTS idx_notif_corr ON notifications(correlation);
