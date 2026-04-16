"""SQLite database schema and helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_DIR = Path.home() / ".autoskill"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "autoskill.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT,
    project_path  TEXT,
    created_at    INTEGER,
    last_message_at INTEGER,
    ingested_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    user_text       TEXT,
    response_summary TEXT,
    timestamp       INTEGER,
    has_code        BOOLEAN DEFAULT 0,
    has_paths       BOOLEAN DEFAULT 0
);

-- Tracks per-session distill progress so we only send new messages
CREATE TABLE IF NOT EXISTS distill_progress (
    session_id       TEXT NOT NULL,
    project_path     TEXT NOT NULL,
    last_message_id  INTEGER NOT NULL,
    distilled_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, project_path)
);

CREATE TABLE IF NOT EXISTS paths (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    path         TEXT UNIQUE,
    alias        TEXT,
    project_path TEXT,
    use_count    INTEGER DEFAULT 1,
    last_used    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS skills (
    id                 TEXT PRIMARY KEY,
    title              TEXT,
    description        TEXT,
    steps              TEXT,
    tags               TEXT,
    project_path       TEXT,
    source_session_ids TEXT,
    created_at         TEXT DEFAULT (datetime('now')),
    -- Structured observation fields (claude-mem inspired)
    obs_type           TEXT DEFAULT 'workflow',  -- workflow|bugfix|feature|decision|discovery
    facts              TEXT DEFAULT '[]',        -- JSON array of discrete assertions
    concepts           TEXT DEFAULT '[]',        -- JSON array of semantic tags
    files_read         TEXT DEFAULT '[]',        -- JSON array of file paths referenced
    files_modified     TEXT DEFAULT '[]',        -- JSON array of file paths changed
    narrative          TEXT DEFAULT '',           -- long-form explanation / context
    recurring          BOOLEAN DEFAULT 1,         -- 1 = recurring pattern, 0 = one-off task
    file_map           TEXT DEFAULT '{}',         -- JSON object: {filepath: [symbols]}
    command_templates  TEXT DEFAULT '[]'          -- JSON array of {name, template, description}
);

CREATE TABLE IF NOT EXISTS embeddings (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_type TEXT,
    ref_id   TEXT,
    vector   TEXT,
    text     TEXT
);

-- Project-specific prompt rules learned from review passes
CREATE TABLE IF NOT EXISTS prompt_rules (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_path TEXT NOT NULL,
    rule_text    TEXT NOT NULL,
    source       TEXT DEFAULT 'review',
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_paths_project ON paths(project_path);
CREATE INDEX IF NOT EXISTS idx_skills_project ON skills(project_path);
CREATE INDEX IF NOT EXISTS idx_embeddings_ref ON embeddings(ref_type, ref_id);
CREATE INDEX IF NOT EXISTS idx_prompt_rules_project ON prompt_rules(project_path);
"""


_MIGRATIONS = [
    ("obs_type", "ALTER TABLE skills ADD COLUMN obs_type TEXT DEFAULT 'workflow'"),
    ("facts", "ALTER TABLE skills ADD COLUMN facts TEXT DEFAULT '[]'"),
    ("concepts", "ALTER TABLE skills ADD COLUMN concepts TEXT DEFAULT '[]'"),
    ("files_read", "ALTER TABLE skills ADD COLUMN files_read TEXT DEFAULT '[]'"),
    ("files_modified", "ALTER TABLE skills ADD COLUMN files_modified TEXT DEFAULT '[]'"),
    ("narrative", "ALTER TABLE skills ADD COLUMN narrative TEXT DEFAULT ''"),
    ("recurring", "ALTER TABLE skills ADD COLUMN recurring BOOLEAN DEFAULT 1"),
    ("file_map", "ALTER TABLE skills ADD COLUMN file_map TEXT DEFAULT '{}'"),
    ("command_templates", "ALTER TABLE skills ADD COLUMN command_templates TEXT DEFAULT '[]'"),
]


def get_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and auto-create) the autoskill database."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    else:
        db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    # Migrate existing databases — add new columns if missing
    existing = {row[1] for row in conn.execute("PRAGMA table_info(skills)").fetchall()}
    for col_name, alter_sql in _MIGRATIONS:
        if col_name not in existing:
            conn.execute(alter_sql)
    conn.commit()
    return conn
