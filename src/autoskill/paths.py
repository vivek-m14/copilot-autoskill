"""Path tracking, aliasing, and clipboard support."""

from __future__ import annotations

import subprocess
from .db import get_db


def list_paths(db_path=None, project: str | None = None, limit: int = 20) -> list[dict]:
    """Return top paths sorted by frequency, optionally filtered by project."""
    conn = get_db(db_path)
    if project:
        rows = conn.execute(
            "SELECT id, path, alias, project_path, use_count, last_used "
            "FROM paths WHERE project_path LIKE ? ORDER BY use_count DESC LIMIT ?",
            (f"%{project}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, path, alias, project_path, use_count, last_used "
            "FROM paths ORDER BY use_count DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_alias(path_or_id: str, alias: str, db_path=None) -> bool:
    """Set a human-friendly alias for a path."""
    conn = get_db(db_path)
    # Try by id first
    if path_or_id.isdigit():
        cur = conn.execute(
            "UPDATE paths SET alias = ? WHERE id = ?", (alias, int(path_or_id))
        )
    else:
        cur = conn.execute(
            "UPDATE paths SET alias = ? WHERE path = ?", (alias, path_or_id)
        )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def add_path(path: str, alias: str | None = None, project: str | None = None, db_path=None):
    """Manually add a path to tracking."""
    conn = get_db(db_path)
    conn.execute(
        "INSERT INTO paths (path, alias, project_path, use_count) "
        "VALUES (?, ?, ?, 1) "
        "ON CONFLICT(path) DO UPDATE SET "
        "use_count = use_count + 1, alias = COALESCE(?, alias)",
        (path, alias, project, alias),
    )
    conn.commit()
    conn.close()


def resolve_path(id_or_alias: str, db_path=None) -> str | None:
    """Resolve an id or alias to the actual path string."""
    conn = get_db(db_path)
    row = None
    if id_or_alias.isdigit():
        row = conn.execute(
            "SELECT path FROM paths WHERE id = ?", (int(id_or_alias),)
        ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT path FROM paths WHERE alias = ?", (id_or_alias,)
        ).fetchone()
    if not row:
        # Fuzzy match on path
        row = conn.execute(
            "SELECT path FROM paths WHERE path LIKE ? ORDER BY use_count DESC LIMIT 1",
            (f"%{id_or_alias}%",),
        ).fetchone()
    conn.close()
    return row["path"] if row else None


def copy_to_clipboard(text: str) -> bool:
    """Copy text to macOS clipboard using pbcopy."""
    try:
        subprocess.run(
            ["pbcopy"], input=text.encode(), check=True, capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
