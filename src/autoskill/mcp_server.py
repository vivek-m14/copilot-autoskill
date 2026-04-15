"""MCP server exposing autoskill data as tools for LLM assistants.

Run via: autoskill serve
Or configure in VS Code settings.json / Claude Code as an MCP server.

Uses stdio transport so the host (VS Code, Claude) can launch it as a subprocess.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .db import get_db

mcp = FastMCP(
    "autoskill",
    instructions=(
        "Autoskill provides distilled skills, patterns, and knowledge from past "
        "Copilot conversations for this project. Use search_skills to find relevant "
        "patterns before solving problems — this saves tokens and avoids re-work."
    ),
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_json_field(val: Any) -> list:
    """Parse a JSON string field into a list, or return empty list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val:
        try:
            result = json.loads(val)
            return result if isinstance(result, list) else [result]
        except (json.JSONDecodeError, TypeError):
            return [val] if val else []
    return []


def _skill_to_dict(row) -> dict:
    """Convert a SQLite Row to a clean skill dict."""
    d = dict(row)
    # Parse JSON list fields for cleaner output
    for field in ("facts", "concepts", "files_read", "files_modified"):
        if field in d:
            d[field] = _parse_json_field(d[field])
    return d


# ── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool()
def search_skills(query: str, project: str | None = None, limit: int = 10) -> str:
    """Search distilled skills by semantic similarity.

    Use this FIRST when the user asks about a workflow, pattern, or task that
    might have been done before. Returns skills ranked by relevance.

    Args:
        query: Natural language description of what you're looking for
        project: Optional project path substring to filter results
        limit: Maximum number of results (default 10)
    """
    from .search import search

    results = search(query, include="skills", project=project, limit=limit)
    if not results:
        return "No matching skills found."

    output = []
    for r in results:
        output.append(
            f"**{r['title']}** (score: {r['score']:.2f})\n"
            f"  {r['description']}\n"
            f"  Project: {r.get('project', 'unknown')}"
        )
    return "\n\n".join(output)


@mcp.tool()
def get_skill(skill_id: str) -> str:
    """Get full details of a specific skill by ID (or ID prefix).

    Returns the complete skill with steps, facts, concepts, and file references.

    Args:
        skill_id: The skill ID or a prefix of it
    """
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM skills WHERE id = ? OR id LIKE ?",
        (skill_id, f"{skill_id}%"),
    ).fetchone()
    conn.close()

    if not row:
        return f"Skill not found: {skill_id}"

    skill = _skill_to_dict(row)
    parts = [
        f"# {skill['title']}",
        f"\n**Type:** {skill.get('obs_type', 'workflow')}",
        f"**Description:** {skill['description']}",
        f"**Tags:** {skill['tags']}",
        f"**Project:** {skill.get('project_path', 'global')}",
    ]

    if skill.get("steps"):
        parts.append(f"\n## Steps\n{skill['steps']}")

    facts = skill.get("facts", [])
    if facts:
        parts.append("\n## Key Facts")
        for f in facts:
            parts.append(f"- {f}")

    concepts = skill.get("concepts", [])
    if concepts:
        parts.append(f"\n**Concepts:** {', '.join(concepts)}")

    files = skill.get("files_read", []) + skill.get("files_modified", [])
    if files:
        parts.append(f"\n**Files:** {', '.join(files)}")

    if skill.get("narrative"):
        parts.append(f"\n## Context\n{skill['narrative']}")

    return "\n".join(parts)


@mcp.tool()
def list_project_skills(project: str) -> str:
    """List all distilled skills for a project.

    Args:
        project: Project path or substring to match
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, description, obs_type, tags FROM skills "
        "WHERE project_path LIKE ? ORDER BY created_at ASC",
        (f"%{project}%",),
    ).fetchall()
    conn.close()

    if not rows:
        return f"No skills found for project matching '{project}'."

    output = []
    for r in rows:
        output.append(
            f"- **{r['title']}** [{r.get('obs_type', 'workflow')}] (id: {r['id'][:10]})\n"
            f"  {r['description']}"
        )
    return f"Found {len(rows)} skills:\n\n" + "\n".join(output)


@mcp.tool()
def list_projects() -> str:
    """List all projects that have distilled skills."""
    conn = get_db()
    rows = conn.execute(
        "SELECT project_path, COUNT(*) as skill_count "
        "FROM skills WHERE project_path IS NOT NULL "
        "GROUP BY project_path ORDER BY skill_count DESC"
    ).fetchall()
    conn.close()

    if not rows:
        return "No projects with skills found. Run 'autoskill distill' first."

    from pathlib import Path

    home = str(Path.home())
    output = []
    for r in rows:
        proj = r["project_path"].replace(home, "~")
        output.append(f"- {proj} ({r['skill_count']} skills)")
    return "\n".join(output)


@mcp.tool()
def search_history(query: str, project: str | None = None, limit: int = 10) -> str:
    """Search past Copilot chat messages by keyword.

    Useful for finding how a specific problem was solved before, even if it
    wasn't distilled into a skill yet.

    Args:
        query: Keyword or phrase to search for
        project: Optional project path substring to filter
        limit: Maximum results (default 10)
    """
    conn = get_db()
    sql = (
        "SELECT m.user_text, m.response_summary, s.project_path "
        "FROM messages m JOIN sessions s ON m.session_id = s.id "
        "WHERE (m.user_text LIKE ? OR m.response_summary LIKE ?)"
    )
    params: list = [f"%{query}%", f"%{query}%"]
    if project:
        sql += " AND s.project_path LIKE ?"
        params.append(f"%{project}%")
    sql += " ORDER BY m.timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        return f"No chat history matching '{query}'."

    from pathlib import Path

    home = str(Path.home())
    output = []
    for r in rows:
        proj = (r["project_path"] or "").replace(home, "~")
        proj_name = proj.split("/")[-1] if proj else "unknown"
        resp = (r["response_summary"] or "")[:200]
        output.append(
            f"**[{proj_name}]** {r['user_text'][:150]}\n"
            f"  → {resp}"
        )
    return "\n\n".join(output)


@mcp.tool()
def get_frequent_paths(project: str | None = None, limit: int = 15) -> str:
    """Get frequently used file/directory paths for a project.

    Useful for auto-completing paths the user commonly references.

    Args:
        project: Optional project path substring
        limit: Maximum paths to return (default 15)
    """
    conn = get_db()
    sql = "SELECT path, alias, use_count FROM paths"
    params: list = []
    if project:
        sql += " WHERE project_path LIKE ?"
        params.append(f"%{project}%")
    sql += " ORDER BY use_count DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        return "No tracked paths found."

    output = []
    for r in rows:
        alias = f" (alias: {r['alias']})" if r["alias"] else ""
        output.append(f"- {r['path']}{alias} (used {r['use_count']}×)")
    return "\n".join(output)


# ── Server entry point ───────────────────────────────────────────────────────


def run_server():
    """Start the MCP server on stdio transport."""
    mcp.run(transport="stdio")
