"""Ingest VS Code Copilot chat history into the autoskill database."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from .db import get_db

VSCODE_STORAGE = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Code"
    / "User"
    / "workspaceStorage"
)

# Regex for absolute paths (Unix-style)
PATH_RE = re.compile(r"(?:/[\w._-]+){2,}")
CODE_FENCE_RE = re.compile(r"```")


def _resolve_project_path(ws_dir: Path) -> str | None:
    """Read workspace.json to get the project folder."""
    ws_json = ws_dir / "workspace.json"
    if not ws_json.exists():
        return None
    try:
        data = json.loads(ws_json.read_text())
        folder = data.get("folder", "")
        if folder.startswith("file://"):
            return folder[7:]  # strip file:// prefix
        return folder or None
    except (json.JSONDecodeError, OSError):
        return None


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes and carriage returns."""
    return ANSI_RE.sub("", text).replace("\r", "")


def _extract_tool_result(item: dict) -> list[str]:
    """Extract rich info from a toolInvocationSerialized item."""
    parts: list[str] = []
    tsd = item.get("toolSpecificData")
    if isinstance(tsd, dict) and tsd.get("kind") == "terminal":
        # Terminal command
        cmd_line = tsd.get("commandLine", {})
        cmd = cmd_line.get("original", "") if isinstance(cmd_line, dict) else ""
        if cmd:
            parts.append(f"[cmd] {cmd}")
        output_obj = tsd.get("terminalCommandOutput", {})
        output_text = output_obj.get("text", "") if isinstance(output_obj, dict) else ""
        state = tsd.get("terminalCommandState", {})
        exit_code = state.get("exitCode", "") if isinstance(state, dict) else ""
        if output_text:
            cleaned = _strip_ansi(output_text).strip()[:500]
            exit_part = f" exit={exit_code}" if exit_code != "" else ""
            parts.append(f"[output{exit_part}] {cleaned}")
        return parts

    # MCP / extension tools with resultDetails
    result_details = item.get("resultDetails")
    tool_id = item.get("toolId", "")
    if isinstance(result_details, dict):
        raw_input = result_details.get("input", "")
        if tool_id:
            input_str = f" input: {raw_input}" if raw_input else ""
            parts.append(f"[tool:{tool_id}]{input_str}")
        output = result_details.get("output")
        if isinstance(output, list):
            for entry in output:
                if isinstance(entry, dict):
                    val = entry.get("value", "")
                    if val:
                        parts.append(f"[tool-output] {str(val).strip()[:300]}")
        elif isinstance(output, str) and output:
            parts.append(f"[tool-output] {output.strip()[:300]}")
        if parts:
            return parts

    # Fallback: invocationMessage or pastTenseMessage
    inv_msg = item.get("invocationMessage", "")
    past = item.get("pastTenseMessage", {})
    past_msg = past.get("value", "") if isinstance(past, dict) else ""
    if past_msg:
        parts.append(f"[tool] {past_msg}")
    elif inv_msg:
        parts.append(f"[tool] {inv_msg}")
    return parts


def _extract_response_text(response: list[dict]) -> str:
    """Extract readable text from the VS Code response array."""
    parts: list[str] = []
    for item in response:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind", "")
        value = item.get("value", "")

        if kind == "markdownContent" and value:
            parts.append(value)
        elif kind == "thinking" and value:
            text = str(value).strip()
            if text and len(text) < 500:
                parts.append(f"[thinking] {text}")
        elif kind == "toolInvocationSerialized":
            parts.extend(_extract_tool_result(item))
        elif kind == "textEditGroup":
            uri = item.get("uri", {})
            file_path = uri.get("path", "") if isinstance(uri, dict) else ""
            edits = item.get("edits", [])
            if file_path:
                parts.append(f"[edit] {file_path} ({len(edits)} change(s))")
            elif edits:
                parts.append(f"[edit] {len(edits)} change(s)")
    return "\n".join(parts)


def iter_chat_sessions(
    storage_dir: Path | None = None,
) -> Iterator[dict]:
    """Yield parsed chat session dicts from VS Code storage."""
    storage = storage_dir or VSCODE_STORAGE
    if not storage.is_dir():
        return

    for ws_dir in storage.iterdir():
        if not ws_dir.is_dir():
            continue
        chat_dir = ws_dir / "chatSessions"
        if not chat_dir.is_dir():
            continue

        workspace_id = ws_dir.name
        project_path = _resolve_project_path(ws_dir)

        for session_file in chat_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            requests = data.get("requests", [])
            if not requests:
                continue

            yield {
                "session_id": data.get("sessionId", session_file.stem),
                "workspace_id": workspace_id,
                "project_path": project_path,
                "created_at": data.get("creationDate"),
                "last_message_at": data.get("lastMessageDate"),
                "requests": requests,
            }


def ingest(db_path=None, storage_dir=None, quiet=False) -> dict:
    """Ingest all new VS Code chat sessions into the database.

    Returns stats dict: {sessions_added, messages_added, paths_found}.
    """
    conn = get_db(db_path)
    existing_ids = {
        row[0] for row in conn.execute("SELECT id FROM sessions").fetchall()
    }

    stats = {"sessions_added": 0, "messages_added": 0, "paths_found": 0}

    for session in iter_chat_sessions(storage_dir):
        sid = session["session_id"]
        if sid in existing_ids:
            continue

        conn.execute(
            "INSERT INTO sessions (id, workspace_id, project_path, created_at, last_message_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                sid,
                session["workspace_id"],
                session["project_path"],
                session["created_at"],
                session["last_message_at"],
            ),
        )
        stats["sessions_added"] += 1

        for req in session["requests"]:
            msg = req.get("message", {})
            user_text = msg.get("text", "") if isinstance(msg, dict) else str(msg)
            if not user_text.strip():
                continue

            response = req.get("response", [])
            resp_text = ""
            if isinstance(response, list):
                resp_text = _extract_response_text(response)

            has_code = bool(CODE_FENCE_RE.search(user_text) or CODE_FENCE_RE.search(resp_text))
            found_paths = PATH_RE.findall(user_text)
            has_paths = bool(found_paths)

            conn.execute(
                "INSERT INTO messages (session_id, user_text, response_summary, timestamp, has_code, has_paths) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    user_text,
                    resp_text[:5000] if resp_text else None,
                    req.get("timestamp"),
                    has_code,
                    has_paths,
                ),
            )
            stats["messages_added"] += 1

            # Track paths
            for p in found_paths:
                stats["paths_found"] += 1
                conn.execute(
                    "INSERT INTO paths (path, project_path, use_count, last_used) "
                    "VALUES (?, ?, 1, datetime('now')) "
                    "ON CONFLICT(path) DO UPDATE SET "
                    "use_count = use_count + 1, last_used = datetime('now')",
                    (p, session["project_path"]),
                )

    conn.commit()
    conn.close()
    return stats
