"""Distill reusable skills from Copilot conversations via gh copilot.

Skills are project-scoped. The LLM receives full conversation context
(user prompts + response summaries) in chunked batches (~10 sessions).
Existing skills are fed back so the LLM can update/merge rather than duplicate.
Skills are stored in both SQLite (for search) and markdown files (for browsing).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .db import get_db, DEFAULT_DB_DIR

SKILLS_DIR = DEFAULT_DB_DIR / "skills"
BATCH_SIZE = 10  # sessions per LLM call
DEFAULT_MODEL = "claude-sonnet-4.6"  # good balance of cost and quality for summarization

# ── Prompt templates ────────────────────────────────────────────────────────

DISTILL_PROMPT = """\
You are a skill extractor for a software project. You produce structured \
observations from developer conversations with an AI coding assistant.

PROJECT: {project}

Below is a batch of conversation history. Each conversation has USER prompts \
and RESPONSE summaries.

Response summaries now include actual terminal commands prefixed with [cmd], their output \
prefixed with [output], and file edits prefixed with [edit]. Pay special attention to [cmd] \
lines — these are the EXACT commands Copilot ran and are the most valuable signal for \
creating reusable skills. When you see the same command pattern repeated across sessions \
(even with different arguments), that's definitely a skill.

{conversations}

{existing_skills_section}

Your task:
1. Identify reusable workflow patterns, recurring tasks, and problem-solution pairs.
2. Pay special attention to SHORT REPEATED COMMANDS — if the user asks for the same \
kind of task multiple times (even with different paths/inputs), that IS a skill. \
Examples: "make a flat dir", "run compare_experiments for X", "copy files from A to B".
3. For each pattern, produce a structured observation. Use placeholders like <PATH>, \
<INPUT>, <DATASET> for variable parts so the skill is reusable.
4. If an existing skill already covers a pattern, output an UPDATED version with \
any new details merged in (keep the same skill_id).
5. If a pattern is genuinely new, create a new skill.
6. Prefer MORE granular skills over fewer broad ones. A specific "Flatten directory \
with symlinks" skill is better than lumping it into a generic "File operations" skill.

Respond with a JSON array of skill objects (no markdown fences). Each object has:
- "skill_id": string — reuse the existing id if updating, or "new" for new skills
- "title": short name (≤8 words)
- "type": one of "workflow", "bugfix", "feature", "decision", "discovery"
- "description": one-sentence summary
- "steps": markdown bullet list of reusable steps
- "tags": list of 3-5 keyword tags
- "facts": list of 2-5 discrete factual assertions (e.g. "script expects --input and --output flags", "output goes to /results/")
- "concepts": list of 2-4 high-level concepts this relates to (e.g. "image-processing", "data-pipeline", "deployment")
- "files": list of file/directory paths mentioned or relevant (can be empty)

Respond ONLY with the JSON array."""

EXISTING_SKILLS_HEADER = """\
EXISTING SKILLS for this project (update these if the conversations add new details):
{skills_text}"""

PROJECT_RULES_HEADER = """\

PROJECT-SPECIFIC RULES (learned from past reviews — follow these strictly):
{rules_text}"""

REVIEW_PROMPT = """\
You are a skill quality reviewer for a software project.

PROJECT: {project}

Below are the CONVERSATIONS that were distilled:

{conversations}

Below are the SKILLS that were extracted:

{skills_text}

Your task: Review whether the skills adequately capture the patterns in the conversations.

Look for:
1. MISSED PATTERNS — repeated tasks the user asked for that didn't become their own skill \
(e.g. user asked "make a flat dir" 5 times across sessions, but no flat-dir skill exists)
2. TOO BROAD — skills that lump together distinct tasks (e.g. "File Operations" covering \
both directory flattening AND file copying — these should be separate)
3. MISSING DETAILS — skills that exist but are missing key steps or context from the conversations

Output a JSON array of rule objects. Each rule is a short instruction for the skill extractor \
to follow in future distillation runs. Examples:
- {{"rule": "Always extract 'flatten directory with symlinks' as its own skill — user does this frequently"}}
- {{"rule": "The user's compare_experiments workflow always involves: run script, generate HTML report, open in browser"}}
- {{"rule": "Prefer granular per-script skills over broad 'run training' umbrella skills"}}

Only output rules for GENUINE gaps. If the skills are good, return an empty array [].
Do NOT repeat rules that would be obvious to any skill extractor.

Respond ONLY with the JSON array (no markdown fences)."""


# ── Helpers ─────────────────────────────────────────────────────────────────


def _call_copilot(prompt: str, model: str | None = None) -> str:
    """Call gh copilot in non-interactive mode and return the response text."""
    cmd = ["gh", "copilot", "--", "-p", prompt, "--model", model or DEFAULT_MODEL]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"gh copilot failed: {result.stderr[:500]}")
    return result.stdout.strip()


def _parse_skills_json(raw: str) -> list[dict]:
    """Parse LLM response as a JSON array of skill objects."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    # Try direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass
    # Find JSON array in text
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    # Try finding a single object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end])
            return [obj] if isinstance(obj, dict) else []
        except json.JSONDecodeError:
            pass
    return []


def _normalize_skill(skill_data: dict) -> dict:
    """Ensure all skill fields are strings (or JSON strings for list fields)."""
    for field in ("steps", "description", "title", "narrative"):
        val = skill_data.get(field)
        if isinstance(val, list):
            skill_data[field] = "\n".join(str(v) for v in val)
        elif val is None:
            skill_data[field] = ""
    tags = skill_data.get("tags", [])
    if isinstance(tags, list):
        skill_data["tags"] = ",".join(str(t) for t in tags)
    elif not isinstance(tags, str):
        skill_data["tags"] = str(tags)
    # Normalize structured observation fields to JSON strings
    for json_field in ("facts", "concepts", "files_read", "files_modified"):
        val = skill_data.get(json_field)
        if isinstance(val, list):
            skill_data[json_field] = json.dumps(val)
        elif val is None:
            skill_data[json_field] = "[]"
        elif isinstance(val, str):
            # Validate it's valid JSON, else wrap
            try:
                json.loads(val)
            except json.JSONDecodeError:
                skill_data[json_field] = json.dumps([val])
    # Handle "files" as a combined field → split into files_read
    if "files" in skill_data and "files_read" not in skill_data:
        files_val = skill_data.pop("files")
        if isinstance(files_val, list):
            skill_data["files_read"] = json.dumps(files_val)
        elif isinstance(files_val, str):
            skill_data["files_read"] = files_val
    elif "files" in skill_data:
        skill_data.pop("files")  # already handled
    # Normalize obs_type
    valid_types = {"workflow", "bugfix", "feature", "decision", "discovery"}
    obs_type = skill_data.get("type", skill_data.get("obs_type", "workflow"))
    skill_data["obs_type"] = obs_type if obs_type in valid_types else "workflow"
    return skill_data


def _project_slug(project_path: str) -> str:
    """Turn a project path into a filesystem-safe folder name."""
    name = Path(project_path).name if project_path else "global"
    return re.sub(r"[^\w.-]", "_", name)


def _write_skill_markdown(skill: dict, project_path: str | None):
    """Write a skill as a markdown file in the skills folder."""
    slug = _project_slug(project_path)
    skill_dir = SKILLS_DIR / slug
    skill_dir.mkdir(parents=True, exist_ok=True)

    title = skill.get("title", "Untitled")
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "-").lower()
    filename = f"{safe_title}.md"

    # Parse JSON list fields for display
    def _fmt_list(val):
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val
        if isinstance(val, list):
            return "\n".join(f"- {item}" for item in val)
        return str(val) if val else ""

    obs_type = skill.get("obs_type", skill.get("type", "workflow"))
    facts = _fmt_list(skill.get("facts", "[]"))
    concepts = _fmt_list(skill.get("concepts", "[]"))
    files = _fmt_list(skill.get("files_read", "[]"))

    content = f"""# {title}

> {skill.get('description', '')}

**Type:** {obs_type}
**Tags:** {skill.get('tags', '')}
**Project:** {project_path or 'global'}
**Skill ID:** {skill.get('id', 'unknown')}

## Steps

{skill.get('steps', '(no steps)')}

## Facts

{facts or '(none)'}

## Concepts

{concepts or '(none)'}

## Files

{files or '(none)'}
"""
    (skill_dir / filename).write_text(content)
    return str(skill_dir / filename)


# ── Load conversation batches ───────────────────────────────────────────────


def _load_project_conversations(conn, project_path: str, full: bool = False) -> list[dict]:
    """Load conversations for a project, grouped by session.

    If full=False (default), only loads messages newer than the last distilled message
    per session — this is the incremental mode that saves tokens.
    """
    if full:
        rows = conn.execute(
            "SELECT s.id as session_id, m.id as msg_id, m.user_text, m.response_summary, m.timestamp "
            "FROM messages m JOIN sessions s ON m.session_id = s.id "
            "WHERE s.project_path = ? AND m.user_text IS NOT NULL "
            "ORDER BY s.created_at ASC, m.timestamp ASC",
            (project_path,),
        ).fetchall()
    else:
        # Only messages after the last distilled point per session
        rows = conn.execute(
            "SELECT s.id as session_id, m.id as msg_id, m.user_text, m.response_summary, m.timestamp "
            "FROM messages m JOIN sessions s ON m.session_id = s.id "
            "LEFT JOIN distill_progress dp ON dp.session_id = s.id AND dp.project_path = ? "
            "WHERE s.project_path = ? AND m.user_text IS NOT NULL "
            "AND (dp.last_message_id IS NULL OR m.id > dp.last_message_id) "
            "ORDER BY s.created_at ASC, m.timestamp ASC",
            (project_path, project_path),
        ).fetchall()

    if not rows:
        return []

    sessions: dict[str, dict] = {}
    for r in rows:
        sid = r["session_id"]
        if sid not in sessions:
            sessions[sid] = {"session_id": sid, "messages": [], "max_msg_id": 0}
        sessions[sid]["messages"].append({
            "user": r["user_text"],
            "response": (r["response_summary"] or "")[:300],
        })
        sessions[sid]["max_msg_id"] = max(sessions[sid]["max_msg_id"], r["msg_id"])
    return list(sessions.values())


def _load_existing_skills(conn, project_path: str) -> list[dict]:
    """Load existing skills for a project."""
    rows = conn.execute(
        "SELECT id, title, description, steps, tags, obs_type, facts, concepts, files_read "
        "FROM skills WHERE project_path = ?",
        (project_path,),
    ).fetchall()
    return [dict(r) for r in rows]


def _format_conversations_for_prompt(sessions: list[dict]) -> str:
    """Format a batch of sessions into text for the LLM prompt."""
    parts = []
    for sess in sessions:
        lines = [f"--- Session {sess['session_id'][:8]} ---"]
        for msg in sess["messages"][:30]:  # cap messages per session
            lines.append(f"USER: {msg['user'][:200]}")
            if msg["response"]:
                lines.append(f"RESPONSE: {msg['response'][:200]}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _format_existing_skills(skills: list[dict]) -> str:
    """Format existing skills for the LLM to see."""
    if not skills:
        return ""
    parts = []
    for s in skills:
        lines = [
            f"[skill_id={s['id']}] {s['title']} (type={s.get('obs_type', 'workflow')}): {s['description']}",
            f"  Steps: {s['steps'][:200]}",
            f"  Tags: {s['tags']}",
        ]
        facts = s.get("facts", "[]")
        if facts and facts != "[]":
            lines.append(f"  Facts: {facts[:200]}")
        concepts = s.get("concepts", "[]")
        if concepts and concepts != "[]":
            lines.append(f"  Concepts: {concepts[:150]}")
        parts.append("\n".join(lines))
    skills_text = "\n\n".join(parts)
    return EXISTING_SKILLS_HEADER.format(skills_text=skills_text)


def _load_project_rules(conn, project_path: str) -> list[str]:
    """Load learned prompt rules for a project."""
    rows = conn.execute(
        "SELECT rule_text FROM prompt_rules WHERE project_path = ? ORDER BY created_at ASC",
        (project_path,),
    ).fetchall()
    return [r["rule_text"] for r in rows]


def _format_project_rules(rules: list[str]) -> str:
    """Format project-specific rules for inclusion in the distill prompt."""
    if not rules:
        return ""
    numbered = "\n".join(f"  {i}. {rule}" for i, rule in enumerate(rules, 1))
    return PROJECT_RULES_HEADER.format(rules_text=numbered)


# ── Review (self-improving prompt) ─────────────────────────────────────────


def review_skills(
    db_path=None,
    project: str | None = None,
    model: str | None = None,
    dry_run: bool = False,
    on_event=None,
) -> dict:
    """Review distilled skills and generate project-specific prompt rules.

    Feeds the LLM the conversations + extracted skills and asks it to identify
    missed patterns, overly-broad skills, and missing details. The output is
    stored as prompt_rules that are automatically included in future distill runs.

    Returns stats: {projects, rules_added, rules_total}.
    """
    def _emit(kind, data=None):
        if on_event:
            on_event(kind, data or {})

    conn = get_db(db_path)

    # Find projects that have skills
    if project:
        projects = conn.execute(
            "SELECT DISTINCT project_path FROM skills WHERE project_path LIKE ?",
            (f"%{project}%",),
        ).fetchall()
    else:
        projects = conn.execute(
            "SELECT DISTINCT project_path FROM skills WHERE project_path IS NOT NULL"
        ).fetchall()

    project_paths = [r["project_path"] for r in projects]
    stats = {"projects": len(project_paths), "rules_added": 0, "rules_total": 0}

    for proj_path in project_paths:
        skills = _load_existing_skills(conn, proj_path)
        existing_rules = _load_project_rules(conn, proj_path)

        if not skills:
            continue

        # Load a sample of conversations for review context (last 20 sessions)
        all_sessions = _load_project_conversations(conn, proj_path, full=True)
        # Take last 20 sessions to keep prompt reasonable
        review_sessions = all_sessions[-20:] if len(all_sessions) > 20 else all_sessions

        conv_text = _format_conversations_for_prompt(review_sessions)
        skills_text = _format_existing_skills(skills)

        _emit("review_start", {
            "path": proj_path,
            "skills": len(skills),
            "sessions_reviewed": len(review_sessions),
            "existing_rules": len(existing_rules),
        })

        if dry_run:
            _emit("review_done", {"path": proj_path, "new_rules": ["(dry run)"], "skipped": False})
            continue

        prompt = REVIEW_PROMPT.format(
            project=proj_path,
            conversations=conv_text,
            skills_text=skills_text,
        )

        _emit("llm_call", {"prompt_len": len(prompt)})

        try:
            raw = _call_copilot(prompt, model=model)
            rules_list = _parse_skills_json(raw)  # reuse JSON parser — same format
        except Exception as e:
            _emit("review_done", {"path": proj_path, "new_rules": [], "error": str(e)[:200]})
            continue

        # Extract rule texts and deduplicate against existing rules
        new_rules: list[str] = []
        existing_lower = {r.lower().strip() for r in existing_rules}

        for item in rules_list:
            rule_text = item.get("rule", "")
            if not rule_text or not isinstance(rule_text, str):
                continue
            if rule_text.lower().strip() in existing_lower:
                continue
            # Avoid near-duplicates (simple substring check)
            is_dup = any(rule_text.lower()[:50] in existing.lower() for existing in existing_rules)
            if is_dup:
                continue

            conn.execute(
                "INSERT INTO prompt_rules (project_path, rule_text) VALUES (?, ?)",
                (proj_path, rule_text),
            )
            new_rules.append(rule_text)
            existing_rules.append(rule_text)
            existing_lower.add(rule_text.lower().strip())

        conn.commit()
        stats["rules_added"] += len(new_rules)
        stats["rules_total"] += len(existing_rules)

        _emit("review_done", {"path": proj_path, "new_rules": new_rules, "skipped": False})

    conn.close()
    return stats


# ── Main distill entrypoint ────────────────────────────────────────────────


def distill(
    db_path=None,
    project: str | None = None,
    model: str | None = None,
    dry_run: bool = False,
    full: bool = False,
    on_event=None,
) -> dict:
    """Distill skills from conversation history, project by project.

    By default, only processes NEW messages since last distill (incremental).
    Set full=True to re-process everything from scratch.

    on_event(kind, data) is called with progress updates:
      ("project_start", {"path": ..., "sessions": N, "existing_skills": N})
      ("batch_start", {"batch": N, "total": N, "sessions": N})
      ("batch_done", {"skills_new": [...], "skills_updated": [...], "failed": bool})
      ("project_done", {"path": ..., "skills_total": N})
      ("skill_written", {"title": ..., "file": ...})

    Returns stats: {projects, batches, skills_created, skills_updated, skills_failed, messages_processed, messages_skipped}.
    """
    def _emit(kind, data=None):
        if on_event:
            on_event(kind, data or {})

    conn = get_db(db_path)

    # Find all projects (or filter to one)
    if project:
        projects = conn.execute(
            "SELECT DISTINCT project_path FROM sessions "
            "WHERE project_path IS NOT NULL AND project_path LIKE ?",
            (f"%{project}%",),
        ).fetchall()
    else:
        projects = conn.execute(
            "SELECT DISTINCT project_path FROM sessions WHERE project_path IS NOT NULL"
        ).fetchall()

    project_paths = [r["project_path"] for r in projects]

    stats = {
        "projects": len(project_paths),
        "batches": 0,
        "skills_created": 0,
        "skills_updated": 0,
        "skills_failed": 0,
        "messages_processed": 0,
        "messages_skipped": 0,
    }

    for proj_path in project_paths:
        # Count total messages for skip stats
        total_msgs = conn.execute(
            "SELECT COUNT(*) as c FROM messages m JOIN sessions s ON m.session_id = s.id "
            "WHERE s.project_path = ? AND m.user_text IS NOT NULL",
            (proj_path,),
        ).fetchone()["c"]

        sessions = _load_project_conversations(conn, proj_path, full=full)
        if not sessions:
            stats["messages_skipped"] += total_msgs
            _emit("project_start", {
                "path": proj_path, "sessions": 0, "existing_skills": 0,
                "batches": 0, "skipped": True, "total_msgs": total_msgs,
            })
            _emit("project_done", {"path": proj_path, "skills_total": 0, "skipped": True})
            continue

        new_msg_count = sum(len(s["messages"]) for s in sessions)
        stats["messages_processed"] += new_msg_count
        stats["messages_skipped"] += total_msgs - new_msg_count

        existing_skills = _load_existing_skills(conn, proj_path)
        project_rules = _load_project_rules(conn, proj_path)
        total_batches = (len(sessions) + BATCH_SIZE - 1) // BATCH_SIZE

        _emit("project_start", {
            "path": proj_path,
            "sessions": len(sessions),
            "existing_skills": len(existing_skills),
            "batches": total_batches,
            "new_messages": new_msg_count,
            "skipped_messages": total_msgs - new_msg_count,
            "rules": len(project_rules),
        })

        # Process in batches of BATCH_SIZE sessions
        for batch_start in range(0, len(sessions), BATCH_SIZE):
            batch = sessions[batch_start : batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            stats["batches"] += 1

            msg_count = sum(len(s["messages"]) for s in batch)
            _emit("batch_start", {
                "batch": batch_num, "total": total_batches,
                "sessions": len(batch), "messages": msg_count,
            })

            if dry_run:
                stats["skills_created"] += 1  # placeholder count
                _emit("batch_done", {"skills_new": ["(dry run)"], "skills_updated": [], "failed": False})
                continue

            conv_text = _format_conversations_for_prompt(batch)
            existing_text = _format_existing_skills(existing_skills)
            rules_text = _format_project_rules(project_rules)
            prompt = DISTILL_PROMPT.format(
                project=proj_path,
                conversations=conv_text,
                existing_skills_section=existing_text + rules_text,
            )

            _emit("llm_call", {"prompt_len": len(prompt)})

            try:
                raw = _call_copilot(prompt, model=model)
                skill_list = _parse_skills_json(raw)
            except Exception as e:
                stats["skills_failed"] += 1
                _emit("batch_done", {"skills_new": [], "skills_updated": [], "failed": True, "error": str(e)[:200]})
                continue

            if not skill_list:
                stats["skills_failed"] += 1
                _emit("batch_done", {"skills_new": [], "skills_updated": [], "failed": True, "error": "empty response"})
                continue

            batch_new: list[str] = []
            batch_updated: list[str] = []

            for skill_data in skill_list:
                skill_data = _normalize_skill(skill_data)
                if not skill_data.get("title"):
                    continue

                skill_id = skill_data.get("skill_id", "new")
                is_update = skill_id != "new" and any(
                    s["id"] == skill_id for s in existing_skills
                )

                if is_update:
                    # Update existing skill
                    conn.execute(
                        "UPDATE skills SET title=?, description=?, steps=?, tags=?, "
                        "obs_type=?, facts=?, concepts=?, files_read=?, files_modified=?, narrative=? "
                        "WHERE id=?",
                        (
                            skill_data["title"],
                            skill_data.get("description", ""),
                            skill_data.get("steps", ""),
                            skill_data.get("tags", ""),
                            skill_data.get("obs_type", "workflow"),
                            skill_data.get("facts", "[]"),
                            skill_data.get("concepts", "[]"),
                            skill_data.get("files_read", "[]"),
                            skill_data.get("files_modified", "[]"),
                            skill_data.get("narrative", ""),
                            skill_id,
                        ),
                    )
                    stats["skills_updated"] += 1
                    batch_updated.append(skill_data["title"])
                    # Update the in-memory list so next batch sees changes
                    for s in existing_skills:
                        if s["id"] == skill_id:
                            s.update(skill_data)
                            break
                else:
                    # Create new skill
                    new_id = hashlib.sha256(
                        f"{proj_path}:{skill_data['title']}:{skill_data.get('steps','')}".encode()
                    ).hexdigest()[:16]

                    # Skip if duplicate title already exists for this project
                    dup = conn.execute(
                        "SELECT id FROM skills WHERE project_path = ? AND title = ?",
                        (proj_path, skill_data["title"]),
                    ).fetchone()
                    if dup:
                        new_id = dup["id"]
                        conn.execute(
                            "UPDATE skills SET description=?, steps=?, tags=?, "
                            "obs_type=?, facts=?, concepts=?, files_read=?, files_modified=?, narrative=? "
                            "WHERE id=?",
                            (
                                skill_data.get("description", ""),
                                skill_data.get("steps", ""),
                                skill_data.get("tags", ""),
                                skill_data.get("obs_type", "workflow"),
                                skill_data.get("facts", "[]"),
                                skill_data.get("concepts", "[]"),
                                skill_data.get("files_read", "[]"),
                                skill_data.get("files_modified", "[]"),
                                skill_data.get("narrative", ""),
                                new_id,
                            ),
                        )
                        stats["skills_updated"] += 1
                        batch_updated.append(skill_data["title"])
                    else:
                        conn.execute(
                            "INSERT INTO skills (id, title, description, steps, tags, project_path, "
                            "source_session_ids, obs_type, facts, concepts, files_read, files_modified, narrative) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                new_id,
                                skill_data["title"],
                                skill_data.get("description", ""),
                                skill_data.get("steps", ""),
                                skill_data.get("tags", ""),
                                proj_path,
                                json.dumps([s["session_id"] for s in batch]),
                                skill_data.get("obs_type", "workflow"),
                                skill_data.get("facts", "[]"),
                                skill_data.get("concepts", "[]"),
                                skill_data.get("files_read", "[]"),
                                skill_data.get("files_modified", "[]"),
                                skill_data.get("narrative", ""),
                            ),
                        )
                        stats["skills_created"] += 1
                        batch_new.append(skill_data["title"])
                        existing_skills.append({"id": new_id, **skill_data})

                    skill_data["id"] = new_id

                # Write markdown file
                try:
                    md_path = _write_skill_markdown(skill_data, proj_path)
                    _emit("skill_written", {"title": skill_data.get("title", ""), "file": md_path})
                except OSError:
                    pass  # non-fatal

            _emit("batch_done", {"skills_new": batch_new, "skills_updated": batch_updated, "failed": False})

            # Record distill progress so we skip these messages next time
            if not dry_run:
                for sess in batch:
                    conn.execute(
                        "INSERT INTO distill_progress (session_id, project_path, last_message_id) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(session_id, project_path) DO UPDATE SET "
                        "last_message_id = MAX(last_message_id, excluded.last_message_id), "
                        "distilled_at = datetime('now')",
                        (sess["session_id"], proj_path, sess["max_msg_id"]),
                    )
                conn.commit()  # commit after each batch so progress is durable

        _emit("project_done", {"path": proj_path, "skills_total": len(existing_skills)})

    conn.commit()
    conn.close()
    return stats
