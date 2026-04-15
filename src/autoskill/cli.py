"""CLI entrypoint for copilot-autoskill."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


@click.group()
@click.version_option(package_name="copilot-autoskill")
def main():
    """autoskill — Learn from your Copilot usage. Never re-solve the same problem."""


# ── ingest ──────────────────────────────────────────────────────────────────


@main.command()
@click.option("--force", is_flag=True, help="Delete all existing sessions/messages and re-ingest from scratch.")
def ingest(force: bool):
    """Scan VS Code Copilot chat history and import new sessions."""
    from .ingest import ingest as do_ingest
    from .db import get_db

    if force:
        console.print("[bold yellow]⚠  --force: deleting all sessions, messages, and distill progress…[/bold yellow]")
        conn = get_db()
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM distill_progress")
        conn.commit()
        conn.close()
        console.print("[dim]Cleared. Re-ingesting…[/dim]")

    with console.status("[bold green]Scanning VS Code chat history…"):
        stats = do_ingest()

    if stats["sessions_added"] == 0:
        console.print("[dim]No new sessions found.[/dim]")
    else:
        console.print(
            f"[green]✓[/green] Ingested [bold]{stats['sessions_added']}[/bold] sessions, "
            f"[bold]{stats['messages_added']}[/bold] messages, "
            f"[bold]{stats['paths_found']}[/bold] path references."
        )


# ── distill ─────────────────────────────────────────────────────────────────


def _show_distill_status(project_filter: str | None = None):
    """Show per-project distillation progress as a rich table."""
    from .db import get_db
    from rich.progress_bar import ProgressBar

    conn = get_db()

    # Get all projects with session/message counts
    query = """
        SELECT
            s.project_path,
            COUNT(DISTINCT s.id) as total_sessions,
            COUNT(m.id) as total_messages,
            (SELECT COUNT(*) FROM skills sk WHERE sk.project_path = s.project_path) as skill_count
        FROM sessions s
        LEFT JOIN messages m ON m.session_id = s.id AND m.user_text IS NOT NULL
        WHERE s.project_path IS NOT NULL
    """
    params: list = []
    if project_filter:
        query += " AND s.project_path LIKE ?"
        params.append(f"%{project_filter}%")
    query += " GROUP BY s.project_path ORDER BY total_messages DESC"

    rows = conn.execute(query, params).fetchall()

    if not rows:
        console.print("[dim]No projects found. Run 'autoskill ingest' first.[/dim]")
        conn.close()
        return

    # For each project, calculate how many messages are already distilled
    home = str(__import__("pathlib").Path.home())

    table = Table(title="Distillation Status", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Project", max_width=45)
    table.add_column("Sessions", width=8, justify="right")
    table.add_column("Messages", width=8, justify="right")
    table.add_column("Distilled", width=9, justify="right")
    table.add_column("Remaining", width=9, justify="right")
    table.add_column("Progress", width=14)
    table.add_column("Skills", width=6, justify="right", style="green")

    total_msgs_all = 0
    total_distilled_all = 0

    for i, r in enumerate(rows, 1):
        proj = r["project_path"]
        total_msgs = r["total_messages"]

        # Count distilled messages for this project
        distilled_count = conn.execute(
            """SELECT COUNT(*) as c FROM messages m
               JOIN sessions s ON m.session_id = s.id
               JOIN distill_progress dp ON dp.session_id = s.id AND dp.project_path = ?
               WHERE s.project_path = ? AND m.user_text IS NOT NULL
               AND m.id <= dp.last_message_id""",
            (proj, proj),
        ).fetchone()["c"]

        remaining = total_msgs - distilled_count
        pct = (distilled_count / total_msgs * 100) if total_msgs > 0 else 0

        total_msgs_all += total_msgs
        total_distilled_all += distilled_count

        # Color the progress
        if pct >= 100:
            pct_str = "[green]██████████[/green] 100%"
        elif pct > 0:
            filled = int(pct / 10)
            empty = 10 - filled
            pct_str = f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim] {pct:.0f}%"
        else:
            pct_str = "[dim]░░░░░░░░░░[/dim]   0%"

        # Color remaining
        if remaining == 0:
            remaining_str = "[green]0[/green]"
        else:
            remaining_str = f"[yellow]{remaining}[/yellow]"

        short_proj = proj.replace(home, "~")
        # Trim to last 2 path components if too long
        if len(short_proj) > 45:
            parts = short_proj.split("/")
            short_proj = "…/" + "/".join(parts[-2:])

        table.add_row(
            str(i),
            short_proj,
            str(r["total_sessions"]),
            str(total_msgs),
            str(distilled_count),
            remaining_str,
            pct_str,
            str(r["skill_count"]),
        )

    conn.close()
    console.print(table)

    # Summary
    overall_pct = (total_distilled_all / total_msgs_all * 100) if total_msgs_all > 0 else 0
    console.print(
        f"\n[bold]Overall:[/bold] {total_distilled_all}/{total_msgs_all} messages distilled "
        f"({overall_pct:.0f}%), {total_msgs_all - total_distilled_all} remaining"
    )
    console.print("[dim]Tip: autoskill distill -p <project> to distill a specific project[/dim]")


@main.command()
@click.option("--project", "-p", default=None, help="Filter to projects matching this substring (e.g. -p engine_clean).")
@click.option("--pick", is_flag=True, help="Interactively pick a project from a numbered list.")
@click.option("--model", "-m", default=None, help="LLM model for distillation (default: claude-sonnet-4.6).")
@click.option("--dry-run", is_flag=True, help="Preview batches and projects without calling the LLM.")
@click.option("--full", is_flag=True, help="Re-distill all messages from scratch, ignoring incremental progress.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output; only show final summary.")
@click.option("--status", "show_status", is_flag=True, help="Show distillation progress per project (messages distilled, remaining, skills) and exit.")
@click.option("--review", is_flag=True, help="After distilling, run a review pass to learn project-specific prompt rules.")
def distill(project, pick, model, dry_run, full, quiet, show_status, review):
    """Distill reusable skills from conversations via gh copilot.

    \b
    Examples:
      autoskill distill                  # distill all projects (incremental)
      autoskill distill -p engine_clean  # distill one project
      autoskill distill --pick           # choose a project interactively
      autoskill distill --dry-run        # preview without calling LLM
      autoskill distill --full           # re-process everything from scratch
      autoskill distill --status         # show progress per project
      autoskill distill --review         # distill + review to improve future prompts
    """
    from .distill import distill as do_distill, review_skills, DEFAULT_MODEL
    from .db import get_db, DEFAULT_DB_DIR
    from pathlib import Path

    if show_status:
        _show_distill_status(project)
        return

    # Interactive project picker
    if pick:
        conn = get_db()
        rows = conn.execute(
            "SELECT project_path, COUNT(*) as sessions, "
            "(SELECT COUNT(*) FROM messages WHERE session_id IN "
            " (SELECT id FROM sessions s2 WHERE s2.project_path = sessions.project_path)) as msgs "
            "FROM sessions WHERE project_path IS NOT NULL "
            "GROUP BY project_path ORDER BY sessions DESC"
        ).fetchall()
        conn.close()

        if not rows:
            console.print("[dim]No projects found. Run 'autoskill ingest' first.[/dim]")
            return

        console.print("\n[bold]Available projects:[/bold]\n")
        home = str(__import__("pathlib").Path.home())
        for i, r in enumerate(rows, 1):
            proj = r["project_path"].replace(home, "~")
            console.print(f"  [cyan]{i:3d}[/cyan]  {r['sessions']:3d} sessions  {r['msgs']:5d} msgs  {proj}")

        console.print()
        choice = click.prompt("Pick a project number", type=int)
        if choice < 1 or choice > len(rows):
            console.print("[red]Invalid choice.[/red]")
            return
        project = rows[choice - 1]["project_path"]
        console.print(f"\n[bold]Selected:[/bold] {project.replace(home, '~')}\n")

    # Progress callback
    def on_event(kind, data):
        if quiet:
            return
        home = str(__import__("pathlib").Path.home())
        if kind == "project_start":
            proj = data["path"].replace(home, "~")
            if data.get("skipped"):
                console.print(
                    f"\n[dim]⏭ Project: {proj} — all {data.get('total_msgs', 0)} messages already distilled[/dim]"
                )
                return
            skipped = data.get("skipped_messages", 0)
            skip_note = f", {skipped} already distilled" if skipped else ""
            rules_note = f", {data.get('rules', 0)} learned rules" if data.get("rules") else ""
            console.print(
                f"\n[bold blue]▶ Project:[/bold blue] {proj}\n"
                f"  [dim]{data['sessions']} sessions with new content, "
                f"{data['new_messages']} new messages{skip_note}, "
                f"{data['existing_skills']} existing skills, "
                f"{data['batches']} batch(es){rules_note}[/dim]"
            )
        elif kind == "batch_start":
            console.print(
                f"  [cyan]⧗[/cyan] Batch {data['batch']}/{data['total']} "
                f"({data['sessions']} sessions, {data['messages']} messages)"
            )
        elif kind == "llm_call":
            console.print(f"    [dim]→ Calling gh copilot · {model or DEFAULT_MODEL} ({data['prompt_len']:,} chars)…[/dim]")
        elif kind == "batch_done":
            if data.get("failed"):
                console.print(f"    [red]✗ Failed:[/red] {data.get('error', 'unknown')}")
            else:
                new = data.get("skills_new", [])
                updated = data.get("skills_updated", [])
                if new:
                    for title in new:
                        console.print(f"    [green]+ NEW:[/green] {title}")
                if updated:
                    for title in updated:
                        console.print(f"    [yellow]↻ UPDATED:[/yellow] {title}")
                if not new and not updated:
                    console.print(f"    [dim]  (no skills extracted)[/dim]")
        elif kind == "skill_written":
            console.print(f"    [dim]  → {data['file']}[/dim]")
        elif kind == "project_done":
            if not data.get("skipped"):
                console.print(
                    f"  [green]✓[/green] Done — {data['skills_total']} total skills for this project"
                )

    if not quiet and not full:
        console.print("[dim]Running in incremental mode (only new messages). Use --full to re-process all.[/dim]")

    stats = do_distill(project=project, model=model, dry_run=dry_run, full=full, on_event=on_event)

    # Final summary
    console.print()
    parts = [
        f"[bold]Summary:[/bold] {stats['projects']} project(s), {stats['batches']} batch(es) →",
        f"[green]{stats['skills_created']}[/green] created,",
        f"[yellow]{stats['skills_updated']}[/yellow] updated",
    ]
    if stats["skills_failed"]:
        parts.append(f", [red]{stats['skills_failed']}[/red] failed")
    if dry_run:
        parts.append("[dim](dry run)[/dim]")
    console.print(" ".join(parts))

    # Show token savings
    processed = stats.get("messages_processed", 0)
    skipped = stats.get("messages_skipped", 0)
    if skipped > 0:
        console.print(
            f"[dim]💰 Token savings: {skipped} messages skipped (already distilled), "
            f"{processed} new messages processed[/dim]"
        )

    if not dry_run and (stats["skills_created"] or stats["skills_updated"]):
        console.print(f"[dim]Skills saved to: {DEFAULT_DB_DIR / 'skills'}/[/dim]")

    # ── Review pass (self-improving prompt) ──
    if review and (stats["skills_created"] or stats["skills_updated"] or dry_run):
        console.print("\n[bold magenta]🔍 Running review pass…[/bold magenta]")

        def on_review_event(kind, data):
            if quiet:
                return
            home = str(Path.home())
            if kind == "review_start":
                proj = data["path"].replace(home, "~")
                console.print(
                    f"  [magenta]▶[/magenta] Reviewing {proj} — "
                    f"{data['skills']} skills, {data['sessions_reviewed']} sessions, "
                    f"{data['existing_rules']} existing rules"
                )
            elif kind == "llm_call":
                console.print(f"    [dim]→ Calling gh copilot · {model or DEFAULT_MODEL} ({data['prompt_len']:,} chars)…[/dim]")
            elif kind == "review_done":
                new_rules = data.get("new_rules", [])
                if data.get("error"):
                    console.print(f"    [red]✗ Review failed:[/red] {data['error']}")
                elif new_rules:
                    for rule in new_rules:
                        console.print(f"    [magenta]+ RULE:[/magenta] {rule}")
                else:
                    console.print(f"    [dim]  (no new rules needed)[/dim]")

        review_stats = review_skills(
            project=project, model=model, dry_run=dry_run, on_event=on_review_event
        )
        console.print(
            f"\n[bold]Review:[/bold] {review_stats['rules_added']} new rules learned, "
            f"{review_stats['rules_total']} total rules"
        )
        if review_stats["rules_added"] and not dry_run:
            console.print("[dim]Tip: Rules will be used automatically in future distill runs. "
                          "View them with: autoskill rules[/dim]")


# ── rules ───────────────────────────────────────────────────────────────────


@main.group()
def rules():
    """View and manage learned prompt rules for distillation."""


@rules.command("list")
@click.option("--project", "-p", default=None, help="Filter by project.")
def rules_list(project):
    """Show all learned prompt rules."""
    from .db import get_db

    conn = get_db()
    if project:
        rows = conn.execute(
            "SELECT project_path, rule_text, source, created_at FROM prompt_rules "
            "WHERE project_path LIKE ? ORDER BY created_at ASC",
            (f"%{project}%",),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT project_path, rule_text, source, created_at FROM prompt_rules "
            "ORDER BY project_path, created_at ASC"
        ).fetchall()
    conn.close()

    if not rows:
        console.print("[dim]No rules yet. Run 'autoskill distill --review' to learn project-specific rules.[/dim]")
        return

    home = str(__import__("pathlib").Path.home())
    current_project = None
    for r in rows:
        proj = r["project_path"].replace(home, "~")
        if proj != current_project:
            current_project = proj
            console.print(f"\n[bold blue]{proj}[/bold blue]")
        console.print(f"  [magenta]•[/magenta] {r['rule_text']}")
        console.print(f"    [dim]({r['source']}, {r['created_at']})[/dim]")


@rules.command("add")
@click.argument("rule")
@click.option("--project", "-p", required=True, help="Project path substring.")
def rules_add(rule, project):
    """Manually add a prompt rule for a project."""
    from .db import get_db

    conn = get_db()
    # Resolve the actual project path
    row = conn.execute(
        "SELECT DISTINCT project_path FROM sessions WHERE project_path LIKE ?",
        (f"%{project}%",),
    ).fetchone()

    if not row:
        console.print(f"[red]✗[/red] No project found matching '{project}'.")
        conn.close()
        return

    conn.execute(
        "INSERT INTO prompt_rules (project_path, rule_text, source) VALUES (?, ?, 'manual')",
        (row["project_path"], rule),
    )
    conn.commit()
    conn.close()

    home = str(__import__("pathlib").Path.home())
    console.print(f"[green]✓[/green] Rule added for {row['project_path'].replace(home, '~')}")


@rules.command("clear")
@click.option("--project", "-p", default=None, help="Clear rules for a specific project only.")
@click.confirmation_option(prompt="Are you sure you want to clear rules?")
def rules_clear(project):
    """Clear all learned rules (or just for one project)."""
    from .db import get_db

    conn = get_db()
    if project:
        conn.execute("DELETE FROM prompt_rules WHERE project_path LIKE ?", (f"%{project}%",))
    else:
        conn.execute("DELETE FROM prompt_rules")
    deleted = conn.total_changes
    conn.commit()
    conn.close()
    console.print(f"[green]✓[/green] Cleared {deleted} rule(s).")


# ── search ──────────────────────────────────────────────────────────────────


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--project", "-p", default=None, help="Filter by project path substring.")
@click.option("--type", "include", type=click.Choice(["all", "skills", "messages"]), default="all")
@click.option("--limit", "-n", default=10, help="Max results.")
def search(query, project, include, limit):
    """Semantic search over skills and past messages."""
    from .search import search as do_search

    query_str = " ".join(query)
    results = do_search(query_str, project=project, include=include, limit=limit)

    if not results:
        console.print("[dim]No matches found.[/dim]")
        return

    table = Table(title=f"Results for '{query_str}'", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Type", width=7)
    table.add_column("Score", width=6)
    table.add_column("Title", max_width=50)
    table.add_column("Description", max_width=60)

    for i, r in enumerate(results, 1):
        type_style = "cyan" if r["type"] == "skill" else "dim"
        table.add_row(
            str(i),
            Text(r["type"], style=type_style),
            f"{r['score']:.2f}",
            r["title"][:50],
            (r["description"] or "")[:60],
        )
    console.print(table)


# ── history ──────────────────────────────────────────────────────────────────


@main.group()
def history():
    """Browse ingested Copilot chat history — sessions and messages."""


@history.command("sessions")
@click.option("--project", "-p", default=None, help="Filter by project path substring.")
@click.option("--limit", "-n", default=20, help="Max sessions to show.")
def history_sessions(project, limit):
    """List ingested chat sessions."""
    from .db import get_db

    conn = get_db()
    query = (
        "SELECT id, project_path, created_at, last_message_at, "
        "(SELECT COUNT(*) FROM messages WHERE session_id = sessions.id) as msg_count "
        "FROM sessions"
    )
    params: list = []
    if project:
        query += " WHERE project_path LIKE ?"
        params.append(f"%{project}%")
    query += " ORDER BY last_message_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        console.print("[dim]No sessions found. Run 'autoskill ingest' first.[/dim]")
        return

    table = Table(title="Chat Sessions", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Session ID", width=12)
    table.add_column("Messages", width=8, justify="right")
    table.add_column("Project", max_width=60)
    table.add_column("Date", width=12)

    from datetime import datetime, timezone

    for i, r in enumerate(rows, 1):
        ts = r["last_message_at"]
        date_str = ""
        if ts:
            try:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d")
            except (OSError, ValueError):
                date_str = str(ts)
        proj = r["project_path"] or ""
        # Shorten home dir
        proj = proj.replace(str(__import__("pathlib").Path.home()), "~")
        table.add_row(
            str(i),
            r["id"][:12],
            str(r["msg_count"]),
            proj,
            date_str,
        )
    console.print(table)
    console.print("[dim]Tip: autoskill history show <session-id> to see messages[/dim]")


@history.command("show")
@click.argument("session_id")
@click.option("--limit", "-n", default=50, help="Max messages to show.")
def history_show(session_id, limit):
    """Show messages from a specific chat session."""
    from .db import get_db
    from rich.markdown import Markdown

    conn = get_db()

    # Match by prefix
    session = conn.execute(
        "SELECT id, project_path, created_at FROM sessions WHERE id = ? OR id LIKE ?",
        (session_id, f"{session_id}%"),
    ).fetchone()

    if not session:
        console.print(f"[red]✗[/red] Session not found: {session_id}")
        raise SystemExit(1)

    rows = conn.execute(
        "SELECT user_text, response_summary, timestamp FROM messages "
        "WHERE session_id = ? ORDER BY timestamp ASC, id ASC LIMIT ?",
        (session["id"], limit),
    ).fetchall()
    conn.close()

    proj = session["project_path"] or "unknown"
    proj = proj.replace(str(__import__("pathlib").Path.home()), "~")
    console.print(Panel(f"[bold]Session:[/bold] {session['id']}\n[bold]Project:[/bold] {proj}", border_style="blue"))

    if not rows:
        console.print("[dim]No messages in this session.[/dim]")
        return

    for r in rows:
        # User message
        console.print(f"\n[bold blue]You:[/bold blue] {r['user_text']}")
        # Response
        resp = r["response_summary"]
        if resp:
            trimmed = resp[:500] + ("…" if len(resp) > 500 else "")
            console.print(f"[bold green]Copilot:[/bold green] {trimmed}")
        else:
            console.print("[dim]  (no text response captured)[/dim]")
        console.rule(style="dim")


@history.command("grep")
@click.argument("pattern")
@click.option("--project", "-p", default=None, help="Filter by project.")
@click.option("--limit", "-n", default=20, help="Max results.")
def history_grep(pattern, project, limit):
    """Search chat history by keyword (exact substring match)."""
    from .db import get_db

    conn = get_db()
    query = (
        "SELECT m.user_text, m.response_summary, s.project_path, m.timestamp "
        "FROM messages m JOIN sessions s ON m.session_id = s.id "
        "WHERE (m.user_text LIKE ? OR m.response_summary LIKE ?)"
    )
    params: list = [f"%{pattern}%", f"%{pattern}%"]
    if project:
        query += " AND s.project_path LIKE ?"
        params.append(f"%{project}%")
    query += " ORDER BY m.timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        console.print(f"[dim]No matches for '{pattern}'.[/dim]")
        return

    table = Table(title=f"History matching '{pattern}'", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Prompt", max_width=50)
    table.add_column("Response", max_width=50)
    table.add_column("Project", max_width=30)

    for i, r in enumerate(rows, 1):
        proj = (r["project_path"] or "").replace(str(__import__("pathlib").Path.home()), "~")
        table.add_row(
            str(i),
            (r["user_text"] or "")[:50],
            (r["response_summary"] or "")[:50],
            proj.split("/")[-1] if proj else "",
        )
    console.print(table)


# ── paths ───────────────────────────────────────────────────────────────────


@main.group()
def paths():
    """Manage frequently-used paths — list, alias, copy to clipboard."""


@paths.command("list")
@click.option("--project", "-p", default=None, help="Filter by project.")
@click.option("--limit", "-n", default=20, help="Max paths to show.")
def paths_list(project, limit):
    """Show frequently-used paths sorted by usage count."""
    from .paths import list_paths

    items = list_paths(project=project, limit=limit)
    if not items:
        console.print("[dim]No paths tracked yet. Run 'autoskill ingest' first.[/dim]")
        return

    table = Table(title="Frequent Paths", show_lines=False)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Uses", width=5, justify="right")
    table.add_column("Alias", style="cyan", width=15)
    table.add_column("Path", max_width=80)

    for p in items:
        table.add_row(
            str(p["id"]),
            str(p["use_count"]),
            p["alias"] or "",
            p["path"],
        )
    console.print(table)
    console.print("[dim]Tip: autoskill paths copy <id-or-alias> to copy a path[/dim]")


@paths.command("copy")
@click.argument("id_or_alias")
def paths_copy(id_or_alias):
    """Copy a path to clipboard by ID or alias."""
    from .paths import resolve_path, copy_to_clipboard

    path = resolve_path(id_or_alias)
    if not path:
        console.print(f"[red]✗[/red] No path found for '{id_or_alias}'.")
        raise SystemExit(1)

    if copy_to_clipboard(path):
        console.print(f"[green]✓[/green] Copied to clipboard: [bold]{path}[/bold]")
    else:
        console.print(f"[yellow]⚠[/yellow] pbcopy not available. Path: {path}")


@paths.command("alias")
@click.argument("path_or_id")
@click.argument("alias")
def paths_alias(path_or_id, alias):
    """Set a short alias for a tracked path."""
    from .paths import set_alias

    if set_alias(path_or_id, alias):
        console.print(f"[green]✓[/green] Alias '{alias}' set.")
    else:
        console.print(f"[red]✗[/red] Path not found: {path_or_id}")
        raise SystemExit(1)


@paths.command("add")
@click.argument("path")
@click.option("--alias", "-a", default=None, help="Optional short alias.")
@click.option("--project", "-p", default=None, help="Associate with project.")
def paths_add(path, alias, project):
    """Manually add a path to tracking."""
    from .paths import add_path

    add_path(path, alias=alias, project=project)
    console.print(f"[green]✓[/green] Tracking: {path}" + (f" (alias: {alias})" if alias else ""))


# ── skills ──────────────────────────────────────────────────────────────────


@main.command("skills")
@click.option("--project", "-p", default=None, help="Filter by project.")
def skills_list(project):
    """List all distilled skills."""
    from .db import get_db

    conn = get_db()
    query = "SELECT id, title, description, tags, project_path, created_at FROM skills"
    params: list = []
    if project:
        query += " WHERE project_path LIKE ?"
        params.append(f"%{project}%")
    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        console.print("[dim]No skills yet. Run 'autoskill distill' after ingesting.[/dim]")
        return

    table = Table(title="Distilled Skills", show_lines=True)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", max_width=40)
    table.add_column("Tags", style="cyan", max_width=30)
    table.add_column("Description", max_width=50)

    for r in rows:
        table.add_row(r["id"][:10], r["title"], r["tags"], (r["description"] or "")[:50])
    console.print(table)


@main.command("skill")
@click.argument("skill_id")
def skill_show(skill_id):
    """Show details of a specific skill."""
    from .db import get_db
    import json as _json

    conn = get_db()
    row = conn.execute(
        "SELECT * FROM skills WHERE id = ? OR id LIKE ?",
        (skill_id, f"{skill_id}%"),
    ).fetchone()
    conn.close()

    if not row:
        console.print(f"[red]✗[/red] Skill not found: {skill_id}")
        raise SystemExit(1)

    def _fmt(val):
        if not val or val == "[]":
            return ""
        try:
            items = _json.loads(val)
            if isinstance(items, list):
                return ", ".join(str(i) for i in items)
        except (ValueError, TypeError):
            pass
        return str(val)

    obs_type = row["obs_type"] or "workflow"
    facts = _fmt(row["facts"])
    concepts = _fmt(row["concepts"])
    files_read = _fmt(row["files_read"])
    files_modified = _fmt(row["files_modified"])

    panel_text = (
        f"[bold]{row['title']}[/bold]\n\n"
        f"{row['description']}\n\n"
        f"[cyan]Type:[/cyan] {obs_type}\n"
        f"[cyan]Tags:[/cyan] {row['tags']}\n"
        f"[cyan]Project:[/cyan] {row['project_path'] or 'global'}\n"
    )
    if facts:
        panel_text += f"\n[bold]Facts:[/bold] {facts}\n"
    if concepts:
        panel_text += f"[bold]Concepts:[/bold] {concepts}\n"
    if files_read:
        panel_text += f"[bold]Files read:[/bold] {files_read}\n"
    if files_modified:
        panel_text += f"[bold]Files modified:[/bold] {files_modified}\n"
    panel_text += f"\n[bold]Steps:[/bold]\n{row['steps']}"
    if row["narrative"]:
        panel_text += f"\n\n[bold]Context:[/bold]\n{row['narrative']}"

    console.print(Panel(panel_text, title=f"Skill {row['id'][:10]}", border_style="green"))


# ── inject ──────────────────────────────────────────────────────────────────


@main.command()
@click.option("--project", "-p", default=None, help="Filter by project path substring.")
@click.option("--pick", is_flag=True, help="Interactively pick a project.")
@click.option("--output", "-o", default=None, help="Override output file path.")
@click.option("--dry-run", is_flag=True, help="Preview the generated file without writing.")
@click.option("--all-projects", is_flag=True, help="Inject for every project that has skills.")
def inject(project, pick, output, dry_run, all_projects):
    """Generate .github/copilot-instructions.md from distilled skills.

    Closes the loop: Copilot reads this file automatically, so your distilled
    skills become context in every future session — saving tokens and re-work.
    """
    from .inject import inject as do_inject, load_project_skills, render_instructions
    from .db import get_db
    from pathlib import Path

    conn = get_db()

    # Build project list
    if pick:
        rows = conn.execute(
            "SELECT project_path, COUNT(*) as skill_count FROM skills "
            "GROUP BY project_path ORDER BY skill_count DESC"
        ).fetchall()

        if not rows:
            console.print("[dim]No skills found. Run 'autoskill distill' first.[/dim]")
            return

        console.print("\n[bold]Projects with skills:[/bold]\n")
        home = str(__import__("pathlib").Path.home())
        for i, r in enumerate(rows, 1):
            proj = r["project_path"].replace(home, "~")
            console.print(f"  [cyan]{i:3d}[/cyan]  {r['skill_count']:3d} skills  {proj}")
        console.print()

        choice = click.prompt("Pick a project number", type=int)
        if choice < 1 or choice > len(rows):
            console.print("[red]Invalid choice.[/red]")
            return
        project_paths = [rows[choice - 1]["project_path"]]
    elif all_projects:
        rows = conn.execute(
            "SELECT DISTINCT project_path FROM skills WHERE project_path IS NOT NULL"
        ).fetchall()
        project_paths = [r["project_path"] for r in rows]
    elif project:
        rows = conn.execute(
            "SELECT DISTINCT project_path FROM skills WHERE project_path LIKE ?",
            (f"%{project}%",),
        ).fetchall()
        project_paths = [r["project_path"] for r in rows]
    else:
        console.print("[red]Specify --project/-p, --pick, or --all-projects.[/red]")
        conn.close()
        return

    conn.close()

    if not project_paths:
        console.print("[dim]No matching projects with skills found.[/dim]")
        return

    home = str(Path.home())
    total_skills = 0

    for proj_path in project_paths:
        short = proj_path.replace(home, "~")

        if dry_run:
            skills = load_project_skills(proj_path)
            if not skills:
                console.print(f"[dim]⏭ {short} — no skills[/dim]")
                continue
            content = render_instructions(skills, proj_path)
            out_target = output or str(Path(proj_path) / ".github" / "copilot-instructions.md")
            console.print(f"\n[bold blue]▶ {short}[/bold blue] → {out_target}")
            console.print(f"  {len(skills)} skills would be written\n")
            console.print(Panel(content[:2000] + ("…" if len(content) > 2000 else ""),
                                title="Preview", border_style="dim"))
            total_skills += len(skills)
        else:
            result = do_inject(proj_path, output=output)
            if result["written"]:
                console.print(
                    f"[green]✓[/green] {short} → [bold]{result['path']}[/bold] "
                    f"({result['skills_count']} skills)"
                )
                total_skills += result["skills_count"]
            else:
                console.print(f"[dim]⏭ {short} — no skills[/dim]")

    if total_skills and not dry_run:
        console.print(
            f"\n[bold green]Done![/bold green] Copilot will now read your skills automatically "
            f"in VS Code sessions."
        )
    elif dry_run and total_skills:
        console.print(f"\n[dim]Dry run — no files written. Remove --dry-run to inject.[/dim]")


# ── status ──────────────────────────────────────────────────────────────────


@main.command()
def status():
    """Show autoskill stats — sessions, messages, skills, top paths."""
    from .db import get_db

    conn = get_db()
    sessions = conn.execute("SELECT COUNT(*) as c FROM sessions").fetchone()["c"]
    messages = conn.execute("SELECT COUNT(*) as c FROM messages").fetchone()["c"]
    skills = conn.execute("SELECT COUNT(*) as c FROM skills").fetchone()["c"]
    path_count = conn.execute("SELECT COUNT(*) as c FROM paths").fetchone()["c"]

    top_paths = conn.execute(
        "SELECT path, alias, use_count FROM paths ORDER BY use_count DESC LIMIT 5"
    ).fetchall()

    projects = conn.execute(
        "SELECT project_path, COUNT(*) as c FROM sessions "
        "WHERE project_path IS NOT NULL GROUP BY project_path ORDER BY c DESC LIMIT 5"
    ).fetchall()
    conn.close()

    console.print(Panel(
        f"[bold]Sessions:[/bold]  {sessions}\n"
        f"[bold]Messages:[/bold]  {messages}\n"
        f"[bold]Skills:[/bold]    {skills}\n"
        f"[bold]Paths:[/bold]     {path_count}",
        title="autoskill status", border_style="blue",
    ))

    if projects:
        console.print("\n[bold]Top Projects:[/bold]")
        for p in projects:
            console.print(f"  {p['c']:4d} sessions  {p['project_path']}")

    if top_paths:
        console.print("\n[bold]Top Paths:[/bold]")
        for p in top_paths:
            alias = f" ({p['alias']})" if p["alias"] else ""
            console.print(f"  {p['use_count']:4d}×  {p['path']}{alias}")


# ── serve (MCP server) ─────────────────────────────────────────────────────


@main.command()
def serve():
    """Start the autoskill MCP server (stdio transport).

    \b
    This exposes your skills as MCP tools that VS Code Copilot, Claude,
    or any MCP-compatible host can query live.

    \b
    VS Code settings.json:
      "mcp": {
        "servers": {
          "autoskill": {
            "command": "autoskill",
            "args": ["serve"]
          }
        }
      }

    \b
    Claude Code (~/.claude/claude_desktop_config.json):
      "mcpServers": {
        "autoskill": {
          "command": "autoskill",
          "args": ["serve"]
        }
      }
    """
    from .mcp_server import run_server

    run_server()
