"""Live file watcher — auto-ingest, distill, and inject on conversation changes.

Monitors VS Code chatSessions/ directories for new or modified JSON files.
When a change is detected, runs the full pipeline:
  1. Ingest new messages
  2. Distill skills for affected projects (incremental)
  3. Re-inject copilot-instructions.md

Usage: autoskill watch [--debounce 30] [--no-inject]
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Callable

from .ingest import VSCODE_STORAGE

# Minimum seconds between pipeline runs (debounce rapid file writes)
DEFAULT_DEBOUNCE = 30


class _ChangeCollector:
    """Collects file change events and debounces them into pipeline runs."""

    def __init__(
        self,
        debounce: int = DEFAULT_DEBOUNCE,
        on_event: Callable[[str], None] | None = None,
        model: str | None = None,
        inject: bool = True,
        distill: bool = False,
    ):
        self.debounce = debounce
        self.on_event = on_event or (lambda msg: None)
        self.model = model
        self.inject = inject
        self.distill = distill
        self._changed_files: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def file_changed(self, path: str) -> None:
        """Record a changed file and schedule a debounced pipeline run."""
        with self._lock:
            self._changed_files.add(path)
            # Reset debounce timer
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce, self._run_pipeline)
            self._timer.daemon = True
            self._timer.start()

    def _run_pipeline(self) -> None:
        """Execute the ingest → distill → inject pipeline."""
        with self._lock:
            changed = self._changed_files.copy()
            self._changed_files.clear()

        if not changed:
            return

        self.on_event(f"📥 Detected {len(changed)} changed file(s), running pipeline…")

        # Step 1: Ingest
        try:
            from .ingest import ingest
            stats = ingest(quiet=True)
            new_msgs = stats["messages_added"]
            new_sessions = stats["sessions_added"]
            if new_msgs == 0:
                self.on_event("  ⏭ No new messages to ingest")
                return
            self.on_event(
                f"  ✓ Ingested {new_sessions} session(s), {new_msgs} message(s)"
            )
        except Exception as e:
            self.on_event(f"  ✗ Ingest failed: {e}")
            return

        # Step 2: Distill (only if explicitly enabled — costs tokens!)
        if not self.distill:
            self.on_event("  ⏭ Skipping distill (run 'autoskill distill' manually to save tokens)")
        else:
            try:
                from .db import get_db
                conn = get_db()
                projects = conn.execute(
                    "SELECT DISTINCT s.project_path FROM sessions s "
                    "JOIN messages m ON m.session_id = s.id "
                    "WHERE s.project_path IS NOT NULL "
                    "AND m.id NOT IN ("
                    "  SELECT dp.last_message_id FROM distill_progress dp "
                    "  WHERE dp.project_path = s.project_path"
                    ") "
                    "GROUP BY s.project_path"
                ).fetchall()
                conn.close()

                if not projects:
                    self.on_event("  ⏭ All projects up to date")
                else:
                    from .distill import distill
                    for row in projects:
                        proj = row["project_path"]
                        proj_name = Path(proj).name
                        self.on_event(f"  🔄 Distilling {proj_name}…")
                        try:
                            result = distill(project_path=proj, model=self.model)
                            n = result.get("skills_created", 0) + result.get("skills_updated", 0)
                            self.on_event(f"  ✓ {proj_name}: {n} skill(s) created/updated")
                        except Exception as e:
                            self.on_event(f"  ✗ Distill failed for {proj_name}: {e}")
                            continue
            except Exception as e:
                self.on_event(f"  ✗ Pipeline error: {e}")

        # Step 3: Re-inject existing skills (free — just rewrites the md file)
        if self.inject:
            try:
                from .db import get_db
                from .inject import inject as do_inject
                conn = get_db()
                projects = conn.execute(
                    "SELECT DISTINCT project_path FROM skills WHERE project_path IS NOT NULL"
                ).fetchall()
                conn.close()
                for row in projects:
                    proj = row["project_path"]
                    try:
                        result = do_inject(proj)
                        if result["written"]:
                            self.on_event(
                                f"  ✓ Injected {result['skills_count']} skill(s) → {Path(result['path']).name}"
                            )
                    except Exception as e:
                        self.on_event(f"  ✗ Inject failed for {Path(proj).name}: {e}")
            except Exception as e:
                self.on_event(f"  ✗ Inject error: {e}")

        self.on_event("✅ Pipeline complete, watching for changes…")


def watch(
    debounce: int = DEFAULT_DEBOUNCE,
    on_event: Callable[[str], None] | None = None,
    model: str | None = None,
    inject: bool = True,
    distill: bool = False,
) -> None:
    """Start watching chatSessions/ directories for changes. Blocks forever."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    except ImportError:
        raise RuntimeError(
            "watchdog is required for watch mode. Install it with: pip install watchdog"
        )

    collector = _ChangeCollector(
        debounce=debounce, on_event=on_event, model=model, inject=inject, distill=distill
    )

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith(".json"):
                collector.file_changed(event.src_path)

        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith(".json"):
                collector.file_changed(event.src_path)

    storage = VSCODE_STORAGE
    if not storage.is_dir():
        raise RuntimeError(f"VS Code storage not found: {storage}")

    # Watch all chatSessions/ subdirectories
    observer = Observer()
    watch_count = 0
    for ws_dir in storage.iterdir():
        chat_dir = ws_dir / "chatSessions"
        if chat_dir.is_dir():
            observer.schedule(_Handler(), str(chat_dir), recursive=False)
            watch_count += 1

    if watch_count == 0:
        raise RuntimeError("No chatSessions directories found to watch")

    on_event = on_event or (lambda msg: None)
    on_event(f"👁 Watching {watch_count} workspace(s) for changes (debounce: {debounce}s)")
    on_event("Press Ctrl+C to stop.\n")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        on_event("\n🛑 Stopped watching.")
    observer.join()
