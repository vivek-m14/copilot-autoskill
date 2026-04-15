# copilot-autoskill — Design & Specification

## Problem Statement

Developers using GitHub Copilot in VS Code face a fundamental limitation: **every chat session starts from scratch**. There's no memory of past conversations, no reuse of solved problems, and no way for Copilot to know what you've already figured out. This leads to:

1. **Repeated explanations** — describing the same project setup, conventions, and workflows every session
2. **Wasted tokens** — re-deriving solutions that were already worked out last week
3. **Lost knowledge** — useful solutions buried in chat history that you can't find or reference
4. **Path fatigue** — typing the same deep file paths repeatedly

`copilot-autoskill` solves this by creating a **passive learning loop**:

```
Ingest → Distill → Inject → Copilot reads skills → Better sessions → Repeat
```

## Core Philosophy

**Reduce Copilot token usage by reusing already-distilled knowledge.**

Every LLM call costs tokens. If Copilot already solved "how to run the training pipeline" last week, there's no reason to re-derive it. `autoskill` captures that solution, distills it into a reusable pattern, and injects it back so Copilot starts every session already knowing the answer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  VS Code Copilot Chat History (read-only source)        │
│  ~/Library/Application Support/Code/User/               │
│    workspaceStorage/<hash>/chatSessions/*.json           │
│    workspaceStorage/<hash>/workspace.json → project path │
└───────────────────────┬─────────────────────────────────┘
                        │  autoskill ingest
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Local SQLite DB  (~/.autoskill/autoskill.db)           │
│                                                         │
│  sessions ─┬─ messages ─── distill_progress             │
│            │                                            │
│  paths     └─ skills ──── skill markdown files          │
│                  │         (~/.autoskill/skills/)        │
└──────────────────┼──────────────────────────────────────┘
                   │
      ┌────────────┼────────────┬──────────────┐
      ▼            ▼            ▼              ▼
   distill      search       paths          inject
   (gh copilot)  (TF-IDF)   (clipboard)       │
      │                                        ▼
      │                       .github/copilot-instructions.md
      └──────────────────────► (VS Code Copilot reads automatically)
```

### Module Breakdown

| Module | File | Responsibility |
|--------|------|----------------|
| **CLI** | `cli.py` | Click commands, Rich output, event handling |
| **Database** | `db.py` | SQLite schema, connection management |
| **Ingest** | `ingest.py` | Parse VS Code chat JSON, extract messages & paths |
| **Distill** | `distill.py` | Project-scoped LLM distillation via `gh copilot` |
| **Inject** | `inject.py` | Generate `.github/copilot-instructions.md` from skills |
| **Search** | `search.py` | TF-IDF semantic search over skills & messages |
| **Paths** | `paths.py` | Path tracking, aliasing, clipboard copy |

---

## Data Model

### Input: VS Code Chat Sessions

**Location:** `~/Library/Application Support/Code/User/workspaceStorage/<hash>/`

- `workspace.json` → `{ "folder": "file:///absolute/path/to/project" }`
- `chatSessions/*.json` → session objects with requests array

**Session JSON structure:**
```json
{
  "sessionId": "uuid",
  "creationDate": 1700000000000,
  "lastMessageDate": 1700001000000,
  "mode": { "id": "agent", "kind": "agent" },
  "requests": [
    {
      "message": { "text": "user prompt" },
      "response": [
        { "kind": "markdownContent", "content": { "value": "response text" } },
        { "kind": "thinking", "value": "reasoning..." },
        { "kind": "toolInvocationSerialized", "value": "{...}" },
        { "kind": "textEditGroup", "value": "..." }
      ],
      "timestamp": 1700000500000
    }
  ]
}
```

**Response kind handling:**
- `markdownContent` — direct text response (used in chat mode)
- `thinking` — LLM reasoning (captured as context)
- `toolInvocationSerialized` — agent-mode tool calls with `pastTenseMessage` summaries
- `textEditGroup` — code edits (noted as "code edit applied")
- `inlineReference`, `mcpServersStarting`, `prepareToolInvocation` — metadata, skipped

> **Note:** Agent-mode sessions often have NO `markdownContent`. The useful content is in `toolInvocationSerialized.pastTenseMessage` fields.

### SQLite Schema

**sessions** — one row per VS Code chat session
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | VS Code session UUID |
| workspace_id | TEXT | Workspace directory hash |
| project_path | TEXT | Resolved absolute project path |
| created_at | INTEGER | Epoch ms |
| last_message_at | INTEGER | Epoch ms |
| ingested_at | TEXT | ISO timestamp |

**messages** — one row per request/response pair
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_id | TEXT FK | → sessions.id |
| user_text | TEXT | The user's prompt |
| response_summary | TEXT | Extracted response text |
| timestamp | INTEGER | Epoch ms |
| has_code | BOOLEAN | Contains code blocks |
| has_paths | BOOLEAN | Contains file paths |

**skills** — distilled reusable patterns
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | SHA-256 hash prefix |
| title | TEXT | Short name (≤8 words) |
| description | TEXT | One-sentence summary |
| steps | TEXT | Markdown bullet list of reusable steps |
| tags | TEXT | Comma-separated keywords |
| project_path | TEXT | Which project this belongs to |
| source_session_ids | TEXT | JSON array of contributing session IDs |
| created_at | TEXT | ISO timestamp |

**distill_progress** — incremental distillation watermark
| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT | → sessions.id |
| project_path | TEXT | Project scope |
| last_message_id | INTEGER | Highest message ID already distilled |
| distilled_at | TEXT | When last distilled |
| PK | | (session_id, project_path) |

**paths** — frequently referenced file/directory paths
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| path | TEXT UNIQUE | Absolute path |
| alias | TEXT | User-friendly short name |
| project_path | TEXT | Associated project |
| use_count | INTEGER | Times referenced |
| last_used | TEXT | ISO timestamp |

---

## Distillation Design

### Goals
1. Extract reusable workflow patterns from conversation history
2. Be project-scoped — each project gets its own skill set
3. Be incremental — never re-process already-distilled messages
4. Support smart merge — update existing skills rather than creating duplicates

### LLM Integration

Uses `gh copilot -- -p "<prompt>" --model <model>` in non-interactive mode.

- Default model: `claude-sonnet-4` (good quality-to-cost ratio for summarization)
- Override with `--model` flag
- Timeout: 180 seconds per call
- No separate API key needed — uses existing Copilot subscription

### Prompt Design

The LLM receives:
1. **Project identifier** — which project these conversations belong to
2. **Conversation batch** — user prompts + response summaries (truncated to 200 chars each)
3. **Existing skills** — so the LLM can update rather than duplicate

The LLM returns a JSON array of skill objects with `skill_id` ("new" or existing ID), title, description, steps, and tags.

### Batching Strategy

- **Batch size:** 10 sessions per LLM call
- **Message cap:** 30 messages per session, 200 chars per message
- Sessions ordered by creation date (oldest first)
- Response summaries capped at 300 chars

### Incremental Processing

The `distill_progress` table tracks `(session_id, project_path, last_message_id)`. On re-run:

1. For each session, query messages WHERE `id > last_message_id`
2. Skip sessions with no new messages
3. After successful batch, record the new watermark
4. Progress is committed per-batch (durable on crash)

**Token savings:** A project with 3000 messages that's already been distilled → 0 messages sent on re-run. Only new conversations since last distill are processed.

### Skill Storage

Dual storage for different use cases:
- **SQLite** `skills` table — for search, querying, and programmatic access
- **Markdown files** at `~/.autoskill/skills/<project-slug>/` — for human browsing
- **copilot-instructions.md** in project `.github/` — for Copilot to read automatically

---

## Inject Design

### Purpose

Closes the feedback loop. Without injection, distilled skills sit in SQLite doing nothing. With injection, they become Copilot's custom instructions.

### How VS Code Copilot Uses Instructions

VS Code Copilot reads `.github/copilot-instructions.md` from the workspace root and includes its contents as system-level context in every chat session. This is a built-in VS Code Copilot feature — no extension needed.

### Generated File Format

```markdown
<!-- Auto-generated by copilot-autoskill — do not edit manually.
     Regenerate with: autoskill inject -p <project>
     Last updated: 2025-01-15 14:30 UTC -->

# Project Skills & Patterns

> These are reusable patterns distilled from past Copilot conversations.
> Copilot will use them as context in every chat session for this project.

## Run Training Pipeline

Standard way to launch model training with specific hyperparams.

**Tags:** training,ml,pytorch,pipeline

- cd to <PROJECT_ROOT>/ml_training/engine_clean
- Activate conda env: conda activate ml_env
- Run: python train.py --config <CONFIG_PATH> --epochs <N>
- Monitor with: tensorboard --logdir runs/

---
```

### Workflow

```bash
autoskill inject -p myproject          # generate from skills
# → writes <project>/.github/copilot-instructions.md

autoskill inject --all-projects        # do it for every project
autoskill inject --pick                # interactive picker
autoskill inject -p myproject --dry-run  # preview
```

---

## Search Design

### Approach: TF-IDF + Cosine Similarity

Pure local search — no API calls, no embeddings API.

- scikit-learn's `TfidfVectorizer` (stop_words="english", max_features=5000)
- Builds corpus from skills (title + description + steps) and messages (user_text + response_summary)
- Query vectorized against corpus, ranked by cosine similarity
- Minimum threshold: 0.05
- Filterable by project, type (skills/messages/all)

### Why not vector embeddings?

- Would require an API call per search (defeats the "reduce token usage" goal)
- TF-IDF works well for keyword-heavy technical content
- Zero latency, zero cost, works offline

---

## Implementation Status

### Completed ✅

| Feature | Status |
|---------|--------|
| Project scaffold (pyproject.toml, package structure) | ✅ |
| SQLite schema with all tables | ✅ |
| VS Code chat history ingestion | ✅ |
| Path tracking with aliases and clipboard | ✅ |
| Semantic search via TF-IDF | ✅ |
| Chat history browsing (sessions, show, grep) | ✅ |
| Skill distillation via gh copilot | ✅ |
| Incremental distillation with progress tracking | ✅ |
| Verbose progress output with event callbacks | ✅ |
| Interactive project picker (--pick) | ✅ |
| Inject: generate copilot-instructions.md | ✅ |
| README and design documentation | ✅ |

### Future Possibilities

| Feature | Description |
|---------|-------------|
| MCP server | Expose skills as a local MCP tool for real-time querying |
| Watch mode | Auto-ingest when new sessions appear |
| Auto-inject on distill | Automatically regenerate instructions after distilling |
| Skill quality scoring | Rate/review skills, prune low-quality ones |
| Cross-project skills | Global skills that apply everywhere |
| VS Code extension | UI for browsing skills, triggering distill from editor |
