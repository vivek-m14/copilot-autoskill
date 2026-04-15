# copilot-autoskill

**Your Copilot already solved that problem last week. `autoskill` remembers so you don't have to.**

A CLI tool that passively learns from your GitHub Copilot chat history in VS Code, distills reusable "skills" via LLM, and serves them back as an **MCP server** — so Copilot and Claude can query your project knowledge live.

```
┌────────────────────────────────────────────────┐
│  VS Code Copilot Chat History (read-only)      │
│  ~/Library/Application Support/Code/User/…     │
└──────────────────┬─────────────────────────────┘
                   │  autoskill ingest
                   ▼
┌────────────────────────────────────────────────┐
│  Local SQLite DB  (~/.autoskill/autoskill.db)  │
│  sessions │ messages │ skills │ paths          │
└──────────────────┬─────────────────────────────┘
                   │
          ┌────────┼────────┬──────────┬──────────┐
          ▼        ▼        ▼          ▼          ▼
      distill   search    paths     inject      serve
      (LLM)    (TF-IDF)  (clipboard) │        (MCP ↕)
          │                           ▼          │
          │              .github/copilot-        │
          │              instructions.md         │
          └──────────────► Copilot / Claude ◄────┘
                          reads skills live
```

---

## Why?

If you use GitHub Copilot daily, you've probably noticed:

- You explain the **same project setup** to Copilot every new session
- You type the **same long file paths** over and over
- You ask Copilot to do the **same task with different inputs** repeatedly
- Each session starts from scratch — **no memory** of past conversations

`autoskill` fixes this. It watches what you've already solved, distills it into structured observations, and serves them back via MCP. The result: **fewer tokens, faster sessions, less repetition**.

---

## Install

```bash
git clone <repo> && cd skill_md
pip install -e .
```

Requires: Python ≥ 3.10, `gh` CLI with Copilot extension, macOS (for VS Code chat history paths and `pbcopy`).

---

## Quick Start

```bash
# 1. Import your VS Code Copilot chat history
autoskill ingest

# 2. Check what was captured
autoskill status

# 3. Distill reusable skills from your conversations
autoskill distill --dry-run          # preview what would be processed
autoskill distill                    # run LLM distillation (uses gh copilot)

# 4. Option A: Inject skills as a static file
autoskill inject -p myproject        # writes .github/copilot-instructions.md

# 4. Option B: Start MCP server (Copilot/Claude queries skills live)
autoskill serve                      # starts stdio MCP server
```

---

## MCP Server — Live Skill Access

The `autoskill serve` command starts an **MCP (Model Context Protocol) server** that exposes your skills as tools. Any MCP-compatible host (VS Code Copilot, Claude Code, etc.) can query your knowledge live.

### Setup in VS Code

Add to `.vscode/settings.json` (or user settings):

```json
{
  "mcp": {
    "servers": {
      "autoskill": {
        "command": "autoskill",
        "args": ["serve"]
      }
    }
  }
}
```

### Setup in Claude Code

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "autoskill": {
      "command": "autoskill",
      "args": ["serve"]
    }
  }
}
```

### Available MCP Tools

| Tool | What it does |
|------|-------------|
| `search_skills` | Semantic search over distilled skills |
| `get_skill` | Full details of a skill by ID |
| `list_project_skills` | All skills for a project |
| `list_projects` | Projects that have skills |
| `search_history` | Keyword search past chat messages |
| `get_frequent_paths` | Frequently used file paths |

When Copilot or Claude encounters a task, it can call `search_skills` first to check if you've solved something similar before — **saving tokens and avoiding re-work**.

---

## Commands

### Core Workflow

| Command | What it does |
|---------|-------------|
| `autoskill ingest` | Scan VS Code chat history, import new sessions & messages |
| `autoskill distill` | Distill reusable skills from conversations via `gh copilot` |
| `autoskill inject` | Generate `.github/copilot-instructions.md` from distilled skills |
| `autoskill serve` | Start MCP server (live skill access for Copilot/Claude) |
| `autoskill status` | Show stats — sessions, messages, skills, top paths |

### Search & Browse

| Command | What it does |
|---------|-------------|
| `autoskill search <query>` | Semantic search over skills & messages (TF-IDF) |
| `autoskill skills` | List all distilled skills |
| `autoskill skill <id>` | Show full details of a specific skill |
| `autoskill history sessions` | List ingested chat sessions |
| `autoskill history show <id>` | Show messages from a specific session |
| `autoskill history grep <pattern>` | Search chat history by keyword |

### Path Clipboard

| Command | What it does |
|---------|-------------|
| `autoskill paths list` | Show frequently-used paths sorted by usage |
| `autoskill paths copy <id\|alias>` | Copy a path to clipboard (`pbcopy`) |
| `autoskill paths alias <id> <name>` | Set a short alias for a tracked path |
| `autoskill paths add <path>` | Manually track a new path |

### Self-Improving Prompt

| Command | What it does |
|---------|-------------|
| `autoskill distill --review` | Distill + review pass to learn prompt rules |
| `autoskill rules list` | Show learned prompt rules |
| `autoskill rules add <rule> -p <proj>` | Manually add a prompt rule |
| `autoskill rules clear` | Clear all learned rules |

---

## Structured Observations

Skills are stored as **structured observations** (inspired by [claude-mem](https://github.com/anthropics/claude-mem)):

```json
{
  "title": "Flatten Directory with Symlinks",
  "type": "workflow",
  "description": "Create a flat directory of symlinks to all images in nested subdirs",
  "steps": "- Find all .jpg/.png files recursively\n- Create symlinks in flat output dir",
  "tags": "image-processing,symlinks,directory-flatten",
  "facts": ["Script uses os.walk + os.symlink", "Output dir must not exist"],
  "concepts": ["data-pipeline", "image-preprocessing"],
  "files": ["scripts/make_flat_dir.py", "/data/images/"]
}
```

Each observation has:
- **type** — `workflow`, `bugfix`, `feature`, `decision`, or `discovery`
- **facts** — discrete factual assertions (what the LLM should remember)
- **concepts** — high-level semantic tags for cross-referencing
- **files** — file/directory paths involved

This structure makes skills more precise and queryable than flat text descriptions.

---

## Distillation

The heart of `autoskill`. The `distill` command sends your conversation history to an LLM in batched chunks, which extracts structured observations.

```bash
autoskill distill                    # incremental (only new messages)
autoskill distill -p engine_clean    # specific project
autoskill distill --pick             # interactive project picker
autoskill distill --dry-run          # preview without calling LLM
autoskill distill --full             # re-process everything
autoskill distill --review           # distill + self-improving review
autoskill distill --status           # show progress per project
autoskill distill --model gpt-4o    # override model
```

### How it works

1. **Project-scoped** — each project is distilled independently
2. **Batched** — 10 sessions per LLM call to stay within context limits
3. **Incremental** — tracks the last distilled message per session; re-runs only process new messages
4. **Smart merge** — existing skills are included in the prompt so the LLM can update them rather than create duplicates
5. **Dual storage** — skills are saved to both SQLite (for search) and markdown files at `~/.autoskill/skills/<project>/`
6. **Self-improving** — `--review` runs a second LLM pass to learn project-specific prompt rules

Default model: `claude-sonnet-4` (override with `--model`).

### Token savings

```
$ autoskill distill -p engine_clean
⏭ Project: ~/projects/engine_clean — all 2997 messages already distilled
💰 Token savings: 2997 messages skipped (already distilled), 0 new messages processed
```

Only new messages since your last distill are sent to the LLM. Use `--full` to force a complete re-process.

---

## Architecture

```
src/autoskill/
├── cli.py         # Click CLI — all commands wired with Rich output
├── db.py          # SQLite schema (sessions, messages, skills, paths, prompt_rules)
├── ingest.py      # VS Code chat history parser
├── distill.py     # Project-scoped incremental distillation via gh copilot
├── inject.py      # Generate copilot-instructions.md from skills
├── search.py      # TF-IDF semantic search (scikit-learn)
├── paths.py       # Path tracking, aliasing, clipboard copy
└── mcp_server.py  # MCP server (FastMCP) — exposes skills as tools
```

### Storage

| What | Where |
|------|-------|
| Database | `~/.autoskill/autoskill.db` (SQLite) |
| Skill markdown files | `~/.autoskill/skills/<project-name>/` |
| Copilot instructions | `<project>/.github/copilot-instructions.md` |

### Dependencies

- **click** — CLI framework
- **rich** — Terminal output formatting
- **scikit-learn** — TF-IDF vectorization for search
- **mcp** — Model Context Protocol server SDK
- **gh copilot** — LLM calls for distillation (uses your existing Copilot subscription)

---

## Requirements

- **Python** ≥ 3.10
- **macOS** (for VS Code chat history paths and `pbcopy`)
- **GitHub Copilot subscription** (for `autoskill distill`)
- **`gh` CLI** with Copilot extension installed (`gh extension install github/gh-copilot`)

---

## License

MIT
