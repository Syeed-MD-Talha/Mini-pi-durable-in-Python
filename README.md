# Mini `pi-durable` in Python

A clean, beginner-friendly Python implementation of the core architecture behind [`@earendil-works/pi-durable`](https://github.com/earendil-works/pi/tree/main/packages/durable), backed by SQLite WAL transactions and any OpenAI-compatible API.

Instead of keeping an AI agent loop in RAM (where a crash or restart loses all progress), every message, tool call, document mutation, and task checkpoint is committed atomically to SQLite **before** anything is shown or treated as complete.

---

## Project Structure

```text
.
├── mini_pi_durable/
│   ├── __init__.py          # Public exports
│   ├── core.py              # SqliteStorage, Tx, Entry, TaskRuntime, Harness, Registry
│   └── llm.py               # OpenAICompatibleLLM (openai SDK) + FauxLLM (offline mock)
├── examples/
│   ├── 01_chat_and_tools.py # Basic prompt -> tool call -> answer + pi.usage
│   ├── 02_crash_and_resume.py # Crash recovery, memo(), and request_id idempotency
│   ├── 03_sleep_and_signals.py # Parking a tool with sleep + external signal() wakeup
│   └── 04_busy_inbox_steer.py # steer vs follow_up vs reject with pi.inbox
├── main.py                  # Interactive & CLI durable agent using OpenAI-compatible APIs
├── requirements.txt         # openai, python-dotenv
├── .env.example             # Safe environment variable template
└── .gitignore
```

---

## Quick Start

### 1. Create a virtual environment & install dependencies

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the offline learning examples (no API key needed)

These four demos use `FauxLLM` so you can watch how durable execution works step by step:

```bash
python examples/01_chat_and_tools.py
python examples/02_crash_and_resume.py
python examples/03_sleep_and_signals.py
python examples/04_busy_inbox_steer.py
```

### 3. Run the interactive agent (`main.py`) with any OpenAI-compatible API

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

> **Using another OpenAI-compatible provider?** Just change `OPENAI_BASE_URL` and `OPENAI_MODEL` in `.env`—for example:
> - **DeepSeek:** `OPENAI_BASE_URL=https://api.deepseek.com/v1`, `OPENAI_MODEL=deepseek-chat`
> - **OpenRouter:** `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, `OPENAI_MODEL=openai/gpt-4o-mini`
> - **Local Ollama:** `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_MODEL=llama3.2`

Start the interactive chat:

```bash
python main.py
```

Or run a single prompt from the command line:

```bash
python main.py "List the files in this project and save a short summary note"
```

#### Interactive Commands

| Command | Description |
|---|---|
| `/entries` | Print all immutable transcript rows (`pi.user`, `pi.assistant`, `pi.tool-result`, `pi.reset`) in `session.sqlite` |
| `/docs` | Print built-in (`pi.usage`, `pi.live`, `pi.inbox`) and custom (`app.notes`) JSON documents |
| `/tasks` | Print all durable task records and checkpoints (`pi.generation`, `pi.tool`) |
| `/reset` | Append a `pi.reset` marker (clears model context while keeping full history in SQLite) |
| `/quit` | Exit (run `python main.py` again later to continue from the same SQLite session) |

---

## Core Concepts (Mapped from `@earendil-works/pi-durable`)

| Concept | Role in `mini_pi_durable` |
|---|---|
| **`SqliteStorage` & `Tx`** | Single-writer atomic SQLite (WAL mode) commits—transcript entries, documents, and task state commit together or roll back together. |
| **`Entry`** | Immutable append-only transcript rows (`pi.user`, `pi.assistant`, `pi.tool-result`, `pi.reset`). |
| **`Doc`** | Mutable JSON state stored alongside the transcript (`pi.live`, `pi.inbox`, `pi.usage`, `pi.agent`, or custom app docs like `app.notes`). |
| **`TaskDef` & `TaskRuntime`** | Durable state machines split into `phases`. Supports `memo()` for idempotent side effects, `sleep_until()`, `wait_for_tasks()`, and `restart_limit` crash-loop protection. |
| **`Harness.signal()`** | Wakes a parked (`sleeping`) task from outside (e.g., human approval or webhook) even across process restarts. |
| **`Submission` & `request_id`** | Idempotent input admission; retrying the same `request_id` after a crash returns the existing submission instead of duplicating work. |
| **`when_busy` (`steer` / `follow_up`)** | Queues messages in `pi.inbox` while a run is active: `steer` injects at the next tool boundary, `follow_up` starts the next run after completion. |

---

## How One Prompt Flows Under the Hood

```text
root.submit(prompt)
  └─► commits `pi.user` entry + creates `pi.generation` task
        ├─► calls LLM (OpenAICompatibleLLM / FauxLLM)
        ├─► commits `pi.assistant` entry + updates `pi.usage`
        ├─► if tool calls exist:
        │     ├─► spawns `pi.tool` child tasks & sets parent to `waiting`
        │     ├─► runs `before_tool` hooks (can block or park on a signal)
        │     └─► commits `pi.tool-result` entries -> checks `steer` inbox -> next LLM turn
        └─► on final answer:
              └─► marks submission `done` -> pops next `follow_up` from `pi.inbox` if any
```

---

## Database Growth Note

Like the original `pi-durable`, transcript `entries` and `tasks` are **append-only** so audit history is never lost (even `/reset` only appends a `pi.reset` marker). For production workloads:
- Use **one SQLite database per session/user/workspace** so old sessions can be archived or deleted as individual files.
- Or run an external retention cleanup job + `VACUUM` on inactive conversations.

