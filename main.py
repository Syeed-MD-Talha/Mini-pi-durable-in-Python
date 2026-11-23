"""
Interactive & CLI Durable Agent powered by any OpenAI-compatible API + SQLite
=============================================================================
1. Copy `.env.example` to `.env` and set your `OPENAI_API_KEY`
   (and optionally `OPENAI_BASE_URL` and `OPENAI_MODEL`).
2. Run interactively:
       python main.py
   Or pass a one-shot prompt:
       python main.py "List the files in this project and save a note summarizing it"

Commands inside interactive mode:
  /entries  -> print all immutable transcript entries stored in session.sqlite
  /docs     -> print pi.usage, pi.live, pi.inbox, and custom app.notes documents
  /tasks    -> print all task checkpoints stored in SQLite
  /reset    -> append a pi.reset entry (model forgets older turns, DB keeps everything)
  /quit     -> exit (re-run later and the agent resumes from session.sqlite)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from mini_pi_durable import (
    Extension,
    Harness,
    OpenAICompatibleLLM,
    Registry,
    SqliteStorage,
    ToolApi,
    ToolDef,
    Tx,
)

WORKSPACE = Path(__file__).parent.resolve()
DB_PATH = WORKSPACE / "session.sqlite"
PLACEHOLDER_KEYS = {"", "your_openai_api_key_here", "your-api-key-here", "sk-your-key-here"}


def build_registry() -> Registry:
    def tool_list_files(_args: dict, _api: ToolApi) -> str:
        items: list[str] = []
        for p in sorted(WORKSPACE.rglob("*")):
            rel = p.relative_to(WORKSPACE)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            if p.is_file() and not p.name.endswith((".sqlite", ".sqlite-wal", ".sqlite-shm")):
                items.append(rel.as_posix())
        return "\n".join(items)

    def tool_read_file(args: dict, _api: ToolApi) -> str:
        target = (WORKSPACE / args["filename"]).resolve()
        try:
            rel = target.relative_to(WORKSPACE)
        except ValueError:
            return f"Cannot read outside workspace: {args['filename']}"
        if not target.is_file() or any(part.startswith(".") for part in rel.parts):
            return f"Cannot read {args['filename']}"
        return target.read_text(encoding="utf-8")[:4000]

    def tool_save_note(args: dict, api: ToolApi) -> str:
        def _write(tx: Tx) -> int:
            doc = tx.doc(api.conversation_id, "app.notes")
            notes = doc.get("items", [])
            notes.append(args["text"])
            doc["items"] = notes
            tx.set_doc(api.conversation_id, "app.notes", doc)
            return len(notes)

        count = api.commit(_write)
        return f"Saved durable note #{count}: {args['text']}"

    def tool_list_notes(_args: dict, api: ToolApi) -> str:
        doc = api.commit(lambda tx: tx.doc(api.conversation_id, "app.notes"))
        notes = doc.get("items", [])
        return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(notes)) if notes else "(no notes saved yet)"

    ext = Extension(
        name="workspace_assistant",
        system_prompt=(
            "You are a concise, helpful assistant running inside a durable SQLite agent harness. "
            "Use tools when helpful. Keep answers short and clear."
        ),
        tools=[
            ToolDef(
                name="list_files",
                description="List project files in the workspace.",
                replay="safe",
                execute=tool_list_files,
                parameters={"type": "object", "properties": {}},
            ),
            ToolDef(
                name="read_file",
                description="Read a file in the workspace (for example README.md or examples/01_chat_and_tools.py).",
                replay="safe",
                execute=tool_read_file,
                parameters={
                    "type": "object",
                    "properties": {"filename": {"type": "string", "description": "Relative file path in the workspace"}},
                    "required": ["filename"],
                },
            ),
            ToolDef(
                name="save_note",
                description="Durably save a note into the conversation's `app.notes` SQLite document.",
                replay="unsafe",
                execute=tool_save_note,
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Note text to save"}},
                    "required": ["text"],
                },
            ),
            ToolDef(
                name="list_notes",
                description="Read all durable notes saved in `app.notes`.",
                replay="safe",
                execute=tool_list_notes,
                parameters={"type": "object", "properties": {}},
            ),
        ],
    )
    reg = Registry()
    reg.install(ext)
    return reg


def run_prompt(harness: Harness, prompt: str) -> None:
    root = harness.root()
    sub = root.submit(prompt)
    settled = sub.wait()
    if settled.status == "done" and settled.answer_entry_id:
        entries = root.entries()
        answer = next((e for e in entries if e.id == settled.answer_entry_id), None)
        if answer:
            print(f"\nAssistant> {answer.data['text']}\n")
    else:
        print(f"\n[Status: {settled.status}, reason: {settled.reason}]\n")


def print_inspection(harness: Harness, cmd: str) -> None:
    root = harness.root()
    if cmd == "/entries":
        print("\n--- Entries in SQLite ---")
        for e in root.entries():
            print(f"  seq={e.seq} {e.kind:<14} {e.data}")
    elif cmd == "/docs":
        print("\n--- Documents in SQLite ---")
        for kind in ("pi.usage", "pi.live", "pi.inbox", "app.notes"):
            print(f"  {kind}: {root.doc(kind)}")
    elif cmd == "/tasks":
        print("\n--- Tasks in SQLite ---")
        rows = harness.storage.conn.execute(
            "SELECT id, kind, status, checkpoint_json, restarts, error FROM tasks ORDER BY rowid ASC"
        ).fetchall()
        for r in rows:
            print(
                f"  {r['id']} {r['kind']:<14} status={r['status']:<10} "
                f"restarts={r['restarts']} ckpt={r['checkpoint_json']} err={r['error']}"
            )
    print()


def main() -> None:
    load_dotenv(WORKSPACE / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    if api_key.lower() in PLACEHOLDER_KEYS:
        print("Missing OPENAI_API_KEY!")
        print(f"Copy `{WORKSPACE / '.env.example'}` to `{WORKSPACE / '.env'}` and set your API key, then run again.")
        sys.exit(1)

    llm = OpenAICompatibleLLM(api_key=api_key, model=model, base_url=base_url)
    harness = Harness.open(SqliteStorage(DB_PATH), llm=llm, registry=build_registry())
    harness.watch(lambda ev, payload: print(f"  [DB COMMIT] {ev}: {payload}"))
    harness.resume()

    if len(sys.argv) > 1:
        run_prompt(harness, " ".join(sys.argv[1:]))
        harness.close()
        return

    endpoint = base_url or "https://api.openai.com/v1"
    print(f"Connected ({model} @ {endpoint}) with durable DB: {DB_PATH.name}")
    print("Commands: /entries, /docs, /tasks, /reset, /quit\n")
    try:
        while True:
            user_input = input("You> ").strip()
            if not user_input:
                continue
            if user_input in ("/quit", "/exit"):
                break
            if user_input in ("/entries", "/docs", "/tasks"):
                print_inspection(harness, user_input)
                continue
            if user_input == "/reset":
                harness.root().reset("User cleared conversation context.")
                print("Appended pi.reset entry (history kept in SQLite, fresh context started).\n")
                continue
            run_prompt(harness, user_input)
    finally:
        harness.close()


if __name__ == "__main__":
    main()
