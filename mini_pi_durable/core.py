"""
mini_pi_durable.core
====================
A minimal, beginner-friendly Python implementation of the core ideas behind
`@earendil-works/pi-durable`, backed by SQLite WAL transactions.

Concept mapping:
- SqliteStorage -> SQLite WAL-backed atomic persistence
- Tx / commit() -> Single-writer atomic transaction (all-or-nothing)
- Entry         -> Immutable transcript rows (`pi.user`, `pi.assistant`, `pi.tool-result`, `pi.reset`)
- Doc           -> Typed JSON state stored next to the transcript (`pi.live`, `pi.inbox`, `pi.usage`, `pi.agent`)
- Task          -> Checkpointed state machine (`phases`, `memo()`, `sleep_until()`, `signals`, `restart_limit`)
- Harness       -> Scheduler, crash recovery, built-in `pi.generation` / `pi.tool` tasks, Extension registry
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .llm import FauxLLM, LLMProvider


class SimulatedCrash(BaseException):
    """Raised in demos to simulate sudden process death mid-phase."""


class ConversationBusy(Exception):
    """Raised when submitting with when_busy='reject' while a run is active."""


class ConversationIdle(Exception):
    """Raised when submitting with when_idle='reject' while idle."""


def new_id(prefix: str = "") -> str:
    short = uuid.uuid4().hex[:12]
    return f"{prefix}_{short}" if prefix else short


# ---------------------------------------------------------------------------
# Data models (Transcript Entries, Tasks, Documents, Submissions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    seq: int
    id: str
    conversation_id: str
    kind: str  # e.g. "pi.user", "pi.assistant", "pi.tool-result", "pi.reset"
    data: dict[str, Any]
    created_at: float


@dataclass
class TaskRecord:
    id: str
    kind: str
    conversation_id: str
    owner_task_id: str | None
    status: str  # pending | running | waiting | sleeping | completed | failed | aborted
    input: Any
    checkpoint: dict[str, Any]
    memos: dict[str, Any]
    signals: dict[str, Any]
    waiting_signals: list[str]
    waiting_on: list[str]
    wake_at: float | None
    wake: dict[str, Any] | None
    restarts: int
    result: Any
    error: str | None


@dataclass
class SubmissionRecord:
    id: str
    conversation_id: str
    request_id: str | None
    type: str  # "input" | "write"
    content: str
    when_busy: str  # "follow_up" | "steer" | "reject"
    when_idle: str  # "start" | "reject"
    status: str  # "queued" | "running" | "done" | "unanswered"
    answer_entry_id: str | None = None
    reason: str | None = None


@dataclass
class ToolDef:
    name: str
    description: str
    execute: Callable[[dict[str, Any], "ToolApi"], Any]
    replay: str = "unsafe"  # "safe" = rerun after crash, "unsafe" = report interrupted
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class TaskDef:
    name: str
    initial: Callable[[Any], dict[str, Any]]
    phases: dict[str, Callable[[TaskRecord, "TaskRuntime"], None]]
    abort: Callable[[TaskRecord, "TaskRuntime"], None] | None = None


@dataclass
class Extension:
    name: str
    system_prompt: str | None = None
    tools: list[ToolDef] = field(default_factory=list)
    tasks: list[TaskDef] = field(default_factory=list)
    before_tool: list[Callable[[str, dict[str, Any], "HookApi"], dict[str, Any] | None]] = field(
        default_factory=list
    )


class Registry:
    def __init__(self) -> None:
        self.extensions: dict[str, Extension] = {}

    def install(self, ext: Extension) -> None:
        # Installing with the same name replaces in place (hot reload)
        self.extensions[ext.name] = ext

    def get_tool(self, name: str) -> ToolDef | None:
        found: ToolDef | None = None
        for ext in self.extensions.values():
            for tool in ext.tools:
                if tool.name == name:
                    found = tool
        return found

    def get_task(self, name: str) -> TaskDef | None:
        for ext in self.extensions.values():
            for task in ext.tasks:
                if task.name == name:
                    return task
        return None

    def all_tools(self) -> list[ToolDef]:
        by_name: dict[str, ToolDef] = {}
        for ext in self.extensions.values():
            for tool in ext.tools:
                by_name[tool.name] = tool
        return list(by_name.values())

    def system_prompt(self) -> str:
        parts = [ext.system_prompt for ext in self.extensions.values() if ext.system_prompt]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# SQLite Storage + Atomic Transaction (Tx)
# ---------------------------------------------------------------------------


class SqliteStorage:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                owner_task_id TEXT,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entries (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                conversation_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS docs (
                conversation_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY (conversation_id, kind)
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                owner_task_id TEXT,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL,
                memos_json TEXT NOT NULL,
                signals_json TEXT NOT NULL,
                waiting_signals_json TEXT NOT NULL,
                waiting_on_json TEXT NOT NULL,
                wake_at REAL,
                wake_json TEXT,
                restarts INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                request_id TEXT UNIQUE,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                when_busy TEXT NOT NULL,
                when_idle TEXT NOT NULL,
                status TEXT NOT NULL,
                answer_entry_id TEXT,
                reason TEXT
            );
            """
        )

    def close(self) -> None:
        self.conn.close()


class Tx:
    """
    Transaction handle passed to commit(fn).
    Everything written through `tx` succeeds together or rolls back together.
    """

    def __init__(self, storage: SqliteStorage, harness: "Harness") -> None:
        self.storage = storage
        self.harness = harness
        self._dirty_docs: dict[tuple[str, str], dict[str, Any]] = {}
        self.events: list[tuple[str, Any]] = []

    def create_conversation(self, conv_id: str | None = None, owner_task_id: str | None = None) -> str:
        cid = conv_id or new_id("conv")
        existing = self.storage.conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
        if existing is not None:
            return cid
        self.storage.conn.execute(
            "INSERT INTO conversations (id, owner_task_id, created_at) VALUES (?, ?, ?)",
            (cid, owner_task_id, self.harness.now()),
        )
        self.set_doc(cid, "pi.agent", {"model": "default"})
        self.set_doc(cid, "pi.live", {"run": None, "tools": {}, "run_failure": None})
        self.set_doc(cid, "pi.inbox", {"items": []})
        self.set_doc(cid, "pi.usage", {"input_tokens": 0, "output_tokens": 0, "requests": 0})
        return cid

    def doc(self, conv_id: str, kind: str) -> dict[str, Any]:
        key = (conv_id, kind)
        if key in self._dirty_docs:
            return self._dirty_docs[key]
        row = self.storage.conn.execute(
            "SELECT value_json FROM docs WHERE conversation_id = ? AND kind = ?",
            (conv_id, kind),
        ).fetchone()
        val = json.loads(row["value_json"]) if row else {}
        self._dirty_docs[key] = val
        return val

    def set_doc(self, conv_id: str, kind: str, value: dict[str, Any]) -> None:
        self._dirty_docs[(conv_id, kind)] = value

    def flush_docs(self) -> None:
        for (conv_id, kind), val in self._dirty_docs.items():
            self.storage.conn.execute(
                """
                INSERT INTO docs (conversation_id, kind, value_json)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id, kind) DO UPDATE SET value_json = excluded.value_json
                """,
                (conv_id, kind, json.dumps(val)),
            )

    def append_entry(self, conv_id: str, kind: str, data: dict[str, Any]) -> Entry:
        eid = new_id("ent")
        ts = self.harness.now()
        cur = self.storage.conn.execute(
            "INSERT INTO entries (id, conversation_id, kind, data_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (eid, conv_id, kind, json.dumps(data), ts),
        )
        entry = Entry(seq=cur.lastrowid or 0, id=eid, conversation_id=conv_id, kind=kind, data=data, created_at=ts)
        self.events.append(("entry", entry))
        return entry

    def create_task(
        self,
        task_def: TaskDef,
        input_data: Any,
        conv_id: str,
        owner_task_id: str | None = None,
    ) -> str:
        tid = new_id("task")
        checkpoint = task_def.initial(input_data)
        self.storage.conn.execute(
            """
            INSERT INTO tasks (
                id, kind, conversation_id, owner_task_id, status,
                input_json, checkpoint_json, memos_json, signals_json,
                waiting_signals_json, waiting_on_json, wake_at, wake_json, restarts, result_json, error
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, '{}', '{}', '[]', '[]', NULL, NULL, 0, NULL, NULL)
            """,
            (tid, task_def.name, conv_id, owner_task_id, json.dumps(input_data), json.dumps(checkpoint)),
        )
        self.events.append(("task_created", {"id": tid, "kind": task_def.name}))
        return tid

    def get_task(self, task_id: str) -> TaskRecord | None:
        row = self.storage.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def get_submission_by_request_id(self, request_id: str) -> SubmissionRecord | None:
        row = self.storage.conn.execute(
            "SELECT * FROM submissions WHERE request_id = ?", (request_id,)
        ).fetchone()
        return _row_to_submission(row) if row else None


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        id=row["id"],
        kind=row["kind"],
        conversation_id=row["conversation_id"],
        owner_task_id=row["owner_task_id"],
        status=row["status"],
        input=json.loads(row["input_json"]),
        checkpoint=json.loads(row["checkpoint_json"]),
        memos=json.loads(row["memos_json"]),
        signals=json.loads(row["signals_json"]),
        waiting_signals=json.loads(row["waiting_signals_json"]),
        waiting_on=json.loads(row["waiting_on_json"]),
        wake_at=row["wake_at"],
        wake=json.loads(row["wake_json"]) if row["wake_json"] else None,
        restarts=row["restarts"],
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=row["error"],
    )


def _row_to_submission(row: sqlite3.Row) -> SubmissionRecord:
    return SubmissionRecord(
        id=row["id"],
        conversation_id=row["conversation_id"],
        request_id=row["request_id"],
        type=row["type"],
        content=row["content"],
        when_busy=row["when_busy"],
        when_idle=row["when_idle"],
        status=row["status"],
        answer_entry_id=row["answer_entry_id"],
        reason=row["reason"],
    )


# ---------------------------------------------------------------------------
# Task Runtime, Hook API, Tool API (passed to tasks, hooks, and tools)
# ---------------------------------------------------------------------------


class TaskRuntime:
    def __init__(self, harness: "Harness", task: TaskRecord) -> None:
        self.harness = harness
        self.task = task

    def now(self) -> float:
        return self.harness.now()

    def memo(self, key: str, value: Any = ...) -> Any:
        """
        Idempotency helper:
        - `memo(key)` reads the saved value (or None).
        - `memo(key, value)` durably stores `value` on first write and returns it.
        """

        def _op(tx: Tx) -> Any:
            t = tx.get_task(self.task.id)
            assert t is not None
            if value is ...:
                return t.memos.get(key)
            if key not in t.memos:
                t.memos[key] = value
                tx.storage.conn.execute(
                    "UPDATE tasks SET memos_json = ? WHERE id = ?",
                    (json.dumps(t.memos), self.task.id),
                )
            self.task.memos = t.memos
            return t.memos[key]

        return self.harness.commit(_op)

    def read_signal(self, signal_name: str, key: str) -> dict[str, Any] | None:
        t = self.harness.get_task(self.task.id)
        if t is None:
            return None
        return t.signals.get(f"{signal_name}:{key}")

    def sleep_until(
        self,
        until: float | None,
        checkpoint: dict[str, Any],
        signals: list[str] | None = None,
    ) -> None:
        """Park this task in SQLite until `until` or a signal arrives."""
        sig_list = signals or []

        def _op(tx: Tx) -> None:
            t = tx.get_task(self.task.id)
            assert t is not None
            for skey in sig_list:
                if skey in t.signals:
                    tx.storage.conn.execute(
                        """
                        UPDATE tasks
                        SET status = 'pending', checkpoint_json = ?, wake_at = NULL,
                            waiting_signals_json = '[]', wake_json = ?, restarts = 0
                        WHERE id = ?
                        """,
                        (
                            json.dumps(checkpoint),
                            json.dumps({"reason": "signal", "key": skey}),
                            self.task.id,
                        ),
                    )
                    return

            tx.storage.conn.execute(
                """
                UPDATE tasks
                SET status = 'sleeping', checkpoint_json = ?, wake_at = ?,
                    waiting_signals_json = ?, wake_json = NULL, restarts = 0
                WHERE id = ?
                """,
                (json.dumps(checkpoint), until, json.dumps(sig_list), self.task.id),
            )

        self.harness.commit(_op)

    def wait_for_tasks(self, child_ids: list[str], checkpoint: dict[str, Any]) -> None:
        """Park until all `child_ids` reach a terminal state."""

        def _op(tx: Tx) -> None:
            tx.storage.conn.execute(
                """
                UPDATE tasks
                SET status = 'waiting', checkpoint_json = ?, waiting_on_json = ?, restarts = 0
                WHERE id = ?
                """,
                (json.dumps(checkpoint), json.dumps(child_ids), self.task.id),
            )

        self.harness.commit(_op)

    def next_phase(self, checkpoint: dict[str, Any], extra_fn: Callable[[Tx], None] | None = None) -> None:
        def _op(tx: Tx) -> None:
            if extra_fn:
                extra_fn(tx)
            tx.storage.conn.execute(
                """
                UPDATE tasks
                SET status = 'pending', checkpoint_json = ?, restarts = 0
                WHERE id = ?
                """,
                (json.dumps(checkpoint), self.task.id),
            )

        self.harness.commit(_op)

    def complete(self, result: Any = None, extra_fn: Callable[[Tx], None] | None = None) -> None:
        def _op(tx: Tx) -> None:
            if extra_fn:
                extra_fn(tx)
            tx.storage.conn.execute(
                """
                UPDATE tasks
                SET status = 'completed', result_json = ?, restarts = 0
                WHERE id = ?
                """,
                (json.dumps(result), self.task.id),
            )

        self.harness.commit(_op)

    def fail(self, error: str, extra_fn: Callable[[Tx], None] | None = None) -> None:
        def _op(tx: Tx) -> None:
            if extra_fn:
                extra_fn(tx)
            tx.storage.conn.execute(
                "UPDATE tasks SET status = 'failed', error = ? WHERE id = ?",
                (error, self.task.id),
            )

        self.harness.commit(_op)


class HookApi:
    def __init__(self, runtime: TaskRuntime) -> None:
        self._rt = runtime

    @property
    def task_id(self) -> str:
        return self._rt.task.id

    def now(self) -> float:
        return self._rt.now()

    def memo(self, key: str, value: Any = ...) -> Any:
        return self._rt.memo(key, value)

    def read_signal(self, signal_name: str, key: str) -> dict[str, Any] | None:
        return self._rt.read_signal(signal_name, key)


class ToolApi:
    def __init__(self, runtime: TaskRuntime) -> None:
        self._rt = runtime

    @property
    def task_id(self) -> str:
        return self._rt.task.id

    @property
    def conversation_id(self) -> str:
        return self._rt.task.conversation_id

    def memo(self, key: str, value: Any = ...) -> Any:
        return self._rt.memo(key, value)

    def commit(self, fn: Callable[[Tx], Any]) -> Any:
        return self._rt.harness.commit(fn)


# ---------------------------------------------------------------------------
# Conversation Handle & Submission Handle
# ---------------------------------------------------------------------------


class Submission:
    def __init__(self, harness: "Harness", sub_id: str) -> None:
        self.harness = harness
        self.id = sub_id

    def status(self) -> SubmissionRecord:
        row = self.harness.storage.conn.execute(
            "SELECT * FROM submissions WHERE id = ?", (self.id,)
        ).fetchone()
        assert row is not None
        return _row_to_submission(row)

    def wait(self) -> SubmissionRecord:
        self.harness.run_until_idle()
        return self.status()


class Conversation:
    def __init__(self, harness: "Harness", conv_id: str) -> None:
        self.harness = harness
        self.id = conv_id

    def commit(self, fn: Callable[[Tx], Any]) -> Any:
        return self.harness.commit(fn)

    def submit(
        self,
        content: str,
        *,
        request_id: str | None = None,
        when_busy: str = "follow_up",
        when_idle: str = "start",
    ) -> Submission:
        def _op(tx: Tx) -> str:
            if request_id is not None:
                existing = tx.get_submission_by_request_id(request_id)
                if existing is not None:
                    return existing.id

            live = tx.doc(self.id, "pi.live")
            is_busy = live.get("run") is not None

            if is_busy and when_busy == "reject":
                raise ConversationBusy(f"Conversation {self.id} is busy")
            if not is_busy and when_idle == "reject":
                raise ConversationIdle(f"Conversation {self.id} is idle")

            sub_id = new_id("sub")
            tx.storage.conn.execute(
                """
                INSERT INTO submissions (
                    id, conversation_id, request_id, type, content, when_busy, when_idle, status
                ) VALUES (?, ?, ?, 'input', ?, ?, ?, ?)
                """,
                (sub_id, self.id, request_id, content, when_busy, when_idle, "queued" if is_busy else "running"),
            )

            if is_busy:
                inbox = tx.doc(self.id, "pi.inbox")
                inbox["items"].append({"submission_id": sub_id, "content": content, "mode": when_busy})
                tx.set_doc(self.id, "pi.inbox", inbox)
                tx.events.append(("inbox_queued", {"submission_id": sub_id, "mode": when_busy}))
            else:
                self.harness._start_run_in_tx(tx, self.id, sub_id, content)

            return sub_id

        sub_id = self.harness.commit(_op)
        return Submission(self.harness, sub_id)

    def entries(self) -> list[Entry]:
        rows = self.harness.storage.conn.execute(
            "SELECT * FROM entries WHERE conversation_id = ? ORDER BY seq ASC", (self.id,)
        ).fetchall()
        return [
            Entry(
                seq=r["seq"],
                id=r["id"],
                conversation_id=r["conversation_id"],
                kind=r["kind"],
                data=json.loads(r["data_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def active_messages(self) -> list[dict[str, Any]]:
        """Entries from the newest reset onward (what the LLM sees)."""
        all_entries = self.entries()
        start = 0
        for i, e in enumerate(all_entries):
            if e.kind == "pi.reset":
                start = i
        msgs: list[dict[str, Any]] = []
        for e in all_entries[start:]:
            if e.kind == "pi.user":
                msgs.append({"role": "user", "content": e.data["content"]})
            elif e.kind == "pi.assistant":
                msgs.append({"role": "assistant", "content": e.data["text"], "tool_calls": e.data.get("tool_calls", [])})
            elif e.kind == "pi.tool-result":
                msgs.append(
                    {
                        "role": "tool_result",
                        "call_id": e.data["call_id"],
                        "tool_name": e.data["tool_name"],
                        "output": e.data["output"],
                        "is_error": e.data["is_error"],
                    }
                )
            elif e.kind == "pi.reset" and e.data.get("handoff"):
                msgs.append({"role": "user", "content": e.data["handoff"]})
        return msgs

    def doc(self, kind: str) -> dict[str, Any]:
        return self.harness.snapshot(self.id, kind)

    def reset(self, handoff: str | None = None) -> Entry:
        return self.commit(lambda tx: tx.append_entry(self.id, "pi.reset", {"handoff": handoff}))


# ---------------------------------------------------------------------------
# Built-in Tasks: pi.generation & pi.tool
# ---------------------------------------------------------------------------


def _build_generation_task_def(harness: "Harness") -> TaskDef:
    def initial(inp: dict[str, Any]) -> dict[str, Any]:
        return {"phase": "request", "turns": 0}

    def phase_request(task: TaskRecord, rt: TaskRuntime) -> None:
        conv = Conversation(harness, task.conversation_id)
        messages = conv.active_messages()
        system_prompt = harness.registry.system_prompt()
        tools = harness.registry.all_tools()
        resp = harness.llm.generate(messages, system_prompt, tools)

        def _after_llm(tx: Tx) -> Entry:
            usage = tx.doc(conv.id, "pi.usage")
            usage["input_tokens"] += resp.input_tokens
            usage["output_tokens"] += resp.output_tokens
            usage["requests"] += 1
            tx.set_doc(conv.id, "pi.usage", usage)

            assistant_entry = tx.append_entry(
                conv.id,
                "pi.assistant",
                {
                    "text": resp.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in resp.tool_calls
                    ],
                },
            )
            tx.storage.conn.execute(
                "UPDATE tasks SET checkpoint_json = ? WHERE id = ?",
                (json.dumps({"phase": "after_response", "last_entry_id": assistant_entry.id}), task.id),
            )
            return assistant_entry

        if resp.tool_calls:
            def _spawn_tools(tx: Tx) -> None:
                _after_llm(tx)
                tool_task_ids: list[str] = []
                for tc in resp.tool_calls:
                    tid = tx.create_task(
                        harness._tool_task_def,
                        {"call_id": tc.id, "name": tc.name, "arguments": tc.arguments},
                        conv.id,
                        owner_task_id=task.id,
                    )
                    tool_task_ids.append(tid)
                tx.storage.conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'waiting', waiting_on_json = ?, checkpoint_json = ?, restarts = 0
                    WHERE id = ?
                    """,
                    (
                        json.dumps(tool_task_ids),
                        json.dumps({"phase": "after_tools", "turns": task.checkpoint["turns"] + 1}),
                        task.id,
                    ),
                )

            harness.commit(_spawn_tools)
            return

        def _finish_run(tx: Tx) -> None:
            assistant_entry = _after_llm(tx)
            live = tx.doc(conv.id, "pi.live")
            run_info = live.get("run") or {}
            for sub_id in run_info.get("submission_ids", []):
                tx.storage.conn.execute(
                    "UPDATE submissions SET status = 'done', answer_entry_id = ? WHERE id = ?",
                    (assistant_entry.id, sub_id),
                )
            live["run"] = None
            tx.set_doc(conv.id, "pi.live", live)

            inbox = tx.doc(conv.id, "pi.inbox")
            if inbox["items"]:
                next_item = inbox["items"].pop(0)
                tx.set_doc(conv.id, "pi.inbox", inbox)
                tx.storage.conn.execute(
                    "UPDATE submissions SET status = 'running' WHERE id = ?",
                    (next_item["submission_id"],),
                )
                harness._start_run_in_tx(tx, conv.id, next_item["submission_id"], next_item["content"])

        rt.complete(result={"status": "answered"}, extra_fn=_finish_run)

    def phase_after_tools(task: TaskRecord, rt: TaskRuntime) -> None:
        def _check_steers(tx: Tx) -> None:
            inbox = tx.doc(task.conversation_id, "pi.inbox")
            steers = [item for item in inbox["items"] if item["mode"] == "steer"]
            remaining = [item for item in inbox["items"] if item["mode"] != "steer"]
            if steers:
                inbox["items"] = remaining
                tx.set_doc(task.conversation_id, "pi.inbox", inbox)
                live = tx.doc(task.conversation_id, "pi.live")
                for item in steers:
                    tx.append_entry(task.conversation_id, "pi.user", {"content": item["content"], "steer": True})
                    tx.storage.conn.execute(
                        "UPDATE submissions SET status = 'running' WHERE id = ?",
                        (item["submission_id"],),
                    )
                    if live.get("run"):
                        live["run"]["submission_ids"].append(item["submission_id"])
                tx.set_doc(task.conversation_id, "pi.live", live)

        rt.next_phase({"phase": "request", "turns": task.checkpoint.get("turns", 1)}, extra_fn=_check_steers)

    return TaskDef(
        name="pi.generation",
        initial=initial,
        phases={"request": phase_request, "after_tools": phase_after_tools},
    )


def _build_tool_task_def(harness: "Harness") -> TaskDef:
    def initial(inp: dict[str, Any]) -> dict[str, Any]:
        return {"phase": "run"}

    def phase_run(task: TaskRecord, rt: TaskRuntime) -> None:
        name = task.input["name"]
        args = task.input["arguments"]
        call_id = task.input["call_id"]

        hook_api = HookApi(rt)
        for ext in harness.registry.extensions.values():
            for hook_fn in ext.before_tool:
                decision = hook_fn(name, args, hook_api)
                if decision is None:
                    continue
                if "block" in decision:
                    rt.complete(
                        {"is_error": True},
                        extra_fn=lambda tx: tx.append_entry(
                            task.conversation_id,
                            "pi.tool-result",
                            {"call_id": call_id, "tool_name": name, "output": f"Blocked: {decision['block']}", "is_error": True},
                        ),
                    )
                    return
                if "sleep" in decision:
                    s = decision["sleep"]
                    rt.sleep_until(s.get("until"), {"phase": "run"}, signals=s.get("signals", []))
                    return

        tool = harness.registry.get_tool(name)
        if tool is None:
            rt.complete(
                {"is_error": True},
                extra_fn=lambda tx: tx.append_entry(
                    task.conversation_id,
                    "pi.tool-result",
                    {"call_id": call_id, "tool_name": name, "output": f"Unknown tool: {name}", "is_error": True},
                ),
            )
            return

        try:
            output = tool.execute(args, ToolApi(rt))
            is_error = False
        except SimulatedCrash:
            raise
        except Exception as exc:
            output = f"Tool error: {exc}"
            is_error = True

        rt.complete(
            {"output": output, "is_error": is_error},
            extra_fn=lambda tx: tx.append_entry(
                task.conversation_id,
                "pi.tool-result",
                {"call_id": call_id, "tool_name": name, "output": output, "is_error": is_error},
            ),
        )

    return TaskDef(name="pi.tool", initial=initial, phases={"run": phase_run})


# ---------------------------------------------------------------------------
# Harness (The Durable Runtime & Scheduler)
# ---------------------------------------------------------------------------


class Harness:
    def __init__(
        self,
        storage: SqliteStorage,
        *,
        llm: LLMProvider | None = None,
        registry: Registry | None = None,
        restart_limit: int = 5,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.storage = storage
        self.llm: LLMProvider = llm or FauxLLM()
        self.registry = registry or Registry()
        self.restart_limit = restart_limit
        self._now = now_fn or time.time
        self._watchers: list[Callable[[str, Any], None]] = []

        self._generation_task_def = _build_generation_task_def(self)
        self._tool_task_def = _build_tool_task_def(self)
        self._builtin_tasks: dict[str, TaskDef] = {
            "pi.generation": self._generation_task_def,
            "pi.tool": self._tool_task_def,
        }

    @classmethod
    def open(
        cls,
        storage: SqliteStorage,
        *,
        llm: LLMProvider | None = None,
        registry: Registry | None = None,
        restart_limit: int = 5,
        now_fn: Callable[[], float] | None = None,
    ) -> "Harness":
        harness = cls(storage, llm=llm, registry=registry, restart_limit=restart_limit, now_fn=now_fn)
        harness._recover_on_open()
        return harness

    def now(self) -> float:
        return self._now()

    def watch(self, fn: Callable[[str, Any], None]) -> None:
        self._watchers.append(fn)

    def commit(self, fn: Callable[[Tx], Any]) -> Any:
        tx = Tx(self.storage, self)
        self.storage.conn.execute("BEGIN IMMEDIATE")
        try:
            result = fn(tx)
            tx.flush_docs()
            self.storage.conn.execute("COMMIT")
        except BaseException:
            self.storage.conn.execute("ROLLBACK")
            raise
        for event_name, payload in tx.events:
            for watcher in self._watchers:
                watcher(event_name, payload)
        return result

    def root(self) -> Conversation:
        self.commit(lambda tx: tx.create_conversation("root"))
        return Conversation(self, "root")

    def snapshot(self, conv_id: str, kind: str) -> dict[str, Any]:
        row = self.storage.conn.execute(
            "SELECT value_json FROM docs WHERE conversation_id = ? AND kind = ?",
            (conv_id, kind),
        ).fetchone()
        return json.loads(row["value_json"]) if row else {}

    def get_task(self, task_id: str) -> TaskRecord | None:
        row = self.storage.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def _start_run_in_tx(self, tx: Tx, conv_id: str, sub_id: str, content: str) -> str:
        tx.append_entry(conv_id, "pi.user", {"content": content})
        gen_task_id = tx.create_task(self._generation_task_def, {"submission_id": sub_id}, conv_id)
        live = tx.doc(conv_id, "pi.live")
        live["run"] = {"generation_task_id": gen_task_id, "submission_ids": [sub_id]}
        tx.set_doc(conv_id, "pi.live", live)
        return gen_task_id

    def _recover_on_open(self) -> None:
        """
        Recovery when reopening SQLite after a previous process crashed or exited:
        Any task still marked `running` was cut off mid-flight.
        """

        def _recover(tx: Tx) -> None:
            rows = tx.storage.conn.execute("SELECT * FROM tasks WHERE status = 'running'").fetchall()
            for row in rows:
                t = _row_to_task(row)
                new_restarts = t.restarts + 1
                if t.kind == "pi.tool":
                    tool = self.registry.get_tool(t.input["name"])
                    if tool is None or tool.replay != "safe":
                        tx.append_entry(
                            t.conversation_id,
                            "pi.tool-result",
                            {
                                "call_id": t.input["call_id"],
                                "tool_name": t.input["name"],
                                "output": "Interrupted by crash (replay='unsafe')",
                                "is_error": True,
                            },
                        )
                        tx.storage.conn.execute(
                            "UPDATE tasks SET status = 'completed', result_json = '{\"is_error\": true}' WHERE id = ?",
                            (t.id,),
                        )
                        continue
                if new_restarts >= self.restart_limit:
                    tx.storage.conn.execute(
                        "UPDATE tasks SET status = 'failed', error = 'restart_limit', restarts = ? WHERE id = ?",
                        (new_restarts, t.id),
                    )
                else:
                    tx.storage.conn.execute(
                        "UPDATE tasks SET status = 'pending', restarts = ? WHERE id = ?",
                        (new_restarts, t.id),
                    )

        self.commit(_recover)

    def signal(self, task_id: str, signal_name: str, key: str, value: Any) -> str:
        """Deliver an external typed signal (webhook / human approval) to a task."""
        skey = f"{signal_name}:{key}"

        def _op(tx: Tx) -> str:
            t = tx.get_task(task_id)
            if t is None:
                return "not_found"
            if t.status in ("completed", "failed", "aborted"):
                return "closed"
            if skey in t.signals:
                return "duplicate"
            t.signals[skey] = {"kind": "value", "value": value}
            if t.status == "sleeping" and skey in t.waiting_signals:
                tx.storage.conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending', signals_json = ?, waiting_signals_json = '[]',
                        wake_at = NULL, wake_json = ?, restarts = 0
                    WHERE id = ?
                    """,
                    (json.dumps(t.signals), json.dumps({"reason": "signal", "key": skey}), t.id),
                )
            else:
                tx.storage.conn.execute(
                    "UPDATE tasks SET signals_json = ? WHERE id = ?",
                    (json.dumps(t.signals), t.id),
                )
            tx.events.append(("signal_delivered", {"task_id": task_id, "key": skey}))
            return "delivered"

        return self.commit(_op)

    def _wake_due_tasks(self) -> bool:
        now = self.now()
        woke_any = False

        def _check(tx: Tx) -> None:
            nonlocal woke_any
            sleeping = tx.storage.conn.execute("SELECT * FROM tasks WHERE status = 'sleeping'").fetchall()
            for row in sleeping:
                t = _row_to_task(row)
                if t.wake_at is not None and t.wake_at <= now:
                    for skey in t.waiting_signals:
                        t.signals.setdefault(skey, {"kind": "expired"})
                    tx.storage.conn.execute(
                        """
                        UPDATE tasks
                        SET status = 'pending', signals_json = ?, waiting_signals_json = '[]',
                            wake_at = NULL, wake_json = '{"reason": "expired"}'
                        WHERE id = ?
                        """,
                        (json.dumps(t.signals), t.id),
                    )
                    woke_any = True

            waiting = tx.storage.conn.execute("SELECT * FROM tasks WHERE status = 'waiting'").fetchall()
            for row in waiting:
                t = _row_to_task(row)
                if not t.waiting_on:
                    continue
                placeholders = ",".join("?" for _ in t.waiting_on)
                children = tx.storage.conn.execute(
                    f"SELECT status FROM tasks WHERE id IN ({placeholders})", t.waiting_on
                ).fetchall()
                if all(c["status"] in ("completed", "failed", "aborted") for c in children):
                    tx.storage.conn.execute(
                        "UPDATE tasks SET status = 'pending', waiting_on_json = '[]' WHERE id = ?",
                        (t.id,),
                    )
                    woke_any = True

        self.commit(_check)
        return woke_any

    def step(self) -> bool:
        """Run ONE pending task phase. Returns True if work ran, False if nothing is runnable."""
        self._wake_due_tasks()
        row = self.storage.conn.execute(
            "SELECT * FROM tasks WHERE status = 'pending' ORDER BY rowid ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return False

        task = _row_to_task(row)
        task_def = self._builtin_tasks.get(task.kind) or self.registry.get_task(task.kind)
        if task_def is None:
            return False

        self.commit(
            lambda tx: tx.storage.conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (task.id,))
        )

        phase_name = task.checkpoint["phase"]
        phase_fn = task_def.phases[phase_name]
        rt = TaskRuntime(self, task)
        try:
            phase_fn(task, rt)
        except SimulatedCrash:
            raise
        except Exception as exc:
            rt.fail(str(exc))
        return True

    def run_until_idle(self) -> None:
        while self.step():
            pass

    def resume(self) -> None:
        self.run_until_idle()

    def close(self) -> None:
        self.storage.close()
