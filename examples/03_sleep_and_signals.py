"""
Demo 3: Parking a Tool with `sleep` + Human/Webhook `signal()`
==============================================================
Run:
    python examples/03_sleep_and_signals.py

Inspired by `packages/durable/test/examples/44-approval-gate.ts`.

Shows:
1. A `before_tool` hook intercepts a dangerous tool (`delete_file`) and returns
   a `sleep` gate waiting on signal `"approvals:<task_id>"`.
2. Process #1 closes completely while the tool is parked in SQLite (0 RAM held!).
3. Process #2 opens the same SQLite file, receives the human's decision via
   `harness.signal(...)`, wakes the parked tool, and completes the conversation!
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_pi_durable import (
    Extension,
    FauxLLM,
    Harness,
    HookApi,
    LLMResponse,
    LLMToolCall,
    Registry,
    SqliteStorage,
    ToolDef,
)


def approval_gate(tool_name: str, args: dict, api: HookApi) -> dict | None:
    if tool_name != "delete_file":
        return None

    sig = api.read_signal("approvals", api.task_id)
    if sig is None:
        print(f"  [HOOK] Pausing `{tool_name}` ({args}) until human approves task {api.task_id}...")
        return {"sleep": {"signals": [f"approvals:{api.task_id}"], "until": api.now() + 3600}}

    if sig.get("kind") == "expired":
        return {"block": "Approval timed out"}

    if sig["value"].get("allow"):
        print("  [HOOK] Approval received! Allowing tool execution.")
        return None
    return {"block": "Human denied permission"}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "approval.sqlite"

        def make_registry() -> Registry:
            reg = Registry()
            reg.install(
                Extension(
                    name="danger_tools",
                    tools=[
                        ToolDef(
                            name="delete_file",
                            description="Delete a file",
                            execute=lambda args, _api: f"Deleted {args['path']}",
                        )
                    ],
                    before_tool=[approval_gate],
                )
            )
            return reg

        # ------------------------------------------------------------------
        # PROCESS #1: LLM asks to delete a file -> tool parks & process exits!
        # ------------------------------------------------------------------
        print("=== PROCESS #1: Running until tool parks for approval ===")
        llm1 = FauxLLM()
        llm1.set_responses(
            [
                LLMResponse(
                    text="I need to remove temp.log.",
                    tool_calls=[LLMToolCall(id="c1", name="delete_file", arguments={"path": "temp.log"})],
                )
            ]
        )
        h1 = Harness.open(SqliteStorage(db_path), llm=llm1, registry=make_registry())
        root1 = h1.root()
        sub = root1.submit("Clean up temp.log")
        h1.run_until_idle()

        row = h1.storage.conn.execute(
            "SELECT id, status, waiting_signals_json FROM tasks WHERE status = 'sleeping'"
        ).fetchone()
        parked_task_id = row["id"]
        print(f"  Parked task in SQLite: id={parked_task_id}, status={row['status']}, signals={row['waiting_signals_json']}")
        h1.close()
        print("  Process #1 closed! (0 threads, 0 memory held while waiting)")

        # ------------------------------------------------------------------
        # PROCESS #2: Later, a human clicks "Approve" -> signal wakes the task!
        # ------------------------------------------------------------------
        print("\n=== PROCESS #2: Reopening & delivering human approval signal ===")
        llm2 = FauxLLM()
        llm2.set_responses([LLMResponse(text="Done! temp.log has been deleted.")])
        h2 = Harness.open(SqliteStorage(db_path), llm=llm2, registry=make_registry())

        delivery = h2.signal(parked_task_id, "approvals", parked_task_id, {"allow": True})
        print(f"  harness.signal(...) -> {delivery}")

        dup = h2.signal(parked_task_id, "approvals", parked_task_id, {"allow": True})
        print(f"  Duplicate signal result -> {dup}")

        h2.resume()

        print("\n=== Final Transcript ===")
        for e in h2.root().entries():
            print(f"  {e.kind:<14} {e.data}")
        print("  Original submission status:", h2.storage.conn.execute(
            "SELECT status FROM submissions WHERE id = ?", (sub.id,)
        ).fetchone()["status"])
        h2.close()


if __name__ == "__main__":
    main()
