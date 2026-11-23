"""
Demo 4: Busy Conversations (`steer` vs `follow_up` & `pi.inbox`)
================================================================
Run:
    python examples/04_busy_inbox_steer.py

Shows:
1. Run #1 starts and requests a tool call.
2. While Run #1 is busy (mid-tool-round), we submit new messages:
   - `when_busy="steer"`     -> injected right after the tool round, inside Run #1!
   - `when_busy="follow_up"` -> waits in `pi.inbox` until Run #1 finishes, then starts Run #2!
   - `when_busy="reject"`    -> raises `ConversationBusy`!
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_pi_durable import (
    ConversationBusy,
    Extension,
    FauxLLM,
    Harness,
    LLMResponse,
    LLMToolCall,
    Registry,
    SqliteStorage,
    ToolDef,
)


def main() -> None:
    llm = FauxLLM()
    llm.set_responses(
        [
            LLMResponse(
                text="Reading config...",
                tool_calls=[LLMToolCall(id="c1", name="read_file", arguments={"path": "config.yaml"})],
            ),
            LLMResponse(text="Got your steer! Using YAML strict mode for config.yaml."),
            LLMResponse(text="Run #2 complete: tests also passed."),
        ]
    )

    reg = Registry()
    reg.install(
        Extension(
            name="files",
            tools=[ToolDef(name="read_file", description="Read a file", execute=lambda args, _api: f"contents of {args['path']}")],
        )
    )

    harness = Harness.open(SqliteStorage(":memory:"), llm=llm, registry=reg)
    root = harness.root()

    print("=== 1. Starting Run #1 ===")
    sub1 = root.submit("Check config.yaml")

    harness.step()
    harness.step()

    print("\n=== 2. Conversation is BUSY mid-run! Submitting steer, follow_up, and reject ===")
    sub_follow = root.submit("Also run tests afterwards", when_busy="follow_up")
    sub_steer = root.submit("Use strict YAML mode!", when_busy="steer")

    try:
        root.submit("Fail if busy", when_busy="reject")
    except ConversationBusy as err:
        print(f"  when_busy='reject' raised ConversationBusy as expected: {err}")

    print("  Current pi.inbox:", root.doc("pi.inbox"))

    print("\n=== 3. Finishing remaining steps ===")
    harness.run_until_idle()

    print("\n=== Final Transcript ===")
    for e in root.entries():
        print(f"  seq={e.seq} kind={e.kind:<14} data={e.data}")

    print("\n=== Submission Statuses ===")
    print("  sub1 (initial):  ", sub1.status().status)
    print("  sub_steer:       ", sub_steer.status().status)
    print("  sub_follow_up:   ", sub_follow.status().status)
    harness.close()


if __name__ == "__main__":
    main()
