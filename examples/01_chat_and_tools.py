"""
Demo 1: Basic Conversation, Tool Calls, Atomic Commits & Usage
===============================================================
Run:
    python examples/01_chat_and_tools.py

Shows:
1. Opening a Harness over SQLite (or Memory) storage.
2. Installing an Extension with a system prompt + a tool (`add_numbers`).
3. Submitting user input (`root.submit(...)`).
4. Watching committed transcript entries (`pi.user` -> `pi.assistant` -> `pi.tool-result` -> `pi.assistant`)
   and reading `pi.usage`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_pi_durable import (
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
    # 1. Mock LLM: first asks to call `add_numbers`, then gives the final answer.
    llm = FauxLLM()
    llm.set_responses(
        [
            LLMResponse(
                text="Let me calculate that for you.",
                tool_calls=[LLMToolCall(id="call_1", name="add_numbers", arguments={"a": 17, "b": 25})],
                input_tokens=40,
                output_tokens=15,
            ),
            LLMResponse(
                text="17 + 25 is 42.",
                input_tokens=65,
                output_tokens=10,
            ),
        ]
    )

    # 2. Register an Extension with a tool and a system prompt section.
    math_ext = Extension(
        name="math_tools",
        system_prompt="You are a helpful math assistant.",
        tools=[
            ToolDef(
                name="add_numbers",
                description="Add two numbers",
                replay="safe",
                execute=lambda args, _api: str(args["a"] + args["b"]),
            )
        ],
    )
    registry = Registry()
    registry.install(math_ext)

    # 3. Open the Harness on an in-memory SQLite database.
    storage = SqliteStorage(":memory:")
    harness = Harness.open(storage, llm=llm, registry=registry)

    # Watch committed events (UI only ever sees committed state!).
    harness.watch(lambda event, payload: print(f"  [WATCH] {event}: {payload}"))

    root = harness.root()
    print("=== Submitting prompt ===")
    sub = root.submit("What is 17 + 25?")

    print("\n=== Running durable tasks (pi.generation -> pi.tool -> pi.generation) ===")
    settled = sub.wait()

    print("\n=== Result ===")
    print("Submission status:", settled.status)
    print("Usage doc:", root.doc("pi.usage"))

    print("\n=== Full Immutable Transcript (entries table) ===")
    for entry in root.entries():
        print(f"  seq={entry.seq} kind={entry.kind:<14} data={entry.data}")

    harness.close()


if __name__ == "__main__":
    main()
