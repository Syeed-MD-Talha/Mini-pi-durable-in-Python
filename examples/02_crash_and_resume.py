"""
Demo 2: Crash Recovery, Checkpoints, memo(), and request_id Idempotency
=======================================================================
Run:
    python examples/02_crash_and_resume.py

Inspired by `packages/durable/test/examples/12-recovery.ts`.

Shows:
1. A custom durable Task (`example.ticker`) counting from 1 to 4.
2. Process #1 crashes right after printing tick 2, BEFORE saving the next checkpoint.
3. Reopening a brand-new Harness on the same SQLite file recovers automatically:
   - Tick 2 runs again, BUT `runtime.memo("printed-2")` prevents printing it twice!
   - Then it finishes ticks 3 and 4.
4. Also shows `request_id` idempotency: retrying the same `submit(..., request_id=...)`
   after a crash returns the exact same Submission instead of duplicating it.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_pi_durable import (
    Extension,
    FauxLLM,
    Harness,
    LLMResponse,
    Registry,
    SimulatedCrash,
    SqliteStorage,
    TaskDef,
    TaskRecord,
    TaskRuntime,
)


def make_ticker_task(crash_on_tick: int | None = None) -> TaskDef:
    def initial(_inp: dict) -> dict:
        return {"phase": "tick", "n": 1}

    def phase_tick(task: TaskRecord, rt: TaskRuntime) -> None:
        n = task.checkpoint["n"]

        # Save the side-effect receipt BEFORE/WITH printing so a crash won't duplicate it!
        if rt.memo(f"printed-{n}") is None:
            rt.memo(f"printed-{n}", True)
            print(f"  [SIDE EFFECT] tick {n}")

        # Simulate sudden power loss on tick 2 before saving the next checkpoint!
        if crash_on_tick is not None and n == crash_on_tick:
            print(f"  [CRASH!] Process died abruptly during tick {n} before saving next checkpoint!")
            raise SimulatedCrash("Power lost")

        if n >= task.input["to"]:
            rt.complete(result=f"counted to {n}")
            return

        # Persist checkpoint for the next step (and sleep 0s)
        rt.sleep_until(rt.now(), {"phase": "tick", "n": n + 1})

    return TaskDef(name="example.ticker", initial=initial, phases={"tick": phase_tick})


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "session.sqlite"

        # ------------------------------------------------------------------
        # PROCESS #1: Crashes mid-task on tick 2!
        # ------------------------------------------------------------------
        print("=== PROCESS #1: Starting task (will crash on tick 2) ===")
        reg1 = Registry()
        reg1.install(Extension(name="ticker", tasks=[make_ticker_task(crash_on_tick=2)]))
        h1 = Harness.open(SqliteStorage(db_path), registry=reg1)
        root1 = h1.root()

        task_id = root1.commit(
            lambda tx: tx.create_task(make_ticker_task(), {"to": 4}, root1.id)
        )
        # Also submit an idempotent chat request before crashing
        sub1 = root1.submit("Hello after crash!", request_id="greeting-1")

        try:
            h1.resume()
        except SimulatedCrash:
            h1.storage.close()

        # ------------------------------------------------------------------
        # PROCESS #2: Reopen SQLite from disk and inspect + resume!
        # ------------------------------------------------------------------
        print("\n=== PROCESS #2: Reopening SQLite file ===")
        inspect_storage = SqliteStorage(db_path)
        saved_row = inspect_storage.conn.execute(
            "SELECT status, checkpoint_json, memos_json FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        print("  Saved in SQLite before recovery:")
        print(f"    status={saved_row['status']}, checkpoint={saved_row['checkpoint_json']}, memos={saved_row['memos_json']}")
        inspect_storage.close()

        llm2 = FauxLLM()
        llm2.set_responses([LLMResponse(text="Welcome back! I survived the crash.")])
        reg2 = Registry()
        reg2.install(Extension(name="ticker", tasks=[make_ticker_task(crash_on_tick=None)]))

        h2 = Harness.open(SqliteStorage(db_path), llm=llm2, registry=reg2)
        root2 = h2.root()

        # Retrying the same request_id does NOT create a duplicate submission!
        sub_retry = root2.submit("Hello after crash!", request_id="greeting-1")
        print(f"\n  request_id='greeting-1' -> same submission? {sub_retry.id == sub1.id} ({sub_retry.id})")

        print("\n=== Resuming unfinished work ===")
        h2.resume()

        finished = h2.get_task(task_id)
        print("\n=== After Recovery ===")
        print("  Ticker status:", finished.status, "result:", finished.result)
        print("  Memos saved:", finished.memos)
        print("  Chat transcript:")
        for e in root2.entries():
            print(f"    {e.kind}: {e.data}")

        h2.close()


if __name__ == "__main__":
    main()
