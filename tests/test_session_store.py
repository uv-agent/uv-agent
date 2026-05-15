from __future__ import annotations

from pathlib import Path

from uv_agent.session.store import ThreadStore


def test_list_threads_returns_latest_first_with_snippet(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    older = store.create_thread("Older")
    newer = store.create_thread("Newer")
    store.append(
        older,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
    )
    store.append(
        newer,
        "item.user",
        turn_id="t2",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]},
    )
    store.append(newer, "turn.completed", turn_id="t2", final_text="done")

    threads = store.list_threads()

    assert threads[0]["thread_id"] == newer
    assert threads[0]["turn_count"] == 1
    assert threads[0]["last_text"] == "new"


def test_thread_digest_starts_after_latest_compaction_and_hides_tools(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Digest")
    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
    )
    store.append(thread_id, "item.compaction", turn_id="t1", text="summary")
    store.append(
        thread_id,
        "item.user",
        turn_id="t2",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]},
    )
    store.append(thread_id, "item.tool_call", turn_id="t2", item={"name": "run_python"})
    store.append(thread_id, "item.assistant_delta", turn_id="t2", text="partial")
    store.append(thread_id, "turn.interrupted", turn_id="t2", reason="user_interrupt")

    digest = store.thread_digest(thread_id)

    assert digest["latest_compaction"]["text"] == "summary"
    assert digest["items"] == [
        {"role": "user", "text": "new"},
        {"role": "assistant", "text": "partial"},
        {"role": "system", "text": "turn interrupted: user_interrupt"},
    ]
