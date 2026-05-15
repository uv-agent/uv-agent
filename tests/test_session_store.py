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
