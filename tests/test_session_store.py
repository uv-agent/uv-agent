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


def test_list_threads_uses_model_response_text_as_latest_snippet(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Model response")
    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    )
    store.append(
        thread_id,
        "item.model_response",
        turn_id="t1",
        response_id="resp_1",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            }
        ],
        usage={},
    )
    store.append(thread_id, "turn.completed", turn_id="t1", final_text="answer")

    assert store.list_threads()[0]["last_text"] == "answer"
    assert store.thread_digest(thread_id)["last_text"] == "answer"


def test_thread_title_update_overrides_created_title(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("New thread")

    store.update_title(thread_id, "Generated title", source="generated")

    assert store.list_threads()[0]["title"] == "Generated title"
    assert store.thread_digest(thread_id)["title"] == "Generated title"


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
    store.append(thread_id, "item.assistant_partial", turn_id="t2", text="partial")
    store.append(thread_id, "turn.interrupted", turn_id="t2", reason="user_interrupt")

    digest = store.thread_digest(thread_id)

    assert digest["latest_compaction"]["text"] == "summary"
    assert digest["items"] == [
        {"role": "user", "text": "new"},
        {"role": "assistant", "text": "partial"},
        {"role": "system", "text": "turn interrupted: user_interrupt"},
    ]


def test_read_after_latest_compaction_returns_only_needed_suffix(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Compact suffix")
    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
    )
    store.append(thread_id, "item.compaction", turn_id="t1", text="summary1")
    store.append(
        thread_id,
        "item.user",
        turn_id="t2",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "middle"}]},
    )
    store.append(thread_id, "item.compaction", turn_id="t2", text="summary2")
    store.append(
        thread_id,
        "item.user",
        turn_id="t3",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]},
    )

    events, compaction = store.read_after_latest_compaction(thread_id)

    assert compaction is not None
    assert compaction["text"] == "summary2"
    assert [event["type"] for event in events] == ["item.user"]
    assert events[0]["turn_id"] == "t3"


def test_subthreads_are_stored_separately_and_listed_by_parent(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    parent = store.create_thread("Parent")
    child = store.create_thread(
        "Subagent: inspect",
        kind="subagent",
        parent_thread_id=parent,
        parent_turn_id="turn_1",
        parent_run_id="run_1",
        parent_script_id="scr_1",
    )
    store.append(child, "turn.completed", turn_id="turn_child", final_text="done")

    assert (tmp_path / "threads" / f"{parent}.jsonl").exists()
    assert (tmp_path / "subthreads" / f"{child}.jsonl").exists()
    assert [thread["thread_id"] for thread in store.list_threads()] == [parent]

    subthreads = store.list_subthreads(parent)

    assert [thread["thread_id"] for thread in subthreads] == [child]
    assert subthreads[0]["kind"] == "subagent"
    assert subthreads[0]["parent_turn_id"] == "turn_1"
