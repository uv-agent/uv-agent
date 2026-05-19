from __future__ import annotations

from pathlib import Path

import pytest

from uv_agent.session.store import ThreadLockedError, ThreadStore


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


def test_thread_digest_includes_turn_error(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Errored")
    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    )
    store.append(
        thread_id,
        "turn.error",
        turn_id="t1",
        error_type="EmptyModelResponseError",
        message="Model returned an empty final response",
    )

    digest = store.thread_digest(thread_id)
    thread = store.list_threads()[0]

    assert digest["items"] == [
        {"role": "user", "text": "hello"},
        {"role": "system", "text": "turn error: Model returned an empty final response"},
    ]
    assert thread["interrupted_turn_count"] == 1
    assert thread["turn_count"] == 0


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


def test_compaction_offset_reads_suffix_without_parsing_old_events(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Compact offset")
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
    path = store.path(thread_id)
    original = path.read_bytes()
    first_newline = original.index(b"\n")
    path.write_bytes(b"{" + (b" " * (first_newline - 1)) + original[first_newline:])

    events, compaction = store.read_after_latest_compaction(thread_id)

    assert compaction is not None
    assert compaction["text"] == "summary"
    assert [event["turn_id"] for event in events] == ["t2"]


def test_history_segment_starts_at_latest_compaction_and_pages_to_previous_compaction(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Segmented history")
    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
    )
    first_compaction = store.append(thread_id, "item.compaction", turn_id="t1", text="summary1")
    store.append(
        thread_id,
        "item.user",
        turn_id="t2",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "middle"}]},
    )
    second_compaction = store.append(thread_id, "item.compaction", turn_id="t2", text="summary2")
    store.append(
        thread_id,
        "item.user",
        turn_id="t3",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]},
    )

    latest = store.read_history_segment(thread_id)
    previous = store.read_history_segment(thread_id, before_offset=latest.start_offset)

    assert latest.start_offset == second_compaction["_jsonl_offset"]
    assert [event["turn_id"] for event in latest.events] == ["t2", "t3"]
    assert latest.events[0]["type"] == "item.compaction"
    assert latest.has_more is True
    assert previous.start_offset == first_compaction["_jsonl_offset"]
    assert [event["turn_id"] for event in previous.events] == ["t1", "t2"]
    assert previous.has_more is True


def test_history_segment_reads_from_start_when_no_previous_compaction(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("History start")
    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
    )
    compaction = store.append(thread_id, "item.compaction", turn_id="t1", text="summary")
    store.append(
        thread_id,
        "item.user",
        turn_id="t2",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]},
    )

    event_types = {"item.user", "item.compaction"}
    latest = store.read_history_segment(thread_id, event_types=event_types)
    oldest = store.read_history_segment(
        thread_id,
        before_offset=latest.start_offset,
        event_types=event_types,
    )

    assert latest.start_offset == compaction["_jsonl_offset"]
    assert latest.has_more is True
    assert oldest.start_offset == 0
    assert [event["turn_id"] for event in oldest.events] == ["t1"]
    assert oldest.has_more is False


def test_thread_lock_blocks_other_store_writes_and_allows_owner_writes(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Locked")
    other = ThreadStore(tmp_path)

    with store.lock_thread(thread_id):
        store.append(
            thread_id,
            "item.user",
            turn_id="t1",
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "owner"}]},
        )
        with pytest.raises(ThreadLockedError):
            other.append(
                thread_id,
                "item.user",
                turn_id="t2",
                item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "other"}]},
            )
        assert store.lock_path(thread_id).exists()

    assert not store.lock_path(thread_id).exists()
    other.append(
        thread_id,
        "item.user",
        turn_id="t3",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "after"}]},
    )


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
