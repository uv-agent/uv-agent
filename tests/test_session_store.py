from __future__ import annotations

from pathlib import Path

import pytest

from uv_agent.blobs import BlobStore
from uv_agent.runner.run_log import RunLogStore
from uv_agent.session.store import (
    HISTORY_SEGMENT_CACHE_MAX_ENTRIES,
    ThreadLockedError,
    ThreadStore,
    _THREAD_LOCK_CONTEXT,
)


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


def test_thread_level_and_model_switch_warning_update_metadata(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("New thread")

    store.append(thread_id, "thread.level_updated", level="large", model="main")
    warning = store.append(
        thread_id,
        "thread.model_switch_warning",
        from_level="medium",
        to_level="large",
        from_model="fast",
        to_model="main",
        message="context conversion is best effort",
    )

    digest = store.thread_digest(thread_id)
    thread = store.list_threads()[0]

    assert digest["active_level"] == "large"
    assert digest["active_model"] == "main"
    assert thread["active_level"] == "large"
    assert thread["active_model"] == "main"
    assert digest["latest_model_switch_warning"]["message"] == "context conversion is best effort"
    assert digest["latest_model_switch_warning"]["_event_id"] == warning["_event_id"]


def test_thread_digest_exposes_extra_metadata(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Worktree")
    worktree_path = tmp_path / "project" / ".uv-agent" / "worktrees" / "feature"
    store.update_thread_metadata(
        thread_id,
        updates={
            "worktree_status": "active",
            "worktree_branch": "feature",
            "worktree_path": str(worktree_path),
            "worktree_base_ref": "HEAD",
            "worktree_origin_root": str(tmp_path / "project"),
            "worktree_head": "abc123",
            "worktree_created_at": "2026-01-01T00:00:00Z",
        },
    )
    store.append(thread_id, "thread.cwd_updated", cwd=str(worktree_path))

    active = store.thread_digest(thread_id)
    assert active["worktree_status"] == "active"
    assert active["worktree_branch"] == "feature"
    assert active["worktree_path"] == str(worktree_path)
    assert active["latest_cwd"] == str(worktree_path)

    store.update_thread_metadata(
        thread_id,
        updates={
            "worktree_status": "deleted",
            "worktree_deleted_head": "def456",
            "worktree_deleted_status": " M file.py",
        },
    )

    deleted = store.thread_digest(thread_id)
    assert deleted["worktree_status"] == "deleted"
    assert deleted["worktree_deleted_head"] == "def456"
    assert deleted["worktree_deleted_status"] == " M file.py"


def test_agent_view_deleted_threads_remain_in_general_thread_lists(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    keep = store.create_thread("Keep")
    hidden = store.create_thread("Hide")

    store.append(hidden, "thread.agent_view_deleted")

    assert [thread["thread_id"] for thread in store.list_threads()] == [hidden, keep]
    digest = store.thread_digest(hidden)
    assert digest["agent_view_deleted"] is True
    assert digest["agent_view_deleted_at"]


def test_agent_view_join_event_marks_thread_and_restores_hidden_thread(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Join me")

    store.append(thread_id, "thread.agent_view_deleted")
    store.append(thread_id, "thread.agent_view_joined", source="thread_command")

    digest = store.thread_digest(thread_id)
    listed = store.list_threads()[0]

    assert digest["agent_view_joined"] is True
    assert digest["agent_view_joined_at"]
    assert digest["agent_view_source"] == "thread_command"
    assert "agent_view_deleted" not in digest
    assert listed["agent_view_joined"] is True


def test_billing_accumulated_events_update_thread_metadata(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Billing")

    store.append(
        thread_id,
        "thread.billing_accumulated",
        turn_id="turn_1",
        amount="0.0001234",
        currency="USD",
        source="model_response",
    )
    store.append(
        thread_id,
        "thread.billing_accumulated",
        turn_id="turn_2",
        amount="0.0000006",
        currency="USD",
        source="compaction",
    )

    digest = store.thread_digest(thread_id)

    assert digest["billing_currency"] == "USD"
    assert digest["billing_total"] == "0.000124"
    assert digest["billing_totals"] == {"USD": "0.000124"}


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


def test_thread_digest_omits_empty_compaction_summary_item(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Empty compact")
    store.append(thread_id, "item.compaction", turn_id="t1", text="")

    digest = store.thread_digest(thread_id, since_last_compaction=False)

    assert digest["items"] == []


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


def test_read_after_latest_compaction_filters_suffix_events(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Filtered suffix")
    store.append(thread_id, "item.agent_epoch_context", text="old")
    store.append(thread_id, "item.compaction", text="summary")
    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]},
    )
    store.append(thread_id, "item.agent_epoch_context", text="new")

    events, compaction = store.read_after_latest_compaction(
        thread_id,
        event_types={"item.agent_epoch_context"},
    )

    assert compaction is not None
    assert compaction["text"] == "summary"
    assert [event["type"] for event in events] == ["item.agent_epoch_context"]
    assert events[0]["text"] == "new"


def test_latest_event_helpers_are_scoped_to_open_epoch(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Epoch helpers")
    store.append(thread_id, "item.rule_index", text="before")
    store.append(thread_id, "item.agent_epoch_context", text="before")
    store.append(thread_id, "item.compaction", text="summary")

    assert not store.has_event_after_latest_compaction(thread_id, event_types={"item.rule_index"})
    assert store.latest_event_after_latest_compaction(thread_id, event_types={"item.agent_epoch_context"}) is None

    store.append(thread_id, "item.rule_index", text="after")
    store.append(thread_id, "item.agent_epoch_context", text="after")

    latest = store.latest_event_after_latest_compaction(thread_id, event_types={"item.agent_epoch_context"})

    assert store.has_event_after_latest_compaction(thread_id, event_types={"item.rule_index"})
    assert latest is not None
    assert latest["text"] == "after"


def test_history_segment_cache_is_lru_bounded_and_cleared_on_append(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Bounded cache")

    for index in range(HISTORY_SEGMENT_CACHE_MAX_ENTRIES + 5):
        store.read_history_segment(thread_id, event_types={f"item.synthetic_{index}"})

    assert len(store._history_segment_cache) == HISTORY_SEGMENT_CACHE_MAX_ENTRIES

    store.append(
        thread_id,
        "item.user",
        turn_id="t1",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]},
    )

    assert store._history_segment_cache == {}


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
    previous = store.read_history_segment(thread_id, before_event_id=latest.start_event_id)

    assert latest.start_event_id == second_compaction["_event_id"]
    assert [event["turn_id"] for event in latest.events] == ["t2", "t3"]
    assert latest.events[0]["type"] == "item.compaction"
    assert latest.has_more is True
    assert previous.start_event_id == first_compaction["_event_id"]
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
        before_event_id=latest.start_event_id,
        event_types=event_types,
    )

    assert latest.start_event_id == compaction["_event_id"]
    assert latest.has_more is True
    assert oldest.start_event_id == 0
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
        assert store._read_lock_owner(thread_id)

    assert not store._read_lock_owner(thread_id)
    other.append(
        thread_id,
        "item.user",
        turn_id="t3",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "after"}]},
    )


def test_thread_lock_permission_does_not_leak_to_other_contexts(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Locked")
    other = ThreadStore(tmp_path)

    with store.lock_thread(thread_id):
        # Simulate an unrelated asyncio task in the same process. A process-wide
        # token would incorrectly let this write through the held thread lock.
        reset_token = _THREAD_LOCK_CONTEXT.set({})
        try:
            with pytest.raises(ThreadLockedError):
                store.append(
                    thread_id,
                    "item.user",
                    turn_id="t_other_context",
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "other context"}],
                    },
                )
            with pytest.raises(ThreadLockedError):
                other.append(
                    thread_id,
                    "item.user",
                    turn_id="t_other_store",
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "other store"}],
                    },
                )
        finally:
            _THREAD_LOCK_CONTEXT.reset(reset_token)


def test_child_threads_are_stored_in_sqlite_and_listed_by_parent(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    parent = store.create_thread("Parent")
    child = store.create_thread(
        "Workflow node: inspect",
        kind="workflow_node",
        parent_thread_id=parent,
        parent_turn_id="turn_1",
        parent_run_id="run_1",
    )
    store.append(child, "turn.completed", turn_id="turn_child", final_text="done")

    assert (tmp_path / "uv-agent.sqlite3").exists()
    assert not (tmp_path / "threads" / f"{parent}.jsonl").exists()
    assert [thread["thread_id"] for thread in store.list_threads()] == [parent]

    children = store.list_child_threads(parent)

    assert [thread["thread_id"] for thread in children] == [child]
    assert children[0]["kind"] == "workflow_node"
    assert children[0]["parent_turn_id"] == "turn_1"


def test_sqlite_store_does_not_create_jsonl_thread_directories(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    parent = store.create_thread("Parent")
    child = store.create_thread("Workflow node", kind="workflow_node", parent_thread_id=parent)
    store.append(
        parent,
        "item.user",
        turn_id="turn_parent",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "parent"}]},
    )
    store.append(
        child,
        "item.user",
        turn_id="turn_child",
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "child"}]},
    )

    assert store.read(parent)
    assert store.thread_digest(child)["items"] == [{"role": "user", "text": "child"}]
    assert [thread["thread_id"] for thread in store.list_threads()] == [parent]
    assert [thread["thread_id"] for thread in store.list_child_threads(parent)] == [child]
    assert (tmp_path / "uv-agent.sqlite3").exists()
    assert not (tmp_path / "threads").exists()


def test_update_thread_metadata_persists_extra_metadata(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Metadata")

    store.update_thread_metadata(
        thread_id,
        updates={
            "goal_mode": {"enabled": True, "objective": "ship it"},
            "token_ratio": {"visible_units": 40, "output_tokens": 10},
        },
    )
    metadata = ThreadStore(tmp_path).thread_metadata(thread_id)

    assert metadata["token_ratio"] == {"visible_units": 40, "output_tokens": 10}
    assert metadata["goal_mode"]["objective"] == "ship it"
    assert "metadata_json" not in metadata


def test_update_thread_metadata_remover_edits_extra_metadata(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    thread_id = store.create_thread("Metadata")
    store.update_thread_metadata(
        thread_id,
        updates={
            "token_ratio": {"visible_units": 40, "output_tokens": 10},
            "custom_flag": True,
        },
    )

    store.update_thread_metadata(
        thread_id,
        remover=lambda extra: extra.pop("token_ratio", None),
    )
    metadata = ThreadStore(tmp_path).thread_metadata(thread_id)

    assert "token_ratio" not in metadata
    assert metadata["custom_flag"] is True


def test_delete_thread_hard_deletes_runs_events_and_blob_refs(tmp_path: Path) -> None:
    store = ThreadStore(tmp_path)
    blobs = BlobStore(tmp_path)
    run_logs = RunLogStore(tmp_path)
    thread_id = store.create_thread("Delete me")
    store.append(thread_id, "item.user", turn_id="turn_1", item={"type": "message", "role": "user", "content": []})
    run_logs.create_run_record(
        run_id="run_delete",
        code="print('bye')",
        script_args=[],
        cwd=tmp_path,
        timeout_s=None,
        started_at="test",
        thread_id=thread_id,
        turn_id="turn_1",
        script_path=None,
    )
    run_logs.writer("run_delete").write({"type": "run.output", "created_at": "test", "text": "bye"})
    blob = blobs.put_bytes(b"payload")
    blobs.add_ref(blob.blob_id, thread_id=thread_id, owner_type="test", owner_id="owner")

    deleted = store.delete_thread(thread_id, blobs=blobs)

    assert deleted["ok"] is True
    assert deleted["deleted_threads"] == 1
    assert deleted["deleted_events"] >= 2
    assert deleted["deleted_runs"] == 1
    assert deleted["deleted_run_events"] == 1
    assert deleted["deleted_blob_refs"] == 1
    assert deleted["deleted_blobs"] == 1
    assert not blob.path.exists()
    assert store.list_threads() == []
    assert run_logs.get_run("run_delete") is None
    assert run_logs.read_events("run_delete") == []
    with pytest.raises(FileNotFoundError):
        blobs.info(blob.blob_id)

