from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

from textual.css.query import NoMatches
from textual.widgets import TextArea
from watchfiles import Change, watch

from uv_agent.mcp_config import discover_mcp_servers
from uv_agent.skills import discover_skills
from uv_agent.tui.formatting import short_thread
from uv_agent.tui.state import PickerItem


CODE_FILE_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".gd",
    ".go",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".lock",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".scss",
    ".toml",
    ".tsx",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
IGNORED_MENTION_DIRS = {
    ".code-search",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-agent",
    ".venv",
    "__pycache__",
    "__pypackages__",
    "build",
    "coverage",
    "dist",
    "env",
    "htmlcov",
    "node_modules",
    "out",
    "site-packages",
    "target",
    "tmp",
    "vendor",
    "venv",
}
MAX_MENTION_ITEMS = 300
MENTION_SCAN_BATCH_SIZE = 50
MENTION_SCAN_DIRECTORY_LIMIT = 20000
MENTION_SCAN_FILE_LIMIT = 100000
MENTION_WATCH_DEBOUNCE_MS = 1000
MENTION_WATCH_POLL_DELAY_MS = 2000


class MentionMixin:
    def _maybe_open_mention_picker(self, composer: TextArea, *, previous: str, current: str) -> None:
        if len(current) <= len(previous):
            return
        trigger = self._mention_trigger_at_cursor(composer)
        if trigger is None:
            return
        inserted = current[len(previous) :]
        if not inserted or not trigger.endswith(inserted):
            return
        kind = "thread" if trigger == "@@" else "file"
        self._open_mention_picker(kind)

    def _mention_trigger_at_cursor(self, composer: TextArea) -> str | None:
        row, column = composer.cursor_location
        lines = composer.text.split("\n")
        if row >= len(lines):
            return None
        prefix = lines[row][:column]
        if prefix.endswith("@@"):
            return "@@"
        if prefix.endswith("@"):
            return "@"
        return None

    def _open_mention_picker(self, kind: str) -> None:
        try:
            composer = self.query_one("#composer", TextArea)
        except NoMatches:
            return
        expected_triggers = {"thread": ("@@",), "file": ("@",)}.get(kind, ("@",))
        if self._mention_trigger_at_cursor(composer) not in expected_triggers:
            return
        if kind == "thread":
            self._open_thread_mention_picker()
            return
        self._open_file_mention_picker()

    def _open_file_mention_picker(self) -> None:
        title, items, subtitle = self._mention_picker_items("file")
        self._open_picker(
            title,
            items,
            self._choose_file_mention,
            subtitle=subtitle,
            mention_kind="file",
            mention_items=self._mention_picker_items,
        )
        self._start_file_mention_scan()

    def _open_thread_mention_picker(self) -> None:
        title, items, subtitle = self._mention_picker_items("thread")
        self._open_picker(
            title,
            items,
            self._choose_thread_mention,
            subtitle=subtitle,
            mention_kind="thread",
            mention_items=self._mention_picker_items,
            initial_filter="@",
        )
        self._start_thread_mention_scan()

    def _mention_picker_items(self, kind: str) -> tuple[str, list[PickerItem], str]:
        if kind == "thread":
            return (
                self._text("mention_threads"),
                self._mention_thread_cache.items,
                self._mention_cache_subtitle("thread"),
            )
        return (
            self._text("mention_files"),
            self._mention_file_cache.items,
            self._mention_cache_subtitle("file"),
        )

    def _mention_cache_subtitle(self, kind: str) -> str:
        if kind == "thread":
            hint = self._text("mention_threads_hint")
            cache = self._mention_thread_cache
        else:
            hint = self._text("mention_files_hint")
            cache = self._mention_file_cache
        if cache.worker is not None and not cache.complete:
            return f"{hint} · {self._text('mention_scanning')}"
        if cache.complete:
            return f"{hint} · {self._text('mention_cached')}"
        return hint

    def _thread_mention_items(self) -> list[PickerItem]:
        threads = self.engine.thread_store.list_threads()
        items = []
        for thread in threads:
            thread_id = str(thread.get("thread_id") or "")
            title = str(thread.get("title") or self._text("new_thread"))
            last_text = str(thread.get("last_text") or "").replace("\n", " ")
            if len(last_text) > 120:
                last_text = last_text[:117].rstrip() + "..."
            marker = f"{self._text('current')} " if thread_id == self.thread_id else ""
            items.append(
                PickerItem(
                    id=thread_id,
                    title=f"{marker}{title}",
                    description=last_text or self._text("no_messages"),
                    meta=f"{short_thread(thread_id)} · {thread.get('turn_count', 0)} {self._text('turns')}",
                )
            )
        return items

    def _file_mention_items(self) -> list[PickerItem]:
        return list(self._iter_file_mention_items(self.project_root.resolve(), generation=None))

    def _start_file_mention_scan(self) -> None:
        cache = self._mention_file_cache
        if cache.complete and not self._mention_file_cache_dirty:
            self._refresh_active_mention_panel("file", cache.generation)
            return
        if cache.worker is not None and not cache.complete:
            if not cache.worker.is_finished:
                return
            cache.worker = None
        cache.generation += 1
        cache.complete = False
        self._mention_file_cache_dirty = False
        generation = cache.generation
        cache.worker = self.run_worker(
            lambda: self._scan_file_mentions_worker(generation),
            name="mention-files",
            group="mention-files",
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )
        self._refresh_active_mention_panel("file", generation)

    def _scan_file_mentions_worker(self, generation: int) -> None:
        root = self.project_root.resolve()
        items: list[PickerItem] = []
        pending: list[PickerItem] = []
        try:
            for item in self._iter_file_mention_items(root, generation=generation):
                if generation != self._mention_file_cache.generation:
                    return
                items.append(item)
                pending.append(item)
                if len(pending) >= MENTION_SCAN_BATCH_SIZE:
                    batch = list(items)
                    self.call_from_thread(self._apply_file_mention_scan_update, generation, batch, False)
                    pending.clear()
        finally:
            self.call_from_thread(self._apply_file_mention_scan_update, generation, items, True)

    def _apply_file_mention_scan_update(self, generation: int, items: list[PickerItem], complete: bool) -> None:
        cache = self._mention_file_cache
        if generation != cache.generation:
            return
        cache.items = list(items)
        cache.complete = complete
        if complete:
            cache.worker = None
            self._start_file_mention_watcher()
        self._refresh_active_mention_panel("file", generation)

    def _start_file_mention_watcher(self) -> None:
        worker = self._mention_file_watcher_worker
        if worker is not None and not worker.is_finished:
            return
        self._mention_file_watcher_stop.clear()
        self._mention_file_watcher_worker = self.run_worker(
            self._watch_file_mentions_worker,
            name="mention-file-watch",
            group="mention-file-watch",
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def _watch_file_mentions_worker(self) -> None:
        root = self.project_root.resolve()
        for changes in watch(
            root,
            watch_filter=self._mention_watch_filter,
            debounce=MENTION_WATCH_DEBOUNCE_MS,
            step=MENTION_WATCH_POLL_DELAY_MS,
            recursive=True,
            ignore_permission_denied=True,
            stop_event=self._mention_file_watcher_stop,
        ):
            if not changes:
                continue
            self.call_from_thread(self._mark_file_mention_cache_dirty)

    def _mention_watch_filter(self, change: Change, path: str) -> bool:
        try:
            relative_parts = Path(path).resolve().relative_to(self.project_root.resolve()).parts
        except (OSError, ValueError):
            return False
        for part in relative_parts[:-1]:
            if part.startswith(".") or part in IGNORED_MENTION_DIRS:
                return False
        name = relative_parts[-1] if relative_parts else ""
        if name.startswith("."):
            return True
        suffix = Path(name).suffix.lower()
        return not suffix or suffix in CODE_FILE_SUFFIXES

    def _mark_file_mention_cache_dirty(self) -> None:
        cache = self._mention_file_cache
        if cache.worker is not None and not cache.worker.is_finished:
            return
        self._mention_file_cache_dirty = True

    def _start_thread_mention_scan(self) -> None:
        cache = self._mention_thread_cache
        if cache.complete:
            self._refresh_active_mention_panel("thread", cache.generation)
            return
        if cache.worker is not None and not cache.complete:
            if not cache.worker.is_finished:
                return
            cache.worker = None
        cache.generation += 1
        cache.complete = False
        generation = cache.generation
        cache.worker = self.run_worker(
            lambda: self._scan_thread_mentions_worker(generation),
            name="mention-threads",
            group="mention-threads",
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )
        self._refresh_active_mention_panel("thread", generation)

    def _start_mention_scan(self, kind: str) -> None:
        if kind == "thread":
            self._start_thread_mention_scan()
        elif kind == "file":
            self._start_file_mention_scan()

    def _scan_thread_mentions_worker(self, generation: int) -> None:
        items: list[PickerItem] = []
        try:
            items = self._thread_mention_items()
        finally:
            if generation == self._mention_thread_cache.generation:
                self.call_from_thread(self._apply_thread_mention_scan_update, generation, items)

    def _apply_thread_mention_scan_update(self, generation: int, items: list[PickerItem]) -> None:
        cache = self._mention_thread_cache
        if generation != cache.generation:
            return
        cache.items = list(items)
        cache.complete = True
        cache.worker = None
        self._refresh_active_mention_panel("thread", generation)

    def _refresh_active_mention_panel(self, kind: str, generation: int) -> None:
        cache = self._mention_thread_cache if kind == "thread" else self._mention_file_cache
        if generation != cache.generation:
            return
        panel = self._active_fullscreen_panel()
        if panel is None or panel.mention_kind != kind:
            return
        panel.update_picker_items(cache.items, subtitle=self._mention_cache_subtitle(kind))

    def _iter_file_mention_items(self, root: Path, *, generation: int | None) -> Any:
        items: list[PickerItem] = []
        directories_seen = 0
        files_seen = 0
        stack = deque([root])
        while stack and len(items) < MAX_MENTION_ITEMS:
            directory = stack.popleft()
            if generation is not None and generation != self._mention_file_cache.generation:
                return
            if directories_seen >= MENTION_SCAN_DIRECTORY_LIMIT or files_seen >= MENTION_SCAN_FILE_LIMIT:
                if generation is None:
                    return
                yield PickerItem(
                    id="",
                    title=self._text("mention_scan_truncated"),
                    description=self._text("mention_scan_truncated_description"),
                )
                return
            try:
                with os.scandir(directory) as entries:
                    children = sorted(entries, key=lambda item: (not item.is_dir(follow_symlinks=False), item.name.casefold()))
            except OSError:
                continue
            directories_seen += 1
            for entry in children:
                if generation is not None and generation != self._mention_file_cache.generation:
                    return
                if len(items) >= MAX_MENTION_ITEMS:
                    break
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    path = Path(entry.path)
                    try:
                        relative = path.relative_to(root)
                    except ValueError:
                        continue
                    mention = relative.as_posix().rstrip("/") + "/"
                    item = PickerItem(
                        id=mention,
                        title=mention,
                        description=(
                            self._text("mention_dot_dir_skipped")
                            if entry.name.startswith(".")
                            else self._text("mention_directory_description")
                        ),
                    )
                    items.append(item)
                    yield item
                    if not entry.name.startswith(".") and entry.name not in IGNORED_MENTION_DIRS:
                        stack.append(path)
                    continue
                try:
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError:
                    continue
                files_seen += 1
                if not is_file:
                    continue
                path = Path(entry.path)
                if path.suffix.lower() not in CODE_FILE_SUFFIXES:
                    continue
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    continue
                mention = relative.as_posix()
                item = PickerItem(
                    id=mention,
                    title=mention,
                    description=self._text("mention_file_description"),
                )
                items.append(item)
                yield item
        return items

    def _mcp_mention_items(self) -> list[PickerItem]:
        items: list[PickerItem] = []
        for server in discover_mcp_servers(self.project_root):
            items.append(
                PickerItem(
                    id=server.name,
                    title=server.name,
                    description=server.description,
                    meta=f"{server.scope} · {server.transport}"
                    + (f" · {server.endpoint}" if server.endpoint else ""),
                )
            )
        return items

    def _skill_mention_items(self) -> list[PickerItem]:
        items: list[PickerItem] = []
        for skill in discover_skills(self.project_root):
            items.append(
                PickerItem(
                    id=skill.name,
                    title=skill.name,
                    description=skill.description,
                    meta=f"{skill.scope} · {skill.path}",
                )
            )
        return items

    def _choose_file_mention(self, path: str) -> None:
        self._insert_mention(f"@{path}", "@")

    def _choose_thread_mention(self, thread_id: str) -> None:
        self._insert_mention(f"@thread:{thread_id}", ("@@", "@"))

    def _choose_mcp_mention(self, name: str) -> None:
        self._insert_mention(f"@mcp:{name}", "")

    def _choose_skill_mention(self, name: str) -> None:
        self._insert_mention(f"@skill:{name}", "")

    def _insert_mention(self, mention: str, triggers: str | tuple[str, ...]) -> None:
        composer = self.query_one("#composer", TextArea)
        row, column = composer.cursor_location
        lines = composer.text.split("\n")
        replacement = mention + " "
        trigger_options = (triggers,) if isinstance(triggers, str) else triggers
        if row < len(lines) and mention.startswith("@thread:") and lines[row][:column].endswith("@"):
            trigger_options = ("@",)
        matched_trigger = next(
            (
                trigger
                for trigger in sorted(trigger_options, key=len, reverse=True)
                if row < len(lines) and lines[row][:column].endswith(trigger)
            ),
            "",
        )
        if matched_trigger:
            composer.replace(
                replacement,
                (row, column - len(matched_trigger)),
                (row, column),
                maintain_selection_offset=False,
            )
        else:
            end_trigger = next(
                (
                    trigger
                    for trigger in sorted(trigger_options, key=len, reverse=True)
                    if composer.text.endswith(trigger)
                ),
                "",
            )
            if end_trigger:
                composer.load_text(composer.text[: -len(end_trigger)] + replacement)
                composer.cursor_location = composer.document.end
            else:
                composer.insert(replacement)
        self._last_composer_text = composer.text
        self._resize_composer()
        composer.focus()
