from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from uv_agent.state_db import connect_state_db
from uv_agent.time import utc_now_iso

PluginScope = Literal["global", "project", "thread"]


@dataclass(frozen=True)
class PluginStorage:
    """Core-managed SQLite storage facade for one plugin."""

    plugin_id: str
    project_data_dir: Path
    global_data_dir: Path
    indexes: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def global_kv(self) -> "PluginKV":
        return PluginKV(self.plugin_id, self.global_data_dir, "global", "")

    def project_kv(self) -> "PluginKV":
        return PluginKV(self.plugin_id, self.project_data_dir, "project", "")

    def thread_kv(self, thread_id: str) -> "PluginKV":
        return PluginKV(self.plugin_id, self.project_data_dir, "thread", thread_id)

    def global_collection(self, name: str) -> "PluginCollection":
        return PluginCollection(self.plugin_id, self.global_data_dir, "global", "", name, self.indexes.get(name, ()))

    def project_collection(self, name: str) -> "PluginCollection":
        return PluginCollection(self.plugin_id, self.project_data_dir, "project", "", name, self.indexes.get(name, ()))

    def thread_collection(self, thread_id: str, name: str) -> "PluginCollection":
        return PluginCollection(self.plugin_id, self.project_data_dir, "thread", thread_id, name, self.indexes.get(name, ()))


@dataclass(frozen=True)
class PluginKV:
    plugin_id: str
    data_dir: Path
    scope: PluginScope
    scope_id: str

    def get(self, key: str, default: Any = None) -> Any:
        with connect_state_db(self.data_dir) as db:
            row = db.execute(
                """
                SELECT value_json FROM plugin_kv
                WHERE plugin_id = ? AND scope = ? AND scope_id = ? AND key = ?
                """,
                (self.plugin_id, self.scope, self.scope_id, str(key)),
            ).fetchone()
        return _loads(row["value_json"]) if row is not None else default

    def set(self, key: str, value: Any) -> dict[str, Any]:
        now = utc_now_iso()
        value_json = _dumps(value)
        with connect_state_db(self.data_dir) as db:
            db.execute(
                """
                INSERT INTO plugin_kv(plugin_id, scope, scope_id, key, value_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(plugin_id, scope, scope_id, key)
                DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                """,
                (self.plugin_id, self.scope, self.scope_id, str(key), value_json, now),
            )
        return {"key": str(key), "updated_at": now}

    def delete(self, key: str) -> dict[str, Any]:
        with connect_state_db(self.data_dir) as db:
            cursor = db.execute(
                """
                DELETE FROM plugin_kv
                WHERE plugin_id = ? AND scope = ? AND scope_id = ? AND key = ?
                """,
                (self.plugin_id, self.scope, self.scope_id, str(key)),
            )
        return {"key": str(key), "deleted": cursor.rowcount > 0}

    def list_prefix(self, prefix: str = "") -> list[dict[str, Any]]:
        pattern = str(prefix) + "%"
        with connect_state_db(self.data_dir) as db:
            rows = db.execute(
                """
                SELECT key, value_json, updated_at FROM plugin_kv
                WHERE plugin_id = ? AND scope = ? AND scope_id = ? AND key LIKE ?
                ORDER BY key ASC
                """,
                (self.plugin_id, self.scope, self.scope_id, pattern),
            ).fetchall()
        return [
            {"key": row["key"], "value": _loads(row["value_json"]), "updated_at": row["updated_at"]}
            for row in rows
        ]

    def update_json(self, key: str, patch: Mapping[str, Any]) -> Any:
        current = self.get(key, {})
        if not isinstance(current, dict):
            current = {}
        current.update(dict(patch))
        self.set(key, current)
        return current


@dataclass(frozen=True)
class PluginCollection:
    plugin_id: str
    data_dir: Path
    scope: PluginScope
    scope_id: str
    name: str
    index_fields: Iterable[str] = ()

    def put(self, doc_id: str, document: Mapping[str, Any]) -> dict[str, Any]:
        doc_id = str(doc_id)
        body = dict(document)
        now = utc_now_iso()
        with connect_state_db(self.data_dir) as db:
            db.execute(
                """
                INSERT INTO plugin_documents(plugin_id, scope, scope_id, collection, doc_id, body_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plugin_id, scope, scope_id, collection, doc_id)
                DO UPDATE SET body_json = excluded.body_json, updated_at = excluded.updated_at
                """,
                (self.plugin_id, self.scope, self.scope_id, self.name, doc_id, _dumps(body), now),
            )
            db.execute(
                """
                DELETE FROM plugin_document_indexes
                WHERE plugin_id = ? AND scope = ? AND scope_id = ? AND collection = ? AND doc_id = ?
                """,
                (self.plugin_id, self.scope, self.scope_id, self.name, doc_id),
            )
            for field in self.index_fields:
                value = _extract_index_value(body, str(field))
                if value is None:
                    continue
                db.execute(
                    """
                    INSERT INTO plugin_document_indexes(plugin_id, scope, scope_id, collection, field, value, doc_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self.plugin_id, self.scope, self.scope_id, self.name, str(field), value, doc_id),
                )
        return {"doc_id": doc_id, "updated_at": now}

    def get(self, doc_id: str, default: Any = None) -> Any:
        with connect_state_db(self.data_dir) as db:
            row = db.execute(
                """
                SELECT body_json FROM plugin_documents
                WHERE plugin_id = ? AND scope = ? AND scope_id = ? AND collection = ? AND doc_id = ?
                """,
                (self.plugin_id, self.scope, self.scope_id, self.name, str(doc_id)),
            ).fetchone()
        return _loads(row["body_json"]) if row is not None else default

    def delete(self, doc_id: str) -> dict[str, Any]:
        doc_id = str(doc_id)
        with connect_state_db(self.data_dir) as db:
            cursor = db.execute(
                """
                DELETE FROM plugin_documents
                WHERE plugin_id = ? AND scope = ? AND scope_id = ? AND collection = ? AND doc_id = ?
                """,
                (self.plugin_id, self.scope, self.scope_id, self.name, doc_id),
            )
            db.execute(
                """
                DELETE FROM plugin_document_indexes
                WHERE plugin_id = ? AND scope = ? AND scope_id = ? AND collection = ? AND doc_id = ?
                """,
                (self.plugin_id, self.scope, self.scope_id, self.name, doc_id),
            )
        return {"doc_id": doc_id, "deleted": cursor.rowcount > 0}

    def list(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with connect_state_db(self.data_dir) as db:
            rows = db.execute(
                """
                SELECT doc_id, body_json, updated_at FROM plugin_documents
                WHERE plugin_id = ? AND scope = ? AND scope_id = ? AND collection = ?
                ORDER BY updated_at DESC, doc_id ASC
                LIMIT ? OFFSET ?
                """,
                (self.plugin_id, self.scope, self.scope_id, self.name, max(1, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [
            {"doc_id": row["doc_id"], "body": _loads(row["body_json"]), "updated_at": row["updated_at"]}
            for row in rows
        ]

    def query_index(self, field: str, value: Any, *, limit: int = 100) -> list[dict[str, Any]]:
        index_value = _json_scalar(value)
        if index_value is None:
            return []
        with connect_state_db(self.data_dir) as db:
            rows = db.execute(
                """
                SELECT d.doc_id, d.body_json, d.updated_at
                FROM plugin_document_indexes i
                JOIN plugin_documents d
                  ON d.plugin_id = i.plugin_id
                 AND d.scope = i.scope
                 AND d.scope_id = i.scope_id
                 AND d.collection = i.collection
                 AND d.doc_id = i.doc_id
                WHERE i.plugin_id = ? AND i.scope = ? AND i.scope_id = ?
                  AND i.collection = ? AND i.field = ? AND i.value = ?
                ORDER BY d.updated_at DESC, d.doc_id ASC
                LIMIT ?
                """,
                (self.plugin_id, self.scope, self.scope_id, self.name, str(field), index_value, max(1, int(limit))),
            ).fetchall()
        return [
            {"doc_id": row["doc_id"], "body": _loads(row["body_json"]), "updated_at": row["updated_at"]}
            for row in rows
        ]


def indexes_from_storage_schema(schema: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    if not isinstance(schema, Mapping):
        return {}
    collections = schema.get("collections")
    if not isinstance(collections, Mapping):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for name, value in collections.items():
        if not isinstance(value, Mapping):
            continue
        indexes = value.get("indexes") or []
        fields: list[str] = []
        if isinstance(indexes, list):
            for item in indexes:
                if isinstance(item, str):
                    fields.append(item)
                elif isinstance(item, Mapping) and isinstance(item.get("field"), str):
                    fields.append(str(item["field"]))
        out[str(name)] = tuple(fields)
    return out


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _loads(value: str) -> Any:
    return json.loads(value)


def _extract_index_value(document: Mapping[str, Any], field: str) -> str | None:
    current: Any = document
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return _json_scalar(current)


def _json_scalar(value: Any) -> str | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return _dumps(value)
    return None
