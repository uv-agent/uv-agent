from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import uuid4


@dataclass(frozen=True)
class HeldFileLock:
    """A small cross-process lock represented by an exclusive lock file."""

    path: Path
    token: str
    fd: int


@contextmanager
def file_lock(
    path: str | Path,
    *,
    timeout_s: float | None = 300.0,
    stale_after_s: float | None = 3600.0,
) -> Iterator[HeldFileLock]:
    """Acquire an exclusive lock file and remove it on exit.

    The helper intentionally uses only ``O_CREAT | O_EXCL`` so it works from
    both uv-agent host code and managed runtime scripts without platform-only
    dependencies. It is meant for coarse project-state mutations such as
    bootstrapping or editing the shared run_python uv environment.
    """

    lock = acquire_file_lock(path, timeout_s=timeout_s, stale_after_s=stale_after_s)
    try:
        yield lock
    finally:
        release_file_lock(lock)


def acquire_file_lock(
    path: str | Path,
    *,
    timeout_s: float | None = 300.0,
    stale_after_s: float | None = 3600.0,
) -> HeldFileLock:
    """Acquire ``path`` as a cross-process lock, waiting up to ``timeout_s``."""

    lock_path = Path(path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
    token = uuid4().hex
    delay = 0.02
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if stale_after_s is not None:
                _remove_stale_lock(lock_path, stale_after_s=stale_after_s)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            sleep_for = delay
            if deadline is not None:
                sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)
            delay = min(delay * 1.5, 0.5)
            continue

        lock = HeldFileLock(path=lock_path, token=token, fd=fd)
        try:
            os.write(fd, _lock_payload(token).encode("utf-8"))
        except BaseException:
            os.close(fd)
            try:
                lock_path.unlink()
            except OSError:
                pass
            raise
        return lock


def release_file_lock(lock: HeldFileLock) -> None:
    """Release a lock acquired by :func:`acquire_file_lock`."""

    owns_path = _read_lock_token(lock.path) == lock.token
    try:
        os.close(lock.fd)
    except OSError:
        pass
    if not owns_path:
        return
    try:
        lock.path.unlink()
    except FileNotFoundError:
        pass


def _lock_payload(token: str) -> str:
    return json.dumps(
        {
            "token": token,
            "pid": os.getpid(),
            "created_at": time.time(),
        },
        separators=(",", ":"),
    )


def _read_lock_token(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    token = data.get("token")
    return token if isinstance(token, str) else None


def _remove_stale_lock(path: Path, *, stale_after_s: float) -> None:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return
    if age < stale_after_s:
        return
    try:
        path.unlink()
    except OSError:
        # Another process may still have the file open or may have won the race.
        return
