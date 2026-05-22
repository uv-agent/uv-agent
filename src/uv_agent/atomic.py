from __future__ import annotations

import os
import stat
import time
from pathlib import Path


# On Windows, ``os.replace`` can fail with ``PermissionError`` (WinError 5) when
# the destination file is transiently held open by another process or by another
# thread in the same process. Common offenders are antivirus/EDR realtime
# scanning, cloud-sync clients (OneDrive, Dropbox, ...), file explorer preview
# panes, and concurrent reads inside this app. A short retry loop with backoff
# is enough to ride out almost every case in practice.
_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.5)


def _clear_readonly(path: Path) -> None:
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if not (mode & stat.S_IWUSR):
        try:
            path.chmod(mode | stat.S_IWUSR)
        except OSError:
            pass


def atomic_replace(tmp_path: Path, dest_path: Path) -> None:
    """Rename ``tmp_path`` to ``dest_path`` with retries for Windows quirks.

    Retries ``PermissionError`` (and falls back ``OSError``) a handful of times
    with short backoff, after clearing a read-only attribute on the destination
    if present. Re-raises the final error if every attempt fails so callers can
    surface a real failure.
    """

    last_error: OSError | None = None
    for attempt, delay in enumerate((0.0, *_RETRY_DELAYS_SECONDS)):
        if delay:
            time.sleep(delay)
        try:
            os.replace(tmp_path, dest_path)
            return
        except PermissionError as exc:
            last_error = exc
            _clear_readonly(dest_path)
        except OSError as exc:
            last_error = exc
        if attempt >= len(_RETRY_DELAYS_SECONDS):
            break
    assert last_error is not None
    raise last_error
