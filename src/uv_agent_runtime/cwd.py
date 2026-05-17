from __future__ import annotations

import os
from pathlib import Path

from .events import emit_event


def enter_dir(path: str | Path) -> Path:
    """Change the script cwd and notify the host to load applicable rules."""
    resolved = Path(path).expanduser().resolve()
    os.chdir(resolved)
    emit_event("enter_dir", cwd=str(resolved))
    return resolved
