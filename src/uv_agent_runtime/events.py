from __future__ import annotations

import json
import sys
from typing import Any


def emit_event(kind: str, **payload: Any) -> None:
    """Emit a structured event on stdout for the runner or user to inspect."""
    print(json.dumps({"kind": kind, **payload}, ensure_ascii=False), flush=True)
