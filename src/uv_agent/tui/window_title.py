from __future__ import annotations

import os
import sys


WINDOW_TITLE_MAX_LEN = 60


def sanitized_window_title(title: str, *, max_len: int = WINDOW_TITLE_MAX_LEN) -> str:
    cleaned = "".join(ch for ch in title if ch >= " " and ch != "\x7f")
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def write_window_title(title: str) -> None:
    sequence = f"\x1b]0;{title}\x07"
    try:
        stream = sys.__stdout__
        if stream is not None:
            stream.write(sequence)
            stream.flush()
            return
    except Exception:
        pass
    try:
        os.write(1, sequence.encode("utf-8", errors="replace"))
    except Exception:
        pass
