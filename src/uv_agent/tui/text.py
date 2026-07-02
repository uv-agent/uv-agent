from __future__ import annotations


def repair_utf16_surrogates(text: str) -> str:
    """Combine UTF-16 surrogate pairs and replace orphaned surrogate code units."""

    if not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        return text
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
