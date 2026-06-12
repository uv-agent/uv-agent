from __future__ import annotations

import codecs
from dataclasses import dataclass, field
from typing import Any

from uv_agent.runner.run_log import EventWriter
from uv_agent.time import utc_now_iso

STREAM_READ_CHUNK_BYTES = 64 * 1024
OUTPUT_TRUNCATION_MARKER = "\n[uv-agent runner output truncated]\n"
# Coalesce many decoded chunks into a small number of strings without dropping
# bytes. The runner may later truncate at ``max_output_bytes``, but chunk-count
# management must never silently discard output because the model receives the
# final joined stdout/stderr as the tool result.
MAX_STDOUT_PARTS = 256
MAX_STDERR_PARTS = 256


@dataclass
class OutputCapture:
    stdout_parts: list[str] = field(default_factory=list)
    stderr_parts: list[str] = field(default_factory=list)
    structured_events: list[dict[str, Any]] = field(default_factory=list)
    byte_count: int = 0
    truncated: bool = False

    def append_structured_event(self, event: dict[str, Any]) -> None:
        """Add a structured event without dropping semantic runtime output."""

        self.structured_events.append(event)


async def pump_stream(
    *,
    stream_name: str,
    stream,
    writer: EventWriter,
    sink: list[str],
    run_id: str,
    max_output_bytes: int,
    capture: OutputCapture,
) -> None:
    if stream is None:
        return
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = await stream.read(STREAM_READ_CHUNK_BYTES)
        if not chunk:
            break
        if capture.truncated:
            continue

        remaining = max_output_bytes - capture.byte_count
        if len(chunk) > remaining:
            captured = chunk[: max(0, remaining)]
            if captured:
                text = decoder.decode(captured)
                record_output_text(
                    stream_name=stream_name,
                    text=text,
                    writer=writer,
                    sink=sink,
                    run_id=run_id,
                )
                tail = decoder.decode(b"", final=True)
                record_output_text(
                    stream_name=stream_name,
                    text=tail,
                    writer=writer,
                    sink=sink,
                    run_id=run_id,
                )
            capture.byte_count += len(chunk)
            capture.truncated = True
            sink.append(OUTPUT_TRUNCATION_MARKER)
            writer.write(
                {
                    "type": "run.output_truncated",
                    "created_at": utc_now_iso(),
                    "run_id": run_id,
                    "max_output_bytes": max_output_bytes,
                }
            )
            continue

        capture.byte_count += len(chunk)
        text = decoder.decode(chunk)
        record_output_text(
            stream_name=stream_name,
            text=text,
            writer=writer,
            sink=sink,
            run_id=run_id,
        )

    if not capture.truncated:
        tail = decoder.decode(b"", final=True)
        if tail:
            record_output_text(
                stream_name=stream_name,
                text=tail,
                writer=writer,
                sink=sink,
                run_id=run_id,
            )


def record_output_text(
    *,
    stream_name: str,
    text: str,
    writer: EventWriter,
    sink: list[str],
    run_id: str,
) -> None:
    if not text:
        return
    sink.append(text)
    _coalesce_output_parts(
        sink,
        max_parts=MAX_STDOUT_PARTS if stream_name == "stdout" else MAX_STDERR_PARTS,
    )
    writer.write(
        {
            "type": f"run.{stream_name}",
            "created_at": utc_now_iso(),
            "run_id": run_id,
            "text": text,
        }
    )


def _coalesce_output_parts(sink: list[str], *, max_parts: int) -> None:
    """Bound chunk references without changing the joined output text."""

    if len(sink) > max_parts:
        sink[:] = ["".join(sink)]
