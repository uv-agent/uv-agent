from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

from uv_agent.config import LoggingConfig


APP_LOGGER_NAME = "uv_agent"
DEFAULT_LOG_FILENAME = "uv-agent.log"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_UV_AGENT_HANDLER_ATTR = "_uv_agent_managed_handler"
_UV_AGENT_LOG_PATH_ATTR = "_uv_agent_log_path"


def log_file_path(data_dir: Path, *, filename: str = DEFAULT_LOG_FILENAME) -> Path:
    """Return the main project log file path for a state directory."""

    return data_dir.resolve() / "log" / filename


def configure_logging(
    data_dir: Path,
    config: LoggingConfig | None = None,
    *,
    level_override: str | int | None = None,
    console: bool | None = None,
    console_stream: TextIO | None = None,
) -> Path:
    """Configure uv-agent's stdlib logging namespace.

    Logging is global process state, while uv-agent engines are short-lived in
    many tests and in workflow-node subprocesses.  To avoid duplicated records
    or leaked file descriptors, every call replaces handlers previously created
    by this function and leaves unrelated user/test handlers alone.
    """

    config = config or LoggingConfig()
    level = parse_log_level(level_override if level_override is not None else config.level)
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)
    app_logger.propagate = False

    close_logging()

    path = log_file_path(data_dir)
    formatter = logging.Formatter(_LOG_FORMAT)
    if config.file_enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=max(0, int(config.max_bytes)),
            backupCount=max(0, int(config.backup_count)),
            encoding="utf-8",
            delay=True,
        )
        setattr(file_handler, _UV_AGENT_HANDLER_ATTR, True)
        setattr(file_handler, _UV_AGENT_LOG_PATH_ATTR, str(path))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        app_logger.addHandler(file_handler)

    if config.console_enabled if console is None else console:
        console_handler = logging.StreamHandler(console_stream)
        setattr(console_handler, _UV_AGENT_HANDLER_ATTR, True)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        app_logger.addHandler(console_handler)

    # Keep plugin loggers aligned with the app-level override while preserving
    # their per-plugin file handlers.  Individual plugin code may still choose a
    # stricter level if it has a reason to be quieter.
    logging.getLogger("uv_agent.plugins").setLevel(level)
    return path


def close_logging() -> None:
    """Flush and close handlers installed by :func:`configure_logging`."""

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    for handler in list(app_logger.handlers):
        if getattr(handler, _UV_AGENT_HANDLER_ATTR, False):
            app_logger.removeHandler(handler)
            handler.close()


def active_log_file() -> Path | None:
    """Return the active uv-agent log file, if file logging is configured."""

    for handler in logging.getLogger(APP_LOGGER_NAME).handlers:
        value = getattr(handler, _UV_AGENT_LOG_PATH_ATTR, None)
        if isinstance(value, str) and value:
            return Path(value)
    return None


def parse_log_level(value: str | int | None) -> int:
    """Normalize user-facing log levels to ``logging`` constants."""

    if isinstance(value, int):
        return value
    if value is None:
        return logging.INFO
    normalized = str(value).strip()
    if not normalized:
        return logging.INFO
    if normalized.isdigit():
        return int(normalized)
    level = logging.getLevelName(normalized.upper())
    if isinstance(level, int):
        return level
    return logging.INFO
