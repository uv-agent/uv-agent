from __future__ import annotations

import logging

from uv_agent.config import LoggingConfig
from uv_agent.logging_config import active_log_file, close_logging, configure_logging, log_file_path, parse_log_level


def test_configure_logging_writes_uv_agent_log_file(tmp_path):
    log_path = configure_logging(
        tmp_path,
        LoggingConfig(level="DEBUG", file_enabled=True, console_enabled=False, max_bytes=1024, backup_count=1),
    )

    logger = logging.getLogger("uv_agent.tests.logging")
    logger.debug("debug log system smoke")
    for handler in logging.getLogger("uv_agent").handlers:
        handler.flush()

    assert log_path == log_file_path(tmp_path)
    assert active_log_file() == log_path
    assert log_path.exists()
    assert "debug log system smoke" in log_path.read_text(encoding="utf-8")

    close_logging()
    assert active_log_file() is None


def test_parse_log_level_accepts_names_numbers_and_fallbacks():
    assert parse_log_level("debug") == logging.DEBUG
    assert parse_log_level("40") == logging.ERROR
    assert parse_log_level("not-a-level") == logging.INFO
