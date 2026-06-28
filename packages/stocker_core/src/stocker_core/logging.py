"""Structured logging setup shared by desktop and server entry points."""

from __future__ import annotations

import logging
from typing import Any


class _FallbackStructuredLogger:
    """Small stdlib-backed logger with a structlog-like call shape."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, **fields: Any) -> None:
        if fields:
            rendered_fields = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
            self._logger.log(level, "%s %s", event, rendered_fields)
            return
        self._logger.log(level, "%s", event)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def critical(self, event: str, **fields: Any) -> None:
        self._log(logging.CRITICAL, event, **fields)


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> Any:
    """Configure structlog and return a logger.

    The project depends on structlog. The fallback keeps bootstrap diagnostics usable if a
    developer runs scripts before syncing dependencies.
    """

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(message)s")

    try:
        import structlog
    except ModuleNotFoundError:
        fallback = logging.getLogger("stocker")
        fallback.setLevel(numeric_level)
        return _FallbackStructuredLogger(fallback)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("stocker")
