"""Request-scoped operational metrics and structured JSON logging."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, TypedDict


class OperationLogFields(TypedDict):
    sqlite_operations: int
    sqlite_duration_ms: float
    parquet_files_examined: int
    parquet_row_groups_examined: int
    parquet_row_groups_read: int
    parquet_input_rows: int
    parquet_output_rows: int


@dataclass
class RequestOperationMetrics:
    sqlite_operations: int = 0
    sqlite_duration_ms: float = 0.0
    parquet_files_examined: int = 0
    parquet_row_groups_examined: int = 0
    parquet_row_groups_read: int = 0
    parquet_input_rows: int = 0
    parquet_output_rows: int = 0

    def log_fields(self) -> OperationLogFields:
        return {
            "sqlite_operations": self.sqlite_operations,
            "sqlite_duration_ms": round(self.sqlite_duration_ms, 3),
            "parquet_files_examined": self.parquet_files_examined,
            "parquet_row_groups_examined": self.parquet_row_groups_examined,
            "parquet_row_groups_read": self.parquet_row_groups_read,
            "parquet_input_rows": self.parquet_input_rows,
            "parquet_output_rows": self.parquet_output_rows,
        }


_CURRENT_METRICS: ContextVar[RequestOperationMetrics | None] = ContextVar(
    "stocker_prospective_request_operation_metrics",
    default=None,
)


def begin_request_metrics() -> tuple[
    RequestOperationMetrics,
    Token[RequestOperationMetrics | None],
]:
    metrics = RequestOperationMetrics()
    return metrics, _CURRENT_METRICS.set(metrics)


def reset_request_metrics(token: Token[RequestOperationMetrics | None]) -> None:
    _CURRENT_METRICS.reset(token)


def record_sqlite_operation(*, duration_ms: float) -> None:
    metrics = _CURRENT_METRICS.get()
    if metrics is None:
        return
    metrics.sqlite_operations += 1
    metrics.sqlite_duration_ms += max(0.0, duration_ms)


def record_parquet_read(
    *,
    files_examined: int,
    row_groups_examined: int,
    row_groups_read: int,
    input_rows: int,
    output_rows: int,
) -> None:
    metrics = _CURRENT_METRICS.get()
    if metrics is None:
        return
    metrics.parquet_files_examined += max(0, files_examined)
    metrics.parquet_row_groups_examined += max(0, row_groups_examined)
    metrics.parquet_row_groups_read += max(0, row_groups_read)
    metrics.parquet_input_rows += max(0, input_rows)
    metrics.parquet_output_rows += max(0, output_rows)


def structured_log(
    logger: logging.Logger,
    *,
    event: str,
    level: int = logging.INFO,
    exception: Exception | None = None,
    **fields: Any,
) -> None:
    """Emit one JSON object; callers supply only allow-listed operational fields."""

    payload = json.dumps(
        {"event": event, **fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    exception_info = (
        None
        if exception is None
        else (type(exception), exception, exception.__traceback__)
    )
    logger.log(level, payload, exc_info=exception_info)


__all__ = [
    "OperationLogFields",
    "RequestOperationMetrics",
    "begin_request_metrics",
    "record_parquet_read",
    "record_sqlite_operation",
    "reset_request_metrics",
    "structured_log",
]
