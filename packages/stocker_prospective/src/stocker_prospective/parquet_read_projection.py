"""Bounded read projections over immutable prospective Parquet evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParquetReadMetrics:
    files_examined: int
    row_groups_examined: int
    row_groups_read: int
    selected_row_group_rows: int
    input_rows: int
    output_rows: int
    columns_read: tuple[str, ...]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParquetProjection:
    rows: tuple[dict[str, Any], ...]
    metrics: ParquetReadMetrics


class ParquetProjectionLimitExceeded(RuntimeError):
    """The bounded web projection would exceed its configured row budget."""

    def __init__(self, metrics: ParquetReadMetrics) -> None:
        super().__init__("bounded Parquet projection input limit exceeded")
        self.metrics = metrics


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _metadata_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _row_group_overlaps(
    metadata: Any,
    *,
    row_group_index: int,
    timestamp_columns: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> bool:
    row_group = metadata.row_group(row_group_index)
    observed_statistics = False
    for column_index in range(row_group.num_columns):
        column = row_group.column(column_index)
        if column.path_in_schema not in timestamp_columns:
            continue
        statistics = column.statistics
        if statistics is None or not statistics.has_min_max:
            continue
        minimum = _metadata_timestamp(statistics.min)
        maximum = _metadata_timestamp(statistics.max)
        if minimum is None or maximum is None:
            continue
        observed_statistics = True
        if maximum >= start and minimum <= end:
            return True
    return not observed_statistics


def _filter_scalar(data_type: Any, value: datetime) -> object:
    import pyarrow as pa

    if pa.types.is_timestamp(data_type):
        return value
    return value.isoformat()


def read_parquet_window(
    path: str | Path,
    *,
    columns: Iterable[str],
    timestamp_columns: Iterable[str],
    start: datetime,
    end: datetime,
    maximum_input_rows: int,
) -> ParquetProjection:
    """Read projected rows inside one timestamp window with a hard row cap."""

    if maximum_input_rows <= 0:
        raise ValueError("maximum_input_rows must be positive")
    start_utc = _aware_utc(start, label="Parquet window start")
    end_utc = _aware_utc(end, label="Parquet window end")
    if end_utc < start_utc:
        raise ValueError("Parquet window end precedes start")

    import pyarrow.dataset as dataset
    import pyarrow.parquet as parquet

    source = Path(path)
    parquet_file = parquet.ParquetFile(source)  # type: ignore[no-untyped-call]
    available = frozenset(parquet_file.schema_arrow.names)
    projected_columns = tuple(dict.fromkeys(name for name in columns if name in available))
    projected_timestamps = tuple(
        dict.fromkeys(name for name in timestamp_columns if name in available)
    )
    metadata = parquet_file.metadata
    row_groups_examined = metadata.num_row_groups
    selected_row_groups = tuple(
        index
        for index in range(row_groups_examined)
        if _row_group_overlaps(
            metadata,
            row_group_index=index,
            timestamp_columns=projected_timestamps,
            start=start_utc,
            end=end_utc,
        )
    )
    selected_row_group_rows = sum(
        metadata.row_group(index).num_rows for index in selected_row_groups
    )
    empty_metrics = ParquetReadMetrics(
        files_examined=1,
        row_groups_examined=row_groups_examined,
        row_groups_read=len(selected_row_groups),
        selected_row_group_rows=selected_row_group_rows,
        input_rows=0,
        output_rows=0,
        columns_read=projected_columns,
    )
    if not projected_columns or not projected_timestamps or not selected_row_groups:
        return ParquetProjection(rows=(), metrics=empty_metrics)

    parquet_dataset = dataset.dataset(source, format="parquet")  # type: ignore[no-untyped-call]
    predicate = None
    for timestamp_column in projected_timestamps:
        data_type = parquet_dataset.schema.field(timestamp_column).type
        field = dataset.field(timestamp_column)  # type: ignore[attr-defined,no-untyped-call]
        within_window = (
            (field >= _filter_scalar(data_type, start_utc))
            & (field <= _filter_scalar(data_type, end_utc))
        )
        predicate = within_window if predicate is None else predicate | within_window
    assert predicate is not None

    rows: list[dict[str, Any]] = []
    input_rows = 0
    scanner = parquet_dataset.scanner(
        columns=list(projected_columns),
        filter=predicate,
        batch_size=min(8192, maximum_input_rows + 1),
        use_threads=False,
    )
    for batch in scanner.to_batches():
        next_count = input_rows + batch.num_rows
        if next_count > maximum_input_rows:
            metrics = ParquetReadMetrics(
                **{
                    **empty_metrics.model_dump(),
                    "input_rows": next_count,
                }
            )
            raise ParquetProjectionLimitExceeded(metrics)
        input_rows = next_count
        rows.extend(batch.to_pylist())
    metrics = ParquetReadMetrics(
        **{
            **empty_metrics.model_dump(),
            "input_rows": input_rows,
            "output_rows": len(rows),
        }
    )
    return ParquetProjection(rows=tuple(rows), metrics=metrics)


def read_parquet_tail(
    path: str | Path,
    *,
    columns: Iterable[str],
    maximum_rows: int,
    maximum_input_rows: int,
) -> ParquetProjection:
    """Read only enough trailing row groups for an explicit raw-detail request."""

    if maximum_rows <= 0 or maximum_input_rows <= 0:
        raise ValueError("Parquet tail limits must be positive")

    import pyarrow.parquet as parquet

    source = Path(path)
    parquet_file = parquet.ParquetFile(source)  # type: ignore[no-untyped-call]
    available = frozenset(parquet_file.schema_arrow.names)
    projected_columns = tuple(dict.fromkeys(name for name in columns if name in available))
    metadata = parquet_file.metadata
    rows: list[dict[str, Any]] = []
    input_rows = 0
    groups_read = 0
    for row_group_index in reversed(range(metadata.num_row_groups)):
        row_group_rows = metadata.row_group(row_group_index).num_rows
        if input_rows + row_group_rows > maximum_input_rows:
            metrics = ParquetReadMetrics(
                files_examined=1,
                row_groups_examined=metadata.num_row_groups,
                row_groups_read=groups_read + 1,
                selected_row_group_rows=input_rows + row_group_rows,
                input_rows=input_rows + row_group_rows,
                output_rows=len(rows),
                columns_read=projected_columns,
            )
            raise ParquetProjectionLimitExceeded(metrics)
        table = parquet_file.read_row_group(  # type: ignore[no-untyped-call]
            row_group_index,
            columns=list(projected_columns),
        )
        input_rows += table.num_rows
        groups_read += 1
        needed = maximum_rows - len(rows)
        start_index = max(0, table.num_rows - needed)
        rows = [*table.slice(start_index).to_pylist(), *rows]
        if len(rows) >= maximum_rows:
            rows = rows[-maximum_rows:]
            break
    return ParquetProjection(
        rows=tuple(rows),
        metrics=ParquetReadMetrics(
            files_examined=1,
            row_groups_examined=metadata.num_row_groups,
            row_groups_read=groups_read,
            selected_row_group_rows=input_rows,
            input_rows=input_rows,
            output_rows=len(rows),
            columns_read=projected_columns,
        ),
    )


__all__ = [
    "ParquetProjection",
    "ParquetProjectionLimitExceeded",
    "ParquetReadMetrics",
    "read_parquet_tail",
    "read_parquet_window",
]
