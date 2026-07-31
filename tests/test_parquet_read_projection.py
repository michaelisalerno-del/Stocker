from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stocker_prospective.parquet_read_projection import (
    ParquetProjectionLimitExceeded,
    read_parquet_tail,
    read_parquet_window,
)


def enlarged_partition(path: Path) -> tuple[datetime, int]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    row_groups = 40
    rows_per_group = 50
    for index in range(row_groups * rows_per_group):
        observed = start + timedelta(hours=index // rows_per_group, seconds=index)
        rows.append(
            {
                "event_id": f"event-{index:05d}",
                "received_timestamp_utc": observed.isoformat(),
                "bid": 100.0 + index / 10_000,
                "ask": 100.1 + index / 10_000,
                "ignored_payload": "x" * 2048,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(rows),
        path,
        row_group_size=rows_per_group,
        compression=None,
    )
    return start, row_groups


def test_window_projection_reads_only_overlapping_row_groups_and_columns(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "enlarged.parquet"
    start, total_row_groups = enlarged_partition(partition)
    requested_start = start + timedelta(hours=21)
    requested_end = requested_start + timedelta(minutes=30)

    projection = read_parquet_window(
        partition,
        columns=("event_id", "received_timestamp_utc", "bid", "ask"),
        timestamp_columns=("received_timestamp_utc",),
        start=requested_start,
        end=requested_end,
        maximum_input_rows=100,
    )

    assert projection.rows
    assert projection.metrics.row_groups_examined == total_row_groups
    assert projection.metrics.row_groups_read == 1
    assert projection.metrics.row_groups_read < projection.metrics.row_groups_examined
    assert projection.metrics.input_rows <= 50
    assert projection.metrics.columns_read == (
        "event_id",
        "received_timestamp_utc",
        "bid",
        "ask",
    )
    assert all("ignored_payload" not in row for row in projection.rows)


def test_window_projection_fails_before_exceeding_python_row_budget(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "enlarged.parquet"
    start, _ = enlarged_partition(partition)

    with pytest.raises(ParquetProjectionLimitExceeded) as blocked:
        read_parquet_window(
            partition,
            columns=("event_id", "received_timestamp_utc"),
            timestamp_columns=("received_timestamp_utc",),
            start=start + timedelta(hours=5),
            end=start + timedelta(hours=5, minutes=30),
            maximum_input_rows=20,
        )

    assert blocked.value.metrics.selected_row_group_rows == 50
    assert blocked.value.metrics.input_rows == 0
    assert blocked.value.metrics.output_rows == 0


def test_window_projection_rejects_oversized_overlapping_row_group_before_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = tmp_path / "oversized-row-group.parquet"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "event_id": f"event-{index:05d}",
            "received_timestamp_utc": (
                start + timedelta(minutes=5) if index == 500 else start + timedelta(days=1)
            ).isoformat(),
        }
        for index in range(1_000)
    ]
    pq.write_table(
        pa.Table.from_pylist(rows),
        partition,
        row_group_size=1_000,
        compression=None,
    )

    def scanner_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("oversized row group reached the dataset scanner")

    monkeypatch.setattr("pyarrow.dataset.dataset", scanner_must_not_run)

    with pytest.raises(ParquetProjectionLimitExceeded) as blocked:
        read_parquet_window(
            partition,
            columns=("event_id", "received_timestamp_utc"),
            timestamp_columns=("received_timestamp_utc",),
            start=start,
            end=start + timedelta(minutes=10),
            maximum_input_rows=100,
        )

    assert blocked.value.metrics.selected_row_group_rows == 1_000
    assert blocked.value.metrics.input_rows == 0
    assert blocked.value.metrics.output_rows == 0


def test_explicit_tail_projection_reads_only_the_last_required_row_group(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "enlarged.parquet"
    _, total_row_groups = enlarged_partition(partition)

    projection = read_parquet_tail(
        partition,
        columns=("event_id", "bid"),
        maximum_rows=5,
        maximum_input_rows=100,
    )

    assert projection.metrics.row_groups_examined == total_row_groups
    assert projection.metrics.row_groups_read == 1
    assert projection.metrics.input_rows == 50
    assert projection.metrics.output_rows == 5
    assert projection.rows[0]["event_id"] == "event-01995"
    assert projection.rows[-1]["event_id"] == "event-01999"
