"""End-of-session recorder data-quality report contract."""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.contract import claims_boundary


class SessionQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_date: date
    generated_at_utc: datetime
    expected_universe_minutes: int = Field(ge=0)
    level1_coverage: float = Field(ge=0.0, le=1.0)
    tick_by_tick_coverage: float = Field(ge=0.0, le=1.0)
    depth_coverage: float = Field(ge=0.0, le=1.0)
    m1c_checkpoint_coverage: float = Field(ge=0.0, le=1.0)
    group_o_coverage: float = Field(ge=0.0, le=1.0)
    m1c_predictions: int = Field(ge=0)
    raw_threshold_rows: int = Field(ge=0)
    fresh_episodes: int = Field(ge=0)
    a1_outputs: int = Field(ge=0)
    c1_outputs: int = Field(ge=0)
    r1_outputs: int = Field(ge=0)
    option_contracts_requested: int = Field(ge=0)
    option_contracts_resolved: int = Field(ge=0)
    option_quote_coverage: float = Field(ge=0.0, le=1.0)
    median_quote_staleness_seconds: float | None = Field(default=None, ge=0.0)
    data_gaps: int = Field(ge=0)
    reconnects: int = Field(ge=0)
    pacing_errors: int = Field(ge=0)
    capacity_denials: int = Field(ge=0)
    raw_event_partition_hashes: tuple[str, ...]
    complete_shadow_horizons: int = Field(ge=0)
    complete: bool
    claims: dict[str, bool | float]

    @field_validator("generated_at_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("report timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @classmethod
    def create(cls, **values: object) -> SessionQualityReport:
        return cls.model_validate({**values, "claims": claims_boundary()})


def write_session_quality_report(
    path: str | Path,
    report: SessionQualityReport,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def build_session_quality_report(
    *,
    database_path: str | Path,
    run_id: str,
    session_date: date,
    generated_at: datetime,
) -> SessionQualityReport:
    """Build one immutable end-of-session report from recorder metadata only."""

    connection = sqlite3.connect(Path(database_path))
    connection.row_factory = sqlite3.Row
    session = session_date.isoformat()
    expected_seconds = 20 * 390 * 60

    def scalar(query: str, parameters: tuple[object, ...]) -> int:
        row = connection.execute(query, parameters).fetchone()
        return 0 if row is None else int(row[0] or 0)

    def event_coverage(event_types: tuple[str, ...]) -> float:
        placeholders = ",".join("?" for _ in event_types)
        rows = connection.execute(
            f"""
            SELECT symbol, MIN(minimum_timestamp_utc), MAX(maximum_timestamp_utc)
            FROM raw_partition_manifest_v0
            WHERE run_id = ? AND session_date = ?
              AND event_type IN ({placeholders})
              AND symbol IN (
                  SELECT symbol FROM universe_membership
                  WHERE run_id = ? AND cohort = 'anchor_frozen_20'
              )
            GROUP BY symbol
            """,
            (run_id, session, *event_types, run_id),
        ).fetchall()
        seconds = sum(
            min(
                390 * 60,
                max(
                    0.0,
                    (
                        datetime.fromisoformat(str(row[2])) - datetime.fromisoformat(str(row[1]))
                    ).total_seconds(),
                ),
            )
            for row in rows
        )
        return min(1.0, seconds / expected_seconds)

    directions = {
        archetype: scalar(
            """
            SELECT COUNT(*) FROM direction_classification_v0 d
            JOIN m1c_episode_v0 e ON e.episode_id = d.episode_id
            WHERE d.run_id = ? AND e.session_date = ? AND d.archetype = ?
            """,
            (run_id, session, archetype),
        )
        for archetype in ("A1", "C1", "R1")
    }
    requested_options = scalar(
        """
        SELECT COUNT(*) FROM episode_option_contract_v0 c
        JOIN m1c_episode_v0 e ON e.episode_id = c.episode_id
        WHERE c.run_id = ? AND e.session_date = ?
        """,
        (run_id, session),
    )
    resolved_options = scalar(
        """
        SELECT COUNT(*) FROM episode_option_contract_v0 c
        JOIN m1c_episode_v0 e ON e.episode_id = c.episode_id
        WHERE c.run_id = ? AND e.session_date = ?
          AND c.resolution_status = 'recording' AND c.con_id IS NOT NULL
        """,
        (run_id, session),
    )
    quoted_options = scalar(
        """
        SELECT COUNT(*) FROM option_quote_state_v0 q
        JOIN episode_option_contract_v0 c ON c.id = q.option_contract_id
        JOIN m1c_episode_v0 e ON e.episode_id = c.episode_id
        WHERE q.run_id = ? AND e.session_date = ?
        """,
        (run_id, session),
    )
    staleness_rows = connection.execute(
        """
        SELECT q.provider_timestamp_utc, q.received_timestamp_utc
        FROM option_quote_state_v0 q
        JOIN episode_option_contract_v0 c ON c.id = q.option_contract_id
        JOIN m1c_episode_v0 e ON e.episode_id = c.episode_id
        WHERE q.run_id = ? AND e.session_date = ?
          AND q.provider_timestamp_utc IS NOT NULL
        """,
        (run_id, session),
    ).fetchall()
    staleness = [
        max(
            0.0,
            (
                datetime.fromisoformat(str(row[1])) - datetime.fromisoformat(str(row[0]))
            ).total_seconds(),
        )
        for row in staleness_rows
    ]
    partitions = connection.execute(
        """
        SELECT content_hash, complete, gap_count
        FROM raw_partition_manifest_v0
        WHERE run_id = ? AND session_date = ?
        ORDER BY content_hash
        """,
        (run_id, session),
    ).fetchall()
    subscription_rows = connection.execute(
        """
        SELECT cancellation_reason, ibkr_error_codes_json, capacity_denied
        FROM subscription_lifecycle_v0
        WHERE run_id = ? AND date(started_at_utc) = ?
        """,
        (run_id, session),
    ).fetchall()
    error_codes = [
        int(code)
        for row in subscription_rows
        for code in json.loads(str(row["ibkr_error_codes_json"]))
    ]
    predictions = scalar(
        "SELECT COUNT(*) FROM m1c_checkpoint_v0 WHERE run_id = ? AND session_date = ?",
        (run_id, session),
    )
    report = SessionQualityReport.create(
        session_date=session_date,
        generated_at_utc=generated_at,
        expected_universe_minutes=20 * 390,
        level1_coverage=event_coverage(("underlying_level1_quote_event",)),
        tick_by_tick_coverage=event_coverage(
            ("underlying_tick_bidask_event", "underlying_tick_trade_event")
        ),
        depth_coverage=event_coverage(("underlying_depth_event", "underlying_depth_snapshot")),
        m1c_checkpoint_coverage=min(1.0, predictions / (20 * 15)),
        group_o_coverage=min(
            1.0,
            scalar(
                """
                SELECT COUNT(*) FROM group_o_session_context_v0
                WHERE run_id = ? AND signal_session = ? AND quality_status = 'valid'
                """,
                (run_id, session),
            )
            / 20,
        ),
        m1c_predictions=predictions,
        raw_threshold_rows=scalar(
            """
            SELECT COUNT(*) FROM m1c_checkpoint_v0
            WHERE run_id = ? AND session_date = ? AND threshold_passed = 1
            """,
            (run_id, session),
        ),
        fresh_episodes=scalar(
            "SELECT COUNT(*) FROM m1c_episode_v0 WHERE run_id = ? AND session_date = ?",
            (run_id, session),
        ),
        a1_outputs=directions["A1"],
        c1_outputs=directions["C1"],
        r1_outputs=directions["R1"],
        option_contracts_requested=requested_options,
        option_contracts_resolved=resolved_options,
        option_quote_coverage=(
            0.0 if resolved_options == 0 else min(1.0, quoted_options / resolved_options)
        ),
        median_quote_staleness_seconds=(None if not staleness else statistics.median(staleness)),
        data_gaps=sum(int(row["gap_count"]) for row in partitions),
        reconnects=sum(
            str(row["cancellation_reason"] or "") == "data_lost_reconnect"
            for row in subscription_rows
        ),
        pacing_errors=sum(code in {100, 420} for code in error_codes),
        capacity_denials=sum(int(row["capacity_denied"]) for row in subscription_rows),
        raw_event_partition_hashes=tuple(str(row["content_hash"]) for row in partitions),
        complete=bool(partitions) and all(bool(row["complete"]) for row in partitions),
        complete_shadow_horizons=scalar(
            """
            SELECT COUNT(*) FROM shadow_quote_outcome_v0 s
            JOIN m1c_episode_v0 e ON e.episode_id = s.episode_id
            WHERE s.run_id = ? AND e.session_date = ? AND s.valid = 1
            """,
            (run_id, session),
        ),
    )
    connection.close()
    return report


__all__ = [
    "SessionQualityReport",
    "build_session_quality_report",
    "write_session_quality_report",
]
