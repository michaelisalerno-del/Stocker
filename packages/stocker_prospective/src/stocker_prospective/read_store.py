"""Read-only SQLite projections for the Stocker web process."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stocker_prospective.database import SchemaVersionTooNew
from stocker_prospective.operational_logging import (
    record_parquet_read,
    record_sqlite_operation,
)
from stocker_prospective.operational_state import (
    OperationalStateProjection,
    OperationalThresholds,
    inactive_operational_projection,
    project_operational_state_from_database,
)
from stocker_prospective.parquet_read_projection import (
    ParquetProjectionLimitExceeded,
    ParquetReadMetrics,
    read_parquet_tail,
    read_parquet_window,
)
from stocker_prospective.quiet_state import (
    BOTTOM_5_THRESHOLD,
    BOTTOM_10_THRESHOLD,
    BOTTOM_20_THRESHOLD,
    HIGH_TAIL_THRESHOLD,
    NEUTRAL_CONTROL_SALT,
    NEUTRAL_CONTROL_SAMPLING_FRACTION,
)
from stocker_prospective.virtual_positions import (
    OpeningReversalVirtualPositionV1,
    QuietStateVirtualCaptureV1,
    QuietStateVirtualPositionV1,
)


class _TimedReadConnection(sqlite3.Connection):
    def execute(
        self,
        sql: str,
        parameters: Any = (),
        /,
    ) -> sqlite3.Cursor:
        started = time.perf_counter()
        try:
            return super().execute(sql, parameters)
        finally:
            record_sqlite_operation(duration_ms=(time.perf_counter() - started) * 1_000.0)


def _record_parquet_metrics(metrics: ParquetReadMetrics) -> None:
    record_parquet_read(
        files_examined=metrics.files_examined,
        row_groups_examined=metrics.row_groups_examined,
        row_groups_read=metrics.row_groups_read,
        input_rows=metrics.input_rows,
        output_rows=metrics.output_rows,
    )


class ProspectiveReadStore:
    """A query-only store that opens SQLite with ``mode=ro``."""

    def __init__(self, database_path: str | Path, *, run_id: str | None = None) -> None:
        self.database_path = Path(database_path)
        self.run_id = run_id
        self._anchor: sqlite3.Connection | None = None
        self._snapshot_connection: ContextVar[sqlite3.Connection | None] = ContextVar(
            f"prospective_read_snapshot_{id(self)}",
            default=None,
        )

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.database_path.resolve()}?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=2.0,
            factory=_TimedReadConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 2000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        snapshot = self._snapshot_connection.get()
        if snapshot is not None:
            yield snapshot
            return
        with self._connect() as connection:
            yield connection

    @contextmanager
    def snapshot_transaction(self) -> Iterator[None]:
        """Give all nested projections one consistent read-only WAL snapshot."""

        if self._snapshot_connection.get() is not None:
            yield
            return
        connection = self._connect()
        token = self._snapshot_connection.set(connection)
        try:
            connection.execute("BEGIN")
            # Establish the SQLite snapshot immediately, before section reads.
            connection.execute("SELECT COUNT(*) FROM sqlite_schema").fetchone()
            yield
            connection.rollback()
        finally:
            self._snapshot_connection.reset(token)
            connection.close()

    def open_anchor(self) -> None:
        """Keep WAL coordination files live for the read-only web process."""

        if self._anchor is not None:
            return
        connection = self._connect()
        try:
            supported = {path.name for path in Path(__file__).with_name("migrations").glob("*.sql")}
            applied = {
                str(row["version"])
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            unsupported = tuple(sorted(applied - supported))
            if unsupported:
                raise SchemaVersionTooNew(
                    "blocked_schema_newer_than_supported: " + ",".join(unsupported)
                )
            connection.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        except Exception:
            connection.close()
            raise
        self._anchor = connection

    def close_anchor(self) -> None:
        """Release the process-lifetime read-only WAL anchor."""

        if self._anchor is None:
            return
        self._anchor.close()
        self._anchor = None

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return None if row is None else dict(row)

    @staticmethod
    def _decoded(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for name, value in tuple(result.items()):
            if name.endswith("_json") and isinstance(value, str):
                try:
                    result[name.removesuffix("_json")] = json.loads(value)
                except json.JSONDecodeError:
                    result[name.removesuffix("_json")] = None
                del result[name]
        return result

    def database_health(self) -> dict[str, Any]:
        try:
            with self._connection() as connection:
                result = connection.execute("PRAGMA quick_check").fetchone()
                migration = connection.execute(
                    "SELECT version, applied_at_utc FROM schema_migrations "
                    "ORDER BY version DESC LIMIT 1"
                ).fetchone()
            return {
                "status": "healthy" if result and result[0] == "ok" else "degraded",
                "quick_check": None if result is None else result[0],
                "latest_migration": self._dict(migration),
                "path_exposed": False,
                "mode": "read_only_wal",
            }
        except sqlite3.Error:
            return {
                "status": "unavailable",
                "quick_check": None,
                "latest_migration": None,
                "path_exposed": False,
                "mode": "read_only_wal",
            }

    def read_only_verification(self) -> dict[str, Any]:
        try:
            with self._connection() as connection:
                query_only = int(connection.execute("PRAGMA query_only").fetchone()[0])
                mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
            return {
                "verified": query_only == 1,
                "query_only": query_only == 1,
                "opened_with_mode_ro": True,
                "journal_mode": mode,
            }
        except sqlite3.Error:
            return {
                "verified": False,
                "query_only": False,
                "opened_with_mode_ro": True,
                "journal_mode": None,
            }

    def latest_run(self) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = self._run_row(connection)
        return self._dict(row)

    def _run_row(self, connection: sqlite3.Connection) -> sqlite3.Row | None:
        if self.run_id is not None:
            configured: sqlite3.Row | None = connection.execute(
                "SELECT * FROM prospective_run WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
            return configured
        latest: sqlite3.Row | None = connection.execute(
            "SELECT * FROM prospective_run ORDER BY created_at_utc DESC LIMIT 1"
        ).fetchone()
        return latest

    def _selected_run_id(self) -> str | None:
        with self._connection() as connection:
            row = self._run_row(connection)
        return None if row is None else str(row["run_id"])

    def runtime_projection(self) -> dict[str, Any]:
        with self._connection() as connection:
            run = self._run_row(connection)
            lease_run_id = (
                self.run_id
                if self.run_id is not None
                else (None if run is None else str(run["run_id"]))
            )
            lease = (
                None
                if lease_run_id is None
                else connection.execute(
                    "SELECT * FROM recorder_lease "
                    "WHERE lease_key = 'prospective_recorder' AND run_id = ?",
                    (lease_run_id,),
                ).fetchone()
            )
            if run is None:
                return {
                    "run": None,
                    "session": None,
                    "recorder_lease": self._dict(lease),
                    "last_completed_bar": None,
                    "latest_score": None,
                    "previous_session_context": None,
                    "ibkr_connection": None,
                    "latest_capture": None,
                    "market_data_budget": None,
                    "parallel_source_capture": None,
                    "latest_signal_episode": None,
                    "blockers": [],
                }
            run_id = str(run["run_id"])
            session = connection.execute(
                "SELECT * FROM runtime_session WHERE run_id = ? "
                "ORDER BY opened_at_utc DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            bar = connection.execute(
                "SELECT * FROM underlying_bar WHERE run_id = ? ORDER BY bar_end_utc DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            score = connection.execute(
                "SELECT * FROM model_score WHERE run_id = ? ORDER BY bar_end_utc DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            context = connection.execute(
                "SELECT * FROM previous_session_options_context WHERE run_id = ? "
                "ORDER BY current_session_date DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            connection_event = connection.execute(
                "SELECT * FROM ibkr_connection_event WHERE run_id = ? "
                "AND COALESCE(json_extract(details_json, '$.event_kind'), "
                "'state_transition') = 'state_transition' "
                "ORDER BY id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            capture = connection.execute(
                "SELECT market_data_type, capture_status, budget_status, "
                "target_timestamp_utc FROM option_surface_capture "
                "WHERE run_id = ? ORDER BY target_timestamp_utc DESC, id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            budget = connection.execute(
                "SELECT * FROM market_data_budget_event WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            parallel_capture = connection.execute(
                "SELECT * FROM source_capture_completion WHERE run_id = ? "
                "ORDER BY session_date DESC, id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            episode = connection.execute(
                "SELECT * FROM signal_episode WHERE run_id = ? "
                "ORDER BY crossing_timestamp_utc DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            blockers = connection.execute(
                """
                SELECT blocker_code, component, message, severity
                FROM web_active_runtime_blocker_v0
                WHERE run_id = ? AND component <> 'parallel_feature_validation'
                ORDER BY event_id
                """,
                (run_id,),
            ).fetchall()
        return {
            "run": self._dict(run),
            "session": self._dict(session),
            "recorder_lease": self._dict(lease),
            "last_completed_bar": self._dict(bar),
            "latest_score": self._dict(score),
            "previous_session_context": self._dict(context),
            "ibkr_connection": self._dict(connection_event),
            "latest_capture": self._dict(capture),
            "market_data_budget": self._dict(budget),
            "parallel_source_capture": self._dict(parallel_capture),
            "latest_signal_episode": self._dict(episode),
            "blockers": [dict(row) for row in blockers],
        }

    def universe(self) -> dict[str, list[dict[str, Any]]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return {
                "anchor_frozen_20": [],
                "prospective_external_universe_exploratory": [],
            }
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM universe_membership WHERE run_id = ? ORDER BY cohort, symbol",
                (run_id,),
            ).fetchall()
        anchor = [dict(row) for row in rows if row["cohort"] == "anchor_frozen_20"]
        exploratory = [
            dict(row)
            for row in rows
            if row["cohort"] == "prospective_external_universe_exploratory"
        ]
        return {
            "anchor_frozen_20": anchor,
            "prospective_external_universe_exploratory": exploratory,
        }

    def signals(self, *, limit: int = 200) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT e.*, s.m1_probability, s.frozen_threshold, s.score_label,
                       COUNT(c.id) AS checkpoint_count
                FROM signal_episode e
                LEFT JOIN model_score s
                  ON s.run_id = e.run_id AND s.cohort = e.cohort
                 AND s.symbol = e.symbol
                 AND s.model_bundle_id = e.model_bundle_id
                 AND s.bar_end_utc = e.crossing_timestamp_utc
                LEFT JOIN signal_checkpoint c ON c.signal_episode_id = e.id
                WHERE e.run_id = ?
                GROUP BY e.id
                ORDER BY e.crossing_timestamp_utc DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def signal_detail(self, signal_id: str) -> dict[str, Any] | None:
        run_id = self._selected_run_id()
        if run_id is None:
            return None
        with self._connection() as connection:
            episode = connection.execute(
                """
                SELECT e.*, v.prospective_start_utc, v.app_version, v.git_commit,
                       v.model_artifact_id, v.universe_id, v.source_timestamps_json,
                       v.recorded_at_utc
                FROM signal_episode e
                JOIN evidence_envelope v ON v.id = e.envelope_id
                WHERE e.id = ? AND e.run_id = ?
                """,
                (signal_id, run_id),
            ).fetchone()
            if episode is None:
                return None
            checkpoints = connection.execute(
                """
                SELECT c.*, s.m0_probability, s.m1_probability, s.frozen_threshold,
                       s.feature_schema_hash, s.eligibility, s.rejection_reason,
                       s.score_label
                FROM signal_checkpoint c
                JOIN model_score s ON s.id = c.model_score_id
                WHERE c.signal_episode_id = ?
                ORDER BY c.checkpoint_timestamp_utc
                """,
                (signal_id,),
            ).fetchall()
            feature = connection.execute(
                """
                SELECT f.* FROM feature_snapshot f
                WHERE f.run_id = ? AND f.symbol = ? AND f.bar_end_utc = ?
                ORDER BY f.id DESC LIMIT 1
                """,
                (
                    episode["run_id"],
                    episode["symbol"],
                    episode["crossing_timestamp_utc"],
                ),
            ).fetchone()
            context = connection.execute(
                """
                SELECT * FROM previous_session_options_context
                WHERE run_id = ? ORDER BY current_session_date DESC LIMIT 1
                """,
                (episode["run_id"],),
            ).fetchone()
            underlying_quotes = connection.execute(
                """
                SELECT * FROM underlying_quote WHERE signal_episode_id = ?
                ORDER BY target_timestamp_utc
                """,
                (signal_id,),
            ).fetchall()
            captures = connection.execute(
                """
                SELECT * FROM option_surface_capture WHERE signal_episode_id = ?
                ORDER BY target_timestamp_utc, dte_bucket
                """,
                (signal_id,),
            ).fetchall()
            option_quotes = connection.execute(
                """
                SELECT q.*, c.con_id, c.local_symbol, c.expiry, c.strike, c.right,
                       c.multiplier, c.exchange, c.trading_class, c.dte_bucket,
                       x.target_timestamp_utc, x.capture_status
                FROM option_quote q
                JOIN option_contract c ON c.id = q.option_contract_id
                JOIN option_surface_capture x ON x.id = q.surface_capture_id
                WHERE x.signal_episode_id = ?
                ORDER BY x.target_timestamp_utc, c.dte_bucket, c.strike, c.right
                """,
                (signal_id,),
            ).fetchall()
            option_computations = connection.execute(
                """
                SELECT g.*, c.con_id, c.local_symbol, c.expiry, c.strike, c.right,
                       x.target_timestamp_utc, x.dte_bucket
                FROM option_quote_computation g
                JOIN option_quote q ON q.id = g.option_quote_id
                JOIN option_contract c ON c.id = q.option_contract_id
                JOIN option_surface_capture x ON x.id = q.surface_capture_id
                WHERE x.signal_episode_id = ?
                ORDER BY x.target_timestamp_utc, c.dte_bucket, c.strike, c.right,
                         g.computation_source
                """,
                (signal_id,),
            ).fetchall()
        return {
            "episode": dict(episode),
            "checkpoints": [dict(row) for row in checkpoints],
            "feature_snapshot": self._dict(feature),
            "previous_session_context": self._dict(context),
            "underlying_quotes": [dict(row) for row in underlying_quotes],
            "captures": [dict(row) for row in captures],
            "option_quotes": [dict(row) for row in option_quotes],
            "option_computations": [dict(row) for row in option_computations],
        }

    def shadow(self, *, limit: int = 500) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT s.*, e.crossing_timestamp_utc,
                       COUNT(h.id) AS horizon_count,
                       MAX(h.horizon_minutes) AS latest_horizon_minutes
                FROM shadow_structure s
                JOIN signal_episode e ON e.id = s.signal_episode_id
                LEFT JOIN shadow_horizon_valuation h ON h.shadow_structure_id = s.id
                WHERE s.run_id = ?
                GROUP BY s.id
                ORDER BY e.crossing_timestamp_utc DESC, s.structure_type
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def shadow_detail(self, structure_id: str) -> dict[str, Any] | None:
        run_id = self._selected_run_id()
        if run_id is None:
            return None
        with self._connection() as connection:
            structure = connection.execute(
                """
                SELECT s.*, e.crossing_timestamp_utc
                FROM shadow_structure s
                JOIN signal_episode e ON e.id = s.signal_episode_id
                WHERE s.id = ? AND s.run_id = ?
                """,
                (structure_id, run_id),
            ).fetchone()
            if structure is None:
                return None
            legs = connection.execute(
                """
                SELECT l.*, c.con_id, c.local_symbol, c.expiry, c.strike, c.right,
                       c.multiplier, c.exchange, c.trading_class
                FROM shadow_leg l JOIN option_contract c ON c.id = l.option_contract_id
                WHERE l.shadow_structure_id = ? ORDER BY l.id
                """,
                (structure_id,),
            ).fetchall()
            horizons = connection.execute(
                """
                SELECT * FROM shadow_horizon_valuation
                WHERE shadow_structure_id = ? ORDER BY horizon_minutes
                """,
                (structure_id,),
            ).fetchall()
        return {
            "structure": dict(structure),
            "legs": [dict(row) for row in legs],
            "horizons": [dict(row) for row in horizons],
            "ledger": "quoted_research_ledger",
            "paper_ledger": {"implemented": False, "records": []},
        }

    def audit(self, *, limit: int = 500) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT a.*, e.recorded_at_utc, e.app_version, e.git_commit,
                       e.model_artifact_id, e.universe_id, e.cohort
                FROM audit_event a
                JOIN evidence_envelope e ON e.id = a.envelope_id
                WHERE a.run_id = ?
                ORDER BY a.run_id, a.sequence LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recorder_operational_state(
        self,
        *,
        now: datetime,
        prospective_start_utc: datetime | None = None,
        thresholds: OperationalThresholds | None = None,
    ) -> OperationalStateProjection:
        with self._connection() as connection:
            run = self._run_row(connection)
            run_id = (
                self.run_id
                if self.run_id is not None
                else (None if run is None else str(run["run_id"]))
            )
            if run_id is None:
                return inactive_operational_projection(now=now)
            if prospective_start_utc is not None:
                start = prospective_start_utc
            elif run is not None:
                start = datetime.fromisoformat(str(run["prospective_start_utc"]))
            else:
                return inactive_operational_projection(now=now)
            return project_operational_state_from_database(
                connection,
                run_id=run_id,
                now=now,
                prospective_start_utc=start,
                thresholds=thresholds or OperationalThresholds(),
            )

    def runtime_artifact_verification(self) -> dict[str, Any]:
        run_id = self.run_id or self._selected_run_id()
        if run_id is None:
            return {
                "verified": False,
                "generation": None,
                "expected_artifact_count": 0,
                "items": [],
                "blockers": [],
            }
        with self._connection() as connection:
            state = connection.execute(
                """
                SELECT recorder_generation, expected_artifact_count
                FROM recorder_operational_state_v1
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if state is None:
                return {
                    "verified": False,
                    "generation": None,
                    "expected_artifact_count": 0,
                    "items": [],
                    "blockers": ["RUNTIME_ARTIFACT_EVIDENCE_ABSENT"],
                }
            generation = int(state["recorder_generation"])
            expected_count = int(state["expected_artifact_count"])
            rows = connection.execute(
                """
                SELECT *
                FROM runtime_artifact_verification_v1
                WHERE run_id = ? AND recorder_generation = ?
                ORDER BY artifact_name
                """,
                (run_id, generation),
            ).fetchall()
        items = [self._decoded(row) for row in rows]
        verified = (
            expected_count > 0
            and len(items) == expected_count
            and all(
                item["verification_result"] == "verified"
                and bool(item["found"])
                and bool(item["loaded"])
                and bool(item["schema_validated"])
                and bool(item["hash_verified"])
                and bool(item["contract_compatible"])
                and bool(item["used_by_active_generation"])
                for item in items
            )
        )
        blockers = sorted(
            {
                str(item["blocker"] or "RUNTIME_ARTIFACT_NOT_VERIFIED")
                for item in items
                if item["verification_result"] != "verified"
            }
        )
        if not items:
            blockers.append("RUNTIME_ARTIFACT_EVIDENCE_ABSENT")
        elif len(items) != expected_count:
            blockers.append("RUNTIME_ARTIFACT_EVIDENCE_INCOMPLETE")
        return {
            "verified": verified,
            "generation": generation,
            "expected_artifact_count": expected_count,
            "items": items,
            "blockers": blockers,
        }

    def recorder_status_v0(
        self,
        *,
        now: datetime | None = None,
        prospective_start_utc: datetime | None = None,
        thresholds: OperationalThresholds | None = None,
        include_gap_details: bool = True,
    ) -> dict[str, Any]:
        observed = datetime.now(UTC) if now is None else now.astimezone(UTC)
        operational = self.recorder_operational_state(
            now=observed,
            prospective_start_utc=prospective_start_utc,
            thresholds=thresholds,
        )
        run_id = self.run_id or self._selected_run_id()
        if run_id is None:
            return {
                "run_id": None,
                "state": operational.state.value,
                "operational": operational.model_dump(mode="json"),
                "latest_checkpoint": None,
                "latest_completed_bar": None,
                "latest_episode": None,
                "last_event_timestamp": None,
                "data_gaps": 0,
                "raw_partition_gap_count": 0,
                "gaps": {
                    "active_gaps": 0,
                    "resolved_recoverable_gaps": 0,
                    "unresolved_scientific_gaps": 0,
                    "connection_interruptions": 0,
                    "optional_feed_degradations": 0,
                },
                "gap_details_included": include_gap_details,
                "subscriptions": {},
                "record_only": True,
                "execution_enabled": False,
            }
        with self._connection() as connection:
            checkpoint = connection.execute(
                """
                SELECT * FROM m1c_checkpoint_v0
                WHERE run_id = ? ORDER BY bar_end_utc DESC, id DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            completed_bar = connection.execute(
                """
                SELECT * FROM completed_bar_state_v0
                WHERE run_id = ? ORDER BY bar_end_utc DESC, symbol LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            episode = connection.execute(
                """
                SELECT * FROM m1c_episode_v0
                WHERE run_id = ? ORDER BY trigger_bar_end_utc DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            partition = connection.execute(
                """
                SELECT last_event_timestamp, data_gaps
                FROM web_run_event_summary_v0
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            gaps = (
                connection.execute(
                    """
                    SELECT
                        SUM(CASE WHEN resolution_timestamp_utc IS NULL THEN 1 ELSE 0 END)
                            AS active_gaps,
                        SUM(CASE WHEN resolution_timestamp_utc IS NOT NULL
                                      AND recoverability = 'recoverable'
                                 THEN 1 ELSE 0 END) AS resolved_recoverable_gaps,
                        SUM(CASE WHEN resolution_timestamp_utc IS NULL
                                      AND severity = 'scientific'
                                 THEN 1 ELSE 0 END) AS unresolved_scientific_gaps,
                        SUM(CASE WHEN cause_code LIKE 'CONNECTION_%'
                                      OR stream_kind = 'connection'
                                 THEN 1 ELSE 0 END) AS connection_interruptions,
                        SUM(CASE WHEN resolution_timestamp_utc IS NULL
                                      AND severity = 'optional'
                                 THEN 1 ELSE 0 END) AS optional_feed_degradations
                    FROM gap_incident_v1
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if include_gap_details
                else None
            )
            subscriptions = connection.execute(
                """
                SELECT subscription_kind, COUNT(*) AS used
                FROM web_latest_subscription_state_v0
                WHERE run_id = ?
                  AND status IN ('pending', 'active', 'cancellation_requested')
                GROUP BY subscription_kind
                """,
                (run_id,),
            ).fetchall()
            connection_event = connection.execute(
                """
                SELECT * FROM ibkr_connection_event
                WHERE run_id = ? ORDER BY id DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return {
            "run_id": run_id,
            "state": operational.state.value,
            "operational": operational.model_dump(mode="json"),
            "latest_checkpoint": (None if checkpoint is None else self._decoded(checkpoint)),
            "latest_completed_bar": (
                None if completed_bar is None else self._decoded(completed_bar)
            ),
            "latest_episode": None if episode is None else self._decoded(episode),
            "last_event_timestamp": (
                None if partition is None else partition["last_event_timestamp"]
            ),
            "data_gaps": (
                int(operational.conditions["unresolved_required_gap_count"] or 0)
                if gaps is None
                else 0
                if gaps["unresolved_scientific_gaps"] is None
                else int(gaps["unresolved_scientific_gaps"])
            ),
            "raw_partition_gap_count": (0 if partition is None else int(partition["data_gaps"])),
            "gaps": {
                name: (None if gaps is None else 0 if gaps[name] is None else int(gaps[name]))
                for name in (
                    "active_gaps",
                    "resolved_recoverable_gaps",
                    "unresolved_scientific_gaps",
                    "connection_interruptions",
                    "optional_feed_degradations",
                )
            },
            "gap_details_included": include_gap_details,
            "subscriptions": {
                str(row["subscription_kind"]): int(row["used"]) for row in subscriptions
            },
            "ibkr_connection": self._dict(connection_event),
            "record_only": True,
            "execution_enabled": False,
            "order_routing": "disabled",
        }

    def dashboard_summary_v0(
        self,
        *,
        now: datetime,
        prospective_start_utc: datetime | None = None,
        thresholds: OperationalThresholds | None = None,
    ) -> dict[str, Any]:
        """Load the bounded state needed by the frequent dashboard poll.

        The caller may wrap this projection in :meth:`snapshot_transaction` so
        every row comes from one read-only WAL snapshot.  Deliberately keep this
        projection independent of filesystem artifacts, Parquet, reports,
        transfer history, universe history, and expanded gap details.
        """

        observed = now.astimezone(UTC)
        with self._connection() as connection:
            if self.run_id is None:
                run = connection.execute(
                    """
                    SELECT run_id, prospective_start_utc
                    FROM prospective_run
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                    """
                ).fetchone()
            else:
                run = connection.execute(
                    """
                    SELECT run_id, prospective_start_utc
                    FROM prospective_run
                    WHERE run_id = ?
                    """,
                    (self.run_id,),
                ).fetchone()
            if run is None:
                operational = inactive_operational_projection(now=observed)
                return {
                    "run_id": None,
                    "operational": operational,
                    "checkpoint": None,
                    "completed_bar": None,
                    "episode": None,
                    "connection": None,
                    "subscriptions": {},
                    "pending_inbox_count": 0,
                    "leased_inbox_count": 0,
                    "alerts": [],
                }

            run_id = str(run["run_id"])
            start = (
                prospective_start_utc.astimezone(UTC)
                if prospective_start_utc is not None
                else datetime.fromisoformat(str(run["prospective_start_utc"])).astimezone(UTC)
            )
            operational = project_operational_state_from_database(
                connection,
                run_id=run_id,
                now=observed,
                prospective_start_utc=start,
                thresholds=thresholds or OperationalThresholds(),
            )
            checkpoint = connection.execute(
                """
                SELECT model_id, symbol, checkpoint, bar_end_utc, probability,
                       threshold, threshold_passed, eligible, feature_freshness,
                       missing_feature_count
                FROM m1c_checkpoint_v0
                WHERE run_id = ?
                ORDER BY bar_end_utc DESC, id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            completed_bar = connection.execute(
                """
                SELECT symbol, session_date, bar_start_utc, bar_end_utc,
                       checkpoint, source, source_completeness,
                       received_timestamp_utc
                FROM completed_bar_state_v0
                WHERE run_id = ?
                ORDER BY bar_end_utc DESC, symbol
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            episode = connection.execute(
                """
                SELECT episode_id, symbol, session_date, trigger_checkpoint,
                       trigger_bar_end_utc, prospective_entry_timestamp_utc,
                       m1c_probability, scientific_recording_valid, phase,
                       completion_status, completed_at_utc
                FROM m1c_episode_v0
                WHERE run_id = ?
                ORDER BY trigger_bar_end_utc DESC, episode_id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            connection_event = connection.execute(
                """
                SELECT id, state, error_code, message, data_maintained,
                       reconnect_attempt
                FROM ibkr_connection_event
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            subscriptions = connection.execute(
                """
                SELECT subscription_kind, COUNT(*) AS used
                FROM web_latest_subscription_state_v0
                WHERE run_id = ?
                  AND status IN ('pending', 'active', 'cancellation_requested')
                GROUP BY subscription_kind
                """,
                (run_id,),
            ).fetchall()
            inbox_counts = connection.execute(
                """
                SELECT status, COUNT(*) AS event_count
                FROM callback_inbox_v1
                WHERE admission_run_id = ? AND status IN ('pending', 'leased')
                GROUP BY status
                """,
                (run_id,),
            ).fetchall()
            alert_rows = connection.execute(
                """
                SELECT blocker_code, message, severity
                FROM web_active_runtime_blocker_v0
                WHERE run_id = ?
                  AND component NOT IN ('parallel_feature_validation', 'feature_parity')
                ORDER BY event_id
                LIMIT 25
                """,
                (run_id,),
            ).fetchall()

        counts = {str(row["status"]): int(row["event_count"]) for row in inbox_counts}
        return {
            "run_id": run_id,
            "operational": operational,
            "checkpoint": self._dict(checkpoint),
            "completed_bar": self._dict(completed_bar),
            "episode": self._dict(episode),
            "connection": self._dict(connection_event),
            "subscriptions": {
                str(row["subscription_kind"]): int(row["used"]) for row in subscriptions
            },
            "pending_inbox_count": counts.get("pending", 0),
            "leased_inbox_count": counts.get("leased", 0),
            "alerts": [dict(row) for row in alert_rows],
        }

    def universe_live_v0(self) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                WITH current_session AS (
                    SELECT COALESCE(
                        (
                            SELECT session_date
                            FROM runtime_session
                            WHERE run_id = ?
                            ORDER BY opened_at_utc DESC, id DESC
                            LIMIT 1
                        ),
                        (
                            SELECT MAX(session_date)
                            FROM m1c_checkpoint_completion_v0
                            WHERE run_id = ?
                        )
                    ) AS session_date
                )
                SELECT u.symbol, u.operational_status,
                       current.session_date AS current_session_date,
                       b.bar_end_utc AS last_completed_bar,
                       c.probability AS m1c_probability,
                       c.threshold AS m1c_threshold,
                       c.threshold_passed,
                       c.eligible AS m1c_scientific_eligible,
                       c.rejection_reasons_json AS m1c_rejection_reasons_json,
                       c.diagnostic_quality_flags_json
                         AS m1c_diagnostic_quality_flags_json,
                       c.checkpoint AS latest_completed_checkpoint,
                       c.bar_end_utc AS latest_completed_checkpoint_utc,
                       e.episode_id AS latest_episode_id,
                       e.session_date AS latest_episode_session_date,
                       COALESCE(schedule.status, e.completion_status)
                         AS latest_episode_status,
                       e.episode_id,
                       COALESCE(schedule.status, e.completion_status)
                         AS episode_status,
                       CASE WHEN e.episode_id IS NULL THEN 0 ELSE 1 END
                         AS has_historical_episode,
                       CASE
                         WHEN e.episode_id IS NOT NULL
                          AND e.run_id = u.run_id
                          AND e.symbol = u.symbol
                          AND e.session_date = current.session_date
                          AND e.checkpoint_id = c.id
                          AND e.trigger_checkpoint = c.checkpoint
                          AND e.trigger_bar_end_utc = c.bar_end_utc
                          AND e.scientific_recording_valid = 1
                          AND COALESCE(
                              schedule.status,
                              e.completion_status
                          ) IN (
                              'active', 'scheduled', 'streaming', 'complete'
                          )
                         THEN 1 ELSE 0
                       END AS fresh_episode,
                       COALESCE(s.bid, q.bid) AS bid,
                       COALESCE(s.ask, q.ask) AS ask,
                       COALESCE(s.bid_size, q.bid_size) AS bid_size,
                       COALESCE(s.ask_size, q.ask_size) AS ask_size,
                       COALESCE(
                         s.received_timestamp_utc,
                         q.receive_timestamp_utc
                       ) AS quote_timestamp_utc,
                       s.microprice_edge_bps,
                       s.tick_by_tick_status,
                       s.depth_status,
                       s.market_data_type,
                       (
                         SELECT action FROM direction_classification_v0 d
                         WHERE d.episode_id = e.episode_id AND d.archetype = 'A1'
                       ) AS a1_classification,
                       (
                         SELECT action FROM direction_classification_v0 d
                         WHERE d.episode_id = e.episode_id AND d.archetype = 'C1'
                       ) AS c1_classification,
                       (
                         SELECT action FROM direction_classification_v0 d
                         WHERE d.episode_id = e.episode_id AND d.archetype = 'R1'
                       ) AS r1_classification
                FROM universe_membership u
                LEFT JOIN current_session current ON 1 = 1
                LEFT JOIN m1c_checkpoint_v0 c ON c.id = (
                    SELECT candidate.id
                    FROM m1c_checkpoint_v0 candidate
                    JOIN m1c_checkpoint_completion_v0 completed
                      ON completed.checkpoint_id = candidate.id
                    WHERE candidate.run_id = u.run_id
                      AND candidate.symbol = u.symbol
                      AND candidate.session_date = current.session_date
                    ORDER BY candidate.checkpoint DESC,
                             candidate.bar_end_utc DESC,
                             candidate.id DESC
                    LIMIT 1
                )
                LEFT JOIN m1c_episode_v0 e ON e.episode_id = (
                    SELECT candidate.episode_id
                    FROM m1c_episode_v0 candidate
                    WHERE candidate.run_id = u.run_id
                      AND candidate.symbol = u.symbol
                    ORDER BY candidate.trigger_bar_end_utc DESC,
                             candidate.episode_id DESC
                    LIMIT 1
                )
                LEFT JOIN option_episode_schedule_v0 schedule
                  ON schedule.episode_id = e.episode_id
                LEFT JOIN completed_bar_state_v0 b
                  ON b.symbol = u.symbol AND b.run_id = u.run_id
                LEFT JOIN underlying_quote q ON q.id = (
                    SELECT candidate.id
                    FROM underlying_quote candidate
                    WHERE candidate.run_id = u.run_id
                      AND candidate.symbol = u.symbol
                    ORDER BY candidate.target_timestamp_utc DESC,
                             candidate.id DESC
                    LIMIT 1
                )
                LEFT JOIN underlying_live_state_v0 s
                  ON s.symbol = u.symbol AND s.run_id = u.run_id
                WHERE u.run_id = ?
                ORDER BY u.symbol
                """,
                (run_id, run_id, run_id),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            probability = item.get("m1c_probability")
            threshold = item.get("m1c_threshold")
            rejection_reasons_json = item.pop(
                "m1c_rejection_reasons_json",
                None,
            )
            diagnostic_quality_flags_json = item.pop(
                "m1c_diagnostic_quality_flags_json",
                None,
            )
            item["m1c_scientific_eligible"] = (
                None if probability is None else bool(item.get("m1c_scientific_eligible"))
            )
            rejection_reasons = (
                []
                if rejection_reasons_json is None
                else list(json.loads(str(rejection_reasons_json)))
            )
            diagnostic_quality_flags = (
                []
                if diagnostic_quality_flags_json is None
                else list(json.loads(str(diagnostic_quality_flags_json)))
            )
            if not diagnostic_quality_flags and "underlying_quote_stale" in rejection_reasons:
                diagnostic_quality_flags.append("underlying_quote_stale")
            item["m1c_rejection_reasons"] = rejection_reasons
            item["m1c_diagnostic_quality_flags"] = diagnostic_quality_flags
            item["distance_from_threshold"] = (
                None
                if probability is None or threshold is None
                else float(probability) - float(threshold)
            )
            bid = item.get("bid")
            ask = item.get("ask")
            bid_size = item.get("bid_size")
            ask_size = item.get("ask_size")
            item["spread"] = None if bid is None or ask is None else float(ask) - float(bid)
            item["quote_imbalance"] = (
                None
                if bid_size is None or ask_size is None
                else (float(bid_size) - float(ask_size))
                / (float(bid_size) + float(ask_size) + 1e-12)
            )
            item["has_historical_episode"] = bool(item["has_historical_episode"])
            item["fresh_episode"] = bool(item["fresh_episode"])
            items.append(item)
        return items

    def episodes_v0(self, *, limit: int = 500) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT e.*,
                       MAX(CASE WHEN d.archetype = 'A1' THEN d.action END) AS a1_action,
                       MAX(CASE WHEN d.archetype = 'C1' THEN d.action END) AS c1_action,
                       MAX(CASE WHEN d.archetype = 'R1' THEN d.action END) AS r1_action
                FROM m1c_episode_v0 e
                LEFT JOIN direction_classification_v0 d ON d.episode_id = e.episode_id
                WHERE e.run_id = ?
                GROUP BY e.episode_id
                ORDER BY e.trigger_bar_end_utc DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [self._decoded(row) for row in rows]

    def episode_v0(self, episode_id: str) -> dict[str, Any] | None:
        run_id = self._selected_run_id()
        if run_id is None:
            return None
        with self._connection() as connection:
            episode = connection.execute(
                """
                SELECT e.*, c.feature_values_json, c.model_hash,
                       c.feature_hash, c.session_context_hash
                FROM m1c_episode_v0 e
                JOIN m1c_checkpoint_v0 c ON c.id = e.checkpoint_id
                WHERE e.episode_id = ? AND e.run_id = ?
                """,
                (episode_id, run_id),
            ).fetchone()
            if episode is None:
                return None
            directions = connection.execute(
                """
                SELECT * FROM direction_classification_v0
                WHERE episode_id = ? ORDER BY archetype
                """,
                (episode_id,),
            ).fetchall()
        return {
            "episode": self._decoded(episode),
            "directional_research_classifications": [self._decoded(row) for row in directions],
        }

    def episode_microstructure_v0(
        self,
        episode_id: str,
    ) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM microstructure_summary_v0
                WHERE run_id = ? AND episode_id = ?
                ORDER BY window_end_utc, window_name
                """,
                (run_id, episode_id),
            ).fetchall()
        return [self._decoded(row) for row in rows]

    def episode_quote_series_v0(
        self,
        episode_id: str,
        *,
        maximum_points: int = 600,
        maximum_input_rows: int = 50_000,
    ) -> list[dict[str, Any]]:
        """Read a bounded chart projection from immutable quote partitions."""

        if maximum_points < 2:
            raise ValueError("maximum quote-series points must be at least two")
        if maximum_input_rows <= 0:
            raise ValueError("maximum quote-series input rows must be positive")
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            episode = connection.execute(
                """
                SELECT symbol, trigger_bar_end_utc, prospective_entry_timestamp_utc
                FROM m1c_episode_v0
                WHERE run_id = ? AND episode_id = ?
                """,
                (run_id, episode_id),
            ).fetchone()
            if episode is None:
                return []
            start = datetime.fromisoformat(str(episode["trigger_bar_end_utc"])) - timedelta(
                minutes=15
            )
            end = datetime.fromisoformat(
                str(episode["prospective_entry_timestamp_utc"])
            ) + timedelta(minutes=30)
            partitions = connection.execute(
                """
                SELECT file_path, event_type
                FROM raw_partition_manifest_v0
                WHERE run_id = ? AND symbol = ?
                  AND event_type IN (
                    'underlying_level1_quote_event',
                    'underlying_tick_bidask_event'
                  )
                  AND maximum_timestamp_utc >= ?
                  AND minimum_timestamp_utc <= ?
                ORDER BY minimum_timestamp_utc, content_hash
                """,
                (run_id, str(episode["symbol"]), start.isoformat(), end.isoformat()),
            ).fetchall()
        points: dict[str, dict[str, Any]] = {}
        input_rows = 0
        for partition in partitions:
            path = Path(str(partition["file_path"]))
            if not path.is_file():
                continue
            remaining_rows = maximum_input_rows - input_rows
            if remaining_rows <= 0:
                return []
            try:
                projection = read_parquet_window(
                    path,
                    columns=(
                        "event_id",
                        "provider_timestamp_utc",
                        "received_timestamp_utc",
                        "bid",
                        "ask",
                        "bid_size",
                        "ask_size",
                    ),
                    timestamp_columns=(
                        "provider_timestamp_utc",
                        "received_timestamp_utc",
                    ),
                    start=start,
                    end=end,
                    maximum_input_rows=remaining_rows,
                )
            except ParquetProjectionLimitExceeded as exc:
                _record_parquet_metrics(exc.metrics)
                return []
            except OSError:
                return []
            _record_parquet_metrics(projection.metrics)
            input_rows += projection.metrics.input_rows
            for row in projection.rows:
                raw_timestamp = row.get("provider_timestamp_utc") or row.get(
                    "received_timestamp_utc"
                )
                if raw_timestamp is None:
                    continue
                observed = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
                if observed.tzinfo is None or observed.utcoffset() is None:
                    continue
                observed = observed.astimezone(UTC)
                bid = row.get("bid")
                ask = row.get("ask")
                if not start <= observed <= end or bid is None or ask is None:
                    continue
                bid_value = float(bid)
                ask_value = float(ask)
                if bid_value <= 0.0 or ask_value < bid_value:
                    continue
                bid_size = row.get("bid_size")
                ask_size = row.get("ask_size")
                bid_size_value = None if bid_size is None else float(bid_size)
                ask_size_value = None if ask_size is None else float(ask_size)
                total_size = (
                    None
                    if bid_size_value is None or ask_size_value is None
                    else bid_size_value + ask_size_value
                )
                microprice = (
                    None
                    if total_size is None
                    or total_size <= 0.0
                    or bid_size_value is None
                    or ask_size_value is None
                    else (ask_value * bid_size_value + bid_value * ask_size_value) / total_size
                )
                event_id = row.get("event_id")
                identity = (
                    str(event_id)
                    if event_id is not None
                    else f"{observed.isoformat()}:{bid_value}:{ask_value}"
                )
                points[identity] = {
                    "event_id": row.get("event_id"),
                    "timestamp_utc": observed.isoformat(),
                    "bid": bid_value,
                    "ask": ask_value,
                    "midpoint": (bid_value + ask_value) / 2.0,
                    "microprice": microprice,
                    "event_type": str(partition["event_type"]),
                }
        ordered = sorted(
            points.values(),
            key=lambda item: (str(item["timestamp_utc"]), str(item["event_id"])),
        )
        if len(ordered) <= maximum_points:
            return ordered
        sample_indexes = tuple(
            round(index * (len(ordered) - 1) / (maximum_points - 1))
            for index in range(maximum_points)
        )
        return [ordered[index] for index in sample_indexes]

    def episode_depth_snapshot_v0(
        self,
        episode_id: str,
        *,
        maximum_input_rows: int = 20_000,
    ) -> dict[str, Any] | None:
        if maximum_input_rows <= 0:
            raise ValueError("maximum depth-snapshot input rows must be positive")
        run_id = self._selected_run_id()
        if run_id is None:
            return None
        with self._connection() as connection:
            episode = connection.execute(
                """
                SELECT symbol, trigger_bar_end_utc, prospective_entry_timestamp_utc
                FROM m1c_episode_v0
                WHERE run_id = ? AND episode_id = ?
                """,
                (run_id, episode_id),
            ).fetchone()
            if episode is None:
                return None
            start = datetime.fromisoformat(str(episode["trigger_bar_end_utc"])) - timedelta(
                minutes=15
            )
            end = datetime.fromisoformat(
                str(episode["prospective_entry_timestamp_utc"])
            ) + timedelta(minutes=30)
            partitions = connection.execute(
                """
                SELECT file_path FROM raw_partition_manifest_v0
                WHERE run_id = ? AND symbol = ?
                  AND event_type = 'underlying_depth_snapshot'
                  AND maximum_timestamp_utc >= ?
                  AND minimum_timestamp_utc <= ?
                ORDER BY minimum_timestamp_utc, content_hash
                """,
                (run_id, str(episode["symbol"]), start.isoformat(), end.isoformat()),
            ).fetchall()
        candidates: list[dict[str, Any]] = []
        input_rows = 0
        for partition in partitions:
            path = Path(str(partition["file_path"]))
            if not path.is_file():
                continue
            remaining_rows = maximum_input_rows - input_rows
            if remaining_rows <= 0:
                return None
            try:
                projection = read_parquet_window(
                    path,
                    columns=(
                        "event_id",
                        "received_timestamp_utc",
                        "snapshot",
                    ),
                    timestamp_columns=("received_timestamp_utc",),
                    start=start,
                    end=end,
                    maximum_input_rows=remaining_rows,
                )
            except ParquetProjectionLimitExceeded as exc:
                _record_parquet_metrics(exc.metrics)
                return None
            except OSError:
                return None
            _record_parquet_metrics(projection.metrics)
            input_rows += projection.metrics.input_rows
            for row in projection.rows:
                raw_timestamp = row.get("received_timestamp_utc")
                snapshot = row.get("snapshot")
                if raw_timestamp is None or snapshot is None:
                    continue
                observed = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
                if observed.tzinfo is None or observed.utcoffset() is None:
                    continue
                if not start <= observed <= end:
                    continue
                decoded = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
                if isinstance(decoded, dict):
                    candidates.append(
                        {
                            **decoded,
                            "received_timestamp_utc": observed.astimezone(UTC).isoformat(),
                        }
                    )
        return (
            None
            if not candidates
            else max(candidates, key=lambda item: str(item["received_timestamp_utc"]))
        )

    def episode_options_v0(self, episode_id: str) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.*, q.provider_timestamp_utc, q.received_timestamp_utc,
                       q.bid, q.bid_size, q.ask, q.ask_size, q.last, q.last_size,
                       q.market_data_type, q.option_model_price,
                       q.implied_volatility, q.delta, q.gamma, q.theta, q.vega,
                       q.underlying_reference_price, q.volume, q.open_interest,
                       q.recording_status, q.quote_quality_flags_json
                FROM episode_option_contract_v0 c
                LEFT JOIN option_quote_state_v0 q ON q.option_contract_id = c.id
                WHERE c.run_id = ? AND c.episode_id = ?
                ORDER BY c.dte, c.selection_rank, c.right
                """,
                (run_id, episode_id),
            ).fetchall()
        return [self._decoded(row) for row in rows]

    def shadow_outcomes_v0(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT s.*, e.symbol, e.trigger_bar_end_utc
                FROM shadow_quote_outcome_v0 s
                JOIN m1c_episode_v0 e ON e.episode_id = s.episode_id
                WHERE s.run_id = ?
                ORDER BY s.target_timestamp_utc DESC, s.id DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
            structures = connection.execute(
                """
                SELECT s.*, e.symbol, e.trigger_bar_end_utc
                FROM shadow_structure_outcome_v0 s
                JOIN m1c_episode_v0 e ON e.episode_id = s.episode_id
                WHERE s.run_id = ?
                ORDER BY e.trigger_bar_end_utc DESC, s.id DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        output = [self._decoded(row) for row in rows]
        for raw in structures:
            item = self._decoded(raw)
            payload = item.get("payload")
            details = payload if isinstance(payload, dict) else {}
            structure_type = str(item["structure_type"])
            output.append(
                {
                    **item,
                    "archetype": (
                        "ATM straddle"
                        if structure_type == "ATM_STRADDLE"
                        else "retrospective oracle — not tradeable"
                    ),
                    "direction": "NEUTRAL" if structure_type == "ATM_STRADDLE" else "ORACLE",
                    "contract_identity": "ATM call + ATM put",
                    "entry_ask": details.get("entry_call_ask_plus_put_ask"),
                    "exit_bid": details.get("exit_call_bid_plus_put_bid"),
                    "ask_to_bid_return": details.get("ask_to_bid_return"),
                    "dollar_pnl_per_contract": None,
                    "quality_flags": [],
                }
            )
        return sorted(
            output,
            key=lambda item: (
                str(item.get("trigger_bar_end_utc", "")),
                int(item.get("horizon_minutes", 0)),
            ),
            reverse=True,
        )[:limit]

    def opening_reversal_virtual_positions_v1(
        self,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return strict V1.1 predicted-leg shadow and scientific evidence."""

        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM opening_reversal_virtual_position_v1
                WHERE run_id = ?
                ORDER BY session_date DESC, entry_timestamp_utc DESC,
                         virtual_position_id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            raw = dict(row)
            encoded_flags = raw.pop("latest_quote_quality_flags_json", None)
            raw["latest_quote_quality_flags"] = (
                () if encoded_flags is None else tuple(json.loads(str(encoded_flags)))
            )
            item = OpeningReversalVirtualPositionV1.model_validate(raw)
            items.append(item.model_dump(mode="json"))
        return items

    def opening_leader_option_accounting_v0(
        self,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return bounded executable option marks from the segregated leader ledger."""

        if limit <= 0:
            raise ValueError("opening-leader option accounting limit must be positive")
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT stable_id, session_date, checkpoint, selected_symbol,
                       observation_name, observed_at_utc,
                       data_quality_flags_json, payload_json
                FROM opening_leader_evidence_v0
                WHERE run_id = ?
                  AND record_type = 'option_strategy_accounting'
                  AND original_stable_id IS NULL
                ORDER BY session_date DESC, checkpoint,
                         observed_at_utc DESC, id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(json.loads(str(row["payload_json"])))
            items.append(
                {
                    "evidence_id": str(row["stable_id"]),
                    "session_date": str(row["session_date"]),
                    "checkpoint": f"C{int(row['checkpoint'])}",
                    "selected_symbol": row["selected_symbol"],
                    "recorded_observation_name": str(row["observation_name"]),
                    "observed_at_utc": str(row["observed_at_utc"]),
                    "evidence_quality_flags": tuple(
                        json.loads(str(row["data_quality_flags_json"]))
                    ),
                    **payload,
                }
            )
        return items

    def quiet_state_virtual_positions_v1(
        self,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return quiet-bottom-10 short-premium evidence in its own projection."""

        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM quiet_state_virtual_position_v1
                WHERE run_id = ?
                ORDER BY session_date DESC, trigger_timestamp_utc DESC,
                         horizon_minutes, virtual_position_id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            raw = dict(row)
            raw["quality_flags"] = tuple(json.loads(str(raw.pop("quality_flags_json"))))
            raw["legs"] = tuple(json.loads(str(raw.pop("legs_json"))))
            item = QuietStateVirtualPositionV1.model_validate(raw)
            items.append(item.model_dump(mode="json"))
        return items

    def quiet_state_virtual_captures_v1(
        self,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return scheduled/capturing quiet episodes with latest quote diagnostics."""

        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            observations = connection.execute(
                """
                SELECT observation.*,
                       (
                           SELECT COUNT(*)
                           FROM quiet_state_shadow_outcome_v0 AS outcome
                           WHERE outcome.run_id = observation.run_id
                             AND outcome.observation_id =
                                 observation.observation_id
                             AND outcome.structure_type IN (
                                 'ATM_IRON_BUTTERFLY',
                                 'DELTA_IRON_CONDOR',
                                 'CALL_CREDIT_SPREAD',
                                 'PUT_CREDIT_SPREAD'
                             )
                       ) AS frozen_short_premium_outcome_count
                FROM quiet_state_observation_v0 AS observation
                WHERE observation.run_id = ?
                  AND observation.observation_kind = 'quiet_bottom_10'
                ORDER BY observation.trigger_timestamp_utc DESC,
                         observation.observation_id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
            observation_ids = tuple(str(row["observation_id"]) for row in observations)
            contract_rows: list[sqlite3.Row] = []
            if observation_ids:
                placeholders = ",".join("?" for _ in observation_ids)
                contract_rows = connection.execute(
                    f"""
                    SELECT contract.id AS option_contract_id,
                           contract.observation_id,
                           contract.con_id, contract.expiry, contract.dte,
                           contract.dte_bucket, contract.strike, contract.right,
                           contract.multiplier, contract.selection_roles_json,
                           contract.resolution_status, contract.rejection_reason,
                           contract.recording_started_at_utc,
                           contract.recording_ends_at_utc,
                           quote.received_timestamp_utc
                               AS latest_quote_received_at_utc,
                           quote.bid AS latest_bid,
                           quote.ask AS latest_ask,
                           quote.market_data_type AS latest_market_data_type,
                           quote.recording_status AS latest_recording_status,
                           quote.quote_quality_flags_json
                               AS latest_quote_quality_flags_json
                    FROM quiet_state_option_contract_v0 AS contract
                    LEFT JOIN quiet_state_option_quote_state_v0 AS quote
                      ON quote.option_contract_id = contract.id
                    WHERE contract.run_id = ?
                      AND contract.observation_id IN ({placeholders})
                    ORDER BY contract.observation_id, contract.dte,
                             contract.selection_rank, contract.strike,
                             contract.right
                    """,
                    (run_id, *observation_ids),
                ).fetchall()
        contracts_by_observation: dict[str, list[dict[str, Any]]] = {
            observation_id: [] for observation_id in observation_ids
        }
        for row in contract_rows:
            raw_contract = dict(row)
            observation_id = str(raw_contract.pop("observation_id"))
            raw_contract["selection_roles"] = tuple(
                json.loads(str(raw_contract.pop("selection_roles_json")))
            )
            encoded_flags = raw_contract.pop("latest_quote_quality_flags_json")
            raw_contract["latest_quote_quality_flags"] = (
                () if encoded_flags is None else tuple(json.loads(str(encoded_flags)))
            )
            contracts_by_observation[observation_id].append(raw_contract)

        items: list[dict[str, Any]] = []
        for row in observations:
            observation = dict(row)
            observation_id = str(observation["observation_id"])
            contracts = tuple(contracts_by_observation[observation_id])
            completion_status = str(observation["completion_status"])
            outcome_count = int(observation["frozen_short_premium_outcome_count"])
            plan_recorded = bool(observation["option_plan_recorded"])
            lifecycle: str
            reason: str | None
            if not bool(observation["scientific_recording_valid"]):
                lifecycle = "INVALID"
                reason = "quiet_scientific_recording_invalid"
            elif completion_status == "incomplete":
                lifecycle = "INVALID"
                reason = "quiet_recording_completed_incomplete"
            elif completion_status == "complete":
                lifecycle = "CLOSED" if outcome_count > 0 else "INVALID"
                reason = None if outcome_count > 0 else "quiet_structure_outcomes_missing"
            elif not plan_recorded:
                lifecycle = "SCHEDULED"
                reason = "awaiting_bounded_quiet_option_plan"
            elif not contracts:
                lifecycle = "INVALID"
                reason = "bounded_quiet_option_contracts_unavailable"
            else:
                lifecycle = "CAPTURING"
                reason = "awaiting_frozen_quiet_structure_outcomes"
            item = QuietStateVirtualCaptureV1.model_validate(
                {
                    "virtual_capture_id": f"quiet-capture:{observation_id}",
                    "ledger_scope": "quiet_state_short_premium_capture",
                    "run_id": observation["run_id"],
                    "observation_id": observation_id,
                    "observation_kind": observation["observation_kind"],
                    "session_date": observation["session_date"],
                    "symbol": observation["symbol"],
                    "trigger_timestamp_utc": observation["trigger_timestamp_utc"],
                    "entry_timestamp_utc": observation["prospective_entry_timestamp_utc"],
                    "lifecycle_state": lifecycle,
                    "status_reason": reason,
                    "option_plan_recorded": plan_recorded,
                    "requested_contract_count": observation["option_plan_requested_contract_count"],
                    "selected_contract_count": observation["option_plan_selected_contract_count"],
                    "option_plan_capacity_reduced": observation["option_plan_capacity_reduced"],
                    "option_plan_missing_buckets": tuple(
                        json.loads(str(observation["option_plan_missing_buckets_json"]))
                    ),
                    "completion_status": completion_status,
                    "completed_at_utc": observation["completed_at_utc"],
                    "frozen_short_premium_outcome_count": outcome_count,
                    "contracts": contracts,
                    "scientific_recording_valid": observation["scientific_recording_valid"],
                    "latest_quotes_are_diagnostic_only": True,
                    "execution_claimed": False,
                    "paper_fill_claimed": False,
                }
            )
            items.append(item.model_dump(mode="json"))
        return items

    @staticmethod
    def _audit_cursor(*, recorded_at_utc: str, audit_id: int) -> str:
        payload = json.dumps(
            {"recorded_at_utc": recorded_at_utc, "audit_id": audit_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(payload).decode().rstrip("=")

    @staticmethod
    def _decode_audit_cursor(cursor: str) -> tuple[str, int]:
        try:
            padded = cursor + ("=" * (-len(cursor) % 4))
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
            recorded_at_utc = str(payload["recorded_at_utc"])
            audit_id = int(payload["audit_id"])
        except (
            binascii.Error,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            raise ValueError("invalid audit cursor") from exc
        if audit_id <= 0:
            raise ValueError("invalid audit cursor")
        return recorded_at_utc, audit_id

    def audit_event_page_v0(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        if limit <= 0:
            raise ValueError("audit page limit must be positive")
        run_id = self._selected_run_id()
        if run_id is None:
            return {
                "items": [],
                "next_cursor": None,
                "has_more": False,
                "limit": limit,
            }
        cursor_timestamp: str | None = None
        cursor_id: int | None = None
        if cursor is not None:
            cursor_timestamp, cursor_id = self._decode_audit_cursor(cursor)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id AS audit_id, audit_type, identity,
                       recorded_at_utc, details
                FROM web_audit_projection_v0
                WHERE run_id = ?
                  AND (
                    ? IS NULL
                    OR recorded_at_utc < ?
                    OR (recorded_at_utc = ? AND id < ?)
                  )
                ORDER BY recorded_at_utc DESC, id DESC
                LIMIT ?
                """,
                (
                    run_id,
                    cursor_timestamp,
                    cursor_timestamp,
                    cursor_timestamp,
                    cursor_id,
                    limit + 1,
                ),
            ).fetchall()
        has_more = len(rows) > limit
        visible = rows[:limit]
        items = [dict(row) for row in visible]
        next_cursor = None
        if has_more and items:
            next_cursor = self._audit_cursor(
                recorded_at_utc=str(items[-1]["recorded_at_utc"]),
                audit_id=int(items[-1]["audit_id"]),
            )
        return {
            "items": items,
            "next_cursor": next_cursor,
            "has_more": has_more,
            "limit": limit,
        }

    def audit_events_v0(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Compatibility wrapper over the indexed first audit page."""

        return list(self.audit_event_page_v0(limit=limit)["items"])

    def raw_event_sample_v0(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent raw-partition identities without opening Parquet files."""

        return [
            item
            for item in self.audit_events_v0(limit=limit)
            if item["audit_type"] == "raw_partition"
        ]

    def raw_event_detail_v0(
        self,
        content_hash: str,
        *,
        limit: int = 100,
        maximum_input_rows: int = 50_000,
    ) -> dict[str, Any] | None:
        """Explicitly inspect one immutable partition with bounded columns and rows."""

        if limit <= 0 or maximum_input_rows <= 0:
            raise ValueError("raw-event detail limits must be positive")
        run_id = self._selected_run_id()
        if run_id is None:
            return None
        with self._connection() as connection:
            partition = connection.execute(
                """
                SELECT content_hash, event_type, symbol, session_date, file_path,
                       row_count, minimum_timestamp_utc, maximum_timestamp_utc,
                       complete, gap_count
                FROM raw_partition_manifest_v0
                WHERE run_id = ? AND content_hash = ?
                """,
                (run_id, content_hash),
            ).fetchone()
        if partition is None:
            return None
        path = Path(str(partition["file_path"]))
        public_partition = dict(partition)
        public_partition.pop("file_path")
        if not path.is_file():
            return {
                "partition": public_partition,
                "items": [],
                "blocked_reason": "partition_unavailable",
                "read_metrics": None,
            }
        try:
            projection = read_parquet_tail(
                path,
                columns=(
                    "event_id",
                    "provider_timestamp_utc",
                    "received_timestamp_utc",
                    "source_sequence",
                    "symbol",
                    "bid",
                    "ask",
                    "last",
                    "price",
                    "size",
                    "operation",
                    "side",
                    "position",
                    "checkpoint",
                    "market_data_type",
                ),
                maximum_rows=limit,
                maximum_input_rows=maximum_input_rows,
            )
        except (OSError, ParquetProjectionLimitExceeded) as exc:
            if isinstance(exc, ParquetProjectionLimitExceeded):
                _record_parquet_metrics(exc.metrics)
            metrics = (
                exc.metrics.model_dump()
                if isinstance(exc, ParquetProjectionLimitExceeded)
                else None
            )
            return {
                "partition": public_partition,
                "items": [],
                "blocked_reason": "projection_limit_exceeded",
                "read_metrics": metrics,
            }
        _record_parquet_metrics(projection.metrics)
        return {
            "partition": public_partition,
            "items": list(projection.rows),
            "blocked_reason": None,
            "read_metrics": projection.metrics.model_dump(),
        }

    def quiet_state_status_v0(self) -> dict[str, Any]:
        """Project frozen quiet-state collection status without broker access."""

        run_id = self._selected_run_id()
        empty: dict[str, Any] = {
            "latest_checkpoint": None,
            "checkpoint_count": 0,
            "observation_counts": {},
            "phase_counts": {},
            "complete_quiet_episodes": 0,
        }
        projection: dict[str, Any]
        if run_id is None:
            projection = empty
        else:
            with self._connection() as connection:
                latest = connection.execute(
                    """
                    SELECT q.*, m.bar_end_utc
                    FROM quiet_state_checkpoint_v0 q
                    JOIN m1c_checkpoint_v0 m ON m.id = q.checkpoint_id
                    WHERE q.run_id = ?
                    ORDER BY m.bar_end_utc DESC, q.id DESC LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                checkpoint_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM quiet_state_checkpoint_v0 WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()[0]
                )
                observation_counts = connection.execute(
                    """
                    SELECT observation_kind, COUNT(*) AS count
                    FROM quiet_state_observation_v0
                    WHERE run_id = ? GROUP BY observation_kind
                    """,
                    (run_id,),
                ).fetchall()
                phase_counts = connection.execute(
                    """
                    SELECT phase, observation_kind, COUNT(*) AS count
                    FROM quiet_state_observation_v0
                    WHERE run_id = ? GROUP BY phase, observation_kind
                    ORDER BY phase, observation_kind
                    """,
                    (run_id,),
                ).fetchall()
                complete_quiet = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM quiet_state_observation_v0
                        WHERE run_id = ? AND observation_kind = 'quiet_bottom_10'
                          AND completion_status = 'complete'
                        """,
                        (run_id,),
                    ).fetchone()[0]
                )
            projection = {
                "latest_checkpoint": (None if latest is None else self._decoded(latest)),
                "checkpoint_count": checkpoint_count,
                "observation_counts": {
                    str(row["observation_kind"]): int(row["count"]) for row in observation_counts
                },
                "phase_counts": [dict(row) for row in phase_counts],
                "complete_quiet_episodes": complete_quiet,
            }
        return {
            **projection,
            "thresholds": {
                "bottom_5": BOTTOM_5_THRESHOLD,
                "bottom_10": BOTTOM_10_THRESHOLD,
                "bottom_20": BOTTOM_20_THRESHOLD,
                "high_tail": HIGH_TAIL_THRESHOLD,
            },
            "neutral_control": {
                "sampling_fraction": NEUTRAL_CONTROL_SAMPLING_FRACTION,
                "salt_sha256": hashlib.sha256(NEUTRAL_CONTROL_SALT.encode()).hexdigest(),
            },
            "phase_boundaries": {
                "prospective_ibkr_evidence_from_first_valid_session": True,
                "cross_vendor_diagnostic_target_sessions": 20,
                "option_development_complete_quiet_episodes": 150,
                "untouched_confirmation_complete_quiet_episodes": 150,
            },
            "market_data_source": "ibkr",
            "historical_research_source": "eodhd",
            "cross_vendor_validation_diagnostic_only": True,
            "record_only": True,
            "order_path": "absent",
            "original_decision": "blocked_insufficient_low_tail_support",
        }

    def quiet_state_universe_v0(self) -> list[dict[str, Any]]:
        """Return the latest quiet classification and Level I state per symbol."""

        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                WITH latest_quiet AS (
                    SELECT q.*
                    FROM quiet_state_checkpoint_v0 q
                    JOIN (
                        SELECT symbol, MAX(id) AS maximum_id
                        FROM quiet_state_checkpoint_v0
                        WHERE run_id = ? GROUP BY symbol
                    ) x ON x.maximum_id = q.id
                ),
                latest_observation AS (
                    SELECT o.*
                    FROM quiet_state_observation_v0 o
                    JOIN (
                        SELECT symbol, MAX(trigger_timestamp_utc) AS maximum_trigger
                        FROM quiet_state_observation_v0
                        WHERE run_id = ? GROUP BY symbol
                    ) x ON x.symbol = o.symbol
                       AND x.maximum_trigger = o.trigger_timestamp_utc
                    WHERE o.run_id = ?
                )
                SELECT u.symbol, u.operational_status,
                       q.session_date, q.checkpoint, q.m1c_probability,
                       q.previous_m1c_probability, q.bottom_5, q.bottom_10,
                       q.bottom_20, q.high_tail, q.distance_from_bottom_10,
                       q.data_quality_status, q.data_quality_flags_json,
                       q.selected_underlying_quote_event_id,
                       q.selected_underlying_quote_timestamp_utc,
                       q.selected_underlying_quote_age_seconds,
                       q.underlying_quote_selection_policy,
                       o.observation_id, o.observation_kind,
                       o.session_date AS observation_session_date,
                       o.trigger_checkpoint AS observation_trigger_checkpoint,
                       o.trigger_timestamp_utc, o.completion_status,
                       o.option_context_valid,
                       s.bid, s.bid_size, s.ask, s.ask_size, s.last, s.last_size,
                       s.spread, s.midpoint, s.quote_size_imbalance,
                       s.microprice_edge_bps, s.received_timestamp_utc,
                       s.market_data_type, s.quote_valid
                FROM universe_membership u
                LEFT JOIN latest_quiet q ON q.symbol = u.symbol
                LEFT JOIN latest_observation o ON o.symbol = u.symbol
                LEFT JOIN underlying_live_state_v0 s
                  ON s.run_id = u.run_id AND s.symbol = u.symbol
                WHERE u.run_id = ?
                ORDER BY u.symbol
                """,
                (run_id, run_id, run_id, run_id),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = self._decoded(row)
            item["fresh_quiet_episode"] = (
                item.get("observation_kind") == "quiet_bottom_10"
                and item.get("checkpoint") == item.get("observation_trigger_checkpoint")
                and item.get("session_date") == item.get("observation_session_date")
            )
            result.append(item)
        return result

    def quiet_state_episodes_v0(
        self,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """List quiet episodes and deterministic controls in reverse chronology."""

        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT o.*,
                       COUNT(DISTINCT c.id) AS option_contract_count,
                       COUNT(DISTINCT s.id) AS shadow_outcome_count
                FROM quiet_state_observation_v0 o
                LEFT JOIN quiet_state_option_contract_v0 c
                  ON c.observation_id = o.observation_id
                LEFT JOIN quiet_state_shadow_outcome_v0 s
                  ON s.observation_id = o.observation_id
                WHERE o.run_id = ?
                GROUP BY o.observation_id
                ORDER BY o.trigger_timestamp_utc DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [self._decoded(row) for row in rows]

    def quiet_state_episode_v0(
        self,
        observation_id: str,
    ) -> dict[str, Any] | None:
        """Return one complete quiet/control evidence projection."""

        run_id = self._selected_run_id()
        if run_id is None:
            return None
        with self._connection() as connection:
            observation = connection.execute(
                """
                SELECT o.*, q.model_hash, q.feature_hash,
                       q.data_quality_status AS checkpoint_data_quality_status,
                       q.selected_underlying_quote_event_id,
                       q.selected_underlying_quote_timestamp_utc,
                       q.selected_underlying_quote_age_seconds,
                       q.underlying_quote_selection_policy
                FROM quiet_state_observation_v0 o
                JOIN quiet_state_checkpoint_v0 q ON q.id = o.quiet_checkpoint_id
                WHERE o.run_id = ? AND o.observation_id = ?
                """,
                (run_id, observation_id),
            ).fetchone()
            if observation is None:
                return None
            microstructure = connection.execute(
                """
                SELECT * FROM quiet_state_microstructure_v0
                WHERE run_id = ? AND observation_id = ?
                ORDER BY window_end_utc, window_name
                """,
                (run_id, observation_id),
            ).fetchall()
            underlying_path = connection.execute(
                """
                SELECT * FROM quiet_state_underlying_path_v0
                WHERE run_id = ? AND observation_id = ?
                ORDER BY target_timestamp_utc
                """,
                (run_id, observation_id),
            ).fetchall()
            shadows = connection.execute(
                """
                SELECT * FROM quiet_state_shadow_outcome_v0
                WHERE run_id = ? AND observation_id = ?
                ORDER BY dte_bucket, structure_type, horizon_minutes
                """,
                (run_id, observation_id),
            ).fetchall()
            risk_observations = connection.execute(
                """
                SELECT * FROM quiet_option_risk_observation_v0
                WHERE run_id = ? AND observation_id = ?
                ORDER BY dte_bucket, horizon_label, candidate_id, observed_at_utc
                """,
                (run_id, observation_id),
            ).fetchall()
            strategy_comparisons = connection.execute(
                """
                SELECT * FROM quiet_option_strategy_comparison_v0
                WHERE run_id = ? AND observation_id = ?
                ORDER BY dte_bucket, horizon_minutes
                """,
                (run_id, observation_id),
            ).fetchall()
        return {
            "episode": self._decoded(observation),
            "frozen_thresholds": {
                "bottom_5": BOTTOM_5_THRESHOLD,
                "bottom_10": BOTTOM_10_THRESHOLD,
                "bottom_20": BOTTOM_20_THRESHOLD,
                "high_tail": HIGH_TAIL_THRESHOLD,
            },
            "underlying_path": [self._decoded(row) for row in underlying_path],
            "microstructure": [self._decoded(row) for row in microstructure],
            "shadow_structures": [self._decoded(row) for row in shadows],
            "risk_observations": [self._decoded(row) for row in risk_observations],
            "strategy_comparisons": [self._decoded(row) for row in strategy_comparisons],
        }

    def quiet_state_episode_options_v0(
        self,
        observation_id: str,
    ) -> list[dict[str, Any]]:
        """Return the bounded contract set and latest recorded option quote state."""

        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.*, q.provider_timestamp_utc, q.received_timestamp_utc,
                       q.bid, q.bid_size, q.ask, q.ask_size, q.last, q.last_size,
                       q.market_data_type, q.option_model_price,
                       q.implied_volatility, q.delta, q.gamma, q.theta, q.vega,
                       q.underlying_reference_price, q.volume, q.open_interest,
                       q.recording_status, q.quote_quality_flags_json
                FROM quiet_state_option_contract_v0 c
                LEFT JOIN quiet_state_option_quote_state_v0 q
                  ON q.option_contract_id = c.id
                WHERE c.run_id = ? AND c.observation_id = ?
                ORDER BY c.dte, c.selection_rank, c.strike, c.right
                """,
                (run_id, observation_id),
            ).fetchall()
        return [self._decoded(row) for row in rows]

    def quiet_state_shadow_structures_v0(
        self,
        *,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        """Return conservative long- and defined-risk shadow outcomes."""

        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT s.*, o.phase, o.observation_kind, o.symbol,
                       o.m1c_probability, o.trigger_timestamp_utc
                FROM quiet_state_shadow_outcome_v0 s
                JOIN quiet_state_observation_v0 o
                  ON o.observation_id = s.observation_id
                WHERE s.run_id = ?
                ORDER BY o.trigger_timestamp_utc DESC,
                         s.horizon_minutes, s.id DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [self._decoded(row) for row in rows]

    def quiet_state_session_quality_v0(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Aggregate attempted and quality-complete structures by session."""

        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                WITH observation_counts AS (
                    SELECT session_date, COUNT(*) AS observations,
                           SUM(CASE WHEN observation_kind = 'quiet_bottom_10'
                                    THEN 1 ELSE 0 END) AS quiet_episodes,
                           SUM(CASE WHEN observation_kind = 'neutral_control'
                                    THEN 1 ELSE 0 END) AS neutral_controls,
                           SUM(CASE WHEN observation_kind = 'high_tail_control'
                                    THEN 1 ELSE 0 END) AS high_tail_controls,
                           SUM(CASE WHEN completion_status = 'complete'
                                    THEN 1 ELSE 0 END) AS complete_observations
                    FROM quiet_state_observation_v0
                    WHERE run_id = ?
                    GROUP BY session_date
                ),
                shadow_counts AS (
                    SELECT o.session_date, COUNT(s.id) AS attempted_structures,
                           COALESCE(SUM(s.complete_quote_quality), 0)
                               AS complete_quote_quality_structures,
                           COALESCE(SUM(s.strict_quote_quality), 0)
                               AS strict_quote_quality_structures
                    FROM quiet_state_observation_v0 o
                    JOIN quiet_state_shadow_outcome_v0 s
                      ON s.observation_id = o.observation_id
                    WHERE o.run_id = ?
                    GROUP BY o.session_date
                )
                SELECT o.*, COALESCE(s.attempted_structures, 0)
                           AS attempted_structures,
                       COALESCE(s.complete_quote_quality_structures, 0)
                           AS complete_quote_quality_structures,
                       COALESCE(s.strict_quote_quality_structures, 0)
                           AS strict_quote_quality_structures
                FROM observation_counts o
                LEFT JOIN shadow_counts s USING (session_date)
                ORDER BY o.session_date DESC LIMIT ?
                """,
                (run_id, run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def session_reports_v0(self, *, limit: int = 100) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM recorder_session_report_v0
                WHERE run_id = ? ORDER BY session_date DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [self._decoded(row) for row in rows]

    def market_data_budget_dashboard_v0(self) -> dict[str, Any]:
        """Return current capacity, ownership, queue, and reconciliation state."""

        run_id = self._selected_run_id()
        if run_id is None:
            return {
                "runtime_capacity": None,
                "current_usage": {},
                "subscriptions_by_priority_class": {},
                "subscriptions_by_symbol": {},
                "subscriptions_by_episode": {},
                "queued_episodes": 0,
                "degraded_episodes": 0,
                "reconciliation_warnings": [],
            }
        with self._connection() as connection:
            capacity = connection.execute(
                """
                SELECT * FROM ibkr_runtime_capacity_v0
                WHERE run_id = ? ORDER BY observed_at_utc DESC, id DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            lifecycle = connection.execute(
                """
                WITH latest AS (
                    SELECT subscription_key, MAX(id) AS id
                    FROM subscription_lifecycle_event_v0
                    WHERE run_id = ? GROUP BY subscription_key
                )
                SELECT e.* FROM subscription_lifecycle_event_v0 e
                JOIN latest l ON l.id = e.id
                ORDER BY e.subscription_class, e.symbol, e.subscription_key
                """,
                (run_id,),
            ).fetchall()
            allocations = connection.execute(
                """
                WITH latest AS (
                    SELECT episode_id, MAX(id) AS id
                    FROM option_episode_allocation_v0
                    WHERE run_id = ? GROUP BY episode_id
                )
                SELECT e.* FROM option_episode_allocation_v0 e
                JOIN latest l ON l.id = e.id
                ORDER BY e.updated_at_utc, e.episode_id
                """,
                (run_id,),
            ).fetchall()
            warnings = connection.execute(
                """
                SELECT * FROM skipped_recording_v0
                WHERE run_id = ?
                  AND recording_kind = 'subscription_reconciliation_warning'
                ORDER BY occurred_at_utc DESC, id DESC LIMIT 100
                """,
                (run_id,),
            ).fetchall()
        active_statuses = {"pending", "active", "cancellation_requested"}
        active = [row for row in lifecycle if str(row["status"]) in active_statuses]
        usage: dict[str, int] = {}
        classes: dict[str, int] = {}
        symbols: dict[str, int] = {}
        episodes: dict[str, int] = {}
        for row in active:
            kind = str(row["subscription_kind"])
            class_name = f"class_{int(row['subscription_class'])}"
            symbol = str(row["symbol"])
            usage[kind] = usage.get(kind, 0) + 1
            classes[class_name] = classes.get(class_name, 0) + 1
            symbols[symbol] = symbols.get(symbol, 0) + 1
            for owner in json.loads(str(row["owner_ids_json"])):
                if str(owner).startswith("episode:"):
                    episode_id = str(owner).split(":", 1)[1]
                    episodes[episode_id] = episodes.get(episode_id, 0) + 1
        decoded_capacity = None if capacity is None else json.loads(str(capacity["manifest_json"]))
        preexisting_internal = (
            0
            if decoded_capacity is None
            else int(decoded_capacity.get("current_internal_level1_lines", 0))
        )
        allocation_rows = [self._decoded(row) for row in allocations]
        oldest_optional = min(
            (str(row["occurred_at_utc"]) for row in active if int(row["subscription_class"]) >= 3),
            default=None,
        )
        return {
            "runtime_capacity": decoded_capacity,
            "current_internal_usage": preexisting_internal + sum(usage.values()),
            "preexisting_internal_usage": preexisting_internal,
            "current_recorder_usage": sum(usage.values()),
            "current_usage": dict(sorted(usage.items())),
            "pending_requests": sum(str(row["status"]) == "pending" for row in active),
            "subscriptions_by_priority_class": dict(sorted(classes.items())),
            "subscriptions_by_symbol": dict(sorted(symbols.items())),
            "subscriptions_by_episode": dict(sorted(episodes.items())),
            "queued_episodes": sum(row.get("state") == "EPISODE_QUEUED" for row in allocation_rows),
            "degraded_episodes": sum(row.get("state") == "DEGRADED" for row in allocation_rows),
            "episode_allocations": allocation_rows,
            "oldest_active_optional_subscription": oldest_optional,
            "reconciliation_warnings": [self._decoded(row) for row in warnings],
            "optional_exhaustion_is_fatal": False,
            "fatal_budget_state": "critical_budget_unavailable",
        }

    def source_transfer_status_v0(self) -> dict[str, Any]:
        run_id = self._selected_run_id()
        if run_id is None:
            return {
                "sessions": [],
                "valid_session_count": 0,
                "decision": None,
                "latest_diagnostic_status": None,
            }
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM source_transfer_session_v0
                WHERE run_id = ? ORDER BY session_date
                """,
                (run_id,),
            ).fetchall()
            health = connection.execute(
                """
                SELECT details_json
                FROM data_health_event
                WHERE run_id = ? AND component = 'parallel_feature_validation'
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        sessions = [self._decoded(row) for row in rows]
        latest = None if not sessions else sessions[-1]
        latest_health = None if health is None else self._decoded(health)
        health_details = None if latest_health is None else latest_health.get("details")
        return {
            "sessions": sessions,
            "valid_session_count": sum(bool(row.get("valid")) for row in sessions),
            "decision": None if latest is None else latest.get("decision"),
            "latest_diagnostic_status": (
                health_details.get("cross_vendor_validation_status")
                if isinstance(health_details, dict)
                else None
            ),
        }

    def opening_leader_continuation_v0(self) -> dict[str, Any]:
        """Project immutable Opening Leader V0 evidence without evaluating it."""

        empty = {
            "title": "Opening Leader Continuation V0",
            "banner": "RECORD ONLY — ORDERS DISABLED",
            "sample_status": "PROSPECTIVE SAMPLE INCOMPLETE",
            "recorder_status": "inactive",
            "record_only": True,
            "orders_disabled": True,
            "primary_checkpoint": "C6",
            "secondary_checkpoint": "C12",
            "checkpoint_pooling_allowed": False,
            "m1c_role": "context_only",
            "option_policy_authorized": False,
            "checkpoints": {
                "C6": self._empty_opening_leader_checkpoint("primary"),
                "C12": self._empty_opening_leader_checkpoint("secondary"),
            },
            "data_quality_warnings": [],
        }
        run_id = self._selected_run_id()
        if run_id is None:
            return empty
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' "
                "AND name = 'opening_leader_evidence_v0'"
            ).fetchone()
            if exists is None:
                return empty
            latest_session_row = connection.execute(
                "SELECT MAX(session_date) AS session_date "
                "FROM opening_leader_evidence_v0 WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            latest_session = (
                None if latest_session_row is None else latest_session_row["session_date"]
            )
            rows = (
                []
                if latest_session is None
                else connection.execute(
                    "SELECT * FROM opening_leader_evidence_v0 "
                    "WHERE run_id = ? AND session_date = ? ORDER BY id",
                    (run_id, latest_session),
                ).fetchall()
            )
            support_rows = connection.execute(
                "SELECT * FROM opening_leader_evidence_v0 WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        support_evidence = [self._decoded(row) for row in support_rows]
        original_support = [
            row for row in support_evidence if row.get("original_stable_id") is None
        ]
        linked_support = {
            str(row["original_stable_id"]): row
            for row in support_evidence
            if row.get("original_stable_id") is not None
        }
        support_by_checkpoint: dict[int, dict[str, int]] = {}
        for support_checkpoint in (6, 12):
            valid_signals: list[dict[str, Any]] = []
            signals = [
                row
                for row in original_support
                if int(row["checkpoint"]) == support_checkpoint
                and row["record_type"] == "signal_receipt"
            ]
            for signal in signals:
                matching = [
                    row
                    for row in original_support
                    if int(row["checkpoint"]) == support_checkpoint
                    and row["session_date"] == signal["session_date"]
                    and row["record_type"] == "underlying_observation"
                ]
                evidence_by_name = {
                    str(row["observation_name"]): linked_support.get(
                        str(row["stable_id"]),
                        row,
                    )
                    for row in matching
                }
                e0_payload = evidence_by_name.get("E0", {}).get("payload") or {}
                final_payload = evidence_by_name.get("FINAL_CONTINUOUS", {}).get("payload") or {}
                e0_quote = e0_payload.get("quote") or {}
                final_quote = final_payload.get("quote") or {}
                identities_match = all(
                    evidence_by_name.get(name, {}).get(identity) == signal.get(identity)
                    for name in ("E0", "FINAL_CONTINUOUS")
                    for identity in ("cohort_hash", "contract_hash", "code_hash")
                )
                if (
                    identities_match
                    and e0_quote.get("valid_for_signal") is True
                    and final_quote.get("valid_for_signal") is True
                    and isinstance(e0_quote.get("ask"), (int, float))
                    and float(e0_quote["ask"]) > 0.0
                    and isinstance(final_quote.get("bid"), (int, float))
                    and float(final_quote["bid"]) > 0.0
                ):
                    valid_signals.append(signal)
            valid_sessions = {str(row["session_date"]) for row in valid_signals}
            support_by_checkpoint[support_checkpoint] = {
                "valid_sessions": len(valid_sessions),
                "calendar_months": len({value[:7] for value in valid_sessions}),
                "distinct_selected_stocks": len(
                    {
                        str(row["selected_symbol"])
                        for row in valid_signals
                        if row.get("selected_symbol") is not None
                    }
                ),
            }
        decoded = [self._decoded(row) for row in rows]
        checkpoints: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        support_complete = True
        for checkpoint, role in ((6, "primary"), (12, "secondary")):
            checkpoint_rows = [row for row in decoded if int(row["checkpoint"]) == checkpoint]
            projection = self._opening_leader_checkpoint_projection(
                checkpoint_rows,
                role=role,
                support=support_by_checkpoint.get(
                    checkpoint,
                    {
                        "valid_sessions": 0,
                        "calendar_months": 0,
                        "distinct_selected_stocks": 0,
                    },
                ),
            )
            checkpoints[f"C{checkpoint}"] = projection
            warnings.extend(str(item) for item in projection["data_quality_warnings"])
            support_complete = support_complete and bool(projection["support"]["complete"])
        return {
            **empty,
            "recorder_status": "recording" if decoded else "waiting_for_signal",
            "latest_session": latest_session,
            "sample_status": (
                "PROSPECTIVE SUPPORT COMPLETE"
                if support_complete
                else "PROSPECTIVE SAMPLE INCOMPLETE"
            ),
            "checkpoints": checkpoints,
            "data_quality_warnings": list(dict.fromkeys(warnings)),
        }

    @staticmethod
    def _empty_opening_leader_checkpoint(role: str) -> dict[str, Any]:
        return {
            "role": role,
            "eligibility": "not_observed",
            "slate_size": None,
            "rank_1": None,
            "rank_2": None,
            "rank_1_return_from_open_bps": None,
            "leader_separation_bps": None,
            "signal_receipt": None,
            "source_feed_status": None,
            "observations": {},
            "latest_hypothetical_underlying_return": None,
            "rank_persistence": None,
            "m1c_context": None,
            "option_snapshots": {},
            "option_strategy_accounting": {},
            "pre_close_observations": {},
            "final_continuous_observation": None,
            "official_close_reference": None,
            "support": {
                "valid_sessions": 0,
                "required_valid_sessions": 60,
                "calendar_months": 0,
                "required_calendar_months": 3,
                "distinct_selected_stocks": 0,
                "required_distinct_selected_stocks": 15,
                "complete": False,
            },
            "data_quality_warnings": [],
        }

    @classmethod
    def _opening_leader_checkpoint_projection(
        cls,
        rows: list[dict[str, Any]],
        *,
        role: str,
        support: dict[str, int],
    ) -> dict[str, Any]:
        projection = cls._empty_opening_leader_checkpoint(role)
        original = [row for row in rows if row.get("original_stable_id") is None]
        linked_by_original = {
            str(row["original_stable_id"]): row
            for row in rows
            if row.get("original_stable_id") is not None
        }
        signal = next(
            (row for row in original if row["record_type"] == "signal_receipt"),
            None,
        )
        failure = next(
            (row for row in original if row["record_type"] == "signal_failure"),
            None,
        )
        support_projection = {
            **support,
            "required_valid_sessions": 60,
            "required_calendar_months": 3,
            "required_distinct_selected_stocks": 15,
            "complete": (
                support["valid_sessions"] >= 60
                and support["calendar_months"] >= 3
                and support["distinct_selected_stocks"] >= 15
            ),
        }
        if signal is None:
            projection.update(
                {
                    "eligibility": "failed" if failure is not None else "not_observed",
                    "signal_receipt": None if failure is None else failure["stable_id"],
                    "support": support_projection,
                    "data_quality_warnings": (
                        [] if failure is None else list(failure.get("data_quality_flags", ()))
                    ),
                }
            )
            return projection
        signal_payload = signal.get("payload") or {}
        ranking = signal_payload.get("ranking") or {}
        signal_quote = signal_payload.get("signal_quote") or {}
        original_observation_rows = [
            row for row in original if row["record_type"] == "underlying_observation"
        ]
        observation_rows = [
            linked_by_original.get(str(row["stable_id"]), row) for row in original_observation_rows
        ]
        observations = {
            str(row["observation_name"]): row.get("payload") for row in observation_rows
        }
        latest_quote = next(
            (
                payload.get("quote")
                for payload in reversed(list(observations.values()))
                if isinstance(payload, dict) and payload.get("quote") is not None
            ),
            signal_quote,
        )
        if not isinstance(latest_quote, dict):
            latest_quote = {}
        latest_shadow = next(
            (
                payload.get("shadow_return")
                for payload in reversed(list(observations.values()))
                if isinstance(payload, dict) and payload.get("shadow_return") is not None
            ),
            None,
        )
        latest_persistence = next(
            (
                payload.get("rank_persistence")
                for payload in reversed(list(observations.values()))
                if isinstance(payload, dict) and payload.get("rank_persistence") is not None
            ),
            None,
        )
        option_snapshots = {
            str(row["observation_name"]): row.get("payload")
            for row in original
            if row["record_type"] == "option_snapshot"
        }
        option_strategy_accounting: dict[str, dict[str, Any]] = {}
        for row in original:
            if row["record_type"] != "option_strategy_accounting":
                continue
            payload = row.get("payload") or {}
            observation_name = str(payload.get("observation_name") or "UNKNOWN")
            strategy_name = str(payload.get("strategy_name") or "UNKNOWN")
            option_strategy_accounting.setdefault(observation_name, {})[strategy_name] = payload
        original_official = next(
            (row for row in original if row["record_type"] == "official_close_reference"),
            None,
        )
        official = (
            None
            if original_official is None
            else linked_by_original.get(
                str(original_official["stable_id"]),
                original_official,
            ).get("payload")
        )
        warning_values = [
            str(flag) for row in rows for flag in row.get("data_quality_flags", ()) if flag
        ]
        warning_values.extend(
            f"linked_{row['record_type']}:{row['stable_id']}"
            for row in rows
            if row.get("original_stable_id") is not None
        )
        rank_1 = ranking.get("rank_1") or {}
        rank_2 = ranking.get("rank_2") or {}
        projection.update(
            {
                "eligibility": "eligible",
                "slate_size": ranking.get("slate_size"),
                "rank_1": rank_1.get("symbol"),
                "rank_2": rank_2.get("symbol"),
                "rank_1_return_from_open_bps": rank_1.get("open_to_checkpoint_return_bps"),
                "leader_separation_bps": ranking.get("rank_1_minus_rank_2_bps"),
                "signal_receipt": signal["stable_id"],
                "source_feed_status": latest_quote.get("market_data_status"),
                "observations": {
                    name: observations.get(name) for name in ("SIGNAL", "E0", "E1", "E2")
                },
                "latest_hypothetical_underlying_return": latest_shadow,
                "rank_persistence": latest_persistence,
                "m1c_context": rank_1.get("m1c_context"),
                "option_snapshots": option_snapshots,
                "option_strategy_accounting": option_strategy_accounting,
                "pre_close_observations": {
                    name: observations.get(name)
                    for name in ("PRE_CLOSE_30", "PRE_CLOSE_15", "PRE_CLOSE_5", "PRE_CLOSE_1")
                },
                "final_continuous_observation": observations.get("FINAL_CONTINUOUS"),
                "official_close_reference": official,
                "support": support_projection,
                "data_quality_warnings": list(dict.fromkeys(warning_values)),
            }
        )
        return projection
