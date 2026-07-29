"""Read-only SQLite projections for the Stocker web process."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stocker_prospective.database import SchemaVersionTooNew
from stocker_prospective.operational_state import (
    OperationalStateProjection,
    OperationalThresholds,
    inactive_operational_projection,
    project_operational_state_from_database,
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
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
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
                FROM data_health_event AS blocked
                WHERE run_id = ? AND blocker_code IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1
                    FROM data_health_event AS resolved
                    WHERE resolved.run_id = blocked.run_id
                      AND resolved.component = blocked.component
                      AND resolved.id > blocked.id
                      AND resolved.blocker_code IS NULL
                      AND resolved.message = 'previous_session_options_context_ready'
                  )
                ORDER BY id
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
                SELECT MAX(maximum_timestamp_utc) AS last_event_timestamp
                FROM raw_partition_manifest_v0 WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            gaps = connection.execute(
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
            subscriptions = connection.execute(
                """
                SELECT subscription_kind, COUNT(*) AS used
                FROM subscription_lifecycle_v0
                WHERE run_id = ? AND cancelled_at_utc IS NULL
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
                0
                if gaps is None or gaps["unresolved_scientific_gaps"] is None
                else int(gaps["unresolved_scientific_gaps"])
            ),
            "gaps": {
                name: (0 if gaps is None or gaps[name] is None else int(gaps[name]))
                for name in (
                    "active_gaps",
                    "resolved_recoverable_gaps",
                    "unresolved_scientific_gaps",
                    "connection_interruptions",
                    "optional_feed_degradations",
                )
            },
            "subscriptions": {
                str(row["subscription_kind"]): int(row["used"]) for row in subscriptions
            },
            "ibkr_connection": self._dict(connection_event),
            "record_only": True,
            "execution_enabled": False,
            "order_routing": "disabled",
        }

    def universe_live_v0(self) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                WITH latest_checkpoint AS (
                    SELECT c.*
                    FROM m1c_checkpoint_v0 c
                    JOIN (
                        SELECT symbol, MAX(bar_end_utc) AS maximum_bar_end
                        FROM m1c_checkpoint_v0 WHERE run_id = ? GROUP BY symbol
                    ) x ON x.symbol = c.symbol
                       AND x.maximum_bar_end = c.bar_end_utc
                    WHERE c.run_id = ?
                ),
                latest_episode AS (
                    SELECT e.*
                    FROM m1c_episode_v0 e
                    JOIN (
                        SELECT symbol, MAX(trigger_bar_end_utc) AS maximum_trigger
                        FROM m1c_episode_v0 WHERE run_id = ? GROUP BY symbol
                    ) x ON x.symbol = e.symbol
                       AND x.maximum_trigger = e.trigger_bar_end_utc
                    WHERE e.run_id = ?
                ),
                latest_legacy_quote AS (
                    SELECT q.*
                    FROM underlying_quote q
                    JOIN (
                        SELECT symbol, MAX(target_timestamp_utc) AS maximum_quote
                        FROM underlying_quote WHERE run_id = ? GROUP BY symbol
                    ) x ON x.symbol = q.symbol
                       AND x.maximum_quote = q.target_timestamp_utc
                    WHERE q.run_id = ?
                )
                SELECT u.symbol, u.operational_status,
                       b.bar_end_utc AS last_completed_bar,
                       c.probability AS m1c_probability,
                       c.threshold AS m1c_threshold,
                       c.threshold_passed,
                       e.episode_id, e.completion_status AS episode_status,
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
                LEFT JOIN latest_checkpoint c ON c.symbol = u.symbol
                LEFT JOIN latest_episode e ON e.symbol = u.symbol
                LEFT JOIN completed_bar_state_v0 b
                  ON b.symbol = u.symbol AND b.run_id = u.run_id
                LEFT JOIN latest_legacy_quote q ON q.symbol = u.symbol
                LEFT JOIN underlying_live_state_v0 s
                  ON s.symbol = u.symbol AND s.run_id = u.run_id
                WHERE u.run_id = ?
                ORDER BY u.symbol
                """,
                (run_id, run_id, run_id, run_id, run_id, run_id, run_id),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            probability = item.get("m1c_probability")
            threshold = item.get("m1c_threshold")
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
            item["fresh_episode"] = item.get("episode_id") is not None
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
    ) -> list[dict[str, Any]]:
        """Read a bounded chart projection from immutable quote partitions."""

        if maximum_points <= 0:
            raise ValueError("maximum quote-series points must be positive")
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
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return []
        points: dict[str, dict[str, Any]] = {}
        for partition in partitions:
            path = Path(str(partition["file_path"]))
            if not path.is_file():
                continue
            table = pq.ParquetFile(path).read()  # type: ignore[no-untyped-call]
            for row in table.to_pylist():
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
                total_size = (
                    None
                    if bid_size is None or ask_size is None
                    else float(bid_size) + float(ask_size)
                )
                microprice = (
                    None
                    if total_size is None or total_size <= 0.0
                    else (ask_value * float(bid_size) + bid_value * float(ask_size)) / total_size
                )
                points[str(row.get("event_id"))] = {
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
        stride = max(1, math.ceil(len(ordered) / maximum_points))
        sampled = ordered[::stride]
        if ordered and sampled[-1] != ordered[-1]:
            sampled.append(ordered[-1])
        return sampled[:maximum_points]

    def episode_depth_snapshot_v0(self, episode_id: str) -> dict[str, Any] | None:
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
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return None
        candidates: list[dict[str, Any]] = []
        for partition in partitions:
            path = Path(str(partition["file_path"]))
            if not path.is_file():
                continue
            table = pq.ParquetFile(path).read()  # type: ignore[no-untyped-call]
            for row in table.to_pylist():
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
        """Return only strict V1.1 predicted-leg virtual position evidence."""

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

    def raw_event_sample_v0(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return a bounded recent sample for audit inspection, never broker state."""

        if limit <= 0:
            return []
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            partitions = connection.execute(
                """
                SELECT file_path, event_type
                FROM raw_partition_manifest_v0
                WHERE run_id = ?
                ORDER BY maximum_timestamp_utc DESC, content_hash DESC
                LIMIT 12
                """,
                (run_id,),
            ).fetchall()
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return []
        sample: list[dict[str, Any]] = []
        for partition in partitions:
            path = Path(str(partition["file_path"]))
            try:
                is_file = path.is_file()
            except OSError:
                is_file = False
            if not is_file:
                continue
            try:
                rows = pq.ParquetFile(path).read().to_pylist()  # type: ignore[no-untyped-call]
            except OSError:
                continue
            for row in rows[-limit:]:
                timestamp = row.get("provider_timestamp_utc") or row.get("received_timestamp_utc")
                detail = {
                    key: row.get(key)
                    for key in (
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
                    )
                    if row.get(key) is not None
                }
                sample.append(
                    {
                        "audit_type": "raw_event",
                        "identity": row.get("event_id"),
                        "recorded_at_utc": timestamp,
                        "details": json.dumps(
                            {
                                "event_type": str(partition["event_type"]),
                                "symbol": row.get("symbol"),
                                "source_sequence": row.get("source_sequence"),
                                **detail,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
        return sorted(
            sample,
            key=lambda item: str(item["recorded_at_utc"]),
            reverse=True,
        )[:limit]

    def audit_events_v0(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        run_id = self._selected_run_id()
        if run_id is None:
            return []
        with self._connection() as connection:
            partitions = connection.execute(
                """
                SELECT 'raw_partition' AS audit_type, content_hash AS identity,
                       maximum_timestamp_utc AS recorded_at_utc,
                       file_path AS details
                FROM raw_partition_manifest_v0 WHERE run_id = ?
                ORDER BY maximum_timestamp_utc DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
            lifecycle = connection.execute(
                """
                SELECT 'subscription' AS audit_type,
                       subscription_key AS identity,
                       started_at_utc AS recorded_at_utc,
                       cancellation_reason AS details
                FROM subscription_lifecycle_v0 WHERE run_id = ?
                ORDER BY started_at_utc DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
            model_events = connection.execute(
                """
                SELECT 'm1c_prediction' AS audit_type,
                       feature_hash AS identity,
                       bar_end_utc AS recorded_at_utc,
                       json_object(
                         'symbol', symbol,
                         'checkpoint', checkpoint,
                         'model_hash', model_hash,
                         'probability', probability,
                         'threshold', threshold,
                         'threshold_passed', threshold_passed,
                         'eligible', eligible,
                         'rejection_reasons', rejection_reasons_json
                       ) AS details
                FROM m1c_checkpoint_v0 WHERE run_id = ?
                ORDER BY bar_end_utc DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
            episode_events = connection.execute(
                """
                SELECT 'episode_decision' AS audit_type,
                       episode_id AS identity,
                       trigger_bar_end_utc AS recorded_at_utc,
                       json_object(
                         'symbol', symbol,
                         'checkpoint', trigger_checkpoint,
                         'entry', prospective_entry_timestamp_utc,
                         'probability', m1c_probability,
                         'previous_probability', previous_m1c_probability,
                         'scientific_recording_valid', scientific_recording_valid,
                         'rejection_reasons', rejection_reasons_json
                       ) AS details
                FROM m1c_episode_v0 WHERE run_id = ?
                ORDER BY trigger_bar_end_utc DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
            shadow_events = connection.execute(
                """
                SELECT 'shadow_quote_selection' AS audit_type,
                       episode_id || ':' || archetype || ':' ||
                         contract_identity || ':' || horizon_minutes AS identity,
                       target_timestamp_utc AS recorded_at_utc,
                       payload_json AS details
                FROM shadow_quote_outcome_v0 WHERE run_id = ?
                ORDER BY target_timestamp_utc DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        combined = [
            *self.raw_event_sample_v0(limit=min(limit, 100)),
            *(
                dict(row)
                for row in (
                    *partitions,
                    *lifecycle,
                    *model_events,
                    *episode_events,
                    *shadow_events,
                )
            ),
        ]
        return sorted(
            combined,
            key=lambda item: str(item["recorded_at_utc"]),
            reverse=True,
        )[:limit]

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
                "engineering_transfer_valid_sessions": 20,
                "option_development_complete_quiet_episodes": 150,
                "untouched_confirmation_complete_quiet_episodes": 150,
            },
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
                       q.data_quality_status AS checkpoint_data_quality_status
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
            return {"sessions": [], "valid_session_count": 0, "decision": None}
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM source_transfer_session_v0
                WHERE run_id = ? ORDER BY session_date
                """,
                (run_id,),
            ).fetchall()
        sessions = [self._decoded(row) for row in rows]
        latest = None if not sessions else sessions[-1]
        return {
            "sessions": sessions,
            "valid_session_count": sum(bool(row.get("valid")) for row in sessions),
            "decision": None if latest is None else latest.get("decision"),
        }
