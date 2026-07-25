"""Read-only SQLite projections for the Stocker web process."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ProspectiveReadStore:
    """A query-only store that opens SQLite with ``mode=ro``."""

    def __init__(self, database_path: str | Path, *, run_id: str | None = None) -> None:
        self.database_path = Path(database_path)
        self.run_id = run_id
        self._anchor: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.database_path.resolve()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 2000")
        return connection

    def open_anchor(self) -> None:
        """Keep WAL coordination files live for the read-only web process."""

        if self._anchor is not None:
            return
        connection = self._connect()
        try:
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

    def database_health(self) -> dict[str, Any]:
        try:
            with self._connect() as connection:
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

    def latest_run(self) -> dict[str, Any] | None:
        with self._connect() as connection:
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
        with self._connect() as connection:
            row = self._run_row(connection)
        return None if row is None else str(row["run_id"])

    def runtime_projection(self) -> dict[str, Any]:
        with self._connect() as connection:
            run = self._run_row(connection)
            lease_run_id = self.run_id if self.run_id is not None else (
                None if run is None else str(run["run_id"])
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
                "SELECT * FROM ibkr_connection_event WHERE run_id = ? ORDER BY id DESC LIMIT 1",
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
            episode = connection.execute(
                "SELECT * FROM signal_episode WHERE run_id = ? "
                "ORDER BY crossing_timestamp_utc DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            blockers = connection.execute(
                "SELECT blocker_code, component, message, severity "
                "FROM data_health_event WHERE run_id = ? AND blocker_code IS NOT NULL "
                "ORDER BY id",
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
        with self._connect() as connection:
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
        with self._connect() as connection:
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
        with self._connect() as connection:
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
        with self._connect() as connection:
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
        with self._connect() as connection:
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
        with self._connect() as connection:
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
