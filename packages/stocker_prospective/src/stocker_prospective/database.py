"""SQLite WAL persistence and append-oriented prospective repositories."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.bundle import SCIENTIFIC_CLASSIFICATION


class RecorderLeaseHeld(RuntimeError):
    """Another recorder currently owns the database lease."""


class EvidenceMetadata(BaseModel):
    """Metadata flattened into every exported runtime evidence record."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    prospective_start_utc: datetime
    app_version: str
    git_commit: str
    model_artifact_id: str
    universe_id: str
    cohort: str
    source_timestamps: list[str]
    recorded_at_utc: datetime

    @field_validator("prospective_start_utc", "recorded_at_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evidence timestamps must be timezone-aware")
        return value.astimezone(UTC)


class ScoreInput(BaseModel):
    """Completed-bar score at the public eventisation seam."""

    model_config = ConfigDict(extra="forbid")

    metadata: EvidenceMetadata
    symbol: str
    bar_end_utc: datetime
    session_date: date
    feature_as_of_utc: datetime
    m0_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    m1_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    frozen_threshold: float = Field(ge=0.0, le=1.0)
    feature_schema_hash: str
    eligibility: bool
    rejection_reason: str | None
    score_label: str


class StoredScore(BaseModel):
    id: int
    input: ScoreInput


class LeaseRecord(BaseModel):
    run_id: str
    owner_id: str
    acquired_at_utc: datetime
    heartbeat_at_utc: datetime
    generation: int
    recovered_stale_owner: bool


class ProspectiveRepository:
    """Single-writer SQLite repository with explicit migrations."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def migrate(self) -> None:
        """Apply package migrations exactly once."""

        migration_root = Path(__file__).with_name("migrations")
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at_utc TEXT NOT NULL
                )
                """
            )
            applied = {
                str(row["version"])
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for path in sorted(migration_root.glob("*.sql")):
                if path.name in applied:
                    continue
                connection.executescript(path.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (path.name, datetime.now(UTC).isoformat()),
                )

    @staticmethod
    def _validate_metadata(metadata: EvidenceMetadata) -> None:
        if metadata.recorded_at_utc < metadata.prospective_start_utc:
            raise ValueError("recorded_at_utc precedes configured prospective_start_utc")

    def create_run(
        self,
        metadata: EvidenceMetadata,
        *,
        mode: Literal["record_only", "shadow"] = "record_only",
    ) -> None:
        self._validate_metadata(metadata)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO prospective_run(
                    run_id, prospective_start_utc, app_version, git_commit,
                    model_artifact_id, universe_id, cohort, created_at_utc,
                    mode, scientific_classification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.run_id,
                    metadata.prospective_start_utc.isoformat(),
                    metadata.app_version,
                    metadata.git_commit,
                    metadata.model_artifact_id,
                    metadata.universe_id,
                    metadata.cohort,
                    metadata.recorded_at_utc.isoformat(),
                    mode,
                    SCIENTIFIC_CLASSIFICATION,
                ),
            )

    def _insert_envelope(
        self,
        connection: sqlite3.Connection,
        metadata: EvidenceMetadata,
    ) -> int:
        self._validate_metadata(metadata)
        cursor = connection.execute(
            """
            INSERT INTO evidence_envelope(
                run_id, prospective_start_utc, app_version, git_commit,
                model_artifact_id, universe_id, cohort, source_timestamps_json,
                recorded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.run_id,
                metadata.prospective_start_utc.isoformat(),
                metadata.app_version,
                metadata.git_commit,
                metadata.model_artifact_id,
                metadata.universe_id,
                metadata.cohort,
                json.dumps(metadata.source_timestamps, separators=(",", ":")),
                metadata.recorded_at_utc.isoformat(),
            ),
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    def acquire_recorder_lease(
        self,
        *,
        run_id: str,
        owner_id: str,
        now: datetime,
        stale_after: timedelta,
    ) -> LeaseRecord:
        """Acquire the singleton recorder lease or reject a concurrent owner."""

        now = now.astimezone(UTC)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM recorder_lease WHERE lease_key = 'prospective_recorder'"
            ).fetchone()
            recovered = False
            if row is not None:
                heartbeat = datetime.fromisoformat(str(row["heartbeat_at_utc"]))
                same_owner = row["owner_id"] == owner_id and row["run_id"] == run_id
                stale = heartbeat <= now - stale_after
                if not same_owner and not stale:
                    raise RecorderLeaseHeld(
                        f"blocked_recorder_lease_held: owner={row['owner_id']} run={row['run_id']}"
                    )
                recovered = not same_owner and stale
                acquired = now.isoformat() if recovered else str(row["acquired_at_utc"])
                generation = int(row["generation"]) + (1 if recovered else 0)
                connection.execute(
                    """
                    UPDATE recorder_lease
                    SET run_id = ?, owner_id = ?, acquired_at_utc = ?,
                        heartbeat_at_utc = ?, generation = ?, recovered_stale_owner = ?
                    WHERE lease_key = 'prospective_recorder'
                    """,
                    (
                        run_id,
                        owner_id,
                        acquired,
                        now.isoformat(),
                        generation,
                        int(recovered),
                    ),
                )
            else:
                acquired = now.isoformat()
                generation = 1
                connection.execute(
                    """
                    INSERT INTO recorder_lease(
                        lease_key, run_id, owner_id, acquired_at_utc,
                        heartbeat_at_utc, generation, recovered_stale_owner
                    ) VALUES ('prospective_recorder', ?, ?, ?, ?, ?, 0)
                    """,
                    (run_id, owner_id, acquired, now.isoformat(), generation),
                )
            connection.commit()
            return LeaseRecord(
                run_id=run_id,
                owner_id=owner_id,
                acquired_at_utc=datetime.fromisoformat(acquired),
                heartbeat_at_utc=now,
                generation=generation,
                recovered_stale_owner=recovered,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat_recorder_lease(
        self,
        *,
        run_id: str,
        owner_id: str,
        now: datetime,
    ) -> LeaseRecord:
        now = now.astimezone(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE recorder_lease SET heartbeat_at_utc = ?
                WHERE lease_key = 'prospective_recorder' AND run_id = ? AND owner_id = ?
                """,
                (now.isoformat(), run_id, owner_id),
            )
            if cursor.rowcount != 1:
                raise RecorderLeaseHeld("blocked_recorder_lease_held: lease ownership changed")
            row = connection.execute(
                "SELECT * FROM recorder_lease WHERE lease_key = 'prospective_recorder'"
            ).fetchone()
        assert row is not None
        return LeaseRecord(
            run_id=str(row["run_id"]),
            owner_id=str(row["owner_id"]),
            acquired_at_utc=datetime.fromisoformat(str(row["acquired_at_utc"])),
            heartbeat_at_utc=now,
            generation=int(row["generation"]),
            recovered_stale_owner=bool(row["recovered_stale_owner"]),
        )

    def record_score(self, score: ScoreInput) -> StoredScore:
        """Idempotently append one model score."""

        self._validate_metadata(score.metadata)
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM model_score
                WHERE run_id = ? AND cohort = ? AND symbol = ?
                  AND model_bundle_id = ? AND bar_end_utc = ?
                """,
                (
                    score.metadata.run_id,
                    score.metadata.cohort,
                    score.symbol,
                    score.metadata.model_artifact_id,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                ),
            ).fetchone()
            if existing is not None:
                return StoredScore(id=int(existing["id"]), input=score)
            envelope_id = self._insert_envelope(connection, score.metadata)
            cursor = connection.execute(
                """
                INSERT INTO model_score(
                    envelope_id, run_id, cohort, symbol, bar_end_utc, session_date,
                    feature_as_of_utc, m0_probability, m1_probability, frozen_threshold,
                    model_bundle_id, feature_schema_hash, eligibility, rejection_reason,
                    score_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    score.metadata.run_id,
                    score.metadata.cohort,
                    score.symbol,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                    score.session_date.isoformat(),
                    score.feature_as_of_utc.astimezone(UTC).isoformat(),
                    score.m0_probability,
                    score.m1_probability,
                    score.frozen_threshold,
                    score.metadata.model_artifact_id,
                    score.feature_schema_hash,
                    int(score.eligibility),
                    score.rejection_reason,
                    score.score_label,
                ),
            )
            assert cursor.lastrowid is not None
            return StoredScore(id=int(cursor.lastrowid), input=score)

    def previous_eligible_score(self, score: ScoreInput) -> sqlite3.Row | None:
        with self._connect() as connection:
            row: sqlite3.Row | None = connection.execute(
                """
                SELECT * FROM model_score
                WHERE run_id = ? AND cohort = ? AND symbol = ? AND model_bundle_id = ?
                  AND eligibility = 1 AND bar_end_utc < ?
                ORDER BY bar_end_utc DESC LIMIT 1
                """,
                (
                    score.metadata.run_id,
                    score.metadata.cohort,
                    score.symbol,
                    score.metadata.model_artifact_id,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                ),
            ).fetchone()
        return row

    def latest_signal_episode(self, score: ScoreInput) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM signal_episode
                WHERE run_id = ? AND cohort = ? AND symbol = ? AND model_bundle_id = ?
                  AND crossing_timestamp_utc <= ?
                ORDER BY crossing_timestamp_utc DESC LIMIT 1
                """,
                (
                    score.metadata.run_id,
                    score.metadata.cohort,
                    score.symbol,
                    score.metadata.model_artifact_id,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                ),
            ).fetchone()
        return None if row is None else str(row["id"])

    def create_signal_episode(self, stored: StoredScore) -> str:
        score = stored.input
        crossing = score.bar_end_utc.astimezone(UTC).isoformat()
        raw_key = "|".join(
            (
                score.metadata.run_id,
                score.metadata.cohort,
                score.symbol,
                score.metadata.model_artifact_id,
                crossing,
            )
        )
        idempotency_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        episode_id = f"sig-{idempotency_key[:24]}"
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM signal_episode WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is None:
                envelope_id = self._insert_envelope(connection, score.metadata)
                connection.execute(
                    """
                    INSERT INTO signal_episode(
                        id, envelope_id, run_id, cohort, symbol, model_bundle_id,
                        crossing_timestamp_utc, idempotency_key, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    """,
                    (
                        episode_id,
                        envelope_id,
                        score.metadata.run_id,
                        score.metadata.cohort,
                        score.symbol,
                        score.metadata.model_artifact_id,
                        crossing,
                        idempotency_key,
                    ),
                )
            else:
                episode_id = str(existing["id"])
        self.add_signal_checkpoint(episode_id, stored)
        return episode_id

    def add_signal_checkpoint(self, episode_id: str, stored: StoredScore) -> None:
        score = stored.input
        assert score.m1_probability is not None
        with self._connect() as connection:
            exists = connection.execute(
                """
                SELECT 1 FROM signal_checkpoint
                WHERE signal_episode_id = ? AND model_score_id = ?
                """,
                (episode_id, stored.id),
            ).fetchone()
            if exists is not None:
                return
            envelope_id = self._insert_envelope(connection, score.metadata)
            connection.execute(
                """
                INSERT INTO signal_checkpoint(
                    envelope_id, run_id, signal_episode_id, model_score_id,
                    checkpoint_timestamp_utc, m1_probability, frozen_threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    score.metadata.run_id,
                    episode_id,
                    stored.id,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                    score.m1_probability,
                    score.frozen_threshold,
                ),
            )

    def record_eventization(
        self,
        stored: StoredScore,
        *,
        status: str,
        episode_id: str | None,
    ) -> None:
        """Append the eventisation decision, including startup-above state."""

        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM signal_eventization WHERE model_score_id = ?",
                (stored.id,),
            ).fetchone()
            if exists is not None:
                return
            envelope_id = self._insert_envelope(connection, stored.input.metadata)
            connection.execute(
                """
                INSERT INTO signal_eventization(
                    envelope_id, run_id, model_score_id, eventization_status,
                    signal_episode_id, recorded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    stored.input.metadata.run_id,
                    stored.id,
                    status,
                    episode_id,
                    stored.input.metadata.recorded_at_utc.isoformat(),
                ),
            )

    def count(self, table: str) -> int:
        allowed = {
            "signal_episode",
            "signal_checkpoint",
            "model_score",
            "evidence_envelope",
            "signal_eventization",
        }
        if table not in allowed:
            raise ValueError("unsupported count table")
        with self._connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        assert row is not None
        return int(row["count"])
