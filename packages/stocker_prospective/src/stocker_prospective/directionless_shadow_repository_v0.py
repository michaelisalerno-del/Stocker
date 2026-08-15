"""Additive SQLite persistence for directionless shadow state and evidence links."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime

from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.m1c_directionless_shadow_v0 import (
    TERMINAL_STATES_V0,
    DirectionlessShadowEpisodeV0,
)


class DirectionlessShadowRepositoryV0:
    def __init__(self, repository: ProspectiveRepository, *, run_id: str) -> None:
        self.repository = repository
        self.run_id = run_id

    def record_run_contract(
        self,
        *,
        activation_timestamp: datetime,
        first_eligible_session: date,
        contract_sha256: str,
        config_hash: str,
        code_hash: str,
        m1c_artifact_hash: str,
    ) -> None:
        values = (
            self.run_id,
            "m1c_directionless_t20_d_v0",
            activation_timestamp.astimezone(UTC).isoformat(),
            first_eligible_session.isoformat(),
            20,
            "LEVEL1_BBO_EXECUTABLE_SIDE",
            contract_sha256,
            config_hash,
            code_hash,
            m1c_artifact_hash,
            0,
            0,
            0,
        )
        with self.repository._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM m1c_directionless_shadow_run_v0 WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
            if existing is not None and tuple(existing) != values:
                raise ValueError("immutable directionless shadow run contract differs")
            connection.execute(
                "INSERT OR IGNORE INTO m1c_directionless_shadow_run_v0 VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    def save(
        self,
        episode: DirectionlessShadowEpisodeV0,
        *,
        source_partition_reference: str | None = None,
        connection_generation: int | None = None,
        record_transition: bool = True,
    ) -> None:
        payload = episode.model_dump_json()
        timestamp = episode.updated_at.astimezone(UTC).isoformat()
        transition_material = "|".join(
            (
                episode.shadow_episode_id,
                episode.state.value,
                episode.last_source_event_id or "timer",
                timestamp,
            )
        )
        transition_id = hashlib.sha256(transition_material.encode()).hexdigest()
        with self.repository._connect() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM m1c_directionless_shadow_v0 WHERE shadow_episode_id = ?",
                (episode.shadow_episode_id,),
            ).fetchone()
            if existing is not None:
                persisted = DirectionlessShadowEpisodeV0.model_validate_json(
                    existing["payload_json"]
                )
                if persisted.terminal and persisted != episode:
                    raise ValueError("terminal directionless shadow episode is immutable")
                if episode.updated_at < persisted.updated_at:
                    raise ValueError("directionless shadow state cannot move backwards")
            connection.execute(
                """
                INSERT INTO m1c_directionless_shadow_v0 (
                    shadow_episode_id, strategy_version, run_id, m1c_episode_id,
                    symbol, con_id, session_date, t0_utc, p0, movement_price,
                    family, probability, consumed_ratio, upper_trigger, lower_trigger,
                    state, direction, entry_timestamp_utc, entry_price,
                    entry_source_event_id, stop_price, target_price,
                    exit_timestamp_utc, exit_price, exit_reason, exit_source_event_id,
                    payload_json, created_at_utc, updated_at_utc, code_hash,
                    config_hash, m1c_artifact_hash
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(shadow_episode_id) DO UPDATE SET
                    state = excluded.state,
                    direction = excluded.direction,
                    entry_timestamp_utc = excluded.entry_timestamp_utc,
                    entry_price = excluded.entry_price,
                    entry_source_event_id = excluded.entry_source_event_id,
                    stop_price = excluded.stop_price,
                    target_price = excluded.target_price,
                    exit_timestamp_utc = excluded.exit_timestamp_utc,
                    exit_price = excluded.exit_price,
                    exit_reason = excluded.exit_reason,
                    exit_source_event_id = excluded.exit_source_event_id,
                    payload_json = excluded.payload_json,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    episode.shadow_episode_id,
                    episode.strategy_version,
                    self.run_id,
                    episode.m1c_episode_id,
                    episode.symbol,
                    episode.con_id,
                    episode.session.isoformat(),
                    episode.t0.isoformat(),
                    episode.p0,
                    episode.movement_price,
                    episode.family,
                    episode.probability,
                    episode.consumed_ratio,
                    episode.upper_trigger,
                    episode.lower_trigger,
                    episode.state.value,
                    episode.direction,
                    None
                    if episode.entry_timestamp is None
                    else episode.entry_timestamp.isoformat(),
                    episode.entry_price,
                    episode.entry_source_event_id,
                    episode.stop_price,
                    episode.target_price,
                    None if episode.exit_timestamp is None else episode.exit_timestamp.isoformat(),
                    episode.exit_price,
                    episode.exit_reason,
                    episode.exit_source_event_id,
                    payload,
                    episode.created_at.isoformat(),
                    timestamp,
                    episode.code_hash,
                    episode.config_hash,
                    episode.m1c_artifact_hash,
                ),
            )
            if record_transition:
                connection.execute(
                    """
                INSERT OR IGNORE INTO m1c_directionless_shadow_transition_v0 (
                    transition_id, shadow_episode_id, run_id, state,
                    transition_timestamp_utc, source_event_id,
                    source_partition_reference, connection_generation, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transition_id,
                        episode.shadow_episode_id,
                        self.run_id,
                        episode.state.value,
                        timestamp,
                        episode.last_source_event_id,
                        source_partition_reference,
                        connection_generation,
                        json.dumps(
                            {
                                "state": episode.state.value,
                                "direction": episode.direction,
                                "entry_gap": episode.entry_gap,
                                "exit_reason": episode.exit_reason,
                                "gap_id": episode.gap_id,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )

    def load(self, m1c_episode_id: str) -> DirectionlessShadowEpisodeV0 | None:
        with self.repository._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM m1c_directionless_shadow_v0 "
                "WHERE run_id = ? AND m1c_episode_id = ? "
                "AND strategy_version = 'm1c_directionless_t20_d_v0'",
                (self.run_id, m1c_episode_id),
            ).fetchone()
        return (
            None
            if row is None
            else DirectionlessShadowEpisodeV0.model_validate_json(row["payload_json"])
        )

    def load_active(self) -> tuple[DirectionlessShadowEpisodeV0, ...]:
        terminal = tuple(state.value for state in TERMINAL_STATES_V0)
        placeholders = ",".join("?" for _ in terminal)
        with self.repository._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM m1c_directionless_shadow_v0 "
                f"WHERE run_id = ? AND state NOT IN ({placeholders}) ORDER BY t0_utc",
                (self.run_id, *terminal),
            ).fetchall()
        return tuple(
            DirectionlessShadowEpisodeV0.model_validate_json(row["payload_json"]) for row in rows
        )


__all__ = ["DirectionlessShadowRepositoryV0"]
