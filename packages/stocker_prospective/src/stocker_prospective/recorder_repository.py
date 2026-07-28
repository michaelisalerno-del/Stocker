"""Append-oriented metadata persistence for the frozen M1C recorder."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

from pydantic import BaseModel

from stocker_prospective.contract import claims_boundary
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.direction import DirectionClassification
from stocker_prospective.direction_features import DirectionFeatureResult
from stocker_prospective.events import (
    FiveMinuteBarEvent,
    OptionQuoteEvent,
    UnderlyingLevel1QuoteEvent,
)
from stocker_prospective.frozen_m1c import EpisodeDecision, FrozenM1CScore
from stocker_prospective.group_o import FrozenGroupOContext
from stocker_prospective.microstructure import MicrostructureWindowSummary
from stocker_prospective.option_budget import EpisodeAllocationRecord
from stocker_prospective.option_ledger import (
    OptionContract,
    OptionContractPlan,
    ShadowOptionOutcome,
)
from stocker_prospective.partition_store import PartitionWriteResult
from stocker_prospective.quality_report import SessionQualityReport
from stocker_prospective.quiet_state import (
    NeutralControlDecision,
    QuietEpisodeDecision,
    QuietStateSnapshot,
)
from stocker_prospective.safety import EpisodeSafetyDecision
from stocker_prospective.subscriptions import PromotionDecision, SubscriptionRecord

TRANSFER_DECISIONS_PERMITTING_OPTION_DEVELOPMENT = frozenset(
    {
        "ibkr_transfer_supported_without_recalibration",
        "ibkr_ranking_supported_probability_scale_shifted",
    }
)


def _json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def _assert_immutable_observation(
    existing: sqlite3.Row,
    *,
    expected: Mapping[str, object],
    label: str,
) -> None:
    mismatches = tuple(key for key, value in expected.items() if existing[key] != value)
    if mismatches:
        raise ValueError(f"immutable {label} differs: {','.join(sorted(mismatches))}")


class FrozenRecorderRepository:
    """Use the existing SQLite database for bounded recorder metadata."""

    def __init__(
        self,
        repository: ProspectiveRepository,
        *,
        configuration_hash: str = "configuration_hash_unavailable",
    ) -> None:
        self.repository = repository
        self.claims_json = _json(claims_boundary())
        self.configuration_hash = configuration_hash

    def recorded_checkpoint_identities(
        self,
        *,
        run_id: str,
    ) -> set[tuple[str, date, int]]:
        """Return immutable checkpoints already recorded for restart deduplication."""

        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT symbol, session_date, checkpoint
                FROM m1c_checkpoint_completion_v0
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
        return {
            (
                str(row["symbol"]),
                date.fromisoformat(str(row["session_date"])),
                int(row["checkpoint"]),
            )
            for row in rows
        }

    def mark_checkpoint_complete(
        self,
        metadata: EvidenceMetadata,
        *,
        checkpoint_id: int,
        symbol: str,
        session: date,
        checkpoint: int,
    ) -> None:
        """Commit the restart marker only after all checkpoint side effects succeed."""

        self._validate(metadata)
        with self.repository._connect() as connection:
            source = connection.execute(
                """
                SELECT run_id, symbol, session_date, checkpoint
                FROM m1c_checkpoint_v0
                WHERE id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            if source is None:
                raise KeyError(checkpoint_id)
            identity = (
                str(source["run_id"]),
                str(source["symbol"]),
                str(source["session_date"]),
                int(source["checkpoint"]),
            )
            expected = (
                metadata.run_id,
                symbol,
                session.isoformat(),
                checkpoint,
            )
            if identity != expected:
                raise ValueError("checkpoint completion identity differs")
            existing = connection.execute(
                """
                SELECT run_id, symbol, session_date, checkpoint
                FROM m1c_checkpoint_completion_v0
                WHERE checkpoint_id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            if existing is not None:
                completed_identity = (
                    str(existing["run_id"]),
                    str(existing["symbol"]),
                    str(existing["session_date"]),
                    int(existing["checkpoint"]),
                )
                if completed_identity != expected:
                    raise ValueError("immutable checkpoint completion differs")
                return
            envelope_id = self.repository._insert_envelope(connection, metadata)
            connection.execute(
                """
                INSERT INTO m1c_checkpoint_completion_v0(
                    checkpoint_id, envelope_id, run_id, symbol, session_date,
                    checkpoint, completed_at_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    envelope_id,
                    metadata.run_id,
                    symbol,
                    session.isoformat(),
                    checkpoint,
                    metadata.recorded_at_utc.astimezone(UTC).isoformat(),
                    self.claims_json,
                ),
            )

    def record_option_episode_schedule(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        checkpoint_id: int,
        symbol: str,
        session: date,
        entry_timestamp: datetime,
        episode_kind: str,
        probability: float,
        quiet_state: bool,
        directional_actions: Mapping[str, str],
        recording_duration: timedelta,
        strike_steps: int,
        maximum_contracts: int,
    ) -> None:
        """Persist option admission inputs before checkpoint completion is marked."""

        self._validate(metadata)
        expected = {
            "run_id": metadata.run_id,
            "symbol": symbol,
            "session_date": session.isoformat(),
            "entry_timestamp_utc": entry_timestamp.astimezone(UTC).isoformat(),
            "episode_kind": episode_kind,
            "probability": probability,
            "quiet_state": int(quiet_state),
            "directional_actions_json": _json(dict(directional_actions)),
            "recording_duration_seconds": int(recording_duration.total_seconds()),
            "strike_steps": strike_steps,
            "maximum_contracts": maximum_contracts,
        }
        with self.repository._connect() as connection:
            source = connection.execute(
                """
                SELECT run_id, symbol, session_date
                FROM m1c_checkpoint_v0
                WHERE id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            if source is None:
                raise KeyError(checkpoint_id)
            if (
                str(source["run_id"]),
                str(source["symbol"]),
                str(source["session_date"]),
            ) != (metadata.run_id, symbol, session.isoformat()):
                raise ValueError("option schedule checkpoint identity differs")
            existing = connection.execute(
                "SELECT * FROM option_episode_schedule_v0 WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if existing is not None:
                _assert_immutable_observation(
                    existing,
                    expected={**expected, "checkpoint_id": checkpoint_id},
                    label="option episode schedule",
                )
                return
            envelope_id = self.repository._insert_envelope(connection, metadata)
            connection.execute(
                """
                INSERT INTO option_episode_schedule_v0(
                    episode_id, checkpoint_id, envelope_id, run_id, symbol,
                    session_date, entry_timestamp_utc, episode_kind, probability,
                    quiet_state, directional_actions_json,
                    recording_duration_seconds, strike_steps, maximum_contracts,
                    status, updated_at_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?)
                """,
                (
                    episode_id,
                    checkpoint_id,
                    envelope_id,
                    expected["run_id"],
                    expected["symbol"],
                    expected["session_date"],
                    expected["entry_timestamp_utc"],
                    expected["episode_kind"],
                    expected["probability"],
                    expected["quiet_state"],
                    expected["directional_actions_json"],
                    expected["recording_duration_seconds"],
                    expected["strike_steps"],
                    expected["maximum_contracts"],
                    metadata.recorded_at_utc.astimezone(UTC).isoformat(),
                    self.claims_json,
                ),
            )

    def update_option_episode_schedule_status(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        status: str,
        degradation_reason: str | None = None,
    ) -> None:
        """Persist the bounded lifecycle needed for safe process reconstruction."""

        if status not in {"scheduled", "streaming", "complete", "rejected", "expired"}:
            raise ValueError("option episode schedule status is invalid")
        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                "SELECT status FROM option_episode_schedule_v0 WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(episode_id)
            current = str(existing["status"])
            if current in {"complete", "rejected", "expired"} and current != status:
                raise ValueError("terminal option episode schedule status differs")
            connection.execute(
                """
                UPDATE option_episode_schedule_v0
                SET status = ?, degradation_reason = ?, updated_at_utc = ?
                WHERE episode_id = ?
                """,
                (
                    status,
                    degradation_reason,
                    metadata.recorded_at_utc.astimezone(UTC).isoformat(),
                    episode_id,
                ),
            )

    def restorable_option_episode_schedules(
        self,
        *,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Load only scheduled or interrupted-streaming option tasks for one run."""

        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM option_episode_schedule_v0
                WHERE run_id = ? AND status IN ('scheduled', 'streaming')
                ORDER BY entry_timestamp_utc, episode_id
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def prospective_phase_for_session(
        self,
        *,
        run_id: str,
        session: date,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[str, bool]:
        """Resolve the immutable cohort phase without consulting any outcome value."""

        owns_connection = connection is None
        active_connection = self.repository._connect() if connection is None else connection
        try:
            transfer_rows = active_connection.execute(
                """
                SELECT session_date, decision
                FROM source_transfer_session_v0
                WHERE run_id = ? AND valid = 1 AND session_date < ?
                ORDER BY session_date
                """,
                (run_id, session.isoformat()),
            ).fetchall()
            if (
                len(transfer_rows) < 20
                or str(transfer_rows[-1]["decision"])
                not in TRANSFER_DECISIONS_PERMITTING_OPTION_DEVELOPMENT
            ):
                return "engineering_transfer", False
            completed_development = int(
                active_connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM quiet_state_observation_v0
                    WHERE run_id = ? AND observation_kind = 'quiet_bottom_10'
                      AND phase = 'option_development'
                      AND completion_status = 'complete'
                    """,
                    (run_id,),
                ).fetchone()[0]
            )
            if completed_development < 150:
                return "option_development", True
            return "untouched_confirmation", True
        finally:
            if owns_connection:
                active_connection.close()

    def _parent_phase(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        parent_table: str,
        parent_id_column: str,
        parent_id: str,
    ) -> tuple[str, bool]:
        if (
            parent_table,
            parent_id_column,
        ) not in {
            ("m1c_episode_v0", "episode_id"),
            ("quiet_state_observation_v0", "observation_id"),
        }:
            raise ValueError("unsupported prospective phase parent")
        row = connection.execute(
            f"""
            SELECT session_date FROM {parent_table}
            WHERE run_id = ? AND {parent_id_column} = ?
            """,
            (run_id, parent_id),
        ).fetchone()
        if row is None:
            raise KeyError(parent_id)
        return self.prospective_phase_for_session(
            run_id=run_id,
            session=date.fromisoformat(str(row["session_date"])),
            connection=connection,
        )

    @staticmethod
    def _validate(metadata: EvidenceMetadata) -> None:
        if metadata.recorded_at_utc < metadata.prospective_start_utc:
            raise ValueError("recorded_at precedes prospective_collection_start")

    def record_group_o_context(
        self,
        metadata: EvidenceMetadata,
        context: FrozenGroupOContext,
    ) -> int:
        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, context_hash FROM group_o_session_context_v0
                WHERE run_id = ? AND symbol = ? AND signal_session = ?
                """,
                (metadata.run_id, context.symbol, context.signal_session.isoformat()),
            ).fetchone()
            if existing is not None:
                if str(existing["context_hash"]) != context.context_hash:
                    raise ValueError("immutable Group O session context differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO group_o_session_context_v0(
                    envelope_id, run_id, symbol, signal_session,
                    required_option_observation_session,
                    actual_option_observation_session, front_expiry, dte,
                    atm_strike, features_json, missing_indicators_json,
                    quality_status, source_receipt_hashes_json, context_hash,
                    claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    context.symbol,
                    context.signal_session.isoformat(),
                    context.required_option_observation_session.isoformat(),
                    (
                        None
                        if context.actual_option_observation_session is None
                        else context.actual_option_observation_session.isoformat()
                    ),
                    None if context.front_expiry is None else context.front_expiry.isoformat(),
                    context.dte,
                    context.atm_strike,
                    _json(context.features),
                    _json(context.missing_indicators),
                    context.quality_status,
                    _json(context.source_receipt_hashes),
                    context.context_hash,
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def update_underlying_live_projection(
        self,
        metadata: EvidenceMetadata,
        event: UnderlyingLevel1QuoteEvent,
        *,
        tick_by_tick_status: str,
        depth_status: str,
    ) -> None:
        """Upsert one bounded UI projection while raw events stay append-only."""

        self._validate(metadata)
        midpoint = None if event.bid is None or event.ask is None else (event.bid + event.ask) / 2.0
        spread = None if event.bid is None or event.ask is None else event.ask - event.bid
        imbalance = (
            None
            if event.bid_size is None or event.ask_size is None
            else (event.bid_size - event.ask_size) / (event.bid_size + event.ask_size + 1e-12)
        )
        microprice_edge_bps = None
        if (
            midpoint is not None
            and midpoint > 0.0
            and event.bid is not None
            and event.ask is not None
            and event.bid_size is not None
            and event.ask_size is not None
        ):
            microprice = (event.ask * event.bid_size + event.bid * event.ask_size) / (
                event.bid_size + event.ask_size + 1e-12
            )
            microprice_edge_bps = (microprice - midpoint) / midpoint * 10_000.0
        with self.repository._connect() as connection:
            connection.execute(
                """
                INSERT INTO underlying_live_state_v0(
                    run_id, symbol, con_id, request_id,
                    provider_timestamp_utc, received_timestamp_utc,
                    received_monotonic_ns, source_sequence, bid, bid_size,
                    ask, ask_size, last, last_size, midpoint, spread,
                    quote_size_imbalance, microprice_edge_bps,
                    market_data_type, quote_valid, tick_by_tick_status,
                    depth_status, claims_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                    con_id = excluded.con_id,
                    request_id = excluded.request_id,
                    provider_timestamp_utc = excluded.provider_timestamp_utc,
                    received_timestamp_utc = excluded.received_timestamp_utc,
                    received_monotonic_ns = excluded.received_monotonic_ns,
                    source_sequence = excluded.source_sequence,
                    bid = excluded.bid,
                    bid_size = excluded.bid_size,
                    ask = excluded.ask,
                    ask_size = excluded.ask_size,
                    last = excluded.last,
                    last_size = excluded.last_size,
                    midpoint = excluded.midpoint,
                    spread = excluded.spread,
                    quote_size_imbalance = excluded.quote_size_imbalance,
                    microprice_edge_bps = excluded.microprice_edge_bps,
                    market_data_type = excluded.market_data_type,
                    quote_valid = excluded.quote_valid,
                    tick_by_tick_status = excluded.tick_by_tick_status,
                    depth_status = excluded.depth_status,
                    claims_json = excluded.claims_json
                WHERE excluded.source_sequence
                    >= underlying_live_state_v0.source_sequence
                """,
                (
                    metadata.run_id,
                    event.symbol,
                    event.con_id,
                    event.request_id,
                    (
                        None
                        if event.provider_timestamp_utc is None
                        else event.provider_timestamp_utc.isoformat()
                    ),
                    event.received_timestamp_utc.isoformat(),
                    event.received_monotonic_ns,
                    event.source_sequence,
                    event.bid,
                    event.bid_size,
                    event.ask,
                    event.ask_size,
                    event.last,
                    event.last_size,
                    midpoint,
                    spread,
                    imbalance,
                    microprice_edge_bps,
                    event.market_data_type.value,
                    int(event.quote_valid),
                    tick_by_tick_status,
                    depth_status,
                    self.claims_json,
                ),
            )

    def session_episode_state(
        self,
        *,
        run_id: str,
        symbol: str,
        session: date,
    ) -> tuple[float | None, datetime | None, int]:
        """Return exact persisted state needed to resume fresh-episode logic."""

        with self.repository._connect() as connection:
            checkpoint = connection.execute(
                """
                SELECT score.probability
                FROM m1c_checkpoint_v0 AS score
                JOIN m1c_checkpoint_completion_v0 AS completed
                  ON completed.checkpoint_id = score.id
                WHERE score.run_id = ? AND score.symbol = ?
                  AND score.session_date = ? AND score.eligible = 1
                ORDER BY score.checkpoint DESC LIMIT 1
                """,
                (run_id, symbol, session.isoformat()),
            ).fetchone()
            episode = connection.execute(
                """
                SELECT episode.trigger_bar_end_utc, episode.episode_number
                FROM m1c_episode_v0 AS episode
                JOIN m1c_checkpoint_completion_v0 AS completed
                  ON completed.checkpoint_id = episode.checkpoint_id
                WHERE episode.run_id = ? AND episode.symbol = ?
                  AND episode.session_date = ?
                ORDER BY episode.episode_number DESC LIMIT 1
                """,
                (run_id, symbol, session.isoformat()),
            ).fetchone()
        return (
            None if checkpoint is None else float(checkpoint["probability"]),
            (
                None
                if episode is None
                else datetime.fromisoformat(str(episode["trigger_bar_end_utc"]))
            ),
            0 if episode is None else int(episode["episode_number"]),
        )

    def quiet_session_state(
        self,
        *,
        run_id: str,
        symbol: str,
        session: date,
    ) -> tuple[float | None, datetime | None, int]:
        """Return the persisted state needed by the bottom-10 crossing tracker."""

        with self.repository._connect() as connection:
            checkpoint = connection.execute(
                """
                SELECT quiet.m1c_probability
                FROM quiet_state_checkpoint_v0 AS quiet
                JOIN m1c_checkpoint_completion_v0 AS completed
                  ON completed.checkpoint_id = quiet.checkpoint_id
                WHERE quiet.run_id = ? AND quiet.symbol = ?
                  AND quiet.session_date = ? AND quiet.eligible = 1
                ORDER BY quiet.checkpoint DESC LIMIT 1
                """,
                (run_id, symbol, session.isoformat()),
            ).fetchone()
            episode = connection.execute(
                """
                SELECT observation.trigger_timestamp_utc, observation.episode_number
                FROM quiet_state_observation_v0 AS observation
                JOIN quiet_state_checkpoint_v0 AS quiet
                  ON quiet.id = observation.quiet_checkpoint_id
                JOIN m1c_checkpoint_completion_v0 AS completed
                  ON completed.checkpoint_id = quiet.checkpoint_id
                WHERE observation.run_id = ? AND observation.symbol = ?
                  AND observation.session_date = ?
                  AND observation.observation_kind = 'quiet_bottom_10'
                ORDER BY observation.episode_number DESC LIMIT 1
                """,
                (run_id, symbol, session.isoformat()),
            ).fetchone()
        return (
            None if checkpoint is None else float(checkpoint["m1c_probability"]),
            (
                None
                if episode is None
                else datetime.fromisoformat(str(episode["trigger_timestamp_utc"]))
            ),
            0 if episode is None else int(episode["episode_number"]),
        )

    def record_quiet_checkpoint(
        self,
        metadata: EvidenceMetadata,
        *,
        checkpoint_id: int,
        symbol: str,
        session: date,
        checkpoint: int,
        snapshot: QuietStateSnapshot,
        eligible: bool,
    ) -> int:
        """Persist every frozen quiet-tail membership beside the original score."""

        self._validate(metadata)
        with self.repository._connect() as connection:
            source = connection.execute(
                """
                SELECT run_id, symbol, session_date, checkpoint, probability,
                       model_hash, feature_hash
                FROM m1c_checkpoint_v0 WHERE id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            if source is None:
                raise KeyError(checkpoint_id)
            identity = (
                str(source["run_id"]),
                str(source["symbol"]),
                str(source["session_date"]),
                int(source["checkpoint"]),
            )
            if identity != (
                metadata.run_id,
                symbol,
                session.isoformat(),
                int(checkpoint),
            ):
                raise ValueError("quiet checkpoint identity differs from M1C checkpoint")
            if (
                float(source["probability"]) != snapshot.probability
                or str(source["model_hash"]) != snapshot.model_hash
                or str(source["feature_hash"]) != snapshot.feature_hash
            ):
                raise ValueError("quiet checkpoint frozen artifact identity differs")
            existing = connection.execute(
                """
                SELECT id, m1c_probability, previous_m1c_probability,
                       data_quality_flags_json
                FROM quiet_state_checkpoint_v0 WHERE checkpoint_id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            encoded_flags = _json(snapshot.data_quality_flags)
            if existing is not None:
                if (
                    float(existing["m1c_probability"]) != snapshot.probability
                    or (
                        None
                        if existing["previous_m1c_probability"] is None
                        else float(existing["previous_m1c_probability"])
                    )
                    != snapshot.previous_probability
                    or str(existing["data_quality_flags_json"]) != encoded_flags
                ):
                    raise ValueError("immutable quiet checkpoint differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO quiet_state_checkpoint_v0(
                    envelope_id, checkpoint_id, run_id, symbol, session_date,
                    checkpoint, m1c_probability, previous_m1c_probability,
                    bottom_5, bottom_10, bottom_20, high_tail,
                    distance_from_bottom_10, model_hash, feature_hash, eligible,
                    data_quality_status, data_quality_flags_json, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    checkpoint_id,
                    metadata.run_id,
                    symbol,
                    session.isoformat(),
                    int(checkpoint),
                    snapshot.probability,
                    snapshot.previous_probability,
                    int(snapshot.bottom_5),
                    int(snapshot.bottom_10),
                    int(snapshot.bottom_20),
                    int(snapshot.high_tail),
                    snapshot.distance_from_bottom_10,
                    snapshot.model_hash,
                    snapshot.feature_hash,
                    int(eligible),
                    snapshot.data_quality_status,
                    encoded_flags,
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def _quiet_checkpoint_row(
        self,
        connection: sqlite3.Connection,
        quiet_checkpoint_id: int,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM quiet_state_checkpoint_v0 WHERE id = ?",
            (quiet_checkpoint_id,),
        ).fetchone()
        if row is None:
            raise KeyError(quiet_checkpoint_id)
        return cast(sqlite3.Row, row)

    def record_quiet_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        quiet_checkpoint_id: int,
        decision: QuietEpisodeDecision,
        scientific_recording_valid: bool,
    ) -> str:
        self._validate(metadata)
        if (
            not decision.fresh_episode
            or decision.quiet_episode_id is None
            or decision.episode_number is None
        ):
            raise ValueError("only fresh bottom-10 quiet episodes may be persisted")
        with self.repository._connect() as connection:
            source = self._quiet_checkpoint_row(connection, quiet_checkpoint_id)
            if (
                str(source["run_id"]) != metadata.run_id
                or str(source["symbol"]) != decision.symbol
                or str(source["session_date"]) != decision.session.isoformat()
                or int(source["checkpoint"]) != decision.checkpoint
            ):
                raise ValueError("quiet episode does not match its checkpoint")
            existing = connection.execute(
                """
                SELECT *
                FROM quiet_state_observation_v0 WHERE observation_id = ?
                """,
                (decision.quiet_episode_id,),
            ).fetchone()
            if existing is not None:
                _assert_immutable_observation(
                    cast(sqlite3.Row, existing),
                    expected={
                        "quiet_checkpoint_id": quiet_checkpoint_id,
                        "run_id": metadata.run_id,
                        "observation_kind": "quiet_bottom_10",
                        "symbol": decision.symbol,
                        "session_date": decision.session.isoformat(),
                        "trigger_checkpoint": decision.checkpoint,
                        "trigger_timestamp_utc": decision.trigger_timestamp.isoformat(),
                        "prospective_entry_timestamp_utc": (
                            decision.prospective_entry_timestamp.isoformat()
                        ),
                        "m1c_probability": decision.probability,
                        "previous_m1c_probability": decision.previous_probability,
                        "bottom_5": int(decision.bottom_5),
                        "bottom_10": int(decision.bottom_10),
                        "bottom_20": int(decision.bottom_20),
                        "high_tail": int(decision.high_tail),
                        "episode_number": decision.episode_number,
                        "minutes_since_previous_quiet_episode": (
                            decision.minutes_since_previous_episode
                        ),
                        "previous_high_tail_within_60_minutes": int(
                            decision.previous_high_tail_within_60_minutes
                        ),
                        "neutral_hash_hex": None,
                        "neutral_hash_fraction": None,
                        "neutral_sampling_fraction": None,
                        "neutral_salt_id": None,
                        "scientific_recording_valid": int(scientific_recording_valid),
                        "data_quality_flags_json": _json(decision.data_quality_flags),
                        "claims_json": self.claims_json,
                    },
                    label="quiet episode",
                )
                return str(existing["observation_id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            connection.execute(
                """
                INSERT INTO quiet_state_observation_v0(
                    observation_id, envelope_id, quiet_checkpoint_id, run_id,
                    observation_kind, symbol, session_date, trigger_checkpoint,
                    trigger_timestamp_utc, prospective_entry_timestamp_utc,
                    m1c_probability, previous_m1c_probability, bottom_5,
                    bottom_10, bottom_20, high_tail, episode_number,
                    minutes_since_previous_quiet_episode,
                    previous_high_tail_within_60_minutes,
                    following_high_tail_within_60_minutes, neutral_hash_hex,
                    neutral_hash_fraction, neutral_sampling_fraction,
                    neutral_salt_id, option_context_valid,
                    scientific_recording_valid, data_quality_flags_json, phase,
                    completion_status, completed_at_utc, claims_json
                ) VALUES (
                    ?, ?, ?, ?, 'quiet_bottom_10', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0, ?, ?, 'pending_completion',
                    'active', NULL, ?
                )
                """,
                (
                    decision.quiet_episode_id,
                    envelope_id,
                    quiet_checkpoint_id,
                    metadata.run_id,
                    decision.symbol,
                    decision.session.isoformat(),
                    decision.checkpoint,
                    decision.trigger_timestamp.isoformat(),
                    decision.prospective_entry_timestamp.isoformat(),
                    decision.probability,
                    decision.previous_probability,
                    int(decision.bottom_5),
                    int(decision.bottom_10),
                    int(decision.bottom_20),
                    int(decision.high_tail),
                    decision.episode_number,
                    decision.minutes_since_previous_episode,
                    int(decision.previous_high_tail_within_60_minutes),
                    int(decision.following_high_tail_within_60_minutes),
                    int(scientific_recording_valid),
                    _json(decision.data_quality_flags),
                    self.claims_json,
                ),
            )
        return decision.quiet_episode_id

    def record_neutral_control(
        self,
        metadata: EvidenceMetadata,
        *,
        quiet_checkpoint_id: int,
        decision: NeutralControlDecision,
        trigger_timestamp: datetime,
        data_quality_flags: tuple[str, ...],
    ) -> str:
        """Persist only a deterministic ten-percent neutral selection."""

        self._validate(metadata)
        if not decision.selected or not decision.population_eligible:
            raise ValueError("only selected neutral controls may be persisted")
        if trigger_timestamp.tzinfo is None or trigger_timestamp.utcoffset() is None:
            raise ValueError("neutral-control timestamp must be timezone-aware")
        observation_id = f"m1c-neutral-{decision.hash_hex[:24]}"
        with self.repository._connect() as connection:
            source = self._quiet_checkpoint_row(connection, quiet_checkpoint_id)
            if (
                str(source["run_id"]) != metadata.run_id
                or str(source["symbol"]) != decision.symbol
                or str(source["session_date"]) != decision.session.isoformat()
                or int(source["checkpoint"]) != decision.checkpoint
            ):
                raise ValueError("neutral control does not match its checkpoint")
            existing = connection.execute(
                """
                SELECT *
                FROM quiet_state_observation_v0 WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
            if existing is not None:
                observed = trigger_timestamp.astimezone(UTC).isoformat()
                _assert_immutable_observation(
                    cast(sqlite3.Row, existing),
                    expected={
                        "quiet_checkpoint_id": quiet_checkpoint_id,
                        "run_id": metadata.run_id,
                        "observation_kind": "neutral_control",
                        "symbol": decision.symbol,
                        "session_date": decision.session.isoformat(),
                        "trigger_checkpoint": decision.checkpoint,
                        "trigger_timestamp_utc": observed,
                        "prospective_entry_timestamp_utc": observed,
                        "m1c_probability": decision.probability,
                        "previous_m1c_probability": (
                            None
                            if source["previous_m1c_probability"] is None
                            else float(source["previous_m1c_probability"])
                        ),
                        "bottom_5": 0,
                        "bottom_10": 0,
                        "bottom_20": 0,
                        "high_tail": 0,
                        "episode_number": None,
                        "minutes_since_previous_quiet_episode": None,
                        "previous_high_tail_within_60_minutes": 0,
                        "neutral_hash_hex": decision.hash_hex,
                        "neutral_hash_fraction": decision.hash_fraction,
                        "neutral_sampling_fraction": decision.sampling_fraction,
                        "neutral_salt_id": decision.salt_id,
                        "scientific_recording_valid": 1,
                        "data_quality_flags_json": _json(tuple(sorted(set(data_quality_flags)))),
                        "claims_json": self.claims_json,
                    },
                    label="neutral control",
                )
                return str(existing["observation_id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            observed = trigger_timestamp.astimezone(UTC).isoformat()
            connection.execute(
                """
                INSERT INTO quiet_state_observation_v0(
                    observation_id, envelope_id, quiet_checkpoint_id, run_id,
                    observation_kind, symbol, session_date, trigger_checkpoint,
                    trigger_timestamp_utc, prospective_entry_timestamp_utc,
                    m1c_probability, previous_m1c_probability, bottom_5,
                    bottom_10, bottom_20, high_tail, episode_number,
                    minutes_since_previous_quiet_episode,
                    previous_high_tail_within_60_minutes,
                    following_high_tail_within_60_minutes, neutral_hash_hex,
                    neutral_hash_fraction, neutral_sampling_fraction,
                    neutral_salt_id, option_context_valid,
                    scientific_recording_valid, data_quality_flags_json, phase,
                    completion_status, completed_at_utc, claims_json
                ) VALUES (
                    ?, ?, ?, ?, 'neutral_control', ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0,
                    NULL, NULL, 0, 0, ?, ?, ?, ?, 0, 1, ?, 'pending_completion',
                    'active', NULL, ?
                )
                """,
                (
                    observation_id,
                    envelope_id,
                    quiet_checkpoint_id,
                    metadata.run_id,
                    decision.symbol,
                    decision.session.isoformat(),
                    decision.checkpoint,
                    observed,
                    observed,
                    decision.probability,
                    (
                        None
                        if source["previous_m1c_probability"] is None
                        else float(source["previous_m1c_probability"])
                    ),
                    decision.hash_hex,
                    decision.hash_fraction,
                    decision.sampling_fraction,
                    decision.salt_id,
                    _json(tuple(sorted(set(data_quality_flags)))),
                    self.claims_json,
                ),
            )
        return observation_id

    def record_high_tail_control(
        self,
        metadata: EvidenceMetadata,
        *,
        quiet_checkpoint_id: int,
        decision: EpisodeDecision,
        scientific_recording_valid: bool,
        data_quality_flags: tuple[str, ...],
    ) -> str:
        """Mirror each already-authorised fresh high episode into the comparison cohort."""

        self._validate(metadata)
        if (
            not decision.fresh_episode
            or decision.episode_id is None
            or decision.episode_number is None
        ):
            raise ValueError("only fresh frozen high-tail episodes may be controls")
        with self.repository._connect() as connection:
            source = self._quiet_checkpoint_row(connection, quiet_checkpoint_id)
            if (
                str(source["run_id"]) != metadata.run_id
                or str(source["symbol"]) != decision.symbol
                or str(source["session_date"]) != decision.session.isoformat()
                or int(source["checkpoint"]) != decision.checkpoint
            ):
                raise ValueError("high-tail control does not match its checkpoint")
            existing = connection.execute(
                """
                SELECT *
                FROM quiet_state_observation_v0 WHERE observation_id = ?
                """,
                (decision.episode_id,),
            ).fetchone()
            if existing is not None:
                _assert_immutable_observation(
                    cast(sqlite3.Row, existing),
                    expected={
                        "quiet_checkpoint_id": quiet_checkpoint_id,
                        "run_id": metadata.run_id,
                        "observation_kind": "high_tail_control",
                        "symbol": decision.symbol,
                        "session_date": decision.session.isoformat(),
                        "trigger_checkpoint": decision.checkpoint,
                        "trigger_timestamp_utc": decision.trigger_bar_end.isoformat(),
                        "prospective_entry_timestamp_utc": (
                            decision.prospective_entry_timestamp.isoformat()
                        ),
                        "m1c_probability": decision.probability,
                        "previous_m1c_probability": decision.previous_probability,
                        "bottom_5": 0,
                        "bottom_10": 0,
                        "bottom_20": 0,
                        "high_tail": 1,
                        "episode_number": decision.episode_number,
                        "minutes_since_previous_quiet_episode": None,
                        "previous_high_tail_within_60_minutes": 0,
                        "neutral_hash_hex": None,
                        "neutral_hash_fraction": None,
                        "neutral_sampling_fraction": None,
                        "neutral_salt_id": None,
                        "scientific_recording_valid": int(scientific_recording_valid),
                        "data_quality_flags_json": _json(tuple(sorted(set(data_quality_flags)))),
                        "claims_json": self.claims_json,
                    },
                    label="high-tail control",
                )
                return str(existing["observation_id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            connection.execute(
                """
                INSERT INTO quiet_state_observation_v0(
                    observation_id, envelope_id, quiet_checkpoint_id, run_id,
                    observation_kind, symbol, session_date, trigger_checkpoint,
                    trigger_timestamp_utc, prospective_entry_timestamp_utc,
                    m1c_probability, previous_m1c_probability, bottom_5,
                    bottom_10, bottom_20, high_tail, episode_number,
                    minutes_since_previous_quiet_episode,
                    previous_high_tail_within_60_minutes,
                    following_high_tail_within_60_minutes, neutral_hash_hex,
                    neutral_hash_fraction, neutral_sampling_fraction,
                    neutral_salt_id, option_context_valid,
                    scientific_recording_valid, data_quality_flags_json, phase,
                    completion_status, completed_at_utc, claims_json
                ) VALUES (
                    ?, ?, ?, ?, 'high_tail_control', ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 1,
                    ?, NULL, 0, 0, NULL, NULL, NULL, NULL, 0, ?, ?,
                    'pending_completion', 'active', NULL, ?
                )
                """,
                (
                    decision.episode_id,
                    envelope_id,
                    quiet_checkpoint_id,
                    metadata.run_id,
                    decision.symbol,
                    decision.session.isoformat(),
                    decision.checkpoint,
                    decision.trigger_bar_end.isoformat(),
                    decision.prospective_entry_timestamp.isoformat(),
                    decision.probability,
                    decision.previous_probability,
                    decision.episode_number,
                    int(scientific_recording_valid),
                    _json(tuple(sorted(set(data_quality_flags)))),
                    self.claims_json,
                ),
            )
        return decision.episode_id

    def mark_following_high_tail_proximity(
        self,
        *,
        run_id: str,
        symbol: str,
        session: date,
        high_tail_timestamp: datetime,
    ) -> int:
        """Complete the descriptive following-60-minute flag after a high episode."""

        if high_tail_timestamp.tzinfo is None or high_tail_timestamp.utcoffset() is None:
            raise ValueError("high-tail timestamp must be timezone-aware")
        high = high_tail_timestamp.astimezone(UTC)
        lower = high - timedelta(minutes=60)
        with self.repository._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE quiet_state_observation_v0
                SET following_high_tail_within_60_minutes = 1
                WHERE run_id = ? AND symbol = ? AND session_date = ?
                  AND observation_kind = 'quiet_bottom_10'
                  AND trigger_timestamp_utc < ?
                  AND trigger_timestamp_utc >= ?
                  AND following_high_tail_within_60_minutes = 0
                """,
                (
                    run_id,
                    symbol,
                    session.isoformat(),
                    high.isoformat(),
                    lower.isoformat(),
                ),
            )
            return int(cursor.rowcount)

    def record_checkpoint(
        self,
        metadata: EvidenceMetadata,
        *,
        symbol: str,
        session: date,
        checkpoint: int,
        bar_start_utc: datetime,
        bar_end_utc: datetime,
        score: FrozenM1CScore,
        session_context_hash: str,
        feature_values: Mapping[str, object],
        eligible: bool,
        feature_freshness: str,
        rejection_reasons: tuple[str, ...],
        model_version: str = "frozen-m1c-v0",
    ) -> int:
        self._validate(metadata)
        if score.threshold_passed != (score.probability >= score.threshold):
            raise ValueError("M1C threshold membership differs")
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, feature_hash, probability FROM m1c_checkpoint_v0
                WHERE run_id = ? AND symbol = ? AND session_date = ? AND checkpoint = ?
                """,
                (metadata.run_id, symbol, session.isoformat(), checkpoint),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["feature_hash"]) != score.feature_hash
                    or float(existing["probability"]) != score.probability
                ):
                    raise ValueError("immutable M1C checkpoint differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO m1c_checkpoint_v0(
                    envelope_id, run_id, symbol, session_date, checkpoint,
                    bar_start_utc, bar_end_utc, feature_as_of_utc,
                    model_id, model_version, model_hash, feature_hash,
                    session_context_hash, feature_values_json, probability,
                    threshold, threshold_passed, eligible, feature_freshness,
                    missing_feature_count, rejection_reasons_json, claims_json,
                    bar_identity, configuration_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'M1C', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    symbol,
                    session.isoformat(),
                    checkpoint,
                    bar_start_utc.astimezone(UTC).isoformat(),
                    bar_end_utc.astimezone(UTC).isoformat(),
                    bar_end_utc.astimezone(UTC).isoformat(),
                    model_version,
                    score.model_hash,
                    score.feature_hash,
                    session_context_hash,
                    _json(dict(feature_values)),
                    score.probability,
                    score.threshold,
                    int(score.threshold_passed),
                    int(eligible),
                    feature_freshness,
                    score.missing_feature_count,
                    _json(rejection_reasons),
                    self.claims_json,
                    (
                        f"IBKR|{symbol}|{session.isoformat()}|"
                        f"{bar_start_utc.astimezone(UTC).isoformat()}|"
                        f"{bar_end_utc.astimezone(UTC).isoformat()}"
                    ),
                    self.configuration_hash,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_episode(
        self,
        metadata: EvidenceMetadata,
        *,
        checkpoint_id: int,
        decision: EpisodeDecision,
        safety: EpisodeSafetyDecision,
    ) -> str:
        self._validate(metadata)
        if not decision.fresh_episode or decision.episode_id is None:
            raise ValueError("only fresh M1C episodes may be persisted as episodes")
        assert decision.episode_number is not None
        with self.repository._connect() as connection:
            existing = connection.execute(
                "SELECT episode_id FROM m1c_episode_v0 WHERE episode_id = ?",
                (decision.episode_id,),
            ).fetchone()
            if existing is not None:
                return str(existing["episode_id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            connection.execute(
                """
                INSERT INTO m1c_episode_v0(
                    episode_id, envelope_id, checkpoint_id, run_id, symbol,
                    session_date, trigger_checkpoint, trigger_bar_end_utc,
                    prospective_entry_timestamp_utc, m1c_probability,
                    previous_m1c_probability, episode_number,
                    minutes_since_previous_episode, scientific_recording_valid,
                    rejection_reasons_json, phase, completion_status,
                    completed_at_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'pending_completion', 'active', NULL, ?)
                """,
                (
                    decision.episode_id,
                    envelope_id,
                    checkpoint_id,
                    metadata.run_id,
                    decision.symbol,
                    decision.session.isoformat(),
                    decision.checkpoint,
                    decision.trigger_bar_end.isoformat(),
                    decision.prospective_entry_timestamp.isoformat(),
                    decision.probability,
                    decision.previous_probability,
                    decision.episode_number,
                    decision.minutes_since_previous_episode,
                    int(safety.scientific_recording_valid),
                    _json(safety.rejection_reasons),
                    self.claims_json,
                ),
            )
        return decision.episode_id

    def record_directions(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        features: DirectionFeatureResult,
        classifications: Mapping[str, DirectionClassification],
        valid: bool,
    ) -> tuple[int, ...]:
        self._validate(metadata)
        if not features.trigger_bar_excluded:
            raise ValueError("trigger bar must be excluded from direction features")
        inserted: list[int] = []
        with self.repository._connect() as connection:
            for archetype in ("A1", "C1", "R1"):
                classification = classifications[archetype]
                existing = connection.execute(
                    """
                    SELECT id, probability_up, action
                    FROM direction_classification_v0
                    WHERE episode_id = ? AND archetype = ?
                    """,
                    (episode_id, archetype),
                ).fetchone()
                if existing is not None:
                    if (
                        float(existing["probability_up"]) != classification.probability_up
                        or str(existing["action"]) != classification.action
                    ):
                        raise ValueError("immutable direction classification differs")
                    inserted.append(int(existing["id"]))
                    continue
                envelope_id = self.repository._insert_envelope(connection, metadata)
                cursor = connection.execute(
                    """
                    INSERT INTO direction_classification_v0(
                        envelope_id, run_id, episode_id, archetype,
                        probability_up, confidence, action, confidence_boundary,
                        classification_label, model_hash, preprocessing_hash,
                        feature_hash, maximum_feature_timestamp_utc,
                        trigger_bar_excluded, valid, payload_json, claims_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        envelope_id,
                        metadata.run_id,
                        episode_id,
                        archetype,
                        classification.probability_up,
                        classification.confidence,
                        classification.action,
                        classification.boundary,
                        classification.label,
                        classification.model_hash,
                        classification.preprocessing_hash,
                        features.feature_hash,
                        features.maximum_direction_feature_timestamp.isoformat(),
                        int(valid),
                        _json(classification),
                        self.claims_json,
                    ),
                )
                assert cursor.lastrowid is not None
                inserted.append(int(cursor.lastrowid))
        return tuple(inserted)

    def record_microstructure_summary(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str | None,
        window_name: str,
        summary: MicrostructureWindowSummary,
        level1_valid: bool,
        tick_valid: bool,
        depth_valid: bool,
        quality_flags: tuple[str, ...],
        archetype_relationships: Mapping[str, str] | None = None,
    ) -> int:
        self._validate(metadata)
        components = {name: score.components for name, score in summary.scores.items()}
        encoded_summary = _json(summary)
        encoded_components = _json(components)
        encoded_relationships = _json(
            {} if archetype_relationships is None else archetype_relationships
        )
        encoded_quality = _json(quality_flags)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM microstructure_summary_v0
                WHERE run_id = ? AND symbol = ? AND window_name = ?
                  AND window_end_utc = ?
                """,
                (
                    metadata.run_id,
                    summary.symbol,
                    window_name,
                    summary.window_end.isoformat(),
                ),
            ).fetchone()
            if existing is not None:
                expected = (
                    episode_id,
                    summary.window_start.isoformat(),
                    int(level1_valid),
                    int(tick_valid),
                    int(depth_valid),
                    encoded_summary,
                    encoded_components,
                    encoded_relationships,
                    encoded_quality,
                )
                actual = (
                    existing["episode_id"],
                    str(existing["window_start_utc"]),
                    int(existing["level1_valid"]),
                    int(existing["tick_valid"]),
                    int(existing["depth_valid"]),
                    str(existing["summary_json"]),
                    str(existing["component_json"]),
                    str(existing["archetype_relationship_json"]),
                    str(existing["quality_flags_json"]),
                )
                if actual != expected:
                    raise ValueError("immutable microstructure summary differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO microstructure_summary_v0(
                    envelope_id, run_id, episode_id, symbol, window_name,
                    window_start_utc, window_end_utc, calculated_at_utc,
                    level1_valid, tick_valid, depth_valid, summary_json,
                    component_json, archetype_relationship_json,
                    quality_flags_json, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    episode_id,
                    summary.symbol,
                    window_name,
                    summary.window_start.isoformat(),
                    summary.window_end.isoformat(),
                    metadata.recorded_at_utc.isoformat(),
                    int(level1_valid),
                    int(tick_valid),
                    int(depth_valid),
                    encoded_summary,
                    encoded_components,
                    encoded_relationships,
                    encoded_quality,
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_quiet_microstructure_summary(
        self,
        metadata: EvidenceMetadata,
        *,
        observation_id: str,
        window_name: str,
        summary: MicrostructureWindowSummary,
        level1_valid: bool,
        tick_valid: bool,
        depth_valid: bool,
        quality_flags: tuple[str, ...],
    ) -> int:
        """Persist a quiet/control window without a directional-model relationship."""

        self._validate(metadata)
        payload = {
            **summary.model_dump(mode="json"),
            "level1_valid": level1_valid,
            "tick_valid": tick_valid,
            "depth_valid": depth_valid,
        }
        encoded = _json(payload)
        encoded_quality = _json(quality_flags)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, summary_json, quality_flags_json
                FROM quiet_state_microstructure_v0
                WHERE observation_id = ? AND window_name = ? AND window_end_utc = ?
                """,
                (
                    observation_id,
                    window_name,
                    summary.window_end.isoformat(),
                ),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["summary_json"]) != encoded
                    or str(existing["quality_flags_json"]) != encoded_quality
                ):
                    raise ValueError("immutable quiet microstructure summary differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO quiet_state_microstructure_v0(
                    envelope_id, run_id, observation_id, window_name,
                    window_start_utc, window_end_utc, summary_json,
                    quality_flags_json, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    observation_id,
                    window_name,
                    summary.window_start.isoformat(),
                    summary.window_end.isoformat(),
                    encoded,
                    encoded_quality,
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_quiet_underlying_path(
        self,
        metadata: EvidenceMetadata,
        *,
        observation_id: str,
        horizon_label: str,
        target_timestamp_utc: datetime,
        payload: dict[str, object],
        quality_flags: tuple[str, ...],
    ) -> int:
        """Persist one deterministic Level-I/trade path projection for a horizon."""

        self._validate(metadata)
        if target_timestamp_utc.tzinfo is None or target_timestamp_utc.utcoffset() is None:
            raise ValueError("quiet underlying-path timestamp must be timezone-aware")
        encoded_payload = _json(payload)
        encoded_quality = _json(tuple(sorted(set(quality_flags))))
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, target_timestamp_utc, payload_json, quality_flags_json
                FROM quiet_state_underlying_path_v0
                WHERE observation_id = ? AND horizon_label = ?
                """,
                (observation_id, horizon_label),
            ).fetchone()
            observed_target = target_timestamp_utc.astimezone(UTC).isoformat()
            if existing is not None:
                if (
                    str(existing["target_timestamp_utc"]) != observed_target
                    or str(existing["payload_json"]) != encoded_payload
                    or str(existing["quality_flags_json"]) != encoded_quality
                ):
                    raise ValueError("immutable quiet underlying path differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO quiet_state_underlying_path_v0(
                    envelope_id, run_id, observation_id, horizon_label,
                    target_timestamp_utc, payload_json, quality_flags_json,
                    claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    observation_id,
                    horizon_label,
                    observed_target,
                    encoded_payload,
                    encoded_quality,
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_partition(
        self,
        metadata: EvidenceMetadata,
        *,
        data_source: str,
        session_date: date,
        symbol: str,
        event_type: str,
        partition: PartitionWriteResult,
    ) -> int:
        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM raw_partition_manifest_v0
                WHERE run_id = ? AND content_hash = ?
                """,
                (metadata.run_id, partition.content_hash),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            cursor = connection.execute(
                """
                INSERT INTO raw_partition_manifest_v0(
                    run_id, data_source, session_date, symbol, event_type,
                    file_path, row_count, minimum_timestamp_utc,
                    maximum_timestamp_utc, schema_version, content_hash,
                    complete, gap_count, recorder_version, contract_version,
                    recorded_at_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.run_id,
                    data_source,
                    session_date.isoformat(),
                    symbol,
                    event_type,
                    str(partition.data_path),
                    partition.row_count,
                    partition.minimum_timestamp_utc.isoformat(),
                    partition.maximum_timestamp_utc.isoformat(),
                    partition.schema_version,
                    partition.content_hash,
                    int(partition.complete),
                    partition.gap_count,
                    partition.recorder_version,
                    partition.contract_version,
                    metadata.recorded_at_utc.isoformat(),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_shadow_outcome(
        self,
        metadata: EvidenceMetadata,
        *,
        archetype: str,
        direction: str,
        outcome: ShadowOptionOutcome,
        valid: bool,
    ) -> int:
        self._validate(metadata)
        identity = (
            f"{outcome.expiry.isoformat()}|{outcome.strike:.12g}|{outcome.right}|{outcome.con_id}"
        )
        encoded_outcome = _json(outcome)
        encoded_quality = _json(outcome.quote_quality_flags)
        with self.repository._connect() as connection:
            cohort_phase, scientific_option_evidence = self._parent_phase(
                connection,
                run_id=metadata.run_id,
                parent_table="m1c_episode_v0",
                parent_id_column="episode_id",
                parent_id=outcome.episode_id,
            )
            existing = connection.execute(
                """
                SELECT * FROM shadow_quote_outcome_v0
                WHERE episode_id = ? AND archetype = ? AND dte_bucket = ?
                  AND contract_identity = ? AND horizon_minutes = ?
                """,
                (
                    outcome.episode_id,
                    archetype,
                    outcome.dte_bucket.value,
                    identity,
                    outcome.horizon_minutes,
                ),
            ).fetchone()
            if existing is not None:
                expected = (
                    direction,
                    outcome.con_id,
                    outcome.horizon_timestamp.isoformat(),
                    outcome.entry_ask,
                    outcome.entry_bid,
                    outcome.exit_bid,
                    outcome.exit_ask,
                    outcome.ask_to_bid_return,
                    outcome.dollar_pnl_per_contract,
                    encoded_outcome,
                    encoded_quality,
                    int(valid),
                )
                actual = (
                    str(existing["direction"]),
                    existing["con_id"],
                    str(existing["target_timestamp_utc"]),
                    existing["entry_ask"],
                    existing["entry_bid"],
                    existing["exit_bid"],
                    existing["exit_ask"],
                    existing["ask_to_bid_return"],
                    existing["dollar_pnl_per_contract"],
                    str(existing["payload_json"]),
                    str(existing["quality_flags_json"]),
                    int(existing["valid"]),
                )
                if actual != expected:
                    raise ValueError("immutable shadow quote outcome differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO shadow_quote_outcome_v0(
                    envelope_id, run_id, episode_id, archetype, direction,
                    dte_bucket, con_id, contract_identity, horizon_minutes,
                    target_timestamp_utc, entry_ask, entry_bid, exit_bid,
                    exit_ask, ask_to_bid_return, dollar_pnl_per_contract,
                    payload_json, quality_flags_json, valid, cohort_phase,
                    scientific_option_evidence, claims_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    outcome.episode_id,
                    archetype,
                    direction,
                    outcome.dte_bucket.value,
                    outcome.con_id,
                    identity,
                    outcome.horizon_minutes,
                    outcome.horizon_timestamp.isoformat(),
                    outcome.entry_ask,
                    outcome.entry_bid,
                    outcome.exit_bid,
                    outcome.exit_ask,
                    outcome.ask_to_bid_return,
                    outcome.dollar_pnl_per_contract,
                    encoded_outcome,
                    encoded_quality,
                    int(valid),
                    cohort_phase,
                    int(scientific_option_evidence),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_shadow_structure(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        structure_type: str,
        dte_bucket: str,
        horizon_minutes: int,
        payload: Mapping[str, object],
        valid: bool,
    ) -> int:
        """Persist straddles and oracle diagnostics outside the live decision surface."""

        self._validate(metadata)
        if structure_type not in {"ATM_STRADDLE", "RETROSPECTIVE_ORACLE"}:
            raise ValueError("shadow structure type is invalid")
        horizon_label = payload.get("horizon_label")
        if horizon_minutes not in {5, 10, 15, 30, 60} and horizon_label != "session_end":
            raise ValueError("shadow structure horizon is not frozen")
        with self.repository._connect() as connection:
            cohort_phase, scientific_option_evidence = self._parent_phase(
                connection,
                run_id=metadata.run_id,
                parent_table="m1c_episode_v0",
                parent_id_column="episode_id",
                parent_id=episode_id,
            )
            existing = connection.execute(
                """
                SELECT id, payload_json FROM shadow_structure_outcome_v0
                WHERE episode_id = ? AND structure_type = ?
                  AND dte_bucket = ? AND horizon_minutes = ?
                """,
                (episode_id, structure_type, dte_bucket, horizon_minutes),
            ).fetchone()
            encoded = _json(payload)
            if existing is not None:
                if str(existing["payload_json"]) != encoded:
                    raise ValueError("immutable shadow structure differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO shadow_structure_outcome_v0(
                    envelope_id, run_id, episode_id, structure_type,
                    dte_bucket, horizon_minutes, payload_json, valid,
                    live_decision_panel_visible, cohort_phase,
                    scientific_option_evidence, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    episode_id,
                    structure_type,
                    dte_bucket,
                    horizon_minutes,
                    encoded,
                    int(valid),
                    cohort_phase,
                    int(scientific_option_evidence),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_option_contract(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        contract: OptionContract,
        selection_rank: int,
        resolution_status: str,
        rejection_reason: str | None,
        recording_started_at_utc: datetime | None,
        recording_ends_at_utc: datetime | None,
    ) -> int:
        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, con_id FROM episode_option_contract_v0
                WHERE episode_id = ? AND expiry = ? AND strike = ? AND right = ?
                """,
                (
                    episode_id,
                    contract.expiry.isoformat(),
                    contract.strike,
                    contract.right,
                ),
            ).fetchone()
            if existing is not None:
                if existing["con_id"] != contract.con_id:
                    raise ValueError("immutable option contract resolution differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO episode_option_contract_v0(
                    envelope_id, run_id, episode_id, underlying_con_id, con_id,
                    expiry, dte, dte_bucket, strike, right, multiplier,
                    exchange, trading_class, selection_rank, resolution_status,
                    rejection_reason, recording_started_at_utc,
                    recording_ends_at_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    episode_id,
                    contract.underlying_con_id,
                    contract.con_id,
                    contract.expiry.isoformat(),
                    contract.dte,
                    contract.dte_bucket.value,
                    contract.strike,
                    contract.right,
                    contract.multiplier,
                    contract.exchange,
                    contract.trading_class,
                    selection_rank,
                    resolution_status,
                    rejection_reason,
                    (
                        None
                        if recording_started_at_utc is None
                        else recording_started_at_utc.astimezone(UTC).isoformat()
                    ),
                    (
                        None
                        if recording_ends_at_utc is None
                        else recording_ends_at_utc.astimezone(UTC).isoformat()
                    ),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def update_option_quote_projection(
        self,
        *,
        option_contract_id: int,
        event: OptionQuoteEvent,
        recording_status: str,
        quote_quality_flags: tuple[str, ...],
    ) -> None:
        """Update only the web projection; immutable raw updates stay in Parquet."""

        with self.repository._connect() as connection:
            contract = connection.execute(
                """
                SELECT run_id, episode_id FROM episode_option_contract_v0
                WHERE id = ?
                """,
                (option_contract_id,),
            ).fetchone()
            if contract is None:
                raise KeyError(option_contract_id)
            observed = event.received_timestamp_utc.isoformat()
            existing = connection.execute(
                """
                SELECT received_timestamp_utc FROM option_quote_state_v0
                WHERE option_contract_id = ?
                """,
                (option_contract_id,),
            ).fetchone()
            if existing is not None and str(existing["received_timestamp_utc"]) > observed:
                raise ValueError("option quote projection cannot move backwards")
            connection.execute(
                """
                INSERT INTO option_quote_state_v0(
                    option_contract_id, run_id, episode_id,
                    provider_timestamp_utc, received_timestamp_utc, bid,
                    bid_size, ask, ask_size, last, last_size, market_data_type,
                    option_model_price, implied_volatility, delta, gamma, theta,
                    vega, underlying_reference_price, volume, open_interest,
                    quote_attributes_json, recording_status,
                    quote_quality_flags_json, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(option_contract_id) DO UPDATE SET
                    provider_timestamp_utc = excluded.provider_timestamp_utc,
                    received_timestamp_utc = excluded.received_timestamp_utc,
                    bid = excluded.bid,
                    bid_size = excluded.bid_size,
                    ask = excluded.ask,
                    ask_size = excluded.ask_size,
                    last = excluded.last,
                    last_size = excluded.last_size,
                    market_data_type = excluded.market_data_type,
                    option_model_price = excluded.option_model_price,
                    implied_volatility = excluded.implied_volatility,
                    delta = excluded.delta,
                    gamma = excluded.gamma,
                    theta = excluded.theta,
                    vega = excluded.vega,
                    underlying_reference_price = excluded.underlying_reference_price,
                    volume = excluded.volume,
                    open_interest = excluded.open_interest,
                    quote_attributes_json = excluded.quote_attributes_json,
                    recording_status = excluded.recording_status,
                    quote_quality_flags_json = excluded.quote_quality_flags_json,
                    claims_json = excluded.claims_json
                """,
                (
                    option_contract_id,
                    str(contract["run_id"]),
                    str(contract["episode_id"]),
                    (
                        None
                        if event.provider_timestamp_utc is None
                        else event.provider_timestamp_utc.isoformat()
                    ),
                    observed,
                    event.bid,
                    event.bid_size,
                    event.ask,
                    event.ask_size,
                    event.last,
                    event.last_size,
                    event.market_data_type.value,
                    event.option_model_price,
                    event.implied_volatility,
                    event.delta,
                    event.gamma,
                    event.theta,
                    event.vega,
                    event.underlying_reference_price,
                    event.volume,
                    event.open_interest,
                    _json(event.quote_attributes),
                    recording_status,
                    _json(quote_quality_flags),
                    self.claims_json,
                ),
            )

    def update_completed_bar_projection(
        self,
        metadata: EvidenceMetadata,
        event: FiveMinuteBarEvent,
    ) -> None:
        """Advance the bounded web bar projection; raw bars remain append-only."""

        self._validate(metadata)
        if not event.finalised:
            raise ValueError("partial bar cannot enter the completed projection")
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT bar_end_utc FROM completed_bar_state_v0
                WHERE run_id = ? AND symbol = ?
                """,
                (metadata.run_id, event.symbol),
            ).fetchone()
            if (
                existing is not None
                and str(existing["bar_end_utc"]) > event.bar_end_utc.isoformat()
            ):
                # IBKR replays historical keepUpToDate bars after reconnect. Raw
                # evidence remains append-only, but an older replay must not
                # regress (or terminate) the bounded latest-bar projection.
                return
            connection.execute(
                """
                INSERT INTO completed_bar_state_v0(
                    run_id, symbol, session_date, bar_start_utc, bar_end_utc,
                    checkpoint, source, source_completeness,
                    received_timestamp_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, symbol) DO UPDATE SET
                    session_date = excluded.session_date,
                    bar_start_utc = excluded.bar_start_utc,
                    bar_end_utc = excluded.bar_end_utc,
                    checkpoint = excluded.checkpoint,
                    source = excluded.source,
                    source_completeness = excluded.source_completeness,
                    received_timestamp_utc = excluded.received_timestamp_utc,
                    claims_json = excluded.claims_json
                """,
                (
                    metadata.run_id,
                    event.symbol,
                    event.session.isoformat(),
                    event.bar_start_utc.isoformat(),
                    event.bar_end_utc.isoformat(),
                    event.checkpoint,
                    event.source,
                    event.source_completeness,
                    event.received_timestamp_utc.isoformat(),
                    self.claims_json,
                ),
            )

    def record_subscription(
        self,
        metadata: EvidenceMetadata,
        record: SubscriptionRecord,
    ) -> int:
        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM subscription_lifecycle_v0
                WHERE run_id = ? AND subscription_key = ? AND started_at_utc = ?
                """,
                (
                    metadata.run_id,
                    record.key,
                    record.started_at_utc.isoformat(),
                ),
            ).fetchone()
            if existing is not None:
                if record.cancelled_at_utc is not None:
                    connection.execute(
                        """
                        UPDATE subscription_lifecycle_v0
                        SET request_id = ?, cancelled_at_utc = ?,
                            cancellation_reason = ?, ibkr_error_codes_json = ?,
                            capacity_denied = ?
                        WHERE id = ? AND cancelled_at_utc IS NULL
                        """,
                        (
                            record.request_id,
                            record.cancelled_at_utc.isoformat(),
                            record.cancellation_reason,
                            _json(record.ibkr_error_codes),
                            int(record.capacity_denied),
                            int(existing["id"]),
                        ),
                    )
                self._insert_subscription_lifecycle_event(
                    connection,
                    metadata=metadata,
                    record=record,
                )
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO subscription_lifecycle_v0(
                    envelope_id, run_id, subscription_key, request_id,
                    subscription_kind, symbol, con_id, priority, owner_episode,
                    started_at_utc, cancelled_at_utc, cancellation_reason,
                    ibkr_error_codes_json, capacity_denied, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    record.key,
                    record.request_id,
                    record.kind.value,
                    record.symbol,
                    record.con_id,
                    int(record.priority),
                    record.owner_episode,
                    record.started_at_utc.isoformat(),
                    (
                        None
                        if record.cancelled_at_utc is None
                        else record.cancelled_at_utc.isoformat()
                    ),
                    record.cancellation_reason,
                    _json(record.ibkr_error_codes),
                    int(record.capacity_denied),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            self._insert_subscription_lifecycle_event(
                connection,
                metadata=metadata,
                record=record,
            )
            return int(cursor.lastrowid)

    def _insert_subscription_lifecycle_event(
        self,
        connection: sqlite3.Connection,
        *,
        metadata: EvidenceMetadata,
        record: SubscriptionRecord,
    ) -> None:
        occurred = (
            record.cancelled_at_utc or record.last_callback_at_utc or metadata.recorded_at_utc
        )
        envelope_id = self.repository._insert_envelope(connection, metadata)
        connection.execute(
            """
            INSERT INTO subscription_lifecycle_event_v0(
                envelope_id, run_id, occurred_at_utc, subscription_key,
                request_id, subscription_kind, subscription_class, symbol,
                con_id, status, owner_ids_json, owner_count, generation,
                reason, payload_json, claims_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                envelope_id,
                metadata.run_id,
                occurred.astimezone(UTC).isoformat(),
                record.key,
                record.request_id,
                record.kind.value,
                int(record.subscription_class),
                record.symbol,
                record.con_id,
                record.status.value,
                _json(tuple(sorted(record.owners))),
                record.owner_count,
                record.generation,
                record.cancellation_reason,
                _json(
                    {
                        "priority": int(record.priority),
                        "protected": record.protected,
                        "capacity_denied": record.capacity_denied,
                        "ibkr_error_codes": record.ibkr_error_codes,
                    }
                ),
                self.claims_json,
            ),
        )

    def record_promotion_decision(
        self,
        metadata: EvidenceMetadata,
        decision: PromotionDecision,
    ) -> int:
        """Persist the scheduler input and rank even when allocation later fails."""

        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM promotion_decision_v0
                WHERE run_id = ? AND promotion_time_utc = ?
                  AND symbol = ? AND subscription_type = ?
                """,
                (
                    metadata.run_id,
                    decision.promotion_time.astimezone(UTC).isoformat(),
                    decision.symbol,
                    decision.subscription_type,
                ),
            ).fetchone()
            expected = (
                decision.m1c_probability,
                decision.rank,
                decision.capacity_available,
                decision.reason,
            )
            if existing is not None:
                actual = (
                    float(existing["m1c_probability"]),
                    int(existing["rank"]),
                    int(existing["capacity_available"]),
                    str(existing["reason"]),
                )
                if actual != expected:
                    raise ValueError("immutable promotion decision differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO promotion_decision_v0(
                    envelope_id, run_id, promotion_time_utc, symbol,
                    m1c_probability, rank, capacity_available,
                    subscription_type, reason, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    decision.promotion_time.astimezone(UTC).isoformat(),
                    decision.symbol,
                    decision.m1c_probability,
                    decision.rank,
                    decision.capacity_available,
                    decision.subscription_type,
                    decision.reason,
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_quiet_option_plan(
        self,
        metadata: EvidenceMetadata,
        *,
        observation_id: str,
        plan: OptionContractPlan,
    ) -> None:
        """Freeze capacity and bucket completeness before recording selected contracts."""

        self._validate(metadata)
        missing_buckets_json = _json(plan.missing_buckets)
        expected = (
            plan.requested_contract_count,
            len(plan.contracts),
            int(plan.capacity_reduced),
            missing_buckets_json,
        )
        with self.repository._connect() as connection:
            observation = connection.execute(
                """
                SELECT run_id, option_plan_recorded,
                       option_plan_requested_contract_count,
                       option_plan_selected_contract_count,
                       option_plan_capacity_reduced,
                       option_plan_missing_buckets_json
                FROM quiet_state_observation_v0
                WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
            if observation is None or str(observation["run_id"]) != metadata.run_id:
                raise KeyError(observation_id)
            if bool(observation["option_plan_recorded"]):
                actual = (
                    int(observation["option_plan_requested_contract_count"]),
                    int(observation["option_plan_selected_contract_count"]),
                    int(observation["option_plan_capacity_reduced"]),
                    str(observation["option_plan_missing_buckets_json"]),
                )
                if actual != expected:
                    raise ValueError("immutable quiet option plan differs")
                return
            connection.execute(
                """
                UPDATE quiet_state_observation_v0
                SET option_plan_recorded = 1,
                    option_plan_requested_contract_count = ?,
                    option_plan_selected_contract_count = ?,
                    option_plan_capacity_reduced = ?,
                    option_plan_missing_buckets_json = ?,
                    option_context_valid = 0
                WHERE observation_id = ?
                """,
                (*expected, observation_id),
            )

    def record_quiet_option_contract(
        self,
        metadata: EvidenceMetadata,
        *,
        observation_id: str,
        contract: OptionContract,
        selection_rank: int,
        selection_roles: tuple[str, ...],
        resolution_status: str,
        rejection_reason: str | None,
        recording_started_at_utc: datetime | None,
        recording_ends_at_utc: datetime | None,
    ) -> int:
        """Persist one exact bounded contract for quiet/control observations."""

        self._validate(metadata)
        with self.repository._connect() as connection:
            observation = connection.execute(
                """
                SELECT run_id FROM quiet_state_observation_v0
                WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
            if observation is None or str(observation["run_id"]) != metadata.run_id:
                raise KeyError(observation_id)
            existing = connection.execute(
                """
                SELECT id, underlying_con_id, con_id, dte, dte_bucket,
                       multiplier, exchange, trading_class, selection_rank,
                       selection_roles_json, resolution_status, rejection_reason,
                       recording_started_at_utc, recording_ends_at_utc
                FROM quiet_state_option_contract_v0
                WHERE observation_id = ? AND expiry = ? AND strike = ? AND right = ?
                """,
                (
                    observation_id,
                    contract.expiry.isoformat(),
                    contract.strike,
                    contract.right,
                ),
            ).fetchone()
            encoded_roles = _json(tuple(sorted(set(selection_roles))))
            started = (
                None
                if recording_started_at_utc is None
                else recording_started_at_utc.astimezone(UTC).isoformat()
            )
            ends = (
                None
                if recording_ends_at_utc is None
                else recording_ends_at_utc.astimezone(UTC).isoformat()
            )
            if existing is not None:
                expected = (
                    contract.underlying_con_id,
                    contract.con_id,
                    contract.dte,
                    contract.dte_bucket.value,
                    contract.multiplier,
                    contract.exchange,
                    contract.trading_class,
                    selection_rank,
                    encoded_roles,
                    resolution_status,
                    rejection_reason,
                    started,
                    ends,
                )
                actual = tuple(
                    existing[name]
                    for name in (
                        "underlying_con_id",
                        "con_id",
                        "dte",
                        "dte_bucket",
                        "multiplier",
                        "exchange",
                        "trading_class",
                        "selection_rank",
                        "selection_roles_json",
                        "resolution_status",
                        "rejection_reason",
                        "recording_started_at_utc",
                        "recording_ends_at_utc",
                    )
                )
                if actual != expected:
                    raise ValueError("immutable quiet option contract differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO quiet_state_option_contract_v0(
                    envelope_id, run_id, observation_id, underlying_con_id,
                    con_id, expiry, dte, dte_bucket, strike, right, multiplier,
                    exchange, trading_class, selection_rank,
                    selection_roles_json, resolution_status, rejection_reason,
                    recording_started_at_utc, recording_ends_at_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    observation_id,
                    contract.underlying_con_id,
                    contract.con_id,
                    contract.expiry.isoformat(),
                    contract.dte,
                    contract.dte_bucket.value,
                    contract.strike,
                    contract.right,
                    contract.multiplier,
                    contract.exchange,
                    contract.trading_class,
                    selection_rank,
                    encoded_roles,
                    resolution_status,
                    rejection_reason,
                    started,
                    ends,
                    self.claims_json,
                ),
            )
            connection.execute(
                """
                UPDATE quiet_state_observation_v0
                SET option_context_valid = CASE
                    WHEN option_plan_recorded = 1
                    AND option_plan_capacity_reduced = 0
                    AND option_plan_selected_contract_count > 0
                    AND option_plan_selected_contract_count =
                        option_plan_requested_contract_count
                    AND option_plan_selected_contract_count = (
                        SELECT COUNT(*)
                        FROM quiet_state_option_contract_v0 AS selected
                        WHERE selected.observation_id = ?
                    )
                    AND EXISTS (
                        SELECT 1 FROM quiet_state_option_contract_v0 AS candidate
                        WHERE candidate.observation_id = ?
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM quiet_state_option_contract_v0 AS failed
                        WHERE failed.observation_id = ?
                          AND failed.resolution_status <> 'recording'
                    )
                    THEN 1 ELSE 0 END
                WHERE observation_id = ?
                """,
                (
                    observation_id,
                    observation_id,
                    observation_id,
                    observation_id,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def update_quiet_option_quote_projection(
        self,
        *,
        option_contract_id: int,
        event: OptionQuoteEvent,
        recording_status: str,
        quote_quality_flags: tuple[str, ...],
    ) -> None:
        """Update only the quiet-option UI projection; raw quotes remain append-only."""

        with self.repository._connect() as connection:
            contract = connection.execute(
                """
                SELECT run_id, observation_id
                FROM quiet_state_option_contract_v0 WHERE id = ?
                """,
                (option_contract_id,),
            ).fetchone()
            if contract is None:
                raise KeyError(option_contract_id)
            observed = event.received_timestamp_utc.isoformat()
            existing = connection.execute(
                """
                SELECT received_timestamp_utc
                FROM quiet_state_option_quote_state_v0
                WHERE option_contract_id = ?
                """,
                (option_contract_id,),
            ).fetchone()
            if existing is not None and str(existing["received_timestamp_utc"]) > observed:
                raise ValueError("quiet option quote projection cannot move backwards")
            connection.execute(
                """
                INSERT INTO quiet_state_option_quote_state_v0(
                    option_contract_id, run_id, observation_id,
                    provider_timestamp_utc, received_timestamp_utc, bid,
                    bid_size, ask, ask_size, last, last_size, market_data_type,
                    option_model_price, implied_volatility, delta, gamma, theta,
                    vega, underlying_reference_price, volume, open_interest,
                    quote_attributes_json, recording_status,
                    quote_quality_flags_json, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(option_contract_id) DO UPDATE SET
                    provider_timestamp_utc = excluded.provider_timestamp_utc,
                    received_timestamp_utc = excluded.received_timestamp_utc,
                    bid = excluded.bid,
                    bid_size = excluded.bid_size,
                    ask = excluded.ask,
                    ask_size = excluded.ask_size,
                    last = excluded.last,
                    last_size = excluded.last_size,
                    market_data_type = excluded.market_data_type,
                    option_model_price = excluded.option_model_price,
                    implied_volatility = excluded.implied_volatility,
                    delta = excluded.delta,
                    gamma = excluded.gamma,
                    theta = excluded.theta,
                    vega = excluded.vega,
                    underlying_reference_price = excluded.underlying_reference_price,
                    volume = excluded.volume,
                    open_interest = excluded.open_interest,
                    quote_attributes_json = excluded.quote_attributes_json,
                    recording_status = excluded.recording_status,
                    quote_quality_flags_json = excluded.quote_quality_flags_json,
                    claims_json = excluded.claims_json
                """,
                (
                    option_contract_id,
                    str(contract["run_id"]),
                    str(contract["observation_id"]),
                    (
                        None
                        if event.provider_timestamp_utc is None
                        else event.provider_timestamp_utc.isoformat()
                    ),
                    observed,
                    event.bid,
                    event.bid_size,
                    event.ask,
                    event.ask_size,
                    event.last,
                    event.last_size,
                    event.market_data_type.value,
                    event.option_model_price,
                    event.implied_volatility,
                    event.delta,
                    event.gamma,
                    event.theta,
                    event.vega,
                    event.underlying_reference_price,
                    event.volume,
                    event.open_interest,
                    _json(event.quote_attributes),
                    recording_status,
                    _json(quote_quality_flags),
                    self.claims_json,
                ),
            )

    def record_quiet_shadow_structure(
        self,
        metadata: EvidenceMetadata,
        *,
        observation_id: str,
        structure_type: str,
        dte_bucket: str,
        horizon_label: str,
        horizon_minutes: int | None,
        payload: Mapping[str, object],
        opening_credit_or_debit: float | None,
        maximum_defined_risk: float | None,
        conservative_pnl: float | None,
        return_on_maximum_risk: float | None,
        short_strike_touched: bool | None,
        protective_wing_touched: bool | None,
        attempted: bool,
        complete_quote_quality: bool,
        strict_quote_quality: bool,
        quality_status: str,
        quality_flags: tuple[str, ...],
    ) -> int:
        """Persist long and defined-risk shadow structures outside any decision panel."""

        self._validate(metadata)
        allowed = {
            "LONG_CALL",
            "LONG_PUT",
            "ATM_STRADDLE",
            "ATM_IRON_BUTTERFLY",
            "DELTA_IRON_CONDOR",
            "CALL_CREDIT_SPREAD",
            "PUT_CREDIT_SPREAD",
        }
        if structure_type not in allowed:
            raise ValueError("quiet shadow structure type is invalid")
        if horizon_label not in {"5m", "10m", "15m", "30m", "60m", "session_end"}:
            raise ValueError("quiet shadow horizon is not frozen")
        encoded = _json(dict(payload))
        with self.repository._connect() as connection:
            cohort_phase, scientific_option_evidence = self._parent_phase(
                connection,
                run_id=metadata.run_id,
                parent_table="quiet_state_observation_v0",
                parent_id_column="observation_id",
                parent_id=observation_id,
            )
            existing = connection.execute(
                """
                SELECT id, payload_json FROM quiet_state_shadow_outcome_v0
                WHERE observation_id = ? AND structure_type = ?
                  AND dte_bucket = ? AND horizon_label = ?
                """,
                (observation_id, structure_type, dte_bucket, horizon_label),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_json"]) != encoded:
                    raise ValueError("immutable quiet shadow outcome differs")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO quiet_state_shadow_outcome_v0(
                    envelope_id, run_id, observation_id, structure_type,
                    dte_bucket, horizon_label, horizon_minutes,
                    opening_credit_or_debit, maximum_defined_risk,
                    conservative_pnl, return_on_maximum_risk,
                    short_strike_touched, protective_wing_touched, attempted,
                    complete_quote_quality, strict_quote_quality, quality_status,
                    quality_flags_json, payload_json,
                    live_decision_panel_visible, cohort_phase,
                    scientific_option_evidence, claims_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    0, ?, ?, ?
                )
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    observation_id,
                    structure_type,
                    dte_bucket,
                    horizon_label,
                    horizon_minutes,
                    opening_credit_or_debit,
                    maximum_defined_risk,
                    conservative_pnl,
                    return_on_maximum_risk,
                    (None if short_strike_touched is None else int(short_strike_touched)),
                    (None if protective_wing_touched is None else int(protective_wing_touched)),
                    int(attempted),
                    int(complete_quote_quality),
                    int(strict_quote_quality),
                    quality_status,
                    _json(quality_flags),
                    encoded,
                    cohort_phase,
                    int(scientific_option_evidence),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def finalise_quiet_observation(
        self,
        *,
        observation_id: str,
        phase: str,
        completion_status: str,
        completed_at_utc: datetime,
    ) -> None:
        if completed_at_utc.tzinfo is None or completed_at_utc.utcoffset() is None:
            raise ValueError("quiet completion timestamp must be timezone-aware")
        completed = completed_at_utc.astimezone(UTC).isoformat()
        with self.repository._connect() as connection:
            row = connection.execute(
                """
                SELECT phase, completion_status, completed_at_utc
                FROM quiet_state_observation_v0 WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(observation_id)
            if row["completed_at_utc"] is not None:
                if (
                    str(row["phase"]) != phase
                    or str(row["completion_status"]) != completion_status
                    or str(row["completed_at_utc"]) != completed
                ):
                    raise ValueError("quiet observation finalisation is immutable")
                return
            connection.execute(
                """
                UPDATE quiet_state_observation_v0
                SET phase = ?, completion_status = ?, completed_at_utc = ?
                WHERE observation_id = ? AND completed_at_utc IS NULL
                """,
                (phase, completion_status, completed, observation_id),
            )

    def finalise_episode(
        self,
        *,
        episode_id: str,
        phase: str,
        completion_status: str,
        completed_at_utc: datetime,
    ) -> None:
        if completed_at_utc.tzinfo is None or completed_at_utc.utcoffset() is None:
            raise ValueError("episode completion timestamp must be timezone-aware")
        completed = completed_at_utc.astimezone(UTC).isoformat()
        with self.repository._connect() as connection:
            row = connection.execute(
                """
                SELECT phase, completion_status, completed_at_utc
                FROM m1c_episode_v0 WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
            if row is None:
                raise KeyError(episode_id)
            if row["completed_at_utc"] is not None:
                if (
                    str(row["phase"]) != phase
                    or str(row["completion_status"]) != completion_status
                    or str(row["completed_at_utc"]) != completed
                ):
                    raise ValueError("episode finalisation is immutable")
                return
            connection.execute(
                """
                UPDATE m1c_episode_v0
                SET phase = ?, completion_status = ?, completed_at_utc = ?
                WHERE episode_id = ? AND completed_at_utc IS NULL
                """,
                (
                    phase,
                    completion_status,
                    completed,
                    episode_id,
                ),
            )

    def record_session_quality_report(
        self,
        metadata: EvidenceMetadata,
        report: SessionQualityReport,
    ) -> int:
        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, report_json FROM recorder_session_report_v0
                WHERE run_id = ? AND session_date = ?
                """,
                (metadata.run_id, report.session_date.isoformat()),
            ).fetchone()
            report_json = _json(report)
            if existing is not None:
                if str(existing["report_json"]) != report_json:
                    raise ValueError("session quality report is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO recorder_session_report_v0(
                    envelope_id, run_id, session_date, report_json,
                    partition_hashes_json, complete, generated_at_utc,
                    claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    report.session_date.isoformat(),
                    report_json,
                    _json(report.raw_event_partition_hashes),
                    int(report.complete),
                    report.generated_at_utc.isoformat(),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_runtime_capacity(
        self,
        metadata: EvidenceMetadata,
        *,
        manifest: Mapping[str, object],
    ) -> int:
        """Persist the startup capacity fact with the same evidence envelope."""

        self._validate(metadata)
        observed = str(manifest["observed_at_utc"])
        encoded = _json(dict(manifest))
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, manifest_json FROM ibkr_runtime_capacity_v0
                WHERE run_id = ? AND observed_at_utc = ?
                """,
                (metadata.run_id, observed),
            ).fetchone()
            if existing is not None:
                if str(existing["manifest_json"]) != encoded:
                    raise ValueError("runtime capacity manifest is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO ibkr_runtime_capacity_v0(
                    envelope_id, run_id, observed_at_utc, manifest_json, claims_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    observed,
                    encoded,
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_option_episode_allocation(
        self,
        metadata: EvidenceMetadata,
        record: EpisodeAllocationRecord,
    ) -> int:
        """Append one explicit queue, degradation, streaming, or release transition."""

        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM option_episode_allocation_v0
                WHERE run_id = ? AND episode_id = ? AND state = ? AND updated_at_utc = ?
                """,
                (
                    metadata.run_id,
                    record.episode_id,
                    record.state.value,
                    record.updated_at_utc.astimezone(UTC).isoformat(),
                ),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO option_episode_allocation_v0(
                    envelope_id, run_id, episode_id, symbol, episode_kind, state,
                    requested_subscriptions_json, approved_subscriptions_json,
                    queued_subscriptions_json, denied_subscriptions_json,
                    degradation_reason, capacity_before_json, capacity_after_json,
                    cohort_phase, scientific_option_evidence, updated_at_utc,
                    claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    record.episode_id,
                    record.symbol,
                    record.kind.value,
                    record.state.value,
                    _json(record.requested_subscriptions),
                    _json(record.approved_subscriptions),
                    _json(record.queued_subscriptions),
                    _json(record.denied_subscriptions),
                    record.degradation_reason,
                    _json(record.capacity_before),
                    _json(record.capacity_after),
                    record.cohort_phase,
                    int(record.scientific_option_evidence),
                    record.updated_at_utc.astimezone(UTC).isoformat(),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_skipped_recording(
        self,
        metadata: EvidenceMetadata,
        *,
        session: date,
        recording_kind: str,
        reason: str,
        requested_payload: Mapping[str, object],
        episode_id: str | None = None,
        symbol: str | None = None,
        cohort_phase: str | None = None,
        scientific_option_evidence: bool | None = None,
    ) -> int:
        self._validate(metadata)
        requested_payload_json = _json(dict(requested_payload))
        with self.repository._connect() as connection:
            if episode_id is not None:
                existing = connection.execute(
                    """
                    SELECT id FROM skipped_recording_v0
                    WHERE run_id = ? AND episode_id = ? AND recording_kind = ?
                      AND reason = ? AND requested_payload_json = ?
                    """,
                    (
                        metadata.run_id,
                        episode_id,
                        recording_kind,
                        reason,
                        requested_payload_json,
                    ),
                ).fetchone()
                if existing is not None:
                    return int(existing["id"])
            resolved_phase, resolved_evidence = self.prospective_phase_for_session(
                run_id=metadata.run_id,
                session=session,
                connection=connection,
            )
            if cohort_phase is not None:
                resolved_phase = cohort_phase
            if scientific_option_evidence is not None:
                resolved_evidence = scientific_option_evidence
            if resolved_phase == "engineering_transfer" and resolved_evidence:
                raise ValueError(
                    "engineering-transfer option records cannot be scientific evidence"
                )
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO skipped_recording_v0(
                    envelope_id, run_id, session_date, episode_id, symbol,
                    recording_kind, reason, requested_payload_json,
                    occurred_at_utc, cohort_phase, scientific_option_evidence,
                    claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    session.isoformat(),
                    episode_id,
                    symbol,
                    recording_kind,
                    reason,
                    requested_payload_json,
                    metadata.recorded_at_utc.isoformat(),
                    resolved_phase,
                    int(resolved_evidence),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_provider_m1c_observation(
        self,
        metadata: EvidenceMetadata,
        *,
        provider: str,
        symbol: str,
        session: date,
        checkpoint: int,
        bar: Mapping[str, object],
        feature_values: Mapping[str, float],
        probability: float,
        quiet_episode: bool,
        high_tail_episode: bool,
        data_quality_status: str,
        model_hash: str,
    ) -> int:
        """Persist one provider-specific score without changing frozen V0 state."""

        self._validate(metadata)
        if provider not in {"ibkr", "eodhd"}:
            raise ValueError("provider M1C observation has an unsupported provider")
        immutable = {
            "bar_identity": str(bar["identity"]),
            "probability": probability,
            "feature_values_json": _json(dict(feature_values)),
        }
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, bar_identity, probability, feature_values_json
                FROM provider_m1c_observation_v0
                WHERE run_id = ? AND provider = ? AND symbol = ?
                  AND session_date = ? AND checkpoint = ?
                """,
                (
                    metadata.run_id,
                    provider,
                    symbol,
                    session.isoformat(),
                    checkpoint,
                ),
            ).fetchone()
            if existing is not None:
                _assert_immutable_observation(
                    existing,
                    expected=immutable,
                    label="provider M1C observation",
                )
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO provider_m1c_observation_v0(
                    envelope_id, run_id, provider, symbol, session_date,
                    checkpoint, bar_identity, bar_start_utc, bar_end_utc,
                    open, high, low, close, feature_values_json, probability,
                    quiet_episode, high_tail_episode, data_quality_status,
                    model_hash, configuration_hash, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    provider,
                    symbol,
                    session.isoformat(),
                    checkpoint,
                    str(bar["identity"]),
                    str(bar["start_utc"]),
                    str(bar["end_utc"]),
                    float(cast(Any, bar["open"])),
                    float(cast(Any, bar["high"])),
                    float(cast(Any, bar["low"])),
                    float(cast(Any, bar["close"])),
                    immutable["feature_values_json"],
                    probability,
                    int(quiet_episode),
                    int(high_tail_episode),
                    data_quality_status,
                    model_hash,
                    self.configuration_hash,
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_source_transfer_session(
        self,
        metadata: EvidenceMetadata,
        *,
        session: date,
        valid: bool,
        decision: str,
        report: Mapping[str, object],
    ) -> int:
        self._validate(metadata)
        encoded = _json(dict(report))
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, report_json FROM source_transfer_session_v0
                WHERE run_id = ? AND session_date = ?
                """,
                (metadata.run_id, session.isoformat()),
            ).fetchone()
            if existing is not None:
                if str(existing["report_json"]) != encoded:
                    raise ValueError("source transfer session report is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO source_transfer_session_v0(
                    envelope_id, run_id, session_date, valid, decision,
                    report_json, generated_at_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    session.isoformat(),
                    int(valid),
                    decision,
                    encoded,
                    metadata.recorded_at_utc.isoformat(),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_prospective_session_phase(
        self,
        metadata: EvidenceMetadata,
        *,
        session: date,
        valid: bool,
        valid_session_ordinal: int | None,
        phase: str,
        source_transfer_decision: str | None,
    ) -> int:
        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, valid, valid_session_ordinal, phase, source_transfer_decision
                FROM prospective_session_phase_v0
                WHERE run_id = ? AND session_date = ?
                """,
                (metadata.run_id, session.isoformat()),
            ).fetchone()
            expected = {
                "valid": int(valid),
                "valid_session_ordinal": valid_session_ordinal,
                "phase": phase,
                "source_transfer_decision": source_transfer_decision,
            }
            if existing is not None:
                _assert_immutable_observation(
                    existing,
                    expected=expected,
                    label="prospective session phase",
                )
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO prospective_session_phase_v0(
                    envelope_id, run_id, session_date, valid_session_ordinal,
                    phase, valid, source_transfer_decision,
                    strategy_rule_changes_allowed, recorded_at_utc, claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    session.isoformat(),
                    valid_session_ordinal,
                    phase,
                    int(valid),
                    source_transfer_decision,
                    metadata.recorded_at_utc.isoformat(),
                    self.claims_json,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)


__all__ = ["FrozenRecorderRepository"]
