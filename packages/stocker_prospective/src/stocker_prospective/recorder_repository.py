"""Append-oriented metadata persistence for the frozen M1C recorder."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel

from stocker_prospective.contract import M1C_FROZEN_THRESHOLD, claims_boundary
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
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
    RESERVED_MARKET_DATA_LINES_V1,
    CapacityDegradationEventV1,
    MarketDataCapacitySnapshotV1,
    OpeningReversalActivationReceiptV1,
    OpeningReversalDecisionReceiptV1,
    OpeningReversalPredictionReceiptV1,
    OpeningReversalUnderlyingOutcomeV1,
    OpeningTransferOperationalEvidenceV1,
    OpeningTransferSessionResultV1,
    PrimaryOptionBidAskOutcomeV1,
    PrimaryOptionPairSelectionV1,
    PromotionSelectionV1,
    build_opening_transfer_decision_receipt_v1,
    validate_primary_option_protocol_v1_1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    OpeningReversalActivationReceiptV1_1,
    OpeningReversalCausalBarrierAuditV1_1,
)
from stocker_prospective.microstructure import MicrostructureWindowSummary
from stocker_prospective.opening_market_transition_v1 import (
    OpeningMarketTransitionStateResultV1,
    OpeningPreEntryWindowV1,
    OpeningTransitionThresholdsV1,
    StockOpeningResponseResultV1,
)
from stocker_prospective.option_budget import EpisodeAllocationRecord
from stocker_prospective.option_ledger import (
    OptionContract,
    OptionContractPlan,
    ShadowOptionOutcome,
)
from stocker_prospective.partition_store import PartitionWriteResult, sha256_path
from stocker_prospective.quality_report import SessionQualityReport
from stocker_prospective.quiet_state import (
    NeutralControlDecision,
    QuietEpisodeDecision,
    QuietStateSnapshot,
)
from stocker_prospective.safety import EpisodeSafetyDecision
from stocker_prospective.signed_market_shock_v1 import (
    CheckpointShockThresholdsV1,
    MarketShockStateResultV1,
    PreentryMarketWindowsV1,
    StockShockResponseResultV1,
)
from stocker_prospective.subscriptions import PromotionDecision, SubscriptionRecord
from stocker_prospective.tail_phase_v1 import (
    M1C_TAIL_PHASE_V1_VERSION,
    MOVEMENT_CONSUMED_LOOKBACK_MINUTES_V1,
    MOVEMENT_CONSUMED_MEDIAN_2024_V1,
    MovementConsumedBucketV1,
    MovementConsumedStateV1,
    TailPhaseStateV1,
    assign_movement_consumed_bucket_v1,
)

TRANSFER_DECISIONS_PERMITTING_OPTION_DEVELOPMENT = frozenset(
    {
        "ibkr_transfer_supported_without_recalibration",
        "ibkr_ranking_supported_probability_scale_shifted",
    }
)
SIGNED_MARKET_SHOCK_COLUMNS_V1: Final[tuple[str, ...]] = (
    "canonical_market_proxy_v1",
    "market_return_w0_v1",
    "market_range_w0_v1",
    "market_return_w1_v1",
    "market_range_w1_v1",
    "market_shock_thresholds_v1_json",
    "market_shock_state_v1",
    "market_shock_event_id_v1",
    "shock_sign_v1",
    "stock_return_w0_v1",
    "stock_absolute_alignment_v1",
    "shock_relative_response_v1",
    "shock_response_class_v1",
    "shock_resisting_subtype_v1",
    "market_shock_complete_v1",
    "shock_response_complete_v1",
    "market_shock_missing_reasons_v1_json",
    "shock_response_missing_reasons_v1_json",
)
OPENING_MARKET_TRANSITION_COLUMNS_V1: Final[tuple[str, ...]] = (
    "opening_market_proxy_v1",
    "vti_session_open_v1",
    "vti_prior_regular_session_close_v1",
    "opening_expected_bar_count_v1",
    "opening_observed_bar_count_v1",
    "market_opening_return_v1",
    "market_opening_range_v1",
    "market_overnight_gap_v1",
    "market_total_transition_v1",
    "market_gap_open_alignment_v1",
    "opening_thresholds_v1_json",
    "opening_market_transition_state_v1",
    "opening_transition_sign_v1",
    "opening_transition_event_id_v1",
    "stock_opening_return_v1",
    "stock_opening_range_v1",
    "stock_opening_alignment_v1",
    "stock_relative_opening_response_v1",
    "stock_opening_response_class_v1",
    "stock_opening_resisting_subtype_v1",
    "opening_market_complete_v1",
    "stock_opening_response_complete_v1",
    "opening_market_missing_reasons_v1_json",
    "stock_opening_response_missing_reasons_v1_json",
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


def _content_hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _assert_immutable_observation(
    existing: sqlite3.Row,
    *,
    expected: Mapping[str, object],
    label: str,
) -> None:
    mismatches = tuple(key for key, value in expected.items() if existing[key] != value)
    if mismatches:
        raise ValueError(f"immutable {label} differs: {','.join(sorted(mismatches))}")


def _signed_market_shock_values_v1(
    *,
    market_windows_v1: PreentryMarketWindowsV1,
    market_shock_state_v1: MarketShockStateResultV1,
    stock_shock_response_v1: StockShockResponseResultV1,
    market_shock_thresholds_v1: CheckpointShockThresholdsV1 | None,
) -> dict[str, object]:
    """Serialise the logging-only fields once for insert and retry validation."""

    values: dict[str, object] = {
        "canonical_market_proxy_v1": market_windows_v1.market_proxy_v1,
        "market_return_w0_v1": market_windows_v1.market_return_w0_v1,
        "market_range_w0_v1": market_windows_v1.market_range_w0_v1,
        "market_return_w1_v1": market_windows_v1.market_return_w1_v1,
        "market_range_w1_v1": market_windows_v1.market_range_w1_v1,
        "market_shock_thresholds_v1_json": (
            None if market_shock_thresholds_v1 is None else _json(market_shock_thresholds_v1)
        ),
        "market_shock_state_v1": market_shock_state_v1.market_shock_state_v1,
        "market_shock_event_id_v1": (market_shock_state_v1.market_shock_event_id_v1),
        "shock_sign_v1": market_shock_state_v1.shock_sign_v1,
        "stock_return_w0_v1": stock_shock_response_v1.stock_return_w0_v1,
        "stock_absolute_alignment_v1": (stock_shock_response_v1.stock_absolute_alignment_v1),
        "shock_relative_response_v1": (stock_shock_response_v1.shock_relative_response_v1),
        "shock_response_class_v1": (stock_shock_response_v1.shock_response_class_v1),
        "shock_resisting_subtype_v1": stock_shock_response_v1.resisting_subtype_v1,
        "market_shock_complete_v1": int(market_shock_state_v1.complete_v1),
        "shock_response_complete_v1": int(stock_shock_response_v1.complete_v1),
        "market_shock_missing_reasons_v1_json": _json(market_shock_state_v1.missing_reasons_v1),
        "shock_response_missing_reasons_v1_json": _json(stock_shock_response_v1.missing_reasons_v1),
    }
    if tuple(values) != SIGNED_MARKET_SHOCK_COLUMNS_V1:
        raise AssertionError("signed market-shock persistence column order drifted")
    return values


def _opening_market_transition_values_v1(
    *,
    opening_window_v1: OpeningPreEntryWindowV1,
    opening_transition_state_v1: OpeningMarketTransitionStateResultV1,
    stock_opening_response_v1: StockOpeningResponseResultV1,
    opening_thresholds_v1: OpeningTransitionThresholdsV1 | None,
) -> dict[str, object]:
    """Serialise opening-transition logging once for insert and retry checks."""

    values: dict[str, object] = {
        "opening_market_proxy_v1": opening_window_v1.market_proxy_v1,
        "vti_session_open_v1": opening_window_v1.market_session_open_v1,
        "vti_prior_regular_session_close_v1": (
            opening_window_v1.market_prior_regular_session_close_v1
        ),
        "opening_expected_bar_count_v1": (opening_window_v1.expected_opening_bar_count_v1),
        "opening_observed_bar_count_v1": (opening_window_v1.observed_opening_bar_count_v1),
        "market_opening_return_v1": opening_window_v1.market_opening_return_v1,
        "market_opening_range_v1": opening_window_v1.market_opening_range_v1,
        "market_overnight_gap_v1": opening_window_v1.market_overnight_gap_v1,
        "market_total_transition_v1": (opening_window_v1.market_total_transition_v1),
        "market_gap_open_alignment_v1": (opening_window_v1.market_gap_open_alignment_v1),
        "opening_thresholds_v1_json": (
            None if opening_thresholds_v1 is None else _json(opening_thresholds_v1)
        ),
        "opening_market_transition_state_v1": (
            opening_transition_state_v1.opening_market_transition_state_v1
        ),
        "opening_transition_sign_v1": (opening_transition_state_v1.opening_transition_sign_v1),
        "opening_transition_event_id_v1": (
            opening_transition_state_v1.opening_transition_event_id_v1
        ),
        "stock_opening_return_v1": (stock_opening_response_v1.stock_opening_return_v1),
        "stock_opening_range_v1": (stock_opening_response_v1.stock_opening_range_v1),
        "stock_opening_alignment_v1": (stock_opening_response_v1.stock_opening_alignment_v1),
        "stock_relative_opening_response_v1": (
            stock_opening_response_v1.stock_relative_opening_response_v1
        ),
        "stock_opening_response_class_v1": (
            stock_opening_response_v1.stock_opening_response_class_v1
        ),
        "stock_opening_resisting_subtype_v1": (stock_opening_response_v1.resisting_subtype_v1),
        "opening_market_complete_v1": int(opening_transition_state_v1.complete_v1),
        "stock_opening_response_complete_v1": int(stock_opening_response_v1.complete_v1),
        "opening_market_missing_reasons_v1_json": _json(
            opening_transition_state_v1.missing_reasons_v1
        ),
        "stock_opening_response_missing_reasons_v1_json": _json(
            stock_opening_response_v1.missing_reasons_v1
        ),
    }
    if tuple(values) != OPENING_MARKET_TRANSITION_COLUMNS_V1:
        raise AssertionError("opening-transition persistence column order drifted")
    return values


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
            SELECT session_date, scientific_recording_valid FROM {parent_table}
            WHERE run_id = ? AND {parent_id_column} = ?
            """,
            (run_id, parent_id),
        ).fetchone()
        if row is None:
            raise KeyError(parent_id)
        phase, phase_allows_scientific_evidence = self.prospective_phase_for_session(
            run_id=run_id,
            session=date.fromisoformat(str(row["session_date"])),
            connection=connection,
        )
        return (
            phase,
            phase_allows_scientific_evidence and bool(row["scientific_recording_valid"]),
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
                    atm_strike, previous_close_implied_movement_15m,
                    features_json, missing_indicators_json,
                    quality_status, source_receipt_hashes_json, context_hash,
                    claims_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    context.previous_close_implied_movement_15m,
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

    def session_tail_phase_history(
        self,
        *,
        run_id: str,
        symbol: str,
        session: date,
    ) -> tuple[tuple[int, datetime, float, bool, str | None], ...]:
        """Return chronological checkpoint inputs needed to restore Tail Phase V1."""

        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT checkpoint, bar_end_utc, probability, eligible,
                       rejection_reasons_json
                FROM m1c_checkpoint_v0
                WHERE run_id = ? AND symbol = ? AND session_date = ?
                ORDER BY checkpoint
                """,
                (run_id, symbol, session.isoformat()),
            ).fetchall()
        output: list[tuple[int, datetime, float, bool, str | None]] = []
        for row in rows:
            reasons = cast(list[object], json.loads(str(row["rejection_reasons_json"])))
            output.append(
                (
                    int(row["checkpoint"]),
                    datetime.fromisoformat(str(row["bar_end_utc"])),
                    float(row["probability"]),
                    bool(row["eligible"]),
                    None if not reasons else ";".join(str(value) for value in reasons),
                )
            )
        return tuple(output)

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
        scientific_recording_valid: bool,
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
                        "scientific_recording_valid": int(scientific_recording_valid),
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
                    NULL, NULL, 0, 0, ?, ?, ?, ?, 0, ?, ?, 'pending_completion',
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
                    int(scientific_recording_valid),
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
        tail_phase_v1: TailPhaseStateV1 | None = None,
        movement_consumed_v1: MovementConsumedStateV1 | None = None,
        movement_consumed_bucket_v1: MovementConsumedBucketV1 | None = None,
        movement_consumed_frozen_median_v1: float | None = None,
        tail_phase_activation_status_v1: str = "not_configured",
    ) -> int:
        self._validate(metadata)
        if score.threshold_passed != (score.probability >= score.threshold):
            raise ValueError("M1C threshold membership differs")
        if score.threshold != M1C_FROZEN_THRESHOLD:
            raise ValueError("M1C threshold differs from the frozen contract")
        if tail_phase_v1 is not None:
            if tail_phase_v1.m1c_high_tail_v1 is not score.threshold_passed:
                raise ValueError("Tail Phase membership differs from frozen M1C")
            if (score.threshold_passed and tail_phase_v1.m1c_tail_phase_v1 == "OUTSIDE_TAIL") or (
                not score.threshold_passed
                and tail_phase_v1.m1c_tail_phase_v1 in {"FIRST_ENTRY", "PERSISTENT", "RE_ENTRY"}
            ):
                raise ValueError("Tail Phase state differs from frozen M1C")
        if (
            movement_consumed_frozen_median_v1 is not None
            and movement_consumed_frozen_median_v1 != MOVEMENT_CONSUMED_MEDIAN_2024_V1
        ):
            raise ValueError("movement-consumed frozen median differs")
        if movement_consumed_v1 is not None and movement_consumed_bucket_v1 is not None:
            expected_bucket = assign_movement_consumed_bucket_v1(
                movement_consumed_v1.movement_consumed_v1,
                frozen_median=movement_consumed_frozen_median_v1,
            )
            if movement_consumed_bucket_v1 != expected_bucket:
                raise ValueError("movement-consumed bucket differs from frozen median")
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, feature_hash, probability, threshold,
                       m1c_high_tail_v1, m1c_tail_phase_v1,
                       tail_entry_number_v1, tail_run_length_checkpoints_v1,
                       tail_run_age_minutes_v1, prior_tail_entries_v1,
                       previous_checkpoint_above_tail_v1,
                       minutes_since_previous_tail_exit_v1,
                       phase_history_complete_v1, phase_missing_reason_v1,
                       movement_consumed_v1, movement_consumed_numerator_v1,
                       movement_consumed_denominator_v1,
                       movement_consumed_complete_v1,
                       movement_consumed_missing_reason_v1,
                       movement_consumed_bucket_v1, tail_phase_source_v1_json
                FROM m1c_checkpoint_v0
                WHERE run_id = ? AND symbol = ? AND session_date = ? AND checkpoint = ?
                """,
                (metadata.run_id, symbol, session.isoformat(), checkpoint),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["feature_hash"]) != score.feature_hash
                    or float(existing["probability"]) != score.probability
                    or float(existing["threshold"]) != score.threshold
                ):
                    raise ValueError("immutable M1C checkpoint differs")
                if tail_phase_v1 is not None:
                    persisted_phase = {
                        "m1c_high_tail_v1": (
                            None
                            if existing["m1c_high_tail_v1"] is None
                            else bool(existing["m1c_high_tail_v1"])
                        ),
                        "m1c_tail_phase_v1": existing["m1c_tail_phase_v1"],
                        "tail_entry_number_v1": existing["tail_entry_number_v1"],
                        "tail_run_length_checkpoints_v1": existing[
                            "tail_run_length_checkpoints_v1"
                        ],
                        "tail_run_age_minutes_v1": existing["tail_run_age_minutes_v1"],
                        "prior_tail_entries_v1": existing["prior_tail_entries_v1"],
                        "previous_checkpoint_above_tail_v1": (
                            None
                            if existing["previous_checkpoint_above_tail_v1"] is None
                            else bool(existing["previous_checkpoint_above_tail_v1"])
                        ),
                        "minutes_since_previous_tail_exit_v1": existing[
                            "minutes_since_previous_tail_exit_v1"
                        ],
                        "phase_history_complete_v1": bool(existing["phase_history_complete_v1"]),
                        "phase_missing_reason_v1": existing["phase_missing_reason_v1"],
                    }
                    if persisted_phase != tail_phase_v1.model_dump(mode="python"):
                        raise ValueError("immutable Tail Phase checkpoint differs")
                    source = json.loads(str(existing["tail_phase_source_v1_json"]))
                    if (
                        source.get("tail_phase_activation_status_v1")
                        != tail_phase_activation_status_v1
                    ):
                        raise ValueError("immutable Tail Phase activation differs")
                if movement_consumed_v1 is not None:
                    persisted_consumed = {
                        "movement_consumed_v1": existing["movement_consumed_v1"],
                        "movement_consumed_numerator_v1": existing[
                            "movement_consumed_numerator_v1"
                        ],
                        "movement_consumed_denominator_v1": existing[
                            "movement_consumed_denominator_v1"
                        ],
                        "movement_consumed_complete_v1": bool(
                            existing["movement_consumed_complete_v1"]
                        ),
                        "movement_consumed_missing_reason_v1": existing[
                            "movement_consumed_missing_reason_v1"
                        ],
                    }
                    if persisted_consumed != movement_consumed_v1.model_dump(mode="python"):
                        raise ValueError("immutable movement-consumed checkpoint differs")
                if (
                    movement_consumed_bucket_v1 is not None
                    and existing["movement_consumed_bucket_v1"] != movement_consumed_bucket_v1
                ):
                    raise ValueError("immutable movement-consumed bucket differs")
                if movement_consumed_frozen_median_v1 is not None:
                    source = json.loads(str(existing["tail_phase_source_v1_json"]))
                    if (
                        source.get("movement_consumed_frozen_median_v1")
                        != movement_consumed_frozen_median_v1
                    ):
                        raise ValueError("immutable Tail Phase source differs")
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
            checkpoint_id = int(cursor.lastrowid)
            if (
                tail_phase_v1 is not None
                or movement_consumed_v1 is not None
                or movement_consumed_bucket_v1 is not None
            ):
                phase = (
                    TailPhaseStateV1(
                        m1c_high_tail_v1=None,
                        m1c_tail_phase_v1="UNKNOWN_INCOMPLETE",
                        tail_entry_number_v1=None,
                        tail_run_length_checkpoints_v1=None,
                        tail_run_age_minutes_v1=None,
                        prior_tail_entries_v1=None,
                        previous_checkpoint_above_tail_v1=None,
                        minutes_since_previous_tail_exit_v1=None,
                        phase_history_complete_v1=False,
                        phase_missing_reason_v1="tail_phase_not_supplied",
                    )
                    if tail_phase_v1 is None
                    else tail_phase_v1
                )
                consumed = (
                    MovementConsumedStateV1(
                        movement_consumed_v1=None,
                        movement_consumed_numerator_v1=None,
                        movement_consumed_denominator_v1=None,
                        movement_consumed_complete_v1=False,
                        movement_consumed_missing_reason_v1=("movement_consumed_not_supplied"),
                    )
                    if movement_consumed_v1 is None
                    else movement_consumed_v1
                )
                bucket: MovementConsumedBucketV1 = (
                    "UNKNOWN_INCOMPLETE"
                    if movement_consumed_bucket_v1 is None
                    else movement_consumed_bucket_v1
                )
                source = {
                    "schema_version": M1C_TAIL_PHASE_V1_VERSION,
                    "m1c_threshold": score.threshold,
                    "movement_consumed_lookback_minutes": (MOVEMENT_CONSUMED_LOOKBACK_MINUTES_V1),
                    "movement_consumed_frozen_median_v1": (movement_consumed_frozen_median_v1),
                    "tail_phase_activation_status_v1": (tail_phase_activation_status_v1),
                    "previous_close_implied_movement_15m_status": (
                        "available"
                        if consumed.movement_consumed_denominator_v1 is not None
                        else consumed.movement_consumed_missing_reason_v1
                    ),
                    "movement_consumed_median_provenance": {
                        "start": "2024-01-01",
                        "end": "2024-12-31",
                        "predictor_values_only": True,
                    },
                    "stock_local": True,
                    "peer_normalisation_used": False,
                    "causal_timestamp": bar_end_utc.astimezone(UTC).isoformat(),
                    "recorder_configuration_hash": self.configuration_hash,
                }
                connection.execute(
                    """
                    UPDATE m1c_checkpoint_v0
                    SET m1c_high_tail_v1 = ?,
                        m1c_tail_phase_v1 = ?,
                        tail_entry_number_v1 = ?,
                        tail_run_length_checkpoints_v1 = ?,
                        tail_run_age_minutes_v1 = ?,
                        prior_tail_entries_v1 = ?,
                        previous_checkpoint_above_tail_v1 = ?,
                        minutes_since_previous_tail_exit_v1 = ?,
                        phase_history_complete_v1 = ?,
                        phase_missing_reason_v1 = ?,
                        movement_consumed_v1 = ?,
                        movement_consumed_numerator_v1 = ?,
                        movement_consumed_denominator_v1 = ?,
                        movement_consumed_complete_v1 = ?,
                        movement_consumed_missing_reason_v1 = ?,
                        movement_consumed_bucket_v1 = ?,
                        tail_phase_source_v1_json = ?
                    WHERE id = ?
                    """,
                    (
                        (None if phase.m1c_high_tail_v1 is None else int(phase.m1c_high_tail_v1)),
                        phase.m1c_tail_phase_v1,
                        phase.tail_entry_number_v1,
                        phase.tail_run_length_checkpoints_v1,
                        phase.tail_run_age_minutes_v1,
                        phase.prior_tail_entries_v1,
                        (
                            None
                            if phase.previous_checkpoint_above_tail_v1 is None
                            else int(phase.previous_checkpoint_above_tail_v1)
                        ),
                        phase.minutes_since_previous_tail_exit_v1,
                        int(phase.phase_history_complete_v1),
                        phase.phase_missing_reason_v1,
                        consumed.movement_consumed_v1,
                        consumed.movement_consumed_numerator_v1,
                        consumed.movement_consumed_denominator_v1,
                        int(consumed.movement_consumed_complete_v1),
                        consumed.movement_consumed_missing_reason_v1,
                        bucket,
                        _json(source),
                        checkpoint_id,
                    ),
                )
            return checkpoint_id

    def record_signed_market_shock_checkpoint_v1(
        self,
        metadata: EvidenceMetadata,
        *,
        checkpoint_id: int,
        symbol: str,
        session: date,
        checkpoint: int,
        market_windows_v1: PreentryMarketWindowsV1,
        market_shock_state_v1: MarketShockStateResultV1,
        stock_shock_response_v1: StockShockResponseResultV1,
        market_shock_thresholds_v1: CheckpointShockThresholdsV1 | None,
        activation_status_v1: str,
    ) -> None:
        """Attach optional signed-shock evidence after core episode promotion."""

        self._validate(metadata)
        expected = _signed_market_shock_values_v1(
            market_windows_v1=market_windows_v1,
            market_shock_state_v1=market_shock_state_v1,
            stock_shock_response_v1=stock_shock_response_v1,
            market_shock_thresholds_v1=market_shock_thresholds_v1,
        )
        shock_source = {
            "schema_version": "m1c-signed-market-shock-transition-v1",
            "activation_status_v1": activation_status_v1,
            "threshold_configuration": (
                None
                if market_shock_thresholds_v1 is None
                else market_shock_thresholds_v1.model_dump(mode="json")
            ),
            "w0_bar_ordinals_v1": market_windows_v1.w0_bar_ordinals_v1,
            "w1_bar_ordinals_v1": market_windows_v1.w1_bar_ordinals_v1,
            "signal_timestamp": market_windows_v1.signal_timestamp,
            "maximum_market_timestamp_v1": (market_windows_v1.maximum_market_timestamp_v1),
            "maximum_stock_timestamp_v1": (stock_shock_response_v1.maximum_stock_timestamp_v1),
            "recorder_configuration_hash": self.configuration_hash,
            "logging_only": True,
            "m1c_scoring_changed": False,
            "episode_promotion_changed": False,
            "recorder_priority_changed": False,
            "subscription_allocation_changed": False,
            "option_contract_selection_changed": False,
            "direction_decision_changed": False,
            "episode_inclusion_changed": False,
            "order_routing_changed": False,
        }
        source_json = _json(shock_source)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT run_id, symbol, session_date, checkpoint,
                       canonical_market_proxy_v1,
                       market_return_w0_v1, market_range_w0_v1,
                       market_return_w1_v1, market_range_w1_v1,
                       market_shock_thresholds_v1_json,
                       market_shock_state_v1, market_shock_event_id_v1,
                       shock_sign_v1, stock_return_w0_v1,
                       stock_absolute_alignment_v1,
                       shock_relative_response_v1,
                       shock_response_class_v1,
                       shock_resisting_subtype_v1,
                       market_shock_complete_v1,
                       shock_response_complete_v1,
                       market_shock_missing_reasons_v1_json,
                       shock_response_missing_reasons_v1_json,
                       signed_market_shock_source_v1_json
                FROM m1c_checkpoint_v0
                WHERE id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            if existing is None:
                raise ValueError("signed-shock checkpoint does not exist")
            if (
                str(existing["run_id"]) != metadata.run_id
                or str(existing["symbol"]) != symbol
                or str(existing["session_date"]) != session.isoformat()
                or int(existing["checkpoint"]) != checkpoint
                or market_windows_v1.session != session
                or market_windows_v1.checkpoint != checkpoint
            ):
                raise ValueError("signed-shock checkpoint identity differs")
            if existing["signed_market_shock_source_v1_json"] is not None:
                _assert_immutable_observation(
                    existing,
                    expected={
                        **expected,
                        "signed_market_shock_source_v1_json": source_json,
                    },
                    label="signed market-shock checkpoint",
                )
                return
            if any(existing[column] is not None for column in SIGNED_MARKET_SHOCK_COLUMNS_V1):
                raise ValueError("signed market-shock checkpoint is partially populated")
            connection.execute(
                """
                UPDATE m1c_checkpoint_v0
                SET canonical_market_proxy_v1 = ?,
                    market_return_w0_v1 = ?,
                    market_range_w0_v1 = ?,
                    market_return_w1_v1 = ?,
                    market_range_w1_v1 = ?,
                    market_shock_thresholds_v1_json = ?,
                    market_shock_state_v1 = ?,
                    market_shock_event_id_v1 = ?,
                    shock_sign_v1 = ?,
                    stock_return_w0_v1 = ?,
                    stock_absolute_alignment_v1 = ?,
                    shock_relative_response_v1 = ?,
                    shock_response_class_v1 = ?,
                    shock_resisting_subtype_v1 = ?,
                    market_shock_complete_v1 = ?,
                    shock_response_complete_v1 = ?,
                    market_shock_missing_reasons_v1_json = ?,
                    shock_response_missing_reasons_v1_json = ?,
                    signed_market_shock_source_v1_json = ?
                WHERE id = ?
                """,
                (
                    *(expected[column] for column in SIGNED_MARKET_SHOCK_COLUMNS_V1),
                    source_json,
                    checkpoint_id,
                ),
            )

    def record_opening_market_transition_checkpoint_v1(
        self,
        metadata: EvidenceMetadata,
        *,
        checkpoint_id: int,
        symbol: str,
        session: date,
        checkpoint: int,
        opening_window_v1: OpeningPreEntryWindowV1,
        opening_transition_state_v1: OpeningMarketTransitionStateResultV1,
        stock_opening_response_v1: StockOpeningResponseResultV1,
        opening_thresholds_v1: OpeningTransitionThresholdsV1 | None,
        activation_status_v1: str,
    ) -> None:
        """Attach optional opening evidence after core episode promotion."""

        self._validate(metadata)
        expected = _opening_market_transition_values_v1(
            opening_window_v1=opening_window_v1,
            opening_transition_state_v1=opening_transition_state_v1,
            stock_opening_response_v1=stock_opening_response_v1,
            opening_thresholds_v1=opening_thresholds_v1,
        )
        source = {
            "schema_version": "m1c-opening-market-transition-v1",
            "activation_status_v1": activation_status_v1,
            "threshold_configuration": (
                None
                if opening_thresholds_v1 is None
                else opening_thresholds_v1.model_dump(mode="json")
            ),
            "previous_session_v1": opening_window_v1.previous_session_v1,
            "session_open_timestamp_v1": (opening_window_v1.session_open_timestamp_v1),
            "signal_timestamp_v1": opening_window_v1.signal_timestamp_v1,
            "entry_timestamp_v1": opening_window_v1.entry_timestamp_v1,
            "opening_bar_ordinals_v1": (opening_window_v1.opening_bar_ordinals_v1),
            "expected_opening_bar_count_v1": (opening_window_v1.expected_opening_bar_count_v1),
            "observed_opening_bar_count_v1": (opening_window_v1.observed_opening_bar_count_v1),
            "entry_bar_included_v1": (opening_window_v1.entry_bar_included_v1),
            "maximum_market_timestamp_v1": (opening_window_v1.maximum_market_timestamp_v1),
            "maximum_stock_timestamp_v1": (stock_opening_response_v1.maximum_stock_timestamp_v1),
            "recorder_configuration_hash": self.configuration_hash,
            "logging_only": True,
            "m1c_scoring_changed": False,
            "episode_promotion_changed": False,
            "recorder_priority_changed": False,
            "subscription_allocation_changed": False,
            "option_contract_selection_changed": False,
            "direction_decision_changed": False,
            "episode_inclusion_changed": False,
            "recorder_capacity_changed": False,
            "order_routing_changed": False,
        }
        source_json = _json(source)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT run_id, symbol, session_date, checkpoint,
                       opening_market_proxy_v1,
                       vti_session_open_v1,
                       vti_prior_regular_session_close_v1,
                       opening_expected_bar_count_v1,
                       opening_observed_bar_count_v1,
                       market_opening_return_v1,
                       market_opening_range_v1,
                       market_overnight_gap_v1,
                       market_total_transition_v1,
                       market_gap_open_alignment_v1,
                       opening_thresholds_v1_json,
                       opening_market_transition_state_v1,
                       opening_transition_sign_v1,
                       opening_transition_event_id_v1,
                       stock_opening_return_v1,
                       stock_opening_range_v1,
                       stock_opening_alignment_v1,
                       stock_relative_opening_response_v1,
                       stock_opening_response_class_v1,
                       stock_opening_resisting_subtype_v1,
                       opening_market_complete_v1,
                       stock_opening_response_complete_v1,
                       opening_market_missing_reasons_v1_json,
                       stock_opening_response_missing_reasons_v1_json,
                       opening_market_transition_source_v1_json
                FROM m1c_checkpoint_v0
                WHERE id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            if existing is None:
                raise ValueError("opening-transition checkpoint does not exist")
            if (
                str(existing["run_id"]) != metadata.run_id
                or str(existing["symbol"]) != symbol
                or str(existing["session_date"]) != session.isoformat()
                or int(existing["checkpoint"]) != checkpoint
                or opening_window_v1.session != session
                or opening_window_v1.checkpoint_v1 != checkpoint
            ):
                raise ValueError("opening-transition checkpoint identity differs")
            if existing["opening_market_transition_source_v1_json"] is not None:
                _assert_immutable_observation(
                    existing,
                    expected={
                        **expected,
                        "opening_market_transition_source_v1_json": source_json,
                    },
                    label="opening market-transition checkpoint",
                )
                return
            if any(existing[column] is not None for column in OPENING_MARKET_TRANSITION_COLUMNS_V1):
                raise ValueError("opening market-transition checkpoint is partially populated")
            connection.execute(
                """
                UPDATE m1c_checkpoint_v0
                SET opening_market_proxy_v1 = ?,
                    vti_session_open_v1 = ?,
                    vti_prior_regular_session_close_v1 = ?,
                    opening_expected_bar_count_v1 = ?,
                    opening_observed_bar_count_v1 = ?,
                    market_opening_return_v1 = ?,
                    market_opening_range_v1 = ?,
                    market_overnight_gap_v1 = ?,
                    market_total_transition_v1 = ?,
                    market_gap_open_alignment_v1 = ?,
                    opening_thresholds_v1_json = ?,
                    opening_market_transition_state_v1 = ?,
                    opening_transition_sign_v1 = ?,
                    opening_transition_event_id_v1 = ?,
                    stock_opening_return_v1 = ?,
                    stock_opening_range_v1 = ?,
                    stock_opening_alignment_v1 = ?,
                    stock_relative_opening_response_v1 = ?,
                    stock_opening_response_class_v1 = ?,
                    stock_opening_resisting_subtype_v1 = ?,
                    opening_market_complete_v1 = ?,
                    stock_opening_response_complete_v1 = ?,
                    opening_market_missing_reasons_v1_json = ?,
                    stock_opening_response_missing_reasons_v1_json = ?,
                    opening_market_transition_source_v1_json = ?
                WHERE id = ?
                """,
                (
                    *(expected[column] for column in OPENING_MARKET_TRANSITION_COLUMNS_V1),
                    source_json,
                    checkpoint_id,
                ),
            )

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
            checkpoint = connection.execute(
                """
                SELECT model_version, threshold, probability,
                       m1c_tail_phase_v1, tail_entry_number_v1,
                       tail_run_length_checkpoints_v1, tail_run_age_minutes_v1,
                       prior_tail_entries_v1, previous_checkpoint_above_tail_v1,
                       minutes_since_previous_tail_exit_v1,
                       phase_history_complete_v1, phase_missing_reason_v1,
                       movement_consumed_v1, movement_consumed_numerator_v1,
                       movement_consumed_denominator_v1,
                       movement_consumed_complete_v1,
                       movement_consumed_missing_reason_v1,
                       movement_consumed_bucket_v1, tail_phase_source_v1_json
                FROM m1c_checkpoint_v0
                WHERE id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            if checkpoint is None:
                raise KeyError(checkpoint_id)
            if float(checkpoint["probability"]) != decision.probability:
                raise ValueError("episode M1C probability differs from checkpoint")
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
            connection.execute(
                """
                UPDATE m1c_episode_v0
                SET m1c_model_version_v1 = ?,
                    m1c_high_tail_threshold_v1 = ?,
                    phase_at_trigger_v1 = ?,
                    tail_entry_number_v1 = ?,
                    tail_run_length_checkpoints_v1 = ?,
                    tail_run_age_at_trigger_v1 = ?,
                    prior_tail_entries_v1 = ?,
                    previous_checkpoint_above_tail_v1 = ?,
                    minutes_since_previous_tail_exit_v1 = ?,
                    phase_history_complete_v1 = ?,
                    phase_missing_reason_v1 = ?,
                    movement_consumed_at_trigger_v1 = ?,
                    movement_consumed_numerator_v1 = ?,
                    movement_consumed_denominator_v1 = ?,
                    movement_consumed_complete_v1 = ?,
                    movement_consumed_missing_reason_v1 = ?,
                    movement_consumed_bucket_v1 = ?,
                    tail_phase_source_v1_json = ?
                WHERE episode_id = ?
                """,
                (
                    str(checkpoint["model_version"]),
                    float(checkpoint["threshold"]),
                    checkpoint["m1c_tail_phase_v1"],
                    checkpoint["tail_entry_number_v1"],
                    checkpoint["tail_run_length_checkpoints_v1"],
                    checkpoint["tail_run_age_minutes_v1"],
                    checkpoint["prior_tail_entries_v1"],
                    checkpoint["previous_checkpoint_above_tail_v1"],
                    checkpoint["minutes_since_previous_tail_exit_v1"],
                    checkpoint["phase_history_complete_v1"],
                    checkpoint["phase_missing_reason_v1"],
                    checkpoint["movement_consumed_v1"],
                    checkpoint["movement_consumed_numerator_v1"],
                    checkpoint["movement_consumed_denominator_v1"],
                    checkpoint["movement_consumed_complete_v1"],
                    checkpoint["movement_consumed_missing_reason_v1"],
                    checkpoint["movement_consumed_bucket_v1"],
                    checkpoint["tail_phase_source_v1_json"],
                    decision.episode_id,
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

    def verify_partition_hashes(
        self,
        *,
        run_id: str,
        content_hashes: tuple[str, ...],
    ) -> bool:
        """Verify exact SQLite manifests and immutable files before batch reuse."""

        expected = tuple(sorted(set(content_hashes)))
        if not expected:
            return True
        with self.repository._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT content_hash, file_path
                FROM raw_partition_manifest_v0
                WHERE run_id = ?
                  AND content_hash IN ({",".join("?" for _ in expected)})
                """,
                (run_id, *expected),
            ).fetchall()
        if tuple(sorted(str(row["content_hash"]) for row in rows)) != expected:
            return False
        for row in rows:
            content_hash = str(row["content_hash"])
            data_path = Path(str(row["file_path"]))
            metadata_path = data_path.with_name(f"part-{content_hash}.metadata.json")
            if (
                not data_path.is_file()
                or not metadata_path.is_file()
                or sha256_path(data_path) != content_hash
            ):
                return False
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if (
                metadata.get("content_hash") != content_hash
                or Path(str(metadata.get("data_path"))) != data_path
                or metadata.get("run_id") != run_id
            ):
                return False
        return True

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

    def opening_reversal_phase_for_session(
        self,
        *,
        run_id: str,
        session: date,
    ) -> tuple[
        Literal[
            "engineering_transfer",
            "prospective_development",
            "untouched_confirmation",
        ],
        str,
    ]:
        """Resolve V1 phase only from immutable aggregate boundary receipts."""

        with self.repository._connect() as connection:
            decisions = connection.execute(
                """
                SELECT receipt_kind, cohort_last_session, decision
                FROM opening_reversal_decision_receipt_v1
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
        by_kind = {str(row["receipt_kind"]): row for row in decisions}
        transfer = by_kind.get("transfer")
        if transfer is None:
            return "engineering_transfer", "engineering_transfer_pending"
        transfer_decision = str(transfer["decision"])
        transfer_last = transfer["cohort_last_session"]
        if transfer_last is None or str(transfer_last) >= session.isoformat():
            return "engineering_transfer", "engineering_transfer_pending"
        if transfer_decision not in {
            "opening_transfer_supported_without_recalibration",
            "opening_transfer_supported_with_predictor_only_mapping",
        }:
            return "engineering_transfer", transfer_decision
        development = by_kind.get("development")
        confirmation = by_kind.get("confirmation_start")
        if (
            development is not None
            and str(development["decision"]) == "prospective_opening_reversal_development_supported"
            and confirmation is not None
            and confirmation["cohort_last_session"] is not None
            and str(confirmation["cohort_last_session"]) < session.isoformat()
        ):
            return "untouched_confirmation", transfer_decision
        return "prospective_development", transfer_decision

    def record_opening_reversal_decision_receipt_v1(
        self,
        metadata: EvidenceMetadata,
        receipt: OpeningReversalDecisionReceiptV1,
    ) -> int:
        """Persist one immutable aggregate phase/decision receipt."""

        self._validate(metadata)
        encoded = _json(receipt)
        with self.repository._connect() as connection:
            self._validate_opening_reversal_decision_sources_v1(
                connection,
                run_id=metadata.run_id,
                receipt=receipt,
            )
            existing = connection.execute(
                """
                SELECT id, receipt_hash_v1, receipt_json
                FROM opening_reversal_decision_receipt_v1
                WHERE run_id = ? AND receipt_kind = ?
                """,
                (metadata.run_id, receipt.receipt_kind),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["receipt_hash_v1"]) != receipt.receipt_hash_v1
                    or str(existing["receipt_json"]) != encoded
                ):
                    raise ValueError(f"{receipt.receipt_kind} decision receipt is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_decision_receipt_v1(
                    envelope_id, run_id, receipt_kind, boundary_timestamp_utc,
                    decision, cohort_first_session, cohort_last_session,
                    receipt_hash_v1, receipt_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    receipt.receipt_kind,
                    receipt.boundary_timestamp_utc.isoformat(),
                    receipt.decision,
                    (
                        None
                        if receipt.cohort_first_session is None
                        else receipt.cohort_first_session.isoformat()
                    ),
                    (
                        None
                        if receipt.cohort_last_session is None
                        else receipt.cohort_last_session.isoformat()
                    ),
                    receipt.receipt_hash_v1,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    @staticmethod
    def _validate_opening_reversal_decision_sources_v1(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        receipt: OpeningReversalDecisionReceiptV1,
    ) -> None:
        """Bind phase receipts to the exact immutable recorder rows."""

        if receipt.receipt_kind == "transfer":
            rows = connection.execute(
                """
                SELECT transfer.session_date, transfer.report_hash_v1,
                       envelope.recorded_at_utc
                FROM opening_reversal_transfer_session_v1 transfer
                JOIN evidence_envelope envelope
                  ON envelope.id = transfer.envelope_id
                WHERE transfer.run_id = ? AND transfer.valid = 1
                  AND transfer.operational_checks_pass = 1
                ORDER BY session_date
                LIMIT 20
                """,
                (run_id,),
            ).fetchall()
            hashes = tuple(str(row["report_hash_v1"]) for row in rows)
            activation = connection.execute(
                """
                SELECT activation_timestamp_utc, new_york_trading_date
                FROM opening_reversal_activation_v1
                WHERE run_id = ? AND experiment_version = ?
                """,
                (run_id, receipt.experiment_version),
            ).fetchone()
            if (
                len(rows) != 20
                or activation is None
                or hashes != receipt.source_receipt_hashes
                or any(
                    date.fromisoformat(str(row["session_date"]))
                    < date.fromisoformat(str(activation["new_york_trading_date"]))
                    or datetime.fromisoformat(str(row["recorded_at_utc"]))
                    <= datetime.fromisoformat(str(activation["activation_timestamp_utc"]))
                    for row in rows
                )
                or receipt.cohort_first_session != date.fromisoformat(str(rows[0]["session_date"]))
                or receipt.cohort_last_session != date.fromisoformat(str(rows[-1]["session_date"]))
            ):
                raise ValueError("transfer receipt sources do not match first 20 valid sessions")
            return

        if receipt.receipt_kind == "confirmation_start":
            development = connection.execute(
                """
                SELECT decision, boundary_timestamp_utc, cohort_last_session,
                       receipt_hash_v1, receipt_json
                FROM opening_reversal_decision_receipt_v1
                WHERE run_id = ? AND receipt_kind = 'development'
                """,
                (run_id,),
            ).fetchone()
            development_receipt = (
                None
                if development is None
                else OpeningReversalDecisionReceiptV1.model_validate_json(
                    str(development["receipt_json"])
                )
            )
            if (
                development is None
                or development_receipt is None
                or development_receipt.experiment_version != receipt.experiment_version
                or str(development["decision"])
                != "prospective_opening_reversal_development_supported"
                or receipt.source_receipt_hashes != (str(development["receipt_hash_v1"]),)
                or receipt.cohort_last_session is None
                or receipt.cohort_last_session.isoformat()
                != str(development["cohort_last_session"])
                or receipt.boundary_timestamp_utc
                <= datetime.fromisoformat(str(development["boundary_timestamp_utc"]))
            ):
                raise ValueError(
                    "confirmation-start source is not the stored supported development"
                )
            return

        if receipt.receipt_kind == "option_economics":
            direction = connection.execute(
                """
                SELECT receipt_hash_v1, cohort_first_session,
                       cohort_last_session, receipt_json
                FROM opening_reversal_decision_receipt_v1
                WHERE run_id = ? AND receipt_kind = 'confirmation'
                  AND decision =
                      'prospective_opening_reversal_direction_supported'
                """,
                (run_id,),
            ).fetchone()
            option_rows = connection.execute(
                """
                SELECT option_outcome.outcome_hash_v1,
                       option_outcome.outcome_json,
                       option_outcome.prediction_receipt_hash_v1,
                       option_outcome.role,
                       option_outcome.expiry,
                       prediction.session_date,
                       prediction.stock,
                       prediction.opening_transition_event_id_v1,
                       prediction.prediction_v1,
                       prediction.receipt_json
                FROM opening_reversal_primary_option_outcome_v1 option_outcome
                JOIN opening_reversal_prediction_v1 prediction
                  ON prediction.receipt_hash_v1 =
                     option_outcome.prediction_receipt_hash_v1
                JOIN opening_reversal_promotion_v1 promotion
                  ON promotion.promoted_receipt_hash_v1 =
                     prediction.receipt_hash_v1
                WHERE option_outcome.run_id = ?
                  AND option_outcome.complete = 1
                  AND prediction.experiment_id = ?
                  AND prediction.experiment_version = ?
                  AND prediction.scientific_outcome_eligible_v1 = 1
                  AND (
                      prediction.experiment_version != '1.1'
                      OR EXISTS(
                          SELECT 1
                          FROM opening_reversal_v1_1_eligible_episode eligible
                          WHERE eligible.run_id = prediction.run_id
                            AND eligible.prediction_receipt_hash_v1 =
                                prediction.receipt_hash_v1
                      )
                  )
                  AND option_outcome.subscription_end_utc <= ?
                ORDER BY option_outcome.prediction_receipt_hash_v1,
                         option_outcome.role
                """,
                (
                    run_id,
                    M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
                    receipt.experiment_version,
                    receipt.boundary_timestamp_utc.isoformat(),
                ),
            ).fetchall()
            direction_receipt = (
                None
                if direction is None
                else OpeningReversalDecisionReceiptV1.model_validate_json(
                    str(direction["receipt_json"])
                )
            )
            if (
                direction is None
                or direction_receipt is None
                or direction_receipt.experiment_version != receipt.experiment_version
            ):
                raise ValueError("option decision requires stored supported confirmation direction")
            by_prediction: dict[str, list[sqlite3.Row]] = {}
            for row in option_rows:
                by_prediction.setdefault(
                    str(row["prediction_receipt_hash_v1"]),
                    [],
                ).append(row)
            complete_pairs = {
                prediction_hash: rows
                for prediction_hash, rows in by_prediction.items()
                if len(rows) == 2
                and {str(row["role"]) for row in rows} == {"predicted_leg", "opposite_leg"}
            }
            option_hashes = tuple(
                sorted(
                    str(row["outcome_hash_v1"]) for rows in complete_pairs.values() for row in rows
                )
            )
            expected_sources = {
                str(direction["receipt_hash_v1"]),
                *option_hashes,
            }
            if set(receipt.source_receipt_hashes) != expected_sources:
                raise ValueError("option decision sources differ from persisted primary pairs")
            representative_rows = tuple(rows[0] for rows in complete_pairs.values())
            if not representative_rows:
                capacity_blocks = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM opening_reversal_degradation_event_v1
                        WHERE run_id = ?
                          AND reason = 'option_economics_blocked_capacity'
                        """,
                        (run_id,),
                    ).fetchone()[0]
                )
                zero_counts = {
                    "complete_promoted_option_episodes": 0,
                    "call_option_episodes": 0,
                    "put_option_episodes": 0,
                    "unique_severe_opening_events": 0,
                    "represented_stocks": 0,
                    "represented_expiries": 0,
                    "maximum_stock_episode_count": 0,
                    "maximum_expiry_episode_count": 0,
                    "maximum_event_episode_count": 0,
                }
                if (
                    receipt.decision != "option_economics_blocked_capacity"
                    or capacity_blocks == 0
                    or dict(receipt.support_counts) != zero_counts
                    or receipt.cohort_first_session is None
                    or receipt.cohort_last_session is None
                    or receipt.cohort_first_session.isoformat()
                    != str(direction["cohort_first_session"])
                    or receipt.cohort_last_session.isoformat()
                    != str(direction["cohort_last_session"])
                ):
                    raise ValueError("empty option decision lacks persisted capacity block")
                return
            sessions = tuple(
                date.fromisoformat(str(row["session_date"])) for row in representative_rows
            )
            if (
                min(sessions) != receipt.cohort_first_session
                or max(sessions) != receipt.cohort_last_session
            ):
                raise ValueError("option decision cohort boundary differs")
            stock_counts = Counter(str(row["stock"]) for row in representative_rows)
            expiry_counts = Counter(str(rows[0]["expiry"]) for rows in complete_pairs.values())
            event_counts = Counter(
                str(row["opening_transition_event_id_v1"]) for row in representative_rows
            )
            expected_counts = {
                "complete_promoted_option_episodes": len(representative_rows),
                "call_option_episodes": sum(
                    str(row["prediction_v1"]) == "CALL" for row in representative_rows
                ),
                "put_option_episodes": sum(
                    str(row["prediction_v1"]) == "PUT" for row in representative_rows
                ),
                "unique_severe_opening_events": len(event_counts),
                "represented_stocks": len(stock_counts),
                "represented_expiries": len(expiry_counts),
                "maximum_stock_episode_count": max(stock_counts.values()),
                "maximum_expiry_episode_count": max(expiry_counts.values()),
                "maximum_event_episode_count": max(event_counts.values()),
            }
            if dict(receipt.support_counts) != expected_counts:
                raise ValueError("option decision support differs from persisted primary pairs")
            from stocker_prospective.m1c_opening_reversal_analysis_v1 import (
                OpeningReversalOptionEpisodeV1,
                analyze_option_economics_v1,
            )

            option_episodes = []
            for prediction_hash, rows in complete_pairs.items():
                by_role = {str(row["role"]): row for row in rows}
                predicted = PrimaryOptionBidAskOutcomeV1.model_validate_json(
                    str(by_role["predicted_leg"]["outcome_json"])
                )
                opposite = PrimaryOptionBidAskOutcomeV1.model_validate_json(
                    str(by_role["opposite_leg"]["outcome_json"])
                )
                prediction = OpeningReversalPredictionReceiptV1.model_validate_json(
                    str(rows[0]["receipt_json"])
                )
                assert predicted.conservative_return_v1 is not None
                assert opposite.conservative_return_v1 is not None
                option_episodes.append(
                    OpeningReversalOptionEpisodeV1(
                        experiment_version=prediction.experiment_version,
                        prediction_receipt_hash_v1=prediction_hash,
                        predicted_leg_outcome_hash_v1=predicted.outcome_hash_v1,
                        opposite_leg_outcome_hash_v1=opposite.outcome_hash_v1,
                        session=prediction.session,
                        stock=prediction.stock,
                        opening_transition_event_id_v1=cast(
                            str,
                            prediction.opening_transition_event_id_v1,
                        ),
                        prediction_v1=cast(
                            Literal["CALL", "PUT"],
                            prediction.prediction_v1,
                        ),
                        expiry=predicted.contract.expiry,
                        predicted_leg_conservative_return_v1=(predicted.conservative_return_v1),
                        opposite_leg_conservative_return_v1=(opposite.conservative_return_v1),
                        actual_bid_ask_evidence=True,
                        quote_quality_passed=True,
                        staleness_passed=True,
                        continuously_or_adequately_quoted=True,
                    )
                )
            option_analysis = analyze_option_economics_v1(
                option_episodes,
                underlying_direction_supported=True,
                capacity_blocked=(receipt.decision == "option_economics_blocked_capacity"),
            )
            if option_analysis.decision != receipt.decision:
                raise ValueError("option decision differs from frozen bid/ask analysis")
            return

        if receipt.receipt_kind not in {"development", "confirmation"}:
            return
        if receipt.cohort_first_session is None or receipt.cohort_last_session is None:
            raise ValueError("direction decision cohort boundary is missing")
        phase = (
            "prospective_development"
            if receipt.receipt_kind == "development"
            else "untouched_confirmation"
        )
        rows = connection.execute(
            """
            SELECT outcome.outcome_receipt_hash_v1,
                   outcome.outcome_created_at_utc,
                   outcome.outcome_json,
                   outcome.session_date,
                   outcome.stock,
                   outcome.opening_transition_event_id_v1,
                   prediction.opening_transition_sign_v1,
                   prediction.receipt_json,
                   EXISTS(
                       SELECT 1
                       FROM opening_reversal_promotion_v1 promotion
                       WHERE promotion.run_id = prediction.run_id
                         AND promotion.promoted_receipt_hash_v1 =
                             prediction.receipt_hash_v1
                   ) AS promoted,
                   (
                       SELECT COUNT(*)
                       FROM opening_reversal_primary_option_outcome_v1 option_outcome
                       WHERE option_outcome.run_id = prediction.run_id
                         AND option_outcome.prediction_receipt_hash_v1 =
                             prediction.receipt_hash_v1
                         AND option_outcome.complete = 1
                   ) AS complete_option_leg_count
            FROM opening_reversal_underlying_outcome_v1 outcome
            JOIN opening_reversal_prediction_v1 prediction
              ON prediction.receipt_hash_v1 =
                 outcome.prediction_receipt_hash_v1
            WHERE outcome.run_id = ?
              AND prediction.experiment_id = ?
              AND prediction.experiment_version = ?
              AND prediction.cohort_phase = ?
              AND prediction.scientific_outcome_eligible_v1 = 1
              AND (
                  prediction.experiment_version != '1.1'
                  OR EXISTS(
                      SELECT 1
                      FROM opening_reversal_v1_1_eligible_episode eligible
                      WHERE eligible.run_id = prediction.run_id
                        AND eligible.prediction_receipt_hash_v1 =
                            prediction.receipt_hash_v1
                  )
              )
              AND outcome.outcome_completeness_v1 = 'complete'
              AND outcome.outcome_created_at_utc <= ?
            ORDER BY outcome.outcome_receipt_hash_v1
            """,
            (
                run_id,
                M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
                receipt.experiment_version,
                phase,
                receipt.boundary_timestamp_utc.isoformat(),
            ),
        ).fetchall()
        hashes = tuple(str(row["outcome_receipt_hash_v1"]) for row in rows)
        if hashes != tuple(sorted(receipt.source_receipt_hashes)) or not rows:
            raise ValueError(f"{receipt.receipt_kind} sources do not match persisted outcomes")
        sessions = tuple(date.fromisoformat(str(row["session_date"])) for row in rows)
        if (
            min(sessions) != receipt.cohort_first_session
            or max(sessions) != receipt.cohort_last_session
        ):
            raise ValueError(f"{receipt.receipt_kind} cohort boundary differs from outcomes")
        event_identity: dict[str, tuple[date, int]] = {}
        for row, session in zip(rows, sessions, strict=True):
            event_id = str(row["opening_transition_event_id_v1"])
            sign = int(row["opening_transition_sign_v1"])
            identity = (session, sign)
            if event_identity.setdefault(event_id, identity) != identity:
                raise ValueError("one persisted opening event has multiple identities")
        stock_counts = Counter(str(row["stock"]) for row in rows)
        event_counts = Counter(str(row["opening_transition_event_id_v1"]) for row in rows)
        expected_counts = {
            "complete_eligible_stock_episodes": len(rows),
            "unique_severe_opening_events": len(event_identity),
            "positive_transition_events": sum(
                sign == 1 for _session, sign in event_identity.values()
            ),
            "negative_transition_events": sum(
                sign == -1 for _session, sign in event_identity.values()
            ),
            "represented_stocks": len(stock_counts),
            "sessions": len(set(sessions)),
            "maximum_stock_episode_count": max(stock_counts.values()),
            "maximum_event_episode_count": max(event_counts.values()),
        }
        if dict(receipt.support_counts) != expected_counts:
            raise ValueError(
                f"{receipt.receipt_kind} support counts differ from persisted outcomes"
            )
        from stocker_prospective.m1c_opening_reversal_analysis_v1 import (
            analyze_direction_cohort_v1,
            build_opening_reversal_analysis_episode_v1,
        )

        episodes = tuple(
            build_opening_reversal_analysis_episode_v1(
                prediction=OpeningReversalPredictionReceiptV1.model_validate_json(
                    str(row["receipt_json"])
                ),
                outcome=OpeningReversalUnderlyingOutcomeV1.model_validate_json(
                    str(row["outcome_json"])
                ),
                promoted=bool(row["promoted"]),
                primary_option_evidence_complete=(int(row["complete_option_leg_count"]) == 2),
            )
            for row in rows
        )
        direction_analysis = analyze_direction_cohort_v1(
            episodes,
            phase=cast(
                Literal[
                    "prospective_development",
                    "untouched_confirmation",
                ],
                phase,
            ),
        )
        if direction_analysis.decision != receipt.decision:
            raise ValueError(f"{receipt.receipt_kind} decision differs from frozen analysis")

    def maybe_record_opening_transfer_decision_v1(
        self,
        metadata: EvidenceMetadata,
    ) -> OpeningReversalDecisionReceiptV1 | None:
        """Sign the first 20 valid predictor-only sessions exactly once."""

        self._validate(metadata)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT receipt_json
                FROM opening_reversal_decision_receipt_v1
                WHERE run_id = ? AND receipt_kind = 'transfer'
                """,
                (metadata.run_id,),
            ).fetchone()
            if existing is not None:
                return OpeningReversalDecisionReceiptV1.model_validate_json(
                    str(existing["receipt_json"])
                )
            rows = connection.execute(
                """
                SELECT report_json
                FROM opening_reversal_transfer_session_v1
                WHERE run_id = ? AND valid = 1
                ORDER BY session_date
                LIMIT 20
                """,
                (metadata.run_id,),
            ).fetchall()
            activation = connection.execute(
                """
                SELECT experiment_version
                FROM opening_reversal_activation_v1
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (metadata.run_id,),
            ).fetchone()
        if len(rows) < 20:
            return None
        if activation is None or str(activation["experiment_version"]) not in {
            "1",
            "1.1",
        }:
            raise ValueError("opening reversal activation is unavailable")
        receipt = build_opening_transfer_decision_receipt_v1(
            sessions=tuple(
                OpeningTransferSessionResultV1.model_validate_json(str(row["report_json"]))
                for row in rows
            ),
            boundary_timestamp_utc=metadata.recorded_at_utc,
            experiment_version=cast(
                Literal["1", "1.1"],
                str(activation["experiment_version"]),
            ),
        )
        self.record_opening_reversal_decision_receipt_v1(metadata, receipt)
        return receipt

    def record_opening_reversal_activation_v1(
        self,
        metadata: EvidenceMetadata,
        receipt: OpeningReversalActivationReceiptV1,
    ) -> int:
        """Persist the activation boundary once; a retry must be byte-identical."""

        self._validate(metadata)
        encoded = _json(receipt)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, activation_receipt_hash, receipt_json
                FROM opening_reversal_activation_v1
                WHERE run_id = ? AND experiment_id = ? AND experiment_version = ?
                """,
                (
                    metadata.run_id,
                    receipt.experiment_id,
                    receipt.experiment_version,
                ),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["activation_receipt_hash"]) != receipt.activation_receipt_hash
                    or str(existing["receipt_json"]) != encoded
                ):
                    raise ValueError("opening reversal activation is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_activation_v1(
                    envelope_id, run_id, experiment_id, experiment_version,
                    activation_timestamp_utc, new_york_trading_date,
                    configuration_hash, frozen_rule_hash,
                    configured_reserved_line_count, order_routing_disabled,
                    activation_receipt_hash, receipt_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 12, 1, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    receipt.experiment_id,
                    receipt.experiment_version,
                    receipt.activation_timestamp_utc.isoformat(),
                    receipt.new_york_trading_date_at_activation.isoformat(),
                    receipt.configuration_hash,
                    receipt.frozen_rule_hash,
                    receipt.activation_receipt_hash,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_opening_reversal_activation_v1_1(
        self,
        metadata: EvidenceMetadata,
        receipt: OpeningReversalActivationReceiptV1_1,
    ) -> int:
        """Persist the immutable timing addendum without replacing V1."""

        self._validate(metadata)
        encoded = _json(receipt)
        with self.repository._connect() as connection:
            base = connection.execute(
                """
                SELECT activation_receipt_hash, configuration_hash,
                       frozen_rule_hash
                FROM opening_reversal_activation_v1
                WHERE run_id = ? AND experiment_id = ?
                  AND experiment_version = '1'
                """,
                (metadata.run_id, receipt.experiment_id),
            ).fetchone()
            if (
                base is None
                or str(base["activation_receipt_hash"])
                != receipt.superseded_activation_receipt_hash_v1
                or str(base["configuration_hash"]) != receipt.frozen_configuration_hash_v1
                or str(base["frozen_rule_hash"]) != receipt.frozen_rule_hash
            ):
                raise ValueError("opening reversal V1.1 lacks the exact V1 activation")
            existing = connection.execute(
                """
                SELECT id, activation_receipt_hash, receipt_json
                FROM opening_reversal_activation_v1
                WHERE run_id = ? AND experiment_id = ?
                  AND experiment_version = '1.1'
                """,
                (metadata.run_id, receipt.experiment_id),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["activation_receipt_hash"]) != receipt.activation_receipt_hash_v1_1
                    or str(existing["receipt_json"]) != encoded
                ):
                    raise ValueError("opening reversal V1.1 activation is immutable")
                return int(existing["id"])
            prior_experiment_rows = sum(
                int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE run_id = ?",
                        (metadata.run_id,),
                    ).fetchone()[0]
                )
                for table in (
                    "opening_reversal_prediction_v1",
                    "opening_reversal_transfer_session_v1",
                    "opening_reversal_underlying_outcome_v1",
                    "opening_reversal_decision_receipt_v1",
                )
            )
            if prior_experiment_rows:
                raise ValueError(
                    "opening reversal V1.1 requires a fresh run so the "
                    "20-session engineering transfer restarts"
                )
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_activation_v1(
                    envelope_id, run_id, experiment_id, experiment_version,
                    activation_timestamp_utc, new_york_trading_date,
                    configuration_hash, frozen_rule_hash,
                    configured_reserved_line_count, order_routing_disabled,
                    activation_receipt_hash, receipt_json
                ) VALUES (?, ?, ?, '1.1', ?, ?, ?, ?, 12, 1, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    receipt.experiment_id,
                    receipt.activation_timestamp_utc.isoformat(),
                    receipt.new_york_trading_date_at_activation.isoformat(),
                    receipt.timing_addendum_configuration_hash_v1_1,
                    receipt.frozen_rule_hash,
                    receipt.activation_receipt_hash_v1_1,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_opening_reversal_prediction_v1(
        self,
        metadata: EvidenceMetadata,
        receipt: OpeningReversalPredictionReceiptV1,
    ) -> int:
        """Append one immutable receipt; corrections use their own table."""

        self._validate(metadata)
        expected_phase, expected_transfer_status = self.opening_reversal_phase_for_session(
            run_id=metadata.run_id,
            session=receipt.session,
        )
        if (
            receipt.cohort_phase != expected_phase
            or receipt.transfer_status != expected_transfer_status
        ):
            raise ValueError("opening reversal prediction phase differs from immutable boundaries")
        encoded = _json(receipt)
        with self.repository._connect() as connection:
            activation = connection.execute(
                """
                SELECT activation_timestamp_utc
                FROM opening_reversal_activation_v1
                WHERE run_id = ? AND experiment_id = ?
                  AND experiment_version = ?
                """,
                (
                    metadata.run_id,
                    receipt.experiment_id,
                    receipt.experiment_version,
                ),
            ).fetchone()
            if activation is None or receipt.receipt_created_at_utc <= datetime.fromisoformat(
                str(activation["activation_timestamp_utc"])
            ):
                raise ValueError("opening reversal prediction lacks prior activation boundary")
            existing = connection.execute(
                """
                SELECT id, receipt_hash_v1, receipt_json
                FROM opening_reversal_prediction_v1
                WHERE run_id = ? AND session_date = ? AND stock = ?
                  AND checkpoint = 6 AND experiment_id = ?
                  AND experiment_version = ?
                """,
                (
                    metadata.run_id,
                    receipt.session.isoformat(),
                    receipt.stock,
                    receipt.experiment_id,
                    receipt.experiment_version,
                ),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["receipt_hash_v1"]) != receipt.receipt_hash_v1
                    or str(existing["receipt_json"]) != encoded
                ):
                    raise ValueError("opening reversal prediction receipt is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_prediction_v1(
                    envelope_id, run_id, experiment_id, experiment_version,
                    session_date, stock, checkpoint, signal_timestamp_utc,
                    entry_timestamp_utc, receipt_created_at_utc,
                    m1c_probability, m1c_threshold, high_tail_membership,
                    fresh_episode_id, tail_phase_v1,
                    market_opening_return_v1, market_opening_range_v1,
                    opening_market_transition_state_v1,
                    opening_transition_sign_v1, opening_transition_event_id_v1,
                    data_source, transfer_status, cohort_phase, prediction_v1,
                    prediction_sign_v1, eligibility_v1,
                    ineligibility_reasons_v1_json, completeness_status_v1,
                    scientific_outcome_eligible_v1,
                    scientific_exclusion_reason_v1, capacity_snapshot_id,
                    previous_close_atm_iv_scale_15m, frozen_comparisons_json,
                    rule_hash_v1, receipt_hash_v1, receipt_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, 6, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    receipt.experiment_id,
                    receipt.experiment_version,
                    receipt.session.isoformat(),
                    receipt.stock,
                    receipt.signal_timestamp_utc.isoformat(),
                    receipt.entry_timestamp_utc.isoformat(),
                    receipt.receipt_created_at_utc.isoformat(),
                    receipt.m1c_probability,
                    receipt.m1c_threshold,
                    int(receipt.high_tail_membership),
                    receipt.fresh_episode_id,
                    receipt.tail_phase_v1,
                    receipt.market_opening_return_v1,
                    receipt.market_opening_range_v1,
                    receipt.opening_market_transition_state_v1,
                    receipt.opening_transition_sign_v1,
                    receipt.opening_transition_event_id_v1,
                    receipt.data_source,
                    receipt.transfer_status,
                    receipt.cohort_phase,
                    receipt.prediction_v1,
                    receipt.prediction_sign_v1,
                    int(receipt.eligibility_v1),
                    _json(receipt.ineligibility_reasons_v1),
                    receipt.completeness_status_v1,
                    int(receipt.scientific_outcome_eligible_v1),
                    receipt.scientific_exclusion_reason_v1,
                    receipt.capacity_snapshot_id,
                    receipt.previous_close_atm_iv_scale_15m,
                    _json(receipt.frozen_comparisons),
                    receipt.rule_hash_v1,
                    receipt.receipt_hash_v1,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def load_opening_reversal_prediction_v1(
        self,
        *,
        run_id: str,
        session: date,
        stock: str,
        experiment_version: Literal["1", "1.1"] = "1",
    ) -> OpeningReversalPredictionReceiptV1 | None:
        """Read an already-frozen receipt so late bars cannot replace it."""

        with self.repository._connect() as connection:
            row = connection.execute(
                """
                SELECT receipt_json
                FROM opening_reversal_prediction_v1
                WHERE run_id = ? AND session_date = ? AND stock = ?
                      AND checkpoint = 6 AND experiment_version = ?
                """,
                (run_id, session.isoformat(), stock, experiment_version),
            ).fetchone()
        if row is None:
            return None
        return OpeningReversalPredictionReceiptV1.model_validate_json(str(row["receipt_json"]))

    def record_opening_reversal_promotion_v1(
        self,
        metadata: EvidenceMetadata,
        selection: PromotionSelectionV1,
    ) -> int:
        """Persist the one-stock promotion ranking and every losing candidate."""

        self._validate(metadata)
        promoted = selection.promoted
        if promoted is None:
            raise ValueError("promotion receipt requires an eligible winner")
        payload = {
            "session": promoted.session,
            "opening_transition_event_id_v1": (promoted.opening_transition_event_id_v1),
            "promoted_receipt_hash_v1": promoted.receipt_hash_v1,
            "promoted_stock": promoted.stock,
            "eligible_count": selection.eligible_count,
            "maximum_promoted_count": selection.maximum_promoted_count,
            "selection_rule": selection.selection_rule,
            "non_promoted": selection.non_promoted,
        }
        promotion_hash = _content_hash(payload)
        encoded_losers = _json(selection.non_promoted)
        with self.repository._connect() as connection:
            if promoted.experiment_version == "1.1":
                barrier = connection.execute(
                    """
                    SELECT audit_json
                    FROM opening_reversal_causal_barrier_audit_v1_1
                    WHERE run_id = ? AND session_date = ?
                      AND barrier_status = 'passed'
                    """,
                    (metadata.run_id, promoted.session.isoformat()),
                ).fetchone()
                if barrier is None:
                    raise ValueError("opening reversal V1.1 causal barrier has not passed")
                audit = OpeningReversalCausalBarrierAuditV1_1.model_validate_json(
                    str(barrier["audit_json"])
                )
                if promoted.receipt_hash_v1 not in audit.prediction_receipt_hashes:
                    raise ValueError("opening reversal V1.1 promotion is outside causal barrier")
            existing = connection.execute(
                """
                SELECT id, promotion_hash_v1
                FROM opening_reversal_promotion_v1
                WHERE run_id = ? AND opening_transition_event_id_v1 = ?
                """,
                (
                    metadata.run_id,
                    promoted.opening_transition_event_id_v1,
                ),
            ).fetchone()
            if existing is not None:
                if str(existing["promotion_hash_v1"]) != promotion_hash:
                    raise ValueError("opening reversal promotion is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_promotion_v1(
                    envelope_id, run_id, session_date,
                    opening_transition_event_id_v1, promoted_receipt_hash_v1,
                    promoted_stock, eligible_count, maximum_promoted_count,
                    selection_rule, non_promoted_json, promotion_hash_v1
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    promoted.session.isoformat(),
                    promoted.opening_transition_event_id_v1,
                    promoted.receipt_hash_v1,
                    promoted.stock,
                    selection.eligible_count,
                    selection.selection_rule,
                    encoded_losers,
                    promotion_hash,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_opening_reversal_causal_barrier_audit_v1_1(
        self,
        metadata: EvidenceMetadata,
        audit: OpeningReversalCausalBarrierAuditV1_1,
    ) -> int:
        """Persist barrier proof before any buffered decision data is released."""

        self._validate(metadata)
        encoded = _json(audit)
        with self.repository._connect() as connection:
            activation = connection.execute(
                """
                SELECT activation_receipt_hash
                FROM opening_reversal_activation_v1
                WHERE run_id = ? AND experiment_id = ?
                  AND experiment_version = '1.1'
                """,
                (metadata.run_id, audit.experiment_id),
            ).fetchone()
            persisted_hashes = tuple(
                sorted(
                    str(row["receipt_hash_v1"])
                    for row in connection.execute(
                        """
                        SELECT receipt_hash_v1
                        FROM opening_reversal_prediction_v1
                        WHERE run_id = ? AND experiment_version = '1.1'
                          AND session_date = ?
                        """,
                        (metadata.run_id, audit.session.isoformat()),
                    ).fetchall()
                )
            )
            if (
                activation is None
                or str(activation["activation_receipt_hash"]) != audit.activation_receipt_hash_v1_1
                or persisted_hashes != audit.prediction_receipt_hashes
            ):
                raise ValueError("opening reversal V1.1 barrier sources are not durable")
            existing = connection.execute(
                """
                SELECT id, audit_hash_v1_1, audit_json
                FROM opening_reversal_causal_barrier_audit_v1_1
                WHERE run_id = ? AND session_date = ?
                """,
                (metadata.run_id, audit.session.isoformat()),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["audit_hash_v1_1"]) != audit.audit_hash_v1_1
                    or str(existing["audit_json"]) != encoded
                ):
                    raise ValueError("opening reversal V1.1 barrier audit is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_causal_barrier_audit_v1_1(
                    envelope_id, run_id, experiment_id, experiment_version,
                    activation_receipt_hash_v1_1, session_date,
                    nominal_entry_timestamp_utc, prediction_receipt_count,
                    prediction_receipt_hashes_json, deferred_event_count,
                    first_deferred_event_received_at_utc,
                    entry_or_post_entry_data_admitted_before_receipts,
                    raw_event_archive_write_allowed, core_recorder_continued,
                    barrier_status, failure_reason, release_authorized_at_utc,
                    audit_hash_v1_1, audit_json
                ) VALUES (
                    ?, ?, ?, '1.1', ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    audit.experiment_id,
                    audit.activation_receipt_hash_v1_1,
                    audit.session.isoformat(),
                    audit.nominal_entry_timestamp_utc.isoformat(),
                    audit.prediction_receipt_count,
                    _json(audit.prediction_receipt_hashes),
                    audit.deferred_event_count,
                    (
                        None
                        if audit.first_deferred_event_received_at_utc is None
                        else audit.first_deferred_event_received_at_utc.isoformat()
                    ),
                    int(audit.entry_or_post_entry_data_admitted_before_receipts),
                    audit.barrier_status,
                    audit.failure_reason,
                    audit.release_authorized_at_utc.isoformat(),
                    audit.audit_hash_v1_1,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def load_opening_reversal_causal_barrier_audit_v1_1(
        self,
        *,
        run_id: str,
        session: date,
    ) -> OpeningReversalCausalBarrierAuditV1_1 | None:
        """Restore the immutable gate disposition after a recorder restart."""

        with self.repository._connect() as connection:
            row = connection.execute(
                """
                SELECT audit_json
                FROM opening_reversal_causal_barrier_audit_v1_1
                WHERE run_id = ? AND session_date = ?
                """,
                (run_id, session.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return OpeningReversalCausalBarrierAuditV1_1.model_validate_json(str(row["audit_json"]))

    def record_opening_reversal_capacity_snapshot_v1(
        self,
        metadata: EvidenceMetadata,
        snapshot: MarketDataCapacitySnapshotV1,
    ) -> int:
        self._validate(metadata)
        encoded = _json(snapshot)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, snapshot_json
                FROM opening_reversal_capacity_snapshot_v1
                WHERE snapshot_hash_v1 = ?
                """,
                (snapshot.snapshot_hash,),
            ).fetchone()
            if existing is not None:
                if str(existing["snapshot_json"]) != encoded:
                    raise ValueError("capacity snapshot hash collision")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_capacity_snapshot_v1(
                    envelope_id, run_id, timestamp_utc, configured_budget,
                    reserved_lines, mandatory_lines, optional_lines,
                    pending_lines, cancelled_lines,
                    lines_awaiting_acknowledgement_or_cleanup,
                    estimated_free_lines, current_promoted_episode_id,
                    snapshot_hash_v1, snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    snapshot.timestamp_utc.isoformat(),
                    snapshot.configured_budget,
                    snapshot.reserved_lines,
                    snapshot.mandatory_lines,
                    snapshot.optional_lines,
                    snapshot.pending_lines,
                    snapshot.cancelled_lines,
                    snapshot.lines_awaiting_acknowledgement_or_cleanup,
                    snapshot.estimated_free_lines,
                    snapshot.current_promoted_episode_id,
                    snapshot.snapshot_hash,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_opening_reversal_degradation_v1(
        self,
        metadata: EvidenceMetadata,
        event: CapacityDegradationEventV1,
        *,
        capacity_snapshot_hash_v1: str | None,
    ) -> int:
        self._validate(metadata)
        payload = {
            **event.model_dump(mode="json"),
            "capacity_snapshot_hash_v1": capacity_snapshot_hash_v1,
        }
        event_hash = _content_hash(payload)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM opening_reversal_degradation_event_v1
                WHERE event_hash_v1 = ?
                """,
                (event_hash,),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_degradation_event_v1(
                    envelope_id, run_id, timestamp_utc, episode_id, feed,
                    subscription_ids_json, reason, raw_capacity_reason,
                    capacity_snapshot_hash_v1,
                    primary_direction_evidence_remains_complete,
                    primary_option_evidence_remains_complete, event_hash_v1
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    event.timestamp_utc.isoformat(),
                    event.episode_id,
                    None if event.feed is None else event.feed.value,
                    _json(event.subscription_ids),
                    event.reason,
                    event.raw_capacity_reason,
                    capacity_snapshot_hash_v1,
                    int(event.primary_option_evidence_remains_complete),
                    event_hash,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_opening_reversal_contract_discovery_v1(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        selection: PrimaryOptionPairSelectionV1,
    ) -> int:
        self._validate(metadata)
        encoded = _json(selection)
        with self.repository._connect() as connection:
            prediction = connection.execute(
                """
                SELECT experiment_id, experiment_version, session_date
                FROM opening_reversal_prediction_v1
                WHERE run_id = ? AND fresh_episode_id = ?
                """,
                (metadata.run_id, episode_id),
            ).fetchone()
            if (
                prediction is None
                or str(prediction["experiment_id"]) != M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID
            ):
                raise ValueError("blocked_opening_reversal_episode_identity_missing")
            if str(prediction["experiment_version"]) == "1.1":
                if (
                    connection.execute(
                        """
                        SELECT 1
                        FROM opening_reversal_v1_1_capture_eligible_episode
                        WHERE run_id = ? AND episode_id = ?
                        """,
                        (metadata.run_id, episode_id),
                    ).fetchone()
                    is None
                ):
                    raise ValueError("blocked_v1_1_episode_not_eligible")
                validate_primary_option_protocol_v1_1(
                    session=date.fromisoformat(str(prediction["session_date"])),
                    contracts=(selection.call, selection.put),
                )
            existing = connection.execute(
                """
                SELECT id, audit_hash_v1, audit_json
                FROM opening_reversal_contract_discovery_v1
                WHERE run_id = ? AND episode_id = ?
                """,
                (metadata.run_id, episode_id),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["audit_hash_v1"]) != selection.selection_hash
                    or str(existing["audit_json"]) != encoded
                ):
                    raise ValueError("contract discovery audit is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_contract_discovery_v1(
                    envelope_id, run_id, episode_id, discovery_timestamp_utc,
                    contract_source, cache_hit, candidates_inspected,
                    call_con_id, put_con_id, expiry, strike, tie_break_rule,
                    live_market_data_lines_consumed, planned_live_market_data_lines,
                    metadata_request_ended,
                    full_chain_live_subscription_created, status, missing_reason,
                    audit_hash_v1, audit_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0,
                          'selected', NULL, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    episode_id,
                    selection.discovery_timestamp_utc.isoformat(),
                    selection.contract_source,
                    int(selection.cache_hit),
                    selection.candidates_inspected,
                    selection.call.con_id,
                    selection.put.con_id,
                    selection.call.expiry.isoformat(),
                    selection.call.strike,
                    selection.frozen_tie_break_rule,
                    selection.live_market_data_lines_consumed,
                    selection.planned_live_market_data_lines,
                    selection.selection_hash,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_opening_reversal_contract_discovery_failure_v1(
        self,
        metadata: EvidenceMetadata,
        *,
        episode_id: str,
        discovery_timestamp_utc: datetime,
        contract_source: str,
        cache_hit: bool,
        candidates_inspected: int,
        missing_reason: str,
    ) -> int:
        """Persist a terminal metadata/qualification failure without streaming."""

        self._validate(metadata)
        payload = {
            "episode_id": episode_id,
            "discovery_timestamp_utc": discovery_timestamp_utc,
            "contract_source": contract_source,
            "cache_hit": cache_hit,
            "candidates_inspected": candidates_inspected,
            "tie_break_rule": (
                "1dte_common_nearest_atm_absolute_distance_then_lower_strike_then_con_id"
            ),
            "metadata_request_ended": True,
            "full_chain_live_subscription_created": False,
            "live_market_data_lines_consumed": 0,
            "planned_live_market_data_lines": 0,
            "status": "failed",
            "missing_reason": missing_reason,
        }
        audit_hash = _content_hash(payload)
        encoded = _json(payload)
        with self.repository._connect() as connection:
            prediction = connection.execute(
                """
                SELECT experiment_id, experiment_version
                FROM opening_reversal_prediction_v1
                WHERE run_id = ? AND fresh_episode_id = ?
                """,
                (metadata.run_id, episode_id),
            ).fetchone()
            if (
                prediction is None
                or str(prediction["experiment_id"]) != M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID
            ):
                raise ValueError("blocked_opening_reversal_episode_identity_missing")
            if (
                str(prediction["experiment_version"]) == "1.1"
                and connection.execute(
                    """
                    SELECT 1
                    FROM opening_reversal_v1_1_capture_eligible_episode
                    WHERE run_id = ? AND episode_id = ?
                    """,
                    (metadata.run_id, episode_id),
                ).fetchone()
                is None
            ):
                raise ValueError("blocked_v1_1_episode_not_eligible")
            existing = connection.execute(
                """
                SELECT id, audit_hash_v1, audit_json
                FROM opening_reversal_contract_discovery_v1
                WHERE run_id = ? AND episode_id = ?
                """,
                (metadata.run_id, episode_id),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["audit_hash_v1"]) != audit_hash
                    or str(existing["audit_json"]) != encoded
                ):
                    raise ValueError("contract discovery audit is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_contract_discovery_v1(
                    envelope_id, run_id, episode_id, discovery_timestamp_utc,
                    contract_source, cache_hit, candidates_inspected,
                    call_con_id, put_con_id, expiry, strike, tie_break_rule,
                    live_market_data_lines_consumed, planned_live_market_data_lines,
                    metadata_request_ended,
                    full_chain_live_subscription_created, status, missing_reason,
                    audit_hash_v1, audit_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?,
                          0, 0, 1, 0, 'failed', ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    episode_id,
                    discovery_timestamp_utc.isoformat(),
                    contract_source,
                    int(cache_hit),
                    candidates_inspected,
                    payload["tie_break_rule"],
                    missing_reason,
                    audit_hash,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_opening_reversal_transfer_session_v1(
        self,
        metadata: EvidenceMetadata,
        result: OpeningTransferSessionResultV1,
    ) -> int:
        """Persist one engineering-only predictor transfer report."""

        self._validate(metadata)
        encoded = _json(result)
        with self.repository._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, report_hash_v1, report_json
                FROM opening_reversal_transfer_session_v1
                WHERE run_id = ? AND session_date = ?
                """,
                (metadata.run_id, result.session.isoformat()),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["report_hash_v1"]) != result.report_hash_v1
                    or str(existing["report_json"]) != encoded
                ):
                    raise ValueError("opening transfer session is immutable")
                return int(existing["id"])
            prior_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM opening_reversal_transfer_session_v1
                    WHERE run_id = ? AND valid = 1 AND session_date < ?
                    """,
                    (metadata.run_id, result.session.isoformat()),
                ).fetchone()[0]
            )
            valid_ordinal = prior_count + 1 if result.valid else None
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_transfer_session_v1(
                    envelope_id, run_id, session_date, valid,
                    valid_session_ordinal, decision, ibkr_opening_return,
                    eodhd_opening_return, ibkr_opening_range,
                    eodhd_opening_range, severe_state_agreement,
                    sign_agreement, timestamp_alignment,
                    checkpoint_6_episode_identity_agreement,
                    operational_checks_pass, operational_evidence_json,
                    outcome_fields_accessed, report_json, report_hash_v1
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0,
                          ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    result.session.isoformat(),
                    int(result.valid),
                    valid_ordinal,
                    result.decision,
                    result.ibkr_opening_return,
                    result.eodhd_opening_return,
                    result.ibkr_opening_range,
                    result.eodhd_opening_range,
                    int(result.severe_state_agreement),
                    int(result.sign_agreement),
                    int(result.bar_timestamp_alignment),
                    int(result.checkpoint_6_episode_identity_agreement),
                    int(result.operational_evidence.critical_checks_pass),
                    _json(result.operational_evidence),
                    encoded,
                    result.report_hash_v1,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def opening_reversal_engineering_operational_evidence_v1(
        self,
        *,
        run_id: str,
        session: date,
    ) -> OpeningTransferOperationalEvidenceV1:
        """Derive one outcome-free engineering guard audit from recorder rows."""

        session_text = session.isoformat()
        with self.repository._connect() as connection:
            activation_row = connection.execute(
                """
                SELECT experiment_version, receipt_json
                FROM opening_reversal_activation_v1
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            active_version = (
                None if activation_row is None else str(activation_row["experiment_version"])
            )
            predictions = connection.execute(
                """
                SELECT experiment_version, receipt_hash_v1,
                       receipt_created_at_utc,
                       entry_timestamp_utc, capacity_snapshot_id,
                       fresh_episode_id, receipt_json
                FROM opening_reversal_prediction_v1
                WHERE run_id = ? AND session_date = ? AND checkpoint = 6
                  AND experiment_version = ?
                ORDER BY stock
                """,
                (run_id, session_text, active_version),
            ).fetchall()
            barrier_row = connection.execute(
                """
                SELECT barrier_status, prediction_receipt_hashes_json,
                       audit_json
                FROM opening_reversal_causal_barrier_audit_v1_1
                WHERE run_id = ? AND session_date = ?
                """,
                (run_id, session_text),
            ).fetchone()
            correction_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM opening_reversal_prediction_correction_v1 c
                    JOIN opening_reversal_prediction_v1 p
                      ON p.receipt_hash_v1 = c.original_receipt_hash_v1
                    WHERE c.run_id = ? AND p.session_date = ?
                    """,
                    (run_id, session_text),
                ).fetchone()[0]
            )
            capacity_rows = connection.execute(
                """
                SELECT snapshot_hash_v1, reserved_lines
                FROM opening_reversal_capacity_snapshot_v1
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
            promoted = connection.execute(
                """
                SELECT prediction.fresh_episode_id
                FROM opening_reversal_promotion_v1 promotion
                JOIN opening_reversal_prediction_v1 prediction
                  ON prediction.receipt_hash_v1 =
                     promotion.promoted_receipt_hash_v1
                WHERE promotion.run_id = ?
                  AND promotion.session_date = ?
                  AND promotion.promoted_receipt_hash_v1 IS NOT NULL
                """,
                (run_id, session_text),
            ).fetchall()
            quality_row = connection.execute(
                """
                SELECT report_json
                FROM recorder_session_report_v0
                WHERE run_id = ? AND session_date = ?
                """,
                (run_id, session_text),
            ).fetchone()
            promoted_ids = tuple(
                str(row["fresh_episode_id"])
                for row in promoted
                if row["fresh_episode_id"] is not None
            )
            discovery_rows = []
            latest_allocations: dict[str, str | None] = {}
            promoted_level1_started: dict[str, bool] = {}
            degradation_rows = []
            for episode_id in promoted_ids:
                promoted_level1_started[episode_id] = (
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM subscription_lifecycle_v0
                        WHERE run_id = ? AND owner_episode = ?
                          AND subscription_kind = 'level1'
                          AND capacity_denied = 0
                        """,
                        (run_id, episode_id),
                    ).fetchone()[0]
                    > 0
                )
                discovery = connection.execute(
                    """
                    SELECT status
                    FROM opening_reversal_contract_discovery_v1
                    WHERE run_id = ? AND episode_id = ?
                    """,
                    (run_id, episode_id),
                ).fetchone()
                if discovery is not None:
                    discovery_rows.append(discovery)
                allocation = connection.execute(
                    """
                    SELECT state
                    FROM option_episode_allocation_v0
                    WHERE run_id = ? AND episode_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (run_id, episode_id),
                ).fetchone()
                latest_allocations[episode_id] = (
                    None if allocation is None else str(allocation["state"])
                )
                degradation_rows.extend(
                    connection.execute(
                        """
                        SELECT reason,
                               primary_direction_evidence_remains_complete
                        FROM opening_reversal_degradation_event_v1
                        WHERE run_id = ? AND episode_id = ?
                        """,
                        (run_id, episode_id),
                    ).fetchall()
                )

        prediction_count = len(predictions)
        prediction_hashes = tuple(sorted(str(row["receipt_hash_v1"]) for row in predictions))
        if active_version == "1.1":
            barrier_hashes = (
                ()
                if barrier_row is None
                else tuple(
                    sorted(
                        cast(
                            list[str],
                            json.loads(str(barrier_row["prediction_receipt_hashes_json"])),
                        )
                    )
                )
            )
            timing_pass = (
                prediction_count == 20
                and barrier_row is not None
                and str(barrier_row["barrier_status"]) == "passed"
                and barrier_hashes == prediction_hashes
                and all(
                    (
                        receipt := (
                            OpeningReversalPredictionReceiptV1.model_validate_json(
                                str(row["receipt_json"])
                            )
                        )
                    ).experiment_version
                    == "1.1"
                    and receipt.timing_evidence_v1_1 is not None
                    and not (
                        receipt.timing_evidence_v1_1.entry_or_post_entry_data_admitted_before_receipt
                    )
                    for row in predictions
                )
            )
        else:
            timing_pass = prediction_count == 20 and all(
                datetime.fromisoformat(str(row["receipt_created_at_utc"]))
                < datetime.fromisoformat(str(row["entry_timestamp_utc"]))
                for row in predictions
            )
        immutability_pass = prediction_count == 20 and correction_count == 0
        capacity_by_hash = {
            str(row["snapshot_hash_v1"]): int(row["reserved_lines"]) for row in capacity_rows
        }
        required_snapshot_ids = tuple(
            str(row["capacity_snapshot_id"])
            for row in predictions
            if row["capacity_snapshot_id"] is not None
        )
        capacity_complete = (
            prediction_count == 20
            and len(required_snapshot_ids) == prediction_count
            and all(snapshot_id in capacity_by_hash for snapshot_id in required_snapshot_ids)
        )
        reserve_pass = capacity_complete and all(
            capacity_by_hash[snapshot_id] >= RESERVED_MARKET_DATA_LINES_V1
            for snapshot_id in required_snapshot_ids
        )
        discovery_complete = len(discovery_rows) == len(promoted_ids)
        selected_count = sum(str(row["status"]) == "selected" for row in discovery_rows)
        promoted_underlying_level1_pass = not promoted_ids or all(promoted_level1_started.values())
        pair_recording_pass = not promoted_ids or (
            discovery_complete
            and selected_count == len(promoted_ids)
            and all(latest_allocations[episode_id] == "COMPLETE" for episode_id in promoted_ids)
        )
        graceful_degradation_pass = all(
            bool(row["primary_direction_evidence_remains_complete"])
            and str(row["reason"])
            in {
                "optional_feed_not_started_capacity_reserved",
                "option_economics_blocked_capacity",
            }
            for row in degradation_rows
        )
        cancellation_recovery_pass = not promoted_ids or all(
            latest_allocations[episode_id] == "COMPLETE" for episode_id in promoted_ids
        )
        quality = (
            {}
            if quality_row is None
            else cast(
                dict[str, object],
                json.loads(str(quality_row["report_json"])),
            )
        )
        universe_uninterrupted = (
            bool(quality.get("complete"))
            and float(cast(float | int, quality.get("m1c_checkpoint_coverage", 0.0))) == 1.0
            and int(cast(float | int, quality.get("m1c_predictions", 0))) >= 20 * 15
        )
        recorder_reliability_pass = (
            bool(quality.get("complete"))
            and int(cast(float | int, quality.get("data_gaps", 1))) == 0
            and int(cast(float | int, quality.get("pacing_errors", 1))) == 0
        )
        activation = (
            {}
            if activation_row is None
            else cast(
                dict[str, object],
                json.loads(str(activation_row["receipt_json"])),
            )
        )
        claims = claims_boundary()
        no_order_guard_pass = (
            activation.get("order_routing_disabled") is True
            and activation.get("order_methods_available") is False
            and claims["paper_orders_allowed"] is False
            and claims["live_orders_allowed"] is False
            and claims["order_methods_available"] is False
        )
        checks = (
            (
                prediction_count == 20,
                "engineering_prediction_receipt_count_not_20",
            ),
            (timing_pass, "engineering_prediction_receipt_timing_failed"),
            (
                immutability_pass,
                "engineering_prediction_receipt_immutability_failed",
            ),
            (capacity_complete, "engineering_capacity_snapshots_incomplete"),
            (reserve_pass, "engineering_reserved_twelve_lines_failed"),
            (
                promoted_underlying_level1_pass,
                "engineering_promoted_underlying_level1_failed",
            ),
            (discovery_complete, "engineering_contract_discovery_incomplete"),
            (
                pair_recording_pass,
                "engineering_primary_option_pair_recording_failed",
            ),
            (
                graceful_degradation_pass,
                "engineering_graceful_degradation_failed",
            ),
            (
                cancellation_recovery_pass,
                "engineering_cancellation_recovery_failed",
            ),
            (
                universe_uninterrupted,
                "engineering_m1c_universe_interrupted",
            ),
            (
                recorder_reliability_pass,
                "engineering_recorder_reliability_failed",
            ),
            (no_order_guard_pass, "engineering_no_order_guard_failed"),
        )
        missing_reasons = tuple(reason for passed, reason in checks if not passed)
        return OpeningTransferOperationalEvidenceV1(
            prediction_receipt_count=prediction_count,
            prediction_receipt_timing_pass=timing_pass,
            prediction_receipt_immutability_pass=immutability_pass,
            capacity_snapshot_count=len(capacity_by_hash),
            capacity_snapshots_complete=capacity_complete,
            reserved_twelve_lines_pass=reserve_pass,
            promoted_episode_count=len(promoted_ids),
            promoted_underlying_level1_pass=(promoted_underlying_level1_pass),
            contract_discovery_audit_count=len(discovery_rows),
            contract_discovery_complete=discovery_complete,
            primary_option_pair_available_count=selected_count,
            primary_option_pair_recording_pass=pair_recording_pass,
            graceful_degradation_pass=graceful_degradation_pass,
            cancellation_recovery_pass=cancellation_recovery_pass,
            m1c_universe_uninterrupted=universe_uninterrupted,
            recorder_reliability_pass=recorder_reliability_pass,
            no_order_guard_pass=no_order_guard_pass,
            orders_placed=0,
            critical_checks_pass=not missing_reasons,
            missing_reasons=missing_reasons,
        )

    def record_opening_reversal_underlying_outcome_v1(
        self,
        metadata: EvidenceMetadata,
        outcome: OpeningReversalUnderlyingOutcomeV1,
    ) -> int:
        self._validate(metadata)
        encoded = _json(outcome)
        with self.repository._connect() as connection:
            parent = connection.execute(
                """
                SELECT scientific_outcome_eligible_v1,
                       experiment_version, session_date
                FROM opening_reversal_prediction_v1
                WHERE receipt_hash_v1 = ?
                """,
                (outcome.prediction_receipt_hash_v1,),
            ).fetchone()
            if parent is None or not bool(parent["scientific_outcome_eligible_v1"]):
                raise ValueError("protected or engineering outcome rejected")
            if str(parent["experiment_version"]) == "1.1":
                barrier = connection.execute(
                    """
                    SELECT audit_json
                    FROM opening_reversal_causal_barrier_audit_v1_1
                    WHERE run_id = ? AND session_date = ?
                      AND barrier_status = 'passed'
                    """,
                    (
                        metadata.run_id,
                        str(parent["session_date"]),
                    ),
                ).fetchone()
                if barrier is None:
                    raise ValueError("V1.1 outcome lacks a passing causal barrier")
                audit = OpeningReversalCausalBarrierAuditV1_1.model_validate_json(
                    str(barrier["audit_json"])
                )
                if outcome.prediction_receipt_hash_v1 not in audit.prediction_receipt_hashes:
                    raise ValueError("V1.1 outcome prediction is outside causal barrier")
            existing = connection.execute(
                """
                SELECT id, outcome_receipt_hash_v1, outcome_json
                FROM opening_reversal_underlying_outcome_v1
                WHERE prediction_receipt_hash_v1 = ?
                """,
                (outcome.prediction_receipt_hash_v1,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["outcome_receipt_hash_v1"]) != outcome.outcome_receipt_hash_v1
                    or str(existing["outcome_json"]) != encoded
                ):
                    raise ValueError("opening reversal outcome is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_underlying_outcome_v1(
                    envelope_id, run_id, prediction_receipt_hash_v1,
                    opening_transition_event_id_v1, session_date, stock,
                    prediction_v1, r_15m, absolute_return_15m, threshold_15m,
                    outcome_state_v1, opening_reversal_aligned_return_v1,
                    correct_predicted_material_direction_v1,
                    accuracy_counting_no_move_as_failure_v1,
                    maximum_favourable_excursion_v1,
                    maximum_adverse_excursion_v1,
                    canonical_post_entry_local_range_share_v1,
                    iv_residual_v1, exceed_iv_v1, outcome_completeness_v1,
                    missing_reason_v1, outcome_created_at_utc,
                    outcome_receipt_hash_v1, outcome_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    outcome.prediction_receipt_hash_v1,
                    outcome.opening_transition_event_id_v1,
                    outcome.session.isoformat(),
                    outcome.stock,
                    outcome.prediction_v1,
                    outcome.r_15m,
                    outcome.absolute_return_15m,
                    outcome.threshold_15m,
                    outcome.outcome_state_v1,
                    outcome.opening_reversal_aligned_return_v1,
                    (
                        None
                        if outcome.correct_predicted_material_direction_v1 is None
                        else int(outcome.correct_predicted_material_direction_v1)
                    ),
                    (
                        None
                        if outcome.accuracy_counting_no_move_as_failure_v1 is None
                        else int(outcome.accuracy_counting_no_move_as_failure_v1)
                    ),
                    outcome.maximum_favourable_excursion_v1,
                    outcome.maximum_adverse_excursion_v1,
                    outcome.canonical_post_entry_local_range_share_v1,
                    outcome.iv_residual_v1,
                    (None if outcome.exceed_iv_v1 is None else int(outcome.exceed_iv_v1)),
                    outcome.outcome_completeness_v1,
                    outcome.missing_reason_v1,
                    outcome.outcome_created_at_utc.isoformat(),
                    outcome.outcome_receipt_hash_v1,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_opening_reversal_primary_option_outcome_v1(
        self,
        metadata: EvidenceMetadata,
        outcome: PrimaryOptionBidAskOutcomeV1,
    ) -> int:
        self._validate(metadata)
        encoded = _json(outcome)
        with self.repository._connect() as connection:
            parent = connection.execute(
                """
                SELECT prediction.prediction_v1, prediction.stock,
                       prediction.experiment_id,
                       prediction.experiment_version,
                       prediction.session_date,
                       prediction.fresh_episode_id,
                       prediction.scientific_outcome_eligible_v1,
                       promotion.promoted_receipt_hash_v1
                FROM opening_reversal_prediction_v1 prediction
                LEFT JOIN opening_reversal_promotion_v1 promotion
                  ON promotion.promoted_receipt_hash_v1 =
                     prediction.receipt_hash_v1
                 AND promotion.run_id = prediction.run_id
                WHERE prediction.run_id = ?
                  AND prediction.receipt_hash_v1 = ?
                """,
                (metadata.run_id, outcome.prediction_receipt_hash_v1),
            ).fetchone()
            if parent is not None and str(parent["experiment_version"]) == "1.1":
                eligible = connection.execute(
                    """
                    SELECT episode_id, session_date
                    FROM opening_reversal_v1_1_capture_eligible_episode
                    WHERE run_id = ? AND prediction_receipt_hash_v1 = ?
                    """,
                    (metadata.run_id, outcome.prediction_receipt_hash_v1),
                ).fetchone()
                discovery = connection.execute(
                    """
                    SELECT call_con_id, put_con_id, expiry, strike,
                           planned_live_market_data_lines, status
                    FROM opening_reversal_contract_discovery_v1
                    WHERE run_id = ? AND episode_id = ?
                    """,
                    (metadata.run_id, parent["fresh_episode_id"]),
                ).fetchone()
                selected_con_id = (
                    None
                    if discovery is None
                    else discovery["call_con_id" if outcome.contract.right == "C" else "put_con_id"]
                )
                if (
                    str(parent["experiment_id"]) != M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID
                    or eligible is None
                    or discovery is None
                    or str(discovery["status"]) != "selected"
                    or int(discovery["planned_live_market_data_lines"]) != 2
                    or selected_con_id != outcome.contract.con_id
                    or str(discovery["expiry"]) != outcome.contract.expiry.isoformat()
                    or float(discovery["strike"]) != outcome.contract.strike
                    or outcome.contract.expiry
                    != date.fromisoformat(str(parent["session_date"])) + timedelta(days=1)
                ):
                    raise ValueError("blocked_v1_1_outcome_not_from_eligible_primary_pair")
            expected_right = (
                "C" if parent is not None and str(parent["prediction_v1"]) == "CALL" else "P"
            )
            if outcome.role == "opposite_leg":
                expected_right = "P" if expected_right == "C" else "C"
            v1_1_capture = parent is not None and str(parent["experiment_version"]) == "1.1"
            if (
                parent is None
                or (not v1_1_capture and not bool(parent["scientific_outcome_eligible_v1"]))
                or parent["promoted_receipt_hash_v1"] is None
                or outcome.contract.underlying != str(parent["stock"])
                or outcome.contract.right != expected_right
            ):
                raise ValueError("primary option outcome is not linked to promoted prediction")
            other_leg = connection.execute(
                """
                SELECT expiry, strike, right
                FROM opening_reversal_primary_option_outcome_v1
                WHERE run_id = ? AND prediction_receipt_hash_v1 = ?
                  AND role <> ?
                """,
                (
                    metadata.run_id,
                    outcome.prediction_receipt_hash_v1,
                    outcome.role,
                ),
            ).fetchone()
            if other_leg is not None and (
                str(other_leg["expiry"]) != outcome.contract.expiry.isoformat()
                or float(other_leg["strike"]) != outcome.contract.strike
                or str(other_leg["right"]) == outcome.contract.right
            ):
                raise ValueError("primary option outcome pair is inconsistent")
            existing = connection.execute(
                """
                SELECT id, outcome_hash_v1, outcome_json
                FROM opening_reversal_primary_option_outcome_v1
                WHERE run_id = ? AND prediction_receipt_hash_v1 = ? AND role = ?
                """,
                (
                    metadata.run_id,
                    outcome.prediction_receipt_hash_v1,
                    outcome.role,
                ),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["outcome_hash_v1"]) != outcome.outcome_hash_v1
                    or str(existing["outcome_json"]) != encoded
                ):
                    raise ValueError("primary option outcome is immutable")
                return int(existing["id"])
            envelope_id = self.repository._insert_envelope(connection, metadata)
            entry = outcome.entry_quote
            exit_quote = outcome.exit_quote
            cursor = connection.execute(
                """
                INSERT INTO opening_reversal_primary_option_outcome_v1(
                    envelope_id, run_id, prediction_receipt_hash_v1, con_id,
                    right, role, expiry, strike, entry_bid, entry_ask,
                    entry_midpoint_diagnostic, entry_quote_timestamp_utc,
                    exit_bid, exit_ask, exit_midpoint_diagnostic,
                    exit_quote_timestamp_utc, entry_spread, exit_spread,
                    entry_quote_age_seconds, exit_quote_age_seconds,
                    entry_locked_or_crossed, exit_locked_or_crossed,
                    entry_stale, exit_stale, subscription_start_utc,
                    subscription_end_utc, capacity_line_owner,
                    conservative_return_v1, complete, missing_reason,
                    outcome_hash_v1, outcome_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    outcome.prediction_receipt_hash_v1,
                    outcome.contract.con_id,
                    outcome.contract.right,
                    outcome.role,
                    outcome.contract.expiry.isoformat(),
                    outcome.contract.strike,
                    entry.bid,
                    entry.ask,
                    outcome.entry_midpoint_diagnostic,
                    entry.timestamp_utc.isoformat(),
                    exit_quote.bid,
                    exit_quote.ask,
                    outcome.exit_midpoint_diagnostic,
                    exit_quote.timestamp_utc.isoformat(),
                    outcome.entry_spread,
                    outcome.exit_spread,
                    entry.quote_age_seconds,
                    exit_quote.quote_age_seconds,
                    int(entry.locked_or_crossed),
                    int(exit_quote.locked_or_crossed),
                    int(entry.stale),
                    int(exit_quote.stale),
                    outcome.subscription_start_utc.isoformat(),
                    outcome.subscription_end_utc.isoformat(),
                    outcome.capacity_line_owner,
                    outcome.conservative_return_v1,
                    int(outcome.complete),
                    outcome.missing_reason,
                    outcome.outcome_hash_v1,
                    encoded,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)


__all__ = ["FrozenRecorderRepository"]
