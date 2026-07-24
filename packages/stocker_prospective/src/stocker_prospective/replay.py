"""Deterministic end-to-end replay for the prospective evidence recorder."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    ScoreInput,
)
from stocker_prospective.market_data import MarketDataBudgetSnapshot, MarketDataType
from stocker_prospective.options import DteBucket, select_expiries
from stocker_prospective.shadow import (
    OptionExecutableQuote,
    ShadowStructureType,
    value_shadow_structures,
)
from stocker_prospective.signal import SignalEventizer
from stocker_prospective.universe import load_registered_universe

REPLAY_MODEL_ID = "synthetic-replay-bundle-v1"
REPLAY_FEATURE_SCHEMA_HASH = hashlib.sha256(b"synthetic-replay-schema-v1\n").hexdigest()
REPLAY_BLOCKERS = (
    "blocked_missing_verified_frozen_bundle",
    "blocked_feature_source_semantics_mismatch",
    "blocked_official_ibkr_api_not_installed",
)


class ReplaySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_path: Path
    run_id: str
    prospective_start_utc: datetime
    app_version: str
    git_commit: str
    universe_path: Path
    owner_id: str
    recorder_lease_stale_seconds: int = Field(gt=0)

    @field_validator("prospective_start_utc")
    @classmethod
    def _aware_start(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("prospective_start_utc must be timezone-aware")
        return value.astimezone(UTC)


class ReplayResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    fixture_id: str
    run_id: str
    score_label: str
    signal_episode_count: int
    signal_checkpoint_count: int
    surface_capture_count: int
    option_quote_count: int
    shadow_structure_count: int
    shadow_horizon_count: int
    capture_horizons_minutes: tuple[int, ...]
    blockers: tuple[str, ...]


def _fixture() -> dict[str, Any]:
    path = Path(__file__).with_name("fixtures") / "replay-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("fixture_version") != "1":
        raise RuntimeError("invalid deterministic replay fixture")
    return payload


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _metadata(
    settings: ReplaySettings,
    *,
    universe_id: str,
    recorded_at: datetime,
    source_timestamps: list[datetime],
) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=settings.run_id,
        prospective_start_utc=settings.prospective_start_utc,
        app_version=settings.app_version,
        git_commit=settings.git_commit,
        model_artifact_id=REPLAY_MODEL_ID,
        universe_id=universe_id,
        cohort="anchor_frozen_20",
        source_timestamps=[item.astimezone(UTC).isoformat() for item in source_timestamps],
        recorded_at_utc=recorded_at.astimezone(UTC),
    )


def _insert_enveloped(
    repository: ProspectiveRepository,
    *,
    metadata: EvidenceMetadata,
    existence_sql: str,
    existence_parameters: tuple[object, ...],
    insert_sql: str,
    insert_parameters: tuple[object, ...],
) -> int:
    """Insert an evidence row without creating orphan envelopes on restart."""

    with repository._connect() as connection:
        existing = connection.execute(existence_sql, existence_parameters).fetchone()
        if existing is not None:
            return int(existing["id"])
        envelope_id = repository._insert_envelope(connection, metadata)
        cursor = connection.execute(insert_sql, (envelope_id, *insert_parameters))
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)


def _record_universe(
    repository: ProspectiveRepository,
    *,
    run_id: str,
    universe_id: str,
    symbols: tuple[str, ...],
    recorded_at: datetime,
) -> None:
    with repository._connect() as connection:
        for symbol in symbols:
            active = symbol == "AAL"
            connection.execute(
                """
                INSERT OR IGNORE INTO universe_membership(
                    run_id, universe_id, cohort, symbol, operational_status,
                    rejection_reason, recorded_at_utc
                ) VALUES (?, ?, 'anchor_frozen_20', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    universe_id,
                    symbol,
                    "synthetic_replay_active" if active else "rejected_missing_replay_data",
                    None if active else "missing_completed_bar",
                    recorded_at.isoformat(),
                ),
            )


def _record_audit(
    repository: ProspectiveRepository,
    metadata: EvidenceMetadata,
    events: tuple[tuple[str, str, dict[str, object]], ...],
) -> None:
    with repository._connect() as connection:
        for sequence, (event_type, message, payload) in enumerate(events, start=1):
            exists = connection.execute(
                "SELECT id FROM audit_event WHERE run_id = ? AND sequence = ?",
                (metadata.run_id, sequence),
            ).fetchone()
            if exists is not None:
                continue
            envelope_id = repository._insert_envelope(connection, metadata)
            connection.execute(
                """
                INSERT INTO audit_event(
                    envelope_id, run_id, sequence, event_type, actor, message, payload_json
                ) VALUES (?, ?, ?, ?, 'deterministic-replay', ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    sequence,
                    event_type,
                    message,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )


def _record_health_and_connection_events(
    repository: ProspectiveRepository,
    metadata: EvidenceMetadata,
) -> None:
    with repository._connect() as connection:
        budget_exists = connection.execute(
            "SELECT 1 FROM market_data_budget_event WHERE run_id = ? LIMIT 1",
            (metadata.run_id,),
        ).fetchone()
    if budget_exists is None:
        repository.record_market_data_budget_event(
            metadata,
            MarketDataBudgetSnapshot(
                line_limit=100,
                reserved_headroom=10,
                usable_lines=90,
                active_lines=0,
                pending_requests=0,
                awaiting_cancellation=0,
                current_request_rate=0,
                waiting_signals=0,
                rejected_signals=1,
            ),
        )
    for blocker in REPLAY_BLOCKERS:
        _insert_enveloped(
            repository,
            metadata=metadata,
            existence_sql=(
                "SELECT id FROM data_health_event WHERE run_id = ? AND blocker_code = ?"
            ),
            existence_parameters=(metadata.run_id, blocker),
            insert_sql=(
                "INSERT INTO data_health_event("
                "envelope_id, run_id, severity, blocker_code, component, message, details_json"
                ") VALUES (?, ?, 'blocker', ?, 'scoring', ?, '{}')"
            ),
            insert_parameters=(metadata.run_id, blocker, blocker),
        )
    _insert_enveloped(
        repository,
        metadata=metadata,
        existence_sql=(
            "SELECT id FROM data_health_event WHERE run_id = ? "
            "AND blocker_code = 'blocked_market_data_budget_exhausted'"
        ),
        existence_parameters=(metadata.run_id,),
        insert_sql=(
            "INSERT INTO data_health_event("
            "envelope_id, run_id, severity, blocker_code, component, message, details_json"
            ") VALUES (?, ?, 'warning', 'blocked_market_data_budget_exhausted', "
            "'market_data_budget', 'synthetic bounded rejection', "
            "'{\"fixture\":true}')"
        ),
        insert_parameters=(metadata.run_id,),
    )
    connection_events = (
        ("disconnected", 1100, "connectivity_lost", 0),
        ("connected", 1102, "connectivity_restored_data_maintained", 1),
        ("disconnected", 1100, "connectivity_lost", 0),
        ("connected", 1101, "connectivity_restored_data_lost", 0),
    )
    for attempt, (state, code, message, maintained) in enumerate(connection_events, start=1):
        _insert_enveloped(
            repository,
            metadata=metadata,
            existence_sql=(
                "SELECT id FROM ibkr_connection_event WHERE run_id = ? AND reconnect_attempt = ?"
            ),
            existence_parameters=(metadata.run_id, attempt),
            insert_sql=(
                "INSERT INTO ibkr_connection_event("
                "envelope_id, run_id, state, error_code, message, data_maintained, "
                "reconnect_attempt, details_json) VALUES (?, ?, ?, ?, ?, ?, ?, '{}')"
            ),
            insert_parameters=(
                metadata.run_id,
                state,
                code,
                message,
                maintained,
                attempt,
            ),
        )


def _record_context(
    repository: ProspectiveRepository,
    metadata: EvidenceMetadata,
    *,
    session_date: date,
    previous_session: date,
) -> None:
    payload = {
        "synthetic": True,
        "label": "exact_previous_session_context_fixture",
        "observation_date": previous_session.isoformat(),
    }
    context_hash = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    _insert_enveloped(
        repository,
        metadata=metadata,
        existence_sql=(
            "SELECT id FROM previous_session_options_context "
            "WHERE run_id = ? AND current_session_date = ? AND context_hash = ?"
        ),
        existence_parameters=(metadata.run_id, session_date.isoformat(), context_hash),
        insert_sql=(
            "INSERT INTO previous_session_options_context("
            "envelope_id, run_id, current_session_date, required_previous_session, "
            "observation_date, provider_identity, source_record_identity_json, "
            "context_hash, schema_hash, feature_hash, completeness, freshness, "
            "eligibility, rejection_reason, payload_json"
            ") VALUES (?, ?, ?, ?, ?, 'synthetic-replay-provider', ?, ?, ?, ?, "
            "'complete', 'exact_previous_session', 1, NULL, ?)"
        ),
        insert_parameters=(
            metadata.run_id,
            session_date.isoformat(),
            previous_session.isoformat(),
            previous_session.isoformat(),
            json.dumps({"fixture_record": "previous-session-v1"}),
            context_hash,
            REPLAY_FEATURE_SCHEMA_HASH,
            REPLAY_FEATURE_SCHEMA_HASH,
            json.dumps(payload, sort_keys=True),
        ),
    )


def _record_bar_and_feature(
    repository: ProspectiveRepository,
    metadata: EvidenceMetadata,
    bar: dict[str, Any],
    *,
    con_id: int,
    threshold: float,
) -> None:
    end = _timestamp(str(bar["end"]))
    start = end - timedelta(minutes=5)
    _insert_enveloped(
        repository,
        metadata=metadata,
        existence_sql=(
            "SELECT id FROM underlying_bar WHERE run_id = ? AND symbol = ? AND bar_end_utc = ?"
        ),
        existence_parameters=(metadata.run_id, "AAL", end.isoformat()),
        insert_sql=(
            "INSERT INTO underlying_bar("
            "envelope_id, run_id, symbol, con_id, bar_start_utc, bar_end_utc, "
            "session_date, open, high, low, close, activity_value, "
            "activity_semantic_label, bar_source, source_timestamp_utc, "
            "receive_timestamp_utc, completeness, feature_as_of_utc, "
            "m0_probability, m1_probability, frozen_threshold, model_bundle_id, "
            "feature_schema_hash, eligibility, rejection_reason"
            ") VALUES (?, ?, 'AAL', ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "'synthetic_replay_activity_not_eodhd_proxy', 'deterministic_replay', "
            "?, ?, 'complete', ?, ?, ?, ?, ?, ?, 1, NULL)"
        ),
        insert_parameters=(
            metadata.run_id,
            con_id,
            start.isoformat(),
            end.isoformat(),
            end.date().isoformat(),
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
            float(bar["activity"]),
            end.isoformat(),
            metadata.recorded_at_utc.isoformat(),
            end.isoformat(),
            float(bar["m0"]),
            float(bar["m1"]),
            threshold,
            REPLAY_MODEL_ID,
            REPLAY_FEATURE_SCHEMA_HASH,
        ),
    )
    feature_values = {
        "synthetic_completed_bar": True,
        "synthetic_activity": bar["activity"],
        "not_frozen_h0": True,
    }
    _insert_enveloped(
        repository,
        metadata=metadata,
        existence_sql=(
            "SELECT id FROM feature_snapshot WHERE run_id = ? AND symbol = 'AAL' "
            "AND bar_end_utc = ? AND feature_schema_hash = ?"
        ),
        existence_parameters=(metadata.run_id, end.isoformat(), REPLAY_FEATURE_SCHEMA_HASH),
        insert_sql=(
            "INSERT INTO feature_snapshot("
            "envelope_id, run_id, symbol, bar_end_utc, feature_as_of_utc, "
            "feature_schema_hash, feature_values_json, parity_status, eligibility, "
            "rejection_reason) VALUES (?, ?, 'AAL', ?, ?, ?, ?, "
            "'synthetic_replay_only', 1, NULL)"
        ),
        insert_parameters=(
            metadata.run_id,
            end.isoformat(),
            end.isoformat(),
            REPLAY_FEATURE_SCHEMA_HASH,
            json.dumps(feature_values, sort_keys=True),
        ),
    )


def _record_contracts(
    repository: ProspectiveRepository,
    metadata: EvidenceMetadata,
    *,
    underlying_con_id: int,
    expiry_by_bucket: dict[DteBucket, date],
    strikes: tuple[float, ...],
) -> dict[tuple[DteBucket, str, float], tuple[int, int]]:
    result: dict[tuple[DteBucket, str, float], tuple[int, int]] = {}
    for bucket, expiry in expiry_by_bucket.items():
        for strike_index, strike in enumerate(strikes):
            for right_index, right in enumerate(("C", "P")):
                con_id = (
                    underlying_con_id * 1000
                    + list(DteBucket).index(bucket) * 100
                    + strike_index * 2
                    + right_index
                    + 1
                )
                local_symbol = f"AAL {expiry.strftime('%y%m%d')}{right}{int(strike * 1000):08d}"
                database_id = _insert_enveloped(
                    repository,
                    metadata=metadata,
                    existence_sql=(
                        "SELECT id FROM option_contract WHERE run_id = ? AND con_id = ?"
                    ),
                    existence_parameters=(metadata.run_id, con_id),
                    insert_sql=(
                        "INSERT INTO option_contract("
                        "envelope_id, run_id, underlying_con_id, con_id, local_symbol, "
                        "expiry, strike, right, multiplier, exchange, trading_class, "
                        "dte_bucket, qualification_status, rejection_reason"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, '100', 'SMART', 'AAL', ?, "
                        "'qualified_exact', NULL)"
                    ),
                    insert_parameters=(
                        metadata.run_id,
                        underlying_con_id,
                        con_id,
                        local_symbol,
                        expiry.isoformat(),
                        strike,
                        right,
                        bucket.value,
                    ),
                )
                result[(bucket, right, strike)] = (database_id, con_id)
    return result


def _surface_quote(
    *,
    con_id: int,
    right: str,
    strike: float,
    timestamp: datetime,
    horizon: int,
    signal_index: int,
    market_data_type: MarketDataType,
    stale: bool,
) -> OptionExecutableQuote:
    time_move = horizon * 0.012
    signal_move = signal_index * 0.03
    if right == "C":
        base_ask = {11.0: 1.45, 12.0: 0.80, 13.0: 0.30}[strike]
        ask = base_ask + time_move + signal_move
    else:
        base_ask = {11.0: 0.25, 12.0: 0.75, 13.0: 1.40}[strike]
        ask = max(0.05, base_ask - time_move / 2 + signal_move)
    bid = max(0.01, ask - 0.05)
    provider_timestamp = timestamp + timedelta(seconds=2)
    return OptionExecutableQuote(
        contract_id=con_id,
        right=right,  # type: ignore[arg-type]
        strike=strike,
        bid=round(bid, 4),
        ask=round(ask, 4),
        bid_size=10.0 + horizon,
        ask_size=12.0 + horizon,
        provider_timestamp=provider_timestamp,
        receive_timestamp=provider_timestamp + timedelta(milliseconds=100),
        market_data_type=market_data_type,
        multiplier=100,
        stale=stale,
    )


def _record_capture(
    repository: ProspectiveRepository,
    metadata: EvidenceMetadata,
    *,
    episode_id: str,
    bucket: DteBucket,
    target: datetime,
    horizon: int,
    signal_index: int,
    contract_ids: dict[tuple[DteBucket, str, float], tuple[int, int]],
    missing_expiry: bool,
) -> dict[tuple[str, float], OptionExecutableQuote]:
    delayed = bucket is DteBucket.ZERO_DTE and horizon == 10
    stale = bucket is DteBucket.ZERO_DTE and horizon == 15
    market_type = MarketDataType.DELAYED if delayed else MarketDataType.LIVE
    if missing_expiry:
        status = "missed"
        freshness = "missing"
        completeness = "missing"
        actual: datetime | None = None
        missing_contract_reason: str | None = "no_expiry_in_bucket"
    else:
        status = "diagnostic_only" if delayed or stale else "captured"
        freshness = "stale" if stale else "fresh"
        completeness = "complete"
        actual = target + timedelta(seconds=2)
        missing_contract_reason = None
    capture_id = _insert_enveloped(
        repository,
        metadata=metadata,
        existence_sql=(
            "SELECT id FROM option_surface_capture WHERE signal_episode_id = ? "
            "AND dte_bucket = ? AND target_timestamp_utc = ?"
        ),
        existence_parameters=(episode_id, bucket.value, target.isoformat()),
        insert_sql=(
            "INSERT INTO option_surface_capture("
            "envelope_id, run_id, signal_episode_id, dte_bucket, target_timestamp_utc, "
            "actual_quote_timestamp_utc, capture_lag_seconds, market_data_type, "
            "quote_freshness, completeness, connection_status, budget_status, "
            "missing_contract_reason, missing_quote_reason, subscription_error, "
            "capture_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'connected', "
            "'within_budget', ?, NULL, NULL, ?)"
        ),
        insert_parameters=(
            metadata.run_id,
            episode_id,
            bucket.value,
            target.isoformat(),
            None if actual is None else actual.isoformat(),
            None if actual is None else 2.0,
            None if missing_expiry else market_type.value,
            freshness,
            completeness,
            missing_contract_reason,
            status,
        ),
    )
    if missing_expiry:
        return {}
    quotes: dict[tuple[str, float], OptionExecutableQuote] = {}
    for right in ("C", "P"):
        for strike in (11.0, 12.0, 13.0):
            database_contract_id, con_id = contract_ids[(bucket, right, strike)]
            quote = _surface_quote(
                con_id=con_id,
                right=right,
                strike=strike,
                timestamp=target,
                horizon=horizon,
                signal_index=signal_index,
                market_data_type=market_type,
                stale=stale,
            )
            quotes[(right, strike)] = quote
            quote_id = _insert_enveloped(
                repository,
                metadata=metadata,
                existence_sql=(
                    "SELECT id FROM option_quote WHERE surface_capture_id = ? "
                    "AND option_contract_id = ?"
                ),
                existence_parameters=(capture_id, database_contract_id),
                insert_sql=(
                    "INSERT INTO option_quote("
                    "envelope_id, run_id, surface_capture_id, option_contract_id, "
                    "bid, ask, bid_size, ask_size, last, last_size, volume, open_interest, "
                    "bid_implied_volatility, ask_implied_volatility, last_implied_volatility, "
                    "model_implied_volatility, bid_delta, ask_delta, last_delta, model_delta, "
                    "provider_timestamp_utc, receive_timestamp_utc, market_data_type, "
                    "staleness_seconds, completeness, permission_error"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, "
                    "0.30, 0.32, NULL, 0.31, NULL, NULL, NULL, 0.50, ?, ?, ?, ?, "
                    "'complete', NULL)"
                ),
                insert_parameters=(
                    metadata.run_id,
                    capture_id,
                    database_contract_id,
                    quote.bid,
                    quote.ask,
                    quote.bid_size,
                    quote.ask_size,
                    quote.provider_timestamp.isoformat(),
                    quote.receive_timestamp.isoformat(),
                    quote.market_data_type.value,
                    120.0 if stale else 0.1,
                ),
            )
            for computation_source, implied_volatility, delta, gamma, theta, vega in (
                ("bid", 0.30, None, None, None, None),
                ("ask", 0.32, None, None, None, None),
                ("model", 0.31, 0.50, 0.04, -0.02, 0.08),
            ):
                _insert_enveloped(
                    repository,
                    metadata=metadata,
                    existence_sql=(
                        "SELECT id FROM option_quote_computation "
                        "WHERE option_quote_id = ? AND computation_source = ?"
                    ),
                    existence_parameters=(quote_id, computation_source),
                    insert_sql=(
                        "INSERT INTO option_quote_computation("
                        "envelope_id, run_id, option_quote_id, computation_source, "
                        "implied_volatility, delta, gamma, theta, vega, option_price, "
                        "present_value_dividend, underlying_reference_price, "
                        "provider_timestamp_utc, receive_timestamp_utc, market_data_type, "
                        "completeness) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, "
                        "12.0, ?, ?, ?, 'complete')"
                    ),
                    insert_parameters=(
                        metadata.run_id,
                        quote_id,
                        computation_source,
                        implied_volatility,
                        delta,
                        gamma,
                        theta,
                        vega,
                        quote.provider_timestamp.isoformat(),
                        quote.receive_timestamp.isoformat(),
                        quote.market_data_type.value,
                    ),
                )
    return quotes


def _record_underlying_quote(
    repository: ProspectiveRepository,
    metadata: EvidenceMetadata,
    *,
    episode_id: str,
    target: datetime,
    horizon: int,
    signal_index: int,
) -> None:
    bid = round(12.0 + signal_index * 0.1 + horizon * 0.005, 4)
    ask = round(bid + 0.02, 4)
    actual = target + timedelta(seconds=1)
    _insert_enveloped(
        repository,
        metadata=metadata,
        existence_sql=(
            "SELECT id FROM underlying_quote WHERE run_id = ? "
            "AND signal_episode_id = ? AND target_timestamp_utc = ?"
        ),
        existence_parameters=(metadata.run_id, episode_id, target.isoformat()),
        insert_sql=(
            "INSERT INTO underlying_quote("
            "envelope_id, run_id, signal_episode_id, target_timestamp_utc, "
            "actual_quote_timestamp_utc, capture_lag_seconds, bid, ask, bid_size, "
            "ask_size, last, last_size, midpoint, spread, provider_timestamp_utc, "
            "receive_timestamp_utc, market_data_type, freshness, completeness, "
            "capture_status, missing_quote_reason"
            ") VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, 100, 120, NULL, NULL, ?, ?, ?, ?, "
            "'live', 'fresh', 'complete', 'captured', NULL)"
        ),
        insert_parameters=(
            metadata.run_id,
            episode_id,
            target.isoformat(),
            actual.isoformat(),
            bid,
            ask,
            round((bid + ask) / 2, 4),
            round(ask - bid, 4),
            actual.isoformat(),
            (actual + timedelta(milliseconds=100)).isoformat(),
        ),
    )


def _record_shadow(
    repository: ProspectiveRepository,
    metadata: EvidenceMetadata,
    *,
    episode_id: str,
    signal_index: int,
    crossing: datetime,
    live_surfaces: dict[int, dict[tuple[str, float], OptionExecutableQuote]],
    contract_ids: dict[tuple[DteBucket, str, float], tuple[int, int]],
) -> None:
    entry = live_surfaces[0]
    valuations_by_horizon = {
        horizon: value_shadow_structures(
            entry_quotes=entry,
            exit_quotes=live_surfaces[horizon],
            atm_strike=12.0,
            lower_strike=11.0,
            upper_strike=13.0,
            target_timestamp=crossing + timedelta(minutes=horizon),
            maximum_capture_lag=timedelta(seconds=15),
            estimated_fee_per_contract=0.65,
        )
        for horizon in (5, 10, 15, 30)
    }
    for structure_index, structure_type in enumerate(ShadowStructureType):
        first = valuations_by_horizon[5][structure_index]
        structure_key = f"{episode_id}|3_TO_5_DTE|{structure_type.value}"
        structure_id = "shadow-" + hashlib.sha256(structure_key.encode()).hexdigest()[:24]
        with repository._connect() as connection:
            exists = connection.execute(
                "SELECT id FROM shadow_structure WHERE id = ?",
                (structure_id,),
            ).fetchone()
            if exists is None:
                envelope_id = repository._insert_envelope(connection, metadata)
                connection.execute(
                    """
                    INSERT INTO shadow_structure(
                        id, envelope_id, run_id, signal_episode_id, cohort, symbol,
                        dte_bucket, structure_type, entry_debit, multiplier,
                        estimated_fees, spread_quality, completeness, rejection_reason,
                        quoted_research_ledger
                    ) VALUES (?, ?, ?, ?, 'anchor_frozen_20', 'AAL', '3_TO_5_DTE',
                              ?, ?, 100, ?, ?, ?, ?, 1)
                    """,
                    (
                        structure_id,
                        envelope_id,
                        metadata.run_id,
                        episode_id,
                        structure_type.value,
                        first.entry_debit,
                        first.estimated_fees or 0.0,
                        first.spread_quality,
                        "complete" if first.rejection_reason is None else "rejected",
                        first.rejection_reason,
                    ),
                )
        for leg in first.legs:
            database_contract_id = contract_ids[
                (DteBucket.THREE_TO_FIVE_DTE, leg.right, leg.strike)
            ][0]
            _insert_enveloped(
                repository,
                metadata=metadata,
                existence_sql=(
                    "SELECT id FROM shadow_leg WHERE shadow_structure_id = ? "
                    "AND option_contract_id = ? AND leg_role = ?"
                ),
                existence_parameters=(structure_id, database_contract_id, leg.side),
                insert_sql=(
                    "INSERT INTO shadow_leg("
                    "envelope_id, run_id, shadow_structure_id, option_contract_id, "
                    "leg_role, quantity, entry_side, entry_price, quote_timestamp_utc"
                    ") VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)"
                ),
                insert_parameters=(
                    metadata.run_id,
                    structure_id,
                    database_contract_id,
                    leg.side,
                    "ask" if leg.side == "long" else "bid",
                    leg.entry_executable_price,
                    leg.entry_quote_timestamp.isoformat(),
                ),
            )
        for horizon, valuations in valuations_by_horizon.items():
            valuation = valuations[structure_index]
            _insert_enveloped(
                repository,
                metadata=metadata,
                existence_sql=(
                    "SELECT id FROM shadow_horizon_valuation "
                    "WHERE shadow_structure_id = ? AND horizon_minutes = ?"
                ),
                existence_parameters=(structure_id, horizon),
                insert_sql=(
                    "INSERT INTO shadow_horizon_valuation("
                    "envelope_id, run_id, shadow_structure_id, horizon_minutes, "
                    "target_timestamp_utc, actual_quote_timestamp_utc, "
                    "capture_lag_seconds, exit_credit, gross_return_on_debit, gross_pnl, "
                    "estimated_fees, market_data_type, completeness, rejection_reason"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                insert_parameters=(
                    metadata.run_id,
                    structure_id,
                    horizon,
                    valuation.target_timestamp.isoformat(),
                    (
                        None
                        if valuation.actual_quote_timestamp is None
                        else valuation.actual_quote_timestamp.isoformat()
                    ),
                    valuation.capture_lag_seconds,
                    valuation.exit_credit,
                    valuation.gross_return_on_debit,
                    valuation.gross_pnl,
                    valuation.estimated_fees or 0.0,
                    (
                        None
                        if valuation.market_data_type is None
                        else valuation.market_data_type.value
                    ),
                    "complete" if valuation.rejection_reason is None else "rejected",
                    valuation.rejection_reason,
                ),
            )


def _count(connection: sqlite3.Connection, table: str) -> int:
    allowed = {
        "signal_episode",
        "signal_checkpoint",
        "option_surface_capture",
        "option_quote",
        "shadow_structure",
        "shadow_horizon_valuation",
    }
    if table not in allowed:
        raise ValueError(table)
    row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def run_deterministic_replay(settings: ReplaySettings) -> ReplayResult:
    """Populate a persistent database from a fixed, explicitly synthetic fixture."""

    fixture = _fixture()
    bars = list(fixture["bars"])
    first_bar_end = _timestamp(str(bars[0]["end"]))
    if settings.prospective_start_utc > first_bar_end:
        raise ValueError("prospective_start_utc is after the first replay evidence timestamp")
    universe = load_registered_universe(settings.universe_path)
    session_date = date.fromisoformat(str(fixture["session_date"]))
    previous_session = date.fromisoformat(str(fixture["required_previous_session"]))
    repository = ProspectiveRepository(settings.database_path)
    repository.migrate()
    run_metadata = _metadata(
        settings,
        universe_id=universe.universe_id,
        recorded_at=first_bar_end,
        source_timestamps=[first_bar_end],
    )
    repository.create_run(run_metadata, mode="shadow")
    repository.acquire_recorder_lease(
        run_id=settings.run_id,
        owner_id=settings.owner_id,
        # Lease time is operational wall time, not synthetic evidence time.
        # This keeps stale-owner recovery meaningful for a continuous replay
        # service while all research records remain fixture-deterministic.
        now=datetime.now(UTC),
        stale_after=timedelta(seconds=settings.recorder_lease_stale_seconds),
    )

    def heartbeat() -> None:
        repository.heartbeat_recorder_lease(
            run_id=settings.run_id,
            owner_id=settings.owner_id,
            now=datetime.now(UTC),
        )

    _record_universe(
        repository,
        run_id=settings.run_id,
        universe_id=universe.universe_id,
        symbols=universe.symbols,
        recorded_at=first_bar_end,
    )
    heartbeat()
    with repository._connect() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO runtime_session(
                run_id, session_date, opened_at_utc, status
            ) VALUES (?, ?, ?, 'replay_complete')
            """,
            (settings.run_id, session_date.isoformat(), first_bar_end.isoformat()),
        )

    _record_audit(
        repository,
        run_metadata,
        (
            (
                "runtime_started",
                "Deterministic replay recorder started",
                {"mode": "shadow", "orders": False},
            ),
            (
                "scientific_boundary",
                "Replay scores are synthetic and are not frozen M1",
                {"score_label": fixture["score_label"]},
            ),
            (
                "recorder_lease_acquired",
                "Single recorder lease acquired",
                {"owner_id": settings.owner_id},
            ),
            (
                "replay_completed",
                "All deterministic capture horizons processed",
                {"horizons": fixture["capture_horizons_minutes"]},
            ),
        ),
    )
    _record_health_and_connection_events(repository, run_metadata)
    _record_context(
        repository,
        run_metadata,
        session_date=session_date,
        previous_session=previous_session,
    )
    heartbeat()

    con_id = int(fixture["underlying_contract_id"])
    _insert_enveloped(
        repository,
        metadata=run_metadata,
        existence_sql=("SELECT id FROM underlying_contract WHERE run_id = ? AND symbol = 'AAL'"),
        existence_parameters=(settings.run_id,),
        insert_sql=(
            "INSERT INTO underlying_contract("
            "envelope_id, run_id, symbol, con_id, exchange, currency, local_symbol, "
            "qualification_status, rejection_reason"
            ") VALUES (?, ?, 'AAL', ?, 'SMART', 'USD', 'AAL', 'qualified_exact', NULL)"
        ),
        insert_parameters=(settings.run_id, con_id),
    )

    threshold = float(fixture["synthetic_threshold"])
    eventizer = SignalEventizer(repository)
    episodes: list[tuple[str, datetime]] = []
    for bar in bars:
        heartbeat()
        end = _timestamp(str(bar["end"]))
        metadata = _metadata(
            settings,
            universe_id=universe.universe_id,
            recorded_at=end + timedelta(seconds=1),
            source_timestamps=[end],
        )
        _record_bar_and_feature(
            repository,
            metadata,
            bar,
            con_id=con_id,
            threshold=threshold,
        )
        event_result = eventizer.record(
            ScoreInput(
                metadata=metadata,
                symbol="AAL",
                bar_end_utc=end,
                session_date=session_date,
                feature_as_of_utc=end,
                m0_probability=float(bar["m0"]),
                m1_probability=float(bar["m1"]),
                frozen_threshold=threshold,
                feature_schema_hash=REPLAY_FEATURE_SCHEMA_HASH,
                eligibility=True,
                rejection_reason=None,
                score_label=str(fixture["score_label"]),
            )
        )
        if event_result.status == "crossing" and event_result.episode_id is not None:
            episodes.append((event_result.episode_id, end))

    expiry_selections = select_expiries(
        session_date,
        [date.fromisoformat(value) for value in fixture["available_expiries"]],
    )
    expiry_by_bucket = {
        bucket: selection.expiry
        for bucket, selection in expiry_selections.items()
        if selection.expiry is not None
    }
    contract_ids = _record_contracts(
        repository,
        run_metadata,
        underlying_con_id=con_id,
        expiry_by_bucket=expiry_by_bucket,
        strikes=tuple(float(value) for value in fixture["strikes"]),
    )
    heartbeat()
    horizons = tuple(int(value) for value in fixture["capture_horizons_minutes"])
    for signal_index, (episode_id, crossing) in enumerate(episodes):
        live_surfaces: dict[int, dict[tuple[str, float], OptionExecutableQuote]] = {}
        for horizon in horizons:
            heartbeat()
            target = crossing + timedelta(minutes=horizon)
            capture_metadata = _metadata(
                settings,
                universe_id=universe.universe_id,
                recorded_at=target + timedelta(seconds=3),
                source_timestamps=[target + timedelta(seconds=2)],
            )
            _record_underlying_quote(
                repository,
                capture_metadata,
                episode_id=episode_id,
                target=target,
                horizon=horizon,
                signal_index=signal_index,
            )
            for bucket in DteBucket:
                quotes = _record_capture(
                    repository,
                    capture_metadata,
                    episode_id=episode_id,
                    bucket=bucket,
                    target=target,
                    horizon=horizon,
                    signal_index=signal_index,
                    contract_ids=contract_ids,
                    missing_expiry=expiry_selections[bucket].expiry is None,
                )
                if bucket is DteBucket.THREE_TO_FIVE_DTE:
                    live_surfaces[horizon] = quotes
        _record_shadow(
            repository,
            run_metadata,
            episode_id=episode_id,
            signal_index=signal_index,
            crossing=crossing,
            live_surfaces=live_surfaces,
            contract_ids=contract_ids,
        )

    with repository._connect() as connection:
        replay_result = ReplayResult(
            fixture_id=str(fixture["fixture_id"]),
            run_id=settings.run_id,
            score_label=str(fixture["score_label"]),
            signal_episode_count=_count(connection, "signal_episode"),
            signal_checkpoint_count=_count(connection, "signal_checkpoint"),
            surface_capture_count=_count(connection, "option_surface_capture"),
            option_quote_count=_count(connection, "option_quote"),
            shadow_structure_count=_count(connection, "shadow_structure"),
            shadow_horizon_count=_count(connection, "shadow_horizon_valuation"),
            capture_horizons_minutes=horizons,
            blockers=REPLAY_BLOCKERS,
        )
    return replay_result
