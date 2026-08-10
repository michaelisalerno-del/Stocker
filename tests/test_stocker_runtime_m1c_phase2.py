from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import stocker_ideas.plugins.frozen_m1c_signal_v0 as m1c_plugin
from stocker_ideas.plugins.frozen_m1c_signal_v0 import (
    COHORT,
    UNIVERSE,
    FrozenM1CSignalV0,
)
from stocker_ideas.plugins.frozen_m1c_v0 import (
    CAUSAL_GROUP_I_FEATURES,
    REQUIRED_GROUP_O_FEATURES,
    build_direction_features,
    build_front_options_context,
    build_group_i_for_symbol,
    classify_directions,
    score_m1c,
)
from stocker_research.daily_soft_regimes_v0 import FrozenDimensionParameters, RobustValueScale
from stocker_research.front_options_soft_regimes_v01 import (
    apply_front_options_dimensions,
    apply_serialized_diag_regime,
)
from stocker_research.legacy_prospective.direction import FrozenDirectionRuntime
from stocker_research.legacy_prospective.direction_features import (
    DirectionFeatureBar,
    FrozenDirectionFeatureBuilder,
)
from stocker_research.legacy_prospective.m1c_features import (
    LiveFeatureBar,
    M1CCausalFeatureBuilder,
)
from stocker_runtime.domain import (
    IdeaOutput,
    JsonValue,
    MarketEvent,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)
from stocker_runtime.ideas.contract import (
    AncestorPageContinuation,
    DiscoveryReceipt,
    ExactEventsContinuation,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    MarketDataRequirement,
)
from stocker_runtime.ideas.discovery import IdeaConfig, discover_plugins, reviewed_code_hash
from stocker_runtime.ideas.identity import deterministic_idea_output_id
from stocker_runtime.ingestion.session_projection import project_required_session_receipts
from stocker_runtime.ingestion.snapshot_projection import project_option_snapshot_captures
from stocker_runtime.storage import connect_v2, initialize_database

ROOT = Path(__file__).resolve().parents[1]
ARCHETYPE_ROOT = (
    ROOT / "research/directional-readiness/20260726-stock-local-directional-archetypes-v0/"
    "artifacts/primary"
)
FRONT_OPTIONS_ROOT = (
    ROOT / "research/cross-market-context/20260723-daily-stock-front-options-context-v01/"
    "artifacts/primary"
)


def _seed(database: Path) -> None:
    initialize_database(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES "
            "('run-1', 'shadow', 'ibkr', 1, NULL, ?, 'fixture', "
            "'shadow_protected', 'running', NULL)",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-1', 1, 'fixture', 1)"
        )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, "
            "currency) VALUES ('AAL', ?, 'stock', 'AAL', 'SMART', 'USD')",
            (hashlib.sha256(b"AAL").hexdigest(),),
        )


def _sessions(count: int) -> tuple[date, ...]:
    result: list[date] = []
    current = date(2026, 6, 1)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return tuple(result)


def _insert_session(
    database: Path,
    session: date,
    *,
    first_sequence: int,
    volume_multiplier: float = 1.0,
    incomplete_bar: int | None = None,
    zero_volume_bar: int | None = None,
    instrument_id: str = "AAL",
    base_price: float = 100.0,
) -> int:
    opening = datetime.combine(session, time(13, 30), tzinfo=UTC)
    with connect_v2(database) as connection:
        for bar_number in range(1, 79):
            sequence = first_sequence + bar_number - 1
            end_at_us = int((opening + timedelta(minutes=5 * bar_number)).timestamp() * 1_000_000)
            complete = incomplete_bar != bar_number
            payload: dict[str, JsonValue] = {
                "bar_number": bar_number,
                "bar_start_at_us": end_at_us - 300_000_000,
                "bar_end_at_us": end_at_us,
                "session": session.isoformat(),
                "source_completeness": "complete" if complete else "incomplete",
                "first_source_sequence": sequence,
                "derived_after_source_sequence": sequence,
                "expected_input_count": 60,
                "input_count": 60 if complete else 59,
            }
            if complete:
                opening_value = base_price + bar_number / 100.0
                payload.update(
                    {
                        "open": opening_value,
                        "high": opening_value + 1.0,
                        "low": opening_value - 1.0,
                        "close": opening_value + 0.25,
                        "volume": (
                            0.0
                            if zero_volume_bar == bar_number
                            else volume_multiplier * float(bar_number)
                        ),
                    }
                )
            payload_json = canonical_json_bytes(payload).decode()
            event_id = hashlib.sha256(
                f"{instrument_id}|{session.isoformat()}|{bar_number}|{sequence}".encode()
            ).hexdigest()
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, "
                "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
                "event_at_us, received_at_us, connection_generation, open_value, high_value, "
                "low_value, close_value, volume_value, payload_json, payload_sha256) "
                "VALUES (?, 'run-1', NULL, ?, ?, 'bars', 'bar_5m', ?, ?, 1, "
                "?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    sequence,
                    instrument_id,
                    end_at_us,
                    end_at_us,
                    payload.get("open"),
                    payload.get("high"),
                    payload.get("low"),
                    payload.get("close"),
                    payload.get("volume"),
                    payload_json,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                ),
            )
    return first_sequence + 78


def _requirement(instrument_id: str = "AAL") -> MarketDataRequirement:
    return MarketDataRequirement(
        instrument_id=instrument_id,
        feed_kind="bars",
        event_kind="bar_5m_session_prefix",
        cadence="5m",
        gaps_block=True,
        staleness_block=True,
    )


def _prefix_fixture(*, session: str, checkpoint: int, base: float = 100.0) -> dict[str, JsonValue]:
    first = max(1, checkpoint - 11)
    trailing: list[dict[str, JsonValue]] = []
    for number in range(first, checkpoint + 1):
        opening = base + number / 10.0
        trailing.append(
            {
                "bar_number": number,
                "event_at_us": 1_000_000 * number,
                "open": opening,
                "high": opening + 1.0,
                "low": opening - 1.0,
                "close": opening + 0.2,
                "volume": 1_000.0 + number,
                "historical_relative_activity": 1.2,
                "return_bps": 10.0,
                "true_range_bps": 200.0,
                "upper_wick_fraction": 0.4,
                "lower_wick_fraction": 0.4,
                "session_vwap": base + number / 20.0,
            }
        )
    return {
        "schema_version": 1,
        "session": session,
        "bar_number": checkpoint,
        "source_completeness": "complete",
        "historical_session_count": 21,
        "historical_relative_activity": 1.2,
        "activity_ready": True,
        "trailing_bars": trailing,
        "session_volumes": [1_000.0 + number for number in range(1, checkpoint + 1)],
        "accumulator": {
            "bar_count": checkpoint,
            "session_open": base,
            "session_high": base + checkpoint / 10.0 + 1.0,
            "session_low": base - 0.9,
            "last_close": base + checkpoint / 10.0 + 0.2,
            "activity_sum": checkpoint * 1.2,
            "range_sum": checkpoint * 200.0,
            "travel_sum": checkpoint * 10.0,
            "return_sum": checkpoint * 10.0,
            "width_sum": checkpoint * 2.0,
            "positive_return_count": checkpoint,
            "negative_return_count": 0,
            "zero_return_count": 0,
        },
    }


def _event(
    event_id: str,
    instrument_id: str,
    event_kind: str,
    at_us: int,
    payload: Mapping[str, JsonValue],
) -> MarketEvent:
    return MarketEvent(
        event_id=event_id,
        instrument_id=instrument_id,
        feed_kind="bars" if event_kind != "option_snapshot_capture" else "quotes",
        event_kind=event_kind,
        event_at_us=at_us,
        received_at_us=at_us,
        payload=payload,
    )


def _exact_continuation_fixture(
    *,
    state: JsonValue,
    session: str,
    checkpoints: tuple[int, ...],
    prior_ids: tuple[str, ...],
    event_overrides: Mapping[tuple[str, int], MarketEvent] | None = None,
) -> tuple[
    JsonValue,
    tuple[str, ...],
    Mapping[int, tuple[MarketEvent, ...]],
    Mapping[int, ExactEventsContinuation],
]:
    events_by_checkpoint: dict[int, tuple[MarketEvent, ...]] = {}
    indexes: dict[str, dict[str, str]] = {}
    overrides = {} if event_overrides is None else dict(event_overrides)
    for checkpoint in checkpoints:
        values = {
            symbol: overrides.get(
                (symbol, checkpoint),
                _event(
                    f"exact-{symbol}-{checkpoint}",
                    symbol,
                    "bar_5m_session_prefix",
                    checkpoint * 1_000_000 + sorted(UNIVERSE).index(symbol),
                    _prefix_fixture(
                        session=session,
                        checkpoint=checkpoint,
                        base=250.0 if symbol == "VTI" else 100.0,
                    ),
                ),
            )
            for symbol in UNIVERSE
        }
        events_by_checkpoint[checkpoint] = tuple(values[symbol] for symbol in sorted(UNIVERSE))
        indexes[f"{session}|{checkpoint:02d}"] = {
            symbol: values[symbol].event_id for symbol in UNIVERSE
        }
    roots = {symbol: f"root-{symbol}" for symbol in UNIVERSE}
    state_value = cast(dict[str, JsonValue], json.loads(canonical_json_bytes(state)))
    state_value.update(
        {
            "schema_version": 2,
            "prefix_roots": {
                symbol: {
                    "i": roots[symbol],
                    "s": session,
                    "n": 78,
                    "t": 78_000_000,
                }
                for symbol in UNIVERSE
            },
            "prefix_index": indexes,
            "processed": state_value.get("processed", {}),
            "statuses": {
                **{
                    symbol: {"s": session, "r": "prior_session_baseline_missing"}
                    for symbol in COHORT
                    if symbol != "AAL"
                },
                **cast(dict[str, JsonValue], state_value.get("statuses", {})),
            },
        }
    )
    root_ids = tuple(roots[symbol] for symbol in UNIVERSE)
    retained = (*prior_ids, *root_ids)
    requests = {
        checkpoint: ExactEventsContinuation(
            root_event_ids=root_ids,
            event_ids=tuple(item.event_id for item in events_by_checkpoint[checkpoint]),
            event_kind="bar_5m_session_prefix",
            input_roles=("prior_receipt",),
        )
        for checkpoint in checkpoints
    }
    return cast(JsonValue, state_value), retained, events_by_checkpoint, requests


def _exact_batch(
    *,
    events: tuple[MarketEvent, ...],
    request: ExactEventsContinuation,
    retained: tuple[str, ...],
) -> IdeaBatch:
    return IdeaBatch(
        mode=RuntimeMode.SHADOW,
        rehydrated_events=events,
        continuation_request=request,
        input_watermark="ordinary-watermark",
        causal_from_at_us=min(item.event_at_us for item in events),
        causal_through_at_us=max(item.event_at_us for item in events),
        prior_state_input_event_ids=retained,
    )


def _full_cohort_context_state() -> tuple[JsonValue, tuple[str, ...]]:
    baselines = {
        symbol: {
            "i": f"baseline-{symbol}",
            "s": "2026-08-07",
            "c": 100.0,
            "v": 0.25,
            "t": 1_000_000,
            "k": 21,
        }
        for symbol in COHORT
    }
    option_context = {
        symbol: {
            "s": "2026-08-07",
            "call": {
                "i": f"capture-{symbol}-call",
                "t": 2_000_000,
                "source_completeness": "complete",
                "bid": 2.0,
                "ask": 2.2,
                "model_implied_volatility": 0.40,
                "option_right": "call",
                "expiry": "20260918",
                "strike": 100.0,
            },
            "put": {
                "i": f"capture-{symbol}-put",
                "t": 2_000_001,
                "source_completeness": "complete",
                "bid": 1.8,
                "ask": 2.0,
                "model_implied_volatility": 0.38,
                "option_right": "put",
                "expiry": "20260918",
                "strike": 100.0,
            },
        }
        for symbol in COHORT
    }
    state = cast(
        JsonValue,
        {
            "schema_version": 2,
            "baselines": baselines,
            "option_context": option_context,
            "option_terminal": {
                symbol: {"s": "2026-08-07", "call": "captured", "put": "captured"}
                for symbol in COHORT
            },
            "prefix_roots": {},
            "prefix_index": {},
            "processed": {},
            "statuses": {},
            "episodes": {},
            "requested": {symbol: "2026-08-07" for symbol in COHORT},
        },
    )
    retained = tuple(
        [*(f"baseline-{symbol}" for symbol in COHORT)]
        + [f"capture-{symbol}-{right}" for symbol in COHORT for right in ("call", "put")]
    )
    return state, retained


def _project_all(database: Path, instruments: tuple[str, ...] = ("AAL",)) -> int:
    total = 0
    while True:
        with connect_v2(database) as connection:
            inserted = project_required_session_receipts(
                connection,
                run_id="run-1",
                requirements=tuple(_requirement(value) for value in instruments),
                after_source_sequence=0,
                limit=256,
            )
        total += inserted
        if inserted == 0:
            return total


def test_session_11_prefix_uses_ten_complete_session_volume_baseline(tmp_path: Path) -> None:
    database = tmp_path / "activity.sqlite3"
    _seed(database)
    sequence = 1
    sessions = _sessions(11)
    for session in sessions[:10]:
        sequence = _insert_session(database, session, first_sequence=sequence)
    assert _project_all(database) == 790

    with connect_v2(database) as connection:
        baseline = connection.execute(
            "SELECT event_id, payload_json FROM market_events "
            "WHERE event_kind='session_volume_baseline' ORDER BY event_at_us DESC LIMIT 1"
        ).fetchone()
    assert baseline is not None
    baseline_payload = json.loads(str(baseline["payload_json"]))
    assert baseline_payload["complete_session_count"] == 10
    assert baseline_payload["ordinal_volume_sums"] == [10.0 * number for number in range(1, 79)]
    assert baseline_payload["realised_volatility_20d"] is None

    sequence = _insert_session(
        database,
        sessions[10],
        first_sequence=sequence,
        volume_multiplier=2.0,
    )
    assert _project_all(database) == 79
    with connect_v2(database) as connection:
        prefix = connection.execute(
            "SELECT event_id, payload_json FROM market_events "
            "WHERE event_kind='bar_5m_session_prefix' "
            "AND json_extract(payload_json, '$.session')=? "
            "AND json_extract(payload_json, '$.bar_number')=1",
            (sessions[10].isoformat(),),
        ).fetchone()
        mappings = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT input_role, input_event_id FROM market_event_derivations "
                "WHERE derived_event_id=? ORDER BY input_ordinal",
                (prefix["event_id"],),
            )
        )
    payload = json.loads(str(prefix["payload_json"]))
    assert payload["historical_session_count"] == 10
    assert payload["historical_relative_activity"] == 2.0
    assert [role for role, _event_id in mappings] == ["constituent", "context"]
    assert _project_all(database) == 0


def test_missing_activity_poison_is_sticky_until_the_next_session(tmp_path: Path) -> None:
    database = tmp_path / "activity-poison.sqlite3"
    _seed(database)
    sequence = 1
    sessions = _sessions(12)
    for session in sessions[:10]:
        sequence = _insert_session(
            database,
            session,
            first_sequence=sequence,
            zero_volume_bar=2,
        )
        _project_all(database)

    sequence = _insert_session(database, sessions[10], first_sequence=sequence)
    _project_all(database)
    with connect_v2(database) as connection:
        poisoned = connection.execute(
            "SELECT payload_json FROM market_events "
            "WHERE event_kind='bar_5m_session_prefix' "
            "AND json_extract(payload_json, '$.session')=? "
            "AND json_extract(payload_json, '$.bar_number')=34",
            (sessions[10].isoformat(),),
        ).fetchone()
    assert poisoned is not None
    poisoned_payload = json.loads(str(poisoned["payload_json"]))
    assert poisoned_payload["activity_ready"] is False
    assert poisoned_payload["accumulator"]["activity_sum"] is None
    with pytest.raises(ValueError, match="activity"):
        build_group_i_for_symbol(poisoned_payload, symbol="AAL", checkpoint=34)

    _insert_session(database, sessions[11], first_sequence=sequence)
    _project_all(database)
    with connect_v2(database) as connection:
        recovered = connection.execute(
            "SELECT payload_json FROM market_events "
            "WHERE event_kind='bar_5m_session_prefix' "
            "AND json_extract(payload_json, '$.session')=? "
            "AND json_extract(payload_json, '$.bar_number')=6",
            (sessions[11].isoformat(),),
        ).fetchone()
    assert recovered is not None
    recovered_payload = json.loads(str(recovered["payload_json"]))
    assert recovered_payload["activity_ready"] is True
    assert isinstance(recovered_payload["accumulator"]["activity_sum"], float)


def test_session_receipt_and_lineage_insert_are_atomic(tmp_path: Path) -> None:
    database = tmp_path / "atomic-session-receipt.sqlite3"
    _seed(database)
    _insert_session(database, _sessions(1)[0], first_sequence=1)
    with connect_v2(database) as connection:
        connection.execute(
            "CREATE TEMP TRIGGER fail_session_lineage "
            "BEFORE INSERT ON market_event_derivations BEGIN "
            "SELECT RAISE(ABORT, 'fixture lineage failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="fixture lineage failure"):
            project_required_session_receipts(
                connection,
                run_id="run-1",
                requirements=(_requirement(),),
                after_source_sequence=0,
                limit=1,
            )
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_kind='bar_5m_session_prefix'"
            ).fetchone()[0]
            == 0
        )
        connection.execute("DROP TRIGGER fail_session_lineage")
        assert (
            project_required_session_receipts(
                connection,
                run_id="run-1",
                requirements=(_requirement(),),
                after_source_sequence=0,
                limit=1,
            )
            == 1
        )
        assert (
            connection.execute("SELECT count(*) FROM market_event_derivations").fetchone()[0] == 1
        )


def test_incomplete_session_does_not_advance_baseline_or_realised_volatility(
    tmp_path: Path,
) -> None:
    database = tmp_path / "incomplete.sqlite3"
    _seed(database)
    sequence = 1
    sessions = _sessions(22)
    for index, session in enumerate(sessions):
        sequence = _insert_session(
            database,
            session,
            first_sequence=sequence,
            volume_multiplier=1.0 + index / 100.0,
            incomplete_bar=17 if index == 10 else None,
        )
        _project_all(database)

    with connect_v2(database) as connection:
        baselines = tuple(
            connection.execute(
                "SELECT payload_json FROM market_events "
                "WHERE event_kind='session_volume_baseline' ORDER BY event_at_us"
            )
        )
    assert len(baselines) == 21
    final = json.loads(str(baselines[-1]["payload_json"]))
    assert final["complete_session_count"] == 21
    assert isinstance(final["realised_volatility_20d"], float)
    assert math.isfinite(final["realised_volatility_20d"])


def test_option_snapshot_capture_is_exact_bounded_and_restart_safe(tmp_path: Path) -> None:
    database = tmp_path / "option-capture.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency, option_expiry, option_strike, option_right, option_multiplier) "
            "VALUES ('AAL-option', ?, 9001, 'option', 'AAL', 'SMART', 'USD', '20260821', "
            "'100', 'call', '100')",
            (hashlib.sha256(b"AAL-option").hexdigest(),),
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us, closed_at_us, snapshot) VALUES "
            "('option-sub', 'run-1', 1, 1, 'AAL-option', 'quotes', 2000000, 'closed', "
            "?, 100, 200, 1)",
            ("b" * 64,),
        )
        payloads: tuple[tuple[str, dict[str, JsonValue]], ...] = (
            ("quote", {"event_at_us": 110, "bid": 2.0}),
            ("quote", {"event_at_us": 120, "ask": 2.2}),
            (
                "quote",
                {
                    "event_at_us": 130,
                    "call_open_interest": 150.0,
                    "call_option_volume": -1.0,
                },
            ),
            (
                "option_computation",
                {
                    "event_at_us": 140,
                    "tick_type": 13,
                    "implied_volatility": 0.4,
                    "delta": 0.52,
                    "option_price": 2.1,
                    "underlying_price": 100.0,
                },
            ),
            ("option_snapshot_end", {"event_at_us": 150, "complete": True}),
        )
        for sequence, (kind, payload) in enumerate(payloads, start=1):
            payload_json = canonical_json_bytes(payload).decode()
            event_id = f"option-source-{sequence}"
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, request_id, callback_kind, "
                "received_at_us, payload_json, payload_sha256, lifecycle) "
                "VALUES (?, ?, 'run-1', 1, 1, 2000000, ?, ?, ?, ?, 'pending')",
                (
                    sequence,
                    event_id,
                    kind,
                    int(payload["event_at_us"]),
                    payload_json,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                ),
            )
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                "bid_value, ask_value, payload_json, payload_sha256) VALUES "
                "(?, 'run-1', ?, 'AAL-option', 'quotes', ?, ?, ?, 1, ?, ?, ?, ?)",
                (
                    event_id,
                    sequence,
                    kind,
                    int(payload["event_at_us"]),
                    int(payload["event_at_us"]),
                    payload.get("bid"),
                    payload.get("ask"),
                    payload_json,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                ),
            )
            connection.execute(
                "UPDATE callback_inbox SET lifecycle='acknowledged', normalized_event_id=?, "
                "acknowledged_at_us=? WHERE source_sequence=?",
                (event_id, int(payload["event_at_us"]), sequence),
            )

        assert project_option_snapshot_captures(connection, run_id="run-1") == 1
        capture = connection.execute(
            "SELECT event_id, payload_json FROM market_events "
            "WHERE event_kind='option_snapshot_capture'"
        ).fetchone()
        mappings = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT input_role, input_event_id FROM market_event_derivations "
                "WHERE derived_event_id=? ORDER BY input_ordinal",
                (capture["event_id"],),
            )
        )
        assert project_option_snapshot_captures(connection, run_id="run-1") == 0
    payload = json.loads(str(capture["payload_json"]))
    assert payload["source_completeness"] == "complete"
    assert payload["bid"] == 2.0
    assert payload["ask"] == 2.2
    assert payload["model_implied_volatility"] == 0.4
    assert payload["model_delta"] == 0.52
    assert payload["open_interest"] == 150.0
    assert payload["open_interest_missing"] is False
    assert payload["option_volume"] is None
    assert payload["option_volume_missing"] is True
    assert [role for role, _event_id in mappings] == [
        "constituent",
        "constituent",
        "constituent",
        "constituent",
        "completion",
    ]


def test_pure_frozen_m1c_runtime_matches_committed_golden_probability() -> None:
    result = score_m1c(
        symbol="AAL",
        checkpoint=6,
        group_o={name: 0.0 for name in REQUIRED_GROUP_O_FEATURES},
        group_i={name: 0.0 for name in CAUSAL_GROUP_I_FEATURES},
    )

    assert result["probability"] == pytest.approx(0.3791098724444006, abs=1e-15)
    assert result["threshold"] == 0.488333710794033
    assert result["threshold_passed"] is False
    assert result["missing_feature_count"] == 0


def test_frozen_m1c_plugin_declares_exact_bounded_prefix_requirements() -> None:
    plugin = FrozenM1CSignalV0()
    parameters = {
        "checkpoints": tuple(range(6, 35, 2)),
        "minimum_activity_sessions": 10,
        "minimum_episode_spacing_minutes": 30,
        "option_minimum_days_to_expiry": 7,
        "option_maximum_days_to_expiry": 45,
        "option_snapshot_lifetime_minutes": 30,
        "threshold": 0.488333710794033,
    }
    activation = IdeaActivation(
        instance_id="m1c-1",
        parameters=parameters,
        parameters_hash=hashlib.sha256(canonical_json_bytes(parameters)).hexdigest(),
        plugin_code_hash="f" * 64,
        activated_at_us=1,
        run_id="run-1",
        protected_data_class=ProtectedDataClass.SHADOW,
        universe=UNIVERSE,
    )

    requirements = plugin.requirements(activation)

    assert (*COHORT, "VTI") == UNIVERSE
    assert len(requirements) == 41
    assert {item.instrument_id for item in requirements} == set(UNIVERSE)
    assert sum(item.event_kind == "bar_5m_session_prefix" for item in requirements) == 21
    assert sum(item.event_kind == "session_volume_baseline" for item in requirements) == 20
    assert all(item.gaps_block and item.staleness_block for item in requirements)
    assert plugin.manifest.maximum_interests_per_batch == 40
    assert set(plugin.manifest.output_kinds) == {"observation", "signal"}


def test_pure_front_options_context_matches_frozen_research_transform() -> None:
    call = {
        "source_completeness": "complete",
        "option_right": "call",
        "expiry": "20260821",
        "strike": 100.0,
        "bid": 2.0,
        "ask": 2.2,
        "model_implied_volatility": 0.42,
    }
    put = {
        "source_completeness": "complete",
        "option_right": "put",
        "expiry": "20260821",
        "strike": 100.0,
        "bid": 1.8,
        "ask": 2.0,
        "model_implied_volatility": 0.40,
    }
    actual = build_front_options_context(
        call_capture=call,
        put_capture=put,
        prior_close=100.0,
        realised_volatility_20d=0.30,
    )

    feature_manifest = json.loads(
        (FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json").read_text()
    )
    mapping = json.loads((FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json").read_text())
    raw = pd.DataFrame(
        [
            {
                "atm_iv": 0.41,
                "straddle_mid_pct": 0.04,
                "call_put_iv_gap": 0.02,
                "skew_25d": math.nan,
                "combined_relative_spread": 0.10,
                "iv_minus_realised_20d": 0.11,
                "near_spot_oi_concentration": math.nan,
                "call_put_oi_imbalance": math.nan,
                "skew_25d_missing": 1.0,
                "near_spot_oi_concentration_missing": 1.0,
                "call_put_oi_imbalance_missing": 1.0,
            }
        ]
    )
    parameters = FrozenDimensionParameters(
        kind="front_options",
        scales={
            name: RobustValueScale(**values) for name, values in feature_manifest["scales"].items()
        },
        imputation_medians=feature_manifest["imputation_medians"],
    )
    expected = apply_serialized_diag_regime(
        apply_front_options_dimensions(raw, parameters),
        mapping,
        prefix="front_options_regime",
    ).iloc[0]

    for name in REQUIRED_GROUP_O_FEATURES:
        assert actual[name] == pytest.approx(float(expected[name]), abs=1e-14)


def test_pure_group_i_and_direction_features_match_frozen_research_builders(
    tmp_path: Path,
) -> None:
    database = tmp_path / "parity.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, "
            "currency) VALUES ('VTI', ?, 'stock', 'VTI', 'SMART', 'USD')",
            (hashlib.sha256(b"VTI").hexdigest(),),
        )
    sequence = 1
    sessions = _sessions(11)
    for session in sessions[:10]:
        sequence = _insert_session(database, session, first_sequence=sequence)
        sequence = _insert_session(
            database,
            session,
            first_sequence=sequence,
            instrument_id="VTI",
            base_price=250.0,
        )
    _project_all(database, ("AAL", "VTI"))
    sequence = _insert_session(
        database,
        sessions[10],
        first_sequence=sequence,
        volume_multiplier=1.5,
    )
    _insert_session(
        database,
        sessions[10],
        first_sequence=sequence,
        volume_multiplier=1.25,
        instrument_id="VTI",
        base_price=250.0,
    )
    _project_all(database, ("AAL", "VTI"))
    with connect_v2(database) as connection:
        rows = {
            str(row["instrument_id"]): json.loads(str(row["payload_json"]))
            for row in connection.execute(
                "SELECT instrument_id, payload_json FROM market_events "
                "WHERE event_kind='bar_5m_session_prefix' "
                "AND json_extract(payload_json, '$.session')=? "
                "AND json_extract(payload_json, '$.bar_number')=6",
                (sessions[10].isoformat(),),
            )
        }
    actual_group_i = build_group_i_for_symbol(rows["AAL"], symbol="AAL", checkpoint=6)
    bars = tuple(
        LiveFeatureBar(
            symbol="AAL",
            session=sessions[10],
            bar_ordinal=int(item["bar_number"]) - 1,
            bar_start_timestamp=datetime.fromtimestamp(
                (int(item["event_at_us"]) - 300_000_000) / 1_000_000,
                tz=UTC,
            ),
            bar_complete_timestamp=datetime.fromtimestamp(
                int(item["event_at_us"]) / 1_000_000,
                tz=UTC,
            ),
            open=float(item["open"]),
            high=float(item["high"]),
            low=float(item["low"]),
            close=float(item["close"]),
            volume=float(item["volume"]),
            historical_relative_activity=float(item["historical_relative_activity"]),
            finalised=True,
            source="test",
        )
        for item in rows["AAL"]["trailing_bars"]
    )
    expected_group_i = M1CCausalFeatureBuilder.from_scaling_artifact(
        ROOT / "research/route-competition/20260722-broad-conflict-advance-hazard-v02/"
        "artifacts/primary/model_configurations.json"
    ).build(symbol="AAL", checkpoint=6, completed_bars=bars)
    assert actual_group_i == pytest.approx(expected_group_i.scaled_features, abs=1e-12)

    actual_direction = build_direction_features(
        symbol="AAL",
        checkpoint=6,
        stock_prefix=rows["AAL"],
        market_prefix=rows["VTI"],
    )
    market_by_number = {int(item["bar_number"]): item for item in rows["VTI"]["trailing_bars"]}
    direction_bars = tuple(
        DirectionFeatureBar(
            symbol="AAL",
            session=sessions[10],
            bar_ordinal=int(item["bar_number"]) - 1,
            bar_start_timestamp=datetime.fromtimestamp(
                (int(item["event_at_us"]) - 300_000_000) / 1_000_000,
                tz=UTC,
            ),
            bar_complete_timestamp=datetime.fromtimestamp(
                int(item["event_at_us"]) / 1_000_000,
                tz=UTC,
            ),
            open=float(item["open"]),
            high=float(item["high"]),
            low=float(item["low"]),
            close=float(item["close"]),
            volume=float(item["volume"]),
            historical_relative_activity=float(item["historical_relative_activity"]),
            stock_log_return=math.log1p(float(item["return_bps"]) / 10_000.0),
            market_log_return=math.log1p(
                float(market_by_number[int(item["bar_number"])]["return_bps"]) / 10_000.0
            ),
            finalised=True,
        )
        for item in rows["AAL"]["trailing_bars"]
    )
    expected_direction = FrozenDirectionFeatureBuilder.from_beta_artifact(
        ARCHETYPE_ROOT / "stock_market_beta_parameters.csv"
    ).build(symbol="AAL", checkpoint=6, completed_bars=direction_bars)
    for name, expected_value in expected_direction.raw_features.items():
        if math.isnan(expected_value):
            assert math.isnan(actual_direction[name])
        else:
            assert actual_direction[name] == pytest.approx(expected_value, abs=1e-14)

    actual_classes = classify_directions(
        symbol="AAL",
        checkpoint=6,
        session=sessions[10].isoformat(),
        raw_features=actual_direction,
    )
    expected_classes = FrozenDirectionRuntime.from_artifacts(
        model_configurations_path=ARCHETYPE_ROOT / "model_configurations.json",
        normalisation_path=ARCHETYPE_ROOT / "stock_local_normalisation_parameters.json",
        thresholds_path=ARCHETYPE_ROOT / "frozen_archetype_thresholds.json",
    ).classify(
        raw_features=expected_direction.raw_features,
        symbol="AAL",
        checkpoint=6,
        checkpoint_category="6",
        day_of_week=sessions[10].strftime("%A"),
    )
    for actual in actual_classes:
        expected = expected_classes[str(actual["model_id"])]
        assert actual["probability_up"] == pytest.approx(expected.probability_up, abs=1e-14)
        assert actual["action"] == expected.action
        assert actual["label"] == expected.label


def test_frozen_m1c_baseline_requests_only_one_bounded_atm_pair() -> None:
    event = _event(
        "baseline-aal",
        "AAL",
        "session_volume_baseline",
        1_000_000,
        {
            "session": "2026-08-07",
            "complete_session_count": 21,
            "session_closes": [100.0, 101.0],
            "realised_volatility_20d": 0.25,
        },
    )
    batch = IdeaBatch(
        mode=RuntimeMode.SHADOW,
        events=(event,),
        input_watermark=event.event_id,
        causal_from_at_us=event.event_at_us,
        causal_through_at_us=event.event_at_us,
    )

    evaluation = FrozenM1CSignalV0().evaluate(batch, {})

    assert len(evaluation.interests) == 2
    assert [item.option_right for item in evaluation.interests] == ["call", "put"]
    assert all(item.strike_offset == 0 for item in evaluation.interests)
    assert all(item.minimum_days_to_expiry == 7 for item in evaluation.interests)
    assert all(item.maximum_days_to_expiry == 45 for item in evaluation.interests)
    assert all(
        item.cadence == "snapshot" and item.maximum_contracts == 1 for item in evaluation.interests
    )
    assert evaluation.retained_input_event_ids == (event.event_id,)
    assert len(evaluation.state_json()) < 65_536


def test_frozen_m1c_catch_up_requests_only_latest_pair_per_symbol() -> None:
    events = tuple(
        _event(
            f"baseline-{session}-{symbol}",
            symbol,
            "session_volume_baseline",
            (session_index + 1) * 1_000_000 + symbol_index,
            {
                "session": session,
                "complete_session_count": 21 + session_index,
                "session_closes": [100.0, 101.0 + session_index],
                "realised_volatility_20d": 0.25,
            },
        )
        for session_index, session in enumerate(("2026-08-06", "2026-08-07"))
        for symbol_index, symbol in enumerate(COHORT)
    )
    evaluation = FrozenM1CSignalV0().evaluate(
        IdeaBatch(
            mode=RuntimeMode.SHADOW,
            events=events,
            input_watermark=events[-1].event_id,
            causal_from_at_us=events[0].event_at_us,
            causal_through_at_us=events[-1].event_at_us,
        ),
        {},
    )

    assert len(evaluation.interests) == 40
    assert all(":2026-08-07:" in item.interest_key for item in evaluation.interests)
    assert {item.underlying_instrument_id for item in evaluation.interests} == set(COHORT)


def test_frozen_m1c_emits_nothing_before_a_complete_cohort_is_available() -> None:
    plugin = FrozenM1CSignalV0()
    first = _event(
        "aal-prefix-6",
        "AAL",
        "bar_5m_session_prefix",
        6_000_000,
        _prefix_fixture(session="2026-08-10", checkpoint=6),
    )
    initial = plugin.evaluate(
        IdeaBatch(
            mode=RuntimeMode.SHADOW,
            events=(first,),
            input_watermark=first.event_id,
            causal_from_at_us=first.event_at_us,
            causal_through_at_us=first.event_at_us,
        ),
        {},
    )
    assert initial.outputs == ()
    assert initial.continuation is None
    assert not initial.interests

    second = _event(
        "aal-prefix-8",
        "AAL",
        "bar_5m_session_prefix",
        8_000_000,
        _prefix_fixture(session="2026-08-10", checkpoint=8),
    )
    repeated = plugin.evaluate(
        IdeaBatch(
            mode=RuntimeMode.SHADOW,
            events=(second,),
            input_watermark=second.event_id,
            causal_from_at_us=second.event_at_us,
            causal_through_at_us=second.event_at_us,
            prior_state_input_event_ids=initial.retained_input_event_ids,
        ),
        initial.state,
    )
    assert repeated.outputs == ()
    assert repeated.continuation is None
    assert len(repeated.state_json()) < 65_536


def test_frozen_m1c_consumes_exact_resolved_snapshot_pair_and_scores() -> None:
    plugin = FrozenM1CSignalV0()
    baseline_event = _event(
        "baseline-aal",
        "AAL",
        "session_volume_baseline",
        1_000_000,
        {
            "session": "2026-08-07",
            "complete_session_count": 21,
            "session_closes": [100.0, 101.0],
            "realised_volatility_20d": 0.25,
        },
    )
    baseline = plugin.evaluate(
        IdeaBatch(
            mode=RuntimeMode.SHADOW,
            events=(baseline_event,),
            input_watermark=baseline_event.event_id,
            causal_from_at_us=baseline_event.event_at_us,
            causal_through_at_us=baseline_event.event_at_us,
        ),
        {},
    )
    receipts = tuple(
        DiscoveryReceipt(
            receipt_id=f"receipt-{right}",
            interest_id=f"interest-{right}",
            interest_key=f"m1c:d1:2026-08-07:AAL:{right}",
            instance_id="m1c-1",
            status="resolved",
            instrument_id=f"AAL-{right}",
            expiry="20260918",
            strike=100.0,
            option_right=right,
            multiplier="100",
            candidates_inspected=20,
            completed_at_us=1_500_000,
        )
        for right in ("call", "put")
    )
    captures = tuple(
        _event(
            f"capture-{right}",
            f"AAL-{right}",
            "option_snapshot_capture",
            2_000_000 + index,
            {
                "source_completeness": "complete",
                "bid": 2.0 if right == "call" else 1.8,
                "ask": 2.2 if right == "call" else 2.0,
                "model_implied_volatility": 0.40 if right == "call" else 0.38,
            },
        )
        for index, right in enumerate(("call", "put"))
    )
    captured = plugin.evaluate(
        IdeaBatch(
            mode=RuntimeMode.SHADOW,
            events=captures,
            input_watermark=captures[-1].event_id,
            causal_from_at_us=captures[0].event_at_us,
            causal_through_at_us=captures[-1].event_at_us,
            prior_state_input_event_ids=baseline.retained_input_event_ids,
            discovery_receipts=receipts,
        ),
        baseline.state,
    )
    stock = _event(
        "aal-current",
        "AAL",
        "bar_5m_session_prefix",
        10_000_000,
        _prefix_fixture(session="2026-08-10", checkpoint=6),
    )
    market = _event(
        "vti-current",
        "VTI",
        "bar_5m_session_prefix",
        10_000_001,
        _prefix_fixture(session="2026-08-10", checkpoint=6, base=250.0),
    )
    continuation_state, retained, events, requests = _exact_continuation_fixture(
        state=captured.state,
        session="2026-08-10",
        checkpoints=(6,),
        prior_ids=captured.retained_input_event_ids,
        event_overrides={("AAL", 6): stock, ("VTI", 6): market},
    )
    scored = plugin.evaluate(
        _exact_batch(events=events[6], request=requests[6], retained=retained),
        continuation_state,
    )

    aal_outputs = tuple(item for item in scored.outputs if item.subject_instrument_id == "AAL")
    assert aal_outputs
    assert aal_outputs[0].kind == "observation"
    assert aal_outputs[0].payload["status"] == "complete"
    assert isinstance(aal_outputs[0].payload["probability"], float)
    assert all(item.kind in {"observation", "signal"} for item in scored.outputs)
    assert len(scored.state_json()) < 65_536


@pytest.mark.parametrize(
    ("call_completeness", "put_strike", "reason"),
    (
        ("incomplete", 100.0, "prior_session_option_pair_incomplete"),
        ("complete", 101.0, "prior_session_option_pair_mismatch"),
    ),
)
def test_frozen_m1c_fails_closed_for_invalid_option_pair(
    call_completeness: str,
    put_strike: float,
    reason: str,
) -> None:
    stock = _event(
        "aal-current",
        "AAL",
        "bar_5m_session_prefix",
        10_000_000,
        _prefix_fixture(session="2026-08-10", checkpoint=6),
    )
    market = _event(
        "vti-current",
        "VTI",
        "bar_5m_session_prefix",
        10_000_001,
        _prefix_fixture(session="2026-08-10", checkpoint=6, base=250.0),
    )
    state = cast(
        JsonValue,
        {
            "schema_version": 1,
            "baselines": {
                "AAL": {
                    "i": "baseline-aal",
                    "s": "2026-08-07",
                    "c": 101.0,
                    "v": 0.25,
                    "t": 1_000_000,
                    "k": 21,
                }
            },
            "option_context": {
                "AAL": {
                    "s": "2026-08-07",
                    "call": {
                        "i": "capture-call",
                        "t": 2_000_000,
                        "source_completeness": call_completeness,
                        "bid": 2.0,
                        "ask": 2.2,
                        "model_implied_volatility": 0.40,
                        "option_right": "call",
                        "expiry": "20260918",
                        "strike": 100.0,
                    },
                    "put": {
                        "i": "capture-put",
                        "t": 2_000_001,
                        "source_completeness": "complete",
                        "bid": 1.8,
                        "ask": 2.0,
                        "model_implied_volatility": 0.38,
                        "option_right": "put",
                        "expiry": "20260918",
                        "strike": put_strike,
                    },
                }
            },
            "pending": {},
            "market": {},
            "statuses": {},
            "episodes": {},
            "requested": {"AAL": "2026-08-07"},
        },
    )
    continuation_state, retained, events, requests = _exact_continuation_fixture(
        state=state,
        session="2026-08-10",
        checkpoints=(6,),
        prior_ids=("baseline-aal", "capture-call", "capture-put"),
        event_overrides={("AAL", 6): stock, ("VTI", 6): market},
    )
    evaluation = FrozenM1CSignalV0().evaluate(
        _exact_batch(events=events[6], request=requests[6], retained=retained),
        continuation_state,
    )

    aal_outputs = tuple(item for item in evaluation.outputs if item.subject_instrument_id == "AAL")
    assert len(aal_outputs) == 1
    assert aal_outputs[0].kind == "observation"
    assert aal_outputs[0].payload["status"] == "unavailable"
    assert aal_outputs[0].payload["reason"] == reason
    assert not evaluation.interests


def test_fresh_m1c_emits_real_labelled_classifications_only_as_generic_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m1c_plugin,
        "score_m1c",
        lambda **_values: {
            "probability": 0.75,
            "threshold_passed": True,
            "missing_feature_count": 0,
            "feature_hash": "1" * 64,
            "model_hash": "2" * 64,
        },
    )
    monkeypatch.setattr(
        m1c_plugin,
        "build_direction_features",
        lambda **_values: {"fixture": 1.0},
    )
    monkeypatch.setattr(
        m1c_plugin,
        "classify_directions",
        lambda **_values: tuple(
            {
                "model_id": model_id,
                "probability_up": 0.6,
                "confidence": 0.1,
                "action": "CALL",
                "boundary": 0.05,
                "label": (
                    "prospective hypothesis — not validated"
                    if model_id == "A1"
                    else "comparison only — not validated"
                ),
                "model_hash": "3" * 64,
                "preprocessing_hash": "4" * 64,
                "feature_hash": "5" * 64,
                "fallback_levels": ("stock_checkpoint",),
            }
            for model_id in ("A1", "C1", "R1")
        ),
    )
    session = "2026-08-10"
    stock = _event(
        "aal-current",
        "AAL",
        "bar_5m_session_prefix",
        10_000_000,
        _prefix_fixture(session=session, checkpoint=6),
    )
    market = _event(
        "vti-current",
        "VTI",
        "bar_5m_session_prefix",
        10_000_001,
        _prefix_fixture(session=session, checkpoint=6, base=250.0),
    )
    state = {
        "schema_version": 1,
        "baselines": {
            "AAL": {
                "i": "aal-baseline",
                "s": "2026-08-07",
                "c": 100.0,
                "v": 0.25,
                "t": 1_000_000,
                "k": 21,
            }
        },
        "option_context": {
            "AAL": {
                "s": "2026-08-07",
                "call": {
                    "i": "aal-call",
                    "t": 2_000_000,
                    "source_completeness": "complete",
                    "bid": 2.0,
                    "ask": 2.2,
                    "model_implied_volatility": 0.4,
                    "option_right": "call",
                    "expiry": "20260918",
                    "strike": 100.0,
                },
                "put": {
                    "i": "aal-put",
                    "t": 2_000_001,
                    "source_completeness": "complete",
                    "bid": 1.8,
                    "ask": 2.0,
                    "model_implied_volatility": 0.38,
                    "option_right": "put",
                    "expiry": "20260918",
                    "strike": 100.0,
                },
            }
        },
        "pending": {},
        "market": {},
        "statuses": {},
        "episodes": {},
        "requested": {},
    }
    prior_ids = ("aal-baseline", "aal-call", "aal-put")
    continuation_state, retained, events, requests = _exact_continuation_fixture(
        state=cast(JsonValue, state),
        session=session,
        checkpoints=(6,),
        prior_ids=prior_ids,
        event_overrides={("AAL", 6): stock, ("VTI", 6): market},
    )
    evaluation = FrozenM1CSignalV0().evaluate(
        _exact_batch(events=events[6], request=requests[6], retained=retained),
        continuation_state,
    )

    aal_outputs = tuple(item for item in evaluation.outputs if item.subject_instrument_id == "AAL")
    assert [item.kind for item in aal_outputs] == [
        "observation",
        "signal",
        "observation",
        "observation",
        "observation",
    ]
    controls = [
        item.payload
        for item in aal_outputs
        if item.payload.get("control") == "direction_classification"
    ]
    assert [item["model_id"] for item in controls] == ["A1", "C1", "R1"]
    assert controls[0]["label"] == "prospective hypothesis — not validated"
    assert controls[1]["label"] == controls[2]["label"] == "comparison only — not validated"
    assert all(item.payload.get("execution_enabled") is False for item in aal_outputs)
    assert {item.kind for item in aal_outputs}.isdisjoint({"proposed_position", "proposed_trade"})
    assert len(evaluation.state_json()) < 65_536


def test_frozen_m1c_backlog_batch_preserves_every_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m1c_plugin,
        "score_m1c",
        lambda **values: {
            "probability": 0.75 if values["checkpoint"] == 6 else 0.25,
            "threshold_passed": values["checkpoint"] == 6,
            "missing_feature_count": 0,
            "feature_hash": f"{values['checkpoint']:064x}",
            "model_hash": "2" * 64,
        },
    )
    monkeypatch.setattr(m1c_plugin, "build_direction_features", lambda **_values: {})
    monkeypatch.setattr(
        m1c_plugin,
        "classify_directions",
        lambda **_values: (
            {
                "model_id": "A1",
                "probability_up": 0.6,
                "confidence": 0.1,
                "action": "CALL",
                "boundary": 0.05,
                "label": "prospective hypothesis — not validated",
                "model_hash": "3" * 64,
                "preprocessing_hash": "4" * 64,
                "feature_hash": "5" * 64,
                "fallback_levels": ("stock_checkpoint",),
            },
        ),
    )

    def initial_state() -> JsonValue:
        return cast(
            JsonValue,
            {
                "schema_version": 1,
                "baselines": {
                    "AAL": {
                        "i": "aal-baseline",
                        "s": "2026-08-07",
                        "c": 100.0,
                        "v": 0.25,
                        "t": 1_000_000,
                        "k": 21,
                    }
                },
                "option_context": {
                    "AAL": {
                        "s": "2026-08-07",
                        "call": {
                            "i": "aal-call",
                            "t": 2_000_000,
                            "source_completeness": "complete",
                            "bid": 2.0,
                            "ask": 2.2,
                            "model_implied_volatility": 0.4,
                            "option_right": "call",
                            "expiry": "20260918",
                            "strike": 100.0,
                        },
                        "put": {
                            "i": "aal-put",
                            "t": 2_000_001,
                            "source_completeness": "complete",
                            "bid": 1.8,
                            "ask": 2.0,
                            "model_implied_volatility": 0.38,
                            "option_right": "put",
                            "expiry": "20260918",
                            "strike": 100.0,
                        },
                    }
                },
                "pending": {},
                "market": {},
                "statuses": {},
                "episodes": {},
                "requested": {},
            },
        )

    session = "2026-08-10"
    events = {
        (instrument_id, checkpoint): _event(
            f"{instrument_id.lower()}-{checkpoint}",
            instrument_id,
            "bar_5m_session_prefix",
            checkpoint * 1_000_000 + (1 if instrument_id == "VTI" else 0),
            _prefix_fixture(
                session=session,
                checkpoint=checkpoint,
                base=250.0 if instrument_id == "VTI" else 100.0,
            ),
        )
        for checkpoint in (6, 8)
        for instrument_id in ("AAL", "VTI")
    }
    prior_ids = ("aal-baseline", "aal-call", "aal-put")
    continuation_state, retained, cohort_events, requests = _exact_continuation_fixture(
        state=initial_state(),
        session=session,
        checkpoints=(6, 8),
        prior_ids=prior_ids,
        event_overrides=events,
    )
    first = FrozenM1CSignalV0().evaluate(
        _exact_batch(events=cohort_events[6], request=requests[6], retained=retained),
        continuation_state,
    )
    assert first.continuation == requests[8]
    second = FrozenM1CSignalV0().evaluate(
        _exact_batch(
            events=cohort_events[8],
            request=cast(ExactEventsContinuation, first.continuation),
            retained=first.retained_input_event_ids,
        ),
        first.state,
    )

    def summary(outputs: tuple[IdeaOutput, ...]) -> list[tuple[str, object, object]]:
        return [
            (item.kind, item.payload.get("checkpoint"), item.payload.get("control"))
            for item in outputs
        ]

    split_outputs = (*first.outputs, *second.outputs)
    assert summary(split_outputs) == [
        ("observation", 6, None),
        ("signal", 6, None),
        ("observation", 6, "direction_classification"),
        ("observation", 8, None),
    ]
    assert (*first.output_input_event_ids, *second.output_input_event_ids) == (
        ("aal-baseline", "aal-call", "aal-put", "aal-6", "vti-6"),
        ("aal-baseline", "aal-call", "aal-put", "aal-6", "vti-6"),
        ("aal-baseline", "aal-call", "aal-put", "aal-6", "vti-6"),
        ("aal-baseline", "aal-call", "aal-put", "aal-8", "vti-8"),
    )
    assert second.continuation is None


def test_frozen_m1c_worst_case_continuation_state_stays_within_contract_bound() -> None:
    def event_id(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    baselines = {
        symbol: {
            "i": event_id(f"baseline-{symbol}"),
            "s": "2026-08-07",
            "c": 999_999.123456789,
            "v": 9.123456789012345,
            "t": 9_999_999_999_999_999,
            "k": 9_999_999,
        }
        for symbol in COHORT
    }
    option_context = {
        symbol: {
            "s": "2026-08-07",
            **{
                right: {
                    "i": event_id(f"capture-{symbol}-{right}"),
                    "t": 9_999_999_999_999_999,
                    "source_completeness": "complete",
                    "bid": 999_999.123456789,
                    "ask": 999_999.987654321,
                    "model_implied_volatility": 4.999999999999999,
                    "option_right": right,
                    "expiry": "20261231",
                    "strike": 999_999.123456789,
                }
                for right in ("call", "put")
            },
        }
        for symbol in COHORT
    }
    prefix_roots = {
        symbol: {
            "i": event_id(f"root-{symbol}"),
            "s": "2026-08-10",
            "n": 78,
            "t": 9_999_999_999_999_999,
        }
        for symbol in UNIVERSE
    }
    prefix_index = {
        f"2026-08-10|{checkpoint:02d}": {
            symbol: event_id(f"prefix-{symbol}-{checkpoint}") for symbol in UNIVERSE
        }
        for checkpoint in range(6, 35, 2)
    }
    state = cast(
        JsonValue,
        {
            "schema_version": 2,
            "baselines": baselines,
            "option_context": option_context,
            "option_terminal": {
                symbol: {"s": "2026-08-07", "call": "captured", "put": "captured"}
                for symbol in COHORT
            },
            "prefix_roots": prefix_roots,
            "prefix_index": prefix_index,
            "processed": {f"2026-08-10|{checkpoint:02d}": 1 for checkpoint in range(6, 34, 2)},
            "statuses": {symbol: {"s": "2026-08-10", "r": "fixture"} for symbol in COHORT},
            "episodes": {
                symbol: {"s": "2026-08-10", "p": 0.999999999, "l": 1, "c": 99} for symbol in COHORT
            },
            "requested": {symbol: "2026-08-07" for symbol in COHORT},
        },
    )
    retained = tuple(
        [*(event_id(f"baseline-{symbol}") for symbol in COHORT)]
        + [event_id(f"capture-{symbol}-{right}") for symbol in COHORT for right in ("call", "put")]
        + [*(event_id(f"root-{symbol}") for symbol in UNIVERSE)]
    )
    continuation = ExactEventsContinuation(
        root_event_ids=tuple(event_id(f"root-{symbol}") for symbol in UNIVERSE),
        event_ids=tuple(prefix_index["2026-08-10|34"][symbol] for symbol in sorted(UNIVERSE)),
        event_kind="bar_5m_session_prefix",
        input_roles=("prior_receipt",),
    )

    evaluation = IdeaEvaluation(
        state=state,
        outputs=(),
        retained_input_event_ids=retained,
        output_input_event_ids=(),
        interests=(),
        continuation=continuation,
    )
    envelope = canonical_json_bytes(
        {
            "_stocker_runner_state_version": 1,
            "plugin_state": state,
            "continuation": continuation.model_dump(mode="json"),
        }
    )

    assert len(evaluation.state_json()) < 60_000
    assert len(envelope) <= 65_536
    assert len(retained) == 81
    assert sum(len(values) for values in prefix_index.values()) == 315


def test_frozen_m1c_skewed_full_cohort_backlog_drains_every_checkpoint_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m1c_plugin,
        "score_m1c",
        lambda **_values: {
            "probability": 0.75,
            "threshold_passed": True,
            "missing_feature_count": 0,
            "feature_hash": "1" * 64,
            "model_hash": "2" * 64,
        },
    )
    monkeypatch.setattr(m1c_plugin, "build_direction_features", lambda **_values: {})
    monkeypatch.setattr(
        m1c_plugin,
        "classify_directions",
        lambda **_values: tuple(
            {
                "model_id": model_id,
                "probability_up": 0.6,
                "confidence": 0.1,
                "action": "CALL",
                "boundary": 0.05,
                "label": (
                    "prospective hypothesis — not validated"
                    if model_id == "A1"
                    else "comparison only — not validated"
                ),
                "model_hash": "3" * 64,
                "preprocessing_hash": "4" * 64,
                "feature_hash": "5" * 64,
                "fallback_levels": ("stock_checkpoint",),
            }
            for model_id in ("A1", "C1", "R1")
        ),
    )
    plugin = FrozenM1CSignalV0()
    session = "2026-08-10"
    stock_events = tuple(
        _event(
            f"prefix-{symbol}-{checkpoint}",
            symbol,
            "bar_5m_session_prefix",
            (symbol_index * 100 + checkpoint) * 1_000_000,
            _prefix_fixture(session=session, checkpoint=checkpoint),
        )
        for symbol_index, symbol in enumerate(COHORT)
        for checkpoint in range(1, 79)
    )
    market_events = tuple(
        _event(
            f"prefix-VTI-{checkpoint}",
            "VTI",
            "bar_5m_session_prefix",
            (10_000 + checkpoint) * 1_000_000,
            _prefix_fixture(session=session, checkpoint=checkpoint, base=250.0),
        )
        for checkpoint in range(1, 79)
    )
    state, retained = _full_cohort_context_state()
    evaluation: IdeaEvaluation | None = None
    events = (*stock_events, *market_events)
    for start in range(0, len(events), 256):
        page = events[start : start + 256]
        evaluation = plugin.evaluate(
            IdeaBatch(
                mode=RuntimeMode.SHADOW,
                events=page,
                input_watermark=page[-1].event_id,
                causal_from_at_us=min(item.event_at_us for item in page),
                causal_through_at_us=max(item.event_at_us for item in page),
                prior_state_input_event_ids=retained,
            ),
            state,
        )
        assert evaluation.outputs == ()
        state = evaluation.state
        retained = evaluation.retained_input_event_ids

    assert evaluation is not None
    assert isinstance(evaluation.continuation, AncestorPageContinuation)
    assert len(evaluation.continuation.root_event_ids) == len(UNIVERSE)
    assert len(retained) <= 81
    assert len(evaluation.state_json()) < 65_536

    split_state, split_retained = _full_cohort_context_state()
    split_evaluation: IdeaEvaluation | None = None
    for start in range(0, len(events), 73):
        page = events[start : start + 73]
        split_evaluation = plugin.evaluate(
            IdeaBatch(
                mode=RuntimeMode.SHADOW,
                events=page,
                input_watermark=page[-1].event_id,
                causal_from_at_us=min(item.event_at_us for item in page),
                causal_through_at_us=max(item.event_at_us for item in page),
                prior_state_input_event_ids=split_retained,
            ),
            split_state,
        )
        split_state = split_evaluation.state
        split_retained = split_evaluation.retained_input_event_ids
    assert split_evaluation is not None
    assert split_evaluation.state == evaluation.state
    assert split_evaluation.continuation == evaluation.continuation
    assert split_retained == retained

    ancestry = tuple(sorted(events, key=lambda item: (item.event_at_us, item.event_id)))
    assert len(ancestry) == len(UNIVERSE) * 78
    request = evaluation.continuation
    for start in range(0, len(ancestry), 64):
        page = ancestry[start : start + 64]
        next_token = (
            f"fixture-page-{start + len(page)}" if start + len(page) < len(ancestry) else None
        )
        evaluation = plugin.evaluate(
            IdeaBatch(
                mode=RuntimeMode.SHADOW,
                rehydrated_events=page,
                continuation_request=request,
                continuation_token=next_token,
                input_watermark="ordinary-watermark",
                causal_from_at_us=min(item.event_at_us for item in page),
                causal_through_at_us=max(item.event_at_us for item in page),
                prior_state_input_event_ids=retained,
            ),
            state,
        )
        assert evaluation.outputs == ()
        assert not evaluation.interests
        state = evaluation.state
        retained = evaluation.retained_input_event_ids
        request = cast(AncestorPageContinuation, evaluation.continuation)

    by_id = {event.event_id: event for event in ancestry}
    continuation_calls = 0
    maximum_outputs = 0
    primary_outputs: list[IdeaOutput] = []
    while isinstance(evaluation.continuation, ExactEventsContinuation):
        request = evaluation.continuation
        exact_events = tuple(by_id[event_id] for event_id in request.event_ids)
        evaluation = plugin.evaluate(
            _exact_batch(events=exact_events, request=request, retained=retained),
            state,
        )
        exact_by_symbol = {item.instrument_id: item.event_id for item in exact_events}
        for output, lineage in zip(
            evaluation.outputs,
            evaluation.output_input_event_ids,
            strict=True,
        ):
            assert {
                f"baseline-{output.subject_instrument_id}",
                f"capture-{output.subject_instrument_id}-call",
                f"capture-{output.subject_instrument_id}-put",
                exact_by_symbol[output.subject_instrument_id],
                exact_by_symbol["VTI"],
            } == set(lineage)
        continuation_calls += 1
        maximum_outputs = max(maximum_outputs, len(evaluation.outputs))
        primary_outputs.extend(
            item
            for item in evaluation.outputs
            if item.kind == "observation"
            and item.payload.get("status") == "complete"
            and item.payload.get("control") is None
        )
        assert len(evaluation.outputs) <= plugin.manifest.maximum_outputs_per_batch
        assert not evaluation.interests
        state = evaluation.state
        retained = evaluation.retained_input_event_ids

    assert continuation_calls == len(range(6, 35, 2))
    assert maximum_outputs == 100
    assert len(primary_outputs) == len(COHORT) * len(range(6, 35, 2))
    assert {item.payload["checkpoint"] for item in primary_outputs} == set(range(6, 35, 2))
    assert evaluation.continuation is None
    assert len(retained) == 81
    assert len(evaluation.state_json()) < 65_536


@pytest.mark.parametrize("market_first", (False, True))
def test_frozen_m1c_prefix_roots_survive_cross_instrument_arrival_skew(
    market_first: bool,
) -> None:
    session = "2026-08-10"
    stock_events = tuple(
        _event(
            f"{symbol}-{checkpoint}",
            symbol,
            "bar_5m_session_prefix",
            checkpoint * 1_000_000,
            _prefix_fixture(session=session, checkpoint=checkpoint),
        )
        for symbol in COHORT
        for checkpoint in (6, 8)
    )
    market_events = tuple(
        _event(
            f"VTI-{checkpoint}",
            "VTI",
            "bar_5m_session_prefix",
            checkpoint * 1_000_000,
            _prefix_fixture(session=session, checkpoint=checkpoint, base=250.0),
        )
        for checkpoint in (6, 8)
    )
    events = (*market_events, *stock_events) if market_first else (*stock_events, *market_events)

    evaluation = FrozenM1CSignalV0().evaluate(
        IdeaBatch(
            mode=RuntimeMode.SHADOW,
            events=events,
            input_watermark=events[-1].event_id,
            causal_from_at_us=min(item.event_at_us for item in events),
            causal_through_at_us=max(item.event_at_us for item in events),
        ),
        {},
    )

    assert evaluation.outputs == ()
    assert isinstance(evaluation.continuation, AncestorPageContinuation)
    roots = cast(Mapping[str, Mapping[str, JsonValue]], evaluation.state)["prefix_roots"]
    assert set(roots) == set(UNIVERSE)
    assert {root["n"] for root in roots.values()} == {8}


def test_frozen_m1c_strict_prefix_makes_multi_session_batches_partition_invariant() -> None:
    plugin = FrozenM1CSignalV0()

    def session_events(session: str, session_index: int) -> tuple[MarketEvent, ...]:
        return tuple(
            _event(
                f"{session}-{checkpoint:02d}-{symbol}",
                symbol,
                "bar_5m_session_prefix",
                (session_index * 100 + checkpoint) * 1_000_000,
                _prefix_fixture(
                    session=session,
                    checkpoint=checkpoint,
                    base=250.0 if symbol == "VTI" else 100.0,
                ),
            )
            for checkpoint in range(1, 7)
            for symbol in UNIVERSE
        )

    first_session = session_events("2026-08-10", 1)
    second_session = session_events("2026-08-11", 2)
    two_sessions = (*first_session, *second_session)
    assert len(two_sessions) == 252

    def run(
        all_events: tuple[MarketEvent, ...],
        candidate_sizes: tuple[int, ...],
    ) -> tuple[JsonValue, tuple[str, ...], tuple[str, ...]]:
        state: JsonValue = {}
        retained: tuple[str, ...] = ()
        event_offset = 0
        output_ids: list[str] = []
        for candidate_size in candidate_sizes:
            remaining = all_events[event_offset : event_offset + candidate_size]
            while remaining:
                candidate = IdeaBatch(
                    mode=RuntimeMode.SHADOW,
                    events=remaining,
                    input_watermark=remaining[-1].event_id,
                    causal_from_at_us=min(item.event_at_us for item in remaining),
                    causal_through_at_us=max(item.event_at_us for item in remaining),
                    prior_state_input_event_ids=retained,
                )
                selected = plugin.select_input_prefix(candidate, state)
                ordinary_events = remaining[:selected]
                evaluation = plugin.evaluate(
                    IdeaBatch(
                        mode=RuntimeMode.SHADOW,
                        events=ordinary_events,
                        input_watermark=ordinary_events[-1].event_id,
                        causal_from_at_us=min(item.event_at_us for item in ordinary_events),
                        causal_through_at_us=max(item.event_at_us for item in ordinary_events),
                        prior_state_input_event_ids=retained,
                    ),
                    state,
                )
                state = evaluation.state
                retained = evaluation.retained_input_event_ids
                request = cast(AncestorPageContinuation, evaluation.continuation)
                ancestry = tuple(
                    sorted(ordinary_events, key=lambda item: (item.event_at_us, item.event_id))
                )
                for page_start in range(0, len(ancestry), 64):
                    page = ancestry[page_start : page_start + 64]
                    next_token = (
                        f"fixture-{page_start + len(page)}"
                        if page_start + len(page) < len(ancestry)
                        else None
                    )
                    evaluation = plugin.evaluate(
                        IdeaBatch(
                            mode=RuntimeMode.SHADOW,
                            rehydrated_events=page,
                            continuation_request=request,
                            continuation_token=next_token,
                            input_watermark=ordinary_events[-1].event_id,
                            causal_from_at_us=min(item.event_at_us for item in page),
                            causal_through_at_us=max(item.event_at_us for item in page),
                            prior_state_input_event_ids=retained,
                        ),
                        state,
                    )
                    state = evaluation.state
                    retained = evaluation.retained_input_event_ids
                    request = cast(AncestorPageContinuation, evaluation.continuation)
                exact_request = cast(ExactEventsContinuation, evaluation.continuation)
                by_id = {event.event_id: event for event in ordinary_events}
                exact_events = tuple(by_id[event_id] for event_id in exact_request.event_ids)
                evaluation = plugin.evaluate(
                    _exact_batch(
                        events=exact_events,
                        request=exact_request,
                        retained=retained,
                    ),
                    state,
                )
                for ordinal, (output, lineage) in enumerate(
                    zip(evaluation.outputs, evaluation.output_input_event_ids, strict=True)
                ):
                    output_ids.append(
                        deterministic_idea_output_id(
                            instance_id="partition-invariant-instance",
                            input_event_ids=lineage,
                            output_kind=output.kind,
                            output_ordinal=ordinal,
                            as_of_at_us=output.as_of_at_us,
                            payload=cast(JsonValue, output.payload),
                        )
                    )
                state = evaluation.state
                retained = evaluation.retained_input_event_ids
                assert evaluation.continuation is None
                event_offset += selected
                remaining = remaining[selected:]
        assert event_offset == len(all_events)
        return state, retained, tuple(output_ids)

    combined = run(two_sessions, (252,))
    split = run(two_sessions, (126, 126))
    assert combined == split
    assert len(combined[2]) == 40
    assert len(set(combined[2])) == 40

    sparse_sessions = tuple(
        event
        for session, session_index in (
            ("2026-08-10", 1),
            ("2026-08-11", 2),
            ("2026-08-12", 3),
        )
        for event in session_events(session, session_index)
        if event.payload["bar_number"] == 6
    )
    assert len(sparse_sessions) == 63
    sparse_combined = run(sparse_sessions, (63,))
    sparse_split = run(sparse_sessions, (21, 21, 21))
    assert sparse_combined == sparse_split
    assert len(sparse_combined[2]) == 60
    assert len(set(sparse_combined[2])) == 60


def test_frozen_m1c_strict_prefix_hides_newer_context_until_continuation_commits() -> None:
    plugin = FrozenM1CSignalV0()
    state, retained = _full_cohort_context_state()
    current_session = "2026-08-10"
    prefix_events = tuple(
        _event(
            f"{current_session}-{checkpoint:02d}-{symbol}",
            symbol,
            "bar_5m_session_prefix",
            checkpoint * 1_000_000,
            _prefix_fixture(
                session=current_session,
                checkpoint=checkpoint,
                base=250.0 if symbol == "VTI" else 100.0,
            ),
        )
        for checkpoint in range(1, 7)
        for symbol in UNIVERSE
    )
    newer_baseline = _event(
        "newer-baseline-AAL",
        "AAL",
        "session_volume_baseline",
        7_000_000,
        {
            "session": "2026-08-09",
            "complete_session_count": 22,
            "session_closes": [100.0, 101.0],
            "realised_volatility_20d": 0.20,
        },
    )
    candidate_events = (*prefix_events, newer_baseline)
    candidate = IdeaBatch(
        mode=RuntimeMode.SHADOW,
        events=candidate_events,
        input_watermark=newer_baseline.event_id,
        causal_from_at_us=prefix_events[0].event_at_us,
        causal_through_at_us=newer_baseline.event_at_us,
        prior_state_input_event_ids=retained,
    )

    selected = plugin.select_input_prefix(candidate, state)

    assert selected == len(prefix_events)
    evaluation = plugin.evaluate(
        IdeaBatch(
            mode=RuntimeMode.SHADOW,
            events=prefix_events,
            input_watermark=prefix_events[-1].event_id,
            causal_from_at_us=prefix_events[0].event_at_us,
            causal_through_at_us=prefix_events[-1].event_at_us,
            prior_state_input_event_ids=retained,
        ),
        state,
    )
    assert isinstance(evaluation.continuation, AncestorPageContinuation)
    baselines = cast(Mapping[str, Mapping[str, JsonValue]], evaluation.state)["baselines"]
    assert baselines["AAL"]["i"] == "baseline-AAL"
    assert "newer-baseline-AAL" not in evaluation.retained_input_event_ids


def test_frozen_m1c_source_graph_includes_reviewed_generated_data() -> None:
    assert len(reviewed_code_hash("stocker_ideas.plugins.frozen_m1c_signal_v0")) == 64


def test_frozen_m1c_starts_through_isolated_reviewed_discovery() -> None:
    module = "stocker_ideas.plugins.frozen_m1c_signal_v0"
    parameters = cast(Mapping[str, JsonValue], m1c_plugin._PARAMETERS)
    discovered = discover_plugins(
        (
            IdeaConfig(
                module=module,
                expected_code_hash=reviewed_code_hash(module),
                expected_manifest_hash=hashlib.sha256(
                    m1c_plugin.MANIFEST.to_canonical_json()
                ).hexdigest(),
                parameters=parameters,
                universe=UNIVERSE,
                enabled=True,
            ),
        )
    )

    assert len(discovered) == 1
    assert discovered[0].manifest.idea_id == "frozen_m1c_signal"
    assert len(discovered[0].requirements) == 41
