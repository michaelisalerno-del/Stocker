from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

import stocker_runtime.ideas.discovery as discovery_module
from stocker_ideas.plugins.opening_leader_continuation_v0 import COHORT, MANIFEST, create_plugin
from stocker_runtime.domain import (
    JsonValue,
    MarketEvent,
    Observation,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)
from stocker_runtime.ideas.contract import (
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    MarketDataRequirement,
)
from stocker_runtime.ideas.discovery import IdeaConfig, discover_plugins, reviewed_code_hash
from stocker_runtime.ideas.runner import IdeaRunner, IdeaRunnerError, _PluginWorker
from stocker_runtime.ingestion.bar_projection import project_required_five_minute_bars
from stocker_runtime.storage.connection import connect_v2, initialize_database
from stocker_runtime.storage.retention import RetentionManager, RetentionPolicy


def _seed(database: Path) -> int:
    initialize_database(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES "
            "('run-1', 'prospective_record', 'ibkr', 1, NULL, ?, 'fixture', "
            "'prospective_protected', 'running', NULL)",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-1', 1, 'fixture', 1)"
        )
        for symbol in COHORT:
            connection.execute(
                "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, "
                "currency) VALUES (?, ?, 'stock', ?, 'SMART', 'USD')",
                (symbol, hashlib.sha256(symbol.encode()).hexdigest(), symbol),
            )
    return int(datetime(2026, 8, 3, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)


def _bar(
    connection: object,
    sequence: int,
    event_at_us: int,
    *,
    received_at_us: int,
    payload_overrides: dict[str, JsonValue] | None = None,
    event_id: str | None = None,
) -> None:
    payload: dict[str, JsonValue] = {
        "event_at_us": event_at_us,
        "open": 100.0 + sequence / 1_000,
        "high": 101.0 + sequence / 1_000,
        "low": 99.0 + sequence / 1_000,
        "close": 100.5 + sequence / 1_000,
        "volume": 1.0,
    }
    payload.update(payload_overrides or {})
    payload_json = canonical_json_bytes(payload).decode()
    event_id = f"raw-{sequence}" if event_id is None else event_id
    connection.execute(  # type: ignore[attr-defined]
        "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
        "recorder_generation, connection_generation, callback_kind, received_at_us, "
        "payload_sha256, lifecycle) VALUES (?, ?, 'run-1', 1, 1, 'bar', ?, ?, 'pending')",
        (sequence, event_id, received_at_us, hashlib.sha256(payload_json.encode()).hexdigest()),
    )
    connection.execute(  # type: ignore[attr-defined]
        "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
        "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
        "payload_json, payload_sha256) VALUES (?, 'run-1', ?, 'AAL', 'bars', 'bar', "
        "?, ?, 1, ?, ?)",
        (
            event_id,
            sequence,
            event_at_us,
            received_at_us,
            payload_json,
            hashlib.sha256(payload_json.encode()).hexdigest(),
        ),
    )


def _requirement() -> MarketDataRequirement:
    return MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar_5m",
        cadence="5m",
        gaps_block=True,
        staleness_block=True,
    )


def _project(database: Path) -> tuple[sqlite3.Row, tuple[sqlite3.Row, ...]]:
    with connect_v2(database) as connection:
        project_required_five_minute_bars(
            connection, run_id="run-1", requirements=(_requirement(),)
        )
        event = connection.execute(
            "SELECT * FROM market_events WHERE event_kind='bar_5m'"
        ).fetchone()
        mappings = tuple(
            connection.execute(
                "SELECT input_ordinal, input_event_id, input_role "
                "FROM market_event_derivations ORDER BY input_ordinal"
            )
        )
    assert event is not None
    return event, mappings


def test_exact_sixty_raw_bars_produce_complete_restart_safe_receipt(tmp_path: Path) -> None:
    database = tmp_path / "complete.sqlite3"
    opening = _seed(database)
    end = opening + 300_000_000
    with connect_v2(database) as connection:
        for index in range(60):
            _bar(
                connection,
                index + 1,
                opening + index * 5_000_000,
                received_at_us=end if index == 59 else opening + index * 5_000_000,
            )
    event, mappings = _project(database)
    payload = json.loads(str(event["payload_json"]))
    assert event["source_sequence"] is None
    assert event["derived_after_source_sequence"] == 60
    assert payload["source_completeness"] == "complete"
    assert payload["input_count"] == 60
    assert len(mappings) == 60
    assert [tuple(row) for row in mappings] == [
        (index, f"raw-{index + 1}", "constituent") for index in range(60)
    ]

    # Replaying projection after a process restart is an immutable no-op.
    event_again, mappings_again = _project(database)
    assert event_again["event_id"] == event["event_id"]
    assert mappings_again == mappings


def test_projection_is_identical_for_reverse_source_arrival(tmp_path: Path) -> None:
    chronological = tmp_path / "chronological.sqlite3"
    reverse = tmp_path / "reverse.sqlite3"
    mixed = tmp_path / "mixed.sqlite3"
    opening = _seed(chronological)
    _seed(reverse)
    _seed(mixed)
    end = opening + 300_000_000
    for database, arrival_order in (
        (chronological, tuple(range(60))),
        (reverse, tuple(reversed(range(60)))),
        (mixed, (*range(0, 60, 2), *range(1, 60, 2))),
    ):
        with connect_v2(database) as connection:
            for source_sequence, chronological_index in enumerate(arrival_order, start=1):
                event_at_us = opening + chronological_index * 5_000_000
                _bar(
                    connection,
                    source_sequence,
                    event_at_us,
                    received_at_us=(
                        end + chronological_index if chronological_index in {0, 59} else event_at_us
                    ),
                    event_id=f"raw-at-{chronological_index}",
                    payload_overrides={
                        "open": 100.0 + chronological_index,
                        "high": 101.0 + chronological_index,
                        "low": 99.0 + chronological_index,
                        "close": 100.5 + chronological_index,
                        "volume": float(chronological_index + 1),
                    },
                )

    first, first_mappings = _project(chronological)
    second, second_mappings = _project(reverse)
    third, third_mappings = _project(mixed)
    first_payload = json.loads(str(first["payload_json"]))
    second_payload = json.loads(str(second["payload_json"]))
    third_payload = json.loads(str(third["payload_json"]))
    for payload in (first_payload, second_payload, third_payload):
        assert payload["source_completeness"] == "complete"
        assert payload["open"] == 100.0
        assert payload["high"] == 160.0
        assert payload["low"] == 99.0
        assert payload["close"] == 159.5
        assert payload["volume"] == 1_830.0
        assert payload["derived_after_source_sequence"] == 60
    assert (
        len(
            {
                first_payload["input_ids_hash"],
                second_payload["input_ids_hash"],
                third_payload["input_ids_hash"],
            }
        )
        == 1
    )
    assert first["event_id"] == second["event_id"] == third["event_id"]
    assert first["received_at_us"] == second["received_at_us"] == third["received_at_us"]
    assert first["received_at_us"] == end + 59
    expected_mappings = [tuple(row) for row in first_mappings]
    assert [tuple(row) for row in second_mappings] == expected_mappings
    assert [tuple(row) for row in third_mappings] == expected_mappings


def test_one_missing_constituent_is_permanently_incomplete(tmp_path: Path) -> None:
    database = tmp_path / "missing.sqlite3"
    opening = _seed(database)
    end = opening + 300_000_000
    with connect_v2(database) as connection:
        sequence = 1
        for index in range(60):
            if index == 17:
                continue
            _bar(
                connection,
                sequence,
                opening + index * 5_000_000,
                received_at_us=opening + index * 5_000_000,
            )
            sequence += 1
        _bar(connection, sequence, end, received_at_us=end)
    event, mappings = _project(database)
    payload = json.loads(str(event["payload_json"]))
    assert payload["source_completeness"] == "incomplete"
    assert payload["missing_count"] == 1
    assert len(mappings) == 60
    assert tuple(mappings[-1]) == (59, f"raw-{sequence}", "progress")
    with connect_v2(database) as connection:
        _bar(
            connection,
            sequence + 1,
            opening + 17 * 5_000_000,
            received_at_us=end + 1,
        )
        assert (
            project_required_five_minute_bars(
                connection, run_id="run-1", requirements=(_requirement(),)
            )
            == 0
        )
        unchanged = connection.execute(
            "SELECT event_id, payload_json FROM market_events WHERE event_kind='bar_5m'"
        ).fetchone()
    assert unchanged["event_id"] == event["event_id"]
    assert json.loads(str(unchanged["payload_json"]))["source_completeness"] == "incomplete"


def _assert_incomplete_receipt_blocks_opening_leader(event: sqlite3.Row) -> None:
    projected = MarketEvent(
        event_id=str(event["event_id"]),
        instrument_id=str(event["instrument_id"]),
        feed_kind=str(event["feed_kind"]),
        event_kind=str(event["event_kind"]),
        event_at_us=int(event["event_at_us"]),
        received_at_us=int(event["received_at_us"]),
        payload=json.loads(str(event["payload_json"])),
    )
    evidence = tuple(
        projected
        if symbol == "AAL" and number == 1
        else _derived_bar(symbol, number, complete=COHORT.index(symbol) < 15)
        for number in range(1, 7)
        for symbol in COHORT
    )
    assert create_plugin().evaluate(_idea_batch(evidence), {}).outputs == ()


def test_duplicate_raw_timestamp_produces_immutable_incomplete_receipt(
    tmp_path: Path,
) -> None:
    database = tmp_path / "duplicate-time.sqlite3"
    opening = _seed(database)
    end = opening + 300_000_000
    with connect_v2(database) as connection:
        for index in range(60):
            event_at_us = opening + index * 5_000_000
            if index == 17:
                event_at_us -= 5_000_000
            _bar(
                connection,
                index + 1,
                event_at_us,
                received_at_us=end if index == 59 else event_at_us,
            )

    event, mappings = _project(database)
    payload = json.loads(str(event["payload_json"]))
    assert payload["source_completeness"] == "incomplete"
    assert payload["input_count"] == 60
    assert payload["missing_count"] == 1
    assert payload["duplicate_count"] == 1
    assert payload["unexpected_count"] == 0
    assert len(mappings) == 60
    replay, replay_mappings = _project(database)
    assert replay["event_id"] == event["event_id"]
    assert replay_mappings == mappings
    _assert_incomplete_receipt_blocks_opening_leader(event)


def test_duplicate_rich_incomplete_receipt_maps_every_unique_input(
    tmp_path: Path,
) -> None:
    database = tmp_path / "duplicate-rich.sqlite3"
    opening = _seed(database)
    end = opening + 300_000_000
    constituent_count = 65
    with connect_v2(database) as connection:
        for index in range(constituent_count):
            chronological_index = min(index, 59)
            _bar(
                connection,
                index + 1,
                opening + chronological_index * 5_000_000,
                received_at_us=opening + chronological_index * 5_000_000,
                event_id=f"duplicate-rich-{index:03d}",
            )
        _bar(
            connection,
            constituent_count + 1,
            end,
            received_at_us=end,
            event_id="duplicate-rich-progress",
        )

    event, mappings = _project(database)
    payload = json.loads(str(event["payload_json"]))
    assert payload["source_completeness"] == "incomplete"
    assert payload["input_count"] == constituent_count
    assert payload["duplicate_count"] == constituent_count - 60
    assert len(mappings) == constituent_count + 1
    assert len({row["input_event_id"] for row in mappings}) == constituent_count + 1
    assert tuple(mappings[-1]) == (
        constituent_count,
        "duplicate-rich-progress",
        "progress",
    )
    replay, replay_mappings = _project(database)
    assert replay["event_id"] == event["event_id"]
    assert replay_mappings == mappings

    RetentionManager(
        database,
        RetentionPolicy(
            raw_market_event_us=1,
            derivation_mapping_us=1,
            completed_bar_us=10_000_000_000,
        ),
    ).run(now_us=end + 2)
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT count(*) FROM market_event_derivations").fetchone()[0] == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_kind='bar'"
            ).fetchone()[0]
            == 0
        )


def test_malformed_numeric_payload_produces_immutable_incomplete_receipt(
    tmp_path: Path,
) -> None:
    database = tmp_path / "malformed-numeric.sqlite3"
    opening = _seed(database)
    end = opening + 300_000_000
    with connect_v2(database) as connection:
        for index in range(60):
            _bar(
                connection,
                index + 1,
                opening + index * 5_000_000,
                received_at_us=end if index == 59 else opening + index * 5_000_000,
                payload_overrides={"open": "NaN"} if index == 17 else None,
            )

    event, mappings = _project(database)
    payload = json.loads(str(event["payload_json"]))
    assert payload["source_completeness"] == "incomplete"
    assert payload["input_count"] == 60
    assert payload["missing_count"] == 0
    assert payload["duplicate_count"] == 0
    assert payload["unexpected_count"] == 0
    assert payload["invalid_numeric_payload"] is True
    assert len(mappings) == 60
    replay, replay_mappings = _project(database)
    assert replay["event_id"] == event["event_id"]
    assert replay_mappings == mappings
    _assert_incomplete_receipt_blocks_opening_leader(event)


def test_incremental_projection_reads_only_after_last_immutable_window(tmp_path: Path) -> None:
    database = tmp_path / "incremental.sqlite3"
    opening = _seed(database)
    first_end = opening + 300_000_000
    second_end = first_end + 300_000_000
    with connect_v2(database) as connection:
        for index in range(60):
            _bar(
                connection,
                index + 1,
                opening + index * 5_000_000,
                received_at_us=first_end if index == 59 else opening + index * 5_000_000,
            )
    first, _ = _project(database)
    with connect_v2(database) as connection:
        for index in range(60):
            _bar(
                connection,
                index + 61,
                first_end + index * 5_000_000,
                received_at_us=(second_end if index == 59 else first_end + index * 5_000_000),
            )
        assert (
            project_required_five_minute_bars(
                connection, run_id="run-1", requirements=(_requirement(),)
            )
            == 1
        )
        derived = connection.execute(
            "SELECT event_id, event_at_us, payload_json FROM market_events "
            "WHERE event_kind='bar_5m' ORDER BY event_at_us"
        ).fetchall()
        second_inputs = tuple(
            row[0]
            for row in connection.execute(
                "SELECT input_event_id FROM market_event_derivations "
                "WHERE derived_event_id=? ORDER BY input_ordinal",
                (derived[1]["event_id"],),
            )
        )
    assert len(derived) == 2
    assert derived[0]["event_id"] == first["event_id"]
    assert json.loads(str(derived[1]["payload_json"]))["source_completeness"] == "complete"
    assert second_inputs == tuple(f"raw-{sequence}" for sequence in range(61, 121))


def test_resolved_overlapping_gap_still_invalidates_bar(tmp_path: Path) -> None:
    database = tmp_path / "resolved-gap.sqlite3"
    opening = _seed(database)
    end = opening + 300_000_000
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES "
            "('sub-1', 'run-1', 1, 1, 'AAL', 'bars', 1, 'active', ?, ?)",
            ("b" * 64, opening),
        )
        connection.execute(
            "INSERT INTO gaps(gap_id, run_id, subscription_id, started_at_us, ended_at_us, "
            "reason, data_loss_possible, continuity_required, resolved_at_us) "
            "VALUES ('gap-1', 'run-1', 'sub-1', ?, ?, 'DISCONNECT', 1, 1, ?)",
            (opening + 10_000_000, opening + 20_000_000, opening + 21_000_000),
        )
        for index in range(60):
            _bar(
                connection,
                index + 1,
                opening + index * 5_000_000,
                received_at_us=end if index == 59 else opening + index * 5_000_000,
            )
    event, _ = _project(database)
    payload = json.loads(str(event["payload_json"]))
    assert payload["source_completeness"] == "incomplete"
    assert payload["overlapping_gap_count"] == 1


def test_derivation_mappings_protect_raw_until_their_thirty_day_tier_expires(
    tmp_path: Path,
) -> None:
    database = tmp_path / "mapping-retention.sqlite3"
    opening = _seed(database)
    end = opening + 300_000_000
    with connect_v2(database) as connection:
        for index in range(60):
            _bar(
                connection,
                index + 1,
                opening + index * 5_000_000,
                received_at_us=end if index == 59 else opening + index * 5_000_000,
            )
    _project(database)
    RetentionManager(
        database,
        RetentionPolicy(
            raw_market_event_us=1,
            derivation_mapping_us=1,
            completed_bar_us=10_000_000_000,
        ),
    ).run(now_us=end + 2)
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT count(*) FROM market_event_derivations").fetchone()[0] == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_kind='bar'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM market_events WHERE event_kind='bar_5m'"
            ).fetchone()[0]
            == 1
        )


def test_raw_source_sequence_remains_unique(tmp_path: Path) -> None:
    database = tmp_path / "unique.sqlite3"
    opening = _seed(database)
    with connect_v2(database) as connection:
        _bar(connection, 1, opening, received_at_us=opening)
        payload_json = canonical_json_bytes({"event_at_us": opening}).decode()
        with pytest.raises(Exception, match="UNIQUE"):
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                "payload_json, payload_sha256) VALUES ('duplicate', 'run-1', 1, 'AAL', "
                "'bars', 'bar', ?, ?, 1, ?, ?)",
                (
                    opening,
                    opening,
                    payload_json,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                ),
            )


def _derived_bar(symbol: str, number: int, *, complete: bool = True) -> MarketEvent:
    opening = int(datetime(2026, 8, 3, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    end = opening + number * 300_000_000
    payload: dict[str, JsonValue] = {
        "session": "2026-08-03",
        "bar_number": number,
        "bar_start_at_us": end - 300_000_000,
        "bar_end_at_us": end,
        "source_completeness": "complete" if complete else "incomplete",
    }
    if complete:
        payload.update(
            {
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 100.0 + COHORT.index(symbol) / 100 + number / 100,
                "volume": 60.0,
            }
        )
    return MarketEvent(
        event_id=f"derived-{symbol}-{number}",
        instrument_id=symbol,
        feed_kind="bars",
        event_kind="bar_5m",
        event_at_us=end,
        received_at_us=end,
        payload=payload,
    )


def _idea_batch(events: tuple[MarketEvent, ...], prior: tuple[str, ...] = ()) -> IdeaBatch:
    return IdeaBatch(
        mode=RuntimeMode.PROSPECTIVE_RECORD,
        events=events,
        input_watermark=events[-1].event_id,
        causal_from_at_us=min(event.event_at_us for event in events),
        causal_through_at_us=max(event.received_at_us for event in events),
        prior_state_input_event_ids=prior,
    )


def test_opening_leader_c6_c12_lineage_and_state_are_bounded() -> None:
    plugin = create_plugin()
    first_events = tuple(
        _derived_bar(symbol, number) for number in range(1, 7) for symbol in COHORT
    )
    c6 = plugin.evaluate(_idea_batch(first_events), {})
    assert len(c6.outputs) == 3
    assert len(c6.retained_input_event_ids) == 120
    assert {len(lineage) for lineage in c6.output_input_event_ids} == {120}
    assert len(canonical_json_bytes(c6.state)) < 65_536

    second_events = tuple(
        _derived_bar(symbol, number) for number in range(7, 13) for symbol in COHORT
    )
    c12 = plugin.evaluate(_idea_batch(second_events, c6.retained_input_event_ids), c6.state)
    assert len(c12.outputs) == 3
    assert {len(lineage) for lineage in c12.output_input_event_ids} == {240}
    assert c12.retained_input_event_ids == ()
    assert len(canonical_json_bytes(c12.state)) < 65_536


@pytest.mark.parametrize(("valid_symbols", "outputs"), ((15, 3), (14, 0)))
def test_opening_leader_exact_bar_completeness_enforces_minimum_slate(
    valid_symbols: int, outputs: int
) -> None:
    plugin = create_plugin()
    events = tuple(
        _derived_bar(symbol, number, complete=COHORT.index(symbol) < valid_symbols)
        for number in range(1, 7)
        for symbol in COHORT
    )
    evaluation = plugin.evaluate(_idea_batch(events), {})
    assert len(evaluation.outputs) == outputs


def _config() -> IdeaConfig:
    module = "stocker_ideas.plugins.opening_leader_continuation_v0"
    return IdeaConfig(
        module=module,
        factory="create_plugin",
        expected_code_hash=reviewed_code_hash(module),
        expected_manifest_hash=hashlib.sha256(MANIFEST.to_canonical_json()).hexdigest(),
        parameters={"checkpoints": (6, 12), "minimum_complete_slate": 15},
        universe=COHORT,
        enabled=True,
    )


@pytest.mark.parametrize(
    ("table", "column", "replacement"),
    (
        ("idea_instances", "parameters_json", '{"checkpoints":[6,12],"minimum_complete_slate":14}'),
        ("idea_instances", "universe_json", '["WULF"]'),
        ("idea_instances", "requirements_json", "[]"),
        ("idea_checkpoints", "state_json", '{"corrupt":true}'),
        ("idea_plugins", "manifest_json", '{"api_version":1}'),
    ),
)
def test_restart_rejects_valid_json_with_broken_identity_binding(
    tmp_path: Path, table: str, column: str, replacement: str
) -> None:
    database = tmp_path / f"corrupt-{column}.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    runner.close()
    with connect_v2(database) as connection:
        connection.execute(f"UPDATE {table} SET {column}=?", (replacement,))
    with pytest.raises(IdeaRunnerError):
        IdeaRunner(database, (discovered,), run_id="run-1")


def _hanging_discovery_worker(*_args: object) -> None:
    time.sleep(10)


def _crashing_discovery_worker(*_args: object) -> None:
    os._exit(19)


@pytest.mark.parametrize("worker", (_hanging_discovery_worker, _crashing_discovery_worker))
def test_discovery_worker_hang_or_crash_is_bounded_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, worker: object
) -> None:
    config = _config()
    parameters_hash = hashlib.sha256(canonical_json_bytes(config.parameters)).hexdigest()
    provisional = IdeaActivation(
        instance_id="discovery",
        parameters=config.parameters,
        parameters_hash=parameters_hash,
        plugin_code_hash=config.expected_code_hash,
        activated_at_us=0,
        run_id="discovery",
        protected_data_class=ProtectedDataClass.PROSPECTIVE,
        universe=config.universe,
    )
    monkeypatch.setattr(discovery_module, "_discovery_worker", worker)
    monkeypatch.setattr(discovery_module, "DISCOVERY_SECONDS", 0.05)
    started = time.monotonic()
    with pytest.raises(discovery_module.IdeaDiscoveryError):
        discovery_module._load_plugin_isolated(config, provisional)
    assert time.monotonic() - started < 0.5


class _BlockedConnection:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate

    def send(self, _value: object) -> None:
        time.sleep(1)

    def close(self) -> None:
        self.delegate.close()  # type: ignore[attr-defined]


def test_request_serialization_delivery_is_inside_fifty_millisecond_bound() -> None:
    worker = _PluginWorker(create_plugin())
    worker.connection = _BlockedConnection(worker.connection)  # type: ignore[assignment]
    batch = _idea_batch((_derived_bar("AAL", 1),))
    started = time.monotonic()
    with pytest.raises(IdeaRunnerError, match="during send"):
        worker.evaluate(batch, {})
    assert time.monotonic() - started < 0.5

    healthy = _PluginWorker(create_plugin())
    assert healthy.evaluate(batch, {}).outputs == ()
    healthy.terminate()


def test_output_causality_checks_exact_current_and_prior_lineage(tmp_path: Path) -> None:
    database = tmp_path / "causality.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation_result = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    activation = discovered.activation(
        instance_id=activation_result.instance_id,
        run_id="run-1",
        data_class=ProtectedDataClass.PROSPECTIVE,
        activated_at_us=100,
    )
    prior = _derived_bar("AAL", 1)
    current = _derived_bar("AAL", 2)
    with connect_v2(database) as connection:
        for sequence, event in enumerate((prior, current), start=1):
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, "
                "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
                "event_at_us, received_at_us, connection_generation, payload_json, "
                "payload_sha256) VALUES (?, 'run-1', NULL, ?, 'AAL', 'bars', 'bar_5m', "
                "?, ?, 1, ?, ?)",
                (
                    event.event_id,
                    sequence,
                    event.event_at_us,
                    event.received_at_us,
                    canonical_json_bytes(event.payload).decode(),
                    hashlib.sha256(canonical_json_bytes(event.payload)).hexdigest(),
                ),
            )
    batch = _idea_batch((current,), (prior.event_id,))
    valid = IdeaEvaluation(
        state={},
        outputs=(
            Observation(subject_instrument_id="AAL", as_of_at_us=prior.event_at_us, payload={}),
        ),
        output_input_event_ids=((prior.event_id,),),
    )
    runner._validate_evaluation(activation, discovered.plugin, batch, valid)
    late = IdeaEvaluation(
        state={},
        outputs=(
            Observation(
                subject_instrument_id="AAL", as_of_at_us=current.event_at_us - 1, payload={}
            ),
        ),
        output_input_event_ids=((current.event_id,),),
    )
    with pytest.raises(IdeaRunnerError, match="as-of"):
        runner._validate_evaluation(activation, discovered.plugin, batch, late)
