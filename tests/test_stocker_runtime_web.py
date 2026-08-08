"""Phase 6 tests for the bounded generic Stocker V2 web surface."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from stocker_runtime.cli import app as runtime_cli
from stocker_runtime.storage import connect_v2, create_backup, initialize_database
from stocker_runtime.web import WebConfig, create_web_app
from stocker_runtime.web import queries as web_queries
from stocker_runtime.web.queries import (
    IDEA_WINDOW_US,
    RESULT_WINDOW_US,
    SQLITE_INTEGER_MAX,
    QueryTimeoutError,
    ReadModel,
    _encode_cursor,
)

EXPECTED_API_ROUTES = {
    "/api/v2/meta",
    "/api/v2/live",
    "/api/v2/ideas",
    "/api/v2/ideas/{instance_id}",
    "/api/v2/results",
    "/api/v2/results/{position_id}",
    "/api/v2/diagnostics",
}


def _config(database: Path, **changes: object) -> WebConfig:
    values: dict[str, object] = {
        "database": database,
        "production": True,
        "git_commit": "c4bb701",
        "config_hash": "a" * 64,
        "allowed_hosts": ["testserver"],
    }
    values.update(changes)
    return WebConfig.model_validate(values)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _seed_live(database: Path) -> None:
    initialize_database(database)
    payload = json.dumps({}, separators=(",", ":"))
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES ('run-live', 'prospective_record', 'ibkr', 100, NULL, ?, "
            "'c4bb701', 'prospective_protected', 'running', NULL)",
            ("b" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-live', 2, 'recorder', 100)"
        )
        connection.execute(
            "INSERT INTO runtime_state(run_id, recorder_generation, lifecycle, reason, "
            "process_heartbeat_at_us, callback_heartbeat_at_us, admission_heartbeat_at_us, "
            "projection_heartbeat_at_us, connection_state, connection_generation, "
            "inbox_nonterminal_count, inbox_bytes, database_bytes, wal_bytes) VALUES "
            "('run-live', 2, 'recording', NULL, 195, 196, 197, 198, 'connected', 4, "
            "3, 256, 4096, 512)"
        )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency) VALUES ('AAPL', ?, 265598, 'stock', 'AAPL', 'SMART', 'USD')",
            (_hash("AAPL"),),
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES "
            "('sub-aapl', 'run-live', 2, 4, 'AAPL', 'quotes', 7, 'active', ?, 100)",
            ("c" * 64,),
        )
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES "
            "(1, 'quote-aapl', 'run-live', 2, 4, 'quote', 190, ?, 'pending')",
            (_hash(payload),),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "bid_value, ask_value, last_value, payload_json, payload_sha256) VALUES "
            "('quote-aapl', 'run-live', 1, 'AAPL', 'quotes', 'quote', 189, 190, 4, "
            "101.25, 101.30, 101.27, ?, ?)",
            (payload, _hash(payload)),
        )
        connection.execute(
            "INSERT INTO market_latest(run_id, instrument_id, feed_kind, event_id, "
            "event_at_us, received_at_us, event_kind, quality_bits, bid_value, "
            "bid_source_event_id, ask_value, ask_source_event_id, last_value, "
            "last_source_event_id) VALUES "
            "('run-live', 'AAPL', 'quotes', 'quote-aapl', 189, 190, 'quote', 0, "
            "101.25, 'quote-aapl', 101.30, 'quote-aapl', 101.27, 'quote-aapl')"
        )


def _seed_ideas(database: Path) -> None:
    _seed_live(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO idea_plugins VALUES "
            "('unknown_alpha', '2026.1', 1, 'Unknown Alpha', 'Generic synthetic plugin', "
            "?, ?, ?, 150)",
            ("d" * 64, "e" * 64, '{"output_kinds":["observation","signal"]}'),
        )
        connection.execute(
            "INSERT INTO idea_plugins VALUES "
            "('discovered_only', '1', 1, 'Discovered only', 'Not activated', ?, ?, '{}', 140)",
            ("f" * 64, "1" * 64),
        )
        for instance_id, activated_at_us, health in (
            ("instance-new", 180, "healthy"),
            ("instance-old", 170, "healthy"),
            ("instance-disabled", 160, "disabled"),
        ):
            connection.execute(
                "INSERT INTO idea_instances(instance_id, idea_id, idea_version, run_id, mode, "
                "parameters_json, parameters_hash, plugin_code_hash, manifest_hash, "
                "universe_json, universe_hash, requirements_json, requirements_hash, "
                "activated_after_source_sequence, activated_at_us, health, data_class) VALUES "
                "(?, 'unknown_alpha', '2026.1', 'run-live', 'prospective_record', '{}', ?, ?, ?, "
                "'[\"AAPL\"]', ?, '[]', ?, 1, ?, ?, 'prospective_protected')",
                (
                    instance_id,
                    _hash("{}"),
                    "e" * 64,
                    "d" * 64,
                    _hash('["AAPL"]'),
                    _hash("[]"),
                    activated_at_us,
                    health,
                ),
            )


def _seed_outputs(database: Path) -> None:
    _seed_ideas(database)
    records = (
        ("output-observation", "observation", 201, "recorded", {"novel": {"score": 3}}),
        ("output-signal", "signal", 202, "recorded", {"state": "watch"}),
        (
            "output-position",
            "proposed_position",
            203,
            "unapproved",
            {"target_weight": 0.1},
        ),
        ("output-trade", "proposed_trade", 204, "unapproved", {"thesis": "generic"}),
    )
    with connect_v2(database) as connection:
        for ordinal, (output_id, kind, as_of_at_us, authority, payload) in enumerate(records):
            payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            connection.execute(
                "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
                "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
                "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
                "payload_json, payload_hash, content_hash, data_class, authority_status) VALUES "
                "(?, 'run-live', 'instance-new', ?, 'AAPL', ?, ?, 'quote-aapl', "
                "'quote-aapl', 'quote-aapl', ?, ?, ?, ?, ?, 'prospective_protected', ?)",
                (
                    output_id,
                    kind,
                    as_of_at_us,
                    as_of_at_us,
                    _hash('["quote-aapl"]'),
                    ordinal,
                    payload_json,
                    _hash(payload_json),
                    _hash(output_id),
                    authority,
                ),
            )
            connection.execute(
                "INSERT INTO idea_output_inputs VALUES (?, 'quote-aapl', 0)",
                (output_id,),
            )
            if kind == "proposed_trade":
                connection.execute(
                    "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                    "target, quantity_value, currency) VALUES (?, 0, 'AAPL', 'buy', 'long', 1, "
                    "'USD')",
                    (output_id,),
                )
            connection.execute("INSERT INTO idea_output_seals VALUES (?)", (output_id,))


def _seed_results(database: Path) -> None:
    _seed_ideas(database)
    payload = "{}"
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES ('run-shadow', 'shadow', 'ibkr', 300, NULL, ?, "
            "'c4bb701', 'shadow_protected', 'running', NULL)",
            ("9" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-shadow', 1, 'shadow-recorder', 300)"
        )
        connection.execute(
            "INSERT INTO idea_instances(instance_id, idea_id, idea_version, run_id, mode, "
            "parameters_json, parameters_hash, plugin_code_hash, manifest_hash, universe_json, "
            "universe_hash, requirements_json, requirements_hash, activated_after_source_sequence, "
            "activated_at_us, health, data_class) VALUES "
            "('shadow-instance', 'unknown_alpha', '2026.1', 'run-shadow', 'shadow', '{}', ?, ?, ?, "
            "'[\"AAPL\"]', ?, '[]', ?, 9, 300, 'healthy', 'shadow_protected')",
            (
                _hash("{}"),
                "e" * 64,
                "d" * 64,
                _hash('["AAPL"]'),
                _hash("[]"),
            ),
        )
        for sequence, event_id, event_at_us, bid, ask in (
            (10, "shadow-input", 300, 100.0, 100.5),
            (11, "shadow-entry", 310, 101.0, 101.5),
            (12, "shadow-exit", 340, 105.0, 105.5),
        ):
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_sha256, lifecycle) VALUES (?, ?, 'run-shadow', 1, 1, 'quote', ?, ?, "
                "'pending')",
                (sequence, event_id, event_at_us, _hash(payload)),
            )
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                "bid_value, ask_value, payload_json, payload_sha256) VALUES "
                "(?, 'run-shadow', ?, 'AAPL', 'quotes', 'quote', ?, ?, 1, ?, ?, ?, ?)",
                (
                    event_id,
                    sequence,
                    event_at_us,
                    event_at_us,
                    bid,
                    ask,
                    payload,
                    _hash(payload),
                ),
            )
        positions = (
            ("position-closed", "proposal-closed", 304, "closed"),
            ("position-open", "proposal-open", 303, "open"),
            ("position-pending", "proposal-pending", 302, "pending"),
            ("position-invalid", "proposal-invalid", 301, "invalid"),
        )
        for ordinal, (position_id, proposal_id, as_of_at_us, lifecycle) in enumerate(positions):
            connection.execute(
                "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
                "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
                "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
                "payload_json, payload_hash, content_hash, data_class, authority_status) VALUES "
                "(?, 'run-shadow', 'shadow-instance', 'proposed_trade', 'AAPL', ?, ?, "
                "'shadow-input', 'shadow-input', 'shadow-input', ?, ?, '{}', ?, ?, "
                "'shadow_protected', 'unapproved')",
                (
                    proposal_id,
                    as_of_at_us,
                    as_of_at_us,
                    _hash('["shadow-input"]'),
                    ordinal,
                    _hash(payload),
                    _hash(proposal_id),
                ),
            )
            connection.execute(
                "INSERT INTO idea_output_inputs VALUES (?, 'shadow-input', 0)",
                (proposal_id,),
            )
            connection.execute(
                "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                "target, quantity_value, currency) VALUES (?, 0, 'AAPL', 'buy', 'long', 1, 'USD')",
                (proposal_id,),
            )
            connection.execute("INSERT INTO idea_output_seals VALUES (?)", (proposal_id,))
            opened = 310 if lifecycle in {"open", "closed"} else None
            closed = 340 if lifecycle == "closed" else None
            invalid_reason = "missing_quote" if lifecycle == "invalid" else None
            connection.execute(
                "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
                "instance_id, opened_at_us, closed_at_us, lifecycle, cost_model_id, fill_model_id, "
                "currency, invalid_reason, data_class) VALUES "
                "(?, ?, 'run-shadow', 'shadow-instance', ?, ?, ?, 'cost-v1', 'quote-v1', 'USD', "
                "?, 'shadow_protected')",
                (position_id, proposal_id, opened, closed, lifecycle, invalid_reason),
            )
        connection.execute(
            "INSERT INTO shadow_legs(position_id, leg_number, instrument_id, side, quantity, "
            "entry_market_event_id, entry_price, exit_market_event_id, exit_price, "
            "entry_bid_market_event_id, entry_ask_market_event_id, exit_bid_market_event_id, "
            "exit_ask_market_event_id) VALUES "
            "('position-closed', 0, 'AAPL', 'buy', 1, 'shadow-entry', 101.5, 'shadow-exit', 105, "
            "'shadow-entry', 'shadow-entry', 'shadow-exit', 'shadow-exit')"
        )
        connection.execute(
            "INSERT INTO shadow_marks VALUES "
            "('position-closed', 320, 103, 1.5, 1.4, 0.014, 0, '{}')"
        )
        connection.execute(
            "INSERT INTO shadow_marks VALUES "
            "('position-closed', 330, 104, 2.5, 2.4, 0.024, 0, '{}')"
        )
        connection.execute(
            "INSERT INTO shadow_outcomes VALUES "
            "('position-closed', 340, 'horizon', 3.5, 3.4, 0.034, 3.5, -0.5, 'complete', '{}')"
        )


def test_openapi_contains_exactly_seven_get_routes(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    client = TestClient(create_web_app(_config(database)))

    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) == EXPECTED_API_ROUTES
    assert all(set(operations) == {"get"} for operations in schema["paths"].values())


def test_live_reads_current_runtime_active_feeds_and_latest_values(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed_live(database)
    payload = TestClient(create_web_app(_config(database))).get("/api/v2/live").json()

    assert payload["run"] == {
        "run_id": "run-live",
        "mode": "prospective_record",
        "status": "running",
        "started_at_us": 100,
    }
    assert payload["recorder"]["lifecycle"] == "recording"
    assert payload["ibkr"] == {
        "connection_state": "connected",
        "connection_generation": 4,
        "freshness_at_us": 198,
    }
    assert payload["feeds"] == {"active": 1, "by_kind": {"quotes": 1}}
    assert payload["instruments"] == [
        {
            "instrument_id": "AAPL",
            "kind": "stock",
            "symbol": "AAPL",
            "exchange": "SMART",
            "currency": "USD",
            "feed_kind": "quotes",
            "event_id": "quote-aapl",
            "event_at_us": 189,
            "received_at_us": 190,
            "quality_bits": 0,
            "bid": 101.25,
            "ask": 101.3,
            "last": 101.27,
            "close": None,
            "size": None,
        }
    ]
    assert payload["callback_inbox"] == {"nonterminal": 3, "bytes": 256}
    assert payload["storage"]["reported_database_bytes"] == 4096
    assert payload["storage"]["reported_wal_bytes"] == 512
    assert payload["gaps"] == {"unresolved": 0, "data_loss_possible": 0}
    assert payload["backup"] == {"available": False, "entries": 0, "latest": None}
    assert "order_capability" not in json.dumps(payload)


def test_live_uses_the_configured_or_latest_shadow_recorder(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed_live(database)
    payload = "{}"
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES ('newer-shadow', 'shadow', 'ibkr', 900, NULL, ?, "
            "'c4bb701', 'shadow_protected', 'running', NULL)",
            ("8" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('newer-shadow', 1, 'shadow-recorder', 900)"
        )
        connection.execute(
            "INSERT INTO runtime_state(run_id, recorder_generation, lifecycle, reason, "
            "process_heartbeat_at_us, callback_heartbeat_at_us, admission_heartbeat_at_us, "
            "projection_heartbeat_at_us, connection_state, connection_generation, "
            "inbox_nonterminal_count, inbox_bytes, database_bytes, wal_bytes) VALUES "
            "('newer-shadow', 1, 'recording', NULL, 995, 996, 997, 998, 'connected', 8, "
            "1, 64, 8192, 1024)"
        )
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES "
            "('sub-shadow-aapl', 'newer-shadow', 1, 8, 'AAPL', 'quotes', 8, 'active', ?, 900)",
            ("7" * 64,),
        )
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES "
            "(2, 'quote-shadow-aapl', 'newer-shadow', 1, 8, 'quote', 990, ?, 'pending')",
            (_hash(payload),),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "bid_value, ask_value, last_value, payload_json, payload_sha256) VALUES "
            "('quote-shadow-aapl', 'newer-shadow', 2, 'AAPL', 'quotes', 'quote', 989, 990, 8, "
            "202.25, 202.30, 202.27, ?, ?)",
            (payload, _hash(payload)),
        )
        connection.execute(
            "DELETE FROM market_latest WHERE instrument_id = 'AAPL' AND feed_kind = 'quotes'"
        )
        connection.execute(
            "INSERT INTO market_latest(run_id, instrument_id, feed_kind, event_id, "
            "event_at_us, received_at_us, event_kind, quality_bits, bid_value, "
            "bid_source_event_id, ask_value, ask_source_event_id, last_value, "
            "last_source_event_id) VALUES "
            "('newer-shadow', 'AAPL', 'quotes', 'quote-shadow-aapl', 989, 990, 'quote', 0, "
            "202.25, 'quote-shadow-aapl', 202.30, 'quote-shadow-aapl', 202.27, "
            "'quote-shadow-aapl')"
        )

    latest = TestClient(create_web_app(_config(database))).get("/api/v2/live").json()
    assert latest["run"]["run_id"] == "newer-shadow"
    assert latest["run"]["mode"] == "shadow"
    assert latest["recorder"]["lifecycle"] == "recording"
    assert latest["ibkr"]["connection_generation"] == 8
    assert latest["feeds"] == {"active": 1, "by_kind": {"quotes": 1}}
    assert latest["instruments"][0]["event_id"] == "quote-shadow-aapl"

    pinned_shadow = TestClient(create_web_app(_config(database, run_id="newer-shadow"))).get(
        "/api/v2/live"
    )
    assert pinned_shadow.status_code == 200
    assert pinned_shadow.json()["run"]["mode"] == "shadow"

    pinned_record = TestClient(create_web_app(_config(database, run_id="run-live"))).get(
        "/api/v2/live"
    )
    assert pinned_record.status_code == 200
    assert pinned_record.json()["run"]["mode"] == "prospective_record"


def test_live_uses_filter_bound_keyset_cursors_and_default_limits(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed_live(database)
    payload = "{}"
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency) VALUES ('MSFT', ?, 272093, 'stock', 'MSFT', 'SMART', 'USD')",
            (_hash("MSFT"),),
        )
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES "
            "(2, 'quote-msft', 'run-live', 2, 4, 'quote', 190, ?, 'pending')",
            (_hash(payload),),
        )
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
            "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
            "bid_value, ask_value, last_value, payload_json, payload_sha256) VALUES "
            "('quote-msft', 'run-live', 2, 'MSFT', 'quotes', 'quote', 189, 190, 4, "
            "201.25, 201.30, 201.27, ?, ?)",
            (payload, _hash(payload)),
        )
        connection.execute(
            "INSERT INTO market_latest(run_id, instrument_id, feed_kind, event_id, "
            "event_at_us, received_at_us, event_kind, quality_bits, bid_value, "
            "bid_source_event_id, ask_value, ask_source_event_id, last_value, "
            "last_source_event_id) VALUES "
            "('run-live', 'MSFT', 'quotes', 'quote-msft', 189, 190, 'quote', 0, "
            "201.25, 'quote-msft', 201.30, 'quote-msft', 201.27, 'quote-msft')"
        )
    client = TestClient(create_web_app(_config(database)))

    first = client.get("/api/v2/live", params={"feed_kind": " QUOTES ", "limit": 1})
    assert first.status_code == 200
    assert [row["instrument_id"] for row in first.json()["instruments"]] == ["AAPL"]
    assert first.json()["limit"] == 1
    assert first.json()["next_cursor"]

    second = client.get(
        "/api/v2/live",
        params={
            "feed_kind": "quotes",
            "limit": 1,
            "cursor": first.json()["next_cursor"],
        },
    )
    assert [row["instrument_id"] for row in second.json()["instruments"]] == ["MSFT"]
    assert second.json()["next_cursor"] is None
    assert client.get("/api/v2/live", params={"cursor": "malformed"}).status_code == 422
    assert (
        client.get(
            "/api/v2/live",
            params={"feed_kind": "trades", "cursor": first.json()["next_cursor"]},
        ).status_code
        == 422
    )
    assert client.get("/api/v2/live", params={"limit": 201}).status_code == 422
    assert client.get("/api/v2/live", params={"feed_kind": "   "}).status_code == 422
    assert client.get("/api/v2/live", params={"cursor": "x" * 2_049}).status_code == 422


def test_ideas_use_route_and_filter_bound_keyset_cursors(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed_ideas(database)
    client = TestClient(create_web_app(_config(database)))

    first = client.get("/api/v2/ideas", params={"health": "healthy", "limit": 1})
    assert first.status_code == 200
    first_page = first.json()
    assert [item["instance_id"] for item in first_page["items"]] == ["instance-new"]
    assert {item["idea_id"] for item in first_page["plugins"]} == {
        "unknown_alpha",
        "discovered_only",
    }
    assert first_page["next_cursor"]

    second = client.get(
        "/api/v2/ideas",
        params={"health": "healthy", "limit": 1, "cursor": first_page["next_cursor"]},
    )
    assert second.status_code == 200
    assert [item["instance_id"] for item in second.json()["items"]] == ["instance-old"]
    assert second.json()["next_cursor"] is None

    assert client.get("/api/v2/ideas", params={"cursor": "malformed"}).status_code == 422
    mismatched = client.get(
        "/api/v2/ideas",
        params={"health": "disabled", "cursor": first_page["next_cursor"]},
    )
    assert mismatched.status_code == 422
    assert mismatched.json()["detail"] == "invalid_cursor"
    assert client.get("/api/v2/ideas", params={"limit": 101}).status_code == 422
    cursor_with_window = _encode_cursor(
        route="/api/v2/ideas",
        filters={"active": None, "health": None, "mode": None},
        timestamp=180,
        identity="instance-new",
        window=(0, 1),
    )
    assert client.get("/api/v2/ideas", params={"cursor": cursor_with_window}).status_code == 422


def test_idea_detail_is_generic_bounded_and_cursor_paginated(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed_outputs(database)
    client = TestClient(create_web_app(_config(database)))
    filters = {"start_us": 0, "end_us": 1_000, "limit": 2}

    first = client.get("/api/v2/ideas/instance-new", params=filters)
    assert first.status_code == 200
    payload = first.json()
    assert payload["instance"]["idea_id"] == "unknown_alpha"
    assert payload["instance"]["health"] == "healthy"
    assert [item["kind"] for item in payload["outputs"]] == [
        "proposed_trade",
        "proposed_position",
    ]
    assert payload["outputs"][0]["authority"] == "unapproved"
    assert payload["outputs"][0]["payload"] == {"thesis": "generic"}
    assert payload["outputs"][0]["legs"][0]["action"] == "buy"
    assert payload["next_cursor"]

    second = client.get(
        "/api/v2/ideas/instance-new",
        params={**filters, "cursor": payload["next_cursor"]},
    )
    assert [item["kind"] for item in second.json()["outputs"]] == [
        "signal",
        "observation",
    ]
    assert second.json()["outputs"][-1]["payload"] == {"novel": {"score": 3}}
    assert second.json()["next_cursor"] is None

    mismatched = client.get(
        "/api/v2/ideas/instance-new",
        params={**filters, "kind": "signal", "cursor": payload["next_cursor"]},
    )
    assert mismatched.status_code == 422
    assert client.get("/api/v2/ideas/missing").status_code == 404
    assert (
        client.get(
            "/api/v2/ideas/instance-new",
            params={"start_us": 0, "end_us": 604_800_000_001},
        ).status_code
        == 422
    )
    mismatched_window = _encode_cursor(
        route="/api/v2/ideas/{instance_id}",
        filters={
            "end_us": 1_000,
            "instance_id": "instance-new",
            "kind": None,
            "start_us": 0,
        },
        timestamp=204,
        identity="output-trade",
        window=(0, 999),
    )
    assert (
        client.get(
            "/api/v2/ideas/instance-new",
            params={"start_us": 0, "end_us": 1_000, "cursor": mismatched_window},
        ).status_code
        == 422
    )
    oversized_timestamp = _encode_cursor(
        route="/api/v2/ideas/{instance_id}",
        filters={
            "end_us": None,
            "instance_id": "instance-new",
            "kind": None,
            "start_us": None,
        },
        timestamp=SQLITE_INTEGER_MAX + 1,
        identity="output-trade",
        window=(0, 1),
    )
    assert (
        client.get(
            "/api/v2/ideas/instance-new",
            params={"cursor": oversized_timestamp},
        ).status_code
        == 422
    )
    assert client.get("/api/v2/ideas/instance-new", params={"limit": 201}).status_code == 422
    oversized_cursor = _encode_cursor(
        route="/api/v2/ideas/{instance_id}",
        filters={
            "end_us": None,
            "instance_id": "instance-new",
            "kind": None,
            "start_us": None,
        },
        timestamp=204,
        identity="output-trade",
        window=(0, IDEA_WINDOW_US + 1),
    )
    assert (
        client.get(
            "/api/v2/ideas/instance-new",
            params={"cursor": oversized_cursor},
        ).status_code
        == 422
    )


def test_results_are_virtual_bounded_and_exactly_traceable(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed_results(database)
    client = TestClient(create_web_app(_config(database)))
    filters = {"start_us": 0, "end_us": 1_000, "limit": 2}

    first = client.get("/api/v2/results", params=filters)
    assert first.status_code == 200
    payload = first.json()
    assert [item["status"] for item in payload["items"]] == ["closed", "open"]
    assert all(
        item["shadow"] is True and item["broker_position"] is False and item["fill"] is False
        for item in payload["items"]
    )
    assert payload["aggregates"]["by_status"] == {
        "closed": 1,
        "open": 1,
        "incomplete": 1,
        "invalid": 1,
    }
    assert payload["next_cursor"]

    second = client.get(
        "/api/v2/results",
        params={**filters, "cursor": payload["next_cursor"]},
    )
    assert [item["status"] for item in second.json()["items"]] == [
        "incomplete",
        "invalid",
    ]

    detail = client.get(
        "/api/v2/results/position-closed",
        params={"start_us": 0, "end_us": 1_000},
    )
    assert detail.status_code == 200
    result = detail.json()
    assert result["language"] == "virtual only; no broker orders, fills, or positions"
    assert result["position"]["shadow"] is True
    assert result["position"]["broker_position"] is False
    assert result["position"]["fill"] is False
    assert result["source_proposal"]["output_id"] == "proposal-closed"
    assert result["source_proposal"]["authority"] == "unapproved"
    assert result["legs"][0]["entry_market_event_id"] == "shadow-entry"
    assert result["legs"][0]["exit_market_event_id"] == "shadow-exit"
    assert [mark["marked_at_us"] for mark in result["marks"]] == [330, 320]
    assert result["outcome"]["completeness"] == "complete"

    first_mark = client.get(
        "/api/v2/results/position-closed",
        params={"start_us": 0, "end_us": 1_000, "limit": 1},
    ).json()
    assert [mark["marked_at_us"] for mark in first_mark["marks"]] == [330]
    assert first_mark["next_cursor"]
    second_mark = client.get(
        "/api/v2/results/position-closed",
        params={
            "start_us": 0,
            "end_us": 1_000,
            "limit": 1,
            "cursor": first_mark["next_cursor"],
        },
    ).json()
    assert [mark["marked_at_us"] for mark in second_mark["marks"]] == [320]
    assert second_mark["next_cursor"] is None

    assert client.get("/api/v2/results/missing").status_code == 404
    assert (
        client.get(
            "/api/v2/results",
            params={"start_us": 0, "end_us": 2_592_000_000_001},
        ).status_code
        == 422
    )
    assert client.get("/api/v2/results", params={"limit": 201}).status_code == 422
    assert (
        client.get(
            "/api/v2/results",
            params={"start_us": SQLITE_INTEGER_MAX, "end_us": SQLITE_INTEGER_MAX},
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/api/v2/results",
            params={"start_us": SQLITE_INTEGER_MAX},
        ).status_code
        == 422
    )


def test_history_windows_and_details_remain_inside_configured_run_scope(
    tmp_path: Path,
) -> None:
    empty_database = tmp_path / "empty.sqlite3"
    initialize_database(empty_database)
    empty_client = TestClient(create_web_app(_config(empty_database)))
    assert (
        empty_client.get(
            "/api/v2/results",
            params={"start_us": 0, "end_us": RESULT_WINDOW_US + 1},
        ).status_code
        == 422
    )

    database = tmp_path / "scoped.sqlite3"
    _seed_results(database)
    record_client = TestClient(create_web_app(_config(database, run_id="run-live")))
    shadow_client = TestClient(create_web_app(_config(database, run_id="run-shadow")))

    assert record_client.get("/api/v2/results/position-closed").status_code == 404
    assert shadow_client.get("/api/v2/ideas/instance-new").status_code == 404
    assert shadow_client.get("/api/v2/results/position-closed").status_code == 200


def test_meta_and_diagnostics_are_bounded_and_authority_free(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    backup_directory = tmp_path / "backups"
    backup_directory.mkdir()
    _seed_ideas(database)
    with connect_v2(database) as connection:
        for number in range(3):
            connection.execute(
                "INSERT INTO incidents(incident_id, run_id, scope, severity, code, opened_at_us, "
                "details_json) VALUES (?, 'run-live', 'ingestion', 'degraded', 'TEST', ?, '{}')",
                (f"incident-{number}", 200 + number),
            )
            connection.execute(
                "INSERT INTO gaps(gap_id, run_id, subscription_id, started_at_us, ended_at_us, "
                "reason, data_loss_possible, continuity_required, resolved_at_us) VALUES "
                "(?, 'run-live', 'sub-aapl', ?, ?, 'disconnect', 1, 1, ?)",
                (f"gap-{number}", 200 + number, 201 + number, 202 + number),
            )
    for number in range(3):
        create_backup(
            database,
            backup_directory,
            tier="daily",
            created_at_us=300 + number,
        )
    (backup_directory / "callback-payload.json").write_text(
        '{"secret":"not diagnostics"}', encoding="utf-8"
    )
    client = TestClient(create_web_app(_config(database, backup_directory=backup_directory)))

    meta = client.get("/api/v2/meta")
    assert meta.status_code == 200
    assert meta.json()["banner"] == "PROSPECTIVE / SHADOW ONLY — NO APPROVAL OR EXECUTION"
    assert meta.json()["modes"] == ["prospective_record", "shadow"]
    assert set(meta.json()["api_routes"]) == EXPECTED_API_ROUTES
    assert meta.json()["authority"] == {
        "approval": False,
        "execution": False,
        "broker_orders": False,
        "broker_fills": False,
        "broker_positions": False,
    }

    response = client.get("/api/v2/diagnostics", params={"limit": 2})
    assert response.status_code == 200
    diagnostics = response.json()
    assert len(diagnostics["incidents"]) == 2
    assert len(diagnostics["gaps"]) == 2
    assert len(diagnostics["subscriptions"]) == 1
    assert len(diagnostics["backups"]["items"]) == 2
    assert diagnostics["backups"]["status"]["state"] == "healthy"
    assert "secret" not in json.dumps(diagnostics)
    assert diagnostics["database"]["query_only"] is True
    assert diagnostics["hashes"]["build"] == "c4bb701"
    assert diagnostics["hashes"]["configuration"] == "b" * 64
    assert diagnostics["hashes"]["plugins"][0]["code_hash"] in {"e" * 64, "1" * 64}
    assert diagnostics["retention"]["database_cap_bytes"] == 8 * 1024**3
    assert diagnostics["retention"]["wal_cap_bytes"] == 64 * 1024**2
    assert "payload_json" not in json.dumps(diagnostics)
    assert "order_capability_observed" not in json.dumps(diagnostics)
    assert all("details" not in item for item in diagnostics["incidents"])
    assert client.get("/api/v2/diagnostics", params={"limit": 201}).status_code == 422


def test_authentication_rate_limit_and_get_only_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    monkeypatch.setenv("STOCKER_WEB_TEST_TOKEN", "correct horse battery staple")
    authenticated = _config(
        database,
        authentication_enabled=True,
        auth_token_env="STOCKER_WEB_TEST_TOKEN",
    )
    client = TestClient(create_web_app(authenticated))

    assert client.get("/").status_code == 401
    denied = client.get("/api/v2/meta")
    assert denied.status_code == 401
    assert denied.json()["detail"] == "authentication_required"
    authorized = client.get(
        "/api/v2/meta",
        headers={"Authorization": "Bearer correct horse battery staple"},
    )
    assert authorized.status_code == 200
    assert authorized.headers["x-content-type-options"] == "nosniff"
    assert authorized.headers["x-frame-options"] == "DENY"
    client.cookies.set("__Host-stocker_session", "correct horse battery staple")
    assert client.get("/api/v2/meta").status_code == 200

    limited = TestClient(create_web_app(_config(database, requests_per_minute=2)))
    assert limited.get("/api/v2/meta").status_code == 200
    assert limited.get("/api/v2/meta").status_code == 200
    exceeded = limited.get("/api/v2/meta")
    assert exceeded.status_code == 429
    assert exceeded.json()["detail"] == "rate_limit_exceeded"
    assert limited.post("/api/v2/live").status_code in {404, 405}


def test_web_config_fails_closed_for_unsafe_network_and_auth_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    for changes in (
        {"host": "0.0.0.0"},
        {"host": "::"},
        {"allowed_hosts": ["*"]},
        {"authentication_enabled": True, "auth_token_env": None},
        {
            "authentication_enabled": True,
            "auth_token_env": "TOKEN",
            "auth_cookie_name": "stocker_session",
        },
        {"trust_proxy_headers": True, "trusted_proxy_ips": []},
    ):
        with pytest.raises(ValidationError):
            _config(database, **changes)

    monkeypatch.delenv("ABSENT_STOCKER_TOKEN", raising=False)
    missing_token = _config(
        database,
        authentication_enabled=True,
        auth_token_env="ABSENT_STOCKER_TOKEN",
    )
    with pytest.raises(RuntimeError, match="authentication token is absent"):
        create_web_app(missing_token)


def test_runtime_cli_owns_the_v2_web_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    config_path = tmp_path / "web.json"
    config_path.write_text(_config(database).model_dump_json(), encoding="utf-8")
    observed: dict[str, object] = {}

    def run(application: object, **settings: object) -> None:
        observed["application"] = application
        observed["settings"] = settings

    monkeypatch.setattr("uvicorn.run", run)
    result = CliRunner().invoke(
        runtime_cli,
        ["web", "run", "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert observed["settings"] == {
        "host": "127.0.0.1",
        "port": 8000,
        "proxy_headers": False,
        "forwarded_allow_ips": "",
        "log_level": "info",
    }
    application = observed["application"]
    assert isinstance(application, FastAPI)
    assert set(application.openapi()["paths"]) == EXPECTED_API_ROUTES


def test_response_and_query_budgets_return_stable_bounded_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)

    monkeypatch.setattr(ReadModel, "meta", lambda _self: {"large": "x" * 2_000})
    oversized = TestClient(
        create_web_app(_config(database, maximum_response_bytes=1_024)),
        raise_server_exceptions=False,
    ).get("/api/v2/meta")
    assert oversized.status_code == 413
    assert oversized.json()["detail"] == "response_too_large"
    assert len(oversized.content) <= 512 * 1024

    def time_out(_self: ReadModel) -> dict[str, object]:
        raise QueryTimeoutError("query_timeout")

    monkeypatch.setattr(ReadModel, "meta", time_out)
    timed_out = TestClient(
        create_web_app(_config(database)),
        raise_server_exceptions=False,
    ).get("/api/v2/meta")
    assert timed_out.status_code == 503
    assert timed_out.json() == {"detail": "query_timeout"}

    def fail_internally(_self: ReadModel) -> dict[str, object]:
        raise ValueError("hidden internal failure")

    monkeypatch.setattr(ReadModel, "meta", fail_internally)
    internal_error = TestClient(
        create_web_app(_config(database)),
        raise_server_exceptions=False,
    ).get("/api/v2/meta")
    assert internal_error.status_code == 500
    assert internal_error.json() == {"detail": "internal_error"}
    assert internal_error.headers["x-content-type-options"] == "nosniff"


def test_query_layer_is_read_only_and_progress_handler_enforces_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    read_model = ReadModel(_config(database))

    with read_model._connection() as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE forbidden_write(id INTEGER)")

    with (
        pytest.raises(QueryTimeoutError, match="query_timeout"),
        read_model._connection(),
    ):
        raise sqlite3.OperationalError("database is locked")

    clock_values = iter((0.0, 1.0))
    monkeypatch.setattr(
        web_queries.time,
        "perf_counter",
        lambda: next(clock_values, 1.0),
    )
    timed_read_model = ReadModel(_config(database, query_budget_ms=1))
    with (
        pytest.raises(QueryTimeoutError, match="query_timeout"),
        timed_read_model._connection() as connection,
    ):
        connection.execute(
            "WITH RECURSIVE counter(value) AS ("
            "SELECT 1 UNION ALL SELECT value + 1 FROM counter WHERE value < 1000000"
            ") SELECT SUM(value) FROM counter"
        ).fetchone()


def test_measured_web_queries_use_timestamp_id_indexes(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)

    def plan(connection: sqlite3.Connection, statement: str) -> str:
        return "\n".join(
            str(row[3]) for row in connection.execute("EXPLAIN QUERY PLAN " + statement).fetchall()
        )

    with connect_v2(database) as connection:
        live_plan = plan(
            connection,
            "SELECT event_id FROM market_latest WHERE run_id = 'run' "
            "ORDER BY event_at_us DESC, event_id LIMIT 51",
        )
        ideas_plan = plan(
            connection,
            "SELECT instance_id FROM idea_instances "
            "ORDER BY activated_at_us DESC, instance_id LIMIT 51",
        )
        run_ideas_plan = plan(
            connection,
            "SELECT instance_id FROM idea_instances WHERE run_id = 'run' "
            "ORDER BY activated_at_us DESC, instance_id LIMIT 51",
        )
        outputs_plan = plan(
            connection,
            "SELECT output_id FROM idea_outputs WHERE instance_id = 'instance' "
            "AND as_of_at_us >= 0 AND as_of_at_us <= 1 "
            "ORDER BY as_of_at_us DESC, output_id LIMIT 51",
        )
        results_plan = plan(
            connection,
            "SELECT position.position_id FROM idea_outputs AS output "
            "JOIN idea_output_seals AS seal ON seal.output_id = output.output_id "
            "JOIN shadow_positions AS position "
            "ON position.proposed_trade_output_id = output.output_id "
            "WHERE output.run_id = 'run' AND output.output_kind = 'proposed_trade' "
            "AND output.as_of_at_us >= 0 AND output.as_of_at_us <= 1 "
            "ORDER BY output.as_of_at_us DESC, position.position_id LIMIT 51",
        )

    assert "market_latest_run_event_time_idx" in live_plan
    assert "idea_instances_activation_time_idx" in ideas_plan
    assert "idea_instances_run_activation_time_idx" in run_ideas_plan
    assert "idea_outputs_instance_time_web_idx" in outputs_plan
    assert "idea_outputs_run_kind_time_idx" in results_plan
    for measured_plan in (
        live_plan,
        ideas_plan,
        run_ideas_plan,
        outputs_plan,
    ):
        assert "TEMP B-TREE" not in measured_plan


def test_frontend_has_exactly_three_accessible_primary_views(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    initialize_database(database)
    client = TestClient(create_web_app(_config(database)))

    page = client.get("/")
    assert page.status_code == 200
    html = page.text
    nav = re.search(r'<nav[^>]+aria-label="Primary views"[^>]*>(.*?)</nav>', html, re.DOTALL)
    assert nav is not None
    labels = re.findall(r'<button[^>]+class="view-tab[^>]*>\s*([^<]+?)\s*</button>', nav[1])
    assert labels == ["Live", "Ideas", "Results"]
    assert 'role="tablist"' in nav[0]
    assert html.count('role="tabpanel"') == 3
    assert '<dialog id="diagnostics-drawer"' in html
    assert 'id="diagnostics-open"' in html
    assert "Diagnostics" not in labels
    assert "PROSPECTIVE / SHADOW ONLY — NO APPROVAL OR EXECUTION" in html
    assert "virtual only; no broker orders, fills, or positions" in html
    assert '<a class="skip-link" href="#main">' in html
    assert '<main id="main" tabindex="-1">' in html
    assert html.count("<h1") == 1

    script = client.get("/assets/app.js")
    stylesheet = client.get("/assets/app.css")
    assert script.status_code == stylesheet.status_code == 200
    assert "genericEntries" in script.text
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet.text
    assert ":focus-visible" in stylesheet.text
    for removed in (
        "opening-leader",
        "quiet-state",
        "source-transfer",
        "report-packages",
        "/api/replay",
        "/api/dashboard-snapshot",
    ):
        assert removed not in script.text.lower()
