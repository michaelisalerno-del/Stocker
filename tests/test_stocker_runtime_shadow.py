"""Phase 5 generic shadow valuation tests; these assert virtual evidence only."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stocker_runtime import ShadowCostPolicy, ShadowFillPolicy
from stocker_runtime.shadow import ShadowEngine, ShadowPolicy
from stocker_runtime.storage import connect_v2, initialize_database


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _seed(database: Path, *, quantity: float | None = 1.0) -> None:
    initialize_database(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES ('run', 'shadow', 'ibkr', 1, NULL, ?, 'deadbee', "
            "'shadow_protected', 'running', NULL)",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run', 1, 'fixture', 1)"
        )
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, currency) "
            "VALUES ('AAPL', ?, 'stock', 'AAPL', 'SMART', 'USD')",
            (_hash("AAPL"),),
        )
        connection.execute(
            "INSERT INTO idea_plugins VALUES ('idea', 'v1', 1, 'Idea', 'fixture', ?, ?, '{}', 1)",
            ("b" * 64, "c" * 64),
        )
        connection.execute(
            "INSERT INTO idea_instances(instance_id, idea_id, idea_version, run_id, mode, "
            "parameters_json, parameters_hash, plugin_code_hash, manifest_hash, universe_json, "
            "universe_hash, requirements_json, requirements_hash, activated_after_source_sequence, "
            "activated_at_us, health, data_class) VALUES ('instance', 'idea', 'v1', 'run', 'shadow', "
            "'{}', ?, ?, ?, '[\"AAPL\"]', ?, '[]', ?, 0, 1, 'healthy', 'shadow_protected')",
            ("d" * 64, "c" * 64, "b" * 64, _hash('["AAPL"]'), _hash("[]")),
        )
        for sequence, event_id, bid, ask, event_at in (
            (1, "input", 100.0, 101.0, 1),
            (2, "entry", 102.0, 103.0, 2),
            (3, "exit", 109.0, 110.0, 12),
        ):
            payload = json.dumps({}, separators=(",", ":"))
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, recorder_generation, "
                "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
                "VALUES (?, ?, 'run', 1, 1, 'quote', ?, ?, 'pending')",
                (sequence, event_id, event_at, _hash(payload)),
            )
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, feed_kind, "
                "event_kind, event_at_us, received_at_us, connection_generation, bid_value, ask_value, "
                "payload_json, payload_sha256) VALUES (?, 'run', ?, 'AAPL', 'quotes', 'quote', ?, ?, 1, ?, ?, ?, ?)",
                (event_id, sequence, event_at, event_at, bid, ask, payload, _hash(payload)),
            )
        payload = "{}"
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, subject_instrument_id, "
            "emitted_at_us, as_of_at_us, first_input_event_id, last_input_event_id, input_watermark, "
            "input_events_hash, output_ordinal, payload_json, payload_hash, content_hash, data_class, authority_status) "
            "VALUES ('proposal', 'run', 'instance', 'proposed_trade', 'AAPL', 1, 1, 'input', 'input', 'input', ?, 0, ?, ?, ?, 'shadow_protected', 'unapproved')",
            (_hash('["input"]'), payload, _hash(payload), "e" * 64),
        )
        connection.execute("INSERT INTO idea_output_inputs VALUES ('proposal', 'input', 0)")
        if quantity is None:
            connection.execute(
                "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, target, notional_value, currency) "
                "VALUES ('proposal', 0, 'AAPL', 'buy', 'long', 100, 'USD')"
            )
        else:
            connection.execute(
                "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, target, quantity_value, currency) "
                "VALUES ('proposal', 0, 'AAPL', 'buy', 'long', ?, 'USD')",
                (quantity,),
            )


def test_shadow_uses_first_causal_ask_then_bid_and_closes_deterministically(tmp_path: Path) -> None:
    database = tmp_path / "shadow.sqlite3"
    _seed(database)
    policy = ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=100),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=10),
        horizons_us=(10,),
    )
    engine = ShadowEngine(database, run_id="run", policy=policy)

    assert engine.run_once(now_us=2) == 1
    assert engine.run_once(now_us=12) == 1
    assert engine.run_once(now_us=12) == 0
    with connect_v2(database) as connection:
        leg = connection.execute(
            "SELECT entry_market_event_id, entry_price, exit_market_event_id, exit_price FROM shadow_legs"
        ).fetchone()
        outcome = connection.execute(
            "SELECT gross_pnl, net_pnl, mfe, mae, completeness FROM shadow_outcomes"
        ).fetchone()
    assert tuple(leg) == ("entry", 103.0, "exit", 109.0)
    assert outcome[0] == 6.0
    assert outcome[1] == pytest.approx(5.794)
    assert outcome[2] == outcome[3] == 6.0
    assert outcome[4] == "complete"


def test_shadow_rejects_non_shadow_runs_and_invalid_quantity(tmp_path: Path) -> None:
    database = tmp_path / "shadow.sqlite3"
    _seed(database, quantity=None)
    engine = ShadowEngine(database, run_id="run")
    assert engine.run_once(now_us=2) == 1
    with connect_v2(database) as connection:
        assert tuple(connection.execute(
            "SELECT lifecycle, invalid_reason FROM shadow_positions"
        ).fetchone()) == (
            "invalid",
            "quantity_unavailable",
        )
