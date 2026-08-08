"""Phase 5 generic shadow valuation tests; these assert virtual evidence only."""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from stocker_runtime import ShadowCostPolicy, ShadowFillPolicy
from stocker_runtime.ingestion import Recorder, RecorderConfig
from stocker_runtime.shadow import ShadowEngine, ShadowPolicy
from stocker_runtime.storage import (
    RetentionManager,
    RetentionPolicy,
    connect_v2,
    initialize_database,
)

ROOT = Path(__file__).resolve().parents[1]
SHADOW_PACKAGE = ROOT / "packages/stocker_runtime/src/stocker_runtime/shadow"

type QuoteSeed = tuple[int, str, str, float | None, float | None, int]
type LegSeed = tuple[str, str, float | None, str]

DEFAULT_QUOTES: tuple[QuoteSeed, ...] = (
    (1, "input", "AAPL", 100.0, 101.0, 1),
    (2, "entry", "AAPL", 102.0, 103.0, 2),
    (3, "exit", "AAPL", 109.0, 110.0, 12),
)
DEFAULT_LEGS: tuple[LegSeed, ...] = (("AAPL", "buy", 1.0, "USD"),)


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
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, "
            "currency) "
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
            "activated_at_us, health, data_class) VALUES "
            "('instance', 'idea', 'v1', 'run', 'shadow', "
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
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, "
                "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
                "VALUES (?, ?, 'run', 1, 1, 'quote', ?, ?, 'pending')",
                (sequence, event_id, event_at, _hash(payload)),
            )
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                "bid_value, ask_value, payload_json, payload_sha256) VALUES "
                "(?, 'run', ?, 'AAPL', 'quotes', 'quote', ?, ?, 1, ?, ?, ?, ?)",
                (event_id, sequence, event_at, event_at, bid, ask, payload, _hash(payload)),
            )
        payload = "{}"
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
            "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
            "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
            "payload_json, payload_hash, content_hash, data_class, authority_status) VALUES "
            "('proposal', 'run', 'instance', 'proposed_trade', 'AAPL', 1, 1, 'input', "
            "'input', 'input', ?, 0, ?, ?, ?, 'shadow_protected', 'unapproved')",
            (_hash('["input"]'), payload, _hash(payload), "e" * 64),
        )
        connection.execute("INSERT INTO idea_output_inputs VALUES ('proposal', 'input', 0)")
        if quantity is None:
            connection.execute(
                "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                "target, notional_value, currency) "
                "VALUES ('proposal', 0, 'AAPL', 'buy', 'long', 100, 'USD')"
            )
        else:
            connection.execute(
                "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                "target, quantity_value, currency) "
                "VALUES ('proposal', 0, 'AAPL', 'buy', 'long', ?, 'USD')",
                (quantity,),
            )


def _insert_quote(
    connection: sqlite3.Connection,
    *,
    sequence: int,
    event_id: str,
    instrument_id: str,
    bid: float | None,
    ask: float | None,
    event_at_us: int,
) -> None:
    payload = json.dumps({}, separators=(",", ":"))
    connection.execute(
        "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, recorder_generation, "
        "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
        "VALUES (?, ?, 'run', 1, 1, 'quote', ?, ?, 'pending')",
        (sequence, event_id, event_at_us, _hash(payload)),
    )
    connection.execute(
        "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, feed_kind, "
        "event_kind, event_at_us, received_at_us, connection_generation, bid_value, ask_value, "
        "payload_json, payload_sha256) VALUES (?, 'run', ?, ?, 'quotes', 'quote', ?, ?, 1, "
        "?, ?, ?, ?)",
        (
            event_id,
            sequence,
            instrument_id,
            event_at_us,
            event_at_us,
            bid,
            ask,
            payload,
            _hash(payload),
        ),
    )


def _seed_case(
    database: Path,
    *,
    quotes: tuple[QuoteSeed, ...],
    legs: tuple[LegSeed, ...] = DEFAULT_LEGS,
) -> None:
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
        instruments = tuple(
            dict.fromkeys(["AAPL", *(quote[2] for quote in quotes), *(leg[0] for leg in legs)])
        )
        connection.executemany(
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, "
            "currency) "
            "VALUES (?, ?, 'stock', ?, 'SMART', 'USD')",
            ((instrument, _hash(instrument), instrument) for instrument in instruments),
        )
        connection.execute(
            "INSERT INTO idea_plugins VALUES ('idea', 'v1', 1, 'Idea', 'fixture', ?, ?, '{}', 1)",
            ("b" * 64, "c" * 64),
        )
        universe = json.dumps(sorted({leg[0] for leg in legs}), separators=(",", ":"))
        connection.execute(
            "INSERT INTO idea_instances(instance_id, idea_id, idea_version, run_id, mode, "
            "parameters_json, parameters_hash, plugin_code_hash, manifest_hash, universe_json, "
            "universe_hash, requirements_json, requirements_hash, activated_after_source_sequence, "
            "activated_at_us, health, data_class) VALUES "
            "('instance', 'idea', 'v1', 'run', 'shadow', "
            "'{}', ?, ?, ?, ?, ?, '[]', ?, 0, 1, 'healthy', 'shadow_protected')",
            ("d" * 64, "c" * 64, "b" * 64, universe, _hash(universe), _hash("[]")),
        )
        for sequence, event_id, instrument_id, bid, ask, event_at_us in quotes:
            _insert_quote(
                connection,
                sequence=sequence,
                event_id=event_id,
                instrument_id=instrument_id,
                bid=bid,
                ask=ask,
                event_at_us=event_at_us,
            )
        payload = "{}"
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
            "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
            "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
            "payload_json, payload_hash, content_hash, data_class, authority_status) "
            "VALUES ('proposal', 'run', 'instance', 'proposed_trade', 'AAPL', 1, 1, "
            "'input', 'input', 'input', ?, 0, ?, ?, ?, 'shadow_protected', 'unapproved')",
            (_hash('["input"]'), payload, _hash(payload), "e" * 64),
        )
        connection.execute("INSERT INTO idea_output_inputs VALUES ('proposal', 'input', 0)")
        for leg_number, (instrument_id, action, quantity, currency) in enumerate(legs):
            target = "long" if action == "buy" else "short"
            if quantity is None:
                connection.execute(
                    "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                    "target, notional_value, currency) VALUES ('proposal', ?, ?, ?, ?, 100, ?)",
                    (leg_number, instrument_id, action, target, currency),
                )
            else:
                connection.execute(
                    "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                    "target, quantity_value, currency) VALUES ('proposal', ?, ?, ?, ?, ?, ?)",
                    (leg_number, instrument_id, action, target, quantity, currency),
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
            "SELECT entry_market_event_id, entry_price, exit_market_event_id, exit_price "
            "FROM shadow_legs"
        ).fetchone()
        outcome = connection.execute(
            "SELECT gross_pnl, net_pnl, mfe, mae, completeness FROM shadow_outcomes"
        ).fetchone()
    assert tuple(leg) == ("entry", 103.0, "exit", 109.0)
    assert outcome[0] == 6.0
    assert outcome[1] == pytest.approx(5.794)
    assert outcome[2] == outcome[3] == 6.0
    assert outcome[4] == "complete"


def test_split_bid_ask_callbacks_form_causal_snapshots_with_exact_side_provenance(
    tmp_path: Path,
) -> None:
    database = tmp_path / "split-quotes.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 99.0, 100.0, 1),
            (2, "entry-bid", "AAPL", 100.0, None, 2),
            (3, "entry-ask", "AAPL", None, 106.0, 3),
            (4, "exit-bid", "AAPL", 105.0, None, 4),
        ),
    )
    policy = ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=2),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=0),
        horizons_us=(1,),
    )

    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=4) == 1

    with connect_v2(database) as connection:
        position = connection.execute(
            "SELECT lifecycle, opened_at_us, closed_at_us FROM shadow_positions"
        ).fetchone()
        leg = connection.execute(
            "SELECT entry_market_event_id, entry_bid_market_event_id, "
            "entry_ask_market_event_id, entry_price, exit_market_event_id, "
            "exit_bid_market_event_id, exit_ask_market_event_id, exit_price "
            "FROM shadow_legs"
        ).fetchone()
        mark = connection.execute("SELECT payload_json FROM shadow_marks").fetchone()
        outcome = connection.execute(
            "SELECT gross_pnl, payload_json FROM shadow_outcomes"
        ).fetchone()

    expected_exit_quote_ids = [
        {
            "ask_event_id": "entry-ask",
            "bid_event_id": "exit-bid",
            "instrument_id": "AAPL",
            "leg_number": 0,
        }
    ]
    assert tuple(position) == ("closed", 3, 4)
    assert tuple(leg) == (
        "entry-ask",
        "entry-bid",
        "entry-ask",
        106.0,
        "exit-bid",
        "exit-bid",
        "entry-ask",
        105.0,
    )
    assert json.loads(mark["payload_json"])["quote_event_ids"] == expected_exit_quote_ids
    assert outcome["gross_pnl"] == -1.0
    outcome_payload = json.loads(outcome["payload_json"])
    assert outcome_payload["exit_quote_event_ids"] == expected_exit_quote_ids
    assert outcome_payload["mfe_quote_event_ids"] == expected_exit_quote_ids
    assert outcome_payload["mae_quote_event_ids"] == expected_exit_quote_ids
    assert (
        outcome_payload["policy_hash"]
        == hashlib.sha256(
            json.dumps(
                {
                    "cost": policy.cost.model_dump(mode="json"),
                    "fill": policy.fill.model_dump(mode="json"),
                    "horizons_us": policy.horizons_us,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )


def test_split_quote_state_survives_restart_and_source_compaction_idempotently(
    tmp_path: Path,
) -> None:
    database = tmp_path / "split-restart.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 99.0, 100.0, 1),
            (2, "entry-bid", "AAPL", 100.0, None, 2),
            (3, "entry-ask", "AAPL", None, 106.0, 3),
            (4, "exit-bid", "AAPL", 105.0, None, 4),
        ),
    )
    policy = ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=2),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=0),
        horizons_us=(1,),
    )
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=3) == 1
    raw = sqlite3.connect(database)
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute("DELETE FROM market_events WHERE event_id IN ('entry-bid', 'entry-ask')")
        raw.commit()
    finally:
        raw.close()

    restarted = ShadowEngine(database, run_id="run", policy=policy)
    assert restarted.run_once(now_us=4) == 1
    assert restarted.run_once(now_us=4) == 0
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute(
                "SELECT lifecycle, exit_bid_market_event_id, exit_ask_market_event_id "
                "FROM shadow_positions JOIN shadow_legs USING(position_id)"
            ).fetchone()
        ) == ("closed", "exit-bid", "entry-ask")
        assert connection.execute("SELECT count(*) FROM shadow_marks").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM shadow_outcomes").fetchone()[0] == 1


def test_shadow_rejects_non_shadow_runs_and_invalid_quantity(tmp_path: Path) -> None:
    database = tmp_path / "shadow.sqlite3"
    _seed(database, quantity=None)
    engine = ShadowEngine(database, run_id="run")
    assert engine.run_once(now_us=2) == 1
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute("SELECT lifecycle, invalid_reason FROM shadow_positions").fetchone()
        ) == (
            "invalid",
            "quantity_unavailable",
        )


def test_shadow_rejects_more_than_eight_trade_legs_without_partial_state(tmp_path: Path) -> None:
    database = tmp_path / "too-many-legs.sqlite3"
    legs = tuple((f"LEG-{index}", "buy", 1.0, "USD") for index in range(9))
    _seed_case(
        database,
        quotes=((1, "input", "AAPL", 100.0, 101.0, 1),),
        legs=legs,
    )

    assert ShadowEngine(database, run_id="run", policy=_policy()).run_once(now_us=1) == 1
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute("SELECT lifecycle, invalid_reason FROM shadow_positions").fetchone()
        ) == ("invalid", "leg_count_exceeds_limit")
        assert connection.execute("SELECT count(*) FROM shadow_quote_state").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM shadow_legs").fetchone()[0] == 0


def _policy(*, max_quote_age_us: int = 100, per_side_bps: float = 10) -> ShadowPolicy:
    return ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=max_quote_age_us),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=per_side_bps),
        horizons_us=(10,),
    )


def test_delayed_entry_matches_prompt_first_causal_snapshot(tmp_path: Path) -> None:
    database = tmp_path / "stale-entry.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 100.0, 101.0, 1),
            (2, "stale-entry", "AAPL", 102.0, 103.0, 2),
        ),
    )
    engine = ShadowEngine(database, run_id="run", policy=_policy(max_quote_age_us=10))

    assert engine.run_once(now_us=100) == 1
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute("SELECT lifecycle, opened_at_us FROM shadow_positions").fetchone()
        ) == ("open", 2)
        assert (
            connection.execute("SELECT entry_market_event_id FROM shadow_legs").fetchone()[0]
            == "stale-entry"
        )


@pytest.mark.parametrize(("bid", "ask"), ((101.0, 100.0), (None, 100.0)))
def test_crossed_or_missing_entry_then_later_valid_quote_opens_atomically(
    tmp_path: Path, bid: float | None, ask: float | None
) -> None:
    database = tmp_path / f"invalid-entry-{bid}-{ask}.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 100.0, 101.0, 1),
            (2, "invalid-entry", "AAPL", bid, ask, 2),
        ),
    )
    engine = ShadowEngine(database, run_id="run", policy=_policy())

    assert engine.run_once(now_us=2) == 0
    with connect_v2(database) as connection:
        assert connection.execute("SELECT lifecycle FROM shadow_positions").fetchone()[0] == (
            "pending"
        )
        assert connection.execute("SELECT count(*) FROM shadow_legs").fetchone()[0] == 0
        _insert_quote(
            connection,
            sequence=3,
            event_id="valid-entry",
            instrument_id="AAPL",
            bid=102.0,
            ask=103.0,
            event_at_us=3,
        )

    assert engine.run_once(now_us=3) == 1
    with connect_v2(database) as connection:
        assert connection.execute("SELECT lifecycle FROM shadow_positions").fetchone()[0] == (
            "open"
        )
        assert (
            connection.execute("SELECT entry_market_event_id FROM shadow_legs").fetchone()[0]
            == "valid-entry"
        )


def test_crossed_only_entry_remains_incomplete(tmp_path: Path) -> None:
    database = tmp_path / "crossed-only.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 100.0, 101.0, 1),
            (2, "crossed", "AAPL", 102.0, 101.0, 2),
        ),
    )
    engine = ShadowEngine(database, run_id="run", policy=_policy())

    assert engine.run_once(now_us=2) == 0
    assert engine.run_once(now_us=50) == 0
    with connect_v2(database) as connection:
        assert connection.execute("SELECT lifecycle FROM shadow_positions").fetchone()[0] == (
            "pending"
        )
        assert connection.execute("SELECT count(*) FROM shadow_legs").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM shadow_outcomes").fetchone()[0] == 0


def test_final_horizon_waits_for_first_valid_quote_at_or_after_target(tmp_path: Path) -> None:
    database = tmp_path / "horizon.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 100.0, 101.0, 1),
            (2, "entry", "AAPL", 102.0, 103.0, 2),
            (3, "pre-horizon", "AAPL", 108.0, 109.0, 11),
            (4, "crossed-at-horizon", "AAPL", 111.0, 110.0, 12),
        ),
    )
    engine = ShadowEngine(database, run_id="run", policy=_policy())

    assert engine.run_once(now_us=2) == 1
    assert engine.run_once(now_us=12) == 0
    with connect_v2(database) as connection:
        assert connection.execute("SELECT lifecycle FROM shadow_positions").fetchone()[0] == (
            "open"
        )
        assert connection.execute("SELECT count(*) FROM shadow_outcomes").fetchone()[0] == 0
        _insert_quote(
            connection,
            sequence=5,
            event_id="post-horizon-valid",
            instrument_id="AAPL",
            bid=109.0,
            ask=110.0,
            event_at_us=13,
        )

    assert engine.run_once(now_us=13) == 1
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute(
                "SELECT position.lifecycle, position.closed_at_us, leg.exit_market_event_id "
                "FROM shadow_positions position JOIN shadow_legs leg USING(position_id)"
            ).fetchone()
        ) == ("closed", 13, "post-horizon-valid")


def _cadence_snapshot(database: Path) -> tuple[tuple[tuple[Any, ...], ...], tuple[Any, ...]]:
    with connect_v2(database) as connection:
        marks = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT marked_at_us, gross_pnl, net_pnl, payload_json FROM shadow_marks "
                "ORDER BY marked_at_us"
            )
        )
        outcome = tuple(
            connection.execute(
                "SELECT gross_pnl, net_pnl, mfe, mae, outcome_at_us FROM shadow_outcomes"
            ).fetchone()
        )
    return marks, outcome


def test_marks_and_excursions_are_identical_across_cadence_and_restart(tmp_path: Path) -> None:
    quotes: tuple[QuoteSeed, ...] = (
        (1, "input", "AAPL", 100.0, 101.0, 1),
        (2, "entry", "AAPL", 100.0, 101.0, 2),
        (3, "adverse", "AAPL", 90.0, 91.0, 4),
        (4, "favourable", "AAPL", 120.0, 121.0, 6),
        (5, "terminal", "AAPL", 109.0, 110.0, 12),
    )
    online_database = tmp_path / "online.sqlite3"
    restarted_database = tmp_path / "restarted.sqlite3"
    for database in (online_database, restarted_database):
        _seed_case(database, quotes=quotes)

    policy = _policy(max_quote_age_us=3, per_side_bps=0)
    online = ShadowEngine(online_database, run_id="run", policy=policy)
    assert online.run_once(now_us=2) == 1
    assert online.run_once(now_us=4) == 0
    assert online.run_once(now_us=6) == 0
    assert online.run_once(now_us=12) == 1

    assert ShadowEngine(restarted_database, run_id="run", policy=policy).run_once(now_us=2) == 1
    assert ShadowEngine(restarted_database, run_id="run", policy=policy).run_once(now_us=12) == 1

    online_snapshot = _cadence_snapshot(online_database)
    restarted_snapshot = _cadence_snapshot(restarted_database)
    assert online_snapshot[1][2:4] == (19.0, -11.0)
    assert restarted_snapshot == online_snapshot


def test_two_horizons_use_first_post_target_snapshot_and_ignore_later_extreme(
    tmp_path: Path,
) -> None:
    database = tmp_path / "two-horizons.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 100.0, 101.0, 1),
            (2, "entry", "AAPL", 100.0, 101.0, 2),
            (3, "first-horizon", "AAPL", 104.0, 105.0, 7),
            (4, "between-extreme", "AAPL", 150.0, 151.0, 9),
            (5, "final-horizon", "AAPL", 109.0, 110.0, 13),
            (6, "post-exit-extreme", "AAPL", 200.0, 201.0, 14),
        ),
    )
    policy = ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=100),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=0),
        horizons_us=(5, 10),
    )

    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=14) == 1
    with connect_v2(database) as connection:
        marks = tuple(
            (int(row["marked_at_us"]), json.loads(str(row["payload_json"])))
            for row in connection.execute("SELECT * FROM shadow_marks ORDER BY marked_at_us")
        )
        outcome = connection.execute(
            "SELECT outcome_at_us, gross_pnl, mfe FROM shadow_outcomes"
        ).fetchone()
    assert tuple(mark[0] for mark in marks) == (7, 12)
    assert tuple(mark[1]["actual_quote_at_us"] for mark in marks) == (7, 13)
    assert tuple(outcome) == (13, 8.0, 49.0)


def test_evidence_projection_is_bounded_and_resumes_after_restart(tmp_path: Path) -> None:
    database = tmp_path / "bounded.sqlite3"
    quotes: list[QuoteSeed] = [(1, "input", "AAPL", 100.0, 101.0, 1)]
    quotes.extend(
        (sequence, f"quote-{sequence}", "AAPL", 100.0 + sequence, 101.0 + sequence, sequence)
        for sequence in range(2, 303)
    )
    _seed_case(database, quotes=tuple(quotes))
    policy = ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=100),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=0),
        horizons_us=(1_000,),
    )

    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=1_000) == 1
    with connect_v2(database) as connection:
        first_cursor = int(
            connection.execute("SELECT next_source_sequence FROM shadow_progress").fetchone()[0]
        )
    assert first_cursor == 10
    restarted = ShadowEngine(database, run_id="run", policy=policy)
    assert restarted.run_once(now_us=1_000) == 0
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT next_source_sequence FROM shadow_progress").fetchone()[0]
            == 18
        )
    for _ in range(40):
        restarted.run_once(now_us=1_000)
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT next_source_sequence FROM shadow_progress").fetchone()[0]
            == 303
        )
        assert connection.execute("SELECT count(*) FROM shadow_marks").fetchone()[0] <= 8


def test_fair_schedule_admits_proposal_33_and_advances_busy_positions_after_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "fair-schedule.sqlite3"
    quotes: list[QuoteSeed] = [(1, "input", "AAPL", 100.0, 101.0, 1)]
    quotes.extend(
        (sequence, f"quote-{sequence}", "AAPL", 100.0, 101.0, sequence)
        for sequence in range(2, 303)
    )
    _seed_case(database, quotes=tuple(quotes))
    with connect_v2(database) as connection:
        for ordinal in range(2, 34):
            output_id = f"proposal-{ordinal:02d}"
            connection.execute(
                "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
                "subject_instrument_id, emitted_at_us, as_of_at_us, valid_until_at_us, "
                "direction, strength, confidence, horizon_us, first_input_event_id, "
                "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
                "payload_json, payload_hash, content_hash, data_class, authority_status) "
                "SELECT ?, run_id, instance_id, output_kind, subject_instrument_id, emitted_at_us, "
                "as_of_at_us, valid_until_at_us, direction, strength, confidence, horizon_us, "
                "first_input_event_id, last_input_event_id, input_watermark, input_events_hash, "
                "?, payload_json, payload_hash, ?, data_class, authority_status "
                "FROM idea_outputs WHERE output_id='proposal'",
                (output_id, ordinal, _hash(output_id)),
            )
            connection.execute(
                "INSERT INTO idea_output_inputs VALUES (?, 'input', 0)", (output_id,)
            )
            connection.execute(
                "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                "target, quantity_value, currency) VALUES (?, 0, 'AAPL', 'buy', 'long', 1, 'USD')",
                (output_id,),
            )
    policy = ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=100),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=0),
        horizons_us=(1_000,),
    )

    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=1_000) == 32
    with connect_v2(database) as connection:
        first_cursor = connection.execute(
            "SELECT progress.next_source_sequence FROM shadow_progress progress "
            "JOIN shadow_positions position ON position.position_id=progress.position_id "
            "WHERE position.proposed_trade_output_id='proposal'"
        ).fetchone()[0]
        assert first_cursor == 10
        assert connection.execute("SELECT count(*) FROM shadow_positions").fetchone()[0] == 32
        assert (
            connection.execute(
                "SELECT 1 FROM shadow_positions WHERE proposed_trade_output_id='proposal-33'"
            ).fetchone()
            is None
        )

    restarted = ShadowEngine(database, run_id="run", policy=policy)
    assert restarted.run_once(now_us=1_000) == 1
    with connect_v2(database) as connection:
        schedule = tuple(
            connection.execute(
                "SELECT position.proposed_trade_output_id, progress.schedule_count, "
                "progress.next_source_sequence FROM shadow_progress progress "
                "JOIN shadow_positions position ON position.position_id=progress.position_id "
                "WHERE position.proposed_trade_output_id IN ('proposal', 'proposal-33') "
                "ORDER BY position.proposed_trade_output_id"
            )
        )
        assert tuple(tuple(row) for row in schedule) == (
            ("proposal", 2, 18),
            ("proposal-33", 1, 10),
        )
        assert connection.execute("SELECT count(*) FROM shadow_positions").fetchone()[0] == 33

    assert restarted.run_once(now_us=1_000) == 0
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT min(schedule_count) FROM shadow_progress").fetchone()[0] == 2
        )


def test_open_recovery_uses_durable_progress_after_entry_event_is_pruned(tmp_path: Path) -> None:
    database = tmp_path / "pruned-entry.sqlite3"
    _seed(database)
    policy = _policy()
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=2) == 1
    raw = sqlite3.connect(database)
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute("DELETE FROM market_events WHERE event_id='entry'")
        raw.commit()
    finally:
        raw.close()

    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=12) == 1
    with connect_v2(database) as connection:
        assert connection.execute("SELECT lifecycle FROM shadow_positions").fetchone()[0] == (
            "closed"
        )


def test_retention_preserves_unprocessed_open_evidence_until_final_horizon(
    tmp_path: Path,
) -> None:
    database = tmp_path / "open-retention.sqlite3"
    _seed(database)
    policy = ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=100),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=0),
        horizons_us=(1_000,),
    )
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=2) == 1

    RetentionManager(database, RetentionPolicy(raw_market_event_us=10)).run(
        now_us=100,
        measured_database_bytes=1,
        measured_wal_bytes=0,
    )

    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT event_id FROM market_events WHERE event_id='exit'"
            ).fetchone()[0]
            == "exit"
        )


def test_retention_does_not_preserve_unrelated_instrument_in_same_shadow_run(
    tmp_path: Path,
) -> None:
    database = tmp_path / "instrument-retention.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, exchange, "
            "currency) "
            "VALUES ('MSFT', ?, 'stock', 'MSFT', 'SMART', 'USD')",
            (_hash("MSFT"),),
        )
        _insert_quote(
            connection,
            sequence=4,
            event_id="unrelated-msft",
            instrument_id="MSFT",
            bid=200.0,
            ask=201.0,
            event_at_us=12,
        )
    policy = ShadowPolicy(
        fill=ShadowFillPolicy(model_id="quote-v1", max_quote_age_us=100),
        cost=ShadowCostPolicy(model_id="cost-v1", per_side_bps=0),
        horizons_us=(1_000,),
    )
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=2) == 1

    RetentionManager(database, RetentionPolicy(raw_market_event_us=10)).run(
        now_us=100,
        measured_database_bytes=1,
        measured_wal_bytes=0,
    )

    with connect_v2(database) as connection:
        retained_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT event_id FROM market_events WHERE event_id IN ('exit', 'unrelated-msft')"
            )
        }
    assert retained_ids == {"exit"}


def test_one_shadow_failure_records_one_incident_and_other_proposal_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "isolated.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
            "subject_instrument_id, emitted_at_us, as_of_at_us, valid_until_at_us, direction, "
            "strength, confidence, horizon_us, first_input_event_id, last_input_event_id, "
            "input_watermark, input_events_hash, output_ordinal, payload_json, payload_hash, "
            "content_hash, data_class, authority_status) SELECT 'proposal-2', run_id, instance_id, "
            "output_kind, subject_instrument_id, emitted_at_us, as_of_at_us, valid_until_at_us, "
            "direction, strength, confidence, horizon_us, first_input_event_id, "
            "last_input_event_id, "
            "input_watermark, input_events_hash, 1, payload_json, payload_hash, ?, data_class, "
            "authority_status FROM idea_outputs WHERE output_id='proposal'",
            ("f" * 64,),
        )
        connection.execute("INSERT INTO idea_output_inputs VALUES ('proposal-2', 'input', 0)")
        connection.execute(
            "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, target, "
            "quantity_value, currency) VALUES ('proposal-2', 0, 'AAPL', 'buy', 'long', 1, 'USD')"
        )
    engine = ShadowEngine(database, run_id="run", policy=_policy())
    original = engine._advance

    def isolated_failure(connection: Any, proposal: Any, **kwargs: Any) -> tuple[int, int]:
        if proposal["output_id"] == "proposal":
            raise RuntimeError("fixture failure")
        return original(connection, proposal, **kwargs)

    monkeypatch.setattr(engine, "_advance", isolated_failure)
    assert engine.run_once(now_us=2) == 1
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM incidents").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT lifecycle FROM shadow_positions WHERE proposed_trade_output_id='proposal-2'"
            ).fetchone()[0]
            == "open"
        )


def test_policy_content_and_hash_survive_restart_and_changed_policy_is_rejected(
    tmp_path: Path,
) -> None:
    database = tmp_path / "policy.sqlite3"
    _seed(database)
    policy = _policy()
    engine = ShadowEngine(database, run_id="run", policy=policy)
    assert engine.run_once(now_us=2) == 1

    with connect_v2(database) as connection:
        policy_json, policy_hash = tuple(
            connection.execute("SELECT policy_json, policy_hash FROM shadow_positions").fetchone()
        )
    assert json.loads(policy_json) == {
        "cost": policy.cost.model_dump(mode="json"),
        "fill": policy.fill.model_dump(mode="json"),
        "horizons_us": [10],
    }
    assert policy_hash == _hash(policy_json)
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=3) == 0

    changed = _policy(per_side_bps=11)
    with pytest.raises(ValueError, match="persisted position policy"):
        ShadowEngine(database, run_id="run", policy=changed).run_once(now_us=3)
    with (
        connect_v2(database) as connection,
        pytest.raises(sqlite3.IntegrityError, match="shadow_policy_immutable"),
    ):
        connection.execute(
            "UPDATE shadow_positions SET policy_json='{}' WHERE proposed_trade_output_id='proposal'"
        )


def test_multileg_invalid_quantity_invalidates_whole_proposal_without_partial_legs(
    tmp_path: Path,
) -> None:
    database = tmp_path / "invalid-multileg.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 100.0, 101.0, 1),
            (2, "aapl-entry", "AAPL", 102.0, 103.0, 2),
            (3, "msft-entry", "MSFT", 202.0, 203.0, 2),
        ),
        legs=(("AAPL", "buy", 1.0, "USD"), ("MSFT", "sell", None, "USD")),
    )

    assert ShadowEngine(database, run_id="run", policy=_policy()).run_once(now_us=2) == 1
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute("SELECT lifecycle, invalid_reason FROM shadow_positions").fetchone()
        ) == ("invalid", "quantity_unavailable")
        assert connection.execute("SELECT count(*) FROM shadow_legs").fetchone()[0] == 0


def test_multileg_entry_is_atomic_when_one_leg_quote_is_invalid(tmp_path: Path) -> None:
    database = tmp_path / "quote-multileg.sqlite3"
    _seed_case(
        database,
        quotes=(
            (1, "input", "AAPL", 100.0, 101.0, 1),
            (2, "aapl-entry", "AAPL", 102.0, 103.0, 2),
            (3, "msft-crossed", "MSFT", 203.0, 202.0, 3),
        ),
        legs=(("AAPL", "buy", 1.0, "USD"), ("MSFT", "sell", 2.0, "USD")),
    )
    engine = ShadowEngine(database, run_id="run", policy=_policy())

    assert engine.run_once(now_us=3) == 0
    with connect_v2(database) as connection:
        assert connection.execute("SELECT lifecycle FROM shadow_positions").fetchone()[0] == (
            "pending"
        )
        assert connection.execute("SELECT count(*) FROM shadow_legs").fetchone()[0] == 0
        _insert_quote(
            connection,
            sequence=4,
            event_id="msft-valid",
            instrument_id="MSFT",
            bid=201.0,
            ask=202.0,
            event_at_us=4,
        )

    assert engine.run_once(now_us=4) == 1
    with connect_v2(database) as connection:
        assert tuple(
            connection.execute("SELECT lifecycle, opened_at_us FROM shadow_positions").fetchone()
        ) == ("open", 4)
        assert tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT entry_market_event_id FROM shadow_legs ORDER BY leg_number"
            )
        ) == (("aapl-entry",), ("msft-valid",))


def test_duplicate_proposal_retry_repeated_runs_and_open_recovery_are_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "idempotent.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO idea_outputs SELECT * FROM idea_outputs "
            "WHERE output_id='proposal'"
        )
        assert connection.execute("SELECT count(*) FROM idea_outputs").fetchone()[0] == 1

    policy = _policy()
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=2) == 1
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=2) == 0
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=12) == 1
    assert ShadowEngine(database, run_id="run", policy=policy).run_once(now_us=12) == 0

    with connect_v2(database) as connection:
        assert tuple(
            connection.execute(
                "SELECT (SELECT count(*) FROM shadow_positions), "
                "(SELECT count(*) FROM shadow_legs), "
                "(SELECT count(*) FROM shadow_outcomes)"
            ).fetchone()
        ) == (1, 1, 1)
        assert connection.execute("SELECT lifecycle FROM shadow_positions").fetchone()[0] == (
            "closed"
        )


class _MarketDataOnlyFake:
    def set_callback(self, callback: Any) -> None:
        self.callback = callback

    def set_disconnect_callback(self, callback: Any) -> None:
        self.disconnect_callback = callback

    def set_status_callback(self, callback: Any) -> None:
        self.status_callback = callback

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def configure_subscriptions(self, subscriptions: Any) -> None:
        del subscriptions

    def subscribe(self, fence: Any) -> None:
        del fence

    def cancel(self, request_id: int) -> None:
        del request_id


def _recorder_config(database: Path, *, mode: str) -> RecorderConfig:
    return RecorderConfig.model_validate(
        {
            "database": database,
            "run_id": f"{mode}-run",
            "owner_id": "test-owner",
            "mode": mode,
            "host": "127.0.0.1",
            "port": 4001,
            "client_id": 71,
            "read_only": True,
            "external_read_only_verified": True,
            "config_hash": "a" * 64,
            "git_commit": "deadbee",
        }
    )


def test_recorder_invokes_shadow_engine_only_in_shadow_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[str] = []
    advanced: list[tuple[str, int]] = []

    class ShadowSpy:
        def __init__(self, database: Path, *, run_id: str) -> None:
            del database
            self.run_id = run_id
            constructed.append(run_id)

        def run_once(self, *, now_us: int) -> int:
            advanced.append((self.run_id, now_us))
            return 0

    monkeypatch.setattr("stocker_runtime.ingestion.recorder.ShadowEngine", ShadowSpy)
    for mode in ("prospective_record", "shadow"):
        database = tmp_path / f"{mode}.sqlite3"
        initialize_database(database)
        recorder = Recorder(_recorder_config(database, mode=mode), _MarketDataOnlyFake())
        recorder.start(now_us=100, instruments=(), subscriptions=())
        assert recorder.drain(now_us=101) == 0
        recorder.stop(now_us=102)

    assert constructed == ["shadow-run"]
    assert advanced == [("shadow-run", 100), ("shadow-run", 101)]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_shadow_package_has_no_broker_risk_execution_or_account_authority() -> None:
    source_paths = tuple(sorted(SHADOW_PACKAGE.rglob("*.py")))
    imported = {module for path in source_paths for module in _imported_modules(path)}
    forbidden_roots = {
        "ibapi",
        "stocker_execution",
        "stocker_prospective",
        "stocker_risk",
    }
    forbidden_runtime = {
        "stocker_runtime.ingestion",
        "stocker_runtime.risk",
        "stocker_runtime.execution",
    }
    assert not ({module.split(".", maxsplit=1)[0] for module in imported} & forbidden_roots)
    assert not any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for module in imported
        for forbidden in forbidden_runtime
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    assert not {
        "account_id",
        "broker_fill_id",
        "broker_order_id",
        "broker_position_id",
        "order_intent_id",
        "place_order",
        "risk_approval_id",
        "submit_order",
    } & set(source.replace("(", " ").replace(")", " ").replace(":", " ").split())


def test_shadow_records_only_virtual_market_evidence_not_broker_claims(tmp_path: Path) -> None:
    database = tmp_path / "claims.sqlite3"
    _seed(database)
    engine = ShadowEngine(database, run_id="run", policy=_policy())
    assert engine.run_once(now_us=2) == 1
    assert engine.run_once(now_us=12) == 1

    forbidden = {
        "account_id",
        "broker_fill_id",
        "broker_order_id",
        "broker_position_id",
        "execution_id",
        "order_id",
        "risk_approval_id",
    }
    with connect_v2(database) as connection:
        columns = {
            str(row[1])
            for table in ("shadow_positions", "shadow_legs", "shadow_marks", "shadow_outcomes")
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        payloads = tuple(
            json.loads(str(row[0]))
            for table in ("shadow_marks", "shadow_outcomes")
            for row in connection.execute(f"SELECT payload_json FROM {table}")
        )
    assert not (columns & forbidden)
    assert all(not (set(payload) & forbidden) for payload in payloads)
