from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from stocker_ideas.plugins.opening_leader_continuation_v0 import MANIFEST
from stocker_runtime.domain import (
    JsonValue,
    MarketEvent,
    Observation,
    ProposedPosition,
    ProposedTradeLeg,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)
from stocker_runtime.ideas import (
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    IdeaPlugin,
    MarketDataRequirement,
    deterministic_idea_output_id,
)
from stocker_runtime.ideas.discovery import (
    IdeaConfig,
    IdeaDiscoveryError,
    aggregate_requirements,
    discover_plugins,
    load_idea_configs,
    reviewed_code_hash,
)
from stocker_runtime.ideas.runner import IdeaRunner
from stocker_runtime.ingestion import (
    IBKRSubscription,
    InstrumentSpec,
    Recorder,
    RecorderConfig,
    SubscriptionSpec,
)
from stocker_runtime.ingestion.official_bridge import create_official_bridge
from stocker_runtime.storage.connection import connect_v2, initialize_database
from stocker_runtime.storage.repository import (
    IdeaOutputRecord,
    IdentityCollisionError,
    OperationalRepository,
    deterministic_output_id,
)
from stocker_runtime.storage.retention import RetentionManager, RetentionPolicy

MODULE = "stocker_ideas.plugins.opening_leader_continuation_v0"
COHORT = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)


def _config(**changes: object) -> IdeaConfig:
    spec = importlib.util.find_spec(MODULE)
    assert spec is not None and spec.origin is not None
    values: dict[str, object] = {
        "module": MODULE,
        "factory": "create_plugin",
        "expected_code_hash": reviewed_code_hash(MODULE),
        "expected_manifest_hash": hashlib.sha256(MANIFEST.to_canonical_json()).hexdigest(),
        "parameters": {"checkpoints": [6, 12], "minimum_complete_slate": 15},
        "universe": COHORT,
    }
    values.update(changes)
    return IdeaConfig.model_validate(values)


def _seed(database: Path, *, mode: str = "prospective_record") -> None:
    initialize_database(database)
    data_class = "prospective_protected" if mode == "prospective_record" else "shadow_protected"
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, 'ibkr', 1, NULL, ?, 'fixture', ?, 'running', NULL)",
            ("run-1", mode, "a" * 64, data_class),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-1', 1, 'fixture', 1)"
        )
        for symbol in COHORT:
            connection.execute(
                "INSERT INTO instruments(instrument_id, identity_hash, kind, symbol, "
                "exchange, currency) "
                "VALUES (?, ?, 'stock', ?, 'SMART', 'USD')",
                (symbol, hashlib.sha256(symbol.encode()).hexdigest(), symbol),
            )


def _event(
    connection: object,
    sequence: int,
    symbol: str,
    close: float,
    *,
    session: str = "2026-08-03",
) -> None:
    del session
    event_at_us = int(datetime(2026, 8, 3, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    payload: Mapping[str, JsonValue] = {
        "event_at_us": event_at_us,
        "open": 100.0,
        "high": max(100.0, close),
        "low": min(100.0, close),
        "close": close,
        "volume": 1.0,
    }
    _persist_market_event(
        connection,
        sequence,
        MarketEvent(
            event_id=f"event-{sequence}",
            instrument_id=symbol,
            feed_kind="bars",
            event_kind="bar",
            event_at_us=event_at_us,
            received_at_us=event_at_us + sequence,
            payload=payload,
        ),
    )


def _persist_market_event(
    connection: object, sequence: int, event: MarketEvent, *, run_id: str = "run-1"
) -> None:
    payload = event.payload
    payload_json = canonical_json_bytes(payload).decode()
    connection.execute(  # type: ignore[attr-defined]
        "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, recorder_generation, "
        "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
        "VALUES (?, ?, ?, 1, 1, 'bar', ?, ?, 'pending')",
        (
            sequence,
            event.event_id,
            run_id,
            event.received_at_us,
            hashlib.sha256(payload_json.encode()).hexdigest(),
        ),
    )
    connection.execute(  # type: ignore[attr-defined]
        "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, feed_kind, "
        "event_kind, event_at_us, received_at_us, connection_generation, payload_json, "
        "payload_sha256) "
        "VALUES (?, ?, ?, ?, 'bars', 'bar', ?, ?, 1, ?, ?)",
        (
            event.event_id,
            run_id,
            sequence,
            event.instrument_id,
            event.event_at_us,
            event.received_at_us,
            payload_json,
            hashlib.sha256(payload_json.encode()).hexdigest(),
        ),
    )


def test_empty_config_discovers_and_runs_nothing() -> None:
    assert discover_plugins(()) == ()
    example = load_idea_configs(Path("configs/ideas/v2.example.json"))
    assert len(example) == 1
    assert discover_plugins(example) == ()


def test_incremental_lineage_contract_is_bounded() -> None:
    with pytest.raises(ValueError, match="at most 256 items"):
        IdeaEvaluation(
            state={},
            outputs=(),
            retained_input_event_ids=tuple(f"event-{index}" for index in range(257)),
        )


def test_discovery_is_deterministic_and_freezes_hashes() -> None:
    first = discover_plugins((_config(),))
    second = discover_plugins((_config(),))
    assert len(first) == 1
    assert first[0].identity == second[0].identity
    assert first[0].manifest.idea_id == "opening_leader_continuation"
    assert len(first[0].code_hash) == len(first[0].manifest_hash) == 64
    assert first[0].manifest_json == first[0].manifest.to_canonical_json().decode()


@pytest.mark.parametrize("module", ["os", "stocker_prospective.opening_leader_continuation_v0"])
def test_discovery_rejects_modules_outside_first_party_package(module: str) -> None:
    with pytest.raises(ValueError, match="first-party"):
        _config(module=module)


def test_discovery_rejects_duplicate_identity_and_bad_parameters() -> None:
    with pytest.raises(IdeaDiscoveryError, match="duplicate"):
        discover_plugins((_config(), _config()))
    with pytest.raises(IdeaDiscoveryError, match="parameter"):
        discover_plugins((_config(parameters={"checkpoints": [9]}),))


@pytest.mark.parametrize("checkpoints", ([6, 6], [12, 6], [12, 12]))
def test_reference_plugin_rejects_nonfrozen_checkpoint_identity(
    checkpoints: list[int],
) -> None:
    with pytest.raises(IdeaDiscoveryError, match="parameter"):
        discover_plugins(
            (
                _config(
                    parameters={
                        "checkpoints": checkpoints,
                        "minimum_complete_slate": 15,
                    }
                ),
            )
        )


@pytest.mark.parametrize("universe", (COHORT[:-1], tuple(reversed(COHORT))))
def test_reference_plugin_rejects_noncanonical_universe(universe: tuple[str, ...]) -> None:
    with pytest.raises(IdeaDiscoveryError, match="requirements"):
        discover_plugins((_config(universe=universe),))


@pytest.mark.parametrize("minimum", (14, 16))
def test_reference_plugin_rejects_nonfrozen_minimum(minimum: int) -> None:
    with pytest.raises(IdeaDiscoveryError, match="parameter"):
        discover_plugins(
            (
                _config(
                    parameters={
                        "checkpoints": [6, 12],
                        "minimum_complete_slate": minimum,
                    }
                ),
            )
        )


def test_discovery_rejects_unreviewed_hash_and_aggregates_requirements() -> None:
    with pytest.raises(IdeaDiscoveryError, match="code hash"):
        discover_plugins((_config(expected_code_hash="0" * 64),))
    plugin = discover_plugins((_config(),))[0]
    aggregated = aggregate_requirements((plugin,))
    assert len(aggregated) == 20
    assert aggregated == tuple(sorted(aggregated, key=lambda item: item.to_canonical_json()))


def test_reference_plugin_uses_supported_phase3_raw_bar_subscription() -> None:
    plugin = discover_plugins((_config(),))[0]
    assert {requirement.feed_kind for requirement in plugin.requirements} == {"bars"}
    assert {requirement.cadence for requirement in plugin.requirements} == {"5s"}


def test_runner_and_repository_share_one_complete_output_identity_contract() -> None:
    leg = ProposedTradeLeg(
        instrument_id="AAL",
        action="buy",
        target="long",
        quantity_value=1.0,
        currency="USD",
    )
    record = IdeaOutputRecord(
        run_id="run-1",
        instance_id="instance-1",
        output_kind="proposed_trade",
        subject_instrument_id="AAL",
        emitted_at_us=200,
        as_of_at_us=150,
        valid_until_at_us=None,
        direction=None,
        strength=None,
        confidence=None,
        horizon_us=None,
        first_input_event_id="event-1",
        last_input_event_id="event-2",
        input_event_ids=("event-1", "event-2"),
        output_ordinal=3,
        payload={"rank": 1},
        data_class="prospective_protected",
        authority_status="unapproved",
        legs=(leg,),
    )
    shared = deterministic_idea_output_id(
        instance_id=record.instance_id,
        input_event_ids=record.input_event_ids,
        output_kind=record.output_kind,
        output_ordinal=record.output_ordinal,
        as_of_at_us=record.as_of_at_us,
        payload=record.payload,
        legs=record.legs,
    )
    assert deterministic_output_id(record) == shared
    variants = (
        replace(record, instance_id="instance-2"),
        replace(
            record,
            input_event_ids=("event-2", "event-1"),
            first_input_event_id="event-2",
            last_input_event_id="event-1",
        ),
        replace(record, output_kind="observation", authority_status="recorded", legs=()),
        replace(record, output_ordinal=4),
        replace(record, as_of_at_us=151),
        replace(record, payload={"rank": 2}),
        replace(
            record,
            legs=(
                ProposedTradeLeg(
                    instrument_id="AAL",
                    action="buy",
                    target="long",
                    quantity_value=2.0,
                    currency="USD",
                ),
            ),
        ),
    )
    assert all(deterministic_output_id(item) != shared for item in variants)


def test_bad_source_pin_is_rejected_before_plugin_module_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(_module: str) -> object:
        raise AssertionError("unreviewed plugin module executed")

    monkeypatch.setattr(
        "stocker_runtime.ideas.discovery.importlib.import_module", unexpected_import
    )
    with pytest.raises(IdeaDiscoveryError, match="code hash"):
        discover_plugins((_config(expected_code_hash="0" * 64),))


def test_review_pin_includes_executed_package_initializers() -> None:
    modules = ("stocker_ideas", "stocker_ideas.plugins", MODULE)
    sources: dict[str, bytes] = {}
    for module in modules:
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None
        sources[module] = Path(spec.origin).read_bytes()
    framed = b"".join(
        len(module.encode()).to_bytes(4, "big")
        + module.encode()
        + len(source).to_bytes(8, "big")
        + source
        for module, source in sorted(sources.items())
    )
    assert reviewed_code_hash(MODULE) == hashlib.sha256(framed).hexdigest()


def test_review_pin_tracks_transitive_helpers_and_rejects_forbidden_helper_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stocker_ideas

    package_spec = stocker_ideas.__spec__
    assert package_spec is not None and package_spec.submodule_search_locations is not None
    package_root = tmp_path / "stocker_ideas"
    plugin_root = package_root / "plugins"
    plugin_root.mkdir(parents=True)
    module = "stocker_ideas.plugins.phase4_test_plugin"
    helper = "stocker_ideas.plugins.phase4_test_helper"
    (plugin_root / "phase4_test_plugin.py").write_text(
        f"from {helper} import VALUE\n", encoding="utf-8"
    )
    helper_path = plugin_root / "phase4_test_helper.py"
    helper_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        package_spec,
        "submodule_search_locations",
        [*package_spec.submodule_search_locations, str(package_root)],
    )

    first = reviewed_code_hash(module)
    helper_path.write_text("VALUE = 2\n", encoding="utf-8")
    assert reviewed_code_hash(module) != first
    helper_path.write_text("import os\nVALUE = 2\n", encoding="utf-8")
    with pytest.raises(IdeaDiscoveryError, match="forbidden plugin import: os"):
        reviewed_code_hash(module)


def test_reference_plugin_has_static_authority_boundary() -> None:
    spec = importlib.util.find_spec(MODULE)
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(
        name.startswith(
            ("stocker_prospective", "stocker_runtime.storage", "stocker_runtime.ingestion")
        )
        for name in imports
    )
    assert not (
        {"place_order", "submit_order", "account_id", "broker_order_id"} & set(source.split())
    )


def test_activation_freezes_identity_and_rejects_cross_mode(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))
    runner = IdeaRunner(database, discovered, run_id="run-1")
    first = runner.activate(run_id="run-1", plugin=discovered[0], activated_at_us=100)
    retry = runner.activate(run_id="run-1", plugin=discovered[0], activated_at_us=100)
    assert first.instance_id == retry.instance_id
    with connect_v2(database) as connection:
        row = connection.execute("SELECT * FROM idea_instances").fetchone()
    assert row["plugin_code_hash"] == discovered[0].code_hash
    assert row["parameters_hash"] == discovered[0].parameters_hash
    assert row["universe_hash"] == discovered[0].universe_hash

    shadow_db = tmp_path / "shadow.sqlite3"
    _seed(shadow_db, mode="shadow")
    incompatible = discover_plugins((_config(),))[0]
    # Opening Leader is deliberately prospective and shadow compatible.
    assert (
        IdeaRunner(shadow_db, (incompatible,), run_id="run-1")
        .activate(run_id="run-1", plugin=incompatible, activated_at_us=100)
        .protected_data_class
        is ProtectedDataClass.SHADOW
    )


def test_runner_has_no_backfill_and_commits_outputs_with_checkpoint(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 150.0)
    plugin = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (plugin,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=plugin, activated_at_us=100)
    boundary = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)
    evidence = [
        _raw_bar_event(100, "AAL", boundary, received_at=boundary),
        *_checkpoint_evidence(6, COHORT[1:], sequence_start=200),
    ]
    with connect_v2(database) as connection:
        for source_sequence, event in enumerate(evidence, start=2):
            _persist_market_event(connection, source_sequence, event)

    results = runner.run_once(now_us=10_000)
    assert results[0].advanced is True
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_market_event_id FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()[0]
        outputs = connection.execute(
            "SELECT output_kind, subject_instrument_id, authority_status, payload_json "
            "FROM idea_outputs ORDER BY output_ordinal"
        ).fetchall()
        legs = connection.execute(
            "SELECT instrument_id, action, target, quantity_value, currency FROM idea_output_legs"
        ).fetchall()
    assert checkpoint == evidence[-1].event_id
    # AAL's later progress closes the barrier but cannot recover its pre-activation open.
    assert [row["output_kind"] for row in outputs] == ["observation", "signal", "proposed_trade"]
    assert outputs[-1]["authority_status"] == "unapproved"
    assert json.loads(outputs[-1]["payload_json"])["selected_identity"] == "rank_1"
    assert [tuple(row) for row in legs] == [
        (outputs[-1]["subject_instrument_id"], "buy", "long", 1.0, "USD")
    ]


def test_one_plugin_failure_does_not_stop_other_instance(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed(database)
    good = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (good,), run_id="run-1")
    failed = runner.activate(run_id="run-1", plugin=good, activated_at_us=100)
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE idea_instances SET deactivated_at_us=100 WHERE instance_id=?",
            (failed.instance_id,),
        )
    healthy = runner.activate(run_id="run-1", plugin=good, activated_at_us=101)
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE idea_instances SET deactivated_at_us=NULL WHERE instance_id=?",
            (failed.instance_id,),
        )
    runner._plugins_by_instance[failed.instance_id] = object()  # explicit fault injection boundary
    with connect_v2(database) as connection:
        for index, symbol in enumerate(COHORT, start=1):
            _event(connection, index, symbol, 101.0 + index / 100)

    outcomes = runner.run_once(now_us=20_000)
    by_instance = {item.instance_id: item for item in outcomes}
    assert by_instance[failed.instance_id].advanced is False
    assert by_instance[healthy.instance_id].advanced is True
    with connect_v2(database) as connection:
        health = dict(connection.execute("SELECT instance_id, health FROM idea_instances"))
    assert health[failed.instance_id] == "degraded"
    assert health[healthy.instance_id] == "healthy"


def test_opening_leader_golden_ranking_is_deterministic() -> None:
    discovered = discover_plugins((_config(),))[0]
    activation = discovered.activation(
        instance_id="fixture",
        run_id="run",
        data_class=ProtectedDataClass.PROSPECTIVE,
        activated_at_us=1,
    )
    plugin = discovered.plugin
    session_open = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)
    boundary = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)
    events: list[MarketEvent] = []
    for index, symbol in enumerate(COHORT):
        events.extend(
            (
                _raw_bar_event(index * 2, symbol, session_open),
                _raw_bar_event(
                    index * 2 + 1,
                    symbol,
                    boundary - timedelta(seconds=5),
                    close=102.0 if symbol in {"AAL", "AAOI"} else 101.0,
                    received_at=boundary,
                ),
            )
        )
    batch = _batch(tuple(events))
    one = plugin.evaluate(batch, {})
    two = plugin.evaluate(batch, {})
    assert one == two
    assert [output.kind for output in one.outputs] == ["observation", "signal", "proposed_trade"]
    assert one.outputs[1].subject_instrument_id == "AAL"
    assert one.outputs[1].payload["rank_1_minus_rank_2_bps"] == 0.0
    assert activation.parameters_hash == discovered.parameters_hash


def test_unresolved_required_gap_blocks_only_that_instance(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES "
            "('sub-aal', 'run-1', 1, 1, 'AAL', 'bars', 1, 'active', ?, 1)",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO gaps(gap_id, run_id, subscription_id, started_at_us, reason, "
            "data_loss_possible, continuity_required) VALUES "
            "('gap-aal', 'run-1', 'sub-aal', 2, 'STREAM_STALE', 0, 1)"
        )
        _event(connection, 1, "AAL", 101.0)

    assert runner.run_once(now_us=1_000)[0].advanced is False
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_market_event_id FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()[0]
    assert checkpoint is None


def test_runner_batches_at_256_without_skipping_the_remainder(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        for sequence in range(1, 258):
            _event(connection, sequence, COHORT[sequence % len(COHORT)], 101.0)

    first = runner.run_once(now_us=2_000)[0]
    second = runner.run_once(now_us=3_000)[0]
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_market_event_id FROM idea_checkpoints"
        ).fetchone()[0]
    assert first.advanced is True
    assert second.advanced is True
    assert checkpoint == "event-257"


@pytest.mark.parametrize("restart", (False, True))
def test_incremental_output_persists_prior_checkpoint_lineage_and_survives_retention(
    tmp_path: Path, restart: bool
) -> None:
    database = tmp_path / f"lineage-{restart}.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    first_evidence = _checkpoint_evidence(6, COHORT[:10])
    with connect_v2(database) as connection:
        for sequence, event in enumerate(first_evidence, start=1):
            _persist_market_event(connection, sequence, event)
    assert runner.run_once(now_us=2_000)[0].output_count == 0
    with connect_v2(database) as connection:
        retained = json.loads(
            connection.execute(
                "SELECT state_input_event_ids_json FROM idea_checkpoints WHERE instance_id=?",
                (activation.instance_id,),
            ).fetchone()[0]
        )
    assert retained == [event.event_id for event in first_evidence]
    retention_now = int(datetime(2026, 8, 3, 15, 0, tzinfo=UTC).timestamp() * 1_000_000)
    RetentionManager(database, policy=RetentionPolicy(completed_bar_us=1)).run(now_us=retention_now)
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM market_events").fetchone()[0] == 20
    if restart:
        runner.close()
        runner = IdeaRunner(database, (discovered,), run_id="run-1")
    remaining = _checkpoint_evidence(6, COHORT[10:], sequence_start=20)
    with connect_v2(database) as connection:
        for sequence, event in enumerate(remaining, start=21):
            _persist_market_event(connection, sequence, event)
    assert runner.run_once(now_us=retention_now + 1)[0].output_count == 3
    with connect_v2(database) as connection:
        rows = connection.execute(
            "SELECT input_ordinal, event_id FROM idea_output_inputs "
            "WHERE output_id=(SELECT output_id FROM idea_outputs WHERE output_ordinal=0) "
            "ORDER BY input_ordinal"
        ).fetchall()
        final_retained = connection.execute(
            "SELECT state_input_event_ids_json FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()[0]
    runner.close()
    expected_lineage = (*first_evidence, *remaining)
    assert [tuple(row) for row in rows] == [
        (ordinal, event.event_id) for ordinal, event in enumerate(expected_lineage)
    ]
    assert json.loads(final_retained) == [
        event.event_id for ordinal, event in enumerate(expected_lineage) if ordinal % 2 == 0
    ]


def test_incomplete_prior_session_lineage_is_released_after_rollover(tmp_path: Path) -> None:
    database = tmp_path / "lineage-release.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    prior = _checkpoint_evidence(6, COHORT[:10])
    with connect_v2(database) as connection:
        for sequence, event in enumerate(prior, start=1):
            _persist_market_event(connection, sequence, event)
    assert runner.run_once(now_us=2_000)[0].output_count == 0
    retention_now = int(datetime(2026, 8, 3, 15, 0, tzinfo=UTC).timestamp() * 1_000_000)
    RetentionManager(database, policy=RetentionPolicy(completed_bar_us=1)).run(now_us=retention_now)
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM market_events").fetchone()[0] == 20
        next_open = _raw_bar_event(100, "AAL", datetime(2026, 8, 4, 13, 30, tzinfo=UTC))
        _persist_market_event(connection, 21, next_open)
    assert runner.run_once(now_us=retention_now + 1)[0].output_count == 0
    RetentionManager(database, policy=RetentionPolicy(completed_bar_us=1)).run(
        now_us=retention_now + 86_400_000_000
    )
    with connect_v2(database) as connection:
        event_ids = tuple(
            row[0] for row in connection.execute("SELECT event_id FROM market_events")
        )
        retained = connection.execute(
            "SELECT state_input_event_ids_json FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()[0]
    runner.close()
    assert event_ids == (next_open.event_id,)
    assert retained == json.dumps([next_open.event_id], separators=(",", ":"))


class _InvalidPlugin:
    def __init__(self, original: IdeaPlugin, failure: str) -> None:
        self._original = original
        self._failure = failure

    @property
    def manifest(self) -> IdeaManifest:
        return self._original.manifest

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        return self._original.requirements(activation)

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        del state
        if self._failure == "forbidden_kind":
            return IdeaEvaluation(
                state={},
                outputs=(
                    ProposedPosition(
                        subject_instrument_id=batch.events[0].instrument_id,
                        as_of_at_us=batch.events[0].event_at_us,
                        payload={"target": "long"},
                    ),
                ),
            )
        if self._failure == "state_overflow":
            return IdeaEvaluation.model_construct(state={"x": "y" * 70_000}, outputs=())
        return IdeaEvaluation(
            state={},
            outputs=(
                Observation(
                    subject_instrument_id="NOT_IN_UNIVERSE",
                    as_of_at_us=batch.events[0].event_at_us,
                    payload={},
                ),
            ),
        )


@pytest.mark.parametrize("failure", ["forbidden_kind", "state_overflow", "subject"])
def test_invalid_plugin_result_degrades_without_partial_checkpoint(
    tmp_path: Path, failure: str
) -> None:
    database = tmp_path / f"{failure}.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    invalid = replace(discovered, plugin=_InvalidPlugin(discovered.plugin, failure))
    runner = IdeaRunner(database, (invalid,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=invalid, activated_at_us=100)
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 101.0)

    result = runner.run_once(now_us=1_000)[0]
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_market_event_id FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()[0]
        output_count = connection.execute("SELECT count(*) FROM idea_outputs").fetchone()[0]
    assert result.advanced is False
    assert checkpoint is None
    assert output_count == 0


def test_activation_uses_durable_inbox_watermark_and_retry_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "activation.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
            "recorder_generation, connection_generation, callback_kind, received_at_us, "
            "payload_sha256, lifecycle) VALUES (1, 'pending-1', 'run-1', 1, 1, "
            "'bar', 10, ?, 'pending')",
            ("1" * 64,),
        )
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    first = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        _event(connection, 2, "AAL", 101.0)
    retry = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        activated_after = connection.execute(
            "SELECT activated_after_source_sequence FROM idea_instances"
        ).fetchone()[0]
    assert activated_after == 1
    assert retry.instance_id == first.instance_id
    assert retry.inserted is False


@pytest.mark.parametrize("evidence", ("market_event", "receipt", "compaction_watermark"))
def test_activation_watermark_survives_terminal_callback_tombstone_retention(
    tmp_path: Path, evidence: str
) -> None:
    database = tmp_path / f"activation-{evidence}.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        if evidence == "market_event":
            payload_json = canonical_json_bytes({"event_at_us": 10}).decode()
            payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
            connection.execute(
                "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, "
                "recorder_generation, connection_generation, callback_kind, received_at_us, "
                "payload_sha256, lifecycle) "
                "VALUES (7, 'retained-event', 'run-1', 1, 1, 'bar', 11, ?, 'pending')",
                (payload_hash,),
            )
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                "payload_json, payload_sha256) VALUES ('retained-event', 'run-1', 7, 'AAL', "
                "'bars', 'bar', 10, 11, 1, ?, ?)",
                (payload_json, payload_hash),
            )
            connection.execute(
                "UPDATE callback_inbox SET lifecycle='acknowledged', "
                "normalized_event_id='retained-event', acknowledged_at_us=12 "
                "WHERE source_sequence=7"
            )
            connection.execute("DELETE FROM callback_inbox WHERE source_sequence=7")
        elif evidence == "receipt":
            connection.execute(
                "INSERT INTO callback_receipts(batch_id, run_id, first_source_sequence, "
                "last_source_sequence, callback_count, first_received_at_us, last_received_at_us, "
                "kind_counts_json, status_counts_json, callback_rows_hash, prior_chain_hash, "
                "chained_payload_hash, created_at_us) VALUES ('receipt-7', 'run-1', 1, 7, 7, "
                "1, 7, '{}', '{}', ?, ?, ?, 8)",
                ("a" * 64, "b" * 64, "c" * 64),
            )
        else:
            connection.execute(
                "INSERT INTO callback_compaction_watermarks(run_id, compacted_through_sequence, "
                "cumulative_callback_count, first_received_at_us, last_received_at_us, "
                "rolled_receipt_chain_hash, last_receipt_chain_hash, updated_at_us) "
                "VALUES ('run-1', 7, 7, 1, 7, ?, ?, 8)",
                ("a" * 64, "b" * 64),
            )
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        activated_after = connection.execute(
            "SELECT activated_after_source_sequence FROM idea_instances"
        ).fetchone()[0]
    runner.close()
    assert activated_after == 7


def test_removing_config_disables_an_active_instance_without_evaluation(tmp_path: Path) -> None:
    database = tmp_path / "disabled.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    activation = IdeaRunner(database, (discovered,), run_id="run-1").activate(
        run_id="run-1", plugin=discovered, activated_at_us=100
    )
    empty_runner = IdeaRunner(database, (), run_id="run-1")
    assert empty_runner.deactivate_unconfigured(run_id="run-1", now_us=101) == (
        activation.instance_id,
    )
    assert empty_runner.run_once(now_us=102) == ()
    with connect_v2(database) as connection:
        row = connection.execute(
            "SELECT health, deactivated_at_us FROM idea_instances WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()
    assert tuple(row) == ("disabled", 101)


def test_runner_is_bound_to_one_active_run_and_never_crosses_stopped_runs(tmp_path: Path) -> None:
    database = tmp_path / "run-bound.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    run1_runner = IdeaRunner(database, (discovered,), run_id="run-1")
    run1 = run1_runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    run1_runner.close()
    with connect_v2(database) as connection:
        connection.execute("UPDATE runs SET status='stopped', ended_at_us=101 WHERE run_id='run-1'")
        connection.execute(
            "INSERT INTO runs VALUES ('run-2', 'prospective_record', 'ibkr', 102, NULL, ?, "
            "'fixture', ?, 'running', NULL)",
            ("b" * 64, "prospective_protected"),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-2', 1, 'fixture', 102)"
        )
    run2_runner = IdeaRunner(database, (discovered,), run_id="run-2")
    run2 = run2_runner.activate(run_id="run-2", plugin=discovered, activated_at_us=103)
    results = run2_runner.run_once(now_us=104)
    run2_runner.close()
    assert tuple(result.instance_id for result in results) == (run2.instance_id,)
    assert run1.instance_id != run2.instance_id


def test_runner_rejects_activation_for_a_different_bound_run(tmp_path: Path) -> None:
    database = tmp_path / "wrong-run-bound.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    with pytest.raises(RuntimeError, match="bound run"):
        runner.activate(run_id="some-other-run", plugin=discovered, activated_at_us=100)


def test_two_simultaneously_active_runs_remain_isolated_by_required_run_id(
    tmp_path: Path,
) -> None:
    database = tmp_path / "two-active-runs.sqlite3"
    _seed(database)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO runs VALUES ('run-2', 'prospective_record', 'ibkr', 2, NULL, ?, "
            "'fixture', ?, 'running', NULL)",
            ("b" * 64, "prospective_protected"),
        )
        connection.execute(
            "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
            "VALUES ('run-2', 1, 'fixture-2', 2)"
        )
    discovered = discover_plugins((_config(),))[0]
    run1_runner = IdeaRunner(database, (discovered,), run_id="run-1")
    run2_runner = IdeaRunner(database, (discovered,), run_id="run-2")
    run1 = run1_runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    run2 = run2_runner.activate(run_id="run-2", plugin=discovered, activated_at_us=100)
    run1_event = _raw_bar_event(1, "AAL", datetime(2026, 8, 3, 13, 30, tzinfo=UTC))
    run2_event = _raw_bar_event(2, "AAOI", datetime(2026, 8, 3, 13, 30, tzinfo=UTC))
    with connect_v2(database) as connection:
        _persist_market_event(connection, 1, run1_event, run_id="run-1")
        _persist_market_event(connection, 2, run2_event, run_id="run-2")

    run1_results = run1_runner.run_once(now_us=1_000)
    run2_results = run2_runner.run_once(now_us=1_000)
    with connect_v2(database) as connection:
        checkpoints = dict(
            connection.execute("SELECT instance_id, last_market_event_id FROM idea_checkpoints")
        )
    run1_runner.close()
    run2_runner.close()
    assert tuple(result.instance_id for result in run1_results) == (run1.instance_id,)
    assert tuple(result.instance_id for result in run2_results) == (run2.instance_id,)
    assert checkpoints == {
        run1.instance_id: run1_event.event_id,
        run2.instance_id: run2_event.event_id,
    }


def _raw_bar_event(
    sequence: int,
    symbol: str,
    event_at: datetime,
    *,
    opening: float = 100.0,
    close: float = 100.0,
    received_at: datetime | None = None,
) -> MarketEvent:
    event_at_us = int(event_at.timestamp() * 1_000_000)
    return MarketEvent(
        event_id=f"raw-{symbol}-{sequence}-{event_at_us}",
        instrument_id=symbol,
        feed_kind="bars",
        event_kind="bar",
        event_at_us=event_at_us,
        received_at_us=int((received_at or event_at).timestamp() * 1_000_000),
        payload={
            "event_at_us": event_at_us,
            "open": opening,
            "high": max(opening, close),
            "low": min(opening, close),
            "close": close,
            "volume": 1.0,
        },
    )


def _checkpoint_evidence(
    checkpoint: int,
    symbols: tuple[str, ...] = COHORT,
    *,
    session: datetime = datetime(2026, 8, 3, 13, 30, tzinfo=UTC),
    sequence_start: int = 0,
) -> tuple[MarketEvent, ...]:
    boundary = session + timedelta(minutes=5 * checkpoint)
    events: list[MarketEvent] = []
    for index, symbol in enumerate(symbols):
        events.extend(
            (
                _raw_bar_event(sequence_start + index * 2, symbol, session),
                _raw_bar_event(
                    sequence_start + index * 2 + 1,
                    symbol,
                    boundary - timedelta(seconds=5),
                    close=101.0 + index / 100,
                    received_at=boundary,
                ),
            )
        )
    return tuple(events)


def _batch(events: tuple[MarketEvent, ...]) -> IdeaBatch:
    return IdeaBatch(
        mode=RuntimeMode.PROSPECTIVE_RECORD,
        events=events,
        input_watermark=events[-1].event_id,
        causal_from_at_us=min(event.event_at_us for event in events),
        causal_through_at_us=max(max(event.event_at_us, event.received_at_us) for event in events),
    )


def test_opening_leader_retains_incremental_state_and_emits_c6_and_c12() -> None:
    plugin = discover_plugins((_config(),))[0].plugin
    first = plugin.evaluate(_batch(_checkpoint_evidence(6, COHORT[:10])), {})
    second = plugin.evaluate(
        _batch(_checkpoint_evidence(6, COHORT[10:15], sequence_start=20)),
        first.state,
    )
    third = plugin.evaluate(
        _batch(_checkpoint_evidence(6, COHORT[15:], sequence_start=30)),
        second.state,
    )
    c12 = plugin.evaluate(_batch(_checkpoint_evidence(12, sequence_start=40)), third.state)
    next_session = plugin.evaluate(
        _batch(
            _checkpoint_evidence(
                6,
                COHORT[:10],
                session=datetime(2026, 8, 4, 13, 30, tzinfo=UTC),
            )
        ),
        c12.state,
    )
    assert first.outputs == ()
    assert second.outputs == ()
    assert [output.kind for output in third.outputs] == ["observation", "signal", "proposed_trade"]
    assert [output.kind for output in c12.outputs] == ["observation", "signal", "proposed_trade"]
    assert next_session.outputs == ()
    next_state = cast(Mapping[str, JsonValue], next_session.state)
    assert next_state["session"] == "2026-08-04"


@pytest.mark.parametrize("checkpoint", (6, 12))
def test_opening_leader_waits_for_complete_cohort_and_is_batch_order_independent(
    checkpoint: int,
) -> None:
    plugin = discover_plugins((_config(),))[0].plugin
    evidence = _checkpoint_evidence(checkpoint)
    one_batch = plugin.evaluate(_batch(evidence), {})
    first = plugin.evaluate(_batch(evidence[:30]), {})
    second = plugin.evaluate(_batch(evidence[30:]), first.state)
    reversed_batch = plugin.evaluate(_batch(tuple(reversed(evidence))), {})

    expected = tuple(output.model_dump(mode="json") for output in one_batch.outputs)
    assert first.outputs == ()
    assert tuple(output.model_dump(mode="json") for output in second.outputs) == expected
    assert tuple(output.model_dump(mode="json") for output in reversed_batch.outputs) == expected
    assert all(output.payload["slate_size"] == 20 for output in one_batch.outputs)


@pytest.mark.parametrize(("valid_count", "output_count"), ((15, 3), (14, 0)))
def test_opening_leader_applies_minimum_after_all_twenty_cross_checkpoint(
    valid_count: int, output_count: int
) -> None:
    plugin = discover_plugins((_config(),))[0].plugin
    session_open = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)
    boundary = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)
    evidence: list[MarketEvent] = []
    for index, symbol in enumerate(COHORT):
        evidence.append(_raw_bar_event(index * 2, symbol, session_open))
        event_at = boundary - timedelta(seconds=5) if index < valid_count else boundary
        evidence.append(
            _raw_bar_event(
                index * 2 + 1,
                symbol,
                event_at,
                close=101.0 + index / 100,
                received_at=boundary,
            )
        )
    evaluation = plugin.evaluate(_batch(tuple(evidence)), {})
    assert len(evaluation.outputs) == output_count
    if evaluation.outputs:
        assert all(output.payload["slate_size"] == valid_count for output in evaluation.outputs)


def test_opening_leader_does_not_emit_incomplete_cohort_at_session_rollover() -> None:
    plugin = discover_plugins((_config(),))[0].plugin
    prior = _checkpoint_evidence(6, COHORT[:15])
    rollover = _raw_bar_event(30, COHORT[0], datetime(2026, 8, 4, 13, 30, tzinfo=UTC))
    evaluation = plugin.evaluate(_batch((*prior, rollover)), {})
    assert evaluation.outputs == ()
    state = cast(Mapping[str, JsonValue], evaluation.state)
    assert state["session"] == "2026-08-04"


def test_opening_leader_excludes_conflicting_raw_duplicate_from_completed_slate() -> None:
    plugin = discover_plugins((_config(),))[0].plugin
    evidence = list(_checkpoint_evidence(6))
    boundary = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)
    evidence.insert(
        -1,
        _raw_bar_event(
            100,
            "AAL",
            boundary - timedelta(seconds=5),
            close=999.0,
            received_at=boundary,
        ),
    )
    evaluation = plugin.evaluate(_batch(tuple(evidence)), {})
    assert [output.kind for output in evaluation.outputs] == [
        "observation",
        "signal",
        "proposed_trade",
    ]
    assert all(output.payload["slate_size"] == 19 for output in evaluation.outputs)


class _SlowPlugin(_InvalidPlugin):
    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        time.sleep(0.2)
        return self._original.evaluate(batch, state)


class _CrashingPlugin(_InvalidPlugin):
    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        del batch, state
        os._exit(17)


def test_hung_plugin_is_terminated_and_does_not_block_healthy_instance(tmp_path: Path) -> None:
    database = tmp_path / "timeout.sqlite3"
    _seed(database)
    good = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (good,), run_id="run-1")
    slow_activation = runner.activate(run_id="run-1", plugin=good, activated_at_us=100)
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE idea_instances SET deactivated_at_us=100 WHERE instance_id=?",
            (slow_activation.instance_id,),
        )
    healthy_activation = runner.activate(run_id="run-1", plugin=good, activated_at_us=101)
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE idea_instances SET deactivated_at_us=NULL WHERE instance_id=?",
            (slow_activation.instance_id,),
        )
        for index, symbol in enumerate(COHORT, start=1):
            _event(connection, index, symbol, 101.0 + index / 100)
    runner._plugins_by_instance[slow_activation.instance_id] = _SlowPlugin(good.plugin, "slow")
    results = {item.instance_id: item for item in runner.run_once(now_us=30_000)}
    runner.close()
    error_code = results[slow_activation.instance_id].error_code
    assert error_code is not None
    assert "50MS" in error_code
    assert results[healthy_activation.instance_id].advanced is True


def test_crashed_worker_is_restarted_and_all_workers_are_closed(tmp_path: Path) -> None:
    database = tmp_path / "crash.sqlite3"
    _seed(database)
    good = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (good,), run_id="run-1")
    crashed = runner.activate(run_id="run-1", plugin=good, activated_at_us=100)
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE idea_instances SET deactivated_at_us=100 WHERE instance_id=?",
            (crashed.instance_id,),
        )
    healthy = runner.activate(run_id="run-1", plugin=good, activated_at_us=101)
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE idea_instances SET deactivated_at_us=NULL WHERE instance_id=?",
            (crashed.instance_id,),
        )
        for sequence, symbol in enumerate(COHORT, start=1):
            _event(connection, sequence, symbol, 101.0 + sequence / 100)
    runner._plugins_by_instance[crashed.instance_id] = _CrashingPlugin(good.plugin, "crash")

    first = {item.instance_id: item for item in runner.run_once(now_us=30_000)}
    assert first[crashed.instance_id].advanced is False
    assert first[healthy.instance_id].advanced is True
    assert crashed.instance_id not in runner._workers
    healthy_process = runner._workers[healthy.instance_id].process
    assert healthy_process.is_alive()

    runner._plugins_by_instance[crashed.instance_id] = good.plugin
    second = {item.instance_id: item for item in runner.run_once(now_us=1_030_000)}
    assert second[crashed.instance_id].advanced is True
    restarted_process = runner._workers[crashed.instance_id].process
    assert restarted_process.is_alive()
    runner.close()
    assert not healthy_process.is_alive()
    assert not restarted_process.is_alive()
    assert runner._workers == {}


def test_repeated_plugin_failure_keeps_one_unresolved_incident(tmp_path: Path) -> None:
    database = tmp_path / "incidents.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    runner._plugins_by_instance[activation.instance_id] = object()
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 101.0)
    runner.run_once(now_us=1_000)
    runner.run_once(now_us=2_000)
    with connect_v2(database) as connection:
        count = connection.execute(
            "SELECT count(*) FROM incidents WHERE plugin_instance_id=? AND resolved_at_us IS NULL",
            (activation.instance_id,),
        ).fetchone()[0]
        failures = connection.execute(
            "SELECT consecutive_failures FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()[0]
    assert count == 1
    assert failures == 1


def test_plugin_incident_reopens_after_recovery_and_recurrence(tmp_path: Path) -> None:
    database = tmp_path / "incident-recurrence.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    runner._plugins_by_instance[activation.instance_id] = object()
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 101.0)
    assert runner.run_once(now_us=1_000)[0].advanced is False
    runner._plugins_by_instance[activation.instance_id] = discovered.plugin
    assert runner.run_once(now_us=1_001_000)[0].advanced is True
    runner._plugins_by_instance[activation.instance_id] = object()
    with connect_v2(database) as connection:
        _event(connection, 2, "AAOI", 102.0)
    assert runner.run_once(now_us=1_002_000)[0].advanced is False
    with connect_v2(database) as connection:
        incidents = connection.execute(
            "SELECT opened_at_us, resolved_at_us FROM incidents WHERE plugin_instance_id=?",
            (activation.instance_id,),
        ).fetchall()
    runner.close()
    assert len(incidents) == 1
    assert tuple(incidents[0]) == (1_002_000, None)


@pytest.mark.parametrize(
    ("reason", "gaps_block", "staleness_block", "initially_blocked"),
    (
        ("STREAM_STALE", False, True, True),
        ("STREAM_STALE", True, False, False),
        ("RECONNECT_UNCERTAINTY", True, False, True),
        ("RECONNECT_UNCERTAINTY", False, True, False),
    ),
)
def test_staleness_and_continuity_flags_block_only_their_gap_class_and_recover(
    tmp_path: Path,
    reason: str,
    gaps_block: bool,
    staleness_block: bool,
    initially_blocked: bool,
) -> None:
    database = tmp_path / f"gap-{reason}-{gaps_block}-{staleness_block}.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    scoped = replace(
        discovered,
        requirements=tuple(
            MarketDataRequirement(
                **{
                    **requirement.model_dump(),
                    "gaps_block": gaps_block,
                    "staleness_block": staleness_block,
                }
            )
            for requirement in discovered.requirements
        ),
    )
    runner = IdeaRunner(database, (scoped,), run_id="run-1")
    runner.activate(run_id="run-1", plugin=scoped, activated_at_us=100)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES "
            "('gap-aal', 'run-1', 1, 1, 'AAL', 'bars', 1, 'active', ?, 1)",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO gaps(gap_id, run_id, subscription_id, started_at_us, reason, "
            "data_loss_possible, continuity_required) VALUES "
            "('scoped-gap', 'run-1', 'gap-aal', 2, ?, 0, 0)",
            (reason,),
        )
        _event(connection, 1, "AAL", 101.0)
    first = runner.run_once(now_us=1_000)[0]
    assert first.advanced is not initially_blocked
    if initially_blocked:
        with connect_v2(database) as connection:
            connection.execute(
                "UPDATE gaps SET ended_at_us=2, resolved_at_us=2 WHERE gap_id='scoped-gap'"
            )
        assert runner.run_once(now_us=2_000)[0].advanced is True
    runner.close()


@pytest.mark.parametrize(
    ("started_at_us", "ended_at_us", "resolved_at_us", "blocked"),
    (
        (1, 99, None, False),
        (1, 100, None, True),
        (99, 101, None, True),
        (101, None, None, True),
        (99, 101, 102, False),
    ),
)
def test_gap_blocking_is_scoped_to_activation_interval(
    tmp_path: Path,
    started_at_us: int,
    ended_at_us: int | None,
    resolved_at_us: int | None,
    blocked: bool,
) -> None:
    database = tmp_path / f"gap-scope-{started_at_us}-{ended_at_us}-{resolved_at_us}.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES "
            "('gap-scope-aal', 'run-1', 1, 1, 'AAL', 'bars', 1, 'active', ?, 1)",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO gaps(gap_id, run_id, subscription_id, started_at_us, ended_at_us, "
            "reason, data_loss_possible, continuity_required, resolved_at_us) VALUES "
            "('scoped-gap', 'run-1', 'gap-scope-aal', ?, ?, 'RECONNECT_UNCERTAINTY', "
            "0, 1, ?)",
            (started_at_us, ended_at_us, resolved_at_us),
        )
        _event(connection, 1, "AAL", 101.0)
    result = runner.run_once(now_us=1_000)[0]
    runner.close()
    assert result.advanced is not blocked


class _RecorderAdapter:
    def set_callback(self, callback: object) -> None:
        self.callback = callback

    def set_disconnect_callback(self, callback: object) -> None:
        self.disconnect_callback = callback

    def set_status_callback(self, callback: object) -> None:
        self.status_callback = callback

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def subscribe(self, fence: object) -> None:
        return None

    def cancel(self, request_id: int) -> None:
        return None


def test_official_raw_bars_flow_through_recorder_into_reference_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clients: list[object] = []

    class EWrapper:
        pass

    class Contract:
        pass

    class EClient:
        def __init__(self, wrapper: object) -> None:
            self.wrapper = wrapper
            self.requests: list[int] = []
            clients.append(self)

        def connect(self, _host: str, _port: int, _client_id: int) -> bool:
            return True

        def run(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def reqRealTimeBars(self, request_id: int, *_args: object) -> None:  # noqa: N802
            self.requests.append(request_id)

        def cancelRealTimeBars(self, _request_id: int) -> None:  # noqa: N802
            return None

    package = types.ModuleType("ibapi")
    client_module = types.ModuleType("ibapi.client")
    contract_module = types.ModuleType("ibapi.contract")
    wrapper_module = types.ModuleType("ibapi.wrapper")
    client_module.EClient = EClient  # type: ignore[attr-defined]
    contract_module.Contract = Contract  # type: ignore[attr-defined]
    wrapper_module.EWrapper = EWrapper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ibapi", package)
    monkeypatch.setitem(sys.modules, "ibapi.client", client_module)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract_module)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper_module)

    database = tmp_path / "official-plugin.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    idea_path.write_text(json.dumps([_config().model_dump(mode="json")]), encoding="utf-8")
    ibkr_subscriptions = tuple(
        IBKRSubscription(index, index, symbol, "STK", "SMART", "USD", "bars")
        for index, symbol in enumerate(COHORT, start=1)
    )
    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4002,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
        subscriptions=ibkr_subscriptions,
    )
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="official-run",
            owner_id="owner",
            mode="prospective_record",
            host="127.0.0.1",
            port=4002,
            client_id=71,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
            idea_config=idea_path,
        ),
        bridge,
    )
    instruments = tuple(
        InstrumentSpec(symbol, index, "stock", symbol, "SMART", "USD")
        for index, symbol in enumerate(COHORT, start=1)
    )
    subscriptions = tuple(
        SubscriptionSpec(
            name=f"{symbol}-bars",
            instrument_id=symbol,
            feed_kind="bars",
            request_id=index,
            continuity_required=True,
            optional=False,
            stale_after_us=10_000_000,
        )
        for index, symbol in enumerate(COHORT, start=1)
    )
    recorder.start(now_us=1, instruments=instruments, subscriptions=subscriptions)
    client = clients[0]
    wrapper = client.wrapper  # type: ignore[attr-defined]
    open_seconds = int(datetime(2026, 8, 3, 13, 30, tzinfo=UTC).timestamp())
    close_seconds = int(datetime(2026, 8, 3, 13, 59, 55, tzinfo=UTC).timestamp())
    for index, _symbol in enumerate(COHORT, start=1):
        wrapper.realtimeBar(index, open_seconds, 100.0, 100.0, 100.0, 100.0, 1, 0, 1)
        wrapper.realtimeBar(
            index,
            close_seconds,
            100.0,
            102.0 + index / 100,
            100.0,
            101.0 + index / 100,
            1,
            0,
            1,
        )
    drain_at = time.time_ns() // 1_000 + 1_000_000
    assert recorder.drain(now_us=drain_at) == 40
    with connect_v2(database) as connection:
        outputs = connection.execute(
            "SELECT * FROM idea_outputs ORDER BY output_ordinal"
        ).fetchall()
        proposed = outputs[-1]
        input_ids = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT event_id FROM idea_output_inputs WHERE output_id=? ORDER BY input_ordinal",
                (proposed["output_id"],),
            )
        )
    recorder.stop(now_us=drain_at + 1)
    assert client.requests == list(range(1, 21))  # type: ignore[attr-defined]
    assert [row["output_kind"] for row in outputs] == [
        "observation",
        "signal",
        "proposed_trade",
    ]
    assert {row["subject_instrument_id"] for row in outputs} == {"WULF"}
    assert {row["as_of_at_us"] for row in outputs} == {
        int(datetime(2026, 8, 3, 14, 0, tzinfo=UTC).timestamp() * 1_000_000)
    }
    repository_record = IdeaOutputRecord(
        run_id=str(proposed["run_id"]),
        instance_id=str(proposed["instance_id"]),
        output_kind="proposed_trade",
        subject_instrument_id=str(proposed["subject_instrument_id"]),
        emitted_at_us=int(proposed["emitted_at_us"]),
        as_of_at_us=int(proposed["as_of_at_us"]),
        valid_until_at_us=None,
        direction=None,
        strength=None,
        confidence=None,
        horizon_us=None,
        first_input_event_id=input_ids[0],
        last_input_event_id=input_ids[-1],
        input_event_ids=input_ids,
        output_ordinal=int(proposed["output_ordinal"]),
        payload=cast(JsonValue, json.loads(str(proposed["payload_json"]))),
        data_class="prospective_protected",
        authority_status="unapproved",
        legs=(
            ProposedTradeLeg(
                instrument_id="WULF",
                action="buy",
                target="long",
                quantity_value=1.0,
                currency="USD",
            ),
        ),
    )
    assert deterministic_output_id(repository_record) == proposed["output_id"]
    repository = OperationalRepository(database)
    retry = repository.put_idea_output(repository_record)
    assert retry.output_id == proposed["output_id"]
    assert retry.inserted is False

    reordered_inputs = (input_ids[1], input_ids[0], *input_ids[2:])
    collisions = (
        replace(repository_record, payload={"changed": True}),
        replace(
            repository_record,
            first_input_event_id=reordered_inputs[0],
            input_event_ids=reordered_inputs,
        ),
        replace(
            repository_record,
            legs=(
                ProposedTradeLeg(
                    instrument_id="WULF",
                    action="buy",
                    target="long",
                    quantity_value=2.0,
                    currency="USD",
                ),
            ),
        ),
        replace(repository_record, subject_instrument_id="AAL"),
    )
    for collision in collisions:
        with pytest.raises(IdentityCollisionError):
            repository.put_idea_output(collision)


def test_recorder_is_default_off_without_idea_configuration(tmp_path: Path) -> None:
    database = tmp_path / "recorder-default-off.sqlite3"
    initialize_database(database)
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-default-off",
            owner_id="owner",
            mode="prospective_record",
            host="127.0.0.1",
            port=4002,
            client_id=1,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
        ),
        _RecorderAdapter(),
    )
    recorder.start(
        now_us=100,
        instruments=(InstrumentSpec("AAL", 1, "stock", "AAL", "SMART", "USD"),),
        subscriptions=(
            SubscriptionSpec(
                name="aal-bars",
                instrument_id="AAL",
                feed_kind="bars",
                request_id=1,
                continuity_required=True,
                optional=False,
                stale_after_us=300_000_000,
            ),
        ),
    )
    with connect_v2(database) as connection:
        instance_count = connection.execute("SELECT count(*) FROM idea_instances").fetchone()[0]
    assert recorder.idea_requirements == ()
    assert instance_count == 0
    recorder.stop(now_us=101)


def test_recorder_explicit_config_wires_generic_requirements_and_activation(tmp_path: Path) -> None:
    database = tmp_path / "recorder.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    config_value = _config().model_dump(mode="json")
    idea_path.write_text(json.dumps([config_value]), encoding="utf-8")
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-wired",
            owner_id="owner",
            mode="prospective_record",
            host="127.0.0.1",
            port=4002,
            client_id=1,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
            idea_config=idea_path,
        ),
        _RecorderAdapter(),
    )
    instruments = tuple(
        InstrumentSpec(symbol, index + 1, "stock", symbol, "SMART", "USD")
        for index, symbol in enumerate(COHORT)
    )
    subscriptions = tuple(
        SubscriptionSpec(
            name=f"{symbol}-bars",
            instrument_id=symbol,
            feed_kind="bars",
            request_id=index + 1,
            continuity_required=True,
            optional=False,
            stale_after_us=300_000_000,
        )
        for index, symbol in enumerate(COHORT)
    )
    recorder.start(now_us=100, instruments=instruments, subscriptions=subscriptions)
    with connect_v2(database) as connection:
        instances = connection.execute("SELECT count(*) FROM idea_instances").fetchone()[0]
    assert len(recorder.idea_requirements) == len(COHORT)
    assert instances == 1
    recorder.stop(now_us=101)
