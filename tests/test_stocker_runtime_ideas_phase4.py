from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from stocker_ideas.plugins.opening_leader_continuation_v0 import MANIFEST
from stocker_runtime.domain import (
    Observation,
    ProposedPosition,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)
from stocker_runtime.ideas import IdeaBatch, IdeaEvaluation
from stocker_runtime.ideas.discovery import (
    IdeaConfig,
    IdeaDiscoveryError,
    aggregate_requirements,
    discover_plugins,
    load_idea_configs,
)
from stocker_runtime.ideas.runner import IdeaRunner
from stocker_runtime.storage.connection import connect_v2, initialize_database

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
        "expected_code_hash": hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),
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


def _event(connection: object, sequence: int, symbol: str, close: float) -> None:
    payload = {
        "session": "2026-08-03",
        "checkpoint": 6,
        "regular_session_open": 100.0,
        "checkpoint_close": close,
        "source_completeness": "complete",
        "duplicate_resolution": "unique",
    }
    payload_json = canonical_json_bytes(payload).decode()
    connection.execute(  # type: ignore[attr-defined]
        "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, recorder_generation, "
        "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
        "VALUES (?, ?, 'run-1', 1, 1, 'bar', ?, ?, 'pending')",
        (
            sequence,
            f"callback-{sequence}",
            1100 + sequence,
            hashlib.sha256(payload_json.encode()).hexdigest(),
        ),
    )
    connection.execute(  # type: ignore[attr-defined]
        "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, feed_kind, "
        "event_kind, event_at_us, received_at_us, connection_generation, payload_json, "
        "payload_sha256) "
        "VALUES (?, 'run-1', ?, ?, 'bars_5m', 'bar', ?, ?, 1, ?, ?)",
        (
            f"event-{sequence}",
            sequence,
            symbol,
            1000 + sequence,
            1100 + sequence,
            payload_json,
            hashlib.sha256(payload_json.encode()).hexdigest(),
        ),
    )


def test_empty_config_discovers_and_runs_nothing() -> None:
    assert discover_plugins(()) == ()
    example = load_idea_configs(Path("configs/ideas/v2.example.json"))
    assert len(example) == 1
    assert discover_plugins(example) == ()


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


def test_discovery_rejects_unreviewed_hash_and_aggregates_requirements() -> None:
    with pytest.raises(IdeaDiscoveryError, match="code hash"):
        discover_plugins((_config(expected_code_hash="0" * 64),))
    plugin = discover_plugins((_config(),))[0]
    aggregated = aggregate_requirements((plugin,))
    assert len(aggregated) == 20
    assert aggregated == tuple(sorted(aggregated, key=lambda item: item.to_canonical_json()))


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
    runner = IdeaRunner(database, discovered)
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
        IdeaRunner(shadow_db, (incompatible,))
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
    runner = IdeaRunner(database, (plugin,))
    activation = runner.activate(run_id="run-1", plugin=plugin, activated_at_us=100)
    with connect_v2(database) as connection:
        for index, symbol in enumerate(COHORT, start=2):
            _event(connection, index, symbol, 104.0 if symbol == "AAOI" else 100.0)

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
    assert checkpoint == "event-21"
    # Pre-activation AAL was not backfilled, so only 19 bars are considered; minimum is 15.
    assert [row["output_kind"] for row in outputs] == ["observation", "signal", "proposed_trade"]
    assert outputs[-1]["authority_status"] == "unapproved"
    assert json.loads(outputs[-1]["payload_json"])["selected_identity"] == "rank_1"


def test_one_plugin_failure_does_not_stop_other_instance(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed(database)
    good = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (good,))
    failed = runner.activate(run_id="run-1", plugin=good, activated_at_us=100)
    healthy = runner.activate(run_id="run-1", plugin=good, activated_at_us=101)
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
    from stocker_runtime.domain import MarketEvent
    from stocker_runtime.ideas import IdeaBatch

    events = tuple(
        MarketEvent(
            event_id=f"e-{index}",
            instrument_id=symbol,
            feed_kind="bars_5m",
            event_kind="bar",
            event_at_us=1000 + index,
            received_at_us=1100 + index,
            payload={
                "session": "2026-08-03",
                "checkpoint": 6,
                "regular_session_open": 100.0,
                "checkpoint_close": 102.0 if symbol in {"AAL", "AAOI"} else 101.0,
                "source_completeness": "complete",
                "duplicate_resolution": "unique",
            },
        )
        for index, symbol in enumerate(COHORT)
    )
    batch = IdeaBatch(
        mode=RuntimeMode.PROSPECTIVE_RECORD,
        events=events,
        input_watermark="e-19",
        causal_from_at_us=1000,
        causal_through_at_us=1019,
    )
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
    runner = IdeaRunner(database, (discovered,))
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
            "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
            "requirements_hash, opened_at_us) VALUES "
            "('sub-aal', 'run-1', 1, 1, 'AAL', 'bars_5m', 1, 'active', ?, 1)",
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
    runner = IdeaRunner(database, (discovered,))
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


class _InvalidPlugin:
    def __init__(self, original: object, failure: str) -> None:
        self._original = original
        self._failure = failure

    @property
    def manifest(self) -> object:
        return self._original.manifest  # type: ignore[attr-defined,no-any-return]

    def requirements(self, activation: object) -> object:
        return self._original.requirements(activation)  # type: ignore[attr-defined,no-any-return]

    def evaluate(self, batch: IdeaBatch, state: object) -> IdeaEvaluation:
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
    runner = IdeaRunner(database, (invalid,))
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
