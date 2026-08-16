from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import threading
import time
import types
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import pytest

import stocker_ideas.plugins.frozen_m1c_signal_v0 as frozen_m1c_plugin
import stocker_ideas.plugins.m1c_opening_reversal_v1_1 as opening_reversal_plugin
import stocker_ideas.plugins.m1c_quiet_state_options_v0 as quiet_plugin
import stocker_runtime.ingestion.recorder as recorder_module
from stocker_ideas.plugins.opening_leader_continuation_v0 import MANIFEST
from stocker_runtime.domain import (
    JsonValue,
    MarketEvent,
    Observation,
    ProposedPosition,
    ProposedTrade,
    ProposedTradeLeg,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)
from stocker_runtime.ideas import (
    AncestorPageContinuation,
    ExactEventsContinuation,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    IdeaPlugin,
    MarketDataInterest,
    MarketDataRequirement,
    deterministic_idea_output_id,
)
from stocker_runtime.ideas.discovery import (
    DiscoveredPlugin,
    IdeaConfig,
    IdeaDiscoveryError,
    IdeaInstrumentConfig,
    aggregate_instruments,
    aggregate_requirements,
    discover_plugins,
    load_idea_configs,
    reviewed_code_hash,
)
from stocker_runtime.ideas.runner import (
    EvaluationResult,
    IdeaRunner,
    IdeaRunnerError,
    _merge_batch_requirements,
)
from stocker_runtime.ingestion import (
    CallbackFence,
    ContractCandidate,
    DuplicateWriterError,
    IBKRMarketData,
    IBKRSubscription,
    InstrumentSpec,
    MarketDataCallback,
    MarketDataPlan,
    MarketDataStatus,
    OptionParameterSet,
    Recorder,
    RecorderConfig,
    RecorderState,
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
        "enabled": True,
    }
    values.update(changes)
    return IdeaConfig.model_validate(values)


def _configured_instruments(
    symbols: tuple[str, ...] = COHORT,
) -> tuple[IdeaInstrumentConfig, ...]:
    return tuple(
        IdeaInstrumentConfig(
            instrument_id=symbol,
            ibkr_con_id=index,
            kind="stock",
            symbol=symbol,
            exchange="SMART",
            currency="USD",
        )
        for index, symbol in enumerate(symbols, start=1)
    )


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
    if event.event_kind == "bar_5m":
        connection.execute(  # type: ignore[attr-defined]
            "INSERT INTO market_events(event_id, run_id, source_sequence, "
            "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
            "event_at_us, received_at_us, connection_generation, payload_json, payload_sha256) "
            "VALUES (?, ?, NULL, ?, ?, 'bars', 'bar_5m', ?, ?, 1, ?, ?)",
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
        return
    connection.execute(  # type: ignore[attr-defined]
        "INSERT INTO callback_inbox(source_sequence, event_uid, run_id, recorder_generation, "
        "connection_generation, callback_kind, received_at_us, payload_sha256, lifecycle) "
        "VALUES (?, ?, ?, 1, 1, ?, ?, ?, 'pending')",
        (
            sequence,
            event.event_id,
            run_id,
            event.event_kind,
            event.received_at_us,
            hashlib.sha256(payload_json.encode()).hexdigest(),
        ),
    )
    connection.execute(  # type: ignore[attr-defined]
        "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, feed_kind, "
        "event_kind, event_at_us, received_at_us, connection_generation, payload_json, "
        "payload_sha256) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (
            event.event_id,
            run_id,
            sequence,
            event.instrument_id,
            event.feed_kind,
            event.event_kind,
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
    assert example[0].instruments == ()
    assert discover_plugins(example) == ()


def test_idea_configuration_requires_explicit_enablement() -> None:
    values = _config().model_dump(mode="python")
    values.pop("enabled")
    omitted = IdeaConfig.model_validate(values)
    assert omitted.enabled is False
    assert discover_plugins((omitted,)) == ()


def test_incremental_lineage_contract_is_bounded() -> None:
    with pytest.raises(ValueError, match="at most 256 items"):
        IdeaEvaluation(
            state={},
            outputs=(),
            retained_input_event_ids=tuple(f"event-{index}" for index in range(257)),
            interests=(),
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


def test_instrument_metadata_aggregation_is_exact_and_deterministic() -> None:
    base = discover_plugins((_config(),))[0]
    first = replace(base, config=_config(instruments=_configured_instruments(("AAL",))))
    identical = replace(base, config=_config(instruments=_configured_instruments(("AAL",))))
    assert aggregate_instruments((first, identical)) == _configured_instruments(("AAL",))
    conflict = replace(
        base,
        config=_config(
            instruments=(
                IdeaInstrumentConfig(
                    instrument_id="AAL",
                    ibkr_con_id=999,
                    kind="stock",
                    symbol="AAL",
                    exchange="SMART",
                    currency="USD",
                ),
            )
        ),
    )
    with pytest.raises(IdeaDiscoveryError, match="conflicting instrument metadata"):
        aggregate_instruments((first, conflict))


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


def test_duplicate_requirements_merge_blocking_flags_conservatively() -> None:
    plugin = discover_plugins((_config(),))[0]
    passive = MarketDataRequirement(
        feed_kind="bars",
        event_kind="bar_5m",
        instrument_id="AAL",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    strict = MarketDataRequirement(
        feed_kind="bars",
        event_kind="bar_5m",
        instrument_id="AAL",
        cadence="5s",
        gaps_block=True,
        staleness_block=True,
    )
    merged = aggregate_requirements(
        (replace(plugin, requirements=(passive,)), replace(plugin, requirements=(strict,)))
    )
    assert merged == (strict,)


def test_dynamic_batch_requirements_merge_cadence_and_blocking_conservatively() -> None:
    snapshot = MarketDataRequirement(
        feed_kind="quotes",
        event_kind="quote",
        instrument_id="ibkr-option-9001",
        cadence="snapshot",
        gaps_block=False,
        staleness_block=False,
    )
    stream = MarketDataRequirement(
        feed_kind="quotes",
        event_kind="quote",
        instrument_id="ibkr-option-9001",
        cadence="stream",
        gaps_block=True,
        staleness_block=True,
    )

    assert _merge_batch_requirements((snapshot, stream)) == (stream,)


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


class _EarlyLineageProposalPlugin:
    def __init__(self, original: IdeaPlugin) -> None:
        self._manifest = original.manifest

    @property
    def manifest(self) -> IdeaManifest:
        return self._manifest

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        del activation
        return ()

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        del state
        first = batch.events[0]
        return IdeaEvaluation(
            state={},
            outputs=(
                ProposedTrade(
                    subject_instrument_id=first.instrument_id,
                    as_of_at_us=first.event_at_us,
                    payload={"reason": "early-lineage-fixture"},
                    legs=(
                        ProposedTradeLeg(
                            instrument_id=first.instrument_id,
                            action="buy",
                            target="long",
                            quantity_value=1,
                            currency="USD",
                        ),
                    ),
                ),
            ),
            output_input_event_ids=((first.event_id,),),
            interests=(),
        )


class _InterestPlugin:
    def __init__(
        self,
        original: IdeaPlugin,
        *,
        lifetime_us: int = 3_600_000_000,
        cadence: Literal["snapshot", "stream"] = "snapshot",
    ) -> None:
        self._manifest = IdeaManifest.model_validate(
            {**original.manifest.model_dump(mode="python"), "maximum_interests_per_batch": 1}
        )
        self._lifetime_us = lifetime_us
        self._cadence = cadence

    @property
    def manifest(self) -> IdeaManifest:
        return self._manifest

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        del activation
        return ()

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        del state
        first = batch.events[0]
        return IdeaEvaluation(
            state={},
            outputs=(),
            output_input_event_ids=(),
            interests=(
                MarketDataInterest(
                    interest_key="primary-1dte-call",
                    underlying_instrument_id=first.instrument_id,
                    minimum_days_to_expiry=1,
                    maximum_days_to_expiry=1,
                    option_right="call",
                    strike_offset=0,
                    reference_price=100.0,
                    cadence=self._cadence,
                    as_of_at_us=first.event_at_us,
                    expires_at_us=first.event_at_us + self._lifetime_us,
                    required=True,
                    priority=100,
                    maximum_contracts=1,
                    input_event_id=first.event_id,
                ),
            ),
        )


def _bounded_interest(interest_key: str, *, option_right: str) -> MarketDataInterest:
    as_of_at_us = int(datetime(2026, 8, 3, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    return MarketDataInterest.model_validate(
        {
            "interest_key": interest_key,
            "underlying_instrument_id": "AAL",
            "minimum_days_to_expiry": 1,
            "maximum_days_to_expiry": 1,
            "option_right": option_right,
            "strike_offset": 0,
            "reference_price": 100.0,
            "cadence": "snapshot",
            "as_of_at_us": as_of_at_us,
            "expires_at_us": as_of_at_us + 3_600_000_000,
            "required": True,
            "priority": 100,
            "input_event_id": "event-1",
        }
    )


def _insert_resolved_interest_fixture(
    connection: sqlite3.Connection,
    activation: IdeaActivation,
    *,
    ordinal: int,
    completed_at_us: int,
    lifecycle: str,
) -> tuple[str, str]:
    interest_key = f"receipt-priority-{ordinal}"
    IdeaRunner._insert_interest(
        connection,
        activation,
        _bounded_interest(interest_key, option_right="call"),
        1_000 + ordinal,
    )
    interest_id = str(
        connection.execute(
            "SELECT interest_id FROM market_data_interests WHERE instance_id=? AND interest_key=?",
            (activation.instance_id, interest_key),
        ).fetchone()[0]
    )
    instrument_id = f"option-priority-{ordinal}"
    connection.execute(
        "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
        "exchange, currency, option_expiry, option_strike, option_right, option_multiplier) "
        "VALUES (?, ?, ?, 'option', 'AAL', 'SMART', 'USD', '20260804', '100', 'call', '100')",
        (instrument_id, hashlib.sha256(instrument_id.encode()).hexdigest(), 20_000 + ordinal),
    )
    connection.execute(
        "INSERT INTO instrument_discovery_receipts(receipt_id, interest_id, run_id, "
        "instance_id, status, instrument_id, expiry, strike, option_right, multiplier, "
        "candidates_inspected, completed_at_us) VALUES (?, ?, 'run-1', ?, 'resolved', ?, "
        "'20260804', 100, 'call', '100', 1, ?)",
        (
            f"receipt-priority-{ordinal}",
            interest_id,
            activation.instance_id,
            instrument_id,
            completed_at_us,
        ),
    )
    connection.execute(
        "UPDATE market_data_interests SET lifecycle='resolved', updated_at_us=? "
        "WHERE interest_id=?",
        (completed_at_us, interest_id),
    )
    if lifecycle != "resolved":
        connection.execute(
            "UPDATE market_data_interests SET lifecycle=?, updated_at_us=? WHERE interest_id=?",
            (lifecycle, completed_at_us + 1, interest_id),
        )
    return interest_id, instrument_id


def test_runner_commits_causal_interest_and_loads_resolved_dynamic_events(tmp_path: Path) -> None:
    database = tmp_path / "runner-interest.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(
        original,
        plugin=_InterestPlugin(original.plugin),
        manifest=_InterestPlugin(original.plugin).manifest,
        manifest_json=_InterestPlugin(original.plugin).manifest.to_canonical_json().decode(),
        manifest_hash=hashlib.sha256(
            _InterestPlugin(original.plugin).manifest.to_canonical_json()
        ).hexdigest(),
        requirements=(requirement,),
    )
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activated = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    activation = discovered.activation(
        instance_id=activated.instance_id,
        run_id="run-1",
        data_class=ProtectedDataClass.PROSPECTIVE,
        activated_at_us=100,
    )
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 100.0)
    result = runner.run_once(now_us=1_000)[0]
    assert result.advanced is True
    with connect_v2(database) as connection:
        interest = connection.execute(
            "SELECT * FROM market_data_interests WHERE instance_id=?", (activation.instance_id,)
        ).fetchone()
        assert interest is not None
        assert interest["input_event_id"] == "event-1"
        assert interest["lifecycle"] == "pending"
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency, option_expiry, option_strike, option_right, option_multiplier) "
            "VALUES ('ibkr-option-9001', ?, 9001, 'option', 'AAL', 'SMART', 'USD', "
            "'20260810', '100', 'call', '100')",
            ("9" * 64,),
        )
        connection.execute(
            "INSERT INTO instrument_discovery_receipts(receipt_id, interest_id, run_id, "
            "instance_id, status, instrument_id, expiry, strike, option_right, multiplier, "
            "candidates_inspected, completed_at_us) VALUES ('receipt-1', ?, 'run-1', ?, "
            "'resolved', 'ibkr-option-9001', '20260810', 100, 'call', '100', 1, ?)",
            (
                interest["interest_id"],
                activation.instance_id,
                int(interest["as_of_at_us"]) + 1,
            ),
        )
        connection.execute(
            "UPDATE market_data_interests SET lifecycle='resolved', updated_at_us=1001 "
            "WHERE interest_id=?",
            (interest["interest_id"],),
        )
        _persist_market_event(
            connection,
            2,
            MarketEvent(
                event_id="option-event-2",
                instrument_id="ibkr-option-9001",
                feed_kind="quotes",
                event_kind="quote",
                event_at_us=int(interest["as_of_at_us"]) + 2,
                received_at_us=int(interest["as_of_at_us"]) + 2,
                payload={"bid": 1.0, "ask": 1.1},
            ),
        )
        _persist_market_event(
            connection,
            3,
            MarketEvent(
                event_id="option-event-after-expiry",
                instrument_id="ibkr-option-9001",
                feed_kind="quotes",
                event_kind="quote",
                event_at_us=int(interest["expires_at_us"]),
                received_at_us=int(interest["expires_at_us"]),
                payload={"bid": 1.2, "ask": 1.3},
            ),
        )
        connection.execute(
            "UPDATE market_data_interests SET lifecycle='expired', "
            "reason_code='INTEREST_EXPIRED', updated_at_us=1002 WHERE interest_id=?",
            (interest["interest_id"],),
        )
    loaded = runner._load_batch(activation.instance_id)
    runner.close()
    assert loaded is not None
    assert tuple(event.event_id for event in loaded[1].events) == ("option-event-2",)
    assert loaded[1].discovery_receipts[0].instrument_id == "ibkr-option-9001"


def test_runner_prioritizes_active_receipts_over_newer_terminal_history(tmp_path: Path) -> None:
    database = tmp_path / "runner-active-receipt-priority.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(original, requirements=(requirement,))
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activated = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    activation = discovered.activation(
        instance_id=activated.instance_id,
        run_id="run-1",
        data_class=ProtectedDataClass.PROSPECTIVE,
        activated_at_us=100,
    )
    base_time = int(datetime(2026, 8, 3, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 100.0)
        _active_interest_id, active_instrument_id = _insert_resolved_interest_fixture(
            connection,
            activation,
            ordinal=0,
            completed_at_us=base_time + 1,
            lifecycle="resolved",
        )
        for ordinal in range(1, 65):
            _insert_resolved_interest_fixture(
                connection,
                activation,
                ordinal=ordinal,
                completed_at_us=base_time + 100 + ordinal,
                lifecycle="fulfilled",
            )
        _persist_market_event(
            connection,
            2,
            MarketEvent(
                event_id="active-option-event",
                instrument_id=active_instrument_id,
                feed_kind="quotes",
                event_kind="quote",
                event_at_us=base_time + 500,
                received_at_us=base_time + 501,
                payload={"bid": 1.0, "ask": 1.1},
            ),
        )

    loaded = runner._load_batch(activation.instance_id)
    runner.close()

    assert loaded is not None
    assert any(
        receipt.instrument_id == active_instrument_id for receipt in loaded[1].discovery_receipts
    )
    assert any(event.event_id == "active-option-event" for event in loaded[1].events)


def test_runner_excludes_discovery_receipts_completed_after_batch_causal_time(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runner-receipt-causality.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(original, requirements=(requirement,))
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activated = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    activation = discovered.activation(
        instance_id=activated.instance_id,
        run_id="run-1",
        data_class=ProtectedDataClass.PROSPECTIVE,
        activated_at_us=100,
    )
    base_time = int(datetime(2026, 8, 3, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 100.0)
        _interest_id, future_instrument_id = _insert_resolved_interest_fixture(
            connection,
            activation,
            ordinal=0,
            completed_at_us=base_time + 1_000,
            lifecycle="resolved",
        )
        _persist_market_event(
            connection,
            2,
            MarketEvent(
                event_id="pre-receipt-option-event",
                instrument_id=future_instrument_id,
                feed_kind="quotes",
                event_kind="quote",
                event_at_us=base_time + 500,
                received_at_us=base_time + 501,
                payload={"bid": 1.0, "ask": 1.1},
            ),
        )

    loaded = runner._load_batch(activation.instance_id)
    runner.close()

    assert loaded is not None
    assert loaded[1].causal_through_at_us < base_time + 1_000
    assert loaded[1].discovery_receipts == ()
    assert all(event.event_id != "pre-receipt-option-event" for event in loaded[1].events)


def test_runner_caps_nonterminal_interests_per_instance(tmp_path: Path) -> None:
    database = tmp_path / "runner-interest-cap.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activated = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    activation = discovered.activation(
        instance_id=activated.instance_id,
        run_id="run-1",
        data_class=ProtectedDataClass.PROSPECTIVE,
        activated_at_us=100,
    )
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 100.0)
        connection.execute("BEGIN IMMEDIATE")
        for index in range(64):
            IdeaRunner._insert_interest(
                connection,
                activation,
                _bounded_interest(f"bounded-{index}", option_right="call"),
                1_000,
            )
        with pytest.raises(IdeaRunnerError, match="nonterminal interest cap"):
            IdeaRunner._insert_interest(
                connection,
                activation,
                _bounded_interest("bounded-overflow", option_right="put"),
                1_000,
            )
        connection.rollback()


def test_runner_commit_boundary_covers_batch_watermark_and_overlapping_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "runner-commit-boundary.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(
        original,
        plugin=_EarlyLineageProposalPlugin(original.plugin),
        requirements=(requirement,),
    )
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        _event(connection, 1, "AAL", 100.0)
        _event(connection, 2, "AAL", 101.0)
    original_commit = runner._commit_evaluation

    def commit_with_overlapping_callback(
        activation: IdeaActivation,
        batch: IdeaBatch,
        evaluation: IdeaEvaluation,
        event_ids: tuple[str, ...],
        starting_checkpoint: str | None,
        starting_state_hash: str,
        now_us: int,
    ) -> None:
        with connect_v2(database) as connection:
            _event(connection, 3, "AAL", 102.0)
        original_commit(
            activation,
            batch,
            evaluation,
            event_ids,
            starting_checkpoint,
            starting_state_hash,
            now_us,
        )

    monkeypatch.setattr(runner, "_commit_evaluation", commit_with_overlapping_callback)

    result = runner.run_once(now_us=1_000)[0]
    with connect_v2(database) as connection:
        output = connection.execute(
            "SELECT output_id, first_input_event_id, last_input_event_id FROM idea_outputs"
        ).fetchone()
        checkpoint = connection.execute(
            "SELECT last_market_event_id, last_source_sequence FROM idea_checkpoints "
            "WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()
        boundary = connection.execute(
            "SELECT committed_after_source_sequence FROM idea_output_commit_boundaries "
            "WHERE output_id=?",
            (output["output_id"],),
        ).fetchone()[0]

    assert result.advanced is True
    assert result.output_count == 1
    assert tuple(output)[1:] == ("event-1", "event-1")
    assert tuple(checkpoint) == ("event-2", 2)
    assert boundary == 3


def test_runner_sealed_retry_survives_real_lineage_compaction(tmp_path: Path) -> None:
    database = tmp_path / "runner-compacted-retry.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(
        original,
        plugin=_EarlyLineageProposalPlugin(original.plugin),
        requirements=(requirement,),
    )
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    try:
        activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
        with connect_v2(database) as connection:
            _event(connection, 1, "AAL", 100.0)
            _event(connection, 2, "AAL", 101.0)
        loaded = runner._load_batch(activation.instance_id)
        assert loaded is not None
        loaded_activation, batch, state, _, _, _ = loaded
        evaluation = discovered.plugin.evaluate(batch, state)
        assert runner.run_once(now_us=1_000)[0].output_count == 1
        output_id = deterministic_idea_output_id(
            instance_id=activation.instance_id,
            input_event_ids=("event-1",),
            output_kind=evaluation.outputs[0].kind,
            output_ordinal=0,
            as_of_at_us=evaluation.outputs[0].as_of_at_us,
            payload=evaluation.outputs[0].payload,
            legs=cast(ProposedTrade, evaluation.outputs[0]).legs,
        )

        RetentionManager(
            database,
            RetentionPolicy(raw_market_event_us=1, idea_shadow_us=10**18),
        ).run(
            now_us=batch.causal_through_at_us + 100,
            measured_database_bytes=1,
            measured_wal_bytes=0,
        )
        with connect_v2(database) as connection:
            assert (
                connection.execute(
                    "SELECT 1 FROM market_events WHERE event_id='event-1'"
                ).fetchone()
                is None
            )
            connection.execute("BEGIN IMMEDIATE")
            runner._insert_output(
                connection,
                loaded_activation,
                batch,
                evaluation.outputs[0],
                0,
                ("event-1",),
                hashlib.sha256(canonical_json_bytes(cast(JsonValue, ("event-1",)))).hexdigest(),
                1_000,
            )
            connection.commit()
            assert (
                connection.execute(
                    "SELECT count(*) FROM idea_outputs WHERE output_id=?", (output_id,)
                ).fetchone()[0]
                == 1
            )
            assert (
                connection.execute(
                    "SELECT count(*) FROM idea_output_seals WHERE output_id=?", (output_id,)
                ).fetchone()[0]
                == 1
            )
    finally:
        runner.close()


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
        _persist_market_event(connection, 1, _derived_progress_event(1))

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
    events: list[MarketEvent] = []
    for event in _checkpoint_evidence(6):
        payload = dict(event.payload)
        if payload["bar_number"] == 6:
            payload["close"] = 102.0 if event.instrument_id in {"AAL", "AAOI"} else 101.0
        events.append(
            MarketEvent(
                **{
                    **event.model_dump(mode="python"),
                    "payload": payload,
                }
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
        _persist_market_event(connection, 1, _derived_progress_event(1))

    # Gap evidence is now embedded permanently in each derived receipt; the generic
    # runner need not stop unrelated receipt processing on an open feed gap.
    assert runner.run_once(now_us=1_000)[0].advanced is True
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_market_event_id FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()[0]
    assert checkpoint == "derived-AAL-1-1"


def test_runner_batches_at_256_without_skipping_the_remainder(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite3"
    _seed(database)
    discovered = discover_plugins((_config(),))[0]
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        for sequence in range(1, 258):
            event = _derived_progress_event(sequence, COHORT[sequence % len(COHORT)])
            event = MarketEvent(
                **{**event.model_dump(mode="python"), "event_id": f"batch-{sequence}"}
            )
            _persist_market_event(connection, sequence, event)

    first = runner.run_once(now_us=2_000)[0]
    second = runner.run_once(now_us=3_000)[0]
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_market_event_id FROM idea_checkpoints"
        ).fetchone()[0]
    assert first.advanced is True
    assert second.advanced is True
    assert checkpoint == "batch-257"


class _CursorRecordingPlugin:
    def __init__(self, original: IdeaPlugin) -> None:
        self._manifest = original.manifest

    @property
    def manifest(self) -> IdeaManifest:
        return self._manifest

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        del activation
        return ()

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        prior = state if isinstance(state, Mapping) else {}
        seen = prior.get("seen", ())
        assert isinstance(seen, tuple | list)
        return IdeaEvaluation(
            state={"seen": (*seen, *(event.event_id for event in batch.events))},
            outputs=(),
            interests=(),
        )


class _ContinuationProbePlugin:
    def __init__(self, original: IdeaPlugin) -> None:
        self._manifest = original.manifest

    @property
    def manifest(self) -> IdeaManifest:
        return self._manifest

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        del activation
        return ()

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        prior = state if isinstance(state, Mapping) else {}
        retained = batch.prior_state_input_event_ids
        if batch.continuation_request is None:
            root = batch.events[-1].event_id
            return IdeaEvaluation(
                state={"seen": ()},
                outputs=(),
                retained_input_event_ids=(root,),
                interests=(),
                continuation=AncestorPageContinuation(
                    root_event_ids=(root,),
                    event_kind="bar_5m_session_prefix",
                    input_roles=("prior_receipt",),
                ),
            )
        if isinstance(batch.continuation_request, AncestorPageContinuation):
            seen_value = prior.get("seen", ())
            assert isinstance(seen_value, tuple | list)
            seen = (*seen_value, *(event.event_id for event in batch.rehydrated_events))
            continuation = (
                AncestorPageContinuation(
                    **{
                        **batch.continuation_request.model_dump(mode="python"),
                        "cursor": batch.continuation_token,
                    }
                )
                if batch.continuation_token is not None
                else ExactEventsContinuation(
                    root_event_ids=batch.continuation_request.root_event_ids,
                    event_ids=(str(seen[0]),),
                    event_kind="bar_5m_session_prefix",
                    input_roles=("prior_receipt",),
                )
            )
            return IdeaEvaluation(
                state={"seen": seen},
                outputs=(),
                retained_input_event_ids=retained,
                interests=(),
                continuation=continuation,
            )
        event = batch.rehydrated_events[0]
        return IdeaEvaluation(
            state={"seen": prior.get("seen", ()), "completed": event.event_id},
            outputs=(
                Observation(
                    subject_instrument_id=event.instrument_id,
                    as_of_at_us=event.event_at_us,
                    payload={"status": "continuation_complete"},
                ),
            ),
            retained_input_event_ids=retained,
            output_input_event_ids=((event.event_id,),),
            interests=(),
        )


class _PrefixSelectingPlugin:
    def __init__(self, original: IdeaPlugin, maximum: int, *, clamp: bool = True) -> None:
        self._manifest = original.manifest
        self._maximum = maximum
        self._clamp = clamp

    @property
    def manifest(self) -> IdeaManifest:
        return self._manifest

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        del activation
        return ()

    def select_input_prefix(self, batch: IdeaBatch, state: JsonValue) -> int:
        del state
        return min(self._maximum, len(batch.events)) if self._clamp else self._maximum

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        prior = state if isinstance(state, Mapping) else {}
        seen = prior.get("seen", ())
        assert isinstance(seen, tuple | list)
        return IdeaEvaluation(
            state={"seen": (*seen, *(event.event_id for event in batch.events))},
            outputs=(),
            interests=(),
        )


def _insert_prefix_chain(
    connection: sqlite3.Connection,
    *,
    count: int,
    prefix: str = "continuation",
    run_id: str = "run-1",
) -> tuple[str, ...]:
    event_ids: list[str] = []
    prior_id: str | None = None
    for sequence in range(1, count + 1):
        event = MarketEvent(
            event_id=f"{prefix}-{sequence:03d}",
            instrument_id="AAL",
            feed_kind="bars",
            event_kind="bar_5m_session_prefix",
            event_at_us=sequence * 1_000_000,
            received_at_us=sequence * 1_000_000,
            payload={"session": "2026-08-10", "bar_number": sequence},
        )
        payload_json = canonical_json_bytes(event.payload).decode()
        connection.execute(
            "INSERT INTO market_events(event_id, run_id, source_sequence, "
            "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
            "event_at_us, received_at_us, connection_generation, payload_json, "
            "payload_sha256) VALUES (?, ?, NULL, ?, 'AAL', 'bars', "
            "'bar_5m_session_prefix', ?, ?, 1, ?, ?)",
            (
                event.event_id,
                run_id,
                sequence,
                event.event_at_us,
                event.received_at_us,
                payload_json,
                hashlib.sha256(payload_json.encode()).hexdigest(),
            ),
        )
        event_ids.append(event.event_id)
        if prior_id is not None:
            connection.execute(
                "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
                "input_ordinal, input_role, created_at_us) VALUES (?, ?, 0, "
                "'prior_receipt', ?)",
                (event.event_id, prior_id, event.received_at_us),
            )
        prior_id = event.event_id
    return tuple(event_ids)


def test_runner_keyset_cursor_does_not_skip_tied_sequence_after_256(
    tmp_path: Path,
) -> None:
    database = tmp_path / "tied-cursor.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind=None,
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(
        original,
        plugin=_CursorRecordingPlugin(original.plugin),
        requirements=(requirement,),
    )
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        for index in range(257):
            event = _derived_progress_event(1, "AAL")
            event = MarketEvent(
                **{
                    **event.model_dump(mode="python"),
                    "event_id": f"tied-{index:03d}",
                    "event_at_us": event.event_at_us + index,
                    "received_at_us": event.received_at_us + index,
                    "payload": {**event.payload, "first_source_sequence": 1},
                }
            )
            _persist_market_event(connection, 1, event)

    assert runner.run_once(now_us=2_000)[0].advanced is True
    runner.close()
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    assert runner.run_once(now_us=3_000)[0].advanced is True
    assert runner.run_once(now_us=4_000)[0].advanced is False
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_source_sequence, last_market_event_id, state_json FROM idea_checkpoints"
        ).fetchone()
    runner.close()
    assert tuple(checkpoint[:2]) == (1, "tied-256")
    assert json.loads(str(checkpoint["state_json"]))["plugin_state"]["seen"] == [
        f"tied-{index:03d}" for index in range(257)
    ]


def test_runner_continuation_pages_restart_and_finish_without_new_input(
    tmp_path: Path,
) -> None:
    database = tmp_path / "continuation.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar_5m_session_prefix",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(
        original,
        plugin=_ContinuationProbePlugin(original.plugin),
        requirements=(requirement,),
    )
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        event_ids = _insert_prefix_chain(connection, count=70)

    assert runner.run_once(now_us=1_000_000)[0] == EvaluationResult(activation.instance_id, True, 0)
    first_page = runner.run_once(now_us=2_000_000)[0]
    assert first_page.error_code is None, first_page.error_code
    assert first_page == EvaluationResult(activation.instance_id, True, 0)
    with connect_v2(database) as connection:
        paged_state = json.loads(
            str(
                connection.execute(
                    "SELECT state_json FROM idea_checkpoints WHERE instance_id=?",
                    (activation.instance_id,),
                ).fetchone()[0]
            )
        )
    assert len(paged_state["plugin_state"]["seen"]) == 64
    assert paged_state["continuation"]["cursor"].startswith("64:")
    runner.close()
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    assert runner.run_once(now_us=3_000_000)[0] == EvaluationResult(activation.instance_id, True, 0)
    completed = runner.run_once(now_us=4_000_000)[0]
    assert completed == EvaluationResult(activation.instance_id, True, 1)
    assert runner.run_once(now_us=5_000_000)[0].advanced is False
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_market_event_id, last_source_sequence, state_json, "
            "state_input_event_ids_json FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()
        output_inputs = tuple(
            row[0]
            for row in connection.execute(
                "SELECT input.event_id FROM idea_output_inputs input "
                "JOIN idea_outputs output USING(output_id) WHERE output.instance_id=?",
                (activation.instance_id,),
            )
        )
    runner.close()

    checkpoint_state = json.loads(str(checkpoint["state_json"]))
    assert tuple(checkpoint[:2]) == (event_ids[-1], 70)
    assert checkpoint_state["continuation"] is None
    assert checkpoint_state["plugin_state"]["completed"] == event_ids[0]
    assert json.loads(str(checkpoint["state_input_event_ids_json"])) == [event_ids[-1]]
    assert output_inputs == (event_ids[0],)


def test_runner_commits_only_selected_prefix_and_reselects_suffix(tmp_path: Path) -> None:
    database = tmp_path / "strict-prefix.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar_5m_session_prefix",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(
        original,
        plugin=_PrefixSelectingPlugin(original.plugin, 2),
        requirements=(requirement,),
    )
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        event_ids = _insert_prefix_chain(connection, count=3)

    assert runner.run_once(now_us=1_000_000)[0] == EvaluationResult(activation.instance_id, True, 0)
    with connect_v2(database) as connection:
        first = connection.execute(
            "SELECT last_market_event_id, last_source_sequence, state_json "
            "FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()
    assert tuple(first[:2]) == (event_ids[1], 2)
    assert json.loads(str(first["state_json"]))["plugin_state"]["seen"] == list(event_ids[:2])

    runner.close()
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    assert runner.run_once(now_us=2_000_000)[0] == EvaluationResult(activation.instance_id, True, 0)
    with connect_v2(database) as connection:
        second = connection.execute(
            "SELECT last_market_event_id, last_source_sequence, state_json "
            "FROM idea_checkpoints WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()
    runner.close()
    assert tuple(second[:2]) == (event_ids[2], 3)
    assert json.loads(str(second["state_json"]))["plugin_state"]["seen"] == list(event_ids)


@pytest.mark.parametrize("selection", (0, 2, True))
def test_runner_rejects_invalid_selected_prefix_without_advancing(
    tmp_path: Path,
    selection: int,
) -> None:
    database = tmp_path / "strict-prefix-invalid.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar_5m_session_prefix",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    discovered = replace(
        original,
        plugin=_PrefixSelectingPlugin(original.plugin, selection, clamp=False),
        requirements=(requirement,),
    )
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        _insert_prefix_chain(connection, count=1)

    result = runner.run_once(now_us=1_000_000)[0]
    with connect_v2(database) as connection:
        checkpoint = connection.execute(
            "SELECT last_market_event_id, last_source_sequence FROM idea_checkpoints "
            "WHERE instance_id=?",
            (activation.instance_id,),
        ).fetchone()
    runner.close()
    assert result.advanced is False
    assert result.error_code is not None and "INVALID ORDINARY INPUT PREFIX" in result.error_code
    assert tuple(checkpoint) == (None, None)


def test_runner_continuation_commit_rejects_stale_state_hash(tmp_path: Path) -> None:
    database = tmp_path / "continuation-cas.sqlite3"
    _seed(database)
    original = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar_5m_session_prefix",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    plugin = _ContinuationProbePlugin(original.plugin)
    discovered = replace(original, plugin=plugin, requirements=(requirement,))
    runner = IdeaRunner(database, (discovered,), run_id="run-1")
    activation = runner.activate(run_id="run-1", plugin=discovered, activated_at_us=100)
    with connect_v2(database) as connection:
        _insert_prefix_chain(connection, count=1)
    assert runner.run_once(now_us=1_000_000)[0].advanced is True

    first = runner._load_batch(activation.instance_id)
    second = runner._load_batch(activation.instance_id)
    assert first is not None and second is not None
    first_evaluation = plugin.evaluate(first[1], first[2])
    second_evaluation = plugin.evaluate(second[1], second[2])
    runner._validate_evaluation(first[0], plugin, first[1], first_evaluation)
    runner._validate_evaluation(second[0], plugin, second[1], second_evaluation)
    runner._commit_evaluation(
        first[0],
        first[1],
        first_evaluation,
        first[3],
        first[4],
        first[5],
        2_000_000,
    )
    with pytest.raises(IdeaRunnerError, match="checkpoint changed"):
        runner._commit_evaluation(
            second[0],
            second[1],
            second_evaluation,
            second[3],
            second[4],
            second[5],
            2_000_001,
        )
    runner.close()


def test_runner_continuation_rejects_unbound_roots_cursors_and_exact_events(
    tmp_path: Path,
) -> None:
    database = tmp_path / "continuation-boundary.sqlite3"
    _seed(database)
    requirement = MarketDataRequirement(
        instrument_id="AAL",
        feed_kind="bars",
        event_kind="bar_5m_session_prefix",
        cadence="5s",
        gaps_block=False,
        staleness_block=False,
    )
    with connect_v2(database) as connection:
        chain = _insert_prefix_chain(connection, count=3)
        unrelated = _insert_prefix_chain(connection, count=1, prefix="unrelated")[0]
        tied_payload = canonical_json_bytes({"session": "2026-08-10", "bar_number": 1}).decode()
        for event_id, event_at_us in (("z-tied-ancestor", 1_000_000), ("a-tied-root", 2_000_000)):
            connection.execute(
                "INSERT INTO market_events(event_id, run_id, source_sequence, "
                "derived_after_source_sequence, instrument_id, feed_kind, event_kind, "
                "event_at_us, received_at_us, connection_generation, payload_json, "
                "payload_sha256) VALUES (?, 'run-1', NULL, 1, 'AAL', 'bars', "
                "'bar_5m_session_prefix', ?, ?, 1, ?, ?)",
                (
                    event_id,
                    event_at_us,
                    event_at_us,
                    tied_payload,
                    hashlib.sha256(tied_payload.encode()).hexdigest(),
                ),
            )
        connection.execute(
            "INSERT INTO market_event_derivations(derived_event_id, input_event_id, "
            "input_ordinal, input_role, created_at_us) VALUES "
            "('a-tied-root', 'z-tied-ancestor', 0, 'prior_receipt', 2000000)"
        )
        valid = AncestorPageContinuation(
            root_event_ids=(chain[-1],),
            event_kind="bar_5m_session_prefix",
            input_roles=("prior_receipt",),
        )
        events, token = IdeaRunner._load_continuation_events(
            connection,
            run_id="run-1",
            last_source_sequence=3,
            last_market_event_id=chain[-1],
            retained_event_ids=(chain[-1],),
            requirements=(requirement,),
            request=valid,
        )
        assert tuple(item.event_id for item in events) == chain
        assert token is None

        tied = AncestorPageContinuation(
            root_event_ids=("a-tied-root",),
            event_kind="bar_5m_session_prefix",
            input_roles=("prior_receipt",),
        )
        tied_events, tied_token = IdeaRunner._load_continuation_events(
            connection,
            run_id="run-1",
            last_source_sequence=1,
            last_market_event_id="a-tied-root",
            retained_event_ids=("a-tied-root",),
            requirements=(requirement,),
            request=tied,
        )
        assert tuple(item.event_id for item in tied_events) == ("a-tied-root",)
        assert tied_token is None
        with pytest.raises(IdeaRunnerError, match="not an allowed causal ancestor"):
            IdeaRunner._load_continuation_events(
                connection,
                run_id="run-1",
                last_source_sequence=1,
                last_market_event_id="a-tied-root",
                retained_event_ids=("a-tied-root",),
                requirements=(requirement,),
                request=ExactEventsContinuation(
                    root_event_ids=("a-tied-root",),
                    event_ids=("z-tied-ancestor",),
                    event_kind="bar_5m_session_prefix",
                    input_roles=("prior_receipt",),
                ),
            )

        with pytest.raises(IdeaRunnerError, match="not retained"):
            IdeaRunner._load_continuation_events(
                connection,
                run_id="run-1",
                last_source_sequence=3,
                last_market_event_id=chain[-1],
                retained_event_ids=(),
                requirements=(requirement,),
                request=valid,
            )
        with pytest.raises(IdeaRunnerError, match="causal watermark"):
            IdeaRunner._load_continuation_events(
                connection,
                run_id="run-1",
                last_source_sequence=2,
                last_market_event_id=chain[1],
                retained_event_ids=(chain[-1],),
                requirements=(requirement,),
                request=valid,
            )
        with pytest.raises(IdeaRunnerError, match="not declared"):
            IdeaRunner._load_continuation_events(
                connection,
                run_id="run-1",
                last_source_sequence=3,
                last_market_event_id=chain[-1],
                retained_event_ids=(chain[-1],),
                requirements=(
                    MarketDataRequirement(
                        **{
                            **requirement.model_dump(mode="python"),
                            "event_kind": "session_volume_baseline",
                        }
                    ),
                ),
                request=valid,
            )
        with pytest.raises(IdeaRunnerError, match="not an allowed causal ancestor"):
            IdeaRunner._load_continuation_events(
                connection,
                run_id="run-1",
                last_source_sequence=3,
                last_market_event_id=chain[-1],
                retained_event_ids=(chain[-1],),
                requirements=(requirement,),
                request=ExactEventsContinuation(
                    root_event_ids=(chain[-1],),
                    event_ids=(unrelated,),
                    event_kind="bar_5m_session_prefix",
                    input_roles=("prior_receipt",),
                ),
            )
        with pytest.raises(IdeaRunnerError, match="cursor does not match"):
            IdeaRunner._load_continuation_events(
                connection,
                run_id="run-1",
                last_source_sequence=3,
                last_market_event_id=chain[-1],
                retained_event_ids=(chain[-1],),
                requirements=(requirement,),
                request=AncestorPageContinuation(
                    **{
                        **valid.model_dump(mode="python"),
                        "cursor": "1:" + "0" * 64,
                    }
                ),
            )


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
        assert connection.execute("SELECT count(*) FROM market_events").fetchone()[0] == len(
            first_evidence
        )
    if restart:
        runner.close()
        runner = IdeaRunner(database, (discovered,), run_id="run-1")
    remaining = _checkpoint_evidence(6, COHORT[10:], sequence_start=20)
    with connect_v2(database) as connection:
        for sequence, event in enumerate(remaining, start=61):
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
    assert json.loads(final_retained) == [event.event_id for event in expected_lineage]


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
        assert connection.execute("SELECT count(*) FROM market_events").fetchone()[0] == len(prior)
        next_open = _checkpoint_evidence(
            1,
            ("AAL",),
            session=datetime(2026, 8, 4, 13, 30, tzinfo=UTC),
            sequence_start=100,
        )[0]
        _persist_market_event(connection, 61, next_open)
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
                interests=(),
            )
        if self._failure == "state_overflow":
            return IdeaEvaluation.model_construct(
                state={"x": "y" * 70_000}, outputs=(), interests=()
            )
        return IdeaEvaluation(
            state={},
            outputs=(
                Observation(
                    subject_instrument_id="NOT_IN_UNIVERSE",
                    as_of_at_us=batch.events[0].event_at_us,
                    payload={},
                ),
            ),
            interests=(),
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
        _persist_market_event(connection, 1, _derived_progress_event(1))

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
    run1_event = _derived_progress_event(1, "AAL")
    run2_event = _derived_progress_event(2, "AAOI")
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
    events: list[MarketEvent] = []
    sequence = sequence_start
    for number in range(1, checkpoint + 1):
        boundary = session + timedelta(minutes=5 * number)
        for symbol in symbols:
            sequence += 1
            close = 101.0 + COHORT.index(symbol) / 100 if number == checkpoint else 100.0
            events.append(
                MarketEvent(
                    event_id=f"derived-{symbol}-{number}-{sequence}",
                    instrument_id=symbol,
                    feed_kind="bars",
                    event_kind="bar_5m",
                    event_at_us=int(boundary.timestamp() * 1_000_000),
                    received_at_us=int(boundary.timestamp() * 1_000_000),
                    payload={
                        "session": session.date().isoformat(),
                        "bar_number": number,
                        "bar_start_at_us": int(
                            (boundary - timedelta(minutes=5)).timestamp() * 1_000_000
                        ),
                        "bar_end_at_us": int(boundary.timestamp() * 1_000_000),
                        "source_completeness": "complete",
                        "first_source_sequence": sequence,
                        "open": 100.0,
                        "high": max(100.0, close),
                        "low": min(100.0, close),
                        "close": close,
                        "volume": 60.0,
                    },
                )
            )
    return tuple(events)


def _derived_progress_event(sequence: int, symbol: str = "AAL") -> MarketEvent:
    return _checkpoint_evidence(1, (symbol,), sequence_start=sequence - 1)[0]


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
    c12_evidence = tuple(
        event
        for event in _checkpoint_evidence(12, sequence_start=40)
        if cast(int, event.payload["bar_number"]) > 6
    )
    c12 = plugin.evaluate(_batch(c12_evidence), third.state)
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
    evidence: list[MarketEvent] = []
    for event in _checkpoint_evidence(6):
        payload = dict(event.payload)
        if COHORT.index(event.instrument_id) >= valid_count:
            payload["source_completeness"] = "incomplete"
        evidence.append(MarketEvent(**{**event.model_dump(mode="python"), "payload": payload}))
    evaluation = plugin.evaluate(_batch(tuple(evidence)), {})
    assert len(evaluation.outputs) == output_count
    if evaluation.outputs:
        assert all(output.payload["slate_size"] == valid_count for output in evaluation.outputs)


def test_opening_leader_does_not_emit_incomplete_cohort_at_session_rollover() -> None:
    plugin = discover_plugins((_config(),))[0].plugin
    prior = _checkpoint_evidence(6, COHORT[:15])
    rollover = _checkpoint_evidence(
        1, (COHORT[0],), session=datetime(2026, 8, 4, 13, 30, tzinfo=UTC)
    )[0]
    evaluation = plugin.evaluate(_batch((*prior, rollover)), {})
    assert evaluation.outputs == ()
    state = cast(Mapping[str, JsonValue], evaluation.state)
    assert state["session"] == "2026-08-04"


def test_opening_leader_excludes_conflicting_raw_duplicate_from_completed_slate() -> None:
    plugin = discover_plugins((_config(),))[0].plugin
    evidence = list(_checkpoint_evidence(6))
    aal_c6 = next(
        event
        for event in evidence
        if event.instrument_id == "AAL" and event.payload["bar_number"] == 6
    )
    evidence.insert(
        -1,
        MarketEvent(
            **{
                **aal_c6.model_dump(mode="python"),
                "event_id": "duplicate-aal-c6",
                "payload": {**dict(aal_c6.payload), "close": 999.0},
            }
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
        _persist_market_event(connection, 1, _derived_progress_event(1))
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
        _persist_market_event(connection, 1, _derived_progress_event(1))
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
        _persist_market_event(connection, 1, _derived_progress_event(1))
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
        _persist_market_event(connection, 1, _derived_progress_event(1))
    assert runner.run_once(now_us=1_000)[0].advanced is False
    runner._plugins_by_instance[activation.instance_id] = discovered.plugin
    assert runner.run_once(now_us=1_001_000)[0].advanced is True
    runner._plugins_by_instance[activation.instance_id] = object()
    with connect_v2(database) as connection:
        _persist_market_event(connection, 2, _derived_progress_event(2, "AAOI"))
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
        _persist_market_event(connection, 1, _derived_progress_event(1))
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
        _persist_market_event(connection, 1, _derived_progress_event(1))
    result = runner.run_once(now_us=1_000)[0]
    runner.close()
    # The derived receipt itself carries permanent gap overlap proof. Generic gap
    # latches therefore do not suppress already-materialized receipt processing.
    assert result.advanced is True


class _RecorderAdapter:
    capabilities = frozenset({"market_data"})

    def __init__(self) -> None:
        self.configured_subscriptions: tuple[IBKRSubscription, ...] = ()
        self.subscribed_request_ids: list[int] = []

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

    def configure_subscriptions(self, subscriptions: tuple[IBKRSubscription, ...]) -> None:
        self.configured_subscriptions = subscriptions

    def subscribe(self, fence: object) -> None:
        request_id = getattr(fence, "request_id", None)
        assert isinstance(request_id, int)
        self.subscribed_request_ids.append(request_id)
        return None

    def retry_subscription(self, fence: object) -> None:
        self.subscribe(fence)

    def cancel(self, request_id: int) -> None:
        return None


class _DynamicRecorderAdapter(_RecorderAdapter):
    def __init__(
        self,
        *,
        fail_dynamic_subscriptions: int = 0,
        fail_cancellations: int = 0,
        fail_parameter_calls: int = 0,
        inline_snapshot_at_us: int | None = None,
        deferred_snapshot_at_us: int | None = None,
        inline_rejection_at_us: int | None = None,
        inline_connection_status: MarketDataStatus | None = None,
    ) -> None:
        super().__init__()
        self.cancelled_request_ids: list[int] = []
        self.parameter_calls: list[tuple[int, str]] = []
        self.contract_calls: list[tuple[str, str, float, str]] = []
        self.subscribe_attempts: list[int] = []
        self.fail_dynamic_subscriptions = fail_dynamic_subscriptions
        self.fail_cancellations = fail_cancellations
        self.fail_parameter_calls = fail_parameter_calls
        self.inline_snapshot_at_us = inline_snapshot_at_us
        self.deferred_snapshot_at_us = deferred_snapshot_at_us
        self.inline_rejection_at_us = inline_rejection_at_us
        self.inline_connection_status = inline_connection_status
        self.release_snapshot = threading.Event()
        self.snapshot_completed = threading.Event()

    def subscribe(self, fence: object) -> None:
        request_id = getattr(fence, "request_id", None)
        assert isinstance(request_id, int)
        self.subscribe_attempts.append(request_id)
        if request_id >= 2_000_000 and self.fail_dynamic_subscriptions:
            self.fail_dynamic_subscriptions -= 1
            raise RuntimeError("synthetic subscription failure")
        super().subscribe(fence)
        if request_id >= 2_000_000 and self.inline_connection_status is not None:
            cast(Callable[[MarketDataStatus], None], self.status_callback)(
                self.inline_connection_status
            )
        if request_id >= 2_000_000 and self.inline_rejection_at_us is not None:
            cast(Callable[[MarketDataStatus], None], self.status_callback)(
                MarketDataStatus(
                    kind="pacing",
                    code=420,
                    request_id=request_id,
                    message="synthetic prompt pacing rejection",
                    received_at_us=self.inline_rejection_at_us,
                )
            )
        if request_id >= 2_000_000 and self.inline_snapshot_at_us is not None:
            cast(Callable[[MarketDataStatus], None], self.status_callback)(
                MarketDataStatus(
                    kind="snapshot_end",
                    code=0,
                    request_id=request_id,
                    message="complete inline",
                    received_at_us=self.inline_snapshot_at_us,
                )
            )
        if request_id >= 2_000_000 and self.deferred_snapshot_at_us is not None:

            def complete_snapshot() -> None:
                self.release_snapshot.wait(timeout=5)
                cast(Callable[[MarketDataStatus], None], self.status_callback)(
                    MarketDataStatus(
                        kind="snapshot_end",
                        code=0,
                        request_id=request_id,
                        message="complete from callback thread",
                        received_at_us=cast(int, self.deferred_snapshot_at_us),
                    )
                )
                self.snapshot_completed.set()

            threading.Thread(target=complete_snapshot, daemon=True).start()

    def cancel(self, request_id: int) -> None:
        self.cancelled_request_ids.append(request_id)
        if self.fail_cancellations:
            self.fail_cancellations -= 1
            raise RuntimeError("synthetic cancellation failure")

    def option_parameters(
        self, *, underlying_con_id: int, symbol: str
    ) -> tuple[OptionParameterSet, ...]:
        self.parameter_calls.append((underlying_con_id, symbol))
        if self.fail_parameter_calls:
            self.fail_parameter_calls -= 1
            raise RuntimeError("synthetic option metadata failure")
        return (
            OptionParameterSet(
                exchange="SMART",
                trading_class=symbol,
                multiplier="100",
                expirations=("20260811",),
                strikes=(99.0, 100.0, 101.0),
            ),
        )

    def option_contracts(
        self,
        *,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        multiplier: str,
        trading_class: str,
    ) -> tuple[ContractCandidate, ...]:
        assert (multiplier, trading_class) == ("100", symbol)
        self.contract_calls.append((symbol, expiry, strike, right))
        return (
            ContractCandidate(
                con_id=9_001,
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                multiplier="100",
                exchange="SMART",
                currency="USD",
                trading_class=trading_class,
            ),
        )


class _AdditiveDynamicRecorderAdapter(_DynamicRecorderAdapter):
    def __init__(self, *, fail_cancellations: int = 0) -> None:
        super().__init__(fail_cancellations=fail_cancellations)
        self.configured_by_request: dict[int, IBKRSubscription] = {}
        self.disconnect_calls = 0

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.configured_by_request.clear()

    def configure_subscriptions(self, subscriptions: tuple[IBKRSubscription, ...]) -> None:
        self.configured_by_request.update(
            {subscription.request_id: subscription for subscription in subscriptions}
        )

    def cancel(self, request_id: int) -> None:
        super().cancel(request_id)
        self.configured_by_request.pop(request_id, None)


class _BlockingConfigureAdditiveRecorderAdapter(_AdditiveDynamicRecorderAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.block_next_configuration = False
        self.configure_entered = threading.Event()
        self.release_configuration = threading.Event()
        self.disconnect_entered = threading.Event()

    def disconnect(self) -> None:
        self.disconnect_entered.set()
        super().disconnect()

    def configure_subscriptions(self, subscriptions: tuple[IBKRSubscription, ...]) -> None:
        if self.block_next_configuration:
            self.block_next_configuration = False
            self.configure_entered.set()
            if not self.release_configuration.wait(timeout=5):
                raise RuntimeError("timed out waiting to release synthetic configuration")
        super().configure_subscriptions(subscriptions)


class _BlockingConnectAdditiveRecorderAdapter(_AdditiveDynamicRecorderAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.connect_calls = 0
        self.first_reconnect_connect_entered = threading.Event()
        self.release_first_reconnect_connect = threading.Event()

    def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_calls == 2:
            self.first_reconnect_connect_entered.set()
            if not self.release_first_reconnect_connect.wait(timeout=5):
                raise RuntimeError("timed out waiting to release first reconnect")


def test_official_raw_bars_flow_through_recorder_into_reference_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clients: list[object] = []
    release_reader = threading.Event()

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
            self.wrapper.nextValidId(1)
            while not release_reader.wait(timeout=1):
                pass

        def disconnect(self) -> None:
            release_reader.set()

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
    monkeypatch.setattr(
        "stocker_runtime.ingestion.official_bridge.require_official_ibkr_api",
        lambda: package,
    )

    database = tmp_path / "official-plugin.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    config_value = _config(instruments=_configured_instruments()).model_dump(mode="json")
    idea_path.write_text(json.dumps([config_value]), encoding="utf-8")
    bridge = create_official_bridge(
        host="127.0.0.1",
        port=4002,
        client_id=71,
        read_only=True,
        external_read_only_verified=True,
        subscriptions=(),
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
        IBKRMarketData(bridge),
    )
    recorder._start_test_only_allow_empty_inputs(now_us=1, instruments=(), subscriptions=())
    client = clients[0]
    wrapper = client.wrapper  # type: ignore[attr-defined]
    private_bridge = cast(Any, bridge)
    private_bridge._callback_context.connection_epoch = private_bridge._active_connection_epoch
    open_seconds = int(datetime(2026, 8, 3, 13, 30, tzinfo=UTC).timestamp())
    callback_count = 0
    for bar_number in range(6 * 60):
        bar_seconds = open_seconds + bar_number * 5
        for index, _symbol in enumerate(COHORT, start=1):
            close = 101.0 + index / 100 if bar_number == 6 * 60 - 1 else 100.0
            wrapper.realtimeBar(
                client.requests[index - 1],  # type: ignore[attr-defined]
                bar_seconds,
                100.0,
                max(100.0, close),
                min(100.0, close),
                close,
                1,
                0,
                1,
            )
            callback_count += 1
    del private_bridge._callback_context.connection_epoch
    drain_at = time.time_ns() // 1_000 + 1_000_000
    assert callback_count == 7_200
    drained = 0
    transactions = 0
    while recorder.inbox.nonterminal_count():
        processed = recorder.drain(now_us=drain_at + transactions, limit=256)
        assert 0 < processed <= 256
        drained += processed
        transactions += 1
    assert drained == callback_count
    assert transactions == 29
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
    assert client.requests == list(range(1_000_000, 1_000_000 + len(COHORT)))  # type: ignore[attr-defined]
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


def test_omitted_enablement_creates_no_subscription_instance_or_evaluation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "recorder-omitted-enable.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    value = _config(instruments=_configured_instruments()).model_dump(mode="json")
    value.pop("enabled")
    idea_path.write_text(json.dumps([value]), encoding="utf-8")
    adapter = _RecorderAdapter()
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-omitted-enable",
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
        adapter,
    )
    recorder._start_test_only_allow_empty_inputs(now_us=100, instruments=(), subscriptions=())
    assert recorder.idea_requirements == ()
    assert adapter.configured_subscriptions == ()
    assert recorder.drain(now_us=101) == 0
    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM idea_instances").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM idea_outputs").fetchone()[0] == 0
    recorder.stop(now_us=102)


def test_recorder_explicit_config_wires_generic_requirements_and_activation(tmp_path: Path) -> None:
    database = tmp_path / "recorder.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    config_value = _config(instruments=_configured_instruments()).model_dump(mode="json")
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
    adapter = cast(_RecorderAdapter, recorder.adapter)
    recorder._start_test_only_allow_empty_inputs(now_us=100, instruments=(), subscriptions=())
    with connect_v2(database) as connection:
        instances = connection.execute("SELECT count(*) FROM idea_instances").fetchone()[0]
    assert len(recorder.idea_requirements) == len(COHORT)
    assert len(adapter.configured_subscriptions) == len(COHORT)
    assert adapter.subscribed_request_ids == sorted(adapter.subscribed_request_ids)
    assert {item.symbol for item in adapter.configured_subscriptions} == set(COHORT)
    assert instances == 1
    recorder.stop(now_us=101)


def _synthetic_discovered(config: IdeaConfig) -> DiscoveredPlugin:
    base = discover_plugins((_config(),))[0]
    requirement = MarketDataRequirement(
        feed_kind="bars",
        event_kind="bar_5m",
        instrument_id="AAL",
        cadence="5s",
        gaps_block=True,
        staleness_block=True,
    )
    return replace(
        base,
        config=config,
        universe_hash=hashlib.sha256(canonical_json_bytes(config.universe)).hexdigest(),
        requirements=(requirement,),
    )


def _dynamic_discovered(
    config: IdeaConfig,
    *,
    interest_lifetime_us: int = 3_600_000_000,
    cadence: Literal["snapshot", "stream"] = "snapshot",
) -> DiscoveredPlugin:
    base = _synthetic_discovered(config)
    plugin = _InterestPlugin(
        base.plugin,
        lifetime_us=interest_lifetime_us,
        cadence=cadence,
    )
    manifest_json = plugin.manifest.to_canonical_json().decode()
    requirement = MarketDataRequirement(
        feed_kind="bars",
        event_kind="bar",
        instrument_id="AAL",
        cadence="5s",
        gaps_block=True,
        staleness_block=True,
    )
    return replace(
        base,
        plugin=plugin,
        manifest=plugin.manifest,
        manifest_json=manifest_json,
        manifest_hash=hashlib.sha256(manifest_json.encode()).hexdigest(),
        requirements=(requirement,),
    )


def _dynamic_recorder(
    database: Path,
    idea_path: Path,
    adapter: _DynamicRecorderAdapter,
    *,
    owner_id: str,
    line_limit: int = 100,
    writer_lease_stale_us: int = 60_000_000,
) -> Recorder:
    return Recorder(
        RecorderConfig(
            database=database,
            run_id="run-dynamic-shadow",
            owner_id=owner_id,
            mode="shadow",
            host="127.0.0.1",
            port=4002,
            client_id=1,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
            market_data_line_limit=line_limit,
            writer_lease_stale_us=writer_lease_stale_us,
            idea_config=idea_path,
        ),
        adapter,
    )


def _activate_dynamic_interest(recorder: Recorder, *, event_at_us: int) -> CallbackFence:
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )
    recorder.drain(now_us=event_at_us + 2)
    assert recorder.state is not None
    return next(
        fence for fence in recorder.state.fences if cast(int, fence.request_id) >= 2_000_000
    )


def test_static_subscription_in_dynamic_numeric_range_is_never_reconciled(
    tmp_path: Path,
) -> None:
    database = tmp_path / "static-high-request-id.sqlite3"
    initialize_database(database)
    adapter = _DynamicRecorderAdapter()
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-static-high-request-id",
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
        adapter,
    )
    instrument = InstrumentSpec("AAL", 1, "stock", "AAL", "SMART", "USD")
    static = SubscriptionSpec(
        name="static-aal-bars",
        instrument_id="AAL",
        feed_kind="bars",
        request_id=2_000_000,
        continuity_required=True,
        optional=False,
        stale_after_us=15_000_000,
    )

    recorder.start(now_us=100, instruments=(instrument,), subscriptions=(static,))
    recorder.drain(now_us=101)
    recorder.drain(now_us=102)

    assert recorder.state is not None
    assert tuple(fence.request_id for fence in recorder.state.fences) == (2_000_000,)
    assert adapter.cancelled_request_ids == []
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT lifecycle FROM subscriptions WHERE request_id=2000000"
            ).fetchone()[0]
            == "active"
        )
    recorder.stop(now_us=103)


def test_dynamic_request_high_water_survives_restart_without_subscription_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dynamic-request-high-water.sqlite3"
    initialize_database(database)

    def create(owner_id: str) -> Recorder:
        return Recorder(
            RecorderConfig(
                database=database,
                run_id="run-high-water",
                owner_id=owner_id,
                mode="shadow",
                host="127.0.0.1",
                port=4002,
                client_id=1,
                read_only=True,
                external_read_only_verified=True,
                config_hash="a" * 64,
                git_commit="deadbee",
                writer_lease_stale_us=15_000_000,
            ),
            _DynamicRecorderAdapter(),
        )

    first = create("owner-1")
    first._start_test_only_allow_empty_inputs(now_us=100, instruments=(), subscriptions=())
    first_request_id = first._next_dynamic_request_ids(1)[0]

    with connect_v2(database) as connection:
        assert connection.execute("SELECT count(*) FROM subscriptions").fetchone()[0] == 0
        assert (
            connection.execute("SELECT dynamic_request_high_water FROM runtime_state").fetchone()[0]
            == first_request_id
        )

    first.abandon_unclean()
    second = create("owner-2")
    second._start_test_only_allow_empty_inputs(now_us=15_000_101, instruments=(), subscriptions=())
    second_request_id = second._next_dynamic_request_ids(1)[0]

    assert (first_request_id, second_request_id) == (2_000_000, 2_000_001)
    second.stop(now_us=15_000_102)


def test_dynamic_interest_records_exact_option_and_recovers_across_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-recorder.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(
        universe=("AAL",),
        instruments=_configured_instruments(("AAL",)),
    )
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)

    first_adapter = _DynamicRecorderAdapter()
    first = _dynamic_recorder(database, idea_path, first_adapter, owner_id="owner-1")
    first_state = first._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    first.receive(
        first_state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )
    first.drain(now_us=event_at_us + 2)

    with connect_v2(database) as connection:
        interest = connection.execute("SELECT * FROM market_data_interests").fetchone()
        receipt = connection.execute("SELECT * FROM instrument_discovery_receipts").fetchone()
        active_dynamic = connection.execute(
            "SELECT * FROM subscriptions WHERE request_id>=2000000 AND lifecycle='active'"
        ).fetchone()
    assert interest is not None and interest["lifecycle"] == "active"
    assert receipt is not None and receipt["instrument_id"] == "ibkr-option-9001"
    assert active_dynamic is not None
    dynamic_request_id = int(active_dynamic["request_id"])
    assert first_adapter.parameter_calls == [(1, "AAL")]
    assert first_adapter.contract_calls == [("AAL", "20260811", 100.0, "C")]
    assert first_adapter.subscribed_request_ids[-1] == dynamic_request_id
    assert first.state is not None
    old_dynamic_fence = next(
        fence for fence in first.state.fences if fence.request_id == dynamic_request_id
    )

    first.abandon_unclean()
    restart_at_us = event_at_us + 120_000_003
    second_adapter = _DynamicRecorderAdapter()
    second = _dynamic_recorder(database, idea_path, second_adapter, owner_id="owner-2")
    second_state = second._start_test_only_allow_empty_inputs(
        now_us=restart_at_us, instruments=(), subscriptions=()
    )
    new_dynamic_fence = next(
        fence for fence in second_state.fences if cast(int, fence.request_id) >= 2_000_000
    )
    assert second_state.recorder_generation == first_state.recorder_generation + 1
    assert new_dynamic_fence.request_id != dynamic_request_id
    assert second_adapter.parameter_calls == []
    assert second_adapter.subscribed_request_ids[-1] == new_dynamic_fence.request_id

    stale = second.receive(
        old_dynamic_fence,
        MarketDataCallback(
            callback_kind="quote",
            received_at_us=restart_at_us + 1,
            provider_at_us=None,
            payload={"event_at_us": restart_at_us + 1, "bid": 1.0, "ask": 1.1},
        ),
    )
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT failure_code FROM callback_inbox WHERE source_sequence=?",
                (stale.source_sequence,),
            ).fetchone()[0]
            == "STALE_RECORDER_GENERATION"
        )

    expires_at_us = int(interest["expires_at_us"])
    second.drain(now_us=expires_at_us + 1)
    with connect_v2(database) as connection:
        lifecycle = connection.execute(
            "SELECT lifecycle FROM market_data_interests WHERE interest_id=?",
            (interest["interest_id"],),
        ).fetchone()[0]
        closed = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
            (new_dynamic_fence.subscription_id,),
        ).fetchone()[0]
    assert lifecycle == "expired"
    assert closed == "closed"
    assert second_adapter.cancelled_request_ids == [new_dynamic_fence.request_id]
    second.stop(now_us=expires_at_us + 2)


def test_repeated_same_contract_snapshot_gets_fresh_fence_and_rejects_late_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-repeated-contract.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    first_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    recorder.market_data_status(
        MarketDataStatus(
            kind="snapshot_end",
            code=0,
            request_id=first_fence.request_id,
            message="first complete",
            received_at_us=event_at_us + 3,
        )
    )
    assert recorder.state is not None
    base_fence = next(
        fence for fence in recorder.state.fences if cast(int, fence.request_id) < 2_000_000
    )
    recorder.receive(
        base_fence,
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 11,
            provider_at_us=event_at_us + 10,
            payload={
                "event_at_us": event_at_us + 10,
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1.0,
            },
        ),
    )
    recorder.drain(now_us=event_at_us + 12)
    assert recorder.state is not None
    second_fence = next(
        fence for fence in recorder.state.fences if cast(int, fence.request_id) >= 2_000_000
    )

    assert second_fence.request_id != first_fence.request_id
    assert second_fence.subscription_id != first_fence.subscription_id
    late = recorder.receive(
        first_fence,
        MarketDataCallback(
            callback_kind="quote",
            received_at_us=event_at_us + 13,
            provider_at_us=None,
            payload={"event_at_us": event_at_us + 13, "bid": 1.0, "ask": 1.1},
        ),
    )
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT failure_code FROM callback_inbox WHERE source_sequence=?",
                (late.source_sequence,),
            ).fetchone()[0]
            == "STALE_REQUEST_GENERATION"
        )
        rows = connection.execute(
            "SELECT request_id, lifecycle FROM subscriptions WHERE request_id>=2000000 "
            "ORDER BY opened_at_us, subscription_id"
        ).fetchall()
    assert len(rows) == 2
    assert rows[-1]["lifecycle"] == "active"
    recorder.stop(now_us=event_at_us + 14)


def test_snapshot_completion_fulfills_only_interests_bound_when_request_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-snapshot-binding.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    recorder = _dynamic_recorder(database, idea_path, _DynamicRecorderAdapter(), owner_id="owner")
    first_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    assert recorder.state is not None
    base_fence = next(
        fence for fence in recorder.state.fences if cast(int, fence.request_id) < 2_000_000
    )
    recorder.receive(
        base_fence,
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 11,
            provider_at_us=event_at_us + 10,
            payload={
                "event_at_us": event_at_us + 10,
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1.0,
            },
        ),
    )
    recorder.drain(now_us=event_at_us + 12)

    recorder.market_data_status(
        MarketDataStatus(
            kind="snapshot_end",
            code=0,
            request_id=first_fence.request_id,
            message="first complete",
            received_at_us=event_at_us + 13,
        )
    )

    with connect_v2(database) as connection:
        interests = connection.execute(
            "SELECT lifecycle, bound_subscription_id FROM market_data_interests "
            "ORDER BY as_of_at_us"
        ).fetchall()
    assert [row["lifecycle"] for row in interests] == ["fulfilled", "resolved"]
    assert interests[0]["bound_subscription_id"] == first_fence.subscription_id
    assert interests[1]["bound_subscription_id"] is None

    recorder.drain(now_us=event_at_us + 14)
    assert recorder.state is not None
    second_fence = next(
        fence for fence in recorder.state.fences if cast(int, fence.request_id) >= 2_000_000
    )
    assert second_fence.request_id != first_fence.request_id
    with connect_v2(database) as connection:
        second = connection.execute(
            "SELECT lifecycle, bound_subscription_id FROM market_data_interests "
            "ORDER BY as_of_at_us DESC LIMIT 1"
        ).fetchone()
    assert tuple(second) == ("active", second_fence.subscription_id)
    recorder.stop(now_us=event_at_us + 15)


@pytest.mark.parametrize(
    ("callback_offset_us", "expected_lifecycle"),
    ((6, "fulfilled"), (3_600_000_004, "expired")),
)
def test_snapshot_interest_merged_into_stream_honors_causal_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    callback_offset_us: int,
    expected_lifecycle: str,
) -> None:
    database = tmp_path / "dynamic-snapshot-on-stream.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured, cadence="stream")
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    recorder = _dynamic_recorder(database, idea_path, _DynamicRecorderAdapter(), owner_id="owner")
    stream_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    recorder._idea_runner = None

    with connect_v2(database) as connection:
        first_interest = connection.execute("SELECT * FROM market_data_interests").fetchone()
        first_receipt = connection.execute("SELECT * FROM instrument_discovery_receipts").fetchone()
        instance_id = str(first_interest["instance_id"])
        activation = discovered.activation(
            instance_id=instance_id,
            run_id="run-dynamic-shadow",
            data_class=ProtectedDataClass.SHADOW,
            activated_at_us=event_at_us,
        )
        snapshot_as_of_us = event_at_us + 3
        IdeaRunner._insert_interest(
            connection,
            activation,
            MarketDataInterest(
                interest_key="snapshot-on-existing-stream",
                underlying_instrument_id="AAL",
                minimum_days_to_expiry=1,
                maximum_days_to_expiry=1,
                option_right="call",
                strike_offset=0,
                reference_price=100.0,
                cadence="snapshot",
                as_of_at_us=snapshot_as_of_us,
                expires_at_us=snapshot_as_of_us + 3_600_000_000,
                required=True,
                priority=100,
                maximum_contracts=1,
                input_event_id=str(first_interest["input_event_id"]),
            ),
            snapshot_as_of_us,
        )
        snapshot_interest = connection.execute(
            "SELECT * FROM market_data_interests WHERE interest_key='snapshot-on-existing-stream'"
        ).fetchone()
        connection.execute(
            "INSERT INTO instrument_discovery_receipts(receipt_id, interest_id, run_id, "
            "instance_id, status, instrument_id, expiry, strike, option_right, multiplier, "
            "candidates_inspected, completed_at_us) VALUES ('receipt-snapshot-on-stream', ?, "
            "'run-dynamic-shadow', ?, 'resolved', ?, ?, ?, ?, ?, 1, ?)",
            (
                snapshot_interest["interest_id"],
                instance_id,
                first_receipt["instrument_id"],
                first_receipt["expiry"],
                first_receipt["strike"],
                first_receipt["option_right"],
                first_receipt["multiplier"],
                snapshot_as_of_us + 1,
            ),
        )
        connection.execute(
            "UPDATE market_data_interests SET lifecycle='resolved', updated_at_us=? "
            "WHERE interest_id=?",
            (snapshot_as_of_us + 1, snapshot_interest["interest_id"]),
        )

    recorder.drain(now_us=event_at_us + 5)
    with connect_v2(database) as connection:
        queued = connection.execute(
            "SELECT lifecycle, bound_subscription_id FROM market_data_interests "
            "WHERE interest_key='snapshot-on-existing-stream'"
        ).fetchone()
    assert tuple(queued) == ("active", stream_fence.subscription_id)

    recorder.receive(
        stream_fence,
        MarketDataCallback(
            callback_kind="quote",
            received_at_us=event_at_us + callback_offset_us,
            provider_at_us=None,
            payload={
                "event_at_us": event_at_us + callback_offset_us,
                "bid": 1.0,
                "ask": 1.1,
            },
        ),
    )
    recorder.drain(now_us=event_at_us + callback_offset_us + 1)
    with connect_v2(database) as connection:
        completed = connection.execute(
            "SELECT lifecycle, bound_subscription_id FROM market_data_interests "
            "WHERE interest_key='snapshot-on-existing-stream'"
        ).fetchone()
    assert tuple(completed) == (expected_lifecycle, stream_fence.subscription_id)
    recorder.stop(now_us=event_at_us + callback_offset_us + 2)


def test_prompt_dynamic_pacing_status_is_retried_and_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-inline-pacing.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter(inline_rejection_at_us=event_at_us + 3)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    first_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)

    with connect_v2(database) as connection:
        first = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
            (first_fence.subscription_id,),
        ).fetchone()
        interest = connection.execute(
            "SELECT attempts, reason_code FROM market_data_interests"
        ).fetchone()
    assert first["lifecycle"] == "disconnected"
    assert tuple(interest) == (1, "SUBSCRIPTION_RETRY")

    adapter.inline_rejection_at_us = None
    recorder.drain(now_us=event_at_us + 1_000_004)
    assert recorder.state is not None
    active_fence = next(
        fence for fence in recorder.state.fences if cast(int, fence.request_id) >= 2_000_000
    )
    assert active_fence.request_id != first_fence.request_id
    with connect_v2(database) as connection:
        open_status_gaps = connection.execute(
            "SELECT count(*) FROM gaps WHERE reason LIKE 'IBKR_STATUS_%' AND resolved_at_us IS NULL"
        ).fetchone()[0]
        open_status_incidents = connection.execute(
            "SELECT count(*) FROM incidents WHERE code LIKE 'IBKR_STATUS_%' "
            "AND resolved_at_us IS NULL"
        ).fetchone()[0]
        state = connection.execute(
            "SELECT lifecycle, reason FROM runtime_state WHERE run_id='run-dynamic-shadow'"
        ).fetchone()
    assert open_status_gaps == open_status_incidents == 0
    assert tuple(state) == ("running", None)
    recorder.stop(now_us=event_at_us + 1_000_005)


def test_dynamic_retry_does_not_resolve_future_status_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-future-status.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    first_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    capacity_incident_id = hashlib.sha256(
        b"run-dynamic-shadow|DYNAMIC_MARKET_DATA_CAPACITY"
    ).hexdigest()
    with connect_v2(database) as connection:
        connection.execute(
            "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
            "opened_at_us, resolved_at_us, details_json) VALUES (?, "
            "'run-dynamic-shadow', 'market_data', 'degraded', "
            "'DYNAMIC_MARKET_DATA_CAPACITY', ?, ?, '{}') ON CONFLICT(incident_id) "
            "DO UPDATE SET opened_at_us=excluded.opened_at_us, "
            "resolved_at_us=excluded.resolved_at_us",
            (capacity_incident_id, event_at_us + 10, event_at_us + 20),
        )
    future_status_at_us = event_at_us + 2_000_000
    retry_at_us = event_at_us + 1_000_000
    recorder.market_data_status(
        MarketDataStatus(
            kind="pacing",
            code=420,
            request_id=first_fence.request_id,
            message="future-dated pacing rejection",
            received_at_us=future_status_at_us,
        )
    )
    with connect_v2(database) as connection:
        connection.execute(
            "UPDATE market_data_interests SET next_attempt_at_us=? WHERE bound_subscription_id=?",
            (retry_at_us, first_fence.subscription_id),
        )

    recorder.drain(now_us=retry_at_us)

    with connect_v2(database) as connection:
        status_gap = connection.execute(
            "SELECT resolved_at_us FROM gaps WHERE reason='IBKR_STATUS_420_PACING'"
        ).fetchone()
        status_incident = connection.execute(
            "SELECT resolved_at_us FROM incidents WHERE code='IBKR_STATUS_420_PACING'"
        ).fetchone()
        active_dynamic = connection.execute(
            "SELECT count(*) FROM subscriptions WHERE request_id>=2000000 AND lifecycle='active'"
        ).fetchone()[0]
        capacity_incident = connection.execute(
            "SELECT opened_at_us, resolved_at_us FROM incidents WHERE incident_id=?",
            (capacity_incident_id,),
        ).fetchone()
    assert status_gap["resolved_at_us"] is None
    assert status_incident["resolved_at_us"] is None
    assert active_dynamic == 1
    assert tuple(capacity_incident) == (future_status_at_us, None)
    assert (
        len([request_id for request_id in adapter.subscribe_attempts if request_id >= 2_000_000])
        == 2
    )
    recorder.stop(now_us=future_status_at_us + 1)


@pytest.mark.parametrize(
    ("kind", "code", "affected", "connection_state", "lifecycle", "reason"),
    (
        ("temporary_disconnect", 1100, (), "disconnected", "degraded", "IBKR_DISCONNECT"),
        (
            "farm_degraded",
            2103,
            ("quotes", "trades"),
            "connected",
            "degraded",
            "IBKR_FARM_2103_DEGRADED",
        ),
    ),
)
def test_connection_status_during_dynamic_subscribe_waits_for_state_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["temporary_disconnect", "farm_degraded"],
    code: int,
    affected: tuple[Literal["quotes", "trades", "bars"], ...],
    connection_state: str,
    lifecycle: str,
    reason: str,
) -> None:
    database = tmp_path / f"dynamic-inline-{kind}.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    status = MarketDataStatus(
        kind=kind,
        code=code,
        request_id=None,
        message="synthetic connection-level status",
        received_at_us=event_at_us + 3,
        affected_feed_kinds=affected,
    )
    adapter = _DynamicRecorderAdapter(inline_connection_status=status)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)

    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT connection_state, lifecycle, reason FROM runtime_state "
            "WHERE run_id='run-dynamic-shadow'"
        ).fetchone()
        subscription = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
            (dynamic_fence.subscription_id,),
        ).fetchone()
        gap = connection.execute(
            "SELECT reason FROM gaps WHERE subscription_id=? AND resolved_at_us IS NULL",
            (dynamic_fence.subscription_id,),
        ).fetchone()
    assert tuple(runtime) == (connection_state, lifecycle, reason)
    assert subscription["lifecycle"] == (
        "disconnected" if kind == "temporary_disconnect" else "degraded"
    )
    assert gap["reason"] == reason
    recorder.stop(now_us=event_at_us + 4)


def test_fully_deferred_required_interest_records_capacity_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-capacity-deferred.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    recorder = _dynamic_recorder(
        database,
        idea_path,
        _DynamicRecorderAdapter(),
        owner_id="owner",
        line_limit=1,
    )

    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )
    recorder.drain(now_us=event_at_us + 2)

    with connect_v2(database) as connection:
        interest = connection.execute(
            "SELECT lifecycle, reason_code FROM market_data_interests"
        ).fetchone()
        incident = connection.execute(
            "SELECT code, resolved_at_us FROM incidents WHERE code='DYNAMIC_MARKET_DATA_CAPACITY'"
        ).fetchone()
    assert tuple(interest) == ("resolved", "CAPACITY_DEFERRED")
    assert incident is not None and incident["resolved_at_us"] is None
    recorder.stop(now_us=event_at_us + 3)


def test_restored_dynamic_snapshot_may_complete_inline_during_subscribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-restored-inline-snapshot.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    first = _dynamic_recorder(database, idea_path, _DynamicRecorderAdapter(), owner_id="owner-1")
    dynamic_fence = _activate_dynamic_interest(first, event_at_us=event_at_us)

    first.abandon_unclean()
    restart_at_us = event_at_us + 120_000_003
    adapter = _DynamicRecorderAdapter(inline_snapshot_at_us=restart_at_us + 1)
    second = _dynamic_recorder(database, idea_path, adapter, owner_id="owner-2")
    second_state = second._start_test_only_allow_empty_inputs(
        now_us=restart_at_us, instruments=(), subscriptions=()
    )
    restored_request_id = next(
        request_id for request_id in adapter.subscribe_attempts if request_id >= 2_000_000
    )
    assert restored_request_id not in {fence.request_id for fence in second_state.fences}

    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT lifecycle FROM market_data_interests").fetchone()[0]
            == "fulfilled"
        )
        assert (
            connection.execute(
                "SELECT lifecycle FROM subscriptions WHERE request_id=?",
                (restored_request_id,),
            ).fetchone()[0]
            == "closed"
        )
    assert restored_request_id != dynamic_fence.request_id
    assert adapter.subscribe_attempts.count(restored_request_id) == 1
    assert adapter.subscribe_attempts.count(cast(int, dynamic_fence.request_id)) == 0
    second.stop(now_us=restart_at_us + 2)


def test_dynamic_subscription_failure_survives_reconnect_and_retries_with_fresh_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-subscribe-retry.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter(fail_dynamic_subscriptions=1)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )
    recorder.drain(now_us=event_at_us + 2)
    dynamic_request_id = next(item for item in adapter.subscribe_attempts if item >= 2_000_000)
    with connect_v2(database) as connection:
        first_row = connection.execute(
            "SELECT subscription_id, lifecycle FROM subscriptions WHERE request_id=?",
            (dynamic_request_id,),
        ).fetchone()
    assert first_row["lifecycle"] == "paused"

    recorder.disconnected(now_us=event_at_us + 3)
    reconnected = recorder.reconnect(now_us=event_at_us + 4)
    reconnected_request_id = next(
        fence.request_id
        for fence in reconnected.fences
        if fence.request_id not in recorder._base_request_ids()
    )
    assert reconnected_request_id != dynamic_request_id
    with connect_v2(database) as connection:
        paused = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE request_id=?",
            (reconnected_request_id,),
        ).fetchone()
        unresolved = connection.execute(
            "SELECT lifecycle, bound_subscription_id FROM market_data_interests"
        ).fetchone()
    assert paused["lifecycle"] == "paused"
    assert tuple(unresolved) == ("resolved", first_row["subscription_id"])

    recorder.drain(now_us=event_at_us + 5)
    assert adapter.subscribe_attempts.count(dynamic_request_id) == 1
    recorder.drain(now_us=event_at_us + 1_000_003)

    dynamic_attempts = tuple(item for item in adapter.subscribe_attempts if item >= 2_000_000)
    with connect_v2(database) as connection:
        rows = connection.execute(
            "SELECT request_id, subscription_id, lifecycle FROM subscriptions "
            "WHERE request_id>=2000000 ORDER BY request_id"
        ).fetchall()
        interest_retry = connection.execute(
            "SELECT attempts, next_attempt_at_us, reason_code FROM market_data_interests"
        ).fetchone()
    assert len(dynamic_attempts) == len(set(dynamic_attempts)) == 2
    assert dynamic_attempts[0] == dynamic_request_id
    assert [row["request_id"] for row in rows] == [
        dynamic_request_id,
        reconnected_request_id,
        dynamic_attempts[1],
    ]
    assert [row["lifecycle"] for row in rows] == ["closed", "closed", "active"]
    assert tuple(interest_retry) == (0, 0, None)
    recorder.stop(now_us=event_at_us + 1_000_004)


def test_dynamic_subscription_retry_exhaustion_releases_the_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-subscribe-exhaustion.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter(fail_dynamic_subscriptions=6)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    first_attempt_at_us = event_at_us + 2
    for elapsed_us in (1_000_000, 3_000_000, 7_000_000, 15_000_000, 31_000_000):
        recorder.drain(now_us=first_attempt_at_us + elapsed_us)

    with connect_v2(database) as connection:
        interest = connection.execute("SELECT * FROM market_data_interests").fetchone()
        subscriptions = connection.execute(
            "SELECT request_id, lifecycle FROM subscriptions WHERE request_id>=2000000 "
            "ORDER BY request_id"
        ).fetchall()
        incident_count = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='DYNAMIC_SUBSCRIBE_FAILED'"
        ).fetchone()[0]
    dynamic_attempts = tuple(item for item in adapter.subscribe_attempts if item >= 2_000_000)
    assert len(dynamic_attempts) == len(set(dynamic_attempts)) == 5
    assert dynamic_attempts[0] == dynamic_fence.request_id
    assert interest["attempts"] == 5
    assert interest["reason_code"] == "SUBSCRIPTION_RETRY_EXHAUSTED"
    assert all(row["lifecycle"] == "closed" for row in subscriptions)
    assert incident_count == 5
    recorder.stop(now_us=first_attempt_at_us + 31_000_001)


def test_dynamic_cancellation_failure_is_tombstoned_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-cancel-retry.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter(fail_cancellations=1)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    with connect_v2(database) as connection:
        expires_at_us = int(
            connection.execute("SELECT expires_at_us FROM market_data_interests").fetchone()[0]
        )

    recorder.drain(now_us=expires_at_us + 1)
    with connect_v2(database) as connection:
        assert (
            connection.execute(
                "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
                (dynamic_fence.subscription_id,),
            ).fetchone()[0]
            == "cancelling"
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM incidents WHERE code='DYNAMIC_CANCEL_FAILED'"
            ).fetchone()[0]
            == 1
        )

    recorder.drain(now_us=expires_at_us + 2)
    with connect_v2(database) as connection:
        incident = connection.execute(
            "SELECT resolved_at_us FROM incidents WHERE code='DYNAMIC_CANCEL_FAILED'"
        ).fetchone()
        assert (
            connection.execute(
                "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
                (dynamic_fence.subscription_id,),
            ).fetchone()[0]
            == "closed"
        )
    assert adapter.cancelled_request_ids == [
        dynamic_fence.request_id,
        dynamic_fence.request_id,
    ]
    assert incident["resolved_at_us"] == expires_at_us + 2
    recorder.stop(now_us=expires_at_us + 3)


def test_dynamic_replacement_cancel_failures_do_not_leak_additive_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-cancel-mapping-bound.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _AdditiveDynamicRecorderAdapter(fail_cancellations=3)
    recorder = _dynamic_recorder(
        database,
        idea_path,
        adapter,
        owner_id="owner",
        line_limit=2,
    )
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    expected_request_ids = {
        cast(int, fence.request_id) for fence in recorder._authority_state().fences
    }
    recorder.market_data_status(
        MarketDataStatus(
            kind="pacing",
            code=420,
            request_id=dynamic_fence.request_id,
            message="force replacement retries",
            received_at_us=event_at_us + 3,
        )
    )
    with connect_v2(database) as connection:
        retry_state_before_cancellations = tuple(
            connection.execute(
                "SELECT attempts, next_attempt_at_us, reason_code FROM market_data_interests"
            ).fetchone()
        )

    for retry_at_us in (event_at_us + 1_000_004, event_at_us + 3_000_004, event_at_us + 7_000_004):
        recorder.drain(now_us=retry_at_us)
        assert set(adapter.configured_by_request) == expected_request_ids
        assert len(adapter.configured_by_request) == 2

    with connect_v2(database) as connection:
        failed_replacements = connection.execute(
            "SELECT request_id, lifecycle FROM subscriptions WHERE request_id>=2000000 "
            "AND request_id!=? ORDER BY request_id",
            (dynamic_fence.request_id,),
        ).fetchall()
        unresolved_incidents = connection.execute(
            "SELECT code, count(*) AS total FROM incidents WHERE resolved_at_us IS NULL "
            "AND code IN ('DYNAMIC_CANCEL_FAILED','DYNAMIC_SUBSCRIBE_FAILED') "
            "GROUP BY code ORDER BY code"
        ).fetchall()
        retry_state = connection.execute(
            "SELECT attempts, next_attempt_at_us, reason_code FROM market_data_interests"
        ).fetchone()
    assert len(failed_replacements) == 3
    assert all(row["lifecycle"] == "closed" for row in failed_replacements)
    assert [(row["code"], row["total"]) for row in unresolved_incidents] == [
        ("DYNAMIC_CANCEL_FAILED", 1)
    ]
    assert tuple(retry_state) == retry_state_before_cancellations
    recorder.stop(now_us=event_at_us + 7_000_005)


def test_dynamic_cancel_failure_does_not_misreport_unattempted_new_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-cancel-aborted-new-start.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _AdditiveDynamicRecorderAdapter(fail_cancellations=1)
    recorder = _dynamic_recorder(
        database,
        idea_path,
        adapter,
        owner_id="owner",
        line_limit=2,
    )
    old_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    recorder._idea_runner = None

    new_as_of_us = event_at_us + 3_600_000_001
    with connect_v2(database) as connection:
        first_interest = connection.execute("SELECT * FROM market_data_interests").fetchone()
        instance_id = str(first_interest["instance_id"])
        activation = discovered.activation(
            instance_id=instance_id,
            run_id="run-dynamic-shadow",
            data_class=ProtectedDataClass.SHADOW,
            activated_at_us=event_at_us,
        )
        IdeaRunner._insert_interest(
            connection,
            activation,
            MarketDataInterest(
                interest_key="unrelated-new-start",
                underlying_instrument_id="AAL",
                minimum_days_to_expiry=1,
                maximum_days_to_expiry=1,
                option_right="put",
                strike_offset=0,
                reference_price=100.0,
                cadence="snapshot",
                as_of_at_us=new_as_of_us,
                expires_at_us=new_as_of_us + 3_600_000_000,
                required=True,
                priority=100,
                maximum_contracts=1,
                input_event_id=str(first_interest["input_event_id"]),
            ),
            new_as_of_us,
        )
        new_interest = connection.execute(
            "SELECT * FROM market_data_interests WHERE interest_key='unrelated-new-start'"
        ).fetchone()
        connection.execute(
            "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, symbol, "
            "exchange, currency, option_expiry, option_strike, option_right, option_multiplier) "
            "VALUES ('ibkr-option-9002', ?, 9002, 'option', 'AAL', 'SMART', 'USD', "
            "'20260811', '100', 'put', '100')",
            ("8" * 64,),
        )
        connection.execute(
            "INSERT INTO instrument_discovery_receipts(receipt_id, interest_id, run_id, "
            "instance_id, status, instrument_id, expiry, strike, option_right, multiplier, "
            "candidates_inspected, completed_at_us) VALUES "
            "('receipt-unrelated-new-start', ?, 'run-dynamic-shadow', ?, 'resolved', "
            "'ibkr-option-9002', '20260811', 100, 'put', '100', 1, ?)",
            (new_interest["interest_id"], instance_id, new_as_of_us + 1),
        )
        connection.execute(
            "UPDATE market_data_interests SET lifecycle='resolved', updated_at_us=? "
            "WHERE interest_id=?",
            (new_as_of_us + 1, new_interest["interest_id"]),
        )
    recorder.drain(now_us=new_as_of_us + 2)
    assert adapter.cancelled_request_ids == [old_fence.request_id]
    assert [item for item in adapter.subscribe_attempts if item >= 2_000_000] == [
        old_fence.request_id
    ]
    with connect_v2(database) as connection:
        unresolved = connection.execute(
            "SELECT code FROM incidents WHERE resolved_at_us IS NULL AND code IN "
            "('DYNAMIC_CANCEL_FAILED','DYNAMIC_SUBSCRIBE_FAILED') ORDER BY code"
        ).fetchall()
        retry_state = connection.execute(
            "SELECT lifecycle, attempts, next_attempt_at_us, reason_code, "
            "bound_subscription_id FROM market_data_interests "
            "WHERE interest_key='unrelated-new-start'"
        ).fetchone()
        aborted_start = connection.execute(
            "SELECT request_id, lifecycle FROM subscriptions WHERE request_id>=2000000 "
            "AND request_id!=?",
            (old_fence.request_id,),
        ).fetchone()
    assert [row["code"] for row in unresolved] == ["DYNAMIC_CANCEL_FAILED"]
    assert tuple(retry_state) == (
        "resolved",
        0,
        new_as_of_us,
        "CAPACITY_DEFERRED",
        None,
    )
    assert aborted_start["lifecycle"] == "closed"
    assert {
        cast(int, fence.request_id)
        for fence in recorder._authority_state().fences
        if cast(int, fence.request_id) >= 2_000_000
    } == {cast(int, old_fence.request_id)}

    recorder.drain(now_us=new_as_of_us + 1_000_003)
    with connect_v2(database) as connection:
        unresolved = connection.execute(
            "SELECT code FROM incidents WHERE resolved_at_us IS NULL AND code IN "
            "('DYNAMIC_CANCEL_FAILED','DYNAMIC_SUBSCRIBE_FAILED') ORDER BY code"
        ).fetchall()
        recovered_interest = connection.execute(
            "SELECT lifecycle, attempts, next_attempt_at_us, reason_code "
            "FROM market_data_interests WHERE interest_key='unrelated-new-start'"
        ).fetchone()
    assert unresolved == []
    assert tuple(recovered_interest) == ("active", 0, 0, None)
    assert adapter.cancelled_request_ids == [old_fence.request_id, old_fence.request_id]
    assert aborted_start["request_id"] not in adapter.subscribe_attempts
    recorder.stop(now_us=new_as_of_us + 1_000_004)


def test_dynamic_discovery_retries_are_backed_off_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-discovery-retry.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter(fail_parameter_calls=5)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )
    first_attempt_at_us = event_at_us + 2
    recorder.drain(now_us=first_attempt_at_us)
    for elapsed_us in (1_500_000, 4_000_000, 8_500_000, 17_000_000):
        recorder.drain(now_us=first_attempt_at_us + elapsed_us)

    with connect_v2(database) as connection:
        interest = connection.execute("SELECT * FROM market_data_interests").fetchone()
        receipt = connection.execute("SELECT * FROM instrument_discovery_receipts").fetchone()
        retry_incidents = connection.execute(
            "SELECT count(*) FROM incidents WHERE code='INSTRUMENT_DISCOVERY_RETRY'"
        ).fetchone()[0]
    assert len(adapter.parameter_calls) == 5
    assert interest["attempts"] == 5
    assert interest["lifecycle"] == "denied"
    assert receipt["status"] == "denied"
    assert receipt["reason_code"] == "DISCOVERY_RETRY_EXHAUSTED"
    assert retry_incidents == 4
    recorder.stop(now_us=first_attempt_at_us + 17_000_001)


def test_dynamic_discovery_retry_never_moves_past_interest_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-discovery-expiry.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured, interest_lifetime_us=500_000)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter(fail_parameter_calls=1)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )
    recorder.drain(now_us=event_at_us + 2)

    with connect_v2(database) as connection:
        interest = connection.execute(
            "SELECT lifecycle, expires_at_us, next_attempt_at_us FROM market_data_interests"
        ).fetchone()
    assert interest["lifecycle"] == "pending"
    assert interest["next_attempt_at_us"] == interest["expires_at_us"]
    assert len(adapter.parameter_calls) == 1

    recorder.drain(now_us=int(interest["expires_at_us"]) + 1)
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT lifecycle FROM market_data_interests").fetchone()[0]
            == "expired"
        )
    recorder.stop(now_us=int(interest["expires_at_us"]) + 2)


def test_dynamic_discovery_response_after_expiry_never_resolves_or_subscribes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-discovery-completes-after-expiry.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured, interest_lifetime_us=500_000)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    elapsed = iter((0, 1_000_000_000))
    monkeypatch.setattr(
        recorder_module,
        "monotonic_ns",
        lambda: next(elapsed, 1_000_000_000),
        raising=False,
    )
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )

    recorder.drain(now_us=event_at_us + 2)

    with connect_v2(database) as connection:
        interest = connection.execute(
            "SELECT lifecycle, reason_code FROM market_data_interests"
        ).fetchone()
        receipt_count = connection.execute(
            "SELECT count(*) FROM instrument_discovery_receipts"
        ).fetchone()[0]
    assert tuple(interest) == ("expired", "INTEREST_EXPIRED_DURING_DISCOVERY")
    assert receipt_count == 0
    assert all(request_id < 2_000_000 for request_id in adapter.subscribe_attempts)
    recorder.stop(now_us=event_at_us + 1_000_003)


def test_dynamic_discovery_refreshes_writer_lease_between_bounded_metadata_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-discovery-lease.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    elapsed_ns = [0]
    monkeypatch.setattr(recorder_module, "monotonic_ns", lambda: elapsed_ns[0])
    contract_started = threading.Event()
    release_contract = threading.Event()

    class SlowMetadataAdapter(_DynamicRecorderAdapter):
        def option_parameters(
            self, *, underlying_con_id: int, symbol: str
        ) -> tuple[OptionParameterSet, ...]:
            result = super().option_parameters(
                underlying_con_id=underlying_con_id,
                symbol=symbol,
            )
            elapsed_ns[0] = 10_000_000_000
            return result

        def option_contracts(
            self,
            *,
            symbol: str,
            expiry: str,
            strike: float,
            right: str,
            multiplier: str,
            trading_class: str,
        ) -> tuple[ContractCandidate, ...]:
            contract_started.set()
            assert release_contract.wait(timeout=5)
            return super().option_contracts(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                multiplier=multiplier,
                trading_class=trading_class,
            )

    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    recorder = _dynamic_recorder(
        database,
        idea_path,
        SlowMetadataAdapter(),
        owner_id="owner-1",
        writer_lease_stale_us=15_000_000,
    )
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )

    failure: list[BaseException] = []

    def drain() -> None:
        try:
            recorder.drain(now_us=event_at_us + 2)
        except BaseException as error:
            failure.append(error)

    worker = threading.Thread(target=drain)
    worker.start()
    assert contract_started.wait(timeout=5)
    contender = _dynamic_recorder(
        database,
        idea_path,
        _DynamicRecorderAdapter(),
        owner_id="owner-2",
        writer_lease_stale_us=15_000_000,
    )
    try:
        with pytest.raises(DuplicateWriterError):
            contender.start(
                now_us=event_at_us + 20_000_000,
                instruments=(),
                subscriptions=(),
            )
    finally:
        release_contract.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert failure == []
    recorder.stop(now_us=event_at_us + 20_000_001)


def test_dynamic_snapshot_completion_is_terminal_across_reconnect_and_late_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-snapshot.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )
    recorder.drain(now_us=event_at_us + 2)
    assert recorder.state is not None
    dynamic_fence = next(
        fence for fence in recorder.state.fences if cast(int, fence.request_id) >= 2_000_000
    )

    recorder.market_data_status(
        MarketDataStatus(
            kind="snapshot_end",
            code=0,
            request_id=dynamic_fence.request_id,
            message="complete",
            received_at_us=event_at_us + 3,
        )
    )
    assert recorder.state is not None
    assert dynamic_fence.request_id not in {fence.request_id for fence in recorder.state.fences}

    recorder.disconnected(now_us=event_at_us + 4)
    reconnected = recorder.reconnect(now_us=event_at_us + 5)
    assert dynamic_fence.request_id not in {fence.request_id for fence in reconnected.fences}
    assert adapter.subscribe_attempts.count(cast(int, dynamic_fence.request_id)) == 1

    late = recorder.receive(
        dynamic_fence,
        MarketDataCallback(
            callback_kind="quote",
            received_at_us=event_at_us + 6,
            provider_at_us=None,
            payload={"event_at_us": event_at_us + 6, "bid": 1.0, "ask": 1.1},
        ),
    )
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT lifecycle FROM market_data_interests").fetchone()[0]
            == "fulfilled"
        )
        assert (
            connection.execute(
                "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
                (dynamic_fence.subscription_id,),
            ).fetchone()[0]
            == "closed"
        )
        assert (
            connection.execute(
                "SELECT failure_code FROM callback_inbox WHERE source_sequence=?",
                (late.source_sequence,),
            ).fetchone()[0]
            == "STALE_REQUEST_GENERATION"
        )
    recorder.drain(now_us=event_at_us + 7)
    assert adapter.cancelled_request_ids == []
    assert adapter.subscribe_attempts.count(cast(int, dynamic_fence.request_id)) == 1
    recorder.stop(now_us=event_at_us + 8)


def test_dynamic_snapshot_completion_after_expiry_records_expired_not_fulfilled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-snapshot-after-expiry.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured, interest_lifetime_us=10_000_000)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    recorder = _dynamic_recorder(
        database,
        idea_path,
        _DynamicRecorderAdapter(),
        owner_id="owner",
    )
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)

    recorder.market_data_status(
        MarketDataStatus(
            kind="snapshot_end",
            code=0,
            request_id=dynamic_fence.request_id,
            message="complete after expiry",
            received_at_us=event_at_us + 10_000_001,
        )
    )

    with connect_v2(database) as connection:
        interest = connection.execute(
            "SELECT lifecycle, reason_code FROM market_data_interests"
        ).fetchone()
        subscription = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
            (dynamic_fence.subscription_id,),
        ).fetchone()
    assert tuple(interest) == ("expired", "INTEREST_EXPIRED")
    assert subscription[0] == "closed"
    recorder.stop(now_us=event_at_us + 10_000_002)


def test_unfinished_snapshot_reconnects_with_fresh_fence_and_exact_interest_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-snapshot-reconnect.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    old_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)

    recorder.disconnected(now_us=event_at_us + 3)
    state = recorder.reconnect(now_us=event_at_us + 4)
    new_fence = next(
        fence for fence in state.fences if fence.request_id not in recorder._base_request_ids()
    )
    assert new_fence.request_id != old_fence.request_id
    with connect_v2(database) as connection:
        interest = connection.execute(
            "SELECT lifecycle, bound_subscription_id FROM market_data_interests"
        ).fetchone()
    assert tuple(interest) == ("active", new_fence.subscription_id)

    recorder.market_data_status(
        MarketDataStatus(
            kind="snapshot_end",
            code=0,
            request_id=old_fence.request_id,
            message="late prior-generation completion",
            received_at_us=event_at_us + 5,
        )
    )
    with connect_v2(database) as connection:
        interest = connection.execute(
            "SELECT lifecycle, bound_subscription_id FROM market_data_interests"
        ).fetchone()
        current = connection.execute(
            "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
            (new_fence.subscription_id,),
        ).fetchone()
    assert tuple(interest) == ("active", new_fence.subscription_id)
    assert current["lifecycle"] == "active"

    recorder.market_data_status(
        MarketDataStatus(
            kind="snapshot_end",
            code=0,
            request_id=new_fence.request_id,
            message="current-generation completion",
            received_at_us=event_at_us + 6,
        )
    )
    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT lifecycle FROM market_data_interests").fetchone()[0]
            == "fulfilled"
        )
    recorder.drain(now_us=event_at_us + 7)
    dynamic_attempts = [
        request_id for request_id in adapter.subscribe_attempts if request_id >= 2_000_000
    ]
    assert dynamic_attempts == [old_fence.request_id, new_fence.request_id]
    recorder.stop(now_us=event_at_us + 8)


def test_direct_dynamic_reconnect_resets_additive_bridge_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-direct-reconnect.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _AdditiveDynamicRecorderAdapter()
    recorder = _dynamic_recorder(
        database,
        idea_path,
        adapter,
        owner_id="owner",
        line_limit=2,
    )
    first_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    dynamic_request_ids = [cast(int, first_fence.request_id)]

    for offset in range(3, 7):
        state = recorder.reconnect(now_us=event_at_us + offset)
        current = next(
            fence for fence in state.fences if fence.request_id not in recorder._base_request_ids()
        )
        dynamic_request_ids.append(cast(int, current.request_id))
        assert len(adapter.configured_by_request) == 2
        assert set(adapter.configured_by_request) == {
            cast(int, fence.request_id) for fence in state.fences
        }

    assert len(dynamic_request_ids) == len(set(dynamic_request_ids)) == 5
    assert adapter.disconnect_calls == 4
    assert [
        request_id for request_id in adapter.subscribe_attempts if request_id >= 2_000_000
    ] == dynamic_request_ids
    recorder.stop(now_us=event_at_us + 7)


def test_dynamic_reconcile_and_direct_reconnect_serialize_adapter_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-reconcile-reconnect-serialization.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _BlockingConfigureAdditiveRecorderAdapter()
    recorder = _dynamic_recorder(
        database,
        idea_path,
        adapter,
        owner_id="owner",
        line_limit=2,
    )
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    adapter.block_next_configuration = True
    errors: list[BaseException] = []
    reconnected: list[RecorderState] = []
    reconnect_attempted = threading.Event()

    def reconcile() -> None:
        try:
            recorder.receive(
                state.fences[0],
                MarketDataCallback(
                    callback_kind="bar",
                    received_at_us=event_at_us + 1,
                    provider_at_us=event_at_us,
                    payload={
                        "event_at_us": event_at_us,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                        "volume": 1.0,
                    },
                ),
            )
            recorder.drain(now_us=event_at_us + 2)
        except BaseException as error:
            errors.append(error)

    def reconnect() -> None:
        try:
            reconnect_attempted.set()
            reconnected.append(recorder.reconnect(now_us=event_at_us + 3))
        except BaseException as error:
            errors.append(error)

    reconcile_thread = threading.Thread(target=reconcile)
    reconcile_thread.start()
    assert adapter.configure_entered.wait(timeout=5)
    reconnect_thread = threading.Thread(target=reconnect)
    reconnect_thread.start()
    assert reconnect_attempted.wait(timeout=5)
    reset_raced_configuration = adapter.disconnect_entered.wait(timeout=0.25)
    adapter.release_configuration.set()
    reconcile_thread.join(timeout=5)
    reconnect_thread.join(timeout=5)

    assert not reconcile_thread.is_alive()
    assert not reconnect_thread.is_alive()
    assert errors == []
    assert reset_raced_configuration is False
    assert len(reconnected) == 1
    current_request_ids = {cast(int, fence.request_id) for fence in reconnected[0].fences}
    assert set(adapter.configured_by_request) == current_request_ids
    assert len(adapter.configured_by_request) == 2
    with connect_v2(database) as connection:
        current_rows = connection.execute(
            "SELECT request_id, lifecycle FROM subscriptions WHERE connection_generation=2 "
            "ORDER BY request_id"
        ).fetchall()
    assert [(row["request_id"], row["lifecycle"]) for row in current_rows] == [
        (request_id, "active") for request_id in sorted(current_request_ids)
    ]
    recorder.stop(now_us=event_at_us + 4)


def test_dynamic_reconnect_is_single_flight_through_subscription_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-concurrent-reconnect.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _BlockingConnectAdditiveRecorderAdapter()
    recorder = _dynamic_recorder(
        database,
        idea_path,
        adapter,
        owner_id="owner",
        line_limit=2,
    )
    initial = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    results: dict[str, RecorderState] = {}
    errors: list[BaseException] = []
    second_reconnect_started = threading.Event()

    def reconnect(label: str, at_us: int) -> None:
        try:
            if label == "second":
                second_reconnect_started.set()
            results[label] = recorder.reconnect(now_us=at_us)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=reconnect, args=("first", event_at_us + 3))
    first.start()
    assert adapter.first_reconnect_connect_entered.wait(timeout=5)
    second = threading.Thread(target=reconnect, args=("second", event_at_us + 4))
    second.start()
    assert second_reconnect_started.wait(timeout=5)
    assert second.is_alive()
    adapter.release_first_reconnect_connect.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results["first"].connection_generation == 2
    assert results["second"].connection_generation == 3
    assert initial.request_id not in {fence.request_id for fence in results["second"].fences}
    current_request_ids = {cast(int, fence.request_id) for fence in results["second"].fences}
    assert set(adapter.configured_by_request) == current_request_ids
    assert len(adapter.configured_by_request) == 2
    current_dynamic_request_id = next(
        request_id for request_id in current_request_ids if request_id >= 2_000_000
    )
    assert adapter.subscribe_attempts.count(current_dynamic_request_id) == 1
    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT connection_generation, connection_state, lifecycle FROM runtime_state"
        ).fetchone()
    assert tuple(runtime) == (3, "connected", "running")
    recorder.stop(now_us=event_at_us + 5)


def test_reconnect_expires_interest_before_cloning_dynamic_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-reconnect-expiry.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured, interest_lifetime_us=10_000_000)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)

    state = recorder.reconnect(now_us=event_at_us + 10_000_001)
    assert dynamic_fence.request_id not in {fence.request_id for fence in state.fences}
    assert all(
        fence.request_id in recorder._base_request_ids()
        for fence in state.fences
        if fence.request_id is not None
    )
    with connect_v2(database) as connection:
        interest = connection.execute(
            "SELECT lifecycle, reason_code FROM market_data_interests"
        ).fetchone()
        dynamic_rows = connection.execute(
            "SELECT request_id, lifecycle FROM subscriptions WHERE request_id>=2000000"
        ).fetchall()
    assert tuple(interest) == ("expired", "INTEREST_EXPIRED")
    assert [(row["request_id"], row["lifecycle"]) for row in dynamic_rows] == [
        (dynamic_fence.request_id, "closed")
    ]
    assert [request_id for request_id in adapter.subscribe_attempts if request_id >= 2_000_000] == [
        dynamic_fence.request_id
    ]
    recorder.stop(now_us=event_at_us + 10_000_002)


def test_dynamic_snapshot_may_complete_inline_during_subscribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-inline-snapshot.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter(inline_snapshot_at_us=event_at_us + 3)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )

    recorder.drain(now_us=event_at_us + 2)

    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT lifecycle FROM market_data_interests").fetchone()[0]
            == "fulfilled"
        )
        subscription = connection.execute(
            "SELECT request_id, lifecycle FROM subscriptions WHERE request_id>=2000000"
        ).fetchone()
    assert tuple(subscription) == (adapter.subscribe_attempts[-1], "closed")
    assert adapter.subscribe_attempts.count(int(subscription["request_id"])) == 1
    recorder.stop(now_us=event_at_us + 4)


def test_dynamic_snapshot_callback_thread_is_held_until_state_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-threaded-snapshot.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter(deferred_snapshot_at_us=event_at_us + 3)
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    original_record_attempt = recorder._record_subscription_attempt

    def release_during_persistence(
        connection: sqlite3.Connection,
        plan: MarketDataPlan,
        spec: SubscriptionSpec,
        *,
        succeeded: bool,
        now_us: int,
    ) -> None:
        adapter.release_snapshot.set()
        assert adapter.snapshot_completed.wait(timeout=5)
        original_record_attempt(
            connection,
            plan,
            spec,
            succeeded=succeeded,
            now_us=now_us,
        )

    monkeypatch.setattr(recorder, "_record_subscription_attempt", release_during_persistence)
    state = recorder._start_test_only_allow_empty_inputs(
        now_us=event_at_us, instruments=(), subscriptions=()
    )
    recorder.receive(
        state.fences[0],
        MarketDataCallback(
            callback_kind="bar",
            received_at_us=event_at_us + 1,
            provider_at_us=event_at_us,
            payload={
                "event_at_us": event_at_us,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
        ),
    )

    recorder.drain(now_us=event_at_us + 2)

    with connect_v2(database) as connection:
        assert (
            connection.execute("SELECT lifecycle FROM market_data_interests").fetchone()[0]
            == "fulfilled"
        )
        assert (
            connection.execute(
                "SELECT lifecycle FROM subscriptions WHERE request_id>=2000000"
            ).fetchone()[0]
            == "closed"
        )
    recorder.stop(now_us=event_at_us + 4)


def test_completed_dynamic_snapshot_cannot_be_republished_by_inflight_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-snapshot-reconcile-race.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    original_configure = adapter.configure_subscriptions
    reconciliation_started = threading.Event()
    release_reconciliation = threading.Event()

    def pause_reconciliation(subscriptions: tuple[IBKRSubscription, ...]) -> None:
        reconciliation_started.set()
        assert release_reconciliation.wait(timeout=5)
        original_configure(subscriptions)

    monkeypatch.setattr(adapter, "configure_subscriptions", pause_reconciliation)
    failure: list[BaseException] = []

    def reconcile() -> None:
        try:
            recorder.drain(now_us=event_at_us + 3)
        except BaseException as error:  # pragma: no cover - asserted below
            failure.append(error)

    worker = threading.Thread(target=reconcile)
    worker.start()
    assert reconciliation_started.wait(timeout=5)
    completion_failure: list[BaseException] = []
    completion_started = threading.Event()

    def complete() -> None:
        try:
            completion_started.set()
            recorder.market_data_status(
                MarketDataStatus(
                    kind="snapshot_end",
                    code=0,
                    request_id=dynamic_fence.request_id,
                    message="complete during reconciliation",
                    received_at_us=event_at_us + 4,
                )
            )
        except BaseException as error:  # pragma: no cover - asserted below
            completion_failure.append(error)

    completion = threading.Thread(target=complete)
    completion.start()
    try:
        assert completion_started.wait(timeout=5)
        assert completion.is_alive()
    finally:
        release_reconciliation.set()
        worker.join(timeout=5)
        completion.join(timeout=5)

    assert not worker.is_alive()
    assert not completion.is_alive()
    assert failure == []
    assert completion_failure == []
    assert recorder.state is not None
    assert dynamic_fence.request_id not in {fence.request_id for fence in recorder.state.fences}
    recorder.disconnected(now_us=event_at_us + 5)
    reconnected = recorder.reconnect(now_us=event_at_us + 6)
    assert dynamic_fence.request_id not in {fence.request_id for fence in reconnected.fences}
    assert adapter.subscribe_attempts.count(cast(int, dynamic_fence.request_id)) == 1
    recorder.stop(now_us=event_at_us + 7)


def test_snapshot_retirement_serializes_concurrent_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-snapshot-reconnect-race.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    original_retire = recorder._retire_dynamic_snapshot_state
    retirement_started = threading.Event()
    release_retirement = threading.Event()

    def pause_retirement(*, request_id: int, fence: CallbackFence, state: RecorderState) -> None:
        retirement_started.set()
        assert release_retirement.wait(timeout=5)
        original_retire(request_id=request_id, fence=fence, state=state)

    monkeypatch.setattr(recorder, "_retire_dynamic_snapshot_state", pause_retirement)
    completion_failure: list[BaseException] = []

    def complete() -> None:
        try:
            recorder.market_data_status(
                MarketDataStatus(
                    kind="snapshot_end",
                    code=0,
                    request_id=dynamic_fence.request_id,
                    message="complete before concurrent reconnect",
                    received_at_us=event_at_us + 3,
                )
            )
        except BaseException as error:  # pragma: no cover - asserted below
            completion_failure.append(error)

    completion = threading.Thread(target=complete)
    completion.start()
    assert retirement_started.wait(timeout=5)
    reconnect_failure: list[BaseException] = []
    reconnect_state: list[RecorderState] = []
    reconnect_started = threading.Event()

    def reconnect() -> None:
        try:
            reconnect_started.set()
            recorder.disconnected(now_us=event_at_us + 4)
            reconnect_state.append(recorder.reconnect(now_us=event_at_us + 5))
        except BaseException as error:  # pragma: no cover - asserted below
            reconnect_failure.append(error)

    reconnecting = threading.Thread(target=reconnect)
    reconnecting.start()
    try:
        assert reconnect_started.wait(timeout=5)
        assert reconnecting.is_alive()
    finally:
        release_retirement.set()
        completion.join(timeout=5)
        reconnecting.join(timeout=5)

    assert not completion.is_alive()
    assert not reconnecting.is_alive()
    assert completion_failure == []
    assert reconnect_failure == []
    assert len(reconnect_state) == 1
    assert dynamic_fence.request_id not in {fence.request_id for fence in reconnect_state[0].fences}
    assert adapter.subscribe_attempts.count(cast(int, dynamic_fence.request_id)) == 1
    recorder.stop(now_us=event_at_us + 6)


def test_clean_stop_serializes_with_inflight_snapshot_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "dynamic-snapshot-stop-race.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    discovered = _dynamic_discovered(configured)
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (discovered,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    event_at_us = int(datetime(2026, 8, 10, 13, 30, tzinfo=UTC).timestamp() * 1_000_000)
    adapter = _DynamicRecorderAdapter()
    recorder = _dynamic_recorder(database, idea_path, adapter, owner_id="owner")
    dynamic_fence = _activate_dynamic_interest(recorder, event_at_us=event_at_us)
    original_retire = recorder._retire_dynamic_snapshot_state
    retirement_started = threading.Event()
    release_retirement = threading.Event()

    def pause_retirement(*, request_id: int, fence: CallbackFence, state: RecorderState) -> None:
        retirement_started.set()
        assert release_retirement.wait(timeout=5)
        original_retire(request_id=request_id, fence=fence, state=state)

    monkeypatch.setattr(recorder, "_retire_dynamic_snapshot_state", pause_retirement)
    completion_errors: list[BaseException] = []
    stop_errors: list[BaseException] = []
    stop_started = threading.Event()
    stop_completed = threading.Event()

    def complete() -> None:
        try:
            recorder.market_data_status(
                MarketDataStatus(
                    kind="snapshot_end",
                    code=0,
                    request_id=dynamic_fence.request_id,
                    message="complete during clean stop",
                    received_at_us=event_at_us + 3,
                )
            )
        except BaseException as error:
            completion_errors.append(error)

    def stop() -> None:
        try:
            stop_started.set()
            recorder.stop(now_us=event_at_us + 4)
        except BaseException as error:
            stop_errors.append(error)
        finally:
            stop_completed.set()

    completion = threading.Thread(target=complete)
    completion.start()
    assert retirement_started.wait(timeout=5)
    stopping = threading.Thread(target=stop)
    stopping.start()
    assert stop_started.wait(timeout=5)
    stop_won_race = stop_completed.wait(timeout=0.25)
    release_retirement.set()
    completion.join(timeout=5)
    stopping.join(timeout=5)

    assert stop_won_race is False
    assert not completion.is_alive()
    assert not stopping.is_alive()
    assert completion_errors == []
    assert stop_errors == []
    assert recorder.state is None
    with connect_v2(database) as connection:
        runtime = connection.execute(
            "SELECT lifecycle, connection_state FROM runtime_state"
        ).fetchone()
        run = connection.execute("SELECT status FROM runs").fetchone()
    assert tuple(runtime) == ("stopped", "disconnected")
    assert run["status"] == "running"


def test_synthetic_idea_config_adds_subscription_without_core_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "synthetic-subscription.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(
        universe=("AAL",),
        instruments=_configured_instruments(("AAL",)),
    )
    monkeypatch.setattr(
        recorder_module,
        "discover_plugins",
        lambda _configs: (_synthetic_discovered(configured),),
    )
    # The synthetic seam is the generic config: no Recorder instrument or
    # SubscriptionSpec is supplied for AAL.
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    adapter = _RecorderAdapter()
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-synthetic",
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
        adapter,
    )
    recorder._start_test_only_allow_empty_inputs(now_us=100, instruments=(), subscriptions=())
    assert adapter.configured_subscriptions == (
        IBKRSubscription(1_000_000, 1, "AAL", "STK", "SMART", "USD", "bars"),
    )
    assert adapter.subscribed_request_ids == [1_000_000]
    recorder.stop(now_us=101)


def test_synthetic_stream_requirement_adds_subscription_without_manual_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "synthetic-stream-subscription.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(
        universe=("AAL",),
        instruments=_configured_instruments(("AAL",)),
    )
    stream = replace(
        _synthetic_discovered(configured),
        requirements=(
            MarketDataRequirement(
                feed_kind="quotes",
                event_kind="quote",
                instrument_id="AAL",
                cadence="stream",
                gaps_block=True,
                staleness_block=True,
            ),
        ),
    )
    monkeypatch.setattr(recorder_module, "discover_plugins", lambda _configs: (stream,))
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    adapter = _RecorderAdapter()
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-synthetic-stream",
            owner_id="owner",
            mode="shadow",
            host="127.0.0.1",
            port=4002,
            client_id=1,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
            idea_config=idea_path,
        ),
        adapter,
    )

    recorder._start_test_only_allow_empty_inputs(now_us=100, instruments=(), subscriptions=())

    assert adapter.configured_subscriptions == (
        IBKRSubscription(1_000_000, 1, "AAL", "STK", "SMART", "USD", "quotes"),
    )
    assert recorder._subscriptions[0].stale_after_us == 15_000_000
    recorder.stop(now_us=101)


@pytest.mark.parametrize("seed_imported", (False, True), ids=("fresh", "imported"))
def test_exact_four_plugin_activation_shape(
    tmp_path: Path,
    seed_imported: bool,
) -> None:
    database = tmp_path / "four-plugin.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    universe = (*COHORT, "VTI")
    instruments = _configured_instruments(universe)
    if seed_imported:
        with connect_v2(database) as connection:
            for instrument in instruments:
                legacy_id = f"legacy-instrument-{instrument.instrument_id.lower()}"
                connection.execute(
                    "INSERT INTO instruments(instrument_id, identity_hash, ibkr_con_id, kind, "
                    "symbol, exchange, currency) VALUES (?, ?, ?, 'stock', ?, 'SMART', 'USD')",
                    (
                        legacy_id,
                        hashlib.sha256(legacy_id.encode()).hexdigest(),
                        instrument.ibkr_con_id,
                        instrument.symbol,
                    ),
                )

    configs = (
        _config(instruments=_configured_instruments()),
        IdeaConfig(
            module="stocker_ideas.plugins.frozen_m1c_signal_v0",
            expected_code_hash=reviewed_code_hash("stocker_ideas.plugins.frozen_m1c_signal_v0"),
            expected_manifest_hash=hashlib.sha256(
                frozen_m1c_plugin.MANIFEST.to_canonical_json()
            ).hexdigest(),
            parameters=cast(Mapping[str, JsonValue], frozen_m1c_plugin._PARAMETERS),
            universe=universe,
            instruments=instruments,
            enabled=True,
        ),
        IdeaConfig(
            module="stocker_ideas.plugins.m1c_quiet_state_options_v0",
            expected_code_hash=reviewed_code_hash(
                "stocker_ideas.plugins.m1c_quiet_state_options_v0"
            ),
            expected_manifest_hash=hashlib.sha256(
                quiet_plugin.MANIFEST.to_canonical_json()
            ).hexdigest(),
            parameters=quiet_plugin.PARAMETERS,
            universe=COHORT,
            instruments=_configured_instruments(),
            enabled=True,
        ),
        IdeaConfig(
            module="stocker_ideas.plugins.m1c_opening_reversal_v1_1",
            expected_code_hash=reviewed_code_hash(
                "stocker_ideas.plugins.m1c_opening_reversal_v1_1"
            ),
            expected_manifest_hash=hashlib.sha256(
                opening_reversal_plugin.MANIFEST.to_canonical_json()
            ).hexdigest(),
            parameters=opening_reversal_plugin.PARAMETERS,
            universe=universe,
            instruments=instruments,
            enabled=True,
        ),
    )
    idea_path.write_text(
        json.dumps([config.model_dump(mode="json") for config in configs]),
        encoding="utf-8",
    )
    adapter = _RecorderAdapter()
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-imported-four-plugin",
            owner_id="owner",
            mode="shadow",
            host="127.0.0.1",
            port=4002,
            client_id=1,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
            market_data_line_limit=100,
            idea_config=idea_path,
        ),
        adapter,
    )

    state = recorder._start_test_only_allow_empty_inputs(
        now_us=100, instruments=(), subscriptions=()
    )

    assert len(state.fences) == len(adapter.configured_subscriptions) == 41
    assert len({item.request_id for item in adapter.configured_subscriptions}) == 41
    assert sum(item.feed_kind == "bars" for item in adapter.configured_subscriptions) == 21
    assert sum(item.feed_kind == "quotes" for item in adapter.configured_subscriptions) == 20
    with connect_v2(database) as connection:
        alias_counts = tuple(
            connection.execute(
                "SELECT ibkr_con_id, count(*) AS aliases FROM instruments "
                "WHERE ibkr_con_id IS NOT NULL GROUP BY ibkr_con_id ORDER BY ibkr_con_id"
            )
        )
        active_subscriptions = connection.execute(
            "SELECT count(*) FROM subscriptions WHERE run_id='run-imported-four-plugin' "
            "AND lifecycle='active'"
        ).fetchone()[0]
        plugin_health = tuple(
            connection.execute(
                "SELECT idea_id, health FROM idea_instances "
                "WHERE run_id='run-imported-four-plugin' ORDER BY idea_id"
            )
        )
        foreign_keys = tuple(connection.execute("PRAGMA foreign_key_check"))
    assert all(row["aliases"] == (2 if seed_imported else 1) for row in alias_counts)
    assert len(alias_counts) == 21
    assert active_subscriptions == 41
    assert [tuple(row) for row in plugin_health] == [
        ("frozen_m1c_signal", "healthy"),
        ("m1c_opening_reversal", "healthy"),
        ("m1c_quiet_state_options", "healthy"),
        ("opening_leader_continuation", "healthy"),
    ]
    assert foreign_keys == ()
    recorder.stop(now_us=101)


def test_idea_requirement_without_same_entry_instrument_metadata_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "missing-instrument.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=())
    monkeypatch.setattr(
        recorder_module,
        "discover_plugins",
        lambda _configs: (_synthetic_discovered(configured),),
    )
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-missing-instrument",
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
    with pytest.raises(Exception, match="instrument metadata"):
        recorder._start_test_only_allow_empty_inputs(now_us=100, instruments=(), subscriptions=())


def test_generated_request_ids_skip_base_specs_and_reconnect_reuses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "request-ids.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    monkeypatch.setattr(
        recorder_module,
        "discover_plugins",
        lambda _configs: (_synthetic_discovered(configured),),
    )
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    adapter = _RecorderAdapter()
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-request-ids",
            owner_id="owner",
            mode="prospective_record",
            host="127.0.0.1",
            port=4002,
            client_id=1,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
            market_data_line_limit=2,
            idea_config=idea_path,
        ),
        adapter,
    )
    instrument = InstrumentSpec("MSFT", 2, "stock", "MSFT", "SMART", "USD")
    base = SubscriptionSpec(
        name="base-msft-bars",
        instrument_id="MSFT",
        feed_kind="bars",
        request_id=1_000_000,
        continuity_required=False,
        optional=True,
        stale_after_us=15_000_000,
    )
    first = recorder._start_test_only_allow_empty_inputs(
        now_us=100, instruments=(instrument,), subscriptions=(base,)
    )
    assert {item.request_id for item in adapter.configured_subscriptions} == {
        1_000_000,
        1_000_001,
    }
    assert {fence.request_id for fence in first.fences} == {1_000_000, 1_000_001}
    second = recorder.reconnect(now_us=101)
    assert second.connection_generation == first.connection_generation + 1
    assert {fence.request_id for fence in second.fences} == {1_000_000, 1_000_001}
    assert adapter.subscribed_request_ids == [1_000_000, 1_000_001] * 2
    recorder.stop(now_us=102)


def test_combined_subscriptions_fail_closed_at_explicit_line_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "line-limit.sqlite3"
    initialize_database(database)
    idea_path = tmp_path / "ideas.json"
    configured = _config(universe=("AAL",), instruments=_configured_instruments(("AAL",)))
    monkeypatch.setattr(
        recorder_module,
        "discover_plugins",
        lambda _configs: (_synthetic_discovered(configured),),
    )
    idea_path.write_text(json.dumps([configured.model_dump(mode="json")]), encoding="utf-8")
    recorder = Recorder(
        RecorderConfig(
            database=database,
            run_id="run-line-limit",
            owner_id="owner",
            mode="prospective_record",
            host="127.0.0.1",
            port=4002,
            client_id=1,
            read_only=True,
            external_read_only_verified=True,
            config_hash="a" * 64,
            git_commit="deadbee",
            market_data_line_limit=1,
            idea_config=idea_path,
        ),
        _RecorderAdapter(),
    )
    with pytest.raises(Exception, match="line limit"):
        recorder._start_test_only_allow_empty_inputs(
            now_us=100,
            instruments=(InstrumentSpec("MSFT", 2, "stock", "MSFT", "SMART", "USD"),),
            subscriptions=(
                SubscriptionSpec(
                    name="base-msft-bars",
                    instrument_id="MSFT",
                    feed_kind="bars",
                    request_id=1_000_000,
                    continuity_required=False,
                    optional=True,
                    stale_after_us=15_000_000,
                ),
            ),
        )
