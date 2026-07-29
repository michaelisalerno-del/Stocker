from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stocker_prospective.activation import ActivationRecord
from stocker_prospective.config import ProspectiveConfig, operational_thresholds
from stocker_prospective.context import previous_xnys_session
from stocker_prospective.contract import SECTOR_PROXY_BY_SYMBOL
from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.fake_ibkr import FakeIBKRAdapter, FakeIBKREvent
from stocker_prospective.frozen_live_application import (
    _require_compatible_existing_activation,
    build_frozen_prospective_application,
)
from stocker_prospective.frozen_m1c import FrozenM1CRuntime
from stocker_prospective.group_o import (
    GROUP_O_FEATURE_MANIFEST_SHA256,
    GROUP_O_REGIME_MAPPING_SHA256,
    FrozenGroupOSessionPackage,
    build_group_o_context,
)
from stocker_prospective.live_bars import xnys_session_bounds
from stocker_prospective.read_store import ProspectiveReadStore
from stocker_prospective.recorder import RecorderDeploymentIdentity

ROOT = Path(__file__).resolve().parents[1]
ARCHETYPE_ROOT = (
    ROOT
    / "research"
    / "directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts"
    / "primary"
)
SCALING_ARTIFACT = (
    ROOT
    / "research"
    / "route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts"
    / "primary"
    / "model_configurations.json"
)
RECORDER_RESEARCH = ROOT / "research/prospective/frozen-m1c-microstructure-recorder-v0"
FAKE_FIXTURE = (
    ROOT
    / "packages/stocker_prospective/src/stocker_prospective/fixtures"
    / "frozen-m1c-recorder-v0.json"
)


def _build_fake_application(
    tmp_path: Path,
    *,
    include_scientific_prerequisites: bool,
    complete_activity_baseline: bool = False,
    bar_compatibility_passed: bool | None = None,
    write_bar_compatibility_report: bool = True,
    events: tuple[FakeIBKREvent, ...] | None = None,
    git_commit: str = "a" * 40,
) -> tuple[object, ProspectiveConfig, FakeIBKRAdapter, tuple[str, ...]]:
    universe = json.loads(
        (ROOT / "configs/prospective/anchor-frozen-20.json").read_text(encoding="utf-8")
    )
    symbols = tuple(str(value) for value in universe["symbols"])
    activity_path = tmp_path / "historical-activity.parquet"
    if include_scientific_prerequisites:
        activity_sessions = ["2026-07-23"]
        activity_ordinals = range(1)
        if complete_activity_baseline:
            latest = previous_xnys_session(datetime.now(UTC).date() + timedelta(days=1))
            reversed_sessions = [latest]
            for _ in range(19):
                reversed_sessions.append(previous_xnys_session(reversed_sessions[-1]))
            activity_sessions = [session.isoformat() for session in reversed(reversed_sessions)]
            activity_ordinals = range(6)
        pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "session": session,
                    "bar_ordinal": ordinal,
                    "volume": 1_000.0,
                }
                for symbol in symbols
                for session in activity_sessions
                for ordinal in activity_ordinals
            ]
        ).to_parquet(activity_path, index=False)
    bar_report = tmp_path / "bar-compatibility.json"
    if include_scientific_prerequisites and write_bar_compatibility_report:
        bar_report.write_text(
            json.dumps(
                {"passed": (True if bar_compatibility_passed is None else bar_compatibility_passed)}
            )
            + "\n",
            encoding="utf-8",
        )
    context_root = tmp_path / "context"
    context_root.mkdir(exist_ok=True)

    config = ProspectiveConfig.model_validate(
        {
            "paths": {
                "database": tmp_path / "prospective.sqlite3",
                "bundle_root": tmp_path / "bundles",
                "feature_parity_report": (ROOT / "configs/prospective/feature-parity-m1.json"),
                "context_root": context_root,
                "raw_event_root": tmp_path / "raw-events",
                "recorder_activation": tmp_path / "activation.json",
                "m1c_live_parity_report": (RECORDER_RESEARCH / "m1c_live_parity_report.json"),
                "direction_live_parity_report": (
                    RECORDER_RESEARCH / "direction_live_parity_report.json"
                ),
                "ibkr_capability_manifest": tmp_path / "capability.json",
                "prospective_phase_ledger": tmp_path / "phases.jsonl",
                "prospective_report_root": tmp_path / "reports",
                "aggregate_transfer_report": tmp_path / "reports" / "aggregate.json",
                "frozen_m1c_artifact_root": ARCHETYPE_ROOT,
                "m1c_scaling_artifact": SCALING_ARTIFACT,
                "direction_beta_artifact": (ARCHETYPE_ROOT / "stock_market_beta_parameters.csv"),
                "historical_activity_bars": activity_path,
                "bar_compatibility_report": bar_report,
            },
            "runtime": {
                "mode": "record_only",
                "source": "ibkr",
                "prospective_start_utc": "2026-07-24T13:00:00Z",
                "instance_id": "fake-recorder-smoke",
                "app_version": "0.1.0",
                "git_commit": git_commit,
                "run_id": "fake-recorder-smoke-v0",
            },
            "risk": {"trading_enabled": False},
            "ibkr": {
                "host": "127.0.0.1",
                "port": 4003,
                "read_only": True,
                "market_data_type_required": "live",
                "tws_or_gateway_version": "Fake Gateway 10.37",
                "market_data_line_budget": 100,
                "reserved_line_headroom": 10,
                "request_rate_per_second": 50,
                "max_option_subscriptions": 30,
            },
            "context": {
                "mode": "signed_import",
                "hmac_secret_env": "NOT_USED_BY_FAKE_SMOKE",
                "import_directory": context_root,
            },
        }
    )
    identity = RecorderDeploymentIdentity(
        model_artifact_id="fake-smoke",
        universe_id=str(universe["universe_id"]),
        universe_hash=str(universe["universe_hash"]),
        symbols=symbols,
        bundle_verified=True,
    )
    adapter = (
        FakeIBKRAdapter.from_fixture(FAKE_FIXTURE)
        if events is None
        else FakeIBKRAdapter(
            fixture_id="engineering-shadow-checkpoint-v0",
            events=events,
        )
    )
    adapter.connect()
    repository = ProspectiveRepository(config.paths.database)
    repository.migrate()

    return (
        build_frozen_prospective_application(
            config=config,
            adapter=adapter,
            repository=repository,
            identity=identity,
            stock_contract_factory=lambda symbol: SimpleNamespace(
                symbol=symbol,
                secType="STK",
                exchange="SMART",
                currency="USD",
            ),
            option_contract_factory=(
                lambda symbol, expiry, strike, right, multiplier, exchange, trading: (
                    SimpleNamespace(
                        symbol=symbol,
                        secType="OPT",
                        lastTradeDateOrContractMonth=expiry.strftime("%Y%m%d"),
                        strike=strike,
                        right=right,
                        multiplier=str(multiplier),
                        exchange=exchange,
                        tradingClass=trading,
                        currency="USD",
                    )
                )
            ),
            ibkr_api_version="fake-10.37",
        ),
        config,
        adapter,
        symbols,
    )


def _completed_bar_events() -> tuple[tuple[FakeIBKREvent, ...], datetime]:
    events: list[FakeIBKREvent] = []
    sequence = 0
    candidate_session = datetime.now(UTC).date() + timedelta(days=1)
    while True:
        try:
            session_open, _session_close = xnys_session_bounds(candidate_session)
            break
        except ValueError:
            candidate_session += timedelta(days=1)
    for checkpoint in range(1, 8):
        bar_end = session_open + timedelta(minutes=5 * checkpoint)
        for symbol in ("AAL", "VTI"):
            events.append(
                FakeIBKREvent(
                    sequence=sequence,
                    scenario="engineering_shadow_checkpoint",
                    kind="five_minute_bar",
                    timestamp_utc=bar_end.isoformat(),
                    payload={"symbol": symbol, "checkpoint": checkpoint},
                )
            )
            sequence += 1
    observed = session_open + timedelta(minutes=35, seconds=1)
    return tuple(events), observed


def _write_valid_group_o_package(
    *,
    config: ProspectiveConfig,
    symbols: tuple[str, ...],
    signal_session: datetime,
) -> None:
    runtime = FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=ARCHETYPE_ROOT / "causal_movement_feature_manifest.json",
        threshold_path=ARCHETYPE_ROOT / "causal_movement_threshold.json",
    )
    session = signal_session.date()
    observation_session = previous_xnys_session(session)
    package = FrozenGroupOSessionPackage(
        contract_version="frozen-m1c-microstructure-recorder-v0/group-o-session-v0",
        signal_session=session,
        generated_from_authorised_cache=True,
        feature_manifest_hash=GROUP_O_FEATURE_MANIFEST_SHA256,
        regime_mapping_hash=GROUP_O_REGIME_MAPPING_SHA256,
        contexts=tuple(
            build_group_o_context(
                symbol=symbol,
                signal_session=session,
                actual_option_observation_session=observation_session,
                front_expiry=session + timedelta(days=3),
                dte=3,
                atm_strike=10.0,
                features={name: 0.0 for name in runtime.required_group_o_features},
                missing_indicators={},
                quality_status="valid",
                source_receipt_hashes=("a" * 64,),
            )
            for symbol in symbols
        ),
    )
    output = config.paths.context_root / "group-o" / f"{session.isoformat()}.json"
    output.parent.mkdir(parents=True)
    output.write_text(package.model_dump_json(indent=2), encoding="utf-8")


def test_engineering_transfer_projects_shadow_checkpoint_without_scientific_claim(
    tmp_path: Path,
) -> None:
    events, observed = _completed_bar_events()
    application, config, adapter, symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        complete_activity_baseline=True,
        bar_compatibility_passed=True,
        events=events,
    )
    _write_valid_group_o_package(
        config=config,
        symbols=symbols,
        signal_session=observed,
    )

    results = tuple(
        application.poll(now=observed + timedelta(seconds=offset)) for offset in range(3)
    )
    checkpoint_results = tuple(
        checkpoint for result in results for checkpoint in result.checkpoint_results
    )

    assert application.live_recorder.shadow_evaluation_enabled is True
    # Runtime artifacts are verified, but the immutable 20-session transfer
    # cohort still keeps this checkpoint out of scientific evidence.
    assert application.live_recorder.scientific_scoring_enabled is True
    assert len(checkpoint_results) == 1, [
        (
            result.callback_count,
            result.raw_event_count,
            result.finalised_bar_count,
            result.checkpoint_count,
            result.blocked_checkpoints,
        )
        for result in results
    ]
    assert "scientific_recording_not_authorized" in (checkpoint_results[0].rejection_reasons)
    universe = ProspectiveReadStore(
        config.paths.database,
        run_id=config.runtime.run_id,
    ).universe_live_v0()
    aal = next(item for item in universe if item["symbol"] == "AAL")
    assert aal["m1c_probability"] is not None
    assert aal["m1c_threshold"] == 0.488333710794033
    assert aal["m1c_scientific_eligible"] is False
    assert "scientific_recording_not_authorized" in aal["m1c_rejection_reasons"]
    assert ("level1", "AAL") in adapter.active_subscriptions.values()
    with sqlite3.connect(config.paths.database) as connection:
        checkpoint_eligibility = connection.execute(
            """
            SELECT eligible
            FROM m1c_checkpoint_v0
            WHERE run_id = ? AND symbol = 'AAL'
            """,
            (config.runtime.run_id,),
        ).fetchone()
        scientific_option_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM option_episode_allocation_v0
            WHERE run_id = ? AND scientific_option_evidence = 1
            """,
            (config.runtime.run_id,),
        ).fetchone()
    assert checkpoint_eligibility == (0,)
    assert scientific_option_rows == (0,)

    application.shutdown(now=observed + timedelta(seconds=1))


def test_missing_bar_compatibility_report_keeps_science_blocked_but_not_shadow(
    tmp_path: Path,
) -> None:
    events, observed = _completed_bar_events()
    application, config, _adapter, symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        complete_activity_baseline=True,
        write_bar_compatibility_report=False,
        events=events,
    )
    _write_valid_group_o_package(
        config=config,
        symbols=symbols,
        signal_session=observed,
    )
    for offset in range(3):
        application.poll(now=observed + timedelta(seconds=offset))

    assert application.live_recorder.shadow_evaluation_enabled is True
    assert application.live_recorder.scientific_scoring_enabled is False

    application.shutdown(now=observed + timedelta(seconds=3))


def test_full_recorder_application_starts_and_polls_with_fake_ibkr(
    tmp_path: Path,
) -> None:
    application, config, adapter, symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
    )
    polled_at = datetime.now(UTC)
    result = application.poll(now=polled_at)

    assert result.checkpoint_results == ()
    assert config.runtime.callback_inbox_batch_limit == 256
    assert config.runtime.raw_storage_heartbeat_stale_seconds == 360
    assert operational_thresholds(config).raw_storage_heartbeat_stale_after == timedelta(minutes=6)
    assert (tmp_path / "activation.json").is_file()
    assert (tmp_path / "capability.json").is_file()
    with sqlite3.connect(config.paths.database) as connection:
        recorded_symbols = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT symbol FROM underlying_contract"
            ).fetchall()
        }
    assert recorded_symbols == {*symbols, "VTI", *SECTOR_PROXY_BY_SYMBOL.values()}
    assert len(adapter.active_subscriptions) == len(recorded_symbols)
    assert {kind for kind, _symbol in adapter.active_subscriptions.values()} == {"bar"}
    assert (tmp_path / "ibkr_runtime_capacity_manifest.json").is_file()
    assert (
        json.loads((tmp_path / "capability.json").read_text(encoding="utf-8"))[
            "scientific_recording_valid"
        ]
        is False
    )

    application.shutdown(now=datetime.now(UTC))
    assert adapter.active_subscriptions == {}


def test_clock_probe_is_retried_after_the_operational_interval(
    tmp_path: Path,
) -> None:
    application, config, adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
    )
    adapter.drain_stream_events()
    application._last_clock_probe_monotonic = 100.0

    assert (
        application._request_clock_probe_if_due(
            monotonic_now=100.0 + config.ibkr.subscription_reconciliation_interval_seconds - 0.001,
        )
        is False
    )
    assert adapter.drain_stream_events() == ()

    assert (
        application._request_clock_probe_if_due(
            monotonic_now=(100.0 + config.ibkr.subscription_reconciliation_interval_seconds),
        )
        is True
    )
    assert tuple(item["kind"] for item in adapter.drain_stream_events()) == ("current_time",)
    assert (
        application._request_clock_probe_if_due(
            monotonic_now=(100.0 + config.ibkr.subscription_reconciliation_interval_seconds),
        )
        is False
    )

    application._last_clock_probe_monotonic = float("-inf")
    application.poll(now=datetime.now(UTC))
    assert tuple(item["kind"] for item in adapter.drain_stream_events()) == ("current_time",)

    application.shutdown(now=datetime.now(UTC))


def test_release_upgrade_preserves_first_activation_and_run_identity(
    tmp_path: Path,
) -> None:
    first_application, config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        git_commit="a" * 40,
    )
    activation_before = (tmp_path / "activation.json").read_bytes()
    first_application.shutdown(now=datetime.now(UTC))

    second_application, second_config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        git_commit="b" * 40,
    )

    assert second_config.runtime.git_commit == "b" * 40
    assert (tmp_path / "activation.json").read_bytes() == activation_before
    with sqlite3.connect(config.paths.database) as connection:
        run_git_commit = connection.execute(
            "SELECT git_commit FROM prospective_run WHERE run_id = ?",
            (config.runtime.run_id,),
        ).fetchone()
    assert run_git_commit == ("a" * 40,)

    second_application.shutdown(now=datetime.now(UTC))


def test_pre_hardening_activation_accepts_added_operational_fields(
    tmp_path: Path,
) -> None:
    application, config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        git_commit="a" * 40,
    )
    activation = ActivationRecord.model_validate_json(
        (tmp_path / "activation.json").read_text(encoding="utf-8")
    )
    application.shutdown(now=datetime.now(UTC))

    legacy_payload = config.model_dump(mode="json")
    for field_name in (
        "callback_inbox_max_unacknowledged",
        "callback_inbox_batch_limit",
        "callback_inbox_lease_seconds",
        "callback_heartbeat_stale_seconds",
        "raw_storage_heartbeat_stale_seconds",
        "callback_acknowledgement_stale_seconds",
        "callback_inbox_healthy_backlog",
        "callback_inbox_oldest_healthy_seconds",
    ):
        legacy_payload["runtime"].pop(field_name)
    legacy_hash = hashlib.sha256(
        json.dumps(
            legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    legacy_activation = activation.model_copy(update={"configuration_hash": legacy_hash})
    upgraded_config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"git_commit": "b" * 40})}
    )

    _require_compatible_existing_activation(
        activation=legacy_activation,
        config=upgraded_config,
        artifact_hashes=legacy_activation.model_artifact_hashes,
        ibkr_api_version=legacy_activation.ibkr_api_version,
        tws_or_gateway_version=legacy_activation.tws_or_gateway_version,
    )


def test_missing_scientific_inputs_degrade_to_live_acquisition(
    tmp_path: Path,
) -> None:
    application, config, adapter, symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=False,
    )
    observed = datetime(2026, 7, 28, 15, 0, tzinfo=UTC)

    result = application.poll(now=observed)

    assert result.checkpoint_results == ()
    assert adapter.active_subscriptions
    assert {kind for kind, _symbol in adapter.active_subscriptions.values()} == {"bar"}
    with sqlite3.connect(config.paths.database) as connection:
        connection.row_factory = sqlite3.Row
        blockers = {
            str(row["blocker_code"])
            for row in connection.execute(
                """
                    SELECT blocker_code FROM data_health_event
                    WHERE run_id = ? AND blocker_code IS NOT NULL
                    """,
                (config.runtime.run_id,),
            )
        }
        connection_state = connection.execute(
            "SELECT state FROM ibkr_connection_event WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (config.runtime.run_id,),
        ).fetchone()
    assert blockers == {
        "blocked_historical_activity_baseline_absent",
        "blocked_missing_previous_session_options_context",
    }
    assert connection_state is not None
    assert connection_state["state"] == "connected"

    runtime = FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=ARCHETYPE_ROOT / "causal_movement_feature_manifest.json",
        threshold_path=ARCHETYPE_ROOT / "causal_movement_threshold.json",
    )
    signal_session = observed.date()
    observation_session = previous_xnys_session(signal_session)
    package = FrozenGroupOSessionPackage(
        contract_version="frozen-m1c-microstructure-recorder-v0/group-o-session-v0",
        signal_session=signal_session,
        generated_from_authorised_cache=True,
        feature_manifest_hash=GROUP_O_FEATURE_MANIFEST_SHA256,
        regime_mapping_hash=GROUP_O_REGIME_MAPPING_SHA256,
        contexts=tuple(
            build_group_o_context(
                symbol=symbol,
                signal_session=signal_session,
                actual_option_observation_session=observation_session,
                front_expiry=signal_session + timedelta(days=3),
                dte=3,
                atm_strike=10.0,
                features={name: 0.0 for name in runtime.required_group_o_features},
                missing_indicators={},
                quality_status="valid",
                source_receipt_hashes=("a" * 64,),
            )
            for symbol in symbols
        ),
    )
    output = config.paths.context_root / "group-o" / f"{signal_session.isoformat()}.json"
    output.parent.mkdir(parents=True)
    output.write_text(package.model_dump_json(indent=2), encoding="utf-8")
    adapter.active_subscriptions.clear()
    application.poll(now=observed + timedelta(seconds=1))

    runtime_projection = ProspectiveReadStore(
        config.paths.database,
        run_id=config.runtime.run_id,
    ).runtime_projection()
    unresolved = {
        str(item["blocker_code"])
        for item in runtime_projection["blockers"]
        if item["blocker_code"] is not None
    }
    assert unresolved == {"blocked_historical_activity_baseline_absent"}

    application.shutdown(now=datetime.now(UTC))
    assert adapter.active_subscriptions == {}
