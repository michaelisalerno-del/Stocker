from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from stocker_prospective.activation import ActivationRecord
from stocker_prospective.config import ProspectiveConfig, operational_thresholds
from stocker_prospective.context import previous_xnys_session
from stocker_prospective.contract import M1C_BOTTOM_10_THRESHOLD, SECTOR_PROXY_BY_SYMBOL
from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.fake_ibkr import FakeIBKRAdapter, FakeIBKREvent, FakeRequestResult
from stocker_prospective.frozen_live_application import (
    _configuration_hash,
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
from stocker_prospective.live_subscriptions import LiveSubscriptionController
from stocker_prospective.market_data import ConnectionState
from stocker_prospective.operational_state import RecorderOperationalRepository
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
OPENING_REVERSAL_V1 = (
    ROOT / "research/prospective/20260729-m1c-prospective-opening-reversal-v1/artifacts/primary"
)
OPENING_REVERSAL_V1_1 = (
    ROOT / "research/prospective/20260729-m1c-prospective-opening-reversal-v1-1/artifacts/primary"
)
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
    app_version: str = "0.1.0",
    recorder_generation: int | None = None,
    run_id: str = "fake-recorder-smoke-v0",
    include_opening_reversal: bool = False,
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

    configured_paths: dict[str, object] = {
        "database": tmp_path / "prospective.sqlite3",
        "bundle_root": tmp_path / "bundles",
        "feature_parity_report": (ROOT / "configs/prospective/feature-parity-m1.json"),
        "context_root": context_root,
        "raw_event_root": tmp_path / "raw-events",
        "recorder_activation": tmp_path / "activation.json",
        "m1c_live_parity_report": (RECORDER_RESEARCH / "m1c_live_parity_report.json"),
        "direction_live_parity_report": (RECORDER_RESEARCH / "direction_live_parity_report.json"),
        "ibkr_capability_manifest": tmp_path / "capability.json",
        "prospective_phase_ledger": tmp_path / "phases.jsonl",
        "prospective_report_root": tmp_path / "reports",
        "aggregate_transfer_report": tmp_path / "reports" / "aggregate.json",
        "frozen_m1c_artifact_root": ARCHETYPE_ROOT,
        "m1c_scaling_artifact": SCALING_ARTIFACT,
        "direction_beta_artifact": (ARCHETYPE_ROOT / "stock_market_beta_parameters.csv"),
        "historical_activity_bars": activity_path,
        "bar_compatibility_report": bar_report,
    }
    if include_opening_reversal:
        configured_paths.update(
            {
                "m1c_prospective_opening_reversal_v1_config": (
                    OPENING_REVERSAL_V1 / "frozen_experiment_configuration_v1.json"
                ),
                "m1c_prospective_opening_reversal_v1_activation": (
                    OPENING_REVERSAL_V1 / "experiment_activation_receipt_v1.json"
                ),
                "m1c_prospective_opening_reversal_v1_1_config": (
                    OPENING_REVERSAL_V1_1 / "frozen_timing_addendum_configuration_v1_1.json"
                ),
                "m1c_prospective_opening_reversal_v1_1_activation": (
                    OPENING_REVERSAL_V1_1 / "experiment_activation_receipt_v1_1.json"
                ),
            }
        )
    config = ProspectiveConfig.model_validate(
        {
            "paths": configured_paths,
            "runtime": {
                "mode": "record_only",
                "source": "ibkr",
                "prospective_start_utc": "2026-07-24T13:00:00Z",
                "instance_id": "fake-recorder-smoke",
                "app_version": app_version,
                "git_commit": git_commit,
                "run_id": run_id,
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
            recorder_generation=recorder_generation,
            recorder_owner_id=(None if recorder_generation is None else "fake-recorder-owner"),
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
        if checkpoint == 7:
            quote_at = session_open + timedelta(minutes=30, seconds=-1)
            events.append(
                FakeIBKREvent(
                    sequence=sequence,
                    scenario="engineering_shadow_checkpoint",
                    kind="level1_quote",
                    timestamp_utc=quote_at.isoformat(),
                    payload={"symbol": "AAL", "bid": 50.0, "ask": 50.02},
                )
            )
            sequence += 1
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


def _delayed_quiet_checkpoint_events(
    *,
    boundary_quote_offset_seconds: float | None = -1.0,
    boundary_bid: float = 50.0,
    boundary_ask: float = 50.02,
    boundary_market_data_type: str = "live",
    processing_bid: float = 50.1,
    processing_ask: float = 50.12,
) -> tuple[tuple[FakeIBKREvent, ...], datetime, datetime]:
    candidate_session = datetime.now(UTC).date() + timedelta(days=1)
    while True:
        try:
            session_open, _session_close = xnys_session_bounds(candidate_session)
            break
        except ValueError:
            candidate_session += timedelta(days=1)
    trigger_end = session_open + timedelta(minutes=30)
    events: list[FakeIBKREvent] = []

    def append(
        *,
        kind: str,
        observed_at: datetime,
        payload: dict[str, object],
    ) -> None:
        events.append(
            FakeIBKREvent(
                sequence=len(events),
                scenario="delayed_quiet_checkpoint",
                kind=kind,
                timestamp_utc=observed_at.isoformat(),
                payload=payload,
            )
        )

    for checkpoint in range(1, 7):
        bar_start = session_open + timedelta(minutes=5 * (checkpoint - 1))
        received_at = bar_start + timedelta(minutes=4, seconds=59)
        for symbol in ("HIMS", "VTI"):
            append(
                kind="five_minute_bar",
                observed_at=received_at,
                payload={
                    "symbol": symbol,
                    "checkpoint": checkpoint,
                    "bar_start_utc": bar_start.isoformat(),
                },
            )
    if boundary_quote_offset_seconds is not None:
        append(
            kind="level1_quote",
            observed_at=trigger_end + timedelta(seconds=boundary_quote_offset_seconds),
            payload={
                "symbol": "HIMS",
                "bid": boundary_bid,
                "ask": boundary_ask,
                "market_data_type": boundary_market_data_type,
            },
        )
    append(
        kind="level1_quote",
        observed_at=trigger_end + timedelta(seconds=9),
        payload={"symbol": "HIMS", "bid": processing_bid, "ask": processing_ask},
    )
    for symbol in ("HIMS", "VTI"):
        append(
            kind="five_minute_bar",
            observed_at=trigger_end + timedelta(seconds=9, milliseconds=698),
            payload={
                "symbol": symbol,
                "checkpoint": 7,
                "bar_start_utc": trigger_end.isoformat(),
            },
        )
    return tuple(events), trigger_end + timedelta(seconds=10), trigger_end


def _install_independent_bar_timestamps(
    adapter: FakeIBKRAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_translate = adapter._translate

    def translate(event: FakeIBKREvent) -> tuple[dict[str, object], ...]:
        if event.kind == "level1_quote" and event.payload.get("market_data_type", "live") != "live":
            received_at = datetime.fromisoformat(event.timestamp_utc.replace("Z", "+00:00"))
            symbol = str(event.payload["symbol"])
            request_id = adapter._request_for(symbol, "level1")
            assert request_id is not None
            return tuple(
                {
                    "kind": "level1_quote_update",
                    "request_id": request_id,
                    "field": field,
                    "value": event.payload[field],
                    "market_data_type": str(event.payload["market_data_type"]),
                    "provider_timestamp_utc": received_at.isoformat(),
                    "received_timestamp_utc": received_at.isoformat(),
                    "received_monotonic_ns": event.sequence * 10 + offset,
                    "source_sequence": event.sequence * 10 + offset,
                }
                for offset, field in enumerate(("bid", "ask"), start=1)
            )
        if event.kind != "five_minute_bar":
            return original_translate(event)
        received_at = datetime.fromisoformat(event.timestamp_utc.replace("Z", "+00:00"))
        symbol = str(event.payload["symbol"])
        request_id = adapter._request_for(symbol, "bar")
        assert request_id is not None
        return (
            {
                "kind": "historical_bar_update",
                "request_id": request_id,
                "bar_start_utc": str(event.payload["bar_start_utc"]),
                "open": 100.0,
                "high": 100.2,
                "low": 99.9,
                "close": 100.1,
                "volume": 1_000.0,
                "wap": 100.05,
                "trade_count": 100,
                "provider_timestamp_utc": received_at.isoformat(),
                "received_timestamp_utc": received_at.isoformat(),
                "received_monotonic_ns": event.sequence * 10 + 1,
                "source_sequence": event.sequence * 10 + 1,
            },
        )

    monkeypatch.setattr(adapter, "_translate", translate)


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


def test_first_ibkr_session_is_authorized_without_transfer_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    monkeypatch.delenv("STOCKER_EODHD_TOKEN_CONFIGURED", raising=False)
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
    # Runtime artifacts and the IBKR observation itself determine eligibility;
    # cross-vendor comparison history is diagnostic only.
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
    assert "scientific_recording_not_authorized" not in (checkpoint_results[0].rejection_reasons)
    universe = ProspectiveReadStore(
        config.paths.database,
        run_id=config.runtime.run_id,
    ).universe_live_v0()
    aal = next(item for item in universe if item["symbol"] == "AAL")
    assert aal["m1c_probability"] is not None
    assert aal["m1c_threshold"] == 0.488333710794033
    # The IBKR signal is usable immediately when its frozen boundary quote is valid.
    assert aal["m1c_scientific_eligible"] is True
    assert "scientific_recording_not_authorized" not in aal["m1c_rejection_reasons"]
    assert "underlying_quote_stale" not in aal["m1c_rejection_reasons"]
    assert aal["m1c_diagnostic_quality_flags"] == []
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
    assert checkpoint_eligibility == (1,)
    assert scientific_option_rows == (0,)

    application.shutdown(now=observed + timedelta(seconds=1))


def test_delayed_checkpoint_uses_fresh_quote_as_of_bar_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, observed, trigger_end = _delayed_quiet_checkpoint_events()
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
    _install_independent_bar_timestamps(adapter, monkeypatch)
    try:
        results = tuple(
            application.poll(now=observed + timedelta(seconds=offset)) for offset in range(3)
        )
        checkpoints = tuple(
            checkpoint for result in results for checkpoint in result.checkpoint_results
        )
        assert checkpoints, [result.blocked_checkpoints for result in results]
        checkpoint = next(
            item
            for item in checkpoints
            if item.episode_decision.symbol == "HIMS" and item.episode_decision.checkpoint == 6
        )

        assert "underlying_quote_stale" not in checkpoint.diagnostic_quality_flags
        with sqlite3.connect(config.paths.database) as connection:
            selected_quote = connection.execute(
                """
                SELECT selected_underlying_quote_event_id,
                       selected_underlying_quote_timestamp_utc,
                       selected_underlying_quote_age_seconds,
                       underlying_quote_selection_policy
                FROM quiet_state_checkpoint_v0
                WHERE run_id = ? AND symbol = 'HIMS' AND checkpoint = 6
                """,
                (config.runtime.run_id,),
            ).fetchone()
        assert selected_quote is not None
        assert selected_quote[0]
        assert selected_quote[1:] == (
            (trigger_end - timedelta(seconds=1)).isoformat(),
            1.0,
            "latest_valid_at_or_before_checkpoint_boundary_v0",
        )
        quiet_universe = ProspectiveReadStore(
            config.paths.database,
            run_id=config.runtime.run_id,
        ).quiet_state_universe_v0()
        hims = next(item for item in quiet_universe if item["symbol"] == "HIMS")
        assert hims["selected_underlying_quote_event_id"] == selected_quote[0]
        assert hims["selected_underlying_quote_timestamp_utc"] == selected_quote[1]
        assert hims["selected_underlying_quote_age_seconds"] == 1.0
        assert hims["underlying_quote_selection_policy"] == selected_quote[3]
    finally:
        application.shutdown(now=observed + timedelta(seconds=1))


def test_quiet_checkpoint_without_quote_inside_contract_window_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, observed, trigger_end = _delayed_quiet_checkpoint_events(
        boundary_quote_offset_seconds=-3.0,
    )
    application, config, adapter, symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        complete_activity_baseline=True,
        bar_compatibility_passed=True,
        events=events,
        run_id="quiet-stale-boundary-v0",
    )
    _write_valid_group_o_package(
        config=config,
        symbols=symbols,
        signal_session=observed,
    )
    _install_independent_bar_timestamps(adapter, monkeypatch)
    try:
        results = tuple(
            application.poll(now=observed + timedelta(seconds=offset)) for offset in range(3)
        )
        checkpoint = next(
            item
            for result in results
            for item in result.checkpoint_results
            if item.episode_decision.symbol == "HIMS" and item.episode_decision.checkpoint == 6
        )
        assert "underlying_quote_stale" in checkpoint.rejection_reasons
        with sqlite3.connect(config.paths.database) as connection:
            quiet_checkpoint = connection.execute(
                """
                SELECT eligible, data_quality_flags_json,
                       selected_underlying_quote_age_seconds
                FROM quiet_state_checkpoint_v0
                WHERE run_id = ? AND symbol = 'HIMS' AND checkpoint = 6
                """,
                (config.runtime.run_id,),
            ).fetchone()
            observation_count = connection.execute(
                """
                SELECT COUNT(*) FROM quiet_state_observation_v0
                WHERE run_id = ? AND symbol = 'HIMS'
                """,
                (config.runtime.run_id,),
            ).fetchone()
        assert quiet_checkpoint == (
            0,
            '["underlying_quote_stale"]',
            3.0,
        )
        assert observation_count == (0,)
    finally:
        application.shutdown(now=observed + timedelta(seconds=3))


@pytest.mark.parametrize(
    ("case", "boundary_bid", "boundary_ask", "market_data_type", "expected_reason"),
    [
        ("invalid", 50.2, 50.1, "live", "underlying_quote_stale"),
        ("non_primary", 50.0, 50.02, "delayed", "market_data_not_live"),
    ],
)
def test_invalid_or_non_primary_boundary_quote_is_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    boundary_bid: float,
    boundary_ask: float,
    market_data_type: str,
    expected_reason: str,
) -> None:
    events, observed, trigger_end = _delayed_quiet_checkpoint_events(
        boundary_bid=boundary_bid,
        boundary_ask=boundary_ask,
        boundary_market_data_type=market_data_type,
    )
    application, config, adapter, symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        complete_activity_baseline=True,
        bar_compatibility_passed=True,
        events=events,
        run_id=f"quiet-{case}-boundary-v0",
    )
    _write_valid_group_o_package(
        config=config,
        symbols=symbols,
        signal_session=observed,
    )
    _install_independent_bar_timestamps(adapter, monkeypatch)
    try:
        results = tuple(
            application.poll(now=observed + timedelta(seconds=offset)) for offset in range(3)
        )
        checkpoint = next(
            item
            for result in results
            for item in result.checkpoint_results
            if item.episode_decision.symbol == "HIMS" and item.episode_decision.checkpoint == 6
        )
        assert expected_reason in checkpoint.rejection_reasons
        with sqlite3.connect(config.paths.database) as connection:
            quiet_checkpoint = connection.execute(
                """
                SELECT eligible FROM quiet_state_checkpoint_v0
                WHERE run_id = ? AND symbol = 'HIMS' AND checkpoint = 6
                """,
                (config.runtime.run_id,),
            ).fetchone()
            observation_count = connection.execute(
                """
                SELECT COUNT(*) FROM quiet_state_observation_v0
                WHERE run_id = ? AND symbol = 'HIMS'
                """,
                (config.runtime.run_id,),
            ).fetchone()
        assert quiet_checkpoint == (0,)
        assert observation_count == (0,)
    finally:
        application.shutdown(now=observed + timedelta(seconds=3))


def test_valid_bottom_10_crossing_begins_quiet_shadow_option_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, observed, trigger_end = _delayed_quiet_checkpoint_events(
        boundary_bid=100.0,
        boundary_ask=100.02,
        processing_bid=100.1,
        processing_ask=100.12,
    )
    application, config, adapter, symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        complete_activity_baseline=True,
        bar_compatibility_passed=True,
        events=events,
        run_id="quiet-valid-bottom-10-crossing-v0",
        include_opening_reversal=True,
    )
    _write_valid_group_o_package(
        config=config,
        symbols=symbols,
        signal_session=observed,
    )
    _install_independent_bar_timestamps(adapter, monkeypatch)

    def current_option_chain_metadata(**kwargs: object) -> FakeRequestResult:
        symbol = str(kwargs["underlying_symbol"])
        con_id = int(str(kwargs["underlying_contract_id"]))
        expiries = {
            (trigger_end.date() + timedelta(days=days)).strftime("%Y%m%d") for days in (1, 3)
        }
        return FakeRequestResult(
            adapter.request_ids.next(),
            (
                SimpleNamespace(
                    underlyingConId=con_id,
                    exchange="SMART",
                    tradingClass=symbol,
                    expirations=expiries,
                    strikes={float(value) for value in range(90, 111)},
                ),
            ),
        )

    monkeypatch.setattr(
        adapter,
        "request_option_chain_metadata",
        current_option_chain_metadata,
    )
    runtime = application.live_recorder.engine.m1c_runtime
    original_score = runtime.score

    def score_bottom_10(**kwargs: object):
        score = original_score(**kwargs)
        if kwargs["symbol"] != "HIMS":
            return score
        probability = 0.13
        assert probability < M1C_BOTTOM_10_THRESHOLD
        return score.model_copy(
            update={
                "probability": probability,
                "threshold_passed": probability >= score.threshold,
            }
        )

    monkeypatch.setattr(runtime, "score", score_bottom_10)

    try:
        results = tuple(
            application.poll(now=observed + timedelta(seconds=offset)) for offset in range(3)
        )
        checkpoint = next(
            item
            for result in results
            for item in result.checkpoint_results
            if item.episode_decision.symbol == "HIMS" and item.episode_decision.checkpoint == 6
        )
        observation_id = checkpoint.quiet_observation_id
        assert observation_id is not None
        assert checkpoint.opening_reversal_prediction_v1 is not None
        assert checkpoint.opening_reversal_prediction_v1.eligibility_v1 is False
        assert checkpoint.opening_reversal_prediction_v1.prediction_v1 == "ABSTAIN"
        assert checkpoint.quiet_state.bottom_10 is True
        assert checkpoint.quiet_episode_decision.fresh_episode is True
        assert checkpoint.diagnostic_quality_flags == ()
        assert application.option_discovery.pending_symbol(observation_id) == "HIMS"
        assert (
            application.live_recorder.episode_window_completed(
                observation_id,
                "T_to_entry",
            )
            is False
        )

        with sqlite3.connect(config.paths.database) as connection:
            observation = connection.execute(
                """
                SELECT observation_kind, completion_status
                FROM quiet_state_observation_v0
                WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
            schedule = connection.execute(
                """
                SELECT symbol, episode_kind, quiet_state, status, degradation_reason
                FROM option_episode_schedule_v0
                WHERE episode_id = ?
                """,
                (observation_id,),
            ).fetchone()
            option_contract_count = connection.execute(
                """
                SELECT COUNT(*) FROM quiet_state_option_contract_v0
                WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
        assert observation == ("quiet_bottom_10", "active")
        assert schedule == ("HIMS", "quiet", 1, "streaming", None), (
            application.option_discovery.rejections
        )
        assert option_contract_count is not None and option_contract_count[0] > 0
        assert application.subscriptions.budget.snapshot()["active"]["option"] > 0  # type: ignore[index,operator]
    finally:
        application.shutdown(now=observed + timedelta(seconds=3))


def test_ineligible_opening_reversal_receipt_does_not_suppress_generic_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, observed, trigger_end = _delayed_quiet_checkpoint_events()
    application, config, adapter, symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        complete_activity_baseline=True,
        bar_compatibility_passed=True,
        events=events,
        run_id="ineligible-opening-generic-capture-v0",
        include_opening_reversal=True,
    )
    _write_valid_group_o_package(
        config=config,
        symbols=symbols,
        signal_session=observed,
    )
    _install_independent_bar_timestamps(adapter, monkeypatch)

    def current_option_chain_metadata(**kwargs: object) -> FakeRequestResult:
        symbol = str(kwargs["underlying_symbol"])
        con_id = int(str(kwargs["underlying_contract_id"]))
        expiries = {
            (trigger_end.date() + timedelta(days=days)).strftime("%Y%m%d") for days in (1, 3)
        }
        return FakeRequestResult(
            adapter.request_ids.next(),
            (
                SimpleNamespace(
                    underlyingConId=con_id,
                    exchange="SMART",
                    tradingClass=symbol,
                    expirations=expiries,
                    strikes={float(value) for value in range(90, 111)},
                ),
            ),
        )

    monkeypatch.setattr(
        adapter,
        "request_option_chain_metadata",
        current_option_chain_metadata,
    )
    runtime = application.live_recorder.engine.m1c_runtime
    original_score = runtime.score

    def score_high_tail(**kwargs: object):
        score = original_score(**kwargs)
        if kwargs["symbol"] != "HIMS":
            return score
        probability = 0.9
        return score.model_copy(
            update={
                "probability": probability,
                "threshold_passed": probability >= score.threshold,
            }
        )

    monkeypatch.setattr(runtime, "score", score_high_tail)

    try:
        results = tuple(
            application.poll(now=observed + timedelta(seconds=offset)) for offset in range(3)
        )
        checkpoint = next(
            item
            for result in results
            for item in result.checkpoint_results
            if item.episode_decision.symbol == "HIMS" and item.episode_decision.checkpoint == 6
        )
        opening_receipt = checkpoint.opening_reversal_prediction_v1
        assert opening_receipt is not None
        assert opening_receipt.eligibility_v1 is False
        assert opening_receipt.prediction_v1 == "ABSTAIN"
        assert checkpoint.episode_decision.fresh_episode is True
        episode_id = checkpoint.episode_decision.episode_id
        assert episode_id is not None
        assert application.option_discovery.pending_symbol(episode_id) == "HIMS"
        with sqlite3.connect(config.paths.database) as connection:
            schedule = connection.execute(
                """
                SELECT symbol, episode_kind, status
                FROM option_episode_schedule_v0
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
        assert schedule == ("HIMS", "high_tail", "streaming")
    finally:
        application.shutdown(now=observed + timedelta(seconds=3))


def test_missing_bar_compatibility_report_is_diagnostic_not_scientific_gate(
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
    assert application.live_recorder.scientific_scoring_enabled is True

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
    assert config.runtime.callback_inbox_batch_limit == 4_096
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
    active = tuple(adapter.active_subscriptions.values())
    assert sum(kind == "bar" for kind, _symbol in active) == len(recorded_symbols)
    assert sum(kind == "level1" for kind, _symbol in active) == len(symbols) + 1
    assert {symbol for kind, symbol in active if kind == "level1"} == {*symbols, "VTI"}
    assert {kind for kind, _symbol in active} == {"bar", "level1"}
    assert (tmp_path / "ibkr_runtime_capacity_manifest.json").is_file()
    assert (
        json.loads((tmp_path / "capability.json").read_text(encoding="utf-8"))[
            "scientific_recording_valid"
        ]
        is False
    )

    application.shutdown(now=datetime.now(UTC))
    assert adapter.active_subscriptions == {}


def test_restart_after_session_report_does_not_rewrite_immutable_report(
    tmp_path: Path,
) -> None:
    report_due_at = datetime(2026, 7, 29, 20, 31, tzinfo=UTC)
    first_application, config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        events=(),
    )
    first_application.poll(now=report_due_at)
    first_application.shutdown(now=report_due_at + timedelta(seconds=1))
    with sqlite3.connect(config.paths.database) as connection:
        report_before_restart = connection.execute(
            """
            SELECT report_json
            FROM recorder_session_report_v0
            WHERE run_id = ? AND session_date = '2026-07-29'
            """,
            (config.runtime.run_id,),
        ).fetchone()
    assert report_before_restart is not None

    restarted_application, _config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        events=(),
    )
    restarted_application.poll(now=report_due_at + timedelta(minutes=1))
    restarted_application.shutdown(now=report_due_at + timedelta(minutes=1, seconds=1))

    with sqlite3.connect(config.paths.database) as connection:
        reports_after_restart = connection.execute(
            """
            SELECT report_json
            FROM recorder_session_report_v0
            WHERE run_id = ? AND session_date = '2026-07-29'
            """,
            (config.runtime.run_id,),
        ).fetchall()
    assert reports_after_restart == [report_before_restart]


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
        recorder_generation=1,
    )
    activation_before = (tmp_path / "activation.json").read_bytes()
    first_application.shutdown(now=datetime.now(UTC))

    second_application, second_config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        git_commit="b" * 40,
        app_version="0.2.0",
        recorder_generation=2,
    )

    assert second_config.runtime.git_commit == "b" * 40
    assert second_config.runtime.app_version == "0.2.0"
    assert (tmp_path / "activation.json").read_bytes() == activation_before
    with sqlite3.connect(config.paths.database) as connection:
        run_identity = connection.execute(
            "SELECT git_commit, app_version FROM prospective_run WHERE run_id = ?",
            (config.runtime.run_id,),
        ).fetchone()
        generation_receipt = connection.execute(
            """
            SELECT details_json
            FROM runtime_artifact_verification_v1
            WHERE run_id = ? AND recorder_generation = 2
            ORDER BY artifact_name
            LIMIT 1
            """,
            (config.runtime.run_id,),
        ).fetchone()
    assert run_identity == ("a" * 40, "0.1.0")
    assert generation_receipt is not None
    receipt_details = json.loads(generation_receipt[0])
    assert receipt_details["activation_app_version"] == "0.1.0"
    assert receipt_details["runtime_app_version"] == "0.2.0"

    second_application.shutdown(now=datetime.now(UTC))


def test_static_artifact_verification_precedes_live_subscription_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    original_record = RecorderOperationalRepository.record_artifact_verification
    original_start = LiveSubscriptionController.start_always_on

    def record_artifact_verification(
        repository: RecorderOperationalRepository,
        verification: object,
    ) -> None:
        call_order.append("artifact")
        original_record(repository, verification)  # type: ignore[arg-type]

    def start_always_on(
        controller: LiveSubscriptionController,
        *args: object,
        **kwargs: object,
    ) -> None:
        call_order.append("subscriptions")
        assert call_order[:-1]
        assert set(call_order[:-1]) == {"artifact"}
        original_start(controller, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        RecorderOperationalRepository,
        "record_artifact_verification",
        record_artifact_verification,
    )
    monkeypatch.setattr(
        LiveSubscriptionController,
        "start_always_on",
        start_always_on,
    )

    application, _config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        recorder_generation=1,
    )

    assert call_order[-1] == "subscriptions"
    assert call_order.count("artifact") > 0
    application.shutdown(now=datetime.now(UTC))


def test_clock_probe_precedes_high_volume_live_subscription_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    original_probe = FakeIBKRAdapter.request_current_time
    original_start = LiveSubscriptionController.start_always_on

    def request_current_time(adapter: FakeIBKRAdapter) -> int:
        call_order.append("clock")
        return original_probe(adapter)

    def start_always_on(
        controller: LiveSubscriptionController,
        *args: object,
        **kwargs: object,
    ) -> None:
        call_order.append("subscriptions")
        original_start(controller, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(FakeIBKRAdapter, "request_current_time", request_current_time)
    monkeypatch.setattr(
        LiveSubscriptionController,
        "start_always_on",
        start_always_on,
    )

    application, _config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        recorder_generation=1,
    )

    assert call_order.index("clock") < call_order.index("subscriptions")
    application.shutdown(now=datetime.now(UTC))


def test_application_rebuilds_actual_subscriptions_after_not_connected_loss(
    tmp_path: Path,
) -> None:
    application, _config, adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        events=(),
        run_id="not-connected-subscription-rebuild-v0",
    )
    original_ids = set(adapter.active_subscriptions)
    original_streams = sorted(adapter.active_subscriptions.values())
    assert original_ids

    adapter.connection.connection_lost(code=504, message="Not connected")
    adapter.active_subscriptions.clear()
    recovered_at = datetime.now(UTC)

    application._recover_if_required(recovered_at)

    health = adapter.connection.health()
    assert health.state is ConnectionState.CONNECTED
    assert health.subscriptions_require_rebuild is False
    assert sorted(adapter.active_subscriptions.values()) == original_streams
    assert set(adapter.active_subscriptions).isdisjoint(original_ids)
    application.shutdown(now=recovered_at + timedelta(seconds=1))


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
        "callback_inbox_retention_enabled",
        "callback_inbox_retention_seconds",
        "callback_inbox_compaction_interval_seconds",
        "callback_inbox_compaction_batch_limit",
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
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "git_commit": "b" * 40,
                    "callback_inbox_retention_enabled": True,
                }
            )
        }
    )

    _require_compatible_existing_activation(
        activation=legacy_activation,
        config=upgraded_config,
        artifact_hashes=legacy_activation.model_artifact_hashes,
        ibkr_api_version=legacy_activation.ibkr_api_version,
        tws_or_gateway_version=legacy_activation.tws_or_gateway_version,
    )


def test_existing_activation_accepts_only_exact_superseded_diagnostic_claims(
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

    legacy_claims = dict(activation.claims_boundary)
    for field_name in (
        "market_data_source",
        "historical_research_source",
        "cross_vendor_validation_diagnostic_only",
        "cross_vendor_validation_required_for_science",
        "prospective_evidence_description",
    ):
        legacy_claims.pop(field_name)
    legacy_claims["engineering_phase_sessions"] = legacy_claims.pop(
        "historical_engineering_phase_sessions"
    )
    legacy_activation = activation.model_copy(
        update={
            "claims_boundary": legacy_claims,
            "configuration_hash": _configuration_hash(
                config,
                git_commit=activation.git_sha,
                web_projection_cache_seconds=60.0,
            ),
        }
    )
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

    unsafe_claims = dict(legacy_claims)
    unsafe_claims["paper_orders_allowed"] = True
    with pytest.raises(ValueError, match="blocked_existing_activation_claims_boundary_mismatch"):
        _require_compatible_existing_activation(
            activation=legacy_activation.model_copy(update={"claims_boundary": unsafe_claims}),
            config=upgraded_config,
            artifact_hashes=legacy_activation.model_artifact_hashes,
            ibkr_api_version=legacy_activation.ibkr_api_version,
            tws_or_gateway_version=legacy_activation.tws_or_gateway_version,
        )


def test_existing_activation_accepts_verified_api_and_gateway_maintenance(
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
        "option_commission_per_contract",
        "option_regulatory_fee_per_contract",
        "option_exchange_fee_per_contract",
    ):
        legacy_payload["ibkr"].pop(field_name)
    legacy_hash = hashlib.sha256(
        json.dumps(
            legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    legacy_activation = activation.model_copy(update={"configuration_hash": legacy_hash})
    maintained_config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"git_commit": "b" * 40}),
            "ibkr": config.ibkr.model_copy(
                update={"tws_or_gateway_version": "10491c-latest-ae1600e7c3a1"}
            ),
        }
    )

    _require_compatible_existing_activation(
        activation=legacy_activation,
        config=maintained_config,
        artifact_hashes=legacy_activation.model_artifact_hashes,
        ibkr_api_version="10.49.1",
        tws_or_gateway_version="10491c-latest-ae1600e7c3a1",
    )
    assert legacy_activation.ibkr_api_version != "10.49.1"
    assert legacy_activation.tws_or_gateway_version != "10491c-latest-ae1600e7c3a1"


def test_existing_activation_accepts_disabling_optional_cross_vendor_diagnostic(
    tmp_path: Path,
) -> None:
    application, config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
    )
    activation = ActivationRecord.model_validate_json(
        (tmp_path / "activation.json").read_text(encoding="utf-8")
    )
    application.shutdown(now=datetime.now(UTC))
    diagnostic_enabled_config = config.model_copy(
        update={
            "parallel_validation": config.parallel_validation.model_copy(update={"enabled": True})
        }
    )
    legacy_activation = activation.model_copy(
        update={
            "configuration_hash": _configuration_hash(
                diagnostic_enabled_config,
                git_commit=activation.git_sha,
            )
        }
    )

    _require_compatible_existing_activation(
        activation=legacy_activation,
        config=config,
        artifact_hashes=legacy_activation.model_artifact_hashes,
        ibkr_api_version=legacy_activation.ibkr_api_version,
        tws_or_gateway_version=legacy_activation.tws_or_gateway_version,
    )


def test_existing_activation_accepts_legacy_bounded_inbox_batch_capacity(
    tmp_path: Path,
) -> None:
    application, config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
    )
    activation = ActivationRecord.model_validate_json(
        (tmp_path / "activation.json").read_text(encoding="utf-8")
    )
    application.shutdown(now=datetime.now(UTC))
    legacy_config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"callback_inbox_batch_limit": 256})}
    )
    legacy_activation = activation.model_copy(
        update={
            "configuration_hash": _configuration_hash(
                legacy_config,
                git_commit=activation.git_sha,
            )
        }
    )

    _require_compatible_existing_activation(
        activation=legacy_activation,
        config=config,
        artifact_hashes=legacy_activation.model_artifact_hashes,
        ibkr_api_version=legacy_activation.ibkr_api_version,
        tws_or_gateway_version=legacy_activation.tws_or_gateway_version,
    )


def test_existing_activation_accepts_legacy_batch_before_retention_controls(
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

    historical_run_id = "historical-retention-upgrade-v0"
    legacy_payload = config.model_dump(mode="json")
    legacy_payload["runtime"]["run_id"] = historical_run_id
    legacy_payload["runtime"]["callback_inbox_batch_limit"] = 256
    for field_name in (
        "callback_inbox_retention_enabled",
        "callback_inbox_retention_seconds",
        "callback_inbox_compaction_interval_seconds",
        "callback_inbox_compaction_batch_limit",
    ):
        legacy_payload["runtime"].pop(field_name)
    legacy_payload["web"]["operational_projection_cache_seconds"] = 60.0
    legacy_payload["parallel_validation"]["enabled"] = True
    for field_name in (
        "option_commission_per_contract",
        "option_regulatory_fee_per_contract",
        "option_exchange_fee_per_contract",
    ):
        legacy_payload["ibkr"].pop(field_name)
    legacy_hash = hashlib.sha256(
        json.dumps(
            legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    legacy_activation = activation.model_copy(update={"configuration_hash": legacy_hash})
    upgraded_config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "git_commit": "b" * 40,
                    "run_id": "replacement-retention-upgrade-v0",
                    "callback_inbox_retention_enabled": True,
                }
            )
        }
    )

    _require_compatible_existing_activation(
        activation=legacy_activation,
        config=upgraded_config,
        artifact_hashes=legacy_activation.model_artifact_hashes,
        ibkr_api_version=legacy_activation.ibkr_api_version,
        tws_or_gateway_version=legacy_activation.tws_or_gateway_version,
        historical_run_ids=(historical_run_id,),
    )


def test_fatal_run_rollover_reuses_activation_only_via_persisted_run_identity(
    tmp_path: Path,
) -> None:
    first_application, first_config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        recorder_generation=1,
        run_id="failed-run-v0",
        include_opening_reversal=True,
    )
    activation_before = (tmp_path / "activation.json").read_bytes()
    activation = ActivationRecord.model_validate_json(activation_before)
    first_application.shutdown(now=datetime.now(UTC))

    replacement_candidate = first_config.model_copy(
        update={
            "runtime": first_config.runtime.model_copy(
                update={"run_id": "isolated-replacement-run-v0"}
            )
        }
    )
    with pytest.raises(ValueError, match="blocked_existing_activation_configuration_mismatch"):
        _require_compatible_existing_activation(
            activation=activation,
            config=replacement_candidate,
            artifact_hashes=activation.model_artifact_hashes,
            ibkr_api_version=activation.ibkr_api_version,
            tws_or_gateway_version=activation.tws_or_gateway_version,
        )
    drifted_candidate = replacement_candidate.model_copy(
        update={
            "ibkr": replacement_candidate.ibkr.model_copy(
                update={
                    "max_option_subscriptions": (
                        replacement_candidate.ibkr.max_option_subscriptions + 1
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="blocked_existing_activation_configuration_mismatch"):
        _require_compatible_existing_activation(
            activation=activation,
            config=drifted_candidate,
            artifact_hashes=activation.model_artifact_hashes,
            ibkr_api_version=activation.ibkr_api_version,
            tws_or_gateway_version=activation.tws_or_gateway_version,
            historical_run_ids=("failed-run-v0",),
        )

    replacement_application, replacement_config, _adapter, _symbols = _build_fake_application(
        tmp_path,
        include_scientific_prerequisites=True,
        git_commit="b" * 40,
        app_version="0.2.0",
        recorder_generation=1,
        run_id="isolated-replacement-run-v0",
        include_opening_reversal=True,
    )

    assert replacement_config.runtime.run_id == "isolated-replacement-run-v0"
    assert replacement_config.runtime.git_commit == "b" * 40
    assert replacement_config.runtime.app_version == "0.2.0"
    assert (tmp_path / "activation.json").read_bytes() == activation_before
    with sqlite3.connect(first_config.paths.database) as connection:
        run_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT run_id FROM prospective_run ORDER BY run_id"
            ).fetchall()
        }
        replacement_identity = connection.execute(
            """
            SELECT git_commit, app_version
            FROM prospective_run
            WHERE run_id = 'isolated-replacement-run-v0'
            """
        ).fetchone()
        activation_rows = connection.execute(
            """
            SELECT id, run_id, experiment_version, activation_receipt_hash,
                   receipt_json, source_activation_id, binding_kind
            FROM opening_reversal_activation_v1
            ORDER BY experiment_version, run_id
            """
        ).fetchall()
        migration_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM schema_migrations
            WHERE version = '0021_opening_reversal_activation_run_binding_v1.sql'
            """
        ).fetchone()
        capture_view_sql = connection.execute(
            """
            SELECT sql
            FROM sqlite_schema
            WHERE type = 'view'
              AND name = 'opening_reversal_v1_1_capture_eligible_episode'
            """
        ).fetchone()
    assert run_ids == {"failed-run-v0", "isolated-replacement-run-v0"}
    assert replacement_identity == ("a" * 40, "0.1.0")
    assert migration_count == (1,)
    assert capture_view_sql is not None
    assert "JOIN opening_reversal_activation_v1 AS activation" in str(capture_view_sql[0])
    assert len(activation_rows) == 4
    for experiment_version in ("1", "1.1"):
        version_rows = [row for row in activation_rows if row[2] == experiment_version]
        assert len(version_rows) == 2
        original = next(row for row in version_rows if row[1] == "failed-run-v0")
        replacement = next(row for row in version_rows if row[1] == "isolated-replacement-run-v0")
        assert original[5:] == (None, "original_activation")
        assert replacement[5:] == (original[0], "audited_run_rollover")
        assert replacement[3:5] == original[3:5]

    with sqlite3.connect(first_config.paths.database) as connection:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="opening reversal activation binding is immutable",
        ):
            connection.execute(
                """
                UPDATE opening_reversal_activation_v1
                SET receipt_json = receipt_json
                WHERE run_id = 'isolated-replacement-run-v0'
                """
            )
        connection.rollback()
        with pytest.raises(
            sqlite3.IntegrityError,
            match="opening reversal activation binding is append-only",
        ):
            connection.execute(
                """
                DELETE FROM opening_reversal_activation_v1
                WHERE run_id = 'isolated-replacement-run-v0'
                """
            )

    replacement_application.shutdown(now=datetime.now(UTC))


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
    assert {kind for kind, _symbol in adapter.active_subscriptions.values()} == {
        "bar",
        "level1",
    }
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
