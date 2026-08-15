from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from stocker_prospective.capacity import (
    CapacityDiscovery,
    RuntimeCapacitySettings,
    resolve_runtime_capacity,
)
from stocker_prospective.config import ProspectiveConfig, ShadowMicrostructureConfig
from stocker_prospective.events import UnderlyingLevel1QuoteEvent
from stocker_prospective.live_subscriptions import QualifiedUnderlying
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.shadow_microstructure_v0 import (
    FORBIDDEN_RAW_RESEARCH_FIELDS,
    ShadowMicrostructureCollectorV0,
    assess_shadow_capacity,
    evaluate_pilot_readiness,
)

ACTIVATION = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
SYMBOLS = (
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


def _contracts() -> tuple[QualifiedUnderlying, ...]:
    return tuple(
        QualifiedUnderlying(
            symbol=symbol,
            con_id=10_000 + index,
            upstream_contract=object(),
            exchange="SMART",
            minimum_tick=0.01,
        )
        for index, symbol in enumerate(SYMBOLS, start=1)
    )


def _capacity(*, total: int = 100):
    return resolve_runtime_capacity(
        settings=RuntimeCapacitySettings(
            configured_total_market_data_lines=total,
            configured_externally_reserved_lines=0,
            reserved_future_trading_lines=12,
            safety_margin_lines=2,
            configured_max_tick_by_tick=2,
            configured_max_depth=0,
        ),
        discovery=CapacityDiscovery(market_data_status="live"),
        environment={},
        observed_at=ACTIVATION,
    )


def _config(*, contract: Path, pilot: bool = True) -> ShadowMicrostructureConfig:
    return ShadowMicrostructureConfig(
        enabled=True,
        pilot_mode=pilot,
        activation_timestamp_utc=ACTIVATION,
        research_contract=contract,
        pilot_readiness_report=None,
    )


def _collector(tmp_path: Path) -> ShadowMicrostructureCollectorV0:
    contract = Path(
        "configs/prospective/shadow-microstructure-v0/shadow_research_contract_v0.json"
    ).resolve()
    return ShadowMicrostructureCollectorV0(
        root=tmp_path,
        config=_config(contract=contract),
        run_id="shadow-pilot-v0",
        git_commit="abcdef1",
        universe_hash="a" * 64,
        m1c_artifact_hash="b" * 64,
        m1c_configuration_hash="c" * 64,
        contracts=_contracts(),
        capacity=_capacity(),
        always_on_bar_lines=28,
        recorder_version="0.1.0",
    )


def _quote(sequence: int, *, field: str, size_delta: float = 0.0):
    received = datetime(2026, 8, 17, 13, 31, tzinfo=UTC)
    return UnderlyingLevel1QuoteEvent(
        event_id=f"source-{sequence}",
        received_timestamp_utc=received,
        received_monotonic_ns=10_000 + sequence,
        provider_timestamp_utc=None,
        source_sequence=sequence,
        session=received.date(),
        symbol="AAL",
        con_id=10_001,
        request_id=100,
        bid=10.0,
        bid_size=100.0 + size_delta,
        ask=10.01,
        ask_size=90.0,
        last=None,
        last_size=None,
        market_data_type=MarketDataType.LIVE,
        source="official_ibkr_tws_socket_api",
        quote_valid=True,
        tick_type=field,
        exchange="SMART",
        connection_generation=3,
    )


def _quote_for_symbol(
    symbol: str,
    symbol_index: int,
    sequence: int,
    *,
    field: str,
    size_delta: float = 0.0,
) -> UnderlyingLevel1QuoteEvent:
    return _quote(sequence, field=field, size_delta=size_delta).model_copy(
        update={
            "event_id": f"source-{symbol}-{sequence}",
            "symbol": symbol,
            "con_id": 10_000 + symbol_index,
        }
    )


def test_shadow_mode_is_disabled_and_analysis_is_fail_closed_by_default(tmp_path: Path) -> None:
    payload = {
        "paths": {
            "database": str(tmp_path / "db.sqlite3"),
            "bundle_root": str(tmp_path / "bundles"),
            "feature_parity_report": str(tmp_path / "parity.json"),
        },
        "runtime": {
            "mode": "shadow",
            "source": "replay",
            "prospective_start_utc": ACTIVATION.isoformat(),
            "instance_id": "test",
            "app_version": "0.1.0",
            "git_commit": "abcdef1",
        },
        "context": {"mode": "signed_import", "hmac_secret_env": "TEST_SECRET"},
    }
    config = ProspectiveConfig.model_validate(payload)
    assert config.shadow_microstructure.enabled is False
    assert config.shadow_microstructure.analysis_enabled is False
    assert config.shadow_microstructure.direction_scoring_enabled is False
    assert config.shadow_microstructure.collect_depth is False

    with pytest.raises(ValidationError):
        ShadowMicrostructureConfig.model_validate(
            {
                "enabled": True,
                "activation_timestamp_utc": ACTIVATION.isoformat(),
                "research_contract": "contract.json",
                "analysis_enabled": True,
            }
        )


def test_capacity_reuses_all_universe_level1_and_fails_closed() -> None:
    allowed = assess_shadow_capacity(_capacity(), universe_size=20, always_on_bar_lines=28)
    assert allowed.frozen_universe_level1_lines == 20
    assert allowed.incremental_shadow_lines == 0
    assert allowed.required_research_lines == 49
    assert allowed.available_research_lines == 86
    assert allowed.allowed is True

    blocked = assess_shadow_capacity(_capacity(total=60), universe_size=20, always_on_bar_lines=28)
    assert blocked.allowed is False
    assert blocked.blocker == "BLOCK_SHADOW_ALL_UNIVERSE_CAPACITY"


def test_every_same_timestamp_bbo_callback_is_retained_in_deterministic_order(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path)
    collector.record_subscriptions_active(
        symbols=SYMBOLS,
        observed_at=ACTIVATION,
        connection_generation=3,
    )
    events = tuple(
        _quote(sequence, field=field, size_delta=float(sequence))
        for sequence, field in enumerate(
            ("bid", "ask", "bid_size", "ask_size", "bid_size"), start=1
        )
    )
    collector.persist_raw_events(events)

    files = sorted(
        collector.root.glob(
            "data_source=ibkr/session_date=*/symbol=AAL/"
            "event_type=underlying_bbo_update/hour=*/*.parquet"
        )
    )
    assert len(files) == 1
    rows = pq.ParquetFile(files[0]).read().to_pylist()
    assert [row["source_sequence"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["callback_arrival_sequence"] for row in rows] == [1, 2, 3, 4, 5]
    assert len({row["event_id"] for row in rows}) == 5
    assert all(row["pipeline_pilot"] for row in rows)
    assert not any(row["confirmatory_eligible"] for row in rows)
    assert all(row["connection_generation"] == 3 for row in rows)
    assert all(row["provider_timestamp_utc"] is None for row in rows)
    assert not FORBIDDEN_RAW_RESEARCH_FIELDS.intersection(rows[0])
    assert not {
        "future_return",
        "next_price_direction",
        "ofi",
        "imbalance",
        "weighted_mid",
    }.intersection(rows[0])


def test_bbo_capture_needs_no_episode_or_optional_feed_and_reports_fairness(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path)
    collector.record_subscriptions_active(
        symbols=SYMBOLS,
        observed_at=ACTIVATION,
        connection_generation=3,
    )
    collector.persist_raw_events(
        tuple(
            _quote(sequence, field=field)
            for sequence, field in enumerate(
                ("bid", "ask", "bid_size", "ask_size"), start=1
            )
        )
    )

    quality_path = collector.root / "collection_quality_summary.csv"
    with quality_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    aal = next(row for row in rows if row["symbol"] == "AAL")
    assert aal["expected_subscription_active"] == "true"
    assert aal["bbo_update_count"] == "4"
    assert aal["optional_trade_stream"] == "BBO_ONLY"
    assert aal["number_universe_stocks"] == "20"
    assert aal["stocks_with_valid_bbo_collection"] == "1"
    assert aal["stocks_missing_bbo_collection"] == "19"
    assert aal["percent_universe_coverage"] == "5.0"


def test_reconnect_gaps_are_explicit_and_never_synthesize_quotes(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    lost = datetime(2026, 8, 17, 13, 35, tzinfo=UTC)
    restored = datetime(2026, 8, 17, 13, 36, tzinfo=UTC)
    collector.record_connection_state(
        state="disconnected",
        observed_at=lost,
        connection_generation=4,
        reason="connectivity_lost",
    )
    collector.record_connection_state(
        state="connected",
        observed_at=restored,
        connection_generation=5,
        reason="connectivity_restored_data_lost",
    )
    collector.record_subscriptions_rebuilt(
        observed_at=restored,
        connection_generation=5,
    )

    root = collector.root
    gap_files = sorted(root.glob("**/event_type=collection_gap_event/**/*.parquet"))
    bbo_files = sorted(root.glob("**/event_type=underlying_bbo_update/**/*.parquet"))
    assert gap_files
    assert bbo_files == []
    rows = [row for path in gap_files for row in pq.ParquetFile(path).read().to_pylist()]
    kinds = {row["gap_kind"] for row in rows}
    assert kinds == {"connection_lost", "connection_restored", "subscription_rebuilt"}
    restored_rows = [row for row in rows if row["gap_kind"] == "connection_restored"]
    assert all(row["gap_start_timestamp_utc"] is not None for row in restored_rows)
    assert all(row["gap_end_timestamp_utc"] is not None for row in restored_rows)


def test_runtime_manifest_freezes_universe_conids_and_collection_contract(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path)
    root = collector.root
    manifest = json.loads((root / "shadow_collection_manifest.json").read_text())
    universe = json.loads((root / "universe_manifest.json").read_text())
    assert manifest["planned_eligible_sessions"] == 20
    assert manifest["frozen_universe_hash"] == "a" * 64
    assert manifest["analysis_enabled"] is False
    assert manifest["execution_enabled"] is False
    assert manifest["capacity_assessment"]["incremental_shadow_lines"] == 0
    assert len(universe["symbols"]) == 20
    assert all(item["con_id"] > 0 for item in universe["symbols"])
    assert all(item["minimum_tick"] == 0.01 for item in universe["symbols"])
    assert manifest["symbols_and_con_ids"] == universe["symbols"]


def test_pre_activation_callbacks_are_ignored_and_pilot_namespace_is_isolated(
    tmp_path: Path,
) -> None:
    pilot = _collector(tmp_path)
    event = _quote(1, field="bid")
    before = event.model_copy(
        update={"received_timestamp_utc": datetime(2026, 8, 17, 11, 59, tzinfo=UTC)}
    )
    assert pilot.persist_raw_events((before,)) == ()
    assert "pipeline_pilot" in pilot.root.parts
    pilot.record_subscriptions_active(
        symbols=SYMBOLS,
        observed_at=before.received_timestamp_utc,
        connection_generation=3,
    )
    pilot.persist_raw_events((event,))

    readiness = tmp_path / "ready.json"
    readiness.write_text('{"classification":"READY_FOR_SHADOW_COLLECTION"}\n')
    contract = Path(
        "configs/prospective/shadow-microstructure-v0/shadow_research_contract_v0.json"
    ).resolve()
    with pytest.raises(RuntimeError, match="IDENTITY_MISMATCH"):
        ShadowMicrostructureCollectorV0(
            root=tmp_path,
            config=ShadowMicrostructureConfig(
                enabled=True,
                pilot_mode=False,
                activation_timestamp_utc=ACTIVATION,
                research_contract=contract,
                pilot_readiness_report=readiness,
            ),
            run_id="shadow-confirmatory-v0",
            git_commit="abcdef1",
            universe_hash="a" * 64,
            m1c_artifact_hash="b" * 64,
            m1c_configuration_hash="c" * 64,
            contracts=_contracts(),
            capacity=_capacity(),
            always_on_bar_lines=28,
            recorder_version="0.1.0",
        )

    events: list[UnderlyingLevel1QuoteEvent] = []
    sequence = 1
    for symbol_index, symbol in enumerate(SYMBOLS, start=1):
        for field, delta in (
            ("bid", 0.0),
            ("ask", 0.0),
            ("bid_size", 0.0),
            ("ask_size", 0.0),
            ("bid_size", 1.0),
        ):
            events.append(
                _quote_for_symbol(
                    symbol,
                    symbol_index,
                    sequence,
                    field=field,
                    size_delta=delta,
                )
            )
            sequence += 1
    pilot.persist_raw_events(tuple(events))
    lost = datetime(2026, 8, 17, 13, 35, tzinfo=UTC)
    restored = datetime(2026, 8, 17, 13, 36, tzinfo=UTC)
    pilot.record_connection_state(
        state="disconnected",
        observed_at=lost,
        connection_generation=3,
        reason="pilot_reconnect_test",
    )
    pilot.record_connection_state(
        state="connected",
        observed_at=restored,
        connection_generation=4,
        reason="pilot_reconnect_test",
    )
    pilot.record_subscriptions_rebuilt(
        observed_at=restored,
        connection_generation=4,
    )
    report = evaluate_pilot_readiness(
        dataset_root=pilot.root,
        universe_symbols=SYMBOLS,
        capacity_allowed=True,
        reconnect_validation_passed=True,
        safety_tests_passed=True,
        disk_capacity_acceptable=True,
        output_path=readiness,
    )
    assert report["classification"] == "READY_FOR_SHADOW_COLLECTION", report
    confirmatory = ShadowMicrostructureCollectorV0(
        root=tmp_path,
        config=ShadowMicrostructureConfig(
            enabled=True,
            pilot_mode=False,
            activation_timestamp_utc=ACTIVATION,
            research_contract=contract,
            pilot_readiness_report=readiness,
        ),
        run_id="shadow-confirmatory-v0",
        git_commit="abcdef1",
        universe_hash="a" * 64,
        m1c_artifact_hash="b" * 64,
        m1c_configuration_hash="c" * 64,
        contracts=_contracts(),
        capacity=_capacity(),
        always_on_bar_lines=28,
        recorder_version="0.1.0",
    )
    assert "confirmatory" in confirmatory.root.parts
    assert confirmatory.root != pilot.root


def test_reconnect_resets_quote_state_and_trade_availability_tracks_stream_lifecycle(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path)
    collector.set_optional_trade_stream(
        symbol="AAL",
        active=True,
        request_id=701,
        observed_at=datetime(2026, 8, 17, 13, 30, tzinfo=UTC),
        connection_generation=3,
    )
    collector.persist_raw_events(
        tuple(
            _quote(sequence, field=field)
            for sequence, field in enumerate(
                ("bid", "ask", "bid_size", "ask_size"), start=1
            )
        )
    )
    collector.record_connection_state(
        state="disconnected",
        observed_at=datetime(2026, 8, 17, 13, 35, tzinfo=UTC),
        connection_generation=3,
        reason="data_lost",
    )
    collector.record_connection_state(
        state="connected",
        observed_at=datetime(2026, 8, 17, 13, 36, tzinfo=UTC),
        connection_generation=4,
        reason="restored",
    )
    post_gap = _quote(5, field="bid").model_copy(update={"connection_generation": 4})
    collector.persist_raw_events((post_gap,))
    collector.set_optional_trade_stream(
        symbol="AAL",
        active=False,
        request_id=701,
        observed_at=datetime(2026, 8, 17, 13, 37, tzinfo=UTC),
        connection_generation=4,
    )
    collector.persist_raw_events(
        (_quote(6, field="ask").model_copy(update={"connection_generation": 4}),)
    )

    files = sorted(collector.root.glob("**/event_type=underlying_bbo_update/**/*.parquet"))
    rows = [row for path in files for row in pq.ParquetFile(path).read().to_pylist()]
    ordered = sorted(rows, key=lambda row: row["source_sequence"])
    assert ordered[3]["stream_availability"] == "BBO_PLUS_TRADES"
    assert ordered[4]["bid"] == 10.0
    assert ordered[4]["ask"] is None
    assert ordered[4]["bid_size"] is None
    assert ordered[4]["ask_size"] is None
    assert ordered[5]["stream_availability"] == "BBO_ONLY"
    lifecycle_files = sorted(
        collector.root.glob("**/event_type=optional_trade_stream_state/**/*.parquet")
    )
    lifecycle_rows = [
        row for path in lifecycle_files for row in pq.ParquetFile(path).read().to_pylist()
    ]
    assert [
        row["stream_state"]
        for row in sorted(lifecycle_rows, key=lambda row: row["source_sequence"])
    ] == ["active", "inactive"]
    with collector.quality_path.open(newline="", encoding="utf-8") as handle:
        quality = next(row for row in csv.DictReader(handle) if row["symbol"] == "AAL")
    assert quality["optional_trade_stream"] == "BBO_PLUS_TRADES"
    assert quality["size_change_count"] == "0"


def test_readiness_report_cannot_claim_ready_without_whole_universe_pilot(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path)
    collector.persist_raw_events((_quote(1, field="bid_size"),))
    report_path = tmp_path / "activation_readiness_report.json"
    report = evaluate_pilot_readiness(
        dataset_root=collector.root,
        universe_symbols=SYMBOLS,
        capacity_allowed=True,
        reconnect_validation_passed=True,
        safety_tests_passed=True,
        disk_capacity_acceptable=True,
        output_path=report_path,
    )
    assert report["classification"] == "BLOCKED_BBO_SIZE_DATA_MISSING"
    assert report["directional_analysis_performed"] is False
    assert report_path.is_file()
