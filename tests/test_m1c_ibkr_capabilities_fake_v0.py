from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from stocker_prospective.capability import (
    CapabilityObservation,
    run_capability_preflight,
)
from stocker_prospective.config import load_prospective_config
from stocker_prospective.contract import FORBIDDEN_BROKER_METHODS
from stocker_prospective.fake_ibkr import FakeIBKRAdapter
from stocker_prospective.market_data import MarketDataType

ROOT = Path(__file__).parents[1]
FIXTURE = (
    ROOT
    / "packages/stocker_prospective/src/stocker_prospective/fixtures"
    / "frozen-m1c-recorder-v0.json"
)


def test_exact_ibkr_environment_variables_are_supported_and_read_only_is_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (ROOT / "configs/prospective/replay.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCKER_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("IBKR_HOST", "127.0.0.1")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "91")
    monkeypatch.setenv("IBKR_READ_ONLY", "true")
    monkeypatch.setenv("IBKR_MARKET_DATA_TYPE_REQUIRED", "live")
    monkeypatch.setenv("IBKR_ENABLE_LEVEL2", "true")
    monkeypatch.setenv("IBKR_LEVEL2_ROWS", "5")
    monkeypatch.setenv("IBKR_MAX_DEPTH_SUBSCRIPTIONS", "3")
    monkeypatch.setenv("IBKR_MAX_TICK_BY_TICK_SUBSCRIPTIONS", "6")
    monkeypatch.setenv("IBKR_CONNECTION_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("IBKR_RECONNECT_BACKOFF_SECONDS", "3")

    config = load_prospective_config(config_path)

    assert config.ibkr.port == 4002
    assert config.ibkr.client_id == 91
    assert config.ibkr.read_only is True
    assert config.ibkr.enable_level2 is True
    assert config.ibkr.level2_rows == 5
    assert config.ibkr.max_depth_subscriptions == 3
    assert config.ibkr.max_tick_by_tick_subscriptions == 6
    assert config.ibkr.connect_timeout_seconds == 8
    assert config.ibkr.reconnect_backoff_seconds == 3

    monkeypatch.setenv("IBKR_READ_ONLY", "false")
    with pytest.raises(ValueError, match="read-only"):
        load_prospective_config(config_path)


def observation(market_data_type: MarketDataType) -> CapabilityObservation:
    return CapabilityObservation(
        connected=True,
        api_server_version=187,
        ibkr_api_version="10.37.01",
        tws_or_gateway_version="IB Gateway 10.37",
        market_data_type=market_data_type,
        underlying_level1_symbols=("AAL", "AAOI"),
        market_proxy_level1_symbols=("VTI",),
        option_level1_available=True,
        option_computation_fields_available=True,
        tick_by_tick_capacity=4,
        depth_capacity=2,
        option_capacity=30,
        depth_exchanges=("NYSE", "NASDAQ"),
        resolved_contracts=("AAL:265598", "AAOI:123"),
        unresolved_contracts=(),
        clock_drift_seconds=0.25,
        new_york_calendar_valid=True,
        timestamps_valid=True,
        permission_errors=(),
    )


def test_capability_preflight_rejects_delayed_data_but_preserves_diagnostic_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ibkr_capability_manifest.json"
    delayed = run_capability_preflight(
        observation(MarketDataType.DELAYED),
        required_underlyings=("AAL", "AAOI"),
        required_market_proxies=("VTI",),
        maximum_clock_drift_seconds=1.0,
        output_path=path,
        observed_at=datetime(2026, 7, 24, 13, 0, tzinfo=UTC),
    )

    assert delayed.scientific_recording_valid is False
    assert "market_data_not_live" in delayed.blockers
    assert delayed.diagnostic_display_allowed is True
    assert json.loads(path.read_text(encoding="utf-8"))["scientific_recording_valid"] is False

    live = run_capability_preflight(
        observation(MarketDataType.LIVE),
        required_underlyings=("AAL", "AAOI"),
        required_market_proxies=("VTI",),
        maximum_clock_drift_seconds=1.0,
        output_path=None,
        observed_at=datetime(2026, 7, 24, 13, 0, tzinfo=UTC),
    )
    assert live.scientific_recording_valid is True
    assert live.blockers == ()


def test_fake_adapter_has_all_engineering_scenarios_and_no_order_surface() -> None:
    adapter = FakeIBKRAdapter.from_fixture(FIXTURE)
    public = {name for name in dir(adapter) if not name.startswith("_")}

    assert FORBIDDEN_BROKER_METHODS.isdisjoint(public)
    assert {
        "bullish_continuation",
        "bearish_continuation",
        "bullish_absorption",
        "bearish_absorption",
        "bad_quote_quality",
        "ibkr_disconnect",
        "unavailable_0dte",
    }.issubset(adapter.scenarios)
    first = tuple(adapter.replay())
    second = tuple(adapter.replay())
    assert first == second
    assert any(item.kind == "connection_loss" for item in first)
    assert any(item.kind == "depth_reset" for item in first)
    assert any(item.kind == "subscription_capacity_error" for item in first)


def test_fake_adapter_exercises_the_production_market_data_interface() -> None:
    adapter = FakeIBKRAdapter.from_fixture(FIXTURE)
    adapter.connect()
    contract = SimpleNamespace(
        symbol="AAL",
        secType="STK",
        exchange="SMART",
        currency="USD",
    )
    qualified = adapter.qualify_exact_contract(contract)
    request_ids = {
        adapter.request_market_data(contract),
        adapter.request_historical_five_minute_updates(contract),
        adapter.request_tick_by_tick(contract, "BidAsk"),
        adapter.request_tick_by_tick(contract, "Last"),
        adapter.request_market_depth(contract),
        adapter.request_tick_by_tick(
            SimpleNamespace(symbol="AAOI"),
            "Last",
        ),
        adapter.request_market_depth(SimpleNamespace(symbol="APLD")),
        adapter.request_market_depth(SimpleNamespace(symbol="CIFR")),
    }
    adapter.request_current_time()
    adapter.request_depth_exchanges()
    callbacks = adapter.drain_stream_events()

    assert len(qualified.items) == 1
    assert qualified.items[0].conId > 0
    assert request_ids.issubset(adapter.active_subscriptions)
    assert {item["kind"] for item in callbacks}.issuperset(
        {
            "current_time",
            "depth_exchanges",
            "level1_quote_update",
            "historical_bar_update",
            "tick_by_tick_bidask",
            "tick_by_tick_trade",
            "depth_update",
            "depth_reset",
        }
    )
    assert adapter.server_version() == 187
    for request_id in request_ids:
        adapter.cancel_market_data(request_id)
    assert adapter.active_subscriptions == {}
