from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.contract import SECTOR_PROXY_BY_SYMBOL
from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.fake_ibkr import FakeIBKRAdapter
from stocker_prospective.frozen_live_application import (
    build_frozen_prospective_application,
)
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


def test_full_recorder_application_starts_and_polls_with_fake_ibkr(
    tmp_path: Path,
) -> None:
    universe = json.loads(
        (ROOT / "configs/prospective/anchor-frozen-20.json").read_text(encoding="utf-8")
    )
    symbols = tuple(str(value) for value in universe["symbols"])
    activity_path = tmp_path / "historical-activity.parquet"
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "session": "2026-07-23",
                "bar_ordinal": 0,
                "volume": 1_000.0,
            }
            for symbol in symbols
        ]
    ).to_parquet(activity_path, index=False)
    bar_report = tmp_path / "bar-compatibility.json"
    bar_report.write_text('{"passed":true}\n', encoding="utf-8")
    context_root = tmp_path / "context"
    context_root.mkdir()

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
                "git_commit": "a" * 40,
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
    adapter = FakeIBKRAdapter.from_fixture(FAKE_FIXTURE)
    adapter.connect()
    repository = ProspectiveRepository(config.paths.database)
    repository.migrate()

    application = build_frozen_prospective_application(
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
            lambda symbol, expiry, strike, right, multiplier, exchange, trading: SimpleNamespace(
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
        ),
        ibkr_api_version="fake-10.37",
    )
    polled_at = datetime.now(UTC)
    result = application.poll(now=polled_at)

    assert result.checkpoint_results == ()
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
