"""Read-only inspection cannot instantiate the causal benchmark or execution engine."""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from stocker_core.config import IbkrConfig
from stocker_core.runs import Environment
from stocker_execution.activity_shortlist import ScannerCapabilities
from stocker_execution.discovery import DiscoveryRow

MODULE = Path(__file__).parents[1] / "scripts/benchmark_scanner_acquisition.py"
spec = importlib.util.spec_from_file_location("benchmark_cli", MODULE)
assert spec and spec.loader
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def arguments(tmp_path, scanner_check=False):
    return argparse.Namespace(
        ibkr_config=tmp_path / "unused.yaml", client_id=292,
        state=tmp_path / "inspection.sqlite", capabilities_only=not scanner_check,
        scanner_check=scanner_check, markets=["US_ALL", "US_NASDAQ"],
        scan_codes=["TOP_TRADE_RATE", "HOT_BY_VOLUME"],
        confirm_dedicated_paper_gateway=False,
    )


@pytest.mark.parametrize("scanner_check", [False, True])
def test_readonly_inspection_isolated_and_bounded(tmp_path, monkeypatch, scanner_check):
    class Broker:
        def __init__(self, config, *, execution_enabled):
            assert not execution_enabled and config.environment is Environment.PAPER
            assert config.client_id == 292
            self.requests = []
            self.active = self.maximum = 0
            self.disconnected = False

        async def connect(self):
            pass

        async def scanner_capabilities(self):
            return ScannerCapabilities(
                frozenset({"STK.US.MAJOR"}),
                frozenset({"TOP_TRADE_RATE", "HOT_BY_VOLUME"}),
                frozenset({"marketCapAbove", "marketCapBelow"}),
                raw_xml="<ScannerParameters/>", server_version="FAKE",
            )

        async def acquisition_scan(self, plan, audit):
            self.requests.append(plan)
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(0)
                audit["cancelled_at"] = "fake-completed"
                if plan.family == "HOT_BY_VOLUME":
                    raise RuntimeError("fake request entitlement failure")
                audit["warnings"] = ["492 request-scoped precision warning"]
                return (DiscoveryRow(11, "S", "SMART", "NASDAQ", "USD", "STK", 3, {}),)
            finally:
                self.active -= 1

        def disconnect(self):
            self.disconnected = True

    config = IbkrConfig(environment=Environment.PAPER, host="localhost", port=4002, client_id=1)
    broker = Broker(config.model_copy(update={"client_id": 292}), execution_enabled=False)
    monkeypatch.setattr(cli, "load_ibkr_config", lambda *args: config)
    monkeypatch.setattr(cli, "IbkrConnection", lambda *a, **kw: broker)

    def forbidden(*args, **kwargs):
        pytest.fail("Inspection must not construct or read strategy state")

    monkeypatch.setattr(cli, "load_runs_config", forbidden)
    monkeypatch.setattr(cli, "AcquiredCandidates", forbidden)
    args = arguments(tmp_path, scanner_check)
    asyncio.run(cli.benchmark(args))
    assert broker.disconnected
    report = json.loads(args.state.with_suffix(".scanner-check.json").read_text())
    assert report["evidence"] == "ACCESS_DIAGNOSTIC_ONLY_NOT_OPENING_RECALL"
    assert len(report["markets"]) == 2
    if scanner_check:
        assert len(broker.requests) == 14  # Identical market-location requests shared.
        assert broker.maximum == 2 and broker.active == 0
        rows = report["markets"][0]["components"]
        assert sum(r["status"] == "COMPLETE" for r in rows) == 7
        assert sum(r["status"] == "FAILED" for r in rows) == 7
        assert rows[0]["hits"][0]["raw_rank"] == 3
        assert rows[0]["audit"]["warnings"]
        assert all(r["shared_request"] for r in report["markets"][1]["components"])
    else:
        assert not broker.requests


def test_full_benchmark_still_requires_exclusive_gateway(tmp_path):
    args = arguments(tmp_path)
    args.capabilities_only = args.scanner_check = False
    args.scan_codes = args.markets = None
    args.runs_config = tmp_path / "runs.yaml"
    args.run_id = "test"
    args.history_cache = tmp_path / "history.sqlite"
    with pytest.raises(ValueError, match="dedicated PAPER Gateway"):
        asyncio.run(cli.benchmark(args))
