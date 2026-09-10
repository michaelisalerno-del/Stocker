"""Opt-in read-only PAPER scanner smoke test; no execution engine is constructed."""

import asyncio
import os
from datetime import UTC, datetime

import pytest

from legacy_discovery_support import UniverseRunBuilder
from stocker_core.config import RunsConfig, load_ibkr_config
from stocker_core.markets import MARKET_CATALOGUE
from stocker_core.methods import LEGACY_SESSION_HARD as SESSION_HARD
from stocker_core.runs import Environment
from stocker_execution.discovery import CandidateDiscovery, DiscoveryStore, watch_identities
from stocker_execution.ibkr import IbkrConnection


@pytest.mark.skipif(
    not os.environ.get("STOCKER_DISCOVERY_IBKR_CONFIG"),
    reason="Set STOCKER_DISCOVERY_IBKR_CONFIG to an explicit PAPER Gateway config",
)
@pytest.mark.parametrize("market", MARKET_CATALOGUE, ids=lambda m: m.market_id.value)
def test_gateway_discovery(tmp_path, market):
    async def scenario():
        broker = IbkrConnection(
            load_ibkr_config(
                os.environ["STOCKER_DISCOVERY_IBKR_CONFIG"],
                Environment.PAPER,
            )
        )
        await broker.connect()
        try:
            now = datetime.now(UTC)
            _, run = UniverseRunBuilder().add(
                RunsConfig(), market_id=market.market_id,
                strategy_id=SESSION_HARD.method_id, strategy_version=SESSION_HARD.version,
                environment=Environment.PAPER,
            )
            result = await CandidateDiscovery(
                broker,
                DiscoveryStore(tmp_path / "gateway-discovery.sqlite"),
            ).discover(run, market, now.date(), now, now)
            capabilities = await broker.scanner_capabilities()
            if run.discovery_profile.scanner_location not in capabilities.locations:
                assert result["status"] == "FAILED"
                assert result["reason"].startswith("SCANNER_NOT_SUPPORTED")
                assert not result["observations"] and not watch_identities(result)
                return
            assert result["status"] in {"READY", "EMPTY"}, result["reason"]
            assert len(result["segments"]) == 5
            assert all(segment["status"] == "COMPLETE" for segment in result["segments"])
            identities = watch_identities(result)
            assert len({i.con_id for i in identities}) == len(identities)
        finally:
            broker.disconnect()

    asyncio.run(scenario())
