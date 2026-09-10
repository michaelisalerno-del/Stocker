"""Opt-in read-only PAPER scanner smoke test; no execution engine is constructed."""

import asyncio
import os
from datetime import UTC, datetime

import pytest

from stocker_core.config import load_ibkr_config
from stocker_core.runs import Environment
from stocker_execution.discovery import CandidateDiscovery, DiscoveryStore, watch_identities
from stocker_execution.ibkr import IbkrConnection
from test_candidate_discovery import MARKET, make_run


@pytest.mark.skipif(
    not os.environ.get("STOCKER_DISCOVERY_IBKR_CONFIG"),
    reason="Set STOCKER_DISCOVERY_IBKR_CONFIG to an explicit PAPER Gateway config",
)
def test_gateway_discovery(tmp_path):
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
            result = await CandidateDiscovery(
                broker,
                DiscoveryStore(tmp_path / "gateway-discovery.sqlite"),
            ).discover(make_run()[1], MARKET, now.date(), now, now)
            assert result["status"] in {"READY", "EMPTY"}, result["reason"]
            assert len(result["segments"]) == 5
            assert all(segment["status"] == "COMPLETE" for segment in result["segments"])
            identities = watch_identities(result)
            assert len({i.con_id for i in identities}) == len(identities)
        finally:
            broker.disconnect()

    asyncio.run(scenario())
