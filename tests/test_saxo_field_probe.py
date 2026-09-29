"""The live field probe is read-only, never refreshes tokens and never prints keys."""

import asyncio
import importlib.util
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stocker_execution.config import SaxoSettings

spec = importlib.util.spec_from_file_location("probe", Path("scripts/saxo_field_probe.py"))
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)

ACCOUNT = "secret-account-key"
NOW = datetime.now(UTC)
EXPIRY = (NOW + timedelta(days=40)).date().isoformat()


class FakeClient:
    def __init__(self):
        self.calls = []

    async def request(self, method, path, params=None, body=None, **kwargs):
        self.calls.append((method, path))
        if path == "/ref/v1/instruments":
            return {
                "Data": [{"Identifier": 7, "Symbol": "CLF9"}, {"Identifier": 8, "Symbol": "CLZ9"}]
            }
        if path.startswith("/ref/v1/instruments/details/"):
            return {
                "Uic": int(path.split("/")[5]),
                "Symbol": "CLZ9",
                "ExpiryDate": EXPIRY,
                "AccountKey": ACCOUNT,
                "RelatedOptionRootsEnhanced": [{"AssetType": "FuturesOption", "OptionRootId": 55}],
                "TradingSessions": {
                    "Sessions": [
                        {
                            "StartTime": "2000-01-01T00:00:00Z",
                            "EndTime": "2100-01-01T00:00:00Z",
                            "State": "AutomatedTrading",
                        }
                    ]
                },
            }
        if path == "/chart/v3/charts":
            return {
                "Data": [
                    {
                        "Time": f"2026-09-29T14:{m:02d}:00Z",
                        "Close": 70 + (m % 3) * 0.05,
                        "Open": 70,
                        "High": 71,
                        "Low": 69,
                        "Volume": 5,
                    }
                    for m in range(30)
                ],
                "ChartInfo": {"DelayedByMinutes": 0},
                "DataVersion": 3,
            }
        if path == "/trade/v1/infoprices" and params["AssetType"] == "ContractFutures":
            return {
                "Quote": {
                    "Bid": 70,
                    "Ask": 70.02,
                    "Mid": 70.01,
                    "PriceTypeBid": "Indicative",
                    "PriceTypeAsk": "Indicative",
                    "MarketState": "Open",
                    "AccountKey": ACCOUNT,
                },
                "PriceInfo": {"High": 71},
                "MarketDepth": {"Bid": [70, 69.99]},
            }
        if path.startswith("/ref/v1/instruments/contractoptionspaces/"):
            return {
                "OptionSpace": [
                    {
                        "Expiry": EXPIRY,
                        "SpecificOptions": [
                            {
                                "Uic": 90 + i,
                                "UnderlyingUic": 7,
                                "StrikePrice": 69 + i // 2,
                                "PutCall": ("Put", "Call")[i % 2],
                            }
                            for i in range(6)
                        ],
                    }
                ]
            }
        if path == "/trade/v1/infoprices":
            return {
                "Quote": {
                    "Bid": 1.1,
                    "Ask": 1.2,
                    "PriceTypeBid": "Indicative",
                    "PriceTypeAsk": "Indicative",
                },
                "Greeks": {"Delta": -0.5, "MidVol": 31.2},
            }
        if method == "POST":
            return {
                "Snapshot": {
                    "Expiries": [
                        {
                            "Expiry": EXPIRY,
                            "MidStrikePrice": 70.01,
                            "Strikes": [
                                {
                                    "Strike": 70,
                                    "MidVolatilityPct": 0.31,
                                    "Put": {"Bid": 1, "Greeks": {"MidVolatility": 0.312}},
                                }
                            ],
                        }
                    ]
                }
            }
        if method == "DELETE":
            return {}
        raise AssertionError((method, path))


def test_probe_reports_live_field_shapes_and_cleans_up():
    client = FakeClient()
    report = asyncio.run(probe_module.probe(client, "CL", ACCOUNT, chain=True))
    assert (
        report["future"]["Uic"] == 7 and report["future"]["current_session"] == "AutomatedTrading"
    )
    assert report["chart_v3_minute"]["samples"] == 30
    assert report["chart_v3_minute"]["all_on_minute_boundaries"] is True
    assert report["chart_v3_minute"]["realised_vol_annualised"] > 0
    assert report["future_infoprice"]["quote"]["PriceTypeBid"] == "Indicative"
    assert [o["right"] for o in report["options"]] == ["Put", "Call"]
    assert report["options"][0]["greeks"]["MidVol"] == 31.2
    assert report["chain_snapshot"]["strikes"][0]["put"]["Greeks"]["MidVolatility"] == 0.312
    methods = [m for m, _ in client.calls]
    assert methods.count("POST") == methods.count("DELETE") == 1
    assert not any("orders" in path for _, path in client.calls)
    assert ACCOUNT not in json.dumps(report)


def test_current_token_reads_but_never_refreshes(tmp_path):
    credentials = tmp_path / "credentials.json"
    credentials.write_text(
        json.dumps(
            {
                "environment": "SAXO_SIM",
                "client_id": "a",
                "client_secret": "b",
                "account_key": ACCOUNT,
            }
        )
    )
    os.chmod(credentials, 0o600)
    tokens = tmp_path / "SAXO_SIM" / "oauth-tokens.json"
    tokens.parent.mkdir()
    tokens.write_text(
        json.dumps(
            {
                "environment": "SAXO_SIM",
                "access_token": "live",
                "refresh_token": "never-used",
                "expires_at": time.time() + 600,
                "refresh_expires_at": time.time() + 3600,
            }
        )
    )
    os.chmod(tokens, 0o600)
    oauth = probe_module.CurrentToken(
        "SAXO_SIM", SaxoSettings(credentials_file=credentials), tmp_path
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("the probe must never exchange or refresh a token")

    oauth.exchange = forbidden
    assert asyncio.run(oauth.access_token()) == "live"
    tokens.write_text(
        json.dumps(
            {"environment": "SAXO_SIM", "access_token": "old", "expires_at": time.time() + 30}
        )
    )
    with pytest.raises(ValueError, match="NEAR_EXPIRY"):
        asyncio.run(oauth.access_token())
    asyncio.run(oauth.close())
