"""Sanitised offline provider fixtures. These tests never contact a broker."""

import asyncio
import gzip
import json
import os
import time

import httpx
import pytest
from pydantic import ValidationError

from stocker_execution.config import MARKETS, FuturesConfig, SaxoSettings
from stocker_execution.contracts import budget, future_identity
from stocker_execution.recorder import Recorder
from stocker_execution.saxo_auth import ENDPOINTS, OAuth, atomic_json
from stocker_execution.saxo_client import SaxoClient, allowed
from stocker_execution.saxo_data import DataService
from stocker_execution.saxo_stream import Frames, PriceState, merge
from stocker_execution.store import Store


def test_no_live_mode_or_crypto_configuration():
    assert MARKETS == ("CL", "GC", "NG", "NQ", "SI")
    for config in (
        {"execution_mode": "LIVE"},
        {"data_environment": "IBKR"},
        {"armed": True},
        {"markets": ["BTC"]},
        {"mappings": {"BTC": {}}},
        {"execution_mode": "SAXO_SIM", "data_environment": "SAXO_LIVE"},
    ):
        with pytest.raises(ValidationError):
            FuturesConfig.model_validate(config)
    assert FuturesConfig(data_environment="SAXO_LIVE", execution_mode="INTERNAL_PAPER")


def test_live_default_deny_and_subscription_lifecycle():
    for path in (
        "/trade/v2/orders",
        "/trade/v1/orders",
        "/trade/v2/positions/123",
        "/atr/v1/cashwithdrawals",
        "/trade/v2/exercise",
        "/root/v1/sessions/capabilities",
        "/trade/v1/prices/subscriptions/../../orders",
        "https://evil.example/ref",
        "/trade/v1/prices/subscriptions%2f..",
        "/trade/v1/prices/subscriptions?x=1",
    ):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert not allowed(method, path)
    assert allowed("POST", "/trade/v1/prices/subscriptions")
    assert allowed("DELETE", "/trade/v1/prices/subscriptions/context/ref")
    assert allowed("GET", "/root/v2/user")
    assert not allowed("GET", "/root/v1/user")
    assert not allowed("POST", "/trade/v2/orders", sim_orders=False)


def oauth_fixture(tmp_path, env="SAXO_SIM", handler=None):
    path = tmp_path / (env + "-credentials.json")
    atomic_json(
        path,
        {
            "environment": env,
            "client_id": "fixture-key",
            "client_secret": "fixture-secret",
            "account_key": "fixture-account",
        },
    )
    settings = SaxoSettings(credentials_file=path)
    return OAuth(
        env, settings, tmp_path, transport=httpx.MockTransport(handler) if handler else None
    )


@pytest.mark.parametrize("status", [200, 201])
def test_oauth_state_rotation_and_environment_binding(tmp_path, status):
    requests = []

    def response(req):
        requests.append(req)
        return httpx.Response(
            status,
            json={
                "access_token": f"access-{len(requests)}",
                "refresh_token": f"refresh-{len(requests)}",
                "expires_in": 1200,
                "refresh_token_expires_in": 2400,
            },
        )

    async def scenario():
        auth = oauth_fixture(tmp_path, handler=response)
        location, binding = auth.begin()
        assert location.startswith(ENDPOINTS["SAXO_SIM"]["auth"] + "/authorize?")
        state = auth.pending[0]
        with pytest.raises(ValueError, match="STATE"):
            await auth.callback(state, "wrong-browser", "code")
        assert not requests
        _, binding = auth.begin()
        state = auth.pending[0]
        await auth.callback(state, binding, "fixture-code")
        assert await auth.access_token() == "access-1"
        with pytest.raises(ValueError, match="STATE"):
            await auth.callback(state, binding, "replay")
        auth.tokens["expires_at"] = time.time() - 1
        await asyncio.gather(*(auth.access_token() for _ in range(10)))
        assert len(requests) == 2
        assert json.loads(auth.token_file.read_text())["refresh_token"] == "refresh-2"
        assert auth.token_file.stat().st_mode & 0o077 == 0
        live = oauth_fixture(tmp_path, "SAXO_LIVE")
        assert live.token_file != auth.token_file and not live.tokens
        await live.close()
        await auth.close()

    asyncio.run(scenario())


def test_secrets_permissions_redirects_and_live_request_transmission(tmp_path):
    for redirect in (
        "https://good.test/other",
        "https://evil.test/oauth/saxo/callback?next=x",
        "http://remote.test/oauth/saxo/callback",
        "https://user@host/oauth/saxo/callback",
    ):
        with pytest.raises(ValidationError):
            SaxoSettings(redirect_uri=redirect)

    async def scenario():
        auth = oauth_fixture(tmp_path, "SAXO_LIVE")
        called = []
        client = SaxoClient(auth, httpx.MockTransport(lambda req: called.append(req)))
        client.sim_account_verified = True  # cannot override the environment boundary
        with pytest.raises(ValueError, match="BLOCKED"):
            await client.request(
                "POST", "/trade/v2/orders", body={"AccountKey": "fixture-account"}, execution=True
            )
        assert not called
        await client.close()
        os.chmod(auth.settings.credentials_file, 0o644)
        with pytest.raises(ValueError, match="0600"):
            OAuth("SAXO_LIVE", auth.settings, tmp_path)

    asyncio.run(scenario())


def test_oauth_can_precede_account_selection_without_enabling_orders(tmp_path):
    async def scenario():
        requests = []

        def response(req):
            requests.append((req.method, req.url.path))
            if req.url.path == "/token":
                return httpx.Response(
                    200,
                    json={
                        "access_token": "fixture-access",
                        "refresh_token": "fixture-refresh",
                        "expires_in": 1200,
                        "refresh_token_expires_in": 2400,
                    },
                )
            if req.url.path.endswith("/root/v2/user"):
                return httpx.Response(200, json={"UserId": "fixture-user"})
            if req.url.path.endswith("/port/v1/accounts/me"):
                return httpx.Response(
                    200,
                    json={
                        "Data": [
                            {
                                "AccountKey": "fixture-account",
                                "AccountId": "fixture-id",
                                "Currency": "GBP",
                            }
                        ]
                    },
                )
            raise AssertionError("Unexpected request")

        auth = oauth_fixture(tmp_path, handler=response)
        path = auth.settings.credentials_file
        credentials = json.loads(path.read_text())
        del credentials["account_key"]
        atomic_json(path, credentials)
        await auth.close()
        auth = OAuth(
            "SAXO_SIM",
            SaxoSettings(credentials_file=path),
            tmp_path,
            transport=httpx.MockTransport(response),
        )
        _, binding = auth.begin()
        await auth.callback(auth.pending[0], binding, "fixture-code")
        assert auth.status == "AUTHENTICATED" and auth.account_key == ""
        client = SaxoClient(auth, httpx.MockTransport(response))
        config = FuturesConfig()
        data = DataService(config, client, Recorder(config.recorder, tmp_path / "events"))
        with pytest.raises(ValueError, match="ACCOUNT_SELECTION_REQUIRED"):
            await data.verify_account()
        assert not data.account_verified and not client.sim_account_verified
        assert not data.connected and not data.subscriptions
        with pytest.raises(ValueError, match="BLOCKED"):
            await client.request(
                "POST", "/trade/v2/orders", execution=True, body={"AccountKey": "fixture-account"}
            )
        assert requests == [
            ("POST", "/token"),
            ("GET", "/sim/openapi/root/v2/user"),
            ("GET", "/sim/openapi/port/v1/accounts/me"),
        ]
        await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("compressed", [False, True])
def test_rest_response_decoded_once_with_rate_headers_preserved(tmp_path, compressed):
    async def scenario():
        auth = oauth_fixture(tmp_path)
        auth.tokens = {"access_token": "fixture-access", "expires_at": time.time() + 1200}
        payload = {"UserId": 123, "GeneralOperations": []}

        def response(request):
            assert request.method == "GET" and request.url.path.endswith("/root/v2/user")
            body = json.dumps(payload).encode()
            headers = {"X-RateLimit-Session-Remaining": "119", "Content-Type": "application/json"}
            if compressed:
                body = gzip.compress(body)
                headers["Content-Encoding"] = "gzip"
            return httpx.Response(200, content=body, headers=headers)

        client = SaxoClient(auth, httpx.MockTransport(response))
        try:
            assert await client.request("GET", "/root/v2/user") == payload
            assert client.rate_headers["x-ratelimit-session-remaining"] == "119"
            assert client.waiters == 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_rest_compressed_response_still_enforces_decoded_size_limit(tmp_path):
    async def scenario():
        auth = oauth_fixture(tmp_path)
        auth.tokens = {"access_token": "fixture-access", "expires_at": time.time() + 1200}
        client = SaxoClient(
            auth,
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=gzip.compress(b"x" * (4 * 1024**2 + 1)),
                    headers={"Content-Encoding": "gzip"},
                )
            ),
        )
        try:
            with pytest.raises(ValueError, match="REST_RESPONSE_LIMIT"):
                await client.request("GET", "/root/v2/user")
            assert client.waiters == 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_merge_explicit_null_array_replacement_and_cleared_depth():
    snapshot = {"Quote": {"Bid": 1, "Ask": 2}, "MarketDepth": {"Bid": [1, 0.9], "BidSize": [2, 3]}}
    updated = merge(snapshot, {"Quote": {"Ask": None}, "MarketDepth": {"Bid": [0.95]}})
    assert updated["Quote"] == {"Bid": 1, "Ask": None}
    assert updated["MarketDepth"]["Bid"] == [0.95]
    assert snapshot["Quote"]["Ask"] == 2
    assert merge(updated, {"MarketDepth": None})["MarketDepth"] is None
    assert merge(updated, {"MarketDepth": {"NoOfBids": 0}})["MarketDepth"]["BidSize"] == []


def test_stream_fragmentation_opaque_ids_duplicates_and_gap():
    import struct

    def frame(mid, value):
        data = json.dumps(value).encode()
        return (
            struct.pack("<QHB", mid, 0, 3) + b"ref" + b"\x00" + struct.pack("<I", len(data)) + data
        )

    parser = Frames()
    raw = frame(900, {"Data": {"Quote": {"Ask": 2}}}) + frame(7, {"Data": {"Quote": {"Bid": 1}}})
    assert not parser.feed(raw[:15])
    messages = parser.feed(raw[15:])
    assert [m["message_id"] for m in messages] == ["900", "7"]
    state = PriceState()
    state.snapshot({"Quote": {"Bid": 1}}, "a", 100)
    assert state.update({"Quote": {"Ask": 2}}, "900", 101)
    assert not state.update({"Quote": {"Ask": 2}}, "900", 102)
    assert state.update({"Quote": {"Ask": 3}}, "7", 103)
    state.gap("RESET")
    assert state.value is None and not state.update({"Quote": {"Bid": 4}}, "2", 104)
    assert not state.depth(104)["bids"]


def test_budget_real_contract_not_ten_contracts_and_no_fractional():
    option = {
        "asset_type": "FuturesOption",
        "minimum_quantity": 1,
        "lot_size": 1,
        "amount_decimals": 0,
        "price_factor": 1000,
        "currency": "USD",
    }
    assert budget(option, 0.01, 1, 0.75)["cash_pennies"] == 850
    with pytest.raises(ValueError, match="MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET"):
        budget(option, 0.013, 1, 0.75)
    with pytest.raises(ValueError, match="WHOLE"):
        budget({**option, "lot_size": 0.1}, 0.01, 1, 0.75)
    with pytest.raises(ValueError, match="DIRECT_FUTURES"):
        budget({**option, "asset_type": "ContractFutures"}, 0.01, 1, 0.75)


def test_contract_reference_rejects_substitutes():
    raw = {
        "AssetType": "ContractFutures",
        "Uic": 123,
        "Symbol": "CLZ6:NYMEX",
        "ContractSize": 1000,
        "PriceToContractFactor": 1000,
        "TickSize": 0.01,
        "ExpiryDate": "2026-12-20",
        "CurrencyCode": "USD",
        "Exchange": {"ExchangeId": "NYMEX"},
    }
    assert future_identity("CL", "SAXO_SIM", raw)["tick_value"] == 10
    november = {**raw, "Symbol": "CLX6:NYMEX", "ExpiryDate": "2026-10-20"}
    assert future_identity("CL", "SAXO_SIM", november)["contract_month"] == "2026-11"
    for substitute in ({**raw, "AssetType": "CfdOnFutures"}, {**raw, "Symbol": "MCLZ6:NYMEX"}):
        with pytest.raises(ValueError):
            future_identity("CL", "SAXO_SIM", substitute)


def event(i):
    return {
        "id": str(i),
        "market": "CL",
        "rule_version": "fixture",
        "signal_at": "2026-09-28T13:00:00+00:00",
        "exit_at": "2026-09-28T14:00:00+00:00",
    }


def test_atomic_capacity_and_provenance(tmp_path):
    import concurrent.futures

    path = tmp_path / "ledger.sqlite3"
    store = Store(path)
    store.bind("SAXO_SIM", "INTERNAL_PAPER")
    for i in range(12):
        store.observe(event(i), "", {})

    def reserve(i):
        other = Store(path)
        try:
            return other.reserve(str(i), {"quantity": 1, "cash_pennies": 100})
        finally:
            other.db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(reserve, range(12)))
    assert results.count("") == 4
    assert store.capacity()["reserved_open_trades"] == 4
    with pytest.raises(ValueError, match="SEPARATE"):
        store.bind("SAXO_LIVE", "INTERNAL_PAPER")
    with pytest.raises(ValueError, match="SEPARATE"):
        store.bind("SAXO_SIM", "SAXO_SIM")
    store.db.close()
