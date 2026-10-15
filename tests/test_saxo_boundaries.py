"""Sanitised offline provider fixtures. These tests never contact a broker."""

import asyncio
import gzip
import json
import os
import time

import httpx
import pytest
from pydantic import ValidationError

from stocker_execution.config import PaperConfig, SaxoSettings
from stocker_execution.saxo_auth import ENDPOINTS, OAuth, SaxoRefused, atomic_json
from stocker_execution.saxo_client import SaxoClient, allowed
from stocker_execution.store import Store


def test_configuration_has_no_trading_switches():
    for config in (
        {"execution_mode": "INTERNAL_PAPER"},
        {"armed": True},
        {"data_environment": "IBKR"},
        {"markets": ["NQ"]},
    ):
        with pytest.raises(ValidationError):
            PaperConfig.model_validate(config)
    assert PaperConfig(data_environment="SAXO_LIVE")


def test_the_client_can_only_read():
    for path in (
        "/trade/v2/orders",
        "/trade/v2/orders/precheck",
        "/trade/v1/orders",
        "/trade/v2/positions/123",
        "/atr/v1/cashwithdrawals",
        "/trade/v2/exercise",
        "/root/v1/sessions/capabilities",
        "/trade/v1/prices/subscriptions",
        "/chart/v3/charts/subscriptions",
    ):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert not allowed(method, path), (method, path)
    for path in (
        "/trade/v1/infoprices",
        "/trade/v1/infoprices/list",
        "/chart/v3/charts",
        "/ref/v1/exchanges/NASDAQ",
        "/ref/v1/instruments/contractoptionspaces/306",
        "/ref/v1/instruments/details/4912/CfdOnIndex",
        "/root/v1/sessions/capabilities",
    ):
        assert allowed("GET", path), path
    for path in (
        "/trade/v2/orders",
        "/port/v1/positions/me",
        "/ref/v1/instruments/details/1/FuturesOption",
        "/trade/v1/infoprices/subscriptions",
        "https://evil.example/ref",
        "/trade/v1/infoprices/../orders",
        "/trade/v1/infoprices%2f..",
        "/trade/v1/infoprices?x=1",
    ):
        assert not allowed("GET", path), path


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


def test_secrets_permissions_and_redirects(tmp_path):
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
        with pytest.raises(SaxoRefused, match="ENDPOINT_NOT_ALLOWED"):
            await client.get("/trade/v2/orders")
        assert not called
        await client.close()
        os.chmod(auth.settings.credentials_file, 0o644)
        with pytest.raises(ValueError, match="0600"):
            OAuth("SAXO_LIVE", auth.settings, tmp_path)

    asyncio.run(scenario())


@pytest.mark.parametrize("compressed", [False, True])
def test_rest_response_decoded_once_with_rate_headers_preserved(tmp_path, compressed):
    async def scenario():
        auth = oauth_fixture(tmp_path)
        auth.tokens = {"access_token": "fixture-access", "expires_at": time.time() + 1200}
        payload = {"UserId": 123, "GeneralOperations": []}

        def response(request):
            assert request.method == "GET" and request.url.path.endswith(
                "/root/v1/sessions/capabilities"
            )
            body = json.dumps(payload).encode()
            headers = {"X-RateLimit-Session-Remaining": "119", "Content-Type": "application/json"}
            if compressed:
                body = gzip.compress(body)
                headers["Content-Encoding"] = "gzip"
            return httpx.Response(200, content=body, headers=headers)

        client = SaxoClient(auth, httpx.MockTransport(response))
        try:
            assert await client.get("/root/v1/sessions/capabilities") == payload
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
                await client.get("/root/v1/sessions/capabilities")
            assert client.waiters == 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_the_record_refuses_another_database_or_environment(tmp_path):
    other = tmp_path / "futures.sqlite3"
    import sqlite3

    with sqlite3.connect(other) as db:
        db.execute("CREATE TABLE futures_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    with pytest.raises(ValueError, match="PAPER_RECORD_DATABASE_REQUIRED"):
        Store(other)
    store = Store(tmp_path / "paper.sqlite3")
    store.bind("SAXO_LIVE")
    store.bind("SAXO_LIVE")
    with pytest.raises(ValueError, match="PROVENANCE"):
        store.bind("SAXO_SIM")
    store.db.close()


def test_a_rate_limit_waits_for_the_exhausted_limit_to_reset():
    import httpx

    from stocker_execution.saxo_client import retry_after

    refdata = httpx.Headers(
        {
            "X-RateLimit-RefDataInstrumentsMinute-Remaining": "0",
            "X-RateLimit-RefDataInstrumentsMinute-Reset": "42",
            "X-RateLimit-AppDay-Remaining": "9999000",
            "X-RateLimit-AppDay-Reset": "80000",
        }
    )
    assert retry_after(refdata) == 42  # not the day limit, which still has plenty left
    assert retry_after(httpx.Headers({"x-ratelimit-session-reset": "7"})) == 7
    assert retry_after(httpx.Headers({})) == 5
    assert (
        retry_after(httpx.Headers({"x-ratelimit-x-remaining": "0", "x-ratelimit-x-reset": "900"}))
        == 60
    )
