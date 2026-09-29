import asyncio
import base64
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stocker_dashboard.security import DashboardSecurity


def app():
    result = FastAPI()
    result.add_middleware(DashboardSecurity)

    @result.api_route("/control", methods=["GET", "POST"])
    def control():
        return {"ok": True}

    return result


def test_local_dashboard_rejects_remote_and_cross_site(monkeypatch):
    monkeypatch.delenv("STOCKER_DASHBOARD_PASSWORD", raising=False)
    with TestClient(app(), base_url="http://127.0.0.1", client=("127.0.0.1", 123)) as client:
        assert client.post("/control").status_code == 200
        assert client.post("/control", headers={"Origin": "https://evil.test"}).status_code == 403
        assert client.get("/control", headers={"Host": "evil.test"}).status_code == 403
    with TestClient(app(), base_url="http://127.0.0.1", client=("192.0.2.1", 123)) as client:
        assert client.post("/control", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 403


def test_protected_backend_requires_credentials_on_every_route(monkeypatch):
    password = "isolated-test-password-123456789"
    monkeypatch.setenv("STOCKER_DASHBOARD_PASSWORD", password)
    monkeypatch.setenv("STOCKER_DASHBOARD_ORIGIN", "https://stocker.example")
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"stocker:{password}".encode()).decode()
    }
    with TestClient(app(), base_url="https://stocker.example") as client:
        assert client.get("/control").status_code == 401
        assert (
            client.post("/control", headers={"X-Authenticated-User": "stocker"}).status_code == 401
        )
        assert client.post("/control", headers=headers).status_code == 200
        assert (
            client.post("/control", headers={**headers, "Origin": "https://evil.test"}).status_code
            == 403
        )
        assert (
            client.post(
                "/control", headers={**headers, "Origin": "https://stocker.example"}
            ).status_code
            == 200
        )


def test_authenticated_proxy_requires_loopback_and_private_credential(monkeypatch):
    monkeypatch.delenv("STOCKER_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setenv("STOCKER_DASHBOARD_PROXY_TOKEN", "isolated-proxy-credential-123456789")
    monkeypatch.setenv("STOCKER_DASHBOARD_ORIGIN", "https://stocker.example")
    headers = {"X-Stocker-Proxy-Token": "isolated-proxy-credential-123456789"}
    with TestClient(app(), base_url="https://stocker.example", client=("127.0.0.1", 123)) as client:
        assert client.get("/control").status_code == 403
        assert (
            client.post("/control", headers={"X-Authenticated-User": "stocker"}).status_code == 403
        )
        assert client.post("/control", headers=headers).status_code == 200
        assert (
            client.post("/control", headers={**headers, "Origin": "https://evil.test"}).status_code
            == 403
        )
    with TestClient(app(), base_url="https://stocker.example", client=("192.0.2.1", 123)) as client:
        assert (
            client.post("/control", headers={**headers, "X-Forwarded-For": "127.0.0.1"}).status_code
            == 403
        )


def test_websocket_boundary_rejects_before_application(monkeypatch):

    monkeypatch.delenv("STOCKER_DASHBOARD_PROXY_TOKEN", raising=False)
    monkeypatch.setenv("STOCKER_DASHBOARD_PASSWORD", "test-only-password-0123456789")
    monkeypatch.setenv("STOCKER_DASHBOARD_ORIGIN", "https://stocker.example")

    async def scenario():
        async def forbidden(scope, receive, send):
            pytest.fail("unauthenticated websocket reached application")

        messages = []

        async def send(message):
            messages.append(message)

        await DashboardSecurity(forbidden)(
            {"type": "websocket", "headers": [(b"host", b"stocker.example")]}, None, send
        )
        assert messages == [{"type": "websocket.close", "code": 1008}]

    asyncio.run(scenario())


def test_futures_server_preserves_authenticated_proxy_socket_peer(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    import httpx
    import uvicorn

    from stocker_execution.__main__ import main
    from stocker_execution.runtime import Runtime

    token = "isolated-proxy-credential-123456789"
    monkeypatch.delenv("STOCKER_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setenv("STOCKER_DASHBOARD_PROXY_TOKEN", token)
    monkeypatch.setenv("STOCKER_DASHBOARD_ORIGIN", "https://stocker.example")
    monkeypatch.setattr(Runtime, "run", AsyncMock())
    monkeypatch.setattr(Runtime, "stop", AsyncMock())
    statuses = []

    class Server:
        def __init__(self, config):
            self.config = config
            self.started = self.should_exit = True  # a normal, completed uvicorn run

        async def serve(self):
            self.config.load()
            transport = httpx.ASGITransport(self.config.loaded_app, client=("127.0.0.1", 123))
            async with httpx.AsyncClient(
                transport=transport, base_url="https://stocker.example"
            ) as client:
                forwarded = {"X-Forwarded-For": "198.51.100.17", "X-Forwarded-Proto": "https"}
                for headers in [
                    {"X-Stocker-Proxy-Token": token},
                    {**forwarded, "X-Stocker-Proxy-Token": token},
                    forwarded,
                ]:
                    statuses.append((await client.get("/api/system", headers=headers)).status_code)

    monkeypatch.setattr(uvicorn, "Server", Server)
    config = tmp_path / "futures.yaml"
    config.write_text("armed: false\n")
    main(["futures-run", "--config", str(config), "--database", str(tmp_path / "state.sqlite")])
    assert statuses == [200, 200, 403]


@pytest.mark.parametrize("token_status", [200, 201])
def test_oauth_redirect_origin_reaches_state_validation_and_landing(
    monkeypatch, tmp_path, token_status
):
    import httpx

    from stocker_dashboard.app import create_dashboard_app
    from stocker_execution.config import FuturesConfig, SaxoSettings
    from stocker_execution.runtime import Runtime
    from stocker_execution.saxo_auth import atomic_json
    from stocker_execution.store import Store

    token = "isolated-proxy-credential-123456789"
    monkeypatch.delenv("STOCKER_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setenv("STOCKER_DASHBOARD_PROXY_TOKEN", token)
    monkeypatch.setenv("STOCKER_DASHBOARD_ORIGIN", "https://stocker.example")
    path = tmp_path / "credentials.json"
    atomic_json(
        path,
        {"environment": "SAXO_SIM", "client_id": "fixture-key", "client_secret": "fixture-secret"},
    )

    async def scenario():
        runtime = Runtime(
            FuturesConfig(
                saxo=SaxoSettings(
                    credentials_file=path,
                    redirect_uri="https://stocker.example/oauth/saxo/callback",
                )
            ),
            Store(tmp_path / "state.sqlite3"),
        )
        exchanges = []

        def exchange(request):
            exchanges.append(request.url.path)
            return httpx.Response(
                token_status,
                json={
                    "access_token": "fixture-access",
                    "refresh_token": "fixture-refresh",
                    "expires_in": 1200,
                    "refresh_token_expires_in": 2400,
                },
            )

        await runtime.data.client.oauth.http.aclose()
        runtime.data.client.oauth.http = httpx.AsyncClient(transport=httpx.MockTransport(exchange))
        application = create_dashboard_app(runtime)
        transport = httpx.ASGITransport(application, client=("127.0.0.1", 123))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://stocker.example",
            headers={"X-Stocker-Proxy-Token": token},
        ) as client:
            navigation = {
                "Origin": "null",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Dest": "document",
            }
            # HTML navigation exceptions must not admit cross-origin writes, API reads,
            # fetches or embedded documents, even with valid dashboard authentication.
            for method, url, headers in [
                ("POST", "/oauth/saxo/start", {"Origin": "null"}),
                ("POST", "/oauth/saxo/callback", navigation),
                ("POST", "/api/entries/resume", navigation),
                ("GET", "/api/system", navigation),
                ("GET", "/system", {**navigation, "Sec-Fetch-Mode": "cors"}),
                ("GET", "/system", {**navigation, "Sec-Fetch-Dest": "iframe"}),
            ]:
                assert (await client.request(method, url, headers=headers)).status_code == 403
            for url in ["/oauth/saxo/callback", "/system"]:
                assert (
                    await client.get(url, headers={**navigation, "X-Stocker-Proxy-Token": ""})
                ).status_code == 403
                assert (
                    await client.get(url, headers={**navigation, "Host": "evil.test"})
                ).status_code == 403
            # State and browser binding, not the redirected Origin, authenticate the grant.
            for invalid in ["state", "binding"]:
                start = await client.post("/oauth/saxo/start")
                assert start.status_code == 303
                state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
                if invalid == "binding":
                    client.cookies.clear()
                failed = await client.get(
                    "/oauth/saxo/callback",
                    params={
                        "state": "wrong" if invalid == "state" else state,
                        "code": "fixture-code",
                    },
                    headers=navigation,
                )
                assert failed.status_code == 400 and exchanges == []
            start = await client.post(
                "/oauth/saxo/start",
                headers={"Origin": "https://stocker.example", "Accept": "application/json"},
            )
            assert start.status_code == 200
            assert start.headers["cache-control"] == "no-store"
            assert start.headers["referrer-policy"] == "no-referrer"
            cookie = start.headers["set-cookie"].lower()
            assert all(
                part in cookie
                for part in ["secure", "httponly", "samesite=lax", "path=/oauth/saxo"]
            )
            assert "fixture-secret" not in start.text
            location = urlsplit(start.json()["authorization_url"])
            assert (location.scheme, location.netloc, location.path) == (
                "https",
                "sim.logonvalidation.net",
                "/authorize",
            )
            state = parse_qs(location.query)["state"][0]
            callback = await client.get(
                "/oauth/saxo/callback",
                params={"state": state, "code": "fixture-code"},
                headers=navigation,
            )
            assert callback.status_code == 303, callback.text
            assert exchanges == ["/token"]
            landing = await client.get(callback.headers["location"], headers=navigation)
            assert landing.status_code == 200, landing.text
            replay = await client.get(
                "/oauth/saxo/callback",
                params={"state": state, "code": "fixture-code"},
                headers=navigation,
            )
            assert replay.status_code == 400 and exchanges == ["/token"]
            assert not runtime.broker.armed and not runtime.data.account_verified
        await runtime.stop()
        runtime.store.db.close()

    asyncio.run(scenario())
