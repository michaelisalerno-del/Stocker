import asyncio
import base64

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


def test_first4_server_preserves_authenticated_proxy_socket_peer(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    import httpx
    import uvicorn
    from typer.testing import CliRunner

    from stocker_core.cli import app as cli
    from stocker_execution.first4_runtime import Runtime

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
    config = tmp_path / "first4.yaml"
    config.write_text("armed: false\n")
    result = CliRunner().invoke(
        cli, ["first4-run", "--config", str(config), "--database", str(tmp_path / "state.sqlite")]
    )
    assert result.exit_code == 0, result.output
    assert statuses == [200, 200, 403]
