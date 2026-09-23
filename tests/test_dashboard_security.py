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
