import asyncio

import httpx

from stocker_mcp.server import build_http_app, doctor


def test_http_mcp_endpoint_requires_auth(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("STOCKER_MCP_TOKEN", "expected-token")
        app = build_http_app(host="127.0.0.1", port=8765, auth_token_env="STOCKER_MCP_TOKEN")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            missing = await client.get("/mcp")
            wrong = await client.get("/mcp", headers={"Authorization": "Bearer wrong-token"})
            valid = await client.get("/mcp", headers={"Authorization": "Bearer expected-token"})

        assert missing.status_code == 401
        assert wrong.status_code == 401
        assert valid.status_code != 401

    asyncio.run(run())


def test_http_default_bind_is_loopback(monkeypatch) -> None:
    monkeypatch.setenv("STOCKER_MCP_TOKEN", "expected-token")

    result = doctor(
        transport="http",
        host="127.0.0.1",
        port=8765,
        auth_token_env="STOCKER_MCP_TOKEN",
    )

    assert result["http"]["local_url"] == "http://127.0.0.1:8765/mcp"
    assert result["http"]["auth_env_var_set"] is True
    assert result["http"]["unsafe_bind_warning"] is None
    assert result["http"]["can_initialise"] is True


def test_http_public_bind_reports_warning(monkeypatch) -> None:
    monkeypatch.setenv("STOCKER_MCP_TOKEN", "expected-token")

    result = doctor(
        transport="http",
        host="0.0.0.0",
        port=8765,
        auth_token_env="STOCKER_MCP_TOKEN",
    )

    assert "public interface" in result["http"]["unsafe_bind_warning"]
