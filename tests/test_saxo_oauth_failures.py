"""Exchange failures are actionable without exposing broker payloads or credentials."""

import asyncio

import httpx
import pytest

from stocker_execution.config import SaxoSettings
from stocker_execution.saxo_auth import OAuth, atomic_json


@pytest.mark.parametrize(
    ("status", "body", "reason"),
    [
        (401, "<html>Untrusted sensitive response</html>", "OAUTH_TOKEN_HTTP_401"),
        (
            400,
            {"error": "invalid_grant", "error_description": "sensitive-code"},
            "OAUTH_INVALID_GRANT",
        ),
        (
            401,
            {"error": "invalid_client", "error_description": "sensitive-secret"},
            "OAUTH_INVALID_CLIENT",
        ),
        (400, {"error": "sensitive-value"}, "OAUTH_TOKEN_HTTP_400"),
        (200, {"access_token": "sensitive-access"}, "OAUTH_TOKEN_RESPONSE_INCOMPLETE"),
        (200, "sensitive-malformed-response", "OAUTH_TOKEN_RESPONSE_INVALID"),
    ],
)
def test_exchange_reports_safe_failure_category(tmp_path, status, body, reason):
    async def scenario():
        path = tmp_path / "credentials.json"
        atomic_json(
            path,
            {
                "environment": "SAXO_SIM",
                "client_id": "fixture-key",
                "client_secret": "fixture-secret",
            },
        )

        def respond(request):
            return (
                httpx.Response(status, json=body)
                if isinstance(body, dict)
                else httpx.Response(status, text=body)
            )

        auth = OAuth(
            "SAXO_SIM",
            SaxoSettings(credentials_file=path),
            tmp_path,
            transport=httpx.MockTransport(respond),
        )
        _, binding = auth.begin()
        with pytest.raises(ValueError, match=f"^{reason}$"):
            await auth.callback(auth.pending[0], binding, "fixture-code")
        assert auth.status == "AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED"
        assert auth.failure_reason == reason
        assert not auth.tokens and not auth.token_file.exists()
        assert auth.pending is None
        await auth.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["network", "storage"])
def test_local_exchange_failure_and_successful_reconnect(tmp_path, monkeypatch, failure):
    async def scenario():
        path = tmp_path / "credentials.json"
        atomic_json(
            path,
            {
                "environment": "SAXO_SIM",
                "client_id": "fixture-key",
                "client_secret": "fixture-secret",
            },
        )
        failing = True

        def respond(request):
            if failing and failure == "network":
                raise httpx.ConnectError("sensitive-transport-details")
            return httpx.Response(
                200,
                json={
                    "access_token": "fixture-access",
                    "refresh_token": "fixture-refresh",
                    "expires_in": 1200,
                    "refresh_token_expires_in": 2400,
                },
            )

        def save(file, tokens):
            if failing and failure == "storage":
                raise OSError("sensitive-filesystem-details")
            atomic_json(file, tokens)

        monkeypatch.setattr("stocker_execution.saxo_auth.atomic_json", save)
        auth = OAuth(
            "SAXO_SIM",
            SaxoSettings(credentials_file=path),
            tmp_path,
            transport=httpx.MockTransport(respond),
        )
        _, binding = auth.begin()
        reason = (
            "OAUTH_TOKEN_NETWORK_ERROR" if failure == "network" else "OAUTH_TOKEN_STORAGE_FAILED"
        )
        with pytest.raises(ValueError, match=f"^{reason}$"):
            await auth.callback(auth.pending[0], binding, "fixture-code")
        assert auth.failure_reason == reason and not auth.tokens
        assert not auth.token_file.exists()
        failing = False
        _, binding = auth.begin()
        await auth.callback(auth.pending[0], binding, "fixture-new-code")
        assert auth.status == "AUTHENTICATED" and auth.failure_reason == ""
        assert await auth.access_token() == "fixture-access"
        await auth.close()

    asyncio.run(scenario())
