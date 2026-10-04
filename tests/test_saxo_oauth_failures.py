"""Exchange failures are actionable without exposing broker payloads or credentials."""

import asyncio
import time

import httpx
import pytest

from stocker_execution.config import SaxoSettings
from stocker_execution.saxo_auth import OAuth, SaxoError, atomic_json
from stocker_execution.saxo_client import SaxoClient

CREDENTIALS = {
    "environment": "SAXO_SIM",
    "client_id": "fixture-key",
    "client_secret": "fixture-secret",
}


def tokens(expires_in=1000):
    return {
        "environment": "SAXO_SIM",
        "access_token": "fixture-old-access",
        "refresh_token": "fixture-keep-refresh",
        "expires_at": time.time() + expires_in,
        "refresh_expires_at": time.time() + 3600,
    }


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
        (201, {"access_token": "sensitive-access"}, "OAUTH_TOKEN_RESPONSE_INCOMPLETE"),
        (201, "sensitive-malformed-response", "OAUTH_TOKEN_RESPONSE_INVALID"),
        (204, "", "OAUTH_TOKEN_HTTP_204"),
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


def test_transport_failure_during_refresh_keeps_the_refresh_token(tmp_path):
    """The request never reached Saxo, so the refresh token has not rotated."""

    async def scenario():
        path = tmp_path / "credentials.json"
        atomic_json(path, CREDENTIALS)
        failing = True

        def respond(request):
            if failing:
                raise httpx.ConnectError("sensitive-transport-details")
            return httpx.Response(
                200,
                json={
                    "access_token": "fixture-new-access",
                    "refresh_token": "fixture-new-refresh",
                    "expires_in": 1200,
                    "refresh_token_expires_in": 2400,
                },
            )

        auth = OAuth(
            "SAXO_SIM",
            SaxoSettings(credentials_file=path),
            tmp_path,
            transport=httpx.MockTransport(respond),
        )
        auth.tokens, auth.status = tokens(expires_in=30), "AUTHENTICATED"
        with pytest.raises(SaxoError, match="^OAUTH_TOKEN_NETWORK_ERROR$"):
            await auth.access_token()
        assert auth.tokens["refresh_token"] == "fixture-keep-refresh"
        assert auth.status == "AUTHENTICATED"
        assert auth.failure_reason == "OAUTH_TOKEN_NETWORK_ERROR"
        failing = False
        assert await auth.access_token() == "fixture-new-access"
        assert auth.failure_reason == "" and auth.token_file.exists()
        await auth.close()

    asyncio.run(scenario())


def test_rejected_access_token_forces_a_refresh_and_a_failed_refresh_stays_expired(tmp_path):
    """A 401 must not be erased by the next access_token() call a second later."""

    async def scenario():
        path = tmp_path / "credentials.json"
        atomic_json(path, CREDENTIALS)
        calls = []

        def respond(request):
            calls.append(request.url.path)
            if request.url.path.endswith("/token"):
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(401, text="denied")

        transport = httpx.MockTransport(respond)
        auth = OAuth("SAXO_SIM", SaxoSettings(credentials_file=path), tmp_path, transport)
        auth.tokens = tokens()
        client = SaxoClient(auth, transport=transport)
        with pytest.raises(SaxoError, match="^HTTP_401$"):
            await client.request("GET", "/root/v2/user")
        assert auth.status == "AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED"
        with pytest.raises(SaxoError, match="^OAUTH_INVALID_GRANT$"):
            await client.request("GET", "/root/v2/user")
        assert [c.rsplit("/", 1)[-1] for c in calls] == ["user", "token"]
        assert auth.status == "AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED" and not auth.tokens
        await client.close()

    asyncio.run(scenario())
