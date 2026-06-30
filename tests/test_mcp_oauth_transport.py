import asyncio
import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import httpx

from stocker_mcp.server import build_http_app, connector_info, doctor

CHATGPT_CALLBACK = "https://chatgpt.com/connector/oauth/LcDotGjs8ocx"


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def test_oauth_metadata_and_registration(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("STOCKER_OAUTH_SETUP_CODE", "setup-code")
        app = build_http_app(
            host="127.0.0.1",
            port=8765,
            auth_mode="oauth",
            oauth_setup_code_env="STOCKER_OAUTH_SETUP_CODE",
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://stocker.test",
        ) as client:
            resource = await client.get("/.well-known/oauth-protected-resource")
            resource_for_mcp = await client.get("/.well-known/oauth-protected-resource/mcp")
            nested_resource = await client.get("/mcp/.well-known/oauth-protected-resource")
            auth_server = await client.get("/.well-known/oauth-authorization-server")
            auth_server_for_mcp = await client.get("/.well-known/oauth-authorization-server/mcp")
            nested_auth_server = await client.get("/mcp/.well-known/oauth-authorization-server")
            openid = await client.get("/.well-known/openid-configuration")
            openid_for_mcp = await client.get("/.well-known/openid-configuration/mcp")
            nested_openid = await client.get("/mcp/.well-known/openid-configuration")
            registered = await client.post(
                "/oauth/register",
                json={
                    "redirect_uris": [CHATGPT_CALLBACK],
                    "client_name": "ChatGPT",
                    "token_endpoint_auth_method": "none",
                },
            )

        assert resource.status_code == 200
        assert resource.json()["authorization_servers"] == ["https://stocker.test"]
        assert resource_for_mcp.status_code == 200
        assert nested_resource.status_code == 200
        assert auth_server.status_code == 200
        assert auth_server.json()["authorization_endpoint"] == (
            "https://stocker.test/oauth/authorize"
        )
        assert auth_server.json()["registration_endpoint"] == "https://stocker.test/oauth/register"
        assert "authorization_code" in auth_server.json()["grant_types_supported"]
        assert auth_server_for_mcp.status_code == 200
        assert nested_auth_server.status_code == 200
        assert openid.status_code == 200
        assert openid_for_mcp.status_code == 200
        assert nested_openid.status_code == 200
        assert registered.status_code == 201
        assert registered.json()["client_id"]
        assert registered.json()["token_endpoint_auth_method"] == "none"

    asyncio.run(run())


def test_oauth_authorization_code_flow_allows_mcp(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("STOCKER_OAUTH_SETUP_CODE", "setup-code")
        app = build_http_app(
            host="127.0.0.1",
            port=8765,
            auth_mode="oauth",
            oauth_setup_code_env="STOCKER_OAUTH_SETUP_CODE",
        )
        transport = httpx.ASGITransport(app=app)
        verifier = "test-verifier-for-stocker-oauth"
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://stocker.test",
        ) as client:
            registered = await client.post(
                "/oauth/register",
                json={"redirect_uris": [CHATGPT_CALLBACK]},
            )
            client_id = registered.json()["client_id"]
            params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": CHATGPT_CALLBACK,
                "state": "abc",
                "scope": "stocker.read",
                "code_challenge": _challenge(verifier),
                "code_challenge_method": "S256",
            }

            consent = await client.get("/oauth/authorize", params=params)
            wrong_setup_code = await client.post(
                "/oauth/authorize",
                data={**params, "setup_code": "wrong"},
            )
            approved = await client.post(
                "/oauth/authorize",
                data={**params, "setup_code": "setup-code"},
                follow_redirects=False,
            )

            redirect = approved.headers["location"]
            parsed = urlparse(redirect)
            query = parse_qs(parsed.query)
            token = await client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "redirect_uri": CHATGPT_CALLBACK,
                    "code": query["code"][0],
                    "code_verifier": verifier,
                },
            )
            payload = token.json()
            missing = await client.get("/mcp")
            authenticated = await client.get(
                "/mcp",
                headers={"Authorization": f"Bearer {payload['access_token']}"},
            )

        assert consent.status_code == 200
        assert "Stocker OAuth Approval" in consent.text
        assert wrong_setup_code.status_code == 403
        assert approved.status_code == 302
        assert parsed.scheme == "https"
        assert parsed.netloc == "chatgpt.com"
        assert query["state"] == ["abc"]
        assert token.status_code == 200
        assert payload["token_type"] == "Bearer"
        assert payload["access_token"]
        assert payload["refresh_token"]
        assert missing.status_code == 401
        assert "oauth-protected-resource" in missing.headers["www-authenticate"]
        assert authenticated.status_code == 200

    asyncio.run(run())


def test_oauth_adopts_cached_chatgpt_client_id_after_restart(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("STOCKER_OAUTH_SETUP_CODE", "setup-code")
        app = build_http_app(
            host="127.0.0.1",
            port=8765,
            auth_mode="oauth",
            oauth_setup_code_env="STOCKER_OAUTH_SETUP_CODE",
        )
        transport = httpx.ASGITransport(app=app)
        verifier = "cached-client-verifier-for-stocker-oauth"
        client_id = "stocker-client-FzVIzLIb69CVU9kMj-cE1rRf-2Ojz8wx"
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": CHATGPT_CALLBACK,
            "scope": "stocker.read",
            "code_challenge": _challenge(verifier),
            "code_challenge_method": "S256",
            "state": "cached-client",
        }
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://stocker.test",
        ) as client:
            approval = await client.get("/oauth/authorize", params=params)
            approved = await client.post(
                "/oauth/authorize",
                data={**params, "setup_code": "setup-code"},
                follow_redirects=False,
            )
            code = parse_qs(urlparse(approved.headers["location"]).query)["code"][0]
            token = await client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "redirect_uri": CHATGPT_CALLBACK,
                    "code": code,
                    "code_verifier": verifier,
                },
            )

        assert approval.status_code == 200
        assert approved.status_code == 302
        assert token.status_code == 200
        assert token.json()["access_token"]

    asyncio.run(run())


def test_oauth_refresh_token_flow(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("STOCKER_OAUTH_SETUP_CODE", "setup-code")
        app = build_http_app(
            host="127.0.0.1",
            port=8765,
            auth_mode="oauth",
            oauth_setup_code_env="STOCKER_OAUTH_SETUP_CODE",
        )
        transport = httpx.ASGITransport(app=app)
        verifier = "another-test-verifier-for-stocker-oauth"
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://stocker.test",
        ) as client:
            registered = await client.post(
                "/oauth/register",
                json={"redirect_uris": [CHATGPT_CALLBACK]},
            )
            client_id = registered.json()["client_id"]
            params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": CHATGPT_CALLBACK,
                "state": "state",
                "code_challenge": _challenge(verifier),
                "code_challenge_method": "S256",
            }
            approved = await client.post(
                "/oauth/authorize",
                data={**params, "setup_code": "setup-code"},
                follow_redirects=False,
            )
            code = parse_qs(urlparse(approved.headers["location"]).query)["code"][0]
            token = await client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "redirect_uri": CHATGPT_CALLBACK,
                    "code": code,
                    "code_verifier": verifier,
                },
            )
            refreshed = await client.post(
                "/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": token.json()["refresh_token"],
                },
            )
            authenticated = await client.get(
                "/mcp",
                headers={"Authorization": f"Bearer {refreshed.json()['access_token']}"},
            )

        assert refreshed.status_code == 200
        assert refreshed.json()["token_type"] == "Bearer"
        assert authenticated.status_code == 200

    asyncio.run(run())


def test_oauth_diagnostics_exclude_setup_code(monkeypatch) -> None:
    monkeypatch.setenv("STOCKER_OAUTH_SETUP_CODE", "setup-code")

    info = connector_info(auth_mode="oauth", oauth_setup_code_env="STOCKER_OAUTH_SETUP_CODE")
    diagnostics = doctor(
        transport="http",
        auth_mode="oauth",
        oauth_setup_code_env="STOCKER_OAUTH_SETUP_CODE",
    )

    assert info["auth"]["type"] == "oauth"
    assert info["auth"]["setup_code_env_var_set"] is True
    assert "setup-code" not in str(info)
    assert diagnostics["http"]["auth_mode"] == "oauth"
    assert diagnostics["http"]["oauth_setup_code_env_var_set"] is True
    assert "setup-code" not in str(diagnostics)
