"""Minimal OAuth 2.1-compatible ASGI wrapper for Stocker MCP HTTP mode."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import secrets
import time
from dataclasses import dataclass, field
from hmac import compare_digest
from typing import Any
from urllib.parse import parse_qs, urlencode

from stocker_mcp.security import StockerMCPContext

ACCESS_TOKEN_TTL_SECONDS = 12 * 60 * 60
AUTH_CODE_TTL_SECONDS = 10 * 60
DEFAULT_SCOPE = "stocker.read"
PROTECTED_RESOURCE_METADATA_PATHS = {
    "/.well-known/oauth-protected-resource",
    "/.well-known/oauth-protected-resource/mcp",
    "/mcp/.well-known/oauth-protected-resource",
}
AUTHORIZATION_SERVER_METADATA_PATHS = {
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-authorization-server/mcp",
    "/mcp/.well-known/oauth-authorization-server",
    "/.well-known/openid-configuration",
    "/.well-known/openid-configuration/mcp",
    "/mcp/.well-known/openid-configuration",
}


@dataclass
class OAuthClient:
    """Dynamically registered OAuth public client."""

    client_id: str
    redirect_uris: list[str]
    client_name: str | None = None


@dataclass
class OAuthCode:
    """Pending authorization-code grant state."""

    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    scope: str
    expires_at: float
    used: bool = False


@dataclass
class OAuthToken:
    """Issued bearer-token state."""

    token: str
    client_id: str
    scope: str
    expires_at: float


@dataclass
class OAuthState:
    """In-memory OAuth state for one local MCP server process."""

    setup_code: str
    clients: dict[str, OAuthClient] = field(default_factory=dict)
    codes: dict[str, OAuthCode] = field(default_factory=dict)
    access_tokens: dict[str, OAuthToken] = field(default_factory=dict)
    refresh_tokens: dict[str, OAuthToken] = field(default_factory=dict)

    def register_client(self, payload: dict[str, Any]) -> OAuthClient:
        redirect_uris = payload.get("redirect_uris")
        if not isinstance(redirect_uris, list) or not redirect_uris:
            raise OAuthError("invalid_client_metadata", "redirect_uris is required", 400)
        safe_redirects = []
        for uri in redirect_uris:
            if not isinstance(uri, str) or not uri.startswith("https://"):
                raise OAuthError("invalid_client_metadata", "redirect_uris must be https URLs", 400)
            safe_redirects.append(uri)
        client_id = f"stocker-client-{secrets.token_urlsafe(24)}"
        client = OAuthClient(
            client_id=client_id,
            redirect_uris=safe_redirects,
            client_name=str(payload.get("client_name")) if payload.get("client_name") else None,
        )
        self.clients[client_id] = client
        return client

    def issue_code(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        scope: str,
    ) -> str:
        code = secrets.token_urlsafe(32)
        self.codes[code] = OAuthCode(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            scope=scope,
            expires_at=time.time() + AUTH_CODE_TTL_SECONDS,
        )
        return code

    def exchange_code(
        self,
        *,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> dict[str, Any]:
        stored = self.codes.get(code)
        if stored is None or stored.used or stored.expires_at < time.time():
            raise OAuthError("invalid_grant", "authorization code is invalid", 400)
        if stored.client_id != client_id or stored.redirect_uri != redirect_uri:
            raise OAuthError("invalid_grant", "authorization code context does not match", 400)
        if not compare_digest(stored.code_challenge, _pkce_challenge(code_verifier)):
            raise OAuthError("invalid_grant", "PKCE verification failed", 400)
        stored.used = True
        return self.issue_tokens(client_id=client_id, scope=stored.scope)

    def refresh(self, *, client_id: str, refresh_token: str) -> dict[str, Any]:
        stored = self.refresh_tokens.get(refresh_token)
        if stored is None or stored.client_id != client_id:
            raise OAuthError("invalid_grant", "refresh token is invalid", 400)
        return self.issue_tokens(client_id=client_id, scope=stored.scope)

    def issue_tokens(self, *, client_id: str, scope: str) -> dict[str, Any]:
        access_token = secrets.token_urlsafe(40)
        refresh_token = secrets.token_urlsafe(40)
        expires_at = time.time() + ACCESS_TOKEN_TTL_SECONDS
        access = OAuthToken(
            token=access_token,
            client_id=client_id,
            scope=scope,
            expires_at=expires_at,
        )
        refresh = OAuthToken(
            token=refresh_token,
            client_id=client_id,
            scope=scope,
            expires_at=expires_at,
        )
        self.access_tokens[access_token] = access
        self.refresh_tokens[refresh_token] = refresh
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL_SECONDS,
            "scope": scope,
        }

    def access_token_valid(self, token: str) -> bool:
        stored = self.access_tokens.get(token)
        return stored is not None and stored.expires_at >= time.time()


class OAuthError(ValueError):
    """OAuth JSON error with a response status."""

    def __init__(self, error: str, description: str, status: int) -> None:
        super().__init__(description)
        self.error = error
        self.description = description
        self.status = status


class OAuthMiddleware:
    """ASGI wrapper that adds OAuth metadata and bearer enforcement."""

    def __init__(self, app: Any, *, state: OAuthState, context: StockerMCPContext) -> None:
        self.app = app
        self.state = state
        self.context = context

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        try:
            if method == "OPTIONS":
                await self._empty(send, 204)
                return
            if path in PROTECTED_RESOURCE_METADATA_PATHS:
                await self._json(send, self._resource_metadata(scope))
                return
            if path in AUTHORIZATION_SERVER_METADATA_PATHS:
                await self._json(send, self._authorization_server_metadata(scope))
                return
            if path == "/oauth/register" and method == "POST":
                await self._register(scope, receive, send)
                return
            if path == "/oauth/authorize" and method in {"GET", "POST"}:
                await self._authorize(scope, receive, send)
                return
            if path == "/oauth/token" and method == "POST":
                await self._token(receive, send)
                return
            if path == "/oauth/revoke" and method == "POST":
                await self._empty(send, 200)
                return
            if path.startswith("/mcp"):
                if not self._request_authorized(scope):
                    await self._unauthorized(scope, send, method, path)
                    return
                if method == "GET" and path == "/mcp" and not self._has_session_header(scope):
                    await self._health(send, method, path)
                    return
            await self.app(scope, receive, self._logging_send(send, method, path))
        except OAuthError as exc:
            await self._json(
                send,
                {"error": exc.error, "error_description": exc.description},
                status=exc.status,
            )

    def _request_authorized(self, scope: dict[str, Any]) -> bool:
        authorization = _headers(scope).get("authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return False
        return self.state.access_token_valid(authorization[len(prefix) :])

    def _has_session_header(self, scope: dict[str, Any]) -> bool:
        return "mcp-session-id" in _headers(scope)

    def _resource_metadata(self, scope: dict[str, Any]) -> dict[str, Any]:
        base = _base_url(scope)
        return {
            "resource": f"{base}/mcp",
            "resource_name": "Stocker Research",
            "authorization_servers": [base],
            "bearer_methods_supported": ["header"],
            "scopes_supported": [DEFAULT_SCOPE],
        }

    def _authorization_server_metadata(self, scope: dict[str, Any]) -> dict[str, Any]:
        base = _base_url(scope)
        return {
            "issuer": base,
            "authorization_endpoint": f"{base}/oauth/authorize",
            "token_endpoint": f"{base}/oauth/token",
            "registration_endpoint": f"{base}/oauth/register",
            "revocation_endpoint": f"{base}/oauth/revoke",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": [DEFAULT_SCOPE],
        }

    async def _register(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        body = await _body(receive)
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise OAuthError("invalid_request", "registration payload must be JSON", 400) from exc
        client = self.state.register_client(payload)
        self.context.log_tool_call("oauth_register", {"client_name": client.client_name})
        await self._json(
            send,
            {
                "client_id": client.client_id,
                "client_id_issued_at": int(time.time()),
                "redirect_uris": client.redirect_uris,
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "client_name": client.client_name,
            },
            status=201,
        )

    async def _authorize(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        params = _query_params(scope)
        if str(scope.get("method")) == "POST":
            params.update(_form_params(await _body(receive)))
        request = _AuthorizationRequest(params=params, oauth_state=self.state)
        if str(scope.get("method")) == "GET":
            await self._html(send, _approval_page(request))
            return
        setup_code = params.get("setup_code", "")
        if not compare_digest(setup_code, self.state.setup_code):
            self.context.log_tool_call("oauth_authorize", {"approved": False})
            await self._html(send, "<h1>Forbidden</h1>", status=403)
            return
        code = self.state.issue_code(
            client_id=request.client_id,
            redirect_uri=request.redirect_uri,
            code_challenge=request.code_challenge,
            scope=request.scope,
        )
        self.context.log_tool_call("oauth_authorize", {"approved": True})
        query = {"code": code}
        if request.request_state:
            query["state"] = request.request_state
        location = f"{request.redirect_uri}?{urlencode(query)}"
        await self._redirect(send, location)

    async def _token(self, receive: Any, send: Any) -> None:
        params = _form_params(await _body(receive))
        grant_type = params.get("grant_type")
        client_id = params.get("client_id", "")
        if client_id not in self.state.clients:
            raise OAuthError("invalid_client", "client_id is invalid", 401)
        if grant_type == "authorization_code":
            payload = self.state.exchange_code(
                code=params.get("code", ""),
                client_id=client_id,
                redirect_uri=params.get("redirect_uri", ""),
                code_verifier=params.get("code_verifier", ""),
            )
        elif grant_type == "refresh_token":
            payload = self.state.refresh(
                client_id=client_id,
                refresh_token=params.get("refresh_token", ""),
            )
        else:
            raise OAuthError("unsupported_grant_type", "grant_type is unsupported", 400)
        self.context.log_tool_call("oauth_token", {"grant_type": grant_type})
        await self._json(send, payload)

    def _logging_send(self, send: Any, method: str, path: str) -> Any:
        async def wrapped(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                self.context.log_tool_call(
                    "http_request",
                    {"method": method, "path": path, "status_code": message.get("status")},
                )
            await send(message)

        return wrapped

    async def _unauthorized(
        self,
        scope: dict[str, Any],
        send: Any,
        method: str,
        path: str,
    ) -> None:
        self.context.log_tool_call(
            "http_request",
            {"method": method, "path": path, "status_code": 401},
        )
        resource_metadata = f"{_base_url(scope)}/.well-known/oauth-protected-resource"
        await self._json(
            send,
            {"error": "unauthorized"},
            status=401,
            extra_headers=[
                (
                    b"www-authenticate",
                    f'Bearer resource_metadata="{resource_metadata}"'.encode(),
                )
            ],
        )

    async def _health(self, send: Any, method: str, path: str) -> None:
        self.context.log_tool_call(
            "http_request",
            {"method": method, "path": path, "status_code": 200},
        )
        await self._json(send, {"status": "ok", "path": path})

    async def _json(
        self,
        send: Any,
        payload: dict[str, Any],
        *,
        status: int = 200,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
        ]
        headers.extend(extra_headers or [])
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def _html(self, send: Any, body_text: str, *, status: int = 200) -> None:
        body = body_text.encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/html; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _redirect(self, send: Any, location: str) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 302,
                "headers": [(b"location", location.encode("utf-8"))],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    async def _empty(self, send: Any, status: int) -> None:
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b""})


@dataclass(frozen=True)
class _AuthorizationRequest:
    params: dict[str, str]
    oauth_state: OAuthState

    def __post_init__(self) -> None:
        if self.params.get("response_type") != "code":
            raise OAuthError("unsupported_response_type", "response_type must be code", 400)
        client_id = self.client_id
        client = self.oauth_state.clients.get(client_id)
        if client is None:
            raise OAuthError("invalid_client", "client_id is invalid", 400)
        if self.redirect_uri not in client.redirect_uris:
            raise OAuthError("invalid_request", "redirect_uri is not registered", 400)
        if not self.code_challenge:
            raise OAuthError("invalid_request", "code_challenge is required", 400)
        if self.params.get("code_challenge_method") != "S256":
            raise OAuthError("invalid_request", "code_challenge_method must be S256", 400)

    @property
    def client_id(self) -> str:
        return self.params.get("client_id", "")

    @property
    def redirect_uri(self) -> str:
        return self.params.get("redirect_uri", "")

    @property
    def code_challenge(self) -> str:
        return self.params.get("code_challenge", "")

    @property
    def scope(self) -> str:
        return self.params.get("scope") or DEFAULT_SCOPE

    @property
    def request_state(self) -> str:
        return self.params.get("state", "")


def _approval_page(request: _AuthorizationRequest) -> str:
    hidden = "\n".join(
        f'<input type="hidden" name="{html.escape(key)}" value="{html.escape(value)}">'
        for key, value in request.params.items()
        if key != "setup_code"
    )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Stocker OAuth Approval</title></head>
<body>
<h1>Stocker OAuth Approval</h1>
<p>Approve read-only ChatGPT access to Stocker MCP tools.</p>
<p>This does not enable trading, broker actions, file writes, or secret access.</p>
<form method="post" action="/oauth/authorize">
{hidden}
<label>Setup code <input name="setup_code" type="password" autofocus></label>
<button type="submit">Approve Stocker Research</button>
</form>
</body>
</html>"""


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _headers(scope: dict[str, Any]) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }


def _base_url(scope: dict[str, Any]) -> str:
    headers = _headers(scope)
    proto = headers.get("x-forwarded-proto") or str(scope.get("scheme") or "http")
    host = headers.get("x-forwarded-host") or headers.get("host") or "127.0.0.1"
    return f"{proto}://{host}"


def _query_params(scope: dict[str, Any]) -> dict[str, str]:
    raw = scope.get("query_string", b"")
    parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def _form_params(body: bytes) -> dict[str, str]:
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


async def _body(receive: Any) -> bytes:
    chunks: list[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        chunks.append(message.get("body", b""))
        more_body = bool(message.get("more_body", False))
    return b"".join(chunks)
