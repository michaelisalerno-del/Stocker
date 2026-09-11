"""Single-operator HTTP boundary, including an authenticated loopback proxy."""

from __future__ import annotations

import base64
import hmac
import os
from urllib.parse import urlsplit

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class DashboardSecurity:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.password = os.environ.get("STOCKER_DASHBOARD_PASSWORD", "")
        self.proxy_token = os.environ.get("STOCKER_DASHBOARD_PROXY_TOKEN", "")
        self.origin = os.environ.get("STOCKER_DASHBOARD_ORIGIN", "").rstrip("/")
        if self.password and self.proxy_token:
            raise ValueError("Choose password or authenticated proxy mode, not both")
        self.protected = bool(self.password or self.proxy_token)
        if self.protected and (
            len(self.password or self.proxy_token) < 24
            or urlsplit(self.origin).scheme != "https"
            or not urlsplit(self.origin).netloc
            or urlsplit(self.origin).path
            or urlsplit(self.origin).query
            or urlsplit(self.origin).fragment
            or urlsplit(self.origin).username
        ):
            raise ValueError(
                "Protected dashboard requires a 24+ character credential and HTTPS origin"
            )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        host = headers.get(b"host", b"").decode("latin1")
        client = (scope.get("client") or ("", 0))[0]
        origin = headers.get(b"origin", b"").decode("latin1").rstrip("/")
        allowed_origin = self.origin if self.protected else f"http://{host}"
        local = client in {"127.0.0.1", "::1"} and (
            urlsplit(f"http://{host}").hostname in {"127.0.0.1", "::1", "localhost"}
        )
        # Starlette's in-process test transport is not a TCP client.
        local = local or (client == "testclient" and host == "testserver")
        denied = None
        if self.proxy_token:
            supplied = headers.get(b"x-stocker-proxy-token", b"")
            if client not in {"127.0.0.1", "::1"} or not hmac.compare_digest(
                supplied, self.proxy_token.encode()
            ):
                denied = (403, "Authenticated proxy required")
        elif self.password:
            expected = base64.b64encode(f"stocker:{self.password}".encode())
            supplied = headers.get(b"authorization", b"")
            if not hmac.compare_digest(supplied, b"Basic " + expected):
                denied = (401, "Authentication required")
        elif not local:
            denied = (403, "Unauthenticated dashboard is loopback-only")
        if self.protected and host != urlsplit(self.origin).netloc:
            denied = (403, "Unexpected dashboard host")
        if origin and origin != allowed_origin:
            denied = (403, "Cross-origin access rejected")
        if headers.get(b"sec-fetch-site") == b"cross-site":
            denied = (403, "Cross-site access rejected")
        if denied:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                response = JSONResponse(
                    {"detail": denied[1]},
                    status_code=denied[0],
                    headers={"WWW-Authenticate": 'Basic realm="Stocker", charset="UTF-8"'}
                    if denied[0] == 401
                    else {},
                )
                await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
