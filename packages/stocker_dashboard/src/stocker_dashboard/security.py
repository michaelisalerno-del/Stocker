"""Single-operator HTTP boundary, including an authenticated loopback proxy."""

from __future__ import annotations

import base64
import hmac
import os
from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# The page has no inline script or style; nothing is embedded and nothing embeds it.
HARDENING = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
}


class DashboardSecurity:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.password = os.environ.get("STOCKER_DASHBOARD_PASSWORD", "")
        self.proxy_token = os.environ.get("STOCKER_DASHBOARD_PROXY_TOKEN", "")
        self.origin = os.environ.get("STOCKER_DASHBOARD_ORIGIN", "").rstrip("/")
        if self.password and self.proxy_token:
            raise ValueError("Choose password or authenticated proxy mode, not both")
        self.protected = bool(self.password or self.proxy_token)
        origin = urlsplit(self.origin)
        self.origin_host = origin.netloc
        if self.protected and (
            len(self.password or self.proxy_token) < 24
            or origin.scheme != "https"
            or not origin.netloc
            or origin.path
            or origin.query
            or origin.fragment
            or origin.username
        ):
            raise ValueError(
                "Protected dashboard requires a 24+ character credential and HTTPS origin"
            )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # the dashboard serves HTTP only
            await self.app(scope, receive, send)
            return

        async def hardened(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                for name, value in HARDENING.items():
                    response_headers.setdefault(name, value)
            await send(message)

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
        if self.protected and host != self.origin_host:
            denied = (403, "Unexpected dashboard host")
        oauth_callback = (
            scope.get("path") == "/oauth/saxo/callback" and scope.get("method") == "GET"
        )
        document_navigation = (
            scope.get("method") == "GET"
            and scope.get("path") in {"/", "/markets", "/opportunities", "/execution", "/system"}
            and headers.get(b"sec-fetch-mode") == b"navigate"
            and headers.get(b"sec-fetch-dest") == b"document"
        )
        # OAuth redirects can carry a null Origin through the landing navigation.
        # The callback validates single-use state and its browser-bound cookie.
        # These exceptions never bypass authentication/host checks or cover APIs/writes.
        cross_site_navigation = oauth_callback or document_navigation
        if origin and origin != allowed_origin and not cross_site_navigation:
            denied = (403, "Cross-origin access rejected")
        if headers.get(b"sec-fetch-site") == b"cross-site" and not cross_site_navigation:
            denied = (403, "Cross-site access rejected")
        if denied:
            response = JSONResponse(
                {"detail": denied[1]},
                status_code=denied[0],
                headers={"WWW-Authenticate": 'Basic realm="SLRNO", charset="UTF-8"'}
                if denied[0] == 401
                else {},
            )
            await response(scope, receive, hardened)
            return
        await self.app(scope, receive, hardened)
