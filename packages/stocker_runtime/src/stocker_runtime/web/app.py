"""FastAPI application exposing only the generic Stocker V2 read model."""

from __future__ import annotations

import os
import secrets
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as ApiPath
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from stocker_runtime.web.config import WebConfig
from stocker_runtime.web.queries import (
    SQLITE_INTEGER_MAX,
    CursorError,
    QueryTimeoutError,
    ReadModel,
    WindowError,
)


class ResponseTooLargeError(RuntimeError):
    """A serialized JSON response exceeded the fixed 512 KiB ceiling."""


def create_web_app(config: WebConfig) -> FastAPI:
    """Create a GET-only web app with no recorder or broker object."""

    authentication_token: str | None = None
    if config.authentication_enabled:
        assert config.auth_token_env is not None
        authentication_token = os.environ.get(config.auth_token_env)
        if not authentication_token:
            raise RuntimeError(
                "blocked_unsafe_runtime_configuration: web authentication token is absent"
            )

    class BoundedJSONResponse(JSONResponse):
        def render(self, content: Any) -> bytes:
            rendered = super().render(content)
            if len(rendered) > config.maximum_response_bytes:
                raise ResponseTooLargeError("response_too_large")
            return rendered

    read_model = ReadModel(config)
    static_root = Path(__file__).with_name("static")
    app = FastAPI(
        title="Stocker V2",
        version=config.app_version,
        docs_url=None if config.production else "/docs",
        redoc_url=None,
        default_response_class=BoundedJSONResponse,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.allowed_hosts)
    app.mount("/assets", StaticFiles(directory=static_root), name="assets")
    rate_windows: OrderedDict[str, deque[float]] = OrderedDict()
    maximum_rate_limit_identities = 4_096

    def security_headers(request_id: str) -> dict[str, str]:
        return {
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-Correlation-ID": request_id,
            "X-Request-ID": request_id,
        }

    @app.middleware("http")
    async def security_boundary(request: Request, call_next: Any) -> Any:
        supplied_request_id = request.headers.get(
            "x-request-id", request.headers.get("x-correlation-id", "")
        )
        request_id = (
            supplied_request_id
            if supplied_request_id
            and len(supplied_request_id) <= 128
            and all(character.isalnum() or character in "-_." for character in supplied_request_id)
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id
        response: Any | None = None
        if request.url.path.startswith("/api/") and request.method not in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            response = JSONResponse(status_code=404, content={"detail": "not_found"})
        if response is None and authentication_token is not None:
            authorization = request.headers.get("authorization", "")
            bearer = (
                authorization.removeprefix("Bearer ").strip()
                if authorization.startswith("Bearer ")
                else ""
            )
            supplied = bearer or request.cookies.get(config.auth_cookie_name, "")
            if not supplied or not secrets.compare_digest(supplied, authentication_token):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "authentication_required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        if response is None:
            client_ip = "unknown" if request.client is None else request.client.host
            if (
                config.trust_proxy_headers
                and client_ip in config.trusted_proxy_ips
                and request.headers.get("x-forwarded-for")
            ):
                client_ip = request.headers["x-forwarded-for"].split(",", 1)[0].strip()
            now = time.monotonic()
            window = rate_windows.setdefault(client_ip, deque())
            rate_windows.move_to_end(client_ip)
            while len(rate_windows) > maximum_rate_limit_identities:
                rate_windows.popitem(last=False)
            while window and window[0] <= now - 60:
                window.popleft()
            if len(window) >= config.requests_per_minute:
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "rate_limit_exceeded"},
                )
            else:
                window.append(now)
        if response is None:
            response = await call_next(request)
        response.headers.update(security_headers(request_id))
        return response

    @app.exception_handler(CursorError)
    async def invalid_cursor(_request: Request, _error: CursorError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "invalid_cursor"})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "invalid_request"})

    @app.exception_handler(WindowError)
    async def invalid_window(_request: Request, _error: WindowError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "invalid_window"})

    @app.exception_handler(QueryTimeoutError)
    async def query_timeout(_request: Request, _error: QueryTimeoutError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "query_timeout"})

    @app.exception_handler(ResponseTooLargeError)
    async def response_too_large(_request: Request, _error: ResponseTooLargeError) -> JSONResponse:
        return JSONResponse(status_code=413, content={"detail": "response_too_large"})

    @app.exception_handler(Exception)
    async def internal_error(request: Request, _error: Exception) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
        response = JSONResponse(status_code=500, content={"detail": "internal_error"})
        response.headers.update(security_headers(request_id))
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/v2/meta")
    def meta() -> dict[str, Any]:
        return read_model.meta()

    @app.get("/api/v2/live")
    def live(
        feed_kind: str | None = Query(
            default=None,
            min_length=1,
            max_length=128,
            pattern=r".*\S.*",
        ),
        limit: int = Query(default=50, ge=1, le=200),
        cursor: str | None = Query(default=None, max_length=2_048),
    ) -> dict[str, Any]:
        return read_model.live(feed_kind=feed_kind, limit=limit, cursor=cursor)

    @app.get("/api/v2/ideas")
    def ideas(
        health: Literal["healthy", "degraded", "disabled"] | None = None,
        mode: Literal["prospective_record", "shadow"] | None = None,
        active: bool | None = None,
        limit: int = Query(default=50, ge=1, le=100),
        cursor: str | None = Query(default=None, max_length=2_048),
    ) -> dict[str, Any]:
        return read_model.ideas(
            health=health,
            mode=mode,
            active=active,
            limit=limit,
            cursor=cursor,
        )

    @app.get("/api/v2/ideas/{instance_id}")
    def idea_detail(
        instance_id: str = ApiPath(min_length=1, max_length=512),
        kind: Literal["observation", "signal", "proposed_position", "proposed_trade"] | None = None,
        start_us: int | None = Query(default=None, ge=0, le=SQLITE_INTEGER_MAX),
        end_us: int | None = Query(default=None, ge=0, le=SQLITE_INTEGER_MAX),
        limit: int = Query(default=50, ge=1, le=200),
        cursor: str | None = Query(default=None, max_length=2_048),
    ) -> dict[str, Any]:
        projection = read_model.idea_detail(
            instance_id,
            kind=kind,
            start_us=start_us,
            end_us=end_us,
            limit=limit,
            cursor=cursor,
        )
        if projection is None:
            raise HTTPException(status_code=404, detail="not_found")
        return projection

    @app.get("/api/v2/results")
    def results(
        status: Literal["open", "closed", "incomplete", "invalid"] | None = None,
        instance_id: str | None = Query(default=None, min_length=1, max_length=512),
        start_us: int | None = Query(default=None, ge=0, le=SQLITE_INTEGER_MAX),
        end_us: int | None = Query(default=None, ge=0, le=SQLITE_INTEGER_MAX),
        limit: int = Query(default=50, ge=1, le=200),
        cursor: str | None = Query(default=None, max_length=2_048),
    ) -> dict[str, Any]:
        return read_model.results(
            status=status,
            instance_id=instance_id,
            start_us=start_us,
            end_us=end_us,
            limit=limit,
            cursor=cursor,
        )

    @app.get("/api/v2/results/{position_id}")
    def result_detail(
        position_id: str = ApiPath(min_length=1, max_length=512),
        start_us: int | None = Query(default=None, ge=0, le=SQLITE_INTEGER_MAX),
        end_us: int | None = Query(default=None, ge=0, le=SQLITE_INTEGER_MAX),
        limit: int = Query(default=50, ge=1, le=200),
        cursor: str | None = Query(default=None, max_length=2_048),
    ) -> dict[str, Any]:
        projection = read_model.result_detail(
            position_id,
            start_us=start_us,
            end_us=end_us,
            limit=limit,
            cursor=cursor,
        )
        if projection is None:
            raise HTTPException(status_code=404, detail="not_found")
        return projection

    @app.get("/api/v2/diagnostics")
    def diagnostics(
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        return read_model.diagnostics(limit=limit)

    return app
