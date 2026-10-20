"""The paper record's pages: Today, Record and System. Display only; nothing here trades."""

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from stocker_dashboard.security import DashboardSecurity
from stocker_execution.runtime import Runtime

PAGES = {"", "record", "system"}
# Revalidate page assets so a deploy never pairs a stale script with new markup.
NO_CACHE = {"Cache-Control": "no-cache"}


class RevalidatedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


def create_dashboard_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="SLRNO — paper record")
    app.add_middleware(DashboardSecurity)
    static = Path(__file__).with_name("static")
    app.mount("/static", RevalidatedStaticFiles(directory=static), name="static")

    @app.exception_handler(sqlite3.Error)
    async def database_unavailable(request: Any, exc: sqlite3.Error) -> JSONResponse:
        # A failed display read never changes the record; the runtime keeps running.
        return JSONResponse(status_code=503, content={"error": "RECORD_UNAVAILABLE"})

    @app.get("/api/health")
    async def health() -> JSONResponse:
        ok = runtime.worker_health == "RUNNING"
        return JSONResponse(status_code=200 if ok else 503, content=runtime.status())

    @app.get("/api/today")
    async def today() -> dict[str, Any]:
        return runtime.today()

    @app.get("/api/record")
    async def record(offset: int = Query(0, ge=0)) -> dict[str, Any]:
        return {**runtime.store.record(offset), "offset": offset, "system": runtime.status()}

    @app.get("/api/system")
    async def system() -> dict[str, Any]:
        return runtime.status()

    @app.get("/api/charts")
    async def charts() -> dict[str, Any]:
        return runtime.charts()

    @app.post("/oauth/saxo/start")
    async def oauth_start(request: Request) -> Response:
        try:
            location, binding = runtime.client.oauth.begin()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        response: Response = (
            JSONResponse({"authorization_url": location})
            if "application/json" in request.headers.get("accept", "")
            else RedirectResponse(location, status_code=303)
        )
        response.set_cookie(
            "slrno_oauth_binding",
            binding,
            max_age=600,
            httponly=True,
            secure=runtime.config.saxo.redirect_uri.startswith("https:"),
            samesite="lax",
            path="/oauth/saxo",
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/oauth/saxo/callback")
    async def oauth_callback(request: Request) -> RedirectResponse:
        try:
            await runtime.client.oauth.callback(
                request.query_params.get("state", ""),
                request.cookies.get("slrno_oauth_binding", ""),
                request.query_params.get("code", ""),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        response = RedirectResponse("/system", status_code=303)
        response.delete_cookie("slrno_oauth_binding", path="/oauth/saxo")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/{page:path}")
    async def page(page: str) -> Any:
        if page not in PAGES:
            raise HTTPException(404, "No such SLRNO page")
        return FileResponse(static / "index.html", headers=NO_CACHE)

    return app
