"""Authenticated futures views, independent of position management."""

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from stocker_dashboard.security import DashboardSecurity
from stocker_execution.config import MARKETS
from stocker_execution.runtime import Runtime


def create_dashboard_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="SLRNO — Futures PAPER")
    app.add_middleware(DashboardSecurity)
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.exception_handler(sqlite3.Error)
    async def database_unavailable(request: Any, exc: sqlite3.Error) -> JSONResponse:
        runtime.broker.fatal_error = "LEDGER_UNAVAILABLE"
        runtime.broker.reconciled = False
        return JSONResponse(status_code=503, content={"error": "LEDGER_UNAVAILABLE"})

    @app.get("/api/health")
    async def health() -> JSONResponse:
        ok = (
            runtime.worker_health == runtime.manager_health == "RUNNING"
            and not runtime.broker.fatal_error
        )
        return JSONResponse(status_code=200 if ok else 503, content=runtime.status())

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return runtime.status()

    @app.get("/api/overview")
    async def overview() -> dict[str, Any]:
        return runtime.overview()

    @app.get("/api/market/{market}")
    async def market_detail(market: str, diagnostics: bool = False) -> dict[str, Any]:
        if market not in MARKETS:
            raise HTTPException(422, "Unknown futures market")
        return runtime.market_detail(market, diagnostics=diagnostics)

    @app.get("/api/execution")
    async def execution() -> dict[str, Any]:
        return runtime.execution_view()

    @app.get("/api/system")
    async def system(diagnostics: bool = False) -> dict[str, Any]:
        return {
            **runtime.status(),
            "configuration": runtime.config.model_dump(mode="json", exclude={"saxo"})
            if diagnostics
            else None,
            "markets": [
                {
                    "market": s.market,
                    "problem": s.problem,
                    "reference_sessions": len(s.references),
                    "capabilities": runtime.data.capability_view(s),
                    "candidates": s.candidates if diagnostics else None,
                }
                for s in runtime.markets.values()
            ],
        }

    @app.get("/api/history")
    async def history(
        market: str | None = None,
        day: date | None = None,
        version: str | None = Query(None, max_length=80),
        offset: int = Query(0, ge=0),
        sort: Literal["asc", "desc"] = "desc",
    ) -> dict[str, Any]:
        if market is not None and market not in MARKETS:
            raise HTTPException(422, "Unknown futures market")
        rows = runtime.store.history(
            market, day.isoformat() if day else None, version, offset, sort
        )
        return {
            "rows": rows,
            "offset": offset,
            "has_more": len(rows) == 100,
            "system": runtime.status(),
        }

    @app.get("/api/detail")
    async def detail(identity: str = Query(max_length=250)) -> dict[str, Any]:
        row = runtime.store.db.execute("SELECT * FROM signals WHERE id=?", (identity,)).fetchone()
        if not row:
            raise HTTPException(404, "Opportunity not found")
        return {
            "signal": dict(row),
            "orders": runtime.store.orders(identity),
            "fills": runtime.store.fills(identity),
            "inputs": json.loads(row["detail"]),
            "l2_observation": runtime.store.depth_summary(identity),
            "lifecycle": [
                dict(r)
                for r in runtime.store.db.execute(
                    "SELECT * FROM lifecycle WHERE reference=? OR reference IN "
                    "(SELECT reference FROM orders WHERE event_id=?) "
                    "ORDER BY sequence DESC LIMIT 100",
                    (identity, identity),
                )
            ],
        }

    @app.post("/api/entries/pause")
    async def pause() -> dict[str, Any]:
        runtime.pause = True
        return runtime.status()

    @app.post("/api/entries/resume")
    async def resume() -> dict[str, Any]:
        runtime.pause = False
        return runtime.status()

    @app.post("/api/paper/preflight")
    async def preflight() -> dict[str, Any]:
        try:
            return await runtime.broker.preflight()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/paper/arm")
    async def arm(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            runtime.broker.arm(payload.get("acknowledgement", ""))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return runtime.status()

    @app.post("/api/paper/disarm")
    async def disarm() -> dict[str, Any]:
        runtime.broker.armed = False
        return runtime.status()

    @app.post("/oauth/saxo/start")
    async def oauth_start(request: Request) -> Response:
        try:
            location, binding = runtime.data.client.oauth.begin()
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
            await runtime.data.client.oauth.callback(
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

    @app.get("/api/recordings")
    async def recordings() -> dict[str, Any]:
        return {
            "active": list(runtime.recorder.active.values()),
            "completed": runtime.recorder.catalog[-100:],
        }

    @app.get("/api/recordings/{segment}")
    async def recording(segment: str) -> FileResponse:
        if len(segment) != 64 or any(c not in "0123456789abcdef" for c in segment):
            raise HTTPException(404, "Unknown recording")
        path = runtime.recorder.directory / (segment + ".jsonl.gz")
        if not path.is_file():
            raise HTTPException(404, "Recording unavailable")
        return FileResponse(path, media_type="application/gzip", filename=path.name)

    @app.get("/api/recordings/{segment}/manifest")
    async def recording_manifest(segment: str) -> FileResponse:
        if len(segment) != 64 or any(c not in "0123456789abcdef" for c in segment):
            raise HTTPException(404, "Unknown recording")
        path = runtime.recorder.directory / (segment + ".manifest.json")
        if not path.is_file():
            raise HTTPException(404, "Manifest unavailable")
        return FileResponse(path, media_type="application/json", filename=path.name)

    @app.post("/api/recordings/{segment}/prune")
    async def prune_recording(segment: str) -> dict[str, Any]:
        referenced = {
            json.loads(r[0]).get("segment", "")
            for r in runtime.store.db.execute("SELECT summary FROM depth_captures")
        }
        try:
            # Completed, unreferenced, unprotected records only. Runs off the risk loop.
            import asyncio

            await asyncio.to_thread(runtime.recorder.prune, segment, referenced)
        except (OSError, ValueError):
            raise HTTPException(409, "ACTIVE_REFERENCED_PROTECTED_OR_UNAVAILABLE") from None
        return {"pruned": segment}

    @app.get("/{page:path}")
    async def page(page: str) -> Any:
        if page not in {"", "markets", "opportunities", "execution", "trades", "system"}:
            raise HTTPException(404, "No such SLRNO page")
        return FileResponse(static / "index.html")

    return app
