"""Authenticated futures views, independent of position management."""

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
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

    @app.get("/api/system")
    async def system() -> dict[str, Any]:
        return {
            **runtime.status(),
            "configuration": runtime.config.model_dump(),
            "markets": [
                {"market": s.market, "problem": s.problem, "reference_sessions": len(s.references)}
                for s in runtime.markets.values()
            ],
            "broker_positions": [
                dict(r)
                for r in runtime.store.db.execute(
                    "SELECT * FROM positions WHERE quantity<>0 ORDER BY con_id LIMIT 100"
                )
            ],
        }

    @app.get("/api/history")
    async def history(
        market: str | None = None,
        day: date | None = None,
        version: str | None = Query(None, max_length=80),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        if market is not None and market not in MARKETS:
            raise HTTPException(422, "Unknown futures market")
        rows = runtime.store.history(market, day.isoformat() if day else None, version, offset)
        return {"rows": rows, "offset": offset, "has_more": len(rows) == 100}

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

    @app.get("/{page:path}")
    async def page(page: str) -> Any:
        if page not in {"", "trades", "system"}:
            raise HTTPException(404, "No such SLRNO page")
        return FileResponse(static / "index.html")

    return app
