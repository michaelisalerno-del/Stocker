"""Authenticated, bounded FIRST4 ledger views and independent health."""

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from stocker_dashboard.security import DashboardSecurity
from stocker_dashboard.views import allocations, economics, opportunities
from stocker_execution.first4 import Q5
from stocker_execution.first4_runtime import Runtime


def create_dashboard_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="SLRNO — FIRST4 / US PAPER", docs_url="/api/docs")
    app.add_middleware(DashboardSecurity)
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.exception_handler(sqlite3.Error)
    async def database_unavailable(request: Any, exc: sqlite3.Error) -> JSONResponse:
        runtime.broker.fatal_error = "LEDGER_UNAVAILABLE: " + str(exc)
        runtime.broker.persistence_failed = True
        return JSONResponse(
            status_code=503, content={"error": "LEDGER_UNAVAILABLE", "system": runtime.status()}
        )

    @app.get("/api/health")
    async def health() -> JSONResponse:
        status = runtime.status()
        healthy = (
            status["worker_health"] == "RUNNING"
            and status["manager_health"] == "RUNNING"
            and status["ledger_available"]
            and not runtime.broker.fatal_error
        )
        return JSONResponse(status_code=200 if healthy else 503, content=status)

    def status(compact: bool = True) -> dict[str, Any]:
        state = {**runtime.status(), "paused": runtime.pause}
        if not compact:
            return state
        keys = (
            "connected",
            "reconciled",
            "armed",
            "paused",
            "session",
            "problem",
            "entry_block_reason",
            "ledger_available",
            "worker_health",
            "manager_health",
            "last_scanner_observation",
            "option_quote_state",
            "last_option_quote_check",
            "outstanding_obligations",
        )
        return {key: state[key] for key in keys}

    @app.get("/api/status")
    async def compact_status() -> dict[str, Any]:
        return status()

    @app.get("/api/system")
    async def system(offset: int = Query(0, ge=0)) -> dict[str, Any]:
        return {
            **status(False),
            "diagnostic_offset": offset,
            "diagnostic_limit": 50,
            "obligations": [
                dict(row)
                for row in runtime.store.db.execute(
                    "SELECT reference,session,symbol,status FROM first4_orders "
                    "WHERE role='ENTRY' AND obligation_done=0 "
                    "ORDER BY session,reference LIMIT 50 OFFSET ?",
                    (offset,),
                )
            ],
            "broker_positions": [
                dict(row)
                for row in runtime.store.db.execute(
                    "SELECT * FROM first4_positions WHERE quantity<>0 "
                    "ORDER BY con_id LIMIT 50 OFFSET ?",
                    (offset,),
                )
            ],
        }

    def selected_session(session: date | None) -> str:
        if session:
            return session.isoformat()
        if runtime.session:
            return runtime.session
        row = runtime.store.db.execute("SELECT MAX(session) FROM first4_sessions").fetchone()
        return str(row[0] or "")

    @app.get("/api/overview")
    async def overview(session: date | None = None) -> dict[str, Any]:
        day = selected_session(session)
        items = allocations(runtime.store, day)
        by_slot = {item["slot"]: item for item in items}
        return {
            "system": status(),
            "session": day,
            "q5": Q5,
            "slots": [
                by_slot.get(slot, {"slot": slot, "state": "WAITING", "symbol": None})
                for slot in range(1, 5)
            ],
            "candidates": opportunities(runtime.store, day, 6, 0, "rejected", top=True),
            "pnl": economics(items),
        }

    @app.get("/api/opportunities")
    async def candidate_page(
        session: date | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        decision: Literal["selected", "rejected", "all"] = "all",
    ) -> dict[str, Any]:
        day = selected_session(session)
        rows = opportunities(runtime.store, day, limit, offset, decision)
        return {
            "system": status(),
            "session": day,
            "q5": Q5,
            "rows": rows,
            "offset": offset,
            "limit": limit,
            "has_more": len(rows) == limit,
        }

    @app.get("/api/execution")
    async def execution(session: date | None = None) -> dict[str, Any]:
        day = selected_session(session)
        items = allocations(runtime.store, day)
        return {"system": status(), "session": day, "allocations": items, "pnl": economics(items)}

    @app.get("/api/detail")
    async def detail(
        session: date,
        symbol: str = Query(min_length=1, max_length=100),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
    ) -> dict[str, Any]:
        day = session.isoformat()
        event = runtime.store.db.execute(
            "SELECT * FROM first4_events WHERE session=? AND symbol=?", (day, symbol)
        ).fetchone()
        if event is None:
            raise HTTPException(404, "No such FIRST4 observation")
        orders = [
            dict(row)
            for row in runtime.store.db.execute(
                "SELECT * FROM first4_orders WHERE session=? AND symbol=? "
                "ORDER BY reference LIMIT ? OFFSET ?",
                (day, symbol, limit, offset),
            )
        ]
        fills = [
            dict(row)
            for row in runtime.store.db.execute(
                "SELECT f.* FROM first4_orders o JOIN first4_fills f USING(reference) "
                "WHERE o.session=? AND o.symbol=? ORDER BY f.time,f.exec_id LIMIT ? OFFSET ?",
                (day, symbol, limit, offset),
            )
        ]
        for order in orders:
            order["payload"] = json.loads(order["payload"])
        evidence = dict(event)
        evidence["detail"] = json.loads(evidence["detail"] or "{}")
        return {
            "event": evidence,
            "orders": orders,
            "fills": fills,
            "offset": offset,
            "limit": limit,
            "has_more": max(len(orders), len(fills)) == limit,
            "quote_basis": "QUOTE_COMPARISON_NOT_BROKER_FILLS",
        }

    @app.post("/api/first4/pause")
    async def pause() -> dict[str, Any]:
        runtime.pause = True
        runtime.broker.opening_verified_session = None
        runtime.broker.management_block = "ENTRIES_PAUSED"
        runtime.store.set_meta("paused", True)
        return status(False)

    @app.get("/{page:path}")
    async def page(page: str) -> Any:
        if page.startswith("api/"):
            raise HTTPException(404, "No such FIRST4 route")
        redirects = {
            "candidates": "opportunities",
            "orders": "execution",
            "positions": "execution",
            "trades": "execution",
            "settings": "system",
        }
        if page in redirects:
            return RedirectResponse("/" + redirects[page], status_code=308)
        if page not in {"", "opportunities", "execution", "system"}:
            raise HTTPException(404, "No such SLRNO page")
        return FileResponse(static / "index.html")

    return app
