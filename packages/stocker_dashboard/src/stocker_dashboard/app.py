"""Authenticated, bounded FIRST4 ledger views and independent health."""

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from stocker_dashboard.security import DashboardSecurity
from stocker_execution.first4_runtime import Runtime


def session_pnl(runtime: Runtime, session: str) -> dict[str, Any]:
    # An indexed session join, aggregated by allocation and actual leg. Display
    # pagination must never change the selected session's accounting.
    legs = runtime.store.db.execute(
        "SELECT o.symbol,f.con_id,"
        "SUM(CASE side WHEN 'BOT' THEN quantity ELSE 0 END) bought,"
        "SUM(CASE side WHEN 'SLD' THEN quantity ELSE 0 END) sold,"
        "SUM(CASE side WHEN 'BOT' THEN quantity*price*multiplier ELSE 0 END) cost,"
        "SUM(CASE side WHEN 'SLD' THEN quantity*price*multiplier ELSE 0 END) proceeds,"
        "SUM(COALESCE(commission,0)) fees,SUM(commission IS NULL) pending "
        "FROM first4_orders o JOIN first4_effective_fills f USING(reference) "
        "WHERE o.session=? GROUP BY o.symbol,f.con_id",
        (session,),
    ).fetchall()
    entries = runtime.store.db.execute(
        "SELECT symbol,obligation_done,payload FROM first4_orders WHERE session=? AND role='ENTRY'",
        (session,),
    ).fetchall()
    completed = {r["symbol"] for r in entries if r["obligation_done"]}
    pending_symbols = {r["symbol"] for r in legs if r["pending"]}
    gross = sum(r["proceeds"] - r["cost"] for r in legs if r["symbol"] in completed)
    realised = sum(
        r["proceeds"] - r["cost"] - r["fees"]
        for r in legs
        if r["symbol"] in completed - pending_symbols
    )
    fees = sum(r["fees"] for r in legs)
    pending = sum(r["pending"] for r in legs)
    cash = sum(r["proceeds"] - r["cost"] for r in legs)
    return {
        "basis": "IBKR_PAPER_SIMULATED_FILLS",
        "currency": "USD",
        "actual_fees_usd": fees,
        "fees_complete": pending == 0,
        "pending_fee_executions": pending,
        "completed_gross_usd": gross,
        "completed_pending_fee_allocations": len(completed & pending_symbols),
        "realised": realised if completed - pending_symbols else (0 if not legs else None),
        "realised_basis": "COMPLETED_ALLOCATIONS_WITH_KNOWN_FEES",
        "partial_close_gross_usd": sum(
            r["proceeds"] - r["cost"] * r["sold"] / r["bought"]
            for r in legs
            if r["symbol"] not in completed and 0 <= r["sold"] <= r["bought"] and r["bought"] > 0
        ),
        "open_owned_legs": [
            {"symbol": r["symbol"], "con_id": r["con_id"], "quantity": r["bought"] - r["sold"]}
            for r in legs
            if r["bought"] != r["sold"]
        ],
        "gross_cash_flow": cash,
        "net_cash_flow": cash - fees,
        "net_cash_flow_basis": "KNOWN_FEES_ONLY" if pending else "ALL_REPORTED_FEES",
        "reserved_fee_allowance_usd": sum(
            json.loads(r["payload"]).get("fee_reserve_usd", 10) for r in entries
        ),
        "session_allocation_usd": sum(
            json.loads(r["payload"]).get("allocation_usd", 260) for r in entries
        ),
        "unrealised": None,
        "status": "COMPLETED_RESULTS_SEPARATE_FROM_OPEN_EXPOSURE_AND_PENDING_FEES",
    }


def create_dashboard_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="Stocker FIRST4 PAPER", docs_url="/api/docs")
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

    @app.get("/api/system")
    @app.get("/api/settings")
    async def system() -> dict[str, Any]:
        return runtime.status()

    def selected_session(session: date | None) -> str:
        if session:
            return session.isoformat()
        if runtime.session:
            return runtime.session
        row = runtime.store.db.execute("SELECT MAX(session) FROM first4_sessions").fetchone()
        return str(row[0] or "")

    @app.get("/api/overview")
    async def overview(
        session: date | None = None,
        limit: int = Query(150, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        day = selected_session(session)
        orders = runtime.store.page("orders", day, limit, offset)
        return {
            "system": runtime.status(),
            "session": day,
            "limit": limit,
            "offset": offset,
            "candidates": runtime.store.page("events", day, limit, offset),
            "orders": orders,
            "fills": runtime.store.page("fills", day, limit, offset),
            "positions": await positions(),
            "errors": [
                dict(r)
                for r in runtime.store.db.execute(
                    "SELECT * FROM first4_meta WHERE key IN "
                    "('broker_error','runtime_error','broker_manager_error',"
                    "'entry_cancellation_error') "
                    "OR key=? ORDER BY key LIMIT 50",
                    ("opening_check:" + day,),
                )
            ],
            "obligations": [
                dict(r)
                for r in runtime.store.db.execute(
                    "SELECT reference,session,symbol,status FROM first4_orders "
                    "WHERE role='ENTRY' AND obligation_done=0 ORDER BY session,reference LIMIT 200"
                )
            ],
            "pnl": session_pnl(runtime, day),
            "quote_comparisons": [
                {
                    "reference": o["reference"],
                    "basis": "QUOTE_COMPARISON_NOT_BROKER_FILLS",
                    **{
                        k: payload.get(k)
                        for k in (
                            "quoted_entry_ask_for_exit_legs_usd",
                            "quoted_exit_bid_usd",
                            "quoted_ask_to_bid_gross_usd",
                            "quotes",
                            "exit_quotes",
                            "exit_quantity",
                            "exit_con_ids",
                        )
                    },
                }
                for o in orders
                if o["role"] == "EXIT"
                for payload in [json.loads(o["payload"])]
            ],
        }

    @app.get("/api/candidates")
    async def candidates(
        session: date | None = None,
        limit: int = Query(150, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list[dict[str, Any]]:
        return runtime.store.page("events", selected_session(session), limit, offset)

    @app.get("/api/orders")
    async def orders(
        session: date | None = None,
        limit: int = Query(150, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list[dict[str, Any]]:
        return runtime.store.page("orders", selected_session(session), limit, offset)

    @app.get("/api/positions")
    async def positions() -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in runtime.store.db.execute(
                "SELECT * FROM first4_positions WHERE quantity<>0 ORDER BY con_id LIMIT 200"
            )
        ]

    @app.get("/api/trades")
    async def trades(
        session: date | None = None,
        limit: int = Query(150, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list[dict[str, Any]]:
        return runtime.store.page("fills", selected_session(session), limit, offset)

    @app.post("/api/first4/pause")
    async def pause() -> dict[str, Any]:
        runtime.pause = True
        runtime.broker.opening_verified_session = None
        runtime.broker.management_block = "ENTRIES_PAUSED"
        runtime.store.set_meta("paused", True)
        return runtime.status()

    @app.get("/{page:path}")
    async def page(page: str) -> FileResponse:
        if page.startswith("api/"):
            raise HTTPException(404, "No such FIRST4 route")
        return FileResponse(static / "index.html")

    return app
