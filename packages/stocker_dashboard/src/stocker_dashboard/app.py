"""Existing Stocker dashboard boundary for the single FIRST4 runtime."""

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from stocker_dashboard.security import DashboardSecurity
from stocker_execution.first4_runtime import Runtime


def create_dashboard_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="Stocker FIRST4 PAPER", docs_url="/api/docs")
    app.add_middleware(DashboardSecurity)
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/api/system")
    @app.get("/api/settings")
    async def system() -> dict[str, Any]:
        return runtime.status()

    @app.get("/api/overview")
    async def overview(
        view: Literal[
            "all", "system", "settings", "candidates", "orders", "positions", "trades"
        ] = "all",
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        orders_page = (
            runtime.store.page("orders", 100, offset if view == "orders" else 0)
            if view in {"all", "orders", "trades"}
            else []
        )
        return {
            "system": runtime.status(),
            "history": {
                "order": "newest_first",
                "candidates_limit": 150,
                "orders_limit": 100,
                "fills_limit": 100,
                "pagination": "limit/offset on history endpoints",
            },
            "candidates": runtime.store.page("events", 150, offset)
            if view in {"all", "candidates"}
            else [],
            "orders": orders_page,
            "positions": runtime.store.rows("positions"),
            "fills": runtime.store.page("fills", 100, offset) if view in {"all", "trades"} else [],
            "errors": runtime.store.page("meta") if view in {"all", "system"} else [],
            "pnl": pnl() if view in {"all", "system", "settings"} else {"session": runtime.session},
            "quote_comparisons": [
                {
                    "reference": order["reference"],
                    **{
                        key: payload.get(key)
                        for key in (
                            "quoted_entry_ask_for_exit_legs_usd",
                            "quoted_exit_bid_usd",
                            "quoted_ask_to_bid_gross_usd",
                            "quotes",
                            "exit_quotes",
                            "exit_quantity",
                            "exit_con_ids",
                        )
                    },
                    "basis": "QUOTE_COMPARISON_NOT_BROKER_FILLS",
                }
                for order in orders_page
                if order["role"] == "EXIT"
                for payload in [json.loads(order["payload"])]
            ],
        }

    def pnl() -> dict[str, Any]:
        session = runtime.session
        if session is None:
            row = runtime.store.db.execute(
                "SELECT session FROM first4_events ORDER BY session DESC LIMIT 1"
            ).fetchone()
            session = row[0] if row else None
        session_orders = [
            dict(r)
            for r in runtime.store.db.execute(
                "SELECT * FROM first4_orders WHERE session=?", (session,)
            )
        ]
        fills = [
            dict(r)
            for r in runtime.store.db.execute(
                "SELECT f.* FROM first4_orders o JOIN first4_fills f USING(reference) "
                "WHERE o.session=?",
                (session,),
            )
        ]
        cash = sum(
            f["quantity"] * f["price"] * f["multiplier"] * (1 if f["side"] == "SLD" else -1)
            for f in fills
        )
        fees = sum(f["commission"] or 0 for f in fills)
        unsettled = runtime.store.db.execute(
            "SELECT 1 FROM first4_orders WHERE session=? AND role='ENTRY' "
            "AND coalesce(json_extract(payload,'$.management_resolved'),0)=0 LIMIT 1",
            (session,),
        ).fetchone()
        flat = not unsettled and runtime.broker.reconciled
        complete = all(f["commission"] is not None for f in fills)
        return {
            "basis": "IBKR_PAPER_SIMULATED_FILLS",
            "currency": "USD",
            "session": session,
            "actual_fees_usd": fees,
            "fees_complete": complete,
            "reserved_fee_allowance_usd": sum(
                json.loads(o["payload"]).get("fee_reserve_usd", 10)
                for o in session_orders
                if o["role"] == "ENTRY"
            ),
            "session_allocation_usd": sum(
                json.loads(o["payload"]).get("allocation_usd", 260)
                for o in session_orders
                if o["role"] == "ENTRY"
            ),
            "net_cash_flow": cash - fees,
            "realised": cash - fees if flat and complete else None,
            "unrealised": None,
            "status": "CLOSED_BROKER_FILLS" if flat and complete else "OPEN_OR_FEES_PENDING",
        }

    @app.get("/api/candidates")
    async def candidates(
        limit: int = Query(150, ge=1, le=500), offset: int = Query(0, ge=0)
    ) -> list[dict[str, Any]]:
        return runtime.store.page("events", limit, offset)

    @app.get("/api/orders")
    async def orders(
        limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)
    ) -> list[dict[str, Any]]:
        return runtime.store.page("orders", limit, offset)

    @app.get("/api/positions")
    async def positions() -> list[dict[str, Any]]:
        return runtime.store.rows("positions")

    @app.get("/api/trades")
    async def trades(
        limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)
    ) -> list[dict[str, Any]]:
        return runtime.store.page("fills", limit, offset)

    @app.post("/api/first4/pause")
    async def pause() -> dict[str, Any]:
        runtime.pause = True
        runtime.store.set_meta("paused", True)
        return runtime.status()

    @app.get("/{page:path}")
    async def page(page: str) -> FileResponse:
        if page.startswith("api/"):
            raise HTTPException(404, "No such FIRST4 route")
        return FileResponse(static / "index.html")

    return app
