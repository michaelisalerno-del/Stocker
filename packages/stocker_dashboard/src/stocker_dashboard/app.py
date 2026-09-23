"""Existing Stocker dashboard boundary for the single FIRST4 runtime."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
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
    async def overview() -> dict[str, Any]:
        return {
            "system": runtime.status(),
            "candidates": runtime.store.rows("events"),
            "orders": runtime.store.rows("orders"),
            "positions": runtime.store.rows("positions"),
            "fills": runtime.store.rows("fills"),
            "errors": runtime.store.rows("meta"),
            "pnl": pnl(),
        }

    def pnl() -> dict[str, Any]:
        fills = runtime.store.rows("fills")
        cash = sum(
            f["quantity"] * f["price"] * f["multiplier"] * (1 if f["side"] == "SLD" else -1)
            for f in fills
        )
        fees = sum(f["commission"] or 0 for f in fills)
        flat = not any(runtime.broker.owned_quantities().values())
        complete = all(f["commission"] is not None for f in fills)
        return {
            "currency": "USD",
            "net_cash_flow": cash - fees,
            "realised": cash - fees if flat and complete else None,
            "unrealised": None,
            "status": "CLOSED_BROKER_FILLS" if flat and complete else "OPEN_OR_FEES_PENDING",
        }

    @app.get("/api/candidates")
    async def candidates() -> list[dict[str, Any]]:
        return runtime.store.rows("events")

    @app.get("/api/orders")
    async def orders() -> list[dict[str, Any]]:
        return runtime.store.rows("orders")

    @app.get("/api/positions")
    async def positions() -> list[dict[str, Any]]:
        return runtime.store.rows("positions")

    @app.get("/api/trades")
    async def trades() -> list[dict[str, Any]]:
        return runtime.store.rows("fills")

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
