"""FastAPI boundary for the replaceable Stage 10 dashboard."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stocker_core.config import load_runs_config
from stocker_core.runs import Environment
from stocker_dashboard.controls import LiveConfirmation, RunControlService
from stocker_dashboard.read_service import DashboardReadService


class ConfirmationBody(BaseModel):
    confirmed: bool = False
    target_account: str = ""


class RunUpdateBody(ConfirmationBody):
    universe: str
    strategy: str
    risk_per_trade: float
    max_concurrent_positions: int | None = None


class EnvironmentBody(ConfirmationBody):
    environment: Environment


def create_dashboard_app(reads: DashboardReadService, controls: RunControlService) -> FastAPI:
    """Create an isolated HTTP consumer over injected read/control boundaries."""

    app = FastAPI(title="Stocker Operational Dashboard", docs_url="/api/docs")
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.exception_handler(Exception)
    async def unavailable(_request: Any, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "dashboard read unavailable", "error": str(exc)},
        )

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return reads.overview()

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return reads.runs()

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        return reads.run_detail(run_id)

    @app.get("/api/candidates")
    def candidates(
        run_id: str | None = None,
        session: date | None = None,
        checkpoint: datetime | None = None,
        status: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return reads.candidates(
            run_id=run_id,
            session=session,
            checkpoint=checkpoint,
            status=status,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/candidates/{signal_id}")
    def candidate_detail(signal_id: str) -> dict[str, Any]:
        return reads.candidate_detail(signal_id)

    @app.get("/api/orders")
    def orders(
        scope: str = "open",
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return reads.orders(scope=scope, limit=limit, offset=offset)

    @app.get("/api/positions")
    def positions() -> list[dict[str, Any]]:
        return reads.positions()

    @app.get("/api/trades")
    def trades(
        environment: Environment | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        run_id: str | None = None,
        symbol: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return reads.trades(
            environment=environment,
            start=start,
            end=end,
            run_id=run_id,
            symbol=symbol,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        return reads.system()

    @app.get("/api/settings")
    def settings() -> dict[str, Any]:
        return reads.settings()

    @app.get("/api/runs/{run_id}/live-confirmation")
    def live_confirmation(run_id: str) -> dict[str, object]:
        return controls.live_confirmation_context(run_id)

    def confirmation(body: ConfirmationBody) -> LiveConfirmation | None:
        if not body.confirmed and not body.target_account:
            return None
        return LiveConfirmation(body.confirmed, body.target_account)

    def changed(run: Any) -> dict[str, Any]:
        reads.config = load_runs_config(controls.runs_config_path)
        return {"run": run.model_dump(mode="json"), "restart_required": True}

    @app.post("/api/runs/{run_id}/enable")
    def enable(run_id: str, body: ConfirmationBody) -> dict[str, Any]:
        try:
            return changed(controls.enable_run(run_id, confirmation=confirmation(body)))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/disable")
    def disable(run_id: str) -> dict[str, Any]:
        try:
            return changed(controls.disable_run(run_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/runs/{run_id}")
    def update(run_id: str, body: RunUpdateBody) -> dict[str, Any]:
        try:
            return changed(
                controls.update_run_config(
                    run_id,
                    universe=body.universe,
                    strategy=body.strategy,
                    risk_per_trade=body.risk_per_trade,
                    max_concurrent_positions=body.max_concurrent_positions,
                    confirmation=confirmation(body),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/environment")
    def environment(run_id: str, body: EnvironmentBody) -> dict[str, Any]:
        try:
            return changed(
                controls.change_execution_environment(
                    run_id, body.environment, confirmation=confirmation(body)
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/{page:path}", response_class=FileResponse)
    def index(page: str = "") -> FileResponse:
        if page.startswith("api/"):
            raise HTTPException(status_code=404)
        return FileResponse(static / "index.html")

    return app
