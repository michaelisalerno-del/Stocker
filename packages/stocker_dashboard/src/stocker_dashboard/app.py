"""FastAPI boundary for the replaceable Stage 10 dashboard."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from stocker_core.config import IbkrConfig, load_runs_config
from stocker_core.markets import MarketId
from stocker_core.runs import Environment
from stocker_dashboard.controls import ControlResult, LiveConfirmation, RunControlService
from stocker_dashboard.performance import PerformancePeriod
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


class BrokerConfigBody(BaseModel):
    host: str
    port: int
    client_id: int
    expected_account: str | None = None
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 60.0
    market_data_line_budget: int = 100


class CustomUniverseBody(BaseModel):
    name: str | None = None
    symbols: list[str]
    exchange: str = "SMART"
    primary_exchange: str | None = None
    currency: str = "USD"
    security_type: str = "STK"


class UniverseRunBody(ConfirmationBody):
    model_config = ConfigDict(extra="forbid")
    market_id: MarketId
    strategy_id: str
    strategy_version: str
    risk_per_trade: float
    max_concurrent_positions: int | None = None


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

    @app.get("/api/runs/{run_id}/performance")
    def run_performance(
        run_id: str,
        period: PerformancePeriod = PerformancePeriod.TODAY,
    ) -> dict[str, Any]:
        return reads.run_performance(run_id, period)

    @app.get("/api/universe-builder/options")
    async def universe_builder_options() -> dict[str, object]:
        return await controls.universe_builder_options()

    @app.get("/api/universe-runs")
    def universe_runs() -> dict[str, list[dict[str, Any]]]:
        return reads.universe_runs()

    async def add_universe_run(
        body: UniverseRunBody, environment: Environment
    ) -> dict[str, object]:
        try:
            return changed(
                await controls.add_universe_run(
                    market_id=body.market_id,
                    strategy_id=body.strategy_id,
                    strategy_version=body.strategy_version,
                    environment=environment,
                    risk_per_trade=body.risk_per_trade,
                    max_concurrent_positions=body.max_concurrent_positions,
                    confirmation=confirmation(body),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/universe-runs/paper")
    async def add_paper_run(body: UniverseRunBody) -> dict[str, object]:
        return await add_universe_run(body, Environment.PAPER)

    @app.post("/api/universe-runs/live")
    async def add_live_run(body: UniverseRunBody) -> dict[str, object]:
        return await add_universe_run(body, Environment.LIVE)

    @app.post("/api/universe-runs/{run_id}/disable")
    async def disable_universe_run(run_id: str) -> dict[str, object]:
        try:
            return changed(await controls.disable_run(run_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/universe-runs/{run_id}/enable")
    async def enable_universe_run(run_id: str, body: ConfirmationBody) -> dict[str, object]:
        try:
            return changed(await controls.enable_run(run_id, confirmation=confirmation(body)))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/screens/{market_id}/{cap_bucket}/{session}")
    def screen(market_id: str, cap_bucket: str, session: date) -> dict[str, Any]:
        try:
            return reads.screen(market_id, cap_bucket, session)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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

    @app.get("/api/orders/{order_plan_id}")
    def order_detail(order_plan_id: str) -> dict[str, Any]:
        return reads.order_detail(order_plan_id)

    @app.get("/api/positions")
    def positions() -> list[dict[str, Any]]:
        return reads.positions()

    @app.get("/api/positions/{environment}/{account}/{con_id}")
    def position_detail(
        environment: Environment,
        account: str,
        con_id: int,
    ) -> dict[str, Any]:
        return reads.position_detail(environment, account, con_id)

    @app.get("/api/trades")
    def trades(
        environment: Environment | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        run_id: str | None = None,
        strategy: str | None = None,
        universe: str | None = None,
        symbol: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return reads.trades(
            environment=environment,
            start=start,
            end=end,
            run_id=run_id,
            strategy=strategy,
            universe=universe,
            symbol=symbol,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        return reads.system()

    @app.get("/api/settings")
    def settings() -> dict[str, Any]:
        return {**reads.settings(), **controls.configuration()}

    @app.get("/api/runs/{run_id}/live-confirmation")
    def live_confirmation(run_id: str) -> dict[str, object]:
        return controls.live_confirmation_context(run_id)

    def confirmation(body: ConfirmationBody) -> LiveConfirmation | None:
        if not body.confirmed and not body.target_account:
            return None
        return LiveConfirmation(body.confirmed, body.target_account)

    def changed(result: ControlResult) -> dict[str, object]:
        reads.config = load_runs_config(controls.runs_config_path)
        return result.as_dict()

    @app.post("/api/runs/{run_id}/enable")
    async def enable(run_id: str, body: ConfirmationBody) -> dict[str, Any]:
        try:
            return changed(await controls.enable_run(run_id, confirmation=confirmation(body)))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/disable")
    async def disable(run_id: str) -> dict[str, Any]:
        try:
            return changed(await controls.disable_run(run_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/runs/{run_id}")
    async def update(run_id: str, body: RunUpdateBody) -> dict[str, Any]:
        try:
            return changed(
                await controls.update_run_config(
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
    async def environment(run_id: str, body: EnvironmentBody) -> dict[str, Any]:
        try:
            return changed(
                await controls.change_execution_environment(
                    run_id, body.environment, confirmation=confirmation(body)
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/settings/broker/{environment}")
    async def broker_config(environment: Environment, body: BrokerConfigBody) -> dict[str, Any]:
        try:
            result = await controls.update_broker_config(
                IbkrConfig(environment=environment, **body.model_dump())
            )
            return result.as_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/settings/universes/{universe_id}")
    async def custom_universe(universe_id: str, body: CustomUniverseBody) -> dict[str, Any]:
        try:
            result = await controls.replace_custom_universe_symbols(
                universe_id,
                body.symbols,
                name=body.name,
                exchange=body.exchange,
                primary_exchange=body.primary_exchange,
                currency=body.currency,
                security_type=body.security_type,
            )
            reads.config = load_runs_config(controls.runs_config_path)
            return result.as_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/{page:path}", response_class=FileResponse)
    def index(page: str = "") -> FileResponse:
        if page.startswith("api/"):
            raise HTTPException(status_code=404)
        return FileResponse(static / "index.html")

    return app
