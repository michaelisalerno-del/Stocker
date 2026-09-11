"""Production and safe standalone composition for the dashboard server."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI

from stocker_core.config import RunsConfig, load_ibkr_config, load_runs_config
from stocker_dashboard.app import create_dashboard_app
from stocker_dashboard.controls import ActiveRuntimeControl, RunControlService
from stocker_dashboard.read_service import DashboardReadService
from stocker_execution.activity_shortlist import ActivityShortlistStore
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.runtime import (
    ApplicationState,
    ExecutionEnvironmentStatus,
    RunRuntimeState,
    RunStatus,
    RuntimeCounters,
    RuntimeStatus,
    RuntimeStore,
)
from stocker_execution.stage5 import Stage5SnapshotStore


def build_dashboard_app(
    *,
    runs_config_path: str | Path,
    ibkr_config_path: str | Path,
    database_path: str | Path,
    runtime_status: Callable[[], RuntimeStatus] | None = None,
    runtime: ActiveRuntimeControl | None = None,
) -> FastAPI:
    """Build the dashboard without owning or starting the trading runtime."""

    config = load_runs_config(runs_config_path)
    status_provider = (
        runtime.status
        if runtime is not None
        else runtime_status or _standalone_status(config, ibkr_config_path)
    )
    reads = DashboardReadService(
        config=config,
        runtime_status=status_provider,
        stage5_store=Stage5SnapshotStore(database_path),
        runtime_store=RuntimeStore(database_path),
        ledger=ExecutionLedger(database_path),
        activity_store=ActivityShortlistStore(database_path),
    )
    controls = RunControlService(runs_config_path, ibkr_config_path, runtime=runtime)
    return create_dashboard_app(reads, controls)


def _standalone_status(
    config: RunsConfig, ibkr_config_path: str | Path
) -> Callable[[], RuntimeStatus]:
    environments = tuple(dict.fromkeys(run.environment for run in config.runs))
    destinations = []
    for environment in environments:
        broker = load_ibkr_config(ibkr_config_path, environment)
        destinations.append(
            ExecutionEnvironmentStatus(
                environment=environment,
                connected=False,
                account=None,
                expected_account=broker.expected_account or "UNCONFIGURED",
                reconciled=False,
                ready=False,
            )
        )
    runs = tuple(
        RunStatus(
            run_id=run.run_id,
            universe=run.universe,
            strategy=run.strategy,
            environment=run.environment,
            state=RunRuntimeState.STOPPED if run.enabled else RunRuntimeState.DISABLED,
            reason="dashboard standalone; runtime status unavailable"
            if run.enabled
            else "disabled",
            session=None,
            market=None,
            instruments_ready=0,
            signals_today=0,
            open_positions=0,
        )
        for run in config.runs
    )

    def status() -> RuntimeStatus:
        return RuntimeStatus(
            application=ApplicationState.STOPPED,
            execution_environments=tuple(destinations),
            runs=runs,
            counters=RuntimeCounters(),
        )

    return status
