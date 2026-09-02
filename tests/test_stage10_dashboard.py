from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from stocker_core.config import RunsConfig, load_runs_config
from stocker_core.runs import Environment, RunConfig, RunRiskConfig, RunWindow
from stocker_core.universes import InstrumentReference, UniverseDefinition
from stocker_dashboard.app import create_dashboard_app
from stocker_dashboard.controls import LiveConfirmation, RunControlService
from stocker_dashboard.read_service import DashboardReadService
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerFill,
    BrokerOrderIds,
    BrokerOrderStatus,
    EntryOrderType,
    OrderAction,
    OrderLifecycle,
    OrderPlan,
)
from stocker_execution.runtime import (
    ApplicationState,
    ExecutionEnvironmentStatus,
    MarketSessionState,
    RunRuntimeState,
    RunStatus,
    RuntimeCounters,
    RuntimeStatus,
    RuntimeStore,
)
from stocker_execution.session_hard_structure_d import (
    PreMoveBand,
    SignalStatus,
    StrategySignal,
)
from stocker_execution.stage5 import (
    STAGE5_CALCULATION_VERSION,
    Stage5FeatureSnapshot,
    Stage5SnapshotStore,
    Stage5Status,
)

NOW = datetime(2026, 9, 2, 14, 41, tzinfo=UTC)


def _config() -> RunsConfig:
    universe = UniverseDefinition(
        universe_id="NASDAQ",
        name="NASDAQ operational",
        members=(
            InstrumentReference(
                symbol="NVDA", exchange="SMART", primary_exchange="NASDAQ", currency="USD"
            ),
            InstrumentReference(
                symbol="AMD", exchange="SMART", primary_exchange="NASDAQ", currency="USD"
            ),
        ),
    )
    session = RunWindow(start="09:30", end="16:00", timezone="America/New_York", calendar="XNYS")
    return RunsConfig(
        universes=(universe,),
        runs=(
            RunConfig(
                run_id="US-SH-LIVE",
                universe="NASDAQ",
                strategy="SESSION_HARD",
                environment=Environment.LIVE,
                risk=RunRiskConfig(risk_per_trade=0.001, max_concurrent_positions=5),
                session=session,
            ),
            RunConfig(
                run_id="US-SH-PAPER",
                enabled=False,
                universe="NASDAQ",
                strategy="SESSION_HARD",
                environment=Environment.PAPER,
                risk=RunRiskConfig(risk_per_trade=0.002, max_concurrent_positions=3),
                session=session,
            ),
        ),
    )


def _runtime_status() -> RuntimeStatus:
    return RuntimeStatus(
        application=ApplicationState.READY,
        execution_environments=(
            ExecutionEnvironmentStatus(Environment.PAPER, True, "DU123456", "DU123456", True, True),
            ExecutionEnvironmentStatus(Environment.LIVE, True, "U123456", "U123456", True, True),
        ),
        runs=(
            RunStatus(
                "US-SH-LIVE",
                "NASDAQ",
                "SESSION_HARD",
                Environment.LIVE,
                RunRuntimeState.ACTIVE,
                "",
                date(2026, 9, 2),
                MarketSessionState.ACTIVE_SESSION,
                2,
                1,
                1,
            ),
            RunStatus(
                "US-SH-PAPER",
                "NASDAQ",
                "SESSION_HARD",
                Environment.PAPER,
                RunRuntimeState.DISABLED,
                "disabled by configuration",
                None,
                None,
                0,
                0,
                0,
            ),
        ),
        counters=RuntimeCounters(
            checkpoints_processed=1, instruments_ready=2, signals=1, orders=1, fills=1
        ),
    )


def _seed_authoritative_state(tmp_path: Path) -> DashboardReadService:
    database = tmp_path / "runtime.sqlite3"
    stage5 = Stage5SnapshotStore(database)
    stage5.save(
        Stage5FeatureSnapshot(
            run_ids=("US-SH-LIVE",),
            universe_id="NASDAQ",
            con_id=101,
            symbol="NVDA",
            session=date(2026, 9, 2),
            t0=NOW,
            status=Stage5Status.READY,
            exclusion_reason="",
            p0=180.0,
            expected_absolute_return_15m=0.01,
            m_price=1.8,
            raw_open_t0_minus_3m=177.5,
            raw_open_t0=179.0,
            alignment_factor=1.00558,
            aligned_pre_open=178.49,
            raw_pre_move_price=1.51,
            pre_move_m=0.84,
            calculation_version=STAGE5_CALCULATION_VERSION,
        )
    )
    runtime_store = RuntimeStore(database)
    signal = StrategySignal(
        strategy_id="SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
        strategy_version="SESSION_HARD_STRUCTURE_D_V1",
        signal_id="signal-live-nvda",
        run_id="US-SH-LIVE",
        underlying_con_id=101,
        symbol="NVDA",
        universe_id="NASDAQ",
        session=date(2026, 9, 2),
        t0=NOW,
        status=SignalStatus.ENTRY_TRIGGERED,
        reason="ENTRY_TRIGGERED",
        pre_move_m=0.84,
        cohort_percentile=94.2,
        band=PreMoveBand.HIGH,
        session_hard_score=0.9998,
        session_hard_checkpoint=12,
        session_hard_qualified=True,
        feature_calculation_version=STAGE5_CALCULATION_VERSION,
        side="SELL",
        direction="DOWN",
        candidate_rank=1,
        selected=True,
        p0=180.0,
        m_price=1.8,
        entry_level=179.64,
        entry_reference=179.60,
        entry_timestamp=NOW,
        signal_timestamp=NOW,
    )
    runtime_store.save_signals((signal,), NOW)
    runtime_store.increment("US-SH-LIVE", NOW.date(), "signals")
    runtime_store.record_reconciliation(
        environment=Environment.LIVE,
        account="U123456",
        connection_epoch=2,
        ok=True,
        detail="reconciled",
        now=NOW,
    )

    ledger = ExecutionLedger(database)
    plan = OrderPlan(
        order_plan_id="plan-live-nvda",
        run_id="US-SH-LIVE",
        signal_id=signal.signal_id,
        strategy_id=signal.strategy_id,
        strategy_version=signal.strategy_version,
        con_id=101,
        symbol="NVDA",
        side=OrderAction.SELL,
        quantity=12,
        entry_order_type=EntryOrderType.MARKET,
        entry_reference=179.60,
        stop_price=180.50,
        target_price=177.80,
        environment=Environment.LIVE,
        created_at=NOW,
    )
    assert ledger.reserve(plan, expected_account="U123456")
    ledger.record_submission(
        "plan-live-nvda",
        BrokerOrderIds(parent=100, stop=101, target=102),
        actual_account="U123456",
    )
    ledger.record_order_status(
        BrokerOrderStatus(
            100, plan.order_plan_id, "U123456", Environment.LIVE, OrderLifecycle.FILLED, 12, 0
        )
    )
    ledger.record_fill(
        BrokerFill(
            execution_id="fill-1",
            order_id=100,
            account="U123456",
            environment=Environment.LIVE,
            con_id=101,
            symbol="NVDA",
            side=OrderAction.SELL,
            quantity=12,
            price=179.68,
            executed_at=NOW,
        )
    )
    return DashboardReadService(
        config=_config(),
        runtime_status=_runtime_status,
        stage5_store=stage5,
        runtime_store=runtime_store,
        ledger=ledger,
        clock=lambda: NOW,
    )


def test_overview_and_runs_expose_runtime_environment_state(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)

    overview = service.overview()
    runs = service.runs()

    assert overview["system"] == "READY"
    assert overview["environments"]["LIVE"]["connected"] is True
    assert overview["active_runs"] == 1
    assert overview["open_positions"] == 1
    assert overview["today"]["signals"] == 1
    assert runs[0]["run_id"] == "US-SH-LIVE"
    assert runs[0]["environment"] == "LIVE"
    assert runs[1]["environment"] == "PAPER"


def test_run_detail_uses_persisted_funnel_counts_and_config(tmp_path: Path) -> None:
    detail = _seed_authoritative_state(tmp_path).run_detail("US-SH-LIVE")

    assert detail["account"] == "U123456"
    assert detail["risk_per_trade"] == 0.001
    assert detail["strategy_version"] == "SESSION_HARD_STRUCTURE_D_V1"
    assert detail["funnel"] == [
        {"stage": "Universe", "count": 2},
        {"stage": "Stage 5 ready", "count": 1},
        {"stage": "Strategy evaluated", "count": 1},
        {"stage": "Strategy qualified", "count": 1},
        {"stage": "Rank selected", "count": 1},
        {"stage": "Entry triggered", "count": 1},
        {"stage": "Orders", "count": 1},
        {"stage": "Positions", "count": 1},
    ]


def test_candidates_expose_stage5_and_stage6_values_without_recalculation(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)

    page = service.candidates(run_id="US-SH-LIVE", session=date(2026, 9, 2), limit=25)
    detail = service.candidate_detail("signal-live-nvda")

    assert page["total"] == 1
    assert page["items"][0]["pre_move_m"] == 0.84
    assert page["items"][0]["cohort_percentile"] == 94.2
    assert page["items"][0]["band"] == "HIGH"
    assert page["items"][0]["session_hard"] is True
    assert detail["m_price"] == 1.8
    assert detail["expected_absolute_return_15m"] == 0.01
    assert detail["signal_id"] == "signal-live-nvda"
    assert detail["order_plan_id"] == "plan-live-nvda"


def test_orders_positions_and_trades_preserve_account_environment(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)

    orders = service.orders(scope="all", limit=25)
    positions = service.positions()
    trades = service.trades(environment=None, start=None, end=None, limit=25)

    assert orders["items"][0]["environment"] == "LIVE"
    assert orders["items"][0]["account"] == "U123456"
    assert [child["role"] for child in orders["items"][0]["orders"]] == ["ENTRY", "STOP", "TARGET"]
    assert positions[0]["source"] == "IBKR_RECONCILED"
    assert positions[0]["environment"] == "LIVE"
    assert positions[0]["account"] == "U123456"
    assert trades["items"] == []
    assert trades["summary"]["trades"] == 0


def _write_control_files(tmp_path: Path) -> tuple[Path, Path]:
    runs_path = tmp_path / "runs.yaml"
    broker_path = tmp_path / "ibkr.yaml"
    runs_path.write_text(yaml.safe_dump(_config().model_dump(mode="json")), encoding="utf-8")
    broker_path.write_text(
        yaml.safe_dump(
            {
                "PAPER": {
                    "environment": "PAPER",
                    "host": "127.0.0.1",
                    "port": 4002,
                    "client_id": 21,
                    "expected_account": "DU123456",
                },
                "LIVE": {
                    "environment": "LIVE",
                    "host": "127.0.0.1",
                    "port": 4001,
                    "client_id": 22,
                    "expected_account": "U123456",
                },
            }
        ),
        encoding="utf-8",
    )
    return runs_path, broker_path


def test_run_controls_validate_through_backend_and_require_live_confirmation(
    tmp_path: Path,
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    observed: list[tuple[str, str]] = []
    controls = RunControlService(
        runs_path,
        broker_path,
        on_change=lambda operation, run: observed.append((operation, run.run_id)),
    )

    controls.disable_run("US-SH-LIVE")
    controls.enable_run(
        "US-SH-LIVE",
        confirmation=LiveConfirmation(confirmed=True, target_account="U123456"),
    )
    controls.update_run_config(
        "US-SH-PAPER",
        risk_per_trade=0.003,
        max_concurrent_positions=4,
        universe="NASDAQ",
        strategy="SESSION_HARD",
    )

    with pytest.raises(ValueError, match="LIVE confirmation"):
        controls.change_execution_environment("US-SH-PAPER", Environment.LIVE)
    changed = controls.change_execution_environment(
        "US-SH-PAPER",
        Environment.LIVE,
        confirmation=LiveConfirmation(confirmed=True, target_account="U123456"),
    )

    loaded = load_runs_config(runs_path)
    assert changed.environment is Environment.LIVE
    assert changed.strategy == "SESSION_HARD"
    assert loaded.runs[1].risk == RunRiskConfig(risk_per_trade=0.003, max_concurrent_positions=4)
    assert observed == [
        ("disable_run", "US-SH-LIVE"),
        ("enable_run", "US-SH-LIVE"),
        ("update_run_config", "US-SH-PAPER"),
        ("change_execution_environment", "US-SH-PAPER"),
    ]


def test_invalid_live_account_and_invalid_config_are_rejected(tmp_path: Path) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)

    with pytest.raises(ValueError, match="target account"):
        controls.change_execution_environment(
            "US-SH-PAPER",
            Environment.LIVE,
            confirmation=LiveConfirmation(confirmed=True, target_account="WRONG"),
        )
    with pytest.raises(ValueError):
        controls.update_run_config(
            "US-SH-PAPER",
            risk_per_trade=-1,
            max_concurrent_positions=4,
            universe="NASDAQ",
            strategy="SESSION_HARD",
        )


def test_http_routes_and_all_navigation_pages_render(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    app = create_dashboard_app(service, RunControlService(runs_path, broker_path))
    client = TestClient(app)

    assert client.get("/api/overview").json()["system"] == "READY"
    for route in (
        "/",
        "/runs",
        "/candidates",
        "/orders",
        "/positions",
        "/trades",
        "/system",
        "/settings",
    ):
        response = client.get(route)
        assert response.status_code == 200
        assert "STOCKER" in response.text
        assert "PAPER" in response.text
        assert "LIVE" in response.text
    index = client.get("/").text
    for label in (
        "Overview",
        "Runs",
        "Candidates",
        "Orders",
        "Positions",
        "Trades",
        "System",
        "Settings",
    ):
        assert f">{label}<" in index


def test_candidate_pagination_is_bounded_and_dashboard_has_no_trading_calculators(
    tmp_path: Path,
) -> None:
    service = _seed_authoritative_state(tmp_path)
    with pytest.raises(ValueError, match="limit"):
        service.candidates(limit=501)

    dashboard_root = Path(__file__).parents[1] / "packages" / "stocker_dashboard"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in dashboard_root.rglob("*.py")
        if path.name != "__init__.py"
    )
    forbidden = (
        "calculate_pre_move",
        "calculate_cohort_percentile",
        "calculate_session_hard_score",
        "calculate_position_size",
        "submit_protected_order",
    )
    assert not any(name in source for name in forbidden)


def test_dashboard_failure_is_confined_to_http_request(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    original_status = service.runtime_status
    service.runtime_status = lambda: (_ for _ in ()).throw(RuntimeError("dashboard read failed"))
    client = TestClient(
        create_dashboard_app(service, RunControlService(runs_path, broker_path)),
        raise_server_exceptions=False,
    )

    response = client.get("/api/overview")

    assert response.status_code == 503
    service.runtime_status = original_status
    assert service.overview()["system"] == "READY"
