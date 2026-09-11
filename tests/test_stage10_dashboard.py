from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from execution_test_support import (
    execution_method,  # noqa: F401
)
from stocker_core.cli import stage10_run
from stocker_core.config import IbkrConfig, RunsConfig, load_ibkr_config, load_runs_config
from stocker_core.markets import MarketId
from stocker_core.runs import Environment, RunConfig, RunRiskConfig, RunWindow
from stocker_core.strategies import SESSION_HARD_HV_METHOD
from stocker_core.universes import InstrumentReference, UniverseDefinition
from stocker_dashboard.app import create_dashboard_app
from stocker_dashboard.controls import LiveConfirmation, RunControlService
from stocker_dashboard.read_service import DashboardReadService
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerFill,
    BrokerOrderIds,
    BrokerOrderStatus,
    BrokerPosition,
    EntryOrderType,
    OrderAction,
    OrderLifecycle,
    OrderPlan,
)
from stocker_execution.ibkr import IbkrConnection
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


pytestmark = pytest.mark.usefixtures("execution_method")


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
    session = RunWindow(
        start=time(9, 30),
        end=time(16),
        timezone="America/New_York",
        calendar="XNYS",
    )
    return RunsConfig(
        universes=(universe,),
        runs=(
            RunConfig(
                run_id="US-SH-LIVE",
                universe="NASDAQ",
                strategy="TEST_EXECUTION",
                environment=Environment.LIVE,
                risk=RunRiskConfig(risk_per_trade=0.001, max_concurrent_positions=5),
                session=session,
            ),
            RunConfig(
                run_id="US-SH-PAPER",
                enabled=False,
                universe="NASDAQ",
                strategy="TEST_EXECUTION",
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
            ExecutionEnvironmentStatus(
                Environment.PAPER,
                True,
                "DU123456",
                "DU123456",
                True,
                True,
                equity=100_000.0,
                buying_power=200_000.0,
            ),
            ExecutionEnvironmentStatus(
                Environment.LIVE,
                True,
                "U123456",
                "U123456",
                True,
                True,
                equity=50_000.0,
                buying_power=100_000.0,
            ),
        ),
        runs=(
            RunStatus(
                "US-SH-LIVE",
                "NASDAQ",
                "TEST_EXECUTION",
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
                "TEST_EXECUTION",
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


def test_overview_surfaces_tick_capacity_and_incomplete_checkpoint(tmp_path):
    from test_ibkr_resources import ConfigClient, config

    service = _seed_authoritative_state(tmp_path)
    resource = IbkrConnection(config(), client=ConfigClient()).resource_status()
    status = _runtime_status()
    service.runtime_status = lambda: replace(
        status,
        ibkr_resources=replace(
            resource,
            capacity_rejects_today=12,
            last_resource_error="10190: tick-by-tick limit reached",
        ),
        runs=(
            replace(
                status.runs[0],
                evaluation_state="INCOMPLETE",
                evaluation_completed=4,
                evaluation_total=100,
                trade_stream_unavailable=95,
            ),
            *status.runs[1:],
        ),
    )
    overview = service.overview()
    assert overview["system"] == "ATTENTION"
    assert any("10190" in item["message"] for item in overview["attention"])
    assert any("4/100" in item["message"] for item in overview["attention"])
    run = service.run_detail("US-SH-LIVE")
    assert run["evaluation_status"] == "INCOMPLETE"
    assert run["evaluation_progress"]["completed"] == 4


class RecordingRuntime:
    def __init__(self) -> None:
        self.run_updates: list[tuple[RunsConfig, set[str]]] = []
        self.broker_updates: list[IbkrConfig] = []
        self._status = _runtime_status()

    def status(self) -> RuntimeStatus:
        return self._status

    async def apply_runs_config(
        self, config: RunsConfig, *, changed_run_ids: frozenset[str]
    ) -> RuntimeStatus:
        self.run_updates.append((config, set(changed_run_ids)))
        self._status = RuntimeStatus(
            application=ApplicationState.READY,
            execution_environments=self._status.execution_environments,
            runs=tuple(
                RunStatus(
                    run.run_id,
                    run.universe,
                    run.strategy,
                    run.environment,
                    RunRuntimeState.ACTIVE if run.enabled else RunRuntimeState.DISABLED,
                    "" if run.enabled else "disabled",
                    date(2026, 9, 2) if run.enabled else None,
                    MarketSessionState.ACTIVE_SESSION if run.enabled else None,
                    2 if run.enabled else 0,
                    0,
                    0,
                )
                for run in config.runs
            ),
            counters=self._status.counters,
        )
        return self._status

    async def replace_broker_config(self, config: IbkrConfig) -> RuntimeStatus:
        self.broker_updates.append(config)
        return self._status


class RejectingRuntime(RecordingRuntime):
    async def apply_runs_config(
        self, config: RunsConfig, *, changed_run_ids: frozenset[str]
    ) -> RuntimeStatus:
        raise ValueError("runtime rejected run config")

    async def replace_broker_config(self, config: IbkrConfig) -> RuntimeStatus:
        raise ValueError("broker exposure exists")


class DegradingRuntime(RecordingRuntime):
    async def apply_runs_config(
        self, config: RunsConfig, *, changed_run_ids: frozenset[str]
    ) -> RuntimeStatus:
        status = await super().apply_runs_config(config, changed_run_ids=changed_run_ids)
        if not changed_run_ids:
            return status
        degraded_id = sorted(changed_run_ids)[0]
        self._status = replace(
            status,
            runs=tuple(
                replace(
                    item,
                    state=RunRuntimeState.DEGRADED,
                    reason="instrument preparation failed",
                )
                if item.run_id == degraded_id
                else item
                for item in status.runs
            ),
        )
        return self._status


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
        strategy_id="TEST_EXECUTION",
        strategy_version="TEST_EXECUTION_V1",
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
    ledger.replace_broker_snapshot(
        environment=Environment.LIVE,
        account="U123456",
        positions=(
            BrokerPosition(
                account="U123456",
                con_id=101,
                symbol="NVDA",
                quantity=-12,
                average_price=179.68,
            ),
        ),
        open_orders=(),
        observed_at=NOW,
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
    assert overview["environments"]["PAPER"]["equity"] == 100_000.0
    assert overview["environments"]["PAPER"]["buying_power"] == 200_000.0
    assert overview["environments"]["LIVE"]["equity"] == 50_000.0
    assert overview["environments"]["LIVE"]["buying_power"] == 100_000.0
    assert overview["active_runs"] == 1
    assert overview["open_positions"] == 1
    assert overview["today"]["signals"] == 1
    assert runs[0]["run_id"] == "US-SH-LIVE"
    assert runs[0]["environment"] == "LIVE"
    assert runs[1]["environment"] == "PAPER"


def test_system_resource_view_is_read_only_and_labels_stocker_budget(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    client = type(
        "ResourceClient",
        (),
        {
            "client": type("Throttle", (), {"MaxRequests": 45, "RequestsInterval": 1})(),
            "isConnected": lambda self: False,
        },
    )()
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=77,
            market_data_line_budget=80,
        ),
        client=client,  # type: ignore[arg-type]
    )
    status_reads = 0

    def status() -> RuntimeStatus:
        nonlocal status_reads
        status_reads += 1
        return replace(_runtime_status(), ibkr_resources=connection.resource_status())

    service.runtime_status = status

    first = service.system()["ibkr_api_resources"]
    second = service.system()["ibkr_api_resources"]

    assert first == second
    assert status_reads == 2
    assert first["market_data_budget_label"] == "Stocker API line budget"
    assert first["market_data_line_budget"] == 80
    assert first["ibkr_account_line_limit"] is None
    assert first["active_market_data_lines"] == 0
    latest = replace(
        status(),
        ibkr_resources=replace(
            connection.resource_status(), historical_requests_today=123, pending_historical_work=4
        ),
    )
    service.runtime_status = lambda: replace(
        latest,
        runs=tuple(
            replace(
                run, next_checkpoint=None, last_scheduled_checkpoint=NOW - timedelta(minutes=10)
            )
            for run in latest.runs
        ),
    )
    detail = service.run_detail("US-SH-LIVE")
    assert detail["evaluation_status"] == "NO_MORE_CHECKPOINTS_TODAY"
    assert detail["downloads"] == {"scope": "ALL_RUNS", "requests": 123, "pending": 4}
    assert detail["last_checkpoint"] is not None


def test_run_detail_uses_persisted_funnel_counts_and_config(tmp_path: Path) -> None:
    detail = _seed_authoritative_state(tmp_path).run_detail("US-SH-LIVE")

    assert detail["account"] == "U123456"
    assert detail["risk_per_trade"] == 0.001
    assert detail["strategy_version"] is None
    assert detail["funnel"] == [
        {"stage": "Universe eligibility", "count": 2},
        {"stage": "Stock eligibility", "count": 2},
        {"stage": "Required data ready", "count": 1},
        {"stage": "Screened", "count": 1},
        {"stage": "Qualified", "count": 1},
        {"stage": "Vetoed", "count": 0},
        {"stage": "Armed", "count": 0},
        {"stage": "Entry triggered", "count": 1},
        {"stage": "Orders", "count": 1},
        {"stage": "Positions", "count": 1},
    ]


def test_run_summary_does_not_load_full_candidate_or_universe_history(tmp_path, monkeypatch):
    reads = _seed_authoritative_state(tmp_path)
    load = reads.runtime_store.load_signals

    def bounded_load(run_id, **filters):
        assert filters.get("signal_ids") is not None
        return load(run_id, **filters)

    def full_provenance(*args):
        raise AssertionError("full audit records belong to the on-demand endpoint")

    monkeypatch.setattr(reads.runtime_store, "load_signals", bounded_load)
    monkeypatch.setattr(reads.runtime_store, "method_run", full_provenance)
    detail = reads.run_detail("US-SH-LIVE")
    assert "provenance" not in detail
    assert detail["funnel"][1]["count"] == 2


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


def test_disabled_run_candidates_stay_in_audit_storage_but_not_operational_views(
    tmp_path: Path,
) -> None:
    service = _seed_authoritative_state(tmp_path)
    source = service.stage5_store.get("NASDAQ", NOW, 101)
    assert source is not None
    service.stage5_store.save(
        replace(
            source,
            run_ids=("US-SH-PAPER",),
            con_id=202,
            symbol="AMD",
        )
    )
    service.config = RunsConfig(
        universes=service.config.universes,
        runs=tuple(reversed(service.config.runs)),
    )

    disabled = next(item for item in service.runs() if item["run_id"] == "US-SH-PAPER")
    explicit = service.candidates(
        run_id="US-SH-PAPER",
        session=date(2026, 9, 2),
    )
    default = service.candidates(session=date(2026, 9, 2))

    assert disabled["candidate_count"] == 0
    assert explicit == {"items": [], "total": 0, "limit": 100, "offset": 0}
    assert default["items"][0]["symbol"] == "NVDA"
    assert service.stage5_store.get("NASDAQ", NOW, 202) is not None


def test_orders_positions_and_trades_preserve_account_environment(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)

    orders = service.orders(scope="all", limit=25)
    positions = service.positions()
    trades = service.trades(environment=None, start=None, end=None, limit=25)

    assert orders["items"][0]["environment"] == "LIVE"
    assert orders["items"][0]["account"] == "U123456"
    assert [child["role"] for child in orders["items"][0]["orders"]] == ["ENTRY", "STOP", "TARGET"]
    assert positions[0]["source"] == "IBKR"
    assert positions[0]["environment"] == "LIVE"
    assert positions[0]["account"] == "U123456"
    assert trades["items"] == []
    assert trades["summary"]["trades"] == 0


def test_archived_runs_leave_operational_lists_but_keep_history(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    service.ledger.record_fill(
        BrokerFill(
            execution_id="archive-target",
            order_id=102,
            account="U123456",
            environment=Environment.LIVE,
            con_id=101,
            symbol="NVDA",
            side=OrderAction.BUY,
            quantity=12,
            price=177.80,
            executed_at=NOW,
        )
    )
    service.config = service.config.model_copy(
        update={
            "runs": tuple(
                run.model_copy(update={"enabled": False, "archived": True})
                for run in service.config.runs
            )
        }
    )
    assert service.runs() == []
    assert service.universe_runs() == {"PAPER": [], "LIVE": []}
    trades = service.trades(environment=None, start=None, end=None)
    assert trades["total"] == 1
    assert trades["items"][0]["run_id"] == "US-SH-LIVE"
    assert service.order_detail("plan-live-nvda")["run_id"] == "US-SH-LIVE"
    payload = service.config.runs[0].model_dump()
    payload["enabled"] = True
    with pytest.raises(ValueError, match="Archived runs cannot be enabled"):
        RunConfig.model_validate(payload)


def test_unknown_broker_position_remains_visible_without_invented_lineage(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    service.ledger.replace_broker_snapshot(
        environment=Environment.LIVE,
        account="U123456",
        positions=(
            BrokerPosition("U123456", 101, "NVDA", -12, 179.68),
            BrokerPosition("U123456", 202, "UNKNOWN", 5, 25.0),
        ),
        open_orders=(),
        observed_at=NOW,
    )

    unknown = next(item for item in service.positions() if item["symbol"] == "UNKNOWN")

    assert unknown["source"] == "IBKR"
    assert unknown["run_id"] is None
    assert unknown["order_plan_id"] is None
    assert unknown["side"] == "LONG"


def test_trade_filters_and_summary_cover_the_complete_filtered_ledger(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    assert service.ledger.record_fill(
        BrokerFill(
            execution_id="fill-target-1",
            order_id=102,
            account="U123456",
            environment=Environment.LIVE,
            con_id=101,
            symbol="NVDA",
            side=OrderAction.BUY,
            quantity=12,
            price=178.0,
            executed_at=NOW + timedelta(minutes=1),
        )
    )

    trades = service.trades(
        environment=Environment.LIVE,
        start=NOW,
        end=NOW + timedelta(minutes=2),
        run_id="US-SH-LIVE",
        strategy="TEST_EXECUTION",
        universe="NASDAQ",
        symbol="nvda",
        limit=1,
    )
    excluded = service.trades(
        environment=None,
        start=None,
        end=None,
        universe="NOT_CONFIGURED",
    )

    assert trades["total"] == 1
    assert trades["items"][0]["universe"] == "NASDAQ"
    assert trades["summary"]["trades"] == 1
    assert trades["summary"]["wins"] == 1
    assert trades["summary"]["total_pnl"] == pytest.approx(20.16)
    assert excluded["total"] == 0
    assert excluded["summary"]["trades"] == 0


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


def test_universe_builder_options_do_not_reload_the_large_runs_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)

    def unexpected_load(_path: Path) -> RunsConfig:
        raise AssertionError("builder options must not parse the runs config")

    monkeypatch.setattr("stocker_dashboard.controls.load_runs_config", unexpected_load)

    options = asyncio.run(controls.universe_builder_options())

    assert options["markets"]
    assert options["strategies"]


def test_dashboard_run_write_preserves_named_snapshot_without_expanding_members(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "us-listed.csv"
    snapshot_path.write_text(
        """# schema_version=1
# source=NASDAQ_TRADER_SYMBOL_DIRECTORY
# source_urls=https://example.test/nasdaq.txt,https://example.test/other.txt
# retrieved_at=2026-09-02T12:00:00+00:00
# nasdaq_source_updated_at=0902202611:55
# other_source_updated_at=0902202611:56
# universes=US_ALL,NASDAQ,NYSE
symbol,primary_exchange
AAPL,NASDAQ
MSFT,NASDAQ
IBM,NYSE
""",
        encoding="utf-8",
    )
    runs_path, broker_path = _write_control_files(tmp_path)
    runs_path.write_text(
        f"named_universe_snapshot: {snapshot_path.name}\nuniverses: []\nruns: []\n",
        encoding="utf-8",
    )
    controls = RunControlService(runs_path, broker_path)

    asyncio.run(
        controls.add_universe_run(
            market_id="US_NASDAQ",
            strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
            strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
            environment=Environment.PAPER,
            risk_per_trade=0.001,
            max_concurrent_positions=1,
        )
    )

    persisted = yaml.safe_load(runs_path.read_text(encoding="utf-8"))
    assert persisted["named_universe_snapshot"] == snapshot_path.name
    assert [item["universe_id"] for item in persisted["universes"]] == [
        "US_NASDAQ_OPENING_CANDIDATES_V1"
    ]
    assert persisted["universes"][0]["members"] == []
    reloaded = load_runs_config(runs_path)
    nasdaq = next(item for item in reloaded.universes if item.universe_id == "NASDAQ")
    derived = next(
        item for item in reloaded.universes if item.universe_id == "US_NASDAQ_OPENING_CANDIDATES_V1"
    )
    assert nasdaq.members  # The fixed catalogue remains available for explicit fixed sources.
    assert len(derived.members) == 2
    assert reloaded.runs[0].uses_candidate_selection
    assert len(reloaded.runs[0].universe_snapshot.members) == 2


def test_run_controls_validate_through_backend_and_require_live_confirmation(
    tmp_path: Path,
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)

    asyncio.run(controls.disable_run("US-SH-LIVE"))
    asyncio.run(
        controls.enable_run(
            "US-SH-LIVE",
            confirmation=LiveConfirmation(confirmed=True, target_account="U123456"),
        )
    )
    asyncio.run(
        controls.update_run_config(
            "US-SH-PAPER",
            risk_per_trade=0.003,
            max_concurrent_positions=4,
            universe="NASDAQ",
            strategy="TEST_EXECUTION",
        )
    )

    with pytest.raises(ValueError, match="LIVE confirmation"):
        asyncio.run(controls.change_execution_environment("US-SH-PAPER", Environment.LIVE))
    changed = asyncio.run(
        controls.change_execution_environment(
            "US-SH-PAPER",
            Environment.LIVE,
            confirmation=LiveConfirmation(confirmed=True, target_account="U123456"),
        )
    )

    loaded = load_runs_config(runs_path)
    assert changed.run is not None
    assert changed.run.environment is Environment.LIVE
    assert changed.run.strategy == "TEST_EXECUTION"
    assert loaded.runs[1].risk == RunRiskConfig(risk_per_trade=0.003, max_concurrent_positions=4)


def test_run_controls_update_session_hard_hv_risk_without_changing_lineage(
    tmp_path: Path,
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)
    created = asyncio.run(
        controls.add_universe_run(
            market_id="US_NASDAQ",
            strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
            strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
            environment=Environment.PAPER,
            risk_per_trade=0.001,
            max_concurrent_positions=1,
        )
    )
    assert created.run is not None

    updated = asyncio.run(
        controls.update_run_config(
            created.run.run_id,
            risk_per_trade=0.002,
            max_concurrent_positions=3,
            universe=created.run.universe,
            strategy=SESSION_HARD_HV_METHOD.config_name,
        )
    )

    assert updated.run is not None
    assert updated.run.strategy_id == SESSION_HARD_HV_METHOD.strategy_id
    assert updated.run.strategy_version == SESSION_HARD_HV_METHOD.strategy_version
    assert updated.run.risk == RunRiskConfig(
        risk_per_trade=0.002,
        max_concurrent_positions=3,
    )


def test_invalid_live_account_and_invalid_config_are_rejected(tmp_path: Path) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)

    with pytest.raises(ValueError, match="target account"):
        asyncio.run(
            controls.change_execution_environment(
                "US-SH-PAPER",
                Environment.LIVE,
                confirmation=LiveConfirmation(confirmed=True, target_account="WRONG"),
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(
            controls.update_run_config(
                "US-SH-PAPER",
                risk_per_trade=-1,
                max_concurrent_positions=4,
                universe="NASDAQ",
                strategy="TEST_EXECUTION",
            )
        )


def test_control_service_persists_and_returns_authoritative_runtime_state(
    tmp_path: Path,
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    runtime = RecordingRuntime()
    controls = RunControlService(runs_path, broker_path, runtime=runtime)

    result = asyncio.run(controls.enable_run("US-SH-PAPER"))

    persisted = next(run for run in load_runs_config(runs_path).runs if run.run_id == "US-SH-PAPER")
    runtime_run = (
        next(run for run in result.runtime.runs if run.run_id == "US-SH-PAPER")
        if result.runtime
        else None
    )
    assert persisted.enabled is True
    assert runtime.run_updates[-1][1] == {"US-SH-PAPER"}
    assert result.runtime_applied is True
    assert runtime_run is not None and runtime_run.state is RunRuntimeState.ACTIVE


def test_rejected_runtime_change_is_not_persisted_or_shown_as_applied(tmp_path: Path) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path, runtime=RejectingRuntime())

    result = asyncio.run(controls.disable_run("US-SH-LIVE"))

    persisted = next(run for run in load_runs_config(runs_path).runs if run.run_id == "US-SH-LIVE")
    assert persisted.enabled is True
    assert result.persisted is False
    assert result.runtime_applied is False


def test_broker_config_edit_persists_and_targets_only_selected_environment(
    tmp_path: Path,
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    runtime = RecordingRuntime()
    controls = RunControlService(runs_path, broker_path, runtime=runtime)
    paper = IbkrConfig(
        environment=Environment.PAPER,
        host="paper-gateway.local",
        port=4002,
        client_id=31,
        expected_account="DU123456",
        market_data_line_budget=80,
    )

    result = asyncio.run(controls.update_broker_config(paper))

    assert load_ibkr_config(broker_path, Environment.PAPER) == paper
    assert load_ibkr_config(broker_path, Environment.LIVE).host == "127.0.0.1"
    assert runtime.broker_updates == [paper]
    assert result.apply_mode == "RECONNECT_ENVIRONMENT"


def test_conflicting_broker_session_identity_is_rejected(tmp_path: Path) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)

    with pytest.raises(ValueError, match="same IBKR session identity"):
        asyncio.run(
            controls.update_broker_config(
                IbkrConfig(
                    environment=Environment.PAPER,
                    host="127.0.0.1",
                    port=4001,
                    client_id=22,
                    expected_account="DU123456",
                )
            )
        )


def test_enabled_paper_broker_requires_expected_account(tmp_path: Path) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)

    with pytest.raises(ValueError, match="PAPER expected_account is required"):
        asyncio.run(
            controls.update_broker_config(
                IbkrConfig(
                    environment=Environment.PAPER,
                    host="127.0.0.1",
                    port=4002,
                    client_id=31,
                    expected_account=None,
                )
            )
        )


def test_runtime_rejected_broker_change_leaves_persisted_config_unchanged(
    tmp_path: Path,
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path, runtime=RejectingRuntime())
    proposed = IbkrConfig(
        environment=Environment.PAPER,
        host="paper-gateway.local",
        port=4002,
        client_id=31,
        expected_account="DU654321",
    )

    result = asyncio.run(controls.update_broker_config(proposed))

    assert load_ibkr_config(broker_path, Environment.PAPER).expected_account == "DU123456"
    assert result.persisted is False
    assert result.runtime_applied is False


def test_custom_universe_edit_normalizes_persists_and_refreshes_affected_run(
    tmp_path: Path,
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    runtime = RecordingRuntime()
    controls = RunControlService(runs_path, broker_path, runtime=runtime)
    asyncio.run(
        controls.replace_custom_universe_symbols(
            "custom_ops", [" aapl ", "AAPL", "msft"], name="Operations"
        )
    )
    asyncio.run(
        controls.update_run_config(
            "US-SH-PAPER",
            risk_per_trade=0.002,
            max_concurrent_positions=3,
            universe="CUSTOM_OPS",
            strategy="TEST_EXECUTION",
        )
    )
    asyncio.run(controls.enable_run("US-SH-PAPER"))

    result = asyncio.run(controls.replace_custom_universe_symbols("CUSTOM_OPS", ["msft", " nvda "]))

    universe = next(
        item for item in load_runs_config(runs_path).universes if item.universe_id == "CUSTOM_OPS"
    )
    assert [member.symbol for member in universe.members] == ["MSFT", "NVDA"]
    assert runtime.run_updates[-1][1] == {"US-SH-PAPER"}
    assert result.runtime_applied is True


def test_custom_universe_edit_reports_authoritative_degraded_run(
    tmp_path: Path,
) -> None:
    runs_path, broker_path = _write_control_files(tmp_path)
    config = load_runs_config(runs_path)
    custom = UniverseDefinition(
        universe_id="CUSTOM_OPS",
        name="Operations",
        members=(
            InstrumentReference(
                symbol="AAPL",
                exchange="SMART",
                primary_exchange="NASDAQ",
                currency="USD",
            ),
        ),
    )
    updated = RunsConfig(
        universes=(*config.universes, custom),
        runs=tuple(
            run.model_copy(update={"enabled": True, "universe": "CUSTOM_OPS"})
            if run.run_id == "US-SH-PAPER"
            else run
            for run in config.runs
        ),
    )
    runs_path.write_text(yaml.safe_dump(updated.model_dump(mode="json")), encoding="utf-8")
    controls = RunControlService(runs_path, broker_path, runtime=DegradingRuntime())

    result = asyncio.run(controls.replace_custom_universe_symbols("CUSTOM_OPS", ["AAPL", "MSFT"]))

    assert result.persisted is True
    assert result.runtime_applied is True
    assert result.detail == ("Saved and applied; runtime degraded. Inspect run diagnostics.")


def test_settings_http_api_exposes_editable_broker_and_custom_universe_fields(
    tmp_path: Path,
) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)
    client = TestClient(create_dashboard_app(service, controls))

    broker_response = client.put(
        "/api/settings/broker/PAPER",
        json={
            "host": "paper-gateway.local",
            "port": 4002,
            "client_id": 31,
            "expected_account": "DU123456",
            "market_data_line_budget": 80,
        },
    )
    universe_response = client.put(
        "/api/settings/universes/CUSTOM_DESK",
        json={"name": "Desk", "symbols": [" aapl ", "AAPL", "msft"]},
    )

    assert broker_response.status_code == 200
    assert broker_response.json()["runtime_applied"] is False
    assert universe_response.status_code == 200
    settings = client.get("/api/settings").json()
    assert settings["broker_configuration"][0]["host"] == "paper-gateway.local"
    assert settings["broker_configuration"][0]["market_data_line_budget"] == 80
    assert [member["symbol"] for member in settings["custom_universes"][0]["members"]] == [
        "AAPL",
        "MSFT",
    ]


@pytest.mark.parametrize("failure", ["exception", "failed_lifespan"])
def test_integrated_dashboard_startup_failure_does_not_stop_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import uvicorn

    import stocker_dashboard.factory
    import stocker_execution.runtime

    class FakeRuntime:
        def __init__(self) -> None:
            self.stopping = False
            self.stop_calls = 0

        async def run_forever(self, *, poll_interval_seconds: float) -> None:
            while not self.stopping:
                await asyncio.sleep(0)

        async def stop(self) -> None:
            self.stop_calls += 1
            self.stopping = True

    class FakeConfig:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    runtime = FakeRuntime()

    class FakeServer:
        attempts = 0

        def __init__(self, _config: object) -> None:
            pass

        async def serve(self) -> None:
            type(self).attempts += 1
            if self.attempts == 1:
                if failure == "exception":
                    raise SystemExit(1)
                self.started = False
                return
            self.started = True
            assert runtime.stop_calls == 0

    monkeypatch.setattr(stocker_execution.runtime, "build_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(
        stocker_dashboard.factory, "build_dashboard_app", lambda **_kwargs: object()
    )
    monkeypatch.setattr(uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)

    stage10_run(
        runs_config=tmp_path / "runs.yaml",
        ibkr_config=tmp_path / "ibkr.yaml",
        database=tmp_path / "runtime.sqlite3",
        host="127.0.0.1",
        port=8000,
        poll_seconds=0.001,
    )

    assert FakeServer.attempts == 2
    assert runtime.stop_calls == 1


def test_http_routes_and_all_navigation_pages_render(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    app = create_dashboard_app(service, RunControlService(runs_path, broker_path))
    client = TestClient(app)

    assert client.get("/api/overview").json()["system"] == "READY"
    assert client.get("/api/orders/plan-live-nvda").json()["signal_id"] == "signal-live-nvda"
    audit = client.get("/api/runs/US-SH-LIVE/provenance")
    assert audit.status_code == 200
    assert audit.headers["content-disposition"] == 'attachment; filename="run-provenance.json"'
    assert client.get("/api/positions/LIVE/U123456/101").json()["order_plan_id"] == "plan-live-nvda"
    for route in (
        "/",
        "/runs",
        "/universes",
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
    assert 'src="/static/dashboard.js?v=20260909-activity-filter"' in index
    for label in (
        "Overview",
        "Runs",
        "Start run",
        "Candidates",
        "Orders",
        "Positions",
        "Trades",
        "System",
        "Settings",
    ):
        assert f">{label}<" in index

    script = client.get("/static/dashboard.js").text
    assert "IBKR API RESOURCES" in script
    assert "Stocker market-data budget" in script


def test_periodic_refresh_preserves_interactive_page_dom(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    client = TestClient(create_dashboard_app(service, RunControlService(runs_path, broker_path)))

    script = client.get("/static/dashboard.js").text

    assert (
        'const INTERACTIVE_ROUTES = new Set(["universes", "candidates", "trades", "settings"]);'
        in script
    )
    assert "if (INTERACTIVE_ROUTES.has(route()))" in script
    assert "timer = setTimeout(refreshCurrentPage" in script
    assert "timer = setTimeout(render" not in script


def test_background_refresh_preserves_last_good_page_on_fetch_failure(
    tmp_path: Path,
) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    client = TestClient(create_dashboard_app(service, RunControlService(runs_path, broker_path)))

    script = client.get("/static/dashboard.js").text

    assert "let pageHasRendered = false;" in script
    assert "pageHasRendered = true;" in script
    assert "if (pageHasRendered) showRefreshWarning();" in script
    assert "Dashboard update delayed. Showing last known data" in script
    assert "clearRefreshWarning();" in script


def test_universe_builder_http_flow_rejects_legacy_values_and_live(tmp_path):
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    client = TestClient(create_dashboard_app(service, RunControlService(runs_path, broker_path)))
    body = {
        "market_id": "US_NASDAQ",
        "strategy_id": SESSION_HARD_HV_METHOD.strategy_id,
        "strategy_version": SESSION_HARD_HV_METHOD.strategy_version,
        "risk_per_trade": 0.001,
        "max_concurrent_positions": 2,
    }
    assert (
        client.post("/api/universe-runs/paper", json={**body, "cap_bucket": "MID"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/universe-runs/paper", json={**body, "strategy_version": "SESSION_HARD_HV_V1"}
        ).status_code
        == 400
    )
    paper = client.post("/api/universe-runs/paper", json=body)
    assert paper.status_code == 200
    live = client.post(
        "/api/universe-runs/live", json={**body, "confirmed": True, "target_account": "U123456"}
    )
    assert live.status_code == 400
    rows = client.get("/api/universe-runs").json()["PAPER"]
    run = next(row for row in rows if row["market_id"] == "US_NASDAQ")
    assert run["display_name"] == "NASDAQ · Session HARD"
    assert client.post(f"/api/universe-runs/{run['run_id']}/disable").status_code == 200


def test_start_acknowledges_before_slow_qualification_and_survives_page_reload(tmp_path):
    async def scenario():
        gate = asyncio.Event()

        class SlowRuntime(RecordingRuntime):
            async def apply_runs_config(self, config, *, changed_run_ids):
                await gate.wait()
                return await super().apply_runs_config(config, changed_run_ids=changed_run_ids)

        reads = _seed_authoritative_state(tmp_path)
        runs_path, broker_path = _write_control_files(tmp_path)
        runtime = SlowRuntime()
        app = create_dashboard_app(
            reads, RunControlService(runs_path, broker_path, runtime=runtime)
        )
        body = {
            "market_id": "US_NASDAQ",
            "strategy_id": SESSION_HARD_HV_METHOD.strategy_id,
            "strategy_version": SESSION_HARD_HV_METHOD.strategy_version,
            "risk_per_trade": 0.001,
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            response = await asyncio.wait_for(
                client.post("/api/universe-runs/paper?background=true", json=body), 1
            )
            assert response.status_code == 202
            assert response.json()["status"] == "STARTING"
            duplicate = await client.post("/api/universe-runs/paper?background=true", json=body)
            assert duplicate.json()["operation_id"] == response.json()["operation_id"]
            assert (await client.get("/api/universe-runs/start-status")).json()[
                "status"
            ] == "STARTING"
            gate.set()
            for _ in range(100):
                await asyncio.sleep(0.01)
                status = (await client.get("/api/universe-runs/start-status")).json()
                if status["status"] != "STARTING":
                    break
            assert status["status"] == "COMPLETED"
            assert status["result"]["persisted"]
            assert len(runtime.run_updates) == 1
            rows = (await client.get("/api/universe-runs")).json()["PAPER"]
            assert any(r["market_id"] == "US_NASDAQ" for r in rows)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_writer", [False, True])
def test_pause_preserves_identity_during_inflight_save(tmp_path, monkeypatch, cancel_writer):
    import threading

    async def scenario():
        loop = asyncio.get_running_loop()
        writing = asyncio.Event()
        paused = asyncio.Event()
        release = threading.Event()

        class PausingRuntime(RecordingRuntime):
            def pause_new_entries(self, run_id):
                paused.set()
                return self.status()

        runs_path, broker_path = _write_control_files(tmp_path)
        controls = RunControlService(runs_path, broker_path, runtime=PausingRuntime())
        config = load_runs_config(runs_path)
        new_run = config.runs[1].model_copy(update={"run_id": "NEW-PAPER"})
        updated = config.model_copy(update={"runs": (*config.runs, new_run)})
        original = controls._write_runs
        writes = []

        def held(config):
            writes.append(config)
            if len(writes) == 1:
                loop.call_soon_threadsafe(writing.set)
                assert release.wait(10)
            original(config)

        monkeypatch.setattr(controls, "_write_runs", held)
        saving = asyncio.create_task(controls._save_runs(updated))
        await asyncio.wait_for(writing.wait(), 5)
        if cancel_writer:
            saving.cancel()
        pausing = asyncio.create_task(controls.disable_run("US-SH-LIVE"))
        try:
            await asyncio.wait_for(paused.wait(), 5)
            assert len(writes) == 1
            assert not pausing.done()
        finally:
            release.set()
        if cancel_writer:
            with pytest.raises(asyncio.CancelledError):
                await saving
        else:
            await saving
        assert (await pausing).persisted
        saved = load_runs_config(runs_path)
        assert [r.run_id for r in saved.runs] == [r.run_id for r in updated.runs]
        assert not next(r for r in saved.runs if r.run_id == "US-SH-LIVE").enabled

    asyncio.run(scenario())


def test_universe_builder_default_risk_is_valid_for_html_number_input(
    tmp_path: Path,
) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    client = TestClient(create_dashboard_app(service, RunControlService(runs_path, broker_path)))

    script = client.get("/static/dashboard.js").text
    match = re.search(
        r'name="risk_per_trade"[^>]*min="([^"]+)"[^>]*max="([^"]+)"'
        r'[^>]*step="([^"]+)"[^>]*value="([^"]+)"',
        script,
    )
    assert match is not None
    minimum, maximum, step, default = match.groups()
    default_value = Decimal(default)
    assert Decimal(minimum) <= default_value <= Decimal(maximum)
    if step != "any":
        assert (default_value - Decimal(minimum)) % Decimal(step) == 0


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


def test_universes_builder_disables_strategies_outside_declared_environments(
    tmp_path: Path,
) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    client = TestClient(create_dashboard_app(service, RunControlService(runs_path, broker_path)))

    script = client.get("/static/dashboard.js").text

    assert "data-environments" in script
    assert 'includes("LIVE")' in script
    assert "is PAPER-only" in script


def test_candidates_page_only_offers_enabled_runs(tmp_path: Path) -> None:
    service = _seed_authoritative_state(tmp_path)
    runs_path, broker_path = _write_control_files(tmp_path)
    client = TestClient(create_dashboard_app(service, RunControlService(runs_path, broker_path)))

    script = client.get("/static/dashboard.js").text

    assert 'const runs = (await api("/api/runs")).filter((run) => run.enabled);' in script


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


def test_first_run_write_failure_never_activates_identity(tmp_path, monkeypatch):
    runs_path, broker_path = _write_control_files(tmp_path)
    runtime = RecordingRuntime()
    controls = RunControlService(runs_path, broker_path, runtime=runtime)
    before = runs_path.read_bytes()

    def fail(config):
        raise OSError("injected write failure")

    monkeypatch.setattr(controls, "_write_runs", fail)
    with pytest.raises(OSError, match="injected"):
        asyncio.run(
            controls.add_universe_run(
                market_id=MarketId.US_NASDAQ,
                strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
                strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
                environment=Environment.PAPER,
                risk_per_trade=0.001,
                max_concurrent_positions=1,
            )
        )
    assert runtime.run_updates == []
    assert runs_path.read_bytes() == before


def test_saved_identity_survives_activation_failure_and_retry(tmp_path):
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path, runtime=RejectingRuntime())
    result = asyncio.run(
        controls.add_universe_run(
            market_id=MarketId.US_NASDAQ,
            strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
            strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
            environment=Environment.PAPER,
            risk_per_trade=0.001,
            max_concurrent_positions=1,
        )
    )
    assert result.persisted and not result.runtime_applied
    assert "Restart will retry" in result.detail
    saved = load_runs_config(runs_path)
    assert sum(r.run_id == result.run.run_id for r in saved.runs) == 1
    # A disconnected restart reads the same identity; no execution is initiated.
    from stocker_dashboard.factory import build_dashboard_app

    with TestClient(
        build_dashboard_app(
            runs_config_path=runs_path,
            ibkr_config_path=broker_path,
            database_path=tmp_path / "restart.sqlite",
        )
    ) as client:
        rows = client.get("/api/runs").json()
    assert next(r for r in rows if r["run_id"] == result.run.run_id)["status"] == "STOPPED"


def test_retry_saved_partial_activation_reuses_identity(tmp_path):
    class PartialRuntime(RecordingRuntime):
        failed = False

        async def apply_runs_config(self, config, *, changed_run_ids):
            status = await super().apply_runs_config(config, changed_run_ids=changed_run_ids)
            if not self.failed:
                self.failed = True
                self._status = replace(
                    status,
                    runs=tuple(replace(r, state=RunRuntimeState.DEGRADED) for r in status.runs),
                )
                raise RuntimeError("injected activation interruption")
            return status

    async def scenario():
        runs_path, broker_path = _write_control_files(tmp_path)
        runtime = PartialRuntime()
        controls = RunControlService(runs_path, broker_path, runtime=runtime)
        body = dict(
            market_id=MarketId.US_NASDAQ,
            strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
            strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
            environment=Environment.PAPER,
            risk_per_trade=0.001,
            max_concurrent_positions=1,
            max_gross_notional=100000,
        )
        first = await controls.add_universe_run(**body)
        assert first.persisted and not first.runtime_applied
        second = await controls.add_universe_run(**body)
        assert second.persisted and second.runtime_applied
        assert second.run.run_id == first.run.run_id
        assert len(runtime.run_updates) == 2
        assert sum(r.run_id == first.run.run_id for r in load_runs_config(runs_path).runs) == 1

    asyncio.run(scenario())


def test_control_exception_returns_safe_correlated_error(tmp_path, monkeypatch):
    runs_path, broker_path = _write_control_files(tmp_path)
    controls = RunControlService(runs_path, broker_path)

    async def fail(run_id):
        raise RuntimeError("password=NEVER-EXPOSE-THIS")

    monkeypatch.setattr(controls, "disable_run", fail)
    with TestClient(
        create_dashboard_app(_seed_authoritative_state(tmp_path), controls),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/api/runs/US-SH-PAPER/disable")
    assert response.status_code == 503
    assert "NEVER-EXPOSE" not in response.text
    assert "reference" in response.json()["detail"]
    assert "Control" in response.json()["detail"]


@pytest.mark.parametrize("phase", ["read", "write", "response"])
def test_pause_gates_before_disk_work_and_keeps_http_responsive(tmp_path, monkeypatch, phase):
    import threading

    import stocker_dashboard.app as app_module
    import stocker_dashboard.controls as controls_module

    async def scenario():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        paused = asyncio.Event()
        entered = asyncio.Event()
        release = threading.Event()

        class PausingRuntime(RecordingRuntime):
            def pause_new_entries(self, run_id):
                paused.set()
                return self.status()

        runs_path, broker_path = _write_control_files(tmp_path)
        service = _seed_authoritative_state(tmp_path)
        controls = RunControlService(runs_path, broker_path, runtime=PausingRuntime())
        if phase == "write":
            owner, name = controls, "_write_runs"
        else:
            owner = controls_module if phase == "read" else app_module
            name = "load_runs_config"
        original = getattr(owner, name)

        def held(*args, **kwargs):
            assert paused.is_set(), "Pause must fence entry before reading configuration"
            assert threading.get_ident() != loop_thread, "YAML work blocked the runtime loop"
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "Test did not release disk work"
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, held)
        app = create_dashboard_app(service, controls)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            request = asyncio.create_task(client.post("/api/runs/US-SH-LIVE/disable"))
            waiting = asyncio.create_task(entered.wait())
            try:
                done, _ = await asyncio.wait(
                    [request, waiting], timeout=10, return_when=asyncio.FIRST_COMPLETED
                )
                if request in done:
                    await request  # Surface the exact blocked-thread/pause-order assertion.
                assert waiting in done
                assert paused.is_set()
                response = await asyncio.wait_for(client.get("/static/dashboard.css"), 5)
                assert response.status_code == 200
            finally:
                release.set()
                waiting.cancel()
            response = await request
            assert response.status_code == 200
            assert response.json()["persisted"]
            assert response.json()["runtime_applied"]
            assert not next(
                r for r in load_runs_config(runs_path).runs if r.run_id == "US-SH-LIVE"
            ).enabled
            assert not next(r for r in service.config.runs if r.run_id == "US-SH-LIVE").enabled

    asyncio.run(scenario())
