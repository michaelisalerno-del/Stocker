from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from stocker_core.markets import CapBucket, MarketId
from stocker_core.runs import CandidateScreen, Environment, RunConfig, RunScreenConfig
from stocker_dashboard.performance import PerformancePeriod, RunPerformanceService
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerFill,
    BrokerOrderIds,
    BrokerPosition,
    EntryOrderType,
    OrderAction,
    OrderPlan,
)


def run(run_id: str, environment: Environment = Environment.PAPER) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        universe="US_NASDAQ_MID_CAP_BUCKETS_V1",
        strategy="SESSION_HARD_HV",
        strategy_id="SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
        strategy_version="SESSION_HARD_HV_V1",
        market_id=MarketId.US_NASDAQ,
        cap_bucket=CapBucket.MID,
        cap_bucket_version="CAP_BUCKETS_V1",
        candidate_screen_id="ACTIVITY_SHORTLIST_V1",
        candidate_screen_version="ACTIVITY_SHORTLIST_V1",
        screen=RunScreenConfig(
            method=CandidateScreen.ACTIVITY_SHORTLIST_V1,
            max_results=50,
            version="ACTIVITY_SHORTLIST_V1",
            scheduled_active_minutes=15,
        ),
        display_name="NASDAQ · HARD · MID",
        environment=environment,
    )


def plan(
    run_id: str,
    plan_id: str,
    *,
    con_id: int,
    quantity: int,
    entry: float,
    environment: Environment = Environment.PAPER,
    created_at: datetime,
) -> OrderPlan:
    return OrderPlan(
        order_plan_id=plan_id,
        run_id=run_id,
        signal_id=f"signal-{plan_id}",
        strategy_id="SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
        strategy_version="SESSION_HARD_HV_V1",
        con_id=con_id,
        symbol="TEST",
        side=OrderAction.SELL,
        quantity=quantity,
        entry_order_type=EntryOrderType.MARKET,
        entry_reference=entry,
        stop_price=entry + 2,
        target_price=entry - 4,
        environment=environment,
        created_at=created_at,
        initial_risk_budget=quantity * 2,
        per_share_initial_risk=2,
    )


def submit(ledger: ExecutionLedger, item: OrderPlan, account: str, base_order_id: int) -> None:
    assert ledger.reserve(item, expected_account=account)
    ledger.record_submission(
        item.order_plan_id,
        BrokerOrderIds(base_order_id, base_order_id + 1, base_order_id + 2),
        actual_account=account,
    )


def fill(
    ledger: ExecutionLedger,
    item: OrderPlan,
    *,
    account: str,
    order_id: int,
    side: OrderAction,
    price: float,
    execution_id: str,
    commission: float = 0,
    when: datetime,
) -> None:
    assert ledger.record_fill(
        BrokerFill(
            execution_id,
            order_id,
            account,
            item.environment,
            item.con_id,
            item.symbol,
            side,
            item.quantity,
            price,
            when,
            commission,
            item.order_plan_id,
        )
    )


def test_realised_performance_is_run_environment_commission_and_original_risk_attributed(
    tmp_path: Path,
) -> None:
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    opened = datetime(2026, 9, 1, 14, tzinfo=UTC)
    closed = datetime(2026, 9, 2, 15, tzinfo=UTC)
    paper = plan("paper", "paper-plan", con_id=1, quantity=10, entry=100, created_at=opened)
    live = plan(
        "live",
        "live-plan",
        con_id=2,
        quantity=10,
        entry=100,
        environment=Environment.LIVE,
        created_at=opened,
    )
    for item, account, base in ((paper, "DU1", 10), (live, "U1", 20)):
        submit(ledger, item, account, base)
        fill(
            ledger,
            item,
            account=account,
            order_id=base,
            side=OrderAction.SELL,
            price=100,
            execution_id=f"{base}-entry",
            commission=1,
            when=opened,
        )
        fill(
            ledger,
            item,
            account=account,
            order_id=base + 2,
            side=OrderAction.BUY,
            price=96,
            execution_id=f"{base}-exit",
            commission=1,
            when=closed,
        )

    service = RunPerformanceService(ledger, clock=lambda: closed)
    result = service.performance(run("paper"), PerformancePeriod.ALL)

    assert result["closed_trades"] == 1
    assert result["wins"] == 1
    assert result["losses"] == 0
    assert result["realised_pnl"] == 38
    assert result["total_r"] == 1.9
    assert result["mean_r"] == 1.9
    assert result["currency"] == "USD"
    assert result["history"] == [{"date": "2026-09-02", "trades": 1, "pnl": 38, "r": 1.9}]


def test_same_conid_multiple_runs_get_lot_attribution_only_after_broker_reconciliation(
    tmp_path: Path,
) -> None:
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 9, 2, 15, tzinfo=UTC)
    first = plan("one", "one-plan", con_id=7, quantity=10, entry=100, created_at=now)
    second = plan("two", "two-plan", con_id=7, quantity=5, entry=110, created_at=now)
    for item, base in ((first, 10), (second, 20)):
        # Reconstruct legacy duplicate lots directly; current admission must reject these.
        import sqlite3
        from dataclasses import replace

        submit(ledger, replace(item, con_id=base), "DU1", base)
        with sqlite3.connect(ledger.path) as connection:
            connection.execute(
                "UPDATE execution_plans SET con_id = ? WHERE order_plan_id = ?",
                (item.con_id, item.order_plan_id),
            )
        fill(
            ledger,
            item,
            account="DU1",
            order_id=base,
            side=OrderAction.SELL,
            price=item.entry_reference,
            execution_id=f"{base}-entry",
            when=now,
        )
    ledger.replace_broker_snapshot(
        environment=Environment.PAPER,
        account="DU1",
        positions=(BrokerPosition("DU1", 7, "TEST", -15, 103.333333),),
        open_orders=(),
        observed_at=now,
    )
    ledger.record_position_mark(
        environment=Environment.PAPER,
        account="DU1",
        con_id=7,
        mark=90,
        observed_at=now,
    )
    service = RunPerformanceService(ledger, clock=lambda: now)
    one = service.performance(run("one"), PerformancePeriod.TODAY)
    two = service.performance(run("two"), PerformancePeriod.TODAY)
    assert one["unrealised_pnl"] == 100
    assert two["unrealised_pnl"] == 100
    assert one["unrealised_status"] == "AVAILABLE"

    ledger.replace_broker_snapshot(
        environment=Environment.PAPER,
        account="DU1",
        positions=(BrokerPosition("DU1", 7, "TEST", -14, 103.333333),),
        open_orders=(),
        observed_at=now,
    )
    mismatch = service.performance(run("one"), PerformancePeriod.TODAY)
    assert mismatch["unrealised_pnl"] is None
    assert mismatch["unrealised_status"] == "RECONCILIATION_REQUIRED"


def test_stale_broker_mark_is_not_used_for_unrealised_pnl(tmp_path: Path) -> None:
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    observed = datetime(2026, 9, 2, 15, tzinfo=UTC)
    item = plan("paper", "paper-plan", con_id=7, quantity=10, entry=100, created_at=observed)
    submit(ledger, item, "DU1", 10)
    fill(
        ledger,
        item,
        account="DU1",
        order_id=10,
        side=OrderAction.SELL,
        price=100,
        execution_id="entry",
        when=observed,
    )
    ledger.replace_broker_snapshot(
        environment=Environment.PAPER,
        account="DU1",
        positions=(BrokerPosition("DU1", 7, "TEST", -10, 100),),
        open_orders=(),
        observed_at=observed,
    )
    ledger.record_position_mark(
        environment=Environment.PAPER,
        account="DU1",
        con_id=7,
        mark=90,
        observed_at=observed,
    )
    result = RunPerformanceService(
        ledger,
        clock=lambda: observed.replace(minute=2),
    ).performance(run("paper"), PerformancePeriod.TODAY)
    assert result["unrealised_pnl"] is None
    assert result["unrealised_status"] == "MARK_UNAVAILABLE"


def test_periods_and_drawdown_use_only_selected_run(tmp_path: Path) -> None:
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    dates = [
        datetime(2026, 8, 28, 15, tzinfo=UTC),
        datetime(2026, 9, 1, 15, tzinfo=UTC),
        datetime(2026, 9, 2, 15, tzinfo=UTC),
    ]
    exits = [98, 104, 99]
    for index, (closed, exit_price) in enumerate(zip(dates, exits, strict=True), start=1):
        opened = closed.replace(hour=14)
        item = plan(
            "paper", f"plan-{index}", con_id=index, quantity=10, entry=100, created_at=opened
        )
        submit(ledger, item, "DU1", index * 10)
        fill(
            ledger,
            item,
            account="DU1",
            order_id=index * 10,
            side=OrderAction.SELL,
            price=100,
            execution_id=f"e{index}",
            when=opened,
        )
        fill(
            ledger,
            item,
            account="DU1",
            order_id=index * 10 + 2,
            side=OrderAction.BUY,
            price=exit_price,
            execution_id=f"x{index}",
            when=closed,
        )
    service = RunPerformanceService(ledger, clock=lambda: dates[-1])
    all_time = service.performance(run("paper"), PerformancePeriod.ALL)
    today = service.performance(run("paper"), PerformancePeriod.TODAY)
    assert all_time["closed_trades"] == 3
    assert all_time["max_realised_drawdown"] == 40
    assert today["closed_trades"] == 1
    assert today["realised_pnl"] == 10
