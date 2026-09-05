"""Calculation-free read models for the Stage 10 dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime, time
from typing import Any, cast
from zoneinfo import ZoneInfo

from stocker_core.config import RunsConfig
from stocker_core.markets import get_market
from stocker_core.runs import Environment, RunConfig
from stocker_core.strategies import installed_strategies
from stocker_core.universes import UniverseDefinition
from stocker_dashboard.performance import PerformancePeriod, RunPerformanceService
from stocker_execution.activity_shortlist import ActivityShortlistStore
from stocker_execution.execution_ledger import ExecutionLedger, ExecutionRecord
from stocker_execution.execution_models import OrderLifecycle
from stocker_execution.runtime import ExecutionEnvironmentStatus, RuntimeStatus, RuntimeStore
from stocker_execution.session_hard_structure_d import (
    STRATEGY_VERSION,
    SignalStatus,
    StrategySignal,
)
from stocker_execution.stage5 import Stage5FeatureSnapshot, Stage5SnapshotStore, Stage5Status


class DashboardReadService:
    """Copy Stage 5--9 outputs into small operational responses."""

    def __init__(
        self,
        *,
        config: RunsConfig,
        runtime_status: Callable[[], RuntimeStatus],
        stage5_store: Stage5SnapshotStore,
        runtime_store: RuntimeStore,
        ledger: ExecutionLedger,
        activity_store: ActivityShortlistStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.runtime_status = runtime_status
        self.stage5_store = stage5_store
        self.runtime_store = runtime_store
        self.ledger = ledger
        self.clock = clock or (lambda: datetime.now(tz=UTC))
        self.activity_store = activity_store
        self.performance_service = RunPerformanceService(ledger, clock=self.clock)

    def overview(self) -> dict[str, Any]:
        status = self.runtime_status()
        positions = self.positions()
        counters = self.runtime_store.counters(session=self.clock().date())
        attention = []
        environments = {}
        for item in status.execution_environments:
            environments[item.environment.value] = self._environment_status(item)
            if not item.connected:
                attention.append({"scope": item.environment.value, "message": "IBKR disconnected"})
            elif not item.reconciled:
                attention.append(
                    {"scope": item.environment.value, "message": "reconciliation required"}
                )
        attention.extend(
            {"scope": run.run_id, "message": run.reason}
            for run in status.runs
            if run.state.value == "DEGRADED" and run.reason
        )
        return {
            "as_of": self.clock().isoformat(),
            "system": status.application.value,
            "environments": environments,
            "active_runs": sum(run.state.value == "ACTIVE" for run in status.runs),
            "open_positions": len(positions),
            "runs": self.runs(),
            "positions": positions,
            "today": {
                "instruments_evaluated": counters.instruments_ready,
                "signals": counters.signals,
                "orders": counters.orders,
                "filled": counters.fills,
                "risk_rejected": counters.risk_rejects,
            },
            "attention": attention[:8],
        }

    def runs(self) -> list[dict[str, Any]]:
        status = self.runtime_status()
        runtime_by_id = {item.run_id: item for item in status.runs}
        accounts = {
            item.environment: item.account or item.expected_account
            for item in status.execution_environments
        }
        result = []
        for run in self.config.runs:
            runtime = runtime_by_id.get(run.run_id)
            market_definition = get_market(run.market_id) if run.market_id is not None else None
            selected_session = (
                runtime.session
                if runtime is not None and runtime.session is not None
                else self.clock()
                .astimezone(ZoneInfo(market_definition.timezone if market_definition else "UTC"))
                .date()
            )
            count = 0
            screen = None
            if run.enabled:
                _rows, count = self.stage5_store.list_snapshots(
                    run_id=run.run_id,
                    session=selected_session,
                    latest_checkpoint=True,
                    limit=1,
                )
                screen = (
                    self.activity_store.get(
                        market_definition.market_id.value,
                        run.cap_bucket,
                        selected_session,
                    )
                    if self.activity_store is not None
                    and market_definition is not None
                    and run.cap_bucket is not None
                    else None
                )
            today = self.performance_service.performance(run, PerformancePeriod.TODAY)
            recent = self.performance_service.performance(run, PerformancePeriod.SESSIONS_20)
            result.append(
                {
                    "run_id": run.run_id,
                    "display_name": run.display_name or run.run_id,
                    "universe": run.universe,
                    "strategy": run.strategy,
                    "strategy_id": run.effective_strategy_id,
                    "strategy_version": run.strategy_version,
                    "market_id": run.market_id.value if run.market_id else None,
                    "cap_bucket": run.cap_bucket.value if run.cap_bucket else None,
                    "environment": run.environment.value,
                    "account": accounts.get(run.environment),
                    "enabled": run.enabled,
                    "status": runtime.state.value if runtime else "CONFIGURED",
                    "market": runtime.market.value if runtime and runtime.market else None,
                    "current_or_next_checkpoint": None,
                    "candidate_count": (
                        sum(item.selected for item in screen.candidates)
                        if screen is not None
                        else count
                    ),
                    "signals_today": runtime.signals_today if runtime else 0,
                    "open_positions": runtime.open_positions if runtime else 0,
                    "reason": runtime.reason if runtime else "",
                    "currency": today["currency"] or self._run_currency(run),
                    "today_realised_pnl": today["realised_pnl"],
                    "unrealised_pnl": today["unrealised_pnl"],
                    "unrealised_status": today["unrealised_status"],
                    "total_r": recent["total_r"],
                }
            )
        return result

    def run_detail(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        status = self.runtime_status()
        runtime = next((item for item in status.runs if item.run_id == run_id), None)
        selected_session = runtime.session if runtime and runtime.session else self.clock().date()
        latest_rows, latest_total = self.stage5_store.list_snapshots(
            run_id=run_id,
            session=selected_session,
            latest_checkpoint=True,
            limit=1,
        )
        latest_checkpoint = latest_rows[0].t0 if latest_rows else None
        ready_count = 0
        if latest_checkpoint is not None:
            _ready_rows, ready_count = self.stage5_store.list_snapshots(
                run_id=run_id,
                session=selected_session,
                checkpoint=latest_checkpoint,
                status=Stage5Status.READY,
                limit=1,
            )
        signals = tuple(
            item
            for item in self.runtime_store.load_signals(run_id)
            if item.session == selected_session and item.t0 == latest_checkpoint
        )
        signal_ids = {item.signal_id for item in signals}
        plans = tuple(
            item
            for item in self.ledger.list_records(run_id=run_id, limit=500)[0]
            if item.signal_id in signal_ids
        )
        positions = self.positions()
        account = self._account(run.environment)
        market = get_market(run.market_id) if run.market_id is not None else None
        screen = None
        if self.activity_store is not None and market is not None and run.cap_bucket is not None:
            screen = self.activity_store.get(
                market.market_id.value, run.cap_bucket, selected_session
            )
        today = self.performance_service.performance(run, PerformancePeriod.TODAY)
        return {
            "run_id": run.run_id,
            "display_name": run.display_name or run.run_id,
            "universe": run.universe,
            "strategy": run.strategy,
            "strategy_id": run.effective_strategy_id,
            "strategy_version": run.strategy_version
            or (STRATEGY_VERSION if run.strategy == "SESSION_HARD_HV" else None),
            "market_id": run.market_id.value if run.market_id else None,
            "market": market.display_name if market else None,
            "cap_bucket": run.cap_bucket.value if run.cap_bucket else None,
            "cap_bucket_version": run.cap_bucket_version,
            "candidate_screen_id": run.effective_candidate_screen_id,
            "candidate_screen_version": run.candidate_screen_version,
            "environment": run.environment.value,
            "account": account,
            "currency": market.currency if market else self._run_currency(run),
            "enabled": run.enabled,
            "status": runtime.state.value if runtime else "CONFIGURED",
            "risk_per_trade": run.risk.risk_per_trade if run.risk else None,
            "max_concurrent_positions": run.risk.max_concurrent_positions if run.risk else None,
            "last_checkpoint": latest_checkpoint.isoformat() if latest_checkpoint else None,
            "next_checkpoint": None,
            "market_state": runtime.market.value if runtime and runtime.market else None,
            "session": selected_session.isoformat(),
            "screen_state": screen.status.value if screen else None,
            "screen_timestamp": screen.screen_timestamp.isoformat() if screen else None,
            "watchlist_size": sum(item.selected for item in screen.candidates) if screen else 0,
            "today_realised_pnl": today["realised_pnl"],
            "current_unrealised_pnl": today["unrealised_pnl"],
            "unrealised_status": today["unrealised_status"],
            "funnel": [
                {
                    "stage": "Market / cap eligible",
                    "count": len(self._universe(run.universe).members),
                },
                {
                    "stage": "Activity scan union",
                    "count": len(screen.candidates) if screen else 0,
                },
                {
                    "stage": "Shortlist selected",
                    "count": sum(item.selected for item in screen.candidates) if screen else 0,
                },
                {"stage": "Stage 2 qualified", "count": latest_total},
                {
                    "stage": "Stage 4 / Stage 5 PRE ready",
                    "count": ready_count,
                },
                {"stage": "Strategy evaluated", "count": min(len(signals), latest_total)},
                {
                    "stage": "Strategy qualified",
                    "count": sum(item.status is not SignalStatus.NOT_QUALIFIED for item in signals),
                },
                {"stage": "Rank selected", "count": sum(item.selected for item in signals)},
                {
                    "stage": "Entry triggered",
                    "count": sum(item.status is SignalStatus.ENTRY_TRIGGERED for item in signals),
                },
                {"stage": "Orders", "count": len(plans)},
                {
                    "stage": "Positions",
                    "count": sum(
                        item["run_id"] == run_id and item["signal_id"] in signal_ids
                        for item in positions
                    ),
                },
            ],
        }

    def run_performance(self, run_id: str, period: PerformancePeriod | str) -> dict[str, Any]:
        return self.performance_service.performance(self._run(run_id), period)

    def universe_runs(self) -> dict[str, list[dict[str, Any]]]:
        rows = self.runs()
        return {
            environment.value: [item for item in rows if item["environment"] == environment.value]
            for environment in Environment
        }

    def screen(self, market_id: str, cap_bucket: str, session: date) -> dict[str, Any]:
        if self.activity_store is None:
            raise ValueError("activity shortlist store is unavailable")
        from stocker_core.markets import CapBucket

        snapshot = self.activity_store.get(market_id, CapBucket(cap_bucket), session)
        if snapshot is None:
            raise ValueError("unknown activity shortlist snapshot")
        return {
            "market_id": snapshot.market_id,
            "cap_bucket": snapshot.cap_bucket.value,
            "cap_bucket_version": snapshot.cap_bucket_version,
            "session": snapshot.session.isoformat(),
            "screen_timestamp": snapshot.screen_timestamp.isoformat(),
            "profile_id": snapshot.profile_id,
            "profile_version": snapshot.profile_version,
            "status": snapshot.status.value,
            "components": [item.value for item in snapshot.components],
            "reason": snapshot.reason,
            "candidates": [asdict(item) for item in snapshot.candidates],
        }

    def candidates(
        self,
        *,
        run_id: str | None = None,
        session: date | None = None,
        checkpoint: datetime | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 500:
            raise ValueError("candidate limit must be between 1 and 500")
        enabled_runs = tuple(run for run in self.config.runs if run.enabled)
        if run_id is None:
            if not enabled_runs:
                return {"items": [], "total": 0, "limit": limit, "offset": offset}
            selected_run = enabled_runs[0].run_id
        else:
            selected = self._run(run_id)
            if not selected.enabled:
                return {"items": [], "total": 0, "limit": limit, "offset": offset}
            selected_run = selected.run_id
        selected_session = session or self.clock().date()
        signal_status = None
        stage5_status = None
        if status is not None:
            try:
                signal_status = SignalStatus(status)
            except ValueError:
                try:
                    stage5_status = Stage5Status(status)
                except ValueError as exc:
                    raise ValueError(f"Unknown candidate status: {status}") from exc
        selected_checkpoint = checkpoint
        if selected_checkpoint is None:
            latest_rows, _ = self.stage5_store.list_snapshots(
                run_id=selected_run,
                session=selected_session,
                latest_checkpoint=True,
                limit=1,
            )
            selected_checkpoint = latest_rows[0].t0 if latest_rows else None
        if signal_status is not None:
            matching_signals = sorted(
                (
                    item
                    for item in self.runtime_store.load_signals(selected_run)
                    if item.session == selected_session
                    and item.t0 == selected_checkpoint
                    and item.status is signal_status
                ),
                key=lambda item: (
                    item.candidate_rank if item.candidate_rank is not None else 10**9,
                    item.symbol,
                    item.signal_id,
                ),
            )
            page = matching_signals[offset : offset + limit]
            items = [
                self._candidate(
                    self.stage5_store.get(
                        item.universe_id,
                        item.t0,
                        item.underlying_con_id or 0,
                        calculation_version=item.feature_calculation_version,
                    ),
                    item,
                )
                for item in page
            ]
            return {
                "items": items,
                "total": len(matching_signals),
                "limit": limit,
                "offset": offset,
            }
        rows, total = self.stage5_store.list_snapshots(
            run_id=selected_run,
            session=selected_session,
            checkpoint=selected_checkpoint,
            latest_checkpoint=False,
            status=stage5_status,
            limit=limit,
            offset=offset,
        )
        signals = {
            (item.underlying_con_id, item.session, item.t0): item
            for item in self.runtime_store.load_signals(selected_run)
        }
        items = [
            self._candidate(row, signals.get((row.con_id, row.session, row.t0))) for row in rows
        ]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def candidate_detail(self, signal_id: str) -> dict[str, Any]:
        signal = self._signal(signal_id)
        snapshot = self.stage5_store.get(
            signal.universe_id,
            signal.t0,
            signal.underlying_con_id or 0,
            calculation_version=signal.feature_calculation_version,
        )
        plan = next(
            (
                item
                for item in self.ledger.list_records(run_id=signal.run_id, limit=500)[0]
                if item.signal_id == signal_id
            ),
            None,
        )
        run = self._run(signal.run_id)
        return {
            **self._candidate(snapshot, signal),
            "con_id": signal.underlying_con_id,
            "run_id": signal.run_id,
            "universe": signal.universe_id,
            "strategy": signal.strategy_id,
            "strategy_version": signal.strategy_version,
            "environment": run.environment.value,
            "account": self._account(run.environment),
            "t0": signal.t0.isoformat(),
            "p0": signal.p0,
            "expected_move_source": snapshot.expected_move_source if snapshot else None,
            "expected_move_observation_at": (
                snapshot.expected_move_observation_at.isoformat()
                if snapshot and snapshot.expected_move_observation_at
                else None
            ),
            "expected_move_calculation_version": (
                snapshot.expected_move_calculation_version if snapshot else None
            ),
            "raw_historical_volatility": (snapshot.raw_historical_volatility if snapshot else None),
            "historical_volatility": snapshot.historical_volatility if snapshot else None,
            "market_regular_minutes": snapshot.market_regular_minutes if snapshot else None,
            "expected_absolute_return_15m": (
                snapshot.expected_absolute_return_15m if snapshot else None
            ),
            "m_price": signal.m_price,
            "raw_pre_move": snapshot.raw_pre_move_price if snapshot else None,
            "entry_reference": signal.entry_reference,
            "signal_id": signal.signal_id,
            "order_plan_id": plan.order_plan_id if plan else None,
        }

    def orders(self, *, scope: str = "open", limit: int = 100, offset: int = 0) -> dict[str, Any]:
        statuses = None
        start = None
        active = {
            OrderLifecycle.PLANNED,
            OrderLifecycle.SUBMITTING,
            OrderLifecycle.SUBMITTED,
            OrderLifecycle.PARTIALLY_FILLED,
            OrderLifecycle.FILLED,
        }
        if scope == "open":
            statuses = frozenset(active)
        elif scope == "rejected":
            statuses = frozenset({OrderLifecycle.REJECTED})
        elif scope == "today":
            start = datetime.combine(self.clock().date(), time.min, tzinfo=UTC)
        elif scope != "all":
            raise ValueError("order scope must be open, today, rejected, or all")
        records, total = self.ledger.list_records(
            statuses=statuses, start=start, limit=limit, offset=offset
        )
        return {"items": [self._order(item) for item in records], "total": total}

    def order_detail(self, order_plan_id: str) -> dict[str, Any]:
        record = self.ledger.get(order_plan_id)
        if record is None or record.diagnostic:
            raise ValueError(f"Unknown order plan: {order_plan_id}")
        return self._order(record)

    def positions(self) -> list[dict[str, Any]]:
        records, _ = self.ledger.list_records(limit=500)
        by_identity: dict[tuple[Environment, str, int], list[ExecutionRecord]] = {}
        for record in records:
            if record.filled_quantity <= record.closed_quantity:
                continue
            account = record.actual_account or record.expected_account
            by_identity.setdefault((record.environment, account, record.con_id), []).append(record)

        rows: list[dict[str, Any]] = []
        for snapshot in self.ledger.broker_position_snapshots():
            identity = (snapshot.environment, snapshot.account, snapshot.con_id)
            attributed = by_identity.get(identity, [])
            signed_quantity = sum(
                (-1 if item.side.value == "SELL" else 1)
                * (item.filled_quantity - item.closed_quantity)
                for item in attributed
            )
            if abs(signed_quantity - snapshot.quantity) > 1e-9:
                rows.append(
                    self._position_row(
                        snapshot,
                        (),
                        run_id=None,
                        quantity=abs(snapshot.quantity),
                        average_entry=None,
                        attribution_status="RECONCILIATION_REQUIRED",
                    )
                )
                continue
            records_by_run: dict[str, list[ExecutionRecord]] = {}
            for record in attributed:
                records_by_run.setdefault(record.run_id, []).append(record)
            for run_id, run_records in sorted(records_by_run.items()):
                quantities = [item.filled_quantity - item.closed_quantity for item in run_records]
                quantity = sum(quantities)
                prices = [item.average_fill_price for item in run_records]
                average_entry = (
                    sum(
                        cast(float, price) * lot
                        for price, lot in zip(prices, quantities, strict=True)
                    )
                    / quantity
                    if quantity > 0 and all(price is not None for price in prices)
                    else None
                )
                rows.append(
                    self._position_row(
                        snapshot,
                        tuple(run_records),
                        run_id=run_id,
                        quantity=quantity,
                        average_entry=average_entry,
                        attribution_status="MARK_UNAVAILABLE",
                    )
                )
        return rows

    def _position_row(
        self,
        snapshot: Any,
        records: tuple[ExecutionRecord, ...],
        *,
        run_id: str | None,
        quantity: float,
        average_entry: float | None,
        attribution_status: str,
    ) -> dict[str, Any]:
        record = records[0] if len(records) == 1 else None
        orders = [
            order for item in records for order in self.ledger.broker_orders(item.order_plan_id)
        ]
        return {
            "symbol": snapshot.symbol,
            "con_id": snapshot.con_id,
            "run_id": run_id,
            "strategy": record.strategy_id if record else None,
            "strategy_version": record.strategy_version if record else None,
            "signal_id": record.signal_id if record else None,
            "order_plan_id": record.order_plan_id if record else None,
            "environment": snapshot.environment.value,
            "account": snapshot.account,
            "side": "SHORT" if snapshot.quantity < 0 else "LONG",
            "quantity": quantity,
            "average_entry": average_entry,
            "current_price": None,
            "stop": record.stop_price if record else None,
            "target": record.target_price if record else None,
            "unrealised_pnl": None,
            "attribution_status": attribution_status,
            "opened_at": record.opened_at.isoformat() if record and record.opened_at else None,
            "observed_at": snapshot.observed_at.isoformat(),
            "source": "IBKR",
            "orders": [
                {
                    "role": order.role.value,
                    "status": order.status.value,
                    "ibkr_order_id": order.order_id,
                }
                for order in orders
            ],
        }

    def position_detail(
        self,
        environment: Environment,
        account: str,
        con_id: int,
    ) -> dict[str, Any]:
        positions = [
            item
            for item in self.positions()
            if item["environment"] == environment.value
            and item["account"] == account
            and item["con_id"] == con_id
        ]
        if not positions:
            raise ValueError(f"Unknown broker position: {environment.value}/{account}/{con_id}")
        if len(positions) == 1:
            return positions[0]
        return {
            **positions[0],
            "run_id": None,
            "strategy": None,
            "signal_id": None,
            "order_plan_id": None,
            "attributions": positions,
        }

    def trades(
        self,
        *,
        environment: Environment | None,
        start: datetime | None,
        end: datetime | None,
        limit: int = 100,
        offset: int = 0,
        run_id: str | None = None,
        strategy: str | None = None,
        universe: str | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        selected_run_ids = frozenset(
            run.run_id
            for run in self.config.runs
            if (run_id is None or run.run_id == run_id)
            and (strategy is None or run.strategy == strategy)
            and (universe is None or run.universe == universe)
        )
        records, total = self.ledger.list_records(
            run_ids=selected_run_ids,
            environment=environment,
            symbol=symbol,
            closed_only=True,
            start=start,
            end=end,
            limit=limit,
            offset=offset,
        )
        summary = self.ledger.closed_trade_summary(
            run_ids=selected_run_ids,
            environment=environment,
            symbol=symbol,
            start=start,
            end=end,
        )
        selected_runs = tuple(run for run in self.config.runs if run.run_id in selected_run_ids)
        run_currency_sets = tuple(self._run_currencies(run) for run in selected_runs)
        currencies = set().union(*run_currency_sets) if run_currency_sets else set()
        unresolved_currency = any(not values for values in run_currency_sets)
        mixed_currency = len(currencies) > 1
        pnl_available = not mixed_currency and not unresolved_currency
        common_currency = next(iter(currencies)) if len(currencies) == 1 else None
        return {
            "items": [self._trade(item) for item in records],
            "total": total,
            "summary": {
                "trades": summary.trades,
                "wins": summary.wins,
                "losses": summary.losses,
                "win_percent": (summary.wins / summary.trades * 100 if summary.trades else None),
                "total_pnl": summary.total_pnl if pnl_available else None,
                "currency": common_currency,
                "pnl_status": (
                    "MULTIPLE_CURRENCIES"
                    if mixed_currency
                    else "CURRENCY_UNAVAILABLE"
                    if unresolved_currency
                    else "AVAILABLE"
                ),
                "total_r": None,
                "mean_r": None,
            },
        }

    def system(self) -> dict[str, Any]:
        status = self.runtime_status()
        broker_positions = self.ledger.broker_position_snapshots()
        broker_orders = self.ledger.broker_open_order_snapshots()
        environments = []
        problems = []
        for item in status.execution_environments:
            environments.append(
                {
                    **self._environment_status(item),
                    "environment": item.environment.value,
                    "open_orders": sum(
                        order.environment is item.environment for order in broker_orders
                    ),
                    "positions": sum(
                        position.environment is item.environment for position in broker_positions
                    ),
                }
            )
            if not item.connected:
                problems.append(f"IBKR {item.environment.value} disconnected")
            elif not item.reconciled:
                problems.append(f"IBKR {item.environment.value} reconciliation required")
        records, _ = self.ledger.list_records(limit=20)
        events = []
        for record in records[:20]:
            timestamp = record.closed_at or record.opened_at or record.submitted_at
            if timestamp:
                events.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "message": (
                            f"{record.symbol} {record.environment.value} "
                            f"{record.status.value.lower()}"
                        ),
                    }
                )
        problems.extend(
            f"{run.run_id}: {run.reason}"
            for run in status.runs
            if run.state.value == "DEGRADED" and run.reason
        )
        return {
            "application": status.application.value,
            "environments": environments,
            "ibkr_api_resources": (
                asdict(status.ibkr_resources) if status.ibkr_resources is not None else None
            ),
            "runtime": {
                "active_runs": sum(run.state.value == "ACTIVE" for run in status.runs),
                "counters": asdict(status.counters),
            },
            "problems": problems,
            "events": events,
        }

    @staticmethod
    def _environment_status(item: ExecutionEnvironmentStatus) -> dict[str, object]:
        return {
            "connected": item.connected,
            "account": item.account,
            "expected_account": item.expected_account,
            "reconciled": item.reconciled,
            "ready": item.ready,
            "equity": item.equity,
            "buying_power": item.buying_power,
        }

    def settings(self) -> dict[str, Any]:
        status = self.runtime_status()
        return {
            "broker": [
                {
                    "environment": item.environment.value,
                    "account": item.expected_account,
                    "connected": item.connected,
                }
                for item in status.execution_environments
            ],
            "universes": [
                {
                    "universe_id": item.universe_id,
                    "name": item.name,
                    "members": len(item.members),
                }
                for item in self.config.universes
            ],
            "runs": [item.model_dump(mode="json") for item in self.config.runs],
            "strategies": [
                {
                    "strategy": item.config_name,
                    "identity": item.strategy_id,
                    "version": item.strategy_version,
                    "environments": list(item.environments),
                    "editable_parameters": [],
                    "description": f"Frozen {item.label} / Structure D strategy definition",
                }
                for item in installed_strategies()
            ],
        }

    @staticmethod
    def _candidate(
        snapshot: Stage5FeatureSnapshot | None, signal: StrategySignal | None
    ) -> dict[str, Any]:
        if signal:
            return {
                "signal_id": signal.signal_id,
                "rank": signal.candidate_rank,
                "symbol": signal.symbol,
                "pre_move_m": signal.pre_move_m,
                "cohort_percentile": signal.cohort_percentile,
                "band": signal.band.value if signal.band else None,
                "session_hard": signal.session_hard_qualified,
                "session_hard_score": signal.session_hard_score,
                "structure": "D" if signal.direction else None,
                "direction": signal.direction,
                "entry": signal.entry_level,
                "status": signal.status.value,
            }
        if snapshot is None:
            raise ValueError("candidate is unavailable")
        return {
            "signal_id": None,
            "rank": None,
            "symbol": snapshot.symbol,
            "con_id": snapshot.con_id,
            "pre_move_m": snapshot.pre_move_m,
            "cohort_percentile": None,
            "band": None,
            "session_hard": None,
            "session_hard_score": None,
            "structure": None,
            "direction": None,
            "entry": None,
            "status": snapshot.status.value,
        }

    def _order(self, record: ExecutionRecord) -> dict[str, Any]:
        return {
            "time": record.submitted_at.isoformat() if record.submitted_at else None,
            "order_plan_id": record.order_plan_id,
            "signal_id": record.signal_id,
            "run_id": record.run_id,
            "universe": self._run(record.run_id).universe,
            "strategy": record.strategy_id,
            "strategy_version": record.strategy_version,
            "environment": record.environment.value,
            "account": record.actual_account or record.expected_account,
            "con_id": record.con_id,
            "symbol": record.symbol,
            "side": record.side.value,
            "quantity": record.intended_quantity,
            "order_type": "MARKET + PROTECTION",
            "entry": record.entry_reference,
            "stop": record.stop_price,
            "target": record.target_price,
            "status": record.status.value,
            "filled": record.filled_quantity,
            "average_fill": record.average_fill_price,
            "rejection_reason": record.rejection_reason,
            "orders": [
                {
                    "role": order.role.value,
                    "status": order.status.value,
                    "ibkr_order_id": order.order_id,
                }
                for order in self.ledger.broker_orders(record.order_plan_id)
            ],
        }

    def _trade(self, record: ExecutionRecord) -> dict[str, Any]:
        run = self._run(record.run_id)
        currency = get_market(run.market_id).currency if run.market_id is not None else None
        return {
            "date_time": record.closed_at.isoformat() if record.closed_at else None,
            "run_id": record.run_id,
            "universe": self._run(record.run_id).universe,
            "strategy": record.strategy_id,
            "strategy_version": record.strategy_version,
            "environment": record.environment.value,
            "account": record.actual_account or record.expected_account,
            "symbol": record.symbol,
            "side": "SHORT" if record.side.value == "SELL" else "LONG",
            "entry": record.average_fill_price,
            "exit": record.average_exit_price,
            "quantity": record.closed_quantity,
            "pnl": record.realized_pnl,
            "currency": currency,
            "r": None,
            "exit_reason": None,
        }

    def _run(self, run_id: str) -> RunConfig:
        run = next((item for item in self.config.runs if item.run_id == run_id), None)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        return run

    def _universe(self, universe_id: str) -> UniverseDefinition:
        universe = next(
            (item for item in self.config.universes if item.universe_id == universe_id), None
        )
        if universe is None:
            raise ValueError(f"Unknown universe: {universe_id}")
        return universe

    def _run_currency(self, run: RunConfig) -> str | None:
        if run.market_id is not None:
            return get_market(run.market_id).currency
        currencies = {item.currency for item in self._universe(run.universe).members}
        return next(iter(currencies)) if len(currencies) == 1 else None

    def _run_currencies(self, run: RunConfig) -> set[str]:
        if run.market_id is not None:
            return {get_market(run.market_id).currency}
        return {item.currency for item in self._universe(run.universe).members}

    def _account(self, environment: Environment) -> str | None:
        return next(
            (
                item.account or item.expected_account
                for item in self.runtime_status().execution_environments
                if item.environment is environment
            ),
            None,
        )

    def _signal(self, signal_id: str) -> StrategySignal:
        for run in self.config.runs:
            for signal in self.runtime_store.load_signals(run.run_id):
                if signal.signal_id == signal_id:
                    return signal
        raise ValueError(f"Unknown candidate signal: {signal_id}")
