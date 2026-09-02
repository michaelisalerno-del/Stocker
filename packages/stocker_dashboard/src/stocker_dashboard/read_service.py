"""Calculation-free read models for the Stage 10 dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime, time
from typing import Any

from stocker_core.config import RunsConfig
from stocker_core.runs import Environment, RunConfig
from stocker_core.universes import UniverseDefinition
from stocker_execution.execution_ledger import ExecutionLedger, ExecutionRecord
from stocker_execution.execution_models import OrderLifecycle
from stocker_execution.pre_context import PriorSessionContextStore
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
        pre_context_store: PriorSessionContextStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.runtime_status = runtime_status
        self.stage5_store = stage5_store
        self.runtime_store = runtime_store
        self.ledger = ledger
        self.pre_context_store = pre_context_store
        self.clock = clock or (lambda: datetime.now(tz=UTC))

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
            _rows, count = self.stage5_store.list_snapshots(
                run_id=run.run_id,
                session=self.clock().date(),
                latest_checkpoint=True,
                limit=1,
            )
            result.append(
                {
                    "run_id": run.run_id,
                    "universe": run.universe,
                    "strategy": run.strategy,
                    "environment": run.environment.value,
                    "account": accounts.get(run.environment),
                    "enabled": run.enabled,
                    "status": runtime.state.value if runtime else "CONFIGURED",
                    "market": runtime.market.value if runtime and runtime.market else None,
                    "current_or_next_checkpoint": None,
                    "candidate_count": count,
                    "signals_today": runtime.signals_today if runtime else 0,
                    "open_positions": runtime.open_positions if runtime else 0,
                    "reason": runtime.reason if runtime else "",
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
        return {
            "run_id": run.run_id,
            "universe": run.universe,
            "strategy": run.strategy,
            "strategy_version": STRATEGY_VERSION if run.strategy == "SESSION_HARD" else None,
            "environment": run.environment.value,
            "account": account,
            "enabled": run.enabled,
            "status": runtime.state.value if runtime else "CONFIGURED",
            "risk_per_trade": run.risk.risk_per_trade if run.risk else None,
            "max_concurrent_positions": run.risk.max_concurrent_positions if run.risk else None,
            "last_checkpoint": latest_checkpoint.isoformat() if latest_checkpoint else None,
            "next_checkpoint": None,
            "funnel": [
                {"stage": "Universe", "count": len(self._universe(run.universe).members)},
                {
                    "stage": "Stage 5 ready",
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
        selected_run = run_id or self.config.runs[0].run_id
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
        context = (
            self.pre_context_store.get_by_identity(
                signal.underlying_con_id or 0,
                session=signal.session,
            )
            if self.pre_context_store is not None
            else None
        )
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
            "call_model_iv": context.call_model_iv if context else None,
            "put_model_iv": context.put_model_iv if context else None,
            "atm_iv": context.atm_iv if context else None,
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
        records_by_identity = {
            (item.environment, item.actual_account or item.expected_account, item.con_id): item
            for item in records
            if item.filled_quantity > item.closed_quantity
        }
        return [
            {
                "symbol": snapshot.symbol,
                "con_id": snapshot.con_id,
                "run_id": record.run_id if record else None,
                "strategy": record.strategy_id if record else None,
                "strategy_version": record.strategy_version if record else None,
                "signal_id": record.signal_id if record else None,
                "order_plan_id": record.order_plan_id if record else None,
                "environment": snapshot.environment.value,
                "account": snapshot.account,
                "side": "SHORT" if snapshot.quantity < 0 else "LONG",
                "quantity": abs(snapshot.quantity),
                "average_entry": snapshot.average_price,
                "current_price": None,
                "stop": record.stop_price if record else None,
                "target": record.target_price if record else None,
                "unrealised_pnl": None,
                "opened_at": record.opened_at.isoformat() if record and record.opened_at else None,
                "observed_at": snapshot.observed_at.isoformat(),
                "source": "IBKR",
                "orders": [
                    {
                        "role": order.role.value,
                        "status": order.status.value,
                        "ibkr_order_id": order.order_id,
                    }
                    for order in (
                        self.ledger.broker_orders(record.order_plan_id) if record else ()
                    )
                ],
            }
            for snapshot in self.ledger.broker_position_snapshots()
            for record in (
                records_by_identity.get(
                    (snapshot.environment, snapshot.account, snapshot.con_id)
                ),
            )
        ]

    def position_detail(
        self,
        environment: Environment,
        account: str,
        con_id: int,
    ) -> dict[str, Any]:
        position = next(
            (
                item
                for item in self.positions()
                if item["environment"] == environment.value
                and item["account"] == account
                and item["con_id"] == con_id
            ),
            None,
        )
        if position is None:
            raise ValueError(
                f"Unknown broker position: {environment.value}/{account}/{con_id}"
            )
        return position

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
        return {
            "items": [self._trade(item) for item in records],
            "total": total,
            "summary": {
                "trades": summary.trades,
                "wins": summary.wins,
                "losses": summary.losses,
                "win_percent": (
                    summary.wins / summary.trades * 100 if summary.trades else None
                ),
                "total_pnl": summary.total_pnl,
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
                    "strategy": "SESSION_HARD",
                    "identity": "SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
                    "version": STRATEGY_VERSION,
                    "editable_parameters": [],
                    "description": "Frozen Session HARD / Structure D strategy definition",
                }
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
