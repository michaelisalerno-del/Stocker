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
from stocker_execution.runtime import RuntimeStatus, RuntimeStore
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
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.runtime_status = runtime_status
        self.stage5_store = stage5_store
        self.runtime_store = runtime_store
        self.ledger = ledger
        self.clock = clock or (lambda: datetime.now(tz=UTC))

    def overview(self) -> dict[str, Any]:
        status = self.runtime_status()
        positions = self.positions()
        counters = self.runtime_store.counters(session=self.clock().date())
        attention = []
        environments = {}
        for item in status.execution_environments:
            environments[item.environment.value] = {
                "connected": item.connected,
                "account": item.account,
                "expected_account": item.expected_account,
                "reconciled": item.reconciled,
                "ready": item.ready,
            }
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
                "strategy_qualified": counters.signals,
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
        rows, _ = self.stage5_store.list_snapshots(
            run_id=run_id, session=selected_session, latest_checkpoint=True, limit=500
        )
        signals = tuple(
            item
            for item in self.runtime_store.load_signals(run_id)
            if item.session == selected_session
        )
        plans, _ = self.ledger.list_records(run_id=run_id, limit=500)
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
            "last_checkpoint": max((item.t0.isoformat() for item in rows), default=None),
            "next_checkpoint": None,
            "funnel": [
                {"stage": "Universe", "count": len(self._universe(run.universe).members)},
                {
                    "stage": "Stage 5 ready",
                    "count": sum(item.status is Stage5Status.READY for item in rows),
                },
                {"stage": "Strategy evaluated", "count": len(signals)},
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
                    "count": sum(item.filled_quantity > item.closed_quantity for item in plans),
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
        rows, total = self.stage5_store.list_snapshots(
            run_id=selected_run,
            session=session or self.clock().date(),
            latest_checkpoint=checkpoint is None,
            limit=limit,
            offset=offset,
        )
        if checkpoint is not None:
            rows = tuple(item for item in rows if item.t0 == checkpoint)
            total = len(rows)
        signals = {
            (item.underlying_con_id, item.session, item.t0): item
            for item in self.runtime_store.load_signals(selected_run)
        }
        items = [
            self._candidate(row, signals.get((row.con_id, row.session, row.t0))) for row in rows
        ]
        if status is not None:
            items = [item for item in items if item["status"] == status]
            total = len(items)
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

    def positions(self) -> list[dict[str, Any]]:
        status = self.runtime_status()
        reconciled = {item.environment: item.reconciled for item in status.execution_environments}
        records, _ = self.ledger.list_records(limit=500)
        return [
            {
                "symbol": item.symbol,
                "run_id": item.run_id,
                "strategy": item.strategy_id,
                "strategy_version": item.strategy_version,
                "signal_id": item.signal_id,
                "order_plan_id": item.order_plan_id,
                "environment": item.environment.value,
                "account": item.actual_account or item.expected_account,
                "side": "SHORT" if item.side.value == "SELL" else "LONG",
                "quantity": item.filled_quantity - item.closed_quantity,
                "average_entry": item.average_fill_price,
                "current_price": None,
                "stop": item.stop_price,
                "target": item.target_price,
                "unrealised_pnl": None,
                "opened_at": item.opened_at.isoformat() if item.opened_at else None,
                "source": "IBKR_RECONCILED" if reconciled.get(item.environment) else "UNRECONCILED",
                "orders": [
                    {
                        "role": order.role.value,
                        "status": order.status.value,
                        "ibkr_order_id": order.order_id,
                    }
                    for order in self.ledger.broker_orders(item.order_plan_id)
                ],
            }
            for item in records
            if item.filled_quantity > item.closed_quantity
        ]

    def trades(
        self,
        *,
        environment: Environment | None,
        start: datetime | None,
        end: datetime | None,
        limit: int = 100,
        offset: int = 0,
        run_id: str | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        records, total = self.ledger.list_records(
            run_id=run_id,
            environment=environment,
            symbol=symbol,
            closed_only=True,
            start=start,
            end=end,
            limit=limit,
            offset=offset,
        )
        pnls = [item.realized_pnl for item in records if item.realized_pnl is not None]
        wins = sum(value > 0 for value in pnls)
        losses = sum(value < 0 for value in pnls)
        return {
            "items": [self._trade(item) for item in records],
            "total": total,
            "summary": {
                "trades": len(records),
                "wins": wins,
                "losses": losses,
                "win_percent": wins / len(pnls) * 100 if pnls else None,
                "total_pnl": sum(pnls),
                "total_r": None,
                "mean_r": None,
            },
        }

    def system(self) -> dict[str, Any]:
        status = self.runtime_status()
        records, _ = self.ledger.list_records(limit=500)
        environments = []
        problems = []
        for item in status.execution_environments:
            scoped = [record for record in records if record.environment is item.environment]
            environments.append(
                {
                    "environment": item.environment.value,
                    "connected": item.connected,
                    "account": item.account,
                    "expected_account": item.expected_account,
                    "reconciled": item.reconciled,
                    "ready": item.ready,
                    "open_orders": sum(
                        record.status
                        not in {
                            OrderLifecycle.CANCELLED,
                            OrderLifecycle.REJECTED,
                            OrderLifecycle.CLOSED,
                        }
                        for record in scoped
                    ),
                    "positions": sum(
                        record.filled_quantity > record.closed_quantity for record in scoped
                    ),
                }
            )
            if not item.connected:
                problems.append(f"IBKR {item.environment.value} disconnected")
            elif not item.reconciled:
                problems.append(f"IBKR {item.environment.value} reconciliation required")
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

    @staticmethod
    def _trade(record: ExecutionRecord) -> dict[str, Any]:
        return {
            "date_time": record.closed_at.isoformat() if record.closed_at else None,
            "run_id": record.run_id,
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
