"""Ledger-backed per-run performance and conservative open-lot attribution."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

from stocker_core.markets import get_market
from stocker_core.runs import Environment, RunConfig
from stocker_data.calendars import get_market_calendar
from stocker_execution.execution_ledger import ExecutionLedger, ExecutionRecord
from stocker_execution.execution_models import OrderAction, OrderLifecycle


class PerformancePeriod(StrEnum):
    TODAY = "TODAY"
    SESSIONS_5 = "5_SESSIONS"
    SESSIONS_20 = "20_SESSIONS"
    ALL = "ALL"


POSITION_MARK_MAX_AGE = timedelta(minutes=1)


class RunPerformanceService:
    def __init__(
        self,
        ledger: ExecutionLedger,
        *,
        clock: Callable[[], datetime] | None = None,
        marks: Mapping[tuple[Environment, str, int], float] | None = None,
    ) -> None:
        self.ledger = ledger
        self.clock = clock or (lambda: datetime.now(tz=UTC))
        self.marks = marks

    def performance(self, run: RunConfig, period: PerformancePeriod | str) -> dict[str, Any]:
        selected_period = PerformancePeriod(period)
        market = get_market(run.market_id) if run.market_id is not None else None
        timezone = ZoneInfo(market.timezone if market is not None else "UTC")
        now = _aware(self.clock()).astimezone(timezone)
        records = self._records(run_id=run.run_id)
        closed = tuple(
            item
            for item in records
            if item.status is OrderLifecycle.CLOSED
            and item.closed_at is not None
            and self._in_period(
                item.closed_at,
                selected_period,
                now.date(),
                market.calendar if market else None,
                timezone,
            )
        )
        ordered = tuple(sorted(closed, key=lambda item: (item.closed_at, item.order_plan_id)))
        pnl_values = tuple(float(item.realized_pnl or 0.0) for item in ordered)
        r_values = tuple(self._realised_r(item) for item in ordered)
        r_available = all(value is not None for value in r_values)
        total_r = sum(value for value in r_values if value is not None) if r_available else None
        history = self._history(ordered, timezone)
        unrealised_pnl, unrealised_status, open_positions = self._unrealised(run, now)
        return {
            "period": selected_period.value,
            "currency": market.currency if market is not None else None,
            "closed_trades": len(ordered),
            "wins": sum(value > 0 for value in pnl_values),
            "losses": sum(value < 0 for value in pnl_values),
            "win_percent": (
                sum(value > 0 for value in pnl_values) / len(ordered) * 100 if ordered else None
            ),
            "realised_pnl": sum(pnl_values),
            "unrealised_pnl": unrealised_pnl,
            "unrealised_status": unrealised_status,
            "open_positions": open_positions,
            "total_r": total_r,
            "mean_r": total_r / len(ordered) if total_r is not None and ordered else None,
            "max_realised_drawdown": _max_drawdown(pnl_values),
            "history": history,
        }

    def _records(self, *, run_id: str | None = None) -> tuple[ExecutionRecord, ...]:
        values: list[ExecutionRecord] = []
        offset = 0
        while True:
            page, total = self.ledger.list_records(run_id=run_id, limit=500, offset=offset)
            values.extend(page)
            offset += len(page)
            if offset >= total or not page:
                return tuple(values)

    @staticmethod
    def _in_period(
        timestamp: datetime,
        period: PerformancePeriod,
        today: date,
        calendar: str | None,
        timezone: ZoneInfo,
    ) -> bool:
        session = _aware(timestamp).astimezone(timezone).date()
        if period is PerformancePeriod.ALL:
            return True
        if period is PerformancePeriod.TODAY:
            return session == today
        count = 5 if period is PerformancePeriod.SESSIONS_5 else 20
        if calendar is None:
            return session >= today - timedelta(days=count * 2)
        schedule = get_market_calendar(calendar).schedule(
            start_date=today - timedelta(days=count * 4), end_date=today
        )
        selected = {item.date() for item in schedule.index[-count:]}
        return session in selected

    @staticmethod
    def _realised_r(record: ExecutionRecord) -> float | None:
        if record.realized_pnl is None or record.per_share_initial_risk is None:
            return None
        original_risk = record.per_share_initial_risk * record.filled_quantity
        if original_risk <= 0:
            return None
        return record.realized_pnl / original_risk

    def _unrealised(self, run: RunConfig, now: datetime) -> tuple[float | None, str, int]:
        all_records = self._records()
        open_records = tuple(
            item
            for item in all_records
            if item.filled_quantity > item.closed_quantity and item.actual_account is not None
        )
        target = tuple(item for item in open_records if item.run_id == run.run_id)
        if not target:
            return 0.0, "AVAILABLE", 0
        broker_positions = {
            (item.environment, item.account, item.con_id): item
            for item in self.ledger.broker_position_snapshots()
        }
        persisted_marks = {
            (item.environment, item.account, item.con_id): item
            for item in self.ledger.broker_position_marks()
        }
        attributed: defaultdict[tuple[Environment, str, int], float] = defaultdict(float)
        for item in open_records:
            assert item.actual_account is not None
            direction = -1.0 if item.side is OrderAction.SELL else 1.0
            attributed[(item.environment, item.actual_account, item.con_id)] += direction * (
                item.filled_quantity - item.closed_quantity
            )
        total = 0.0
        identities = {
            (item.environment, item.actual_account or item.expected_account, item.con_id)
            for item in target
        }
        for identity in identities:
            broker = broker_positions.get(identity)
            if broker is None or abs(attributed[identity] - broker.quantity) > 1e-9:
                return None, "RECONCILIATION_REQUIRED", len(identities)
            if self.marks is not None:
                mark = self.marks.get(identity)
            else:
                persisted = persisted_marks.get(identity)
                mark = (
                    persisted.mark
                    if persisted is not None
                    and _aware(now) - _aware(persisted.observed_at) <= POSITION_MARK_MAX_AGE
                    else None
                )
            if mark is None:
                return None, "MARK_UNAVAILABLE", len(identities)
            lots = [
                item
                for item in target
                if (item.environment, item.actual_account, item.con_id) == identity
            ]
            for item in lots:
                if item.average_fill_price is None:
                    return None, "RECONCILIATION_REQUIRED", len(identities)
                quantity = item.filled_quantity - item.closed_quantity
                if item.side is OrderAction.SELL:
                    total += quantity * (item.average_fill_price - mark)
                else:
                    total += quantity * (mark - item.average_fill_price)
        return total, "AVAILABLE", len(identities)

    @staticmethod
    def _history(records: tuple[ExecutionRecord, ...], timezone: ZoneInfo) -> list[dict[str, Any]]:
        grouped: defaultdict[date, list[ExecutionRecord]] = defaultdict(list)
        for item in records:
            assert item.closed_at is not None
            grouped[_aware(item.closed_at).astimezone(timezone).date()].append(item)
        results = []
        for session in sorted(grouped):
            items = grouped[session]
            rs = [RunPerformanceService._realised_r(item) for item in items]
            results.append(
                {
                    "date": session.isoformat(),
                    "trades": len(items),
                    "pnl": sum(float(item.realized_pnl or 0.0) for item in items),
                    "r": sum(value for value in rs if value is not None)
                    if all(value is not None for value in rs)
                    else None,
                }
            )
        return results


def _max_drawdown(values: tuple[float, ...]) -> float:
    peak = 0.0
    cumulative = 0.0
    maximum = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("performance timestamps must be timezone-aware")
    return value.astimezone(UTC)
