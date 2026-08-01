"""Audited completed five-minute bar adapter with exact XNYS checkpoint indexing."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, field_validator


class HistoricalBarUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: int
    symbol: str
    con_id: int
    bar_start_utc: datetime
    provider_timestamp_utc: datetime
    received_timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    wap: float | None
    trade_count: int | None
    source: str
    explicitly_finalised: bool

    @field_validator(
        "bar_start_utc",
        "provider_timestamp_utc",
        "received_timestamp_utc",
    )
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bar timestamps must be timezone-aware")
        return value.astimezone(UTC)


class AuditedLiveBar(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    bar_start_utc: datetime
    bar_end_utc: datetime
    checkpoint: int
    open: float
    high: float
    low: float
    close: float
    volume_or_activity_field: float | None
    wap_where_available: float | None
    trade_count_where_available: int | None
    source: str
    source_completeness: str
    finalised: bool
    provider_timestamp_utc: datetime
    received_timestamp_utc: datetime


@lru_cache(maxsize=512)
def _xnys_session_bounds(session: date) -> tuple[datetime, datetime]:
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("XNYS").schedule(
        start_date=session,
        end_date=session,
    )
    if schedule.empty:
        raise ValueError("bar session is not an XNYS trading session")
    row = schedule.iloc[0]
    return (
        row["market_open"].to_pydatetime().astimezone(UTC),
        row["market_close"].to_pydatetime().astimezone(UTC),
    )


def xnys_session_bounds(session: date) -> tuple[datetime, datetime]:
    """Return audited regular-session bounds for an exact XNYS session."""

    return _xnys_session_bounds(session)


def checkpoint_for_bar(bar_start_utc: datetime) -> int:
    if bar_start_utc.tzinfo is None or bar_start_utc.utcoffset() is None:
        raise ValueError("bar timestamp must be timezone-aware")
    start = bar_start_utc.astimezone(UTC)
    # The UTC date equals the New York session date for regular US hours.
    market_open, market_close = _xnys_session_bounds(start.date())
    if start < market_open or start >= market_close:
        raise ValueError("bar start is outside XNYS regular trading hours")
    elapsed = start - market_open
    if elapsed.total_seconds() % 300 != 0:
        raise ValueError("bar start is not aligned to a five-minute checkpoint")
    return int(elapsed.total_seconds() // 300) + 1


class KeepUpToDateBarFinalizer:
    """Finalize an IBKR bar only when the next five-minute bar begins."""

    def __init__(self, *, prospective_collection_start: datetime) -> None:
        if (
            prospective_collection_start.tzinfo is None
            or prospective_collection_start.utcoffset() is None
        ):
            raise ValueError("prospective_collection_start must be timezone-aware")
        self.prospective_collection_start = prospective_collection_start.astimezone(UTC)
        self._contracts: dict[int, tuple[str, int]] = {}
        self._pending: dict[int, HistoricalBarUpdate] = {}

    def register(self, request_id: int, *, symbol: str, con_id: int) -> None:
        identity = (symbol, con_id)
        existing = self._contracts.get(request_id)
        if existing is not None and existing != identity:
            raise ValueError("historical bar request ownership differs")
        self._contracts[request_id] = identity

    def add(
        self,
        *,
        request_id: int,
        bar_start_utc: datetime,
        provider_timestamp_utc: datetime,
        received_timestamp_utc: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float | None,
        wap: float | None,
        trade_count: int | None,
    ) -> tuple[HistoricalBarUpdate, ...]:
        if request_id not in self._contracts:
            raise ValueError("unknown historical bar request")
        symbol, con_id = self._contracts[request_id]
        current = HistoricalBarUpdate(
            request_id=request_id,
            symbol=symbol,
            con_id=con_id,
            bar_start_utc=bar_start_utc,
            provider_timestamp_utc=provider_timestamp_utc,
            received_timestamp_utc=received_timestamp_utc,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            wap=wap,
            trade_count=trade_count,
            source="ibkr_historical_keep_up_to_date",
            explicitly_finalised=False,
        )
        previous = self._pending.get(request_id)
        if previous is not None and current.bar_start_utc < previous.bar_start_utc:
            raise ValueError("historical bar updates moved backwards")
        self._pending[request_id] = current
        if previous is None or current.bar_start_utc == previous.bar_start_utc:
            return ()
        finalised = previous.model_copy(update={"explicitly_finalised": True})
        if finalised.bar_start_utc < self.prospective_collection_start:
            return ()
        return (finalised,)

    def pending_for_symbol_session(
        self,
        *,
        symbol: str,
        session: date,
    ) -> HistoricalBarUpdate | None:
        """Expose the latest received update without mutating bar finalization state."""

        candidates = tuple(
            update
            for update in self._pending.values()
            if update.symbol == symbol and update.bar_start_utc.date() == session
        )
        return max(
            candidates,
            key=lambda update: (
                update.bar_start_utc,
                update.received_timestamp_utc,
                update.request_id,
            ),
            default=None,
        )


class AuditedFiveMinuteBarAdapter:
    """Emit each explicitly completed keepUpToDate bar exactly once."""

    def __init__(self) -> None:
        self._latest: dict[tuple[int, datetime], HistoricalBarUpdate] = {}
        self._finalised: dict[tuple[int, datetime], AuditedLiveBar] = {}

    def add(self, update: HistoricalBarUpdate) -> tuple[AuditedLiveBar, ...]:
        key = (update.request_id, update.bar_start_utc)
        existing_final = self._finalised.get(key)
        if existing_final is not None:
            candidate = self._to_bar(update)
            if candidate != existing_final:
                raise ValueError("finalised bar changed after emission")
            return ()
        self._latest[key] = update
        if not update.explicitly_finalised:
            return ()
        bar = self._to_bar(update)
        self._finalised[key] = bar
        return (bar,)

    @staticmethod
    def _to_bar(update: HistoricalBarUpdate) -> AuditedLiveBar:
        if update.high < max(update.open, update.close, update.low) or update.low > min(
            update.open,
            update.close,
            update.high,
        ):
            raise ValueError("invalid completed bar OHLC")
        checkpoint = checkpoint_for_bar(update.bar_start_utc)
        return AuditedLiveBar(
            symbol=update.symbol,
            session=update.bar_start_utc.date(),
            bar_start_utc=update.bar_start_utc,
            bar_end_utc=update.bar_start_utc + timedelta(minutes=5),
            checkpoint=checkpoint,
            open=update.open,
            high=update.high,
            low=update.low,
            close=update.close,
            volume_or_activity_field=update.volume,
            wap_where_available=update.wap,
            trade_count_where_available=update.trade_count,
            source=update.source,
            source_completeness="complete",
            finalised=True,
            provider_timestamp_utc=update.provider_timestamp_utc,
            received_timestamp_utc=update.received_timestamp_utc,
        )


class BarCompatibilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows_compared: int
    timestamp_mismatches: int
    rth_mismatches: int
    checkpoint_mismatches: int
    ohlc_mismatches: int
    missing_bar_mismatches: int
    corporate_action_mismatches: int
    maximum_ohlc_difference: float
    passed: bool


def compare_bar_semantics(
    research: tuple[AuditedLiveBar, ...],
    live: tuple[AuditedLiveBar, ...],
    *,
    tolerance: float = 1e-12,
    research_rth_only: bool = True,
    live_rth_only: bool = True,
    research_corporate_action_policy: str = "unadjusted_intraday",
    live_corporate_action_policy: str = "unadjusted_intraday",
) -> BarCompatibilityReport:
    research_by_key = {(item.symbol, item.session, item.checkpoint): item for item in research}
    live_by_key = {(item.symbol, item.session, item.checkpoint): item for item in live}
    all_keys = set(research_by_key).union(live_by_key)
    timestamp_mismatches = checkpoint_mismatches = ohlc_mismatches = 0
    maximum = 0.0
    for key in sorted(all_keys):
        left = research_by_key.get(key)
        right = live_by_key.get(key)
        if left is None or right is None:
            continue
        timestamp_mismatches += int(
            left.bar_start_utc != right.bar_start_utc or left.bar_end_utc != right.bar_end_utc
        )
        checkpoint_mismatches += int(left.checkpoint != right.checkpoint)
        differences = [
            abs(getattr(left, field) - getattr(right, field))
            for field in ("open", "high", "low", "close")
        ]
        maximum = max(maximum, *differences)
        ohlc_mismatches += int(max(differences) > tolerance)
    missing = len(set(research_by_key).symmetric_difference(live_by_key))
    rth_mismatches = int(research_rth_only != live_rth_only)
    corporate_action_mismatches = int(
        research_corporate_action_policy != live_corporate_action_policy
    )
    passed = (
        timestamp_mismatches == 0
        and rth_mismatches == 0
        and checkpoint_mismatches == 0
        and ohlc_mismatches == 0
        and missing == 0
        and corporate_action_mismatches == 0
        and maximum <= tolerance
    )
    return BarCompatibilityReport(
        rows_compared=len(set(research_by_key).intersection(live_by_key)),
        timestamp_mismatches=timestamp_mismatches,
        rth_mismatches=rth_mismatches,
        checkpoint_mismatches=checkpoint_mismatches,
        ohlc_mismatches=ohlc_mismatches,
        missing_bar_mismatches=missing,
        corporate_action_mismatches=corporate_action_mismatches,
        maximum_ohlc_difference=maximum,
        passed=passed,
    )
