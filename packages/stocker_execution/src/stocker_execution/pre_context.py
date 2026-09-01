"""Canonical prior-session volatility context for Stage 4."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from itertools import product
from math import inf, isfinite, log, pi, sqrt
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from stocker_execution.history import (
    HistorySemantics,
    HistorySnapshot,
    HistoryStatus,
    IbkrHistoryCache,
    IbkrHistoryService,
)
from stocker_execution.ibkr import (
    IbkrConnection,
    IbkrError,
    OptionChainDefinition,
    OptionContractRequest,
    OptionMarketSnapshot,
    QualifiedInstrument,
    QualifiedOption,
)

PRE_CONTEXT_CALCULATION_VERSION = "PRE_CONTEXT_V1"
IBKR_MODEL_OPTION_COMPUTATION_TICK_TYPE = 13
IBKR_MODEL_OPTION_IV_SOURCE = "IBKR_MODEL_OPTION_COMPUTATION_TICK_13"
PRE_CONTEXT_NOT_READY = "PRE_CONTEXT_NOT_READY"
PRE_UNDERLYING_HISTORY = HistorySemantics("5 mins", "TRADES", True)
PRE_CONTEXT_OBSERVATION_MINUTE = timedelta(minutes=1)


class ContextStatus(StrEnum):
    """Availability of the complete prior-session volatility context."""

    READY = "READY"
    NOT_READY = PRE_CONTEXT_NOT_READY


@dataclass(frozen=True, slots=True)
class PriorSessionVolatilityContext:
    """Persisted IBKR-only lineage needed by Stage 5 to calculate M_price."""

    underlying_con_id: int
    symbol: str
    security_type: str
    exchange: str
    primary_exchange: str | None
    currency: str
    target_session: date
    observation_session: date
    option_observation_at: datetime
    prior_session_reference_price: float
    call_con_id: int
    put_con_id: int
    option_exchange: str
    expiry: date
    strike: float
    call_model_iv: float
    put_model_iv: float
    call_market_data_type: int
    put_market_data_type: int
    atm_iv: float
    expected_absolute_return_15m: float
    iv_source: str
    calculation_version: str
    captured_at: datetime
    calculated_at: datetime


@dataclass(frozen=True, slots=True)
class PriorSessionContextResult:
    """Fail-closed outcome for one qualified instrument and target session."""

    status: ContextStatus
    context: PriorSessionVolatilityContext | None
    reason: str
    reused: bool = False


@dataclass(frozen=True, slots=True)
class VolatilityCalculation:
    """Pure canonical volatility arithmetic, independent of any broker."""

    atm_iv: float
    expected_absolute_return_15m: float


@dataclass(frozen=True, slots=True)
class OptionCandidate:
    """One normalized contract-specific IBKR option snapshot."""

    con_id: int
    symbol: str
    exchange: str
    currency: str
    expiry: date
    strike: float
    right: str
    bid: float | None
    ask: float | None
    open_interest: float | None
    model_iv: float | None
    model_delta: float | None = None
    model_gamma: float | None = None

    @property
    def midpoint(self) -> float | None:
        """Return the unrounded quote midpoint when both sides are finite."""

        if self.bid is None or self.ask is None:
            return None
        if not isfinite(self.bid) or not isfinite(self.ask):
            return None
        return (self.bid + self.ask) / 2


@dataclass(frozen=True, slots=True)
class CanonicalOptionPair:
    """The single call/put pair selected by the frozen research ordering."""

    call: OptionCandidate
    put: OptionCandidate
    expiry: date
    dte: int
    strike: float


def calculate_volatility(*, call_model_iv: float, put_model_iv: float) -> VolatilityCalculation:
    """Calculate the frozen call/put ATM IV and 15-minute expected absolute return."""

    if not all(isfinite(value) and 0.005 <= value <= 5 for value in (call_model_iv, put_model_iv)):
        raise ValueError("call and put model IV must each be finite and in [0.005, 5]")
    atm_iv = (call_model_iv + put_model_iv) / 2
    expected_absolute_return_15m = atm_iv * sqrt(15 / (252 * 390)) * sqrt(2 / pi)
    return VolatilityCalculation(
        atm_iv=atm_iv,
        expected_absolute_return_15m=expected_absolute_return_15m,
    )


def select_canonical_option_pair(
    candidates: tuple[OptionCandidate, ...],
    *,
    previous_close: float,
    observation_date: date,
) -> CanonicalOptionPair:
    """Select exactly one option pair using the frozen research ordering."""

    if not isfinite(previous_close) or previous_close <= 0:
        raise ValueError("previous close must be finite and positive")
    eligible = tuple(
        item
        for item in candidates
        if item.con_id > 0
        and item.right.upper() in {"C", "CALL", "P", "PUT"}
        and 7 <= (item.expiry - observation_date).days <= 45
        and previous_close * 0.75 <= item.strike <= previous_close * 1.25
        and isfinite(item.strike)
        and item.strike > 0
    )
    expiries: list[tuple[int, date]] = []
    for expiry in sorted({item.expiry for item in eligible}):
        expiry_items = tuple(item for item in eligible if item.expiry == expiry)
        call_strikes = {item.strike for item in expiry_items if _is_call(item)}
        put_strikes = {item.strike for item in expiry_items if not _is_call(item)}
        if call_strikes & put_strikes:
            expiries.append(((expiry - observation_date).days, expiry))
    if not expiries:
        raise ValueError("no eligible common-strike expiry")

    selected_dte, selected_expiry = min(expiries, key=lambda item: (item[0], item[1]))
    at_expiry = tuple(item for item in eligible if item.expiry == selected_expiry)
    ranked: list[tuple[tuple[object, ...], OptionCandidate, OptionCandidate]] = []
    common_strikes = sorted(
        {item.strike for item in at_expiry if _is_call(item)}
        & {item.strike for item in at_expiry if not _is_call(item)}
    )
    for strike in common_strikes:
        calls = (item for item in at_expiry if _is_call(item) and item.strike == strike)
        puts = (item for item in at_expiry if not _is_call(item) and item.strike == strike)
        for call, put in product(calls, puts):
            minimum_oi = min(
                -inf if call.open_interest is None else call.open_interest,
                -inf if put.open_interest is None else put.open_interest,
            )
            iv_gap = (
                inf
                if call.model_iv is None or put.model_iv is None
                else abs(call.model_iv - put.model_iv)
            )
            rank: tuple[object, ...] = (
                abs(log(strike / previous_close)),
                -minimum_oi,
                _combined_relative_spread(call, put),
                iv_gap,
                strike,
                str(call.con_id),
                str(put.con_id),
            )
            ranked.append((rank, call, put))
    _, call, put = min(ranked, key=lambda item: item[0])
    for side, item in (("call", call), ("put", put)):
        reason = _quality_reason(side, item, observation_date=observation_date)
        if reason is not None:
            raise ValueError(reason)
    return CanonicalOptionPair(
        call=call,
        put=put,
        expiry=selected_expiry,
        dte=selected_dte,
        strike=call.strike,
    )


def _is_call(candidate: OptionCandidate) -> bool:
    return candidate.right.upper() in {"C", "CALL"}


def _relative_spread(candidate: OptionCandidate) -> float:
    midpoint = candidate.midpoint
    if (
        candidate.bid is None
        or candidate.ask is None
        or midpoint is None
        or midpoint <= 0
        or candidate.ask < candidate.bid
    ):
        return inf
    return (candidate.ask - candidate.bid) / midpoint


def _combined_relative_spread(call: OptionCandidate, put: OptionCandidate) -> float:
    values = (call.bid, call.ask, put.bid, put.ask, call.midpoint, put.midpoint)
    if any(value is None for value in values):
        return inf
    call_bid, call_ask, put_bid, put_ask, call_midpoint, put_midpoint = cast(
        tuple[float, float, float, float, float, float], values
    )
    denominator = call_midpoint + put_midpoint
    if denominator <= 0 or call_ask < call_bid or put_ask < put_bid:
        return inf
    return (call_ask - call_bid + put_ask - put_bid) / denominator


def _quality_reason(side: str, candidate: OptionCandidate, *, observation_date: date) -> str | None:
    if (
        candidate.model_iv is None
        or not isfinite(candidate.model_iv)
        or not 0.005 <= candidate.model_iv <= 5
    ):
        return f"selected pair {side} model IV invalid"
    if candidate.bid is None or not isfinite(candidate.bid) or candidate.bid < 0:
        return f"selected pair {side} bid invalid"
    if candidate.ask is None or not isfinite(candidate.ask) or candidate.ask < candidate.bid:
        return f"selected pair {side} ask invalid"
    midpoint = candidate.midpoint
    if midpoint is None or midpoint <= 0:
        return f"selected pair {side} midpoint not positive"
    if (
        candidate.open_interest is None
        or not isfinite(candidate.open_interest)
        or candidate.open_interest < 10
    ):
        return "selected pair open interest below 10"
    if _relative_spread(candidate) > 1:
        return f"selected pair {side} relative spread above 1"
    if candidate.model_delta is not None and (
        not isfinite(candidate.model_delta) or abs(candidate.model_delta) > 1.05
    ):
        return f"selected pair {side} delta implausible"
    if candidate.model_gamma is not None and (
        not isfinite(candidate.model_gamma) or candidate.model_gamma < 0
    ):
        return f"selected pair {side} gamma negative"
    if candidate.expiry < observation_date:
        return f"selected pair {side} expiration before trade"
    return None


class PriorSessionContextStore:
    """One SQLite table for reusable IBKR prior-session volatility contexts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ibkr_pre_volatility_contexts (
                    underlying_con_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    security_type TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    primary_exchange TEXT,
                    currency TEXT NOT NULL,
                    target_session TEXT NOT NULL,
                    observation_session TEXT NOT NULL,
                    option_observation_at_utc TEXT NOT NULL,
                    prior_session_reference_price REAL NOT NULL,
                    call_con_id INTEGER NOT NULL,
                    put_con_id INTEGER NOT NULL,
                    option_exchange TEXT NOT NULL,
                    expiry TEXT NOT NULL,
                    strike REAL NOT NULL,
                    call_model_iv REAL NOT NULL,
                    put_model_iv REAL NOT NULL,
                    call_market_data_type INTEGER NOT NULL,
                    put_market_data_type INTEGER NOT NULL,
                    atm_iv REAL NOT NULL,
                    expected_absolute_return_15m REAL NOT NULL,
                    iv_source TEXT NOT NULL CHECK (
                        iv_source = 'IBKR_MODEL_OPTION_COMPUTATION_TICK_13'
                    ),
                    calculation_version TEXT NOT NULL,
                    captured_at_utc TEXT NOT NULL,
                    calculated_at_utc TEXT NOT NULL,
                    PRIMARY KEY (underlying_con_id, target_session, calculation_version)
                )
                """
            )

    def get(
        self, instrument: QualifiedInstrument, *, session: date
    ) -> PriorSessionVolatilityContext | None:
        """Load one environment-independent context by conId, session, and version."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM ibkr_pre_volatility_contexts
                WHERE underlying_con_id = ?
                  AND target_session = ?
                  AND calculation_version = ?
                  AND iv_source = ?
                """,
                (
                    instrument.con_id,
                    session.isoformat(),
                    PRE_CONTEXT_CALCULATION_VERSION,
                    IBKR_MODEL_OPTION_IV_SOURCE,
                ),
            ).fetchone()
        if row is None:
            return None
        return _context_from_row(row)

    def _store_from_ibkr(self, context: PriorSessionVolatilityContext) -> None:
        """Persist a context created only by the concrete Stage 2 IBKR ingress."""

        if context.iv_source != IBKR_MODEL_OPTION_IV_SOURCE:
            raise ValueError("Prior-session context source must be IBKR tick-13 model IV")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ibkr_pre_volatility_contexts VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT (underlying_con_id, target_session, calculation_version)
                DO NOTHING
                """,
                _context_row(context),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


class PriorSessionContextService:
    """Compose Stage 2 IBKR data, cache, selector, calculation, and persistence."""

    def __init__(
        self,
        ibkr: IbkrConnection,
        history_cache: IbkrHistoryCache,
        context_store: PriorSessionContextStore,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ibkr, IbkrConnection):
            raise TypeError("PRE context ingress requires the concrete Stage 2 IbkrConnection")
        self._ibkr = ibkr
        self._history_cache = history_cache
        self._history_service = IbkrHistoryService(ibkr, history_cache)
        self._context_store = context_store
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    async def get_or_create(
        self, instrument: QualifiedInstrument, *, session: date
    ) -> PriorSessionContextResult:
        """Return or causally capture one complete IBKR prior-session context."""

        try:
            existing = self._context_store.get(instrument, session=session)
            if existing is not None:
                return PriorSessionContextResult(ContextStatus.READY, existing, "cache hit", True)
            observation, previous_open, previous_close, target_open = _session_contract(session)
            option_observation_at = _option_observation_at(observation)
            now = _aware_utc(self._clock())
            if (
                now < option_observation_at
                or now >= option_observation_at + PRE_CONTEXT_OBSERVATION_MINUTE
                or now >= target_open
            ):
                raise ValueError(
                    "uncached option context must be captured during the canonical "
                    "16:00 America/New_York observation minute"
                )
            required = _five_minute_starts(previous_open, previous_close)
            history = self._history_cache.get_required_history(
                instrument,
                PRE_UNDERLYING_HISTORY,
                required,
                as_of=previous_close,
            )
            if history.status is HistoryStatus.NOT_READY:
                await self._fill_missing_history(
                    instrument,
                    history,
                    required_count=len(required),
                    previous_close=previous_close,
                )
                history = self._history_cache.get_required_history(
                    instrument,
                    PRE_UNDERLYING_HISTORY,
                    required,
                    as_of=previous_close,
                )
            if history.status is not HistoryStatus.READY:
                raise ValueError(history.reason)
            previous_reference = history.bars[-1].close

            chains = await self._ibkr.option_chains(instrument)
            request_groups = _option_request_groups(
                instrument,
                chains,
                previous_close=previous_reference,
                observation_date=observation,
            )
            nearest_qualified: tuple[QualifiedOption, ...] = ()
            for requests in request_groups:
                qualified = await self._ibkr.qualify_options(requests)
                try:
                    nearest_qualified = _nearest_common_qualified_options(
                        qualified,
                        previous_close=previous_reference,
                    )
                except ValueError:
                    continue
                break
            if not nearest_qualified:
                raise ValueError("no eligible qualified common-strike expiry")
            snapshots = await self._ibkr.option_snapshots(nearest_qualified)
            if any(
                not option_observation_at
                <= _aware_utc(item.captured_at)
                < option_observation_at + PRE_CONTEXT_OBSERVATION_MINUTE
                for item in snapshots
            ):
                raise ValueError("option model IV snapshot is not causal for the target session")
            candidates = tuple(_candidate_from_snapshot(item) for item in snapshots)
            pair = select_canonical_option_pair(
                candidates,
                previous_close=previous_reference,
                observation_date=observation,
            )
            calculation = calculate_volatility(
                call_model_iv=_required_iv(pair.call),
                put_model_iv=_required_iv(pair.put),
            )
            by_con_id = {item.option.con_id: item for item in snapshots}
            captured_at = max(
                _aware_utc(by_con_id[pair.call.con_id].captured_at),
                _aware_utc(by_con_id[pair.put.con_id].captured_at),
            )
            context = PriorSessionVolatilityContext(
                underlying_con_id=instrument.con_id,
                symbol=instrument.symbol,
                security_type=instrument.security_type,
                exchange=instrument.exchange,
                primary_exchange=instrument.primary_exchange,
                currency=instrument.currency,
                target_session=session,
                observation_session=observation,
                option_observation_at=option_observation_at,
                prior_session_reference_price=previous_reference,
                call_con_id=pair.call.con_id,
                put_con_id=pair.put.con_id,
                option_exchange=pair.call.exchange,
                expiry=pair.expiry,
                strike=pair.strike,
                call_model_iv=_required_iv(pair.call),
                put_model_iv=_required_iv(pair.put),
                call_market_data_type=_required_market_data_type(
                    by_con_id[pair.call.con_id]
                ),
                put_market_data_type=_required_market_data_type(
                    by_con_id[pair.put.con_id]
                ),
                atm_iv=calculation.atm_iv,
                expected_absolute_return_15m=calculation.expected_absolute_return_15m,
                iv_source=IBKR_MODEL_OPTION_IV_SOURCE,
                calculation_version=PRE_CONTEXT_CALCULATION_VERSION,
                captured_at=captured_at,
                calculated_at=now,
            )
            self._context_store._store_from_ibkr(context)
            persisted = self._context_store.get(instrument, session=session)
            if persisted is None:
                raise ValueError("calculated PRE context was not persisted")
            return PriorSessionContextResult(ContextStatus.READY, persisted, "calculated")
        except (IbkrError, ValueError) as exc:
            return PriorSessionContextResult(
                ContextStatus.NOT_READY,
                None,
                f"{PRE_CONTEXT_NOT_READY}: {exc}",
            )

    async def _fill_missing_history(
        self,
        instrument: QualifiedInstrument,
        history: HistorySnapshot,
        *,
        required_count: int,
        previous_close: datetime,
    ) -> None:
        ranges: tuple[tuple[str, datetime], ...]
        if len(history.missing_timestamps) == required_count:
            ranges = (("1 D", previous_close),)
        else:
            ranges = tuple(
                (
                    f"{len(group) * 300} S",
                    group[-1] + timedelta(minutes=5),
                )
                for group in _contiguous_five_minute_ranges(history.missing_timestamps)
            )
        for duration, end_time in ranges:
            await self._history_service.fetch_and_store(
                instrument,
                bar_size=PRE_UNDERLYING_HISTORY.bar_size,
                duration=duration,
                what_to_show=PRE_UNDERLYING_HISTORY.what_to_show,
                regular_trading_hours=PRE_UNDERLYING_HISTORY.regular_trading_hours,
                end_time=end_time,
            )


def _session_contract(session: date) -> tuple[date, datetime, datetime, datetime]:
    from stocker_data.calendars import get_market_calendar

    calendar = get_market_calendar("XNYS")
    schedule = calendar.schedule(
        start_date=session - timedelta(days=14),
        end_date=session,
    )
    rows = [
        (index.date(), _aware_utc(row["market_open"]), _aware_utc(row["market_close"]))
        for index, row in schedule.iterrows()
    ]
    target_index = next((index for index, row in enumerate(rows) if row[0] == session), None)
    if target_index is None or target_index == 0:
        raise ValueError("target and previous trading sessions could not be established")
    observation, previous_open, previous_close = rows[target_index - 1]
    _, target_open, _ = rows[target_index]
    return observation, previous_open, previous_close, target_open


def _five_minute_starts(start: datetime, end: datetime) -> tuple[datetime, ...]:
    values: list[datetime] = []
    cursor = _aware_utc(start)
    end_utc = _aware_utc(end)
    while cursor < end_utc:
        values.append(cursor)
        cursor += timedelta(minutes=5)
    if not values or cursor != end_utc:
        raise ValueError("session cannot be represented as complete five-minute bars")
    return tuple(values)


def _option_observation_at(observation_session: date) -> datetime:
    return datetime.combine(
        observation_session,
        time(16, 0),
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(UTC)


def _contiguous_five_minute_ranges(
    timestamps: tuple[datetime, ...],
) -> tuple[tuple[datetime, ...], ...]:
    groups: list[list[datetime]] = []
    for timestamp in timestamps:
        if not groups or timestamp - groups[-1][-1] != timedelta(minutes=5):
            groups.append([timestamp])
        else:
            groups[-1].append(timestamp)
    return tuple(tuple(group) for group in groups)


def _option_request_groups(
    instrument: QualifiedInstrument,
    chains: tuple[OptionChainDefinition, ...],
    *,
    previous_close: float,
    observation_date: date,
) -> tuple[tuple[OptionContractRequest, ...], ...]:
    smart = tuple(item for item in chains if item.exchange.upper() == "SMART")
    eligible_expiries = sorted(
        {
            expiry
            for chain in smart
            for expiry in chain.expirations
            if 7 <= (expiry - observation_date).days <= 45
        },
        key=lambda expiry: ((expiry - observation_date).days, expiry),
    )
    if not eligible_expiries:
        raise ValueError("no eligible SMART option expiry")
    groups: list[tuple[OptionContractRequest, ...]] = []
    for expiry in eligible_expiries:
        requests: list[OptionContractRequest] = []
        for chain in smart:
            if expiry not in chain.expirations:
                continue
            for strike in chain.strikes:
                if not previous_close * 0.75 <= strike <= previous_close * 1.25 or strike <= 0:
                    continue
                for right in ("C", "P"):
                    requests.append(
                        OptionContractRequest(
                            symbol=instrument.symbol,
                            exchange=chain.exchange,
                            currency=instrument.currency,
                            expiry=expiry,
                            strike=strike,
                            right=right,
                            multiplier=chain.multiplier,
                            trading_class=chain.trading_class,
                        )
                    )
        if requests:
            groups.append(tuple(requests))
    if not groups:
        raise ValueError("no eligible option strike between 75% and 125% of previous close")
    return tuple(groups)


def _nearest_common_qualified_options(
    options: tuple[QualifiedOption, ...], *, previous_close: float
) -> tuple[QualifiedOption, ...]:
    calls = {item.strike for item in options if item.right.upper() in {"C", "CALL"}}
    puts = {item.strike for item in options if item.right.upper() in {"P", "PUT"}}
    common = calls & puts
    if not common:
        raise ValueError("qualified expiry has no common call/put strike")
    nearest_distance = min(abs(log(strike / previous_close)) for strike in common)
    nearest_strikes = {
        strike
        for strike in common
        if abs(log(strike / previous_close)) == nearest_distance
    }
    return tuple(
        item
        for item in options
        if item.strike in nearest_strikes
        and item.right.upper() in {"C", "CALL", "P", "PUT"}
    )


def _candidate_from_snapshot(snapshot: OptionMarketSnapshot) -> OptionCandidate:
    option = snapshot.option
    return OptionCandidate(
        con_id=option.con_id,
        symbol=option.symbol,
        exchange=option.exchange,
        currency=option.currency,
        expiry=option.expiry,
        strike=option.strike,
        right=option.right,
        bid=snapshot.bid,
        ask=snapshot.ask,
        open_interest=snapshot.open_interest,
        model_iv=snapshot.model_iv,
        model_delta=snapshot.model_delta,
        model_gamma=snapshot.model_gamma,
    )


def _required_iv(candidate: OptionCandidate) -> float:
    if candidate.model_iv is None:
        raise ValueError("selected pair model IV missing")
    return candidate.model_iv


def _required_market_data_type(snapshot: OptionMarketSnapshot) -> int:
    if snapshot.market_data_type not in {1, 2}:
        raise ValueError("selected pair did not originate from tick-13 market data")
    return snapshot.market_data_type


def _aware_utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Stage 4 context timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _context_row(context: PriorSessionVolatilityContext) -> tuple[object, ...]:
    return (
        context.underlying_con_id,
        context.symbol,
        context.security_type,
        context.exchange,
        context.primary_exchange,
        context.currency,
        context.target_session.isoformat(),
        context.observation_session.isoformat(),
        _aware_utc(context.option_observation_at).isoformat(),
        context.prior_session_reference_price,
        context.call_con_id,
        context.put_con_id,
        context.option_exchange,
        context.expiry.isoformat(),
        context.strike,
        context.call_model_iv,
        context.put_model_iv,
        context.call_market_data_type,
        context.put_market_data_type,
        context.atm_iv,
        context.expected_absolute_return_15m,
        context.iv_source,
        context.calculation_version,
        _aware_utc(context.captured_at).isoformat(),
        _aware_utc(context.calculated_at).isoformat(),
    )


def _context_from_row(row: sqlite3.Row) -> PriorSessionVolatilityContext:
    context = PriorSessionVolatilityContext(
        underlying_con_id=int(row["underlying_con_id"]),
        symbol=str(row["symbol"]),
        security_type=str(row["security_type"]),
        exchange=str(row["exchange"]),
        primary_exchange=(
            None if row["primary_exchange"] is None else str(row["primary_exchange"])
        ),
        currency=str(row["currency"]),
        target_session=date.fromisoformat(str(row["target_session"])),
        observation_session=date.fromisoformat(str(row["observation_session"])),
        option_observation_at=_aware_utc(
            datetime.fromisoformat(str(row["option_observation_at_utc"]))
        ),
        prior_session_reference_price=float(row["prior_session_reference_price"]),
        call_con_id=int(row["call_con_id"]),
        put_con_id=int(row["put_con_id"]),
        option_exchange=str(row["option_exchange"]),
        expiry=date.fromisoformat(str(row["expiry"])),
        strike=float(row["strike"]),
        call_model_iv=float(row["call_model_iv"]),
        put_model_iv=float(row["put_model_iv"]),
        call_market_data_type=int(row["call_market_data_type"]),
        put_market_data_type=int(row["put_market_data_type"]),
        atm_iv=float(row["atm_iv"]),
        expected_absolute_return_15m=float(row["expected_absolute_return_15m"]),
        iv_source=str(row["iv_source"]),
        calculation_version=str(row["calculation_version"]),
        captured_at=_aware_utc(datetime.fromisoformat(str(row["captured_at_utc"]))),
        calculated_at=_aware_utc(datetime.fromisoformat(str(row["calculated_at_utc"]))),
    )
    _validate_context(context)
    return context


def _validate_context(context: PriorSessionVolatilityContext) -> None:
    if min(context.underlying_con_id, context.call_con_id, context.put_con_id) <= 0:
        raise ValueError("persisted PRE context contains invalid conId lineage")
    if (
        not isfinite(context.prior_session_reference_price)
        or context.prior_session_reference_price <= 0
    ):
        raise ValueError("persisted PRE context contains invalid prior-session price")
    calculation = calculate_volatility(
        call_model_iv=context.call_model_iv,
        put_model_iv=context.put_model_iv,
    )
    if (
        calculation.atm_iv != context.atm_iv
        or calculation.expected_absolute_return_15m
        != context.expected_absolute_return_15m
    ):
        raise ValueError("persisted PRE context does not match canonical arithmetic")
    if context.iv_source != IBKR_MODEL_OPTION_IV_SOURCE:
        raise ValueError("persisted PRE context has a non-canonical IV source")
    if {context.call_market_data_type, context.put_market_data_type} - {1, 2}:
        raise ValueError("persisted PRE context is not tick-13 market data")
    expected_observation = _option_observation_at(context.observation_session)
    if context.option_observation_at != expected_observation or not (
        expected_observation
        <= context.captured_at
        < expected_observation + PRE_CONTEXT_OBSERVATION_MINUTE
    ):
        raise ValueError("persisted PRE context has invalid 16:00 observation timing")
