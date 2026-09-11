"""IBKR-only inputs for the Session HARD package's frozen causal feature contract."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_core.logging import configure_logging
from stocker_core.markets import MarketDefinition
from stocker_core.runs import RunConfig
from stocker_data.calendars import get_market_calendar
from stocker_execution.expected_move import (
    ExpectedMoveResult,
    ExpectedMoveStatus,
    calculate_hv_expected_move,
)
from stocker_execution.history import (
    HistorySemantics,
    HistoryStatus,
    IbkrHistoryCache,
    IbkrHistoryService,
)
from stocker_execution.ibkr import HistoricalBar, IbkrConnection, IbkrError, QualifiedInstrument
from stocker_execution.session_hard_method import TradeEvent
from stocker_execution.session_hard_payoff import is_baseline_payoff_candidate
from stocker_execution.session_hard_structure_d import (
    CohortOpportunity,
    EntryBar,
    SessionHardAssessment,
    SignalStatus,
    StrategyContext,
    StrategyOpportunityKey,
    StrategySignal,
)
from stocker_execution.stage5 import (
    Stage5FeatureSnapshot,
    Stage5Status,
    calculate_session_hard_inputs,
)

SOURCE = "IBKR_PRIOR20_FINAL_RTH_MINUTE_CLOSES"
SEMANTICS = HistorySemantics("1 min", "TRADES", True)


class PriorSessionExpectedMoveService:
    """Original prior-20-return HV formula; tick 104 cannot substitute for it."""

    def __init__(self, broker: IbkrConnection, cache: IbkrHistoryCache) -> None:
        self.cache = cache
        self.history = IbkrHistoryService(broker, cache)
        self.prepared: dict[tuple[int, date], ExpectedMoveResult] = {}
        self._prior_closes: dict[tuple[str, date], tuple[datetime, ...]] = {}

    async def prepare_expected_move(
        self,
        instrument: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> None:
        key = instrument.con_id, session
        if key in self.prepared:
            return
        calendar_key = market.calendar, session
        if calendar_key not in self._prior_closes:
            calendar = get_market_calendar(market.calendar)
            schedule = calendar.schedule(
                start_date=session - timedelta(days=90), end_date=session - timedelta(days=1)
            ).tail(21)
            self._prior_closes[calendar_key] = tuple(
                (v - pd.Timedelta(minutes=1)).to_pydatetime() for v in schedule.market_close
            )
        required = self._prior_closes[calendar_key]
        try:
            if len(required) != 21:
                raise ValueError("21 prior exchange sessions are required")
            snapshot = self.cache.get_required_history(instrument, SEMANTICS, required, as_of=t0)
            if snapshot.status is not HistoryStatus.READY:
                for timestamp in required:
                    one = self.cache.get_required_history(
                        instrument, SEMANTICS, (timestamp,), as_of=t0
                    )
                    if one.status is not HistoryStatus.READY:
                        await self.history.fetch_and_store(
                            instrument,
                            bar_size="1 min",
                            duration="60 S",
                            what_to_show="TRADES",
                            regular_trading_hours=True,
                            end_time=timestamp + timedelta(minutes=1),
                        )
                snapshot = self.cache.get_required_history(
                    instrument, SEMANTICS, required, as_of=t0
                )
            if snapshot.status is not HistoryStatus.READY:
                raise ValueError(snapshot.reason)
            closes = pd.Series([bar.close for bar in snapshot.bars])
            # Identical frozen rolling sample-std expression, evaluated strictly on prior closes.
            hv = float(
                cast(pd.Series, np.log(closes))
                .diff()
                .rolling(20, min_periods=20)
                .std(ddof=1)
                .iloc[-1]
                * math.sqrt(252)
            )
            calculation = calculate_hv_expected_move(
                hv, market_regular_minutes=market.active_regular_minutes
            )
            result = ExpectedMoveResult(
                ExpectedMoveStatus.READY,
                calculation.expected_absolute_return_15m,
                SOURCE,
                required[-1] + timedelta(minutes=1),
                "PRIOR20_HV_V1",
                "",
                hv,
                hv,
                market.active_regular_minutes,
            )
        except ValueError as exc:
            result = ExpectedMoveResult(
                ExpectedMoveStatus.NOT_READY, None, SOURCE, None, "PRIOR20_HV_V1", str(exc)
            )
        self.prepared[key] = result

    async def get_expected_move(
        self,
        instrument: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> ExpectedMoveResult:
        return self.prepared.get(
            (instrument.con_id, session),
            ExpectedMoveResult(
                ExpectedMoveStatus.NOT_READY,
                None,
                SOURCE,
                None,
                "PRIOR20_HV_V1",
                "Required prior-session close history was not prepared",
            ),
        )


def completed_five_minute_bars(bars: Sequence[HistoricalBar]) -> tuple[HistoricalBar, ...]:
    """Original research aggregation of an already verified exact 1-minute prefix."""
    if len(bars) % 5:
        raise ValueError("Incomplete five-minute aggregation")
    return tuple(
        HistoricalBar(
            bars[i].timestamp,
            bars[i].open,
            max(b.high for b in bars[i : i + 5]),
            min(b.low for b in bars[i : i + 5]),
            bars[i + 4].close,
            sum(b.volume for b in bars[i : i + 5]),
        )
        for i in range(0, len(bars), 5)
    )


def directional_inputs(
    bars: Sequence[HistoricalBar],
    t0: datetime,
    checkpoint: int,
    *,
    active_minutes: Sequence[datetime] | None = None,
) -> dict[str, float]:
    """The four frozen T0 predictors, cut at T0-1m; no trigger-minute OHLC.

    Expressions retained from t0_direction_v0.direction_features / _symbol / _segment.
    Other research feature columns are not calculated in runtime.
    """
    prefix = pd.DataFrame(
        [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
    ).set_index("timestamp")
    expected = (
        pd.DatetimeIndex(list(active_minutes))
        if active_minutes is not None
        else pd.date_range(
            t0 - timedelta(minutes=checkpoint * 5), t0 - timedelta(minutes=1), freq="1min"
        )
    )
    if (
        len(expected) != checkpoint * 5
        or not expected.is_unique
        or not expected.is_monotonic_increasing
        or expected[-1] >= t0
        or not prefix.index.equals(expected)
    ):
        raise ValueError("Missing exact causal qualification prefix")
    typical = (prefix.high + prefix.low + prefix.close) / 3
    vwap = (typical * prefix.volume).cumsum() / prefix.volume.cumsum().replace(0, np.nan)
    five = prefix.groupby(np.arange(len(prefix)) // 5).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    body = (five.close - five.open) / (five.high - five.low).replace(0, np.nan)
    return {
        "typical_vwap_distance": float(((prefix.close / vwap - 1) * 10000).iloc[-1]),
        "return_3m": float((prefix.close.pct_change(3, fill_method=None) * 10000).iloc[-1]),
        "session_open_distance": float(((prefix.close / prefix.open.iloc[0] - 1) * 10000).iloc[-1]),
        "completed_5m_body": float(body.iloc[-1]),
    }


class IbkrSessionDataSource:
    """Supply Stage 6 causal score inputs and entry bars from the shared IBKR cache."""

    _FIVE_MINUTES = HistorySemantics("5 mins", "TRADES", True)
    _ONE_MINUTE = HistorySemantics("1 min", "TRADES", True)

    def __init__(
        self,
        ibkr: IbkrConnection,
        history_cache: IbkrHistoryCache,
        *,
        logger: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._broker = ibkr
        self._cache = history_cache
        self._history = IbkrHistoryService(ibkr, history_cache)
        self._logger = logger or configure_logging()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._prepared_trades: set[int] = set()

    async def context_for(
        self,
        run: RunConfig,
        rows: Sequence[Stage5FeatureSnapshot],
        checkpoint: int,
        instruments: Mapping[int, QualifiedInstrument],
        cohort_history: Sequence[CohortOpportunity],
    ) -> StrategyContext:
        from stocker_execution.runtime import ExchangeSessionResolver, _aware

        assessments: dict[StrategyOpportunityKey, SessionHardAssessment] = {}
        risk_features: dict[StrategyOpportunityKey, dict[str, float]] = {}
        required_history_ready: set[int] = set()
        for row in rows:
            if row.status is not Stage5Status.READY or row.con_id is None:
                continue
            instrument = instruments.get(row.con_id)
            if instrument is None:
                self._logger.warning(
                    "session_hard_input_unavailable",
                    run_id=run.run_id,
                    con_id=row.con_id,
                    symbol=row.symbol,
                    reason="qualified instrument identity missing",
                )
                continue
            t0 = _aware(row.t0)
            market_session = ExchangeSessionResolver().resolve(run, t0)
            required = market_session.active_bar_starts[:checkpoint]
            if len(required) != checkpoint:
                self._logger.warning(
                    "session_hard_input_unavailable",
                    run_id=run.run_id,
                    con_id=row.con_id,
                    symbol=row.symbol,
                    reason="active trading-bar prefix unavailable",
                )
                continue
            session_open = required[0]
            try:
                semantics = self._ONE_MINUTE if run.method_spec else self._FIVE_MINUTES
                history_required = (
                    tuple(start + timedelta(minutes=i) for start in required for i in range(5))
                    if run.method_spec
                    else required
                )
                snapshot = self._cache.get_required_history(
                    instrument, semantics, history_required, as_of=t0
                )
                if snapshot.status is not HistoryStatus.READY:
                    await self._history.fetch_and_store(
                        instrument,
                        bar_size=semantics.bar_size,
                        duration=f"{int((t0 - session_open).total_seconds()) + 300} S",
                        what_to_show="TRADES",
                        regular_trading_hours=True,
                        end_time=t0,
                    )
                    snapshot = self._cache.get_required_history(
                        instrument, semantics, history_required, as_of=t0
                    )
                if snapshot.status is not HistoryStatus.READY:
                    self._logger.debug(
                        "session_hard_input_unavailable",
                        run_id=run.run_id,
                        con_id=instrument.con_id,
                        symbol=instrument.symbol,
                        reason=snapshot.reason,
                    )
                    continue
                features = calculate_session_hard_inputs(
                    completed_five_minute_bars(snapshot.bars) if run.method_spec else snapshot.bars,
                    checkpoint=checkpoint,
                    session_open=session_open,
                    bar_starts=required,
                )
                key = StrategyOpportunityKey(instrument.con_id, row.session, t0)
                assessments[key] = SessionHardAssessment.from_features(
                    checkpoint=checkpoint, features=features
                )
                if run.method_spec is not None:
                    if row.expected_move_source != SOURCE:
                        raise ValueError("Frozen MODEL_T0 requires prior20 HV, not tick104")
                    one_required = history_required
                    one = self._cache.get_required_history(
                        instrument, self._ONE_MINUTE, one_required, as_of=t0
                    )
                    if one.status is not HistoryStatus.READY:
                        await self._history.fetch_and_store(
                            instrument,
                            bar_size="1 min",
                            duration=f"{int((t0 - session_open).total_seconds())} S",
                            what_to_show="TRADES",
                            regular_trading_hours=True,
                            end_time=t0,
                        )
                        one = self._cache.get_required_history(
                            instrument, self._ONE_MINUTE, one_required, as_of=t0
                        )
                    if one.status is not HistoryStatus.READY:
                        raise ValueError("MODEL_T0 exact 1-minute prefix unavailable")
                    from stocker_core.methods import ARTIFACTS

                    columns = json.loads((ARTIFACTS / "MODEL_T0_parameters.json").read_text())[
                        "input_columns"
                    ]
                    assert row.m_price is not None and row.p0 is not None
                    assert row.pre_move_m is not None and row.historical_volatility is not None
                    inputs = {
                        **features,
                        **directional_inputs(one.bars, t0, checkpoint, active_minutes=one_required),
                        "score": assessments[key].score,
                        "PRE_MOVE_M": row.pre_move_m,
                        "cohort_percentile": float("nan"),
                        "M_over_P0_bps": row.m_price / row.p0 * 10000,
                        "hv": row.historical_volatility,
                    }
                    risk_features[key] = {name: inputs[name] for name in columns}
                required_history_ready.add(instrument.con_id)

            except (IbkrError, ValueError) as exc:
                self._logger.warning(
                    "session_hard_input_failed",
                    run_id=run.run_id,
                    con_id=instrument.con_id,
                    symbol=instrument.symbol,
                    reason=str(exc),
                )
                continue
        return StrategyContext(
            run_id=run.run_id,
            session_hard=assessments,
            cohort_history=tuple(cohort_history),
            whipsaw_features=risk_features,
            available_at=self._clock(),
            required_history_ready=frozenset(required_history_ready),
        )

    def prepare_trades(self, instrument: QualifiedInstrument) -> None:
        self._broker.prepare_trade_events(instrument)
        self._prepared_trades.add(instrument.con_id)

    def trade_stream_status(self, instrument: QualifiedInstrument, *, t0: datetime) -> str:
        return self._broker.trade_stream_status(instrument, t0=t0)

    def release_trades(self, con_id: int) -> None:
        self._broker.release_trade_events(con_id)
        self._prepared_trades.discard(con_id)

    def release_unused_trades(self, retained: set[int]) -> None:
        for con_id in self._prepared_trades - retained:
            self.release_trades(con_id)

    async def cohort_bars(
        self, instrument: QualifiedInstrument, signal: StrategySignal, now: datetime
    ) -> tuple[EntryBar, ...]:
        required = tuple(signal.t0 + timedelta(minutes=i) for i in range(5))
        snapshot = self._cache.get_required_history(
            instrument, self._ONE_MINUTE, required, as_of=now
        )
        if snapshot.status is not HistoryStatus.READY:
            await self._history.fetch_and_store(
                instrument,
                bar_size="1 min",
                duration="300 S",
                what_to_show="TRADES",
                regular_trading_hours=True,
                end_time=signal.t0 + timedelta(minutes=5),
            )
            snapshot = self._cache.get_required_history(
                instrument, self._ONE_MINUTE, required, as_of=now
            )
        if snapshot.status is not HistoryStatus.READY:
            raise ValueError(snapshot.reason)
        return tuple(
            EntryBar(cast(datetime, b.timestamp), b.open, b.high, b.low, b.close)
            for b in snapshot.bars
        )

    async def trades_for(
        self, instruments: Mapping[int, QualifiedInstrument], signals: Sequence[StrategySignal]
    ) -> dict[int, tuple[TradeEvent, ...]]:
        self.trade_errors: dict[int, str] = {}
        result = {}
        for con_id in {s.underlying_con_id for s in signals if s.underlying_con_id is not None}:
            if con_id not in instruments:
                continue
            try:
                result[con_id] = self._broker.trade_events(
                    instruments[con_id],
                    t0=min(s.t0 for s in signals if s.underlying_con_id == con_id),
                )
            except IbkrError as exc:
                self.trade_errors[con_id] = str(exc)
                self._logger.warning(
                    "candidate_trade_prefix_unavailable", con_id=con_id, reason=str(exc)
                )
        return result

    async def bars_for(
        self,
        run: RunConfig,
        instruments: Mapping[int, QualifiedInstrument],
        *,
        session: date,
        now: datetime,
        signals: Sequence[StrategySignal],
        fetch_missing: bool = True,
    ) -> Mapping[int, Sequence[EntryBar]]:
        from stocker_execution.runtime import _aware, _intraday_timestamp

        causal_now = _aware(now)
        completed_minute = causal_now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        required_by_con_id: dict[tuple[int, date], set[datetime]] = {}
        for signal in signals:
            shadow = signal.baseline_eligible and is_baseline_payoff_candidate(signal)
            if (
                signal.run_id != run.run_id
                or signal.underlying_con_id is None
                or (
                    not shadow
                    and (
                        signal.session != session
                        or signal.status is not SignalStatus.WAITING_FOR_ENTRY
                    )
                )
            ):
                continue
            start = (
                _aware(signal.entry_timestamp)
                if shadow and signal.entry_timestamp
                else _aware(signal.t0)
            )
            if not shadow and causal_now > start + timedelta(minutes=5):
                self._logger.info(
                    "entry_window_not_replayed",
                    run_id=run.run_id,
                    signal_id=signal.signal_id,
                    con_id=signal.underlying_con_id,
                )
                continue
            horizon_end = (
                _aware(signal.t0) + timedelta(minutes=14)
                if shadow
                else start + timedelta(minutes=4)
            )
            end = min(horizon_end, completed_minute)
            if end < start:
                continue
            required_by_con_id.setdefault((signal.underlying_con_id, signal.session), set()).update(
                start + timedelta(minutes=index)
                for index in range(int((end - start).total_seconds() // 60) + 1)
            )

        result: dict[int, tuple[EntryBar, ...]] = {}
        for (con_id, _session), required_set in required_by_con_id.items():
            instrument = instruments.get(con_id)
            if instrument is None:
                self._logger.warning(
                    "entry_data_unavailable",
                    run_id=run.run_id,
                    con_id=con_id,
                    reason="qualified instrument identity missing",
                )
                continue
            required = tuple(sorted(required_set))
            try:
                snapshot = self._cache.get_required_history(
                    instrument, self._ONE_MINUTE, required, as_of=causal_now
                )
                if snapshot.status is not HistoryStatus.READY and fetch_missing:
                    duration_seconds = max(
                        900, int((required[-1] - required[0]).total_seconds()) + 60
                    )
                    await self._history.fetch_and_store(
                        instrument,
                        bar_size="1 min",
                        duration=f"{duration_seconds} S",
                        what_to_show="TRADES",
                        regular_trading_hours=True,
                        end_time=min(causal_now, required[-1] + timedelta(minutes=1)),
                    )
                    snapshot = self._cache.get_required_history(
                        instrument, self._ONE_MINUTE, required, as_of=causal_now
                    )
                if snapshot.status is not HistoryStatus.READY:
                    self._logger.debug(
                        "entry_data_incomplete",
                        run_id=run.run_id,
                        con_id=instrument.con_id,
                        symbol=instrument.symbol,
                        reason=snapshot.reason,
                    )
                result[con_id] = (
                    *result.get(con_id, ()),
                    *tuple(
                        EntryBar(
                            timestamp=_intraday_timestamp(bar.timestamp),
                            open=bar.open,
                            high=bar.high,
                            low=bar.low,
                            close=bar.close,
                        )
                        for bar in snapshot.bars
                    ),
                )
            except (IbkrError, ValueError) as exc:
                self._logger.warning(
                    "entry_data_failed",
                    run_id=run.run_id,
                    con_id=instrument.con_id,
                    symbol=instrument.symbol,
                    reason=str(exc),
                )
                continue
        return result
