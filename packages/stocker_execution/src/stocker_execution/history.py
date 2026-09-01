"""Persistent IBKR historical bars without PRE calculation assumptions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Protocol

from stocker_execution.ibkr import HistoricalBar, QualifiedInstrument

IBKR_HISTORY_SOURCE = "IBKR"


class _Stage2HistoryBoundary(Protocol):
    async def historical_bars(
        self,
        instrument: QualifiedInstrument,
        *,
        bar_size: str,
        duration: str,
        what_to_show: str,
        regular_trading_hours: bool,
        end_time: datetime | None = None,
        minimum_bars: int = 1,
    ) -> tuple[HistoricalBar, ...]: ...


class HistoryStatus(StrEnum):
    """Whether every exact bar required by a caller is present."""

    READY = "READY"
    NOT_READY = "NOT_READY"


@dataclass(frozen=True, slots=True)
class HistorySemantics:
    """Request fields that make two IBKR historical bars interchangeable."""

    bar_size: str
    what_to_show: str
    regular_trading_hours: bool

    def __post_init__(self) -> None:
        bar_size = " ".join(self.bar_size.strip().lower().split())
        what_to_show = self.what_to_show.strip().upper()
        if not bar_size or not what_to_show:
            raise ValueError("History bar size and what_to_show are required")
        object.__setattr__(self, "bar_size", bar_size)
        object.__setattr__(self, "what_to_show", what_to_show)


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    """Exact causal cached bars plus explicit completeness and lineage."""

    con_id: int
    symbol: str
    as_of: datetime
    semantics: HistorySemantics
    bars: tuple[HistoricalBar, ...]
    missing_timestamps: tuple[datetime, ...]
    status: HistoryStatus
    reason: str
    source: str = IBKR_HISTORY_SOURCE

    @property
    def history_start(self) -> datetime | None:
        """Return the first included bar timestamp."""

        if not self.bars:
            return None
        return _require_aware_datetime(self.bars[0].timestamp)

    @property
    def history_end(self) -> datetime | None:
        """Return the last included bar timestamp."""

        if not self.bars:
            return None
        return _require_aware_datetime(self.bars[-1].timestamp)


class IbkrHistoryCache:
    """One small SQLite store for bars obtained through the Stage 2 IBKR boundary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ibkr_history_bars (
                    con_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    security_type TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    primary_exchange TEXT,
                    currency TEXT NOT NULL,
                    bar_size TEXT NOT NULL,
                    what_to_show TEXT NOT NULL,
                    regular_trading_hours INTEGER NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT NOT NULL CHECK (source = 'IBKR'),
                    fetched_at_utc TEXT NOT NULL,
                    PRIMARY KEY (
                        con_id,
                        bar_size,
                        what_to_show,
                        regular_trading_hours,
                        timestamp_utc
                    )
                )
                """
            )

    def store(
        self,
        instrument: QualifiedInstrument,
        semantics: HistorySemantics,
        bars: Iterable[HistoricalBar],
        *,
        fetched_at: datetime | None = None,
    ) -> None:
        """Persist validated Stage 2 bars as IBKR-originated history."""

        if instrument.con_id <= 0:
            raise ValueError("Qualified instrument con_id must be positive")
        fetched = _to_utc(fetched_at or datetime.now(tz=UTC))
        rows = [self._storage_row(instrument, semantics, item, fetched_at=fetched) for item in bars]
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO ibkr_history_bars (
                    con_id, symbol, security_type, exchange, primary_exchange, currency,
                    bar_size, what_to_show, regular_trading_hours, timestamp_utc,
                    open, high, low, close, volume, source, fetched_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    con_id, bar_size, what_to_show, regular_trading_hours, timestamp_utc
                ) DO UPDATE SET
                    symbol = excluded.symbol,
                    security_type = excluded.security_type,
                    exchange = excluded.exchange,
                    primary_exchange = excluded.primary_exchange,
                    currency = excluded.currency,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    source = excluded.source,
                    fetched_at_utc = excluded.fetched_at_utc
                """,
                rows,
            )

    def get_required_history(
        self,
        instrument: QualifiedInstrument,
        semantics: HistorySemantics,
        required_timestamps: Sequence[datetime],
        *,
        as_of: datetime,
    ) -> HistorySnapshot:
        """Return exact cached bars or NOT_READY; never fill or substitute data."""

        if instrument.con_id <= 0:
            raise ValueError("Qualified instrument con_id must be positive")
        causal_cutoff = _to_utc(as_of)
        required = tuple(sorted({_to_utc(timestamp) for timestamp in required_timestamps}))
        if not required:
            raise ValueError("At least one required history timestamp is required")

        found: dict[datetime, HistoricalBar] = {}
        with self._connect() as connection:
            for chunk in _chunks(required, size=500):
                placeholders = ", ".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    SELECT timestamp_utc, open, high, low, close, volume
                    FROM ibkr_history_bars
                    WHERE con_id = ?
                      AND bar_size = ?
                      AND what_to_show = ?
                      AND regular_trading_hours = ?
                      AND source = ?
                      AND timestamp_utc <= ?
                      AND timestamp_utc IN ({placeholders})
                    """,  # noqa: S608 - placeholders are generated, not user supplied
                    (
                        instrument.con_id,
                        semantics.bar_size,
                        semantics.what_to_show,
                        int(semantics.regular_trading_hours),
                        IBKR_HISTORY_SOURCE,
                        _serialize_timestamp(causal_cutoff),
                        *(_serialize_timestamp(timestamp) for timestamp in chunk),
                    ),
                ).fetchall()
                for row in rows:
                    timestamp = _deserialize_timestamp(str(row["timestamp_utc"]))
                    found[timestamp] = HistoricalBar(
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )

        bars = tuple(found[timestamp] for timestamp in required if timestamp in found)
        missing = tuple(timestamp for timestamp in required if timestamp not in found)
        status = HistoryStatus.READY if not missing else HistoryStatus.NOT_READY
        reason = ""
        if missing:
            reason = (
                f"incomplete IBKR history: {len(missing)} of {len(required)} required bars missing"
            )
        return HistorySnapshot(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            as_of=causal_cutoff,
            semantics=semantics,
            bars=bars,
            missing_timestamps=missing,
            status=status,
            reason=reason,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _storage_row(
        instrument: QualifiedInstrument,
        semantics: HistorySemantics,
        bar: HistoricalBar,
        *,
        fetched_at: datetime,
    ) -> tuple[object, ...]:
        timestamp = _to_utc(_require_aware_datetime(bar.timestamp))
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if not all(isfinite(value) for value in values) or bar.volume < 0:
            raise ValueError("Historical bar contains invalid values")
        if bar.high < max(bar.open, bar.low, bar.close) or bar.low > min(
            bar.open, bar.high, bar.close
        ):
            raise ValueError("Historical bar contains inconsistent OHLC values")
        return (
            instrument.con_id,
            instrument.symbol,
            instrument.security_type,
            instrument.exchange,
            instrument.primary_exchange,
            instrument.currency,
            semantics.bar_size,
            semantics.what_to_show,
            int(semantics.regular_trading_hours),
            _serialize_timestamp(timestamp),
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            IBKR_HISTORY_SOURCE,
            _serialize_timestamp(fetched_at),
        )


class IbkrHistoryService:
    """The sole Stage 4 ingress from the Stage 2 IBKR history boundary."""

    def __init__(
        self,
        ibkr: _Stage2HistoryBoundary,
        cache: IbkrHistoryCache,
    ) -> None:
        self._ibkr = ibkr
        self._cache = cache

    async def fetch_and_store(
        self,
        instrument: QualifiedInstrument,
        *,
        bar_size: str,
        duration: str,
        what_to_show: str,
        regular_trading_hours: bool,
        end_time: datetime | None = None,
        minimum_bars: int = 1,
    ) -> tuple[HistoricalBar, ...]:
        """Request history from IBKR and persist it only after Stage 2 validation."""

        semantics = HistorySemantics(
            bar_size=bar_size,
            what_to_show=what_to_show,
            regular_trading_hours=regular_trading_hours,
        )
        bars = await self._ibkr.historical_bars(
            instrument,
            bar_size=semantics.bar_size,
            duration=duration,
            what_to_show=semantics.what_to_show,
            regular_trading_hours=semantics.regular_trading_hours,
            end_time=end_time,
            minimum_bars=minimum_bars,
        )
        self._cache.store(instrument, semantics, bars)
        return bars


def _require_aware_datetime(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Historical cache timestamps must be timezone-aware datetimes")
    return value


def _to_utc(value: datetime) -> datetime:
    return _require_aware_datetime(value).astimezone(UTC)


def _serialize_timestamp(value: datetime) -> str:
    return _to_utc(value).isoformat(timespec="microseconds")


def _deserialize_timestamp(value: str) -> datetime:
    return _to_utc(datetime.fromisoformat(value))


def _chunks(values: Sequence[datetime], *, size: int) -> Iterable[Sequence[datetime]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]
