"""Causal, non-trading Stage 5 current-session PRE feature production."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from math import isfinite, log1p
from pathlib import Path
from typing import Protocol, TypedDict

from stocker_core.markets import market_for_instrument
from stocker_core.runs import CandidateScreen, RunInstance, RunState
from stocker_core.universes import InstrumentReference
from stocker_execution.activity_shortlist import (
    ActivityShortlistSnapshot,
    ActivityShortlistStatus,
)
from stocker_execution.expected_move import ExpectedMoveService, ExpectedMoveStatus
from stocker_execution.history import (
    HistorySemantics,
    HistorySnapshot,
    HistoryStatus,
    IbkrHistoryCache,
    IbkrHistoryService,
)
from stocker_execution.ibkr import (
    HISTORICAL_REQUEST_CONCURRENCY,
    HistoricalBar,
    IbkrConnection,
    IbkrError,
    QualifiedInstrument,
)

STAGE5_CALCULATION_VERSION = "STAGE5_PRE_MOVE_HV_V1"
STAGE5_HV_CALCULATION_VERSION = "STAGE5_PRE_MOVE_HV_V1"
STAGE5_FIVE_MINUTE_HISTORY = HistorySemantics("5 mins", "TRADES", True)
STAGE5_ONE_MINUTE_HISTORY = HistorySemantics("1 min", "TRADES", True)


class Stage5Status(StrEnum):
    READY = "READY"
    INELIGIBLE = "INELIGIBLE"
    PRE_CONTEXT_NOT_READY = "PRE_CONTEXT_NOT_READY"
    PRE_MOVE_NOT_READY = "PRE_MOVE_NOT_READY"


@dataclass(frozen=True, slots=True)
class PreMoveCalculation:
    """Auditable result of the frozen current-session PRE_MOVE arithmetic."""

    p0: float
    expected_absolute_return_15m: float
    m_price: float
    alignment_factor: float
    aligned_pre_open: float
    raw_pre_move_price: float
    pre_move_m: float


@dataclass(frozen=True, slots=True)
class Stage5FeatureResult:
    """One canonical conId/checkpoint feature outcome."""

    con_id: int
    symbol: str
    session: date
    t0: datetime
    status: Stage5Status
    exclusion_reason: str
    p0: float | None = None
    expected_absolute_return_15m: float | None = None
    m_price: float | None = None
    raw_open_t0_minus_3m: float | None = None
    raw_open_t0: float | None = None
    alignment_factor: float | None = None
    aligned_pre_open: float | None = None
    raw_pre_move_price: float | None = None
    pre_move_m: float | None = None
    calculation_version: str = STAGE5_CALCULATION_VERSION
    expected_move_source: str | None = None
    expected_move_observation_at: datetime | None = None
    expected_move_calculation_version: str | None = None
    raw_historical_volatility: float | None = None
    historical_volatility: float | None = None
    market_regular_minutes: int | None = None


@dataclass(frozen=True, slots=True)
class Stage5Membership:
    """One active run/universe association for a qualified conId."""

    run_id: str
    universe_id: str

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.universe_id.strip():
            raise ValueError("Stage 5 run_id and universe_id are required")


@dataclass(frozen=True, slots=True)
class Stage5QualifiedRequest:
    """Stage 2 identity plus every active membership that requires Stage 5 work."""

    instrument: QualifiedInstrument
    memberships: tuple[Stage5Membership, ...]

    def __post_init__(self) -> None:
        if self.instrument.con_id <= 0:
            raise ValueError("Stage 5 requires a positive qualified conId")
        if not self.memberships:
            raise ValueError("Stage 5 requires at least one active membership")


@dataclass(frozen=True, slots=True)
class Stage5IneligibleInstrument:
    """One qualification failure isolated from the remaining active population."""

    symbol: str
    memberships: tuple[Stage5Membership, ...]
    reason: str
    status: Stage5Status = Stage5Status.INELIGIBLE


@dataclass(frozen=True, slots=True)
class Stage5QualificationResult:
    requests: tuple[Stage5QualifiedRequest, ...]
    ineligible: tuple[Stage5IneligibleInstrument, ...]


@dataclass(frozen=True, slots=True)
class Stage5FeatureSnapshot:
    """One deterministic reusable feature row projected to a universe."""

    run_ids: tuple[str, ...]
    universe_id: str
    con_id: int | None
    symbol: str
    session: date
    t0: datetime
    status: Stage5Status
    exclusion_reason: str
    p0: float | None
    expected_absolute_return_15m: float | None
    m_price: float | None
    raw_open_t0_minus_3m: float | None
    raw_open_t0: float | None
    alignment_factor: float | None
    aligned_pre_open: float | None
    raw_pre_move_price: float | None
    pre_move_m: float | None
    calculation_version: str
    expected_move_source: str | None = None
    expected_move_observation_at: datetime | None = None
    expected_move_calculation_version: str | None = None
    raw_historical_volatility: float | None = None
    historical_volatility: float | None = None
    market_regular_minutes: int | None = None


class _FeatureService(Protocol):
    async def get_feature(
        self, instrument: QualifiedInstrument, *, session: date, t0: datetime
    ) -> Stage5FeatureResult: ...


class _Stage5Lineage(TypedDict):
    calculation_version: str
    expected_move_source: str | None
    expected_move_observation_at: datetime | None
    expected_move_calculation_version: str | None
    raw_historical_volatility: float | None
    historical_volatility: float | None
    market_regular_minutes: int | None


class Stage5CurrentDataService:
    """Load Stage 4 context and the minimum exact current-session IBKR bars."""

    def __init__(
        self,
        ibkr: IbkrConnection,
        history_cache: IbkrHistoryCache,
        expected_move_service: ExpectedMoveService,
        *,
        calculation_version: str = STAGE5_CALCULATION_VERSION,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ibkr, IbkrConnection):
            raise TypeError("Stage 5 current-session data requires the Stage 2 IbkrConnection")
        self._history_cache = history_cache
        self._history_service = IbkrHistoryService(ibkr, history_cache)
        self._expected_move_service = expected_move_service
        self._calculation_version = calculation_version
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    async def prepare_expected_moves(
        self,
        requests: Sequence[Stage5QualifiedRequest],
        *,
        session: date,
        t0: datetime,
    ) -> None:
        """Capture source-specific observations before the causal T0 cutoff."""

        pending = iter(requests)

        async def prepare_next() -> None:
            for request in pending:
                # Isolate each symbol, and yield even when its history is already cached.
                await asyncio.gather(
                    self._prepare_expected_move(request, session=session, t0=t0),
                    return_exceptions=True,
                )

        await asyncio.gather(
            *(prepare_next() for _ in range(min(len(requests), HISTORICAL_REQUEST_CONCURRENCY)))
        )

    async def _prepare_expected_move(
        self,
        request: Stage5QualifiedRequest,
        *,
        session: date,
        t0: datetime,
    ) -> None:
        instrument = request.instrument
        await self._expected_move_service.prepare_expected_move(
            instrument,
            market=market_for_instrument(
                primary_exchange=instrument.primary_exchange,
                exchange=instrument.exchange,
                currency=instrument.currency,
            ),
            session=session,
            t0=t0,
        )

    async def get_feature(
        self, instrument: QualifiedInstrument, *, session: date, t0: datetime
    ) -> Stage5FeatureResult:
        signal_timestamp = _aware_utc(t0)
        now = _aware_utc(self._clock())
        if now < signal_timestamp:
            return Stage5FeatureResult(
                con_id=instrument.con_id,
                symbol=instrument.symbol,
                session=session,
                t0=signal_timestamp,
                status=Stage5Status.PRE_MOVE_NOT_READY,
                exclusion_reason="PRE_MOVE_NOT_READY: T0 opening print is not yet causal",
                calculation_version=self._calculation_version,
            )

        expected_move = await self._expected_move_service.get_expected_move(
            instrument,
            market=market_for_instrument(
                primary_exchange=instrument.primary_exchange,
                exchange=instrument.exchange,
                currency=instrument.currency,
            ),
            session=session,
            t0=signal_timestamp,
        )
        lineage: _Stage5Lineage = {
            "calculation_version": self._calculation_version,
            "expected_move_source": expected_move.source,
            "expected_move_observation_at": expected_move.observation_timestamp,
            "expected_move_calculation_version": expected_move.calculation_version,
            "raw_historical_volatility": expected_move.raw_historical_volatility,
            "historical_volatility": expected_move.historical_volatility,
            "market_regular_minutes": expected_move.market_regular_minutes,
        }
        if (
            expected_move.status is not ExpectedMoveStatus.READY
            or expected_move.expected_absolute_return_15m is None
        ):
            return Stage5FeatureResult(
                con_id=instrument.con_id,
                symbol=instrument.symbol,
                session=session,
                t0=signal_timestamp,
                status=Stage5Status.PRE_CONTEXT_NOT_READY,
                exclusion_reason=expected_move.reason,
                **lineage,
            )

        five_required = (signal_timestamp,)
        one_required = (signal_timestamp - timedelta(minutes=3), signal_timestamp)
        try:
            five = await self._required_history(
                instrument,
                STAGE5_FIVE_MINUTE_HISTORY,
                five_required,
                as_of=signal_timestamp,
                duration="600 S",
                end_time=min(now, signal_timestamp + timedelta(minutes=5)),
            )
            one = await self._required_history(
                instrument,
                STAGE5_ONE_MINUTE_HISTORY,
                one_required,
                as_of=signal_timestamp,
                duration="300 S",
                end_time=min(now, signal_timestamp + timedelta(minutes=1)),
            )
        except (IbkrError, ValueError) as exc:
            return Stage5FeatureResult(
                con_id=instrument.con_id,
                symbol=instrument.symbol,
                session=session,
                t0=signal_timestamp,
                status=Stage5Status.PRE_MOVE_NOT_READY,
                exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
                expected_absolute_return_15m=expected_move.expected_absolute_return_15m,
                **lineage,
            )
        return calculate_stage5_feature(
            instrument=instrument,
            session=session,
            t0=signal_timestamp,
            expected_absolute_return_15m=expected_move.expected_absolute_return_15m,
            five_minute_bars=five.bars,
            one_minute_bars=one.bars,
            **lineage,
        )

    async def _required_history(
        self,
        instrument: QualifiedInstrument,
        semantics: HistorySemantics,
        required_timestamps: Sequence[datetime],
        *,
        as_of: datetime,
        duration: str,
        end_time: datetime,
    ) -> HistorySnapshot:
        snapshot = self._history_cache.get_required_history(
            instrument, semantics, required_timestamps, as_of=as_of
        )
        if snapshot.status is HistoryStatus.READY:
            return snapshot
        await self._history_service.fetch_and_store(
            instrument,
            bar_size=semantics.bar_size,
            duration=duration,
            what_to_show=semantics.what_to_show,
            regular_trading_hours=semantics.regular_trading_hours,
            end_time=end_time,
        )
        return self._history_cache.get_required_history(
            instrument, semantics, required_timestamps, as_of=as_of
        )


class Stage5SnapshotStore:
    """Small SQLite audit store for reusable Stage 5 feature snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stage5_feature_snapshots (
                    universe_id TEXT NOT NULL,
                    run_ids_json TEXT NOT NULL,
                    con_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    session TEXT NOT NULL,
                    t0_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    exclusion_reason TEXT NOT NULL,
                    p0 REAL,
                    expected_absolute_return_15m REAL,
                    m_price REAL,
                    raw_open_t0_minus_3m REAL,
                    raw_open_t0 REAL,
                    alignment_factor REAL,
                    aligned_pre_open REAL,
                    raw_pre_move_price REAL,
                    pre_move_m REAL,
                    calculation_version TEXT NOT NULL,
                    expected_move_source TEXT,
                    expected_move_observation_at_utc TEXT,
                    expected_move_calculation_version TEXT,
                    raw_historical_volatility REAL,
                    historical_volatility REAL,
                    market_regular_minutes INTEGER,
                    PRIMARY KEY (universe_id, t0_utc, con_id, calculation_version)
                )
                """
            )
            existing_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(stage5_feature_snapshots)")
            }
            for name, data_type in (
                ("expected_move_source", "TEXT"),
                ("expected_move_observation_at_utc", "TEXT"),
                ("expected_move_calculation_version", "TEXT"),
                ("raw_historical_volatility", "REAL"),
                ("historical_volatility", "REAL"),
                ("market_regular_minutes", "INTEGER"),
            ):
                if name not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE stage5_feature_snapshots ADD COLUMN {name} {data_type}"
                    )

    def save(self, snapshot: Stage5FeatureSnapshot) -> None:
        """Persist one qualified row; a transient rerun cannot replace a READY row."""

        self.save_many((snapshot,))

    def save_many(self, snapshots: Sequence[Stage5FeatureSnapshot]) -> None:
        """Commit a checkpoint together, retaining the existing READY-row protection."""

        if not snapshots:
            return
        if any(snapshot.con_id is None for snapshot in snapshots):
            raise ValueError("cannot persist an unqualified Stage 5 row without conId")

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO stage5_feature_snapshots (
                    universe_id, run_ids_json, con_id, symbol, session, t0_utc,
                    status, exclusion_reason, p0, expected_absolute_return_15m, m_price,
                    raw_open_t0_minus_3m, raw_open_t0, alignment_factor, aligned_pre_open,
                    raw_pre_move_price, pre_move_m, calculation_version, expected_move_source,
                    expected_move_observation_at_utc, expected_move_calculation_version,
                    raw_historical_volatility, historical_volatility, market_regular_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (universe_id, t0_utc, con_id, calculation_version)
                DO UPDATE SET
                    run_ids_json = excluded.run_ids_json,
                    symbol = excluded.symbol,
                    session = excluded.session,
                    status = excluded.status,
                    exclusion_reason = excluded.exclusion_reason,
                    p0 = excluded.p0,
                    expected_absolute_return_15m = excluded.expected_absolute_return_15m,
                    m_price = excluded.m_price,
                    raw_open_t0_minus_3m = excluded.raw_open_t0_minus_3m,
                    raw_open_t0 = excluded.raw_open_t0,
                    alignment_factor = excluded.alignment_factor,
                    aligned_pre_open = excluded.aligned_pre_open,
                    raw_pre_move_price = excluded.raw_pre_move_price,
                    pre_move_m = excluded.pre_move_m,
                    expected_move_source = excluded.expected_move_source,
                    expected_move_observation_at_utc = excluded.expected_move_observation_at_utc,
                    expected_move_calculation_version = excluded.expected_move_calculation_version,
                    raw_historical_volatility = excluded.raw_historical_volatility,
                    historical_volatility = excluded.historical_volatility,
                    market_regular_minutes = excluded.market_regular_minutes
                WHERE stage5_feature_snapshots.status != ?
                  AND excluded.status = ?
                """,
                (
                    (
                        snapshot.universe_id,
                        json.dumps(snapshot.run_ids, separators=(",", ":")),
                        snapshot.con_id,
                        snapshot.symbol,
                        snapshot.session.isoformat(),
                        _aware_utc(snapshot.t0).isoformat(timespec="microseconds"),
                        snapshot.status.value,
                        snapshot.exclusion_reason,
                        snapshot.p0,
                        snapshot.expected_absolute_return_15m,
                        snapshot.m_price,
                        snapshot.raw_open_t0_minus_3m,
                        snapshot.raw_open_t0,
                        snapshot.alignment_factor,
                        snapshot.aligned_pre_open,
                        snapshot.raw_pre_move_price,
                        snapshot.pre_move_m,
                        snapshot.calculation_version,
                        snapshot.expected_move_source,
                        (
                            _aware_utc(snapshot.expected_move_observation_at).isoformat(
                                timespec="microseconds"
                            )
                            if snapshot.expected_move_observation_at is not None
                            else None
                        ),
                        snapshot.expected_move_calculation_version,
                        snapshot.raw_historical_volatility,
                        snapshot.historical_volatility,
                        snapshot.market_regular_minutes,
                        Stage5Status.READY.value,
                        Stage5Status.READY.value,
                    )
                    for snapshot in snapshots
                ),
            )

    def get(
        self,
        universe_id: str,
        t0: datetime,
        con_id: int,
        *,
        calculation_version: str = STAGE5_CALCULATION_VERSION,
    ) -> Stage5FeatureSnapshot | None:
        """Load one persisted generic feature snapshot by its stable audit key."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM stage5_feature_snapshots
                WHERE universe_id = ?
                  AND t0_utc = ?
                  AND con_id = ?
                  AND calculation_version = ?
                """,
                (
                    universe_id,
                    _aware_utc(t0).isoformat(timespec="microseconds"),
                    con_id,
                    calculation_version,
                ),
            ).fetchone()
        if row is None:
            return None
        run_ids = json.loads(str(row["run_ids_json"]))
        if not isinstance(run_ids, list) or not all(isinstance(item, str) for item in run_ids):
            raise ValueError("invalid run_ids_json in Stage 5 feature snapshot")
        return Stage5FeatureSnapshot(
            run_ids=tuple(run_ids),
            universe_id=str(row["universe_id"]),
            con_id=int(row["con_id"]),
            symbol=str(row["symbol"]),
            session=date.fromisoformat(str(row["session"])),
            t0=datetime.fromisoformat(str(row["t0_utc"])),
            status=Stage5Status(str(row["status"])),
            exclusion_reason=str(row["exclusion_reason"]),
            p0=_optional_float(row["p0"]),
            expected_absolute_return_15m=_optional_float(row["expected_absolute_return_15m"]),
            m_price=_optional_float(row["m_price"]),
            raw_open_t0_minus_3m=_optional_float(row["raw_open_t0_minus_3m"]),
            raw_open_t0=_optional_float(row["raw_open_t0"]),
            alignment_factor=_optional_float(row["alignment_factor"]),
            aligned_pre_open=_optional_float(row["aligned_pre_open"]),
            raw_pre_move_price=_optional_float(row["raw_pre_move_price"]),
            pre_move_m=_optional_float(row["pre_move_m"]),
            calculation_version=str(row["calculation_version"]),
            expected_move_source=_optional_str(row["expected_move_source"]),
            expected_move_observation_at=_optional_datetime(
                row["expected_move_observation_at_utc"]
            ),
            expected_move_calculation_version=_optional_str(
                row["expected_move_calculation_version"]
            ),
            raw_historical_volatility=_optional_float(row["raw_historical_volatility"]),
            historical_volatility=_optional_float(row["historical_volatility"]),
            market_regular_minutes=_optional_int(row["market_regular_minutes"]),
        )

    def list_snapshots(
        self,
        *,
        run_id: str | None = None,
        universe_id: str | None = None,
        session: date | None = None,
        checkpoint: datetime | None = None,
        latest_checkpoint: bool = False,
        status: Stage5Status | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[tuple[Stage5FeatureSnapshot, ...], int]:
        """Page persisted feature rows without interpreting their values."""

        if not 1 <= limit <= 500:
            raise ValueError("Stage 5 snapshot limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("Stage 5 snapshot offset cannot be negative")
        clauses: list[str] = []
        values: list[object] = []
        if run_id is not None:
            clauses.append("EXISTS (SELECT 1 FROM json_each(run_ids_json) WHERE value = ?)")
            values.append(run_id)
        if universe_id is not None:
            clauses.append("universe_id = ?")
            values.append(universe_id)
        if session is not None:
            clauses.append("session = ?")
            values.append(session.isoformat())
        if checkpoint is not None:
            clauses.append("t0_utc = ?")
            values.append(checkpoint.isoformat(timespec="microseconds"))
        if status is not None:
            clauses.append("status = ?")
            values.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            if latest_checkpoint:
                latest = connection.execute(
                    f"SELECT MAX(t0_utc) FROM stage5_feature_snapshots {where}", values
                ).fetchone()[0]
                clauses.append("t0_utc = ?")
                values.append(latest)
                where = f"WHERE {' AND '.join(clauses)}"
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM stage5_feature_snapshots {where}", values
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM stage5_feature_snapshots {where}
                ORDER BY t0_utc DESC, symbol, con_id LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
        snapshots: list[Stage5FeatureSnapshot] = []
        for row in rows:
            run_ids = json.loads(str(row["run_ids_json"]))
            if not isinstance(run_ids, list) or not all(isinstance(item, str) for item in run_ids):
                raise ValueError("invalid run_ids_json in Stage 5 feature snapshot")
            snapshots.append(
                Stage5FeatureSnapshot(
                    run_ids=tuple(run_ids),
                    universe_id=str(row["universe_id"]),
                    con_id=int(row["con_id"]),
                    symbol=str(row["symbol"]),
                    session=date.fromisoformat(str(row["session"])),
                    t0=datetime.fromisoformat(str(row["t0_utc"])),
                    status=Stage5Status(str(row["status"])),
                    exclusion_reason=str(row["exclusion_reason"]),
                    p0=_optional_float(row["p0"]),
                    expected_absolute_return_15m=_optional_float(
                        row["expected_absolute_return_15m"]
                    ),
                    m_price=_optional_float(row["m_price"]),
                    raw_open_t0_minus_3m=_optional_float(row["raw_open_t0_minus_3m"]),
                    raw_open_t0=_optional_float(row["raw_open_t0"]),
                    alignment_factor=_optional_float(row["alignment_factor"]),
                    aligned_pre_open=_optional_float(row["aligned_pre_open"]),
                    raw_pre_move_price=_optional_float(row["raw_pre_move_price"]),
                    pre_move_m=_optional_float(row["pre_move_m"]),
                    calculation_version=str(row["calculation_version"]),
                    expected_move_source=_optional_str(row["expected_move_source"]),
                    expected_move_observation_at=_optional_datetime(
                        row["expected_move_observation_at_utc"]
                    ),
                    expected_move_calculation_version=_optional_str(
                        row["expected_move_calculation_version"]
                    ),
                    raw_historical_volatility=_optional_float(row["raw_historical_volatility"]),
                    historical_volatility=_optional_float(row["historical_volatility"]),
                    market_regular_minutes=_optional_int(row["market_regular_minutes"]),
                )
            )
        return tuple(snapshots), total

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


async def qualify_active_runs(
    ibkr: IbkrConnection,
    runs: Sequence[RunInstance],
    *,
    activity_snapshots: Mapping[str, ActivityShortlistSnapshot] | None = None,
    candidate_identities: Mapping[str, Sequence[QualifiedInstrument]] | None = None,
) -> Stage5QualificationResult:
    """Screen active memberships, qualify them, then deduplicate physical stocks by conId."""

    if not isinstance(ibkr, IbkrConnection):
        raise TypeError("Stage 5 qualification requires the Stage 2 IbkrConnection")
    activity_snapshots = activity_snapshots or {}
    candidate_identities = candidate_identities or {}
    qualified: dict[int, tuple[QualifiedInstrument, set[Stage5Membership]]] = {}
    screened_symbols_by_run: dict[str, frozenset[str]] = {}
    ineligible: list[Stage5IneligibleInstrument] = []
    screened_runs = [
        run
        for run in runs
        if run.state is RunState.ACTIVE
        and run.config.screen is not None
        and run.config.screen.method is CandidateScreen.HOT_BY_VOLUME
    ]
    if screened_runs:
        max_results = max(
            run.config.screen.max_results for run in screened_runs if run.config.screen is not None
        )
        try:
            ranked_symbols = await ibkr.hot_us_stocks_by_volume(max_results=max_results)
        except IbkrError as exc:
            for run in screened_runs:
                screened_symbols_by_run[run.config.run_id] = frozenset()
                ineligible.append(
                    Stage5IneligibleInstrument(
                        "HOT_BY_VOLUME",
                        (Stage5Membership(run.config.run_id, run.universe.universe_id),),
                        f"candidate screen HOT_BY_VOLUME failed: {exc}",
                    )
                )
        else:
            for run in screened_runs:
                screen = run.config.screen
                assert screen is not None
                selected_symbols = frozenset(ranked_symbols[: screen.max_results])
                screened_symbols_by_run[run.config.run_id] = selected_symbols
                if not any(
                    reference.symbol in selected_symbols for reference in run.universe.members
                ):
                    ineligible.append(
                        Stage5IneligibleInstrument(
                            "HOT_BY_VOLUME",
                            (Stage5Membership(run.config.run_id, run.universe.universe_id),),
                            "candidate screen HOT_BY_VOLUME returned no "
                            f"{run.universe.universe_id} universe members",
                        )
                    )

    references: dict[InstrumentReference, set[Stage5Membership]] = {}
    for run in runs:
        if run.state is not RunState.ACTIVE:
            continue
        membership = Stage5Membership(run.config.run_id, run.universe.universe_id)
        if run.config.run_id in candidate_identities:
            for instrument in candidate_identities[run.config.run_id]:
                if instrument.con_id <= 0 or instrument.security_type != "STK":
                    ineligible.append(Stage5IneligibleInstrument(
                        instrument.symbol, (membership,), "INVALID_CONTRACT",
                    ))
                    continue
                existing = qualified.get(instrument.con_id)
                if existing is None:
                    qualified[instrument.con_id] = (instrument, {membership})
                else:
                    existing[1].add(membership)
            continue
        if run.config.uses_activity_shortlist:
            snapshot = activity_snapshots.get(run.config.run_id)
            if snapshot is None or snapshot.status is not ActivityShortlistStatus.READY:
                reason = (
                    snapshot.reason or snapshot.status.value
                    if snapshot is not None
                    else "ACTIVITY_SHORTLIST_NOT_READY"
                )
                ineligible.append(
                    Stage5IneligibleInstrument(
                        run.config.activity_profile_id,
                        (membership,),
                        reason,
                    )
                )
                continue
            for candidate in snapshot.candidates:
                if not candidate.selected:
                    continue
                reference = InstrumentReference(
                    symbol=candidate.symbol,
                    exchange=candidate.exchange or "SMART",
                    primary_exchange=candidate.primary_exchange,
                    currency=candidate.currency,
                )
                references.setdefault(reference, set()).add(membership)
            continue
        allowed_symbols = screened_symbols_by_run.get(run.config.run_id)
        for reference in run.universe.members:
            if allowed_symbols is not None and reference.symbol not in allowed_symbols:
                continue
            references.setdefault(reference, set()).add(membership)

    reference_order = sorted(
        references,
        key=lambda item: (
            item.symbol,
            item.exchange,
            item.primary_exchange or "",
            item.currency,
            item.security_type,
        ),
    )
    for reference in reference_order:
        memberships = references[reference]
        if reference.security_type != "STK":
            ineligible.append(
                Stage5IneligibleInstrument(
                    reference.symbol,
                    tuple(sorted(memberships, key=lambda item: (item.universe_id, item.run_id))),
                    f"unsupported security type: {reference.security_type}",
                )
            )
            continue
        try:
            instrument = await ibkr.resolve_stock(
                reference.symbol,
                exchange=reference.exchange,
                primary_exchange=reference.primary_exchange,
                currency=reference.currency,
            )
        except IbkrError as exc:
            ineligible.append(
                Stage5IneligibleInstrument(
                    reference.symbol,
                    tuple(sorted(memberships, key=lambda item: (item.universe_id, item.run_id))),
                    str(exc),
                )
            )
            continue
        existing = qualified.get(instrument.con_id)
        if existing is None:
            qualified[instrument.con_id] = (instrument, set(memberships))
        else:
            existing[1].update(memberships)

    requests = tuple(
        Stage5QualifiedRequest(
            instrument,
            tuple(sorted(memberships, key=lambda item: (item.universe_id, item.run_id))),
        )
        for _con_id, (instrument, memberships) in sorted(qualified.items())
    )
    failures = tuple(
        sorted(
            ineligible,
            key=lambda item: (
                item.symbol,
                tuple((member.universe_id, member.run_id) for member in item.memberships),
            ),
        )
    )
    return Stage5QualificationResult(requests, failures)


class Stage5Analyzer:
    """Deduplicate conId feature work and project results to active universe memberships."""

    def __init__(
        self,
        feature_service: _FeatureService,
        *,
        snapshot_store: Stage5SnapshotStore | None = None,
        calculation_version: str = STAGE5_CALCULATION_VERSION,
    ) -> None:
        self._feature_service = feature_service
        self._snapshot_store = snapshot_store
        self._calculation_version = calculation_version

    async def prepare_expected_moves(
        self,
        requests: Sequence[Stage5QualifiedRequest],
        *,
        session: date,
        t0: datetime,
    ) -> None:
        prepare = getattr(self._feature_service, "prepare_expected_moves", None)
        if prepare is not None:
            await prepare(requests, session=session, t0=t0)

    async def analyze_active_runs(
        self,
        ibkr: IbkrConnection,
        runs: Sequence[RunInstance],
        *,
        session: date,
        t0: datetime,
    ) -> tuple[Stage5FeatureSnapshot, ...]:
        """Compose active-run qualification, conId analysis, and failure projection."""

        qualification = await qualify_active_runs(ibkr, runs)
        return await self.analyze(
            qualification.requests,
            ineligible=qualification.ineligible,
            session=session,
            t0=t0,
        )

    async def analyze(
        self,
        requests: Sequence[Stage5QualifiedRequest],
        *,
        ineligible: Sequence[Stage5IneligibleInstrument] = (),
        session: date,
        t0: datetime,
    ) -> tuple[Stage5FeatureSnapshot, ...]:
        grouped: dict[int, tuple[QualifiedInstrument, set[Stage5Membership]]] = {}
        for request in requests:
            existing = grouped.get(request.instrument.con_id)
            if existing is None:
                grouped[request.instrument.con_id] = (
                    request.instrument,
                    set(request.memberships),
                )
            else:
                existing[1].update(request.memberships)

        features: dict[int, Stage5FeatureResult] = {}
        for con_id in sorted(grouped):
            instrument, _memberships = grouped[con_id]
            try:
                features[con_id] = await self._feature_service.get_feature(
                    instrument, session=session, t0=t0
                )
            except Exception as exc:  # isolate one instrument at the public batch boundary
                features[con_id] = Stage5FeatureResult(
                    con_id=instrument.con_id,
                    symbol=instrument.symbol,
                    session=session,
                    t0=_aware_utc(t0),
                    status=Stage5Status.PRE_MOVE_NOT_READY,
                    exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
                    calculation_version=self._calculation_version,
                )

        rows: list[Stage5FeatureSnapshot] = []
        for con_id in sorted(grouped):
            _instrument, memberships = grouped[con_id]
            feature = features[con_id]
            by_universe: dict[str, list[str]] = {}
            for membership in memberships:
                by_universe.setdefault(membership.universe_id, []).append(membership.run_id)
            for universe_id in sorted(by_universe):
                snapshot = _snapshot_from_feature(
                    feature,
                    universe_id=universe_id,
                    run_ids=tuple(sorted(set(by_universe[universe_id]))),
                )
                rows.append(snapshot)
        if self._snapshot_store is not None:
            await asyncio.to_thread(self._snapshot_store.save_many, rows)
        rows.extend(
            _ineligible_snapshots(
                ineligible,
                session=session,
                t0=t0,
                calculation_version=self._calculation_version,
            )
        )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.universe_id,
                    row.con_id is None,
                    row.con_id or 0,
                    row.symbol,
                ),
            )
        )


def calculate_pre_move(
    *,
    expected_absolute_return_15m: float,
    p0: float,
    raw_open_t0: float,
    raw_open_t0_minus_3m: float,
) -> PreMoveCalculation:
    """Calculate the frozen split-aligned PRE_MOVE feature without data access."""

    values = {
        "expected_absolute_return_15m": expected_absolute_return_15m,
        "p0": p0,
        "raw_open_t0": raw_open_t0,
        "raw_open_t0_minus_3m": raw_open_t0_minus_3m,
    }
    for name, value in values.items():
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    m_price = p0 * expected_absolute_return_15m
    if not isfinite(m_price) or m_price <= 0.0:
        raise ValueError("M_price must be finite and positive")
    alignment_factor = p0 / raw_open_t0
    aligned_pre_open = raw_open_t0_minus_3m * alignment_factor
    raw_pre_move_price = abs(p0 - aligned_pre_open)
    pre_move_m = raw_pre_move_price / m_price
    if not all(
        isfinite(value)
        for value in (alignment_factor, aligned_pre_open, raw_pre_move_price, pre_move_m)
    ):
        raise ValueError("PRE_MOVE calculation produced a non-finite value")
    return PreMoveCalculation(
        p0=p0,
        expected_absolute_return_15m=expected_absolute_return_15m,
        m_price=m_price,
        alignment_factor=alignment_factor,
        aligned_pre_open=aligned_pre_open,
        raw_pre_move_price=raw_pre_move_price,
        pre_move_m=pre_move_m,
    )


def calculate_session_hard_inputs(
    bars: Sequence[HistoricalBar],
    *,
    checkpoint: int,
    session_open: datetime,
    bar_starts: Sequence[datetime] | None = None,
) -> dict[str, float]:
    """Calculate the frozen 15 causal Model B inputs from completed native bars."""

    opened = _aware_utc(session_open)
    if checkpoint < 6 or len(bars) != checkpoint:
        raise ValueError("Session HARD requires the exact completed checkpoint prefix")
    ordered = tuple(sorted(bars, key=_historical_bar_timestamp))
    expected = (
        tuple(_aware_utc(item) for item in bar_starts)
        if bar_starts is not None
        else tuple(opened + timedelta(minutes=5 * index) for index in range(checkpoint))
    )
    if len(expected) != checkpoint:
        raise ValueError("Session HARD requires the exact active trading-bar prefix")
    observed = tuple(_historical_bar_timestamp(bar) for bar in ordered)
    if observed != expected:
        raise ValueError("Session HARD requires exact completed five-minute bar timestamps")
    for bar in ordered:
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if (
            not all(isfinite(value) for value in values)
            or min(bar.open, bar.high, bar.low, bar.close) <= 0.0
            or bar.volume < 0.0
            or bar.high < max(bar.open, bar.low, bar.close)
            or bar.low > min(bar.open, bar.high, bar.close)
        ):
            raise ValueError("Session HARD received an invalid completed bar")

    previous_closes = (ordered[0].open, *(bar.close for bar in ordered[:-1]))
    widths = tuple(bar.high - bar.low for bar in ordered)
    ranges = tuple(
        10_000.0 * max(width, abs(bar.high - previous), abs(bar.low - previous)) / previous
        for bar, previous, width in zip(ordered, previous_closes, widths, strict=True)
    )
    returns = tuple(
        10_000.0 * (bar.close / previous - 1.0)
        for bar, previous in zip(ordered, previous_closes, strict=True)
    )
    total_travel = sum(abs(value) for value in returns)
    net = sum(returns)
    cumulative = 10_000.0 * (ordered[-1].close / ordered[0].open - 1.0)
    if abs(cumulative) <= 1e-12:
        persistence = 0.5
    else:
        direction = 1.0 if cumulative > 0.0 else -1.0
        persistence = sum(_sign(value) == direction for value in returns) / checkpoint
    trailing_ranges = ranges[-6:]
    trailing_returns = returns[-6:]
    mean_range = sum(trailing_ranges) / len(trailing_ranges)
    current = ordered[-1]
    current_width = widths[-1]
    body_fraction = abs(current.close - current.open) / max(current_width, 1e-12)
    upper_wick = (current.high - max(current.open, current.close)) / max(current_width, 1e-12)
    lower_wick = (min(current.open, current.close) - current.low) / max(current_width, 1e-12)
    volumes = tuple(bar.volume for bar in ordered)
    prior_six = volumes[max(0, checkpoint - 7) : checkpoint - 1]
    if not prior_six:
        raise ValueError("Session HARD prior-six volume is unavailable")
    first_three = volumes[:3]
    last_three = volumes[-3:]
    return {
        "range_effort": log1p(sum(ranges)),
        "travel_effort": log1p(total_travel),
        "absolute_efficiency": abs(net / max(total_travel, 1e-12)),
        "close_retention": abs(current.close - ordered[0].open) / max(sum(widths), 1e-12),
        "directional_persistence": persistence,
        "prior_6_mean_range": mean_range,
        "prior_6_price_travel": sum(abs(value) for value in trailing_returns),
        "prior_6_absolute_net_movement": abs(sum(trailing_returns)),
        "recent_vs_earlier_range_ratio": _mean(trailing_ranges[3:])
        / max(_mean(trailing_ranges[:3]), 1e-12),
        "current_bar_range_vs_prior_6": ranges[-1] / max(mean_range, 1e-12),
        "current_bar_body_fraction": min(max(body_fraction, 0.0), 1.0),
        "current_bar_extreme_wick_fraction": max(
            min(max(upper_wick, 0.0), 1.0),
            min(max(lower_wick, 0.0), 1.0),
        ),
        "current_volume_vs_session_mean": volumes[-1] / max(_mean(volumes), 1e-12),
        "current_volume_vs_prior6_mean": volumes[-1] / max(_mean(prior_six), 1e-12),
        "last3_vs_first3_volume": _mean(last_three) / max(_mean(first_three), 1e-12),
    }


def calculate_stage5_feature(
    *,
    instrument: QualifiedInstrument,
    session: date,
    t0: datetime,
    expected_absolute_return_15m: float | None,
    five_minute_bars: Sequence[HistoricalBar],
    one_minute_bars: Sequence[HistoricalBar],
    calculation_version: str = STAGE5_CALCULATION_VERSION,
    expected_move_source: str | None = None,
    expected_move_observation_at: datetime | None = None,
    expected_move_calculation_version: str | None = None,
    raw_historical_volatility: float | None = None,
    historical_volatility: float | None = None,
    market_regular_minutes: int | None = None,
) -> Stage5FeatureResult:
    """Select exact causal IBKR bar inputs and calculate one conId/checkpoint feature."""

    signal_timestamp = _aware_utc(t0)
    lineage: _Stage5Lineage = {
        "calculation_version": calculation_version,
        "expected_move_source": expected_move_source,
        "expected_move_observation_at": expected_move_observation_at,
        "expected_move_calculation_version": expected_move_calculation_version,
        "raw_historical_volatility": raw_historical_volatility,
        "historical_volatility": historical_volatility,
        "market_regular_minutes": market_regular_minutes,
    }
    if expected_absolute_return_15m is None:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=Stage5Status.PRE_CONTEXT_NOT_READY,
            exclusion_reason="PRE_CONTEXT_NOT_READY: expected_absolute_return_15m unavailable",
            **lineage,
        )

    try:
        p0 = _exact_open(five_minute_bars, signal_timestamp)
    except ValueError as exc:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=Stage5Status.PRE_MOVE_NOT_READY,
            exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
            expected_absolute_return_15m=expected_absolute_return_15m,
            **lineage,
        )
    if p0 is None:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=Stage5Status.PRE_MOVE_NOT_READY,
            exclusion_reason="PRE_MOVE_NOT_READY: missing native 5-minute open at T0",
            expected_absolute_return_15m=expected_absolute_return_15m,
            **lineage,
        )
    t0_minus_3m = signal_timestamp - timedelta(minutes=3)
    try:
        raw_open_t0_minus_3m = _exact_open(one_minute_bars, t0_minus_3m)
    except ValueError as exc:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=Stage5Status.PRE_MOVE_NOT_READY,
            exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
            **lineage,
        )
    if raw_open_t0_minus_3m is None:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=Stage5Status.PRE_MOVE_NOT_READY,
            exclusion_reason="PRE_MOVE_NOT_READY: missing exact 1-minute open at T0-3m",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
            **lineage,
        )
    try:
        raw_open_t0 = _exact_open(one_minute_bars, signal_timestamp)
    except ValueError as exc:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=Stage5Status.PRE_MOVE_NOT_READY,
            exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
            raw_open_t0_minus_3m=raw_open_t0_minus_3m,
            **lineage,
        )
    if raw_open_t0 is None:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=Stage5Status.PRE_MOVE_NOT_READY,
            exclusion_reason="PRE_MOVE_NOT_READY: missing exact 1-minute open at T0",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
            raw_open_t0_minus_3m=raw_open_t0_minus_3m,
            **lineage,
        )
    try:
        calculation = calculate_pre_move(
            expected_absolute_return_15m=expected_absolute_return_15m,
            p0=p0,
            raw_open_t0=raw_open_t0,
            raw_open_t0_minus_3m=raw_open_t0_minus_3m,
        )
    except ValueError as exc:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=Stage5Status.PRE_MOVE_NOT_READY,
            exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
            raw_open_t0_minus_3m=raw_open_t0_minus_3m,
            raw_open_t0=raw_open_t0,
            **lineage,
        )
    return Stage5FeatureResult(
        con_id=instrument.con_id,
        symbol=instrument.symbol,
        session=session,
        t0=signal_timestamp,
        status=Stage5Status.READY,
        exclusion_reason="",
        p0=calculation.p0,
        expected_absolute_return_15m=calculation.expected_absolute_return_15m,
        m_price=calculation.m_price,
        raw_open_t0_minus_3m=raw_open_t0_minus_3m,
        raw_open_t0=raw_open_t0,
        alignment_factor=calculation.alignment_factor,
        aligned_pre_open=calculation.aligned_pre_open,
        raw_pre_move_price=calculation.raw_pre_move_price,
        pre_move_m=calculation.pre_move_m,
        **lineage,
    )


def _exact_open(bars: Sequence[HistoricalBar], timestamp: datetime) -> float | None:
    exact: list[float] = []
    for bar in bars:
        if not isinstance(bar.timestamp, datetime):
            continue
        try:
            observed = _aware_utc(bar.timestamp)
        except ValueError:
            continue
        if observed == timestamp:
            exact.append(bar.open)
    if len(exact) > 1:
        raise ValueError(f"duplicate bars at required timestamp {timestamp.isoformat()}")
    return exact[0] if exact else None


def _snapshot_from_feature(
    feature: Stage5FeatureResult,
    *,
    universe_id: str,
    run_ids: tuple[str, ...],
) -> Stage5FeatureSnapshot:
    return Stage5FeatureSnapshot(
        run_ids=run_ids,
        universe_id=universe_id,
        con_id=feature.con_id,
        symbol=feature.symbol,
        session=feature.session,
        t0=feature.t0,
        status=feature.status,
        exclusion_reason=feature.exclusion_reason,
        p0=feature.p0,
        expected_absolute_return_15m=feature.expected_absolute_return_15m,
        m_price=feature.m_price,
        raw_open_t0_minus_3m=feature.raw_open_t0_minus_3m,
        raw_open_t0=feature.raw_open_t0,
        alignment_factor=feature.alignment_factor,
        aligned_pre_open=feature.aligned_pre_open,
        raw_pre_move_price=feature.raw_pre_move_price,
        pre_move_m=feature.pre_move_m,
        calculation_version=feature.calculation_version,
        expected_move_source=feature.expected_move_source,
        expected_move_observation_at=feature.expected_move_observation_at,
        expected_move_calculation_version=feature.expected_move_calculation_version,
        raw_historical_volatility=feature.raw_historical_volatility,
        historical_volatility=feature.historical_volatility,
        market_regular_minutes=feature.market_regular_minutes,
    )


def _ineligible_snapshots(
    items: Sequence[Stage5IneligibleInstrument],
    *,
    session: date,
    t0: datetime,
    calculation_version: str = STAGE5_CALCULATION_VERSION,
) -> tuple[Stage5FeatureSnapshot, ...]:
    rows: list[Stage5FeatureSnapshot] = []
    signal_timestamp = _aware_utc(t0)
    for item in items:
        by_universe: dict[str, set[str]] = {}
        for membership in item.memberships:
            by_universe.setdefault(membership.universe_id, set()).add(membership.run_id)
        for universe_id in sorted(by_universe):
            rows.append(
                Stage5FeatureSnapshot(
                    run_ids=tuple(sorted(by_universe[universe_id])),
                    universe_id=universe_id,
                    con_id=None,
                    symbol=item.symbol,
                    session=session,
                    t0=signal_timestamp,
                    status=Stage5Status.INELIGIBLE,
                    exclusion_reason=item.reason,
                    p0=None,
                    expected_absolute_return_15m=None,
                    m_price=None,
                    raw_open_t0_minus_3m=None,
                    raw_open_t0=None,
                    alignment_factor=None,
                    aligned_pre_open=None,
                    raw_pre_move_price=None,
                    pre_move_m=None,
                    calculation_version=calculation_version,
                )
            )
    return tuple(rows)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("T0 and bar timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _historical_bar_timestamp(bar: HistoricalBar) -> datetime:
    if not isinstance(bar.timestamp, datetime):
        raise ValueError("Session HARD requires intraday datetime bars")
    return _aware_utc(bar.timestamp)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot calculate a mean from no values")
    return sum(values) / len(values)


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError("invalid numeric value in Stage 5 feature snapshot")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError("invalid integer value in Stage 5 feature snapshot")


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_datetime(value: object) -> datetime | None:
    return _aware_utc(datetime.fromisoformat(str(value))) if value is not None else None
