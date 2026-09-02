"""Causal, non-trading Stage 5 PRE features and cohort descriptions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Protocol

from stocker_core.runs import RunInstance, RunState
from stocker_core.universes import InstrumentReference
from stocker_execution.history import (
    HistorySemantics,
    HistorySnapshot,
    HistoryStatus,
    IbkrHistoryCache,
    IbkrHistoryService,
)
from stocker_execution.ibkr import HistoricalBar, IbkrConnection, IbkrError, QualifiedInstrument
from stocker_execution.pre_context import ContextStatus, PriorSessionContextResult

PRE_MOVE_M_THRESHOLD = 0.475764059845861
STAGE5_CALCULATION_VERSION = "STAGE5_PRE_MOVE_V1"
COHORT_LOOKBACK_SESSIONS = 20
COHORT_MINIMUM_PRIOR_OBSERVATIONS = 30
LOW_PERCENTILE_MAX = 33.33
MID_PERCENTILE_MAX = 66.67
STAGE5_FIVE_MINUTE_HISTORY = HistorySemantics("5 mins", "TRADES", True)
STAGE5_ONE_MINUTE_HISTORY = HistorySemantics("1 min", "TRADES", True)


class PREMoveBand(StrEnum):
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


class CandidateStatus(StrEnum):
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
    passes_pre_move_threshold: bool


@dataclass(frozen=True, slots=True)
class Stage5FeatureResult:
    """One canonical conId/checkpoint feature outcome before cohort projection."""

    con_id: int
    symbol: str
    session: date
    t0: datetime
    status: CandidateStatus
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
    passes_pre_move_threshold: bool | None = None
    calculation_version: str = STAGE5_CALCULATION_VERSION


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
    status: CandidateStatus = CandidateStatus.INELIGIBLE


@dataclass(frozen=True, slots=True)
class Stage5QualificationResult:
    requests: tuple[Stage5QualifiedRequest, ...]
    ineligible: tuple[Stage5IneligibleInstrument, ...]


@dataclass(frozen=True, slots=True)
class Stage5CandidateSnapshot:
    """One deterministic non-trading candidate row projected to a universe."""

    run_ids: tuple[str, ...]
    universe_id: str
    cohort_id: str | None
    con_id: int | None
    symbol: str
    session: date
    t0: datetime
    status: CandidateStatus
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
    passes_pre_move_threshold: bool | None
    cohort_size: int | None
    cohort_pre_move_percentile: float | None
    pre_move_band: PREMoveBand | None
    cohort_reason: str
    calculation_version: str


class _FeatureService(Protocol):
    async def get_feature(
        self, instrument: QualifiedInstrument, *, session: date, t0: datetime
    ) -> Stage5FeatureResult: ...


class _ContextService(Protocol):
    async def get_or_create(
        self, instrument: QualifiedInstrument, *, session: date
    ) -> PriorSessionContextResult: ...


class Stage5CurrentDataService:
    """Load Stage 4 context and the minimum exact current-session IBKR bars."""

    def __init__(
        self,
        ibkr: IbkrConnection,
        history_cache: IbkrHistoryCache,
        context_service: _ContextService,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ibkr, IbkrConnection):
            raise TypeError("Stage 5 current-session data requires the Stage 2 IbkrConnection")
        self._history_cache = history_cache
        self._history_service = IbkrHistoryService(ibkr, history_cache)
        self._context_service = context_service
        self._clock = clock or (lambda: datetime.now(tz=UTC))

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
                status=CandidateStatus.PRE_MOVE_NOT_READY,
                exclusion_reason="PRE_MOVE_NOT_READY: T0 opening print is not yet causal",
            )

        context_result = await self._context_service.get_or_create(instrument, session=session)
        if context_result.status is not ContextStatus.READY or context_result.context is None:
            return Stage5FeatureResult(
                con_id=instrument.con_id,
                symbol=instrument.symbol,
                session=session,
                t0=signal_timestamp,
                status=CandidateStatus.PRE_CONTEXT_NOT_READY,
                exclusion_reason=context_result.reason,
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
                status=CandidateStatus.PRE_MOVE_NOT_READY,
                exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
                expected_absolute_return_15m=(context_result.context.expected_absolute_return_15m),
            )
        return calculate_stage5_feature(
            instrument=instrument,
            session=session,
            t0=signal_timestamp,
            expected_absolute_return_15m=(context_result.context.expected_absolute_return_15m),
            five_minute_bars=five.bars,
            one_minute_bars=one.bars,
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
    """Small SQLite audit store and causal cohort-history source."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stage5_candidate_snapshots (
                    cohort_id TEXT,
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
                    passes_pre_move_threshold INTEGER,
                    cohort_size INTEGER,
                    cohort_pre_move_percentile REAL,
                    pre_move_band TEXT,
                    cohort_reason TEXT NOT NULL,
                    calculation_version TEXT NOT NULL,
                    PRIMARY KEY (universe_id, t0_utc, con_id, calculation_version)
                )
                """
            )

    def save(self, snapshot: Stage5CandidateSnapshot) -> None:
        """Persist one qualified row; a transient rerun cannot replace a READY row."""

        if snapshot.con_id is None:
            raise ValueError("cannot persist an unqualified Stage 5 row without conId")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO stage5_candidate_snapshots (
                    cohort_id, universe_id, run_ids_json, con_id, symbol, session, t0_utc,
                    status, exclusion_reason, p0, expected_absolute_return_15m, m_price,
                    raw_open_t0_minus_3m, raw_open_t0, alignment_factor, aligned_pre_open,
                    raw_pre_move_price, pre_move_m, passes_pre_move_threshold, cohort_size,
                    cohort_pre_move_percentile, pre_move_band, cohort_reason,
                    calculation_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (universe_id, t0_utc, con_id, calculation_version)
                DO UPDATE SET
                    cohort_id = excluded.cohort_id,
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
                    passes_pre_move_threshold = excluded.passes_pre_move_threshold,
                    cohort_size = excluded.cohort_size,
                    cohort_pre_move_percentile = excluded.cohort_pre_move_percentile,
                    pre_move_band = excluded.pre_move_band,
                    cohort_reason = excluded.cohort_reason
                WHERE stage5_candidate_snapshots.status != ?
                  AND excluded.status = ?
                """,
                (
                    snapshot.cohort_id,
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
                    (
                        None
                        if snapshot.passes_pre_move_threshold is None
                        else int(snapshot.passes_pre_move_threshold)
                    ),
                    snapshot.cohort_size,
                    snapshot.cohort_pre_move_percentile,
                    snapshot.pre_move_band.value if snapshot.pre_move_band else None,
                    snapshot.cohort_reason,
                    snapshot.calculation_version,
                    CandidateStatus.READY.value,
                    CandidateStatus.READY.value,
                ),
            )

    def prior_qualifying_pre_move_m(
        self,
        cohort_id: str,
        *,
        before_session: date,
    ) -> tuple[float, ...]:
        """Load qualifying rows from the preceding distinct cohort session dates."""

        with self._connect() as connection:
            session_rows = connection.execute(
                """
                SELECT DISTINCT session
                FROM stage5_candidate_snapshots
                WHERE cohort_id = ?
                  AND session < ?
                  AND status = ?
                  AND passes_pre_move_threshold = 1
                  AND pre_move_m IS NOT NULL
                  AND calculation_version = ?
                ORDER BY session DESC
                LIMIT ?
                """,
                (
                    cohort_id,
                    before_session.isoformat(),
                    CandidateStatus.READY.value,
                    STAGE5_CALCULATION_VERSION,
                    COHORT_LOOKBACK_SESSIONS,
                ),
            ).fetchall()
            sessions = tuple(str(row["session"]) for row in session_rows)
            if not sessions:
                return ()
            placeholders = ", ".join("?" for _ in sessions)
            rows = connection.execute(
                f"""
                SELECT pre_move_m
                FROM stage5_candidate_snapshots
                WHERE cohort_id = ?
                  AND session IN ({placeholders})
                  AND status = ?
                  AND passes_pre_move_threshold = 1
                  AND pre_move_m IS NOT NULL
                  AND calculation_version = ?
                ORDER BY session, t0_utc, con_id
                """,  # noqa: S608 - placeholders are generated, not user supplied
                (
                    cohort_id,
                    *sessions,
                    CandidateStatus.READY.value,
                    STAGE5_CALCULATION_VERSION,
                ),
            ).fetchall()
        return tuple(float(row["pre_move_m"]) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


async def qualify_active_runs(
    ibkr: IbkrConnection, runs: Sequence[RunInstance]
) -> Stage5QualificationResult:
    """Qualify active universe members once, then deduplicate physical stocks by conId."""

    if not isinstance(ibkr, IbkrConnection):
        raise TypeError("Stage 5 qualification requires the Stage 2 IbkrConnection")
    references: dict[InstrumentReference, set[Stage5Membership]] = {}
    ineligible: list[Stage5IneligibleInstrument] = []
    for run in runs:
        if run.state is not RunState.ACTIVE:
            continue
        membership = Stage5Membership(run.config.run_id, run.universe.universe_id)
        if run.config.environment is not ibkr.config.environment:
            for reference in run.universe.members:
                ineligible.append(
                    Stage5IneligibleInstrument(
                        reference.symbol,
                        (membership,),
                        "run environment does not match this IBKR connection",
                    )
                )
            continue
        for reference in run.universe.members:
            references.setdefault(reference, set()).add(membership)

    qualified: dict[int, tuple[QualifiedInstrument, set[Stage5Membership]]] = {}
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
    ) -> None:
        self._feature_service = feature_service
        self._snapshot_store = snapshot_store

    async def analyze_active_runs(
        self,
        ibkr: IbkrConnection,
        runs: Sequence[RunInstance],
        *,
        session: date,
        t0: datetime,
    ) -> tuple[Stage5CandidateSnapshot, ...]:
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
    ) -> tuple[Stage5CandidateSnapshot, ...]:
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
                    status=CandidateStatus.PRE_MOVE_NOT_READY,
                    exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
                )

        rows: list[Stage5CandidateSnapshot] = []
        for con_id in sorted(grouped):
            _instrument, memberships = grouped[con_id]
            feature = features[con_id]
            by_universe: dict[str, list[str]] = {}
            for membership in memberships:
                by_universe.setdefault(membership.universe_id, []).append(membership.run_id)
            for universe_id in sorted(by_universe):
                candidate = _candidate_from_feature(
                    feature,
                    universe_id=universe_id,
                    run_ids=tuple(sorted(set(by_universe[universe_id]))),
                )
                rows.append(candidate)
                if self._snapshot_store is not None:
                    self._snapshot_store.save(candidate)
        rows.extend(_ineligible_candidates(ineligible, session=session, t0=t0))
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
        passes_pre_move_threshold=pre_move_m > PRE_MOVE_M_THRESHOLD,
    )


def calculate_cohort_percentile(
    current_pre_move_m: float,
    prior_qualifying_pre_move_m: Sequence[float],
) -> float | None:
    """Return the frozen causal percentile against prior qualifying observations."""

    if not isfinite(current_pre_move_m) or current_pre_move_m < 0.0:
        raise ValueError("current PRE_MOVE_M must be finite and nonnegative")
    history = tuple(prior_qualifying_pre_move_m)
    if any(not isfinite(value) or value < 0.0 for value in history):
        raise ValueError("cohort history must contain only finite nonnegative PRE_MOVE_M values")
    if len(history) < COHORT_MINIMUM_PRIOR_OBSERVATIONS:
        return None
    less_than_or_equal_count = sum(value <= current_pre_move_m for value in history)
    return 100.0 * less_than_or_equal_count / len(history)


def classify_pre_move_band(percentile: float | None) -> PREMoveBand | None:
    """Map an unrounded cohort percentile to the frozen LOW/MID/HIGH boundaries."""

    if percentile is None:
        return None
    if not isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("cohort percentile must be finite and between 0 and 100")
    if percentile <= LOW_PERCENTILE_MAX:
        return PREMoveBand.LOW
    if percentile <= MID_PERCENTILE_MAX:
        return PREMoveBand.MID
    return PREMoveBand.HIGH


def calculate_stage5_feature(
    *,
    instrument: QualifiedInstrument,
    session: date,
    t0: datetime,
    expected_absolute_return_15m: float | None,
    five_minute_bars: Sequence[HistoricalBar],
    one_minute_bars: Sequence[HistoricalBar],
) -> Stage5FeatureResult:
    """Select exact causal IBKR bar inputs and calculate one conId/checkpoint feature."""

    signal_timestamp = _aware_utc(t0)
    if expected_absolute_return_15m is None:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=CandidateStatus.PRE_CONTEXT_NOT_READY,
            exclusion_reason="PRE_CONTEXT_NOT_READY: expected_absolute_return_15m unavailable",
        )

    try:
        p0 = _exact_open(five_minute_bars, signal_timestamp)
    except ValueError as exc:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=CandidateStatus.PRE_MOVE_NOT_READY,
            exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
            expected_absolute_return_15m=expected_absolute_return_15m,
        )
    if p0 is None:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=CandidateStatus.PRE_MOVE_NOT_READY,
            exclusion_reason="PRE_MOVE_NOT_READY: missing native 5-minute open at T0",
            expected_absolute_return_15m=expected_absolute_return_15m,
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
            status=CandidateStatus.PRE_MOVE_NOT_READY,
            exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
        )
    if raw_open_t0_minus_3m is None:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=CandidateStatus.PRE_MOVE_NOT_READY,
            exclusion_reason="PRE_MOVE_NOT_READY: missing exact 1-minute open at T0-3m",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
        )
    try:
        raw_open_t0 = _exact_open(one_minute_bars, signal_timestamp)
    except ValueError as exc:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=CandidateStatus.PRE_MOVE_NOT_READY,
            exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
            raw_open_t0_minus_3m=raw_open_t0_minus_3m,
        )
    if raw_open_t0 is None:
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=session,
            t0=signal_timestamp,
            status=CandidateStatus.PRE_MOVE_NOT_READY,
            exclusion_reason="PRE_MOVE_NOT_READY: missing exact 1-minute open at T0",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
            raw_open_t0_minus_3m=raw_open_t0_minus_3m,
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
            status=CandidateStatus.PRE_MOVE_NOT_READY,
            exclusion_reason=f"PRE_MOVE_NOT_READY: {exc}",
            p0=p0,
            expected_absolute_return_15m=expected_absolute_return_15m,
            raw_open_t0_minus_3m=raw_open_t0_minus_3m,
            raw_open_t0=raw_open_t0,
        )
    return Stage5FeatureResult(
        con_id=instrument.con_id,
        symbol=instrument.symbol,
        session=session,
        t0=signal_timestamp,
        status=CandidateStatus.READY,
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
        passes_pre_move_threshold=calculation.passes_pre_move_threshold,
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


def _candidate_from_feature(
    feature: Stage5FeatureResult,
    *,
    universe_id: str,
    run_ids: tuple[str, ...],
) -> Stage5CandidateSnapshot:
    return Stage5CandidateSnapshot(
        run_ids=run_ids,
        universe_id=universe_id,
        cohort_id=None,
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
        passes_pre_move_threshold=feature.passes_pre_move_threshold,
        cohort_size=None,
        cohort_pre_move_percentile=None,
        pre_move_band=None,
        cohort_reason="CANONICAL_COHORT_MEMBERSHIP_UNRESOLVED",
        calculation_version=feature.calculation_version,
    )


def _ineligible_candidates(
    items: Sequence[Stage5IneligibleInstrument],
    *,
    session: date,
    t0: datetime,
) -> tuple[Stage5CandidateSnapshot, ...]:
    rows: list[Stage5CandidateSnapshot] = []
    signal_timestamp = _aware_utc(t0)
    for item in items:
        by_universe: dict[str, set[str]] = {}
        for membership in item.memberships:
            by_universe.setdefault(membership.universe_id, set()).add(membership.run_id)
        for universe_id in sorted(by_universe):
            rows.append(
                Stage5CandidateSnapshot(
                    run_ids=tuple(sorted(by_universe[universe_id])),
                    universe_id=universe_id,
                    cohort_id=None,
                    con_id=None,
                    symbol=item.symbol,
                    session=session,
                    t0=signal_timestamp,
                    status=CandidateStatus.INELIGIBLE,
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
                    passes_pre_move_threshold=None,
                    cohort_size=None,
                    cohort_pre_move_percentile=None,
                    pre_move_band=None,
                    cohort_reason="CANONICAL_COHORT_MEMBERSHIP_UNRESOLVED",
                    calculation_version=STAGE5_CALCULATION_VERSION,
                )
            )
    return tuple(rows)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("T0 and bar timestamps must be timezone-aware")
    return value.astimezone(UTC)
