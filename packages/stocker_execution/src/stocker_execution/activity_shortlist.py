"""Versioned, causal IBKR activity shortlist and immutable session storage."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from stocker_core.markets import ActivityScanner, CapBucket, MarketDefinition

ACTIVITY_SHORTLIST_ID = "ACTIVITY_SHORTLIST_V1"
ACTIVITY_SHORTLIST_VERSION = "ACTIVITY_SHORTLIST_V1"
ACTIVITY_SHORTLIST_WATCH_LIMIT = 50
ACTIVITY_SHORTLIST_COMPONENT_LIMIT = 50
ACTIVITY_SHORTLIST_CAPTURE_WINDOW = timedelta(minutes=1)


class ActivityShortlistStatus(StrEnum):
    READY = "READY"
    SCHEDULED = "SCHEDULED"
    MISSED = "SCREEN_MISSED"
    NOT_AVAILABLE = "ACTIVITY_SHORTLIST_NOT_AVAILABLE"
    CAP_FILTER_UNAVAILABLE = "CAP_FILTER_UNAVAILABLE"
    SCANNER_NOT_AVAILABLE = "SCANNER_NOT_AVAILABLE"
    DATA_NOT_ENTITLED = "DATA_NOT_ENTITLED"
    BROKER_NOT_CONNECTED = "BROKER_NOT_CONNECTED"


@dataclass(frozen=True, slots=True)
class ScannerCandidate:
    component: ActivityScanner
    rank: int
    symbol: str
    con_id: int | None
    exchange: str
    primary_exchange: str | None
    currency: str


@dataclass(frozen=True, slots=True)
class ActivityCandidate:
    symbol: str
    con_id: int | None
    exchange: str
    primary_exchange: str | None
    currency: str
    top_trade_rate_rank: int | None
    top_volume_rate_rank: int | None
    hot_by_volume_rank: int | None
    scan_hit_count: int
    best_component_rank: int
    aggregate_screen_score: float
    final_shortlist_rank: int | None
    selected: bool


@dataclass(frozen=True, slots=True)
class ActivityShortlistSnapshot:
    market_id: str
    cap_bucket: CapBucket
    cap_bucket_version: str
    session: date
    screen_timestamp: datetime
    profile_id: str
    profile_version: str
    status: ActivityShortlistStatus
    components: tuple[ActivityScanner, ...]
    candidates: tuple[ActivityCandidate, ...]
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ScannerCapabilities:
    locations: frozenset[str]
    scan_codes: frozenset[str]
    filters: frozenset[str]


class ActivityScannerBoundary(Protocol):
    async def scanner_capabilities(self) -> ScannerCapabilities: ...

    async def activity_scan(
        self,
        *,
        market: MarketDefinition,
        cap_bucket: CapBucket,
        component: ActivityScanner,
        max_results: int = 50,
    ) -> tuple[ScannerCandidate, ...]: ...


def rank_activity_candidates(
    component_rows: Mapping[ActivityScanner, Sequence[ScannerCandidate]],
    *,
    watch_limit: int = ACTIVITY_SHORTLIST_WATCH_LIMIT,
) -> tuple[ActivityCandidate, ...]:
    """Merge scanner components using the frozen equal-weight deterministic V1 rule."""

    if not 1 <= watch_limit <= ACTIVITY_SHORTLIST_WATCH_LIMIT:
        raise ValueError("Activity Shortlist V1 watch limit must be between 1 and 50")
    if len(component_rows) < 2:
        raise ValueError("ACTIVITY_SHORTLIST_NOT_AVAILABLE")
    grouped: dict[tuple[str, int | None], dict[ActivityScanner, ScannerCandidate]] = {}
    for component in ActivityScanner:
        for row in component_rows.get(component, ())[:ACTIVITY_SHORTLIST_COMPONENT_LIMIT]:
            if row.component is not component or not 1 <= row.rank <= 50:
                continue
            symbol = row.symbol.strip().upper()
            if not symbol:
                continue
            key = (symbol, row.con_id)
            current = grouped.setdefault(key, {})
            previous = current.get(component)
            if previous is None or row.rank < previous.rank:
                current[component] = row

    scored: list[
        tuple[tuple[object, ...], tuple[str, int | None], dict[ActivityScanner, ScannerCandidate]]
    ] = []
    for key, hits in grouped.items():
        component_ranks = tuple(item.rank for item in hits.values())
        aggregate = sum((51 - rank) / 50 for rank in component_ranks)
        scored.append(
            (
                (-len(hits), -aggregate, min(component_ranks), key[0], key[1] or 0),
                key,
                hits,
            )
        )
    scored.sort(key=lambda item: item[0])

    results: list[ActivityCandidate] = []
    for index, (_order, (symbol, con_id), hits) in enumerate(scored, start=1):
        exemplar = min(hits.values(), key=lambda item: (item.rank, item.component.value))
        rank_by_component = {component: item.rank for component, item in hits.items()}
        selected = index <= watch_limit
        results.append(
            ActivityCandidate(
                symbol=symbol,
                con_id=con_id,
                exchange=exemplar.exchange,
                primary_exchange=exemplar.primary_exchange,
                currency=exemplar.currency,
                top_trade_rate_rank=rank_by_component.get(ActivityScanner.TOP_TRADE_RATE),
                top_volume_rate_rank=rank_by_component.get(ActivityScanner.TOP_VOLUME_RATE),
                hot_by_volume_rank=rank_by_component.get(ActivityScanner.HOT_BY_VOLUME),
                scan_hit_count=len(hits),
                best_component_rank=min(rank_by_component.values()),
                aggregate_screen_score=sum((51 - rank) / 50 for rank in rank_by_component.values()),
                final_shortlist_rank=index if selected else None,
                selected=selected,
            )
        )
    return tuple(results)


class ActivityShortlistStore:
    """Persist exactly one immutable activity population per generic market/session key."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS activity_shortlist_snapshots (
                    market_id TEXT NOT NULL,
                    cap_bucket TEXT NOT NULL,
                    cap_bucket_version TEXT NOT NULL,
                    session TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    screen_timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    components_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY (
                        market_id, cap_bucket, cap_bucket_version, session,
                        profile_id, profile_version
                    )
                );
                CREATE TABLE IF NOT EXISTS activity_shortlist_candidates (
                    market_id TEXT NOT NULL,
                    cap_bucket TEXT NOT NULL,
                    cap_bucket_version TEXT NOT NULL,
                    session TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    con_id INTEGER,
                    exchange TEXT NOT NULL,
                    primary_exchange TEXT,
                    currency TEXT NOT NULL,
                    top_trade_rate_rank INTEGER,
                    top_volume_rate_rank INTEGER,
                    hot_by_volume_rank INTEGER,
                    scan_hit_count INTEGER NOT NULL,
                    best_component_rank INTEGER NOT NULL,
                    aggregate_screen_score REAL NOT NULL,
                    final_shortlist_rank INTEGER,
                    selected INTEGER NOT NULL,
                    PRIMARY KEY (
                        market_id, cap_bucket, cap_bucket_version, session,
                        profile_id, profile_version, symbol, con_id
                    )
                );
                """
            )

    def get(
        self,
        market_id: str,
        cap_bucket: CapBucket,
        session: date,
        *,
        cap_bucket_version: str = "CAP_BUCKETS_V1",
        profile_id: str = ACTIVITY_SHORTLIST_ID,
        profile_version: str = ACTIVITY_SHORTLIST_VERSION,
    ) -> ActivityShortlistSnapshot | None:
        key = (
            market_id,
            cap_bucket.value,
            cap_bucket_version,
            session.isoformat(),
            profile_id,
            profile_version,
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM activity_shortlist_snapshots
                WHERE market_id = ? AND cap_bucket = ? AND cap_bucket_version = ?
                  AND session = ? AND profile_id = ? AND profile_version = ?
                """,
                key,
            ).fetchone()
            if row is None:
                return None
            candidate_rows = connection.execute(
                """
                SELECT * FROM activity_shortlist_candidates
                WHERE market_id = ? AND cap_bucket = ? AND cap_bucket_version = ?
                  AND session = ? AND profile_id = ? AND profile_version = ?
                ORDER BY selected DESC, COALESCE(final_shortlist_rank, 999999), symbol, con_id
                """,
                key,
            ).fetchall()
        return ActivityShortlistSnapshot(
            market_id=str(row["market_id"]),
            cap_bucket=CapBucket(str(row["cap_bucket"])),
            cap_bucket_version=str(row["cap_bucket_version"]),
            session=date.fromisoformat(str(row["session"])),
            screen_timestamp=datetime.fromisoformat(str(row["screen_timestamp"])),
            profile_id=str(row["profile_id"]),
            profile_version=str(row["profile_version"]),
            status=ActivityShortlistStatus(str(row["status"])),
            components=tuple(ActivityScanner(item) for item in json.loads(row["components_json"])),
            candidates=tuple(self._candidate(item) for item in candidate_rows),
            reason=str(row["reason"]),
        )

    def save_once(self, snapshot: ActivityShortlistSnapshot) -> ActivityShortlistSnapshot:
        existing = self.get(
            snapshot.market_id,
            snapshot.cap_bucket,
            snapshot.session,
            cap_bucket_version=snapshot.cap_bucket_version,
            profile_id=snapshot.profile_id,
            profile_version=snapshot.profile_version,
        )
        if existing is not None:
            return existing
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO activity_shortlist_snapshots (
                    market_id, cap_bucket, cap_bucket_version, session, profile_id,
                    profile_version, screen_timestamp, status, components_json, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.market_id,
                    snapshot.cap_bucket.value,
                    snapshot.cap_bucket_version,
                    snapshot.session.isoformat(),
                    snapshot.profile_id,
                    snapshot.profile_version,
                    snapshot.screen_timestamp.astimezone(UTC).isoformat(timespec="microseconds"),
                    snapshot.status.value,
                    json.dumps([item.value for item in snapshot.components]),
                    snapshot.reason,
                ),
            )
            for item in snapshot.candidates:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO activity_shortlist_candidates VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        snapshot.market_id,
                        snapshot.cap_bucket.value,
                        snapshot.cap_bucket_version,
                        snapshot.session.isoformat(),
                        snapshot.profile_id,
                        snapshot.profile_version,
                        item.symbol,
                        item.con_id,
                        item.exchange,
                        item.primary_exchange,
                        item.currency,
                        item.top_trade_rate_rank,
                        item.top_volume_rate_rank,
                        item.hot_by_volume_rank,
                        item.scan_hit_count,
                        item.best_component_rank,
                        item.aggregate_screen_score,
                        item.final_shortlist_rank,
                        int(item.selected),
                    ),
                )
        return (
            self.get(
                snapshot.market_id,
                snapshot.cap_bucket,
                snapshot.session,
                cap_bucket_version=snapshot.cap_bucket_version,
                profile_id=snapshot.profile_id,
                profile_version=snapshot.profile_version,
            )
            or snapshot
        )

    @staticmethod
    def _candidate(row: sqlite3.Row) -> ActivityCandidate:
        return ActivityCandidate(
            symbol=str(row["symbol"]),
            con_id=int(row["con_id"]) if row["con_id"] is not None else None,
            exchange=str(row["exchange"]),
            primary_exchange=(str(row["primary_exchange"]) if row["primary_exchange"] else None),
            currency=str(row["currency"]),
            top_trade_rate_rank=(
                int(row["top_trade_rate_rank"]) if row["top_trade_rate_rank"] is not None else None
            ),
            top_volume_rate_rank=(
                int(row["top_volume_rate_rank"])
                if row["top_volume_rate_rank"] is not None
                else None
            ),
            hot_by_volume_rank=(
                int(row["hot_by_volume_rank"]) if row["hot_by_volume_rank"] is not None else None
            ),
            scan_hit_count=int(row["scan_hit_count"]),
            best_component_rank=int(row["best_component_rank"]),
            aggregate_screen_score=float(row["aggregate_screen_score"]),
            final_shortlist_rank=(
                int(row["final_shortlist_rank"])
                if row["final_shortlist_rank"] is not None
                else None
            ),
            selected=bool(row["selected"]),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


class ActivityShortlistService:
    def __init__(self, store: ActivityShortlistStore) -> None:
        self.store = store

    async def get_or_create(
        self,
        broker: ActivityScannerBoundary,
        *,
        market: MarketDefinition,
        cap_bucket: CapBucket,
        session: date,
        screen_at: datetime,
        now: datetime,
        allowed_symbols: frozenset[str] | None = None,
    ) -> ActivityShortlistSnapshot:
        existing = self.store.get(market.market_id.value, cap_bucket, session)
        if existing is not None:
            return existing
        if now < screen_at:
            return self._status(
                market, cap_bucket, session, screen_at, ActivityShortlistStatus.SCHEDULED
            )
        if now >= screen_at + ACTIVITY_SHORTLIST_CAPTURE_WINDOW:
            return self.store.save_once(
                self._status(
                    market,
                    cap_bucket,
                    session,
                    screen_at,
                    ActivityShortlistStatus.MISSED,
                    "SCREEN_MISSED",
                )
            )
        try:
            capabilities = await broker.scanner_capabilities()
        except Exception:
            return self.store.save_once(
                self._status(
                    market,
                    cap_bucket,
                    session,
                    screen_at,
                    ActivityShortlistStatus.BROKER_NOT_CONNECTED,
                    "BROKER_NOT_CONNECTED",
                )
            )
        if market.scanner_location not in capabilities.locations:
            return self.store.save_once(
                self._status(
                    market,
                    cap_bucket,
                    session,
                    screen_at,
                    ActivityShortlistStatus.SCANNER_NOT_AVAILABLE,
                    "SCANNER_NOT_AVAILABLE",
                )
            )
        if cap_bucket is not CapBucket.ALL and not _supports_cap_bucket(
            capabilities.filters, cap_bucket
        ):
            return self.store.save_once(
                self._status(
                    market,
                    cap_bucket,
                    session,
                    screen_at,
                    ActivityShortlistStatus.CAP_FILTER_UNAVAILABLE,
                    "CAP_FILTER_UNAVAILABLE",
                )
            )
        components = tuple(
            item for item in ActivityScanner if item.value in capabilities.scan_codes
        )
        if len(components) < 2:
            return self.store.save_once(
                self._status(
                    market,
                    cap_bucket,
                    session,
                    screen_at,
                    ActivityShortlistStatus.NOT_AVAILABLE,
                    "ACTIVITY_SHORTLIST_NOT_AVAILABLE",
                    components=components,
                )
            )
        rows: dict[ActivityScanner, tuple[ScannerCandidate, ...]] = {}
        for component in components:
            try:
                scanned = await broker.activity_scan(
                    market=market,
                    cap_bucket=cap_bucket,
                    component=component,
                    max_results=ACTIVITY_SHORTLIST_COMPONENT_LIMIT,
                )
            except Exception as exc:
                text = str(exc).lower()
                entitled = any(
                    word in text for word in ("entitle", "subscription", "market data permission")
                )
                status = (
                    ActivityShortlistStatus.DATA_NOT_ENTITLED
                    if entitled
                    else ActivityShortlistStatus.SCANNER_NOT_AVAILABLE
                )
                return self.store.save_once(
                    self._status(
                        market,
                        cap_bucket,
                        session,
                        screen_at,
                        status,
                        status.value,
                        components=components,
                    )
                )
            if allowed_symbols is not None:
                scanned = tuple(item for item in scanned if item.symbol in allowed_symbols)
            rows[component] = scanned
        snapshot = ActivityShortlistSnapshot(
            market_id=market.market_id.value,
            cap_bucket=cap_bucket,
            cap_bucket_version="CAP_BUCKETS_V1",
            session=session,
            screen_timestamp=screen_at.astimezone(UTC),
            profile_id=ACTIVITY_SHORTLIST_ID,
            profile_version=ACTIVITY_SHORTLIST_VERSION,
            status=ActivityShortlistStatus.READY,
            components=components,
            candidates=rank_activity_candidates(rows),
        )
        return self.store.save_once(snapshot)

    @staticmethod
    def _status(
        market: MarketDefinition,
        cap_bucket: CapBucket,
        session: date,
        screen_at: datetime,
        status: ActivityShortlistStatus,
        reason: str = "",
        *,
        components: tuple[ActivityScanner, ...] = (),
    ) -> ActivityShortlistSnapshot:
        return ActivityShortlistSnapshot(
            market.market_id.value,
            cap_bucket,
            "CAP_BUCKETS_V1",
            session,
            screen_at.astimezone(UTC),
            ACTIVITY_SHORTLIST_ID,
            ACTIVITY_SHORTLIST_VERSION,
            status,
            components,
            (),
            reason,
        )


def _supports_cap_bucket(filters: frozenset[str], bucket: CapBucket) -> bool:
    above = bool({"marketCapAbove", "usdMarketCapAbove"} & filters)
    below = bool({"marketCapBelow", "usdMarketCapBelow"} & filters)
    if bucket is CapBucket.MEGA:
        return above
    return above and below
