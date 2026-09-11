"""Audited candidate discovery. No history, features, trade scores or orders belong here."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from stocker_core.discovery import DiscoveryProfile
from stocker_core.markets import CAP_BUCKETS_V1, CapBucket, MarketDefinition
from stocker_core.methods import content_hash
from stocker_core.runs import RunConfig
from stocker_execution.activity_shortlist import ScannerCapabilities
from stocker_execution.ibkr import QualifiedInstrument


@dataclass(frozen=True)
class DiscoveryScan:
    cap_band: CapBucket
    instrument: str
    location: str
    scanner: str
    rows: int
    stock_type: str
    minimum_cap_millions: float | None
    maximum_cap_millions: float | None
    minimum_price: float
    minimum_volume: int
    minimum_average_volume: int
    price_currency: str = "LOCAL"
    cap_currency: str = "USD"


@dataclass(frozen=True)
class DiscoveryFx:
    """Observed local currency per USD, used only to translate scanner cap boundaries."""

    currency: str
    local_per_usd: float
    con_id: int
    symbol: str
    bid: float
    ask: float
    observed_at: str


@dataclass(frozen=True)
class DiscoveryRow:
    """One unfiltered broker observation. Rank is IBKR's original zero-based rank."""

    con_id: int
    symbol: str
    exchange: str
    primary_exchange: str | None
    currency: str
    security_type: str
    raw_rank: int
    metadata: dict[str, str]


class DiscoveryBroker(Protocol):
    async def scanner_capabilities(self) -> ScannerCapabilities: ...
    async def discovery_fx(self, currency: str) -> DiscoveryFx: ...
    async def discovery_scan(self, request: DiscoveryScan) -> tuple[DiscoveryRow, ...]: ...
    async def qualify_discovery_candidate(
        self, row: DiscoveryRow
    ) -> tuple[QualifiedInstrument, str]: ...


def scanner_requests(
    profile: DiscoveryProfile, market: MarketDefinition, capabilities: ScannerCapabilities,
    *, local_per_usd: float | None = None,
) -> tuple[DiscoveryScan, ...]:
    """Validate the entire profile before issuing any scan; never substitute scan codes."""
    if local_per_usd is not None and (not isfinite(local_per_usd) or local_per_usd <= 0):
        raise ValueError("MARKET_DATA_UNAVAILABLE: invalid scanner FX conversion")
    location = profile.scanner_location or market.scanner_location
    price_filter = "usdPriceAbove" if profile.price_currency == "USD" else "priceAbove"
    required = {price_filter, "volumeAbove", "avgVolumeAbove"}
    available = capabilities.filters_for(location)
    missing = sorted(required - available)
    for names in (
        {"marketCapAbove", "marketCapAbove1e6"},
        {"marketCapBelow", "marketCapBelow1e6"},
    ):
        if not available & names:
            missing.append("/".join(sorted(names)))
    if (
        location not in capabilities.locations
        or not capabilities.supports_instrument(location, market.scanner_instrument)
        or profile.scanner.value not in capabilities.scan_codes_for(location)
        or missing
    ):
        raise ValueError(
            f"SCANNER_NOT_SUPPORTED: {market.scanner_instrument}/{location}/"
            f"{profile.scanner.value}; missing filters: {', '.join(missing) or 'none'}"
        )
    def local_millions(usd: int | None) -> float | None:
        rate = local_per_usd if local_per_usd is not None else 1
        return usd * rate / 1_000_000 if usd is not None else None

    return tuple(
        DiscoveryScan(
            band,
            market.scanner_instrument,
            location,
            profile.scanner.value,
            profile.results_per_band,
            profile.stock_type_filter,
            local_millions(CAP_BUCKETS_V1.definition(band).minimum_usd),
            local_millions(CAP_BUCKETS_V1.definition(band).maximum_usd_exclusive),
            profile.minimum_price,
            profile.minimum_volume,
            profile.minimum_average_volume,
            profile.price_currency,
            market.currency if local_per_usd is not None else "USD",
        )
        for band in profile.cap_bands
    )


class DiscoveryStore:
    """One durable document per attempt, updated at stage boundaries, never per quote tick."""

    def __init__(self, path: Path, *, initialize: bool = True):
        self.path = path
        if initialize:
            with self._connect() as connection:
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS candidate_discovery_runs (
                        discovery_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                        session TEXT NOT NULL, config_hash TEXT NOT NULL,
                        generation INTEGER NOT NULL, document_json TEXT NOT NULL,
                        UNIQUE(run_id, session, config_hash, generation)
                    );
                    CREATE TABLE IF NOT EXISTS candidate_discovery_refresh (
                        run_id TEXT PRIMARY KEY, generation INTEGER NOT NULL
                    );
                """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def generation(self, run_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT generation FROM candidate_discovery_refresh WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def request_refresh(self, run_id: str) -> None:
        # Caller serializes this with run controls and requires the run to be disabled.
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO candidate_discovery_refresh VALUES (?, 1) "
                "ON CONFLICT(run_id) DO UPDATE SET generation=generation+1",
                (run_id,),
            )

    def save(self, document: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO candidate_discovery_runs VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(discovery_id) DO UPDATE SET document_json=excluded.document_json",
                (
                    document["discovery_id"],
                    document["run_id"],
                    document["session"],
                    document["config_hash"],
                    document["generation"],
                    json.dumps(document, sort_keys=True, allow_nan=False),
                ),
            )

    def find(self, run_id: str, session: date, digest: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT document_json FROM candidate_discovery_runs "
                "WHERE run_id=? AND session=? AND config_hash=? AND generation=?",
                (run_id, session.isoformat(), digest, self.generation(run_id)),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def history(
        self,
        run_id: str,
        *,
        limit: int = 100,
        session: date | None = None,
        successful_only: bool = False,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._connect() as connection:
            if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='candidate_discovery_runs'"
            ).fetchone():
                return []
            rows = connection.execute(
                "SELECT document_json FROM candidate_discovery_runs WHERE run_id=? "
                "AND (? IS NULL OR session=?) "
                "AND (?=0 OR json_extract(document_json, '$.status')='READY') "
                "ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (
                    run_id,
                    session.isoformat() if session else None,
                    session.isoformat() if session else None,
                    int(successful_only),
                    limit,
                    offset,
                ),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]


def discovery_summary(document: dict[str, Any]) -> dict[str, Any]:
    observations = document["observations"]
    candidates = document["candidates"]
    return {
        key: document[key]
        for key in ("discovery_id", "status", "reason", "started_at", "completed_at", "session")
    } | {
        "per_cap_band": {
            segment["request"]["cap_band"]: segment["returned"] for segment in document["segments"]
        },
        "raw_candidates": len(observations),
        "unique_candidates": len({r["con_id"] for r in observations if r["con_id"] > 0}),
        "rejected_candidates": sum(bool(c["rejection_reason"]) for c in candidates),
        "rejection_counts": dict(
            Counter(c["rejection_reason"] for c in candidates if c["rejection_reason"])
        ),
        "duplicate_observations": sum(
            r.get("rejection_reason") == "DUPLICATE_CONID" for r in observations
        ),
        "warnings": sorted({
            r["metadata"]["warning"] for r in observations if r["metadata"].get("warning")
        }),
        "fx_conversion": document.get("fx_conversion"),
        "watch_pool_size": (
            sum(c["in_watch_pool"] for c in candidates) if document["status"] == "READY" else 0
        ),
        "captured_at": document.get("captured_at"),
        "ready_at": document["completed_at"] if document["status"] == "READY" else None,
        "observation_rejection_counts": dict(
            Counter(r["rejection_reason"] for r in observations if r["rejection_reason"])
        ),
    }


class CandidateDiscovery:
    def __init__(
        self,
        broker: DiscoveryBroker,
        store: DiscoveryStore,
        clock: Callable[[], datetime] | None = None,
    ):
        self.broker = broker
        self.store = store
        self.clock = clock
        self._lock = asyncio.Lock()

    async def discover(
        self,
        run: RunConfig,
        market: MarketDefinition,
        session: date,
        now: datetime,
        due: datetime,
    ) -> dict[str, Any]:
        async with self._lock:
            return await self._discover(run, market, session, now, due)

    async def _discover(
        self,
        run: RunConfig,
        market: MarketDefinition,
        session: date,
        now: datetime,
        due: datetime,
    ) -> dict[str, Any]:
        profile = run.discovery_profile
        assert profile is not None
        configuration = {
            "profile": profile.model_dump(mode="json"),
            "method_spec_hash": run.method_spec_hash,
            "cap_bucket_version": CAP_BUCKETS_V1.version,
            "source": run.universe_source,
        }
        digest = content_hash(configuration)
        document = self.store.find(run.run_id, session, digest)
        if document is not None and (document["status"] != "SCHEDULED" or now < due):
            if document["status"] == "RUNNING":
                document.update(status="FAILED", reason="DISCOVERY_INTERRUPTED")
                self.store.save(document)
            return document
        if document is None:
            document = {
                "discovery_id": str(uuid4()),
                "run_id": run.run_id,
                "session": session.isoformat(),
                "config_hash": digest,
                "generation": self.store.generation(run.run_id),
                "market": market.market_id.value,
                "method": run.strategy_id,
                "strategy_version": run.strategy_version,
                "configuration": configuration,
                "started_at": now.isoformat(),
                "completed_at": None,
                "status": "RUNNING",
                "reason": "",
                "segments": [],
                "observations": [],
                "candidates": [],
            }
        self.store.save(document)
        try:
            requests = scanner_requests(profile, market, await self.broker.scanner_capabilities())
        except Exception as exc:
            completed = self.clock() if self.clock else datetime.now(now.tzinfo)
            document.update(status="FAILED", reason=str(exc), completed_at=completed.isoformat())
            self.store.save(document)
            return document
        document["segments"] = [
            {"request": asdict(request), "returned": 0, "status": "PENDING", "reason": ""}
            for request in requests
        ]
        if now < due:
            document["status"] = "SCHEDULED"
            self.store.save(document)
            return document
        document["status"] = "RUNNING"
        document["scan_started_at"] = now.isoformat()
        self.store.save(document)
        semaphore = asyncio.Semaphore(profile.scan_concurrency)

        async def scan(request: DiscoveryScan) -> tuple[DiscoveryRow, ...]:
            async with semaphore:
                return await self.broker.discovery_scan(request)

        try:
            if market.currency != "USD":
                fx = await self.broker.discovery_fx(market.currency)
                if fx.currency != market.currency:
                    raise ValueError("MARKET_DATA_UNAVAILABLE: scanner FX currency mismatch")
                document["fx_conversion"] = asdict(fx)
                requests = scanner_requests(
                    profile, market, await self.broker.scanner_capabilities(),
                    local_per_usd=fx.local_per_usd,
                )
                for segment, request in zip(document["segments"], requests, strict=True):
                    segment["request"] = asdict(request)
                self.store.save(document)
            results = await asyncio.gather(*(scan(r) for r in requests), return_exceptions=True)
            received_at = self.clock() if self.clock else datetime.now(now.tzinfo)
            document["captured_at"] = received_at.isoformat()
            for segment, result in zip(document["segments"], results, strict=True):
                if isinstance(result, BaseException):
                    segment.update(status="FAILED", reason=str(result))
                    continue
                segment.update(status="COMPLETE", returned=len(result))
                for row in result:
                    document["observations"].append(
                        asdict(row)
                        | {
                            "cap_band": segment["request"]["cap_band"],
                            "scanner": segment["request"]["scanner"],
                            "discovered_at": received_at.isoformat(),
                            "rejection_reason": "",
                        }
                    )
            # Preserve all returned data even if another segment failed.
            self.store.save(document)
            if any(segment["status"] == "FAILED" for segment in document["segments"]):
                document.update(status="FAILED", reason="SCANNER_FAILED: see segment diagnostics")
                for observation in document["observations"]:
                    observation["rejection_reason"] = "SCANNER_FAILED"
            else:
                await self._reduce(document, profile, market)
                document["status"] = (
                    "READY" if any(c["in_watch_pool"] for c in document["candidates"]) else "EMPTY"
                )
                document["reason"] = "" if document["status"] == "READY" else "NO_WATCH_CANDIDATES"
        except asyncio.CancelledError:
            document.update(status="FAILED", reason="DISCOVERY_INTERRUPTED")
            self.store.save(document)
            raise
        except Exception as exc:
            document.update(status="FAILED", reason=str(exc))
            for candidate in document["candidates"]:
                candidate["in_watch_pool"] = False
        completed = self.clock() if self.clock else datetime.now(now.tzinfo)
        document["completed_at"] = completed.isoformat()
        document["ready_at"] = completed.isoformat() if document["status"] == "READY" else None
        self.store.save(document)
        return document

    async def _reduce(
        self,
        document: dict[str, Any],
        profile: DiscoveryProfile,
        market: MarketDefinition,
    ) -> None:
        candidates: dict[int, dict[str, Any]] = {}
        # Interleave equal raw ranks across cap bands so one cap cannot consume the budget.
        observations = sorted(
            enumerate(document["observations"]),
            key=lambda pair: (
                pair[1]["raw_rank"],
                profile.cap_bands.index(CapBucket(pair[1]["cap_band"])),
                pair[0],
            ),
        )
        for index, row in observations:
            if row["con_id"] > 0 and row["con_id"] in candidates:
                candidates[row["con_id"]]["observation_indices"].append(index)
                row["rejection_reason"] = "DUPLICATE_CONID"
                continue
            candidate: dict[str, Any] = {
                "identity": {
                    key: row[key]
                    for key in (
                        "con_id",
                        "symbol",
                        "exchange",
                        "primary_exchange",
                        "currency",
                        "security_type",
                    )
                },
                "observation_indices": [index],
                "rejection_reason": "",
                "stages": {
                    "deduplication": "PASSED",
                    "eligibility": "NOT_RUN",
                    "price": "PASSED_BY_SCANNER",
                    "liquidity": "PASSED_BY_SCANNER",
                    "contract": "NOT_RUN",
                    "watch_pool": "NOT_RUN",
                },
                "in_watch_pool": False,
            }
            document["candidates"].append(candidate)
            if row["con_id"] > 0:
                candidates[row["con_id"]] = candidate
            reason = (
                "INVALID_CONTRACT"
                if row["con_id"] <= 0
                or not row["symbol"]
                or row["raw_rank"] < 0
                or row["currency"] != market.currency
                or not row["exchange"]
                else "WRONG_SECURITY_TYPE"
                if row["security_type"] != "STK"
                else ""
            )
            candidate["rejection_reason"] = reason
            candidate["stages"]["eligibility"] = "REJECTED" if reason else "PASSED"
            if not reason and row["raw_rank"] >= profile.results_per_band:
                candidate.update(rejection_reason="RESOURCE_LIMIT", resource_stage="SCAN_RESULTS")
        eligible = [c for c in document["candidates"] if not c["rejection_reason"]]
        for candidate in eligible[profile.merged_candidate_limit :]:
            candidate.update(rejection_reason="RESOURCE_LIMIT", resource_stage="MERGED_POOL")
        bounded = eligible[: profile.merged_candidate_limit]

        async def qualify(candidate: dict[str, Any]) -> None:
            observation = document["observations"][candidate["observation_indices"][0]]
            row = DiscoveryRow(
                **{key: observation[key] for key in DiscoveryRow.__dataclass_fields__}
            )
            try:
                instrument, stock_type = await self.broker.qualify_discovery_candidate(row)
                candidate["stock_type"] = stock_type
                if instrument.con_id != row.con_id or instrument.currency != market.currency:
                    candidate["rejection_reason"] = "INVALID_CONTRACT"
                elif (
                    instrument.security_type != "STK"
                    or stock_type not in profile.allowed_stock_types
                ):
                    candidate["rejection_reason"] = "WRONG_SECURITY_TYPE"
                else:
                    candidate["identity"] = asdict(instrument)
            except Exception as exc:
                candidate.update(
                    rejection_reason="INVALID_CONTRACT"
                    if str(exc).startswith("INVALID_CONTRACT")
                    else "MARKET_DATA_UNAVAILABLE",
                    detail=str(exc),
                )
            candidate["stages"]["contract"] = (
                "REJECTED" if candidate["rejection_reason"] else "PASSED"
            )

        # Batch contract metadata only; no quote, history or live trade subscriptions.
        for offset in range(0, len(bounded), 4):
            await asyncio.gather(*(qualify(c) for c in bounded[offset : offset + 4]))
        promoted = 0
        for candidate in bounded:
            if candidate["rejection_reason"]:
                continue
            if promoted >= profile.monitoring_limit:
                candidate.update(rejection_reason="RESOURCE_LIMIT", resource_stage="MONITORING")
                candidate["stages"]["watch_pool"] = "REJECTED"
            else:
                promoted += 1
                candidate.update(in_watch_pool=True, discovery_order=promoted)
                candidate["stages"]["watch_pool"] = "PASSED"


def watch_identities(document: dict[str, Any]) -> tuple[QualifiedInstrument, ...]:
    if document["status"] != "READY":
        return ()
    return tuple(
        QualifiedInstrument(**candidate["identity"])
        for candidate in document["candidates"]
        if candidate["in_watch_pool"]
    )
