from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_core.markets import ActivityScanner, CapBucket, MarketId, get_market
from stocker_execution.activity_shortlist import (
    ACTIVITY_SHORTLIST_VERSION,
    ActivityShortlistService,
    ActivityShortlistStatus,
    ActivityShortlistStore,
    ScannerCandidate,
    rank_activity_candidates,
)
from stocker_execution.ibkr import ScannerCapabilities


def candidate(component: ActivityScanner, rank: int, symbol: str, con_id: int) -> ScannerCandidate:
    return ScannerCandidate(component, rank, symbol, con_id, "SMART", "NASDAQ", "USD")


def test_rank_union_is_deduplicated_deterministic_and_component_hits_dominate() -> None:
    rows = rank_activity_candidates(
        {
            ActivityScanner.TOP_TRADE_RATE: (
                candidate(ActivityScanner.TOP_TRADE_RATE, 50, "THREE", 3),
                candidate(ActivityScanner.TOP_TRADE_RATE, 1, "ONE", 1),
                candidate(ActivityScanner.TOP_TRADE_RATE, 40, "TWO", 2),
            ),
            ActivityScanner.TOP_VOLUME_RATE: (
                candidate(ActivityScanner.TOP_VOLUME_RATE, 50, "THREE", 3),
                candidate(ActivityScanner.TOP_VOLUME_RATE, 40, "TWO", 2),
            ),
            ActivityScanner.HOT_BY_VOLUME: (
                candidate(ActivityScanner.HOT_BY_VOLUME, 50, "THREE", 3),
            ),
        }
    )

    assert [row.symbol for row in rows] == ["THREE", "TWO", "ONE"]
    assert [row.scan_hit_count for row in rows] == [3, 2, 1]
    assert rows[0].aggregate_screen_score == pytest.approx(0.06)
    assert [row.final_shortlist_rank for row in rows] == [1, 2, 3]
    assert all(row.selected for row in rows)


def test_deduplication_merges_missing_con_id_with_the_known_contract() -> None:
    ranked = rank_activity_candidates(
        {
            ActivityScanner.TOP_TRADE_RATE: (
                ScannerCandidate(
                    ActivityScanner.TOP_TRADE_RATE,
                    1,
                    "AAPL",
                    265598,
                    "SMART",
                    "NASDAQ",
                    "USD",
                ),
            ),
            ActivityScanner.TOP_VOLUME_RATE: (
                ScannerCandidate(
                    ActivityScanner.TOP_VOLUME_RATE,
                    2,
                    "AAPL",
                    None,
                    "SMART",
                    "NASDAQ",
                    "USD",
                ),
            ),
        }
    )
    assert len(ranked) == 1
    assert ranked[0].con_id == 265598
    assert ranked[0].scan_hit_count == 2


def test_rank_union_caps_at_50_but_retains_all_when_fewer_exist() -> None:
    many = tuple(
        candidate(ActivityScanner.TOP_TRADE_RATE, (index % 50) + 1, f"S{index:03}", index)
        for index in range(1, 56)
    )
    volume = tuple(
        candidate(ActivityScanner.TOP_VOLUME_RATE, (index % 50) + 1, f"S{index:03}", index)
        for index in range(1, 56)
    )
    ranked = rank_activity_candidates(
        {ActivityScanner.TOP_TRADE_RATE: many, ActivityScanner.TOP_VOLUME_RATE: volume}
    )
    assert sum(item.selected for item in ranked) == 50
    assert len([item for item in ranked[:4] if item.selected]) == 4


class FakeScanner:
    def __init__(self, components: tuple[ActivityScanner, ...]) -> None:
        self.components = components
        self.calls: list[tuple[ActivityScanner, CapBucket, int]] = []

    async def scanner_capabilities(self) -> ScannerCapabilities:
        market = get_market(MarketId.US_NASDAQ)
        return ScannerCapabilities(
            locations=frozenset({market.scanner_location}),
            scan_codes=frozenset(item.value for item in self.components),
            filters=frozenset({"marketCapAbove", "marketCapBelow"}),
        )

    async def activity_scan(self, *, market, cap_bucket, component, max_results=50):
        self.calls.append((component, cap_bucket, max_results))
        return (candidate(component, 1, "AAPL", 265598),)


class OneRejectedScanner(FakeScanner):
    async def activity_scan(self, *, market, cap_bucket, component, max_results=50):
        if component is ActivityScanner.HOT_BY_VOLUME:
            raise RuntimeError("SCANNER_NOT_AVAILABLE")
        return await super().activity_scan(
            market=market,
            cap_bucket=cap_bucket,
            component=component,
            max_results=max_results,
        )


class OneEntitlementFailureScanner(FakeScanner):
    async def activity_scan(self, *, market, cap_bucket, component, max_results=50):
        if component is ActivityScanner.HOT_BY_VOLUME:
            raise RuntimeError("market data subscription not entitled")
        return await super().activity_scan(
            market=market,
            cap_bucket=cap_bucket,
            component=component,
            max_results=max_results,
        )


class CapRejectedScanner(FakeScanner):
    async def activity_scan(self, *, market, cap_bucket, component, max_results=50):
        raise RuntimeError("CAP_FILTER_UNAVAILABLE: market cap filter rejected")


def test_two_components_are_allowed_and_fewer_than_two_is_not_available(tmp_path: Path) -> None:
    screen_at = datetime(2026, 9, 2, 13, 45, tzinfo=UTC)
    service = ActivityShortlistService(ActivityShortlistStore(tmp_path / "screens.sqlite3"))
    two = FakeScanner((ActivityScanner.TOP_TRADE_RATE, ActivityScanner.HOT_BY_VOLUME))

    ready = asyncio.run(
        service.get_or_create(
            two,
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at,
        )
    )
    assert ready.status is ActivityShortlistStatus.READY
    assert ready.components == (
        ActivityScanner.TOP_TRADE_RATE,
        ActivityScanner.HOT_BY_VOLUME,
    )
    assert all(call[2] == 50 for call in two.calls)

    one = FakeScanner((ActivityScanner.TOP_TRADE_RATE,))
    unavailable = asyncio.run(
        service.get_or_create(
            one,
            market=get_market(MarketId.US_NYSE),
            cap_bucket=CapBucket.ALL,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at,
        )
    )
    assert unavailable.status is ActivityShortlistStatus.NOT_AVAILABLE
    assert unavailable.reason == "ACTIVITY_SHORTLIST_NOT_AVAILABLE"
    assert one.calls == []


def test_one_rejected_component_keeps_two_successful_components(tmp_path: Path) -> None:
    screen_at = datetime(2026, 9, 2, 13, 45, tzinfo=UTC)
    result = asyncio.run(
        ActivityShortlistService(
            ActivityShortlistStore(tmp_path / "screens.sqlite3")
        ).get_or_create(
            OneRejectedScanner(tuple(ActivityScanner)),
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at,
        )
    )
    assert result.status is ActivityShortlistStatus.READY
    assert result.components == (
        ActivityScanner.TOP_TRADE_RATE,
        ActivityScanner.TOP_VOLUME_RATE,
    )


def test_one_unentitled_component_keeps_two_successful_components(tmp_path: Path) -> None:
    screen_at = datetime(2026, 9, 2, 13, 45, tzinfo=UTC)
    result = asyncio.run(
        ActivityShortlistService(
            ActivityShortlistStore(tmp_path / "screens.sqlite3")
        ).get_or_create(
            OneEntitlementFailureScanner(tuple(ActivityScanner)),
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at,
        )
    )
    assert result.status is ActivityShortlistStatus.READY
    assert result.components == (
        ActivityScanner.TOP_TRADE_RATE,
        ActivityScanner.TOP_VOLUME_RATE,
    )


def test_broker_cap_filter_rejection_remains_explicit(tmp_path: Path) -> None:
    screen_at = datetime(2026, 9, 2, 13, 45, tzinfo=UTC)
    result = asyncio.run(
        ActivityShortlistService(
            ActivityShortlistStore(tmp_path / "screens.sqlite3")
        ).get_or_create(
            CapRejectedScanner(tuple(ActivityScanner)),
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at,
        )
    )
    assert result.status is ActivityShortlistStatus.CAP_FILTER_UNAVAILABLE
    assert result.reason == "CAP_FILTER_UNAVAILABLE"


def test_snapshot_is_frozen_and_reconnect_reuses_it_without_scanning(tmp_path: Path) -> None:
    store = ActivityShortlistStore(tmp_path / "screens.sqlite3")
    service = ActivityShortlistService(store)
    screen_at = datetime(2026, 9, 2, 13, 45, tzinfo=UTC)
    first_broker = FakeScanner(tuple(ActivityScanner))
    first = asyncio.run(
        service.get_or_create(
            first_broker,
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at,
        )
    )
    reconnect_broker = FakeScanner(tuple(ActivityScanner))
    reused = asyncio.run(
        service.get_or_create(
            reconnect_broker,
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at + timedelta(minutes=30),
        )
    )

    assert first == reused
    assert len(first_broker.calls) == 3
    assert reconnect_broker.calls == []
    assert first.profile_version == ACTIVITY_SHORTLIST_VERSION


def test_missed_screen_does_not_create_late_population(tmp_path: Path) -> None:
    broker = FakeScanner(tuple(ActivityScanner))
    screen_at = datetime(2026, 9, 2, 13, 45, tzinfo=UTC)
    result = asyncio.run(
        ActivityShortlistService(
            ActivityShortlistStore(tmp_path / "screens.sqlite3")
        ).get_or_create(
            broker,
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at + timedelta(minutes=1),
        )
    )
    assert result.status is ActivityShortlistStatus.MISSED
    assert result.reason == "SCREEN_MISSED"
    assert broker.calls == []


def test_non_all_cap_requires_broker_cap_filters(tmp_path: Path) -> None:
    class NoCapFilters(FakeScanner):
        async def scanner_capabilities(self) -> ScannerCapabilities:
            market = get_market(MarketId.US_NASDAQ)
            return ScannerCapabilities(
                locations=frozenset({market.scanner_location}),
                scan_codes=frozenset(item.value for item in ActivityScanner),
                filters=frozenset(),
            )

    screen_at = datetime(2026, 9, 2, 13, 45, tzinfo=UTC)
    result = asyncio.run(
        ActivityShortlistService(
            ActivityShortlistStore(tmp_path / "screens.sqlite3")
        ).get_or_create(
            NoCapFilters(tuple(ActivityScanner)),
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at,
        )
    )
    assert result.status is ActivityShortlistStatus.CAP_FILTER_UNAVAILABLE
    assert result.reason == "CAP_FILTER_UNAVAILABLE"


def test_current_ibkr_million_cap_filter_names_are_supported(tmp_path: Path) -> None:
    class CurrentCapFilters(FakeScanner):
        async def scanner_capabilities(self) -> ScannerCapabilities:
            market = get_market(MarketId.US_NASDAQ)
            return ScannerCapabilities(
                locations=frozenset({market.scanner_location}),
                scan_codes=frozenset(item.value for item in ActivityScanner),
                filters=frozenset({"marketCapAbove1e6", "marketCapBelow1e6"}),
            )

    screen_at = datetime(2026, 9, 2, 13, 45, tzinfo=UTC)
    result = asyncio.run(
        ActivityShortlistService(
            ActivityShortlistStore(tmp_path / "screens.sqlite3")
        ).get_or_create(
            CurrentCapFilters(tuple(ActivityScanner)),
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            session=date(2026, 9, 2),
            screen_at=screen_at,
            now=screen_at,
        )
    )

    assert result.status is ActivityShortlistStatus.READY
