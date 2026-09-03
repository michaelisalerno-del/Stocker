from __future__ import annotations

from stocker_core.markets import CAP_BUCKETS_V1, MARKET_CATALOGUE, CapBucket, MarketId


def test_supported_market_catalogue_has_stable_v1_identity_and_metadata() -> None:
    assert tuple(item.market_id for item in MARKET_CATALOGUE) == tuple(MarketId)
    assert {item.market_id for item in MARKET_CATALOGUE} == {
        MarketId.US_NASDAQ,
        MarketId.US_NYSE,
        MarketId.US_ALL,
        MarketId.CANADA_TSX,
        MarketId.UK_LSE,
        MarketId.GERMANY_XETRA,
        MarketId.FRANCE_PARIS,
        MarketId.NETHERLANDS_AMSTERDAM,
        MarketId.SWITZERLAND_SIX,
        MarketId.AUSTRALIA_ASX,
        MarketId.HONG_KONG_HKEX,
        MarketId.JAPAN_TSE,
        MarketId.SOUTH_KOREA_KRX,
        MarketId.SOUTH_AFRICA_JSE,
    }
    nasdaq = next(item for item in MARKET_CATALOGUE if item.market_id is MarketId.US_NASDAQ)
    assert (nasdaq.currency, nasdaq.calendar, nasdaq.timezone) == (
        "USD",
        "XNYS",
        "America/New_York",
    )
    assert nasdaq.scanner_location == "STK.US.MAJOR"
    assert nasdaq.listing_membership == "NASDAQ"

    asx = next(item for item in MARKET_CATALOGUE if item.market_id is MarketId.AUSTRALIA_ASX)
    assert asx.scanner_location == "STK.HK.ASX"
    assert asx.scanner_instrument == "STOCK.HK"
    assert asx.security_type == "STK"

    lse = next(item for item in MARKET_CATALOGUE if item.market_id is MarketId.UK_LSE)
    assert lse.scanner_location == "STK.EU.LSE"
    assert lse.scanner_instrument == "STOCK.EU"
    assert lse.security_type == "STK"

    expected_scanner_identities = {
        MarketId.CANADA_TSX: ("STK.NA.CANADA", "STOCK.NA"),
        MarketId.GERMANY_XETRA: ("STK.EU.IBIS", "STOCK.EU"),
        MarketId.FRANCE_PARIS: ("STK.EU.SBF", "STOCK.EU"),
        MarketId.NETHERLANDS_AMSTERDAM: ("STK.EU.AEB", "STOCK.EU"),
        MarketId.SWITZERLAND_SIX: ("STK.EU.EBS", "STOCK.EU"),
        MarketId.HONG_KONG_HKEX: ("STK.HK.SEHK", "STOCK.HK"),
        MarketId.JAPAN_TSE: ("STK.HK.TSE_JPN", "STOCK.HK"),
        MarketId.SOUTH_KOREA_KRX: ("STK.HK.KRX", "STOCK.HK"),
    }
    for market_id, expected in expected_scanner_identities.items():
        market = next(item for item in MARKET_CATALOGUE if item.market_id is market_id)
        assert (market.scanner_location, market.scanner_instrument) == expected


def test_cap_buckets_v1_use_the_frozen_usd_equivalent_boundaries() -> None:
    assert CAP_BUCKETS_V1.version == "CAP_BUCKETS_V1"
    assert CAP_BUCKETS_V1.bounds(CapBucket.ALL) == (None, None)
    assert CAP_BUCKETS_V1.bounds(CapBucket.MEGA) == (200_000_000_000, None)
    assert CAP_BUCKETS_V1.bounds(CapBucket.LARGE) == (10_000_000_000, 200_000_000_000)
    assert CAP_BUCKETS_V1.bounds(CapBucket.MID) == (2_000_000_000, 10_000_000_000)
    assert CAP_BUCKETS_V1.bounds(CapBucket.SMALL) == (300_000_000, 2_000_000_000)
    assert CAP_BUCKETS_V1.bounds(CapBucket.MICRO) == (50_000_000, 300_000_000)
