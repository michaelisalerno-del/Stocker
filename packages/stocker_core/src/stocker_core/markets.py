"""Finite market and capitalisation contracts for the Stage 10 run builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class MarketId(StrEnum):
    US_NASDAQ = "US_NASDAQ"
    US_NYSE = "US_NYSE"
    US_ALL = "US_ALL"
    CANADA_TSX = "CANADA_TSX"
    UK_LSE = "UK_LSE"
    GERMANY_XETRA = "GERMANY_XETRA"
    FRANCE_PARIS = "FRANCE_PARIS"
    NETHERLANDS_AMSTERDAM = "NETHERLANDS_AMSTERDAM"
    SWITZERLAND_SIX = "SWITZERLAND_SIX"
    AUSTRALIA_ASX = "AUSTRALIA_ASX"
    HONG_KONG_HKEX = "HONG_KONG_HKEX"
    JAPAN_TSE = "JAPAN_TSE"
    SOUTH_KOREA_KRX = "SOUTH_KOREA_KRX"
    SOUTH_AFRICA_JSE = "SOUTH_AFRICA_JSE"


class CapBucket(StrEnum):
    ALL = "ALL"
    MEGA = "MEGA"
    LARGE = "LARGE"
    MID = "MID"
    SMALL = "SMALL"
    MICRO = "MICRO"


class ActivityScanner(StrEnum):
    TOP_TRADE_RATE = "TOP_TRADE_RATE"
    TOP_VOLUME_RATE = "TOP_VOLUME_RATE"
    HOT_BY_VOLUME = "HOT_BY_VOLUME"


class MarketAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    DATA_NOT_ENTITLED = "DATA_NOT_ENTITLED"
    SCANNER_NOT_AVAILABLE = "SCANNER_NOT_AVAILABLE"
    CONTRACT_QUALIFICATION_FAILED = "CONTRACT_QUALIFICATION_FAILED"
    BROKER_NOT_CONNECTED = "BROKER_NOT_CONNECTED"


class MarketUniverseSpec(BaseModel):
    """Stable rule-based eligible market/capitalisation space."""

    model_config = ConfigDict(frozen=True)

    market_id: MarketId
    cap_bucket: CapBucket
    cap_bucket_version: str = "CAP_BUCKETS_V1"


@dataclass(frozen=True, slots=True)
class CapBucketDefinition:
    bucket: CapBucket
    label: str
    minimum_usd: int | None
    maximum_usd_exclusive: int | None

    @property
    def scanner_minimum_millions(self) -> float | None:
        return self.minimum_usd / 1_000_000 if self.minimum_usd is not None else None

    @property
    def scanner_maximum_millions(self) -> float | None:
        return (
            self.maximum_usd_exclusive / 1_000_000
            if self.maximum_usd_exclusive is not None
            else None
        )


@dataclass(frozen=True, slots=True)
class CapBucketContract:
    version: str
    buckets: tuple[CapBucketDefinition, ...]

    def definition(self, bucket: CapBucket | str) -> CapBucketDefinition:
        selected = CapBucket(bucket)
        return next(item for item in self.buckets if item.bucket is selected)

    def bounds(self, bucket: CapBucket | str) -> tuple[int | None, int | None]:
        item = self.definition(bucket)
        return item.minimum_usd, item.maximum_usd_exclusive


CAP_BUCKETS_V1 = CapBucketContract(
    version="CAP_BUCKETS_V1",
    buckets=(
        CapBucketDefinition(CapBucket.ALL, "All Caps", None, None),
        CapBucketDefinition(CapBucket.MEGA, "Mega Cap", 200_000_000_000, None),
        CapBucketDefinition(CapBucket.LARGE, "Large Cap", 10_000_000_000, 200_000_000_000),
        CapBucketDefinition(CapBucket.MID, "Mid Cap", 2_000_000_000, 10_000_000_000),
        CapBucketDefinition(CapBucket.SMALL, "Small Cap", 300_000_000, 2_000_000_000),
        CapBucketDefinition(CapBucket.MICRO, "Micro Cap", 50_000_000, 300_000_000),
    ),
)


@dataclass(frozen=True, slots=True)
class RegularSessionSegment:
    opens_at: time
    closes_at: time


@dataclass(frozen=True, slots=True)
class MarketDefinition:
    market_id: MarketId
    display_name: str
    short_name: str
    region: str
    country: str
    currency: str
    calendar: str
    timezone: str
    regular_sessions: tuple[RegularSessionSegment, ...]
    scanner_location: str
    scanner_instrument: str = "STK"
    security_type: str = "STK"
    listing_membership: str | None = None
    experimental: bool = False

    @property
    def active_regular_minutes(self) -> int:
        return sum(
            (segment.closes_at.hour * 60 + segment.closes_at.minute)
            - (segment.opens_at.hour * 60 + segment.opens_at.minute)
            for segment in self.regular_sessions
        )


def _segment(open_value: str, close_value: str) -> RegularSessionSegment:
    return RegularSessionSegment(time.fromisoformat(open_value), time.fromisoformat(close_value))


MARKET_CATALOGUE = (
    MarketDefinition(
        MarketId.US_NASDAQ,
        "US / NASDAQ",
        "NASDAQ",
        "Americas",
        "US",
        "USD",
        "XNYS",
        "America/New_York",
        (_segment("09:30", "16:00"),),
        "STK.US.MAJOR",
        listing_membership="NASDAQ",
    ),
    MarketDefinition(
        MarketId.US_NYSE,
        "US / NYSE",
        "NYSE",
        "Americas",
        "US",
        "USD",
        "XNYS",
        "America/New_York",
        (_segment("09:30", "16:00"),),
        "STK.US.MAJOR",
        listing_membership="NYSE",
    ),
    MarketDefinition(
        MarketId.US_ALL,
        "US / All",
        "US ALL",
        "Americas",
        "US",
        "USD",
        "XNYS",
        "America/New_York",
        (_segment("09:30", "16:00"),),
        "STK.US.MAJOR",
        listing_membership="US_ALL",
    ),
    MarketDefinition(
        MarketId.CANADA_TSX,
        "Canada / TSX",
        "TSX",
        "Americas",
        "Canada",
        "CAD",
        "XTSE",
        "America/Toronto",
        (_segment("09:30", "16:00"),),
        "STK.CA.TSE",
    ),
    MarketDefinition(
        MarketId.UK_LSE,
        "UK / LSE",
        "LSE",
        "Europe",
        "United Kingdom",
        "GBP",
        "XLON",
        "Europe/London",
        (_segment("08:00", "16:30"),),
        "STK.EU.LSE",
        scanner_instrument="STOCK.EU",
    ),
    MarketDefinition(
        MarketId.GERMANY_XETRA,
        "Germany / Xetra",
        "XETRA",
        "Europe",
        "Germany",
        "EUR",
        "XETR",
        "Europe/Berlin",
        (_segment("09:00", "17:30"),),
        "STK.EU.IBIS",
    ),
    MarketDefinition(
        MarketId.FRANCE_PARIS,
        "France / Euronext Paris",
        "PARIS",
        "Europe",
        "France",
        "EUR",
        "XPAR",
        "Europe/Paris",
        (_segment("09:00", "17:30"),),
        "STK.EU.SBF",
    ),
    MarketDefinition(
        MarketId.NETHERLANDS_AMSTERDAM,
        "Netherlands / Euronext Amsterdam",
        "AMSTERDAM",
        "Europe",
        "Netherlands",
        "EUR",
        "XAMS",
        "Europe/Amsterdam",
        (_segment("09:00", "17:30"),),
        "STK.EU.AEB",
    ),
    MarketDefinition(
        MarketId.SWITZERLAND_SIX,
        "Switzerland / SIX",
        "SIX",
        "Europe",
        "Switzerland",
        "CHF",
        "XSWX",
        "Europe/Zurich",
        (_segment("09:00", "17:30"),),
        "STK.EU.SWX",
    ),
    MarketDefinition(
        MarketId.AUSTRALIA_ASX,
        "Australia / ASX",
        "ASX",
        "Asia-Pacific",
        "Australia",
        "AUD",
        "XASX",
        "Australia/Sydney",
        (_segment("10:00", "16:00"),),
        "STK.HK.ASX",
        scanner_instrument="STOCK.HK",
    ),
    MarketDefinition(
        MarketId.HONG_KONG_HKEX,
        "Hong Kong / HKEX",
        "HKEX",
        "Asia-Pacific",
        "Hong Kong",
        "HKD",
        "XHKG",
        "Asia/Hong_Kong",
        (_segment("09:30", "12:00"), _segment("13:00", "16:00")),
        "STK.HK.SEHK",
    ),
    MarketDefinition(
        MarketId.JAPAN_TSE,
        "Japan / TSE",
        "TSE",
        "Asia-Pacific",
        "Japan",
        "JPY",
        "XTKS",
        "Asia/Tokyo",
        (_segment("09:00", "11:30"), _segment("12:30", "15:30")),
        "STK.JP.TSE",
    ),
    MarketDefinition(
        MarketId.SOUTH_KOREA_KRX,
        "South Korea / KRX",
        "KRX",
        "Asia-Pacific",
        "South Korea",
        "KRW",
        "XKRX",
        "Asia/Seoul",
        (_segment("09:00", "15:30"),),
        "STK.KR.KSE",
    ),
    MarketDefinition(
        MarketId.SOUTH_AFRICA_JSE,
        "South Africa / JSE",
        "JSE",
        "Experimental",
        "South Africa",
        "ZAR",
        "XJSE",
        "Africa/Johannesburg",
        (_segment("09:00", "17:00"),),
        "STK.ZA.JSE",
        experimental=True,
    ),
)

_MARKETS_BY_ID = {item.market_id: item for item in MARKET_CATALOGUE}

_MARKET_EXCHANGE_ALIASES = {
    "NASDAQ": MarketId.US_NASDAQ,
    "ISLAND": MarketId.US_NASDAQ,
    "NYSE": MarketId.US_NYSE,
    "TSE": MarketId.CANADA_TSX,
    "LSE": MarketId.UK_LSE,
    "LSEETF": MarketId.UK_LSE,
    "IBIS": MarketId.GERMANY_XETRA,
    "IBIS2": MarketId.GERMANY_XETRA,
    "SBF": MarketId.FRANCE_PARIS,
    "AEB": MarketId.NETHERLANDS_AMSTERDAM,
    "SWX": MarketId.SWITZERLAND_SIX,
    "ASX": MarketId.AUSTRALIA_ASX,
    "SEHK": MarketId.HONG_KONG_HKEX,
    "TSEJ": MarketId.JAPAN_TSE,
    "KSE": MarketId.SOUTH_KOREA_KRX,
    "JSE": MarketId.SOUTH_AFRICA_JSE,
}


def get_market(market_id: MarketId | str) -> MarketDefinition:
    """Return one supported market or fail explicitly."""

    try:
        return _MARKETS_BY_ID[MarketId(market_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported Stocker market: {market_id}") from exc


def market_for_instrument(
    *, primary_exchange: str | None, exchange: str, currency: str
) -> MarketDefinition:
    """Resolve the small catalogue entry needed for market-local PRE semantics."""

    for value in (primary_exchange, exchange):
        if value and value.upper() in _MARKET_EXCHANGE_ALIASES:
            return get_market(_MARKET_EXCHANGE_ALIASES[value.upper()])
    currency_matches = tuple(item for item in MARKET_CATALOGUE if item.currency == currency.upper())
    if len(currency_matches) == 1:
        return currency_matches[0]
    if currency.upper() == "USD":
        return get_market(MarketId.US_ALL)
    raise ValueError("instrument cannot be mapped to a supported Stocker market")
