"""Canonical market-data schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CANONICAL_COLUMNS: tuple[str, ...] = (
    "source",
    "symbol",
    "instrument_type",
    "timeframe",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "currency",
    "timezone",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "bid",
    "ask",
    "spread",
    "spread_bps",
    "adjusted_close",
    "corporate_action_flag",
    "session",
)

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
ALL_SCHEMA_COLUMNS: tuple[str, ...] = CANONICAL_COLUMNS + OPTIONAL_COLUMNS

InstrumentType = Literal["stock", "index", "forex", "crypto", "future", "option", "fund", "other"]


@dataclass(frozen=True)
class MarketDataSpec:
    """Metadata that identifies one OHLCV dataset."""

    source: str
    symbol: str
    instrument_type: str
    timeframe: str
    timezone: str
    currency: str = "USD"

    def normalized_symbol(self) -> str:
        """Return a storage-safe symbol."""

        return self.symbol.upper().replace("/", "-").replace(":", "-")
